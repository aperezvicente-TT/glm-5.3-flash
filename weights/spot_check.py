#!/usr/bin/env python3
"""Spot-check the GLM-5.3-Flash weight cache against the fp8/bf16 checkpoint (CPU only, no ttnn).

For N random tensors per dtype group (bfp4 / bfp8 / bf16 / fp32): verify sha256 against the manifest, decode
the .tensorbin with tile_codec (pure numpy), rebuild the expected tensor from the HF checkpoint (fp8 block
dequant -> transform, float32) and report PCC, max |err| and relative RMS error.
Gate: bfp8 PCC >= 0.999, bf16/fp32 must match the bf16-rounded / exact source bit-for-bit; bfp4 is reported.

    python weights/spot_check.py --per-group 32 --out doc/ttnn/00c-weights-goldens/spot_check.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tile_codec import decode, pcc  # noqa: E402
from glm53_cache_spec import dtype_group, read_headers  # noqa: E402
from convert_weights import dequant_fp8_block  # noqa: E402

_HANDLES = {}


def load_hf(snapshot, headers, name):
    from safetensors import safe_open

    dt, shape, shard = headers[name]
    if shard not in _HANDLES:
        _HANDLES[shard] = safe_open(f"{snapshot}/{shard}", framework="pt")
    w = _HANDLES[shard].get_tensor(name)
    if dt == "F8_E4M3":
        s = load_hf(snapshot, headers, name + "_scale_inv").float()
        return dequant_fp8_block(w, s)
    return w.float()


def expected(entry, snapshot, headers) -> torch.Tensor:
    src = [load_hf(snapshot, headers, s) for s in entry["source_tensors"]]
    tr = entry["transform"]
    if tr == "experts_gate_up":
        return torch.cat(src, 0).T
    if tr == "linear_t":
        return src[0].T
    if tr == "kv_b_uk":
        return src[0].view(64, 512, 512)[:, :256, :]
    if tr == "kv_b_uv":
        return src[0].view(64, 512, 512)[:, 256:, :].transpose(1, 2)
    if tr == "conv_taps":
        return src[0][:, 0, :].T
    if tr == "copy":
        return src[0]
    raise ValueError(tr)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "manifest.json"))
    ap.add_argument("--snapshot", default=os.environ.get("GLM53_HF_SNAPSHOT"))
    ap.add_argument("--cache", default=None, help="override manifest cache_dir")
    ap.add_argument("--per-group", type=int, default=32)
    ap.add_argument("--seed", type=int, default=53)
    ap.add_argument("--out", default=None)
    ap.add_argument("--done-dir", default=None, help="use <cache>/.done records instead of a manifest (testing)")
    args = ap.parse_args()
    torch.set_num_threads(32)

    if args.done_dir:
        tensors = {}
        for f in os.listdir(args.done_dir):
            for e in json.load(open(os.path.join(args.done_dir, f)))["entries"]:
                tensors[e["name"]] = e
        cache = args.cache
    else:
        man = json.load(open(args.manifest))
        tensors = man["tensors"]
        cache = args.cache or man["cache_dir"]
    headers = read_headers(args.snapshot)

    by_group = defaultdict(list)
    for name, e in tensors.items():
        by_group[dtype_group(e["dtype"])].append(name)
    rng = random.Random(args.seed)
    results = []
    t0 = time.time()
    for g in sorted(by_group):
        names = sorted(by_group[g])
        pick = names if len(names) <= args.per_group else rng.sample(names, args.per_group)
        for name in sorted(pick):
            e = tensors[name]
            path = os.path.join(cache, e["path"])
            sha_ok = sha256_file(path) == e["sha256"]
            got = decode(path, e["shape"], e["dtype"], e["layout"])
            exp = expected(e, args.snapshot, headers).contiguous()
            assert list(exp.shape) == list(e["shape"]), (name, exp.shape, e["shape"])
            exp_np = exp.numpy().astype(np.float32)
            err = got.astype(np.float64) - exp_np.astype(np.float64)
            r = {
                "name": name,
                "group": g,
                "dtype": e["dtype"],
                "shape": e["shape"],
                "sha256_ok": sha_ok,
                "pcc": pcc(got, exp_np),
                "max_abs_err": float(np.abs(err).max()),
                "rel_rms_err": float(np.sqrt((err**2).mean()) / max(np.sqrt((exp_np.astype(np.float64) ** 2).mean()), 1e-30)),
            }
            if g in ("bf16", "fp32"):
                ref = exp.to(torch.bfloat16).float().numpy() if g == "bf16" else exp_np
                r["bit_exact_vs_rounded_source"] = bool(np.array_equal(got, ref))
            results.append(r)
            print(
                f"{g:5s} {name:60s} pcc={r['pcc']:.6f} max_err={r['max_abs_err']:.3e} rel_rms={r['rel_rms_err']:.3e} sha={'ok' if sha_ok else 'BAD'}"
                + (f" exact={r['bit_exact_vs_rounded_source']}" if "bit_exact_vs_rounded_source" in r else ""),
                flush=True,
            )
    summary = {}
    for g in sorted(by_group):
        rs = [r for r in results if r["group"] == g]
        summary[g] = {
            "n_checked": len(rs),
            "n_in_cache": len(by_group[g]),
            "min_pcc": min(r["pcc"] for r in rs),
            "mean_pcc": float(np.mean([r["pcc"] for r in rs])),
            "max_rel_rms_err": max(r["rel_rms_err"] for r in rs),
            "all_sha256_ok": all(r["sha256_ok"] for r in rs),
        }
        if g in ("bf16", "fp32"):
            summary[g]["all_bit_exact"] = all(r["bit_exact_vs_rounded_source"] for r in rs)
    gates = {
        "bfp8_min_pcc>=0.999": summary.get("bfp8", {}).get("min_pcc", 0) >= 0.999,
        "bf16_bit_exact": summary.get("bf16", {}).get("all_bit_exact", False),
        "fp32_bit_exact": summary.get("fp32", {}).get("all_bit_exact", False),
        "all_sha256_ok": all(s["all_sha256_ok"] for s in summary.values()),
        "bfp4": "reported (no gate)",
    }
    out = {"seconds": time.time() - t0, "seed": args.seed, "per_group": args.per_group, "summary": summary, "gates": gates, "results": results}
    print(json.dumps({"summary": summary, "gates": gates}, indent=1))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=1)
    ok = all(v for k, v in gates.items() if k != "bfp4")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
