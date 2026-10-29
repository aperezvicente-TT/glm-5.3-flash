#!/usr/bin/env python3
"""Build the shared GLM-5.3-Flash bfp4/bfp8 tile cache from the fp8 HF checkpoint (CPU only).

Run under the device guard (no Tenstorrent device may be opened in this stage):
    source env.sh; glm53_activate_metal /opt/glm53/src/tt-metal
    .agents/scripts/no-tt-devices python weights/convert_weights.py --workers 40

- Streams the 62 shards in order (work units sorted by source shard; tensors are read one at a time through
  safetensors mmap, so resident memory stays ~ workers x one expert).
- fp8 e4m3 + 128x128 f32 `*_scale_inv` block scales are dequantised with the tt-metal-dsv4-flash
  `quant.dequantize_fp8_block` semantics (w.float() * per-block scale), rounded to bf16 and packed with
  `ttnn._ttnn.fast_bfp.to_ttnn` (byte-identical to ttnn's host packer, commit 5786a850eb4); bf16/f32 tensors
  go through `ttnn.from_torch` on host. Everything is written with `ttnn.dump_tensor`.
- Resumable: each finished work unit drops `<out>/.done/<unit>.json`; reruns skip finished units.
- At the end the manifest (`weights/manifest.json`) is assembled from the done records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from glm53_cache_spec import BF16, BFP4, BFP8, FP32, Task, build_tasks, dtype_group, read_headers  # noqa: E402

EXPERTS_PER_UNIT = 16

_G = {}  # per-worker globals


def sha256_file(path: str, bufsize: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def dequant_fp8_block(w, scale, block=(128, 128)):
    """tt-metal-dsv4-flash quant.dequantize_fp8_block semantics with a float32 scale_inv: w * scale per block."""
    import torch

    R, C = w.shape
    bh, bw = block
    wf = w.to(torch.float32)
    if R % bh == 0 and C % bw == 0:
        return (wf.view(R // bh, bh, C // bw, bw) * scale.view(R // bh, 1, C // bw, 1)).view(R, C)
    s = scale.repeat_interleave(bh, 0).repeat_interleave(bw, 1)[:R, :C]
    return wf * s


def _worker_init(snapshot: str, out_dir: str, threads: int, headers: dict):
    import torch

    torch.set_num_threads(threads)
    import ttnn  # noqa: F401  (host-only use; the device guard hides /dev/tenstorrent)

    _G.update(snapshot=snapshot, out=out_dir, threads=threads, headers=headers, handles={})


def _handle(shard: str):
    from safetensors import safe_open

    hs = _G["handles"]
    if shard not in hs:
        if len(hs) > 4:
            hs.clear()
        hs[shard] = safe_open(f"{_G['snapshot']}/{shard}", framework="pt")
    return hs[shard]


def load_source(hf: str):
    """HF tensor as float32 (fp8 dequantised with its block scale). Returns (tensor, [shards])."""
    import torch

    dt, shape, shard = _G["headers"][hf]
    w = _handle(shard).get_tensor(hf)
    shards = [shard]
    if dt == "F8_E4M3":
        sname = hf + "_scale_inv"
        sdt, sshape, sshard = _G["headers"][sname]
        s = _handle(sshard).get_tensor(sname).to(torch.float32)
        shards.append(sshard)
        return dequant_fp8_block(w, s), shards
    return w.to(torch.float32), shards


def to_tt(task: Task, src: list):
    """Apply the task transform to float32 sources and return a host ttnn.Tensor."""
    import torch
    import ttnn

    fb = ttnn._ttnn.fast_bfp
    tt_dtype = {BFP4: ttnn.bfloat4_b, BFP8: ttnn.bfloat8_b, BF16: ttnn.bfloat16, FP32: ttnn.float32}[task.dtype]
    layout = ttnn.TILE_LAYOUT if task.layout == "TILE" else ttnn.ROW_MAJOR_LAYOUT
    tr = task.transform
    if tr == "experts_gate_up":
        x = torch.cat([src[0], src[1]], dim=0)  # [2I, H] rows: gate then up (HF Concatenate(dim=1) on [E,2I,H])
        transpose = True
    elif tr == "linear_t":
        x = src[0]
        transpose = True
    elif tr == "kv_b_uk":
        x = src[0].view(64, 512, 512)[:, :256, :]  # [h, nope, latent]
        transpose = False
    elif tr == "kv_b_uv":
        x = src[0].view(64, 512, 512)[:, 256:, :].transpose(1, 2)  # [h, latent, v]
        transpose = False
    elif tr == "conv_taps":
        x = src[0][:, 0, :].transpose(0, 1)  # [C,1,K] -> [K,C]
        transpose = False
    elif tr == "copy":
        x = src[0]
        transpose = False
    else:
        raise ValueError(tr)

    if task.dtype in (BFP4, BFP8):
        xb = x.contiguous().to(torch.bfloat16).contiguous()
        t = fb.to_ttnn(xb.view(torch.int16), tt_dtype, transpose=transpose, nthreads=_G["threads"])
    else:
        if transpose:
            x = x.transpose(-1, -2)
        x = x.contiguous().to(torch.bfloat16 if task.dtype == BF16 else torch.float32).contiguous()
        t = ttnn.from_torch(x, dtype=tt_dtype, layout=layout)
    if list(t.shape) != list(task.shape):
        raise RuntimeError(f"{task.name}: produced shape {list(t.shape)} != spec {task.shape}")
    return t


def run_unit(unit):
    """Convert one work unit (list of tasks). Returns (unit_id, entries, seconds, bytes_read)."""
    import ttnn

    unit_id, tasks = unit
    done_path = f"{_G['out']}/.done/{unit_id}.json"
    if os.path.exists(done_path):
        return unit_id, json.load(open(done_path))["entries"], 0.0, 0
    t0 = time.time()
    entries = []
    nread = 0
    cache: dict = {}
    for task in tasks:
        src = []
        shards = []
        for hf in task.sources:
            if hf not in cache:
                cache.clear() if len(cache) > 2 else None
                cache[hf] = load_source(hf)
            x, sh = cache[hf]
            src.append(x)
            shards.extend(sh)
            nread += x.numel()
        t = to_tt(task, src)
        final = f"{_G['out']}/{task.path}"
        os.makedirs(os.path.dirname(final), exist_ok=True)
        tmp = final[: -len(".tensorbin")] + f".tmp{os.getpid()}.tensorbin"
        ttnn.dump_tensor(tmp, t)
        del t
        digest = sha256_file(tmp)
        os.replace(tmp, final)
        e = {
            "name": task.name,
            "path": task.path,
            "shape": task.shape,
            "dtype": task.dtype,
            "layout": task.layout,
            "sha256": digest,
            "bytes": os.path.getsize(final),
            "source_shard": sorted(set(shards)),
            "source_tensors": task.sources,
            "transform": task.transform,
            "group": task.group,
            "layer": task.layer,
        }
        entries.append(e)
    dt = time.time() - t0
    os.makedirs(os.path.dirname(done_path), exist_ok=True)
    with open(done_path + ".tmp", "w") as f:
        json.dump({"unit": unit_id, "seconds": dt, "entries": entries}, f)
    os.replace(done_path + ".tmp", done_path)
    return unit_id, entries, dt, nread


def make_units(tasks: list[Task], headers: dict):
    """Group tasks into work units (expert chunks / per-layer misc / singletons) ordered by source shard."""
    by_key = defaultdict(list)
    for t in tasks:
        if t.group == "routed_expert":
            eid = int(t.name.split(".")[4])
            by_key[f"L{t.layer:02d}_experts_{eid // EXPERTS_PER_UNIT:02d}"].append(t)
        elif t.layer is None:
            by_key[t.name.replace(".", "_")].append(t)
        else:
            by_key[f"L{t.layer:02d}_misc"].append(t)

    def first_shard(ts):
        return min(headers[s][2] for t in ts for s in t.sources)

    return sorted(by_key.items(), key=lambda kv: (first_shard(kv[1]), kv[0]))


class _NullPool:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def imap_unordered(self, *a, **k):
        return iter(())


def git_sha(path: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=os.environ.get("GLM53_HF_SNAPSHOT"))
    ap.add_argument("--out", default=os.environ.get("GLM53_CACHE_DIR"))
    ap.add_argument("--manifest", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "manifest.json"))
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--threads", type=int, default=4, help="torch / fast_bfp threads per worker")
    ap.add_argument("--only", default=None, help="regex on unit id (testing)")
    ap.add_argument("--no-manifest", action="store_true")
    ap.add_argument("--manifest-only", action="store_true",
                    help="rebuild the manifest from <out>/.done records without converting anything")
    args = ap.parse_args()

    if os.path.exists("/dev/tenstorrent") and os.listdir("/dev/tenstorrent"):
        sys.exit("refusing to run: Tenstorrent devices visible; run under .agents/scripts/no-tt-devices")

    import multiprocessing as mp
    import re

    tt_home = os.environ.get("TT_METAL_HOME", "")
    tt_sha = git_sha(tt_home)
    t_start = time.time()
    headers = read_headers(args.snapshot)
    tasks = build_tasks(headers)
    units = make_units(tasks, headers)
    if args.only:
        units = [u for u in units if re.search(args.only, u[0])]
    print(f"{len(tasks)} tensors in {len(units)} units; tt-metal {tt_sha} ({tt_home}); out {args.out}", flush=True)
    os.makedirs(args.out, exist_ok=True)

    all_entries = []
    done_units = 0
    elems = 0
    unit_seconds = 0.0
    if args.manifest_only:
        for unit_id, _ in units:
            rec = json.load(open(f"{args.out}/.done/{unit_id}.json"))
            all_entries.extend(rec["entries"])
            unit_seconds += rec["seconds"]
        units = []
    ctx = mp.get_context("fork")
    with ctx.Pool(0 if args.manifest_only else args.workers, initializer=_worker_init,
                  initargs=(args.snapshot, args.out, args.threads, headers)) if not args.manifest_only else _NullPool() as pool:
        for unit_id, entries, dt, nread in pool.imap_unordered(run_unit, units, chunksize=1):
            done_units += 1
            elems += nread
            all_entries.extend(entries)
            if done_units % 25 == 0 or done_units == len(units):
                el = time.time() - t_start
                print(
                    f"[{el:7.1f}s] {done_units}/{len(units)} units, {len(all_entries)} tensors, "
                    f"{elems / 1e9:.1f} G source elems this run ({elems / 1e9 / max(el, 1e-9):.2f} G/s)",
                    flush=True,
                )
    wall = time.time() - t_start
    print(f"conversion wall {wall:.1f}s", flush=True)
    if args.no_manifest or args.only:
        return

    for e in all_entries:
        e["tt_metal_sha"] = tt_sha
        e["logical_shape"] = e["shape"]
    if not args.manifest_only:
        done_dir = f"{args.out}/.done"
        unit_seconds = sum(json.load(open(os.path.join(done_dir, f)))["seconds"] for f in os.listdir(done_dir))
    all_entries.sort(key=lambda e: e["name"])
    names = [e["name"] for e in all_entries]
    assert len(names) == len(set(names)) == len(tasks), (len(names), len(set(names)), len(tasks))
    totals = defaultdict(int)
    per_layer = defaultdict(lambda: defaultdict(int))
    for e in all_entries:
        g = dtype_group(e["dtype"])
        totals[g] += e["bytes"]
        per_layer["global" if e["layer"] is None else str(e["layer"])][g] += e["bytes"]
    manifest = {
        "format": "glm53-weight-cache/v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hf_model": "zai-org/GLM-5.3-Flash",
        "hf_snapshot": args.snapshot,
        "cache_dir": args.out,
        "tt_metal_sha": tt_sha,
        "tt_metal_home": tt_home,
        "writer": "ttnn.dump_tensor (fast_bfp.to_ttnn for bfp4/bfp8; ttnn.from_torch host for bf16/f32)",
        "conventions": {
            "linear": "2-D linear weights stored transposed as [in, out] (y = x @ W)",
            "routed_experts": "gate_up_proj [4096, 4096] = [gate(2048) | up(2048)] columns; down_proj [2048, 4096]; bfloat4_b",
            "kv_b_proj": "W_uk [64,256,512] = kv_b.view(64,512,512)[:, :256, :]; W_uv [64,512,256] = kv_b.view(64,512,512)[:, 256:, :].mT",
            "conv_taps": "[C,1,K] -> [K,C]; row j multiplies input row t-(K-1)+j",
            "vectors": "1-D tensors keep HF shape (TILE pads to [32,N])",
            "embedding": "ROW_MAJOR bf16 (ttnn.embedding requires ROW_MAJOR weights)",
            "fp8_dequant": "w.float() * scale_inv per 128x128 block -> bf16 (RNE) -> packer",
            "mtp_layer": 45,
        },
        "totals_bytes_by_group": dict(totals),
        "bytes_by_layer": {k: dict(v) for k, v in sorted(per_layer.items(), key=lambda kv: (kv[0] == "global", kv[0].zfill(3)))},
        "total_bytes": sum(totals.values()),
        "num_tensors": len(all_entries),
        "conversion_wall_seconds_last_run": None if args.manifest_only else wall,
        "conversion_runs": "run 1 2026-10-15T02:39:50Z, 53/823 units before a host crash; run 2 02:50:36Z-03:13:35Z "
                           "(1378 s wall, 40 workers x 4 threads); see doc/ttnn/00c-weights-goldens/work_log.md",
        "worker_unit_seconds_total": round(unit_seconds, 1),
        "tensors": {e["name"]: e for e in all_entries},
    }
    tmp = args.manifest + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=0, separators=(",", ":"))
    os.replace(tmp, args.manifest)
    print(f"manifest: {args.manifest} ({len(all_entries)} tensors, {manifest['total_bytes'] / 1e9:.2f} GB)")
    for g, b in sorted(totals.items()):
        print(f"  {g}: {b / 1e9:.3f} GB")


if __name__ == "__main__":
    main()
