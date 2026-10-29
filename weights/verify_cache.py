#!/usr/bin/env python3
"""Re-hash every file of the GLM-5.3-Flash weight cache against weights/manifest.json (parallel, read-only).

Also checks that the logical-shape/dtype -> expected byte size relation holds for every TILE tensor and that the
cache directory holds no stray / missing .tensorbin files.

    python weights/verify_cache.py [--workers 32] [--out doc/ttnn/00c-weights-goldens/verify_cache.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
TILE_BYTES = {"bfloat8_b": 1088, "bfloat4_b": 576, "bfloat16": 2048, "float32": 4096}
ELEM_BYTES = {"bfloat16": 2, "float32": 4}


def data_bytes(path):
    with open(path, "rb") as f:
        (hdr,) = struct.unpack("<Q", f.read(8))
    return os.path.getsize(path) - 8 - hdr


def expected_data_bytes(e):
    shape = e["shape"]
    if e["layout"] == "ROW_MAJOR":
        return math.prod(shape) * ELEM_BYTES[e["dtype"]]
    h, w = (1, shape[0]) if len(shape) == 1 else (shape[-2], shape[-1])
    batch = math.prod(shape[:-2]) if len(shape) > 2 else 1
    return batch * math.ceil(h / 32) * math.ceil(w / 32) * TILE_BYTES[e["dtype"]]


def check(args):
    name, e, cache = args
    p = os.path.join(cache, e["path"])
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(16 << 20), b""):
            h.update(b)
    return name, h.hexdigest() == e["sha256"], data_bytes(p) == expected_data_bytes(e), os.path.getsize(p) == e["bytes"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(HERE, "manifest.json"))
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    man = json.load(open(args.manifest))
    cache = man["cache_dir"]
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        res = list(pool.map(check, [(n, e, cache) for n, e in man["tensors"].items()]))
    on_disk = set()
    for d, _, fs in os.walk(cache):
        for f in fs:
            if f.endswith(".tensorbin"):
                on_disk.add(os.path.relpath(os.path.join(d, f), cache))
    listed = {e["path"] for e in man["tensors"].values()}
    out = {
        "manifest": args.manifest,
        "cache_dir": cache,
        "n_tensors": len(res),
        "sha256_mismatch": [n for n, ok, _, _ in res if not ok],
        "size_formula_mismatch": [n for n, _, ok, _ in res if not ok],
        "bytes_field_mismatch": [n for n, _, _, ok in res if not ok],
        "missing_files": sorted(listed - on_disk),
        "stray_files": sorted(on_disk - listed),
        "total_bytes": sum(e["bytes"] for e in man["tensors"].values()),
        "seconds": round(time.time() - t0, 1),
    }
    out["ok"] = not any(out[k] for k in ("sha256_mismatch", "size_formula_mismatch", "bytes_field_mismatch", "missing_files", "stray_files"))
    print(json.dumps({k: (v if not isinstance(v, list) else len(v)) for k, v in out.items()}, indent=1))
    if args.out:
        json.dump(out, open(args.out, "w"), indent=1)
    raise SystemExit(0 if out["ok"] else 1)


if __name__ == "__main__":
    main()
