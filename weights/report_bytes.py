#!/usr/bin/env python3
"""Markdown tables of the weight-cache byte counts (per dtype group, per layer, per tensor group) from manifest.json."""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))


def gb(b):
    return f"{b / 1e9:.3f}"


def main():
    man = json.load(open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "manifest.json")))
    groups = ["bfp4", "bfp8", "bf16", "fp32"]
    dt2g = {"bfloat4_b": "bfp4", "bfloat8_b": "bfp8", "bfloat16": "bf16", "float32": "fp32"}
    cnt = defaultdict(int)
    for e in man["tensors"].values():
        cnt[dt2g[e["dtype"]]] += 1
    print(f"Total: {man['num_tensors']} tensors, {gb(man['total_bytes'])} GB (tt-metal {man['tt_metal_sha'][:12]})\n")
    print("| dtype group | tensors | GB |\n|---|---:|---:|")
    for g in groups:
        print(f"| {g} | {cnt[g]} | {gb(man['totals_bytes_by_group'].get(g, 0))} |")
    by_kind = defaultdict(lambda: defaultdict(int))
    for e in man["tensors"].values():
        by_kind[e["group"]][dt2g[e["dtype"]]] += e["bytes"]
    print("\n| tensor group | " + " | ".join(groups) + " | total GB |\n|---|" + "---:|" * (len(groups) + 1))
    for k in sorted(by_kind):
        row = by_kind[k]
        print(f"| {k} | " + " | ".join(gb(row.get(g, 0)) for g in groups) + f" | {gb(sum(row.values()))} |")
    print("\n| layer | kind | " + " | ".join(groups) + " | total GB |\n|---|---|" + "---:|" * (len(groups) + 1))
    kinds = {}
    for e in man["tensors"].values():
        if e["layer"] is not None and ("q_proj" in e["path"] or "q_a_proj" in e["path"]):
            kinds[str(e["layer"])] = "KDA" if "q_proj" in e["path"] and "q_a_proj" not in e["path"] else "MLA"
    for layer, row in man["bytes_by_layer"].items():
        k = kinds.get(layer, "")
        if layer not in ("global",):
            mlp = "dense" if int(layer) < 3 else "MoE"
            k = f"{k}+{mlp}" + (" (MTP)" if layer == "45" else "")
        print(f"| {layer} | {k} | " + " | ".join(gb(row.get(g, 0)) for g in groups) + f" | {gb(sum(row.values()))} |")


if __name__ == "__main__":
    main()
