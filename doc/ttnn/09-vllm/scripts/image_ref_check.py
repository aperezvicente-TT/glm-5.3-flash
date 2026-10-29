"""An image's CONTENT against a known-good reference image of the same construction (program 2026-10-27T04:09:06Z: the
Galaxy pin 629a5c1a384e, written under KMD 2.9.1-pre / dma_mask_bits 64, differed from host-8chip's 1530493535ed in 6,376 of
6,712 4-MiB blocks on 7 of 8 chips -- the restore's own check compares DRAM only against the image's hashes, so a bad image
restores 'clean'). PASS = the same segments / block size and, per chip, at most --max-diff blocks whose hash differs (the
harvest-dependent L1 / arg-table blocks: 4 per chip on a correct Galaxy build). Chips are matched by MESH COORDINATE
(the manifest's per-chip 'coord'), as the restore matches them (tp_image.chip_view): device ids differ across hosts (the
Galaxy's device 0 is coordinate (1, 0), host-8chip's is (1, 3)); 2026-10-27T04:29:33Z the coordinator found my first,
device-id version flagging valid images. Host-only (reads two manifest.json files).
usage: image_ref_check.py <image manifest.json> <reference manifest.json> [--max-diff N] -> one line; rc 0 pass, 1 fail, 2 error"""
import json
import sys

args = [a for a in sys.argv[1:] if not a.startswith("--")]
max_diff = 16
if "--max-diff" in sys.argv:
    max_diff = int(sys.argv[sys.argv.index("--max-diff") + 1])
    args = [a for a in args if a != str(max_diff)]
try:
    img, ref = (json.load(open(p)) for p in args[:2])
except (OSError, ValueError, IndexError) as e:
    print(f"image_ref_check ERROR: {type(e).__name__}: {e}")
    sys.exit(2)
if img.get("segments") != ref.get("segments") or img.get("block_bytes") != ref.get("block_bytes"):
    print(f"image_ref_check FAIL: layout differs (segments equal {img.get('segments') == ref.get('segments')}, block_bytes "
          f"{img.get('block_bytes')} vs {ref.get('block_bytes')})")
    sys.exit(1)
by_coord = {tuple(c["coord"]): (k, c) for k, c in img["chips"].items() if "coord" in c}
per, mapping = {}, {}
for chip, c in sorted(ref["chips"].items(), key=lambda kv: int(kv[0])):
    coord = tuple(c.get("coord", ()))
    k, ic = by_coord.get(coord, (None, {}))
    a, b = c["block_hashes"], ic.get("block_hashes", [])
    key = f"{list(coord)}"
    per[key] = (sum(1 for x, y in zip(a, b) if x != y) + abs(len(a) - len(b))) if k is not None else len(a)
    mapping[key] = f"img {k} / ref {chip}"
if len(by_coord) != len(img["chips"]) or set(by_coord) != {tuple(c.get("coord", ())) for c in ref["chips"].values()}:
    print(f"image_ref_check FAIL: mesh coordinates differ (image {sorted(by_coord)}, reference "
          f"{sorted(tuple(c.get('coord', ())) for c in ref['chips'].values())})")
    sys.exit(1)
bad = {k: v for k, v in per.items() if v > max_diff}
nblk = len(next(iter(ref["chips"].values()))["block_hashes"])
print(("image_ref_check PASS" if not bad else "image_ref_check FAIL")
      + f": differing {ref['block_bytes'] >> 20}-MiB blocks per mesh coordinate {per} (bar <= {max_diff} of {nblk} per chip; "
        f"chips by coordinate {mapping})")
sys.exit(1 if bad else 0)
