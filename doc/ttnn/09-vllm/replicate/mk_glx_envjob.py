"""The Galaxy form of a served env job (the fleet kit, program 2026-10-27T16:54:15Z; as int27n29h7p2hk4b_img_env_glx.done,
11:07:17Z (2)): the host-8chip env job with GLM53_GRID_X=11 (the grid pin the Galaxy image was built under) and
GLM53_TP_IMAGE_DIR = the Galaxy's own image (same image key as host-8chip's), in args.env AND stage9_overlay.keys (the
pre-mesh check's env compare), the authority note extended. Refuses an env job whose image key differs from the pin's.
usage: mk_glx_envjob.py <host-8chip env job> <out> [--pin fde4697e51b2-tp-ds-v1-2x4-629a5c1a384e]"""
import json
import os
import sys

src, out = sys.argv[1], sys.argv[2]
pin = sys.argv[sys.argv.index("--pin") + 1] if "--pin" in sys.argv else "fde4697e51b2-tp-ds-v1-2x4-629a5c1a384e"
j = json.load(open(src))
env = j["args"]["env"]
old = env.get("GLM53_TP_IMAGE_DIR", "")
key = pin.split("-tp-")[0]
if os.path.basename(old).split("-tp-")[0] != key:
    sys.exit(f"{src}: its image {os.path.basename(old)} is not the pin's image key {key}")
new_dir = os.path.join(os.path.dirname(old), pin)
for d in (env, (j.get("stage9_overlay") or {}).get("keys")):
    if d is None:
        continue
    d["GLM53_GRID_X"] = "11"
    d["GLM53_TP_IMAGE_DIR"] = new_dir
o = j.setdefault("stage9_overlay", {})
o["authority"] = (o.get("authority") or "") + (
    f" | glx variant (fleet kit, program 2026-10-27T16:54:15Z; mk_glx_envjob.py from {os.path.basename(src)}): GLM53_GRID_X=11"
    f" and GLM53_TP_IMAGE_DIR = the Galaxy's own image {pin.rsplit('-', 1)[1]} (image key {key}, as host-8chip's {os.path.basename(old).rsplit('-', 1)[1]})")
json.dump(j, open(out + ".tmp", "w"), indent=1)
os.replace(out + ".tmp", out)
print(f"{out}: GRID_X 11, TP_IMAGE_DIR {new_dir}")
