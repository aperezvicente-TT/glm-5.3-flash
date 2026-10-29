#!/usr/bin/env bash
# Run one Stage 9 row on a Galaxy 2x4 block (program 2026-10-27T00:08:40Z: '<stage> glx:blkN: GO'), unchanged scripts:
# the host profile + the path namespace (ns_paths.sh) + the block's lock group (tt-lock-galaxy blk<N> exports TT_VISIBLE_DEVICES).
# usage: glx_block_run.sh <N> '<the full GO line>' [script, default the resident target]   (no device opens before the
# operator's go; the first light: script = scripts/targets/glx_firstlight_k3.sh)
set -u
N=${1:?block 0..3}; GO=${2:?GO line}; SCRIPT=${3:-/opt/glm53/work-s9/resident_target.sh}
case "$N" in 0|1|2|3) ;; *) echo "block $N?"; exit 2;; esac
K=$(dirname "$(readlink -f "$0")"); . "$K/host_glx.sh"
# the GO route (no share on the Galaxy): the coordinator's GO line, given here in full, is mirrored into the note the
# launcher reads (/opt/glm53/work-s9/COORDINATOR_NOTE.txt in the namespace) before the launch; launch_resident.sh
# then checks it as on host-8chip (in the note, <= 85 s old, no later '9 HOLD'): run this within 85 s of the GO
# the fleet (program 2026-10-27T16:54:15Z): one GO line '9 glx:fleet: GO (dp4)' covers blk0..blk3 (glx_fleet_run.sh)
case "$GO" in *"9 glx:blk$N: GO"*|*"9 mesh: GO"*|*"9 glx:fleet: GO"*) ;; *) echo "not a 9 GO line for blk$N: $GO"; exit 2;; esac
# both notes: launch_resident.sh reads work-s9's, serve_window.sh the block's own ($GLM53_WORK; host_glx.sh, 04:01:01Z (2))
for NOTE9 in $R_GLM53/work-s9/COORDINATOR_NOTE.txt ${GLX_WORK:+$GLX_WORK/COORDINATOR_NOTE.txt}; do
  mkdir -p "$(dirname "$NOTE9")"; grep -qxF -- "$GO" "$NOTE9" 2>/dev/null || echo "$GO" >> "$NOTE9"; done
# the block's devices (program RULING 2026-10-28T01:08:26Z; the device-id rule): the lock tool is the truth -- blkN =
# /dev/tenstorrent 8N..8N+7 in groups.conf, holders checked on those nodes -- while TT_VISIBLE_DEVICES takes UMD's BDF-sorted
# ranks, a different order on this box (bisect arm a under 'blk3' opened UMD 24-31 = /dev 16-23 = the lock tool's blk2).
# Translated here exactly as 14's glx14_row.sh: /dev id -> /sys/class/tenstorrent/tenstorrent!<id>/device (BDF) -> its rank
# among the sorted BDFs; the inverse (rank -> BDF -> /dev id) must give back the group's /dev set; unmappable = refused.
# serve_window.sh passes GLX_TT_VISIBLE_DEVICES after the lock tool's own export (which stays untouched).
GC=${TT_LOCK_DIR:?host_glx.sh}/groups.conf
DEVS=$(grep -E "^blk$N=" "$GC" 2>/dev/null | head -1 | cut -d= -f2-)
[ -n "$DEVS" ] || { echo "refusing: no blk$N in $GC"; exit 7; }
RANKS=$(python3 -I - "$DEVS" <<'PY2'
import glob, os, sys
m = {}
for d in glob.glob("/sys/class/tenstorrent/tenstorrent!*"):
    m[int(d.rsplit("!", 1)[1])] = os.path.basename(os.path.realpath(os.path.join(d, "device")))
if not m:
    sys.exit("no /sys/class/tenstorrent entries")
order = sorted(m.values())
devs = [int(v) for v in sys.argv[1].replace(" ", "").split(",") if v != ""]
miss = [v for v in devs if v not in m]
if miss:
    sys.exit(f"/dev ids without a sysfs entry: {miss}")
ranks = sorted(order.index(m[v]) for v in devs)
back = sorted(next(k for k, b in m.items() if b == order[r]) for r in ranks)
if back != sorted(devs):
    sys.exit(f"inverse mismatch: ranks {ranks} -> /dev {back} != {sorted(devs)}")
print(",".join(str(r) for r in ranks))
PY2
) || { echo "refusing: cannot map blk$N's /dev ids $DEVS to UMD ranks"; exit 7; }
echo "$(date -u +%FT%TZ) glx_block_run blk$N: groups.conf /dev ids $DEVS -> TT_VISIBLE_DEVICES (UMD BDF ranks) $RANKS (inverse checked: the ranks' /dev set == the lock group's nodes)"
export GLX_TT_VISIBLE_DEVICES=$RANKS GLX_DEV_IDS=$DEVS
exec env GLM53_LOCK_GROUP=blk$N GLM53_MESH_TAG=blk$N GLM53_GO_LINE="$GO" bash "$K/ns_paths.sh" bash "$SCRIPT"
