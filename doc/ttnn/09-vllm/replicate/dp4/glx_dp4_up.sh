#!/usr/bin/env bash
# The Galaxy TP8 x DP4 bring-up (operator delivery, program 2026-10-28T12:20:55Z items (2) + (3); form 12:25:35Z): four standing
# TP8 servers, one per 2x4 block, launched IN PARALLEL (one process per block, no open / close churn), each a HYBRID tray of
# pd/glx_pd_role.sh (GLX_PD_ROLE=H, GLX_PD_TAG=<form tag>dp4b<N>: its own env job + target from mk_dp4_envjobs.py, port 8100 + N,
# the block's TT_VISIBLE set from groups.conf through glx_block_run.sh, the block's OWN image root, the strict startup self-test
# before the socket binds, TT_METAL_OPERATION_TIMEOUT_SECONDS=0 + serve_window.sh's watchdog). Then the per-block acceptance
# (item (3)): the strict self-test verdict + the cold ladder 128 / 2,048 / 32,000 / 131,072 on every block (selftest_loop.py).
# The coordinator runs it on the box host, outside the namespace (the trays enter it through glx_block_run.sh).
# usage: glx_dp4_up.sh '<the 9 glx:fleet: GO (dp4...) line>'     (run within 85 s of the GO: launch_resident.sh's age check)
#        GLX_DP4_FORM=t6d|f2 (default t6d = k4d8t6d; f2 = the fallback)   GLX_DP4_BLOCKS='0 1 2 3'   GLX_DP4_DRY=1 (checks only)
#        GLX_DP4_ACCEPT=0 (no acceptance cells)   GLX_DP4_UP_S (health wait, default 2400 s)
# The image per block: GLX_DP4_IMAGES (default $R_GLM53/work-s9/dp4_images.conf), one line '<N> <image root, namespace path> <pin>'
# per block (dp4/README.md: blk2 / blk3 = the blk3 root's 1530 image; blk0 / blk1 = the 629a map or a device build of their own;
# the 1530 map fails blk0 / blk1 at upload #179). Given to the tray as GLX_IMAGE_ROOT + GLX_PIN.
# Refuses before any launch: a block without its image line, the image (<root>/<pin>/manifest.json; NO fallback to another
# root), its H mock verdict next to that image (mock_verdict_glx_<tag>dp4b<N>pdH.json, pass true), its env job (role H,
# block N, form) or its target. A tray that refuses or is not healthy in time STOPs this launch's trays
# (GLX_DP4_FAIL_KEEP=1 keeps them). The trays stay up after the acceptance (glx_dp4_stop.sh closes them in order).
set -u
D4=$(dirname "$(readlink -f "$0")"); K=$(dirname "$D4"); PD=$K/pd
. "$K/host_glx.sh"
FORM=${GLX_DP4_FORM:-t6d}
case $FORM in t6d) TAG=n29h7p2hk4d8t6d;; f2) TAG=n29h7p2hk4d8t5wp6f2;; t6di) TAG=n29h7p2hk4d8t6di;; f2i) TAG=n29h7p2hk4d8t5wp6f2i;; f2i3) TAG=n29h7p2hk4d8t5wp6f2i3;; f2i3b) TAG=n29h7p2hk4d8t5wp6f2i3b;; f2i3c) TAG=n29h7p2hk4d8t5wp6f2i3c;; f2i3cxb) TAG=n29h7p2hk4d8t5wp6f2i3cxb;; f2i3d) TAG=n29h7p2hk4d8t5wp6f2i3d;; f2i3d2) TAG=n29h7p2hk4d8t5wp6f2i3d2;; f2i3d3) TAG=n29h7p2hk4d8t5wp6f2i3d3;; smu2v4m2xbnp) TAG=n29h7p2hk4d8t5wsmu2v4m2xbnp;; f2i3d4) TAG=n29h7p2hk4d8t5wp6f2i3d4;; smu2v4m2xbv4) TAG=n29h7p2hk4d8t5wsmu2v4m2xbv4;; df2gv4c4xbv4) TAG=n29h7p2hk4d8t5wdf2gv4c4xbv4;; f2i3d5) TAG=n29h7p2hk4d8t5wp6f2i3d5;; smu2v4m2xbv5) TAG=n29h7p2hk4d8t5wsmu2v4m2xbv5;; f2i3d6) TAG=n29h7p2hk4d8t5wp6f2i3d6;; smu2v4m2xbv6) TAG=n29h7p2hk4d8t5wsmu2v4m2xbv6;; f2i3d7) TAG=n29h7p2hk4d8t5wp6f2i3d7;; smu2v4m2xbv7) TAG=n29h7p2hk4d8t5wsmu2v4m2xbv7;; smu2v4m2xb) TAG=n29h7p2hk4d8t5wsmu2v4m2xb;; smu2v4m2xbgc) TAG=n29h7p2hk4d8t5wsmu2v4m2xbgc;; df2gv4c4xb) TAG=n29h7p2hk4d8t5wdf2gv4c4xb;; df2g4v2) TAG=n29h7p2hdf2g4v2;; smu2g4n4) TAG=n29h7p2hsmu2g4n4;; t6d2) TAG=n29h7p2hk4d8t6d2;; smu2g) TAG=n29h7p2hsmu2g;; smu2g4) TAG=n29h7p2hsmu2g4;; df2g) TAG=n29h7p2hdf2g;; df2g4) TAG=n29h7p2hdf2g4;; *) echo "refusing: GLX_DP4_FORM=$FORM (t6d, f2, t6di, f2i, f2i3, f2i3b, smu2g, smu2g4, df2g or df2g4)"; exit 2;; esac
BLOCKS=${GLX_DP4_BLOCKS:-0 1 2 3}
IMGS=${GLX_DP4_IMAGES:-$R_GLM53/work-s9/dp4_images.conf}
NSD=/opt/glm53/data
img() { awk -v b="$1" '$1 == b {print $'"$2"'}' "$IMGS" 2>/dev/null | tail -1; }  # <block> <2 = root | 3 = pin>
OUT=${GLX_DP4_OUT:-$R_GLM53/work-s9/dp4_runs/dp4_${FORM}_$(date -u +%Y%m%dT%H%M%S)}; mkdir -p "$OUT"
PY=${GLX_VENV:-$R_GLM53/venv-lmcache}/bin/python
log() { echo "$(date -u +%FT%TZ) glx_dp4_up: $*" | tee -a "$OUT/run.log"; }
work() { [ "$1" = 0 ] && echo "$R_GLM53/work-s9" || echo "$R_GLM53/work-s9-blk$1"; }
GO=${1:?the '9 glx:fleet: GO (dp4...)' line}
case "$GO" in *"9 glx:fleet: GO"*) ;; *) log "refusing: not the fleet GO form '9 glx:fleet: GO (...)': $GO"; exit 2;; esac
# (1) every block's inputs, before anything starts
for b in $BLOCKS; do
  bt=${TAG}dp4b$b
  ej=$R_GLM53/work-s9/envjobs/int27${bt}pdH_img_env.done
  [ -s "$ej" ] || { log "refusing: blk$b has no env job $ej (mk_dp4_envjobs.py --form $FORM, in the kit tar)"; exit 2; }
  /usr/bin/python3 -I -c 'import json, sys; o = json.load(open(sys.argv[1]))["stage9_overlay"]
sys.exit(0 if (o.get("role"), o.get("block"), o.get("dp4_form")) == ("H", int(sys.argv[2]), sys.argv[3]) else 1)' "$ej" $b $FORM \
    || { log "refusing: $ej is not the H overlay of blk$b, form $FORM"; exit 2; }
  [ -s "$R_WS/doc/ttnn/09-vllm/scripts/targets/int27${bt}_resident.sh" ] || { log "refusing: no target int27${bt}_resident.sh in $R_WS"; exit 2; }
  root=$(img $b 2); pin=$(img $b 3)
  [ -n "$root" ] && [ -n "$pin" ] || { log "refusing: blk$b has no image line in $IMGS ('<N> <ns image root> <pin>', dp4/README.md)"; exit 2; }
  own=$R_DATA${root#$NSD}
  [ -s "$own/$pin/manifest.json" ] || { log "refusing: blk$b's image $own/$pin has no manifest (dp4/README.md, the image step)"; exit 2; }
  mv_=$own/$pin/mock_verdict_glx_${bt}pdH.json
  /usr/bin/python3 -I -c 'import json, sys; sys.exit(0 if json.load(open(sys.argv[1])).get("pass") else 1)' "$mv_" 2>/dev/null \
    || { log "refusing: blk$b has no passing H mock verdict $mv_ (GLX_MOCK_IMAGE_ROOT=$root GLX_TAG=$bt GLX_PD_ROLE=H glx_image_mock.sh $pin <blk$b descriptor>)"; exit 2; }
  log "blk$b: env job, target, image $root/$pin and its H mock verdict present (port $((8100 + b)))"
done
if [ "${GLX_DP4_DRY:-0}" = 1 ]; then
  for b in $BLOCKS; do log "DRY blk$b: env GLX_IMAGE_ROOT=$(img $b 2) GLX_PIN=$(img $b 3) GLX_PD_ROLE=H GLX_PD_TAG=${TAG}dp4b$b TT_METAL_OPERATION_TIMEOUT_SECONDS=0 bash $K/glx_block_run.sh $b '<GO>' $PD/glx_pd_role.sh"; done
  exit 0
fi
tray_run() {  # <block>: this launch's H run dir on the block ('' until it exists)
  local r; r=$(ls -td $(work $1)/runs/resident_int27${TAG}dp4b$1pdH_glxb$1_* 2>/dev/null | grep -v '\.json$' | head -1)
  [ -n "$r" ] && [[ ! "${r##*_}" < "$LSTAMP" ]] && echo "$r"
}
stop_trays() { GLX_DP4_FORM=$FORM GLX_DP4_BLOCKS="$BLOCKS" bash "$D4/glx_dp4_stop.sh" --since "$LSTAMP" 2>&1 | tee -a "$OUT/run.log"; }
fail() { log "launch FAIL: $*"; [ "${GLX_DP4_FAIL_KEEP:-0}" = 1 ] || stop_trays; exit 4; }
# (2) the four trays in parallel
LSTAMP=$(date -u +%Y%m%dT%H%M%S)
# program [program] 2026-10-28T14:59:19Z: (a) serve_window.lock outlives the device holder ~3 s after an orderly close, so a
# launch keyed on the holder alone was refused (3 times): wait for each block's lock itself (flock -n), at most GLX_DP4_LOCK_WAIT_S;
# (b) the blk1 H mock bound *:8102 and the blk2 relaunch died 'Address already in use': vLLM's own ZMQ ports come from
# vllm.utils.network_utils.get_open_port(), which starts at $VLLM_PORT and walks up past busy ports (or takes an OS ephemeral port):
# every tray pins VLLM_PORT to 8200 + 10 N (the image mocks to their port + 1000), never into the fleet's 8100..8103
for b in $BLOCKS; do
  lk=$(work $b)/serve_window.lock; t0=$(date +%s)
  while [ -e "$lk" ] && ! flock -n "$lk" true 2>/dev/null; do
    [ $(( $(date +%s) - t0 )) -ge ${GLX_DP4_LOCK_WAIT_S:-90} ] && fail "blk$b: $lk still held after ${GLX_DP4_LOCK_WAIT_S:-90} s"
    sleep 1
  done
  [ $(( $(date +%s) - t0 )) -gt 0 ] && log "blk$b: serve_window.lock free after $(( $(date +%s) - t0 )) s"
done
for b in $BLOCKS; do
  env GLX_IMAGE_ROOT=$(img $b 2) GLX_PIN=$(img $b 3) GLX_PD_ROLE=H GLX_PD_TAG=${TAG}dp4b$b TT_METAL_OPERATION_TIMEOUT_SECONDS=0 \
    VLLM_PORT=$((8200 + 10 * b)) setsid bash "$K/glx_block_run.sh" $b "$GO" "$PD/glx_pd_role.sh" > "$OUT/H_blk$b.log" 2>&1 < /dev/null &
done
log "trays launched (form $FORM = $TAG) on blk $(echo $BLOCKS | tr ' ' '/'), ports $(for b in $BLOCKS; do printf '%s ' $((8100 + b)); done); logs $OUT/H_blk*.log"
# (3) all healthy (the socket binds only after the strict startup self-test passed); a refusal or a dead tray fails the launch
t0=$(date +%s)
while :; do
  up=0
  for b in $BLOCKS; do
    hit=$(grep -a -m1 'refusing' "$OUT/H_blk$b.log" 2>/dev/null) && fail "blk$b refused: $hit"
    r=$(tray_run $b)
    [ -n "$r" ] && grep -qE 'server rc=|server exited' "$r/window.log" 2>/dev/null && fail "blk$b's server ended before healthy ($(grep -E 'server rc=|server exited' "$r/window.log" | tail -1 | cut -c1-200))"
    curl -sf -m 5 -o /dev/null http://127.0.0.1:$((8100 + b))/health && up=$((up + 1))
  done
  [ $up = $(echo $BLOCKS | wc -w) ] && break
  [ $(( $(date +%s) - t0 )) -ge ${GLX_DP4_UP_S:-2400} ] && fail "$up of $(echo $BLOCKS | wc -w) trays healthy after ${GLX_DP4_UP_S:-2400} s"
  sleep 15
done
log "all trays healthy after $(( $(date +%s) - t0 )) s"
for b in $BLOCKS; do
  r=$(tray_run $b)
  sv=$(ls -t $R_GLM53/work-s9/runs/glxdp4_${TAG}dp4b${b}_selftest_*.json 2>/dev/null | head -1)
  log "blk$b: run $(basename "${r:-?}"); strict self-test verdict ${sv:-none found}: $([ -n "$sv" ] && /usr/bin/python3 -I -c 'import json, sys; j = json.load(open(sys.argv[1])); print("pass" if j.get("pass") else "FAIL", {k: j.get(k) for k in ("prefill_top1", "prefill_top5", "decode_top1", "decode_top5") if k in j})' "$sv")"
done
[ "${GLX_DP4_ACCEPT:-1}" = 1 ] || { log "trays up; acceptance skipped (GLX_DP4_ACCEPT=0)"; exit 0; }
# (4) the per-block acceptance, all blocks at once (each block's ladder hits only its own server): cold 128 / 2,048 / 32,000 /
# 131,072 (+ one 2,048 cold prompt, selftest_loop.py's prefill loop)
for b in $BLOCKS; do
  r=$(tray_run $b)
  $PY -I "$K/glxk4d7lg/selftest_loop.py" --port $((8100 + b)) --run-glob "$r" --out "$OUT/accept_blk$b" --plugin-src $R_GLM53/srv/s9hp5bc/src \
    --prefill 1 --prefill-tokens 2048 --prefill-max-tokens 4 --prefill-timeout 300 --ladder 128,2048,32000,131072 \
    > "$OUT/accept_blk$b.log" 2>&1 &
done
wait
rc=0
for b in $BLOCKS; do
  s=$(grep -E 'LADDER|NO STALL|STALL|ERROR' "$OUT/accept_blk$b.log" | tr '\n' ' ' | cut -c1-600)
  grep -q 'NO STALL' "$OUT/accept_blk$b.log" || rc=5
  log "blk$b acceptance: $s"
done
log "acceptance rc=$rc ($OUT); the trays stay up (glx_dp4_stop.sh closes them in order)"
exit $rc
