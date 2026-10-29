#!/usr/bin/env bash
# Stage 9 mesh window: ONE long-lived vLLM server on all 8 chips (tt-lock group `all`), a workload against it, an
# orderly shutdown. Launch detached (setsid nohup ... &): the harness must never own a device process.
#   usage: GLM53_RUN=<name> [GLM53_WINDOW_MIN=55] [GLM53_TRACKING=1] [GLM53_MAX_SEQS=32] [GLM53_MAX_MODEL_LEN=2051]
#          [GLM53_MAX_BATCHED_TOKENS=2048] [GLM53_EARLIEST_START=<UTC>] [GLM53_LATEST_START=<UTC>] [GLM53_HARD_STOP=<UTC>]
#          [GLM53_EXTRA_SERVER_ARGS=...] [GLM53_KEEP_UP=1] serve_window.sh <workload.sh>
# DEVICE RULES (2026-10-18): MESH_REQUEST.stage9 announces the window; the server starts only when no other stage's
# MESH_BUSY.* exists and no lock group is HELD; MESH_BUSY.stage9 is held for the window; both are removed on exit.
# COORDINATOR_NOTE.txt gates the launch: the latest Stage 9 GO line ("<UTC>Z [COORDINATOR — | COORDINATOR-SCHEDULER — ]9 mesh: GO") or "9 HOLD" line decides;
# launched on a GO (GLM53_GO_LINE), a later "9 HOLD" cancels and a RESET in progress pauses; work-s9/NEEDS_RESET.all
# refuses before the mesh is taken. The device process tree only ever gets SIGINT (to the vLLM API server, which stops
# the engine core, which closes the mesh) and only once the server is up (before that SIGINT is ignored: a deadline
# that passes during startup stops the server the moment it is ready, without the workload). At the window deadline
# the workload's host-only client group gets SIGTERM, the server its SIGINT 15 s later, and clients still alive after
# 30 s more are SIGKILLed (no device process is in that group). Kernel-log anomalies of the window go to the run log
# and work-s9/kernel_anomalies.log. A server that ends without its orderly-shutdown markers, or with a device-failure
# signature in its log, marks the box dirty (dirty_check.sh: work-s9/NEEDS_RESET.all with the failed chips).
set -u
source /opt/glm53/kit/doc/ttnn/09-vllm/scripts/env.sh
# program 2026-10-28T04:04:42Z RULE: every window runs with TT_METAL_OPERATION_TIMEOUT_SECONDS=0 (env.sh's 300 made a hung engine
# exit and wipe the device state before the read: k4d8s 03:57Z); the stall is declared by window_watchdog.sh, with no kill and no
# reset. GLM53_OP_TIMEOUT_S overrides.
export TT_METAL_OPERATION_TIMEOUT_SECONDS=${GLM53_OP_TIMEOUT_S:-0}
WORKLOAD=${1:?workload script}
RUN=$GLM53_WORK/runs/${GLM53_RUN:?run name}
mkdir -p $RUN
cd $GLM53_WORK  # neutral cwd: tt-metal resolves kernel sources cwd-first
# host-parametric (program 2026-10-27T00:08:40Z, the Galaxy blocks): the lock tool and the group come from the host profile
# (replicate/host_glx.sh on the Galaxy: tt-lock-galaxy + blk<N>); host-8chip keeps the defaults
TL=${GLM53_TT_LOCK:-/opt/glm53/kit/.agents/scripts/tt-lock}; LG=${GLM53_LOCK_GROUP:-all}
# GLM53_MESH_TAG (the Galaxy: blk<N>, set by replicate/glx_block_run.sh): the busy / request files are the block's own
# (MESH_BUSY.blk<N>.stage9), so rows on other blocks are not peers; host-8chip leaves it unset (unchanged names)
MT=${GLM53_MESH_TAG:+$GLM53_MESH_TAG.}
# program [session 10] 2026-10-28T11:02:13Z (a stale work-s9-blk2/NEEDS_RESET.all refused k3srvo after the base tool's
# reset): the dirty marker clears ITSELF -- moved to NEEDS_RESET.all.cleared.<UTC> -- when this host's reset journal (TT_LOCK_DIR's,
# ~/.tt-locks', the Galaxy base tool's /opt/glm53/locks) shows a done rc=0 reset of its chips (host-8chip) / of its block's
# group or 'all' (Galaxy: GLM53_MESH_TAG) after it, or when the box rebooted since (boot id) with its pid gone; never a hand-clear
autoclear_marker() {
  local m=$GLM53_WORK/NEEDS_RESET.all j dc=/opt/glm53/kit/doc/ttnn/09-vllm/scripts/dirty_check.sh
  [ -f $m ] || return 0
  for j in ${GLM53_RESET_JOURNAL:-} ${TT_LOCK_DIR:+$TT_LOCK_DIR/reset.journal} $HOME/.tt-locks/reset.journal /opt/glm53/locks/reset.journal /dev/null; do
    [ "$j" = /dev/null ] || [ -s "$j" ] || continue
    if GLM53_DIRTY_GROUP=${GLM53_MESH_TAG:-} GLM53_DIRTY_RUNS=$GLM53_WORK/runs bash $dc cleared $m $j; then
      local c=$m.cleared.$(date -u +%Y%m%dT%H%M%SZ); mv $m $c
      log "dirty marker cleared ($( [ "$j" = /dev/null ] && echo "boot id changed, its pid gone" || echo "a done reset after it in $j")): $(head -1 $c | cut -c1-200) -> $c"
      return 0
    fi
  done
}
BUSY=/opt/glm53/MESH_BUSY.${MT}stage9
REQ=/opt/glm53/MESH_REQUEST.${MT}stage9
NOTE=$GLM53_WORK/COORDINATOR_NOTE.txt
WIN=${GLM53_WINDOW_MIN:-55}
PORT=${GLM53_PORT:-8000}
LOG=$RUN/window.log
log() { echo "[s9 $(date -u +%FT%TZ)] $*" | tee -a $LOG >&2; }
held() { $TL status 2>/dev/null | awk '$1 ~ /^(all|glm|card[0-3]|card23|chip[0-7])$/ && $3 == "HELD" {print $1}' | tr '\n' ' '; }
peers() { ls /opt/glm53/MESH_BUSY.${MT}* 2>/dev/null | grep -v "MESH_BUSY.${MT}stage9$" | tr '\n' ' '; }
# coordinator lines after our GO (GLM53_GO_LINE = the exact GO line's "<UTC>Z [COORDINATOR(-SCHEDULER) — ]9 mesh: GO" prefix) that match $1
post_go() { local n; n=$(grep -n -F -- "$GLM53_GO_LINE" $NOTE | tail -1 | cut -d: -f1); [ -n "$n" ] && tail -n +$((n + 1)) $NOTE | grep -E -- "$1"; }

# ONE Stage 9 window at a time (review round 21): a second instance refuses before it touches MESH_REQUEST / MESH_BUSY
# (its EXIT trap would remove the first one's); the API server inherits the lock fd (vLLM spawns its EngineCore,
# which closes inherited fds), so the lock is held while serve_window or its API server lives
exec 8> $GLM53_WORK/serve_window.lock
flock -n 8 || { log "refusing: another Stage 9 serve_window (or its server) holds $GLM53_WORK/serve_window.lock"
                 exit 75; }
# Checked at LAUNCH, before the mesh is requested (review round 14 P2-1: install-time checks missed the resident target
# and every target installed before them; a hand swap of resident_target.sh bypassed them):
# (1) the target's own plugin carries every listed plugin fix (plugin_fixes_check.py), or the target names the
#     coordinator ruling that waives its gap (GLM53_PLUGIN_FIXES_WAIVER, e.g. "e79ee93 waived for cf3h-c, ruled
#     2026-10-24T05:57:05Z"); the waiver goes into the launch record (server_cmd.txt keeps GLM53_*)
# (2) the d0 capture rule (DECISIONS 2026-10-24T06:01:58Z): a target installed with GLM53_CAPTURE_GATE=PENDING (a
#     captured-path change whose d0 pair test WITH trace capture has not passed) never launches
REFUSE() { log "refusing: $1"; echo "$(date -u +%FT%TZ) stage9: ${GLM53_TARGET:-?} launch REFUSED before the mesh request: $1" >> $GLM53_WORK/COORDINATOR_REQUEST.txt; exit 75; }
if [ -n "${GLM53_PLUGIN_SRC:-}" ]; then
  pfc=$(/opt/glm53/kit/.agents/scripts/no-tt-devices /usr/bin/python3 -I \
        /opt/glm53/kit/doc/ttnn/09-vllm/scripts/plugin_fixes_check.py ${GLM53_PLUGIN_SRC%/src} 2>&1)
  pfc_rc=$?
  miss=$(echo "$pfc" | grep -E 'MISSING|UNKNOWN|REVERTED' | sed 's/^ *//' | tr '\n' ';')
  if [ $pfc_rc -ne 0 ] && [ -z "${GLM53_PLUGIN_FIXES_WAIVER:-}" ]; then
    REFUSE "plugin $(basename ${GLM53_PLUGIN_SRC%/src}) misses a listed fix (${miss:-rc $pfc_rc}) and the target names no ruled waiver (GLM53_PLUGIN_FIXES_WAIVER)"
  fi
  # the waiver names the ruled shas before its ':' ("e79ee93,1306ec1: <ruling>"; review round 15 O1): any other gap, a
  # REVERTED / UNKNOWN result or a checker that produced no MISSING line is refused
  if [ $pfc_rc -ne 0 ]; then
    ruled=$(echo "${GLM53_PLUGIN_FIXES_WAIVER%%:*}" | tr ', ' '\n\n' | grep -E '^[0-9a-f]{7,}$' | sort -u)
    gaps=$(echo "$pfc" | awk '$1 == "MISSING" {print $2}' | sort -u)
    other=$(echo "$pfc" | grep -cE '^ *(REVERTED|UNKNOWN)')
    outside=$(comm -23 <(echo "$gaps") <(echo "$ruled") | grep . | tr '\n' ' ')
    [ -z "$gaps" ] || [ "$other" != 0 ] || [ -n "$outside" ] || [ -z "$ruled" ] && \
      REFUSE "plugin $(basename ${GLM53_PLUGIN_SRC%/src}): the gap (${miss:-rc $pfc_rc}) is not inside the ruled waiver's shas (${ruled:-none named})"
  fi
  if [ $pfc_rc -ne 0 ]; then log "plugin fixes: ${miss} WAIVED: $GLM53_PLUGIN_FIXES_WAIVER"; else log "plugin fixes: $(basename ${GLM53_PLUGIN_SRC%/src}) carries every listed fix"; fi
fi
# a SET gate must start with PASS (review round 16: PENDING, FAIL or any other text refuses); unset = a target from before
# the rule (2026-10-24T06:01:58Z)
[ -n "${GLM53_CAPTURE_GATE:-}" ] && [[ "$GLM53_CAPTURE_GATE" != PASS* ]] && REFUSE "GLM53_CAPTURE_GATE=${GLM53_CAPTURE_GATE%% *}: the d0 pair test with trace capture has not passed for this target (DECISIONS 2026-10-24T06:01:58Z)"
# hugetlbfs pages (program 2026-10-27T19:02:14Z): a dead server's PinnedPagePool file (16 GiB) left the next window
# ENOMEM in initialize_kv_cache (twindf 18:15Z). Sweep dead-pid pool files when the box is not dirty (an orderly close or a
# reset since: no stale page-DMA kernel), then refuse below GLM53_HUGEPAGES_MIN free 1-GiB pages (24 = the image pull's 8 +
# the 16-GiB pool), naming the pool files
# host-8chip only by default (one mesh = every chip, so an orderly close or a reset quiets every page-DMA kernel); another
# host (the shared Galaxy: other blocks' chips and pool files) runs it only with GLM53_HUGEPAGES_MIN set
PSW=/opt/glm53/kit/doc/ttnn/09-vllm/scripts/stage9_ops/pool_sweep.sh
PSW_ON=0; { [ "$(hostname -s)" = host-8chip ] || [ -n "${GLM53_HUGEPAGES_MIN:-}" ] || [ "${GLM53_POOL_SWEEP:-0}" = 1 ]; } && PSW_ON=1  # GLM53_POOL_SWEEP=1: the Galaxy roles (program 2026-10-28T00:05:00Z, pair 31: three stale 16-GiB pool files of dead engines)
[ $PSW_ON = 1 ] && [ ! -f $GLM53_WORK/NEEDS_RESET.all ] && log "$(bash $PSW sweep)"
# the shared pool's size (program 2026-10-27T21:32:00Z, PD pair 30: D at 1 + 15 GiB could not hold the 1M pages; pair 29:
# a tier key alone died 'the shared pool is 1 + 15 GiB, asked 1 + 48'): one pool = the page-DMA ring
# (GLM53_LMCACHE_PAGE_DMA_GIB) + the LMCache CPU tier (GLM53_LMCACHE_POOL_CPU_GB, which the role env job sets equal to the
# tier GLM53_LMCACHE_CPU_GB: pd_envjobs.py b6af9b23). Refused here, on every host (a read-only check), when the two keys
# differ or the free 1-GiB pages cannot hold the image pull (8) + the pool; GLM53_HUGEPAGES_MIN overrides the count
pool=0
if [ "${GLM53_LMCACHE:-0}" = 1 ] && [ "${GLM53_LMCACHE_POOL_CPU_GB:-0}" -gt 0 ] 2>/dev/null; then
  [ -n "${GLM53_LMCACHE_CPU_GB:-}" ] && [ "${GLM53_LMCACHE_CPU_GB}" != "${GLM53_LMCACHE_POOL_CPU_GB}" ] \
    && REFUSE "GLM53_LMCACHE_CPU_GB ${GLM53_LMCACHE_CPU_GB} != GLM53_LMCACHE_POOL_CPU_GB ${GLM53_LMCACHE_POOL_CPU_GB}: LMCache would ask a tier the shared pool does not hold (the init dies 'the shared pool is ... asked ...')"
  pool=$(( ${GLM53_LMCACHE_PAGE_DMA_GIB:-1} + GLM53_LMCACHE_POOL_CPU_GB ))
fi
need=${GLM53_HUGEPAGES_MIN:-$((8 + pool))}
hp=$(bash $PSW check $need) || REFUSE "the shared pool cannot be allocated: this launch needs $need free 1-GiB pages (the image pull 8 + the pool $pool = ring ${GLM53_LMCACHE_PAGE_DMA_GIB:-1} + tier ${GLM53_LMCACHE_POOL_CPU_GB:-0} GiB); $hp"
log "$hp (this launch needs $need: image pull 8 + pool $pool)"
echo "$(date -u +%FT%TZ) stage9 requests the mesh for ~${WIN} min: vLLM server window ${GLM53_RUN} (one process, tt-lock all)${GLM53_EARLIEST_START:+; announced slot from $GLM53_EARLIEST_START}${GLM53_HARD_STOP:+ to $GLM53_HARD_STOP}" > $REQ
trap 'rm -f $REQ' EXIT
waited=0
while :; do
  # Stage 6's acceptance phases have priority (DEVICE RULES): yield while its request is pending
  # GLM53_AFTER_PID: the coordinator queued us right after that process (Stage 6 run 3): once it is gone, a new
  # Stage 6 request no longer has priority over this launch
  # a HOLD for Stage 9 stands until a later GO headline: the latest "9 HOLD" / "9 mesh: GO" line decides
  h=$(held); p=$(peers); hold=$(grep -sE "^[^ ]+Z ((COORDINATOR(-SCHEDULER)? — )?9 mesh: GO( |$)|.*\b9 HOLD\b)" $NOTE | tail -1 | grep -E "\b9 HOLD\b" | cut -c1-140)
  s6=$(ls /opt/glm53/MESH_REQUEST.stage6 2>/dev/null)
  if [ -n "${GLM53_GO_LINE:-}" ]; then  # launched on a GO: a later HOLD cancels it, a reset in progress pauses it
    ph=$(post_go "9 HOLD" | tail -1 | cut -c1-160)
    [ -n "$ph" ] && { log "coordinator HOLD after our GO ($GLM53_GO_LINE): $ph -- not starting"; exit 75; }
    case "$(post_go "[Rr][Ee][Ss][Ee][Tt]" | tail -1 | tr '[:upper:]' '[:lower:]')" in *"in progress"*) s6="${s6:+$s6; }reset in progress";; esac
    [ $waited -ge ${GLM53_POST_GO_WAIT_S:-600} ] && { log "our GO is ${waited}s old without the mesh; not starting (ask again)"; exit 75; }
    if [ -n "${GLM53_GO_MAX_AGE_S:-}" ]; then  # the scheduler forfeits a turn without MESH_BUSY 90 s after its GO
      go_age=$(( $(date -u +%s) - $(date -u -d "${GLM53_GO_LINE%%Z *}Z" +%s 2>/dev/null || echo 0) ))
      [ $go_age -le $GLM53_GO_MAX_AGE_S ] || { log "our GO is ${go_age}s old (> ${GLM53_GO_MAX_AGE_S}s: past the scheduler's 90-s turn); not starting"; exit 75; }
    fi
  fi
  if [ -n "${GLM53_AFTER_PID:-}" ]; then
    if kill -0 $GLM53_AFTER_PID 2>/dev/null; then s6="waiting for pid $GLM53_AFTER_PID"; else s6=""; fi
  fi
  # GLM53_AFTER_BUSY=<stage>: start only after that stage's MESH_BUSY file has appeared and gone again (its slot ran)
  if [ -n "${GLM53_AFTER_BUSY:-}" ]; then
    [ -f /opt/glm53/MESH_BUSY.$GLM53_AFTER_BUSY ] && seen_after=1
    [ "${seen_after:-0}" = 1 ] && [ ! -f /opt/glm53/MESH_BUSY.$GLM53_AFTER_BUSY ] || s6="${s6:+$s6; }waiting for the $GLM53_AFTER_BUSY slot"
    [ -f /opt/glm53/MESH_REQUEST.$GLM53_AFTER_BUSY ] && s6="${s6:+$s6; }$GLM53_AFTER_BUSY requested"
  fi
  # GLM53_EARLIEST_START=<UTC>: the announced slot opens then (no earlier start even if the mesh is free)
  if [ -n "${GLM53_EARLIEST_START:-}" ] && [ "$(date -u +%s)" -lt "$(date -u -d "$GLM53_EARLIEST_START" +%s)" ]; then
    s6="${s6:+$s6; }slot opens $GLM53_EARLIEST_START"
  fi
  # GLM53_LATEST_START=<UTC>: the window must not start later (it has to end inside the announced maintenance window)
  if [ -n "${GLM53_LATEST_START:-}" ] && [ "$(date -u +%s)" -gt "$(date -u -d "$GLM53_LATEST_START" +%s)" ]; then
    log "latest start $GLM53_LATEST_START passed without the mesh; not starting (next maintenance window)"; exit 75
  fi
  autoclear_marker
  [ -f $GLM53_WORK/NEEDS_RESET.all ] && { log "refusing: box marked dirty ($(cat $GLM53_WORK/NEEDS_RESET.all)) -- it clears itself on a done reset of its chips / block after it"; exit 4; }
  [ -z "$h" ] && [ -z "$p" ] && [ -z "$hold" ] && [ -z "$s6" ] && break
  [ $waited -ge ${GLM53_LOCK_WAIT_S:-21600} ] && { log "gave up after ${waited}s (held: $h; busy: $p; $hold)"; exit 75; }
  [ $((waited % 300)) = 0 ] && log "waiting for the mesh (held: $h; busy: $p; ${hold:-no HOLD}; ${s6:-no stage6 request})"
  sleep 20; waited=$((waited + 20))
done
echo "$(date -u +%FT%TZ) stage9 pid $$: vLLM server window ${GLM53_RUN} (~${WIN} min)" > $BUSY
trap 'rm -f $BUSY $REQ' EXIT
log "mesh taken after ${waited}s; coordinator note (latest): $(tail -1 $NOTE 2>/dev/null | cut -c1-400)"
# this host's chip table at the mesh take (dev, BDF, root port, tt-lock chip): the dirty check names chips with this
# copy, so a reset / re-enumeration before the scan cannot shift the numbers (review round 22)
/opt/glm53/kit/doc/ttnn/09-vllm/scripts/dirty_check.sh table > $RUN/tt_chips.txt
log "chip table saved: $(grep -c . $RUN/tt_chips.txt) rows (8 expected) -> $RUN/tt_chips.txt"
# host load at the mesh take: every mock process on the box (any stage's; a Stage 9 mock's yield watcher SIGTERMs its
# mock server within ~5 s of MESH_BUSY.stage9), recorded with the run
mocks=$(ps -eo pid,etime,args | grep -iE "[m]ock" | grep -v shell-snapshots | cut -c1-140 | tr "\n" ";")
[ -n "$mocks" ] && log "host: mock processes running at the mesh take: $mocks"
autoclear_marker
[ -f $GLM53_WORK/NEEDS_RESET.all ] && { log "refusing: box marked dirty ($(cat $GLM53_WORK/NEEDS_RESET.all))"; exit 4; }

T0=$(date '+%Y-%m-%d %H:%M:%S'); START=$(date +%s)
SEQS=${GLM53_MAX_SEQS:-32}; MML=${GLM53_MAX_MODEL_LEN:-2051}
# vLLM automatic prefix caching: ON only with the model's prefix store (GLM53_PREFIX_CACHE=1, SRV prefixcache; the
# adapter refuses APC without it), else off as before
APC=--no-enable-prefix-caching; [ "${GLM53_PREFIX_CACHE:-0}" = 1 ] && APC=--enable-prefix-caching
TT_CFG=$(printf '{"sample_on_device_mode": "all", "trace_mode": "all", "enable_model_warmup": true, "trace_region_size": %d, "prefill_chunk_alignment": 128%s}' ${GLM53_TRACE_REGION:-800000000} "${GLM53_TT_EXTRA_CFG:+, $GLM53_TT_EXTRA_CFG}")
SERVER_ARGS=(--model zai-org/GLM-5.3-Flash --block-size 64 --max-num-seqs $SEQS --max-model-len $MML
  --max-num-batched-tokens ${GLM53_MAX_BATCHED_TOKENS:-2048} $APC --override-generation-config '{"top_k": 32}'
  --max-logprobs -1 --tool-call-parser glm47 --enable-auto-tool-choice --host 127.0.0.1 --port $PORT
  --shutdown-timeout 900 --additional-config "{\"tt\": $TT_CFG}" ${GLM53_EXTRA_SERVER_ARGS:-})
# MTP speculative decoding (Stage 17; launch_resident.sh checks the pair against the rehearsal). The adapter's
# verify / propose split is synchronous (no supports_async_spec_decode): the plugin refuses async scheduling with it,
# unless GLM53_SPEC_ASYNC=1 (program 2026-10-23T10:33:28Z / 11:00:30Z: the adapter declares
# supports_async_spec_decode, so the launch keeps vLLM's async scheduling and the plain multi-user steps their overlap)
[ -n "${GLM53_SPEC_CONFIG:-}" ] && SERVER_ARGS+=(--speculative-config "$GLM53_SPEC_CONFIG") \
  && { [ "${GLM53_SPEC_ASYNC:-0}" = 1 ] || SERVER_ARGS+=(--no-async-scheduling); }
# LMCache (operator 2026-10-22 18:30Z, Stage 12's join: plugin srv/stack-s12lmc + LMCache-tt j11j12), opt-in with
# GLM53_LMCACHE=1 (an env-job overlay key, so the image rehearsal binds it): the LMCacheConnectorV1 + the TT paged
# connector, CPU tier GLM53_LMCACHE_CPU_GB, 128-token chunks (= the prefill alignment / prefix boundary), a fresh KDA
# checkpoint dir per launch, LMCache-tt's fix worktree ahead of the venv's editable install, a short TMPDIR (LMCache's
# ZMQ ipc socket paths are capped at 107 chars) -- the settings of 12's rehearsed mock (mock_lmcache_int27.sh)
if [ "${GLM53_LMCACHE:-0}" = 1 ]; then
  export PYTHONPATH=${GLM53_LMCACHE_TREE:-/opt/glm53/LMCache-tt-j12}:$PYTHONPATH
  export TMPDIR=${GLM53_SHORT_TMP:-/opt/glm53/tz}; mkdir -p $TMPDIR
  # GLM53_KDA_CKPT_ROOT (an env-job overlay key; program 2026-10-23T02:09:37Z: the checkpoints live on the NFS data
  # tree, not the root disk): <root>/<run>; unset: the run dir
  # GLM53_KDA_CKPT_DIR (12's restart reuse, program 2026-10-24T05:53:59Z; doc/ttnn/12-fast-start/remote_tier/README.md):
  # a STABLE checkpoint dir per target + trees (KEY), used as is, so a restart of the same target finds its checkpoints
  KDA=${GLM53_KDA_CKPT_DIR:-${GLM53_KDA_CKPT_ROOT:+$GLM53_KDA_CKPT_ROOT/$(basename $RUN)}}; KDA=${KDA:-$RUN/kda_ckpt}; mkdir -p $KDA
  # GLM53_LMCACHE_DISK_GB (an env-job overlay key; program 2026-10-23T02:58:17Z: local_disk 40 GB on root): a per-run
  # store under GLM53_LMCACHE_DISK_DIR (default /opt/glm53/lmcache_disk), removed when the window ends
  LMC_DISK=""; LMC_DISK_DIR=""
  # program [session 10] 2026-10-28T23:29:44Z (2): work-s9/LMCACHE_DISK_CAP.<tag> (one number, GB) caps the per-run local tier for that
  # target below its env job's value (host-8chip local disk before the 1M rungs; an operational knob: no mock / levers key reads it)
  CAPF_=/opt/glm53/work-s9/LMCACHE_DISK_CAP.${GLM53_TARGET#int27}
  if [ -s "$CAPF_" ] && [ "${GLM53_LMCACHE_DISK_GB:-0}" != 0 ] && [ "$(head -1 $CAPF_)" -lt "${GLM53_LMCACHE_DISK_GB}" ] 2>/dev/null; then
    log "LMCache local tier capped: GLM53_LMCACHE_DISK_GB ${GLM53_LMCACHE_DISK_GB} -> $(head -1 $CAPF_) ($CAPF_)"; GLM53_LMCACHE_DISK_GB=$(head -1 $CAPF_)
  fi
  if [ "${GLM53_LMCACHE_DISK_GB:-0}" != 0 ]; then
    LMC_DISK_DIR=${GLM53_LMCACHE_DISK_DIR:-/opt/glm53/lmcache_disk}/$(basename $RUN); mkdir -p $LMC_DISK_DIR
    LMC_DISK="local_disk: \"file://$LMC_DISK_DIR/\""$'\n'"max_local_disk_size: ${GLM53_LMCACHE_DISK_GB}"
  fi
  # GLM53_LMCACHE_REMOTE_URL (an env-job overlay key; 12's 16 x 1M tiers, lmcache_tiers_16x1M.yaml): the cold level,
  # '<run>' replaced by the run name; it has no cap, so the run dir is removed after the window (by hand)
  LMC_REMOTE=""
  [ -n "${GLM53_LMCACHE_REMOTE_URL:-}" ] && LMC_REMOTE="remote_url: \"${GLM53_LMCACHE_REMOTE_URL//<run>/$(basename $RUN)}\""$'\n'"remote_serde: naive"
  # a STABLE remote dir (no '<run>': 12's restart reuse) is never removed, and LMCache's fs remote never evicts: prune it
  # oldest-first to GLM53_LMCACHE_REMOTE_MAX_GB (default 64) before the server starts, so no lookup races the prune
  # (12's snippet, remote_tier/README.md)
  if [ -n "${GLM53_LMCACHE_REMOTE_URL:-}" ] && [[ "$GLM53_LMCACHE_REMOTE_URL" != *"<run>"* ]]; then
    RDIR=${GLM53_LMCACHE_REMOTE_URL#fs://localhost:0}; mkdir -p "$RDIR"
    CAP=$(( ${GLM53_LMCACHE_REMOTE_MAX_GB:-64} * 1073741824 ))
    find "$RDIR" -type f -printf '%T@\t%s\t%p\0' | sort -z -rn \
      | awk -v RS='\0' -v ORS='\0' -F'\t' -v cap=$CAP '{t+=$2} t>cap {print $3}' | xargs -0 -r rm -f
    log "stable remote dir $RDIR: $(timeout 30 du -sh "$RDIR" 2>/dev/null | cut -f1) after the prune to ${GLM53_LMCACHE_REMOTE_MAX_GB:-64} GB"
  fi
  cat > $RUN/lmcache_tt.yaml <<EOF
chunk_size: ${GLM53_LMCACHE_CHUNK:-128}
local_cpu: true
max_local_cpu_size: ${GLM53_LMCACHE_CPU_GB:-8}
local_cpu_use_hugepages: false
$LMC_DISK
$LMC_REMOTE
use_layerwise: false
save_decode_cache: false
save_unfull_chunk: true
enable_p2p: false
lmcache_rpc_port: glm53s9_${GLM53_RUN}
extra_config:
  gpu_connector: vllm_tt_plugin.lmcache.connector:VLLMPagedMemTTConnector
  tt_tokens_per_page: 64
  save_only_first_rank: false
  use_odirect: false
$([ -n "${GLM53_LMCACHE_EXTERNAL_CPU_BUFFER:-}" ] && echo "  external_cpu_buffer: $GLM53_LMCACHE_EXTERNAL_CPU_BUFFER")
$([ -n "${GLM53_LMCACHE_KEY_NAMESPACE:-}" ] && echo "  key_namespace: \"$GLM53_LMCACHE_KEY_NAMESPACE\"")
$(if [ -f "${GLM53_LMCACHE_EXTRA_YAML:-}" ]; then sed 's/^/  /' "$GLM53_LMCACHE_EXTRA_YAML"; elif [ -n "${GLM53_LMCACHE_EXTRA_YAML:-}" ]; then printf '%s\n' "$GLM53_LMCACHE_EXTRA_YAML"; fi)
EOF
  # GLM53_LMCACHE_KEY_NAMESPACE (cf3h-k3, program 16:12:35Z B2 / LMCache fork ef4689db): appended to every LMCache key
  # (quoted: "#tt-tile-v2" is a YAML comment unquoted); GLM53_KV_TILE_OBJECTS=1 refuses keys without it
  # GLM53_LMCACHE_EXTERNAL_CPU_BUFFER=<name> (lever (c) step 2, program 2026-10-26T03:59:21Z; LMCache fork 357a8f98):
  # the CPU tier lives in the buffer the plugin registered under <name> (its pinned page pool); unset = LMCache's arena
  # GLM53_LMCACHE_EXTRA_YAML (the Galaxy PD kit, program 2026-10-27T08:02:51Z; replicate/pd/): extra_config lines, e.g.
  # 12's Mooncake client keys (doc/ttnn/12-fast-start/mooncake/served_config.md) -- a FILE of unindented 'key: value' lines
  # (pd_envjobs.py's int27<tag>pd_mc_extra.yaml; indented here), else pre-indented TEXT (program 2026-10-27T13:47:37Z:
  # the kit shipped both forms; a box whose hook read a file got none of the keys -> Mooncake setup rc=-600)
  export LMCACHE_CONFIG_FILE=$RUN/lmcache_tt.yaml
  # kv_load_failure_policy pinned to fail (2026-10-23): a request whose counted hit holds a block LMCache did not
  # load is failed by vLLM; 'recompute' would restart it mid-prompt, where no KDA state exists
  SERVER_ARGS+=(--enable-prompt-tokens-details --kv-transfer-config
    "{\"kv_connector\": \"LMCacheConnectorV1\", \"kv_role\": \"kv_both\", \"kv_load_failure_policy\": \"fail\", \"kv_connector_extra_config\": {\"tt_kda_checkpoint_dir\": \"$KDA\"}}")
fi
# the served chat template (program 2026-10-23T15:02:43Z, review round 5 P2-4 stop/sequence): the model's own
# template with enable_thinking=false rendering '<|assistant|><think></think>', every other render byte-identical
# (scripts/chat_template_thinking_switch.jinja; GLM53_CHAT_TEMPLATE= (empty) serves the model's own). The render check
# is the gate: on a failure the flag is not passed and the launch says so.
CT=${GLM53_CHAT_TEMPLATE-$GLM53_WORK/chat_template_thinking_switch.jinja}
if [ -n "$CT" ]; then
  if HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 timeout 120 /opt/glm53/kit/.agents/scripts/no-tt-devices \
       python -I /opt/glm53/kit/doc/ttnn/09-vllm/scripts/check_chat_template.py ${GLM53_TOKENIZER_DIR:-${HF_SNAPSHOT:-zai-org/GLM-5.3-Flash}} "$CT" > $RUN/chat_template_check.json 2>&1; then
    SERVER_ARGS+=(--chat-template "$CT"); log "chat template: $CT ($(tail -1 $RUN/chat_template_check.json))"
  else
    log "chat template check FAILED ($(tail -1 $RUN/chat_template_check.json | cut -c1-200)): serving the model's own template"
  fi
fi
# the server's interpreter by path (2026-10-27 01:36Z, the Galaxy: a PATH lookup fell through to /usr/bin/python)
PYBIN=python; [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ] && PYBIN=$VIRTUAL_ENV/bin/python
TRK=()
[ "${GLM53_TRACKING:-0}" = 1 ] && TRK=(TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0)
# torch intra-op threads of the server (program 2026-10-23T21:27:40Z (5), the cf3g host profile): unset, every small
# host torch op of the per-step path forks across all cores of the taskset (one 8-chip page-table conversion: 129.6 ms at
# 32 threads vs 6.6 ms at 8); GLM53_OMP_THREADS overrides
TRK+=(OMP_NUM_THREADS=${GLM53_OMP_THREADS:-8})
TRK+=(TT_METAL_OPERATION_TIMEOUT_SECONDS=$TT_METAL_OPERATION_TIMEOUT_SECONDS)  # explicit on the server command (the 04:04:42Z rule)
# the Galaxy block's UMD ranks (program RULING 2026-10-28T01:08:26Z): replicate/glx_block_run.sh translates the lock group's
# /dev ids to UMD's BDF-sorted ranks and checks the inverse; given after the lock tool's own TT_VISIBLE_DEVICES export, it wins.
# Unset on host-8chip / host-p300 (there the tt-lock ids ARE the UMD ids).
if [ -n "${GLX_TT_VISIBLE_DEVICES:-}" ]; then
  TRK+=(TT_VISIBLE_DEVICES=$GLX_TT_VISIBLE_DEVICES)
  log "TT_VISIBLE_DEVICES=$GLX_TT_VISIBLE_DEVICES (the UMD ranks of the lock group's /dev ids ${GLX_DEV_IDS:-?}, glx_block_run.sh)"
fi
# the inspector RPC server's address (program 2026-10-28T02:03:11Z: bisect arm c's server bound the default 127.0.0.1:50051,
# held by another engine on the box, so its stall triage was blind; the arm script's per-block export never reached this env):
# with TT_METAL_INSPECTOR_RPC=1 the address is passed on the server command (server_cmd.txt shows it); on the Galaxy
# (GLX_TT_VISIBLE_DEVICES set) an unset address = the block's own 127.0.0.1:<50160 + N> (blkN). The triage reads
# <run>/inspector_rpc_address.txt: tools/triage/<script>.py --inspector-rpc-host=<host> --inspector-rpc-port=<port>
if [ "${TT_METAL_INSPECTOR_RPC:-0}" = 1 ]; then
  IRA=${TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS:-}
  if [ -z "$IRA" ] && [ -n "${GLX_TT_VISIBLE_DEVICES:-}" ]; then
    case "${GLM53_MESH_TAG:-}" in blk[0-9]|blk[0-9][0-9]) IRA=127.0.0.1:$((50160 + ${GLM53_MESH_TAG#blk}));; esac
  fi
  if [ -n "$IRA" ]; then
    TRK+=(TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS=$IRA); echo "$IRA" > $RUN/inspector_rpc_address.txt
    log "inspector RPC server address $IRA (triage: --inspector-rpc-host=${IRA%:*} --inspector-rpc-port=${IRA##*:})"
  else
    log "inspector RPC on at the default address 127.0.0.1:50051 (no TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS)"
  fi
fi
export GLM53_VLLM_EVENTS=$RUN/events.jsonl
# startup timeline (the adapter writes it at model ready; Stage 12's fast-start timeline next to it) and a fresh
# Prometheus multiprocess dir, so the API server's /metrics carries the EngineCore's glm53_startup_seconds{phase}
export GLM53_STARTUP_TIMELINE=$RUN/startup_timeline.json GLM53_TIMELINE_PATH=$RUN/fast_start_timeline.json
rm -rf $RUN/prometheus && mkdir -p $RUN/prometheus && export PROMETHEUS_MULTIPROC_DIR=$RUN/prometheus
{ echo "code=$GLM53_CODE_SHA tracking=${GLM53_TRACKING:-0} seqs=$SEQS max_model_len=$MML"; env | grep -E "^(TT_|VLLM_|MESH_|GLM53_|PMK_|HF_|PYTHONPATH|PYTHONHASHSEED|LD_LIBRARY_PATH|LMCACHE_|TMPDIR)" | sort; \
  printf '%q ' env "${TRK[@]}" taskset -c ${GLM53_SERVER_CPUS:-0-95} "$PYBIN" -m vllm.entrypoints.openai.api_server "${SERVER_ARGS[@]}"; echo; } > $RUN/server_cmd.txt
log "launch: $(tail -1 $RUN/server_cmd.txt)"
# the API server + EngineCore on cores 0-95 (program 2026-10-23T11:36:16Z: host clients / suites / mocks use 128-255)
$TL run $LG -- env "${TRK[@]}" taskset -c ${GLM53_SERVER_CPUS:-0-95} bash -c 'echo $$ > "$0"; py=$1; shift; exec "$py" -m vllm.entrypoints.openai.api_server "$@"' \
  $RUN/server.pid "$PYBIN" "${SERVER_ARGS[@]}" > $RUN/server.log 2>&1 &
TLPID=$!
sleep 5
SPID=$(cat $RUN/server.pid 2>/dev/null)
log "tt-lock pid $TLPID, api server pid ${SPID:-?}"
# the window watchdog (program 2026-10-28T04:04:42Z): declares a startup / serving stall, holds the stop path, never signals
[ -n "${SPID:-}" ] && setsid nohup bash /opt/glm53/kit/doc/ttnn/09-vllm/scripts/window_watchdog.sh $RUN $SPID ${PORT:-8000} \
  > $RUN/watchdog.log 2>&1 < /dev/null &
stop_server() {
  # a declared stall (window_watchdog.sh: <run>/HOLD_FOR_READ) holds the stop: the hung engine stays for 14b's read until the
  # coordinator ends it by pid and removes the file (program 2026-10-28T04:04:42Z)
  if [ -e $RUN/HOLD_FOR_READ ]; then
    log "stop requested ($1) while HOLD_FOR_READ exists: waiting for the coordinator's read ($(head -1 $RUN/HOLD_FOR_READ | cut -c1-160))"
    while [ -e $RUN/HOLD_FOR_READ ] && kill -0 ${SPID:-0} 2>/dev/null; do sleep 10; done
  fi
  if [ -n "${SPID:-}" ] && kill -0 $SPID 2>/dev/null; then
    log "SIGINT -> api server $SPID ($1)"; kill -INT $SPID
  fi
}
deadline=$((START + WIN * 60))
# GLM53_HARD_STOP=<UTC>: the announced window end; the workload and the server stop there at the latest
if [ -n "${GLM53_HARD_STOP:-}" ]; then hs=$(date -u -d "$GLM53_HARD_STOP" +%s); [ $hs -lt $deadline ] && deadline=$hs; fi
[ "${GLM53_KEEP_UP:-0}" = 1 ] && deadline=$((START + 7 * 86400))
healthy=0; late=0
while kill -0 $TLPID 2>/dev/null; do
  curl -sf http://127.0.0.1:$PORT/health >/dev/null && { healthy=1; break; }
  # past the deadline while starting: no signal now (the server starts as a background job with SIGINT ignored, and
  # vLLM installs its SIGINT handler only once it serves; SIGTERM is never sent to a device process tree) -> stop it
  # the moment it is ready, without the workload
  if [ $late = 0 ] && [ $(date +%s) -ge $deadline ]; then
    log "window deadline before ready: stopping it as soon as it is up"; late=1
    echo "$(date -u +%FT%TZ) stage9: ${GLM53_RUN} still starting at its window end ($(( $(date +%s) - START )) s after launch): no signal while it starts (SIGINT is ignored until vLLM serves); it is stopped the moment it is ready, without the workload -- the mesh stays held until then" >> $GLM53_WORK/COORDINATOR_REQUEST.txt
  fi
  sleep 10
done
if [ $healthy = 1 ] && [ $late = 1 ]; then stop_server "window deadline passed during startup"; healthy=0; fi
UP=/opt/glm53/SERVER_UP
if [ $healthy = 1 ]; then
  log "server healthy after $(( $(date +%s) - START )) s"
  echo "$(( $(date +%s) - START ))" > $RUN/ready_s.txt
  /opt/glm53/kit/.agents/scripts/no-tt-devices /usr/bin/python3 -I -c '
import json, sys, time
p, start = sys.argv[1], int(sys.argv[2])
try: d = json.load(open(p))
except (OSError, ValueError): d = {}
d["launcher"] = {"launch_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start)), "health_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "launch_to_health_s": round(time.time() - start, 1)}
json.dump(d, open(p, "w"), indent=1)' $GLM53_STARTUP_TIMELINE $START
  log "startup timeline: $(tr -d '\n ' < $GLM53_STARTUP_TIMELINE | head -c 400)"
  if [ "${GLM53_KEEP_UP:-0}" = 1 ]; then  # SERVE-FIRST (program 2026-10-19 00:06Z): other stages are clients
    for _ in $(seq 60); do curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null && break; sleep 5; done
    { echo "$(date -u +%FT%TZ) UP host=127.0.0.1 (host-8chip) port=$PORT model=zai-org/GLM-5.3-Flash max_model_len=$MML" \
        "max_num_seqs=$SEQS max_logprobs=-1(unlimited) tracking=${GLM53_TRACKING:-0} start_utc=$(date -u -d @$START +%FT%TZ)" \
        "sha=$GLM53_CODE_SHA run=$RUN metrics=http://127.0.0.1:$PORT/metrics stop=touch $RUN/STOP"; } >> $UP
    rm -f /opt/glm53/SERVER_DOWN
    log "published $UP"
  fi
  export SERVER_URL=http://127.0.0.1:$PORT RUN GLM53_WINDOW_DEADLINE=$deadline  # clients plan around the window end
  setsid bash $WORKLOAD > $RUN/workload.log 2>&1 &
  WPID=$!  # its own session / process group: the deadline signals every client of it
  past=0
  while kill -0 $WPID 2>/dev/null; do
    if [ $(date +%s) -ge $deadline ]; then
      # a background job of a non-interactive bash ignores SIGINT, and so do the clients it starts: TERM the
      # host-only client group, then stop the server after a short grace whatever the clients do
      log "window deadline: SIGTERM -> workload process group $WPID; the server stops in 15 s"
      kill -TERM -- -$WPID 2>/dev/null; sleep 15; past=1; break
    fi
    kill -0 $TLPID 2>/dev/null || { log "server exited during the workload"; break; }
    # a planned close (coordinator PAUSE / STOP) while the workload still runs: end the host-only client group now --
    # the keep-up wait below then sees STOP and stops the server in order (2026-10-23 00:02Z: STOP sat unread 10 min
    # behind a running w17 because it was checked only after the workload)
    if [ -f $RUN/STOP ]; then
      log "STOP during the workload ($(head -c 200 $RUN/STOP)): SIGTERM -> workload process group $WPID"
      kill -TERM -- -$WPID 2>/dev/null; break
    fi
    sleep 10
  done
  [ $past = 1 ] && stop_server "window deadline"
  # host-only clients: a bounded wait, then KILL of their group (no device process is in it)
  for _ in $(seq 15); do kill -0 -- -$WPID 2>/dev/null || break; sleep 2; done  # the whole group, not just its leader
  kill -0 -- -$WPID 2>/dev/null && { log "workload group still alive after TERM: SIGKILL (host-only clients)"; kill -KILL -- -$WPID 2>/dev/null; }
  wait $WPID 2>/dev/null; log "workload rc=$?"
  if [ "${GLM53_KEEP_UP:-0}" = 1 ]; then
    log "keep-up mode: server stays up until $RUN/STOP exists"
    while kill -0 $TLPID 2>/dev/null && [ ! -f $RUN/STOP ]; do sleep 10; done
    [ -f $RUN/STOP ] && echo "$(date -u +%FT%TZ) DOWN (planned): $(cat $RUN/STOP) run=$RUN" > /opt/glm53/SERVER_DOWN
    kill -0 $TLPID 2>/dev/null || echo "$(date -u +%FT%TZ) DOWN (server exited on its own; see $RUN/server.log)" > /opt/glm53/SERVER_DOWN
  fi
  [ $past = 1 ] || stop_server "workload done / stop requested"
fi
for _ in $(seq 120); do kill -0 $TLPID 2>/dev/null || break; sleep 10; done
if kill -0 $TLPID 2>/dev/null; then log "server still alive 20 min after SIGINT: NOT killing it (unclean-death rule); operator needed"; fi
wait $TLPID; rc=$?
log "server rc=$rc after $(( $(date +%s) - START )) s"
[ -n "${LMC_DISK_DIR:-}" ] && rm -rf -- "$LMC_DISK_DIR" && log "LMCache disk store $LMC_DISK_DIR removed (per run)"
# the plugin (s9drain 277c08e) logs how the engine process ended after vLLM's process-manager join and when its drain
# budget aborted requests: surface both (exit status 7 = tt-umd's SIGBUS exit = a device reset under the live process)
grep -aE "\[shutdown\] TT(:| drain budget)" $RUN/server.log | tail -3 | sed "s/^/[engine] /" >> $LOG
holders=$(lsof /dev/tenstorrent/* 2>/dev/null | awk 'NR>1{print $1"/"$2}' | sort -u | tr '\n' ' ')
log "device node holders after exit: [${holders}]"
ps -eo pid,args | grep -E "EngineCore|vllm.entrypoints" | grep -v grep >> $LOG
# the WHOLE kernel log of the window -> $RUN/kernel_window.txt (no cap: DEVICE RULES record every Hardware Error line);
# the matching lines are counted, listed per device, and all appended to the run log and kernel_anomalies.log
timeout 20 journalctl -k --since "$T0" --no-pager -o short-iso 2>/dev/null > $RUN/kernel_window.txt
K=$(grep -iE "ghes|hardware error|aer|mce|machine check|tenstorrent|all-1s|all-ones|0xffffffff|Completion Timeout|NonFatalErr|RxErr|BadTLP|BadDLLP|ReplayTimer|CmpltTO|MalfTLP|UnsupReq|bnxt|infiniband|rdma|qp_async|async event" $RUN/kernel_window.txt | grep -v "Configured ACS")
# every other non-ACS line, counted by its source (e.g. a USB serial disconnect): listed in the summary, kept in the file
O=$(grep -v "Configured ACS" $RUN/kernel_window.txt | grep -viE "ghes|hardware error|aer|mce|machine check|tenstorrent|all-1s|all-ones|0xffffffff|Completion Timeout|NonFatalErr|RxErr|BadTLP|BadDLLP|ReplayTimer|CmpltTO|MalfTLP|UnsupReq|bnxt|infiniband|rdma|qp_async|async event" | grep -oE "kernel: [^:]+" | sort | uniq -c | tr -s ' ' | tr '\n' ';')
if [ -n "$K" ]; then
  log "KERNEL LOG during the window: $(echo "$K" | wc -l) matching lines ($(echo "$K" | grep -c 'Hardware Error') [Hardware Error], severities: $(echo "$K" | grep -oiE 'severity: [a-z]+|severity=[a-z]+( \([a-z -]+\))?|, type: [a-z]+' | sort | uniq -c | tr -s ' ' | tr '\n' ';')) on $(echo "$K" | grep -oE '(device_id: |pcieport |bnxt_en |tenstorrent )?0000:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9]' | grep -oE '0000:[0-9a-f:.]+' | sort -u | tr '\n' ' '); other lines: ${O:-none}; full: $RUN/kernel_window.txt"
  echo "$K" | tee -a $LOG >/dev/null; { echo "$(date -u +%FT%TZ) stage9 ${GLM53_RUN}"; echo "$K"; } >> $GLM53_WORK/kernel_anomalies.log
else log "kernel log clean for the window (other lines: ${O:-none}; $RUN/kernel_window.txt)"; fi
# the box is dirty (review rounds 19 / 20; dirty_check.sh) after a device-failure line in server.log even when UMD then
# closed cleanly (open / FW-init / firmware startup failure, device timeout or hang, NOC hang, all-1s read, DMA or
# training timeout, watcher stop), after a Completion Timeout / uncorrected AER on a program chip in this window's
# kernel log (collected above), or after no orderly mesh close once the server touched the devices:
# work-s9/NEEDS_RESET.all with the failed tt-lock chips; the launcher withdraws MESH_REQUEST.stage9 and declines GOs
# until tt-lock's reset.journal shows those chips reset after it (or a coordinator ruling, applied by hand)
dirty=$(/opt/glm53/kit/doc/ttnn/09-vllm/scripts/dirty_check.sh scan $RUN/server.log $RUN/kernel_window.txt)
if [ -n "$dirty" ]; then
  log "box dirty: $dirty -> $GLM53_WORK/NEEDS_RESET.all (check fuser /dev/tenstorrent/*)"
  echo "$(date -u +%FT%TZ) stage9 ${GLM53_RUN} rc=$rc: $dirty boot=$(cat /proc/sys/kernel/random/boot_id 2>/dev/null) pid=${SPID:-}" > $GLM53_WORK/NEEDS_RESET.all
  echo "$(date -u +%FT%TZ) stage9: DEVICE — ${GLM53_RUN} (rc=$rc): $dirty. Box marked dirty (work-s9/NEEDS_RESET.all); MESH_REQUEST.stage9 withdrawn until tt-lock's reset.journal shows those chips reset (or your ruling). Evidence: $RUN/server.log" >> $GLM53_WORK/COORDINATOR_REQUEST.txt
fi
# the dead server's pool file (program 2026-10-27T19:02:14Z): swept after an orderly close; a dirty box keeps it until
# the reset (a stale page-DMA kernel could still write those pages) and the next start sweeps it then
if [ $PSW_ON = 0 ]; then :
elif [ -z "$dirty" ]; then log "$(bash $PSW sweep)"
else log "pool files kept until the reset (dirty box): $(ls /dev/hugepages-1G 2>/dev/null | grep glm53_pagepool | tr '\n' ' ')"; fi
exit $rc
