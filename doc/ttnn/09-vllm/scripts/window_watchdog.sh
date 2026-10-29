#!/usr/bin/env bash
# The window watchdog (program 2026-10-28T04:04:42Z RULE: every host-8chip window runs with TT_METAL_OPERATION_TIMEOUT_SECONDS=0, so
# a hung engine stays hung with its cores live; the stall is DECLARED here, with NO kill and NO reset). Started by serve_window.sh
# beside the server. A stall =
#   - before the 'server healthy' line: server.log AND events.jsonl (if any) idle >= GLM53_WATCHDOG_IDLE_S (600);
#   - after it: vLLM running + waiting > 0 while events.jsonl is idle >= GLM53_WATCHDOG_IDLE_S (an idle resident is not a stall).
# On a stall: <run>/HOLD_FOR_READ (serve_window.sh's stop path then waits instead of signalling the server), a copy of
# work-s9/generated/inspector into <run>/inspector_at_stall, one DEVICE line to the coordinator file; then it exits. 14b reads the
# live chips (inspector RPC / halted dispatch); the coordinator ends the engine by pid, resets, and removes HOLD_FOR_READ.
# usage: window_watchdog.sh <run dir> <api server pid> <port>
set -u
RUN=${1:?run dir}; SPID=${2:?api server pid}; PORT=${3:-8000}
W=/opt/glm53/work-s9; CQ=$W/COORDINATOR_REQUEST.txt; IDLE=${GLM53_WATCHDOG_IDLE_S:-600}
log() { echo "$(date -u +%FT%TZ) [watchdog] $*"; }
mt() { stat -c %Y "$1" 2>/dev/null || echo 0; }
q() { curl -s -m 5 http://127.0.0.1:$PORT/metrics | grep -E '^vllm:num_requests_(running|waiting)\{' | sed -E 's/\{.*\}//' | awk '{s += $2} END {print s + 0}'; }
log "start: run $(basename $RUN), api server $SPID, idle limit $IDLE s, TT_METAL_OPERATION_TIMEOUT_SECONDS=${TT_METAL_OPERATION_TIMEOUT_SECONDS:-unset}"
while kill -0 $SPID 2>/dev/null && [ ! -e $RUN/STOP ]; do
  now=$(date +%s); sl=$(mt $RUN/server.log); ev=$(mt $RUN/events.jsonl)
  if grep -q 'server healthy' $RUN/window.log 2>/dev/null; then
    n=$(q); [ "${n:-0}" -gt 0 ] && [ $ev -gt 0 ] && [ $((now - ev)) -ge $IDLE ] && { phase="serving: $n request(s) in flight, events.jsonl idle $((now - ev)) s"; break; }
  else
    last=$sl; [ $ev -gt $last ] && last=$ev
    [ $last -gt 0 ] && [ $((now - last)) -ge $IDLE ] && { phase="startup (no healthy line): server.log idle $((now - sl)) s, events.jsonl $([ $ev -gt 0 ] && echo "idle $((now - ev)) s" || echo absent)"; break; }
  fi
  sleep 15
done
[ -n "${phase:-}" ] || { log "ended without a stall ($([ -e $RUN/STOP ] && echo STOP || echo 'server gone'))"; exit 0; }
EC=$(pgrep -P $SPID -f 'EngineCore' | head -1); EC=${EC:-$(ps -eo pid,args | awk '/VLLM::EngineCore/ && !/awk/ {print $1}' | head -1)}
echo "stall declared $(date -u +%FT%TZ) by window_watchdog.sh ($phase); EngineCore ${EC:-?} left hung; the stop path waits for the coordinator" > $RUN/HOLD_FOR_READ
[ -d $W/generated/inspector ] && cp -a $W/generated/inspector $RUN/inspector_at_stall 2>/dev/null
log "STALL: $phase; HOLD_FOR_READ set; inspector copied"
echo "$(date -u +%FT%TZ) stage9: DEVICE — $(basename $RUN) STALL declared by the window watchdog ($phase). TT_METAL_OPERATION_TIMEOUT_SECONDS=${TT_METAL_OPERATION_TIMEOUT_SECONDS:-?}: the engine is LEFT HUNG (EngineCore ${EC:-?}, api server $SPID), no kill, no reset; inspector logs copied to $RUN/inspector_at_stall; $RUN/HOLD_FOR_READ holds the stop path. Next: 14b's live read, then the coordinator ends the engine by pid, resets and removes HOLD_FOR_READ." >> $CQ
