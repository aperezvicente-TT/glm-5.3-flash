#!/usr/bin/env bash
# 12 dp4router v6f runner (Galaxy stand-alone form, program 2026-10-28T12:20:55Z): the front router for TP8 x DP4,
# one OpenAI-compatible endpoint in front of the four block servers; v5 refuses host-sampling parameters (400)
# by default (--host-sampling reject | strip | allow); v6 routes on each block's KV-budget headroom (plugin v3 metrics). CODE only: no device, no tt-lock; the block
# servers are not started, stopped or changed. Foreground (run it under the box's standing-server wrapper / nohup).
# usage: glx12_dp4router.sh <router_port> <log_dir> [<blocks csv, default http://127.0.0.1:8100..8103>] [extra args]
#        glx12_dp4router.sh --selftest            (fake blocks in-process; host-only)
# python: $PYTHON, else <extract root>/glm53/venv-lmcache/bin/python ($GLX_VENV/bin/python if set), else python3;
# it must import aiohttp (a vLLM 0.26 dependency). Exit 3 = refused (no aiohttp / the port is taken).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
PY=${PYTHON:-}
for c in "${GLX_VENV:+$GLX_VENV/bin/python}" "$ROOT/venv-lmcache/bin/python" "$(command -v python3)"; do
  [ -z "$PY" ] && [ -n "$c" ] && [ -x "$c" ] && "$c" -I -c "import aiohttp" 2>/dev/null && PY=$c
done
[ -n "$PY" ] || { echo "REFUSED: no python with aiohttp (set PYTHON=<the vLLM venv python>)"; exit 3; }
if [ "${1:-}" = "--selftest" ]; then
  exec "$PY" -I "$HERE/dp4router/test_dp4router.py"
fi
PORT=${1:?router port}
LOGD=${2:?log dir}
BLOCKS=${3:-http://127.0.0.1:8100,http://127.0.0.1:8101,http://127.0.0.1:8102,http://127.0.0.1:8103}
if [ $# -ge 3 ]; then shift 3; else shift $#; fi
mkdir -p "$LOGD" || { echo "REFUSED: cannot create $LOGD"; exit 3; }
# (v4: the bind test uses SO_REUSEADDR + SO_REUSEPORT like the router; with --replace <old url> the router itself takes
# the port over from the old instance, so no refusal here)
case " $* " in *" --replace "*) ;; *)
if "$PY" -I -c "import socket,sys; s=socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1); s.bind(('127.0.0.1', int(sys.argv[1])))" "$PORT" 2>/dev/null; then :; else
  echo "REFUSED: port $PORT is taken (pass --replace http://127.0.0.1:$PORT to take it over)" | tee -a "$LOGD/router.log"; exit 3
fi;;
esac
echo "# $(date -u +%FT%TZ) 12 dp4router v6f on $(hostname): port $PORT, blocks $BLOCKS, python $PY, extra: $*" | tee -a "$LOGD/router.log"
exec "$PY" -u -I "$HERE/dp4router/dp4router.py" --port "$PORT" --blocks "$BLOCKS" --log-dir "$LOGD" "$@" >>"$LOGD/router.log" 2>&1
