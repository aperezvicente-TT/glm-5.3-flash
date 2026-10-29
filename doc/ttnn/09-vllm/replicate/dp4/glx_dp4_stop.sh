#!/usr/bin/env bash
# Stop the Galaxy DP4 trays IN ORDER (operator delivery, program 2026-10-28T12:20:55Z item (2)): one block at a time, blk0 ->
# blk3: a STOP file in the tray's run dir = serve_window.sh's orderly close (workload SIGTERM, server SIGINT, the device close,
# the holder check), then wait for that tray's 'device node holders after exit' line (or its server pid gone) before the next
# block. No signal from here, no reset. A tray that does not close within GLX_DP4_CLOSE_S (default 300 s) is reported and the
# script goes on to the next block (the coordinator decides on the stuck one).
# usage: glx_dp4_stop.sh [--since <UTC stamp>]   GLX_DP4_FORM=t6d|f2 (default t6d)   GLX_DP4_BLOCKS='0 1 2 3'
set -u
D4=$(dirname "$(readlink -f "$0")"); K=$(dirname "$D4")
. "$K/host_glx.sh"
FORM=${GLX_DP4_FORM:-t6d}
case $FORM in t6d) TAG=n29h7p2hk4d8t6d;; f2) TAG=n29h7p2hk4d8t5wp6f2;; t6di) TAG=n29h7p2hk4d8t6di;; f2i) TAG=n29h7p2hk4d8t5wp6f2i;; f2i3) TAG=n29h7p2hk4d8t5wp6f2i3;; f2i3b) TAG=n29h7p2hk4d8t5wp6f2i3b;; f2i3c) TAG=n29h7p2hk4d8t5wp6f2i3c;; f2i3cxb) TAG=n29h7p2hk4d8t5wp6f2i3cxb;; f2i3d) TAG=n29h7p2hk4d8t5wp6f2i3d;; f2i3d2) TAG=n29h7p2hk4d8t5wp6f2i3d2;; f2i3d3) TAG=n29h7p2hk4d8t5wp6f2i3d3;; smu2v4m2xbnp) TAG=n29h7p2hk4d8t5wsmu2v4m2xbnp;; f2i3d4) TAG=n29h7p2hk4d8t5wp6f2i3d4;; smu2v4m2xbv4) TAG=n29h7p2hk4d8t5wsmu2v4m2xbv4;; df2gv4c4xbv4) TAG=n29h7p2hk4d8t5wdf2gv4c4xbv4;; f2i3d5) TAG=n29h7p2hk4d8t5wp6f2i3d5;; smu2v4m2xbv5) TAG=n29h7p2hk4d8t5wsmu2v4m2xbv5;; f2i3d6) TAG=n29h7p2hk4d8t5wp6f2i3d6;; smu2v4m2xbv6) TAG=n29h7p2hk4d8t5wsmu2v4m2xbv6;; f2i3d7) TAG=n29h7p2hk4d8t5wp6f2i3d7;; smu2v4m2xbv7) TAG=n29h7p2hk4d8t5wsmu2v4m2xbv7;; smu2v4m2xb) TAG=n29h7p2hk4d8t5wsmu2v4m2xb;; smu2v4m2xbgc) TAG=n29h7p2hk4d8t5wsmu2v4m2xbgc;; df2gv4c4xb) TAG=n29h7p2hk4d8t5wdf2gv4c4xb;; df2g4v2) TAG=n29h7p2hdf2g4v2;; smu2g4n4) TAG=n29h7p2hsmu2g4n4;; t6d2) TAG=n29h7p2hk4d8t6d2;; smu2g) TAG=n29h7p2hsmu2g;; smu2g4) TAG=n29h7p2hsmu2g4;; df2g) TAG=n29h7p2hdf2g;; df2g4) TAG=n29h7p2hdf2g4;; *) echo "GLX_DP4_FORM=$FORM (t6d, f2, t6di, f2i, f2i3, f2i3b, smu2g, smu2g4, df2g or df2g4)"; exit 2;; esac
SINCE=0; [ "${1:-}" = --since ] && SINCE=${2:?stamp}
work() { [ "$1" = 0 ] && echo "$R_GLM53/work-s9" || echo "$R_GLM53/work-s9-blk$1"; }
say() { echo "$(date -u +%FT%TZ) glx_dp4_stop: $*"; }
rc=0
for b in ${GLX_DP4_BLOCKS:-0 1 2 3}; do
  r=$(ls -td $(work $b)/runs/resident_int27${TAG}dp4b${b}pdH_glxb${b}_* 2>/dev/null | grep -v '\.json$' | head -1)
  if [ -z "$r" ] || [[ "${r##*_}" < "$SINCE" ]]; then say "blk$b: no DP4 tray run"; continue; fi
  pid=$(cat "$r/server.pid" 2>/dev/null || echo 0)
  if grep -q 'device node holders after exit' "$r/window.log" 2>/dev/null; then say "blk$b: $(basename $r) already closed"; continue; fi
  [ -e "$r/STOP" ] || echo "glx_dp4_stop $(date -u +%FT%TZ): the DP4 fleet closes in order" > "$r/STOP"
  say "blk$b: STOP -> $(basename $r) (server pid $pid); waiting for its orderly close"
  t0=$(date +%s)
  until grep -q 'device node holders after exit' "$r/window.log" 2>/dev/null; do
    if [ $(( $(date +%s) - t0 )) -ge ${GLX_DP4_CLOSE_S:-300} ]; then say "blk$b: NOT closed after ${GLX_DP4_CLOSE_S:-300} s (server pid $pid alive: $(kill -0 $pid 2>/dev/null && echo yes || echo no)); the coordinator decides"; rc=3; break; fi
    sleep 3
  done
  grep -q 'device node holders after exit' "$r/window.log" 2>/dev/null \
    && say "blk$b: closed: $(grep -E 'server rc=|device node holders after exit|box dirty' "$r/window.log" | tail -3 | tr '\n' ' ' | cut -c1-300)"
done
exit $rc
