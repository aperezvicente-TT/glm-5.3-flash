# Sourced by the host mocks (mock_workload.sh, mock_image_serve.sh): yield to a Stage 9 device run that opens while a
# mock runs (the persistent launcher takes GO lines at any time; review rounds 16-19). A mock holds no device
# (no-tt-devices + the mock cluster), so its yield is bounded: SIGTERM to the mock API server (a background job starts
# with SIGINT ignored; vLLM handles SIGTERM, but once serving that is a drain of up to --shutdown-timeout), then after
# GLM53_MOCK_YIELD_GRACE_S (30 s) SIGKILL of the server's whole process tree (API server, EngineCore, workers),
# then a record of any process of that tree still alive -> $RUN/yielded.txt.
#   mock_yield_watch <server pid> <port>   (background watcher; needs RUN, BUSY9)
#   mock_yielded                           (wait for the watcher's verdict line, print the record)

mock_tree() {  # the pid and every descendant (collected before any kill: children re-parent once the parent dies)
  local p; echo "$1"; for p in $(pgrep -P "$1" 2>/dev/null); do mock_tree "$p"; done
}

mock_yield_watch() {
  local srv=$1 port=$2 grace=${GLM53_MOCK_YIELD_GRACE_S:-30} tree left
  ( while kill -0 "$srv" 2>/dev/null; do
      if [ -f "$BUSY9" ] && [ "${GLM53_MOCK_DURING_RUN:-0}" != 1 ]; then
        tree=$(mock_tree "$srv" | tr '\n' ' ')
        echo "$(date -u +%FT%TZ) a Stage 9 device run opened ($BUSY9): SIGTERM to the mock server (tree: $tree)" \
          >> $RUN/yielded.txt
        kill -TERM "$srv" 2>/dev/null
        pkill -TERM -f "vllm.entrypoints.openai.api_server .*--port $port( |$)" 2>/dev/null
        for _ in $(seq "$grace"); do kill -0 "$srv" 2>/dev/null || break; sleep 1; done
        tree="$tree $(mock_tree "$srv" 2>/dev/null | tr '\n' ' ')"  # + anything forked during the grace
        left=$(for p in $tree; do kill -0 "$p" 2>/dev/null && echo "$p"; done | sort -u | tr '\n' ' ')
        if [ -n "$left" ]; then
          echo "$(date -u +%FT%TZ) after ${grace} s still alive: [$left] -> SIGKILL (mock processes, no device)" \
            >> $RUN/yielded.txt
          kill -KILL $left 2>/dev/null; sleep 2
        fi
        left=$(for p in $tree; do kill -0 "$p" 2>/dev/null && echo "$p"; done | sort -u | tr '\n' ' ')
        echo "$(date -u +%FT%TZ) mock server gone; leftover processes of its tree: [${left:-none}]" >> $RUN/yielded.txt
        break
      fi
      sleep 5
    done ) &
}

mock_yielded() {
  for _ in $(seq $(( ${GLM53_MOCK_YIELD_GRACE_S:-30} + 30 ))); do
    grep -q "mock server gone" $RUN/yielded.txt && break; sleep 1
  done
  cat $RUN/yielded.txt
}
