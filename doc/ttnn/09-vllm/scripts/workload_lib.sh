# Shared by the Stage 9 workloads (sourced). Needs SERVER_URL, RUN. Every python call runs under no-tt-devices: the
# clients and the readiness runner only talk HTTP (the server process owns the mesh).
S=/opt/glm53/kit/doc/ttnn/09-vllm/scripts
NTD=/opt/glm53/kit/.agents/scripts/no-tt-devices
# the client python: venv-vllm on host-8chip; a host without it (the Galaxy box, program 2026-10-27T11:28:36Z: the
# firstlight lone-user step died rc 127) uses venv-lmcache; GLM53_CLIENT_PY overrides
PY=${GLM53_CLIENT_PY:-/opt/glm53/venv-vllm/bin/python}; [ -x "$PY" ] || PY=/opt/glm53/venv-lmcache/bin/python
RT=/opt/glm53/kit/vendor/tenstorrent-skills/plugins/tt-model-bringup/runtime
TREE=${GLM53_S9_TREE:-/opt/glm53/tt-metal-s9}
SEQS=${GLM53_MAX_SEQS:-32}
# the runner writes <model dir>/readiness_vllm; mock dry runs point it at a scratch dir
RMD=${GLM53_RUNNER_MODEL_DIR:-$TREE/models/autoports/zai_org_glm_5_3_flash}
MML=${GLM53_MAX_MODEL_LEN:-2051}
export HF_HUB_CACHE=${HF_HUB_CACHE:-/opt/glm53/hf} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# the plugin tests (sampling suite, needles) come from the SERVED plugin's checkout when the target sets GLM53_PLUGIN_SRC
_PLUG=${GLM53_PLUGIN_SRC:+${GLM53_PLUGIN_SRC%/src}}
# (the served plugin wins over an inherited VLLM_TT_PLUGIN_ROOT: env.sh exports the old checkout, and the runner refuses a
# root that is not the imported plugin's)
export VLLM_TT_PLUGIN_ROOT=${_PLUG:-${VLLM_TT_PLUGIN_ROOT:-/opt/glm53/vllm-tt-plugin}} VLLM_NO_USAGE_STATS=1
export GLM53_WS=${GLM53_WS:-/opt/glm53/kit}
step() { echo "[workload $(date -u +%FT%TZ)] $*"; }
checks() { ( cd $RUN && taskset -c ${GLM53_CLIENT_CPUS:-128-159} $NTD $PY -I $S/serve_checks.py "$@" --url $SERVER_URL --out $RUN/checks.jsonl \
  --events $RUN/events.jsonl --max-model-len $MML ); echo "rc=$? checks $1"; }
runner() {  # the shared readiness runner against the live server; artifacts in <model dir>/readiness_vllm
  touch $RUN/.runner_start
  ( cd $TREE && PYTHONPATH=${GLM53_PLUGIN_SRC:+$GLM53_PLUGIN_SRC:}$TREE/ttnn:$TREE:$RT taskset -c ${GLM53_CLIENT_CPUS:-128-159} $NTD $PY -m readiness_check.run_vllm_server \
      --model-dir $RMD --hf-model zai-org/GLM-5.3-Flash --max-num-seqs $SEQS \
      --max-model-len $MML --server-url $SERVER_URL "$@" ); echo "rc=$? runner $*"; }
snap_readiness() {  # keep a per-run copy of only what the last runner call wrote (the model dir accumulates runs)
  mkdir -p $RUN/readiness_vllm/$1 && ( cd $RMD/readiness_vllm && find . -type f -newer $RUN/.runner_start \
    -exec cp -a --parents {} $RUN/readiness_vllm/$1/ \; ); }
needles() {  # the plugin's canonical chunked-prefill needle tests at budget $1 (= the server's --max-num-batched-tokens)
  ( cd $VLLM_TT_PLUGIN_ROOT && PYTHONPATH=${GLM53_PLUGIN_SRC:+$GLM53_PLUGIN_SRC:}$TREE/ttnn:$TREE:$RT taskset -c ${GLM53_CLIENT_CPUS:-128-159} $NTD $PY -m pytest \
      tests/tt/test_chunked_prefill.py -v -p no:cacheprovider --tt-server-url=$SERVER_URL --tt-model-name=zai-org/GLM-5.3-Flash \
      --tt-max-num-seqs=$SEQS --tt-chunked-prefill-budget=$1 ) > $RUN/needles_$1.log 2>&1; echo "rc=$? needles budget $1"; }
dups() { ( cd $RUN && taskset -c ${GLM53_CLIENT_CPUS:-128-159} $NTD $PY -I $S/probe_dups.py $RUN/checks.jsonl $SERVER_URL ); echo "rc=$? duplicates"; }
smoke() {  # the build's five-minute smoke (smoke.py): rows + verdict in checks.jsonl, verdict JSON in $RUN/smoke.json
  ( cd $RUN && taskset -c ${GLM53_CLIENT_CPUS:-128-159} $NTD $PY -I $S/smoke.py --url $SERVER_URL --out $RUN/checks.jsonl \
      --max-model-len $MML --events $RUN/events.jsonl ${GLM53_SMOKE_ARGS:-} ) > $RUN/smoke.log 2>&1
  local rc=$?; tail -1 $RUN/smoke.log > $RUN/smoke.json; echo "rc=$rc smoke $(head -c 400 $RUN/smoke.json)"; return $rc; }
