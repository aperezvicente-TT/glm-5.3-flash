#!/usr/bin/env bash
# Resident-image rehearsal (program 2026-10-21 00:28:13Z SPEED RULES: a Stage 9 mesh open restores the served
# layout's TP DRAM image, no tensor loader): the REAL vLLM API server + TT plugin + Stage 9 adapter on a MOCK 2x4
# Blackhole cluster, run from the target's tree (GLM53_S9_TREE: s9i8 = 14b int8 + Stage 9, s9i9 = int9 + Stage 9, C++ =
# tt-metal-s12m5; s9i10 / s9i11 = int10 / int11 + Stage 9, C++ = tt-metal-s14bx) with the device run's venv + plugin
# (venv-lmcache, the LMCache join src), the FULL 45-layer stack with 14b's served knob env, its construction through
# the PRODUCTION restore of the pinned image (tests/vllm/stack_image_mock.py: restore_tp_backend ->
# tp_image.restore_backend; only the on-device hashes, the image-file prefetch and pulls, and the sidecar writes are
# stubbed). Pass = image_replay.json status "pass" with the stack env == 14b's env job + all 11 requests completed + an
# orderly close -> mock_verdict.json (+ work-s9/mock_verdict_<target>.json)
# with the code identity, the image's manifest sha256, the venv and plugin. JIT: the DEVICE run's cache dir (env.sh
# TT_METAL_CACHE), so a rehearsal pre-compiles the kernels a later mesh window would otherwise compile cold.
# usage: GLM53_RUN=<name> [GLM53_TARGET=int9 GLM53_S9_TREE=... GLM53_14B_ENV_JOB=...] mock_image_serve.sh
set -u
# no mock while a Stage 9 device run is open: host load during its measurements, and the mock compiles into the same
# JIT cache (review rounds 16 / 17); GLM53_MOCK_DURING_RUN=1 overrides. Checked at the start, and a watcher
# (mock_yield.sh) stops the mock the moment a run opens: SIGTERM, then SIGKILL of the mock's process tree after 30 s.
BUSY9=${GLM53_MOCK_BUSY_FILE:-/opt/glm53/MESH_BUSY.stage9}  # (override: tests of the yield)
if [ -f $BUSY9 ] && [ "${GLM53_MOCK_DURING_RUN:-0}" != 1 ]; then
  echo "a Stage 9 device run is open ($BUSY9): not starting a mock"; exit 75
fi
export GLM53_S9_TREE=${GLM53_S9_TREE:-/opt/glm53/tt-metal-s9i8}
# the device run's interpreter + plugin (launch_resident.sh): venv-lmcache, the LMCache join's src ahead of the venv's
export GLM53_VENV=${GLM53_VENV:-/opt/glm53/venv-lmcache}
export GLM53_PLUGIN_SRC=${GLM53_PLUGIN_SRC:-/opt/glm53/vllm-tt-plugin-lmcache-join/src}
source /opt/glm53/kit/doc/ttnn/09-vllm/scripts/env.sh
NTD=/opt/glm53/kit/.agents/scripts/no-tt-devices
RUN=$GLM53_WORK/mockruns/${GLM53_RUN:-image_mock}; mkdir -p $RUN
# GLM53_MOCK_PF_WARM=1: the TP prefill's warm-up forwards run on the mock for their allocations (int27 14:41Z: a warm-up
# transient that does not fit device DRAM must fail here, not at the device open)
# GLM53_MOCK_CLUSTER_DESC: another host's cluster descriptor (the Galaxy W5 mock, 2026-10-27: its blocks' harvest masks
# give a 12x10 worker grid vs host-8chip's 11x10, so the construction allocates another map); default host-8chip's
export TT_METAL_MOCK_CLUSTER_DESC_PATH=${GLM53_MOCK_CLUSTER_DESC:-$GLM53_WORK/mock/glm8.yaml} GLM53_MOCK=1 TT_METAL_SHM_TRACKING_DISABLED=1 GLM53_MOCK_PF_WARM=${GLM53_MOCK_PF_WARM:-1}
# the capture phase too (program 20:29:42Z, the int27m28c2 gap: it passed this mock and died on the device inside
# the prefill warm, 0.41 MB/bank short): GLM53_MOCK_CAPTURE=1 runs every captured forward once for its allocations, and
# the DRAM probe records the smallest largest-free-block per bank after any op of the prepare / capture phases
# (tt/mock_dram.py, trees with it); GLM53_MOCK_DRAM_BALLAST_MIB holds that much DRAM per bank through both phases.
# Calibration (image_mock_int27m28c2cal2_2112 = m28c2's config, ballast 0): the mock's headroom at the allocation that
# died on the device was 3.44 MiB/bank (capture; prepare 4.70), the device's -0.39 -> gap 3.83 MiB/bank (2.15 MB/bank
# of device-only allocations: int27c steady state 30.888 device vs 30.872 GiB/chip mock, same 543 programs). Ballast
# 4 MiB/bank: m28c2 then passes prepare and fails capture on the mock, as on the device
export GLM53_MOCK_CAPTURE=${GLM53_MOCK_CAPTURE:-1} GLM53_MOCK_DRAM_PROBE=${GLM53_MOCK_DRAM_PROBE:-1} GLM53_MOCK_DRAM_BALLAST_MIB=${GLM53_MOCK_DRAM_BALLAST_MIB:-4}
export TMPDIR=$GLM53_WORK/tmp TT_METAL_TRACE_ALLOC_TRACKING=1  # TT_METAL_CACHE: env.sh's (the device runs')
export GLM53_BACKEND_MODULE=models.autoports.zai_org_glm_5_3_flash.tests.vllm.stack_image_mock
# 14b's served construction (tests/s14b/m5common.serving): native 32-row step + a shared weight cache (DSA fabrics for
# t = 1..4 + the chunk), the served knob env of 14b's env job (below; default u01_env = int8's)
export GLM53_BACKEND_KW='{"native_t": true, "weight_cache": {}}'
export GLM53_STACK_KNOBS=s7a283 GLM53_TP_XSL=1 GLM53_TP_XSL_SLICED=1 GLM53_STACK_WIDTHS=none
export GLM53_STACK_PREFILL_TRACES=0 GLM53_STACK_SHARE_FABRIC=1 GLM53_STACK_KNOB_SET="s:GLM53_MHC_PRE_SPLIT=1"
export GLM53_STACK_NATIVE_WIDTHS=${GLM53_STACK_NATIVE_WIDTHS:-8,16}
export GLM53_TP_IMAGE_ROOT=${GLM53_TP_IMAGE_ROOT:-/opt/glm53/data/dram_images/tp14b}
# 14b's env job decides the stack / TP env (int21+ changed the knob set): its .done in q_mesh, or a staged .job before
# 14b's gate runs it (GLM53_14B_ENV_FILE); every GLM53_STACK_ / GLM53_TP_ key of it is exported over the defaults
# above, and so is every other key it sets but the launch-policy switches GLM53_IMAGE_BUILD / _REQUIRED and
# GLM53_WEIGHT_SOURCE (this mock restores the image by construction): int21+'s image restore mode, kernel knobs
# (e.g. GLM53_MLA_ATT_NKMAX). The values the mock ran go to the verdict's job_env (launch_resident.sh requires it to
# equal the env job's; a key the mock overrides below then shows as a mismatch, never as rehearsed); the env file and
# the keys go to run.txt
ENVF=${GLM53_14B_ENV_FILE:-/opt/glm53/work-s14b/q_mesh/${GLM53_14B_ENV_JOB:-u01_env}.done}
ENVX=$(/opt/glm53/kit/.agents/scripts/no-tt-devices /usr/bin/python3 -I -c "
import json, shlex, sys
env = json.load(open(sys.argv[1]))['args']['env']
skip = ('GLM53_IMAGE_BUILD', 'GLM53_IMAGE_REQUIRED', 'GLM53_WEIGHT_SOURCE')
img = sorted(k for k in env if not k.startswith(('GLM53_STACK_', 'GLM53_TP_')) and k not in skip)
for k, v in sorted(env.items()):
    if k.startswith(('GLM53_STACK_', 'GLM53_TP_')) or k in img:
        print('export %s=%s' % (k, shlex.quote(str(v))))
print('export GLM53_MOCK_JOB_KEYS=%s' % ','.join(img))" "$ENVF") || { echo "cannot read the env job $ENVF"; exit 3; }
eval "$ENVX"
# GLM53_MOCK_DESC_CACHE=verify (program 2026-10-27T02:14:40Z: the k4 image mock runs the descriptor cache in VERIFY
# mode -- every descriptor rebuilt and checked against the cache -- while the served env job says 1). launch_resident.sh
# takes a verdict that ran verify as rehearsing 1
[ -n "${GLM53_MOCK_DESC_CACHE:-}" ] && export GLM53_PMK_DESC_CACHE=$GLM53_MOCK_DESC_CACHE
# the image launch_resident.sh pins (it launches only on a passing verdict of this image, manifest and code)
export GLM53_IMAGE_PIN=${GLM53_IMAGE_PIN:-fde4697e51b2-tp-ds-v1-2x4-1530493535ed}
# GLM53_MOCK_IMAGE_ROOT (2026-10-27, the Galaxy W5 mock): another host's image, its host-side files copied under this
# root (no chip files: the mock's pulls are hash-gated). The env job's TP_IMAGE_DIR / ROOT are re-pointed; the launch
# compares the image by pin + manifest sha, its location is the host's (launch_resident.sh LOCATION)
[ -n "${GLM53_MOCK_IMAGE_ROOT:-}" ] && export GLM53_TP_IMAGE_ROOT=$GLM53_MOCK_IMAGE_ROOT GLM53_TP_IMAGE_DIR=$GLM53_MOCK_IMAGE_ROOT/$GLM53_IMAGE_PIN
unset GLM53_IMAGE_REQUIRED  # stack_image_mock.py sets the required restore loop for its own call only
export GLM53_IMAGE_RESIDENT=1  # the resident path (hash-gated pulls: nothing stale on the mock); never a full pull
export GLM53_IMAGE_FINE_BLOCK=0  # int9+ tp_image: manifest-size blocks only (the mock's stubbed hashes are those)
export GLM53_IMAGE_MOCK_RESULT=$RUN/image_replay.json; rm -f $GLM53_IMAGE_MOCK_RESULT
export GLM53_MOCK_DRAM_OUT=$RUN/mock_dram.jsonl; rm -f $GLM53_MOCK_DRAM_OUT
# the selftest REQUEST path runs in every mock (program 22:30:25Z: n29g's strict selftest 400'd every request --
# logprobs under MTP -- and the mock had it off): warn mode (the mock's logits are fake: accuracy is not scored here), a
# budget the mock's slow LMCache checkpoints cannot overrun, and any selftest REQUEST error fails the rehearsal below
if [ "${GLM53_SELFTEST:-off}" = off ]; then echo "mock: GLM53_SELFTEST=off is not allowed (the selftest request path must run): warn"; fi
export GLM53_SELFTEST=warn GLM53_SELFTEST_BUDGET_S=${GLM53_MOCK_SELFTEST_BUDGET_S:-3600}
export GLM53_SELFTEST_OUT=${GLM53_SELFTEST_OUT:-$RUN/selftest_verdict.json}; rm -f $GLM53_SELFTEST_OUT
# the code the rehearsal ran = what a device run executes: the git tree of tt/ + tests/common.py (stack_model imports
# it), "-dirty" when the checkout differs (launch_resident.sh computes the same and launches only on a passing
# verdict of THIS code: $GLM53_WORK/mock_verdict_<GLM53_TARGET>.json); doc / test-only commits keep it
MD=models/autoports/zai_org_glm_5_3_flash; T=$TT_METAL_HOME
CODE=tt:$(git -C $T rev-parse --short=12 HEAD:$MD/tt)+common:$(git -C $T rev-parse --short=8 HEAD:$MD/tests/common.py)$( [ -n "$(git -C $T status --porcelain -- $MD/tt $MD/tests/common.py)" ] && echo -dirty)
export GLM53_VLLM_EVENTS=$RUN/events.jsonl GLM53_STARTUP_TIMELINE=$RUN/startup_timeline.json
export PROMETHEUS_MULTIPROC_DIR=$RUN/prometheus; rm -rf $PROMETHEUS_MULTIPROC_DIR; mkdir -p $PROMETHEUS_MULTIPROC_DIR
PORT=${GLM53_PORT:-18052}; SEQS=${GLM53_MAX_SEQS:-32}
# GLM53_TT_CFG_EXTRA: more additional_config['tt'] keys (an env-job overlay key, e.g. '"prefill_chunk_run": 4';
# launch_resident.sh appends the same to the device launch) -- GLM53_EXTRA_SERVER_ARGS likewise (serve_window.sh)
WL1_DEFAULT='"worker_l1_size": 1264512, '; case "${GLM53_TT_CFG_EXTRA:-}" in *'"worker_l1_size"'*) WL1_DEFAULT='';; esac  # review round 30 O4: named once
TT_CFG='{"tt": {"sample_on_device_mode": "all", "trace_mode": "all", "enable_model_warmup": true, "trace_region_size": '"${GLM53_TRACE_REGION:-536870912}"', '"$WL1_DEFAULT"'"prefill_chunk_alignment": 128'"${GLM53_TT_CFG_EXTRA:+, $GLM53_TT_CFG_EXTRA}"'}}'
PLUGIN_REV=$(git -C ${GLM53_PLUGIN_SRC%/src} rev-parse --short=11 HEAD 2>/dev/null)$( [ -n "$(git -C ${GLM53_PLUGIN_SRC%/src} status --porcelain -- src 2>/dev/null)" ] && echo -dirty)
MANIFEST_SHA=$(sha256sum $GLM53_TP_IMAGE_ROOT/$GLM53_IMAGE_PIN/manifest.json | cut -c1-16)
echo "$(date -u +%FT%TZ) target ${GLM53_TARGET:-int8} tree $TT_METAL_HOME ($GLM53_CODE_SHA, code $CODE) image $GLM53_TP_IMAGE_ROOT/$GLM53_IMAGE_PIN (manifest sha256 $MANIFEST_SHA) venv $GLM53_VENV plugin $GLM53_PLUGIN_SRC ($PLUGIN_REV)" | tee $RUN/run.txt
echo "env job: $ENVF (imported: GLM53_STACK_ / GLM53_TP_ + ${GLM53_MOCK_JOB_KEYS:-no other keys})" \
  | tee -a $RUN/run.txt
cd $GLM53_WORK
# max_model_len: the env job's GLM53_MAX_MODEL_LEN when it sets one (a context-ladder rung, e.g. int26's 32,768; serve_window
# reads the same key), else 2,051
# LMCache (GLM53_LMCACHE=1 from the env job's overlay keys; the same settings serve_window.sh applies on the device)
LMC_ARGS=()
if [ "${GLM53_LMCACHE:-0}" = 1 ]; then
  export PYTHONPATH=${GLM53_LMCACHE_TREE:-/opt/glm53/LMCache-tt-j12}:$PYTHONPATH
  export TMPDIR=${GLM53_SHORT_TMP:-/opt/glm53/tz}; mkdir -p $TMPDIR
  KDA=$RUN/kda_ckpt; mkdir -p $KDA
  # GLM53_LMCACHE_DISK_GB: the mock's store stays in its own run dir, at most 2 GB (a mock's artefacts never leave it)
  LMC_DISK=""
  if [ "${GLM53_LMCACHE_DISK_GB:-0}" != 0 ]; then mkdir -p $RUN/lmcache_disk
    LMC_DISK="local_disk: \"file://$RUN/lmcache_disk/\""$'\n'"max_local_disk_size: $(( GLM53_LMCACHE_DISK_GB < 2 ? GLM53_LMCACHE_DISK_GB : 2 ))"; fi
  cat > $RUN/lmcache_tt.yaml <<EOF
chunk_size: ${GLM53_LMCACHE_CHUNK:-128}
local_cpu: true
max_local_cpu_size: $([ -n "${GLM53_LMCACHE_EXTERNAL_CPU_BUFFER:-}" ] && echo ${GLM53_LMCACHE_POOL_CPU_GB:-${GLM53_LMCACHE_CPU_GB:-8}} || echo $(( ${GLM53_LMCACHE_CPU_GB:-8} < 8 ? ${GLM53_LMCACHE_CPU_GB:-8} : 8 )))
local_cpu_use_hugepages: false
$LMC_DISK
use_layerwise: false
save_decode_cache: false
save_unfull_chunk: true
enable_p2p: false
lmcache_rpc_port: glm53s9m_${GLM53_RUN}
extra_config:
  gpu_connector: vllm_tt_plugin.lmcache.connector:VLLMPagedMemTTConnector
  tt_tokens_per_page: 64
  save_only_first_rank: false
  use_odirect: false
$([ -n "${GLM53_LMCACHE_EXTERNAL_CPU_BUFFER:-}" ] && echo "  external_cpu_buffer: $GLM53_LMCACHE_EXTERNAL_CPU_BUFFER")
$([ -n "${GLM53_LMCACHE_KEY_NAMESPACE:-}" ] && echo "  key_namespace: \"$GLM53_LMCACHE_KEY_NAMESPACE\"")
EOF
  # GLM53_LMCACHE_KEY_NAMESPACE (cf3h-k3, program 16:12:35Z B2 / LMCache fork ef4689db): appended to every LMCache key
  # (quoted: "#tt-tile-v2" is a YAML comment unquoted); GLM53_KV_TILE_OBJECTS=1 refuses keys without it
  # GLM53_LMCACHE_EXTERNAL_CPU_BUFFER=<name> (lever (c) step 2, program 2026-10-26T03:59:21Z; LMCache fork 357a8f98):
  # with it the mock's tier = GLM53_LMCACHE_POOL_CPU_GB, not the 8-GB mock cap: the model's shared pool is created at that
  # size and LMCache must ask for the same (page_pool.shared_pool refuses a mismatch: the j3cl2 mock 04:43Z)
  # the CPU tier lives in the buffer the plugin registered under <name> (its pinned page pool); unset = LMCache's arena
  export LMCACHE_CONFIG_FILE=$RUN/lmcache_tt.yaml
  # kv_load_failure_policy pinned to fail (2026-10-23): a request whose counted hit holds a block LMCache did not
  # load is failed by vLLM; 'recompute' would restart it mid-prompt, where no KDA state exists
  LMC_ARGS=(--enable-prompt-tokens-details --kv-transfer-config
    "{\"kv_connector\": \"LMCacheConnectorV1\", \"kv_role\": \"kv_both\", \"kv_load_failure_policy\": \"fail\", \"kv_connector_extra_config\": {\"tt_kda_checkpoint_dir\": \"$KDA\"}}")
  echo "LMCache on: $LMCACHE_CONFIG_FILE, LMCache-tt ${GLM53_LMCACHE_TREE:-/opt/glm53/LMCache-tt-j12} ($(git -C ${GLM53_LMCACHE_TREE:-/opt/glm53/LMCache-tt-j12} rev-parse --short=8 HEAD 2>/dev/null))" | tee -a $RUN/run.txt
fi
# host-memory guard (memory: a mock must never OOM a device process on this box). A mock's artefacts stay in its own
# dir (program 2026-10-23T01:09:36Z): the server's DSA slice sidecar dir is $RUN/dsa_slices, never the shared one
# (the harness keeps the job's value, so the verdict's job_env still equals the env job's)
# GLM53_SPEC_ASYNC=1 (program 2026-10-23T10:33:28Z): a speculative launch keeps async scheduling (as serve_window.sh)
NOASYNC=--no-async-scheduling; [ "${GLM53_SPEC_ASYNC:-0}" = 1 ] && NOASYNC=
# GLM53_MOCK_PYSPY=1 (program 2026-10-23T19:17:55Z: the host profile of the per-step paths): py-spy records the server
# tree as its parent (inside no-tt-devices' user namespace; ptrace_scope 1 forbids attaching to a foreign server), raw
# collapsed stacks of every process to $RUN/pyspy.txt, 100 Hz, non-blocking
PYSPY=(); [ "${GLM53_MOCK_PYSPY:-0}" = 1 ] && PYSPY=(/opt/glm53/home/.local/bin/py-spy record --subprocesses --threads --nonblocking -r 100 -f raw -o $RUN/pyspy.txt --)
systemd-run --user --scope -q -p MemoryMax=${GLM53_MOCK_MEM:-40G} \
  env ${GLM53_DSA_SLICE_CACHE:+GLM53_DSA_SLICE_CACHE=$RUN/dsa_slices} OMP_NUM_THREADS=${GLM53_OMP_THREADS:-8} taskset -c ${GLM53_CPUS:-224-255} nice -n 19 $NTD "${PYSPY[@]}" python -m vllm.entrypoints.openai.api_server --model zai-org/GLM-5.3-Flash \
  --block-size 64 --max-num-seqs $SEQS --max-model-len ${GLM53_MAX_MODEL_LEN:-2051} --max-num-batched-tokens ${GLM53_MAX_BATCHED_TOKENS:-2048} $( [ "${GLM53_PREFIX_CACHE:-0}" = 1 ] && echo --enable-prefix-caching || echo --no-enable-prefix-caching ) \
  --override-generation-config '{"top_k": 32}' --max-logprobs -1 --host 127.0.0.1 --port $PORT --shutdown-timeout 900 \
  --additional-config "$TT_CFG" ${GLM53_SPEC_CONFIG:+--speculative-config "$GLM53_SPEC_CONFIG" $NOASYNC} "${LMC_ARGS[@]}" ${GLM53_EXTRA_SERVER_ARGS:-} > $RUN/server.log 2>&1 &
SRV=$!

# yield to a Stage 9 device run that opens while the mock runs (mock_yield.sh: SIGTERM, then SIGKILL of the tree)
rm -f $RUN/yielded.txt
source /opt/glm53/kit/doc/ttnn/09-vllm/scripts/mock_yield.sh
mock_yield_watch $SRV $PORT
echo "server pid $SRV (log $RUN/server.log)"
# GLM53_MOCK_HEALTH_S (default 3600): the mock's health deadline; a 2,048-only build pays its strict self-test on the
# slow mock W7 path before the socket binds (cf3h-k2 2026-10-26: ~37 min) -- raise it rather than lose the run
# program 2026-10-26T15:59:55Z: 7,200 s is the default for a 2,048-only build (GLM53_PREFILL_BUCKETS=2048)
HS=${GLM53_MOCK_HEALTH_S:-$([ "${GLM53_PREFILL_BUCKETS:-}" = 2048 ] && echo 7200 || echo 3600)}
up=0; for _ in $(seq $HS); do curl -sf http://127.0.0.1:$PORT/health >/dev/null && { up=1; break; }; kill -0 $SRV 2>/dev/null || { [ -f $RUN/yielded.txt ] && { mock_yielded; echo "MOCK YIELDED to a device run"; exit 75; }; echo "server died"; tail -30 $RUN/server.log; cat $GLM53_IMAGE_MOCK_RESULT 2>/dev/null; exit 3; }; sleep 1; done
[ $up = 1 ] || { echo "not healthy after $HS s: SIGINT"; kill -INT $SRV; wait $SRV; exit 3; }
echo "healthy at $(date -u +%FT%TZ)"
c() { curl -s -m 1800 http://127.0.0.1:$PORT/v1/completions -H 'Content-Type: application/json' -d "$1"; echo; }
{ echo "1 user (width 8):"; c '{"model": "zai-org/GLM-5.3-Flash", "prompt": [1, 2, 3, 4, 5, 6, 7], "max_tokens": 4, "temperature": 0, "ignore_eos": true}'
  echo "10 concurrent (width 16):"
  P=(); for i in $(seq 10); do c "{\"model\": \"zai-org/GLM-5.3-Flash\", \"prompt\": [$i, 12, 13], \"max_tokens\": 3, \"temperature\": 0, \"ignore_eos\": true}" & P+=($!); done
  wait "${P[@]}"  # (a bare wait would also wait for the server)
} > $RUN/client.log 2>&1
# GLM53_MOCK_PROFILE_CLIENT="<cmd>" (the profile mock): run after the standard requests with URL=http://127.0.0.1:$PORT and
# RUN=$RUN exported; its output goes to $RUN/profile_client.log (not part of the pass verdict)
if [ -n "${GLM53_MOCK_PROFILE_CLIENT:-}" ]; then
  URL=http://127.0.0.1:$PORT RUN=$RUN bash -c "$GLM53_MOCK_PROFILE_CLIENT" > $RUN/profile_client.log 2>&1
  echo "profile client rc=$? ($RUN/profile_client.log)"
fi
# program 2026-10-23T07:13:20Z: every SPECULATIVE build passes a concurrent cell before install (n29df2e died when a
# second request arrived while a lone request speculated): (a) a lone request decoding (speculating) and a plain request
# arriving mid-generation; (b) 16 requests at once (mixed lengths: mixed widths). Every request must finish.
conc=skip
if [ -n "${GLM53_SPEC_CONFIG:-}" ] || [ "${GLM53_MOCK_CONCURRENT_CELL:-0}" = 1 ]; then
  { echo "(a) a speculating request + a plain arrival mid-generation:"
    c '{"model": "zai-org/GLM-5.3-Flash", "prompt": [21, 22, 23, 24, 25, 26, 27, 28], "max_tokens": 48, "temperature": 0, "ignore_eos": true}' & A=$!
    for _ in $(seq 240); do  # the first request is running (vLLM's gauge), then a few steps in
      n=$(curl -s -m 5 http://127.0.0.1:$PORT/metrics | awk '/^vllm:num_requests_running/ {s += $2} END {print int(s)}')
      [ "${n:-0}" -ge 1 ] && break; sleep 0.5
    done
    sleep 3
    c '{"model": "zai-org/GLM-5.3-Flash", "prompt": [31, 32, 33], "max_tokens": 8, "temperature": 0, "ignore_eos": true}'
    wait $A
    echo "(b) 16 at once:"
    P=(); for i in $(seq 16); do c "{\"model\": \"zai-org/GLM-5.3-Flash\", \"prompt\": [$((40 + i)), 12, 13, $i], \"max_tokens\": $((8 + i)), \"temperature\": 0, \"ignore_eos\": true}" & P+=($!); done
    wait "${P[@]}"
  } > $RUN/client_concurrent.log 2>&1
  conc=$(grep -o '"finish_reason":"length"' $RUN/client_concurrent.log | wc -l)
fi
echo "concurrent cell: $conc / 18"
# under GLM53_MOCK_PYSPY the server is py-spy's child: SIGINT the API server, py-spy writes its file when it exits
TGT=$SRV; [ ${#PYSPY[@]} -gt 0 ] && TGT=$(pgrep -P $SRV -f vllm.entrypoints | head -1); kill -INT ${TGT:-$SRV}
for _ in $(seq 600); do kill -0 $SRV 2>/dev/null || break; sleep 1; done
wait $SRV; echo "server rc=$?"
[ -f $RUN/yielded.txt ] && { mock_yielded; echo "MOCK YIELDED to a device run (not a pass)"; exit 75; }
closed=$(grep -c 'Closing devices in cluster completed' $RUN/server.log); aborts=$(grep -cE 'Process received signal|Assertion failed' $RUN/server.log)
echo "orderly close: $closed; aborts: $aborts"; cat $GLM53_IMAGE_MOCK_RESULT | head -c 600; echo
served=$(grep -o '"finish_reason":"length"' $RUN/client.log | wc -l); echo "requests completed: $served / 11"
envchk=$($NTD /usr/bin/python3 -I -c "
import json, sys
r = json.load(open(sys.argv[1]))
pre = ('GLM53_STACK_', 'GLM53_TP_')
LOC = ('GLM53_TP_IMAGE_DIR', 'GLM53_TP_IMAGE_ROOT')  # the image's location is the host's (launch_resident.sh LOCATION)
want = {k: v for k, v in json.load(open(sys.argv[2]))['args']['env'].items() if k.startswith(pre) and k not in LOC}
have = {k: v for k, v in (r.get('stack_env') or {}).items() if k not in LOC}
print('OK' if r.get('status') == 'pass' and have == want else 'replay %s, env diff %s' % (
    r.get('status'), sorted(set(have.items()) ^ set(want.items()))))" \
  $GLM53_IMAGE_MOCK_RESULT "$ENVF" 2>&1)
echo "image replay / env: $envchk"
# the DRAM probe's phases (tt/mock_dram.py): a phase that raised fails the rehearsal like any other startup error
dram=$($NTD /usr/bin/python3 -I -c "
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])] if __import__('os').path.exists(sys.argv[1]) else []
print(json.dumps(dict(phases=len(rows), ok=all(r.get('ok') for r in rows), ballast_mib_per_bank=max([r.get('ballast_mib_per_bank', 0) for r in rows] or [0]),
                      min_largest_free_mib_per_bank=min([r['min_largest_free_mib_per_bank'] for r in rows] or [None]) if rows else None,
                      min_at=[(r['label'], r['min_at_op']) for r in sorted(rows, key=lambda r: r['min_largest_free_per_bank'])[:1]])))" $GLM53_MOCK_DRAM_OUT 2>&1)
echo "mock DRAM probe: $dram"
# the selftest request path: every selftest request must have been answered (errors = 400s / refusals / exceptions)
stest=$($NTD /usr/bin/python3 -I -c "
import json, os, sys
p = sys.argv[1]
if not os.path.exists(p):
    print('no selftest verdict (the selftest did not run)'); sys.exit()
v = json.load(open(p))
errs = v.get('errors') or []
print('OK %s/%s answered' % (v.get('responses'), v.get('requests')) if not errs else
      'selftest request errors %d: %s' % (len(errs), str(errs[0])[:300]))" $GLM53_SELFTEST_OUT 2>&1)
echo "selftest request path: $stest"
[ "$closed" -ge 1 ] && [ "$aborts" = 0 ] && [ "$served" = 11 ] && [ "$envchk" = OK ] && [ "${stest%% *}" = OK ] \
  && { [ "$conc" = skip ] || [ "$conc" = 18 ]; } || { echo "IMAGE REHEARSAL FAILED"; exit 8; }
# the verdict (written only on a full pass): what launch_resident.sh checks before it takes the mesh
$NTD /usr/bin/python3 -I -c "
import json, os, sys, time
r = json.load(open(sys.argv[1]))
v = dict(pass_=True, target=os.environ.get('GLM53_TARGET', 'int8'), image=os.path.basename(r['image']), code=sys.argv[2], backend_kw=os.environ['GLM53_BACKEND_KW'],
         stack_env=r['stack_env'], uploads=r.get('uploads'), served=int(sys.argv[3]), closed=int(sys.argv[4]),
         job_env={k: os.environ.get(k) for k in os.environ.get('GLM53_MOCK_JOB_KEYS', '').split(',') if k},
         aborts=int(sys.argv[5]), run=sys.argv[6], utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
         manifest_sha=sys.argv[8], venv=os.environ['GLM53_VENV'], plugin_src=os.environ['GLM53_PLUGIN_SRC'], plugin_rev=sys.argv[9],
         spec={k: os.environ[k] for k in ('GLM53_SPEC_K', 'GLM53_SPEC_CONFIG') if os.environ.get(k)},
         mock_capture=os.environ.get('GLM53_MOCK_CAPTURE'), mock_dram=json.loads(sys.argv[10]) if sys.argv[10].startswith('{') else sys.argv[10],
         selftest_request_path=sys.argv[11], concurrent_cell=sys.argv[12])
v['pass'] = v.pop('pass_')
json.dump(v, open(sys.argv[7], 'w'), indent=1)" $GLM53_IMAGE_MOCK_RESULT "$CODE" $served $closed $aborts $RUN $RUN/mock_verdict.json "$MANIFEST_SHA" "$PLUGIN_REV" "$dram" "$stest" "$conc" \
  && cp $RUN/mock_verdict.json $GLM53_WORK/mock_verdict_${GLM53_TARGET:-int8}.json \
  && echo "verdict: $RUN/mock_verdict.json -> mock_verdict_${GLM53_TARGET:-int8}.json (code $CODE)"
echo MOCK_IMAGE_DONE
