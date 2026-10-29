#!/usr/bin/env bash
# Resident-image serving run (program 2026-10-21 00:28:13Z SPEED RULES: a Stage 9 open restores the served layout's
# TP DRAM image -- no tensor loader) of a serving TARGET: a run tree (GLM53_S9_TREE: s9i8 = 14b int8 + Stage 9, s9i9 =
# int9 + Stage 9, ...; sparse /models/ with 14b's C++ symlinked), 14b's env job + gate rows, the pinned image and a
# passing mock rehearsal verdict of that exact code / image / env / venv / plugin. Defaults = int8 (resident1); the
# target files (scripts/targets/<int>_resident.sh, run by watch_sched_go.sh) export another int's values first.
# GLM53_WEIGHT_SOURCE=image + GLM53_IMAGE_REQUIRED=1 (no usable image -> the server raises before any tensor load),
# 14b's SERVED construction (native 32-row step + width buckets + the shared weight cache), connector OFF, the plugin
# from the LMCache join @ 2bd556d, the w12 workload.
# Start ONLY on the coordinator's GO: GLM53_GO_LINE="<the GO line's '<UTC>Z [COORDINATOR(-SCHEDULER) — ]9 mesh: GO'
# prefix>" launch_resident.sh. serve_window.sh re-checks the note for a later "9 HOLD" / a RESET in progress, writes
# MESH_REQUEST / MESH_BUSY.stage9, runs ONE vLLM process and closes clean.
# GLM53_14B_ENV_JOB / GLM53_14B_GATE: 14b's job server files (work-s14b/q_mesh) of the open that built / validated the
# image: its env job and the gate jobs whose verdicts (result.ok + bars) must hold before the mesh is taken.
set -u
W=/opt/glm53/work-s9
NOTE=$W/COORDINATOR_NOTE.txt
[ -n "${GLM53_GO_LINE:-}" ] || { echo "GLM53_GO_LINE unset"; exit 75; }
n=$(grep -n -F -- "$GLM53_GO_LINE" $NOTE | tail -1 | cut -d: -f1)
[ -n "$n" ] || { echo "GO line not in the note"; exit 75; }
tail -n +$((n + 1)) $NOTE | grep -qw "9 HOLD" && { echo "a later '9 HOLD' supersedes the GO"; exit 75; }
# a GO is good for its own turn only (the scheduler wants MESH_BUSY within 90 s): an older one was used or forfeited
export GLM53_GO_MAX_AGE_S=${GLM53_GO_MAX_AGE_S:-85}  # serve_window.sh re-checks it right before it takes the mesh
age=$(( $(date -u +%s) - $(date -u -d "${GLM53_GO_LINE%%Z *}Z" +%s 2>/dev/null || echo 0) ))
[ $age -le $GLM53_GO_MAX_AGE_S ] || { echo "the GO is ${age} s old (max $GLM53_GO_MAX_AGE_S s): not this turn's"; exit 75; }
export GLM53_RUN=${GLM53_RUN:-resident_$(date -u +%Y%m%dT%H%M)}  # a fresh run dir
# this GO is used from here on, launched or refused: a restarting launcher must not take it again (watch_sched_go.sh
# reads work-s9/go_seen; it also records the GOs it handles itself)
echo "${GLM53_GO_LINE%%Z *}Z taken by launch_resident.sh for $GLM53_RUN (pid $$)" >> $W/go_seen
# a retired target (scripts/targets/<int>_resident.sh exports the reason) never launches: refused, the coordinator told
if [ -n "${GLM53_TARGET_RETIRED:-}" ]; then
  echo "target ${GLM53_TARGET:-?} is retired: $GLM53_TARGET_RETIRED"
  msg="resident run NOT started -- target ${GLM53_TARGET:-?} is retired ($GLM53_TARGET_RETIRED)"
  echo "$(date -u +%FT%TZ) stage9: $msg" >> $W/COORDINATOR_REQUEST.txt
  exit 75
fi
# the serving target (tree, 14b's env, gate jobs, image): defaults = int8 (resident1); a target file
# (work-s9/resident_target.sh, run by watch_sched_go.sh) exports another int's values, then runs this launcher
export GLM53_S9_TREE=${GLM53_S9_TREE:-/opt/glm53/tt-metal-s9i8}
# the plugin: the LMCache join by default; a target may pin another (MTP speculative decoding: glm53-s9-serving); the
# rehearsal verdict must name the same src and rev
export GLM53_VENV=/opt/glm53/venv-lmcache
export GLM53_PLUGIN_SRC=${GLM53_PLUGIN_SRC:-/opt/glm53/vllm-tt-plugin-lmcache-join/src}
# MTP speculative decoding (Stage 17): GLM53_SPEC_K (the backend builds its K-draft speculative step at load) and
# GLM53_SPEC_CONFIG (serve_window.sh passes it as vLLM's --speculative-config); both unset = no speculation
export GLM53_BACKEND_MODULE=models.autoports.zai_org_glm_5_3_flash.tt.stack_model
export GLM53_BACKEND_KW='{"native_t": true, "weight_cache": {}}'  # = tests/s14b/m5common.serving's construction
export GLM53_WEIGHT_SOURCE=image GLM53_IMAGE_REQUIRED=1 GLM53_IMAGE_BUILD=0 GLM53_EXPECT_WEIGHT_SOURCE=image
# GLM53_LAUNCH_WEIGHTS=tensor (a launch-owned key; program / Stage 16 2026-10-22 09:04Z: the tp14b DRAM image held
# stale prefill weights, int25's prefill is right with TENSOR-loaded weights -- serve with tensors until the image is
# rebuilt): the server loads the tensor cache (GLM53_WEIGHT_SOURCE / _IMAGE_REQUIRED unset, the startup timeline's
# weight source not pinned); the pre-mesh check then treats the env job's GLM53_WEIGHT_SOURCE as a policy switch and
# matches the rehearsal verdict on everything but the image (code, env, backend kw, venv, plugin, spec). Load: 4-11
# min inside the window (serve_window.sh waits for health until the window deadline).
if [ "${GLM53_LAUNCH_WEIGHTS:-image}" = tensor ]; then
  unset GLM53_WEIGHT_SOURCE GLM53_IMAGE_REQUIRED GLM53_EXPECT_WEIGHT_SOURCE
fi
# GLM53_LAUNCH_WEIGHTS=build (program 2026-10-27T01:03:02Z: the Galaxy's first light, W4): a host with no image yet.
# The server finds no image under GLM53_TP_IMAGE_ROOT, loads the tensor cache under the recorder and WRITES the image
# there (GLM53_IMAGE_BUILD=1; tt/tp_image.py build_or_restore). Nothing is pinned or required. The pre-mesh check skips
# the pinned manifest and the mock rehearsal (none exists for a new host) and takes the cache paths as host paths; the
# env job, its gate rows and every model knob are checked as for an image launch.
if [ "${GLM53_LAUNCH_WEIGHTS:-image}" = build ]; then
  export GLM53_WEIGHT_SOURCE=image GLM53_IMAGE_BUILD=1
  unset GLM53_IMAGE_REQUIRED GLM53_TP_IMAGE_DIR GLM53_IMAGE_PIN GLM53_IMAGE_MOCK_OK GLM53_EXPECT_WEIGHT_SOURCE
fi
# GLM53_LAUNCH_PINNED=off (program 2026-10-27T01:29:52Z (2): under the Galaxy's KMD 2.9.1-pre dma_mask_bits=64 +
# DMA-FQ the PINNED host-memory paths can be silently misdirected). The launch has no LMCache connector, so no external
# pinned CPU pool and no page DMA: the adapter starts its page-DMA only in kv_cache_connector_views, which the plugin
# calls only with a connector. The adapter's page-DMA switches are off as well. The image BUILD (read_region) pins
# nothing. The pre-mesh check exempts exactly these keys.
if [ "${GLM53_LAUNCH_PINNED:-on}" = off ]; then
  export GLM53_LMCACHE=0 GLM53_KV_T2=0 GLM53_KDA_CKPT_DMA=0
fi
# 14b's served env for the target (default int8: work-s14b q_mesh/u01_env, the image-building open of 2026-10-21
# 00:41Z); the pre-mesh check compares the whole GLM53_STACK_ / GLM53_TP_ set with the target's env job
: "${GLM53_STACK_KNOBS:=s7a283}" "${GLM53_TP_XSL:=1}" "${GLM53_TP_XSL_SLICED:=1}" "${GLM53_STACK_WIDTHS:=none}"
: "${GLM53_STACK_PREFILL_TRACES:=0}" "${GLM53_STACK_SHARE_FABRIC:=1}" "${GLM53_STACK_KNOB_SET:=s:GLM53_MHC_PRE_SPLIT=1}"
: "${GLM53_STACK_NATIVE_WIDTHS:=8,16}" "${GLM53_EXPECT_CHUNK:=32}"
: "${GLM53_TP_IMAGE_ROOT:=/opt/glm53/data/dram_images/tp14b}"
export GLM53_STACK_KNOBS GLM53_TP_XSL GLM53_TP_XSL_SLICED GLM53_STACK_WIDTHS GLM53_STACK_PREFILL_TRACES
export GLM53_STACK_SHARE_FABRIC GLM53_STACK_KNOB_SET GLM53_STACK_NATIVE_WIDTHS GLM53_EXPECT_CHUNK GLM53_TP_IMAGE_ROOT
# the pinned image (resident1 restored it; 14b re-ruled it in place at 01:50Z); any change = a new rehearsal
[ "${GLM53_LAUNCH_WEIGHTS:-image}" = build ] || export GLM53_IMAGE_PIN=${GLM53_IMAGE_PIN:-fde4697e51b2-tp-ds-v1-2x4-1530493535ed}
export TMPDIR=/opt/glm53/tz
ENV_JOB=${GLM53_14B_ENV_JOB:-u01_env}; GATE=${GLM53_14B_GATE:-u02_step_B32n u03_tflag_B32n u05_gendet32 u04_widths}
# before the mesh: the pinned stack / TP env == 14b's env job (same prefixes), that open's gate verdicts hold, the
# pinned image's manifest exists and a mock rehearsal verdict (GLM53_IMAGE_MOCK_OK: mock_image_serve.sh's, full pass
# only) matches this launch: image, manifest sha, env, image restore mode, backend kw, code, venv, plugin; the env job
# and the gate rows ran on GLM53_14B_TREE (when set); otherwise (or no explicit OK: a timeout / kill) refuse and tell
# the coordinator
MOCK_OK=${GLM53_IMAGE_MOCK_OK:-$W/mock_verdict_${GLM53_TARGET:-int8}.json}  # mock_image_serve.sh, full pass only
MD=models/autoports/zai_org_glm_5_3_flash; T=$GLM53_S9_TREE  # the identity mock_image_serve.sh records: code,
# the pinned image's manifest (14b re-rules images in place: 01:50Z), the venv + plugin the server runs
PLUGIN_REV=$(git -C ${GLM53_PLUGIN_SRC%/src} rev-parse --short=11 HEAD 2>/dev/null)$( [ -n "$(git -C ${GLM53_PLUGIN_SRC%/src} status --porcelain -- src 2>/dev/null)" ] && echo -dirty)
MANIFEST_SHA=$(timeout 60 sha256sum $GLM53_TP_IMAGE_ROOT/$(basename ${GLM53_IMAGE_PIN:-none})/manifest.json 2>/dev/null | cut -c1-16)
CODE=tt:$(git -C $T rev-parse --short=12 HEAD:$MD/tt)+common:$(git -C $T rev-parse --short=8 HEAD:$MD/tests/common.py)$( [ -n "$(git -C $T status --porcelain -- $MD/tt $MD/tests/common.py)" ] && echo -dirty)
chk=$(timeout 60 /opt/glm53/kit/.agents/scripts/no-tt-devices /usr/bin/python3 -I -c "
import json, os, sys
sys.excepthook = lambda t, e, tb: print('pre-mesh check error: %s: %s' % (t.__name__, e))  # one line, never a traceback
q = '/opt/glm53/work-s14b/q_mesh'
pre = ('GLM53_STACK_', 'GLM53_TP_')
tree = os.environ.get('GLM53_14B_TREE', '').rstrip('/')  # the 14b tree of the env job's and gate rows' scripts
# the env job: 14b's <job>.done in q_mesh, or GLM53_14B_ENV_FILE = an env-job file of the same form outside it (a
# coordinator-ruled overlay on a gated env job: the file names its base .done, the overlay keys and the ruling under
# 'stage9_overlay'; the base must exist and the file's env must equal the base's env + the overlay - its dropped keys)
env_file = os.environ.get('GLM53_14B_ENV_FILE', '')
job = json.load(open(env_file or q + '/' + sys.argv[1] + '.done'))
if env_file:
    ov = job.get('stage9_overlay') or {}
    try:
        base_env = json.load(open(ov.get('base', '')))['args']['env']
    except (OSError, ValueError, KeyError, TypeError):
        base_env = None
    # 'dropped': base keys a ruling removes (program 15:32:39Z / 15:33:21Z: MTP off, per-user admission)
    drop = set(ov.get('dropped') or [])
    ruled = {k: v for k, v in {**(base_env or {}), **(ov.get('keys') or {})}.items() if k not in drop}
    if base_env is None or not ov.get('authority') or drop & set(ov.get('keys') or {}) or job['args']['env'] != ruled:
        raise ValueError('GLM53_14B_ENV_FILE %s: not a ruled overlay of an existing env job' % env_file)
jenv = job['args']['env']
want = {k: v for k, v in jenv.items() if k.startswith(pre)}
BUILD = os.environ.get('GLM53_LAUNCH_WEIGHTS', 'image') == 'build'  # the image-building first launch of a new host
if BUILD:  # no image is pinned: the env job's pinned image dir is not this launch's
    want.pop('GLM53_TP_IMAGE_DIR', None)
# where the image lives is the host's (2026-10-27, the Galaxy W5: its own pin under its own root); WHICH image is the pin
# + its manifest sha, checked against the mock verdict below
LOCATION = ('GLM53_TP_IMAGE_DIR', 'GLM53_TP_IMAGE_ROOT')
for k in LOCATION:
    want.pop(k, None)
have = {k: v for k, v in os.environ.items() if k.startswith(pre) and k not in LOCATION}
err = [] if have == want else ['env differs: %s' % sorted(set(have.items()) ^ set(want.items()))]
# every other key the env job sets (the weight source, the image restore mode 14b's gate ran: int21+ sets
# GLM53_IMAGE_ORDER / _COLD_ORDER / _EMULATE_REAL, a kernel knob like GLM53_MLA_ATT_NKMAX, ...) must have the same value
# in this launch: nothing the gate ran is dropped silently (unset == unset). Two launch-policy switches are exempt: the
# image build switch (a 14b env job may build the image, e.g. u01_env / s01_env set 1; a Stage 9 launch never builds,
# 0 above) and the required restore (a Stage 9 launch always requires the image, 1 above; the mock requires it within
# its own restore call)
POLICY = ('GLM53_IMAGE_BUILD', 'GLM53_IMAGE_REQUIRED')
if os.environ.get('GLM53_LAUNCH_PINNED', 'on') == 'off':  # (the pinned-memory paths off: the launch's own choice)
    POLICY = POLICY + ('GLM53_LMCACHE', 'GLM53_KV_T2', 'GLM53_KDA_CKPT_DMA')
TENSORS = os.environ.get('GLM53_LAUNCH_WEIGHTS', 'image') == 'tensor'
if TENSORS:  # the weight source is this launch's choice (above), not the env job's
    POLICY = POLICY + ('GLM53_WEIGHT_SOURCE',)
# a ruled env-match waiver for ONE key (program 2026-10-27T19:06:00Z, k4dg's trace-region death: the d0 mock never
# captures traces, so its verdict stays valid for a trace-only knob): GLM53_ENV_MATCH_WAIVER = <KEY>=<value> then the ruling text;
# this launch may carry exactly that value for that one key where the env job (and so its mock) set another; never a
# GLM53_STACK_ / GLM53_TP_ key (the image construction); every other key stands
wkv = os.environ.get('GLM53_ENV_MATCH_WAIVER', '').split(' ', 1)[0]
wk, _, wval = wkv.partition('=')
waived = lambda k: bool(wk) and k == wk and not k.startswith(pre) and os.environ.get(k) == wval
err += ['env job sets %s=%s, this launch %s' % (k, v, os.environ.get(k)) for k, v in sorted(jenv.items())
        if not k.startswith(pre) and k not in POLICY and os.environ.get(k) != v and not waived(k)]
# and the other way: any other GLM53_ key this launch carries that the env job does not set is a knob the gate never
# ran and the rehearsal never imported (a model knob such as GLM53_PF_BATCH or GLM53_MOCK, an image restore mode, a
# server-config override such as GLM53_EXTRA_SERVER_ARGS / GLM53_MAX_BATCHED_TOKENS / GLM53_MAX_MODEL_LEN) -> refused.
# Exempt: the launch's own keys (the policy switches, the pin / rehearsal pointer, the target's identity, the GO, the
# lock wait, the interpreter / plugin / backend construction compared below), the window / workload controls the
# caller may set (when, ports, client CPUs, step budgets: no model or server-config effect; not GLM53_KEEP_UP, which
# would hold the mesh past the scheduler's cap) and the keys the program / Stage 9 env.sh set unconditionally (paths:
# a caller's value is overwritten)
OWN = POLICY + ('GLM53_IMAGE_PIN', 'GLM53_IMAGE_MOCK_OK', 'GLM53_TARGET', 'GLM53_RUN', 'GLM53_S9_TREE', 'GLM53_LAUNCH_WEIGHTS',
                'GLM53_LAUNCH_PINNED',
                'GLM53_14B_ENV_JOB', 'GLM53_14B_GATE', 'GLM53_14B_TREE', 'GLM53_GO_LINE', 'GLM53_GO_MAX_AGE_S',
                'GLM53_WINDOW_MIN', 'GLM53_LOCK_WAIT_S', 'GLM53_VENV', 'GLM53_PLUGIN_SRC', 'GLM53_BACKEND_MODULE',
                'GLM53_BACKEND_KW', 'GLM53_EXPECT_WEIGHT_SOURCE', 'GLM53_EXPECT_CHUNK', 'GLM53_TARGET_RETIRED',
                'GLM53_SPEC_K', 'GLM53_SPEC_CONFIG',  # the spec pair: compared with the rehearsal's below
                'GLM53_14B_ENV_FILE',
                # launch policy read by serve_window.sh before the mesh request (review rounds 14-15): the ruled plugin
                # fixes waiver and the d0 capture gate -- not model knobs
                'GLM53_PLUGIN_FIXES_WAIVER', 'GLM53_CAPTURE_GATE',
                # program 2026-10-27T10:00:40Z: the PD D-role's mock verdict waived as its launch gate (pdD targets only)
                'GLM53_IMAGE_MOCK_WAIVER',
                # program 2026-10-27T19:06:00Z: the one-key env-match waiver (read above)
                'GLM53_ENV_MATCH_WAIVER')
# GLM53_KEEP_UP: the standing server (program 2026-10-22 09:42:26Z: int25t stays up as the resident production server
# when its strict TF passes; serve_window.sh keeps the server until <run>/STOP) -- a target sets it only under that ruling
# GLM53_SELFTEST(=strict|warn|off) / GLM53_SELFTEST_OUT: the selftest plugin's startup accuracy gate (coordinator
# 2026-10-22 10:03:12Z: srv/selftest; strict = the socket never binds on a miss) -- a readiness gate, not a model knob
# GLM53_DECODE_PROBE_EVERY: the adapter's host-overhead probe (a drained decode step every N-th steady step; timing only,
# tokens unchanged: test_host_step_overhead_probe_and_off_hot_path_events; program 10:05:13Z)
# GLM53_WORKLOAD: the client workload (default w12_int3_rerun.sh; w15_prefix.sh = the APC-on launch's cells)
# GLM53_VLLM_KV_POOL_TOKENS: the KV pool 14b's kv_pool_probe measured at a context rung (int26: 32K); set only to a value
# <= the rehearsal's pool (the adapter default, max_num_seqs x max_model_len): a smaller pool only lowers DRAM use
CONTROL = ('GLM53_AFTER_BUSY', 'GLM53_AFTER_PID', 'GLM53_EARLIEST_START', 'GLM53_LATEST_START', 'GLM53_HARD_STOP', 'GLM53_KEEP_UP',
           'GLM53_SELFTEST', 'GLM53_SELFTEST_OUT', 'GLM53_DECODE_PROBE_EVERY', 'GLM53_VLLM_KV_POOL_TOKENS', 'GLM53_WORKLOAD',
           'GLM53_PORT', 'GLM53_POST_GO_WAIT_S', 'GLM53_CLIENT_CPUS', 'GLM53_LONE_2048_BUDGET_S',
           'GLM53_WINDOW_DEADLINE',
           # the host profile (scripts/host_profile.sh, replicate/host_glx.sh): which lock tool / group and which block's
           # busy files -- where the row runs, not what it runs
           'GLM53_TT_LOCK', 'GLM53_LOCK_GROUP', 'GLM53_MESH_TAG', 'GLM53_SERVER_CPUS',
           # GLM53_GRID_X (ruling 2026-10-27T02:12:31Z): pins another host's compute grid to host-8chip's 11 x 10, so the
           # construction equals the env job's (image map, traces, bitwise units)
           'GLM53_GRID_X',
           # the hugepage pool hygiene switches (serve_window.sh: the dead-pid sweep + the free-page refusal; coordinator
           # 2026-10-28T00:05:00Z) -- launch policy, no model effect
           'GLM53_HUGEPAGES_MIN', 'GLM53_POOL_SWEEP',
           # program [session 10] 2026-10-29T02:30:48Z (3) / 02:32:31Z: the no-preempt admission cap on every mesh window until 12's
           # fix (B) lands: the plugin's KV-budget reserve counts the full max_tokens (vllm_tt_plugin/scheduler.py:507-510, 0 = no cap),
           # so no request outgrows its reservation; an admission policy (which requests wait), no model / numerics effect, an
           # env-only change with no re-mock (02:32:31Z). Only the value 0 is accepted (checked below).
           'GLM53_ADMIT_RESERVE_TOKENS')
ENVSH_FIXED = ('GLM53_WS', 'GLM53_LOCAL', 'GLM53_EVIDENCE', 'GLM53_GOLDENS', 'GLM53_HF_SNAPSHOT', 'GLM53_MODEL_DIR',
               'GLM53_REF_VENV', 'GLM53_WORK', 'GLM53_E9', 'GLM53_CODE_SHA', 'GLM53_PLUGIN_RUNTIME')
# the env.sh keys that keep a caller's value (${X:-default}): the rehearsal ran env.sh's defaults, so a launch may
# carry only those values (env.sh lines: the program's GLM53_CACHE_DIR, Stage 9's GLM53_FUSED_EXPERTS_CACHE and
# GLM53_PARALLEL_WARMUP / _CAPTURE) -- unless the env job sets the key: then the env job's value rules (checked above,
# and the rehearsal imported it)
ENVSH_DEFAULT = {'GLM53_CACHE_DIR': '/opt/glm53/tt-cache/glm53-flash',
                 'GLM53_FUSED_EXPERTS_CACHE': '/opt/glm53/tt-cache/glm53-flash-fused-perm',
                 'GLM53_PARALLEL_WARMUP': '1', 'GLM53_PARALLEL_CAPTURE': '1'}
err += ['this launch sets %s=%s, env.sh (the rehearsal) %s' % (k, os.environ[k], v) for k, v in ENVSH_DEFAULT.items()
        if k not in jenv and k in os.environ and os.environ[k] != v
        and k not in ('GLM53_CACHE_DIR', 'GLM53_FUSED_EXPERTS_CACHE')]  # host paths (2026-10-27, the Galaxy); the content is
        # the weights manifest's, which the image identity (weights id) pins
extra = sorted(k for k in os.environ if k.startswith('GLM53_') and not k.startswith(pre)
               and k not in OWN + CONTROL + ENVSH_FIXED and k not in ENVSH_DEFAULT and k not in jenv)
err += ['this launch sets %s, the env job does not' % k for k in extra]
if os.environ.get('GLM53_ADMIT_RESERVE_TOKENS') not in (None, '0') and 'GLM53_ADMIT_RESERVE_TOKENS' not in jenv:  # 02:32:31Z: only the no-preempt 0
    err.append('GLM53_ADMIT_RESERVE_TOKENS=%s: a launch may set only 0 (the no-preempt cap, program 2026-10-29T02:32:31Z)' % os.environ['GLM53_ADMIT_RESERVE_TOKENS'])
# Stage 16's batched prefill moves each slot's KDA state only inside its trace: prefill_tp.py builds the state I/O
# only for GLM53_PF_STATE_IO=trace (any other value = the single-request SlotOps path) and write_batch /
# forward_batch skip it otherwise, so a batch key (GLM53_PF_BATCH = nb > 1) with any other mode would leave the
# batched slots' recurrent state unwritten
try:
    pf_nb = int(os.environ.get('GLM53_PF_BATCH', '0') or 0)
except ValueError:
    pf_nb = None
if pf_nb is None or pf_nb < 0:
    err.append('GLM53_PF_BATCH=%s is not a count >= 0' % os.environ.get('GLM53_PF_BATCH'))
elif pf_nb > 1 and os.environ.get('GLM53_PF_STATE_IO', 'trace') != 'trace':
    err.append('GLM53_PF_BATCH=%s with GLM53_PF_STATE_IO=%s: the batched prefill moves no KDA state outside '
               'state-I/O mode trace' % (pf_nb, os.environ.get('GLM53_PF_STATE_IO')))
# what the rehearsal must have run: every key of the env job outside the GLM53_STACK_ / GLM53_TP_ prefixes (compared
# above as stack_env) but the policy switches and GLM53_WEIGHT_SOURCE (the image mock restores the image by
# construction: tests/vllm/stack_image_mock.py) -- e.g. int21's image restore mode, GLM53_MLA_ATT_NKMAX
BY_CONSTRUCTION = ('GLM53_WEIGHT_SOURCE',)
want_job = {k: v for k, v in jenv.items() if not k.startswith(pre) and k not in POLICY + BY_CONSTRUCTION}
def on_tree(j):  # gate rows are matched by name; names repeat across 14b's ints: the job's declared script must be
    if not tree:  # under this target's 14b tree
        return True
    try:
        return json.load(open(q + '/' + j + '.done')).get('script', '').startswith(tree + '/')
    except (OSError, ValueError):
        return False
env_ok = job.get('script', '').startswith(tree + '/') if (env_file and tree) else on_tree(sys.argv[1])
err += [] if env_ok else ['%s: its script is not under %s' % (env_file or sys.argv[1], tree)]
def verdict(j):  # 14b's job server: top-level ok = the job did not raise; the verdict is result.ok (+ the bars)
    try:
        d = json.load(open(q + '/' + j + '.out'))
    except (OSError, ValueError):
        return j + ' missing'
    r = d.get('result') or {}
    if os.path.exists(q + '/' + j + '.failed') or not d.get('ok') or not r.get('ok'):
        return j + ' failed'
    if not on_tree(j):
        return j + ': its script is not under ' + tree
    if 'tflag' in j and not (r.get('top1', 0) >= 0.90 and r.get('top5', 0) >= 0.98 and r.get('top100', 0) >= 1.0):
        return j + ' below the TF bar'
    if 'gendet' in j and not all(r.get('outputs_equal_to_run0') or [False]):
        return j + ' not deterministic'
    return ''
err += [e for e in map(verdict, sys.argv[2].split()) if e]
pin = os.path.join(os.environ['GLM53_TP_IMAGE_ROOT'], os.path.basename(os.environ.get('GLM53_IMAGE_PIN', '')))
err += [] if TENSORS or BUILD or os.path.exists(os.path.join(pin, 'manifest.json')) else ['no manifest for the pinned image ' + pin]
try:
    m = json.load(open(sys.argv[3]))
except (OSError, ValueError):
    m = {}
code = sys.argv[4]
# the speculative-decoding pair this launch serves with must be the one the rehearsal served with (unset == unset); the
# K the backend builds must be the K vLLM asks for
SPEC = ('GLM53_SPEC_K', 'GLM53_SPEC_CONFIG')
spec = {k: os.environ[k] for k in SPEC if os.environ.get(k)}
if spec:
    try:
        nk = json.loads(spec.get('GLM53_SPEC_CONFIG', '{}')).get('num_speculative_tokens')
    except ValueError:
        nk = None
    if set(spec) != set(SPEC) or str(nk) != spec.get('GLM53_SPEC_K'):
        err.append('speculative decoding needs GLM53_SPEC_K and GLM53_SPEC_CONFIG with the same K (%s)' % spec)
def tr_ok(mj, wj):  # == except GLM53_TRACE_REGION, which may only grow (program 2026-10-23T15:36:38Z, n29h7s2 TR768)
    # a mock that ran the PMK descriptor cache in verify mode (every descriptor rebuilt + checked) rehearsed the cache ON
    # (program 2026-10-27T02:14:40Z, the k4 image mock)
    if mj.get('GLM53_PMK_DESC_CACHE') == 'verify' and wj.get('GLM53_PMK_DESC_CACHE') == '1':
        mj = dict(mj, GLM53_PMK_DESC_CACHE='1')
    k = 'GLM53_TRACE_REGION'
    strip = lambda d: {x: v for x, v in d.items() if x != k}
    try:
        grows = int(wj.get(k, 0) or 0) >= int(mj.get(k, 0) or 0)
    except ValueError:
        grows = wj.get(k) == mj.get(k)
    return strip(mj) == strip(wj) and grows


ok = (m.get('pass') is True and (m.get('spec') or {}) == spec and (TENSORS or m.get('image') == os.path.basename(pin)) and {k: v for k, v in (m.get('stack_env') or {}).items() if k not in LOCATION} == want
      and tr_ok(m.get('job_env', m.get('image_env')) or {}, want_job)  # image_env: the older verdicts' name
      and m.get('backend_kw') == os.environ['GLM53_BACKEND_KW'] and m.get('code') == code and not code.endswith('-dirty')
      and (TENSORS or (sys.argv[5] and m.get('manifest_sha') == sys.argv[5])) and m.get('venv') == os.environ['GLM53_VENV']
      and m.get('plugin_src') == os.environ['GLM53_PLUGIN_SRC'] and m.get('plugin_rev') == sys.argv[6]
      and not sys.argv[6].endswith('-dirty'))
# program 2026-10-27T10:00:40Z: the PD D role cannot serve the mock's fresh-prefill self-test by design, so its mock
# verdict is WAIVED as its launch gate: GLM53_IMAGE_MOCK_WAIVER names the ruling, and only a PD D-role window (GLX_PD_ROLE=D,
# GLM53_RUN ...pdD_...: replicate/pd/glx_pd_role.sh) may use it; every pre-mesh check that does not read the verdict stands
# (the D mock writes no verdict -- its self-test never ends -- so the verdict's identity match goes with it; the served code
# and plugin must still be clean trees: review round 40)
if not ok and os.environ.get('GLM53_IMAGE_MOCK_WAIVER') and os.environ.get('GLX_PD_ROLE') == 'D' and 'pdD_' in os.environ.get('GLM53_RUN', ''):
    ok = not code.endswith('-dirty') and not sys.argv[6].endswith('-dirty')
err += [] if ok or BUILD else ['no passing mock rehearsal (served, clean close) of the pinned image (manifest %s) with this env, '
                      'backend kw, code %s and plugin %s (%s: pass %s, image %s, code %s, manifest %s, plugin %s)' % (
                          sys.argv[5] or '?', code, sys.argv[6], sys.argv[3], m.get('pass'), m.get('image'), m.get('code'),
                          m.get('manifest_sha'), m.get('plugin_rev'))]
print('; '.join(err) if err else 'OK')" "$ENV_JOB" "$GATE" "$MOCK_OK" "$CODE" "$MANIFEST_SHA" "$PLUGIN_REV" 2>&1)
# build mode NEVER pulls (01:29:52Z (2)): tt/tp_image.py restores any finished image of these weights / layout / mesh under
# the root before it builds, so a root already holding one is refused here (move it away, or launch in image mode)
if [ "$chk" = OK ] && [ "${GLM53_LAUNCH_WEIGHTS:-image}" = build ]; then
  wid=$(sha256sum /opt/glm53/kit/weights/manifest.json 2>/dev/null | cut -c1-12)
  have_img=$(ls -d $GLM53_TP_IMAGE_ROOT/${wid:-none}-${GLM53_TP_LAYOUT:-tp-ds-v1}-2x4-*/manifest.json 2>/dev/null | head -1)
  [ -n "$have_img" ] && chk="build mode: $GLM53_TP_IMAGE_ROOT already holds $(dirname $have_img) -- the server would restore (pull) it, not build"
fi
# F2 (ruling 2026-10-27T02:40:15Z: OFF in k4): TT_METAL_ALL_GATHER_UNICAST=1 needs a C++ build that carries it; on a tree
# linking a build without it the switch would be silently inert
if [ "$chk" = OK ] && [ "${TT_METAL_ALL_GATHER_UNICAST:-0}" = 1 ] && ! grep -qa TT_METAL_ALL_GATHER_UNICAST $GLM53_S9_TREE/build_Release/ttnn/_ttnncpp.so 2>/dev/null; then
  chk="TT_METAL_ALL_GATHER_UNICAST=1 (F2) but $GLM53_S9_TREE's C++ build lacks it (F2 is k5, with its own build)"
fi
[ "$chk" = OK ] || { chk=${chk:-no verdict (timeout / killed)}; echo "pre-mesh check failed: $chk"; echo "$(date -u +%FT%TZ) stage9: resident run NOT started -- pre-mesh check: $chk" >> $W/COORDINATOR_REQUEST.txt; exit 75; }
[ -n "${GLM53_ENV_MATCH_WAIVER:-}" ] && echo "$(date -u +%FT%TZ) stage9: ${GLM53_RUN:-$GLM53_TARGET} launches with a one-key env-match WAIVER: $GLM53_ENV_MATCH_WAIVER" >> $W/COORDINATOR_REQUEST.txt
[ -n "${GLM53_IMAGE_MOCK_WAIVER:-}" ] && [ "${GLX_PD_ROLE:-}" = D ] && case "${GLM53_RUN:-}" in *pdD_*) echo "$(date -u +%FT%TZ) stage9: ${GLM53_RUN} (the PD D role) launches with its mock verdict WAIVED: $GLM53_IMAGE_MOCK_WAIVER" >> $W/COORDINATOR_REQUEST.txt;; esac
cd $W
# review round 30 O4: the tag check at launch time (the 21:44Z ERROR class: a GO-launched cf3h target whose tag missed a
# gate-window list ran a window without its isolation verdict; ruling 04:07:52Z (c): the readiness profile early). A
# cf3h-family tag (n29h7p2h*) missing from a list is staged here, before the window starts, and the launch log names it
t9=${GLM53_TARGET#int27}
case "$t9" in n29h7p2h*)
  staged=""
  for f in GATE_WINDOW ISOCTL_FIRST WARMCTX CELLS_1M READINESS_FIRST; do
    grep -qx "$t9" $W/$f 2>/dev/null || { echo "$t9" >> $W/$f; staged="$staged $f"; }
  done
  [ -e $W/CELLS_1M_PLAN.$t9 ] || { cp $W/CELLS_1M_PLAN.n29h7p2hjw0 $W/CELLS_1M_PLAN.$t9 && staged="$staged CELLS_1M_PLAN"; }
  [ -n "$staged" ] && echo "launch-time tag check: staged $t9 for the gate window:$staged";;
esac
# program 2026-10-26T03:01:58Z (1), the starvation study's launcher finding: the tt keys a launch serves are worker_l1 +
# GLM53_TT_CFG_EXTRA (the env job's key; mock_image_serve.sh rehearses exactly that). GLM53_TT_EXTRA_CFG is rebuilt
# below from those two, so a target's own GLM53_TT_EXTRA_CFG is NOT served (e.g. the i-family's
# '"decode_interleave_prefill_steps": 1' never reached a server). Policy keys go in GLM53_TT_CFG_EXTRA; the dropped value
# is named in the launch log
[ -n "${GLM53_TT_EXTRA_CFG:-}" ] && echo "note: the target's GLM53_TT_EXTRA_CFG ($GLM53_TT_EXTRA_CFG) is not served; the tt keys are worker_l1 + GLM53_TT_CFG_EXTRA (${GLM53_TT_CFG_EXTRA:-none})"
# review round 30 O4: the default worker_l1_size only when GLM53_TT_CFG_EXTRA does not name one (it was passed twice, the
# last winning; the served value is unchanged)
WL1_DEFAULT='"worker_l1_size": 1264512'; case "${GLM53_TT_CFG_EXTRA:-}" in *'"worker_l1_size"'*) WL1_DEFAULT='';; esac
TT_KEYS="$WL1_DEFAULT${WL1_DEFAULT:+${GLM53_TT_CFG_EXTRA:+, }}${GLM53_TT_CFG_EXTRA:-}"
GLM53_TRACKING=1 GLM53_MAX_SEQS=${GLM53_MAX_SEQS:-32} GLM53_WINDOW_MIN=${GLM53_WINDOW_MIN:-20} GLM53_TRACE_REGION=${GLM53_TRACE_REGION:-536870912} \
  GLM53_TT_EXTRA_CFG="$TT_KEYS" GLM53_LOCK_WAIT_S=${GLM53_LOCK_WAIT_S:-3600} \
  exec /opt/glm53/kit/doc/ttnn/09-vllm/scripts/serve_window.sh \
  ${GLM53_WORKLOAD:-/opt/glm53/kit/doc/ttnn/09-vllm/scripts/w12_int3_rerun.sh}
