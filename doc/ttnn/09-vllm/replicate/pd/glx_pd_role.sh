#!/usr/bin/env bash
# A Galaxy PD ROLE window (program 2026-10-27T08:02:51Z; Form B v2, design/pd_disagg.md section 10): P = the
# prefill-only role on blk0 (port 8100), D = the decode-only role on blk1 (port 8101), each one TP8 vLLM resident of the
# same served line; the proxy (pd_proxy.py :8000) is their only client. Run through the kit's block runner:
#   GLX_PD_ROLE=P replicate/glx_block_run.sh 0 '<9 glx:blk0: GO ...>' replicate/pd/glx_pd_role.sh
#   GLX_PD_ROLE=D replicate/glx_block_run.sh 1 '<9 glx:blk1: GO ...>' replicate/pd/glx_pd_role.sh
# (or glx_pd_run.sh, which launches both + the proxy). Everything else is the served launch (launch_resident.sh), as
# scripts/targets/glx_w5_k3.sh:
#  - the base target's exports (GLX_PD_TAG, default n29h7p2hk4b: the PD hand-off needs resume-as-decode, the k4 line);
#  - the ROLE overlay env job as GLM53_14B_ENV_FILE (pd_envjobs.py: 14b's role knobs + the shared P / D keys + 12's
#    Mooncake LMCache keys) and every GLM53_ key of its overlay exported, so the pre-mesh check's env compare holds;
#  - 12's plugin with the P / D hand-off (GLX_PD_PLUGIN_SRC), 12's LMCache tree with the Mooncake connector
#    (GLX_PD_LMCACHE_TREE) and the Mooncake process env (MC_GID_INDEX from GLX_MC_GID_INDEX, MC_MS_AUTO_DISC=0);
#  - the block's image root (host_glx.sh: blk1 -> dram_images/tp14b_int27_blk1) + its pin (GLX_PIN) + a mock verdict of
#    THIS role's env (GLX_MOCK_OK, default <image dir>/mock_verdict_glx_<tag>pd<role>.json);
#  - GLM53_GRID_X=11, the role's port 8100 + block, the workload w_pd_role.sh (a health line, then the keep-up resident).
# A D window needs the serve_window_mooncake.patch hook (GLM53_LMCACHE_EXTRA_YAML) in the workspace copy it runs, and
# P's KDA checkpoint dir on the same host (GLM53_KDA_CKPT_DIR is shared by construction).
set -u
S=/opt/glm53/kit/doc/ttnn/09-vllm/scripts
PD=$(dirname "$(readlink -f "$0")")
B=${GLM53_MESH_TAG:-blk0}; B=${B#blk}
ROLE=${GLX_PD_ROLE:-$([ "$B" = 0 ] && echo P || echo D)}
# H = the fleet's HYBRID tray (program 2026-10-27T16:54:15Z, Mode 1 DP4): the served form + the shared store (glx_fleet_run.sh)
case $ROLE in P|D|H) ;; *) echo "refusing: GLX_PD_ROLE=$ROLE (P, D or H)"; exit 2;; esac
TAG=${GLX_PD_TAG:-n29h7p2hk4b}
T=$S/targets/int27${TAG}_resident.sh
[ -s $T ] || { echo "refusing: no base target $T"; exit 2; }
eval "$(grep -E '^export ' $T)"
# (1) host_glx.sh exports GLM53_WORK only for blocks > 0 (program 2026-10-27T11:38:03Z)
GLM53_WORK=${GLM53_WORK:-/opt/glm53/work-s9}
EJ=${GLX_PD_ENVJOB:-$GLM53_WORK/envjobs/int27${TAG}pd${ROLE}_img_env.done}
[ -s "$EJ" ] || { echo "refusing: no role env job $EJ (compose it: pd_envjobs.py pd_config.json $GLM53_WORK/envjobs)"; exit 2; }
OV=$(/usr/bin/python3 -I - "$EJ" "$ROLE" <<'PY'
import json, shlex, sys
j = json.load(open(sys.argv[1])); o = j.get("stage9_overlay") or {}
if o.get("role") != sys.argv[2]:
    sys.exit(f"echo 'refusing: {sys.argv[1]} is the {o.get('role')} overlay, not {sys.argv[2]}'; exit 2")
bad = sorted(k for k, v in (o.get("keys") or {}).items() if "<" in str(v) and ">" in str(v))
if bad:
    print(f"echo 'refusing: placeholders in the {sys.argv[2]} overlay: {', '.join(bad)} (fill pd_config.json)'; exit 2"); sys.exit()
for k, v in (o.get("keys") or {}).items():
    print(f"export {k}={shlex.quote(str(v))}")
PY
) || exit 2
eval "$OV"
export GLM53_14B_ENV_FILE=$EJ
# the transport's PROCESS env (program 2026-10-28T00:02:56Z, pair 32: mooncake.process_env reached no process, and D's
# batched 1M get cancelled at LMCache's 10 s default): pd_envjobs.py's stage9_overlay.process_env (the LMCACHE_* / MC_* keys of
# pd_config's mooncake.process_env, e.g. LMCACHE_BLOCKING_TIMEOUT_SECS=600 + the MC_TCP_* queue sizes) exported here, so the
# API server and its EngineCore inherit them (LMCache's update_config_from_env reads LMCACHE_<KEY> over the yaml; the
# engine's Mooncake client reads MC_*). Not env-job keys: the pre-mesh compare and the mock verdicts are unchanged.
PE=$(/usr/bin/python3 -I -c 'import json, shlex, sys; o = json.load(open(sys.argv[1])).get("stage9_overlay") or {}
for k, v in (o.get("process_env") or {}).items():
    if k.startswith(("LMCACHE_", "MC_")): print("export " + k + "=" + shlex.quote(str(v)))' "$EJ") || exit 2
eval "$PE"
[ -n "$PE" ] && echo "process env (stage9_overlay.process_env): $(printf '%s' "$PE" | sed 's/^export //' | tr '\n' ' ')"
# the dead-pid pool sweep + the free-page refusal before every role launch (program 2026-10-28T00:05:00Z, pair 31: three
# stale 16-GiB glm53_pagepool files of dead engines held 48 GiB, D's 49-GiB pool ENOMEM): serve_window.sh's hugepage block
export GLM53_POOL_SWEEP=1
# (5) the launch paths default to the composed env job's (stage9_overlay.launch = pd_config.json's plugin_src / lmcache_tree /
# tree; the role mock reads the same), GLX_PD_* override (program 2026-10-27T11:41:23Z)
eval "$(/usr/bin/python3 -I -c 'import json, shlex, sys; l = (json.load(open(sys.argv[1])).get("stage9_overlay") or {}).get("launch") or {}
for k, v in (("EJ_PLUGIN", l.get("plugin_src")), ("EJ_LMCACHE", l.get("lmcache_tree")), ("EJ_TREE", l.get("tree")),
             ("EJ_PLUGIN_SHA", l.get("plugin_sha")), ("EJ_LMCACHE_SHA", l.get("lmcache_sha")),
             ("EJ_IMAGE_PIN", l.get("image_pin")), ("EJ_IMAGE_ROOT", l.get("image_root"))):
    print(k + "=" + shlex.quote(v or str()))' "$EJ")"
export GLM53_PLUGIN_SRC=${GLX_PD_PLUGIN_SRC:-${EJ_PLUGIN:-/opt/glm53/srv/s12lmc_s9h8t/src}}
# the kit's plugin pin (pd_config.json plugin_sha, program 2026-10-27T15:15:59Z): the plugin worktree must be AT it
if [ -n "${EJ_PLUGIN_SHA:-}" ] && [ -z "${GLX_PD_PLUGIN_SRC:-}" ]; then
  ph=$(git -C "$GLM53_PLUGIN_SRC/.." rev-parse HEAD 2>/dev/null)
  case "$ph" in "$EJ_PLUGIN_SHA"*) ;; *) echo "refusing: the plugin $GLM53_PLUGIN_SRC is at ${ph:-?}, the kit pins $EJ_PLUGIN_SHA (pd_config plugin_sha)"; exit 2;; esac
fi
LMT=${GLX_PD_LMCACHE_TREE:-$EJ_LMCACHE}; [ -n "$LMT" ] && export GLM53_LMCACHE_TREE=$LMT
# the kit's LMCache pin (pd_config lmcache_sha, program 2026-10-28T00:07:30Z (3): 23babb83 = the extra_config
# blocking_timeout_secs / mooncake_env overrides): the LMCache tree must be AT it
if [ -n "${EJ_LMCACHE_SHA:-}" ] && [ -z "${GLX_PD_LMCACHE_TREE:-}" ]; then
  lh=$(git -C "$GLM53_LMCACHE_TREE" rev-parse HEAD 2>/dev/null)
  case "$lh" in "$EJ_LMCACHE_SHA"*) ;; *) echo "refusing: the LMCache tree $GLM53_LMCACHE_TREE is at ${lh:-?}, the kit pins $EJ_LMCACHE_SHA (pd_config lmcache_sha)"; exit 2;; esac
fi
PDT=${GLX_PD_TREE:-$EJ_TREE}; [ -n "$PDT" ] && export GLM53_S9_TREE=$PDT
[ -n "${GLX_MC_GID_INDEX:-}" ] && export MC_GID_INDEX=$GLX_MC_GID_INDEX
export MC_MS_AUTO_DISC=0
case "${GLM53_LMCACHE_REMOTE_URL:-}" in mooncakestore://*)
  # the pair-33 overlay check (program 2026-10-28T00:27:56Z: in the kit's own launch path, inside the namespace where the
  # overlay's yaml path resolves): LMCache-tt 23babb83's blocking_timeout_secs >= 600 + mooncake_env in the extra_config yaml
  /usr/bin/python3 -I $PD/pd_overlay_check.py "$EJ" || { echo "refusing: the $ROLE overlay fails the pair-33 check (pd_overlay_check.py)"; exit 2; }
  grep -q 'GLM53_LMCACHE_EXTRA_YAML' $S/serve_window.sh \
    || { echo "refusing: serve_window.sh lacks the GLM53_LMCACHE_EXTRA_YAML hook (apply replicate/pd/serve_window_mooncake.patch)"; exit 2; }
  # 12's kit (program 2026-10-27T13:37:07Z): the WINDOW's interpreter + PYTHONPATH must import Mooncake, else LMCache
  # skips every lookup / store ('Please install mooncake') and D recomputes every prompt
  bash /opt/glm53/kit/doc/ttnn/12-fast-start/pd_galaxy/mc_role_import_check.sh \
    || { echo "refusing: the $ROLE window cannot import mooncake (12's mc_galaxy_up.sh installs venv-lmcache's zz_mooncake_pyshim.pth)"; exit 2; };;
esac
[ -n "${GLX_IMAGE_ROOT:-}" ] && export GLM53_TP_IMAGE_ROOT=$GLX_IMAGE_ROOT
# the role's own image (pd_config roles.<R>.image_pin / image_root via the env job's stage9_overlay.launch; coordinator
# 2026-10-28T01:51:52Z (b)): an explicit GLX_PIN wins, then the env job's pin, then the fleet pin; an env-job image_root wins
# over the block root host_glx.sh gives
[ -n "${EJ_IMAGE_ROOT:-}" ] && export GLM53_TP_IMAGE_ROOT=$EJ_IMAGE_ROOT
PIN=${GLX_PIN:-${EJ_IMAGE_PIN:-fde4697e51b2-tp-ds-v1-2x4-629a5c1a384e}}
# (2) / (6) a block whose own image root lacks this pin restores the default root's (blk1 shares blk0's harvest layout;
# its root may exist empty: program 2026-10-27T11:38:03Z / 11:41:23Z)
DEF_ROOT=/opt/glm53/data/dram_images/tp14b_int27
if [ "${GLM53_LAUNCH_WEIGHTS:-image}" = build ]; then
  # build mode (program 2026-10-28T04:37:44Z (2): the blk0 build was refused twice): tt/tp_image.py restores ANY finished image of
  # these weights / layout / mesh under the root before it builds, so a build needs an EMPTY root of its own -- the block's
  # GLX_IMAGE_ROOT, kept as given (no fallback to the shared root, which holds the 1530 image)
  [ -n "${GLX_IMAGE_ROOT:-}" ] || { echo "refusing: an image BUILD needs GLX_IMAGE_ROOT = an empty root of this block (e.g. .../dram_images/tp14b_int27_blk0)"; exit 2; }
  export GLM53_TP_IMAGE_ROOT=$GLX_IMAGE_ROOT
  echo "image BUILD root $GLM53_TP_IMAGE_ROOT (kept; no default-root fallback)"
elif [ ! -s "$GLM53_TP_IMAGE_ROOT/$PIN/manifest.json" ] && [ -s "$DEF_ROOT/$PIN/manifest.json" ]; then
  echo "image root $GLM53_TP_IMAGE_ROOT has no $PIN: the default root $DEF_ROOT"; export GLM53_TP_IMAGE_ROOT=$DEF_ROOT
fi
export GLM53_IMAGE_PIN=$PIN GLM53_TP_IMAGE_DIR=$GLM53_TP_IMAGE_ROOT/$PIN
export GLM53_IMAGE_MOCK_OK=${GLX_MOCK_OK:-$GLM53_TP_IMAGE_ROOT/$PIN/mock_verdict_glx_${TAG}pd${ROLE}.json}
# program 2026-10-27T10:00:40Z: the D-role mock verdict is WAIVED as the Galaxy launch gate (the construction is proven:
# pool 32,848 blocks = 2.1M tokens, DRAM min 217.7 MiB/bank, no Traceback; the role cannot serve the mock's fresh-prefill
# self-test); D's readiness = P's store + the first D request with tt.expect_hit = n (glx_pd_run.sh's smoke)
[ "$ROLE" = D ] && [ ! -s "$GLM53_IMAGE_MOCK_OK" ] && export GLM53_IMAGE_MOCK_WAIVER="program 2026-10-27T10:00:40Z (the D role's fresh-prefill self-test; construction proven on the host-p300 mock image_mock_int27n29h7p2hk4bpdD_0844)"
export GLM53_GRID_X=11 GLM53_PORT=${GLM53_PORT:-$((8100 + B))} GLM53_WORKLOAD=$PD/w_pd_role.sh GLX_PD_ROLE=$ROLE
export GLM53_RUN=resident_int27${TAG}pd${ROLE}_glxb${B}_$(date -u +%Y%m%dT%H%M%S)
export GLM53_CACHE_DIR=${GLX_CACHE_DIR:-/opt/glm53/tt-cache/glm53-flash}
export GLM53_FUSED_EXPERTS_CACHE=${GLX_FUSED_CACHE:-/opt/glm53/tt-cache/glm53-flash-fused-perm}
for f in /sys/bus/pci/drivers/tenstorrent/*/dma_mask_bits; do
  [ -e "$f" ] || continue
  [ "$(cat $f)" = 58 ] || { echo "refusing: $f = $(cat $f), not 58 (tt-kmd < 2.10: the 64-bit DMA-mask hazard)"; exit 76; }
done
REFM=${GLX_REF_MANIFEST:-/opt/glm53/kit/doc/ttnn/09-vllm/replicate/ref_1530493535ed_hashes.json}
# GLM53_LAUNCH_WEIGHTS=build (program 2026-10-28T04:02:52Z): no image exists yet on this block -- the server writes it (launch_
# resident.sh unsets the pin / dir / mock keys), so there is no manifest to check
if [ "${GLM53_LAUNCH_WEIGHTS:-image}" = build ]; then
  echo "image BUILD launch (GLM53_LAUNCH_WEIGHTS=build): the server writes the image under $GLM53_TP_IMAGE_ROOT; no content check"
else
  /usr/bin/python3 -I $S/image_ref_check.py $GLM53_TP_IMAGE_DIR/manifest.json $REFM --max-diff ${GLX_REF_MAX_DIFF:-16} \
    || { echo "refusing: the image content check against $REFM failed"; exit 76; }
fi
echo "$(date -u +%FT%TZ) PD role $ROLE on blk$B: tag $TAG, env job $EJ, plugin $GLM53_PLUGIN_SRC, port $GLM53_PORT, remote ${GLM53_LMCACHE_REMOTE_URL:-none}"
# the env check (program 2026-10-28T00:02:56Z (1)): every overlay key (incl. roles.<R>.extra_env) + every process-env key,
# read back from /proc/<pid>/environ of the API server and its EngineCore once they run -> <run>/env_check.json + one line
mkdir -p ${GLM53_WORK:-/opt/glm53/work-s9}/logs
setsid nohup /usr/bin/python3 -I $PD/pd_env_check.py "$EJ" "${GLM53_WORK:-/opt/glm53/work-s9}/runs/$GLM53_RUN" \
  > ${GLM53_WORK:-/opt/glm53/work-s9}/logs/env_check_$GLM53_RUN.log 2>&1 < /dev/null &
bash $S/launch_resident.sh; rc=$?
R=models/autoports/zai_org_glm_5_3_flash/readiness_vllm
git -C $GLM53_S9_TREE checkout -- $R 2>/dev/null || echo "readiness restore skipped in $GLM53_S9_TREE"
exit $rc
