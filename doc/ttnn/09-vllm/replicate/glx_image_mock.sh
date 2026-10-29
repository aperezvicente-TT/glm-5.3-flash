#!/usr/bin/env bash
# The Galaxy W5 image mock (program 2026-10-27T02:44:50Z), run ON the Galaxy, host-only: GLM53_MOCK=1 with the block's
# cluster descriptor (its harvest masks place the bank-adjacent L1 tables as the image recorded), no device opened. It is
# the host-8chip image mock (scripts/mock_image_serve.sh) with the k3 env job UNCHANGED (the W5 launch uses that env job, so
# the verdict's job_env must equal it), the model tree at the host-8chip path (the Galaxy's gridpin checkout), GLM53_GRID_X=11,
# the plugin / LMCache / venv at their host-8chip paths (ns_paths.sh), and the pinned image under the data root.
# The verdict: the run dir, work-s9/mock_verdict_int27n29h7p2hk3.json on the Galaxy, and next to the image
# (mock_verdict_glx.json, which the W5 target names as GLM53_IMAGE_MOCK_OK).
# GLX_TAG (2026-10-27 03:4xZ, the W5 'spec' variant, re 03:37:57Z): another served tag on the same image, e.g. n29h7p2hk3s3
# (the MTP K=3 twin): its target's sizing + spec keys (max_model_len / seats / pool / GLM53_SPEC_*), its env job, its own
# run name and verdict file (mock_verdict_glx_<tag>.json; k3 keeps mock_verdict_glx.json). The env job must be on the
# Galaxy (work-s9/envjobs/int27<tag>_img_env.done, in the kit bundle).
# GLX_PD_ROLE=P|D (the PD kit, program 2026-10-27T10:00:40Z): a mock of THAT role's launch (pd/glx_pd_role.sh): the
# role overlay env job work-s9/envjobs/int27<tag>pd<role>_img_env.done with its overlay keys exported, the k4 line's tree
# (GLX_PD_TREE, default tt-metal-s9cfk4b), 12's PD plugin (GLX_PD_PLUGIN_SRC) and LMCache (GLX_PD_LMCACHE_TREE); the verdict
# mock_verdict_glx_<tag>pd<role>.json next to the image = glx_pd_role.sh's default GLX_MOCK_OK. (D's is waived as its launch
# gate, 10:00:40Z: the D role cannot serve the mock's fresh-prefill self-test.)
# usage: [GLX_TAG=<tag>] [GLX_PD_ROLE=P|D] glx_image_mock.sh <pin> <cluster descriptor yaml> [port]
set -u
PIN=${1:?image pin}; DESC=$(readlink -f "${2:?cluster descriptor yaml}"); PORT=${3:-18071}
K=$(dirname "$(readlink -f "$0")"); . "$K/host_glx.sh"
[ -s "$DESC" ] || { echo "no cluster descriptor $DESC"; exit 2; }
# GLX_MOCK_IMAGE_ROOT (namespace path; program 2026-10-28T00:28:36Z, the k4d7lg arms): the image root of the mocked pin,
# e.g. a block's own root .../dram_images/tp14b_int27_blk3 (its fde4697e51b2-tp-ds-v1-2x4-1530493535ed = the host-8chip k4-line
# map); default the shared tp14b_int27. The verdict lands next to that image = glx_pd_role.sh's default GLX_MOCK_OK there.
NSD=/opt/glm53/data
IROOT=${GLX_MOCK_IMAGE_ROOT:-$NSD/dram_images/tp14b_int27}; HROOT=$R_DATA${IROOT#$NSD}
[ -f "$HROOT/$PIN/manifest.json" ] || { echo "no image $HROOT/$PIN"; exit 2; }
S=/opt/glm53/kit/doc/ttnn/09-vllm/scripts
NTD=/opt/glm53/kit/.agents/scripts/no-tt-devices
TAG=${GLX_TAG:-n29h7p2hk3}; VNAME=mock_verdict_glx.json; [ $TAG = n29h7p2hk3 ] || VNAME=mock_verdict_glx_$TAG.json
ROLE=${GLX_PD_ROLE:-}; case "$ROLE" in ""|P|D|H) ;; *) echo "GLX_PD_ROLE=$ROLE (P, D or H: the fleet's HYBRID tray)"; exit 2;; esac
[ -n "$ROLE" ] && VNAME=mock_verdict_glx_${TAG}pd$ROLE.json
# the target is read from the host checkout ($R_WS): this check runs OUTSIDE the namespace, where the share mount does not exist
# (program 2026-10-27T04:01:01Z (1))
TH=$R_WS/doc/ttnn/09-vllm/scripts/targets/int27${TAG}_resident.sh
[ -f $TH ] || { echo "no target $TH"; exit 2; }
EJN=int27${TAG}$([ -n "$ROLE" ] && echo pd$ROLE)_img_env.done
[ -f "$R_GLM53/work-s9/envjobs/$EJN" ] || { echo "no env job $R_GLM53/work-s9/envjobs/$EJN on this host"; exit 2; }
# program [program] 2026-10-28T14:59:19Z: the f2i blk1 H mock (--port 18311) also bound *:8102 (vLLM's get_open_port() ZMQ
# sockets start at $VLLM_PORT / an ephemeral port) and the drained blk2 tray could not rebind: pin them to this mock's port + 1000
export VLLM_PORT=${GLX_MOCK_VLLM_PORT:-$((PORT + 1000))}
export GLM53_RUN=image_mock_glx_$([ $TAG = n29h7p2hk3 ] || echo ${TAG}$([ -n "$ROLE" ] && echo pd$ROLE)_)${PIN##*-}_$(date -u +%H%M) GLM53_TARGET=int27$TAG GLM53_IMAGE_PIN=$PIN
# GLM53_MOCK_IMAGE_ROOT: mock_image_serve.sh re-points the env job's GLM53_TP_IMAGE_DIR / ROOT (imported with the GLM53_TP_ keys)
# to this root AFTER the import (program 2026-10-28T00:37:53Z: without it the engine kept the env job's shared root)
export GLM53_TP_IMAGE_ROOT=$IROOT GLM53_MOCK_IMAGE_ROOT=$IROOT
export GLM53_S9_TREE=/opt/glm53/tt-metal-s9cfk3 GLM53_14B_ENV_JOB=p00_env
export GLM53_14B_ENV_FILE=/opt/glm53/work-s9/envjobs/int27${TAG}_img_env.done
export GLM53_PLUGIN_SRC=/opt/glm53/srv/s9h8w/src GLM53_LMCACHE_TREE=/opt/glm53/LMCache-tt-s9k3
export GLM53_VENV=/opt/glm53/venv-lmcache GLM53_SELFTEST=off
export GLM53_SELFTEST_OUT=/opt/glm53/work-s9/mockruns/$GLM53_RUN.selftest.json
export GLM53_DECODE_PROBE_EVERY=64 GLM53_VLLM_KV_POOL_TOKENS=1048576
# another tag: its target's construction keys (the pool, max_model_len, seats, the spec pair) over the k3 defaults
[ $TAG = n29h7p2hk3 ] || eval "$(grep -E '^export GLM53_(MAX_MODEL_LEN|MAX_SEQS|SPEC_[A-Z_]+|VLLM_KV_POOL_TOKENS)=' $TH)"
# a PD role: its launch's tree / plugin / LMCache and env job, and every key of its overlay (as pd/glx_pd_role.sh exports them)
if [ -n "$ROLE" ]; then
  # the launch paths default to the composed env job's (stage9_overlay.launch), as pd/glx_pd_role.sh reads them, so the role
  # mock and the role run match (program 2026-10-27T11:41:23Z (5)); GLX_PD_* override
  eval "$(/usr/bin/python3 -I -c 'import json, shlex, sys; l = (json.load(open(sys.argv[1])).get("stage9_overlay") or {}).get("launch") or {}
for k, v in (("EJ_PLUGIN", l.get("plugin_src")), ("EJ_LMCACHE", l.get("lmcache_tree")), ("EJ_TREE", l.get("tree"))):
    print(k + "=" + shlex.quote(v or str()))' "$R_GLM53/work-s9/envjobs/$EJN")"
  export GLM53_S9_TREE=${GLX_PD_TREE:-${EJ_TREE:-/opt/glm53/tt-metal-s9cfk4b}} GLM53_14B_ENV_FILE=/opt/glm53/work-s9/envjobs/$EJN
  export GLM53_PLUGIN_SRC=${GLX_PD_PLUGIN_SRC:-${EJ_PLUGIN:-/opt/glm53/srv/s12lmc_s9h8t/src}}
  LMT=${GLX_PD_LMCACHE_TREE:-$EJ_LMCACHE}; [ -n "$LMT" ] && export GLM53_LMCACHE_TREE=$LMT
  OV=$(/usr/bin/python3 -I - "$R_GLM53/work-s9/envjobs/$EJN" "$ROLE" <<'PY2'
import json, shlex, sys
o = json.load(open(sys.argv[1])).get("stage9_overlay") or {}
if o.get("role") != sys.argv[2]:
    sys.exit(f"echo 'refusing: {sys.argv[1]} is the {o.get('role')} overlay, not {sys.argv[2]}'; exit 2")
for k, v in (o.get("keys") or {}).items():
    print(f"export {k}={shlex.quote(str(v))}")
PY2
) || exit 2
  eval "$OV"
fi
export GLM53_MOCK_DURING_RUN=1 GLM53_MOCK_CAPTURE=1 GLM53_MOCK_DRAM_PROBE=1 GLM53_MOCK_DRAM_BALLAST_MIB=4 GLM53_MOCK_CONCURRENT_CELL=0
export GLM53_MOCK_PROFILE_CLIENT="$NTD /opt/glm53/venv-lmcache/bin/python -I $S/profile_client.py --url \$URL --lengths '' --hit 32768 --max-tokens 4"
export GLM53_MOCK_CLUSTER_DESC=$DESC GLM53_GRID_X=11 GLM53_PORT=$PORT GLM53_CPUS=0-$(( $(nproc --all) - 1 ))
export GLM53_CACHE_DIR=${GLX_CACHE_DIR:-/opt/glm53/tt-cache/glm53-flash}
export GLM53_FUSED_EXPERTS_CACHE=${GLX_FUSED_CACHE:-/opt/glm53/tt-cache/glm53-flash-fused-perm}
mkdir -p "$R_GLM53/work-s9/mockruns" "$R_GLM53/work-s9/logs"
echo "$(date -u +%FT%TZ) glx image mock $GLM53_RUN ($TAG): pin $PIN, descriptor $DESC, log $R_GLM53/work-s9/logs/$GLM53_RUN.log"
# program [session 10] 2026-10-28T16:47:35Z RULE (14b 16:46:58Z): no H / PD mock on a serving box may open a device node -- a PD
# H2D / D2H socket connector builds a raw UMD Cluster that opens EVERY visible chip non-power-aware (one ARC POWER_SETTING each) and
# the mock-mode sysinfo read opens every node: by default the mock runs under the no-tt-devices shim (tmpfs over /dev/tenstorrent
# inside the namespace: UMD sees 0 chips; a connector fails fast instead of poking a serving chip). GLX_MOCK_NTD=0 = the old open path.
MWRAP=; [ "${GLX_MOCK_NTD:-1}" = 1 ] && MWRAP=$NTD
echo "$(date -u +%FT%TZ) glx image mock: device nodes $([ -n "$MWRAP" ] && echo HIDDEN by no-tt-devices || echo "VISIBLE (GLX_MOCK_NTD=0)")"
bash "$K/ns_paths.sh" $MWRAP bash -c "cd /opt/glm53/work-s9 && bash $S/mock_image_serve.sh > /opt/glm53/work-s9/logs/\$GLM53_RUN.log 2>&1; rc=\$?
  v=/opt/glm53/work-s9/mockruns/\$GLM53_RUN/mock_verdict.json
  [ -s \$v ] && cp \$v \$GLM53_TP_IMAGE_ROOT/\$GLM53_IMAGE_PIN/$VNAME
  tail -4 /opt/glm53/work-s9/logs/\$GLM53_RUN.log; exit \$rc"
rc=$?
echo "$(date -u +%FT%TZ) glx image mock ($TAG) rc=$rc; verdict $HROOT/$PIN/$VNAME $( [ -s $HROOT/$PIN/$VNAME ] && echo written || echo MISSING)"
exit $rc
