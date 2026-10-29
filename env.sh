# GLM-5.3-Flash bring-up environment. Source from bash on host-p300 or host-8chip.
export GLM53_WS=/opt/glm53/kit
export GLM53_HF_SNAPSHOT=/opt/glm53/hf/models--zai-org--GLM-5.3-Flash/snapshots/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a
export GLM53_MODEL_DIR=models/autoports/zai_org_glm_5_3_flash   # relative to TT_METAL_HOME
# Local-disk checkouts (host-8chip). On host-p300 override TT_METAL_HOME to your own checkout.
export GLM53_LOCAL=/opt/glm53
export TT_METAL_HOME=${TT_METAL_HOME:-$GLM53_LOCAL/tt-metal}
export TTLANG_HOME=${TTLANG_HOME:-$GLM53_LOCAL/tt-lang-ops-and-models}
export GLM53_CACHE_DIR=${GLM53_CACHE_DIR:-/opt/glm53/tt-cache/glm53-flash}   # bfp4/bfp8 tile cache on the shared mpi-shfs volume (user decision 2026-10-15)
export GLM53_GOLDENS=$GLM53_WS/goldens
export GLM53_EVIDENCE=$GLM53_WS/doc
export TT_LOCK_DIR=/opt/glm53/home/.tt-locks
export PYTHONPATH=$TT_METAL_HOME:${PYTHONPATH:-}
# Activate the tt-metal python env the way the user's scripts do (LD_LIBRARY_PATH=build/install/lib, python_env).
# Usage: glm53_activate_metal [tt-metal dir]
glm53_activate_metal() { local d=${1:-$TT_METAL_HOME}; pushd "$d" >/dev/null && { set +u; source ./env_vars_setup.sh; }; popd >/dev/null; }
export GLM53_REF_VENV=/opt/glm53/venv-ref   # torch-cpu + transformers>=5.16 (glm5_next) on host-8chip

# Kernel JIT: cross-tree compile reuse via ccache (operator ask 2026-10-19 18:05Z). Active only once ccache is installed
# (`sudo apt install ccache` on host-8chip). tt-metal keys TT_METAL_CACHE by tree hash, so sibling trees recompile the
# same kernels; ccache keys by preprocessed source and hits across trees. One shared store on local disk.
# host-8chip has ccache 4.5.1, but tt-metal invokes "ccache sloppiness=pch_defines,time_macros <compiler>" (>= 4.8 syntax):
# /opt/glm53/bin/ccache is a shim that turns leading KEY=VALUE args into CCACHE_* env and execs /usr/bin/ccache
# (tested 18:25Z: probe OK, second identical compile = hit). Without the shim the first JIT of every process fails.
if [ -x /opt/glm53/bin/ccache ] && command -v /usr/bin/ccache >/dev/null 2>&1; then
  export PATH=/opt/glm53/bin:$PATH
  export TT_METAL_CCACHE_KERNEL_SUPPORT=1 CCACHE_DIR=/opt/glm53/ccache CCACHE_MAXSIZE=40G CCACHE_SLOPPINESS=pch_defines,time_macros
fi

# Mock runs must not write into the physical chips' /dev/shm memory-tracking regions (tt-mgmt sums per-process slots per
# asic; a 2x4 full-model mock reported 24 GiB on every real chip, 2026-10-20 14:20Z). Launchers that set GLM53_MOCK=1
# before sourcing get it automatically; others export TT_METAL_SHM_TRACKING_DISABLED=1 themselves.
[ "${GLM53_MOCK:-0}" = "1" ] && export TT_METAL_SHM_TRACKING_DISABLED=1
