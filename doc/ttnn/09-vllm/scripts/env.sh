# Stage 9 (vLLM on device) environment. Source from bash. Worktree /opt/glm53/tt-metal-s9 (branch
# glm53/glm53-flash-s9) with a copy of Stage 6's runtime (= Stage 3/4 C++), the vLLM venv, the 2x4 mesh.
source /opt/glm53/kit/env.sh
export TT_METAL_HOME=${GLM53_S9_TREE:-/opt/glm53/tt-metal-s9}  # GLM53_S9_TREE: the run tree (s9i8: int8 + Stage 9, C++ = s12m5)
export TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME
# GLM53_VENV: another venv for the server (LMCache rungs: venv-lmcache = vLLM 0.26.0 + LMCache-tt, same torch / python)
export VIRTUAL_ENV=${GLM53_VENV:-/opt/glm53/venv-vllm}; export PATH=$VIRTUAL_ENV/bin:$PATH
export GLM53_PLUGIN_RUNTIME=/opt/glm53/kit/vendor/tenstorrent-skills/plugins/tt-model-bringup/runtime
# ttnn and the model package from the Stage 9 tree (ahead of the venv's python_env .pth entries)
export PYTHONPATH=$TT_METAL_HOME/ttnn:$TT_METAL_HOME:$TT_METAL_HOME/tools:$GLM53_PLUGIN_RUNTIME
# GLM53_PLUGIN_SRC: a vllm-tt-plugin checkout's src/ ahead of the venv's installed plugin (e.g. the LMCache join)
[ -n "${GLM53_PLUGIN_SRC:-}" ] && export PYTHONPATH=$GLM53_PLUGIN_SRC:$PYTHONPATH
export TT_METAL_INSTALL_DIR=$TT_METAL_HOME/build/install
export LD_LIBRARY_PATH=$TT_METAL_HOME/build/lib:${LD_LIBRARY_PATH:-}
export TT_METAL_CACHE=${TT_METAL_CACHE:-/opt/glm53/work-s9/tt-metal-cache}
export TT_METAL_OPERATION_TIMEOUT_SECONDS=${TT_METAL_OPERATION_TIMEOUT_SECONDS:-300}
export GLM53_FUSED_EXPERTS_CACHE=${GLM53_FUSED_EXPERTS_CACHE:-/opt/glm53/tt-cache/glm53-flash-fused-perm}
export GLM53_WORK=${GLM53_WORK:-/opt/glm53/work-s9}  # a Galaxy block's own work dir (replicate/host_glx.sh, program 2026-10-27T04:01:01Z (2))
export GLM53_E9=/opt/glm53/kit/doc/ttnn/09-vllm
export GLM53_CODE_SHA=$(git -C $TT_METAL_HOME rev-parse --short HEAD)$( [ -n "$(git -C $TT_METAL_HOME status --porcelain -- models/autoports/zai_org_glm_5_3_flash)" ] && echo -dirty)
# vLLM + plugin
export VLLM_TT_PLUGIN_ROOT=/opt/glm53/vllm-tt-plugin
# HF_HUB_CACHE: a host profile may set it (the Galaxy: replicate/host_glx.sh; the namespace maps only /opt/glm53 and
# the two share roots, not /opt/glm53/home/mpi-shfs); host-8chip keeps its hub
export HF_HUB_CACHE=${HF_HUB_CACHE:-/opt/glm53/hf} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
# the 8-chip engine needs ~5-20 min to build + warm up; its teardown ~20 s: no vLLM timeout may fire (a timed-out
# startup or a shutdown past VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS ends in SIGKILL = unclean device death)
export VLLM_RPC_TIMEOUT=1800000 VLLM_ENGINE_READY_TIMEOUT_S=7200 VLLM_ENGINE_ITERATION_TIMEOUT_S=900
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=900 VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS=900
# the physical 2x4 Ethernet mesh + its graph descriptor (the adapter declares the fabric, checks the descriptor)
export MESH_DEVICE="(2, 4)"
export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/models/autoports/zai_org_glm_5_3_flash/mesh/glm8_2x4_mesh_graph_descriptor.textproto
# program 2026-10-19 18:00-18:25Z: shared kernel ccache across trees (shim for tt-metal's >= 4.8 ccache syntax, same
# block as the workspace env.sh) and per-chip parallel warm-up / capture for every 8-chip warm-up (Stage 12 fast_start
# options; effective when the generator comes from tt/fast_start.py, i.e. GLM53_WEIGHT_SOURCE is set)
if [ -x /opt/glm53/bin/ccache ] && command -v /usr/bin/ccache >/dev/null 2>&1; then
  export PATH=/opt/glm53/bin:$PATH
  export TT_METAL_CCACHE_KERNEL_SUPPORT=1 CCACHE_DIR=/opt/glm53/ccache CCACHE_MAXSIZE=40G CCACHE_SLOPPINESS=pch_defines,time_macros
fi
export GLM53_PARALLEL_WARMUP=${GLM53_PARALLEL_WARMUP:-1} GLM53_PARALLEL_CAPTURE=${GLM53_PARALLEL_CAPTURE:-1}
