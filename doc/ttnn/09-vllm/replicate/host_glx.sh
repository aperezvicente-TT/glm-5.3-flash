# Host profile for the Galaxy (program 2026-10-27T00:08:40Z; source it, then run a row through glx_block_run.sh).
# Everything lives under /opt/glm53; inside ns_paths.sh the host-8chip paths the scripts name map onto it.
export GLX_ROOT=${GLX_ROOT:-/opt/glm53}
export R_GLM53=$GLX_ROOT/glm53 R_WS=$GLX_ROOT/glm53-flash-bringup R_DATA=$GLX_ROOT/data
export TT_LOCK_DIR=$GLX_ROOT/locks
# inside the namespace the workspace sits at the host-8chip path, so the lock tool is named by that path
export GLM53_TT_LOCK=/opt/glm53/kit/doc/ttnn/09-vllm/replicate/tt-lock-galaxy
# the HF hub holding models--zai-org--GLM-5.3-Flash (vLLM loads the config / tokenizer offline from it): GLX_HF_HUB_CACHE,
# else the first such hub dir found under $GLX_ROOT (depth <= 3); scripts/env.sh keeps a caller's HF_HUB_CACHE
if [ -z "${HF_HUB_CACHE:-}" ]; then
  [ -z "${GLX_HF_HUB_CACHE:-}" ] && [ -d /models/hf/hub/models--zai-org--GLM-5.3-Flash ] && GLX_HF_HUB_CACHE=/models/hf/hub  # the Galaxy's hub (01:01Z)
  _hf=${GLX_HF_HUB_CACHE:-$(ls -d $GLX_ROOT/models--zai-org--GLM-5.3-Flash $GLX_ROOT/*/models--zai-org--GLM-5.3-Flash \
        $GLX_ROOT/*/*/models--zai-org--GLM-5.3-Flash 2>/dev/null | head -1 | xargs -r dirname)}
  if [ -n "$_hf" ]; then export HF_HUB_CACHE=$_hf; else echo "host_glx.sh: no HF hub with models--zai-org--GLM-5.3-Flash under $GLX_ROOT (set GLX_HF_HUB_CACHE)" >&2; fi
fi
# the venv, activated here as well (01:36:03Z; scripts/env.sh sets the same from GLM53_VENV inside the launch)
export VIRTUAL_ENV=${GLX_VENV:-$R_GLM53/venv-lmcache}; export PATH=$VIRTUAL_ENV/bin:$PATH
# CPU sets per block (four blocks share the host; host-8chip's 0-95 / 128-159 may not exist here): the block's quarter of
# the CPUs, 3/4 for the server and 1/4 for the client cells (GLX_SERVER_CPUS / GLX_CLIENT_CPUS override). N = the block
# Per-block work dir (program 2026-10-27T04:01:01Z (2): parallel blocks collided on $GLM53_WORK/serve_window.lock):
# blocks 1-3 run under /opt/glm53/work-s9-blk<N> (namespace path; GLX_WORK = its host path) with a copy of
# work-s9's envjobs and chat template, the JIT cache linked to work-s9's, and port 8100+N; block 0 keeps work-s9
if [ -n "${N:-}" ] && [ "$N" != 0 ]; then
  export GLX_WORK=$R_GLM53/work-s9-blk$N GLM53_WORK=/opt/glm53/work-s9-blk$N GLM53_PORT=${GLM53_PORT:-$((8100 + N))}
  mkdir -p "$GLX_WORK/runs" "$GLX_WORK/logs" "$GLX_WORK/envjobs"
  cp -a "$R_GLM53/work-s9/envjobs/." "$GLX_WORK/envjobs/" 2>/dev/null
  [ -f "$R_GLM53/work-s9/chat_template_thinking_switch.jinja" ] && cp -a "$R_GLM53/work-s9/chat_template_thinking_switch.jinja" "$GLX_WORK/"
  [ -e "$GLX_WORK/tt-metal-cache" ] || ln -s ../work-s9/tt-metal-cache "$GLX_WORK/tt-metal-cache"
  # the block's own image root (04:47:11Z (a)): namespace path; its host dir under $R_DATA
  export GLX_IMAGE_ROOT=${GLX_IMAGE_ROOT:-/opt/glm53/data/dram_images/tp14b_int27_blk$N}
  mkdir -p "$R_DATA/dram_images/tp14b_int27_blk$N"
elif [ -n "${N:-}" ]; then
  export GLX_WORK=$R_GLM53/work-s9
fi
if [ -n "${N:-}" ]; then
  _n=$(nproc --all); _q=$((_n / 4)); _b=$((N * _q)); _s=$((_q * 3 / 4))
  export GLM53_SERVER_CPUS=${GLX_SERVER_CPUS:-$_b-$((_b + _s - 1))} GLM53_CLIENT_CPUS=${GLX_CLIENT_CPUS:-$((_b + _s))-$((_b + _q - 1))}
fi
