#!/usr/bin/env bash
# Stage 9's layout on the Galaxy for the namespace launch (program 2026-10-27T01:03:02Z; HOST ONLY, run once before
# the first light, as the launching user). ns_paths.sh binds $GLX_ROOT/glm53 onto /opt/glm53 and the workspace /
# data dirs onto the share paths, so the host-8chip names the k3 target uses must exist there:
#  - srv/s9h8w -> the kit's vllm-tt-plugin (1be2e33 = srv/s9h8w's HEAD on host-8chip);
#  - LMCache-tt-s9k3 -> the kit's LMCache (ef4689db = the target's GLM53_LMCACHE_TREE);
#  - the per-host files the launcher's pre-mesh check and the server read (the bundle, from the share:
#    glm53-flash-bringup-data/galaxy/s9_blk0_bundle.tar): our env-job overlay, 14b's base env job and gate rows t01_widths /
#    t12_idle, the chat template, the 1M cell plan, and the workspace's teacher-forcing goldens (the client cells' prompts);
#  - empty dirs: run / log dirs, the DSA slice sidecar (written by the first attach), the JIT cache (cold), TMPDIR, the
#    LMCache disk tier.
# usage: glx_s9_prepare.sh <bundle tar>
set -eu
G=${GLX_ROOT:-/opt/glm53}; L=$G/glm53; B=${1:?bundle tar (s9_blk0_bundle.tar)}
mkdir -p $L/srv
[ -e $L/srv/s9h8w ] || ln -s ../vllm-tt-plugin $L/srv/s9h8w
[ -e $L/LMCache-tt-s9k3 ] || ln -s LMCache $L/LMCache-tt-s9k3
[ -e $L/venv-vllm ] || ln -s venv-lmcache $L/venv-vllm  # the client cells' interpreter (scripts/workload_lib.sh PY)
mkdir -p $L/work-s9/runs $L/work-s9/logs $L/work-s9/envjobs $L/work-s9/dsa_slices $L/work-s9/tt-metal-cache \
         $L/work-s14b/q_mesh $L/work-s14b/q_t28p $L/tz $L/lmcache_nvme $G/data/dram_images/tp14b_int27 $G/locks
tar -C $G --keep-directory-symlink -xf $B
bad=0
for p in tt-metal-s9cfk3/models/autoports/zai_org_glm_5_3_flash/tt/generator.py tt-metal-s9cfk3/build_Release/lib \
         venv-lmcache/bin/python srv/s9h8w/src/vllm_tt_plugin LMCache-tt-s9k3/lmcache \
         work-s9/envjobs/int27n29h7p2hk3_img_env.done work-s14b/q_t28p/p00_env.hold work-s14b/q_mesh/t01_widths.out \
         work-s14b/q_mesh/t12_idle.out work-s9/chat_template_thinking_switch.jinja; do
  if [ -e "$L/$p" ]; then echo "ok      glm53/$p"; else echo "MISSING glm53/$p"; bad=1; fi
done
for p in glm53-flash-bringup/doc/ttnn/09-vllm/scripts/targets/glx_firstlight_k3.sh \
         glm53-flash-bringup/goldens_gpu_fp32/teacher_forcing/32768.pt tt-cache/glm53-flash tt-cache/glm53-flash-fused-perm; do
  if [ -e "$G/$p" ]; then echo "ok      $p"; else echo "MISSING $p"; bad=1; fi
done
grep -q 'GLM53_LAUNCH_WEIGHTS:-image}" = build' $G/glm53-flash-bringup/doc/ttnn/09-vllm/scripts/launch_resident.sh \
  && echo "ok      launch_resident.sh has the build mode" || { echo "MISSING the build mode: update the workspace checkout"; bad=1; }
( unset HF_HUB_CACHE; . "$(dirname "$(readlink -f "$0")")/host_glx.sh" 2>&1; if [ -n "${HF_HUB_CACHE:-}" ] && [ -d "$HF_HUB_CACHE/models--zai-org--GLM-5.3-Flash/snapshots" ]; then echo "ok      HF hub $HF_HUB_CACHE ($(ls $HF_HUB_CACHE/models--zai-org--GLM-5.3-Flash/snapshots | head -1))"; else echo "MISSING the HF hub (models--zai-org--GLM-5.3-Flash/snapshots): set GLX_HF_HUB_CACHE"; exit 1; fi ) || bad=1
# the venv rules the first light found (program 2026-10-27T02:01:46Z): an interpreter outside /home (uv-python under
# $G), CPU torchvision 0.26.0+cpu, no torchaudio / torchcodec / nvidia-* (host-8chip's venv has none), setuptools 79.0.1,
# wheel 0.48.0
V=$L/venv-lmcache; py=$(readlink -f $V/bin/python 2>/dev/null)
case "$py" in /home/*|"") echo "MISSING a venv interpreter outside /home ($py)"; bad=1;; *) echo "ok      venv interpreter $py";; esac
pk=$($V/bin/python -m pip list --format=freeze 2>/dev/null)
for want in 'torchvision==0.26.0+cpu' 'setuptools==79.0.1' 'wheel==0.48.0' 'vllm==0.26.0'; do  # (vllm: 0.26.0 or a local 0.26.0+<tag> build)
  echo "$pk" | grep -qixE "$(echo "$want" | sed 's/[.+]/\\&/g')(\\+[a-z0-9.]+)?" && echo "ok      $want" || { echo "MISSING $want ($(echo "$pk" | grep -i "^${want%%==*}==" | head -1))"; bad=1; }
done
extra=$(echo "$pk" | grep -iE '^(torchaudio|torchcodec|nvidia-[a-z0-9-]+)==' | tr '\n' ' ')
[ -z "$extra" ] && echo "ok      no torchaudio / torchcodec / nvidia-*" || { echo "EXTRA   remove: $extra"; bad=1; }
df -h $G/data | tail -1 | awk '{print "image root free: " $4 " (the k3 image: 8 chip files, 198 GB on host-8chip)"}'
exit $bad
