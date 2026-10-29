#!/usr/bin/env bash
# Replication kit: verify a host matches the served cf3h-k3 stack before any device run (read-only; no device opened).
# Set the variables of env_k3.sh first (GLM53_TT_METAL, GLM53_TT_CXX = the C++ build tree, GLM53_PLUGIN_REPO,
# GLM53_LMCACHE_REPO, GLM53_VENV, GLM53_DATA). Prints one line per check and a final OK / N FAIL.
fail=0
chk() { if eval "$2" >/dev/null 2>&1; then echo "ok    $1"; else echo "FAIL  $1"; fail=$((fail + 1)); fi; }
sha() { git -C "$1" rev-parse --short=11 HEAD 2>/dev/null; }
chk "tt-metal (model/python/kernels) = 48d2fda005a"        '[ "$(sha $GLM53_TT_METAL)" = 48d2fda005a ]'
chk "tt-metal host C++ tree = 8f28c0fd0a8"                  '[ "$(sha ${GLM53_TT_CXX:-$GLM53_TT_METAL})" = 8f28c0fd0a8 ]'
chk "the C++ build has the page-DMA driver (_ttnn.so page_dma symbols)" 'nm -D --defined-only ${GLM53_TT_CXX:-$GLM53_TT_METAL}/build_Release/lib/_ttnn.so | grep -qiE "page_?dma"'
chk "the model tree's ttnn/ttnn/_ttnn.so IS the C++ build's (md5)" '[ "$(md5sum < $GLM53_TT_METAL/ttnn/ttnn/_ttnn.so)" = "$(md5sum < ${GLM53_TT_CXX:-$GLM53_TT_METAL}/build_Release/lib/_ttnn.so)" ]'
chk "plugin = 1be2e33"                                      '[ "$(git -C $GLM53_PLUGIN_REPO rev-parse --short=7 HEAD)" = 1be2e33 ]'
chk "LMCache fork = ef4689db"                               '[ "$(git -C $GLM53_LMCACHE_REPO rev-parse --short=8 HEAD)" = ef4689db ]'
chk "venv python 3.10"                                      '$GLM53_VENV/bin/python -c "import sys; assert sys.version_info[:2] == (3, 10)"'
chk "vllm == 0.26.0"                                        '$GLM53_VENV/bin/python -c "import vllm; assert vllm.__version__ == \"0.26.0\""'
chk "torch 2.11.0 (cpu)"                                    '$GLM53_VENV/bin/python -c "import torch; assert torch.__version__.startswith(\"2.11.0\")"'
chk "vllm_tt_plugin registered (entry point)"               '$GLM53_VENV/bin/python -c "from importlib.metadata import entry_points as e; assert any(x.name for x in e(group=\"vllm.platform_plugins\"))"'
chk "sfpi 7.86.0"                                           'grep -q "7.86.0" $GLM53_TT_METAL/tt_metal/sfpi-version'
chk ">= 32 configured 1 GiB hugepages (a k3 window held 24 of 32 on host-8chip; free now: $(cat /sys/kernel/mm/hugepages/hugepages-1048576kB/free_hugepages))" '[ $(cat /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages) -ge 32 ]'
chk "TP image manifest present (fde4697e51b2-tp-ds-v1-2x4-1530493535ed)" '[ -s $GLM53_DATA/dram_images/tp14b_int27/fde4697e51b2-tp-ds-v1-2x4-1530493535ed/manifest.json ]'
chk ">= 200 GB free under GLM53_DATA (KDA checkpoints skip below 80 GB)" '[ $(df -BG --output=avail $GLM53_DATA | tail -1 | tr -dc 0-9) -ge 200 ]'
[ $fail = 0 ] && echo OK || echo "$fail FAIL"
