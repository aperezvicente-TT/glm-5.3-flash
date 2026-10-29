# GLM-5.3-Flash release 0.1.0 — build from source

Everything the TP8 x DP4 deployment runs, from the release branches, plus the exact procedure that turns them
into a serving fleet on a fresh host. No container.

## 1. The sources

| Component | Repository, branch, tag | Commit | Over |
|---|---|---|---|
| tt-metal: the model tree `models/autoports/zai_org_glm_5_3_flash` (the TP8 persistent megakernels, the DRAM-image writer, the device kernels, the vLLM generator) and the ttnn / tt_metal changes it needs | `aperezvicente-TT/tt-metal`, branch `glm53/release-0.1.0`, tag `glm53-flash-0.1.0` | dfd0e5b0ac1 | public tt-metal main dc0b7bea9ad (2026-10-12) |
| vllm-tt-plugin: the model runner, the admission policy, the LMCache adapter, the self-test | `aperezvicente-TT/vllm-tt-plugin`, branch `glm53/release-0.1.0`, tag `glm53-flash-0.1.0` | 5107a3f | tenstorrent/vllm-tt-plugin a7230306 |
| LMCache: the TT connector, the capped fs cold tier | `aperezvicente-TT/LMCache`, branch `glm53/release-0.1.0`, tag `glm53-flash-0.1.0` | 93a3baa3 | LMCache/LMCache c9439c65 |
| this repository: the deployment kit, the router, the converter, the per-tray configuration | `aperezvicente-TT/glm-5.3-flash` | | |

Not in git (large, derived or third party):

- The weights: HuggingFace `zai-org/GLM-5.3-Flash`, snapshot eb9eb208eb0d988989d07a6a12d0fdeb5f52574a.
- The tile cache (bfp4 experts, bfp8 attention, 174 GB) and the fused-experts cache (107 GB): derived from the
  snapshot by the converter in this repository (section 4).
- The DRAM images (193 GB per harvest layout): built on the target's own chips (section 5).

## 2. Host requirements (as deployed)

| | the reference host (a 32-chip Blackhole box) |
|---|---|
| OS / Python | Ubuntu 22.04.5, Python 3.10 (the venv's interpreter 3.10.19, outside `/home`) |
| Compiler / build | clang-20, Ninja, CMake; sfpi is fetched by the tt-metal build |
| Tenstorrent | KMD `tenstorrent` 2.11.1-pre, tt-smi 5.3.0, 32 Blackhole chips in four 2x4 trays (8 chips per TP8 server) |
| Host memory | 566 GB RAM, 128 x 1 GiB hugepages (each TP8 server pins 8 for its image puller / page pool) |
| Disk | one dataset holding the checkouts, the build, the venv, the caches (281 GB), the images (193 GB per layout), the LMCache tiers; keep >= 250 GB free at all times |

## 3. Build

`R` is the root of the install (the Galaxy uses `/opt/glm53`; the kit's `ns_paths.sh` binds `$R/glm53` onto
the host-8chip path names the launchers use, so the directory names below are the ones they expect).

1. Clone the release branch of tt-metal with its submodules:

```
git clone -b glm53/release-0.1.0 https://github.com/aperezvicente-TT/tt-metal.git $R/tt-metal
git -C $R/tt-metal submodule update --init --recursive
```

   (The submodule shas of record: umd 538094128060, tt-cluster-descriptors 9bcf698e971a, tracy 9c3ca7e5fe48.)
   The branch is one commit over public tt-metal main dc0b7bea9ad; `git diff dc0b7bea9ad` shows everything it adds.

2. Build the host C++ once, from that checkout (the configure line of record):

```
cd $R/tt-metal
cmake -B build_Release -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$R/tt-metal/build_Release \
  -DCMAKE_TOOLCHAIN_FILE=cmake/x86_64-linux-clang-20-libstdcpp-toolchain.cmake \
  -DENABLE_TRACY=ON -DENABLE_DISTRIBUTED=ON -DTT_INSTALL=ON -DTT_UNITY_BUILDS=ON -DTT_ENABLE_LIGHT_METAL_TRACE=ON
cmake --build build_Release -j 32 && cmake --install build_Release && ln -sfn build_Release build
```

   40 to 60 minutes. Tests, programming examples and tt-train are off; Python bindings on.

3. Give the model tree the kit's expected name and point it at the build (the kit launches the server from a
   tree directory named after its form; the release branch is one tree, so a symlink is enough):

```
ln -sfn $R/tt-metal $R/tt-metal-s9cfk4d8t5wgxb
ln -sfn build_Release $R/tt-metal/build && cp $R/tt-metal/build_Release/lib/_ttnn.so $R/tt-metal/ttnn/ttnn/_ttnn.so
```

   The device kernels under the model tree are JIT-compiled at the first launch (a cold first launch adds
   ~15 minutes; the kit seeds the JIT cache from a host mock afterwards).

4. The serving venv (the plugin and LMCache are installed editable from their clones):

```
uv venv -p 3.10 $R/venv-lmcache && . $R/venv-lmcache/bin/activate
uv pip install vllm==0.26.0 torch==2.11.0 torchvision==0.26.0 --extra-index-url https://download.pytorch.org/whl/cpu
git clone -b glm53/release-0.1.0 https://github.com/aperezvicente-TT/vllm-tt-plugin.git $R/vllm-tt-plugin
git clone -b glm53/release-0.1.0 https://github.com/aperezvicente-TT/LMCache.git $R/LMCache-tt-s9cap
uv pip install -e $R/tt-metal -e $R/vllm-tt-plugin
NO_NATIVE_EXT=1 pip install -e $R/LMCache-tt-s9cap --no-build-isolation --no-deps
```

   Pin the rest to `VERSIONS`
   (transformers 5.16.1, numpy 1.26.4, tt-umd, tt-exalens; no torchaudio / torchcodec / nvidia-* packages).
   The kit's per-form prep (`glx_k4d7lg_tree_prep.sh`, `glx_lmcache_prep.sh`) sets the tree, plugin and LMCache
   directories a tray uses.

5. The kit is this repository:

```
git clone https://github.com/aperezvicente-TT/glm-5.3-flash.git $R/glm53-flash-bringup
mkdir -p $R/glm53/work-s12 && cp -r $R/glm53-flash-bringup/router $R/glm53/work-s12/dp4router && cp $R/glm53-flash-bringup/router/glx12_dp4router.sh $R/glm53/work-s12/
```

## 4. Weights and caches (host only, once)

1. Download the snapshot (`huggingface-cli download zai-org/GLM-5.3-Flash --revision eb9eb208eb0d988989d07a6a12d0fdeb5f52574a`).
2. Convert to the tile cache (bfp4 experts, bfp8 attention, the fused-K layouts; 26,051 tensors, 174 GB):

```
GLM53_HF_SNAPSHOT=<snapshot dir> GLM53_CACHE_DIR=$R/tt-cache/glm53-flash \
  python $R/glm53-flash-bringup/weights/convert_weights.py --workers 40 --threads 4
python $R/glm53-flash-bringup/weights/verify_cache.py   # against weights/manifest.json
```

   About 23 minutes on a 128-core host. The fused-experts cache (`GLM53_FUSED_EXPERTS_CACHE`, 107 GB) is derived
   from the tile cache by the model on its first load (the per-core `[gate_c | up_c]` tile permutation).

## 5. DRAM images (one device job per harvest layout)

The engine does not load safetensors at start. It DMAs a per-chip image (26 GB per chip, 8 per block) laid out
against the exact grid of the chips it was built on: which DRAM bank holds which expert, which core column
carries the MLA sync lane. The image id (for example `fde4697e51b2-tp-ds-v1-2x4-629a5c1a384e`) ends in the hash
of the block's chip descriptor. A block whose chips have a different harvest mask cannot use it.

- This box has two layouts: trays 1 and 2 share `629a5c1a384e`, trays 3 and 4 share `1530493535ed`.
- A new box builds its own: launch one block in build mode (`GLM53_LAUNCH_WEIGHTS=build`; the server finds no
  image under `GLM53_TP_IMAGE_ROOT`, loads the tile cache and writes the image, `tt/tp_image.py
  build_or_restore`), once per distinct layout, about 30 minutes each. Build mode never pulls an image.
- Then run the host-only image mock per block (`GLX_MOCK_IMAGE_ROOT=<root> GLX_TAG=<tag>dp4b<N> GLX_PD_ROLE=H
  bash glx_image_mock.sh <pin> <that block's step-1 descriptor> <port>`); its verdict file next to the image is
  what the fleet launcher checks. A block with no verdict is refused.

## 6. Bring the fleet up

Follow `GETTING-STARTED.md` section "Replicating the deployment": `glx_s9_prepare.sh` (layout + checks),
`check_versions.sh` (shas, venv, hugepages, image manifest), the per-form env jobs (`dp4/mk_dp4_envjobs.py`), the
H mock per block, `dp4/glx_dp4_up.sh '<GO line>'` (four standing TP8 servers on 8100..8103, ~12 min per block warm),
then the router:

```
bash $R/glm53/work-s12/glx12_dp4router.sh 8110 $R/glm53/logs/router \
  http://127.0.0.1:8100,http://127.0.0.1:8101,http://127.0.0.1:8102,http://127.0.0.1:8103 \
  --block-reserve-cap 0 --block-max-total-tokens 1048576,1048576,1048576,1048576 \
  --default-max-tokens 16384 --max-max-tokens 65536 --max-tokens-over clamp
```

The form of record (release 0.1.0, uniform plain fleet from 2026-10-29 ~06:00Z):

| Tray | Form | tt-metal | Plugin | LMCache | Keys |
|---|---|---|---|---|---|
| 0, 1, 2, 3 | f2i3d6 | tt-metal dfd0e5b0ac1 | plugin 5107a3f | LMCache 93a3baa3 | context 1,048,576; `GLM53_PREFILL_SHORT_FIRST=32768`; `GLM53_ADMIT_KV_BUDGET=1 GLM53_ADMIT_RESERVE_TOKENS=0` (no preemption) |

The exact per-tray environment is in `deploy/envjobs/f2i3d6_blk<N>_env.json` (the env job the kit generates and the
launcher reads); the key lists in prose are a summary, the job file is the record.

## 7. Verify

- `check_versions.sh` prints OK.
- The strict startup self-test per block passes before its socket binds (`glx_dp4_up.sh` waits for it).
- The cold ladder per block (`selftest_loop.py`, 128 / 2,048 / 32,000 / 131,072 tokens) matches the release
  numbers within noise (rolled form, 2026-10-29: 0.20 s at 128, 6.4 s at 32,000, 26.4 s at 131,072).
- `GET http://127.0.0.1:8110/health` lists all four blocks up.
