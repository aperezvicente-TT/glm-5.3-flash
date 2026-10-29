# GLM-5.3-Flash on Tenstorrent Blackhole

Production serving of [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (321B parameters, 18B
active, KDA + MLA/DSA hybrid attention, 288-expert MoE) on Tenstorrent Blackhole: one tensor-parallel 8-chip server
per 2x4 tray running a persistent megakernel for decode and prefill, a vLLM 0.26 front end through the TT platform
plugin, LMCache for KV offload and prefix reuse at 1,048,576 tokens of context, and a front router that spreads
requests over several trays.

This repository is the entry point. It holds the deployment kit, the router, the weight converter and the two
documents you need:

- `BUILD-FROM-SOURCE.md`: how to build every component from the release branches, convert the weights, build the
  per-tray DRAM image and bring a fleet up.
- `GETTING-STARTED.md`: how to call the endpoint, what to expect, the capacity rules, how to restart a tray.

The code lives in three release branches, pinned in `VERSIONS`:

| Component | Repository | Branch / tag |
|---|---|---|
| tt-metal with the GLM-5.3-Flash model tree and the ttnn operations it adds | [aperezvicente-TT/tt-metal](https://github.com/aperezvicente-TT/tt-metal) | `glm53/release-0.1.0`, tag `glm53-flash-0.1.0` |
| vllm-tt-plugin with the model runner, the admission policy and the LMCache adapter | [aperezvicente-TT/vllm-tt-plugin](https://github.com/aperezvicente-TT/vllm-tt-plugin) | `glm53/release-0.1.0`, tag `glm53-flash-0.1.0` |
| LMCache with the TT connector and the capped cold tier | [aperezvicente-TT/LMCache](https://github.com/aperezvicente-TT/LMCache) | `glm53/release-0.1.0`, tag `glm53-flash-0.1.0` |

## Layout

```
BUILD-FROM-SOURCE.md          build, convert, image, bring-up
GETTING-STARTED.md            calling the endpoint, operating a fleet
VERSIONS                      the pinned commits of every component
env.sh                        the program environment (paths; edit for your host)
.agents/scripts/tt-lock       the device lock tool (one holder per chip group)
doc/ttnn/09-vllm/replicate/   the fleet kit: prepare, tree, image mock, block launch, dp4 up / stop, env jobs
doc/ttnn/09-vllm/scripts/     the server launcher, the serving window, the host mock, the chat template, checks
deploy/envjobs/               the per-tray serving configuration of record (release 0.1.0, four plain 1M trays)
router/                       the front router (dp4router.py v6f), its test and launcher
weights/                      the HF -> tile cache converter, its manifest and verifier
```

The kit keeps the directory names it was developed under (`doc/ttnn/09-vllm/...`) because the scripts reference
each other by those paths; set `GLM53_WS` to this checkout and they resolve.

## What is served (release 0.1.0)

Four identical trays, each a TP8 server with a 1,048,576-token KV pool, behind one OpenAI-compatible endpoint.
One user: first token in 0.20 s at 128 tokens of prompt, 6.4 s at 32K, 26 s at 131K, about 270 s at 1M cold;
decode 17 to 20 ms per token. Sixteen users: decode 29 to 35 ms per token. Stored context returns in 0.4 to 1.4 s
at 8K to 128K. The numbers and their conditions are in the release report that accompanies this release.

## License

tt-metal, vllm-tt-plugin and LMCache are Apache-2.0; the release branches carry their upstream licenses. The model
weights are distributed by their authors under the GLM-5.3-Flash model license and are not part of this release.
