# GLM-5.3-Flash on Blackhole Galaxy (TP8 x DP4): getting started

OpenAI-compatible serving of zai-org/GLM-5.3-Flash (321B, 18B active) on one Blackhole Galaxy (32 chips) as four independent TP8 servers, one per 2x4 tray, behind one router. Release 0.1.0 (2026-10-29). Report: `RELEASE-0.1.0-draft.md`; operations: `OPERATING.md`.

## The endpoint

| What | Where |
|---|---|
| Router (the one endpoint) | `http://127.0.0.1:8110` on the box, localhost only, no auth |
| Blocks | `:8100` .. `:8103` = blk0 .. blk3; direct calls bypass the router's rules |
| Model id | `zai-org/GLM-5.3-Flash` |
| Context | 1,048,576 tokens on every block (uniform plain fleet since 2026-10-29) |
| Concurrency | 8 decode slots per block, 32 fleet-wide; one resident 1M request per block |

## What to expect

| | one user, cold | returning user (stored context) | 16 users |
|---|---|---|---|
| 128 tokens | 0.20 s to first token, 58 tok/s | | 0.35 s P50, 34 tok/s each |
| 8K | 2.3 s, 52 tok/s | 0.42 s | 7.0 s, 32 each |
| 32K | 5.3 s, 52 | 0.61 s | 16 s, 29 each |
| 128K | 27 s, 51 | 1.43 s | 19 s P50, 31 each |
| 1M | 262 to 286 s, 52 | 8.8 s | one per block; the 4th waits |

Decode is 17 to 20 ms per token for one user at every context length and 29 to 35 ms at 16 users. The MTP speculative form (x1.4 to x1.6 for 1 to 2 users, short prompts, 131K pool) was measured on one block and is reported as an option; the deployment runs the plain form on all four blocks. Full tables and caveats: the release report.

## Capacity rules

- A 1M request takes a block's whole KV pool for its 4.5-minute prefill; concurrent 1M users need one block each, at most 4, and prompts over 131K run on at most 3 blocks at once (the 4th waits in the router).
- Contexts are stored in LMCache tiers (host memory, NVMe, disk) when a request finishes; a repeated prefix is a hit (the "returning user" column). The disk tier grows under long-context load; a pruner holds the box above 150 GB free.
- Keep `max_tokens x concurrent users` within a block's 1M pool when calling a block directly; through the router the caps are enforced (default max_tokens 16,384, ceiling 65,536, clamped).

## Replicating the deployment (the box kit)

Everything lives on the box under `/opt/glm53` (`R`): the kit checkout `R/glm53-flash-bringup` (git, fast-forwarded by bundles only), the serving trees and plugins under `R/glm53/`, the DRAM images under `R/data/dram_images/`.

1. Fast-forward the kit to the release's commit (bundles in `galaxy/`), untar the form package `galaxy/rows/dp4_<form>_v<N>.tar` under `R`; it carries one env job and one target per block, the plugin bundle and the tree bundle.
2. Prepare the plugin and the tree outside the serving namespace: `glx_plugin_prep.sh <bundle> <branch> <sha> <srv>` and `glx_k4d7lg_tree_prep.sh <bundle> <branch> <sha> <tree> <dir>` (the package's result line names them).
3. Run the host-only image mock per block (`GLX_MOCK_IMAGE_ROOT=<root> GLX_TAG=<tag>dp4b<N> GLX_PD_ROLE=H bash glx_image_mock.sh <pin> <step1 descriptor> <port>`); its verdict file is what the launcher checks. blk0 / blk1 use the `tp14b_int27` root with the 629a pin, blk2 / blk3 the `tp14b_int27_blk3` root with the 1530 pin (the pins carry each tray's harvest layout).
4. Bring the trays up: `GLX_DP4_FORM=<form> dp4/glx_dp4_up.sh '<GO line>'` (all four, or `GLX_DP4_BLOCKS=N` for one). Healthy in about 12 minutes per block with a warm kernel cache (26 cold); the launcher runs each block's self-test and ladder.
5. Start the router: `glm53/work-s12/glx12_dp4router.sh 8110 <dir> http://127.0.0.1:8100,...,:8103 [--replace http://127.0.0.1:8110] <flags>`; check `GET /health` lists all four blocks `up`.
6. Stop in order with `GLX_DP4_FORM=<form> dp4/glx_dp4_stop.sh` (a STOP file per tray; no signals, no resets).

## When something is wrong

- `GET http://127.0.0.1:8110/health`: `up`, `draining`, per-block `inflight` and `kv_usage`. Each block's `/metrics` has `vllm:num_requests_running` and the `glm53_admit_*_tokens` gauges.
- Logs: the block's run dir `R/glm53/work-s9[-blkN]/runs/resident_<tag>dp4b<N>pdH_glxb<N>_<UTC>/server.log` (and `events.jsonl`); the router's log beside its launcher.
- Restart one block: `POST /router/drain?block=N&wait_s=600`, then `glx_dp4_stop.sh` for that block, then `glx_dp4_up.sh` for it, then `POST /router/undrain?block=N`. The fleet serves on the other three meanwhile.
- A block whose engine exited cleanly (handles closed) is relaunched directly. A block left with device nodes open needs a chip reset, which on this shared box requires the whole fleet stopped and is the operator's call.
- Known issues and their fix status: the release report, section 3.
