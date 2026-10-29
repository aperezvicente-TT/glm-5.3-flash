12 dp4router v6f -- the front router for TP8 x DP4 on the Galaxy (program 2026-10-28T12:20:55Z; v6c 2026-10-29T00:12:19Z; v6d 01:17:33Z; v6e 01:56:07Z; v6f 02:45Z)

WHAT: one OpenAI-compatible endpoint (/v1/completions, /v1/chat/completions, /v1/models, /tokenize, /detokenize)
in front of the four block servers. It is a thin asyncio proxy (aiohttp) with streaming passthrough: the block's
bytes are forwarded as they arrive, and X-Request-Id is passed through. It is not vLLM's data-parallel front end,
which needs the four engines launched as DP ranks of one deployment; the TT plugin's DP is lane-DP or one mesh.

V3 (program 2026-10-28T13:27:02Z, load run 1: a block's prefill is FIFO; shorts behind a 1M waited 255-325 s):
  LONG CAP  prompts over --long-cap-tokens (131,072) prefill on at most --long-cap blocks at once (default N - 1 = 3
            of 4; 0 = no cap); a further long waits IN THE ROUTER, FIFO, until one of them reaches its first token
            (dp4router_long_queue_depth / _queued_total / _wait_seconds_sum)
  LONG-FREE every other prompt goes to a block WITHOUT a long prefill in flight when one is up (a session / prefix
            match on a block with a long prefill is passed over); only when none is free, the next rules on all blocks

V4 (program 2026-10-28T13:47:25Z / 13:48:04Z, load run 2: v3's cap did not engage):
  the cap counts a long at ADMISSION (dp4router_long_admitted; released at its first token / end); the prompt is
  classified (tokens, prefix keys) BEFORE admission; the block choice and its counters are taken with no await between
  them. The listener binds with SO_REUSEADDR + SO_REUSEPORT (a restart binds through TIME_WAIT at once).
  --replace <old router URL>: take the port over. A v4 old instance gets POST /router/retire (it stops listening at
  once, finishes its in-flight streams, exits); an older one is waited on until its in-flight streams are 0, then the
  bind is retried until the port frees (--replace-wait-s 600).

V6b (program 2026-10-28T18:10:22Z, the MIXED FLEET candidate: 3 x 1M f2i3 blocks + 1 x 131,072-cap spec block):
PER-BLOCK max_model_len.
  the cap   from each block's /v1/models (max_model_len, re-read every 30 s), or --block-max-model-len
            1048576,1048576,131072,1048576 (a comma list per block; 0 / empty = learn it); metric
            dp4router_block_max_model_len{block}.
  the rule  a block never takes a request whose prompt + min(max_tokens, 32,768) exceeds its cap (no max_tokens: the
            32,768 reserve cap counts), nor one whose prompt + an explicit max_tokens does (vLLM would refuse it there);
            counted in dp4router_ctx_steered_total. With no block that fits, the largest cap gets it.
  shorts    prompts <= --spec-prefer-tokens (32,768) PREFER the smaller-cap block: in the least-loaded rule it counts
            --spec-prefer-slack (4) requests less load, so shorts go to the spec block until it carries 4 more than the
            others. The long cap (3 of 4) and the headroom routing are v6's.
  Self-test: blocks 0 / 1 / 3 at 1M + block 2 at 131K -> no 200K prompt, no 130K + max_tokens 2,000, no 100K without
  max_tokens on block 2; a 2K short to block 2 at equal load, elsewhere once block 2 queues 10.

V6 (program 2026-10-28T16:15:39Z / 16:15:53Z, DP4 acceptance run 1: blk3's shorts waited 318-330 s behind its
1M): KV-HEADROOM ROUTING. Under GLM53_ADMIT_KV_BUDGET a block admits a request only while every resident's reserve
(prompt + min(max_tokens, 32,768)) plus its own fits the KV pool. On a 1M pool (1,049,536 tokens) a resident 1M leaves
~9.4K tokens, so a block holding a 1M can take only arrivals under ~9.4K until it finishes (~4.5 min).
  headroom of a block = min(engine headroom - queued reserves - what the router sent since that poll, pool - the
            router's ledger of the open reserves it sent there). The engine part comes from plugin v3's
            glm53_admit_{pool,reserved,headroom,queued}_tokens on the block's /metrics. Without them, only the ledger
            with --pool-tokens (0 = unknown: the block counts as fitting, the v5 behaviour).
  an arrival's reserve = blocks(min(prompt + min(max_tokens, --admit-reserve-cap 32,768), --max-model-len)) x
            --kv-block-size 64 (no max_tokens: max_model_len - prompt, capped). A prompt <= --short-overtake-tokens
            (32,768 = SHORT_FIRST T) is checked against the RESIDENT headroom only: plugin v3 lets it pass a queued
            request the gate defers.
  routing   only blocks whose headroom covers the reserve, then the usual rules (session, prefix, long, least); a long
            arrival prefers the most headroom. No block fits: the arrival WAITS IN THE ROUTER until one does (each poll
            and each finished request re-checks), at most --headroom-wait-max-s 1,800, then the most headroom takes it.
  metrics   dp4router_block_headroom_tokens / _engine_headroom_tokens / _reserved_tokens {block}, dp4router_headroom_
            queue_depth / _waits_total / _wait_seconds_sum / _fallbacks_total / _steered_total.
  --headroom-routing off = v5. Fleet: plugin v3 (c5051f1) on every block, else pass --pool-tokens 1049536.

V5 (program 2026-10-28T14:26:28Z RULING (A), the width-8 mechanism of record: a host-sampled step reads the
whole [width x vocab] logits in ONE D2H -- 2.36 MiB at w8 -- the read that makes 14b's dispatch-CB credit leak bite):
  --host-sampling reject (DEFAULT) | strip | allow.  On /v1/completions and /v1/chat/completions a request carrying
  any parameter that makes the TT plugin HOST-sample (model_runner.check_perform_device_sampling + the model's
  capabilities: max_device_top_k 32, supports_device_logprobs False, supports_device_penalties False) is
    reject: answered 400 {"error": {"type": "router_host_sampling", "param": <the first one>, "message": ...}};
            it never reaches a block
    strip:  forwarded without them (response header X-Router-Stripped lists what was removed / set)
  The parameters: logprobs (ANY: completions' logprobs 0 too) / top_logprobs; min_p > 0; logit_bias; bad_words;
  allowed_token_ids; min_tokens > 0; an explicit top_k outside 1..--device-top-k (32, incl. -1) on a sampling request
  (temperature not 0); presence_penalty / frequency_penalty != 0, repetition_penalty != 1; structured outputs
  (response_format json_object / json_schema, guided_json / _regex / _choice / _grammar, structured_outputs).
  In every mode but allow, a sampling request (temperature not 0 or not given) with NO top_k is sent with top_k = 32
  (X-Router-Stripped: top_k=32), so it samples on device; greedy requests (temperature 0) are untouched.
  Counted in dp4router_host_sampling_total{params=...,action=reject|strip|top_k_clamped}. --device-top-k 32.
  Clients of record send tokens only: dp4load_v3 and sfcheck_v2 (no logprobs anywhere).

ROUTING, in order (reported in the X-Routed-Block / X-Route-Reason response headers and counted per block):
  session  X-Session-Id header, or the body's "session_id" / "user" -> the block that session used before
  prefix   the longest known prompt prefix (token ids hashed at 256 / 1K / 4K / 16K / 64K / 256K / 1M tokens; text
           and chat messages at 4.0 chars per token) -> the block that last served it, so LMCache stored hits stay
           local. Only when the match is >= 4,096 tokens (--affinity-min-tokens) or half the prompt (a short shared
           system prompt pins nothing) AND that block is within 4 requests (--affinity-slack) of the least-loaded one
           (else counted in dp4router_affinity_overflow_total and routed by the next rules)
  long     a NEW prompt over 32,768 tokens -> the block with the fewest long prefills in flight (concurrent 1M
           arrivals spread one per block), then the fewest active decode streams, in-flight, running
  least    otherwise -> the least-loaded block (router in-flight + the block's vllm:num_requests_waiting)
Down blocks (--down-after failed /health probes, polled every --poll-s) and draining blocks are skipped. A connect
failure before any response byte retries the request on another block.

CONTROL / OBSERVABILITY:
  GET  /health        200 while >= 1 block is up (JSON: per block up / draining / in-flight / decoding / queue depth /
                      running / KV usage / routed by reason / errors / mean TTFT); 503 when none is up
  GET  /metrics       Prometheus text: dp4router_block_{up,draining,inflight,decoding,queue_depth,running,kv_usage},
                      dp4router_routed_total{block,reason}, dp4router_errors_total{block,kind},
                      dp4router_block_ttft_seconds_{sum,count}
  GET  /router/status JSON
  POST /router/drain?block=N[&wait_s=S]   no new requests to N; with wait_s, returns once N's in-flight is 0 or S passed
  POST /router/undrain?block=N
  access log: <log_dir>/router_access.jsonl, one row per request (block, reason, prompt_tokens, ttft_s, total_s,
  status, error); progress in <log_dir>/router.log

INPUTS (stated): the four block servers already running on the box (default http://127.0.0.1:8100..8103), a python
with aiohttp (the box's vLLM venv: $PYTHON, $GLX_VENV/bin/python, <root>/venv-lmcache/bin/python, else python3), and
a free router port. No device, no tt-lock.

RUN (from the extract root):
  glm53/work-s12/glx12_dp4router.sh <router_port> <log_dir> [<blocks csv>] [extra args]   (foreground)
  extra args: --long-tokens 32768 --chars-per-token 4.0 --poll-s 1.0 --down-after 3 --read-timeout-s 7200 --host H
SELF-TEST (host-only, fake blocks): glm53/work-s12/glx12_dp4router.sh --selftest

v6f (kit v3's AgentX trace check on host-8chip, 2026-10-29T02:26Z): the replay through v6e + mock blocks aborted in its
warm-up -- a 133,803-token (and 134,778 / 138,337) chat prompt went to the 131,072 block and was refused (400), and
AgentX aborts the whole run on one refused root request. Cause: the router estimates text at 4 chars per token; the program
Code traces (code, JSON) tokenize at ~3. The live v6b has the same estimate: the Galaxy trace replay would abort too.
- --ctx-retry on (default): a block's 400 whose text says the prompt is over its context (vLLM: "maximum context
  length", "maximum model length", "max_model_len"), before anything streamed, retries on a block with a LARGER cap
  (blocks with a cap no larger than the refusing one are skipped); none left: the block's own 400 goes back
  (dp4router_ctx_refused_total, dp4router_ctx_retried_total). A capped long's admission carries over to the retry.
- --ctx-margin 1.25 (default): a block whose cap is below the largest takes a prompt only if (estimate x 1.25) + its
  reserve fits -- most over-cap traces never reach the small block (1.0 = v6e).
- --default-max-tokens N (0 = off) and --max-max-tokens M (0 = off) with --max-tokens-over clamp|reject (coordinator
  02:32:09Z, for the no-preempt roll): under --block-reserve-cap 0 a request WITHOUT max_tokens would reserve its
  block's whole cap (vLLM's default max_tokens = cap - prompt) = one such user per block. The router fills
  max_tokens = N into it (not for a prompt whose estimate x --ctx-margin + N passes the largest cap: it takes a whole
  block anyway), and an explicit max_tokens / max_completion_tokens over M is clamped to M (default) or answered 400
  naming the limit. Header X-Router-Stripped names the change (max_tokens=16384(default) / max_tokens=65536(clamped));
  dp4router_max_tokens_total{action=defaulted|clamped|rejected}.
  The roll: ... --block-reserve-cap 0 --block-max-total-tokens 1048576,1048576,131072,1048576 --default-max-tokens 16384
  --max-max-tokens 65536
TESTS: test_the_router_v6f_max_tokens (default 500 / limit 1000): no max_tokens -> 500 filled; 5,000 -> clamped to
1,000 (reject mode: 400 naming 1000); 800 untouched; a prompt near the cap gets no default.
test_the_router_v6f (blocks refuse text at a real 3 chars/token, block 2's cap 13,107): 8 concurrent 13,500-token
prompts (estimate 10,125: eligible with the margin) -> every one 200, 5 refusals on block 2 retried (v6e: 5 x 400);
a 14,667-token prompt steered off block 2 by the margin (v6e: 400).

v6e (program 2026-10-29T01:56:07Z P1 / 01:57:40Z, the INTERIM until the generator fix): blk2 (the 131K MTP block)
died at 01:53:29Z -- 8 concurrent requests at a 65,536 generation cap outgrew its pool once their outputs passed the
32,768 reserve cap, vLLM preempted one, and its recompute prefill raised in the generator (the engine dies with every
request in flight). The router can keep ITS traffic from oversubscribing a block:
- --block-reserve-cap  per block (comma list, or ONE value for every block): -1 / empty = --admit-reserve-cap (32,768,
  the plugin's rule); 0 = no cap -- the reserve is the FULL prompt + max_tokens (no max_tokens: the block's whole
  max_model_len), so the router never sends a block more than its pool holds at full generation; N = max_tokens
  counted up to N. A block with its own cap is a NO-PREEMPT block: its ledger binds every arrival (shorts too), and it
  takes a no-fit fallback only when no other block is usable.
  blk2 only:  --block-reserve-cap -1,-1,0,-1   (blocks in --blocks order: :8100..:8103)
  demo-wide:  --block-reserve-cap 0            (cost: a request WITHOUT max_tokens reserves a whole block)
- --block-max-total-tokens  per block (comma list or one value; 0 = none): prompt + max_tokens (no max_tokens: the
  block's max_model_len) over it goes to another block (dp4router_total_cap_steered_total; none fits:
  dp4router_total_cap_unfit_total, the usual cap fallback).
- metrics / status per block: dp4router_block_reserve_cap, dp4router_block_max_total_tokens; reserve_cap,
  max_total_tokens.
- ROUTER TRAFFIC ONLY: a client on a block's own port (the kit's parity arm on :8102) passes neither cap.
TESTS: test_the_router_v6e (blk2 pool 13,000, global reserve cap 2,000 scaled, 8 concurrent shorts with max_tokens
6,000): blk2 peak 1 in flight with --block-reserve-cap -1,-1,0,-1 vs v6d 5 (oversubscribed); --block-max-total-tokens
0,0,4000,0: long answers and no-max_tokens requests never on blk2, a 100-token answer still is.

v6d (program 2026-10-29T01:10:11Z (2) and 01:17:33Z (a) + (b)):
- ROOT CAUSE of (a) 16 shorts all on blk3 while blk0 / blk1 sat idle and (b) two 1M returns 12+ min in the router with
  every 1M block empty: plugin v3 publishes glm53_admit_{pool,reserved,headroom,queued}_tokens from scheduler.schedule()
  only (at most every 0.5 s), and a drained engine schedules nothing -- an EMPTY block's gauge keeps its last BUSY value
  (after a 1M request ~0 headroom) until a new request arrives, which the router withheld for want of headroom. v6c did
  NOT fix this (its unsatisfiable guard covers only a reserve over a whole pool).
  v6d: at every poll a block with nothing running or waiting on its engine and nothing of the router's open reads as its
  WHOLE pool (dp4router /router/status: drained, engine_headroom_published = the gauge as read).
- no headroom wait unless a request of the router's is open on a block whose pool could hold the reserve: its end is
  the only change the router can see (frozen gauges, vLLM's running gauge after an abort, load the router did not send:
  the most headroom takes the arrival now and the engine's own KV gate judges) (dp4router_headroom_nothing_open_total).
- the no-fit fallback keeps EVERY block tied at the most headroom and load decides (v6-v6c's max() took the first index
  of a tie); long prompts (> --long-tokens) rank load (router in-flight + engine queued) before headroom.
- a client that left while its arrival waited in the router is not routed (aiohttp does not cancel a handler on
  disconnect: v6c sent it on, a whole prefill for nobody; the first write fails only after the first token)
  (dp4router_client_gone_before_route_total); the long-cap FIFO is a line (a cancelled waiter leaves it; v3-v6c's
  tickets left an unserved ticket that stalled every later long arrival) and a capped arrival that leaves while it
  waits for headroom gives its long-cap slot back (dp4router_long_cancelled_total).
- STORED-CONTEXT PIN (01:10:11Z (2)): a huge arrival (>= --huge-tokens) whose prefix the router last sent to a block that
  is busy with another huge prompt WAITS for that block when expected_free + --hit-reload-s < its cold prefill
  (--prefill-s-per-mtok 262 x tokens / 1M); expected_free from the open huge request's timeline (prefill left at
  --prefill-s-per-mtok, then its remaining tokens at --decode-tok-s 50); at most --pin-wait-max-s (900), else cold
  elsewhere. Other huge arrivals avoid a block a pinned arrival waits for. --pin-wait off = v6c (dp4router_pin_total{decision}).
TESTS: test_the_router_v6d (pin: late return waits 0.38 s TTFT vs 1.25 s cold; early return goes cold; pin off: cold),
test_the_router_v6d_drained_blocks (plugin-like frozen gauges: tie [3,3,7,3] vs v6c [0,0,0,16]; drained 1M returns
1.27 s vs v6c the full --headroom-wait-max-s), test_the_router_v6d_client_gone (not routed: 2 seen vs v6c 3; the line
after a cancel: admitted vs v6c stuck). Plugin side (staged, not in this package): publish the gauges on drain / abort.

v6c (program 2026-10-29T00:11:38Z / 00:12:19Z, operator priority; the 1M c=4 sweep row ran 8 x 1M serialised on blk0
while blk1 / blk3 sat idle, ~4 min each):
- ROOT CAUSE found in v6b: a prompt the 4-chars-per-token estimate reads over EVERY block's max_model_len (a random-
  token 1M prompt reads ~1.35M) left ONE candidate -- the first block of the largest cap -- so every such prompt piled
  onto it. v6c keeps every block of the largest cap (the engine judges the real token count); counter
  dp4router_ctx_unfit_total.
- prefix affinity YIELDS to admission: only a block whose KV headroom covers the arrival's reserve NOW and with no long
  prompt queued there takes it by affinity; else the emptiest block, cold (dp4router_affinity_yield_total{why}).
- a prompt >= --huge-tokens (524,288) takes a whole block: at most --huge-per-block (1) open per block; an arrival
  waits in the router (FIFO; a waiter that leaves never blocks the line) until such a block is free, at most
  --headroom-wait-max-s. For huge prompts this per-block cap replaces the global N-1 long cap (8 x 1M at c=4 keeps 4
  blocks busy on a 4 x 1M fleet, 3 on the mixed fleet) (dp4router_huge_*; dp4router_block_huge_inflight{block}).
- the spec-block preference is BOUNDED: a smaller-cap block is preferred for shorts only while it holds fewer than
  --spec-prefer-max (8) open requests.
- a reserve no block's whole pool can hold (an over-read estimate) no longer waits --headroom-wait-max-s for nothing:
  the most-headroom block takes it at once (dp4router_headroom_unsatisfiable_total).
- --huge-per-block 0 = v6b's behaviour for huge prompts.
TESTS: test_dp4router.py test_the_router_v6c (fake blocks, tonight's pattern: 8 identical huge prompts at c=4) and
test_the_router_v6b_piles_huge_prompts_on_one_block (the negative control on dp4router_v6b.py: all 8 on block 0).
