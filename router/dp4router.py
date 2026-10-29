#!/usr/bin/env python3
"""Stage 12 dp4router v6f -- the front router for TP8 x DP4 on the Galaxy (operator delivery, coordinator
2026-10-28T12:20:55Z): ONE OpenAI-compatible endpoint in front of the four block servers (:8100..:8103 on the box), a
thin asyncio proxy (aiohttp, a vLLM hard dependency) with streaming passthrough. The block servers are untouched.

v3 (program 2026-10-28T13:27:02Z, DP4 load run 1: a block's prefill is FIFO, so a short arrival behind an in-flight
1M waited the whole 1M, 255-325 s): (a) LONG prefills (> --long-cap-tokens, 131,072) run on at most --long-cap blocks at
once (default N - 1 = 3 of 4); a further long waits IN THE ROUTER (FIFO) until one of them reaches its first token;
(b) every other prompt prefers the LONG-FREE blocks (no long prefill in flight; then no > --long-tokens prefill), falling
back to the rest only when none is up -- a session / prefix match on a block with a long prefill in flight is passed over.
v4 (program 2026-10-28T13:47:25Z, load run 2: the v3 cap did not engage -- four 1M arrivals within ~5 s all passed
it): the cap counts a long at ADMISSION (inside the cap's FIFO lock; released at its first token or its end), the
prompt is classified (tokens, prefix keys) BEFORE admission, and the block choice + its counters happen with no await
between them (a > 1 MB body's parse / hash ran in a thread there, and the other longs passed the cap meanwhile).
v6c (program 2026-10-29T00:11:38Z / 00:12:19Z): a prompt the estimate reads over every cap stays on every block of
the largest cap (v6b kept only the first: 8 x 1M piled onto blk0); prefix affinity only where the block can admit now;
>= --huge-tokens (524,288) at most --huge-per-block (1) per block (the global long cap is for the prompts below it);
the spec preference bounded by --spec-prefer-max; an unsatisfiable reserve does not wait. See README.txt.
v6d (program 2026-10-29T01:10:11Z (2) / 01:17:33Z): a huge arrival whose stored context sits on a busy block waits
for it when expected free + the hit beats a cold prefill (--pin-wait); a DRAINED block (nothing running / waiting on its
engine, nothing of ours open) reads as its whole pool -- plugin v3's admit gauges freeze at their last busy value when
the engine stops scheduling; no headroom wait unless a request of ours is open; the no-fit fallback keeps every block
tied at the most headroom (load decides); long prompts rank load before headroom; a client gone while it waited is not
routed; the long-cap FIFO is cancel-safe. See README.txt.
v6e (program 2026-10-29T01:56:07Z P1, the interim until the generator fix): --block-reserve-cap (per block, or one
value for all; 0 = a NO-PREEMPT block: its reserve is the full prompt + max_tokens, so the router never sends it more
than its pool holds at full generation -- blk2 died on a recompute prefill after outputs past the 32,768 reserve cap
outgrew its pool) and --block-max-total-tokens (prompt + max_tokens over it -> another block). Router traffic only.
v6f (kit v3's trace check, 2026-10-29T02:26Z): a block's 400 for a prompt over its context (nothing streamed yet)
retries on a LARGER-cap block (--ctx-retry), and a smaller-cap block takes a prompt only if (estimate x --ctx-margin
1.25) + its reserve fits: agent traces tokenize at ~3 chars/token, the router estimates 4, so >131K-token traces went to
the 131K block, were refused, and one refused root request aborts a whole AgentX warm-up. And (02:32:09Z) for the
no-preempt roll: --default-max-tokens fills max_tokens into a request that sends none, --max-max-tokens bounds it.
v6b (program 2026-10-28T18:10:22Z, the MIXED FLEET: 3 x 1M f2i3 blocks + 1 x 131,072-cap spec block): PER-BLOCK
max_model_len. Each block's cap comes from its /v1/models (max_model_len, re-read every 30 s) or --block-max-model-len
(a comma list per block; 0 = learn). A block never takes a request whose prompt + min(max_tokens, --admit-reserve-cap
32,768) exceeds its cap (no max_tokens: the reserve cap), nor one whose prompt + an explicit max_tokens does (vLLM would
refuse it there). Short prompts (<= --spec-prefer-tokens 32,768) PREFER the smaller-cap blocks: in the least-loaded rule
such a block counts --spec-prefer-slack (4) requests less load. Everything else is v6.

v6 (program 2026-10-28T16:15:39Z / 16:15:53Z, the DP4 acceptance's blk3: a 1M resident on a 1M pool leaves
~9.4K tokens of KV-budget headroom, so a 32K / 131K arrival there waits the 1M out): KV-HEADROOM ROUTING. Each block's
admission headroom comes from its /metrics (plugin v3: glm53_admit_{pool,reserved,headroom,queued}_tokens = the
GLM53_ADMIT_KV_BUDGET reserve rule in tokens) and from the router's own ledger of the reserves it sent there and that are
still open (--pool-tokens when a block publishes no pool). An arrival's reserve = blocks(min(prompt + min(max_tokens,
--admit-reserve-cap 32,768), --max-model-len)) x --kv-block-size; it is routed only to a block whose headroom covers it:
min(engine headroom - queued reserves - what the router sent since that poll, pool - ledger); a prompt <= --short-overtake-
tokens (32,768, the plugin's SHORT_FIRST T) is checked against the RESIDENT headroom only (plugin v3 lets it pass a queued
request the gate defers). Among the blocks that fit, the usual rules (session, prefix, long, least); a long arrival
prefers the most headroom. No block fits: the arrival waits IN THE ROUTER (dp4router_headroom_queue_depth) until one does
(a poll or a finished request re-checks), at most --headroom-wait-max-s (1,800), then goes to the block with the most
headroom. --headroom-routing off = v5.

v5 (program 2026-10-28T14:23:03Z, the width-8 mechanism of record: a host-sampled step reads the whole [width x
vocab] logits in ONE D2H, 2.36 MiB at w8 > the 512-KiB dispatch CB -> a prefetcher / dispatcher credit deadlock):
--host-sampling reject (default) | strip | allow. A request whose parameters force HOST sampling in the TT plugin
(logprobs / top_logprobs -- the model has no device logprobs --, min_p > 0, logit_bias, bad_words, allowed_token_ids,
min_tokens > 0, an explicit top_k outside 1..--device-top-k (32) on a sampling request, structured outputs, and a
presence / frequency / repetition penalty off its neutral value: the model declares supports_device_penalties False) is
refused with a 400 naming them, or stripped of them (header X-Router-Stripped; an explicit top_k is clamped to 32); and a
sampling request (temperature not 0, or not given) with NO top_k gets top_k = 32 in every mode (the plugin host-samples
it otherwise; greedy requests are untouched), so the fleet never runs a host-sampled decode step.

Routing, in order (the reason is counted and logged per request):
  1. session  -- X-Session-Id header, or the body's "session_id" / "user": the block the session went to before;
  2. prefix   -- the longest known prompt prefix: the prompt (token ids, text, or chat messages) is hashed at fixed cut
                 points (256 .. 1M tokens; x4 in characters for text) and a request goes to the block that last served
                 that prefix, so LMCache stored hits (and the KDA checkpoints) stay local -- when the match is >=
                 --affinity-min-tokens (4,096) or half the prompt (a short common system prompt pins nothing), and
                 the block is within --affinity-slack (4) requests of the least-loaded one;
  3. long     -- a NEW prompt over --long-tokens (32,768) goes to the block with the fewest long prefills in flight,
                 then the fewest active decode streams (an arrival prefilling under 8 decoders runs them at ~62 pct:
                 the in-flight prefill interference);
  4. least    -- otherwise the least-loaded block: router in-flight + the block's vllm:num_requests_waiting.
A block that is down (its /health failed --down-after times) or draining is skipped by every rule; a connection
failure before any response byte retries the request on another block.

Endpoints: POST /v1/completions, /v1/chat/completions (streamed or not, passed through byte for byte), GET /v1/models,
POST /tokenize, /detokenize; GET /health (200 while >= 1 block is up; per-block detail), GET /metrics (Prometheus:
per-block up / draining / in-flight / decoding / queue depth / running / KV usage, routed-by-reason, errors, the
router-observed TTFT sum / count), GET /router/status (JSON); POST /router/drain?block=N[&wait_s=S] (no new requests
to N; with wait_s, returns when N's in-flight is 0 or S passed), POST /router/undrain?block=N.
Access log: --log-dir/router_access.jsonl (one row per request: block, reason, prompt tokens, ttft_s, total_s, status).

usage: python3 -u dp4router.py [--port 8000] [--host 127.0.0.1] [--blocks http://127.0.0.1:8100,...,:8103]
       [--long-tokens 32768] [--chars-per-token 4.0] [--poll-s 1.0] [--down-after 3] [--log-dir DIR]
       (the python of the box's vLLM venv: aiohttp)
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import sys
import time
from array import array
from collections import OrderedDict

from aiohttp import ClientConnectorError, ClientSession, ClientTimeout, TCPConnector, web

VERSION = "dp4router v6f"
CUTS_TOKENS = (256, 1024, 4096, 16384, 65536, 262144, 1048576)
HOP = {"host", "content-length", "transfer-encoding", "connection", "keep-alive", "proxy-connection", "upgrade", "te",
       "trailer", "proxy-authorization", "proxy-authenticate"}  # fmt: skip
log = logging.getLogger("dp4router")


class Block:
    def __init__(self, idx: int, url: str):
        self.idx, self.url = idx, url.rstrip("/")
        self.up, self.fails, self.draining = True, 0, False
        self.inflight = 0  # router requests open on this block
        self.decoding = 0  # of them: streams past their first response byte
        self.long_pf = 0  # of them: prompts over --long-tokens still before their first byte (a long prefill)
        self.capped_pf = 0  # of them: prompts over --long-cap-tokens still before their first byte (v3 cap / long-free)
        self.waiting = self.running = 0.0
        self.kv_usage = None
        self.last_poll = None
        self.routed: dict = {}
        self.errors: dict = {}
        self.ttft_sum, self.ttft_n = 0.0, 0
        self.adm_pool = self.adm_headroom = self.adm_queued = None  # v6: the engine's glm53_admit_*_tokens (plugin v3)
        self.reserved = 0  # v6: the router's ledger -- the reserves of the requests it sent here that are still open
        self.since_poll = 0  # v6: of them, the reserves sent since the last metrics poll
        self.adm_raw = None  # v6d: the engine's published headroom as read (adm_headroom = the pool when drained)
        self.res_cap = -1  # v6e: --block-reserve-cap (-1: the global --admit-reserve-cap; 0: NO cap = the full
        # prompt + max_tokens -- a no-preempt block; N: max_tokens counted up to N)
        self.total_cap = 0  # v6e: --block-max-total-tokens (0: max_model_len only): prompt + max_tokens above it go elsewhere
        self.drained = False  # v6d: at the last poll nothing ran / waited on the engine and nothing of ours was open
        self.mml = None  # v6b: the block's max_model_len (/v1/models or --block-max-model-len)
        self.huge = 0  # v6c: open requests with a prompt >= --huge-tokens (one takes the whole pool: --huge-per-block)
        self.huge_reqs: dict = {}  # v6d: those requests' timelines {key: {t, ntok, max_tokens, t_first, out}}
        self.mml_fixed = False
        self.mml_t = 0.0

    def view(self) -> dict:
        return {"block": self.idx, "url": self.url, "up": self.up, "draining": self.draining, "inflight": self.inflight,
                "decoding": self.decoding, "long_prefills": self.long_pf, "capped_long_prefills": self.capped_pf, "huge_inflight": self.huge, "queue_depth": self.waiting, "running": self.running,
                "kv_usage": self.kv_usage, "routed": dict(self.routed), "errors": dict(self.errors),
                "admit_headroom_tokens": self.adm_headroom, "admit_queued_tokens": self.adm_queued,
                "engine_headroom_published": self.adm_raw, "drained": self.drained,
                "admit_pool_tokens": self.adm_pool, "ledger_reserved_tokens": self.reserved, "max_model_len": self.mml,
                "reserve_cap": self.res_cap, "max_total_tokens": self.total_cap,
                "ttft_mean_s": round(self.ttft_sum / self.ttft_n, 3) if self.ttft_n else None}  # fmt: skip


def prompt_of(body: dict):
    """(kind, payload) of a request's prompt: ('ids', [int]) / ('text', str) / (None, None)"""
    p = body.get("prompt")
    if p is None and isinstance(body.get("messages"), list):
        parts = []
        for m in body["messages"]:
            c = m.get("content") if isinstance(m, dict) else None
            if isinstance(c, list):
                c = "".join(x.get("text", "") for x in c if isinstance(x, dict))
            parts.append(f"{m.get('role', '') if isinstance(m, dict) else ''}\x1f{c or ''}")
        return "text", "\x1e".join(parts)
    if isinstance(p, list) and p and isinstance(p[0], list):
        p = p[0]  # a batch: route by its first prompt
    if isinstance(p, list) and p and isinstance(p[0], str):
        p = p[0]
    if isinstance(p, list) and all(isinstance(x, int) for x in p[:16]):
        return "ids", p
    if isinstance(p, str):
        return "text", p
    return None, None


PENALTIES = (("presence_penalty", 0.0), ("frequency_penalty", 0.0), ("repetition_penalty", 1.0))
HOST_ONLY_STRUCTURED = ("response_format", "guided_json", "guided_regex", "guided_choice", "guided_grammar",
                        "structured_outputs")


def _num(x):
    """a JSON number field as a float: None when absent, NaN when malformed (a malformed top_k is refused; the rest is
    left to the block server's own validation)"""
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def host_sampling_params(body: dict, device_top_k: int) -> list:
    """the request fields that force host sampling in the TT plugin (model_runner.check_perform_device_sampling)"""
    out = []
    lp, tl = body.get("logprobs"), body.get("top_logprobs")
    # ANY logprobs (completions' logprobs 0 included: max_num_logprobs is not None) -- the model declares
    # supports_device_logprobs False; chat's logprobs false is none. (`0 in (None, False)` is True: no tuple test.)
    if (lp is not None and lp is not False) or (tl is not None and tl is not False and tl != 0):
        out.append("logprobs")
    if (_num(body.get("min_p")) or 0) > 0:
        out.append("min_p")
    for k in ("logit_bias", "bad_words", "allowed_token_ids"):
        if body.get(k):
            out.append(k)
    if (_num(body.get("min_tokens")) or 0) > 0:
        out.append("min_tokens")
    for k, neutral in PENALTIES:  # the model declares supports_device_penalties False
        v = _num(body.get(k))
        if v is not None and v != neutral:
            out.append(k)
    tk = _num(body.get("top_k"))  # an EXPLICIT top_k outside 1..K on a sampling request (ruling 14:26:28Z: refuse it)
    t = _num(body.get("temperature"))
    if tk is not None and t != 0.0 and not (1 <= tk <= device_top_k):
        out.append("top_k")
    rf = body.get("response_format")
    if (isinstance(rf, dict) and rf.get("type") not in (None, "text")) or any(body.get(k) for k in HOST_ONLY_STRUCTURED[1:]):
        out.append("structured_outputs")
    return out


def clamp_top_k(body: dict, device_top_k: int) -> dict | None:
    """a sampling request (temperature not 0, or not given: the model's generation defaults may sample) whose top_k is
    unset / < 1 / > the device sampler's K is HOST-sampled by the plugin (check_perform_device_sampling): the router sets
    top_k = K (both modes; a greedy request is unaffected). Returns the new body, or None when nothing changes."""
    if _num(body.get("temperature")) == 0.0:
        return None
    tk = _num(body.get("top_k"))
    if tk is not None and 1 <= tk <= device_top_k:
        return None
    b = dict(body)
    b["top_k"] = int(device_top_k)
    return b


def strip_host_sampling(body: dict, found: list) -> dict:
    b = dict(body)
    for f in found:
        if f == "logprobs":
            b.pop("logprobs", None)
            b.pop("top_logprobs", None)
        elif f == "structured_outputs":
            for k in HOST_ONLY_STRUCTURED:
                b.pop(k, None)
        elif f == "top_k":
            b.pop("top_k", None)  # (clamp_top_k then sets the device K)
        else:
            b.pop(f, None)
    return b


def prefix_keys(kind: str, payload, chars_per_token: float) -> list:
    """[(cut, key)] for every cut point the prompt reaches, shortest first (one incremental sha1 pass)"""
    if kind == "ids":
        data, unit, cuts = array("q", payload).tobytes(), 8, CUTS_TOKENS
    elif kind == "text":
        data, unit, cuts = payload.encode("utf-8", "replace"), 1, tuple(int(c * chars_per_token) for c in CUTS_TOKENS)
    else:
        return []
    h, pos, out = hashlib.sha1(kind.encode()), 0, []
    for cut in cuts:
        end = cut * unit
        if end > len(data):
            break
        h.update(data[pos:end])
        pos = end
        out.append((cut, h.copy().hexdigest()))
    return out


def prompt_tokens(kind: str, payload, chars_per_token: float) -> int:
    if kind == "ids":
        return len(payload)
    if kind == "text":
        return int(len(payload) / chars_per_token)
    return 0


_METRIC = re.compile(r"^((?:vllm:|glm53_admit_)[a-z_]+)(\{[^}]*\})?\s+([0-9.eE+-]+)\s*$")


# v6f: a block's refusal of a prompt over its context (vLLM's 400 texts; the mock's "max_model_len")
CTX_REFUSAL = re.compile(r"max_model_len|maximum context length|maximum model length", re.I)


class Need(int):
    """v6e: an arrival's KV reserve -- the int is the global rule (--admit-reserve-cap); .per[block idx] overrides it on
    a block with its own --block-reserve-cap (a NO-PREEMPT block: the full prompt + max_tokens)"""

    def __new__(cls, base: int, per: dict | None = None):
        o = super().__new__(cls, base)
        o.per = per or {}
        return o

    def on(self, b) -> int:
        return self.per.get(b.idx, int(self))


def need_on(need, b) -> int:
    return need.on(b) if isinstance(need, Need) else int(need or 0)


def per_block_ints(spec: str, n: int, default: int) -> list:
    """a comma list per block ("" = default everywhere; ONE value = every block)"""
    vals = [x.strip() for x in (spec or "").split(",") if x.strip()]
    if len(vals) == 1:
        vals = vals * n
    out = [default] * n
    for i, v in enumerate(vals[:n]):
        out[i] = int(v)
    return out


def parse_metrics(text: str) -> dict:
    out: dict = {}
    for line in text.splitlines():
        m = _METRIC.match(line)
        if m:
            try:
                out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(3))
            except ValueError:
                pass
    return out


class Router:
    def __init__(self, a: argparse.Namespace):
        self.a = a
        self.blocks = [Block(i, u) for i, u in enumerate(a.blocks)]
        self.sessions: OrderedDict = OrderedDict()  # session id -> block idx
        self.prefixes: OrderedDict = OrderedDict()  # prefix key -> block idx (LRU)
        self.max_entries = int(a.max_affinity_entries)
        self.http: ClientSession | None = None
        self.t0 = time.time()
        self.rr = 0
        self.access = None
        self.affinity_overflow = 0  # prefix matches passed over: their block was over the slack
        self.long_cv: asyncio.Condition | None = None  # v3: the long-prefill cap's FIFO
        self.long_queue = 0
        self.long_queued_total = 0
        self.long_wait_s = 0.0
        self.long_next = self.long_served = 0  # FIFO tickets
        self.long_admitted = 0  # v4: longs admitted and not yet at their first token / end
        self.sites: list = []  # v4: the listening sites (/router/retire stops them)
        self.host_sampling_hits: dict = {}  # v5: refused / stripped requests by the parameters found
        self.hr_cv: asyncio.Condition | None = None  # v6: arrivals waiting for KV headroom
        self.hr_queue = 0
        self.hr_waits = self.hr_fallbacks = self.hr_steered = 0
        self.ctx_steered = 0  # v6b: choices narrowed by a block's max_model_len
        self.ctx_unfit = 0  # v6c: arrivals no block's cap takes by the router's estimate (kept on every block: the engine
        # judges -- the 4-chars-per-token estimate reads a random-token 1M prompt as ~1.35M)
        self.affinity_yield_headroom = self.affinity_yield_long = 0  # v6c: affinity matches passed over (cannot admit)
        self.huge_waits, self.huge_wait_s, self.huge_queue = 0, 0.0, 0
        self.huge_cv = None
        self.huge_line: list = []  # v6c: the huge arrivals waiting, oldest first
        self.pin_waits = self.pin_cold = self.pin_timeouts = 0  # v6d: stored-context arrivals: waited / went cold
        self.pinned: dict = {}  # v6d: block idx -> arrivals waiting for that block (unpinned arrivals avoid it)
        for i, v in enumerate(x for x in (a.block_max_model_len or "").split(",") if x.strip()):
            if i < len(self.blocks) and int(v) > 0:
                self.blocks[i].mml, self.blocks[i].mml_fixed = int(v), True
        for b, rc, tc in zip(self.blocks, per_block_ints(a.block_reserve_cap, len(self.blocks), -1),
                             per_block_ints(a.block_max_total_tokens, len(self.blocks), 0)):  # fmt: skip
            b.res_cap, b.total_cap = rc, tc
        self.total_steered = self.total_unfit = 0  # v6e
        self.ctx_refused = self.ctx_retried = 0  # v6f
        self.mt_defaulted = self.mt_clamped = self.mt_rejected = 0  # v6f: --default-max-tokens / --max-max-tokens
        self.hr_wait_s = 0.0
        self.retiring = False

    # -- v3: the long-prefill cap --------------------------------------------------------------------------------------
    def long_cap(self) -> int:
        n = int(self.a.long_cap)
        return max(1, len(self.blocks) - 1) if n < 0 else n

    def capped_inflight(self) -> int:
        return sum(1 for b in self.blocks if b.capped_pf > 0)

    async def long_admit(self) -> None:
        """wait (FIFO) until fewer than --long-cap blocks hold a long prefill. v6d: a LINE, not tickets -- a waiter that
        leaves (the client gone, a cancel) leaves the line; with v3-v6c's tickets its ticket was never served and every
        later long arrival waited forever with nothing in flight (01:17:33Z (b): two 1M returns 12+ min in the router)"""
        if self.long_cv is None:
            self.long_cv = asyncio.Condition()
        t0 = time.time()
        me = object()
        line = self.__dict__.setdefault("long_line", [])
        async with self.long_cv:
            line.append(me)
            queued = False
            try:
                while line[0] is not me or self.long_admitted >= self.long_cap():
                    if not queued:
                        queued = True
                        self.long_queue += 1
                        self.long_queued_total += 1
                    await self.long_cv.wait()
                self.long_admitted += 1
            finally:
                line.remove(me)
                if queued:
                    self.long_queue -= 1
                    self.long_wait_s += time.time() - t0
                self.long_cv.notify_all()

    async def long_release(self, counted: bool = False) -> None:
        if counted:
            self.long_admitted = max(0, self.long_admitted - 1)
        if self.long_cv is not None:
            async with self.long_cv:
                self.long_cv.notify_all()

    # -- block state -------------------------------------------------------------------------------------------------
    async def poll_once(self) -> None:
        async def one(b: Block):
            try:
                async with self.http.get(b.url + "/health", timeout=ClientTimeout(total=self.a.health_timeout_s)) as r:
                    ok = r.status == 200
            except Exception:  # noqa: BLE001 -- a failed probe is a down signal
                ok = False
            if ok:
                b.fails, b.up = 0, True
                try:
                    async with self.http.get(b.url + "/metrics", timeout=ClientTimeout(total=self.a.health_timeout_s)) as r:
                        m = parse_metrics(await r.text())
                    b.waiting = m.get("vllm:num_requests_waiting", 0.0)
                    b.running = m.get("vllm:num_requests_running", 0.0)
                    b.kv_usage = m.get("vllm:kv_cache_usage_perc", m.get("vllm:gpu_cache_usage_perc"))
                    if not b.mml_fixed and (b.mml is None or time.time() - b.mml_t > 30):  # v6b: the block's cap
                        async with self.http.get(b.url + "/v1/models", timeout=ClientTimeout(total=self.a.health_timeout_s)) as r:
                            data = (await r.json()).get("data") or [{}]
                        if data[0].get("max_model_len"):
                            b.mml, b.mml_t = int(data[0]["max_model_len"]), time.time()
                    if "glm53_admit_headroom_tokens" in m:  # v6 (plugin v3)
                        b.adm_pool = m.get("glm53_admit_pool_tokens")
                        b.adm_headroom = b.adm_raw = m["glm53_admit_headroom_tokens"]
                        b.adm_queued = m.get("glm53_admit_queued_tokens", 0.0)
                        b.since_poll = 0
                        # v6d (01:17:33Z (a) + (b)): plugin v3 publishes these gauges from schedule() only (at most
                        # every 0.5 s) and a drained engine schedules nothing -- the gauge keeps its last BUSY value
                        # (a 1M reserve) until the next request arrives, which the router held for want of headroom.
                        # A block with nothing running or waiting on the engine and nothing of ours open is EMPTY: its
                        # headroom is the whole pool. (v6-v6c read the frozen value: shorts all went to the one block
                        # last seen with room; 1M returns waited --headroom-wait-max-s with every block empty.)
                        b.drained = (b.inflight == 0 and not b.running and not b.waiting and b.adm_pool is not None)
                        if b.drained:
                            b.adm_headroom, b.adm_queued = b.adm_pool, 0.0
                except Exception:  # noqa: BLE001 -- metrics are advisory
                    pass
            else:
                b.fails += 1
                if b.fails >= self.a.down_after and b.up:
                    b.up = False
                    log.warning("block %d (%s) DOWN after %d failed health probes", b.idx, b.url, b.fails)
            b.last_poll = time.time()

        await asyncio.gather(*(one(b) for b in self.blocks))
        await self.headroom_notify()

    async def poll_loop(self) -> None:
        while True:
            await self.poll_once()
            await asyncio.sleep(self.a.poll_s)

    def usable(self, exclude=()) -> list:
        return [b for b in self.blocks if b.up and not b.draining and b.idx not in exclude]

    # -- v6: KV-headroom routing ---------------------------------------------------------------------------------------
    def reserve_tokens(self, ntok: int, body: dict | None, cap: int | None = None, mml: int | None = None) -> int:
        """an arrival's GLM53_ADMIT_KV_BUDGET reserve in tokens (the plugin's rule): blocks(min(prompt + min(max_tokens,
        cap), max_model_len)) x block; no max_tokens = the server's default max_model_len - prompt. v6e: cap / mml of
        one block (cap 0 = none: the full prompt + max_tokens)"""
        mml = int(mml or self.a.max_model_len)
        body = body or {}
        mx = _num(body.get("max_tokens", body.get("max_completion_tokens")))
        out = int(mx) if mx is not None and mx == mx else max(0, mml - ntok)
        cap = int(self.a.admit_reserve_cap) if cap is None else int(cap)
        if cap > 0:
            out = min(out, cap)
        bs = max(1, int(self.a.kv_block_size))
        return -(-min(int(ntok) + max(0, out), mml) // bs) * bs

    def need_of(self, ntok: int, body: dict | None) -> Need:
        """v6e: the global reserve + the per-block ones of the blocks with their own --block-reserve-cap"""
        base = self.reserve_tokens(ntok, body)
        per = {}
        for b in self.blocks:
            if b.res_cap >= 0:
                per[b.idx] = self.reserve_tokens(ntok, body, cap=b.res_cap, mml=b.mml)
        return Need(base, per)

    def headroom(self, b: Block, short: bool = False) -> float | None:
        """the admission headroom of block b for a NEW arrival, tokens (None = unknown: no metric, no --pool-tokens):
        min(engine headroom - queued reserves - sent since that poll, pool - the router's ledger); short=True (a prompt
        <= --short-overtake-tokens): the resident headroom only (plugin v3 lets it pass a queued deferred request)"""
        vals = []
        if b.adm_headroom is not None:
            vals.append(b.adm_headroom - b.since_poll - (0 if short else (b.adm_queued or 0)))
        pool = b.adm_pool if b.adm_pool is not None else (self.a.pool_tokens or None)
        if pool and (not (short and b.adm_headroom is not None) or b.res_cap >= 0):  # v6e: a no-preempt block's
            vals.append(pool - b.reserved)  # ledger binds every arrival (the engine's reserves are capped at 32,768)
        return min(vals) if vals else None

    def fits_ctx(self, b: Block, ntok: int, body: dict | None) -> bool:
        """v6b: the request fits block b's max_model_len -- prompt + min(max_tokens, cap) (no max_tokens: the cap) and
        prompt + an explicit max_tokens both within it; an unknown cap fits"""
        if b.mml is None:
            return True
        if self.a.ctx_margin > 1.0 and b.mml < max(x.mml or 0 for x in self.blocks):
            ntok = int(ntok * self.a.ctx_margin)  # v6f: a smaller-cap block takes only a prompt that fits with margin
        body = body or {}
        mx = _num(body.get("max_tokens", body.get("max_completion_tokens")))
        mx = int(mx) if mx is not None and mx == mx else None
        cap = int(self.a.admit_reserve_cap) or b.mml
        res = min(mx, cap) if mx is not None else cap
        return ntok + res <= b.mml and (mx is None or ntok + mx <= b.mml) and self.fits_total(b, ntok, mx)

    def fits_total(self, b: Block, ntok: int, mx: int | None) -> bool:
        """v6e (program 2026-10-29T01:56:07Z P1, the interim): prompt + max_tokens (none: the block's cap) within
        the block's --block-max-total-tokens"""
        if not b.total_cap:
            return True
        return ntok + (mx if mx is not None else (b.mml or 1 << 62)) <= b.total_cap

    def fits(self, b: Block, need: int, ntok: int) -> bool:
        h = self.headroom(b, short=ntok <= self.a.short_overtake_tokens)
        return h is None or h >= need_on(need, b)

    def waitable(self, need: int, exclude=()) -> list:
        """v6d: the blocks worth waiting for -- a request of OURS is open there (its end frees a reserve the router sees:
        the ledger, the next poll) and its pool, if known, holds the reserve. None: waiting cannot change the answer
        (an empty fleet whose engine gauges froze busy, an abort the engine never re-published) -- the arrival takes
        the most headroom now and the engine's own gate judges"""
        out = []
        for b in self.usable(exclude):
            pool = b.adm_pool if b.adm_pool is not None else (self.a.pool_tokens or None)
            if b.inflight > 0 and (pool is None or pool >= need_on(need, b)):
                out.append(b)
        return out

    async def headroom_admit(self, need: int, ntok: int, exclude=()) -> None:
        """wait while no usable block fits the arrival (re-checked at every poll and every finished request), at most
        --headroom-wait-max-s; then the arrival takes the block with the most headroom (choose)"""
        if not self.usable(exclude) or any(self.fits(b, need, ntok) for b in self.usable(exclude)):
            return
        pools = [(b.adm_pool if b.adm_pool is not None else (self.a.pool_tokens or None), need_on(need, b))
                 for b in self.usable(exclude)]  # fmt: skip
        if pools and all(p is not None for p, _ in pools) and all(n > p for p, n in pools):
            # v6c: a reserve no block's WHOLE pool holds (an over-read estimate) never fits: waiting would burn
            # --headroom-wait-max-s for nothing -- the most-headroom block takes it now, the engine judges
            self.hr_unsat = getattr(self, "hr_unsat", 0) + 1
            return
        if not self.waitable(need, exclude):
            self.hr_nothing_open = getattr(self, "hr_nothing_open", 0) + 1
            return
        if self.hr_cv is None:
            self.hr_cv = asyncio.Condition()
        t0 = time.time()
        self.hr_waits += 1
        self.hr_queue += 1
        try:
            async with self.hr_cv:
                while not any(self.fits(b, need, ntok) for b in self.usable(exclude)):
                    left = self.a.headroom_wait_max_s - (time.time() - t0)
                    if not self.waitable(need, exclude):  # v6d: the last of ours ended and it still does not fit
                        self.hr_nothing_open = getattr(self, "hr_nothing_open", 0) + 1
                        break
                    if left <= 0 or not self.usable(exclude):
                        self.hr_fallbacks += 1
                        log.warning("no block has %d tokens of KV headroom after %.0f s: the most headroom takes it",
                                    need, time.time() - t0)  # fmt: skip
                        break
                    try:
                        await asyncio.wait_for(self.hr_cv.wait(), timeout=min(left, max(1.0, 2 * self.a.poll_s)))
                    except asyncio.TimeoutError:
                        pass
        finally:
            self.hr_queue -= 1
            self.hr_wait_s += time.time() - t0

    # -- v6c: one huge prompt (>= --huge-tokens) per block ------------------------------------------------------------
    def _huge_cands(self, ntok: int, body: dict | None, exclude=()) -> list:
        cand = self.usable(exclude)
        ctx = [x for x in cand if self.fits_ctx(x, ntok, body)]
        if ctx:
            return ctx
        top = max((x.mml or 1 << 62) for x in cand) if cand else 0
        return [x for x in cand if (x.mml or 1 << 62) == top]

    def _unpinned_free(self, ntok: int, body: dict | None, exclude=()) -> list:
        free = [x for x in self._huge_cands(ntok, body, exclude) if x.huge < self.a.huge_per_block]
        open_ = [x for x in free if not self.pinned.get(x.idx)]  # v6d: a block a stored-context arrival waits for
        return open_ or free

    def huge_free(self, ntok: int, body: dict | None, exclude=()) -> bool:
        return bool(self._unpinned_free(ntok, body, exclude))

    # -- v6d: a huge arrival with a stored context waits for its block when that beats a cold prefill elsewhere ------
    def expected_free_s(self, b: Block, now: float) -> float:
        """seconds until block b's open huge requests end, by the router's own record: a request still before its first
        byte = its expected prefill (--prefill-s-per-mtok x its tokens) minus the elapsed, plus its decode (max_tokens at
        --decode-tok-s); one decoding = its remaining tokens at --decode-tok-s"""
        rate, dec = self.a.prefill_s_per_mtok / 1e6, max(1e-3, self.a.decode_tok_s)
        left = 0.0
        for r in b.huge_reqs.values():
            mt = r["max_tokens"] if r["max_tokens"] is not None else 1024
            if r["t_first"] is None:
                t = max(0.0, r["ntok"] * rate - (now - r["t"])) + mt / dec
            else:
                t = max(0, mt - r["out"]) / dec
            left = max(left, t)
        return left

    def affinity_block(self, ntok: int, keys: list):
        """the block holding the longest stored prefix of this prompt (the affinity rule's match), else None"""
        for cut, key in reversed(keys):
            b = self.prefixes.get(key)
            if b is None:
                continue
            if cut < self.a.affinity_min_tokens and cut * 2 < ntok:
                return None
            return self.blocks[b]
        return None

    def pin_decision(self, ntok: int, keys: list, body: dict | None):
        """(block to wait for | None, why): wait for the affinity block when its expected free time + a hit reload
        (--hit-reload-s) beats a cold prefill elsewhere (--prefill-s-per-mtok x the prompt); a free affinity block
        needs no decision (v6c routes there), a free other block makes the comparison"""
        if self.a.pin_wait != "on":
            return None, "off"
        b = self.affinity_block(ntok, keys)
        if b is None or not b.up or b.draining or not self.fits_ctx(b, ntok, body):
            return None, "no stored context"
        if b.huge < self.a.huge_per_block:
            return None, "affinity block free"
        wait = self.expected_free_s(b, time.time()) + self.a.hit_reload_s
        cold = ntok * self.a.prefill_s_per_mtok / 1e6
        if wait < cold:
            return b, f"wait {wait:.0f} s < cold {cold:.0f} s"
        return None, f"cold {cold:.0f} s <= wait {wait:.0f} s"

    async def huge_admit(self, ntok: int, body: dict | None, exclude=(), target: Block | None = None) -> bool:
        """wait until a block that can take the arrival holds fewer than --huge-per-block huge prompts (each takes the
        whole pool), at most --headroom-wait-max-s; FIFO (the head of the line goes first; a waiter that leaves --
        timeout, client gone -- leaves the line, so it never blocks the ones behind it). v6d target: wait for THAT block
        only (outside the line: it holds no one back), at most --pin-wait-max-s; False = it did not free in time"""
        if self.huge_cv is None:
            self.huge_cv = asyncio.Condition()
        t0 = time.time()
        if target is not None:
            self.pinned[target.idx] = self.pinned.get(target.idx, 0) + 1
            try:
                async with self.huge_cv:
                    while target.huge >= self.a.huge_per_block:
                        left = self.a.pin_wait_max_s - (time.time() - t0)
                        if left <= 0 or not target.up or target.draining:
                            self.pin_timeouts += 1
                            return False
                        try:
                            await asyncio.wait_for(self.huge_cv.wait(), timeout=min(left, max(1.0, 2 * self.a.poll_s)))
                        except asyncio.TimeoutError:
                            pass
                return True
            finally:  # (no await between here and choose(): the block is still free when it is taken)
                self.pinned[target.idx] -= 1
                if not self.pinned[target.idx]:
                    del self.pinned[target.idx]
        me = object()
        async with self.huge_cv:
            self.huge_line.append(me)
            queued = False
            try:
                while self.huge_line[0] is not me or not self.huge_free(ntok, body, exclude):
                    left = self.a.headroom_wait_max_s - (time.time() - t0)
                    if left <= 0 or not self.usable(exclude):
                        log.warning("no block free for a %d-token prompt after %.0f s: it queues in an engine", ntok,
                                    time.time() - t0)  # fmt: skip
                        break
                    if not queued:
                        queued = True
                        self.huge_waits += 1
                        self.huge_queue += 1
                    try:
                        await asyncio.wait_for(self.huge_cv.wait(), timeout=min(left, max(1.0, 2 * self.a.poll_s)))
                    except asyncio.TimeoutError:
                        pass
            finally:
                self.huge_line.remove(me)
                if queued:
                    self.huge_queue -= 1
                    self.huge_wait_s += time.time() - t0
                self.huge_cv.notify_all()
        return True

    async def huge_notify(self) -> None:
        if self.huge_cv is not None:
            async with self.huge_cv:
                self.huge_cv.notify_all()

    async def headroom_notify(self) -> None:
        if self.hr_cv is not None and self.hr_queue:
            async with self.hr_cv:
                self.hr_cv.notify_all()

    # -- routing -------------------------------------------------------------------------------------------------------
    def classify(self, body: dict | None) -> tuple:
        """(prompt tokens, prefix keys) of a request body (one sha1 pass: run in a thread for a big body)"""
        kind, payload = prompt_of(body or {})
        return (prompt_tokens(kind, payload, self.a.chars_per_token),
                prefix_keys(kind, payload, self.a.chars_per_token) if kind else [])  # fmt: skip

    def choose(self, body: dict | None, headers, exclude=(), pre: tuple | None = None, need: int | None = None,
               prefer: Block | None = None) -> tuple:
        """(block, reason, prompt tokens, prefix keys) -- block None when no block is usable. pre = classify(body);
        need (v6) = the arrival's KV reserve: only blocks whose headroom covers it, else the most headroom"""
        cand = self.usable(exclude)
        if not cand:
            return None, "none", 0, []
        body = body or {}
        ntok, keys = pre if pre is not None else self.classify(body)
        if pre is not None or body:  # v6b: only the blocks whose max_model_len takes it (v6e: and the total caps)
            if any(x.total_cap for x in cand):
                mx_ = _num(body.get("max_tokens", body.get("max_completion_tokens")))
                mx_ = int(mx_) if mx_ is not None and mx_ == mx_ else None
                tot = [x for x in cand if self.fits_total(x, ntok, mx_)]
                if tot and len(tot) < len(cand):
                    self.total_steered += 1
                elif not tot:
                    self.total_unfit += 1
            ctx = [x for x in cand if self.fits_ctx(x, ntok, body)]
            if ctx and len(ctx) < len(cand):
                self.ctx_steered += 1
                cand = ctx
            elif not ctx:  # v6c: no cap takes it BY THE ESTIMATE (text at --chars-per-token: a random-token 1M prompt
                # reads ~1.35M): every block of the largest cap stays a candidate -- the engine judges. (v6b kept ONE
                # block, the first of the largest cap: every such prompt piled onto it.)
                self.ctx_unfit += 1
                top = max(x.mml or 1 << 62 for x in cand)
                cand = [x for x in cand if (x.mml or 1 << 62) == top]
        huge = self.a.huge_per_block > 0 and ntok >= self.a.huge_tokens
        if huge and prefer is not None and prefer in cand and prefer.huge < self.a.huge_per_block:
            return prefer, "prefix-wait", ntok, keys  # v6d: it waited for its stored context's block
        if huge:  # v6c: a prompt >= --huge-tokens takes the whole pool: at most --huge-per-block per block
            free = [x for x in cand if x.huge < self.a.huge_per_block]
            open_ = [x for x in free if not self.pinned.get(x.idx)]  # v6d: leave a block a stored context waits for
            cand = open_ or free or cand  # (huge_admit waited for a free one; none left = a race: the engine queues it)
        hr_rank = lambda x: 0.0  # noqa: E731
        if need:
            short = ntok <= self.a.short_overtake_tokens
            fit = [x for x in cand if self.fits(x, need, ntok)]
            if fit and len(fit) < len(cand):
                self.hr_steered += 1
            if not fit:  # (headroom_admit waited it out): the most headroom -- v6d: EVERY block tied at it stays (load
                # decides below; v6-v6c's max() took the first index of a tie); v6e: a no-preempt block only if no other
                cand = [x for x in cand if x.res_cap < 0] or cand
                h = lambda x: float("inf") if self.headroom(x, short) is None else self.headroom(x, short)  # noqa: E731
                best = max(h(x) for x in cand)
                fit = [x for x in cand if h(x) == best]
            cand = fit
            hr_rank = lambda x: -(self.headroom(x, short) if self.headroom(x, short) is not None else float("inf"))  # noqa: E731
        ok = {b.idx for b in cand}
        sess = headers.get("X-Session-Id") or body.get("session_id") or body.get("user")
        if sess is not None:
            sess = str(sess)
            b = self.sessions.get(sess)
            if b is not None and b in ok:
                self.sessions.move_to_end(sess)
                return self.blocks[b], "session", ntok, keys
        is_long_cap = ntok > self.a.long_cap_tokens
        if not is_long_cap:  # v3: short / medium prompts avoid blocks with a long prefill in flight when they can
            free = [x for x in cand if x.capped_pf == 0]
            if free:
                cand = free
                ok = {b.idx for b in cand}
        least_load = min(x.inflight + x.waiting for x in cand)
        for cut, key in reversed(keys):
            b = self.prefixes.get(key)
            if b is None or b not in ok:
                continue
            # a SHORT shared prefix (a common system prompt) must not pin every request to one block: affinity needs
            # >= --affinity-min-tokens or half the prompt, and the block within --affinity-slack of the least loaded
            if cut < self.a.affinity_min_tokens and cut * 2 < ntok:
                break
            blk = self.blocks[b]
            if blk.inflight + blk.waiting > least_load + self.a.affinity_slack:
                self.affinity_overflow += 1
                break
            # v6c: affinity YIELDS to admission -- only a block that can admit the arrival NOW: its KV headroom covers
            # the reserve, and no long prompt is queued there; else the emptiest block takes it cold (00:11:38Z: eight
            # 1M prompts on one block, 4 min each, serialised, while two blocks sat idle)
            if need and not self.fits(blk, need, ntok):
                self.affinity_yield_headroom += 1
                break
            if blk.long_pf > 0 and (blk.waiting > 0 or blk.long_pf > 1):
                self.affinity_yield_long += 1
                break
            return blk, "prefix", ntok, keys
        self.rr += 1
        if ntok > self.a.long_tokens:
            # concurrent long arrivals spread first (a block mid-way through a long prefill shows 0 decoding streams),
            # then the fewest active decode streams
            # v6d (01:17:33Z (a)): among the blocks that fit, LOAD (router in-flight + engine queued) before headroom,
            # then the index -- a headroom tie (or a block a little roomier) no longer takes every arrival
            pick = min(cand, key=lambda x: (x.long_pf, x.inflight + x.waiting, hr_rank(x), x.decoding, x.running,
                                            (x.idx - self.rr) % len(self.blocks)))  # fmt: skip
            return pick, "long", ntok, keys
        top = max((x.mml or 1 << 62) for x in cand)
        cap_n = int(self.a.spec_prefer_max)  # v6c: the preference is BOUNDED -- a smaller-cap (spec) block is preferred
        # only while it holds fewer than --spec-prefer-max open requests (its decode width), then load decides alone
        pref = (lambda x: self.a.spec_prefer_slack if (x.mml or 1 << 62) < top and (cap_n <= 0 or x.inflight < cap_n)
                else 0) if ntok <= self.a.spec_prefer_tokens else (lambda x: 0)  # v6b: shorts prefer the smaller cap
        pick = min(cand, key=lambda x: (x.long_pf > 0, x.inflight + x.waiting - pref(x), x.decoding,
                                        (x.idx - self.rr) % len(self.blocks)))  # fmt: skip
        return pick, "least", ntok, keys

    def remember(self, block: Block, body: dict | None, headers, keys: list) -> None:
        sess = headers.get("X-Session-Id") or (body or {}).get("session_id") or (body or {}).get("user")
        if sess is not None:
            self.sessions[str(sess)] = block.idx
            self.sessions.move_to_end(str(sess))
        for _cut, key in keys:
            self.prefixes[key] = block.idx
            self.prefixes.move_to_end(key)
        for d in (self.sessions, self.prefixes):
            while len(d) > self.max_entries:
                d.popitem(last=False)

    # -- proxy ---------------------------------------------------------------------------------------------------------
    @staticmethod
    async def _dumps(body: dict, size: int) -> bytes:
        """re-serialize a rewritten body (a 1M-token prompt is ~7 MB: off the event loop)"""
        return json.dumps(body).encode() if size < (1 << 20) else await asyncio.to_thread(lambda: json.dumps(body).encode())

    async def forward(self, request: web.Request) -> web.StreamResponse:
        raw = await request.read()
        t_in = time.time()
        body = None
        if raw and request.method == "POST":
            try:  # (a 1M-token prompt is ~7 MB of JSON: parsed off the event loop, the other streams keep flowing)
                body = json.loads(raw) if len(raw) < (1 << 20) else await asyncio.to_thread(json.loads, raw)
            except ValueError:
                body = None
        routed = request.path in ("/v1/completions", "/v1/chat/completions")
        tried: set = set()
        stripped: list = []
        if routed and isinstance(body, dict) and self.a.host_sampling != "allow":  # v5: no host-sampled decode step
            found = host_sampling_params(body, self.a.device_top_k)
            if found:
                self.host_sampling_hits[",".join(found)] = self.host_sampling_hits.get(",".join(found), 0) + 1
                if self.a.host_sampling == "reject":
                    return web.json_response({"error": {
                        "message": f"parameters {found} force host sampling, which this fleet does not run (the "
                                   f"width-8 D2H hazard); resend without them", "type": "router_host_sampling",
                        "param": found[0]}}, status=400)  # fmt: skip
                body = strip_host_sampling(body, found)
                raw = await self._dumps(body, len(raw))
                stripped = found
            clamped = clamp_top_k(body, self.a.device_top_k)
            if clamped is not None:
                body, raw = clamped, await self._dumps(clamped, len(raw))
                stripped = stripped + ["top_k=%d" % self.a.device_top_k]
                self.host_sampling_hits["top_k_clamped"] = self.host_sampling_hits.get("top_k_clamped", 0) + 1
        pre = None
        if routed:  # classify BEFORE admission (v4): the token count + the prefix keys (a thread for a big body)
            pre = await asyncio.to_thread(self.classify, body) if raw and len(raw) >= (1 << 20) else self.classify(body)
        if routed and pre and isinstance(body, dict) and (self.a.default_max_tokens > 0 or self.a.max_max_tokens > 0):
            # v6f (program 2026-10-29T02:32:09Z): under --block-reserve-cap 0 a request without max_tokens reserves
            # its block's whole cap (vLLM's default max_tokens = cap - prompt): one such user per block. The router
            # fills a default (--default-max-tokens) -- not for a prompt that nearly fills the largest cap by the
            # estimate (it takes a whole block anyway; a filled default could push it over the cap) -- and bounds an
            # explicit one (--max-max-tokens: clamp, or 400 naming the limit)
            given = [k for k in ("max_tokens", "max_completion_tokens") if body.get(k) is not None]
            new, tag = None, None
            top = max([b.mml for b in self.blocks if b.mml] or [int(self.a.max_model_len)])
            if not given and self.a.default_max_tokens > 0:
                if pre[0] * max(1.0, self.a.ctx_margin) + self.a.default_max_tokens <= top:
                    new, tag = dict(body, max_tokens=int(self.a.default_max_tokens)), "max_tokens=%d(default)"
                    self.mt_defaulted += 1
            elif self.a.max_max_tokens > 0:
                over = [k for k in given if (_num(body.get(k)) or 0) > self.a.max_max_tokens]
                if over and self.a.max_tokens_over == "reject":
                    self.mt_rejected += 1
                    return web.json_response({"error": {
                        "message": f"{over[0]} {body.get(over[0])} exceeds this endpoint's limit of "
                                   f"{self.a.max_max_tokens} output tokens; resend with {over[0]} <= "
                                   f"{self.a.max_max_tokens}", "type": "router_max_tokens", "param": over[0]}},
                        status=400)  # fmt: skip
                if over:
                    new, tag = dict(body, **{k: int(self.a.max_max_tokens) for k in over}), "max_tokens=%d(clamped)"
                    self.mt_clamped += 1
            if new is not None:
                body, raw = new, await self._dumps(new, len(raw))
                stripped = stripped + [tag % (self.a.default_max_tokens if "default" in tag else self.a.max_max_tokens)]
        huge = bool(routed and pre and self.a.huge_per_block > 0 and pre[0] >= self.a.huge_tokens)
        # the global long cap (N - 1 blocks with a long prefill) for the prompts between --long-cap-tokens and
        # --huge-tokens; a huge prompt is bounded per block instead (v6c: 8 x 1M at c=4 keeps 4 blocks busy)
        capped = bool(routed and pre and self.long_cap() > 0 and pre[0] > self.a.long_cap_tokens and not huge)
        admitted = False
        need = 0
        pin = None
        try:
            if capped:
                await self.long_admit()  # counts it at once (v4), FIFO behind the cap
                admitted = True
            if routed and pre and self.a.headroom_routing == "on":  # v6: the KV reserve; wait for a block that fits
                need = self.need_of(pre[0], body if isinstance(body, dict) else None)
                await self.headroom_admit(need, pre[0])
            if huge:  # (the last await before choose(): the block it picks is still free)
                bd = body if isinstance(body, dict) else None
                pin, why = self.pin_decision(pre[0], pre[1], bd)
                if pin is not None:
                    self.pin_waits += 1
                    if not await self.huge_admit(pre[0], bd, target=pin):
                        pin = None  # its block did not free in time: cold, like any huge arrival
                elif why.startswith("cold"):
                    self.pin_cold += 1
                if pin is None:
                    await self.huge_admit(pre[0], bd)
        except BaseException:  # v6d: the client left while it waited: give the long-cap slot back (it leaked before)
            if admitted:
                self.long_cancelled = getattr(self, "long_cancelled", 0) + 1
                await self.long_release(counted=True)
            raise
        if routed and (request.transport is None or request.transport.is_closing()):
            # v6d: the client left while the arrival waited (aiohttp does not cancel a handler on disconnect): routed,
            # it would cost a whole prefill for nobody -- the first write fails only after the first token
            self.gone_before_route = getattr(self, "gone_before_route", 0) + 1
            if admitted:
                await self.long_release(counted=True)
            return web.Response(status=499)
        while True:
            # (no await between the choice and the counters below: v4)
            block, reason, ntok, keys = self.choose(body if routed else None, request.headers, tried, pre=pre,
                                                    need=need or None, prefer=pin)  # fmt: skip
            if block is None:
                if admitted:
                    admitted = False
                    await self.long_release(counted=True)
                return web.json_response({"error": {"message": "no block available", "type": "router"}}, status=503)
            if not routed:
                reason = "aux"
            tried.add(block.idx)
            hdrs = {k: v for k, v in request.headers.items() if k.lower() not in HOP}
            url = block.url + request.path_qs
            block.inflight += 1
            hrec = None
            if huge:
                block.huge += 1
                mx = _num((body or {}).get("max_tokens", (body or {}).get("max_completion_tokens")))
                hrec = {"t": time.time(), "ntok": ntok, "max_tokens": int(mx) if mx is not None and mx == mx else None,
                        "t_first": None, "out": 0}  # fmt: skip
                block.huge_reqs[id(hrec)] = hrec
            held = need_on(need, block)  # v6e: this block's reserve
            block.reserved += held  # v6: the ledger (released when the request ends)
            block.since_poll += held
            is_long = routed and ntok > self.a.long_tokens
            if is_long:
                block.long_pf += 1
            if capped:
                block.capped_pf += 1
            started = False
            out = None
            status = None
            t_first = None
            err = None
            try:
                async with self.http.request(request.method, url, data=raw if raw else None, headers=hdrs) as up:
                    status = up.status
                    if routed and up.status == 400 and self.a.ctx_retry == "on":
                        # v6f (kit v3 trace check, 2026-10-29T02:26Z): the 4-chars-per-token estimate reads agent
                        # traces (code, JSON) low -- a 133,803-token prompt went to the 131,072 block, which refused
                        # it, and one refused root request aborts a whole AgentX warm-up. A context refusal (nothing
                        # streamed yet) retries on a LARGER-cap block; none left: the block's own 400 goes back.
                        buf = await up.content.read(1 << 16)
                        if CTX_REFUSAL.search(buf.decode("utf-8", "replace")):
                            block.errors["ctx_refused"] = block.errors.get("ctx_refused", 0) + 1
                            self.ctx_refused += 1
                            for x in self.blocks:  # (a cap no larger than the refusing block's refuses it too)
                                if x.mml and block.mml and x.mml <= block.mml:
                                    tried.add(x.idx)
                            if self.usable(tried):
                                self.ctx_retried += 1
                                err = "ctx-refused"
                                log.warning("block %d refused a ~%d-token prompt for its context (%s): retrying on a "
                                            "larger cap", block.idx, ntok, block.mml)  # fmt: skip
                                continue
                        resp = web.Response(status=400, body=buf, content_type="application/json")
                        resp.headers["X-Routed-Block"] = str(block.idx)
                        resp.headers["X-Route-Reason"] = reason
                        return resp
                    if routed:
                        self.remember(block, body, request.headers, keys)
                    block.routed[reason] = block.routed.get(reason, 0) + 1
                    out = web.StreamResponse(status=up.status, headers={k: v for k, v in up.headers.items()
                                                                        if k.lower() not in HOP})  # fmt: skip
                    out.headers["X-Routed-Block"] = str(block.idx)
                    out.headers["X-Route-Reason"] = reason
                    if stripped:
                        out.headers["X-Router-Stripped"] = ",".join(stripped)
                    await out.prepare(request)
                    started = True
                    async for chunk in up.content.iter_any():
                        if hrec is not None:  # v6d: the block's free-time estimate (SSE: ~1 token a chunk)
                            hrec["out"] += 1
                            if hrec["t_first"] is None:
                                hrec["t_first"] = time.time()
                        if t_first is None:
                            t_first = time.time()
                            block.decoding += 1
                            if is_long:
                                block.long_pf -= 1
                            if capped:
                                block.capped_pf -= 1
                                if admitted:
                                    admitted = False
                                    await self.long_release(counted=True)
                        await out.write(chunk)
                    await out.write_eof()
                    return out
            except ClientConnectorError as e:  # nothing reached the client yet: another block
                block.errors["connect"] = block.errors.get("connect", 0) + 1
                block.fails += 1
                if block.fails >= self.a.down_after:
                    block.up = False
                err = f"connect: {e}"
                if not started:
                    log.warning("block %d connect failed (%s): retrying elsewhere", block.idx, e)
                    continue
                raise
            except ConnectionResetError as e:  # the client went away: leaving the block closes the upstream stream
                block.errors["client_gone"] = block.errors.get("client_gone", 0) + 1
                err = f"client gone: {type(e).__name__}"
                return out if out is not None else web.Response(status=499)
            except asyncio.CancelledError:
                block.errors["client_gone"] = block.errors.get("client_gone", 0) + 1
                err = "client gone: cancelled"
                raise
            except Exception as e:  # noqa: BLE001
                block.errors["upstream"] = block.errors.get("upstream", 0) + 1
                err = f"{type(e).__name__}: {e}"[:200]
                if not started:
                    return web.json_response({"error": {"message": err, "type": "router_upstream"}}, status=502)
                raise
            finally:
                block.inflight -= 1
                if huge:
                    block.huge -= 1
                    if hrec is not None:
                        block.huge_reqs.pop(id(hrec), None)
                    await self.huge_notify()
                block.reserved -= held
                if held:
                    await self.headroom_notify()
                if is_long and t_first is None:
                    block.long_pf -= 1
                if capped and t_first is None:
                    block.capped_pf -= 1
                    if admitted and not (isinstance(err, str) and err.startswith(("connect", "ctx-refused"))):
                        admitted = False  # (a connect failure / v6f context refusal retries elsewhere: it carries over)
                        await self.long_release(counted=True)
                if t_first is not None:
                    block.decoding -= 1
                    if routed:
                        block.ttft_sum += t_first - t_in
                        block.ttft_n += 1
                if routed and (started or err):
                    self.log_access({"t": round(t_in, 3), "id": request.headers.get("X-Request-Id"), "path": request.path,
                                     "block": block.idx, "reason": reason, "prompt_tokens": ntok,
                                     "stream": bool((body or {}).get("stream")), "status": status,
                                     "ttft_s": round(t_first - t_in, 3) if t_first else None,
                                     "total_s": round(time.time() - t_in, 3), "error": err})  # fmt: skip

    def log_access(self, row: dict) -> None:
        if self.access is not None:
            self.access.write(json.dumps(row) + "\n")
            self.access.flush()

    # -- control / observability ----------------------------------------------------------------------------------------
    async def health(self, _req: web.Request) -> web.Response:
        up = [b.idx for b in self.blocks if b.up]
        return web.json_response({"version": VERSION, "up": up, "draining": [b.idx for b in self.blocks if b.draining],
                                  "blocks": [b.view() for b in self.blocks]}, status=200 if up else 503)  # fmt: skip

    async def status(self, _req: web.Request) -> web.Response:
        return web.json_response({"version": VERSION, "uptime_s": round(time.time() - self.t0, 1),
                                  "blocks": [b.view() for b in self.blocks], "sessions": len(self.sessions),
                                  "prefixes": len(self.prefixes), "long_tokens": self.a.long_tokens})  # fmt: skip

    async def metrics(self, _req: web.Request) -> web.Response:
        L = [f"# {VERSION}"]
        for b in self.blocks:
            lab = f'block="{b.idx}"'
            L += [f"dp4router_block_up{{{lab}}} {int(b.up)}", f"dp4router_block_draining{{{lab}}} {int(b.draining)}",
                  f"dp4router_block_long_prefills{{{lab}}} {b.long_pf}",
                  f"dp4router_block_inflight{{{lab}}} {b.inflight}", f"dp4router_block_decoding{{{lab}}} {b.decoding}",
                  f"dp4router_block_queue_depth{{{lab}}} {b.waiting}", f"dp4router_block_running{{{lab}}} {b.running}",
                  f"dp4router_block_ttft_seconds_sum{{{lab}}} {round(b.ttft_sum, 4)}",
                  f"dp4router_block_ttft_seconds_count{{{lab}}} {b.ttft_n}"]  # fmt: skip
            if b.kv_usage is not None:
                L.append(f"dp4router_block_kv_usage{{{lab}}} {b.kv_usage}")
            L.append(f"dp4router_block_reserved_tokens{{{lab}}} {b.reserved}")  # v6: the ledger
            if b.mml is not None:
                L.append(f"dp4router_block_max_model_len{{{lab}}} {b.mml}")  # v6b
            L.append(f"dp4router_block_reserve_cap{{{lab}}} {b.res_cap}")  # v6e
            L.append(f"dp4router_block_max_total_tokens{{{lab}}} {b.total_cap}")
            h = self.headroom(b)
            if h is not None:
                L.append(f"dp4router_block_headroom_tokens{{{lab}}} {int(h)}")
            if b.adm_headroom is not None:
                L.append(f"dp4router_block_engine_headroom_tokens{{{lab}}} {int(b.adm_headroom)}")
            for r, n in b.routed.items():
                L.append(f'dp4router_routed_total{{{lab},reason="{r}"}} {n}')
            for k, n in b.errors.items():
                L.append(f'dp4router_errors_total{{{lab},kind="{k}"}} {n}')
        L.append(f"dp4router_affinity_sessions {len(self.sessions)}")
        L.append(f"dp4router_affinity_prefixes {len(self.prefixes)}")
        L.append(f"dp4router_affinity_overflow_total {self.affinity_overflow}")
        L.append(f"dp4router_affinity_yield_total{{why=\"headroom\"}} {self.affinity_yield_headroom}")
        L.append(f"dp4router_affinity_yield_total{{why=\"long_queued\"}} {self.affinity_yield_long}")
        L.append(f"dp4router_ctx_unfit_total {self.ctx_unfit}")
        L.append(f"dp4router_headroom_unsatisfiable_total {getattr(self, 'hr_unsat', 0)}")
        L.append(f"dp4router_headroom_nothing_open_total {getattr(self, 'hr_nothing_open', 0)}")  # v6d
        L.append(f"dp4router_client_gone_before_route_total {getattr(self, 'gone_before_route', 0)}")  # v6d
        L.append(f"dp4router_total_cap_steered_total {self.total_steered}")  # v6e
        L.append(f"dp4router_ctx_refused_total {self.ctx_refused}")  # v6f
        L.append(f'dp4router_max_tokens_total{{action="defaulted"}} {self.mt_defaulted}')
        L.append(f'dp4router_max_tokens_total{{action="clamped"}} {self.mt_clamped}')
        L.append(f'dp4router_max_tokens_total{{action="rejected"}} {self.mt_rejected}')
        L.append(f"dp4router_ctx_retried_total {self.ctx_retried}")
        L.append(f"dp4router_total_cap_unfit_total {self.total_unfit}")
        L.append(f"dp4router_huge_waits_total {self.huge_waits}")
        L.append(f'dp4router_pin_total{{decision="wait"}} {self.pin_waits}')
        L.append(f'dp4router_pin_total{{decision="cold"}} {self.pin_cold}')
        L.append(f'dp4router_pin_total{{decision="timeout"}} {self.pin_timeouts}')
        L.append(f"dp4router_huge_queue_depth {self.huge_queue}")
        L.append(f"dp4router_huge_wait_seconds_sum {round(self.huge_wait_s, 3)}")
        for b in self.blocks:
            L.append(f'dp4router_block_huge_inflight{{block="{b.idx}"}} {b.huge}')
        L.append(f"dp4router_long_cap {self.long_cap()}")
        L.append(f"dp4router_long_cancelled_total {getattr(self, 'long_cancelled', 0)}")
        L.append(f"dp4router_long_admitted {self.long_admitted}")
        for k, n in self.host_sampling_hits.items():
            L.append(f'dp4router_host_sampling_total{{params="{k}",action="{self.a.host_sampling}"}} {n}')
        L.append(f"dp4router_long_queue_depth {self.long_queue}")
        L.append(f"dp4router_long_queued_total {self.long_queued_total}")
        L.append(f"dp4router_long_wait_seconds_sum {round(self.long_wait_s, 3)}")
        L.append(f"dp4router_headroom_queue_depth {self.hr_queue}")
        L.append(f"dp4router_headroom_waits_total {self.hr_waits}")
        L.append(f"dp4router_headroom_wait_seconds_sum {round(self.hr_wait_s, 3)}")
        L.append(f"dp4router_headroom_fallbacks_total {self.hr_fallbacks}")
        L.append(f"dp4router_headroom_steered_total {self.hr_steered}")
        L.append(f"dp4router_ctx_steered_total {self.ctx_steered}")
        return web.Response(text="\n".join(L) + "\n", content_type="text/plain")

    def _block_arg(self, req: web.Request) -> Block | None:
        try:
            i = int(req.query.get("block", ""))
        except ValueError:
            return None
        return self.blocks[i] if 0 <= i < len(self.blocks) else None

    async def drain(self, req: web.Request) -> web.Response:
        b = self._block_arg(req)
        if b is None:
            return web.json_response({"error": "block=<index> required"}, status=400)
        b.draining = True
        log.warning("block %d DRAINING (in-flight %d)", b.idx, b.inflight)
        wait_s = float(req.query.get("wait_s", "0") or 0)
        t0 = time.time()
        while wait_s > 0 and b.inflight > 0 and time.time() - t0 < wait_s:
            await asyncio.sleep(0.2)
        return web.json_response({"block": b.idx, "draining": True, "inflight": b.inflight,
                                  "drained": b.inflight == 0, "waited_s": round(time.time() - t0, 2)})  # fmt: skip

    async def retire(self, _req: web.Request) -> web.Response:
        """v4 (--replace): stop LISTENING (a new instance on the same port takes the new connections), finish every
        in-flight stream, then exit the process"""
        if self.retiring:
            return web.json_response({"retiring": True})
        self.retiring = True
        for site in list(self.sites):
            await site.stop()
        log.warning("RETIRING: listener closed; exiting once %d in-flight streams end", sum(b.inflight for b in self.blocks))

        async def finish():
            t0 = time.time()
            while sum(b.inflight for b in self.blocks) > 0 and time.time() - t0 < self.a.retire_max_s:
                await asyncio.sleep(0.5)
            log.warning("RETIRED after %.1f s", time.time() - t0)
            if self.access is not None:
                self.access.flush()
            os._exit(0)

        asyncio.get_running_loop().create_task(finish())
        return web.json_response({"retiring": True, "inflight": sum(b.inflight for b in self.blocks)})

    async def undrain(self, req: web.Request) -> web.Response:
        b = self._block_arg(req)
        if b is None:
            return web.json_response({"error": "block=<index> required"}, status=400)
        b.draining = False
        log.warning("block %d back in rotation", b.idx)
        return web.json_response({"block": b.idx, "draining": False})


async def build_app(a: argparse.Namespace) -> web.Application:
    r = Router(a)
    if a.log_dir:
        os.makedirs(a.log_dir, exist_ok=True)
        r.access = open(os.path.join(a.log_dir, "router_access.jsonl"), "a")
    app = web.Application(client_max_size=1 << 30)  # (a 1M-token prompt body is ~7 MB of JSON)
    app["router"] = r

    async def on_start(_app):
        r.http = ClientSession(connector=TCPConnector(limit=0, force_close=False),
                               timeout=ClientTimeout(total=None, sock_connect=a.connect_timeout_s,
                                                     sock_read=a.read_timeout_s))  # fmt: skip
        await r.poll_once()
        _app["poller"] = asyncio.create_task(r.poll_loop())
        log.warning("%s on %s:%d -> %s", VERSION, a.host, a.port, ", ".join(b.url for b in r.blocks))

    async def on_stop(_app):
        _app["poller"].cancel()
        await r.http.close()
        if r.access is not None:
            r.access.close()

    app.on_startup.append(on_start)
    app.on_cleanup.append(on_stop)
    app.router.add_get("/health", r.health)
    app.router.add_get("/metrics", r.metrics)
    app.router.add_get("/router/status", r.status)
    app.router.add_post("/router/drain", r.drain)
    app.router.add_post("/router/undrain", r.undrain)
    app.router.add_post("/router/retire", r.retire)
    for path in ("/v1/completions", "/v1/chat/completions", "/tokenize", "/detokenize"):
        app.router.add_post(path, r.forward)
    app.router.add_get("/v1/models", r.forward)
    return app


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=VERSION)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--blocks", default=",".join(f"http://127.0.0.1:{p}" for p in range(8100, 8104)))
    ap.add_argument("--long-tokens", type=int, default=32768)
    ap.add_argument("--chars-per-token", type=float, default=4.0)
    ap.add_argument("--poll-s", type=float, default=1.0)
    ap.add_argument("--health-timeout-s", type=float, default=3.0)
    ap.add_argument("--down-after", type=int, default=3)
    ap.add_argument("--connect-timeout-s", type=float, default=10.0)
    ap.add_argument("--read-timeout-s", type=float, default=7200.0)  # (a 1M prefill's first byte)
    ap.add_argument("--max-affinity-entries", type=int, default=200000)
    ap.add_argument("--affinity-min-tokens", type=int, default=4096)
    ap.add_argument("--affinity-slack", type=int, default=4)
    ap.add_argument("--long-cap-tokens", type=int, default=131072)
    ap.add_argument("--long-cap", type=int, default=-1, help="blocks with a long prefill at once (-1: N - 1; 0: no cap)")
    ap.add_argument("--replace", help="the old router's URL: take its port over (v4: /router/retire; else wait for its "
                                      "in-flight streams to end), binding with SO_REUSEADDR / SO_REUSEPORT")
    ap.add_argument("--replace-wait-s", type=float, default=600.0)
    ap.add_argument("--host-sampling", choices=("reject", "strip", "allow"), default="reject")
    ap.add_argument("--device-top-k", type=int, default=32)
    ap.add_argument("--retire-max-s", type=float, default=3600.0)
    ap.add_argument("--headroom-routing", choices=("on", "off"), default="on")  # v6
    ap.add_argument("--pool-tokens", type=int, default=0, help="KV pool per block when a block publishes none (0: unknown)")
    ap.add_argument("--kv-block-size", type=int, default=64)
    ap.add_argument("--admit-reserve-cap", type=int, default=32768)
    ap.add_argument("--max-model-len", type=int, default=1048576)
    ap.add_argument("--short-overtake-tokens", type=int, default=32768)
    ap.add_argument("--headroom-wait-max-s", type=float, default=1800.0)
    ap.add_argument("--block-max-model-len", default="", help="v6b: per-block caps, comma list (0 / empty = /v1/models)")
    ap.add_argument("--default-max-tokens", type=int, default=0, help="v6f: fill max_tokens = N into a request that "
                    "sends none (0: off; not for a prompt that nearly fills the largest cap)")
    ap.add_argument("--max-max-tokens", type=int, default=0, help="v6f: the largest max_tokens / max_completion_tokens "
                    "a request may ask (0: no limit); over it -> --max-tokens-over")
    ap.add_argument("--max-tokens-over", choices=("clamp", "reject"), default="clamp", help="v6f: clamp to "
                    "--max-max-tokens (header X-Router-Stripped names it) or answer 400 naming the limit")
    ap.add_argument("--ctx-retry", choices=("on", "off"), default="on", help="v6f: a block's 400 for a prompt over its "
                    "context (nothing streamed yet) retries on a larger-cap block")
    ap.add_argument("--ctx-margin", type=float, default=1.25, help="v6f: a block whose cap is below the largest takes "
                    "a prompt only if (estimate x this) + its reserve fits (agent traces tokenize at ~3 chars/token)")
    ap.add_argument("--block-reserve-cap", default="", help="v6e: per-block KV reserve cap, comma list or ONE value for "
                    "every block (-1 / empty: --admit-reserve-cap; 0: none = the full prompt + max_tokens, a no-preempt "
                    "block: the router never sends it more than its pool holds at full generation)")
    ap.add_argument("--block-max-total-tokens", default="", help="v6e: per-block cap on prompt + max_tokens (no "
                    "max_tokens: the block's max_model_len), comma list or ONE value (0 / empty: none); over it -> "
                    "another block")
    ap.add_argument("--spec-prefer-tokens", type=int, default=32768)
    ap.add_argument("--spec-prefer-slack", type=int, default=4)
    ap.add_argument("--spec-prefer-max", type=int, default=8, help="v6c: a smaller-cap block is preferred only while it "
                    "holds fewer open requests than this (0: unbounded)")
    ap.add_argument("--huge-tokens", type=int, default=524288, help="v6c: prompts at or above this take a whole block")
    ap.add_argument("--huge-per-block", type=int, default=1, help="v6c: open huge prompts per block (0: off = v6b)")
    ap.add_argument("--pin-wait", choices=("on", "off"), default="on", help="v6d: a huge arrival with a stored context "
                    "waits for its block when expected free time + --hit-reload-s < a cold prefill elsewhere")
    ap.add_argument("--prefill-s-per-mtok", type=float, default=262.0, help="v6d: cold prefill seconds per 1M tokens "
                    "(the lone 1M on the frozen form: 262.5 s)")
    ap.add_argument("--decode-tok-s", type=float, default=50.0, help="v6d: a lone long user's decode rate (52 after a 1M)")
    ap.add_argument("--hit-reload-s", type=float, default=30.0, help="v6d: a stored-context hit's reload + tail on its "
                    "own block (local tiers ~3-6 s + the KDA state + the tail chunk; 9-80 s seen)")
    ap.add_argument("--pin-wait-max-s", type=float, default=900.0, help="v6d: the most a pinned arrival waits")
    ap.add_argument("--log-dir")
    a = ap.parse_args(argv)
    a.blocks = [u.strip() for u in a.blocks.split(",") if u.strip()]
    return a


async def _replace_old(a: argparse.Namespace) -> None:
    """--replace: ask the old router to retire (v4); an older one: wait until its in-flight streams are 0"""
    import json as _json  # noqa: PLC0415
    import urllib.request  # noqa: PLC0415

    def call(path: str, method: str = "GET"):
        req = urllib.request.Request(a.replace.rstrip("/") + path, method=method, data=b"" if method == "POST" else None)
        with urllib.request.urlopen(req, timeout=10) as r:
            return _json.loads(r.read() or b"{}")

    try:
        got = await asyncio.to_thread(call, "/router/retire", "POST")
        log.warning("--replace: the old router retires (%s)", got)
        return
    except Exception as e:  # noqa: BLE001 -- an older router (no /router/retire): wait for it to go idle
        log.warning("--replace: no /router/retire on %s (%s): waiting for its in-flight streams", a.replace, e)
    t0 = time.time()
    while time.time() - t0 < a.replace_wait_s:
        try:
            st = await asyncio.to_thread(call, "/router/status")
            n = sum(int(b.get("inflight", 0)) for b in st.get("blocks", []))
        except Exception:  # noqa: BLE001 -- gone
            return
        if n == 0:
            log.warning("--replace: the old router is idle; stop it now (this instance binds as soon as the port frees)")
            return
        await asyncio.sleep(1.0)


async def _serve(a: argparse.Namespace) -> None:
    app = await build_app(a)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    if a.replace:
        await _replace_old(a)
    t0 = time.time()
    while True:  # SO_REUSEADDR (TIME_WAIT) + SO_REUSEPORT (an old v4 still finishing its streams)
        try:
            site = web.TCPSite(runner, a.host, a.port, reuse_address=True, reuse_port=True)
            await site.start()
            break
        except OSError as e:
            if not a.replace or time.time() - t0 > a.replace_wait_s:
                raise
            log.warning("port %d busy (%s): retrying", a.port, e)
            await asyncio.sleep(1.0)
    app["router"].sites.append(site)
    log.warning("%s listening on %s:%d", VERSION, a.host, a.port)
    while True:
        await asyncio.sleep(3600)


def main(argv=None) -> int:
    a = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s dp4router %(levelname)s %(message)s",
                        stream=sys.stderr)  # fmt: skip
    try:
        asyncio.run(_serve(a))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
