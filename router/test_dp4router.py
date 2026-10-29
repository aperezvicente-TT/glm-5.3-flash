"""dp4router.py against four fake block servers (aiohttp, in-process): least-loaded spread, prefix and session
affinity, the long-prompt rule, a down block (skipped; a connect failure retried elsewhere), drain / undrain, /health,
/metrics, streaming passthrough, and a client disconnect closing the upstream stream.
usage: <the vLLM venv's python> -I test_dp4router.py"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import socket
import sys
import time

import aiohttp
from aiohttp import web

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("dp4router", os.path.join(_HERE, "dp4router.py"))
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class Fake:
    """one block server: /health, /metrics, /v1/models, /tokenize, /v1/completions (SSE, one token per DEC s)"""

    DEC = 0.01
    PF_US = 0.0
    WAITING = 0  # reported as vllm:num_requests_waiting (the router's poller reads it)
    ADM = None  # v6: {"pool", "headroom", "queued"} published as glm53_admit_*_tokens (plugin v3), None = not published
    MML = None  # v6b: the max_model_len /v1/models reports (None = not reported)
    HIT_S = None  # v6d: a prompt this block saw before is a stored-context hit (this many seconds, not PF_US x len)
    RES = None  # v6d: with ADM, one active request's reserve: the gauge is LIVE while busy (pool - active x RES) and
    # FROZEN at its last busy value when idle -- plugin v3 publishes glm53_admit_* from schedule() only
    RUNNING = None  # v6d: report this running count, not the live one (vLLM's gauge after an abort: frozen)
    DECODE_MAX = None  # v6e: decode at most this many tokens (a body's max_tokens is what the router reserves on)
    CTX_CPT = None  # v6f: refuse (400, vLLM's text) a prompt over MML -- text counted at this many chars per token

    def __init__(self, idx: int):
        self.idx, self.port, self.runner = idx, free_port(), None
        self.seen: list = []  # (prompt len, X-Request-Id)
        self.active = 0
        self.cancelled = 0
        self.healthy = True

    async def start(self):
        app = web.Application(client_max_size=1 << 28)
        app.router.add_get("/health", self.health)
        app.router.add_get("/metrics", self.metrics)
        app.router.add_get("/v1/models", lambda r: web.json_response(
            {"data": [dict({"id": "glm"}, **({"max_model_len": self.MML} if self.MML else {}))]}))
        app.router.add_post("/tokenize", lambda r: web.json_response({"tokens": [1, 2, 3], "block": self.idx}))
        app.router.add_post("/v1/completions", self.completions)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        await web.TCPSite(self.runner, "127.0.0.1", self.port).start()

    async def stop(self):
        await self.runner.cleanup()

    async def health(self, _r):
        return web.Response(status=200 if self.healthy else 503)

    async def metrics(self, _r):
        if self.ADM and self.RES is not None and self.active > 0:
            self.ADM = dict(self.ADM, headroom=self.ADM["pool"] - self.active * self.RES, queued=0)
        adm = "".join(f"glm53_admit_{k}_tokens {float(v)}\n" for k, v in (self.ADM or {}).items())
        running = self.active if self.RUNNING is None else self.RUNNING
        return web.Response(text=f"vllm:num_requests_running {running}\nvllm:num_requests_waiting {self.WAITING}\n"
                                 f"vllm:kv_cache_usage_perc 0.25\n" + adm)  # fmt: skip

    async def completions(self, r):
        body = await r.json()
        p = body.get("prompt")
        if self.CTX_CPT and self.MML:
            n = len(p) if isinstance(p, list) else int(len(p or "") / self.CTX_CPT)
            if n + int(body.get("max_tokens", 4)) > self.MML:
                self.refused = getattr(self, "refused", 0) + 1
                return web.json_response({"error": {"message": f"This model's maximum context length is {self.MML} "
                                                    f"tokens. However, your request has {n} input tokens.",
                                                    "type": "BadRequestError"}}, status=400)  # fmt: skip
        self.seen.append((len(p) if isinstance(p, (list, str)) else 0, r.headers.get("X-Request-Id")))
        self.bodies = getattr(self, "bodies", [])
        self.bodies.append({k: body.get(k) for k in ("top_k", "logprobs", "temperature", "max_tokens")})
        out = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await out.prepare(r)
        self.active += 1
        try:
            if self.HIT_S is not None and isinstance(p, list):  # v6d: a stored context reloads, an unseen one prefills
                key = hash(tuple(p))
                stored = getattr(self, "stored", set())
                await asyncio.sleep(self.HIT_S if key in stored else len(p) * self.PF_US)
                stored.add(key)
                self.stored = stored
            elif self.PF_US and isinstance(p, list):
                await asyncio.sleep(len(p) * self.PF_US)  # the prefill (v3 tests)
            for _ in range(min(int(body.get("max_tokens", 4)), self.DECODE_MAX or 1 << 30)):
                await asyncio.sleep(self.DEC)
                await out.write(b'data: {"choices": [{"text": "x", "token_ids": [5]}]}\n\n')
            await out.write(b"data: [DONE]\n\n")
        except (ConnectionResetError, asyncio.CancelledError):
            self.cancelled += 1
            return out
        finally:
            self.active -= 1
        return out


async def start_router(fakes, **kw):
    a = R.parse_args(["--port", str(free_port()), "--blocks", ",".join(f"http://127.0.0.1:{f.port}" for f in fakes),
                      "--poll-s", "0.1", "--down-after", "1", "--health-timeout-s", "0.5"])  # fmt: skip
    for k, v in kw.items():
        setattr(a, k, v)
    app = await R.build_app(a)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", a.port).start()
    return runner, f"http://127.0.0.1:{a.port}", app["router"]


async def complete(s, url, prompt, max_tokens=3, headers=None, stream=True):
    t0 = time.time()
    async with s.post(url + "/v1/completions", json={"model": "glm", "prompt": prompt, "max_tokens": max_tokens,
                                                     "stream": stream}, headers=headers or {}) as r:  # fmt: skip
        times = []
        async for _chunk in r.content.iter_any():
            times.append(time.time())
        return int(r.headers.get("X-Routed-Block", -1)), r.headers.get("X-Route-Reason"), r.status, times, t0


async def scenario():
    fakes = [Fake(i) for i in range(4)]
    for f in fakes:
        await f.start()
    runner, url, router = await start_router(fakes)
    results = {}
    try:
        async with aiohttp.ClientSession() as s:
            # 1. least-loaded spread: 8 concurrent new prompts -> 2 per block
            got = await asyncio.gather(*(complete(s, url, [1000 + i] * 300, max_tokens=20) for i in range(8)))
            per = [sum(1 for g in got if g[0] == b) for b in range(4)]
            assert per == [2, 2, 2, 2] and all(g[1] == "least" for g in got), (per, [g[1] for g in got])
            results["least"] = per
            # 2. prefix affinity: a repeat with the same first 1,024 tokens + more goes back to that block
            base = list(range(5000, 5000 + 2048))
            b1, why1, *_ = await complete(s, url, base)
            fakes[b1].WAITING = 3  # its queue (within the affinity slack): least-loaded would pick elsewhere
            await asyncio.sleep(0.35)  # (the router's 0.1-s poller reads it)
            b2, why2, *_ = await complete(s, url, base[:1024] + [7] * 900)
            assert b2 == b1 and why2 == "prefix", (b1, b2, why2)
            results["prefix"] = (b1, why2)
            # text prompts too (cut points x chars-per-token)
            txt = "lorem ipsum dolor " * 400
            t1, _w, *_ = await complete(s, url, txt)
            t2, tw, *_ = await complete(s, url, txt[:4096] + " a different tail")
            assert t2 == t1 and tw == "prefix", (t1, t2, tw)
            fakes[b1].WAITING = 0
            await asyncio.sleep(0.35)
            # 2b. a SHORT shared prefix (a common 1,024-token system prompt) pins nothing: 8 concurrent -> 2 per block
            sysp = list(range(70000, 71024))
            await complete(s, url, sysp + [1] * 2000)
            got = await asyncio.gather(*(complete(s, url, sysp + [10 + i] * 2000, max_tokens=20) for i in range(8)))
            per = [sum(1 for g in got if g[0] == b) for b in range(4)]
            assert per == [2, 2, 2, 2] and all(g[1] == "least" for g in got), (per, [g[1] for g in got])
            # 2c. an affinity block far over the slack: the repeat goes elsewhere (counted as overflow)
            fakes[b1].WAITING = 20
            await asyncio.sleep(0.35)
            b3, why3, *_ = await complete(s, url, base)
            assert b3 != b1 and why3 == "least" and router.affinity_overflow >= 1, (b1, b3, why3)
            fakes[b1].WAITING = 0
            await asyncio.sleep(0.35)
            # 3. session affinity
            s1, _ = (await complete(s, url, [1] * 10, headers={"X-Session-Id": "u42"}))[:2]
            s2, sw = (await complete(s, url, [2] * 10, headers={"X-Session-Id": "u42"}))[:2]
            assert s1 == s2 and sw == "session", (s1, s2, sw)
            # 4. the long rule: blocks 0..2 each hold one decoding stream -> a new 40K prompt goes to block 3
            hold = [asyncio.create_task(complete(s, url, [9] * 10, max_tokens=300, headers={"X-Session-Id": f"h{b}"}))
                    for b in range(3)]  # fmt: skip
            await asyncio.sleep(0.3)
            homes = sorted(router.sessions[f"h{b}"] for b in range(3))
            free = ({0, 1, 2, 3} - set(homes))
            if len(free) == 1:
                lb, lw, *_ = await complete(s, url, list(range(40000)), max_tokens=2)
                assert lw == "long" and lb in free, (lb, lw, homes)
                results["long"] = (lb, lw)
            for t in hold:
                t.cancel()
            await asyncio.gather(*hold, return_exceptions=True)
            await asyncio.sleep(0.3)
            # 5. drain: block 1 drains while a stream is in flight; new requests avoid it; undrain restores it
            router.sessions["d"] = 1
            inflight = asyncio.create_task(complete(s, url, [3] * 10, max_tokens=40, headers={"X-Session-Id": "d"}))
            await asyncio.sleep(0.1)
            async with s.post(url + "/router/drain?block=1&wait_s=5") as r:
                dj = await r.json()
            assert dj["drained"] and dj["waited_s"] > 0.1, dj
            await inflight
            got = await asyncio.gather(*(complete(s, url, [2000 + i] * 50) for i in range(6)))
            assert all(g[0] != 1 for g in got), [g[0] for g in got]
            async with s.post(url + "/router/undrain?block=1") as r:
                assert (await r.json())["draining"] is False
            results["drain"] = dj
            # 6. a down block: health fails -> skipped
            fakes[2].healthy = False
            await asyncio.sleep(0.4)
            got = await asyncio.gather(*(complete(s, url, [3000 + i] * 50) for i in range(6)))
            assert all(g[0] != 2 for g in got) and not router.blocks[2].up
            fakes[2].healthy = True
            await asyncio.sleep(0.4)
            assert router.blocks[2].up
            # 7. a connect failure before the poll notices: retried elsewhere, the client sees 200
            await fakes[3].stop()
            router.blocks[3].up, router.blocks[3].fails = True, 0
            router.sessions["c"] = 3
            cb, cw, cst, *_ = await complete(s, url, [4] * 10, headers={"X-Session-Id": "c"})
            assert cst == 200 and cb != 3, (cb, cw, cst)
            results["retry"] = (cb, cw)
            # 8. streaming passthrough: the chunks arrive spread over the decode time, not in one burst
            b8, _w8, _st8, times, t0 = await complete(s, url, [6] * 10, max_tokens=30)
            assert len(times) >= 10 and times[-1] - times[0] > 0.15, (len(times), times[-1] - times[0])
            # 9. a client that goes away closes the upstream stream
            before = sum(f.cancelled for f in fakes[:3])
            async with s.post(url + "/v1/completions", json={"prompt": [8] * 10, "max_tokens": 500, "stream": True}) as r:
                n = 0
                async for _c in r.content.iter_any():
                    n += 1
                    if n >= 2:
                        break
            await asyncio.sleep(0.5)
            assert sum(f.cancelled for f in fakes[:3]) > before, "the upstream stream was not closed"
            # 10. /health and /metrics aggregate; /v1/models and /tokenize pass through
            async with s.get(url + "/health") as r:
                h = await r.json()
                assert r.status == 200 and len(h["blocks"]) == 4 and 3 not in h["up"], h["up"]
            async with s.get(url + "/metrics") as r:
                m = await r.text()
                assert 'dp4router_routed_total{block="0",reason="least"}' in m and "dp4router_block_queue_depth" in m
                assert 'dp4router_block_kv_usage{block="0"} 0.25' in m
            async with s.get(url + "/v1/models") as r:
                assert (await r.json())["data"][0]["id"] == "glm"
            async with s.post(url + "/tokenize", json={"prompt": "hi"}) as r:
                assert (await r.json())["tokens"] == [1, 2, 3]
            # the X-Request-Id passes through
            await complete(s, url, [11] * 10, headers={"X-Request-Id": "pdk-dp4-test"})
            # 11. v5 host-sampling policy (default reject): logprobs -> 400; a sampling request gets top_k = 32
            async with s.post(url + "/v1/completions", json={"model": "glm", "prompt": [5] * 10, "max_tokens": 2,
                                                             "logprobs": 5}) as r:  # fmt: skip
                j = await r.json()
                assert r.status == 400 and j["error"]["type"] == "router_host_sampling" and j["error"]["param"] == "logprobs", j
            async with s.post(url + "/v1/completions", json={"model": "glm", "prompt": [6] * 10, "max_tokens": 2,
                                                             "temperature": 0.7, "stream": True}) as r:  # fmt: skip
                assert r.status == 200 and r.headers.get("X-Router-Stripped") == "top_k=32", dict(r.headers)
                async for _c in r.content.iter_any():
                    pass
            assert any(b.get("top_k") == 32 and b.get("temperature") == 0.7 for f in fakes[:3] for b in getattr(f, "bodies", []))
            async with s.post(url + "/v1/completions", json={"model": "glm", "prompt": [7] * 10, "max_tokens": 2,
                                                             "temperature": 0, "stream": True}) as r:  # fmt: skip
                assert r.status == 200 and r.headers.get("X-Router-Stripped") is None
                async for _c in r.content.iter_any():
                    pass
            assert R.strip_host_sampling({"logprobs": 5, "min_p": 0.1, "x": 1}, ["logprobs", "min_p"]) == {"x": 1}
            async with s.post(url + "/v1/completions", json={"model": "glm", "prompt": [8] * 10, "max_tokens": 2,
                                                             "temperature": 0.8, "top_k": 50}) as r:  # fmt: skip
                assert r.status == 400 and (await r.json())["error"]["param"] == "top_k"
            assert R.host_sampling_params({"temperature": 0, "top_k": 50}, 32) == []  # greedy: top_k irrelevant
            assert R.host_sampling_params({"temperature": 0.7, "top_k": "x"}, 32) == ["top_k"]  # malformed: refused
            assert R.host_sampling_params({"temperature": 0.7, "top_k": -1}, 32) == ["top_k"]  # -1 = full vocab
            assert R.host_sampling_params({"top_k": 20, "min_p": "0", "min_tokens": None}, 32) == []
            assert R.clamp_top_k({"temperature": "0"}, 32) is None and R.clamp_top_k({}, 32) == {"top_k": 32}
            assert R.host_sampling_params({"logprobs": 0}, 32) == ["logprobs"]  # completions' 0 is still logprobs
            assert R.host_sampling_params({"logprobs": False, "top_logprobs": 0, "top_k": 8}, 32) == []  # chat: off
            assert R.host_sampling_params({"top_k": 8, "repetition_penalty": 1.0, "presence_penalty": 0}, 32) == []
            assert R.host_sampling_params({"top_k": 8, "repetition_penalty": 1.1, "frequency_penalty": 0.2}, 32) == [
                "frequency_penalty", "repetition_penalty"]  # fmt: skip
            async with s.get(url + "/metrics") as r:
                m = await r.text()
                assert 'dp4router_host_sampling_total{params="logprobs",action="reject"} 1' in m, m[-600:]
            assert any(rid == "pdk-dp4-test" for f in fakes[:3] for _n, rid in f.seen)
    finally:
        await runner.cleanup()
        for f in fakes[:3]:
            await f.stop()
    return results


async def scenario_v3():
    """the v3 long cap (N - 1 = 3 blocks with a > 131,072-token prefill at once; the 4th long waits in the router) and
    the long-free routing of short prompts"""
    fakes = [Fake(i) for i in range(4)]
    for f in fakes:
        f.PF_US = 6e-6  # 200,000 tokens -> 1.2 s of prefill (a > 1 MB body: the classify runs in a thread, v4)
        await f.start()
    runner, url, router = await start_router(fakes)
    peak = {"capped": 0}
    stop = asyncio.Event()

    async def sample():
        while not stop.is_set():
            peak["capped"] = max(peak["capped"], router.capped_inflight(), router.long_admitted)
            await asyncio.sleep(0.01)

    try:
        async with aiohttp.ClientSession() as s:
            smp = asyncio.create_task(sample())
            longs = [asyncio.create_task(complete(s, url, [50000 + i] * 200000, max_tokens=2)) for i in range(4)]
            for _ in range(100):  # three longs admitted + prefilling, the fourth queued in the router
                if router.long_queue == 1 and router.capped_inflight() == 3:
                    break
                await asyncio.sleep(0.02)
            assert router.long_queue == 1 and router.long_admitted == 3 and router.capped_inflight() == 3, (
                router.long_queue, router.long_admitted, router.capped_inflight())
            busy = {b.idx for b in router.blocks if b.capped_pf}
            sb, sw, *_ = await complete(s, url, [7] * 500, max_tokens=2)  # a short arrival: the long-free block
            assert sb not in busy and sw == "least", (sb, sw, busy)
            got = await asyncio.gather(*longs)
            stop.set()
            await smp
            assert all(g[2] == 200 for g in got) and peak["capped"] <= 3 and router.long_queued_total == 1, peak
            async with s.get(url + "/metrics") as r:
                m = await r.text()
            assert "dp4router_long_cap 3" in m and "dp4router_long_queued_total 1" in m and "dp4router_long_admitted 0" in m, m[-500:]
    finally:
        await runner.cleanup()
        for f in fakes:
            await f.stop()
    return {"peak_capped": peak["capped"], "short_block": sb, "busy": sorted(busy)}


async def scenario_v6():
    """v6 KV-headroom routing (program 2026-10-28T16:15:39Z / 16:15:53Z): block 0 holds a 1M (headroom 9,408 tokens)"""
    POOL = 1049536
    fakes = [Fake(i) for i in range(4)]
    for f in fakes:
        f.ADM = {"pool": POOL, "headroom": 1000000, "queued": 0}
        await f.start()
    fakes[0].ADM = {"pool": POOL, "headroom": 9408, "queued": 0}
    fakes[0].RUNNING = 1  # (v6d: it holds a 1M -- a block with nothing running reads as drained: its whole pool)
    for f in fakes[1:]:
        f.WAITING = 50  # least-loaded alone would pick block 0
    runner, url, router = await start_router(fakes)
    res = {}
    try:
        await asyncio.sleep(0.3)
        async with aiohttp.ClientSession() as s:
            # 1. a 32K arrival (reserve 32,832 > 9,408) never lands on block 0; a 2K (2,112) does
            b32 = [(await complete(s, url, [9] * 32768, max_tokens=2))[0] for _ in range(4)]
            assert 0 not in b32 and router.hr_steered >= 4, (b32, router.hr_steered)
            b2, why, *_ = await complete(s, url, [8] * 2048, max_tokens=2)
            assert b2 == 0 and why == "least", (b2, why)
            # 2. a short is checked against the RESIDENT headroom (plugin v3 lets it pass a queued deferred request)
            fakes[0].ADM = {"pool": POOL, "headroom": 9408, "queued": 200000}
            await asyncio.sleep(0.25)
            assert (await complete(s, url, [7] * 2048, max_tokens=2))[0] == 0
            # 3. no block fits a 131K: it waits IN THE ROUTER until one does -- v6d: while a request of the router's is
            # open (its end is a change the router sees; 3b: nothing of its open, no wait)
            for f in fakes:
                f.ADM = {"pool": POOL, "headroom": 9408, "queued": 0}
                f.RUNNING = 1  # (each holds a 1M)
            await asyncio.sleep(0.25)
            hold = asyncio.create_task(complete(s, url, [4] * 1024, max_tokens=300))  # ~3 s open (reserve 1,344)
            for _ in range(50):
                if any(b.inflight for b in router.blocks):
                    break
                await asyncio.sleep(0.02)
            t = asyncio.create_task(complete(s, url, [3] * 131072, max_tokens=2))
            for _ in range(50):
                if router.hr_queue == 1:
                    break
                await asyncio.sleep(0.02)
            await asyncio.sleep(0.3)
            assert router.hr_queue == 1 and not t.done(), router.hr_queue
            fakes[2].ADM = {"pool": POOL, "headroom": 1000000, "queued": 0}
            bl, why, st, *_ = await asyncio.wait_for(t, 10)
            assert bl == 2 and st == 200 and router.hr_waits == 1 and router.hr_fallbacks == 0, (bl, why, st)
            async with s.get(url + "/metrics") as r:
                m = await r.text()
            assert 'dp4router_block_engine_headroom_tokens{block="0"} 9408' in m and "dp4router_headroom_waits_total 1" in m, m[-900:]
            await hold
            # 3b. v6d: nothing of the router's open, no block fits: no wait (it would last --headroom-wait-max-s
            # if the gauges froze) -- the most headroom (a tie: load decides) takes it and the engine's gate judges
            fakes[2].ADM = {"pool": POOL, "headroom": 9408, "queued": 0}
            await asyncio.sleep(0.25)
            t0 = time.time()
            bl3b, *_ = await asyncio.wait_for(complete(s, url, [2] * 131072, max_tokens=2), 10)
            assert time.time() - t0 < 2.0 and router.hr_waits == 1 and router.hr_nothing_open >= 1, (
                time.time() - t0, router.hr_waits)
            res["blocks_32k"] = b32
            res["nothing_open_block"] = bl3b
    finally:
        await runner.cleanup()
    # 4. the ledger alone (blocks without the metric, --pool-tokens 40,000): an open 30,000-token request (reserve
    # 30,144) leaves 9,856 on its block; the next 30,000 goes elsewhere though least-loaded prefers that block
    for f in fakes:
        f.ADM, f.WAITING = None, 50
    fakes[1].WAITING = 0
    runner, url, router = await start_router(fakes, pool_tokens=40000)
    try:
        await asyncio.sleep(0.3)
        async with aiohttp.ClientSession() as s:
            first = asyncio.create_task(complete(s, url, [5] * 30000, max_tokens=100))  # ~1 s open on block 1
            for _ in range(50):
                if router.blocks[1].reserved:
                    break
                await asyncio.sleep(0.02)
            assert router.blocks[1].reserved == 30144, [b.reserved for b in router.blocks]
            second = (await complete(s, url, [6] * 30000, max_tokens=2))[0]
            assert second != 1, second
            assert (await first)[0] == 1 and router.blocks[1].reserved == 0
            res["ledger_second_block"] = second
    finally:
        await runner.cleanup()
        for f in fakes:
            await f.stop()
    return res


async def scenario_v6b():
    """v6b per-block max_model_len (program 2026-10-28T18:10:22Z): the MIXED FLEET -- blocks 0 / 1 / 3 at 1,048,576,
    block 2 a 131,072-cap spec block"""
    fakes = [Fake(i) for i in range(4)]
    for f in fakes:
        f.MML = 1048576
        await f.start()
    fakes[2].MML = 131072
    runner, url, router = await start_router(fakes)
    res = {}
    try:
        for _ in range(50):
            if all(b.mml for b in router.blocks):
                break
            await asyncio.sleep(0.05)
        assert [b.mml for b in router.blocks] == [1048576, 1048576, 131072, 1048576], [b.mml for b in router.blocks]
        async with aiohttp.ClientSession() as s:
            # 1. a 200,000-token prompt never lands on the spec block
            b200 = [(await complete(s, url, [9] * 200000, max_tokens=2))[0] for _ in range(4)]
            assert 2 not in b200, b200
            # 2. prompt + an explicit max_tokens over the cap: not the spec block (vLLM would refuse it there)
            b130 = [(await complete(s, url, [8] * 130000, max_tokens=2000))[0] for _ in range(3)]
            assert 2 not in b130, b130
            # 3. no max_tokens: the reserve cap counts (100,000 + 32,768 > 131,072)
            async def nomax(n):
                async with s.post(url + "/v1/completions", json={"model": "glm", "prompt": [7] * n, "stream": True}) as r:
                    await r.read()
                    return int(r.headers.get("X-Routed-Block", -1))
            fakes[2].MAXT = None
            assert all(b != 2 for b in [await nomax(100000) for _ in range(3)])
            # 4. a short PREFERS the spec block at equal load ...
            b2k, why, *_ = await complete(s, url, [6] * 2048, max_tokens=2)
            assert b2k == 2 and why == "least", (b2k, why)
            # 5. ... until it carries more than --spec-prefer-slack (4) extra requests
            fakes[2].WAITING = 10
            await asyncio.sleep(0.3)
            b2k2, *_ = await complete(s, url, [5] * 2048, max_tokens=2)
            assert b2k2 != 2, b2k2
            async with s.get(url + "/metrics") as r:
                m = await r.text()
            assert 'dp4router_block_max_model_len{block="2"} 131072' in m and "dp4router_ctx_steered_total" in m, m[-700:]
            res = {"200k": b200, "130k+2000": b130, "short": [b2k, b2k2]}
    finally:
        await runner.cleanup()
        for f in fakes:
            await f.stop()
    return res


def test_the_router_v6b():
    print(json.dumps(asyncio.run(scenario_v6b())))


def test_the_router_v6():
    print(json.dumps(asyncio.run(scenario_v6())))


def test_the_router_v3():
    print(json.dumps(asyncio.run(scenario_v3())))


async def start_router_mod(mod, fakes, **kw):
    a = mod.parse_args(["--port", str(free_port()), "--blocks", ",".join(f"http://127.0.0.1:{f.port}" for f in fakes),
                        "--poll-s", "0.1", "--down-after", "1", "--health-timeout-s", "0.5"])  # fmt: skip
    for k, v in kw.items():
        setattr(a, k, v)
    app = await mod.build_app(a)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", a.port).start()
    return runner, f"http://127.0.0.1:{a.port}", app["router"]


HUGE, MML = 40000, 70000  # (scaled: a "1M" prompt = 60,000 tokens, the blocks' cap 70,000, huge from 40,000)


async def eight_huge_at_c4(mod, mmls, prompt, max_tokens=2, **kw) -> dict:
    """00:11:38Z's pattern: 8 identical huge prompts, at most 4 in flight, through the router; the busy blocks sampled"""
    fakes = [Fake(i) for i in range(4)]
    for f, m in zip(fakes, mmls):
        f.MML, f.PF_US = m, 20e-6  # (Fake.PF_US is seconds per token: 60,000 x 20 us = a 1.2-s prefill)
        await f.start()
    runner, url, router = await start_router_mod(mod, fakes, **kw)
    seen = {"max_busy": 0, "max_per_block": 0, "blocks": set()}
    stop = asyncio.Event()

    async def sample():
        while not stop.is_set():
            busy = [f.idx for f in fakes if f.active > 0]
            seen["max_busy"] = max(seen["max_busy"], len(busy))
            seen["max_per_block"] = max([seen["max_per_block"]] + [f.active for f in fakes])
            seen["blocks"].update(busy)
            await asyncio.sleep(0.02)

    try:
        for _ in range(50):
            if all(b.mml for b in router.blocks):
                break
            await asyncio.sleep(0.05)
        sem = asyncio.Semaphore(4)
        async with aiohttp.ClientSession() as s:
            async def one():
                async with sem:
                    return await complete(s, url, prompt, max_tokens=max_tokens)
            smp = asyncio.create_task(sample())
            t0 = time.time()
            got = await asyncio.gather(*[one() for _ in range(8)])
            wall = time.time() - t0
            stop.set()
            await smp
        return {"routed": [g[0] for g in got], "status": sorted({g[2] for g in got}), "max_busy": seen["max_busy"],
                "max_per_block": seen["max_per_block"], "blocks": sorted(seen["blocks"]), "wall_s": round(wall, 2),
                "ctx_unfit": getattr(router, "ctx_unfit", None), "huge_waits": getattr(router, "huge_waits", None)}
    finally:
        await runner.cleanup()
        for f in fakes:
            await f.stop()


async def scenario_v6c(mod=None) -> dict:
    """v6c (program 2026-10-28T00:11:38Z / 00:12:19Z, operator priority): tonight 8 x 1M at c=4 ran serialised on
    blk0 while blk1 / blk3 sat idle. (a) token prompts, one shared prefix: affinity yields to admission and a huge
    prompt takes a block of its own -> 4 blocks busy on a 4 x 1M fleet, 3 on the mixed fleet (block 2 = the 131K spec
    block), never 2 huge on one block; (b) TEXT prompts the 4-chars-per-token estimate reads over every cap (a random-
    token 1M prompt reads ~1.35M): v6b kept one block for them, v6c keeps every block of the largest cap; (c) the
    spec-block preference is bounded by --spec-prefer-max"""
    mod = mod or R
    tok = [7] * 60000
    words = " ".join(f"w{i % 997:x}q" for i in range(60000))  # (~7 chars a word: the estimate reads ~108K tokens)
    kw = dict(pool_tokens=MML + 1000, huge_tokens=HUGE, max_model_len=MML)  # (the fleet: cap 1,048,576 < pool)
    res = {"a_4x1M": await eight_huge_at_c4(mod, [MML] * 4, tok, **kw),
           "a_mixed": await eight_huge_at_c4(mod, [MML, MML, 13107, MML], tok, **kw),
           "b_text_over_cap": await eight_huge_at_c4(mod, [MML] * 4, words, **kw)}
    # (c) bounded spec preference: the spec block takes the first --spec-prefer-max shorts, then load decides
    fakes = [Fake(i) for i in range(4)]
    for f in fakes:
        f.MML = MML
        f.DEC = 0.05
        await f.start()
    fakes[2].MML = 13107
    runner, url, router = await start_router_mod(mod, fakes, spec_prefer_max=2)
    try:
        for _ in range(50):
            if all(b.mml for b in router.blocks):
                break
            await asyncio.sleep(0.05)
        async with aiohttp.ClientSession() as s:
            got = await asyncio.gather(*[complete(s, url, [5 + i] * 512, max_tokens=20) for i in range(6)])
        res["c_spec_bounded"] = [g[0] for g in got]
    finally:
        await runner.cleanup()
        for f in fakes:
            await f.stop()
    return res


def test_the_router_v6c():
    res = asyncio.run(scenario_v6c())
    print(json.dumps(res))
    for k in ("a_4x1M", "b_text_over_cap"):
        assert res[k]["status"] == [200] and res[k]["max_per_block"] == 1 and res[k]["max_busy"] == 4, (k, res[k])
    m = res["a_mixed"]
    assert m["status"] == [200] and m["max_per_block"] == 1 and m["max_busy"] == 3 and 2 not in m["blocks"], m
    assert res["b_text_over_cap"]["ctx_unfit"] >= 8
    assert res["c_spec_bounded"].count(2) <= 2, res["c_spec_bounded"]


async def scenario_v6d(pin="on") -> dict:
    """v6d (program 2026-10-29T01:10:11Z (2)): a huge arrival whose stored context sits on a block that is busy
    waits for that block when the expected wait + the hit beats a cold prefill elsewhere; else it goes cold. P is stored
    on block B; Q (P's prefix, a different tail: no hit) then runs cold on B by affinity; P returns (a) late in Q's
    prefill -> waits for B, a hit; (b) right after Q starts -> cold elsewhere."""
    fakes = [Fake(i) for i in range(4)]
    for f in fakes:
        f.MML, f.PF_US, f.HIT_S = MML, 20e-6, 0.1  # (60,000 tokens cold = 1.2 s; a hit 0.1 s)
        await f.start()
    runner, url, router = await start_router_mod(R, fakes, pool_tokens=MML + 1000, huge_tokens=HUGE, max_model_len=MML,
                                                 prefill_s_per_mtok=20.0, decode_tok_s=100.0, hit_reload_s=0.1,
                                                 pin_wait=pin)  # fmt: skip
    P = [3] * 60000
    res = {}
    try:
        for _ in range(50):
            if all(b.mml for b in router.blocks):
                break
            await asyncio.sleep(0.05)
        async with aiohttp.ClientSession() as s:
            b_store, *_ = await complete(s, url, P, max_tokens=2)  # P stored on B
            for case, delay in (("late", 1.0), ("early", 0.05)):
                Q = P[:59000] + [case == "late" and 11 or 12] * 1000  # P's prefix: affinity -> B, no hit
                q = asyncio.create_task(complete(s, url, Q, max_tokens=2))
                await asyncio.sleep(delay)
                bp, why, st, times, t0 = await complete(s, url, P, max_tokens=2)
                bq, *_ = await q
                res[case] = {"q_block": bq, "p_block": bp, "p_reason": why, "p_ttft_s": round(times[0] - t0, 2)}
        res["store_block"] = b_store
        res["pin_total"] = {"wait": router.pin_waits, "cold": router.pin_cold, "timeout": router.pin_timeouts}
    finally:
        await runner.cleanup()
        for f in fakes:
            await f.stop()
    return res


def test_the_router_v6d():
    res = asyncio.run(scenario_v6d())
    print(json.dumps(res))
    B = res["store_block"]
    late, early = res["late"], res["early"]
    assert late["q_block"] == B and late["p_block"] == B and late["p_reason"] == "prefix-wait", res
    assert late["p_ttft_s"] < 1.0, res  # waited ~0.2 s for B + a 0.1-s hit, vs a 1.2-s cold prefill
    assert early["q_block"] == B and early["p_block"] != B, res  # B busy for ~1.2 s more: cold elsewhere wins
    assert res["pin_total"]["wait"] >= 1 and res["pin_total"]["cold"] >= 1, res
    off = asyncio.run(scenario_v6d(pin="off"))
    print(json.dumps(off))
    assert off["late"]["p_block"] != B and off["late"]["p_ttft_s"] > 1.0, off  # v6c: cold elsewhere, the full prefill


def load_mod(name: str):
    sp = importlib.util.spec_from_file_location(name, os.path.join(_HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod


async def fleet(mod, mmls, adms, **kw):
    """four Fake blocks (caps mmls, admit gauges adms) behind a router of module mod, polled once"""
    fakes = [Fake(i) for i in range(len(mmls))]
    for f, m, adm in zip(fakes, mmls, adms):
        f.MML, f.ADM = m, adm
        await f.start()
    runner, url, router = await start_router_mod(mod, fakes, **kw)
    for _ in range(50):
        if all(b.mml for b in router.blocks) and all(b.adm_pool is not None for b in router.blocks):
            break
        await asyncio.sleep(0.05)
    return fakes, runner, url, router


async def stop_fleet(fakes, runner):
    await runner.cleanup()
    for f in fakes:
        await f.stop()


async def scenario_v6d_drain(mod=None) -> dict:
    """v6d (program 2026-10-29T01:17:33Z): plugin v3 publishes glm53_admit_* from schedule() only, so a DRAINED
    engine's gauge keeps its last busy value. (a) TIE: blk0 / blk1 drained after 1M work (gauge frozen at 500 tokens of
    headroom), blk2 the 131K spec block (frozen too), blk3 fresh: 16 concurrent shorts -- v6c sends all 16 to blk3;
    v6d reads a drained block as its whole pool and spreads them. (b) RESERVE WAIT: every block drained, gauges frozen
    low, two 1M returns -- v6c waits --headroom-wait-max-s (6 s here) for headroom no frozen gauge ever shows; v6d
    routes at once. (b2) vLLM's running gauge frozen too (after an abort: the block is not provably drained): v6d waits
    at most for the router's own open request (nothing of its open: no wait), never --headroom-wait-max-s"""
    mod = mod or R
    res = {}
    frozen = {"pool": MML + 1000, "headroom": 500, "queued": 0}
    fakes, runner, url, router = await fleet(
        mod, [MML, MML, 13107, MML], [frozen, frozen, {"pool": 13000, "headroom": 300, "queued": 0},
                                      {"pool": MML + 1000, "headroom": MML + 1000, "queued": 0}])  # fmt: skip
    try:
        for f in fakes:
            f.RES, f.DEC = 600, 0.05
        async with aiohttp.ClientSession() as s:
            got = await asyncio.gather(*[complete(s, url, [9 + i] * 512, max_tokens=20) for i in range(16)])
        res["a_tie"] = [sum(1 for g in got if g[0] == b) for b in range(4)]
        res["a_status"] = sorted({g[2] for g in got})
    finally:
        await stop_fleet(fakes, runner)
    for case, running in (("b_reserve", None), ("b2_aborted", 1)):
        low = {"pool": MML + 1000, "headroom": 1000, "queued": 0}
        fakes, runner, url, router = await fleet(mod, [MML] * 4, [low] * 4, pool_tokens=MML + 1000, huge_tokens=HUGE,
                                                 max_model_len=MML, headroom_wait_max_s=6.0)  # fmt: skip
        try:
            for f in fakes:
                f.PF_US, f.RES, f.RUNNING = 20e-6, 61000, running
            async with aiohttp.ClientSession() as s:
                got = await asyncio.gather(*[complete(s, url, [4 + i] * 60000, max_tokens=2) for i in range(2)])
            res[case] = {"blocks": [g[0] for g in got], "status": sorted({g[2] for g in got}),
                         "ttft_s": [round(g[3][0] - g[4], 2) for g in got]}  # fmt: skip
        finally:
            await stop_fleet(fakes, runner)
    return res


def test_the_router_v6d_drained_blocks():
    res = asyncio.run(scenario_v6d_drain())
    print(json.dumps(res))
    tie = res["a_tie"]
    assert res["a_status"] == [200] and sum(tie) == 16, res
    assert min(tie[0], tie[1], tie[3]) >= 2 and tie[3] <= 6, res  # spread over the plain blocks (v6c: [0, 0, 0, 16])
    r = res["b_reserve"]
    assert r["status"] == [200] and max(r["ttft_s"]) < 3.0 and len(set(r["blocks"])) == 2, r
    r = res["b2_aborted"]
    assert r["status"] == [200] and min(r["ttft_s"]) < 2.0 and max(r["ttft_s"]) < 4.0, r
    old = asyncio.run(scenario_v6d_drain(load_mod("dp4router_v6c")))
    print(json.dumps(old))
    assert old["a_tie"][3] == 16, old  # the negative control: v6c reads the frozen gauges
    assert min(old["b_reserve"]["ttft_s"]) > 6.0, old


async def scenario_v6d_gone(mod=None) -> dict:
    """v6d: a client that leaves while its arrival waits in the router is not routed (aiohttp does not cancel the
    handler: v6c sent it on, a whole prefill for nobody); and the long-cap line is cancel-safe (v6c's tickets: a
    cancelled waiter's ticket was never served -- every later long arrival waited forever)"""
    mod = mod or R
    res = {}
    fakes, runner, url, router = await fleet(mod, [MML] * 2, [None] * 2, pool_tokens=MML + 1000, huge_tokens=HUGE,
                                             max_model_len=MML, headroom_wait_max_s=10.0)  # fmt: skip
    try:
        for f in fakes:
            f.PF_US = 20e-6
        async with aiohttp.ClientSession() as s:
            busy = [asyncio.create_task(complete(s, url, [6 + i] * 60000, max_tokens=2)) for i in range(2)]
            await asyncio.sleep(0.3)
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=0.4)) as s2:
                    await complete(s2, url, [8] * 60000, max_tokens=2)
                res["gone_status"] = "answered"
            except asyncio.TimeoutError:
                res["gone_status"] = "client timeout"
            await asyncio.gather(*busy)
            await asyncio.sleep(1.5)  # (v6c routes it once a block frees; it is cancelled after its first token)
        res["seen"] = sum(len(f.seen) for f in fakes)
        res["gone_before_route"] = getattr(router, "gone_before_route", None)
        # the long-cap line: A holds the one slot (2 blocks: cap N - 1), B and C wait, B is cancelled, A ends
        await router.long_admit()
        b = asyncio.create_task(router.long_admit())
        c = asyncio.create_task(router.long_admit())
        await asyncio.sleep(0.1)
        b.cancel()
        await asyncio.sleep(0.1)
        await router.long_release(counted=True)
        try:
            await asyncio.wait_for(c, timeout=1.0)
            res["line_after_cancel"] = "admitted"
        except asyncio.TimeoutError:
            res["line_after_cancel"] = "stuck"
    finally:
        await stop_fleet(fakes, runner)
    return res


def test_the_router_v6d_client_gone():
    res = asyncio.run(scenario_v6d_gone())
    print(json.dumps(res))
    assert res["gone_status"] == "client timeout" and res["seen"] == 2 and res["gone_before_route"] == 1, res
    assert res["line_after_cancel"] == "admitted", res
    old = asyncio.run(scenario_v6d_gone(load_mod("dp4router_v6c")))
    print(json.dumps(old))
    assert old["seen"] == 3 and old["line_after_cancel"] == "stuck", old  # the negative control


async def scenario_v6e(mod=None, **kw) -> dict:
    """v6e (program 2026-10-29T01:56:07Z P1, the interim): the 131K MTP block died when outputs past the 32,768
    reserve cap outgrew its pool (preemption -> a recompute prefill the generator refused). blk2 here: pool 13,000,
    the global reserve cap 2,000 (scaled). (a) 8 concurrent shorts with max_tokens 6,000 (full reserve 6,528): under
    the global rule blk2 takes ~5 (spec preference); with --block-reserve-cap -1,-1,0,-1 at most what its pool holds at
    FULL generation (1). (b) --block-max-total-tokens 0,0,4000,0: prompt + max_tokens over 4,000 (or no max_tokens)
    never lands on blk2; a 100-token answer still may."""
    mod = mod or R
    res = {}
    big = {"pool": MML + 1000, "headroom": MML + 1000, "queued": 0}
    fakes, runner, url, router = await fleet(mod, [MML, MML, 13107, MML],
                                             [big, big, {"pool": 13000, "headroom": 13000, "queued": 0}, big],
                                             admit_reserve_cap=2000, **kw)  # fmt: skip
    peak = {f.idx: 0 for f in fakes}
    stop = asyncio.Event()

    async def sample():
        while not stop.is_set():
            for f in fakes:
                peak[f.idx] = max(peak[f.idx], f.active)
            await asyncio.sleep(0.005)

    try:
        for f in fakes:
            f.RES, f.DEC, f.DECODE_MAX = 2560, 0.05, 20  # (the engine's reserve: capped like the plugin's; ~1 s each)
        smp = asyncio.create_task(sample())
        async with aiohttp.ClientSession() as s:
            got = await asyncio.gather(*[complete(s, url, [30 + i] * 512, max_tokens=6000) for i in range(8)])
            stop.set()
            await smp
            res["a_status"] = sorted({g[2] for g in got})
            res["a_peak"] = [peak[i] for i in range(4)]
            # (b) the total cap (when given): long answers / no max_tokens never on blk2, a short answer may
            if any(getattr(b, "total_cap", 0) for b in router.blocks):
                long_ = await asyncio.gather(*[complete(s, url, [60 + i] * 512, max_tokens=6000) for i in range(4)])
                async with s.post(url + "/v1/completions", json={"model": "glm", "prompt": [77] * 512,
                                                                 "stream": True}) as r:  # (no max_tokens)
                    await r.read()
                    nomax = int(r.headers.get("X-Routed-Block", -1))
                short = await complete(s, url, [90] * 512, max_tokens=100)
                res["b_long_blocks"] = [g[0] for g in long_]
                res["b_nomax_block"] = nomax
                res["b_short_block"] = short[0]
                res["b_steered"] = router.total_steered
    finally:
        await stop_fleet(fakes, runner)
    return res


def test_the_router_v6e():
    assert R.per_block_ints("0", 4, -1) == [0, 0, 0, 0] and R.per_block_ints("-1,-1,0", 4, -1) == [-1, -1, 0, -1]
    res = asyncio.run(scenario_v6e(block_reserve_cap="-1,-1,0,-1"))
    print(json.dumps(res))
    assert res["a_status"] == [200] and res["a_peak"][2] <= 1, res  # blk2 never past its pool at full generation
    old = asyncio.run(scenario_v6e(load_mod("dp4router_v6d")))
    print(json.dumps(old))
    assert old["a_peak"][2] >= 3, old  # the negative control: the 2,000-capped reserve oversubscribes blk2
    tot = asyncio.run(scenario_v6e(block_max_total_tokens="0,0,4000,0"))
    print(json.dumps(tot))
    assert 2 not in tot["b_long_blocks"] and tot["b_nomax_block"] != 2 and tot["b_steered"] >= 5, tot
    assert tot["b_short_block"] == 2, tot  # (an idle fleet: the spec preference still sends a short answer there)


async def scenario_v6f(mod=None, **kw) -> dict:
    """v6f (kit v3 trace check, 2026-10-29T02:26Z): agent traces tokenize at ~3 chars per token, the router estimates 4,
    so a prompt over the 131K block's cap went there and was refused (400) -- one refused root request aborts a whole
    AgentX warm-up. Blocks here refuse text at 3 chars/token; block 2 = the small cap (13,107). (a) 8 concurrent text
    prompts of 13,500 real tokens (40,500 chars: estimate 10,125, x1.25 = 12,656: block 2 still eligible) -> v6f retries
    a refusal on a larger cap, every request 200; (b) 14,667 real tokens (estimate 11,000, x1.25 = 13,750 > 13,107): steered off
    block 2 by the margin, no refusal."""
    mod = mod or R
    res = {}
    fakes, runner, url, router = await fleet(mod, [MML, MML, 13107, MML], [None] * 4, **kw)
    try:
        for f in fakes:
            f.CTX_CPT = 3.0
        text = lambda n_chars, salt: (f"z{salt}q " * (n_chars // len(f"z{salt}q ") + 1))[:n_chars]  # noqa: E731
        async with aiohttp.ClientSession() as s:
            got = await asyncio.gather(*[complete(s, url, text(40500, i), max_tokens=2) for i in range(8)])
            res["a_status"] = [g[2] for g in got]
            res["a_blocks"] = [g[0] for g in got]
            b = await complete(s, url, text(44000, 99), max_tokens=2)
            res["b_block"], res["b_status"] = b[0], b[2]
        res["refused_on_block2"] = getattr(fakes[2], "refused", 0)
        res["retried"] = getattr(router, "ctx_retried", None)
    finally:
        await stop_fleet(fakes, runner)
    return res


async def scenario_v6f_max_tokens(over="clamp") -> dict:
    """v6f (program 2026-10-29T02:32:09Z): --default-max-tokens fills max_tokens into a request that sends none (so
    its no-preempt reserve is prompt + the default, not the whole cap); --max-max-tokens bounds an explicit one (clamp,
    or 400 naming the limit); a prompt that nearly fills the largest cap gets no default"""
    res = {}
    fakes, runner, url, router = await fleet(R, [MML] * 4, [None] * 4, default_max_tokens=500, max_max_tokens=1000,
                                             max_tokens_over=over)  # fmt: skip
    try:
        for f in fakes:
            f.DECODE_MAX = 3

        async def post(prompt, **extra):
            async with aiohttp.ClientSession() as s:
                async with s.post(url + "/v1/completions", json=dict({"model": "glm", "prompt": prompt, "stream": True},
                                                                     **extra)) as r:  # fmt: skip
                    text = await r.text()
                    return r.status, r.headers.get("X-Router-Stripped"), text

        def max_tokens_seen(n):  # the max_tokens the block received with the n-token prompt
            for f in fakes:
                for (ln, _), b in zip(f.seen, getattr(f, "bodies", [])):
                    if ln == n:
                        return b["max_tokens"]
            return "none"

        st, hdr, _ = await post([5] * 100)
        res["default"] = (st, hdr, max_tokens_seen(100))
        st, hdr, text = await post([6] * 101, max_tokens=5000)
        res["over"] = (st, hdr, max_tokens_seen(101) if st == 200 else text[:160])
        st, hdr, _ = await post([7] * 102, max_tokens=800)
        res["within"] = (st, hdr, max_tokens_seen(102))
        st, hdr, _ = await post([8] * 56000)  # (x1.25 = 70,000 + 500 > the 70,000 cap: no default)
        res["near_cap"] = (st, hdr, max_tokens_seen(56000))
        res["counts"] = (router.mt_defaulted, router.mt_clamped, router.mt_rejected)
    finally:
        await stop_fleet(fakes, runner)
    return res


def test_the_router_v6f_max_tokens():
    res = asyncio.run(scenario_v6f_max_tokens())
    print(json.dumps(res))
    assert res["default"] == (200, "top_k=32,max_tokens=500(default)", 500), res  # (top_k: the router's device pin)
    assert res["over"] == (200, "top_k=32,max_tokens=1000(clamped)", 1000), res
    assert res["within"] == (200, "top_k=32", 800), res
    assert res["near_cap"] == (200, "top_k=32", None) and res["counts"] == (1, 1, 0), res
    rej = asyncio.run(scenario_v6f_max_tokens("reject"))
    print(json.dumps(rej))
    assert rej["over"][0] == 400 and "1000" in rej["over"][2] and rej["counts"] == (1, 0, 1), rej


def test_the_router_v6f():
    res = asyncio.run(scenario_v6f())
    print(json.dumps(res))
    assert res["a_status"] == [200] * 8 and 2 not in res["a_blocks"], res  # every refusal retried on a larger cap
    assert res["refused_on_block2"] >= 1 and res["retried"] == res["refused_on_block2"], res
    assert res["b_status"] == 200 and res["b_block"] != 2, res
    old = asyncio.run(scenario_v6f(load_mod("dp4router_v6e")))
    print(json.dumps(old))
    assert 400 in old["a_status"], old  # the negative control: v6e passes the block's refusal to the client


def test_the_router_v6b_piles_huge_prompts_on_one_block():
    """the negative control: the same patterns through v6b (dp4router_v6b.py) -- one block busy"""
    sp = importlib.util.spec_from_file_location("dp4router_v6b", os.path.join(_HERE, "dp4router_v6b.py"))
    old = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(old)

    async def run():
        kw = dict(pool_tokens=MML + 1000, max_model_len=MML)
        return {"a_4x1M": await eight_huge_at_c4(old, [MML] * 4, [7] * 60000, **kw),
                "b_text_over_cap": await eight_huge_at_c4(old, [MML] * 4, " ".join(f"w{i % 997:x}q" for i in range(60000)),
                                                          **kw)}  # fmt: skip
    res = asyncio.run(run())
    print(json.dumps(res))
    assert res["b_text_over_cap"]["max_busy"] == 1, res  # every over-the-estimate prompt on the first block


def test_the_router():
    res = asyncio.run(scenario())
    print(json.dumps(res, default=str))


if __name__ == "__main__":
    t0 = time.time()
    test_the_router()
    test_the_router_v3()
    test_the_router_v6()
    test_the_router_v6b()
    test_the_router_v6c()
    test_the_router_v6d()
    test_the_router_v6d_drained_blocks()
    test_the_router_v6d_client_gone()
    test_the_router_v6e()
    test_the_router_v6f()
    test_the_router_v6f_max_tokens()
    test_the_router_v6b_piles_huge_prompts_on_one_block()
    print(f"12 passed in {time.time() - t0:.1f} s")
