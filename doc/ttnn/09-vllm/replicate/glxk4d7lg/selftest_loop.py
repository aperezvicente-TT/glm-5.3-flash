"""The k4d7lg-stall bisect client (program 2026-10-27T23:16Z; 14b's plan doc/ttnn/14b-persistent-stack/hang_k4d7lg/
bisect_plan.md): the startup self-test's own request sequence LOOPED on one standing server, then request 36 alone.
- The sequence = the served plugin's selftest plan (vllm_tt_plugin/selftest/score.py of --plugin-src: 39 requests, tf_steps
  32, then the fork-TF prefills of the smoke rows that forked, 2 on host-8chip = 41), sent one at a time like the startup run
  (GLM53_SELFTEST_CONCURRENCY 1), greedy, logprobs 100, ignore_eos -- the body of runner.run_http. Index 36 = decode/len=192,
  the request that hung on host-8chip at 22:48:54Z (prefill [0, 192) on vslot 0, then the first decode).
- Leg 1: the sequence x --loops (10). Leg 2: request 36 alone x --lone (50).
- Stall detector: a request without a response in --timeout s (60) is a STALL: the row (loop, index, name) and
  <out>/bisect_stall.json are written, nothing more is sent and the server is LEFT UP for the halt-read (14b's steps 1-3;
  GLM53_ENGINE_STALL_S=0 keeps the engine alive). A non-200 response (an engine death) stops the client the same way.
- Loop 0 = the server's own startup self-test (in-process, before the socket binds): while waiting for /health the client
  watches the run dir's events.jsonl; after its 'startup' event, no new event for --startup-stall-s (60) with the socket
  still closed is a loop-0 STALL (the host-8chip hang happened there).
PREFILL mode (program 2026-10-28T03:20:19Z: the Galaxy proof arm for hang class A = a LOOPED 131K-PREFILL client on k4d6, each
loop a fresh 131K prompt, the stall detector as today, not the self-test): --prefill N replaces legs 1-2 with N requests of
--prefill-tokens (131,072) pseudo-random ids (a fresh seed per loop: no prefix / KDA hit), kv_transfer_params {"tt.cold": true}
(nothing stored: the share stays flat), max_tokens --prefill-max-tokens (4), one at a time; a request without a response in
--prefill-timeout s (180; a 131K cold prefill is ~28 s on host-8chip) is the STALL, handled as above.
Rows: <out>/selftest_loop_rows.jsonl (loop, index, name, kind, prompt_tokens, max_tokens, status, latency_s, the first 8
ids, ids_sha1, ids_eq_loop1); summary <out>/selftest_loop_summary.json. --out defaults to the run dir.
usage: selftest_loop.py --port <8100 + block> --run-glob '<work>/runs/resident_int27<tag>pdH_glxb<B>_*' [--out <dir>]
       [--loops 10] [--lone 50] [--timeout 60] [--startup-stall-s 60] [--wait-s 7200] [--plugin-src <src>] [--dry-run]
       [--prefill N [--prefill-tokens 131072] [--prefill-timeout 180] [--prefill-max-tokens 4] [--ladder 128,2048,32000]]
       [--trigger-legs 1,2,3,4 | h1,h2,h3,h4 [--trigger-timeout 180] [--decoder-tokens 2048] [--decoder-steps 20] [--bw-max-tokens 10] [--hs-max-tokens 2148]]  (k3srv)
exit: 0 no stall; 3 a stall (the server is left up); 4 an engine error or the server died; 2 could not run"""
import argparse
import glob
import hashlib
import importlib.util
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--run-glob", default="")
ap.add_argument("--out", default="")
ap.add_argument("--loops", type=int, default=10)
ap.add_argument("--lone", type=int, default=50)
ap.add_argument("--lone-name", default="decode/len=192")
ap.add_argument("--timeout", type=float, default=60.0)
ap.add_argument("--startup-stall-s", type=float, default=60.0)
ap.add_argument("--idle-stall-s", type=float, default=600.0, help="loop-0 stall without a 'startup' event: events.jsonl idle this long")
ap.add_argument("--heartbeat-s", type=float, default=120.0)
ap.add_argument("--wait-s", type=float, default=7200.0)
ap.add_argument("--logprobs", type=int, default=100)
ap.add_argument("--plugin-src", default=os.environ.get("GLM53_PLUGIN_SRC", "/opt/glm53/srv/s9k4d3/src"))
ap.add_argument("--dry-run", action="store_true", help="print the plan and exit (no HTTP)")
ap.add_argument("--prefill", type=int, default=0, help="the class-A proof: N fresh cold prompts instead of the self-test legs")
ap.add_argument("--prefill-tokens", type=int, default=131072)
ap.add_argument("--prefill-timeout", type=float, default=180.0)
ap.add_argument("--prefill-max-tokens", type=int, default=4)
ap.add_argument("--trigger-legs", default="", help="the k3srv trigger (program 2026-10-28T09:46:53Z + 09:49:31Z): legs, e.g. 1,2,3,4")
ap.add_argument("--trigger-timeout", type=float, default=180.0, help="a request without its response in this long = the STALL")
ap.add_argument("--decoder-tokens", type=int, default=2048)
ap.add_argument("--decoder-max-tokens", type=int, default=400)
ap.add_argument("--decoder-steps", type=int, default=20, help="leg 3: the decoder's streamed tokens before the bad_words request")
ap.add_argument("--bw-max-tokens", type=int, default=10)
ap.add_argument("--ladder", default="", help="kt5w (program 2026-10-28T09:51:19Z (3)): cold TTFT rungs before the prefill loop, e.g. 128,2048,32000")
ap.add_argument("--hs-max-tokens", type=int, default=2148, help="legs h1-h4: the readiness test's max_tokens (100 + the runner's reasoning budget 2048)")
a = ap.parse_args()

sp = os.path.join(a.plugin_src, "vllm_tt_plugin", "selftest", "score.py")
spec = importlib.util.spec_from_file_location("glm53_selftest_score", sp)
st = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = st  # dataclasses resolve the module by name
spec.loader.exec_module(st)
pack = st.load_pack(st.DEFAULT_PACK)
plan = st.build_plan(pack, tf_steps=int(os.environ.get("GLM53_SELFTEST_TF_STEPS", "32")))
names = [r.name for r in plan]
if a.lone_name not in names:
    sys.exit(f"{a.lone_name} is not in the plan ({sp})")
if a.dry_run:
    for i, r in enumerate(plan):
        print(i, r.name, r.kind, len(r.prompt_ids), r.max_tokens)
    print(f"plan {len(plan)} (+ the fork-TF prefills of forked smoke rows); lone = index {names.index(a.lone_name)} {a.lone_name}")
    sys.exit(0)

url = f"http://{a.host}:{a.port}"
t_start = time.time()
utc = lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def say(msg):
    print(f"{utc()} selftest_loop :{a.port}: {msg}", flush=True)


def run_dir():
    ds = [d for d in glob.glob(a.run_glob) if os.path.isdir(d) and os.path.getmtime(d) >= t_start - 300] if a.run_glob else []
    return max(ds, key=os.path.getmtime) if ds else None


def health():
    try:
        with urllib.request.urlopen(url + "/health", timeout=5) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001 -- not up yet
        return False


def server_dead(rd):
    try:
        pid = int(open(os.path.join(rd, "server.pid")).read().split()[0])
    except (OSError, ValueError, IndexError):
        return False
    return not os.path.exists(f"/proc/{pid}")


# wait for the socket; loop 0 = the startup self-test (events.jsonl stagnation after the 'startup' event)
# 2026-10-28T03:38:20Z (arm d: its server stalled at 03:01:02Z and this client never declared it, log empty): the run dir is
# RE-PICKED every poll (the newest match; a refused / earlier launch's dir no longer latches the watch), a stall no longer
# needs the 'startup' event (events.jsonl idle > --startup-stall-s after the startup event, or idle > --idle-stall-s (600)
# without one, socket closed, server pid alive), and a heartbeat line every --heartbeat-s (120) names the watched dir / state
rd, seen_startup, last_sig, last_change, last_hb = None, False, None, time.time(), 0.0
while not health():
    if time.time() - t_start > a.wait_s:
        say(f"no /health in {a.wait_s:.0f} s; giving up")
        sys.exit(2)
    nd = run_dir()
    if nd and nd != rd:
        if rd:
            say(f"watching the newer run dir {nd} (was {rd})")
        rd, seen_startup, last_sig, last_change = nd, False, None, time.time()
    if time.time() - last_hb > a.heartbeat_s:
        last_hb = time.time()
        ev_ = os.path.join(rd, "events.jsonl") if rd else None
        say(f"loop 0 waiting: run dir {rd or 'none yet (glob ' + a.run_glob + ')'}; events.jsonl "
            f"{os.path.getsize(ev_) if ev_ and os.path.exists(ev_) else 'absent'} B, idle {time.time() - last_change:.0f} s, "
            f"startup event {'seen' if seen_startup else 'not seen'}")
    if rd:
        ev = os.path.join(rd, "events.jsonl")
        if os.path.exists(ev):
            st_ = os.stat(ev)
            sig = (st_.st_size, st_.st_mtime)
            if sig != last_sig:
                last_sig, last_change = sig, time.time()
                if not seen_startup:
                    with open(ev) as f:
                        seen_startup = any('"event": "startup"' in ln or '"event":"startup"' in ln for ln in f)
            else:
                idle = time.time() - last_change
                if (seen_startup and idle > a.startup_stall_s) or idle > a.idle_stall_s:
                    with open(ev) as f:
                        lines = f.readlines()
                    tail = lines[-1][:600] if lines else ""
                    out = a.out or rd
                    os.makedirs(out, exist_ok=True)
                    stall = {"utc": utc(), "loop": 0, "index": None, "name": "the startup self-test (in-process)",
                             "events_idle_s": round(idle, 1), "startup_event_seen": seen_startup, "last_event": tail, "run": rd}
                    json.dump(stall, open(os.path.join(out, "bisect_stall.json"), "w"), indent=1)
                    say(f"STALL in loop 0 (the startup self-test): events.jsonl idle {stall['events_idle_s']} s, socket closed "
                        f"(startup event {'seen' if seen_startup else 'not seen'}); the server is left up for the halt-read ({rd})")
                    sys.exit(3)
        if server_dead(rd):
            say(f"the server died before binding ({rd})")
            sys.exit(4)
    time.sleep(5)
rd = rd or run_dir()
out = a.out or rd or "."
os.makedirs(out, exist_ok=True)
with urllib.request.urlopen(url + "/v1/models", timeout=30) as r:
    model = json.loads(r.read())["data"][0]["id"]
say(f"server up ({rd}); model {model}; " + (f"TRIGGER mode: legs {a.trigger_legs}" if a.trigger_legs else (f"PREFILL mode: {a.prefill} fresh cold prompts of {a.prefill_tokens} tokens" if a.prefill
    else f"plan {len(plan)} requests; {a.loops} loops, then {a.lone_name} x {a.lone}")))
rows_f = open(os.path.join(out, "selftest_loop_rows.jsonl"), "a")
ref_ids, lat = {}, {}


def send(loop, index, req, cold=False, timeout=None):
    body = {"model": model, "prompt": list(req.prompt_ids), "max_tokens": req.max_tokens, "temperature": 0.0,
            "logprobs": a.logprobs, "return_tokens_as_token_ids": True, "return_token_ids": True, "ignore_eos": True}
    if cold:
        body["kv_transfer_params"] = {"tt.cold": True}
    row = {"utc": utc(), "loop": loop, "index": index, "name": req.name, "kind": req.kind,
           "prompt_tokens": len(req.prompt_ids), "max_tokens": req.max_tokens}
    t0 = time.perf_counter()
    obs = None
    try:
        q = urllib.request.Request(url + "/v1/completions", data=json.dumps(body).encode(),
                                   headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(q, timeout=timeout or a.timeout) as r:
            payload = json.loads(r.read().decode())
        obs = st.Observation(token_ids=[], top=[])
        ch = payload["choices"][0]
        lp = ch.get("logprobs") or {}
        obs.top = [[(int(str(k).split(":")[-1]), float(v)) for k, v in (d or {}).items()] for d in (lp.get("top_logprobs") or [])]
        obs.token_ids = [int(x) for x in (ch.get("token_ids") or [str(t).split(":")[-1] for t in (lp.get("tokens") or [])])]
        ids = obs.token_ids
        key = (req.name, index)
        ref_ids.setdefault(key, ids)
        row.update(status="ok", ids=ids[:8], ids_sha1=hashlib.sha1(json.dumps(ids).encode()).hexdigest()[:12],
                   ids_eq_loop1=ids == ref_ids[key])
    except urllib.error.HTTPError as e:
        row.update(status="error", error=f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}")
    except Exception as e:  # noqa: BLE001 -- a timeout is the stall signal
        s = f"{type(e).__name__}: {e}"
        row.update(status="timeout" if ("timed out" in s or isinstance(e, TimeoutError)) else "error", error=s[:300])
    row["latency_s"] = round(time.perf_counter() - t0, 3)
    rows_f.write(json.dumps(row) + "\n")
    rows_f.flush()
    if row["status"] == "ok":
        lat.setdefault(req.kind, []).append(row["latency_s"])
    return row, obs


def stop(row, code):
    stall = {**row, "run": rd, "port": a.port}
    json.dump(stall, open(os.path.join(out, "bisect_stall.json"), "w"), indent=1)
    what = "STALL" if row["status"] == "timeout" else "ENGINE ERROR"
    say(f"{what} at loop {row['loop']} index {row['index']} ({row['name']}): {row.get('error', '')[:160]}; nothing more is "
        f"sent; the server is left up for the halt-read ({out}/bisect_stall.json)")
    summary(what.lower().replace(" ", "_"), row)
    sys.exit(code)


def summary(status, row=None):
    pct = lambda xs, p: sorted(xs)[min(len(xs) - 1, int(p * len(xs)))] if xs else None
    s = {"utc": utc(), "status": status, "run": rd, "port": a.port, "loops": a.loops, "lone": a.lone, "prefill": a.prefill,
         "prefill_tokens": a.prefill_tokens if a.prefill else None,
         "first_bad": row, "requests_ok": sum(len(v) for v in lat.values()),
         "latency_s": {k: {"n": len(v), "p50": pct(v, 0.5), "p99": pct(v, 0.99), "max": max(v)} for k, v in lat.items()}}
    json.dump(s, open(os.path.join(out, "selftest_loop_summary.json"), "w"), indent=1)


if a.trigger_legs:
    # the k3srv TRIGGER (program [session 10] 2026-10-28T09:46:53Z ORDER + 09:49:31Z ADDENDUM, 14b 09:48:39Z): the k4d8k3 decode stall
    # (host-8chip 08:38:47Z) = the TRANSITION step: a host-sampled request (bad_words, the readiness suite's test_host_only_params::
    # test_bad_words) joining while an ASYNC device-sampled decoder is in flight (a D2H read of io.sampled + a whole-buffer H2D write
    # + the w8 replay + the D2H logits gather, no sync between). Legs against the one standing server, LMCache store OFF ({"tt.cold":
    # true}) in 1-3: (1) the bad_words request ALONE; (2) two device-sampled requests together (the 2-slot w8 first replay, async);
    # (3) one 2K device-sampled decoder in flight for >= --decoder-steps streamed tokens (w1), THEN the bad_words request = the
    # transition; (4) = (3) with the decoder STORING (no tt.cold; the bad_words request goes in right after the decoder's first
    # token, while its KV store is in flight). Each leg prints 'K3SRV LEG<n> CLEAN | STALL | ERROR'; a STALL stops sending and
    # leaves the server up for 14b's v3b read (bisect_stall.json, exit 3). Expected: 1 and 2 clean, 3 stalls.
    import threading
    BW = ["hello", "Hello", "hi", "Hi", "hey", "Hey"]  # test_bad_words' list (plugin tests/tt/test_host_only_params.py)

    def _post(path, body, timeout, stream=False, on_token=None):
        q = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(q, timeout=timeout) as r:
            if not stream:
                return json.loads(r.read().decode())
            n = 0
            for raw in r:
                ln = raw.decode(errors="replace").strip()
                if not ln.startswith("data: ") or ln == "data: [DONE]":
                    continue
                ch = (json.loads(ln[6:]).get("choices") or [{}])[0]
                if ch.get("text") or ch.get("token_ids"):
                    n += 1
                    if on_token:
                        on_token(n)
            return {"streamed_tokens": n}

    def _call(leg, name, path, body, timeout, res, stream=False, on_token=None):
        row = {"utc": utc(), "loop": f"leg{leg}", "index": name, "name": name, "kind": "trigger", "path": path,
               "cold": bool(body.get("kv_transfer_params")), "max_tokens": body.get("max_tokens")}
        t0 = time.perf_counter()
        try:
            out_ = _post(path, body, timeout, stream, on_token)
            row.update(status="ok", detail=(out_.get("streamed_tokens") if stream else
                       ((out_.get("choices") or [{}])[0].get("finish_reason"))))
        except urllib.error.HTTPError as e:
            row.update(status="error", error=f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}")
        except Exception as e:  # noqa: BLE001 -- a timeout is the stall signal
            s_ = f"{type(e).__name__}: {e}"
            row.update(status="timeout" if ("timed out" in s_ or isinstance(e, TimeoutError)) else "error", error=s_[:300])
        row["latency_s"] = round(time.perf_counter() - t0, 3)
        rows_f.write(json.dumps(row) + "\n")
        rows_f.flush()
        if row["status"] == "ok":
            lat.setdefault(f"leg{leg}", []).append(row["latency_s"])
        res[name] = row
        return row

    def _bw(seed, cold=True):
        b = {"model": model, "messages": [{"role": "user", "content": "Say hello to me"}], "max_tokens": a.bw_max_tokens,
             "bad_words": BW, "temperature": 1.0, "seed": seed}
        if cold:
            b["kv_transfer_params"] = {"tt.cold": True}
        return b

    def _dev(k, n_tok, max_tokens, cold=True, stream=False):
        rnd = random.Random(7_000_003 * k + n_tok)
        b = {"model": model, "prompt": [rnd.randrange(1000, 100000) for _ in range(n_tok)], "max_tokens": max_tokens,
             "temperature": 1.0, "ignore_eos": True, "stream": stream}
        if cold:
            b["kv_transfer_params"] = {"tt.cold": True}
        return b

    def _verdict(leg, res):
        bad = [r for r in res.values() if r["status"] != "ok"]
        if not bad:
            say(f"K3SRV LEG{leg} CLEAN: " + "; ".join(f"{r['name']} {r['latency_s']} s ({r.get('detail')})" for r in res.values()))
            return
        r = next((x for x in bad if x["status"] == "timeout"), bad[0])
        say(f"K3SRV LEG{leg} {'STALL' if r['status'] == 'timeout' else 'ERROR'}: " + "; ".join(f"{x['name']} {x['status']} {x.get('error', '')[:120]}" for x in bad))
        stop(r, 3 if r["status"] == "timeout" else 4)

    def _transition(leg, cold_decoder, steps):
        res, hit = {}, threading.Event()
        dec = threading.Thread(target=_call, args=(leg, "decoder", "/v1/completions",
                               _dev(leg, a.decoder_tokens, a.decoder_max_tokens, cold=cold_decoder, stream=True), a.trigger_timeout, res, True,
                               lambda n: hit.set() if n >= steps else None), daemon=True)
        dec.start()
        if not hit.wait(a.trigger_timeout):
            dec.join(5)
            res.setdefault("decoder", {"utc": utc(), "loop": f"leg{leg}", "index": "decoder", "name": "decoder", "kind": "trigger",
                                       "status": "timeout", "error": f"fewer than {steps} streamed tokens in {a.trigger_timeout} s", "latency_s": a.trigger_timeout})
            _verdict(leg, res)
        say(f"leg {leg}: the decoder streamed >= {steps} token(s){'' if cold_decoder else ' (storing)'}; sending the bad_words request")
        _call(leg, "bad_words", "/v1/chat/completions", _bw(leg), a.trigger_timeout, res)
        dec.join(a.trigger_timeout)
        if dec.is_alive():
            res["decoder"] = {"utc": utc(), "loop": f"leg{leg}", "index": "decoder", "name": "decoder", "kind": "trigger",
                              "status": "timeout", "error": f"the decoder did not finish within {a.trigger_timeout} s of the transition", "latency_s": a.trigger_timeout}
        _verdict(leg, res)

    def _host_batch(leg, n, cold):
        # 14b 09:53:19Z (from the k4d8k3 server.log): the stall = an ALL-HOST-SAMPLED 2-live w8 step reached by STAGGERED admission
        # (test_bad_words' 5 chat requests sent together; GLM53_PREFILL_ONE_REQUEST=1 admits them one at a time: A decodes at w1, B
        # joins), more prefills queued. n concurrent bad_words chat requests, seeds 0..n-1, max_tokens --hs-max-tokens (2148 = the
        # test's 100 + the runner's reasoning budget 2048); LMCache store as the test (on) unless cold.
        res = {}
        def body(i):
            b_ = {"model": model, "messages": [{"role": "user", "content": "Say hello to me"}], "max_tokens": a.hs_max_tokens,
                  "bad_words": BW, "temperature": 1.0, "seed": i}
            if cold:
                b_["kv_transfer_params"] = {"tt.cold": True}
            return b_
        th = [threading.Thread(target=_call, args=(leg, f"bw{i}", "/v1/chat/completions", body(i), a.trigger_timeout, res), daemon=True) for i in range(n)]
        for t in th:
            t.start()
        for t in th:
            t.join(a.trigger_timeout + 5)
        for i in range(n):
            res.setdefault(f"bw{i}", {"utc": utc(), "loop": f"leg{leg}", "index": f"bw{i}", "name": f"bw{i}", "kind": "trigger",
                                      "status": "timeout", "error": "no response", "latency_s": a.trigger_timeout})
        _verdict(leg, res)

    for leg in [x.strip() for x in a.trigger_legs.split(",") if x.strip()]:
        say(f"K3SRV leg {leg} starts")
        if leg in ("h1", "h2", "h3", "h4"):  # 14b's revised legs (09:53:19Z): h1 one alone, h2 two together, h3 the test as run (5), h4 = h3 store off
            _host_batch(leg, {"h1": 1, "h2": 2, "h3": 5, "h4": 5}[leg], cold=leg == "h4")
            continue
        leg = int(leg)
        if leg == 1:
            res = {}
            _call(1, "bad_words", "/v1/chat/completions", _bw(1), a.trigger_timeout, res)
            _verdict(1, res)
        elif leg == 2:
            res = {}
            th = [threading.Thread(target=_call, args=(2, f"dev{i}", "/v1/completions", _dev(20 + i, 128, 64), a.trigger_timeout, res), daemon=True) for i in (0, 1)]
            for t in th:
                t.start()
            for t in th:
                t.join(a.trigger_timeout + 5)
            for i in (0, 1):
                res.setdefault(f"dev{i}", {"utc": utc(), "loop": "leg2", "index": f"dev{i}", "name": f"dev{i}", "kind": "trigger",
                                           "status": "timeout", "error": "no response", "latency_s": a.trigger_timeout})
            _verdict(2, res)
        elif leg == 3:
            _transition(3, True, a.decoder_steps)
        elif leg == 4:
            _transition(4, False, 1)
        else:
            sys.exit(f"unknown trigger leg {leg}")
    summary("no_stall")
    say(f"K3SRV NO STALL: legs {a.trigger_legs} all clean ({out}/selftest_loop_summary.json)")
    sys.exit(0)
if a.prefill and a.ladder:
    # the ladder cell (program [session 10] 2026-10-28T09:51:19Z (3), kt5w): per rung one COLD streamed request ({"tt.cold":
    # true}: nothing stored, no hit) of n pseudo-random ids (a fresh seed per rung), 64 greedy tokens, ignore_eos; TTFT = the first
    # streamed token, decode tok/s = (tokens - 1) / (last - first). A rung without its response in --prefill-timeout s is the STALL.
    for n in [int(x) for x in a.ladder.split(",") if x.strip()]:
        rnd = random.Random(5_000_011 * n + 17)
        body = {"model": model, "prompt": [rnd.randrange(1000, 100000) for _ in range(n)], "max_tokens": 64, "temperature": 0.0,
                "ignore_eos": True, "stream": True, "kv_transfer_params": {"tt.cold": True}}
        row = {"utc": utc(), "loop": "ladder", "index": n, "name": f"ladder/{n}", "kind": "ladder", "prompt_tokens": n, "max_tokens": 64}
        t0 = time.perf_counter(); stamps = []
        try:
            q = urllib.request.Request(url + "/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(q, timeout=a.prefill_timeout) as r:
                for raw in r:
                    ln = raw.decode(errors="replace").strip()
                    if ln.startswith("data: ") and ln != "data: [DONE]":
                        ch = (json.loads(ln[6:]).get("choices") or [{}])[0]
                        if ch.get("text") or ch.get("token_ids"):
                            stamps.append(time.perf_counter())
            row.update(status="ok" if stamps else "error", ttft_s=round(stamps[0] - t0, 3) if stamps else None, tokens=len(stamps),
                       decode_tok_s=round((len(stamps) - 1) / (stamps[-1] - stamps[0]), 2) if len(stamps) > 1 and stamps[-1] > stamps[0] else None)
        except urllib.error.HTTPError as e:
            row.update(status="error", error=f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}")
        except Exception as e:  # noqa: BLE001 -- a timeout is the stall signal
            s_ = f"{type(e).__name__}: {e}"
            row.update(status="timeout" if ("timed out" in s_ or isinstance(e, TimeoutError)) else "error", error=s_[:300])
        row["latency_s"] = round(time.perf_counter() - t0, 3)
        rows_f.write(json.dumps(row) + "\n")
        rows_f.flush()
        if row["status"] != "ok":
            stop(row, 3 if row["status"] == "timeout" else 4)
        say(f"LADDER {n}: cold TTFT {row['ttft_s']} s, decode {row['decode_tok_s']} tok/s ({row['tokens']} tokens)")
if a.prefill:
    class _Req:  # a fresh cold prompt per loop (the selftest Request's fields send() reads)
        def __init__(self, k):
            rnd = random.Random(1_000_003 * k + a.prefill_tokens)
            self.prompt_ids = [rnd.randrange(1000, 100000) for _ in range(a.prefill_tokens)]
            self.max_tokens, self.name, self.kind = a.prefill_max_tokens, f"prefill/{a.prefill_tokens}", "prefill"
    for k in range(1, a.prefill + 1):
        row, _ = send(k, 0, _Req(k), cold=True, timeout=a.prefill_timeout)
        if row["status"] != "ok":
            stop(row, 3 if row["status"] == "timeout" else 4)
        say(f"prefill {k}/{a.prefill}: {a.prefill_tokens} tokens in {row['latency_s']} s")
    summary("no_stall")
    say(f"NO STALL: {a.prefill} fresh cold prompts of {a.prefill_tokens} tokens ({out}/selftest_loop_summary.json)")
    sys.exit(0)
for loop in range(1, a.loops + 1):
    t0, obs_loop = time.time(), {}
    for i, req in enumerate(plan):
        row, obs = send(loop, i, req)
        if row["status"] != "ok":
            stop(row, 3 if row["status"] == "timeout" else 4)
        obs_loop[req.name] = obs
    for j, req in enumerate(st.fork_tf_requests(pack, plan, obs_loop)):
        row, _ = send(loop, len(plan) + 1 + j, req)
        if row["status"] != "ok":
            stop(row, 3 if row["status"] == "timeout" else 4)
    say(f"loop {loop}/{a.loops} done in {time.time() - t0:.1f} s")
li = names.index(a.lone_name)
for n in range(1, a.lone + 1):
    row, _ = send(f"lone{n}", li, plan[li])
    if row["status"] != "ok":
        stop(row, 3 if row["status"] == "timeout" else 4)
say(f"lone leg done: {a.lone_name} x {a.lone}")
summary("no_stall")
say(f"NO STALL: {a.loops} loops + {a.lone} lone ({out}/selftest_loop_summary.json)")
