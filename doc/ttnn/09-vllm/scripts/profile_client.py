"""The profile mock's requests (program 2026-10-23T19:17:55Z: py-spy as the parent of the cf3g image mock, a 131,072
cold request + 64 decode steps at 1M max_model_len; the first-step cost of a 128-token request too; 19:20:12Z: a 4-user x
131,072 concurrent case). The --lengths requests one at a time, then each --concurrent group "NxL" at once; tt.cold,
streamed; one JSON row per request with the client TTFT, the decode span and the wall stamps the profile's samples are
cut by. Host only (the mock server).
A group spec ending in "s" ("4x131072s", 19:24:42Z: the LMCache store path) sends its requests without tt.cold (stored).
--hit N (the cf3g verdict mocks' hit request, 19:04:11Z): the same stored N-token prompt twice, one after the other; the
second row's cached_tokens is LMCache's hit (0 on a cache whose checkpoints cannot resume: kv_checkpoints_resumable).
--mixed L (cf3h, program 2026-10-23T22:43:55Z: one user prefills L tokens while one decodes): a cold 128-token decoder
streams; at its 8th token a cold L-token request starts; when that request ends the decoder disconnects (an abort). The
"mixed/summary" row: the decoder's tokens inside the long request's prefill window [t0, t_first], its inter-token gaps
before that window (the decode-only step) and overlapping it (p50 / max). With mixed steps the overlapping gaps stay
~one step; without them the decoder waits out prefill steps.
usage: profile_client.py --url URL [--lengths 128,2048,131072,128] [--concurrent 4x131072,4x131072s] [--hit 2560]
       [--mixed 32768]"""

import argparse
import json
import random
import threading
import time

import requests

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True)
ap.add_argument("--lengths", default="128,2048,131072,128")
ap.add_argument("--max-tokens", type=int, default=64)
ap.add_argument("--concurrent", default="")
ap.add_argument("--hit", type=int, default=0)
ap.add_argument("--mixed", type=int, default=0)
# --prefix L,S (cf3h-e, the provenance restore): store an L-token prompt, then its S-token prefix twice (the same seed =
# the same prefix). The first S request matches LMCache past its KDA hit (the provenance case: it replaces the span);
# its repeat resumes at the last grid boundary before S (hit depth)
ap.add_argument("--prefix", default="")
a = ap.parse_args()


def one(n, seed, tag, cold=True, max_tokens=None, on_token=None, stop=None):
    rnd = random.Random(seed)
    ids = [rnd.randrange(1000, 100000) for _ in range(n)]
    body = {"model": "zai-org/GLM-5.3-Flash", "prompt": ids, "max_tokens": max_tokens or a.max_tokens, "temperature": 0,
            "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}  # fmt: skip
    if cold:
        body["kv_transfer_params"] = {"tt.cold": True}
    t0, stamps, err, usage = time.time(), [], None, None
    try:
        with requests.post(f"{a.url}/v1/completions", json=body, stream=True, timeout=3600) as r:
            if r.status_code != 200:
                err = f"http {r.status_code}: {r.text[:200]}"
            for line in r.iter_lines():
                if line.startswith(b"data: ") and line != b"data: [DONE]":
                    ev = json.loads(line[6:])
                    usage = ev.get("usage") or usage
                    if (ev.get("choices") or [{}])[0].get("text") is not None:
                        stamps.append(time.time())
                        if on_token is not None:
                            on_token(len(stamps))
                if stop is not None and stop.is_set():
                    break  # closing the stream aborts the request
    except requests.RequestException as e:
        err = f"{type(e).__name__}: {e}"[:300]
    row = {"case": tag, "prompt_tokens": n, "t0": round(t0, 3), "t_first": round(stamps[0], 3) if stamps else None,
                      "t_end": round(stamps[-1], 3) if stamps else None, "chunks": len(stamps),
                      "ttft_s": round(stamps[0] - t0, 3) if stamps else None,
                      "decode_ms_per_chunk": round(1e3 * (stamps[-1] - stamps[0]) / (len(stamps) - 1), 2) if len(stamps) > 1 else None,
                      "completion_tokens": (usage or {}).get("completion_tokens"),
                      "cached_tokens": ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens"),
                      "error": err}  # fmt: skip
    print(json.dumps(row), flush=True)
    return dict(row, stamps=stamps)


for i, n in enumerate(int(v) for v in a.lengths.split(",") if v):
    one(n, 9_000_000 + i, "solo")
for g, spec in enumerate(v for v in a.concurrent.split(",") if v):
    stored = spec.lower().endswith("s")
    nu, n = (int(x) for x in spec.lower().rstrip("s").split("x"))
    th = [threading.Thread(target=one, args=(n, 9_100_000 + 1000 * g + u, f"{spec}/{u}", not stored)) for u in range(nu)]
    for t in th:
        t.start()
    for t in th:
        t.join()
if a.hit:
    for turn in ("hit/store", "hit/repeat"):
        one(a.hit, 9_200_000, turn, cold=False)
if a.prefix:
    big, small = (int(v) for v in a.prefix.split(","))
    one(big, 9_400_000, f"prefix/store{big}", cold=False, max_tokens=4)
    for turn in ("first", "repeat"):
        one(small, 9_400_000, f"prefix/{small}-{turn}", cold=False, max_tokens=4)
if a.mixed:
    started, stop, dec = threading.Event(), threading.Event(), {}
    td = threading.Thread(target=lambda: dec.update(one(128, 9_300_000, "mixed/decoder", True, max_tokens=65536,
                                                        on_token=lambda k: k >= 8 and started.set(), stop=stop)))
    td.start()
    started.wait(1800)
    lr = one(a.mixed, 9_300_001, f"mixed/prefill{a.mixed}", True, max_tokens=4)
    stop.set()
    td.join()
    s, t0, t1 = dec.get("stamps") or [], lr["t0"], (lr["stamps"] or [None])[0]
    gaps = list(zip(s, s[1:]))
    before = sorted(b - x for x, b in gaps if b < t0)
    over = sorted(b - x for x, b in gaps if t1 is not None and b > t0 and x < t1)

    def ms(v, q):
        return round(1e3 * v[min(len(v) - 1, int(q * len(v)))], 1) if v else None

    print(json.dumps({"case": "mixed/summary", "prefill_tokens": a.mixed, "prefill_ttft_s": lr["ttft_s"],
                      "decoder_tokens_in_prefill_window": sum(1 for x in s if t1 is not None and t0 <= x <= t1),
                      "decoder_gap_ms_before_p50": ms(before, 0.5), "decoder_gap_ms_in_window_p50": ms(over, 0.5),
                      "decoder_gap_ms_in_window_max": ms(over, 1.0), "decoder_error": dec.get("error"),
                      "prefill_error": lr["error"]}), flush=True)  # fmt: skip
