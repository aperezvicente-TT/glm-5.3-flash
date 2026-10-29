# GLM-5.3-Flash on Blackhole: getting started

OpenAI-compatible serving of zai-org/GLM-5.3-Flash (321B, 18B active) on Tenstorrent Blackhole: four TP8 servers,
one per 2x4 tray, behind one router. Release 0.1.0 (2026-10-29).

Deploying it from source: the `README.md` and `BUILD-FROM-SOURCE.md` in this repository
(`github.com/aperezvicente-TT/glm-5.3-flash`).

## Call it

The router is the one endpoint: `http://<host>:8110` (localhost on the serving host; put your own proxy and auth
in front of it). Model id `zai-org/GLM-5.3-Flash`. Always send `max_tokens`.

```
curl -s http://127.0.0.1:8110/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "zai-org/GLM-5.3-Flash",
  "messages": [{"role": "user", "content": "Summarize the attached report in five bullets."}],
  "max_tokens": 1024, "temperature": 0.6, "top_p": 0.95,
  "chat_template_kwargs": {"enable_thinking": false}
}'
```

`/v1/completions`, `/v1/chat/completions`, streaming and tool calls (`tool_choice: auto`) work. Host-sampling
parameters (`logprobs`, `min_p`, `logit_bias`, `bad_words`, `allowed_token_ids`, `min_tokens`, penalties,
`response_format`, `tool_choice: required`) are refused with HTTP 400 in this release.

## What to expect

Through the router, random-token prompts, 1,024 output tokens each. "Users" = fleet-wide concurrency over the four
trays. Each cell: time to first token (P50), then that user's own decode rate in tokens per second.

| Prompt | 1 user | 4 users | 8 users | 16 users | fleet output at 16 users | returning user (stored context) |
|---|---|---|---|---|---|---|
| 128 | 0.20 s, 58 tok/s | 0.25 s, 36 tok/s | 0.34 s, 40 tok/s | 0.35 s, 34 tok/s | 467 tok/s | |
| 1K | 0.38 s, 58 tok/s | 0.42 s, 40 tok/s | 0.46 s, 39 tok/s | 1.3 s, 35 tok/s | 483 tok/s | |
| 8K | 2.3 s, 52 tok/s | 4.5 s, 35 tok/s | 4.8 s, 36 tok/s | 7.0 s, 32 tok/s | 343 tok/s | 0.42 s |
| 32K | 5.3 s, 52 tok/s | 14.9 s, 33 tok/s | 13.8 s, 33 tok/s | 16 s, 29 tok/s | 227 tok/s | 0.61 s |
| 128K | 27 s, 51 tok/s | 28 s, 42 tok/s | 16 s, 36 tok/s | 19 s, 31 tok/s | 105 tok/s | 1.4 s |
| 1M | ~270 s, 52 tok/s | one per tray | | | | 8.8 s |

Real-text prompts (AIME) on one tray: 58 tok/s for one user, 38 at 2, 33 at 4, 26.5 at 8 (131 tok/s aggregate per
tray); a mixed 16-user load with 1M contexts in it decodes at about 21 tok/s per user. Every tray holds 1,048,576
tokens of context and 8 decode slots (32 fleet-wide); finished contexts are stored and a repeated prefix returns in the
"returning user" time.

## Four rules

1. A 1M prompt takes a whole tray for its prefill (about 4.5 minutes); run at most one per tray.
2. Spread long prompts over trays. A long arrival queued behind a resident 1M waits for it.
3. Through the router, `max_tokens` defaults to 16,384 and is clamped at 65,536.
4. Calling a tray directly (`:8100` to `:8103`) bypasses the router's rules; keep `max_tokens x users` within its pool.

## When something is wrong

`GET http://<host>:8110/health` lists the trays that are up. A down tray restarts with the kit's stop and up scripts
(about 12 minutes); the router routes around it meanwhile. Known issues and fix status: the release report, section 3.
