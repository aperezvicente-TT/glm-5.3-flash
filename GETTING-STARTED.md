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

| Prompt | One user, cold | Returning user (stored context) | 16 users |
|---|---|---|---|
| 128 tokens | 0.20 s to first token, 58 tok/s | | 0.35 s, 34 tok/s each |
| 8K | 2.3 s, 52 tok/s | 0.42 s | 7.0 s, 32 each |
| 32K | 5.3 s, 52 | 0.61 s | 16 s, 29 each |
| 128K | 27 s, 51 | 1.43 s | 19 s, 31 each |
| 1M | about 270 s, 52 | 8.8 s | one per tray |

Every tray holds 1,048,576 tokens of context and 8 decode slots (32 fleet-wide). Finished contexts are stored and
a repeated prefix returns in the "returning user" time.

## Four rules

1. A 1M prompt takes a whole tray for its prefill (about 4.5 minutes); run at most one per tray.
2. Spread long prompts over trays. A long arrival queued behind a resident 1M waits for it.
3. Through the router, `max_tokens` defaults to 16,384 and is clamped at 65,536.
4. Calling a tray directly (`:8100` to `:8103`) bypasses the router's rules; keep `max_tokens x users` within its pool.

## When something is wrong

`GET http://<host>:8110/health` lists the trays that are up. A tray that is down restarts with the kit's stop and
up scripts (about 12 minutes); the router routes around it meanwhile. Known issues and their fix status: the
release report, section 3.
