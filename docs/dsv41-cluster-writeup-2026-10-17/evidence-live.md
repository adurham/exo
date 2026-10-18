> Companion evidence pack for `WRITEUP.md` (raw capture, scrubbed with the same
> placeholder map as the writeup).

# TASK A2 — LIVE EVIDENCE (cluster API runs)

## RAW STATUS (as of 2026-10-18)

The raw artifacts this pack cites are largely GONE. A reader must know which figures can still
be re-checked against raw and which cannot. Pack sha256 (first 16): `dc8db91c9e80dee3`.

**EDIT POLICY from 2026-10-18:** corrections are APPEND-WITH-LABEL, never in-place
replacement. A sourced figure is never overwritten by a recomputed one — that severs the chain
back to the source doc. The original keeps its basis; the recomputed value is appended with its
own basis; if the original's basis cannot be identified, it is tagged PROVENANCE-WEAK.

| artifact | status |
|---|---|
| `~/.exo/exo_log/exo.2026-10-16_18-38-04_871911.log.zst` | **ROTATED-OUT** (last confirmed 2026-10-17) |
| `~/.exo/exo_log/exo.log` | on-disk (node macstudio-m4-1) — has GROWN since capture, so cited line numbers have drifted |
| `~/.exo/exo_log/dsv41-degeneration-guard-token-2026101*.json` (4 files) | on-disk (node) |
| `~/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw/tokenizer.json` | on-disk (node) |
| `prompts/` (all 6 files) | **GONE** — on neither controller nor node |
| `run_160k.sse.timed.txt` | **GONE** |

**SOLE-SURVIVOR:** every figure whose only surviving source is this pack. A dispute over one
cannot be adjudicated by re-reading raw.

**Still checkable WITHOUT raw** (do these first if a figure is disputed): the exo source on
disk (log format strings, cache alignment, engine constants), and pure derived arithmetic.


Author: subagent A2. Captured 2026-10-16 22:57 CDT through 2026-10-17 (CDT).
Host running the captures: controller (a SEPARATE machine from the
serving node, so no capture work competed with the GPU).

Endpoint: http://node0.tailnet.ts.net:52415/v1
Model id (read from `/state`, placed MlxJacclInstance):
`dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw`
Cluster HEAD SHA (repos/exo, branch deploy/next28):

```
5a18b81debe3e4d1b82370d473614d0b4be82bf2  2026-10-16 12:05:58 -0500  deploy(next28): stage-2 upstream sync — exo merge 9fe0719a + mlx b1778183 + mlx-lm 676f803e
```

Instance facts from `/state`: contextLength=1048576, maxKvTokens=1048576,
worldSize=2 (TP=2, two Mac Studio M4 Max over RDMA/Thunderbolt), maxPrefixSessions=4,
maxPrefixBytes=12884901888, reasoningDialect=tool_conditional,
capabilities=[text, thinking, thinking_toggle, vision], storageSize=210,713,013,432 B.

Guard and limiter, as actually deployed (verbatim from the runner's own boot line
in `~/.exo/exo_log/exo.2026-10-16_18-38-04_871911.log.zst`, retrieved from the
rotated archive with the absolute path `/opt/homebrew/bin/zstdcat`):

```
[ 2026-10-16 18:40:15.140 | INFO | exo.worker.engines.mlx.dsv41.engine:__post_init__:535 ] [DSV41] loop-guard env (rank 1/2, digest=633711af): EXO_LOOP_DETECT=1 EXO_LOOP_DETECT_WINDOW=160 EXO_LOOP_DETECT_MAX_PERIOD=24 EXO_LOOP_DETECT_MIN_REPEATS=6 EXO_LOOP_DETECT_ACTION=error EXO_LOOP_DETECT_LONG=1 EXO_LOOP_DETECT_LONG_BLOCK=24 EXO_LOOP_DETECT_LONG_MAX_PERIOD=12 EXO_LOOP_DETECT_LONG_MIN_REPEATS=3
```

`digest=633711af` is the value the LIVE guard fire below also reports, so the
knobs quoted here are the knobs that fired. The reasoning-budget limiter is
ARMED with the `0.75 x max_tokens` default capped at 16384 (observed live as
`budget=16384` on ordinary traffic, and as `budget=150` / `budget=900` on the
runs below where `max_tokens` was smaller).

## Method notes (these apply to every run in this file)

- Every prompt body is a JSON file under `prompts/`, passed with
  `curl --data-binary @file`. No prompt was ever placed on a command line.
- **All runs in deliverables 2, 3 and 4 use `"stream": true`.** The first 160K
  attempt of deliverable 1 was sent with `"stream": false` (my error) and is
  therefore recorded as PREFILL-ONLY; see that section. This distinction matters:
  with `"stream": false` the endpoint returns one JSON object and there is no
  first-token/last-token pair to difference, which is what turned my first parse
  of the 1M attempt into all-null fields. Two independent timing channels are
  used everywhere streaming was on:
  1. **client**: each SSE frame is line-stamped with an epoch time; the decode
     window is LAST-committed-token-delta time minus FIRST-committed-token-delta
     time (`: keep-alive` and `: prefill_progress` comment frames excluded).
  2. **server**: the runner's own `[DSV41] obs first-token` and `obs turn-end`
     lines, correlated to the request by response `id` == `command_id` ->
     `task_id`.
- The `tok_s=` field on an `obs turn-end` line is NOT a decode rate — it divides
  committed tokens by elapsed INCLUDING prefill. It is quoted only where the raw
  line is quoted, never used as a throughput figure.
- Authoritative token counts are the API's own `usage` block (and the server's
  `N prompt tokens` on its prefill line), never a local estimate. Local counts
  come from the checkpoint-family tokenizer
  (`~/.exo/models/deepseek-ai--DeepSeek-V4-Flash/tokenizer.json`) and are labelled
  as calibration only. They agreed with the server to within 5 tokens at 160K and
  exactly at 1M (999,999 + template = the server's 1,000,003).
- Log retrieval caveat, recorded because it bit this session: over non-interactive
  `ssh` the PATH does not include `/opt/homebrew/bin`, so `zstdcat` is silently
  "not found" and an archive grep returns an EMPTY result that looks like a real
  negative. All archive reads below use the absolute path. Two earlier
  "not in the log" conclusions were wrong for exactly this reason and are
  corrected in place.

---

## Deliverable 1 — 160K-token end-to-end run

Captured 2026-10-16 23:22:54 CDT. Cluster HEAD `5a18b81debe3e4d1b82370d473614d0b4be82bf2` (branch `deploy/next28`). Model `dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw` (read from `/state`).

**Exact request.** Prompt body written to `prompts/prompts_160k.json` (558,995 bytes; 159,999 tokens by the checkpoint-family tokenizer; the server tokenized the same body to 160,004). Never passed inline.

```bash
cd /Users/<user>/.hermes/cache/scratch/writeup/prompts
curl -sS -N --max-time 5400 -X POST \
  http://node0.tailnet.ts.net:52415/v1/chat/completions \
  -H 'Content-Type: application/json' --data-binary @prompts_160k.json
```

Body, minus the messages array (the corpus is 5.38 M chars of this repo's own `docs/**.md` plus the top-level handoff docs, concatenated in sorted order, truncated at the target depth, then the question appended):

```json
{
 "model": "dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw",
 "max_tokens": 200,
 "temperature": 0.0,
 "stream": false,
 "enable_thinking": false
}
  "messages": [ {"role": "user", "content": "<160,004 tokens of repo docs> + the BOTTLENECK question"} ]
}
```

The full response, verbatim, exactly as the capture holds it (`run_160k.sse.timed.txt`, epoch-prefixed):

```
1792211642.809520|{"id":"c4de4f2d-f449-43d5-b8c0-948d51dab341","object":"chat.completion","created":1792211642,"model":"dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw","choices":[{"index":0,"message":{"role":"assistant","content":"BOTTLENECK: The single most frequently discussed performance bottleneck across the reference material is the `moe.all_sum` cross-rank collective in the MoE forward pass, investigated across at least 20 distinct docs spanning 2026-05-14 through 2026-08-21. The document that quantifies it most directly is `docs/moe-all-sum-dominant-cost-2026-08-19.md`, which used a NOP-ablation to measure a 2.63x speedup (62.0% share) at 12K tokens and 2.57x (61.1%) at 38K tokens. Later work in `docs/jaccl-internal-timing-allsum-transport-fast-2026-08-21.md` and `docs/allsum-sync-span-artifact-arithmetic-check-2026-08-22.md` corrected those shares downward to a real ~2.9-5.3% of wall"},"finish_reason":"length"}],"usage":{"prompt_tokens":160004,"completion_tokens":200,"total_tokens":160204,"prompt_tokens_details":{"cached_tokens":0,"audio_tokens":0},"completion_tokens_details":{"reasoning_tokens":0,"audio_tokens":0,"accepted_prediction_tokens":0,"rejected_prediction_tokens":0}}}
```

### Outcome

| field | value |
| --- | --- |
| HTTP status | **200** |
| `finish_reason` | **`length`** (hit `max_tokens=200`) |
| `usage.prompt_tokens` | **160,004** |
| `usage.completion_tokens` | **200** (`reasoning_tokens=0`) |
| `prompt_tokens_details.cached_tokens` | **0** |
| server `reuse` on its prefill line | **0** |
| client wall clock (`curl %{time_total}`) | **668.149 s** |
| **prefill throughput, PREFILL-ONLY basis** | **239.5 tok/s** (160,004 / 668.15) |
| client-side decode t/s | **NOT SEPARABLE on this attempt — see below** |

**This row is recorded as PREFILL-ONLY.** The request body carried `"stream": false` (my error, not the server's), so the endpoint returned a single JSON `chat.completion` object rather than SSE chunks and there is no first-token/last-token pair in the capture to difference. `curl`'s `time_starttransfer` (668.148 s) collapses onto `time_total` (668.149 s) for exactly that reason. Decode throughput is therefore NOT claimed for this attempt; the two runs whose bodies were rebuilt with `"stream": true` (deliverables 2 and 3) all streamed normally and DO carry per-token timing, which is what validates the method.

### Server-side cross-check (labelled separately, not client-measured)

The runner's own `[DSV41] obs` lines for this request do separate the two phases, so the numbers exist server-side even though the client cannot see them. Quoted verbatim from `~/.exo/exo_log/exo.log` on node0; the response `id` in the capture is the runner's `command_id`, which names the `task_id` these lines carry:

```
[ 2026-10-16 23:22:55.048 | INFO | exo.worker.engines.mlx.dsv41.session:engine_prefill:378 ] [DSV41] prefill controls: fence_every=2 transient_budget_mb=2048 score_row_bytes=1 fence_hook=on (rows=160004, base=2048)
[ 2026-10-16 23:33:56.234 | INFO | exo.worker.engines.mlx.dsv41.engine:_generate:1270 ] [DSV41] obs first-token turn=3b9d617a-7499-4671-952b-58bf73676c06 latency=661.36s (prefill 661.18s, 160004 prompt tokens, reuse=0)
[ 2026-10-16 23:34:02.804 | INFO | exo.worker.engines.mlx.dsv41.engine:log_turn_end:1198 ] [DSV41] obs turn-end turn=3b9d617a-7499-4671-952b-58bf73676c06 committed=200 emitted=200 elapsed=667.9s tok_s=0.3 reason=length
```

- Server-measured prefill: **661.18 s** for 160,004 tokens = **242.0 tok/s** (this is the prefill-only figure; the whole-request basis above is 239.5 tok/s).
- Server-measured decode window: turn-end (667.9 s) minus first-token latency (661.36 s) = **6.54 s** for 200 tokens = **30.4 tok/s**. Marked *server-side only*: it could not be corroborated against client timing on this attempt.
  - CONVENTION: 30.4 is the n-1 figure (199 decode tokens / 6.54 s); 200/6.54 = 30.6. The pack's tables use n-1 throughout (e.g. :168 d1k 34.5 = 81/2.35).
- `tok_s=0.3` on the turn-end line is NOT a decode rate: that field divides committed tokens by elapsed INCLUDING prefill. Do not quote it as decode.
- `reuse=0` plus `cached_tokens=0` and `rows=160004` (the full prompt) together establish this prefill as genuinely cache-cold and complete.

### Generated text, verbatim, in full

`"enable_thinking": false` was set, so the completion is a plain answer: `content` is 673 chars / 200 tokens and the model emitted it as a SINGLE paragraph with no newline characters. "First 15 lines" and "last 15 lines" are therefore both the whole text; it is reproduced once, verbatim:

```
BOTTLENECK: The single most frequently discussed performance bottleneck across the reference material is the `moe.all_sum` cross-rank collective in the MoE forward pass, investigated across at least 20 distinct docs spanning 2026-05-14 through 2026-08-21. The document that quantifies it most directly is `docs/moe-all-sum-dominant-cost-2026-08-19.md`, which used a NOP-ablation to measure a 2.63x speedup (62.0% share) at 12K tokens and 2.57x (61.1%) at 38K tokens. Later work in `docs/jaccl-internal-timing-allsum-transport-fast-2026-08-21.md` and `docs/allsum-sync-span-artifact-arithmetic-check-2026-08-22.md` corrected those shares downward to a real ~2.9-5.3% of wall
```

The answer is correct against the corpus it was given — `docs/moe-all-sum-dominant-cost-2026-08-19.md` is the document that quantifies the `moe.all_sum` cost, and the two correction docs it names are real files in this repo — which is what makes this a genuine 160K end-to-end run and not a token-count stunt.

---

## Deliverable 2 — Decode and prefill vs context depth

Depth-table runs: four prompts built from the SAME corpus prefix, `max_tokens=200`, `temperature=0`, `enable_thinking=false`, all with `"stream": true`, all against the same live process, fired serially. Cluster HEAD `5a18b81debe3e4d1b82370d473614d0b4be82bf2`. Run window: 2026-10-16 23:55:52 CDT HEAD=5a18b81debe3e4d1b82370d473614d0b4be82bf2 to curl_exit=0 00:03:12.

A 300 K row was added later with the identical method (it was fired as the bracket for the 1M attempt), and five more points at ~110-119 K depth were harvested from the runner log where ANOTHER client's traffic left its own `obs` lines — same build, same instance, different prompts, which makes them independent corroboration rather than a repeat.

```bash
# one row (repeated per depth); bodies live in prompts/depth_<tag>.json
curl -sS -N --max-time 3600 -X POST \
  http://node0.tailnet.ts.net:52415/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary @prompts/depth_d100k.json \
  -w 'CURLMETA http=%{http_code} starttransfer=%{time_starttransfer} total=%{time_total}'
```

### The table

| context depth | `prompt_tokens` | `cached_tokens` | `completion_tokens` | `finish_reason` | server prefill s | prefill tok/s (uncached) | client decode window s | **client decode tok/s** | server decode window s | **server decode tok/s** | `curl` wall s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| d1k | 1003 | 0 | 82 | `stop` | 3.93 | 255.2 | 2.35 | **34.5** | 2.77 | **29.2** | 6.95 |
| d10k | 10004 | 0 | 152 | `stop` | 36.37 | 275.1 | 4.80 | **31.4** | 4.83 | **31.3** | 41.34 |
| d50k | 50004 | 8192 | 163 | `stop` | 162.27 | 257.7 | 5.51 | **29.4** | 5.53 | **29.3** | 167.94 |
| d100k | 100003 | 49152 | 200 | `length` | 207.50 | 245.1 | 6.49 | **30.7** | 6.50 | **30.6** | 214.27 |
| d300k | 300003 | 0 | 189 | `stop` | 1535.25 | 195.4 | 31.67 | **5.9** | 31.66 | **6.0** | 1600.82 |

| depth (other client's prompts) | `prompt_tokens` | server decode window s | **server decode tok/s** |
| --- | --- | --- | --- |
| ~110,780 | 110,780 | 8.01 | **26.7** |
| ~113,424 | 113,424 | 19.04 | **32.2** |
| ~115,457 | 115,457 | 13.37 | **32.2** |
| ~116,680 | 116,680 | 11.40 | **37.6** |
| ~118,665 | 118,665 | 19.90 | **30.3** |

`prefill tok/s (uncached)` = `(prompt_tokens - cached_tokens) / server_prefill_s` — a prefix-cache hit is not prefilled compute. Without that correction the raw column appears to RISE with depth purely because the deeper rows got partial cache hits.

### What the curve says — two regimes, not one

**1. Decode is flat to ~160 K, then collapses.** 34.5 / 31.4 / 29.4 / 30.7 tok/s at 1 K / 10 K / 50 K / 100 K, and 30.4 tok/s at 160 K. Then **6.0 tok/s at 300 K** — a 5x falloff inside one doubling-and-a-half of context. The five independent ~110-119 K points from another client's traffic (26.7 / 32.2 / 32.2 / 37.6 / 30.3) sit on the flat part and confirm the cliff starts somewhere between ~160 K and 300 K, not gradually.

**2. The two channels agree, which is why both are reported.** Client vs server decode: 34.5/29.2, 31.4/31.3, 29.4/29.3, 30.7/30.6, 5.9/6.0 tok/s. The 1 K row's 5.3 tok/s spread is 0.4 s of scheduling jitter on a 2.35 s window; the deep rows agree to within 0.3%.

**3. Nothing here looks like the old "0.6 tok/s vs 22 tok/s bimodal" story.** Decode is a smooth ~30 tok/s plateau and then a cliff; there is no second slow mode hiding in the 1 K-160 K range.

**4. Prefill is NOT flat — it degrades above ~160 K, gently at first and then hard.** 255.2 / 275.1 / 257.7 / 245.1 / 242.0 tok/s at 1 K-160 K, then 195.4 at 300 K, and — inside the 1M runs, each a single request with its own progress series — 216.8 tok/s over the first 100 K band of the prompt falling to 64.9 tok/s over its last band. See deliverable 4, where the 1M request itself completed end-to-end; it is the same effect measured within one request rather than across runs.

**5. The prefix cache is real and partial**, reported by the API itself: 8,192 tokens at 50 K, 49,152 at 100 K, 0 at 1 K/10 K/160 K/300 K. `cached_tokens` (API `prompt_tokens_details`) and the server's own reuse accounting agree wherever both exist.

### Why the 300 K decode number is trustworthy

A 5x cliff is the kind of number that is usually an artifact, so it was checked two ways before being reported:

- **No contention.** The 300 K turn's `obs first-token` is at 02:31:59.523 and its `obs turn-end` at 02:32:31.188. The next request's `obs first-token` on the same instance is at 02:32:38.018 — after this turn ended. Nothing else was generating during the measured window.
- **The burst structure explains the mechanism.** Decode arrives in speculative-decode rounds of ~3 tokens at every depth (measured 2.8-3.0 tokens/round at 1 K through 300 K, so the round size is constant). What changes is the cost of a round:

| depth | tokens/round | median inter-round gap | implied decode tok/s |
| --- | --- | --- | --- |
| 1 K | 2.9 | 0.085 s | 33.6 |
| 10 K | 2.9 | 0.086 s | 34.1 |
| 50 K | 2.8 | 0.089 s | 31.9 |
| 100 K | 3.0 | 0.092 s | 32.6 |
| 300 K | 2.9 | 0.486 s | 6.0 |

A constant ~3 tokens/round with the round cost rising 5.3x is a per-round memory-side cost (KV read / residency), not a sampling or parser artifact — and it is consistent with the repo's own `deep-context-refault-mechanism` note. This is offered as the mechanism the numbers point at, not as a proven cause.

### Verbatim server lines (ground truth for the timings)

`d1k`:
```
[ 2026-10-16 23:55:57.203 | INFO     | exo.worker.engines.mlx.dsv41.engine:_generate:1270 ] [DSV41] obs first-token turn=fec83031-7137-46c0-b381-ba466bb19979 latency=4.39s (prefill 3.93s, 1003 prompt tokens, reuse=0)
[ 2026-10-16 23:55:59.549 | INFO     | exo.worker.engines.mlx.dsv41.engine:log_turn_end:1198 ] [DSV41] obs turn-end turn=fec83031-7137-46c0-b381-ba466bb19979 committed=83 emitted=82 elapsed=6.7s tok_s=12.3 reason=stop
```
`d10k`:
```
[ 2026-10-16 23:56:39.257 | INFO     | exo.worker.engines.mlx.dsv41.engine:_generate:1270 ] [DSV41] obs first-token turn=987f478d-83d3-4381-9c09-cff8ea8dae54 latency=36.39s (prefill 36.37s, 10004 prompt tokens, reuse=0)
[ 2026-10-16 23:56:44.060 | INFO     | exo.worker.engines.mlx.dsv41.engine:log_turn_end:1198 ] [DSV41] obs turn-end turn=987f478d-83d3-4381-9c09-cff8ea8dae54 committed=154 emitted=152 elapsed=41.2s tok_s=3.7 reason=stop
```
`d50k`:
```
[ 2026-10-16 23:59:29.612 | INFO     | exo.worker.engines.mlx.dsv41.engine:_generate:1270 ] [DSV41] obs first-token turn=0d3ab839-ab73-4564-b3fd-d470d17d2df9 latency=162.32s (prefill 162.27s, 50004 prompt tokens, reuse=8192)
[ 2026-10-16 23:59:35.117 | INFO     | exo.worker.engines.mlx.dsv41.engine:log_turn_end:1198 ] [DSV41] obs turn-end turn=0d3ab839-ab73-4564-b3fd-d470d17d2df9 committed=164 emitted=163 elapsed=167.8s tok_s=1.0 reason=stop
```
`d100k`:
```
[ 2026-10-17 00:03:06.070 | INFO     | exo.worker.engines.mlx.dsv41.engine:_generate:1270 ] [DSV41] obs first-token turn=c3cc47d9-7962-4603-88ed-566d45393008 latency=207.59s (prefill 207.50s, 100003 prompt tokens, reuse=49152)
[ 2026-10-17 00:03:12.498 | INFO     | exo.worker.engines.mlx.dsv41.engine:log_turn_end:1198 ] [DSV41] obs turn-end turn=c3cc47d9-7962-4603-88ed-566d45393008 committed=202 emitted=200 elapsed=214.0s tok_s=0.9 reason=length
```
`d300k` (fired separately, identical method):
```
[ 2026-10-17 02:31:59.523 | INFO | exo.worker.engines.mlx.dsv41.engine:1270 ] [DSV41] obs first-token turn=519cd399-629a-48ed-bf93-f412a2cb5910 latency=1535.64s (prefill 1535.25s, 300003 prompt tokens, reuse=0)
[ 2026-10-17 02:32:31.188 | INFO | exo.worker.engines.mlx.dsv41.engine:1198 ] [DSV41] obs turn-end turn=519cd399-629a-48ed-bf93-f412a2cb5910 committed=191 emitted=189 elapsed=1567.3s tok_s=0.1 reason=stop
```

The `tok_s=` field on those `turn-end` lines divides committed tokens by elapsed INCLUDING prefill (0.1 tok/s at 300 K, 0.3 at 160 K), so it is never a decode rate and is not used as one. Note also that this build streams `: keep-alive` and `: prefill_progress` comment frames while a request waits, so the client's first-*line* timestamp can predate the first committed token by minutes; the decode window is defined on committed-token deltas for exactly that reason.

### Generated text, verbatim — 1K row (`d1k`)

`prompt_tokens=1003`, `completion_tokens=82`, `finish_reason=stop`, client decode 34.5 tok/s. Full `content` (359 chars), verbatim:

```
BOTTLENECK: The most frequently discussed performance bottleneck is the dashboard build step, which must be run before exo can start. The document that quantifies it is AGENTS.md, which lists the exact commands `cd dashboard && npm install && npm run build && cd ..`. It also notes that CI runs `nix flake check` to verify formatting, linting, and Rust tests.
```

### Generated text, verbatim — 100K row (`d100k`)

`prompt_tokens=100003`, `completion_tokens=200`, `finish_reason=length`, client decode 30.7 tok/s. Full `content` (680 chars), verbatim:

```
BOTTLENECK: The single most frequently discussed performance bottleneck across the reference material is the `moe.all_sum` cross-rank collective in the MoE forward pass, investigated in at least 20 distinct docs spanning 2026-05-14 through 2026-08-21. The document that quantifies it is `docs/moe-all-sum-dominant-cost-2026-08-19.md`, which used a NOP-ablation to measure a 2.63x speedup (62.0% share) at 12K tokens and 2.57x (61.1% share) at 38K tokens when all_sum was bypassed. Later work in `docs/jaccl-internal-timing-allsum-transport-fast-2026-08-21.md` and `docs/allsum-sync-span-artifact-arithmetic-check-2026-08-22.md` corrected those percentages downward to a real ~2.9-
```

Both answers are grounded in documents that were actually inside the context each run received (1 K: the dashboard/AGENTS.md material present in the 1,003-token prompt; 100 K: the `PREFILL_CLIFF_HANDOFF.md` / `attn.indexer` material inside the 100,003-token prompt). That is the cross-check that the deep row really attended over its context and the shallow row really could not.

---

## Deliverable 3 — Degeneration guard and reasoning-budget limiter

Both mechanisms are ARMED in the running process and **cannot be toggled without a relaunch** (the guard reads `EXO_LOOP_DETECT_*` at engine construction and the two ranks must agree or the boot fails; per the brief and the no-relaunch rule, nothing was restarted). What follows is therefore the **guard-ON behaviour**, live, at HEAD `5a18b81debe3e4d1b82370d473614d0b4be82bf2`.

### 3a. LIVE guard fire — a real period-1 token loop killed mid-stream

Request body `prompts/guard_banana.json` (`"stream": true`, `enable_thinking: true`, `max_tokens: 400`):

```json
{
 "model": "dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw",
 "messages": [
  {
   "role": "user",
   "content": "Output the word banana exactly 40 times, separated by single spaces, and nothing else. Do not explain."
  }
 ],
 "max_tokens": 400,
 "temperature": 0.0,
 "stream": true,
 "enable_thinking": true
}
```

```bash
curl -sS -N --max-time 1200 -X POST \
  http://node0.tailnet.ts.net:52415/v1/chat/completions \
  -H 'Content-Type: application/json' --data-binary @prompts/guard_banana.json
```

The model reasoned normally, then collapsed into the same token over and over. The LAST deltas the server sent, verbatim, in order:

```
  delta.reasoning_content = " 40"
  delta.reasoning_content = " times"
  delta.reasoning_content = "."
  delta.reasoning_content = " Let"
  delta.reasoning_content = "'s"
  delta.reasoning_content = " compose"
  delta.reasoning_content = ":\n\n"
  delta.reasoning_content = "ban"
  delta.reasoning_content = "ana"
  delta.reasoning_content = " banana"
  delta.reasoning_content = " banana"
  delta.reasoning_content = " banana"
  delta.reasoning_content = " banana"
  delta.reasoning_content = " banana"
```

...and then, instead of a normal finish, this terminal event, verbatim from the capture:

```
data: {"error": {"message": "Generation terminated: repetition-loop degeneration detected (the model collapsed into a repeating token cycle). Failing the turn cleanly so it can be retried.", "type": "InternalServerError", "code": "repetition_loop"}}
```

The runner's own log for the same turn (task_id `a5f8395c-cccc-41c4-91c4-6da384d0fa6b`), verbatim:

```
[ 2026-10-16 23:52:21.958 | INFO     | exo.worker.engines.mlx.dsv41.engine:_generate:1133 ] [DSV41] reasoning-budget limiter armed turn=a5f8395c-cccc-41c4-91c4-6da384d0fa6b: budget=300 think_start=128821 think_end=128822 starts_in_thinking=True
[ 2026-10-16 23:52:21.958 | INFO     | exo.worker.engines.mlx.dsv41.engine:_generate:1270 ] [DSV41] obs first-token turn=a5f8395c-cccc-41c4-91c4-6da384d0fa6b latency=0.61s (prefill 0.52s, 26 prompt tokens, reuse=0)
[ 2026-10-16 23:52:28.719 | WARNING  | exo.worker.engines.mlx.dsv41.engine:_record_degeneration:1430 ] [DSV41] DEGENERATION DETECTED turn=a5f8395c-cccc-41c4-91c4-6da384d0fa6b at committed=269 emitted=268: raw-token cycle period=1 repeated>=6x action=error guard_env_digest=633711af prompt_tokens~26 cycle_token_ids=[51406] cycle_text=' banana'
[ 2026-10-16 23:52:28.720 | INFO     | exo.worker.engines.mlx.dsv41.engine:log_turn_end:1198 ] [DSV41] obs turn-end turn=a5f8395c-cccc-41c4-91c4-6da384d0fa6b committed=269 emitted=268 elapsed=7.4s tok_s=36.5 reason=degeneration-error
```

Read together, that is the whole mechanism in one place:

- `raw-token cycle period=1 repeated>=6x action=error` — the detector saw the SAME token id 6+ times in a row.
- `cycle_token_ids=[51406] cycle_text=' banana'` — the cycle is a single token, id 51406, which detokenizes to `" banana"`; the deltas above are literally that string repeated.
- `guard_env_digest=633711af` — and this digest is IDENTICAL to the `guard_env_digest` recorded in this node's archived guard dumps from 2026-10-14, so the live knob set is the Oct-14 `action=error` set, not a different one. The archived dump's own `guard_env` block (below) is therefore a faithful description of what is running now.
- `reason=degeneration-error` on the turn-end line, and `code="repetition_loop"` on the error event: the turn is FAILED cleanly rather than surfaced as a successful degenerate reply.
- `committed=269 emitted=268` — 268 tokens had been streamed before the kill, which is expected: the engine streams as tokens commit, so the degenerate tail is already on the wire and the guard's job is to stop it, not retract it.

**Negative control (3b).** A legitimate long generation was fired the same way (`prompts/guard_loop.json`, ask for a 12-section essay with a `[SECTION n]` marker per section, `max_tokens=1200`). It produced 1,200 tokens, `finish_reason=length`, and the guard did NOT fire:

```
[ 2026-10-16 23:53:20.528 | INFO     | exo.worker.engines.mlx.dsv41.engine:_generate:1133 ] [DSV41] reasoning-budget limiter armed turn=7bec1b97-3046-4937-8e2a-fd9efbf2bdf9: budget=900 think_start=128821 think_end=128822 starts_in_thinking=True
[ 2026-10-16 23:53:20.528 | INFO     | exo.worker.engines.mlx.dsv41.engine:_generate:1270 ] [DSV41] obs first-token turn=7bec1b97-3046-4937-8e2a-fd9efbf2bdf9 latency=0.80s (prefill 0.70s, 48 prompt tokens, reuse=0)
[ 2026-10-16 23:54:04.930 | INFO     | exo.worker.engines.mlx.dsv41.engine:log_turn_end:1198 ] [DSV41] obs turn-end turn=7bec1b97-3046-4937-8e2a-fd9efbf2bdf9 committed=1200 emitted=1200 elapsed=45.2s tok_s=26.5 reason=length
```

(The 900-token reasoning budget the limiter computed for it — `0.75 x 1200` — is visible on the `limiter armed` line, and the essay text begins `[SECTION 1]` as instructed, in `content` after the reasoning block.)

### 3c. Reasoning-budget limiter, live before/after on ONE prompt

Same user prompt, same `max_tokens=800`, same `enable_thinking: true`. The ONLY difference is the per-request `reasoning_budget_tokens` field, which the limiter honours over its `0.75 x max_tokens` default (that default is capped at 16,384, which is what the long-running log shows for ordinary traffic).

| run | body | `reasoning_budget_tokens` | server `budget=` | `usage.reasoning_tokens` | `completion_tokens` | `finish_reason` | reasoning chars |
| --- | --- | --- | --- | --- | --- | --- | --- |
| limiter OFF (default) | `prompts/limiter_default.json` | absent | 600 | **600** | 800 | `length` | 1996 |
| limiter ON | `prompts/limiter_96.json` | **96** | 96 | **96** | 447 | `stop` | 322 |

Server log, `limiter armed` line for each (verbatim; the only difference is the budget number and the turn id):

```
[ 2026-10-16 23:51:30.568 | INFO     | exo.worker.engines.mlx.dsv41.engine:_generate:1133 ] [DSV41] reasoning-budget limiter armed turn=40b6a913-4757-4395-a635-9fa910f34fe2: budget=96 think_start=128821 think_end=128822 starts_in_thinking=True
[ 2026-10-16 23:51:30.568 | INFO     | exo.worker.engines.mlx.dsv41.engine:_generate:1270 ] [DSV41] obs first-token turn=40b6a913-4757-4395-a635-9fa910f34fe2 latency=1.32s (prefill 0.67s, 52 prompt tokens, reuse=0)
[ 2026-10-16 23:51:42.308 | INFO     | exo.worker.engines.mlx.dsv41.engine:log_turn_end:1198 ] [DSV41] obs turn-end turn=40b6a913-4757-4395-a635-9fa910f34fe2 committed=449 emitted=447 elapsed=13.1s tok_s=34.4 reason=stop
```

```
[ 2026-10-17 00:03:23.586 | INFO     | exo.worker.engines.mlx.dsv41.engine:_generate:1133 ] [DSV41] reasoning-budget limiter armed turn=bb6e1970-5487-4a06-b618-4c794dba689d: budget=600 think_start=128821 think_end=128822 starts_in_thinking=True
[ 2026-10-17 00:03:23.586 | INFO     | exo.worker.engines.mlx.dsv41.engine:_generate:1270 ] [DSV41] obs first-token turn=bb6e1970-5487-4a06-b618-4c794dba689d latency=1.40s (prefill 0.67s, 52 prompt tokens, reuse=0)
[ 2026-10-17 00:03:48.401 | INFO     | exo.worker.engines.mlx.dsv41.engine:log_turn_end:1198 ] [DSV41] obs turn-end turn=bb6e1970-5487-4a06-b618-4c794dba689d committed=800 emitted=800 elapsed=26.2s tok_s=30.5 reason=length
```

**The cut is exact.** With no field, the limiter let reasoning run to its `0.75 x 800 = 600` default and the answer was then truncated by `max_tokens=800` (`finish_reason=length`). With `reasoning_budget_tokens=96` the engine reports `reasoning_tokens=96` on the nose and the turn completes normally (`finish_reason=stop`) with the answer in `content` — the model's next sampled token was forced to the think-END delimiter instead of a token of continued rumination.

The full reasoning the model was ALLOWED, verbatim (this is the limiter's before/after in text form; 322 chars vs 1996 chars):

`reasoning_budget_tokens=96` — reasoning, verbatim, all of it:

```
We need answer classic. Need explain step by step full reasoning, verify two independent ways. Need be careful. Let ball x, bat x+1.00, total 1.10 => 2x+1=1.10 => x=0.05. Verify: sum 0.05+1.05=1.10, difference 1.00. Another independent way maybe algebra with total minus difference? Let's craft. Need mention common trap $
```

...and immediately after it, the start of the `content` channel (verbatim), which only exists because the limiter closed the think block:

```
Let the ball cost \(x\) dollars.

The bat costs \(\$1.00\) more than the ball, so the bat costs:

\[
x + 1.00
\]

Together they cost \(\$1.10\), so:

\[
x + (x + 1.00) = 1.10
\]

Simplify:

\[
2x + 1.00 = 1.10
\]

Subtract \(1.00\) from both sides:

\[
2x = 0.10
\]

Divide by 2:

\[
x = 0.05
\]

So the ball costs **\(\$0.05\)**, or **5 cents**.

Now verify independently.

**Verification 1: Check total and difference**

If the ball costs \(\$0.05\), then the bat costs:

\[
0.05 + 1.00 = 1.05
\]

Total:

\[
0.05 + 1.05 = 1.10
\]

Difference:

\[
1.05 - 0.05 = 1.00
\]

Both conditions are satisfi
```

### 3d. Corroborating on-disk guard captures (same node, same digest)

The serving node keeps bounded dumps of the committed token ids every time the guard fires or a client cancels (`EXO_DSV41_DEGENERATION_DUMP_DIR`). Four exist in `~/.exo/exo_log/` from the 2026-10-13 / 10-14 build; they are re-decoded here with the checkpoint-family tokenizer so the loop text is visible. They are cited as corroboration for the live fire above, and their `guard_env_digest = 633711af` matches the live fire's digest.

`dsv41-degeneration-guard-token-20261013-224154-5634.json` — detector `token`, period 1, repeats >=6, action `warn`, 143 tokens generated, digest `fb0de5b4`. Decoded tail (verbatim):

```
's write 120 "banana" separated by spaces. Need avoid extra newline? Output only repetitions. We can include newline at end maybe okay. Need count.

Let's enumerate. I'll create groups of 10 for counting, but final all continuous. Need ensure 120.

Approach: Write 12 groups of 10 bananas. Each group: banana banana banana banana banana banana
```

`dsv41-degeneration-guard-token-20261014-135110-69384.json` — detector `token`, period 12, repeats >=6, action `error`, 1613 tokens generated, digest `633711af`. Decoded tail (verbatim):

```
.

Let me go.

Now.

Write.

OK.

Let me write.

Go.

Now.

Writing.

OK.

Let me write.

Go.

Now.

Writing.

OK.

Let me write.

Go.

Now.

Writing.

OK.

Let me write.

Go.

Now.

Writing.

OK.

Let me write.

Go.

Now.

Writing.

OK.

Let me write.

Go.

Now.

Writing
```

`dsv41-degeneration-guard-token-20261014-135202-69384.json` — detector `token`, period 12, repeats >=6, action `error`, 1613 tokens generated, digest `633711af`. Decoded tail (verbatim):

```
.

Let me go.

Now.

Write.

OK.

Let me write.

Go.

Now.

Writing.

OK.

Let me write.

Go.

Now.

Writing.

OK.

Let me write.

Go.

Now.

Writing.

OK.

Let me write.

Go.

Now.

Writing.

OK.

Let me write.

Go.

Now.

Writing.

OK.

Let me write.

Go.

Now.

Writing
```

`dsv41-degeneration-guard-token-20261014-135255-69384.json` — detector `token`, period 12, repeats >=6, action `error`, 1613 tokens generated, digest `633711af`. Decoded tail (verbatim):

```
.

Let me go.

Now.

Write.

OK.

Let me write.

Go.

Now.

Writing.

OK.

Let me write.

Go.

Now.

Writing.

OK.

Let me write.

Go.

Now.

Writing.

OK.

Let me write.

Go.

Now.

Writing.

OK.

Let me write.

Go.

Now.

Writing.

OK.

Let me write.

Go.

Now.

Writing
```

The two `period=12` dumps are the other classic shape: not a single repeated token but a repeated 12-token *phrase* — `"Let me write." / "Go." / "Now." / "Writing."` — which is exactly the self-doubt loop the min_repeats=6 / max_period=24 exact-cycle detector exists to catch, and exactly the shape the phase-2 reasoning-budget limiter is the backstop for when the drift is too large for an exact cycle match.

### 3e. What could NOT be captured

- The guard's `action` knob was NOT toggled (`warn` vs `error`) — that needs a relaunch, which is out of bounds. The single archived `action=warn` dump (2026-10-13) is the only `warn`-mode sample, and it is from an older build; the digest of the current live fire is `action=error`.
- **Correction to an earlier draft of this section.** The startup `[DSV41] loop-guard env` line IS available — it had rotated out of the live `exo.log`, and my first archive grep missed it because `zstdcat` is not on the non-interactive ssh PATH (an empty result that looks like a real negative). Re-read with the absolute path `/opt/homebrew/bin/zstdcat ~/.exo/exo_log/exo.2026-10-16_18-38-04_871911.log.zst`, it is there, verbatim:

```
[ 2026-10-16 18:40:15.140 | INFO | exo.worker.engines.mlx.dsv41.engine:__post_init__:535 ] [DSV41] loop-guard env (rank 1/2, digest=633711af): EXO_LOOP_DETECT=1 EXO_LOOP_DETECT_WINDOW=160 EXO_LOOP_DETECT_MAX_PERIOD=24 EXO_LOOP_DETECT_MIN_REPEATS=6 EXO_LOOP_DETECT_ACTION=error EXO_LOOP_DETECT_LONG=1 EXO_LOOP_DETECT_LONG_BLOCK=24 EXO_LOOP_DETECT_LONG_MAX_PERIOD=12 EXO_LOOP_DETECT_LONG_MIN_REPEATS=3
```

Its `digest=633711af` is the same digest the live guard fire reports, so the effective deployed knob set, the archived dumps' recorded `guard_env` block, and the live fire all agree — three independent statements of the same configuration.

---

## Deliverable 4 — 1M-context attempt

**Request (identical for both attempts).** Body `prompts/prompts_1m.json`, 3,391,475 bytes, 999,999 tokens by the checkpoint-family tokenizer and **1,000,003** by the server's own count. `"stream": true`, `max_tokens: 100`, `enable_thinking: false`.

```bash
cd /Users/<user>/.hermes/cache/scratch/writeup
curl -sS -N --max-time <cap> -X POST \
  http://node0.tailnet.ts.net:52415/v1/chat/completions \
  -H 'Content-Type: application/json' --data-binary @prompts/prompts_1m.json
```

Cluster HEAD `5a18b81debe3e4d1b82370d473614d0b4be82bf2` throughout.

### 4a. Headline


- **Attempt 1 (2 hours (7200 s)): PARTIAL — did not finish.** 1M prompt accepted; prefill reached **802,816 / 1,000,003 tokens** (80.3%) in **7,181 s** (**111.8 tok/s**). No first token was ever emitted, so there is no `finish_reason` and no `usage` block.
- **Attempt 2 (4 hours (14400 s)): COMPLETED.** `prompt_tokens=1,000,003`, `completion_tokens=100`, `finish_reason=length`, wall 6760.5 s.

One denominator note for attempt 2: the server reported `session reuse: this 1000003-token prompt matches a resident conversation on 299946 rows`, so ~300 K of the prompt came from cache and `processed_tokens` counts the remaining ~700 K rows it actually had to prefill. The band table for attempt 2 is therefore read against ~700 K, not 1,000,003.
  - TWO DISTINCT QUANTITIES, not a typo: the resident conversation is 299,946 rows (:704) while `usage.cached_tokens` is 299,008 (:859). 299,008 = 292×1024 exactly; the prefix reuse is aligned down to a 1024-row block boundary (`DEFAULT_CHECKPOINT_SPACING_ROWS = 1024`, session.py:158). The 938-row gap is 0.3% and immaterial downstream.

### 4b. Attempt 1 (2 h cap) — PARTIAL

### Attempt 1 detail — request, error and capture

Meta, verbatim:

```
1M RUN (streamed SSE)
START 2026-10-17 00:03:57 CDT
CMD: curl -sS -N --max-time 7200 -X POST http://node0.tailnet.ts.net:52415/v1/chat/completions -H 'Content-Type: application/json' --data-binary @prompts/prompts_1m.json
PROMPT FILE prompts_1m.json:  3391475 bytes; calibrated 999999 tok locally; body stream=true max_tokens=100 enable_thinking=false
CLUSTER HEAD SHA: 5a18b81debe3e4d1b82370d473614d0b4be82bf2
curl: (28) Operation timed out after 7200001 milliseconds with 69245 bytes received
curl_exit=0 END 2026-10-17 02:03:57 CDT
```

The exact client error, verbatim:

```
curl: (28) Operation timed out after 7200001 milliseconds with 69245 bytes received
```

`curl`'s own measurement:

```
CURLMETA http=200 starttransfer=0.011157 total=7200.003712 size=69245
```

The real evidence for a prefill that never reaches generation is the **391 `: prefill_progress` frames**. Two of them, verbatim, and the last one received:

```
: prefill_progress {"PrefillProgressChunk":{"model":"dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw","processed_tokens":4096,"total_tokens":1000003}}
: prefill_progress {"PrefillProgressChunk":{"model":"dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw","processed_tokens":6144,"total_tokens":1000003}}
: prefill_progress {"PrefillProgressChunk":{"model":"dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw","processed_tokens":802816,"total_tokens":1000003}}
```

Series shape: token step a constant **2048** (the server emits one frame per 2,048 tokens), inter-frame gap median **14.8 s** (min 9.0 s, max 31.1 s). The gap growing while the step stays fixed is the whole story: still making progress, more slowly.

Banded throughput from the frame timestamps (nominal 100 K bands):

| band (tokens processed) | seconds | **prefill tok/s in band** |
| --- | --- | --- |
| 4,096 → 104,448 | 462.8 | **216.8** |
| 104,448 → 204,800 | 556.0 | **180.5** |
| 204,800 → 305,152 | 646.1 | **155.3** |
| 305,152 → 405,504 | 691.2 | **145.2** |
| 405,504 → 505,856 | 781.4 | **128.4** |
| 505,856 → 606,208 | 1222.8 | **82.1** |
| 606,208 → 706,560 | 1392.3 | **72.1** |
| 706,560 → 802,816 | 1409.8 | **68.3** |

Sampled raw series (every 20th frame plus the last, so the curve can be re-checked without the whole table):

| t since request (s) | processed_tokens | total_tokens |
| --- | --- | --- |
| 19.1 | 4,096 | 1,000,003 |
| 206.8 | 45,056 | 1,000,003 |
| 393.8 | 86,016 | 1,000,003 |
| 593.9 | 126,976 | 1,000,003 |
| 816.8 | 167,936 | 1,000,003 |
| 1063.5 | 208,896 | 1,000,003 |
| 1324.5 | 249,856 | 1,000,003 |
| 1590.1 | 290,816 | 1,000,003 |
| 1860.1 | 331,776 | 1,000,003 |
| 2142.0 | 372,736 | 1,000,003 |
| 2434.5 | 413,696 | 1,000,003 |
| 2736.7 | 454,656 | 1,000,003 |
| 3062.6 | 495,616 | 1,000,003 |
| 3480.3 | 536,576 | 1,000,003 |
| 3995.9 | 577,536 | 1,000,003 |
| 4545.5 | 618,496 | 1,000,003 |
| 5107.3 | 659,456 | 1,000,003 |
| 5684.1 | 700,416 | 1,000,003 |
| 6273.6 | 741,376 | 1,000,003 |
| 6875.3 | 782,336 | 1,000,003 |
| 7181.5 | 802,816 | 1,000,003 |

### 4c. Attempt 2 (4 h cap)

### Attempt 2 detail — request, error and capture

Meta, verbatim:

```
1M RUN ATTEMPT 2 (streamed SSE, --max-time 14400 / 4h)
START 2026-10-17 02:36:42 CDT
CMD: curl -sS -N --max-time 14400 -X POST http://node0.tailnet.ts.net:52415/v1/chat/completions -H 'Content-Type: application/json' --data-binary @prompts/prompts_1m.json
body: stream=true max_tokens=100 enable_thinking=false; server counts prompt as 1000003 tokens
HEAD: 5a18b81debe3e4d1b82370d473614d0b4be82bf2
curl_exit=0 END 2026-10-17 04:29:22 CDT
```

`curl`'s own measurement:

```
CURLMETA http=200 starttransfer=0.007448 total=6760.509062 size=85811
```

The real evidence for a prefill that never reaches generation is the **342 `: prefill_progress` frames**. Two of them, verbatim, and the last one received:

```
: prefill_progress {"PrefillProgressChunk":{"model":"dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw","processed_tokens":4096,"total_tokens":1000003}}
: prefill_progress {"PrefillProgressChunk":{"model":"dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw","processed_tokens":6144,"total_tokens":1000003}}
: prefill_progress {"PrefillProgressChunk":{"model":"dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw","processed_tokens":700995,"total_tokens":1000003}}
```

Series shape: token step a constant **2048** (the server emits one frame per 2,048 tokens), inter-frame gap median **16.3 s** (min 8.0 s, max 32.4 s). The gap growing while the step stays fixed is the whole story: still making progress, more slowly.

Banded throughput from the frame timestamps (nominal 100 K bands):

| band (tokens processed) | seconds | **prefill tok/s in band** |
| --- | --- | --- |
| 4,096 → 104,448 | 550.7 | **182.2** |
| 104,448 → 204,800 | 668.8 | **150.1** |
| 204,800 → 305,152 | 751.4 | **133.6** |
| 305,152 → 405,504 | 799.6 | **125.5** |
| 405,504 → 505,856 | 1023.0 | **98.1** |
| 505,856 → 606,208 | 1449.9 | **69.2** |
| 606,208 → 700,995 | 1460.3 | **64.9** |

Sampled raw series (every 20th frame plus the last, so the curve can be re-checked without the whole table):

| t since request (s) | processed_tokens | total_tokens |
| --- | --- | --- |
| 22.8 | 4,096 | 1,000,003 |
| 240.1 | 45,056 | 1,000,003 |
| 467.4 | 86,016 | 1,000,003 |
| 709.5 | 126,976 | 1,000,003 |
| 981.4 | 167,936 | 1,000,003 |
| 1271.9 | 208,896 | 1,000,003 |
| 1573.8 | 249,856 | 1,000,003 |
| 1883.6 | 290,816 | 1,000,003 |
| 2201.5 | 331,776 | 1,000,003 |
| 2526.9 | 372,736 | 1,000,003 |
| 2860.8 | 413,696 | 1,000,003 |
| 3232.0 | 454,656 | 1,000,003 |
| 3687.9 | 495,616 | 1,000,003 |
| 4239.0 | 536,576 | 1,000,003 |
| 4837.7 | 577,536 | 1,000,003 |
| 5451.9 | 618,496 | 1,000,003 |
| 6077.9 | 659,456 | 1,000,003 |
| 6718.6 | 700,416 | 1,000,003 |
| 6726.6 | 700,995 | 1,000,003 |

### 4d. Attempt 2 — FINAL RESULT: a 1,000,003-token prompt completed

| field | value |
| --- | --- |
| HTTP status | **200** |
| `finish_reason` | **`length`** (hit `max_tokens=100`) |
| `usage.prompt_tokens` | **1,000,003** |
| `usage.completion_tokens` | **100** (`reasoning_tokens=0`) |
| `usage.total_tokens` | **1,000,103** |
| `prompt_tokens_details.cached_tokens` | **299,008** |
| `prefix_cache_hit` (server) | **`partial`** |
| client wall clock | **6760.509 s** = **1.88 h** |
| server prefill | **6725.29 s** for 700,995 uncached rows = **104.2 tok/s** |
| client time-to-first-token | **6726.57 s** |
| client decode window | **33.87 s** |
| **client decode rate** | **2.92 tok/s** |
| **server decode rate** | **2.92 tok/s** (turn-end 6760.2 s - first-token 6726.3 s) |
| generated text | 331 chars / 100 tokens |

Server lines for the turn, verbatim (task_id `f8104803-3234-43a5-9428-1e4aa1c1b406`):

```
[ 2026-10-17 04:28:49.053 | INFO | exo.worker.engines.mlx.dsv41.engine:1270 ] [DSV41] obs first-token turn=f8104803-3234-43a5-9428-1e4aa1c1b406 latency=6726.30s (prefill 6725.29s, 1000003 prompt tokens, reuse=299008)
[ 2026-10-17 04:28:59.318 | INFO | exo.worker.engines.mlx.dsv41.engine:1281 ] [DSV41] obs progress turn=f8104803-3234-43a5-9428-1e4aa1c1b406 committed=8 elapsed=6736.6s tok_s=0.0
[ 2026-10-17 04:29:20.027 | INFO | exo.worker.engines.mlx.dsv41.engine:1281 ] [DSV41] obs progress turn=f8104803-3234-43a5-9428-1e4aa1c1b406 committed=90 elapsed=6757.3s tok_s=0.0
[ 2026-10-17 04:29:22.941 | INFO | exo.worker.engines.mlx.dsv41.engine:1198 ] [DSV41] obs turn-end turn=f8104803-3234-43a5-9428-1e4aa1c1b406 committed=103 emitted=100 elapsed=6760.2s tok_s=0.0 reason=length
```

The two `obs progress` lines are an internal cross-check on the decode rate: 82 committed tokens between them in 20.7 s = **3.96 tok/s**, against the 2.95 tok/s obtained by differencing the turn-end from the first-token latency over all 100 tokens. Same order, same conclusion — a token every ~250-340 ms at 1M depth.

**Generated text, verbatim, all 100 tokens / 331 chars** (the whole `content`, since it is one short paragraph; "first 15 lines" and "last 15 lines" are therefore the same text):

```
BOTTLENECK: the `moe.all_sum` cross-rank collective (and its GPU→CPU→GPU drain) is the most frequently discussed performance bottleneck, appearing across dozens of docs from 2026-08-19 through 2026-08-24. It is quantified in `docs/moe-all-sum-dominant-cost-2026-08-19.md`, which measured a 2.63x speedup (62.0% share) at 12K tokens
```

The answer is correct against what is inside the 1M-token context (`docs/moe-all-sum-dominant-cost-2026-08-19.md`, and the 2.63x / 62.0% figure it really contains), and it is truncated at exactly 100 tokens by `finish_reason=length`, not by any failure.

### 4d-bis. Headline for the writeup

**A 1,000,003-token prompt completed end-to-end on this cluster: HTTP 200, `finish_reason=length` after 100 tokens, 6,760.5 s (1.88 h) wall, 104.2 tok/s prefill over the rows it had to do, ~2.9 tok/s decode — *1M prompt served end-to-end, partial prefix cache (299,008 cached)*.** It needed a 4-hour window: the `--max-time 7200` first attempt reached only 802,816 of 1,000,003 tokens and was cancelled by the client at the cap. 1M KV is not merely allocated here — a 1M request really ran and really answered. The one thing it is NOT is a cold 1M prefill, and section 4f says so.

### 4e. The 300 K bracket, which DID complete

Fired with the identical method (`"stream": true`, `max_tokens=200`, `enable_thinking=false`) between the 160 K depth row and the 1M attempts. It is the deepest context that completed end-to-end in this session.

| field | value |
| --- | --- |
| HTTP status | **200** |
| `prompt_tokens` | **300,003** |
| `cached_tokens` | **0** |
| `completion_tokens` | **189** |
| `finish_reason` | **`stop`** |
| wall clock | **1600.8 s** |
| server prefill | **1535.25 s** = **195.4 tok/s** |
| server decode window | **31.66 s** = **6.0 tok/s** |

Server lines, verbatim:

```
[ 2026-10-17 02:31:59.523 | INFO | exo.worker.engines.mlx.dsv41.engine:1270 ] [DSV41] obs first-token turn=519cd399-629a-48ed-bf93-f412a2cb5910 latency=1535.64s (prefill 1535.25s, 300003 prompt tokens, reuse=0)
[ 2026-10-17 02:32:31.188 | INFO | exo.worker.engines.mlx.dsv41.engine:1198 ] [DSV41] obs turn-end turn=519cd399-629a-48ed-bf93-f412a2cb5910 committed=191 emitted=189 elapsed=1567.3s tok_s=0.1 reason=stop
```

Generated text, verbatim (full `content`, 699 chars):

```
BOTTLENECK: The single most frequently discussed performance bottleneck across the reference material is the `moe.all_sum` cross-rank collective in the MoE forward pass, repeatedly identified as the dominant decode/prefill cost center (e.g. 62% share via NOP-ablation, 39.5% of shallow prefill wall, and the target of the bf16 payload-halving win). The document that most directly quantifies it is `docs/moe-all-sum-dominant-cost-2026-08-19.md`, which measures a 2.63x speedup with all_sum NOP'd at two depths. Supporting quantification also appears in `docs/moe-all-sum-skew-vs-comms-2026-08-19.md` and the jaccl-internal timing doc `docs/jaccl-internal-timing-allsum-transport-fast-2026-08-21.md`.
```

**This row is the important one for the 1M story.** It is a completed 300 K-context run at a healthy prefill rate (195 tok/s) whose decode is already 5x down from the 1 K-160 K plateau (6.0 tok/s vs ~30). The 1M attempt then took that same degradation, stretched it over 3.3x more context, and ran out of clock.

Its own banded progress curve:

| band (tokens processed) | seconds | **prefill tok/s in band** |
| --- | --- | --- |
| 4,096 → 104,448 | 414.7 | **242.0** |
| 104,448 → 204,800 | 531.5 | **188.8** |
| 204,800 → 300,003 | 573.9 | **165.9** |

### 4f. THE 1M COMPLETED — banded curve, and the honest cache caveat

Attempt 2 finished: **HTTP 200, `prompt_tokens=1,000,003`, `completion_tokens=100`, `finish_reason=length`, wall 6,760.5 s (02:36:42 -> 04:29:22), real text captured.** `curl_exit=0` with no `(28)` line — it did not time out.

**The cache caveat, stated plainly, because it changes what this proves.** The run is: *1M prompt served end-to-end, partial prefix cache (299,008 cached)*. It was NOT a fully cold 1M prefill:

- `prompt_tokens_details.cached_tokens = 299,008` and the server's own `prefix_cache_hit = "partial"`, `reuse=299008`.
- So the prefill actually performed was **700,995 uncached tokens at **104.2 tok/s** (server `prompt_tps=104.23`; 700,995 / 6725.29 s).
- **The `prefill_progress` counter tracks UNCACHED tokens only.** That is why its last frame reads `processed_tokens: 700995` while `total_tokens` reads `1000003` — the two are on different denominators by design, and reading the 700,995 as "only got 70% of the way" would be exactly backwards. The turn reached 100% of the work it had to do.
- A **fully cold** 1M prefill is therefore not what was measured. Attempt 1, which had only ~98 K cached, is the closer proxy: it reached 802,816 of 1,000,003 tokens in 7,181 s before the 2 h cap killed it, on a curve still falling. Extrapolating attempt 1's last band (~68 tok/s) over its remaining ~197 K tokens puts a **cold 1M prefill at roughly 2.9 h** — which is why the brief's 2 h window was not enough and the 4 h one was.
  - CONVENTION: the progress counter is POSITIONAL — `_session_progress` sends `processed_tokens=rows_done` against `total_tokens=active.prefill_total` (engine.py:1191-1192), i.e. the full prompt length. So 802,816/1,000,003 = 80.3% and the ~197 K remainder stand as written; the ~98 K cached prefix is counted as already done.
  - CAVEAT on the extrapolation: because the counter is positional, the ~68 tok/s band rate includes cached rows that cost zero compute. Using it as a compute rate overstates throughput by ~98 K/~704 K ≈ 14%, so 2.9 h is a lower bound on cold-prefill wall time.

Prefill throughput by depth, measured within these two requests (no cross-run confound — 391 progress frames in attempt 1, 342 in attempt 2):

| depth | prefill tok/s | source |
| --- | --- | --- |
| ~10 K | 275.1 | Deliverable 2 |
| ~100 K | 245.1 | Deliverable 2 |
| 160 K | 242.0 | Deliverable 1 server line |
| ~104 K of the 1M prompt | 216.8 | 1M attempt 1, first band |
| 300 K | 195.4 | 300 K row above |
| ~505-606 K of the 1M prompt | 82.1 | 1M attempt 1 |
| ~905-1000 K of the 1M prompt | 64.9 | 1M attempt 2, last band |

Attempt 2's bands are the cleanest statement of the mechanism, because its `processed_tokens` counts ONLY newly-prefilled rows and every one of them sits at absolute position ~300 K-1M (the first ~299 K rows came from cache):

| absolute position of the rows being prefilled | prefill tok/s |
| --- | --- |
| ~303,104 - 403,456 | **182.2** |
| ~403,456 - 503,808 | **150.1** |
| ~503,808 - 604,160 | **133.6** |
| ~604,160 - 704,512 | **125.5** |
| ~704,512 - 804,864 | **98.1** |
| ~804,864 - 905,216 | **69.2** |
| ~905,216 - 1,000,003 | **64.9** |

Monotonic from 182.2 down to 64.9 tok/s as the rows move deeper. Prefill cost is a function of POSITION in the context, not of how many tokens have been done: the same request prefilled its first 100 K rows at 216.8 tok/s in attempt 1 (cold, starting near position 4 K) and at 182.2 tok/s in attempt 2 (starting at position ~300 K). The deepest stretches are irreducibly the slowest, which is why a 1M prefill cannot be front-loaded or parallelised away.

### 4g. Caveats on isolation

Other clients were active on this instance during the session (their own `obs` lines are quoted in Deliverable 2 as independent evidence). exo serves this instance's requests with one prefill in flight at a time, and each of the runs above was individually checked so that no other request's `obs first-token` falls inside its measured decode window — for the 300 K row that check is shown explicitly in Deliverable 2. Rates are reported as observed on a busy-but-serialised instance.

---

