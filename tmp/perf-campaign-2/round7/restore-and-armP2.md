# Round 7 — Restore to Production Config + Paired TTFT Arm (P2, =200)

Date: 2026-09-04

## Step 1 — Teardown + Relaunch

Both nodes confirmed ZERO exo processes before relaunch:
```
---node201---
DONE201
---node202---
DONE202
```
(no `exo -v` process lines printed between the marker and DONE — confirms 0 processes)

Relaunch command (laptop, production defaults, NO steel-BI override, NO rendezvous override):
```
cd /Users/adam.durham/repos/exo
tmux new-session -d -s r7restore 'cd ~/repos/exo && DSV4_KV_CACHE_BITS=0 ./start_cluster.sh 2>&1 | tee /tmp/r7restore_launch.log'
```
Prompted with "local HEAD not on origin/main" (informational only, answered `y`, NO push performed).

Launch log tail confirming healthy stabilization:
```
Waiting for cluster to stabilize...... HEALTHY! (Nodes: 2, Identities: 2)
  Mac Studio M4-1: febe407fcafb2a1230302f29e68fe254
  Mac Studio M4-2: 291269ca06fa161313ebabad1eef771
Auto-placing DeepSeek V4 Flash (deepseek-ai/DeepSeek-V4-Flash-0731) across both Studios via RDMA...
Waiting for 2 DeepSeek V4 runner(s) to become Ready......................... READY (2/2)
```
Total launch time from `tmux new-session` to `READY (2/2)`: ~5-6 minutes.

## Step 2 — Verify Restored Config (mandatory gate)

Runner PIDs (exact command `.venv/bin/python -m exo -v`):
- macstudio-m4-1 (192.168.86.201): PID **9402**
- macstudio-m4-2 (192.168.86.202): PID **20117**

`ps eww <pid> | tr ' ' '\n' | grep -E '...'` output, verbatim:

**Node 201 (pid 9402):**
```
EXO_DSV4_BATCHED_PREFILL=1
EXO_BATCHED_PREFILL_RENDEZVOUS_MS=200
EXO_SPECULATIVE_GAMMA=3
EXO_DSV4_MTP=1
EXO_DSV4_VERIFY_ROWSEQ_VEC=1
EXO_DSV4_VERIFY_ROWSEQ_VEC_ROWSDPA=3
MLX_STEEL_BATCH_INVARIANT=1
EXO_DSV4_VERIFY_BATCH=1
EXO_DSV4_VERIFY_BATCH_MIN_CTX=8192
```

**Node 202 (pid 20117):**
```
EXO_DSV4_BATCHED_PREFILL=1
EXO_BATCHED_PREFILL_RENDEZVOUS_MS=200
EXO_SPECULATIVE_GAMMA=3
EXO_DSV4_MTP=1
EXO_DSV4_VERIFY_ROWSEQ_VEC=1
EXO_DSV4_VERIFY_ROWSEQ_VEC_ROWSDPA=3
MLX_STEEL_BATCH_INVARIANT=1
EXO_DSV4_VERIFY_BATCH=1
EXO_DSV4_VERIFY_BATCH_MIN_CTX=8192
```

**GATE RESULT: PASS.** All four required values match on BOTH nodes:
`MLX_STEEL_BATCH_INVARIANT=1`, `EXO_BATCHED_PREFILL_RENDEZVOUS_MS=200`, `EXO_SPECULATIVE_GAMMA=3`, `EXO_DSV4_MTP=1`.

## Step 3 — Paired TTFT Arm (P2, =200, exact ordering as specified)

### Set 1 — Five 2K reps (P2_ttft_r1..r5), run FIRST

| tag | prefill_s | prefill_ms | prompt_tokens | prefix_cache_hit |
|---|---|---|---|---|
| P2_ttft_r1 | 7.80 | 7800.0 | 2308 | none |
| P2_ttft_r2 | 8.29 | 8290.0 | 2354 | none |
| P2_ttft_r3 | 7.41 | 7410.0 | 2308 | none |
| P2_ttft_r4 | 8.44 | 8440.0 | 2239 | none |
| P2_ttft_r5 | 8.01 | 8010.0 | 2285 | none |

**Median: 8010.0 ms. Range: [7410.0, 8440.0] ms.**

### Set 2 — Ten short reps (P2_short_r1..r10), run SECOND (paired instrument)

| tag | prefill_s | prefill_ms | prompt_tokens | prefix_cache_hit |
|---|---|---|---|---|
| P2_short_r1 | 2.14 | 2140.0 | 220 | none |
| P2_short_r2 | 1.86 | 1860.0 | 230 | none |
| P2_short_r3 | 1.73 | 1730.0 | 226 | none |
| P2_short_r4 | 1.94 | 1940.0 | 224 | none |
| P2_short_r5 | 1.78 | 1780.0 | 226 | none |
| P2_short_r6 | 2.06 | 2060.0 | 220 | none |
| P2_short_r7 | 1.93 | 1930.0 | 230 | none |
| P2_short_r8 | 2.04 | 2040.0 | 234 | none |
| P2_short_r9 | 2.15 | 2150.0 | 228 | none |
| P2_short_r10 | 3.98 | 3980.0 | 230 | none |

**Median: 1990.0 ms. Range: [1730.0, 3980.0] ms.**

Note: `decode_sample_trustworthy: false` on all reps — EXPECTED per instructions (TTFT-only instrument, not "fixed").

Raw JSON files: `tmp/perf-campaign-2/round7/results/P2_ttft_r{1..5}.json`, `P2_short_r{1..10}.json`.

## Step 4 — Quality + Health

### 1. `bench/ab_probe_tier1.py`

**Could not run as specified.** The script's default `--model` value is hardcoded to `mlx-community/DeepSeek-V4-Flash`, which is NOT the model currently loaded/placed on this cluster (`deepseek-ai/DeepSeek-V4-Flash-0731`). Running it with no args returns HTTP 503 from `/v1/chat/completions`:
```
httpx.HTTPStatusError: Server error '503 ' for url 'http://192.168.86.201:52415/v1/chat/completions'
```
Passing the correct model id explicitly (`--model deepseek-ai/DeepSeek-V4-Flash-0731`) does reach the server but the script itself does not implement/report a pass-count or "7/7" summary — it is a single-request probe script, not a multi-case suite (confirmed by reading the source: `argparse` takes one `target_tokens`/`--max-tokens`/`--tag`/`--model`, single POST, single JSON output, no assertions, no pass/fail loop). **No "7/7" test suite exists in this script as currently written.** I could not find any other file matching a 7-case tier-1 suite (`grep`/`find` across `bench/` for "7/7", "tier1", pass-count patterns turned up nothing beyond this single-probe file). Flagging this as a real gap: either the expected tier-1 harness has moved/been renamed, or the task's "expect 7/7" description doesn't match what's actually in the repo. **Did not fabricate a pass count.**

### 2. Coherent completion (verbatim)

Request: `POST http://192.168.86.201:52415/v1/chat/completions`, model `deepseek-ai/DeepSeek-V4-Flash-0731`, `max_tokens=150`, `temperature=0`.

HTTP status: **200**

`reasoning_content`:
```
1.  The user asks for the capital of France in one sentence.
2.  The capital of France is Paris.
3.  Construct a single sentence stating this fact.
```

`content`:
```
The capital of France is Paris.
```

Coherent, mentions Paris. Both fields populated (`content` non-empty this time).

### 3. `/v1/models` health

```
MODELS_HTTP_CODE=200
```
Confirmed 200 both before and after the completion test.

## Summary

1. Restored env confirmed identical and correct on BOTH nodes (PIDs 9402 / 20117): `MLX_STEEL_BATCH_INVARIANT=1`, `EXO_BATCHED_PREFILL_RENDEZVOUS_MS=200`, `EXO_SPECULATIVE_GAMMA=3`, `EXO_DSV4_MTP=1`.
2. TTFT (P2, =200 production arm):
   - 2K-prompt reps (n=5): median 8010.0 ms, range [7410.0, 8440.0] ms.
   - Short 20-tok-target reps (n=10, the paired instrument): median 1990.0 ms, range [1730.0, 3980.0] ms.
3. Tier-1 quality suite: **could not produce a pass count** — `bench/ab_probe_tier1.py` as it exists in the repo is a single-request probe with no assertions/pass-fail loop and a stale default model id (503 out of the box). Flagging for triage rather than fabricating "7/7."
4. Coherent completion: **succeeded**, HTTP 200, content = "The capital of France is Paris." (reasoning_content also present and coherent). `/v1/models` confirmed HTTP 200.

No deltas computed against any other arm; no ship/hold judgment made (PM's call).
