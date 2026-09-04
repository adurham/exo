# Task A — Arm P (baseline, RENDEZVOUS_MS=200) TTFT measurement

Measured on the currently-running cluster, no relaunch/restart/kill/edit performed.

## Step 1 — Config confirmation (real runner PIDs, both nodes)

### 192.168.86.201
```
PID=50175
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

### 192.168.86.202
```
PID=60177
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

**Confirmed:** `EXO_BATCHED_PREFILL_RENDEZVOUS_MS=200` and `EXO_DSV4_BATCHED_PREFILL=1` present and matching expected values on both nodes. Proceeded to measurement.

## Step 2 — API health

`curl -s -o /tmp/models.json -w "HTTP_STATUS:%{http_code}\n" http://192.168.86.201:52415/v1/models`

Result: `HTTP_STATUS:200`

## Step 3/4 — 5 reps, cache-cold ~2K prompt, max-tokens=200 (TTFT only; decode window intentionally not trustworthy)

| rep | prefill_s | prefill_ms | prompt_tokens | finish_reason | prefix_cache_hit |
|-----|-----------|------------|----------------|----------------|-------------------|
| 1   | 7.80      | 7800.0     | 2285           | length         | none              |
| 2   | 7.66      | 7660.0     | 2216           | length         | none              |
| 3   | 8.03      | 8030.0     | 2331           | length         | none              |
| 4   | 7.94      | 7940.0     | 2285           | length         | none              |
| 5   | 7.90      | 7900.0     | 2285           | length         | none              |

- **Median prefill_s (arm P):** 7900.0 ms
- **Min–max range:** 7660.0 ms – 8030.0 ms

## Step 5 — Sanity checks

- `prompt_tokens`: all reps in range 2216–2331, within expected ~2,000–2,600 window. ✅
- `prefix_cache_hit`: `none` on all 5 reps — no contamination detected. ✅
- `server_stats` was non-null on all 5 reps (no missing-data cases to flag).
- All 5 reps `finish_reason: length` (expected, given `--max-tokens 200` cap; not used for decode-tps conclusions per probe design note).

No relaunch, restart, process kill, file edit, or env var change was performed at any point.
