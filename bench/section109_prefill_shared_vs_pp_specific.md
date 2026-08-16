# Section 109: Is the dominant prefill cost SHARED code or PP-specific code?

**Verdict: the dominant prefill cost is SHARED — because the per-chunk
`forward` span (98% of tracked prefill time, per Section 99 Track C) wraps a
`model(chunk_tokens, cache=...)` call plus `maybe_quantize_kv_cache(...)`,
and BOTH of those calls execute, unconditionally, on the TP path
(`prefill_batched`) as well as the PP path
(`_pipeline_parallel_prefill_steps`). Nothing PP-specific sits between the
measured cost and TP's own chunk loop. This DIRECTLY caps TP's prefill
ceiling — it is not a PP tax that disappears when PP ships dead.**

This document is CODE READING ONLY. No cluster runs were performed (cluster
is live and was not touched). Calibration: per the skill
`exo-dsv4-prefill-tuning` and doc Section 55, the 202-225 tok/s figure is the
HONEST post-accounting-fix number (TTFT, the timing denominator, never
moved) — there is no "regression" to chase, only "is this shared or
PP-only."

## Method

Read `src/exo/worker/engines/mlx/generator/generate.py`:
- `_pipeline_parallel_prefill_steps()` (~line 421) — PP's real chunked-prefill
  driver, wrapped eagerly by `pipeline_parallel_prefill()` (~line 671).
- `prefill()` (~line 709) — the router: picks
  `pipeline_parallel_prefill()` when `is_pipeline and num_tokens >=
  prefill_step_size`, else `stream_generate()` (mlx-lm).
- `prefill_batched()` (~line 1240) — the TP-aware batched path (this is what
  actually runs multi-stream prefill under TP; falls back to serial
  `prefill()`→`stream_generate()` for single-stream or SSM-cache cases).
- `src/exo/worker/engines/mlx/cache.py` — `CacheSnapshot`,
  `snapshot_ssm_states()`, `_LEAF_SNAPSHOT_RETENTION` /
  `EXO_LEAF_SNAPSHOT_RETENTION`, `is_non_trimmable_cache_entry()`.
- `mlx-lm/mlx_lm/generate.py` — `maybe_quantize_kv_cache()` (line 293),
  `stream_generate()` (line 787), `_merge_caches`.

## Candidate ranking

### #1 (top candidate): the per-chunk `forward` span — `model(...)` call + KV quantize, CONFLATED by the timer

**CONFIRMED, and CONFIRMED shared.**

- PP: `generate.py` line ~568-571, inside `_pipeline_parallel_prefill_steps`:
  ```
  model(chunk_tokens, cache=_prompt_cache)     # (non-interruptible branch)
  quantize_cache_fn(_prompt_cache)             # = maybe_quantize_kv_cache(...)
  request_trace.record(f"prefill.chunk{i}.forward({chunk_size}tok)", _t_fwd)
  ```
  The `_t_fwd` timer starts *before* the `model(...)` call (line ~549) and is
  only recorded *after* `quantize_cache_fn` returns — so the logged `forward`
  span conflates model compute with KV cache quantization. This exact
  instrumentation gap is documented in the `exo-dsv4-prefill-tuning` skill
  and doc Section 99 Track C.

- TP: `generate.py` line ~1408-1412, inside `prefill_batched`'s chunk loop:
  ```
  model(padded_tokens[:, offset : offset + n_to_process], cache=batched_cache)
  request_trace.record(f"prefill_batched.chunk{chunk_idx}.forward({n_to_process}tok)", _t_fwd)
  ```
  Same structural pattern: a `model(...)` forward call per chunk, timed with
  the identical span-naming convention (`prefill_batched.chunk{i}.forward`).

- `maybe_quantize_kv_cache` (`mlx-lm/mlx_lm/generate.py:293`) is imported
  once at `generate.py:15` and used via a `functools.partial` in BOTH the PP
  chunk loop (`generate.py:506-510`, called at line 570/653) and — critically
  — inside mlx-lm's own `stream_generate`/`generate_step`
  (`mlx-lm/mlx_lm/generate.py:373-374,412,489`), which is what
  `prefill_batched`'s serial fallback (`_serial_prefill_fallback` →
  `prefill()` → `stream_generate()`) and any single-stream TP prefill goes
  through. So quantize is on every path, PP or TP, chunked or eager.

- Gate that determines which chunk-loop runs: `prefill()` line ~867,
  `is_pipeline = _has_pipeline_communication_layer(model)`, then
  `if is_pipeline and num_tokens >= prefill_step_size: pipeline_parallel_prefill(...) else: stream_generate(...)`.
  This gate selects the OUTER driver (PP loop vs. mlx-lm's own loop), NOT
  whether `model(...)` + quantize run — those run in every branch. The live
  cluster's `MLX_JACCL_SHARDING_MODE=Pipeline` env var controls which layer
  classes (`PipelineFirstLayer`/`MetaFramedPipelineFirstLayer` vs
  TP-sharded equivalents) get installed on `model.layers`, which in turn
  decides `_has_pipeline_communication_layer`'s answer — but it does not
  gate `maybe_quantize_kv_cache` or the existence of a per-chunk `model(...)`
  call at all. Under TP, `is_pipeline` is False, so the multi-stream case
  uses `prefill_batched`'s own dedicated chunk loop (same `model(...)` +
  `mx.eval` pattern, no PP dummy-iteration bookkeeping) rather than
  `pipeline_parallel_prefill`; the single-stream case falls through
  `prefill()` → `stream_generate()`.

- **TP executes this? YES**, both sub-paths (`prefill_batched`'s own loop
  for multi-stream, `stream_generate`'s loop for single-stream/SSM
  fallback). Whatever fraction of the "forward" 98%/1901ms-intercept/
  +2.03ms-per-chunk cost is actual model compute (DeepSeek-V4 forward pass:
  MoE dispatch, indexer GEMM, attention) is architecturally identical
  compute under TP — same model, same layers, same math — just sharded
  differently across ranks. The KV-quantize sub-fraction is the *exact
  same function* on both paths.

### #2: the per-chunk `mx.eval` of cache state (`eval_cache` span) — measured FLAT, not the driver, but confirmed shared

**CONFIRMED, and CONFIRMED shared, but ruled out as the depth-scaling
driver already (Section 99 Track C: ~2% of tracked time, +0.033ms/chunk).**

- PP: `generate.py` line ~582, `mx.eval([c.state for c in _prompt_cache])`,
  timed as `prefill.chunk{i}.eval_cache`.
- TP: `generate.py` line ~1429-1432, `mx.eval([c.state for c in
  batched_cache])`, timed as `prefill_batched.chunk{chunk_idx}.eval_cache`.
- Same `mx.eval`-of-cache-state pattern on both paths, same relative
  position in the chunk loop (right after the forward+quantize). Since it
  was independently measured flat on the PP trace, it is not expected to be
  the depth-scaling driver on TP either, but it IS shared code and DOES
  execute under TP.
- **TP executes this? YES**, but it is not the dominant cost (2% of tracked
  time on the measured PP trace; no reason structurally to expect a
  different share under TP since it's the identical `mx.eval` call over an
  identical cache-state list shape per stream).

### #3: `CacheSnapshot` / SSM-state snapshot-copy machinery — CONFIRMED PP-relevant, but gated OFF for the TP batched path; residual exposure only via the shared serial fallback

**CONFIRMED, and CONFIRMED PP-specific in the dominant TP code path, with one caveat.**

- `snapshot_ssm_states()` (`cache.py:320`) does a per-chunk deep COPY of
  non-sliceable cache layers via `copy_rotating_kv_cache` (numpy round-trip
  detach, `cache.py:171-189`) and `_copy_arrays_cache`/`_copy_cache_list`
  (`cache.py:360-361`) for `ArraysCache`/non-trimmable `CacheList` entries.
  This is analogous in mechanism (a per-turn/per-chunk state copy) to the
  `d5f6c421` incident, but NOT identical: `d5f6c421` was a full deepcopy of
  the whole cache; this is a scoped, type-dispatched copy of only the
  non-sliceable sub-layers, retention-capped (`_SNAPSHOT_RETENTION = 2` at
  the generator level, `_LEAF_SNAPSHOT_RETENTION`/`EXO_LEAF_SNAPSHOT_RETENTION
  = 4` at the trie level, `cache.py:81`).
- **Called from:** `prefill()`'s `progress_callback` closure
  (`generate.py:~814`, inside `if has_ssm:`), which fires on EVERY chunk
  boundary of the serial prefill path (both the PP branch via
  `pipeline_parallel_prefill`'s `prompt_progress_callback`, and the
  `stream_generate` branch via `combined_progress_callback`).
- **`has_ssm = has_non_kv_caches(cache)`** (`generate.py:791`) gates whether
  this fires at all — `has_non_kv_caches` → `is_non_trimmable_cache_entry`
  → True only for `ArraysCache`/non-trimmable `CacheList` entries
  (DeepSeek-V4's DeltaNet/SSM sparse layers).
- **Does TP execute this?** For the dominant multi-stream TP path
  (`prefill_batched`): **NO for the snapshot-copy cost itself** —
  `prefill_batched` explicitly checks `_has_arrays_cache(cache_list)` at
  entry (`generate.py:~1272-1285`) and if any stream's cache holds an
  `ArraysCache` layer, it **falls back to `_serial_prefill_fallback` →
  `prefill()` → the serial path** (which DOES call `snapshot_ssm_states`).
  So under TP: if the model's cache has no SSM/ArraysCache layers,
  `prefill_batched` runs its own loop and `snapshot_ssm_states` never fires
  — genuinely absent from the batched-TP forward-cost budget. If the model
  DOES have ArraysCache layers (DeepSeek-V4-Flash's sparse/DeltaNet layers
  — need to confirm cache composition on the live checkpoint), `prefill_batched`
  silently reroutes to the serial path and per-chunk snapshotting resumes —
  meaning **the caveat is live and depends on whether DSv4-Flash's cache
  actually contains ArraysCache entries**, which this read did not confirm
  either way (comment at `generate.py` cache.py:960 references DeepSeek-V4
  sliding-window RotatingKVCache + PoolingCache layers being SSM/pooling
  related — this needs a runtime check, not a code-only one, to resolve
  definitively).
- Whether it scales with context depth: the copy is bounded per-call by the
  cache layer's *current* size (not cumulative depth), and retention is
  capped, so its cost per chunk should be roughly flat, not growing — this
  is INFERRED from the code's shape, not measured. It is a plausible partial
  explanation for the small residual growth if the serial fallback path
  is in fact what's running on the live 202-225 tok/s trace, but Section 99's
  `eval_cache` (flat) measurement doesn't cover this because it's a distinct
  span (`prefill.cache_trim_and_rollback`/`snapshot_ssm_states` isn't in the
  Track C span list quoted).

## What this means for TP's prefill ceiling

Candidate #1 (the forward+quantize span, 98% of tracked time) is
**confirmed present on both PP and TP code paths, unconditionally**. It is
not gated by `MLX_JACCL_SHARDING_MODE` or `is_pipeline` — that flag only
picks which *driver loop* wraps the same `model(...)` + quantize call. So
whatever the live cluster's Pipeline-mode 202-225 tok/s reflects as
"forward cost growing with depth," the SAME model-forward + KV-quantize
machinery runs under TP's `prefill_batched` loop. **This caps TP's prefill
ceiling too** — TP does not get a free pass on this cost just because PP is
being deprioritized.

The one PP-flavored candidate found (#3, SSM-state snapshotting) is
correctly GATED OFF the dominant TP batched-prefill path when the cache has
no ArraysCache entries, but the gate's real-world state on the live
DeepSeek-V4-Flash checkpoint was not confirmed by this read — flagged as
the one open INFERRED item.

## Confirmed vs Inferred summary

| Claim | Status |
|---|---|
| `model(...)` forward call executes every chunk on both PP (`_pipeline_parallel_prefill_steps`) and TP (`prefill_batched`) loops | CONFIRMED (code read, generate.py:568-571, 1408-1412) |
| `maybe_quantize_kv_cache` runs on both PP's explicit call and inside mlx-lm's `stream_generate`/`generate_step` (TP's single-stream/fallback path) | CONFIRMED (generate.py:506-510,570,653; mlx-lm/generate.py:293,373-374,412,489) |
| The `forward` span timer conflates model compute + KV quantize (can't separate from existing logs) | CONFIRMED (generate.py timer placement, matches skill/Section 99's stated instrumentation gap) |
| `eval_cache` span (mx.eval of cache state) runs on both PP and TP loops | CONFIRMED (generate.py:582, 1429) |
| `eval_cache` is flat/~2%, not the depth-scaling driver | CONFIRMED BY REFERENCE (Section 99 Track C measurement, not re-derived here) |
| `snapshot_ssm_states` (CacheSnapshot machinery) does NOT run on `prefill_batched`'s batched-TP loop when no ArraysCache layers are present | CONFIRMED (generate.py:~1272-1285, explicit `_has_arrays_cache` gate to serial fallback) |
| DeepSeek-V4-Flash's actual live cache composition (does it contain ArraysCache layers, forcing the serial-fallback/snapshot path even under `prefill_batched`?) | **NOT CONFIRMED — requires a runtime check**, out of scope for code-only reading |
| Snapshot-copy cost is roughly flat per chunk (bounded by current layer size, capped retention), not scaling with depth | INFERRED from code shape, not measured |
| Whether growth in candidate #1 is dominated by model compute vs KV quantize | **UNKNOWN — this is the real open question**, blocked on the same instrumentation gap on both PP and TP |

## Cheapest experiment to confirm the top candidate under TP

**Split the conflated `forward` span into two spans** — this is the
change Section 99 already identifies as the prerequisite, and it applies
identically to both loops since they share the exact same
`model(...)` / `quantize_cache_fn(...)` call-then-time pattern:

1. In `prefill_batched`'s chunk loop (`generate.py` ~1408-1421) and in
   `_pipeline_parallel_prefill_steps` (~549-571), start a second timer
   immediately after `model(...)` returns and before `quantize_cache_fn`/
   the TP loop's own quantize call, recording `forward.model_only` and
   `forward.quantize_only` as separate `request_trace` entries instead of
   one combined `forward` span.
2. This requires a tiny code change (not a pure env toggle), but it's the
   **cheapest** because: (a) it needs no new instrumentation machinery —
   `request_trace`/`T()` already exist and are gated by
   `EXO_TRACING_ENABLED` (no-op cost when off); (b) it applies to BOTH
   loops with the same one-line split, so a single fix answers the
   question for TP and PP simultaneously; (c) it requires no relaunch of
   the cluster's sharding mode — just re-running the existing
   `bench/phase3_precheck_depth_throughput.py`-style probe (with the
   chars/token bug already fixed) under whichever mode is live, with
   `EXO_TRACING_ENABLED=1`.
3. If a live TP run with this split shows `forward.quantize_only` flat and
   `forward.model_only` carrying the ~2ms/chunk growth (mirroring PP's
   measured shape), that CONFIRMS candidate #1's growth is real model
   compute (MoE/indexer/attention), shared and unavoidable under TP too —
   not an artifact of KV quantize. If `quantize_only` instead grows with
   depth, that points at cache-size-dependent quantize cost, still shared
   with TP, but suggests a different optimization target (e.g. faster
   quantize, not model-side compute).

No existing env toggle isolates model-forward from quantize directly, so
this experiment requires the smallest possible code change (a timer split)
rather than a pure toggle — flagged honestly rather than claiming a
toggle exists that doesn't.
