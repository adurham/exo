# Phase 0a — per-call time decomposition of the real 42-call turn

**Session** `20261007_092009_9a2ed7` · `provider='custom'` · model
`dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw` (engine `dsv41`, TP2 over jaccl;
this window is `m4-1` = `studio1` = rank 1, the API/master node).

**Question.** Split each call's `latency_seconds` (state.db `api_calls`) into
**measured** prefill and decode time from exo.log markers — replacing the old
`output/20.66` estimate — and decide cache hit/miss per call.

**Artifacts.** `bench/phase20_turn_decomp.py` (extractor, stdlib only),
`bench/phase20_tests/test_phase20_turn_decomp.py` (13 tests),
`docs/benchmarks/phase20-throughput/turn_decomp.csv` (42 rows),
`.../turn_decomp_summary.json`.

## Sources

* **Ledger** — state.db `api_calls`, opened read-only
  (`file:...?mode=ro`, `uri=True`), **exact** `provider='custom' AND
  model='dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw'` (SQLite `LIKE` is
  case-insensitive and would also match the unrelated ollama model
  `deepseek-v4.1-flash`). → 42 rows, `sum(latency)=1530.587 s`.
* **Log** — `/Users/adam.durham/.hermes/cache/scratch/phase20/m41_0901_1400.log.zst`
  (m4-1 boot 09:01–14:00), read via `zstdcat` (decompressed form never written
  into the repo). Window 09:20:00–09:52:00 CDT contains the whole turn; it also
  holds **5 other-session POSTs** (09:29:43.356 … 09:30:55.944) that do not
  belong to this session — matched out by time-nearest-assignment and prompt
  equality.

## Marker semantics (verified: source + real lines)

| marker | emitted where | meaning |
|---|---|---|
| `API request: POST /v1/chat/completions` | `api.main:_log_requests` | client arrival `t_post` |
| `[DSV41] prefill controls: … (rows=M, base=2048)` | `dsv41.session:engine_prefill` (line 333) | prefill **entry**; `rows=` = rows this turn re-feeds (== `prefill=`), or the full prompt on a cold call |
| `[DSV41] session reuse: this N-token prompt matches a resident conversation on R rows` | `dsv41.session:get` (line 1004) | a resident session was matched; emitted at the same instant as `prefill controls` |
| `[DSV41] turn reuse: prompt=N prefill=M reuse=R cache=C [rewind=W] UNCOMMITTED` | `dsv41.engine:_start_turn` (line 957) | emitted **immediately after `session.prefill()` returns** → prefill-**completion** `t_prefill_end`. Emitted **only when `reuse>0`**; `rewind=W` present only when the cache rolled back (a prefix collapsed), otherwise absent. |
| `Executing command: TaskFinished(command_id=…, finished_command_id=…)` | `master.main:_command_processor` | request **end** `t_end`; `runner idle: reclaimed MLX allocator pool` + `runner ready` follow it |
| `[DSV41] hier geometry: … estrip=E block=8 …` (stderr, WARNING) | `mlx_lm...dsv41.indexer._log_hier_geometry` | one line per (shape,strip) key, **per prefill CHUNK** (deduped per key). `estrip` ⇒ the chunk row-count `n` via `estrip = max(8, (128MiB/(n·(6·128+4)))//8·8)`. This is the only cold-prefill progress evidence. |

Definitions used: `t_prefill_start` = the session-reuse/prefill-controls line
(prompt-matched, else the first prefill-controls) — 0.2–0.34 s after `t_post`;
`t_prefill_end` = the turn-reuse line; `t_end` = the first `TaskFinished` after
`t_prefill_end` (or after `t_post` for the cold call).
`prefill_s = t_prefill_end − t_prefill_start`; `decode_s = t_end − t_prefill_end`;
`pre_s = t_prefill_start − t_post`; `post_s = ledger.ended_at − t_end`.

## Which calls lack which markers, and why

* **Call 1 (cold) has no `turn reuse` line and no `session reuse` line.** It
  re-feeds the full 24095-row prompt on an empty cache, so `reuse==0` and the
  turn-reuse INFO is suppressed. Its prefill end is therefore **not** marked —
  the brief forbids estimating it, so call 1's `prefill_s`/`decode_s` are left
  **blank** and it is reported **`UNSPLIT_COLD`**, not guessed. Its
  `delta_rows_prefilled = 24095` is taken from the cold call's own
  `prefill controls (rows=24095)` line. Its two `hier geometry` lines
  (estrip 80 → n≈2022, then estrip 104 → n≈1552) are the only cold-prefill
  progress evidence: consistent with the 2048-row chunked cold prefill
  (~13 chunks), but with per-chunk time absent the split cannot be timed.
* **Calls 11 and 21 have no `hier geometry` line.** Both are tiny deltas (1567
  and 4 rows) whose window still shows one clean in-window chunk (estrip 880 →
  n≈196; estrip 128 → n≈1280, i.e. the deduped key was already logged). The
  geometry census is per unique (shape,strip) key; a repeat key prints nothing.
  Absence is therefore **not** a signal — it does not mean "no prefill".
* **`reuse undershoot` warning** fires only when `refed (prefill) > 256` and
  `reuse>0` (engine.py:958) — its absence on the smallest deltas (calls 2, 21)
  is expected, not a signal.
* No other marker type is involved; all 42 calls have exactly one `TaskFinished`
  in their own window.

## Clock alignment

`started_at`(UTC) − 5 h ↔ the matching POST line (node-local CDT): every call's
nearest POST is within **0.214 s** (mean 0.033 s). That is the measured
laptop/ledger ↔ node skew implied by the match (the ledger `started_at` is the
agent-side send time; the POST line is the node-receive time).

## Results

| quantity | value |
|---|---:|
| `sum(latency)` (42 calls) | **1530.587 s** |
| `sum(prefill_s)` (41 split) | **327.385 s** |
| `sum(decode_s)` (41 split) | **1085.405 s** |
| `sum(prefill_s + decode_s)` | **1412.79 s** |
| **Gate 0a ratio** = model / latency | **0.9230** ✅ within ±10% |
| model-side unaccounted | **117.797 s** = 98.798 s (whole UNSPLIT cold call 1) + 17.14 s `sum(pre_s)` + 1.30 s `sum(post_s)` + ≈0.77 s ledger↔POST offset |
| `sum(gap_to_next_call_s)` | **283.858 s** (brief expected 283.86 ✅) |
| wall span (first POST → last end) | 1814.44 s |
| **cache misses** | **0 / 41** (call 1 cold, excluded) ✅ reproduces the prior audit |
| degenerate rows (dctx ≤ 0) | 0 |

**Shares.** decode 76.8% / prefill 23.2% of model time; model time = 77.9% of the
1814.44 s wall span.

**Decode tok/s** (per call, n=41): min 17.75, median 21.66, max 25.76;
weighted mean `sum(output)/sum(decode_s)` = **19.92 t/s**.

**Prefill rows/s** (per call): over all 41 split calls min 18.9 / median 221.7 /
max 256.1; over the 32 calls with `delta_rows ≥ 500` (below that, fixed per-call
overhead dominates the prefill window and drags the apparent rate down) min
186.0 / median 229.1 / max 256.1, weighted mean **238.4 rows/s**. *This last
distribution is an artifact of `prefill_s` including the whole turn-reuse window
(entry→return), which also covers the post-prefill ladder/checkpoint work; it is
not a pure kernel rows/s and should not be read as one.*

**`prompt == reuse + prefill`** holds on **all 41** turn-reuse calls (0
exceptions) — establishing that `rows == tokens` for prefill.

## Reconstruction honesty

* Call 1: `prefill_s` / `decode_s` = **UNKNOWN** (`UNSPLIT_COLD`). Its total time
  is known (98.798 s) but cannot be split from the available markers; it is left
  in the model-side-unaccounted line.
* Every other cell is derived from a real marker timestamp or a real ledger
  field; nothing is estimated.

## Caveat on `prefill_s`

`prefill_s` is measured as `t_prefill_end − t_prefill_start`, i.e. from the
prefill-entry marker to the turn-reuse return. `session.prefill()` includes the
margin-rung/ladder checkpoint bookkeeping, so `prefill_s` is a *turn entry→
return* window, slightly larger than pure forward-pass time. It is nonetheless
the only measured prefill interval available, and it is what the gate consumes.
