# Phase 2 — Delta-prefill audit + prefill entry-point × knob matrix

**Scope.** READ-ONLY forensic audit of the real 42-call Hermes turn (`session_id=20261007_092009_9a2ed7`, provider `custom` / dsv41 engine), plus a source-read of the dsv41 prefill path.

**Sources.**
- **A1 per-call split** (`/private/tmp/next14-gamma/docs/benchmarks/phase19-latency/raw/A1-real-turn-split.md`): `delta_rows (prefill=)`, `reuse_rows (reuse=)`, `rewind=`, plus the exo.log line numbers. Values cross-checked there against `macstudio-m4-1:~/exo.log`.
- **Ledger** `/Users/adam.durham/.hermes/state.db`, table `api_calls`, opened `file:...?mode=ro`. `input_tokens` (= `prompt_tokens_total` − `cache_read_tokens`) is the *uncached delta actually prefilled* and equals A1's `delta_rows` for all 42 calls (0 mismatches).
- **Source** (worktree `/private/tmp/levers-wt`, branch `deploy/next15-levers` @ `71a8c94c2`): `src/exo/worker/engines/mlx/dsv41/{engine,session,park,builder,load,thinking,engram}.py`, the launcher `start_cluster.sh`, and the mlx-lm submodule at `/Users/adam.durham/repos/exo/mlx-lm/mlx_lm/models/deepseek_v41/{session_cache,indexer,indexer_hierarchical,prefill}.py` (worktree `mlx-lm/` is an empty submodule dir, so the live main-repo copy was read).

> The hypothesis under test — *“some of the 32/42 delta-dominated calls prefill far more rows than the true delta (prefix-cache miss / system-prompt churn / a missed prefill entry point)”* — is **refuted** for this trace: **0 of 41 warm calls over-prefill**. See §A.

---

## Mechanistic resolution of the “prefill < promptΔ and reuse > previous prompt” nuance

Two entry points decide how many rows a turn re-feeds:

- `SessionCache.plan(ids)` (mlx-lm `session_cache.py:460-473`): `lcp = common_prefix_len(ids, self._ids)`; if `lcp >= cache.offset` → `boundary = cache.offset` (exact hit, no rewind), else `boundary = self._boundary_le(lcp)` (**newest checkpoint at or below `lcp`**). Re-fed rows = `prompt_len − boundary`.
- The resident token history `self._ids` is **not** the previous prompt: after each turn `sync_history` (`dsv41/session.py:840-866`) sets `cache._ids = prompt_prev + gen_prev[:-1]`, and `append_turn` sets `self._ids = ids.copy()` then `stats["reused"] += boundary` (`session_cache.py:635,637`).

So:

1. **`reuse_rows` (=`boundary`) counts the resident cache’s committed rows, which include the previous turn’s *generated* tokens** (all but the last, which is the next anchor). That is why `reuse_rows > prompt[i−1]` (e.g. call 3 `reuse=24431` vs `prompt[2]=24340`: 91 of call 2’s ~95 generated tokens matched the re-rendered reply).
2. **The typical boundary lands exactly on `prompt[i−1]`** — the *prompt-end checkpoint* taken in `Conversation.prefill` (`dsv41/session.py:684-686`, “Checkpoint the prompt end: a follow-up turn whose re-rendered reply differs from the generated tokens then reuses this whole prompt”). When the client re-renders the assistant reply differently from the generated tokens, `lcp` stops at `prompt[i−1]`, the checkpoint at `prompt[i−1]` is the newest one ≤ lcp, and re-fed rows become exactly `prompt[i] − prompt[i−1]`. This is the observed `act == expected` for 36/41 warm calls.
3. **`act < expected`** (calls 3, 8, 20, 21, 41) means the re-rendered reply *did* match further, so the boundary rose **above** `prompt[i−1]` → a reuse **hit**, not a miss. (`act == expected` for 36, `act < expected` for 5, `act > expected` for **0**.)

The `reuse undershoot` warning (`engine.py:971-980`: fires when `turn.reused_tokens` is truthy **and** `prefill_tokens > 256`) is therefore **not** a miss signal: it fires whenever a warm call re-feeds more than a small chunk, and it fired on 39/42 calls even though the re-feed equals the genuine client delta. The “undershoot” is the 1–3-row BPE-seam rewind below the newest boundary (`rewind=` in the log), which the margin rung (`session.py:732-751`) exists to bound — not an over-prefill.

**Correct denominator for “miss”.** `expected_delta(i) = prompt[i] − prompt[i−1]` (= the tokens the client actually added: new tool output + new assistant message + injected content). It is the right denominator precisely because the reuse boundary is pinned at (or a few rows above) the previous prompt end. A **miss** is `act > expected` (ratio > 1). `Σ(prompt − cread)`/`cread` are *not* a second denominator — they describe cache residency, not the new-token count.

---

## (A) Delta-prefill audit

Ratio = `actual_prefill / expected_delta`; **MISS** = ratio > 1.5. Call 1 is cold (expected = full prompt).

| seq | prompt | cach_read | actual (delta_rows) | expected (promptΔ) | ratio | excess | MISS |
|---:|---:|---:|---:|---:|---:|---:|:--:|
| 1 | 24095 | 0 | 24095 | 24095 (cold full) | 1.000 | 0 | — |
| 2 | 24340 | 24095 | 245 | 245 | 1.000 | 0 | no |
| 3 | 25300 | 24431 | 869 | 960 | 0.905 | −91 | no |
| 4 | 28743 | 25300 | 3443 | 3443 | 1.000 | 0 | no |
| 5 | 30178 | 28743 | 1435 | 1435 | 1.000 | 0 | no |
| 6 | 39293 | 30178 | 9115 | 9115 | 1.000 | 0 | no |
| 7 | 53564 | 39293 | 14271 | 14271 | 1.000 | 0 | no |
| 8 | 56313 | 54588 | 1725 | 2749 | 0.628 | −1024 | no |
| 9 | 57541 | 56313 | 1228 | 1228 | 1.000 | 0 | no |
| 10 | 59602 | 57541 | 2061 | 2061 | 1.000 | 0 | no |
| 11 | 61169 | 59602 | 1567 | 1567 | 1.000 | 0 | no |
| 12 | 63904 | 61169 | 2735 | 2735 | 1.000 | 0 | no |
| 13 | 64853 | 63904 | 949 | 949 | 1.000 | 0 | no |
| 14 | 66620 | 64853 | 1767 | 1767 | 1.000 | 0 | no |
| 15 | 68147 | 66620 | 1527 | 1527 | 1.000 | 0 | no |
| 16 | 69862 | 68147 | 1715 | 1715 | 1.000 | 0 | no |
| 17 | 70159 | 69862 | 297 | 297 | 1.000 | 0 | no |
| 18 | 70622 | 70159 | 463 | 463 | 1.000 | 0 | no |
| 19 | 71044 | 70622 | 422 | 422 | 1.000 | 0 | no |
| 20 | 72678 | 72070 | 608 | 1634 | 0.372 | −1026 | no |
| 21 | 73568 | 73564 | 4 | 890 | 0.004 | −886 | no |
| 22 | 74893 | 73568 | 1325 | 1325 | 1.000 | 0 | no |
| 23 | 75338 | 74893 | 445 | 445 | 1.000 | 0 | no |
| 24 | 83663 | 75338 | 8325 | 8325 | 1.000 | 0 | no |
| 25 | 84980 | 83663 | 1317 | 1317 | 1.000 | 0 | no |
| 26 | 86015 | 84980 | 1035 | 1035 | 1.000 | 0 | no |
| 27 | 86529 | 86015 | 514 | 514 | 1.000 | 0 | no |
| 28 | 87509 | 86529 | 980 | 980 | 1.000 | 0 | no |
| 29 | 88853 | 87509 | 1344 | 1344 | 1.000 | 0 | no |
| 30 | 90793 | 88853 | 1940 | 1940 | 1.000 | 0 | no |
| 31 | 92236 | 90793 | 1443 | 1443 | 1.000 | 0 | no |
| 32 | 94402 | 92236 | 2166 | 2166 | 1.000 | 0 | no |
| 33 | 95238 | 94402 | 836 | 836 | 1.000 | 0 | no |
| 34 | 95611 | 95238 | 373 | 373 | 1.000 | 0 | no |
| 35 | 96008 | 95611 | 397 | 397 | 1.000 | 0 | no |
| 36 | 97134 | 96008 | 1126 | 1126 | 1.000 | 0 | no |
| 37 | 98382 | 97134 | 1248 | 1248 | 1.000 | 0 | no |
| 38 | 99585 | 98382 | 1203 | 1203 | 1.000 | 0 | no |
| 39 | 100945 | 99585 | 1360 | 1360 | 1.000 | 0 | no |
| 40 | 102063 | 100945 | 1118 | 1118 | 1.000 | 0 | no |
| 41 | 104396 | 103088 | 1308 | 2333 | 0.561 | −1025 | no |
| 42 | 104880 | 104396 | 484 | 484 | 1.000 | 0 | no |

### Summary

- **Misses (ratio > 1.5): 0** of calls 2–42 (and 0 of 42 including the cold call, whose ratio is exactly 1.0 by definition). No cache-miss, no system-prompt churn, no missed entry point manifested as over-prefill in this turn.
- **Reverse sign found:** 5 calls re-feed **less** than the prompt delta (3, 8, 20, 21, 41 with excess −91…−1026) — reuse hits past the previous prompt boundary, i.e. **under**-prefill, the benign direction.
- **Σ actual_prefill:** all 42 = **100 828**; warm calls 2–42 = **76 733**.
- **Σ expected_delta:** warm calls 2–42 = **80 785** (**act is 4.0 % *below* the client delta**); including the cold call-as-full-prompt = 104 880.
- **Ratio distribution** (all 42): min **0.004** (call 21), median **1.000**, max **1.000**. Warm-only: min 0.004, median 1.000, max 1.000.
- **Top-5 calls by absolute excess (act − expected):** none positive. The five largest **negative** excesses are call 20 (−1026), call 8 (−1024), call 41 (−1025), call 21 (−886), call 3 (−91). Largest **positive** excess = 0.
- **Max single-call delta_rows:** **24 095** (call 1, cold full prefill). Excluding the cold call: **14 271** (call 7).

### Go-condition for the later chunk-4096 experiment

Calls with `delta_rows > 4096`: **call 1 (24095, cold), call 6 (9115), call 7 (14271), call 24 (8325)** → Σ = **55 806**.

- Share of **Σ delta_rows (all 42 = 100 828)** from `delta_rows > 4096` calls: **55.3 %**.
- Share of **Σ delta_rows warm-only (calls 2–42 = 76 733)** from `delta_rows > 4096` warm calls: **41.3 %**.

**Verdict: the ≥25 % go-condition is MET under either reading (41.3 % warm-only, 55.3 % all-calls).** The single call able to exercise a >4096-row chunk is the 14 271-row warm delta at call 7 (plus 9115 @ call 6, 8325 @ call 24).

---

## (B) Prefill entry-point × knob matrix

Chain for a served turn: `Dsv41Engine._start_turn` → `Conversation.prefill` → `SessionCache.append_turn` → `_prefill_call` **or** `_prefill_planned` → `engine_prefill` (the actual chunk loop). Parked resume rebuilds a `Conversation` (same driver wiring) via `ParkedStore.try_restore → restore_conversation`. `chunked_prefill` is the mlx-lm **fallback** driver, replaced by `engine_prefill` whenever the session gets exo’s `prefill_fn` partial.

| # | entry_point (file:line) | function | knobs read (file:line) | launcher forwards? | gap |
|---|---|---|---|---|---|
| 1 | `dsv41/engine.py:938` | `_start_turn` (delta + cold full prefill) | chunk `self._chunk` (`engine.py:348`); fence via driver (`session.py:346`) | chunk ← `EXO_PREFILL_STEP_SIZE` (launcher `2019`) Y; fence Y (`2721`) | — |
| 2 | `dsv41/session.py:617` | `Conversation.prefill` | draft base; prompt-end checkpoint (`686`); margin rung (`729,732`) | — | — |
| 3 | mlx-lm `session_cache.py:584` | `SessionCache.append_turn` | boundary from `plan()` (`460`); `chunk_plan` → planned path | — | — |
| 4 | mlx-lm `session_cache.py:483` | `_prefill_call` → exo `prefill_fn` partial | `chunk`/`long_chunk`/`long_threshold`; `fence_every`/`transient_budget_bytes`/`fence_hook` via partial (`dsv41/session.py:545-555`) | Y | — |
| 5 | **mlx-lm `session_cache.py:506`** | **`_prefill_planned`** (image-span / explicit `chunk_plan`) | forces `chunk=long_chunk=piece_size`, `long_threshold=10**9` (`529-530`) | — | **GAP: pins the fixed-crossover branch, so `choose_prefill_step` / `EXO_PREFILL_TRANSIENT_BUDGET_MB` is NOT honored on this entry point** (see Finding 1) |
| 6 | `dsv41/session.py:235` | `engine_prefill` (the real chunk loop) | `EXO_PREFILL_FENCE_EVERY` (`92,228-232`); `EXO_PREFILL_TRANSIENT_BUDGET_MB` (`99,218-225`); `base`/`long_chunk`/`long_threshold` (`312-317`); policy (`361-378`) | Y (`2720,2721`) | — |
| 7 | mlx-lm `session_cache.py:145` | `chunked_prefill` (**fallback driver if no `prefill_fn`**) | `plan_step` fixed crossover only; **ignores `fence_every`/`async_depth`** (`154-157`) | — | **GAP (latent): the transient-budget/FENCE knobs are dead if the engine partial is not installed** (see Finding 2) |
| 8 | `dsv41/park.py:436` | `restore_conversation` (parked resume) | rebuilds `Conversation` with all of `conv_kw` = `engine.kw` + fresh fence hook (`session.py:1040`); honors chunk/threshold/transient/checkpoint knobs | Y for the knobs it carries | — (honors the full set → contrast case) |
| 9 | `dsv41/session.py:979 / 1052 / 1078` | `Dsv41Sessions.get` / `_restore_parked` / `_open` | `MIN_REUSE_TOKENS=64` (`70`); park gates `EXO_DSV41_PARK*` (`73-77`) | `EXO_DSV41_PARK*` **N** (see Finding 3) | park is env-gated downstream, not by the launcher |
| 10 | `dsv41/session.py:753 / 515-522` | `maybe_checkpoint` / `Conversation.__init__` | `EXO_DSV41_CHECKPOINT_{SPACING_ROWS,MARGIN_ROWS,KEEP}` (`112,120,127`) | Y (`2728-2730`) | — |
| 11 | mlx-lm `indexer.py:138` | `_HIER_CONSUMER_SKIP` (indexer coarse pass) | `DSV41_INDEXER_CONSUMER_SKIP` (**import-time constant**) | Y (`2689`) | — (global; active for every entry point) |
| 12 | mlx-lm `indexer.py:111-138`, `indexer_hierarchical.py:132-134` | indexer row/hier knobs | `DSV41_INDEXER_ROW_BF16 / HIER / HIER_* / TILE*` (import-time) | partial: `ROW_BF16, HIER, HIER_*` Y (`2683-2689`); `TILE*/TILED_IMPL/TILE_MIN_NB/COMPILE` **N** | see Finding 4 |
| 13 | `dsv41/load.py:43,100` | layer selection | `EXO_DSV41_LAYERS` | **N** | load-time; not prefill-entry-specific |
| 14 | `dsv41/builder.py:117,43` / `thinking.py:68` / `load.py:46`+`engram.py:27` | vision / speculative / think-markers / engram | `EXO_DSV41_{VISION,SPECULATIVE,THINK_MARKERS,ENGRAM_DIR,ENGRAM_TOKEN_MAP}` | **N** | orthogonal to prefill chunking |

### Findings (called out)

**Finding 1 — “missed entry point”: the image-span / `chunk_plan` path bypasses the transient-budget policy.** `_start_turn` passes a `chunk_plan` only when the request carries images (`engine.py:954`; `plan = _first_chunk_covers(...) if embeddings is not None else None`). That routes through `SessionCache._prefill_planned`, which pins `{"chunk": piece, "long_chunk": piece, "long_threshold": 10**9}` (`session_cache.py:529-530`). In `engine_prefill`, a non-None `long_threshold` selects the fixed-crossover branch (`session.py:317, 361-362`) and **never calls `choose_prefill_step`** (`session.py:372`). Consequence: on **any image request**, `EXO_PREFILL_TRANSIENT_BUDGET_MB` is inert (the partial still passes `transient_budget_bytes`, but it is unused) and the chunk no longer shrinks with context — the very memory policy the knob exists for. The fence knob is *not* affected (it rides the partial at `session.py:547`, independent of the threshold). This is the single entry point that fails to honor a knob the other prefill entry points (`#1/#4/#6/#8`) honor.

**Finding 2 — latent fallback gap.** The engine installs its own driver as the `prefill_fn` partial (`session.py:545-555`). If that wiring were absent, `resolve_prefill_fn(None)` (`session_cache.py:205`) falls back to mlx-lm’s `chunked_prefill`, whose docstring states `fence_every`/`async_depth` are “accepted and ignored” (`session_cache.py:154-157`) and which uses `plan_step` (fixed crossover) instead of `choose_prefill_step`. On that path both `EXO_PREFILL_TRANSIENT_BUDGET_MB` and `EXO_PREFILL_FENCE_EVERY` are dead. Not live in production (the partial is always installed), but it is the same knob-drop class the launcher comment at `2715-2719` warns about.

**Finding 3 — `EXO_DSV41_PARK*` are not in the launcher forwarding list.** `grep` over the whole `EXO_ENV` allow-list returns no `EXO_DSV41_PARK`, `EXO_DSV41_PARK_DIR`, `_MAX_GB`, `_MIN_TOKENS`, `_MAX_COUNT`, nor `EXO_DSV41_LAYERS / VISION / SPECULATIVE / THINK_MARKERS / ENGRAM_*`. The code reads them (`park.py:100-104`; `load.py:43,46`; `builder.py:43,117`; `thinking.py:68`; `engram.py:27`), so any shell value set for them on the launcher host is silently dropped to the runner — the read-at-code / dead-in-deployment class. (Parking is on by default, so the *default* path still works; only the overrides are unreachable.) `EXO_DSV41_LAYERS` and `EXO_DSV41_VISION` are load-time, not prefill-entry, knobs.

**Finding 4 — indexer tiled-path knobs not forwarded.** The launcher forwards `DSV41_INDEXER_ROW_BF16 / HIER / HIER_{EXACT_MB,EXACT_STRIP,STRIP,OVERFETCH} / CONSUMER_SKIP` (`2683-2689`) but **not** `DSV41_INDEXER_TILE / TILE_MB / TILE_MIN_NB / TILED_IMPL / TILE_FORCE / COMPILE`, which the model still reads (`indexer.py:32,34,42,48,67-99,210-267`). The tiled-indexer A/B (`INDEXER_TILED_P_PLAN.md`) is therefore not reachable from the launcher. Orthogonal to the prefill entry points (all read at mlx-lm import).

**No consumer-skip gap.** `DSV41_INDEXER_CONSUMER_SKIP` (`indexer.py:138`, default ON, bit-exact) is a module-level constant read once at import, so it is active uniformly for every prefill entry point — it is **not** bypassed by `_prefill_planned` or the fallback driver.

---

## (C) Current prefill chunk size + delta-size distribution

**Current chunk size.**
- Engine base chunk: `DEFAULT_PREFILL_CHUNK = 512` (`dsv41/engine.py:121`); `self._chunk = self.prefill_chunk_size or 512` (`engine.py:348`).
- `prefill_chunk_size` comes from `_engine_kwargs_from_instance` (`builder.py:153-158`), which reads `EXO_PREFILL_STEP_SIZE`. **The launcher sets `EXO_PREFILL_STEP_SIZE=2048`** (`start_cluster.sh:88`) and forwards it unconditionally (`2019`). So the **effective deployed base chunk = 2048 rows** (`engine_prefill` `base = chunk` at `session.py:312`).
- The effective per-chunk size then **shrinks with context** under `choose_prefill_step` (`session.py:184-215`): `step = clamp(budget_bytes // (row_bytes·offset), floor=128, ·)`, with `budget = EXO_PREFILL_TRANSIENT_BUDGET_MB` (default **2048 MB**, `session.py:99`) and `row_bytes = _indexer_row_bytes()` = **1** under the shipped M2 default (`_HIER=1`, `block=8` → `ceil(8/8)=1`; `session.py:164-181`). So `step ≈ 2048e6 / offset`, i.e. 2048 rows until offset ≈ 10⁶, flooring at 128 (never reached in this 105 K-turn). No `world_size` divisor applies on the dsv41 path (in-loader TP; `load.py:73-82`).
- Long-chunk crossover: `long_threshold` is `None` for the engine path (`engine.py:356`), so the budget policy governs; `long_chunk` defaults equal `self._chunk` (`engine.py:349`).
- Separately, `DSV41_SPARSE_PREFILL_CHUNK` (`attention.py:51`) is a *sparse-attention internal* tile, not the session prefill chunk.

**delta_rows histogram (all 42 calls):**

| bucket | count | calls |
|---|---:|---|
| ≤ 256 | 2 | 2, 21 |
| 257 – 1024 | 13 | 3, 13, 17, 18, 19, 20, 23, 27, 28, 33, 34, 35, 42 |
| 1025 – 4096 | 23 | 4, 5, 8, 9, 10, 11, 12, 14, 15, 16, 22, 25, 26, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41 |
| 4097 – 16384 | 3 | 6, 7, 24 |
| > 16384 | 1 | 1 (cold) |

(Warm-only, calls 2–42: ≤256 → 2; 257–1024 → 13; 1025–4096 → 23; 4097–16384 → 3; >16384 → 0.)

---

## Bottom line

- **Hypothesis refuted:** over the 41 warm calls, `actual_prefill` never exceeds the genuinely-new tokens (`expected_delta = prompt[i]−prompt[i−1]`). Ratio max = **1.000**, median = 1.000; **0 misses**. The 32/42 “prefill-dominated” calls are prefill-dominated because the *client delta itself* is large relative to the short generations (new tool output dominates), **not** because of a cache miss or system-prompt churn.
- 5 warm calls **under**-feed vs the prompt delta (3, 8, 20, 21, 41); the `reuse undershoot` warning (39/42) is a seam-rewind artifact, not a miss signal.
- **Go-condition MET:** 41.3 % (warm) / 55.3 % (all) of Σ delta rows live in `delta_rows > 4096` calls; max warm single-call delta = 14 271 rows.
- **Missed entry point:** the image-span / `chunk_plan` path (`_prefill_planned`) pins the fixed-crossover branch and thereby **bypasses `choose_prefill_step` → `EXO_PREFILL_TRANSIENT_BUDGET_MB` is not honored there**, unlike every other prefill entry point.
