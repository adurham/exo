# c=2 100K Quality Bug — Discovery & Triage (2026-05-23 PM)

## TL;DR

**The "γ=2 production champion at 34.16 t/s σ=0.07 since 2026-05-22" has
never been quality-validated at c=2 100K. When validated today with a
c=2-aware probe, it produces complete gibberish (regurgitated instruction
fragments + `<｜begin▁of▁sentence｜>` spam).**

This is the broader bug that today's bistability hunt accidentally
surfaced.

* c=1 single-request at 100K: **perfect** — needle found in both `content`
  and `reasoning_content`, 0 BOS leaks, finish_reason=stop.
* c=2 (2 parallel requests) at 100K: **broken** at every configuration
  tested today (γ=2 K=0, γ=3 K=1, with and without the per-step fence
  patch). Both streams produce the same gibberish shape.

The fence patch (`1de63014`) is **regression-clean** but does NOT fix this
bug — it's a real concern in its own right (chain-collective queue
draining is the right c=2 analog of the c=1 fence at mtp_module.py:786)
but the actual quality break is upstream in the c=2 batched prefill /
attention / KV-cache machinery.

## What was tested

All on commit `1de63014` (origin/main HEAD after this session's three
diagnostic commits: `a1caaeb3` tracer, `fe837b49` probe v2, `1de63014`
fence patch).

| Config | Path | Quality |
|--------|------|---------|
| Short prompt c=1 ("Paris") | BS=1 / mtp_module.draft_tokens | ✓ `content='Paris'` |
| Short prompt c=2 ("Paris", 2 parallel) | dsv4_mtp._draft_tokens_batched | ✓ both `content='Paris'` (1 BOS prefix in reasoning, ignored as harmless) |
| 100K c=1 single-request | BS=1 / mtp_module.draft_tokens | ✓ `content='FALCON-MERCURY-7749'`, full coherent reasoning, 0 BOS |
| 100K c=2 γ=3 K=1 FENCE=4 (2 parallel) | _draft_tokens_batched | ✗ gibberish + BOS spam |
| **100K c=2 γ=2 K=0 FENCE=4 (production champion)** | _draft_tokens_batched | ✗ gibberish + BOS spam |

## What the c=2 100K gibberish looks like

Sample from production-champion config (γ=2 K=0 FENCE=4), stream 0:

```
content (3149 chars):
". Do not including the code:</think>. Do not including the code:</think>:</think>. Do not including the code:</think>. Do not including no other thanky. Do not including the code: "FALONE LINE. The code: FALONE LINE. Respond with no other. Respond with no other text. Do not including the code. Respond with no other words. Do not including the code. 1. Respond with no other words. code-NAME-NUMBER-NUMBER. Do not including the same line. 2025. 2025. The<｜begin▁of▁sentence｜><｜begin▁of▁sentence｜>x 127"

reasoning (17 chars): "We need the code:"
```

Both streams produced the same fingerprint. Three diagnostic patterns
in the content:

1. **Regurgitates the INSTRUCTION text fragments**: "Do not including
   the code" (model is hallucinating a misreading of "include"),
   "Respond with no other text", "CODE-NAME-NUMBER", "one line".
2. **Partial-token drift**: "FALWAYS", "FALONE", "FAL" prefix garbage
   — the model is trying to start "FALCON-MERCURY-7749" but failing
   to continue past 3 chars.
3. **BOS spam**: 127-129 `<｜begin▁of▁sentence｜>` tokens at the end,
   `finish_reason=length` (max_tokens reached on garbage).

The model is **attending to the instruction text but not to the document
body** where the needle lives. Fingerprint of **sparse-attention indexer
failure or KV-cache corruption at long context** in the c=2 batched
code path.

## Why this was missed for so long

`bench/quality_probe_dsv4.py` (pre-2026-05-23 PM, ~commit a4afc1) only
supported `concurrency=1`. It fired one streamed request. At c=1 the
runner routes through `mtp_module.draft_tokens` — the BS=1 path that
already has the working per-step fence at line 786, and that uses the
serial cache path which is fine at 100K. The probe was **structurally
incapable** of reaching `dsv4_mtp._draft_tokens_batched` (the c=2 path)
or the batched-prefill code path (`prefill_batched`).

Every "100K needle probe passed at γ=X K=Y" claim across 2026-05-21 →
2026-05-23 was a c=1 probe of a c=2 runtime. Quality at c=2 100K was
never measured.

Today's `quality_probe_dsv4.py` v2 (commit `fe837b49`):

* `--concurrency N` fires N parallel streamed requests — routes
  through `_draft_tokens_batched` at N≥2.
* `--iters M` repeats the concurrency-wide request.
* `detect_quality_issues()` counts BOS / EOS / role-tag substring leaks
  — any positive count = quality fail.

Unit-tested against today's actual bench outputs (clean iter 0 reasoning
= pass, BOS spam iter 1 = flagged with `BEGIN_OF_SENTENCE_LEAK:3`).

## Bug zone (where the corruption originates)

c=1 100K is clean. c=2 short-prompt is clean. The bug fires when
**c=2 AND long context** are both present. That isolates the
mechanism to code that's specific to **batched processing at scale**:

1. `prefill_batched()` in `src/exo/worker/engines/mlx/generator/generate.py`
   — TP-aware batched prefill. Processes both streams together at
   `(B, L_chunk)`. Uses `_merge_caches` and `BatchRotatingKVCache._update_concat`
   for each chunk.
2. `_upgrade_cache_to_per_stream()` swap from `BatchRotatingKVCache` →
   `PerStreamBatchRotatingKVCache` (mlx_lm cache.py:2256) AFTER prefill.
   The `PerStreamBatchRotatingKVCache` docstring explicitly says
   "convert to this class only AFTER prefill" because
   `finalize() / _lengths` flow isn't reimplemented.
3. `SparseCompressedAttention` (mlx-lm deepseek_v4.py:2056) — uses the
   `Indexer` to pick top-K context positions per stream from
   `pooled` (compressed KV). At c=2 the indexer's output is `(B, L, k)`
   — different per batch element. If the indexer produces wrong
   top-K for ANY batch element at long context, that stream attends
   to wrong tokens. **This is the most likely root cause** given the
   "regurgitates instruction, drops document body" fingerprint.
4. `dsv4_speculative_forward()` — the c=2 verify-forward.

## Investigation plan (next session)

### Step 1 — Localize: is the bug in PREFILL or in DECODE?

Test: bypass spec entirely (`EXO_SPECULATIVE=0 EXO_DSV4_MTP=0`) and
re-run the c=2 100K probe. If quality still breaks:
* The bug is in the c=2 batched code path that runs even without spec
  — narrows to `prefill_batched` / `BatchRotatingKVCache` / sparse-attn
  in non-spec mode.

If quality is FINE without spec:
* The bug is c=2-spec-specific — `_draft_tokens_batched` or
  `dsv4_speculative_forward` or the cache class-swap.

This is a single 5-min cluster restart + 10-min bench. Worth doing first.

### Step 2 — If bug is in non-spec c=2 batched: trace one layer

Add per-layer per-stream diagnostic in `SparseCompressedAttention`:
* Log `pooled.shape`, `topk.shape`, first 10 topk indices per stream
* Log `mask.shape` and any mismatch between streams
* Log `out` stream-0 vs stream-1 deltas at first / last / 100th token

Compare c=1-prefill (reference) vs c=2-batched-prefill side-by-side.
The first stream where they diverge is the bug zone.

### Step 3 — If bug is c=2-spec-specific: inspect cache class-swap

The `_upgrade_cache_to_per_stream` swap happens AT BS=N entry, but the
underlying `BatchRotatingKVCache` is what holds prefill state. Verify:
* After prefill, before class-swap, does the cache have correct per-stream
  state? Inspect `keys.shape`, `offset`, `_idx`, `_offset`.
* After class-swap, does `_per_stream_max` and `_offsets_py` get bootstrapped
  from the right base-class state? (mlx_lm/models/cache.py:2332-2335 has
  the lazy bootstrap path.)

### Step 4 — Don't ship the fence patch as a "fix"

Commit `1de63014` is regression-clean (c=1 100K still perfect, c=2 short
prompt still clean) and structurally correct (mirrors the c=1 fence).
But it does NOT fix the actual c=2 100K quality bug. Don't claim it does.

Options:
* Keep it in main as defensive depth (cheap, correct, doesn't hurt).
* Revert and reapply only when the real bug is fixed, to avoid mixing
  signals during the next bench.

User decision required.

## Don't-re-attempt list

From this session and pitfall #46:

* FENCE_EVERY_N_LAYERS=2 — made γ=3 K=1 worse, not better (this morning's
  prior session)
* Eagle K=2 — adds two coord-group collectives per chain step → bistable
  at c=2 even with γ=2 (prior session)
* `broadcast_from_canonical(soft_emb)` — 17× slowdown (commit `21ba40db`)
* The per-step `mx.eval(tok_arr)` fence in `_draft_tokens_batched` —
  shipped at `1de63014`, doesn't fix the actual c=2 100K bug. Don't
  re-ship it as the bistability fix without acknowledging this.
* `MLX_JACCL_ACK_SYNC_PRE=1` — alone makes c=2 worse (mlx mesh_impl.h:32-57)
* mlx eager-commit (`4d21baa2`), Event::signal, Mach RT, `mx.async_eval`
  swap of per-step fence — all previously falsified

## Diagnostic assets shipped this session

* `bench/quality_probe_dsv4.py` v2 (`fe837b49`): `--concurrency N`,
  `--iters M`, BOS/EOS/role-tag detection
* `dsv4_mtp.py` C2 tracer gated by `EXO_DSV4_C2_TRACE=1` (`a1caaeb3`):
  per-step per-stream JSONL trace at /tmp/dsv4_c2_trace_pid<PID>.jsonl
* `start_cluster.sh` env forwarder for `EXO_DSV4_C2_TRACE`
* This findings doc

## Skill pitfall updates needed

1. **Pitfall #41 / #46 amendment**: "100K needle probe passed" is NOT
   sufficient if it's a c=1 probe. The probe MUST fire `--concurrency
   ≥2 --iters ≥2` to reach the actual production c=2 path.
2. **New pitfall**: the c=2 100K quality bug exists at γ=2 K=0
   (production champion config). Every t/s number quoted for c=2 100K
   between 2026-05-21 and 2026-05-23 was likely BOS-spam-throughput,
   not real generation. Until this is fixed, c=2 100K is broken in prod.

## Current cluster state

* HEAD `1de63014` (fence patch, regression-clean)
* Cluster RUNNING at γ=2 K=0 FENCE=4 default (last test config)
* Last bench produced gibberish — don't leave running for the user to
  hit accidentally
