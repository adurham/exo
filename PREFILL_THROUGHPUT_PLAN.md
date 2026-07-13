# DSv4-Flash Prefill Throughput Plan — ≥300 tok/s everywhere, 350 steady-state (2026-07-13)

Status: PLANNED — nothing executed. Approval + idle cluster windows required per phase.
Scope: per-request prefill throughput at normal context (≤100K). The 340K+ cliff is a
separate problem — see PREFILL_CLIFF_HANDOFF.md. Indexer-window changes are OFF THE
TABLE (quality degradation, owner decision 2026-07-13).

## Problem statement (measured on current code, logs from 2026-07-12/13 sessions)

- Steady-state prefill ≈ 365 tok/s and is fully explained by
  `throughput = step_size / per_chunk_cost`: each 1024-token chunk costs a near-flat
  ~2.3–2.8s regardless of tokens-in-chunk or context depth. Implied within-chunk
  marginal rate ~1,900 tok/s — the math is fast; the per-chunk fixed work is not.
- Every prefill <1024 tokens pays one full chunk (~2.3s) + ~0.4s request overhead
  (0.32s tokenize/cache-lookup + 0.07s MTP cache prefill). That is the whole
  sub-300 story.
- Bucketed evidence (exo.log, both Studios, 2 days, n=200 prefills):

  | size bucket | n | avg rate (tok/s) |
  |---|---|---|
  | <512 | 53 | 131 |
  | 512–2K | 94 | 225 |
  | 2–8K | 27 | 310 |
  | 8–32K | 12 | 337 |
  | 32K+ | 14 | 325 |

- Depth is NOT the driver: small prefills at <20K ctx average 139 tok/s vs 194–210
  at 50K+. Fixed per-chunk cost dominates at every depth.
- 147 of 200 logged prefills are <2K tokens (the agentic-loop common case).
- Sample timeline (298-token continuation @ 71K ctx, 2026-07-13 01:21):
  received→cache-hit 0.32s, prefill() 2.03s (one chunk), MTP cache 0.07s.
- Chunk-cost arithmetic from the 1500-token trace (chunk1 1024 tok / 2.8s,
  chunk2 476 tok / 2.5s): fixed ≈ 2.3s/chunk, marginal ≈ 1,900 tok/s. Model
  predicts ~365 tok/s multi-chunk steady state — matches observation.

## Success criteria (fixed now so it can't drift)

- Metric: per-request effective prefill rate = uncached tokens ÷ prefill wall time.
- Targets: ≥300 at probe sizes 300 / 1K / 4K / 32K / 100K; ≥350 at ≥4K.
- Gates on every accepted config: needle pass (quality_probe_dsv4.py, 100K,
  FALCON-MERCURY-7749 protocol) AND decode within noise of current (~35 tok/s
  @ 30–70K ctx, c=1). Perf wins that lose quality are not wins.

## Phase 0 — Validate the baseline before trusting it

No restarts; two probe requests against an idle cluster.

1. **Needle-gate the RUNNING config.** The May sweep found
   `EXO_PREFILL_STEP_SIZE=512` and `64` QUALITY-BROKEN ("sweet spot is narrow:
   128 ✓, 256 ✓, 512 ✗, 64 ✗" — docs/fork-notes.md). Production now runs **1024**
   on the newer batched-prefill/seq-split path. If 1024 was never needle-gated on
   current code, confirm it FIRST. If it fails: roll back to 256/128, accept the
   throughput hit, re-baseline, and continue the plan from there.
2. Formal baselines via bench/quality_probe_dsv4.py + concurrent_bench.py (c=1,
   100K) plus per-size effective-rate probes at 300 / 1K / 4K / 32K / 100K
   uncached tokens. One JSONL row each, tagged `baseline-2026-07-13`.

## Phase 1 — Span-profile the 2.3s chunk (1 restart)

Restart with `EXO_PROFILER=spans EXO_PROFILER_SYNC_SPANS=1` (same rig as the cliff
investigation). Capture:

- one ~300-token prefill at ~70K ctx (the exact slow shape),
- one 4K prefill,
- one request immediately after `runner idle: reclaimed MLX allocator pool` vs one
  back-to-back (quantifies the cold-allocator penalty),
- what the 0.32s tokenize/lookup actually is (full-prompt retokenization? hashing?).

Output: ranked table splitting ~2.3s into indexer / sdpa / MoE / allreduce /
fence-sync / launch-eval. **Decision gate:** the dominant bucket picks which Phase 2
lever runs first. No blind sweeps.

## Phase 2 — Quality-gated knob sweeps (one variable at a time)

Protocol per experiment (the established one, ~/.hermes/scripts/run_exp.sh):
teardown both Studios → start_cluster.sh with baseline env + ONE override →
READY (2/2) → needle gate (skip bench on fail) → bench → append
/tmp/exp_results.jsonl.

| # | Lever | Test | Expected win | Risk / notes |
|---|---|---|---|---|
| 2a | `EXO_PREFILL_STEP_SIZE` | 1024 (control) vs 2048 vs 4096 | If chunk cost stays near-flat: ~600+ tok/s at 2048 | QUALITY (512 was broken in May — gate mandatory); activation memory at 100K; GPU queue depth / WindowServer watchdog (`EXO_MAX_ACTIVE_TASKS=5` exists for a reason) |
| 2b | `EXO_DSV4_FENCE_EVERY_N_LAYERS` | 4 (current) vs 8 vs 16 | May data: decode +13% at 16; prefill effect unmeasured — 43 layers × sync is a plausible chunk-cost component | Establish why prod is at 4; check interplay with `EXO_DSV4_FENCE_ASYNC=1` + jaccl reliability flags before sweeping |
| 2c | `MLX_MAX_OPS_PER_BUFFER` / `MLX_MAX_MB_PER_BUFFER` | 200 (current) vs 400 vs 800 | Fewer command-buffer submits per chunk | Same watchdog caveat |
| 2d | Idle allocator-pool reclaim | Skip/delay reclaim for N s after last request | Removes cold-alloc penalty on the next small prefill | Likely small code change, not env; memory held longer |
| 2e | Per-request trims | Incremental tokenization / prompt-hash cache; skip MTP cache prefill below a token threshold | ~0.3–0.4s off every request — biggest mover for the <1024 bucket after 2a | Code changes in the prefill submit path |

## Phase 3 — Only if Phase 2 lands short of 350

Code-level, each with its own needle + decode gate: overlap indexer with MoE within
a chunk; batch fence signaling; pipeline chunk N+1 tokenization under chunk N
compute. These ride the mlx-lm submodule pin dance (commit submodule → bump gitlink
→ start_cluster reinstall) — slow iteration, so last.

## Logistics & guardrails

- Every phase needs the cluster free. Owner schedules windows; NO restarts while a
  session is live.
- One variable per experiment. Accepted config = needle pass + decode within noise
  + prefill win.
- start_cluster.sh env baseline recorded before anything changes; rollback is one
  revert.
- New `EXO_*` flags need explicit passthrough lines in start_cluster.sh or they
  silently no-op (PREFILL_CLIFF_HANDOFF.md warning).
- Results: /tmp/exp_results.jsonl during the run; fork-notes.md entry at the end.

## Expected outcome

- Realistic: 2a alone reaches ~500–600 steady-state if chunk cost is flat at 2048
  and quality holds; 2d+2e move the small-continuation bucket from ~150 to 300+.
- Honest caveat: if the needle gate kills 2048 the way it killed 512, steady-state
  350 must come from 2b/2c/Phase 3, and the small-prefill bucket becomes the main
  battleground.
