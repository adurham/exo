# PHASE 20 — M0 milestone (Phase 0 progress)

Campaign: pure throughput (prefill rows/s, decode tok/s). Cluster production `deploy/next13 @ f4bb14746`, TP2, never relaunched.
Started from a dead PM's scaffold; nothing was deployed before this PM either. Relaunch budget: **0 of 3 consumed**.

## Phase 0a — per-call decomposition of the real 42-call turn (BRIEF Z) — DONE
- Artifacts: `turn_decomp.csv` (42 rows), `turn_decomp_summary.json`, `turn_decomp.md`.
- `sum(latency)` = **1530.587 s** (expected 1530.59).
- `sum(prefill_s)` = **327.385 s**, `sum(decode_s)` = **1085.405 s**, model total 1412.79 s.
- **Gate 0a: ratio 0.9230 → within ±10% PASS.** Model-side unaccounted = **117.8 s** (whole cold call 1 = 98.8 s UNSPLIT + 17.1 s pre_s + 1.3 s post_s).
- Decode weighted mean **19.92 t/s** (median 21.66). `sum(gap_to_next)` = 283.858 s (expected 283.86).
- **Cache misses = 0 / 41** (call 1 cold, excluded) — reproduces the prior audit independently.
- `prompt == reuse + prefill` holds on all 41 reuse calls → **rows == tokens** for prefill.

## Phase 0b — prefix-cache integrity — CLOSED
0 misses in 0a → item closed; no client-side prompt mutation to chase.

## Phase 0c — delta-prefill ladder (BRIEF L) — IN PROGRESS (live)
Live chunk1 complete (canary healthy both nodes each run):

| ctx | delta | actual rows (med) | n | rows/s med | min | max |
|---|---|---|---|---|---|---|
| 20k | 2048 | 3866 | 3 | **272.6** | 272.3 | 273.2 |
| 50k | 2048 | 2501 | 3 | **249.6** | 249.2 | 251.5 |

20k→50k delta rows/s = **−8.4%** (needs 110k to test the PREREG >15% ctx-depth rule). Remaining live: chunk2a (110k cold), chunk2b (110k deltas), chunk3 (delta-size sweep 256/1024/4096/8192 at 50k), fresh100k reference.
`delta rows/s` is essentially flat across the ctx tested so far; the delta path (≈250 rows/s) sits well below the fresh-feed reference.

### Tool bugs found and fixed live (each was blocking or silently corrupting measurement)
1. **Guard S2 own-request race** (blocked every chunk): the watcher snapshotted the own-request list before the multi-second ssh log read, so the harness's *own* POST was flagged non-own → instant abort. Fixed: re-read own list after the log read.
2. **Guard S1 TP2 false positive** (blocked every chunk): `_state_active_tasks` counted `RunnerRunning` *runners*; on TP2 one request runs on BOTH ranks → active=2 → "user arrived". Fixed: count cluster-wide `TextGeneration` *tasks* (one task_id shared by both ranks).
3. **Guard registry not persistent** (stalled chunks 10 min): a chunk aborted/killed mid-run left its own POSTs unregistered for the next run's idle-check. Fixed: delta-ladder persists own requests to `raw/own_requests.jsonl` and loads them + passes them into ChunkGuard.
4. **Ladder delta reps not unique** → reps 2/3 were byte-identical to rep 1 and served from cache (prefill=0, measured nothing). Fixed: per-rep nonce in the delta text; cache hits (`prefill==0`) now detected and excluded.
5. **Ladder wall-cap double-count** (`elapsed + pred_cum`) falsely tripped NOT_RUN_WALL_CAP. Fixed: gate on true elapsed only.
6. **Collapsed threshold mis-calibrated**: the engine's 2048-row checkpoint ladder legitimately rewinds a base to the largest rung ≤ base end. Fixed: `collapsed` only when `reuse < base_rows − 2048`.
7. **mtrace `export` silently emitted an empty 65-byte result** on a `--xpath` no-match and reported success. Fixed: validate exported XML, list schemas, exit 2 on no-match. (Laptop + studio2 toy validated: recovered busy fraction within 0.04 of known duty cycle.)

## Phase 0d — GPU-busy fraction of the decode round (BRIEF P) — TOOL READY, LIVE WINDOWS PENDING
- `bench/phase20_gpu_busy.py` built, 24 tests. Idle fixtures parsed: m4-1 3.98% / m4-2 3.71% HW-active residency (elapsed-ms-weighted), P-state and frequency bins ordered. Live decode window (benign + agentic) not yet captured.
- Fallback B (`bench/phase20_mtrace.py`) validated end-to-end.

## Design forensics (feeds Phase 1-3)
- **R1** (`phase1_round_mechanics.md`): the "one sync per round" premise is only syntactically true — a compressing layer's `sparse_attention._column_boundary` does an **ungated host round-trip** (`int(mx.min(...))`, `.item()`) every body forward at default env (gate default mismatch between two modules). INFERRED 5–15 ms/round; the top Phase-3 lever. `spec.rollback`/`head.append_ctx` contain no syncs and are O(1)/O(window), not O(ctx).
- **R2** (`phase1_tp_comm_head_policy.md`): 44+γ collectives/round (or 84+γ if `attn.all_sum` fires). **Head replication = +3.44 GB/rank but busts the 115 GB wired guardrail** (needs a limit raise / ≥3.5 GB reclaim / experts-only). Draft-state cache is already per-rank. `GammaPolicy.update()` is never called on the serving path — Phase 2 must wire it; γ=5 needs code (candidate set is (1,2,3,4)). Per-round γ switching is zero-copy.

## Budget / status
- Relaunches used: **0 / 3**. Cluster untouched (production, idle, canary 14.8–14.86 TFLOPS both nodes).
- Next: finish 0c (chunk2a/2b/3, fresh100k) + 0d live window → M0 final; then relaunch #1 (`deploy/next16-instr`, built: merge + round_prof plumbing, 234 dsv41 tests green).
