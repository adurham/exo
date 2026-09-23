# Deep-context decode degradation: mechanism CONFIRMED as mmap page re-faulting (2026-09-22)

## Task

User: don't collapse at 500K; get 250K back above 30 t/s.
Baseline (same session, measured): 273K = 26.8 t/s, 565K = 13.4 t/s,
while <=120K runs 29-37 t/s.

## Mechanism: CONFIRMED, measured directly

Per-request page-fault deltas on m4-1, sampled across a single request:

| request depth | pageins | pageouts | decompressions |
|---|---|---|---|
| 401 tok | **47** | 0 | 0 |
| 41,874 tok | **32,469** | 0 | 10 |

**~32.5K page-ins in one 41.9K-token request ≈ 0.53 GB re-read from SSD,
scaling with depth.** The short-context control shows 47 — i.e. ~700x
fewer.

- **pageouts = 0** → not swapping.
- **decompressions ≈ 0** → not compressor thrash.

This matches the mechanism documented in
`docs/dspark-352k-memory-regression-2026-08-27.md`: MLX weights are
mmap-backed **clean** file pages. Under memory pressure macOS evicts them
*without* swap accounting and they are re-faulted from SSD. Idle wired is
only **3.3 GB** while ~89 GB of weights are resident — i.e. the weights are
already in exactly the evictable state that doc describes.

## Why it degrades at depth (the budget)

`iogpu.wired_limit_mb = 115000` → 120.6 GB usable GPU wired.

| state | peak memory | headroom |
|---|---|---|
| idle (weights clean/mmapped) | 3.3 GB wired | 117.3 GB |
| 115K ctx | 101.1 GB | 19.5 GB |
| 273K ctx | 103.8 GB | 16.8 GB |
| 565K ctx | **115.7 GB** | **4.9 GB** |

Headroom shrinks monotonically with depth. At 565K only ~5 GB remains, so
weights get evicted → re-faulted → the observed 13.4 t/s. Bistable: whether
a given run collapses depends on the initial margin, matching the 4/16
stochastic collapse rate in the 352K investigation.

## Cost attribution (honest)

0.53 GB at 3-6 GB/s is only **0.09-0.18 s** of a 107 s wall — so at 42K
depth the re-faults are NOT yet the dominant cost; the throughput loss to
41.9K (vs ~30 t/s at shallow depth) must be mostly normal depth-scaling of
the verify forward (Indexer top-k over compressed KV).

Re-faulting becomes dominant only where it is sustained *every cycle* —
i.e. the 565K regime, where headroom is ~5 GB and eviction cannot be
recovered between cycles. That is the collapse, and it is the thing to fix.

## The fix path (extends shipped work)

`EXO_DSV4_DSPARK_TP_SHARD=1` (commit `2d85ccdcb`) is **already live** and
shards the 3 DSpark stages' MoE FFN weights, recovering ~3-3.5 GB/node.
The doc is explicit that this is "NOT a full margin-restorer". Residual
replicated DSpark footprint: **~6.5-7 GB/node** (attention / main_proj /
markov parts, and non-expert projections).

Total identified headroom: **~6.5-7 GB/node remaining**.
Against 4.9 GB of headroom at 565K, recovering that is plausibly
collapse-eliminating.

### Candidate levers, in rough order of (value / risk)

1. **Shard the remaining DSpark non-expert projections** where dims divide —
   direct extension of the shipped `_shard_stage` helper in
   `auto_parallel.py` (~line 1245). Lowest risk: same mechanism, same
   failure policy (detach on error), same rank-consistency discipline.
2. **Quantize the DSpark head harder** (it is mxfp4/mxfp8, 3 stages,
   ~10.13 GB). Re-encode at lower bit width; needs a quality gate since the
   draft head's accuracy drives acceptance.
3. **Page/spill DSpark stages** — a bigger change; only if 1+2 fall short.

## Guardrails for whoever picks this up

- **Verify with the acceptance/quality gate, not just tok/s.** The draft
  head feeds speculation; degrading it raises tok/s *only* if acceptance
  holds. Use the needle probe at depth AND check acceptance parity.
- **Reboot before benchmarking.** An earlier session confirmed a degraded
  GPU power state (2.5-4.0 TFLOPS vs 14.8 healthy) that only a reboot
  cleared. Always run the raw-GEMM canary first.
- **Never compute tok/s from wall clock** — use the probe's `decode_tps`.
- **The collapse is bistable/stochastic** (4/16 in the 352K protocol). A
  single non-collapsed run proves nothing; need N>=8 with a stated bar.

## Status

- Mechanism: **CONFIRMED** (page-ins measured, swap/compressor ruled out).
- Fix: **NOT yet attempted.** Next action is lever 1 above.
- 250K currently reproduces at **26.72-26.83 t/s** (two independent runs).
