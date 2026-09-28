# Phase 5 — Sizing Plan B, and the per-rank budget under real pipeline sharding

Date: 2026-09-28
Host: macstudio-m4-1 (measurement) + hermes-gw-01 (analysis)
Status: Plan B sized with real data; a stale claim in the phase-2 doc corrected

Phase 2's recommended next step #1 was: *"Size Plan B before choosing it —
instrument per-token top-6 expert indices, compute LRU hit rate and cold-miss
bytes/token at both budgets, then compare against the 6.5 GB/s sustained SSD
read."* This phase does that, and in doing so corrects a sharding assumption
that materially changes the answer.

## Bottom line

1. **exo shards by PIPELINE, not by expert parallelism.** At TP=2 each rank
   owns **20 of the 40 layers and ALL 384 experts within them**. Every
   per-rank budget number in phase 2 that divided experts across ranks is
   therefore wrong by ~2x.
2. **Native (mxfp4) experts do not fit, and cannot:** 20 x 7.219 = **144.38 GB**
   against a **115 GB** wired limit. Streaming is mandatory for Plan B.
3. **EXL3's experts DO fit:** **97.16-98.86 GB** per rank, leaving
   **9.0-10.7 GB** for attention/dense/embed/MTP/KV. EXL3 is the only format of
   the two that fits without streaming — which reframes the whole comparison.
4. **Plan B's SSD ceiling is ~40 tok/s, not dead.** On a real routing trace the
   steady-state LRU hit rate at a 252-experts/layer budget is **92.8%**, giving
   **162 MB/token** cold and a **~40 tok/s** ceiling — *above* the 25 tok/s bar.
   Plan B is viable on I/O; it is not viable on *memory*, and that is the
   finding that matters.
5. **The phase-2 doc's "this retuning has not been attempted" is FALSE.** The
   v3 chunked-prologue retune was built and measured later the same day:
   verify R=4 went **12.25 -> 3.17 ms (3.86x)** and R=8 **19.08 -> 6.05 ms
   (3.15x)**. Corrected in that doc as of this phase.

## 1. Sharding model (read from exo source, not assumed)

`auto_parallel.py` / `utils_mlx.py` shard with
`PipelineShardMetadata(start_layer, end_layer, device_rank, world_size)` — a
**pipeline** split. There is no expert-parallel split: a rank that owns a layer
owns all of that layer's experts.

```
per rank (TP=2) = 20 layers x per-layer expert bytes
```

This is the correction that flipped the earlier read of the data.

## 2. Per-rank expert footprint

| format | per-layer expert bytes | per rank (20 layers) | vs 115 GB wired |
|---|---|---|---|
| native mxfp4 | 7.219 GB | **144.38 GB** | **+29.38 GB — DOES NOT FIT** |
| EXL3 2.9bpw (layers 0-17, 23-39) | 5.113 GB | 98.86 GB (layers 0-19) | fits |
| EXL3 2.9bpw (layers 18-22) | 3.414 GB | 97.16 GB (layers 20-39) | fits |

Two independent routes agree on the EXL3 total: tiered layer sum above, and the
whole-directory figure `210.60 / 2 = 105.30 GB/rank` (which lands at 104.3-106.0
GB once the tiers are summed with the 7.18 GB/rank of non-expert bytes).

**EXL3 headroom: 9.0-10.7 GB** for attention + dense + embed + MTP + KV.

### The 2-bit tier is real, and it is exactly layers 18-22

Measured across **all 40** EXL3 layers (`p8f_exl3_layer_hist.py`):

```
5.113 GB : 35 layers [0..17, 23..39]
3.414 GB :  5 layers [18, 19, 20, 21, 22]      <- 33.2% smaller
```

Confirmed independently of phase 1's claim. Note the consequence for
extrapolation: a flat `40 x layer-0` estimate gives 204.53 GB against an exact
196.03 GB — it **overstates by 8.49 GB**. The native checkpoint has only layers
0-3 locally, so any native `x40` extrapolation carries the same systematic
error and should be treated as an upper bound.

## 3. Routing trace: how concentrated is expert selection?

`p8c_trace_dump.py` hooks every `Gate` in the real port and records the actual
top-6 selections (the checkpoint's own gate weights and
`e_score_correction_bias`) while decoding 256 tokens from a 490-token
multi-domain prompt (code, technical prose, math, list, chat, SQL, narrative,
numeric). `p8e_residency_sim2.py` then simulates per-layer LRU caches offline.

| budget/layer | cap | oracle | LRU cold | LRU steady |
|---|---|---|---|---|
| 24.1G | 64 | 66.1% | 76.6% | 76.9% |
| 36.1G | 96 | 76.3% | 82.3% | 82.7% |
| 48.1G | 128 | 83.5% | 85.5% | 85.9% |
| 60.2G | 160 | 89.0% | 87.9% | 88.4% |
| 72.2G | 192 | 92.9% | 89.7% | 90.2% |
| 84.2G | 224 | 95.7% | 91.3% | 91.9% |
| 96.3G | 256 | 97.7% | 92.2% | **92.8%** |
| 120.3G | 320 | 99.6% | 93.1% | 93.7% |
| 144.4G | 384 | 100.0% | 93.2% | 93.8% |

Two structural notes:

- **The LRU beats the oracle at small caps.** That is not a paradox: eviction is
  *temporal* (recently used experts survive), which tracks this workload's
  burstiness better than a frequency ranking chosen in hindsight. It means the
  realistic column is the one to plan against, and it is not worse than the
  idealised one.
- **Routing is diffuse, not Zipf-extreme.** Over a longer trace, layer 0 touches
  **361 of 384** experts and the top-10 share is only 12.6%. The hit rates above
  are therefore an **optimistic ceiling** for a full-model, multi-day workload:
  more tokens can only raise distinct-expert counts, which lowers hit rate.

### Cold-miss bytes/token and the SSD ceiling

Steady-state LRU, 20 layers/rank x 6 picks = 120 picks/token:

| cap | hit | miss/tok | native MB/tok | native tok/s | EXL3 MB/tok | EXL3 tok/s |
|---|---|---|---|---|---|---|
| 128 | 85.9% | 16.9 | 317.6 | 20.5 | 225.1 | 28.9 |
| 160 | 88.4% | 13.9 | 260.7 | 24.9 | 184.7 | 35.2 |
| 192 | 90.2% | 11.7 | 220.5 | 29.5 | 156.3 | 41.6 |
| 256 | 92.8% | 8.7 | 162.7 | **39.9** | 115.3 | 56.4 |
| 384 | 93.8% | 7.4 | 139.9 | 46.5 | 99.1 | 65.6 |

**Plan B at a 95 GB expert budget** (252 experts/layer, 92.8% hit) gives
8.6 cold picks/token = **162.4 MB/token** = **~40 tok/s** — above the 25 tok/s
bar. Plan B dies on **memory** (144.38 GB needed vs 115 GB wired), not on I/O.

## 4. The v3 retune DID happen — phase 2's doc is stale

Phase 2 closed with "**This retuning has not been attempted.**" That was true
when written and is now false. The v3 chunked-prologue kernel
(`~/exl3-moe-v3-chunked-prologue.patch`, applied to
`PonyExl3/ponyexl3/mlx/exl3_moe.py`) was built and measured at the real shape
E=384, D=5120, H=2304, with `_v2_ok()=True` (i.e. the pathological `_prefill`
fallthrough is gone):

| shape | v2 ms | v2 ratio | v3 ms | v3 ratio | speedup |
|---|---|---|---|---|---|
| decode R=1 | 1.305 | 3.38 | 1.029 | **2.65** | 1.27x |
| verify R=4 (DSpark) | 12.251 | 11.95 | 3.171 | **3.04** | **3.86x** |
| verify R=8 | 19.079 | 9.78 | 6.050 | **3.11** | 3.15x |

The verify hole that dominated phase 2 — the 9.5-12x — is **closed to ~3.0x**.
The remaining miss is a ~2.65-3.1x decode/verify gap against production MXFP4,
and prefill R=512 is 3.10x (worse than v2's 1.98x at E=128, but that comparison
is not like-for-like: E=384 is the real expert count and costs more at prefill
in *both* formats).

The phase-2 README has been corrected in place to state this.

## 5. What this changes about the decision

- **EXL3 is no longer only a memory play with a broken speed story.** Its
  experts are the only ones that fit resident (97-99 GB vs 144 GB), and its
  verify cost is now ~3.0x rather than ~12x.
- **Plan B is not "dead by construction."** Its I/O ceiling is ~40 tok/s, above
  the bar. If Plan B is rejected, it must be on the resident-memory argument
  (144 GB > 115 GB ⇒ mandatory streaming ⇒ a streamed-expert decode path that
  does not exist yet), not on the SSD.
- **The honest comparison is now two *unbuilt* paths**, not one built path
  against one broken path: Plan B needs a streamed-expert MoE implementation;
  EXL3 needs its kernels ported into exo. Neither exists today.
- **Still unmeasured:** real MTP/DSpark acceptance rate (needs a production
  relaunch with `EXO_DSV4_MTP_LOG_INTERVAL=64`; see phase 4).

## Artifacts

- `scripts/p8a_expert_bytes.py` — exact expert byte accounting from
  safetensors headers (native + EXL3)
- `scripts/p8b_routing_trace.py` — first-pass routing trace + concentration
- `scripts/p8c_trace_dump.py` — full trace to JSON for offline simulation
- `scripts/p8d_residency_sim.py` — first simulator (superseded; kept as the
  worked example of the wrong sharding assumption)
- `scripts/p8e_residency_sim2.py` — corrected simulator (pipeline sharding,
  cold vs steady LRU)
- `scripts/p8f_exl3_layer_hist.py` — per-layer histogram proving the 18-22 tier
- `raw/p8a.out` .. `raw/p8f.out` — captured runs

## Corrections carried into this phase

| claim | where it was wrong | now |
|---|---|---|
| experts split across ranks at TP=2 | phase 2 §"residence" reasoning | pipeline sharding: 20 layers x all 384 experts/rank |
| "retuning has not been attempted" | phase 2 §"Cross-check" | v3 measured: R=4 3.86x faster, 12.25 -> 3.17 ms |
| EXL3 = 40 x 5.113 GB uniform | implicit in flat extrapolation | tiered: 35 x 5.113 + 5 x 3.414 = 196.03 GB exact |
