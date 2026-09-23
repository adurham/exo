# TARGET (b) SYNTHESIS — my attribution was directionally right but I stopped one level too shallow (2026-09-23)

## What changed

Last night I localized the 250K cycle cost to `attn.sdpa.compressed` (369.96 µs,
5.8% of decode wall) and framed that as "the target" for target (b). A
source-reading pass over that exact path (read-only, no cluster contact) plus the
repo's own August measurement (`docs/p3-worker-c-attn-kernel-walltime-2026-08-23.md`)
shows that framing was **one level too shallow**, in two specific ways.

## Correction 1: the model-level overhead IS real, but it is bounded to <0.5%

The reader found genuine, unconditional per-step waste in the compressed path:

- `deepseek_v4.py:5159` — `kv = mx.concatenate([kv, pooled[:, None]], axis=2)`
  runs **every decode step, in all 20 ratio-128 layers**, with no size or
  step-invariant guard. It is a pure copy on the context-scaling path: ~85 MB/step
  at 250K across 20 layers (vs ~15 MB/step at 30K). The repo already contains the
  cheap idiom it should use (pre-allocated storage + slice-assign, cache.py:1474-1483).
- `deepseek_v4.py:1867/1873` — `_extend_mask` manufactures an **all-True** mask
  with `mx.ones` when `pooled_mask is None`, then concatenates it. The pooled mask
  is None at decode **by construction** (cache.py:1582-1583 for PoolingCache,
  2238-2241 for BatchPoolingCache). So a mask is passed where no masking is
  needed, forcing the bool-mask kernel specialization and a per-key mask test over
  the whole pooled range. In-repo measurement of exactly this at these shapes:
  mask=None 20,255.7 µs vs all-True mask 21,408.6 µs — a 5.4% delta attributed to
  "the fixed cost of reading/materializing the mask, NOT work skipping."
- The enabling asymmetry: `RotatingKVCache.make_mask` returns None at N==1
  (cache.py:894-908), but the batch classes serving actually uses
  (`BatchRotatingKVCache`, `PerStreamBatchRotatingKVCache`) have no such path and
  always build a rolled bool array. The model source documents this consequence
  itself at 2205-2216.

**But the honest bound:** these all live inside the `attn.mask` span, and in the
sync-mode table `attn.mask` is **15.79 µs median / 0.3%** — of the same order as
`attn.kv_cache` (16.96 µs). So the entire model-level materialization cost on
this path is **under ~0.5% of the 69.5 ms cycle** and *cannot* account for the
+22.9% growth. Real waste, wrong size class.

## Correction 2: the compressed-SDPA span is at a structural ceiling, not a fixable target

The reader also diagnosed where the span's time actually goes, and the repo
already measured the same thing in August:

- `sdpa_vector_2pass_1_gqa` (the duplication-free variant — reads each K/V byte
  once instead of G/`HPT` times) is **unreachable for DSv4 for four independent
  reasons**: wrong `gqa_factor` (32, gate wants 8/12/16), wrong `head_dim` (512,
  gate wants 64/128), no matching Metal instantiation (only (64,64,8,8),
  (128,128,8,4), (128,128,12,4), (128,128,16,2) exist), and it additionally
  requires `!mask && !sinks` — both always false here (the mask is materialized
  per corrections above; `attn_sink` is a loaded parameter).
- The August doc's achieved-bandwidth table is the decisive datum:
  **sdpa-compressed runs at 47.6 GB/s against a 404.7 GB/s measured streaming
  ceiling** — i.e. 8-12% of streaming bandwidth, latency/occupancy-bound.
- And critically: *"the top-k and compressed-SDPA kernels are latency/occupancy-
  bound... they have headroom in principle — but they are small in absolute
  terms (+0.088 and +0.261 ms over the range), so optimizing them cannot recover
  multiple ms."*

So `attn.sdpa.compressed` being 5.8% of wall does **not** mean it is the fixable
cause of the depth gap. Its per-depth GROWTH is small; it is a large-but-flat
term.

## The real remaining question — and it is already named in the repo

The August doc's "honest gap" section is the most useful thing I have:

| | 100,026 | 352,599 | Δ |
|---|---|---|---|
| live total per token | 35.79 ms | 42.59 ms | **+6.80** |
| attention path only (bench) | 16.57 ms | 19.13 ms | **+2.56** |
| **residual (non-attention)** | 19.22 ms | 23.46 ms | **+4.24** |

Consistency check against tonight: +6.80 ms/token x 1.896 tok/cycle ≈ 12.9
ms/cycle vs my measured +13.0 ms/cycle. **Same order, but different depth ranges
(100K→352K vs 30K→279K) and his 30K endpoint isn't in the table — so this is a
consistency check, NOT an agreement. I am not claiming they match.**

That residual is outside anything the attention bench measured. Four candidates
are listed, none tested, and the doc flags **#4 as "the single most testable
follow-up"**: *pool-write donation intermittently failing in production*, worth
"up to +6.35 ms/token over 100K→352.6K for the compressor pool alone."

Candidate #1 is **MoE all_sum arrival skew**: the collective's payload is fixed
(1,1,4096) and cannot grow with L, but if one rank's attention runs slower at
depth the collective waits, and 43 calls/token amplifies small per-layer skew.

Tonight's data bears on #1 only weakly: `moe.all_sum` median went 79.12 → 93.08 µs
(+17.6%) from 30K to 250K. If that applied to all 43 layers it would be ~1.14
ms/cycle ≈ **9% of the +13.0 ms/cycle** — right order of magnitude but a
*single-rank span median cannot distinguish "waiting for the peer" from "doing
work"*, and the profiler does not establish that every layer pays the median.
**Wrong instrument, not a measurement.**

## What this means for the user's ask (b)

The honest position is now sharper and less flattering to my own framing:

- It is **not** "fix the indexer" (falsified last night, 0.0% of wall).
- It is **not** "fix the compressed-SDPA kernel" — that path is at 8-12% of
  streaming bandwidth with only +0.261 ms of depth growth; the model-level waste
  around it is real but bounded under ~0.5% of the cycle.
- The measured +13.0 ms/cycle lives largely in a **non-attention residual** the
  repo already identified in August and never chased: all_sum arrival skew and/or
  intermittent pool-write donation failure. 45-46% of the live budget is
  attention; the rest is where the depth term has room to hide.

## Concrete next steps (ranked, none of them a knob sweep)

1. **Pool-write donation failure** (the doc's own #1 follow-up): instrument or
   count donation failures per step at depth, or capture a per-call DISTRIBUTION
   of `attn.compressor` rather than median/max. A median cannot detect a
   fraction-of-steps +6 ms cost. Note tonight's `attn.compressor` medians are
   similar at 30K and 250K (99.25 vs 117.58 µs), which is *not* evidence of
   absence — it is the wrong statistic for an intermittent effect.
2. **Cross-rank all_sum skew**: compare both nodes' `moe.all_sum` timing for the
   same request (not one rank's median). This is the instrument the span profiler
   cannot provide.
3. **The bounded model-level fixes** (the `mx.concatenate` at 5159 and the
   all-True mask at 1867/1873) are worth doing — they are pure waste with an
   existing in-repo idiom, and the mask fix may be c=1-scoped only. But they must
   be presented as hygiene, with the <0.5% bound stated, **not** as the answer to
   target (b).

## Verification status

The reader's eight load-bearing source claims are being independently
re-verified against the source (a separate read-only pass, since a child's
summary is a self-report and not fact). This document will be amended if any
claim fails.
