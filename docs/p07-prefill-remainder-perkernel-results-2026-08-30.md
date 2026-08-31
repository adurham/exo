# P07 results — per-kernel GPU capture of prefill's non-GEMM remainder (2026-08-30)

Companion to `docs/p07-prefill-remainder-perkernel-preregister-2026-08-30.md`
(gates registered before measurement). Raw artifacts: `tmp/p07-20260830/`
(`results.json`, `results_topup.json`, `results_topk_ab.json`, harnesses, node logs).

**Cluster untouched throughout.** All captures ran as standalone processes beside
the live runner (p01/P03-proven recipe). PID 59909 (m4-1) / 60392 (m4-2),
`Sun Aug 30 19:34:04 2026`, verified identical before and after every run. Zero
relaunches. All verbon3 production flags verified live on both nodes.

## 1. Headline

The remaining prefill non-GEMM remainder is **not hiding an async-fence-class
bug**, and T10's "legitimate real compute near its own ceilings" conclusion
survives per-kernel scrutiny for every op **except one**: the indexer's top-k.

**`attn.indexer` top-k (`argpartition`) is ~2.9% of prefill wall time and runs
at 16-30 GB/s effective — roughly 4-7% of the 424 GB/s achievable streaming
bandwidth.** It is the single largest identified inefficiency remaining in the
remainder. It is real, it is measured (not inferred), and it is NOT fixable by
any existing knob — see §4.

Also settled: **~5.3pp of T7's original 19.3% was genuinely fixable and is
already fixed and shipped** (hc_expand 08-24, hc_collapse 08-25). The remainder
today is ~14.0%, not 19.3%.

## 2. Measured table (prefill shape, L_band=1024, ctx=220K → P=55000, m4-1)

Denominator chosen per the pre-registered intensity rule (<10 FLOP/byte →
bandwidth; ≥10 → FLOPS). FLOPS ceiling 14.34 TF (see §5 on the ceiling correction).

| op | µs/call | disp | intensity | ceiling | % achieved | verdict |
|---|---|---|---|---|---|---|
| `attn.indexer.score_gemm` (folded) | 1013.0 | 13 | 113.5 F/B | FLOPS | **99.3%** | AT CEILING |
| `attn.indexer.topk` (argpartition) | 15406.3 | 13 | <1 F/B | BW | **~4-7%** | **REAL HEADROOM** |
| `attn.indexer.pmask.apply` | 561.2 | 5 | ~1 F/B | BW | 47.3% | MODEST |
| `moe.shared_experts` (mxfp8 MLP) | 2062.0 | 4 | 421 F/B | FLOPS | 87.2% | AT CEILING |
| `moe.post_combine` (elementwise) | 359.4 | 3 | 0.78 F/B | BW | 49.6% | MODEST |
| `moe.gate.routed` | 299.2 | 7 | 170 F/B | FLOPS | 50.1% | MODEST |
| `moe.gate.hash` | 273.3 | 6 | 85 F/B | FLOPS | 54.8% | MODEST |
| `tail.rmsnorm` | 29.0 | 1 | 0.25 F/B | BW | >100%* | LATENCY FLOOR |
| `tail.rope.q` | 269.9 | 1 | — | BW | >100%* | LATENCY FLOOR |
| `tail.rope.q_idx` | 66.7 | 1 | — | BW | >100%* | LATENCY FLOOR |
| `tail.kv_cache.write` | 4.7 | 2 | — | BW | >100%* | LATENCY FLOOR |

\* **These four exceed 100% of the bandwidth ceiling, which is physically
impossible for real DRAM traffic — published with that caveat, not as real
bandwidth.** Cause is byte-model over-count, not a hardware violation: these
tensors are freshly written by the immediately-preceding op (`hc_expand` for
rmsnorm's `x`, `wq_b` for rope's `q`) and are therefore L2-resident, so the
modeled full DRAM read+write never happens. `kv_cache.write` (4.7µs, 2
dispatches) is additionally at GPU-timer granularity. All four are sub-ms/chunk;
the LATENCY FLOOR verdict holds regardless of which explanation dominates.

**Two headline claims CONFIRMED that were previously only inferred:**
- `indexer.score_gemm` at **99.3%** of ceiling — the OPT-6 folded GEMM really is
  at hardware peak. `indexer-prefill-decomposition-2026-08-24.md:254` explicitly
  flagged this as "inferred, not directly measured." Now measured. Confirmed.
- `moe.shared_experts` at **87.2%** — consistent with T10's 1.05x-of-peak
  estimate, reached by a completely different instrument.

## 3. Why the span profiler could never have found the top-k cost

The pre-registration predicted this and it held. `indexer.topk`'s span reads
**5.81 µs/call** (`dsv4-220k-prefill-span-profile-2026-08-18.md:112`) for an op
that really costs **7.7-15.4 ms**. That is a ~1300-2600x under-read, because
under MLX's lazy executor a span timer not terminated by an eval barrier measures
*graph construction*, not GPU work — real compute is charged to whichever later
span forces the sync. `indexer-prefill-decomposition-2026-08-24.md:116-121`
documents this exact effect and derives the same tell (a 14.42 GFLOP GEMM in
8.34µs would be 1.7 PFLOPS).

**This is the methodological point of the phase**: every prior "at ceiling"
verdict inside this remainder rested on an instrument that is structurally blind
to it. Per-kernel GPU capture is what resolves it.

## 4. The 15.41 ms "impossible" number — reconciled, then re-adjudicated

The capture's 15406 µs/call top-k initially looked impossible: it exceeds the
entire parent `attn.indexer` span (24,596.71 ms / 2310 calls = **10.65 ms/call**).
A sub-op cannot exceed its parent.

**Resolution — the two numbers are measured at different shapes, and reconcile
exactly once that is corrected.** The span figure is a run-AVERAGE over 110
chunks whose context grows 0 → 220K, so mean pool length is ~P_final/2. The
capture measured at FINAL-chunk shape (P=55000). Top-k is ~linear in P (measured:
0.20 → 0.31 µs per unit P across a 25x sweep). Run-averaged:

```
topk(P=55000)            = 15.40 ms/call   (final chunk, as captured)
topk run-average         =  7.70 ms/call   (mean P = P_final/2)
  as share of the 10.65 ms/call attn.indexer span   = 72.3%   <- fits
attn.indexer = 4.0% of prefill wall
  => topk = 4.0% x 72.3%                            = ~2.9% of prefill wall
```

72.3% of its parent is a large but entirely coherent share, and it is consistent
with the rest of the indexer being the score GEMM (at ceiling) plus small ops.

**A review pass wrongly called this an artifact** on the grounds that
`argpartition` is "default off," reading only the code's
`os.environ.get("EXO_DSV4_PREFILL_ARGPARTITION", "0")` fallback default. That is
wrong: **`start_cluster.sh:553` sets `: "${EXO_DSV4_PREFILL_ARGPARTITION:=1}"`
— default ON — and the live runner's env confirms it.** Production really does
run argpartition at prefill. Finding recorded here because the failure mode is
reusable: *a code-level env default is not the production default; check the
launcher and the live process env.*

## 5. Is production's default-ON argpartition a pessimization? NO — measured.

The obvious cheap fix would be flipping to the `argsort` fallback, whose code
comment claims it is "~5% faster on Metal." Tested directly (isolated GPU-timed
A/B, bf16, k=512, L_band=1024, distinct-input rotation banks, median of 5,
`mx.eval` per call; `tmp/p07-20260830/results_topk_ab.json`):

| P | ~ctx | argpartition µs | argsort µs | ratio | dispatches | top-k sets equal |
|---|---|---|---|---|---|---|
| 5,000 | 20K | 1003.1 | 1003.0 | 1.0002 | 7 | YES |
| 12,500 | 50K | 2871.6 | 2873.4 | 0.9994 | 9 | YES |
| 25,000 | 100K | 6336.8 | 6337.2 | 1.0000 | 11 | YES |
| 55,000 | 220K | 15396.0 | 15402.5 | 0.9996 | 13 | YES |
| 125,000 | 500K | 39149.3 | 39174.1 | 0.9994 | 15 | YES |

**Dead heat at every production P (ratio 1.000 ± 0.001), with identical dispatch
counts** — the tell that both expressions lower to the *same* radix-sort kernel
family in this MLX build (0.32.1.dev). Set-equality of selected indices holds at
all P including forced-tie inputs, so the equivalence claim in the code comment
is sound.

Consequences:
- **`EXO_DSV4_PREFILL_ARGPARTITION` / `EXO_DSV4_ARGPARTITION_MIN_P` are not
  levers.** Nothing to win by flipping either. No config change recommended.
- The historical "295→163 tok/s at P=500" argpartition collapse **does not
  reproduce** on the current MLX build (divergence only appears at P≤2000, +6.1%,
  on L2-resident inputs). The `MIN_P` gate is harmless but today unnecessary.
- The cost is therefore **structural to MLX's sort kernel**, not a wrong-branch
  bug. `mx.sort` (values only, no index gather) is only ~3% faster, so the index
  side-channel isn't the problem either.

## 6. Verdict against the pre-registered gates

| Span | Verdict | Basis |
|---|---|---|
| `indexer.score_gemm` | **AT CEILING** (99.3%) | measured, confirms prior inference |
| `moe.shared_experts` | **AT CEILING** (87.2%) | measured |
| `indexer.topk` | **REAL HEADROOM** (~4-7% of BW, ~2.9% of prefill wall) | measured; not knob-fixable |
| `moe.gate`, `post_combine`, `pmask` | **MODEST** (47-55%) | each <1.0% e2e → below ship gate |
| norms / rope / kv_cache | **DISPATCH/LATENCY FLOOR** | 1-2 dispatches, sub-ms/chunk |

So the answer to the phase's question is **both, in known proportion**:
- ~5.3pp of T7's original 19.3% = **real fixable bottleneck, already shipped**.
- The small ops (norms/rope/kv_cache) = **genuine dispatch/latency floor**,
  matching decode's own ~1.9ms/token floor. Confirmed, not assumed.
- **One real, open, non-floor item remains: indexer top-k at ~2.9% of prefill
  wall**, running at a few percent of achievable bandwidth.

## 7. Open item and next step (top-k)

**Status: OPEN, not fixed.** It clears the ≥1.0% e2e ship-gate threshold on size
(~2.9%), but does NOT have a shippable fix today, and per the pre-registered ship
gate a fix must reduce *dispatch count* and survive a *pipelined* microbench —
neither is satisfied by anything currently available:
- Both existing branches are the same kernel (§5). No config change helps.
- A custom Metal top-k (threshold-select / blocked partial-sort exploiting
  k=512 ≪ P=55000) is the only remaining path. MLX's kernel is a full radix
  sort of all P elements to extract the top 512 — asymptotically far more work
  than needed.
- **Do not treat this as a quick win.** This repo's history is emphatic that
  hand-rolled kernels which merely re-implement an MLX primitive lose once
  pipelined (whole-indexer fused kernel measured 0.54x = *slower*; wq_a+wkv
  fusion -0.48%). A top-k rewrite only clears the bar if it reduces real work,
  not just dispatch bookkeeping — and it must be validated pipelined, in situ,
  not per-call in isolation.
- Realistic ceiling if it were perfect: ~2.9% of prefill wall, i.e. **~1.03x
  prefill** — worth a scoped spike, not a campaign.

## 8. Secondary: SDPA ceiling denominators (T7's never-checked flag)

Audited by code reading + arithmetic (`bench/attn_production_class_bench.py`
+ real call sites). Full detail in the pre-registration §4:
- **MLA absorption: not in play.** DSv4-Flash has no `kv_lora_rank`; KV projects
  directly `hidden(4096)→head_dim(512)`. Naive D=512 formula correct.
- **`attn.sdpa` (sparse): denominator CORRECT, 61.7% stands** —
  `_sparse_pooled_attention_inner` does a dense local matmul over all 2175 keys
  and masks *after*, so masked positions genuinely are computed.
- **`attn.sdpa.compressed`: denominator inflated ~1.65x** — the bench counted
  full `L_band × CATTN_KV` with a synthetic 95%-dense mask, but production passes
  a real causal mask. Real causal work 158.3 vs 261.3 GFLOP counted →
  **~48%, not 79.1%**. **UNDETERMINED pending one runtime test** (causal-vs-dense
  timing at production shape): if the fused kernel exploits the mask, ~48% is
  right and a span T7 wrote off as "at ceiling, dead end" has real headroom.
  This sits in the 71.2% GEMM-covered majority, not in the remainder.

## 9. Measurement-integrity notes

- **FLOPS ceiling corrected 11.66 → 14.34 TF.** The 11.66 figure
  (`bench/attn_production_class_bench.py:136-145`) was measured on the **laptop**
  M4 Max (32-core GPU), not on the Studio nodes (40-core). Internal proof the old
  denominator is wrong for this node: the capture measured `score_gemm` at 14.23
  TFLOPS *achieved on-node* — 122% of 11.66, impossible; 99.3% of 14.34,
  plausible. **Caveat: 14.34 is theoretical (40 cores × 128 lanes × 2 × ~1.4 GHz),
  not an on-node measured dense-fp16 peak.** Since achieved is ≥14.23, the true
  peak is ≥14.23, so 14.34 is slightly conservative and defensible. The rigorous
  fix is to run `measure_peak_gemm()` on a Studio node — not done here.
  **Any prior doc quoting 11.66 against a Studio-node measurement understated
  efficiency by ~19%.**
- `attn.mask` (windowed) reported GPU-busy 3852µs > wall 3026µs for the same op —
  MLX GPU-timer over-count across 4 small dispatches. Wall treated as
  authoritative; the anomaly did not appear on any large-per-call-work op, so the
  rest of the table is unaffected.
- Harness bugs found and fixed mid-run (a stray `score_full` call, a missing
  `rmsnorm.dram` variant) via a top-up pass rather than by patching numbers.
- `.eval()` applied to all constructed modules (the `training=True` trap);
  no cancelling arithmetic used to force evaluation; rotation banks sized past
  the L2 boundary with a small-bank-vs-large-bank sanity check reported.

## 10. Not measured

- HyperConnection `attn_hc`/`ffn_hc` and `hc_expand` at prefill shape with both
  fused kernels ON. Ran out of capture window. These are the two spans already
  *reduced* by shipped kernels (4.6%→~1.9%, 4.4%→~0.5%), so they are the least
  likely remaining source of headroom — but their post-ship prefill-shape
  per-kernel numbers are genuinely UNMEASURED, and any future claim about them
  should say so rather than reusing the pre-ship figures.
