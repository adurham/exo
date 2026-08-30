# P02d — Reconciling decode's 14-20% roofline figure against 78-85% occupancy and 88-97% switch_mlp kernel efficiency (2026-08-29)

**Scope: PURE DESK ANALYSIS.** No cluster access, no relaunch, no probes —
arithmetic against numbers already measured, dated, and committed. Every input
below carries its source. Analysis scripts + raw output:
`tmp/p02d-20260829/` (`reconcile.py`, `weighted_avg.py`,
`reconcile_output.json`).

## 1. The question

Since 2026-08-22, three findings have coexisted without ever being explicitly
multiplied together:

1. **Aggregate roofline** (`docs/roofline-recalculated-post-fix-2026-08-22.md`):
   post-fence-fix decode runs at **14.0-20.2% of the theoretical
   bandwidth-bound peak** (100K: 17.5%, 300K: 15.9%, 500K: 14.0%,
   short: 19.0-20.2%) — "5-7x slower than the hardware ceiling," flagged as
   the highest-priority open question.
2. **GPU occupancy** (T2/T5/C2, 2026-08-22/23): request-window-isolated
   interval-union occupancy is **78.6-78.9% (short), 82.4-82.7% (300K),
   82.98-83.06% (100K, 50s window)**.
3. **Per-kernel efficiency** (`docs/p01-switch-mlp-gputrace-recapture-2026-08-29.md`):
   the `moe.switch_mlp` kernel — measured DRAM-real, rotated indices — runs
   at **97.5% (fused_gate_up) / 88.5% (down_proj) of the 546 GB/s spec**
   (stage-sum 91.4%).

**The apparent paradox:** 0.80 occupancy × 0.91 kernel efficiency ≈ 0.73, yet
the aggregate figure says 0.14-0.20. That is a **3.8x mismatch** — and the
question is where it lives.

**A premise correction before any arithmetic:** the task framing (and several
older docs) calls switch_mlp "~30-45% of wall time." That figure is the
**RETRACTED 2026-08-22 span-profiling artifact** — span profiling inserts
~430 per-section `mx.eval` syncs/token and over-attributes exactly like the
serial-sync microbench did. The retraction doc's corrected arithmetic and the
p01 recapture agree: real switch_mlp is **~5.0 ms/token ≈ 14-15% of wall**
(chained 117µs/call × 43 layers), GPU-busy stage sum 4.07 ms = **13.7% of
busy time**. The dominant single kernel is attention (43-46% of wall per
worker C's census), not switch_mlp. Any reconciliation that starts from
"switch_mlp is 30-45% of wall" starts from a retracted number.

## 2. Method

Two computations over documented numbers only:

1. **Exact factorization of the headline.** The headline ratio is
   `6.51 ms / wall`. Multiplying and dividing by the same quantities gives
   the identity
   `headline% = coverage × occupancy × busy-blend`, where
   `coverage = 3.56 GB / B_true` (the roofline's byte denominator vs the
   true per-rank byte inventory), `occupancy` is the measured interval-union
   fraction, and `busy-blend = (B_true / busy_ms) / 546 GB/s` is the
   achieved-bandwidth efficiency of the GPU-busy time under the true byte
   accounting. **Honest status of this identity: it is bookkeeping, not
   validation — it closes to 0.00pp at every depth by construction, because
   busy-blend is derived, not measured.** Its value is that it names the
   three levers and their sizes. The actual empirical validation of
   `B_true` is independent: worker C's kernel census (§3).
2. **Weighted average across ALL busy-time buckets** (the candidate-(b)
   computation): switch_mlp at its measured 91.4%, the attention path at
   its census-measured GB/s, and the remaining busy time as the implied
   residual bucket. Cross-checked against the observed headline.

**The true per-rank byte inventory `B_true`** (bytes that must move per
decode token, per rank, spec-off plain-decode regime — the same regime the
roofline was computed in):

| component | bytes/rank/token | source |
|---|---|---|
| attention-path weights + fixed caches (REPLICATED per rank — exo's TP shards MoE only, `auto_parallel.py:1032-1034`) | 5.2976 GB | worker A code-derived inventory, R1-verified, checkpoint-index cross-checked (mxfp8 1.03125 B/param etc.) |
| depth-linear indexer/pool reads | 1930.25 B/context-token | worker A (B1-B7 terms) |
| `moe.switch_mlp` routed experts (SHARDED, both ranks hold all 256 experts at half width) | 47.186 MB × 43 = 2.029 GB | microbench ground truth; retraction doc and p01 recapture agree exactly |
| shared expert + moe.gate (sharded) | ~0.29 GB | 13B active − routed − attention − head, mxfp4-ish 0.625 B/param, ÷2 |
| lm_head (~129K vocab × 4096, replicated or sharded — carried as ±0.28 GB sensitivity) | ~0.55 GB | param arithmetic |

`B_true = 8.17 GB` (short) → `9.14 GB` (500K). **The roofline's denominator
(3.56 GB) covers only 0.390-0.436 of it.** Independent validation: worker C's
measured attention-path wall time at L=520 (12.876 ms) vs the byte model's
prediction at his own session-measured streaming bandwidth (5.298 GB ÷ 405
GB/s = 13.1 ms) — **2% agreement**; and C2's measured idle at 100K (6.06-6.09
ms/token) matches this analysis's idle column exactly (6.09).

## 3. Results

### 3.1 The three-factor factorization

| case (wall, occupancy pairing) | headline | coverage | occupancy | busy-blend |
|---|---|---|---|---|
| short ~520 (B1 anchor; T2 occ) | 19.3% | 0.436 | 0.786 | 0.564 |
| 100K (B1 anchor; C2 50s occ, same-day) | 18.2% | 0.426 | 0.830 | 0.516 |
| 300K (T1 anchor; T5 9s occ) ⚠ | 15.9% | 0.407 | 0.824 | 0.475 |
| 352.6K (B1 anchor; T5 occ) ⚠ | 15.3% | 0.402 | 0.824 | 0.462 |
| 500K (T1 anchor; occ never captured, 82.4% carried) ⚠ | 14.0% | 0.390 | 0.824 | 0.437 |

⚠ = T1-era anchors carry the documented EOS-ban probe bug and the 300K/500K
occupancy pairings are cross-run (flagged in C2 §5); the clean-anchor rows
(short, 100K, 352.6K) are the primary claim, the ⚠ rows are caveated
extensions.

**The 3.8x mismatch splits as:**

- **2.29-2.57x from byte accounting** (candidate (a), CONFIRMED, largest
  factor). The roofline's 3.56 GB counts active MoE expert bytes only, at a
  whole-model 0.588 B/param average, halved for TP. It omits: (i) the
  **attention path, which is REPLICATED not sharded** — 5.30 GB/rank of
  mxfp8-class weights read every token, roughly 2.4x the roofline's entire
  denominator by itself; (ii) KV/pool/indexer reads (1930.25 B per
  context-token); (iii) lm_head, shared expert, gate. The 0.588 B/param
  average is also wrong per-tensor: attention weights are mxfp8
  (~1.03 B/param), switch experts mxfp4 (~0.63 B/param with scales).
- **1.63-2.08x from busy-time composition** (candidate (b), CONFIRMED,
  second factor). Weighted-average efficiency across ALL busy-time buckets:

| case | busy ms | switch_mlp (13.7% of busy) | attention census (49-58% of busy) | implied small-op bucket (27-31% of busy) | blend over busy | × occupancy |
|---|---|---|---|---|---|---|
| short | 26.54 | 91.4% of spec | 75.4% | 16.1% | 0.564 | 0.444 |
| 100K | 29.70 | 91.4% | 60.7% | 20.9% | 0.528 | 0.438 |
| 352.6K | 35.11 | 91.4% | 57.2% | 23.4% | 0.497 | 0.410 |

  `coverage × (blend × occupancy)` lands within ~1pp of the observed
  headline at every clean depth (short: 0.436 × 0.444 = 19.4% vs 19.3%;
  100K: 0.426 × 0.438 = 18.7% vs 18.2%; 352.6K: 0.402 × 0.410 = 16.5% vs
  15.3% — the ~1pp residual is the ~5% spread between the two independent
  attention-byte estimates). **A single kernel at 91% cannot propagate: it
  is 13.7% of busy. The blend is dominated by attention (49-58% of busy at
  57-75% of spec) and dragged by the small-op bucket (27-31% of busy at
  16-23% of spec).**
- **Candidate (c) — genuine headroom — is real but far smaller than the
  "5-7x" framing implied.** With the true denominator, decode runs at:

| case | % of 546 spec | % of 424 real-streaming | slower than spec | slower than real ceiling |
|---|---|---|---|---|
| short | 44.3% | 57.1% | 2.26x | **1.75x** |
| 100K | 42.8% | 55.1% | 2.34x | **1.81x** |
| 352.6K | 38.1% | 49.0% | 2.63x | **2.04x** |
| 500K ⚠ | 36.0% | 46.3% | 2.78x | **2.16x** |

  Both denominators are shown deliberately: 546 GB/s is the spec-sheet
  ceiling the 14-20% figure used; 424 GB/s is the measured real streaming
  ceiling on these machines (`exo-perf-tuning` "Hardware Truths"; worker C's
  same-session measurement was 404.7 GB/s — the ~5% spread is noted). Note
  the 424 GB/s figure is itself a transplant from a different workload
  (bulk streaming copy) and is a ceiling for kernels WITHOUT L2 reuse;
  worker C measured indexer-score at 477-558 GB/s at depth (above
  "streaming") precisely because of L2 reuse — so the real-ceiling column
  is conservative at depth.

### 3.2 Where the true gap lives (vs the real-streaming floor)

Decomposition of `wall − B_true/424 GB/s` (accounting identity — the
buckets are exhaustive by construction, so this is a naming of costs, not a
discovery of an unexplained remainder):

| case | total gap | GPU idle (measured) | small-op latency-bound excess | attention+switch above byte floor |
|---|---|---|---|---|
| short | 14.5 ms | 7.2 | 6.6 | 0.6 |
| 100K | 16.1 | 6.1 | 6.1 | 3.9 |
| 352.6K | 21.7 | 7.5 | 9.0 | 5.3 |
| 500K ⚠ | 24.9 | 8.2 | 9.8 | 7.0 |

Three honest caveats on this table: (i) idle matches C2's direct
measurement at 100K exactly (6.09-6.47 measured vs 6.1 here) — that bucket
is solid; (ii) the "attention+switch above byte floor" column is an UPPER
BOUND on inefficiency, because B_true is an upper bound on DRAM traffic (L2
reuse means real DRAM bytes are lower, so real efficiency is higher);
(iii) the small-op bucket is **implied, never directly measured** — no
instrument has ever produced a per-kernel table for moe.gate, shared
expert, combine, norms, residuals, rope, or lm_head.

## 4. Verdict

1. **The three numbers were never in contradiction.** "14-20% of peak" and
   "80% busy with the dominant gather kernel at 91%" are mutually
   consistent once the accounting is corrected. The naive product
   (0.80 × 0.91 = 0.73 vs 0.14-0.20) is a 3.8x mismatch whose largest
   single piece (2.29-2.57x) is the roofline's byte denominator omitting
   the replicated attention path and everything but routed-expert bytes;
   the next piece (1.63-2.08x) is that a 91%-efficient kernel occupying
   13.7% of busy time cannot set the blend, which is instead set by
   attention (49-58% of busy at 57-75%) and the small-op bucket (27-31%
   at 16-23%). **Candidate (a) and candidate (b) are both confirmed and
   quantified; this closes the "5-7x slower than the hardware ceiling"
   open question as an accounting artifact, not a performance mystery.**
2. **The real headroom is 1.75-2.16x vs the measured streaming ceiling**
   (2.26-2.78x vs spec), not 5-7x — and it decomposes into: idle
   (~6-8 ms/token, already bounded by C2 at 100K), the small-op
   latency-bound bucket (~6-10 ms/token, never characterized), and
   depth-growing kernel/byte-model divergence (0.6-7.0 ms, partially an
   L2-reuse artifact of the byte model). The campaign's open
   **+1.67..+2.52 ms/tok residual band (P01a/1b/2c, 2026-08-29) lives
   inside these buckets** — this analysis bounds it but does not attribute
   it.
3. **What is NOT dissolved:** the small-op bucket. It is 27-31% of GPU-busy
   time running at an implied 16-23% of spec bandwidth (~88-140 GB/s) —
   ~6-10 ms/token of latency-bound dispatch-small-op time that has never
   been directly measured per-kernel, in any campaign, at any depth.

## 5. Recommended next investigation target (not executed — analysis-only task)

**Characterize the small-op bucket with the already-proven capture recipe.**
One `mx.metal.start_capture()` + `MLX_GPU_TIME=1` bracketing pass over a
real decode window (the exact method the 2026-08-29 p01 recapture proved
works on production silicon, `METAL_CAPTURE_ENABLED=1`), with per-stage
brackets around moe.gate / shared_experts / post_combine / norms /
residuals / lm_head — building the first per-kernel time table for the
27-31%-of-busy bucket that this reconciliation identifies as the largest
never-measured cost. Run it **in both the spec-off regime (comparability
with the census/anchors used here) and the current production spec-ON
(verbon3) regime** — the spec-ON wall times differ (p01a: 27.46 ms/tok
@100K), so the roofline-anchored picture above must be re-based before it
is applied to today's production numbers. Expected outcomes: either the
bucket is irreducible per-op dispatch latency (→ the practical decode
ceiling is the current wall minus idle, and decode tuning is done), or a
concentration is found (e.g. one op class dominating) → a concrete fusion
target. Cheap, low-risk, zero new infrastructure.

## 6. Limitations (read before citing)

- **The factorization is bookkeeping, not validation.** Busy-blend is
  derived from the identity; its "closure" is guaranteed. The empirical
  legs are: worker C's census (independent, 2% agreement at L=520), C2's
  measured idle (exact match at 100K), and the switch_mlp microbench bytes
  (two independent docs agree to 4 significant figures). The attention
  byte model is directly census-validated only at short context; at
  100K/352.6K the census measures TIME (validated) while the byte split
  relies on worker A's code-derived model (R1-verified, but a model).
- **B_true is an upper bound on DRAM traffic** (assumes every weight byte
  misses L2 every token). Real efficiency is therefore somewhat higher
  than the corrected columns state — the 1.75-2.16x "slower than real
  ceiling" is itself slightly pessimistic. The indexer-score kernel
  (477-558 GB/s measured at depth, above the 405-424 streaming figure) is
  the demonstrated case of this.
- **Regime mixing.** Anchors, census, and occupancy are 2026-08-22/23
  spec-off plain-decode era; the roofline's 1-forward-pass-per-token
  assumption does NOT hold in today's spec-ON production (batched verify
  amortizes a forward over ~2.3 tokens at 100K). The reconciliation is
  internally consistent (all inputs from the same era/regime) but applies
  to that era; re-basing for spec-ON is part of the recommended next step.
- The 300K/500K rows pair T1-era (EOS-bug-flagged) anchors with cross-run
  occupancy windows; the clean-anchor claim is the short/100K/352.6K rows.
- lm_head sharding under TP is unverified (±0.28 GB on B_true → ±1.5pp on
  coverage — does not change any conclusion).

## 7. Artifacts

- `tmp/p02d-20260829/reconcile.py` — factorization + corrected-efficiency +
  gap-decomposition arithmetic (all inputs sourced in-line).
- `tmp/p02d-20260829/weighted_avg.py` — candidate-(b) weighted-average table.
- `tmp/p02d-20260829/reconcile_output.json` — machine-readable output.

## 8. Cluster state

Not touched. Pure offline analysis, zero SSH/relaunch/probe activity, per
task scope.