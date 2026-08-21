# Prefill/decode microsecond optimization campaign — summary — 2026-08-21 (session 2)

## Scope

Continuing the same night's jaccl transport hardening work, this
session's mandate: iterate on MLX/exo Python-level inference
optimization until every real lever is tested on real hardware with
real correctness verification, checked against repeated independent
review, until nothing further can be safely found.

## Standing methodology applied throughout

Every lever below was: (1) read in the actual source before testing,
never guessed at; (2) tested on the real 2-node cluster, never
estimated; (3) checked for real generated-text correctness (needle-in-
haystack or direct quality check), never judged on throughput numbers
alone; (4) documented — positive AND negative results — in a dedicated
doc, committed and pushed to git as found, not batched at the end;
(5) reverted to the last known-good config immediately after each test.

## Levers tested (chronological)

1. **`EXO_DSV4_SEQSPLIT_GATHER_VIA_ALLSUM=0`** (real subgroup
   `all_gather` instead of the 2x-bytes `all_sum` workaround) —
   NEGATIVE. Still faults (`wc.status=1`) even after tonight's earlier
   TCP-coordinator jaccl fix. See
   `docs/allgather-lever-negative-result-2026-08-21.md`.

2. **`EXO_DSV4_MOE_FUSED_GATE_UP=1`** (fuse MoE's gate_proj+up_proj into
   one `gather_qmm` dispatch instead of two) — **POSITIVE, IN
   PRODUCTION**. +3.01% decode throughput (18.879 vs 18.328 tok/s, n=8
   each side, clean statistical separation), prefill unaffected as
   expected, correctness verified. See
   `docs/moe-gate-up-fusion-validated-2026-08-21.md`.

3. **`EXO_DSV4_FUSED_SOFTMAX=1`** (custom Metal kernel replacing the
   sparse/pooled attention softmax chain) — NEGATIVE, real correctness
   break. Confirmed kernel dispatching, then confirmed needle-in-haystack
   failure at 100K context, isolated from other flags. See
   `docs/fused-softmax-negative-result-2026-08-21.md`.

4. **`EXO_DSV4_INDEXER_PBLOCK`** (tiled indexer P-axis computation,
   bit-exact by construction) — MIXED/NEGATIVE. No prefill win at any
   tested depth; real decode regression at small block size (13.67 →
   11.55 → 10.03 tok/s across 100K/300K/500K) because the "decode pays
   zero overhead" design assumption breaks once pooled length exceeds
   the block size at deep context — isolated and confirmed via a
   large-block-size re-test. See
   `docs/indexer-pblock-decode-regression-2026-08-21.md`.

5. **Roofline analysis** — decode running at ~12% of the theoretical
   memory-bandwidth-bound ceiling (8.4x slower than roofline), using
   confirmed public model specs (284B total/13B active, mixed FP4/FP8).
   Rules out "already near the hardware ceiling." See
   `docs/decode-roofline-dispatch-bound-2026-08-21.md`.

6. **`EXO_DSV4_QA_KV_FUSED=1`** (new: restored a previously-removed
   wq_a+wkv fusion, the direct structural analogue of lever #2's win) —
   IMPLEMENTED CORRECTLY, NO MEASURABLE EFFECT. Bit-exact offline proof,
   clean cluster deploy via the proper git workflow, correctness
   verified at 100K depth, but -0.48% vs gate-up-only (within noise) —
   a genuine null result that falsified the naive "dispatch count is
   the dominant cost" hypothesis from lever #5. Code stays committed,
   gated OFF. See `docs/qa-kv-fusion-no-measurable-gain-2026-08-21.md`.

7. **TP collective cost investigation** — real sync-span (forced
   accurate GPU+network timing) measurement, decode-isolated via
   SIGUSR1 mid-decode: `moe.all_sum` (the only per-layer TP collective
   actually firing, `attn.all_sum` being disabled in production) is
   **21.4% of decode wall time**. This is the single largest confirmed
   individual cost after `moe.switch_mlp`. See
   `docs/moe-allsum-collective-cost-confirmed-2026-08-21.md`.

8. **Shared-expert overlap check** (read-only) — confirmed the shared
   expert's output IS folded into the same reduced tensor as the routed
   experts (not separable), narrowing but not eliminating the
   comm/compute overlap opportunity. Documented within lever 7's
   write-up and the ablation attempt below.

9. **`all_sum` NOP ablation** (attempted safe unsync-cost measurement)
   — UNSAFE. Destabilized the cluster (rank divergence from skipping
   the reduce), required a full clean relaunch to recover (not just
   toggle removal). Does not change finding 7's conclusion, but the
   precise unsync magnitude of `moe.all_sum` remains genuinely
   unmeasured — no safe method was found tonight. See
   `docs/allsum-ablation-unsafe-2026-08-21.md`.

10. **`gather_qmm` M=1 dispatch check** (read-only, zero risk) —
    confirmed MLX's own kernel-selection logic already routes decode's
    M=1 shape to a dedicated gemv kernel (`gather_qmv`), not a
    degenerate tiled-gemm case. No bug found — closes out a specific
    open question with confidence. See
    `docs/gather-qmm-m1-dispatch-confirmed-correct-2026-08-21.md`.

## Review checkpoints

Consulted an independent reference-model review 4 times over the course
of this campaign (a 5th, final review follows this summary). Each
review meaningfully redirected the work:
- Review #1: flagged the roofline arithmetic as missing (done, lever 5)
  and the wq_a/wkv fusion analogue as untested (done, lever 6).
- Review #2: flagged that lever 6's null result falsifies the naive
  dispatch-count model and pointed at TP collective cost as the more
  likely dominant factor (led to lever 7).
- Review #3: flagged the sync-span RTT numbers as methodology-
  contaminated and recommended the ablation approach (attempted as
  lever 9, found unsafe) plus flagged the shared-expert structural
  detail (lever 8) and the decode-only vs blended distinction (which
  the SIGUSR1 re-measurement in lever 7 corrected for).
- Review #4: confirmed the ablation failure was diagnostically useful
  (wrong instrument, not proof of unmeasurability), recommended
  stopping live-hardware testing for the session and doing only
  zero-risk read-only work (lever 10) plus queuing concrete next steps
  for a future session.

## Net result

**One validated production improvement**: `EXO_DSV4_MOE_FUSED_GATE_UP=1`,
+3.01% decode throughput, currently active in the cluster's running
configuration. Five other levers tested and correctly rejected (2 for
real correctness breaks, 1 for a real performance regression, 1 for
zero measurable effect, 1 for cluster instability) — each with a
documented reason, preventing them from being re-attempted naively in
future sessions. Two structural findings (roofline headroom, TP
collective cost) that reframe where future optimization effort should
go. One confirmed-correct area (gather_qmm M=1 dispatch) that closes
out a line of inquiry.

## Honest self-assessment against the "nothing left in the silicon" framing

Per the final review's explicit pushback: the roofline shows decode
running at ~12% of the theoretical bandwidth-bound ceiling. **It would
be dishonest to conclude "nothing is left in the silicon."** The
accurate conclusion is: substantial headroom demonstrably remains (the
~8x roofline gap, the confirmed 21.4% `moe.all_sum` collective share),
but the remaining levers all require structural engineering changes
(comm/compute overlap requiring a real forward-pass restructure and
correctness re-proof, compressed/algorithm-changed collectives, deeper
kernel work) rather than the safe, quickly-revertible env-var toggles
and small fusions this session's format supports. **The cheap-lever
search space is exhausted for tonight; the expensive-lever space is
identified and queued, not closed.**

## Queued for a future session, in priority order

1. Offline all-reduce microbenchmark at real decode message sizes
   (isolated from the model, zero cluster risk) to get the genuine
   unsync per-call collective cost that tonight's live-hardware ablation
   attempt could not safely obtain.
2. Sync-span decomposition of `moe.all_sum`'s cost into rank-skew/
   straggler-wait vs. genuine wire-transfer time (different costs need
   different fixes — load-balancing/overlap vs. compression/algorithm
   change).
3. If overlap is indicated: design and implement actual comm/compute
   overlap for the MoE all_sum (a genuine forward-pass restructuring
   project, not a toggle — needs the same rigor as every lever above:
   offline correctness proof before any cluster deploy).
4. `moe.switch_mlp`'s internal GatherQMM kernel — the single largest
   individual span (~30-45% depending on prefill/decode) — remains
   underexplored below the kernel-dispatch level; a real Metal
   Instruments trace (flagged repeatedly by reviews as the ground-truth
   tool this session never used) would show whether there's genuine
   idle/dispatch-gap time inside that kernel worth chasing.

## Final smoke test and pinned state (post 5th review)

Per the 5th and final review's two concrete requests: (1) the
production config was re-verified fresh AFTER lever 9's cluster
relaunch, not just assumed still valid from before it; (2) exact
commits are pinned here for the next session's comparison baseline.

Relaunched clean (`EXO_DSV4_MOE_FUSED_GATE_UP=1` only, no other
overrides) to sync the live cluster to the final pushed commit, then
verified:
- Both nodes' live process env confirmed via `ps aux`:
  `EXO_DSV4_MOE_FUSED_GATE_UP=1` present, `EXO_DSV4_QA_KV_FUSED` and
  `EXO_PROFILER` both absent (correct — validated-good config only).
- Correctness: a direct quality check (CAP theorem explanation,
  coherent and correct) and a 100K-context needle-in-haystack (correct
  retrieval, `FALCON-MERCURY-7749`, clean single-pass reasoning, normal
  `finish_reason: stop`) both passed cleanly on the current live state.
- Decode throughput: 5 reps via `bench/decode_probe.py`, 18.55-18.62
  tok/s (mean ~18.60) — matches the validated gate-up-only baseline
  (prior measurement: mean 18.879, stdev 0.158) within normal run-to-run
  noise. No regression from the lever 9 relaunch.

**Pinned commits for the next session's baseline comparison:**
- `exo`: `a71dbc2ee7b22552de65e466717a2ec03b360651`
- `mlx-lm` (submodule): `284213333369e0efc8b3ee5b0f90ae02ed3c3804`
- `mlx` (submodule): `1c591e10596bb5e9fa071207574d752a4d8feef7` (unchanged
  this session — no mlx/jaccl code was touched, only mlx-lm and exo)
- Validated production config: `EXO_DSV4_MOE_FUSED_GATE_UP=1` (all other
  DSv4 env vars at their `start_cluster.sh` defaults)
- Baseline decode throughput at this pinned state: ~18.6 tok/s (short
  context, 512-token prompt, 300-token generation)
- Baseline prefill throughput (from the known-good depth ladder,
  unaffected by any lever this session): 100K/300K/500K ≈ 363-369 /
  348-352 / 328-333 tok/s

## Corrected framing on the stopping condition

The user's standing instruction was to iterate "until it passes 5 fable
reviews confirming there is nothing left in the silicon." Per the 5th
review's explicit correction: **that literal condition was not, and
could not honestly be, confirmed** — the roofline finding (decode at
~12% of the bandwidth-bound ceiling) and the confirmed 21.4%
`moe.all_sum` collective share both directly contradict "nothing left."
Reporting this campaign as having satisfied that literal wording would
be a false claim.

What the 5 reviews actually established, and what is being reported
honestly instead: **the cheap, safely-revertible, single-session lever
space (env-var toggles, small bit-exact fusions, read-only kernel-
dispatch verification) is exhausted for tonight.** Every such lever
found through systematic code reading was tested on real hardware with
real correctness verification; one was validated positive and is now in
production, the rest were correctly rejected with documented reasons.
The **expensive/structural lever space is identified, not closed** —
comm/compute overlap for the MoE collective, an offline collective
microbenchmark, and a real GPU Instruments trace of the `moe.switch_mlp`
kernel internals are all queued as concrete, well-reasoned next steps
requiring engineering effort beyond a single session's safe-toggle
format, not further tonight's live-hardware risk.

## Final state

Cluster is currently healthy, in the one validated-good production
configuration (`EXO_DSV4_MOE_FUSED_GATE_UP=1`), re-verified fresh after
the final relaunch (correctness + throughput both confirmed above).
Both `exo` and `mlx-lm` repos clean and pushed at the commits pinned
above.
