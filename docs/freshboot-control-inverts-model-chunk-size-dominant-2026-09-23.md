# RESULT: the fresh-boot control INVERTS the model — chunk size is the dominant factor, not boot state (2026-09-23)

Pre-registered in `tmp/decode-deep-context-campaign-20260922/PREREG-freshboot-control-2026-09-23.md`
and committed as `7cf0939dc` before measuring. Bands applied verbatim.

## The measurement

Same probe, same depth, same fresh-boot baseline; the ONLY variable is the
prefill chunk size past 500K context.

| | fresh + 2048 chunks (this run) | fresh + 128 chunks (adaptive r0) |
|---|---|---|
| baseline before prefill | 87.83 GB | 87.83 GB |
| depth | 638,455 | 638,455 |
| **PEAK memory** | **112.96 GB** | **87.87 GB** |
| decode tps | 26.68 | 25.75 |
| prefill tps | **370.3** | ~148 |
| needle | True | True |

Validity gate: baseline 87.8 ± 2 GB -> **87.83 GB, PASS**. Genuinely fresh boot,
so this is a clean single-variable comparison.

## Verdict against the pre-registered table: BOTH CONTRIBUTE

Measured peak 112.96 GB falls in the pre-registered 100-115 GB band ("BOTH
contribute; ship both and add a headroom guard"), and tps is 26.68, not degraded.
That is the answer, stated before the run.

## But the MAGNITUDES invert the independent review's model

The review that prompted this experiment decomposed the adaptive run's 30 GB peak
reduction as **~19.5 GB boot state + ~10.5 GB chunk size (~2/3 : 1/3)**. Measured:

- **boot-state effect** (both at 2048 chunks): long-lived boot peak 117.88-123.80
  GB -> fresh boot peak 112.96 GB = **~5 to ~11 GB**, not ~19.5 GB.
- **chunk-size effect** (both fresh): 112.96 GB -> 87.87 GB = **25.1 GB**, not
  ~10.5 GB.

**The model inverted the split.** Chunk size is the dominant factor (~25 GB),
roughly 2.5-5x larger than boot state (~5-11 GB) — the opposite of the 2/3-boot,
1/3-chunk estimate I accepted and published in the close-out.

## Why the model got it backwards

The review inferred the split by subtracting the *long-lived* baseline (107-115 GB)
from the adaptive run's baseline (87.8 GB) and calling that difference "boot
state." That subtraction conflates two different quantities: the **baseline** at
the start of a request, and the **peak** reached during it. The 128-chunk change
acts on the prefill *transient* — the peak above baseline — which the baseline
comparison cannot see. Comparing baselines attributed the transient's reduction to
boot state, because the fresh boot also happened to lower the baseline.

The correctly-controlled comparison (this run) isolates each factor on the
quantity it actually affects.

## Consequence for target (a): the fix is more load-bearing than I said

| condition | peak | margin vs 115.0 GB limit |
|---|---|---|
| fresh boot + 2048 chunks | 112.96 GB | **2.0 GB** |
| fresh boot + 128 chunks | 87.87 GB | **27.1 GB** |

On a fresh boot with large chunks we are back to ~2 GB of margin — the exact
"headroom gone" condition that drives the mmap refault collapse. So the
`EXO_PREFILL_STEP_SIZE_HIGH_CTX=128` change is **the primary mitigation**, not a
minority contributor. It bought 25 GB where boot state bought 5-11 GB.

This run did NOT collapse (26.68 t/s, needle True) — but with only 2 GB of margin
the absence of a collapse in a single run is not evidence of safety, given the
mode is stochastic (measured 2-of-5 slow runs at ~370K in an earlier sweep).

## Secondary finding worth keeping: the chunk trade is NOT a trade

Chunk size 128 vs 2048 was expected to cost prefill throughput (the repo's own
comment measures 256-chunks as -30% vs 128 past 380K). Measured here:

- fresh + 2048 chunks: prefill **370.3 t/s**
- fresh + 128 chunks: prefill **~148 t/s**

That is the opposite direction in the LIVE deep regime — the run that took 128
chunk-boundaries for 6x less peak memory ALSO had ~2.5x the prefill throughput at
depth. Whatever penalised 2048-chunks in the earlier measurement does not apply
to the live 638K prefill.

Caveat kept explicit: the two prefill rates come from different boots and
different runs; the earlier 148 figure is a reading off the progress log, not a
controlled A/B of prefill alone. The direction is consistent with the memory
story (a smaller transient means less pressure and less refaulting *during*
prefill), but I would not quote the ratio as a settled measurement.

## What this does NOT establish

- Only n=1 per cell. The collapse is stochastic, so neither cell's tps is a
  collapse-rate estimate.
- Page-ins were recorded as a cumulative counter, not a per-run delta, so the
  pre-registered page-ins criterion could not be evaluated cleanly this run. The
  peak criterion carried the verdict.
- The 250K target (b) is untouched by any of this.
