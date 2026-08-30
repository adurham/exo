# P02c — MoE-at-depth interplay vs allocator/memory-pressure, attacking the +1.67..+2.52 ms/tok residual (2026-08-29)

## 1. Question and why it was still open

After Phase 1(a) (all_sum arrival-skew, measured inside the collective) and
Phase 1(b) (inter-layer pipelining loss), the +1.67..+2.52 ms/tok on-GPU-busy
residual (100K→352.6K, same-build clean anchors 34.55→39.49 = +4.94 ms/tok
total, kernel band +2.56..+3.34 subtracted, collective growth -0.07..-0.14
subtracted) remained fully open. Two candidates were still live:

1. **MoE-at-depth interplay** — MoE decode cost is nominally
   depth-independent (fixed weights, top_k=6, M=1 per decode step), but
   nobody had tested whether the MoE module has hidden depth-dependent
   interplay with the rest of the forward pass (memory pressure, cache
   locality, scheduling) that a static per-kernel census cannot see.
2. **Allocator/memory-pressure at the ~90 GB-resident regime** — production
   telemetry (spec-ON, `CLEAR_CACHE_INTERVAL=64`) previously showed cache
   growth rate 3.7x faster at 352.6K-class depth than at 100K, with
   active+cache riding to within ~0.6 GB of `gc_limit` (114.557 GB) at
   depth. P01a's own probe-gap quintile read (spec-ON production telemetry)
   reported 352.6K "worsens monotonically WITHIN the window" (32.0→42.4
   ms/event Q1→Q5) while 100K stayed flat (26.5→25.8).

A prior session designed a real A/B protocol for both candidates
(`tmp/p02c-20260829/DESIGN.md`), executed the probes, and ran out of
iteration budget before analysis or write-up. This document is that
analysis, run entirely offline against the already-collected files in
`tmp/p02c-20260829/` — **no cluster access, no new probes.**

## 2. Method (as executed, deviations from DESIGN.md noted)

- **Arm A** ("production-mirrored"): spec-off (`SPECULATIVE=0 MTP=0
  DSPARK=0`), band-era allocator env (`EXO_MLX_CLEAR_CACHE_INTERVAL=0`,
  `EXO_GC_COLLECT_INTERVAL=0`), temp=0.0. This is the same regime the
  +4.94 ms/tok clean-anchor total was measured in (P3 Part III).
- **Arm B** ("MoE-NOP"): `/tmp/dsv4_nop_targets` containing `moe` — the
  first line of `DeepseekV4MoE.__call__` returns `mx.zeros_like(x)`
  immediately (verified at `mlx_lm/models/deepseek_v4.py:2932`), skipping
  gate/switch_mlp/shared_experts/combine and the MoE-local `all_sum`. The
  global coordination collective and seq-split prefill gathers are
  untouched. Temp=1.0 (garbage logits at temp=0 hit the exact-repeat
  long-period degeneration detector at a deterministic ~72 tokens — see
  §5 for a data-quality note on how well this mitigation actually held).
- **Probe**: `bench/p3_depth_anchor_probe.py --max-tokens 2000`, the
  EOS-banning `/bench` route, `--target-tokens {100000, 352600}`.
- **Order**: 100K block ran A,B,B,A (`block_100k_meta.json`); two of the
  B reps (`100k_1B`, `100k_2B`) hit the degeneration kill at 71 tokens
  (see §5) and were **excluded**, then re-run as a standalone `100kb`
  block (B,B — `block_100kb_meta.json`), both clean. 352.6K block ran
  the full intended A,B,B,A (`block_352k_meta.json`), all four clean.
- **Telemetry**: `MLX_LOG_NEW_BUFFER_PATH` (fresh-allocation log,
  `newbuf_m4-1_snap2.log`), `EXO_MEMORY_PROFILE_INTERVAL=8` structured JSON
  dump (`mem_p02c_m4-1_snap1.jsonl`), and a 1Hz `vm_stat` sampler
  (`sampler_m4-1_snap1.log`) — **m4-1 (one rank) only**, no m4-2 copies of
  these three files exist in the collected set.

All raw analysis scripts used to produce the numbers below live in
`tmp/p02c-20260829/analysis/` (created during this write-up pass, offline,
read-only against the existing probe/meta/log files).

## 3. Candidate 1: MoE-at-depth interplay

### 3.1 Canonical numbers

| Arm | 100K reps (ms/tok, usage-based) | 100K mean | 352.6K reps | 352.6K mean | delta |
|---|---|---|---|---|---|
| A (prod-mirrored) | 35.685, 35.586 | **35.635** | 40.310, 40.362 | **40.336** | **dA = +4.70** |
| B (MoE-NOP) | 27.261, 27.206 (from `100kb_0B`/`100kb_1B`) | **27.233** | 32.282, 32.277 | **32.280** | **dB = +5.05** |

Both cells are tight (<0.3% spread between reps, n=2/cell) — the numbers
above are not a small-n noise artifact.

**Reproduction gate (DESIGN.md: "arm A @100K/@352.6K should REPRODUCE
~34.5/39.5 ms/tok — that reproduction is itself a validity gate"): PASSES.**
35.64 vs 34.55 expected (+3.1%), 40.34 vs 39.49 expected (+2.1%). dA=+4.70
is within 4.8% of the design doc's own +4.94 clean-anchor expectation — a
third independent occasion (P3 Part III → P01a → this) landing in the same
band.

### 3.2 Validity gate on dB: FAILS

DESIGN.md's explicit instruction: *"dB must be compared against worker C's
kernel census (+2.56 ms/tok expectation) as a live calibration. If dB is
wildly off census, the overlap structure changed too much → report both
numbers as weakly-related observations, do NOT force attribution."*

dB = **+5.05 ms/tok** vs kernel census **+2.56 ms/tok** (worker C's primary
run) to **+3.34 ms/tok** (C's noisiest of three fencing-mode runs):

- vs census low (2.56): dB is **1.97x** (+97%)
- vs census high (3.34): dB is **1.51x** (+51%)

Even against the generous top of the three-run census range, dB overshoots
by half again. **This is a literal, non-marginal gate failure**, not a
rounding-distance miss. Per DESIGN.md's own instruction, the naive
delta-of-deltas (dA − dB = +4.70 − 5.05 = **−0.35 ms/tok**) must **not** be
forced into an attribution. It is reported below as an informational,
gate-failed observation only.

### 3.3 Digging into why dB overshoots (root-cause discipline, not forcing the fit)

Three checks before accepting "gate failed, move on":

**(a) Is it noise?** No — reps are tight (<0.3% spread) at every cell; the
overshoot is reproducible within this run.

**(b) Does the DESIGN.md-flagged arm-B bias explain it?** DESIGN.md flags:
*"garbage hidden states → possibly degenerate topk indices → gather
locality could make arm B attention slightly cheaper at depth."* This bias
predicts dB should be biased **down** (an undershoot vs the true
attention+framework growth). Observed dB **overshoots** census by ~50-97%.
The stated bias points the wrong direction — it does not explain the
overshoot, and if anything makes the puzzle a little more interesting (the
true "cheaper" bias, if real, means the underlying uncorrected number could
be even larger than +5.05).

**(c) Mechanistic candidate: dB structurally includes a cost the bare kernel
census structurally excludes.** DESIGN.md defines dB as *"attention +
framework depth growth"* — not just attention kernel time. The NOP only
short-circuits `DeepseekV4MoE.__call__`; it does not touch KV/pool cache
bookkeeping. A separate, already-completed phase of this campaign (P3
worker C3, `docs/p3-worker-c3-donation-failure-insitu-2026-08-23.md`,
independently re-verified by reviewer R2) isolated a **+1.91 ms/tok**
depth-growth cost from `BatchPoolingCache`'s per-flush `mx.concatenate`
pool reallocation — a cost on the live production critical path that
worker C's synthetic single-layer kernel census structurally never paid
(C's harness amortized flushes 1-in-256; production flushes every 4th
token). That +1.91 ms/tok was established as **additive** with the kernel
census, not a subset of it.

If dB — running through the *real* production decode loop's cache
management, just with MoE zeroed — is compared against kernel+C3 combined
rather than bare kernel alone:

| comparison | band | dB=+5.05 vs band |
|---|---|---|
| bare kernel | +2.56 .. +3.34 | overshoot **+1.71 to +2.49** ms/tok (51-97%) |
| kernel + C3 (+1.91) | +4.47 .. +5.25 | overshoot **−0.20 to +0.58** ms/tok |

This mechanistic explanation is **plausible and evidence-grounded**, not
invented — the C3 mechanism runs on both arms A and B identically (only
MoE math is stubbed) and was independently verified in an earlier phase.
Under it, most (not all) of the overshoot resolves.

**This is presented as a secondary, tentative root-cause hypothesis for
the gate failure — not as a retroactive pass of the literal gate, and not
as a redefinition of the campaign's official residual band.** Two honest
caveats:

1. The campaign's official residual band (+1.67..+2.52) was derived by
   subtracting **bare** kernel census from the total; it does not already
   have C3 subtracted out. Folding C3 into dB's calibration target without
   also removing it from the residual-band side would double-count C3.
   Recomputing the band with C3 folded in on both sides gives a band that
   straddles zero (**−0.24 to +0.61 ms/tok**) — informational only, not
   adopted as the campaign's residual definition here; that is a call for
   whoever owns the campaign's bookkeeping, not something this analysis
   pass unilaterally changes.
2. Even under the generous kernel+C3 framing, a small (~0.2-0.6 ms/tok)
   gap remains at the edges of the band — the reinterpretation narrows the
   miss substantially but does not fully close it.

### 3.4 What the (gate-failed) numbers are still weakly suggestive of

Reported per DESIGN.md's instruction as a weakly-related observation, not
an attribution: dA (+4.70) and dB (+5.05) are close to each other (within
~7%, dA−dB = −0.35 ms/tok). If MoE cost genuinely does not scale with
depth (DESIGN.md's own opening hypothesis), removing it from both ends of
the delta should leave the delta roughly unchanged — which is roughly what
happened. −0.35 ms/tok is about 3x the previously-established
collective-growth contamination floor (~0.12 ms/tok, P01a) so it isn't
pure noise-floor dust, but it is small relative to the ~2 ms/tok residual
band and, again, **rests on a failed validity gate** — it is not being
carried forward as a validated attribution.

**Candidate 1 verdict: INCONCLUSIVE.** The validity gate specified by the
design doc fails under its literal definition. A plausible, mechanistically
grounded explanation for most (not all) of the overshoot exists (shared
cache-management cost dB pays that bare-kernel census structurally
excludes), but adopting it requires a bookkeeping change (removing C3 from
the residual band) that this pass flags rather than makes unilaterally.
Net contribution of MoE-at-depth interplay to the residual: **not
established, zero ms/tok confidently attributed either way.**

## 4. Candidate 2: allocator/memory-pressure at the ~90 GB-resident regime

### 4.1 Critical data gap: 352.6K telemetry does not exist

Before any analysis, the collected files were cross-checked against the
352.6K block's timing window (`block_352k_meta.json`: 16:45:38–17:47:40).

| telemetry file | m4-1 window | covers 352.6K block? |
|---|---|---|
| `sampler_m4-1_snap1.log` (1Hz vm_stat) | 16:15:34–16:36:49 | **NO** |
| `mem_p02c_m4-1_snap1.jsonl` (structured mem profile) | 16:17:01–16:35:44 | **NO** |
| `newbuf_m4-1_snap2.log` (fresh-alloc log) | byte range ends at offset 2,677,020 = exactly the `100kb_1B` post-watermark | **NO** |

All three per-node telemetry streams stop at (or just after) the end of
the **100kb** block, roughly nine minutes before the 352.6K block even
starts. Cross-checked two independent ways (wall-clock timestamp ranges,
and newbuf's byte-offset watermarks recorded in the meta JSON files before
and after every probe) — both agree. There is no rank-2 (m4-2) copy of any
of these three files either. **The design doc's entire R1 in-band
discriminator — comparing newbuf rate, cache growth, and gc_limit crossings
at BOTH depths from the SAME telemetry — cannot be executed.** Only a
100K-only characterization is possible from the real data that exists. No
speculation is offered here about *why* the telemetry stopped (sampler
script death, a background process not surviving between blocks, or
simply not restarted for the 352K block are all consistent with the
evidence but none is confirmed by anything in the collected files) —
flagging the gap honestly matters more than guessing at its cause.

### 4.2 What CAN be checked at 100K only (from `mem_p02c_m4-1_snap1.jsonl`, `newbuf_m4-1_snap2.log`, `sampler_m4-1_snap1.log`)

**gc_limit crossings: NONE.** Peak (active+cache) observed anywhere in the
100K-block telemetry is **80.57 GB** (step 504, mid-`100k_0A` decode) vs
`gc_limit`=114.557 GB — **34 GB of headroom**, nowhere close to a crossing.
`active_bytes` itself is essentially flat through arm-A's decode window
(79.06–79.31 GB, mean 79.11 GB); `cache_bytes` oscillates in a narrow
0–1.4 GB band with no runaway growth within the window.

**Note on regime mismatch:** production spec-ON telemetry (cited in
DESIGN.md, `CLEAR_CACHE_INTERVAL=64`) showed active ~89.6–92.7 GB at 100K —
noticeably higher than this R1 spec-off measurement's ~79 GB. This is
expected and not a discrepancy to chase: R1 deliberately runs spec-off (no
resident MTP/DSpark draft-model buffers), matching the P3 clean-anchor
measurement regime, not the spec-ON production regime the "3.7x cache
growth rate" figure was measured in. **This means even the 100K side of
R1 cannot be used to directly confirm or refute the spec-ON 3.7x
cache-growth-rate claim — it's a different execution regime.** Combined
with the missing 352.6K telemetry (§4.1), the 3.7x claim is simply
**untested by this data**, neither confirmed nor refuted.

**Fresh-allocation (newbuf) activity is continuous, not rare, even at
100K** — arm A: ~130 events/s (~22 events per completion token); arm B:
~365-420 events/s (~39 events per completion token) in the contamination-
free `100kb` reps. This confirms newbuf events are a routine, ongoing part
of decode at this depth (consistent with DESIGN.md's "fresh allocs
happening continuously at depth" framing) but, without a depth-paired
comparison, says nothing about whether the *rate* grows with depth. The
~1.8x higher per-token newbuf rate in arm B vs arm A at the *same* depth is
a real, interesting arm-comparison observation but is orthogonal to the
depth question this candidate is chasing — not pursued further here in
the interest of the diminishing-returns framing in §6.

**vm_stat "wired pages" swings are dominated by a prefill-phase transient,
not decode-time churn — and are independently known to be an unreliable
proxy.** Splitting `100k_0A`'s sampler data at the probe's own recorded
TTFT boundary (264.1s): wired pages jump from ~3 GB to ~25-28 GB **during
prefill** (building the 100K KV/pool cache), then continue climbing during
decode. Taking the raw vm_stat wired-pages delta over a probe's *full*
window (prefill+decode together, as an early pass of this analysis did
before catching the issue) would misattribute a prefill KV-cache-build
transient to decode-phase allocator churn. Separately, the
`exo-cluster-debugging` skill's own documented pitfall applies directly
here: `vm_stat`'s "Pages wired down" is a known noisy proxy on Apple
Silicon (can show multi-GB swings from driver housekeeping alone, with
zero real memory change) — the more trustworthy signal is MLX's own
`active_bytes`, which stayed flat (79.06→79.31 GB) through the same decode
window while wired-pages swung by tens of GB. Wired-pages telemetry is not
treated as decode-churn evidence in this analysis for exactly that reason.

### 4.3 What CAN be checked at both depths: within-window quintile trend (self-contained in probe data, doesn't need the missing telemetry)

DESIGN.md cites P01a's prior finding (spec-ON production telemetry):
*"352.6K worsens monotonically WITHIN the window (32.0 → 42.4 ms/event
Q1→Q5) while 100K is flat (26.5 → 25.8)."* Each probe JSON stores its own
per-event inter-token gaps, so this specific check — unlike the rest of
Candidate 2 — **can** be run at both depths from data that does exist,
using arm A (spec-off, this session's regime):

| depth | rep | Q1 (ms) | Q2 | Q3 | Q4 | Q5 | Q5−Q1 |
|---|---|---|---|---|---|---|---|
| 100K | `100k_0A` | 37.45 | 36.77 | 36.38 | 36.26 | 36.70 | **−0.75** |
| 100K | `100k_3A` | 38.16 | 47.29 | 46.28 | 38.32 | 36.53 | **−1.63** |
| 352.6K | `352k_0A` | 40.37 | 40.78 | 40.40 | 40.42 | 40.69 | **+0.32** |
| 352.6K | `352k_3A` | 41.49 | 40.77 | 40.50 | 40.52 | 40.78 | **−0.71** |

**Neither depth shows the strong monotonic worsening P01a reported.**
352.6K is essentially flat (±0.3-0.7 ms across the whole window, no
consistent direction) and 100K is, if anything, flat-to-improving (one rep
has a mid-window bump in Q2/Q3 that resolves by Q5, more consistent with a
transient than a trend). Median-based quintiles (outlier-resistant) and a
tail-frequency check (fraction of >3x-median slow gaps per quintile, arm A
352.6K) both confirm this — slow-gap frequency stays in a tight 0.2-1.5%
band across all five quintiles with no escalating trend.

**This does not reproduce the P01a quintile-worsening claim in the regime
tested here.** The honest read is a **scope-limited non-reproduction, not
a clean refutation**: P01a's claim was measured on spec-ON production
telemetry (MTP+DSpark, batched verify, ~16-17 collective calls/token);
this R1 data is spec-off (raw per-layer decode, no batched verify). The
mechanism P01a's claim points at may be specific to the spec-ON
batched-verify execution path (which this experiment did not test) rather
than a universal property of long-decode allocator behavior. Both
possibilities — (a) the effect is spec-mode-specific, or (b) something
about how the two measurements characterize "worsening" differs — are
consistent with the data; neither is confirmed here. What IS established:
**in the spec-off regime, at both depths, arm A does not show the
allocator-churn-driven escalating-within-window signature the design doc
hypothesized as one candidate discriminator.**

### 4.4 Candidate 2 verdict

The core R1 experimental design (cross-depth telemetry comparison — newbuf
rate, cache growth, gc_limit crossings, at both 100K and 352.6K from the
same instrumentation) **could not be executed**: the 352.6K block's
allocator telemetry was never captured or did not survive to the analysis
stage. What 100K-only data shows (no gc_limit crossings, 34 GB of
headroom, flat `active_bytes`, modest bounded cache-buffer usage) is
consistent with "no allocator pressure at 100K" but says nothing about
352.6K by itself. The one check that *does* span both depths (quintile
trend from each probe's self-contained gap data) leans **against** the
"allocator churn escalates within a long deep-context decode window"
mechanism in the spec-off regime — but that regime differs from the one
the original P01a claim was measured in, so this is a partial,
regime-scoped negative, not a full refutation.

**Candidate 2 verdict: DATA GAP PREVENTS THE INTENDED TEST.** The
available partial evidence leans against the within-window-worsening
allocator mechanism (in the spec-off regime), but the design's central
cross-depth comparison is simply missing from the collected data. Net
contribution to explaining the residual: **not established.**

## 5. Data-quality note: arm-B temp=1.0 mitigation only partially held

`run_block.py`'s own comment states MoE-NOP's garbage logits at temp=0
"emit byte-exact block cycles that trip the LONG-period degeneration
detector... at ~72 tokens," and sets temp=1.0 for arm B specifically to
break exact repeats. On the first attempt (`100k_1B`, `100k_2B`), **both**
reps still hit the degeneration kill at exactly 71 streamed events, with
an identical repeating token (`непотпу`) despite temp=1.0 sampling. The
re-run (`100kb_0B`, `100kb_1B`, same config, no code change) completed
cleanly to the full 2000 tokens both times. This is flagged rather than
silently worked around: the temp=1.0 mitigation reduces but does not
reliably prevent the degenerate loop when driving genuinely zeroed
(near-uniform or otherwise pathological) logits through
`mx.random.categorical` — plausibly because a sufficiently degenerate
logit distribution can still collapse into a short random cycle some
fraction of the time. No further root-cause was pursued (out of scope for
this analysis pass; the contaminated reps were correctly excluded and
valid replacements exist). Anyone re-running MoE-NOP arms in the future
should expect this failure mode at roughly this rate and budget a re-run.

## 6. Residual reconciliation and recommendation

### 6.1 How much of +1.67..+2.52 ms/tok is now explained?

**None of it, cleanly.** Neither candidate produced a validated
attribution:

- Candidate 1's validity gate fails under its literal, DESIGN.md-specified
  definition (dB vs bare kernel census). A plausible mechanistic
  explanation for most of the gap exists (shared BatchPoolingCache cost
  dB pays that the census excludes) but formalizing it requires a
  residual-band bookkeeping change this pass does not make unilaterally,
  and even the generous framing leaves a small residual gap. Best
  available (unvalidated) point signal: dA≈dB, mildly suggesting MoE is
  not a large residual contributor — informational, not an attribution.
- Candidate 2's core cross-depth experimental design could not be run at
  all due to a telemetry collection gap at 352.6K. The one same-regime,
  both-depths check available (quintile trend) leans against the
  hypothesized escalating-churn mechanism, but is scope-limited to the
  spec-off regime and does not test the spec-ON regime P01a's original
  claim was made in.

The residual remains **fully open** at +1.67..+2.52 ms/tok. This phase's
useful contribution is narrowing (a plausible, partial explanation for
Candidate 1's gate failure; a partial negative for Candidate 2's escalating
within-window mechanism) and a fresh third independent reproduction of
the total (+4.70 this session vs +4.94 established, within 4.8%) — not a
closure.

### 6.2 Recommendation: diminishing returns reached for now

This session (and the two before it in the same campaign day) has already
ruled out kernel ceiling (no headroom), inter-layer pipelining (refuted/
weak evidence), and collective/skew (closed from both the idle side and
the in-collective side). This phase leaves both remaining candidates
inconclusive rather than closed, but for different reasons than "wrong
hypothesis" — one hit a genuine validity-gate failure with a plausible
but unconfirmed explanation, the other hit a data-collection gap, not a
negative result.

Given:
- the residual is small in absolute terms (+1.67..+2.52 ms/tok against a
  ~35-40 ms/tok decode-time base, roughly 4-7% of total),
- multiple relaunch-gated, multi-hour phases have already been spent on
  it today,
- both remaining paths to a cleaner answer are non-trivial: Candidate 1
  would need either a formal residual-band bookkeeping decision (fold C3
  in on both sides) or a new third "cache-management-NOP" arm to cleanly
  separate MoE from cache cost; Candidate 2 would need a full redo of R1
  with telemetry verified to actually span the 352.6K block this time,

**recommend treating this residual as a documented, bounded, low-priority
open item rather than continuing to chase it in the current session.** If
resumed later, the two concrete, cheap-to-specify next steps are recorded
above (§3.3 caveat 1, §4.4) rather than repeated here as new open threads.

## 7. Cluster state after

**Not touched.** Per the task's explicit instructions, this was a pure
offline analysis pass — no SSH, no relaunch, no probe execution. The
production cluster's live state was not queried, verified, or modified by
this analysis; the supervisor's own verification (both nodes on verbon3
config, real generation smoke-tested) stands as the last known state.

## 8. Artifacts

- Raw data (unmodified, as found): `tmp/p02c-20260829/` — `DESIGN.md`,
  `block_*_meta.json`, `block_*.log`, `probe_*.json`, `newbuf_m4-1_snap2.log`,
  `sampler_m4-1_snap1.log`, `mem_p02c_m4-1_snap1.jsonl`, `r1_*`,
  `run_block.py`, `node_sampler.sh`, `nop_dryrun_2k.json`.
- Analysis scripts (new, this pass, read-only against the above):
  `tmp/p02c-20260829/analysis/*.py` — probe validity triage, Candidate 1
  delta-of-deltas + root-cause digging, Candidate 2 quintile/allocator/
  sampler/telemetry-coverage checks, final residual reconciliation.
