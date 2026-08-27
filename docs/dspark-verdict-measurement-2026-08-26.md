# DSpark/MTP spec-decode verdict — 24-run measurement (2026-08-26)

**Step 4 (measurement) of the corrected spec-decode verdict protocol.** Ran
12 spec-ON + 12 spec-OFF runs of `bench/golden_v1_probe.py` at 100K context
target (`--target-tokens 100000 --max-tokens 2000`, temp=0 greedy), one probe
at a time, ~60s cooldown between runs. Metric: **256-token fixed-window decode
tok/s** (`(N-1)/(t[N-1]-t[0])` over the first 256 decoded tokens), which
amortizes away the prefill+startup difference so it measures pure decode.

Pre-registered decision rule (from `docs/dspark-cs-profile-2026-08-26.md`):
**PROMOTE** iff median fixed-window delta (on−off) ≥ +10% AND lower 95%
bootstrap CI ≥ +5% AND Tier-1 byte-identical AND Gate A clean. **REVERT** iff
median < +5% OR CI includes 0. Marginal (between) → recommend 352.6K block
(not run unless asked).

## Result: REVERT

| metric | value | bar | pass? |
|---|---|---|---|
| median % delta (on−off) | **+1.87%** | ≥ +10% | **FAIL** |
| 95% bootstrap CI (median % delta) | **[−0.82%, +9.45%]** | lower ≥ +5% | **FAIL** |
| CI includes 0? | **YES** | no | **FAIL (REVERT trigger)** |
| Tier-1 byte-identical | 2/3 (partial) | all 7 | **FAIL** |
| Gate A (acceptance = strict argmax) | clean | clean | PASS |

Median % delta (+1.87%) is far below the +10% PROMOTE bar, and the 95% CI
straddles 0 (lower bound −0.82%), which is an independent REVERT trigger.
This **matches the C_s-profile prediction** (step 1): at C_s=3.20 and
a≈2.26, the mechanism is at the break-even knife-edge (a*=2.199, measured
clears a* by 2.6%); a clean fixed-window measurement was always going to land
near break-even, not +10%. The protocol was run to completion for the
record; the arithmetic prediction held.

## The 24-run table

12 spec-ON + 12 spec-OFF, paired by index (time-adjacent: phase ON ran
18:48–19:52, phase OFF ran 19:57–21:10). fixed_window_tok_s = 256-token
fixed-window decode tok/s.

| idx | ON tok/s | ON n_tok | ON finish | OFF tok/s | OFF n_tok | OFF finish | Δ (on−off) | Δ % |
|---:|---:|---:|---|---:|---:|---|---:|---:|
| 0 | 27.029 | 1531 | length | 27.486 | 423 | stop | −0.457 | −1.66% |
| 1 | 30.103 | 435 | stop | 27.295 | 971 | stop | +2.808 | +10.29% |
| 2 | 21.604 | 64 | **null** | 27.757 | 501 | stop | −6.153 | **−22.17%** |
| 3 | 32.293 | 859 | stop | 27.509 | 365 | stop | +4.784 | +17.39% |
| 4 | 27.444 | 224 | stop | 27.015 | 1477 | stop | +0.429 | +1.59% |
| 5 | 28.663 | 356 | stop | 27.187 | 863 | stop | +1.476 | +5.43% |
| 6 | 29.896 | 392 | stop | 27.483 | 366 | stop | +2.413 | +8.78% |
| 7 | 25.989 | 356 | stop | 27.057 | 913 | stop | −1.068 | −3.95% |
| 8 | 28.164 | 242 | stop | 27.573 | 448 | stop | +0.591 | +2.14% |
| 9 | 28.437 | 482 | stop | 28.233 | 345 | stop | +0.204 | +0.72% |
| 10 | 27.616 | 291 | stop | 27.609 | 569 | stop | +0.007 | +0.02% |
| 11 | 29.838 | 983 | stop | 27.099 | 945 | stop | +2.740 | +10.11% |

Per-arm fixed-window tok/s:

| arm | n | median | IQR | min | max |
|---|---:|---:|---|---:|---:|
| spec-ON | 12 | 28.30 | [27.44, 29.90] | 21.60 | 32.29 |
| spec-OFF | 12 | 27.49 | [27.19, 27.61] | 27.02 | 28.23 |

The ON arm has ~3.7× wider spread than OFF (range 10.7 vs 1.2 tok/s). This
is the spec-decode cycle's known bimodality — some cycles verify a long
draft chain (high tok/s), some reject early (low tok/s). The OFF arm is
tight (sequential decode, no cycle-to-cycle acceptance variance). Run
#02-ON (64 tokens, `finish_reason: null`) is the anomaly: the model
emitted a reasonable 64-token summary then stopped with no finish flag —
the same EOS-bypass/early-stop family documented in the Stage-2c campaign
(`references/dspark-campaign-results-2026-08-26.md`), where the spec verify
path applies no logits processors and the raw-argmax bonus can be EOS.
Excluding #02, the median % delta is +2.14% — still well below +10%.

## Tier 2 quality (natural-EOS completions)

scipy not available in the runner env → descriptives only (no KS test).

| metric | spec-ON | spec-OFF | delta |
|---|---|---|---|
| finish: stop / length / other | 10 / 1 / 1 | 12 / 0 / 0 | ON has 1 anomaly |
| n_tokens median (all) | 374 | 535 | ON −161 (shorter) |
| n_tokens IQR | [291, 859] | [423, 945] | — |
| rep16-gram fraction median | 0.0186 | 0.0688 | ON −0.050 (lower) |
| loop flags (16-gram repeated ≥3×) | 1/12 | 1/12 | tie |

**Interpretation.** The ON arm produces shorter completions (median 374 vs
535 tokens) and contains one anomalous null-finish run (64 tokens) —
consistent with the known spec-path EOS-emission tendency (the verify
forward applies no logits processors, so the raw-argmax bonus token can be
EOS, stopping early). The lower rep16-gram fraction for ON is an artifact
of shorter completions (fewer tokens → fewer 16-gram windows → less
opportunity for overlap), NOT a quality improvement. The 1 loop flag each
(on#9, off#4) is task-structural, not degeneration: the golden probe
summarizes a repetitive corpus, so 16-gram repetition at maxrep=3 is the
corpus structure itself, not a model loop. **Net Tier-2 signal: neutral-to-
negative for ON** (shorter, 1 anomaly), consistent with the REVERT decision.

## Decision: REVERT to spec-off

All three independent decision inputs point the same way:
1. **Throughput**: median +1.87% ≪ +10% bar; CI [−0.82%, +9.45%] includes 0.
2. **Tier-1 byte-identity**: 2/3 (partial) — the shipped MoE-rowseq residual
   (0.023%/row) deterministically flips a near-tie, so spec-ON is NOT a
   bit-identical trajectory.
3. **C_s arithmetic (step 1)**: at C_s=3.20 / a≈2.26 the mechanism is at
   break-even; +10% is an arithmetic impossibility without verify-path
   batching (the real fix direction).

The cluster is restored to the production spec-off pattern
(`EXO_SPECULATIVE=0 EXO_DSV4_MTP=0 EXO_DSV4_HC_COLLAPSE_KERNEL=1`, DSpark
head loaded but not drafting) — screen `exorun_specoff` on both nodes, env
verified via `ps eww`. This matches the `dspark_prod` / `dspark_revert`
pattern; no further relaunch needed (the measurement phase already left the
cluster in the production state).

## Artifacts

- `/tmp/ab/protocol/summary_on.jsonl`, `summary_off.jsonl` — per-run summaries
- `/tmp/ab/protocol/run_{on,off}_{00..11}.json` — full per-token captures
- `/tmp/ab/protocol/stats_result.json` — statistics + decision
- `/tmp/ab/protocol/tier2_result.json` — Tier-2 quality
- `/tmp/protocol_stats.py`, `/tmp/tier2_quality.py` — analysis scripts
- Prior steps: `docs/dspark-cs-profile-2026-08-26.md` (step 1, C_s=3.20),
  `docs/dspark-tier1-byte-identity-2026-08-26.md` (step 3, 2/3 byte-identical)
- Pre-reg doc: `docs/dspark-mtp-ab-preregister-2026-08-25.md` (this protocol's
  decision rule supersedes the Stage-2/3 bars there for the corrected
  fixed-window protocol)