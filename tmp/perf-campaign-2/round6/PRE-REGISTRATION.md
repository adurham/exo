# CAMPAIGN 2, ROUND 6 — PRE-REGISTRATION

**Committed BEFORE any Phase-0 or sweep measurement.** Everything below is fixed in advance.
If a decision rule is not written here, it was not pre-registered, and any post-hoc rule must
be labelled as such in the report.

**Date:** 2026-09-04 | **PM:** round-6 delegation
**Predecessor:** `tmp/perf-campaign-2/round5/REPORT.md` (NO SHIP — throughput never validly
measured; acceptance γ=3 = 1.820/3, γ=4 = 2.420/4, +33%).

---

## 0. THE RULE OF THIS ROUND: no new harness

Round 5 built a new throughput harness and it produced believable-looking garbage twice
(a burst-timed stream reading ~14x physically impossible, then a CHUNK rate mislabelled as a
token rate that passed its own 38–55 t/s sanity check). This round builds nothing.

**Primary metric: `stats.generation_tps`** — decode-only, `perf_counter`-timed *inside the
generator* at `batch_generate.py:1255-1257`, returned by the server in the completion
response. It cannot be fooled by burst delivery or chunking because it is not measured on the
client at all.

**Harness: `bench/long_decode_probe.py`**, the exact probe P12–P15 used, reused as-is with its
`decode_sample_trustworthy` gate.

### The one permitted code change, and its strict limit
`long_decode_probe.py` as written computes `decode_tps = completion_tokens / wall_clock`
CLIENT-SIDE. That is precisely the class of number round 5 got wrong. The probe does not
currently capture the server's `stats` object.

**Permitted:** add read-only capture + printing of the server-emitted `stats` field (and
`prompt_tokens`, already present) to the probe's result dict.
**FORBIDDEN:** any new timing arithmetic, any client-side rate computation used as a decision
input, any change to how the request is issued, any change to prompt construction other than
the depth argument. If `stats` is not obtainable on the streaming path, the fallback is to
issue the SAME request non-streamed and read `stats` from the response body — still no new
timing math.

The probe's pre-existing client-side `decode_tps` is retained ONLY as a cross-check and is
**never** a decision input. Recorded in every artifact next to `generation_tps`.

---

## 1. Measurement protocol (fixed)

Per arm, at ~89K depth:
- Relaunch the cluster (`EXO_SPECULATIVE_GAMMA` is an import-time read; it cannot be changed live).
- Verify the gamma value on the **real runner PIDs** via `ps eww` on **both** nodes before probing.
- 1 warmup rep — **discarded, never reported as data**.
- n = 3 measured reps.
- `decode_sample_trustworthy == true` required on **all 3**. Any false → rep invalid, re-run.
- `prompt_tokens >= 85000` required on **all 3**, printed for every rep. The probe's depth
  argument uses a ~4 chars/token heuristic; the TARGET IS NOT THE ACHIEVED DEPTH. Round 5's
  `--depth 89000` produced 62K. Calibrate the argument against measured `prompt_tokens`.
- `EXO_DSV4_MTP_LOG_INTERVAL` set on every arm so counted acceptance is recorded alongside
  throughput (see §4 for the confound this introduces and its pre-registered handling).

**Reported per arm: median AND full range (min, max) of the 3 reps. Ranges, never bare means.**

Derived `(1+a)/cycle` is OPTIONAL this round and is NOT a decision input: obtaining a cycle
time requires `EXO_DSV4_MTP_PROFILE`, which brackets `mx.eval` every cycle
(`dsv4_mtp.py:240-243`) and would corrupt the very t/s under test. Measured `generation_tps`
is the decision input.

### Arm order (fixed)
**γ=3 (A) → γ=4 → γ=3 (B) → γ=2 → γ=5.**
Both γ=3 boots are mandatory. They bracket the sweep and their spread IS this round's
boot-variance reading. Round 5 skipped the closing bracket; that alone blocked its ship claim.

---

## 2. Phase 0 — mechanical calibration gate

Run on the CURRENT production boot (γ=3). **No relaunch for this.** Both numbers from
`stats.generation_tps`, with `prompt_tokens` printed.

| probe | reference | pass band (reference ± 6 t/s known boot variance) |
|---|---|---|
| ~2K prompt | ~26.6 t/s (round-4 restored-config smoke) | **20.6 – 32.6 t/s** |
| ~89K prompt | 30–34 t/s (record band) | **24.0 – 40.0 t/s** |

**If either lands outside its band: STOP. No sweep number is believed and no arm runs.**
The calibration table is reported FIRST in REPORT.md regardless of outcome.

Rationale for the ±6: the task specifies the known ~6 t/s boot variance as the tolerance. The
gate is deliberately generous — it is a check that the *measurement path* is sane (i.e. not
reading a chunk rate or a burst artifact), not a tight performance assertion.

---

## 3. THE BAND (pre-registered verbatim; measured `stats.generation_tps` only)

Definitions, fixed before data:
- `g3_reps` = all 6 measured reps across γ=3 (A) and γ=3 (B).
- `g3_union_max` = max(`g3_reps`); `g3_union_min` = min(`g3_reps`).
- **`g3_spread` ≡ |median(γ=3 A) − median(γ=3 B)|** — the boot-to-boot shift in central
  tendency. (The union width is also reported, but the SHIP test uses the median difference.
  Fixing this now so it cannot be chosen after seeing the data.)
- `arm_min` = min of the arm's 3 reps; `arm_max` = max.

**An arm SHIPS iff BOTH hold:**
1. `arm_min > g3_union_max` — the arm's 3-rep range lies ENTIRELY ABOVE the union of both
   γ=3 ranges; AND
2. `(arm_min − g3_union_max) > g3_spread` — the gap exceeds the boot variance just measured,
   i.e. the effect is larger than the noise floor of a reboot.

**Verdicts:**
- Both hold → **SHIP CANDIDATE** (proceeds to the Phase-2 quality gate; the quality gate can
  still disqualify it).
- Ranges overlap (`arm_min <= g3_union_max`) → **"inside boot variance, not shippable."**
- `arm_max < g3_union_min` → **CLOSED** (arm is worse).

**The I8 question, pre-registered:** if γ=4's +33% counted acceptance does NOT show up in
`generation_tps`, the report must state where it went — the expected sink is cycle-time growth
from the extra draft/verify row, and the magnitude implied by the measured t/s and acceptance
must be stated explicitly rather than hand-waved.

---

## 4. PRE-REGISTERED CONFOUND: `EXO_DSV4_MTP_LOG_INTERVAL`

This is the only env delta between the current production boot (Phase 0) and every swept arm.
The operations record carries an explicit warning against it during a throughput bench:

> `EXO_DSV4_MTP_LOG_INTERVAL=50` appears to re-trigger JACCL stalls — diagnostic-only env, do
> NOT use during a reported-tok/s bench (May 15 2026: champion env + log_interval gave clean
> warmup 31.76 t/s but iter1 10.57 / iter2 6.28).

The task requires it on every arm. It rides all arms equally, so a *stable* offset would leave
the A/B valid. The May-15 evidence is not a stable offset — it is **instability** (31.76 → 10.57
→ 6.28). That is the actual risk, and it is checked, not assumed away:

**Check C1 (level):** γ=3 (A)'s 89K median must be within ±6 t/s of the Phase-0 89K median.
**Check C2 (stability):** every arm's 3-rep range width must be ≤ 8 t/s (~25% of a ~32 t/s
level). A wider range means the arm is bistable and its median is meaningless.

**If C1 fails, or C2 fails on any arm:** the working hypothesis is LOG_INTERVAL-induced
instability. Remedy, in priority order:
1. Re-run the offending arm WITHOUT `EXO_DSV4_MTP_LOG_INTERVAL`. If it recovers, run the
   remaining sweep without it and report counted acceptance as **not measured this round**
   (round 5 already measured it; throughput is this round's missing half and takes priority).
2. If budget forbids a re-run, the arm is declared **INVALID**, not merely "slow" — an
   unstable measurement is not evidence in either direction.

Recording this in advance so that a bad number cannot be retro-explained after the fact.

---

## 5. Phase 2 — quality gate (runs ONLY on a ship candidate; all four mandatory)

| gate | pass criterion | disqualifies on |
|---|---|---|
| `bench/ab_probe_tier1.py` | **7/7** | anything < 7/7 |
| needle exact-match @ 89K (`bench/quality_probe_dsv4.py`, round-4 harness) | exact string hit | miss |
| **temp=0 byte-identity vs γ=3, 3 prompts** | **byte-identical on all 3** | **any divergence** |
| 2 DSML tool-call prompts | both parse as well-formed tool calls | either malformed |

**Bit-equivalence is a HARD gate.** Speculative decoding is mathematically lossless: the
verify step guarantees the accepted token sequence is exactly what the target model would have
emitted alone. Therefore **any** byte divergence at temp=0 is a verify bug, not a tolerance
question, and it disqualifies the arm outright regardless of throughput.

---

## 6. Phase 3 — ship or hold

- **Clean candidate:** change the `EXO_SPECULATIVE_GAMMA` default in `start_cluster.sh` with a
  dated comment citing rounds 5+6 and the counted acceptance; relaunch; verify healthy on the
  new config.
- **No candidate:** leave γ=3; cluster healthy and verified.

Either way the cluster ends HEALTHY, verified by the PM on real PIDs — not self-reported by a
worker.

---

## 7. Degrade path (pre-registered, per the task's time box)

Time box: **4 hours**, 5 relaunches budgeted. If time runs out after γ=3(A), γ=4, γ=3(B):
**that is a valid round.** The γ=4 decision is made on those three arms and γ=2 / γ=5 are
reported as NOT RUN. γ=2 is the first drop, γ=5 the second. The two γ=3 boots and γ=4 are
never dropped — without both brackets there is no boot-variance reading and therefore no band.

---

## 8. Hard constraints

1. **NO new harness.** See §0 for the single narrowly-scoped permitted edit.
2. **NO pushes to any remote.** Local commits only; the supervisor pushes.
3. **Never `git add -A`. Never `git stash`.**
4. Ranges, never bare means. `prompt_tokens` printed on every rep.
5. Bit-equivalence is a hard gate.
6. `gh` requires `--repo adurham/exo`.
7. Cluster HEALTHY at the end — PM-verified.

## 9. Explicit predictions (so this round can be wrong in public)

Stated before any measurement:
1. Phase 0 calibration **passes** — `stats.generation_tps` is the number the real-usage study
   already validated (34.06 t/s at matched depth vs bench 31.84), so the path is expected sane.
2. γ=4 lands **above** γ=3 on median but I expect the gap to be **much smaller than the +33%
   acceptance gain** — most of the acceptance win is expected to be eaten by cycle-time growth
   from the extra draft/verify row.
3. The most likely single outcome is **"inside boot variance, not shippable"** — i.e. γ=4 wins
   on median but its range overlaps the γ=3 union. Round 4's cycle model implies ≈1.14x, which
   would clear; the acceptance-to-throughput transfer is exactly the untested link.
4. γ=2 below γ=3; γ=5 at or below γ=4 (γ is clamped at `block_size = 5`, so γ=5 is the
   reachable ceiling and nothing above it is ever worth a boot).
