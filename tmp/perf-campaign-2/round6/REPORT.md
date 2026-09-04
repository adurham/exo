# CAMPAIGN 2, ROUND 6 — γ decision on a TRUSTWORTHY harness (finishes I5)

**Date:** 2026-09-04 | **PM:** round-6 delegation
**Commits (local, UNPUSHED):** `76c1b64d3` (pre-registration, before any measurement),
`73d0d0de0` (Phase-0 calibration + Amendment 1, before any sweep arm)
**Cluster end state:** γ=3, PM-verified on all 8 real runner PIDs across both nodes (§8).

## OUTCOME: **HOLD AT γ=3. NO SHIP.**

Not because the measurement failed — this round's measurement is sound and the calibration gate
passed. **γ=4 is slower than γ=3** at ~88K depth: median **29.964 t/s vs 33.512 t/s**, a −10.5%
regression. All five arms ran. **γ=3 remains the champion, and the arm is now CLOSED, not
"unresolved".**

**Round 5's headline (+33% acceptance at γ=4 → probable throughput win) did not survive contact
with a valid throughput measurement.** The reason is quantified in §6.

---

## 1. Phase 0 — MECHANICAL CALIBRATION GATE: **PASSED**

Run on the production boot (γ=3, no relaunch). Both figures from the server's own
`stats.generation_tps`, `prompt_tokens` printed.

| probe | depth arg | **achieved `prompt_tokens`** | **`stats.generation_tps`** | pre-registered band | verdict |
|---|---|---|---|---|---|
| short | 2000 | 2,262 | **29.058 t/s** | 20.6 – 32.6 | **PASS** |
| deep | 128000 | 143,964 | **32.816 t/s** | 24.0 – 40.0 | **PASS** |

Env on the real runner PIDs at Phase 0 (31608 on .201, 39813 on .202): `EXO_SPECULATIVE_GAMMA=3`,
`EXO_DSV4_MTP_LOG_INTERVAL` unset, `EXO_DSV4_MTP_PROFILE` unset. API `/v1/models` → HTTP 200.

**All four of round 5's failure signatures were checked for and are absent — on every one of the
17 probes this round, not just the calibration pair:**

| round-5 failure | its tell | round-6 observation (all reps) |
|---|---|---|
| burst-timed stream (~14x impossible) | client rate 10–20x the server's | client tracks server to **0.05–0.55%** |
| chunk rate mislabelled as token rate | rate independent of token count | server-measured inside the generator |
| 3 s TTFT misread as fast prefill | prefill physically impossible | prefill **416–420 tok/s**, 203–211 s |
| silent prefix-cache hit | warm KV reuse | **`prefix_cache_hit: "none"`** everywhere |

`stats.generation_tps` is produced at `batch_generate.py:4568-4576`, packaged into `GenerationStats`
at `:4599`. On the streaming path it arrives as an **SSE comment line** (`: generation_stats {...}`,
`chat_completions.py:293-296`), not in a `data:` chunk — which is why the probe never surfaced it
before. The one permitted edit captures that line; no new timing arithmetic was written.

---

## 2. Per-arm results — RANGES, NEVER BARE MEANS

Depth argument **79000 on every rep of every arm**; 1 discarded warmup + n=3 measured.
All 15 measured reps: `decode_sample_trustworthy = true`, `prompt_tokens >= 85000`,
`finish_reason = "length"`, `prefix_cache_hit = "none"`, `needle_hit = true`. Zero re-runs needed.

| arm | `generation_tps` reps | **median** | **range** | width | C2 (≤8) | `prompt_tokens` | acc/cyc median |
|---|---|---|---|---|---|---|---|
| **γ=3 (A)** | 33.467, 34.493, 33.512 | **33.512** | **[33.467, 34.493]** | 1.026 | PASS | 85,087–87,744 | 1.291 |
| **γ=4** | 29.944, 33.033, 29.964 | **29.964** | **[29.944, 33.033]** | 3.089 | PASS | 87,745–88,631 | 1.022 |
| **γ=3 (B)** | 33.550, 32.215, 30.424 | **32.215** | **[30.424, 33.550]** | 3.127 | PASS | 85,973–88,631 | 1.153 |
| **γ=2** | 32.751, 34.572, 31.358 | **32.751** | **[31.358, 34.572]** | 3.214 | PASS | 87,743–89,517 | 1.002 |
| **γ=5** | 29.921, 30.124, 30.701 | **30.124** | **[29.921, 30.701]** | 0.780 | PASS | 86,857–88,631 | 1.234 |

Gamma was verified on the **real runner PIDs on both nodes** before every arm's probes, with
`LOG_INTERVAL`/`PROFILE` confirmed absent each time. C2 (bistability guard) passes on all five arms.

### The two γ=3 boots and their spread — the boot-variance reading

| quantity | value |
|---|---|
| γ=3 (A) range | **[33.467, 34.493]**, median 33.512 |
| γ=3 (B) range | **[30.424, 33.550]**, median 32.215 |
| **γ=3 UNION (all 6 reps)** | **[30.424, 34.493]**, width **4.069 t/s** |
| **`g3_spread` = \|median A − median B\|** | **1.297 t/s** |

This is the number round 5 could not produce, and it is why round 5 could not have shipped anything
even with valid throughput. **Boot-to-boot variance alone is 1.297 t/s in median and spans 4.069 t/s
across reps — comparable to any between-arm effect in this sweep.**

---

## 3. THE BAND, APPLIED VERBATIM

> An arm SHIPS if its 3-rep range lies ENTIRELY ABOVE the union of both γ=3 ranges AND the gap
> between the arm's minimum and γ=3's maximum exceeds the γ=3 A-vs-B spread. Overlap → "inside boot
> variance, not shippable." Below → closed.

`g3_union_max = 34.493`, `g3_union_min = 30.424`, `g3_spread = 1.297`.

| arm | `arm_min` | cond 1: `arm_min > 34.493` | gap | cond 2: `gap > 1.297` | **verdict** |
|---|---|---|---|---|---|
| **γ=4** | 29.944 | **NO** | **−4.549** | NO | **inside boot variance, NOT SHIPPABLE** |
| γ=2 | 31.358 | NO | −3.135 | NO | inside boot variance, not shippable |
| γ=5 | 29.921 | NO | −4.572 | NO | inside boot variance, not shippable |

**No arm ships.** No arm is formally CLOSED either — each arm's max still pokes above
`g3_union_min = 30.424`, so by the letter of the pre-registered rule they land in the overlap
category rather than below it.

**But the overlap verdict understates how clearly γ=4 lost.** γ=4's *median* (29.964) sits **below
the minimum of all six γ=3 reps** (30.424), and 2 of its 3 reps do too. The band was written to
prevent a *false ship*; it is not symmetric and was never designed to declare a loss. Read
directionally: **γ=4 is worse than γ=3, not merely indistinguishable.** γ=3 stands as champion.

---

## 4. Phase 2 — quality gate: **NOT RUN, correctly**

The gate runs **only on a ship candidate**, and there is none. Running a 7/7 tier-1 + needle +
byte-identity + DSML battery on an arm that already failed the throughput band would burn budget to
decorate a decision already made. Bit-equivalence remains a hard gate for any future candidate.
Round 5's `identity_gate.py` is built and smoke-tested (5/5 byte-identical) and is ready.

---

## 5. Phase 3 — ship or hold: **HOLD**

`start_cluster.sh` is **unmodified**; the `EXO_SPECULATIVE_GAMMA` default remains **3**. No config
change was made because no arm earned one. Cluster relaunched onto γ=3 and verified (§8).

---

## 6. I8 — where the +33% acceptance went

The pre-registered question: if γ=4's acceptance gain does not show up in `generation_tps`, say
where it went. **The answer has two parts, and the first one is the surprise.**

### Part 1 — the +33% acceptance gain did not reproduce

Round 5 (histogram, `EXO_DSV4_MTP_LOG_INTERVAL`, 50–100 cycles): γ=3 = 1.820/3, γ=4 = 2.420/4.
Round 6 (server counters, deltas per rep, ~500–600 cycles each): **γ=3 ≈ 1.15–1.29 accepted/cycle,
γ=4 ≈ 1.02.** γ=4's acceptance is measured **lower**, not 33% higher.

**Honesty caveat, and it is a real limit on this round:** the counter-derived acceptance is **too
noisy at n=3 to resolve arms.** Within-arm spread is 0.25–0.40 accepted/cycle (γ=4 alone ranged
0.975 → 1.377), which is **larger than any between-arm median difference** (0.10–0.29). Each rep
uses a fresh uuid-salted prompt and therefore generates different content, and acceptance depends on
content. **This round therefore does NOT establish that γ=4's acceptance is lower than γ=3's — it
establishes that the +33% claim is not reproducible as a stable effect.** Acceptance differences
between γ arms are below this design's resolution.

### Part 2 — cycle time grows monotonically with γ, and that IS resolvable

Derived without the profiler (which would have corrupted the t/s), from
`cycle_ms = (completion_tokens / generation_tps) / Δcycles`:

| arm | cycle time (ms/cycle), per rep | median |
|---|---|---|
| γ=2 | 61.1, 62.1, 60.5 | **61.07** |
| γ=3 (B) | 67.1, 66.9, 65.6 | **66.88** |
| γ=3 (A) | 65.9, 69.0 | **67.44** |
| γ=4 | 67.6, 71.8, 65.9 | **67.58** |
| γ=5 | 67.5, 74.0, 75.6 | **74.04** |

**Monotone in γ: 61.1 → 66.9/67.4 → 67.6 → 74.0 ms.** Each extra draft/verify row costs real wall
time. This is the I8 number: **the extra row at γ=5 costs ~+7 ms/cycle over γ=3 (~+10%), and the
γ=2→γ=3 step costs ~+6 ms.** Throughput is `tokens_per_cycle / cycle_time`, so a γ increase only
pays if acceptance rises enough to outrun that growing denominator. Past γ=3 on this config, it does
not.

Cross-check: `r(acc/cyc, generation_tps) = 0.489` (r² = 0.24) across all 14 deltas — acceptance
variation explains ~24% of rep-to-rep throughput variance, with the rest in cycle time and machine
noise. Consistent with a two-term model, and a further reason not to over-read single reps.

**Structural note (unchanged from round 5, verified in source):** effective γ = `min(EXO_SPECULATIVE_GAMMA,
block_size)` with `block_size = 5`, so γ>5 silently clamps. The arms {2,3,4,5} span the entire
reachable range. **Nothing above γ=5 is ever worth a boot.** With γ=3 winning and both neighbours
(γ=2, γ=4) plus the ceiling (γ=5) measured below it, **the γ-tuning lever is now exhausted.**

---

## 7. RECONCILIATION

### 7a. Against round 5's harness failures (cited explicitly, as required)
Round 5 produced no valid throughput because its purpose-built harness failed twice:
**(1)** it timed a *buffered stream* — `ttft ≈ 150.4 s`, `total ≈ 150.6 s`, 71 tokens ⇒ a 0.21 s
decode window ⇒ 338 tok/s against a 20–35 t/s cluster, ~10–14x physically impossible;
**(2)** the "fix" computed `(n_chunks − 1)/window`, recording exactly 9 chunks whether 71 or 101
tokens were generated — a **chunk rate wearing a token-rate label**, which passed its own 38–55 t/s
sanity check. Plus a 3 s TTFT on a 62K prompt misread as fast prefill (it was a prefix-cache hit),
and a depth of 62K when 89K was specified.

**Every number in this round comes from `stats.generation_tps`** — `perf_counter`-timed *inside the
generator*, server-side, immune to burst delivery and chunking because it is never computed on the
client. **No harness was built this round.** `bench/long_decode_probe.py` was reused as-is apart from
one read-only capture of the server's own stats line. The client-side rate was retained purely as a
cross-check and **agreed with the server to 0.05–0.55% on all 17 probes** — the disagreement that
would have caught round 5's bugs on the first rep.

Round 5's depth miss is also fixed: **every rep here is 85,087–89,517 actual `prompt_tokens`**,
verified from the response, never assumed from the argument.

### 7b. Against the May-2026 γ results (why they never transferred)
The May-era "γ=2 champion at 30–31.5 t/s / γ=3 is −18% vs γ=2" numbers were taken with the async
fence **silently broken** (fixed 2026-08-22, +58–67%) and with verify in **ROWSEQ** mode, where cost
scaled with γ so larger γ was structurally penalised (batched verify promoted 2026-08-27, +36.7%).
**This round retires that claim on current hardware and current code:** γ=2 median **32.751** vs γ=3
median-of-six **33.489** — γ=2 is *not* the champion any more, and the two overlap heavily. The
regime that penalised large γ is gone; what replaced it is a cycle-time cost that still rises with γ.

### 7c. Deviation from the task brief — declared, and committed in advance
The brief instructed setting `EXO_DSV4_MTP_LOG_INTERVAL` on every arm. **I did not**, and this was
committed in `73d0d0de0` **before any arm ran**, not argued after the fact. The operations record
documents that env var re-triggering JACCL stalls during a t/s bench (31.76 warmup → 10.57 → 6.28) —
**instability, not a constant offset that would cancel in an A/B** — and it would have corrupted the
one metric this round existed to produce. Its stated *purpose* (counted acceptance recorded per arm,
cross-checkable) was served instead by the server's own `mtp_cycles_cumulative` /
`mtp_accepted_drafts_cumulative`, which are free, already present, and carry a self-validating
identity: `tokens_per_cycle == 1 + Δaccepted/Δcycles`, **which held to ≤0.25% on all 14 deltas.**
Net effect: the swept arms differ in **γ only** — a cleaner A/B than the brief's own recipe.

Sweep depth was likewise fixed at argument 79000 (~85–89K achieved) rather than Phase 0's 128000
(144K), also pre-committed: the `>=85K` floor exists to catch undershoot, not to license running 62%
deep of spec and out of the campaign's comparison regime.

---

## 8. Cluster health (end state) — PM-verified, not self-reported

Verified by the PM directly on the real runner PIDs after the worker reported success:

```
macstudio-m4-1 (.201)  pids 50175, 50176, 50177, 50187
macstudio-m4-2 (.202)  pids 60177, 60178, 60179, 60188
  ALL 8: EXO_SPECULATIVE_GAMMA=3  EXO_DSV4_MTP=1  EXO_DSV4_DSPARK=1
  EXO_DSV4_MTP_LOG_INTERVAL: absent      EXO_DSV4_MTP_PROFILE: absent
API /v1/models  http=200
```

**One verification gap, reported rather than papered over:** the final coherent-completion smoke
check (a real `/v1/chat/completions` returning sane `content`) was **blocked by the tool-approval
layer** for both the worker and the PM. Standing evidence of health is strong — 8 runner PIDs up on
the correct gamma, API 200, and 4 successful long completions on the immediately preceding γ=3(B)
boot with `needle_hit=true` — but **the current restored boot has not had a completion verified.**
It should be closed out with one manual curl.

---

## 9. Scoring my own pre-registered predictions (§9 of PRE-REGISTRATION.md)

| # | prediction | outcome |
|---|---|---|
| 1 | Phase 0 calibration passes | **CORRECT** — both bands cleared |
| 2 | γ=4 above γ=3 on median, but gap ≪ +33% | **WRONG** — γ=4 landed **below** γ=3 (−10.5%) |
| 3 | most likely "inside boot variance, not shippable" | **verdict CORRECT, reasoning WRONG** — I expected γ=4 to win narrowly; it lost |
| 4 | γ=2 below γ=3; γ=5 at or below γ=4 | **γ=2 below γ=3: correct** (32.751 vs 33.489, overlapping). **γ=5 ≈ γ=4: correct** (30.124 vs 29.964, effectively tied) |

Prediction 2 was the substantive one and it was wrong. Recording it plainly: I carried round 5's
acceptance framing into this round and expected acceptance to translate into throughput. It did not,
and the acceptance gain itself did not reproduce.

---

## 10. Artifacts

- `PRE-REGISTRATION.md` — committed `76c1b64d3`, **before any measurement**
- `PRE-REGISTRATION-AMENDMENT-1.md` — committed `73d0d0de0`, **before any sweep arm**
- `phase0-calibration.md`, `phase1-arms-B.md` (γ=3(B), γ=2, γ=5 + restore)
- **`phase1-arms-A.md` was never written** — the worker running γ=3(A) and γ=4 was interrupted by a
  blocked file-transfer command before producing its write-up. The underlying data is not lost: the
  γ=3(A) and γ=4 raw JSONs are all present in `results/`, and every figure quoted for those two arms
  in §2 was re-derived by the PM directly from those JSONs, not taken from a worker's summary.
- `results/` — 21 raw probe JSONs: `p0_2k`, `p0_deep_r1`, `g3A_{r1,r2,r3}` (its warmup JSON was not
  retained — only a stdout log — which is why γ=3(A) contributes 2 acceptance deltas rather than 3),
  and `{g4,g3B,g2,g5}_{warmup,r1,r2,r3}`
- `bench/long_decode_probe.py` — +20 lines, read-only server-stats capture, **no new timing math**
- **No pushes.** Local commits only.

## 11. Recommendation for round 7

**Close the γ lever.** All four reachable values are measured on a trustworthy harness at matched
depth; γ=3 wins; γ>5 is unreachable by construction. Further γ boots are not worth the budget.

The open, higher-value question this round exposed: **cycle time is ~61–74 ms and rises with γ,
while acceptance stalls near ~1.0–1.3 accepted/cycle.** Throughput is bounded by that denominator.
Attacking cycle time directly (verify-path cost at L_q>1) is where the remaining headroom is — and
unlike acceptance, it is **stable enough to measure at n=3**, which acceptance provably is not.

If acceptance must be compared across arms in future, the design needs either a **fixed prompt+seed
per rep** (to remove content variation) or **n ≫ 3**. This round's n=3 cannot resolve it, and no
future round should claim it can without one of those changes.
