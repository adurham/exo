# CAMPAIGN 2, ROUND 5 — Speculative γ re-tune against the TRUE acceptance (I5)

**Date:** 2026-09-03/04 | **PM:** round-5 delegation
**Commits (local, UNPUSHED):** `62aeca65e` (pre-registration, committed BEFORE any measurement)
**Cluster end state:** restored to shipped γ=3, PM-verified healthy (see §7).
**Outcome: NO SHIP. γ=3 stands.** Not because an arm lost — because **decode throughput
was never validly measured.** Acceptance was, and it is the round's real result.

---

## 0. Verdicts up front

1. **Phase 0 histogram: the acceptance histogram had NEVER BEEN EMITTED** — not in 48h,
   not ever on the current config. `EXO_DSV4_MTP_LOG_INTERVAL` is an import-time read and
   was unset on all 8 live runner PIDs. It rode the Phase-1 relaunch, exactly as the task
   specified. **We now have the first real acceptance measurement of this config.**
2. **TRUE acceptance at γ=3, measured: `mean_accept = 1.820/3`** (`hist=0:11,1:9,2:8,3:22`,
   n=50 cycles). Not 1.411, not 1.73. Both prior figures are superseded.
3. **My pre-registered ordering was CONTRADICTED, and that is the finding.** I predicted
   a cliff past position 3 and that no arm could clear the band. The measured per-position
   curve **does not cliff** — the conditional continue-probability is FLAT at 73–83% out to
   position 4. γ=4 raised acceptance +33% (1.820→2.420 tok/cycle).
4. **γ=4 is a genuine SHIP CANDIDATE on acceptance, but it CANNOT BE SHIPPED on this
   evidence** because the measured-t/s half of the pre-registered band was never validly
   obtained. Per the band's own wording, both conditions are required. **Reported as
   unresolved, not as a win.**
5. **The throughput harness is broken in two distinct ways** (§4). Both are documented with
   the arithmetic that proves it, so round 6 does not rediscover them.

---

## 1. Phase 0 — the histogram (zero cluster cost)

| Question | Answer | Evidence |
|---|---|---|
| Was the histogram emitted in the last 48h? | **NO — never emitted at all** | 0 matches for `accept_hist=\|mean_accept=\|k_hist\|bypos` in both nodes' live logs and all of round1–4 |
| Why? | `EXO_DSV4_MTP_LOG_INTERVAL` absent on all 8 runner PIDs | `ps eww`, 4 PIDs × 2 nodes |
| Can it be turned on live? | **No — import-time read** | `_LOG_INTERVAL = int(os.environ.get(..., "0"))`, `dsv4_mtp.py:118`, module level |
| Does `bypos` exist? | **No, nowhere, ever** | needs `EXO_DSV4_SPEC_SHADOW=1` (`_ShadowStats`, `dsv4_mtp.py:552-654`, emit `:4521`); set in no launcher on either node |

Per the task's instruction for exactly this case, the interval **rode the Phase-1 relaunch**;
no boot was spent on it alone.

### Two structural facts I verified in source (they shaped the prediction)
`dsv4_mtp.py:3946-3955` — **effective γ = `min(EXO_SPECULATIVE_GAMMA, block_size)`, and
`block_size = 5`.** The arms {2,3,4,5} span *exactly* the reachable range; γ>5 silently
clamps. Nothing above 5 is worth a boot, ever.

`dsv4_mtp.py:3913-3914` — "the DSpark head is a 3-stage block (n_stages = n_mtp_layers = 3)
**trained for width-3 draft/verify**. block_size is 5 (anchor + 4)." I used this as the
structural substitute for the missing `bypos` data to predict a cliff past position 3.
**The measurement refuted that inference** (§3).

### A hypothesis I formed and killed before it cost a boot
I suspected the 85%-efficiency stale histogram vs 47% production was `EXO_DSV4_MTP_DEDICATED`
selecting a better draft head. **Refuted by code audit:** DEDICATED has one read site
(`utils_mlx.py:362`) and overlays weights onto `model.model.mtp[0]` — the *classic* MTP head.
Production drafts with `model.model.dspark`, a different module that ignores DEDICATED
entirely. Recorded so the dead end is not re-walked.

---

## 2. Pre-registration (committed `62aeca65e` BEFORE boot 1)

Model: `a(γ) = Σ_{k=1..γ} p^k · d^max(0,k−3)`, cycle `= 56.1 + 4.25·γ` ms (round-4 ground truth).

| regime | damping | γ2 | γ3 | γ4 | γ5 | clears 1.10x? |
|---|---|---|---|---|---|---|
| A true (p=.668) | none | 0.934 | 1.000 | 1.019 | 1.012 | NO |
| A true | soft d=.5 | 0.934 | 1.000 | 0.981 | 0.939 | NO |
| A true | hard d=.15 | 0.934 | 1.000 | 0.954 | 0.902 | NO |
| B stale (p=.923) | none | 0.830 | 1.000 | 1.134 | 1.239 | **YES** |
| B stale | soft d=.5 | 0.830 | 1.000 | 1.038 | 1.023 | NO |
| B stale | hard d=.15 | 0.830 | 1.000 | 0.971 | 0.921 | NO |

**Predicted:** no arm clears the band (5 of 6 cells); ordering **γ3 ≥ γ4 > γ5 > γ2**;
γ=2 worst in every cell.

---

## 3. Phase 1 — measured acceptance (the trustworthy result)

Both arms: env confirmed on **both** nodes' real `.venv/bin/python -m exo -v` child before
measuring; API 200; both nodes READY.

| arm | cycles | mean_accept | hist | P(ceiling) | P(0) |
|---|---|---|---|---|---|
| **γ=3** (control) | 50 | **1.820 / 3** | `0:11, 1:9, 2:8, 3:22` | 44.0% | 22.0% |
| **γ=4** | 100 | **2.420 / 4** | `0:17, 1:14, 2:18, 3:12, 4:39` | 39.0% | 17.0% |
| γ=4 (2nd sample) | 50 Δ | 2.501 / 4 | `0:5, 1:11, 2:7, 3:8, 4:19` | 38.0% | 10.0% |

Node-identical on both ranks, as expected.

### The `bypos` curve — reconstructed from the two histograms
This is the per-position data the task wanted and that has never existed on this cluster:

| | pos 1 | pos 2 | pos 3 | pos 4 |
|---|---|---|---|---|
| γ=3 `P(accept ≥ k)` | 78.0% | 60.0% | 44.0% | — |
| γ=3 conditional continue | 78.0% | 76.9% | 73.3% | — |
| γ=4 `P(accept ≥ k)` | 83.0% | 69.0% | 51.0% | 39.0% |
| γ=4 conditional continue | 83.0% | 83.1% | 73.9% | **76.5%** |

**The curve does NOT cliff.** The conditional continue-probability is flat at 73–83% all the
way through position 4 — including *past* the head's 3 trained stages. The distribution is
**strongly bimodal** (mass piled at 0 and at the ceiling), not geometric: once a draft is
on-track it stays on-track; when it derails it derails immediately.

**Consequences:**
- A geometric fit to γ=3 (p=0.7701) predicts `a(4)=2.172`; **measured 2.420** — the geometric
  model *under*-predicts position-4 acceptance by +0.248 tok/cycle.
- **My pre-registered ordering is contradicted.** The "trained for width-3" comment is a poor
  predictor of *acceptance* behaviour. That is a real finding about the draft head, and per
  the task's framing it is the round's result, not a failure.
- γ=2 was never measured (§6), so the May-2026 "γ=2 champion" claim is **not** re-tested here.

### What this implies for throughput — and why it is NOT a ship decision
If round-4's cycle model (`56.1 + 4.25γ`) holds, γ=4 derives to ≈1.14x — above the band.
**I am explicitly NOT claiming that.** It substitutes a *model* for the measurement the band
requires, and §4 shows this round produced no valid cycle time either. It is a hypothesis for
round 6, nothing more.

---

## 4. Why throughput is UNMEASURED (the honest core of this round)

Two independent defects. Both proven by arithmetic on the harness's own recorded fields.

### Defect 1 — the original harness timed a buffered stream
`arm_g3_boot1.json`, every rep: `ttft_s ≈ 150.4s`, `total_s ≈ 150.6s`, 71 tokens.
Implied decode window: **0.21 s for 71 tokens = 338 tok/s.** Against a cluster whose
established decode is ~20–35 t/s, that is ~10–14x physically impossible. `finish_reason: null`
corroborates improper stream termination. The reported "363–647 tok/s" figures are an
**artifact of burst delivery**, not throughput. Discarded.

### Defect 2 — the "fix" replaced a token rate with a CHUNK rate
The repaired harness computes `decode_tps = (n_chunks − 1) / decode_window_s`. Every rep
recorded **exactly 9 token-bearing chunks** regardless of whether 71 or 101 tokens were
generated — so this measures *chunks per second*, not tokens per second. Re-deriving from its
own fields:

| rep | tokens | chunks | window s | tok/chunk | implied **token**/s |
|---|---|---|---|---|---|
| 0 | 71 | 9 | 0.150 | 7.9 | 472 |
| 1 | 101 | 9 | 0.170 | 11.2 | 595 |
| 2 | 74 | 9 | 0.146 | 8.2 | 507 |
| 3 | 101 | 9 | 0.210 | 11.2 | 480 |

The "believable 38–55 tok/s" was 9 chunks over a ~0.15s window — the same artifact wearing a
different unit. `stream_looks_buffered=False` is a false negative: its heuristic looks for
tokens piling into the last 10% of the window, but here the *entire* window is the burst.

### Defect 3 — the prefill collapsed between runs (cache reuse)
Same prompt, same depth: `ttft_s` fell from **~150 s** (first boot) to **~3 s** (later run).
A ~62K-token prefill at this cluster's measured ~350 tok/s ceiling must take ~170 s. A 3 s
TTFT means the prefill did not happen — a warm prefix/KV path was hit. Any A/B across those
two runs compares different cluster *states*, which this campaign's own standing rule forbids.

### Also: the depth was ~62K, not 89K
`--depth 89000` produced **61,684–62,062 actual prompt tokens** every rep. The prompt was
identical across arms, so the *acceptance* A/B is unaffected — but **every number in this
round is at ~62K, not the specified 89K.** Stated so no downstream reader mis-cites it.

**Net: `derived_tps` is `null` in every artifact, cycle time was never measured, and the
measured-t/s half of the band has no valid input.** The harness emitted explicit nulls +
warnings rather than fabricating values — the one thing that worked as designed.

---

## 5. Band application (verbatim, as pre-registered)

> derived `(1+a)/cycle` ≥ **1.10x** the γ=3 control (both γ=3 boots) **AND** measured t/s
> range entirely above γ=3's range → SHIP CANDIDATE. **1.03–1.10x** → inside boot variance.
> **<1.03x** → closed.

| arm | derived (1+a)/cycle | measured t/s range | band verdict |
|---|---|---|---|
| γ=3 (boot 1) | **null** (no cycle time) | invalid | control |
| γ=3 (boot 2 bracket) | **NOT RUN** | — | see §6 |
| γ=4 | **null** (no cycle time) | invalid | **INDETERMINATE — cannot evaluate** |
| γ=2 | NOT RUN | — | — |
| γ=5 | NOT RUN | — | — |

**Verdict: no arm satisfies the band, because the band's inputs do not exist.** γ=4 is
*acceptance*-promising (+33%) but that is one of the two required conditions. **NO SHIP.**
Shipping γ=4 on acceptance alone would be exactly the "no single-boot t/s delta as evidence"
violation the constraints forbid.

---

## 6. What was not done, and why

- **γ=2 and γ=5 arms: NOT RUN.** Time box. Each arm is a full ~15-minute relaunch and the
  measurement defect consumed the budget. Per the pre-registered degrade path, γ=2 is the
  designated first drop.
- **Closing γ=3 bracket: NOT RUN** — a deviation from my own protocol, recorded as such.
  Boot variance is therefore **unquantified**, which independently blocks any ship claim.
- **Phase 2 quality gate: NOT RUN.** Correct per protocol — it runs only on a ship candidate,
  and there is none. `identity_gate.py` is built and smoke-tested (5/5 byte-identical,
  exit 0, same-config control) and is ready for round 6.
- **Phase 3 ship: NOT DONE.** γ=3 left in place, as specified when no candidate clears.
- **I8 per-draft cost / phase ratios: UNMEASURED.** Requires `EXO_DSV4_MTP_PROFILE`, which
  brackets with `mx.eval` *every* cycle ("serialises pipelining — measurements are upper
  bounds", `dsv4_mtp.py:240-243`) and would corrupt the t/s under test. Pre-registered as
  deferred; the sweep did not reach it. **Not guessed.**

---

## 7. Cluster health (end state) — PM-verified, not self-reported

Restored to the shipped default and independently re-checked by the PM after the worker
reported success:

```
macstudio-m4-1  runner_pid=31608  EXO_SPECULATIVE_GAMMA=3  EXO_DSV4_MTP=1  EXO_DSV4_DSPARK=1
                leftover_diag_count=0
macstudio-m4-2  runner_pid=39813  EXO_SPECULATIVE_GAMMA=3  EXO_DSV4_MTP=1  EXO_DSV4_DSPARK=1
                leftover_diag_count=0
API /v1/models  http=200
```

Zero `EXO_DSV4_MTP_LOG_INTERVAL` / `EXO_DSV4_MTP_PROFILE` on either runner. Smoke completion
returned coherent output with `finish_reason` present.

---

## 8. RECONCILIATION — why the May-2026 γ results do not transfer

**Named explicitly, as required.**

1. **The async fence was SILENTLY BROKEN.** Every May γ number (γ=2 champion at 30–31.5 t/s,
   "γ=3 is −18% vs γ=2", γ=3 bistability) was measured with the fence broken; fixing it
   (2026-08-22) moved throughput **+58–67%**. Those numbers describe a machine that no
   longer exists.
2. **Verify was ROWSEQ.** Per-row attention + per-row TP all_reduces meant verify cost scaled
   with γ, so larger γ was structurally expensive. Batched verify was promoted 2026-08-27
   (**+36.7%**); at M=γ+1 weights are read once for all rows. **The regime that penalised
   large γ is gone.** Round 4 independently measured the verify GPU at 117% busy on real
   compute — no idle gap.
3. **γ=3 arrived with the 08-27 batched-verify promotion and had never been swept post-fix.**
   Confirmed: this round is the first acceptance measurement of the config at all.
4. **The acceptance figure prior tuning used was wrong** — and so was its replacement. Tuning
   used 1.73; the campaign-1 wall-attribution gave 1.411; **direct counter measurement gives
   1.820/3.** Anything derived from 1.73 or 1.411 should be re-derived.
5. **The stale 2026-08-29 histogram (`mean_accept=2.561/3`, 85.4%) does not transfer either.**
   Its launcher set `EXO_SPECULATIVE_GAMMA=2` while the log's own denominator reads `/3`.
   Explained by `dsv4_mtp.py:3913-3925`: before 2026-08-26 the DSpark branch **silently
   ignored `EXO_SPECULATIVE_GAMMA` and ran width-5 drafts**. That run is from a different
   gamma-resolution regime and is not comparable. Superseded by the 1.820/3 measurement.

---

## 9. Recommendations for round 6 (ranked)

1. **Fix the throughput harness before spending another boot.** Required: count *tokens* not
   chunks; verify TTFT is consistent with the known ~350 tok/s prefill ceiling (a 3 s TTFT on
   62K is a cache hit, not a fast prefill); bust the prefix cache between reps or vary the
   prompt per rep; assert `finish_reason` is non-null. **Gate: the harness must reproduce the
   known ~26 t/s at 2K before any sweep number is believed.**
2. **Then re-run 3,4,5,3 with valid t/s.** γ=4's +33% acceptance makes it the strongest
   candidate this campaign has produced. γ=5 is the reachable ceiling.
3. **Set `EXO_DSV4_SPEC_SHADOW=1` on one boot** for a real `bypos` curve, rather than
   reconstructing it from histogram differences as I did here.
4. **Measure at true 89K** — `--depth 89000` yields ~62K.
5. `EXO_DSV4_MTP_EAGLE_K`: the audit found it **LIVE** on the BS≥2 batched path
   (`dsv4_mtp.py:1108` → `:3545`), contradicting `start_cluster.sh:192`'s "DORMANT under
   DSpark" comment. The comment is wrong for c=1-vs-batched reasons; worth a follow-up.

## 10. Artifacts
- `PRE-REGISTRATION.md` — committed `62aeca65e` **before** boot 1
- `phase0-histogram.md`, `harness-map.md`, `gamma-resolution-audit.md`
- `measure_arm.py`, `identity_gate.py` (identity gate verified working; measure_arm **known
  broken for throughput** — see §4, do not reuse its t/s without the fix)
- `results/arm_g3_boot1.json`, `arm_g4_boot1.json`, `arm_g4_boot1_fixed.json`
- **No pushes.** Local commits only.
