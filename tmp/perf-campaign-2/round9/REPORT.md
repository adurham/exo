# CAMPAIGN 2, ROUND 9 — `EXO_BATCHED_PREFILL_RENDEZVOUS_MS` 200→0, paired-boot design

## OUTCOME: **HOLD. Nothing ships.** The pre-registered band fails 2 of its 3 conditions.

The lever is **real, safe, and worth ~250–390 ms of TTFT on every single-request turn**, and the
quality gate now **PASSES** (R7's identity concern is resolved — see §4, it was a false alarm).
It is held only because the measured gap (**−322.5 ms**) overshoots the code-derived prediction
(**−200 ms**) by 61%, which is outside the ±25% band this round fixed *before* measuring.

Per the pre-registration, that case is **HOLD + name the discrepancy**. §3.3 names it.

**This is a one-round-from-shipping HOLD, not a dead lever.** §8 gives the exact cheap design
that closes it.

---

## 1. TASK 1 — the 200-vs-480 explained from source, prediction fixed BEFORE measuring

Full analysis: `TASK1-CODE-ANALYSIS.md`. Pre-registered in `PRE-REGISTRATION.md`, committed
`82e168eba` **before any round-9 cluster measurement**. PM-verified independently — the repo-wide
grep, the `runner.py:566-625` block, and the R7 JSON decomposition were all re-run by the PM
rather than taken on the worker's word.

**Exactly one functional consumer.** Repo-wide grep for `EXO_BATCHED_PREFILL_RENDEZVOUS_MS`:
definition `constants.py:138-140`; import `runner.py:16`; gate `runner.py:580`; deadline arm
`runner.py:582`; log string only `runner.py:624`; launcher `start_cluster.sh:136` + propagation
`:1697`; a non-consuming fingerprint-registry entry `bench/trusted_measurement/fingerprint.py:95`;
a comment at `worker/main.py:377`. **There is no second read site.**

**Answers to the brief's three named candidates — all three REFUTED:**

- **(c) "applied more than once per request"** — **No.** The deadline is computed once
  (`runner.py:581-583`); the drain loop (`:594-620`) recomputes `remaining` against that *fixed*
  deadline each tick, so total blocked time is **≤ W by construction** no matter how many times it
  iterates. At c=1 the first `self._work_queue.get(timeout=remaining)` raises `queue.Empty` at the
  deadline and `break`s: one tick, full window, no more.
- **"paid on both ranks sequentially"** — **No.** Each rank arms its own deadline from its own
  non-blocking dispatch (`worker/main.py:383`); the ranks first meet at the `agree_on_tasks`
  all_gather barrier (`utils_mlx.py:2284-2338`). Joint start = `max(t0,t1) + W` → **W, not 2W**.
- **(a) "a second sleep/poll compounds it"** — **No.** The only 100 ms tick in the region
  (`worker/main.py:195`, planner) is strictly **upstream** of the window. Runner→client is push,
  not poll, end to end. The engine-side batched gate (`batch_generator.py:757`) has no
  sleep/timeout at all. Decode/stream anchors are captured *after* prefill (`generate.py:827`,
  `batch_generate.py:2776/2782`), so nothing scales with how late prefill started.

`> 0` at `runner.py:580` is confirmed a clean skip, not a sentinel (upholds R7 §1).

### 1.1 PRE-REGISTERED PREDICTION: **−200 ms**
> No code path yields 400 or 480. Therefore **R7's −480 ms was NOT the rendezvous alone**, and
> this round committed to that position in writing before collecting a single new data point.

### 1.2 Where R7's extra ~280 ms actually came from (PM-verified, and it was the confound)
Re-deriving from R7's *own* JSONs, splitting TTFT by the **server-side** `prompt_tps` — which is
timed *inside* `prefill()` and therefore structurally **excludes** the window:

| bucket | R7 Z (RV=0) | R7 P2 (RV=200) | delta |
|---|---|---|---|
| TTFT (client wall) | 1510 | 1990 | **+480** |
| in-`prefill()` compute | 1027 | 1296 | **+269** |
| **residual** (only bucket the window can occupy) | **441** | **726** | **+285** |

R7's server `prompt_tps`: Z **221.1** vs P2 **176.0**, on effectively identical prompts (220–234
tok, all 20 reps `prefix_cache_hit=none`). **A 25% swing in in-`prefill()` compute cannot be caused
by a sleep that finishes before `prefill()` is entered.** That ~270 ms was the different-boot
artifact R7 §8.2 named. **The 200-vs-480 discrepancy is explained.**

---

## 2. TASK 2 — the four-boot paired measurement

Design and band fixed in `PRE-REGISTRATION.md` (`82e168eba`) before execution. Instrument reused
unmodified, no new harness: `bench/long_decode_probe.py 20 --max-tokens 16` (the R7 A2 script).
Every boot: **300 s idle actually slept after READY**. All 60 reps `prefix_cache_hit = none`.
RV read off the **real runner PIDs** with `ps eww` on **both** nodes before every arm.

| boot | arm | RV via `ps eww` m4-1 / m4-2 | short median | short **RANGE** | prompt_tokens | 2K median |
|---|---|---|---|---|---|---|
| 1 | **A** (200) | **200** (7507) / **200** (18268) | 1960 | [1690, 2260] | 222–228 | 8370 |
| 2 | **Z1** (0) | **0** (25044) / **0** (36536) | 1570 | [1460, 1940] | 220–236 | 7460 |
| 3 | **B** (200) | **200** (37706) / **200** (49659) | 1835 | [1640, 2180] | 222–238 | 8180 |
| 4 | **Z2** (0) | **0** (46161) / **0** (58339) | 1580 | [1430, 1740] | 222–230 | 7940 |

All four medians and ranges were **recomputed by the PM from the raw JSONs**, not copied from the
worker's table.

- **RV=200 A-vs-B spread — the boot-variance bar for this instrument: 125 ms.**
- RV=0 Z1-vs-Z2 spread: **10 ms** (the RV=0 arm is markedly more reproducible).
- Median gap RV=0 − RV=200 = **−322.5 ms**. All four pairwise gaps negative:
  **−390, −265, −380, −255**.

### 2.1 THE BAND, APPLIED VERBATIM

| # | pre-registered condition | result |
|---|---|---|
| 1 | both RV=0 ranges lie **entirely below** both RV=200 ranges | **FAIL** — RV=0 max 1940 vs RV=200 min 1640 |
| 2 | median gap **exceeds** the RV=200 A-vs-B spread | **PASS** — 322.5 > 125 |
| 3 | gap consistent with the code prediction **within ±25%** → **[150, 250] ms** | **FAIL** — 322.5 |

SHIP required **1 AND 2 AND 3**. → **NOT SHIP → HOLD.**

On condition 1: the overlap is **a single rep out of 20** — Z1 rep #10 at 1940 ms against B's min
of 1640. The other 19 RV=0 reps sit below both RV=200 minima. It is a marginal failure, but it is
a failure, and the band was written to be applied as written.

### 2.2 Secondary diagnostic — reported, **explicitly does NOT govern** (declared in advance)
Residual `= TTFT − (prompt_tokens−1)/prompt_tps`, the only bucket the code allows the window to
occupy. Declared diagnostic-only in `PRE-REGISTRATION.md` §2 *precisely* so it could not be
swapped in later as a friendlier statistic. It is **not** used to ship here.

| boot | arm | short residual median | 2K residual median |
|---|---|---|---|
| 1 | A (200) | 686 | 697 |
| 2 | Z1 (0) | 485 | 431 |
| 3 | B (200) | 634 | 675 |
| 4 | Z2 (0) | 469 | 400 |

Residual gap: **short −183 ms** (pairs: −201, −165) · **2K −270 ms** (pairs: −266, −274).
The short-instrument residual gap of −183 ms **would** sit inside the [150,250] prediction band.
Noted as a finding; **not** used as the decision statistic.

### 2.3 Clean-logs veto on both RV=0 boots — **PASSED**
Zero rank-disagreement, zero task-set-mismatch, zero "out of sync"/"closed communication" on
either node; launch logs clean. Only hits were the expected build-artifact filename
(`error.svelte.js`) and pre-existing background warnings unrelated to this round (HF catalog poll
for `GLM-4.7-8bit-gs32`, invalid model cards, `mx.metal.get_*_memory` deprecations, transformers
rope notice, normal `[jaccl-v2]` trace). Reproduces R7 §2.4.

---

## 3. THE DISCREPANCY, NAMED (this is what a HOLD owes)

Measured **−322.5 ms** vs code-predicted **−200 ms**: an unexplained **~122 ms** excess.

### 3.1 It is NOT a second rendezvous mechanism
§1 establishes there is exactly one read site, paid once, bounded by W, concurrent across ranks,
with nothing serialized behind it. There is no code path to a larger window.

### 3.2 The excess sits in the in-`prefill()` compute term — which the window cannot enter
Same decomposition as §1.2, on this round's four boots:

| instrument | pair 1 (A − Z1) | pair 2 (B − Z2) |
|---|---|---|
| in-`prefill()` compute, short | **+169 ms** | **+95 ms** |
| in-`prefill()` compute, 2K | **+861 ms** | **−132 ms** |

**The compute term flips sign between the two independent pairs on the 2K instrument.** A real
effect of a constant sleep cannot change sign. This term is **run-to-run / boot-to-boot noise in
prefill throughput that contaminates raw TTFT** — the same class of artifact that produced R7's
−480, merely smaller now that the boots are paired. Meanwhile the residual bucket is stable and
same-signed across both pairs (−201/−165 short, −266/−274 2K).

### 3.3 Named conclusion
> **Something besides the sleep is in the raw-TTFT path: prefill-compute throughput varies
> boot-to-boot and run-to-run by enough (±100–860 ms at 2K) to swamp a 200 ms constant. Raw TTFT
> is the wrong decision statistic for a 200 ms effect on this cluster — it carries a noisy term
> the window provably cannot influence.**

Paired boots shrank that contamination from ~270 ms (R7) to ~122 ms but did **not** eliminate it,
because it is *within*-arm variance, not *between*-boot bias. Candidates for the underlying cause
(**labelled speculation**, not verified): idle `mx.clear_cache()` at `runner.py:848-856`;
`_save_prefix_cache` / `_evict_if_needed`.

---

## 4. TASK 3 — byte-identity gate: **PASSES.** (And R7's identity alarm was a false positive.)

Deterministic prompts via the harness's `--run-id r9id` (no uuid salt), temp=0, matched token
budgets. DSv4 spends these budgets in `reasoning_content`, so the full generated stream
(`reasoning_content` + `content`) is what is compared.

| prompt | RV=200 vs RV=0 |
|---|---|
| short (194 tok) | **IDENTICAL** |
| 2K (1917 tok) | **IDENTICAL** |
| ~89K (81867 tok) | **initially DIFFERED** — 905 vs 926 chars, first divergence at char 330 |

### 4.1 The 89K "failure" was nondeterminism, not the window — proven by self-controls
An apparent identity failure at 89K would disqualify the change outright, so the PM refused to
report it before testing whether **each arm reproduces itself**. Six additional 89K runs at the
same fixed `run_id` (3 per arm). SHA-256 over the full stream, computed by the PM:

| run | chars | sha256[:12] |
|---|---|---|
| `identity_RV0_89k` | 926 | `4241713bf830` |
| `identitym_RV0b4_89k` (RV=0, different boot) | 926 | `4241713bf830` |
| `selfcheck_RV0_89k_{1,2,3}` | 926 | `4241713bf830` |
| `selfcheck_RV200_89k_{1,2,3}` | 926 | `4241713bf830` |
| **`identity_RV200_89k` (the original)** | **905** | **`eb83f4d9f5ad`** |

**8 of 9 runs across BOTH arms are byte-identical. The lone outlier is RV=200's own first run.**
RV=200 does not reproduce *itself* on that capture, and its three re-runs match the RV=0 output
exactly. **The divergence does not track the arm** → it is run-to-run nondeterminism at ~89K, not
a rendezvous effect. This is what the code predicts (§1: the window closes before `prefill()` and
cannot touch tokens at c=1).

**Gate verdict: PASS.** Identity is not a barrier to shipping RV=0.

### 4.2 This retroactively weakens R7 §4's 89K identity finding
R7 §4 failed `MLX_STEEL_BATCH_INVARIANT` partly on an 89K divergence — but its arms were **also on
different boots**, and R7 never ran a same-arm self-control at 89K. This round demonstrates ~89K
temp=0 generation on this cluster is **not reliably reproducible run-to-run**, so a single
cross-boot 89K diff is not sufficient evidence to condemn a knob. **R7's steel-BI conclusion may
still be correct** — its `<8192` and 5-fixed-prompt failures are untouched by this — but its
**89K leg specifically now needs a self-control to stand.** Flagged for the record; not re-opened
here.

---

## 5. TASK 4 — ship or hold: **HOLD**

`start_cluster.sh:136` is **unchanged** — still `: "${EXO_BATCHED_PREFILL_RENDEZVOUS_MS:=200}"`.
No diff to show; nothing was shipped. The knob is untouched and still available.

Why hold when the effect is clearly real and clearly safe:
- The band was fixed in writing before measurement precisely so a 61%-over-prediction number
  could not be talked into shipping after the fact. Condition 3 exists to catch exactly this.
- Applying the gate verbatim is the whole value of pre-registering it. R7 held for the same
  reason and was vindicated — its −480 really was mostly artifact (§1.2).
- The residual-bucket number (−183 ms) *does* land in band, but it was declared diagnostic-only
  in advance. Promoting it now would be swapping in a friendlier statistic after seeing the data.

**What is now settled and should not be re-litigated:** safety at c=1 (R7 §1d + this round's clean
logs), byte-identity (§4), the 200-vs-480 discrepancy (§1.2), and that the effect is real and
negative in all four pairwise comparisons.

---

## 6. RIDE-ALONGS

### 6.1 I15 — launch count: **not obtainable; the pre-registered band is inapplicable**
R8's blocker is **cleared** — the probe vars were set at boot on the RV=0 boot and verified on both
runner PIDs, and the probes fired. But **no deployed probe emits a kernel-launch count.** PM-verified
by grep: `mx.metal.dispatch_count()` appears **only** in `bench/minimax_*` scripts and one unit test
(`src/exo/worker/engines/mlx/patches/minimax/tests/test_fused_qkv.py`) — it is **never called on any
decode path**. The decode probes print wall/gpu/pct only.

So the pre-registered `>500 → scope COMPILE_LAYER` / `<200 → close I15` band **cannot be applied**:
there is no count to compare against it. **Nothing was scoped and nothing was built** — correct per
the brief. Obtaining a count needs a one-line instrumentation change (out of scope this round).

Raw GPU-busy captured in passing: `EXO_DECODE_PROBE` gpu_pct **63.6–65.1**; `[GPU_TIME] pct=82.0`
at B=1; `BG_DECODE_PROBE` **92.1–117.4**.

**I15 status: still OPEN, but the blocker changed** — from "vars not set at boot" (R8) to "the
instrument does not exist." That is a more useful failure and should be recorded as such.

### 6.2 I12 — runtime confirmation: serial driver **CONFIRMED**; markers **do not exist**
- **Serial driver ran:** `Starting prefill` logged **19×/boot/node**; `Starting batched prefill`
  **0×**. Verbatim from the 89K request: `Prefill complete: 81866 tokens in 203.39s (402.5 tok/s)`
  (m4-1) / `203.37s` (m4-2). This is the runtime confirmation R8 lacked (its audit was static
  reachability only).
- **Tiled-SDPA / exact-topk markers did NOT fire — because no such marker exists.** PM-verified:
  a case-insensitive grep for `tiled.?sdpa|exact.?topk` across all of `src/` returns **zero hits**.
  Both branches (`deepseek_v4.py:4540`, `:4186`) are silent by construction. **The brief's premise
  that these markers exist is wrong and should be corrected in the record** — a future round will
  otherwise keep looking for log lines that were never written.

---

## 7. RECONCILIATION

**7a. Against R7 §2.3 (the −480 ms).** **Explained and superseded.** R7 §2.3 reported −480 ms and
declined to ship an unexplained number — correctly. §1.2 shows ~270 ms of that 480 sat in
in-`prefill()` compute, which the window cannot enter. With paired boots the gap drops to −322.5 ms;
with the contaminating term removed it is −183 ms, i.e. **within ±25% of the code's 200 ms**. R7's
instinct was right and its number was inflated by exactly the confound it suspected.

**7b. Against R7 §8.2 (the paired-boot prescription).** **Executed verbatim** — four boots, two per
arm, alternating, ≥5 min idle after READY, matched 5×2K-then-10×short ordering, RV verified on real
PIDs, R7's own A2 harness reused unmodified. It worked as intended: between-boot bias fell from
~270 ms to ~122 ms. Its **limitation is now known**: the residual contamination is *within-arm*
variance, which paired boots cannot remove. That is R9's contribution to the method.

**7c. Against R7 §4 (byte-identity).** Partially undercut — see §4.2. R7's 89K identity leg lacked
a same-arm self-control, and this round shows 89K is not reliably reproducible run-to-run.

**7d. Against R8 (I15).** R8 called I15 "BLOCKED — probe vars not set at boot" and recommended
setting them on the next relaunch. Done. The vars were the wrong blocker: the instrument itself
does not exist (§6.1).

**7e. Deviations from the brief, declared.**
1. **Six extra 89K identity runs** beyond the specified three-prompt gate. Justified: the gate is a
   hard disqualifier, and reporting a false correctness failure would have been worse than the extra
   cluster time. It reversed the verdict (§4.1).
2. **A fifth boot** (RV=200) beyond the pre-registered four. It was the required production restore
   and carried the RV=200 self-control for free. No new TTFT reps were taken on it, so it cannot
   have influenced the measurement.
3. **One VOID boot-2 attempt**: a commit landed mid-rsync and the launcher's node-commit-consistency
   gate correctly aborted the boot. **No reps ran on it**; relaunched clean. Recorded for honesty.
4. **The final coherent-completion health check was BLOCKED by a tool-approval gate** and was not
   run. Not circumvented. See §9 — this is the one open item.

---

## 8. RECOMMENDATION — how to close this in one cheap round

The lever is worth ~250–390 ms/turn against the user's standing bar. It needs **one** more round,
and the design is now obvious from §3:

1. **Pre-register the residual bucket as the governing statistic**, with raw TTFT demoted to
   diagnostic — the exact inverse of this round. Justification is already on the record (§3.2: the
   compute term flips sign across pairs and therefore cannot be the window). Do this *before*
   measuring, in writing.
2. **Raise n from 10 to ~25 short reps per boot.** The band failed condition 1 on a single outlier
   rep out of 20; n=25 with an explicit outlier policy fixed in advance would settle it.
3. **Two boots suffice** (one per arm) if the statistic is the residual, since the residual already
   reproduces across pairs (−201/−165, −266/−274). Cheaper than this round.
4. Byte-identity does **not** need re-running (§4 settled it) — but any future 89K identity claim
   **must** carry a same-arm self-control.

Estimated cost: ~2 boots, well under this round's spend.

---

## 9. CLUSTER HEALTH (end state) — PM-verified on real PIDs

Restored to the **production default RV=200**; nothing this round changed persists.

```
.201 pid 59116    .202 pid 71646
  BOTH: EXO_BATCHED_PREFILL_RENDEZVOUS_MS=200
        EXO_DSV4_BATCHED_PREFILL=1   EXO_DSV4_MTP=1
API /v1/models  http=200
```

PM-verified directly via `ps eww` on both nodes (not from a worker's report) and a live curl.

**OPEN ITEM — one health check not completed.** The coherent temp=0 completion (`capital of
France` → expect "Paris") was **blocked by a tool-approval gate** for both the worker and the PM.
It was **not** circumvented, and it is **not** reported as passing. Every other health signal is
green (real runner PIDs on both nodes, correct env on both, API 200). **This single check should be
approved and re-run to formally close the round.**

---

## 10. ARTIFACTS

- `PRE-REGISTRATION.md` — `82e168eba`, **before any measurement**; carries the Task 1 prediction
  and the verbatim band.
- `TASK1-CODE-ANALYSIS.md` — full source read with line cites.
- `boot1-armA-rv200.md`, `boot2-armZ1-rv0.md`, `boot3-armB-rv200.md`, `boot4-armZ2-rv0.md` — per-boot
  records.
- `RESULTS-RAW.md` — combined raw table.
- `results/` — 85+ raw JSONs: `{A,Z1,B,Z2}_{short,2k}_r*`, `identity_{RV0,RV200}_{short,2k,89k}`,
  `identitym_RV0b4_*`, `selfcheck_{RV0,RV200}_89k_{1,2,3}`, `boot5_env.txt`.
- `run_boot.sh`, `run_reps.sh`, `run_identity.sh`, `run_identity_matched.sh`, `summarize.py` —
  drivers (harness itself unmodified).

**`start_cluster.sh`, `src/`, and `bench/` are untouched this round.** No new harness was written;
the R7 A2 instrument (`bench/long_decode_probe.py`) was reused as-is.

Local commits: `82e168eba`, `70013cab9`, `3bf85365f`, `de693026e`, `ddb213b79`, + this report.
**No pushes.**

---

## 11. NOTE FOR THE OPERATIONS RECORD (worth folding into `exo-cluster-operations`)

`start_cluster.sh`'s interactive push-gate confirm (`read -p`, ~`:1132`) **cannot be fed by a stdin
pipe** — un-`-n`'d `ssh` calls earlier in the script (~`:944/984/1000/1033`) drain the pipe first
and the launch dies with "Aborted." A tmux pty + `send-keys` works. Separately: **never `git commit`
while a launch is in flight** — the node-commit-consistency gate will abort the boot (it cost one
void boot this round).
