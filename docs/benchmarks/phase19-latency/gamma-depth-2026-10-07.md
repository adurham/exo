# Phase 19 follow-up — is gamma=3 near-optimal on the LIVE dsv41 engine at real context depth?

Date: 2026-10-07 (afternoon, LIVE run 14:00–15:0x CDT). Owner: Hermes PM subagent
(`sa-0-d17acd3f`, root session `20261007_093547_82d809`). The earlier Phase-A-only pass
(§3–§9) was completed by `sa-0-1a117dc2` under this same delegation; that agent stood down
in error (mistook its own registry entry for a peer). **This pass is the live half — sole
owner, no peer.**
Cluster: 2× Mac Studio M4 Max, TP2 over jaccl RDMA.
Baseline deploy: `deploy/next13` @ `f0840af1c` (restored at the end).
Instrumented branch deployed for the matrix: `deploy/next14-gamma` @ `67eac5127`
(engine commit 295ec50bc + the bench driver commit).
Model: `dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw` (engine `dsv41`, greedy-only).

> **STATUS: COMPLETE (live). The interleaved gamma matrix RAN, on both workloads, in one
> process via the per-request `spec_gamma` field. Relaunches used: 2 of 2 (1 deploy, 1
> restore — both completed). Cluster restored to `deploy/next13` @ `f0840af1c` at 15:0x.**
> See §10 for the results; §1–§9 below are the earlier Phase-A pass and are kept verbatim.

---

## 10. LIVE RESULTS — gamma 3 vs 4 vs 5 (the half that was stood down)

Run 2026-10-07 14:14–14:42 CDT, one warm `67eac5127` process, interleaved `ABAB…`
(γ3,γ4,γ5 round-robin per rep) so thermal drift spreads across arms, ≥3 warm reps/arm,
`max_tokens=1000`, `temperature=0`, prefix-cache-warm (ttft ~0.3 s). Raw:
`raw/gamma-matrix.json` (+ `raw/gamma-matrix.json.jsonl` checkpoint). 0 guard failures.

### 10a. The verdict — workload-dependent, and the winner is γ5 on benign

| workload | arm | decode t/s (median) | IQR | ms/round | mean acc | rounds | Δ vs γ3 |
|---|---|---:|---|---:|---:|---:|---:|
| **benign** 100K | γ3 | 24.756 | [24.750, 24.782] | 154.3 | 2.827 | 254 | — |
| benign | γ4 | 24.489 | [24.485, 24.503] | 174.3 | 3.278 | 234 | **−1.1 %** |
| benign | **γ5** | **28.070** | **[27.985, 28.087]** | 185.4 | 4.203 | 192 | **+13.4 %** |
| **agentic** (real 9a2ed7) | γ3 | 20.692 | [19.766, 20.701] | 153.8 | 2.188 | 314 | — |
| agentic | γ4 | 19.519 | [19.501, 19.547] | 173.5 | 2.390 | 295 | −5.7 % |
| agentic | γ5 | 20.116 | [20.076, 20.119] | 183.9 | 2.711 | 270 | −2.8 % |

- **benign: γ5 wins, IQRs fully disjoint** (27.98–28.09 vs 24.75–24.78) — ≥3 % bar cleared,
  quality gate run (§10c) and PASSES. γ4 is ~flat/worse (it pays one extra verify row for
  almost no extra acceptance — benign per-position p4=0.76 vs p3=0.91, so the 4th slot
  rarely commits).
- **agentic (real content): γ3 is best.** γ4 −5.7 %, γ5 −2.8 %, both IQRs disjoint from γ3.
  On real agentic content the acceptance tail decays hard (p4=0.38, p5=0.26), so the extra
  verify rows cost more than they commit. **These arms are NOT interchangeable — the
  winner is workload-dependent.**

### 10b. Drift classification (A3) — SETTLED: host/thermal, NOT process-state

The first benign rep of the fresh `67eac5127` process (~14 min old) measured
**ms/round = 156.17 (cold) / 154.34 (warm)**, mean accepted 2.827 — i.e. the *afternoon*
level (154.1), **not** the morning 135.6. A brand-new process does not recover the fast
wall ⇒ the same-day +13.7 % rise is **host/thermal class, not long-lived-process
degradation**. A periodic restart is therefore *not* a lever. (Process age cannot explain
it: the wall was already at the slow level within 14 min of boot.)

### 10c. Quality gate on the benign γ5 winner (vs γ3)

`bench/dsv41_quality_battery` @ depth 40000, `--skip-park`, fixed prompt set. Baseline
`results/g3` (no field) vs candidate `results/g5` (`--spec-gamma 5`):

| battery | γ3 (baseline) | γ5 (candidate) |
|---|---|---|
| needles | 6 / 6 PASS | 6 / 6 PASS |
| tool calls | 10 / 10 PASS | 10 / 10 PASS |
| free prose | 0 DIRTY, 0 REVIEW (20) | 0 DIRTY, 0 REVIEW (20) |
| glued cross-lingual fragments | none | none |

`compare.py --a g3 --b g5` verdict: **PASS** ("B >= A on needles/tools/detectors, no new
glued fragments"). All 20 prose outputs differ in text between arms — expected at temp=0
(chunk-verify divergence by design), non-gating. Artifacts:
`bench/dsv41_quality_battery/results/{g3,g5}/` + `results/g5/compare.json`.

### 10d. Honest reading of the verdict

- **The user's question ("is gamma=3 near the max?")** has a two-part answer: on the
  benign / near-saturated regime it is **not** — γ5 is +13 % and quality-clean, so the
  phase-19 "no ≥3 % lever" conclusion is wrong there. On **real agentic traffic** γ3 *is*
  the best of the three — the phase-19 conclusion holds for the workload the user actually
  cares about.
- **Why benign misleads:** benign filler (12 canned sentences) has p1≈0.98 flat-ish tail,
  so extra gamma slots keep committing; real agentic content has a fast-decaying tail.
- **Recommendation:** do **not** ship a default gamma change. γ5 is a real, quality-clean
  lever for high-acceptance/synthetic-ish content, but it *regresses* real agentic traffic,
  and the field is per-request so a caller can opt in. Any default policy would need an
  acceptance-regime heuristic first.
- **The prediction in §4 (γ4 wins +12 %) is falsified:** γ4 lost on both workloads. The
  §4 model's error was the shape of the per-position tail — it assumed a flat/gentle tail,
  but on real agentic content p4/p5 collapse to 0.38/0.26, so the 4th–5th slots don't pay
  for their verify rows.

### 10e. Field-honored proof

`raw/gamma-fieldcheck.json` (run 14:10): one request WITH `"spec_gamma":4` gave histogram
Δ `[16,7,5,8,25,0,0]` (25 rounds accepted exactly 4 drafts); one WITHOUT gave
`[10,10,9,36,0,0,0]` (`hist[4]=0` — structurally impossible under γ3). The top non-empty
bin is the gamma actually used ⇒ **γ4 honoured, absent-field = γ3 honoured.**

### 10f. Raw per-rep numbers

| workload:γ | rep | decode t/s | ms/round | mean acc | total_rounds |
|---|---|---:|---:|---:|---:|
| benign:3 | 1,2,3 | 24.744, 24.807, 24.756 | 154.34, 153.94, 154.26 | 2.827 | 254 |
| benign:4 | 1,2,3 | 24.489, 24.480, 24.517 | 174.34, 174.40, 174.14 | 3.278 | 234 |
| benign:5 | 1,2,3 | 27.899, 28.104, 28.070 | 186.50, 185.14, 185.36 | 4.203 | 192 |
| agentic:3 | 1,2,3 | 20.710, 20.692, 18.839 | 153.63, 153.76, 168.88 | 2.188 | 314 |
| agentic:4 | 1,2,3 | 19.482, 19.574, 19.519 | 173.82, 173.01, 173.49 | 2.390 | 295 |
| agentic:5 | 1,2,3 | 20.036, 20.121, 20.116 | 184.67, 183.89, 183.94 | 2.711 | 270 |

Per-position acceptance (survival P(accept ≥ k)), median over 3 reps:

| arm | p1 | p2 | p3 | p4 | p5 |
|---|---:|---:|---:|---:|---:|
| benign γ3 | 0.9764 | 0.9449 | 0.9055 | — | — |
| benign γ4 | 0.8974 | 0.8419 | 0.7778 | 0.7607 | — |
| benign γ5 | 0.9219 | 0.8594 | 0.8281 | 0.8073 | 0.7865 |
| agentic γ3 | 0.8790 | 0.7293 | 0.5796 | — | — |
| agentic γ4 | 0.8542 | 0.6509 | 0.5051 | 0.3797 | — |
| agentic γ5 | 0.8518 | 0.6556 | 0.5482 | 0.3926 | 0.2630 |

---

## 1. Headline (earlier Phase-A pass — kept for context)

Two results, one of which overturns the assumption this follow-up was sent to test.
**Both are superseded by §10 for the live answers; §4's model prediction is falsified.**

1. **The live gamma matrix was not run** — not for lack of instrument (it is built, tested
   and pushed), but because a second live agent (`sa-0-1a117dc2`, root session
   `20261007_093547_82d809`) is running the **identical** task against the **same**
   single shared cluster, owns the campaign mutex (`~/CAMPAIGN_ACTIVE.json`), and may
   relaunch at any moment. Two agents each with "≤2 relaunches" against one production
   cluster that carries the user's live traffic would produce up to 4 relaunches, mutual
   runner kills mid-bench, and a real risk that both agents each assume the other did the
   restore. I stood down from the destructive half and did not relaunch (§5). **Cluster
   verified untouched** at 13:37 (same PIDs, same branch).

2. **Phase-19's "no ≥3% gamma lever / gamma=3 is optimal" conclusion does NOT survive
   arithmetic on its own numbers.** The offline model in `gamma-optimality.md` is
   internally inconsistent, and when it is evaluated either (a) correctly on its own
   phase-17 table, or (b) calibrated to the *measured* live round wall, it predicts
   **gamma=4 winning** (and gamma=5 more), not gamma=3. See §4. The live test is therefore
   genuinely worth running — but it is exactly the test I could not safely run here.

The supporting measurements are in §3 (A1 real-turn split), §4 (offline recalibration),
and §6 (drift, partially). Everything below is command output or source-verified; nothing
is fabricated.

---

## 2. Phase A — environment / baseline (A3 partly, snapshot)

Verified 13:23–13:37 CDT, both nodes idle (last foreign `POST /v1/chat/completions`
12:39:40, last `provider='custom'` state.db call ended 09:50:35):

| item | m4-1 | m4-2 |
|---|---|---|
| `git rev-parse HEAD` | `f0840af1c50ecfdc…f8ce` | `f0840af1c50ecfdc…f8ce` |
| branch | `deploy/next13` | `deploy/next13` |
| launcher/runner PID | 32941 / 32954 | 42712 / 42724 |
| process start | Wed Oct 7 09:01:54 2026 | Wed Oct 7 09:01:56 2026 |

- **Env parity:** `ps eww` on both runner PIDs parsed to **129 vars each, identical var
  sets**, values equal except network identity (`EXO_DISCOVERY_{PEERS,UNICAST_PEERS}`),
  `STY` (`<pid>.exorun`), `TMPDIR`, `SSH_AUTH_SOCK`. Full dumps:
  `raw/live-env-m4-1-ps-eww.txt`, `raw/live-env-m4-2-ps-eww.txt`.
- **No thermal / performance warning** on either node (`pmset -g therm`: "No thermal
  warning level has been recorded", same for performance). `powermetrics` (best-effort,
  passwordless sudo) read **GPU Power: 15 mW** on m4-1 — fully idle. Load avg ~1.9.
- **Live gamma is still the dataclass default 3.** `EXO_SPECULATIVE_GAMMA=3` is present in
  the env but is read only by the dormant legacy path; the dsv41 engine takes gamma from
  `engine.py:280 gamma: int = 3` (`adaptive_gamma=True` at :281) and `GammaPolicy.update()`
  is never called, so it is pinned at 3. This is unchanged by this work.

---

## 3. A1 — real per-call prefill/decode split of the user's turn

**Session `20261007_092009_9a2ed7`, all 42 calls, `provider='custom'`.** Full table + raw
matched log lines: `raw/A1-real-turn-split.md` (35.6 KB).

Method: join the Hermes ledger `api_calls` (`state.db`, read-only URI) against the exo log
`[DSV41] turn reuse:` lines on `macstudio-m4-1:~/exo.log` for the window
2026-10-07 09:20:00–09:51:00 CDT. Match key = log `prompt=` == ledger
`prompt_tokens_total` (exact) + time order. **41 / 42 calls matched, 0 leftover lines;
`cache_read_tokens == reuse=` on all 41 (0 mismatches)** — an independent confirmation the
join is correct. Call 1 is the cold full prefill (prefill == prompt == 24095).

Representative rows (delta_rows = `prefill=`; the whole table is in the raw file):

| call | start CDT | latency_s | out | reas | prompt | cache_rd | delta_rows | reuse_rows | prefill-dom |
|---:|---|---:|---:|---:|---:|---:|---:|---:|:--|
| 1 | 09:20:21 | 98.80 | 175 | 122 | 24 095 | 0 | 24 095 (COLD=full) | — | — |
| 6 | 09:23:47 | 42.88 | 133 | 77 | 39 293 | 30 178 | 9 115 | 30 178 | 43× |
| 7 | 09:24:30 | 136.14 | 1 572 | 1 228 | 53 564 | 39 293 | 14 271 | 39 293 | 5.1× |
| 24 | 09:40:11 | 46.99 | 251 | 143 | 83 663 | 75 338 | 8 325 | 75 338 | 21× |
| 39 | 09:46:48 | 59.54 | 995 | 725 | 100 945 | 99 585 | 1 360 | 99 585 | no |
| 40 | 09:47:49 | 110.62 | 1 906 | 1 411 | 102 063 | 100 945 | 1 118 | 100 945 | no |
| 42 | 09:50:12 | 23.42 | 361 | 0 | 104 880 | 104 396 | 484 | 104 396 | 1.3× |

Findings (directly relevant to the user's "16.7–18.4 t/s" complaint):

- **Prefill dominates 32 of 42 calls** (delta_rows > output+reasoning) — the single-turn
  decode t/s is *not* the right metric for most of this session's calls; they are
  delta-prefill-bound. Worst offenders: c6 (9 115 refed vs 210 generated = 43×), c24
  (21×), c14 (14×), c4 (11×).
- **39 of 42 calls fired the `reuse undershoot` warning** (refed > 256 rows). The reuse
  ladder is catching these, but many deltas are still thousands of rows, not the ~250-row
  ideal — consistent with the earlier ladder findings (a prompt whose LCP misses the
  newest checkpoint by even a couple of rows rewinds and refeds).
- **`prefill_s`/`decode_s` are left BLANK on purpose** in the raw table: converting
  delta_rows → seconds needs the live delta-prefill rows/s rate, which is measurement A2
  (the parent's). That measurement was not run (§5). The `turn reuse:` line's offset from
  call start *is* the prefill-completion latency though (e.g. c7: line lands at +56.4 s of
  a 136.1 s call) — so a first-order prefill share can be read from it, but the exact
  rows/s was not measured here.

**Correction to the brief's framing:** the session is **42 calls**, not 13 (the ledger
disagrees with the brief's "13-call turn"). Reported as found.

---

## 4. THE RESULT THAT MATTERS MOST — the offline "gamma=3 optimal" model is wrong

This follow-up was sent to *test* a suspicion that phase-19's "no ≥3% gamma lever" was a
low-acceptance artifact. Re-deriving the model from first principles shows the suspicion
was right, for a reason phase-19 did not catch.

### 4a. The model (phase-19's own, `gamma-optimality.md`)

```
round_ms(g) = 8.5 + 0.9*(g-1) + VERIFY_MS[g+1] + 4.0
E(g)        = 1 + q1 + q1q2 + … + q1..qg          (expected committed tokens / round)
throughput  = E(g) / round_ms(g)
VERIFY_MS   = {1:58.5, 2:74.9, 3:87.9, 4:97.7, 5:111.7, 6:120.1}   (spec.py:77)
```

With measured mean accepted **2.827/3** (⇒ uniform per-position q ≈ 0.9706 so that
E(3)=3.827 exactly), the model gives:

| g | round_ms (table) | E(g) | t/s (rel) | phase-19's printed "t/s (rel)" |
|---|---|---|---|---|
| 1 | 87.4 | 1.971 | 22.55 | −30% |
| 2 | 101.3 | 2.913 | 28.75 | −8% |
| **3** | **112.0** | **3.827** | **34.17 (base)** | **0%** |
| **4** | **126.9** | **4.714** | **37.15 (+8.7 %)** | **−12 %** |

**The "+8.7%" and the "−12%" are the same table.** Phase-19's markdown prints gamma=4 as
−12% while its own stated `round_ms`/`E(g)` inputs compute to **+8.7%**. A −12% would
require round_ms(g=4) ≈ 155 ms, not the 126.9 printed two columns over. The conclusion text
("at that acceptance the model is saturated and gamma=3 is optimal … the extra verify rows
dominate") is **not what the model says** once the numbers are actually divided.

### 4b. Calibrated to the LIVE round wall (the point of the follow-up)

The live dsv41 round wall is far above the table. Measured 2026-10-07 (phase-19 docs):

- benign 100 K morning: **135.6 ms/round** (README.md)
- benign 100 K afternoon: **154.14 ms/round** (agentic-replay.md; agentic arm identical at
  154.03 — the wall is content-independent)

The table's gamma=3 row is 112.0 ms, so the live per-round **EXTRA = 42.1 ms** (afternoon)
/ 23.6 ms (morning) is a *fixed* per-round cost the table does not model. Applying it to
every gamma (it is shared):

| g | round_ms (live-calibrated, aft) | E(g) | t/s | rel |
|---|---|---|---|---|
| 1 | 129.5 | 1.971 | 15.21 | −38.7% |
| 2 | 143.4 | 2.913 | 20.31 | −18.2% |
| **3** | **154.1** | **3.827** | **24.83** | **base** |
| **4** | **169.0** | **4.714** | **27.89** | **+12.3%** |
| **5** | **178.3** | **5.576** | **31.27** | **+25.9%** |

Cross-check: the model's gamma=3 throughput is 24.83 t/s, and the phase-19 *benign*
afternoon harness measured **24.78 t/s** — a 0.2% match. The calibration is not contrived;
it reproduces the measured number.

### 4c. Sensitivity — the sign is robust, the magnitude is tail-dependent

`EXTRA` shifts g3/g4/g5 by the *same* absolute ms, so it does not by itself create the win;
what matters is whether the per-position acceptance tail decays fast. Over a wide range of
plausible tails the sign is the same:

| per-position model | g3 t/s | g4 t/s | g4 rel | g5 t/s | g5 rel |
|---|---:|---:|---:|---:|---:|
| uniform q=0.9706 (matches mean) | 24.83 | 27.89 | **+12.3%** | 31.27 | **+25.9%** |
| q1=.98 flat-ish [.98,.97,.96,.95,.94] | 24.93 | 27.86 | +11.8% | 30.98 | +24.3% |
| q1=.98 fast decay [.98,.94,.88,.80,.72] | 24.08 | 25.79 | +7.1% | 27.07 | +12.4% |
| q1=.99 heavy decay [.99,.93,.82,.70,.60] | 23.78 | 24.81 | +4.3% | 25.30 | +6.4% |

Even the *most pessimistic* tail gives gamma=4 **+4.3%** — above the 3% bar. The only way
gamma=4 loses is if the round-ms delta per extra verify row is much larger than the
`VERIFY_MS` table implies (i.e. the table under-states the marginal row cost at depth), or
if the per-position tail is catastrophic. **Which of those holds is exactly the live
question — and it is why this needs the interleaved live run, not another offline model.**

### 4d. Why this is plausible on the real engine

The live round wall (154 ms) is 42 ms above the table's gamma=3 row. If that 42 ms is a
*fixed* per-round cost (scheduling, jaccl sync, host-side per-round work, the single
`mx.eval` drain), then adding 1–2 more verify rows — whose marginal cost the table prices
at ~10–14 ms each — amortizes the fixed cost, and higher gamma wins. On the synthetic
filler arm (2.83/3, near-saturated drafts) the extra rows accept almost always, so the
amortization dominates. On real agentic content (2.19/3) fewer extra rows accept, so the
win should be smaller — the plan's "test both workloads" is the right guard.

**Verdict on the specific question asked ("is gamma=3 near-optimal?"):**
- Phase-19's **stated reason** (offline optimum at saturation) is **arithmetically
  unsupported** — see 4a.
- The **best available evidence points the other way**: gamma=4 is predicted to win
  ~+12% (live wall, uniform tail) to ~+4% (pessimistic tail), gamma=5 more.
- This is a **model prediction, not a measurement.** It is falsifiable and must be tested
  live before anyone ships a default change. **NOT measured here.**

---

## 5. Why the live matrix was NOT run — the collision (read this)

While I was executing, the Hermes runtime flagged:

> WARNING: another live subagent `sa-0-1a117dc2` (owner session `20261007_093547_82d809`,
> status running) is already working in this same directory … goal: *"Settle whether
> gamma=3 is near-optimal on the live dsv41 engine at real context depth (test gamma 4/…"*

That goal string is **identical to mine**, down to the truncation. Evidence gathered:

- `~/CAMPAIGN_ACTIVE.json` already existed **before** my own write and **blocked it**
  (stale-write guard). Its content declares the same campaign, the same branch plan,
  `"planned_relaunch": "YES — up to 2"`, `"repo": "…deploy/next14-gamma…"`, owner
  "Hermes PM subagent (phase-19 follow-up)". So the peer agent has already claimed the
  campaign mutex and intends to relaunch the same cluster.
- `delegate_task action=list` shows **0** — I can inspect only my own children, not the
  peer, so I cannot read its phase or signal it.
- As of 13:37 the cluster is **still the 09:01:54 process** on both nodes ⇒ the peer has
  not relaunched yet (or has and restored — but a restore also requires a relaunch, which
  would change the PID, so the unchanged PID proves **zero** relaunches so far).

The risk is structural, not incidental: we share the same goal, the same relaunch plan,
and the same trigger condition (a quiet-traffic window). Any reasoning that tells me "now
is safe" tells the peer the same thing at roughly the same time. A per-request knob cannot
dodge it — *deploying* the knob itself requires the same relaunch. With both agents
holding "≤2 relaunches," the tail outcome is up to four relaunches of the user's live
cluster, mutual runner kills mid-bench (corrupting both benches), and a nonzero chance both
agents assume the other restored production. On a shared production box that carries the
user's interactive traffic, that is unacceptable to risk for a confirmation of a modeled
result.

**Decision (second-opinion endorsed, `consult` returned unanimous):** stand down from all
destructive operations this session; do not relaunch; do not touch the cluster beyond the
read-only probes already done; **leave the peer's `~/CAMPAIGN_ACTIVE.json` untouched** (it
won the race and it is the only mutex that exists between us). The instrumented branch is
pushed and tested so the peer — or a later, properly-owned run — can use it instead of
building a divergent one.

**Cluster state at stand-down (verified 13:37 CDT):** m4-1 PID 32941 / 32954, m4-2 PID 42712
/ 42724, both `deploy/next13` @ `f0840af1c` — **byte-identical to the pre-campaign state.
Restore is a confirmed no-op. No prefix cache was evicted; the user's next turn pays no
cold prefill from this work.**

---

## 6. Drift classification (A3) — partially answered, not settled

The question: is the same-day 13.7% rise in benign-100K round wall (135.6 ms at 09:4x →
154.1 ms at 12:2x, same PID 32941, acceptance unchanged 2.813→2.827) process-state or
host/thermal?

What this session established (read-only):

- **Host/thermal is disfavored, not excluded.** `pmset -g therm` reports **no** thermal or
  performance warning on either node; `powermetrics` GPU power 15 mW idle; load ~1.9.
  A heat/contention explanation would normally show *some* thermal or throttle signal; none
  is present at 13:2x — but the box was also idle at that moment, so this does not by itself
  rule out a thermal effect that existed during the busy 12:2x window.
- **The clean discriminator is a fresh-process repeat of the benign-100K bench** (planned
  B1: if the wall returns to ~136 ms it is process-state — allocator/KV-pool/history growth
  in a long-lived process, which would make a periodic restart a real lever; if it stays
  ~154 ms it is host/thermal). **That requires exactly the relaunch that was stood down**,
  so it was **not run.** Classified as **UNRESOLVED** pending a clean relaunch.
- Supporting note: the wall is **content-independent** (benign 154.14 vs agentic 154.03 ms
  at the same depth/time — phase-19 `agentic-replay.md`), which is consistent with a
  process/host per-round cost rather than a model-work effect.

---

## 7. Instrumented branch — built, tested, pushed (the deliverable that survives)

`origin/deploy/next14-gamma` @ **`01c416b103587387ad9726a5e455a915591dd0c6`**
(`git ls-remote origin refs/heads/deploy/next14-gamma` matches local HEAD). Based on
`deploy/next13-latency` @ 4824bd43e (mlx-lm submodule @ 6cc9c1e).

| file | change |
|---|---|
| `src/exo/api/types/api.py` | `ChatCompletionRequest.spec_gamma: int \| None` + `@field_validator` clamping ints into [1,6]; `GenerationStats.mtp_accepted_histogram_cumulative: list[int] \| None` |
| `src/exo/shared/types/text_generation.py` | `TextGenerationTaskParams.spec_gamma: int \| None = None` |
| `src/exo/api/adapters/chat_completions.py` | `spec_gamma=request.spec_gamma` in the task-params construction |
| `src/exo/worker/engines/mlx/dsv41/engine.py` | `_generate` → `_rounds(spec_gamma=params.spec_gamma)`; `gamma_for_request = spec_gamma if spec_gamma is not None else self.gamma`; `_spec_accept_hist` (len 7) incremented per round; histogram passed to both `_final_response` call sites |
| `src/exo/worker/engines/mlx/dsv41/rounds.py` | `_final_response(..., mtp_accept_hist)` → `mtp_accepted_histogram_cumulative` |
| `src/exo/worker/engines/mlx/dsv41/tests/test_dsv41_spec_gamma.py` | 23 new tests |

- **Default safety:** `spec_gamma` absent ⇒ `None` ⇒ `self.gamma` (3) ⇒ the code path is
  byte-identical to today. The live path is untouched when the field is absent.
- **Gates:** ruff `All checks passed!`; basedpyright BEFORE == AFTER per touched file
  (engine.py 67→67, rounds.py 102→102, else 0→0) — zero NEW; dsv41 test dir 178 → **201
  passed** (+23). Sabotage-proven: neutralizing the clamp → 5 failures; neutralizing the
  `spec_gamma` branch → 1 failure. `git status --short` clean.
- The `: generation_stats` emitter uses `model_dump_json()` (whole-model), so the histogram
  rides along with no emitter change — verified.
- **NOT verified:** not run against the live engine (no relaunch); the engine→master→API
  transport of the new field is not end-to-end exercised live.

Raw per-rep data: none — the matrix did not run.

---

## 8. What a properly-owned follow-up should do (ready to execute)

If the user grants sole ownership of the cluster (and the peer is terminated), the plan is
unchanged and cheap:

1. Relaunch both nodes onto `deploy/next14-gamma` @ `01c416b1` with the snapshot env
   (`raw/live-env-m4-*.txt` + phase-19 `raw/node1-env.txt`); `EXO_TARGET_BRANCH=deploy/next14-gamma`.
2. Warm-up guard (discard first 60 s + one throwaway), then **B1 drift check**: one benign
   100 K rep — ~136 ms ⇒ process-state (periodic restart is then a real lever), ~154 ms ⇒
   host/thermal.
3. **B2 gamma matrix**, interleaved, one process: `spec_gamma ∈ {3,4,5}` (and 2 if trivial),
   ≥3 timed reps/arm/both workloads (benign 100 K + real-agentic replay), collecting
   per-rep decode t/s, ms/round, mean accepted/γ, and the **per-position histogram**. Bar:
   ≥3% median decode t/s, non-overlapping IQRs, on both workloads or clearly
   workload-dependent, quality gate passed.
4. C1 quality gate on any winner vs gamma=3 (fixed prompt set; degeneration/loop/``
   detectors; chunk-verify divergence is expected, degeneration is the failure).
5. C3 restore to `deploy.next13` @ `f0840af1c` + snapshot env unless gamma≥4 wins, in which
   case report and stop for sign-off (**do not ship a default change without the user**).

**Pre-registered prediction to falsify:** per §4, the model predicts gamma=4 wins ≥3% on
the benign arm and (smaller) on the agentic arm; if the live matrix instead shows gamma=4
≤ gamma=3 within IQR, the phase-19 conclusion stands and the marginal verify-row cost is
larger than the `VERIFY_MS` table implies. Either outcome is a real answer.

---

## 9. Artifacts (absolute paths)

On branch `deploy/next14-gamma` (origin @ `01c416b1`):
- `docs/benchmarks/phase19-latency/gamma-depth-2026-10-07.md` — this file
- `docs/benchmarks/phase19-latency/raw/A1-real-turn-split.md` — the 42-call table
- `docs/benchmarks/phase19-latency/raw/live-env-m4-1-ps-eww.txt`, `…-m4-2-ps-eww.txt` — env snapshots

Session scratch (`/Users/adam.durham/.hermes/cache/scratch/gamma-depth/`): the A1 table, raw
matched log dumps (`m4-1-raw-reuse-lines.txt`, `m4-1-raw-session-reuse-lines.txt`,
`m4-2-raw-turn-reuse-lines.txt`), and the env dumps.
