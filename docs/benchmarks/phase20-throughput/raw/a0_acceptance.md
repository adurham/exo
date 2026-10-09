# A0 — Draft-acceptance histogram by draft position (agentic vs benign)

Author: mid-tier subagent (LEAD-A0). Date: 2026-10-09 (CDT). Worktree `/private/tmp/phase20-campaign`.
**OFFLINE, READ-ONLY, 0 boots** — no ssh, no API request, no launch, no implementation. Every number is
re-derived from local files; source cited inline.

---

## 0. Headline / honesty statement (read first)

- **Per-ROUND `n_accepted` histogram (0..3): AVAILABLE** for both arms (§1).
- **Per-POSITION acceptance: AVAILABLE**, but only as the *survival* vector `S_k = P(n_accepted ≥ k)`
  (equivalently `q_k`, the implied per-position conditional accept prob). It exists in
  `raw/gamma-matrix.json` (phase-19), benign **and** agentic, gamma 3/4/5 (§2). It is derived from the
  coarse 0..gamma round histogram (`per_position` producer: `bench/phase19_gamma_matrix.py:191`), **not**
  from a direct per-position tap.
- **Per-CONTENT-TYPE breakdown: NOT AVAILABLE beyond a 2-way split** (benign-synthetic vs agentic-real).
  There is **no** tool-call-JSON vs prose vs echoed-tool-output labeling anywhere in the local data — no
  file tags rounds or accept positions by content kind (§4). **Therefore the data CANNOT resolve whether
  echoed tool output dominates the rejects; no prompt-lookup / n-gram-drafting claim is supported.**
- There is **no per-position × content-type cross-tab** in local data.

---

## 1. Per-ROUND `n_accepted` histogram (0..3) — the coarse acceptance distribution

`n_accepted` = number of the γ=3 drafted tokens accepted in one round (0..3). `committed/round = 1 + mean_accepted`.
Source files carry only `{round_idx, gamma, n_accepted, draft_build_ms, verify_block_ms, tail_bookkeep_ms, round_total_ms, emit_ms, rank}` — **no content label**.

| arm | source | n rounds | n=0 | n=1 | n=2 | n=3 | mean acc | committed/round |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **agentic 91K g3** (real session) | `raw/p3/off_s1_rank1.round_prof.jsonl`, `verify_block_ms ≥ 95 ms` stratum | 1056 | 152 (14.4%) | 180 (17.1%) | 165 (15.6%) | 559 (52.9%) | **2.071** | **3.071** |
| **benign 20K g3** (synthetic) | same file, `verify_block_ms < 95 ms` stratum | 2016 | 92 (4.6%) | 83 (4.1%) | 60 (3.0%) | 1781 (88.3%) | **2.751** | **3.751** |
| benign 20K g3 (levers OFF) | `raw/p3/levers_off_measure.json` (Δ of `mtp_accepted_histogram_cumulative`) | 2030 | 92 (4.5%) | 84 (4.1%) | 70 (3.4%) | 1784 (87.9%) | **2.747** | **3.747** |
| benign 20K, next16-instr PROF | `raw/prof/m4-1.round_prof.jsonl` (m4-2 identical) | 480 | 142 (29.6%) | 122 (25.4%) | 48 (10.0%) | 168 (35.0%) | **1.504** ⚠️ | 2.504 ⚠️ |

⚠️ **Provenance flag (triage):** the `raw/prof/*` run's `n_accepted` mean is **1.504**, which *disagrees*
with the clean benign measurement documented in `PHASE1-M1.md:7` (`mean_accepted 2.756`, same build, same
depth). Reported verbatim; **do not** treat the `raw/prof` run as benign acceptance. The p3 and
`levers_off_measure` rows agree on ~2.75 and match the docs — use those.

**Cross-check vs arm-level stats (`mean accepted = mtp_accepted_drafts/cycles`):**
- `raw/p5r1c/ship_smoke_agentic.json` (real agentic, 91K, g3): **2.0651** — matches the p3 agentic stratum (2.071).
- `raw/p5r1c/ship_smoke_benign.json` (20K, g3): **2.8239** (rep1 2.7163 / rep2 2.9314) — matches the p3 benign stratum (2.751).
- `raw/p3/levers_off_measure.json` (pooled): pooled mean acc **2.4394**; split → benign **2.747**, agentic stratum 2.071.

**Pooling caveat (load-bearing):** the four `raw/p3/*.round_prof.jsonl` files are **byte-identical**
(md5: `off_s1==on_s1==948fcda1…`, `off_s2==on_s2==51b01454…`). The `--arm both` driver wrote **one shared
rank file** interleaving the benign (20K) and agentic (91K) arms; they are separated here only by the
`verify_block_ms` bimodality (benign ≈92 ms / agentic ≈97.8 ms; ~24 transition rounds straddle the 95 ms
split; 8 round_outlier rows with `round_total_ms ≥ 200 ms` are excluded by the threshold). The `on_*`
copies were **not** independently measured.

---

## 2. Per-POSITION acceptance (survival vector) — the real per-position data

Source: `/private/tmp/next14-gamma/docs/benchmarks/phase19-latency/raw/gamma-matrix.json` (field
`per_position` / `per_position_median`), produced by `bench/phase19_gamma_matrix.py:191`
(`S_k = sum(hist_delta[k:]) / total_rounds`). Live interleaved run 2026-10-07 14:14–14:42 CDT, depth 100K,
temp 0, gamma set per-request via `spec_gamma`. **This is the per-ROUND `hist_delta` read as a survival
vector, not a direct per-position tap.** Consistency: `Σ_k S_k = mean_accepted` holds exactly (e.g.
agentic:3 → 0.879+0.729+0.580 = 2.188).

| arm | S₁ (p1) | S₂ (p2) | S₃ (p3) | S₄ (p4) | S₅ (p5) | mean acc | ms/round | t/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **agentic:3** | 0.879 | 0.729 | 0.580 | — | — | 2.188 | 153.8 | 20.69 |
| agentic:4 | 0.854 | 0.651 | 0.505 | 0.380 | — | 2.390 | 173.5 | 19.52 |
| agentic:5 | 0.852 | 0.656 | 0.548 | 0.393 | 0.263 | 2.711 | 183.9 | 20.12 |
| **benign:3** | 0.976 | 0.945 | 0.906 | — | — | 2.827 | 154.3 | 24.76 |
| benign:4 | 0.897 | 0.842 | 0.778 | 0.761 | — | 3.278 | 174.3 | 24.49 |
| benign:5 | 0.922 | 0.859 | 0.828 | 0.807 | 0.787 | 4.203 | 185.4 | 28.07 |

**Implied per-position conditional accept prob `q_k = S_k / S_{k-1}` (γ=3):**
- agentic: q1=0.879, q2=0.830, q3=0.795 → survival decays across positions.
- benign:  q1=0.976, q2=0.968, q3=0.958 → near-flat.

**Reading:** at γ=3 the agentic survival drops to 0.580 by position 3 vs benign 0.906 — that per-position
decay *is* the acceptance mechanism behind the agentic throughput gap. (This reproduces
`gamma-depth-2026-10-07.md:44`, which cites agentic tail p4=0.38 / p5=0.26.)

---

## 3. GAMMA-PRICING note (analysis only — NO implementation)

**Mechanism & config-only claim — verified.** γ is a **per-request field `spec_gamma`** (server clamps
[1,6]) on branch `deploy/next14-gamma`; changing it is config/request-only, **no code**. Live proof:
`gamma-fieldcheck.json` — one request WITH `spec_gamma=4` → `hist_delta [16,7,5,8,25,0,0]` (top non-empty
bin = 4); one WITHOUT → `hist[4]=0` (structurally impossible under γ3) ⇒ γ4 honoured, absent-field = γ3.

**γ4 vs γ3 is already priced LIVE (both workloads).** From `gamma-matrix.json` `by_arm`:

| workload | γ3 t/s | γ4 t/s | Δ | γ5 t/s | Δ |
|---|---:|---:|---:|---:|---:|
| benign 100K | 24.756 | 24.489 | **−1.1%** | 28.070 | +13.4% |
| agentic (real) | 20.692 | 19.519 | **−5.7%** | 20.116 | −2.8% |

IQRs fully disjoint. **A static γ4 is WORSE than γ3 on both workloads** — it pays one extra verify row
(round 154.3 → 174.3 ms) for almost no extra acceptance (agentic S₄=0.380). **γ is pinned at 3 on the
serving path, and the prior data already shows no static γ4 lever exists.**

**γ2 is NOT in the live matrix (matrix = 3,4,5).** It is priced only by the offline model
`phase19-latency/gamma-optimality.md` (`round_ms(g)=8.5+0.9·(g−1)+VERIFY_MS[g+1]+4.0`,
`VERIFY_MS={1:58.5,2:74.9,3:87.9,4:97.7,5:111.7,6:120.1}`):
- at benign saturation γ2 ≈ **−8%** vs γ3;
- only in a *much worse* acceptance regime (mean ≈1.27/3) does γ2 open to **≈+5%** — and that requires a
  static pin change (γ is not adaptive on dsv41).
No live γ2 measurement exists.

**Does prior gamma-matrix data already price a static change? YES.** `gamma-matrix.json` prices γ3/4/5 live
on both workloads; `gamma-optimality.md` prices γ1..6 offline; `gamma-depth-2026-10-07.md §4c` gives a
per-position-tail sensitivity table (γ4 = **+7.1% to +12.3%** across *optimistic* tails — falsified live,
where γ4 lost); phase-18 `p63_spec2.py` + `p63-run1.log` show *adaptive* γ picking γ2 on low-acceptance
prompts (acc 0.90–1.29, γ mix `{2: 75–90}`) for +1.1 tok/s mean.

**Adaptive γ: NOT-FUNDED / feature-blocked.** `GammaPolicy.update()` is never called on the serving path
(dsv41 engine γ hard-pinned at 3); wiring it is new code → out of scope. This section is analysis only.

---

## 4. Content-type attribution — honest "cannot resolve"

A **2-way** split exists: **benign-synthetic** (12 canned sentences + "count 1..450") vs **agentic-real**
(the flattened real session `20261007_092009_9a2ed7` = system preamble 22,770 ch + 98 messages with
`content` + `reasoning_content` + `tool_calls`, rendered into one user message — i.e. **code, terminal
output, tool JSON, and prose all mixed**, `agentic-replay.md`).

There is **no** finer label — no local file tags rounds or accept positions as tool-call-JSON vs prose vs
echoed-tool-output. The `round_prof` files carry no content field; the arm JSONs carry only workload-level
meta (`n_messages_total`, `preamble_chars`, `prompt_chars`); the histogram metric
(`exo_mtp_acceptance_bucket_total`) buckets only by `accepted` count (`metrics.py:149-154`).

**Conclusion:** the aggregate per-position decay (§2) is consistent with *hard agentic content*, but the
data **cannot attribute the rejects to echoed tool output**. Do **not** claim prompt-lookup / n-gram
drafting is evidenced — that remains a *untested* future feature-gated hypothesis.

---

## 5. Files & verifications

- Wrote `raw/a0_acceptance.md` (this file) and `raw/a0_acceptance.json` (machine-readable).
- Re-derived: p3 histogram + benign/agentic strata (Python, exact); `levers_off_measure` cumulative-delta
  histogram (Σhist=rounds, Σk·hist=accepted verified); `gamma-matrix.json` per_position (Σ S_k = mean_acc
  verified). All source paths absolute and cited inline.
- **Not verified / blockers:** no per-content-type data exists; `raw/prof/*` n_accepted mean (1.504)
  disagrees with the documented benign 2.756 (flagged); p3 `on_*` files are byte-identical copies of
  `off_*` (md5), and the shared file pools two workloads separated only by a `verify_block` bimodality.
