# Survivorship sanity check on the 21.9% "model's own completions re-prefilled" figure

Data: `tmp/real-usage-capture-20260902/phase1/requests.jsonl` (57 records, provenance-wrapped; 55 main-chat rows = call_seq 33-87 + 2 aux). Read-only, no cluster/inference. This file contains **ONE contiguous warm run**: all 57 rows carry `instance_id = 339f04f8-...`. The earlier instance `25ae372c` (32 calls, same client session, before the runner relaunch) is **not** in this file.

Every number below is **MEASURED** (from `requests.jsonl`) or **DERIVED** (arithmetic on measurements, labeled). Arithmetic shown so any reader can reproduce.

---

## 1. Reproducing the pooled 21.9% — CONFIRMED, exact

`uncached(n) = prompt_tokens(n) − cached_tokens(n)`. All sums over the 55 main rows (2 aux excluded — they are separate tiny helper threads, not this chat).

| line | value |
|---|---|
| total prompt presented | 7,883,591 | MEASURED (sum prompt_tokens) |
| served from cache | 7,694,581 | MEASURED |
| **uncached prefill** | **189,010** | DERIVED = 7,883,591 − 7,694,581 |
| Bucket A (cold start) | 92,594 | MEASURED = uncached of call_seq 33 only (`prefix_cache_hit="none"`, `cached=0`) |
| Bucket B (= Σ completion(n−1), n=2..55) | 41,414 | DERIVED = Σ int(completion_tokens[i−1]) over the 54 turn pairs |
| Bucket C (residual) | 55,002 | DERIVED = 189,010 − 92,594 − 41,414 |
| **B share of uncached** | **21.911%** | DERIVED = 100 × 41,414 / 189,010 |

Check: A+B+C = 92,594 + 41,414 + 55,002 = 189,010 ✓. Residual per pair (`uncached(n) − completion(n−1)`) is **positive on all 54 pairs**, median 447 (min 29) — matching the prior report's §2 (median 446). The pooled 21.9% **reproduces exactly**. The prior figure is not an arithmetic error.

**Critical framing caveat for this check:** "pooled across 7,883,591 tokens" is **itself a single-session pool** — there is only one session in this file. So "pooled 21.9%" and "this session's 21.9%" are the *same number*. The survivorship question is therefore not "does pooling distort across sessions" (unanswerable from n=1 session) but "is 21.9%, earned at the tail of a 55-turn run, representative of a *typical* session's value from Fix B?" — and that is a **length-dependence** question, which this data *can* bound (next section).

---

## 2. Length-dependence: bucket B's share rises monotonically with run length (the survivorship effect)

Fix B's payoff is `Σ completion(n−1)` — a **per-turn cost that accumulates with turn count**. A cold start (bucket A, 92,594 tokens) is a **one-time, per-session fixed cost**. So for a run of length L (1 cold start + k = L−1 warm pairs), B-share = `Σ_{i=1..k} completion(i−1) / ( 92,594 + Σ_{i=1..k} uncached(i) )`.

Measured B-share vs. run length (DERIVED, monotone increasing):

| run length L (main turns) | warm pairs k | B-share of that run's uncached |
|---|---|---:|
| 2 | 1 | 2.09% |
| 3 | 2 | 3.11% |
| 4 | 3 | 3.43% |
| 6 | 5 | 4.23% |
| 11 | 10 | 5.61% |
| 21 | 20 | 7.53% |
| 31 | 30 | 12.25% |
| 41 | 40 | 16.99% |
| 51 | 50 | 21.58% |
| **55** | **54** | **21.91%** |

Monotonicity is structural, not a quirk of this run: `uncached(n)` is bounded below by `completion(n−1)` (all 54 residuals positive), while bucket A is a fixed 92,594 one-time term, so B/A-share strictly increases as the warm-run grows. To hit each threshold from this run's own turn ordering: **k = 7 warm pairs → ~5%, k = 25 → ~10%, k = 39 → ~15%, k = 50 → ~20%, all 54 → 21.9%.** A session needs roughly **50 warm turns before it earns a 20%+ B-share** in this data.

"First N turns vs turns N+" (MEASURED, split of the run's 54 warm pairs):

| segment | bucket-B tokens | that segment's warm uncached | B-share of segment (warm-only) | B-share of full session incl. cold A |
|---|---:|---:|---:|---:|
| first 27 warm pairs | 16,641 | 53,378 | 31.18% | 11.40% |
| last 27 warm pairs | 24,773 | 43,038 | 57.56% | 18.26% |

Incremental ceiling: **warm-only** (excluding the one-time cold start) B/(B+C) = **42.95%** — this is the share of per-warm-turn uncached prefill that would be reclaimed once a session is fully ramped and bucket A amortizes to ~nothing.

The incremental (steady-state, per-warm-turn) B share is **~43%**; the *session-aggregate* number (21.9%) is dragged down by the one-time cold-start term. The two numbers answer different questions: 43% = "once warm, what fraction of my per-turn prefill does Fix B reclaim"; 21.9% = "what fraction of *this whole run's* prefill work (incl. cold start) was bucket B."

---

## 3. The reviewer's question answered precisely

**Does 21.9% OVERSTATE, UNDERSTATE, or fairly represent the per-session expectation?**

- **The number itself is exact and honestly labeled** in the source report (REPORT.md §2 calls it "of this session's prefill work").
- **The survivorship concern is directionally VALID for bucket B, via length, not via cross-session pooling.** Pooling *within* a single long session does not fabricate anything — but using "21.9%" as if it were the *expected savings of Fix B for a typical/aggregate session* **overstates the value proposition if deployed sessions are typically shorter than this run.** B-share scales from ~2% (a 2-turn session) to ~22% (a 55-turn session).
- Conversely, if sessions are typically **longer-lived** than this one (this run had long user-idle gaps: 68m/51m/32m), then 21.9% **understates** Fix B's value — the incremental warm-only share is ~43% and only rises the longer a session survives.
- **Verdict: UNDETERMINED from this data, direction depends on the true session-length distribution, which n=1 session cannot establish.** What IS established (DERIVED, monotone): B-share as a function of run length is a steep, increasing curve, so any aggregate figure requires weighting by the real session-length distribution. Treat "21.9%" as an **upper-bound-ish value-proposition number anchored to an atypically long run**, not a fleet-average expectation.

**Concrete bound we can state (honest, DERIVED):** for any collection of sessions where the average session-length is L turns, Fix B's aggregate uncached-reclaim fraction is bounded between the value at run length L and the incremental ceiling. E.g. mean session length 6 → ≲ 4.2%; length 21 → ≲ 7.5%; length 55 → 21.9%; fully-ramped steady state → 42.95%.

---

## 4. What this file CANNOT establish, and what WOULD settle it

This capture is **one client session with one relaunch** (instance 25ae372c → 339f04f8; only the post-relaunch 55-turn window is in this file). A single session **cannot establish a cross-session distribution** — that is acknowledged outright.

**What would settle it** (proposed data collection):
- **N ≥ ~20-30 full sessions**, each captured to completion (or a stable cutoff), all with the same per-request fields already present in requests.jsonl: `seq_client_call`, `instance_id`, `prompt_tokens`, `cached_tokens`, `completion_tokens`, `prefix_cache_hit`.
- Then the fleet B-share is the **length-weighted** expectation: `Σ_sessions (ΣB / Σuncached)`, or equivalently weight each session's B-share by its total uncached tokens — this both corrects for survivorship and handles the bucket-A-per-session fixed cost.
- Report both (a) unweighted across sessions and (b) uncached-token-weighted, side by side.
- Also record **relaunch frequency** (how often `instance_id` changes mid-client-session) — that governs bucket A amortization and is independent of B.

**Do not** estimate the session-length mix from this one run — it is one draw from an unknown distribution, with long idle gaps (3h31m wall, ~2h of it idle) that likely make it unrepresentatively long.

---

## 5. Bottom line

1. **21.9% is correctly computed and exactly reproduces (21.911%).** Not an overstatement by arithmetic.
2. But it is a **single long-session** number (55 turns). Fix B's value is **length-dependent and monotonically increasing** in run length; as a *per-session expectation* it is **NOT established by this data**.
3. If typical sessions are shorter than ~50 warm turns, claiming 21.9% as the aggregate Fix B value **overstates** it (a mean session length of 6 → ~4%, of 21 → ~7.5%). If sessions are comparably long-lived, it is representative or even conservative (fully-warm incremental share ≈ 43%).
4. Fix B remains the **safest first build** (in-process, no cross-node serialization, evenly spread across turns) — but its *size* should be quoted as **"~21.9% in this 55-turn session; length-dependent; fleet value requires the length-weighted average over N≥20-30 sessions,"** not as a flat fleet expectation.
5. Bucket A (49%, from a single relaunch) is the *more* survivorship-laden bucket: one event per session, its aggregate value governed purely by relaunch frequency. The reviewer's survivorship instinct is correct in general; here it bites both buckets, and only A is the one where even the *direction* (under-vs-overstatement) depends on relaunch cadence.

**Files written:** this report + `survivorship.json` (key numbers).
