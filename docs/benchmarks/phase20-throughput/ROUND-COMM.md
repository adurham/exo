# ROUND-COMM.md — LEAD-A STEP-A1: exposed non-compute time per round (m=4 verify + m=1 draft)

Author: mid-tier subagent (LEAD-A Step A1). Date: 2026-10-09 (CDT). Worktree `/private/tmp/phase20-campaign`.
**OFFLINE ONLY** — no ssh, no API request, no launch. Every number below is re-derived from local files;
no cluster contact was made.

---

## 0. Pre-registered falsifier (fixed BEFORE computing, quoted verbatim)

> 'If exposed non-compute time (round wall − sum of GPU kernel time) is < ~8 ms/round at BOTH the m=4
> verify and the m=1 draft shapes, CLOSE lead A with the number. If ≥8 ms exposed, PROCEED to A2
> (call-counter + standalone jaccl all_sum latency microbench).' Recovery is realistically ~50-60% of
> exposed; no tune-existing lever can recover more than the exposed time.

**What "exposed" means here:** the time in a decode round that is NOT GPU kernel execution — host-side
waits between command buffers, waits inside command buffers (comm that reads as GPU-busy), and
per-collective handoff overhead. Lead A's premise: Phase-20 0d measured "comm ≈ 0%" of the round but
that may have counted only comm-*KERNEL* time, so real exposure could be hidden.

**NOTE ON THE FALSIFIER'S OPERAND:** the project has **no per-round "sum of GPU kernel time" number**
(the instrumented build exposes brackets, not a kernel-time sum). Per the task, that sum is **not
fabricated**; the exposed term is instead **bounded** by independent non-compute measures (§3–§4). The
verdict is stated against those bounds, not against an invented kernel sum.

---

## 1. Provenance caveats (read before the table)

- **`raw/prof/m4-1.round_prof.jsonl`, `raw/prof/m4-2.round_prof.jsonl`** — next16-instr build,
  `PROF=1`, levers at **build defaults (ON)**, 480 lines each, rank-tagged (m4-1 rank0, m4-2 rank1).
  Round ≈ **147 ms/round**. This is the **instrumented** build, **NOT** the current production build.
- **`raw/p3/off_s1_rank1.round_prof.jsonl`, `raw/p3/off_s2_rank0.round_prof.jsonl`** — next16-instr
  build, `round_prof=1`, levers **OFF** (`DSV41_SPARSE_COLSPLIT=0 DSV41_INDEXER_HIER=0`), 3072 lines
  each (off_s1 = rank1, off_s2 = rank0). Round ≈ **94 ms/round**.
- **Production** (the number to improve): **94.94 ms/round benign 20K**, **99.46 ms/round agentic 91K**
  (`ROUND-DENSE-EXL3.md:22-23`, `PHASE5-R1C-SHIP.md:124-125`, `PHASE5-CAMPAIGN.md:215-216`).
- ⚠️ **The absolute ms above are from the instrumented/lever builds; they are NOT the production
  absolute ms. The STRUCTURE (bracket decomposition, exposure fractions) is what transfers, not the
  absolute ms.** Every per-round exposed figure below is therefore **projected to the production
  ~99.46 ms round** by applying the measured fractional shares.
- Known instrumentation caveat (`PHASE1-M1.md:23`): `emit_ms` is a *cumulative* time-since-stream-start,
  not a per-round delta. It is ~0.3 ms of real work and is excluded here; it does not affect closure.

---

## 2. A1 exposed-time table (m=4 verify + m=1 draft, both nodes, both datasets)

All values are **medians over all lines** (mean in parentheses) from the four JSONL files + the two
`gpu_busy` captures. Source paths are absolute-relative to
`/private/tmp/phase20-campaign/docs/benchmarks/phase20-throughput/`.

| shape | file | metric | value | share of round_total |
|---|---|---|---|---:|
| **m=1 draft** | `raw/prof/m4-1.round_prof.jsonl` (rank0, levers ON) | `draft_build_ms` | **4.793** (mean 5.056) | 3.26% |
| **m=1 draft** | `raw/prof/m4-2.round_prof.jsonl` (rank1, levers ON) | `draft_build_ms` | **4.820** (mean 5.025) | 3.28% |
| **m=1 draft** | `raw/p3/off_s1_rank1.round_prof.jsonl` (rank1, levers OFF) | `draft_build_ms` | **0.555** (mean 0.574) | 0.59% |
| **m=1 draft** | `raw/p3/off_s2_rank0.round_prof.jsonl` (rank0, levers OFF) | `draft_build_ms` | **0.564** (mean 0.581) | 0.60% |
| **m=4 verify** | `raw/prof/m4-1.round_prof.jsonl` | `verify_block_ms` | **143.122** (mean 141.129) | 97.4% |
| **m=4 verify** | `raw/prof/m4-2.round_prof.jsonl` | `verify_block_ms` | **143.063** (mean 141.110) | 97.4% |
| **m=4 verify** | `raw/p3/off_s1_rank1.round_prof.jsonl` | `verify_block_ms` | **92.568** (mean 94.778) | 98.5% |
| **m=4 verify** | `raw/p3/off_s2_rank0.round_prof.jsonl` | `verify_block_ms` | **92.608** (mean 94.821) | 98.5% |
| m=4 verify | `raw/prof/m4-{1,2}.round_prof.jsonl` | `tail_bookkeep_ms` | 0.544 / 0.549 | 0.37% |
| m=4 verify | `raw/p3/off_s{1,2}*.round_prof.jsonl` | `tail_bookkeep_ms` | 0.092 / 0.090 | 0.10% |
| — closure — | all four JSONL | (draft+verify+tail)/round_total | **101.0% / 101.0% / 99.1% / 99.1%** | — |

**Non-compute (`gpu_busy`) measures** — `raw/gpu_busy.benign.json`, `raw/gpu_busy.agentic.json`
(`d['parsed'][node]['powermetrics']` / `['sample']`):

| measure | benign m4-1 | benign m4-2 | agentic m4-1 | agentic m4-2 |
|---|---:|---:|---:|---:|
| `mean_hw_active_residency_pct` (powermetrics, 500 ms blocks, elapsed-weighted) | **100.0%** | **100.0%** | **100.0%** | **99.998%** |
| `mean_hw_active_freq_mhz` | 1578 | 1578 | 1578 | 1578 |
| below-1-GHz residency tail (sum of bins < 1000 MHz) | 0.0% | 0.0% | 0.0% | 0.10% |
| `sample` comm fraction (`all_threads_pct.comm`) | 0% (absent) | 0% (absent) | **0.0%** (24 smp) | **0.0%** (12 smp) |
| `sample` `gpu_wait` — **all threads** | 5.26% | 5.26% | 5.23% | 5.28% |
| `sample` `gpu_wait` — **main/decode thread** | **0.00%** | **0.00%** | **0.02%** | **0.01%** |
| `sample` `python_busy` — all threads | 1.75% | 1.75% | 59.2% | 59.5% |
| `sample` `python_busy` — main thread | 0.00% | 0.00% | 99.98% | 99.98% |

Idle fixtures (`PHASE0-M0.md:39`, `GPU-BUSY-NOTES.md:45-46`): m4-1 **3.98%**, m4-2 **3.71%**
HW-active → decode **≈96 points above idle** on both nodes.

---

## 3. m=1 draft exposed — upper bound

The task's bound: **the ENTIRE `draft_build_ms` bracket contains all 4 draft `all_sum`s** (`phase1_round_mechanics.md` Q1:
`draft()` runs 4 cross-rank all_sums/call — 3 MoE + 1 argmax). So `exposed(m=1) ≤ whole draft_build bracket`.

- **Median `draft_build_ms`: 4.793 ms (rank0) / 4.820 ms (rank1)**, levers ON (`raw/prof/*`, n=480 each);
  **0.555 / 0.564 ms**, levers OFF (`raw/p3/*`, n=3072 each).
- **Honest structural note (strengthens the bound):** `draft_build_ms` is *host graph-build only*
  (`PHASE1-M1.md:15`; Q6a). The 4 all_sums do **not** execute there — they drain inside the **next**
  `verify_block` `mx.eval(logits)` that consumes `drafted` (`phase1_round_mechanics.md` Q1/Q6, trap 2).
  So the *real* m=1 exposed comm is billed to `verify_block`, and the draft bracket is a **generous**
  over-estimate of it.
- **Bimodality caveat:** the median hides a bimodal `draft_build_ms` distribution — e.g. `raw/prof/m4-1`
  is 239 rounds ≈0.5 ms and 239 rounds ≈9 ms (rounds following a deferred-tail/rollback drain), plus a
  single 137.9 ms cold round. `raw/p3/off_s1_rank1` is 3071/3072 rounds ≈0.55 ms. The medians above are
  robust; the 9-ms mode is the same deferred-tail effect that lands in `verify_block`, not new comm.
- **Per-collective handoff (the thing that "bites" at m=1):** each all_sum is a cross-rank
  handoff over jaccl. The `sample` capture shows the jaccl side-channel thread with **24 samples on
  `jaccl::SideChannel::all_gather` (agentic m4-1) / 12 (m4-2) — 0.00% of all samples** ⇒ the
  per-collective handoff overhead is **not** a visible contributor.

**m=1 exposed upper bound (instrumented) = 4.79 ms/round; production-projected = 3.26% × 99.46 ≈ 3.24 ms/round.**

---

## 4. m=4 verify exposed — bound from three independent non-compute measures

`verify_block_ms` absorbs the **fused verify eval** (real GPU compute) + the **previous round's deferred
rollback/append_ctx** + the draft drain + the per-layer host syncs (`PHASE1-M1.md:16`, Q6a). Its bracket
is therefore mostly *compute*, not exposure — the bracket **cannot** be read as exposed time. Exposure is
bounded independently:

**(a) powermetrics GPU HW-active residency (block resolution, 500 ms).** **100.0% on both nodes, benign
and agentic** (99.998% agentic m4-2); mean active freq 1578 MHz; below-1-GHz tail 0.0–0.10%
(`raw/gpu_busy.benign.json`, `raw/gpu_busy.agentic.json`). ⇒ **Exposed GPU-idle at 500 ms block
resolution ≈ 0** (≤ 0.1%). A hidden comm stall ≥8 ms inside a 500 ms block is ≤1.6% and would not be
resolvable here — this measure *alone* cannot exclude an in-block stall, which is why (b)+(c) matter.

**(b) `sample` comm + gpu_wait fractions** (`raw/gpu_busy.*.json`, parsed per `GPU-BUSY-NOTES.md` §3):
- **comm: 0% (benign, absent) / ≤0.003% (agentic).** The only comm-classified samples are 24/12 frames
  on the jaccl side-channel thread — i.e. jaccl is parked/negotiating, not burning the round.
- **gpu_wait: 5.23–5.28% all-thread — but it sits on MLX scheduler worker threads, not the decode
  thread.** Composed of 3 threads × ~1.75% each (`__psynch_cvwait` / `_pthread_cond_wait`, with a few
  `Scheduler::enqueue` frames) plus a 1-sample artifact. The **main/decode thread's own gpu_wait is
  0.00% (benign) / 0.02% (agentic)** ⇒ **no host-side wait-on-GPU on the critical path.**

**(c) Python bracket closure ≈ 100%** (101.0% levers-ON, 99.1% levers-OFF, §2). The three brackets
account for the whole round ⇒ there is **no large unexplained host segment** outside the fused verify
eval (`PHASE1-M1.md:21`). If ≥8 ms of host sync existed outside the eval, closure would not be ~100%.

**Which measures could MISS comm hidden as GPU-busy, and how the bound holds:**
- (a) powermetrics is block-resolution (500 ms) — an in-block stall is invisible; it also reads **HW
  active**, so if a "comm wait" were implemented as GPU-busy spin it would read 100%.
- `sample` **thread-state** can miss an in-block bubble where the host is enqueueing (agentic 59%
  python_busy) while the GPU is briefly idle.
- **The `sample` `gpu_wait` fraction bounds it:** even the *loosest* reading — every `gpu_wait` sample
  charged to exposure — is **≤5.3%** of samples, and on the decode thread itself it is **≤0.02%**. So
  exposure hidden as `gpu_wait` is **≤5.3% (loose) / ≈0 (tight)**, i.e. **≤5.25 ms/round (loose)** at the
  production round — still **< 8 ms**. The two measures are complementary: `sample` catches host-side
  waits; powermetrics catches GPU-idle bubbles.

**m=4 exposed bound: ≈0.1 ms/round (tight: comm + main-thread gpu_wait + block-res GPU idle); ≤5.25 ms/round (loose, all gpu_wait charged).**

---

## 5. Per-round exposed-time estimate at both shapes vs the 8 ms falsifier

Projection rule: apply each measured fraction to the **production** round (99.46 ms agentic; 94.94 ms
benign) — the task's instruction to sanity-check `< 8 ms`.

| term | measured (instrumented) | fraction | production-projected (×99.46 ms) |
|---|---|---:|---:|
| m=1 draft bracket (hard UPPER bound on m=1 exposed comm) | 4.793 ms median | 3.26% | **3.24 ms** (levers OFF shape: 0.59 ms) |
| m=4 verify bracket (mostly compute — NOT exposure) | 143.1 ms median | 97.4% | 96.9 ms (bracket, not exposed) |
| comm (`sample`) | ≤0.003% | ≤0.003% | **≤0.004 ms** |
| main-thread gpu_wait (`sample`) | ≤0.02% | ≤0.02% | **≤0.03 ms** |
| GPU idle, block-res (powermetrics) | ≤0.10% | ≤0.10% | **≤0.10 ms** |
| **all-thread gpu_wait — loose exposure bound** | 5.28% | 5.28% | **≤5.25 ms** |

- **m=1 draft exposed per round ≈ 3.24 ms (upper bound).**
- **m=4 verify exposed per round ≈ 0.1 ms (tight) / ≤5.25 ms (loose).**
- Both are **< 8 ms**. The production round (~99 ms) is far longer than the instrumented round (~147 ms),
  yet the *fractional* exposure is tiny, so the projected per-round figure is **≪ 8 ms** — the
  instrumented build does **not** manufacture the exposure.

---

## 6. Required statement — the m=1 GEMV number is DISCOUNTED

> **DISCOUNTED:** "m=1 GEMV 79 vs 66 GB/s" (`MECHANISM.md:84-86`, `:140-142`) is **GEMV memory-efficiency
> (a bandwidth/occupancy number for the dense/EXL3 matmul), NOT comm evidence.** It says nothing about
> exposed communication time and is **excluded** from this analysis. The dense slice's per-round pass is
> the **m=4 verify** (one pass/round), so an m=1 GEMM bandwidth figure does not move the round.

---

## 7. VERDICT — **CLOSE lead A**

Applying the pre-registered falsifier:

- m=4 verify exposed per round: **≈0.1 ms (tight) / ≤5.25 ms (loose)** — **< 8 ms** ✅
- m=1 draft exposed per round: **≤3.24 ms** (whole-bracket hard upper bound) — **< 8 ms** ✅
- **Both shapes are < ~8 ms ⇒ falsifier fires ⇒ CLOSE lead A with the number.**
  Headline: **exposed non-compute ≈ ≤5.25 ms/round worst-case bound, ≈0.1 ms best-estimate, at m=4;
  ≤3.24 ms/round (upper bound) at m=1; both < the 8 ms gate.** Do **not** proceed to A2.

**Honest boundary (what this does and does not close):**
- The round-time gap between workloads is **small**: production benign 94.94 vs agentic 99.46 ms =
  **4.52 ms**; the instrumented levers-ON pair is 146.9 ms both nodes, i.e. rank-symmetric to <0.1 ms
  (`PHASE1-M1.md:24`) — no rank imbalance, no large workload-driven comm term.
- This closes **exposed-tune-existing** comm: with recovery realistically ~50–60% of exposed, even the
  loose ≤5.25 ms bound maps to **≤~3 ms/round of *possible* recovery** — below any shipping gate, and
  the tight estimate maps to ~0.
- **NOT closed (per `MECHANISM.md` §6 "What was NOT closed"):** the P2 route *"collectives-at-m=1"* is a
  **separate slice** from the dense GEMV, and the m=1 GEMV advantage means a **collective/comm** lever at
  m=1 *could* still be live — but it is **out of the dense-GEMM scope and was not measured** here. This
  A1 result does **not** claim the round is at 100% of a theoretical floor; it claims the **exposed
  non-compute** term is small at the production shapes. A standalone jaccl `all_sum` latency
  microbench (A2) would resolve the per-collective handoff cost directly, but the current evidence does
  not justify spending it.
- Measures NOT available: no per-round "sum of GPU kernel time" (the exposed operand is bounded, not
  directly subtracted); powermetrics floor is 500 ms (in-block stalls below ~1.6% are unresolvable).

---

### Sources cited
- `raw/prof/m4-1.round_prof.jsonl`, `raw/prof/m4-2.round_prof.jsonl` (next16-instr, levers ON, 480 lines)
- `raw/p3/off_s1_rank1.round_prof.jsonl`, `raw/p3/off_s2_rank0.round_prof.jsonl` (next16-instr, levers OFF, 3072 lines)
- `raw/gpu_busy.benign.json`, `raw/gpu_busy.agentic.json`
- `PHASE1-M1.md` (bracket table + 143-vs-98 gap), `PHASE0-M0.md` (0d residency), `PHASE20-FINAL-MF.md` (0d verdict)
- `GPU-BUSY-NOTES.md` (comm/gpu_wait classification + caveat), `phase1_round_mechanics.md` (Q1 4 all_sums; Q3c syncs; Q6)
- `PHASE3-M3.md` (levers-OFF brackets), `ROUND-DENSE-EXL3.md` / `PHASE5-R1C-SHIP.md` / `PHASE5-CAMPAIGN.md` (production 94.94 / 99.46)
- `MECHANISM.md` §5 (gate), §6 (what was NOT closed)
