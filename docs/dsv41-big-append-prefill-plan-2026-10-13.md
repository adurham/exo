# DSv4.1/exo — big-append prefill felt-latency workstream — EXECUTABLE PLAN (consult, saved 2026-10-13)

**Provenance:** requested by the owner during the route-cliff deploy ("while that is running consult with
fable on what's needed for the big tool-result prefills you called out"). Produced by the reference-model
consult (Fable) on 2026-10-13; saved verbatim below with this header. **NOT executed.** A fresh session
executes it. Stage 0 is offline (free); Stage 1 is one live round and needs its own go.

**Base state at save time:** exo `deploy/next23b-qsplit @ 60008939` + mlx-lm `6a5c78f4` serving;
`deploy/next24-routing @ 9fe1d593` (the small-n wide-tile fix) was IN GATES at save time — **check its
outcome first** (idxprof8) before any Stage-1 planning; flags `EXO_DSV41_CKPT=1` + `DSV41_INDEXER_QSPLIT=1`;
chunk 2048; KV cap 1,048,576; DSpark γ3.

**House-convention mapping:** deliverable docs at `docs/dsv41-<name>-<date>.md` on `main`; branches per the
plan's own naming; every number labelled MEASURED/DERIVED/BOUND/MODELED; Stage-1 live round gets its own
go and a rollback-ready header.

---

# Plan: DSv4.1/exo felt prefill latency for large appends at deep context

**Plan date:** 2026-10-13. **Base build:** deploy/next23b-qsplit @ 60008939 + mlx-lm 6a5c78f4, plus tonight's decode-cliff fix once its gates pass.

**Verdict up front:** This workstream should start as a measurement, not an optimisation. Three facts point that way:

* The per-chunk prefill cost is dominated by depth-independent work.
* The depth-independent work sits mostly in already-closed areas: MoE/ffn, EXL3 dequant, compile.
* Every unclosed Python-level lever is bounded in the low single digits.

The most probable outcomes are:

* (a) A request-path or prefix-cache defect that is cheap to find and fix.
* (b) An evidence-backed closure, with the real levers handed to the owner. Those levers are Hermes tool-result sizing, routing, and a 3rd node.

Stage 0 is entirely offline. Stage 1 is one live round. Do not write optimisation code before Stage 1 reads.

---

## Pre-flight consistency checks (do these first, 15 min, offline)

**P1. Is the depth curve consistent with the indexer share?**
* DERIVED per-chunk time:
  * at ≤105K: 2048 / 309 = 6.63 s
  * at 500K: 2048 / 234 = 8.75 s
* The depth-dependent increment over 105K→500K is therefore about 2.1 s.
* DERIVED indexer time at 105K: 22% × 6.63 ≈ 1.46 s.
* If the indexer scaled linearly with depth, it alone would be about 6.9 s at 500K. The chunk would then take more than 12 s, which is under 170 rows/s. That contradicts the measured 234 rows/s.
* So one of two things is true:
  * the 234 rows/s figure predates qsplit/HIER, or
  * HIER makes the indexer sublinear in depth.
* Find the provenance of the 234 rows/s figure: commit, date, and flags.
* Either way, the depth curve is re-measured in Stage 1. **Do not use either rate in any projection until it is re-measured.**

**P2. How was "new rows" in state.db computed?**
* If it came from ollama-cloud's `prompt_eval_count` or another server-side figure, it reflects the cloud's cache behaviour.
* In that case, cloud cache misses inflate the ≥20K bucket.
* Recompute "new rows" as the token-level longest common prefix against the same session's previous call, using the DSv4.1 tokenizer and chat template. Count everything after the common prefix as new.
* Report how much the ≥20K bucket changes. If it shrinks by ≥25%, **surface immediately**, because the target itself moves.

---

## Stage 0: offline census (timebox 3 h, no cluster contact)

All inputs come from Hermes state.db and existing logs. Produce one CSV and a one-page summary.

### 0.1 Classify the ≥20K-row calls (≈270 calls)

Sort each call into one of three classes:

| Class | Definition |
|---|---|
| (i) True append | Common prefix = whole previous prompt + previous assistant turn |
| (ii) Divergent prefix | Divergence before the end of the previous prompt. Examples: timestamp in the system prompt, edited or rewritten history, reasoning stripped from prior assistant turns by the template |
| (iii) Compaction-rebuilt context | Effectively a cold prefill |

For each class, record count, rows, and depth distribution.

**Pre-registered reading:**
* If classes (ii) + (iii) hold ≥30% of the ≥20K rows, the dominant lever is prefix stability or cache fidelity, not kernel speed. Promote L1 and L2 (see the ranking table).
* If class (iii) dominates, the problem is cold prefill at shallow-to-mid depth, and the depth-related levers are irrelevant.

### 0.2 Template reasoning-strip check

Render two consecutive turns through the exact DSv4.1 chat template that exo uses, with a real generated reasoning block included.

* Locate the token index where the next request's prompt diverges from what the cluster's KV holds after its own decode.
* If the template drops `reasoning_content` from prior turns, every turn re-prefills from the last assistant message onward.
* That is quality-neutral to fix only if the fix keeps the exact rendered prompt and re-prefills just the true suffix. It is **not** acceptable to keep stale reasoning KV, because that is output-affecting.
* Quantify the re-prefilled rows per turn: median and p99.

### 0.3 Concurrency census

Measure the fraction of calls whose interval overlaps another in-flight call, including subagents and parallel tools.

* **Kill L6 (prefill blocking decode) if fewer than 5% of call-seconds overlap.**

### 0.4 Felt-cost share

Model each session's wall-clock using cluster rates: prefill at 309/234 (flagged per P1), decode at the measured deep decode rate after the cliff fix, and tool time from state.db timestamps.

* Compute S = Σ(prefill of ≥20K appends) / Σ(session wall time). Label it MODELED.
* **If S < 8%**, the felt-latency case is weak at any achievable speed-up. Surface this before spending live time.

### 0.5 Owner-side lever sizing (data only, no recommendation)

Recompute the share of prefill rows removed if Hermes capped tool results at 8K, 16K, or 32K tokens, or stored them by reference.

* This is an agent-behaviour change, not a model-identity change. It is the owner's call.
* Its MODELED gain is likely larger than every engine lever combined.

### 0.6 Routing context

From state.db, note what ollama-cloud's TTFT was on the same ≥20K calls, if it was recorded. This sets the expectation gap.

**Stage 0 deliverable:** `census-largeappend.md` plus the CSV. Sign and push.

---

## Stage 1: one live measurement round (needs its own go; timebox 2 h; read-only, no code change)

**Preconditions:**
* Tonight's cliff-fix gates have passed and the build has been stable for ≥12 h.
* The build and flags are recorded exactly.
* The cluster is idle.

**Mechanism:**
* Launcher-only relaunch with the existing profiler and timing harness, i.e. the Oct 6 attribution tooling.
* No new env vars beyond the fixed forward list.
* If timing hooks need a new env var, that becomes a separate plan.

### 1A. Depth curve, one long prefill (~35 min)

* Run one request of 480K synthetic-but-realistic tokens, using concatenated real tool-result text.
* Log per-chunk wall time at eval boundaries.
* Each chunk sits at a known depth, so a single run yields chunk time versus depth.
* Fit `t_chunk = a + b·depth`. Report a, b, residuals, and whether the indexer contribution is sublinear. This resolves P1.

### 1B. Component split at 117K and 300K

* Profile 2 chunks at each depth. Report each component's share and absolute ms:
  * sdpa
  * indexer
  * ffn/MoE (split dequant vs GEMM if the profiler allows)
  * attention projections
  * all_sum
  * MTP-head prefill
  * KV write/growth
  * residual "unattributed"

### 1C. Request-path decomposition

After 1A, send 3 append requests of 20K rows each at the 480K session's checkpoints, or in a fresh session at about 117K.

Timestamp each request through these phases:

| Phase | Covers |
|---|---|
| HTTP arrival | |
| Template render + tokenize | ~180K-token prompt |
| Prefix-cache match | |
| First chunk start | |
| Last chunk end | |
| First token emitted | |

Also record the cache-hit length versus the expected common prefix, to the token.

### 1D. Cache-fidelity replay

* Replay 3 real short Hermes sessions (5–20K depth; fidelity does not depend on depth) through the cluster with real generation.
* Construct each next prompt exactly as Hermes would, from the cluster's own output.
* Record the cache hit versus the ideal at every turn.

### 1E. Memory

Record peak wired memory per rank during the 480K prefill. This is the input for the M3 reopen question.

**Rollback:** none needed, since the round is read-only. Relaunch to the stock flags afterwards and verify a 5-prompt identity smoke.

### Pre-registered interpretation (write the result next to each row)

| Observation | Reading | Action |
|---|---|---|
| Cache hit < ideal by ≥1K rows on any turn (1C/1D) | Fidelity defect | L1 becomes top priority; surface same day |
| Template + tokenize + match ≥ 2% of TTFT for a 20K append at 117K (≥1.3 s) | Host request-path overhead | Run L2 |
| `a` ≥ 75% of `t_chunk` at 117K | Depth-independent work dominates | Depth-specific levers are capped; L5 stays closed |
| all_sum ≥ 10% of chunk at 117K | Comm is bigger than assumed | L3 (overlap) is worth a feasibility probe |
| all_sum < 6% | L3 ceiling below the 1% EV bar after the success-probability discount | Close L3 with this number |
| Unattributed + boundary ≥ 5% **and** peak wired ≤ 120 GB at 480K | M3 fence-removal gate is reachable | Reopen M3 as its own plan |
| Indexer ≥ 25% at 117K or ≥ 35% at 300K | Indexer grows with depth | Write the L5 plan (metal_kernel, no core rebuild) |
| MoE/ffn + dequant ≥ 50% | Bulk of time is in closed, memory-blocked work | Record for the 3rd-node decision; nothing to do now |
| Sum of open-lever ceilings (L2+L3+L4+L5 BOUNDs) < 12% of TTFT and L1 clean | Engine-level work cannot move felt latency perceptibly (65 s → ≥57 s) | **Close the workstream with evidence**; hand owner-side levers over |

---

## Lever ranking

Ranked by expected end-to-end gain × feasibility. Every number below is a BOUND or MODELED estimate pending Stage 1.

| # | Lever | Ceiling | Feasibility | Identity risk | Kill criterion |
|---|---|---|---|---|---|
| L1 | Prefix-cache fidelity / template-strip re-prefill / prefix stability | Unknown; 10–100% on affected calls if a defect exists | High (Python) | None if exact-prefix reuse only | 0.1 class (ii) <5% of rows **and** 1C/1D hit = ideal on every turn |
| L2 | Request-path host work (template render, tokenize, Python prefix match on 180K tokens) | BOUND = measured share (1C) | High (Python) | None | Share <2% of TTFT |
| L3 | Sub-chunk comm/compute overlap (micro-batch A/B pipelining per layer) | BOUND ≤ post-IDLE all_sum share. DERIVED expectation 3–4.5% (6–9% at 2 GB/s, halved at 4 GB/s). The W15 10–16% figure is inconsistent with this unless 1B shows all_sum ≥10% | Low–medium. Needs the prebuilt MLX to run the JACCL all_sum truly concurrently with GPU compute on a separate stream | Medium: a smaller M can change GEMM kernel selection, so it must pass the identity class | all_sum <6% at 117K, **or** concurrency probe shows serialisation |
| L4 | Eval-boundary / schedule overhead (M3 fence-removal reopen) | BOUND ≤ 3.2–7.4% (GPU-idle fraction at 110 ms sampling) plus sub-sample launch gaps | Gated on memory | Low | Boundary <4%, or peak wired >120 GB at 480K |
| L5 | Fused indexer score kernel via `mx.fast.metal_kernel`, JIT at runtime with no core rebuild. **Verify this first, because it changes the "core can't be built" assumption for this lever only** | 22% × (1 − 1/speedup) ≈ ≤15% at 117K if 3×; grows with depth | Medium–low; documented modest odds | Medium; fp32 reduction order | Stays closed unless the 1B thresholds above are hit |
| L6 | Prefill/decode interleave in continuous batching | ≈0 for single-stream use | Medium | Low | 0.3 overlap <5% |
| L7 | all_sum host copy/zero work (4 → 10 GB/s) | ≤ ~2–3% DERIVED (all_sum share × 0.6) | Low (JACCL is prebuilt; env knobs only) | None | Same bound as L3; fold into L3 |
| — | Overlap the append's prefill with the previous decode, or stream append chunks | **0 by causality.** The tool result does not exist until the tool returns, and the client sends whole prompts | — | — | Killed on logic; no experiment. Reopen only if Hermes streams tool output incrementally into the request |
| — | "~80 KB decode-cliff chunk-size" | Decode-side, not prefill | — | — | Out of scope unless 1B shows it alters prefill chunk scheduling |
| O1 | **Owner-side:** Hermes tool-result caps / by-reference results | MODELED from 0.5; likely the largest single lever | Trivial technically | Changes agent behaviour, so it is the owner's call | — |
| O2 | **Owner-side:** route ≥20K-append calls elsewhere | — | — | Privacy/uncensored trade; owner's call | — |

### Closed-list status changes this plan could trigger

* **M3 fence-removal.** Reopen evidence = 1B boundary overhead ≥4% **and** 1E peak wired ≤120 GB during a 480K prefill with qsplit + cliff fix.
* **Fused indexer kernel.** Reopen evidence = 1B indexer share thresholds **plus** confirmation that `mx.fast.metal_kernel` works in the installed binary.
* **Native-experts requant / dequant caching.** No change. 1B records the dequant share so the 3rd-node business case has a number.
* **Everything else on the closed list** stays closed. Nothing in this plan produces new evidence for it.

---

## Stage 2: staged experiments (only those Stage 1 triggers; each with its own live plan and go)

**2-L1: fidelity fix**
* Timebox: 1 day offline + 1 live round.
* Offline: a unit test that renders turn N and N+1 and asserts that the cache-reuse length equals the token-exact common prefix.
* Implement the fix in exo or mlx-lm Python.
* Gates: identity class (5-prompt greedy + battery), then a 1D replay showing hit = ideal.
* Rollback: revert the commit and relaunch.

**2-L2: request-path fix**
* Timebox: 0.5 day offline.
* Profile tokenize, render, and match on a saved 180K prompt on any machine.
* Typical fixes: incremental tokenization of the suffix only, and array-based prefix matching instead of Python lists.
* Output-identical by construction. Gate on identity anyway.

**2-L3: feasibility probe only**
* Timebox: 1 day.
* Offline-first: a minimal 2-rank script that issues an `all_sum` on one stream and a large GEMM on another. Check for wall-time overlap.
* This needs both nodes, so it needs a go. Run it in the same window as another live round if possible.
* Proceed to implementation only if overlap ≥70% of all_sum time is observed **and** the 1B all_sum share is ≥6%. Then the micro-batch restructure becomes its own plan with identity gating.

**2-L4 and 2-L5**
* If triggered, write separate plans. Do not fold them into this workstream's timebox.

**Total workstream timebox:** Stage 0 + Stage 1 within 1 week. If nothing is triggered, closure is written by the end of week 1.

---

## Surface to owner

**S0, now (with Stage 0 kickoff):**
* The cluster does not currently serve Hermes.
* Ask: is moving Hermes to the cluster intended? If not, this workstream optimises a hypothetical, and its priority should drop.
* No blocking needed; Stage 0 is free.

**S1, end of Stage 0 (≤1 day):**
* Report the census summary: P2 rebaseline, class split, S share, and the O1 sizing table.
* Request the go for the Stage 1 live round.
* **Surface immediately** (not at stage end) if any of these hold:
  * P2 shrinks the target by ≥25%
  * 0.2 finds per-turn reasoning-strip re-prefill
  * S < 8%

**S2, after Stage 1:**
* Deliver a one-page decision memo: the filled-in interpretation table, which levers are triggered, and the closure-or-proceed recommendation.
* If closing, state plainly: "engine-level ceiling ≈ X% of TTFT (BOUND); the felt cost is moved only by O1/O2/3rd node". Attach the dequant/MoE share for the 3rd-node decision.

**Ad hoc, same day:**
* Any cache-fidelity defect. It affects benchmarks and any future real use.
* Any identity-gate failure in Stage 2.

---

## Executor notes

* Every number in the deliverables must be labelled MEASURED, DERIVED, BOUND, or MODELED. The two rates (309/234) stay flagged until 1A replaces them.
* Do not co-mingle with tonight's cliff-fix deploy. Stage 1 waits for its gates, and both the commit and the flags are recorded in the Stage 1 header.
* Live rounds use launcher-only relaunches with regenerated node helpers and the fixed env forward list. Restore the stock config and run a smoke after each round.
* Sign and push the docs (census, Stage 1 results, decision memo).
