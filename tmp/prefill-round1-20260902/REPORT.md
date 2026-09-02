# PREFILL ROUND 1 — PM report (feedback-loop round 1)

**Date:** 2026-09-02 · **Repo:** exo @ `cb1f91903` (main) · **PM:** delegation subagent
**Mode:** Phase 0 + Phase 1 complete (zero-cost analysis + pre-registration). **Phase 2 STOP honored — no cluster benchmark was run.**
**Inputs:** fable's ranked suggestions (2026-09-02 consult); prefix-cache dive (`tmp/prefix-cache-dive-20260902/REPORT.md`); three read-only audits under `findings/`.

---

## HEADLINE

**Four of fable's six items rest on premises that are false at the source level. The audit killed them for free.**
**The highest-value remaining work needs ZERO cluster time — its first validation step is an offline token diff.**

The approval ask that comes out of this round is therefore much smaller than the ~6 cluster-hours fable's list implied. It is **~0 GPU-hours of mandatory work**, plus one optional ~30-min confirmation arm.

| # | fable's item | Verdict | Why |
|---|---|---|---|
| 1 | Prefix-cache: 0 full hits | **CLOSED on hit rate → REFRAMED** | Cache is at 97.60% session-wide / 99.9986% turn-to-turn. ~0.001% of hit rate left. But the dive found a **~22% prefill-avoidance win via KV lifetime** — promoted to #1. |
| 2 | Re-decompose "390ms/chunk" | **PREMISE UNSOURCED → REFRAMED** | The 390ms number is a **prose code comment** (commit `c0d012d2`, 2026-06-21), never a measurement, and is contradicted by a real instrumented run (<0.02% of wall). |
| 3 | CLEAR_CACHE_INTERVAL sweep | **KILL** | The knob is **dead code on the production path**. Also already refuted live at deep context (+0.48%). |
| 4 | Chunk-size re-sweep | **KILL** (except 1 optional arm) | 8% regression measured at deep context; upside bounded by a <0.02%-of-wall bracket. |
| 5 | RDMA all_sum fusion | **KILL** | Its gate was "item 2 shows collectives dominate fixed overhead." Item 2's premise is unsourced and the one real measurement contradicts it. Gate cannot be met. |
| 6 | Indexer window bound | **KILL** | Indexer is **4.0% of prefill wall**, already scored CLOSED (2026-08-24), and its env knob is dead plumbing — this is a code project for a ≤4% ceiling. |

---

## 1. PREFIX-CACHE VERDICT (item 1) — folded in

The dive (`tmp/prefix-cache-dive-20260902/REPORT.md`) refuted all three of its own pre-registered hypotheses.

**What is closed:**
- Cache key is **exact token ids** — no hash, timestamp, session id, or template tag anywhere in the key path (`cache.py:2324`, `_longest_prefix_match`).
- Matching is **token-granular**, not chunk-quantized (`cache.py:1035`).
- Turn-to-turn reuse: **min 99.9978% / median 99.9986%**, and the shortfall is a *constant* 2 tokens across growths spanning 77→11,535.
- Session-wide: **97.60% of 7,883,591 prompt tokens served from cache.**
- **"0 full hits" is correct-by-design.** `is_exact = match_length >= max_length - 1` (`cache.py:1284`) requires the incoming prompt to be ≤1 token longer than what's cached. Measured per-turn growth was min 77 / median 1,324. A full hit was arithmetically impossible on all 54 turns.

**On fable's framing claim.** "A 20% cache-hit improvement beats a 20% prefill_tps improvement" is **arithmetically correct but inapplicable** — there is no 20% of hit rate left to win; remaining headroom on that axis is ~0.001%. This is worth stating plainly back to fable, because the ranking that put item 1 first was built on it.

**But the dive opened something better.** It decomposed the 189,010 tokens that *were* actually prefilled:

| Bucket | Tokens | % of prefill | Mechanism | Fixable |
|---|---:|---:|---|---|
| A. Cold start after runner relaunch | 92,594 | **49.0%** | Trie is in-memory only; dies with the process (`builder.py:156`). No persistence path exists. | Yes |
| B. Model's own completions re-prefilled | 41,414 | **21.9%** | Leaf stores prompt tokens only; decode-produced KV is never inserted. | Yes |
| C. Genuinely new tool output + framing | 55,002 | 29.1% | Irreducible by caching | No |

A+B ≈ **67% of prefill wall time** in that session. **This is the same class of win the loop is hunting — ~22% (B) and ~49% (A) of prefill avoided — but won by retaining KV, not by matching prefixes better.** Fable's instinct that item 1 was the top of the list turns out to be right for a reason fable did not have.

**Do NOT touch the cache key.** It is working at 99.9986% efficiency; any hashing or normalization can only make it worse.

### Implementation workstream (replaces "improve hit rate")

**W1 — Fix B: retain decode-produced KV in the trie.** Target: 41,414 tok (21.9%, ~181 s in the studied session), spread evenly across all turns, fully in-process, no cross-node serialization. **Recommended first build.**

**W2 — Fix A: survive a runner relaunch.** Target: 92,594 tok (49.0%, ~222 s). Larger bucket but it came from a *single* relaunch event, so amortized value depends entirely on relaunch frequency — which one session cannot establish. **Measure relaunch frequency first (cheap); it sets the entire value of this item.**

Both are pre-registered in §5.

---

## 2. ITEM 2 — the 390ms/chunk number does not exist

This is the most consequential finding of the round, because items 2 and 5 both rest on it and item 3's rationale borrows from it.

**Provenance (`findings/item2-instrumentation.md`):** "~390ms" appears in exactly two places, both **prose comments**, both authored in the same commit `c0d012d2` (2026-06-21): `mlx-lm/mlx_lm/generate.py:438-440` and `start_cluster.sh:89-91`. A repo-wide grep finds **zero** measurements of it in `docs/`, `bench/`, or `tmp/`.

**It is contradicted by real instrumentation.** `docs/prefill-trace-instrumentation-findings-2026-08-21.md` measured that exact bracket — `prefill.clear_cache` + `prefill.barrier` + `prefill.mem_checkpoint` — at **single-digit ms total (worst case ~35 ms) against 190–1057 s prefill walls: under 0.02% of wall**, at 100K–500K context. That is one to two orders of magnitude below the claimed 390 ms.

So fable's premise (a) — "prefill exhausted was kernel-level only and never accounted for fixed per-chunk overhead" — is inverted: the fixed per-chunk overhead **was** measured, at the operating regime, and it is negligible.

**What existing logs can and cannot give.**
- The 2026-09-01 depth-scan logs contain **zero prefill span data**. They are `[MTP-PROF]` dumps — a decode/speculative-decode cycle profiler (`dsv4_mtp.py:790`). Grep for any prefill token across all 6 files returns nothing. No estimate should be manufactured from them.
- Per-chunk **TOTAL** wall IS derivable from `bench_*.json` (TP uses the full 2048 step; the `//min(4,size)` halving is PP-only, `generate.py:497` vs `:1454`):

| depth | real prompt tok | prefill wall | tok/s | chunks @2048 | per-chunk TOTAL |
|---|---:|---:|---:|---:|---:|
| 89.4K | 89,408 | 210.2 s | 425.4 | 44 | **4777 ms** |
| 150.0K | 150,013 | 358.4 s | 418.6 | 74 | **4843 ms** |
| 250.0K | 250,019 | 615.1 s | 406.5 | 123 | **5000 ms** |

- 3-point regression of per-chunk wall vs mean depth (n=3, R²=0.995): **intercept ≈ 4637 ms/chunk, slope ≈ 0.00288 ms/token**.

**Read that intercept correctly — this is a trap.** It is *not* "fixed overhead." It is overwhelmingly **L-proportional real compute** (the MoE FFN, attention, etc. for 2048 tokens), which does not amortize away when you enlarge chunks. The genuinely *amortizable* per-chunk bracket is the launch/sync/clear_cache cost, and that is the one measured at <0.02% of wall. 390 ms would be 8.4% of the intercept, but the mechanism it names was measured at ~0.02%.

**Consequence:** there is almost no amortizable fixed overhead to recover. This single fact is what kills items 3, 4, and 5 together.

**Reframed item 2:** do not fund a re-decomposition to chase a number that has no source. If a future workstream genuinely needs per-chunk attribution, the measurement is cheap and specified in §5.4 — but **nothing downstream currently needs it**, so it is not in the approval ask.

---

## 3. PRE-REGISTERED EXPERIMENT DESIGNS (items 3, 4, 6)

Designs are written out in full so the decision is auditable and so they are ready if the user overrides the recommendation. **All three carry a NOT-FUND recommendation with an explicit condition that would make them fundable.**

### 3.1 CLEAR_CACHE_INTERVAL sweep (item 3) — **NOT-FUND: the knob is dead code**

**Design as fable specified it:** sweep `EXO_PREFILL_CLEAR_CACHE_INTERVAL` ∈ {1,2,4,8} at 150K. Pre-registered band: 5–15% wall reduction at interval 4–8.

**Kill reason 1 — the knob does not exist on the production path.** I verified this by direct file read rather than taking the subagent's word:
- Production TP prefill is `prefill_batched` (`src/exo/worker/engines/mlx/generator/generate.py:1269`, called at `batch_generate.py:3068`).
- At `generate.py:1527` it calls `mx.clear_cache()` **unconditionally**. There is no interval gate in that loop.
- The **only** reader of `EXO_PREFILL_CLEAR_CACHE_INTERVAL` in the entire repo is `mlx-lm/mlx_lm/generate.py:544` — the **eager `stream_generate` fallback path**, not the batched production path.

Sweeping the variable would change nothing on the production path. The experiment cannot produce a signal.

**Kill reason 2 — already refuted live, at deep context.** `docs/dsv4-clear-cache-interval-2-test-2026-08-19.md` (commit `dd777f60a`): interval=2 vs 1 at **~180K tokens** measured **332.8 vs 331.2 tok/s = +0.48%**, i.e. noise. Verdict quoted: "The clear-cache-interval fix does not work."

**Kill reason 3 — the mechanism has no room.** Even if the knob were wired, the bracket it targets is <0.02% of wall (§2). A pre-registered 5–15% band is ~3 orders of magnitude above what the mechanism can physically deliver.

**Memory-headroom gate, for the record.** Fable's "76% headroom at 150K" is a paraphrase, not a citation. The real measurement (2026-08-09, Section 27): 2×concurrent-150K at peak resident **87.3 GB / 84.1 GB** against a 115 GB wired ceiling, `vm.swapusage` ~0. **Critically, it was measured WITH clear-every-chunk** — so it does *not* bound allocator-pool growth when 7 of 8 clears are skipped. Had this experiment been fundable, it would have needed its own fresh headroom measurement under interval=8 before being trusted.

**Would become fundable if:** someone first wires an interval gate into `prefill_batched` AND states a concrete mechanism by which a <0.02%-of-wall bracket could yield >1% e2e. Neither exists.

**Action instead:** file a defect — `EXO_PREFILL_CLEAR_CACHE_INTERVAL` is documented in `start_cluster.sh:107`, forwarded to the runner at `:1640`, and registered in `bench/trusted_measurement/fingerprint.py:153` as a live tunable, but is inert on the production path. It is a config foot-gun and a trusted-measurement fingerprint that records a value with no effect.

### 3.2 Chunk-size re-sweep (item 4) — **NOT-FUND, except one optional 3072 arm**

**Design as fable specified it:** 2048/3072/4096/6144 × 90K/150K/250K (~1.5 cluster-hr).

**Prior evidence.** 4096 vs 2048 measured **331.2 vs 358.6 tok/s at ~191K = ~8% end-to-end regression** (`docs/dsv4-prefill-step-size-4096-retest-2026-08-18.md`). Note this e2e number was **already taken at deep context** — fable's "the old sweep was not controlled for the deep-context regime" is factually wrong about the e2e measurement.

**On the mechanism — I am recording a caveat the repo does not.** The root-cause doc attributes the regression to SDPA cost scaling linearly with SEQ_SPLIT per-rank query rows, citing a sync-mode A/B: `attn.sdpa` 0.4153 → 0.8428 **ms/token** = 2.029x. I verified the units are ms/token (not ms/call), so the "half as many chunks cancels it" objection does not apply — at fixed total tokens, a 2x per-token cost is a real 2x. **However**, that A/B was run at **12,068 tokens**, and a 2x per-token jump from doubling L alone is steeper than a simple attend-to-prefix cost model predicts at that depth. The doc declares the thread "FULLY closed"; that confidence is **not fully earned**, and I am not repeating it as settled.

**The kill does not depend on resolving that**, which is why I am comfortable killing anyway. Two independent reasons:
1. **The e2e regression was measured at deep context directly** (191K). Whatever the mechanism, the outcome at the operating regime is known.
2. **The upside is bounded to near-zero regardless of mechanism.** The only thing a larger chunk *buys* is amortization of per-chunk fixed overhead — measured at <0.02% of wall (§2). Meanwhile every chunk-size-dependent effect shrinks as a fraction of wall as P grows, because the depth-dependent per-token attention cost grows with P and is chunk-size-independent. So chunk size matters **less** at 250K, not more. For 4096 to flip to a win at 250K you would need a per-chunk fixed cost that *grows with P* — the only candidate is `mx.clear_cache` scaling with resident memory, which the 2026-08-21 measurement at 100K–500K already bounds at <0.02%.

**Never-tested arms.** 3072 and 6144 have genuinely never been run (grep-confirmed). Under the linear-in-L model: 3072 → 1536 rows/rank ≈ 1.5x SDPA (near-breakeven at best); 6144 → 3072 rows/rank ≈ 3x SDPA (guaranteed worse than the already-regressing 4096).

**Optional arm, if the user wants one confirmation:** a **single 3072 arm at ~191K only** — one relaunch, ~30 min. Pre-registered bands, fixed now:
- **>+2% vs 2048** → genuinely new information; re-open the chunk-size question.
- **−2% to +2%** → confirms the model; close permanently.
- **<−2%** → confirms the model; close permanently.
Do **not** run 6144 (mechanism guarantees a regression) and do **not** run the 3-depth matrix (depth adds no information per the argument above).

### 3.3 Indexer sliding-window bound (item 6) — **NOT-FUND**

**Design as fable specified it:** bound the indexer's lookback window + correctness ablation; claimed to flatten the 426→406 t/s curve (~1 cluster-hr).

**Premise (a) is TRUE.** The lookback genuinely is unbounded: `_indexer_score` scores every query row against the full pooled prefix — `q_weighted @ pooled.swapaxes(-1,-2)` → `(B,L,P)` at `deepseek_v4.py:3849`. No window, no cap. `EXO_DSV4_INDEXER_PBLOCK` tiles P for *allocation* only and reassembles the full `(B,L,P)`.

**Premise (b) is FALSE — it is not the only O(P) term.** Compressed-layer SDPA (20 of 43 layers) concatenates the full pooled KV onto local KV (`deepseek_v4.py:4452`) and runs SDPA over all of it — also O(P) per token, and not bounded by `index_topk` at all.

**Premise (c) — the cost ceiling kills it.** Measured span profile at 220K (`docs/dsv4-220k-prefill-span-profile-2026-08-18.md`): `attn.indexer` = **4.0% of prefill wall**. For comparison: MoE 26.9%, `attn.sdpa` 13.6%, `attn.sdpa.compressed` 11.8%, collectives ~18%. A *perfectly free* indexer buys ≤4–4.7% e2e. And `docs/indexer-prefill-decomposition-2026-08-24.md` already scored this **CLOSED** — "no un-tried candidate with predicted e2e ≥1% exists."

**Premise (d) — the knob is dead plumbing, so this is a code project not a config A/B.** `EXO_DSV4_INDEXER_WINDOW` appears in `start_cluster.sh:26`, `bench/trusted_measurement/fingerprint.py:220-223`, and a test fixture — with **zero readers in any model or generator code**. I verified this myself. Bounding the window means writing a reader into `_indexer_score`, which introduces a real long-range-recall correctness risk for a ≤4% ceiling.

**Correctness-harness inventory (needed regardless, so recording it).** Best available: `bench/ab_probe_tier1.py` — builds a fresh ~N-token prompt with a uuid cache-buster, seeds needle `FALCON-MERCURY-7749` at **40% depth**, asserts `needle_hit` and `bos_spam`, reports prefill/decode t/s. Depth-agnostic via `target_tokens`. **Two real limitations for a window ablation:** it tests a *single* needle position, so a windowed indexer that still reaches 40% but loses early content would pass it; and it uses a ~4 chars/token heuristic rather than the real ~5.68 for prose. Any genuine long-range-recall ablation needs multi-position needles first. Supplementary: `bench/quality_probe_dsv4.py` (special-token-leak / bistability gate), `bench/context_stress.py` (memory-pressure + needle loop).

**Would become fundable if:** a span profile at 250K–500K shows `attn.indexer` has grown past ~15% of wall (it was 4.0% at 220K), *and* multi-position needle coverage exists to gate the correctness risk.

### 3.4 What a real item-2 measurement would look like (specified, NOT requested)

Recorded so it is ready if a future workstream needs it. Not in this round's approval ask.

Add a dedicated span around the in-loop `mx.clear_cache()` at `generate.py:1527` (none exists today), run with `EXO_TRACING_ENABLED=1` plus `EXO_PROFILER=spans EXO_PROFILER_SYNC_SPANS=1`. **Two mandatory gotchas:**
1. **Pair sync-spans with `EXO_RUNNER_HANG_TIMEOUT_SECONDS=300`.** Sync mode forces a GPU sync per span boundary, which can push the progress-callback gap past the 45 s watchdog and SIGKILL the runner mid-request (documented failure, `supervisor.py`).
2. **Instrument the batched production path, not the eager fallback.** The item-3 finding proves these paths differ materially; a measurement on the wrong one is worthless. Confirm the 2026-08-21 <0.02% figure was itself taken on the batched path before treating it as the production number.

---

## 4. ITEM 5 GATE MEMO (RDMA all_sum batching/fusion)

**Gate status: NOT MET. Do not fund.**

Fable correctly gated item 5 on item 2 first confirming that collectives are a large fraction of fixed per-chunk overhead. That evidence does not exist and the available evidence points the other way. The gating number (390 ms/chunk) is an unsourced code comment (§2), and the one real instrumented measurement of the launch/sync/clear bracket puts it under 0.02% of prefill wall at 100K–500K.

There is also a specific, expensive precedent in this repo for funding collective work on a derived number. A previously accepted figure of "178 ms/call all_sum" was later proven **mathematically impossible** — 4730 calls × 178 ms = 842 s against a real 612 s total wall for the same run — and corrected to **~5–12 ms/call** via two independent instrumentation sources. That correction invalidated the downstream "all_sum is 61–64% of prefill wall" conclusion that had motivated a whole line of work. Item 5's gating number has *weaker* provenance than the 178 ms number did: 178 ms was at least derived from measurements, whereas 390 ms was never measured at all.

**The precise evidence threshold that would fund item 5:** a **sync-mode span profile on the batched production path**, at ≥150K context, showing `moe.all_sum` + `attn.all_gather` + `prefill_batched.chunk*.barrier` summing to **≥20% of prefill wall**, with per-call cost cross-checked against total wall clock (calls × per-call ≤ measured wall — the reductio that caught the 178 ms error). Absent that, engineering effort on collective fusion is unfunded.

Note the current 220K span profile shows `moe.all_sum` 9.5% + `attn.all_gather` 8.5% ≈ 18% — close to, but under, that bar, and those are *per-layer collectives inside the forward*, not the cross-chunk-boundary overhead item 5 proposes to fuse. Fusing across chunk boundaries does not address per-layer collectives. **The item as scoped targets the wrong 18%.**

---

## 5. PRE-REGISTERED DESIGNS FOR THE PROMOTED WORKSTREAM

### 5.1 W1 — Fix B: retain decode-produced KV (RECOMMENDED FIRST)

**Target:** 41,414 tok = 21.9% of prefill work (~181 s in the studied session), spread across all turns.

**Mechanism.** The trie leaf stores `all_prompt_tokens` — prompt only. KV computed during decode is discarded, then recomputed as prompt next turn. Extend the leaf with the generated token ids plus their already-resident decode KV at end-of-turn.

**Step 1 is OFFLINE and costs zero cluster time.** For each of the 54 recorded turn pairs in `tmp/real-usage-capture-20260902/phase1/requests.jsonl`, take turn *n−1*'s stored tokens + generated ids, and diff against turn *n*'s prompt as tokenized **through the actual production chat-template path**.

**Metric — this must be LCP coverage, not mean token agreement.** The trie is a prefix matcher: a single divergence at position *k* forfeits everything after *k*. A pair could show 99% token agreement with the divergence at position 1 and yield zero benefit. **Pre-registered metric: longest-common-prefix coverage — the fraction of the completion's tokens covered by the LCP — reported as a distribution across all 54 pairs, not a mean.**

**Pre-registered outcome bands (fixed before measurement):**
- **median LCP coverage ≥80%** → build it. Expected ~18% cut in warm-turn prefill.
- **40–80%** → build only with a token-level agreement check at insert (insert the agreeing prefix, discard the tail).
- **<40%** → abandon and say so.

**Pre-registered risks — the first is a potential hard cap, not a footnote:**
1. **Thinking-marker stripping is a POSITION problem, not a token-id problem.** If `<think>`/`</think>` spans are stripped before the completion re-enters the next prompt (`_strip_v4_thinking_markers`, `utils_mlx.py:1489`), every subsequent token's RoPE position shifts. Decode KV was computed at the *original* positions and is unusable past the strip point **even if the surviving text retokenizes identically**. Reusable KV is capped at the prefix up to the first stripped span. If completions routinely open with thinking, the recoverable fraction of the 22% could approach zero. **The offline check must run the real production template path including stripping** — an idealized retokenization would pass a scenario production cannot reproduce.
2. **Boundary-token BPE merges.** The final completion token can merge with template-appended boundary tokens on retokenization. Distinguish *edge* divergence (costs a few tokens, acceptable) from *body* divergence (kills the tail).
3. **TP cross-rank consistency.** KV is sharded across 2 ranks; insertion must be atomic/consistent on both, or a partial insert yields garbage attention. Add a rank-divergence assertion at insert.
4. **Quantized-KV numerics.** Decode-time and prefill-time paths may use different quantization group boundaries. One-off logit-equivalence spot check.
5. **Trie growth / eviction.** Inserting decode KV every turn grows the trie; confirm eviction policy covers it before shipping.

**Correctness gate (automatic NO-GO on failure, regardless of speedup):** needle probe at 100K returning `needle_hit=true`, `bos_spam=false`, and **bit-identical output vs the cache-off path** on a fixed prompt. Precedent: a wrong-position restore previously produced `' his his his'` degeneration (`cache.py:~1380`, `strict_snapshot` history).

### 5.2 W2 — Fix A: survive a runner relaunch (SECOND)

**Target:** 92,594 tok = 49.0% of prefill work (~222 s) — the largest single bucket.

**Step 1 is cheap and gates everything: measure relaunch frequency.** The 49% came from one relaunch in one session; amortized value is entirely a function of how often runners actually cycle. This is **read-only log analysis** (allowed under the task's constraint 4) — but note the logs live on the cluster nodes; there is no `~/exo.log` on this host, so it needs read-only SSH. **Zero GPU time.**

**Step 2 (only if frequency justifies it):** measure bytes-on-disk for one 146K-token leaf and real write/read wall time.

**Pre-registered bands:**
- Restore wall **<25%** of equivalent re-prefill (~222 s → <55 s) **and** relaunches more than once per ~10 sessions → build.
- Restore **25–60%** → build only if relaunch frequency is high; else shelve.
- Restore **>60%**, or serialization measurably degrades steady-state turns → **abandon persistence and pivot to making the runner stop relaunching** — a zero-GPU-cost fix to half the problem, and likely the better lever if relaunches are crashes or routine deploys rather than intrinsic.

**Gate:** a restored trie must pass the permanently-on `[PREFIX_CACHE_INTEGRITY_VIOLATION]` check on first hit, plus a needle probe. A restored-but-subtly-wrong KV cache is far worse than a cold prefill.

### 5.3 W3 — Latency hiding (new, not on fable's list)

Bucket C (29.1%, genuinely new tool output) cannot be *avoided* — but TTFT is **perceived** latency, so it can be **hidden**. Eagerly prefill appended tool output the moment the tool returns, before the next request is issued. At 96.6–97.0% GPU saturation there is no idle capacity to exploit *during* a request, but there is real idle time *between* a tool returning and the user's next turn. Recorded as a candidate for round 2; needs a design pass before it can be pre-registered.

---

## 6. THE APPROVAL REQUEST (single bundle)

**Mandatory cluster GPU time requested: ZERO.**

| # | Item | Cluster cost | Needs approval? | Blocking |
|---|---|---|---|---|
| A1 | W1 step 1 — offline LCP-coverage probe on 54 recorded turn pairs | **0** (local CPU) | **No** — read-only, existing artifacts | Nothing |
| A2 | W2 step 1 — relaunch frequency from node log archive | **0 GPU** | **Yes, narrowly** — needs read-only SSH to cluster nodes | Awaiting green light |
| A3 | File 2 dead-knob defects (`EXO_PREFILL_CLEAR_CACHE_INTERVAL`, `EXO_DSV4_INDEXER_WINDOW`) | 0 | **Yes** — repo write | Awaiting green light |
| A4 | *Optional* single 3072 arm @191K | ~30 min, 1 relaunch | **Yes** | Only if you want the confirmation |

**On relaunch cost, since it shapes any future ask:** all three sweep env vars (`EXO_PREFILL_CLEAR_CACHE_INTERVAL`, `EXO_PREFILL_STEP_SIZE`, `EXO_DSV4_SEQ_SPLIT`) are read from the process-global env fixed at worker launch. `EXO_DSV4_SEQ_SPLIT` is frozen at module import (`auto_parallel.py:124`); the other two are read per-call but from the same fixed `os.environ`. The runner subprocess inherits the worker env (`supervisor.py:294`) with no external injection path, so **spawning a fresh runner does not re-read anything** — every arm costs one full top-level relaunch, ~20 min bring-up + ~9 min for a 191K prefill ≈ **~30 min per arm**. Fable's ~2.5 cluster-hour estimate for the two sweeps was accurate; the sweeps just should not be run.

**Recommended execution order:**
1. **A1 now** (no approval needed, zero cost) — it either unlocks a ~22% prefill-avoidance win or kills it for free. Highest information per unit cost in the entire round.
2. **A3** — cheap hygiene; both knobs currently lie to `start_cluster.sh` and to the trusted-measurement fingerprint.
3. **A2** — sets the entire value of the 49% bucket.
4. **A4 only if wanted** — I do not recommend it; the mechanism is understood well enough and the upside is bounded by a <0.02%-of-wall bracket.

---

## 7. FOR THE NEXT CONSULT ROUND (feedback to fable)

Worth putting to fable directly, since it shapes the next ranking:

1. **The 390ms/chunk premise is unsourced** — a 2026-06-21 code comment, contradicted by a 2026-08-21 instrumented measurement (<0.02% of wall). Items 2, 3, and 5 all inherit from it. This is the single correction that most changes the ranking.
2. **The "20% hit rate beats 20% prefill_tps" arithmetic is right but the cache is already at 97.60%/99.9986%.** The correct target is the same *shape* of win — avoided prefill — but via KV lifetime.
3. **Item 3's knob is dead code on the production path** (`generate.py:1527` clears unconditionally; only the eager fallback reads the interval). Worth knowing that this fork's env surface contains inert knobs — future proposals should verify a reader exists.
4. **Two of fable's items were already-closed investigations** (clear-cache interval refuted 2026-08-19; indexer closed 2026-08-24). If fable had access to `docs/` it would likely not have proposed them — consider passing a doc index into the next consult.
5. **Open question fable could help with:** the 4096 SDPA attribution measured 2.029x ms/token at only 12K context and the repo declares it "fully closed." That confidence looks unearned. It does not change this round's decision, but it is a live crack in a doc that future work will cite.
6. **The genuinely open question for round 2:** given GPUs measured 96.6–97.0% saturated during prefill, is there any lever *other* than work-avoidance (KV lifetime) and latency-hiding? If not, the prefill-tuning line of work is effectively complete and the campaign should move to KV lifetime wholesale.

---

## Provenance

- **Prefix-cache verdict:** `tmp/prefix-cache-dive-20260902/REPORT.md` (deleg_14b6b593).
- **Audits:** `findings/item2-instrumentation.md`, `findings/item34-provenance-and-envvars.md`, `findings/item6-indexer-and-gates.md`.
- **PM-verified independently** (not taken on a subagent's word): the unconditional `mx.clear_cache()` at `generate.py:1527` and its call chain from `batch_generate.py:3068`; the sole `EXO_PREFILL_CLEAR_CACHE_INTERVAL` reader at `mlx-lm/mlx_lm/generate.py:544`; `EXO_DSV4_INDEXER_WINDOW` having zero code readers; the ms/token units in the 4096 root-cause doc's sync-mode table.
- **External review:** the kill decisions and the W1 gate design were pressure-tested by an independent reference model, which corrected the W1 metric from token-agreement to LCP coverage, elevated the thinking-marker position-shift risk from footnote to primary, flagged the "intercept ≠ fixed overhead" framing error, and prompted the addition of §5.3.
- **Constraints honored:** no cluster benchmark run, no code changes, no commits. Only files created are this report and the three findings files under `tmp/prefill-round1-20260902/`.
