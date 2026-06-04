# DeepSeek-V4-Flash + MTP on the 2-node M4 Max Cluster — Performance Writeup

**Date:** 2026-06-04
**Cluster:** 2× Mac Studio M4 Max (128 GB each), RDMA over Thunderbolt 5, exo
tensor-parallel (2 ranks).
**Model:** `mlx-community/DeepSeek-V4-Flash-8bit` (DSv4-Flash, MoE + sparse-pooled
attention, served via the fork's `deepseek_v4.py`).
**Champion (default via `start_cluster.sh`) — 2026-06-04 post-upstream-integration:**
exo `main`=`652d8224` (zenoh networking #2132 + mlx upstream sync), mlx
`main`=`419cf7efe`, mlx-lm `main`=`f8b277f`. Validated-state tag:
`validated-upstream-integration-20260604-153609`. (Supersedes the pre-integration
champion `c840bc2d` / mlx `db757dcb0`; throughput unchanged within noise.)
**Config:**
`EXO_SPECULATIVE=1 EXO_DSV4_MTP=1 EXO_SPECULATIVE_GAMMA=2 EXO_DSV4_MTP_EAGLE_K=8`
`EXO_KV_CACHE_BITS=0 EXO_DSV4_INDEX_TOPK=512 EXO_PREFILL_STEP_SIZE=128`
`EXO_DSV4_FENCE_EVERY_N_LAYERS=4 EXO_DSV4_MTP_TIEBREAK_FIX=1 EXO_DSV4_MTP_TIEBREAK_EPS=0.5`.

**Latest validated bench (this integrated champion):** c=1 100K γ=2, 10 scored
iters — mean **30.6 t/s**, median 30.7, min 29.9, max 30.8, 0/10 bad, errors=0.
Quality: Paris clean, 100K needle 3/3 (0 leak), aistupid hourly correctness 100%.

---

## 1. TL;DR

DeepSeek-V4-Flash runs on this 2-node cluster at **30.8 tok/s decode (c=1,
100K context)** with **MTP self-speculation on**, and it does so at **100%
correctness** on the ported AI-Stupid-Meter execution suite — matching the
MTP-off reference exactly. This is the result of two things landing together:

1. **MTP self-speculation** (the model's own Eagle/MTP head drafting γ=2 tokens
   per main-model forward), which buys ~2× the per-forward token yield.
2. **The tie-break losslessness fix** (`c840bc2d`), which closed the one quality
   crack MTP had — a temp-0 tie-flip that cascaded into spurious `</think>` and
   degenerate reasoning on hard prompts.

Before the fix these were mutually exclusive: you could have MTP's speed (~31
t/s) *or* full correctness (MTP-off, ~27 t/s), not both. Now you get both.

## 2. What "MTP" means here (and what it is not)

**MTP = Multi-Token Prediction self-speculation.** DSv4-Flash ships with an
Eagle-style MTP head. At decode the MTP head proposes γ draft tokens from the
current hidden state; the main DSv4 forward then verifies all γ+1 positions in a
single batched pass and accepts the longest correct prefix. There is **no
separate draft model** — the model speculates against itself.

This is distinct from **classic speculative decoding**, which uses a separate
smaller draft model (on this stack that path falls through to a Qwen draft and
produces gibberish — never run `EXO_SPECULATIVE=1` with `EXO_DSV4_MTP=0` on
DSv4). The two are not interchangeable knobs; MTP is the only speculation mode
used for DSv4 here.

## 3. Throughput

All numbers c=1 (single in-flight request — the project's serving target),
`bench/concurrent_bench.py`, 100K context, γ=2, temp 0.

### Headline (champion, MTP-on + tie-break fix, 10 scored iters)

| metric | value |
|--------|-------|
| mean agg tok/s | **30.77** |
| median | 30.8 |
| min / max | 30.6 / 30.8 |
| σ | 0.067 |
| errors | 0 / 10 |

Passes the project bar (γ=2: ≥10 iters, all ≥29 t/s, σ<0.5, errors=0). The
tie-break fix costs ~0 throughput — it is an argmax/where over logits already in
flight, no extra forward.

### MTP vs no-MTP (where the speedup comes from)

From the c=1 100K tuning sweep (`fork-notes.md`), stacking the three confirmed
wins on the control baseline:

```
                                          wall    decode (tok/s)   needle
control (no extras)                       735s    15.4             ✓
+ EXO_PREFILL_STEP_SIZE=128               457s    15.5             ✓
+ EXO_DSV4_FENCE_EVERY_N_LAYERS=16        446s    17.3             ✓
+ EXO_DSV4_MTP=1 + EXO_SPECULATIVE=1      444s    21.6             ✓
```

MTP alone is the **+25% decode** step (17.3 → 21.6 in that sweep's units; the
champion's higher absolute 30.8 reflects later fence/EAGLE_K tuning and the
8-bit checkpoint). The prefill knobs (`step=128`, `fence=16`) cut wall time
~39% but barely move decode — they pay off the prefill, MTP pays off decode.

### Why ~2× and not 3×

γ=2 means the theoretical ceiling is 3 tokens per main-model forward (1
guaranteed + 2 accepted drafts). Measured acceptance:

```
[MTP] mean_accept = 1.04 / 2 drafts   (hist  0:34%  1:28%  2:38%)
→ ~2.04 tokens per forward = 2.04/3 = 68% of the γ=2 ceiling
```

MTP cycle decomposition (`EXO_DSV4_MTP_PROFILE`):

```
[MTP-PROF] B=1 draft      4.84 ms   ( 5.5%)
[MTP-PROF] B=1 verify    81.7  ms   (93.4%)   ← the main DSv4 forward dominates
[MTP-PROF] B=1 accept     0.79 ms   ( 0.9%)
[MTP-PROF] B=1 rollback   0.17 ms   ( 0.2%)
[MTP-PROF] B=1 total     87.5  ms
```

Verify (the full DSv4 forward) is 93% of the cycle, so the draft + accept +
rollback bookkeeping is nearly free — the speedup is governed entirely by
acceptance rate. **γ=2 is the sweet spot**: γ=1 is −6% (less amortization of the
verify cost), γ=3 is −18% (the third draft has too-low acceptance and wastes
forward work).

## 4. Quality — the part that almost didn't survive MTP

Throughput-clean is meaningless without quality-clean (a hard-won lesson on this
project: symmetric per-stream throughput once turned out to be pure BOS spam).
Every config below was quality-gated.

### AI-Stupid-Meter execution suite (the comparison that started this)

`aistupidlevel.info` runs an execution-based 7-axis suite (correctness, spec
compliance, code quality, efficiency, stability, refusal, recovery). We ported
it offline to run against the cluster's OpenAI-compatible endpoint
(`bench/aistupid_harness.py`): 8 coding tasks × 5 trials, deterministic graders
that actually execute the generated code.

**MTP-on + tie-break fix** (`aistupid_tbv2`) vs **MTP-off reference**
(`aistupid_specoff_full`) — identical profiles:

| axis | MTP-on + fix | MTP-off ref |
|------|-------------:|------------:|
| correctness | **1.0000** | 1.0000 |
| spec compliance | 1.0000 | 1.0000 |
| code quality | 0.684 | 0.681 |
| efficiency | 0.517 | 0.488 |
| stability | 0.950 | 0.950 |
| refusal | 1.000 | 1.000 |
| recovery | 1.000 | 1.000 |
| **absolute correctness %** | **100.0** | 100.0 |
| `</think>` leaks | **0** | 0 |

Every one of the 8 tasks scored 100% correctness across all 5 trials in both
configs (CI `[100.0, 100.0]`, SE 0.00). The minor code-quality / efficiency
deltas are within trial noise and favor neither config systematically. The
8-bit cluster shows **no measurable quality crack vs the MTP-off reference** on
this suite.

### The bug the tie-break fix closed

Before `c840bc2d`, `optimize_fibonacci` failed intermittently under MTP-on
(3/5, with a literal `</think>` leaking mid-content and a doubled answer). Root
cause (confirmed by controlled A/B, full writeup in
`docs/mtp-tiebreak-losslessness-fix.md`):

- MTP's **batched** verify forward differs from a **sequential** single-token
  greedy decode by ~1 ulp (different reduction/batching order).
- At temp 0 that flips **tied** tokens. One early tie-flip (`Iteration` vs `So`
  at char 335 of an identical prefix) sent the whole generation onto a
  degenerate trajectory.
- This is a genuine losslessness violation — speculative decoding at temp 0 is
  supposed to be token-identical to non-speculative greedy. It was not, at ties.

The fix: deterministic tie-break on the bonus token (among tokens within `eps`
logits of the max, pick the lowest id). Both batched and sequential forwards see
the *same tied set*, so the pick is stable across both → the cascade is cut at
its source. As a free bonus, high-id specials like `</think>` (128822) are
de-prioritized at ties. Costs no extra forward, mutates no cache state, leaves
draft acceptance bit-exact.

### Long-context integrity

The 100K needle-in-haystack quality gate (`bench/quality_probe_dsv4.py`, secret
`FALCON-MERCURY-7749` in the middle third) passes at the champion config — the
needle is recovered, no BOS spam, no `</think>` leak. The gate caught two
regressions during tuning that pure-throughput metrics would have shipped
(`EXO_DSV4_INDEX_TOPK=128` → model emitted a single `.`; `PREFILL_STEP_SIZE`
64/512 → needle missed), which is why `INDEX_TOPK` is floored at 512 and the
step size is pinned at 128.

## 5. How this compares to the published reference

`aistupidlevel.info`'s DeepSeek-V4-Flash entry (model 260) reports ~1.00
correctness on its hourly execution suite. Our local port reproduces that:
**100% absolute correctness**, hourly and combined both 100.0, on the 8-bit
checkpoint served over 2-node RDMA TP. The gauge sits at 50.0 because the
offline port uses `self_baseline(PROXY)` (no cross-model z-scoring), so the
meaningful comparison is the **absolute correctness and per-axis profile**, not
the centered gauge — and on those, the cluster matches the reference.

The takeaway: serving DSv4-Flash 8-bit on this cluster, with MTP on, is **not
quality-degraded relative to the reference** on every task we can grade
deterministically — and it does it at 30.8 tok/s.

## 6. Knob reference (what's load-bearing)

| knob | champion value | effect | notes |
|------|---------------:|--------|-------|
| `EXO_DSV4_MTP` + `EXO_SPECULATIVE` | 1, 1 | decode +25% | the speedup; self-spec, no draft model |
| `EXO_SPECULATIVE_GAMMA` | 2 | sweet spot | γ=1 −6%, γ=3 −18% |
| `EXO_DSV4_MTP_TIEBREAK_FIX` | 1 | ~0 t/s, fixes correctness | set 0 to opt out |
| `EXO_DSV4_MTP_TIEBREAK_EPS` | 0.5 | covers 82% of observed flips | all observed flips < 3.0 logits |
| `EXO_DSV4_MTP_EAGLE_K` | 8 | MTP soft-emb mixture width | K=1 fast-path / K>1 mixture |
| `EXO_PREFILL_STEP_SIZE` | 128 | wall −38% | narrow sweet spot; 64/512 break quality |
| `EXO_DSV4_FENCE_EVERY_N_LAYERS` | 4 | decode +13% (asymptote ~16) | cross-rank fence batching |
| `EXO_DSV4_INDEX_TOPK` | 512 | quality floor | **never below 192**; 128 broke quality |
| `EXO_KV_CACHE_BITS` | 0 (bf16) | quality floor | **do not quantize KV** |

**Forbidden levers** (proven to break quality on this stack, do not propose):
`EXO_KV_CACHE_BITS != 0`, `EXO_DSV4_INDEX_TOPK < 512`.

## 7. Reproduce

```bash
# 1. champion cluster (defaults bake in the whole champion config)
./start_cluster.sh

# 2. throughput (c=1, 100K, γ=2, 10 iters)
python bench/concurrent_bench.py --concurrency 1 --context 100000 --iters 10

# 3. quality — execution suite (8 tasks × 5 trials)
python bench/aistupid_harness.py --trials 5

# 4. quality — 100K needle
python bench/quality_probe_dsv4.py
```

Result files from this writeup: `bench/aistupid_tbv2.json` (MTP-on + fix),
`bench/aistupid_results_specoff_full.json` (MTP-off reference).

## 8. Related docs

- `docs/mtp-tiebreak-losslessness-fix.md` — the root-cause investigation and fix.
- `docs/thinking-parser-fused-delimiter-fix.md` — companion parser fix (the one
  piece that *is* upstreamable; filed as exo#2149).
- `docs/fork-notes.md` — full c=1 100K tuning sweep, all 24 experiments, profile
  decomposition.
- `docs/upstream-prs.md` — fork lineage and upstreamability verdict (nothing in
  mlx/mlx-lm has a clean upstream target; the DSv4 stack is fork-only).
