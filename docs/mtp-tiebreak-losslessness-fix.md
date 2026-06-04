# MTP Speculative Decoding — Tie-Break Losslessness Fix

**Status:** Shipped, default-on (2026-06-04). Fork commit `c840bc2d` on
`adurham/exo` main.

**TL;DR:** DSv4-Flash MTP self-speculation was *not* token-identical to
sequential greedy at temperature 0. A ~1 ulp difference between the batched
verify forward and a sequential single-token decode flips **tied** tokens; one
flipped tie early in a generation cascades the whole output onto a different
(often degenerate / repetition) trajectory — producing a spurious `</think>`
mid-reasoning and a wrong final answer on hard prompts. A deterministic
tie-break on the bonus-token selection (lowest token id among tokens within
`eps` logits of the max) restores losslessness at ~0 throughput cost.

---

## 1. Symptom

On a sequential benchmark batch (the ported `aistupidlevel.info` 7-axis suite,
`bench/aistupid_harness.py`), the `optimize_fibonacci` task failed
intermittently and position-dependently:

- Failed on the **first 1-2 trials** of the task when it appeared 8th in a
  batch (after warmup), then **self-recovered** on trials 3-5.
- Failing outputs were long (content 2000-5300 chars, `finish_reason=stop`) and
  contained a literal `</think>` token **inside the content stream**, followed
  by a second copy of the answer.
- Did **not** reproduce on isolated single requests (14/14 clean single-shot).
- Disappeared entirely with speculation off.

Per-axis: hourly coding tasks 1-7 were unaffected (they emit short answers);
only the harder task that triggered longer reasoning failed.

## 2. What it was NOT (ruled out with evidence)

A multi-stage investigation (diagnostic commits `730cd460`, `c7c57b37`,
`75a31587`, `fcbe1cec`) ruled out the obvious suspects:

1. **Not the chat-template / `</think>` detection.** End-to-end the serving
   stack correctly splits `reasoning_content` from `content` and terminates on
   EOS. `tokenizer.think_end` resolves correctly (transformers
   `get_vocab(with_added_tokens=True)` includes `</think>` = 128822).
2. **Not the acceptance criterion.** Static analysis confirmed accept-iff
   `draft == argmax(verify_logits)` at temp 0, correct positions, correct tree
   walk. The accept logic is sound.
3. **Not PoolingCache contamination.** A runtime audit
   (`EXO_DSV4_MTP_VERIFY_AUDIT`) captured 28 special-token cycles; every one had
   `pools=[]` (no PoolingCache active in this config). The 2026-05-29 pool
   snapshot/restore fix is irrelevant here.
4. **Not a confidently-wrong verify forward.** An every-cycle reference check
   (`EXO_DSV4_MTP_REFCHECK_ALL`) compared the batched verify argmax against a
   clean single-token forward at the committed prefix. Across 95 cycles / 89
   divergences: **zero** `</think>` divergences at commit time, and **all 89
   divergences were near-ties** (top-2 logit gap median 0.25, max < 3.0).
5. **Not quantization.** 8-bit on the cluster produced correct answers on the
   same prompts when speculation was off.

## 3. Root cause (confirmed via controlled A/B)

Direct A/B: identical warmup + prompt, temp 0, MTP-on vs MTP-off. The two
reasoning streams are **token-identical for the first 335 characters**, then
diverge at a single token:

```
shared:   ...the problem says "no recursion, use memoization or iteration". 
MTP-off:  Iteration is fine. For n up to 10000...      -> short correct answer
MTP-on:   So iterative is fine. However, for n=10000...-> 14.5K-char reasoning,
                                                           degenerate repetition,
                                                           spurious </think>,
                                                           wrong fast-doubling code
```

The first divergent token is a near-tie: `Iteration` vs `So` had near-equal
logits. The **batched** MTP verify forward (which processes `[y, draft_0..
draft_{γ-1}]` in one pass) differs from a **sequential** single-token decode by
~1 ulp due to different reduction/batching order. At temperature 0 that tiny
difference is enough to flip which tied token wins the argmax. Once one token
differs, the two generations are on entirely different — each individually
plausible — trajectories. For hard prompts, MTP-on's perturbed path tends to
land in a degenerate over-thinking / repetition basin.

This is a genuine **losslessness violation**: speculative decoding at temp 0 is
supposed to be token-identical to non-speculative greedy. It was not, at tie
positions.

## 4. The fix

`src/exo/worker/engines/mlx/speculative/dsv4_mtp.py`, `_speculative_next`
(the c=1 linear path), in the temp==0 branch:

```python
# Among tokens within eps logits of the per-position max, pick the LOWEST id.
_maxlogit = mx.max(_vl0, axis=-1, keepdims=True)
_tied = _vl0 >= (_maxlogit - _tb_eps)           # tied set per position
_cand = mx.where(_tied, mx.arange(vocab), vocab) # untied -> sentinel
all_next = mx.argmin(_cand, axis=-1)             # lowest tied id
```

`all_next` feeds **only the bonus token** (`bonus_val`). The bonus is the token
emitted after the accepted drafts; it becomes next cycle's `y`.

**Why it works:** both the batched verify and a hypothetical sequential forward
see the *same tied set* (the ~1 ulp difference changes ordering within the set,
not membership), so picking the lowest id is stable across both → the early
tie-flip cannot happen → the cascade is cut at its source. As a bonus, high-id
specials like `</think>` (128822) are naturally de-prioritized at ties while
clear-margin (legitimate) picks are untouched.

**Why it is safe (and why fix v1 was not):**
- Only `all_next` / `bonus_val` change. The bonus is **not yet in the KV
  cache** (it is written next cycle as `y`), so changing it mutates **no cache
  state**, needs **no extra forward**, and cannot desync `pre_norm`.
- **Draft acceptance is unchanged** — it compares against `target_tokens`
  (a separate argmax), so accepted drafts stay bit-exact in the KV cache.
- The earlier failed approach (`552a4193`, reverted in `63c5120c`) tried to
  *recompute* the bonus via `trim(1) + re-feed` after rollback. That swapped the
  bonus token while leaving `pre_norm` pointing at the old token's hidden state,
  desyncing the MTP draft head → a worse `We'll.We'll.We'll...` repetition loop.
  The extra forward also double-advanced cache state. v2 touches no cache and
  has no such coupling.

**Config:** `EXO_DSV4_MTP_TIEBREAK_FIX` (default `1`; set `0` to opt out),
`EXO_DSV4_MTP_TIEBREAK_EPS` (default `0.5` logits — covers the 82% of observed
flips with gap < 0.5; all observed flips were < 3.0).

Scope: the c=1 linear `_speculative_next` path only (per the project's c=1-only
serving target). The c>1 `_speculative_next_batch` and tree
`_speculative_next_tree` paths are not patched.

## 5. Validation

**Correctness** (`bench/aistupid_harness.py`, MTP-on + fix, temp 0):
- `optimize_fibonacci`: 5/5 PASS (was 3/5 with `</think>` leaks pre-fix).
- Full 8-task suite: **100.0% absolute correctness** — matches MTP-off.
- Zero `</think>` leaks, zero BOS-spam, no repetition.

**Throughput** (`bench/concurrent_bench.py`, c=1, 100K ctx, γ=2, 10 scored iters):

| metric | value |
|--------|-------|
| mean agg_tps | 30.77 |
| median | 30.8 |
| min / max | 30.6 / 30.8 |
| σ | 0.067 |
| errors | 0 / 10 |

Passes the project bar (γ=2: ≥10 iters, all ≥29 t/s, σ<0.5, errors=0) and
matches the prior MTP-on champion (30.7 t/s) — the tie-break costs ~0
throughput (it is an argmax/where over the logits already in flight, no extra
forward).

**Quality probe at the benched config:** "capital of France?" → "Paris"
(finish=stop, 0 BOS, 0 `</think>` leak); code-fix prompt → correct
`def add(a,b): return a + b`.

## 6. Result

MTP self-speculation is now both **fast** (30.8 t/s c=1 100K) and **lossless on
hard prompts** (100% suite correctness). Previously these were mutually
exclusive: MTP-on was fast but hit the spurious-`</think>` / degeneration bug,
and correctness required disabling MTP (~27 t/s). This is the new default
champion config.

## 7. Diagnostic instruments (env-gated, off by default, kept in tree)

- `EXO_DSV4_MTP_VERIFY_AUDIT=<path>` — JSONL dump of special-token cycles
  (commit `730cd460`).
- `EXO_DSV4_MTP_REFCHECK=<path>` — per-special-token reference-forward check
  (`c7c57b37`).
- `EXO_DSV4_MTP_REFCHECK_ALL=1` — every-cycle reference check, logs divergences
  (`75a31587`).

These produced the evidence in §2-3 and remain available for regression
analysis. They are no-ops unless their env var is set.

## 8. Upstreamability

The MTP speculative subsystem (`dsv4_mtp.py` and the rest of
`worker/engines/mlx/speculative/`) is **fork-only** — `exo-explore/exo` does not
carry it — so this specific fix is not a standalone upstream PR (there is no
`_speculative_next` upstream to patch). The *general lesson* (a batched
spec-verify forward is not bit-identical to sequential greedy, so temp-0
speculative decoding needs a deterministic tie-break to stay lossless) applies
to any speculative-decoding implementation and is worth raising upstream if/when
the speculative subsystem is contributed. See
`docs/thinking-parser-fused-delimiter-fix.md` for the companion parser fix,
which *is* directly upstreamable.
