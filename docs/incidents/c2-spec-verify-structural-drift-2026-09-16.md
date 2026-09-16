# c>=2 speculative verify: a STRUCTURAL drift class the B==1 losslessness gates never covered

**Date:** 2026-09-16
**Status:** ROOT-CAUSE LOCALIZED (leading hypothesis, source-verified). No
runtime behaviour changed by this commit — documentation + a false-comment
correction only.
**Affects:** DeepSeek-V4-Flash(-Vision-Exp) serving with speculative decoding
ON at c>=2 (`EXO_DSV4_MTP_C2_MAX_CTX=0`). Not reachable at production defaults.

---

## 1. TL;DR

A real text-corruption defect exists in the c>=2 (multi-stream) +
speculative-decoding-ON regime: 2 of 12 measured streams emitted cross-script
glued subword fragments (`redundancyфа.`, `purposeфабрика.`) on a boot where
the already-root-caused mxfp8-lm_head defect was provably OFF
(`EXO_DSV4_LMHEAD_MXFP8=0`). Task correctness was unaffected (needle 12/12) —
only text *form* was damaged.

The localization below says the mechanism is **structural, not kernel
rounding**: at c>=2 the speculative verify forward **cannot run the
structurally-sequential path at all**, because the "losslessness stack" that
makes an L>1 verify bitwise-equivalent to L sequential decode steps is gated
to B==1 at four independent sites. That class of divergence (which KV rows a
query row can see; when a pooled window becomes visible; which pooled blocks
top-k selects) is upstream of the GEMM and is therefore **not repairable by
kernel-level batch invariance** — which is exactly why the 2026-07-11
"bitexact build" probe did not clear the defect.

**No runtime fix ships with this document.** The candidate fix is
relaunch-tier and needs an on-hardware A/B (see §5); it is deliberately left
for a user-approved decision rather than bundled here.

---

## 2. The observation (what was actually measured)

Source: the `EXO_DSV4_BOOKKEEP_FAST` production A/B's **BK0 control arm**
(flag unset), step `c2_30k` — 30K target depth (30,083 real tokens), **c=2**,
6 measured iterations + 1 warmup, `max_tokens=700`, temp=0, thinking off.

Live pid env at the time (read off the REAL process, not a fresh ssh session):

```
EXO_DSV4_LMHEAD_MXFP8=0      <- the already-fixed cause, genuinely OFF
EXO_DSV4_MTP_C2_MAX_CTX=0    <- the c>=2 spec gate deliberately opened
EXO_DSV4_BS_MIN_ACCEPT=1
EXO_DSV4_VERIFY_ROWSEQ=1     EXO_DSV4_ROWSEQ_FULLBLOCK=1
EXO_DSV4_VERIFY_ROWSEQ_VEC=1 EXO_DSV4_VERIFY_ROWSEQ_VEC_ROWSDPA=3
MLX_STEEL_BATCH_INVARIANT=1  MLX_GEMV_BATCH_INVARIANT=1
```

| metric | value |
|---|---|
| per-stream decode | 5.171 ± 0.012 tok/s |
| AGG decode (2 streams) | 10.342 ± 0.023 tok/s |
| prefill | 215.87 ± 1.13 tok/s |
| needle | **12/12** |
| glue fragments | **2** |

The two defects, read directly out of the generated text (not from a counter):

- record 2 (iter 1, stream 0): `...compiled with a significant amount of redundancyфа.`
- record 8 (iter 4, stream 0): `...the high-level description of the model's purposeфабрика.`

Both are real vocabulary tokens, not detokenizer artifacts. The Vision-Exp
tokenizer encodes `фабрика` as `[37244, 93911, 2525]` and `37244` decodes to
`"фа"` — so the model genuinely argmaxed a Cyrillic subword at a
sentence-final, low-margin position. This is the *symptom class* of the
already-fixed mxfp8 defect, but a different cause: that flag was 0.

Raw text: `~/ops_ab/results/bk0/texts_c2_30k.json` (gateway mirror
`/home/hermes/ops-ab/results/bk0/`).

### Measurement pitfall (carried forward from the earlier writeup)

The naive adjacent-script regex
(`[\x00-\x7f][^\x00-\x7f]|[^\x00-\x7f][\x00-\x7f]`) flags **7** records on
this same 14-record file, but 5 are FALSE POSITIVES (em-dash U+2014 / curly
apostrophe U+2019 next to ASCII). The harness's own `glues=N` counter (2) was
the accurate one. The discriminator is a real script transition *inside a
word*, not merely a non-ASCII codepoint adjacent to ASCII. Always read the
actual substring.

---

## 3. Root-cause localization: the verify path is B==1-only by construction

### 3.1 What the losslessness stack actually fixes

`mlx-lm/models/deepseek_v4.py` (~L2160-2178) documents the mechanism
explicitly, and it is **not rounding**:

> an L>1 decode-time pass is NOT equivalent to L sequential steps because the
> attention-side CACHE STATE evolves differently —
> * `RotatingKVCache._update_in_place` writes all L tokens BEFORE any row
>   attends, so rows 0..L-2 have their window's oldest L-1..1 tokens already
>   OVERWRITTEN (**a mask cannot restore overwritten keys**): row j attends a
>   truncated window vs its sequential twin.
> * `PoolingCache` prompt-mode `accumulate_windows` flushes a straddled
>   window visible to **ALL** rows in the pass; sequentially it flushes at the
>   boundary token and (deferred bump) becomes visible a step later.
> * the indexer score GEMM runs at M=L (steel gemm) vs M=1 (gemv) with a
>   different K-reduction order; near-cutoff score ties then select a
>   **different top-k pooled set**.

The shipped fix (`EXO_DSV4_VERIFY_ROWSEQ` + `EXO_DSV4_ROWSEQ_FULLBLOCK`) is
to run the block per row so window contents, pool-flush timing, deferred
bumps and indexer scoring "all evolve bitwise-identically to sequential
decode".

### 3.2 It is disabled at B>=2 — at four independent sites

Every one of these requires `h.shape[0] == 1` (i.e. B==1):

| site | line (2026-09-16) | what it disables at B>=2 |
|---|---|---|
| block-level FULLBLOCK per-row loop | ~5944 | whole per-row block (`attn_hc`, `attn_norm`, attention, `ffn_hc`, `ffn_norm`) |
| model-level `hc_head` per-row | ~7985 | model-level HyperHead hc-ops |
| `EXO_DSV4_VERIFY_BATCH` batched alternative | ~7918 | the depth-gated replacement path |
| `_rowseq_min_ctx()` | ~2366-2373 | even per-row **attention** below 32768 ctx |

The last one is the sharpest:

```python
def _rowseq_min_ctx(batch_size: int) -> int:
    if batch_size == 1:
        return _VERIFY_ROWSEQ_MIN_CTX      # prod exports 0 -> rowseq everywhere
    return max(_VERIFY_ROWSEQ_MIN_CTX, 32768)   # B>=2: only at >=32K ctx
```

`EXO_DSV4_VERIFY_ROWSEQ_MIN_CTX=0` is exported by the launcher (row-seq at
*all* contexts) — but that only takes effect at B==1. At B>=2 the floor is
hardcoded to 32768.

**Net effect.** At c>=2 there is no configuration in which the verify forward
is structurally sequential:

- **ctx < 32768** (where this defect was observed, at 30K): the *entire*
  verify forward is the classic batched path — attention, hc-ops, norms.
- **ctx >= 32768**: only the attention is per-row'd; the hc-head and norm
  ops remain batched at B>=2 permanently.

And even the ctx >= 32768 attention path is not mask-identical to a real
decode step at B>=2: `EXO_DSV4_ROWSEQ_ROWMASK` is applied only when
`normed.shape[0] == 1` (~L6132 — "the c>=2 rowseq path was validated with
mask=None rows"), while the batched cache classes return an explicit ARRAY
mask at every N including 1. That is the same "different SDPA specialization
than real decode steps" deviation the source itself records at ~L2214 as
producing ulp-level drift at pool-flush rows.

Three-deep redundant gating (`h.shape[0] == 1` in three separate places plus
a batch-aware context floor) is itself evidence this was **deliberately**
scoped to B==1 — not an oversight — most likely because the c>=2 path predates
the B==1 losslessness work and was left on its "previously-validated" path.

### 3.3 Why the 2026-07-11 refutation does not cover this

`start_cluster.sh` records, verbatim:

> Steel-level batch invariance ... : required before re-enabling spec at c>=2
> (kernel layer is proven bitexact with it, mlx ac73d0c9) but costs ~5% c=1
> decode and the **c>=2 spec corruption ALSO has an unresolved serving-logic
> component (2026-07-11 probe: degens persist on the bitexact build)**

That probe tested the **kernel-rounding leg only**. `MLX_STEEL_BATCH_INVARIANT`
makes a batched GEMM reduce in the same order as the unbatched GEMV — a
property of the arithmetic *given identical inputs and identical attention /
masking structure*. It cannot touch:

1. which KV rows are visible to which query row (window-overwrite ordering),
2. when a pooled window becomes visible to other rows (flush timing),
3. which tokens win top-k selection under an M=L vs M=1 reduction order.

Those are **input / masking / selection** differences that occur *upstream* of
the GEMM. So the "degens persist on the bitexact build" result is consistent
with the structural cause and does not falsify it. Kernel-BI was never the
complete fix for this class.

### 3.4 Why this is a *text-form* defect and not an incoherence

The failure mode is a low-margin argmax flip: the target distribution at that
position is nearly flat, and a shifted context/attention view is enough to
promote one token ahead of another. That is precisely what a structurally
different attention view produces, and precisely why:

- needle-in-haystack still passes (task content lives in high-margin
  retrievals), and
- the damage shows up only as *which of two near-tied tokens* won.

It also explains the character of the observed fragments: a real, well-formed
Cyrillic subword (not punctuation, not a byte artifact) — the model picked a
legitimate but wrong token from a flat distribution.

---

## 4. What is NOT established

- **The mechanism is the leading hypothesis, not proven.** No logprobs capture
  (`logprobs=True, top_logprobs=10`) was taken on the c>=2 spec-ON path. Until
  that exists, this could still be, e.g., a KV/position defect of the June-2026
  shape rather than an argmax flip. Do not present it as confirmed.
- **Rate is a single small sample.** 2 defects in 12 measured streams
  (~1 in 6) at 30K only. Depth dependence is unknown.
- **The causal link from the gate asymmetry to these two specific tokens is
  not observed directly.** What is established: the gates exist, they are
  B==1-only, and the kernel-BI probe therefore did not test this leg.
- The four `== 1` sites were located by line-range grep on a large file. A
  future session should re-grep (line numbers move) and confirm there is no
  fifth consumer of these gate booleans.

---

## 5. Why no runtime fix shipped here

The candidate fix is: remove the `h.shape[0] == 1` gates and set the B>=2
rowseq context floor to 0, so the c>=2 verify runs the structurally-sequential
path. That is **not** a low-risk change:

1. It lives in the **mlx-lm submodule** → fork commit + gitlink bump in the
   parent, not a local edit.
2. It changes verify **numerics** for the c>=2 regime, with a documented
   ~**1.6x** verify-time cost at short context.
3. It cannot be validated on the Linux gateway box — that host has no
   `mlx.core` (MLX is macOS/ARM-only), so only the pre-existing collection
   errors reproduce there.
4. Validating it needs a **production relaunch + A/B**, which the local
   operating discipline treats as a separate, user-approved risk tier.
5. Most importantly, it would be a fix for a **hypothesis**, shipped into a
   path that is currently **disabled in production**.

Shipping it now would be the wrong order of operations: changing verify
numerics in an untestable-locally regime with no ability to confirm the fix
worked or that it introduced something new.

### The next step, in order

1. Take a **c>=2 spec-ON repro with `logprobs=True, top_logprobs=10`** and
   check whether the defect positions show a flat cross-lingual top-10 (the
   argmax-flip signature) or something else. This is the cheap discriminator
   and it needs a launch, not a code change.
2. If the signature matches, the fix is the gate removal in §5 — run it as a
   **measurement boot + a separate restore boot** (never leave production
   flipped as a side effect of measuring).
3. Keep `EXO_DSV4_MTP_C2_MAX_CTX=1` armed until then.

---

## 6. Why this document also corrects a comment

Two places asserted things that are false of the **deployed** system, and the
error is dangerous in one direction — it invites a future session to open the
gate believing nothing would change:

- `start_cluster.sh` (~L2230-2240) said the gate "was REMOVED", that setting
  the var to 0 "is a no-op (no gate exists to disable)", and that "MTP-on at
  c>=2 high context is now clean through 500K".
- `dsv4_mtp.py` (~L2366-2376) said the var "no longer has a
  default-threshold effect".

Both are **half** true and were read as fully true. What actually holds: the
gate no longer *default-disables* spec at c>=2 (that is the part that was
removed on 2026-06-24), but it is **still a live threshold gate** —
`dsv4_mtp.py:2377-2400` reads the var and forces `spec_eligible=False` at
c>=2 whenever it is nonzero and the max cache offset exceeds it. The code
default is `0`, the **launcher exports `1`**, so production has c>=2 spec
disabled for every real generation.

The `through 500K all_needles=True` claim is also not a text-quality claim —
needle retrieval passes in this regime (12/12 here too). It is needle recall,
not coherence of prose, and the 2026-09-16 defect above is the counterexample.

Both comments are corrected in this commit. The rule that prevents the next
misreading is already in the ops skill and is repeated here: **read the var off
the live pid, never from the comment.**
