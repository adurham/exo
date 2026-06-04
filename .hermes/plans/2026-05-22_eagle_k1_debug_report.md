# Eagle K=1 c=2 Regression — Debug Report

**Verdict (one-line):** A new hypothesis **H5** — not on the original list — is the bug.
The Eagle soft-embedding is computed from **rank-local `prev_logits`**, bypassing the
cross-rank token-broadcast that the c=2 draft loop installs specifically to mask
MLX's documented per-rank logit drift. Once drift flips argmax in *one* batch
element on *one* rank, that batch element's soft-emb diverges between ranks, the
next MTP forward runs with materially different inputs across ranks, and the
divergence compounds through subsequent chain steps and spec cycles. H1–H4
are all falsified.

---

## Section 1 — Data flow map

### Where `soft_emb` is computed

There are **two callers**, one per concurrency mode:

**c=1 path** — `src/exo/worker/engines/mlx/speculative/mtp_module.py:664–698`
```python
_eagle_k     = int(getattr(mtp_pred, "eagle_k", 0) or 0)
_eagle_embed = getattr(mtp_pred, "embed_tokens", None) if _eagle_k > 0 else None
_eagle_active = _eagle_k > 0 and _eagle_embed is not None
prev_logits = None

for i in range(gamma):
    _eagle_installed = _eagle_active and prev_logits is not None
    if _eagle_installed:
        soft_emb = _compute_eagle_soft_emb(prev_logits, _eagle_embed, _eagle_k)
        mtp_pred.set_eagle_soft_emb(soft_emb)
    try:
        logits, h = mtp_pred.predict(h, tok_arr, return_hidden=True,
                                     draft_mode=fast_lm_head)
    finally:
        if _eagle_installed:
            mtp_pred.set_eagle_soft_emb(None)
    prev_logits = logits                                  # un-synced
    if temp == 0:
        tok_arr = mx.argmax(logits, axis=-1).reshape(1, 1)
        if sync_drafts and not _disable_broadcast:
            tok_arr = broadcast_from_canonical(            # only tok_arr is synced
                tok_arr.astype(mx.int32), sync_group
            )
```

**c=2 path** — `src/exo/worker/engines/mlx/speculative/dsv4_mtp.py:1542–1605`
```python
_eagle_k     = int(getattr(self.mtp, "eagle_k", 0) or 0)
_eagle_embed = getattr(self.mtp, "embed_tokens", None) if _eagle_k > 0 else None
_eagle_active = _eagle_k > 0 and _eagle_embed is not None
prev_logits = None

for i in range(gamma):
    _eagle_installed = _eagle_active and prev_logits is not None
    if _eagle_installed:
        from .mtp_module import _compute_eagle_soft_emb
        soft_emb = _compute_eagle_soft_emb(prev_logits, _eagle_embed, _eagle_k)
        self.mtp.set_eagle_soft_emb(soft_emb)
    try:
        logits, h = self.mtp.predict(h, tok_arr, return_hidden=True,
                                      draft_mode=False)
    finally:
        if _eagle_installed:
            self.mtp.set_eagle_soft_emb(None)
    prev_logits = logits                                  # un-synced
    if temp == 0:
        tok_pre_sync = mx.argmax(logits, axis=-1).reshape(-1, 1)
        if sync_drafts:
            tok_arr = broadcast_from_canonical(            # only tok_arr is synced
                tok_pre_sync.astype(mx.int32), coord_group
            )
        else:
            tok_arr = tok_pre_sync
```

`_compute_eagle_soft_emb` (`mtp_module.py:591–622`):
```python
def _compute_eagle_soft_emb(logits, embed_tokens, K):
    if logits.ndim == 2:
        logits = logits[:, None, :]                          # (B,1,vocab)
    probs       = mx.softmax(logits, axis=-1)
    topk_ids    = mx.argsort(-logits, axis=-1)[..., :K]      # (B,1,K)
    topk_probs  = mx.take_along_axis(probs, topk_ids, axis=-1)
    topk_probs  = topk_probs / topk_probs.sum(axis=-1, keepdims=True)
    topk_embs   = embed_tokens(topk_ids)                     # (B,1,K,hidden)
    return (topk_embs * topk_probs[..., None]).sum(axis=-2)  # (B,1,hidden)
```

### Where it is consumed

`mlx-lm/mlx_lm/models/deepseek_v4.py:2500–2504`:
```python
_eagle_soft = _EAGLE_CTX.get("soft_emb")
if _eagle_soft is not None:
    emb = _eagle_soft                          # (B, S, hidden)
else:
    emb = embed_tokens(next_token)             # standard hard path
e_normed = self.enorm(emb)
h_normed = self.hnorm(prev_hidden)
x        = self.e_proj(e_normed) + self.h_proj(h_normed)
```

`_EAGLE_CTX` is a single module-level dict (`deepseek_v4.py:192`);
`_set_eagle_soft_emb` (line 195) is a plain dict mutation. The dict is read
at Python call time of `__call__`, i.e. at graph-construction time — the
soft-emb tensor is **captured into the graph by reference** at that instant.

### Predictor → embed_tokens resolution

`DSv4MTPPredictor.__init__` (`dsv4_mtp.py:279–296`):
```python
inner = getattr(model, "model", None) or model.language_model.model
self._inner = inner
self.embed_tokens = inner.embed_tokens          # THE TRUNK INPUT EMBEDDING
self.final_norm   = inner.norm
self.lm_head      = (
    getattr(model, "lm_head", None) or ...
)
self.mtp_module = inner.mtp[mtp_idx]
```

And `predict()` forwards exactly that reference into the MLX MTP module
(`dsv4_mtp.py:545–554`):
```python
out = self.mtp_module(
    prev_hidden=hidden_state,
    next_token=token_ids,
    embed_tokens=self.embed_tokens,             # same nn.Embedding object
    ...
)
```

So `getattr(predictor, "embed_tokens", ...)` returns the *trunk's*
`nn.Embedding`, identical by `id()` to what the MTP module uses for its
hard-embed branch.

---

## Section 2 — H1 (lazy → eager eval) — FALSIFIED

Claim: holding `prev_logits = logits` across the loop iteration forces mlx
to materialize the logits where the prior code didn't, killing overlap.

**Counter-evidence:**
1. The prior code already needed `logits` for `mx.argmax(logits, axis=-1)`
   to produce `tok_arr`. The reference-lifetime difference is "the same
   tensor across the next predict() call instead of dropping after argmax."
2. mlx is lazy. Holding an extra Python reference does not trigger eval —
   nothing in `_compute_eagle_soft_emb` calls `mx.eval` and the consumer
   sites (softmax / argsort / take_along_axis / embed_tokens / sum) are all
   lazy graph nodes.
3. Memory pressure is trivial: `(B=2, vocab≈129280, bf16)` ≈ 516 KB held
   for ~one chain step. Negligible on 128 GB-per-node M4 Max.
4. If H1 were the cause, the symptom would be uniform slowdown across both
   streams (longer cycle wall, same per-stream rate). The brief reports
   **asymmetric per-stream throughput** ([11.52 / 9.96]). That signal is
   incompatible with a graph-overlap regression — overlap loss is symmetric
   across batch elements within one `predict()` call.

**Verdict: falsified.** The reference-lifetime change does not force eval
and does not match the asymmetric symptom.

---

## Section 3 — H2 (K=1 algebra) — FALSIFIED

Claim: `_compute_eagle_soft_emb` at K=1 is not bit-equivalent to
`embed_tokens(argmax)`.

**Walkthrough at K=1, shape (B,1,vocab):**

1. `probs = softmax(logits)` — lazy.
2. `topk_ids = argsort(-logits)[..., :1]` — shape `(B,1,1)`. For untied
   logits this is the same index as `argmax(logits)`. Tie-breaking between
   `argsort` (stable sort by value) and `argmax` (first-index of max) can
   differ in theory; in practice bf16 logits over a 129 280-entry vocab
   have effectively zero exact ties. Even if a tie occurred, the *value*
   selected is by definition equal; only the *index* could differ — and
   that would only matter if two distinct token IDs shared the maximum
   logit exactly, which is vanishingly improbable for natural LM logits.
3. `topk_probs = take_along_axis(probs, topk_ids, -1)` — shape `(B,1,1)`.
4. `topk_probs / topk_probs.sum(-1, keepdims=True)` — for K=1 this is
   `x / x = 1.0` exactly (no FP error; the same value divides itself).
5. `topk_embs = embed_tokens(topk_ids)` — shape `(B,1,1,hidden)`.
6. `(topk_embs * 1.0).sum(axis=-2)` — collapses K dim, yields
   `embed_tokens(topk_ids).squeeze(-2)`.

Net result at K=1: `soft_emb == embed_tokens(argmax(logits))` modulo a
near-impossible tie-break. The algebra is correct.

**Verdict: falsified.** K=1 numerically collapses to the hard-embed value
on a single rank.

---

## Section 4 — H3 (clear-timing race) — FALSIFIED

Claim: `set_eagle_soft_emb(None)` in `finally` races with the lazy compute,
so the MTP forward sometimes sees `None` and falls back to hard-embed.

**Counter-evidence — read the consumer:**

`_EAGLE_CTX.get("soft_emb")` at `deepseek_v4.py:2500` runs in Python during
`DeepseekV4MTPModule.__call__`. By the time `self.mtp_module(...)` *returns*
in the predictor, that Python dict read has already happened — the result
is captured into the lazy graph as a `mx.array` reference. Subsequent
mutations of `_EAGLE_CTX["soft_emb"]` cannot change which tensor is in
that graph node; they only affect *future* `__call__` invocations.

Sequence:
1. Caller does `set_eagle_soft_emb(soft_emb)` → dict["soft_emb"] = soft_emb.
2. Caller does `self.mtp.predict(...)`:
   - inside predict: `self.mtp_module(...)` runs.
   - `__call__` reads `_EAGLE_CTX.get("soft_emb")` → returns the soft_emb
     ref. Branch picks `emb = _eagle_soft`. Graph captures this reference.
   - `__call__` returns `(logits, hidden)` (both lazy arrays whose graph
     transitively references soft_emb).
3. Caller's `try` block exits; `finally` runs `set_eagle_soft_emb(None)`.
   - Mutates the dict but does NOT touch the graph already built.
4. Later `mx.eval` flushes the graph using the captured soft_emb.

Single-threaded per process per the side-channel design comment
(`deepseek_v4.py:163–164`), so no re-entry can occur between predict()
return and the `finally` clear.

Also, if H3 were the bug, the symptom would be "iter ≥1 silently behaves
like hard-embed" — i.e., K=1 would be identical to hard-embed, not worse.
The opposite of what's observed.

**Verdict: falsified.**

---

## Section 5 — H4 (wrong embed_tokens) — FALSIFIED

Claim: `getattr(self.mtp, "embed_tokens", None)` returns the MTP module's
own embedding (or `None`), not the trunk's input embedding.

**Counter-evidence:** Both call sites do `getattr(mtp_pred, "embed_tokens",
...)` or `getattr(self.mtp, "embed_tokens", ...)` where the target is the
`DSv4MTPPredictor` (NOT the mlx-lm `DeepseekV4MTPModule`):

- c=2 path: `self.mtp` in `DSv4MTPBatchGenerator._draft_tokens_batched`
  references the predictor (used as `self.mtp.predict(...)`,
  `self.mtp._cache`, `self.mtp.activate_for_uids(...)` elsewhere in the
  same file).
- c=1 path: `mtp_pred` is the function's first parameter and the only
  caller (`dsv4_mtp.py:1677`) passes `self.mtp` (the predictor).

`DSv4MTPPredictor.__init__` explicitly assigns:
```python
self.embed_tokens = inner.embed_tokens     # dsv4_mtp.py:288
```
where `inner` is `model.model` (DeepseekV4Model). This is the **same**
`nn.Embedding` object the trunk uses at `deepseek_v4.py:2585` (`h =
self.embed_tokens(inputs)` in `DeepseekV4Model.__call__`) and the same one
passed into the MTP module's hard-embed branch via the `embed_tokens=...`
kwarg at `dsv4_mtp.py:548`. There is exactly one embedding object on the
node and both branches reach it through the same reference.

**Verdict: falsified.** Embed lookup is correct.

---

## Section 6 — H5 (NEW): un-synced `prev_logits` breaks cross-rank determinism — CONFIRMED

### The mechanism

The c=2 batched draft loop comment (`dsv4_mtp.py:1501–1521`) explicitly
documents:

> the MTP module is unsharded ... so its forward SHOULD be bit-exact across
> ranks given bit-exact inputs. In practice ... MLX produces tiny logit
> drift between ranks at cycle 5+, flipping argmax for tied/near-tied
> positions ... Without intervention the divergent draft tokens cascade
> through the verify forward into asymmetric n_accepted_per → asymmetric
> _num_tokens → asymmetric _filter_finished_uid → JACCL LEN_ERR cluster
> wedge AND garbled output even when the wedge is bandaged downstream.

The fix shipped for that drift was the **post-argmax broadcast at
`dsv4_mtp.py:1570–1572`**:
```python
tok_arr = broadcast_from_canonical(tok_pre_sync.astype(mx.int32), coord_group)
```
which forces rank 0's chosen token onto every other rank, guaranteeing the
next chain step's hard-embed input `embed_tokens(tok_arr)` is bit-identical
across ranks.

**The Eagle integration silently bypasses that guarantee:**

```python
prev_logits = logits                              # rank-local, NOT broadcast
...
soft_emb = _compute_eagle_soft_emb(prev_logits, _eagle_embed, _eagle_k)
```

`prev_logits` is captured BEFORE the broadcast. At chain step *i+1* every
rank computes its OWN `argmax(prev_logits)` inside `_compute_eagle_soft_emb`
(via `argsort(-logits)[..., :K]`). When MLX's per-rank drift flips that
argmax for any batch element on any rank — exactly the failure mode the
broadcast was added to suppress — the soft-emb feeding the next MTP
forward **differs** between ranks for that batch element.

The MTP module forward consumes that soft-emb. The hidden state flowing
into the MTP head was produced by the sharded main-trunk forward (whose
`all_sum` collectives drift by ~1 ulp per the comment above), and that
hidden state already differs between ranks at cycle 5+. Pre-Eagle, the
hard-embed branch's `embed_tokens(tok_arr)` was forced identical by the
broadcast, so the cross-rank drift was contained to whatever drift the
shared hidden state already carried. With Eagle's un-synced soft-emb, both
inputs to `e_proj(enorm(emb)) + h_proj(hnorm(prev_hidden))` carry rank-
local drift — and the soft-emb drift is no longer bounded by 1 ulp once
argmax flips for a batch element. The next predict() then produces
materially different logits across ranks for that batch element. The
broadcast at `dsv4_mtp.py:1570` still aligns the *next* `tok_arr` to rank
0's choice, masking the symptom on outputs, but the per-rank graphs have
already evolved divergently and rank 0's "winning" argmax is now derived
from logits that disagree with every other rank's logits by more than the
near-tie margin the broadcast was designed to handle. Acceptance against
the (also-broadcasted) verify-pass tokens drops for batch elements whose
divergence happened to cross the tie boundary.

### Why this matches the observed symptom

The brief reports:
- K=1 c=2 100K iter 0: per-stream **[11.52 / 9.96]**, total 21.48 tok/s
- FENCE=4 baseline (hard-embed) iter 0: **23.29 symmetric**

Per-stream asymmetry is exactly the signature H5 predicts: drift may flip
argmax for batch-element 0 (uid 0) while leaving batch-element 1 (uid 1)
alone, so soft-emb diverges for stream 0 only. Stream 0's drafts degrade
(low acceptance), stream 1's don't, giving asymmetric per-stream
throughput. Total drops below baseline because the slow stream stalls the
joint per-cycle wall (both streams advance in lockstep through verify and
spec-cycle bookkeeping, so the slow stream paces the whole cluster).

Cycle-5+ onset of drift (per the documented comment) is consistent with
seeing the regression already at iter 0 of a 100K-context bench — 100K
prefill warms up enough cycles for drift to manifest before measurement
begins.

### Why H5 also rules out the other candidates

- **H1** would be symmetric across batch elements; observed signal is
  asymmetric. ✗
- **H2** would either work for everyone (rare ties) or fail uniformly;
  not asymmetric per-stream. ✗
- **H3** would silently make K=1 ≡ hard-embed (best case), not slower. ✗
- **H4** would mean soft-emb is computed from a different embedding
  matrix for *every* batch element on every rank — uniform garbage, not
  asymmetric. ✗

H5 is the only hypothesis that produces *per-stream* asymmetry, and it
exactly reproduces the documented drift-cascade failure mode that the
broadcast was originally designed to prevent.

---

## Proposed patch (do NOT apply — for review)

**Principle:** Make Eagle's soft-emb path inherit the same cross-rank
determinism enforcement as the hard-embed path. Two options, ranked.

### Option A (recommended) — broadcast the soft-emb after compute

Mirror the `broadcast_from_canonical(tok_arr, ...)` pattern but on the
soft-emb tensor itself. Tensor shape is `(B, 1, hidden)` ≈ 16 KB at B=2,
hidden=4096, bf16 — negligible vs the ~516 KB-per-call logits and the
4-byte-per-stream tok_arr broadcast that already runs each step.

`src/exo/worker/engines/mlx/speculative/dsv4_mtp.py` around lines 1547–1564:

```python
# BEFORE
for i in range(gamma):
    hidden_for_dump = h if drift_dump else None
    _eagle_installed = _eagle_active and prev_logits is not None
    if _eagle_installed:
        from .mtp_module import _compute_eagle_soft_emb
        soft_emb = _compute_eagle_soft_emb(
            prev_logits, _eagle_embed, _eagle_k
        )
        self.mtp.set_eagle_soft_emb(soft_emb)

# AFTER
for i in range(gamma):
    hidden_for_dump = h if drift_dump else None
    _eagle_installed = _eagle_active and prev_logits is not None
    if _eagle_installed:
        from .mtp_module import _compute_eagle_soft_emb
        soft_emb = _compute_eagle_soft_emb(
            prev_logits, _eagle_embed, _eagle_k
        )
        # Cross-rank determinism: prev_logits is rank-local and can flip
        # argmax under MLX's documented per-rank drift (see comment at
        # lines 1501-1521 above). Broadcast rank-0's soft_emb so every
        # rank's next predict() sees identical input — same guarantee the
        # post-argmax broadcast (line 1570) gives the hard-embed path.
        if sync_drafts:
            soft_emb = broadcast_from_canonical(soft_emb, coord_group)
        self.mtp.set_eagle_soft_emb(soft_emb)
```

`src/exo/worker/engines/mlx/speculative/mtp_module.py` around lines 688–691:

```python
# BEFORE
_eagle_installed = _eagle_active and prev_logits is not None
if _eagle_installed:
    soft_emb = _compute_eagle_soft_emb(prev_logits, _eagle_embed, _eagle_k)
    mtp_pred.set_eagle_soft_emb(soft_emb)

# AFTER
_eagle_installed = _eagle_active and prev_logits is not None
if _eagle_installed:
    soft_emb = _compute_eagle_soft_emb(prev_logits, _eagle_embed, _eagle_k)
    # Same cross-rank determinism fix as the c=2 batched path.
    if sync_drafts and not _disable_broadcast:
        soft_emb = broadcast_from_canonical(soft_emb, sync_group)
    mtp_pred.set_eagle_soft_emb(soft_emb)
```

**Note:** `broadcast_from_canonical` (`mtp_module.py:586–588`) is a
masked-all_gather that returns rank-0's slice; it takes a tensor of any
dtype/shape. The existing c=2 path casts `tok_pre_sync` to int32 only
because the source is int32 already and the cast is a no-op signal; for
bf16 soft-emb no cast is needed.

### Option B (simpler, K=1-only) — short-circuit K=1 to embed(tok_arr)

Since K=1 collapses algebraically to `embed_tokens(argmax)` and `tok_arr`
is already the cross-rank-broadcast argmax, skip `_compute_eagle_soft_emb`
entirely at K=1 and use `embed_tokens(tok_arr)` to build the soft-emb.
This restores K=1 ≡ hard-embed exactly. It does NOT fix K>1.

Useful as a sanity gate (validates the side-channel plumbing end-to-end)
but doesn't generalize. **Option A is strictly more general and should
be preferred.**

---

## Sanity check the patch

For K=1 specifically:
- Pre-fix: rank-local `soft_emb_r = embed_tokens(rank_r_argmax(prev_logits))`.
  Ranks disagree when argmax flips due to drift.
- Post-fix (Option A): `soft_emb` computed on every rank, then overwritten
  with rank-0's via `broadcast_from_canonical`. Every rank sees rank-0's
  `embed_tokens(rank_0_argmax(prev_logits))`.
- The hard-embed path at iter i+1 would use `embed_tokens(broadcast(
  rank_0_argmax(prev_logits)))` = `embed_tokens(rank_0_argmax(prev_logits))`.
- These are now **bit-identical** (modulo the negligible argsort-vs-argmax
  tiebreak from Section 3). K=1 ≡ hard-embed restored across ranks.

For K>1: rank-0's soft-emb is rank-0's mixture of its top-K embeddings.
Every rank uses rank-0's value, so all ranks' next MTP forward sees the
same input. Cross-rank determinism is restored at K>1 too — Eagle now has
a chance to either help or hurt, but the integration is no longer
corrupted by silent per-stream divergence.

---

## Risk assessment for K>1

Option A's broadcast fixes the *integration* bug at all K. With the bug
in place, K>1 was suffering the same divergence-cascade as K=1 (plus the
additional honest issue that K>1 is OOD relative to the trained MTP
head's expected input distribution — a non-bug). After the fix, K>1's
remaining risk is purely *quality*: the soft mixture changes the input
distribution in a way the MTP head was not trained on, so acceptance
rates may suffer even though the cross-rank state is now consistent.
That is a separate question — the patch makes it answerable by isolating
it from the determinism bug.

**Recommendation:** ship Option A and re-bench K=1 first. K=1 post-fix
should match the FENCE=4 hard-embed baseline (~23.3 symmetric at iter 0,
~34.16 ± 0.07 at steady state). If it does, the integration is sound and
K>1 becomes a pure quality investigation. If K=1 is still off, look next
at the (small) `argsort`-vs-`argmax` tiebreak discrepancy and consider
short-circuiting K=1 via Option B.

---

## K>1 outlook

With Option A in place, K=2 / K=4 should at minimum no longer
asymmetrically degrade the c=2 cluster. Whether they *help* depends on
the MTP head's tolerance for soft-embedding inputs:

- The DSv4 MTP head was trained on hard `embed_tokens(token_id)` inputs.
  Feeding a mixture changes the input manifold. The head may produce
  poorer step-2 logits, lowering γ=2 acceptance even though cross-rank
  state is now coherent.
- Counter-balance: Eagle is supposed to help precisely when the step-1
  argmax is uncertain — the soft mixture carries the uncertainty forward
  into step-2 so step-2's argmax doesn't lock in a wrong commitment. If
  acceptance is gated on step-1 top-1 probability, K=2 may lift it.
- Worth measuring: P(top-1) at step 1 (with K=1 hard-equiv) vs K=2, and
  cross-rank logit divergence at iter 5+ (verify the broadcast actually
  flattens it). The `EXO_MTP_DRIFT_DUMP=1` instrumentation at
  `dsv4_mtp.py:1534` already supports the latter probe.

P(K=2 wins ≥ +1 tok/s/stream over K=1-fixed): moderate.
P(K>2 wins more than K=2): low — soft mixture's signal degrades fast past
K=2 for MTP heads. Worth one cycle of testing at K=2 specifically after
the patch lands.
