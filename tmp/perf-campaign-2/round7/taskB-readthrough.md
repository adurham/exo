# Task B Read-Through — STEEL_BATCH_INVARIANT kernel scope + ROWSEQ_VEC liveness

Read-only source investigation. No cluster commands run. `build/` directories
ignored throughout (stale mirrors).

Prior audit consulted for context: `tmp/perf-campaign-2/round1/I2-C2-TAX-AUDIT.md`
rows 2/7 and its "Interaction constraint verification" section. Its core claim —
that the `MLX_STEEL_BATCH_INVARIANT=0` ⇒ `EXO_DSV4_VERIFY_ROWSEQ_VEC=0` pairing
is a comment-only, not code-enforced, requirement — is verified independently
below and confirmed accurate as of this read.

---

## Q1 — Kernel scope of `MLX_STEEL_BATCH_INVARIANT`

### All call sites of `steel_batch_invariant_enabled()`

Defined `mlx/mlx/backend/metal/matmul.cpp:107-113` (static, `getenv`-backed,
resolved once per process on first call — a relaunch is required to change it).
Sibling flag `MLX_GEMV_BATCH_INVARIANT` / `gemv_batch_invariant_enabled()` is
defined right above it (`matmul.cpp:93-99`) and is a **separate, independently
gated flag** — do not conflate the two; only `MLX_STEEL_BATCH_INVARIANT` is in
scope for this question.

Four call sites, all in `mlx/mlx/backend/metal/matmul.cpp`:

1. **`matmul.cpp:144-146`** — inside the `GEMM_TPARAM_MACRO` macro's "Large
   device" (`devc == 'd'`, i.e. Mac Studio Ultra-class) branch, used by every
   steel-dispatch tile-selection call site (`steel_matmul_regular_axpby_nax`,
   `steel_matmul_regular_axpby`). When true, the tile-size threshold check uses
   `(size_t)M * N >= 1<<20` (per-batch-element M,N only); when false it uses
   `(size_t)batch_size_out * M * N >= 1<<20` (the full batch multiplies in).
   <br>**Effect:** with the flag OFF, a GEMM's tile config (bm/bn/bk/wm/wn) can
   change depending on how many *other* batch elements ride along, even though
   each element's own (M,N,K) is identical — e.g. a lone stream's attention
   projection could pick a different tile than the same stream's projection
   when batched with a sibling stream. ON pins the tile choice to the
   per-element shape alone, independent of batch_size_out.

2. **`matmul.cpp:963-964`** (`steel_matmul_regular_axpby_nax`, "Case 1: Small
   M×N with large K, use SIMD split-K") — guard:
   `if (!steel_batch_invariant_enabled() && !use_nax && batch_size_out == 1 && (_tm*_tn) <= min_tmn_threshold && _tk >= 8 && K >= std::max(M,N))`.
   <br>**Effect:** with the flag ON, this branch is unreachable — split-K is
   never used for this shape class; the regular (non-split-K) steel kernel
   runs instead. The comment at `matmul.cpp:959-962` states directly: *"split-K
   is a different reduction order and only exists for batch==1, so a row would
   round differently solo vs inside a batch (observed on the sdpa-fallback
   output matmul M=64, K=kv_len)."*

3. **`matmul.cpp:987-989`** (`steel_matmul_regular_axpby_nax`, "Case 2: Large K
   with sufficient M, N, and NAX is available, use NAX split-K") — same pin,
   NAX-hardware variant of case 1: `!steel_batch_invariant_enabled() && use_nax && batch_size_out == 1 && (...)`.
   Same effect (disables the NAX split-K route) for the same batch-invariance
   reason.

4. **`matmul.cpp:1511-1522`** (`Matmul::eval_gpu`, "Batch-invariance for the
   collapse itself") — guard:
   `else if (steel_batch_invariant_enabled() && batch_size_out > 1 && !a_transposed && batch_shape.size() >= 2 && ...)`.
   <br>**Effect:** this branch only *exists* (does anything) when the flag is
   ON. It generalizes the batch-into-M collapse to fold the *last* batch
   dimension into M even when a real leading batch dim (rank ≥ 2, e.g.
   multiple concurrent decode streams) remains — per the comment
   (`matmul.cpp:1499-1510`), without this, "a leading real batch dim... skipped
   the collapse entirely and fell to batched gemv — a different kernel class
   with a different accumulation order (~2 ulp bf16... 2026-07-11), so a
   stream's attention rounds differently solo vs batched." With the flag OFF
   this collapse never runs and such shapes take the pre-existing (rank-1
   collapse only) path, i.e. more cases fall through to batched-gemv instead
   of a uniform steel kernel.

**Summary of what the flag changes:** it forces (a) tile selection to depend
only on the per-row (M,N) shape, never on how many batch elements/streams ride
along; (b) split-K (both classic-SIMD and NAX variants) to be disabled
entirely, since split-K's reduction order is batch-size-dependent; (c) an
additional batch→M collapse path to activate so more shapes take a single
uniform steel kernel rather than splitting between steel and batched-gemv
depending on batch rank. All three together are what the comment at
`matmul.cpp:101-106` summarizes: *"extends batch invariance to the steel
dispatch heuristics (generalized collapse-into-M, tile selection by M*N,
split-K pinned off). Needed for cross-batch-size bitexactness (c=1 rows ==
c>=2 rows); costs ~5% single-stream decode on DSv4 (split-K retirement on the
attention output matmul)."*

### Does the c=1 decode path (M=1 draft, M=4 batched verify) reach this code?

The dispatch entry point is `Matmul::eval_gpu` (`matmul.cpp:1447` onward). The
decisive guard is at **`matmul.cpp:1554`**: `if (std::min(M, N) == 1) { return gemv(...); }`
— i.e. **any matmul where either dimension is 1 is routed to the GEMV kernel
family, never to steel**, before any `steel_batch_invariant_enabled()` check is
reached. (`AddMM::eval_gpu` has the equivalent guard at `matmul.cpp:1754`:
`if (std::min(M, N) == 1) { return gemv_axbpy(...); }`.)

For an ordinary FFN/router/projection GEMM with a large output dimension N
(hidden size, vocab, etc.):
- **M=1** (draft row) → `min(M,N)==1` is true (M=1) → routed to `gemv`/`gemv_axbpy`,
  which **never calls `steel_batch_invariant_enabled()`** — none of the four
  call sites above sit inside the gemv path. **M=1 rows on this class of
  matmul do not reach the flag-sensitive code.**
- **M=4** (batched verify, γ=3 ⇒ 1 real + 3 draft rows) → `min(M,N)==4` (for
  N≫4) → falls through the `min(M,N)==1` guard, past the
  `gemv_batch_invariant_enabled()`-gated small-M-batch-of-gemvs route at
  `matmul.cpp:1593` (which only fires for `MLX_GEMV_BATCH_INVARIANT=1`, a
  **separate** flag, and is unconditional on `MLX_STEEL_BATCH_INVARIANT`), and
  reaches `steel_matmul` (`matmul.cpp:1626`) → `steel_matmul_axpby<false>` →
  `steel_matmul_regular_axpby` — which **does** contain call sites 1-3 above
  (tile selection + both split-K gates). **M=4 verify rows on this class of
  matmul DO reach the flag-sensitive code.**

However, the comment's own worked example (`matmul.cpp:961`, *"observed on the
sdpa-fallback output matmul M=64, K=kv_len"*) is **not** talking about the raw
decode-row count as M. The attention output matmul (P·V, the SDPA fallback)
collapses the per-head batch dimension into M via the "Collapse batches into
M if needed" logic at `matmul.cpp:900-923` / `1488-1498` (and, when the flag
is ON, the additional collapse at `1511-1522`) — so for this specific matmul,
M is effectively **num_attention_heads** (e.g. 64), not the decode-batch row
count, and this collapse-to-M=64 happens **regardless of whether the decode
step is a single token (draft, "rows"=1) or a 4-row batched verify** — each
row's own P·V attention-output matmul is dispatched with heads folded into M
the same way. This is the mechanism by which the documented "~5% c=1 decode"
cost is paid even on ordinary single-token (M=1-row) decode: the *decode row
count* is 1, but the *matmul's effective M* (after head-collapse) is not, so
this particular GEMM (attention output projection) reaches the split-K gates
(call sites 2-3) on every decode step, draft or verify, independent of the
γ-driven row count.

**Net answer:** whether the flag-sensitive code is reached depends on which
matmul within a decode/verify step you're asking about, not on decode-row-count
alone. Large-N linear-layer GEMMs (FFN, router, lm-head, o-proj as issued per
raw token row) at M=1 (draft) are routed to gemv and bypass the flag entirely;
the same layers at M=4 (batched verify) are routed to steel and are affected
by all three code-live gates (1-3, plus 4 when batch_size_out>1). The
attention **output** matmul (P·V) is affected at **both** M=1 draft and M=4
verify, because its effective M is the collapsed head count, not the decode
row count — this is the specific GEMM the code comment cites as the source of
the ~5% figure.

---

## Q2 — Is `EXO_DSV4_VERIFY_ROWSEQ_VEC` live at production depth (89K, `VERIFY_BATCH=1`)?

### Correction to task framing

`EXO_DSV4_VERIFY_ROWSEQ_VEC` **does not default to 1 in the Python source** —
`mlx_lm/models/deepseek_v4.py:5646`: `_VERIFY_ROWSEQ_VEC = os.environ.get("EXO_DSV4_VERIFY_ROWSEQ_VEC", "0") == "1"`.
The value 1 is the **launcher's** default: `start_cluster.sh:1953`:
`: "${EXO_DSV4_VERIFY_ROWSEQ_VEC:=1}"`. In production (via `start_cluster.sh`)
it is effectively 1 unless overridden; the code-level default is 0. Likewise
`EXO_DSV4_VERIFY_ROWSEQ` defaults to 0 in code (`deepseek_v4.py:1602`) but 1 in
the launcher (`start_cluster.sh:331`), and `EXO_DSV4_VERIFY_BATCH` defaults to
0 in code (`deepseek_v4.py:1695`) but 1 in the launcher (`start_cluster.sh:344`,
"default ON, 2026-08-27 PROMOTED to production"), with
`EXO_DSV4_VERIFY_BATCH_MIN_CTX` defaulting to 8192 in **both** code
(`deepseek_v4.py:1708`) and launcher (`start_cluster.sh:350`).

### The dispatch chain

`_forward_steps` computes the activation gate once per verify forward
(`deepseek_v4.py:7093-7103`):
```
_vb_ctx_len = _rowseq_ctx(cache[0]) if cache is not None else 0
_vb_active = (
    _VERIFY_BATCH
    and h.shape[0] == 1
    and 2 <= h.shape[1] <= _VERIFY_ROWSEQ_MAX_L
    and _vb_ctx_len >= _VERIFY_BATCH_MIN_CTX
)
if _VERIFY_BATCH:
    _set_verify_batch_ctx(active=False)
    if _vb_active:
        _set_verify_batch_ctx(active=True, L=h.shape[1])
```
(`deepseek_v4.py:7094-7103`). `_VERIFY_ROWSEQ_MAX_L` defaults to 8
(`deepseek_v4.py:1610`). At 89,000 tokens of context, `_vb_ctx_len` (≈89000)
is far above `_VERIFY_BATCH_MIN_CTX` (8192), and a γ=3 verify forward has
`h.shape[1] == 4` (1 anchor + 3 draft, within `[2, 8]`) with `h.shape[0] == 1`
(single stream) — so `_vb_active` evaluates **True**, and
`_VERIFY_BATCH_CTX["active"]` is set True for the duration of that forward
(cleared again at `deepseek_v4.py:7134-7135` after the per-layer loop).

Every rowseq-family branch — including the ROWSEQ_VEC call site — is gated by
the identical pattern `_VERIFY_ROWSEQ and not (_VERIFY_BATCH and _VERIFY_BATCH_CTX["active"])`:

- Attention rowseq / rowseq_vec dispatch, `deepseek_v4.py:5440-5443`:
  ```
  if (
      _VERIFY_ROWSEQ
      and not (_VERIFY_BATCH and _VERIFY_BATCH_CTX["active"])
      and 2 <= normed.shape[1] <= _VERIFY_ROWSEQ_MAX_L
      and normed.shape[0] * normed.shape[1] <= 8
      and (...)
  ):
      if (getattr(self.attn, "rowseq_vec_supported", None) is not None
              and self.attn.rowseq_vec_supported(cache)):
          x = self.attn.rowseq_vec(normed, cache)   # <-- ROWSEQ_VEC call site (5468)
      else:
          x = mx.concatenate([...])                  # per-row loop fallback
  else:
      x = self.attn(normed, mask=mask, cache=cache)  # batched path
  ```
  (`deepseek_v4.py:5440-5494`, ROWSEQ_VEC dispatch specifically at
  `5460-5468`.)
- FULLBLOCK per-row loop, `deepseek_v4.py:5292-5294`: same
  `_VERIFY_ROWSEQ and _VERIFY_ROWSEQ_FULLBLOCK and not (_VERIFY_BATCH and _VERIFY_BATCH_CTX["active"])` guard.

**At 89K with `VERIFY_BATCH=1`: when `_vb_active` is True (which it is at this
depth for any in-range verify call), the `not (_VERIFY_BATCH and _VERIFY_BATCH_CTX["active"])`
term is `not (True and True) == False`, so the entire `if` condition is False
regardless of `_VERIFY_ROWSEQ`'s value — the block falls to the `else` branch
(`deepseek_v4.py:5493-5494`), which is the plain batched `self.attn(...)` call.
`self.attn.rowseq_vec(...)` (line 5468) is never reached. ROWSEQ_VEC is DEAD
at 89K under this config** — not merely unused-by-preference, but structurally
unreachable: the boolean short-circuits before `rowseq_vec_supported()` is
even evaluated.

### Does ROWSEQ_VEC execute below 8192 ctx, or on the draft path?

**Below 8192 ctx:** `_vb_ctx_len < _VERIFY_BATCH_MIN_CTX` ⇒ `_vb_active` is
False in `_forward_steps` (`deepseek_v4.py:7098`) ⇒ `_VERIFY_BATCH_CTX["active"]`
stays False for that forward ⇒ the `not (_VERIFY_BATCH and _VERIFY_BATCH_CTX["active"])`
term is `not (True and False) == True` ⇒ the outer `_VERIFY_ROWSEQ and True and (shape checks)`
condition can be satisfied ⇒ if `_VERIFY_ROWSEQ_VEC` (launcher default 1) is
set and `rowseq_vec_supported(cache)` returns True (`deepseek_v4.py:6087-6088`:
`return _VERIFY_ROWSEQ_VEC and _rowseq_vec_ring_ok(cache)`, and the
`CompressedAttention`/`SparseCompressedAttention` variants at
`deepseek_v4.py:6175-6176` / `6325-6326` with the same `_VERIFY_ROWSEQ_VEC and isinstance(cache, CacheList)`
form), `self.attn.rowseq_vec(normed, cache)` **does execute** below 8192 ctx.
This is the live case the flag exists for.

**Draft path:** the gated block above is inside `DeepseekV4Block.__call__`
and fires whenever that forward is a "verify-shaped" call (`h.shape[0]==1`
single stream, `2 <= h.shape[1] <= 8` multi-row) — this shape condition does
not itself distinguish "verify forward" from any other multi-row forward the
model executes with the same shape (e.g. a chained-MTP draft forward that
also submits multiple candidate rows in one call). The gate is driven purely
by tensor shape and the `_VERIFY_BATCH_CTX["active"]` side channel set in
`_forward_steps`, both of which are set once per call regardless of whether
the caller is drafting or verifying — `deepseek_v4.py:5443-5452`'s shape/ctx
checks make no reference to a draft-vs-verify distinguishing flag. **A true
single-row (M=1) draft step does not qualify** (`normed.shape[1]` would be 1,
failing the `2 <= ... <= 8` lower bound at line 5443/5451), so **M=1 draft
calls never reach this branch at all** (draft and verify are architecturally
distinct call shapes — γ=3 draft is issued as sequential M=1 steps per the
`exo-speculative-decode-correctness` skill's MTP/DSpark description, not a
multi-row draft batch). Any *batched* multi-row call in the 2-8 row range
(which in this codebase is the verify call, not draft) is eligible on shape
alone; no other call site of this specific gated block exists in the file
(`deepseek_v4.py`'s only occurrences of this exact conditional pattern are
the two cited above, both inside `DeepseekV4Block.__call__`/its FULLBLOCK
sibling).

### Guard quotes (verbatim, for the record)

- `deepseek_v4.py:7094-7098`:
  ```
  _vb_active = (
      _VERIFY_BATCH
      and h.shape[0] == 1
      and 2 <= h.shape[1] <= _VERIFY_ROWSEQ_MAX_L
      and _vb_ctx_len >= _VERIFY_BATCH_MIN_CTX
  )
  ```
- `deepseek_v4.py:5440-5443`:
  ```
  if (
      _VERIFY_ROWSEQ
      and not (_VERIFY_BATCH and _VERIFY_BATCH_CTX["active"])
      and 2 <= normed.shape[1] <= _VERIFY_ROWSEQ_MAX_L
  ```
- `deepseek_v4.py:6087-6088`:
  ```
  def rowseq_vec_supported(self, cache: Any) -> bool:
      return _VERIFY_ROWSEQ_VEC and _rowseq_vec_ring_ok(cache)
  ```

---

## Verdicts

1. **STEEL-BI affects the c=1 decode path at M=1/M=4: YES** because the
   attention output (P·V) matmul collapses per-head batch into M
   (`matmul.cpp:900-923,1488-1522`), so it hits the flag's split-K gates
   (`matmul.cpp:963-964,987-989`) regardless of decode-row-count — on both
   M=1 draft and M=4 verify steps; and separately, ordinary large-N linear
   GEMMs at M=4 (verify) additionally hit the flag's tile-selection macro
   (`matmul.cpp:144-146`), while the same GEMMs at M=1 (draft) are routed to
   `gemv` (`matmul.cpp:1554`) and bypass the flag entirely for that class of
   op. The documented "~5% c=1 decode cost" is real and live in the current
   code, not stale, because it is anchored to the always-present attention
   output matmul, not to a steel-path-only GEMM that M=1 would skip.

2. **ROWSEQ_VEC at 89K with VERIFY_BATCH=1: DEAD** because
   `_vb_active` evaluates True at that depth (`deepseek_v4.py:7094-7098`,
   89000 ≥ `_VERIFY_BATCH_MIN_CTX`=8192), which sets
   `_VERIFY_BATCH_CTX["active"]=True`, making the guard
   `not (_VERIFY_BATCH and _VERIFY_BATCH_CTX["active"])` at
   `deepseek_v4.py:5442` evaluate False and short-circuit the whole
   ROWSEQ/ROWSEQ_VEC branch before `self.attn.rowseq_vec(...)`
   (`deepseek_v4.py:5468`) is ever reached — the call falls through to the
   plain batched `self.attn(normed, mask=mask, cache=cache)` at
   `deepseek_v4.py:5494` instead. ROWSEQ_VEC remains LIVE below 8192 ctx
   (where `_vb_active` is always False) but not on any true M=1 draft call
   (shape guard `2 <= normed.shape[1]` at `deepseek_v4.py:5443` excludes
   single-row calls).

3. **The steel-BI=0 test CANNOT be run alone for a general depth sweep — it
   depends on which depth is under test.** At/above `EXO_DSV4_VERIFY_BATCH_MIN_CTX`
   (8192, production default per `start_cluster.sh:350`), ROWSEQ_VEC is dead
   per verdict 2, so the "REQUIRES `MLX_STEEL_BATCH_INVARIANT=1` for per-row
   bitexactness" docstring constraint (`deepseek_v4.py:6011-6023` per the task
   description) is moot there and `MLX_STEEL_BATCH_INVARIANT=0` can be tested
   alone at ≥8192-token depth. **Below 8192 ctx, ROWSEQ_VEC is live** (verdict
   2), so the constraint still binds there — any bit-equivalence gate that
   includes a short/sub-8192 prompt in its test matrix must also flip
   `EXO_DSV4_VERIFY_ROWSEQ_VEC=0` when testing `MLX_STEEL_BATCH_INVARIANT=0`,
   per the launcher's own paired-flag comment (`start_cluster.sh:259`). Also
   note (independent of depth): the pairing itself remains
   **comment-only, not code-enforced** — no runtime assertion anywhere ties
   `MLX_STEEL_BATCH_INVARIANT` to `EXO_DSV4_VERIFY_ROWSEQ_VEC`'s value, so an
   operator flipping one without the other below 8192 ctx gets no warning,
   confirming the round-1 audit's "Interaction constraint verification"
   finding still holds.
