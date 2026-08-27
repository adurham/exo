# Phase 0 addendum — G0 GATE FAILED: verify-batch Indexer pmask/scores shape mismatch (REVERT)

**Date:** 2026-08-27
**Author:** PM agent (GLM-5.2 via Ollama Cloud), verify-path batching campaign
**Status:** G0 FAIL → REVERT. `EXO_DSV4_VERIFY_BATCH` stays default OFF. Cluster restored to production spec-off.
**Super HEAD:** `a1ba8c27e18bed8d26a430325357851b5cf29492`
**Submodule HEAD:** `93afab74a27f40ec747663833407b215de653366`
**Companion:** `docs/verify-batch-phase0-2026-08-26.md` (the Phase 0 design doc this supersedes at the gates)

## TL;DR (plain language first)

The indexer-stream-sharing implementation (`EXO_DSV4_VERIFY_BATCH=1`,
submodule `93afab7`) crashes on the **first verify cycle** with a
deterministic shape-mismatch in the sparse-attention Indexer. The
Phase 0 design doc's core assumption — "the verify-time pmask is None"
(`deepseek_v4.py:3870-3880`, `docs/verify-batch-phase0-2026-08-26.md`
lines 3870-3880) — is **wrong**: `_dispatch_pmask` returns a non-None,
3D pmask sized for the full verify width, which does not broadcast
against the per-row scores tensor. The crash is 100% reproducible on
both nodes (4 crashes each), in the very first warmup request. The
rowseq baseline (`EXO_DSV4_VERIFY_BATCH=0`, identical env otherwise) is
clean — 0 errors, 0 crashes, 256-token probe returns coherent output.
Because the verify-batch path cannot produce a single cycle of output,
G0 (cycle-level bitwise diff vs rowseq) cannot be run: there is no
VERIFY_BATCH=1 trace to diff. **Verdict: REVERT** — keep
`EXO_DSV4_VERIFY_BATCH=0` (the shipped default). Downstream gates
(G2, G3, sanity A/B, 24-run verdict) are moot: the path crashes before
they can execute.

## The crash (exact)

```
ValueError: [broadcast_shapes] Shapes (1,1,3) and (1,1,2) cannot be broadcast.
```

**Call chain** (both nodes, identical):
- `dsv4_mtp.py:2299` `_next` → `:3893` `_speculative_next`
- → `dsv4_mtp.py:1420` `dsv4_speculative_forward` → `model(inputs, cache)`
- → `deepseek_v4.py:7055` `__call__` → `:7038` → `:6875` `_forward_steps`
- → `:6856` `_set_verify_batch_ctx(active=True, L=h.shape[1])` (activates
  the side channel for `2 <= L <= _VERIFY_ROWSEQ_MAX_L`)
- → `:6875` `layer(h, mask, layer_cache, inputs)` → `:5108` `__call__`
- → `:4706` `attn(...)` → `:4706` `self.indexer(...)`
- → `:4008` `Indexer.__call__`:
  ```python
  scores = mx.where(
      pmask if pmask.ndim == 3 else pmask[None],  # pmask = (1,1,3)
      scores,                                      # scores = (1,1,2)
      mx.finfo(scores.dtype).min,
  )
  ```
  → `ValueError: [broadcast_shapes] Shapes (1,1,3) and (1,1,2) cannot be broadcast.`

**Crash count:** 4 crashes per node (8 total), all on the first warmup
request (`"Say hi"`, max_tokens=5). The runner supervisor restarts the
worker child after each crash; the parent (`python -m exo`) survives.

## Root cause

The Phase 0 design doc (`docs/verify-batch-phase0-2026-08-26.md`,
Indexer section) and the in-code comment
(`deepseek_v4.py:3870-3880`) state the bit-exactness guarantee rests
on: *"The verify-time pmask is None (PoolingCache.make_mask returns None
for L<=_POOL_VERIFY_MAX_L), so the length check — not the pmask — is
what guarantees bit-exactness."*

This assumption is **violated** in the shipped implementation:

1. `_VERIFY_BATCH_CTX["active"]=True` activates the indexer-stream-sharing
   at `deepseek_v4.py:3894` (snapshot `pooled` on row 0, reuse for rows
   1..L-1).
2. `pmask = _dispatch_pmask(pool_cache, L_full, offset)` at `:3950`
   returns a **non-None, 3D pmask** shaped `(1, 1, L_full)` — NOT None.
3. The `_tail_ok` fast-path guard (`:3960`) requires `pmask.ndim == 2`
   to apply the tail-restricted band optimization; with a 3D pmask it
   falls through to the `else` branch at `:4007`.
4. The `else` branch broadcasts `pmask` `(1,1,3)` against `scores`
   `(1,1,2)` → the last axes (3 vs 2) mismatch → crash.

The `scores` tensor has last-axis 2 (not the pool length P) because the
stream-sharing path computes scores per-row (L=1 after the row band
slice), but the `pmask` is built for the full `L_full=3` verify width and
never sliced to the per-row band in the non-`_tail_ok` branch.

**Net:** the indexer-stream-sharing code reuses row 0's `pooled` snapshot
correctly, but the companion `pmask` handling was written assuming
pmask is None at verify time. When pmask is non-None (which it is in this
config), the shape contract breaks.

## G0 result

| side | config | result |
|---|---|---|
| VERIFY_BATCH=1 | `EXO_DSV4_VERIFY_BATCH=1 MTP_PROFILE=50 SPEC_TRACE=1` + full spec-ON (native DSpark head, gamma=3, rowseq, FULLBLOCK) | **CRASH** — 0 cycles produced, deterministic `broadcast_shapes` on first verify |
| VERIFY_BATCH=0 (rowseq baseline) | identical env minus the 3 verify-batch flags | **CLEAN** — 0 errors, 0 crashes, 256-token probe coherent (`completion_tokens=256, finish_reason=length`) |

G0 expects 0-ulp between the two sides. VERIFY_BATCH=1 cannot produce a
trace, so there is nothing to diff. **G0 = FAIL (nonzero-ulp by
construction: one side crashes).**

## REAL C_s

**Not obtainable.** `EXO_DSV4_MTP_PROFILE=50` brackets the draft/verify
phases with `[MTP-PROF]` log lines, but the verify-batch path crashes
before the first `[MTP-PROF]` line emits (the crash is inside the verify
forward, before the phase-timer dump). The rowseq baseline
(`VERIFY_BATCH=0`) was launched WITHOUT `MTP_PROFILE` (the G0-OFF side
doesn't need it), so its logs have no `[MTP-PROF]` lines either. The
documented rowseq-baseline C_s=3.20 (from the prior 2026-08-26 campaign,
`docs/dspark-cs-profile-2026-08-26.md`) remains the relevant number:
at C_s=3.20 the +10% PROMOTE bar is arithmetically unreachable
(break-even a*=2.199 vs measured a≈2.256). The verify-batch path was the
proposed fix to bring C_s down to ~1.3; since it crashes, C_s stays at
3.20 and the verdict arithmetic is unchanged.

## Downstream gates (moot)

G2 (Tier-1 7-prompt byte-identity), G3 (10K soak), the 4-run sanity A/B,
and the 24-run verdict protocol all require a functioning VERIFY_BATCH=1
path. Since the path crashes on the first cycle, none can execute. They
are cancelled, not skipped-for-time.

## Env-audit (production final state)

`ps eww` on both nodes (post-REVERT, production spec-off):

```
EXO_SPECULATIVE=0
EXO_DSV4_MTP=0
EXO_DSV4_DSPARK=1
EXO_DSV4_DSPARK_NATIVE=1
EXO_DSV4_DSPARK_FORCE_LOAD=1
(EXO_DSV4_VERIFY_BATCH absent = default OFF)
```

DSpark head resident (FORCE_LOAD=1), not drafting (SPECULATIVE=0,
MTP=0) — the documented production state. Warmup `"Say hello"` returns
`"Hello!"` cleanly (`completion_tokens=20`). The 4 `Traceback` lines in
each fresh `~/exo.log` are the pre-existing, non-fatal ModelCard
validation warnings (`failed to validate model card at .../*.toml` —
pydantic missing-field in model-card TOMLs), not crashes.

## Sync + launch method

`start_cluster.sh` is blocked by expired sudo on both studios
(`sudo -n true` → "a password is required"). Manual per-node screen
launch (the proven 18:34-19:46 CDT pattern):

1. SIGTERM old runners (SIGTERM only, 15s RDMA drain, never kill -9).
2. Per node: `cd ~/repos/exo && git fetch origin && git reset --hard a1ba8c27e`, then `cd mlx-lm && git fetch origin && git reset --hard 93afab7`.
3. Reinstall mlx-lm (copy install, not editable): `zsh -l -c 'cd ~/repos/exo && uv pip install --no-deps --force-reinstall ./mlx-lm'` on each node.
4. Verify: `grep -c EXO_DSV4_VERIFY_BATCH .venv/lib/python3.13/site-packages/mlx_lm/models/deepseek_v4.py` = 8 on both nodes.
5. Build launch files from `/tmp/ab/tier1/node{1,2}_specon_launch.txt` (inject `EXO_DSV4_VERIFY_BATCH=1 EXO_DSV4_MTP_PROFILE=50 EXO_DSV4_SPEC_TRACE=1` before `EXO_DSV4_SPEC_EOS_BAN=0`).
6. scp to each node, `screen -dmS exorun_verbon bash -c 'bash /tmp/verbon_launch.sh'`.
7. For the REVERT: same with the specoff templates (`EXO_SPECULATIVE=0 EXO_DSV4_MTP=0`), screen `exorun_specoff`.

## Next-step direction (not executed this session)

The fix is in the Indexer `pmask` handling on the verify-batch path
(`deepseek_v4.py:3894-4010`): when `_VERIFY_BATCH_CTX["active"]` and
`pmask is not None`, the pmask must be sliced to the per-row band
(matching the `seq_band` slice at `:3948`) OR the `_tail_ok` fast-path
must handle the 3D-pmask case the stream-sharing produces. The Phase 0
doc's claim that verify-time pmask is None needs re-verification against
`_dispatch_pmask` / `PoolingCache.make_mask` for the actual L values the
verify path passes (the crash shows L_full=3, not the L<=MAX_L the
None-pmask branch was gated on). This is a code fix in the submodule
mlx-lm fork, scoped to the `EXO_DSV4_VERIFY_BATCH` path, with no change
to the rowseq baseline.