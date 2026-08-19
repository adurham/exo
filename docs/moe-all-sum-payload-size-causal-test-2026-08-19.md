P1: moe.all_sum payload-size causal test — instrumentation + run plan (2026-08-19)
==================================================================================

Question
--------

`moe.all_sum` is now confirmed to be ~61-64% of TP prefill wall time
(`docs/moe-all-sum-dominant-cost-2026-08-19.md`, NOP ablation at 12K and
38K). That measured the cost of the collective **existing at all**. It
does NOT say *where* the cost lives. Two hypotheses with opposite fixes:

- **Bandwidth-bound** — cost scales with bytes moved over the TB5/RDMA
  link. Fix direction: shrink the payload (dtype, sharded/reduce-scatter
  form, compression). Ceiling set by the link.
- **Overhead-bound** — cost is a per-call fixed price (fence/sync,
  graph-eval boundary, QP/handshake, `mx.eval` serialization at the call
  site) that barely moves with size. Fix direction: cut the CALL COUNT
  (43/forward) or overlap it — shrinking bytes would buy nothing.

Discriminating experiment
-------------------------

Hold the call site, the call count (43/forward), and everything else
fixed; change ONLY the number of bytes each call reduces. Sweep the
payload fraction `f` and read prefill tok/s:

| f | expected if bandwidth-bound | expected if overhead-bound |
|---|---|---|
| 1.00 | control | control |
| 0.50 | ~half the all_sum cost recovered | ~no change |
| 0.25 | ~3/4 recovered | ~no change |
| 0.00 | full recovery (== existing `all_sum` NOP result) | full recovery |

The `f=0.0` arm is the anchor: it must land at/near the already-measured
NOP numbers (427-430 tok/s). If it does not, the harness is wrong and
the whole sweep is invalid — treat that as INCONCLUSIVE, not as a result.

Instrumentation landed
----------------------

`mlx-lm/mlx_lm/models/deepseek_v4.py`, riding the existing
`/tmp/dsv4_nop_targets` file-toggle infra (1s cache, no relaunch needed
to change `f` between runs):

- `_get_allsum_fraction()` — parses a `all_sum_frac=<f>` token out of the
  NOP-targets file; returns `None` when absent.
- `_all_sum_fractional(y, group, frac)` — reduces only the first
  `ceil(f*L)` **sequence rows** of `y` `(B, L, H)` and passes the rest
  through unreduced, via one slice + one `all_sum` + one `concatenate`.
- Wired at the single `moe.all_sum` span in `DeepseekV4MoE.__call__`.
  Absent token ⇒ the untouched production call, byte-identical.

Design notes / why this shape:

- Slicing the SEQUENCE axis (not hidden) keeps the reduced sub-tensor
  contiguous, so the collective sees one dense buffer of exactly `f`×
  the bytes — no gather/scatter cost polluting the measurement.
- Call count is unchanged at every `f > 0` (still one collective per MoE
  layer). Only bytes change. That is the whole point — it isolates
  payload from per-call overhead, which the existing binary `all_sum`
  NOP target cannot do (it changes both at once).
- `f = 1.0` still routes through `_all_sum_fractional` (which then takes
  its `_n >= L` fast path and issues the identical full `all_sum`), so
  the f=1.0 arm is a fair control for the sliced arms.
- **Output is GARBAGE at any `f < 1.0`** — unreduced rows carry only the
  local rank's partial MoE result. Same contract as the existing
  `all_sum` NOP target: bench tok/s only, decode will produce nonsense
  and may crash. NEVER leave the token set.

Verification done locally (pre-deploy)
--------------------------------------

- `ast.parse` clean.
- `ruff check --statistics` on the edited file: **161 errors, exactly
  matching the `git show HEAD:` baseline of the same file** (this file's
  pre-existing baseline is nonzero; the exit criterion is match, not 0).
  One transient `N806` introduced by a first draft was fixed.
- Slice/ceil/concat logic exercised standalone under real `mlx.core`
  with a stand-in reduce (`x -> 2x`): shapes preserved at every `f`,
  row counts `ceil(f*L)` correct for `f ∈ {1.0, 0.5, 0.25, 0.0}`, and
  `f=1.0` bit-equal to the unsliced reduce.

NOT deployed. Landing this on the cluster needs a `git push` + per-node
`git reset --hard` + `start_cluster.sh` restart, which requires its own
explicit go-ahead (and the mlx-lm submodule pin bump — the nodes install
mlx-lm from the submodule at launch, so a submodule-only commit that
isn't pinned will silently not take effect).

Run plan (after deploy go-ahead)
--------------------------------

Standing config (`EXO_PREFILL_STEP_SIZE=2048`, `SEQ_SPLIT=1`), 2 nodes.

1. Confirm cluster healthy; confirm `/tmp/dsv4_nop_targets` is EMPTY on
   BOTH nodes.
2. Warm the cluster (throwaway request), then take a **warm** baseline at
   12K with the token absent — this is the production reference.
3. For `f` in `1.0, 0.5, 0.25, 0.0`: write `all_sum_frac=<f>` to
   `/tmp/dsv4_nop_targets` on **BOTH** nodes (per-process file, 1s
   cache), issue a **fresh** prompt (unique secret code) at the same
   token depth, and record tok/s. Verify `cached_tokens: 0` in the
   response every time — a KV-cache hit silently invalidates the arm.
4. Clear the file on both nodes immediately after the last arm.
5. Sanity gates before believing anything:
   - `f=1.0` must land within noise of the warm production baseline
     (proves the slice wrapper itself is free).
   - `f=0.0` must land near the known NOP numbers (~427-430 tok/s at
     12K) (proves the sweep spans the full known range).
   - If either gate fails: INCONCLUSIVE, fix the harness, do not report
     a bandwidth-vs-overhead verdict.
6. Repeat at 38K to confirm the verdict isn't depth-specific (the
   original NOP result held to within 1pp across 12K/38K).

Reading the result
------------------

Convert each arm to an all_sum time share:
`t_allsum(f) = 1/tok_s(f) - 1/tok_s(f=0)` (per-token wall attributable
to the collective). Then:

- `t_allsum(0.5) ≈ 0.5 · t_allsum(1.0)` ⇒ bandwidth-bound; the fix
  space is payload reduction, and the TB5 link is the ceiling.
- `t_allsum(0.5) ≈ t_allsum(1.0)` ⇒ overhead-bound; payload work is
  wasted effort, attack call count / fencing / overlap instead.
- Anything in between ⇒ mixed; report the fitted intercept (fixed
  per-call cost) and slope (per-byte cost) rather than picking a side.

Do not skip step 5. The prior session on this exact question produced a
false-positive from an unnoticed KV-cache hit.
