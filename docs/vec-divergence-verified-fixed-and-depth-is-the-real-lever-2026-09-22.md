# Vec-vs-loop divergence: VERIFIED FIXED on the current checkpoint; the 35-40 t/s target is depth-limited, not correctness-limited (2026-09-22)

## Task as given

"Chase the divergence" — root-cause why the vec verify path diverged from
the loop path, so the measured 35.4-35.5 t/s (vec+ROWSDPA=1) could ship
losslessly instead of the ~29 t/s loop champion.

## Outcome: the divergence was ALREADY root-caused and fixed — and the fix
## is verified intact on the current checkpoint. But it no longer buys the
## speedup, because the checkpoint and the depth regime both changed.

### 1. The fix is present and correct (verified, not assumed)

The 2026-07-12 campaign's RESOLUTION section (line 113 of
`docs/vec-rowsdpa-campaign-2026-07-12.md` — easy to miss, it is 113 lines
down and contradicts the doc's own earlier "per-row hypothesis is dead"
headline) documents the actual root cause:

> `DeepseekV4ShardingStrategy` replicates attention but sets
> `attn.sharding_group` on Compressed/SparseCompressedAttention when
> `EXO_DSV4_SEQ_SPLIT=1` (prod default) — so the loop's per-row `__call__`
> ends with `elif self.sharding_group is not None: all_sum(out)`, a
> distributed reduction **no vec path ever performed**. Deterministic
> rank-level numeric difference on every compressed/sparse forward; the
> single-rank ldiff harness (sharding_group=None) is structurally blind to
> it — which is why four increments of bitwise-proven mechanics never moved
> the serving gate.

Fixed in mlx-lm `095c98c` + exo `4b9932322` (`_rowsdpa_sharding_allsum`
mirrors the loop tail in all five vec tails).

**Verified present**: `git merge-base --is-ancestor 095c98c HEAD` in the
mlx-lm submodule → YES (pinned at `5c5328b`). exo `4b9932322` present.

**Verified effective** (fresh 3-prompt gate, this session, temp=0, 400-token
generations, vec arm vs loop arm on the same boot pair):

| prompt | vec sha | loop sha | result |
|---|---|---|---|
| p0 | `487cd04ac825f93a` | `487cd04ac825f93a` | IDENTICAL |
| p1 | `a3540b5784a20113` | `a3540b5784a20113` | IDENTICAL |
| p2 | `2330ec3661ba3152` | `2330ec3661ba3152` | IDENTICAL |

**Verdict: LOSSLESS.** The vec path is byte-identical to the loop path on
the current checkpoint. There is no divergence left to chase.

### 2. But the vec speedup did NOT survive the checkpoint change

| arm | mean tok/s |
|---|---|
| vec (prod default) | 27.39 |
| loop (`ROWSEQ_VEC=0`) | 27.33 |
| **delta** | **+0.06 (+0.2%, noise)** |

On the retired text-only `-0731` checkpoint this same path comparison was
29.5 (loop) → 33.7-35.5 (vec), i.e. +14-20%. Production switched to
**Vision-Exp** on 2026-09-09 (Phase 5 port), with the launcher default
corrected 2026-09-12 (`eb05307ec`). Vision-Exp is a different model (46
layers vs 43 text, vision tower, different MoE/attention config), and on it
the vec path is a wash.

So the "28 vs 36" gap I initially treated as a regression is a
**cross-checkpoint comparison**, not a loss. The July numbers are simply not
transferable.

### 3. Where the throughput actually goes: context depth

Measured this session, current config, 400-token generations:

| context depth | tok/s |
|---|---|
| ~20 tok | 27.02 |
| ~605 tok | 23.22 |
| ~1771 tok | 18.99 |
| ~3521 tok | 15.78 |
| ~7021 tok | 12.30 |

**~20% loss per doubling of context.** This is the real decode lever, and it
also explains the campaign's headline: the July "36.1-36.3 t/s short-ctx"
figure was explicitly *short-ctx*. Our 4K probe sits several doublings
deeper. Same config, different operating point.

Generation length matters too (standing ">400 tokens" rule understates it):

| generation length | tok/s |
|---|---|
| 400 tok | 27.08 |
| 800 tok | 33.20 |

### 4. Correction to a prior reading (record hygiene)

`docs/verify-batch-g0-fail-2026-08-27.md` reads as a REVERT of the
batched-verify path (G0 shape-mismatch crash). That doc is **superseded**:
`start_cluster.sh:342-353` shows `EXO_DSV4_VERIFY_BATCH` default-ON,
PROMOTED to production later the same day, with the rationale recorded
inline — the depth-gated batched path is acceptable because base decode is
*already* nondeterministic at depth (MLX Metal dispatch drift, ~0.6-logit
run-to-run), so small drift is inside the base's own envelope (G0'' bar
74.7% <= base-vs-base 99.3%). 24-run paired @100K: **+36.7% median tok/s**
(CI +28.3..+51.0), verify 83.8→60.6ms, C_s 3.20→2.14.

Live env confirms it is active: `VERIFY_BATCH=1`,
`VERIFY_BATCH_MIN_CTX=8192`.

**This is a documentation hazard of exactly the class flagged this session:**
a superseded failure doc sitting next to (and contradicting) the later
promotion, with no inline status marker.

## Conclusion and next step

- The divergence is fixed and verified **lossless**. Nothing to chase there.
- Decode is **depth-limited**: 27 t/s at short ctx falling to 12 t/s at 7K.
- The 35-40 t/s target is therefore reachable only by attacking the
  **context-scaling term**, not the vec/loop question.
- The batched-verify path (+36.7% @100K, already shipped) is the main
  reason depth scaling has improved at all.

Highest-value next work: characterize and reduce the per-cycle cost growth
with depth. Candidate mechanisms are already mapped in
`docs/dspark-14k-cliff-investigation-2026-08-27.md` (Indexer top-k over
compressed KV; branch flip compressed→sparse at the depth threshold;
per-cycle pmask build). None of those were measured under the current
batched-verify regime at intermediate depths — a depth ladder of verify-cycle
timings (not just end-to-end tok/s) would localize it.
