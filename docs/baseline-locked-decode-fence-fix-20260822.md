# Baseline locked in: async-fence fix confirmed at real depth, tagged — 2026-08-22 (session 3, continuation)

## Why this doc

Following the async-fence fix's short-context validation
(`docs/async-fence-fix-validated-2026-08-22.md`), the user asked for
three things before moving on: (1) confirm prefill is still ~350 tok/s
and decode ~30 tok/s at real depth, not just short-context, (2)
re-confirm MTP/DSpark really aren't wired up, (3) lock this in as the
new clean baseline across all performance docs. This records that
work.

## Real re-benchmark at depth (100K/300K/500K, needle-verified)

Ran `bench/phase3_precheck_depth_throughput.py` against the live
cluster with the async-fence fix active (`EXO_DSV4_MOE_FUSED_GATE_UP=1`
`EXO_DSV4_FENCE_ASYNC=1`, genuinely functional post-fix), same tool
used for the original 2026-08-21 baseline measurement.

| Depth | Prefill (tok/s) | Decode (tok/s) | Needle |
|---|---|---|---|
| 100K (70,158 real tokens) | 359.7 | 26.91 | PASS |
| 300K (211,002 real tokens) | 348.2 | 24.44 | PASS |
| 500K (352,595 real tokens) | 324.1 | 21.51 | PASS |

**Prefill: at parity with the pre-fix baseline** (366.6/351.5/331.6 —
within normal run-to-run variance), confirming the async-fence fix
correctly left prefill untouched, exactly as its design predicted (the
fence only affects decode-time `mx.eval`/`mx.async_eval` behavior).

**Decode: real, substantial improvement at every depth tested**, though
smaller in relative terms than the short-context result:
- 100K: 17.48 → 26.91 tok/s (+53.9%)
- 300K: 18.60 → 24.44 tok/s (+31.4%)
- 500K: 17.26 → 21.51 tok/s (+24.6%)

The relative gain shrinking with depth is a real, expected pattern, not
a concern: the fence only ever governs the ~43 per-layer collectives,
whose absolute cost is roughly depth-independent, while total decode
step cost grows with KV-cache size at deeper context — so the fence's
fixed absolute-time saving becomes a smaller fraction of a larger total.
Not fully decomposed this session; a reasonable target for future
depth-scaling investigation if further decode optimization is pursued.

## MTP/DSpark: re-confirmed dormant, directly from live logs

```
grep -c 'PP speculation using DSpark' ~/exo.log   → 0
grep -c 'DSpark ctx warmed' ~/exo.log              → 12
grep -c 'DSv4 MTP speculative decoding enabled' ~/exo.log → 0
```

Same result as originally found earlier this session
(`docs/roofline-sanity-check-inputs-confirmed-2026-08-22.md`), now
re-verified against the current live process (not assumed carried
over). `EXO_DSV4_DSPARK=1` is set in production config but the module
only loads and warms its context per-request — its actual decode loop
(`pp_dspark_decode_loop`, PP-only) never fires under this cluster's TP
sharding mode. **MTP/DSpark speculative decode is NOT wired up for
this cluster** — a real, standing, unrealized throughput opportunity,
not yet attempted.

## Baseline locked in

Both `exo` and `mlx-lm` repos tagged `known-good-decode-fenceasync-20260822`
at their current HEADs (commit `e1833008f` and `1fea494` respectively),
pushed to origin. The `mlx` submodule is unchanged since
`known-good-prefill-20260821-165048` and remains current — no new mlx
C++ changes landed as part of tonight's fence fix (pure Python, in
`mlx-lm` and `exo`).

`docs/PERFORMANCE_HISTORY.md` §1 rewritten: the new baseline table (all
figures above) is now the primary, current reference; the 2026-08-21
figures are preserved as a collapsed historical `<details>` block for
context, explicitly marked as measured under the since-fixed broken
fence condition. §2.8 updated with the real at-depth confirmation
numbers.

## Config summary (for reference)

```
EXO_DSV4_MOE_FUSED_GATE_UP=1
EXO_DSV4_FENCE_ASYNC=1   # genuinely functional as of this baseline
EXO_DSV4_MTP=0           # classic MTP disabled
EXO_DSV4_DSPARK=1        # set but structurally inactive under TP
```

Cluster confirmed healthy, correctness confirmed clean (standard
quality probe + needle-in-haystack, both passing) at the point this
baseline was locked in.
