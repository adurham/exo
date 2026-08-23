# P5: hc_expand bf16-comb-cast fix — rejection RE-LITIGATED and UPHELD (multi-seed) — 2026-08-22

## Question

`docs/t10-final-decomposition-closed-2026-08-22.md` Check 1 rejected a
1.41x-speedup candidate for `_hc_expand_op` (cast tiny `comb` to bf16
instead of upcasting `residual` to fp32) on a SINGLE precision check
(~1.08% mean relative error, seed 0 only). Re-run the check 5+ times
with different seeds; if the error is real and stable, test
fp32-internal-accumulation variants for a better tradeoff; ship only
if a variant clears <0.2% mean relative error AND still wins on speed.

## Method

Standalone microbench, `/tmp/p5_hc_expand_multiseed.py`, run with the
cluster's own venv (`.venv/bin/python`), production shape
B=1, L=2048, HC=4, D=4096, realistic mixed dtypes (bf16 x/residual,
fp32 post/comb), 7 seeds (0-6), all variants `@mx.compile`d, timing
with warmup + `mx.eval` + `mx.synchronize` per repo methodology.

Variants:
- `orig` — current production op (`residual.astype(fp32)` + fp32 matmul)
- `cand_bf16` — the rejected candidate (comb→bf16, bf16 matmul)
- `var_fp32acc` — comb stays fp32; the tiny K=4 contraction done as
  broadcast-multiply + `sum(axis)` so accumulation happens in fp32 with
  no standalone fp32 residual materialization
- `var_bf16comb_fp32acc` — comb→bf16 but fp32 accumulation

## Real results

Precision vs `orig` (fp32 reference path), across all 7 seeds:

| variant | mean rel err (range across seeds) | max abs diff | speed (µs) |
|---|---|---|---|
| orig | — | — | 3103.8 |
| cand_bf16 (rejected candidate) | 1.083%–1.136% | 0.125 every seed | 2113.6 (1.47x faster) |
| var_fp32acc | ~0.000% (all seeds) | ≤0.0625 | 5017.9 (1.6x SLOWER) |
| var_bf16comb_fp32acc | 1.301%–1.373% | 0.125 | 5057.5 (1.6x SLOWER) |

(The huge max_rel values on the bf16 variants are near-zero-denominator
artifacts; max_abs 0.125 is the meaningful figure and matches the
original rejection's number exactly.)

## Findings

1. **The original rejection was sound.** The ~1.08% mean relative
   error is not a seed-0 fluke — it is stable at 1.08–1.14% across 7
   independent seeds with an identical 0.125 max-abs diff every time.
   The error comes from quantizing `comb` itself to bf16 (a genuine
   information loss on the combination weights), not from unlucky
   inputs. Confirms `var_bf16comb_fp32acc`: keeping fp32 accumulation
   but bf16 comb is WORSE (1.30–1.37%) — the dominant error source is
   comb's bf16 rounding, so no accumulation trick can rescue the
   bf16-comb approach.
2. **The only variant that clears the <0.2% precision gate
   (`var_fp32acc`, ~0.000% mean rel err) is 1.6x SLOWER than the
   current production op** (5018µs vs 3104µs) — the broadcast-sum
   pattern materializes a (B,L,HC,HC,D) intermediate, costing more
   bandwidth than the fp32-residual upcast it avoids. Nothing to ship.

## Outcome

**(b)/(c): rejection upheld with multi-seed evidence; no shippable
variant exists in this design space.** No code change, no relaunch.
The bf16-comb 1.41x-class speedup is fundamentally coupled to a real,
stable ~1.1% numeric divergence on the residual stream (compounding
across 43 layers) and remains unshippable without a full quality
validation that the ~1.3-point prefill payoff does not justify.
P5 CLOSED.
