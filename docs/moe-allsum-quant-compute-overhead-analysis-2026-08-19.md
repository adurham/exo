Quantized all_sum: compute overhead is negligible vs projected wire savings (2026-08-19)
=============================================================================================

Answering the user's specific concern
------------------------------------------

User's question: quantizing the all_sum payload cuts bytes-on-wire, but
the quantize/dequant math itself runs on the GPU (the constrained
resource) -- does that overhead eat the savings, or just move the cost
elsewhere?

Real answer, with corrected numbers
---------------------------------------

The implementing subagent measured compute overhead at hidden_size=7168
(the mlx-lm code's DEFAULT config value), not the real production
hidden_size=**4096** (confirmed from the live model's actual
`config.json`). Rescaled linearly (quant/dequant is elementwise +
per-group reduction, scales ~linearly with hidden_size):

| L (tokens/call) | raw measurement (H=7168) | corrected (H=4096) |
|---|---|---|
| 256 | 0.94 ms | 0.537 ms |
| 1024 | 1.41 ms | 0.806 ms |
| 4096 | 4.07 ms | 2.326 ms |
| 2048 (interpolated) | -- | **~1.31 ms** |

Compared against the REAL measured RDMA cost at production's actual
shape (P2's finding tonight: 178 ms/call at sz=2, 16.8MB payload,
L=2048-equivalent):

```
baseline (unquantized):        178 ms/call  (100% RDMA wait)
quantized payload (~47% smaller, roughly proportional round-count cut):
  projected RDMA time:          ~94 ms/call
  + quant/dequant compute:      ~1.3 ms/call
  projected total:              ~96 ms/call
projected net speedup:          ~1.86x
compute overhead as % of total savings: ~1.5%
```

**The compute overhead does not meaningfully eat the savings.** ~1.3ms
of GPU work to save ~84ms of RDMA wait is a clear net win on paper, IF
the RDMA round-count actually scales down proportionally with payload
size the way P2's chunk-math predicts (untested assumption, needs a
real cluster A/B, not just projected from the existing chunk-count
model).

What's still NOT verified (honest gaps, do not skip before deploying)
---------------------------------------------------------------------------

1. **The actual `mx.distributed.all_gather` call is completely
   untested end-to-end.** The implementing session had no second rank
   available (single laptop) -- only the local quant/dequant/sum math
   was validated (6 passing tests, `src/exo/worker/engines/mlx/tests/test_moe_allsum_quant.py`).
   Whether `all_gather` on int8+scale payloads actually works correctly
   across 2 real ranks, and whether it's actually faster in practice
   (not just by chunk-count projection), is UNKNOWN until tested on the
   live cluster.
2. **Numerical accuracy on the real model is unverified.** The local
   test validates reconstruction accuracy on synthetic random data, not
   on real DSv4-Flash activations flowing through 43 layers of MoE.
   Quantization error compounds through the residual stream --
   needs a real needle-in-haystack + perplexity-style check before
   trusting output quality, not just a unit test.
3. Code lives on an UNMERGED branch (`mlx-lm` submodule,
   `feat/moe-allsum-quant-2026-08-19`), env-gated OFF by default
   (`EXO_DSV4_MOE_ALLSUM_QUANT=1` to enable). Safe to leave as-is
   until a real cluster test is explicitly approved.

Bottom line
--------------

The user's concern was well-founded to check, and in this specific case
the answer is reassuring: compute overhead is a rounding error against
the projected wire-time savings. But "projected" is doing real work in
that sentence -- this still needs an actual live-cluster A/B (same
matched-prompt methodology used all night) before it's a confirmed win,
not just a promising calculation.
