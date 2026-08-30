"""P05 Phase A diagnosis: WHY is live acceptance 0.000 at real depth?

Facts from the live run (knob ON):
  - trivial ctx ("Say OK", 13 tokens): acceptance HEALTHY (1.3-1.6/3)
  - 5K+ probes: acceptance 0.000/3 on EVERY cycle, ~550ms draft, 5 tok/s
  - needle still hits (main model output coherent) — the TARGET logits
    are fine; it's the draft/verify MATCH that broke.

Hypothesis space (ranked):
  H1 M-dependence of the quantized head: draft computes base_logits at
     M=width (3), verify computes the same tokens' logits at M=4 (batched)
     or M=1 rows (rowseq). If quantized-head output differs ACROSS M,
     draft-vs-verify argmax can systematically disagree... but offline
     v2 measured 0.0 divergence at M=1/3/4 on the SAME inputs.
     -> BUT offline I called qmod(x) with x shape (M, D) FLAT. Production
     calls lm_head on 3D (B, L, D) tensors! MLX flattens internally, but
     the (1, 3, D) shape might route differently than (3, D).
  H2 The verify comparison uses lm_head BIASES/absmax quantization noise
     at near-ties: at real depth, hidden states have outlier channels;
     quantization error 0.53 mean >> real margins (p10=0.25) flips MOST
     tokens' argmax. Then the target (verify) and draft DISAGREE because
     BOTH go through the quantized head but with DIFFERENT x (draft's x is
     from the draft head's own stages, verify's x is the real model's) —
     and near-tie tokens flip to DIFFERENT tokens in each. Acceptance
     = P(draft argmax == verify argmax). If per-call argmax flip rate
     vs the BF16 head is ~16%, acceptance at position 0 should be ~84%,
     NOT 0%. -> H2 alone cannot explain 0.000.
  H3 The draft side uses a SEPARATE quantization of the head... no, both
     use model.lm_head.
  H4 The verify path at depth runs BATCHED verify (M=4) through the
     quantized head, whose output differs from M=1 per-row by >0 — the
     0-ulp claim was for the BF16 head under GEMV/STEEL flags. The
     quantized qmm/qmv_wide paths have NO invariance flags. If verify's
     batched M=4 quantized logits differ from draft's M=3 quantized
     logits IN ARGMAX for near-ties... still can't be 100% mismatch.

  H5 SOMETHING ELSE SYSTEMATIC: e.g. the draft passes lm_head as an
     ARGUMENT and calls it inside @mx.compile'd code — a QuantizedLinear
     inside mx.compile might behave differently, or the .scales access
     breaks the compiled graph, producing garbage draft logits (coherent
     target output, 0% match, and 550ms draft = recompilation churn!).

Test: reproduce the draft-vs-verify M-shape divergence offline with the
3D call shapes production actually uses, M-grid {1,3,4}, comparing
per-row argmax AND full logits.
"""
import os
import sys
from pathlib import Path

os.environ["MLX_GEMV_BATCH_INVARIANT"] = "1"
os.environ["MLX_STEEL_BATCH_INVARIANT"] = "1"

import numpy as np

import mlx.core as mx
import mlx.nn as nn

HERE = Path(__file__).parent
W_BF16 = HERE / "head_weight.bf16"
V, D = 129280, 4096


def load_head_bf16():
    raw = np.fromfile(W_BF16, dtype=np.uint16)
    f32 = (raw.astype(np.uint32) << 16).view(np.float32)
    return f32.reshape(V, D)


def main():
    w_np = load_head_bf16()
    lm_q = nn.Linear(D, V, bias=False)
    lm_q.weight = mx.array(w_np).astype(mx.bfloat16)
    qmod = lm_q.to_quantized(group_size=32, bits=8, mode="mxfp8")
    qmod.eval()
    mx.eval(qmod.parameters())

    rng = np.random.default_rng(20260830)
    Hs = rng.standard_normal((16, D)).astype(np.float32)
    Hn = Hs / (np.sqrt((Hs ** 2).mean(-1, keepdims=True)) + 1e-6)
    x = mx.array(Hn).astype(mx.bfloat16)  # (16, D)

    # production call shapes: (B=1, L, D) 3D
    print("3D-shape M-invariance of the QUANTIZED head (B=1, L, D):")
    for L in (1, 3, 4):
        n_rows = (16 // L) * L
        outs = []
        for start in range(0, n_rows, L):
            xb = x[start : start + L].reshape(1, L, D)   # (B, L, D)
            y = qmod(xb)
            mx.eval(y)
            outs.append(y.reshape(L, V))
        y_batch = mx.concatenate(outs, axis=0)
        per = []
        for i in range(n_rows):
            y = qmod(x[i].reshape(1, 1, D))
            mx.eval(y)
            per.append(y.reshape(1, V))
        y_per = mx.concatenate(per, axis=0)
        d = (y_batch - y_per).astype(mx.float32)
        print(f"  L={L}: batched-3D vs per-row-3D max={float(mx.abs(d).max()):.3e} "
              f"argmax_equal={int((mx.argmax(y_batch, -1) == mx.argmax(y_per, -1)).sum())}/{n_rows}")

    # 2D vs 3D of the SAME rows, same M
    print("\n2D (M, D) vs 3D (1, M, D) at same M:")
    for M in (1, 3, 4):
        x2 = x[:M]
        y2 = qmod(x2)
        y3 = qmod(x2.reshape(1, M, D)).reshape(M, V)
        mx.eval(y2, y3)
        d = (y2 - y3).astype(mx.float32)
        print(f"  M={M}: max={float(mx.abs(d).max()):.3e}")

    # the draft-vs-verify pairing: same hidden state computed at M=3 (draft
    # width) vs M=4 (verify width) — do ARGMAX agree?
    print("\ndraft-M=3 vs verify-M=4 pairing (same row):")
    y3 = qmod(x[:3])          # 3 rows at M=3
    y4 = qmod(x[:4])          # 4 rows at M=4
    mx.eval(y3, y4)
    a3 = mx.argmax(y3, -1)
    a4 = mx.argmax(y4[:3], -1)
    print(f"  argmax M=3 vs M=4 agree: {int((a3 == a4).sum())}/3")

    # CRITICAL TEST H5: QuantizedLinear inside @mx.compile
    print("\nH5: quantized head inside @mx.compile (draft() is compiled?):")

    @mx.compile
    def compiled_call(h):
        return qmod(h)

    try:
        yc = compiled_call(x[:4].reshape(1, 4, D))
        mx.eval(yc)
        ye = qmod(x[:4].reshape(1, 4, D))
        mx.eval(ye)
        d = (yc.reshape(4, V) - ye.reshape(4, V)).astype(mx.float32)
        print(f"  compiled vs eager max={float(mx.abs(d).max()):.3e}")
    except Exception as e:
        print(f"  compiled call FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()