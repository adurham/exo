"""P05 Phase A (step 1): offline lm_head mxfp8 quantization numerics.

Real `head.weight` (BF16 [129280, 4096], 1.059 GB, md5-verified copy from the
production checkpoint shard 45) + the REAL mlx quantization machinery
(`nn.Linear.to_quantized(group_size=32, bits=8, mode='mxfp8')` — the exact
scheme the attention path already uses in production).

Questions (per task A2):
  1. Logit-level divergence: BF16 lm_head vs mxfp8 lm_head on realistic
     hidden-state inputs. (Temp=0 sampling consumes argmax of these logits.)
  2. Does mxfp8 shift top-1 (greedy) token selection? At what rate?
  3. M-batch invariance: production runs lm_head at M in {1, 3, 4} (draft L=3,
     verify L=4, decode L=1). Does the quantized lm_head's per-row output
     depend on M (the qmv vs qmv_wide kernel split at M>=2)? If yes, the
     quantized head would introduce a NEW M-dependence the BF16 head has
     under MLX_GEMV_BATCH_INVARIANT... actually check: is BF16 lm_head
     M-invariant under the production flags? The flags pin batch-invariance
     for UNQUANTIZED matmul; the quantized path (quantized_matmul) has NO
     invariance flag. Compare div(M=1 vs M=4) for both dtypes.
  4. Throughput microbench of the quantized head at production shapes (the
     projected win is -0.7ms/token; verify directionally on the laptop GPU,
     final A/B on cluster).

Input realism: N(0,1) synthetic hidden states at decode shapes + the final
RMSNorm applied (production applies self.norm(x) before lm_head — norms to
~unit RMS anyway). A real-x capture rides the Phase-A relaunch (dump knob)
for the live quality gate; this offline pass is the numerics gate.
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2] / "mlx-lm"
sys.path.insert(0, str(REPO))

import mlx.core as mx
import mlx.nn as nn

HERE = Path(__file__).parent
W_BF16 = HERE / "head_weight.bf16"
V, D = 129280, 4096


def load_head_bf16():
    raw = np.fromfile(W_BF16, dtype=np.uint16)
    assert raw.size == V * D, (raw.size, V * D)
    f32 = (raw.astype(np.uint32) << 16).view(np.float32)
    return f32.reshape(V, D)


def main():
    w_np = load_head_bf16()
    print(f"head.weight BF16 loaded: {w_np.shape}, "
          f"absmax={np.abs(w_np).max():.3f} std={w_np.std():.4f}")

    # --- real module pair ---
    lm_bf16 = nn.Linear(D, V, bias=False)
    lm_bf16.weight = mx.array(w_np.astype(np.float32)).astype(mx.bfloat16)
    lm_bf16.eval()
    mx.eval(lm_bf16.parameters())

    lm_q = nn.Linear(D, V, bias=False)
    lm_q.weight = mx.array(w_np.astype(np.float32)).astype(mx.bfloat16)
    qmod = lm_q.to_quantized(group_size=32, bits=8, mode="mxfp8")
    qmod.eval()
    mx.eval(qmod.parameters())
    print(f"quantized head: weight {qmod.weight.dtype} {qmod.weight.shape} "
          f"({qmod.weight.nbytes/1e6:.1f} MB packed) + scales "
          f"{qmod.scales.nbytes/1e6:.1f} MB -> "
          f"{(qmod.weight.nbytes + qmod.scales.nbytes)/1e9:.3f} GB vs "
          f"1.059 GB BF16")

    # dequantized-weight error (weight fidelity)
    # QuantizedLinear exposes weight in packed form; reconstruct via matmul probe:
    # compare BF16 W vs dequant(W) on identity-like inputs is not needed —
    # what matters is output divergence. Compute output stats directly.

    rng = np.random.default_rng(20260830)
    results = {"per_M": {}, "flops": {}}

    for M in (1, 3, 4, 8):
        # N independent hidden states, each evaluated as its own M=1 call
        # AND as rows of an M-row batched call.
        Hs = rng.standard_normal((M * 8, D)).astype(np.float32)
        # production applies final RMSNorm before lm_head; emulate
        Hn = Hs / (np.sqrt((Hs ** 2).mean(axis=-1, keepdims=True)) + 1e-6)
        x_b = mx.array(Hn).astype(mx.bfloat16)

        # BF16 head: batched vs per-row
        logits_bf16_batch = lm_bf16(x_b.reshape(M, 8, D).reshape(-1, D))
        mx.eval(logits_bf16_batch)
        per_b = []
        for i in range(M * 8):
            l = lm_bf16(x_b[i : i + 1])
            mx.eval(l)
            per_b.append(l)
        logits_bf16_per = mx.concatenate(per_b, axis=0)

        # quantized head: batched vs per-row
        logits_q_batch = qmod(x_b.reshape(-1, D))
        mx.eval(logits_q_batch)
        per_q = []
        for i in range(M * 8):
            l = qmod(x_b[i : i + 1])
            mx.eval(l)
            per_q.append(l)
        logits_q_per = mx.concatenate(per_q, axis=0)

        # normalize to per-row M=1 semantics: the batched M*8 call is a
        # DIFFERENT M than the per-row calls; the M-invariance question is
        # "does row i's output depend on how many rows shared the batch".
        # logits_*_batch used M = M*8 rows; per-row used M=1.
        def stats(a, b):
            d = (a.astype(mx.float32) - b.astype(mx.float32))
            return (float(mx.abs(d).max()), float(mx.abs(d).mean()))

        bf16_invar = stats(logits_bf16_batch, logits_bf16_per)
        q_invar = stats(logits_q_batch, logits_q_per)

        # quantization error (per-row, M=1): quantized vs BF16
        qerr_per = stats(logits_q_per, logits_bf16_per)

        # top-1 flip rate at temp=0 (greedy) per-row M=1
        arg_b = mx.argmax(logits_bf16_per, axis=-1)
        arg_q = mx.argmax(logits_q_per, axis=-1)
        flips = int((arg_b != arg_q).sum())
        n = arg_b.shape[0]

        # top-5 set overlap
        k = 5
        # cheap top-k via argpartition on numpy
        lb = np.asarray(logits_bf16_per.astype(mx.float32))
        lq = np.asarray(logits_q_per.astype(mx.float32))
        top_b = np.argpartition(-lb, k, axis=-1)[:, :k]
        top_q = np.argpartition(-lq, k, axis=-1)[:, :k]
        top5_match = float(np.mean([
            len(set(top_b[i]) & set(top_q[i])) / k for i in range(n)
        ]))

        # logit margin distribution (how close is the top-1 race?)
        srt = -np.sort(-lb, axis=-1)
        margins = srt[:, 0] - srt[:, 1]

        # quantization error relative to margin
        qerr_rows = np.abs(lq - lb).max(axis=-1)

        results["per_M"][M] = {
            "bf16_M_invariance": bf16_invar,
            "quant_M_invariance": q_invar,
            "quant_vs_bf16_perrow": qerr_per,
            "top1_flips": flips, "top1_total": int(n),
            "top1_flip_rate": flips / n,
            "top5_set_overlap_frac": top5_match,
            "margin_p50": float(np.median(margins)),
            "margin_p10": float(np.percentile(margins, 10)),
            "qerr_gt_margin_rows": int((qerr_rows > margins).sum()),
        }
        print(f"\nM={M} rows/call: bf16 batch-vs-row max={bf16_invar[0]:.2e} | "
              f"quant batch-vs-row max={q_invar[0]:.2e}")
        print(f"  quant-vs-bf16 per-row: max={qerr_per[0]:.4f} mean={qerr_per[1]:.5f}")
        print(f"  top-1 flips: {flips}/{n} ({100*flips/n:.2f}%) | top-5 set overlap: "
              f"{top5_match*100:.1f}%")
        print(f"  greedy margin p50={np.median(margins):.3f} p10={np.percentile(margins,10):.3f} | "
              f"rows where qerr>margin: {int((qerr_rows > margins).sum())}/{n}")

    out = HERE / "lmhead_numerics.json"
    out.write_text(json.dumps(results, indent=1, default=str))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()