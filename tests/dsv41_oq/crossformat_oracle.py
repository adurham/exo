# Copyright © 2026 Adam Durham (hermes-gw)
"""Cross-format weight oracle: oQ (native-MLX) vs EXL3 1017 pack, layer 0.

The oQ loader (``oq_build``) and the EXL3 loader (``exl3.loader``) read two
*different* quantizations of the same bf16 base model. Neither stores the base
weights, so "is the oQ loader right?" cannot be answered by comparing against
the source -- only by reconstructing both formats to float and asking whether
they agree to within the noise the two quantizers add.

EXL3 side: ``load_dense_layer`` + ``reconstruct_public_mlx`` -- the vendored
public-weight reconstruction (trellis decode -> Hadamard -> sign vectors), the
same code the runtime's ``W`` cache would hold.

oQ side: ``OqCheckpoint.dequant`` / ``mx.dequantize`` with the per-module
bits/group/mode from ``config.json``.

A loader bug that permutes blocks, transposes a projection or reads the wrong
expert shows up as a cosine near 0 on that tensor while its neighbours stay at
~0.999; a *different quantization* of the same weights shows up as a uniform
0.99x. That separation is the whole point of the report.

Usage (needs the EXL3 layer-0 shard locally, ~5.3 GiB):

    PYTHONPATH=$PWD/src:$PWD/mlx-lm .venv/bin/python \
        tests/dsv41_oq/crossformat_oracle.py \
        --oq-pack ~/.hermes/cache/scratch/oqmini \
        --exl3-pack ~/.hermes/cache/scratch/exl3_1017_l0 \
        --experts 8 --json /tmp/crossformat.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (os.path.join(ROOT, "mlx-lm"), os.path.join(ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mlx.core as mx  # noqa: E402

from mlx_lm.models.deepseek_v41 import oq_build  # noqa: E402
from mlx_lm.models.exl3.loader import Exl3Checkpoint, load_dense_layer  # noqa: E402
from mlx_lm.models.exl3.reconstruct import reconstruct_public_mlx  # noqa: E402


# --------------------------------------------------------------------------
# readers
# --------------------------------------------------------------------------
def oq_arr(ck, name: str, sel=None) -> mx.array:
    """``OqCheckpoint.array`` with an optional numpy row selection.

    Selecting in numpy first keeps a 384-expert stack from being copied into
    MLX in one piece -- the mini pack's w1 stack alone is 1.1 GiB packed.
    """
    flat, mx_dt = ck._raw(name)                      # noqa: SLF001
    a = flat if sel is None else flat[sel]
    arr = mx.array(np.ascontiguousarray(a))
    return arr if arr.dtype == mx_dt else arr.view(mx_dt)


def oq_dense(ck, module: str) -> mx.array:
    """Dequantized ``[out, in]`` for a 2-D oQ module."""
    if not ck.is_quantized(module):
        return ck.array(f"{module}.weight").astype(mx.float32)
    w, s, b, qi = ck._parts(module)                  # noqa: SLF001
    bits, group, mode = oq_build._qparams(qi)        # noqa: SLF001
    out = mx.dequantize(w, s, b, group_size=group, bits=bits, mode=mode)
    mx.eval(out)
    return out.astype(mx.float32)


def oq_expert(ck, module: str, e: int) -> mx.array:
    """Dequantized ``[out, in]`` of expert ``e`` from a stacked oQ projection."""
    w, s, b, qi = ck._parts(module)                  # noqa: SLF001
    bits, group, mode = oq_build._qparams(qi)        # noqa: SLF001
    w = oq_arr(ck, f"{module}.weight", e)
    s = oq_arr(ck, f"{module}.scales", e)
    b = (oq_arr(ck, f"{module}.biases", e)
         if ck.has(f"{module}.biases") else None)
    out = mx.dequantize(w, s, b, group_size=group, bits=bits, mode=mode)
    mx.eval(out)
    return out.astype(mx.float32)


def exl3_dense(ck, prefix: str) -> mx.array:
    """Reconstructed ``[out, in]`` float32 for one EXL3 dense group.

    ``reconstruct_public_mlx`` hands back exllamav3's stored ``[in, out]``
    orientation; the oQ reader (and MLX ``nn.Linear``) use ``[out, in]``, so
    the transpose here is what makes the two comparable at all. Without it
    every comparison degenerates into a shape mismatch.
    """
    w = reconstruct_public_mlx(load_dense_layer(ck, prefix, mul1=True))
    return w.T.astype(mx.float32)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def compare(a: mx.array, b: mx.array) -> dict:
    """Cosine similarity + relative Frobenius error between two matrices."""
    if tuple(a.shape) != tuple(b.shape):
        return {"shape_mismatch": [list(a.shape), list(b.shape)]}
    af = a.reshape(-1).astype(mx.float32)
    bf = b.reshape(-1).astype(mx.float32)
    dot = float((af * bf).sum())
    na = float(mx.sqrt((af * af).sum()))
    nb = float(mx.sqrt((bf * bf).sum()))
    diff = af - bf
    nerr = float(mx.sqrt((diff * diff).sum()))
    return {
        "cos": dot / (na * nb) if na and nb else float("nan"),
        "rel_fro": nerr / na if na else float("nan"),
        "n": int(af.size),
        "shape": list(a.shape),
    }


def _free():
    try:
        mx.clear_cache()
    except Exception:            # pragma: no cover - older mlx
        pass


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
DENSE = ["attn.wq_a", "attn.wq_b", "attn.wkv", "attn.wo_b",
         "ffn.shared_experts.w1", "ffn.shared_experts.w2",
         "ffn.shared_experts.w3"]
EXPERTS = ["ffn.experts.w1", "ffn.experts.w2", "ffn.experts.w3"]
#: Plain (unquantized in both packs) tensors -- an exact-match table proves the
#: two packs are the same base model with the same tensor names, which is the
#: precondition for reading anything into the quantized rows.
PLAIN = ["attn_norm.weight", "ffn_norm.weight", "attn.q_norm.weight",
         "attn.kv_norm.weight", "attn.attn_sink", "ffn.gate.weight",
         "ffn.gate.bias", "hc_attn_base", "hc_attn_fn", "hc_attn_scale",
         "hc_ffn_base", "hc_ffn_fn", "hc_ffn_scale"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oq-pack", default=os.path.expanduser(
        "~/.hermes/cache/scratch/oqmini"))
    ap.add_argument("--exl3-pack", default=os.path.expanduser(
        "~/.hermes/cache/scratch/exl3_1017_l0"))
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--experts", type=int, default=8,
                    help="how many of the 384 routed experts to sample")
    ap.add_argument("--json", default=None)
    ap.add_argument("--skip-plain", action="store_true")
    args = ap.parse_args()

    oq = oq_build.OqCheckpoint(args.oq_pack)
    ex = Exl3Checkpoint(args.exl3_pack)
    pre = f"layers.{args.layer}."
    n_exp = ex.n_experts(args.layer)
    report: dict = {"layer": args.layer, "n_experts_total": n_exp,
                    "oq_pack": args.oq_pack, "exl3_pack": args.exl3_pack,
                    "dense": {}, "experts": {}, "plain": {},
                    "plain_mismatch": [], "wo_a": {}}

    print(f"# layer {args.layer}: oQ={args.oq_pack}  EXL3={args.exl3_pack}")
    print(f"# {n_exp} routed experts; sampling {args.experts}\n")

    # -- dense projections ---------------------------------------------------
    print("| tensor | oQ bits/group/mode | shape | cos | rel Frobenius |")
    print("|---|---|---|---|---|")
    for tail in DENSE:
        module = pre + tail
        qi = oq.qinfo(module) or {}
        bits, group, mode = (oq_build._qparams(qi) if qi   # noqa: SLF001
                             else (None, None, "plain"))
        a = oq_dense(oq, module)
        b = exl3_dense(ex, module)
        m = compare(a, b)
        m.update(bits=bits, group=group, mode=mode)
        report["dense"][tail] = m
        print(f"| {tail} | {bits}/{group}/{mode} | {m.get('shape')} | "
              f"{m.get('cos', float('nan')):.6f} | {m.get('rel_fro', 0):.4f} |"
              if "cos" in m else f"| {tail} | {bits}/{group}/{mode} | "
              f"MISMATCH {m} | | |")
        del a, b
        _free()

    # -- wo_a: 8 per-group EXL3 slices vs one plain oQ tensor ----------------
    wo_a_mod = pre + "attn.wo_a"
    if ex.has(f"{wo_a_mod}.slice.0.trellis") and oq.has(f"{wo_a_mod}.weight"):
        slices = []
        g = 0
        while ex.has(f"{wo_a_mod}.slice.{g}.trellis"):
            slices.append(exl3_dense(ex, f"{wo_a_mod}.slice.{g}"))
            g += 1
        # reconstruct_public_mlx returns [in, out]; after exl3_dense's transpose
        # each group slice is [o_lora, heads_per_group*head_dim] and the oQ
        # tensor stacks those groups on axis 0: [groups*o_lora, heads*dim].
        stacked = mx.concatenate(slices, axis=0) if len(slices) > 1 else slices[0]
        a = oq_dense(oq, wo_a_mod)
        m = compare(a, stacked)
        m.update(slices=g, bits="plain", group=None, mode="plain")
        report["wo_a"] = m
        print(f"| attn.wo_a (group-stacked, {g} EXL3 slices) | plain | "
              f"{m.get('shape')} | {m.get('cos', float('nan')):.6f} | "
              f"{m.get('rel_fro', 0):.4f} |" if "cos" in m else
              f"| attn.wo_a | plain | MISMATCH {m} | | |")
        del a, stacked, slices
        _free()

    # -- routed experts ------------------------------------------------------
    picks = sorted(set(np.linspace(0, n_exp - 1, args.experts).astype(int)
                       .tolist()))
    print(f"\n## routed experts (sampled {picks})\n")
    print("| tensor | expert | oQ bits/group/mode | cos | rel Frobenius |")
    print("|---|---|---|---|---|")
    for tail in EXPERTS:
        module = pre + tail
        qi = oq.qinfo(module) or {}
        bits, group, mode = (oq_build._qparams(qi) if qi   # noqa: SLF001
                             else (None, None, "plain"))
        rows = []
        for e in picks:
            a = oq_expert(oq, module, e)
            b = exl3_dense(ex, f"{pre}ffn.experts.{e}.{tail.split('.')[-1]}")
            m = compare(a, b)
            m.update(bits=bits, group=group, mode=mode, expert=e)
            rows.append(m)
            print(f"| {tail} | {e} | {bits}/{group}/{mode} | "
                  f"{m.get('cos', float('nan')):.6f} | "
                  f"{m.get('rel_fro', 0):.4f} |")
            del a, b
            _free()
        report["experts"][tail] = rows

    # -- plain tensors -------------------------------------------------------
    if not args.skip_plain:
        print("\n## plain (unquantized in both packs) - exact match expected\n")
        print("| tensor | shape | bit-exact | max |diff| |")
        print("|---|---|---|---|")
        for tail in PLAIN:
            name = pre + tail
            if not (oq.has(name) and ex.has(name)):
                continue
            a = oq_arr(oq, name).astype(mx.float32)
            b = mx.array(ex.np(name)).astype(mx.float32)
            if tuple(a.shape) != tuple(b.shape):
                report["plain_mismatch"].append(
                    {"tensor": tail, "reason": f"shape {tuple(a.shape)} vs "
                                               f"{tuple(b.shape)}"})
                print(f"| {tail} | MISMATCH SHAPE | no | |")
                continue
            d = mx.abs(a - b)
            maxd = float(d.max())
            exact = bool(maxd == 0.0)
            report["plain"][tail] = {"shape": list(a.shape), "exact": exact,
                                     "max_abs_diff": maxd}
            if not exact:
                report["plain_mismatch"].append(
                    {"tensor": tail, "max_abs_diff": maxd})
            print(f"| {tail} | {tuple(a.shape)} | {'yes' if exact else 'NO'} | "
                  f"{maxd:.3e} |")
            del a, b, d
            _free()

    # -- embed (plain bf16 in both; lives in its own EXL3 shard) -------------
    if ex.has("embed.weight") and oq.has("embed.weight"):
        a = oq_arr(oq, "embed.weight")
        # ex.np widens BF16 -> float32 exactly; compare both sides in float32
        b = mx.array(ex.np("embed.weight")).astype(mx.float32)
        a32 = a.astype(mx.float32)
        maxd = float(mx.abs(a32 - b).max())
        m = compare(a32, b)
        m.update(exact=bool(maxd == 0.0), max_abs_diff=maxd)
        report["embed"] = m
        print(f"\n## embed (plain bf16 both)  cos={m['cos']:.6f}  "
              f"max|diff|={maxd:.3e}  bit-exact={'yes' if maxd == 0 else 'NO'}")
        del a, b, a32
        _free()

    # -- verdict -------------------------------------------------------------
    dense_cos = [m["cos"] for m in report["dense"].values() if "cos" in m]
    exp_cos = [m["cos"] for rows in report["experts"].values()
               for m in rows if "cos" in m]
    worst = sorted(
        [(m["cos"], f"dense.{k}") for k, m in report["dense"].items()
         if "cos" in m] +
        [(m["cos"], f"{k}[e{m['expert']}]") for k, rows in
         report["experts"].items() for m in rows if "cos" in m])[:8]
    report["summary"] = {
        "dense_min_cos": min(dense_cos) if dense_cos else None,
        "dense_median_cos": float(np.median(dense_cos)) if dense_cos else None,
        "experts_min_cos": min(exp_cos) if exp_cos else None,
        "experts_median_cos": float(np.median(exp_cos)) if exp_cos else None,
        "worst": [[round(c, 6), t] for c, t in worst],
        "plain_mismatches": len(report["plain_mismatch"]),
        "suspect_zero_cos": [t for c, t in worst if c < 0.5],
    }
    print("\n## summary")
    print(json.dumps(report["summary"], indent=2))

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(report, fh, indent=1)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
