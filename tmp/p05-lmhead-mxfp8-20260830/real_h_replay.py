#!/usr/bin/env python3
"""P05 real-h replay harness for the lm_head mxfp8 A/B (2026-08-30).

Runs AFTER the PM re-runs the A/B with EXO_DSV4_PRENORM_H_DUMP set —
the knob captures REAL hidden states from the model tail as bf16 .bin
+ json sidecar (shape list + "n").

INPUT-MODE NOTE (verified by code reading 2026-08-30, see summary):
  The dump site in deepseek_v4.py Model.__call__ sits AFTER
  `h = self.model(inputs, cache)`, and _forward_steps' tail ALREADY
  applies `out = finalize(self.norm(self.hc_head(h)))` INSIDE the model
  — so what the knob actually captures is the POST-norm lm_head INPUT
  (3D (B, L, 4096)), not the pre-tail hidden. (The pre-tail hidden is
  4D (B, L, hc_mult=4, 4096) — HyperHead's fn is [4, 16384] and its
  rms_norm(z) @ fn.T math only type-checks for a 16384-wide last-but-one
  layout, i.e. a 4D input.)

  This harness therefore AUTO-DETECTS per capture:
    ndim==4 (B, L, 4, 4096)  -> PRENORM: reconstruct the lm_head input
        exactly as production does, x = self.norm(self.hc_head(h)), by
        IMPORTING AND CALLING the real classes from the vendored mlx-lm
        submodule (HyperHead from mlx_lm/models/hyper_connection.py,
        nn.RMSNorm), loaded with the REAL 0731 weights hc_head_fn /
        hc_head_base / hc_head_scale / norm.weight. Never reimplements
        the math.
    ndim==3 (B, L, 4096)     -> POSTNORM: the capture IS the lm_head
        input (production: `_logits = self.lm_head(h)` directly); using
        it through norm(hc_head(...)) again would DOUBLE-APPLY the tail.
  Override with --input-mode {auto,prenorm,postnorm}.

Then it runs those inputs through BOTH heads: the production BF16
nn.Linear (head_weight.bf16, md5-checked) and its
to_quantized(group_size=32, bits=8, mode='mxfp8') version — exactly the
knob's quantization — and reports, per-capture and pooled: logit err
max/mean/rms, top-1 flip count/rate, BF16 top-1 margin distribution
(p10/p50/p90), flips stratified by margin band (<1.44 / 1.44-3.62 /
>3.62 — the bands from lmhead_numerics_v2), top-5 set overlap, and
M-batch invariance on the REAL hiddens (argmax of the quantized head
at M=1 per-row vs the capture's native L as one batched call).

Usage:
  # real dumps (scp'd off the nodes into the default dir, or any dir):
  .venv/bin/python real_h_replay.py --dump-dir <dir-with-prenorm_h_*.bin>
  # synthetic surrogate plumbing test:
  .venv/bin/python real_h_replay.py --dump-dir surrogate_dumps

Prints a compact machine-readable JSON summary at the end.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

# Production env parity for the BF16 arm (production pins BF16 batch
# invariance with these flags) — same as lmhead_numerics_v2.py. MUST be
# set before mlx is imported.
os.environ.setdefault("MLX_GEMV_BATCH_INVARIANT", "1")
os.environ.setdefault("MLX_STEEL_BATCH_INVARIANT", "1")

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] / "mlx-lm"
sys.path.insert(0, str(REPO))

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
from mlx_lm.models.hyper_connection import HyperHead  # noqa: E402

# Real 0731 tail weights + the production BF16 lm_head weight.
WEIGHTS_DIR = Path.home() / "repos/exo/tmp/p05-sinkhorn-real-20260830/p05_weights"
HEAD_BF16 = HERE / "head_weight.bf16"
HEAD_BF16_MD5 = "d21485e51a7d9fe9da83b0b0643f972b"

# ModelArgs defaults for DeepSeek-V4-Flash-0731 (deepseek_v4.py):
#   hidden_size=4096, hc_mult=4, rms_norm_eps=1e-6, hc_eps=1e-6
HIDDEN, HC_MULT, RMS_EPS, HC_EPS = 4096, 4, 1e-6, 1e-6
VOCAB = 129280

# Default: real dumps scp'd into the local tmp workspace.
DEFAULT_DUMP_DIR = HERE / "prenorm_h_dumps"


class _Cfg:
    """Minimal config stand-in with exactly the attrs HyperHead reads."""
    hc_mult = HC_MULT
    rms_norm_eps = RMS_EPS
    hc_eps = HC_EPS
    hidden_size = HIDDEN


def load_manifest_weight(key: str) -> np.ndarray:
    manifest = json.loads((WEIGHTS_DIR / "manifest.json").read_text())
    m = manifest[key]
    # manifest paths point at /tmp/p05_weights (the studio origin); the
    # local copy sits beside manifest.json — resolve by basename.
    raw = (WEIGHTS_DIR / Path(m["path"]).name).read_bytes()
    if m["dtype"] == "F32":
        return np.frombuffer(raw, dtype=np.float32).reshape(m["shape"])
    if m["dtype"] == "BF16":
        u16 = np.frombuffer(raw, dtype=np.uint16)
        return (u16.astype(np.uint32) << 16).view(np.float32).reshape(m["shape"])
    raise ValueError(f"unsupported dtype {m['dtype']} for {key}")


def build_production_tail():
    """The REAL hc_head + norm, loaded with the REAL 0731 weights."""
    hc_head = HyperHead(_Cfg())
    hc_head.fn = mx.array(load_manifest_weight("hc_head_fn"))        # f32 [4, 16384]
    hc_head.base = mx.array(load_manifest_weight("hc_head_base"))    # f32 [4]
    hc_head.scale = mx.array(load_manifest_weight("hc_head_scale"))  # f32 [1]
    norm = nn.RMSNorm(HIDDEN, eps=RMS_EPS)
    norm.weight = mx.array(load_manifest_weight("norm.weight")).astype(mx.bfloat16)
    mx.eval(hc_head.parameters(), norm.parameters())
    return hc_head, norm


def build_heads():
    """BF16 production lm_head + the knob's mxfp8 quantized version."""
    md5 = hashlib.md5(HEAD_BF16.read_bytes()).hexdigest()
    if md5 != HEAD_BF16_MD5:
        raise SystemExit(f"head_weight.bf16 md5 mismatch: {md5} != {HEAD_BF16_MD5}")
    raw = np.fromfile(HEAD_BF16, dtype=np.uint16)
    assert raw.size == VOCAB * HIDDEN, raw.size
    w_np = (raw.astype(np.uint32) << 16).view(np.float32).reshape(VOCAB, HIDDEN)

    lm_bf16 = nn.Linear(HIDDEN, VOCAB, bias=False)
    lm_bf16.weight = mx.array(w_np).astype(mx.bfloat16)
    lm_q = nn.Linear(HIDDEN, VOCAB, bias=False)
    lm_q.weight = mx.array(w_np).astype(mx.bfloat16)
    qmod = lm_q.to_quantized(group_size=32, bits=8, mode="mxfp8")
    mx.eval(lm_bf16.parameters(), qmod.parameters())
    return lm_bf16, qmod


def load_dumps(dump_dir: Path):
    """All prenorm_h_*.bin + .json sidecars, sorted by name."""
    bins = sorted(dump_dir.glob("prenorm_h_*.bin"))
    if not bins:
        raise SystemExit(f"no prenorm_h_*.bin files in {dump_dir}")
    dumps = []
    for b in bins:
        side = b.with_name(b.name + ".json")
        if not side.exists():
            print(f"WARNING: {b.name} has no sidecar; skipping", file=sys.stderr)
            continue
        meta = json.loads(side.read_text())
        shape = meta["shape"]
        raw = np.fromfile(b, dtype=np.uint16)
        expected = int(np.prod(shape))
        if raw.size != expected:
            print(f"WARNING: {b.name}: {raw.size} u16 != shape {shape} "
                  f"({expected}); skipping", file=sys.stderr)
            continue
        h_np = (raw.astype(np.uint32) << 16).view(np.float32).reshape(shape)
        dumps.append({"name": b.name, "shape": shape, "n": meta.get("n"),
                      "h": mx.array(h_np).astype(mx.bfloat16)})
    if not dumps:
        raise SystemExit("no valid dumps loaded")
    return dumps


def detect_input_mode(shape: list, override: str) -> str:
    if override != "auto":
        return override
    if len(shape) == 4 and shape[2] == HC_MULT and shape[3] == HIDDEN:
        return "prenorm"
    if len(shape) == 3 and shape[2] == HIDDEN:
        return "postnorm"
    raise SystemExit(
        f"cannot auto-detect input mode for shape {shape}: expected 4D "
        f"(B,L,{HC_MULT},{HIDDEN}) pre-norm or 3D (B,L,{HIDDEN}) post-norm; "
        f"pass --input-mode explicitly")


def percentile(margins: np.ndarray, p: float) -> float:
    return float(np.percentile(margins, p)) if margins.size else float("nan")


def analyze_rows(lb_np: np.ndarray, lq_np: np.ndarray) -> dict:
    """Row-wise metrics shared by per-capture and pooled reporting."""
    d = lq_np - lb_np
    srt = np.sort(lb_np, axis=-1)[:, ::-1]
    margin = srt[:, 0] - srt[:, 1]
    flips = lb_np.argmax(-1) != lq_np.argmax(-1)

    bands = [
        ("<1.44", margin < 1.44),
        ("1.44-3.62", (margin >= 1.44) & (margin < 3.62)),
        (">3.62", margin >= 3.62),
    ]
    strat = []
    for label, sel in bands:
        n = int(sel.sum())
        strat.append({
            "band": label, "n": n,
            "flips": int(flips[sel].sum()) if n else 0,
            "flip_rate": float(flips[sel].mean()) if n else None,
        })

    k = 5
    top_b = np.argpartition(-lb_np, k, axis=-1)[:, :k]
    top_q = np.argpartition(-lq_np, k, axis=-1)[:, :k]
    top5 = float(np.mean([len(set(top_b[i]) & set(top_q[i])) / k
                          for i in range(lb_np.shape[0])]))

    return {
        "rows": int(lb_np.shape[0]),
        "logit_err_max": float(np.abs(d).max()),
        "logit_err_mean": float(np.abs(d).mean()),
        "logit_err_rms": float(np.sqrt((d ** 2).mean())),
        "flips": int(flips.sum()),
        "flip_rate": float(flips.mean()),
        "margin_p10": percentile(margin, 10),
        "margin_p50": percentile(margin, 50),
        "margin_p90": percentile(margin, 90),
        "flips_by_margin_band": strat,
        "top5_overlap": top5,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", default=str(DEFAULT_DUMP_DIR),
                    help="dir with prenorm_h_*.bin (+ .json sidecars); "
                         "also accepts scp'd copies. Default: %(default)s")
    ap.add_argument("--input-mode", default="auto",
                    choices=["auto", "prenorm", "postnorm"],
                    help="prenorm: capture is the pre-tail hidden "
                         "(B,L,4,4096); reconstruct lm_head input via the "
                         "real HyperHead+RMSNorm. postnorm: capture IS the "
                         "lm_head input (what the current dump site writes).")
    ap.add_argument("--max-rows-per-capture", type=int, default=0,
                    help="cap rows per capture (0 = all)")
    args = ap.parse_args()

    dump_dir = Path(args.dump_dir)
    print(f"dump dir: {dump_dir}")
    dumps = load_dumps(dump_dir)
    print(f"loaded {len(dumps)} capture(s): "
          + ", ".join(f"{d['name']} shape={d['shape']}" for d in dumps))

    hc_head = norm = None
    if any(detect_input_mode(d["shape"], args.input_mode) == "prenorm"
           for d in dumps):
        hc_head, norm = build_production_tail()
        print("prenorm mode: real HyperHead + RMSNorm loaded with 0731 weights")
    lm_bf16, qmod = build_heads()
    print("heads built (BF16 + mxfp8); head_weight.bf16 md5 verified")

    per_capture = []
    all_lb, all_lq = [], []
    for dmp in dumps:
        h = dmp["h"]
        mode = detect_input_mode(dmp["shape"], args.input_mode)
        if mode == "prenorm":
            # Production tail (deepseek_v4.py _forward_steps, non-rowseq
            # branch): out = finalize(self.norm(self.hc_head(h))).
            # finalize() is a no-op pass-through unless a profiler hook
            # is registered (profiler.py) — not part of the input math.
            assert hc_head is not None and norm is not None
            x = norm(hc_head(h))
        else:
            # Production (Model.__call__): `_logits = self.lm_head(h)`
            # directly — the capture already IS the lm_head input.
            x = h
        if args.max_rows_per_capture and x.shape[-2] > args.max_rows_per_capture:
            x = x[:, : args.max_rows_per_capture]
        mx.eval(x)
        rows2d = x.reshape(-1, HIDDEN)  # (B*L, D)

        # --- both heads, M=1 per-row (the decode-shaped reference) ---
        lb_rows, lq_rows = [], []
        for i in range(rows2d.shape[0]):
            lb_rows.append(lm_bf16(rows2d[i: i + 1]))
            lq_rows.append(qmod(rows2d[i: i + 1]))
        lb1 = mx.concatenate(lb_rows, axis=0)
        lq1 = mx.concatenate(lq_rows, axis=0)
        mx.eval(lb1, lq1)
        lb_np = np.asarray(lb1.astype(mx.float32))
        lq_np = np.asarray(lq1.astype(mx.float32))

        stats = analyze_rows(lb_np, lq_np)
        stats["name"] = dmp["name"]
        stats["shape"] = dmp["shape"]
        stats["input_mode"] = mode

        # --- M-batch invariance on REAL hiddens: the capture's native L
        # as ONE batched call vs per-row M=1 argmax ---
        L = dmp["shape"][1] if len(dmp["shape"]) >= 2 else 1
        if L > 1:
            lq_batch = qmod(x)  # (B, L, V) batched at native L
            mx.eval(lq_batch)
            batch_argmax = np.asarray(
                lq_batch.astype(mx.float32)).reshape(-1, VOCAB).argmax(-1)
            mbatch_flips = int((batch_argmax != lq_np.argmax(-1)).sum())
            stats["mbatch"] = {
                "native_L": L,
                "argmax_flips_vs_M1": mbatch_flips,
                "flip_rate": float(mbatch_flips / batch_argmax.size),
            }
        else:
            stats["mbatch"] = None

        per_capture.append(stats)
        print(f"\n--- {dmp['name']} shape={dmp['shape']} mode={mode} ---")
        print(json.dumps(stats, indent=1))

        all_lb.append(lb_np)
        all_lq.append(lq_np)

    pooled = analyze_rows(np.concatenate(all_lb, axis=0),
                          np.concatenate(all_lq, axis=0))
    modes = sorted({s["input_mode"] for s in per_capture})

    summary = {
        "dump_dir": str(dump_dir),
        "n_captures": len(dumps),
        "input_modes": modes,
        "pooled": pooled,
        "per_capture": per_capture,
    }
    print("\n" + "=" * 62)
    print("REAL-H REPLAY SUMMARY (machine-readable)")
    print("=" * 62)
    print(json.dumps(summary))
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())