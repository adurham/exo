#!/usr/bin/env python3
"""Phase 2 microbench: EXL3SwitchGLU vs affine SwitchGLU at REAL V4.1 shapes.

Answers the plan's decisive Phase-2 question ("EXL3 gather-GEMV/GEMM on a real
MoE") using the ALREADY-CONVERTED V4.1 EXL3 checkpoint -- no conversion, no exo
integration, no production relaunch. Runs in the isolated phase1-exl3 venv.

V4.1 shapes: hidden 5120, moe_intermediate 2304, top-6 routed.
Layer 1 is a k=3 layer (layers 18-22 are k=2 -- do not use those for k=3).

Both sides are built from THE SAME expert weights, so this is format-vs-format:
  EXL3   k=3 trellis (mul1)           <- read straight from the checkpoint
  affine 4-bit gs=64                  <- dequantize EXL3 -> requantize affine
  affine 3-bit gs=64                  <- bit-width control (EXL3 is 2.9 bpw)
A dense fp16 reference validates orientation at small R.

PRODUCTION ACTIVATION NOTE: DeepseekV4MoE uses
    LimitedSwiGLU(10.0): silu(min(gate,10)) * clip(up,-10,10)
while EXL3SwitchGLU implements only plain silu: silu(g)*u. This script uses
PLAIN silu on both sides (apples-to-apples for orientation + timing) and
separately measures how often the +-10 clamp actually binds, because that is a
real fidelity gap the integration has to close.
"""
from __future__ import annotations

import gc
import json
import os
import time

import numpy as np

import mlx.core as mx
import mlx.nn as nn

MODEL = os.path.expanduser(
    "~/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
)
LAYER = int(os.environ.get("BENCH_LAYER", "1"))
E = int(os.environ.get("BENCH_EXPERTS", "128"))
KK = 6
HID = 5120
INT = 2304
SWIGLU_LIMIT = 10.0
CORRECTNESS_ONLY = os.environ.get("BENCH_CORRECTNESS_ONLY") == "1"


def log(*a):
    print(*a, flush=True)


def gib(n):
    return f"{n / 2**30:.2f} GiB"


def mem(tag):
    if hasattr(mx, "get_active_memory"):
        log(f"[mem] {tag}: active={gib(mx.get_active_memory())} "
            f"peak={gib(mx.get_peak_memory())}")


def flush():
    gc.collect()
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()


log("=" * 78)
log(f"P2 EXL3 MoE microbench  layer={LAYER}  experts_stacked={E}  top_k={KK}")
log(f"model: {MODEL}")
log("=" * 78)
mem("start")

# ---------------------------------------------------------------- read tensors
idx = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))["weight_map"]
SUFFIX = ("w1.trellis", "w1.suh", "w1.svh",
          "w2.trellis", "w2.suh", "w2.svh",
          "w3.trellis", "w3.suh", "w3.svh")
by_shard: dict[str, list[str]] = {}
for e in range(E):
    for s in SUFFIX:
        k = f"layers.{LAYER}.ffn.experts.{e}.{s}"
        by_shard.setdefault(idx[k], []).append(k)

from safetensors import safe_open  # noqa: E402

tens: dict[str, np.ndarray] = {}
t0 = time.perf_counter()
for s, ks in sorted(by_shard.items()):
    with safe_open(os.path.join(MODEL, s), framework="np") as f:
        for k in ks:
            tens[k] = f.get_tensor(k)
    flush()
log(f"[load] {len(tens)} tensors from {len(by_shard)} shards in "
    f"{time.perf_counter() - t0:.1f}s")

p0 = f"layers.{LAYER}.ffn.experts.0."
K_BITS = int(tens[p0 + "w1.trellis"].shape[2] * 16 // 256)
log(f"[shape] w1.trellis={tens[p0 + 'w1.trellis'].shape} -> k={K_BITS} "
    f"(packed_size {tens[p0 + 'w1.trellis'].shape[2]})")
log(f"[shape] w1: in={len(tens[p0 + 'w1.suh'])} out={len(tens[p0 + 'w1.svh'])} | "
    f"w2: in={len(tens[p0 + 'w2.suh'])} out={len(tens[p0 + 'w2.svh'])}")
assert len(tens[p0 + "w1.suh"]) == HID and len(tens[p0 + "w1.svh"]) == INT
assert len(tens[p0 + "w2.suh"]) == INT and len(tens[p0 + "w2.svh"]) == HID

# ------------------------------------------------------------- EXL3 module
from ponyexl3.mlx.exl3_moe import EXL3SwitchGLU  # noqa: E402
from ponyexl3.ref.codebook import codebook_mode_from_flags  # noqa: E402

CB = codebook_mode_from_flags(mcg=False, mul1=True)   # config: codebook=mul1
log(f"[cb] codebook_mode={int(CB)} (mul1 per config.json)")

gu_parts, up_parts, gu_suh, gu_svh = [], [], [], []
dn_parts, dn_suh, dn_svh = [], [], []
for e in range(E):
    p = f"layers.{LAYER}.ffn.experts.{e}."
    gu_parts.append(mx.array(tens[p + "w1.trellis"]))    # w1 = gate
    up_parts.append(mx.array(tens[p + "w3.trellis"]))    # w3 = up
    gu_suh.append(mx.stack([mx.array(tens[p + "w1.suh"]),
                            mx.array(tens[p + "w3.suh"])]))
    gu_svh.append(mx.concatenate([mx.array(tens[p + "w1.svh"]),
                                  mx.array(tens[p + "w3.svh"])]))
    dn_parts.append(mx.array(tens[p + "w2.trellis"]))    # w2 = down
    dn_suh.append(mx.array(tens[p + "w2.suh"]))
    dn_svh.append(mx.array(tens[p + "w2.svh"]))

t0 = time.perf_counter()
exl3 = EXL3SwitchGLU(
    gu_trellis=mx.concatenate(gu_parts + up_parts, axis=1).view(mx.uint16),
    gu_suh=mx.stack(gu_suh).astype(mx.float16),
    gu_svh=mx.stack(gu_svh).astype(mx.float16),
    dn_trellis=mx.concatenate(dn_parts, axis=1).view(mx.uint16),
    dn_suh=mx.stack(dn_suh).astype(mx.float16),
    dn_svh=mx.stack(dn_svh).astype(mx.float16),
    k=K_BITS, cb=CB, activation="silu",
)
mx.eval(exl3._gu_trellis, exl3._gu_suh, exl3._gu_svh,
        exl3._dn_trellis, exl3._dn_suh, exl3._dn_svh)
log(f"[exl3] built in {time.perf_counter() - t0:.1f}s  "
    f"gu_trellis={exl3._gu_trellis.shape} dn_trellis={exl3._dn_trellis.shape}")
log(f"[exl3] in={exl3.input_dims} hidden={exl3.hidden_dims} "
    f"E={exl3.num_experts} gu_tiles={exl3._gu_tiles} dn_tiles={exl3._dn_tiles}")
log(f"[exl3] _v2_ok()={exl3._v2_ok()}  (v2 needs hidden<=512; V4.1 is {INT})")
import ponyexl3.mlx.exl3_moe as _m  # noqa: E402
log(f"[exl3] env: UNFUSED={_m._MOE_UNFUSED} V2={_m._MOE_V2} MM={_m._MOE_MM} "
    f"SEG_BM={_m._SEG_BM} MM_MAX_ROWS={_m._MM_MAX_ROWS}")
mem("after exl3 build")
# NOTE: we deliberately KEEP tens[] (incl. trellis) — deq_public() needs it for
# the affine baselines and the fp16 reference. ~3 GiB for 128 experts.

# ------------------------------------------- affine baselines + fp16 reference
from ponyexl3.ref.layer import EXL3Layer  # noqa: E402
from ponyexl3.mlx.reconstruct import reconstruct_public_mlx  # noqa: E402
from mlx_lm.models.switch_layers import (  # noqa: E402
    QuantizedSwitchLinear, SwiGLU, SwitchGLU,
)


def deq_public(prefix: str) -> mx.array:
    """EXL3 -> public fp16 matrix, shape (in, out)."""
    lay = EXL3Layer(
        key=prefix, in_features=len(tens[prefix + "suh"]),
        out_features=len(tens[prefix + "svh"]), k=K_BITS,
        trellis=tens[prefix + "trellis"], suh=tens[prefix + "suh"],
        svh=tens[prefix + "svh"], mul1=True,
    )
    return reconstruct_public_mlx(lay)          # (in, out) fp16


class AffineSwitchLinear(QuantizedSwitchLinear):
    """Production QuantizedSwitchLinear.__call__ VERBATIM, cheap init.

    The stock __init__ allocates a random (E, out, in) fp32 tensor (6+ GiB at
    these dims) which we would immediately throw away; __call__ (the only thing
    being timed) is inherited unchanged.
    """

    def __init__(self, W: np.ndarray, bits: int, group_size: int):
        nn.Module.__init__(self)
        self.group_size = group_size
        self.bits = bits
        self.mode = "affine"
        w, scales, *biases = mx.quantize(mx.array(W), group_size=group_size,
                                         bits=bits, mode="affine")
        self.weight = w
        self.scales = scales
        if biases:
            self.biases = biases[0]
        mx.eval(self.weight, self.scales)


def _stack(prefixes: list[str]) -> np.ndarray:
    """Stack per-expert public matrices transposed to mlx-lm's (E, out, in)."""
    out = None
    for e, p in enumerate(prefixes):
        m = np.array(deq_public(p)).T          # (out, in)
        if out is None:
            out = np.empty((len(prefixes),) + m.shape, dtype=np.float16)
        out[e] = m
        del m
    return out


def build_affine(bits: int, group_size: int = 64) -> dict:
    mods = {}
    for name, pre in (("gate", "w1"), ("up", "w3"), ("down", "w2")):
        t0 = time.perf_counter()
        P = f"layers.{LAYER}.ffn.experts."
        W = _stack([f"{P}{e}.{pre}." for e in range(E)])
        log(f"[affine{bits}] {name}: fp16 stack {W.shape} {gib(W.nbytes)} "
            f"in {time.perf_counter() - t0:.1f}s")
        mods[name] = AffineSwitchLinear(W, bits, group_size)
        del W
        flush()
        mem(f"after affine{bits} {name}")
    mods["bits"] = bits
    mods["mod"] = BenchSwitchGLU(mods)
    return mods


class BenchSwitchGLU(SwitchGLU):
    """Production SwitchGLU with EXL3-derived affine projections.

    __init__ is replaced ONLY because the stock one allocates random
    (E, out, in) fp32 tensors (GiBs at these dims) that we immediately
    discard.  __call__ is inherited UNCHANGED, so the timed path is
    byte-for-byte the production MoE call path.
    """

    def __init__(self, mods: dict):
        nn.Module.__init__(self)
        self.gate_proj = mods["gate"]
        self.up_proj = mods["up"]
        self.down_proj = mods["down"]
        self.activation = SwiGLU()


def affine_forward(x: mx.array, indices: mx.array, mods: dict) -> mx.array:
    """Exactly production's call: switch_mlp(x, indices) -> (B, S, kk, D)."""
    return mods["mod"](x, indices)


# ---------------------------------------------------------------- correctness
_REFCACHE: dict = {}


def pub(expert: int, proj: str) -> mx.array:
    """Lazily dequantize ONE expert's public matrix (in, out), cached.

    A full (E, in, out) fp16 stack would be ~9 GiB; the reference test only
    touches a handful of experts, so materialize on demand.
    """
    key = (expert, proj)
    if key not in _REFCACHE:
        _REFCACHE[key] = mx.array(
            deq_public(f"layers.{LAYER}.ffn.experts.{expert}.{proj}."))
        mx.eval(_REFCACHE[key])
    return _REFCACHE[key]


def dense_ref(x: mx.array, indices: mx.array, limit: float | None = None):
    """fp16 dense reference. Activation matches production LimitedSwiGLU:
    silu(min(gate,limit)) * clip(up,-limit,limit); limit=None -> plain silu."""
    B, S, kk = indices.shape
    R = B * S
    xr = x.reshape(R, HID)
    idxnp = np.array(indices).reshape(R, kk)
    out = np.zeros((R, kk, HID), dtype=np.float32)
    stats = {"max_abs_gate": 0.0, "max_abs_up": 0.0, "clipped": 0, "total": 0}
    for r in range(R):
        for j in range(kk):
            e = int(idxnp[r, j])
            g = np.array(xr[r] @ pub(e, "w1"))          # w1 = gate
            u = np.array(xr[r] @ pub(e, "w3"))          # w3 = up
            stats["max_abs_gate"] = max(stats["max_abs_gate"], float(np.abs(g).max()))
            stats["max_abs_up"] = max(stats["max_abs_up"], float(np.abs(u).max()))
            stats["total"] += g.size + u.size
            if limit:
                stats["clipped"] += int((g > limit).sum())
                stats["clipped"] += int((np.abs(u) > limit).sum())
            gg = mx.minimum(mx.array(g), limit) if limit else mx.array(g)
            uu = mx.clip(mx.array(u), -limit, limit) if limit else mx.array(u)
            # silu(GATE) * UP  (matches mlx-lm SwiGLU + EXL3 gateup kernel)
            out[r, j] = np.array(((gg * mx.sigmoid(gg)) * uu) @ pub(e, "w2"))
    return out.reshape(B, S, kk, HID), stats


def rel(a, b):
    a = np.array(a, dtype=np.float32).ravel()
    b = np.array(b, dtype=np.float32).ravel()
    return float(np.sqrt(((a - b) ** 2).mean()) / (np.sqrt((b ** 2).mean()) + 1e-12))


mx.random.seed(0)
log("\n" + "=" * 78)
log("CORRECTNESS (proves orientation; plain silu on all three sides)")
log("=" * 78)
AFF4 = build_affine(4)
mem("after affine4 complete")
for R in (1, 4):
    x = mx.random.normal((1, R, HID)).astype(mx.float16) * 0.1
    ind = mx.array(np.stack([np.random.choice(E, KK, replace=False)
                             for _ in range(R)]).reshape(1, R, KK).astype(np.int32))
    ref, st = dense_ref(x, ind)
    y_ex = exl3(x, ind)
    y_af = affine_forward(x, ind, AFF4)
    mx.eval(ref, y_ex, y_af)
    log(f"  R={R}: EXL3-vs-dense={rel(y_ex, ref):.5f}  "
        f"aff4-vs-dense={rel(y_af, ref):.5f}  "
        f"EXL3-vs-aff4={rel(y_ex, y_af):.5f}")
    log(f"        shapes exl3={y_ex.shape} aff={y_af.shape} ref={ref.shape}")
    log(f"        max|gate|={st['max_abs_gate']:.2f} max|up|={st['max_abs_up']:.2f}"
        f" -> SWIGLU_LIMIT={SWIGLU_LIMIT} binds on "
        f"{100.0 * st['clipped'] / st['total']:.4f}% of elements")

if CORRECTNESS_ONLY:
    log("\nBENCH_CORRECTNESS_ONLY=1 -> stopping before timing.")
    raise SystemExit(0)

_REFCACHE.clear()
flush()

# ---------------------------------------------------------------------- timing
def bench(fn, x, ind, reps, warmup=6):
    for _ in range(warmup):
        mx.eval(fn(x, ind))
    mx.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(fn(x, ind))
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn(x, ind)
    mx.eval(fn(x, ind))
    mx.synchronize()
    return {"median_ms": ts[len(ts) // 2], "min_ms": ts[0],
            "amort_ms": (time.perf_counter() - t0) * 1e3 / reps, "reps": reps}


CASES = [(1, 300), (4, 200), (512, 20), (2048, 8)]
results: dict = {}
log("\n" + "=" * 78)
log(f"TIMING per MoE block (gate+up+silu+down, top-{KK} of {E} stacked experts)")
log("=" * 78)
AFF3 = build_affine(3)
mem("after affine3 complete")
log(f"\n{'rows':>6} {'EXL3 k%d med' % K_BITS:>17} {'aff4 med':>12} {'aff3 med':>12} "
    f"{'ex/aff4':>9} {'ex/aff3':>9}")
for R, reps in CASES:
    x = mx.random.normal((1, R, HID)).astype(mx.float16) * 0.1
    ind = mx.array(np.stack([np.random.choice(E, KK, replace=False)
                             for _ in range(R)]).reshape(1, R, KK).astype(np.int32))
    mx.eval(x, ind)
    r_ex = bench(exl3, x, ind, reps)
    r_4 = bench(lambda a, b: affine_forward(a, b, AFF4), x, ind, reps)
    r_3 = bench(lambda a, b: affine_forward(a, b, AFF3), x, ind, reps)
    results[R] = {"exl3": r_ex, "aff4": r_4, "aff3": r_3}
    log(f"{R:>6} {r_ex['median_ms']:>15.3f}ms {r_4['median_ms']:>10.3f}ms "
        f"{r_3['median_ms']:>10.3f}ms "
        f"{r_ex['median_ms'] / r_4['median_ms']:>9.3f} "
        f"{r_ex['median_ms'] / r_3['median_ms']:>9.3f}")

log("\n--- amortized (one sync per batch; strips per-call dispatch) ---")
log(f"{'rows':>6} {'EXL3 ms':>12} {'aff4 ms':>12} {'aff3 ms':>12} "
    f"{'ex/aff4':>9} {'ex/aff3':>9}")
for R, _ in CASES:
    r = results[R]
    log(f"{R:>6} {r['exl3']['amort_ms']:>12.3f} {r['aff4']['amort_ms']:>12.3f} "
        f"{r['aff3']['amort_ms']:>12.3f} "
        f"{r['exl3']['amort_ms'] / r['aff4']['amort_ms']:>9.3f} "
        f"{r['exl3']['amort_ms'] / r['aff3']['amort_ms']:>9.3f}")

log("\n" + "=" * 78)
log("VERDICT vs plan gates (EXL3 <= 1.25x affine decode, <= 1.7x affine prefill)")
log("=" * 78)
rr = {R: results[R]["exl3"]["median_ms"] / results[R]["aff4"]["median_ms"]
      for R, _ in CASES}
for R, lbl in ((1, "decode R=1"), (4, "verify R=4"), (512, "prefill R=512"),
               (2048, "prefill R=2048")):
    gate = 1.25 if R <= 4 else 1.7
    log(f"  {lbl:<16} ratio vs aff4 = {rr[R]:.3f}   "
        f"{'PASS' if rr[R] <= gate else 'MISS'} (gate {gate})")
log("  note: aff3 (3-bit) is the closer bit-width control; EXL3 is 2.9 bpw.")

out = {"layer": LAYER, "experts_stacked": E, "top_k": KK, "k_bits": K_BITS,
       "hidden": HID, "inter": INT, "v2_ok": bool(exl3._v2_ok()),
       "swiglu_limit": SWIGLU_LIMIT, "results": results,
       "ratios_vs_aff4": {str(k): v for k, v in rr.items()}}
with open(os.path.expanduser("~/p2_exl3_bench_results.json"), "w") as f:
    json.dump(out, f, indent=1)
log("\nwrote ~/p2_exl3_bench_results.json")
log("DONE")
