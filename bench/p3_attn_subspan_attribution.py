"""P3 Worker C — per-sub-span attribution inside ONE SparseCompressedAttention
layer, at decode shapes, vs context depth L.

Uses the fork's OWN instrumentation (EXO_DSV4_SECTION_TIME=1 ->
_SECTION_TIME_ENABLED at deepseek_v4.py:180 -> the _stsub fenced blocks inside
SparseCompressedAttention.__call__ that accumulate into _ATTN_SUB_ACC:
compressor / proj_qkv / qk_prep / indexer / sdpa / out_proj).

Every sub-span is separated by an explicit mx.eval + mx.synchronize, so the
TOTAL here is the fully-serialized upper bound and is NOT comparable to the
chained totals in p3_attn_depth_walltime_microbench.py. What IS meaningful is
(a) the relative split and (b) how each sub-span's time changes with L.

Run: python bench/p3_attn_subspan_attribution.py --depths 100026,352599,500000
"""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

_PROD_ENV = {
    "EXO_DSV4_INDEX_TOPK": "512", "EXO_KV_CACHE_BITS": "0",
    "EXO_COMPUTE_DTYPE": "bf16", "EXO_DSV4_SPARSE_SDPA_TILE": "128",
    "EXO_DSV4_SEQ_SPLIT": "1", "EXO_DSV4_EXACT_TOPK": "1",
    "EXO_DSV4_TOPK_FUSED": "0", "EXO_DSV4_SPARSE_FUSED_SDPA": "0",
    "EXO_DSV4_ATTN_ALLSUM": "0", "EXO_DSV4_SINGLE_GATHER": "1",
    "EXO_DSV4_POOL_DEFER_COPY_MAX_BYTES": "8388608",
    "EXO_DSV4_SECTION_TIME": "1",          # <-- the only non-production var
}
for _k, _v in _PROD_ENV.items():
    os.environ[_k] = _v

import mlx.core as mx           # noqa: E402

_REPO = Path(__file__).resolve().parent.parent
_MLXLM = _REPO / "mlx-lm"
if not _MLXLM.is_dir():
    _MLXLM = Path.home() / "repos" / "exo" / "mlx-lm"
sys.path.insert(0, str(_MLXLM))

# IMPORTANT: import deepseek_v4 BEFORE the microbench module. The microbench
# pops EXO_DSV4_SECTION_TIME from the environment (it wants clean production
# defaults), and _SECTION_TIME_ENABLED is read once at deepseek_v4 import time
# (deepseek_v4.py:180). Importing here freezes it True.
from mlx_lm.models import deepseek_v4 as dv4               # noqa: E402
assert dv4._SECTION_TIME_ENABLED, "section-time instrumentation did not arm"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p3_attn_depth_walltime_microbench import (            # noqa: E402
    CFG, DTYPE, build_attn, make_cache,
)

SUBS = ("compressor", "proj_qkv", "qk_prep", "indexer", "sdpa", "out_proj")


def run(L, steps, warmup):
    attn = build_attn(4)
    cache, meta = make_cache(4, L)
    x = mx.random.normal((1, 1, CFG["hidden_size"])).astype(DTYPE)
    mx.eval(x)
    for _ in range(warmup):
        mx.eval(attn(x, mask=None, cache=cache))
    mx.synchronize()

    for k in dv4._ATTN_SUB_ACC:
        dv4._ATTN_SUB_ACC[k] = 0.0

    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        mx.eval(attn(x, mask=None, cache=cache))
    mx.synchronize()
    wall = (time.perf_counter() - t0) * 1e3 / steps

    acc = {k: dv4._ATTN_SUB_ACC[k] * 1e3 / steps for k in SUBS}
    del attn, cache
    mx.clear_cache()
    return acc, wall, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", default="520,100026,352599,500000")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    depths = [int(d) for d in a.depths.split(",")]

    print(f"MLX {mx.__version__} host={os.uname().nodename}")
    print("SparseCompressedAttention (compress_ratio=4, 21 of 43 layers), "
          "B=1 L_q=1 decode, SECTION_TIME fences ON (serialized upper bound)")
    print(f"steps={a.steps} warmup={a.warmup}\n")

    out = {}
    hdr = f"{'L':>10s} " + "".join(f"{s:>11s}" for s in SUBS) + f"{'wall':>11s}"
    print(hdr)
    for L in depths:
        acc, wall, meta = run(L, a.steps, a.warmup)
        print(f"{L:>10,} " + "".join(f"{acc[s]:11.4f}" for s in SUBS)
              + f"{wall:11.4f}")
        out[L] = dict(subs=acc, wall_ms=wall, **meta)

    print("\n=== per-sub-span DELTA vs depth (ms per layer per decode step) ===")
    print(f"{'segment':>24s} " + "".join(f"{s:>11s}" for s in SUBS))
    for i in range(1, len(depths)):
        a0, b0 = depths[i - 1], depths[i]
        d = {s: out[b0]["subs"][s] - out[a0]["subs"][s] for s in SUBS}
        print(f"{a0:>10,}->{b0:<12,} " + "".join(f"{d[s]:+11.4f}" for s in SUBS))
    print("\n=== same deltas SCALED x21 sparse layers (ms/token) ===")
    print(f"{'segment':>24s} " + "".join(f"{s:>11s}" for s in SUBS)
          + f"{'sum':>11s}")
    for i in range(1, len(depths)):
        a0, b0 = depths[i - 1], depths[i]
        d = {s: 21 * (out[b0]["subs"][s] - out[a0]["subs"][s]) for s in SUBS}
        print(f"{a0:>10,}->{b0:<12,} " + "".join(f"{d[s]:+11.4f}" for s in SUBS)
              + f"{sum(d.values()):+11.4f}")

    if a.out:
        Path(a.out).write_text(json.dumps(
            dict(host=os.uname().nodename, mlx=mx.__version__,
                 steps=a.steps, results={str(k): v for k, v in out.items()}),
            indent=2))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
