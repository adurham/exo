# P07b: top-up capture — only the ops that errored / need re-examination in the
# completed P07 run. Same env parity + timing recipe as p07 run. Appends
# results to results_topup.json (separate file; P07a results.json untouched).
import os

for v in ("METAL_CAPTURE_ENABLED", "MLX_GPU_TIME", "MLX_DISPATCH_COUNT"):
    assert os.environ.get(v) == "1", f"{v}=1 required before mlx import"
for v in ("MLX_GEMV_BATCH_INVARIANT", "MLX_STEEL_BATCH_INVARIANT",
          "EXO_DSV4_HC_COLLAPSE_KERNEL", "EXO_DSV4_HC_EXPAND_KERNEL"):
    assert os.environ.get(v) == "1", f"{v}=1 required (production parity)"

import json, math, sys, time, random
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, "/Users/adam.durham/repos/exo/mlx-lm")
from mlx_lm.models.deepseek_v4 import _moe_post_combine, _gate_route

OUT = Path("/Users/adam.durham/repos/exo/tmp/p07-20260830")
RESULTS = {"meta": {}, "ops": {}, "errors": []}
H, N_EXPERTS, TOP_K = 4096, 256, 6
L_BAND, P, IDX_HEADS, IDX_DIM, IDX_TOPK = 1024, 55000, 64, 128, 512
SPEC, REAL, TFLOPS = 546.0, 424.0, 14.34e12  # 40-core M4 Max fp32-FMA peak @~1.4GHz

def log(*a): print(*a, flush=True)
def save(): (OUT / "results_topup.json").write_text(json.dumps(RESULTS, indent=1, default=str))

RESULTS["meta"] = {
    "host": os.uname().nodename, "mlx": mx.__version__,
    "gpu": mx.device_info()["architecture"],
    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    "note": "top-up for P07: score_full (fixed call), rmsnorm dram, gate dram, "
            "post_combine gputrace evidence, tail pmask FULL variant",
    "spec_gbps": 546.0, "real_streaming_gbps": 424.0,
    "flops_ceiling_40core_fp32_fma": 14.34,
}

def bench(name, mk_call, n, warmup, bytes_modeled=None, flops_modeled=None,
          L=None, extra=None, passes=3):
    try:
        for j in range(warmup):
            out = mk_call(j % 512)
            mx.eval(*out) if isinstance(out, (list, tuple)) else mx.eval(out)
        mx.synchronize()
        recs = []
        for _p in range(passes):
            t0 = time.perf_counter()
            mx.metal.reset_gpu_time()
            d0 = mx.metal.dispatch_count()
            for i in range(n):
                out = mk_call(i)
                mx.eval(*out) if isinstance(out, (list, tuple)) else mx.eval(out)
            mx.synchronize()
            recs.append({"wall_us": (time.perf_counter() - t0) / n * 1e6,
                         "gpu_us": mx.metal.gpu_time_ns() / n / 1e3,
                         "dispatches": (mx.metal.dispatch_count() - d0) / n})
        srt = {k: sorted(x[k] for x in recs) for k in recs[0]}
        r = {k: srt[k][len(srt[k]) // 2] for k in srt}
        r["bytes"] = bytes_modeled
        r["flops"] = flops_modeled
        if bytes_modeled and flops_modeled:
            inten = flops_modeled / bytes_modeled
            r["intensity_flop_per_byte"] = inten
            if inten < 10:
                r["ceiling"] = "bandwidth"
                r["gbps"] = bytes_modeled / (r["gpu_us"] * 1e-6) / 1e9
                r["pct_spec"] = 100 * r["gbps"] / 546
                r["pct_real"] = 100 * r["gbps"] / 424
            else:
                r["ceiling"] = "flops"
                r["tflops"] = flops_modeled / (r["gpu_us"] * 1e-6) / 1e12
                r["pct_ceiling"] = 100 * r["tflops"] / 14.34
        elif bytes_modeled:
            r["ceiling"] = "bandwidth(bytes-only)"
            r["gbps"] = bytes_modeled / (r["gpu_us"] * 1e-6) / 1e9
            r["pct_spec"] = 100 * r["gbps"] / 546
            r["pct_real"] = 100 * r["gbps"] / 424
        r["passes"] = [{k: x[k] for k in ("wall_us", "gpu_us", "dispatches")} for x in recs]
        if L is not None:
            r["L"] = L
        if extra:
            r.update(extra)
        RESULTS["ops"].setdefault(name, {})[str(L) if L is not None else "-"] = r
        log(f"{name:<34} gpu={r['gpu_us']:9.2f} wall={r['wall_us']:9.2f} disp={r['dispatches']:6.1f}")
        save()
        return r
    except Exception as e:
        RESULTS["errors"].append(f"{name}: {type(e).__name__}: {e}")
        log(f"ERROR {name}: {e}")
        save()
        return None

# weight banks (rotated) — reuse P07 seed pattern
rng = random.Random(20260831)
mx.random.seed(20260831)
N_ROT = 64
gate_ws = [(mx.random.normal((N_EXPERTS, H)) * 0.02).astype(mx.bfloat16) for _ in range(N_ROT)]
gate_bs = [mx.zeros((N_EXPERTS,), dtype=mx.float32) for _ in range(N_ROT)]
perm = list(range(N_ROT)); rng.shuffle(perm)
wproj = (mx.random.normal((IDX_HEADS, H)) * 0.02).astype(mx.bfloat16)
mx.eval(*gate_ws, *gate_bs, wproj)
log("bank ready")

def xs_bank(L, n=8, seed=1):
    mx.random.seed(seed)
    return [mx.random.normal((1, L, H)).astype(mx.bfloat16) for _ in range(n)]

# ---- 1) FIXED attn.indexer.score_full ----
L_BAND_ = L_BAND
S_FLOPS = 2.0 * L_BAND * IDX_DIM * P + 2.0 * L_BAND * IDX_HEADS * IDX_DIM  # GEMM + weight-fold
pooled = (mx.random.normal((1, P, IDX_DIM)) * 0.05).astype(mx.bfloat16)
q4 = mx.random.normal((1, IDX_HEADS, L_BAND, IDX_DIM)).astype(mx.bfloat16)
xs = xs_bank(L_BAND, 4, 500)
mx.eval(pooled, q4)
# score_full byte model: q fold + GEMM in/out (pooled NOT rotated in this topup run)
S_BYTES = L_BAND * IDX_DIM * 2 + P * IDX_DIM * 2 + L_BAND * P * 2 + L_BAND * H * 2 + IDX_HEADS * L_BAND * 4 + IDX_DIM * L_BAND * 2
def mk_full(i):
    w = (mx.sigmoid(xs[i % 4] @ wproj.T) * (IDX_DIM ** -0.5 * IDX_HEADS ** -0.5))
    q_blhd = q4.transpose(0, 2, 1, 3)
    q_weighted = (w[..., None] * q_blhd).sum(axis=2)
    return q_weighted @ pooled.swapaxes(-1, -2)
r = bench("attn.indexer.score_full", mk_full, 20, 3, S_BYTES, S_FLOPS, L_BAND,
          extra={"note": "sigmoid weights_proj + H-fold + GEMM: fills the P07a gap "
                         "(POTENTIAL CONCERN: pooled NOT rotated here, single set is 14 MB, partially L2-resident)"})

# ---- 2) rmsnorm DRAM variant (P07a hot was 106% spec — suspicious) ----
per = L_BAND * H * 2 * 2 + H * 2
nsets = max(8, min(512, int(192e6 // per)))
mx.random.seed(700)
norm_xs = [mx.random.normal((1, L_BAND, H)).astype(mx.bfloat16) for _ in range(nsets)]
norm_ws = [mx.random.normal((H,)).astype(mx.bfloat16) for _ in range(nsets)]
mx.eval(*norm_xs, *norm_ws)
mk = lambda i: norm_xs[i % nsets] * norm_ws[i % nsets] / mx.sqrt(mx.mean(norm_xs[i % nsets] * norm_xs[i % nsets], axis=-1, keepdims=True) + 1e-6)
# use the real nn.RMSNorm math (weights * x / rms) — matches mlx nn.RMSNorm fast path? Use mx.fast.rms_norm:
def mk_fast(i):
    return mx.fast.rms_norm(norm_xs[i % nsets], norm_ws[i % nsets], 1e-6)
bench("tail.rmsnorm.dram", mk_fast, min(nsets, 256), 8, per, None, L_BAND,
      extra={"note": f"rotated {nsets} sets ({nsets * per / 1e6:.0f} MB); fast.rms_norm — P07a hot said 579 GB/s (over spec)"})

# ---- 3) moe.gate DRAM variant (P07a was weight-rotated 64 sets, fine, but x input bank was only 8 — add dram input rotation) ----
G_FLOPS = 2.0 * L_BAND * N_EXPERTS * H
G_BYTES = N_EXPERTS * H * 2 + L_BAND * H * 2 + L_BAND * N_EXPERTS * 8 + N_EXPERTS * 4
gx = xs_bank(L_BAND, 8, 100)
mk = lambda i: _gate_route(gx[i % 8], gate_ws[perm[i % N_ROT]], gate_bs[perm[i % N_ROT]],
                           TOP_K, 1.5, True, "sqrtsoftplus")
bench("moe.gate.check", mk, 100, 8, G_BYTES, G_FLOPS, L_BAND,
      extra={"note": "repeat of P07a moe.gate.routed for cross-run stability"})

# ---- 4) post_combine gputrace evidence (unique names, assert bundle exists) ----
try:
    per = TOP_K * H * 2 * L_BAND + TOP_K * 4 * L_BAND + 3 * H * 2 * L_BAND
    ys = mx.random.normal((1, L_BAND, TOP_K, H)).astype(mx.bfloat16)
    sc = mx.random.normal((1, L_BAND, TOP_K)).astype(mx.float32)
    sh = mx.random.normal((1, L_BAND, H)).astype(mx.bfloat16)
    mx.eval(ys, sc, sh)
    path = str(OUT / "captures" / "p07b_post_combine.gputrace")
    import os as _os
    if _os.path.exists(path):
        _os.remove(path)
    mx.metal.start_capture(path)
    for _ in range(3):
        o = _moe_post_combine(ys, sc, sh)
        mx.eval(o)
    mx.synchronize()
    mx.metal.stop_capture()
    sz = _os.path.getsize(path)
    RESULTS["meta"]["post_combine_gputrace"] = {"path": path, "bytes": sz}
    log(f"gputrace post_combine: {sz/1e6:.1f} MB")
except Exception as e:
    RESULTS["errors"].append(f"gputrace: {e}")
    log(f"gputrace ERROR: {e}")

save()
log("=== P07b TOPUP COMPLETE ===")
save()