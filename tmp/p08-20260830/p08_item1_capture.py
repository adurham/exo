# P08 Item 1: decisive causal-vs-dense SDPA measurement + denominator-free
# direct headroom for attn.sdpa.compressed at production prefill shape.
#
# p01/P03/P07 recipe: standalone process on macstudio-m4-1 beside the live
# runner (PID 59909, DO NOT TOUCH). MLX_GPU_TIME=1 + MLX_DISPATCH_COUNT=1,
# fresh graph per timed call, mx.eval + mx.synchronize per call, median of
# >=5 timed passes after >=3 warmup. Rotation banks past L2. No *0 tricks.
#
# Production shape provenance (all verified in code, cited in results):
#   - attn.sdpa.compressed span: mlx-lm/mlx_lm/models/deepseek_v4.py:4385
#     (CompressedAttention.__call__); fused call at :4413-4425 via
#     scaled_dot_product_attention (base.py:122) ->
#     mx.fast.scaled_dot_product_attention (base.py:193-201).
#   - q: (1, 32, 1024, 512) bf16 — n_heads 64//TP2 (deepseek_v4.py:7390),
#     L_band = chunk 2048 / 2 ranks (EXO_DSV4_SEQ_SPLIT=1, start_cluster
#     .sh:108; slice at deepseek_v4.py:4381),
#   - kv: (1, 1, 3894, 512): local ring 128 (sliding_window, dv4:865) post
#     update = max_size + S - 1 = 2175 (cache.py:633), pooled r=128 at
#     ctx 220000 -> 1719 (bench:69), concatenated at dv4:4359.
#   - mask: MATERIALIZED bool (1, 1, 1024, 3894). Model-level
#     create_attention_mask(window_size=128, return_array=True) at
#     dv4:6766-6771 -> RotatingKVCache.make_mask -> create_causal_mask
#     (base.py:24-42, windowed-causal, (1,1,2048,2175) bool);
#     _extend_mask (dv4:1361-1391) concatenates the pooled row-causal mask
#     (PoolingCache.make_mask, cache.py:1605-1627 formula) broadcast to
#     (B,H,L,1719) -> (1,1,2048,3894), then band row-slice at dv4:4382-4383.
#   The bench (bench/attn_production_class_bench.py:212-236) models this
#   shape correctly BUT gives the mask 95%-dense RANDOM content instead of
#   the production causal+window content.
import os

for v in ("MLX_GPU_TIME", "MLX_DISPATCH_COUNT", "METAL_CAPTURE_ENABLED"):
    assert os.environ.get(v) == "1", f"{v}=1 required before mlx import"

import json
import socket
import statistics
import time
from pathlib import Path

import mlx.core as mx

OUT = Path("/Users/adam.durham/repos/exo/tmp/p08-20260830")
OUT.mkdir(parents=True, exist_ok=True)
RESULTS = {"meta": {}, "runtime_shape_probe": {}, "gemm_peak": {},
           "bandwidth": {}, "sdpa": {}, "floor": {}, "errors": []}

# ------------- production constants (cited above) -------------
N_HEADS = 32          # 64 // TP=2 (dv4.py:7390)
HEAD_DIM = 512
L_BAND = 1024         # seq-split v2 band per rank
LOCAL_KV = 2175       # 128 ring + 2048 - 1 (cache.py:633)
POOL_128 = 1719       # ceil(220000/128)
CATTN_KV = LOCAL_KV + POOL_128   # 3894
SLIDING = 128
OFF_LAST = 218880     # final-chunk band start (110 chunks x 2048 - 2048+...)
RATIO_C = 128
SCALE = HEAD_DIM ** -0.5
SPEC_546, REAL_424 = 546.0, 424.0

def save():
    (OUT / "item1_results.json").write_text(json.dumps(RESULTS, indent=1, default=str))

def log(*a):
    print(*a, flush=True)

log("=== P08 Item 1: sdpa compressed causal-vs-dense + direct floor ===")
RESULTS["meta"] = {
    "host": socket.gethostname(),
    "mlx": mx.__version__,
    "arch": mx.device_info(),
    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    "shape_constants": {
        "batch": 1, "n_heads_per_rank": N_HEADS, "L_band": L_BAND,
        "local_kv": LOCAL_KV, "pool_r128": POOL_128, "CATTN_KV": CATTN_KV,
        "head_dim": HEAD_DIM, "dtype": "bfloat16", "scale": SCALE,
        "offset_final_chunk": OFF_LAST, "sliding_window": SLIDING,
    },
    "assumed_by_bench": {
        "bench": "bench/attn_production_class_bench.py:212-236",
        "q": [1, 32, 1024, 512], "kv": [1, 1, 3894, 512],
        "mask_content": "random.uniform > 0.05 (95% dense), NOT production causal",
        "mask_shape": [1, 1, 1024, 3894], "mask_dtype": "bool",
    },
}
save()

# ------------------------------------------------------------- timing helpers
def timed(mk, warmup, iters):
    """mk() builds a FRESH lazy graph per call (real data deps, no folding).
    Returns per-call median GPU-busy us + wall us + dispatch count."""
    for _ in range(warmup):
        mx.eval(mk())
    mx.synchronize()
    recs = []
    for _ in range(iters):
        t0 = time.perf_counter()
        mx.metal.reset_gpu_time()
        d0 = mx.metal.dispatch_count()
        mx.eval(mk())
        mx.synchronize()
        wall_us = (time.perf_counter() - t0) * 1e6
        gpu_us = mx.metal.gpu_time_ns() / 1e3
        disp = mx.metal.dispatch_count() - d0
        recs.append((wall_us, gpu_us, disp))
    wall = statistics.median([r[0] for r in recs])
    gpu = statistics.median([r[1] for r in recs])
    disp = statistics.median([r[2] for r in recs])
    return {"median_wall_us": round(wall, 2), "median_gpu_us": round(gpu, 2),
            "median_dispatches": int(disp),
            "all_wall_us": [round(r[0], 1) for r in recs],
            "all_gpu_us": [round(r[1], 1) for r in recs],
            "warmup": warmup, "iters": iters}

# ------------------------------------------------------- runtime shape check
# Rebuild the production mask EXACTLY as the code does, print runtime shapes.
log("--- (A) runtime production-mask reconstruction ---")
masks = {}
# model-level windowed causal mask, exactly create_attention_mask(return_array
# =True) -> RotatingKVCache.make_mask -> create_causal_mask(N=2048, offset=127,
# window=128): (1,1,2048,2175) bool. RotatingKVCache.make_mask clamps offset to
# max_size-1 (cache.py:886-891) and returns the windowed causal array.
N_CHUNK_L = 2048
m_local = mx.arange(SLIDING - 1, SLIDING - 1 + N_CHUNK_L)[:, None] >= mx.arange(SLIDING - 1 + N_CHUNK_L)[None]
m_local = m_local & (mx.arange(SLIDING - 1, SLIDING - 1 + N_CHUNK_L)[:, None]
                     < mx.arange(SLIDING - 1 + N_CHUNK_L)[None] + SLIDING)
m_local = m_local[None, None]  # (1,1,2048,2175) bool
# pooled row-causal mask, exactly PoolingCache.make_mask(L=2048, offset=218880)
# (cache.py:1625-1627): pool_idx < query_pos // ratio, P=1719 visible at end.
P_vis = POOL_128
pool_idx = mx.arange(P_vis)[None, :]
query_pos = OFF_LAST + mx.arange(1, N_CHUNK_L + 1)[:, None]
m_pool = pool_idx < (query_pos // RATIO_C)          # (2048, 1719) bool
m_pool = mx.broadcast_to(m_pool[None, None], (1, 1, N_CHUNK_L, P_vis))
mask_full = mx.concatenate([m_local, m_pool], axis=-1)  # (1,1,2048,3894)
# seq-split band slice (dv4:4381-4383), rank 0 = rows [0,1024)
mask_rank0 = mask_full[..., 0:L_BAND, :]
mask_rank1 = mask_full[..., L_BAND:N_CHUNK_L, :]
mx.eval(mask_rank0, mask_rank1)
dens0 = float(mx.mean(mask_rank0.astype(mx.float32)).item())
dens1 = float(mx.mean(mask_rank1.astype(mx.float32)).item() if False else mx.mean(mask_rank1.astype(mx.float32)).item())
vis0 = int(mask_rank0.sum().item())
vis1 = int(mask_rank1.sum().item())
avg_vis = (vis0 + vis1) / 2.0 / L_BAND
RESULTS["runtime_shape_probe"] = {
    "mask_rank0": {"shape": list(mask_rank0.shape), "dtype": str(mask_rank0.dtype),
                   "density": round(dens0, 6), "visible_keys_total": vis0,
                   "avg_visible_keys_per_row": vis0 / L_BAND},
    "mask_rank1": {"shape": list(mask_rank1.shape), "dtype": str(mask_rank1.dtype),
                   "visible_keys_total": vis1, "avg_visible_keys_per_row": vis1 / L_BAND},
 "mean_visible_keys_per_row_both_ranks": avg_vis,
    "q_shape_production": [1, N_HEADS, L_BAND, HEAD_DIM],
    "kv_shape_production": [1, 1, CATTN_KV, HEAD_DIM],
    "mask_shape_production": list(mask_rank0.shape),
    "mask_dtype": "bool (mx.concatenate of bools, MATERIALIZED, broadcast over heads by the kernel)",
    "mask_materialized_bytes": int(L_BAND * CATTN_KV),
    "bench_assumed_mask": "random >=95% dense bool, same shape/dtype (bench:219)",
    "shape_matches_bench": True,
    "content_differs_from_bench": True,
}
log(f"mask shape {mask_rank0.shape} dtype {mask_rank0.dtype} density {dens0:.4f}")
log(f"avg visible keys/row (both ranks): {avg_vis:.1f} of {CATTN_KV}")
save()

# ---------------------------------------------------------------- tensors
q = mx.random.normal((1, N_HEADS, L_BAND, HEAD_DIM)).astype(mx.bfloat16)
sinks = mx.zeros((N_HEADS,), dtype=mx.bfloat16)
N_KV_BANKS = 16   # 16 x 3.99MB = 63.8MB >> 16MB L2
kv_banks = [mx.random.normal((1, 1, CATTN_KV, HEAD_DIM)).astype(mx.bfloat16)
            for _ in range(N_KV_BANKS)]
mx.eval(q, sinks, *kv_banks)
log(f"q 33.5MB; kv bank {CATTN_KV*HEAD_DIM*2/1e6:.2f}MB x {N_KV_BANKS} banks")

def dense_mask():
    return mx.ones((1, 1, L_BAND, CATTN_KV), dtype=mx.bool_)

MASK_CAUSAL = [mask_rank0, mask_rank1, mask_rank0, mask_rank1]  # rotate 2 real
MASK_DENSE = [dense_mask() for _ in range(4)]
mx.eval(*MASK_CAUSAL, *MASK_DENSE)

SDPA = {}
_ctr = {"i": 0}

def sdpa_rows():
    global _ctr
    for kind in ("causal", "dense", "none"):
        def mk(kind=kind):
            i = _ctr["i"]
            _ctr["i"] = i + 1
            k = kv_banks[i % N_KV_BANKS]
            if kind == "causal":
                m = MASK_CAUSAL[i % 4]
            elif kind == "dense":
                m = MASK_DENSE[i % 4]
            else:
                m = None
            return mx.fast.scaled_dot_product_attention(
                q, k, k, scale=SCALE, mask=m, sinks=sinks)
        r = timed(mk, warmup=6, iters=11)
        r["mask"] = kind
        SDPA[kind] = r
        log(f"  sdpa[{kind:>6}] gpu={r['median_gpu_us']:9.2f}us "
            f"wall={r['median_wall_us']:9.2f}us disp={r['median_dispatches']}")
        save()
SDPA_rows = sdpa_rows()
RESULTS["sdpa"] = SDPA

t_causal = SDPA["causal"]["median_gpu_us"]
t_dense = SDPA["dense"]["median_gpu_us"]
t_none = SDPA["none"]["median_gpu_us"]
R = t_causal / t_dense

# f from the ACTUAL production mask content
FLOP_DENSE = 2 * 2 * N_HEADS * L_BAND * CATTN_KV * HEAD_DIM   # QK^T + PV
FLOP_CAUSAL = FLOP_DENSE * (avg_vis / CATTN_KV)
f_ratio = FLOP_CAUSAL / FLOP_DENSE
RESULTS["sdpa_flops"] = {
    "flop_dense": FLOP_DENSE, "flop_actual_mask": FLOP_CAUSAL,
    "f_flop_ratio": f_ratio,
    "avg_visible_keys_per_row": avg_vis,
    "note": ("f computed from the production mask CONTENT (windowed-causal "
             "local 128/2175 + row-causal pooled): mean visible keys/row = "
             f"{avg_vis:.1f}. P07's 0.6058 was a run-average over growing "
             "context; at the final chunk the real content density is "
             f"{f_ratio:.4f}."),
}
log(f"  f(flop ratio, actual mask content) = {f_ratio:.4f}; R = {R:.4f}")

# L2 sanity for the sdpa: 2 banks (L2-resident KV) vs 16 banks
for label, nb in (("kvbanks2", 2), ("kvbanks16", 16)):
    ctr = {"i": 0}
    def mk(nb=nb, ctr=ctr):
        i = ctr["i"]
        ctr["i"] = i + 1
        return mx.fast.scaled_dot_product_attention(
            q, kv_banks[i % nb], kv_banks[i % nb], scale=SCALE,
            mask=MASK_CAUSAL[i % 4], sinks=sinks)
    RESULTS.setdefault("l2_sanity", {})[f"sdpa_{label}"] = timed(mk, 5, 9)
    save()
log("  l2sanity:", json.dumps(RESULTS["l2_sanity"]))

# ------------------------------------------------------------- (C) GEMM peak
log("--- (C) on-node dense GEMM peak sweep ---")
peak_best = None
for dt, dtname in ((mx.bfloat16, "bf16"), (mx.float16, "fp16")):
    for M, K, N in ((2048, 4096, 4096), (4096, 4096, 4096),
                    (8192, 4096, 4096), (16384, 4096, 4096),
                    (8192, 8192, 8192)):
        nsets = 4 if M <= 8192 else 2
        ab = [(mx.random.normal((M, K)).astype(dt),
               mx.random.normal((K, N)).astype(dt)) for _ in range(nsets)]
        mx.eval(*[t for pair in ab for t in pair])
        ctr = {"i": 0}
        def mk(ab=ab, nsets=nsets, ctr=ctr):
            i = ctr["i"]
            ctr["i"] = i + 1
            a, b = ab[i % nsets]
            return a @ b
        r = timed(mk, warmup=3, iters=7)
        tf = 2.0 * M * K * N / (r["median_gpu_us"] * 1e-6) / 1e12
        rec = {"dtype": str(dt).replace("mlx.core", ""), "shape": [M, K, N],
               "gpu_us": r["median_gpu_us"], "tflops": round(tf, 3),
               "dispatches": r["median_dispatches"]}
        RESULTS["gemm_peak"][f"{dtname}_{M}x{K}x{N}"] = rec
        if peak_best is None or tf > peak_best["tflops"]:
            peak_best = rec
        log(f"  {dtname} {M:>6}x{K}x{N}: {tf:6.2f} TF")
        save()
RESULTS["gemm_peak"]["best"] = peak_best
log(f"  PEAK: {peak_best['tflops']:.2f} TF at {peak_best['shape']} {peak_best['dtype']}")

# streaming bandwidth (roofline denominator for the softmax floor)
n_bw = 256 * 1024 * 1024 // 2
a_bw = mx.random.normal((n_bw,)).astype(mx.bfloat16)
mx.eval(a_bw)
ctr = {"i": 0}
def bw_mk(ctr=ctr):
    return a_bw + 1.0
r_bw = timed(bw_mk, warmup=3, iters=7)
bw_meas = a_bw.nbytes * 2 / (r_bw["median_gpu_us"] * 1e-6) / 1e9
RESULTS["bandwidth"] = {"gpu_us": r_bw["median_gpu_us"],
                        "read_write_gbps": bw_meas,
                        "note": "256MB bf16 stream, a+1.0"}
log(f"  streaming: {bw_meas:.1f} GB/s (r+w)")
save()

# ------------------------------------------------------- (D) direct floor
log("--- (D) direct floor ---")
H_, L_, KVN, D_ = N_HEADS, L_BAND, CATTN_KV, HEAD_DIM
# QK^T exact shape: (H, L, D) @ (D, KV) -> (H, L, KV); PV: (H, L, KV) @ (KV, D)
kk = kv_banks[0][0, 0].T      # (512, 3894) bf16
vv = kv_banks[1][0, 0]        # (3894, 512) bf16
n_bank = 3                    # score tensor 32x1024x3894x2B = 256MB each
q_banks = [mx.random.normal((H_, L_, D_)).astype(mx.bfloat16) for _ in range(3)]
s_banks = [mx.random.normal((H_, L_, KVN)).astype(mx.bfloat16) for _ in range(n_bank)]
mx.eval(*q_banks, *s_banks, kk, vv)

floor = {}

def bench_variant(name, mk):
    ctr = {"i": 0}
    def call():
        i = ctr["i"]
        ctr["i"] = i + 1
        return mk(i)
    r = timed(call, warmup=4, iters=7)
    floor[name] = r
    log(f"  {name}: gpu={r['median_gpu_us']:9.2f}us disp={r['median_dispatches']}")
    save()
    return r

# QK^T variants: broadcast-KV single matmul vs 32 separate matmuls (min kept)
def qk_broadcast(i):
    return q_banks[i % 3] @ kk
def qk_loop(i):
    # 32 separate (L,D)@(D,KV) matmuls sharing one KV — dispatch-heavy variant
    part = q_banks[i % 3]
    return [q_banks[i % 3][h] @ kk for h in range(H_)]
r_qkb = bench_variant("QK_broadcast", qk_broadcast)
r_qkl = bench_variant("QK_loop32", qk_loop)

def pv_broadcast(i):
    return s_banks[i % 3] @ vv
r_pvb = bench_variant("PV_broadcast", pv_broadcast)

# softmax pass over the score tensor (production: fused in kernel; floor probe)
def softmax_chain(i):
    s = s_banks[i % 3]
    e = mx.exp(s * SCALE - mx.max(s, axis=-1, keepdims=True))
    return e / mx.sum(e, axis=-1, keepdims=True)
r_sm = bench_variant("softmax_pass", softmax_chain)

# pure streaming bound for one read+write of the score tensor at measured bw
score_bytes = H_ * L_ * KVN * 2
t_sm_stream_us = score_bytes / (bw_meas * 1e9) * 1e6
floor["softmax_stream_bound_us"] = t_sm_stream_us
floor["score_tensor_bytes"] = score_bytes

t_qk = min(r_qkb["median_gpu_us"], r_qkl["median_gpu_us"])
t_pv = r_pvb["median_gpu_us"]
t_sm = r_sm["median_gpu_us"]
matmul_sum = t_qk + t_pv
floor_us = max(matmul_sum, t_sm_stream_us, 0.0)
direct_headroom = t_causal / floor_us
# secondary convention: using the MEASURED softmax chain instead of the
# streaming bound (registered letter: 'time a softmax pass' -> measured)
floor_measured_us = max(matmul_sum, t_sm)
direct_headroom_measured = t_causal / floor_measured_us

RESULTS["floor"] = {
    "matmul_QK_us": t_qk, "matmul_QK_variant_used":
        "QK_broadcast" if r_qkb["median_gpu_us"] <= r_qkl["median_gpu_us"] else "QK_loop32",
    "matmul_QK_broadcast_us": r_qkb["median_gpu_us"],
    "matmul_QK_loop32_us": r_qkl["median_gpu_us"],
    "matmul_PV_us": t_pv, "softmax_measured_us": t_sm,
    "softmax_stream_bound_us": t_sm_stream_us,
    "matmul_sum_us": matmul_sum,
    "floor_us_max_convention": floor_us,
    "floor_us_measured_softmax": floor_measured_us,
    "direct_headroom": direct_headroom,
    "direct_headroom_measured_softmax": direct_headroom_measured,
}
save()

# ------------------------------------------------------------- headline math
verdict = None
if R <= 0.75:
    verdict = "KERNEL EXPLOITS MASK"
elif R >= 0.92:
    verdict = "KERNEL DOES FULL WORK"
else:
    verdict = "PARTIAL"

span_share_pct = 11.8  # cited artifact below
lever = (direct_headroom >= 1.40
         and span_share_pct / 100.0 * (1 - 1 / direct_headroom) >= 0.01)
RESULTS["verdict"] = {
    "R_causal_over_dense": R,
    "t_causal_us": t_causal, "t_dense_us": t_dense, "t_none_us": t_none,
    "dispatches": {k: SDPA[k]["median_dispatches"] for k in SDPA},
    "f_flop_ratio": f_ratio,
    "pre_registered_verdict": verdict,
    "measured_onnode_peak_tflops": peak_best["tflops"],
    "peak_shape": peak_best["shape"], "peak_dtype": peak_best["dtype"],
    "direct_headroom": direct_headroom,
    "direct_headroom_measured_softmax": direct_headroom_measured,
    "lever_gate": {"headroom_ge_140": direct_headroom >= 1.40,
                   "e2e_ge_1pct": bool(span_share_pct / 100.0 * (1 - 1 / direct_headroom) >= 0.01),
                   "verdict": "REAL LEVER (P09 candidate)" if lever
                              else "DENOMINATOR CORRECTED, NO ACTIONABLE LEVER"},
    "span_share_citation": ("docs/dsv4-220k-prefill-span-profile-2026-08-18.md:84 — "
                            "attn.sdpa.compressed 2200 calls, 73396.81 ms total, "
                            "11.8% of prefill wall"),
}
save()

# re-derive headline from raw JSON before printing (integrity rule 9)
with open(OUT / "item1_results.json") as fh:
    chk = json.load(fh)
r2 = chk["verdict"]["R_causal_over_dense"]
h2 = chk["verdict"]["direct_headroom"]
assert abs(r2 - chk["sdpa"]["causal"]["median_gpu_us"]
           / chk["sdpa"]["dense"]["median_gpu_us"]) < 1e-9
assert abs(h2 - chk["sdpa"]["causal"]["median_gpu_us"]
           / chk["floor"]["floor_us_max_convention"]) < 1e-9 or True
chk_floor = max(chk["floor"]["matmul_QK_us"] + chk["floor"]["matmul_PV_us"],
                chk["floor"]["softmax_stream_bound_us"])
assert abs(chk_floor - h2 * 0) >= 0  # noop; real check below
f2 = chk["sdpa"]["causal"]["median_gpu_us"] / chk_floor
log(f"RECHECK from json: R={r2:.4f} headroom={f2:.4f} (should equal "
    f"{h2:.4f})")
assert abs(f2 - h2) < 1e-6, "headroom mismatch after json round-trip"

# ------------------------------------------------------------- summary table
log("\n=== P08 Item 1 SUMMARY ===")
log(f"shape: q (1,{N_HEADS},{L_BAND},{HEAD_DIM}) kv (1,1,{CATTN_KV},{HEAD_DIM}) "
    f"bf16, mask bool (1,1,{L_BAND},{CATTN_KV}) materialized")
log(f"t_causal={t_causal:.2f}us  t_dense={t_dense:.2f}us  t_none={t_none:.2f}us")
log(f"R = t_causal/t_dense = {R:.4f}   -> {verdict}")
log(f"f = causal/dense FLOP (actual mask content) = {f_ratio:.4f}")
log(f"dispatches: causal={SDPA['causal']['median_dispatches']} "
    f"dense={SDPA['dense']['median_dispatches']} none={SDPA['none']['median_dispatches']}")
log(f"on-node GEMM peak: {peak_best['tflops']:.2f} TF at {peak_best['shape']} {peak_best['dtype']}")
log(f"floor: QK {floor['matmul_QK_us']:.1f} + PV {floor['matmul_PV_us']:.1f} = "
    f"{matmul_sum:.1f}us | softmax stream-bound {t_sm_stream_us:.1f}us "
    f"(measured chain {t_sm:.1f}us)")
log(f"floor (max convention) = {floor_us:.1f}us -> direct_headroom = {direct_headroom:.3f}x")
log(f"lever gate: headroom>=1.40 {direct_headroom >= 1.40}, "
    f"e2e>=1.0% {chk['verdict']['lever_gate']['e2e_ge_1pct']}")
log("=== CAPTURE COMPLETE ===")
