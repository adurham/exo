"""Production-class DeepSeek-V4-Flash ATTENTION sub-kernel efficiency benchmark.

Measures achieved-vs-ceiling for the 4 largest attention sub-spans in the
220K-token TP prefill span profile (docs/dsv4-220k-prefill-span-profile-2026-08-18.md):

    attn.proj_qkv          8.9% of prefill wall
    attn.o_proj           10.0%
    attn.sdpa             13.6%  (SparseCompressedAttention, compress_ratio=4)
    attn.sdpa.compressed  11.8%  (CompressedAttention,       compress_ratio=128)

Methodology mirrors bench/moe_production_class_bench.py:
  * real mlx-lm classes (nn.Linear / MultiLinear quantized exactly as
    DeepseekV4 make_quantization_config does -> mxfp8 g=32 b=8 for ALL
    attention weights; the real _sparse_pooled_attention for the sparse SDPA)
  * exact production shapes at TP=2 with EXO_DSV4_SEQ_SPLIT=1 and
    EXO_PREFILL_STEP_SIZE=2048
  * matched-shape dense fp16 GEMM ceiling measured in the SAME session
  * SDPA compared against an analytic roofline (compute + memory bound)

Run:  .venv/bin/python bench/attn_production_class_bench.py
"""

import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mlx-lm"))
from mlx_lm.models.deepseek_v4 import _sparse_pooled_attention  # noqa: E402
from mlx_lm.models.mla import MultiLinear  # noqa: E402

# ─────────────────── production config (DeepSeek-V4-Flash-0731) ───────────────────
HIDDEN = 4096
N_HEADS_FULL = 64
HEAD_DIM = 512
Q_LORA = 1024
O_LORA = 1024
O_GROUPS = 8
INDEX_TOPK = 512
SLIDING_WINDOW = 128
TP = 2

# quantization: make_quantization_config() maps every ".attn.w*" key to mxfp8
QGROUP, QBITS, QMODE = 32, 8, "mxfp8"
DTYPE = mx.bfloat16

# serving regime
PREFILL_CHUNK = 2048          # EXO_PREFILL_STEP_SIZE
CONTEXT = 220_000             # profiled context
SPARSE_SDPA_TILE = 128        # EXO_DSV4_SPARSE_SDPA_TILE

# derived per-TP-rank shapes
N_HEADS = N_HEADS_FULL // TP              # model.shard(): attn.n_heads //= N  -> 32
L_FULL = PREFILL_CHUNK                    # kv / wq_a / wkv side stays FULL
L_BAND = PREFILL_CHUNK // TP              # seq-split v2 query band -> 1024
WQB_OUT = N_HEADS * HEAD_DIM              # wq_b sharded all-to-sharded -> 16384
WOA_IN = N_HEADS * HEAD_DIM // O_GROUPS   # wo_a sharded-to-all       -> 2048

# KV lengths at 220K context
#   local: RotatingKVCache(max_size=sliding_window) fetched during an S-token
#          prefill chunk returns max_size + S - 1 rows
LOCAL_KV = SLIDING_WINDOW + L_FULL - 1                     # 2175
POOL_R128 = math.ceil(CONTEXT / 128)                       # 1719
POOL_R4 = math.ceil(CONTEXT / 4)                           # 55000  (>> topk -> sparse)
CATTN_KV = LOCAL_KV + POOL_R128                            # 3894

WARMUP = 5
ITERS = 20

# span-profile ground truth (docs/dsv4-220k-prefill-span-profile-2026-08-18.md)
SPAN_PROFILE = {
    # span: (pct_of_wall, avg_us_per_call, n_calls)
    "attn.sdpa": (13.6, 13_315.29, 2530),
    "attn.sdpa.compressed": (11.8, 33_362.18, 2200),
    "attn.o_proj": (10.0, 13_072.01, 4730),
    "attn.proj_qkv": (8.9, 11_725.31, 4730),
}
N_CHUNKS = 110                 # 2530/23 == 2200/20 == 4730/43 == 110
N_LAYERS_SPARSE = 21           # compress_ratio == 4
N_LAYERS_COMPRESSED = 20       # compress_ratio == 128
N_LAYERS_LOCAL = 2             # compress_ratio == 0 (also emits attn.sdpa)
N_LAYERS = 43


def bench_call(fn, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()
    samples = []
    for _ in range(iters):
        mx.synchronize()
        t0 = time.perf_counter()
        y = fn()
        mx.eval(y)
        mx.synchronize()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples), samples


def qlinear(in_dims, out_dims):
    m = nn.Linear(in_dims, out_dims, bias=False)
    m.set_dtype(DTYPE)
    m = m.to_quantized(group_size=QGROUP, bits=QBITS, mode=QMODE)
    mx.eval(m.parameters())
    return m


def qmulti(in_dims, out_dims, groups):
    m = MultiLinear(in_dims, out_dims, groups)
    m.set_dtype(DTYPE)
    m = m.to_quantized(group_size=QGROUP, bits=QBITS, mode=QMODE)
    mx.eval(m.parameters())
    return m


def dense_gemm_ceiling(shapes, label):
    """shapes: list of (M, K, N). Returns (median_s, tflops) for fp16 dense."""
    ops = []
    flops = 0
    for M, K, N in shapes:
        a = mx.random.normal((M, K)).astype(mx.float16)
        b = mx.random.normal((K, N)).astype(mx.float16)
        mx.eval(a, b)
        ops.append((a, b))
        flops += 2 * M * K * N
    t, _ = bench_call(lambda: tuple(a @ b for a, b in ops))
    return t, flops / t / 1e12, flops


def measure_peak_gemm():
    """Big square fp16 GEMM: this session's practical dense-GEMM peak."""
    best = 0.0
    for n in (2048, 4096):
        a = mx.random.normal((n, n)).astype(mx.float16)
        b = mx.random.normal((n, n)).astype(mx.float16)
        mx.eval(a, b)
        t, _ = bench_call(lambda: a @ b)
        best = max(best, 2 * n**3 / t / 1e12)
    return best


def measure_bandwidth():
    """Streaming copy bandwidth (read+write) -- the memory-roofline slope."""
    n = 256 * 1024 * 1024 // 2  # 256MB of bf16
    a = mx.random.normal((n,)).astype(mx.bfloat16)
    mx.eval(a)
    t, _ = bench_call(lambda: a + 1.0, warmup=3, iters=10)
    return (a.nbytes * 2) / t / 1e9  # read + write


# ───────────────────────────── kernels under test ─────────────────────────────


def bench_proj_qkv():
    wq_a = qlinear(HIDDEN, Q_LORA)
    wkv = qlinear(HIDDEN, HEAD_DIM)
    wq_b = qlinear(Q_LORA, WQB_OUT)
    q_norm = nn.RMSNorm(Q_LORA, eps=1e-6)
    kv_norm = nn.RMSNorm(HEAD_DIM, eps=1e-6)
    q_norm.set_dtype(DTYPE)
    kv_norm.set_dtype(DTYPE)
    x = mx.random.normal((1, L_FULL, HIDDEN)).astype(DTYPE)
    mx.eval(x, q_norm.parameters(), kv_norm.parameters())

    def fn():
        q_lora = wq_a(x)
        kv_pre = wkv(x)
        q_res = q_norm(q_lora)
        q = wq_b(q_res[:, :L_BAND, :])
        kv = kv_norm(kv_pre)
        return q, kv

    t, s = bench_call(fn)
    shapes = [
        (L_FULL, HIDDEN, Q_LORA),      # wq_a  (full L)
        (L_FULL, HIDDEN, HEAD_DIM),    # wkv   (full L)
        (L_BAND, Q_LORA, WQB_OUT),     # wq_b  (banded L)
    ]
    flops = sum(2 * M * K * N for M, K, N in shapes)
    return t, s, flops, shapes


def bench_o_proj():
    wo_a = qmulti(WOA_IN, O_LORA, O_GROUPS)
    wo_b = qlinear(O_GROUPS * O_LORA, HIDDEN)
    # out after sdpa: (B, H, L_band, D) -> _o_pre_a -> (1, o_groups, L_band, WOA_IN)
    out = mx.random.normal((1, N_HEADS, L_BAND, HEAD_DIM)).astype(DTYPE)
    mx.eval(out)

    def fn():
        y = out.reshape(1, O_GROUPS, -1, L_BAND, HEAD_DIM)
        y = y.transpose(0, 1, 3, 2, 4).flatten(-2)         # _o_pre_a
        y = wo_a(y)
        y = y.transpose(0, 2, 1, 3).flatten(-2)            # _o_pre_b
        return wo_b(y)

    t, s = bench_call(fn)
    shapes = [
        (O_GROUPS * L_BAND, WOA_IN, O_LORA),               # wo_a (batched over groups)
        (L_BAND, O_GROUPS * O_LORA, HIDDEN),               # wo_b
    ]
    flops = sum(2 * M * K * N for M, K, N in shapes)
    return t, s, flops, shapes


def bench_sdpa_compressed(with_mask=True):
    """CompressedAttention: one dense fast-SDPA over local||pooled KV."""
    q = mx.random.normal((1, N_HEADS, L_BAND, HEAD_DIM)).astype(DTYPE)
    kv = mx.random.normal((1, 1, CATTN_KV, HEAD_DIM)).astype(DTYPE)
    sinks = mx.zeros((N_HEADS,), dtype=DTYPE)
    mask = None
    if with_mask:
        mask = mx.random.uniform(shape=(1, 1, L_BAND, CATTN_KV)) > 0.05
    mx.eval(q, kv, sinks, mask if mask is not None else mx.array(0))
    scale = HEAD_DIM**-0.5

    def fn():
        return mx.fast.scaled_dot_product_attention(
            q, kv, kv, scale=scale, mask=mask, sinks=sinks
        )

    t, s = bench_call(fn)
    # QK^T + PV, both (L_band x KV x D) per head
    flops = 2 * 2 * N_HEADS * L_BAND * CATTN_KV * HEAD_DIM
    # KV is shared across heads (MLA, num_key_value_heads=1) -> read once
    kv_bytes = CATTN_KV * HEAD_DIM * 2 * 2   # K and V alias the same tensor here
    q_bytes = N_HEADS * L_BAND * HEAD_DIM * 2
    o_bytes = q_bytes
    mask_bytes = (L_BAND * CATTN_KV) if with_mask else 0
    return t, s, flops, kv_bytes + q_bytes + o_bytes + mask_bytes


def bench_sdpa_sparse(with_mask=True):
    """SparseCompressedAttention: production tiled _sparse_pooled_attention."""
    q = mx.random.normal((1, N_HEADS, L_BAND, HEAD_DIM)).astype(DTYPE)
    local_kv = mx.random.normal((1, 1, LOCAL_KV, HEAD_DIM)).astype(DTYPE)
    pooled = mx.random.normal((1, POOL_R4, HEAD_DIM)).astype(DTYPE)
    topk = mx.random.randint(
        0, POOL_R4, (1, L_BAND, INDEX_TOPK)
    ).astype(mx.int32)
    sinks = mx.zeros((N_HEADS,), dtype=DTYPE)
    lmask = smask = None
    if with_mask:
        lmask = mx.random.uniform(shape=(1, 1, L_BAND, LOCAL_KV)) > 0.05
        smask = mx.random.uniform(shape=(1, 1, L_BAND, INDEX_TOPK)) > 0.05
    mx.eval(q, local_kv, pooled, topk, sinks,
            lmask if lmask is not None else mx.array(0),
            smask if smask is not None else mx.array(0))
    scale = HEAD_DIM**-0.5

    def fn():
        # OPT-11 single gather for the full band, then tile the SDPA
        pooled_flat = pooled.reshape(POOL_R4, HEAD_DIM)
        pg_full = pooled_flat[topk.reshape(-1)].reshape(
            1, L_BAND, INDEX_TOPK, HEAD_DIM
        )
        parts = []
        for s0 in range(0, L_BAND, SPARSE_SDPA_TILE):
            e0 = min(s0 + SPARSE_SDPA_TILE, L_BAND)
            parts.append(
                _sparse_pooled_attention(
                    q[:, :, s0:e0, :],
                    local_kv,
                    pooled,
                    topk[:, s0:e0, :],
                    lmask[..., s0:e0, :] if lmask is not None else None,
                    smask[:, :, s0:e0, :] if smask is not None else None,
                    scale,
                    sinks,
                    pooled_gathered=pg_full[:, s0:e0, :, :],
                )
            )
        return mx.concatenate(parts, axis=2)

    t, s = bench_call(fn)
    # local part: L_band x LOCAL_KV per head; pooled part: L_band x topk per head
    flops = 2 * 2 * N_HEADS * L_BAND * (LOCAL_KV + INDEX_TOPK) * HEAD_DIM
    local_bytes = LOCAL_KV * HEAD_DIM * 2
    # gathered pooled is PER QUERY ROW -- this is the dominant memory term
    gathered_bytes = L_BAND * INDEX_TOPK * HEAD_DIM * 2
    q_bytes = N_HEADS * L_BAND * HEAD_DIM * 2
    mask_bytes = (L_BAND * (LOCAL_KV + INDEX_TOPK)) if with_mask else 0
    return t, s, flops, local_bytes + 2 * gathered_bytes + 2 * q_bytes + mask_bytes


def bench_fused_equivalent_sdpa():
    """ACHIEVABLE ceiling for attn.sdpa: the same FLOPs (local_kv + topk pooled
    rows) issued as ONE mx.fast.scaled_dot_product_attention over a dense
    concatenated KV, instead of the hand-rolled split-softmax chain.

    This is not currently expressible in production at L_q>1 (each query row has
    its own gathered pooled KV), but it bounds what a fused Metal kernel with the
    same arithmetic could reach -- a far more meaningful ceiling than the abstract
    roofline, since it is a real kernel measured on this hardware.
    """
    kve = LOCAL_KV + INDEX_TOPK
    q = mx.random.normal((1, N_HEADS, L_BAND, HEAD_DIM)).astype(DTYPE)
    kv = mx.random.normal((1, 1, kve, HEAD_DIM)).astype(DTYPE)
    sinks = mx.zeros((N_HEADS,), dtype=DTYPE)
    mask = mx.random.uniform(shape=(1, 1, L_BAND, kve)) > 0.05
    mx.eval(q, kv, sinks, mask)
    t, _ = bench_call(
        lambda: mx.fast.scaled_dot_product_attention(
            q, kv, kv, scale=HEAD_DIM**-0.5, mask=mask, sinks=sinks
        )
    )
    flops = 2 * 2 * N_HEADS * L_BAND * kve * HEAD_DIM
    return t, flops / t / 1e12


def main():
    print(f"MLX {mx.__version__}   device={mx.default_device()}")
    print(
        f"prod shapes @TP={TP}, seq_split=1, chunk={PREFILL_CHUNK}, ctx={CONTEXT}:\n"
        f"  n_heads/rank={N_HEADS} head_dim={HEAD_DIM} L_full={L_FULL} "
        f"L_band={L_BAND}\n"
        f"  quant={QMODE} g={QGROUP} b={QBITS}  wq_b_out={WQB_OUT} wo_a_in={WOA_IN}\n"
        f"  local_kv={LOCAL_KV} pool(r=128)={POOL_R128} -> cattn_kv={CATTN_KV}  "
        f"pool(r=4)={POOL_R4} topk={INDEX_TOPK} tile={SPARSE_SDPA_TILE}"
    )

    print("\n=== session hardware baselines ===")
    peak = measure_peak_gemm()
    bw = measure_bandwidth()
    print(f"  dense fp16 GEMM peak (square)      : {peak:6.2f} TFLOPS")
    print(f"  streaming memory bandwidth (r+w)   : {bw:6.1f} GB/s")

    rows = []

    # ---------------- GEMM spans ----------------
    for name, bfn in (("attn.proj_qkv", bench_proj_qkv), ("attn.o_proj", bench_o_proj)):
        t, s, flops, shapes = bfn()
        tf = flops / t / 1e12
        td, tfd, _ = dense_gemm_ceiling(shapes, name)
        print(f"\n[{name}]  shapes(M,K,N)={shapes}")
        print(
            f"  achieved (mxfp8, real classes): {t*1e3:8.3f} ms  {tf:6.2f} TFLOPS"
        )
        print(
            f"  matched-shape DENSE fp16 GEMM : {td*1e3:8.3f} ms  {tfd:6.2f} TFLOPS"
            f"   -> {tf/tfd*100:5.1f}% of matched-dense"
        )
        print(f"  vs session square-GEMM peak   : {tf/peak*100:5.1f}%")
        rows.append(
            dict(span=name, kind="GEMM", achieved_ms=t * 1e3, achieved_tflops=tf,
                 ceiling_ms=td * 1e3, ceiling_tflops=tfd, pct_of_ceiling=tf / tfd * 100,
                 ceiling_type="matched-shape dense fp16 GEMM, same session")
        )

    # ---------------- SDPA spans ----------------
    for name, bfn in (
        ("attn.sdpa.compressed", bench_sdpa_compressed),
        ("attn.sdpa", bench_sdpa_sparse),
    ):
        for with_mask in (True, False):
            t, s, flops, bytes_ = bfn(with_mask=with_mask)
            tf = flops / t / 1e12
            gbs = bytes_ / t / 1e9
            t_compute = flops / (peak * 1e12)
            t_mem = bytes_ / (bw * 1e9)
            t_roof = max(t_compute, t_mem)
            bound = "compute" if t_compute >= t_mem else "memory"
            tag = "with mask" if with_mask else "no mask "
            print(
                f"\n[{name}] ({tag})  L_q={L_BAND} H={N_HEADS} D={HEAD_DIM}"
            )
            print(
                f"  achieved: {t*1e3:8.3f} ms  {tf:6.2f} TFLOPS  {gbs:7.1f} GB/s"
                f"  ({flops/1e9:.1f} GFLOP, {bytes_/1e6:.1f} MB)"
            )
            print(
                f"  roofline: compute-bound {t_compute*1e3:7.3f} ms | "
                f"memory-bound {t_mem*1e3:7.3f} ms -> {bound}-bound "
                f"ceiling {t_roof*1e3:7.3f} ms  -> {t_roof/t*100:5.1f}% of roofline"
            )
            if with_mask:
                extra = {}
                if name == "attn.sdpa":
                    tf_ms, tf_tf = bench_fused_equivalent_sdpa()
                    print(
                        f"  ACHIEVABLE ceiling (same-FLOP single fused "
                        f"mx.fast.sdpa): {tf_ms*1e3:7.3f} ms  {tf_tf:5.2f} TFLOPS"
                        f"  -> {tf_ms/t*100:5.1f}% of fused-equivalent"
                    )
                    extra = dict(fused_equiv_ms=tf_ms * 1e3,
                                 fused_equiv_tflops=tf_tf,
                                 pct_of_fused_equiv=tf_ms / t * 100)
                rows.append(
                    dict(span=name, kind="SDPA", achieved_ms=t * 1e3,
                         achieved_tflops=tf, achieved_gbs=gbs,
                         ceiling_ms=t_roof * 1e3, pct_of_ceiling=t_roof / t * 100,
                         bound=bound,
                         ceiling_type=f"analytic roofline ({bound}-bound), "
                                      f"peak={peak:.2f} TF / bw={bw:.0f} GB/s",
                         **extra)
                )

    # ---------------- cross-check vs span profile ----------------
    print("\n=== CROSS-CHECK: isolated kernel time x calls  vs  span-profile wall ===")
    print("(this laptop = M4 Max 32-core GPU; cluster nodes = M4 Max 40-core "
          "=> scale isolated ms by ~0.8 for a cluster-equivalent estimate)")
    per_span_calls = {
        "attn.proj_qkv": N_LAYERS * N_CHUNKS,
        "attn.o_proj": N_LAYERS * N_CHUNKS,
        "attn.sdpa": (N_LAYERS_SPARSE + N_LAYERS_LOCAL) * N_CHUNKS,
        "attn.sdpa.compressed": N_LAYERS_COMPRESSED * N_CHUNKS,
    }
    for r in rows:
        span = r["span"]
        pct, avg_us, ncalls = SPAN_PROFILE[span]
        iso = r["achieved_ms"]
        iso_scaled = iso * 0.8
        print(
            f"  {span:22s} span avg/call {avg_us/1000:8.3f} ms | isolated "
            f"{iso:8.3f} ms (laptop) / {iso_scaled:8.3f} ms (40-core est) | "
            f"ratio span/isolated = {avg_us/1000/iso_scaled:6.2f}x  "
            f"[{ncalls} calls, {pct}% of wall]"
        )
        r.update(span_avg_ms=avg_us / 1000, span_pct_wall=pct, span_calls=ncalls,
                 isolated_40core_est_ms=iso_scaled,
                 span_over_isolated=avg_us / 1000 / iso_scaled)

    # ---------------- ranked headroom table ----------------
    print("\n=== RANKED LEVERS (headroom_fraction x wall_share) ===")
    print(f"{'span':22s} {'%wall':>6s} {'kind':>5s} {'%ceil':>7s} "
          f"{'headroom':>9s} {'max e2e speedup':>16s} {'lever score':>12s}")
    for r in sorted(rows, key=lambda r: -(1 - r["pct_of_ceiling"] / 100)
                    * r["span_pct_wall"]):
        head = max(0.0, 1 - r["pct_of_ceiling"] / 100)
        # if this span ran at ceiling, its wall share shrinks by pct_of_ceiling
        saved = r["span_pct_wall"] * head / 100
        speedup = 1 / (1 - saved) if saved < 1 else float("inf")
        r.update(headroom=head, max_e2e_speedup=speedup, lever_score=head * r["span_pct_wall"])
        print(
            f"{r['span']:22s} {r['span_pct_wall']:6.1f} {r['kind']:>5s} "
            f"{r['pct_of_ceiling']:7.1f} {head*100:8.1f}% {speedup:15.3f}x "
            f"{head*r['span_pct_wall']:12.2f}"
        )

    outp = Path(__file__).resolve().parent / "attn_production_class_bench_results.json"
    outp.write_text(json.dumps(dict(
        mlx=mx.__version__, peak_tflops=peak, bandwidth_gbs=bw,
        shapes=dict(n_heads=N_HEADS, head_dim=HEAD_DIM, L_full=L_FULL,
                    L_band=L_BAND, local_kv=LOCAL_KV, cattn_kv=CATTN_KV,
                    pool_r4=POOL_R4, topk=INDEX_TOPK),
        rows=rows), indent=2))
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    os.environ.setdefault("EXO_DSV4_SPARSE_FUSED_SDPA", "0")
    main()
