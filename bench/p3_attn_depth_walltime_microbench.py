"""P3 Worker C — real attention-path kernel WALL TIME per single-token decode
step for DeepSeek-V4-Flash, as a function of context depth L.

WHAT THIS MEASURES
------------------
One instance of EACH production attention class (LocalAttention r=0,
SparseCompressedAttention r=4, CompressedAttention r=128) built from the REAL
mlx-lm fork classes at the REAL production config (config.json of
deepseek-ai--DeepSeek-V4-Flash-0731) with production quantization
(make_quantization_config rules: mxfp8 g=32 b=8 for ``.attn.w*`` and
``.attn.indexer.wq*``; affine g=64 b=8 elsewhere), driven with a synthetic
pre-filled KV / compressor-pool / indexer-pool cache at depth L, then stepped
through N consecutive REAL single-token (B=1, L_q=1) decode calls.

Per-layer-class median ms is then multiplied by the production layer census
(2 x r=0, 21 x r=4, 20 x r=128 -- from config.compress_ratios[:43]) to get a
whole-model attention-path ms/token.

WHY ONE LAYER PER CLASS, SCALED
-------------------------------
Memory. 43 instantiated attention blocks would be ~5.3 GB of weights, and the
studio nodes are running the live 85 GB model. One block per class is ~370 MB
of weights + ~170 MB of synthetic cache at L=500K. Attention layers are
independent and identical within a class, so per-class median x count is exact
up to inter-layer scheduling overlap (which this bench, being serialized by
mx.synchronize fences, does not capture -- see LIMITATIONS in the doc).

N consecutive steps matter: the Compressor only emits a pooled entry every
``ratio`` decode steps (1-in-4 for sparse layers, 1-in-128 for compressed
layers) and PoolingCache reallocs every 256 pooled entries, so the average over
>= 256 steps amortizes both correctly, exactly as production does.

TP NOTE: exo replicates attention on both ranks (only MoE is sharded --
src/exo/worker/engines/mlx/auto_parallel.py:1032-1034), and seq-split needs
L_q >= 16, so single-token decode does no attention sharding at all. A
single-process microbench therefore measures the full per-rank attention work.

Run:  .venv/bin/python bench/p3_attn_depth_walltime_microbench.py --depths 520,100026,352599,500000
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

# ── production env config (start_cluster.sh defaults) — MUST precede import ──
_PROD_ENV = {
    "EXO_DSV4_INDEX_TOPK": "512",          # start_cluster.sh:33
    "EXO_KV_CACHE_BITS": "0",              # start_cluster.sh:151 (bf16 KV)
    "EXO_COMPUTE_DTYPE": "bf16",           # start_cluster.sh:153
    "EXO_DSV4_SPARSE_SDPA_TILE": "128",    # start_cluster.sh:103
    "EXO_DSV4_SEQ_SPLIT": "1",             # start_cluster.sh:108
    "EXO_DSV4_EXACT_TOPK": "1",            # deepseek_v4.py:3426 default
    "EXO_DSV4_TOPK_FUSED": "0",            # deepseek_v4.py:3893 default
    "EXO_DSV4_SPARSE_FUSED_SDPA": "0",     # deepseek_v4.py:1716 default
    "EXO_DSV4_ATTN_ALLSUM": "0",           # start_cluster.sh:1755
    "EXO_DSV4_SINGLE_GATHER": "1",         # deepseek_v4.py:4684 default
    "EXO_DSV4_POOL_DEFER_COPY_MAX_BYTES": "8388608",   # start_cluster.sh:2069
    "EXO_DSV4_PREFILL_ARGPARTITION": "1",  # start_cluster.sh:528 (prefill only)
    "EXO_DSV4_ARGPARTITION_MIN_P": "8192",
}
for _k, _v in _PROD_ENV.items():
    os.environ[_k] = _v
# explicitly UNSET the ones production leaves unset
for _k in ("EXO_DSV4_INDEXER_PBLOCK", "EXO_DSV4_QA_KV_FUSED", "EXO_DSV4_FP32_ACT",
           "EXO_DSV4_MTP", "EXO_PROFILER", "EXO_DSV4_SECTION_TIME"):
    os.environ.pop(_k, None)

import mlx.core as mx           # noqa: E402
import mlx.nn as nn             # noqa: E402

_REPO = Path(__file__).resolve().parent.parent
_MLXLM = _REPO / "mlx-lm"
if not _MLXLM.is_dir():          # running from /tmp on a studio node
    _MLXLM = Path.home() / "repos" / "exo" / "mlx-lm"
sys.path.insert(0, str(_MLXLM))

from mlx_lm.models import deepseek_v4 as dv4              # noqa: E402
from mlx_lm.models.cache import (                          # noqa: E402
    CacheList, PoolingCache, RotatingKVCache,
)

# ── production config (config.json, DeepSeek-V4-Flash-0731) ──
COMPRESS_RATIOS = (
    [0, 0] + [4, 128] * 20 + [4]          # first 43 entries, verified from config.json
)
assert len(COMPRESS_RATIOS) == 43
assert COMPRESS_RATIOS.count(0) == 2
assert COMPRESS_RATIOS.count(4) == 21
assert COMPRESS_RATIOS.count(128) == 20

CFG = dict(
    model_type="deepseek_v4",
    vocab_size=129280,
    hidden_size=4096,
    intermediate_size=18432,
    moe_intermediate_size=2048,
    num_hidden_layers=43,
    num_attention_heads=64,
    num_key_value_heads=1,
    n_shared_experts=1,
    n_routed_experts=256,
    num_experts_per_tok=6,
    head_dim=512,
    index_head_dim=128,
    index_n_heads=64,
    index_topk=512,
    o_groups=8,
    o_lora_rank=1024,
    q_lora_rank=1024,
    qk_rope_head_dim=64,
    sliding_window=128,
    max_position_embeddings=1048576,
    rms_norm_eps=1e-6,
    rope_theta=10000,
    compress_rope_theta=160000,
    rope_scaling=dict(beta_fast=32, beta_slow=1, factor=16,
                      original_max_position_embeddings=65536, type="yarn"),
    routed_scaling_factor=1.5,
    scoring_func="sqrtsoftplus",
    topk_method="noaux_tc",
    norm_topk_prob=True,
    attention_bias=False,
    compress_ratios=COMPRESS_RATIOS,
    num_nextn_predict_layers=1,
    hidden_act="silu",
    swiglu_limit=10.0,
    tie_word_embeddings=False,
)

DTYPE = mx.bfloat16
HEAD_DIM = 512
INDEX_DIM = 128
SW = 128

LAYER_CENSUS = {0: 2, 4: 21, 128: 20}
# representative layer index for each class (must have the matching ratio)
LAYER_IDX = {0: 0, 4: 2, 128: 3}


def build_args():
    return dv4.ModelArgs.from_dict(CFG)


def _quant_predicate(path, module):
    """Replicates make_quantization_config() (deepseek_v4.py:899-931) for a
    module rooted at ``model.layers.<i>.attn``. ``path`` here is relative to the
    attention module, so prefix it to match the production key test."""
    if not hasattr(module, "to_quantized"):
        return False
    full = "model.layers.0.attn." + path if path else "model.layers.0.attn"
    if ".attn.w" in full or ".attn.indexer.wq" in full:
        return {"group_size": 32, "bits": 8, "mode": "mxfp8"}       # mxfp8
    return {"group_size": 64, "bits": 8, "mode": "affine"}          # default


def build_attn(ratio):
    args = build_args()
    attn = dv4.v4_attention_factory(args, LAYER_IDX[ratio])
    attn.set_dtype(DTYPE)
    nn.quantize(attn, class_predicate=_quant_predicate)
    mx.eval(attn.parameters())
    return attn


# ─────────────────── synthetic cache pre-fill at depth L ───────────────────

def _fill_rotating(rc: RotatingKVCache, L: int, B=1, D=HEAD_DIM):
    """Put a RotatingKVCache into steady-state rotation at absolute offset L.

    Production shape: keys (B, 1, max_size, D) bf16, values (B, 1, max_size, 0)
    (DSv4 passes a zero-width values placeholder — _zero_values). At L >> 128
    the buffer is exactly max_size wide and _idx has wrapped.
    """
    rc.keys = mx.random.normal((B, 1, rc.max_size, D)).astype(DTYPE)
    rc.values = mx.zeros((B, 1, rc.max_size, 0), dtype=DTYPE)
    rc.offset = L
    rc._idx = rc.max_size          # forces the rotate branch on the next update
    mx.eval(rc.keys, rc.values)


def _fill_pool(pc: PoolingCache, L: int, dim: int, B=1):
    """Pre-fill a PoolingCache to P = L // ratio entries with a realistic
    step-allocated storage buffer and a realistic partial remainder."""
    P = L // pc.ratio
    # storage is allocated in `step` chunks; mimic that exactly
    alloc = max(pc.step, ((P + 1 + pc.step - 1) // pc.step) * pc.step)
    pc._pool_storage = mx.random.normal((B, alloc, dim)).astype(DTYPE)
    pc._pool_offset = P
    pc._pending_offset_bump = 0
    # remainder: tokens seen since the last full window
    rem = L % pc.ratio
    out_dim = dim * (2 if pc.ratio == 4 else 1)
    if rem:
        pc.buf_kv = mx.random.normal((B, pc.ratio, out_dim)).astype(DTYPE)
        pc.buf_gate = mx.random.normal((B, pc.ratio, out_dim)).astype(DTYPE)
        pc.remainder = rem
    if pc.ratio == 4:   # overlap layers carry a cross-call window tail
        half = out_dim // 2
        pc._overlap_kv_carry = mx.random.normal((B, 1, pc.ratio, half)).astype(DTYPE)
        pc._overlap_gate_carry = mx.random.normal((B, 1, pc.ratio, half)).astype(DTYPE)
    mx.eval(pc._pool_storage)
    if pc.buf_kv is not None:
        mx.eval(pc.buf_kv, pc.buf_gate)
    if pc._overlap_kv_carry is not None:
        mx.eval(pc._overlap_kv_carry, pc._overlap_gate_carry)
    return P


def make_cache(ratio, L):
    """Mirror Model.make_cache (deepseek_v4.py:6956-6979) for one layer."""
    rc = RotatingKVCache(max_size=SW)
    _fill_rotating(rc, L)
    if ratio == 0:
        return rc, dict(P_comp=0, P_idx=0)
    comp = PoolingCache(ratio)
    P_comp = _fill_pool(comp, L, HEAD_DIM)
    if ratio == 4:
        idx = PoolingCache(ratio)
        P_idx = _fill_pool(idx, L, INDEX_DIM)
        return CacheList(rc, comp, idx), dict(P_comp=P_comp, P_idx=P_idx)
    return CacheList(rc, comp), dict(P_comp=P_comp, P_idx=0)


def cache_bytes(ratio, L):
    b = SW * HEAD_DIM * 2
    if ratio == 4:
        b += (L // 4) * HEAD_DIM * 2 + (L // 4) * INDEX_DIM * 2
    elif ratio == 128:
        b += (L // 128) * HEAD_DIM * 2
    return b


# ───────────────────────────── timing ─────────────────────────────

def time_decode_steps(attn, cache, steps, warmup, chain=1, mode="async"):
    """Run consecutive REAL single-token decode steps; return per-step ms.

    ``chain`` controls the fencing discipline, which matters enormously:

      chain=1  -- one mx.synchronize + mx.eval fence around EVERY step. This is
                  the cleanest attribution but SERIALIZES CPU dispatch against
                  GPU execution, so each sample carries a full command-buffer
                  round-trip (~0.5 ms on this hardware). Production does NOT
                  run this way.
      chain=K  -- K consecutive decode steps issued back-to-back with a single
                  fence pair around the group, then divided by K. The lazy
                  graph pipelines, but K live pool-storage views coexist, which
                  DEFEATS PoolingCache's buffer-donation optimization
                  (cache.py:1547-1556) and silently turns the pool write into
                  an O(P*D) copy at depth. Results grow with K. Use only as a
                  sensitivity probe.
      mode="async" -- K steps issued back-to-back but each one's output passed
                  to mx.async_eval immediately. This is what production does
                  (EXO_DSV4_FENCE_EVERY_N_LAYERS=4 + async fence, and the
                  explicit mx.async_eval(self._pool_storage) at cache.py:1551):
                  the GPU pipeline stays full, but the previous step's graph is
                  already submitted so no stale view is held and donation
                  succeeds. THIS IS THE PRODUCTION-FAITHFUL NUMBER.
    """
    B, Lq, H = 1, 1, CFG["hidden_size"]
    x = mx.random.normal((B, Lq, H)).astype(DTYPE)
    mx.eval(x)

    for _ in range(warmup):
        out = attn(x, mask=None, cache=cache)
        mx.eval(out)
    mx.synchronize()

    samples = []
    groups = max(1, steps // chain)
    for _ in range(groups):
        mx.synchronize()
        t0 = time.perf_counter()
        if mode == "async":
            last = None
            for _ in range(chain):
                last = attn(x, mask=None, cache=cache)
                mx.async_eval(last)
            mx.eval(last)
        else:
            outs = []
            for _ in range(chain):
                outs.append(attn(x, mask=None, cache=cache))
            mx.eval(outs)
        mx.synchronize()
        samples.append((time.perf_counter() - t0) * 1e3 / chain)
    return samples


def measure_fence_floor(reps=200):
    """Cost of one mx.eval+mx.synchronize round trip on a trivial op — the
    per-fence overhead that chain=1 adds to every single sample."""
    a = mx.random.normal((16,)).astype(DTYPE)
    mx.eval(a)
    for _ in range(20):
        mx.eval(a + 1.0)
    mx.synchronize()
    ts = []
    for _ in range(reps):
        mx.synchronize()
        t0 = time.perf_counter()
        mx.eval(a + 1.0)
        mx.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def measure_bandwidth():
    n = 256 * 1024 * 1024 // 2
    a = mx.random.normal((n,)).astype(mx.bfloat16)
    mx.eval(a)
    for _ in range(3):
        mx.eval(a + 1.0)
    mx.synchronize()
    ts = []
    for _ in range(10):
        mx.synchronize()
        t0 = time.perf_counter()
        mx.eval(a + 1.0)
        mx.synchronize()
        ts.append(time.perf_counter() - t0)
    t = statistics.median(ts)
    return (a.nbytes * 2) / t / 1e9


# ───────── component microbench: indexer score GEMM + exact top-k ─────────

def bench_indexer_components(L, steps=64, warmup=16, chain=16):
    """Isolate the three depth-dependent indexer/gather kernels at exact decode
    shapes. Uses the SAME chained-fence discipline as the layer bench so the
    numbers are comparable (a per-call fence would add ~0.17 ms to each)."""
    P = L // 4
    B, H, Lq, D = 1, CFG["index_n_heads"], 1, INDEX_DIM
    q = mx.random.normal((B, H, Lq, D)).astype(DTYPE)
    pooled = mx.random.normal((B, P, D)).astype(DTYPE)
    wx = mx.random.normal((B, Lq, H)).astype(DTYPE)
    scale = D ** -0.5
    inv = H ** -0.5
    mx.eval(q, pooled, wx)

    def _chained(fn):
        for _ in range(warmup):
            mx.eval(fn())
        mx.synchronize()
        out = []
        for _ in range(max(1, steps // chain)):
            mx.synchronize()
            t0 = time.perf_counter()
            outs = [fn() for _ in range(chain)]
            mx.eval(outs)
            mx.synchronize()
            out.append((time.perf_counter() - t0) * 1e3 / chain)
        return statistics.median(out)

    score_ms = _chained(lambda: dv4._indexer_score(q, pooled, wx, scale, inv))

    scores = dv4._indexer_score(q, pooled, wx, scale, inv)
    mx.eval(scores)
    k = min(512, P)
    topk_ms = _chained(lambda: dv4._exact_topk(scores, k))

    # KV gather at exact decode shape (OPT-10 reshape+gather)
    pooled_kv = mx.random.normal((B, P, HEAD_DIM)).astype(DTYPE)
    topk = mx.random.randint(0, P, (B, Lq, k)).astype(mx.int32)
    mx.eval(pooled_kv, topk)

    def gather():
        pf = pooled_kv.reshape(B * P, HEAD_DIM)
        off = (mx.arange(B) * P).reshape(B, 1, 1)
        return pf[(topk + off).reshape(-1)].reshape(B, Lq, k, HEAD_DIM)

    gather_ms = _chained(gather)

    # core SDPA at the sparse-decode shape: 128 local rows + k gathered rows
    qs = mx.random.normal((B, CFG["num_attention_heads"], Lq, HEAD_DIM)).astype(DTYPE)
    kvc = mx.random.normal((B, 1, SW + k, HEAD_DIM)).astype(DTYPE)
    sinks = mx.zeros((CFG["num_attention_heads"],), dtype=DTYPE)
    mx.eval(qs, kvc, sinks)
    sdpa_ms = _chained(lambda: mx.fast.scaled_dot_product_attention(
        qs, kvc, kvc, scale=HEAD_DIM ** -0.5, mask=None, sinks=sinks))

    # compressed-class SDPA: 128 local rows + L/128 pooled rows (grows with L)
    Pc = L // 128
    kvd = mx.random.normal((B, 1, SW + Pc, HEAD_DIM)).astype(DTYPE)
    mx.eval(kvd)
    sdpa_c_ms = _chained(lambda: mx.fast.scaled_dot_product_attention(
        qs, kvd, kvd, scale=HEAD_DIM ** -0.5, mask=None, sinks=sinks))

    return dict(P=P, k=k, P_c=Pc,
                score_ms=score_ms, topk_ms=topk_ms, gather_ms=gather_ms,
                sdpa_sparse_ms=sdpa_ms, sdpa_compressed_ms=sdpa_c_ms,
                score_bytes=P * D * 2,
                topk_bytes=4 * P * 2,
                sdpa_c_bytes=(SW + Pc) * HEAD_DIM * 2)


# ───────────────────────────── main ─────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", default="520,100026,352599,500000")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--chain", type=int, default=16,
                    help="decode steps issued per fence group (production-like "
                         "pipelining); 1 = fully serialized")
    ap.add_argument("--section-time", action="store_true",
                    help="enable the fork's _ATTN_SUB_ACC per-sub-span fences "
                         "(sparse class only); perturbs totals, run separately")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    depths = [int(d) for d in args.depths.split(",")]
    chain = args.chain

    print(f"MLX {mx.__version__}  device={mx.default_device()}")
    print(f"host={os.uname().nodename}")
    print("env:", json.dumps(_PROD_ENV))
    print(f"layer census: r=0 x{LAYER_CENSUS[0]}, r=4 x{LAYER_CENSUS[4]}, "
          f"r=128 x{LAYER_CENSUS[128]}  (43 total)")
    print(f"steps/depth/class = {args.steps} (warmup {args.warmup}), chain={chain}")

    bw = measure_bandwidth()
    fence = measure_fence_floor()
    print(f"session streaming bandwidth (r+w): {bw:.1f} GB/s")
    print(f"per-fence round-trip floor (mx.eval+synchronize on a 16-elem op): "
          f"{fence:.4f} ms")

    results = {}
    for L in depths:
        print(f"\n{'='*72}\n=== DEPTH L = {L:,} ===")
        per_class = {}
        for ratio in (0, 4, 128):
            attn = build_attn(ratio)
            cache, meta = make_cache(ratio, L)
            s_f = time_decode_steps(attn, cache, args.steps, args.warmup,
                                    chain=1, mode="sync")
            s_c = time_decode_steps(attn, cache, args.steps, args.warmup,
                                    chain=chain, mode="async")
            s_h = time_decode_steps(attn, cache, args.steps, args.warmup,
                                    chain=chain, mode="sync")
            med_f = statistics.median(s_f)
            med_c = statistics.median(s_c)
            med_h = statistics.median(s_h)
            per_class[ratio] = dict(
                fenced_median_ms=med_f,
                chained_median_ms=med_c,          # async == production-faithful
                heldview_median_ms=med_h,         # held-view sensitivity probe
                chained_mean_ms=statistics.mean(s_c),
                chained_p10_ms=statistics.quantiles(s_c, n=10)[0]
                if len(s_c) >= 10 else med_c,
                chained_p90_ms=statistics.quantiles(s_c, n=10)[8]
                if len(s_c) >= 10 else med_c,
                chained_stdev_ms=statistics.pstdev(s_c),
                n_fenced=len(s_f), n_chained=len(s_c), **meta,
            )
            print(f"  r={ratio:<4d} ({LAYER_CENSUS[ratio]:2d} layers)  "
                  f"async(x{chain}) {med_c:7.4f} ms | heldview(x{chain}) "
                  f"{med_h:7.4f} | fenced {med_f:7.4f}  "
                  f"P_comp={meta['P_comp']:,} P_idx={meta['P_idx']:,}")
            del attn, cache
            mx.clear_cache()
            print(f"        peak GPU mem so far: "
                  f"{mx.get_peak_memory()/1e9:.2f} GB")

        total_f = sum(per_class[r]["fenced_median_ms"] * LAYER_CENSUS[r]
                      for r in (0, 4, 128))
        total = sum(per_class[r]["chained_median_ms"] * LAYER_CENSUS[r]
                    for r in (0, 4, 128))
        contrib = {r: per_class[r]["chained_median_ms"] * LAYER_CENSUS[r]
                   for r in (0, 4, 128)}
        print(f"  --> 43-layer attention-path total (CHAINED, production-like): "
              f"{total:8.3f} ms/token")
        print(f"      (fenced/serialized upper bound: {total_f:8.3f} ms/token)")
        print(f"      breakdown: r=0 {contrib[0]:7.3f} | r=4 {contrib[4]:7.3f} "
              f"| r=128 {contrib[128]:7.3f} ms")

        comp = bench_indexer_components(L)
        print(f"  components (chained) @P_idx={comp['P']:,} k={comp['k']} "
              f"P_c={comp['P_c']:,}:")
        print(f"    indexer.score  {comp['score_ms']:7.4f} ms  x21 = "
              f"{21*comp['score_ms']:7.3f} ms")
        print(f"    indexer.topk   {comp['topk_ms']:7.4f} ms  x21 = "
              f"{21*comp['topk_ms']:7.3f} ms")
        print(f"    kv gather      {comp['gather_ms']:7.4f} ms  x21 = "
              f"{21*comp['gather_ms']:7.3f} ms")
        print(f"    sdpa sparse    {comp['sdpa_sparse_ms']:7.4f} ms  x21 = "
              f"{21*comp['sdpa_sparse_ms']:7.3f} ms")
        print(f"    sdpa compress  {comp['sdpa_compressed_ms']:7.4f} ms  x20 = "
              f"{20*comp['sdpa_compressed_ms']:7.3f} ms")
        score_gbs = comp["score_bytes"] / (comp["score_ms"] / 1e3) / 1e9
        topk_gbs = comp["topk_bytes"] / (comp["topk_ms"] / 1e3) / 1e9
        sdpac_gbs = comp["sdpa_c_bytes"] / (comp["sdpa_compressed_ms"] / 1e3) / 1e9
        print(f"  achieved GB/s on the depth-dependent reads: "
              f"indexer-score {score_gbs:7.1f}  exact-topk {topk_gbs:7.1f}  "
              f"sdpa-compressed {sdpac_gbs:7.1f}   (session peak {bw:.0f})")
        score_gbs_nf = topk_gbs_nf = float("nan")

        results[L] = dict(per_class=per_class, total_ms=total,
                          total_fenced_ms=total_f,
                          contrib=contrib, components=comp,
                          score_gbs=score_gbs, topk_gbs=topk_gbs,
                          score_gbs_nofence=score_gbs_nf,
                          topk_gbs_nofence=topk_gbs_nf,
                          cache_bytes_43=sum(cache_bytes(r, L) * n
                                             for r, n in LAYER_CENSUS.items()))

    # ── scaling fit ──
    print(f"\n{'='*72}\n=== SCALING ===")
    print(f"{'L':>10s} {'attn ms/tok':>12s} {'delta vs prev':>14s} "
          f"{'ms per 100K':>12s}")
    prev = None
    for L in depths:
        t = results[L]["total_ms"]
        if prev is None:
            print(f"{L:>10,} {t:12.3f} {'-':>14s} {'-':>12s}")
        else:
            dL = L - prev[0]
            dt = t - prev[1]
            print(f"{L:>10,} {t:12.3f} {dt:14.3f} {dt/dL*1e5:12.3f}")
        prev = (L, t)

    deep = [L for L in depths if L >= 50_000]
    if len(deep) >= 2:
        n = len(deep)
        sx = sum(deep); sy = sum(results[L]["total_ms"] for L in deep)
        sxx = sum(L * L for L in deep)
        sxy = sum(L * results[L]["total_ms"] for L in deep)
        slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
        icept = (sy - slope * sx) / n
        print(f"\nlinear fit over L>=50K: ms = {icept:.4f} + {slope*1e5:.4f} "
              f"per 100K tokens")
        for L in deep:
            pred = icept + slope * L
            act = results[L]["total_ms"]
            print(f"  L={L:>9,}  actual {act:8.3f}  fit {pred:8.3f}  "
                  f"resid {act-pred:+8.4f} ms ({(act-pred)/act*100:+6.2f}%)")

    if args.out:
        Path(args.out).write_text(json.dumps(
            dict(host=os.uname().nodename, mlx=mx.__version__, bandwidth_gbs=bw,
                 env=_PROD_ENV, steps=args.steps,
                 results={str(k): v for k, v in results.items()}), indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
