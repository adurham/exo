"""DSv4 prefill-scale fused-SDPA feasibility microbench.

Per Fable gate-sequence step 2 (2026-07-14): traffic-aware microbench at
PREFILL shape to split the 28.97% NOP-lift ceiling into:
  - matmul share (4 GEMMs — a fused kernel can fold these via online-softmax tiling)
  - overhead share (logsumexp + logaddexp + split_softmax + transposes —
    a fused kernel eliminates by construction)

Also measures intermediate memory traffic: at L_q=2048 the score matrices are
(B, H, L_q, sw+k) bf16. On M4 Max ~546 GB/s the fusion win is eliminating the
round-trips, not the exp/logsumexp math. So we compute bytes_moved per op and
compare to the measured time × bandwidth.

Shape matches _sparse_pooled_attention_inner production prefill path:
  q_scaled:  (B, H, L_q, D)      = (1, 64, 2048, 512) bf16
  local_kv:  (B, 1, sw, D)       = (1, 1, 128, 512)  bf16
  pooled_g:  (B, 1, L_q, k, D)  = (1, 1, 2048, 512, 512) bf16  [gathered]
  local_scores:  (B, H, L_q, sw)  = (1, 64, 2048, 128)
  pooled_scores: (B, H, L_q, k)   = (1, 64, 2048, 512)

Ops in the inner kernel (deepseek_v4.py:1292-1313):
  1. local_scores  = q_scaled @ local_kv.swapaxes(-1,-2)     [matmul A]
  2. local_scores  = _apply_score_mask(local_scores, lmask)   [where/add]
  3. normalizer    = logsumexp(local_scores, -1, keepdims)    [reduce]
  4. pooled_sq     = pooled_gathered.squeeze(1)              [view]
  5. q_bl          = q_scaled.transpose(0,2,1,3)              [transpose]
  6. pooled_scores = q_bl @ pooled_sq.swapaxes(-1,-2)         [matmul B]
  7. pooled_scores = pooled_scores.transpose(0,2,1,3)        [transpose]
  8. pooled_scores = _apply_score_mask(pooled_scores, pmask) [where/add]
  9. normalizer    = logaddexp(normalizer, logsumexp(pooled_scores,-1,kd)) [reduce+add]
 10. local_w, pooled_w = _split_softmax(normalizer, ls, ps, sinks) [exp×3]
 11. out           = local_w @ local_kv                       [matmul C]
 12. pw_bl         = pooled_weights.transpose(0,2,1,3)        [transpose]
 13. out           = out + (pw_bl @ pooled_sq).transpose(0,2,1,3) [matmul D + add]

Usage:
  cd ~/repos/exo && uv run python bench/sdpa_prefill_fused_microbench.py
"""

from __future__ import annotations

import statistics
import time

import mlx.core as mx

# Production prefill shape
B = 1
H = 64
L_Q = 2048
D = 512
SW = 128
K_SEL = 512
SCALE = D ** -0.5
N_LAYERS = 21  # sparse layers in DSv4
N_ITERS = 50
N_WARMUP = 10
DTYPE = mx.bfloat16


def make_inputs(seed: int):
    mx.random.seed(seed)
    q_scaled = mx.random.normal((B, H, L_Q, D), dtype=DTYPE) * SCALE
    local_kv = mx.random.normal((B, 1, SW, D), dtype=DTYPE)
    # pooled_gathered: (B, 1, L_q, k, D) — the gathered pooled KV
    pooled_gathered = mx.random.normal((B, 1, L_Q, K_SEL, D), dtype=DTYPE)
    # bool masks (production uses bool for the fused path; inner takes either)
    local_mask = mx.ones((B, 1, L_Q, SW), dtype=mx.bool_)
    pooled_mask = mx.ones((B, H, L_Q, K_SEL), dtype=mx.bool_)
    # sinks_expanded: (1, H, 1, 1) — the form _sparse_pooled_attention passes
    sinks = mx.random.normal((H,), dtype=DTYPE) * 0.1
    sinks_expanded = sinks[None, :, None, None]
    return q_scaled, local_kv, pooled_gathered, local_mask, pooled_mask, sinks_expanded


# ─────────────── Production inner kernel (reference) ───────────────
def _apply_score_mask(scores, mask):
    if mask is None:
        return scores
    if mask.dtype == mx.bool_:
        return mx.where(mask, scores, mx.finfo(scores.dtype).min)
    return scores + mask.astype(scores.dtype)


def _split_softmax(log_normalizer, logits_a, logits_b, sinks=None):
    if sinks is not None:
        log_normalizer = mx.logaddexp(log_normalizer, sinks)
    weights_a = mx.exp(logits_a - log_normalizer)
    weights_b = mx.exp(logits_b - log_normalizer)
    return weights_a, weights_b


def inner_full(q_scaled, local_kv, pooled_gathered, local_mask, pooled_mask, sinks_expanded):
    """Exact replica of _sparse_pooled_attention_inner. sinks_expanded=(1,H,1,1)."""
    local_scores = q_scaled @ local_kv.swapaxes(-1, -2)
    local_scores = _apply_score_mask(local_scores, local_mask)
    normalizer = mx.logsumexp(local_scores, -1, keepdims=True)

    pooled_sq = pooled_gathered.squeeze(1)
    q_bl = q_scaled.transpose(0, 2, 1, 3)
    pooled_scores = q_bl @ pooled_sq.swapaxes(-1, -2)
    pooled_scores = pooled_scores.transpose(0, 2, 1, 3)
    pooled_scores = _apply_score_mask(pooled_scores, pooled_mask)
    normalizer = mx.logaddexp(
        normalizer, mx.logsumexp(pooled_scores, -1, keepdims=True)
    )

    local_weights, pooled_weights = _split_softmax(
        normalizer, local_scores, pooled_scores, sinks_expanded
    )

    out = local_weights @ local_kv
    pw_bl = pooled_weights.transpose(0, 2, 1, 3)
    out = out + (pw_bl @ pooled_sq).transpose(0, 2, 1, 3)
    return out.astype(q_scaled.dtype)


# ─────────────── Op-isolation timing ───────────────
def time_op(fn, *args, n_iters=N_ITERS, n_warmup=N_WARMUP):
    """Time a single op with forced eval + sync."""
    for _ in range(n_warmup):
        out = fn(*args)
        mx.eval(out)
        mx.synchronize()
    samples = []
    for _ in range(n_iters):
        mx.synchronize()
        t0 = time.perf_counter()
        out = fn(*args)
        mx.eval(out)
        mx.synchronize()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples) * 1e6  # µs


def time_chain(fn, args_list, n_iters=N_ITERS, n_warmup=N_WARMUP):
    """Pipelined: queue N_LAYERS calls, one eval at end (captures chain-level effects)."""
    def _chain():
        return [fn(*a) for a in args_list]

    for _ in range(n_warmup):
        outs = _chain()
        mx.eval(*outs)
        mx.synchronize()
    samples = []
    for _ in range(n_iters):
        mx.synchronize()
        t0 = time.perf_counter()
        outs = _chain()
        mx.eval(*outs)
        mx.synchronize()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples) * 1e6  # µs total


def bytes_of(arr):
    return arr.size * arr.itemsize


def fmt_bytes(n):
    if n >= 1e9:
        return f"{n/1e9:.2f} GB"
    if n >= 1e6:
        return f"{n/1e6:.2f} MB"
    if n >= 1e3:
        return f"{n/1e3:.2f} KB"
    return f"{n} B"


def main():
    print("=" * 78)
    print("DSv4 prefill fused-SDPA feasibility microbench")
    print(f"  Shape: B={B} H={H} L_q={L_Q} D={D} sw={SW} k_sel={K_SEL}")
    print(f"  Layers: {N_LAYERS}, iters: {N_ITERS}, dtype: {DTYPE}")
    print(f"  Device: {mx.default_device()}")
    print("=" * 78)
    print()

    args = make_inputs(42)
    q_scaled, local_kv, pooled_gathered, local_mask, pooled_mask, sinks_expanded = args
    mx.eval(*args)

    # Derived intermediates (for shape/bytes reporting)
    local_scores_shape = (B, H, L_Q, SW)
    pooled_scores_shape = (B, H, L_Q, K_SEL)
    local_weights_shape = local_scores_shape
    out_shape = (B, H, L_Q, D)

    print("Intermediate tensor sizes (bf16):")
    print(f"  local_scores:  {local_scores_shape} = {fmt_bytes(bytes_of(mx.zeros(local_scores_shape, dtype=DTYPE)))}")
    print(f"  pooled_scores: {pooled_scores_shape} = {fmt_bytes(bytes_of(mx.zeros(pooled_scores_shape, dtype=DTYPE)))}")
    print(f"  local_weights: {local_weights_shape} = {fmt_bytes(bytes_of(mx.zeros(local_weights_shape, dtype=DTYPE)))}")
    print(f"  out:           {out_shape} = {fmt_bytes(bytes_of(mx.zeros(out_shape, dtype=DTYPE)))}")
    pooled_sq = pooled_gathered.squeeze(1)
    print(f"  pooled_sq:     {pooled_sq.shape} = {fmt_bytes(bytes_of(pooled_sq))}")
    print()

    # ─────────────── Op-isolation timing ───────────────
    print("=" * 78)
    print("Op-isolation timing (median µs, forced eval+sync):")
    print("=" * 78)

    # Pre-compute intermediates needed by later ops
    local_scores = q_scaled @ local_kv.swapaxes(-1, -2)
    local_scores = _apply_score_mask(local_scores, local_mask)
    normalizer = mx.logsumexp(local_scores, -1, keepdims=True)
    q_bl = q_scaled.transpose(0, 2, 1, 3)
    pooled_scores = q_bl @ pooled_sq.swapaxes(-1, -2)
    pooled_scores = pooled_scores.transpose(0, 2, 1, 3)
    pooled_scores = _apply_score_mask(pooled_scores, pooled_mask)
    mx.eval(local_scores, normalizer, pooled_scores)
    mx.synchronize()

    results = {}

    # 1. Matmul A: local_scores = q_scaled @ local_kv.swapaxes
    t = time_op(lambda q, kv: q @ kv.swapaxes(-1, -2), q_scaled, local_kv)
    results["1. matmul A (q@local_kv)"] = t
    print(f"  1. matmul A (q_scaled @ local_kv.T):      {t:8.1f} µs")

    # 2. mask local_scores
    t = time_op(lambda s, m: _apply_score_mask(s, m), local_scores, local_mask)
    results["2. mask local"] = t
    print(f"  2. _apply_score_mask(local):              {t:8.1f} µs")

    # 3. logsumexp local
    t = time_op(lambda s: mx.logsumexp(s, -1, keepdims=True), local_scores)
    results["3. logsumexp local"] = t
    print(f"  3. logsumexp(local_scores):              {t:8.1f} µs")

    # 5. transpose q
    t = time_op(lambda q: q.transpose(0, 2, 1, 3), q_scaled)
    results["5. transpose q"] = t
    print(f"  5. q.transpose(0,2,1,3):                 {t:8.1f} µs")

    # 6. Matmul B: pooled_scores = q_bl @ pooled_sq.swapaxes
    t = time_op(lambda q, p: q @ p.swapaxes(-1, -2), q_bl, pooled_sq)
    results["6. matmul B (q_bl@pooled_sq)"] = t
    print(f"  6. matmul B (q_bl @ pooled_sq.T):         {t:8.1f} µs  <-- the big one")

    # 7. transpose pooled_scores back
    t = time_op(lambda s: s.transpose(0, 2, 1, 3), pooled_scores)
    results["7. transpose ps back"] = t
    print(f"  7. pooled_scores.transpose(0,2,1,3):     {t:8.1f} µs")

    # 8. mask pooled_scores
    t = time_op(lambda s, m: _apply_score_mask(s, m), pooled_scores, pooled_mask)
    results["8. mask pooled"] = t
    print(f"  8. _apply_score_mask(pooled):             {t:8.1f} µs")

    # 9. logsumexp pooled + logaddexp
    t = time_op(
        lambda n, ps: mx.logaddexp(n, mx.logsumexp(ps, -1, keepdims=True)),
        normalizer, pooled_scores,
    )
    results["9. logsumexp+logaddexp"] = t
    print(f"  9. logsumexp(pooled)+logaddexp:           {t:8.1f} µs")

    # 10. split_softmax (exp × 3, with sinks)
    t = time_op(
        lambda n, la, lb, s: _split_softmax(n, la, lb, s),
        normalizer, local_scores, pooled_scores, sinks_expanded,
    )
    results["10. split_softmax (exp×3)"] = t
    print(f"  10. split_softmax (exp×3 + sinks):       {t:8.1f} µs")

    # 11. Matmul C: out = local_weights @ local_kv
    local_weights, pooled_weights = _split_softmax(
        normalizer, local_scores, pooled_scores, sinks_expanded
    )
    mx.eval(local_weights, pooled_weights)
    mx.synchronize()
    t = time_op(lambda w, kv: w @ kv, local_weights, local_kv)
    results["11. matmul C (lw@local_kv)"] = t
    print(f"  11. matmul C (local_weights @ local_kv): {t:8.1f} µs")

    # 12. transpose pooled_weights
    t = time_op(lambda w: w.transpose(0, 2, 1, 3), pooled_weights)
    results["12. transpose pw"] = t
    print(f"  12. pooled_weights.transpose(0,2,1,3):   {t:8.1f} µs")

    # 13. Matmul D + add: out + (pw_bl @ pooled_sq).transpose
    pw_bl = pooled_weights.transpose(0, 2, 1, 3)
    mx.eval(pw_bl)
    mx.synchronize()
    out_partial = local_weights @ local_kv
    mx.eval(out_partial)
    mx.synchronize()
    t = time_op(
        lambda op, pw, ps: op + (pw @ ps).transpose(0, 2, 1, 3),
        out_partial, pw_bl, pooled_sq,
    )
    results["13. matmul D + trans + add"] = t
    print(f"  13. matmul D (pw@pooled_sq)+trans+add:   {t:8.1f} µs  <-- 2nd big")

    print()

    # ─────────────── Full inner kernel ───────────────
    t_full = time_op(inner_full, *args)
    results["FULL inner"] = t_full
    print(f"  FULL _sparse_pooled_attention_inner:      {t_full:8.1f} µs")
    print()

    # ─────────────── Aggregation ───────────────
    matmul_keys = [k for k in results if k.startswith(("1.", "6.", "11.", "13."))]
    overhead_keys = [k for k in results if k.startswith((
        "2.", "3.", "5.", "7.", "8.", "9.", "10.", "12."
    ))]
    matmul_total = sum(results[k] for k in matmul_keys)
    overhead_total = sum(results[k] for k in overhead_keys)
    sum_all = matmul_total + overhead_total

    print("=" * 78)
    print("AGGREGATION (op-isolation, unweighted sum):")
    print(f"  4 matmuls total:     {matmul_total:8.1f} µs  ({matmul_total/sum_all*100:.1f}%)")
    print(f"  overhead total:      {overhead_total:8.1f} µs  ({overhead_total/sum_all*100:.1f}%)")
    print(f"  sum(isolated):       {sum_all:8.1f} µs")
    print(f"  FULL inner (measured): {t_full:8.1f} µs")
    print(f"  overhead/overhead+matmul = {overhead_total/(overhead_total+matmul_total)*100:.1f}%")
    print()
    print("FUSION-CEILING ANALYSIS:")
    print("  A fused kernel can fold the 4 matmuls into online-softmax tiling,")
    print("  eliminating the intermediate score/weight round-trips.")
    print("  The overhead (logsumexp/logaddexp/split_softmax/transposes) is")
    print("  eliminated by construction in a fused kernel.")
    print()
    overhead_share = overhead_total / t_full * 100
    matmul_share = matmul_total / t_full * 100
    print(f"  overhead share of FULL: {overhead_share:.1f}%  <-- epilogue-fusion ceiling")
    print(f"  matmul share of FULL:    {matmul_share:.1f}%  <-- must beat steel GEMM")
    print()
    print("DECISION:")
    if overhead_share >= 40:
        print(f"  overhead = {overhead_share:.1f}% >= 40% → EPILOGUE FUSION is viable")
        print("  Fuse only logsumexp+logaddexp+split_softmax+exp, keep matmuls on steel GEMMs.")
        print("  Days not weeks. Near-zero slowdown risk.")
    else:
        print(f"  overhead = {overhead_share:.1f}% < 40% → epilogue fusion ceiling is low")
        print("  Full fusion needed, but per-row top-k makes FA tiling infeasible for pooled portion.")
    print()

    # ─────────────── Pipelined chain (21 layers) ───────────────
    print("=" * 78)
    print(f"Pipelined {N_LAYERS}-layer chain (captures MLX chain-level pipelining):")
    print("=" * 78)
    args_list = []
    for layer in range(N_LAYERS):
        a = make_inputs(42 + layer)
        mx.eval(*a)
        args_list.append(a)
    p_full = time_chain(inner_full, args_list)
    print(f"  FULL inner pipelined: {p_full:8.1f} µs total = {p_full/N_LAYERS:.1f} µs/layer")
    print()

    # ─────────────── Memory traffic estimate ───────────────
    print("=" * 78)
    print("MEMORY TRAFFIC (intermediate round-trips a fused kernel eliminates):")
    print("=" * 78)
    # Each intermediate is written once (producer) + read once (consumer) = 2× bytes
    intermeds = {
        "local_scores (B,H,L,sw)": (B, H, L_Q, SW),
        "pooled_scores (B,H,L,k)": (B, H, L_Q, K_SEL),
        "normalizer (B,H,L,1)": (B, H, L_Q, 1),
        "local_weights (B,H,L,sw)": (B, H, L_Q, SW),
        "pooled_weights (B,H,L,k)": (B, H, L_Q, K_SEL),
        "out (B,H,L,D)": (B, H, L_Q, D),
    }
    total_bytes = 0
    for name, shape in intermeds.items():
        sz = 1
        for d in shape:
            sz *= d
        b = sz * 2  # bf16
        total_bytes += 2 * b  # write + read
        print(f"  {name}: {fmt_bytes(b)} → {fmt_bytes(2*b)} (write+read)")
    print(f"  TOTAL intermediate traffic per layer: {fmt_bytes(total_bytes)}")
    print(f"  Per token (L_q={L_Q}): {fmt_bytes(total_bytes/L_Q)}")
    print(f"  At 546 GB/s M4 Max: {total_bytes/546e9*1e6:.2f} µs/layer floor (traffic alone)")
    print(f"  Measured pipelined: {p_full/N_LAYERS:.1f} µs/layer")
    print(f"  Traffic fraction: {total_bytes/546e9*1e6 / (p_full/N_LAYERS) * 100:.1f}%")
    print("=" * 78)


if __name__ == "__main__":
    main()