# P08 Item 1 (b): REACHABILITY of the ~2.1x arithmetically-unnecessary work in
# attn.sdpa.compressed using ONLY existing MLX primitives (call restructuring,
# NOT hand-written Metal). Standalone on macstudio-m4-1 beside live runner 59909.
#
# Sections:
#   A. mask structure: visible-keys/row local ring [0:2175] vs pooled [2175:3894]
#   B. SDPA string-mask/windowed support introspection + GPU-timed variants
#   C. decomposition: pooled + local sub-calls merged via log-sum-exp
#   D. windowed-local query-block tiling (B=128/256/512) x row assignments
#   E. best-of verdict assembly
#
# Timing discipline = p08_item1_capture.py recipe (wrapper sets MLX_GPU_TIME=1
# + MLX_DISPATCH_COUNT=1 BEFORE import; fresh lazy graph per call;
# mx.eval+mx.synchronize; median of >=5 timed after >=3 warmup; bank rotation;
# no *0 folding; physics sanity vs 15.21 TF / 488 GB/s).

import json
import math
import socket
import statistics
import time
from pathlib import Path

import mlx.core as mx

OUT = Path("/Users/adam.durham/repos/exo/tmp/p08-20260830")
OUT.mkdir(parents=True, exist_ok=True)
R = {
    "meta": {}, "mask_structure": {}, "sdpa_support": {}, "baseline": {},
    "sink_check": {}, "decomposition": {}, "windowed_local": {},
    "row_assignment_locality": {}, "verdict_input": {}, "errors": [],
}


def save():
    (OUT / "item1b_reachability_results.json").write_text(
        json.dumps(R, indent=1, default=str))


def log(*a):
    print(*a, flush=True)


N_HEADS, HEAD_DIM, L_BAND = 32, 512, 1024
LOCAL_KV, POOL_128, CATTN_KV = 2175, 1719, 3894
SLIDING, N_CHUNK_L, OFF_LAST, RATIO_C = 128, 2048, 218880, 128
SCALE = HEAD_DIM ** -0.5
BASELINE_CAUSAL_US = 21423.38   # item1_results.json causal median
NUMERICS_GATE = 0.002

R["meta"] = {
    "host": socket.gethostname(), "mlx": mx.__version__,
    "arch": mx.device_info(), "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    "shape": {"q": [1, N_HEADS, L_BAND, HEAD_DIM],
              "kv": [1, 1, CATTN_KV, HEAD_DIM],
              "mask": [1, 1, L_BAND, CATTN_KV], "dtype": "bfloat16"},
    "baseline_anchor_us": BASELINE_CAUSAL_US,
}
save()

# ===================== (A) mask structure ====================================
# Production mask, reconstructed exactly (cache.py:633 make_mask windowed
# causal; PoolingCache.make_mask cache.py:1605-1627; concat dv4:4359; band
# slice dv4:4381-4383).
q_pos = mx.arange(N_CHUNK_L) + 127                 # 127..2174 (create_causal_mask exact)
k_idx = mx.arange(LOCAL_KV)                        # 0..2174
# create_causal_mask(base.py:31-37): mask = linds >= rinds & linds < rinds + W
#   linds = query positions 127..2174, rinds = key index 0..2174.
# EMPIRICAL (diag_mask.py on-node): query row r sees keys [r .. r+127],
# i.e. the window sits AHEAD of the row (rinds <= linds < rinds+W).
m_local = (q_pos[:, None] >= k_idx[None, :]) & (q_pos[:, None] < k_idx[None, :] + SLIDING)
pool_idx = mx.arange(POOL_128)[None, :]
query_pos = OFF_LAST + mx.arange(1, N_CHUNK_L + 1)[:, None]
m_pool = pool_idx < (query_pos // RATIO_C)          # (2048, 1719)
mask_full = mx.concatenate([m_local[None, None], m_pool[None, None]], axis=-1)
mask_rank = mask_full[..., 0:L_BAND, :]             # rank-0 band (dv4:4381)
ml, mp = m_local[0:L_BAND], m_pool[0:L_BAND]
mx.eval(mask_rank, ml, mp)


def vis_stats(mm):
    cnt = mm.sum(axis=1).astype(mx.float32)
    return {"min": int(cnt.min().item()), "max": int(cnt.max().item()),
            "mean": round(float(cnt.mean().item()), 2),
            "region_cols": int(mm.shape[1]),
            "density": round(float(mm.astype(mx.float32).mean().item()), 6)}


R["mask_structure"] = {
    "split_point": {"local_cols": LOCAL_KV, "pooled_cols": POOL_128,
                    "provenance": "concat at deepseek_v4.py:4359 (mlx-lm live "
                    "checkout a6eb893); local = RotatingKVCache post-update len "
                    "= max_size(128)+N(2048)-1 = 2175 (cache.py:633); pooled = "
                    "ceil(220000/128) = 1719. Split is EXACTLY 2175/1719."},
    "local_ring_first2175_vis_per_row": vis_stats(ml),
    "pooled_last1719_vis_per_row": vis_stats(mp),
    "window_direction_note": "row i sees local keys [i, i+127] (window AHEAD of "
    "the row: q_pos >= k_idx & q_pos < k_idx + 128), NOT [i-127, i]",
}
save()
log("=== (A) mask structure ===")
log(json.dumps(R["mask_structure"], indent=1))

# =========================== tensors / banks =================================
N_KV_BANKS = 16
q = mx.random.normal((1, N_HEADS, L_BAND, HEAD_DIM)).astype(mx.bfloat16)
sinks = mx.zeros((N_HEADS,), dtype=mx.bfloat16)
kv_banks = [mx.random.normal((1, 1, CATTN_KV, HEAD_DIM)).astype(mx.bfloat16)
            for _ in range(N_KV_BANKS)]
kv_local = kv_banks[0][..., 0:LOCAL_KV, :]
kv_pool = kv_banks[0][..., LOCAL_KV:, :]
mask_bool = mask_rank
mask_local = mask_bool[..., 0:LOCAL_KV]
mask_pool = mask_bool[..., LOCAL_KV:]
mx.eval(q, sinks, *kv_banks, kv_local, kv_pool, mask_bool, mask_local, mask_pool)
_ctr = {"i": 0}


def bank():
    _ctr["i"] += 1
    return kv_banks[_ctr["i"] % N_KV_BANKS]


def timed(mk, warmup=6, iters=11):
    for _ in range(warmup):
        mx.eval(mk())
    mx.synchronize()
    recs = []
    for _ in range(iters):
        t0 = time.perf_counter()
        mx.metal.reset_gpu_time()
        d0 = mx.metal.dispatch_count()
        out = mk()
        mx.eval(out)
        mx.synchronize()
        wall_us = (time.perf_counter() - t0) * 1e6
        gpu_us = mx.metal.gpu_time_ns() / 1e3
        disp = mx.metal.dispatch_count() - d0
        recs.append((wall_us, gpu_us, disp))
    return {"median_wall_us": round(statistics.median([r[0] for r in recs]), 2),
            "median_gpu_us": round(statistics.median([r[1] for r in recs]), 2),
            "median_dispatches": int(statistics.median([r[2] for r in recs])),
            "all_gpu_us": [round(r[1], 1) for r in recs]}


def rel_err_mean(a, b):
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    return float((mx.abs(a - b) / (mx.abs(b) + 1e-6)).mean().item())


def rel_err_max(a, b):
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    return float((mx.abs(a - b) / (mx.abs(b) + 1e-6)).max().item())


# ============================ baseline =======================================
log("=== baseline re-run: single SDPA over 3894 keys, production mask ===")
R["baseline"] = timed(lambda: mx.fast.scaled_dot_product_attention(
    q, bank(), bank(), scale=SCALE, mask=mask_bool, sinks=sinks))
save()
log(json.dumps(R["baseline"]))

# ================ (B) SDPA support introspection + variants ==================
log("=== (B) SDPA variant support probe ===")
variants = {}
variants["array_bool_mask_production"] = timed(
    lambda: mx.fast.scaled_dot_product_attention(
        q, bank(), bank(), scale=SCALE, mask=mask_bool, sinks=sinks))
variants["mask_none"] = timed(
    lambda: mx.fast.scaled_dot_product_attention(
        q, bank(), bank(), scale=SCALE, mask=None, sinks=sinks))


def mk_causal_str():
    return mx.fast.scaled_dot_product_attention(
        q, bank(), bank(), scale=SCALE, mask="causal", sinks=sinks)


try:
    mx.eval(mk_causal_str())
    variants["mask_string_causal"] = dict(timed(mk_causal_str), accepted=True)
except Exception as e:
    variants["mask_string_causal"] = {"accepted": False, "error": repr(e)[:300]}
save()
for kw_name, kwargs in (("kwarg_window_size", {"window_size": SLIDING}),
                        ("kwarg_sliding_window", {"sliding_window": SLIDING}),
                        ("kwarg_block_sparse", {"block_sparse": True})):
    def mk(kwargs=kwargs):
        return mx.fast.scaled_dot_product_attention(
            q, bank(), bank(), scale=SCALE, sinks=sinks, **kwargs)
    try:
        mx.eval(mk())
        variants[kw_name] = dict(timed(mk), accepted=True)
    except Exception as e:
        variants[kw_name] = {"accepted": False, "error": repr(e)[:300]}
    save()

# 'causal' string vs materialized lower-right-causal bool mask at SAME geometry
geom_q = mx.arange(L_BAND)[:, None] + (CATTN_KV - L_BAND)
geom_k = mx.arange(CATTN_KV)[None, :]
mask_lr = geom_q >= geom_k
mx.eval(mask_lr)
out_str = mx.fast.scaled_dot_product_attention(
    q, kv_banks[0], kv_banks[0], scale=SCALE, mask="causal", sinks=sinks)
out_lr = mx.fast.scaled_dot_product_attention(
    q, kv_banks[0], kv_banks[0], scale=SCALE, mask=mask_lr, sinks=sinks)
mx.eval(out_str, out_lr)
R["sdpa_support"] = {
    "signature": "scaled_dot_product_attention(q, k, v, *, scale: float, "
    "mask: Union[None, str, array] = None, sinks: Optional[array] = None, "
    "stream=None) -> array",
    "accepted_mask_types": "array (bool or additive) OR the single string "
    "'causal' (lower-right aligned). NO window_size / sliding_window / "
    "block_sparse / segment kwarg exists in this build (probe results in "
    "variants).",
    "variants": variants,
    "string_causal_vs_lrcausal_bool_max_rel_diff": rel_err_max(out_str, out_lr),
    "lrcausal_bool_tensor_time": timed(
        lambda: mx.fast.scaled_dot_product_attention(
            q, bank(), bank(), scale=SCALE, mask=mask_lr, sinks=sinks)),
    "note": "string 'causal' cannot express the production mask (windowed-"
    "causal ring + pooled row-causal suffix): the pooled region is ~dense for "
    "ALL rows while string-causal would hide trailing pooled columns from "
    "early rows. So no faster path for the production op is reachable via "
    "string masks; timed only as kernel-path evidence.",
}
save()
log(json.dumps(R["sdpa_support"], indent=1, default=str))

# ======== sink-formula verification (needed for the exact LSE merge) =========
log("=== sink formula check: reproduces single-call SDPA with explicit scores? ===")
# mx.fast.sdpa with sinks: production base.py:193-201 passes sinks=zeros(32,).
# MLX semantics (mlx-lm sinks): logits get an extra sink column of value
# sinks[h]; with sinks=0 that contributes exp((s_max - s_max)) = 1 to Z.
# Verify: explicit softmax WITH a zeros-sink column == mx.fast.sdpa output.
NEG = -1e30  # finite very-negative instead of -inf: keeps -inf out of gather paths
s_full = (q @ kv_banks[0].transpose(0, 1, 3, 2)) * SCALE      # bf16 scores
s32 = s_full.astype(mx.float32)
NEG_MASK = mx.full((1, 1, 1, CATTN_KV), NEG, dtype=s32.dtype)  # broadcasts over heads
s32m = mx.where(mask_bool, s32, NEG_MASK)
smax = mx.max(s32m, axis=-1, keepdims=True)
p = mx.exp(s32m - smax)
Z = mx.sum(p, axis=-1, keepdims=True)              # sinks=0 adds exp(-m), NOT +1
# PV in fp32 (kernel accumulates fp32 internally; bf16 p was the 3.5% error)
o_explicit = (p @ kv_banks[0].astype(mx.float32)) / Z
o_explicit = o_explicit.astype(mx.bfloat16)
o_sdpa_ref = mx.fast.scaled_dot_product_attention(
    q, kv_banks[0], kv_banks[0], scale=SCALE, mask=mask_bool, sinks=sinks)
mx.eval(o_explicit, o_sdpa_ref)
R["sink_formula_check"] = {
    "formula": "Z = sum(exp(s - max over visible keys)); sinks=0 contributes "
    "exp(0 - smax_keys), which is inside the running max already since smax is "
    "computed over masked scores (NEG for invisible) -- verified by diag3.py: "
    "fast(sinks=0) == fast(None) within bf16 rounding, and fast(sinks=5) == "
    "Z + exp(5 - smax). A '+1.0' constant was WRONG (diag3 measured 0.026 mean "
    "rel diff with +1 vs 0.0013 with the correct formula); p@V in fp32.",
    "max_rel_diff_vs_fast_sdpa": rel_err_max(o_explicit, o_sdpa_ref),
    "mean_rel_diff_vs_fast_sdpa": rel_err_mean(o_explicit, o_sdpa_ref),
}
save()
log(json.dumps(R["sink_formula_check"], indent=1))

# =============== (C) decomposition: pooled + local, LSE merge ================
log("=== (C) two-call decomposition (pooled 1719 + local 2175) with LSE merge ===")
# mx.fast.sdpa does NOT return per-row logsumexp, so partial pieces that must
# be re-merged use explicit matmul+softmax+matmul; merge via running-max trick.
# Sink-correct: each partial's denominator gets +1.0 (zeros sink), and the
# merged Z gets +1.0 once (the single production sink).
KV0 = kv_banks[0]


def partial_explicit(qx, kx, vkx, m):
    """(out, per-row logsumexp) with NEG-filled masked scores, fp32 softmax.
    Returns UN-normalized p@V (p = exp(s - smax)) plus lse = log(sum p) + smax,
    so the true normalizer is exp(lse) and the merged denominator gets the
    zeros-sink as +1.0 (sinks=0 contributes exp(0)=1 to Z)."""
    s = ((qx @ kx.transpose(0, 1, 3, 2)) * SCALE).astype(mx.float32)
    sm = mx.where(m, s, mx.full(s.shape, NEG, dtype=s.dtype))
    smax = mx.max(sm, axis=-1, keepdims=True)
    p = mx.exp(sm - smax)
    lse = mx.log(mx.sum(p, axis=-1, keepdims=True)) + smax    # fp32 (…,Tq,1)
    o = p @ vkx.astype(mx.float32)                             # fp32 PV
    return o, lse


def merge2(o1, lse1, o2, lse2):
    """Exact two-way logsumexp merge of un-normalized partials.
    Each partial o_i = (sum_j exp(s_ij - lse_i) v_j) i.e. o_i = S_i/Z_i with
    Z_i = exp(lse_i) here normalized differently: partial_attn returns
    un-normalized p@V with p = exp(s - smax_i), lse_i = log(sum p) + smax_i,
    so TRUE normalizer Zi = exp(lse_i). Merge: o = (o1*w1 + o2*w2)/(w1+w2)."""
    lmax = mx.maximum(lse1, lse2)
    w1 = mx.exp(lse1 - lmax)
    w2 = mx.exp(lse2 - lmax)
    Z = w1 + w2
    return (o1.astype(mx.float32) * w1 + o2.astype(mx.float32) * w2) / Z


out_prod = mx.fast.scaled_dot_product_attention(
    q, kv_banks[0], kv_banks[0], scale=SCALE, mask=mask_bool, sinks=sinks)
mx.eval(out_prod)


def mk_decomposed():
    o_p, lse_p = partial_explicit(q, kv_pool, kv_pool, mask_pool)
    o_l, lse_l = partial_explicit(q, kv_local, kv_local, mask_local)
    # sinks=0 contributes exp(0 - s_max_keys) to the true Z (diag3.py). Each
    # partial's lse is relative to its OWN max; the shared sink is folded in
    # by adding exp(0 - lmax_merged) to the merged normalizer.
    lmax = mx.maximum(lse_p, lse_l)
    w_p = mx.exp(lse_p - lmax)
    w_l = mx.exp(lse_l - lmax)
    Zm = w_p + w_l + mx.exp(-lmax)   # <- the zeros-sink term, exp(0-lmax)
    o_merge = (o_p * w_p + o_l * w_l) / Zm
    return o_merge.astype(mx.bfloat16)


try:
    out_dec = mk_decomposed()
    mx.eval(out_dec, out_prod)
    dec = timed(mk_decomposed)
    R["decomposition"] = {
        "description": "pooled partial (1719) + local partial (2175), each "
        "explicit matmul+softmax+matmul, merged via log-sum-exp running-max "
        "with the shared zeros-sink folded into Z (+1.0). mx.fast.sdpa does "
        "not return per-row LSE, so an SDPA-based exact split is not "
        "expressible; explicit path measured.",
        "time": timed(mk_decomposed),
        "mean_rel_err_vs_single_call": rel_err_mean(out_dec, out_prod),
        "max_rel_err": rel_err_max(out_dec, out_prod),
    }
except Exception as e:
    R["errors"].append(f"decomposition: {e!r}")
    R["decomposition"] = {"error": repr(e)}
save()
log(json.dumps(R["decomposition"], indent=1, default=str))

# ============= (D) windowed-local tiling =====================================
log("=== (D) windowed-local tiling B=128/256/512 x row-assignment ===")
# Row assignments (chunk L=2048, N=2 ranks, band=1024):
#  contiguous (LIVE production, dv4:4375-4383): rank0 = rows [0:1024).
#  subchunk-balanced (d2b28e21, REVERTED): rank0 = sub-chunks 0,2 = chunk rows
#    [0:512) + [1024:1536), gathered via q[:,:,idx,:].
#  rowstride2 (pure-interleave variant): buffer row j = chunk row 2j (+1).
def band_rows_balanced_subchunks():
    sub = N_CHUNK_L // 4  # 512
    rows = []
    for lo, hi in ((0, sub), (2 * sub, 3 * sub)):
        rows.extend(range(lo, hi))
    return rows


def band_rows_rowstride2():
    # rank 0 gets even rows, rank 1 odd (the OTHER natural interleave)
    return list(range(0, N_CHUNK_L, 2))


R["row_assignment_locality"] = {
    "balanced_provenance": {
        "live_env_flag": "EXO_DSV4_SEQSPLIT_BALANCED=1 IS set in the live "
        "runner env (ps eww 59909, verified this session; start_cluster.sh:97)",
        "but_code_reverted": "the balanced implementation (mlx-lm commit "
        "d2b28e21, _seqsplit_band(): sub-chunks interleaved rank0=[0,2,4..]) "
        "was REVERTED at bf8cbad5 (2026-07-13, 'throughput-neutral, slight "
        "regression'; Fable-5-reviewed A/B: 309.8 vs 313.3 cumulative at 500K, "
        "within noise). The live mlx-lm checkout (a6eb893) contains NO "
        "balanced code: grep SEQSPLIT_BALANCED in mlx-lm/mlx_lm = 0 hits; "
        "deepseek_v4.py:4376-4383 assigns the CONTIGUOUS band "
        "_seq_lo = _sg.rank()*_band (rank0 rows [0:1024)).",
        "verdict": "the env var is INERT in the live build; this rank's 1024 "
        "query rows ARE contiguous in the original sequence (rows 0..1023 of "
        "the 2048 chunk on rank 0). Both assignment patterns measured anyway.",
    },
    "interleaved_note": "under a hypothetical balanced assignment (sub-chunks "
    "0,2 of 512) a B-row block in the band buffer spans chunk rows within one "
    "sub-chunk for B<=512 OR straddles the two sub-chunks for the block "
    "crossing the boundary — NOT the full ring. Under a worst-case pure "
    "row-stride-2 interleave, a B=256 block spans 512 chunk rows and its "
    "local window union covers min..max+127 = 512+127 = 639 distinct keys, "
    "not ~383. Both measured.",
}

# ---- row-assignment locality measurement (pure arithmetic, no GPU) ----------
def keyspans(rows):
    """rows: list mapping band-buffer row j -> chunk row position. For each B,
    per-block union of visible local keys = |[min-0 .. max+127] clipped|."""
    out = {}
    for B in (32, 128, 256, 512):
        n_blocks = math.ceil(len(rows) / B)
        widths = []
        for b in range(n_blocks):
            blk = rows[b * B:(b + 1) * B]
            lo_k = max(0, min(blk))                    # pos itself visible
            hi = min(LOCAL_KV, max(blk) + SLIDING)     # + window ahead
            widths.append(hi - lo_k)
        out[f"B{B}"] = {"n_blocks": n_blocks,
                        "distinct_local_keys_per_block": {
                            "min": min(widths), "max": max(widths)}}
    return out


rows_contig = list(range(L_BAND))
_sub = N_CHUNK_L // 4
rows_subchunk = list(range(0, _sub)) + list(range(2 * _sub, 3 * _sub))
rows_stride2 = list(range(0, N_CHUNK_L, 2))
R["row_assignment_locality"]["distinct_local_keys_per_block"] = {
    "contiguous_live_production": keyspans(rows_contig),
    "balanced_subchunks_reverted_commit": keyspans(rows_subchunk),
    "row_stride2_worst_interleave": keyspans(rows_stride2),
    "note": "window: band row i sees local keys [i, i+127]; a block of B "
    "contiguous chunk rows needs B+127 distinct keys (clipped at ring edges); "
    "an interleaved assignment needs the span between min and max row + 127.",
}
save()
log(json.dumps(R["row_assignment_locality"], indent=1))

# ---- untiled local-region SDPA reference (the D comparison anchor) ----------
t_untiled_local = timed(lambda: mx.fast.scaled_dot_product_attention(
    q, bank()[..., :LOCAL_KV, :], bank()[..., :LOCAL_KV, :],
    scale=SCALE, mask=mask_bool[..., :LOCAL_KV], sinks=sinks))
R["windowed_local"]["untiled_local_fullband"] = t_untiled_local
o_local_ref = mx.fast.scaled_dot_product_attention(
    q, kv_local, kv_local, scale=SCALE, mask=mask_local, sinks=sinks)
mx.eval(o_local_ref)
log("untiled local:", json.dumps(t_untiled_local))


def tiled_local_contiguous(B):
    """Query rows tiled into blocks of B; per block SDPA over exactly the
    needed local-key window; outputs concatenated. Contiguous rows."""
    outs = []
    for b0 in range(0, L_BAND, B):
        b1 = b0 + B
        k_lo = max(0, b0)                    # min chunk pos of block = b0
        k_hi = min(LOCAL_KV, b1 - 1 + SLIDING)
        n_k = k_hi - k_lo
        kb = kv_banks[0][..., k_lo:k_hi, :]
        pos_q = mx.arange(b0, b1, dtype=mx.int32)[:, None] - b0
        kp = mx.arange(n_k, dtype=mx.int32)[None, :]
        m = (pos_q + b0 >= kp + k_lo) & (pos_q + b0 < kp + k_lo + SLIDING)
        outs.append(mx.fast.scaled_dot_product_attention(
            q[:, :, b0:b1, :], kb, kb, scale=SCALE,
            mask=mx.broadcast_to(m[None, None], (1, 1, B, n_k)), sinks=sinks))
    return mx.concatenate(outs, axis=2)


def tiled_local_balanced(B):
    """Same but rows re-indexed to the balanced sub-chunk order (chunk rows
    [0:512) then [1024:1536)); per block, exactly the needed window."""
    outs = []
    for b0 in range(0, L_BAND, B):
        b1 = b0 + B
        blk = rows_subchunk[b0:b1]
        k_lo = max(0, min(blk))
        k_hi = min(LOCAL_KV, max(blk) + SLIDING)
        kb = kv_banks[0][..., k_lo:k_hi, :]
        qb = q[:, :, b0:b1, :]
        kp = mx.arange(k_lo, k_hi, dtype=mx.int32)[None, :]
        pos = mx.array(blk, dtype=mx.int32)[:, None]
        m = (pos >= kp) & (pos < kp + SLIDING)
        outs.append(mx.fast.scaled_dot_product_attention(
            qb, kb, kb, scale=SCALE, mask=m[None, None], sinks=sinks))
    return mx.concatenate(outs, axis=2)


def tiled_local_stride2(B):
    """Worst-case row-stride-2 interleave: buffer blocks span wide chunk rows."""
    outs = []
    for b0 in range(0, L_BAND, B):
        b1 = b0 + B
        blk = rows_stride2[b0:b1]
        lo_k = max(0, min(blk))
        hi_k = min(LOCAL_KV, max(blk) + SLIDING)
        kb = kv_banks[0][..., lo_k:hi_k, :]
        kp = mx.arange(lo_k, hi_k, dtype=mx.int32)[None, :]
        pos = mx.array(blk, dtype=mx.int32)[:, None]
        m = (pos >= kp) & (pos < kp + SLIDING)
        outs.append(mx.fast.scaled_dot_product_attention(
            q[:, :, b0:b1, :], kb, kb, scale=SCALE, mask=m[None, None],
            sinks=sinks))
    return mx.concatenate(outs, axis=2)


for tag, fn, ref_rows in (
        ("tiled_local_contiguous", tiled_local_contiguous, None),
        ("tiled_local_balanced", tiled_local_balanced, rows_subchunk),
        ("tiled_local_stride2", tiled_local_stride2, rows_stride2)):
    for B in (128, 256, 512):
        key = f"{tag}_B{B}"
        try:
            o_t = fn(B)
            mx.eval(o_t)
            t = timed(lambda f=fn, B=B: f(B))
            R["windowed_local"][key] = {"time": t}
        except Exception as e:
            R["windowed_local"][key] = {"error": repr(e)}
            R["errors"].append(f"{key}: {e!r}")
        save()
        log(key, json.dumps(R["windowed_local"].get(key, {}), default=str))

# numerics of tiled-contiguous vs the untiled local reference
idx_probe = {}
for B in (128, 256, 512):
    try:
        o_t = tiled_local_contiguous(B)
        mx.eval(o_t)
        R["windowed_local"][f"tiled_local_contiguous_B{B}"]["numerics_vs_untiled_local"] = {
            "mean": rel_err_mean(o_t, o_local_ref),
            "max": rel_err_max(o_t, o_local_ref),
        }
    except Exception as e:
        R["windowed_local"][f"tiled_local_contiguous_B{B}"]["numerics_error"] = repr(e)
        R["errors"].append(f"tile numerics B{B}: {e!r}")
save()

# ============== (D2) full composite per query block ==========================
log("=== (D2) full-op composite: pooled SDPA + tiled local + LSE merge ===")


def composite_full(B, interleaved):
    """Full op per query block: local SDPA-windowed partial (explicit, gives
    LSE) + pooled partial (explicit, needed for its LSE anyway), merged with
    the log-sum-exp running-max and the single shared zeros-sink in Z."""
    rows = rows_stride2 if interleaved else rows_contig
    kvs_pool = kv_pool
    outs = []
    for b0 in range(0, L_BAND, B):
        b1 = b0 + B
        blk = rows[b0:b1]
        lo = max(0, min(blk))
        hi = min(LOCAL_KV, max(blk) + SLIDING)
        kb = kv_banks[0][..., lo:hi, :]
        qb = q[:, :, b0:b1, :]
        kp = mx.arange(lo, hi, dtype=mx.int32)[None, :]
        pos = mx.array(blk, dtype=mx.int32)[:, None]
        m_loc = (pos >= kp) & (pos < kp + SLIDING)
        o_loc, lse_loc = partial_explicit(qb, kb, kb, m_loc[None, None])
        m_p = mask_pool[:, :, b0:b1, :]
        o_p, lse_p = partial_explicit(qb, kv_pool, kv_pool, m_p)
        lmax = mx.maximum(lse_loc, lse_p)
        w_loc = mx.exp(lse_loc - lmax)
        w_p = mx.exp(lse_p - lmax)
        Zm = w_loc + w_p + mx.exp(-lmax)   # zeros-sink = exp(0-lmax)
        outs.append(((o_loc * w_loc + o_p * w_p) / Zm).astype(mx.bfloat16))
    return mx.concatenate(outs, axis=2)


for tag, il in (("contiguous", False), ("interleaved_stride2", True)):
    for B in (128, 256, 512):
        key = f"composite_{tag}_B{B}"
        try:
            o_c = composite_full(B, il)
            mx.eval(o_c)
            err = rel_err_mean(o_c, out_prod)
            t = timed(lambda B=B, il2=il: composite_full(B, il2))
            R["windowed_local"][key] = {
                "time": t, "mean_rel_err_vs_single_call": err,
                "numerics_pass": err < NUMERICS_GATE}
        except Exception as e:
            R["windowed_local"][key] = {"error": repr(e)}
            R["errors"].append(f"{key}: {e!r}")
        save()
        log(key, json.dumps(R["windowed_local"].get(key, {}), default=str))

# ===================== (E) verdict assembly ==================================
passing = []
for name, rec in R["windowed_local"].items():
    t = rec.get("time", {}).get("median_gpu_us")
    e = rec.get("mean_rel_err_vs_single_call")
    if t is not None and (e is not None and e < NUMERICS_GATE):
        passing.append((name, t, e))
R["verdict_input"] = {
    "baseline_us": BASELINE_CAUSAL_US, "numerics_gate": NUMERICS_GATE,
    "variants_passing": [list(p) for p in passing]}
if passing:
    best_name, best_t, best_e = min(passing, key=lambda x: x[1])
    R["verdict_input"].update({
        "best_variant": best_name, "best_time_us": best_t,
        "best_rel_err": best_e,
        "speedup_vs_baseline": round(best_t / BASELINE_CAUSAL_US, 4),
        "implied_e2e_win_pct": round(11.8 * (1 - best_t / BASELINE_CAUSAL_US), 3),
    })
save()

log("\n=== SUMMARY ===")
log(f"baseline single-call causal: {R['baseline']['median_gpu_us']}us "
    f"(item1 anchor 21423.4us)")
for k, v in R["windowed_local"].items():
    log(f"  {k}: {json.dumps(v, default=str)}")
log("verdict:", json.dumps(R["verdict_input"], default=str))