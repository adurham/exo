"""P4 (2026-08-23): does TP=2 width-sharding create SKINNY K/N GEMM shapes that
hurt tile / simdgroup efficiency for DSv4-Flash's prefill-dominant GEMMs?

QUESTION
--------
Under exo's `DeepseekV4ShardingStrategy` (MoE-only TP), the routed-expert FFN is
WIDTH-sharded:

    gate_proj / up_proj : all_to_sharded -> N halved   (2048 -> 1024 per rank)
    down_proj           : sharded_to_all -> K halved   (2048 -> 1024 per rank)

Attention is REPLICATED (no head-sharding), so it has no per-rank skinny dim at
all. The open question is whether the HALVED N (gate/up) and HALVED K (down)
measurably degrade the efficiency of the actually-dispatched Metal kernel
relative to the hypothetical unsharded width.

This is an EFFICIENCY question, not a wall-time question: the per-rank GEMM
legitimately does half the work. So every arm is scored on achieved GB/s and
achieved TFLOPS (work / time), never on raw ms.

WHAT IS ACTUALLY DISPATCHED (traced in mlx/backend/metal/quantized.cpp)
----------------------------------------------------------------------
`SwitchGLU.__call__` does `mx.expand_dims(x, (-2,-3))`, so `GatherQMM::eval_gpu`
always sees outer M == 1. With a 2048-token TP prefill chunk:

    B = 2048 tokens * top_k 6 = 12288 rows,  E = 256 experts,  B/E = 48

    tier 1  gather_qmv_rhs   needs B/E <= gather_qmv_rhs_max_be() == 6   -> NO
    tier 2  gather_qmm_rhs   needs M==1 && B>=16 && sorted && B/E>=4     -> YES

So prefill runs `gather_qmm_rhs`, tile geometry bm=16 bn=32 bk=32 wm=1 wn=2
(quantized.cpp; the bm=64 `_nax` variant is gated on `is_nax_available()`, which
is false on M4 Max -- architecture `applegpu_g16s`, gen 16 < the required 17).

Grid is `((N + bn - 1)/bn, (M + bm - 1)/bm, 1)`, so partial-tile waste in the
N direction is `ceil(N/32)*32 / N` and in the K direction the loop count is
`K/bk`. Both per-rank widths (1024) and unsharded widths (2048) are exact
multiples of 32 -> ZERO partial-tile waste either way, analytically. This bench
tests the second-order concern the tile arithmetic cannot answer: whether the
shorter K reduction (down_proj) or the narrower N (gate/up) costs achieved
bandwidth/FLOPs through reduced arithmetic intensity or weaker latency hiding.

METHOD (per exo-perf-tuning "Microbench Accuracy" + the 2026-08-22 P0 retraction)
--------------------------------------------------------------------------------
* NEVER `mx.eval()` inside the timed loop. The P0 retraction
  (`docs/switch-mlp-bandwidth-artifact-retraction-2026-08-22.md`) showed a
  per-iteration eval charges ~172us/call of host/dispatch overhead to the
  kernel and manufactured a fake 3.6x "headroom". We time a dependency-CHAINED
  graph with ONE eval per REPS calls, which is what production's per-layer
  chain actually looks like.
* >=3 warm-up iterations, `mx.eval` + `mx.synchronize` around the timed region.
* Identical routing indices are reused across every width arm, so per-expert
  raggedness is held EXACTLY constant and width is the only variable.
* Roofline framing is `max(compute_time, memory_time)`, not `compute + memory`.

Run:  .venv/bin/python bench/p4_tp_width_shard_gemm_efficiency.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mlx-lm"))
from mlx_lm.models.switch_layers import SwitchGLU  # noqa: E402

# --- DeepSeek-V4-Flash-0731 config (config.json, verified not assumed) ---
HIDDEN = 4096
INTER_FULL = 2048  # moe_intermediate_size
N_EXPERTS = 256  # n_routed_experts
TOP_K = 6  # num_experts_per_tok
GROUP_SIZE = 32  # mxfp4
BITS = 4
QMODE = "mxfp4"
DTYPE = mx.bfloat16

# attention is mxfp8 g=32 b=8 (make_quantization_config), and REPLICATED under
# exo's DeepseekV4ShardingStrategy -- no per-rank narrowing at all.
ATTN_BITS = 8
ATTN_MODE = "mxfp8"
Q_LORA_RANK = 1024
HEAD_DIM = 512
N_HEADS_FULL = 64

# TP prefill chunk: prefill_batched() uses the FULL EXO_PREFILL_STEP_SIZE.
# (The `// min(4, group.size())` halving at generate.py:497 is on the PIPELINE
# driver, not TP's prefill_batched loop.) SEQ_SPLIT halves only the attention
# QUERY band; MoE sees all 2048 rows on every rank.
L_CHUNK = 2048

WARMUP = 4
REPS = 12  # calls chained per timed graph
TRIALS = 5

# mxfp4: 4 bits/param + one fp8 (1 byte) scale per 32 params
BYTES_PER_PARAM_MXFP4 = 0.5 + 1.0 / 32.0
BYTES_PER_PARAM_MXFP8 = 1.0 + 1.0 / 32.0

# Session-measured M4 Max hardware truths (exo-perf-tuning "Hardware Truths").
PEAK_BW_SPEC = 546.0  # GB/s marketing spec
PEAK_BW_REAL = 424.0  # measured streaming bf16 read+write
# dense fp16 square-GEMM peak measured on this same laptop M4 Max in the
# 2026-08-18 attention session (docs/dsv4-attention-kernel-efficiency-2026-08-18.md)
PEAK_TFLOPS = 11.66


def _make_routing(n_rows: int, skew: str, seed: int) -> mx.array:
    """Top-k-of-256 routing indices with production-like raggedness.

    `skew="uniform"` gives a binomial spread (mean 48, sd ~7 at L=2048) --
    far TIGHTER than production. `skew="power"` reproduces the real measured
    shape (median ~14, mean ~48, long tail to >1000) by drawing expert
    popularity from a power law. Both are held IDENTICAL across width arms.
    """
    rng = mx.random.key(seed)
    if skew == "uniform":
        idx = mx.random.randint(0, N_EXPERTS, (n_rows, TOP_K), key=rng)
    else:
        # power-law expert popularity -> heavy raggedness
        u = mx.random.uniform(shape=(n_rows, TOP_K), key=rng)
        # alpha tuned so median rows/expert lands near the measured ~14
        idx = mx.floor(mx.power(u, 3.0) * N_EXPERTS).astype(mx.uint32)
        idx = mx.clip(idx, 0, N_EXPERTS - 1)
    mx.eval(idx)
    return idx.astype(mx.uint32)


def _routing_stats(idx: mx.array) -> dict:
    import numpy as np

    flat = np.array(idx).reshape(-1)
    counts = np.bincount(flat, minlength=N_EXPERTS)
    hit = counts[counts > 0]
    return {
        "rows_total": int(flat.size),
        "experts_hit": int(hit.size),
        "median_rows_per_hit_expert": float(np.median(hit)) if hit.size else 0.0,
        "mean_rows_per_hit_expert": float(hit.mean()) if hit.size else 0.0,
        "max_rows_per_expert": int(counts.max()),
        "min_rows_per_hit_expert": int(hit.min()) if hit.size else 0,
    }


def _build_moe(inter: int) -> SwitchGLU:
    moe = SwitchGLU(HIDDEN, inter, N_EXPERTS)
    nn.quantize(moe, group_size=GROUP_SIZE, bits=BITS, mode=QMODE)
    mx.eval(moe.parameters())
    return moe


def _time_chained(fn, x0: mx.array, reps: int, trials: int) -> list[float]:
    """Time `reps` dependency-CHAINED calls with ONE eval per trial.

    Mirrors production's per-layer data dependency. Crucially there is no
    `mx.eval` inside the loop -- that was the 2026-08-22 P0 artifact.
    """
    for _ in range(WARMUP):
        y = fn(x0)
        mx.eval(y)
    mx.synchronize()

    out = []
    for _ in range(trials):
        t0 = time.perf_counter()
        y = x0
        for _ in range(reps):
            y = fn(y)
        mx.eval(y)
        mx.synchronize()
        out.append((time.perf_counter() - t0) / reps)
    return out


def bench_moe_width(inter: int, idx: mx.array, l_chunk: int, skew: str) -> dict:
    """Real SwitchGLU (the production class) at one intermediate width."""
    moe = _build_moe(inter)
    x = mx.random.normal((1, l_chunk, HIDDEN)).astype(DTYPE)
    mx.eval(x)

    def call(v: mx.array) -> mx.array:
        # keep the chain shape-stable: MoE maps hidden->hidden
        return moe(v, idx).astype(DTYPE)

    times = _time_chained(call, x, REPS, TRIALS)
    t = statistics.median(times)

    rows = l_chunk * TOP_K
    # every expert is touched at these row counts, so the whole weight set moves
    params_per_expert = 3 * inter * HIDDEN  # gate + up + down
    weight_bytes = N_EXPERTS * params_per_expert * BYTES_PER_PARAM_MXFP4
    # activations: x in, gate/up out, down out
    act_bytes = (rows * HIDDEN + 2 * rows * inter + rows * HIDDEN) * 2
    total_bytes = weight_bytes + act_bytes
    flops = 2 * rows * (2 * HIDDEN * inter + inter * HIDDEN)  # gate+up+down

    gbs = total_bytes / t / 1e9
    tflops = flops / t / 1e12
    # roofline: this shape is weight-bandwidth-bound (already established
    # 2026-08-19), so report both arms of max(compute, memory)
    t_mem_real = total_bytes / (PEAK_BW_REAL * 1e9)
    return {
        "arm": "switchglu",
        "inter": inter,
        "l_chunk": l_chunk,
        "skew": skew,
        "ms_per_call": t * 1e3,
        "ms_stdev": statistics.stdev(times) * 1e3 if len(times) > 1 else 0.0,
        "achieved_gbs": gbs,
        "achieved_tflops": tflops,
        "pct_of_real_bw": 100.0 * gbs / PEAK_BW_REAL,
        "pct_of_spec_bw": 100.0 * gbs / PEAK_BW_SPEC,
        "bytes_per_call_gb": total_bytes / 1e9,
        "roofline_mem_ms": t_mem_real * 1e3,
        "roofline_efficiency_pct": 100.0 * t_mem_real / t,
    }


def bench_dense_qmm(m: int, k: int, n: int, bits: int, mode: str, label: str) -> dict:
    """Isolated dense quantized matmul -- no routing, no gather.

    Isolates the pure SHAPE effect of halving K or N from every MoE-specific
    (raggedness / gather / scatter) confound.
    """
    gs = GROUP_SIZE
    bpp = BYTES_PER_PARAM_MXFP4 if bits == 4 else BYTES_PER_PARAM_MXFP8
    # mxfp4/mxfp8 are SCALE-ONLY formats: mx.quantize returns (w_q, scales),
    # no biases (affine returns 3). Unpack defensively so this works for both.
    w = mx.random.normal((n, k)).astype(DTYPE)
    _qw = mx.quantize(w, group_size=gs, bits=bits, mode=mode)
    wq, ws = _qw[0], _qw[1]
    wb = _qw[2] if len(_qw) > 2 else None
    x = mx.random.normal((m, k)).astype(DTYPE)
    # projection back to k so the chain is shape-stable
    back = mx.random.normal((k, n)).astype(DTYPE)
    _qb = mx.quantize(back, group_size=gs, bits=bits, mode=mode)
    bq, bs = _qb[0], _qb[1]
    bb = _qb[2] if len(_qb) > 2 else None
    mx.eval(wq, ws, x, bq, bs)

    def call(v: mx.array) -> mx.array:
        y = mx.quantized_matmul(
            v, wq, scales=ws, biases=wb, transpose=True, group_size=gs, bits=bits, mode=mode
        )
        return mx.quantized_matmul(
            y, bq, scales=bs, biases=bb, transpose=True, group_size=gs, bits=bits, mode=mode
        )

    times = _time_chained(call, x, REPS, TRIALS)
    t = statistics.median(times) / 2.0  # two matmuls per chained step

    weight_bytes = n * k * bpp
    act_bytes = (m * k + m * n) * 2
    total_bytes = weight_bytes + act_bytes
    flops = 2 * m * n * k
    gbs = total_bytes / t / 1e9
    tflops = flops / t / 1e12
    return {
        "arm": "dense_qmm",
        "label": label,
        "M": m,
        "K": k,
        "N": n,
        "bits": bits,
        "mode": mode,
        "ms_per_call": t * 1e3,
        "achieved_gbs": gbs,
        "achieved_tflops": tflops,
        "pct_of_real_bw": 100.0 * gbs / PEAK_BW_REAL,
        "pct_of_compute_peak": 100.0 * tflops / PEAK_TFLOPS,
        "n_tiles": (n + 31) // 32,
        "n_tile_waste_pct": 100.0 * (((n + 31) // 32) * 32 - n) / n,
        "k_iters": k / 32.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skew", default="power", choices=["power", "uniform", "both"])
    ap.add_argument("--out", default="bench/results/p4_tp_width_shard.json")
    args = ap.parse_args()

    results: dict = {
        "meta": {
            "device": str(mx.device_info().get("device_name")),
            "architecture": str(mx.device_info().get("architecture")),
            "mlx": mx.__version__,
            "note": "gather_qmm_rhs bm=16 bn=32 bk=32 (nax variant gated off: gen 16 < 17)",
            "l_chunk": L_CHUNK,
            "reps_chained": REPS,
            "trials": TRIALS,
        },
        "moe": [],
        "dense": [],
    }

    skews = ["power", "uniform"] if args.skew == "both" else [args.skew]

    # ---- Arm 1: the real production class at several intermediate widths ----
    for skew in skews:
        idx = _make_routing(L_CHUNK * 1, skew, seed=1234)
        # SwitchGLU wants (B, L, top_k) index shape
        idx3 = idx.reshape(1, L_CHUNK, TOP_K)
        stats = _routing_stats(idx3)
        results.setdefault("routing", {})[skew] = stats
        print(f"\n=== routing skew={skew}: {stats}")
        for inter in (512, 1024, 2048):
            r = bench_moe_width(inter, idx3, L_CHUNK, skew)
            tp = {512: 4, 1024: 2, 2048: 1}[inter]
            r["implied_tp"] = tp
            results["moe"].append(r)
            print(
                f"  inter={inter:5d} (TP={tp}) {r['ms_per_call']:8.3f} ms  "
                f"{r['achieved_gbs']:7.1f} GB/s  {r['achieved_tflops']:6.2f} TF  "
                f"{r['pct_of_real_bw']:5.1f}% of real BW"
            )
            mx.clear_cache()

    # ---- Arm 2: isolated dense mxfp4 qmm, per-rank vs unsharded widths ----
    rows = L_CHUNK * TOP_K
    print("\n=== dense mxfp4 qmm: pure shape effect (no routing) ===")
    for label, m, k, n in (
        ("gate/up  per-rank  N=1024", rows, HIDDEN, 1024),
        ("gate/up  unsharded N=2048", rows, HIDDEN, 2048),
        ("gate/up  TP=4      N=512", rows, HIDDEN, 512),
        ("down     per-rank  K=1024", rows, 1024, HIDDEN),
        ("down     unsharded K=2048", rows, 2048, HIDDEN),
        ("down     TP=4      K=512", rows, 512, HIDDEN),
    ):
        r = bench_dense_qmm(m, k, n, BITS, QMODE, label)
        results["dense"].append(r)
        print(
            f"  {label:28s} {r['ms_per_call']:8.3f} ms  {r['achieved_gbs']:7.1f} GB/s  "
            f"{r['achieved_tflops']:6.2f} TF  {r['pct_of_real_bw']:5.1f}% real BW"
        )
        mx.clear_cache()

    # ---- Arm 3: attention wq_b -- REPLICATED under exo, so full n_heads ----
    print("\n=== attention wq_b (mxfp8): exo replicates, so N is FULL width ===")
    for label, n_heads in (("replicated (exo real, 64 heads)", 64), ("head-sharded hypo (32)", 32)):
        r = bench_dense_qmm(
            L_CHUNK, Q_LORA_RANK, n_heads * HEAD_DIM, ATTN_BITS, ATTN_MODE, label
        )
        results["dense"].append(r)
        print(
            f"  {label:34s} N={n_heads * HEAD_DIM:6d} {r['ms_per_call']:8.3f} ms  "
            f"{r['achieved_gbs']:7.1f} GB/s  {r['achieved_tflops']:6.2f} TF"
        )
        mx.clear_cache()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
