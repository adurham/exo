#!/usr/bin/env python3
"""I11 STEP 1 -- MoE gather_qmm microbench at bits {6,5,4}, M=4.

METHOD OF RECORD: the R1 CHAINED-GRAPH construction
(tmp/perf-campaign-2/round1/i3_microbench_chained.py, adopted after the
2026-08-22 serial-sync artifact retraction). SERIAL-SYNC TIMING IS NOT
USED ANYWHERE IN THIS FILE:

  - Build a dependency-chained graph of CHAIN_LEN SwitchGLU-equivalent
    calls, where call i+1's input is a real data dependency of call i's
    output (carry = x + 1e-9 * mean(out, axis=-2)). MLX cannot elide any
    call. ONE mx.eval() at the very end of the chain.
  - ROTATED routing indices from a pool of N_POOL independently-drawn
    (M, top_k) index sets, so no repeat-hit expert weight set can sit in
    cache across the chain (the "fictitious cache" artifact class).
  - A measured EMPTY-GRAPH BASELINE (the identical carry chain with the
    three gather_qmm calls removed) is subtracted, so the reported
    us/call is the marginal cost of the MoE kernels, not the chain
    scaffolding.
  - A chain-length SCALING CHECK (elision detector): us/call must stay
    flat as CHAIN_LEN grows. If the chain were being optimised away,
    total wall would be ~constant and us/call would collapse.

ARMS ARE INTERLEAVED: each repetition runs all three bit widths
back-to-back, and the arm ORDER ALTERNATES between reps (6,5,4 /
4,5,6 / ...). Any monotonic drift from live-cluster GPU contention is
therefore spread across arms rather than loaded onto whichever arm ran
first. Ranges (min/median/max over reps) are reported, never bare means.

WHY AFFINE MODE: the deployed routed experts are mxfp4, and the mxfp
family only defines 4-bit (mxfp4) and 8-bit (mxfp8) -- there is no
mxfp5 or mxfp6. A 6- vs 5- vs 4-bit comparison is therefore necessarily
an AFFINE-mode comparison, which is the mode mlx-community 5-/6-bit
conversions actually use. Stated in the writeup, not silently assumed.

MEMORY DISCIPLINE (the cluster is LIVE): the full-precision expert
tensor at the deployed shape is ~4.3 GB per projection, which would not
fit in the headroom the live runner leaves. Weights are therefore
quantized in EXPERT CHUNKS (real mx.quantize on real random weights --
NOT synthetic bit patterns) and only one bit-width arm is resident at a
time.
"""

import argparse
import gc
import json
import statistics
import sys
import time

import mlx.core as mx
import numpy as np

# ---- DEPLOYED SHAPES (provenance in the writeup) -------------------------
# ~/.exo/models/deepseek-ai--DeepSeek-V4-Flash-0731/config.json:
#   hidden_size=4096, moe_intermediate_size=2048, n_routed_experts=256,
#   num_experts_per_tok=6, num_hidden_layers=43
# TP worldSize=2 (MLX_JACCL_SHARDING_MODE=Tensor on the live cluster), so
# the per-rank expert intermediate is 2048//2 = 1024.
HIDDEN = 4096
MOE_INTERMEDIATE = 2048
TP = 2
PER_RANK_INTER = MOE_INTERMEDIATE // TP  # 1024
N_ROUTED_EXPERTS = 256
TOP_K = 6

M_ROWS = 4  # the campaign's batch/rows dimension

PEAK_BW_GBPS = 546.0  # M4 Max spec, the repo's standing reference ceiling

CHAIN_LEN = 300
N_POOL = 64
QUANT_CHUNK = 32  # experts quantized per chunk (memory discipline)


def sync():
    if hasattr(mx, "synchronize"):
        mx.synchronize()


def quantize_chunked(out_dims, in_dims, n_experts, group_size, bits, chunk, seed):
    """Real mx.quantize over expert chunks, so peak memory stays low.

    Returns (weight, scales, biases|None) concatenated over the expert axis.
    """
    mx.random.seed(seed)
    scale = (1.0 / in_dims) ** 0.5
    w_parts, s_parts, b_parts = [], [], []
    has_biases = None
    for start in range(0, n_experts, chunk):
        n = min(chunk, n_experts - start)
        src = mx.random.uniform(
            low=-scale, high=scale, shape=(n, out_dims, in_dims)
        ).astype(mx.bfloat16)
        mx.eval(src)
        packed = mx.quantize(src, group_size=group_size, bits=bits, mode="affine")
        mx.eval(packed)
        w, s, *rest = packed
        has_biases = bool(rest)
        w_parts.append(w)
        s_parts.append(s)
        if rest:
            b_parts.append(rest[0])
        del src, packed, w, s, rest
        print(
            f"[i11]     chunk@{start} active={mx.get_active_memory() / 1e9:.2f}GB "
            f"cache={mx.get_cache_memory() / 1e9:.2f}GB",
            file=sys.stderr,
            flush=True,
        )
    weight = mx.concatenate(w_parts, axis=0)
    scales = mx.concatenate(s_parts, axis=0)
    biases = mx.concatenate(b_parts, axis=0) if has_biases else None
    mx.eval(weight, scales, biases if biases is not None else mx.array(0))
    del w_parts, s_parts, b_parts
    gc.collect()
    mx.clear_cache()
    return weight, scales, biases


class Projection:
    __slots__ = ("weight", "scales", "biases", "group_size", "bits")

    def __init__(self, weight, scales, biases, group_size, bits):
        self.weight = weight
        self.scales = scales
        self.biases = biases
        self.group_size = group_size
        self.bits = bits

    def __call__(self, x, indices):
        # EXACT production call shape: QuantizedSwitchLinear.__call__ in
        # mlx-lm/mlx_lm/models/switch_layers.py:76-91 -- rhs_indices only,
        # transpose=True, affine mode, sorted_indices=False (M=4 gives
        # indices.size=24 < 64, so SwitchGLU's do_sort gate is False).
        return mx.gather_qmm(
            x,
            self.weight,
            self.scales,
            self.biases,
            rhs_indices=indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode="affine",
            sorted_indices=False,
        )

    def nbytes(self):
        total = self.weight.nbytes + self.scales.nbytes
        if self.biases is not None:
            total += self.biases.nbytes
        return total


def build_arm(bits, group_size, seed):
    """gate/up: (E, PER_RANK_INTER, HIDDEN); down: (E, HIDDEN, PER_RANK_INTER)."""
    gate = Projection(
        *quantize_chunked(
            PER_RANK_INTER, HIDDEN, N_ROUTED_EXPERTS, group_size, bits, QUANT_CHUNK, seed
        ),
        group_size,
        bits,
    )
    up = Projection(
        *quantize_chunked(
            PER_RANK_INTER,
            HIDDEN,
            N_ROUTED_EXPERTS,
            group_size,
            bits,
            QUANT_CHUNK,
            seed + 1,
        ),
        group_size,
        bits,
    )
    down = Projection(
        *quantize_chunked(
            HIDDEN,
            PER_RANK_INTER,
            N_ROUTED_EXPERTS,
            group_size,
            bits,
            QUANT_CHUNK,
            seed + 2,
        ),
        group_size,
        bits,
    )
    return gate, up, down


def switch_forward(projs, x, idx):
    """SwitchGLU-equivalent forward (mlx-lm switch_layers.py:177-203)."""
    gate, up, down = projs
    xe = mx.expand_dims(x, (-2, -3))  # (M, 1, 1, HIDDEN)
    x_up = up(xe, idx)
    x_gate = gate(xe, idx)
    x_act = mx.sigmoid(x_gate) * x_gate * x_up  # silu-glu, same op count
    out = down(x_act, idx)
    return out.squeeze(-2)  # (M, TOP_K, HIDDEN)


def run_chained(projs, x, idx_arrs, chain_len):
    """Chained dependency graph, ONE eval at the end. No serial sync."""
    n_pool = len(idx_arrs)
    carry = x
    last = None
    for i in range(chain_len):
        out = switch_forward(projs, carry, idx_arrs[i % n_pool])
        last = out
        carry = x + 1e-9 * mx.mean(out, axis=-2).astype(x.dtype)
    t0 = time.perf_counter()
    mx.eval(last, carry)
    sync()
    return time.perf_counter() - t0


def run_chained_baseline(x, idx_arrs, chain_len, hidden):
    """The IDENTICAL carry chain with the gather_qmm calls removed.

    Measures the scaffolding cost (expand_dims / mean / add) so it can be
    subtracted from the arm timings -- the reported us/call is then the
    marginal cost of the three MoE kernels, not the harness.
    """
    n_pool = len(idx_arrs)
    carry = x
    last = None
    for i in range(chain_len):
        idx = idx_arrs[i % n_pool]
        fake = mx.broadcast_to(
            mx.expand_dims(carry, -2), (carry.shape[0], idx.shape[1], hidden)
        )
        last = fake
        carry = x + 1e-9 * mx.mean(fake, axis=-2).astype(x.dtype)
    t0 = time.perf_counter()
    mx.eval(last, carry)
    sync()
    return time.perf_counter() - t0


def make_index_pool(m_rows, n_pool, n_experts, top_k, seed):
    rng = np.random.default_rng(seed)
    return [
        mx.array(rng.integers(0, n_experts, size=(m_rows, top_k)).astype(np.uint32))
        for _ in range(n_pool)
    ]


def gpu_usage():
    try:
        import urllib.request

        with urllib.request.urlopen("http://localhost:52415/metrics", timeout=5) as r:
            for line in r.read().decode().splitlines():
                if line.startswith("exo_gpu_usage_ratio"):
                    return float(line.rsplit(" ", 1)[1])
    except Exception:
        return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, nargs="+", default=[6, 5, 4])
    ap.add_argument("--group-size", type=int, default=32)
    ap.add_argument("--chain-len", type=int, default=CHAIN_LEN)
    ap.add_argument("--n-pool", type=int, default=N_POOL)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warmup-chain", type=int, default=30)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument(
        "--skip-baseline",
        action="store_true",
        help="diagnostic: skip the empty-graph baseline phase entirely",
    )
    args = ap.parse_args()

    log = lambda m: print(m, file=sys.stderr, flush=True)  # noqa: E731

    log(f"[i11] mlx {getattr(mx, '__version__', '?')} device {mx.default_device()}")
    log(
        f"[i11] shapes: E={N_ROUTED_EXPERTS} hidden={HIDDEN} "
        f"per_rank_inter={PER_RANK_INTER} top_k={TOP_K} M={M_ROWS}"
    )
    log(f"[i11] mode=affine group_size={args.group_size} bits={args.bits}")
    log(f"[i11] chain_len={args.chain_len} n_pool={args.n_pool} reps={args.reps}")

    results = {
        "mlx_version": getattr(mx, "__version__", "?"),
        "mode": "affine",
        "group_size": args.group_size,
        "chain_len": args.chain_len,
        "n_pool": args.n_pool,
        "reps": args.reps,
        "M_rows": M_ROWS,
        "top_k": TOP_K,
        "n_pairs_per_call": M_ROWS * TOP_K,
        "shapes": {
            "hidden": HIDDEN,
            "per_rank_inter": PER_RANK_INTER,
            "n_routed_experts": N_ROUTED_EXPERTS,
            "tp": TP,
        },
        "gpu_usage_before": gpu_usage(),
        "arms": {},
        "rep_order": [],
    }

    x = mx.random.normal(shape=(M_ROWS, HIDDEN)).astype(mx.bfloat16)
    mx.eval(x)
    idx_arrs = make_index_pool(M_ROWS, args.n_pool, N_ROUTED_EXPERTS, TOP_K, seed=1011)
    mx.eval(idx_arrs)

    # ---- empty-graph baseline (scaffolding cost, subtracted later) ----
    if args.skip_baseline:
        base_samples = [0.0]
    else:
        run_chained_baseline(x, idx_arrs, args.warmup_chain, HIDDEN)
        base_samples = []
        for _ in range(5):
            w = run_chained_baseline(x, idx_arrs, args.chain_len, HIDDEN)
            base_samples.append(w / args.chain_len * 1e6)
    base_us = statistics.median(base_samples)
    results["baseline_us_per_call"] = {
        "median": base_us,
        "min": min(base_samples),
        "max": max(base_samples),
        "samples": base_samples,
    }
    log(
        f"[i11] empty-graph baseline: {base_us:.2f} us/call "
        f"(range {min(base_samples):.2f}-{max(base_samples):.2f})"
    )

    # The baseline phase leaves MLX's buffer cache populated. On a node
    # whose GPU working set is already dominated by the LIVE runner's
    # resident weights, that reclaimable-but-unreclaimed cache is enough
    # to push the subsequent multi-GB quantize over the Metal working-set
    # limit ("Insufficient Memory" on the command buffer). Release it
    # explicitly before allocating arm weights.
    gc.collect()
    mx.clear_cache()
    log(
        f"[i11] post-baseline: active={mx.get_active_memory() / 1e9:.2f}GB "
        f"cache={mx.get_cache_memory() / 1e9:.2f}GB"
    )

    per_arm_samples = {b: [] for b in args.bits}
    per_arm_bytes = {}
    scaling = {}

    # ---- ONE ARM RESIDENT AT A TIME -------------------------------------
    # All three arms together are ~7.3 GB of quantized expert weights. On a
    # node whose Metal working set is already dominated by the LIVE runner's
    # resident model, holding all three at once intermittently exceeds the
    # working-set limit and the command buffer fails with "Insufficient
    # Memory" -- observed transiently, tracking the live cluster's own
    # allocation. So each arm is built, measured, and freed within the rep.
    #
    # Interleaving is preserved at the REP level: every rep visits all three
    # bit widths, and the visit order alternates (6,5,4 / 4,5,6 / ...), so a
    # monotonic contention drift cannot load preferentially onto one arm.
    # Rebuilding also means each rep's weights land on different physical
    # pages, which guards against a fixed-page residency artifact.
    def build_arm_retry(bits, attempts=5):
        for attempt in range(attempts):
            try:
                return build_arm(bits, args.group_size, seed=7000 + bits + 97 * attempt)
            except RuntimeError as exc:
                if "Insufficient Memory" not in str(exc) or attempt == attempts - 1:
                    raise
                log(f"[i11]   b={bits} build OOM (attempt {attempt + 1}) -- backing off")
                gc.collect()
                mx.clear_cache()
                time.sleep(20)
        raise RuntimeError("unreachable")

    for rep in range(args.reps):
        order = list(args.bits) if rep % 2 == 0 else list(reversed(args.bits))
        results["rep_order"].append(order)
        log(f"[i11] --- rep {rep + 1}/{args.reps} order={order} ---")
        for bits in order:
            projs = build_arm_retry(bits)

            if bits not in per_arm_bytes:
                per_expert = sum(p.nbytes() for p in projs) / N_ROUTED_EXPERTS
                per_arm_bytes[bits] = {
                    "bytes_per_expert_all3_projs": per_expert,
                    "bytes_per_call": per_expert * M_ROWS * TOP_K,
                    "arrays": {
                        name: {
                            "weight_shape": list(p.weight.shape),
                            "weight_dtype": str(p.weight.dtype),
                            "weight_nbytes": p.weight.nbytes,
                            "scales_shape": list(p.scales.shape),
                            "scales_dtype": str(p.scales.dtype),
                            "scales_nbytes": p.scales.nbytes,
                            "biases_nbytes": (
                                None if p.biases is None else p.biases.nbytes
                            ),
                        }
                        for name, p in zip(("gate", "up", "down"), projs)
                    },
                }
                log(
                    f"[i11]   b={bits} bytes/expert(3 projs)="
                    f"{per_expert:,.0f} B  bytes/call="
                    f"{per_expert * M_ROWS * TOP_K:,.0f} B"
                )

            # warmup (JIT-compiles this kernel variant; not timed)
            run_chained(projs, x, idx_arrs, args.warmup_chain)

            # elision detector, once per bit width
            if bits not in scaling:
                sc = {}
                for L in (args.chain_len // 3, args.chain_len * 2 // 3, args.chain_len):
                    w = run_chained(projs, x, idx_arrs, L)
                    sc[L] = {"wall_s": w, "us_per_call": w / L * 1e6}
                    log(
                        f"[i11]   b={bits} scaling chain_len={L} "
                        f"us/call={w / L * 1e6:.2f}"
                    )
                scaling[bits] = sc

            gpu_pre = gpu_usage()
            wall = run_chained(projs, x, idx_arrs, args.chain_len)
            raw_us = wall / args.chain_len * 1e6
            net_us = raw_us - base_us
            gpu_post = gpu_usage()
            per_arm_samples[bits].append(
                {
                    "raw_us": raw_us,
                    "net_us": net_us,
                    "gpu_before": gpu_pre,
                    "gpu_after": gpu_post,
                }
            )
            log(
                f"[i11]   b={bits} raw={raw_us:.2f} us/call net={net_us:.2f} "
                f"us/call gpu={gpu_pre}->{gpu_post}"
            )

            del projs
            gc.collect()
            mx.clear_cache()

    results["gpu_usage_after"] = gpu_usage()
    results["scaling_check"] = scaling

    for bits in args.bits:
        nets = sorted(s["net_us"] for s in per_arm_samples[bits])
        raws = sorted(s["raw_us"] for s in per_arm_samples[bits])
        gpu_obs = [
            v
            for s in per_arm_samples[bits]
            for v in (s.get("gpu_before"), s.get("gpu_after"))
            if v is not None
        ]
        bpc = per_arm_bytes[bits]["bytes_per_call"]
        med = statistics.median(nets)
        results["arms"][str(bits)] = {
            "bytes": per_arm_bytes[bits],
            "raw_us_per_call": {
                "min": raws[0],
                "median": statistics.median(raws),
                "max": raws[-1],
                "samples": raws,
            },
            "net_us_per_call": {
                "min": nets[0],
                "median": med,
                "max": nets[-1],
                "samples": nets,
            },
            "gbps": {
                "at_median": bpc / (med * 1e-6) / 1e9,
                "at_min_us": bpc / (nets[0] * 1e-6) / 1e9,
                "at_max_us": bpc / (nets[-1] * 1e-6) / 1e9,
            },
            "pct_peak_at_median": 100 * (bpc / (med * 1e-6) / 1e9) / PEAK_BW_GBPS,
            "gpu_usage_observed": {
                "min": min(gpu_obs) if gpu_obs else None,
                "max": max(gpu_obs) if gpu_obs else None,
            },
            "per_rep": per_arm_samples[bits],
        }

    # ratios vs the 6-bit baseline arm
    if 6 in args.bits:
        b6_net = results["arms"]["6"]["net_us_per_call"]["median"]
        b6_raw = results["arms"]["6"]["raw_us_per_call"]["median"]
        for bits in args.bits:
            a = results["arms"][str(bits)]
            a["ratio_vs_6bit_net"] = a["net_us_per_call"]["median"] / b6_net
            a["ratio_vs_6bit_raw"] = a["raw_us_per_call"]["median"] / b6_raw
            r = a["ratio_vs_6bit_net"]
            a["band"] = (
                "FAST" if r <= 0.90 else ("MARGINAL" if r < 0.98 else "SLOW/generic")
            )

    for bits in args.bits:
        a = results["arms"][str(bits)]
        log(
            f"[i11] RESULT b={bits}: net {a['net_us_per_call']['min']:.2f}-"
            f"{a['net_us_per_call']['max']:.2f} us/call "
            f"(median {a['net_us_per_call']['median']:.2f}), "
            f"{a['gbps']['at_median']:.1f} GB/s "
            f"({a['pct_peak_at_median']:.1f}% of {PEAK_BW_GBPS}), "
            f"ratio={a.get('ratio_vs_6bit_net', float('nan')):.4f} "
            f"band={a.get('band', '?')}"
        )

    blob = json.dumps(results, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(blob)
        log(f"[i11] wrote {args.out}")
    else:
        print(blob)


if __name__ == "__main__":
    main()
