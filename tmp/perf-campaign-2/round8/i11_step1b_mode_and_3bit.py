#!/usr/bin/env python3
"""I11 STEP 1B -- deployed-mxfp4 mode check + downward 3-bit extension.

WHY THIS EXISTS. The step-1 harness
(tmp/perf-campaign-2/round8/i11_gather_qmm_bits_microbench.py) hardcoded
mode="affine" everywhere (quantize_chunked():96 and
Projection.__call__():145). Its bits=4 arm therefore measured
`affine_gather_qmv_fast_bfloat16_t_gs_32_b_4`, NOT the kernel production
runs. Production routed experts are mode="mxfp4", group_size=32, bits=4
(deepseek_v4.py make_quantization_config), which dispatches the SEPARATE
`mxfp4_gather_qmv_fast_bfloat16_t_gs_32_b_4` kernel from
mlx/backend/metal/kernels/fp_quantized.metal and carries a DIFFERENT byte
footprint (uint8 e8m0 scales, NO biases). This harness measures the
deployed configuration head-to-head against the original affine arm, and
extends the sweep DOWN to 3-bit.

METHOD OF RECORD is unchanged from step 1 -- the R1 CHAINED-GRAPH
construction. SERIAL-SYNC TIMING IS NOT USED ANYWHERE IN THIS FILE:

  - Dependency-chained graph of CHAIN_LEN SwitchGLU-equivalent calls,
    each a real data dependency of the next (carry = x + 1e-9*mean(out)),
    with ONE mx.eval() at the very end. No mx.eval() inside a timing loop.
  - ROTATED routing indices from a pool of N_POOL independently-drawn
    (M, top_k) index sets, so no expert weight set sits warm in cache.
  - A measured EMPTY-GRAPH BASELINE (identical carry chain, gather_qmm
    calls removed) is subtracted; reported us/call is the marginal cost
    of the MoE kernels.
  - Chain-length SCALING CHECK (elision detector) per arm.
  - Ranges (min/median/max over reps) reported, never bare means.

ARMS ARE INTERLEAVED. With 6 arms, plain forward/reverse alternation is
not enough to decorrelate arm from position, so each rep ROTATES the arm
list by the rep index and reverses on odd reps. Every arm therefore
visits a spread of positions across reps, so monotonic drift from
live-cluster GPU contention cannot load preferentially onto one arm.

MODE SUPPORT IS PROBED, NEVER ASSUMED. probe_support() actually calls
mx.quantize + mx.gather_qmm for every requested (mode, group_size, bits)
combination and records the real exception text for the unsupported ones.
An unsupported combination is reported as UNSUPPORTED; it is NEVER
silently substituted with affine.

MEMORY DISCIPLINE (the cluster is LIVE): weights are quantized in EXPERT
CHUNKS (real mx.quantize on real random weights) and only ONE arm is
resident at a time -- built, measured, and freed inside each rep.
"""

import argparse
import gc
import itertools
import json
import statistics
import sys
import time

import mlx.core as mx
import numpy as np

# ---- DEPLOYED SHAPES (identical to step 1; provenance in that writeup) ----
HIDDEN = 4096
MOE_INTERMEDIATE = 2048
TP = 2
PER_RANK_INTER = MOE_INTERMEDIATE // TP  # 1024
N_ROUTED_EXPERTS = 256
TOP_K = 6

M_ROWS = 4
PEAK_BW_GBPS = 546.0

CHAIN_LEN = 300
N_POOL = 64
QUANT_CHUNK = 32

# ---- ARMS -----------------------------------------------------------------
# (mode, group_size, bits, label). The DEPLOYED arm is mxfp4/32/4 -- it is
# the 1.000x reference for every ratio this harness reports.
DEPLOYED_ARM = ("mxfp4", 32, 4)

DEFAULT_ARMS = [
    ("affine", 32, 3),  # the remaining DOWNWARD lever
    ("mxfp4", 32, 4),  # DEPLOYED routed-expert kernel  <-- reference
    ("affine", 32, 4),  # what step 1 actually measured as "bits=4"
    ("affine", 32, 5),  # step 1's 5-bit arm
    ("affine", 32, 6),  # step 1's 6-bit arm
    ("mxfp8", 32, 8),  # DEPLOYED shared_experts / attention precision
]

# Combinations to PROBE for support. Anything that throws is reported as
# UNSUPPORTED with its real error string.
PROBE_COMBOS = [
    ("mxfp3", 32, 3),
    ("mxfp4", 32, 3),
    ("mxfp5", 32, 5),
    ("mxfp6", 32, 6),
    ("mxfp4", 32, 4),
    ("mxfp4", 64, 4),
    ("mxfp8", 32, 8),
    ("nvfp4", 16, 4),
    ("affine", 32, 3),
    ("affine", 32, 4),
    ("affine", 32, 5),
    ("affine", 32, 6),
    ("affine", 32, 8),
]


def arm_key(mode, group_size, bits):
    return f"{mode}/gs{group_size}/b{bits}"


def sync():
    if hasattr(mx, "synchronize"):
        mx.synchronize()


# ---------------------------------------------------------------------------
# SUPPORT PROBE -- real calls, real errors, no substitution
# ---------------------------------------------------------------------------
def probe_support(log):
    """Actually run quantize + gather_qmm for each candidate combination.

    Uses a tiny but dispatch-legal shape so a failure is a genuine
    mode/bits/group_size rejection, not a shape artifact.
    """
    out = {}
    e_small, n_small, k_small = 4, 64, 512  # N%8==0, K%512==0 -> fast gate
    for mode, gs, bits in PROBE_COMBOS:
        key = arm_key(mode, gs, bits)
        rec = {"mode": mode, "group_size": gs, "bits": bits}
        try:
            src = mx.random.uniform(
                low=-0.05, high=0.05, shape=(e_small, n_small, k_small)
            ).astype(mx.bfloat16)
            mx.eval(src)
            packed = mx.quantize(src, group_size=gs, bits=bits, mode=mode)
            mx.eval(packed)
            w, s, *rest = packed
            b = rest[0] if rest else None
            rec["quantize"] = "OK"
            rec["has_biases"] = bool(rest)
            rec["weight_dtype"] = str(w.dtype)
            rec["scales_dtype"] = str(s.dtype)
            rec["weight_shape"] = list(w.shape)
            rec["scales_shape"] = list(s.shape)
        except Exception as exc:  # noqa: BLE001 -- the error text IS the finding
            rec["quantize"] = "UNSUPPORTED"
            rec["error"] = f"{type(exc).__name__}: {exc}"
            out[key] = rec
            log(f"[i11b] PROBE {key}: quantize UNSUPPORTED -- {rec['error']}")
            continue
        try:
            xq = mx.random.normal(shape=(2, 1, 1, k_small)).astype(mx.bfloat16)
            idx = mx.array(
                np.random.default_rng(0)
                .integers(0, e_small, size=(2, 3))
                .astype(np.uint32)
            )
            y = mx.gather_qmm(
                xq,
                w,
                s,
                b,
                rhs_indices=idx,
                transpose=True,
                group_size=gs,
                bits=bits,
                mode=mode,
                sorted_indices=False,
            )
            mx.eval(y)
            rec["gather_qmm"] = "OK"
            rec["supported"] = True
            log(
                f"[i11b] PROBE {key}: SUPPORTED "
                f"(biases={rec['has_biases']} scales={rec['scales_dtype']})"
            )
        except Exception as exc:  # noqa: BLE001
            rec["gather_qmm"] = "UNSUPPORTED"
            rec["error"] = f"{type(exc).__name__}: {exc}"
            rec["supported"] = False
            log(f"[i11b] PROBE {key}: gather_qmm UNSUPPORTED -- {rec['error']}")
        out[key] = rec
        del src, packed, w, s
        gc.collect()
        mx.clear_cache()
    return out


# ---------------------------------------------------------------------------
# BUILD
# ---------------------------------------------------------------------------
def quantize_chunked(out_dims, in_dims, n_experts, group_size, bits, mode, chunk, seed):
    """Real mx.quantize over expert chunks so peak memory stays low.

    Returns (weight, scales, biases|None) concatenated over the expert axis.
    biases is None for the mx/nv fp modes, which genuinely have none.
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
        packed = mx.quantize(src, group_size=group_size, bits=bits, mode=mode)
        mx.eval(packed)
        w, s, *rest = packed
        has_biases = bool(rest)
        w_parts.append(w)
        s_parts.append(s)
        if rest:
            b_parts.append(rest[0])
        del src, packed, w, s, rest
    weight = mx.concatenate(w_parts, axis=0)
    scales = mx.concatenate(s_parts, axis=0)
    biases = mx.concatenate(b_parts, axis=0) if has_biases else None
    mx.eval(weight, scales, biases if biases is not None else mx.array(0))
    del w_parts, s_parts, b_parts
    gc.collect()
    mx.clear_cache()
    return weight, scales, biases


class Projection:
    __slots__ = ("weight", "scales", "biases", "group_size", "bits", "mode")

    def __init__(self, weight, scales, biases, group_size, bits, mode):
        self.weight = weight
        self.scales = scales
        self.biases = biases
        self.group_size = group_size
        self.bits = bits
        self.mode = mode

    def __call__(self, x, indices):
        # EXACT production call shape: QuantizedSwitchLinear.__call__ in
        # mlx-lm/mlx_lm/models/switch_layers.py:76-91 -- rhs_indices only,
        # transpose=True, sorted_indices=False (M=4 -> indices.size=24 < 64,
        # so SwitchGLU's do_sort gate is False). The ONLY thing that varies
        # across arms is (mode, group_size, bits).
        return mx.gather_qmm(
            x,
            self.weight,
            self.scales,
            self.biases,
            rhs_indices=indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            sorted_indices=False,
        )

    def nbytes(self):
        total = self.weight.nbytes + self.scales.nbytes
        if self.biases is not None:
            total += self.biases.nbytes
        return total


def build_arm(mode, group_size, bits, seed):
    """gate/up: (E, PER_RANK_INTER, HIDDEN); down: (E, HIDDEN, PER_RANK_INTER)."""
    specs = [
        (PER_RANK_INTER, HIDDEN, seed),
        (PER_RANK_INTER, HIDDEN, seed + 1),
        (HIDDEN, PER_RANK_INTER, seed + 2),
    ]
    return tuple(
        Projection(
            *quantize_chunked(
                od, idim, N_ROUTED_EXPERTS, group_size, bits, mode, QUANT_CHUNK, sd
            ),
            group_size,
            bits,
            mode,
        )
        for od, idim, sd in specs
    )


# ---------------------------------------------------------------------------
# CHAINED TIMING (no serial sync anywhere)
# ---------------------------------------------------------------------------
def switch_forward(projs, x, idx):
    """SwitchGLU-equivalent forward (mlx-lm switch_layers.py:177-203)."""
    gate, up, down = projs
    xe = mx.expand_dims(x, (-2, -3))
    x_up = up(xe, idx)
    x_gate = gate(xe, idx)
    x_act = mx.sigmoid(x_gate) * x_gate * x_up
    out = down(x_act, idx)
    return out.squeeze(-2)


def run_chained(projs, x, idx_arrs, chain_len):
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
    """The IDENTICAL carry chain with the gather_qmm calls removed."""
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


def rep_order(arms, rep):
    """Rotate by rep, reverse on odd reps.

    With 6 arms and plain forward/reverse alternation an arm would only ever
    occupy 2 of 6 positions. Rotation spreads each arm across positions so a
    monotonic drift is decorrelated from arm identity.
    """
    rotated = list(arms[rep % len(arms) :]) + list(arms[: rep % len(arms)])
    return rotated if rep % 2 == 0 else list(reversed(rotated))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--arms",
        type=str,
        nargs="+",
        default=None,
        help="arm specs as mode:group_size:bits, e.g. mxfp4:32:4",
    )
    ap.add_argument("--chain-len", type=int, default=CHAIN_LEN)
    ap.add_argument("--n-pool", type=int, default=N_POOL)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warmup-chain", type=int, default=30)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--probe-only", action="store_true")
    args = ap.parse_args()

    log = lambda m: print(m, file=sys.stderr, flush=True)  # noqa: E731

    if args.arms:
        arms = []
        for spec in args.arms:
            mode, gs, bits = spec.split(":")
            arms.append((mode, int(gs), int(bits)))
    else:
        arms = list(DEFAULT_ARMS)

    log(f"[i11b] mlx {getattr(mx, '__version__', '?')} device {mx.default_device()}")
    log(
        f"[i11b] shapes: E={N_ROUTED_EXPERTS} hidden={HIDDEN} "
        f"per_rank_inter={PER_RANK_INTER} top_k={TOP_K} M={M_ROWS}"
    )
    log(f"[i11b] arms={[arm_key(*a) for a in arms]}")
    log(f"[i11b] chain_len={args.chain_len} n_pool={args.n_pool} reps={args.reps}")

    results = {
        "mlx_version": getattr(mx, "__version__", "?"),
        "chain_len": args.chain_len,
        "n_pool": args.n_pool,
        "reps": args.reps,
        "M_rows": M_ROWS,
        "top_k": TOP_K,
        "n_pairs_per_call": M_ROWS * TOP_K,
        "deployed_arm": arm_key(*DEPLOYED_ARM),
        "arms_requested": [arm_key(*a) for a in arms],
        "shapes": {
            "hidden": HIDDEN,
            "per_rank_inter": PER_RANK_INTER,
            "n_routed_experts": N_ROUTED_EXPERTS,
            "tp": TP,
        },
        "gpu_usage_before": gpu_usage(),
        "support_probe": {},
        "arms": {},
        "rep_order": [],
    }

    # ---- support probe FIRST: never bench an arm we have not proven runs ----
    results["support_probe"] = probe_support(log)
    unsupported = [
        k for k, v in results["support_probe"].items() if not v.get("supported")
    ]
    results["unsupported"] = unsupported
    log(f"[i11b] UNSUPPORTED combinations: {unsupported}")

    if args.probe_only:
        blob = json.dumps(results, indent=2)
        if args.out:
            with open(args.out, "w") as f:
                f.write(blob)
            log(f"[i11b] wrote {args.out}")
        else:
            print(blob)
        return

    # Drop any requested arm the probe proved unsupported, rather than
    # substituting affine for it.
    kept = []
    for a in arms:
        k = arm_key(*a)
        rec = results["support_probe"].get(k)
        if rec is not None and not rec.get("supported"):
            log(f"[i11b] SKIPPING unsupported arm {k}")
            continue
        kept.append(a)
    arms = kept
    results["arms_benched"] = [arm_key(*a) for a in arms]

    x = mx.random.normal(shape=(M_ROWS, HIDDEN)).astype(mx.bfloat16)
    mx.eval(x)
    idx_arrs = make_index_pool(M_ROWS, args.n_pool, N_ROUTED_EXPERTS, TOP_K, seed=1011)
    mx.eval(idx_arrs)

    # ---- empty-graph baseline (scaffolding cost, subtracted later) ----
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
        f"[i11b] empty-graph baseline: {base_us:.2f} us/call "
        f"(range {min(base_samples):.2f}-{max(base_samples):.2f})"
    )

    gc.collect()
    mx.clear_cache()

    per_arm_samples = {arm_key(*a): [] for a in arms}
    per_arm_bytes = {}
    scaling = {}

    seed_counter = itertools.count(7000)

    def build_arm_retry(mode, gs, bits, attempts=5):
        for attempt in range(attempts):
            try:
                return build_arm(mode, gs, bits, seed=next(seed_counter) * 3)
            except RuntimeError as exc:
                if "Insufficient Memory" not in str(exc) or attempt == attempts - 1:
                    raise
                log(
                    f"[i11b]   {arm_key(mode, gs, bits)} build OOM "
                    f"(attempt {attempt + 1}) -- backing off"
                )
                gc.collect()
                mx.clear_cache()
                time.sleep(20)
        raise RuntimeError("unreachable")

    for rep in range(args.reps):
        order = rep_order(arms, rep)
        results["rep_order"].append([arm_key(*a) for a in order])
        log(f"[i11b] --- rep {rep + 1}/{args.reps} order={[arm_key(*a) for a in order]} ---")
        for mode, gs, bits in order:
            key = arm_key(mode, gs, bits)
            projs = build_arm_retry(mode, gs, bits)

            if key not in per_arm_bytes:
                per_expert = sum(p.nbytes() for p in projs) / N_ROUTED_EXPERTS
                per_arm_bytes[key] = {
                    "bytes_per_expert_all3_projs": per_expert,
                    "bytes_per_call": per_expert * M_ROWS * TOP_K,
                    "has_biases": projs[0].biases is not None,
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
                    f"[i11b]   {key} bytes/expert(3 projs)="
                    f"{per_expert:,.0f} B  bytes/call="
                    f"{per_expert * M_ROWS * TOP_K:,.0f} B  "
                    f"biases={projs[0].biases is not None}"
                )

            # warmup (JIT-compiles this kernel variant; not timed)
            run_chained(projs, x, idx_arrs, args.warmup_chain)

            # elision detector, once per arm
            if key not in scaling:
                sc = {}
                for L in (args.chain_len // 3, args.chain_len * 2 // 3, args.chain_len):
                    w = run_chained(projs, x, idx_arrs, L)
                    sc[L] = {"wall_s": w, "us_per_call": w / L * 1e6}
                    log(f"[i11b]   {key} scaling chain_len={L} us/call={w / L * 1e6:.2f}")
                scaling[key] = sc

            gpu_pre = gpu_usage()
            wall = run_chained(projs, x, idx_arrs, args.chain_len)
            raw_us = wall / args.chain_len * 1e6
            net_us = raw_us - base_us
            gpu_post = gpu_usage()
            per_arm_samples[key].append(
                {
                    "raw_us": raw_us,
                    "net_us": net_us,
                    "gpu_before": gpu_pre,
                    "gpu_after": gpu_post,
                }
            )
            log(
                f"[i11b]   {key} raw={raw_us:.2f} us/call net={net_us:.2f} "
                f"us/call gpu={gpu_pre}->{gpu_post}"
            )

            del projs
            gc.collect()
            mx.clear_cache()

    results["gpu_usage_after"] = gpu_usage()
    results["scaling_check"] = scaling

    for mode, gs, bits in arms:
        key = arm_key(mode, gs, bits)
        nets = sorted(s["net_us"] for s in per_arm_samples[key])
        raws = sorted(s["raw_us"] for s in per_arm_samples[key])
        gpu_obs = [
            v
            for s in per_arm_samples[key]
            for v in (s.get("gpu_before"), s.get("gpu_after"))
            if v is not None
        ]
        bpc = per_arm_bytes[key]["bytes_per_call"]
        med = statistics.median(nets)
        results["arms"][key] = {
            "mode": mode,
            "group_size": gs,
            "bits": bits,
            "is_mx_mode": mode != "affine",
            "bytes": per_arm_bytes[key],
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
            "per_rep": per_arm_samples[key],
        }

    # ---- ratios keyed to the DEPLOYED mxfp4 arm ----
    dep_key = arm_key(*DEPLOYED_ARM)
    if dep_key in results["arms"]:
        dep = results["arms"][dep_key]
        dep_net = dep["net_us_per_call"]["median"]
        dep_bytes = dep["bytes"]["bytes_per_call"]
        for key, a in results["arms"].items():
            a["ratio_vs_deployed_mxfp4_net"] = a["net_us_per_call"]["median"] / dep_net
            a["byte_ratio_vs_deployed_mxfp4"] = (
                a["bytes"]["bytes_per_call"] / dep_bytes
            )
            r = a["ratio_vs_deployed_mxfp4_net"]
            # Band logic restated against the deployed 4-bit baseline. For an
            # arm moving FEWER bytes than deployed, FAST means the time ratio
            # tracks the byte ratio; falling off the fast path shows up as
            # time decoupled from bytes.
            br = a["byte_ratio_vs_deployed_mxfp4"]
            a["time_over_byte_ratio"] = r / br
            if br < 1.0:
                a["band"] = (
                    "FAST"
                    if r <= br + 0.05
                    else ("MARGINAL" if r < br + 0.15 else "SLOW/generic")
                )
            else:
                a["band"] = "reference" if key == dep_key else "UPWARD (costs more)"

    for key, a in results["arms"].items():
        log(
            f"[i11b] RESULT {key}: net {a['net_us_per_call']['min']:.2f}-"
            f"{a['net_us_per_call']['max']:.2f} us/call "
            f"(median {a['net_us_per_call']['median']:.2f}), "
            f"{a['gbps']['at_median']:.1f} GB/s "
            f"({a['pct_peak_at_median']:.1f}% of {PEAK_BW_GBPS}), "
            f"ratio_vs_deployed={a.get('ratio_vs_deployed_mxfp4_net', float('nan')):.4f} "
            f"byte_ratio={a.get('byte_ratio_vs_deployed_mxfp4', float('nan')):.4f} "
            f"band={a.get('band', '?')}"
        )

    blob = json.dumps(results, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(blob)
        log(f"[i11b] wrote {args.out}")
    else:
        print(blob)


if __name__ == "__main__":
    main()
