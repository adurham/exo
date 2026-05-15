"""DSv4 top-K microbench - replace argsort(-scores)[..., :K] with a real top-K kernel."""

from __future__ import annotations
import time
import statistics
import mlx.core as mx

B = 1
L = 1
P = 25000
K = 160
N_LAYERS = 21
N_ITERS = 300
N_WARMUP = 30
SEED = 42
DTYPE = mx.bfloat16
T_THREADS = 256
K_LOCAL = 4
CANDIDATES = T_THREADS * K_LOCAL
assert CANDIDATES >= K


def make_scores(seed):
    mx.random.seed(seed)
    s = mx.random.normal((B, L, P), dtype=DTYPE)
    s = mx.maximum(s, 0)
    return s


def ref_topk(scores):
    return mx.argsort(-scores, axis=-1)[..., :K]


def ref_argpartition(scores):
    return mx.argpartition(scores, kth=P - K, axis=-1)[..., -K:]


def _read_kernel_src():
    with open("/tmp/topk_kernel.metal") as f:
        return f.read().replace("AND_OP", chr(38))  # chr(38) is ampersand, single


def _build_topk_kernel():
    return mx.fast.metal_kernel(
        name="dsv4_topk_v1",
        input_names=["scores"],
        output_names=["out_idx"],
        source=_read_kernel_src(),
        header=f"""
        constant uint B_ = {B};
        constant uint L_ = {L};
        constant uint P_ = {P};
        constant uint K_ = {K};
        constant uint T_ = {T_THREADS};
        constant uint K_LOCAL_ = {K_LOCAL};
        constant uint CANDIDATES_ = {CANDIDATES};
        """,
        ensure_row_contiguous=True,
    )


def fused_topk(kernel, scores):
    outs = kernel(
        inputs=[scores],
        grid=(T_THREADS * B * L, 1, 1),
        threadgroup=(T_THREADS, 1, 1),
        output_shapes=[(B, L, K)],
        output_dtypes=[mx.int32],
    )
    return outs[0]


def check_equiv(kernel, n_seeds=5):
    print("Numerical equivalence:")
    overlaps = []
    for seed in range(n_seeds):
        scores = make_scores(seed)
        ref_idx = ref_topk(scores)
        fu_idx  = fused_topk(kernel, scores)
        mx.eval(ref_idx, fu_idx)
        ref_set = set(ref_idx.tolist()[0][0])
        fu_set  = set(fu_idx.tolist()[0][0])
        overlap = len(ref_set.intersection(fu_set))
        overlaps.append(overlap)
        print(f"  seed={seed}: top-{K} overlap = {overlap}/{K}")
    return overlaps


def time_call(fn, n=N_ITERS, warmup=N_WARMUP):
    for _ in range(warmup):
        out = fn(); mx.eval(out)
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        out = fn(); mx.eval(out)
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples) * 1e6


def time_chain(fn_list, n=N_ITERS, warmup=N_WARMUP):
    for _ in range(warmup):
        outs = [f() for f in fn_list]; mx.eval(*outs)
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        outs = [f() for f in fn_list]; mx.eval(*outs)
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples) * 1e6


def main():
    print("=" * 70)
    print(f"DSv4 top-K microbench (B={B} L={L} P={P} K={K})")
    print(f"  T={T_THREADS} K_LOCAL={K_LOCAL} CANDIDATES={CANDIDATES}")
    print(f"  Device: {mx.default_device()}")
    print("=" * 70)
    print("\nBuilding kernel...")
    kernel = _build_topk_kernel()
    print("  ok")

    overlaps = check_equiv(kernel)
    avg = sum(overlaps) / len(overlaps)
    print(f"  Avg overlap: {avg:.1f}/{K} ({avg/K*100:.1f}%)")

    scores = make_scores(SEED)
    mx.eval(scores)

    print(f"\nPer-call timing (median us over {N_ITERS}):")
    t_argsort = time_call(lambda: ref_topk(scores))
    print(f"  argsort + slice (current):   {t_argsort:7.2f} us")
    t_argpart = time_call(lambda: ref_argpartition(scores))
    print(f"  argpartition + slice:        {t_argpart:7.2f} us")
    t_fu = time_call(lambda: fused_topk(kernel, scores))
    print(f"  fused topk kernel:           {t_fu:7.2f} us")
    print(f"  speedup vs argsort:          {t_argsort/t_fu:.2f}x")

    print(f"\nPipelined {N_LAYERS}-call chain (median us):")
    score_list = []
    for layer in range(N_LAYERS):
        s = make_scores(SEED + layer)
        mx.eval(s)
        score_list.append(s)

    fn_argsort = [lambda s=s: ref_topk(s) for s in score_list]
    fn_fused   = [lambda s=s: fused_topk(kernel, s) for s in score_list]

    p_argsort = time_chain(fn_argsort)
    p_fu = time_chain(fn_fused)
    print(f"  argsort chain:  {p_argsort:8.1f} us total = {p_argsort/N_LAYERS:5.1f} us/call")
    print(f"  fused chain:    {p_fu:8.1f} us total = {p_fu/N_LAYERS:5.1f} us/call")
    print(f"  speedup:        {p_argsort/p_fu:.2f}x")

    print()
    print("=" * 70)
    speedup = p_argsort / p_fu
    print(f"PIPELINED SPEEDUP vs argsort: {speedup:.2f}x")
    if speedup >= 1.7:
        print(f"  ABOVE 1.7x GATE")
    elif speedup >= 1.3:
        print(f"  marginal")
    else:
        print(f"  below gate")

    saved = (p_argsort - p_fu) / 1000.0
    proj = 1000.0 / max(0.1, 28.5 - max(0, saved))
    print(f"\nCluster projection (21 calls/token, baseline 28.5 ms = 29.2 t/s):")
    print(f"  saved per token: {saved:+.2f} ms")
    print(f"  projected:       {proj:.1f} t/s")
    print("=" * 70)


if __name__ == "__main__":
    main()
