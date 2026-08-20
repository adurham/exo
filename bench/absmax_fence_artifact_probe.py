"""Isolate whether local_absmax's ~400-420ms/call cost (from the shared-scale
probe, see docs/moe-allsum-sharedscale-root-cause-found-2026-08-19.md) is a
REAL cost of `mx.max(mx.abs(y))`, or an artifact of the probe's own
mx.eval(local_absmax) fence being the FIRST eval encountered after a long
backlog of unevaluated lazy-graph ops (the preceding MoE layer's compute),
so it ends up paying for that entire backlog's GPU round-trip instead of
just the reduction itself.

Shape matches the live probe data: (1, 2048, 4096) float32/bf16.

Three conditions, each timed with time.time() around a single mx.eval():
  A) "warm" y: y is created and mx.eval()'d BEFORE the timed region, so the
     timed mx.max(mx.abs(y)) has nothing else queued -- isolates the true
     cost of the reduction + a bare device round-trip.
  B) "cold behind matmul chain" y: y is the lazy output of N chained matmuls
     (unevaluated), mimicking "y is the tail of the MoE layer's still-lazy
     graph" -- timed region does mx.eval(mx.max(mx.abs(y))) covering BOTH the
     upstream chain AND the reduction, same shape as the probe's real
     in-situ call site.
  C) same cold chain, but with an explicit mx.eval(y) BEFORE the timed
     region (i.e. what happens if a fence already ran right before this
     phase) -- timed region is then a bare warm reduction again, isolating
     just the incremental cost of the reduction op itself when nothing else
     is backlogged.

If B >> A/C, the ~400ms figure is real ONLY as "cost of flushing the whole
backlogged graph the first time you force it" -- i.e. it's an artifact of
where in the lazy graph the FIRST eval fence happens to land, not a
per-call fixed cost of the abs-max reduction itself.
"""

import time

import mlx.core as mx

SHAPE = (1, 2048, 4096)
N_WARMUP = 5
N_TRIALS = 30
CHAIN_LEN = 8  # rough proxy for "several MoE-layer matmuls worth of lazy graph"


def make_chain(seed_key):
    """Return a fresh, UNEVALUATED array that is the tail of a matmul chain,
    same shape as the real y (the pre-all_sum MoE partial-sum activation)."""
    x = mx.random.normal(SHAPE, key=seed_key)
    w = mx.random.normal((SHAPE[-1], SHAPE[-1]), key=seed_key)
    for _ in range(CHAIN_LEN):
        x = mx.matmul(x, w)
        x = x * 1.0001  # cheap elementwise to keep the graph from collapsing
    return x


def bench_A_warm(n_trials):
    key = mx.random.key(0)
    times = []
    for i in range(n_trials):
        y = make_chain(key)
        mx.eval(y)  # fully materialize BEFORE the timed region
        t0 = time.time()
        local_absmax = mx.max(mx.abs(y)).reshape(1).astype(mx.float32)
        mx.eval(local_absmax)
        t1 = time.time()
        times.append((t1 - t0) * 1000.0)
    return times


def bench_B_cold(n_trials):
    key = mx.random.key(0)
    times = []
    for i in range(n_trials):
        y = make_chain(key)  # NOT evaluated yet -- whole chain is lazy
        t0 = time.time()
        local_absmax = mx.max(mx.abs(y)).reshape(1).astype(mx.float32)
        mx.eval(local_absmax)  # this is the FIRST eval fence hit
        t1 = time.time()
        times.append((t1 - t0) * 1000.0)
    return times


def bench_C_prefenced(n_trials):
    key = mx.random.key(0)
    times = []
    for i in range(n_trials):
        y = make_chain(key)
        mx.eval(y)  # pre-fence: simulates "a fence already ran before this phase"
        t0 = time.time()
        local_absmax = mx.max(mx.abs(y)).reshape(1).astype(mx.float32)
        mx.eval(local_absmax)
        t1 = time.time()
        times.append((t1 - t0) * 1000.0)
    return times


def bench_D_chain_alone(n_trials):
    """Cost of JUST evaluating the chain (no reduction at all) -- isolates
    how much of B's cost is "flush the backlog" vs "the reduction op"."""
    key = mx.random.key(0)
    times = []
    for i in range(n_trials):
        y = make_chain(key)
        t0 = time.time()
        mx.eval(y)
        t1 = time.time()
        times.append((t1 - t0) * 1000.0)
    return times


def summarize(name, times):
    s = sorted(times)
    n = len(s)
    p50 = s[n // 2]
    p99 = s[min(n - 1, int(n * 0.99))]
    print(
        f"{name:28s} n={n:3d} min={s[0]:8.3f}ms p50={p50:8.3f}ms "
        f"p99={p99:8.3f}ms max={s[-1]:8.3f}ms mean={sum(s)/n:8.3f}ms"
    )


if __name__ == "__main__":
    print(f"device={mx.default_device()} shape={SHAPE} chain_len={CHAIN_LEN}")
    print("warming up (compile caches, memory pool, etc.)...")
    bench_A_warm(N_WARMUP)
    bench_B_cold(N_WARMUP)
    bench_C_prefenced(N_WARMUP)
    bench_D_chain_alone(N_WARMUP)

    print()
    print("=== Results (isolated single-process, no distributed) ===")
    a = bench_A_warm(N_TRIALS)
    summarize("A: warm y, timed reduction", a)

    b = bench_B_cold(N_TRIALS)
    summarize("B: cold/lazy y, timed reduction+chain", b)

    c = bench_C_prefenced(N_TRIALS)
    summarize("C: pre-fenced y, timed reduction", c)

    d = bench_D_chain_alone(N_TRIALS)
    summarize("D: cold y, timed chain eval only (no reduction)", d)

    print()
    print("=== Interpretation ===")
    import statistics as st

    print(
        f"A (warm reduction) median: {st.median(a):.3f}ms  <- true reduction-only cost"
    )
    print(
        f"B (cold, first-eval) median: {st.median(b):.3f}ms  <- what the probe actually measured"
    )
    print(
        f"C (pre-fenced reduction) median: {st.median(c):.3f}ms  <- reduction cost with a fence just before it"
    )
    print(
        f"D (chain eval alone, no reduction) median: {st.median(d):.3f}ms  <- backlog-flush cost alone"
    )
    print()
    if st.median(b) > 5 * max(st.median(a), 1e-6):
        print(
            "VERDICT: B >> A. The probe's local_absmax fence is dominated by "
            "flushing an unevaluated upstream graph (backlog), NOT by the "
            "abs-max reduction itself. The ~400-420ms figure is largely an "
            "ARTIFACT of where the first mx.eval() fence lands in the lazy "
            "graph, not a real per-call fixed cost of local_absmax."
        )
    else:
        print(
            "VERDICT: B ~ A. The reduction's own eval cost is comparable to "
            "the cold/backlogged case -- the ~400-420ms figure looks like a "
            "REAL cost, not primarily a backlog-flush artifact."
        )
