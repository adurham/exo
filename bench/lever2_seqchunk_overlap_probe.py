#!/Users/adam.durham/repos/exo/.venv/bin/python3
"""LEVER 2 probe: can moe.all_sum comm overlap GPU compute via SEQUENCE-CHUNK
pipelining?

Run:  .venv/bin/mlx.launch -n 2 --backend ring bench/lever2_seqchunk_overlap_probe.py

Structural question this answers (NOT an absolute-perf number -- ring/loopback
on one laptop GPU, both ranks share the GPU):

  Does MLX schedule an in-flight distributed collective (which lands on the
  group's pinned CPU stream) CONCURRENTLY with GPU compute belonging to a
  DIFFERENT, independent sequence chunk?

Arms:
  compute_only : GPU block on chunk, no collective
  comm_only    : all_sum on the chunk-sized payload, no block compute
  serial       : for each chunk: compute -> all_sum -> eval   (today's shape)
  pipelined    : issue all_sum(chunk i) async, then compute chunk i+1 on GPU
                 before waiting.  Overlap iff pipelined < serial.

Ideal overlap => pipelined ~= max(compute, comm) * N
No overlap    => pipelined ~= serial ~= (compute + comm) * N
"""

import os
import time

import mlx.core as mx
import mlx.nn as nn

_e = os.environ.get
HID = int(_e("P_HID", "7168"))    # DSv4 hidden dim
L = int(_e("P_L", "2048"))        # tokens per sequence chunk
NCHUNK = int(_e("P_NCHUNK", "4")) # chunks pipelined
FFN = int(_e("P_FFN", "2048"))    # fake expert width -> real GPU-bound matmul
REPS = int(_e("P_REPS", "1"))     # inner matmul reps: tunes compute cost
ITERS = int(_e("P_ITERS", "10"))

world = mx.distributed.init()
rank, size = world.rank(), world.size()


def log(*a):
    if rank == 0:
        print(*a, flush=True)


mx.random.seed(0)
W1 = mx.random.normal((HID, FFN)).astype(mx.bfloat16)
W2 = mx.random.normal((FFN, HID)).astype(mx.bfloat16)
XS = [mx.random.normal((1, L, HID)).astype(mx.bfloat16) for _ in range(NCHUNK)]
mx.eval(W1, W2, *XS)


def block(x):
    """Stand-in for the MoE expert compute that PRECEDES moe.all_sum."""
    y = x
    for _ in range(REPS):
        y = nn.gelu(y @ W1) @ W2
    return y


def asum(y):
    return mx.distributed.all_sum(y, group=world)


def timeit(fn, iters=ITERS):
    fn()                      # warm
    mx.synchronize()
    world.barrier() if hasattr(world, "barrier") else mx.eval(asum(mx.zeros((1,))))
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    mx.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


# ---- arms -------------------------------------------------------------
def arm_compute_only():
    outs = [block(x) for x in XS]
    mx.eval(outs)


def arm_comm_only():
    outs = [asum(x) for x in XS]
    mx.eval(outs)


def arm_serial():
    for x in XS:
        y = asum(block(x))
        mx.eval(y)            # <-- exactly what deepseek_v4.py does per layer


def arm_pipelined():
    """Sequence-chunk pipelining: chunk i's all_sum is in flight (CPU/comm
    stream) while chunk i+1's expert compute is dispatched to the GPU."""
    pend = []
    for x in XS:
        s = asum(block(x))
        mx.async_eval(s)      # kick off; do NOT block
        pend.append(s)
    mx.eval(pend)


def arm_lazy_noeval():
    """CONTROL: same graph as `serial` but WITHOUT the per-chunk mx.eval and
    WITHOUT async_eval. Isolates 'gain from dropping the fence' from
    'gain from genuine comm/compute overlap'. If lazy_noeval ~= pipelined,
    the pipelining added nothing beyond removing the fence."""
    outs = [asum(block(x)) for x in XS]
    mx.eval(outs)


def arm_pipelined_deep():
    """Stronger pipelining: explicitly async_eval chunk i's all_sum, THEN
    build+dispatch chunk i+1's compute, only forcing chunk i at the end."""
    pend = None
    outs = []
    for x in XS:
        s = asum(block(x))
        mx.async_eval(s)
        if pend is not None:
            mx.eval(pend)
        pend = s
        outs.append(s)
    mx.eval(outs)


# --- correctness: pipelined must be bit-identical to serial ---
_ref = [asum(block(x)) for x in XS]
mx.eval(_ref)
_p = []
for _x in XS:
    _s = asum(block(_x))
    mx.async_eval(_s)
    _p.append(_s)
mx.eval(_p)
_ok = all(bool(mx.all(a == b)) for a, b in zip(_ref, _p))
if rank == 0:
    print(f"[correctness] pipelined bit-identical to serial: {_ok}", flush=True)

# INTERLEAVED A/B: alternate serial and pipelined within one loop so
# thermal/scheduler drift hits both arms equally. Non-interleaved
# sequential-arm timing showed 0.87x-1.27x run-to-run swings.
ARMS = [
    ("compute_only", arm_compute_only),
    ("comm_only", arm_comm_only),
    ("serial", arm_serial),
    ("lazy_noeval", arm_lazy_noeval),
    ("pipelined", arm_pipelined),
    ("pipelined_deep", arm_pipelined_deep),
]
import statistics
samples = {n: [] for n, _ in ARMS}
for _n, _f in ARMS:      # warm all
    _f()
mx.synchronize()
for _ in range(ITERS):
    for _n, _f in ARMS:
        mx.synchronize()
        _t = time.perf_counter()
        _f()
        mx.synchronize()
        samples[_n].append((time.perf_counter() - _t) * 1000.0)
results = {n: statistics.median(v) for n, v in samples.items()}
if rank == 0:
    print("\n[interleaved medians +- IQR, n=%d]" % ITERS)
    for n, v in samples.items():
        q = statistics.quantiles(v, n=4)
        print(f"  {n:16s} med={statistics.median(v):8.2f}  p25={q[0]:8.2f}  p75={q[2]:8.2f}")
    # paired per-iteration speedup: immune to global drift
    pair = [s / p for s, p in zip(samples["serial"], samples["pipelined"])]
    print(f"  PAIRED serial/pipelined speedup: med={statistics.median(pair):.3f} "
          f"min={min(pair):.3f} max={max(pair):.3f}")
    pair2 = [s / p for s, p in zip(samples["serial"], samples["pipelined_deep"])]
    print(f"  PAIRED serial/pipelined_deep   : med={statistics.median(pair2):.3f}")

log(f"\n=== LEVER 2 seq-chunk overlap probe (rank {rank}/{size}) ===")
log(f"HID={HID} L={L} NCHUNK={NCHUNK} FFN={FFN} REPS={REPS} "
    f"payload/chunk={XS[0].nbytes/1e6:.1f}MB backend={os.environ.get('MLX_DIST_BACKEND','?')}")
for k, v in results.items():
    log(f"  {k:14s} {v:9.2f} ms")
c, m = results["compute_only"], results["comm_only"]
log(f"\n  sum(compute+comm)   = {c + m:9.2f} ms   (no-overlap floor)")
log(f"  max(compute,comm)   = {max(c, m):9.2f} ms   (perfect-overlap floor)")
log(f"  serial              = {results['serial']:9.2f} ms")
log(f"  pipelined           = {results['pipelined']:9.2f} ms")
denom = (c + m) - max(c, m)
if denom > 0:
    frac = ((c + m) - results["pipelined"]) / denom
    log(f"  => overlap achieved  = {100*frac:6.1f}% of theoretical max")
log(f"  => pipelined vs serial speedup = {results['serial']/results['pipelined']:.3f}x")
