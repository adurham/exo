#!/usr/bin/env python3
"""gather_qmm achieved-bandwidth microbench at DSv4 prefill production shape.

Measures wall time of the sorted gather_qmm path exactly as SwitchGLU uses it
during a 256-token prefill chunk: 1536 (token,expert) pairs over 256 experts,
mxfp4 gate/up + down projections, sharded dims (intermediate 2048 -> 1024
per node under tensor sharding of moe_intermediate across 2 nodes... actually
gate/up are all-to-sharded so N=1024/node; down is sharded-to-all K=1024).

Reports ms/layer-equivalent and achieved GB/s vs the ~546 GB/s M4 Max peak.
Run ON A STUDIO: python3 /tmp/qmm_bench.py
"""
import time

import mlx.core as mx

B_PAIRS = 1536          # 256 tokens x 6 experts
N_EXPERTS = 256
HIDDEN = 4096
INTER = 1024            # per-node shard of moe_intermediate 2048
GROUP = 32
BITS = 4
MODE = "mxfp4"
N_ITERS = 30
N_WARM = 5


def make_qweights(n_exp, out_d, in_d):
    w = mx.random.normal((n_exp, out_d, in_d), dtype=mx.float32) * 0.02
    return mx.quantize(w, group_size=GROUP, bits=BITS, mode=MODE)


def main():
    mx.random.seed(0)
    # Sorted expert indices as _gather_sort produces (contiguous same-expert runs)
    reps = B_PAIRS // N_EXPERTS
    idx_sorted = mx.sort(mx.concatenate([
        mx.arange(N_EXPERTS, dtype=mx.uint32)] * reps))
    x = mx.random.normal((B_PAIRS, 1, HIDDEN), dtype=mx.bfloat16)

    gate_w = make_qweights(N_EXPERTS, INTER, HIDDEN)
    up_w = make_qweights(N_EXPERTS, INTER, HIDDEN)
    down_w = make_qweights(N_EXPERTS, HIDDEN, INTER)

    def one_layer():
        g = mx.gather_qmm(x, *gate_w, rhs_indices=idx_sorted, transpose=True,
                          group_size=GROUP, bits=BITS, mode=MODE,
                          sorted_indices=True)
        u = mx.gather_qmm(x, *up_w, rhs_indices=idx_sorted, transpose=True,
                          group_size=GROUP, bits=BITS, mode=MODE,
                          sorted_indices=True)
        h = mx.minimum(g, 10.0) * mx.clip(u, -10.0, 10.0)  # stand-in act
        d = mx.gather_qmm(h, *down_w, rhs_indices=idx_sorted, transpose=True,
                          group_size=GROUP, bits=BITS, mode=MODE,
                          sorted_indices=True)
        return d

    for _ in range(N_WARM):
        mx.eval(one_layer())
    mx.synchronize()

    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        mx.eval(one_layer())
    mx.synchronize()
    dt = (time.perf_counter() - t0) / N_ITERS

    # Weight bytes actually streamed per layer (all experts touched):
    def qbytes(qw):
        return sum(a.nbytes for a in qw)
    wbytes = qbytes(gate_w) + qbytes(up_w) + qbytes(down_w)
    # Activation traffic (read x 2x, write inter 2x, read inter, write out)
    abytes = (2 * x.nbytes + 2 * B_PAIRS * INTER * 2 +
              B_PAIRS * INTER * 2 + B_PAIRS * HIDDEN * 2)
    gbs = (wbytes + abytes) / dt / 1e9
    print(f"pairs={B_PAIRS} experts={N_EXPERTS} inter={INTER} hidden={HIDDEN}")
    print(f"weights={wbytes/1e9:.2f}GB acts={abytes/1e9:.3f}GB")
    print(f"per-layer wall: {dt*1000:.2f} ms")
    print(f"achieved: {gbs:.0f} GB/s (M4 Max peak ~546)")
    print(f"bandwidth floor at peak: {(wbytes+abytes)/546e9*1000:.2f} ms")


if __name__ == "__main__":
    main()
