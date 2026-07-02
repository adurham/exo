#!/usr/bin/env python3
"""Prototype: M-batched mxfp4 GEMV — does one weight stream at GEMV bandwidth
amortize across M activation rows?

Hypothesis: qmv_fast hits ~450 GB/s at M=1; steel qmm collapses to ~140 GB/s
at M=6-16 (production MoE prefill regime). A GEMV-shaped kernel holding M x
rows in registers and accumulating M results per weight read should keep
GEMV-class bandwidth while doing M rows of work -> ~2.7x on the MoE layer.

This is a THROWAWAY validation kernel (dense, single expert weight matrix,
mxfp4 g32) — not production. If it sustains >300 GB/s at M=4-6 with correct
output, the C++ gather variant is justified.

Run ON A STUDIO: ~/scratch/tilesweep-venv/bin/python /tmp/qmv_mbatch_proto.py
"""
import time

import mlx.core as mx

N_OUT = 1024      # rows of w (output features per expert shard)
K_IN = 4096       # cols (input dim)
GROUP = 32
M_ROWS = 6        # activation rows sharing this weight stream

SRC = """
    // One threadgroup = 2 simdgroups x 4 output rows each = 8 output rows.
    // Each lane streams 2 packs of 8 fp4 values (16 values) per K-block.
    // M_ activation rows are held in registers: x_thread[M_][16].
    constexpr int packs_per_thread = 2;
    constexpr int num_simdgroups = 2;
    constexpr int results_per_simdgroup = 4;
    constexpr int pack_factor = 8;          // fp4 values per uint32
    constexpr int values_per_thread = pack_factor * packs_per_thread; // 16
    constexpr int block_size = values_per_thread * 32;  // per-simdgroup K step

    uint simd_gid = thread_position_in_threadgroup.x / 32;
    uint simd_lid = thread_position_in_threadgroup.x % 32;

    const int out_row = threadgroup_position_in_grid.y * (num_simdgroups * results_per_simdgroup)
                      + simd_gid * results_per_simdgroup;

    const int K = K_;
    const int K_w = K / pack_factor;        // uint32 per w row
    const int K_g = K / GROUP_;             // scale bytes per w row

    const device uint32_t* ws = w + out_row * K_w
                              + simd_lid * packs_per_thread;
    const device uint8_t* ss = scales + out_row * K_g
                             + (simd_lid * values_per_thread) / GROUP_;

    float x_thread[M_][values_per_thread];
    float result[M_][results_per_simdgroup];
    for (int m = 0; m < M_; m++)
      for (int r = 0; r < results_per_simdgroup; r++)
        result[m][r] = 0.0f;

    // fp4 e2m1 LUT
    const float lut[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                           -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

    const device T* xb = x + simd_lid * values_per_thread;

    for (int k = 0; k < K; k += block_size) {
      // load M activation slices for this K-block
      for (int m = 0; m < M_; m++) {
        const device T* xr = xb + m * K + k;
        for (int v = 0; v < values_per_thread; v++)
          x_thread[m][v] = float(xr[v]);
      }
      // stream weights once, accumulate into all M rows
      for (int r = 0; r < results_per_simdgroup; r++) {
        const device uint32_t* wr = ws + r * K_w;
        const device uint8_t* sr = ss + r * K_g;
        // e8m0 scale: 2^(s-127)
        float s = exp2(float(sr[0]) - 127.0f);
        for (int p = 0; p < packs_per_thread; p++) {
          uint32_t pack = wr[p];
          for (int v = 0; v < pack_factor; v++) {
            float wv = lut[(pack >> (4 * v)) & 0xF] * s;
            int vi = p * pack_factor + v;
            for (int m = 0; m < M_; m++)
              result[m][r] += wv * x_thread[m][vi];
          }
        }
      }
      ws += block_size / pack_factor;
      ss += block_size / GROUP_;
      xb += block_size;
    }

    for (int m = 0; m < M_; m++)
      for (int r = 0; r < results_per_simdgroup; r++) {
        float tot = simd_sum(result[m][r]);
        if (simd_lid == 0)
          y[m * N_ + out_row + r] = T(tot);
      }
"""


def main():
    mx.random.seed(0)
    wf = mx.random.normal((N_OUT, K_IN), dtype=mx.float32) * 0.02
    w_q, w_s = mx.quantize(wf, group_size=GROUP, bits=4, mode="mxfp4")
    x = mx.random.normal((M_ROWS, K_IN), dtype=mx.bfloat16)

    # Reference
    ref = x @ mx.dequantize(w_q, w_s, group_size=GROUP, bits=4, mode="mxfp4").T
    mx.eval(ref)

    kernel = mx.fast.metal_kernel(
        name=f"qmv_mbatch_m{M_ROWS}",
        input_names=["w", "scales", "x"],
        output_names=["y"],
        source=SRC,
        header=f"""
        constant int M_ = {M_ROWS};
        constant int N_ = {N_OUT};
        constant int K_ = {K_IN};
        constant int GROUP_ = {GROUP};
        """,
        ensure_row_contiguous=True,
    )

    def run():
        return kernel(
            inputs=[w_q, w_s, x],
            template=[("T", mx.bfloat16)],
            grid=(1, N_OUT // 8, 1),
            threadgroup=(64, 1, 1),
            output_shapes=[(M_ROWS, N_OUT)],
            output_dtypes=[mx.bfloat16],
        )[0]

    out = run()
    mx.eval(out)
    diff = mx.max(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32))).item()
    rel = diff / mx.max(mx.abs(ref.astype(mx.float32))).item()
    print(f"max abs diff vs dequant-ref: {diff:.4f} (rel {rel:.2e})")

    for _ in range(10):
        mx.eval(run())
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        mx.eval(run())
    mx.synchronize()
    dt = (time.perf_counter() - t0) / 50

    wbytes = w_q.nbytes + w_s.nbytes
    print(f"M={M_ROWS}: {dt*1e6:.0f} us  weight-stream {wbytes/dt/1e9:.0f} GB/s")
    print(f"(qmv_fast M=1 reference on this shape: ~450 GB/s; steel qmm M=6: ~150)")


if __name__ == "__main__":
    main()
