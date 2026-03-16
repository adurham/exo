/**
 * CPU-assisted attention for Apple Silicon unified memory.
 *
 * Uses Accelerate BLAS (cblas_sgemm) for batched matmul — processes all
 * Q heads in a GQA group in a single BLAS call via AMX.
 *
 * Build:
 *   clang -O3 -shared -DACCELERATE_NEW_LAPACK -o cpu_attention.dylib \
 *     cpu_attention.c -framework Accelerate -arch arm64
 */

#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

/**
 * Compute partial scaled dot-product attention on CPU.
 *
 * Uses batched BLAS for all Q heads per KV head simultaneously.
 *
 * q:        (num_q_heads, D) float32
 * k:        (num_kv_heads, N, D) float32
 * v:        (num_kv_heads, N, D) float32
 * out:      (num_q_heads, D) float32
 * max_out:  (num_q_heads,) float32
 * sum_out:  (num_q_heads,) float32
 */
void cpu_attention_f32(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    float* max_out,
    float* sum_out,
    float scale,
    int num_q_heads,
    int num_kv_heads,
    int N,
    int D) {

    int gqa = num_q_heads / num_kv_heads;

    // Temp buffers
    float* scores = (float*)malloc(gqa * N * sizeof(float));
    float* exp_buf = (float*)malloc(gqa * N * sizeof(float));

    for (int kv_h = 0; kv_h < num_kv_heads; kv_h++) {
        const float* k_head = k + kv_h * N * D;  // (N, D)
        const float* v_head = v + kv_h * N * D;  // (N, D)
        const float* q_group = q + kv_h * gqa * D;  // (gqa, D)
        float* out_group = out + kv_h * gqa * D;  // (gqa, D)
        float* max_group = max_out + kv_h * gqa;
        float* sum_group = sum_out + kv_h * gqa;

        // Scores = Q @ K^T: (gqa, D) × (D, N) → (gqa, N)
        // cblas_sgemm: C = alpha * A * B + beta * C
        // A = Q (gqa × D, row-major), B = K^T (D × N, K is N × D so transpose)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    gqa, N, D,
                    scale, q_group, D,
                    k_head, D,
                    0.0f, scores, N);

        // Per-Q-head softmax + weighted V
        for (int qi = 0; qi < gqa; qi++) {
            float* s = scores + qi * N;

            // Find max
            float max_s;
            vDSP_maxv(s, 1, &max_s, N);

            // exp(scores - max)
            float neg_max = -max_s;
            vDSP_vsadd(s, 1, &neg_max, exp_buf + qi * N, 1, N);
            int n_int = N;
            vvexpf(exp_buf + qi * N, exp_buf + qi * N, &n_int);

            // Sum of exp
            float sum_e;
            vDSP_sve(exp_buf + qi * N, 1, &sum_e, N);

            max_group[qi] = max_s;
            sum_group[qi] = sum_e;
        }

        // Output = exp_scores @ V: (gqa, N) × (N, D) → (gqa, D)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    gqa, D, N,
                    1.0f, exp_buf, N,
                    v_head, D,
                    0.0f, out_group, D);
    }

    free(scores);
    free(exp_buf);
}
