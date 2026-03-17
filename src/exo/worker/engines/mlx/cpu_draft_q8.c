/**
 * INT8 quantized CPU draft forward pass.
 * Weights stored as uint8 with per-group scale+bias.
 * Dequantizes on the fly during vector-matrix multiply.
 *
 * Build:
 *   clang -O3 -shared -DACCELERATE_NEW_LAPACK -o cpu_draft_q8.dylib \
 *     cpu_draft_q8.c -framework Accelerate -arch arm64
 */

#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <arm_neon.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ── Quantized vector-matrix multiply ────────────────────────── */
/* out[j] = sum_i( input[i] * dequant(weight[j][i]) )           */
/* weight is packed: 4 uint8 values per uint32                   */
/* scales/biases are per-group (group_size elements share one)   */

/* Dequantize a quantized weight matrix into a float32 scratch buffer,
 * then use cblas_sgemv for the actual matmul. This reads 4× less from
 * DRAM (uint8 vs float32) and the scratch buffer stays in L2 cache. */
static float* _scratch = NULL;
static int _scratch_size = 0;

static float* get_scratch(int n) {
    if (n > _scratch_size) {
        free(_scratch);
        _scratch = (float*)malloc(n * sizeof(float));
        _scratch_size = n;
    }
    return _scratch;
}

static void dequant_to_scratch(
    float* dst,
    const uint32_t* src,   /* packed uint8 */
    const float* scales,
    const float* biases,
    int out_dim, int in_dim, int group_size
) {
    int pack = 4;
    int in_packed = in_dim / pack;
    int n_groups = in_dim / group_size;

    for (int j = 0; j < out_dim; j++) {
        const uint32_t* wrow = src + j * in_packed;
        float* drow = dst + j * in_dim;
        int idx = 0;

        for (int g = 0; g < n_groups; g++) {
            float s = scales[j * n_groups + g];
            float b = biases ? biases[j * n_groups + g] : 0.0f;

            for (int p = 0; p < group_size / pack; p++) {
                uint32_t packed = wrow[idx++];
                for (int k = 0; k < pack; k++) {
                    uint8_t q = (packed >> (8 * k)) & 0xFF;
                    *drow++ = s * (float)q + b;
                }
            }
        }
    }
}

static void qvm_8bit(
    float* out,
    const float* input,
    const uint32_t* weight,
    const float* scales,
    const float* biases,
    int out_dim,
    int in_dim,
    int group_size
) {
    /* Dequantize to scratch buffer, then use optimized BLAS */
    float* scratch = get_scratch(out_dim * in_dim);
    dequant_to_scratch(scratch, weight, scales, biases, out_dim, in_dim, group_size);

    /* y = A * x : A is (out_dim × in_dim), x is (in_dim,), y is (out_dim,) */
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                out_dim, in_dim, 1.0f, scratch, in_dim, input, 1, 0.0f, out, 1);
}

/* ── Same helpers as cpu_draft.c ─────────────────────────────── */

static void rms_norm(float* out, const float* x, const float* w, int n, float eps) {
    float sum = 0;
    for (int i = 0; i < n; i++) sum += x[i] * x[i];
    float scale = 1.0f / sqrtf(sum / n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * scale * w[i];
}

static void silu_mul(float* out, const float* gate, const float* up, int n) {
    for (int i = 0; i < n; i++) {
        float g = gate[i];
        out[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}

static void rope_inplace(float* x, int n_heads, int head_dim, int offset, float theta) {
    int half = head_dim / 2;
    for (int h = 0; h < n_heads; h++) {
        float* xh = x + h * head_dim;
        for (int d = 0; d < half; d++) {
            float freq = 1.0f / powf(theta, (float)(2 * d) / head_dim);
            float angle = offset * freq;
            float c = cosf(angle), s = sinf(angle);
            float x0 = xh[d], x1 = xh[d + half];
            xh[d]        = x0 * c - x1 * s;
            xh[d + half]  = x0 * s + x1 * c;
        }
    }
}

static void softmax(float* x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float sum = 0;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

/* ── Layer weights (quantized) ───────────────────────────────── */
typedef struct {
    const float* in_norm;     /* (hidden,) f32 */
    const float* post_norm;   /* (hidden,) f32 */
    /* Quantized projections: weight is uint32-packed, scales/biases are f32 */
    const uint32_t* q_w;  const float* q_s;  const float* q_b;
    const uint32_t* k_w;  const float* k_s;  const float* k_b;
    const uint32_t* v_w;  const float* v_s;  const float* v_b;
    const uint32_t* o_w;  const float* o_s;  const float* o_b;
    const uint32_t* gate_w; const float* gate_s; const float* gate_b;
    const uint32_t* up_w;   const float* up_s;   const float* up_b;
    const uint32_t* down_w; const float* down_s; const float* down_b;
    const float* q_norm;  /* (hd,) or NULL */
    const float* k_norm;  /* (hd,) or NULL */
    int n_heads, n_kv, head_dim, hidden, inter, group_size;
    float scale;
} Q8LayerWeights;

typedef struct {
    float* k;
    float* v;
    int seq_len;
    int max_seq;
} KVCache;

/* ── Forward pass ────────────────────────────────────────────── */

void cpu_draft_q8_forward(
    int token_id,
    const float* embed,       /* (vocab, hidden) f32 dequantized */
    const Q8LayerWeights* layers,
    int n_layers,
    const float* final_norm,
    const uint32_t* lm_head_w, const float* lm_head_s, const float* lm_head_b,
    int lm_head_gs,
    int vocab_size,
    int hidden,
    KVCache* caches,
    int offset,
    float rope_theta,
    float* logits_out
) {
    float* h     = (float*)malloc(hidden * sizeof(float));
    float* norm  = (float*)malloc(hidden * sizeof(float));
    int max_proj = hidden * 4; /* largest projection output */
    float* buf1  = (float*)malloc(max_proj * sizeof(float));
    float* buf2  = (float*)malloc(max_proj * sizeof(float));
    float* buf3  = (float*)malloc(max_proj * sizeof(float));
    float* proj  = (float*)malloc(hidden * sizeof(float));

    memcpy(h, embed + token_id * hidden, hidden * sizeof(float));

    for (int l = 0; l < n_layers; l++) {
        const Q8LayerWeights* lw = &layers[l];
        KVCache* kv = &caches[l];
        int nh = lw->n_heads, nkv = lw->n_kv, hd = lw->head_dim;
        int gqa = nh / nkv;
        int q_dim = nh * hd, kv_dim = nkv * hd;
        int gs = lw->group_size;

        rms_norm(norm, h, lw->in_norm, hidden, 1e-6f);

        /* Quantized Q/K/V projections */
        qvm_8bit(buf1, norm, lw->q_w, lw->q_s, lw->q_b, q_dim, hidden, gs);
        qvm_8bit(buf2, norm, lw->k_w, lw->k_s, lw->k_b, kv_dim, hidden, gs);
        qvm_8bit(buf3, norm, lw->v_w, lw->v_s, lw->v_b, kv_dim, hidden, gs);

        if (lw->q_norm) {
            for (int i = 0; i < nh; i++)
                rms_norm(buf1 + i*hd, buf1 + i*hd, lw->q_norm, hd, 1e-6f);
            for (int i = 0; i < nkv; i++)
                rms_norm(buf2 + i*hd, buf2 + i*hd, lw->k_norm, hd, 1e-6f);
        }

        rope_inplace(buf1, nh, hd, offset, rope_theta);
        rope_inplace(buf2, nkv, hd, offset, rope_theta);

        /* KV cache append */
        int seq = kv->seq_len;
        for (int i = 0; i < nkv; i++) {
            memcpy(kv->k + i*kv->max_seq*hd + seq*hd, buf2 + i*hd, hd*sizeof(float));
            memcpy(kv->v + i*kv->max_seq*hd + seq*hd, buf3 + i*hd, hd*sizeof(float));
        }
        kv->seq_len = seq + 1;
        int cur_seq = kv->seq_len;

        /* GQA attention (same as float32 version — KV cache is always f32) */
        float* attn = buf3; /* reuse buf3 */
        memset(attn, 0, q_dim * sizeof(float));
        float* scores = (float*)malloc(cur_seq * sizeof(float));

        for (int kh = 0; kh < nkv; kh++) {
            float* kcache = kv->k + kh*kv->max_seq*hd;
            float* vcache = kv->v + kh*kv->max_seq*hd;
            for (int g = 0; g < gqa; g++) {
                int qh = kh * gqa + g;
                cblas_sgemv(CblasRowMajor, CblasNoTrans,
                            cur_seq, hd, lw->scale,
                            kcache, hd, buf1 + qh*hd, 1, 0.0f, scores, 1);
                softmax(scores, cur_seq);
                cblas_sgemv(CblasRowMajor, CblasTrans,
                            cur_seq, hd, 1.0f,
                            vcache, hd, scores, 1, 0.0f, attn + qh*hd, 1);
            }
        }
        free(scores);

        /* Output projection (quantized) */
        qvm_8bit(proj, attn, lw->o_w, lw->o_s, lw->o_b, hidden, q_dim, gs);
        for (int i = 0; i < hidden; i++) h[i] += proj[i];

        /* MLP (quantized) */
        rms_norm(norm, h, lw->post_norm, hidden, 1e-6f);
        qvm_8bit(buf1, norm, lw->gate_w, lw->gate_s, lw->gate_b, lw->inter, hidden, gs);
        qvm_8bit(buf2, norm, lw->up_w, lw->up_s, lw->up_b, lw->inter, hidden, gs);
        silu_mul(buf1, buf1, buf2, lw->inter);
        qvm_8bit(proj, buf1, lw->down_w, lw->down_s, lw->down_b, hidden, lw->inter, gs);
        for (int i = 0; i < hidden; i++) h[i] += proj[i];
    }

    rms_norm(norm, h, final_norm, hidden, 1e-6f);

    /* LM head (quantized) */
    qvm_8bit(logits_out, norm, lm_head_w, lm_head_s, lm_head_b,
             vocab_size, hidden, lm_head_gs);

    free(h); free(norm); free(buf1); free(buf2); free(buf3); free(proj);
}
