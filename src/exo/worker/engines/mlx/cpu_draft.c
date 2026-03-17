/**
 * CPU-only draft model forward pass for speculative decoding.
 * Uses Accelerate BLAS (cblas_sgemm) for all matmuls.
 * Runs entirely on CPU — zero GPU contention.
 *
 * Build:
 *   clang -O3 -shared -DACCELERATE_NEW_LAPACK -o cpu_draft.dylib \
 *     cpu_draft.c -framework Accelerate -arch arm64
 */

#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ── Helpers ─────────────────────────────────────────────────── */

static void rms_norm(float* out, const float* x, const float* w, int n, float eps) {
    float sum = 0;
    for (int i = 0; i < n; i++) sum += x[i] * x[i];
    float scale = 1.0f / sqrtf(sum / n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * scale * w[i];
}

static void silu_mul(float* out, const float* gate, const float* up, int n) {
    for (int i = 0; i < n; i++) {
        float g = gate[i];
        float s = g / (1.0f + expf(-g));  /* SiLU */
        out[i] = s * up[i];
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

/* ── Layer weights (passed as flat pointers) ─────────────────── */
typedef struct {
    const float* in_norm;    /* (hidden,) */
    const float* post_norm;  /* (hidden,) */
    const float* q;          /* (n_heads*hd, hidden) */
    const float* k;          /* (n_kv*hd, hidden) */
    const float* v;          /* (n_kv*hd, hidden) */
    const float* o;          /* (hidden, n_heads*hd) */
    const float* gate;       /* (inter, hidden) */
    const float* up;         /* (inter, hidden) */
    const float* down;       /* (hidden, inter) */
    const float* q_norm;     /* (hd,) or NULL */
    const float* k_norm;     /* (hd,) or NULL */
    int n_heads, n_kv, head_dim, hidden, inter;
    float scale;
} LayerWeights;

typedef struct {
    float* k;   /* (n_kv, max_seq, hd) */
    float* v;   /* (n_kv, max_seq, hd) */
    int seq_len;
    int max_seq;
} KVCache;

/* ── Single-token forward pass ───────────────────────────────── */

void cpu_draft_forward(
    int token_id,
    const float* embed,       /* (vocab, hidden) */
    const LayerWeights* layers,
    int n_layers,
    const float* final_norm,  /* (hidden,) */
    const float* lm_head,     /* (vocab, hidden) */
    int vocab_size,
    int hidden,
    KVCache* caches,
    int offset,               /* position in sequence */
    float rope_theta,
    float* logits_out         /* (vocab,) output */
) {
    /* Scratch buffers */
    float* h     = (float*)malloc(hidden * sizeof(float));
    float* norm  = (float*)malloc(hidden * sizeof(float));
    float* qbuf  = (float*)malloc(hidden * 2 * sizeof(float)); /* max(n_heads*hd) */
    float* kbuf  = (float*)malloc(hidden * sizeof(float));
    float* vbuf  = (float*)malloc(hidden * sizeof(float));
    float* attn  = (float*)malloc(hidden * 2 * sizeof(float));
    float* mlp1  = (float*)malloc(hidden * 4 * sizeof(float)); /* inter can be up to 4x */
    float* mlp2  = (float*)malloc(hidden * 4 * sizeof(float));
    float* proj  = (float*)malloc(hidden * sizeof(float));

    /* Embedding */
    memcpy(h, embed + token_id * hidden, hidden * sizeof(float));

    for (int l = 0; l < n_layers; l++) {
        const LayerWeights* lw = &layers[l];
        KVCache* kv = &caches[l];
        int nh = lw->n_heads, nkv = lw->n_kv, hd = lw->head_dim;
        int gqa = nh / nkv;
        int q_dim = nh * hd, kv_dim = nkv * hd;

        /* Input norm */
        rms_norm(norm, h, lw->in_norm, hidden, 1e-6f);

        /* Q/K/V projections: (1, hidden) × (hidden, out_dim)^T = (1, out_dim)
         * Weight layout: (out_dim, hidden), so we do x @ W^T via
         * cblas_sgemv with W as (out_dim × hidden) row-major, x as vector. */
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    q_dim, hidden, 1.0f, lw->q, hidden, norm, 1, 0.0f, qbuf, 1);
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    kv_dim, hidden, 1.0f, lw->k, hidden, norm, 1, 0.0f, kbuf, 1);
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    kv_dim, hidden, 1.0f, lw->v, hidden, norm, 1, 0.0f, vbuf, 1);

        /* Q/K norms */
        if (lw->q_norm) {
            for (int i = 0; i < nh; i++)
                rms_norm(qbuf + i*hd, qbuf + i*hd, lw->q_norm, hd, 1e-6f);
            for (int i = 0; i < nkv; i++)
                rms_norm(kbuf + i*hd, kbuf + i*hd, lw->k_norm, hd, 1e-6f);
        }

        /* RoPE */
        rope_inplace(qbuf, nh, hd, offset, rope_theta);
        rope_inplace(kbuf, nkv, hd, offset, rope_theta);

        /* KV cache append */
        int seq = kv->seq_len;
        for (int i = 0; i < nkv; i++) {
            memcpy(kv->k + i * kv->max_seq * hd + seq * hd,
                   kbuf + i * hd, hd * sizeof(float));
            memcpy(kv->v + i * kv->max_seq * hd + seq * hd,
                   vbuf + i * hd, hd * sizeof(float));
        }
        kv->seq_len = seq + 1;
        int cur_seq = kv->seq_len;

        /* GQA attention */
        memset(attn, 0, q_dim * sizeof(float));
        float* scores = (float*)malloc(cur_seq * sizeof(float));

        for (int kh = 0; kh < nkv; kh++) {
            float* kcache = kv->k + kh * kv->max_seq * hd;  /* (cur_seq, hd) */
            float* vcache = kv->v + kh * kv->max_seq * hd;

            for (int g = 0; g < gqa; g++) {
                int qh = kh * gqa + g;
                float* qhead = qbuf + qh * hd;  /* (hd,) */

                /* scores = Q @ K^T: (1,hd) × (hd,seq) */
                cblas_sgemv(CblasRowMajor, CblasNoTrans,
                            cur_seq, hd, lw->scale,
                            kcache, hd, qhead, 1, 0.0f, scores, 1);

                softmax(scores, cur_seq);

                /* out = scores @ V: (1,seq) × (seq,hd) */
                cblas_sgemv(CblasRowMajor, CblasTrans,
                            cur_seq, hd, 1.0f,
                            vcache, hd, scores, 1, 0.0f, attn + qh * hd, 1);
            }
        }
        free(scores);

        /* Output projection: (1, q_dim) × (q_dim, hidden)^T */
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    hidden, q_dim, 1.0f, lw->o, q_dim, attn, 1, 0.0f, proj, 1);

        /* Residual */
        for (int i = 0; i < hidden; i++) h[i] += proj[i];

        /* Post-attention norm */
        rms_norm(norm, h, lw->post_norm, hidden, 1e-6f);

        /* MLP: gate + up + silu_mul + down */
        int inter = lw->inter;
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    inter, hidden, 1.0f, lw->gate, hidden, norm, 1, 0.0f, mlp1, 1);
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    inter, hidden, 1.0f, lw->up, hidden, norm, 1, 0.0f, mlp2, 1);

        silu_mul(mlp1, mlp1, mlp2, inter);

        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    hidden, inter, 1.0f, lw->down, inter, mlp1, 1, 0.0f, proj, 1);

        /* Residual */
        for (int i = 0; i < hidden; i++) h[i] += proj[i];
    }

    /* Final norm */
    rms_norm(norm, h, final_norm, hidden, 1e-6f);

    /* LM head: (1, hidden) × (hidden, vocab)^T */
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                vocab_size, hidden, 1.0f, lm_head, hidden, norm, 1, 0.0f, logits_out, 1);

    free(h); free(norm); free(qbuf); free(kbuf); free(vbuf);
    free(attn); free(mlp1); free(mlp2); free(proj);
}
