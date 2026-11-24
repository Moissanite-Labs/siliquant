/*
 * Native Intel Core i9-11900H AVX-512 (vpermb + vpopcntdq + vpdpbusd)
 * Full 24-Layer Qwen2.5-0.5B A_16 CPU-Only Inference Engine.
 *
 * Consumes the EXACT same 4.500-bpw A_16 [16,11,4] nibble-packed weights
 * (16 B biased u4 nibbles + 2 B fp16 scale per 32-weight group = 18 B / 32 w)
 * and 122 preserved BF16 tensors (norms, QKV biases, tied embed_tokens/lm_head)
 * as qwen25_sm86_a16_engine.cu, enabling exact apples-to-apples GPU vs. CPU
 * latency-energy (mJ/tok) Pareto measurement on live Qwen/Qwen2.5-0.5B.
 */

#include <immintrin.h>
#include <omp.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define QWEN_LAYERS 24
#define QWEN_H 896
#define QWEN_I 4864
#define QWEN_Q_HEADS 14
#define QWEN_KV_HEADS 2
#define QWEN_HEAD_DIM 64
#define QWEN_KV_DIM (QWEN_KV_HEADS * QWEN_HEAD_DIM)
#define QWEN_VOCAB 151936
#define QWEN_RMS_EPS 1.0e-6f
#define QWEN_ROPE_THETA 1000000.0f
#define MAX_SEQ_LEN 512

static const int PROJ_M[7] = {QWEN_H, QWEN_KV_DIM, QWEN_KV_DIM, QWEN_H, QWEN_I, QWEN_I, QWEN_H};
static const int PROJ_K[7] = {QWEN_H, QWEN_H,      QWEN_H,      QWEN_H, QWEN_H, QWEN_H, QWEN_I};

static inline float fp16_to_fp32(uint16_t h) {
    __m128i vh = _mm_cvtsi32_si128((int)h);
    __m128 vf = _mm_cvtph_ps(vh);
    return _mm_cvtss_f32(vf);
}

static inline float bf16_to_fp32(uint16_t b) {
    uint32_t u = ((uint32_t)b) << 16;
    float f;
    memcpy(&f, &u, sizeof(float));
    return f;
}

typedef struct {
    uint8_t*  w_nib[7];    /* [M, K/2] packed biased u4 nibbles */
    float*    w_scale[7];  /* [M, K/32] unpacked fp32 scales (from fp16) */
    float*    bias[3];     /* q, k, v biases */
    float*    in_norm;     /* [896] */
    float*    post_norm;   /* [896] */
} QwenLayerCPU;

typedef struct {
    QwenLayerCPU layers[QWEN_LAYERS];
    uint16_t* embed_bf16;   /* [151936, 896] */
    float*    final_norm;   /* [896] */
    float*    k_cache;      /* [24, MAX_SEQ_LEN, 128] */
    float*    v_cache;      /* [24, MAX_SEQ_LEN, 128] */

    float     h[QWEN_H];
    float     h_norm[QWEN_H];
    float     q[QWEN_H];
    float     k[QWEN_KV_DIM];
    float     v[QWEN_KV_DIM];
    float     attn_out[QWEN_H];
    float     gate[QWEN_I];
    float     up[QWEN_I];

    int8_t    x_q8[QWEN_I] __attribute__((aligned(64)));
    float     x_scale[QWEN_I / 32] __attribute__((aligned(64)));
    int32_t   x_gsum[QWEN_I / 32] __attribute__((aligned(64)));
    uint64_t  syn_errors;
    int       num_threads;
} QwenEngineCPU;

static void* alloc_aligned64(size_t bytes) {
    void* p = NULL;
    size_t sz = (bytes + 63u) & ~63u;
    if (sz == 0) sz = 64u;
    if (posix_memalign(&p, 64, sz) != 0) return NULL;
    memset(p, 0, sz);
    return p;
}

/* Pure AVX-512 RMSNorm + per-32-group symmetric int8 quantization */
static void rmsnorm_q8_cpu(
    const float* x_in, const float* norm_w,
    int8_t* x_q8, float* x_scale, int32_t* x_gsum, int H
) {
    __m512 vsq = _mm512_setzero_ps();
    for (int i = 0; i < H; i += 16) {
        __m512 vx = _mm512_loadu_ps(x_in + i);
        vsq = _mm512_fmadd_ps(vx, vx, vsq);
    }
    float inv_rms = 1.0f / sqrtf(_mm512_reduce_add_ps(vsq) / (float)H + QWEN_RMS_EPS);
    __m512 vinv = _mm512_set1_ps(inv_rms);

    int ng = H >> 5;
    for (int g = 0; g < ng; ++g) {
        int base = g << 5;
        __m512 v0 = _mm512_mul_ps(_mm512_mul_ps(_mm512_loadu_ps(x_in + base),      vinv), _mm512_loadu_ps(norm_w + base));
        __m512 v1 = _mm512_mul_ps(_mm512_mul_ps(_mm512_loadu_ps(x_in + base + 16), vinv), _mm512_loadu_ps(norm_w + base + 16));
        __m512 absm = _mm512_max_ps(_mm512_abs_ps(v0), _mm512_abs_ps(v1));
        float amax = _mm512_reduce_max_ps(absm);
        float sc = (amax > 1.0e-12f) ? (amax / 127.0f) : 1.0e-12f;
        float inv_sc = 1.0f / sc;
        __m512 vis = _mm512_set1_ps(inv_sc);
        __m512 vmin = _mm512_set1_ps(-127.0f);
        __m512 vmax = _mm512_set1_ps(127.0f);

        __m512i q0 = _mm512_cvtps_epi32(_mm512_min_ps(vmax, _mm512_max_ps(vmin, _mm512_mul_ps(v0, vis))));
        __m512i q1 = _mm512_cvtps_epi32(_mm512_min_ps(vmax, _mm512_max_ps(vmin, _mm512_mul_ps(v1, vis))));
        int32_t gsum = _mm512_reduce_add_epi32(_mm512_add_epi32(q0, q1));
        _mm_storeu_si128((__m128i*)(x_q8 + base),      _mm512_cvtsepi32_epi8(q0));
        _mm_storeu_si128((__m128i*)(x_q8 + base + 16), _mm512_cvtsepi32_epi8(q1));
        x_scale[g] = sc;
        x_gsum[g] = gsum;
    }
}

static void quantize_fp32_q8_cpu(const float* x_in, int8_t* x_q8, float* x_scale, int32_t* x_gsum, int N) {
    int ng = N >> 5;
    __m512 vmin = _mm512_set1_ps(-127.0f);
    __m512 vmax = _mm512_set1_ps(127.0f);
    for (int g = 0; g < ng; ++g) {
        int base = g << 5;
        __m512 v0 = _mm512_loadu_ps(x_in + base);
        __m512 v1 = _mm512_loadu_ps(x_in + base + 16);
        float amax = _mm512_reduce_max_ps(_mm512_max_ps(_mm512_abs_ps(v0), _mm512_abs_ps(v1)));
        float sc = (amax > 1.0e-12f) ? (amax / 127.0f) : 1.0e-12f;
        __m512 vis = _mm512_set1_ps(1.0f / sc);
        __m512i q0 = _mm512_cvtps_epi32(_mm512_min_ps(vmax, _mm512_max_ps(vmin, _mm512_mul_ps(v0, vis))));
        __m512i q1 = _mm512_cvtps_epi32(_mm512_min_ps(vmax, _mm512_max_ps(vmin, _mm512_mul_ps(v1, vis))));
        _mm_storeu_si128((__m128i*)(x_q8 + base),      _mm512_cvtsepi32_epi8(q0));
        _mm_storeu_si128((__m128i*)(x_q8 + base + 16), _mm512_cvtsepi32_epi8(q1));
        x_scale[g] = sc;
        x_gsum[g] = _mm512_reduce_add_epi32(_mm512_add_epi32(q0, q1));
    }
}

static void silu_mul_q8_cpu(
    const float* gate, const float* up,
    int8_t* out_q8, float* out_scale, int32_t* out_gsum, int N, int num_threads
) {
    int ng = N >> 5;
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int g = 0; g < ng; ++g) {
        float tmp[32] __attribute__((aligned(64)));
        int base = g << 5;
        for (int j = 0; j < 32; ++j) {
            float gv = gate[base + j];
            tmp[j] = (gv / (1.0f + expf(-gv))) * up[base + j];
        }
        __m512 v0 = _mm512_load_ps(tmp);
        __m512 v1 = _mm512_load_ps(tmp + 16);
        float amax = _mm512_reduce_max_ps(_mm512_max_ps(_mm512_abs_ps(v0), _mm512_abs_ps(v1)));
        float sc = (amax > 1.0e-12f) ? (amax / 127.0f) : 1.0e-12f;
        __m512 vis = _mm512_set1_ps(1.0f / sc);
        __m512 vmin = _mm512_set1_ps(-127.0f);
        __m512 vmax = _mm512_set1_ps(127.0f);
        __m512i q0 = _mm512_cvtps_epi32(_mm512_min_ps(vmax, _mm512_max_ps(vmin, _mm512_mul_ps(v0, vis))));
        __m512i q1 = _mm512_cvtps_epi32(_mm512_min_ps(vmax, _mm512_max_ps(vmin, _mm512_mul_ps(v1, vis))));
        _mm_storeu_si128((__m128i*)(out_q8 + base),      _mm512_cvtsepi32_epi8(q0));
        _mm_storeu_si128((__m128i*)(out_q8 + base + 16), _mm512_cvtsepi32_epi8(q1));
        out_scale[g] = sc;
        out_gsum[g] = _mm512_reduce_add_epi32(_mm512_add_epi32(q0, q1));
    }
}

/*
 * AVX-512 VBMI (vpermb) + VNNI (vpdpbusd) + VPOPCNTDQ (vpopcntq) A_16 GEMV:
 * Processes 2 groups (64 weights = 32 packed bytes) per ZMM iteration!
 *   - Hardware vpermb unpacks low/high nibbles using 64-byte in-register LUT.
 *   - Hardware vpdpbusd computes 64 u8*s8 MACs per instruction.
 *   - Optional live vpopcntq checks all 4 Extended Hamming [16,11,4] subblocks in parallel!
 */
static void a16_gemv_avx512_vnni(
    const uint8_t* w_nib,
    const float*   w_scale,
    const int8_t*  x_q8,
    const float*   x_scale,
    const int32_t* x_gsum,
    const float*   bias,
    float*         y_out,
    int M, int K,
    int add_residual,
    int live_verify,
    uint64_t* syn_err_acc,
    int num_threads
) {
    int ng = K >> 5;  /* K is 896 (28 groups) or 4864 (152 groups) -> ALWAYS even! */

    /* 64-byte identity nibble LUT for vpermb + permutation index to interleave even/odd bytes */
    static const uint8_t NIB_LUT[64] __attribute__((aligned(64))) = {
        0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
        0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
        0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
        0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
    };
    /* Interleave 32 lo bytes [0..31] and 32 hi bytes [32..63] -> [lo0,hi0,lo1,hi1,...,lo31,hi31] */
    static const uint8_t INTERLEAVE_IDX[64] __attribute__((aligned(64))) = {
         0,32, 1,33, 2,34, 3,35, 4,36, 5,37, 6,38, 7,39,
         8,40, 9,41,10,42,11,43,12,44,13,45,14,46,15,47,
        16,48,17,49,18,50,19,51,20,52,21,53,22,54,23,55,
        24,56,25,57,26,58,27,59,28,60,29,61,30,62,31,63
    };

    const __m512i v_lut  = _mm512_load_si512(NIB_LUT);
    const __m512i v_perm = _mm512_load_si512(INTERLEAVE_IDX);
    const __m512i v_mask0f = _mm512_set1_epi8(0x0F);

    /* Parity masks for 4x16-weight subblocks stored in 4 x 64-bit lanes (each 64-bit lane = 16 nibbles) */
    const __m256i m_all = _mm256_set1_epi64x(0x1111111111111111LL);
    const __m256i m_s0  = _mm256_set1_epi64x(0x1010101010101010LL);
    const __m256i m_s1  = _mm256_set1_epi64x(0x1100110011001100LL);
    const __m256i m_s2  = _mm256_set1_epi64x(0x1111000011110000LL);
    const __m256i m_s3  = _mm256_set1_epi64x(0x1111111100000000LL);
    const __m256i m_one = _mm256_set1_epi64x(1LL);

    uint64_t total_syn = 0;

    #pragma omp parallel for num_threads(num_threads) schedule(static) reduction(+:total_syn)
    for (int r = 0; r < M; ++r) {
        const uint8_t* wp = w_nib + (size_t)r * (size_t)(K >> 1);
        const float*   sp = w_scale + (size_t)r * (size_t)ng;
        float acc = 0.0f;

        for (int g = 0; g < ng; g += 2) {
            /* Load 32 packed bytes (= 2 groups of 32 weights = 64 weights) */
            __m256i raw256 = _mm256_loadu_si256((const __m256i*)(wp + (g << 4)));

            if (live_verify) {
                /* Each 64-bit lane holds 16 biased 4-bit nibbles (1 Extended Hamming [16,11,4] block) */
                __m256i p_all = _mm256_and_si256(_mm256_popcnt_epi64(_mm256_and_si256(raw256, m_all)), m_one);
                __m256i p_s0  = _mm256_and_si256(_mm256_popcnt_epi64(_mm256_and_si256(raw256, m_s0)),  m_one);
                __m256i p_s1  = _mm256_and_si256(_mm256_popcnt_epi64(_mm256_and_si256(raw256, m_s1)),  m_one);
                __m256i p_s2  = _mm256_and_si256(_mm256_popcnt_epi64(_mm256_and_si256(raw256, m_s2)),  m_one);
                __m256i p_s3  = _mm256_and_si256(_mm256_popcnt_epi64(_mm256_and_si256(raw256, m_s3)),  m_one);
                __m256i bad   = _mm256_or_si256(_mm256_or_si256(p_all, p_s0), _mm256_or_si256(_mm256_or_si256(p_s1, p_s2), p_s3));
                if (!_mm256_testz_si256(bad, bad)) {
                    total_syn += 1u;
                }
            }

            /* Place 32 lo nibbles in low 256 bits and 32 hi nibbles in high 256 bits */
            __m256i lo256 = _mm256_and_si256(raw256, _mm256_set1_epi8(0x0F));
            __m256i hi256 = _mm256_and_si256(_mm256_srli_epi16(raw256, 4), _mm256_set1_epi8(0x0F));
            __m512i pair  = _mm512_inserti64x4(_mm512_castsi256_si512(lo256), hi256, 1);

            /* Hardware AVX-512 VBMI vpermb: interleave to exact weight order + LUT decode */
            __m512i w_inter = _mm512_permutexvar_epi8(v_perm, pair);
            __m512i w_u8    = _mm512_permutexvar_epi8(_mm512_and_si512(w_inter, v_mask0f), v_lut);

            /* Load 64 signed int8 activations & run hardware VNNI vpdpbusd! */
            __m512i x_s8 = _mm512_loadu_si512(x_q8 + (g << 5));
            __m512i dot16 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w_u8, x_s8);

            /* Low 256 bits = group g (8 int32 lanes = 32 weights); High 256 bits = group g+1 */
            __m256i d0 = _mm512_castsi512_si256(dot16);
            __m256i d1 = _mm512_extracti64x4_epi64(dot16, 1);

            /* Horizontal sum of 8 int32 lanes for d0 and d1 */
            __m128i s0 = _mm_add_epi32(_mm256_castsi256_si128(d0), _mm256_extracti128_si256(d0, 1));
            s0 = _mm_add_epi32(s0, _mm_shuffle_epi32(s0, _MM_SHUFFLE(1, 0, 3, 2)));
            s0 = _mm_add_epi32(s0, _mm_shuffle_epi32(s0, _MM_SHUFFLE(2, 3, 0, 1)));
            int32_t idot0 = _mm_cvtsi128_si32(s0) - (x_gsum[g] << 3);

            __m128i s1 = _mm_add_epi32(_mm256_castsi256_si128(d1), _mm256_extracti128_si256(d1, 1));
            s1 = _mm_add_epi32(s1, _mm_shuffle_epi32(s1, _MM_SHUFFLE(1, 0, 3, 2)));
            s1 = _mm_add_epi32(s1, _mm_shuffle_epi32(s1, _MM_SHUFFLE(2, 3, 0, 1)));
            int32_t idot1 = _mm_cvtsi128_si32(s1) - (x_gsum[g + 1] << 3);

            acc += (float)idot0 * (sp[g] * x_scale[g]) + (float)idot1 * (sp[g + 1] * x_scale[g + 1]);
        }

        if (bias != NULL) acc += bias[r];
        y_out[r] = add_residual ? (y_out[r] + acc) : acc;
    }

    if (syn_err_acc != NULL) *syn_err_acc += total_syn;
}

static void rope_and_gqa_cpu(
    float* q, float* k, const float* v,
    float* k_cache, float* v_cache, float* attn_out, int pos
) {
    float cos_t[32], sin_t[32];
    for (int j = 0; j < 32; ++j) {
        float angle = (float)pos * powf(QWEN_ROPE_THETA, -(float)(2 * j) / (float)QWEN_HEAD_DIM);
        cos_t[j] = cosf(angle);
        sin_t[j] = sinf(angle);
    }
    for (int h = 0; h < QWEN_Q_HEADS; ++h) {
        float* qh = q + h * QWEN_HEAD_DIM;
        for (int j = 0; j < 32; ++j) {
            float q0 = qh[j], q1 = qh[j + 32];
            qh[j]      = q0 * cos_t[j] - q1 * sin_t[j];
            qh[j + 32] = q1 * cos_t[j] + q0 * sin_t[j];
        }
    }
    float* kc_dst = k_cache + (size_t)pos * QWEN_KV_DIM;
    float* vc_dst = v_cache + (size_t)pos * QWEN_KV_DIM;
    for (int h = 0; h < QWEN_KV_HEADS; ++h) {
        float* kh = k + h * QWEN_HEAD_DIM;
        float* kdst = kc_dst + h * QWEN_HEAD_DIM;
        float* vdst = vc_dst + h * QWEN_HEAD_DIM;
        const float* vh = v + h * QWEN_HEAD_DIM;
        for (int j = 0; j < 32; ++j) {
            float k0 = kh[j], k1 = kh[j + 32];
            kdst[j]      = k0 * cos_t[j] - k1 * sin_t[j];
            kdst[j + 32] = k1 * cos_t[j] + k0 * sin_t[j];
            vdst[j]      = vh[j];
            vdst[j + 32] = vh[j + 32];
        }
    }

    int seq_len = pos + 1;
    float scores[MAX_SEQ_LEN];
    for (int qh = 0; qh < QWEN_Q_HEADS; ++qh) {
        int kvh = qh / 7;
        const float* qp = q + qh * QWEN_HEAD_DIM;
        float smax = -1.0e30f;
        for (int t = 0; t < seq_len; ++t) {
            const float* kp = k_cache + (size_t)t * QWEN_KV_DIM + kvh * QWEN_HEAD_DIM;
            float dot = 0.0f;
            for (int d = 0; d < QWEN_HEAD_DIM; ++d) dot += qp[d] * kp[d];
            float sc = dot * 0.125f;
            scores[t] = sc;
            if (sc > smax) smax = sc;
        }
        float ssum = 0.0f;
        for (int t = 0; t < seq_len; ++t) {
            scores[t] = expf(scores[t] - smax);
            ssum += scores[t];
        }
        float inv = 1.0f / ssum;
        float* op = attn_out + qh * QWEN_HEAD_DIM;
        memset(op, 0, QWEN_HEAD_DIM * sizeof(float));
        for (int t = 0; t < seq_len; ++t) {
            float w = scores[t] * inv;
            const float* vp = v_cache + (size_t)t * QWEN_KV_DIM + kvh * QWEN_HEAD_DIM;
            for (int d = 0; d < QWEN_HEAD_DIM; ++d) op[d] += w * vp[d];
        }
    }
}

/* AVX-512 tied BF16 lm_head GEMV (151936 x 896) */
static void bf16_lm_head_avx512(
    const uint16_t* embed_bf16, const float* h_norm, float* logits_out, int num_threads
) {
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int v = 0; v < QWEN_VOCAB; ++v) {
        const uint16_t* rp = embed_bf16 + (size_t)v * QWEN_H;
        __m512 vacc = _mm512_setzero_ps();
        for (int i = 0; i < QWEN_H; i += 16) {
            __m256i b16 = _mm256_loadu_si256((const __m256i*)(rp + i));
            __m512i u32 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(b16), 16);
            __m512 vf = _mm512_castsi512_ps(u32);
            __m512 vh = _mm512_loadu_ps(h_norm + i);
            vacc = _mm512_fmadd_ps(vf, vh, vacc);
        }
        logits_out[v] = _mm512_reduce_add_ps(vacc);
    }
}

/* Public C ABI */
void* qwen25_cpu_create(int num_threads) {
    QwenEngineCPU* eng = (QwenEngineCPU*)alloc_aligned64(sizeof(QwenEngineCPU));
    eng->num_threads = (num_threads > 0) ? num_threads : 8;
    eng->embed_bf16 = (uint16_t*)alloc_aligned64((size_t)QWEN_VOCAB * QWEN_H * sizeof(uint16_t));
    eng->final_norm = (float*)alloc_aligned64(QWEN_H * sizeof(float));
    eng->k_cache    = (float*)alloc_aligned64((size_t)QWEN_LAYERS * MAX_SEQ_LEN * QWEN_KV_DIM * sizeof(float));
    eng->v_cache    = (float*)alloc_aligned64((size_t)QWEN_LAYERS * MAX_SEQ_LEN * QWEN_KV_DIM * sizeof(float));
    for (int l = 0; l < QWEN_LAYERS; ++l) {
        for (int p = 0; p < 7; ++p) {
            size_t ng = (size_t)PROJ_M[p] * (size_t)(PROJ_K[p] / 32);
            eng->layers[l].w_nib[p]   = (uint8_t*)alloc_aligned64(ng * 16u);
            eng->layers[l].w_scale[p] = (float*)alloc_aligned64(ng * sizeof(float));
        }
        eng->layers[l].bias[0]   = (float*)alloc_aligned64(QWEN_H * sizeof(float));
        eng->layers[l].bias[1]   = (float*)alloc_aligned64(QWEN_KV_DIM * sizeof(float));
        eng->layers[l].bias[2]   = (float*)alloc_aligned64(QWEN_KV_DIM * sizeof(float));
        eng->layers[l].in_norm   = (float*)alloc_aligned64(QWEN_H * sizeof(float));
        eng->layers[l].post_norm = (float*)alloc_aligned64(QWEN_H * sizeof(float));
    }
    return eng;
}

void qwen25_cpu_destroy(void* handle) {
    if (!handle) return;
    QwenEngineCPU* eng = (QwenEngineCPU*)handle;
    for (int l = 0; l < QWEN_LAYERS; ++l) {
        for (int p = 0; p < 7; ++p) {
            free(eng->layers[l].w_nib[p]);
            free(eng->layers[l].w_scale[p]);
        }
        for (int b = 0; b < 3; ++b) free(eng->layers[l].bias[b]);
        free(eng->layers[l].in_norm);
        free(eng->layers[l].post_norm);
    }
    free(eng->embed_bf16);
    free(eng->final_norm);
    free(eng->k_cache);
    free(eng->v_cache);
    free(eng);
}

void qwen25_cpu_load_globals(void* handle, const uint16_t* embed_bf16, const float* final_norm_f32) {
    QwenEngineCPU* eng = (QwenEngineCPU*)handle;
    memcpy(eng->embed_bf16, embed_bf16, (size_t)QWEN_VOCAB * QWEN_H * sizeof(uint16_t));
    memcpy(eng->final_norm, final_norm_f32, QWEN_H * sizeof(float));
}

void qwen25_cpu_load_layer_norms_biases(
    void* handle, int layer,
    const float* in_norm, const float* post_norm,
    const float* q_bias, const float* k_bias, const float* v_bias
) {
    QwenEngineCPU* eng = (QwenEngineCPU*)handle;
    memcpy(eng->layers[layer].in_norm,   in_norm,   QWEN_H * sizeof(float));
    memcpy(eng->layers[layer].post_norm, post_norm, QWEN_H * sizeof(float));
    memcpy(eng->layers[layer].bias[0],   q_bias,    QWEN_H * sizeof(float));
    memcpy(eng->layers[layer].bias[1],   k_bias,    QWEN_KV_DIM * sizeof(float));
    memcpy(eng->layers[layer].bias[2],   v_bias,    QWEN_KV_DIM * sizeof(float));
}

void qwen25_cpu_load_proj(void* handle, int layer, int proj, const uint8_t* nib_host, const uint16_t* scale_fp16_host) {
    QwenEngineCPU* eng = (QwenEngineCPU*)handle;
    size_t ng = (size_t)PROJ_M[proj] * (size_t)(PROJ_K[proj] / 32);
    memcpy(eng->layers[layer].w_nib[proj], nib_host, ng * 16u);
    for (size_t i = 0; i < ng; ++i) {
        eng->layers[layer].w_scale[proj][i] = fp16_to_fp32(scale_fp16_host[i]);
    }
}

void qwen25_cpu_reset_kv(void* handle) {
    QwenEngineCPU* eng = (QwenEngineCPU*)handle;
    memset(eng->k_cache, 0, (size_t)QWEN_LAYERS * MAX_SEQ_LEN * QWEN_KV_DIM * sizeof(float));
    memset(eng->v_cache, 0, (size_t)QWEN_LAYERS * MAX_SEQ_LEN * QWEN_KV_DIM * sizeof(float));
    eng->syn_errors = 0;
}

uint64_t qwen25_cpu_get_syn_errors(void* handle) {
    return ((QwenEngineCPU*)handle)->syn_errors;
}

void qwen25_cpu_forward_token(void* handle, int token_id, int pos, int live_verify, float* logits_out) {
    QwenEngineCPU* eng = (QwenEngineCPU*)handle;
    const uint16_t* ep = eng->embed_bf16 + (size_t)token_id * QWEN_H;
    for (int i = 0; i < QWEN_H; ++i) eng->h[i] = bf16_to_fp32(ep[i]);

    int nt = eng->num_threads;
    for (int l = 0; l < QWEN_LAYERS; ++l) {
        QwenLayerCPU* ly = &eng->layers[l];
        rmsnorm_q8_cpu(eng->h, ly->in_norm, eng->x_q8, eng->x_scale, eng->x_gsum, QWEN_H);

        a16_gemv_avx512_vnni(ly->w_nib[0], ly->w_scale[0], eng->x_q8, eng->x_scale, eng->x_gsum, ly->bias[0], eng->q, QWEN_H,      QWEN_H, 0, live_verify, &eng->syn_errors, nt);
        a16_gemv_avx512_vnni(ly->w_nib[1], ly->w_scale[1], eng->x_q8, eng->x_scale, eng->x_gsum, ly->bias[1], eng->k, QWEN_KV_DIM, QWEN_H, 0, live_verify, &eng->syn_errors, nt);
        a16_gemv_avx512_vnni(ly->w_nib[2], ly->w_scale[2], eng->x_q8, eng->x_scale, eng->x_gsum, ly->bias[2], eng->v, QWEN_KV_DIM, QWEN_H, 0, live_verify, &eng->syn_errors, nt);

        float* lk = eng->k_cache + (size_t)l * MAX_SEQ_LEN * QWEN_KV_DIM;
        float* lv = eng->v_cache + (size_t)l * MAX_SEQ_LEN * QWEN_KV_DIM;
        rope_and_gqa_cpu(eng->q, eng->k, eng->v, lk, lv, eng->attn_out, pos);

        quantize_fp32_q8_cpu(eng->attn_out, eng->x_q8, eng->x_scale, eng->x_gsum, QWEN_H);
        a16_gemv_avx512_vnni(ly->w_nib[3], ly->w_scale[3], eng->x_q8, eng->x_scale, eng->x_gsum, NULL, eng->h, QWEN_H, QWEN_H, 1, live_verify, &eng->syn_errors, nt);

        rmsnorm_q8_cpu(eng->h, ly->post_norm, eng->x_q8, eng->x_scale, eng->x_gsum, QWEN_H);
        a16_gemv_avx512_vnni(ly->w_nib[4], ly->w_scale[4], eng->x_q8, eng->x_scale, eng->x_gsum, NULL, eng->gate, QWEN_I, QWEN_H, 0, live_verify, &eng->syn_errors, nt);
        a16_gemv_avx512_vnni(ly->w_nib[5], ly->w_scale[5], eng->x_q8, eng->x_scale, eng->x_gsum, NULL, eng->up,   QWEN_I, QWEN_H, 0, live_verify, &eng->syn_errors, nt);

        silu_mul_q8_cpu(eng->gate, eng->up, eng->x_q8, eng->x_scale, eng->x_gsum, QWEN_I, nt);
        a16_gemv_avx512_vnni(ly->w_nib[6], ly->w_scale[6], eng->x_q8, eng->x_scale, eng->x_gsum, NULL, eng->h, QWEN_H, QWEN_I, 1, live_verify, &eng->syn_errors, nt);
    }

    if (logits_out != NULL) {
        __m512 vsq = _mm512_setzero_ps();
        for (int i = 0; i < QWEN_H; i += 16) {
            __m512 vx = _mm512_loadu_ps(eng->h + i);
            vsq = _mm512_fmadd_ps(vx, vx, vsq);
        }
        float inv_rms = 1.0f / sqrtf(_mm512_reduce_add_ps(vsq) / (float)QWEN_H + QWEN_RMS_EPS);
        for (int i = 0; i < QWEN_H; ++i) eng->h_norm[i] = eng->h[i] * inv_rms * eng->final_norm[i];
        bf16_lm_head_avx512(eng->embed_bf16, eng->h_norm, logits_out, nt);
    }
}

/* Sustained zero-Python-overhead CPU decode loop */
int qwen25_cpu_bench_decode_sustained(
    void* handle,
    int seed_token,
    int n_iters,
    int live_verify,
    int include_lm_head
) {
    float* logits_buf = NULL;
    if (include_lm_head) {
        logits_buf = (float*)alloc_aligned64((size_t)QWEN_VOCAB * sizeof(float));
    }
    int cur_tok = seed_token;
    for (int it = 0; it < n_iters; ++it) {
        int pos = it & 63;
        qwen25_cpu_forward_token(handle, cur_tok, pos, live_verify, logits_buf);
        if (include_lm_head) {
            int best_i = 0;
            float best_v = logits_buf[0];
            for (int v = 1; v < QWEN_VOCAB; ++v) {
                if (logits_buf[v] > best_v) { best_v = logits_buf[v]; best_i = v; }
            }
            cur_tok = best_i;
        }
    }
    if (logits_buf) free(logits_buf);
    return cur_tok;
}
