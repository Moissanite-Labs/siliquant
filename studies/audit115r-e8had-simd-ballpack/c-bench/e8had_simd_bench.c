/* Track S15c (AUDIT-115): E_8_Had Zero-Table WHT_8 Leaf Transform, Batched SIMD/AVX2
 * Unranking, Fused Dequant-GEMV (Materialized vs Adjoint WHT_8), Matched Q4_K_M-Style
 * AVX2/FMA GEMV Baseline, and Stage-by-Stage Cycle Breakdown in pure C11 + AVX2/FMA.
 *
 * Zero hardware division instructions on every hot unranking & GEMV path.
 * Build (0 warnings under gcc & clang):
 *   cc -O3 -march=native -Wall -Wextra -Werror -std=c11 -shared -fPIC -pthread \
 *      -o libe8had_simd.so e8had_simd_bench.c -lm
 */
#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__AVX2__)
#include <immintrin.h>
#define HAS_AVX2 1
#else
#define HAS_AVX2 0
#endif

typedef unsigned __int128 u128;

typedef struct {
    uint64_t w[3];
} u192;

typedef struct {
    uint64_t m[3];
    uint32_t shift;
    uint32_t pad;
} Recip192;

typedef struct {
    u128 m;
    uint32_t shift;
    uint32_t pad;
} Recip128;

typedef struct {
    uint64_t m;
    uint32_t shift;
    uint32_t pad;
} Recip64;

typedef struct {
    uint32_t m;
    uint32_t shift;
} Recip32;

int e8had_has_avx2(void) {
    return HAS_AVX2;
}

static inline uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static inline uint64_t thread_cpu_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/* ============================================================================
 * Zero-Division Multiply-Shift & Branchless Checkpoint Search
 * ============================================================================ */

static inline int u192_le(const u192 *a, const u192 *b) {
    if (a->w[2] != b->w[2]) return a->w[2] < b->w[2];
    if (a->w[1] != b->w[1]) return a->w[1] < b->w[1];
    return a->w[0] <= b->w[0];
}

static inline int u192_ge(const u192 *a, const u192 *b) {
    return u192_le(b, a);
}

static inline u192 u192_sub(const u192 *a, const u192 *b) {
    u192 out;
    u128 borrow = 0;
    for (int i = 0; i < 3; ++i) {
        u128 d = (u128)a->w[i] - (u128)b->w[i] - borrow;
        out.w[i] = (uint64_t)d;
        borrow = (d >> 64) ? 1u : 0u;
    }
    return out;
}

static inline u128 mul_shr_u192_to_u128(const u192 *a, const uint64_t b[3], uint32_t shift) {
    uint64_t p[6] = {0, 0, 0, 0, 0, 0};
    for (int i = 0; i < 3; ++i) {
        u128 carry = 0;
        for (int j = 0; j < 3; ++j) {
            u128 cur = (u128)a->w[i] * b[j] + p[i + j] + carry;
            p[i + j] = (uint64_t)cur;
            carry = cur >> 64;
        }
        p[i + 3] = (uint64_t)carry;
    }
    uint32_t w = shift >> 6;
    uint32_t bit = shift & 63u;
    uint64_t lo = (w < 6u) ? p[w] : 0u;
    uint64_t mid = (w + 1u < 6u) ? p[w + 1u] : 0u;
    uint64_t hi = (w + 2u < 6u) ? p[w + 2u] : 0u;
    if (bit != 0u) {
        lo = (lo >> bit) | (mid << (64u - bit));
        mid = (mid >> bit) | (hi << (64u - bit));
    }
    return ((u128)mid << 64) | lo;
}

static inline uint64_t mul_shr_u128_to_u64(u128 a, u128 b, uint32_t shift) {
    uint64_t a0 = (uint64_t)a, a1 = (uint64_t)(a >> 64);
    uint64_t b0 = (uint64_t)b, b1 = (uint64_t)(b >> 64);
    u128 p00 = (u128)a0 * b0;
    u128 p01 = (u128)a0 * b1;
    u128 p10 = (u128)a1 * b0;
    u128 p11 = (u128)a1 * b1;
    uint64_t r0 = (uint64_t)p00;
    u128 t = (p00 >> 64) + (uint64_t)p01 + (uint64_t)p10;
    uint64_t r1 = (uint64_t)t;
    u128 hi = p11 + (p01 >> 64) + (p10 >> 64) + (t >> 64);
    uint64_t p[4] = {r0, r1, (uint64_t)hi, (uint64_t)(hi >> 64)};
    uint32_t w = shift >> 6;
    uint32_t bit = shift & 63u;
    uint64_t lo = (w < 4u) ? p[w] : 0u;
    uint64_t mid = (w + 1u < 4u) ? p[w + 1u] : 0u;
    if (bit != 0u) {
        lo = (lo >> bit) | (mid << (64u - bit));
    }
    return lo;
}

static inline uint64_t mul_shr_u64_to_u64(uint64_t a, uint64_t b, uint32_t shift) {
    return (uint64_t)(((u128)a * b) >> shift);
}

static inline uint32_t mul_shr_u32_to_u32(uint32_t a, uint32_t b, uint32_t shift) {
    return (uint32_t)(((uint64_t)a * b) >> shift);
}

static inline u192 load_u192_le(const uint8_t *in, int nbytes) {
    u192 v = {{0, 0, 0}};
    for (int i = 0; i < nbytes; ++i) {
        v.w[i >> 3] |= (uint64_t)in[i] << (8 * (i & 7));
    }
    return v;
}

static inline uint32_t find_cp_u128(const u128 *cp, uint32_t nblk, u128 rem) {
    uint32_t lo = 0, len = nblk;
    while (len > 1u) {
        uint32_t half = len >> 1;
        uint32_t mid = lo + half;
        lo = (cp[mid] <= rem) ? mid : lo;
        len -= half;
    }
    return lo;
}

static inline uint32_t find_cp_u64(const uint64_t *cp, uint32_t nblk, uint64_t rem) {
    uint32_t lo = 0, len = nblk;
    while (len > 1u) {
        uint32_t half = len >> 1;
        uint32_t mid = lo + half;
        lo = (cp[mid] <= rem) ? mid : lo;
        len -= half;
    }
    return lo;
}

static inline uint32_t find_cp_u32(const uint32_t *cp, uint32_t nblk, uint32_t rem) {
    uint32_t lo = 0, len = nblk;
    while (len > 1u) {
        uint32_t half = len >> 1;
        uint32_t mid = lo + half;
        lo = (cp[mid] <= rem) ? mid : lo;
        len -= half;
    }
    return lo;
}

/* ============================================================================
 * E_8^4 Context & Init (Identical Exact DP Tables for Both E_8 & E_8_Had!)
 * ============================================================================ */

#define E8_STRIDE    16
#define E8_IN_STRIDE 16
#define E8_Q_STRIDE  16

typedef struct {
    int m_max;
    int r_max;
    int t_max;
    int bytes;
    int use_fast_inner; /* 1 for 136b (m=391), 0 for L1d-resident 88b (m=48) */
    u192 total_count;
    u192 *c_root;
    u128 *s_blk2;
    u128 *w_blk2;
    uint64_t *s_blk1;
    uint64_t *w_blk1;
    Recip192 *rec_root;
    Recip128 *rec_b1;
    Recip128 *rec_s1;
    uint64_t *even_c8;
    uint32_t *even_c4[2];
    uint16_t *even_c2[2];
    uint16_t *even_p2_off[2];
    int8_t *even_p2_pts[2];
    uint32_t *odd_c4[2];
    uint16_t *odd_c2[2];
    uint16_t *odd_p2_off[2];
    int8_t *odd_p2_pts[2];
    Recip64 *rec_even_c4[2];
    Recip32 *rec_even_c2[2];
    Recip64 *rec_odd_c4[2];
    Recip32 *rec_odd_c2[2];
    /* Packed uint32 tables for AVX2 vpgatherdd SIMD8 experiment */
    uint32_t *even_c2_u32[2];
    uint32_t *odd_c2_u32[2];
    /* Strided checkpoints */
    uint32_t *cp_m_off;
    uint32_t *cp_m_len;
    u128 *cp_s_blk2;
    u128 *cp_w_blk2;
    uint64_t *e8_even_p0_tot;
    uint64_t *e8_odd_p0_tot;
    uint32_t *cp_ec8_off;
    uint32_t *cp_ec8_len;
    uint64_t *cp_ec8[2];
    uint32_t *cp_oc8_off;
    uint32_t *cp_oc8_len;
    uint64_t *cp_oc8[2];
    uint32_t *q_even_p0_tot[2];
    uint32_t *q_odd_p0_tot[2];
    uint32_t *cp_eq_off;
    uint32_t *cp_eq_len;
    uint32_t *cp_eq[4];
    uint32_t *cp_oq_off;
    uint32_t *cp_oq_len;
    uint32_t *cp_oq[4];
} E8Ctx;

static E8Ctx g_e8;
static int g_e8_ready = 0;

static void e8_free(void) {
    if (!g_e8_ready) return;
    free(g_e8.c_root);
    free(g_e8.s_blk2);
    free(g_e8.w_blk2);
    free(g_e8.s_blk1);
    free(g_e8.w_blk1);
    free(g_e8.rec_root);
    free(g_e8.rec_b1);
    free(g_e8.rec_s1);
    free(g_e8.even_c8);
    for (int p = 0; p < 2; ++p) {
        free(g_e8.even_c4[p]);
        free(g_e8.even_c2[p]);
        free(g_e8.even_c2_u32[p]);
        free(g_e8.even_p2_off[p]);
        free(g_e8.even_p2_pts[p]);
        free(g_e8.odd_c4[p]);
        free(g_e8.odd_c2[p]);
        free(g_e8.odd_c2_u32[p]);
        free(g_e8.odd_p2_off[p]);
        free(g_e8.odd_p2_pts[p]);
        free(g_e8.rec_even_c4[p]);
        free(g_e8.rec_even_c2[p]);
        free(g_e8.rec_odd_c4[p]);
        free(g_e8.rec_odd_c2[p]);
        free(g_e8.cp_ec8[p]);
        free(g_e8.cp_oc8[p]);
        free(g_e8.q_even_p0_tot[p]);
        free(g_e8.q_odd_p0_tot[p]);
    }
    for (int q = 0; q < 4; ++q) {
        free(g_e8.cp_eq[q]);
        free(g_e8.cp_oq[q]);
    }
    free(g_e8.cp_m_off);
    free(g_e8.cp_m_len);
    free(g_e8.cp_s_blk2);
    free(g_e8.cp_w_blk2);
    free(g_e8.e8_even_p0_tot);
    free(g_e8.e8_odd_p0_tot);
    free(g_e8.cp_ec8_off);
    free(g_e8.cp_ec8_len);
    free(g_e8.cp_oc8_off);
    free(g_e8.cp_oc8_len);
    free(g_e8.cp_eq_off);
    free(g_e8.cp_eq_len);
    free(g_e8.cp_oq_off);
    free(g_e8.cp_oq_len);
    memset(&g_e8, 0, sizeof(g_e8));
    g_e8_ready = 0;
}

int e8had_init(
    int m_max,
    int bytes,
    const uint64_t *c_root_limbs,
    const uint64_t *s_blk2_limbs,
    const uint64_t *w_blk2_limbs,
    const uint64_t *s_blk1_in,
    const uint64_t *w_blk1_in,
    const uint64_t *rec_root_m,
    const uint32_t *rec_root_s,
    const uint64_t *rec_b1_m,
    const uint32_t *rec_b1_s,
    const uint64_t *rec_s1_m,
    const uint32_t *rec_s1_s,
    const uint64_t *even_c8_in,
    const uint32_t *even_c4_in,
    const uint16_t *even_c2_in,
    const uint16_t *even_off_in,
    const int8_t *even_pts0_in, int even_n0,
    const int8_t *even_pts1_in, int even_n1,
    const uint32_t *odd_c4_in,
    const uint16_t *odd_c2_in,
    const uint16_t *odd_off_in,
    const int8_t *odd_pts0_in, int odd_n0,
    const int8_t *odd_pts1_in, int odd_n1,
    const uint64_t *rec_ec4_m, const uint32_t *rec_ec4_s,
    const uint32_t *rec_ec2_m, const uint32_t *rec_ec2_s,
    const uint64_t *rec_oc4_m, const uint32_t *rec_oc4_s,
    const uint32_t *rec_oc2_m, const uint32_t *rec_oc2_s
) {
    if (m_max < 1 || m_max > 2048 || bytes < 1 || bytes > 24) return 1;
    e8_free();
    int r_max = 2 * m_max;
    int t_max = m_max - 1;
    size_t M1 = (size_t)m_max + 1u;
    size_t M2 = (size_t)m_max + 2u;
    size_t R1 = (size_t)r_max + 1u;
    size_t R2 = (size_t)r_max + 2u;
    size_t T1 = (size_t)t_max + 1u;
    size_t T2 = (size_t)t_max + 2u;

    g_e8.m_max = m_max;
    g_e8.r_max = r_max;
    g_e8.t_max = t_max;
    g_e8.bytes = bytes;
    g_e8.use_fast_inner = (m_max > 64) ? 1 : 0;

    g_e8.c_root = (u192 *)malloc(M2 * sizeof(u192));
    g_e8.s_blk2 = (u128 *)malloc(M1 * sizeof(u128));
    g_e8.w_blk2 = (u128 *)malloc(M1 * sizeof(u128));
    g_e8.s_blk1 = (uint64_t *)malloc(M1 * sizeof(uint64_t));
    g_e8.w_blk1 = (uint64_t *)malloc(M1 * sizeof(uint64_t));
    g_e8.rec_root = (Recip192 *)calloc(M1, sizeof(Recip192));
    g_e8.rec_b1 = (Recip128 *)calloc(M1, sizeof(Recip128));
    g_e8.rec_s1 = (Recip128 *)calloc(M1, sizeof(Recip128));
    g_e8.even_c8 = (uint64_t *)malloc(M1 * sizeof(uint64_t));

    for (size_t i = 0; i < M2; ++i) {
        g_e8.c_root[i].w[0] = c_root_limbs[3u * i + 0u];
        g_e8.c_root[i].w[1] = c_root_limbs[3u * i + 1u];
        g_e8.c_root[i].w[2] = c_root_limbs[3u * i + 2u];
    }
    g_e8.total_count = g_e8.c_root[m_max + 1];
    for (size_t i = 0; i < M1; ++i) {
        g_e8.s_blk2[i] = ((u128)s_blk2_limbs[2u * i + 1u] << 64) | s_blk2_limbs[2u * i];
        g_e8.w_blk2[i] = ((u128)w_blk2_limbs[2u * i + 1u] << 64) | w_blk2_limbs[2u * i];
        g_e8.s_blk1[i] = s_blk1_in[i];
        g_e8.w_blk1[i] = w_blk1_in[i];
        g_e8.even_c8[i] = even_c8_in[i];
        g_e8.rec_root[i].m[0] = rec_root_m[3u * i + 0u];
        g_e8.rec_root[i].m[1] = rec_root_m[3u * i + 1u];
        g_e8.rec_root[i].m[2] = rec_root_m[3u * i + 2u];
        g_e8.rec_root[i].shift = rec_root_s[i];
        g_e8.rec_b1[i].m = ((u128)rec_b1_m[2u * i + 1u] << 64) | rec_b1_m[2u * i];
        g_e8.rec_b1[i].shift = rec_b1_s[i];
        g_e8.rec_s1[i].m = ((u128)rec_s1_m[2u * i + 1u] << 64) | rec_s1_m[2u * i];
        g_e8.rec_s1[i].shift = rec_s1_s[i];
    }

    const int even_ns[2] = {even_n0, even_n1};
    const int8_t *even_pts[2] = {even_pts0_in, even_pts1_in};
    const int odd_ns[2] = {odd_n0, odd_n1};
    const int8_t *odd_pts[2] = {odd_pts0_in, odd_pts1_in};
    for (int p = 0; p < 2; ++p) {
        g_e8.even_c4[p] = (uint32_t *)malloc(R1 * sizeof(uint32_t));
        g_e8.even_c2[p] = (uint16_t *)malloc(R1 * sizeof(uint16_t));
        g_e8.even_c2_u32[p] = (uint32_t *)malloc(R1 * sizeof(uint32_t));
        g_e8.even_p2_off[p] = (uint16_t *)malloc(R2 * sizeof(uint16_t));
        g_e8.even_p2_pts[p] = (int8_t *)malloc((size_t)even_ns[p] * 2u * sizeof(int8_t));
        memcpy(g_e8.even_c4[p], even_c4_in + (size_t)p * R1, R1 * sizeof(uint32_t));
        memcpy(g_e8.even_c2[p], even_c2_in + (size_t)p * R1, R1 * sizeof(uint16_t));
        for (size_t i = 0; i < R1; ++i) g_e8.even_c2_u32[p][i] = g_e8.even_c2[p][i];
        memcpy(g_e8.even_p2_off[p], even_off_in + (size_t)p * R2, R2 * sizeof(uint16_t));
        memcpy(g_e8.even_p2_pts[p], even_pts[p], (size_t)even_ns[p] * 2u * sizeof(int8_t));

        g_e8.odd_c4[p] = (uint32_t *)malloc(T1 * sizeof(uint32_t));
        g_e8.odd_c2[p] = (uint16_t *)malloc(T1 * sizeof(uint16_t));
        g_e8.odd_c2_u32[p] = (uint32_t *)malloc(T1 * sizeof(uint32_t));
        g_e8.odd_p2_off[p] = (uint16_t *)malloc(T2 * sizeof(uint16_t));
        g_e8.odd_p2_pts[p] = (int8_t *)malloc((size_t)odd_ns[p] * 2u * sizeof(int8_t));
        memcpy(g_e8.odd_c4[p], odd_c4_in + (size_t)p * T1, T1 * sizeof(uint32_t));
        memcpy(g_e8.odd_c2[p], odd_c2_in + (size_t)p * T1, T1 * sizeof(uint16_t));
        for (size_t i = 0; i < T1; ++i) g_e8.odd_c2_u32[p][i] = g_e8.odd_c2[p][i];
        memcpy(g_e8.odd_p2_off[p], odd_off_in + (size_t)p * T2, T2 * sizeof(uint16_t));
        memcpy(g_e8.odd_p2_pts[p], odd_pts[p], (size_t)odd_ns[p] * 2u * sizeof(int8_t));

        g_e8.rec_even_c4[p] = (Recip64 *)calloc(R1, sizeof(Recip64));
        g_e8.rec_even_c2[p] = (Recip32 *)calloc(R1, sizeof(Recip32));
        for (size_t i = 0; i < R1; ++i) {
            g_e8.rec_even_c4[p][i].m = rec_ec4_m[(size_t)p * R1 + i];
            g_e8.rec_even_c4[p][i].shift = rec_ec4_s[(size_t)p * R1 + i];
            g_e8.rec_even_c2[p][i].m = rec_ec2_m[(size_t)p * R1 + i];
            g_e8.rec_even_c2[p][i].shift = rec_ec2_s[(size_t)p * R1 + i];
        }
        g_e8.rec_odd_c4[p] = (Recip64 *)calloc(T1, sizeof(Recip64));
        g_e8.rec_odd_c2[p] = (Recip32 *)calloc(T1, sizeof(Recip32));
        for (size_t i = 0; i < T1; ++i) {
            g_e8.rec_odd_c4[p][i].m = rec_oc4_m[(size_t)p * T1 + i];
            g_e8.rec_odd_c4[p][i].shift = rec_oc4_s[(size_t)p * T1 + i];
            g_e8.rec_odd_c2[p][i].m = rec_oc2_m[(size_t)p * T1 + i];
            g_e8.rec_odd_c2[p][i].shift = rec_oc2_s[(size_t)p * T1 + i];
        }
    }

    /* Checkpoints */
    g_e8.cp_m_off = (uint32_t *)malloc(M1 * sizeof(uint32_t));
    g_e8.cp_m_len = (uint32_t *)malloc(M1 * sizeof(uint32_t));
    uint32_t total_cp = 0;
    for (int m = 0; m <= m_max; ++m) {
        uint32_t nblk = (uint32_t)(m / E8_STRIDE) + 1u;
        g_e8.cp_m_off[m] = total_cp;
        g_e8.cp_m_len[m] = nblk;
        total_cp += nblk;
    }
    g_e8.cp_s_blk2 = (u128 *)malloc((size_t)total_cp * sizeof(u128));
    g_e8.cp_w_blk2 = (u128 *)malloc((size_t)total_cp * sizeof(u128));
    for (int m = 0; m <= m_max; ++m) {
        uint32_t base = g_e8.cp_m_off[m];
        u128 as2 = 0, aw2 = 0;
        for (int m_l = 0; m_l <= m; ++m_l) {
            if ((m_l & (E8_STRIDE - 1)) == 0) {
                uint32_t slot = base + (uint32_t)(m_l / E8_STRIDE);
                g_e8.cp_s_blk2[slot] = as2;
                g_e8.cp_w_blk2[slot] = aw2;
            }
            as2 += (u128)g_e8.s_blk1[m_l] * g_e8.s_blk1[m - m_l];
            aw2 += (u128)g_e8.s_blk1[m_l] * g_e8.w_blk1[m - m_l];
        }
    }

    g_e8.e8_even_p0_tot = (uint64_t *)calloc(M1, sizeof(uint64_t));
    g_e8.e8_odd_p0_tot = (uint64_t *)calloc(M1, sizeof(uint64_t));
    g_e8.cp_ec8_off = (uint32_t *)malloc(M1 * sizeof(uint32_t));
    g_e8.cp_ec8_len = (uint32_t *)malloc(M1 * sizeof(uint32_t));
    uint32_t tot_ec8 = 0;
    for (int m = 0; m <= m_max; ++m) {
        uint32_t nblk = (uint32_t)((2 * m) / E8_IN_STRIDE) + 1u;
        g_e8.cp_ec8_off[m] = tot_ec8;
        g_e8.cp_ec8_len[m] = nblk;
        tot_ec8 += nblk;
    }
    g_e8.cp_oc8_off = (uint32_t *)malloc(T1 * sizeof(uint32_t));
    g_e8.cp_oc8_len = (uint32_t *)malloc(T1 * sizeof(uint32_t));
    uint32_t tot_oc8 = 0;
    for (int t = 0; t <= t_max; ++t) {
        uint32_t nblk = (uint32_t)(t / E8_IN_STRIDE) + 1u;
        g_e8.cp_oc8_off[t] = tot_oc8;
        g_e8.cp_oc8_len[t] = nblk;
        tot_oc8 += nblk;
    }
    for (int p = 0; p < 2; ++p) {
        g_e8.cp_ec8[p] = (uint64_t *)malloc((size_t)tot_ec8 * sizeof(uint64_t));
        g_e8.cp_oc8[p] = (uint64_t *)malloc((size_t)tot_oc8 * sizeof(uint64_t));
    }
    for (int m = 0; m <= m_max; ++m) {
        int r8 = 2 * m;
        uint32_t base = g_e8.cp_ec8_off[m];
        for (int p = 0; p < 2; ++p) {
            uint64_t acc = 0;
            for (int a = 0; a <= r8; ++a) {
                if ((a & (E8_IN_STRIDE - 1)) == 0) {
                    g_e8.cp_ec8[p][base + (uint32_t)(a / E8_IN_STRIDE)] = acc;
                }
                acc += (uint64_t)g_e8.even_c4[p][a] * g_e8.even_c4[p][r8 - a];
            }
            if (p == 0) g_e8.e8_even_p0_tot[m] = acc;
        }
        if (m >= 1) {
            int t8 = m - 1;
            uint32_t obase = g_e8.cp_oc8_off[t8];
            for (int p = 0; p < 2; ++p) {
                uint64_t acc = 0;
                for (int a = 0; a <= t8; ++a) {
                    if ((a & (E8_IN_STRIDE - 1)) == 0) {
                        g_e8.cp_oc8[p][obase + (uint32_t)(a / E8_IN_STRIDE)] = acc;
                    }
                    acc += (uint64_t)g_e8.odd_c4[p][a] * g_e8.odd_c4[p][t8 - a];
                }
                if (p == 0) g_e8.e8_odd_p0_tot[m] = acc;
            }
        }
    }

    g_e8.cp_eq_off = (uint32_t *)malloc(R1 * sizeof(uint32_t));
    g_e8.cp_eq_len = (uint32_t *)malloc(R1 * sizeof(uint32_t));
    uint32_t tot_eq = 0;
    for (int r = 0; r <= r_max; ++r) {
        uint32_t nblk = (uint32_t)(r / E8_Q_STRIDE) + 1u;
        g_e8.cp_eq_off[r] = tot_eq;
        g_e8.cp_eq_len[r] = nblk;
        tot_eq += nblk;
    }
    g_e8.cp_oq_off = (uint32_t *)malloc(T1 * sizeof(uint32_t));
    g_e8.cp_oq_len = (uint32_t *)malloc(T1 * sizeof(uint32_t));
    uint32_t tot_oq = 0;
    for (int t = 0; t <= t_max; ++t) {
        uint32_t nblk = (uint32_t)(t / E8_Q_STRIDE) + 1u;
        g_e8.cp_oq_off[t] = tot_oq;
        g_e8.cp_oq_len[t] = nblk;
        tot_oq += nblk;
    }
    for (int p = 0; p < 2; ++p) {
        g_e8.q_even_p0_tot[p] = (uint32_t *)calloc(R1, sizeof(uint32_t));
        g_e8.q_odd_p0_tot[p] = (uint32_t *)calloc(T1, sizeof(uint32_t));
    }
    for (int q = 0; q < 4; ++q) {
        g_e8.cp_eq[q] = (uint32_t *)malloc((size_t)tot_eq * sizeof(uint32_t));
        g_e8.cp_oq[q] = (uint32_t *)malloc((size_t)tot_oq * sizeof(uint32_t));
    }
    for (int pa = 0; pa < 2; ++pa) {
        for (int pb = 0; pb < 2; ++pb) {
            int qid = (pa << 1) | pb;
            for (int r = 0; r <= r_max; ++r) {
                uint32_t base = g_e8.cp_eq_off[r];
                uint32_t acc = 0;
                for (int na = 0; na <= r; ++na) {
                    if ((na & (E8_Q_STRIDE - 1)) == 0) {
                        g_e8.cp_eq[qid][base + (uint32_t)(na / E8_Q_STRIDE)] = acc;
                    }
                    acc += (uint32_t)g_e8.even_c2[pa][na] * g_e8.even_c2[pb][r - na];
                }
                if (pa == 0) g_e8.q_even_p0_tot[pb][r] = acc;
            }
            for (int t = 0; t <= t_max; ++t) {
                uint32_t base = g_e8.cp_oq_off[t];
                uint32_t acc = 0;
                for (int na = 0; na <= t; ++na) {
                    if ((na & (E8_Q_STRIDE - 1)) == 0) {
                        g_e8.cp_oq[qid][base + (uint32_t)(na / E8_Q_STRIDE)] = acc;
                    }
                    acc += (uint32_t)g_e8.odd_c2[pa][na] * g_e8.odd_c2[pb][t - na];
                }
                if (pa == 0) g_e8.q_odd_p0_tot[pb][t] = acc;
            }
        }
    }
    g_e8_ready = 1;
    return 0;
}

/* ============================================================================
 * WHT_8 Leaf Transforms (Exact Integer z = (H_8 * u) / 4 & Float AVX2)
 * ============================================================================ */

static inline void wht8_u_to_z_scalar(const int8_t *u, int8_t *z) {
    int a0 = u[0] + u[1], a1 = u[0] - u[1];
    int a2 = u[2] + u[3], a3 = u[2] - u[3];
    int a4 = u[4] + u[5], a5 = u[4] - u[5];
    int a6 = u[6] + u[7], a7 = u[6] - u[7];

    int b0 = a0 + a2, b1 = a1 + a3, b2 = a0 - a2, b3 = a1 - a3;
    int b4 = a4 + a6, b5 = a5 + a7, b6 = a4 - a6, b7 = a5 - a7;

    z[0] = (int8_t)((b0 + b4) >> 2);
    z[1] = (int8_t)((b1 + b5) >> 2);
    z[2] = (int8_t)((b2 + b6) >> 2);
    z[3] = (int8_t)((b3 + b7) >> 2);
    z[4] = (int8_t)((b0 - b4) >> 2);
    z[5] = (int8_t)((b1 - b5) >> 2);
    z[6] = (int8_t)((b2 - b6) >> 2);
    z[7] = (int8_t)((b3 - b7) >> 2);
}

#if HAS_AVX2
/* Exact AVX2 32-weight (4 x WHT_8) int8 -> int8 leaf transform: z = (H_8 * u) >> 2 */
static inline void wht8x4_u_to_z_avx2(const int8_t *u32, int8_t *z32) {
    /* Load 32 int8s as two 16xint16 vectors (blocks 0,1 and blocks 2,3) */
    __m128i lo16 = _mm_loadu_si128((const __m128i *)u32);
    __m128i hi16 = _mm_loadu_si128((const __m128i *)(u32 + 16));
    __m256i v0 = _mm256_cvtepi8_epi16(lo16); /* [blk0(8), blk1(8)] */
    __m256i v1 = _mm256_cvtepi8_epi16(hi16); /* [blk2(8), blk3(8)] */

    /* Stage 1 (stride 1): pairs (0,1), (2,3), (4,5), (6,7) inside each 128b lane */
    const __m256i sgn1 = _mm256_set1_epi32((int)0xffff0001u); /* [+1, -1] in int16 */
    __m256i p0 = _mm256_shufflelo_epi16(_mm256_shufflehi_epi16(v0, 0xB1), 0xB1);
    __m256i p1 = _mm256_shufflelo_epi16(_mm256_shufflehi_epi16(v1, 0xB1), 0xB1);
    v0 = _mm256_add_epi16(_mm256_sign_epi16(v0, sgn1), p0);
    v1 = _mm256_add_epi16(_mm256_sign_epi16(v1, sgn1), p1);

    /* Stage 2 (stride 2): quads (0,1)<->(2,3), (4,5)<->(6,7) */
    const __m256i sgn2 = _mm256_setr_epi16(
        1, 1, -1, -1, 1, 1, -1, -1,
        1, 1, -1, -1, 1, 1, -1, -1
    );
    __m256i q0 = _mm256_shuffle_epi32(v0, 0xB1);
    __m256i q1 = _mm256_shuffle_epi32(v1, 0xB1);
    v0 = _mm256_add_epi16(_mm256_sign_epi16(v0, sgn2), q0);
    v1 = _mm256_add_epi16(_mm256_sign_epi16(v1, sgn2), q1);

    /* Stage 3 (stride 4): halves (0..3)<->(4..7) inside each 128b lane */
    const __m256i sgn4 = _mm256_setr_epi16(
        1, 1, 1, 1, -1, -1, -1, -1,
        1, 1, 1, 1, -1, -1, -1, -1
    );
    __m256i r0 = _mm256_shuffle_epi32(v0, 0x4E);
    __m256i r1 = _mm256_shuffle_epi32(v1, 0x4E);
    v0 = _mm256_srai_epi16(_mm256_add_epi16(_mm256_sign_epi16(v0, sgn4), r0), 2);
    v1 = _mm256_srai_epi16(_mm256_add_epi16(_mm256_sign_epi16(v1, sgn4), r1), 2);

    /* Pack two 16xint16 back to 32xint8 preserving lane order [blk0, blk1, blk2, blk3] */
    __m256i pk = _mm256_packs_epi16(v0, v1); /* lanes: [blk0, blk2 | blk1, blk3] */
    __m256i ordered = _mm256_permute4x64_epi64(pk, 0xD8); /* [blk0, blk1, blk2, blk3] */
    _mm256_storeu_si256((__m256i *)z32, ordered);
}

static inline float hsum256_ps(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

static inline float dot_i8_32_f32_avx2(const int8_t *w32, const float *x32) {
    __m128i b0 = _mm_loadl_epi64((const __m128i *)(w32 + 0));
    __m128i b1 = _mm_loadl_epi64((const __m128i *)(w32 + 8));
    __m128i b2 = _mm_loadl_epi64((const __m128i *)(w32 + 16));
    __m128i b3 = _mm_loadl_epi64((const __m128i *)(w32 + 24));
    __m256 f0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b0));
    __m256 f1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b1));
    __m256 f2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b2));
    __m256 f3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b3));
    __m256 acc = _mm256_mul_ps(f0, _mm256_loadu_ps(x32 + 0));
    acc = _mm256_fmadd_ps(f1, _mm256_loadu_ps(x32 + 8), acc);
    acc = _mm256_fmadd_ps(f2, _mm256_loadu_ps(x32 + 16), acc);
    acc = _mm256_fmadd_ps(f3, _mm256_loadu_ps(x32 + 24), acc);
    return hsum256_ps(acc);
}
#else
static inline void wht8x4_u_to_z_avx2(const int8_t *u32, int8_t *z32) {
    for (int b = 0; b < 4; ++b) wht8_u_to_z_scalar(u32 + 8 * b, z32 + 8 * b);
}

static inline float dot_i8_32_f32_avx2(const int8_t *w32, const float *x32) {
    float acc = 0.0f;
    for (int i = 0; i < 32; ++i) acc += (float)w32[i] * x32[i];
    return acc;
}
#endif

/* Activation-side exact WHT_8 pre-transform: x_tilde = (H_8 / sqrt(8)) * x, amortized over M rows */
static inline void wht8_f32_vector(const float *x, float *x_tilde, size_t k_cols) {
    const float inv_sqrt8 = 0.3535533905932737622f;
    for (size_t j = 0; j < k_cols; j += 8u) {
        float a0 = x[j + 0] + x[j + 1], a1 = x[j + 0] - x[j + 1];
        float a2 = x[j + 2] + x[j + 3], a3 = x[j + 2] - x[j + 3];
        float a4 = x[j + 4] + x[j + 5], a5 = x[j + 4] - x[j + 5];
        float a6 = x[j + 6] + x[j + 7], a7 = x[j + 6] - x[j + 7];
        float b0 = a0 + a2, b1 = a1 + a3, b2 = a0 - a2, b3 = a1 - a3;
        float b4 = a4 + a6, b5 = a5 + a7, b6 = a4 - a6, b7 = a5 - a7;
        x_tilde[j + 0] = (b0 + b4) * inv_sqrt8;
        x_tilde[j + 1] = (b1 + b5) * inv_sqrt8;
        x_tilde[j + 2] = (b2 + b6) * inv_sqrt8;
        x_tilde[j + 3] = (b3 + b7) * inv_sqrt8;
        x_tilde[j + 4] = (b0 - b4) * inv_sqrt8;
        x_tilde[j + 5] = (b1 - b5) * inv_sqrt8;
        x_tilde[j + 6] = (b2 - b6) * inv_sqrt8;
        x_tilde[j + 7] = (b3 - b7) * inv_sqrt8;
    }
}

/* ============================================================================
 * E_8^4 Zero-Division Tree Unrankers (Outer + Inner Compact / Fast / SIMD8)
 * ============================================================================ */

static inline int root_split_e8_4(const uint8_t *in, int *m_l_out, u128 *idx_l_out, u128 *idx_r_out) {
    u192 rk = load_u192_le(in, g_e8.bytes);
    if (u192_ge(&rk, &g_e8.total_count)) return 1;
    uint32_t lo = 0, len = (uint32_t)g_e8.m_max + 1u;
    const u192 *c_root = g_e8.c_root;
    while (len > 1u) {
        uint32_t half = len >> 1;
        uint32_t mid = lo + half;
        if (u192_le(&c_root[mid], &rk)) lo = mid;
        len -= half;
    }
    int m_l = (int)lo;
    int m_r = g_e8.m_max - m_l;
    u192 rem = u192_sub(&rk, &c_root[m_l]);
    const Recip192 *rec = &g_e8.rec_root[m_r];
    u128 q = mul_shr_u192_to_u128(&rem, rec->m, rec->shift);
    u128 rem_lo128 = ((u128)rem.w[1] << 64) | rem.w[0];
    u128 r = rem_lo128 - q * g_e8.w_blk2[m_r];
    *m_l_out = m_l;
    *idx_l_out = q;
    *idx_r_out = r;
    return 0;
}

static inline void split_e8_blk2_shell(int m_l, u128 idx_l, int *a0_out, uint64_t *l0_out, uint64_t *r0_out) {
    const uint64_t *s1 = g_e8.s_blk1;
    int a0;
    u128 rem0;
    if (g_e8.use_fast_inner) {
        const u128 *cp0 = g_e8.cp_s_blk2 + g_e8.cp_m_off[m_l];
        uint32_t blk0 = find_cp_u128(cp0, g_e8.cp_m_len[m_l], idx_l);
        rem0 = idx_l - cp0[blk0];
        a0 = (int)(blk0 * E8_STRIDE);
    } else {
        rem0 = idx_l;
        a0 = 0;
    }
    for (; a0 <= m_l; ++a0) {
        u128 term = (u128)s1[a0] * s1[m_l - a0];
        if (rem0 < term) break;
        rem0 -= term;
    }
    Recip128 rec0 = g_e8.rec_s1[m_l - a0];
    uint64_t l0 = mul_shr_u128_to_u64(rem0, rec0.m, rec0.shift);
    uint64_t r0 = (uint64_t)rem0 - l0 * s1[m_l - a0];
    *a0_out = a0; *l0_out = l0; *r0_out = r0;
}

static inline void split_e8_blk2_ball(int b_r, u128 idx_r, int *a1_out, uint64_t *l1_out, uint64_t *r1_out) {
    const uint64_t *s1 = g_e8.s_blk1;
    const uint64_t *w1 = g_e8.w_blk1;
    int a1;
    u128 rem1;
    if (g_e8.use_fast_inner) {
        const u128 *cp1 = g_e8.cp_w_blk2 + g_e8.cp_m_off[b_r];
        uint32_t blk1 = find_cp_u128(cp1, g_e8.cp_m_len[b_r], idx_r);
        rem1 = idx_r - cp1[blk1];
        a1 = (int)(blk1 * E8_STRIDE);
    } else {
        rem1 = idx_r;
        a1 = 0;
    }
    for (; a1 <= b_r; ++a1) {
        u128 term = (u128)s1[a1] * w1[b_r - a1];
        if (rem1 < term) break;
        rem1 -= term;
    }
    Recip128 rec1 = g_e8.rec_b1[b_r - a1];
    uint64_t l1 = mul_shr_u128_to_u64(rem1, rec1.m, rec1.shift);
    uint64_t r1 = (uint64_t)rem1 - l1 * w1[b_r - a1];
    *a1_out = a1; *l1_out = l1; *r1_out = r1;
}

static inline void unrank_e8_quad_compact(uint32_t rem, int coset, int norm4, int parity4, int8_t *out) {
    uint32_t p0_tot = (coset == 0) ? g_e8.q_even_p0_tot[parity4][norm4] : g_e8.q_odd_p0_tot[parity4][norm4];
    int pa = (rem < p0_tot) ? 0 : 1;
    if (pa) rem -= p0_tot;
    int pb = parity4 ^ pa;
    const uint16_t *ca = (coset == 0) ? g_e8.even_c2[pa] : g_e8.odd_c2[pa];
    const uint16_t *cb = (coset == 0) ? g_e8.even_c2[pb] : g_e8.odd_c2[pb];
    int na = 0;
    for (; na <= norm4; ++na) {
        uint32_t term = (uint32_t)ca[na] * cb[norm4 - na];
        if (rem < term) break;
        rem -= term;
    }
    int nb = norm4 - na;
    Recip32 rec = (coset == 0) ? g_e8.rec_even_c2[pb][nb] : g_e8.rec_odd_c2[pb][nb];
    uint32_t d = cb[nb];
    uint32_t il = mul_shr_u32_to_u32(rem, rec.m, rec.shift);
    uint32_t ir = rem - il * d;
    uint32_t pl = (uint32_t)((coset == 0) ? g_e8.even_p2_off[pa][na] : g_e8.odd_p2_off[pa][na]) + il;
    uint32_t pr = (uint32_t)((coset == 0) ? g_e8.even_p2_off[pb][nb] : g_e8.odd_p2_off[pb][nb]) + ir;
    const int8_t *pt_a = (coset == 0) ? g_e8.even_p2_pts[pa] : g_e8.odd_p2_pts[pa];
    const int8_t *pt_b = (coset == 0) ? g_e8.even_p2_pts[pb] : g_e8.odd_p2_pts[pb];
    out[0] = pt_a[2u * pl];
    out[1] = pt_a[2u * pl + 1u];
    out[2] = pt_b[2u * pr];
    out[3] = pt_b[2u * pr + 1u];
}

static inline void unrank_e8_quad_fast(uint32_t rem, int coset, int norm4, int parity4, int8_t *out) {
    uint32_t p0_tot = (coset == 0) ? g_e8.q_even_p0_tot[parity4][norm4] : g_e8.q_odd_p0_tot[parity4][norm4];
    int pa = (rem < p0_tot) ? 0 : 1;
    if (pa) rem -= p0_tot;
    int pb = parity4 ^ pa;
    int qid = (pa << 1) | pb;
    const uint16_t *ca = (coset == 0) ? g_e8.even_c2[pa] : g_e8.odd_c2[pa];
    const uint16_t *cb = (coset == 0) ? g_e8.even_c2[pb] : g_e8.odd_c2[pb];
    const uint32_t *cp = (coset == 0) ? (g_e8.cp_eq[qid] + g_e8.cp_eq_off[norm4])
                                      : (g_e8.cp_oq[qid] + g_e8.cp_oq_off[norm4]);
    uint32_t nblk = (coset == 0) ? g_e8.cp_eq_len[norm4] : g_e8.cp_oq_len[norm4];
    uint32_t blk = find_cp_u32(cp, nblk, rem);
    rem -= cp[blk];
    int na = (int)(blk * E8_Q_STRIDE);
    for (; na <= norm4; ++na) {
        uint32_t term = (uint32_t)ca[na] * cb[norm4 - na];
        if (rem < term) break;
        rem -= term;
    }
    int nb = norm4 - na;
    Recip32 rec = (coset == 0) ? g_e8.rec_even_c2[pb][nb] : g_e8.rec_odd_c2[pb][nb];
    uint32_t d = cb[nb];
    uint32_t il = mul_shr_u32_to_u32(rem, rec.m, rec.shift);
    uint32_t ir = rem - il * d;
    uint32_t pl = (uint32_t)((coset == 0) ? g_e8.even_p2_off[pa][na] : g_e8.odd_p2_off[pa][na]) + il;
    uint32_t pr = (uint32_t)((coset == 0) ? g_e8.even_p2_off[pb][nb] : g_e8.odd_p2_off[pb][nb]) + ir;
    const int8_t *pt_a = (coset == 0) ? g_e8.even_p2_pts[pa] : g_e8.odd_p2_pts[pa];
    const int8_t *pt_b = (coset == 0) ? g_e8.even_p2_pts[pb] : g_e8.odd_p2_pts[pb];
    out[0] = pt_a[2u * pl];
    out[1] = pt_a[2u * pl + 1u];
    out[2] = pt_b[2u * pr];
    out[3] = pt_b[2u * pr + 1u];
}

static inline void unrank_e8_block_auto(uint64_t rank, int is_ball, int m_bound, int8_t *out) {
    int m;
    uint64_t rem = rank;
    if (is_ball) {
        const uint64_t *w1 = g_e8.w_blk1;
        uint32_t lo = 0, len = (uint32_t)m_bound + 1u;
        while (len > 1u) {
            uint32_t half = len >> 1;
            uint32_t mid = lo + half;
            lo = (w1[mid - 1u] <= rem) ? mid : lo;
            len -= half;
        }
        m = (int)lo;
        if (m > 0) rem -= w1[m - 1];
    } else {
        m = m_bound;
    }

    int coset, n_tot;
    uint64_t p0_tot;
    uint64_t ec8 = g_e8.even_c8[m];
    if (rem < ec8) {
        coset = 0;
        n_tot = 2 * m;
        p0_tot = g_e8.e8_even_p0_tot[m];
    } else {
        coset = 1;
        rem -= ec8;
        n_tot = m - 1;
        p0_tot = g_e8.e8_odd_p0_tot[m];
    }
    int p = (rem < p0_tot) ? 0 : 1;
    if (p) rem -= p0_tot;
    const uint32_t *c4 = (coset == 0) ? g_e8.even_c4[p] : g_e8.odd_c4[p];
    const Recip64 *recs = (coset == 0) ? g_e8.rec_even_c4[p] : g_e8.rec_odd_c4[p];
    int n_l;
    if (g_e8.use_fast_inner) {
        const uint64_t *cp = (coset == 0) ? (g_e8.cp_ec8[p] + g_e8.cp_ec8_off[m])
                                          : (g_e8.cp_oc8[p] + g_e8.cp_oc8_off[n_tot]);
        uint32_t nblk = (coset == 0) ? g_e8.cp_ec8_len[m] : g_e8.cp_oc8_len[n_tot];
        uint32_t blk = find_cp_u64(cp, nblk, rem);
        rem -= cp[blk];
        n_l = (int)(blk * E8_IN_STRIDE);
    } else {
        n_l = 0;
    }
    for (; n_l <= n_tot; ++n_l) {
        uint64_t term = (uint64_t)c4[n_l] * c4[n_tot - n_l];
        if (rem < term) break;
        rem -= term;
    }
    int n_r = n_tot - n_l;
    Recip64 rec = recs[n_r];
    uint64_t d = c4[n_r];
    uint32_t il = (uint32_t)mul_shr_u64_to_u64(rem, rec.m, rec.shift);
    uint32_t ir = (uint32_t)(rem - (uint64_t)il * d);
    if (g_e8.use_fast_inner) {
        unrank_e8_quad_fast(il, coset, n_l, p, out);
        unrank_e8_quad_fast(ir, coset, n_r, p, out + 4);
    } else {
        unrank_e8_quad_compact(il, coset, n_l, p, out);
        unrank_e8_quad_compact(ir, coset, n_r, p, out + 4);
    }
}

/* Explicit AVX2 8-wide SIMD quad unranker (tests vpgatherdd + _mm256_mul_epu32 on real Zen-1 silicon!) */
#if HAS_AVX2
static inline void unrank_e8_quad8_avx2(
    const uint32_t rem_in[8],
    const int coset_in[8],
    const int norm4_in[8],
    const int par4_in[8],
    int8_t *out_ptrs[8]
) {
    uint32_t rem[8], pa[8], pb[8], na[8], nb[8], d_arr[8], m_arr[8], sh_arr[8];
    for (int k = 0; k < 8; ++k) {
        int c = coset_in[k], n4 = norm4_in[k], p4 = par4_in[k];
        uint32_t r = rem_in[k];
        uint32_t p0 = (c == 0) ? g_e8.q_even_p0_tot[p4][n4] : g_e8.q_odd_p0_tot[p4][n4];
        uint32_t a = (r < p0) ? 0u : 1u;
        if (a) r -= p0;
        uint32_t b = (uint32_t)p4 ^ a;
        pa[k] = a; pb[k] = b;
        const uint16_t *ca = (c == 0) ? g_e8.even_c2[a] : g_e8.odd_c2[a];
        const uint16_t *cb = (c == 0) ? g_e8.even_c2[b] : g_e8.odd_c2[b];
        int cur = 0;
        if (g_e8.use_fast_inner) {
            int qid = (int)((a << 1) | b);
            const uint32_t *cp = (c == 0) ? (g_e8.cp_eq[qid] + g_e8.cp_eq_off[n4])
                                          : (g_e8.cp_oq[qid] + g_e8.cp_oq_off[n4]);
            uint32_t blk = find_cp_u32(cp, (c == 0) ? g_e8.cp_eq_len[n4] : g_e8.cp_oq_len[n4], r);
            r -= cp[blk];
            cur = (int)(blk * E8_Q_STRIDE);
        }
        for (; cur <= n4; ++cur) {
            uint32_t term = (uint32_t)ca[cur] * cb[n4 - cur];
            if (r < term) break;
            r -= term;
        }
        na[k] = (uint32_t)cur;
        nb[k] = (uint32_t)(n4 - cur);
        rem[k] = r;
        d_arr[k] = cb[nb[k]];
        Recip32 rec = (c == 0) ? g_e8.rec_even_c2[b][nb[k]] : g_e8.rec_odd_c2[b][nb[k]];
        m_arr[k] = rec.m;
        sh_arr[k] = rec.shift;
    }

    /* 8-lane AVX2 zero-division multiply-shift: il[k] = (uint64(rem[k]) * m[k]) >> sh[k] */
    __m256i v_rem = _mm256_loadu_si256((const __m256i *)rem);
    __m256i v_m   = _mm256_loadu_si256((const __m256i *)m_arr);
    __m256i v_sh  = _mm256_loadu_si256((const __m256i *)sh_arr);
    __m256i v_d   = _mm256_loadu_si256((const __m256i *)d_arr);

    /* Even lanes (0, 2, 4, 6) in 64-bit */
    __m256i prod_ev = _mm256_mul_epu32(v_rem, v_m);
    __m256i sh_ev   = _mm256_and_si256(v_sh, _mm256_set1_epi64x(0xffffffffLL));
    __m256i q_ev    = _mm256_srlv_epi64(prod_ev, sh_ev);

    /* Odd lanes (1, 3, 5, 7) shifted down to even positions */
    __m256i rem_od  = _mm256_srli_epi64(v_rem, 32);
    __m256i m_od    = _mm256_srli_epi64(v_m, 32);
    __m256i sh_od   = _mm256_srli_epi64(v_sh, 32);
    __m256i prod_od = _mm256_mul_epu32(rem_od, m_od);
    __m256i q_od    = _mm256_slli_epi64(_mm256_srlv_epi64(prod_od, sh_od), 32);

    /* Recombine 8 x uint32 quotients and compute remainders in 1 AVX2 mul + sub */
    __m256i v_il = _mm256_blend_epi32(q_ev, q_od, 0xAA);
    __m256i v_ir = _mm256_sub_epi32(v_rem, _mm256_mullo_epi32(v_il, v_d));

    uint32_t il[8], ir[8];
    _mm256_storeu_si256((__m256i *)il, v_il);
    _mm256_storeu_si256((__m256i *)ir, v_ir);

    for (int k = 0; k < 8; ++k) {
        int c = coset_in[k];
        uint32_t a = pa[k], b = pb[k];
        uint32_t pl = (uint32_t)((c == 0) ? g_e8.even_p2_off[a][na[k]] : g_e8.odd_p2_off[a][na[k]]) + il[k];
        uint32_t pr = (uint32_t)((c == 0) ? g_e8.even_p2_off[b][nb[k]] : g_e8.odd_p2_off[b][nb[k]]) + ir[k];
        const int8_t *pt_a = (c == 0) ? g_e8.even_p2_pts[a] : g_e8.odd_p2_pts[a];
        const int8_t *pt_b = (c == 0) ? g_e8.even_p2_pts[b] : g_e8.odd_p2_pts[b];
        int8_t *dst = out_ptrs[k];
        dst[0] = pt_a[2u * pl];
        dst[1] = pt_a[2u * pl + 1u];
        dst[2] = pt_b[2u * pr];
        dst[3] = pt_b[2u * pr + 1u];
    }
}
#endif

static inline size_t unrank_e8_block_32(const uint8_t *in_blk, int8_t *u32) {
    int m_l;
    u128 idx_l, idx_r;
    if (root_split_e8_4(in_blk, &m_l, &idx_l, &idx_r)) {
        memset(u32, 0, 32u);
        return 1u;
    }
    int a0, a1;
    uint64_t l0, r0, l1, r1;
    split_e8_blk2_shell(m_l, idx_l, &a0, &l0, &r0);
    split_e8_blk2_ball(g_e8.m_max - m_l, idx_r, &a1, &l1, &r1);
    unrank_e8_block_auto(l0, 0, a0, u32);
    unrank_e8_block_auto(r0, 0, m_l - a0, u32 + 8);
    unrank_e8_block_auto(l1, 0, a1, u32 + 16);
    unrank_e8_block_auto(r1, 1, (g_e8.m_max - m_l) - a1, u32 + 24);
    return 0u;
}

/* ============================================================================
 * Exported Decode Kernels (0 Hardware Divisions!)
 * ============================================================================ */

size_t e8_decode_std_scalar(const uint8_t *in, size_t groups, int8_t *U) {
    size_t bad = 0;
    const size_t nb = (size_t)g_e8.bytes;
    for (size_t g = 0; g < groups; ++g) {
        bad += unrank_e8_block_32(in + g * nb, U + g * 32u);
    }
    return bad;
}

size_t e8had_decode_wht_scalar(const uint8_t *in, size_t groups, int8_t *Z) {
    size_t bad = 0;
    const size_t nb = (size_t)g_e8.bytes;
    int8_t u32[32];
    for (size_t g = 0; g < groups; ++g) {
        if (unrank_e8_block_32(in + g * nb, u32)) {
            ++bad;
            memset(Z + g * 32u, 0, 32u);
            continue;
        }
        int8_t *z32 = Z + g * 32u;
        for (int b = 0; b < 4; ++b) {
            wht8_u_to_z_scalar(u32 + 8 * b, z32 + 8 * b);
        }
    }
    return bad;
}

size_t e8had_decode_wht_avx2(const uint8_t *in, size_t groups, int8_t *Z) {
    size_t bad = 0;
    const size_t nb = (size_t)g_e8.bytes;
    int8_t u32[32];
    for (size_t g = 0; g < groups; ++g) {
        if (unrank_e8_block_32(in + g * nb, u32)) {
            ++bad;
            memset(Z + g * 32u, 0, 32u);
            continue;
        }
        wht8x4_u_to_z_avx2(u32, Z + g * 32u);
    }
    return bad;
}

/* 8-quad-per-block SIMD AVX2 inner quad unranker + AVX2 WHT_8 (exact same bijection!) */
size_t e8had_decode_simd8_avx2(const uint8_t *in, size_t groups, int8_t *Z) {
#if HAS_AVX2
    size_t bad = 0;
    const size_t nb = (size_t)g_e8.bytes;
    const int M = g_e8.m_max;
    int8_t u32[32];
    for (size_t g = 0; g < groups; ++g) {
        int m_l;
        u128 idx_l, idx_r;
        if (root_split_e8_4(in + g * nb, &m_l, &idx_l, &idx_r)) {
            ++bad;
            memset(Z + g * 32u, 0, 32u);
            continue;
        }
        int a0, a1;
        uint64_t l0, r0, l1, r1;
        split_e8_blk2_shell(m_l, idx_l, &a0, &l0, &r0);
        split_e8_blk2_ball(M - m_l, idx_r, &a1, &l1, &r1);

        uint64_t rks[4] = {l0, r0, l1, r1};
        int is_b[4] = {0, 0, 0, 1};
        int mb[4] = {a0, m_l - a0, a1, (M - m_l) - a1};

        uint32_t q_rem[8];
        int q_coset[8], q_n4[8], q_p4[8];
        int8_t *q_out[8];
        for (int b = 0; b < 4; ++b) {
            int m;
            uint64_t rem = rks[b];
            if (is_b[b]) {
                const uint64_t *w1 = g_e8.w_blk1;
                uint32_t lo = 0, len = (uint32_t)mb[b] + 1u;
                while (len > 1u) {
                    uint32_t half = len >> 1;
                    uint32_t mid = lo + half;
                    lo = (w1[mid - 1u] <= rem) ? mid : lo;
                    len -= half;
                }
                m = (int)lo;
                if (m > 0) rem -= w1[m - 1];
            } else {
                m = mb[b];
            }
            int coset, n_tot;
            uint64_t p0_tot;
            uint64_t ec8 = g_e8.even_c8[m];
            if (rem < ec8) {
                coset = 0; n_tot = 2 * m; p0_tot = g_e8.e8_even_p0_tot[m];
            } else {
                coset = 1; rem -= ec8; n_tot = m - 1; p0_tot = g_e8.e8_odd_p0_tot[m];
            }
            int p = (rem < p0_tot) ? 0 : 1;
            if (p) rem -= p0_tot;
            const uint32_t *c4 = (coset == 0) ? g_e8.even_c4[p] : g_e8.odd_c4[p];
            const Recip64 *recs = (coset == 0) ? g_e8.rec_even_c4[p] : g_e8.rec_odd_c4[p];
            int n_l = 0;
            if (g_e8.use_fast_inner) {
                const uint64_t *cp = (coset == 0) ? (g_e8.cp_ec8[p] + g_e8.cp_ec8_off[m])
                                                  : (g_e8.cp_oc8[p] + g_e8.cp_oc8_off[n_tot]);
                uint32_t blk = find_cp_u64(cp, (coset == 0) ? g_e8.cp_ec8_len[m] : g_e8.cp_oc8_len[n_tot], rem);
                rem -= cp[blk];
                n_l = (int)(blk * E8_IN_STRIDE);
            }
            for (; n_l <= n_tot; ++n_l) {
                uint64_t term = (uint64_t)c4[n_l] * c4[n_tot - n_l];
                if (rem < term) break;
                rem -= term;
            }
            int n_r = n_tot - n_l;
            Recip64 rec = recs[n_r];
            uint32_t il = (uint32_t)mul_shr_u64_to_u64(rem, rec.m, rec.shift);
            uint32_t ir = (uint32_t)(rem - (uint64_t)il * c4[n_r]);
            q_rem[2 * b] = il;     q_coset[2 * b] = coset;     q_n4[2 * b] = n_l; q_p4[2 * b] = p; q_out[2 * b] = u32 + 8 * b;
            q_rem[2 * b + 1] = ir; q_coset[2 * b + 1] = coset; q_n4[2 * b + 1] = n_r; q_p4[2 * b + 1] = p; q_out[2 * b + 1] = u32 + 8 * b + 4;
        }
        /* Unrank all 8 quads of this 32-weight block in one AVX2 8-lane multiply-shift! */
        unrank_e8_quad8_avx2(q_rem, q_coset, q_n4, q_p4, q_out);
        wht8x4_u_to_z_avx2(u32, Z + g * 32u);
    }
    return bad;
#else
    return e8had_decode_wht_scalar(in, groups, Z);
#endif
}

/* ============================================================================
 * Fused Dequant-GEMV Kernels (y[0..rows-1] = W * x, 0 Hardware Divisions!)
 * Packed record layout per 32-weight block: [1B scale_idx | nb bytes rank]
 * ============================================================================ */

/* Mode A: Materialized E_8_Had GEMV (unrank u -> AVX2 WHT_8 -> z -> AVX2/FMA dot(z, x) * scale/sqrt(2)) */
size_t e8had_gemv_materialized_avx2(
    const uint8_t *records,
    const float *scale_lut,
    const float *x,
    size_t rows,
    size_t cols,
    float *y_out
) {
    size_t bad = 0;
    const size_t nb = (size_t)g_e8.bytes;
    const size_t rec_sz = nb + 1u;
    const size_t blk_per_row = cols >> 5;
    const float inv_sqrt2 = 0.7071067811865475244f;
    int8_t u32[32], z32[32];

    for (size_t r = 0; r < rows; ++r) {
        const uint8_t *row_ptr = records + r * blk_per_row * rec_sz;
        float acc = 0.0f;
        for (size_t b = 0; b < blk_per_row; ++b) {
            const uint8_t *rec = row_ptr + b * rec_sz;
            float s = scale_lut[rec[0]] * inv_sqrt2;
            if (unrank_e8_block_32(rec + 1u, u32)) {
                ++bad;
                continue;
            }
            wht8x4_u_to_z_avx2(u32, z32);
            acc += s * dot_i8_32_f32_avx2(z32, x + (b << 5));
        }
        y_out[r] = acc;
    }
    return bad;
}

/* Mode B: Adjoint E_8_Had GEMV (pre-rotate x -> x_tilde ONCE per GEMV, 0 per-block WHT_8 ops!) */
/* Exact identity: <(H_8*u)/(4*sqrt(2)), x> == <u/2, (H_8/sqrt(8))*x> == 0.5 * <u, x_tilde>! */
size_t e8had_gemv_adjoint_avx2(
    const uint8_t *records,
    const float *scale_lut,
    const float *x,
    size_t rows,
    size_t cols,
    float *y_out
) {
    float *x_tilde = (float *)malloc(cols * sizeof(float));
    if (!x_tilde) return rows;
    wht8_f32_vector(x, x_tilde, cols);

    size_t bad = 0;
    const size_t nb = (size_t)g_e8.bytes;
    const size_t rec_sz = nb + 1u;
    const size_t blk_per_row = cols >> 5;
    int8_t u32[32];

    for (size_t r = 0; r < rows; ++r) {
        const uint8_t *row_ptr = records + r * blk_per_row * rec_sz;
        float acc = 0.0f;
        for (size_t b = 0; b < blk_per_row; ++b) {
            const uint8_t *rec = row_ptr + b * rec_sz;
            float s = scale_lut[rec[0]] * 0.5f;
            if (unrank_e8_block_32(rec + 1u, u32)) {
                ++bad;
                continue;
            }
            acc += s * dot_i8_32_f32_avx2(u32, x_tilde + (b << 5));
        }
        y_out[r] = acc;
    }
    free(x_tilde);
    return bad;
}

/* Mode C: 4-Block Interleaved Adjoint E_8_Had GEMV (hides tree-search latency across 4 blocks!) */
size_t e8had_gemv_adjoint_batch4_avx2(
    const uint8_t *records,
    const float *scale_lut,
    const float *x,
    size_t rows,
    size_t cols,
    float *y_out
) {
    float *x_tilde = (float *)malloc(cols * sizeof(float));
    if (!x_tilde) return rows;
    wht8_f32_vector(x, x_tilde, cols);

    size_t bad = 0;
    const int M = g_e8.m_max;
    const size_t nb = (size_t)g_e8.bytes;
    const size_t rec_sz = nb + 1u;
    const size_t blk_per_row = cols >> 5;
    int8_t u32[4][32];

    for (size_t r = 0; r < rows; ++r) {
        const uint8_t *row_ptr = records + r * blk_per_row * rec_sz;
        float acc = 0.0f;
        size_t b = 0;
        for (; b + 4u <= blk_per_row; b += 4u) {
            int m_l[4], ok[4];
            u128 idx_l[4], idx_r[4];
            float sc[4];
            for (int k = 0; k < 4; ++k) {
                const uint8_t *rec = row_ptr + (b + (size_t)k) * rec_sz;
                sc[k] = scale_lut[rec[0]] * 0.5f;
                ok[k] = !root_split_e8_4(rec + 1u, &m_l[k], &idx_l[k], &idx_r[k]);
                if (!ok[k]) ++bad;
            }
            int a0[4], a1[4];
            uint64_t l0[4], r0[4], l1[4], r1[4];
            for (int k = 0; k < 4; ++k) {
                if (!ok[k]) continue;
                split_e8_blk2_shell(m_l[k], idx_l[k], &a0[k], &l0[k], &r0[k]);
                split_e8_blk2_ball(M - m_l[k], idx_r[k], &a1[k], &l1[k], &r1[k]);
            }
            for (int k = 0; k < 4; ++k) {
                if (!ok[k]) continue;
                unrank_e8_block_auto(l0[k], 0, a0[k], u32[k]);
                unrank_e8_block_auto(r0[k], 0, m_l[k] - a0[k], u32[k] + 8);
                unrank_e8_block_auto(l1[k], 0, a1[k], u32[k] + 16);
                unrank_e8_block_auto(r1[k], 1, (M - m_l[k]) - a1[k], u32[k] + 24);
                acc += sc[k] * dot_i8_32_f32_avx2(u32[k], x_tilde + ((b + (size_t)k) << 5));
            }
        }
        for (; b < blk_per_row; ++b) {
            const uint8_t *rec = row_ptr + b * rec_sz;
            float s = scale_lut[rec[0]] * 0.5f;
            if (unrank_e8_block_32(rec + 1u, u32[0])) {
                ++bad;
                continue;
            }
            acc += s * dot_i8_32_f32_avx2(u32[0], x_tilde + (b << 5));
        }
        y_out[r] = acc;
    }
    free(x_tilde);
    return bad;
}

/* ============================================================================
 * Matched 4.500-bpw Q4_K_M-Style Linear Affine AVX2/FMA GEMV Baseline
 * Layout per 32 weights: 18 bytes = [1B scale_idx | 1B zero_pt | 16B packed 4-bit nibbles]
 * Identical 2,359,296-byte matrix footprint for 4096 x 1024 as EBG32:b136s8!
 * ============================================================================ */

size_t q4km_style_gemv_avx2(
    const uint8_t *records,
    const float *scale_lut,
    const float *x,
    size_t rows,
    size_t cols,
    float *y_out
) {
    const size_t blk_per_row = cols >> 5;
    const size_t rec_sz = 18u;
#if HAS_AVX2
    const __m128i mask_lo = _mm_set1_epi8(0x0f);
    const __m128i eight = _mm_set1_epi8(8);
    for (size_t r = 0; r < rows; ++r) {
        const uint8_t *row_ptr = records + r * blk_per_row * rec_sz;
        __m256 row_acc = _mm256_setzero_ps();
        for (size_t b = 0; b < blk_per_row; ++b) {
            const uint8_t *rec = row_ptr + b * rec_sz;
            __m256 v_sc = _mm256_set1_ps(scale_lut[rec[0]]);
            __m128i packed = _mm_loadu_si128((const __m128i *)(rec + 2u));
            __m128i lo_n = _mm_sub_epi8(_mm_and_si128(packed, mask_lo), eight);
            __m128i hi_n = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(packed, 4), mask_lo), eight);
            const float *xp = x + (b << 5);
            __m256 f0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo_n));
            __m256 f1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo_n, 8)));
            __m256 f2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi_n));
            __m256 f3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi_n, 8)));
            __m256 blk_acc = _mm256_mul_ps(f0, _mm256_loadu_ps(xp + 0));
            blk_acc = _mm256_fmadd_ps(f1, _mm256_loadu_ps(xp + 8), blk_acc);
            blk_acc = _mm256_fmadd_ps(f2, _mm256_loadu_ps(xp + 16), blk_acc);
            blk_acc = _mm256_fmadd_ps(f3, _mm256_loadu_ps(xp + 24), blk_acc);
            row_acc = _mm256_fmadd_ps(blk_acc, v_sc, row_acc);
        }
        y_out[r] = hsum256_ps(row_acc);
    }
#else
    for (size_t r = 0; r < rows; ++r) {
        const uint8_t *row_ptr = records + r * blk_per_row * rec_sz;
        float acc = 0.0f;
        for (size_t b = 0; b < blk_per_row; ++b) {
            const uint8_t *rec = row_ptr + b * rec_sz;
            float s = scale_lut[rec[0]];
            const float *xp = x + (b << 5);
            float bacc = 0.0f;
            for (int i = 0; i < 16; ++i) {
                int q0 = (int)(rec[2 + i] & 0x0fu) - 8;
                int q1 = (int)(rec[2 + i] >> 4) - 8;
                bacc += (float)q0 * xp[i] + (float)q1 * xp[16 + i];
            }
            acc += s * bacc;
        }
        y_out[r] = acc;
    }
#endif
    return 0u;
}

/* ============================================================================
 * Multi-Core (pthread) Dual-Clock Benchmark Harness & Stage Breakdown
 * ============================================================================ */

typedef size_t (*decode_fn_t)(const uint8_t *, size_t, int8_t *);

typedef struct {
    decode_fn_t fn;
    const uint8_t *in;
    size_t groups;
    int8_t *out;
    size_t bad;
    uint64_t cpu_ns;
} DecWorker;

static void *dec_worker_entry(void *arg) {
    DecWorker *w = (DecWorker *)arg;
    uint64_t c0 = thread_cpu_ns();
    w->bad = w->fn(w->in, w->groups, w->out);
    w->cpu_ns = thread_cpu_ns() - c0;
    return NULL;
}

/* decode_mode: 0=e8_std_scalar, 1=e8had_wht_scalar, 2=e8had_wht_avx2, 3=e8had_simd8_avx2 */
uint64_t bench_decode_ns(
    int decode_mode,
    int n_threads,
    const uint8_t *in,
    size_t groups,
    int8_t *out,
    size_t *bad_out,
    uint64_t *cpu_ns_out
) {
    decode_fn_t fn = NULL;
    if (decode_mode == 0) fn = e8_decode_std_scalar;
    else if (decode_mode == 1) fn = e8had_decode_wht_scalar;
    else if (decode_mode == 2) fn = e8had_decode_wht_avx2;
    else if (decode_mode == 3) fn = e8had_decode_simd8_avx2;
    else { *bad_out = groups; if (cpu_ns_out) *cpu_ns_out = 0; return 0; }

    size_t nb = (size_t)g_e8.bytes;
    if (n_threads <= 1) {
        uint64_t w0 = now_ns(), c0 = thread_cpu_ns();
        *bad_out = fn(in, groups, out);
        uint64_t c1 = thread_cpu_ns(), w1 = now_ns();
        if (cpu_ns_out) *cpu_ns_out = c1 - c0;
        return w1 - w0;
    }
    if (n_threads > 8) n_threads = 8;
    pthread_t th[8];
    DecWorker wa[8];
    size_t chunk = groups / (size_t)n_threads;
    size_t rem = groups % (size_t)n_threads;
    size_t off = 0;
    uint64_t t0 = now_ns();
    for (int i = 0; i < n_threads; ++i) {
        size_t cnt = chunk + ((size_t)i < rem ? 1u : 0u);
        wa[i].fn = fn;
        wa[i].in = in + off * nb;
        wa[i].groups = cnt;
        wa[i].out = out + off * 32u;
        wa[i].bad = 0;
        wa[i].cpu_ns = 0;
        off += cnt;
        pthread_create(&th[i], NULL, dec_worker_entry, &wa[i]);
    }
    size_t tb = 0; uint64_t tc = 0;
    for (int i = 0; i < n_threads; ++i) {
        pthread_join(th[i], NULL);
        tb += wa[i].bad; tc += wa[i].cpu_ns;
    }
    uint64_t wall = now_ns() - t0;
    *bad_out = tb;
    if (cpu_ns_out) *cpu_ns_out = tc;
    return wall;
}

typedef size_t (*gemv_fn_t)(const uint8_t *, const float *, const float *, size_t, size_t, float *);

typedef struct {
    gemv_fn_t fn;
    const uint8_t *records;
    const float *scale_lut;
    const float *x;
    size_t rows;
    size_t cols;
    float *y_out;
    size_t bad;
    uint64_t cpu_ns;
} GemvWorker;

static void *gemv_worker_entry(void *arg) {
    GemvWorker *w = (GemvWorker *)arg;
    uint64_t c0 = thread_cpu_ns();
    w->bad = w->fn(w->records, w->scale_lut, w->x, w->rows, w->cols, w->y_out);
    w->cpu_ns = thread_cpu_ns() - c0;
    return NULL;
}

/* gemv_mode: 0=e8had_materialized_avx2, 1=e8had_adjoint_avx2, 2=e8had_adjoint_batch4_avx2, 3=q4km_style_avx2 */
uint64_t bench_gemv_ns(
    int gemv_mode,
    int n_threads,
    const uint8_t *records,
    size_t rec_sz,
    const float *scale_lut,
    const float *x,
    size_t rows,
    size_t cols,
    float *y_out,
    size_t *bad_out,
    uint64_t *cpu_ns_out
) {
    gemv_fn_t fn = NULL;
    if (gemv_mode == 0) fn = e8had_gemv_materialized_avx2;
    else if (gemv_mode == 1) fn = e8had_gemv_adjoint_avx2;
    else if (gemv_mode == 2) fn = e8had_gemv_adjoint_batch4_avx2;
    else if (gemv_mode == 3) fn = q4km_style_gemv_avx2;
    else { *bad_out = rows; if (cpu_ns_out) *cpu_ns_out = 0; return 0; }

    if (n_threads <= 1) {
        uint64_t w0 = now_ns(), c0 = thread_cpu_ns();
        *bad_out = fn(records, scale_lut, x, rows, cols, y_out);
        uint64_t c1 = thread_cpu_ns(), w1 = now_ns();
        if (cpu_ns_out) *cpu_ns_out = c1 - c0;
        return w1 - w0;
    }
    if (n_threads > 8) n_threads = 8;
    pthread_t th[8];
    GemvWorker wa[8];
    size_t row_bytes = (cols >> 5) * rec_sz;
    size_t chunk = rows / (size_t)n_threads;
    size_t rem = rows % (size_t)n_threads;
    size_t off = 0;
    uint64_t t0 = now_ns();
    for (int i = 0; i < n_threads; ++i) {
        size_t cnt = chunk + ((size_t)i < rem ? 1u : 0u);
        wa[i].fn = fn;
        wa[i].records = records + off * row_bytes;
        wa[i].scale_lut = scale_lut;
        wa[i].x = x;
        wa[i].rows = cnt;
        wa[i].cols = cols;
        wa[i].y_out = y_out + off;
        wa[i].bad = 0;
        wa[i].cpu_ns = 0;
        off += cnt;
        pthread_create(&th[i], NULL, gemv_worker_entry, &wa[i]);
    }
    size_t tb = 0; uint64_t tc = 0;
    for (int i = 0; i < n_threads; ++i) {
        pthread_join(th[i], NULL);
        tb += wa[i].bad; tc += wa[i].cpu_ns;
    }
    uint64_t wall = now_ns() - t0;
    *bad_out = tb;
    if (cpu_ns_out) *cpu_ns_out = tc;
    return wall;
}

/* Fine-grained 4-stage CPU-time breakdown (ns over `groups` blocks):
 *   stage_ns[0]: outer tree splits (root_split_e8_4 + 2 x split_e8_blk2)
 *   stage_ns[1]: inner 4 x E_8 block unrank (8->4->2->LUT)
 *   stage_ns[2]: materialized AVX2 4 x WHT_8 transform
 *   stage_ns[3]: AVX2/FMA 32-weight dot-product
 */
size_t bench_stage_breakdown_ns(
    const uint8_t *in_ranks,
    size_t groups,
    const float *x32,
    uint64_t stage_ns[4],
    float *checksum_out
) {
    const size_t nb = (size_t)g_e8.bytes;
    const int M = g_e8.m_max;
    uint64_t *blk_rk = (uint64_t *)malloc(groups * 4u * sizeof(uint64_t));
    int16_t *blk_m = (int16_t *)malloc(groups * 4u * sizeof(int16_t));
    int8_t *u_buf = (int8_t *)malloc(groups * 32u);
    int8_t *z_buf = (int8_t *)malloc(groups * 32u);
    if (!blk_rk || !blk_m || !u_buf || !z_buf) {
        free(blk_rk); free(blk_m); free(u_buf); free(z_buf);
        return groups;
    }

    size_t bad = 0;
    uint64_t t0 = thread_cpu_ns();
    for (size_t g = 0; g < groups; ++g) {
        int m_l;
        u128 idx_l, idx_r;
        if (root_split_e8_4(in_ranks + g * nb, &m_l, &idx_l, &idx_r)) {
            ++bad;
            blk_rk[4 * g + 0] = blk_rk[4 * g + 1] = blk_rk[4 * g + 2] = blk_rk[4 * g + 3] = 0;
            blk_m[4 * g + 0] = blk_m[4 * g + 1] = blk_m[4 * g + 2] = blk_m[4 * g + 3] = 0;
            continue;
        }
        int a0, a1;
        uint64_t l0, r0, l1, r1;
        split_e8_blk2_shell(m_l, idx_l, &a0, &l0, &r0);
        split_e8_blk2_ball(M - m_l, idx_r, &a1, &l1, &r1);
        blk_rk[4 * g + 0] = l0; blk_m[4 * g + 0] = (int16_t)a0;
        blk_rk[4 * g + 1] = r0; blk_m[4 * g + 1] = (int16_t)(m_l - a0);
        blk_rk[4 * g + 2] = l1; blk_m[4 * g + 2] = (int16_t)a1;
        blk_rk[4 * g + 3] = r1; blk_m[4 * g + 3] = (int16_t)((M - m_l) - a1);
    }
    uint64_t t1 = thread_cpu_ns();
    for (size_t g = 0; g < groups; ++g) {
        int8_t *dst = u_buf + g * 32u;
        unrank_e8_block_auto(blk_rk[4 * g + 0], 0, blk_m[4 * g + 0], dst);
        unrank_e8_block_auto(blk_rk[4 * g + 1], 0, blk_m[4 * g + 1], dst + 8);
        unrank_e8_block_auto(blk_rk[4 * g + 2], 0, blk_m[4 * g + 2], dst + 16);
        unrank_e8_block_auto(blk_rk[4 * g + 3], 1, blk_m[4 * g + 3], dst + 24);
    }
    uint64_t t2 = thread_cpu_ns();
    for (size_t g = 0; g < groups; ++g) {
        wht8x4_u_to_z_avx2(u_buf + g * 32u, z_buf + g * 32u);
    }
    uint64_t t3 = thread_cpu_ns();
    float chk = 0.0f;
    for (size_t g = 0; g < groups; ++g) {
        chk += dot_i8_32_f32_avx2(z_buf + g * 32u, x32);
    }
    uint64_t t4 = thread_cpu_ns();

    stage_ns[0] = t1 - t0;
    stage_ns[1] = t2 - t1;
    stage_ns[2] = t3 - t2;
    stage_ns[3] = t4 - t3;
    if (checksum_out) *checksum_out = chk;

    free(blk_rk); free(blk_m); free(u_buf); free(z_buf);
    return bad;
}
