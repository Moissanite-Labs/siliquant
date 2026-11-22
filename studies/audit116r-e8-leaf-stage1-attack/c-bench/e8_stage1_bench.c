/* Track S15d (AUDIT-116): Attacking the 68%-75% Stage-1 Inner E_8 Leaf Bottleneck
 * in Pure C11 + AVX2/FMA (Zero Hardware Divisions on Every Hot Path).
 *
 * Implements & benchmarks head-to-head on EBG32:b88s8 (m_max=48) & EBG32:b136s8 (m_max=391):
 *   0. baseline          : AUDIT-115 reference (strided-16 checkpoint / linear 8->4->2)
 *   1. (A) guide_narrow  : Pure uint32_t (88b) / uint64_t (136b) + O(1) top-bit CDF-prefix
 *                          bucket guide tables (ZERO binary search, uint16 c4, uint8 c2)
 *   2. (B) eytz_lut2     : Branchless Eytzinger cmov search + flat int8[8] LUT for m <= 2 (19.2 KB L1d)
 *   3. (B) eytz_lut3     : Branchless Eytzinger cmov search + flat int8[8] LUT for m <= 3 (73.0 KB L2)
 *   4. (C1) d8_compact   : D_8^+ signed-leader / permutation-orbit + 502B binomial-mask unranker
 *   5. (C2) d8_pshufb    : D_8^+ signed-leader / permutation-orbit + 22-shape pshufb/psignb LUT
 *   + Matched 4.500-bpw Q4_K_M-style AVX2/FMA GEMV baseline.
 *
 * Build (0 warnings under gcc & clang):
 *   cc -O3 -march=native -Wall -Wextra -Werror -std=c11 -shared -fPIC -pthread \
 *      -o libe8_stage1.so e8_stage1_bench.c -lm
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

typedef struct {
    uint64_t cdf;
    int8_t vals[8];
} D8Leader16;

int e8s1_has_avx2(void) {
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
 * Zero-Division Multiply-Shift & Search Primitives
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

/* Branchless Eytzinger search: returns 1-indexed node ans of largest a[j] <= rem */
static inline uint32_t eytz_search_u64(const uint64_t *e, uint32_t n, uint64_t rem) {
    uint32_t k = 1u;
    while (k <= n) {
        k = (k << 1) + (uint32_t)(e[k] <= rem);
    }
    return k >> ((unsigned)__builtin_ctz(k) + 1u);
}

static inline uint32_t eytz_search_u32(const uint32_t *e, uint32_t n, uint32_t rem) {
    uint32_t k = 1u;
    while (k <= n) {
        k = (k << 1) + (uint32_t)(e[k] <= rem);
    }
    return k >> ((unsigned)__builtin_ctz(k) + 1u);
}

static inline uint32_t eytz_search_u16(const uint16_t *e, uint32_t n, uint16_t rem) {
    uint32_t k = 1u;
    while (k <= n) {
        k = (k << 1) + (uint32_t)(e[k] <= rem);
    }
    return k >> ((unsigned)__builtin_ctz(k) + 1u);
}

static uint32_t eytz_fill_u64(const uint64_t *src, uint32_t n, uint64_t *dst, uint16_t *idx_dst, uint32_t k, uint32_t i) {
    if (k <= n) {
        i = eytz_fill_u64(src, n, dst, idx_dst, k << 1, i);
        dst[k] = src[i];
        idx_dst[k] = (uint16_t)i;
        ++i;
        i = eytz_fill_u64(src, n, dst, idx_dst, (k << 1) + 1u, i);
    }
    return i;
}

static uint32_t eytz_fill_u32(const uint32_t *src, uint32_t n, uint32_t *dst, uint16_t *idx_dst, uint32_t k, uint32_t i) {
    if (k <= n) {
        i = eytz_fill_u32(src, n, dst, idx_dst, k << 1, i);
        dst[k] = src[i];
        idx_dst[k] = (uint16_t)i;
        ++i;
        i = eytz_fill_u32(src, n, dst, idx_dst, (k << 1) + 1u, i);
    }
    return i;
}

static uint32_t eytz_fill_u16(const uint16_t *src, uint32_t n, uint16_t *dst, uint16_t *idx_dst, uint32_t k, uint32_t i) {
    if (k <= n) {
        i = eytz_fill_u16(src, n, dst, idx_dst, k << 1, i);
        dst[k] = src[i];
        idx_dst[k] = (uint16_t)i;
        ++i;
        i = eytz_fill_u16(src, n, dst, idx_dst, (k << 1) + 1u, i);
    }
    return i;
}

static inline uint32_t bit_length_u64(uint64_t v) {
    return (v == 0u) ? 0u : (64u - (uint32_t)__builtin_clzll(v));
}

static inline uint32_t bit_length_u32(uint32_t v) {
    return (v == 0u) ? 0u : (32u - (uint32_t)__builtin_clz(v));
}

static Recip32 make_recip32(uint32_t d, uint32_t max_div) {
    Recip32 out = {1u, 0u};
    if (d <= 1u) return out;
    if ((d & (d - 1u)) == 0u) {
        out.m = 1u;
        out.shift = (uint32_t)__builtin_ctz(d);
        return out;
    }
    uint32_t k = bit_length_u32(max_div);
    if (k < 1u) k = 1u;
    for (uint32_t extra = 0u; extra <= 32u; ++extra) {
        uint32_t sh = k + extra;
        if (sh >= 64u) break;
        uint64_t two_s = 1ULL << sh;
        uint64_t mult = (two_s + (uint64_t)d - 1ULL) / (uint64_t)d;
        if (mult < (1ULL << 32) && mult * (uint64_t)d >= two_s &&
            (mult * (uint64_t)d - two_s) * (uint64_t)max_div < two_s) {
            out.m = (uint32_t)mult;
            out.shift = sh;
            return out;
        }
    }
    return out;
}

/* ============================================================================
 * Context: Baseline + (A) Guide Narrow + (B) Eytzinger+LUT + (C) D8+ Leaders
 * ============================================================================ */

#define E8_STRIDE     16
#define E8_IN_STRIDE  16
#define E8_Q_STRIDE   16
#define E8_EYTZ_STR8   8
#define G8_KBITS_88    5
#define G8_NB_88      32
#define G4_KBITS_88    4
#define G4_NB_88      16
#define G8_KBITS_136   6
#define G8_NB_136     64
#define G4_KBITS_136   5
#define G4_NB_136     32
#define D8_G_KBITS     5
#define D8_G_NB       32

typedef struct {
    int m_max;
    int r_max;
    int t_max;
    int bytes;
    int use_u32; /* 1 for m_max <= 64 (88b), 0 for 136b */
    u192 total_count;
    u192 *c_root;
    u128 *s_blk2;
    u128 *w_blk2;
    uint64_t *s_blk1;
    uint64_t *w_blk1;
    uint32_t *w_blk1_u32;
    Recip192 *rec_root;
    Recip128 *rec_b1;
    Recip128 *rec_s1;

    /* Base inner DP tables (shared) */
    uint64_t *even_c8;
    uint32_t *even_c8_u32;
    uint32_t *even_c4[2];
    uint16_t *even_c4_u16[2];
    uint16_t *even_c2[2];
    uint8_t *even_c2_u8[2];
    uint16_t *even_p2_off[2];
    int8_t *even_p2_pts[2];
    uint16_t *even_p2_u16[2];
    uint32_t *odd_c4[2];
    uint16_t *odd_c4_u16[2];
    uint16_t *odd_c2[2];
    uint8_t *odd_c2_u8[2];
    uint16_t *odd_p2_off[2];
    int8_t *odd_p2_pts[2];
    uint16_t *odd_p2_u16[2];
    Recip64 *rec_even_c4[2];
    Recip32 *rec_even_c4_u32[2];
    Recip32 *rec_even_c2[2];
    Recip64 *rec_odd_c4[2];
    Recip32 *rec_odd_c4_u32[2];
    Recip32 *rec_odd_c2[2];

    /* Baseline checkpoints */
    uint32_t *cp_m_off;
    uint32_t *cp_m_len;
    u128 *cp_s_blk2;
    u128 *cp_w_blk2;
    uint64_t *e8_even_p0_tot;
    uint64_t *e8_odd_p0_tot;
    uint32_t *e8_even_p0_u32;
    uint32_t *e8_odd_p0_u32;
    uint32_t *cp_ec8_off;
    uint32_t *cp_ec8_len;
    uint64_t *cp_ec8[2];
    uint32_t *cp_oc8_off;
    uint32_t *cp_oc8_len;
    uint64_t *cp_oc8[2];
    uint32_t *q_even_p0_tot[2];
    uint32_t *q_odd_p0_tot[2];
    uint16_t *q_even_p0_u16[2];
    uint16_t *q_odd_p0_u16[2];
    uint32_t *cp_eq_off;
    uint32_t *cp_eq_len;
    uint32_t *cp_eq[4];
    uint32_t *cp_oq_off;
    uint32_t *cp_oq_len;
    uint32_t *cp_oq[4];

    /* Accelerator (A): O(1) Top-Bit CDF-Prefix Bucket Guide */
    /* 88b flat CDF + tiny uint8 guide index (ZERO multiply in search!) */
    uint32_t *f88_ec8_off;
    uint32_t *f88_oc8_off;
    uint32_t *f88_ec8_cdf[2];
    uint32_t *f88_oc8_cdf[2];
    uint8_t *g88_ec8_sh[2];
    uint8_t *g88_oc8_sh[2];
    uint8_t *g88_ec8_idx[2];
    uint8_t *g88_oc8_idx[2];
    uint32_t *f88_eq_off;
    uint32_t *f88_oq_off;
    uint16_t *f88_eq_cdf[4];
    uint16_t *f88_oq_cdf[4];
    uint8_t *g88_eq_sh[4];
    uint8_t *g88_oq_sh[4];
    uint8_t *g88_eq_idx[4];
    uint8_t *g88_oq_idx[4];

    /* 136b compact O(1) guide (64 buckets for 8->4, 32 buckets for 4->2) */
    uint8_t *g136_ec8_sh[2];
    uint8_t *g136_oc8_sh[2];
    uint16_t *g136_ec8_idx[2];
    uint64_t *g136_ec8_cdf[2];
    uint16_t *g136_oc8_idx[2];
    uint64_t *g136_oc8_cdf[2];
    uint8_t *g136_eq_sh[4];
    uint8_t *g136_oq_sh[4];
    uint32_t *g136_eq_pack[4]; /* (cdf16 << 16) | na16 */
    uint32_t *g136_oq_pack[4];

    /* Accelerator (B): Flat int8[8] LUT (m <= 3, 9,121 vecs = 72,968 B) + Branchless Eytzinger */
    int8_t *flat_lut;
    uint32_t flat_lut_off[5];
    /* 88b: full Eytzinger CDF (ZERO tail scan!) */
    uint32_t *ez88_ec8_off;
    uint32_t *ez88_oc8_off;
    uint32_t *ez88_ec8_val[2];
    uint16_t *ez88_ec8_a[2];
    uint32_t *ez88_oc8_val[2];
    uint16_t *ez88_oc8_a[2];
    uint32_t *ez88_eq_off;
    uint32_t *ez88_oq_off;
    uint16_t *ez88_eq_val[4];
    uint16_t *ez88_eq_a[4];
    uint16_t *ez88_oq_val[4];
    uint16_t *ez88_oq_a[4];
    /* 136b: stride-8 Eytzinger checkpoints (L2/L3-friendly) */
    uint32_t *ez136_ec8_off;
    uint32_t *ez136_ec8_len;
    uint64_t *ez136_ec8_val[2];
    uint16_t *ez136_ec8_a[2];
    uint32_t *ez136_oc8_off;
    uint32_t *ez136_oc8_len;
    uint64_t *ez136_oc8_val[2];
    uint16_t *ez136_oc8_a[2];
    uint32_t *ez136_eq_off;
    uint32_t *ez136_eq_len;
    uint16_t *ez136_eq_val[4];
    uint16_t *ez136_eq_a[4];
    uint32_t *ez136_oq_off;
    uint32_t *ez136_oq_len;
    uint16_t *ez136_oq_val[4];
    uint16_t *ez136_oq_a[4];

    /* Accelerator (C): D_8^+ Signed-Leader / Permutation-Orbit */
    uint32_t d8_total_leaders;
    D8Leader16 *d8_leaders;
    uint8_t *d8_meta;       /* (coset << 7) | shape_id (0..21) */
    uint8_t *d8_sbits;      /* 0..8 */
    uint32_t *d8_shell_off; /* length m_max + 2 */
    uint8_t *d8_guide_sh;   /* length m_max + 1 */
    uint32_t *d8_guide_idx; /* (m_max + 1) * D8_G_NB */
    /* Universal 22-shape pshufb table (95,503 x 8 = 764,024 B) + 510B binomial mask table */
    uint32_t shape_perm_off[23];
    uint8_t shape_k[22];
    uint8_t shape_runs[22][8];
    uint8_t *shape_pshufb; /* 95,503 * 8 bytes */
    uint8_t comb_masks[9][9][70];
    uint8_t comb_cnt[9][9];
    Recip32 comb_recip[71];
    int8_t sign8_lut[256][8]; /* for single-instruction _mm_sign_epi8 */

    /* Memory footprints (bytes) reported to Python */
    size_t mem_bytes_baseline;
    size_t mem_bytes_guide_a;
    size_t mem_bytes_eytz_b2;
    size_t mem_bytes_eytz_b3;
    size_t mem_bytes_d8_c1;
    size_t mem_bytes_d8_c2;
} E8S1Ctx;

static E8S1Ctx g_e8;
static int g_e8_ready = 0;

/* ============================================================================
 * D_8^+ Universal 22-Shape & Shell Leader Builders (Pure C11, Run Once at Init)
 * ============================================================================ */

static const uint32_t FACT8[9] = {1u, 1u, 2u, 6u, 24u, 120u, 720u, 5040u, 40320u};

static void build_comb_table(void) {
    memset(g_e8.comb_cnt, 0, sizeof(g_e8.comb_cnt));
    for (int n = 1; n <= 8; ++n) {
        for (uint32_t mask = 0u; mask < (1u << n); ++mask) {
            int c = __builtin_popcount(mask);
            if (c >= 1 && c <= n) {
                uint8_t idx = g_e8.comb_cnt[n][c]++;
                g_e8.comb_masks[n][c][idx] = (uint8_t)mask;
            }
        }
    }
    for (uint32_t d = 1u; d <= 70u; ++d) {
        g_e8.comb_recip[d] = make_recip32(d, 40320u);
    }
    for (int s = 0; s < 256; ++s) {
        for (int i = 0; i < 8; ++i) {
            g_e8.sign8_lut[s][i] = ((s >> i) & 1) ? (int8_t)-1 : (int8_t)1;
        }
    }
}

/* Generate the 22 canonical integer partitions of 8 (sorted descending) and all 95,503 pshufb perms */
static int g_shape_cnt_tmp = 0;
static void gen_shapes_rec(int rem, int max_v, uint8_t cur[8], int depth) {
    if (rem == 0) {
        int id = g_shape_cnt_tmp++;
        int d = (depth <= 8) ? depth : 8;
        g_e8.shape_k[id] = (uint8_t)d;
        for (int i = 0; i < d; ++i) g_e8.shape_runs[id][i] = cur[i];
        return;
    }
    if (depth >= 8) return;
    int hi = (rem < max_v) ? rem : max_v;
    for (int v = hi; v >= 1; --v) {
        cur[depth] = (uint8_t)v;
        gen_shapes_rec(rem - v, v, cur, depth + 1);
    }
}

static uint8_t find_shape_id_from_runs(const uint8_t runs[8], int k) {
    /* Sort run counts descending into canonical partition of 8 */
    uint8_t s[8];
    for (int i = 0; i < k; ++i) s[i] = runs[i];
    for (int i = 0; i < k; ++i) {
        for (int j = i + 1; j < k; ++j) {
            if (s[j] > s[i]) {
                uint8_t tmp = s[i]; s[i] = s[j]; s[j] = tmp;
            }
        }
    }
    for (int id = 0; id < 22; ++id) {
        if (g_e8.shape_k[id] != (uint8_t)k) continue;
        int eq = 1;
        for (int i = 0; i < k; ++i) {
            if (g_e8.shape_runs[id][i] != s[i]) { eq = 0; break; }
        }
        if (eq) return (uint8_t)id;
    }
    return 0u;
}

static void build_universal_pshufb_table(void) {
    g_shape_cnt_tmp = 0;
    uint8_t cur[8];
    gen_shapes_rec(8, 8, cur, 0);
    uint32_t off = 0u;
    for (int id = 0; id < 22; ++id) {
        g_e8.shape_perm_off[id] = off;
        uint32_t denom = 1u;
        for (int j = 0; j < g_e8.shape_k[id]; ++j) denom *= FACT8[g_e8.shape_runs[id][j]];
        off += FACT8[8] / denom;
    }
    g_e8.shape_perm_off[22] = off; /* == 95,503 */
    g_e8.shape_pshufb = (uint8_t *)malloc((size_t)off * 8u);

    /* Populate each shape's pshufb index map: dst[slot] = src[pshufb[slot]] */
    for (int id = 0; id < 22; ++id) {
        int k = g_e8.shape_k[id];
        uint32_t nperm = g_e8.shape_perm_off[id + 1] - g_e8.shape_perm_off[id];
        uint8_t *base = g_e8.shape_pshufb + (size_t)g_e8.shape_perm_off[id] * 8u;
        /* Canonical source indices for each run j start at run_src[j] */
        uint8_t run_src[8];
        uint8_t acc = 0u;
        for (int j = 0; j < k; ++j) {
            run_src[j] = acc;
            acc = (uint8_t)(acc + g_e8.shape_runs[id][j]);
        }
        for (uint32_t rk = 0u; rk < nperm; ++rk) {
            uint8_t *dst = base + (size_t)rk * 8u;
            uint8_t avail[8] = {0, 1, 2, 3, 4, 5, 6, 7};
            int n_rem = 8;
            uint32_t r = rk;
            for (int j = 0; j < k - 1; ++j) {
                int c = g_e8.shape_runs[id][j];
                uint32_t cnt = g_e8.comb_cnt[n_rem][c];
                uint32_t q = r / cnt;
                uint32_t ci = r - q * cnt;
                r = q;
                uint8_t mask = g_e8.comb_masks[n_rem][c][ci];
                uint8_t next_avail[8];
                int nn = 0;
                uint8_t src_pos = run_src[j];
                for (int p = 0; p < n_rem; ++p) {
                    if ((mask >> p) & 1u) {
                        dst[avail[p]] = src_pos++;
                    } else {
                        next_avail[nn++] = avail[p];
                    }
                }
                for (int p = 0; p < nn; ++p) avail[p] = next_avail[p];
                n_rem = nn;
            }
            uint8_t src_pos = run_src[k - 1];
            for (int p = 0; p < n_rem; ++p) dst[avail[p]] = src_pos++;
        }
    }
}

/* Reorder leader vals[8] so runs appear in descending multiplicity order matching canonical shape_id! */
static void canonicalize_leader_by_shape(int8_t vals[8], uint8_t *shape_id_out) {
    int8_t rv[8];
    uint8_t rc[8];
    int k = 0;
    for (int i = 0; i < 8; ++i) {
        if (k > 0 && rv[k - 1] == vals[i]) {
            rc[k - 1]++;
        } else {
            rv[k] = vals[i];
            rc[k] = 1u;
            ++k;
        }
    }
    /* Sort runs by count descending (tie-break by value descending) */
    for (int i = 0; i < k; ++i) {
        for (int j = i + 1; j < k; ++j) {
            if (rc[j] > rc[i] || (rc[j] == rc[i] && rv[j] > rv[i])) {
                uint8_t tc = rc[i]; rc[i] = rc[j]; rc[j] = tc;
                int8_t tv = rv[i]; rv[i] = rv[j]; rv[j] = tv;
            }
        }
    }
    int pos = 0;
    for (int j = 0; j < k; ++j) {
        for (int t = 0; t < rc[j]; ++t) vals[pos++] = rv[j];
    }
    *shape_id_out = find_shape_id_from_runs(rc, k);
}

/* Two-pass fast C11 generator of ALL D_8^+ leaders across m = 0..m_max */
typedef struct {
    int m_max;
    int max_ev;
    int max_od;
    int pass;
    uint32_t *counts;
    uint32_t *cursor;
    D8Leader16 *leaders;
    uint8_t *meta;
    uint8_t *sbits;
    int8_t cur[8];
} D8Gen;

static void d8_emit(D8Gen *g, int m, int coset, int nz, const int8_t raw[8], uint32_t prod_f) {
    if (g->pass == 0) {
        g->counts[m]++;
        return;
    }
    uint32_t slot = g->cursor[m]++;
    int8_t canon[8];
    memcpy(canon, raw, 8u);
    uint8_t sid = 0u;
    canonicalize_leader_by_shape(canon, &sid);
    memcpy(g->leaders[slot].vals, canon, 8u);
    uint32_t pcnt = FACT8[8] / prod_f;
    uint8_t sb = (coset == 0) ? (uint8_t)nz : 7u;
    g->leaders[slot].cdf = (uint64_t)pcnt << sb; /* temporarily store orbit size; prefix-summed next */
    g->meta[slot] = (uint8_t)((coset << 7) | sid);
    g->sbits[slot] = sb;
}

static void d8_gen_ev(D8Gen *g, int idx, int max_v, int rem_sq, int par, int nz, int run_v, int run_c, uint32_t prod_f) {
    if (idx == 8) {
        if ((rem_sq & 1) == 0 && par == 0) {
            int m = (g->max_ev - rem_sq) >> 1;
            d8_emit(g, m, 0, nz, g->cur, prod_f * FACT8[run_c]);
        }
        return;
    }
    int rem_slots = 8 - idx;
    if ((rem_sq & 1) == 0 && par == 0) {
        int m = (g->max_ev - rem_sq) >> 1;
        for (int j = idx; j < 8; ++j) g->cur[j] = 0;
        uint32_t pf = (run_v == 0) ? (prod_f * FACT8[run_c + rem_slots]) : (prod_f * FACT8[run_c] * FACT8[rem_slots]);
        d8_emit(g, m, 0, nz, g->cur, pf);
    }
    int hi = (int)sqrt((double)rem_sq);
    while ((hi + 1) * (hi + 1) <= rem_sq) ++hi;
    while (hi * hi > rem_sq) --hi;
    if (max_v < hi) hi = max_v;
    for (int v = 1; v <= hi; ++v) {
        g->cur[idx] = (int8_t)(2 * v);
        if (v == run_v) {
            d8_gen_ev(g, idx + 1, v, rem_sq - v * v, par ^ (v & 1), nz + 1, v, run_c + 1, prod_f);
        } else {
            d8_gen_ev(g, idx + 1, v, rem_sq - v * v, par ^ (v & 1), nz + 1, v, 1, prod_f * FACT8[run_c]);
        }
    }
}

static void d8_gen_od(D8Gen *g, int idx, int max_v, int rem_sq, int run_v, int run_c, uint32_t prod_f) {
    if (idx == 8) {
        int used = g->max_od - rem_sq;
        if ((used & 7) == 0) {
            int m = used >> 3;
            d8_emit(g, m, 1, 8, g->cur, prod_f * FACT8[run_c]);
        }
        return;
    }
    int rem_slots = 8 - idx;
    int avail_sq = rem_sq - (rem_slots - 1);
    if (avail_sq < 1) return;
    int hi = (int)sqrt((double)avail_sq);
    while ((hi + 1) * (hi + 1) <= avail_sq) ++hi;
    while (hi * hi > avail_sq) --hi;
    if (max_v < hi) hi = max_v;
    if ((hi & 1) == 0) --hi;
    for (int v = 1; v <= hi; v += 2) {
        g->cur[idx] = (int8_t)v;
        if (v == run_v) {
            d8_gen_od(g, idx + 1, v, rem_sq - v * v, v, run_c + 1, prod_f);
        } else {
            d8_gen_od(g, idx + 1, v, rem_sq - v * v, v, 1, prod_f * FACT8[run_c]);
        }
    }
}

static void build_d8plus_tables(int m_max) {
    size_t M1 = (size_t)m_max + 1u;
    uint32_t *counts = (uint32_t *)calloc(M1, sizeof(uint32_t));
    uint32_t *cursor = (uint32_t *)calloc(M1, sizeof(uint32_t));
    D8Gen gen;
    memset(&gen, 0, sizeof(gen));
    gen.m_max = m_max;
    gen.max_ev = 2 * m_max;
    gen.max_od = 8 * m_max;
    gen.counts = counts;
    gen.cursor = cursor;

    int hi_ev = (int)sqrt((double)gen.max_ev) + 1;
    int hi_od = (int)sqrt((double)gen.max_od) + 1;
    if ((hi_od & 1) == 0) --hi_od;

    gen.pass = 0;
    d8_gen_ev(&gen, 0, hi_ev, gen.max_ev, 0, 0, -1, 0, 1u);
    d8_gen_od(&gen, 0, hi_od, gen.max_od, -1, 0, 1u);

    g_e8.d8_shell_off = (uint32_t *)malloc((M1 + 1u) * sizeof(uint32_t));
    uint32_t tot = 0u;
    for (int m = 0; m <= m_max; ++m) {
        g_e8.d8_shell_off[m] = tot;
        cursor[m] = tot;
        tot += counts[m];
    }
    g_e8.d8_shell_off[m_max + 1] = tot;
    g_e8.d8_total_leaders = tot;

    g_e8.d8_leaders = (D8Leader16 *)malloc((size_t)tot * sizeof(D8Leader16));
    g_e8.d8_meta = (uint8_t *)malloc((size_t)tot * sizeof(uint8_t));
    g_e8.d8_sbits = (uint8_t *)malloc((size_t)tot * sizeof(uint8_t));
    gen.leaders = g_e8.d8_leaders;
    gen.meta = g_e8.d8_meta;
    gen.sbits = g_e8.d8_sbits;

    gen.pass = 1;
    d8_gen_ev(&gen, 0, hi_ev, gen.max_ev, 0, 0, -1, 0, 1u);
    d8_gen_od(&gen, 0, hi_od, gen.max_od, -1, 0, 1u);

    /* Prefix-sum orbit sizes into CDFs and build O(1) 32-bucket guide per shell */
    g_e8.d8_guide_sh = (uint8_t *)malloc(M1 * sizeof(uint8_t));
    g_e8.d8_guide_idx = (uint32_t *)malloc(M1 * D8_G_NB * sizeof(uint32_t));
    for (int m = 0; m <= m_max; ++m) {
        uint32_t off = g_e8.d8_shell_off[m];
        uint32_t end = g_e8.d8_shell_off[m + 1];
        uint32_t len = end - off;
        uint64_t acc = 0u;
        for (uint32_t i = off; i < end; ++i) {
            uint64_t orb = g_e8.d8_leaders[i].cdf;
            g_e8.d8_leaders[i].cdf = acc;
            acc += orb;
        }
        uint32_t tb = bit_length_u64(acc > 0u ? acc - 1u : 0u);
        uint8_t sh = (tb > D8_G_KBITS) ? (uint8_t)(tb - D8_G_KBITS) : 0u;
        g_e8.d8_guide_sh[m] = sh;
        uint32_t *gbase = g_e8.d8_guide_idx + (size_t)m * D8_G_NB;
        uint32_t cur_i = 0u;
        for (uint32_t b = 0u; b < D8_G_NB; ++b) {
            uint64_t lo_r = (uint64_t)b << sh;
            if (lo_r >= acc) {
                gbase[b] = len - 1u;
                continue;
            }
            while (cur_i + 1u < len && g_e8.d8_leaders[off + cur_i + 1u].cdf <= lo_r) {
                ++cur_i;
            }
            gbase[b] = cur_i;
        }
    }

    free(counts);
    free(cursor);
}

/* ============================================================================
 * Context Free & Init
 * ============================================================================ */

static void e8s1_free(void) {
    if (!g_e8_ready) return;
    free(g_e8.c_root);
    free(g_e8.s_blk2);
    free(g_e8.w_blk2);
    free(g_e8.s_blk1);
    free(g_e8.w_blk1);
    free(g_e8.w_blk1_u32);
    free(g_e8.rec_root);
    free(g_e8.rec_b1);
    free(g_e8.rec_s1);
    free(g_e8.even_c8);
    free(g_e8.even_c8_u32);
    for (int p = 0; p < 2; ++p) {
        free(g_e8.even_c4[p]);
        free(g_e8.even_c4_u16[p]);
        free(g_e8.even_c2[p]);
        free(g_e8.even_c2_u8[p]);
        free(g_e8.even_p2_off[p]);
        free(g_e8.even_p2_pts[p]);
        free(g_e8.even_p2_u16[p]);
        free(g_e8.odd_c4[p]);
        free(g_e8.odd_c4_u16[p]);
        free(g_e8.odd_c2[p]);
        free(g_e8.odd_c2_u8[p]);
        free(g_e8.odd_p2_off[p]);
        free(g_e8.odd_p2_pts[p]);
        free(g_e8.odd_p2_u16[p]);
        free(g_e8.rec_even_c4[p]);
        free(g_e8.rec_even_c4_u32[p]);
        free(g_e8.rec_even_c2[p]);
        free(g_e8.rec_odd_c4[p]);
        free(g_e8.rec_odd_c4_u32[p]);
        free(g_e8.rec_odd_c2[p]);
        free(g_e8.cp_ec8[p]);
        free(g_e8.cp_oc8[p]);
        free(g_e8.q_even_p0_tot[p]);
        free(g_e8.q_odd_p0_tot[p]);
        free(g_e8.q_even_p0_u16[p]);
        free(g_e8.q_odd_p0_u16[p]);
        free(g_e8.f88_ec8_cdf[p]);
        free(g_e8.f88_oc8_cdf[p]);
        free(g_e8.g88_ec8_sh[p]);
        free(g_e8.g88_oc8_sh[p]);
        free(g_e8.g88_ec8_idx[p]);
        free(g_e8.g88_oc8_idx[p]);
        free(g_e8.g136_ec8_sh[p]);
        free(g_e8.g136_oc8_sh[p]);
        free(g_e8.g136_ec8_idx[p]);
        free(g_e8.g136_ec8_cdf[p]);
        free(g_e8.g136_oc8_idx[p]);
        free(g_e8.g136_oc8_cdf[p]);
        free(g_e8.ez88_ec8_val[p]);
        free(g_e8.ez88_ec8_a[p]);
        free(g_e8.ez88_oc8_val[p]);
        free(g_e8.ez88_oc8_a[p]);
        free(g_e8.ez136_ec8_val[p]);
        free(g_e8.ez136_ec8_a[p]);
        free(g_e8.ez136_oc8_val[p]);
        free(g_e8.ez136_oc8_a[p]);
    }
    for (int q = 0; q < 4; ++q) {
        free(g_e8.cp_eq[q]);
        free(g_e8.cp_oq[q]);
        free(g_e8.f88_eq_cdf[q]);
        free(g_e8.f88_oq_cdf[q]);
        free(g_e8.g88_eq_sh[q]);
        free(g_e8.g88_oq_sh[q]);
        free(g_e8.g88_eq_idx[q]);
        free(g_e8.g88_oq_idx[q]);
        free(g_e8.g136_eq_sh[q]);
        free(g_e8.g136_oq_sh[q]);
        free(g_e8.g136_eq_pack[q]);
        free(g_e8.g136_oq_pack[q]);
        free(g_e8.ez88_eq_val[q]);
        free(g_e8.ez88_eq_a[q]);
        free(g_e8.ez88_oq_val[q]);
        free(g_e8.ez88_oq_a[q]);
        free(g_e8.ez136_eq_val[q]);
        free(g_e8.ez136_eq_a[q]);
        free(g_e8.ez136_oq_val[q]);
        free(g_e8.ez136_oq_a[q]);
    }
    free(g_e8.cp_m_off);
    free(g_e8.cp_m_len);
    free(g_e8.cp_s_blk2);
    free(g_e8.cp_w_blk2);
    free(g_e8.e8_even_p0_tot);
    free(g_e8.e8_odd_p0_tot);
    free(g_e8.e8_even_p0_u32);
    free(g_e8.e8_odd_p0_u32);
    free(g_e8.cp_ec8_off);
    free(g_e8.cp_ec8_len);
    free(g_e8.cp_oc8_off);
    free(g_e8.cp_oc8_len);
    free(g_e8.cp_eq_off);
    free(g_e8.cp_eq_len);
    free(g_e8.cp_oq_off);
    free(g_e8.cp_oq_len);
    free(g_e8.f88_ec8_off);
    free(g_e8.f88_oc8_off);
    free(g_e8.f88_eq_off);
    free(g_e8.f88_oq_off);
    free(g_e8.flat_lut);
    free(g_e8.ez88_ec8_off);
    free(g_e8.ez88_oc8_off);
    free(g_e8.ez88_eq_off);
    free(g_e8.ez88_oq_off);
    free(g_e8.ez136_ec8_off);
    free(g_e8.ez136_ec8_len);
    free(g_e8.ez136_oc8_off);
    free(g_e8.ez136_oc8_len);
    free(g_e8.ez136_eq_off);
    free(g_e8.ez136_eq_len);
    free(g_e8.ez136_oq_off);
    free(g_e8.ez136_oq_len);
    free(g_e8.d8_leaders);
    free(g_e8.d8_meta);
    free(g_e8.d8_sbits);
    free(g_e8.d8_shell_off);
    free(g_e8.d8_guide_sh);
    free(g_e8.d8_guide_idx);
    free(g_e8.shape_pshufb);
    memset(&g_e8, 0, sizeof(g_e8));
    g_e8_ready = 0;
}

/* Forward declaration for populating exact BallPack flat LUT (m = 0..3) */
static inline void unrank_e8_baseline(uint64_t rank, int is_ball, int m_bound, int8_t *out);

int e8s1_init(
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
    if (m_max < 3 || m_max > 2048 || bytes < 1 || bytes > 24) return 1;
    e8s1_free();
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
    g_e8.use_u32 = (m_max <= 64) ? 1 : 0;

    g_e8.c_root = (u192 *)malloc(M2 * sizeof(u192));
    g_e8.s_blk2 = (u128 *)malloc(M1 * sizeof(u128));
    g_e8.w_blk2 = (u128 *)malloc(M1 * sizeof(u128));
    g_e8.s_blk1 = (uint64_t *)malloc(M1 * sizeof(uint64_t));
    g_e8.w_blk1 = (uint64_t *)malloc(M1 * sizeof(uint64_t));
    g_e8.w_blk1_u32 = (uint32_t *)malloc(M1 * sizeof(uint32_t));
    g_e8.rec_root = (Recip192 *)calloc(M1, sizeof(Recip192));
    g_e8.rec_b1 = (Recip128 *)calloc(M1, sizeof(Recip128));
    g_e8.rec_s1 = (Recip128 *)calloc(M1, sizeof(Recip128));
    g_e8.even_c8 = (uint64_t *)malloc(M1 * sizeof(uint64_t));
    g_e8.even_c8_u32 = (uint32_t *)malloc(M1 * sizeof(uint32_t));

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
        g_e8.w_blk1_u32[i] = (uint32_t)w_blk1_in[i];
        g_e8.even_c8[i] = even_c8_in[i];
        g_e8.even_c8_u32[i] = (uint32_t)even_c8_in[i];
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
        g_e8.even_c4_u16[p] = (uint16_t *)malloc(R1 * sizeof(uint16_t));
        g_e8.even_c2[p] = (uint16_t *)malloc(R1 * sizeof(uint16_t));
        g_e8.even_c2_u8[p] = (uint8_t *)malloc(R1 * sizeof(uint8_t));
        g_e8.even_p2_off[p] = (uint16_t *)malloc(R2 * sizeof(uint16_t));
        g_e8.even_p2_pts[p] = (int8_t *)malloc((size_t)even_ns[p] * 2u);
        g_e8.even_p2_u16[p] = (uint16_t *)malloc((size_t)even_ns[p] * sizeof(uint16_t));
        memcpy(g_e8.even_c4[p], even_c4_in + (size_t)p * R1, R1 * sizeof(uint32_t));
        memcpy(g_e8.even_c2[p], even_c2_in + (size_t)p * R1, R1 * sizeof(uint16_t));
        memcpy(g_e8.even_p2_off[p], even_off_in + (size_t)p * R2, R2 * sizeof(uint16_t));
        memcpy(g_e8.even_p2_pts[p], even_pts[p], (size_t)even_ns[p] * 2u);
        memcpy(g_e8.even_p2_u16[p], even_pts[p], (size_t)even_ns[p] * 2u);
        for (size_t i = 0; i < R1; ++i) {
            g_e8.even_c4_u16[p][i] = (uint16_t)g_e8.even_c4[p][i];
            g_e8.even_c2_u8[p][i] = (uint8_t)g_e8.even_c2[p][i];
        }

        g_e8.odd_c4[p] = (uint32_t *)malloc(T1 * sizeof(uint32_t));
        g_e8.odd_c4_u16[p] = (uint16_t *)malloc(T1 * sizeof(uint16_t));
        g_e8.odd_c2[p] = (uint16_t *)malloc(T1 * sizeof(uint16_t));
        g_e8.odd_c2_u8[p] = (uint8_t *)malloc(T1 * sizeof(uint8_t));
        g_e8.odd_p2_off[p] = (uint16_t *)malloc(T2 * sizeof(uint16_t));
        g_e8.odd_p2_pts[p] = (int8_t *)malloc((size_t)odd_ns[p] * 2u);
        g_e8.odd_p2_u16[p] = (uint16_t *)malloc((size_t)odd_ns[p] * sizeof(uint16_t));
        memcpy(g_e8.odd_c4[p], odd_c4_in + (size_t)p * T1, T1 * sizeof(uint32_t));
        memcpy(g_e8.odd_c2[p], odd_c2_in + (size_t)p * T1, T1 * sizeof(uint16_t));
        memcpy(g_e8.odd_p2_off[p], odd_off_in + (size_t)p * T2, T2 * sizeof(uint16_t));
        memcpy(g_e8.odd_p2_pts[p], odd_pts[p], (size_t)odd_ns[p] * 2u);
        memcpy(g_e8.odd_p2_u16[p], odd_pts[p], (size_t)odd_ns[p] * 2u);
        for (size_t i = 0; i < T1; ++i) {
            g_e8.odd_c4_u16[p][i] = (uint16_t)g_e8.odd_c4[p][i];
            g_e8.odd_c2_u8[p][i] = (uint8_t)g_e8.odd_c2[p][i];
        }

        g_e8.rec_even_c4[p] = (Recip64 *)calloc(R1, sizeof(Recip64));
        g_e8.rec_even_c4_u32[p] = (Recip32 *)calloc(R1, sizeof(Recip32));
        g_e8.rec_even_c2[p] = (Recip32 *)calloc(R1, sizeof(Recip32));
        for (size_t i = 0; i < R1; ++i) {
            g_e8.rec_even_c4[p][i].m = rec_ec4_m[(size_t)p * R1 + i];
            g_e8.rec_even_c4[p][i].shift = rec_ec4_s[(size_t)p * R1 + i];
            g_e8.rec_even_c2[p][i].m = rec_ec2_m[(size_t)p * R1 + i];
            g_e8.rec_even_c2[p][i].shift = rec_ec2_s[(size_t)p * R1 + i];
            if (g_e8.use_u32 && g_e8.even_c4[p][i] > 0u) {
                uint32_t d = g_e8.even_c4[p][i];
                uint32_t max_l = 0u;
                for (size_t j = 0; j + i < R1; ++j) {
                    if (g_e8.even_c4[p][j] > max_l) max_l = g_e8.even_c4[p][j];
                }
                g_e8.rec_even_c4_u32[p][i] = make_recip32(d, max_l * d - 1u);
            }
        }
        g_e8.rec_odd_c4[p] = (Recip64 *)calloc(T1, sizeof(Recip64));
        g_e8.rec_odd_c4_u32[p] = (Recip32 *)calloc(T1, sizeof(Recip32));
        g_e8.rec_odd_c2[p] = (Recip32 *)calloc(T1, sizeof(Recip32));
        for (size_t i = 0; i < T1; ++i) {
            g_e8.rec_odd_c4[p][i].m = rec_oc4_m[(size_t)p * T1 + i];
            g_e8.rec_odd_c4[p][i].shift = rec_oc4_s[(size_t)p * T1 + i];
            g_e8.rec_odd_c2[p][i].m = rec_oc2_m[(size_t)p * T1 + i];
            g_e8.rec_odd_c2[p][i].shift = rec_oc2_s[(size_t)p * T1 + i];
            if (g_e8.use_u32 && g_e8.odd_c4[p][i] > 0u) {
                uint32_t d = g_e8.odd_c4[p][i];
                uint32_t max_l = 0u;
                for (size_t j = 0; j + i < T1; ++j) {
                    if (g_e8.odd_c4[p][j] > max_l) max_l = g_e8.odd_c4[p][j];
                }
                g_e8.rec_odd_c4_u32[p][i] = make_recip32(d, max_l * d - 1u);
            }
        }
    }

    /* Outer checkpoints */
    g_e8.cp_m_off = (uint32_t *)malloc(M1 * sizeof(uint32_t));
    g_e8.cp_m_len = (uint32_t *)malloc(M1 * sizeof(uint32_t));
    uint32_t total_cp = 0u;
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

    /* Inner baseline checkpoints + parity totals */
    g_e8.e8_even_p0_tot = (uint64_t *)calloc(M1, sizeof(uint64_t));
    g_e8.e8_odd_p0_tot = (uint64_t *)calloc(M1, sizeof(uint64_t));
    g_e8.e8_even_p0_u32 = (uint32_t *)calloc(M1, sizeof(uint32_t));
    g_e8.e8_odd_p0_u32 = (uint32_t *)calloc(M1, sizeof(uint32_t));
    g_e8.cp_ec8_off = (uint32_t *)malloc(M1 * sizeof(uint32_t));
    g_e8.cp_ec8_len = (uint32_t *)malloc(M1 * sizeof(uint32_t));
    uint32_t tot_ec8 = 0u;
    for (int m = 0; m <= m_max; ++m) {
        uint32_t nblk = (uint32_t)((2 * m) / E8_IN_STRIDE) + 1u;
        g_e8.cp_ec8_off[m] = tot_ec8;
        g_e8.cp_ec8_len[m] = nblk;
        tot_ec8 += nblk;
    }
    g_e8.cp_oc8_off = (uint32_t *)malloc(T1 * sizeof(uint32_t));
    g_e8.cp_oc8_len = (uint32_t *)malloc(T1 * sizeof(uint32_t));
    uint32_t tot_oc8 = 0u;
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
            uint64_t acc = 0u;
            for (int a = 0; a <= r8; ++a) {
                if ((a & (E8_IN_STRIDE - 1)) == 0) {
                    g_e8.cp_ec8[p][base + (uint32_t)(a / E8_IN_STRIDE)] = acc;
                }
                acc += (uint64_t)g_e8.even_c4[p][a] * g_e8.even_c4[p][r8 - a];
            }
            if (p == 0) {
                g_e8.e8_even_p0_tot[m] = acc;
                g_e8.e8_even_p0_u32[m] = (uint32_t)acc;
            }
        }
        if (m >= 1) {
            int t8 = m - 1;
            uint32_t obase = g_e8.cp_oc8_off[t8];
            for (int p = 0; p < 2; ++p) {
                uint64_t acc = 0u;
                for (int a = 0; a <= t8; ++a) {
                    if ((a & (E8_IN_STRIDE - 1)) == 0) {
                        g_e8.cp_oc8[p][obase + (uint32_t)(a / E8_IN_STRIDE)] = acc;
                    }
                    acc += (uint64_t)g_e8.odd_c4[p][a] * g_e8.odd_c4[p][t8 - a];
                }
                if (p == 0) {
                    g_e8.e8_odd_p0_tot[m] = acc;
                    g_e8.e8_odd_p0_u32[m] = (uint32_t)acc;
                }
            }
        }
    }

    g_e8.cp_eq_off = (uint32_t *)malloc(R1 * sizeof(uint32_t));
    g_e8.cp_eq_len = (uint32_t *)malloc(R1 * sizeof(uint32_t));
    uint32_t tot_eq = 0u;
    for (int r = 0; r <= r_max; ++r) {
        uint32_t nblk = (uint32_t)(r / E8_Q_STRIDE) + 1u;
        g_e8.cp_eq_off[r] = tot_eq;
        g_e8.cp_eq_len[r] = nblk;
        tot_eq += nblk;
    }
    g_e8.cp_oq_off = (uint32_t *)malloc(T1 * sizeof(uint32_t));
    g_e8.cp_oq_len = (uint32_t *)malloc(T1 * sizeof(uint32_t));
    uint32_t tot_oq = 0u;
    for (int t = 0; t <= t_max; ++t) {
        uint32_t nblk = (uint32_t)(t / E8_Q_STRIDE) + 1u;
        g_e8.cp_oq_off[t] = tot_oq;
        g_e8.cp_oq_len[t] = nblk;
        tot_oq += nblk;
    }
    for (int p = 0; p < 2; ++p) {
        g_e8.q_even_p0_tot[p] = (uint32_t *)calloc(R1, sizeof(uint32_t));
        g_e8.q_odd_p0_tot[p] = (uint32_t *)calloc(T1, sizeof(uint32_t));
        g_e8.q_even_p0_u16[p] = (uint16_t *)calloc(R1, sizeof(uint16_t));
        g_e8.q_odd_p0_u16[p] = (uint16_t *)calloc(T1, sizeof(uint16_t));
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
                uint32_t acc = 0u;
                for (int na = 0; na <= r; ++na) {
                    if ((na & (E8_Q_STRIDE - 1)) == 0) {
                        g_e8.cp_eq[qid][base + (uint32_t)(na / E8_Q_STRIDE)] = acc;
                    }
                    acc += (uint32_t)g_e8.even_c2[pa][na] * g_e8.even_c2[pb][r - na];
                }
                if (pa == 0) {
                    g_e8.q_even_p0_tot[pb][r] = acc;
                    g_e8.q_even_p0_u16[pb][r] = (uint16_t)acc;
                }
            }
            for (int t = 0; t <= t_max; ++t) {
                uint32_t base = g_e8.cp_oq_off[t];
                uint32_t acc = 0u;
                for (int na = 0; na <= t; ++na) {
                    if ((na & (E8_Q_STRIDE - 1)) == 0) {
                        g_e8.cp_oq[qid][base + (uint32_t)(na / E8_Q_STRIDE)] = acc;
                    }
                    acc += (uint32_t)g_e8.odd_c2[pa][na] * g_e8.odd_c2[pb][t - na];
                }
                if (pa == 0) {
                    g_e8.q_odd_p0_tot[pb][t] = acc;
                    g_e8.q_odd_p0_u16[pb][t] = (uint16_t)acc;
                }
            }
        }
    }

    /* ========================================================================
     * Build Accelerator (A) & (B) Tables (Exact Branch-Specialized for 88b / 136b)
     * ======================================================================== */
    if (g_e8.use_u32) {
        /* 88b: Flat uint32 8->4 CDF + uint8 32-bucket guide + full Eytzinger */
        g_e8.f88_ec8_off = (uint32_t *)malloc(M1 * sizeof(uint32_t));
        g_e8.ez88_ec8_off = (uint32_t *)malloc(M1 * sizeof(uint32_t));
        uint32_t f_ec = 0u, e_ec = 0u;
        for (int m = 0; m <= m_max; ++m) {
            g_e8.f88_ec8_off[m] = f_ec;
            f_ec += (uint32_t)(2 * m + 2);
            g_e8.ez88_ec8_off[m] = e_ec;
            e_ec += (uint32_t)(2 * m + 2); /* 1-indexed Eytzinger of size 2*m + 1 */
        }
        g_e8.f88_oc8_off = (uint32_t *)malloc(T1 * sizeof(uint32_t));
        g_e8.ez88_oc8_off = (uint32_t *)malloc(T1 * sizeof(uint32_t));
        uint32_t f_oc = 0u, e_oc = 0u;
        for (int t = 0; t <= t_max; ++t) {
            g_e8.f88_oc8_off[t] = f_oc;
            f_oc += (uint32_t)(t + 2);
            g_e8.ez88_oc8_off[t] = e_oc;
            e_oc += (uint32_t)(t + 2);
        }
        for (int p = 0; p < 2; ++p) {
            g_e8.f88_ec8_cdf[p] = (uint32_t *)malloc((size_t)f_ec * sizeof(uint32_t));
            g_e8.f88_oc8_cdf[p] = (uint32_t *)malloc((size_t)f_oc * sizeof(uint32_t));
            g_e8.g88_ec8_sh[p] = (uint8_t *)calloc(M1, sizeof(uint8_t));
            g_e8.g88_oc8_sh[p] = (uint8_t *)calloc(T1, sizeof(uint8_t));
            g_e8.g88_ec8_idx[p] = (uint8_t *)calloc(M1 * G8_NB_88, sizeof(uint8_t));
            g_e8.g88_oc8_idx[p] = (uint8_t *)calloc(T1 * G8_NB_88, sizeof(uint8_t));
            g_e8.ez88_ec8_val[p] = (uint32_t *)calloc((size_t)e_ec, sizeof(uint32_t));
            g_e8.ez88_ec8_a[p] = (uint16_t *)calloc((size_t)e_ec, sizeof(uint16_t));
            g_e8.ez88_oc8_val[p] = (uint32_t *)calloc((size_t)e_oc, sizeof(uint32_t));
            g_e8.ez88_oc8_a[p] = (uint16_t *)calloc((size_t)e_oc, sizeof(uint16_t));

            for (int m = 0; m <= m_max; ++m) {
                int r8 = 2 * m;
                uint32_t *cdf = g_e8.f88_ec8_cdf[p] + g_e8.f88_ec8_off[m];
                uint32_t acc = 0u;
                for (int a = 0; a <= r8; ++a) {
                    cdf[a] = acc;
                    acc += g_e8.even_c4[p][a] * g_e8.even_c4[p][r8 - a];
                }
                cdf[r8 + 1] = UINT32_MAX;
                uint32_t tb = bit_length_u32(acc > 0u ? acc - 1u : 0u);
                uint8_t sh = (tb > G8_KBITS_88) ? (uint8_t)(tb - G8_KBITS_88) : 0u;
                g_e8.g88_ec8_sh[p][m] = sh;
                uint8_t *gi = g_e8.g88_ec8_idx[p] + (size_t)m * G8_NB_88;
                int a_cur = 0;
                for (uint32_t b = 0u; b < G8_NB_88; ++b) {
                    uint32_t lo_r = b << sh;
                    if (acc == 0u || lo_r >= acc) { gi[b] = (uint8_t)r8; continue; }
                    while (a_cur < r8 && cdf[a_cur + 1] <= lo_r) ++a_cur;
                    gi[b] = (uint8_t)a_cur;
                }
                eytz_fill_u32(cdf, (uint32_t)(r8 + 1),
                              g_e8.ez88_ec8_val[p] + g_e8.ez88_ec8_off[m],
                              g_e8.ez88_ec8_a[p] + g_e8.ez88_ec8_off[m], 1u, 0u);
            }
            for (int t = 0; t <= t_max; ++t) {
                uint32_t *cdf = g_e8.f88_oc8_cdf[p] + g_e8.f88_oc8_off[t];
                uint32_t acc = 0u;
                for (int a = 0; a <= t; ++a) {
                    cdf[a] = acc;
                    acc += g_e8.odd_c4[p][a] * g_e8.odd_c4[p][t - a];
                }
                cdf[t + 1] = UINT32_MAX;
                uint32_t tb = bit_length_u32(acc > 0u ? acc - 1u : 0u);
                uint8_t sh = (tb > G8_KBITS_88) ? (uint8_t)(tb - G8_KBITS_88) : 0u;
                g_e8.g88_oc8_sh[p][t] = sh;
                uint8_t *gi = g_e8.g88_oc8_idx[p] + (size_t)t * G8_NB_88;
                int a_cur = 0;
                for (uint32_t b = 0u; b < G8_NB_88; ++b) {
                    uint32_t lo_r = b << sh;
                    if (acc == 0u || lo_r >= acc) { gi[b] = (uint8_t)t; continue; }
                    while (a_cur < t && cdf[a_cur + 1] <= lo_r) ++a_cur;
                    gi[b] = (uint8_t)a_cur;
                }
                eytz_fill_u32(cdf, (uint32_t)(t + 1),
                              g_e8.ez88_oc8_val[p] + g_e8.ez88_oc8_off[t],
                              g_e8.ez88_oc8_a[p] + g_e8.ez88_oc8_off[t], 1u, 0u);
            }
        }

        /* 88b: Flat uint16 4->2 CDF + uint8 16-bucket guide + full Eytzinger */
        g_e8.f88_eq_off = (uint32_t *)malloc(R1 * sizeof(uint32_t));
        g_e8.ez88_eq_off = (uint32_t *)malloc(R1 * sizeof(uint32_t));
        uint32_t f_eq = 0u;
        for (int r = 0; r <= r_max; ++r) {
            g_e8.f88_eq_off[r] = f_eq;
            g_e8.ez88_eq_off[r] = f_eq;
            f_eq += (uint32_t)(r + 2);
        }
        g_e8.f88_oq_off = (uint32_t *)malloc(T1 * sizeof(uint32_t));
        g_e8.ez88_oq_off = (uint32_t *)malloc(T1 * sizeof(uint32_t));
        uint32_t f_oq = 0u;
        for (int t = 0; t <= t_max; ++t) {
            g_e8.f88_oq_off[t] = f_oq;
            g_e8.ez88_oq_off[t] = f_oq;
            f_oq += (uint32_t)(t + 2);
        }
        for (int pa = 0; pa < 2; ++pa) {
            for (int pb = 0; pb < 2; ++pb) {
                int q = (pa << 1) | pb;
                g_e8.f88_eq_cdf[q] = (uint16_t *)malloc((size_t)f_eq * sizeof(uint16_t));
                g_e8.f88_oq_cdf[q] = (uint16_t *)malloc((size_t)f_oq * sizeof(uint16_t));
                g_e8.g88_eq_sh[q] = (uint8_t *)calloc(R1, sizeof(uint8_t));
                g_e8.g88_oq_sh[q] = (uint8_t *)calloc(T1, sizeof(uint8_t));
                g_e8.g88_eq_idx[q] = (uint8_t *)calloc(R1 * G4_NB_88, sizeof(uint8_t));
                g_e8.g88_oq_idx[q] = (uint8_t *)calloc(T1 * G4_NB_88, sizeof(uint8_t));
                g_e8.ez88_eq_val[q] = (uint16_t *)calloc((size_t)f_eq, sizeof(uint16_t));
                g_e8.ez88_eq_a[q] = (uint16_t *)calloc((size_t)f_eq, sizeof(uint16_t));
                g_e8.ez88_oq_val[q] = (uint16_t *)calloc((size_t)f_oq, sizeof(uint16_t));
                g_e8.ez88_oq_a[q] = (uint16_t *)calloc((size_t)f_oq, sizeof(uint16_t));
                for (int r = 0; r <= r_max; ++r) {
                    uint16_t *cdf = g_e8.f88_eq_cdf[q] + g_e8.f88_eq_off[r];
                    uint32_t acc = 0u;
                    for (int na = 0; na <= r; ++na) {
                        cdf[na] = (uint16_t)acc;
                        acc += (uint32_t)g_e8.even_c2_u8[pa][na] * g_e8.even_c2_u8[pb][r - na];
                    }
                    cdf[r + 1] = UINT16_MAX;
                    uint32_t tb = bit_length_u32(acc > 0u ? acc - 1u : 0u);
                    uint8_t sh = (tb > G4_KBITS_88) ? (uint8_t)(tb - G4_KBITS_88) : 0u;
                    g_e8.g88_eq_sh[q][r] = sh;
                    uint8_t *gi = g_e8.g88_eq_idx[q] + (size_t)r * G4_NB_88;
                    int na_cur = 0;
                    for (uint32_t b = 0u; b < G4_NB_88; ++b) {
                        uint32_t lo_r = b << sh;
                        if (acc == 0u || lo_r >= acc) { gi[b] = (uint8_t)r; continue; }
                        while (na_cur < r && cdf[na_cur + 1] <= lo_r) ++na_cur;
                        gi[b] = (uint8_t)na_cur;
                    }
                    eytz_fill_u16(cdf, (uint32_t)(r + 1),
                                  g_e8.ez88_eq_val[q] + g_e8.ez88_eq_off[r],
                                  g_e8.ez88_eq_a[q] + g_e8.ez88_eq_off[r], 1u, 0u);
                }
                for (int t = 0; t <= t_max; ++t) {
                    uint16_t *cdf = g_e8.f88_oq_cdf[q] + g_e8.f88_oq_off[t];
                    uint32_t acc = 0u;
                    for (int na = 0; na <= t; ++na) {
                        cdf[na] = (uint16_t)acc;
                        acc += (uint32_t)g_e8.odd_c2_u8[pa][na] * g_e8.odd_c2_u8[pb][t - na];
                    }
                    cdf[t + 1] = UINT16_MAX;
                    uint32_t tb = bit_length_u32(acc > 0u ? acc - 1u : 0u);
                    uint8_t sh = (tb > G4_KBITS_88) ? (uint8_t)(tb - G4_KBITS_88) : 0u;
                    g_e8.g88_oq_sh[q][t] = sh;
                    uint8_t *gi = g_e8.g88_oq_idx[q] + (size_t)t * G4_NB_88;
                    int na_cur = 0;
                    for (uint32_t b = 0u; b < G4_NB_88; ++b) {
                        uint32_t lo_r = b << sh;
                        if (acc == 0u || lo_r >= acc) { gi[b] = (uint8_t)t; continue; }
                        while (na_cur < t && cdf[na_cur + 1] <= lo_r) ++na_cur;
                        gi[b] = (uint8_t)na_cur;
                    }
                    eytz_fill_u16(cdf, (uint32_t)(t + 1),
                                  g_e8.ez88_oq_val[q] + g_e8.ez88_oq_off[t],
                                  g_e8.ez88_oq_a[q] + g_e8.ez88_oq_off[t], 1u, 0u);
                }
            }
        }
        g_e8.mem_bytes_baseline = 5490u;
        g_e8.mem_bytes_guide_a = (size_t)(f_ec + f_oc) * 8u + (M1 + T1) * 2u * G8_NB_88
                                 + (size_t)(f_eq + f_oq) * 8u + (R1 + T1) * 4u * G4_NB_88;
        g_e8.mem_bytes_eytz_b2 = (size_t)(e_ec + e_oc) * 12u + (size_t)(f_eq + f_oq) * 16u + 2401u * 8u;
        g_e8.mem_bytes_eytz_b3 = (size_t)(e_ec + e_oc) * 12u + (size_t)(f_eq + f_oq) * 16u + 9121u * 8u;
    } else {
        /* 136b: Compact O(1) 64-bucket (8->4) & 32-bucket (4->2) guides + Stride-8 Eytzinger */
        uint64_t *tmp64 = (uint64_t *)malloc((R1 + 2u) * sizeof(uint64_t));
        uint16_t *tmp16 = (uint16_t *)malloc((R1 + 2u) * sizeof(uint16_t));
        uint64_t *cp64 = (uint64_t *)malloc((R1 + 2u) * sizeof(uint64_t));
        uint16_t *cp16 = (uint16_t *)malloc((R1 + 2u) * sizeof(uint16_t));

        g_e8.ez136_ec8_off = (uint32_t *)malloc(M1 * sizeof(uint32_t));
        g_e8.ez136_ec8_len = (uint32_t *)malloc(M1 * sizeof(uint32_t));
        uint32_t e_ec = 0u;
        for (int m = 0; m <= m_max; ++m) {
            uint32_t nblk = (uint32_t)((2 * m) / E8_EYTZ_STR8) + 1u;
            g_e8.ez136_ec8_off[m] = e_ec;
            g_e8.ez136_ec8_len[m] = nblk;
            e_ec += nblk + 1u;
        }
        g_e8.ez136_oc8_off = (uint32_t *)malloc(T1 * sizeof(uint32_t));
        g_e8.ez136_oc8_len = (uint32_t *)malloc(T1 * sizeof(uint32_t));
        uint32_t e_oc = 0u;
        for (int t = 0; t <= t_max; ++t) {
            uint32_t nblk = (uint32_t)(t / E8_EYTZ_STR8) + 1u;
            g_e8.ez136_oc8_off[t] = e_oc;
            g_e8.ez136_oc8_len[t] = nblk;
            e_oc += nblk + 1u;
        }
        for (int p = 0; p < 2; ++p) {
            g_e8.g136_ec8_sh[p] = (uint8_t *)calloc(M1, sizeof(uint8_t));
            g_e8.g136_oc8_sh[p] = (uint8_t *)calloc(T1, sizeof(uint8_t));
            g_e8.g136_ec8_idx[p] = (uint16_t *)calloc(M1 * G8_NB_136, sizeof(uint16_t));
            g_e8.g136_ec8_cdf[p] = (uint64_t *)calloc(M1 * G8_NB_136, sizeof(uint64_t));
            g_e8.g136_oc8_idx[p] = (uint16_t *)calloc(T1 * G8_NB_136, sizeof(uint16_t));
            g_e8.g136_oc8_cdf[p] = (uint64_t *)calloc(T1 * G8_NB_136, sizeof(uint64_t));
            g_e8.ez136_ec8_val[p] = (uint64_t *)calloc((size_t)e_ec, sizeof(uint64_t));
            g_e8.ez136_ec8_a[p] = (uint16_t *)calloc((size_t)e_ec, sizeof(uint16_t));
            g_e8.ez136_oc8_val[p] = (uint64_t *)calloc((size_t)e_oc, sizeof(uint64_t));
            g_e8.ez136_oc8_a[p] = (uint16_t *)calloc((size_t)e_oc, sizeof(uint16_t));

            for (int m = 0; m <= m_max; ++m) {
                int r8 = 2 * m;
                uint64_t acc = 0u;
                for (int a = 0; a <= r8; ++a) {
                    tmp64[a] = acc;
                    if ((a & (E8_EYTZ_STR8 - 1)) == 0) cp64[a / E8_EYTZ_STR8] = acc;
                    acc += (uint64_t)g_e8.even_c4_u16[p][a] * g_e8.even_c4_u16[p][r8 - a];
                }
                uint32_t tb = bit_length_u64(acc > 0u ? acc - 1u : 0u);
                uint8_t sh = (tb > G8_KBITS_136) ? (uint8_t)(tb - G8_KBITS_136) : 0u;
                g_e8.g136_ec8_sh[p][m] = sh;
                uint16_t *gi = g_e8.g136_ec8_idx[p] + (size_t)m * G8_NB_136;
                uint64_t *gc = g_e8.g136_ec8_cdf[p] + (size_t)m * G8_NB_136;
                int a_cur = 0;
                for (uint32_t b = 0u; b < G8_NB_136; ++b) {
                    uint64_t lo_r = (uint64_t)b << sh;
                    if (acc == 0u || lo_r >= acc) { gi[b] = (uint16_t)r8; gc[b] = tmp64[r8]; continue; }
                    while (a_cur < r8 && tmp64[a_cur + 1] <= lo_r) ++a_cur;
                    gi[b] = (uint16_t)a_cur;
                    gc[b] = tmp64[a_cur];
                }
                eytz_fill_u64(cp64, g_e8.ez136_ec8_len[m],
                              g_e8.ez136_ec8_val[p] + g_e8.ez136_ec8_off[m],
                              g_e8.ez136_ec8_a[p] + g_e8.ez136_ec8_off[m], 1u, 0u);
            }
            for (int t = 0; t <= t_max; ++t) {
                uint64_t acc = 0u;
                for (int a = 0; a <= t; ++a) {
                    tmp64[a] = acc;
                    if ((a & (E8_EYTZ_STR8 - 1)) == 0) cp64[a / E8_EYTZ_STR8] = acc;
                    acc += (uint64_t)g_e8.odd_c4_u16[p][a] * g_e8.odd_c4_u16[p][t - a];
                }
                uint32_t tb = bit_length_u64(acc > 0u ? acc - 1u : 0u);
                uint8_t sh = (tb > G8_KBITS_136) ? (uint8_t)(tb - G8_KBITS_136) : 0u;
                g_e8.g136_oc8_sh[p][t] = sh;
                uint16_t *gi = g_e8.g136_oc8_idx[p] + (size_t)t * G8_NB_136;
                uint64_t *gc = g_e8.g136_oc8_cdf[p] + (size_t)t * G8_NB_136;
                int a_cur = 0;
                for (uint32_t b = 0u; b < G8_NB_136; ++b) {
                    uint64_t lo_r = (uint64_t)b << sh;
                    if (acc == 0u || lo_r >= acc) { gi[b] = (uint16_t)t; gc[b] = tmp64[t]; continue; }
                    while (a_cur < t && tmp64[a_cur + 1] <= lo_r) ++a_cur;
                    gi[b] = (uint16_t)a_cur;
                    gc[b] = tmp64[a_cur];
                }
                eytz_fill_u64(cp64, g_e8.ez136_oc8_len[t],
                              g_e8.ez136_oc8_val[p] + g_e8.ez136_oc8_off[t],
                              g_e8.ez136_oc8_a[p] + g_e8.ez136_oc8_off[t], 1u, 0u);
            }
        }

        g_e8.ez136_eq_off = (uint32_t *)malloc(R1 * sizeof(uint32_t));
        g_e8.ez136_eq_len = (uint32_t *)malloc(R1 * sizeof(uint32_t));
        uint32_t e_eq = 0u;
        for (int r = 0; r <= r_max; ++r) {
            uint32_t nblk = (uint32_t)(r / E8_EYTZ_STR8) + 1u;
            g_e8.ez136_eq_off[r] = e_eq;
            g_e8.ez136_eq_len[r] = nblk;
            e_eq += nblk + 1u;
        }
        g_e8.ez136_oq_off = (uint32_t *)malloc(T1 * sizeof(uint32_t));
        g_e8.ez136_oq_len = (uint32_t *)malloc(T1 * sizeof(uint32_t));
        uint32_t e_oq = 0u;
        for (int t = 0; t <= t_max; ++t) {
            uint32_t nblk = (uint32_t)(t / E8_EYTZ_STR8) + 1u;
            g_e8.ez136_oq_off[t] = e_oq;
            g_e8.ez136_oq_len[t] = nblk;
            e_oq += nblk + 1u;
        }
        for (int pa = 0; pa < 2; ++pa) {
            for (int pb = 0; pb < 2; ++pb) {
                int q = (pa << 1) | pb;
                g_e8.g136_eq_sh[q] = (uint8_t *)calloc(R1, sizeof(uint8_t));
                g_e8.g136_oq_sh[q] = (uint8_t *)calloc(T1, sizeof(uint8_t));
                g_e8.g136_eq_pack[q] = (uint32_t *)calloc(R1 * G4_NB_136, sizeof(uint32_t));
                g_e8.g136_oq_pack[q] = (uint32_t *)calloc(T1 * G4_NB_136, sizeof(uint32_t));
                g_e8.ez136_eq_val[q] = (uint16_t *)calloc((size_t)e_eq, sizeof(uint16_t));
                g_e8.ez136_eq_a[q] = (uint16_t *)calloc((size_t)e_eq, sizeof(uint16_t));
                g_e8.ez136_oq_val[q] = (uint16_t *)calloc((size_t)e_oq, sizeof(uint16_t));
                g_e8.ez136_oq_a[q] = (uint16_t *)calloc((size_t)e_oq, sizeof(uint16_t));
                for (int r = 0; r <= r_max; ++r) {
                    uint32_t acc = 0u;
                    for (int na = 0; na <= r; ++na) {
                        tmp16[na] = (uint16_t)acc;
                        if ((na & (E8_EYTZ_STR8 - 1)) == 0) cp16[na / E8_EYTZ_STR8] = (uint16_t)acc;
                        acc += (uint32_t)g_e8.even_c2_u8[pa][na] * g_e8.even_c2_u8[pb][r - na];
                    }
                    uint32_t tb = bit_length_u32(acc > 0u ? acc - 1u : 0u);
                    uint8_t sh = (tb > G4_KBITS_136) ? (uint8_t)(tb - G4_KBITS_136) : 0u;
                    g_e8.g136_eq_sh[q][r] = sh;
                    uint32_t *gp = g_e8.g136_eq_pack[q] + (size_t)r * G4_NB_136;
                    int na_cur = 0;
                    for (uint32_t b = 0u; b < G4_NB_136; ++b) {
                        uint32_t lo_r = b << sh;
                        if (acc == 0u || lo_r >= acc) {
                            gp[b] = ((uint32_t)tmp16[r] << 16) | (uint32_t)r;
                            continue;
                        }
                        while (na_cur < r && tmp16[na_cur + 1] <= lo_r) ++na_cur;
                        gp[b] = ((uint32_t)tmp16[na_cur] << 16) | (uint32_t)na_cur;
                    }
                    eytz_fill_u16(cp16, g_e8.ez136_eq_len[r],
                                  g_e8.ez136_eq_val[q] + g_e8.ez136_eq_off[r],
                                  g_e8.ez136_eq_a[q] + g_e8.ez136_eq_off[r], 1u, 0u);
                }
                for (int t = 0; t <= t_max; ++t) {
                    uint32_t acc = 0u;
                    for (int na = 0; na <= t; ++na) {
                        tmp16[na] = (uint16_t)acc;
                        if ((na & (E8_EYTZ_STR8 - 1)) == 0) cp16[na / E8_EYTZ_STR8] = (uint16_t)acc;
                        acc += (uint32_t)g_e8.odd_c2_u8[pa][na] * g_e8.odd_c2_u8[pb][t - na];
                    }
                    uint32_t tb = bit_length_u32(acc > 0u ? acc - 1u : 0u);
                    uint8_t sh = (tb > G4_KBITS_136) ? (uint8_t)(tb - G4_KBITS_136) : 0u;
                    g_e8.g136_oq_sh[q][t] = sh;
                    uint32_t *gp = g_e8.g136_oq_pack[q] + (size_t)t * G4_NB_136;
                    int na_cur = 0;
                    for (uint32_t b = 0u; b < G4_NB_136; ++b) {
                        uint32_t lo_r = b << sh;
                        if (acc == 0u || lo_r >= acc) {
                            gp[b] = ((uint32_t)tmp16[t] << 16) | (uint32_t)t;
                            continue;
                        }
                        while (na_cur < t && tmp16[na_cur + 1] <= lo_r) ++na_cur;
                        gp[b] = ((uint32_t)tmp16[na_cur] << 16) | (uint32_t)na_cur;
                    }
                    eytz_fill_u16(cp16, g_e8.ez136_oq_len[t],
                                  g_e8.ez136_oq_val[q] + g_e8.ez136_oq_off[t],
                                  g_e8.ez136_oq_a[q] + g_e8.ez136_oq_off[t], 1u, 0u);
                }
            }
        }
        free(tmp64);
        free(tmp16);
        free(cp64);
        free(cp16);
        g_e8.mem_bytes_baseline = (size_t)(tot_ec8 + tot_oc8) * 16u + (size_t)(tot_eq + tot_oq) * 16u + 56850u;
        g_e8.mem_bytes_guide_a = (M1 + T1) * 2u * G8_NB_136 * 10u + (R1 + T1) * 4u * G4_NB_136 * 4u;
        g_e8.mem_bytes_eytz_b2 = (size_t)(e_ec + e_oc) * 20u + (size_t)(e_eq + e_oq) * 16u + 2401u * 8u;
        g_e8.mem_bytes_eytz_b3 = (size_t)(e_ec + e_oc) * 20u + (size_t)(e_eq + e_oq) * 16u + 9121u * 8u;
    }

    /* Populate Flat BallPack LUT for m = 0..3 (9,121 vectors = 72,968 B) */
    g_e8.flat_lut_off[0] = 0u;
    g_e8.flat_lut_off[1] = 1u;
    g_e8.flat_lut_off[2] = 241u;
    g_e8.flat_lut_off[3] = 2401u;
    g_e8.flat_lut_off[4] = 9121u;
    g_e8.flat_lut = (int8_t *)malloc(9121u * 8u);
    g_e8_ready = 1;
    for (int m = 0; m <= 3; ++m) {
        uint64_t cnt = g_e8.s_blk1[m];
        uint32_t off = g_e8.flat_lut_off[m];
        for (uint64_t rk = 0u; rk < cnt; ++rk) {
            unrank_e8_baseline(rk, 0, m, g_e8.flat_lut + (size_t)(off + rk) * 8u);
        }
    }

    /* Build Accelerator (C): D_8^+ Signed-Leader + 510B Comb + 22-Shape pshufb Tables */
    build_comb_table();
    build_universal_pshufb_table();
    build_d8plus_tables(m_max);
    size_t d8_core = (size_t)g_e8.d8_total_leaders * (sizeof(D8Leader16) + 2u) + M1 * D8_G_NB * 4u;
    g_e8.mem_bytes_d8_c1 = d8_core + 502u;
    g_e8.mem_bytes_d8_c2 = d8_core + (size_t)g_e8.shape_perm_off[22] * 8u;
    return 0;
}

void e8s1_get_mem_bytes(uint64_t out6[6], uint32_t *d8_total_leaders_out) {
    out6[0] = (uint64_t)g_e8.mem_bytes_baseline;
    out6[1] = (uint64_t)g_e8.mem_bytes_guide_a;
    out6[2] = (uint64_t)g_e8.mem_bytes_eytz_b2;
    out6[3] = (uint64_t)g_e8.mem_bytes_eytz_b3;
    out6[4] = (uint64_t)g_e8.mem_bytes_d8_c1;
    out6[5] = (uint64_t)g_e8.mem_bytes_d8_c2;
    *d8_total_leaders_out = g_e8.d8_total_leaders;
}

/* ============================================================================
 * WHT_8 Leaf & Activation Pre-Transform + AVX2/FMA Dot Product
 * ============================================================================ */

#if HAS_AVX2
static inline void wht8x4_u_to_z_avx2(const int8_t *u32, int8_t *z32) {
    __m256i v0 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)u32));
    __m256i v1 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(u32 + 16)));
    const __m256i sgn1 = _mm256_set1_epi32((int)0xffff0001u);
    v0 = _mm256_add_epi16(_mm256_sign_epi16(v0, sgn1), _mm256_shufflelo_epi16(_mm256_shufflehi_epi16(v0, 0xB1), 0xB1));
    v1 = _mm256_add_epi16(_mm256_sign_epi16(v1, sgn1), _mm256_shufflelo_epi16(_mm256_shufflehi_epi16(v1, 0xB1), 0xB1));
    const __m256i sgn2 = _mm256_setr_epi16(1,1,-1,-1,1,1,-1,-1, 1,1,-1,-1,1,1,-1,-1);
    v0 = _mm256_add_epi16(_mm256_sign_epi16(v0, sgn2), _mm256_shuffle_epi32(v0, 0xB1));
    v1 = _mm256_add_epi16(_mm256_sign_epi16(v1, sgn2), _mm256_shuffle_epi32(v1, 0xB1));
    const __m256i sgn4 = _mm256_setr_epi16(1,1,1,1,-1,-1,-1,-1, 1,1,1,1,-1,-1,-1,-1);
    v0 = _mm256_srai_epi16(_mm256_add_epi16(_mm256_sign_epi16(v0, sgn4), _mm256_shuffle_epi32(v0, 0x4E)), 2);
    v1 = _mm256_srai_epi16(_mm256_add_epi16(_mm256_sign_epi16(v1, sgn4), _mm256_shuffle_epi32(v1, 0x4E)), 2);
    _mm256_storeu_si256((__m256i *)z32, _mm256_permute4x64_epi64(_mm256_packs_epi16(v0, v1), 0xD8));
}

static inline float hsum256_ps(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

static inline float dot_i8_32_f32_avx2(const int8_t *w32, const float *x32) {
    __m256 f0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(w32 + 0))));
    __m256 f1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(w32 + 8))));
    __m256 f2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(w32 + 16))));
    __m256 f3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(w32 + 24))));
    __m256 acc = _mm256_mul_ps(f0, _mm256_loadu_ps(x32 + 0));
    acc = _mm256_fmadd_ps(f1, _mm256_loadu_ps(x32 + 8), acc);
    acc = _mm256_fmadd_ps(f2, _mm256_loadu_ps(x32 + 16), acc);
    acc = _mm256_fmadd_ps(f3, _mm256_loadu_ps(x32 + 24), acc);
    return hsum256_ps(acc);
}
#else
static inline void wht8_u_to_z_scalar(const int8_t *u, int8_t *z) {
    int a0 = u[0] + u[1], a1 = u[0] - u[1];
    int a2 = u[2] + u[3], a3 = u[2] - u[3];
    int a4 = u[4] + u[5], a5 = u[4] - u[5];
    int a6 = u[6] + u[7], a7 = u[6] - u[7];
    int b0 = a0 + a2, b1 = a1 + a3, b2 = a0 - a2, b3 = a1 - a3;
    int b4 = a4 + a6, b5 = a5 + a7, b6 = a4 - a6, b7 = a5 - a7;
    z[0] = (int8_t)((b0 + b4) >> 2); z[1] = (int8_t)((b1 + b5) >> 2);
    z[2] = (int8_t)((b2 + b6) >> 2); z[3] = (int8_t)((b3 + b7) >> 2);
    z[4] = (int8_t)((b0 - b4) >> 2); z[5] = (int8_t)((b1 - b5) >> 2);
    z[6] = (int8_t)((b2 - b6) >> 2); z[7] = (int8_t)((b3 - b7) >> 2);
}

static inline void wht8x4_u_to_z_avx2(const int8_t *u32, int8_t *z32) {
    for (int b = 0; b < 4; ++b) wht8_u_to_z_scalar(u32 + 8 * b, z32 + 8 * b);
}

static inline float dot_i8_32_f32_avx2(const int8_t *w32, const float *x32) {
    float acc = 0.0f;
    for (int i = 0; i < 32; ++i) acc += (float)w32[i] * x32[i];
    return acc;
}
#endif

static inline void wht8_f32_vector(const float *x, float *x_tilde, size_t k_cols) {
    const float inv_sqrt8 = 0.3535533905932737622f;
    for (size_t j = 0; j < k_cols; j += 8u) {
        float a0 = x[j + 0] + x[j + 1], a1 = x[j + 0] - x[j + 1];
        float a2 = x[j + 2] + x[j + 3], a3 = x[j + 2] - x[j + 3];
        float a4 = x[j + 4] + x[j + 5], a5 = x[j + 4] - x[j + 5];
        float a6 = x[j + 6] + x[j + 7], a7 = x[j + 6] - x[j + 7];
        float b0 = a0 + a2, b1 = a1 + a3, b2 = a0 - a2, b3 = a1 - a3;
        float b4 = a4 + a6, b5 = a5 + a7, b6 = a4 - a6, b7 = a5 - a7;
        x_tilde[j + 0] = (b0 + b4) * inv_sqrt8; x_tilde[j + 1] = (b1 + b5) * inv_sqrt8;
        x_tilde[j + 2] = (b2 + b6) * inv_sqrt8; x_tilde[j + 3] = (b3 + b7) * inv_sqrt8;
        x_tilde[j + 4] = (b0 - b4) * inv_sqrt8; x_tilde[j + 5] = (b1 - b5) * inv_sqrt8;
        x_tilde[j + 6] = (b2 - b6) * inv_sqrt8; x_tilde[j + 7] = (b3 - b7) * inv_sqrt8;
    }
}

/* ============================================================================
 * Outer Stage-0 Splits (Identical Across All Variants!)
 * ============================================================================ */

static inline int root_split_e8_4(const uint8_t *in, int *m_l_out, u128 *idx_l_out, u128 *idx_r_out) {
    u192 rk = load_u192_le(in, g_e8.bytes);
    if (u192_ge(&rk, &g_e8.total_count)) return 1;
    uint32_t lo = 0u, len = (uint32_t)g_e8.m_max + 1u;
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
    *m_l_out = m_l; *idx_l_out = q; *idx_r_out = r;
    return 0;
}

static inline void split_e8_blk2_shell(int m_l, u128 idx_l, int *a0_out, uint64_t *l0_out, uint64_t *r0_out) {
    const uint64_t *s1 = g_e8.s_blk1;
    int a0;
    u128 rem0;
    if (!g_e8.use_u32) {
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
    if (!g_e8.use_u32) {
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

static inline int resolve_leaf_ball_u64(uint64_t *rem_io, int is_ball, int m_bound) {
    if (!is_ball) return m_bound;
    uint64_t rem = *rem_io;
    const uint64_t *w1 = g_e8.w_blk1;
    uint32_t lo = 0u, len = (uint32_t)m_bound + 1u;
    while (len > 1u) {
        uint32_t half = len >> 1;
        uint32_t mid = lo + half;
        lo = (w1[mid - 1u] <= rem) ? mid : lo;
        len -= half;
    }
    int m = (int)lo;
    if (m > 0) rem -= w1[m - 1];
    *rem_io = rem;
    return m;
}

static inline int resolve_leaf_ball_u32(uint32_t *rem_io, int is_ball, int m_bound) {
    if (!is_ball) return m_bound;
    uint32_t rem = *rem_io;
    const uint32_t *w1 = g_e8.w_blk1_u32;
    uint32_t lo = 0u, len = (uint32_t)m_bound + 1u;
    while (len > 1u) {
        uint32_t half = len >> 1;
        uint32_t mid = lo + half;
        lo = (w1[mid - 1u] <= rem) ? mid : lo;
        len -= half;
    }
    int m = (int)lo;
    if (m > 0) rem -= w1[m - 1];
    *rem_io = rem;
    return m;
}

/* ============================================================================
 * STAGE-1 VARIANT 0: AUDIT-115 Baseline (Exact Reference)
 * ============================================================================ */

static inline void unrank_quad_baseline(uint32_t rem, int coset, int norm4, int parity4, int8_t *out) {
    uint32_t p0_tot = (coset == 0) ? g_e8.q_even_p0_tot[parity4][norm4] : g_e8.q_odd_p0_tot[parity4][norm4];
    int pa = (rem < p0_tot) ? 0 : 1;
    if (pa) rem -= p0_tot;
    int pb = parity4 ^ pa;
    const uint16_t *ca = (coset == 0) ? g_e8.even_c2[pa] : g_e8.odd_c2[pa];
    const uint16_t *cb = (coset == 0) ? g_e8.even_c2[pb] : g_e8.odd_c2[pb];
    int na = 0;
    if (!g_e8.use_u32) {
        int qid = (pa << 1) | pb;
        const uint32_t *cp = (coset == 0) ? (g_e8.cp_eq[qid] + g_e8.cp_eq_off[norm4])
                                          : (g_e8.cp_oq[qid] + g_e8.cp_oq_off[norm4]);
        uint32_t nblk = (coset == 0) ? g_e8.cp_eq_len[norm4] : g_e8.cp_oq_len[norm4];
        uint32_t blk = find_cp_u32(cp, nblk, rem);
        rem -= cp[blk];
        na = (int)(blk * E8_Q_STRIDE);
    }
    for (; na <= norm4; ++na) {
        uint32_t term = (uint32_t)ca[na] * cb[norm4 - na];
        if (rem < term) break;
        rem -= term;
    }
    int nb = norm4 - na;
    Recip32 rec = (coset == 0) ? g_e8.rec_even_c2[pb][nb] : g_e8.rec_odd_c2[pb][nb];
    uint32_t il = mul_shr_u32_to_u32(rem, rec.m, rec.shift);
    uint32_t ir = rem - il * cb[nb];
    uint32_t pl = (uint32_t)((coset == 0) ? g_e8.even_p2_off[pa][na] : g_e8.odd_p2_off[pa][na]) + il;
    uint32_t pr = (uint32_t)((coset == 0) ? g_e8.even_p2_off[pb][nb] : g_e8.odd_p2_off[pb][nb]) + ir;
    const int8_t *pt_a = (coset == 0) ? g_e8.even_p2_pts[pa] : g_e8.odd_p2_pts[pa];
    const int8_t *pt_b = (coset == 0) ? g_e8.even_p2_pts[pb] : g_e8.odd_p2_pts[pb];
    out[0] = pt_a[2u * pl]; out[1] = pt_a[2u * pl + 1u];
    out[2] = pt_b[2u * pr]; out[3] = pt_b[2u * pr + 1u];
}

static inline void unrank_e8_baseline(uint64_t rank, int is_ball, int m_bound, int8_t *out) {
    uint64_t rem = rank;
    int m = resolve_leaf_ball_u64(&rem, is_ball, m_bound);
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
    if (!g_e8.use_u32) {
        const uint64_t *cp = (coset == 0) ? (g_e8.cp_ec8[p] + g_e8.cp_ec8_off[m])
                                          : (g_e8.cp_oc8[p] + g_e8.cp_oc8_off[n_tot]);
        uint32_t nblk = (coset == 0) ? g_e8.cp_ec8_len[m] : g_e8.cp_oc8_len[n_tot];
        uint32_t blk = find_cp_u64(cp, nblk, rem);
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
    unrank_quad_baseline(il, coset, n_l, p, out);
    unrank_quad_baseline(ir, coset, n_r, p, out + 4);
}

/* ============================================================================
 * STAGE-1 VARIANT 1 (A): Pure u32/u64 + O(1) Top-Bit CDF-Prefix Bucket Guide
 * ============================================================================ */

static inline void write_pair_u16(int8_t *dst, uint16_t packed) {
    memcpy(dst, &packed, 2u);
}

static inline void unrank_quad_guide_88(uint32_t rem, int coset, int norm4, int parity4, int8_t *out) {
    uint32_t p0_tot = (coset == 0) ? g_e8.q_even_p0_u16[parity4][norm4] : g_e8.q_odd_p0_u16[parity4][norm4];
    int pa = (rem < p0_tot) ? 0 : 1;
    if (pa) rem -= p0_tot;
    int pb = parity4 ^ pa;
    int q = (pa << 1) | pb;
    const uint16_t *cdf;
    uint32_t b;
    int na;
    if (coset == 0) {
        cdf = g_e8.f88_eq_cdf[q] + g_e8.f88_eq_off[norm4];
        b = rem >> g_e8.g88_eq_sh[q][norm4];
        na = g_e8.g88_eq_idx[q][(size_t)norm4 * G4_NB_88 + b];
    } else {
        cdf = g_e8.f88_oq_cdf[q] + g_e8.f88_oq_off[norm4];
        b = rem >> g_e8.g88_oq_sh[q][norm4];
        na = g_e8.g88_oq_idx[q][(size_t)norm4 * G4_NB_88 + b];
    }
    while (cdf[na + 1] <= rem) ++na;
    rem -= cdf[na];
    int nb = norm4 - na;
    const uint8_t *cb = (coset == 0) ? g_e8.even_c2_u8[pb] : g_e8.odd_c2_u8[pb];
    Recip32 rec = (coset == 0) ? g_e8.rec_even_c2[pb][nb] : g_e8.rec_odd_c2[pb][nb];
    uint32_t il = mul_shr_u32_to_u32(rem, rec.m, rec.shift);
    uint32_t ir = rem - il * (uint32_t)cb[nb];
    uint32_t pl = (uint32_t)((coset == 0) ? g_e8.even_p2_off[pa][na] : g_e8.odd_p2_off[pa][na]) + il;
    uint32_t pr = (uint32_t)((coset == 0) ? g_e8.even_p2_off[pb][nb] : g_e8.odd_p2_off[pb][nb]) + ir;
    const uint16_t *pt_a = (coset == 0) ? g_e8.even_p2_u16[pa] : g_e8.odd_p2_u16[pa];
    const uint16_t *pt_b = (coset == 0) ? g_e8.even_p2_u16[pb] : g_e8.odd_p2_u16[pb];
    write_pair_u16(out, pt_a[pl]);
    write_pair_u16(out + 2, pt_b[pr]);
}

static inline void unrank_quad_guide_136(uint32_t rem, int coset, int norm4, int parity4, int8_t *out) {
    uint32_t p0_tot = (coset == 0) ? g_e8.q_even_p0_u16[parity4][norm4] : g_e8.q_odd_p0_u16[parity4][norm4];
    int pa = (rem < p0_tot) ? 0 : 1;
    if (pa) rem -= p0_tot;
    int pb = parity4 ^ pa;
    int q = (pa << 1) | pb;
    uint32_t pk;
    const uint8_t *ca, *cb;
    if (coset == 0) {
        uint32_t b = rem >> g_e8.g136_eq_sh[q][norm4];
        pk = g_e8.g136_eq_pack[q][(size_t)norm4 * G4_NB_136 + b];
        ca = g_e8.even_c2_u8[pa];
        cb = g_e8.even_c2_u8[pb];
    } else {
        uint32_t b = rem >> g_e8.g136_oq_sh[q][norm4];
        pk = g_e8.g136_oq_pack[q][(size_t)norm4 * G4_NB_136 + b];
        ca = g_e8.odd_c2_u8[pa];
        cb = g_e8.odd_c2_u8[pb];
    }
    int na = (int)(pk & 0xffffu);
    rem -= (pk >> 16);
    for (; na <= norm4; ++na) {
        uint32_t term = (uint32_t)ca[na] * (uint32_t)cb[norm4 - na];
        if (rem < term) break;
        rem -= term;
    }
    int nb = norm4 - na;
    Recip32 rec = (coset == 0) ? g_e8.rec_even_c2[pb][nb] : g_e8.rec_odd_c2[pb][nb];
    uint32_t il = mul_shr_u32_to_u32(rem, rec.m, rec.shift);
    uint32_t ir = rem - il * (uint32_t)cb[nb];
    uint32_t pl = (uint32_t)((coset == 0) ? g_e8.even_p2_off[pa][na] : g_e8.odd_p2_off[pa][na]) + il;
    uint32_t pr = (uint32_t)((coset == 0) ? g_e8.even_p2_off[pb][nb] : g_e8.odd_p2_off[pb][nb]) + ir;
    const uint16_t *pt_a = (coset == 0) ? g_e8.even_p2_u16[pa] : g_e8.odd_p2_u16[pa];
    const uint16_t *pt_b = (coset == 0) ? g_e8.even_p2_u16[pb] : g_e8.odd_p2_u16[pb];
    write_pair_u16(out, pt_a[pl]);
    write_pair_u16(out + 2, pt_b[pr]);
}

static inline void unrank_e8_guide_a(uint64_t rank, int is_ball, int m_bound, int8_t *out) {
    if (g_e8.use_u32) {
        /* PURE 32-BIT uint32_t Stage-1 for 88b (m_max=48, W1 < 2^29) + O(1) Guide! */
        uint32_t rem = (uint32_t)rank;
        int m = resolve_leaf_ball_u32(&rem, is_ball, m_bound);
        int coset, n_tot;
        uint32_t p0_tot;
        uint32_t ec8 = g_e8.even_c8_u32[m];
        if (rem < ec8) {
            coset = 0; n_tot = 2 * m; p0_tot = g_e8.e8_even_p0_u32[m];
        } else {
            coset = 1; rem -= ec8; n_tot = m - 1; p0_tot = g_e8.e8_odd_p0_u32[m];
        }
        int p = (rem < p0_tot) ? 0 : 1;
        if (p) rem -= p0_tot;
        const uint32_t *cdf;
        int n_l;
        if (coset == 0) {
            cdf = g_e8.f88_ec8_cdf[p] + g_e8.f88_ec8_off[m];
            uint32_t b = rem >> g_e8.g88_ec8_sh[p][m];
            n_l = g_e8.g88_ec8_idx[p][(size_t)m * G8_NB_88 + b];
        } else {
            cdf = g_e8.f88_oc8_cdf[p] + g_e8.f88_oc8_off[n_tot];
            uint32_t b = rem >> g_e8.g88_oc8_sh[p][n_tot];
            n_l = g_e8.g88_oc8_idx[p][(size_t)n_tot * G8_NB_88 + b];
        }
        while (cdf[n_l + 1] <= rem) ++n_l;
        rem -= cdf[n_l];
        int n_r = n_tot - n_l;
        Recip32 rec = (coset == 0) ? g_e8.rec_even_c4_u32[p][n_r] : g_e8.rec_odd_c4_u32[p][n_r];
        uint32_t d = (coset == 0) ? g_e8.even_c4_u16[p][n_r] : g_e8.odd_c4_u16[p][n_r];
        uint32_t il = mul_shr_u32_to_u32(rem, rec.m, rec.shift);
        uint32_t ir = rem - il * d;
        unrank_quad_guide_88(il, coset, n_l, p, out);
        unrank_quad_guide_88(ir, coset, n_r, p, out + 4);
    } else {
        /* PURE 64-BIT uint64_t Stage-1 for 136b (m_max=391, ZERO u128!) + O(1) 64-bucket Guide! */
        uint64_t rem = rank;
        int m = resolve_leaf_ball_u64(&rem, is_ball, m_bound);
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
        const uint16_t *c4;
        const Recip64 *recs;
        int n_l;
        if (coset == 0) {
            uint32_t b = (uint32_t)(rem >> g_e8.g136_ec8_sh[p][m]);
            size_t slot = (size_t)m * G8_NB_136 + b;
            n_l = g_e8.g136_ec8_idx[p][slot];
            rem -= g_e8.g136_ec8_cdf[p][slot];
            c4 = g_e8.even_c4_u16[p];
            recs = g_e8.rec_even_c4[p];
        } else {
            uint32_t b = (uint32_t)(rem >> g_e8.g136_oc8_sh[p][n_tot]);
            size_t slot = (size_t)n_tot * G8_NB_136 + b;
            n_l = g_e8.g136_oc8_idx[p][slot];
            rem -= g_e8.g136_oc8_cdf[p][slot];
            c4 = g_e8.odd_c4_u16[p];
            recs = g_e8.rec_odd_c4[p];
        }
        for (; n_l <= n_tot; ++n_l) {
            uint32_t term = (uint32_t)c4[n_l] * (uint32_t)c4[n_tot - n_l];
            if (rem < (uint64_t)term) break;
            rem -= (uint64_t)term;
        }
        int n_r = n_tot - n_l;
        Recip64 rec = recs[n_r];
        uint32_t il = (uint32_t)mul_shr_u64_to_u64(rem, rec.m, rec.shift);
        uint32_t ir = (uint32_t)(rem - (uint64_t)il * (uint64_t)c4[n_r]);
        unrank_quad_guide_136(il, coset, n_l, p, out);
        unrank_quad_guide_136(ir, coset, n_r, p, out + 4);
    }
}

/* ============================================================================
 * STAGE-1 VARIANT 2/3 (B): Branchless Eytzinger cmov + Flat int8[8] LUT (m<=2/3)
 * ============================================================================ */

static inline void unrank_quad_eytz_88(uint32_t rem, int coset, int norm4, int parity4, int8_t *out) {
    uint32_t p0_tot = (coset == 0) ? g_e8.q_even_p0_u16[parity4][norm4] : g_e8.q_odd_p0_u16[parity4][norm4];
    int pa = (rem < p0_tot) ? 0 : 1;
    if (pa) rem -= p0_tot;
    int pb = parity4 ^ pa;
    int q = (pa << 1) | pb;
    const uint16_t *ev = (coset == 0) ? (g_e8.ez88_eq_val[q] + g_e8.ez88_eq_off[norm4])
                                      : (g_e8.ez88_oq_val[q] + g_e8.ez88_oq_off[norm4]);
    const uint16_t *ea = (coset == 0) ? (g_e8.ez88_eq_a[q] + g_e8.ez88_eq_off[norm4])
                                      : (g_e8.ez88_oq_a[q] + g_e8.ez88_oq_off[norm4]);
    uint32_t ans = eytz_search_u16(ev, (uint32_t)(norm4 + 1), (uint16_t)rem);
    int na = (int)ea[ans];
    rem -= ev[ans];
    int nb = norm4 - na;
    const uint8_t *cb = (coset == 0) ? g_e8.even_c2_u8[pb] : g_e8.odd_c2_u8[pb];
    Recip32 rec = (coset == 0) ? g_e8.rec_even_c2[pb][nb] : g_e8.rec_odd_c2[pb][nb];
    uint32_t il = mul_shr_u32_to_u32(rem, rec.m, rec.shift);
    uint32_t ir = rem - il * (uint32_t)cb[nb];
    uint32_t pl = (uint32_t)((coset == 0) ? g_e8.even_p2_off[pa][na] : g_e8.odd_p2_off[pa][na]) + il;
    uint32_t pr = (uint32_t)((coset == 0) ? g_e8.even_p2_off[pb][nb] : g_e8.odd_p2_off[pb][nb]) + ir;
    const uint16_t *pt_a = (coset == 0) ? g_e8.even_p2_u16[pa] : g_e8.odd_p2_u16[pa];
    const uint16_t *pt_b = (coset == 0) ? g_e8.even_p2_u16[pb] : g_e8.odd_p2_u16[pb];
    write_pair_u16(out, pt_a[pl]);
    write_pair_u16(out + 2, pt_b[pr]);
}

static inline void unrank_quad_eytz_136(uint32_t rem, int coset, int norm4, int parity4, int8_t *out) {
    uint32_t p0_tot = (coset == 0) ? g_e8.q_even_p0_u16[parity4][norm4] : g_e8.q_odd_p0_u16[parity4][norm4];
    int pa = (rem < p0_tot) ? 0 : 1;
    if (pa) rem -= p0_tot;
    int pb = parity4 ^ pa;
    int q = (pa << 1) | pb;
    const uint16_t *ev = (coset == 0) ? (g_e8.ez136_eq_val[q] + g_e8.ez136_eq_off[norm4])
                                      : (g_e8.ez136_oq_val[q] + g_e8.ez136_oq_off[norm4]);
    const uint16_t *ea = (coset == 0) ? (g_e8.ez136_eq_a[q] + g_e8.ez136_eq_off[norm4])
                                      : (g_e8.ez136_oq_a[q] + g_e8.ez136_oq_off[norm4]);
    uint32_t nblk = (coset == 0) ? g_e8.ez136_eq_len[norm4] : g_e8.ez136_oq_len[norm4];
    uint32_t ans = eytz_search_u16(ev, nblk, (uint16_t)rem);
    rem -= ev[ans];
    int na = (int)ea[ans] * E8_EYTZ_STR8;
    const uint8_t *ca = (coset == 0) ? g_e8.even_c2_u8[pa] : g_e8.odd_c2_u8[pa];
    const uint8_t *cb = (coset == 0) ? g_e8.even_c2_u8[pb] : g_e8.odd_c2_u8[pb];
    for (; na <= norm4; ++na) {
        uint32_t term = (uint32_t)ca[na] * (uint32_t)cb[norm4 - na];
        if (rem < term) break;
        rem -= term;
    }
    int nb = norm4 - na;
    Recip32 rec = (coset == 0) ? g_e8.rec_even_c2[pb][nb] : g_e8.rec_odd_c2[pb][nb];
    uint32_t il = mul_shr_u32_to_u32(rem, rec.m, rec.shift);
    uint32_t ir = rem - il * (uint32_t)cb[nb];
    uint32_t pl = (uint32_t)((coset == 0) ? g_e8.even_p2_off[pa][na] : g_e8.odd_p2_off[pa][na]) + il;
    uint32_t pr = (uint32_t)((coset == 0) ? g_e8.even_p2_off[pb][nb] : g_e8.odd_p2_off[pb][nb]) + ir;
    const uint16_t *pt_a = (coset == 0) ? g_e8.even_p2_u16[pa] : g_e8.odd_p2_u16[pa];
    const uint16_t *pt_b = (coset == 0) ? g_e8.even_p2_u16[pb] : g_e8.odd_p2_u16[pb];
    write_pair_u16(out, pt_a[pl]);
    write_pair_u16(out + 2, pt_b[pr]);
}

static inline void unrank_e8_eytz_lut(uint64_t rank, int is_ball, int m_bound, int lut_m_max, int8_t *out) {
    if (g_e8.use_u32) {
        uint32_t rem = (uint32_t)rank;
        int m = resolve_leaf_ball_u32(&rem, is_ball, m_bound);
        if (m <= lut_m_max) {
            memcpy(out, g_e8.flat_lut + (size_t)(g_e8.flat_lut_off[m] + rem) * 8u, 8u);
            return;
        }
        int coset, n_tot;
        uint32_t p0_tot;
        uint32_t ec8 = g_e8.even_c8_u32[m];
        if (rem < ec8) {
            coset = 0; n_tot = 2 * m; p0_tot = g_e8.e8_even_p0_u32[m];
        } else {
            coset = 1; rem -= ec8; n_tot = m - 1; p0_tot = g_e8.e8_odd_p0_u32[m];
        }
        int p = (rem < p0_tot) ? 0 : 1;
        if (p) rem -= p0_tot;
        const uint32_t *ev = (coset == 0) ? (g_e8.ez88_ec8_val[p] + g_e8.ez88_ec8_off[m])
                                          : (g_e8.ez88_oc8_val[p] + g_e8.ez88_oc8_off[n_tot]);
        const uint16_t *ea = (coset == 0) ? (g_e8.ez88_ec8_a[p] + g_e8.ez88_ec8_off[m])
                                          : (g_e8.ez88_oc8_a[p] + g_e8.ez88_oc8_off[n_tot]);
        uint32_t ans = eytz_search_u32(ev, (uint32_t)(n_tot + 1), rem);
        int n_l = (int)ea[ans];
        rem -= ev[ans];
        int n_r = n_tot - n_l;
        Recip32 rec = (coset == 0) ? g_e8.rec_even_c4_u32[p][n_r] : g_e8.rec_odd_c4_u32[p][n_r];
        uint32_t d = (coset == 0) ? g_e8.even_c4_u16[p][n_r] : g_e8.odd_c4_u16[p][n_r];
        uint32_t il = mul_shr_u32_to_u32(rem, rec.m, rec.shift);
        uint32_t ir = rem - il * d;
        unrank_quad_eytz_88(il, coset, n_l, p, out);
        unrank_quad_eytz_88(ir, coset, n_r, p, out + 4);
    } else {
        uint64_t rem = rank;
        int m = resolve_leaf_ball_u64(&rem, is_ball, m_bound);
        if (m <= lut_m_max) {
            memcpy(out, g_e8.flat_lut + (size_t)(g_e8.flat_lut_off[m] + (uint32_t)rem) * 8u, 8u);
            return;
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
        const uint64_t *ev = (coset == 0) ? (g_e8.ez136_ec8_val[p] + g_e8.ez136_ec8_off[m])
                                          : (g_e8.ez136_oc8_val[p] + g_e8.ez136_oc8_off[n_tot]);
        const uint16_t *ea = (coset == 0) ? (g_e8.ez136_ec8_a[p] + g_e8.ez136_ec8_off[m])
                                          : (g_e8.ez136_oc8_a[p] + g_e8.ez136_oc8_off[n_tot]);
        uint32_t nblk = (coset == 0) ? g_e8.ez136_ec8_len[m] : g_e8.ez136_oc8_len[n_tot];
        uint32_t ans = eytz_search_u64(ev, nblk, rem);
        rem -= ev[ans];
        int n_l = (int)ea[ans] * E8_EYTZ_STR8;
        const uint16_t *c4 = (coset == 0) ? g_e8.even_c4_u16[p] : g_e8.odd_c4_u16[p];
        const Recip64 *recs = (coset == 0) ? g_e8.rec_even_c4[p] : g_e8.rec_odd_c4[p];
        for (; n_l <= n_tot; ++n_l) {
            uint32_t term = (uint32_t)c4[n_l] * (uint32_t)c4[n_tot - n_l];
            if (rem < (uint64_t)term) break;
            rem -= (uint64_t)term;
        }
        int n_r = n_tot - n_l;
        Recip64 rec = recs[n_r];
        uint32_t il = (uint32_t)mul_shr_u64_to_u64(rem, rec.m, rec.shift);
        uint32_t ir = (uint32_t)(rem - (uint64_t)il * (uint64_t)c4[n_r]);
        unrank_quad_eytz_136(il, coset, n_l, p, out);
        unrank_quad_eytz_136(ir, coset, n_r, p, out + 4);
    }
}

/* ============================================================================
 * STAGE-1 VARIANT 4/5 (C1/C2): D_8^+ Signed-Leader / Permutation-Orbit
 * ============================================================================ */

static inline uint32_t find_d8_leader_slot(int m, uint64_t rem) {
    uint32_t off = g_e8.d8_shell_off[m];
    uint32_t end = g_e8.d8_shell_off[m + 1];
    uint32_t b = (uint32_t)(rem >> g_e8.d8_guide_sh[m]);
    uint32_t idx = off + g_e8.d8_guide_idx[(size_t)m * D8_G_NB + b];
    const D8Leader16 *ldrs = g_e8.d8_leaders;
    while (idx + 1u < end && ldrs[idx + 1u].cdf <= rem) {
        ++idx;
    }
    return idx;
}

static inline void apply_d8_signs(int8_t out[8], int coset, uint32_t sign_bits) {
    if (coset == 0) {
        uint32_t s = sign_bits;
        for (int i = 0; i < 8; ++i) {
            if (out[i] != 0) {
                if (s & 1u) out[i] = (int8_t)(-out[i]);
                s >>= 1;
            }
        }
    } else {
        int sum7 = 0;
        for (int i = 0; i < 7; ++i) {
            int8_t v = ((sign_bits >> i) & 1u) ? (int8_t)(-out[i]) : out[i];
            out[i] = v;
            sum7 += v;
        }
        if (((sum7 + out[7]) & 3) != 0) {
            out[7] = (int8_t)(-out[7]);
        }
    }
}

/* (C1) Ultra-compact 510-byte binomial-mask permutation unranker (0 divisions!) */
static inline void unrank_e8_d8_compact(uint64_t rank, int is_ball, int m_bound, int8_t *out) {
    uint64_t rem = rank;
    int m = resolve_leaf_ball_u64(&rem, is_ball, m_bound);
    uint32_t slot = find_d8_leader_slot(m, rem);
    const D8Leader16 *ldr = &g_e8.d8_leaders[slot];
    rem -= ldr->cdf;
    uint8_t meta = g_e8.d8_meta[slot];
    int coset = (int)(meta >> 7);
    uint8_t sid = meta & 0x7fu;
    uint8_t sb = g_e8.d8_sbits[slot];
    uint32_t sign_bits = (uint32_t)(rem & ((1ULL << sb) - 1ULL));
    uint32_t pr = (uint32_t)(rem >> sb);

    int k = g_e8.shape_k[sid];
    const uint8_t *runs = g_e8.shape_runs[sid];
    uint8_t avail[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    int n_rem = 8;
    int src_pos = 0;
    for (int j = 0; j < k - 1; ++j) {
        int c = runs[j];
        int8_t val = ldr->vals[src_pos];
        src_pos += c;
        uint32_t cnt = g_e8.comb_cnt[n_rem][c];
        Recip32 rec = g_e8.comb_recip[cnt];
        uint32_t q = mul_shr_u32_to_u32(pr, rec.m, rec.shift);
        uint32_t ci = pr - q * cnt;
        pr = q;
        uint8_t mask = g_e8.comb_masks[n_rem][c][ci];
        uint8_t next_avail[8];
        int nn = 0;
        for (int p = 0; p < n_rem; ++p) {
            if ((mask >> p) & 1u) {
                out[avail[p]] = val;
            } else {
                next_avail[nn++] = avail[p];
            }
        }
        for (int p = 0; p < nn; ++p) avail[p] = next_avail[p];
        n_rem = nn;
    }
    int8_t last_val = ldr->vals[src_pos];
    for (int p = 0; p < n_rem; ++p) out[avail[p]] = last_val;
    apply_d8_signs(out, coset, sign_bits);
}

/* (C2) Universal 22-shape pshufb (_mm_shuffle_epi8) + sign unranker (0 divisions, 0 multiplies!) */
static inline void unrank_e8_d8_pshufb(uint64_t rank, int is_ball, int m_bound, int8_t *out) {
    uint64_t rem = rank;
    int m = resolve_leaf_ball_u64(&rem, is_ball, m_bound);
    uint32_t slot = find_d8_leader_slot(m, rem);
    const D8Leader16 *ldr = &g_e8.d8_leaders[slot];
    rem -= ldr->cdf;
    uint8_t meta = g_e8.d8_meta[slot];
    int coset = (int)(meta >> 7);
    uint8_t sid = meta & 0x7fu;
    uint8_t sb = g_e8.d8_sbits[slot];
    uint32_t sign_bits = (uint32_t)(rem & ((1ULL << sb) - 1ULL));
    uint32_t pr = (uint32_t)(rem >> sb);
    const uint8_t *shuf = g_e8.shape_pshufb + ((size_t)g_e8.shape_perm_off[sid] + pr) * 8u;

#if HAS_AVX2
    __m128i v_ldr = _mm_loadl_epi64((const __m128i *)ldr->vals);
    __m128i v_sh  = _mm_loadl_epi64((const __m128i *)shuf);
    __m128i v_out = _mm_shuffle_epi8(v_ldr, v_sh);
    if (coset == 1) {
        /* Odd coset: all 8 entries nonzero; 7 bits directly select signs[0..6], 8th fixed by sum%4 */
        __m128i v_sgn = _mm_loadl_epi64((const __m128i *)g_e8.sign8_lut[sign_bits]);
        v_out = _mm_sign_epi8(v_out, v_sgn);
        _mm_storel_epi64((__m128i *)out, v_out);
        int s8 = (int)out[0] + out[1] + out[2] + out[3] + out[4] + out[5] + out[6] + out[7];
        if ((s8 & 3) != 0) out[7] = (int8_t)(-out[7]);
    } else {
        _mm_storel_epi64((__m128i *)out, v_out);
        apply_d8_signs(out, 0, sign_bits);
    }
#else
    for (int i = 0; i < 8; ++i) out[i] = ldr->vals[shuf[i]];
    apply_d8_signs(out, coset, sign_bits);
#endif
}

/* ============================================================================
 * Unified Block-32 Unranker Dispatch (Variant 0..5)
 * ============================================================================ */

static inline size_t unrank_block32_variant(int variant, const uint8_t *in_blk, int8_t *u32) {
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
    int b0 = a0, b1 = m_l - a0, b2 = a1, b3 = (g_e8.m_max - m_l) - a1;

    switch (variant) {
    case 0: /* baseline */
        unrank_e8_baseline(l0, 0, b0, u32);
        unrank_e8_baseline(r0, 0, b1, u32 + 8);
        unrank_e8_baseline(l1, 0, b2, u32 + 16);
        unrank_e8_baseline(r1, 1, b3, u32 + 24);
        break;
    case 1: /* (A) guide_narrow */
        unrank_e8_guide_a(l0, 0, b0, u32);
        unrank_e8_guide_a(r0, 0, b1, u32 + 8);
        unrank_e8_guide_a(l1, 0, b2, u32 + 16);
        unrank_e8_guide_a(r1, 1, b3, u32 + 24);
        break;
    case 2: /* (B) eytz_lut2 (m<=2 L1d) */
        unrank_e8_eytz_lut(l0, 0, b0, 2, u32);
        unrank_e8_eytz_lut(r0, 0, b1, 2, u32 + 8);
        unrank_e8_eytz_lut(l1, 0, b2, 2, u32 + 16);
        unrank_e8_eytz_lut(r1, 1, b3, 2, u32 + 24);
        break;
    case 3: /* (B) eytz_lut3 (m<=3 L2) */
        unrank_e8_eytz_lut(l0, 0, b0, 3, u32);
        unrank_e8_eytz_lut(r0, 0, b1, 3, u32 + 8);
        unrank_e8_eytz_lut(l1, 0, b2, 3, u32 + 16);
        unrank_e8_eytz_lut(r1, 1, b3, 3, u32 + 24);
        break;
    case 4: /* (C1) d8_compact */
        unrank_e8_d8_compact(l0, 0, b0, u32);
        unrank_e8_d8_compact(r0, 0, b1, u32 + 8);
        unrank_e8_d8_compact(l1, 0, b2, u32 + 16);
        unrank_e8_d8_compact(r1, 1, b3, u32 + 24);
        break;
    default: /* (C2) d8_pshufb */
        unrank_e8_d8_pshufb(l0, 0, b0, u32);
        unrank_e8_d8_pshufb(r0, 0, b1, u32 + 8);
        unrank_e8_d8_pshufb(l1, 0, b2, u32 + 16);
        unrank_e8_d8_pshufb(r1, 1, b3, u32 + 24);
        break;
    }
    return 0u;
}

/* ============================================================================
 * Exported Hot Decode, Stage-1-Only & Adjoint GEMV Kernels (0 Hardware Divs!)
 * ============================================================================ */

size_t e8s1_decode_z_baseline(const uint8_t *in, size_t groups, int8_t *Z) {
    size_t bad = 0u;
    const size_t nb = (size_t)g_e8.bytes;
    int8_t u32[32];
    for (size_t g = 0; g < groups; ++g) {
        if (unrank_block32_variant(0, in + g * nb, u32)) { ++bad; memset(Z + g * 32u, 0, 32u); continue; }
        wht8x4_u_to_z_avx2(u32, Z + g * 32u);
    }
    return bad;
}

size_t e8s1_decode_z_guide_a(const uint8_t *in, size_t groups, int8_t *Z) {
    size_t bad = 0u;
    const size_t nb = (size_t)g_e8.bytes;
    int8_t u32[32];
    for (size_t g = 0; g < groups; ++g) {
        if (unrank_block32_variant(1, in + g * nb, u32)) { ++bad; memset(Z + g * 32u, 0, 32u); continue; }
        wht8x4_u_to_z_avx2(u32, Z + g * 32u);
    }
    return bad;
}

size_t e8s1_decode_z_eytz_lut2(const uint8_t *in, size_t groups, int8_t *Z) {
    size_t bad = 0u;
    const size_t nb = (size_t)g_e8.bytes;
    int8_t u32[32];
    for (size_t g = 0; g < groups; ++g) {
        if (unrank_block32_variant(2, in + g * nb, u32)) { ++bad; memset(Z + g * 32u, 0, 32u); continue; }
        wht8x4_u_to_z_avx2(u32, Z + g * 32u);
    }
    return bad;
}

size_t e8s1_decode_z_eytz_lut3(const uint8_t *in, size_t groups, int8_t *Z) {
    size_t bad = 0u;
    const size_t nb = (size_t)g_e8.bytes;
    int8_t u32[32];
    for (size_t g = 0; g < groups; ++g) {
        if (unrank_block32_variant(3, in + g * nb, u32)) { ++bad; memset(Z + g * 32u, 0, 32u); continue; }
        wht8x4_u_to_z_avx2(u32, Z + g * 32u);
    }
    return bad;
}

size_t e8s1_decode_z_d8_compact(const uint8_t *in, size_t groups, int8_t *Z) {
    size_t bad = 0u;
    const size_t nb = (size_t)g_e8.bytes;
    int8_t u32[32];
    for (size_t g = 0; g < groups; ++g) {
        if (unrank_block32_variant(4, in + g * nb, u32)) { ++bad; memset(Z + g * 32u, 0, 32u); continue; }
        wht8x4_u_to_z_avx2(u32, Z + g * 32u);
    }
    return bad;
}

size_t e8s1_decode_z_d8_pshufb(const uint8_t *in, size_t groups, int8_t *Z) {
    size_t bad = 0u;
    const size_t nb = (size_t)g_e8.bytes;
    int8_t u32[32];
    for (size_t g = 0; g < groups; ++g) {
        if (unrank_block32_variant(5, in + g * nb, u32)) { ++bad; memset(Z + g * 32u, 0, 32u); continue; }
        wht8x4_u_to_z_avx2(u32, Z + g * 32u);
    }
    return bad;
}

/* Direct standalone D_8^+ leaf unranker for exhaustive m=0..3 bijection verification */
void e8s1_d8_unrank_shell_vec(int m, uint64_t rk, int use_pshufb, int8_t *out8) {
    if (use_pshufb) unrank_e8_d8_pshufb(rk, 0, m, out8);
    else            unrank_e8_d8_compact(rk, 0, m, out8);
}

/* Adjoint E_8_Had GEMV helper parameterized by Stage-1 variant (0..5) */
static inline size_t gemv_adjoint_impl(
    int variant,
    const uint8_t *records,
    const float *scale_lut,
    const float *x,
    size_t rows,
    size_t cols,
    float *y_out
) {
    size_t bad = 0u;
    const size_t nb = (size_t)g_e8.bytes;
    const size_t rec_sz = nb + 1u;
    const size_t blk_per_row = cols >> 5;
    float *x_tilde = (float *)malloc(cols * sizeof(float));
    if (!x_tilde) return rows;
    wht8_f32_vector(x, x_tilde, cols);
    int8_t u32[32];
    for (size_t r = 0; r < rows; ++r) {
        float acc = 0.0f;
        const uint8_t *row_ptr = records + r * blk_per_row * rec_sz;
        for (size_t b = 0; b < blk_per_row; ++b) {
            const uint8_t *rec = row_ptr + b * rec_sz;
            float sc = scale_lut[rec[0]] * 0.5f;
            if (unrank_block32_variant(variant, rec + 1u, u32)) { ++bad; continue; }
            acc += sc * dot_i8_32_f32_avx2(u32, x_tilde + (b << 5));
        }
        y_out[r] = acc;
    }
    free(x_tilde);
    return bad;
}

size_t e8s1_gemv_adjoint_baseline(const uint8_t *rec, const float *sc, const float *x, size_t rows, size_t cols, float *y) {
    return gemv_adjoint_impl(0, rec, sc, x, rows, cols, y);
}
size_t e8s1_gemv_adjoint_guide_a(const uint8_t *rec, const float *sc, const float *x, size_t rows, size_t cols, float *y) {
    return gemv_adjoint_impl(1, rec, sc, x, rows, cols, y);
}
size_t e8s1_gemv_adjoint_eytz_lut2(const uint8_t *rec, const float *sc, const float *x, size_t rows, size_t cols, float *y) {
    return gemv_adjoint_impl(2, rec, sc, x, rows, cols, y);
}
size_t e8s1_gemv_adjoint_eytz_lut3(const uint8_t *rec, const float *sc, const float *x, size_t rows, size_t cols, float *y) {
    return gemv_adjoint_impl(3, rec, sc, x, rows, cols, y);
}
size_t e8s1_gemv_adjoint_d8_compact(const uint8_t *rec, const float *sc, const float *x, size_t rows, size_t cols, float *y) {
    return gemv_adjoint_impl(4, rec, sc, x, rows, cols, y);
}
size_t e8s1_gemv_adjoint_d8_pshufb(const uint8_t *rec, const float *sc, const float *x, size_t rows, size_t cols, float *y) {
    return gemv_adjoint_impl(5, rec, sc, x, rows, cols, y);
}

/* Materialized reference GEMV (for adjoint exactness verification) */
size_t e8s1_gemv_materialized_variant(
    int variant, const uint8_t *records, const float *scale_lut, const float *x,
    size_t rows, size_t cols, float *y_out
) {
    size_t bad = 0u;
    const size_t nb = (size_t)g_e8.bytes;
    const size_t rec_sz = nb + 1u;
    const size_t blk_per_row = cols >> 5;
    const float inv_sqrt2 = 0.7071067811865475244f;
    int8_t u32[32], z32[32];
    for (size_t r = 0; r < rows; ++r) {
        float acc = 0.0f;
        const uint8_t *row_ptr = records + r * blk_per_row * rec_sz;
        for (size_t b = 0; b < blk_per_row; ++b) {
            const uint8_t *rec = row_ptr + b * rec_sz;
            float sc = scale_lut[rec[0]] * inv_sqrt2;
            if (unrank_block32_variant(variant, rec + 1u, u32)) { ++bad; continue; }
            wht8x4_u_to_z_avx2(u32, z32);
            acc += sc * dot_i8_32_f32_avx2(z32, x + (b << 5));
        }
        y_out[r] = acc;
    }
    return bad;
}

/* Matched 4.500-bpw Q4_K_M-style AVX2/FMA GEMV baseline (18B / 32 weights) */
size_t q4km_style_gemv_avx2(
    const uint8_t *records, const float *scale_lut, const float *x,
    size_t rows, size_t cols, float *y_out
) {
    const size_t blk_per_row = cols >> 5;
    for (size_t r = 0; r < rows; ++r) {
        const uint8_t *row_ptr = records + r * blk_per_row * 18u;
#if HAS_AVX2
        __m256 row_acc = _mm256_setzero_ps();
        const __m128i mask0f = _mm_set1_epi8(0x0f);
        for (size_t b = 0; b < blk_per_row; ++b) {
            const uint8_t *rec = row_ptr + b * 18u;
            __m256 v_sc = _mm256_set1_ps(scale_lut[rec[0]]);
            __m256 v_zp = _mm256_set1_ps((float)rec[1]);
            __m128i pk = _mm_loadu_si128((const __m128i *)(rec + 2u));
            __m128i lo = _mm_and_si128(pk, mask0f);
            __m128i hi = _mm_and_si128(_mm_srli_epi16(pk, 4), mask0f);
            const float *xb = x + (b << 5);
            __m256 f0 = _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(lo)), v_zp);
            __m256 f1 = _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(lo, 8))), v_zp);
            __m256 f2 = _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(hi)), v_zp);
            __m256 f3 = _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(hi, 8))), v_zp);
            __m256 dot = _mm256_mul_ps(f0, _mm256_loadu_ps(xb + 0));
            dot = _mm256_fmadd_ps(f1, _mm256_loadu_ps(xb + 8), dot);
            dot = _mm256_fmadd_ps(f2, _mm256_loadu_ps(xb + 16), dot);
            dot = _mm256_fmadd_ps(f3, _mm256_loadu_ps(xb + 24), dot);
            row_acc = _mm256_fmadd_ps(dot, v_sc, row_acc);
        }
        y_out[r] = hsum256_ps(row_acc);
#else
        float acc = 0.0f;
        for (size_t b = 0; b < blk_per_row; ++b) {
            const uint8_t *rec = row_ptr + b * 18u;
            float sc = scale_lut[rec[0]], zp = (float)rec[1];
            const float *xb = x + (b << 5);
            float dot = 0.0f;
            for (int i = 0; i < 16; ++i) {
                uint8_t byte = rec[2 + i];
                dot += ((float)(byte & 0x0fu) - zp) * xb[i];
                dot += ((float)(byte >> 4) - zp) * xb[16 + i];
            }
            acc += sc * dot;
        }
        y_out[r] = acc;
#endif
    }
    return 0u;
}

/* ============================================================================
 * Isolated Stage-1-Only Head-to-Head Microbenchmark (On Pre-Extracted Leaves!)
 * ============================================================================ */

/* Pre-extract 4*groups exact (m, rank) leaves from packed block stream via Stage 0,
 * and report exact LUT hit counts for m <= 2 and m <= 3! */
size_t e8s1_extract_leaves_from_blocks(
    const uint8_t *in_blocks,
    size_t groups,
    uint64_t *leaf_ranks_out,
    int32_t *leaf_m_out,
    uint64_t *hits_m_le_2_out,
    uint64_t *hits_m_le_3_out,
    uint64_t *stage0_cpu_ns_out
) {
    size_t bad = 0u;
    const size_t nb = (size_t)g_e8.bytes;
    uint64_t h2 = 0u, h3 = 0u;
    uint64_t c0 = thread_cpu_ns();
    for (size_t g = 0; g < groups; ++g) {
        int m_l;
        u128 idx_l, idx_r;
        if (root_split_e8_4(in_blocks + g * nb, &m_l, &idx_l, &idx_r)) {
            ++bad;
            continue;
        }
        int a0, a1;
        uint64_t l0, r0, l1, r1;
        split_e8_blk2_shell(m_l, idx_l, &a0, &l0, &r0);
        split_e8_blk2_ball(g_e8.m_max - m_l, idx_r, &a1, &l1, &r1);
        uint64_t r3 = r1;
        int m3 = resolve_leaf_ball_u64(&r3, 1, (g_e8.m_max - m_l) - a1);
        int ms[4] = {a0, m_l - a0, a1, m3};
        uint64_t rks[4] = {l0, r0, l1, r3};
        for (int b = 0; b < 4; ++b) {
            leaf_m_out[g * 4u + (size_t)b] = ms[b];
            leaf_ranks_out[g * 4u + (size_t)b] = rks[b];
            if (ms[b] <= 2) ++h2;
            if (ms[b] <= 3) ++h3;
        }
    }
    uint64_t c1 = thread_cpu_ns();
    *hits_m_le_2_out = h2;
    *hits_m_le_3_out = h3;
    *stage0_cpu_ns_out = c1 - c0;
    return bad;
}

/* Time isolated Stage-1-only leaf unranking over n_leaves (m, rk) pairs for variant 0..5 */
uint64_t e8s1_bench_stage1_only_ns(
    int variant,
    const uint64_t *leaf_ranks,
    const int32_t *leaf_m,
    size_t n_leaves,
    int8_t *out_buf,
    uint64_t *cpu_ns_out
) {
    uint64_t w0 = now_ns();
    uint64_t c0 = thread_cpu_ns();
    switch (variant) {
    case 0:
        for (size_t i = 0; i < n_leaves; ++i) unrank_e8_baseline(leaf_ranks[i], 0, leaf_m[i], out_buf + i * 8u);
        break;
    case 1:
        for (size_t i = 0; i < n_leaves; ++i) unrank_e8_guide_a(leaf_ranks[i], 0, leaf_m[i], out_buf + i * 8u);
        break;
    case 2:
        for (size_t i = 0; i < n_leaves; ++i) unrank_e8_eytz_lut(leaf_ranks[i], 0, leaf_m[i], 2, out_buf + i * 8u);
        break;
    case 3:
        for (size_t i = 0; i < n_leaves; ++i) unrank_e8_eytz_lut(leaf_ranks[i], 0, leaf_m[i], 3, out_buf + i * 8u);
        break;
    case 4:
        for (size_t i = 0; i < n_leaves; ++i) unrank_e8_d8_compact(leaf_ranks[i], 0, leaf_m[i], out_buf + i * 8u);
        break;
    default:
        for (size_t i = 0; i < n_leaves; ++i) unrank_e8_d8_pshufb(leaf_ranks[i], 0, leaf_m[i], out_buf + i * 8u);
        break;
    }
    uint64_t c1 = thread_cpu_ns();
    uint64_t w1 = now_ns();
    *cpu_ns_out = c1 - c0;
    return w1 - w0;
}

/* Multi-threaded dual-clock GEMV benchmark harness (mode 0..5 = adjoint variants, 6 = Q4) */
typedef struct {
    int mode;
    const uint8_t *records;
    size_t rec_sz;
    const float *scale_lut;
    const float *x;
    size_t rows;
    size_t cols;
    float *y_out;
    size_t bad;
    uint64_t cpu_ns;
} GemvTask;

static void *gemv_worker(void *arg) {
    GemvTask *t = (GemvTask *)arg;
    uint64_t c0 = thread_cpu_ns();
    if (t->mode == 6) {
        t->bad = q4km_style_gemv_avx2(t->records, t->scale_lut, t->x, t->rows, t->cols, t->y_out);
    } else {
        t->bad = gemv_adjoint_impl(t->mode, t->records, t->scale_lut, t->x, t->rows, t->cols, t->y_out);
    }
    t->cpu_ns = thread_cpu_ns() - c0;
    return NULL;
}

uint64_t e8s1_bench_gemv_ns(
    int mode, int threads, const uint8_t *records, size_t rec_sz,
    const float *scale_lut, const float *x, size_t rows, size_t cols,
    float *y_out, size_t *bad_out, uint64_t *cpu_ns_out
) {
    if (threads < 1) threads = 1;
    if (threads > 16) threads = 16;
    if (threads == 1) {
        GemvTask task = {mode, records, rec_sz, scale_lut, x, rows, cols, y_out, 0u, 0u};
        uint64_t w0 = now_ns();
        gemv_worker(&task);
        uint64_t w1 = now_ns();
        *bad_out = task.bad;
        *cpu_ns_out = task.cpu_ns;
        return w1 - w0;
    }
    pthread_t th[16];
    GemvTask tasks[16];
    size_t blk_per_row = cols >> 5;
    size_t base = rows / (size_t)threads;
    size_t rem = rows % (size_t)threads;
    size_t off = 0u;
    uint64_t w0 = now_ns();
    for (int i = 0; i < threads; ++i) {
        size_t n = base + ((size_t)i < rem ? 1u : 0u);
        tasks[i].mode = mode;
        tasks[i].records = records + off * blk_per_row * rec_sz;
        tasks[i].rec_sz = rec_sz;
        tasks[i].scale_lut = scale_lut;
        tasks[i].x = x;
        tasks[i].rows = n;
        tasks[i].cols = cols;
        tasks[i].y_out = y_out + off;
        tasks[i].bad = 0u;
        tasks[i].cpu_ns = 0u;
        off += n;
        pthread_create(&th[i], NULL, gemv_worker, &tasks[i]);
    }
    size_t bad = 0u;
    uint64_t cpu_sum = 0u;
    for (int i = 0; i < threads; ++i) {
        pthread_join(th[i], NULL);
        bad += tasks[i].bad;
        cpu_sum += tasks[i].cpu_ns;
    }
    uint64_t w1 = now_ns();
    *bad_out = bad;
    *cpu_ns_out = cpu_sum;
    return w1 - w0;
}
