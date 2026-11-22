/* Zero-division binary-tree shell-convolution unrankers & rankers in pure C11
 * for Track S15b (AUDIT-112 empirical companion to Moissanite-Labs/siliquant#23/#25).
 *
 * Implements:
 *   1. Z^32 (136-bit, B=783) zero-division binary-tree bijection (32->16->8->4->2->L1 LUT):
 *      - ztree_unrank_compact      (70,780 B 1D tables + Z^2 L1 LUT)
 *      - ztree_unrank_fast         (L2-resident strided checkpoint prefix sums)
 *      - ztree_unrank_fast_batch4  (4-block interleaved ILP unrolling)
 *      - ztree_rank                (exact inverse bijection)
 *   2. E_8^4 (136-bit m=391 & 88-bit m=48) zero-division reduced-norm binary-tree codec:
 *      - e8tree_unrank_compact     (56,850 B @ 136b / 5,490 B @ 88b)
 *      - e8tree_unrank_fast        (L2-resident strided checkpoints, depth 2 -> 4 E_8 blocks)
 *      - e8tree_rank               (exact inverse bijection in doubled coordinates u = 2y)
 *   3. Built-in 1-core and multi-core (pthread) CLOCK_MONOTONIC microbenchmark harness.
 *
 * Zero hardware division instructions on the hot unranking path: every split uses
 * precomputed exact Granlund-Montgomery / Barrett multiply-shift pairs verified by AUDIT-112.
 *
 * Build (0 warnings under gcc & clang):
 *   cc -O3 -march=native -Wall -Wextra -Werror -std=c11 -shared -fPIC -pthread -o libballpack_tree.so ballpack_tree.c
 */
#define _POSIX_C_SOURCE 200809L

#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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

/* ============================================================================
 * Zero-Division Exact Multiply-Shift Helpers
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

static inline u192 u192_add(const u192 *a, const u192 *b) {
    u192 out;
    u128 carry = 0;
    for (int i = 0; i < 3; ++i) {
        u128 s = (u128)a->w[i] + (u128)b->w[i] + carry;
        out.w[i] = (uint64_t)s;
        carry = s >> 64;
    }
    return out;
}

static inline u192 u128_mul_to_u192(u128 a, u128 b) {
    uint64_t a0 = (uint64_t)a, a1 = (uint64_t)(a >> 64);
    uint64_t b0 = (uint64_t)b, b1 = (uint64_t)(b >> 64);
    u128 p00 = (u128)a0 * b0;
    u128 p01 = (u128)a0 * b1;
    u128 p10 = (u128)a1 * b0;
    u128 p11 = (u128)a1 * b1;
    u192 out;
    out.w[0] = (uint64_t)p00;
    u128 mid = (p00 >> 64) + (uint64_t)p01 + (uint64_t)p10;
    out.w[1] = (uint64_t)mid;
    u128 hi = p11 + (p01 >> 64) + (p10 >> 64) + (mid >> 64);
    out.w[2] = (uint64_t)hi;
    return out;
}

/* Exact floor((a * b) / 2^shift) -> u128 for 192-bit a, b (0 hardware divisions!) */
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

/* Exact floor((a * b) / 2^shift) -> uint64_t for 128-bit a, b (0 hardware divisions!) */
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

static inline void store_u192_le(const u192 *v, uint8_t *out, int nbytes) {
    for (int i = 0; i < nbytes; ++i) {
        out[i] = (uint8_t)(v->w[i >> 3] >> (8 * (i & 7)));
    }
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
 * PART 1: Z^32 (B = 783, 136-bit) Binary-Tree Shell-Convolution Codec
 * ============================================================================ */

#define Z16_STRIDE 32
#define Z8_STRIDE  16
#define Z4_STRIDE  8

typedef struct {
    int bound;
    int bytes;
    int z2_pairs;
    int z2_nonzero_count;
    u192 total_count;
    u192 *c_root;          /* [bound + 2] */
    u128 *s16;             /* [bound + 1] */
    u128 *w16;             /* [bound + 1] */
    uint64_t *s8;          /* [bound + 1] */
    uint64_t *w8;          /* [bound + 1] */
    uint32_t *s4;          /* [bound + 1] */
    uint32_t *w4;          /* [bound + 1] */
    uint16_t *s2;          /* [bound + 1] */
    uint16_t *w2;          /* [bound + 1] */
    uint16_t *z2_off;      /* [bound + 2] */
    int8_t *z2_pts;        /* [2 * z2_pairs] */
    uint16_t *z2_nz_r;     /* [z2_nonzero_count] non-zero shells of Z^2 */
    /* Inverse LUT for O(1) Z^2 ranking: indexed by (x + max_c) * side + (y + max_c) */
    int z2_max_c;
    int z2_side;
    uint16_t *z2_ball_idx;
    uint16_t *z2_shell_idx;
    /* Exact zero-division reciprocals */
    Recip192 *rec_root;    /* [bound + 1] divisors W_16[b_R] */
    Recip128 *rec_b8;      /* [bound + 1] divisors W_8[b_R] */
    Recip128 *rec_s8;      /* [bound + 1] divisors S_8[r_R] */
    Recip64 *rec_b4;       /* [bound + 1] divisors W_4[b_R] */
    Recip64 *rec_s4;       /* [bound + 1] divisors S_4[r_R] */
    Recip32 *rec_b2;       /* [bound + 1] divisors W_2[b_R] */
    Recip32 *rec_s2;       /* [bound + 1] divisors S_2[r_R] */
    /* Level-tuned L2-resident strided checkpoint prefix sums (Z16=32, Z8=16, Z4=8) */
    uint32_t *cp16_off;    /* [bound + 1] */
    uint32_t *cp16_len;    /* [bound + 1] */
    uint32_t *cp8_off;     /* [bound + 1] */
    uint32_t *cp8_len;     /* [bound + 1] */
    uint32_t *cp4_off;     /* [bound + 1] */
    uint32_t *cp4_len;     /* [bound + 1] */
    u128 *cp_s16;          /* strided prefix sums for Shell(16, r) */
    u128 *cp_w16;          /* strided prefix sums for Ball(16, b) */
    uint64_t *cp_s8;       /* strided prefix sums for Shell(8, r) */
    uint64_t *cp_w8;       /* strided prefix sums for Ball(8, b) */
    uint32_t *cp_s4;       /* strided prefix sums for Shell(4, r) */
    uint32_t *cp_w4;       /* strided prefix sums for Ball(4, b) */
    size_t checkpoint_bytes;
} Z32Ctx;

static Z32Ctx g_z32;
static int g_z32_ready = 0;

static void z32_free(void) {
    if (!g_z32_ready) return;
    free(g_z32.c_root);
    free(g_z32.s16);
    free(g_z32.w16);
    free(g_z32.s8);
    free(g_z32.w8);
    free(g_z32.s4);
    free(g_z32.w4);
    free(g_z32.s2);
    free(g_z32.w2);
    free(g_z32.z2_off);
    free(g_z32.z2_pts);
    free(g_z32.z2_nz_r);
    free(g_z32.z2_ball_idx);
    free(g_z32.z2_shell_idx);
    free(g_z32.rec_root);
    free(g_z32.rec_b8);
    free(g_z32.rec_s8);
    free(g_z32.rec_b4);
    free(g_z32.rec_s4);
    free(g_z32.rec_b2);
    free(g_z32.rec_s2);
    free(g_z32.cp16_off);
    free(g_z32.cp16_len);
    free(g_z32.cp8_off);
    free(g_z32.cp8_len);
    free(g_z32.cp4_off);
    free(g_z32.cp4_len);
    free(g_z32.cp_s16);
    free(g_z32.cp_w16);
    free(g_z32.cp_s8);
    free(g_z32.cp_w8);
    free(g_z32.cp_s4);
    free(g_z32.cp_w4);
    memset(&g_z32, 0, sizeof(g_z32));
    g_z32_ready = 0;
}

int ztree_init(
    int bound,
    int bytes,
    int z2_pairs,
    const uint64_t *c_root_limbs,  /* (bound + 2) * 3 */
    const uint64_t *s16_limbs,     /* (bound + 1) * 2 */
    const uint64_t *w16_limbs,     /* (bound + 1) * 2 */
    const uint64_t *s8_in,
    const uint64_t *w8_in,
    const uint32_t *s4_in,
    const uint32_t *w4_in,
    const uint16_t *s2_in,
    const uint16_t *w2_in,
    const uint16_t *z2_off_in,     /* bound + 2 */
    const int8_t *z2_pts_in,       /* 2 * z2_pairs */
    const uint64_t *rec_root_m,    /* (bound + 1) * 3 */
    const uint32_t *rec_root_s,
    const uint64_t *rec_b8_m,      /* (bound + 1) * 2 */
    const uint32_t *rec_b8_s,
    const uint64_t *rec_s8_m,      /* (bound + 1) * 2 */
    const uint32_t *rec_s8_s,
    const uint64_t *rec_b4_m,
    const uint32_t *rec_b4_s,
    const uint64_t *rec_s4_m,
    const uint32_t *rec_s4_s,
    const uint32_t *rec_b2_m,
    const uint32_t *rec_b2_s,
    const uint32_t *rec_s2_m,
    const uint32_t *rec_s2_s
) {
    if (bound < 0 || bound > 4096 || bytes < 1 || bytes > 24 || z2_pairs < 1) return 1;
    z32_free();
    size_t B1 = (size_t)bound + 1u;
    size_t B2 = (size_t)bound + 2u;

    g_z32.bound = bound;
    g_z32.bytes = bytes;
    g_z32.z2_pairs = z2_pairs;
    g_z32.c_root = (u192 *)malloc(B2 * sizeof(u192));
    g_z32.s16 = (u128 *)malloc(B1 * sizeof(u128));
    g_z32.w16 = (u128 *)malloc(B1 * sizeof(u128));
    g_z32.s8 = (uint64_t *)malloc(B1 * sizeof(uint64_t));
    g_z32.w8 = (uint64_t *)malloc(B1 * sizeof(uint64_t));
    g_z32.s4 = (uint32_t *)malloc(B1 * sizeof(uint32_t));
    g_z32.w4 = (uint32_t *)malloc(B1 * sizeof(uint32_t));
    g_z32.s2 = (uint16_t *)malloc(B1 * sizeof(uint16_t));
    g_z32.w2 = (uint16_t *)malloc(B1 * sizeof(uint16_t));
    g_z32.z2_off = (uint16_t *)malloc(B2 * sizeof(uint16_t));
    g_z32.z2_pts = (int8_t *)malloc((size_t)z2_pairs * 2u * sizeof(int8_t));
    g_z32.z2_nz_r = (uint16_t *)malloc(B1 * sizeof(uint16_t));
    g_z32.rec_root = (Recip192 *)calloc(B1, sizeof(Recip192));
    g_z32.rec_b8 = (Recip128 *)calloc(B1, sizeof(Recip128));
    g_z32.rec_s8 = (Recip128 *)calloc(B1, sizeof(Recip128));
    g_z32.rec_b4 = (Recip64 *)calloc(B1, sizeof(Recip64));
    g_z32.rec_s4 = (Recip64 *)calloc(B1, sizeof(Recip64));
    g_z32.rec_b2 = (Recip32 *)calloc(B1, sizeof(Recip32));
    g_z32.rec_s2 = (Recip32 *)calloc(B1, sizeof(Recip32));
    g_z32.cp16_off = (uint32_t *)malloc(B1 * sizeof(uint32_t));
    g_z32.cp16_len = (uint32_t *)malloc(B1 * sizeof(uint32_t));
    g_z32.cp8_off = (uint32_t *)malloc(B1 * sizeof(uint32_t));
    g_z32.cp8_len = (uint32_t *)malloc(B1 * sizeof(uint32_t));
    g_z32.cp4_off = (uint32_t *)malloc(B1 * sizeof(uint32_t));
    g_z32.cp4_len = (uint32_t *)malloc(B1 * sizeof(uint32_t));

    for (size_t i = 0; i < B2; ++i) {
        g_z32.c_root[i].w[0] = c_root_limbs[3u * i + 0u];
        g_z32.c_root[i].w[1] = c_root_limbs[3u * i + 1u];
        g_z32.c_root[i].w[2] = c_root_limbs[3u * i + 2u];
    }
    g_z32.total_count = g_z32.c_root[bound + 1];

    int nz = 0;
    for (size_t i = 0; i < B1; ++i) {
        g_z32.s16[i] = ((u128)s16_limbs[2u * i + 1u] << 64) | s16_limbs[2u * i];
        g_z32.w16[i] = ((u128)w16_limbs[2u * i + 1u] << 64) | w16_limbs[2u * i];
        g_z32.s8[i] = s8_in[i];
        g_z32.w8[i] = w8_in[i];
        g_z32.s4[i] = s4_in[i];
        g_z32.w4[i] = w4_in[i];
        g_z32.s2[i] = s2_in[i];
        g_z32.w2[i] = w2_in[i];
        if (s2_in[i] > 0u) g_z32.z2_nz_r[nz++] = (uint16_t)i;
        g_z32.rec_root[i].m[0] = rec_root_m[3u * i + 0u];
        g_z32.rec_root[i].m[1] = rec_root_m[3u * i + 1u];
        g_z32.rec_root[i].m[2] = rec_root_m[3u * i + 2u];
        g_z32.rec_root[i].shift = rec_root_s[i];
        g_z32.rec_b8[i].m = ((u128)rec_b8_m[2u * i + 1u] << 64) | rec_b8_m[2u * i];
        g_z32.rec_b8[i].shift = rec_b8_s[i];
        g_z32.rec_s8[i].m = ((u128)rec_s8_m[2u * i + 1u] << 64) | rec_s8_m[2u * i];
        g_z32.rec_s8[i].shift = rec_s8_s[i];
        g_z32.rec_b4[i].m = rec_b4_m[i];
        g_z32.rec_b4[i].shift = rec_b4_s[i];
        g_z32.rec_s4[i].m = rec_s4_m[i];
        g_z32.rec_s4[i].shift = rec_s4_s[i];
        g_z32.rec_b2[i].m = rec_b2_m[i];
        g_z32.rec_b2[i].shift = rec_b2_s[i];
        g_z32.rec_s2[i].m = rec_s2_m[i];
        g_z32.rec_s2[i].shift = rec_s2_s[i];
    }
    g_z32.z2_nonzero_count = nz;
    memcpy(g_z32.z2_off, z2_off_in, B2 * sizeof(uint16_t));
    memcpy(g_z32.z2_pts, z2_pts_in, (size_t)z2_pairs * 2u * sizeof(int8_t));

    /* Inverse Z^2 lookup tables for exact O(1) ranking */
    int mc = 0;
    while ((mc + 1) * (mc + 1) <= bound) ++mc;
    g_z32.z2_max_c = mc;
    g_z32.z2_side = 2 * mc + 1;
    size_t grid_n = (size_t)g_z32.z2_side * (size_t)g_z32.z2_side;
    g_z32.z2_ball_idx = (uint16_t *)malloc(grid_n * sizeof(uint16_t));
    g_z32.z2_shell_idx = (uint16_t *)malloc(grid_n * sizeof(uint16_t));
    memset(g_z32.z2_ball_idx, 0xff, grid_n * sizeof(uint16_t));
    memset(g_z32.z2_shell_idx, 0xff, grid_n * sizeof(uint16_t));
    for (int r = 0; r <= bound; ++r) {
        uint16_t start = g_z32.z2_off[r];
        uint16_t end = g_z32.z2_off[r + 1];
        for (uint16_t bi = start; bi < end; ++bi) {
            int x = g_z32.z2_pts[2u * bi];
            int y = g_z32.z2_pts[2u * bi + 1u];
            size_t pos = (size_t)(x + mc) * (size_t)g_z32.z2_side + (size_t)(y + mc);
            g_z32.z2_ball_idx[pos] = bi;
            g_z32.z2_shell_idx[pos] = (uint16_t)(bi - start);
        }
    }

    /* Build level-tuned strided checkpoint prefix sums (Z16=32, Z8=16, Z4=8) */
    uint32_t tot16 = 0, tot8 = 0, tot4 = 0;
    for (int b = 0; b <= bound; ++b) {
        uint32_t n16 = (uint32_t)(b / Z16_STRIDE) + 1u;
        uint32_t n8 = (uint32_t)(b / Z8_STRIDE) + 1u;
        uint32_t n4 = (uint32_t)(b / Z4_STRIDE) + 1u;
        g_z32.cp16_off[b] = tot16; g_z32.cp16_len[b] = n16; tot16 += n16;
        g_z32.cp8_off[b] = tot8;   g_z32.cp8_len[b] = n8;   tot8 += n8;
        g_z32.cp4_off[b] = tot4;   g_z32.cp4_len[b] = n4;   tot4 += n4;
    }
    g_z32.cp_s16 = (u128 *)malloc((size_t)tot16 * sizeof(u128));
    g_z32.cp_w16 = (u128 *)malloc((size_t)tot16 * sizeof(u128));
    g_z32.cp_s8 = (uint64_t *)malloc((size_t)tot8 * sizeof(uint64_t));
    g_z32.cp_w8 = (uint64_t *)malloc((size_t)tot8 * sizeof(uint64_t));
    g_z32.cp_s4 = (uint32_t *)malloc((size_t)tot4 * sizeof(uint32_t));
    g_z32.cp_w4 = (uint32_t *)malloc((size_t)tot4 * sizeof(uint32_t));
    for (int b = 0; b <= bound; ++b) {
        uint32_t b16 = g_z32.cp16_off[b];
        uint32_t b8 = g_z32.cp8_off[b];
        uint32_t b4 = g_z32.cp4_off[b];
        u128 as16 = 0, aw16 = 0;
        uint64_t as8 = 0, aw8 = 0;
        uint32_t as4 = 0, aw4 = 0;
        for (int r_l = 0; r_l <= b; ++r_l) {
            if ((r_l & (Z16_STRIDE - 1)) == 0) {
                uint32_t s = b16 + (uint32_t)(r_l / Z16_STRIDE);
                g_z32.cp_s16[s] = as16;
                g_z32.cp_w16[s] = aw16;
            }
            if ((r_l & (Z8_STRIDE - 1)) == 0) {
                uint32_t s = b8 + (uint32_t)(r_l / Z8_STRIDE);
                g_z32.cp_s8[s] = as8;
                g_z32.cp_w8[s] = aw8;
            }
            if ((r_l & (Z4_STRIDE - 1)) == 0) {
                uint32_t s = b4 + (uint32_t)(r_l / Z4_STRIDE);
                g_z32.cp_s4[s] = as4;
                g_z32.cp_w4[s] = aw4;
            }
            int rem_b = b - r_l;
            as16 += (u128)g_z32.s8[r_l] * g_z32.s8[rem_b];
            aw16 += (u128)g_z32.s8[r_l] * g_z32.w8[rem_b];
            as8 += (uint64_t)g_z32.s4[r_l] * g_z32.s4[rem_b];
            aw8 += (uint64_t)g_z32.s4[r_l] * g_z32.w4[rem_b];
            as4 += (uint32_t)g_z32.s2[r_l] * g_z32.s2[rem_b];
            aw4 += (uint32_t)g_z32.s2[r_l] * g_z32.w2[rem_b];
        }
    }
    g_z32.checkpoint_bytes = (size_t)tot16 * 32u + (size_t)tot8 * 16u + (size_t)tot4 * 8u;
    g_z32_ready = 1;
    return 0;
}

size_t ztree_checkpoint_bytes(void) {
    return g_z32_ready ? g_z32.checkpoint_bytes : 0u;
}

/* --- Leaf Z^2 emission (O(1) L1 LUT load!) --- */
static inline void emit_z2_shell(int r, uint32_t idx, int8_t *out) {
    uint32_t pos = (uint32_t)g_z32.z2_off[r] + idx;
    out[0] = g_z32.z2_pts[2u * pos];
    out[1] = g_z32.z2_pts[2u * pos + 1u];
}

static inline void emit_z2_ball(uint32_t idx, int8_t *out) {
    out[0] = g_z32.z2_pts[2u * idx];
    out[1] = g_z32.z2_pts[2u * idx + 1u];
}

/* --- Compact 1D-table node unrankers (70.8 KB footprint, 0 hardware divisions!) --- */
static inline void unrank_z4_shell_compact(int r, uint32_t rem, int8_t *out) {
    const uint16_t *s2 = g_z32.s2;
    int r_l = 0;
    for (; r_l <= r; ++r_l) {
        uint32_t term = (uint32_t)s2[r_l] * s2[r - r_l];
        if (rem < term) break;
        rem -= term;
    }
    int r_r = r - r_l;
    Recip32 rec = g_z32.rec_s2[r_r];
    uint32_t d = s2[r_r];
    uint32_t idx_l = mul_shr_u32_to_u32(rem, rec.m, rec.shift);
    uint32_t idx_r = rem - idx_l * d;
    emit_z2_shell(r_l, idx_l, out);
    emit_z2_shell(r_r, idx_r, out + 2);
}

static inline void unrank_z4_ball_compact(int b, uint32_t rem, int8_t *out) {
    const uint16_t *s2 = g_z32.s2;
    const uint16_t *w2 = g_z32.w2;
    int r_l = 0;
    for (; r_l <= b; ++r_l) {
        uint32_t term = (uint32_t)s2[r_l] * w2[b - r_l];
        if (rem < term) break;
        rem -= term;
    }
    int b_r = b - r_l;
    Recip32 rec = g_z32.rec_b2[b_r];
    uint32_t d = w2[b_r];
    uint32_t idx_l = mul_shr_u32_to_u32(rem, rec.m, rec.shift);
    uint32_t idx_r = rem - idx_l * d;
    emit_z2_shell(r_l, idx_l, out);
    emit_z2_ball(idx_r, out + 2);
}

static inline void unrank_z8_shell_compact(int r, uint64_t rem, int8_t *out) {
    const uint32_t *s4 = g_z32.s4;
    int r_l = 0;
    for (; r_l <= r; ++r_l) {
        uint64_t term = (uint64_t)s4[r_l] * s4[r - r_l];
        if (rem < term) break;
        rem -= term;
    }
    int r_r = r - r_l;
    Recip64 rec = g_z32.rec_s4[r_r];
    uint64_t d = s4[r_r];
    uint32_t idx_l = (uint32_t)mul_shr_u64_to_u64(rem, rec.m, rec.shift);
    uint32_t idx_r = (uint32_t)(rem - (uint64_t)idx_l * d);
    unrank_z4_shell_compact(r_l, idx_l, out);
    unrank_z4_shell_compact(r_r, idx_r, out + 4);
}

static inline void unrank_z8_ball_compact(int b, uint64_t rem, int8_t *out) {
    const uint32_t *s4 = g_z32.s4;
    const uint32_t *w4 = g_z32.w4;
    int r_l = 0;
    for (; r_l <= b; ++r_l) {
        uint64_t term = (uint64_t)s4[r_l] * w4[b - r_l];
        if (rem < term) break;
        rem -= term;
    }
    int b_r = b - r_l;
    Recip64 rec = g_z32.rec_b4[b_r];
    uint64_t d = w4[b_r];
    uint32_t idx_l = (uint32_t)mul_shr_u64_to_u64(rem, rec.m, rec.shift);
    uint32_t idx_r = (uint32_t)(rem - (uint64_t)idx_l * d);
    unrank_z4_shell_compact(r_l, idx_l, out);
    unrank_z4_ball_compact(b_r, idx_r, out + 4);
}

static inline void unrank_z16_shell_compact(int r, u128 rem, int8_t *out) {
    const uint64_t *s8 = g_z32.s8;
    int r_l = 0;
    for (; r_l <= r; ++r_l) {
        u128 term = (u128)s8[r_l] * s8[r - r_l];
        if (rem < term) break;
        rem -= term;
    }
    int r_r = r - r_l;
    Recip128 rec = g_z32.rec_s8[r_r];
    uint64_t d = s8[r_r];
    uint64_t idx_l = mul_shr_u128_to_u64(rem, rec.m, rec.shift);
    uint64_t idx_r = (uint64_t)rem - idx_l * d;
    unrank_z8_shell_compact(r_l, idx_l, out);
    unrank_z8_shell_compact(r_r, idx_r, out + 8);
}

static inline void unrank_z16_ball_compact(int b, u128 rem, int8_t *out) {
    const uint64_t *s8 = g_z32.s8;
    const uint64_t *w8 = g_z32.w8;
    int r_l = 0;
    for (; r_l <= b; ++r_l) {
        u128 term = (u128)s8[r_l] * w8[b - r_l];
        if (rem < term) break;
        rem -= term;
    }
    int b_r = b - r_l;
    Recip128 rec = g_z32.rec_b8[b_r];
    uint64_t d = w8[b_r];
    uint64_t idx_l = mul_shr_u128_to_u64(rem, rec.m, rec.shift);
    uint64_t idx_r = (uint64_t)rem - idx_l * d;
    unrank_z8_shell_compact(r_l, idx_l, out);
    unrank_z8_ball_compact(b_r, idx_r, out + 8);
}

static inline int root_split_z32(const uint8_t *in, int *r_l_out, u128 *idx_l_out, u128 *idx_r_out) {
    u192 rk = load_u192_le(in, g_z32.bytes);
    if (u192_ge(&rk, &g_z32.total_count)) return 1;
    int lo = 0, hi = g_z32.bound;
    const u192 *c_root = g_z32.c_root;
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (u192_le(&c_root[mid], &rk)) lo = mid;
        else hi = mid - 1;
    }
    int r_l = lo;
    int b_r = g_z32.bound - r_l;
    u192 rem = u192_sub(&rk, &c_root[r_l]);
    const Recip192 *rec = &g_z32.rec_root[b_r];
    u128 q = mul_shr_u192_to_u128(&rem, rec->m, rec->shift);
    u128 rem_lo128 = ((u128)rem.w[1] << 64) | rem.w[0];
    u128 r = rem_lo128 - q * g_z32.w16[b_r];
    *r_l_out = r_l;
    *idx_l_out = q;
    *idx_r_out = r;
    return 0;
}

size_t ztree_unrank_compact(const uint8_t *in, size_t groups, int8_t *Y) {
    size_t bad = 0;
    const int B = g_z32.bound;
    const int nb = g_z32.bytes;
    for (size_t g = 0; g < groups; ++g) {
        int r_l;
        u128 idx_l, idx_r;
        int8_t *out = Y + g * 32u;
        if (root_split_z32(in + g * (size_t)nb, &r_l, &idx_l, &idx_r)) {
            ++bad;
            memset(out, 0, 32u);
            continue;
        }
        unrank_z16_shell_compact(r_l, idx_l, out);
        unrank_z16_ball_compact(B - r_l, idx_r, out + 16);
    }
    return bad;
}

/* --- Fast branchless strided binary-search node unrankers (0 hardware divisions!) --- */
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

static inline void unrank_z4_shell_fast(int r, uint32_t rem, int8_t *out) {
    const uint16_t *s2 = g_z32.s2;
    const uint32_t *cp = g_z32.cp_s4 + g_z32.cp4_off[r];
    uint32_t blk = find_cp_u32(cp, g_z32.cp4_len[r], rem);
    rem -= cp[blk];
    int r_l = (int)(blk * Z4_STRIDE);
    for (; r_l <= r; ++r_l) {
        uint32_t term = (uint32_t)s2[r_l] * s2[r - r_l];
        if (rem < term) break;
        rem -= term;
    }
    int r_r = r - r_l;
    Recip32 rec = g_z32.rec_s2[r_r];
    uint32_t d = s2[r_r];
    uint32_t idx_l = mul_shr_u32_to_u32(rem, rec.m, rec.shift);
    uint32_t idx_r = rem - idx_l * d;
    emit_z2_shell(r_l, idx_l, out);
    emit_z2_shell(r_r, idx_r, out + 2);
}

static inline void unrank_z4_ball_fast(int b, uint32_t rem, int8_t *out) {
    const uint16_t *s2 = g_z32.s2;
    const uint16_t *w2 = g_z32.w2;
    const uint32_t *cp = g_z32.cp_w4 + g_z32.cp4_off[b];
    uint32_t blk = find_cp_u32(cp, g_z32.cp4_len[b], rem);
    rem -= cp[blk];
    int r_l = (int)(blk * Z4_STRIDE);
    for (; r_l <= b; ++r_l) {
        uint32_t term = (uint32_t)s2[r_l] * w2[b - r_l];
        if (rem < term) break;
        rem -= term;
    }
    int b_r = b - r_l;
    Recip32 rec = g_z32.rec_b2[b_r];
    uint32_t d = w2[b_r];
    uint32_t idx_l = mul_shr_u32_to_u32(rem, rec.m, rec.shift);
    uint32_t idx_r = rem - idx_l * d;
    emit_z2_shell(r_l, idx_l, out);
    emit_z2_ball(idx_r, out + 2);
}

static inline void unrank_z8_shell_fast(int r, uint64_t rem, int8_t *out) {
    const uint32_t *s4 = g_z32.s4;
    const uint64_t *cp = g_z32.cp_s8 + g_z32.cp8_off[r];
    uint32_t blk = find_cp_u64(cp, g_z32.cp8_len[r], rem);
    rem -= cp[blk];
    int r_l = (int)(blk * Z8_STRIDE);
    for (; r_l <= r; ++r_l) {
        uint64_t term = (uint64_t)s4[r_l] * s4[r - r_l];
        if (rem < term) break;
        rem -= term;
    }
    int r_r = r - r_l;
    Recip64 rec = g_z32.rec_s4[r_r];
    uint64_t d = s4[r_r];
    uint32_t idx_l = (uint32_t)mul_shr_u64_to_u64(rem, rec.m, rec.shift);
    uint32_t idx_r = (uint32_t)(rem - (uint64_t)idx_l * d);
    unrank_z4_shell_fast(r_l, idx_l, out);
    unrank_z4_shell_fast(r_r, idx_r, out + 4);
}

static inline void unrank_z8_ball_fast(int b, uint64_t rem, int8_t *out) {
    const uint32_t *s4 = g_z32.s4;
    const uint32_t *w4 = g_z32.w4;
    const uint64_t *cp = g_z32.cp_w8 + g_z32.cp8_off[b];
    uint32_t blk = find_cp_u64(cp, g_z32.cp8_len[b], rem);
    rem -= cp[blk];
    int r_l = (int)(blk * Z8_STRIDE);
    for (; r_l <= b; ++r_l) {
        uint64_t term = (uint64_t)s4[r_l] * w4[b - r_l];
        if (rem < term) break;
        rem -= term;
    }
    int b_r = b - r_l;
    Recip64 rec = g_z32.rec_b4[b_r];
    uint64_t d = w4[b_r];
    uint32_t idx_l = (uint32_t)mul_shr_u64_to_u64(rem, rec.m, rec.shift);
    uint32_t idx_r = (uint32_t)(rem - (uint64_t)idx_l * d);
    unrank_z4_shell_fast(r_l, idx_l, out);
    unrank_z4_ball_fast(b_r, idx_r, out + 4);
}

static inline void split_z16_shell_fast(int r, u128 rem, int *r_l_out, uint64_t *idx_l_out, uint64_t *idx_r_out) {
    const uint64_t *s8 = g_z32.s8;
    const u128 *cp = g_z32.cp_s16 + g_z32.cp16_off[r];
    uint32_t blk = find_cp_u128(cp, g_z32.cp16_len[r], rem);
    rem -= cp[blk];
    int r_l = (int)(blk * Z16_STRIDE);
    for (; r_l <= r; ++r_l) {
        u128 term = (u128)s8[r_l] * s8[r - r_l];
        if (rem < term) break;
        rem -= term;
    }
    int r_r = r - r_l;
    Recip128 rec = g_z32.rec_s8[r_r];
    uint64_t d = s8[r_r];
    uint64_t idx_l = mul_shr_u128_to_u64(rem, rec.m, rec.shift);
    uint64_t idx_r = (uint64_t)rem - idx_l * d;
    *r_l_out = r_l;
    *idx_l_out = idx_l;
    *idx_r_out = idx_r;
}

static inline void split_z16_ball_fast(int b, u128 rem, int *r_l_out, uint64_t *idx_l_out, uint64_t *idx_r_out) {
    const uint64_t *s8 = g_z32.s8;
    const uint64_t *w8 = g_z32.w8;
    const u128 *cp = g_z32.cp_w16 + g_z32.cp16_off[b];
    uint32_t blk = find_cp_u128(cp, g_z32.cp16_len[b], rem);
    rem -= cp[blk];
    int r_l = (int)(blk * Z16_STRIDE);
    for (; r_l <= b; ++r_l) {
        u128 term = (u128)s8[r_l] * w8[b - r_l];
        if (rem < term) break;
        rem -= term;
    }
    int b_r = b - r_l;
    Recip128 rec = g_z32.rec_b8[b_r];
    uint64_t d = w8[b_r];
    uint64_t idx_l = mul_shr_u128_to_u64(rem, rec.m, rec.shift);
    uint64_t idx_r = (uint64_t)rem - idx_l * d;
    *r_l_out = r_l;
    *idx_l_out = idx_l;
    *idx_r_out = idx_r;
}

size_t ztree_unrank_fast(const uint8_t *in, size_t groups, int8_t *Y) {
    size_t bad = 0;
    const int B = g_z32.bound;
    const int nb = g_z32.bytes;
    for (size_t g = 0; g < groups; ++g) {
        int r_l;
        u128 idx_l, idx_r;
        int8_t *out = Y + g * 32u;
        if (root_split_z32(in + g * (size_t)nb, &r_l, &idx_l, &idx_r)) {
            ++bad;
            memset(out, 0, 32u);
            continue;
        }
        int a0, a1;
        uint64_t l0, r0, l1, r1;
        split_z16_shell_fast(r_l, idx_l, &a0, &l0, &r0);
        split_z16_ball_fast(B - r_l, idx_r, &a1, &l1, &r1);
        unrank_z8_shell_fast(a0, l0, out);
        unrank_z8_shell_fast(r_l - a0, r0, out + 8);
        unrank_z8_shell_fast(a1, l1, out + 16);
        unrank_z8_ball_fast((B - r_l) - a1, r1, out + 24);
    }
    return bad;
}

/* 4-block level-by-level ILP interleaved unranker (identical bijection!) */
size_t ztree_unrank_fast_batch4(const uint8_t *in, size_t groups, int8_t *Y) {
    size_t bad = 0;
    const int B = g_z32.bound;
    const int nb = g_z32.bytes;
    size_t g = 0;
    for (; g + 4u <= groups; g += 4u) {
        int r_l[4], ok[4];
        u128 idx_l[4], idx_r[4];
        for (int k = 0; k < 4; ++k) {
            ok[k] = !root_split_z32(in + (g + (size_t)k) * (size_t)nb, &r_l[k], &idx_l[k], &idx_r[k]);
            if (!ok[k]) {
                ++bad;
                memset(Y + (g + (size_t)k) * 32u, 0, 32u);
            }
        }
        int a0[4], a1[4];
        uint64_t l0[4], r0[4], l1[4], r1[4];
        for (int k = 0; k < 4; ++k) {
            if (!ok[k]) continue;
            split_z16_shell_fast(r_l[k], idx_l[k], &a0[k], &l0[k], &r0[k]);
            split_z16_ball_fast(B - r_l[k], idx_r[k], &a1[k], &l1[k], &r1[k]);
        }
        for (int k = 0; k < 4; ++k) {
            if (!ok[k]) continue;
            int8_t *out = Y + (g + (size_t)k) * 32u;
            unrank_z8_shell_fast(a0[k], l0[k], out);
            unrank_z8_shell_fast(r_l[k] - a0[k], r0[k], out + 8);
            unrank_z8_shell_fast(a1[k], l1[k], out + 16);
            unrank_z8_ball_fast((B - r_l[k]) - a1[k], r1[k], out + 24);
        }
    }
    if (g < groups) {
        bad += ztree_unrank_fast(in + g * (size_t)nb, groups - g, Y + g * 32u);
    }
    return bad;
}

/* --- Exact C11 Z^32 Binary-Tree Ranker (for 100,000/100,000 roundtrip proof!) --- */
static inline uint32_t rank_z2_shell_c(const int8_t *y) {
    int mc = g_z32.z2_max_c;
    size_t pos = (size_t)(y[0] + mc) * (size_t)g_z32.z2_side + (size_t)(y[1] + mc);
    return g_z32.z2_shell_idx[pos];
}

static inline uint32_t rank_z2_ball_c(const int8_t *y) {
    int mc = g_z32.z2_max_c;
    size_t pos = (size_t)(y[0] + mc) * (size_t)g_z32.z2_side + (size_t)(y[1] + mc);
    return g_z32.z2_ball_idx[pos];
}

static inline uint32_t rank_z4_shell_c(const int8_t *y, int r) {
    int r_l = y[0] * y[0] + y[1] * y[1];
    uint32_t blk = (uint32_t)(r_l / Z4_STRIDE);
    uint32_t off = g_z32.cp_s4[g_z32.cp4_off[r] + blk];
    for (int a = (int)(blk * Z4_STRIDE); a < r_l; ++a) {
        off += (uint32_t)g_z32.s2[a] * g_z32.s2[r - a];
    }
    return off + rank_z2_shell_c(y) * g_z32.s2[r - r_l] + rank_z2_shell_c(y + 2);
}

static inline uint32_t rank_z4_ball_c(const int8_t *y, int b) {
    int r_l = y[0] * y[0] + y[1] * y[1];
    uint32_t blk = (uint32_t)(r_l / Z4_STRIDE);
    uint32_t off = g_z32.cp_w4[g_z32.cp4_off[b] + blk];
    for (int a = (int)(blk * Z4_STRIDE); a < r_l; ++a) {
        off += (uint32_t)g_z32.s2[a] * g_z32.w2[b - a];
    }
    return off + rank_z2_shell_c(y) * g_z32.w2[b - r_l] + rank_z2_ball_c(y + 2);
}

static inline uint64_t rank_z8_shell_c(const int8_t *y, int r) {
    int r_l = 0;
    for (int i = 0; i < 4; ++i) r_l += y[i] * y[i];
    uint32_t blk = (uint32_t)(r_l / Z8_STRIDE);
    uint64_t off = g_z32.cp_s8[g_z32.cp8_off[r] + blk];
    for (int a = (int)(blk * Z8_STRIDE); a < r_l; ++a) {
        off += (uint64_t)g_z32.s4[a] * g_z32.s4[r - a];
    }
    return off + (uint64_t)rank_z4_shell_c(y, r_l) * g_z32.s4[r - r_l] + rank_z4_shell_c(y + 4, r - r_l);
}

static inline uint64_t rank_z8_ball_c(const int8_t *y, int b) {
    int r_l = 0;
    for (int i = 0; i < 4; ++i) r_l += y[i] * y[i];
    uint32_t blk = (uint32_t)(r_l / Z8_STRIDE);
    uint64_t off = g_z32.cp_w8[g_z32.cp8_off[b] + blk];
    for (int a = (int)(blk * Z8_STRIDE); a < r_l; ++a) {
        off += (uint64_t)g_z32.s4[a] * g_z32.w4[b - a];
    }
    return off + (uint64_t)rank_z4_shell_c(y, r_l) * g_z32.w4[b - r_l] + rank_z4_ball_c(y + 4, b - r_l);
}

static inline u128 rank_z16_shell_c(const int8_t *y, int r) {
    int r_l = 0;
    for (int i = 0; i < 8; ++i) r_l += y[i] * y[i];
    uint32_t blk = (uint32_t)(r_l / Z16_STRIDE);
    u128 off = g_z32.cp_s16[g_z32.cp16_off[r] + blk];
    for (int a = (int)(blk * Z16_STRIDE); a < r_l; ++a) {
        off += (u128)g_z32.s8[a] * g_z32.s8[r - a];
    }
    return off + (u128)rank_z8_shell_c(y, r_l) * g_z32.s8[r - r_l] + rank_z8_shell_c(y + 8, r - r_l);
}

static inline u128 rank_z16_ball_c(const int8_t *y, int b) {
    int r_l = 0;
    for (int i = 0; i < 8; ++i) r_l += y[i] * y[i];
    uint32_t blk = (uint32_t)(r_l / Z16_STRIDE);
    u128 off = g_z32.cp_w16[g_z32.cp16_off[b] + blk];
    for (int a = (int)(blk * Z16_STRIDE); a < r_l; ++a) {
        off += (u128)g_z32.s8[a] * g_z32.w8[b - a];
    }
    return off + (u128)rank_z8_shell_c(y, r_l) * g_z32.w8[b - r_l] + rank_z8_ball_c(y + 8, b - r_l);
}

size_t ztree_rank(const int8_t *Y, size_t groups, uint8_t *out) {
    size_t bad = 0;
    const int B = g_z32.bound;
    const int nb = g_z32.bytes;
    for (size_t g = 0; g < groups; ++g) {
        const int8_t *y = Y + g * 32u;
        int r_l = 0, r_r = 0;
        for (int i = 0; i < 16; ++i) r_l += y[i] * y[i];
        for (int i = 16; i < 32; ++i) r_r += y[i] * y[i];
        if (r_l + r_r > B) {
            ++bad;
            memset(out + g * (size_t)nb, 0xff, (size_t)nb);
            continue;
        }
        u128 idx_l = rank_z16_shell_c(y, r_l);
        u128 idx_r = rank_z16_ball_c(y + 16, B - r_l);
        u192 prod = u128_mul_to_u192(idx_l, g_z32.w16[B - r_l]);
        u192 idx_r192 = {{(uint64_t)idx_r, (uint64_t)(idx_r >> 64), 0}};
        u192 sum = u192_add(&g_z32.c_root[r_l], &prod);
        sum = u192_add(&sum, &idx_r192);
        store_u192_le(&sum, out + g * (size_t)nb, nb);
    }
    return bad;
}

/* ============================================================================
 * PART 2: E_8^4 (136-bit m=391 & 88-bit m=48) Reduced-Norm Binary-Tree Codec
 * ============================================================================ */

#define E8_STRIDE    16
#define E8_IN_STRIDE 16
#define E8_Q_STRIDE  16

typedef struct {
    int m_max;
    int r_max;    /* 2 * m_max */
    int t_max;    /* m_max - 1 */
    int bytes;
    u192 total_count;
    /* Outer E_8^4 -> E_8^2 -> E_8 */
    u192 *c_root;          /* [m_max + 2] */
    u128 *s_blk2;          /* [m_max + 1] */
    u128 *w_blk2;          /* [m_max + 1] */
    uint64_t *s_blk1;      /* [m_max + 1] */
    uint64_t *w_blk1;      /* [m_max + 1] */
    Recip192 *rec_root;    /* [m_max + 1] divisors W_blk2[m_R] */
    Recip128 *rec_b1;      /* [m_max + 1] divisors W_blk1[m_R] */
    Recip128 *rec_s1;      /* [m_max + 1] divisors S_blk1[m_R] */
    /* Inner E_8: 8 -> 4 -> 2 -> L1 LUT (even & odd cosets) */
    uint64_t *even_c8;     /* [m_max + 1] */
    uint32_t *even_c4[2];  /* [r_max + 1] */
    uint16_t *even_c2[2];  /* [r_max + 1] */
    uint16_t *even_p2_off[2]; /* [r_max + 2] */
    int8_t *even_p2_pts[2];   /* 2 * total pairs */
    uint32_t *odd_c4[2];   /* [t_max + 1] */
    uint16_t *odd_c2[2];   /* [t_max + 1] */
    uint16_t *odd_p2_off[2];  /* [t_max + 2] */
    int8_t *odd_p2_pts[2];    /* 2 * total pairs */
    Recip64 *rec_even_c4[2];  /* [r_max + 1] */
    Recip32 *rec_even_c2[2];  /* [r_max + 1] */
    Recip64 *rec_odd_c4[2];   /* [t_max + 1] */
    Recip32 *rec_odd_c2[2];   /* [t_max + 1] */
    /* Inverse LUT for O(1) E_8 pair ranking */
    int u_max;
    int u_side;
    uint16_t *even_p2_idx;    /* [u_side * u_side] */
    uint16_t *odd_p2_idx;     /* [u_side * u_side] */
    /* L2-resident strided checkpoints for outer E_8^2 & inner E_8 8->4 & 4->2 splits */
    uint32_t *cp_m_off;
    uint32_t *cp_m_len;
    u128 *cp_s_blk2;
    u128 *cp_w_blk2;
    uint64_t *e8_even_p0_tot; /* [m_max + 1] */
    uint64_t *e8_odd_p0_tot;  /* [m_max + 1] */
    uint32_t *cp_ec8_off;     /* [m_max + 1] */
    uint32_t *cp_ec8_len;     /* [m_max + 1] */
    uint64_t *cp_ec8[2];      /* even 8->4 strided prefix sums */
    uint32_t *cp_oc8_off;     /* [t_max + 1] */
    uint32_t *cp_oc8_len;     /* [t_max + 1] */
    uint64_t *cp_oc8[2];      /* odd 8->4 strided prefix sums */
    uint32_t *q_even_p0_tot[2]; /* [parity4][r_max + 1] */
    uint32_t *q_odd_p0_tot[2];  /* [parity4][t_max + 1] */
    uint32_t *cp_eq_off;      /* [r_max + 1] */
    uint32_t *cp_eq_len;      /* [r_max + 1] */
    uint32_t *cp_eq[4];       /* even 4->2 strided prefix sums, pair_id = (pa<<1)|pb */
    uint32_t *cp_oq_off;      /* [t_max + 1] */
    uint32_t *cp_oq_len;      /* [t_max + 1] */
    uint32_t *cp_oq[4];       /* odd 4->2 strided prefix sums, pair_id = (pa<<1)|pb */
    size_t checkpoint_bytes;
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
        free(g_e8.even_p2_off[p]);
        free(g_e8.even_p2_pts[p]);
        free(g_e8.odd_c4[p]);
        free(g_e8.odd_c2[p]);
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
    free(g_e8.even_p2_idx);
    free(g_e8.odd_p2_idx);
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

int e8tree_init(
    int m_max,
    int bytes,
    const uint64_t *c_root_limbs,  /* (m_max + 2) * 3 */
    const uint64_t *s_blk2_limbs,  /* (m_max + 1) * 2 */
    const uint64_t *w_blk2_limbs,  /* (m_max + 1) * 2 */
    const uint64_t *s_blk1_in,
    const uint64_t *w_blk1_in,
    const uint64_t *rec_root_m,    /* (m_max + 1) * 3 */
    const uint32_t *rec_root_s,
    const uint64_t *rec_b1_m,      /* (m_max + 1) * 2 */
    const uint32_t *rec_b1_s,
    const uint64_t *rec_s1_m,      /* (m_max + 1) * 2 */
    const uint32_t *rec_s1_s,
    const uint64_t *even_c8_in,
    const uint32_t *even_c4_in,    /* 2 * (r_max + 1) */
    const uint16_t *even_c2_in,    /* 2 * (r_max + 1) */
    const uint16_t *even_off_in,   /* 2 * (r_max + 2) */
    const int8_t *even_pts0_in, int even_n0,
    const int8_t *even_pts1_in, int even_n1,
    const uint32_t *odd_c4_in,     /* 2 * (t_max + 1) */
    const uint16_t *odd_c2_in,     /* 2 * (t_max + 1) */
    const uint16_t *odd_off_in,    /* 2 * (t_max + 2) */
    const int8_t *odd_pts0_in, int odd_n0,
    const int8_t *odd_pts1_in, int odd_n1,
    const uint64_t *rec_ec4_m, const uint32_t *rec_ec4_s, /* 2 * (r_max + 1) */
    const uint32_t *rec_ec2_m, const uint32_t *rec_ec2_s,
    const uint64_t *rec_oc4_m, const uint32_t *rec_oc4_s, /* 2 * (t_max + 1) */
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
        g_e8.even_p2_off[p] = (uint16_t *)malloc(R2 * sizeof(uint16_t));
        g_e8.even_p2_pts[p] = (int8_t *)malloc((size_t)even_ns[p] * 2u * sizeof(int8_t));
        memcpy(g_e8.even_c4[p], even_c4_in + (size_t)p * R1, R1 * sizeof(uint32_t));
        memcpy(g_e8.even_c2[p], even_c2_in + (size_t)p * R1, R1 * sizeof(uint16_t));
        memcpy(g_e8.even_p2_off[p], even_off_in + (size_t)p * R2, R2 * sizeof(uint16_t));
        memcpy(g_e8.even_p2_pts[p], even_pts[p], (size_t)even_ns[p] * 2u * sizeof(int8_t));

        g_e8.odd_c4[p] = (uint32_t *)malloc(T1 * sizeof(uint32_t));
        g_e8.odd_c2[p] = (uint16_t *)malloc(T1 * sizeof(uint16_t));
        g_e8.odd_p2_off[p] = (uint16_t *)malloc(T2 * sizeof(uint16_t));
        g_e8.odd_p2_pts[p] = (int8_t *)malloc((size_t)odd_ns[p] * 2u * sizeof(int8_t));
        memcpy(g_e8.odd_c4[p], odd_c4_in + (size_t)p * T1, T1 * sizeof(uint32_t));
        memcpy(g_e8.odd_c2[p], odd_c2_in + (size_t)p * T1, T1 * sizeof(uint16_t));
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

    /* Inverse LUT for O(1) E_8 pair ranking */
    int um = 0;
    while ((um + 1) * (um + 1) <= 8 * m_max) ++um;
    g_e8.u_max = um;
    g_e8.u_side = 2 * um + 1;
    size_t ugrid = (size_t)g_e8.u_side * (size_t)g_e8.u_side;
    g_e8.even_p2_idx = (uint16_t *)malloc(ugrid * sizeof(uint16_t));
    g_e8.odd_p2_idx = (uint16_t *)malloc(ugrid * sizeof(uint16_t));
    memset(g_e8.even_p2_idx, 0xff, ugrid * sizeof(uint16_t));
    memset(g_e8.odd_p2_idx, 0xff, ugrid * sizeof(uint16_t));
    for (int p = 0; p < 2; ++p) {
        for (int r = 0; r <= r_max; ++r) {
            uint16_t s = g_e8.even_p2_off[p][r], e = g_e8.even_p2_off[p][r + 1];
            for (uint16_t idx = s; idx < e; ++idx) {
                int u0 = g_e8.even_p2_pts[p][2u * idx];
                int u1 = g_e8.even_p2_pts[p][2u * idx + 1u];
                g_e8.even_p2_idx[(size_t)(u0 + um) * (size_t)g_e8.u_side + (size_t)(u1 + um)] = (uint16_t)(idx - s);
            }
        }
        for (int t = 0; t <= t_max; ++t) {
            uint16_t s = g_e8.odd_p2_off[p][t], e = g_e8.odd_p2_off[p][t + 1];
            for (uint16_t idx = s; idx < e; ++idx) {
                int u0 = g_e8.odd_p2_pts[p][2u * idx];
                int u1 = g_e8.odd_p2_pts[p][2u * idx + 1u];
                g_e8.odd_p2_idx[(size_t)(u0 + um) * (size_t)g_e8.u_side + (size_t)(u1 + um)] = (uint16_t)(idx - s);
            }
        }
    }

    /* Outer E_8^2 strided checkpoints + inner p=0 branch totals */
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

    /* Inner E_8 quad (4->2) strided checkpoints + pa=0 totals */
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
    g_e8.checkpoint_bytes =
        (size_t)total_cp * 32u + M1 * 16u +
        (size_t)(tot_ec8 + tot_oc8) * 16u +
        (size_t)(tot_eq + tot_oq) * 16u +
        (R1 + T1) * 8u;
    g_e8_ready = 1;
    return 0;
}

size_t e8tree_checkpoint_bytes(void) {
    return g_e8_ready ? g_e8.checkpoint_bytes : 0u;
}

static inline void unrank_e8_quad(uint32_t rem, int coset, int norm4, int parity4, int8_t *out) {
    const uint16_t *c2_0 = (coset == 0) ? g_e8.even_c2[0] : g_e8.odd_c2[0];
    const uint16_t *c2_1 = (coset == 0) ? g_e8.even_c2[1] : g_e8.odd_c2[1];
    const uint16_t *const *off = (coset == 0) ? (const uint16_t *const *)g_e8.even_p2_off : (const uint16_t *const *)g_e8.odd_p2_off;
    const int8_t *const *pts = (coset == 0) ? (const int8_t *const *)g_e8.even_p2_pts : (const int8_t *const *)g_e8.odd_p2_pts;
    const Recip32 *const *recs = (coset == 0) ? (const Recip32 *const *)g_e8.rec_even_c2 : (const Recip32 *const *)g_e8.rec_odd_c2;

    for (int pa = 0; pa < 2; ++pa) {
        int pb = parity4 ^ pa;
        const uint16_t *ca = (pa == 0) ? c2_0 : c2_1;
        const uint16_t *cb = (pb == 0) ? c2_0 : c2_1;
        for (int na = 0; na <= norm4; ++na) {
            int nb = norm4 - na;
            uint32_t term = (uint32_t)ca[na] * cb[nb];
            if (rem < term) {
                Recip32 rec = recs[pb][nb];
                uint32_t d = cb[nb];
                uint32_t il = mul_shr_u32_to_u32(rem, rec.m, rec.shift);
                uint32_t ir = rem - il * d;
                uint32_t pl = (uint32_t)off[pa][na] + il;
                uint32_t pr = (uint32_t)off[pb][nb] + ir;
                out[0] = pts[pa][2u * pl];
                out[1] = pts[pa][2u * pl + 1u];
                out[2] = pts[pb][2u * pr];
                out[3] = pts[pb][2u * pr + 1u];
                return;
            }
            rem -= term;
        }
    }
}

static inline void unrank_e8_block(uint64_t rank, int is_ball, int m_bound, int8_t *out) {
    int m;
    uint64_t rem = rank;
    if (is_ball) {
        int lo = 0, hi = m_bound;
        const uint64_t *w1 = g_e8.w_blk1;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (rem < w1[mid]) hi = mid;
            else lo = mid + 1;
        }
        m = lo;
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
    int p;
    if (rem < p0_tot) {
        p = 0;
    } else {
        p = 1;
        rem -= p0_tot;
    }
    const uint32_t *c4 = (coset == 0) ? g_e8.even_c4[p] : g_e8.odd_c4[p];
    const Recip64 *recs = (coset == 0) ? g_e8.rec_even_c4[p] : g_e8.rec_odd_c4[p];
    int n_l = 0;
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
    unrank_e8_quad(il, coset, n_l, p, out);
    unrank_e8_quad(ir, coset, n_r, p, out + 4);
}

static inline int root_split_e8_4(const uint8_t *in, int *m_l_out, u128 *idx_l_out, u128 *idx_r_out) {
    u192 rk = load_u192_le(in, g_e8.bytes);
    if (u192_ge(&rk, &g_e8.total_count)) return 1;
    int lo = 0, hi = g_e8.m_max;
    const u192 *c_root = g_e8.c_root;
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (u192_le(&c_root[mid], &rk)) lo = mid;
        else hi = mid - 1;
    }
    int m_l = lo;
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

size_t e8tree_unrank_compact(const uint8_t *in, size_t groups, int8_t *U) {
    size_t bad = 0;
    const int M = g_e8.m_max;
    const int nb = g_e8.bytes;
    const uint64_t *s1 = g_e8.s_blk1;
    const uint64_t *w1 = g_e8.w_blk1;
    for (size_t g = 0; g < groups; ++g) {
        int m_l;
        u128 idx_l, idx_r;
        int8_t *out = U + g * 32u;
        if (root_split_e8_4(in + g * (size_t)nb, &m_l, &idx_l, &idx_r)) {
            ++bad;
            memset(out, 0, 32u);
            continue;
        }
        /* Level 1 left: Shell(E8^2, m_l) */
        int a0 = 0;
        u128 rem0 = idx_l;
        for (; a0 <= m_l; ++a0) {
            u128 term = (u128)s1[a0] * s1[m_l - a0];
            if (rem0 < term) break;
            rem0 -= term;
        }
        Recip128 rec0 = g_e8.rec_s1[m_l - a0];
        uint64_t l0 = mul_shr_u128_to_u64(rem0, rec0.m, rec0.shift);
        uint64_t r0 = (uint64_t)rem0 - l0 * s1[m_l - a0];
        /* Level 1 right: Ball(E8^2, M - m_l) */
        int b_r = M - m_l;
        int a1 = 0;
        u128 rem1 = idx_r;
        for (; a1 <= b_r; ++a1) {
            u128 term = (u128)s1[a1] * w1[b_r - a1];
            if (rem1 < term) break;
            rem1 -= term;
        }
        Recip128 rec1 = g_e8.rec_b1[b_r - a1];
        uint64_t l1 = mul_shr_u128_to_u64(rem1, rec1.m, rec1.shift);
        uint64_t r1 = (uint64_t)rem1 - l1 * w1[b_r - a1];
        /* Level 2: 4 independent native uint64/uint32 E_8 blocks! */
        unrank_e8_block(l0, 0, a0, out);
        unrank_e8_block(r0, 0, m_l - a0, out + 8);
        unrank_e8_block(l1, 0, a1, out + 16);
        unrank_e8_block(r1, 1, b_r - a1, out + 24);
    }
    return bad;
}

static inline void unrank_e8_quad_fast(uint32_t rem, int coset, int norm4, int parity4, int8_t *out) {
    uint32_t p0_tot = (coset == 0) ? g_e8.q_even_p0_tot[parity4][norm4] : g_e8.q_odd_p0_tot[parity4][norm4];
    int pa;
    if (rem < p0_tot) {
        pa = 0;
    } else {
        pa = 1;
        rem -= p0_tot;
    }
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

static inline void unrank_e8_block_fast(uint64_t rank, int is_ball, int m_bound, int8_t *out) {
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
    int p;
    if (rem < p0_tot) {
        p = 0;
    } else {
        p = 1;
        rem -= p0_tot;
    }
    const uint32_t *c4 = (coset == 0) ? g_e8.even_c4[p] : g_e8.odd_c4[p];
    const Recip64 *recs = (coset == 0) ? g_e8.rec_even_c4[p] : g_e8.rec_odd_c4[p];
    const uint64_t *cp = (coset == 0) ? (g_e8.cp_ec8[p] + g_e8.cp_ec8_off[m])
                                      : (g_e8.cp_oc8[p] + g_e8.cp_oc8_off[n_tot]);
    uint32_t nblk = (coset == 0) ? g_e8.cp_ec8_len[m] : g_e8.cp_oc8_len[n_tot];
    uint32_t blk = find_cp_u64(cp, nblk, rem);
    rem -= cp[blk];
    int n_l = (int)(blk * E8_IN_STRIDE);
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
    unrank_e8_quad_fast(il, coset, n_l, p, out);
    unrank_e8_quad_fast(ir, coset, n_r, p, out + 4);
}

static inline void split_e8_blk2_shell_fast(int m_l, u128 idx_l, int *a0_out, uint64_t *l0_out, uint64_t *r0_out) {
    const uint64_t *s1 = g_e8.s_blk1;
    const u128 *cp0 = g_e8.cp_s_blk2 + g_e8.cp_m_off[m_l];
    uint32_t blk0 = find_cp_u128(cp0, g_e8.cp_m_len[m_l], idx_l);
    u128 rem0 = idx_l - cp0[blk0];
    int a0 = (int)(blk0 * E8_STRIDE);
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

static inline void split_e8_blk2_ball_fast(int b_r, u128 idx_r, int *a1_out, uint64_t *l1_out, uint64_t *r1_out) {
    const uint64_t *s1 = g_e8.s_blk1;
    const uint64_t *w1 = g_e8.w_blk1;
    const u128 *cp1 = g_e8.cp_w_blk2 + g_e8.cp_m_off[b_r];
    uint32_t blk1 = find_cp_u128(cp1, g_e8.cp_m_len[b_r], idx_r);
    u128 rem1 = idx_r - cp1[blk1];
    int a1 = (int)(blk1 * E8_STRIDE);
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

size_t e8tree_unrank_fast(const uint8_t *in, size_t groups, int8_t *U) {
    size_t bad = 0;
    const int M = g_e8.m_max;
    const int nb = g_e8.bytes;
    for (size_t g = 0; g < groups; ++g) {
        int m_l;
        u128 idx_l, idx_r;
        int8_t *out = U + g * 32u;
        if (root_split_e8_4(in + g * (size_t)nb, &m_l, &idx_l, &idx_r)) {
            ++bad;
            memset(out, 0, 32u);
            continue;
        }
        int a0, a1;
        uint64_t l0, r0, l1, r1;
        split_e8_blk2_shell_fast(m_l, idx_l, &a0, &l0, &r0);
        split_e8_blk2_ball_fast(M - m_l, idx_r, &a1, &l1, &r1);
        unrank_e8_block_fast(l0, 0, a0, out);
        unrank_e8_block_fast(r0, 0, m_l - a0, out + 8);
        unrank_e8_block_fast(l1, 0, a1, out + 16);
        unrank_e8_block_fast(r1, 1, (M - m_l) - a1, out + 24);
    }
    return bad;
}

size_t e8tree_unrank_fast_batch4(const uint8_t *in, size_t groups, int8_t *U) {
    size_t bad = 0;
    const int M = g_e8.m_max;
    const int nb = g_e8.bytes;
    size_t g = 0;
    for (; g + 4u <= groups; g += 4u) {
        int m_l[4], ok[4];
        u128 idx_l[4], idx_r[4];
        for (int k = 0; k < 4; ++k) {
            ok[k] = !root_split_e8_4(in + (g + (size_t)k) * (size_t)nb, &m_l[k], &idx_l[k], &idx_r[k]);
            if (!ok[k]) {
                ++bad;
                memset(U + (g + (size_t)k) * 32u, 0, 32u);
            }
        }
        int a0[4], a1[4];
        uint64_t l0[4], r0[4], l1[4], r1[4];
        for (int k = 0; k < 4; ++k) {
            if (!ok[k]) continue;
            split_e8_blk2_shell_fast(m_l[k], idx_l[k], &a0[k], &l0[k], &r0[k]);
            split_e8_blk2_ball_fast(M - m_l[k], idx_r[k], &a1[k], &l1[k], &r1[k]);
        }
        for (int k = 0; k < 4; ++k) {
            if (!ok[k]) continue;
            int8_t *out = U + (g + (size_t)k) * 32u;
            unrank_e8_block_fast(l0[k], 0, a0[k], out);
            unrank_e8_block_fast(r0[k], 0, m_l[k] - a0[k], out + 8);
            unrank_e8_block_fast(l1[k], 0, a1[k], out + 16);
            unrank_e8_block_fast(r1[k], 1, (M - m_l[k]) - a1[k], out + 24);
        }
    }
    if (g < groups) {
        bad += e8tree_unrank_fast(in + g * (size_t)nb, groups - g, U + g * 32u);
    }
    return bad;
}

/* --- Exact C11 E_8^4 Ranker (for 100,000/100,000 exact roundtrip + validity proof!) --- */
static inline int floor_div2(int v) {
    return (v >= 0) ? (v >> 1) : -((-v + 1) >> 1);
}

static inline int valid_e8_block_c(const int8_t *u) {
    int c = u[0] & 1;
    int s = 0;
    for (int i = 0; i < 8; ++i) {
        if ((u[i] & 1) != c) return 0;
        s += floor_div2((int)u[i]);
    }
    return (s & 1) == 0;
}

static inline uint32_t rank_e8_quad_c(const int8_t *u4, int coset, int norm4, int parity4) {
    const uint16_t *c2_0 = (coset == 0) ? g_e8.even_c2[0] : g_e8.odd_c2[0];
    const uint16_t *c2_1 = (coset == 0) ? g_e8.even_c2[1] : g_e8.odd_c2[1];
    const uint16_t *idx_tbl = (coset == 0) ? g_e8.even_p2_idx : g_e8.odd_p2_idx;
    int k0 = floor_div2(u4[0]), k1 = floor_div2(u4[1]);
    int p_l = (k0 + k1) & 1;
    int n_l = (coset == 0) ? (k0 * k0 + k1 * k1) : (k0 * (k0 + 1) / 2 + k1 * (k1 + 1) / 2);
    int n_r = norm4 - n_l;
    int um = g_e8.u_max, side = g_e8.u_side;
    uint32_t il = idx_tbl[(size_t)(u4[0] + um) * (size_t)side + (size_t)(u4[1] + um)];
    uint32_t ir = idx_tbl[(size_t)(u4[2] + um) * (size_t)side + (size_t)(u4[3] + um)];
    uint32_t off = 0;
    for (int pa = 0; pa < 2; ++pa) {
        int pb = parity4 ^ pa;
        const uint16_t *ca = (pa == 0) ? c2_0 : c2_1;
        const uint16_t *cb = (pb == 0) ? c2_0 : c2_1;
        for (int na = 0; na <= norm4; ++na) {
            if (pa == p_l && na == n_l) {
                return off + il * cb[n_r] + ir;
            }
            off += (uint32_t)ca[na] * cb[norm4 - na];
        }
    }
    return 0;
}

static inline uint64_t rank_e8_block_c(const int8_t *u, int is_ball, int m) {
    uint64_t base = (is_ball && m > 0) ? g_e8.w_blk1[m - 1] : 0u;
    int coset = u[0] & 1;
    int n_tot = (coset == 0) ? (2 * m) : (m - 1);
    int n_l = 0, p_l = 0;
    for (int i = 0; i < 4; ++i) {
        int k = floor_div2(u[i]);
        p_l ^= (k & 1);
        n_l += (coset == 0) ? (k * k) : (k * (k + 1) / 2);
    }
    uint64_t off = (coset == 0) ? 0u : g_e8.even_c8[m];
    if (p_l == 1) {
        off += (coset == 0) ? g_e8.e8_even_p0_tot[m] : g_e8.e8_odd_p0_tot[m];
    }
    const uint32_t *c4 = (coset == 0) ? g_e8.even_c4[p_l] : g_e8.odd_c4[p_l];
    for (int a = 0; a < n_l; ++a) {
        off += (uint64_t)c4[a] * c4[n_tot - a];
    }
    uint32_t il = rank_e8_quad_c(u, coset, n_l, p_l);
    uint32_t ir = rank_e8_quad_c(u + 4, coset, n_tot - n_l, p_l);
    return base + off + (uint64_t)il * c4[n_tot - n_l] + ir;
}

size_t e8tree_rank(const int8_t *U, size_t groups, uint8_t *out) {
    size_t bad = 0;
    const int M = g_e8.m_max;
    const int nb = g_e8.bytes;
    for (size_t g = 0; g < groups; ++g) {
        const int8_t *u = U + g * 32u;
        int ok = 1;
        int mb[4] = {0, 0, 0, 0};
        for (int b = 0; b < 4; ++b) {
            if (!valid_e8_block_c(u + 8 * b)) { ok = 0; break; }
            int sq = 0;
            for (int i = 0; i < 8; ++i) sq += u[8 * b + i] * u[8 * b + i];
            if ((sq & 7) != 0) { ok = 0; break; }
            mb[b] = sq >> 3;
        }
        int m_l = mb[0] + mb[1];
        int m_r = mb[2] + mb[3];
        if (!ok || m_l + m_r > M) {
            ++bad;
            memset(out + g * (size_t)nb, 0xff, (size_t)nb);
            continue;
        }
        /* Left Shell(E8^2, m_l) */
        uint32_t blk0 = (uint32_t)(mb[0] / E8_STRIDE);
        u128 off0 = g_e8.cp_s_blk2[g_e8.cp_m_off[m_l] + blk0];
        for (int a = (int)(blk0 * E8_STRIDE); a < mb[0]; ++a) {
            off0 += (u128)g_e8.s_blk1[a] * g_e8.s_blk1[m_l - a];
        }
        u128 idx_l = off0 + (u128)rank_e8_block_c(u, 0, mb[0]) * g_e8.s_blk1[mb[1]] + rank_e8_block_c(u + 8, 0, mb[1]);

        /* Right Ball(E8^2, M - m_l) */
        int b_r = M - m_l;
        uint32_t blk1 = (uint32_t)(mb[2] / E8_STRIDE);
        u128 off1 = g_e8.cp_w_blk2[g_e8.cp_m_off[b_r] + blk1];
        for (int a = (int)(blk1 * E8_STRIDE); a < mb[2]; ++a) {
            off1 += (u128)g_e8.s_blk1[a] * g_e8.w_blk1[b_r - a];
        }
        u128 idx_r = off1 + (u128)rank_e8_block_c(u + 16, 0, mb[2]) * g_e8.w_blk1[b_r - mb[2]] + rank_e8_block_c(u + 24, 1, mb[3]);

        u192 prod = u128_mul_to_u192(idx_l, g_e8.w_blk2[b_r]);
        u192 r192 = {{(uint64_t)idx_r, (uint64_t)(idx_r >> 64), 0}};
        u192 total = u192_add(&g_e8.c_root[m_l], &prod);
        total = u192_add(&total, &r192);
        store_u192_le(&total, out + g * (size_t)nb, nb);
    }
    return bad;
}

/* ============================================================================
 * PART 3: Multi-Core (pthread) & 1-Core CLOCK_MONOTONIC Benchmark Runner
 * ============================================================================ */

typedef size_t (*unrank_fn_t)(const uint8_t *, size_t, int8_t *);

typedef struct {
    unrank_fn_t fn;
    const uint8_t *in;
    size_t groups;
    int8_t *out;
    size_t bad;
    uint64_t cpu_ns;
} WorkerArgs;

static void *worker_entry(void *arg) {
    WorkerArgs *w = (WorkerArgs *)arg;
    uint64_t c0 = thread_cpu_ns();
    w->bad = w->fn(w->in, w->groups, w->out);
    w->cpu_ns = thread_cpu_ns() - c0;
    return NULL;
}

static uint64_t run_bench_fn(
    unrank_fn_t fn,
    int n_threads,
    const uint8_t *in,
    size_t groups,
    int in_stride_bytes,
    int8_t *out,
    size_t *bad_out,
    uint64_t *cpu_ns_out
) {
    if (n_threads <= 1) {
        uint64_t w0 = now_ns();
        uint64_t c0 = thread_cpu_ns();
        *bad_out = fn(in, groups, out);
        uint64_t c1 = thread_cpu_ns();
        uint64_t w1 = now_ns();
        if (cpu_ns_out) *cpu_ns_out = c1 - c0;
        return w1 - w0;
    }
    if (n_threads > 8) n_threads = 8;
    pthread_t th[8];
    WorkerArgs wa[8];
    size_t chunk = groups / (size_t)n_threads;
    size_t rem = groups % (size_t)n_threads;
    size_t off = 0;
    uint64_t t0 = now_ns();
    for (int i = 0; i < n_threads; ++i) {
        size_t cnt = chunk + ((size_t)i < rem ? 1u : 0u);
        wa[i].fn = fn;
        wa[i].in = in + off * (size_t)in_stride_bytes;
        wa[i].groups = cnt;
        wa[i].out = out + off * 32u;
        wa[i].bad = 0;
        wa[i].cpu_ns = 0;
        off += cnt;
        pthread_create(&th[i], NULL, worker_entry, &wa[i]);
    }
    size_t tot_bad = 0;
    uint64_t tot_cpu = 0;
    for (int i = 0; i < n_threads; ++i) {
        pthread_join(th[i], NULL);
        tot_bad += wa[i].bad;
        tot_cpu += wa[i].cpu_ns;
    }
    uint64_t elapsed = now_ns() - t0;
    *bad_out = tot_bad;
    if (cpu_ns_out) *cpu_ns_out = tot_cpu;
    return elapsed;
}

/* mode: 0=ztree_compact, 1=ztree_fast, 2=ztree_batch4, 3=e8_compact, 4=e8_fast, 5=e8_batch4 */
uint64_t bench_unrank_ns(
    int mode,
    int n_threads,
    const uint8_t *in,
    size_t groups,
    int in_stride_bytes,
    int8_t *out,
    size_t *bad_out,
    uint64_t *cpu_ns_out
) {
    unrank_fn_t fn = NULL;
    if (mode == 0) fn = ztree_unrank_compact;
    else if (mode == 1) fn = ztree_unrank_fast;
    else if (mode == 2) fn = ztree_unrank_fast_batch4;
    else if (mode == 3) fn = e8tree_unrank_compact;
    else if (mode == 4) fn = e8tree_unrank_fast;
    else if (mode == 5) fn = e8tree_unrank_fast_batch4;
    else { *bad_out = groups; if (cpu_ns_out) *cpu_ns_out = 0; return 0; }
    return run_bench_fn(fn, n_threads, in, groups, in_stride_bytes, out, bad_out, cpu_ns_out);
}

/* Also time any external C unrank_fn_t (e.g., baseline ballpack.c zb_unrank) identically! */
uint64_t bench_external_unrank_ns(
    void *fn_ptr,
    int n_threads,
    const uint8_t *in,
    size_t groups,
    int in_stride_bytes,
    int8_t *out,
    size_t *bad_out,
    uint64_t *cpu_ns_out
) {
    unrank_fn_t fn;
    memcpy(&fn, &fn_ptr, sizeof(fn));
    return run_bench_fn(fn, n_threads, in, groups, in_stride_bytes, out, bad_out, cpu_ns_out);
}
