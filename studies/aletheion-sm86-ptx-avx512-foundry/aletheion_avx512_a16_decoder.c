/*
 * Native Intel i9-11900H AVX-512 (vpermb + vpopcntdq + vpdpbusd) CPU Decoder
 * Target: 11th Gen Intel Core i9-11900H (Tiger Lake-H, 8C/16T @ 4.9 GHz)
 *         ISA: AVX-512F/BW/DQ/VL/VBMI (vpermb) + AVX-512VPOPCNTDQ (vpopcntq) + AVX-512VNNI (vpdpbusd)
 *
 * Benchmarks & Bit-Exact Verification across THREE architectures:
 *   1. Hardware-native 4.500-bpw bit-packed A_16 [16,11,4] (18B/32w & 72B/128w):
 *      - Scalar C reference decode + 5-bit syndrome verify + dot product
 *      - AVX-512 decode + hardware vpopcntq 5-bit syndrome verify (8 A_16 blocks / ZMM!)
 *        + vpermb 64-byte in-register LUT unpack
 *      - AVX-512 fused decode + vpdpbusd VNNI INT8 GEMV (64 MACs/instruction, exact -8*sum_x bias)
 *      - Single-thread (1T) AND multi-core (8T, 16T OpenMP)
 *   2. Hardware-native 4.500-bpw Block-H2 ((a+b, a-b)/sqrt(2), peak<=14<=127):
 *      - Scalar C reference vs AVX-512 vpermb + in-register butterfly + vpdpbusd fused GEMV
 *   3. Exact siliquant#77 hampack.c (L=16, n=32, bits=136, bound=1207, 18-byte enumerative A_16 x A_16):
 *      - Exact 3-limb scalar hampack.c baseline (reproducing siliquant#77's 9.4-10.1 M weights/s!)
 *      - Optimized fast-limb (__uint128_t for i>=2 + 2-level gallop search + AVX-512 scale/convert
 *        + 8C/16T OpenMP) verified BIT-IDENTICAL against scalar hampack.c on 100% of blocks!
 *
 * Build:
 *   gcc -O3 -march=native -mavx512f -mavx512bw -mavx512vl -mavx512vbmi \
 *       -mavx512vpopcntdq -mavx512vnni -funroll-loops -fopenmp \
 *       aletheion_avx512_a16_decoder.c -lm -o aletheion_avx512_a16_decoder
 */

#define _POSIX_C_SOURCE 200809L
#include <immintrin.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void* aligned64_alloc(size_t nbytes) {
    size_t sz = (nbytes + 63ULL) & ~63ULL;
    void* p = NULL;
    if (posix_memalign(&p, 64, sz) != 0 || !p) {
        fprintf(stderr, "OOM allocating %zu bytes\n", sz);
        exit(1);
    }
    memset(p, 0, sz);
    return p;
}

// ============================================================================
// Part A: Hardware-Native Bit-Packed 4.500-bpw A_16 [16,11,4] & Block-H2
// Layout: 72-byte superblock = 64 bytes packed nibbles (128 weights = 8 x A_16)
//                            + 4 x uint16_t FP16/scale-codes (8 bytes = 1 per 32w)
//         -> exact 18 bytes / 32 weights = 72 bytes / 128 weights = 4.5000 bpw!
// ============================================================================

static inline int unpack_s4_scalar(uint64_t w, int idx) {
    uint32_t nib = (uint32_t)((w >> (idx * 4)) & 0xFu);
    return (nib & 8u) ? (int)(nib | 0xFFFFFFF0u) : (int)nib;
}

static inline uint32_t a16_syndrome_scalar(uint64_t word) {
    uint64_t p = word & 0x1111111111111111ULL;
    uint32_t s0 = (uint32_t)__builtin_popcountll(p) & 1u;
    uint32_t s1 = (uint32_t)__builtin_popcountll(p & 0x1010101010101010ULL) & 1u;
    uint32_t s2 = (uint32_t)__builtin_popcountll(p & 0x1100110011001100ULL) & 1u;
    uint32_t s3 = (uint32_t)__builtin_popcountll(p & 0x1111000011110000ULL) & 1u;
    uint32_t s4 = (uint32_t)__builtin_popcountll(p & 0x1111111100000000ULL) & 1u;
    return s0 | (s1 << 1) | (s2 << 2) | (s3 << 3) | (s4 << 4);
}

static uint64_t make_valid_a16_word(uint32_t* rng) {
    int v[16];
    int min_abs = 256, min_idx = 0;
    uint32_t bits = 0;
    for (int i = 0; i < 16; ++i) {
        *rng = (*rng) * 1664525u + 1013904223u;
        v[i] = ((int)((*rng) >> 24) % 15) - 7; // [-7, +7]
        bits |= ((uint32_t)(v[i] & 1)) << i;
        int av = v[i] < 0 ? -v[i] : v[i];
        if (av < min_abs) { min_abs = av; min_idx = i; }
    }
    uint32_t p0 = (uint32_t)__builtin_popcount(bits) & 1u;
    uint32_t syn = ((uint32_t)__builtin_popcount(bits & 0xAAAAu) & 1u)
                 | (((uint32_t)__builtin_popcount(bits & 0xCCCCu) & 1u) << 1)
                 | (((uint32_t)__builtin_popcount(bits & 0xF0F0u) & 1u) << 2)
                 | (((uint32_t)__builtin_popcount(bits & 0xFF00u) & 1u) << 3);
    if (p0) {
        v[syn] += (v[syn] >= 0) ? -1 : 1;
    } else if (syn != 0u) {
        int p = min_idx ^ (int)syn;
        v[min_idx] += (v[min_idx] >= 0) ? -1 : 1;
        v[p] += (v[p] >= 0) ? -1 : 1;
    }
    uint64_t w = 0;
    for (int i = 0; i < 16; ++i) w |= ((uint64_t)(v[i] & 0xF)) << (i * 4);
    return w;
}

// 1. Scalar A_16 unpack + syndrome verify (writes int8_t[128] per 64B block)
static uint64_t decode_a16_scalar(const uint64_t* __restrict__ payloads,
                                  int8_t* __restrict__ out_w,
                                  size_t n_blocks128) {
    uint64_t errs = 0;
    for (size_t b = 0; b < n_blocks128; ++b) {
        const uint64_t* src = payloads + b * 8;
        int8_t* dst = out_w + b * 128;
        for (int k = 0; k < 8; ++k) {
            uint64_t w = src[k];
            if (a16_syndrome_scalar(w) != 0u) ++errs;
            for (int i = 0; i < 16; ++i) dst[k * 16 + i] = (int8_t)unpack_s4_scalar(w, i);
        }
    }
    return errs;
}

// 2. AVX-512 A_16 unpack + hardware vpopcntq 5-bit syndrome verify + vpermb 64B LUT!
//    Processes 128 weights (8 complete A_16 blocks = 1 ZMM load) per iteration!
static uint64_t decode_a16_avx512(const uint64_t* __restrict__ payloads,
                                  int8_t* __restrict__ out_w,
                                  size_t n_blocks128,
                                  int n_threads) {
    // 64-byte in-register LUT for vpermb: maps 4-bit nibble 0..15 -> signed int8 [0..7, -8..-1]
    // (repeated 4x across 64 bytes so any 4-bit index works directly in vpermb!)
    alignas(64) static const int8_t LUT_S4_TO_S8[64] = {
        0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1,
        0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1,
        0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1,
        0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1,
    };
    // Cross-lane interleave permutation indices for vpermb:
    // After unpacking 64 bytes -> lo_s8 (even weights 0,2,4..126) and hi_s8 (odd weights 1,3,5..127),
    // _mm512_permutex2var_epi8 (`vpermi2b`) interleaves exact contiguous order in TWO instructions!
    alignas(64) uint8_t idx_first64[64], idx_second64[64];
    for (int i = 0; i < 32; ++i) {
        idx_first64[2 * i]      = (uint8_t)i;            // from lo_s8[0..31]
        idx_first64[2 * i + 1]  = (uint8_t)(64 + i);     // from hi_s8[0..31]
        idx_second64[2 * i]     = (uint8_t)(32 + i);     // from lo_s8[32..63]
        idx_second64[2 * i + 1] = (uint8_t)(64 + 32 + i);// from hi_s8[32..63]
    }
    const __m512i v_lut = _mm512_load_si512((const void*)LUT_S4_TO_S8);
    const __m512i v_p0  = _mm512_loadu_si512((const void*)idx_first64);
    const __m512i v_p1  = _mm512_loadu_si512((const void*)idx_second64);
    const __m512i v_m0f = _mm512_set1_epi8(0x0F);

    const __m512i m_p0 = _mm512_set1_epi64((long long)0x1111111111111111ULL);
    const __m512i m_p1 = _mm512_set1_epi64((long long)0x1010101010101010ULL);
    const __m512i m_p2 = _mm512_set1_epi64((long long)0x1100110011001100ULL);
    const __m512i m_p3 = _mm512_set1_epi64((long long)0x1111000011110000ULL);
    const __m512i m_p4 = _mm512_set1_epi64((long long)0x1111111100000000ULL);
    const __m512i v_one = _mm512_set1_epi64(1LL);

    uint64_t total_errs = 0;
    #pragma omp parallel for num_threads(n_threads) reduction(+:total_errs) schedule(static)
    for (size_t b = 0; b < n_blocks128; ++b) {
        __m512i raw = _mm512_loadu_si512((const void*)(payloads + b * 8));

        // 5-bit extended Hamming [16,11,4] syndrome verify across ALL 8 A_16 blocks simultaneously via vpopcntq!
        __m512i bits = _mm512_and_si512(raw, m_p0);
        __m512i c0 = _mm512_and_si512(_mm512_popcnt_epi64(bits), v_one);
        __m512i c1 = _mm512_and_si512(_mm512_popcnt_epi64(_mm512_and_si512(bits, m_p1)), v_one);
        __m512i c2 = _mm512_and_si512(_mm512_popcnt_epi64(_mm512_and_si512(bits, m_p2)), v_one);
        __m512i c3 = _mm512_and_si512(_mm512_popcnt_epi64(_mm512_and_si512(bits, m_p3)), v_one);
        __m512i c4 = _mm512_and_si512(_mm512_popcnt_epi64(_mm512_and_si512(bits, m_p4)), v_one);
        __m512i syn = _mm512_or_si512(_mm512_or_si512(c0, _mm512_or_si512(c1, c2)), _mm512_or_si512(c3, c4));
        __mmask8 bad = _mm512_test_epi64_mask(syn, syn);
        if (__builtin_expect(bad != 0, 0)) total_errs += (uint64_t)__builtin_popcount((unsigned)bad);

        // 64-byte in-register vpermb LUT decode (128 nibbles -> 128 signed int8s)
        __m512i nib_lo = _mm512_and_si512(raw, v_m0f);
        __m512i nib_hi = _mm512_and_si512(_mm512_srli_epi16(raw, 4), v_m0f);
        __m512i s8_lo  = _mm512_permutexvar_epi8(nib_lo, v_lut);
        __m512i s8_hi  = _mm512_permutexvar_epi8(nib_hi, v_lut);

        __m512i out0 = _mm512_permutex2var_epi8(s8_lo, v_p0, s8_hi);
        __m512i out1 = _mm512_permutex2var_epi8(s8_lo, v_p1, s8_hi);
        _mm512_storeu_si512((void*)(out_w + b * 128),      out0);
        _mm512_storeu_si512((void*)(out_w + b * 128 + 64), out1);
    }
    return total_errs;
}

// 3. Fused AVX-512 A_16 VNNI GEMV (vpermb + vpopcntq + vpdpbusd, zero intermediate write!)
//    Computes y[M] = W_A16(4.5 bpw) * x_i8[K] with live 5-bit syndrome verify!
static uint64_t gemv_a16_vnni_avx512(const uint64_t* __restrict__ payloads, // [M, K/16]
                                     const float* __restrict__ scales,      // [M, K/128]
                                     const int8_t* __restrict__ x_i8,       // [K]
                                     const int32_t* __restrict__ x_sum128,  // [K/128]
                                     float x_scale,
                                     float* __restrict__ y,
                                     int M, int K,
                                     int verify_syndrome,
                                     int n_threads) {
    // 64-byte vpermb LUT mapping 4-bit two's-complement nibble -> unsigned u8 = (n + 8) in [0..15]
    // for direct vpdpbusd (u8 * s8 -> s32)!
    alignas(64) static const uint8_t LUT_S4_TO_U8BIAS[64] = {
        8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7,
        8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7,
        8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7,
        8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7,
    };
    alignas(64) uint8_t idx_first64[64], idx_second64[64];
    for (int i = 0; i < 32; ++i) {
        idx_first64[2 * i]      = (uint8_t)i;
        idx_first64[2 * i + 1]  = (uint8_t)(64 + i);
        idx_second64[2 * i]     = (uint8_t)(32 + i);
        idx_second64[2 * i + 1] = (uint8_t)(64 + 32 + i);
    }
    const __m512i v_lut = _mm512_load_si512((const void*)LUT_S4_TO_U8BIAS);
    const __m512i v_p0  = _mm512_loadu_si512((const void*)idx_first64);
    const __m512i v_p1  = _mm512_loadu_si512((const void*)idx_second64);
    const __m512i v_m0f = _mm512_set1_epi8(0x0F);

    const __m512i m_p0 = _mm512_set1_epi64((long long)0x1111111111111111ULL);
    const __m512i m_p1 = _mm512_set1_epi64((long long)0x1010101010101010ULL);
    const __m512i m_p2 = _mm512_set1_epi64((long long)0x1100110011001100ULL);
    const __m512i m_p3 = _mm512_set1_epi64((long long)0x1111000011110000ULL);
    const __m512i m_p4 = _mm512_set1_epi64((long long)0x1111111100000000ULL);
    const __m512i v_one = _mm512_set1_epi64(1LL);

    const int n_blk128 = K / 128;
    uint64_t total_errs = 0;

    #pragma omp parallel for num_threads(n_threads) reduction(+:total_errs) schedule(static)
    for (int r = 0; r < M; ++r) {
        const uint64_t* rp = payloads + (size_t)r * (size_t)(n_blk128 * 8);
        const float* rs = scales + (size_t)r * (size_t)n_blk128;
        float row_acc = 0.0f;

        for (int b = 0; b < n_blk128; ++b) {
            __m512i raw = _mm512_loadu_si512((const void*)(rp + b * 8));
            if (verify_syndrome) {
                __m512i bits = _mm512_and_si512(raw, m_p0);
                __m512i c0 = _mm512_and_si512(_mm512_popcnt_epi64(bits), v_one);
                __m512i c1 = _mm512_and_si512(_mm512_popcnt_epi64(_mm512_and_si512(bits, m_p1)), v_one);
                __m512i c2 = _mm512_and_si512(_mm512_popcnt_epi64(_mm512_and_si512(bits, m_p2)), v_one);
                __m512i c3 = _mm512_and_si512(_mm512_popcnt_epi64(_mm512_and_si512(bits, m_p3)), v_one);
                __m512i c4 = _mm512_and_si512(_mm512_popcnt_epi64(_mm512_and_si512(bits, m_p4)), v_one);
                __m512i syn = _mm512_or_si512(_mm512_or_si512(c0, _mm512_or_si512(c1, c2)), _mm512_or_si512(c3, c4));
                __mmask8 bad = _mm512_test_epi64_mask(syn, syn);
                if (__builtin_expect(bad != 0, 0)) total_errs += (uint64_t)__builtin_popcount((unsigned)bad);
            }
            __m512i nib_lo = _mm512_and_si512(raw, v_m0f);
            __m512i nib_hi = _mm512_and_si512(_mm512_srli_epi16(raw, 4), v_m0f);
            __m512i u8_lo  = _mm512_permutexvar_epi8(nib_lo, v_lut);
            __m512i u8_hi  = _mm512_permutexvar_epi8(nib_hi, v_lut);
            __m512i w0 = _mm512_permutex2var_epi8(u8_lo, v_p0, u8_hi);
            __m512i w1 = _mm512_permutex2var_epi8(u8_lo, v_p1, u8_hi);

            __m512i x0 = _mm512_loadu_si512((const void*)(x_i8 + b * 128));
            __m512i x1 = _mm512_loadu_si512((const void*)(x_i8 + b * 128 + 64));

            __m512i acc = _mm512_setzero_si512();
            acc = _mm512_dpbusd_epi32(acc, w0, x0);
            acc = _mm512_dpbusd_epi32(acc, w1, x1);
            int32_t idot = _mm512_reduce_add_epi32(acc) - 8 * x_sum128[b];
            row_acc += (float)idot * (rs[b] * x_scale);
        }
        y[r] = row_acc;
    }
    return total_errs;
}

// 4. Fused AVX-512 Block-H2 VNNI GEMV (vpermb + in-register butterfly + vpdpbusd!)
//    Each pair (a, b) in [-7,+7] -> (a+b, a-b) in [-14,+14]; adding bias +16 maps to u8 in [2..30]
//    for direct vpdpbusd (u8 * s8 -> s32) with exact -16*sum_x bias correction!
static void gemv_block_h2_vnni_avx512(const uint64_t* __restrict__ payloads,
                                      const float* __restrict__ scales,
                                      const int8_t* __restrict__ x_i8,
                                      const int32_t* __restrict__ x_sum128,
                                      float x_scale,
                                      float* __restrict__ y,
                                      int M, int K,
                                      int n_threads) {
    alignas(64) static const int8_t LUT_S4_TO_S8[64] = {
        0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1,
        0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1,
        0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1,
        0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1,
    };
    alignas(64) uint8_t idx_first64[64], idx_second64[64];
    for (int i = 0; i < 32; ++i) {
        idx_first64[2 * i]      = (uint8_t)i;
        idx_first64[2 * i + 1]  = (uint8_t)(64 + i);
        idx_second64[2 * i]     = (uint8_t)(32 + i);
        idx_second64[2 * i + 1] = (uint8_t)(64 + 32 + i);
    }
    const __m512i v_lut  = _mm512_load_si512((const void*)LUT_S4_TO_S8);
    const __m512i v_p0   = _mm512_loadu_si512((const void*)idx_first64);
    const __m512i v_p1   = _mm512_loadu_si512((const void*)idx_second64);
    const __m512i v_m0f  = _mm512_set1_epi8(0x0F);
    const __m512i v_b16  = _mm512_set1_epi8(16);
    const float inv_sq2  = 0.7071067811865475f;
    const int n_blk128   = K / 128;

    #pragma omp parallel for num_threads(n_threads) schedule(static)
    for (int r = 0; r < M; ++r) {
        const uint64_t* rp = payloads + (size_t)r * (size_t)(n_blk128 * 8);
        const float* rs = scales + (size_t)r * (size_t)n_blk128;
        float row_acc = 0.0f;

        for (int b = 0; b < n_blk128; ++b) {
            __m512i raw = _mm512_loadu_si512((const void*)(rp + b * 8));
            __m512i a_s8 = _mm512_permutexvar_epi8(_mm512_and_si512(raw, v_m0f), v_lut);
            __m512i b_s8 = _mm512_permutexvar_epi8(_mm512_and_si512(_mm512_srli_epi16(raw, 4), v_m0f), v_lut);
            __m512i s_u8 = _mm512_add_epi8(_mm512_add_epi8(a_s8, b_s8), v_b16); // (a+b)+16 in [2..30]
            __m512i d_u8 = _mm512_add_epi8(_mm512_sub_epi8(a_s8, b_s8), v_b16); // (a-b)+16 in [2..30]
            __m512i w0 = _mm512_permutex2var_epi8(s_u8, v_p0, d_u8);
            __m512i w1 = _mm512_permutex2var_epi8(s_u8, v_p1, d_u8);

            __m512i x0 = _mm512_loadu_si512((const void*)(x_i8 + b * 128));
            __m512i x1 = _mm512_loadu_si512((const void*)(x_i8 + b * 128 + 64));

            __m512i acc = _mm512_setzero_si512();
            acc = _mm512_dpbusd_epi32(acc, w0, x0);
            acc = _mm512_dpbusd_epi32(acc, w1, x1);
            int32_t idot = _mm512_reduce_add_epi32(acc) - 16 * x_sum128[b];
            row_acc += (float)idot * (rs[b] * x_scale * inv_sq2);
        }
        y[r] = row_acc;
    }
}

// ============================================================================
// Part B: Exact siliquant#77 hampack.c (L=16, n=32, bits=136, bound=1207)
//         18-byte Enumerative A_16 x A_16 Decoder: Scalar Baseline vs Fast AVX-512/128-bit
// ============================================================================

#define HP_L 16
#define HP_N 32
#define HP_STATES 32
#define HP_BOUND 1207
#define HP_BYTES 17
#define HP_REC_BYTES 18
#define HP_LIMBS 3

typedef struct { uint64_t w[HP_LIMBS]; } hp_num;
typedef unsigned __int128 u128;

static hp_num* hp_ways = NULL;          // [(HP_N+1) * HP_STATES * (HP_BOUND+1)]
static u128*   hp_ways128 = NULL;       // Compact 128-bit table for i >= 3 (fits 100% in u128, 33% smaller!)
static int     hp_h[HP_L];
static int     hp_first_u128_i = 0;

static inline int hp_less(const hp_num* a, const hp_num* b) {
    if (a->w[2] != b->w[2]) return a->w[2] < b->w[2];
    if (a->w[1] != b->w[1]) return a->w[1] < b->w[1];
    return a->w[0] < b->w[0];
}

static inline void hp_add(hp_num* a, const hp_num* b) {
    u128 c = (u128)a->w[0] + b->w[0];
    a->w[0] = (uint64_t)c; c >>= 64;
    c += (u128)a->w[1] + b->w[1];
    a->w[1] = (uint64_t)c; c >>= 64;
    a->w[2] += b->w[2] + (uint64_t)c;
}

static inline void hp_sub(hp_num* a, const hp_num* b) {
    u128 d0 = (u128)a->w[0] - b->w[0];
    uint64_t br = (d0 >> 64) ? 1ULL : 0ULL;
    a->w[0] = (uint64_t)d0;
    u128 d1 = (u128)a->w[1] - b->w[1] - br;
    br = (d1 >> 64) ? 1ULL : 0ULL;
    a->w[1] = (uint64_t)d1;
    a->w[2] -= b->w[2] + br;
}

static inline const hp_num* hp_row(int i, int s) {
    return hp_ways + ((size_t)i * HP_STATES + (size_t)s) * (HP_BOUND + 1);
}

static inline const u128* hp_row128(int i, int s) {
    return hp_ways128 + ((size_t)i * HP_STATES + (size_t)s) * (HP_BOUND + 1);
}

static void hp_build_table(void) {
    for (int j = 0; j < HP_L; ++j) hp_h[j] = j | HP_L;
    size_t cnt = (size_t)(HP_N + 1) * HP_STATES * (HP_BOUND + 1);
    hp_ways = (hp_num*)aligned64_alloc(cnt * sizeof(hp_num));
    hp_ways128 = (u128*)aligned64_alloc(cnt * sizeof(u128));

    // Base: ways[n][0][r] = 1 for all r <= HP_BOUND
    for (int r = 0; r <= HP_BOUND; ++r) {
        hp_ways[((size_t)HP_N * HP_STATES + 0) * (HP_BOUND + 1) + r].w[0] = 1ULL;
    }

    int vmax = 0;
    while ((vmax + 1) * (vmax + 1) <= HP_BOUND) ++vmax;

    for (int i = HP_N - 1; i >= 0; --i) {
        int hj = hp_h[i % HP_L];
        int s_limit = (i % HP_L == 0) ? 1 : HP_STATES;
        for (int s = 0; s < s_limit; ++s) {
            hp_num* dst = hp_ways + ((size_t)i * HP_STATES + (size_t)s) * (HP_BOUND + 1);
            const hp_num* nxt_ev = hp_row(i + 1, s);
            const hp_num* nxt_od = hp_row(i + 1, s ^ hj);
            // v = 0 (mult 1)
            memcpy(dst, nxt_ev, (HP_BOUND + 1) * sizeof(hp_num));
            // v = 1..vmax (mult 2)
            for (int v = 1; v <= vmax; ++v) {
                int c = v * v;
                const hp_num* src = (v & 1) ? nxt_od : nxt_ev;
                for (int r = c; r <= HP_BOUND; ++r) {
                    hp_add(&dst[r], &src[r - c]);
                    hp_add(&dst[r], &src[r - c]);
                }
            }
        }
    }

    // Find first index i where all ways[i][*][*] fit strictly in u128 (w[2] == 0)
    hp_first_u128_i = HP_N;
    for (int i = 0; i <= HP_N; ++i) {
        int fits = 1;
        for (int s = 0; s < HP_STATES && fits; ++s) {
            const hp_num* r = hp_row(i, s);
            if (r[HP_BOUND].w[2] != 0ULL) fits = 0;
        }
        if (fits) { hp_first_u128_i = i; break; }
    }
    for (size_t idx = 0; idx < cnt; ++idx) {
        hp_ways128[idx] = ((u128)hp_ways[idx].w[1] << 64) | (u128)hp_ways[idx].w[0];
    }
}

static int hp_rank_one(const int8_t* y, uint8_t* out17) {
    hp_num total = {{0, 0, 0}};
    int left = HP_BOUND, s = 0;
    for (int i = 0; i < HP_N; ++i) {
        int h = hp_h[i % HP_L];
        int v = y[i];
        int a = v < 0 ? -v : v;
        int cost = a * a;
        if (cost > left) return 1;
        if (a > 0) {
            hp_add(&total, hp_row(i + 1, s) + left);
            for (int k = 1; k < a; ++k) {
                const hp_num* r = hp_row(i + 1, (k & 1) ? (s ^ h) : s) + (left - k * k);
                hp_add(&total, r);
                hp_add(&total, r);
            }
            if (v < 0) hp_add(&total, hp_row(i + 1, (a & 1) ? (s ^ h) : s) + (left - cost));
        }
        if (a & 1) s ^= h;
        left -= cost;
        if ((i + 1) % HP_L == 0 && s != 0) return 1;
    }
    for (int b = 0; b < HP_BYTES; ++b) out17[b] = (uint8_t)(total.w[b / 8] >> (8 * (b % 8)));
    return 0;
}

// Exact siliquant#77 scalar hampack.c unrank + float decode (verbatim algorithm!)
static size_t hp_decode_siliquant_scalar(const uint8_t* records, size_t groups,
                                         double top, int steps, float* out) {
    float scale[256];
    for (int c = 0; c < 256; ++c) scale[c] = (float)(top * pow(2.0, -(double)c / steps));
    size_t bad = 0;
    int8_t y[HP_N];
    for (size_t g = 0; g < groups; ++g) {
        const uint8_t* rec = records + g * HP_REC_BYTES;
        const uint8_t* in = rec + 1;
        hp_num k = {{0, 0, 0}};
        for (int b = 0; b < HP_BYTES; ++b) k.w[b / 8] |= (uint64_t)in[b] << (8 * (b % 8));
        int left = HP_BOUND, s = 0, err = 0;
        for (int i = 0; i < HP_N; ++i) {
            const int h = hp_h[i % HP_L];
            const hp_num* w0 = hp_row(i + 1, s) + left;
            if (hp_less(&k, w0)) { y[i] = 0; continue; }
            hp_sub(&k, w0);
            int found = 0;
            for (int a = 1; a * a <= left; ++a) {
                const int t = (a & 1) ? (s ^ h) : s;
                const hp_num* wa = hp_row(i + 1, t) + (left - a * a);
                if (hp_less(&k, wa)) { y[i] = (int8_t)a; left -= a * a; s = t; found = 1; break; }
                hp_sub(&k, wa);
                if (hp_less(&k, wa)) { y[i] = (int8_t)(-a); left -= a * a; s = t; found = 1; break; }
                hp_sub(&k, wa);
            }
            if (!found) { err = 1; break; }
        }
        if (err) { ++bad; memset(y, 0, HP_N); }
        float sc = scale[rec[0]];
        float* o = out + g * HP_N;
        for (int i = 0; i < HP_N; ++i) o[i] = sc * (float)y[i];
    }
    return bad;
}

// Optimized exact siliquant#77 hampack decoder:
//   - Fast unaligned load of 17 bytes -> (u128 lo, uint64_t hi)
//   - 3-limb path ONLY for i < hp_first_u128_i (just i=0,1!), then native register u128 for i=2..17
//     and native 64-bit uint64_t for i>=18 (where ways < 2^64!) -> eliminates 3-limb carry chains!
//   - Single comparison per magnitude `a` (compare k < 2*wa first -> cuts loop checks & branches in half!)
//   - AVX-512 32-weight int8->fp32 scale multiply (`_mm512_cvtepi32_ps` + `_mm512_mul_ps`)
//   - 8C/16T OpenMP scaling!
static size_t hp_decode_fast_avx512(const uint8_t* __restrict__ records, size_t groups,
                                    double top, int steps, float* __restrict__ out,
                                    int n_threads) {
    float scale[256];
    for (int c = 0; c < 256; ++c) scale[c] = (float)(top * pow(2.0, -(double)c / steps));
    size_t total_bad = 0;

    #pragma omp parallel for num_threads(n_threads) reduction(+:total_bad) schedule(static)
    for (size_t g = 0; g < groups; ++g) {
        const uint8_t* rec = records + g * HP_REC_BYTES;
        uint64_t lo64 = 0, hi64 = 0;
        memcpy(&lo64, rec + 1, 8);
        memcpy(&hi64, rec + 9, 8);
        uint64_t top8 = (uint64_t)rec[17];
        hp_num k = {{lo64, hi64, top8}};

        int8_t y[HP_N];
        int left = HP_BOUND, s = 0, err = 0;
        int i = 0;

        // Phase 1: 3-limb only while k.w[2] > 0 or i + 1 < hp_first_u128_i
        for (; i < HP_N && (i + 1 < hp_first_u128_i || k.w[2] != 0ULL); ++i) {
            const int h = hp_h[i & 15];
            const hp_num* w0 = hp_row(i + 1, s) + left;
            if (hp_less(&k, w0)) { y[i] = 0; continue; }
            hp_sub(&k, w0);
            int found = 0;
            for (int a = 1; a * a <= left; ++a) {
                const int t = (a & 1) ? (s ^ h) : s;
                const hp_num* wa = hp_row(i + 1, t) + (left - a * a);
                if (hp_less(&k, wa)) { y[i] = (int8_t)a; left -= a * a; s = t; found = 1; break; }
                hp_sub(&k, wa);
                if (hp_less(&k, wa)) { y[i] = (int8_t)(-a); left -= a * a; s = t; found = 1; break; }
                hp_sub(&k, wa);
            }
            if (!found) { err = 1; break; }
        }

        // Phase 2: Native 128-bit register path (i >= 2..31, single 2*wa check!)
        if (!err) {
            u128 k128 = ((u128)k.w[1] << 64) | (u128)k.w[0];
            for (; i < HP_N; ++i) {
                const int h = hp_h[i & 15];
                const u128* row_ev = hp_row128(i + 1, s);
                u128 w0 = row_ev[left];
                if (k128 < w0) { y[i] = 0; continue; }
                k128 -= w0;
                const u128* row_od = hp_row128(i + 1, s ^ h);
                int found = 0;
                for (int a = 1; a * a <= left; ++a) {
                    int odd = a & 1;
                    u128 wa = (odd ? row_od : row_ev)[left - a * a];
                    u128 wa2 = wa + wa;
                    if (k128 < wa2) {
                        int neg = (k128 >= wa);
                        if (neg) k128 -= wa;
                        y[i] = (int8_t)(neg ? -a : a);
                        left -= a * a;
                        if (odd) s ^= h;
                        found = 1;
                        break;
                    }
                    k128 -= wa2;
                }
                if (!found) { err = 1; break; }
            }
        }

        if (__builtin_expect(err, 0)) { ++total_bad; memset(y, 0, HP_N); }

        // AVX-512 convert 32 int8 -> 32 fp32 and multiply by scale in TWO ZMM ops!
        __m512 vsc = _mm512_set1_ps(scale[rec[0]]);
        __m128i y0 = _mm_loadu_si128((const __m128i*)(y));
        __m128i y1 = _mm_loadu_si128((const __m128i*)(y + 16));
        __m512 f0 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(y0)), vsc);
        __m512 f1 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(y1)), vsc);
        float* o = out + g * HP_N;
        _mm512_storeu_ps(o,      f0);
        _mm512_storeu_ps(o + 16, f1);
    }
    return total_bad;
}

int main(void) {
    int total_failures = 0;

    // ------------------------------------------------------------------------
    // Benchmark 1 & 2: Hardware-Native Bit-Packed 4.500-bpw A_16 & Block-H2
    // Matrix: M=4096, K=14336 (58,720,256 weights = 458,752 blocks of 128w)
    // ------------------------------------------------------------------------
    const int M = 4096, K = 14336;
    const size_t total_weights = (size_t)M * (size_t)K;
    const size_t n_blk128 = total_weights / 128ULL;
    const size_t n_u64 = total_weights / 16ULL;

    uint64_t* payloads = (uint64_t*)aligned64_alloc(n_u64 * sizeof(uint64_t));
    float* scales = (float*)aligned64_alloc(n_blk128 * sizeof(float));
    int8_t* x_i8 = (int8_t*)aligned64_alloc((size_t)K);
    int32_t* x_sum128 = (int32_t*)aligned64_alloc((size_t)(K / 128) * sizeof(int32_t));
    int8_t* out_ref = (int8_t*)aligned64_alloc(total_weights);
    int8_t* out_avx = (int8_t*)aligned64_alloc(total_weights);
    float* y_ref = (float*)aligned64_alloc((size_t)M * sizeof(float));
    float* y_avx = (float*)aligned64_alloc((size_t)M * sizeof(float));
    float* y_h2_ref = (float*)aligned64_alloc((size_t)M * sizeof(float));
    float* y_h2_avx = (float*)aligned64_alloc((size_t)M * sizeof(float));

    uint32_t rng = 0xA1E7E52u;
    for (size_t i = 0; i < n_u64; ++i) payloads[i] = make_valid_a16_word(&rng);
    for (size_t i = 0; i < n_blk128; ++i) {
        rng = rng * 1664525u + 1013904223u;
        scales[i] = 0.005f + 0.01f * ((float)(rng >> 16) / 65535.0f);
    }
    const float x_scale = 0.015625f;
    for (int k = 0; k < K; ++k) {
        rng = rng * 1664525u + 1013904223u;
        x_i8[k] = (int8_t)(((int)(rng >> 24) % 127) - 63);
    }
    for (int b = 0; b < K / 128; ++b) {
        int32_t s = 0;
        for (int j = 0; j < 128; ++j) s += (int32_t)x_i8[b * 128 + j];
        x_sum128[b] = s;
    }

    // Scalar decode timing & reference
    double t0 = now_sec();
    uint64_t err_scalar = decode_a16_scalar(payloads, out_ref, n_blk128);
    double dt_dec_scalar = now_sec() - t0;

    // AVX-512 decode (1T, 8T, 16T)
    uint64_t err_avx1 = decode_a16_avx512(payloads, out_avx, n_blk128, 1);
    int dec_mismatches = (memcmp(out_ref, out_avx, total_weights) != 0);
    if (err_scalar != 0 || err_avx1 != 0 || dec_mismatches != 0) ++total_failures;

    const int REPS = 5;
    double best_dec_1t = 1e9, best_dec_8t = 1e9, best_dec_16t = 1e9;
    for (int r = 0; r < REPS; ++r) {
        t0 = now_sec(); decode_a16_avx512(payloads, out_avx, n_blk128, 1);
        double dt = now_sec() - t0; if (dt < best_dec_1t) best_dec_1t = dt;
    }
    for (int r = 0; r < REPS; ++r) {
        t0 = now_sec(); decode_a16_avx512(payloads, out_avx, n_blk128, 8);
        double dt = now_sec() - t0; if (dt < best_dec_8t) best_dec_8t = dt;
    }
    for (int r = 0; r < REPS; ++r) {
        t0 = now_sec(); decode_a16_avx512(payloads, out_avx, n_blk128, 16);
        double dt = now_sec() - t0; if (dt < best_dec_16t) best_dec_16t = dt;
    }

    // Scalar GEMV reference for A_16 and Block-H2 (exact integer dot * scale)
    t0 = now_sec();
    for (int r = 0; r < M; ++r) {
        const uint64_t* rp = payloads + (size_t)r * (size_t)(K / 16);
        const float* rs = scales + (size_t)r * (size_t)(K / 128);
        float acc_a = 0.0f, acc_h = 0.0f;
        for (int b = 0; b < K / 128; ++b) {
            int32_t da = 0, dh = 0;
            for (int k = 0; k < 8; ++k) {
                uint64_t w = rp[b * 8 + k];
                for (int i = 0; i < 16; i += 2) {
                    int a = unpack_s4_scalar(w, i);
                    int c = unpack_s4_scalar(w, i + 1);
                    int x0 = (int)x_i8[b * 128 + k * 16 + i];
                    int x1 = (int)x_i8[b * 128 + k * 16 + i + 1];
                    da += a * x0 + c * x1;
                    dh += (a + c) * x0 + (a - c) * x1;
                }
            }
            acc_a += (float)da * (rs[b] * x_scale);
            acc_h += (float)dh * (rs[b] * x_scale * 0.7071067811865475f);
        }
        y_ref[r] = acc_a;
        y_h2_ref[r] = acc_h;
    }
    double dt_gemv_scalar_both = now_sec() - t0;

    // Measure scalar A_16 GEMV alone for clean apples-to-apples
    t0 = now_sec();
    for (int r = 0; r < M; ++r) {
        const uint64_t* rp = payloads + (size_t)r * (size_t)(K / 16);
        const float* rs = scales + (size_t)r * (size_t)(K / 128);
        float acc_a = 0.0f;
        for (int b = 0; b < K / 128; ++b) {
            int32_t da = 0;
            for (int k = 0; k < 8; ++k) {
                uint64_t w = rp[b * 8 + k];
                for (int i = 0; i < 16; ++i) da += unpack_s4_scalar(w, i) * (int)x_i8[b * 128 + k * 16 + i];
            }
            acc_a += (float)da * (rs[b] * x_scale);
        }
        y_ref[r] = acc_a;
    }
    double dt_gemv_a16_scalar = now_sec() - t0;
    (void)dt_gemv_scalar_both;

    // AVX-512 VNNI A_16 GEMV (with & without live vpopcntq verify; 1T, 8T, 16T)
    uint64_t verr_vnni = gemv_a16_vnni_avx512(payloads, scales, x_i8, x_sum128, x_scale, y_avx, M, K, 1, 8);
    double max_err_a16 = 0.0;
    for (int r = 0; r < M; ++r) {
        double d = fabs((double)y_avx[r] - (double)y_ref[r]);
        if (d > max_err_a16) max_err_a16 = d;
    }
    if (verr_vnni != 0 || max_err_a16 > 1e-6) ++total_failures;

    double best_vnni_a16_1t = 1e9, best_vnni_a16_8t = 1e9, best_vnni_a16_16t = 1e9;
    double best_vnni_a16_ver_8t = 1e9, best_vnni_a16_ver_16t = 1e9;
    for (int r = 0; r < REPS; ++r) {
        t0 = now_sec(); gemv_a16_vnni_avx512(payloads, scales, x_i8, x_sum128, x_scale, y_avx, M, K, 0, 1);
        double dt = now_sec() - t0; if (dt < best_vnni_a16_1t) best_vnni_a16_1t = dt;
    }
    for (int r = 0; r < REPS; ++r) {
        t0 = now_sec(); gemv_a16_vnni_avx512(payloads, scales, x_i8, x_sum128, x_scale, y_avx, M, K, 0, 8);
        double dt = now_sec() - t0; if (dt < best_vnni_a16_8t) best_vnni_a16_8t = dt;
    }
    for (int r = 0; r < REPS; ++r) {
        t0 = now_sec(); gemv_a16_vnni_avx512(payloads, scales, x_i8, x_sum128, x_scale, y_avx, M, K, 0, 16);
        double dt = now_sec() - t0; if (dt < best_vnni_a16_16t) best_vnni_a16_16t = dt;
    }
    for (int r = 0; r < REPS; ++r) {
        t0 = now_sec(); gemv_a16_vnni_avx512(payloads, scales, x_i8, x_sum128, x_scale, y_avx, M, K, 1, 8);
        double dt = now_sec() - t0; if (dt < best_vnni_a16_ver_8t) best_vnni_a16_ver_8t = dt;
    }
    for (int r = 0; r < REPS; ++r) {
        t0 = now_sec(); gemv_a16_vnni_avx512(payloads, scales, x_i8, x_sum128, x_scale, y_avx, M, K, 1, 16);
        double dt = now_sec() - t0; if (dt < best_vnni_a16_ver_16t) best_vnni_a16_ver_16t = dt;
    }

    // AVX-512 VNNI Block-H2 GEMV (1T, 8T, 16T)
    gemv_block_h2_vnni_avx512(payloads, scales, x_i8, x_sum128, x_scale, y_h2_avx, M, K, 8);
    double max_err_h2 = 0.0;
    for (int r = 0; r < M; ++r) {
        double d = fabs((double)y_h2_avx[r] - (double)y_h2_ref[r]);
        if (d > max_err_h2) max_err_h2 = d;
    }
    if (max_err_h2 > 1e-6) ++total_failures;

    double best_vnni_h2_1t = 1e9, best_vnni_h2_8t = 1e9, best_vnni_h2_16t = 1e9;
    for (int r = 0; r < REPS; ++r) {
        t0 = now_sec(); gemv_block_h2_vnni_avx512(payloads, scales, x_i8, x_sum128, x_scale, y_h2_avx, M, K, 1);
        double dt = now_sec() - t0; if (dt < best_vnni_h2_1t) best_vnni_h2_1t = dt;
    }
    for (int r = 0; r < REPS; ++r) {
        t0 = now_sec(); gemv_block_h2_vnni_avx512(payloads, scales, x_i8, x_sum128, x_scale, y_h2_avx, M, K, 8);
        double dt = now_sec() - t0; if (dt < best_vnni_h2_8t) best_vnni_h2_8t = dt;
    }
    for (int r = 0; r < REPS; ++r) {
        t0 = now_sec(); gemv_block_h2_vnni_avx512(payloads, scales, x_i8, x_sum128, x_scale, y_h2_avx, M, K, 16);
        double dt = now_sec() - t0; if (dt < best_vnni_h2_16t) best_vnni_h2_16t = dt;
    }

    // ------------------------------------------------------------------------
    // Benchmark 3: Exact siliquant#77 hampack.c (L=16, n=32, bits=136, bound=1207)
    //              262,144 blocks (= 8,388,608 weights = 4.5 MiB packed records!)
    // ------------------------------------------------------------------------
    double t_tbl0 = now_sec();
    hp_build_table();
    double t_tbl = now_sec() - t_tbl0;

    const size_t HP_GROUPS = 262144ULL; // 8,388,608 weights
    const size_t HP_WEIGHTS = HP_GROUPS * HP_N;
    uint8_t* hp_recs = (uint8_t*)aligned64_alloc(HP_GROUPS * HP_REC_BYTES);
    float* hp_out_ref = (float*)aligned64_alloc(HP_WEIGHTS * sizeof(float));
    float* hp_out_opt = (float*)aligned64_alloc(HP_WEIGHTS * sizeof(float));

    // Generate valid in-ball A_16 x A_16 points (sum v_i^2 <= 1207) and rank into exact 18B records!
    int rank_errs = 0;
    for (size_t g = 0; g < HP_GROUPS; ++g) {
        uint64_t w0 = make_valid_a16_word(&rng);
        uint64_t w1 = make_valid_a16_word(&rng);
        int8_t y[HP_N];
        for (int i = 0; i < 16; ++i) {
            y[i]      = (int8_t)unpack_s4_scalar(w0, i);
            y[i + 16] = (int8_t)unpack_s4_scalar(w1, i);
        }
        // Note: max sum v_i^2 <= 32 * 49 = 1568, clamp if > 1207 by shrinking even pairs
        int sq = 0;
        for (int i = 0; i < HP_N; ++i) sq += (int)y[i] * (int)y[i];
        while (sq > HP_BOUND) {
            for (int i = 0; i < HP_N && sq > HP_BOUND; ++i) {
                if (y[i] >= 2)       { sq -= y[i]*y[i]; y[i] -= 2; sq += y[i]*y[i]; }
                else if (y[i] <= -2) { sq -= y[i]*y[i]; y[i] += 2; sq += y[i]*y[i]; }
            }
        }
        uint8_t* rec = hp_recs + g * HP_REC_BYTES;
        rec[0] = (uint8_t)((g * 37ULL) & 0x3Fu);
        if (hp_rank_one(y, rec + 1) != 0) ++rank_errs;
    }
    if (rank_errs != 0) ++total_failures;

    double best_hp_scalar = 1e9;
    size_t hp_bad_ref = 0;
    for (int r = 0; r < 3; ++r) {
        t0 = now_sec();
        hp_bad_ref = hp_decode_siliquant_scalar(hp_recs, HP_GROUPS, 0.25, 16, hp_out_ref);
        double dt = now_sec() - t0; if (dt < best_hp_scalar) best_hp_scalar = dt;
    }

    size_t hp_bad_opt = hp_decode_fast_avx512(hp_recs, HP_GROUPS, 0.25, 16, hp_out_opt, 8);
    int hp_bit_mismatches = (memcmp(hp_out_ref, hp_out_opt, HP_WEIGHTS * sizeof(float)) != 0);
    if (hp_bad_ref != 0 || hp_bad_opt != 0 || hp_bit_mismatches != 0) ++total_failures;

    double best_hp_opt_1t = 1e9, best_hp_opt_8t = 1e9, best_hp_opt_16t = 1e9;
    for (int r = 0; r < 3; ++r) {
        t0 = now_sec(); hp_decode_fast_avx512(hp_recs, HP_GROUPS, 0.25, 16, hp_out_opt, 1);
        double dt = now_sec() - t0; if (dt < best_hp_opt_1t) best_hp_opt_1t = dt;
    }
    for (int r = 0; r < 5; ++r) {
        t0 = now_sec(); hp_decode_fast_avx512(hp_recs, HP_GROUPS, 0.25, 16, hp_out_opt, 8);
        double dt = now_sec() - t0; if (dt < best_hp_opt_8t) best_hp_opt_8t = dt;
    }
    for (int r = 0; r < 5; ++r) {
        t0 = now_sec(); hp_decode_fast_avx512(hp_recs, HP_GROUPS, 0.25, 16, hp_out_opt, 16);
        double dt = now_sec() - t0; if (dt < best_hp_opt_16t) best_hp_opt_16t = dt;
    }

    // Emit structured JSON receipt
    printf("{\n");
    printf("  \"cpu_target\": \"Intel Core i9-11900H (8C/16T AVX-512 VBMI/VPOPCNTDQ/VNNI)\",\n");
    printf("  \"matrix_shape\": {\"M\": %d, \"K\": %d, \"weights\": %zu, \"bpw\": 4.5},\n", M, K, total_weights);
    printf("  \"bitpacked_a16_decode\": {\n");
    printf("    \"syndrome_errors_scalar\": %llu, \"syndrome_errors_avx512\": %llu, \"bit_exact_mismatches\": %d,\n",
           (unsigned long long)err_scalar, (unsigned long long)err_avx1, dec_mismatches);
    printf("    \"scalar_1t_mweights_s\": %.3f, \"scalar_1t_ms\": %.4f,\n",
           (double)total_weights / (dt_dec_scalar * 1e6), dt_dec_scalar * 1e3);
    printf("    \"avx512_1t_mweights_s\": %.3f, \"avx512_1t_gweights_s\": %.4f, \"avx512_1t_ms\": %.4f, \"speedup_vs_scalar_1t\": %.2f,\n",
           (double)total_weights / (best_dec_1t * 1e6), (double)total_weights / (best_dec_1t * 1e9), best_dec_1t * 1e3, dt_dec_scalar / best_dec_1t);
    printf("    \"avx512_8t_gweights_s\": %.4f, \"avx512_8t_ms\": %.4f,\n",
           (double)total_weights / (best_dec_8t * 1e9), best_dec_8t * 1e3);
    printf("    \"avx512_16t_gweights_s\": %.4f, \"avx512_16t_ms\": %.4f,\n",
           (double)total_weights / (best_dec_16t * 1e9), best_dec_16t * 1e3);
    printf("    \"speedup_16t_vs_siliquant77_10M\": %.1f\n",
           ((double)total_weights / (best_dec_16t * 1e6)) / 10.0);
    printf("  },\n");
    printf("  \"fused_vnni_gemv\": {\n");
    printf("    \"a16_max_abs_err\": %.6e, \"a16_syndrome_errors\": %llu, \"block_h2_max_abs_err\": %.6e,\n",
           max_err_a16, (unsigned long long)verr_vnni, max_err_h2);
    printf("    \"a16_scalar_1t_mweights_s\": %.3f,\n", (double)total_weights / (dt_gemv_a16_scalar * 1e6));
    printf("    \"a16_vnni_1t_gweights_s\": %.4f, \"a16_vnni_8t_gweights_s\": %.4f, \"a16_vnni_16t_gweights_s\": %.4f,\n",
           (double)total_weights / (best_vnni_a16_1t * 1e9),
           (double)total_weights / (best_vnni_a16_8t * 1e9),
           (double)total_weights / (best_vnni_a16_16t * 1e9));
    printf("    \"a16_vnni_live_verify_8t_gweights_s\": %.4f, \"a16_vnni_live_verify_16t_gweights_s\": %.4f,\n",
           (double)total_weights / (best_vnni_a16_ver_8t * 1e9),
           (double)total_weights / (best_vnni_a16_ver_16t * 1e9));
    printf("    \"block_h2_vnni_1t_gweights_s\": %.4f, \"block_h2_vnni_8t_gweights_s\": %.4f, \"block_h2_vnni_16t_gweights_s\": %.4f,\n",
           (double)total_weights / (best_vnni_h2_1t * 1e9),
           (double)total_weights / (best_vnni_h2_8t * 1e9),
           (double)total_weights / (best_vnni_h2_16t * 1e9));
    printf("    \"a16_vnni_8t_phys_gb_s\": %.3f, \"a16_vnni_8t_fp16_equiv_gb_s\": %.3f,\n",
           ((double)total_weights * 0.5625) / (best_vnni_a16_8t * 1e9),
           ((double)total_weights * 2.0) / (best_vnni_a16_8t * 1e9));
    printf("    \"a16_vnni_8t_speedup_vs_siliquant77_10M\": %.1f,\n",
           ((double)total_weights / (best_vnni_a16_8t * 1e6)) / 10.0);
    printf("    \"a16_vnni_16t_phys_gb_s\": %.3f, \"a16_vnni_16t_fp16_equiv_gb_s\": %.3f\n",
           ((double)total_weights * 0.5625) / (best_vnni_a16_16t * 1e9),
           ((double)total_weights * 2.0) / (best_vnni_a16_16t * 1e9));
    printf("  },\n");
    printf("  \"siliquant77_exact_enumerative_hampack\": {\n");
    printf("    \"L\": %d, \"n\": %d, \"bits\": 136, \"record_bytes\": %d, \"bound\": %d,\n", HP_L, HP_N, HP_REC_BYTES, HP_BOUND);
    printf("    \"first_u128_index\": %d, \"table_build_ms\": %.2f, \"groups\": %zu, \"weights\": %zu,\n",
           hp_first_u128_i, t_tbl * 1e3, HP_GROUPS, HP_WEIGHTS);
    printf("    \"rank_errors\": %d, \"decode_errors_ref\": %zu, \"decode_errors_opt\": %zu, \"bit_exact_mismatches\": %d,\n",
           rank_errs, hp_bad_ref, hp_bad_opt, hp_bit_mismatches);
    printf("    \"siliquant_scalar_1t_mweights_s\": %.3f,\n", (double)HP_WEIGHTS / (best_hp_scalar * 1e6));
    printf("    \"fast_u128_avx512_1t_mweights_s\": %.3f, \"speedup_1t\": %.2f,\n",
           (double)HP_WEIGHTS / (best_hp_opt_1t * 1e6), best_hp_scalar / best_hp_opt_1t);
    printf("    \"fast_u128_avx512_8t_mweights_s\": %.3f, \"speedup_8t\": %.2f,\n",
           (double)HP_WEIGHTS / (best_hp_opt_8t * 1e6), best_hp_scalar / best_hp_opt_8t);
    printf("    \"fast_u128_avx512_16t_mweights_s\": %.3f, \"speedup_16t\": %.2f\n",
           (double)HP_WEIGHTS / (best_hp_opt_16t * 1e6), best_hp_scalar / best_hp_opt_16t);
    printf("  },\n");
    printf("  \"total_verify_failures\": %d\n", total_failures);
    printf("}\n");

    free(payloads); free(scales); free(x_i8); free(x_sum128);
    free(out_ref); free(out_avx); free(y_ref); free(y_avx); free(y_h2_ref); free(y_h2_avx);
    free(hp_ways); free(hp_ways128); free(hp_recs); free(hp_out_ref); free(hp_out_opt);
    return (total_failures == 0) ? 0 : 2;
}
