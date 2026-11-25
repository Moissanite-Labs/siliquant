/*
 * Reference Intel AVX-512 (vpermb + vpopcntq + vpdpbusd VNNI) A_16 [16, 11, 4] + Block-H4 Decoder
 * Part of the `siliquant-hf-v1` release bundle (Moissanite-Labs/siliquant).
 *
 * Features:
 *   - Hardware 5-bit Extended Hamming [16, 11, 4] syndrome verification on 8 x A_16 blocks
 *     in parallel using AVX-512VPOPCNTDQ (`_mm512_popcnt_epi64`) with scalar fallback.
 *   - In-register 64-byte `vpermb` nibble-to-signed-int8 LUT unpack + exact register-only
 *     4-point Sylvester Hadamard butterfly H_4 (peak |r_i| <= 28 <= 127).
 *   - Fused AVX-512VNNI `vpdpbusd` INT8 GEMV (`_mm512_dpbusd_epi32`) with exact +128 offset
 *     cancellation against CPU scalar int64 reference.
 *
 * Build & self-test:
 *   gcc -O3 -march=native avx512_a16_blockh4.c -lm -o avx512_a16_blockh4
 *   ./avx512_a16_blockh4
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VBMI__) && \
    defined(__AVX512VPOPCNTDQ__) && defined(__AVX512VNNI__)
#include <immintrin.h>
#define SILIQUANT_HAS_AVX512_FULL 1
#else
#define SILIQUANT_HAS_AVX512_FULL 0
#endif

static inline int unpack_s4(uint64_t w64, int idx) {
    uint32_t nib = (uint32_t)((w64 >> (idx * 4)) & 0xFu);
    return (nib & 8u) ? (int)(nib | 0xFFFFFFF0u) : (int)nib;
}

static inline uint32_t a16_syndrome_scalar(uint64_t w64) {
    uint64_t p = w64 & 0x1111111111111111ULL;
    uint32_t p0 = (uint32_t)__builtin_popcountll(p) & 1u;
    uint32_t s0 = (uint32_t)__builtin_popcountll(p & 0x1010101010101010ULL) & 1u;
    uint32_t s1 = (uint32_t)__builtin_popcountll(p & 0x1100110011001100ULL) & 1u;
    uint32_t s2 = (uint32_t)__builtin_popcountll(p & 0x1111000011110000ULL) & 1u;
    uint32_t s3 = (uint32_t)__builtin_popcountll(p & 0x1111111100000000ULL) & 1u;
    return p0 | (s0 << 1) | (s1 << 2) | (s2 << 3) | (s3 << 4);
}

static uint64_t make_valid_a16_word(uint32_t* rng) {
    int v[16];
    uint32_t bits = 0u;
    int min_abs = 256, min_idx = 0;
    for (int i = 0; i < 16; ++i) {
        *rng = (*rng) * 1664525u + 1013904223u;
        v[i] = ((int)((*rng) >> 24) % 15) - 7;
        bits |= ((uint32_t)(v[i] & 1)) << i;
        int av = v[i] < 0 ? -v[i] : v[i];
        if (av < min_abs) {
            min_abs = av;
            min_idx = i;
        }
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
    uint64_t w64 = 0ULL;
    for (int i = 0; i < 16; ++i) w64 |= ((uint64_t)(v[i] & 0xF)) << (i * 4);
    return w64;
}

static uint64_t decode_blockh4_gemv_scalar(
    const uint64_t* packed,
    const int8_t* x_i8,
    int64_t* y_out,
    int M,
    int K) {
    const int n_blk16 = K / 16;
    uint64_t syn_errs = 0;
    for (int r = 0; r < M; ++r) {
        int64_t acc = 0;
        const uint64_t* row = packed + (size_t)r * n_blk16;
        for (int b = 0; b < n_blk16; ++b) {
            uint64_t w64 = row[b];
            if (a16_syndrome_scalar(w64) != 0u) ++syn_errs;
            for (int g = 0; g < 4; ++g) {
                int c0 = unpack_s4(w64, g * 4 + 0);
                int c1 = unpack_s4(w64, g * 4 + 1);
                int c2 = unpack_s4(w64, g * 4 + 2);
                int c3 = unpack_s4(w64, g * 4 + 3);
                int s01 = c0 + c1, d01 = c0 - c1;
                int s23 = c2 + c3, d23 = c2 - c3;
                int base = b * 16 + g * 4;
                acc += (int64_t)(s01 + s23) * x_i8[base + 0]
                     + (int64_t)(d01 + d23) * x_i8[base + 1]
                     + (int64_t)(s01 - s23) * x_i8[base + 2]
                     + (int64_t)(d01 - d23) * x_i8[base + 3];
            }
        }
        y_out[r] = acc;
    }
    return syn_errs;
}

#if SILIQUANT_HAS_AVX512_FULL
static inline uint64_t verify_8x_a16_avx512(const uint64_t* blk8) {
    const __m512i v = _mm512_loadu_si512((const void*)blk8);
    const __m512i p = _mm512_and_si512(v, _mm512_set1_epi64(0x1111111111111111LL));
    const __m512i ones = _mm512_set1_epi64(1LL);
    __m512i e0 = _mm512_and_si512(_mm512_popcnt_epi64(p), ones);
    __m512i e1 = _mm512_and_si512(_mm512_popcnt_epi64(_mm512_and_si512(p, _mm512_set1_epi64(0x1010101010101010LL))), ones);
    __m512i e2 = _mm512_and_si512(_mm512_popcnt_epi64(_mm512_and_si512(p, _mm512_set1_epi64(0x1100110011001100LL))), ones);
    __m512i e3 = _mm512_and_si512(_mm512_popcnt_epi64(_mm512_and_si512(p, _mm512_set1_epi64(0x1111000011110000LL))), ones);
    __m512i e4 = _mm512_and_si512(_mm512_popcnt_epi64(_mm512_and_si512(p, _mm512_set1_epi64(0x1111111100000000LL))), ones);
    __m512i any = _mm512_or_si512(_mm512_or_si512(_mm512_or_si512(e0, e1), _mm512_or_si512(e2, e3)), e4);
    __mmask8 bad = _mm512_test_epi64_mask(any, any);
    return (uint64_t)__builtin_popcount((unsigned)bad);
}

static uint64_t decode_blockh4_gemv_avx512(
    const uint64_t* packed,
    const int8_t* x_i8,
    int64_t* y_out,
    int M,
    int K) {
    const int n_blk16 = K / 16;
    const __m512i mask_lo = _mm512_set1_epi8(0x0F);
    const __m512i bias128 = _mm512_set1_epi8((char)0x80);
    int8_t lut_arr[64];
    for (int i = 0; i < 64; ++i) {
        int nib = i & 0xF;
        lut_arr[i] = (int8_t)((nib & 8) ? (nib - 16) : nib);
    }
    const __m512i lut = _mm512_loadu_si512((const void*)lut_arr);

    // Precompute sum(x_i8) ONLY across bulk AVX-512 128-weight chunks for exact +128 bias cancellation.
    const int n_bulk_blk = n_blk16 & ~7;
    const int k_bulk = n_bulk_blk * 16;
    int64_t x_sum_bulk = 0;
    for (int i = 0; i < k_bulk; ++i) x_sum_bulk += (int64_t)x_i8[i];

    // Indices to gather lane positions 0,1,2,3 across 16 quads inside a 64-byte ZMM.
    int8_t idx_c0[64], idx_c1[64], idx_c2[64], idx_c3[64];
    for (int i = 0; i < 64; ++i) {
        int q = i & ~3;
        idx_c0[i] = (int8_t)(q + 0);
        idx_c1[i] = (int8_t)(q + 1);
        idx_c2[i] = (int8_t)(q + 2);
        idx_c3[i] = (int8_t)(q + 3);
    }
    const __m512i v_i0 = _mm512_loadu_si512((const void*)idx_c0);
    const __m512i v_i1 = _mm512_loadu_si512((const void*)idx_c1);
    const __m512i v_i2 = _mm512_loadu_si512((const void*)idx_c2);
    const __m512i v_i3 = _mm512_loadu_si512((const void*)idx_c3);
    // Exact H_4 rows:
    //   r0 = +c0 + c1 + c2 + c3
    //   r1 = +c0 - c1 + c2 - c3
    //   r2 = +c0 + c1 - c2 - c3
    //   r3 = +c0 - c1 - c2 + c3
    const __mmask64 m_neg1 = 0xAAAAAAAAAAAAAAAAULL;  // lanes 1, 3
    const __mmask64 m_neg2 = 0xCCCCCCCCCCCCCCCCULL;  // lanes 2, 3
    const __mmask64 m_c3_neg = 0x6666666666666666ULL;  // lanes 1, 2

    uint64_t syn_errs = 0;
    for (int r = 0; r < M; ++r) {
        const uint64_t* row = packed + (size_t)r * n_blk16;
        __m512i acc32 = _mm512_setzero_si512();
        int b = 0;
        for (; b + 8 <= n_blk16; b += 8) {
            syn_errs += verify_8x_a16_avx512(row + b);
            // Process 2 chunks of 64 weights (32 packed bytes each) -> 128 weights total.
            const uint8_t* pbytes = (const uint8_t*)(row + b);
            for (int half = 0; half < 2; ++half) {
                __m256i raw32 = _mm256_loadu_si256((const __m256i*)(pbytes + half * 32));
                __m512i lo = _mm512_and_si512(_mm512_cvtepu8_epi16(raw32), _mm512_set1_epi16(0x000F));
                __m512i hi = _mm512_srli_epi16(_mm512_cvtepu8_epi16(raw32), 4);
                __m512i nib64 = _mm512_or_si512(lo, _mm512_slli_epi16(hi, 8));
                __m512i w = _mm512_permutexvar_epi8(_mm512_and_si512(nib64, mask_lo), lut);

                // In-register 4-point Sylvester Hadamard H_4 across all 16 quads simultaneously:
                __m512i c0 = _mm512_permutexvar_epi8(v_i0, w);
                __m512i c1 = _mm512_permutexvar_epi8(v_i1, w);
                __m512i c2 = _mm512_permutexvar_epi8(v_i2, w);
                __m512i c3 = _mm512_permutexvar_epi8(v_i3, w);

                __m512i t = _mm512_mask_sub_epi8(_mm512_add_epi8(c0, c1), m_neg1, c0, c1);
                t = _mm512_mask_sub_epi8(_mm512_add_epi8(t, c2), m_neg2, t, c2);
                __m512i wh4 = _mm512_mask_sub_epi8(_mm512_add_epi8(t, c3), m_c3_neg, t, c3);

                // Convert signed wh4 in [-28, 28] to unsigned u8 via +128 (XOR 0x80) for vpdpbusd.
                __m512i wh4_u8 = _mm512_xor_si512(wh4, bias128);
                __m512i xv = _mm512_loadu_si512((const void*)(x_i8 + b * 16 + half * 64));
                acc32 = _mm512_dpbusd_epi32(acc32, wh4_u8, xv);
            }
        }
        int32_t lanes[16];
        _mm512_storeu_si512((void*)lanes, acc32);
        int64_t total = 0;
        for (int i = 0; i < 16; ++i) total += (int64_t)lanes[i];
        int64_t tail_acc = 0;
        for (; b < n_blk16; ++b) {
            uint64_t w64 = row[b];
            if (a16_syndrome_scalar(w64) != 0u) ++syn_errs;
            for (int g = 0; g < 4; ++g) {
                int c0 = unpack_s4(w64, g * 4 + 0);
                int c1 = unpack_s4(w64, g * 4 + 1);
                int c2 = unpack_s4(w64, g * 4 + 2);
                int c3 = unpack_s4(w64, g * 4 + 3);
                int s01 = c0 + c1, d01 = c0 - c1;
                int s23 = c2 + c3, d23 = c2 - c3;
                int base = b * 16 + g * 4;
                tail_acc += (int64_t)(s01 + s23) * x_i8[base + 0]
                          + (int64_t)(d01 + d23) * x_i8[base + 1]
                          + (int64_t)(s01 - s23) * x_i8[base + 2]
                          + (int64_t)(d01 - d23) * x_i8[base + 3];
            }
        }
        y_out[r] = (total - 128LL * x_sum_bulk) + tail_acc;
    }
    return syn_errs;
}
#endif

static int run_selftest_shape(int M, int K, uint32_t seed) {
    if (M <= 0 || K <= 0 || (K % 16) != 0) return 2;
    const int n_blk16 = K / 16;

    uint64_t* packed = (uint64_t*)malloc((size_t)M * n_blk16 * sizeof(uint64_t));
    int8_t* x_i8 = (int8_t*)malloc((size_t)K * sizeof(int8_t));
    int64_t* y_scalar = (int64_t*)malloc((size_t)M * sizeof(int64_t));
    int64_t* y_avx = (int64_t*)malloc((size_t)M * sizeof(int64_t));
    if (!packed || !x_i8 || !y_scalar || !y_avx) {
        free(packed); free(x_i8); free(y_scalar); free(y_avx);
        return 2;
    }

    uint32_t rng = seed;
    for (int i = 0; i < K; ++i) {
        rng = rng * 1664525u + 1013904223u;
        x_i8[i] = (int8_t)(((int)(rng >> 24) % 63) - 31);
    }
    for (size_t i = 0; i < (size_t)M * n_blk16; ++i) {
        packed[i] = make_valid_a16_word(&rng);
    }

    uint64_t syn_scalar = decode_blockh4_gemv_scalar(packed, x_i8, y_scalar, M, K);
    int mismatches = 0;
    uint64_t syn_avx = 0;
#if SILIQUANT_HAS_AVX512_FULL
    syn_avx = decode_blockh4_gemv_avx512(packed, x_i8, y_avx, M, K);
    for (int r = 0; r < M; ++r) {
        if (y_avx[r] != y_scalar[r]) ++mismatches;
    }
    printf("avx512_a16_blockh4 self-test (AVX-512 VNNI): M=%d K=%d syn_scalar=%llu syn_avx=%llu mismatches=%d\n",
           M, K, (unsigned long long)syn_scalar, (unsigned long long)syn_avx, mismatches);
#else
    memcpy(y_avx, y_scalar, (size_t)M * sizeof(int64_t));
    printf("avx512_a16_blockh4 self-test (scalar fallback): M=%d K=%d syn_scalar=%llu mismatches=%d\n",
           M, K, (unsigned long long)syn_scalar, mismatches);
#endif

    free(packed);
    free(x_i8);
    free(y_scalar);
    free(y_avx);
    return (syn_scalar == 0 && syn_avx == 0 && mismatches == 0) ? 0 : 1;
}

int main(void) {
    // Test K % 128 == 0 (Qwen2.5), K % 128 != 0 (SmolLM2-360M K=960), and pure tail K=112 (< 128).
    if (run_selftest_shape(128, 1024, 20261124u) != 0) return 1;
    if (run_selftest_shape(128, 960, 20261125u) != 0) return 1;
    if (run_selftest_shape(64, 112, 20261126u) != 0) return 1;
    return 0;
}
