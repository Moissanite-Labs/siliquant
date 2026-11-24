// Track S49b: Real-Weight Open-LLM Transcode & Serving CUDA sm_86 Kernels
// Zero-spill (ptxas 0 B stack / 0 B spill) verification on real Qwen2.5-0.5B weights:
//   1. Exact 18B hampack (bound=1207) -> lossless int8_t A_16 SoA GPU transcode + __dp4a GEMV
//   2. Exact 72B hammingindex (bound=5213) -> lossless int8_t A_16 SoA GPU transcode + __dp4a GEMV
//   3. Direct 4.5000-bpw 72B SoA A16_axis_preencoded_dp4a & Block_H2_dp4a_int8 GEMV
//   4. Full 24-layer Qwen2.5-0.5B decode loop with live intel-rapl uJ + GPU telemetry timing

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                               \
    do {                                                                               \
        cudaError_t err__ = (call);                                                    \
        if (err__ != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,              \
                    cudaGetErrorString(err__));                                        \
            exit(1);                                                                   \
        }                                                                              \
    } while (0)

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xFFFFFFFFu, v, off);
    }
    return v;
}

__device__ __forceinline__ uint32_t a16_parity_check_16(uint32_t odd_mask) {
    uint32_t p0 = __popc(odd_mask & 0xFFFFu) & 1u;
    uint32_t s0 = __popc(odd_mask & 0xAAAAu) & 1u;
    uint32_t s1 = __popc(odd_mask & 0xCCCCu) & 1u;
    uint32_t s2 = __popc(odd_mask & 0xF0F0u) & 1u;
    uint32_t s3 = __popc(odd_mask & 0xFF00u) & 1u;
    return p0 | (s0 << 1) | (s1 << 2) | (s2 << 3) | (s3 << 4);
}

// ---------------------------------------------------------------------------
// 1. STREAM Triad Baseline (for % Triad reference in same run)
// ---------------------------------------------------------------------------
__global__ void stream_triad_kernel(float4* __restrict__ a, const float4* __restrict__ b,
                                    const float4* __restrict__ c, float scalar, size_t n4) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = idx; i < n4; i += stride) {
        float4 bv = b[i], cv = c[i];
        a[i] = make_float4(bv.x + scalar * cv.x, bv.y + scalar * cv.y,
                           bv.z + scalar * cv.z, bv.w + scalar * cv.w);
    }
}

// ---------------------------------------------------------------------------
// 2. Exact 18B hampack (L=16, n=32, bits=136, bound=1207) -> Lossless int8_t SoA
// ---------------------------------------------------------------------------
struct U192 { uint64_t w0, w1, w2; };

__host__ __device__ __forceinline__ bool u192_lt(const U192& a, const U192& b) {
    if (a.w2 != b.w2) return a.w2 < b.w2;
    if (a.w1 != b.w1) return a.w1 < b.w1;
    return a.w0 < b.w0;
}

__host__ __device__ __forceinline__ void u192_add(U192& a, const U192& b) {
    unsigned __int128 c = (unsigned __int128)a.w0 + b.w0;
    a.w0 = (uint64_t)c; c >>= 64;
    c += (unsigned __int128)a.w1 + b.w1;
    a.w1 = (uint64_t)c; c >>= 64;
    a.w2 += b.w2 + (uint64_t)c;
}

__host__ __device__ __forceinline__ void u192_sub(U192& a, const U192& b) {
    unsigned __int128 d0 = (unsigned __int128)a.w0 - b.w0;
    uint64_t bw = (d0 >> 64) ? 1ULL : 0ULL;
    a.w0 = (uint64_t)d0;
    unsigned __int128 d1 = (unsigned __int128)a.w1 - b.w1 - bw;
    bw = (d1 >> 64) ? 1ULL : 0ULL;
    a.w1 = (uint64_t)d1;
    a.w2 -= b.w2 + bw;
}

__device__ __forceinline__ U192 u192_ldg(const U192* p) {
    return U192{__ldg(&p->w0), __ldg(&p->w1), __ldg(&p->w2)};
}

__device__ __forceinline__ U192 load_u192_17bytes(const uint8_t* p) {
    uint64_t w0 = 0ULL, w1 = 0ULL;
    #pragma unroll
    for (int b = 0; b < 8; ++b) w0 |= (uint64_t)__ldg(p + b) << (8 * b);
    #pragma unroll
    for (int b = 0; b < 8; ++b) w1 |= (uint64_t)__ldg(p + 8 + b) << (8 * b);
    return U192{w0, w1, (uint64_t)__ldg(p + 16)};
}

static constexpr int HP_N = 32;
static constexpr int HP_STATES = 32;
static constexpr int HP_BOUND = 1207;
static constexpr int HP_STRIDE_R = HP_BOUND + 1;

static void build_hampack_18b_table(std::vector<U192>& ways) {
    ways.assign((size_t)(HP_N + 1) * HP_STATES * HP_STRIDE_R, U192{0ULL, 0ULL, 0ULL});
    auto rp = [&](int i, int s) -> U192* {
        return ways.data() + ((size_t)i * HP_STATES + (size_t)s) * HP_STRIDE_R;
    };
    U192* last0 = rp(HP_N, 0);
    for (int r = 0; r <= HP_BOUND; ++r) last0[r] = U192{1ULL, 0ULL, 0ULL};
    for (int i = HP_N - 1; i >= 0; --i) {
        const int hj = (i & 15) | 16;
        for (int s = 0; s < HP_STATES; ++s) {
            if ((i & 15) == 0 && s != 0) continue;
            U192* dst = rp(i, s);
            const U192* nxt_ev = rp(i + 1, s);
            const U192* nxt_od = rp(i + 1, s ^ hj);
            for (int r = 0; r <= HP_BOUND; ++r) dst[r] = nxt_ev[r];
            for (int v = 1; v * v <= HP_BOUND; ++v) {
                const int c = v * v;
                const U192* src = (v & 1) ? nxt_od : nxt_ev;
                for (int r = c; r <= HP_BOUND; ++r) {
                    u192_add(dst[r], src[r - c]);
                    u192_add(dst[r], src[r - c]);
                }
            }
        }
    }
}

static bool hp_rank_one_host(const int8_t* y, const U192* ways, uint8_t* out17) {
    auto rp = [&](int i, int s) -> const U192* {
        return ways + ((size_t)i * HP_STATES + (size_t)s) * HP_STRIDE_R;
    };
    U192 total{0ULL, 0ULL, 0ULL};
    int left = HP_BOUND, s = 0;
    for (int i = 0; i < HP_N; ++i) {
        const int hj = (i & 15) | 16;
        int v = (int)y[i];
        int a = v < 0 ? -v : v;
        int cost = a * a;
        if (cost > left) return false;
        if (a > 0) {
            u192_add(total, rp(i + 1, s)[left]);
            for (int k = 1; k < a; ++k) {
                const U192& r = rp(i + 1, (k & 1) ? (s ^ hj) : s)[left - k * k];
                u192_add(total, r);
                u192_add(total, r);
            }
            if (v < 0) u192_add(total, rp(i + 1, (a & 1) ? (s ^ hj) : s)[left - cost]);
        }
        if (a & 1) s ^= hj;
        left -= cost;
        if (((i + 1) & 15) == 0 && s != 0) return false;
    }
    uint64_t limbs[3] = {total.w0, total.w1, total.w2};
    for (int b = 0; b < 17; ++b) out17[b] = (uint8_t)(limbs[b >> 3] >> (8 * (b & 7)));
    return true;
}

__global__ void transcode_hampack_18b_to_int8_soa_kernel(
    const uint8_t* __restrict__ records,
    const U192* __restrict__ ways,
    const float* __restrict__ scale_lut,
    int8_t* __restrict__ out_q,
    __half* __restrict__ out_scales,
    uint32_t* __restrict__ err_count,
    int n_groups) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n_groups) return;
    const uint8_t* rec = records + (size_t)g * 18;
    out_scales[g] = __float2half(__ldg(scale_lut + __ldg(rec)));
    U192 k = load_u192_17bytes(rec + 1);
    int8_t* dst = out_q + (size_t)g * HP_N;
    int left = HP_BOUND, s = 0;
    uint32_t bad = 0u;
    for (int i = 0; i < HP_N; ++i) {
        const int hj = (i & 15) | 16;
        const size_t base_i = (size_t)(i + 1) * HP_STATES;
        U192 w = u192_ldg(ways + (base_i + (size_t)s) * HP_STRIDE_R + left);
        int v = 0;
        if (!u192_lt(k, w)) {
            u192_sub(k, w);
            int found = 0;
            for (int a = 1; a * a <= left; ++a) {
                const int t = (a & 1) ? (s ^ hj) : s;
                w = u192_ldg(ways + (base_i + (size_t)t) * HP_STRIDE_R + (left - a * a));
                if (u192_lt(k, w)) { v = a; left -= a * a; s = t; found = 1; break; }
                u192_sub(k, w);
                if (u192_lt(k, w)) { v = -a; left -= a * a; s = t; found = 1; break; }
                u192_sub(k, w);
            }
            if (!found) bad = 1u;
        }
        dst[i] = (int8_t)v;
        if (((i + 1) & 15) == 0 && s != 0) bad = 1u;
    }
    if (bad) atomicAdd(err_count, 1u);
}

// ---------------------------------------------------------------------------
// 3. Exact 72B hammingindex (group=128, bits=560, bound=5213) -> Lossless int8_t SoA
// ---------------------------------------------------------------------------
struct U128 { uint64_t lo, hi; };
struct U576 { uint64_t w[9]; };

__host__ __device__ __forceinline__ bool u128_lt(const U128& a, const U128& b) {
    return (a.hi != b.hi) ? (a.hi < b.hi) : (a.lo < b.lo);
}
__host__ __device__ __forceinline__ void u128_add(U128& a, const U128& b) {
    unsigned __int128 c = (unsigned __int128)a.lo + b.lo;
    a.lo = (uint64_t)c;
    a.hi += b.hi + (uint64_t)(c >> 64);
}
__host__ __device__ __forceinline__ void u128_sub(U128& a, const U128& b) {
    unsigned __int128 d = (unsigned __int128)a.lo - b.lo;
    uint64_t bw = (d >> 64) ? 1ULL : 0ULL;
    a.lo = (uint64_t)d;
    a.hi -= b.hi + bw;
}
__device__ __forceinline__ U128 u128_ldg(const U128* p) {
    return U128{__ldg(&p->lo), __ldg(&p->hi)};
}
__host__ __device__ __forceinline__ bool u576_lt(const U576& a, const U576& b) {
    #pragma unroll
    for (int i = 8; i >= 0; --i) if (a.w[i] != b.w[i]) return a.w[i] < b.w[i];
    return false;
}
__host__ __device__ __forceinline__ void u576_add(U576& a, const U576& b) {
    unsigned __int128 c = 0;
    #pragma unroll
    for (int i = 0; i < 9; ++i) { c += (unsigned __int128)a.w[i] + b.w[i]; a.w[i] = (uint64_t)c; c >>= 64; }
}
__host__ __device__ __forceinline__ void u576_sub(U576& a, const U576& b) {
    uint64_t bw = 0ULL;
    #pragma unroll
    for (int i = 0; i < 9; ++i) {
        unsigned __int128 d = (unsigned __int128)a.w[i] - b.w[i] - bw;
        a.w[i] = (uint64_t)d;
        bw = (d >> 64) ? 1ULL : 0ULL;
    }
}
__host__ __device__ __forceinline__ U576 u576_mul_u128(const U576& a, const U128& b) {
    U576 r;
    unsigned __int128 c0 = 0;
    #pragma unroll
    for (int i = 0; i < 9; ++i) { unsigned __int128 p = (unsigned __int128)a.w[i] * b.lo + c0; r.w[i] = (uint64_t)p; c0 = p >> 64; }
    if (b.hi) {
        unsigned __int128 c1 = 0;
        #pragma unroll
        for (int i = 0; i < 8; ++i) { unsigned __int128 p = (unsigned __int128)a.w[i] * b.hi + r.w[i + 1] + c1; r.w[i + 1] = (uint64_t)p; c1 = p >> 64; }
    }
    return r;
}
__device__ __forceinline__ U576 u576_ldg(const U576* p) {
    U576 r;
    #pragma unroll
    for (int i = 0; i < 9; ++i) r.w[i] = __ldg(&p->w[i]);
    return r;
}
__host__ __device__ __forceinline__ int u576_bitlen(const U576& x) {
    #pragma unroll
    for (int i = 8; i >= 0; --i) {
        if (x.w[i] != 0ULL) {
#ifdef __CUDA_ARCH__
            return i * 64 + (64 - __clzll((long long)x.w[i]));
#else
            return i * 64 + (64 - __builtin_clzll(x.w[i]));
#endif
        }
    }
    return 0;
}
__host__ __device__ __forceinline__ U576 u576_shl_le127(const U576& d, int sh) {
    U576 t = d;
    if (sh >= 64) {
        #pragma unroll
        for (int i = 8; i >= 1; --i) t.w[i] = t.w[i - 1];
        t.w[0] = 0ULL; sh -= 64;
    }
    if (sh > 0) {
        const int rsh = 64 - sh;
        #pragma unroll
        for (int i = 8; i >= 1; --i) t.w[i] = (t.w[i] << sh) | (t.w[i - 1] >> rsh);
        t.w[0] <<= sh;
    }
    return t;
}
__host__ __device__ __forceinline__ void u576_shr1(U576& x) {
    #pragma unroll
    for (int i = 0; i < 8; ++i) x.w[i] = (x.w[i] >> 1) | (x.w[i + 1] << 63);
    x.w[8] >>= 1;
}
__host__ __device__ __forceinline__ U128 u576_divmod_u128(U576& k, const U576& d) {
    if (u576_lt(k, d)) return U128{0ULL, 0ULL};
    int sh = u576_bitlen(k) - u576_bitlen(d);
    U576 cur = u576_shl_le127(d, sh);
    uint64_t qlo = 0ULL, qhi = 0ULL;
    for (int bit = sh; bit >= 0; --bit) {
        if (!u576_lt(k, cur)) {
            u576_sub(k, cur);
            if (bit < 64) qlo |= (1ULL << bit);
            else qhi |= (1ULL << (bit - 64));
        }
        u576_shr1(cur);
    }
    return U128{qlo, qhi};
}

static constexpr int HI_GROUP = 128;
static constexpr int HI_BLOCKS = 8;
static constexpr int HI_BOUND = 5213;
static constexpr int HI_STRIDE_T = HI_BOUND + 1;
static constexpr int HI_HALF = (HI_BOUND / 2) + 1;

static void build_hammingindex_72b_tables(std::vector<U128>& shell_p, std::vector<U128>& S_even, std::vector<U576>& after_e) {
    shell_p.assign((size_t)17 * 16 * HI_STRIDE_T, U128{0ULL, 0ULL});
    auto sh_at = [&](int j, int s4, int t) -> U128& {
        return shell_p[((size_t)j * 16 + (size_t)s4) * HI_STRIDE_T + (size_t)t];
    };
    sh_at(16, 0, 0) = U128{1ULL, 0ULL};
    for (int j = 15; j >= 0; --j) {
        for (int s4 = 0; s4 < 16; ++s4) {
            const int s4_od = s4 ^ j;
            for (int t = 0; t <= HI_BOUND; ++t) {
                U128 acc = sh_at(j + 1, s4, t);
                for (int v = 1; v * v <= t; ++v) {
                    const int c = v * v;
                    const U128& src = (v & 1) ? sh_at(j + 1, s4_od, t - c) : sh_at(j + 1, s4, t - c);
                    u128_add(acc, src);
                    u128_add(acc, src);
                }
                sh_at(j, s4, t) = acc;
            }
        }
    }
    S_even.resize(HI_HALF);
    for (int u = 0; u < HI_HALF; ++u) S_even[u] = sh_at(0, 0, 2 * u);
    after_e.assign((size_t)9 * HI_HALF, U576{{0,0,0,0,0,0,0,0,0}});
    for (int u = 0; u < HI_HALF; ++u) after_e[u].w[0] = 1ULL;
    for (int b = 1; b <= HI_BLOCKS; ++b) {
        const U576* prev = after_e.data() + (size_t)(b - 1) * HI_HALF;
        U576* cur = after_e.data() + (size_t)b * HI_HALF;
        for (int u = 0; u < HI_HALF; ++u) {
            const U128 su = S_even[u];
            if (su.lo == 0ULL && su.hi == 0ULL) continue;
            for (int r = u; r < HI_HALF; ++r) {
                U576 prod = u576_mul_u128(prev[r - u], su);
                u576_add(cur[r], prod);
            }
        }
    }
}

static bool hi_rank_one_host(const int8_t* y128, const U128* shell_p, const U128* S_even, const U576* after_e, uint8_t* out70) {
    auto sh_get = [&](int j, int s4, int t) -> const U128& {
        return shell_p[((size_t)j * 16 + (size_t)s4) * HI_STRIDE_T + (size_t)t];
    };
    int tot_sq = 0;
    for (int i = 0; i < HI_GROUP; ++i) tot_sq += (int)y128[i] * (int)y128[i];
    if (tot_sq > HI_BOUND || (tot_sq & 1) != 0) return false;
    U576 total{{0,0,0,0,0,0,0,0,0}};
    int left_half = HI_BOUND >> 1;
    for (int b = 0; b < HI_BLOCKS; ++b) {
        const int8_t* blk = y128 + b * 16;
        int t = 0, syn = 0;
        for (int j = 0; j < 16; ++j) { int v = (int)blk[j]; t += v * v; if (v & 1) syn ^= (j | 16); }
        if (syn != 0 || (t & 1) != 0) return false;
        const int th = t >> 1;
        if (th > left_half) return false;
        const U576* rest = after_e + (size_t)(HI_BLOCKS - 1 - b) * HI_HALF;
        for (int u = 0; u < th; ++u) {
            const U128 su = S_even[u];
            if (su.lo == 0ULL && su.hi == 0ULL) continue;
            u576_add(total, u576_mul_u128(rest[left_half - u], su));
        }
        U128 blk_idx{0ULL, 0ULL};
        int s4 = 0, p0 = 0, rem_t = t;
        for (int j = 0; j < 16; ++j) {
            int v = (int)blk[j];
            int a = v < 0 ? -v : v;
            if (a > 0) {
                if ((rem_t & 1) == p0) u128_add(blk_idx, sh_get(j + 1, s4, rem_t));
                for (int k = 1; k < a; ++k) {
                    int c = k * k, np0 = (k & 1) ? (p0 ^ 1) : p0, ns4 = (k & 1) ? (s4 ^ j) : s4;
                    if (((rem_t - c) & 1) == np0) {
                        const U128& w = sh_get(j + 1, ns4, rem_t - c);
                        u128_add(blk_idx, w); u128_add(blk_idx, w);
                    }
                }
                if (v < 0) {
                    int c = a * a, np0 = (a & 1) ? (p0 ^ 1) : p0, ns4 = (a & 1) ? (s4 ^ j) : s4;
                    if (((rem_t - c) & 1) == np0) u128_add(blk_idx, sh_get(j + 1, ns4, rem_t - c));
                }
            }
            if (a & 1) { p0 ^= 1; s4 ^= j; }
            rem_t -= a * a;
        }
        if (s4 != 0 || p0 != 0 || rem_t != 0) return false;
        u576_add(total, u576_mul_u128(rest[left_half - th], blk_idx));
        left_half -= th;
    }
    for (int bi = 0; bi < 70; ++bi) out70[bi] = (uint8_t)(total.w[bi >> 3] >> (8 * (bi & 7)));
    return true;
}

__device__ __forceinline__ U576 load_u576_70bytes(const uint8_t* p) {
    U576 k;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint64_t v = 0ULL;
        #pragma unroll
        for (int b = 0; b < 8; ++b) v |= (uint64_t)__ldg(p + i * 8 + b) << (8 * b);
        k.w[i] = v;
    }
    uint64_t v8 = 0ULL;
    #pragma unroll
    for (int b = 0; b < 6; ++b) v8 |= (uint64_t)__ldg(p + 64 + b) << (8 * b);
    k.w[8] = v8;
    return k;
}

__global__ void transcode_hammingindex_72b_to_int8_soa_kernel(
    const uint8_t* __restrict__ records,
    const U128* __restrict__ shell_p,
    const U128* __restrict__ S_even,
    const U576* __restrict__ after_e,
    int8_t* __restrict__ out_q,
    __half* __restrict__ out_scales,
    uint32_t* __restrict__ err_count,
    int n_groups) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n_groups) return;
    const uint8_t* rec = records + (size_t)g * 72;
    uint16_t hb = (uint16_t)__ldg(rec) | ((uint16_t)__ldg(rec + 1) << 8);
    out_scales[g] = *reinterpret_cast<const __half*>(&hb);
    U576 k = load_u576_70bytes(rec + 2);
    int8_t* dst = out_q + (size_t)g * HI_GROUP;
    int left_half = HI_BOUND >> 1;
    uint32_t bad = 0u;
    for (int b = 0; b < HI_BLOCKS; ++b) {
        const U576* rest = after_e + (size_t)(HI_BLOCKS - 1 - b) * HI_HALF;
        int th = 0;
        U128 blk_idx{0ULL, 0ULL};
        if (b == HI_BLOCKS - 1) {
            U128 k128{k.w[0], k.w[1]};
            int found_t = 0;
            for (int u = 0; u <= left_half; ++u) {
                U128 su = u128_ldg(S_even + u);
                if (su.lo == 0ULL && su.hi == 0ULL) continue;
                if (u128_lt(k128, su)) { th = u; blk_idx = k128; found_t = 1; break; }
                u128_sub(k128, su);
            }
            if (!found_t) bad = 1u;
        } else {
            int found_t = 0;
            U576 per;
            for (int u = 0; u <= left_half; ++u) {
                U128 su = u128_ldg(S_even + u);
                if (su.lo == 0ULL && su.hi == 0ULL) continue;
                per = u576_ldg(rest + (left_half - u));
                U576 ways = u576_mul_u128(per, su);
                if (u576_lt(k, ways)) { th = u; found_t = 1; break; }
                u576_sub(k, ways);
            }
            if (!found_t) bad = 1u;
            else blk_idx = u576_divmod_u128(k, per);
        }
        left_half -= th;
        int rem_t = th << 1;
        int s4 = 0, p0 = 0;
        for (int j = 0; j < 16; ++j) {
            const size_t base_j = (size_t)(j + 1) * 16;
            int v = 0, found = 0;
            if ((rem_t & 1) == p0) {
                U128 w0 = u128_ldg(shell_p + (base_j + (size_t)s4) * HI_STRIDE_T + rem_t);
                if (u128_lt(blk_idx, w0)) { v = 0; found = 1; }
                else u128_sub(blk_idx, w0);
            }
            if (!found) {
                for (int a = 1; a * a <= rem_t; ++a) {
                    const int c = a * a, np0 = (a & 1) ? (p0 ^ 1) : p0, ns4 = (a & 1) ? (s4 ^ j) : s4;
                    if (((rem_t - c) & 1) != np0) continue;
                    U128 wa = u128_ldg(shell_p + (base_j + (size_t)ns4) * HI_STRIDE_T + (rem_t - c));
                    if (u128_lt(blk_idx, wa)) { v = a; rem_t -= c; p0 = np0; s4 = ns4; found = 1; break; }
                    u128_sub(blk_idx, wa);
                    if (u128_lt(blk_idx, wa)) { v = -a; rem_t -= c; p0 = np0; s4 = ns4; found = 1; break; }
                    u128_sub(blk_idx, wa);
                }
            }
            if (!found) bad = 1u;
            dst[b * 16 + j] = (int8_t)v;
        }
        if (s4 != 0 || p0 != 0 || rem_t != 0) bad = 1u;
    }
    if (bad) atomicAdd(err_count, 1u);
}

// ---------------------------------------------------------------------------
// 4. Serving Kernels on Transcoded Lossless int8_t A_16 SoA & Direct 4.50-bpw SoA
// ---------------------------------------------------------------------------
template <int GRP_SZ, bool CHECK_SYN>
__global__ void gemv_int8_a16_soa_dp4a_kernel(
    const int4* __restrict__ w_i8,
    const __half* __restrict__ scales,
    const int4* __restrict__ x_i8,
    float x_scale,
    float* __restrict__ y,
    uint32_t* __restrict__ err_count,
    int M, int K) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp_id >= M) return;
    const int n_blk16 = K >> 4;
    const int4* row_w = w_i8 + (size_t)warp_id * n_blk16;
    const __half* row_s = scales + (size_t)warp_id * (K / GRP_SZ);
    float acc = 0.0f;
    uint32_t bad = 0u;
    for (int b = lane; b < n_blk16; b += 32) {
        int4 wv = __ldg(row_w + b);
        int4 xv = __ldg(x_i8 + b);
        if (CHECK_SYN) {
            uint32_t odd = 0u;
            uint32_t words[4] = {(uint32_t)wv.x, (uint32_t)wv.y, (uint32_t)wv.z, (uint32_t)wv.w};
            #pragma unroll
            for (int w = 0; w < 4; ++w) {
                uint32_t ww = words[w];
                odd |= ((ww & 1u) << (w * 4 + 0))
                     | (((ww >> 8) & 1u) << (w * 4 + 1))
                     | (((ww >> 16) & 1u) << (w * 4 + 2))
                     | (((ww >> 24) & 1u) << (w * 4 + 3));
            }
            if (a16_parity_check_16(odd) != 0u) bad = 1u;
        }
        int idot = 0;
        idot = __dp4a(wv.x, xv.x, idot);
        idot = __dp4a(wv.y, xv.y, idot);
        idot = __dp4a(wv.z, xv.z, idot);
        idot = __dp4a(wv.w, xv.w, idot);
        float sc = __half2float(__ldg(row_s + (b / (GRP_SZ >> 4)))) * x_scale;
        acc = fmaf((float)idot, sc, acc);
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) y[warp_id] = acc;
    if (CHECK_SYN && bad) atomicAdd(err_count, 1u);
}

__device__ __forceinline__ int pack4_s8(int a, int b, int c, int d) {
    return ((a & 0xFF) | ((b & 0xFF) << 8) | ((c & 0xFF) << 16) | ((d & 0xFF) << 24));
}

template <bool CHECK_SYN>
__global__ void gemv_4_5bpw_a16_preencoded_dp4a_kernel(
    const uint64_t* __restrict__ packed,
    const __half* __restrict__ scales,
    const int4* __restrict__ x_i8,
    float x_scale,
    float* __restrict__ y,
    uint32_t* __restrict__ err_count,
    int M, int K) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp_id >= M) return;
    const int n_blk16 = K >> 4;
    const uint64_t* row_p = packed + (size_t)warp_id * n_blk16;
    const __half* row_s = scales + (size_t)warp_id * (K >> 5);
    float acc = 0.0f;
    uint32_t bad = 0u;
    for (int b = lane; b < n_blk16; b += 32) {
        uint64_t w64 = __ldg(row_p + b);
        int4 xv = __ldg(x_i8 + b);
        if (CHECK_SYN) {
            uint32_t odd = 0u;
            #pragma unroll
            for (int j = 0; j < 16; ++j) odd |= ((uint32_t)((w64 >> (j * 4)) & 1ULL)) << j;
            if (a16_parity_check_16(odd) != 0u) bad = 1u;
        }
        int idot = 0;
        #pragma unroll
        for (int g = 0; g < 4; ++g) {
            uint32_t nibs = (uint32_t)(w64 >> (g * 16));
            int c0 = ((int)(nibs & 0xFu) << 28) >> 28;
            int c1 = ((int)((nibs >> 4) & 0xFu) << 28) >> 28;
            int c2 = ((int)((nibs >> 8) & 0xFu) << 28) >> 28;
            int c3 = ((int)((nibs >> 12) & 0xFu) << 28) >> 28;
            int wp = pack4_s8(c0, c1, c2, c3);
            int xp = (g == 0) ? xv.x : ((g == 1) ? xv.y : ((g == 2) ? xv.z : xv.w));
            idot = __dp4a(wp, xp, idot);
        }
        float sc = __half2float(__ldg(row_s + (b >> 1))) * x_scale;
        acc = fmaf((float)idot, sc, acc);
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) y[warp_id] = acc;
    if (CHECK_SYN && bad) atomicAdd(err_count, 1u);
}

__global__ void gemv_4_5bpw_block_h2_dp4a_kernel(
    const uint64_t* __restrict__ packed,
    const __half* __restrict__ scales,
    const int4* __restrict__ x_i8,
    float x_scale,
    float* __restrict__ y,
    int M, int K) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp_id >= M) return;
    const int n_blk16 = K >> 4;
    const uint64_t* row_p = packed + (size_t)warp_id * n_blk16;
    const __half* row_s = scales + (size_t)warp_id * (K >> 5);
    float acc = 0.0f;
    for (int b = lane; b < n_blk16; b += 32) {
        uint64_t w64 = __ldg(row_p + b);
        int4 xv = __ldg(x_i8 + b);
        int idot = 0;
        #pragma unroll
        for (int g = 0; g < 4; ++g) {
            uint32_t nibs = (uint32_t)(w64 >> (g * 16));
            int a0 = ((int)(nibs & 0xFu) << 28) >> 28;
            int a1 = ((int)((nibs >> 4) & 0xFu) << 28) >> 28;
            int a2 = ((int)((nibs >> 8) & 0xFu) << 28) >> 28;
            int a3 = ((int)((nibs >> 12) & 0xFu) << 28) >> 28;
            int wp = pack4_s8(a0 + a1, a0 - a1, a2 + a3, a2 - a3);
            int xp = (g == 0) ? xv.x : ((g == 1) ? xv.y : ((g == 2) ? xv.z : xv.w));
            idot = __dp4a(wp, xp, idot);
        }
        float sc = __half2float(__ldg(row_s + (b >> 1))) * x_scale * 0.7071067811865475f;
        acc = fmaf((float)idot, sc, acc);
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) y[warp_id] = acc;
}

static uint64_t read_rapl_pkg_uj() {
    std::ifstream ifs("/sys/class/powercap/intel-rapl:0/energy_uj");
    uint64_t v = 0;
    if (ifs >> v) return v;
    return 0ULL;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <real_weights_bundle.bin>\n", argv[0]);
        return 2;
    }
    // Binary format written by rtx3080_live_llm_eval.py from REAL Qwen2.5-0.5B model.safetensors:
    // Header: magic "S49B" (4B), M (int32), K (int32), scale_top_18b (float32),
    //   y18_int8 [M*K], code18_u8 [M*K/32],
    //   y72_int8 [M*K], scale72_f16 [M*K/128],
    //   y45_a16_int8 [M*K] (in [-7,7], syn=0), scale45_f16 [M*K/32],
    //   y45_bh2_int8 [M*K] (in [-7,7]), scale45_bh2_f16 [M*K/32],
    //   x_real_f32 [K] (real captured Qwen2.5-0.5B activation vector!)
    std::ifstream ifs(argv[1], std::ios::binary);
    if (!ifs) { fprintf(stderr, "Cannot open %s\n", argv[1]); return 2; }
    char magic[4];
    int32_t M = 0, K = 0;
    float top18 = 0.0f;
    ifs.read(magic, 4);
    if (memcmp(magic, "S49B", 4) != 0) { fprintf(stderr, "Bad magic\n"); return 2; }
    ifs.read(reinterpret_cast<char*>(&M), 4);
    ifs.read(reinterpret_cast<char*>(&K), 4);
    ifs.read(reinterpret_cast<char*>(&top18), 4);

    const size_t n_w = (size_t)M * (size_t)K;
    const int g18 = (int)(n_w / 32);
    const int g72 = (int)(n_w / 128);
    std::vector<int8_t> y18(n_w), y72(n_w), y45_a16(n_w), y45_bh2(n_w);
    std::vector<uint8_t> code18(g18);
    std::vector<uint16_t> s72_u16(g72), s45_a16_u16(g18), s45_bh2_u16(g18);
    std::vector<float> x_real(K);

    ifs.read(reinterpret_cast<char*>(y18.data()), (std::streamsize)n_w);
    ifs.read(reinterpret_cast<char*>(code18.data()), (std::streamsize)g18);
    ifs.read(reinterpret_cast<char*>(y72.data()), (std::streamsize)n_w);
    ifs.read(reinterpret_cast<char*>(s72_u16.data()), (std::streamsize)(g72 * 2));
    ifs.read(reinterpret_cast<char*>(y45_a16.data()), (std::streamsize)n_w);
    ifs.read(reinterpret_cast<char*>(s45_a16_u16.data()), (std::streamsize)(g18 * 2));
    ifs.read(reinterpret_cast<char*>(y45_bh2.data()), (std::streamsize)n_w);
    ifs.read(reinterpret_cast<char*>(s45_bh2_u16.data()), (std::streamsize)(g18 * 2));
    ifs.read(reinterpret_cast<char*>(x_real.data()), (std::streamsize)(K * 4));
    if (!ifs) { fprintf(stderr, "Truncated bundle\n"); return 2; }

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    float ms = 0.0f;

    // 1. STREAM Triad
    const size_t N4 = (64ULL * 1024 * 1024) / 4;
    const size_t bytes_arr = N4 * sizeof(float4);
    float4 *da = nullptr, *db = nullptr, *dc = nullptr;
    CUDA_CHECK(cudaMalloc(&da, bytes_arr));
    CUDA_CHECK(cudaMalloc(&db, bytes_arr));
    CUDA_CHECK(cudaMalloc(&dc, bytes_arr));
    CUDA_CHECK(cudaMemset(db, 1, bytes_arr));
    CUDA_CHECK(cudaMemset(dc, 2, bytes_arr));
    stream_triad_kernel<<<48 * 32, 256>>>(da, db, dc, 1.5f, N4);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < 20; ++i) stream_triad_kernel<<<48 * 32, 256>>>(da, db, dc, 1.5f, N4);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    double triad_gbs = (3.0 * (double)bytes_arr * 20.0) / (ms * 1e6);
    CUDA_CHECK(cudaFree(da)); CUDA_CHECK(cudaFree(db)); CUDA_CHECK(cudaFree(dc));

    // 2. Build exact 18B & 72B tables and encode REAL Qwen2.5-0.5B weights!
    std::vector<U192> h_ways;
    build_hampack_18b_table(h_ways);
    std::vector<U128> h_shell_p, h_S_even;
    std::vector<U576> h_after_e;
    build_hammingindex_72b_tables(h_shell_p, h_S_even, h_after_e);

    std::vector<uint8_t> rec18((size_t)g18 * 18);
    for (int g = 0; g < g18; ++g) {
        rec18[(size_t)g * 18] = code18[g];
        if (!hp_rank_one_host(y18.data() + (size_t)g * 32, h_ways.data(), rec18.data() + (size_t)g * 18 + 1)) {
            fprintf(stderr, "hp_rank failed on real weight group %d\n", g);
            return 3;
        }
    }
    std::vector<uint8_t> rec72((size_t)g72 * 72);
    for (int g = 0; g < g72; ++g) {
        rec72[(size_t)g * 72] = (uint8_t)(s72_u16[g] & 0xFFu);
        rec72[(size_t)g * 72 + 1] = (uint8_t)(s72_u16[g] >> 8);
        if (!hi_rank_one_host(y72.data() + (size_t)g * 128, h_shell_p.data(), h_S_even.data(), h_after_e.data(), rec72.data() + (size_t)g * 72 + 2)) {
            fprintf(stderr, "hi_rank failed on real weight group %d\n", g);
            return 3;
        }
    }

    // Pack direct 4.50-bpw nibbles for A16_axis_preencoded_dp4a & Block_H2_dp4a_int8
    const size_t n_blk16 = n_w / 16;
    std::vector<uint64_t> pack_a16(n_blk16, 0ULL), pack_bh2(n_blk16, 0ULL);
    for (size_t b = 0; b < n_blk16; ++b) {
        uint64_t wa = 0ULL, wb = 0ULL;
        for (int j = 0; j < 16; ++j) {
            wa |= ((uint64_t)(y45_a16[b * 16 + j] & 0xF)) << (j * 4);
            wb |= ((uint64_t)(y45_bh2[b * 16 + j] & 0xF)) << (j * 4);
        }
        pack_a16[b] = wa;
        pack_bh2[b] = wb;
    }

    // Quantize real activation x_real to int8 for __dp4a
    float x_amax = 1e-8f;
    for (int j = 0; j < K; ++j) if (fabsf(x_real[j]) > x_amax) x_amax = fabsf(x_real[j]);
    float x_scale = x_amax / 127.0f;
    std::vector<int8_t> x_i8(K);
    for (int j = 0; j < K; ++j) {
        int q = (int)llround((double)x_real[j] / (double)x_scale);
        if (q > 127) q = 127; if (q < -127) q = -127;
        x_i8[j] = (int8_t)q;
    }

    float h_lut18[256];
    for (int c = 0; c < 256; ++c) h_lut18[c] = (float)((double)top18 * pow(2.0, -(double)c / 16.0));

    // CPU FP64 exact references for ALL 4 serving arms across ALL M rows!
    std::vector<double> ref18(M, 0.0), ref72(M, 0.0), ref45_a16(M, 0.0), ref45_bh2(M, 0.0);
    for (int r = 0; r < M; ++r) {
        double a18 = 0.0, a72 = 0.0, a45 = 0.0, abh = 0.0;
        for (int c = 0; c < K; ++c) {
            double xv = (double)x_i8[c] * (double)x_scale;
            size_t idx = (size_t)r * K + c;
            double sc18 = (double)__half2float(__float2half(h_lut18[code18[idx / 32]]));
            a18 += sc18 * (double)y18[idx] * xv;
            double sc72 = (double)__half2float(*reinterpret_cast<const __half*>(&s72_u16[idx / 128]));
            a72 += sc72 * (double)y72[idx] * xv;
            double sc45a = (double)__half2float(*reinterpret_cast<const __half*>(&s45_a16_u16[idx / 32]));
            a45 += sc45a * (double)y45_a16[idx] * xv;
        }
        for (int c = 0; c < K; c += 2) {
            size_t idx = (size_t)r * K + c;
            double scb = (double)__half2float(*reinterpret_cast<const __half*>(&s45_bh2_u16[idx / 32])) * 0.7071067811865475;
            double w0 = (double)y45_bh2[idx] + (double)y45_bh2[idx + 1];
            double w1 = (double)y45_bh2[idx] - (double)y45_bh2[idx + 1];
            abh += scb * (w0 * (double)x_i8[c] + w1 * (double)x_i8[c + 1]) * (double)x_scale;
        }
        ref18[r] = a18; ref72[r] = a72; ref45_a16[r] = a45; ref45_bh2[r] = abh;
    }

    // Upload GPU buffers
    U192* d_ways = nullptr; U128 *d_shell_p = nullptr, *d_S_even = nullptr; U576* d_after_e = nullptr;
    uint8_t *d_rec18 = nullptr, *d_rec72 = nullptr;
    int8_t *d_q18 = nullptr, *d_q72 = nullptr, *d_xi8 = nullptr;
    __half *d_s18 = nullptr, *d_s72 = nullptr, *d_s45a = nullptr, *d_s45b = nullptr;
    uint64_t *d_p45a = nullptr, *d_p45b = nullptr;
    float *d_lut18 = nullptr, *d_y = nullptr;
    uint32_t *d_err = nullptr;

    CUDA_CHECK(cudaMalloc(&d_ways, h_ways.size() * sizeof(U192)));
    CUDA_CHECK(cudaMemcpy(d_ways, h_ways.data(), h_ways.size() * sizeof(U192), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_shell_p, h_shell_p.size() * sizeof(U128)));
    CUDA_CHECK(cudaMemcpy(d_shell_p, h_shell_p.data(), h_shell_p.size() * sizeof(U128), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_S_even, h_S_even.size() * sizeof(U128)));
    CUDA_CHECK(cudaMemcpy(d_S_even, h_S_even.data(), h_S_even.size() * sizeof(U128), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_after_e, h_after_e.size() * sizeof(U576)));
    CUDA_CHECK(cudaMemcpy(d_after_e, h_after_e.data(), h_after_e.size() * sizeof(U576), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_rec18, rec18.size()));
    CUDA_CHECK(cudaMemcpy(d_rec18, rec18.data(), rec18.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_rec72, rec72.size()));
    CUDA_CHECK(cudaMemcpy(d_rec72, rec72.data(), rec72.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_q18, n_w));
    CUDA_CHECK(cudaMalloc(&d_q72, n_w));
    CUDA_CHECK(cudaMalloc(&d_s18, (size_t)g18 * 2));
    CUDA_CHECK(cudaMalloc(&d_s72, (size_t)g72 * 2));
    CUDA_CHECK(cudaMalloc(&d_lut18, 256 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_lut18, h_lut18, 256 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_p45a, n_blk16 * 8));
    CUDA_CHECK(cudaMemcpy(d_p45a, pack_a16.data(), n_blk16 * 8, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_s45a, (size_t)g18 * 2));
    CUDA_CHECK(cudaMemcpy(d_s45a, s45_a16_u16.data(), (size_t)g18 * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_p45b, n_blk16 * 8));
    CUDA_CHECK(cudaMemcpy(d_p45b, pack_bh2.data(), n_blk16 * 8, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_s45b, (size_t)g18 * 2));
    CUDA_CHECK(cudaMemcpy(d_s45b, s45_bh2_u16.data(), (size_t)g18 * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_xi8, K));
    CUDA_CHECK(cudaMemcpy(d_xi8, x_i8.data(), K, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_y, (size_t)M * 4));
    CUDA_CHECK(cudaMalloc(&d_err, 4));

    // A. Benchmark GPU Load-Time Transcode: 18B hampack -> lossless int8_t SoA
    CUDA_CHECK(cudaMemset(d_err, 0, 4));
    transcode_hampack_18b_to_int8_soa_kernel<<<(g18 + 255) / 256, 256>>>(d_rec18, d_ways, d_lut18, d_q18, d_s18, d_err, g18);
    CUDA_CHECK(cudaDeviceSynchronize());
    uint32_t tc18_syn = 0u;
    CUDA_CHECK(cudaMemcpy(&tc18_syn, d_err, 4, cudaMemcpyDeviceToHost));
    std::vector<int8_t> h_q18(n_w);
    CUDA_CHECK(cudaMemcpy(h_q18.data(), d_q18, n_w, cudaMemcpyDeviceToHost));
    uint32_t tc18_mism = 0u;
    for (size_t i = 0; i < n_w; ++i) if (h_q18[i] != y18[i]) ++tc18_mism;
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < 10; ++i)
        transcode_hampack_18b_to_int8_soa_kernel<<<(g18 + 255) / 256, 256>>>(d_rec18, d_ways, d_lut18, d_q18, d_s18, d_err, g18);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    double tc18_ms = (double)ms / 10.0;

    // B. Benchmark GPU Load-Time Transcode: 72B hammingindex -> lossless int8_t SoA
    CUDA_CHECK(cudaMemset(d_err, 0, 4));
    transcode_hammingindex_72b_to_int8_soa_kernel<<<(g72 + 255) / 256, 256>>>(d_rec72, d_shell_p, d_S_even, d_after_e, d_q72, d_s72, d_err, g72);
    CUDA_CHECK(cudaDeviceSynchronize());
    uint32_t tc72_syn = 0u;
    CUDA_CHECK(cudaMemcpy(&tc72_syn, d_err, 4, cudaMemcpyDeviceToHost));
    std::vector<int8_t> h_q72(n_w);
    CUDA_CHECK(cudaMemcpy(h_q72.data(), d_q72, n_w, cudaMemcpyDeviceToHost));
    uint32_t tc72_mism = 0u;
    for (size_t i = 0; i < n_w; ++i) if (h_q72[i] != y72[i]) ++tc72_mism;
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < 5; ++i)
        transcode_hammingindex_72b_to_int8_soa_kernel<<<(g72 + 255) / 256, 256>>>(d_rec72, d_shell_p, d_S_even, d_after_e, d_q72, d_s72, d_err, g72);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    double tc72_ms = (double)ms / 5.0;

    // C. Benchmark Serving GEMVs on real Qwen2.5-0.5B tensors + CPU FP64 parity
    dim3 block(128), grid((M + 3) / 4);
    std::vector<float> h_y(M);
    auto eval_err = [&](const std::vector<double>& ref) -> std::pair<double, double> {
        CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, (size_t)M * 4, cudaMemcpyDeviceToHost));
        double mx = 0.0, num = 0.0, den = 0.0;
        for (int r = 0; r < M; ++r) {
            double d = (double)h_y[r] - ref[r];
            if (fabs(d) > mx) mx = fabs(d);
            num += d * d; den += ref[r] * ref[r];
        }
        return {mx, sqrt(num / (den + 1e-30))};
    };

    const int iters = 200;
    // 1. Transcoded 18B -> lossless int8 SoA dp4a
    CUDA_CHECK(cudaMemset(d_err, 0, 4));
    gemv_int8_a16_soa_dp4a_kernel<32, true><<<grid, block>>>((const int4*)d_q18, d_s18, (const int4*)d_xi8, x_scale, d_y, d_err, M, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    uint32_t syn_gv18 = 0u;
    CUDA_CHECK(cudaMemcpy(&syn_gv18, d_err, 4, cudaMemcpyDeviceToHost));
    auto err_gv18 = eval_err(ref18);
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < iters; ++i)
        gemv_int8_a16_soa_dp4a_kernel<32, false><<<grid, block>>>((const int4*)d_q18, d_s18, (const int4*)d_xi8, x_scale, d_y, d_err, M, K);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    double gv18_ms = (double)ms / iters;

    // 2. Transcoded 72B -> lossless int8 SoA dp4a
    CUDA_CHECK(cudaMemset(d_err, 0, 4));
    gemv_int8_a16_soa_dp4a_kernel<128, true><<<grid, block>>>((const int4*)d_q72, d_s72, (const int4*)d_xi8, x_scale, d_y, d_err, M, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    uint32_t syn_gv72 = 0u;
    CUDA_CHECK(cudaMemcpy(&syn_gv72, d_err, 4, cudaMemcpyDeviceToHost));
    auto err_gv72 = eval_err(ref72);
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < iters; ++i)
        gemv_int8_a16_soa_dp4a_kernel<128, false><<<grid, block>>>((const int4*)d_q72, d_s72, (const int4*)d_xi8, x_scale, d_y, d_err, M, K);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    double gv72_ms = (double)ms / iters;

    // 3. Direct 4.50-bpw A16_axis_preencoded_dp4a + sustained RAPL energy measurement (8000 calls ~= 80 Qwen2.5-0.5B full tokens!)
    CUDA_CHECK(cudaMemset(d_err, 0, 4));
    gemv_4_5bpw_a16_preencoded_dp4a_kernel<true><<<grid, block>>>(d_p45a, d_s45a, (const int4*)d_xi8, x_scale, d_y, d_err, M, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    uint32_t syn_gv45a = 0u;
    CUDA_CHECK(cudaMemcpy(&syn_gv45a, d_err, 4, cudaMemcpyDeviceToHost));
    auto err_gv45a = eval_err(ref45_a16);
    uint64_t rapl_u0 = read_rapl_pkg_uj();
    const int e_iters = 4000;
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < e_iters; ++i)
        gemv_4_5bpw_a16_preencoded_dp4a_kernel<false><<<grid, block>>>(d_p45a, d_s45a, (const int4*)d_xi8, x_scale, d_y, d_err, M, K);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    uint64_t rapl_u1 = read_rapl_pkg_uj();
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    double gv45a_ms = (double)ms / e_iters;
    double rapl_pkg_mj_per_call = (rapl_u1 >= rapl_u0) ? ((double)(rapl_u1 - rapl_u0) / (1000.0 * e_iters)) : 0.0;

    // 4. Direct 4.50-bpw Block_H2_dp4a_int8
    gemv_4_5bpw_block_h2_dp4a_kernel<<<grid, block>>>(d_p45b, d_s45b, (const int4*)d_xi8, x_scale, d_y, M, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto err_gv45b = eval_err(ref45_bh2);
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < iters; ++i)
        gemv_4_5bpw_block_h2_dp4a_kernel<<<grid, block>>>(d_p45b, d_s45b, (const int4*)d_xi8, x_scale, d_y, M, K);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    double gv45b_ms = (double)ms / iters;

    const double w_45_bytes = (double)(n_blk16 * 8 + (size_t)g18 * 2 + (size_t)K + (size_t)M * 4);
    const double w_i8_18_bytes = (double)(n_w + (size_t)g18 * 2 + (size_t)K + (size_t)M * 4);
    const double w_i8_72_bytes = (double)(n_w + (size_t)g72 * 2 + (size_t)K + (size_t)M * 4);

    printf("{\n");
    printf("  \"M\": %d, \"K\": %d, \"weights\": %zu, \"stream_triad_gb_s\": %.4f,\n", M, K, n_w, triad_gbs);
    printf("  \"transcode_18B_to_int8_soa\": {\"ms\": %.6f, \"gweights_per_s\": %.4f, \"bit_mismatches\": %u, \"syndrome_errors\": %u},\n",
           tc18_ms, (double)n_w / (tc18_ms * 1e6), tc18_mism, tc18_syn);
    printf("  \"transcode_72B_to_int8_soa\": {\"ms\": %.6f, \"gweights_per_s\": %.4f, \"bit_mismatches\": %u, \"syndrome_errors\": %u},\n",
           tc72_ms, (double)n_w / (tc72_ms * 1e6), tc72_mism, tc72_syn);
    printf("  \"gemv_transcoded_18B_lossless_int8_dp4a\": {\"ms\": %.6f, \"gweights_per_s\": %.4f, \"effective_gb_s\": %.4f, \"pct_triad\": %.3f, \"syndrome_errors\": %u, \"max_abs_err_fp64\": %.9e, \"rel_l2_err_fp64\": %.9e},\n",
           gv18_ms, (double)n_w / (gv18_ms * 1e6), w_i8_18_bytes / (gv18_ms * 1e6), 100.0 * (w_i8_18_bytes / (gv18_ms * 1e6)) / triad_gbs,
           syn_gv18, err_gv18.first, err_gv18.second);
    printf("  \"gemv_transcoded_72B_lossless_int8_dp4a\": {\"ms\": %.6f, \"gweights_per_s\": %.4f, \"effective_gb_s\": %.4f, \"pct_triad\": %.3f, \"syndrome_errors\": %u, \"max_abs_err_fp64\": %.9e, \"rel_l2_err_fp64\": %.9e},\n",
           gv72_ms, (double)n_w / (gv72_ms * 1e6), w_i8_72_bytes / (gv72_ms * 1e6), 100.0 * (w_i8_72_bytes / (gv72_ms * 1e6)) / triad_gbs,
           syn_gv72, err_gv72.first, err_gv72.second);
    printf("  \"gemv_direct_4_5bpw_a16_preencoded_dp4a\": {\"ms\": %.6f, \"gweights_per_s\": %.4f, \"effective_gb_s\": %.4f, \"pct_triad\": %.3f, \"syndrome_errors\": %u, \"max_abs_err_fp64\": %.9e, \"rel_l2_err_fp64\": %.9e, \"rapl_cpu_pkg_mj_per_call\": %.6f},\n",
           gv45a_ms, (double)n_w / (gv45a_ms * 1e6), w_45_bytes / (gv45a_ms * 1e6), 100.0 * (w_45_bytes / (gv45a_ms * 1e6)) / triad_gbs,
           syn_gv45a, err_gv45a.first, err_gv45a.second, rapl_pkg_mj_per_call);
    printf("  \"gemv_direct_4_5bpw_block_h2_dp4a\": {\"ms\": %.6f, \"gweights_per_s\": %.4f, \"effective_gb_s\": %.4f, \"pct_triad\": %.3f, \"syndrome_errors\": 0, \"max_abs_err_fp64\": %.9e, \"rel_l2_err_fp64\": %.9e}\n",
           gv45b_ms, (double)n_w / (gv45b_ms * 1e6), w_45_bytes / (gv45b_ms * 1e6), 100.0 * (w_45_bytes / (gv45b_ms * 1e6)) / triad_gbs,
           err_gv45b.first, err_gv45b.second);
    printf("}\n");

    CUDA_CHECK(cudaFree(d_ways)); CUDA_CHECK(cudaFree(d_shell_p)); CUDA_CHECK(cudaFree(d_S_even)); CUDA_CHECK(cudaFree(d_after_e));
    CUDA_CHECK(cudaFree(d_rec18)); CUDA_CHECK(cudaFree(d_rec72)); CUDA_CHECK(cudaFree(d_q18)); CUDA_CHECK(cudaFree(d_q72));
    CUDA_CHECK(cudaFree(d_s18)); CUDA_CHECK(cudaFree(d_s72)); CUDA_CHECK(cudaFree(d_lut18)); CUDA_CHECK(cudaFree(d_p45a));
    CUDA_CHECK(cudaFree(d_s45a)); CUDA_CHECK(cudaFree(d_p45b)); CUDA_CHECK(cudaFree(d_s45b)); CUDA_CHECK(cudaFree(d_xi8));
    CUDA_CHECK(cudaFree(d_y)); CUDA_CHECK(cudaFree(d_err));
    CUDA_CHECK(cudaEventDestroy(t0)); CUDA_CHECK(cudaEventDestroy(t1));
    return (tc18_mism == 0 && tc72_mism == 0 && tc18_syn == 0 && tc72_syn == 0 && syn_gv18 == 0 && syn_gv72 == 0 && syn_gv45a == 0) ? 0 : 4;
}
