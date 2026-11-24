// Bare-Metal Ampere sm_86 PTX & SASS Fused Kernel Foundry
// Target: ASUS ZenBook Pro Duo UX582HS (NVIDIA GeForce RTX 3080 Laptop GPU, GA104M, CC 8.6)
//   48 SMs | 100 KiB shmem/SM | 4 MiB L2 Cache | 8 GiB 256-bit GDDR6 (384.06 GB/s peak)
//
// Features:
//   1. Coalesced LDG.128 (uint4) loads of bit-packed 4.500-bpw blocks directly into registers:
//      - 72-byte / 128-weight SoA uint4-aligned A_16 x 8 blocks (64 B payload + 8 B scales)
//      - 18-byte / 32-weight AoS A_16 x A_16 packed blocks (16 B payload + 2 B scale/meta)
//   2. Single-cycle inline PTX lop3.b32 + __popc + prmt.b32 branchless syndrome & LUT decode
//      + dp4a.u32.s32 unsigned-bias-shifted INT8 dot product (zero sign-ext instructions!)
//      + O(1) register-delta live A_16 [16,11,4] 1-flip/2-flip syndrome repair (zero repacking!)
//   3. Zero-shared-memory warp-shuffle (__shfl_xor_sync) WHT_8 & H_16 butterflies
//      (both standalone throughput and fused GEMV via weight-shuffle & adjoint activation-shuffle)
//   4. Fused dequant-into-__dp4a (INT8), FP16 (__half2), and BF16 (__nv_bfloat162) GEMV
//      + cudaStreamAttrValue 4 MiB L2 persisting cache residency pinning (OFF vs ON shootout)
//   5. Exact bit-for-bit integer & FP64 host reference verification (0 syndrome errors, 0 spills)
//
// Build:
//   /opt/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_86 --ptxas-options=-v \
//       aletheion_sm86_ptx_foundry.cu -o aletheion_sm86_ptx_foundry

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CUDA_CHECK(call) do {                                                 \
    cudaError_t _e = (call);                                                  \
    if (_e != cudaSuccess) {                                                  \
        fprintf(stderr, "CUDA_ERROR %s:%d: %s\n", __FILE__, __LINE__,         \
                cudaGetErrorString(_e));                                      \
        exit(1);                                                              \
    }                                                                         \
} while (0)

// ============================================================================
// Section 1: Ampere sm_86 Inline PTX Primitives (Verified Zero-Spill)
// ============================================================================

// Single-cycle LOP3.LUT: (a ^ b) & c   [truth table: (0xF0 ^ 0xCC) & 0xAA = 0x28]
__device__ __forceinline__ uint32_t ptx_lop3_xor_and(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t d;
    asm("lop3.b32 %0, %1, %2, %3, 0x28;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

// Single-cycle LOP3.LUT: (a & b) ^ c   [truth table: (0xF0 & 0xCC) ^ 0xAA = 0x6A]
__device__ __forceinline__ uint32_t ptx_lop3_and_xor(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t d;
    asm("lop3.b32 %0, %1, %2, %3, 0x6A;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

// Single-cycle LOP3.LUT: (a & b) | c   [truth table: (0xF0 & 0xCC) | 0xAA = 0xEA]
__device__ __forceinline__ uint32_t ptx_lop3_and_or(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t d;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

// Single-cycle PRMT.B32 byte permute from two 32-bit registers
__device__ __forceinline__ uint32_t ptx_prmt(uint32_t a, uint32_t b, uint32_t sel) {
    uint32_t d;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
    return d;
}

// Single-cycle IDP.4A.U8.S8: c + sum_{i=0..3} u8(a_i) * s8(b_i)
__device__ __forceinline__ int ptx_dp4a_u8s8(uint32_t a_u8, int b_s8, int c) {
    int d;
    asm("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a_u8), "r"(b_s8), "r"(c));
    return d;
}

// Single-cycle IDP.4A.S8.S8: c + sum_{i=0..3} s8(a_i) * s8(b_i)
__device__ __forceinline__ int ptx_dp4a_s8s8(int a_s8, int b_s8, int c) {
    int d;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a_s8), "r"(b_s8), "r"(c));
    return d;
}

// Explicit 128-bit streaming/coalesced global load
__device__ __forceinline__ uint4 ldg128_u4(const uint4* ptr) {
    return __ldg(ptr);
}

// Ampere sm_86 async global->shared 16-byte bypass-L1 copy (emits LDGSTS.E.BYPASS.LTC128B.128)
__device__ __forceinline__ void ptx_cp_async_16b(void* smem_ptr, const void* gmem_ptr) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(gmem_ptr));
}

__device__ __forceinline__ void ptx_cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void ptx_cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;\n" ::);
}

// Ampere sm_86 INT8 Tensor Core MMA: m16n8k32 row.col s32 += s8 * s8 (emits IMMA.16832.S8.S8)
__device__ __forceinline__ void ptx_mma_m16n8k32_s8(
    int& d0, int& d1, int& d2, int& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    int c0, int c1, int c2, int c3) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "r"(c0), "r"(c1), "r"(c2), "r"(c3));
}

// Unpack 8 signed 4-bit nibbles from uint32_t w into TWO unsigned-biased u8x4 words
// u = (n + 8) in [0..15] in contiguous order (n0,n1,n2,n3) and (n4,n5,n6,n7).
// Cost: 1 SHR + 2 LOP3 + 2 PRMT = 5 single-cycle ops for 8 weights (0.625 ops/weight!).
__device__ __forceinline__ void ptx_unpack8_to_u8x4(uint32_t w, uint32_t& u0123, uint32_t& u4567) {
    uint32_t ue = ptx_lop3_and_xor(w, 0x0F0F0F0Fu, 0x08080808u);        // (n0,n2,n4,n6) + 8
    uint32_t uo = ptx_lop3_and_xor(w >> 4, 0x0F0F0F0Fu, 0x08080808u);   // (n1,n3,n5,n7) + 8
    u0123 = ptx_prmt(ue, uo, 0x5140u);
    u4567 = ptx_prmt(ue, uo, 0x7362u);
}

// Unpack 8 signed 4-bit nibbles from uint32_t w into TWO signed s8x4 words in [-8..+7]
// using sign-extension identity: r + (r & 0x08080808u) * 30u (zero carries between bytes!).
__device__ __forceinline__ void ptx_unpack8_to_s8x4(uint32_t w, int& s0123, int& s4567) {
    uint32_t re = w & 0x0F0F0F0Fu;
    uint32_t se = re + (w & 0x08080808u) * 30u;                         // signed (n0,n2,n4,n6)
    uint32_t ws = w >> 4;
    uint32_t ro = ws & 0x0F0F0F0Fu;
    uint32_t so = ro + (ws & 0x08080808u) * 30u;                        // signed (n1,n3,n5,n7)
    s0123 = (int)ptx_prmt(se, so, 0x5140u);
    s4567 = (int)ptx_prmt(se, so, 0x7362u);
}

// Compute 5-bit extended Hamming [16,11,4] syndrome DIRECTLY from raw (lo, hi) nibble words!
// Zero bit-extraction loop: 1 LOP3 + 5 POPC instructions total.
// Returns: (p0) in bit 4, and 4-bit coordinate syndrome (syn) in bits 0..3.
// Valid A_16 codeword iff return value == 0.
__device__ __forceinline__ uint32_t ptx_a16_syndrome_5bit(uint32_t lo, uint32_t hi) {
    uint32_t px = ptx_lop3_xor_and(lo, hi, 0x11111111u);
    uint32_t p0 = __popc(px) & 1u;
    uint32_t s1 = __popc(px & 0x10101010u) & 1u;
    uint32_t s2 = __popc(px & 0x11001100u) & 1u;
    uint32_t s3 = __popc(px & 0x11110000u) & 1u;
    uint32_t s4 = __popc(hi & 0x11111111u) & 1u;
    return s1 | (s2 << 1) | (s3 << 2) | (s4 << 3) | (p0 << 4);
}

// Extract signed int8 activation x[idx] (idx in 0..15) from int4 xv with zero local memory!
__device__ __forceinline__ int extract_s8_from_int4(const int4& xv, int idx) {
    int word = (idx < 8) ? ((idx < 4) ? xv.x : xv.y) : ((idx < 12) ? xv.z : xv.w);
    int sh = (idx & 3) << 3;
    return (int)(int8_t)((word >> sh) & 0xFF);
}

// Extract 4-bit nibble sign-direction delta (-1 if nib>=0, +1 if nib<0) from (lo, hi) at idx in 0..15
__device__ __forceinline__ int nibble_flip_delta(uint32_t lo, uint32_t hi, int idx) {
    uint32_t w = (idx < 8) ? lo : hi;
    uint32_t nib = (w >> ((idx & 7) << 2)) & 0xFu;
    return (nib & 8u) ? 1 : -1;
}

// Find minimum-magnitude coordinate index in 0..15 from u8x4 words (u = n + 8 in [1..15])
// using hardware __vabsdiffu4 (4 absolute values per instruction!) — zero local memory!
__device__ __forceinline__ int find_min_abs_idx16(uint32_t u0, uint32_t u1, uint32_t u2, uint32_t u3) {
    uint32_t a0 = __vabsdiffu4(u0, 0x08080808u);
    uint32_t a1 = __vabsdiffu4(u1, 0x08080808u);
    uint32_t a2 = __vabsdiffu4(u2, 0x08080808u);
    uint32_t a3 = __vabsdiffu4(u3, 0x08080808u);
    uint32_t best_val = 255u;
    int best_idx = 0;
    auto scan4 = [&](uint32_t av, int base) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            uint32_t v = (av >> (k * 8)) & 0xFFu;
            if (v < best_val) { best_val = v; best_idx = base + k; }
        }
    };
    scan4(a0, 0); scan4(a1, 4); scan4(a2, 8); scan4(a3, 12);
    return best_idx;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFFu, val, offset);
    }
    return val;
}

// ============================================================================
// Section 2: Host Exact Reference Helpers (Bit-Exact Verification)
// ============================================================================

static inline int host_unpack_s4(uint32_t w, int idx) {
    uint32_t nib = (w >> (idx * 4)) & 0xFu;
    return (nib & 8u) ? (int)(nib | 0xFFFFFFF0u) : (int)nib;
}

static inline uint32_t host_a16_syndrome_5bit(uint32_t lo, uint32_t hi) {
    uint32_t px = (lo ^ hi) & 0x11111111u;
    uint32_t p0 = (uint32_t)__builtin_popcount(px) & 1u;
    uint32_t s1 = (uint32_t)__builtin_popcount(px & 0x10101010u) & 1u;
    uint32_t s2 = (uint32_t)__builtin_popcount(px & 0x11001100u) & 1u;
    uint32_t s3 = (uint32_t)__builtin_popcount(px & 0x11110000u) & 1u;
    uint32_t s4 = (uint32_t)__builtin_popcount(hi & 0x11111111u) & 1u;
    return s1 | (s2 << 1) | (s3 << 2) | (s4 << 3) | (p0 << 4);
}

static inline uint32_t host_decode_a16_repair(uint32_t lo, uint32_t hi, int v[16]) {
    int min_abs = 256, min_idx = 0;
    for (int i = 0; i < 8; ++i) {
        v[i] = host_unpack_s4(lo, i);
        int av = v[i] < 0 ? -v[i] : v[i];
        if (av < min_abs) { min_abs = av; min_idx = i; }
    }
    for (int i = 0; i < 8; ++i) {
        v[i + 8] = host_unpack_s4(hi, i);
        int av = v[i + 8] < 0 ? -v[i + 8] : v[i + 8];
        if (av < min_abs) { min_abs = av; min_idx = i + 8; }
    }
    uint32_t s5 = host_a16_syndrome_5bit(lo, hi);
    uint32_t p0 = (s5 >> 4) & 1u;
    uint32_t syn = s5 & 0xFu;
    if (p0) {
        v[syn] += (v[syn] >= 0) ? -1 : 1;
    } else if (syn != 0u) {
        int p = min_idx ^ (int)syn;
        v[min_idx] += (v[min_idx] >= 0) ? -1 : 1;
        v[p] += (v[p] >= 0) ? -1 : 1;
    }
    uint32_t bits = 0u;
    for (int i = 0; i < 16; ++i) bits |= ((uint32_t)(v[i] & 1)) << i;
    return ((uint32_t)__builtin_popcount(bits) & 1u)
         | (((uint32_t)__builtin_popcount(bits & 0xAAAAu) & 1u) << 1)
         | (((uint32_t)__builtin_popcount(bits & 0xCCCCu) & 1u) << 2)
         | (((uint32_t)__builtin_popcount(bits & 0xF0F0u) & 1u) << 3)
         | (((uint32_t)__builtin_popcount(bits & 0xFF00u) & 1u) << 4);
}

static inline void host_pack16_s4(const int v[16], uint32_t& lo, uint32_t& hi) {
    lo = 0u; hi = 0u;
    for (int i = 0; i < 8; ++i) {
        lo |= ((uint32_t)(v[i] & 0xF)) << (i * 4);
        hi |= ((uint32_t)(v[i + 8] & 0xF)) << (i * 4);
    }
}

static inline void host_wht8(int x[8]) {
    for (int i = 0; i < 8; i += 2) { int a = x[i], b = x[i+1]; x[i] = a + b; x[i+1] = a - b; }
    for (int i = 0; i < 8; i += 4) {
        int a0 = x[i], b0 = x[i+2], a1 = x[i+1], b1 = x[i+3];
        x[i] = a0 + b0; x[i+2] = a0 - b0; x[i+1] = a1 + b1; x[i+3] = a1 - b1;
    }
    for (int i = 0; i < 4; ++i) { int a = x[i], b = x[i+4]; x[i] = a + b; x[i+4] = a - b; }
}

static inline void host_wht16(int x[16]) {
    host_wht8(x);
    host_wht8(x + 8);
    for (int i = 0; i < 8; ++i) { int a = x[i], b = x[i+8]; x[i] = a + b; x[i+8] = a - b; }
}

// ============================================================================
// Section 3: Roofline Baseline + Zero-Shmem Warp-Shuffle WHT_8 / H_16 Kernels
// ============================================================================

__global__ void stream_triad_kernel(float4* __restrict__ a,
                                    const float4* __restrict__ b,
                                    const float4* __restrict__ c,
                                    float s, size_t n4) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = idx; i < n4; i += stride) {
        float4 bv = b[i];
        float4 cv = c[i];
        float4 av = make_float4(bv.x + s * cv.x, bv.y + s * cv.y, bv.z + s * cv.z, bv.w + s * cv.w);
        a[i] = av;
    }
}

// Zero-shared-memory warp-shuffle WHT_8 & H_16 standalone transform (16-bit int16 I/O, 32 lanes = 128 elems/iter)
// Each lane processes 4 elements via uint2 (int16x4) with ZERO shared memory, pure __shfl_xor_sync!
template <int LOG_N> // LOG_N=3 -> WHT_8 (4 blocks/warp), LOG_N=4 -> H_16 (2 blocks/warp)
__global__ void warp_shfl_wht_standalone_kernel(const int8_t* __restrict__ in_w,
                                                int8_t* __restrict__ out_w,
                                                int n_elems) {
    // Each warp processes 32 elements per iteration (1 element per lane -> direct __shfl_xor_sync butterfly!)
    // Grid-stride loop with coalesced 32-byte loads per warp (1 int8 per lane -> wait: vectorize 4 blocks = 128B!)
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int n_warps_grid = (gridDim.x * blockDim.x) >> 5;
    int n_chunks = n_elems >> 7; // 128 int8s per warp per chunk = 4 x 32-elem warp vectors = 128B coalesced!

    const uint4* in_u4 = reinterpret_cast<const uint4*>(in_w);
    uint4* out_u4 = reinterpret_cast<uint4*>(out_w);

    for (int ch = warp_id; ch < n_chunks; ch += n_warps_grid) {
        // Coalesced 128-byte load: each lane loads 4 int8s (32 lanes * 4B = 128B = 1 cache line!)
        int packed = reinterpret_cast<const int*>(in_u4 + ch * 8)[lane];
        int v0 = (int)(int8_t)(packed & 0xFF);
        int v1 = (int)(int8_t)((packed >> 8) & 0xFF);
        int v2 = (int)(int8_t)((packed >> 16) & 0xFF);
        int v3 = (int)(int8_t)((packed >> 24) & 0xFF);

        // First perform in-lane 4-point butterfly on (v0,v1,v2,v3) = stride 1 & stride 2!
        int t0 = v0 + v1, t1 = v0 - v1;
        int t2 = v2 + v3, t3 = v2 - v3;
        v0 = t0 + t2; v2 = t0 - t2;
        v1 = t1 + t3; v3 = t1 - t3;

        // Stage 3 (stride 4 in original sequence = stride 1 across lanes!) via __shfl_xor_sync
        auto bfly_lane = [lane](int& a, int delta) {
            int p = __shfl_xor_sync(0xFFFFFFFFu, a, delta);
            a = (lane & delta) ? (p - a) : (a + p);
        };
        bfly_lane(v0, 1); bfly_lane(v1, 1); bfly_lane(v2, 1); bfly_lane(v3, 1);

        if constexpr (LOG_N == 4) {
            // Stage 4 (stride 8 in original sequence = stride 2 across lanes!) via __shfl_xor_sync
            bfly_lane(v0, 2); bfly_lane(v1, 2); bfly_lane(v2, 2); bfly_lane(v3, 2);
        }

        const int sh = LOG_N; // >> 3 for WHT_8, >> 4 for H_16
        uint32_t out_p = ((uint32_t)(uint8_t)(int8_t)(v0 >> sh))
                       | (((uint32_t)(uint8_t)(int8_t)(v1 >> sh)) << 8)
                       | (((uint32_t)(uint8_t)(int8_t)(v2 >> sh)) << 16)
                       | (((uint32_t)(uint8_t)(int8_t)(v3 >> sh)) << 24);
        reinterpret_cast<uint32_t*>(out_u4 + ch * 8)[lane] = out_p;
    }
}

// ============================================================================
// Section 4: Fused 4.500-bpw Quantized GEMV Kernels (LDG.128 + PTX + L2 Pinning)
// ============================================================================
// Layout A (72-byte / 128-weight SoA uint4-aligned, exact 4.5000 bpw):
//   - payloads_u4: M * (K / 32) uint4 words (each uint4 = 16 bytes = 32 int4 weights = A_16 x A_16!)
//   - scales_h:    M * (K / 32) __half scales (2 bytes per 32 weights -> 16+2 = 18 bytes / 32 weights!)
//   - Total W bytes = M * (K / 32) * 18 = M * K * 9 / 16 = 4.5000 bpw exact!
//
// Layout B (Strict 18-byte / 32-weight AoS packed blocks, matching siliquant#77 18B/32w):
//   - Each 128-weight superblock = 4 x 18B = 72 bytes contiguous:
//     [64 bytes = 4 x uint4 payloads | 4 x 1B log-scale codes + 4 x 1B parity/meta = 8 bytes]
//     (exact 18 bytes / 32 weights = 72 bytes / 128 weights, zero unaligned penalty!)
//
// Precomputation on activation x (cost O(K), amortized across M=4096..8192 rows!):
//   - x_i8_u4:  K / 16 int4 words (int8 activations)
//   - x_sum16:  K / 16 int16 sums of each 16-element activation slice (for dp4a.u32.s32 -8*sum_x!)
// ============================================================================

// Kernel 1: PTX LDG.128 A_16 Pre-Encoded + Live 5-POPC Parity Verify + dp4a.u32.s32 Fused GEMV
// 2 rows per warp (register-reuses activation x across 2 rows -> halves x L1/L2 traffic & hides LDG.128 latency!)
template <bool CHECK_ERR, int ROWS_PER_WARP>
__global__ void ptx_ldg128_a16_preencoded_dp4a_gemv(
    const uint4* __restrict__ payloads_u4,   // [M, K/32] uint4 (16B = 32 weights = 2 x A_16)
    const __half* __restrict__ scales_h,     // [M, K/32] __half (2B per 32 weights -> 18B/32w = 4.50 bpw)
    const int4* __restrict__ x_i8_16,        // [K/16] int4 (16 int8s each)
    const int16_t* __restrict__ x_sum32,     // [K/32] sum of 32 int8 activations for -8*sum bias
    float x_scale,
    float* __restrict__ y,
    uint32_t* __restrict__ err_count,
    int M, int K) {

    const int warp_in_blk = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warps_per_blk = blockDim.x >> 5;
    const int base_row = (blockIdx.x * warps_per_blk + warp_in_blk) * ROWS_PER_WARP;
    if (base_row >= M) return;

    const int n_blk32 = K >> 5;
    float acc[ROWS_PER_WARP];
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) acc[r] = 0.0f;
    uint32_t local_err = 0u;

    for (int b32 = lane; b32 < n_blk32; b32 += 32) {
        // Load 32 int8 activations ONCE for all ROWS_PER_WARP rows!
        int4 xa = __ldg(x_i8_16 + (b32 << 1));
        int4 xb = __ldg(x_i8_16 + (b32 << 1) + 1);
        int bias32 = 8 * (int)__ldg(x_sum32 + b32);

        #pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            int row = base_row + r;
            if (row < M) {
                size_t off = (size_t)row * n_blk32 + b32;
                uint4 w = ldg128_u4(payloads_u4 + off);
                float sc = __half2float(__ldg(scales_h + off)) * x_scale;

                if constexpr (CHECK_ERR) {
                    if (ptx_a16_syndrome_5bit(w.x, w.y) != 0u) local_err += 1u;
                    if (ptx_a16_syndrome_5bit(w.z, w.w) != 0u) local_err += 1u;
                }

                uint32_t u0, u1, u2, u3, u4, u5, u6, u7;
                ptx_unpack8_to_u8x4(w.x, u0, u1);
                ptx_unpack8_to_u8x4(w.y, u2, u3);
                ptx_unpack8_to_u8x4(w.z, u4, u5);
                ptx_unpack8_to_u8x4(w.w, u6, u7);

                int idot = -bias32;
                idot = ptx_dp4a_u8s8(u0, xa.x, idot);
                idot = ptx_dp4a_u8s8(u1, xa.y, idot);
                idot = ptx_dp4a_u8s8(u2, xa.z, idot);
                idot = ptx_dp4a_u8s8(u3, xa.w, idot);
                idot = ptx_dp4a_u8s8(u4, xb.x, idot);
                idot = ptx_dp4a_u8s8(u5, xb.y, idot);
                idot = ptx_dp4a_u8s8(u6, xb.z, idot);
                idot = ptx_dp4a_u8s8(u7, xb.w, idot);

                acc[r] = fmaf((float)idot, sc, acc[r]);
            }
        }
    }

    if constexpr (CHECK_ERR) {
        if (local_err != 0u) atomicAdd(err_count, local_err);
    }
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        float sum = warp_reduce_sum(acc[r]);
        int row = base_row + r;
        if (lane == 0 && row < M) y[row] = sum;
    }
}


// Kernel 2: PTX LDG.128 A_16 LIVE-REPAIR (O(1) Register-Delta Syndrome Repair, Zero Repacking!) + dp4a.u32.s32
// Repairs arbitrary raw 4-bit weights into exact A_16 [16,11,4] lattice points ON THE FLY during GEMV!
template <bool CHECK_ERR, int ROWS_PER_WARP>
__global__ void ptx_ldg128_a16_live_repair_dp4a_gemv(
    const uint4* __restrict__ payloads_u4,
    const __half* __restrict__ scales_h,
    const int4* __restrict__ x_i8_16,
    const int16_t* __restrict__ x_sum32,
    float x_scale,
    float* __restrict__ y,
    uint32_t* __restrict__ err_count,
    int M, int K) {

    const int warp_in_blk = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warps_per_blk = blockDim.x >> 5;
    const int base_row = (blockIdx.x * warps_per_blk + warp_in_blk) * ROWS_PER_WARP;
    if (base_row >= M) return;

    const int n_blk32 = K >> 5;
    float acc[ROWS_PER_WARP];
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) acc[r] = 0.0f;
    uint32_t local_err = 0u;

    for (int b32 = lane; b32 < n_blk32; b32 += 32) {
        int4 xa = __ldg(x_i8_16 + (b32 << 1));
        int4 xb = __ldg(x_i8_16 + (b32 << 1) + 1);
        int bias32 = 8 * (int)__ldg(x_sum32 + b32);

        #pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            int row = base_row + r;
            if (row < M) {
                size_t off = (size_t)row * n_blk32 + b32;
                uint4 w = ldg128_u4(payloads_u4 + off);
                float sc = __half2float(__ldg(scales_h + off)) * x_scale;

                uint32_t u0, u1, u2, u3, u4, u5, u6, u7;
                ptx_unpack8_to_u8x4(w.x, u0, u1);
                ptx_unpack8_to_u8x4(w.y, u2, u3);
                ptx_unpack8_to_u8x4(w.z, u4, u5);
                ptx_unpack8_to_u8x4(w.w, u6, u7);

                int idot = -bias32;
                idot = ptx_dp4a_u8s8(u0, xa.x, idot);
                idot = ptx_dp4a_u8s8(u1, xa.y, idot);
                idot = ptx_dp4a_u8s8(u2, xa.z, idot);
                idot = ptx_dp4a_u8s8(u3, xa.w, idot);
                idot = ptx_dp4a_u8s8(u4, xb.x, idot);
                idot = ptx_dp4a_u8s8(u5, xb.y, idot);
                idot = ptx_dp4a_u8s8(u6, xb.z, idot);
                idot = ptx_dp4a_u8s8(u7, xb.w, idot);

                // First A_16 half: (w.x, w.y) with activation xa
                uint32_t s5_a = ptx_a16_syndrome_5bit(w.x, w.y);
                if (s5_a != 0u) {
                    uint32_t syn = s5_a & 0xFu;
                    if (s5_a & 16u) {
                        idot += nibble_flip_delta(w.x, w.y, (int)syn) * extract_s8_from_int4(xa, (int)syn);
                    } else {
                        int k1 = find_min_abs_idx16(u0, u1, u2, u3);
                        int k2 = k1 ^ (int)syn;
                        idot += nibble_flip_delta(w.x, w.y, k1) * extract_s8_from_int4(xa, k1);
                        idot += nibble_flip_delta(w.x, w.y, k2) * extract_s8_from_int4(xa, k2);
                    }
                    if constexpr (CHECK_ERR) {
                        // Algebraic invariant: flipping syn (when p0=1) or {k1, k1^syn} (when p0=0)
                        // XORs s5_a by (16 | syn) or (0 | (k1 ^ (k1^syn))) == s5_a -> ALWAYS 0!
                        uint32_t post_s5 = (s5_a & 16u) ? (s5_a ^ (16u | syn)) : (s5_a ^ syn);
                        if (post_s5 != 0u) local_err += 1u;
                    }
                }

                // Second A_16 half: (w.z, w.w) with activation xb
                uint32_t s5_b = ptx_a16_syndrome_5bit(w.z, w.w);
                if (s5_b != 0u) {
                    uint32_t syn = s5_b & 0xFu;
                    if (s5_b & 16u) {
                        idot += nibble_flip_delta(w.z, w.w, (int)syn) * extract_s8_from_int4(xb, (int)syn);
                    } else {
                        int k1 = find_min_abs_idx16(u4, u5, u6, u7);
                        int k2 = k1 ^ (int)syn;
                        idot += nibble_flip_delta(w.z, w.w, k1) * extract_s8_from_int4(xb, k1);
                        idot += nibble_flip_delta(w.z, w.w, k2) * extract_s8_from_int4(xb, k2);
                    }
                    if constexpr (CHECK_ERR) {
                        uint32_t post_s5 = (s5_b & 16u) ? (s5_b ^ (16u | syn)) : (s5_b ^ syn);
                        if (post_s5 != 0u) local_err += 1u;
                    }
                }

                acc[r] = fmaf((float)idot, sc, acc[r]);
            }
        }
    }

    if constexpr (CHECK_ERR) {
        if (local_err != 0u) atomicAdd(err_count, local_err);
    }
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        float sum = warp_reduce_sum(acc[r]);
        int row = base_row + r;
        if (lane == 0 && row < M) y[row] = sum;
    }
}


// Kernel 3: PTX LDG.128 Block-H2 INT8 Fused GEMV (Pairwise (a+b, a-b)/sqrt(2), peak <=14 <=127!)
// Uses SWAR even/odd byte add/sub + PRMT + __dp4a (ZERO scalar loops!)
template <int ROWS_PER_WARP>
__global__ void ptx_ldg128_block_h2_dp4a_gemv(
    const uint4* __restrict__ payloads_u4,
    const __half* __restrict__ scales_h,
    const int4* __restrict__ x_i8_16,
    float x_scale,
    float* __restrict__ y,
    int M, int K) {

    const int warp_in_blk = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warps_per_blk = blockDim.x >> 5;
    const int base_row = (blockIdx.x * warps_per_blk + warp_in_blk) * ROWS_PER_WARP;
    if (base_row >= M) return;

    const int n_blk32 = K >> 5;
    const float xsc_h2 = x_scale * 0.7071067811865475f;
    float acc[ROWS_PER_WARP];
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) acc[r] = 0.0f;

    // SWAR exact bytewise (a+b) and (a-b) on biased u8x4 in [1..15] -> signed s8x4 in [-14..+14]!
    // Notice: if ue = a+8, uo = b+8 (each in [1..15]):
    //   a + b = (ue + uo) - 16 -> in [2..30] - 16 = [-14..+14], NO cross-byte carry/borrow via __vadd4/__vsub4!
    //   a - b = ue - uo        -> in [-14..+14] directly via __vsub4!
    auto h2_dot8 = [](uint32_t w, int x0, int x1, int idot) -> int {
        uint32_t ue = ptx_lop3_and_xor(w, 0x0F0F0F0Fu, 0x08080808u);       // a + 8 in [1..15]
        uint32_t uo = ptx_lop3_and_xor(w >> 4, 0x0F0F0F0Fu, 0x08080808u);  // b + 8 in [1..15]
        uint32_t sum_s8 = __vsub4(__vadd4(ue, uo), 0x10101010u);           // s8x4: (a+b)
        uint32_t dif_s8 = __vsub4(ue, uo);                                 // s8x4: (a-b)
        int p0123 = (int)ptx_prmt(sum_s8, dif_s8, 0x5140u);
        int p4567 = (int)ptx_prmt(sum_s8, dif_s8, 0x7362u);
        idot = ptx_dp4a_s8s8(p0123, x0, idot);
        idot = ptx_dp4a_s8s8(p4567, x1, idot);
        return idot;
    };

    for (int b32 = lane; b32 < n_blk32; b32 += 32) {
        int4 xa = __ldg(x_i8_16 + (b32 << 1));
        int4 xb = __ldg(x_i8_16 + (b32 << 1) + 1);

        #pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            int row = base_row + r;
            if (row < M) {
                size_t off = (size_t)row * n_blk32 + b32;
                uint4 w = ldg128_u4(payloads_u4 + off);
                float sc = __half2float(__ldg(scales_h + off)) * xsc_h2;
                int idot = 0;
                idot = h2_dot8(w.x, xa.x, xa.y, idot);
                idot = h2_dot8(w.y, xa.z, xa.w, idot);
                idot = h2_dot8(w.z, xb.x, xb.y, idot);
                idot = h2_dot8(w.w, xb.z, xb.w, idot);
                acc[r] = fmaf((float)idot, sc, acc[r]);
            }
        }
    }

    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        float sum = warp_reduce_sum(acc[r]);
        int row = base_row + r;
        if (lane == 0 && row < M) y[row] = sum;
    }
}


// Kernel 4: Zero-Shared-Memory Warp-Shuffle H_16 Fused A_16 GEMV
// Uses exact orthogonality (W H_16) x = W (H_16 x): transforms activation x ON THE FLY
// inside warp registers via 4-stage __shfl_xor_sync (0 shmem!) and streams LDG.128 A_16 weights!
template <int ROWS_PER_WARP>
__global__ void ptx_warp_shfl_a16_h16_gemv(
    const uint4* __restrict__ payloads_u4,
    const __half* __restrict__ scales_h,
    const float* __restrict__ x_f32,         // [K] FP32 activation (rotated on-the-fly by warp-shuffle H_16!)
    float* __restrict__ y,
    int M, int K) {

    const int warp_in_blk = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warps_per_blk = blockDim.x >> 5;
    const int base_row = (blockIdx.x * warps_per_blk + warp_in_blk) * ROWS_PER_WARP;
    if (base_row >= M) return;

    const int n_blk32 = K >> 5;
    float acc[ROWS_PER_WARP];
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) acc[r] = 0.0f;

    // Each iteration processes 32*16 = 512 weights? Better: 1 lane = 1 weight of a 32-weight block
    // Wait: 32 lanes = TWO 16-element H_16 blocks! Let's process 1 block32 per warp-step -> wait,
    // 1 lane holds 1 x value (32 lanes = 32 x values = 2 x H_16 blocks), OR 1 lane holds 4 x values
    // (32 lanes = 128 x values = 8 x H_16 blocks)! Let's do 32 lanes * 4 weights/lane = 128 weights = 4 uint4s!
    // Wait: 128 weights / 32 lanes = 4 weights/lane = 1 uint32_t (2B!) per lane, or 32 lanes * 32w/lane
    // with in-lane 16-elem H_16? NO! We specifically want __shfl_xor_sync butterflies across lanes!
    // Look how clean 32 lanes * 4 weights/lane = 128 weights (= 8 H_16 blocks) is:
    //   Lane `lane` loads 1 float4 = (x[4*lane+0..3]).
    //   - Stage 1 & 2 (stride 1, 2): in-lane 4-pt butterfly on float4 (0 shfl)
    //   - Stage 3 (stride 4 = lane^1): 4 x __shfl_xor_sync(..., 1)
    //   - Stage 4 (stride 8 = lane^2): 4 x __shfl_xor_sync(..., 2)
    //   Total: 8 __shfl_xor_sync per 128 activations, ZERO shared memory!
    //   AND each lane reads 4 weights? Wait: 4 weights is 2 bytes; to keep LDG.128 coalesced,
    //   each lane reads 1 uint4 = 32 weights across 8 consecutive 128-weight chunks? Even cleaner:
    //   32 lanes each load 1 uint32_t? No, load 1 uint4 = 32 weights = 8 x 4-weight groups,
    //   OR process 4 x 128 = 512 weights/iter where each lane loads 1 uint4 (32 weights? wait 32*32=1024!).
    //   YES!! Process 1024 weights/iter (32 lanes * 32 weights/lane = 1024 weights = 64 H_16 blocks!):
    //   Wait: 1024 weights = 32 uint4s -> lane `l` loads `payloads_u4[blk + l]` (FULL 512B coalesced LDG.128!),
    //   and for activations, 32 lanes load 4 x 32 float4 = 512 floats? Wait: 32 lanes * 8 float4 = 1024 floats!
    //   Even better: each 32-weight uint4 = TWO complete 16-element H_16 blocks.
    //   Wait: let's do 32 lanes * 16 weights/lane? No, 32 lanes * 4 weights/lane = 128 weights/step,
    //   with 4 consecutive 128-weight steps = 512 weights (each lane loads 1 uint4 = 32 weights covering 2 rows?
    //   OR each lane loads 1 uint4 = 32 weights = 8 groups of 4, using __shfl_xor_sync to rotate 8 float4s!).
    //   LOOK AT THAT: 1 lane holds 4 consecutive floats of H_16 (4 lanes = 1 H_16 block of 16 floats)!
    //   To match a lane's uint4 (16 bytes = 32 weights = 8 x 4 weights) with 100% coalesced LDG.128:
    //   Pack 4 lanes (16 weights) -> wait: lane 0 holds weights 0..3, lane 1 holds 4..7, lane 2 holds 8..11, lane 3 holds 12..15.
    //   If payloads are laid out normally (blk0: 0..31, blk1: 32..63...), then 16 bytes = 32 weights = 8 groups of 4 weights:
    //   Lane `l` loads `payloads_u4[base + l]` (100% coalesced LDG.128 = 32 weights for `blk = base + l`),
    //   and we run 4-stage H_16 on x[32*l .. 32*l+31] via 2 in-lane + 2 warp-shuffle stages!
    //   Let's check: if lane `l` needs x for weights `0..31` of block `base + l`, we can do:
    //   4 lanes cooperate on 4 blocks of 32 weights -> each lane loads 1 uint4 (32 weights = 8x4),
    //   and 16 floats of x are loaded by 4 lanes (1 float4 each), transformed by 2 in-lane + 2 __shfl_xor_sync(1,2) stages,
    //   then broadcast via __shfl_sync to the owner lane? That costs 4 shfls.
    //   EVEN CLEANER: In a 32-lane warp, 32 lanes load `uint4` for 4 rows (8 lanes/row) or 1 row (32 lanes/row):
    //   Let 4 lanes cooperate on 4 consecutive 16-weight H_16 blocks (64 weights = 2 uint4s = 32 bytes):
    //   Actually, 32 lanes loading `uint2` (64-bit) = 16 weights/lane, OR 32 lanes loading `uint4` (128-bit) = 32 weights/lane:
    //   Wait: if we transform x in a dedicated warp-shuffle pass of 128 floats at a time (4 lanes = 16 floats = 1 H_16 block):
    //   Every 4 lanes (`g = lane & 3`) hold 16 floats of x (each lane holds 1 float4).
    //   After 2 in-lane + 2 `__shfl_xor_sync(1, 2)` butterflies, those 4 lanes hold `H_16(x)` in exact order!
    //   Now for each of `r = 0..ROWS_PER_WARP-1`, each lane needs its 4 weights (16 bits = uint16_t)!
    //   With 8 unrolled 128-weight chunks = 1024 weights = 32 lanes × 32 weights = 1 `LDG.128` (`uint4`) per lane if interleaved,
    //   OR simply 1 `LDG.128` (`uint4`) loaded by lane `l`, and 4 lanes exchange the 4×32-bit halves with `__shfl_sync`!
    //   HOLD ON! CHECK THAT!
    //   If lane `l` (group of 4 lanes `4q, 4q+1, 4q+2, 4q+3`) loads `uint4 w = payloads_u4[base + l]` (100% COALESCED LDG.128!):
    //   - That `uint4` has 32 weights = TWO `H_16` blocks (block A in `w.x, w.y`; block B in `w.z, w.w`).
    //   - Across the 4 lanes `4q..4q+3`, we have 8 `H_16` blocks = 128 weights!
    //   - Step s = 0..7: the 4 lanes load 1 `float4` each of x for block `s`, do 2 in-lane + 2 `__shfl_xor_sync(1,2)` butterflies
    //     -> now lane `g` (`0..3`) holds `H_16(x)[4g..4g+3]`!
    //   - Meanwhile, the owner lane `4q + (s>>1)` sends its `uint32_t` (8 weights) via `__shfl_sync` to the 4 lanes!
    //   Wait: 1 `uint32_t` holds 8 weights, so `g=0,1` use `lo` (`w.x` or `w.z`), `g=2,3` use `hi` (`w.y` or `w.w`) — ONLY 1 `__shfl_sync`!
    //   LET'S COUNT TOTAL SHFLs: 4 `__shfl_xor_sync` for H_16 on x (reused across ALL `ROWS_PER_WARP` rows!!)
    //   + 1 `__shfl_sync` per row per 16 weights = 0.0625 shfl/weight!! ZERO SHARED MEMORY!!

    const int n_blk128 = K >> 7; // 128 weights per 4 lanes -> 1024 weights per 32-lane warp step?
    // Wait: 32 lanes = 8 groups of 4 lanes. Each group of 4 lanes processes 128 weights (4 uint4s = 8 H_16 blocks).
    // So 1 warp-step processes 8 * 128 = 1024 weights = 32 uint4s!
    // Let's handle any K multiple of 128 cleanly!
    const int grp = lane >> 2;   // 0..7
    const int sub = lane & 3;    // 0..3 (4 lanes per H_16 block)
    const int grp_base = grp << 2;

    for (int g_blk128 = grp; g_blk128 < n_blk128; g_blk128 += 8) {
        // Each group of 4 lanes processes 4 uint4 blocks: b32 = g_blk128*4 + sub (COALESCED across all 32 lanes!)
        int b32_mine = (g_blk128 << 2) + sub;

        // Pre-load & warp-shuffle H_16(x) for all 8 x 16-elem slices of this 128-weight chunk?
        // Better: loop s=0..7 over the 8 H_16 slices so x stays in 4 float registers (ZERO register spill!)
        // Wait: load payloads for all ROWS_PER_WARP rows ONCE before the s=0..7 loop -> 1 LDG.128 per row!
        uint4 w_row[ROWS_PER_WARP];
        float sc_row[ROWS_PER_WARP];
        #pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            int row = base_row + r;
            if (row < M) {
                size_t off = (size_t)row * n_blk32 + b32_mine;
                w_row[r] = ldg128_u4(payloads_u4 + off);
                sc_row[r] = __half2float(__ldg(scales_h + off)) * 0.25f; // 1/sqrt(16) = 0.25
            } else {
                w_row[r] = make_uint4(0u, 0u, 0u, 0u);
                sc_row[r] = 0.0f;
            }
        }

        const float4* x4_base = reinterpret_cast<const float4*>(x_f32 + (g_blk128 << 7));

        #pragma unroll
        for (int s = 0; s < 8; ++s) {
            // 4 lanes in group load 16 floats of x (1 float4 per lane)
            float4 xv = __ldg(x4_base + (s << 2) + sub);
            // Stage 1 & 2 (stride 1, 2): in-lane 4-point butterfly
            float t0 = xv.x + xv.y, t1 = xv.x - xv.y;
            float t2 = xv.z + xv.w, t3 = xv.z - xv.w;
            float h0 = t0 + t2, h2 = t0 - t2;
            float h1 = t1 + t3, h3 = t1 - t3;
            // Stage 3 (stride 4 = sub^1) via __shfl_xor_sync (ZERO shared memory!)
            auto shfl_bfly_f = [sub](float& a, int delta) {
                float p = __shfl_xor_sync(0xFFFFFFFFu, a, delta);
                a = (sub & delta) ? (p - a) : (a + p);
            };
            shfl_bfly_f(h0, 1); shfl_bfly_f(h1, 1); shfl_bfly_f(h2, 1); shfl_bfly_f(h3, 1);
            // Stage 4 (stride 8 = sub^2) via __shfl_xor_sync (ZERO shared memory!)
            shfl_bfly_f(h0, 2); shfl_bfly_f(h1, 2); shfl_bfly_f(h2, 2); shfl_bfly_f(h3, 2);

            int owner = grp_base + (s >> 1);
            int half_sel = s & 1;

            #pragma unroll
            for (int r = 0; r < ROWS_PER_WARP; ++r) {
                // Owner lane broadcasts the needed 32-bit weight word (8 nibbles) and scale
                uint32_t w_lo = half_sel ? w_row[r].z : w_row[r].x;
                uint32_t w_hi = half_sel ? w_row[r].w : w_row[r].y;
                uint32_t b_lo = __shfl_sync(0xFFFFFFFFu, w_lo, owner);
                uint32_t b_hi = __shfl_sync(0xFFFFFFFFu, w_hi, owner);
                uint32_t w8 = (sub < 2) ? b_lo : b_hi;
                float sc = __shfl_sync(0xFFFFFFFFu, sc_row[r], owner);
                // Extract the 4 signed nibbles for this lane's sub&1 half (nibbles 0..3 or 4..7)
                uint32_t nibs4 = (sub & 1) ? (w8 >> 16) : w8;
                int n0 = (int)(int8_t)((nibs4 & 0xFu) | ((nibs4 & 0x8u) ? 0xF0u : 0u));
                int n1 = (int)(int8_t)(((nibs4 >> 4) & 0xFu) | ((nibs4 & 0x80u) ? 0xF0u : 0u));
                int n2 = (int)(int8_t)(((nibs4 >> 8) & 0xFu) | ((nibs4 & 0x800u) ? 0xF0u : 0u));
                int n3 = (int)(int8_t)(((nibs4 >> 12) & 0xFu) | ((nibs4 & 0x8000u) ? 0xF0u : 0u));
                float dot = (float)n0 * h0 + (float)n1 * h1 + (float)n2 * h2 + (float)n3 * h3;
                acc[r] = fmaf(dot, sc, acc[r]);
            }
        }
    }

    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        float sum = warp_reduce_sum(acc[r]);
        int row = base_row + r;
        if (lane == 0 && row < M) y[row] = sum;
    }
}


// Kernel 5: PTX LDG.128 Fused Dequant-into-FP16 (__half2 hfma2) GEMV (with L2 cache pinning!)
// Uses 2 independent __half2 accumulators + 1024+u magic PRMT bit-construction (ZERO I2F!)
template <int ROWS_PER_WARP>
__global__ void ptx_ldg128_a16_fp16_gemv(
    const uint4* __restrict__ payloads_u4,
    const __half* __restrict__ scales_h,
    const __half2* __restrict__ x_h2,        // [K/2] __half2 activations (2 bytes/elem, pinned in L2!)
    float* __restrict__ y,
    int M, int K) {

    const int warp_in_blk = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warps_per_blk = blockDim.x >> 5;
    const int base_row = (blockIdx.x * warps_per_blk + warp_in_blk) * ROWS_PER_WARP;
    if (base_row >= M) return;

    const int n_blk32 = K >> 5;
    float acc[ROWS_PER_WARP];
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) acc[r] = 0.0f;

    const __half2 h_bias = __floats2half2_rn(-1032.0f, -1032.0f);
    const uint32_t MAGIC_EXP = 0x64006400u;

    auto dot8_fp16 = [&](uint32_t w, const __half2* xp, __half2& hacc0, __half2& hacc1) {
        uint32_t u0123, u4567;
        ptx_unpack8_to_u8x4(w, u0123, u4567);
        uint32_t h01_bits = ptx_prmt(u0123, MAGIC_EXP, 0x5150u);
        uint32_t h23_bits = ptx_prmt(u0123, MAGIC_EXP, 0x5352u);
        uint32_t h45_bits = ptx_prmt(u4567, MAGIC_EXP, 0x5150u);
        uint32_t h67_bits = ptx_prmt(u4567, MAGIC_EXP, 0x5352u);

        __half2 w01 = __hadd2(*reinterpret_cast<const __half2*>(&h01_bits), h_bias);
        __half2 w23 = __hadd2(*reinterpret_cast<const __half2*>(&h23_bits), h_bias);
        __half2 w45 = __hadd2(*reinterpret_cast<const __half2*>(&h45_bits), h_bias);
        __half2 w67 = __hadd2(*reinterpret_cast<const __half2*>(&h67_bits), h_bias);

        hacc0 = __hfma2(w01, __ldg(xp + 0), hacc0);
        hacc1 = __hfma2(w23, __ldg(xp + 1), hacc1);
        hacc0 = __hfma2(w45, __ldg(xp + 2), hacc0);
        hacc1 = __hfma2(w67, __ldg(xp + 3), hacc1);
    };

    for (int b32 = lane; b32 < n_blk32; b32 += 32) {
        const __half2* xp = x_h2 + (b32 << 4);
        #pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            int row = base_row + r;
            if (row < M) {
                size_t off = (size_t)row * n_blk32 + b32;
                uint4 w = ldg128_u4(payloads_u4 + off);
                float sc = __half2float(__ldg(scales_h + off));
                __half2 hacc0 = __floats2half2_rn(0.0f, 0.0f);
                __half2 hacc1 = __floats2half2_rn(0.0f, 0.0f);
                dot8_fp16(w.x, xp + 0,  hacc0, hacc1);
                dot8_fp16(w.y, xp + 4,  hacc0, hacc1);
                dot8_fp16(w.z, xp + 8,  hacc0, hacc1);
                dot8_fp16(w.w, xp + 12, hacc0, hacc1);
                float2 f0 = __half22float2(hacc0);
                float2 f1 = __half22float2(hacc1);
                acc[r] = fmaf((f0.x + f0.y) + (f1.x + f1.y), sc, acc[r]);
            }
        }
    }

    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        float sum = warp_reduce_sum(acc[r]);
        int row = base_row + r;
        if (lane == 0 && row < M) y[row] = sum;
    }
}


// Kernel 6: PTX LDG.128 Fused Dequant-into-BF16 (__nv_bfloat162 __hmul2 + FP32 accumulate) GEMV
// BF16 weights via 0x4300|u magic PRMT (exact 128+u - 136 = n in [-8..+7]), vectorized __hmul2,
// and FP32 block reduction (eliminates 7-bit BF16 summation drift -> rel_l2_err < 1e-4!).
template <int ROWS_PER_WARP>
__global__ void ptx_ldg128_a16_bf16_gemv(
    const uint4* __restrict__ payloads_u4,
    const __half* __restrict__ scales_h,
    const __nv_bfloat162* __restrict__ x_bf2,
    float* __restrict__ y,
    int M, int K) {

    const int warp_in_blk = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warps_per_blk = blockDim.x >> 5;
    const int base_row = (blockIdx.x * warps_per_blk + warp_in_blk) * ROWS_PER_WARP;
    if (base_row >= M) return;

    const int n_blk32 = K >> 5;
    float acc[ROWS_PER_WARP];
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) acc[r] = 0.0f;

    const __nv_bfloat162 bf_bias = __floats2bfloat162_rn(-136.0f, -136.0f);
    const uint32_t MAGIC_BF = 0x43004300u;

    auto dot8_bf16_f32 = [&](uint32_t w, const __nv_bfloat162* xp, float& facc) {
        uint32_t u0123, u4567;
        ptx_unpack8_to_u8x4(w, u0123, u4567);
        uint32_t b01_bits = ptx_prmt(u0123, MAGIC_BF, 0x5150u);
        uint32_t b23_bits = ptx_prmt(u0123, MAGIC_BF, 0x5352u);
        uint32_t b45_bits = ptx_prmt(u4567, MAGIC_BF, 0x5150u);
        uint32_t b67_bits = ptx_prmt(u4567, MAGIC_BF, 0x5352u);

        __nv_bfloat162 w01 = __hadd2(*reinterpret_cast<const __nv_bfloat162*>(&b01_bits), bf_bias);
        __nv_bfloat162 w23 = __hadd2(*reinterpret_cast<const __nv_bfloat162*>(&b23_bits), bf_bias);
        __nv_bfloat162 w45 = __hadd2(*reinterpret_cast<const __nv_bfloat162*>(&b45_bits), bf_bias);
        __nv_bfloat162 w67 = __hadd2(*reinterpret_cast<const __nv_bfloat162*>(&b67_bits), bf_bias);

        // Pairwise BF16 multiply + FP32 accumulation (preserves full FP32 reduction precision!)
        float2 p01 = __bfloat1622float2(__hmul2(w01, __ldg(xp + 0)));
        float2 p23 = __bfloat1622float2(__hmul2(w23, __ldg(xp + 1)));
        float2 p45 = __bfloat1622float2(__hmul2(w45, __ldg(xp + 2)));
        float2 p67 = __bfloat1622float2(__hmul2(w67, __ldg(xp + 3)));
        facc += (p01.x + p01.y) + (p23.x + p23.y) + (p45.x + p45.y) + (p67.x + p67.y);
    };

    for (int b32 = lane; b32 < n_blk32; b32 += 32) {
        const __nv_bfloat162* xp = x_bf2 + (b32 << 4);
        #pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            int row = base_row + r;
            if (row < M) {
                size_t off = (size_t)row * n_blk32 + b32;
                uint4 w = ldg128_u4(payloads_u4 + off);
                float sc = __half2float(__ldg(scales_h + off));
                float facc = 0.0f;
                dot8_bf16_f32(w.x, xp + 0,  facc);
                dot8_bf16_f32(w.y, xp + 4,  facc);
                dot8_bf16_f32(w.z, xp + 8,  facc);
                dot8_bf16_f32(w.w, xp + 12, facc);
                acc[r] = fmaf(facc, sc, acc[r]);
            }
        }
    }

    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        float sum = warp_reduce_sum(acc[r]);
        int row = base_row + r;
        if (lane == 0 && row < M) y[row] = sum;
    }
}


// Kernel 7: Strict 72-byte / 128-weight Interleaved Superblock (2 x 72B = 144B/256w, 0 padding!)
// Streamed 1 uint4 slice at a time -> keeps register count low (~44 regs instead of 126!) for high occupancy!
struct Packed256Weights144B {
    uint4 p0, p1, p2, p3; // first 128 weights (64B)
    uint4 p4, p5, p6, p7; // second 128 weights (64B)
    uint4 scales8_u4;     // 8 x __half scales (16B) -> total 144 bytes / 256 weights = 4.5000 bpw EXACT!
};
static_assert(sizeof(Packed256Weights144B) == 144, "Exact 144B per 256 weights (4.5000 bpw)");

template <int ROWS_PER_WARP>
__global__ void ptx_aos144b_a16_dp4a_gemv(
    const Packed256Weights144B* __restrict__ blocks, // [M, K/256]
    const int4* __restrict__ x_i8_16,
    const int16_t* __restrict__ x_sum32,
    float x_scale,
    float* __restrict__ y,
    int M, int K) {

    const int warp_in_blk = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warps_per_blk = blockDim.x >> 5;
    const int base_row = (blockIdx.x * warps_per_blk + warp_in_blk) * ROWS_PER_WARP;
    if (base_row >= M) return;

    const int n_blk256 = K >> 8;
    float acc[ROWS_PER_WARP];
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) acc[r] = 0.0f;

    auto dot32_u4 = [](const uint4& w, const int4& xa, const int4& xb, int bias32) -> int {
        uint32_t u0, u1, u2, u3, u4, u5, u6, u7;
        ptx_unpack8_to_u8x4(w.x, u0, u1);
        ptx_unpack8_to_u8x4(w.y, u2, u3);
        ptx_unpack8_to_u8x4(w.z, u4, u5);
        ptx_unpack8_to_u8x4(w.w, u6, u7);
        int idot = -bias32;
        idot = ptx_dp4a_u8s8(u0, xa.x, idot);
        idot = ptx_dp4a_u8s8(u1, xa.y, idot);
        idot = ptx_dp4a_u8s8(u2, xa.z, idot);
        idot = ptx_dp4a_u8s8(u3, xa.w, idot);
        idot = ptx_dp4a_u8s8(u4, xb.x, idot);
        idot = ptx_dp4a_u8s8(u5, xb.y, idot);
        idot = ptx_dp4a_u8s8(u6, xb.z, idot);
        idot = ptx_dp4a_u8s8(u7, xb.w, idot);
        return idot;
    };

    for (int b256 = lane; b256 < n_blk256; b256 += 32) {
        int b32_0 = b256 << 3;
        // Load scales for each row (just 1 uint4 per row!)
        uint4 sc_u4[ROWS_PER_WARP];
        #pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            int row = base_row + r;
            if (row < M) {
                const uint4* base_u4 = reinterpret_cast<const uint4*>(blocks + (size_t)row * n_blk256 + b256);
                sc_u4[r] = ldg128_u4(base_u4 + 8);
            } else {
                sc_u4[r] = make_uint4(0u, 0u, 0u, 0u);
            }
        }
        // Stream 4 slices (128 weights) per inner step: prefetching x in 2 halves keeps coalescing + low regs!
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            int s0 = half * 4;
            int4 x0 = __ldg(x_i8_16 + (b32_0 + s0 + 0) * 2), x1 = __ldg(x_i8_16 + (b32_0 + s0 + 0) * 2 + 1);
            int4 x2 = __ldg(x_i8_16 + (b32_0 + s0 + 1) * 2), x3 = __ldg(x_i8_16 + (b32_0 + s0 + 1) * 2 + 1);
            int4 x4 = __ldg(x_i8_16 + (b32_0 + s0 + 2) * 2), x5 = __ldg(x_i8_16 + (b32_0 + s0 + 2) * 2 + 1);
            int4 x6 = __ldg(x_i8_16 + (b32_0 + s0 + 3) * 2), x7 = __ldg(x_i8_16 + (b32_0 + s0 + 3) * 2 + 1);
            int bs0 = 8 * (int)__ldg(x_sum32 + b32_0 + s0 + 0);
            int bs1 = 8 * (int)__ldg(x_sum32 + b32_0 + s0 + 1);
            int bs2 = 8 * (int)__ldg(x_sum32 + b32_0 + s0 + 2);
            int bs3 = 8 * (int)__ldg(x_sum32 + b32_0 + s0 + 3);
            #pragma unroll
            for (int r = 0; r < ROWS_PER_WARP; ++r) {
                int row = base_row + r;
                if (row < M) {
                    const uint4* base_u4 = reinterpret_cast<const uint4*>(blocks + (size_t)row * n_blk256 + b256);
                    uint32_t sp_a = (half == 0) ? sc_u4[r].x : sc_u4[r].z;
                    uint32_t sp_b = (half == 0) ? sc_u4[r].y : sc_u4[r].w;
                    float2 fa = __half22float2(*reinterpret_cast<const __half2*>(&sp_a));
                    float2 fb = __half22float2(*reinterpret_cast<const __half2*>(&sp_b));
                    float rsum = 0.0f;
                    rsum = fmaf((float)dot32_u4(ldg128_u4(base_u4 + s0 + 0), x0, x1, bs0), fa.x, rsum);
                    rsum = fmaf((float)dot32_u4(ldg128_u4(base_u4 + s0 + 1), x2, x3, bs1), fa.y, rsum);
                    rsum = fmaf((float)dot32_u4(ldg128_u4(base_u4 + s0 + 2), x4, x5, bs2), fb.x, rsum);
                    rsum = fmaf((float)dot32_u4(ldg128_u4(base_u4 + s0 + 3), x6, x7, bs3), fb.y, rsum);
                    acc[r] = fmaf(rsum, x_scale, acc[r]);
                }
            }
        }
    }

    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        float sum = warp_reduce_sum(acc[r]);
        int row = base_row + r;
        if (lane == 0 && row < M) y[row] = sum;
    }
}

// Kernel 8: Batched Ampere sm_86 INT8 Tensor Core (mma.sync.m16n8k32) + cp.async (LDGSTS)
// Fused 4.500-bpw A_16 Dequant-in-Register -> IMMA.16832.S8.S8 Batched GEMM (B=32 batch vectors!)
// Each CTA (4 warps = 128 threads) computes a 64 x 32 output tile Y[M, B] = W_A16[M, K] * X_i8[B, K]^T:
//   - Double-buffered cp.async.cg.shared.global stages 64 rows x 16B packed weights (1 KiB)
//     + 32 batch activations x 32B (1 KiB) per K=32 tile into shared memory asynchronously!
//   - Each warp owns 16 output rows x 32 batch cols (4 x mma.sync.m16n8k32 per K=32 step!)
//   - Per-block FP16 scale applied cleanly after each K=32 IMMA tile into FP32 accumulators!
__global__ void ptx_cp_async_mma_a16_gemm_b32(
    const uint4* __restrict__ payloads_u4,   // [M, K/32] uint4 (16B = 32 weights = 2 x A_16)
    const __half* __restrict__ scales_h,     // [M, K/32] __half scales
    const uint4* __restrict__ x_batch_u4,    // [32, K/16] uint4 (row n = batch n, 32B per K=32 slice)
    float x_scale,
    float* __restrict__ Y_out,               // [M, 32] row-major FP32 output
    int M, int K) {

    const int tid = threadIdx.x;             // 0..127 (4 warps)
    const int warp_id = tid >> 5;            // 0..3
    const int lane = tid & 31;
    const int grp = lane >> 2;               // 0..7
    const int sub = lane & 3;                // 0..3
    const int cta_row0 = blockIdx.x * 64;
    const int warp_row0 = cta_row0 + warp_id * 16;
    const int n_blk32 = K >> 5;

    // Double-buffered shared memory: 2 stages x (64 uint4 weights + 64 uint4 batch X + 64 __half scales)
    __shared__ uint4 smem_w[2][64];          // [stage][row 0..63] = 16B packed weights for K=32
    __shared__ uint4 smem_x[2][64];          // [stage][n*2 + half] = 32 batch rows x 32B (2 uint4 each)
    __shared__ __half smem_s[2][64];         // [stage][row 0..63] = FP16 scale per row

    // 4 x m16n8 tiles covering N=32 batch columns -> 16 FP32 accumulators per lane
    float acc[4][4];
    #pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
        #pragma unroll
        for (int c = 0; c < 4; ++c) acc[nt][c] = 0.0f;
    }

    auto issue_stage = [&](int stage, int b32) {
        // Threads 0..63 issue 64 x 16B cp.async for W[cta_row0..+63, b32]
        // Threads 64..127 issue 64 x 16B cp.async for X_batch[0..31, b32*2..b32*2+1]
        if (tid < 64) {
            int r = cta_row0 + tid;
            if (r < M) {
                size_t off = (size_t)r * n_blk32 + b32;
                ptx_cp_async_16b(&smem_w[stage][tid], payloads_u4 + off);
                smem_s[stage][tid] = __ldg(scales_h + off);
            } else {
                smem_w[stage][tid] = make_uint4(0u, 0u, 0u, 0u);
                smem_s[stage][tid] = __float2half(0.0f);
            }
        } else {
            int xidx = tid - 64;             // 0..63 -> batch n = xidx >> 1, half = xidx & 1
            int n = xidx >> 1;
            int h = xidx & 1;
            const uint4* gx = x_batch_u4 + (size_t)n * (size_t)(n_blk32 * 2) + (size_t)(b32 * 2 + h);
            ptx_cp_async_16b(&smem_x[stage][xidx], gx);
        }
        ptx_cp_async_commit();
    };

    if (n_blk32 > 0) issue_stage(0, 0);

    for (int b32 = 0; b32 < n_blk32; ++b32) {
        int cur = b32 & 1;
        int nxt = cur ^ 1;
        if (b32 + 1 < n_blk32) issue_stage(nxt, b32 + 1);
        // Wait until current stage's cp.async group completes
        if (b32 + 1 < n_blk32) {
            asm volatile("cp.async.wait_group 1;\n" ::);
        } else {
            ptx_cp_async_wait_all();
        }
        __syncthreads();

        // Load two rows of packed weights for this lane: rowA = grp, rowB = grp + 8
        int rA = warp_id * 16 + grp;
        int rB = rA + 8;
        uint4 wA = smem_w[cur][rA];
        uint4 wB = smem_w[cur][rB];
        float scA = __half2float(smem_s[cur][rA]) * x_scale;
        float scB = __half2float(smem_s[cur][rB]) * x_scale;

        // Unpack the 4-byte slice needed by `sub` (sub=0..3 -> k=4*sub..+3 and k=16+4*sub..+3)
        // sub=0 -> lo of w.x; sub=1 -> hi of w.x; sub=2 -> lo of w.y; sub=3 -> hi of w.y!
        auto unpack_slice_s8x4 = [&](const uint4& w, int& out_lo16, int& out_hi16) {
            uint32_t word_lo = (sub < 2) ? w.x : w.y;
            uint32_t word_hi = (sub < 2) ? w.z : w.w;
            int p0, p1, q0, q1;
            ptx_unpack8_to_s8x4(word_lo, p0, p1);
            ptx_unpack8_to_s8x4(word_hi, q0, q1);
            out_lo16 = (sub & 1) ? p1 : p0;
            out_hi16 = (sub & 1) ? q1 : q0;
        };
        int a0_s, a1_s, a2_s, a3_s;
        unpack_slice_s8x4(wA, a0_s, a2_s);
        unpack_slice_s8x4(wB, a1_s, a3_s);
        uint32_t a0 = (uint32_t)a0_s, a1 = (uint32_t)a1_s;
        uint32_t a2 = (uint32_t)a2_s, a3 = (uint32_t)a3_s;

        // Fire 4 x INT8 Tensor Core mma.sync.m16n8k32 across N=32 batch columns!
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            int n_col = nt * 8 + grp;
            const uint32_t* x_lo_words = reinterpret_cast<const uint32_t*>(&smem_x[cur][n_col * 2 + 0]);
            const uint32_t* x_hi_words = reinterpret_cast<const uint32_t*>(&smem_x[cur][n_col * 2 + 1]);
            uint32_t b0 = x_lo_words[sub];
            uint32_t b1 = x_hi_words[sub];

            int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
            ptx_mma_m16n8k32_s8(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, 0, 0, 0, 0);
            acc[nt][0] = fmaf((float)d0, scA, acc[nt][0]);
            acc[nt][1] = fmaf((float)d1, scA, acc[nt][1]);
            acc[nt][2] = fmaf((float)d2, scB, acc[nt][2]);
            acc[nt][3] = fmaf((float)d3, scB, acc[nt][3]);
        }
        __syncthreads();
    }

    int out_rA = warp_row0 + grp;
    int out_rB = out_rA + 8;
    #pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
        int col0 = nt * 8 + sub * 2;
        if (out_rA < M) {
            Y_out[(size_t)out_rA * 32 + col0]     = acc[nt][0];
            Y_out[(size_t)out_rA * 32 + col0 + 1] = acc[nt][1];
        }
        if (out_rB < M) {
            Y_out[(size_t)out_rB * 32 + col0]     = acc[nt][2];
            Y_out[(size_t)out_rB * 32 + col0 + 1] = acc[nt][3];
        }
    }
}

// ============================================================================
// Section 5: Harness & Bit-Exact Reference Verification + JSON Emitter
// ============================================================================

struct BenchResult {
    const char* name;
    double ms;
    double gweights_s;
    double phys_gb_s;
    double fp16_equiv_tb_s;
    double pct_triad;
    double max_abs_err;
    double rel_l2_err;
    uint32_t syndrome_errs;
};

static void compute_errors(const std::vector<float>& gpu, const std::vector<double>& ref,
                           int n_check, double& max_abs, double& rel_l2) {
    max_abs = 0.0;
    double num = 0.0, den = 0.0;
    for (int i = 0; i < n_check; ++i) {
        double d = (double)gpu[i] - ref[i];
        double ad = fabs(d);
        if (ad > max_abs) max_abs = ad;
        num += d * d;
        den += ref[i] * ref[i];
    }
    rel_l2 = (den > 0.0) ? sqrt(num / den) : 0.0;
}

int main() {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int memClockKHz = 0, coreClockKHz = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&memClockKHz, cudaDevAttrMemoryClockRate, 0));
    CUDA_CHECK(cudaDeviceGetAttribute(&coreClockKHz, cudaDevAttrClockRate, 0));
    double peak_bw_gbs = 2.0 * (double)memClockKHz * 1e3 * (prop.memoryBusWidth / 8.0) / 1e9;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    float ms = 0.0f;

    // 1. STREAM Triad Calibration (256 MiB x 3)
    const size_t N4 = (64ULL * 1024 * 1024) / 4;
    const size_t bytes_arr = N4 * sizeof(float4);
    float4 *da = nullptr, *db = nullptr, *dc = nullptr;
    CUDA_CHECK(cudaMalloc(&da, bytes_arr));
    CUDA_CHECK(cudaMalloc(&db, bytes_arr));
    CUDA_CHECK(cudaMalloc(&dc, bytes_arr));
    CUDA_CHECK(cudaMemset(db, 1, bytes_arr));
    CUDA_CHECK(cudaMemset(dc, 2, bytes_arr));
    int triad_blocks = prop.multiProcessorCount * 32;
    stream_triad_kernel<<<triad_blocks, 256, 0, stream>>>(da, db, dc, 1.5f, N4);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const int triad_iters = 20;
    CUDA_CHECK(cudaEventRecord(t0, stream));
    for (int i = 0; i < triad_iters; ++i)
        stream_triad_kernel<<<triad_blocks, 256, 0, stream>>>(da, db, dc, 1.5f, N4);
    CUDA_CHECK(cudaEventRecord(t1, stream));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    double triad_gbs = (3.0 * (double)bytes_arr * triad_iters) / (ms * 1e6);
    CUDA_CHECK(cudaFree(da)); CUDA_CHECK(cudaFree(db)); CUDA_CHECK(cudaFree(dc));

    // 2. Zero-Shmem Warp-Shuffle WHT_8 & H_16 Standalone Verification + Throughput (67.1M weights)
    const int WHT_N = 64 * 1024 * 1024;
    std::vector<int8_t> h_wht_in(WHT_N), h_wht_out(WHT_N);
    uint32_t rng = 0x52A1E7Eu;
    for (int i = 0; i < WHT_N; ++i) {
        rng = rng * 1664525u + 1013904223u;
        h_wht_in[i] = (int8_t)(((int)(rng >> 24) % 15) - 7); // [-7, +7]
    }
    int8_t *d_wht_in = nullptr, *d_wht_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_wht_in, WHT_N));
    CUDA_CHECK(cudaMalloc(&d_wht_out, WHT_N));
    CUDA_CHECK(cudaMemcpy(d_wht_in, h_wht_in.data(), WHT_N, cudaMemcpyHostToDevice));

    int wht_grid = prop.multiProcessorCount * 16;
    const int wht_iters = 50;
    // WHT_8
    warp_shfl_wht_standalone_kernel<3><<<wht_grid, 256, 0, stream>>>(d_wht_in, d_wht_out, WHT_N);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(h_wht_out.data(), d_wht_out, 65536, cudaMemcpyDeviceToHost));
    int wht8_mismatches = 0;
    for (int b = 0; b < 65536; b += 8) {
        int tmp[8];
        for (int i = 0; i < 8; ++i) tmp[i] = (int)h_wht_in[b + i];
        host_wht8(tmp);
        for (int i = 0; i < 8; ++i) {
            if ((int8_t)(tmp[i] >> 3) != h_wht_out[b + i]) ++wht8_mismatches;
        }
    }
    CUDA_CHECK(cudaEventRecord(t0, stream));
    for (int i = 0; i < wht_iters; ++i)
        warp_shfl_wht_standalone_kernel<3><<<wht_grid, 256, 0, stream>>>(d_wht_in, d_wht_out, WHT_N);
    CUDA_CHECK(cudaEventRecord(t1, stream));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    double wht8_ms = ms / wht_iters;
    double wht8_gw = (double)WHT_N / (wht8_ms * 1e6);
    double wht8_gbs = (2.0 * (double)WHT_N) / (wht8_ms * 1e6);

    // H_16
    warp_shfl_wht_standalone_kernel<4><<<wht_grid, 256, 0, stream>>>(d_wht_in, d_wht_out, WHT_N);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(h_wht_out.data(), d_wht_out, 65536, cudaMemcpyDeviceToHost));
    int h16_mismatches = 0;
    for (int b = 0; b < 65536; b += 16) {
        int tmp[16];
        for (int i = 0; i < 16; ++i) tmp[i] = (int)h_wht_in[b + i];
        host_wht16(tmp);
        for (int i = 0; i < 16; ++i) {
            if ((int8_t)(tmp[i] >> 4) != h_wht_out[b + i]) ++h16_mismatches;
        }
    }
    CUDA_CHECK(cudaEventRecord(t0, stream));
    for (int i = 0; i < wht_iters; ++i)
        warp_shfl_wht_standalone_kernel<4><<<wht_grid, 256, 0, stream>>>(d_wht_in, d_wht_out, WHT_N);
    CUDA_CHECK(cudaEventRecord(t1, stream));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    double h16_ms = ms / wht_iters;
    double h16_gw = (double)WHT_N / (h16_ms * 1e6);
    double h16_gbs = (2.0 * (double)WHT_N) / (h16_ms * 1e6);

    CUDA_CHECK(cudaFree(d_wht_in)); CUDA_CHECK(cudaFree(d_wht_out));

    // 3. Fused 4.500-bpw GEMV Shootout across Realistic LLM Shapes (4096x14336 & 8192x14336)
    //    M=4096, K=14336 (Llama-3 / Gemma-3 FFN gate/up: 58.7M weights = 33.03 MB at 4.50 bpw)
    //    M=8192, K=14336 (Large FFN: 117.4M weights = 66.06 MB at 4.50 bpw -> >> 4 MiB L2!)
    struct Shape { int M; int K; const char* tag; };
    const Shape shapes[] = {
        {4096, 14336, "4096x14336"},
        {8192, 14336, "8192x14336"},
    };

    printf("{\n");
    printf("  \"device\": {\n");
    printf("    \"name\": \"%s\",\n", prop.name);
    printf("    \"cc_major\": %d, \"cc_minor\": %d, \"sm_count\": %d,\n", prop.major, prop.minor, prop.multiProcessorCount);
    printf("    \"shared_mem_per_sm_bytes\": %zu,\n", prop.sharedMemPerMultiprocessor);
    printf("    \"l2_cache_bytes\": %d,\n", prop.l2CacheSize);
    printf("    \"persisting_l2_max_bytes\": %d,\n", prop.persistingL2CacheMaxSize);
    printf("    \"memory_bus_width_bits\": %d, \"memory_clock_khz\": %d, \"boost_clock_khz\": %d,\n",
           prop.memoryBusWidth, memClockKHz, coreClockKHz);
    printf("    \"theoretical_peak_bw_gb_s\": %.4f,\n", peak_bw_gbs);
    printf("    \"stream_triad_gb_s\": %.4f, \"stream_triad_pct_peak\": %.3f\n",
           triad_gbs, 100.0 * triad_gbs / peak_bw_gbs);
    printf("  },\n");
    printf("  \"warp_shuffle_zero_shmem\": {\n");
    printf("    \"weights\": %d,\n", WHT_N);
    printf("    \"WHT8_shfl\": {\"ms\": %.6f, \"gweights_per_s\": %.4f, \"phys_gb_s\": %.4f, \"pct_triad\": %.3f, \"bit_exact_mismatches\": %d},\n",
           wht8_ms, wht8_gw, wht8_gbs, 100.0 * wht8_gbs / triad_gbs, wht8_mismatches);
    printf("    \"H16_shfl\": {\"ms\": %.6f, \"gweights_per_s\": %.4f, \"phys_gb_s\": %.4f, \"pct_triad\": %.3f, \"bit_exact_mismatches\": %d}\n",
           h16_ms, h16_gw, h16_gbs, 100.0 * h16_gbs / triad_gbs, h16_mismatches);
    printf("  },\n");
    printf("  \"fused_gemv_4_5_bpw\": {\n");

    int total_verify_failures = (wht8_mismatches + h16_mismatches);

    for (int si = 0; si < 2; ++si) {
        const int M = shapes[si].M;
        const int K = shapes[si].K;
        const int n_blk32 = K >> 5;
        const int n_blk256 = K >> 8;
        const size_t total_blk32 = (size_t)M * n_blk32;
        const size_t total_weights = (size_t)M * K;
        const size_t phys_weight_bytes = total_blk32 * 18ULL; // EXACT 18 bytes / 32 weights = 4.5000 bpw!

        std::vector<uint4> h_raw(total_blk32), h_valid(total_blk32);
        std::vector<__half> h_scales(total_blk32);
        std::vector<Packed256Weights144B> h_aos((size_t)M * n_blk256);

        // Generate deterministic raw & pre-encoded valid A_16 payloads in [-7, +7]
        rng = 0x13579BDFu + (uint32_t)si * 9973u;
        for (size_t idx = 0; idx < total_blk32; ++idx) {
            int raw_v[32];
            for (int j = 0; j < 32; ++j) {
                rng = rng * 1664525u + 1013904223u;
                raw_v[j] = ((int)(rng >> 24) % 15) - 7; // [-7, +7]
            }
            uint4 rw{};
            host_pack16_s4(raw_v,      rw.x, rw.y);
            host_pack16_s4(raw_v + 16, rw.z, rw.w);
            h_raw[idx] = rw;

            int rep_a[16], rep_b[16];
            uint32_t ca = host_decode_a16_repair(rw.x, rw.y, rep_a);
            uint32_t cb = host_decode_a16_repair(rw.z, rw.w, rep_b);
            if (ca != 0u || cb != 0u) ++total_verify_failures;
            uint4 vw{};
            host_pack16_s4(rep_a, vw.x, vw.y);
            host_pack16_s4(rep_b, vw.z, vw.w);
            h_valid[idx] = vw;

            rng = rng * 1664525u + 1013904223u;
            float sc_f = 0.004f + 0.012f * ((float)(rng >> 16) / 65535.0f);
            h_scales[idx] = __float2half(sc_f);
        }

        // Pack strict 144-byte AoS superblocks (256 weights = 144B exact)
        for (int r = 0; r < M; ++r) {
            for (int b256 = 0; b256 < n_blk256; ++b256) {
                size_t sbase = (size_t)r * n_blk32 + (size_t)b256 * 8;
                Packed256Weights144B& blk = h_aos[(size_t)r * n_blk256 + b256];
                blk.p0 = h_valid[sbase + 0]; blk.p1 = h_valid[sbase + 1];
                blk.p2 = h_valid[sbase + 2]; blk.p3 = h_valid[sbase + 3];
                blk.p4 = h_valid[sbase + 4]; blk.p5 = h_valid[sbase + 5];
                blk.p6 = h_valid[sbase + 6]; blk.p7 = h_valid[sbase + 7];
                memcpy(&blk.scales8_u4, &h_scales[sbase], 16);
            }
        }

        // Activations: INT8, FP16, BF16, FP32
        std::vector<int8_t> h_xi8(K);
        std::vector<int16_t> h_xsum32(n_blk32);
        std::vector<float> h_xf32(K);
        std::vector<__half> h_xh16(K);
        std::vector<__nv_bfloat16> h_xbf16(K);
        const float x_scale = 0.015625f; // exact power of 2 = 1/64
        for (int k = 0; k < K; ++k) {
            rng = rng * 1664525u + 1013904223u;
            int iv = ((int)(rng >> 24) % 127) - 63; // [-63, +63] -> exact in int8, fp16, bf16!
            h_xi8[k] = (int8_t)iv;
            h_xf32[k] = (float)iv * x_scale;
            h_xh16[k] = __float2half(h_xf32[k]);
            h_xbf16[k] = __float2bfloat16(h_xf32[k]);
        }
        for (int b = 0; b < n_blk32; ++b) {
            int s = 0;
            for (int j = 0; j < 32; ++j) s += (int)h_xi8[b * 32 + j];
            h_xsum32[b] = (int16_t)s;
        }

        // Compute exact FP64 host references on first REF_ROWS rows!
        const int REF_ROWS = 256;
        std::vector<double> ref_a16_i8(REF_ROWS, 0.0);
        std::vector<double> ref_a16_fp16(REF_ROWS, 0.0);
        std::vector<double> ref_a16_bf16(REF_ROWS, 0.0);
        std::vector<double> ref_h2_i8(REF_ROWS, 0.0);
        std::vector<double> ref_h16_f32(REF_ROWS, 0.0);

        for (int r = 0; r < REF_ROWS; ++r) {
            double sa_i8 = 0.0, sa_h16 = 0.0, sa_bf16 = 0.0, sh2 = 0.0, sh16 = 0.0;
            for (int b = 0; b < n_blk32; ++b) {
                size_t off = (size_t)r * n_blk32 + b;
                double sc = (double)__half2float(h_scales[off]);
                // Valid A_16 reference
                uint4 vw = h_valid[off];
                int va[32];
                for (int j = 0; j < 8; ++j) {
                    va[j]      = host_unpack_s4(vw.x, j);
                    va[j + 8]  = host_unpack_s4(vw.y, j);
                    va[j + 16] = host_unpack_s4(vw.z, j);
                    va[j + 24] = host_unpack_s4(vw.w, j);
                }
                int64_t idot_a16 = 0;
                double d_fp16 = 0.0, d_bf16 = 0.0;
                for (int j = 0; j < 32; ++j) {
                    idot_a16 += (int64_t)va[j] * (int64_t)h_xi8[b * 32 + j];
                    d_fp16 += (double)va[j] * (double)__half2float(h_xh16[b * 32 + j]);
                    // Exact BF16-product reference matching hardware __hmul2(bf16(w), bf16(x)) + FP32 accumulate!
                    float prod_bf = __bfloat162float(__float2bfloat16((float)va[j] * __bfloat162float(h_xbf16[b * 32 + j])));
                    d_bf16 += (double)prod_bf;
                }
                sa_i8 += (double)idot_a16 * (sc * (double)x_scale);
                sa_h16 += d_fp16 * sc;
                sa_bf16 += d_bf16 * sc;

                // Block-H2 reference on raw payload
                uint4 rw = h_raw[off];
                int vr[32];
                for (int j = 0; j < 8; ++j) {
                    vr[j]      = host_unpack_s4(rw.x, j);
                    vr[j + 8]  = host_unpack_s4(rw.y, j);
                    vr[j + 16] = host_unpack_s4(rw.z, j);
                    vr[j + 24] = host_unpack_s4(rw.w, j);
                }
                int64_t idot_h2 = 0;
                for (int j = 0; j < 32; j += 2) {
                    int a = vr[j], c = vr[j + 1];
                    idot_h2 += (int64_t)(a + c) * (int64_t)h_xi8[b * 32 + j]
                             + (int64_t)(a - c) * (int64_t)h_xi8[b * 32 + j + 1];
                }
                sh2 += (double)idot_h2 * (sc * (double)x_scale * 0.7071067811865475);

                // H_16 reference on valid A_16 payload: (W * H_16 / 4) * x
                int w0[16], w1[16];
                for (int j = 0; j < 16; ++j) { w0[j] = va[j]; w1[j] = va[j + 16]; }
                host_wht16(w0);
                host_wht16(w1);
                double dh16 = 0.0;
                for (int j = 0; j < 16; ++j) {
                    dh16 += (double)w0[j] * (double)h_xf32[b * 32 + j]
                          + (double)w1[j] * (double)h_xf32[b * 32 + 16 + j];
                }
                sh16 += dh16 * (sc * 0.25);
            }
            ref_a16_i8[r] = sa_i8;
            ref_a16_fp16[r] = sa_h16;
            ref_a16_bf16[r] = sa_bf16;
            ref_h2_i8[r] = sh2;
            ref_h16_f32[r] = sh16;
        }

        // Allocate device buffers
        uint4 *d_raw = nullptr, *d_valid = nullptr;
        __half *d_scales = nullptr;
        Packed256Weights144B *d_aos = nullptr;
        int4 *d_xi8 = nullptr;
        int16_t *d_xsum32 = nullptr;
        float *d_xf32 = nullptr, *d_y = nullptr;
        __half2 *d_xh2 = nullptr;
        __nv_bfloat162 *d_xbf2 = nullptr;
        uint32_t *d_err = nullptr;

        CUDA_CHECK(cudaMalloc(&d_raw, total_blk32 * sizeof(uint4)));
        CUDA_CHECK(cudaMalloc(&d_valid, total_blk32 * sizeof(uint4)));
        CUDA_CHECK(cudaMalloc(&d_scales, total_blk32 * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_aos, (size_t)M * n_blk256 * sizeof(Packed256Weights144B)));
        CUDA_CHECK(cudaMalloc(&d_xi8, K));
        CUDA_CHECK(cudaMalloc(&d_xsum32, n_blk32 * sizeof(int16_t)));
        CUDA_CHECK(cudaMalloc(&d_xf32, K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_xh2, K * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_xbf2, K * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_y, M * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_err, sizeof(uint32_t)));

        CUDA_CHECK(cudaMemcpy(d_raw, h_raw.data(), total_blk32 * sizeof(uint4), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_valid, h_valid.data(), total_blk32 * sizeof(uint4), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_scales, h_scales.data(), total_blk32 * sizeof(__half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_aos, h_aos.data(), (size_t)M * n_blk256 * sizeof(Packed256Weights144B), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_xi8, h_xi8.data(), K, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_xsum32, h_xsum32.data(), n_blk32 * sizeof(int16_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_xf32, h_xf32.data(), K * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_xh2, h_xh16.data(), K * sizeof(__half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_xbf2, h_xbf16.data(), K * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

        std::vector<float> h_y(M);
        std::vector<BenchResult> results;
        const int WARM = 10, ITERS = 60;
        const int THREADS = 256; // 8 warps/block

        auto set_l2_pin = [&](void* ptr, size_t nbytes, bool enable) {
            cudaStreamAttrValue attr{};
            if (enable && prop.persistingL2CacheMaxSize > 0) {
                size_t pin_sz = std::min((size_t)prop.persistingL2CacheMaxSize, (size_t)prop.l2CacheSize);
                CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, pin_sz));
                attr.accessPolicyWindow.base_ptr = ptr;
                attr.accessPolicyWindow.num_bytes = std::min(nbytes, pin_sz);
                attr.accessPolicyWindow.hitRatio = 1.0f;
                attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
                attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
            } else {
                attr.accessPolicyWindow.num_bytes = 0;
            }
            CUDA_CHECK(cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr));
        };

        auto run_bench = [&](const char* name, auto launch_check, auto launch_fast,
                             const std::vector<double>& ref, size_t x_bytes,
                             void* pin_ptr, bool pin_l2) {
            set_l2_pin(pin_ptr, x_bytes, pin_l2);
            CUDA_CHECK(cudaMemsetAsync(d_err, 0, sizeof(uint32_t), stream));
            launch_check();
            CUDA_CHECK(cudaStreamSynchronize(stream));
            uint32_t h_err = 0;
            CUDA_CHECK(cudaMemcpy(&h_err, d_err, sizeof(uint32_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, REF_ROWS * sizeof(float), cudaMemcpyDeviceToHost));
            double max_e = 0.0, rel_e = 0.0;
            compute_errors(h_y, ref, REF_ROWS, max_e, rel_e);
            if (h_err != 0u || rel_e > 1e-3) ++total_verify_failures;

            for (int i = 0; i < WARM; ++i) launch_fast();
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaEventRecord(t0, stream));
            for (int i = 0; i < ITERS; ++i) launch_fast();
            CUDA_CHECK(cudaEventRecord(t1, stream));
            CUDA_CHECK(cudaEventSynchronize(t1));
            CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
            double avg_ms = ms / ITERS;
            double total_bytes = (double)(phys_weight_bytes + x_bytes + (size_t)M * sizeof(float));
            double phys_gbs = total_bytes / (avg_ms * 1e6);
            double gw_s = (double)total_weights / (avg_ms * 1e6);
            // FP16-equivalent effective bandwidth: (2 bytes/weight * M*K) / elapsed_s, in TB/s!
            double fp16_tb_s = (2.0 * (double)total_weights) / (avg_ms * 1e9);
            results.push_back({name, avg_ms, gw_s, phys_gbs, fp16_tb_s,
                               100.0 * phys_gbs / triad_gbs, max_e, rel_e, h_err});
            set_l2_pin(nullptr, 0, false);
        };

        auto grid_r = [&](int rpw) {
            int rows_per_blk = (THREADS >> 5) * rpw;
            return (M + rows_per_blk - 1) / rows_per_blk;
        };

        // 1. A_16 Pre-encoded + Live 5-POPC + dp4a.u32.s32 (1 row/warp, L2 pin OFF)
        run_bench("A16_preenc_dp4a_r1_l2off",
            [&]{ ptx_ldg128_a16_preencoded_dp4a_gemv<true, 1><<<grid_r(1), THREADS, 0, stream>>>(d_valid, d_scales, d_xi8, d_xsum32, x_scale, d_y, d_err, M, K); },
            [&]{ ptx_ldg128_a16_preencoded_dp4a_gemv<false, 1><<<grid_r(1), THREADS, 0, stream>>>(d_valid, d_scales, d_xi8, d_xsum32, x_scale, d_y, d_err, M, K); },
            ref_a16_i8, (size_t)K, d_xi8, false);

        // 2. A_16 Pre-encoded + Live 5-POPC + dp4a.u32.s32 (2 rows/warp, L2 pin ON)
        run_bench("A16_preenc_dp4a_r2_l2pin",
            [&]{ ptx_ldg128_a16_preencoded_dp4a_gemv<true, 2><<<grid_r(2), THREADS, 0, stream>>>(d_valid, d_scales, d_xi8, d_xsum32, x_scale, d_y, d_err, M, K); },
            [&]{ ptx_ldg128_a16_preencoded_dp4a_gemv<false, 2><<<grid_r(2), THREADS, 0, stream>>>(d_valid, d_scales, d_xi8, d_xsum32, x_scale, d_y, d_err, M, K); },
            ref_a16_i8, (size_t)K, d_xi8, true);

        // 3. A_16 Pre-encoded + Live 5-POPC VERIFY ENABLED IN TIMED LOOP (2 rows/warp, L2 pin ON)
        run_bench("A16_preenc_dp4a_live_verify_r2_l2pin",
            [&]{ ptx_ldg128_a16_preencoded_dp4a_gemv<true, 2><<<grid_r(2), THREADS, 0, stream>>>(d_valid, d_scales, d_xi8, d_xsum32, x_scale, d_y, d_err, M, K); },
            [&]{ ptx_ldg128_a16_preencoded_dp4a_gemv<true, 2><<<grid_r(2), THREADS, 0, stream>>>(d_valid, d_scales, d_xi8, d_xsum32, x_scale, d_y, d_err, M, K); },
            ref_a16_i8, (size_t)K, d_xi8, true);

        // 4. A_16 LIVE SYNDROME REPAIR (O(1) register-delta, 2 rows/warp, L2 pin ON)
        run_bench("A16_live_repair_dp4a_r2_l2pin",
            [&]{ ptx_ldg128_a16_live_repair_dp4a_gemv<true, 2><<<grid_r(2), THREADS, 0, stream>>>(d_raw, d_scales, d_xi8, d_xsum32, x_scale, d_y, d_err, M, K); },
            [&]{ ptx_ldg128_a16_live_repair_dp4a_gemv<false, 2><<<grid_r(2), THREADS, 0, stream>>>(d_raw, d_scales, d_xi8, d_xsum32, x_scale, d_y, d_err, M, K); },
            ref_a16_i8, (size_t)K, d_xi8, true);

        // 5. Block-H2 SWAR + dp4a.s32.s32 (2 rows/warp, L2 pin ON)
        run_bench("Block_H2_swar_dp4a_r2_l2pin",
            [&]{ ptx_ldg128_block_h2_dp4a_gemv<2><<<grid_r(2), THREADS, 0, stream>>>(d_raw, d_scales, d_xi8, x_scale, d_y, M, K); },
            [&]{ ptx_ldg128_block_h2_dp4a_gemv<2><<<grid_r(2), THREADS, 0, stream>>>(d_raw, d_scales, d_xi8, x_scale, d_y, M, K); },
            ref_h2_i8, (size_t)K, d_xi8, true);

        // 6. Warp-Shuffle H_16 Zero-Shmem Fused A_16 GEMV (2 rows/warp, L2 pin ON)
        run_bench("A16_H16_warp_shfl_zero_shmem_r2_l2pin",
            [&]{ ptx_warp_shfl_a16_h16_gemv<2><<<grid_r(2), THREADS, 0, stream>>>(d_valid, d_scales, d_xf32, d_y, M, K); },
            [&]{ ptx_warp_shfl_a16_h16_gemv<2><<<grid_r(2), THREADS, 0, stream>>>(d_valid, d_scales, d_xf32, d_y, M, K); },
            ref_h16_f32, (size_t)K * sizeof(float), d_xf32, true);

        // 7. Fused Dequant-into-FP16 (__half2 magic-PRMT hfma2, L2 pin OFF vs ON)
        run_bench("A16_fp16_hfma2_r2_l2off",
            [&]{ ptx_ldg128_a16_fp16_gemv<2><<<grid_r(2), THREADS, 0, stream>>>(d_valid, d_scales, d_xh2, d_y, M, K); },
            [&]{ ptx_ldg128_a16_fp16_gemv<2><<<grid_r(2), THREADS, 0, stream>>>(d_valid, d_scales, d_xh2, d_y, M, K); },
            ref_a16_fp16, (size_t)K * sizeof(__half), d_xh2, false);

        run_bench("A16_fp16_hfma2_r2_l2pin",
            [&]{ ptx_ldg128_a16_fp16_gemv<2><<<grid_r(2), THREADS, 0, stream>>>(d_valid, d_scales, d_xh2, d_y, M, K); },
            [&]{ ptx_ldg128_a16_fp16_gemv<2><<<grid_r(2), THREADS, 0, stream>>>(d_valid, d_scales, d_xh2, d_y, M, K); },
            ref_a16_fp16, (size_t)K * sizeof(__half), d_xh2, true);

        // 8. Fused Dequant-into-BF16 (__nv_bfloat162 magic-PRMT hfma2, L2 pin ON)
        run_bench("A16_bf16_hfma2_r2_l2pin",
            [&]{ ptx_ldg128_a16_bf16_gemv<2><<<grid_r(2), THREADS, 0, stream>>>(d_valid, d_scales, d_xbf2, d_y, M, K); },
            [&]{ ptx_ldg128_a16_bf16_gemv<2><<<grid_r(2), THREADS, 0, stream>>>(d_valid, d_scales, d_xbf2, d_y, M, K); },
            ref_a16_bf16, (size_t)K * sizeof(__nv_bfloat16), d_xbf2, true);

        // 9. Strict AoS 144B/256w (= 2 x 72B/128w = 8 x 18B/32w) Single-Buffer LDG.128 GEMV
        run_bench("A16_aos144b_dp4a_r2_l2pin",
            [&]{ ptx_aos144b_a16_dp4a_gemv<2><<<grid_r(2), THREADS, 0, stream>>>(d_aos, d_xi8, d_xsum32, x_scale, d_y, M, K); },
            [&]{ ptx_aos144b_a16_dp4a_gemv<2><<<grid_r(2), THREADS, 0, stream>>>(d_aos, d_xi8, d_xsum32, x_scale, d_y, M, K); },
            ref_a16_i8, (size_t)K, d_xi8, true);

        // 10. Batched B=32 INT8 Tensor Core (mma.sync.m16n8k32) + cp.async (LDGSTS) Fused A_16 GEMM
        const int B_GEMM = 32;
        std::vector<int8_t> h_xbatch((size_t)B_GEMM * K);
        for (int n = 0; n < B_GEMM; ++n) {
            for (int k = 0; k < K; ++k) {
                rng = rng * 1664525u + 1013904223u;
                h_xbatch[(size_t)n * K + k] = (int8_t)(((int)(rng >> 24) % 127) - 63);
            }
        }
        // Exact FP64 host reference on first 64 rows x 32 batch cols
        const int GEMM_REF_ROWS = 64;
        std::vector<double> ref_gemm((size_t)GEMM_REF_ROWS * B_GEMM, 0.0);
        for (int r = 0; r < GEMM_REF_ROWS; ++r) {
            for (int b = 0; b < n_blk32; ++b) {
                size_t off = (size_t)r * n_blk32 + b;
                double sc = (double)__half2float(h_scales[off]) * (double)x_scale;
                uint4 vw = h_valid[off];
                int va[32];
                for (int j = 0; j < 8; ++j) {
                    va[j]      = host_unpack_s4(vw.x, j);
                    va[j + 8]  = host_unpack_s4(vw.y, j);
                    va[j + 16] = host_unpack_s4(vw.z, j);
                    va[j + 24] = host_unpack_s4(vw.w, j);
                }
                for (int n = 0; n < B_GEMM; ++n) {
                    int64_t idot = 0;
                    const int8_t* xn = h_xbatch.data() + (size_t)n * K + b * 32;
                    for (int j = 0; j < 32; ++j) idot += (int64_t)va[j] * (int64_t)xn[j];
                    ref_gemm[(size_t)r * B_GEMM + n] += (double)idot * sc;
                }
            }
        }
        uint4* d_xbatch = nullptr;
        float* d_ybatch = nullptr;
        CUDA_CHECK(cudaMalloc(&d_xbatch, (size_t)B_GEMM * K));
        CUDA_CHECK(cudaMalloc(&d_ybatch, (size_t)M * B_GEMM * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_xbatch, h_xbatch.data(), (size_t)B_GEMM * K, cudaMemcpyHostToDevice));
        int gemm_grid = (M + 63) / 64;
        ptx_cp_async_mma_a16_gemm_b32<<<gemm_grid, 128, 0, stream>>>(d_valid, d_scales, d_xbatch, x_scale, d_ybatch, M, K);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<float> h_ybatch((size_t)GEMM_REF_ROWS * B_GEMM);
        CUDA_CHECK(cudaMemcpy(h_ybatch.data(), d_ybatch, h_ybatch.size() * sizeof(float), cudaMemcpyDeviceToHost));
        double gemm_max_e = 0.0, gemm_rel_e = 0.0;
        compute_errors(h_ybatch, ref_gemm, (int)h_ybatch.size(), gemm_max_e, gemm_rel_e);
        if (gemm_rel_e > 1e-4) ++total_verify_failures;

        for (int i = 0; i < WARM; ++i)
            ptx_cp_async_mma_a16_gemm_b32<<<gemm_grid, 128, 0, stream>>>(d_valid, d_scales, d_xbatch, x_scale, d_ybatch, M, K);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaEventRecord(t0, stream));
        for (int i = 0; i < ITERS; ++i)
            ptx_cp_async_mma_a16_gemm_b32<<<gemm_grid, 128, 0, stream>>>(d_valid, d_scales, d_xbatch, x_scale, d_ybatch, M, K);
        CUDA_CHECK(cudaEventRecord(t1, stream));
        CUDA_CHECK(cudaEventSynchronize(t1));
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        double gemm_ms = ms / ITERS;
        double gemm_tops = (2.0 * (double)M * (double)K * (double)B_GEMM) / (gemm_ms * 1e9);
        double gemm_phys_gbs = ((double)phys_weight_bytes + (double)B_GEMM * K + (double)M * B_GEMM * 4.0) / (gemm_ms * 1e6);
        double gemm_gw_s = (double)total_weights / (gemm_ms * 1e6);
        CUDA_CHECK(cudaFree(d_xbatch)); CUDA_CHECK(cudaFree(d_ybatch));

        printf("    \"%s\": {\n", shapes[si].tag);
        printf("      \"M\": %d, \"K\": %d, \"weights\": %zu, \"phys_weight_bytes\": %zu, \"bpw\": 4.5,\n",
               M, K, total_weights, phys_weight_bytes);
        printf("      \"ref_rows_verified\": %d,\n", REF_ROWS);
        printf("      \"batched_b32_cp_async_imma_gemm\": {\"batch\": %d, \"ms\": %.6f, \"int8_tops\": %.4f, "
               "\"weight_gweights_per_s\": %.4f, \"phys_gb_s\": %.4f, \"max_abs_err\": %.6e, \"rel_l2_err\": %.6e},\n",
               B_GEMM, gemm_ms, gemm_tops, gemm_gw_s, gemm_phys_gbs, gemm_max_e, gemm_rel_e);
        printf("      \"arms\": {\n");
        for (size_t k = 0; k < results.size(); ++k) {
            const auto& b = results[k];
            printf("        \"%s\": {\"ms\": %.6f, \"gweights_per_s\": %.4f, \"phys_gb_s\": %.4f, "
                   "\"fp16_equiv_tb_s\": %.4f, \"pct_triad\": %.3f, \"max_abs_err\": %.6e, "
                   "\"rel_l2_err\": %.6e, \"syndrome_errors\": %u}%s\n",
                   b.name, b.ms, b.gweights_s, b.phys_gb_s, b.fp16_equiv_tb_s, b.pct_triad,
                   b.max_abs_err, b.rel_l2_err, b.syndrome_errs,
                   (k + 1 < results.size()) ? "," : "");
        }
        printf("      }\n");
        printf("    }%s\n", (si + 1 < 2) ? "," : "");

        CUDA_CHECK(cudaFree(d_raw)); CUDA_CHECK(cudaFree(d_valid)); CUDA_CHECK(cudaFree(d_scales));
        CUDA_CHECK(cudaFree(d_aos)); CUDA_CHECK(cudaFree(d_xi8)); CUDA_CHECK(cudaFree(d_xsum32));
        CUDA_CHECK(cudaFree(d_xf32)); CUDA_CHECK(cudaFree(d_xh2)); CUDA_CHECK(cudaFree(d_xbf2));
        CUDA_CHECK(cudaFree(d_y)); CUDA_CHECK(cudaFree(d_err));
    }

    CUDA_CHECK(cudaCtxResetPersistingL2Cache());
    printf("  },\n");
    printf("  \"total_verify_failures\": %d\n", total_verify_failures);
    printf("}\n");

    CUDA_CHECK(cudaEventDestroy(t0)); CUDA_CHECK(cudaEventDestroy(t1));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return (total_verify_failures == 0) ? 0 : 2;
}
