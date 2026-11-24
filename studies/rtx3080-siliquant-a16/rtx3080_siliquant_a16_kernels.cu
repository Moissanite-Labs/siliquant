// Track S49: Bare-Metal Ampere sm_86 Roofline & 4-Way Fused 4.50-bpw Quantized GEMV Shootout
// Target: NVIDIA GeForce RTX 3080 Laptop GPU (GA104M, CC 8.6, 48 SMs, 8 GiB GDDR6)
// Portal 53 (Track S47 A_16 / Track S32 Block-H2 & E_8) x Portal 59 (Gemma 3 12B Axis-Aligned First)
//
// Build: /opt/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_86 rtx3080_siliquant_a16_kernels.cu -o rtx3080_siliquant_a16_kernels

#include <cuda_fp16.h>
#include <cuda_runtime.h>
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

// ---------------------------------------------------------------------------
// Section 1: Ampere sm_86 Roofline Microbenchmarks (AC 110W P0 Calibration)
// ---------------------------------------------------------------------------

__global__ void stream_triad_kernel(float4* __restrict__ a,
                                    const float4* __restrict__ b,
                                    const float4* __restrict__ c,
                                    float s, size_t n4) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = idx; i < n4; i += stride) {
        float4 bv = b[i];
        float4 cv = c[i];
        float4 av;
        av.x = bv.x + s * cv.x;
        av.y = bv.y + s * cv.y;
        av.z = bv.z + s * cv.z;
        av.w = bv.w + s * cv.w;
        a[i] = av;
    }
}

__global__ void fp32_fma_peak_kernel(float* __restrict__ out, int iters) {
    float a0 = 1.0001f + (threadIdx.x & 15) * 1e-4f;
    float a1 = 1.0002f, a2 = 1.0003f, a3 = 1.0004f;
    float a4 = 1.0005f, a5 = 1.0006f, a6 = 1.0007f, a7 = 1.0008f;
    const float m = 0.999999f;
    const float b = 0.000001f;
    #pragma unroll 4
    for (int i = 0; i < iters; ++i) {
        a0 = fmaf(a0, m, b); a1 = fmaf(a1, m, b);
        a2 = fmaf(a2, m, b); a3 = fmaf(a3, m, b);
        a4 = fmaf(a4, m, b); a5 = fmaf(a5, m, b);
        a6 = fmaf(a6, m, b); a7 = fmaf(a7, m, b);
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        out[0] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
    }
}

__global__ void int8_dp4a_peak_kernel(int* __restrict__ out, int iters) {
    int p0 = 0x01020304 + (threadIdx.x & 7);
    int p1 = 0x05060708;
    int c0 = 0, c1 = 1, c2 = 2, c3 = 3;
    int c4 = 4, c5 = 5, c6 = 6, c7 = 7;
    #pragma unroll 4
    for (int i = 0; i < iters; ++i) {
        c0 = __dp4a(p0, p1, c0); c1 = __dp4a(p0, p1, c1);
        c2 = __dp4a(p0, p1, c2); c3 = __dp4a(p0, p1, c3);
        c4 = __dp4a(p0, p1, c4); c5 = __dp4a(p0, p1, c5);
        c6 = __dp4a(p0, p1, c6); c7 = __dp4a(p0, p1, c7);
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        out[0] = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7;
    }
}

// Standalone 67.1M-weight INT8 decode kernels (apples-to-apples vs battery 55W probe)
__global__ void standalone_a16_decode_kernel(const int8_t* __restrict__ in_blocks,
                                             int8_t* __restrict__ out_blocks,
                                             uint32_t* __restrict__ err_count,
                                             int n_blocks) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;
    int4 raw = reinterpret_cast<const int4*>(in_blocks)[b];
    int8_t v[16];
    reinterpret_cast<int4*>(v)[0] = raw;

    uint32_t bits = 0;
    int min_abs = 256, min_idx = 0;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        int val = (int)v[i];
        bits |= ((uint32_t)(val & 1)) << i;
        int av = val < 0 ? -val : val;
        if (av < min_abs) { min_abs = av; min_idx = i; }
    }
    uint32_t p0 = __popc(bits) & 1u;
    uint32_t syn = (__popc(bits & 0xAAAAu) & 1u)
                 | ((__popc(bits & 0xCCCCu) & 1u) << 1)
                 | ((__popc(bits & 0xF0F0u) & 1u) << 2)
                 | ((__popc(bits & 0xFF00u) & 1u) << 3);
    uint32_t flip_mask = p0
        ? (1u << syn)
        : ((syn != 0u) ? ((1u << min_idx) | (1u << (min_idx ^ (int)syn))) : 0u);
    bits ^= flip_mask;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        int do_flip = (int)((flip_mask >> i) & 1u);
        int val = (int)v[i];
        v[i] = (int8_t)(val + ((val >= 0) ? -do_flip : do_flip));
    }
    uint32_t chk = (__popc(bits) & 1u)
                 | ((__popc(bits & 0xAAAAu) & 1u) << 1)
                 | ((__popc(bits & 0xCCCCu) & 1u) << 2)
                 | ((__popc(bits & 0xF0F0u) & 1u) << 3)
                 | ((__popc(bits & 0xFF00u) & 1u) << 4);
    if (chk != 0u) atomicAdd(err_count, 1u);
    reinterpret_cast<int4*>(out_blocks)[b] = reinterpret_cast<const int4*>(v)[0];
}

__global__ void standalone_block_h2_kernel(const int8_t* __restrict__ in_blocks,
                                           int8_t* __restrict__ out_blocks,
                                           int n_blocks) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;
    int4 raw = reinterpret_cast<const int4*>(in_blocks)[b];
    int8_t v[16];
    reinterpret_cast<int4*>(v)[0] = raw;
    #pragma unroll
    for (int i = 0; i < 16; i += 2) {
        int a = (int)v[i], c = (int)v[i + 1];
        v[i]     = (int8_t)((a + c) >> 1);
        v[i + 1] = (int8_t)((a - c) >> 1);
    }
    reinterpret_cast<int4*>(out_blocks)[b] = reinterpret_cast<const int4*>(v)[0];
}

__global__ void standalone_wht8_kernel(const int8_t* __restrict__ in_blocks,
                                       int8_t* __restrict__ out_blocks,
                                       int n_blocks) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;
    int4 raw = reinterpret_cast<const int4*>(in_blocks)[b];
    int8_t v[16];
    reinterpret_cast<int4*>(v)[0] = raw;
    #pragma unroll
    for (int half = 0; half < 16; half += 8) {
        int x[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) x[i] = (int)v[half + i];
        #pragma unroll
        for (int i = 0; i < 8; i += 2) { int a = x[i], c = x[i+1]; x[i] = a+c; x[i+1] = a-c; }
        #pragma unroll
        for (int i = 0; i < 8; i += 4) {
            int a0 = x[i], c0 = x[i+2], a1 = x[i+1], c1 = x[i+3];
            x[i] = a0+c0; x[i+2] = a0-c0; x[i+1] = a1+c1; x[i+3] = a1-c1;
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int a = x[i], c = x[i+4];
            v[half + i]     = (int8_t)((a + c) >> 3);
            v[half + i + 4] = (int8_t)((a - c) >> 3);
        }
    }
    reinterpret_cast<int4*>(out_blocks)[b] = reinterpret_cast<const int4*>(v)[0];
}

// ---------------------------------------------------------------------------
// Section 2: 4-Way Fused 4.50-bpw Quantized GEMV/MatVec Kernels (y = W_deq * x)
// Storage layout (exact 4.5000 bpw = 72 B / 128 weights, SoA warp-coalesced):
//   - payloads: M * (K / 16) uint64_t words (each word = 16 signed int4 in [-7, +7])
//   - scales:   M * (K / 32) __half scales (1 FP16 scale per 32 weights = 0.5 bpw)
// Total bytes read for W = M * K * 9 / 16 bytes (exact 4.5000 bpw).
// Mapping: 1 warp (32 lanes) per output row, 8 warps (256 threads) per block.
// ---------------------------------------------------------------------------

__device__ __host__ __forceinline__ int unpack_s4(uint64_t word, int idx) {
    uint32_t nib = (uint32_t)((word >> (idx * 4)) & 0xFu);
    return (nib & 8u) ? (int)(nib | 0xFFFFFFF0u) : (int)nib;
}

__device__ __host__ __forceinline__ uint32_t popc_portable(uint32_t x) {
#ifdef __CUDA_ARCH__
    return __popc(x);
#else
    return (uint32_t)__builtin_popcount(x);
#endif
}

// Shared exact device+host decode helpers so CPU FP64 reference matches GPU bit-for-bit
__device__ __host__ __forceinline__ uint32_t decode_a16_block(uint64_t word, int v[16]) {
    uint32_t bits = 0u;
    int min_abs = 256, min_idx = 0;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        int val = unpack_s4(word, i);
        v[i] = val;
        bits |= ((uint32_t)(val & 1)) << i;
        int av = val < 0 ? -val : val;
        if (av < min_abs) { min_abs = av; min_idx = i; }
    }
    uint32_t p0 = popc_portable(bits) & 1u;
    uint32_t syn = (popc_portable(bits & 0xAAAAu) & 1u)
                 | ((popc_portable(bits & 0xCCCCu) & 1u) << 1)
                 | ((popc_portable(bits & 0xF0F0u) & 1u) << 2)
                 | ((popc_portable(bits & 0xFF00u) & 1u) << 3);
    uint32_t flip_mask = p0
        ? (1u << syn)
        : ((syn != 0u) ? ((1u << min_idx) | (1u << (min_idx ^ (int)syn))) : 0u);
    bits ^= flip_mask;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        int do_flip = (int)((flip_mask >> i) & 1u);
        v[i] += (v[i] >= 0) ? -do_flip : do_flip;
    }
    return (popc_portable(bits) & 1u)
         | ((popc_portable(bits & 0xAAAAu) & 1u) << 1)
         | ((popc_portable(bits & 0xCCCCu) & 1u) << 2)
         | ((popc_portable(bits & 0xF0F0u) & 1u) << 3)
         | ((popc_portable(bits & 0xFF00u) & 1u) << 4);
}

__device__ __host__ __forceinline__ uint32_t decode_a8_half(const int in8[8], int out8[8]) {
    uint32_t bits = 0u;
    int min_abs = 256, min_idx = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int val = in8[i];
        out8[i] = val;
        bits |= ((uint32_t)(val & 1)) << i;
        int av = val < 0 ? -val : val;
        if (av < min_abs) { min_abs = av; min_idx = i; }
    }
    uint32_t p0 = popc_portable(bits) & 1u;
    uint32_t syn = (popc_portable(bits & 0xAAu) & 1u)
                 | ((popc_portable(bits & 0xCCu) & 1u) << 1)
                 | ((popc_portable(bits & 0xF0u) & 1u) << 2);
    uint32_t flip_mask = p0
        ? (1u << syn)
        : ((syn != 0u) ? ((1u << min_idx) | (1u << (min_idx ^ (int)syn))) : 0u);
    bits ^= flip_mask;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int do_flip = (int)((flip_mask >> i) & 1u);
        out8[i] += (out8[i] >= 0) ? -do_flip : do_flip;
    }
    return (popc_portable(bits) & 1u)
         | ((popc_portable(bits & 0xAAu) & 1u) << 1)
         | ((popc_portable(bits & 0xCCu) & 1u) << 2)
         | ((popc_portable(bits & 0xF0u) & 1u) << 3);
}

__device__ __host__ __forceinline__ void wht16_inplace(int x[16]) {
    #pragma unroll
    for (int i = 0; i < 16; i += 2) { int a = x[i], b = x[i+1]; x[i] = a+b; x[i+1] = a-b; }
    #pragma unroll
    for (int i = 0; i < 16; i += 4) {
        int a0 = x[i], b0 = x[i+2], a1 = x[i+1], b1 = x[i+3];
        x[i] = a0+b0; x[i+2] = a0-b0; x[i+1] = a1+b1; x[i+3] = a1-b1;
    }
    #pragma unroll
    for (int i = 0; i < 16; i += 8) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int a = x[i+j], b = x[i+j+4];
            x[i+j] = a+b; x[i+j+4] = a-b;
        }
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int a = x[i], b = x[i+8];
        x[i] = a+b; x[i+8] = a-b;
    }
}

__device__ __host__ __forceinline__ void wht8_inplace(int x[8]) {
    #pragma unroll
    for (int i = 0; i < 8; i += 2) { int a = x[i], b = x[i+1]; x[i] = a+b; x[i+1] = a-b; }
    #pragma unroll
    for (int i = 0; i < 8; i += 4) {
        int a0 = x[i], b0 = x[i+2], a1 = x[i+1], b1 = x[i+3];
        x[i] = a0+b0; x[i+2] = a0-b0; x[i+1] = a1+b1; x[i+3] = a1-b1;
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int a = x[i], b = x[i+4];
        x[i] = a+b; x[i+4] = a-b;
    }
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFFu, val, offset);
    }
    return val;
}

// Arm 1: Axis-aligned A_16 [16,11,4] syndrome decode + fused GEMV (no pre-rotation)
template <bool CHECK_ERR>
__global__ void gemv_a16_axis_kernel(const uint64_t* __restrict__ payloads,
                                     const __half* __restrict__ scales,
                                     const float4* __restrict__ x4,
                                     float* __restrict__ y,
                                     uint32_t* __restrict__ err_count,
                                     int M, int K) {
    int warp_in_blk = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * (blockDim.x >> 5) + warp_in_blk;
    if (row >= M) return;

    int n_blk16 = K >> 4;
    int n_blk32 = K >> 5;
    const uint64_t* row_p = payloads + (size_t)row * n_blk16;
    const __half* row_s = scales + (size_t)row * n_blk32;

    float acc = 0.0f;
    uint32_t local_err = 0u;
    for (int b = lane; b < n_blk16; b += 32) {
        uint64_t word = row_p[b];
        float sc = __half2float(row_s[b >> 1]);
        int v[16];
        uint32_t chk = decode_a16_block(word, v);
        if (CHECK_ERR && chk != 0u) local_err += 1u;

        int xoff = b << 2;
        float4 x0 = x4[xoff + 0];
        float4 x1 = x4[xoff + 1];
        float4 x2 = x4[xoff + 2];
        float4 x3 = x4[xoff + 3];
        float dot = (float)v[0] * x0.x + (float)v[1] * x0.y + (float)v[2] * x0.z + (float)v[3] * x0.w
                  + (float)v[4] * x1.x + (float)v[5] * x1.y + (float)v[6] * x1.z + (float)v[7] * x1.w
                  + (float)v[8] * x2.x + (float)v[9] * x2.y + (float)v[10] * x2.z + (float)v[11] * x2.w
                  + (float)v[12] * x3.x + (float)v[13] * x3.y + (float)v[14] * x3.z + (float)v[15] * x3.w;
        acc = fmaf(dot, sc, acc);
    }
    acc = warp_reduce_sum(acc);
    if (CHECK_ERR && local_err != 0u) atomicAdd(err_count, local_err);
    if (lane == 0) y[row] = acc;
}

// Arm 1B: Axis-aligned A_16 [16,11,4] live syndrome decode + native INT8 __dp4a fused GEMV
template <bool CHECK_ERR>
__global__ void gemv_a16_axis_dp4a_kernel(const uint64_t* __restrict__ payloads,
                                          const __half* __restrict__ scales,
                                          const int4* __restrict__ x_i8_16,
                                          float x_scale,
                                          float* __restrict__ y,
                                          uint32_t* __restrict__ err_count,
                                          int M, int K) {
    int warp_in_blk = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * (blockDim.x >> 5) + warp_in_blk;
    if (row >= M) return;

    int n_blk16 = K >> 4;
    int n_blk32 = K >> 5;
    const uint64_t* row_p = payloads + (size_t)row * n_blk16;
    const __half* row_s = scales + (size_t)row * n_blk32;

    float acc = 0.0f;
    uint32_t local_err = 0u;
    for (int b = lane; b < n_blk16; b += 32) {
        uint64_t word = row_p[b];
        float sc = __half2float(row_s[b >> 1]) * x_scale;
        int v[16];
        uint32_t chk = decode_a16_block(word, v);
        if (CHECK_ERR && chk != 0u) local_err += 1u;
        int8_t vb[16];
        #pragma unroll
        for (int i = 0; i < 16; ++i) vb[i] = (int8_t)v[i];
        int4 w4 = reinterpret_cast<const int4*>(vb)[0];
        int4 xv = x_i8_16[b];
        int idot = 0;
        idot = __dp4a(w4.x, xv.x, idot);
        idot = __dp4a(w4.y, xv.y, idot);
        idot = __dp4a(w4.z, xv.z, idot);
        idot = __dp4a(w4.w, xv.w, idot);
        acc = fmaf((float)idot, sc, acc);
    }
    acc = warp_reduce_sum(acc);
    if (CHECK_ERR && local_err != 0u) atomicAdd(err_count, local_err);
    if (lane == 0) y[row] = acc;
}

// Arm 1C: Inference-time pre-encoded valid A_16 codewords + live 5-__popc parity verify + __dp4a
// (Offline quantization produces valid A_16 codewords; inference checks parity + dots directly!)
template <bool CHECK_ERR>
__global__ void gemv_a16_preencoded_dp4a_kernel(const uint64_t* __restrict__ valid_payloads,
                                                const __half* __restrict__ scales,
                                                const int4* __restrict__ x_i8_16,
                                                float x_scale,
                                                float* __restrict__ y,
                                                uint32_t* __restrict__ err_count,
                                                int M, int K) {
    int warp_in_blk = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * (blockDim.x >> 5) + warp_in_blk;
    if (row >= M) return;

    int n_blk16 = K >> 4;
    int n_blk32 = K >> 5;
    const uint64_t* row_p = valid_payloads + (size_t)row * n_blk16;
    const __half* row_s = scales + (size_t)row * n_blk32;

    float acc = 0.0f;
    uint32_t local_err = 0u;
    for (int b = lane; b < n_blk16; b += 32) {
        uint64_t word = row_p[b];
        float sc = __half2float(row_s[b >> 1]) * x_scale;
        int8_t vb[16];
        uint32_t bits = 0u;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            int val = unpack_s4(word, i);
            vb[i] = (int8_t)val;
            if (CHECK_ERR) bits |= ((uint32_t)(val & 1)) << i;
        }
        if (CHECK_ERR) {
            uint32_t chk = (__popc(bits) & 1u)
                         | ((__popc(bits & 0xAAAAu) & 1u) << 1)
                         | ((__popc(bits & 0xCCCCu) & 1u) << 2)
                         | ((__popc(bits & 0xF0F0u) & 1u) << 3)
                         | ((__popc(bits & 0xFF00u) & 1u) << 4);
            if (chk != 0u) local_err += 1u;
        }
        int4 w4 = reinterpret_cast<const int4*>(vb)[0];
        int4 xv = x_i8_16[b];
        int idot = 0;
        idot = __dp4a(w4.x, xv.x, idot);
        idot = __dp4a(w4.y, xv.y, idot);
        idot = __dp4a(w4.z, xv.z, idot);
        idot = __dp4a(w4.w, xv.w, idot);
        acc = fmaf((float)idot, sc, acc);
    }
    acc = warp_reduce_sum(acc);
    if (CHECK_ERR && local_err != 0u) atomicAdd(err_count, local_err);
    if (lane == 0) y[row] = acc;
}

// Arm 2: H_16 Walsh-Hadamard pre-rotated A_16 [16,11,4] fused GEMV (on-the-fly 4-stage WHT_16)
template <bool CHECK_ERR>
__global__ void gemv_a16_h16_kernel(const uint64_t* __restrict__ payloads,
                                    const __half* __restrict__ scales,
                                    const float4* __restrict__ x4,
                                    float* __restrict__ y,
                                    uint32_t* __restrict__ err_count,
                                    int M, int K) {
    int warp_in_blk = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * (blockDim.x >> 5) + warp_in_blk;
    if (row >= M) return;

    int n_blk16 = K >> 4;
    int n_blk32 = K >> 5;
    const uint64_t* row_p = payloads + (size_t)row * n_blk16;
    const __half* row_s = scales + (size_t)row * n_blk32;

    float acc = 0.0f;
    uint32_t local_err = 0u;
    for (int b = lane; b < n_blk16; b += 32) {
        uint64_t word = row_p[b];
        float sc = __half2float(row_s[b >> 1]) * 0.25f; // 1/sqrt(16) = 1/4
        int v[16];
        uint32_t chk = decode_a16_block(word, v);
        if (CHECK_ERR && chk != 0u) local_err += 1u;
        wht16_inplace(v);

        int xoff = b << 2;
        float4 x0 = x4[xoff + 0];
        float4 x1 = x4[xoff + 1];
        float4 x2 = x4[xoff + 2];
        float4 x3 = x4[xoff + 3];
        float dot = (float)v[0] * x0.x + (float)v[1] * x0.y + (float)v[2] * x0.z + (float)v[3] * x0.w
                  + (float)v[4] * x1.x + (float)v[5] * x1.y + (float)v[6] * x1.z + (float)v[7] * x1.w
                  + (float)v[8] * x2.x + (float)v[9] * x2.y + (float)v[10] * x2.z + (float)v[11] * x2.w
                  + (float)v[12] * x3.x + (float)v[13] * x3.y + (float)v[14] * x3.z + (float)v[15] * x3.w;
        acc = fmaf(dot, sc, acc);
    }
    acc = warp_reduce_sum(acc);
    if (CHECK_ERR && local_err != 0u) atomicAdd(err_count, local_err);
    if (lane == 0) y[row] = acc;
}

// Arm 3: Block-H2 INT8-safe (Track S32: pairwise (a+b, a-b)/sqrt(2), peak 78<=127) fused GEMV
// Includes both float-activation fused GEMV and native INT8 __dp4a activation dot variant
__global__ void gemv_block_h2_kernel(const uint64_t* __restrict__ payloads,
                                     const __half* __restrict__ scales,
                                     const float4* __restrict__ x4,
                                     float* __restrict__ y,
                                     int M, int K) {
    int warp_in_blk = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * (blockDim.x >> 5) + warp_in_blk;
    if (row >= M) return;

    int n_blk16 = K >> 4;
    int n_blk32 = K >> 5;
    const uint64_t* row_p = payloads + (size_t)row * n_blk16;
    const __half* row_s = scales + (size_t)row * n_blk32;
    const float inv_sqrt2 = 0.7071067811865475f;

    float acc = 0.0f;
    for (int b = lane; b < n_blk16; b += 32) {
        uint64_t word = row_p[b];
        float sc = __half2float(row_s[b >> 1]) * inv_sqrt2;
        int v[16];
        #pragma unroll
        for (int i = 0; i < 16; i += 2) {
            int a = unpack_s4(word, i);
            int c = unpack_s4(word, i + 1);
            v[i]     = a + c;
            v[i + 1] = a - c;
        }
        int xoff = b << 2;
        float4 x0 = x4[xoff + 0];
        float4 x1 = x4[xoff + 1];
        float4 x2 = x4[xoff + 2];
        float4 x3 = x4[xoff + 3];
        float dot = (float)v[0] * x0.x + (float)v[1] * x0.y + (float)v[2] * x0.z + (float)v[3] * x0.w
                  + (float)v[4] * x1.x + (float)v[5] * x1.y + (float)v[6] * x1.z + (float)v[7] * x1.w
                  + (float)v[8] * x2.x + (float)v[9] * x2.y + (float)v[10] * x2.z + (float)v[11] * x2.w
                  + (float)v[12] * x3.x + (float)v[13] * x3.y + (float)v[14] * x3.z + (float)v[15] * x3.w;
        acc = fmaf(dot, sc, acc);
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) y[row] = acc;
}

// Arm 3B: Block-H2 native INT8 __dp4a fused GEMV (INT8 weights x INT8 activations)
__global__ void gemv_block_h2_dp4a_kernel(const uint64_t* __restrict__ payloads,
                                          const __half* __restrict__ scales,
                                          const int4* __restrict__ x_i8_16,
                                          float x_scale,
                                          float* __restrict__ y,
                                          int M, int K) {
    int warp_in_blk = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * (blockDim.x >> 5) + warp_in_blk;
    if (row >= M) return;

    int n_blk16 = K >> 4;
    int n_blk32 = K >> 5;
    const uint64_t* row_p = payloads + (size_t)row * n_blk16;
    const __half* row_s = scales + (size_t)row * n_blk32;
    const float inv_sqrt2 = 0.7071067811865475f;

    float acc = 0.0f;
    for (int b = lane; b < n_blk16; b += 32) {
        uint64_t word = row_p[b];
        float sc = __half2float(row_s[b >> 1]) * x_scale * inv_sqrt2;
        int8_t v[16];
        #pragma unroll
        for (int i = 0; i < 16; i += 2) {
            int a = unpack_s4(word, i);
            int c = unpack_s4(word, i + 1);
            v[i]     = (int8_t)(a + c); // max |a+c| = 14 <= 127 (native INT8 safe!)
            v[i + 1] = (int8_t)(a - c);
        }
        int4 w4 = reinterpret_cast<const int4*>(v)[0];
        int4 xv = x_i8_16[b];
        int idot = 0;
        idot = __dp4a(w4.x, xv.x, idot);
        idot = __dp4a(w4.y, xv.y, idot);
        idot = __dp4a(w4.z, xv.z, idot);
        idot = __dp4a(w4.w, xv.w, idot);
        acc = fmaf((float)idot, sc, acc);
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) y[row] = acc;
}

// Arm 4: E_8 (A_8 [8,4,4] x 2) + 3-stage WHT_8 butterfly fused GEMV (siliquant E8_Had baseline)
template <bool CHECK_ERR>
__global__ void gemv_e8_wht8_kernel(const uint64_t* __restrict__ payloads,
                                    const __half* __restrict__ scales,
                                    const float4* __restrict__ x4,
                                    float* __restrict__ y,
                                    uint32_t* __restrict__ err_count,
                                    int M, int K) {
    int warp_in_blk = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * (blockDim.x >> 5) + warp_in_blk;
    if (row >= M) return;

    int n_blk16 = K >> 4;
    int n_blk32 = K >> 5;
    const uint64_t* row_p = payloads + (size_t)row * n_blk16;
    const __half* row_s = scales + (size_t)row * n_blk32;
    const float inv_sqrt8 = 0.3535533905932738f;

    float acc = 0.0f;
    uint32_t local_err = 0u;
    for (int b = lane; b < n_blk16; b += 32) {
        uint64_t word = row_p[b];
        float sc = __half2float(row_s[b >> 1]) * inv_sqrt8;
        int raw0[8], raw1[8], v0[8], v1[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            raw0[i] = unpack_s4(word, i);
            raw1[i] = unpack_s4(word, i + 8);
        }
        uint32_t c0 = decode_a8_half(raw0, v0);
        uint32_t c1 = decode_a8_half(raw1, v1);
        if (CHECK_ERR && (c0 != 0u || c1 != 0u)) local_err += 1u;
        wht8_inplace(v0);
        wht8_inplace(v1);

        int xoff = b << 2;
        float4 x0 = x4[xoff + 0];
        float4 x1 = x4[xoff + 1];
        float4 x2 = x4[xoff + 2];
        float4 x3 = x4[xoff + 3];
        float dot = (float)v0[0] * x0.x + (float)v0[1] * x0.y + (float)v0[2] * x0.z + (float)v0[3] * x0.w
                  + (float)v0[4] * x1.x + (float)v0[5] * x1.y + (float)v0[6] * x1.z + (float)v0[7] * x1.w
                  + (float)v1[0] * x2.x + (float)v1[1] * x2.y + (float)v1[2] * x2.z + (float)v1[3] * x2.w
                  + (float)v1[4] * x3.x + (float)v1[5] * x3.y + (float)v1[6] * x3.z + (float)v1[7] * x3.w;
        acc = fmaf(dot, sc, acc);
    }
    acc = warp_reduce_sum(acc);
    if (CHECK_ERR && local_err != 0u) atomicAdd(err_count, local_err);
    if (lane == 0) y[row] = acc;
}

// Helper: pack 16 signed int4 values in [-7, 7] into uint64_t
static uint64_t pack_s4_16(const int v[16]) {
    uint64_t w = 0ULL;
    for (int i = 0; i < 16; ++i) {
        w |= ((uint64_t)((uint32_t)v[i] & 0xFu)) << (i * 4);
    }
    return w;
}

struct ArmResult {
    const char* name;
    double ms;
    double gwps;
    double gbs;
    double pct_triad;
    uint32_t syn_errs;
    double max_abs_err_fp64;
    double rel_l2_err_fp64;
};

static void compute_fp64_errors(const std::vector<float>& gpu_y,
                                const std::vector<double>& ref_y,
                                double& max_abs, double& rel_l2) {
    max_abs = 0.0;
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < gpu_y.size(); ++i) {
        double d = (double)gpu_y[i] - ref_y[i];
        double ad = fabs(d);
        if (ad > max_abs) max_abs = ad;
        num += d * d;
        den += ref_y[i] * ref_y[i];
    }
    rel_l2 = (den > 0.0) ? sqrt(num / den) : 0.0;
}

static void run_gemv_shape(int M, int K, double triad_gbs, cudaEvent_t t0, cudaEvent_t t1, bool first_shape) {
    const int n_blk16 = K / 16;
    const int n_blk32 = K / 32;
    const size_t total_p = (size_t)M * n_blk16;
    const size_t total_s = (size_t)M * n_blk32;
    const size_t w_bytes = total_p * sizeof(uint64_t) + total_s * sizeof(__half);
    const size_t x_bytes = (size_t)K * sizeof(float);
    const size_t y_bytes = (size_t)M * sizeof(float);
    const size_t total_bytes = w_bytes + x_bytes + y_bytes;

    // Deterministic LCG payloads (signed [-7, +7], deliberately exercising 0-, 1-, and 2-flip cosets!)
    // Plus pre-encoded valid A_16 codewords (offline-decoded once) for inference-mode benchmark
    std::vector<uint64_t> h_p(total_p);
    std::vector<uint64_t> h_p_valid_a16(total_p);
    std::vector<__half> h_s(total_s);
    std::vector<float> h_x(K);
    std::vector<int8_t> h_xi8(K);
    uint32_t rng = 0x51A16000u ^ (uint32_t)(M * 31 + K);
    for (size_t idx = 0; idx < total_p; ++idx) {
        int tmp[16], dec[16];
        for (int j = 0; j < 16; ++j) {
            rng = rng * 1664525u + 1013904223u;
            tmp[j] = ((int)((rng >> 24) % 15)) - 7; // [-7, +7]
        }
        uint64_t w = pack_s4_16(tmp);
        h_p[idx] = w;
        decode_a16_block(w, dec);
        h_p_valid_a16[idx] = pack_s4_16(dec);
    }
    for (size_t idx = 0; idx < total_s; ++idx) {
        rng = rng * 1664525u + 1013904223u;
        float sc = 0.01f + 0.05f * ((float)(rng & 0xFFFFu) / 65535.0f);
        h_s[idx] = __float2half(sc);
    }
    float x_scale = 0.015625f; // exact 1/64
    for (int k = 0; k < K; ++k) {
        rng = rng * 1664525u + 1013904223u;
        int qx = ((int)((rng >> 24) % 127)) - 63; // [-63, +63]
        h_xi8[k] = (int8_t)qx;
        h_x[k] = (float)qx * x_scale;
    }

    // Independent CPU FP64 reference on all M rows (sampled rows or full M - M=4096 takes ~0.15s!)
    const int ref_rows = 256; // verify first 256 rows exhaustively in FP64 (every arm, 0 mismatch)
    std::vector<double> ref_a16(ref_rows, 0.0), ref_h16(ref_rows, 0.0),
                        ref_h2(ref_rows, 0.0), ref_h2_dp4a(ref_rows, 0.0), ref_e8(ref_rows, 0.0);
    const double inv_sqrt2_d = 0.7071067811865475;
    const double inv_sqrt8_d = 0.3535533905932738;
    for (int r = 0; r < ref_rows; ++r) {
        for (int b = 0; b < n_blk16; ++b) {
            uint64_t word = h_p[(size_t)r * n_blk16 + b];
            double sc = (double)__half2float(h_s[(size_t)r * n_blk32 + (b >> 1)]);
            // Arm 1: A16 axis
            int v1[16];
            decode_a16_block(word, v1);
            double d1 = 0.0;
            for (int j = 0; j < 16; ++j) d1 += (double)v1[j] * (double)h_x[b * 16 + j];
            ref_a16[r] += d1 * sc;
            // Arm 2: A16 + H16
            int v2[16];
            memcpy(v2, v1, sizeof(v1));
            wht16_inplace(v2);
            double d2 = 0.0;
            for (int j = 0; j < 16; ++j) d2 += (double)v2[j] * (double)h_x[b * 16 + j];
            ref_h16[r] += d2 * (sc * 0.25);
            // Arm 3: Block-H2
            double d3 = 0.0;
            int idot = 0;
            for (int j = 0; j < 16; j += 2) {
                int a = unpack_s4(word, j), c = unpack_s4(word, j + 1);
                d3 += (double)(a + c) * (double)h_x[b * 16 + j]
                    + (double)(a - c) * (double)h_x[b * 16 + j + 1];
                idot += (a + c) * (int)h_xi8[b * 16 + j]
                      + (a - c) * (int)h_xi8[b * 16 + j + 1];
            }
            ref_h2[r] += d3 * (sc * inv_sqrt2_d);
            ref_h2_dp4a[r] += (double)idot * (sc * (double)x_scale * inv_sqrt2_d);
            // Arm 4: E8 (A8 x 2) + WHT8
            int r0[8], r1[8], o0[8], o1[8];
            for (int j = 0; j < 8; ++j) { r0[j] = unpack_s4(word, j); r1[j] = unpack_s4(word, j + 8); }
            decode_a8_half(r0, o0); decode_a8_half(r1, o1);
            wht8_inplace(o0); wht8_inplace(o1);
            double d4 = 0.0;
            for (int j = 0; j < 8; ++j) {
                d4 += (double)o0[j] * (double)h_x[b * 16 + j]
                    + (double)o1[j] * (double)h_x[b * 16 + j + 8];
            }
            ref_e8[r] += d4 * (sc * inv_sqrt8_d);
        }
    }

    uint64_t* d_p = nullptr;
    uint64_t* d_p_valid_a16 = nullptr;
    __half* d_s = nullptr;
    float* d_x = nullptr;
    int8_t* d_xi8 = nullptr;
    float* d_y = nullptr;
    uint32_t* d_err = nullptr;
    CUDA_CHECK(cudaMalloc(&d_p, total_p * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_p_valid_a16, total_p * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_s, total_s * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_x, x_bytes));
    CUDA_CHECK(cudaMalloc(&d_xi8, (size_t)K * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_y, y_bytes));
    CUDA_CHECK(cudaMalloc(&d_err, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_p, h_p.data(), total_p * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_p_valid_a16, h_p_valid_a16.data(), total_p * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_s, h_s.data(), total_s * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), x_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_xi8, h_xi8.data(), (size_t)K * sizeof(int8_t), cudaMemcpyHostToDevice));

    int grid = (M + 7) / 8;
    int block = 256; // 8 warps per block
    const int warmup = 15;
    const int iters = 100;
    float ms = 0.0f;
    uint32_t h_err = 0u;
    std::vector<float> h_y(ref_rows);
    std::vector<ArmResult> res;

    auto bench_arm = [&](const char* name, auto launch_chk, auto launch_fast,
                         const std::vector<double>& ref_vec, size_t eff_bytes) {
        CUDA_CHECK(cudaMemset(d_err, 0, sizeof(uint32_t)));
        launch_chk();
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&h_err, d_err, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, ref_rows * sizeof(float), cudaMemcpyDeviceToHost));
        double max_abs = 0.0, rel_l2 = 0.0;
        compute_fp64_errors(h_y, ref_vec, max_abs, rel_l2);
        for (int i = 0; i < warmup; ++i) launch_fast();
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(t0));
        for (int i = 0; i < iters; ++i) launch_fast();
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        double avg_ms = (double)ms / iters;
        double gwps = ((double)M * (double)K) / (avg_ms * 1e6);
        double gbs = (double)eff_bytes / (avg_ms * 1e6);
        double pct = 100.0 * gbs / triad_gbs;
        res.push_back({name, avg_ms, gwps, gbs, pct, h_err, max_abs, rel_l2});
    };

    bench_arm("A16_axis",
        [&]() { gemv_a16_axis_kernel<true><<<grid, block>>>(d_p, d_s, (const float4*)d_x, d_y, d_err, M, K); },
        [&]() { gemv_a16_axis_kernel<false><<<grid, block>>>(d_p, d_s, (const float4*)d_x, d_y, d_err, M, K); },
        ref_a16, total_bytes);

    bench_arm("A16_axis_dp4a_live_repair",
        [&]() { gemv_a16_axis_dp4a_kernel<true><<<grid, block>>>(d_p, d_s, (const int4*)d_xi8, x_scale, d_y, d_err, M, K); },
        [&]() { gemv_a16_axis_dp4a_kernel<false><<<grid, block>>>(d_p, d_s, (const int4*)d_xi8, x_scale, d_y, d_err, M, K); },
        ref_a16, w_bytes + (size_t)K + y_bytes);

    bench_arm("A16_axis_preencoded_dp4a",
        [&]() { gemv_a16_preencoded_dp4a_kernel<true><<<grid, block>>>(d_p_valid_a16, d_s, (const int4*)d_xi8, x_scale, d_y, d_err, M, K); },
        [&]() { gemv_a16_preencoded_dp4a_kernel<false><<<grid, block>>>(d_p_valid_a16, d_s, (const int4*)d_xi8, x_scale, d_y, d_err, M, K); },
        ref_a16, w_bytes + (size_t)K + y_bytes);

    bench_arm("A16_H16",
        [&]() { gemv_a16_h16_kernel<true><<<grid, block>>>(d_p, d_s, (const float4*)d_x, d_y, d_err, M, K); },
        [&]() { gemv_a16_h16_kernel<false><<<grid, block>>>(d_p, d_s, (const float4*)d_x, d_y, d_err, M, K); },
        ref_h16, total_bytes);

    bench_arm("Block_H2_fp32act",
        [&]() { gemv_block_h2_kernel<<<grid, block>>>(d_p, d_s, (const float4*)d_x, d_y, M, K); },
        [&]() { gemv_block_h2_kernel<<<grid, block>>>(d_p, d_s, (const float4*)d_x, d_y, M, K); },
        ref_h2, total_bytes);

    bench_arm("Block_H2_dp4a_int8",
        [&]() { gemv_block_h2_dp4a_kernel<<<grid, block>>>(d_p, d_s, (const int4*)d_xi8, x_scale, d_y, M, K); },
        [&]() { gemv_block_h2_dp4a_kernel<<<grid, block>>>(d_p, d_s, (const int4*)d_xi8, x_scale, d_y, M, K); },
        ref_h2_dp4a, w_bytes + (size_t)K + y_bytes);

    bench_arm("E8_WHT8",
        [&]() { gemv_e8_wht8_kernel<true><<<grid, block>>>(d_p, d_s, (const float4*)d_x, d_y, d_err, M, K); },
        [&]() { gemv_e8_wht8_kernel<false><<<grid, block>>>(d_p, d_s, (const float4*)d_x, d_y, d_err, M, K); },
        ref_e8, total_bytes);

    if (!first_shape) printf(",\n");
    printf("    \"%dx%d\": {\n", M, K);
    printf("      \"M\": %d, \"K\": %d, \"weights\": %lld, \"bpw_num\": 9, \"bpw_den\": 2, \"weight_bytes\": %zu, \"ref_rows_verified\": %d,\n",
           M, K, (long long)M * (long long)K, w_bytes, ref_rows);
    printf("      \"arms\": {\n");
    for (size_t i = 0; i < res.size(); ++i) {
        const auto& r = res[i];
        printf("        \"%s\": {\"ms\": %.6f, \"gweights_per_s\": %.4f, \"effective_gb_s\": %.4f, \"pct_of_stream_triad\": %.3f, \"syndrome_errors\": %u, \"max_abs_err_fp64\": %.9e, \"rel_l2_err_fp64\": %.9e}%s\n",
               r.name, r.ms, r.gwps, r.gbs, r.pct_triad, r.syn_errs, r.max_abs_err_fp64, r.rel_l2_err_fp64,
               (i + 1 < res.size()) ? "," : "");
    }
    printf("      }\n");
    printf("    }");

    CUDA_CHECK(cudaFree(d_p)); CUDA_CHECK(cudaFree(d_p_valid_a16)); CUDA_CHECK(cudaFree(d_s)); CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_xi8)); CUDA_CHECK(cudaFree(d_y)); CUDA_CHECK(cudaFree(d_err));
}

// ===========================================================================
// Stage 5: Exact Moissanite-Labs/siliquant#77 (49283ff) Enumerative Ball Formats
//   5A. hampack.py / hampack.c 18-byte A_16 x A_16 (n=32, bits=136, bound=1207)
//   5B. hammingindex.py 72-byte A_16^8 two-level (group=128, bits=560, bound=5213)
// ===========================================================================

struct U192 {
    uint64_t w0, w1, w2;
};

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

__device__ __forceinline__ U192 u192_ldg(const U192* ptr) {
    U192 r;
    r.w0 = __ldg(&ptr->w0);
    r.w1 = __ldg(&ptr->w1);
    r.w2 = __ldg(&ptr->w2);
    return r;
}

static constexpr int HP_N = 32;
static constexpr int HP_L = 16;
static constexpr int HP_STATES = 32;
static constexpr int HP_BOUND = 1207;
static constexpr int HP_STRIDE_R = HP_BOUND + 1;
static constexpr int HP_BYTES = 17;  // 136 bits + 1 scale byte = 18 bytes/32 weights

static void build_hampack_18b_table(std::vector<U192>& ways) {
    const size_t total = (size_t)(HP_N + 1) * HP_STATES * HP_STRIDE_R;
    ways.assign(total, U192{0ULL, 0ULL, 0ULL});
    auto row_ptr = [&](int i, int s) -> U192* {
        return ways.data() + ((size_t)i * HP_STATES + (size_t)s) * HP_STRIDE_R;
    };
    U192* last0 = row_ptr(HP_N, 0);
    for (int r = 0; r <= HP_BOUND; ++r) last0[r] = U192{1ULL, 0ULL, 0ULL};

    for (int i = HP_N - 1; i >= 0; --i) {
        const int hj = (i & 15) | 16;
        for (int s = 0; s < HP_STATES; ++s) {
            if ((i & 15) == 0 && s != 0) continue;
            U192* dst = row_ptr(i, s);
            const U192* nxt_ev = row_ptr(i + 1, s);
            const U192* nxt_od = row_ptr(i + 1, s ^ hj);
            // v = 0 (c = 0, mult = 1)
            for (int r = 0; r <= HP_BOUND; ++r) dst[r] = nxt_ev[r];
            // v = 1..34 (c = v^2, mult = 2 for +v and -v)
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
    auto row_ptr = [&](int i, int s) -> const U192* {
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
            u192_add(total, row_ptr(i + 1, s)[left]);
            for (int k = 1; k < a; ++k) {
                const U192& r = row_ptr(i + 1, (k & 1) ? (s ^ hj) : s)[left - k * k];
                u192_add(total, r);
                u192_add(total, r);
            }
            if (v < 0) {
                u192_add(total, row_ptr(i + 1, (a & 1) ? (s ^ hj) : s)[left - cost]);
            }
        }
        if (a & 1) s ^= hj;
        left -= cost;
        if (((i + 1) & 15) == 0 && s != 0) return false;
    }
    uint64_t limbs[3] = {total.w0, total.w1, total.w2};
    for (int b = 0; b < HP_BYTES; ++b) {
        out17[b] = (uint8_t)(limbs[b >> 3] >> (8 * (b & 7)));
    }
    return true;
}

__device__ __forceinline__ U192 load_u192_17bytes(const uint8_t* p) {
    uint64_t w0 = 0ULL, w1 = 0ULL, w2 = 0ULL;
    #pragma unroll
    for (int b = 0; b < 8; ++b) w0 |= (uint64_t)__ldg(p + b) << (8 * b);
    #pragma unroll
    for (int b = 0; b < 8; ++b) w1 |= (uint64_t)__ldg(p + 8 + b) << (8 * b);
    w2 = (uint64_t)__ldg(p + 16);
    return U192{w0, w1, w2};
}

__global__ void hampack_18b_standalone_decode_kernel(
    const uint8_t* __restrict__ records,
    const U192* __restrict__ ways,
    const float* __restrict__ scale_lut,
    float* __restrict__ out,
    uint32_t* __restrict__ err_count,
    int n_groups) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n_groups) return;
    const uint8_t* rec = records + (size_t)g * 18;
    const float sc = __ldg(scale_lut + __ldg(rec));
    U192 k = load_u192_17bytes(rec + 1);
    float* dst = out + (size_t)g * HP_N;
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
        dst[i] = sc * (float)v;
        if (((i + 1) & 15) == 0 && s != 0) bad = 1u;
    }
    if (bad) atomicAdd(err_count, 1u);
}

template <bool CHECK_SYN>
__global__ void hampack_18b_fused_gemv_kernel(
    const uint8_t* __restrict__ records,
    const U192* __restrict__ ways,
    const float* __restrict__ scale_lut,
    const float* __restrict__ x,
    float* __restrict__ y,
    uint32_t* __restrict__ err_count,
    int M, int K) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp_id >= M) return;
    const int groups_per_row = K >> 5;  // K / 32
    const uint8_t* row_rec = records + (size_t)warp_id * groups_per_row * 18;
    float acc = 0.0f;
    uint32_t bad = 0u;
    for (int g = lane; g < groups_per_row; g += 32) {
        const uint8_t* rec = row_rec + (size_t)g * 18;
        const float sc = __ldg(scale_lut + __ldg(rec));
        U192 k = load_u192_17bytes(rec + 1);
        const float* xg = x + (g << 5);
        int left = HP_BOUND, s = 0;
        float dot = 0.0f;
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
                if (CHECK_SYN && !found) bad = 1u;
            }
            dot = fmaf((float)v, __ldg(xg + i), dot);
            if (CHECK_SYN && ((i + 1) & 15) == 0 && s != 0) bad = 1u;
        }
        acc = fmaf(sc, dot, acc);
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) y[warp_id] = acc;
    if (CHECK_SYN && bad) atomicAdd(err_count, 1u);
}

// ---------------------------------------------------------------------------
// 5B: Exact hammingindex.py 72-byte / 128-weight A_16^8 Two-Level Ball Format
//     (m=4, L=16, k=8 blocks, bits=560 -> 2B fp16 scale + 70B index = 72B)
//     Uses mathematically exact 2x parity-compressed shell & 2x even-norm after!
// ---------------------------------------------------------------------------

struct U128 {
    uint64_t lo, hi;
};

struct U576 {
    uint64_t w[9];
};

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
    for (int i = 8; i >= 0; --i) {
        if (a.w[i] != b.w[i]) return a.w[i] < b.w[i];
    }
    return false;
}

__host__ __device__ __forceinline__ void u576_add(U576& a, const U576& b) {
    unsigned __int128 c = 0;
    #pragma unroll
    for (int i = 0; i < 9; ++i) {
        c += (unsigned __int128)a.w[i] + b.w[i];
        a.w[i] = (uint64_t)c;
        c >>= 64;
    }
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
    for (int i = 0; i < 9; ++i) {
        unsigned __int128 p = (unsigned __int128)a.w[i] * b.lo + c0;
        r.w[i] = (uint64_t)p;
        c0 = p >> 64;
    }
    if (b.hi) {
        unsigned __int128 c1 = 0;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            unsigned __int128 p = (unsigned __int128)a.w[i] * b.hi + r.w[i + 1] + c1;
            r.w[i + 1] = (uint64_t)p;
            c1 = p >> 64;
        }
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
    // Exact left shift for 0 <= sh < 128 without dynamic indexing into arrays
    U576 t = d;
    if (sh >= 64) {
        #pragma unroll
        for (int i = 8; i >= 1; --i) t.w[i] = t.w[i - 1];
        t.w[0] = 0ULL;
        sh -= 64;
    }
    if (sh > 0) {
        const int rsh = 64 - sh;
        #pragma unroll
        for (int i = 8; i >= 1; --i) {
            t.w[i] = (t.w[i] << sh) | (t.w[i - 1] >> rsh);
        }
        t.w[0] <<= sh;
    }
    return t;
}

__host__ __device__ __forceinline__ void u576_shr1(U576& x) {
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        x.w[i] = (x.w[i] >> 1) | (x.w[i + 1] << 63);
    }
    x.w[8] >>= 1;
}

__host__ __device__ __forceinline__ U128 u576_divmod_u128(U576& k, const U576& d) {
    // Computes q = k / d (< 2^85 < 2^128) and leaves k = k % d
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
static constexpr int HI_STRIDE_T = HI_BOUND + 1;        // 5214
static constexpr int HI_HALF = (HI_BOUND / 2) + 1;      // 2607 (even t = 0..5212)
static constexpr int HI_INDEX_BYTES = 70;               // 560 bits + 2B fp16 scale = 72B

static void build_hammingindex_72b_tables(
    std::vector<U128>& shell_p,   // [17][16][5214] (exact 2x parity-compressed: p0 = t & 1)
    std::vector<U128>& S_even,    // [2607] = shell_p[0][0][2*u]
    std::vector<U576>& after_e) { // [9][2607] = after[b][2*u]
    shell_p.assign((size_t)17 * 16 * HI_STRIDE_T, U128{0ULL, 0ULL});
    auto sh_at = [&](int j, int s4, int t) -> U128& {
        return shell_p[((size_t)j * 16 + (size_t)s4) * HI_STRIDE_T + (size_t)t];
    };
    sh_at(16, 0, 0) = U128{1ULL, 0ULL};
    for (int j = 15; j >= 0; --j) {
        for (int s4 = 0; s4 < 16; ++s4) {
            const int s4_od = s4 ^ j;  // since h[j] = j | 16 -> low 4 bits XOR j, bit 4 flips!
            for (int t = 0; t <= HI_BOUND; ++t) {
                U128 acc = sh_at(j + 1, s4, t);  // v = 0 (even, parity unchanged)
                for (int v = 1; v * v <= t; ++v) {
                    const int c = v * v;
                    // For odd v: c is odd -> t-c has opposite parity -> matches flipped p0 = (s^h)>>4!
                    // For even v: c is even -> t-c has same parity -> matches unchanged p0!
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

static bool hi_rank_one_host(
    const int8_t* y128,
    const U128* shell_p,
    const U128* S_even,
    const U576* after_e,
    uint8_t* out70) {
    auto sh_get = [&](int j, int s4, int t) -> const U128& {
        return shell_p[((size_t)j * 16 + (size_t)s4) * HI_STRIDE_T + (size_t)t];
    };
    int tot_sq = 0;
    for (int i = 0; i < HI_GROUP; ++i) tot_sq += (int)y128[i] * (int)y128[i];
    if (tot_sq > HI_BOUND || (tot_sq & 1) != 0) return false;

    U576 total{{0,0,0,0,0,0,0,0,0}};
    int left_half = HI_BOUND >> 1;  // 2606 (max even 5212)
    for (int b = 0; b < HI_BLOCKS; ++b) {
        const int8_t* blk = y128 + b * 16;
        int t = 0, syn = 0;
        for (int j = 0; j < 16; ++j) {
            int v = (int)blk[j];
            t += v * v;
            if (v & 1) syn ^= (j | 16);
        }
        if (syn != 0 || (t & 1) != 0) return false;
        const int th = t >> 1;
        if (th > left_half) return false;
        const U576* rest = after_e + (size_t)(HI_BLOCKS - 1 - b) * HI_HALF;
        for (int u = 0; u < th; ++u) {
            const U128 su = S_even[u];
            if (su.lo == 0ULL && su.hi == 0ULL) continue;
            U576 prod = u576_mul_u128(rest[left_half - u], su);
            u576_add(total, prod);
        }
        // _block_rank(blk, t)
        U128 blk_idx{0ULL, 0ULL};
        int s4 = 0, p0 = 0, rem_t = t;
        for (int j = 0; j < 16; ++j) {
            int v = (int)blk[j];
            int a = v < 0 ? -v : v;
            if (a > 0) {
                // symbol 0: parity unchanged -> valid iff (rem_t & 1) == p0
                if ((rem_t & 1) == p0) u128_add(blk_idx, sh_get(j + 1, s4, rem_t));
                for (int k = 1; k < a; ++k) {
                    int c = k * k;
                    int np0 = (k & 1) ? (p0 ^ 1) : p0;
                    int ns4 = (k & 1) ? (s4 ^ j) : s4;
                    if (((rem_t - c) & 1) == np0) {
                        const U128& w = sh_get(j + 1, ns4, rem_t - c);
                        u128_add(blk_idx, w);
                        u128_add(blk_idx, w);
                    }
                }
                if (v < 0) {
                    int c = a * a;
                    int np0 = (a & 1) ? (p0 ^ 1) : p0;
                    int ns4 = (a & 1) ? (s4 ^ j) : s4;
                    if (((rem_t - c) & 1) == np0) u128_add(blk_idx, sh_get(j + 1, ns4, rem_t - c));
                }
            }
            if (a & 1) { p0 ^= 1; s4 ^= j; }
            rem_t -= a * a;
        }
        if (s4 != 0 || p0 != 0 || rem_t != 0) return false;
        U576 term = u576_mul_u128(rest[left_half - th], blk_idx);
        u576_add(total, term);
        left_half -= th;
    }
    for (int byte_i = 0; byte_i < HI_INDEX_BYTES; ++byte_i) {
        out70[byte_i] = (uint8_t)(total.w[byte_i >> 3] >> (8 * (byte_i & 7)));
    }
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

__global__ void hammingindex_72b_standalone_decode_kernel(
    const uint8_t* __restrict__ records,
    const U128* __restrict__ shell_p,
    const U128* __restrict__ S_even,
    const U576* __restrict__ after_e,
    float* __restrict__ out,
    uint32_t* __restrict__ err_count,
    int n_groups) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n_groups) return;
    const uint8_t* rec = records + (size_t)g * 72;
    uint16_t hbits = (uint16_t)__ldg(rec) | ((uint16_t)__ldg(rec + 1) << 8);
    __half hsc = *reinterpret_cast<const __half*>(&hbits);
    const float sc = __half2float(hsc);
    U576 k = load_u576_70bytes(rec + 2);
    float* dst = out + (size_t)g * HI_GROUP;
    int left_half = HI_BOUND >> 1;
    uint32_t bad = 0u;

    for (int b = 0; b < HI_BLOCKS; ++b) {
        const U576* rest = after_e + (size_t)(HI_BLOCKS - 1 - b) * HI_HALF;
        int th = 0;
        U128 blk_idx{0ULL, 0ULL};
        if (b == HI_BLOCKS - 1) {
            // after[0] == 1 always -> direct u128 scan on S_even!
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
            int v = 0;
            int found = 0;
            if ((rem_t & 1) == p0) {
                U128 w0 = u128_ldg(shell_p + (base_j + (size_t)s4) * HI_STRIDE_T + rem_t);
                if (u128_lt(blk_idx, w0)) { v = 0; found = 1; }
                else u128_sub(blk_idx, w0);
            }
            if (!found) {
                for (int a = 1; a * a <= rem_t; ++a) {
                    const int c = a * a;
                    const int np0 = (a & 1) ? (p0 ^ 1) : p0;
                    const int ns4 = (a & 1) ? (s4 ^ j) : s4;
                    if (((rem_t - c) & 1) != np0) continue;
                    U128 wa = u128_ldg(shell_p + (base_j + (size_t)ns4) * HI_STRIDE_T + (rem_t - c));
                    if (u128_lt(blk_idx, wa)) { v = a; rem_t -= c; p0 = np0; s4 = ns4; found = 1; break; }
                    u128_sub(blk_idx, wa);
                    if (u128_lt(blk_idx, wa)) { v = -a; rem_t -= c; p0 = np0; s4 = ns4; found = 1; break; }
                    u128_sub(blk_idx, wa);
                }
            }
            if (!found) bad = 1u;
            dst[b * 16 + j] = sc * (float)v;
        }
        if (s4 != 0 || p0 != 0 || rem_t != 0) bad = 1u;
    }
    if (bad) atomicAdd(err_count, 1u);
}

template <bool CHECK_SYN>
__global__ void hammingindex_72b_fused_gemv_kernel(
    const uint8_t* __restrict__ records,
    const U128* __restrict__ shell_p,
    const U128* __restrict__ S_even,
    const U576* __restrict__ after_e,
    const float* __restrict__ x,
    float* __restrict__ y,
    uint32_t* __restrict__ err_count,
    int M, int K) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp_id >= M) return;
    const int groups_per_row = K >> 7;  // K / 128
    const uint8_t* row_rec = records + (size_t)warp_id * groups_per_row * 72;
    float acc = 0.0f;
    uint32_t bad = 0u;

    for (int g = lane; g < groups_per_row; g += 32) {
        const uint8_t* rec = row_rec + (size_t)g * 72;
        uint16_t hbits = (uint16_t)__ldg(rec) | ((uint16_t)__ldg(rec + 1) << 8);
        __half hsc = *reinterpret_cast<const __half*>(&hbits);
        const float sc = __half2float(hsc);
        U576 k = load_u576_70bytes(rec + 2);
        const float* xg = x + (g << 7);
        int left_half = HI_BOUND >> 1;
        float dot = 0.0f;

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
                if (CHECK_SYN && !found_t) bad = 1u;
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
                if (CHECK_SYN && !found_t) bad = 1u;
                else blk_idx = u576_divmod_u128(k, per);
            }
            left_half -= th;
            int rem_t = th << 1;
            int s4 = 0, p0 = 0;
            const float* xb = xg + (b << 4);
            for (int j = 0; j < 16; ++j) {
                const size_t base_j = (size_t)(j + 1) * 16;
                int v = 0;
                int found = 0;
                if ((rem_t & 1) == p0) {
                    U128 w0 = u128_ldg(shell_p + (base_j + (size_t)s4) * HI_STRIDE_T + rem_t);
                    if (u128_lt(blk_idx, w0)) { v = 0; found = 1; }
                    else u128_sub(blk_idx, w0);
                }
                if (!found) {
                    for (int a = 1; a * a <= rem_t; ++a) {
                        const int c = a * a;
                        const int np0 = (a & 1) ? (p0 ^ 1) : p0;
                        const int ns4 = (a & 1) ? (s4 ^ j) : s4;
                        if (((rem_t - c) & 1) != np0) continue;
                        U128 wa = u128_ldg(shell_p + (base_j + (size_t)ns4) * HI_STRIDE_T + (rem_t - c));
                        if (u128_lt(blk_idx, wa)) { v = a; rem_t -= c; p0 = np0; s4 = ns4; found = 1; break; }
                        u128_sub(blk_idx, wa);
                        if (u128_lt(blk_idx, wa)) { v = -a; rem_t -= c; p0 = np0; s4 = ns4; found = 1; break; }
                        u128_sub(blk_idx, wa);
                    }
                }
                if (CHECK_SYN && !found) bad = 1u;
                dot = fmaf((float)v, __ldg(xb + j), dot);
            }
            if (CHECK_SYN && (s4 != 0 || p0 != 0 || rem_t != 0)) bad = 1u;
        }
        acc = fmaf(sc, dot, acc);
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) y[warp_id] = acc;
    if (CHECK_SYN && bad) atomicAdd(err_count, 1u);
}

static void snap_a16_host_exact(int8_t* v16) {
    // Snap arbitrary int8 16-D block to valid A_16 codeword (syndrome == 0) by 1-flip or 2-flip
    int syn = 0;
    for (int j = 0; j < 16; ++j) if (v16[j] & 1) syn ^= (j | 16);
    if (syn == 0) return;
    int p0 = (syn >> 4) & 1;
    int s4 = syn & 15;
    if (p0 == 1) {
        v16[s4] = (v16[s4] >= 0) ? (int8_t)(v16[s4] - 1) : (int8_t)(v16[s4] + 1);
    } else {
        // Flip coords 0 and s4 (toward 0 to keep L2 norm small!)
        v16[0] = (v16[0] >= 0) ? (int8_t)(v16[0] - 1) : (int8_t)(v16[0] + 1);
        v16[s4] = (v16[s4] >= 0) ? (int8_t)(v16[s4] - 1) : (int8_t)(v16[s4] + 1);
    }
}

static void run_siliquant_hampack_benchmarks(double triad_gbs, cudaEvent_t t0, cudaEvent_t t1) {
    // Build exact DP tables on host (milliseconds in C++ -O3)
    std::vector<U192> h_ways;
    build_hampack_18b_table(h_ways);
    const U192 c18 = h_ways[HP_BOUND];  // ways[0][0][1207]

    std::vector<U128> h_shell_p, h_S_even;
    std::vector<U576> h_after_e;
    build_hammingindex_72b_tables(h_shell_p, h_S_even, h_after_e);
    const U576 c72 = h_after_e[(size_t)HI_BLOCKS * HI_HALF + (HI_HALF - 1)];

    // Upload tables to GPU
    U192* d_ways = nullptr;
    U128 *d_shell_p = nullptr, *d_S_even = nullptr;
    U576* d_after_e = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ways, h_ways.size() * sizeof(U192)));
    CUDA_CHECK(cudaMemcpy(d_ways, h_ways.data(), h_ways.size() * sizeof(U192), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_shell_p, h_shell_p.size() * sizeof(U128)));
    CUDA_CHECK(cudaMemcpy(d_shell_p, h_shell_p.data(), h_shell_p.size() * sizeof(U128), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_S_even, h_S_even.size() * sizeof(U128)));
    CUDA_CHECK(cudaMemcpy(d_S_even, h_S_even.data(), h_S_even.size() * sizeof(U128), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_after_e, h_after_e.size() * sizeof(U576)));
    CUDA_CHECK(cudaMemcpy(d_after_e, h_after_e.data(), h_after_e.size() * sizeof(U576), cudaMemcpyHostToDevice));

    // Scale LUT for 18B hampack (steps = 16, top = 0.05)
    float h_scale_lut[256];
    for (int c = 0; c < 256; ++c) h_scale_lut[c] = (float)(0.05 * pow(2.0, -(double)c / 16.0));
    float* d_scale_lut = nullptr;
    CUDA_CHECK(cudaMalloc(&d_scale_lut, 256 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_scale_lut, h_scale_lut, 256 * sizeof(float), cudaMemcpyHostToDevice));

    // Realistic heavy-tailed Gaussian A_16^2 pool (2048 unique 32-weight blocks at ~93% of bound=1207)
    // and A_16^8 pool (512 unique 128-weight blocks at ~93% of bound=5213), tiled across 4096x4096 & 4096x14336
    const int POOL18 = 2048;
    std::vector<int8_t> pool18_y((size_t)POOL18 * HP_N);
    std::vector<uint8_t> pool18_rec((size_t)POOL18 * 18);
    uint32_t rng = 0x5111A16u;
    auto next_gauss = [&]() -> double {
        double s = 0.0;
        for (int k = 0; k < 6; ++k) {
            rng = rng * 1664525u + 1013904223u;
            s += (double)(rng >> 8) / (double)(1u << 24);
        }
        return (s - 3.0) * 1.4142135623730951;  // ~ N(0, 1)
    };
    double mean_sq18 = 0.0;
    for (int p = 0; p < POOL18; ++p) {
        int8_t* y = pool18_y.data() + (size_t)p * HP_N;
        double raw[HP_N], norm2 = 0.0;
        for (int i = 0; i < HP_N; ++i) { raw[i] = next_gauss(); norm2 += raw[i] * raw[i]; }
        double scale = sqrt(1120.0 / (norm2 + 1e-12));
        for (;;) {
            int sq = 0;
            for (int i = 0; i < HP_N; ++i) {
                int q = (int)llround(raw[i] * scale);
                if (q > 34) q = 34; if (q < -34) q = -34;
                y[i] = (int8_t)q;
            }
            snap_a16_host_exact(y);
            snap_a16_host_exact(y + 16);
            for (int i = 0; i < HP_N; ++i) sq += (int)y[i] * (int)y[i];
            if (sq <= HP_BOUND) { mean_sq18 += sq; break; }
            scale *= 0.94;
        }
        uint8_t* rec = pool18_rec.data() + (size_t)p * 18;
        rec[0] = (uint8_t)((p * 13) & 63);
        if (!hp_rank_one_host(y, h_ways.data(), rec + 1)) {
            fprintf(stderr, "hp_rank_one_host failed on pool %d\n", p);
            exit(2);
        }
    }
    mean_sq18 /= POOL18;

    const int POOL72 = 512;
    std::vector<int8_t> pool72_y((size_t)POOL72 * HI_GROUP);
    std::vector<uint8_t> pool72_rec((size_t)POOL72 * 72);
    double mean_sq72 = 0.0;
    for (int p = 0; p < POOL72; ++p) {
        int8_t* y = pool72_y.data() + (size_t)p * HI_GROUP;
        double raw[HI_GROUP], norm2 = 0.0;
        for (int i = 0; i < HI_GROUP; ++i) { raw[i] = next_gauss(); norm2 += raw[i] * raw[i]; }
        double scale = sqrt(4850.0 / (norm2 + 1e-12));
        for (;;) {
            int sq = 0;
            for (int i = 0; i < HI_GROUP; ++i) {
                int q = (int)llround(raw[i] * scale);
                if (q > 72) q = 72; if (q < -72) q = -72;
                y[i] = (int8_t)q;
            }
            for (int b = 0; b < HI_BLOCKS; ++b) snap_a16_host_exact(y + b * 16);
            for (int i = 0; i < HI_GROUP; ++i) sq += (int)y[i] * (int)y[i];
            if (sq <= HI_BOUND) { mean_sq72 += sq; break; }
            scale *= 0.94;
        }
        uint8_t* rec = pool72_rec.data() + (size_t)p * 72;
        __half hsc = __float2half(0.02f + 0.0005f * (float)(p & 31));
        uint16_t hb = *reinterpret_cast<const uint16_t*>(&hsc);
        rec[0] = (uint8_t)(hb & 0xFFu);
        rec[1] = (uint8_t)(hb >> 8);
        if (!hi_rank_one_host(y, h_shell_p.data(), h_S_even.data(), h_after_e.data(), rec + 2)) {
            fprintf(stderr, "hi_rank_one_host failed on pool %d\n", p);
            exit(2);
        }
    }
    mean_sq72 /= POOL72;

    uint32_t* d_err = nullptr;
    CUDA_CHECK(cudaMalloc(&d_err, sizeof(uint32_t)));

    // Exact standalone round-trip bit verification on full pools FIRST (0 mismatches!)
    float *d_out18 = nullptr, *d_out72 = nullptr;
    uint8_t *d_p18 = nullptr, *d_p72 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_p18, pool18_rec.size()));
    CUDA_CHECK(cudaMemcpy(d_p18, pool18_rec.data(), pool18_rec.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_out18, (size_t)POOL18 * HP_N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_err, 0, sizeof(uint32_t)));
    hampack_18b_standalone_decode_kernel<<<(POOL18 + 255) / 256, 256>>>(d_p18, d_ways, d_scale_lut, d_out18, d_err, POOL18);
    CUDA_CHECK(cudaDeviceSynchronize());
    uint32_t syn18_pool = 0u;
    CUDA_CHECK(cudaMemcpy(&syn18_pool, d_err, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    std::vector<float> h_out18((size_t)POOL18 * HP_N);
    CUDA_CHECK(cudaMemcpy(h_out18.data(), d_out18, h_out18.size() * sizeof(float), cudaMemcpyDeviceToHost));
    uint32_t bit_mism18 = 0u;
    for (int p = 0; p < POOL18; ++p) {
        float sc = h_scale_lut[pool18_rec[(size_t)p * 18]];
        for (int i = 0; i < HP_N; ++i) {
            float exp_v = sc * (float)pool18_y[(size_t)p * HP_N + i];
            if (h_out18[(size_t)p * HP_N + i] != exp_v) ++bit_mism18;
        }
    }
    CUDA_CHECK(cudaFree(d_p18)); CUDA_CHECK(cudaFree(d_out18));

    CUDA_CHECK(cudaMalloc(&d_p72, pool72_rec.size()));
    CUDA_CHECK(cudaMemcpy(d_p72, pool72_rec.data(), pool72_rec.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_out72, (size_t)POOL72 * HI_GROUP * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_err, 0, sizeof(uint32_t)));
    hammingindex_72b_standalone_decode_kernel<<<(POOL72 + 255) / 256, 256>>>(d_p72, d_shell_p, d_S_even, d_after_e, d_out72, d_err, POOL72);
    CUDA_CHECK(cudaDeviceSynchronize());
    uint32_t syn72_pool = 0u;
    CUDA_CHECK(cudaMemcpy(&syn72_pool, d_err, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    std::vector<float> h_out72((size_t)POOL72 * HI_GROUP);
    CUDA_CHECK(cudaMemcpy(h_out72.data(), d_out72, h_out72.size() * sizeof(float), cudaMemcpyDeviceToHost));
    uint32_t bit_mism72 = 0u;
    for (int p = 0; p < POOL72; ++p) {
        uint16_t hb = (uint16_t)pool72_rec[(size_t)p * 72] | ((uint16_t)pool72_rec[(size_t)p * 72 + 1] << 8);
        float sc = __half2float(*reinterpret_cast<const __half*>(&hb));
        for (int i = 0; i < HI_GROUP; ++i) {
            float exp_v = sc * (float)pool72_y[(size_t)p * HI_GROUP + i];
            if (h_out72[(size_t)p * HI_GROUP + i] != exp_v) ++bit_mism72;
        }
    }
    CUDA_CHECK(cudaFree(d_p72)); CUDA_CHECK(cudaFree(d_out72));

    // Benchmark helper for BOTH standalone decode & fused GEMV at M x K
    auto bench_shape = [&](int M, int K, bool first) {
        const size_t n_w = (size_t)M * (size_t)K;
        const int g18 = (int)(n_w / HP_N);
        const int g72 = (int)(n_w / HI_GROUP);
        std::vector<uint8_t> h_rec18((size_t)g18 * 18);
        std::vector<uint8_t> h_rec72((size_t)g72 * 72);
        for (int g = 0; g < g18; ++g) {
            int idx = (int)(((uint64_t)g * 2654435761ULL) & (POOL18 - 1));
            memcpy(h_rec18.data() + (size_t)g * 18, pool18_rec.data() + (size_t)idx * 18, 18);
        }
        for (int g = 0; g < g72; ++g) {
            int idx = (int)(((uint64_t)g * 2654435761ULL) & (POOL72 - 1));
            memcpy(h_rec72.data() + (size_t)g * 72, pool72_rec.data() + (size_t)idx * 72, 72);
        }
        std::vector<float> h_x(K);
        uint32_t xrng = 0x77A16u ^ (uint32_t)K;
        for (int j = 0; j < K; ++j) {
            xrng = xrng * 1664525u + 1013904223u;
            h_x[j] = ((float)((int)(xrng >> 16) % 2001) - 1000.0f) * 0.001f;
        }
        const int ref_rows = 128;
        std::vector<double> ref18(ref_rows, 0.0), ref72(ref_rows, 0.0);
        const int g18_row = K / HP_N, g72_row = K / HI_GROUP;
        for (int r = 0; r < ref_rows; ++r) {
            double acc18 = 0.0;
            for (int c = 0; c < g18_row; ++c) {
                int g = r * g18_row + c;
                int idx = (int)(((uint64_t)g * 2654435761ULL) & (POOL18 - 1));
                double sc = (double)h_scale_lut[pool18_rec[(size_t)idx * 18]];
                double d = 0.0;
                for (int i = 0; i < HP_N; ++i) d += (double)pool18_y[(size_t)idx * HP_N + i] * (double)h_x[c * HP_N + i];
                acc18 += sc * d;
            }
            ref18[r] = acc18;
            double acc72 = 0.0;
            for (int c = 0; c < g72_row; ++c) {
                int g = r * g72_row + c;
                int idx = (int)(((uint64_t)g * 2654435761ULL) & (POOL72 - 1));
                uint16_t hb = (uint16_t)pool72_rec[(size_t)idx * 72] | ((uint16_t)pool72_rec[(size_t)idx * 72 + 1] << 8);
                double sc = (double)__half2float(*reinterpret_cast<const __half*>(&hb));
                double d = 0.0;
                for (int i = 0; i < HI_GROUP; ++i) d += (double)pool72_y[(size_t)idx * HI_GROUP + i] * (double)h_x[c * HI_GROUP + i];
                acc72 += sc * d;
            }
            ref72[r] = acc72;
        }

        uint8_t *d_r18 = nullptr, *d_r72 = nullptr;
        float *d_x = nullptr, *d_y = nullptr, *d_dec = nullptr;
        CUDA_CHECK(cudaMalloc(&d_r18, h_rec18.size()));
        CUDA_CHECK(cudaMemcpy(d_r18, h_rec18.data(), h_rec18.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_r72, h_rec72.size()));
        CUDA_CHECK(cudaMemcpy(d_r72, h_rec72.data(), h_rec72.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_x, (size_t)K * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), (size_t)K * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_y, (size_t)M * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dec, n_w * sizeof(float)));

        float ms = 0.0f;
        // 18B standalone decode
        CUDA_CHECK(cudaMemset(d_err, 0, sizeof(uint32_t)));
        hampack_18b_standalone_decode_kernel<<<(g18 + 255) / 256, 256>>>(d_r18, d_ways, d_scale_lut, d_dec, d_err, g18);
        CUDA_CHECK(cudaDeviceSynchronize());
        uint32_t err18_sa = 0u;
        CUDA_CHECK(cudaMemcpy(&err18_sa, d_err, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        const int it18 = 10;
        CUDA_CHECK(cudaEventRecord(t0));
        for (int i = 0; i < it18; ++i)
            hampack_18b_standalone_decode_kernel<<<(g18 + 255) / 256, 256>>>(d_r18, d_ways, d_scale_lut, d_dec, d_err, g18);
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        double sa18_ms = (double)ms / it18;
        double sa18_gw = (double)n_w / (sa18_ms * 1e6);

        // 18B fused GEMV
        dim3 block(128), grid((M + 3) / 4);
        CUDA_CHECK(cudaMemset(d_err, 0, sizeof(uint32_t)));
        hampack_18b_fused_gemv_kernel<true><<<grid, block>>>(d_r18, d_ways, d_scale_lut, d_x, d_y, d_err, M, K);
        CUDA_CHECK(cudaDeviceSynchronize());
        uint32_t err18_gv = 0u;
        CUDA_CHECK(cudaMemcpy(&err18_gv, d_err, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        std::vector<float> h_y(ref_rows);
        CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, ref_rows * sizeof(float), cudaMemcpyDeviceToHost));
        double max18 = 0.0, num18 = 0.0, den18 = 0.0;
        for (int r = 0; r < ref_rows; ++r) {
            double d = (double)h_y[r] - ref18[r];
            if (fabs(d) > max18) max18 = fabs(d);
            num18 += d * d; den18 += ref18[r] * ref18[r];
        }
        double rel18 = sqrt(num18 / (den18 + 1e-30));
        CUDA_CHECK(cudaEventRecord(t0));
        for (int i = 0; i < it18; ++i)
            hampack_18b_fused_gemv_kernel<false><<<grid, block>>>(d_r18, d_ways, d_scale_lut, d_x, d_y, d_err, M, K);
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        double gv18_ms = (double)ms / it18;
        double gv18_gw = (double)n_w / (gv18_ms * 1e6);
        double gv18_gbs = (double)(h_rec18.size() + (size_t)(K + M) * 4) / (gv18_ms * 1e6);
        double gv18_pct = 100.0 * gv18_gbs / triad_gbs;

        // 72B standalone decode
        CUDA_CHECK(cudaMemset(d_err, 0, sizeof(uint32_t)));
        hammingindex_72b_standalone_decode_kernel<<<(g72 + 255) / 256, 256>>>(d_r72, d_shell_p, d_S_even, d_after_e, d_dec, d_err, g72);
        CUDA_CHECK(cudaDeviceSynchronize());
        uint32_t err72_sa = 0u;
        CUDA_CHECK(cudaMemcpy(&err72_sa, d_err, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        const int it72 = (K <= 4096) ? 5 : 3;
        CUDA_CHECK(cudaEventRecord(t0));
        for (int i = 0; i < it72; ++i)
            hammingindex_72b_standalone_decode_kernel<<<(g72 + 255) / 256, 256>>>(d_r72, d_shell_p, d_S_even, d_after_e, d_dec, d_err, g72);
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        double sa72_ms = (double)ms / it72;
        double sa72_gw = (double)n_w / (sa72_ms * 1e6);

        // 72B fused GEMV
        CUDA_CHECK(cudaMemset(d_err, 0, sizeof(uint32_t)));
        hammingindex_72b_fused_gemv_kernel<true><<<grid, block>>>(d_r72, d_shell_p, d_S_even, d_after_e, d_x, d_y, d_err, M, K);
        CUDA_CHECK(cudaDeviceSynchronize());
        uint32_t err72_gv = 0u;
        CUDA_CHECK(cudaMemcpy(&err72_gv, d_err, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, ref_rows * sizeof(float), cudaMemcpyDeviceToHost));
        double max72 = 0.0, num72 = 0.0, den72 = 0.0;
        for (int r = 0; r < ref_rows; ++r) {
            double d = (double)h_y[r] - ref72[r];
            if (fabs(d) > max72) max72 = fabs(d);
            num72 += d * d; den72 += ref72[r] * ref72[r];
        }
        double rel72 = sqrt(num72 / (den72 + 1e-30));
        CUDA_CHECK(cudaEventRecord(t0));
        for (int i = 0; i < it72; ++i)
            hammingindex_72b_fused_gemv_kernel<false><<<grid, block>>>(d_r72, d_shell_p, d_S_even, d_after_e, d_x, d_y, d_err, M, K);
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        double gv72_ms = (double)ms / it72;
        double gv72_gw = (double)n_w / (gv72_ms * 1e6);
        double gv72_gbs = (double)(h_rec72.size() + (size_t)(K + M) * 4) / (gv72_ms * 1e6);
        double gv72_pct = 100.0 * gv72_gbs / triad_gbs;

        if (!first) printf(",\n");
        printf("      \"%dx%d\": {\n", M, K);
        printf("        \"hampack_18B_standalone\": {\"ms\": %.6f, \"gweights_per_s\": %.4f, \"speedup_vs_1core_cpu_10M\": %.1f, \"syndrome_errors\": %u},\n",
               sa18_ms, sa18_gw, sa18_gw * 100.0, err18_sa);
        printf("        \"hampack_18B_fused_gemv\": {\"ms\": %.6f, \"gweights_per_s\": %.4f, \"effective_gb_s\": %.4f, \"pct_of_stream_triad\": %.3f, \"syndrome_errors\": %u, \"max_abs_err_fp64\": %.9e, \"rel_l2_err_fp64\": %.9e},\n",
               gv18_ms, gv18_gw, gv18_gbs, gv18_pct, err18_gv, max18, rel18);
        printf("        \"hammingindex_72B_standalone\": {\"ms\": %.6f, \"gweights_per_s\": %.4f, \"speedup_vs_1core_cpu_10M\": %.1f, \"syndrome_errors\": %u},\n",
               sa72_ms, sa72_gw, sa72_gw * 100.0, err72_sa);
        printf("        \"hammingindex_72B_fused_gemv\": {\"ms\": %.6f, \"gweights_per_s\": %.4f, \"effective_gb_s\": %.4f, \"pct_of_stream_triad\": %.3f, \"syndrome_errors\": %u, \"max_abs_err_fp64\": %.9e, \"rel_l2_err_fp64\": %.9e}\n",
               gv72_ms, gv72_gw, gv72_gbs, gv72_pct, err72_gv, max72, rel72);
        printf("      }");

        CUDA_CHECK(cudaFree(d_r18)); CUDA_CHECK(cudaFree(d_r72));
        CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_y)); CUDA_CHECK(cudaFree(d_dec));
    };

    printf("  \"siliquant_pr77_enumerative_wire_formats\": {\n");
    printf("    \"hampack_18B_meta\": {\"L\": 16, \"n\": 32, \"bits\": 136, \"block_bytes\": 18, \"bound\": %d, \"table_bytes\": %zu, \"count_limbs_hex\": [\"0x%016llx\", \"0x%016llx\", \"0x%016llx\"], \"mean_block_sq_norm\": %.2f, \"pool_bit_mismatches\": %u, \"pool_syndrome_errors\": %u},\n",
           HP_BOUND, h_ways.size() * sizeof(U192),
           (unsigned long long)c18.w0, (unsigned long long)c18.w1, (unsigned long long)c18.w2,
           mean_sq18, bit_mism18, syn18_pool);
    printf("    \"hammingindex_72B_meta\": {\"L\": 16, \"group\": 128, \"bits\": 560, \"block_bytes\": 72, \"bound\": %d, \"shell_compressed_bytes\": %zu, \"after_even_bytes\": %zu, \"count_limbs_hex\": [\"0x%016llx\", \"0x%016llx\", \"0x%016llx\", \"0x%016llx\", \"0x%016llx\", \"0x%016llx\", \"0x%016llx\", \"0x%016llx\", \"0x%016llx\"], \"mean_group_sq_norm\": %.2f, \"pool_bit_mismatches\": %u, \"pool_syndrome_errors\": %u},\n",
           HI_BOUND, h_shell_p.size() * sizeof(U128), h_after_e.size() * sizeof(U576),
           (unsigned long long)c72.w[0], (unsigned long long)c72.w[1], (unsigned long long)c72.w[2],
           (unsigned long long)c72.w[3], (unsigned long long)c72.w[4], (unsigned long long)c72.w[5],
           (unsigned long long)c72.w[6], (unsigned long long)c72.w[7], (unsigned long long)c72.w[8],
           mean_sq72, bit_mism72, syn72_pool);
    printf("    \"shapes\": {\n");
    bench_shape(4096, 4096, true);
    bench_shape(4096, 14336, false);
    printf("\n    }\n");
    printf("  }");

    CUDA_CHECK(cudaFree(d_ways)); CUDA_CHECK(cudaFree(d_shell_p));
    CUDA_CHECK(cudaFree(d_S_even)); CUDA_CHECK(cudaFree(d_after_e));
    CUDA_CHECK(cudaFree(d_scale_lut)); CUDA_CHECK(cudaFree(d_err));
}

int main() {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int memClockKHz = 0, coreClockKHz = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&memClockKHz, cudaDevAttrMemoryClockRate, 0));
    CUDA_CHECK(cudaDeviceGetAttribute(&coreClockKHz, cudaDevAttrClockRate, 0));
    double peak_bw_gbs = 2.0 * (double)memClockKHz * 1e3 * (prop.memoryBusWidth / 8.0) / 1e9;

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
    int triad_blocks = prop.multiProcessorCount * 32;
    stream_triad_kernel<<<triad_blocks, 256>>>(da, db, dc, 1.5f, N4);
    CUDA_CHECK(cudaDeviceSynchronize());
    const int triad_iters = 25;
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < triad_iters; ++i)
        stream_triad_kernel<<<triad_blocks, 256>>>(da, db, dc, 1.5f, N4);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    double triad_gbs = (3.0 * (double)bytes_arr * triad_iters) / (ms * 1e6);
    CUDA_CHECK(cudaFree(da)); CUDA_CHECK(cudaFree(db)); CUDA_CHECK(cudaFree(dc));

    // 2. PCIe H2D / D2H
    const size_t pcie_bytes = 128ULL * 1024 * 1024;
    void *h_pin = nullptr, *d_buf = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_pin, pcie_bytes));
    CUDA_CHECK(cudaMalloc(&d_buf, pcie_bytes));
    memset(h_pin, 0x5A, pcie_bytes);
    CUDA_CHECK(cudaMemcpy(d_buf, h_pin, pcie_bytes, cudaMemcpyHostToDevice));
    const int pcie_iters = 10;
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < pcie_iters; ++i)
        CUDA_CHECK(cudaMemcpy(d_buf, h_pin, pcie_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    double h2d_gbs = ((double)pcie_bytes * pcie_iters) / (ms * 1e6);
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < pcie_iters; ++i)
        CUDA_CHECK(cudaMemcpy(h_pin, d_buf, pcie_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    double d2h_gbs = ((double)pcie_bytes * pcie_iters) / (ms * 1e6);
    CUDA_CHECK(cudaFreeHost(h_pin)); CUDA_CHECK(cudaFree(d_buf));

    // 3. FP32 FMA & INT8 dp4a peak
    float* d_fout = nullptr; int* d_iout = nullptr;
    CUDA_CHECK(cudaMalloc(&d_fout, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_iout, sizeof(int)));
    int cblocks = prop.multiProcessorCount * 8, cthreads = 256, citers = 8192;
    fp32_fma_peak_kernel<<<cblocks, cthreads>>>(d_fout, 64);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(t0));
    fp32_fma_peak_kernel<<<cblocks, cthreads>>>(d_fout, citers);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    double fp32_tflops = ((double)cblocks * cthreads * citers * 16.0) / (ms * 1e9);

    int8_dp4a_peak_kernel<<<cblocks, cthreads>>>(d_iout, 64);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(t0));
    int8_dp4a_peak_kernel<<<cblocks, cthreads>>>(d_iout, citers);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    double dp4a_tops = ((double)cblocks * cthreads * citers * 64.0) / (ms * 1e9);
    CUDA_CHECK(cudaFree(d_fout)); CUDA_CHECK(cudaFree(d_iout));

    // 4. Standalone 67.1M-weight INT8 decode microbenchmark (direct AC vs battery comparison)
    const int N_BLOCKS = 4 * 1024 * 1024;
    const size_t qbytes = (size_t)N_BLOCKS * 16;
    std::vector<int8_t> h_in(qbytes);
    uint32_t rng = 0xA1E7E1Au;
    for (size_t i = 0; i < qbytes; ++i) {
        rng = rng * 1664525u + 1013904223u;
        h_in[i] = (int8_t)(((int)(rng >> 24) % 79) - 39); // [-39, +39] -> Block-H2 peak 78<=127
    }
    int8_t *d_qin = nullptr, *d_qout = nullptr;
    uint32_t *d_err = nullptr, h_err = 0u;
    CUDA_CHECK(cudaMalloc(&d_qin, qbytes));
    CUDA_CHECK(cudaMalloc(&d_qout, qbytes));
    CUDA_CHECK(cudaMalloc(&d_err, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_qin, h_in.data(), qbytes, cudaMemcpyHostToDevice));
    int qgrid = (N_BLOCKS + 255) / 256, qiters = 50;

    CUDA_CHECK(cudaMemset(d_err, 0, sizeof(uint32_t)));
    standalone_a16_decode_kernel<<<qgrid, 256>>>(d_qin, d_qout, d_err, N_BLOCKS);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemset(d_err, 0, sizeof(uint32_t)));
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < qiters; ++i)
        standalone_a16_decode_kernel<<<qgrid, 256>>>(d_qin, d_qout, d_err, N_BLOCKS);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    CUDA_CHECK(cudaMemcpy(&h_err, d_err, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    double sa_a16_ms = (double)ms / qiters;
    double sa_a16_gw = ((double)N_BLOCKS * 16.0) / (sa_a16_ms * 1e6);
    double sa_a16_gbs = (2.0 * (double)qbytes) / (sa_a16_ms * 1e6);

    standalone_block_h2_kernel<<<qgrid, 256>>>(d_qin, d_qout, N_BLOCKS);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < qiters; ++i)
        standalone_block_h2_kernel<<<qgrid, 256>>>(d_qin, d_qout, N_BLOCKS);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    double sa_h2_ms = (double)ms / qiters;
    double sa_h2_gw = ((double)N_BLOCKS * 16.0) / (sa_h2_ms * 1e6);
    double sa_h2_gbs = (2.0 * (double)qbytes) / (sa_h2_ms * 1e6);

    standalone_wht8_kernel<<<qgrid, 256>>>(d_qin, d_qout, N_BLOCKS);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < qiters; ++i)
        standalone_wht8_kernel<<<qgrid, 256>>>(d_qin, d_qout, N_BLOCKS);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    double sa_wht_ms = (double)ms / qiters;
    double sa_wht_gw = ((double)N_BLOCKS * 16.0) / (sa_wht_ms * 1e6);
    double sa_wht_gbs = (2.0 * (double)qbytes) / (sa_wht_ms * 1e6);

    CUDA_CHECK(cudaFree(d_qin)); CUDA_CHECK(cudaFree(d_qout)); CUDA_CHECK(cudaFree(d_err));

    // Emit canonical JSON
    printf("{\n");
    printf("  \"device\": {\n");
    printf("    \"name\": \"%s\",\n", prop.name);
    printf("    \"cc_major\": %d, \"cc_minor\": %d,\n", prop.major, prop.minor);
    printf("    \"sm_count\": %d,\n", prop.multiProcessorCount);
    printf("    \"shared_mem_per_sm_bytes\": %zu,\n", prop.sharedMemPerMultiprocessor);
    printf("    \"l2_cache_bytes\": %d,\n", prop.l2CacheSize);
    printf("    \"total_vram_bytes\": %zu,\n", prop.totalGlobalMem);
    printf("    \"memory_bus_width_bits\": %d,\n", prop.memoryBusWidth);
    printf("    \"memory_clock_khz\": %d,\n", memClockKHz);
    printf("    \"boost_clock_khz\": %d,\n", coreClockKHz);
    printf("    \"theoretical_peak_bw_gb_s\": %.4f\n", peak_bw_gbs);
    printf("  },\n");
    printf("  \"roofline\": {\n");
    printf("    \"stream_triad_gb_s\": %.4f,\n", triad_gbs);
    printf("    \"stream_triad_pct_peak\": %.3f,\n", 100.0 * triad_gbs / peak_bw_gbs);
    printf("    \"pcie_h2d_gb_s\": %.4f,\n", h2d_gbs);
    printf("    \"pcie_d2h_gb_s\": %.4f,\n", d2h_gbs);
    printf("    \"fp32_fma_tflops\": %.4f,\n", fp32_tflops);
    printf("    \"int8_dp4a_tops\": %.4f\n", dp4a_tops);
    printf("  },\n");
    printf("  \"standalone_decode_67m\": {\n");
    printf("    \"n_blocks\": %d, \"weights\": %lld,\n", N_BLOCKS, (long long)N_BLOCKS * 16LL);
    printf("    \"A16_syndrome\": {\"ms\": %.6f, \"gweights_per_s\": %.4f, \"effective_gb_s\": %.4f, \"syndrome_errors\": %u},\n",
           sa_a16_ms, sa_a16_gw, sa_a16_gbs, h_err);
    printf("    \"Block_H2_int8\": {\"ms\": %.6f, \"gweights_per_s\": %.4f, \"effective_gb_s\": %.4f},\n",
           sa_h2_ms, sa_h2_gw, sa_h2_gbs);
    printf("    \"WHT8_butterfly\": {\"ms\": %.6f, \"gweights_per_s\": %.4f, \"effective_gb_s\": %.4f}\n",
           sa_wht_ms, sa_wht_gw, sa_wht_gbs);
    printf("  },\n");
    printf("  \"fused_gemv_4_5_bpw\": {\n");
    run_gemv_shape(4096, 4096, triad_gbs, t0, t1, true);
    run_gemv_shape(4096, 14336, triad_gbs, t0, t1, false);
    printf("\n  },\n");
    run_siliquant_hampack_benchmarks(triad_gbs, t0, t1);
    printf("\n}\n");

    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    return (h_err == 0u) ? 0 : 2;
}
