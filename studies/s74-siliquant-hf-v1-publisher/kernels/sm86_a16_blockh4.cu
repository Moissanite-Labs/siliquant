// Reference Ampere sm_86 PTX Fused A_16 [16,11,4] + Block-H4 dp4a.s32.s32 GEMV Kernel
// Part of the `siliquant-hf-v1` release bundle (Moissanite-Labs/siliquant).
//
// Features:
//   - Register-only 4-point Sylvester Hadamard butterfly H_4 with exact dyadic 0.5f scale.
//   - Live Extended Hamming [16, 11, 4] 5-bit syndrome check via hardware __popc().
//   - PTX `dp4a.s32.s32` fused INT8xINT8->INT32 dot product (zero register spill on sm_86).
//   - Optional shared-RMSNorm input permutation gather (`perm_in == nullptr` => 0% gather overhead).
//   - Built-in bit-exact GPU int32 vs CPU int64 verification self-test.
//
// Build & self-test:
//   nvcc -O3 -std=c++17 -arch=sm_86 --ptxas-options=-v sm86_a16_blockh4.cu -o sm86_a16_blockh4
//   ./sm86_a16_blockh4

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define SILIQUANT_CUDA_CHECK(call)                                                      \
    do {                                                                                \
        cudaError_t err__ = (call);                                                     \
        if (err__ != cudaSuccess) {                                                     \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,               \
                    cudaGetErrorString(err__));                                         \
            exit(1);                                                                    \
        }                                                                               \
    } while (0)

__device__ __forceinline__ float warp_reduce_sum_f32(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xFFFFFFFFu, v, off);
    }
    return v;
}

__device__ __forceinline__ int warp_reduce_sum_i32(int v) {
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

__device__ __forceinline__ int pack4_s8(int a, int b, int c, int d) {
    return ((a & 0xFF) | ((b & 0xFF) << 8) | ((c & 0xFF) << 16) | ((d & 0xFF) << 24));
}

__device__ __forceinline__ int dp4a_s32_ptx(int a, int b, int c) {
    int out;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(out) : "r"(a), "r"(b), "r"(c));
    return out;
}

// One-time shared-RMSNorm INT8 gather kernel (amortized across Q/K/V or Gate/Up projections).
__global__ void siliquant_gather_x_i8_kernel(
    const int8_t* __restrict__ x_src,
    const int32_t* __restrict__ perm_in,
    int8_t* __restrict__ x_dst,
    int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < K) {
        x_dst[idx] = __ldg(x_src + __ldg(perm_in + idx));
    }
}

// Fused 4.5-bpw A_16 + Block-H4 (H_4 / 2) GEMV kernel with optional live syndrome verification.
// Packed layout: 16 signed 4-bit weights per uint64_t (1 A_16 codeword), 1 FP16 scale per 32 weights.
template <bool CHECK_SYN>
__global__ void siliquant_a16_blockh4_dp4a_gemv_kernel(
    const uint64_t* __restrict__ packed,
    const __half* __restrict__ scales,
    const int4* __restrict__ x_i8,
    float x_scale,
    float* __restrict__ y,
    uint32_t* __restrict__ err_count,
    int M,
    int K) {
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
            for (int j = 0; j < 16; ++j) {
                odd |= ((uint32_t)((w64 >> (j * 4)) & 1ULL)) << j;
            }
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

            // Exact register-only 4-point Sylvester Hadamard butterfly:
            //   [r0, r1, r2, r3] = H_4 * [c0, c1, c2, c3], peak |r_i| <= 28 <= 127 (exact int8!).
            int s01 = c0 + c1;
            int d01 = c0 - c1;
            int s23 = c2 + c3;
            int d23 = c2 - c3;
            int r0 = s01 + s23;
            int r1 = d01 + d23;
            int r2 = s01 - s23;
            int r3 = d01 - d23;

            int wp = pack4_s8(r0, r1, r2, r3);
            int xp = (g == 0) ? xv.x : ((g == 1) ? xv.y : ((g == 2) ? xv.z : xv.w));
            idot = dp4a_s32_ptx(wp, xp, idot);
        }

        // Exact dyadic 0.5f = 1/sqrt(4) factor absorbed with FP16 group scale.
        float sc = 0.5f * __half2float(__ldg(row_s + (b >> 1))) * x_scale;
        acc = fmaf((float)idot, sc, acc);
    }

    acc = warp_reduce_sum_f32(acc);
    if (lane == 0) y[warp_id] = acc;
    if (CHECK_SYN && bad) atomicAdd(err_count, 1u);
}

// Exact integer-accumulator parity kernel (for bit-exact GPU vs CPU int64 self-test).
__global__ void siliquant_a16_blockh4_exact_i32_kernel(
    const uint64_t* __restrict__ packed,
    const int4* __restrict__ x_i8,
    int32_t* __restrict__ y_i32,
    uint32_t* __restrict__ err_count,
    int M,
    int K) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp_id >= M) return;

    const int n_blk16 = K >> 4;
    const uint64_t* row_p = packed + (size_t)warp_id * n_blk16;
    int acc = 0;
    uint32_t bad = 0u;

    for (int b = lane; b < n_blk16; b += 32) {
        uint64_t w64 = __ldg(row_p + b);
        int4 xv = __ldg(x_i8 + b);
        uint32_t odd = 0u;
        #pragma unroll
        for (int j = 0; j < 16; ++j) odd |= ((uint32_t)((w64 >> (j * 4)) & 1ULL)) << j;
        if (a16_parity_check_16(odd) != 0u) bad = 1u;

        #pragma unroll
        for (int g = 0; g < 4; ++g) {
            uint32_t nibs = (uint32_t)(w64 >> (g * 16));
            int c0 = ((int)(nibs & 0xFu) << 28) >> 28;
            int c1 = ((int)((nibs >> 4) & 0xFu) << 28) >> 28;
            int c2 = ((int)((nibs >> 8) & 0xFu) << 28) >> 28;
            int c3 = ((int)((nibs >> 12) & 0xFu) << 28) >> 28;
            int s01 = c0 + c1, d01 = c0 - c1;
            int s23 = c2 + c3, d23 = c2 - c3;
            int wp = pack4_s8(s01 + s23, d01 + d23, s01 - s23, d01 - d23);
            int xp = (g == 0) ? xv.x : ((g == 1) ? xv.y : ((g == 2) ? xv.z : xv.w));
            acc = dp4a_s32_ptx(wp, xp, acc);
        }
    }
    acc = warp_reduce_sum_i32(acc);
    if (lane == 0) y_i32[warp_id] = acc;
    if (bad) atomicAdd(err_count, 1u);
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
    uint64_t w = 0ULL;
    for (int i = 0; i < 16; ++i) w |= ((uint64_t)(v[i] & 0xF)) << (i * 4);
    return w;
}

int main() {
    const int M = 256;
    const int K = 1024;
    const int n_blk16 = K / 16;

    std::vector<uint64_t> h_packed((size_t)M * n_blk16);
    std::vector<int8_t> h_x(K);
    std::vector<int64_t> h_ref(M, 0);

    uint32_t rng = 20261124u;
    for (int i = 0; i < K; ++i) {
        rng = rng * 1664525u + 1013904223u;
        h_x[i] = (int8_t)(((int)(rng >> 24) % 63) - 31);
    }
    for (int r = 0; r < M; ++r) {
        int64_t acc = 0;
        for (int b = 0; b < n_blk16; ++b) {
            uint64_t w64 = make_valid_a16_word(&rng);
            h_packed[(size_t)r * n_blk16 + b] = w64;
            for (int g = 0; g < 4; ++g) {
                uint32_t nibs = (uint32_t)(w64 >> (g * 16));
                int c0 = ((int)(nibs & 0xFu) << 28) >> 28;
                int c1 = ((int)((nibs >> 4) & 0xFu) << 28) >> 28;
                int c2 = ((int)((nibs >> 8) & 0xFu) << 28) >> 28;
                int c3 = ((int)((nibs >> 12) & 0xFu) << 28) >> 28;
                int s01 = c0 + c1, d01 = c0 - c1;
                int s23 = c2 + c3, d23 = c2 - c3;
                int base = b * 16 + g * 4;
                acc += (int64_t)(s01 + s23) * h_x[base + 0]
                     + (int64_t)(d01 + d23) * h_x[base + 1]
                     + (int64_t)(s01 - s23) * h_x[base + 2]
                     + (int64_t)(d01 - d23) * h_x[base + 3];
            }
        }
        h_ref[r] = acc;
    }

    uint64_t* d_packed = nullptr;
    int8_t* d_x = nullptr;
    int32_t* d_y = nullptr;
    uint32_t* d_err = nullptr;
    SILIQUANT_CUDA_CHECK(cudaMalloc(&d_packed, h_packed.size() * sizeof(uint64_t)));
    SILIQUANT_CUDA_CHECK(cudaMalloc(&d_x, h_x.size() * sizeof(int8_t)));
    SILIQUANT_CUDA_CHECK(cudaMalloc(&d_y, M * sizeof(int32_t)));
    SILIQUANT_CUDA_CHECK(cudaMalloc(&d_err, sizeof(uint32_t)));
    SILIQUANT_CUDA_CHECK(cudaMemcpy(d_packed, h_packed.data(), h_packed.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
    SILIQUANT_CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(int8_t), cudaMemcpyHostToDevice));
    SILIQUANT_CUDA_CHECK(cudaMemset(d_err, 0, sizeof(uint32_t)));

    int warps_per_block = 4;
    int threads = warps_per_block * 32;
    int blocks = (M + warps_per_block - 1) / warps_per_block;
    siliquant_a16_blockh4_exact_i32_kernel<<<blocks, threads>>>(
        d_packed, reinterpret_cast<const int4*>(d_x), d_y, d_err, M, K);
    SILIQUANT_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int32_t> h_y(M, 0);
    uint32_t h_err = 0;
    SILIQUANT_CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, M * sizeof(int32_t), cudaMemcpyDeviceToHost));
    SILIQUANT_CUDA_CHECK(cudaMemcpy(&h_err, d_err, sizeof(uint32_t), cudaMemcpyDeviceToHost));

    int mismatches = 0;
    for (int r = 0; r < M; ++r) {
        if ((int64_t)h_y[r] != h_ref[r]) ++mismatches;
    }
    printf("sm86_a16_blockh4 self-test: M=%d K=%d syndrome_errors=%u int64_mismatches=%d\n",
           M, K, h_err, mismatches);

    cudaFree(d_packed);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_err);
    return (h_err == 0u && mismatches == 0) ? 0 : 1;
}
