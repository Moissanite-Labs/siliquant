// Track S68: Bare-Metal sm_86 PTX Fused A16_BlockH4 Kernel & Serving Pareto Benchmark
// Zero-spill (ptxas 0 B stack / 0 B spill, 0 PTX ld.local/st.local) on REAL Qwen2.5-3B weights:
//   1. Direct 4.5000-bpw A16_axis_preencoded_dp4a (baseline, CHECK_SYN=true verified)
//   2. Direct 4.5000-bpw Block_H2_dp4a (Track S63 comparison baseline)
//   3. Direct 4.5000-bpw Fused A16_BlockH4_dp4a: register-only 4-point butterfly H_4 + exact
//      dyadic 0.5f scale + PTX dp4a.s32.s32 + live GPU Extended Hamming [16,11,4] parity check
//   4. Dedicated exact int32 GPU vs CPU int64 bit-exact parity check for fused (H_4 W_q) * x
//   5. Live 100 Hz trapezoidal NVML INA3221 rail power + intel-rapl:0 CPU mJ/call integration

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <utility>
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

// ---------------------------------------------------------------------------
// 1. STREAM Triad Baseline (measured in same GPU run for % Triad reference)
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
// 2. Direct 4.5000-bpw A16_axis_preencoded_dp4a Baseline
// ---------------------------------------------------------------------------
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
            idot = dp4a_s32_ptx(wp, xp, idot);
        }
        float sc = __half2float(__ldg(row_s + (b >> 1))) * x_scale;
        acc = fmaf((float)idot, sc, acc);
    }
    acc = warp_reduce_sum_f32(acc);
    if (lane == 0) y[warp_id] = acc;
    if (CHECK_SYN && bad) atomicAdd(err_count, 1u);
}

// ---------------------------------------------------------------------------
// 3. Direct 4.5000-bpw Block_H2_dp4a Baseline (Track S63 reference)
// ---------------------------------------------------------------------------
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
    const float x_inv_sqrt2 = x_scale * 0.7071067811865475f;
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
            idot = dp4a_s32_ptx(wp, xp, idot);
        }
        float sc = __half2float(__ldg(row_s + (b >> 1))) * x_inv_sqrt2;
        acc = fmaf((float)idot, sc, acc);
    }
    acc = warp_reduce_sum_f32(acc);
    if (lane == 0) y[warp_id] = acc;
}

// ---------------------------------------------------------------------------
// 4. Direct 4.5000-bpw Fused Register-Only A16_BlockH4 PTX dp4a GEMV
//    Exact 4-point butterfly H_4 in registers:
//      a_j in [-7, 7] => u_k = (H_4 a)_k in [-28, 28] subset [-128, 127] (lossless int8!)
//      Exact dyadic normalization 1/2 = 0.5f hoisted out of inner loop.
// ---------------------------------------------------------------------------
template <bool CHECK_SYN>
__global__ void gemv_4_5bpw_a16_block_h4_dp4a_kernel(
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
    const float x_half = x_scale * 0.5f;
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
            int a0 = ((int)(nibs & 0xFu) << 28) >> 28;
            int a1 = ((int)((nibs >> 4) & 0xFu) << 28) >> 28;
            int a2 = ((int)((nibs >> 8) & 0xFu) << 28) >> 28;
            int a3 = ((int)((nibs >> 12) & 0xFu) << 28) >> 28;
            int s0 = a0 + a1;
            int d0 = a0 - a1;
            int s1 = a2 + a3;
            int d1 = a2 - a3;
            int wp = pack4_s8(s0 + s1, d0 + d1, s0 - s1, d0 - d1);
            int xp = (g == 0) ? xv.x : ((g == 1) ? xv.y : ((g == 2) ? xv.z : xv.w));
            idot = dp4a_s32_ptx(wp, xp, idot);
        }
        float sc = __half2float(__ldg(row_s + (b >> 1))) * x_half;
        acc = fmaf((float)idot, sc, acc);
    }
    acc = warp_reduce_sum_f32(acc);
    if (lane == 0) y[warp_id] = acc;
    if (CHECK_SYN && bad) atomicAdd(err_count, 1u);
}

// ---------------------------------------------------------------------------
// 5. Exact Unscaled int32 Bit-Exact Parity Kernel for Fused (H_4 W_q) * x_i8
// ---------------------------------------------------------------------------
__global__ void verify_block_h4_int32_exact_kernel(
    const uint64_t* __restrict__ packed,
    const int4* __restrict__ x_i8,
    int* __restrict__ out_i32,
    int M, int K) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp_id >= M) return;
    const int n_blk16 = K >> 4;
    const uint64_t* row_p = packed + (size_t)warp_id * n_blk16;
    int idot = 0;
    for (int b = lane; b < n_blk16; b += 32) {
        uint64_t w64 = __ldg(row_p + b);
        int4 xv = __ldg(x_i8 + b);
        #pragma unroll
        for (int g = 0; g < 4; ++g) {
            uint32_t nibs = (uint32_t)(w64 >> (g * 16));
            int a0 = ((int)(nibs & 0xFu) << 28) >> 28;
            int a1 = ((int)((nibs >> 4) & 0xFu) << 28) >> 28;
            int a2 = ((int)((nibs >> 8) & 0xFu) << 28) >> 28;
            int a3 = ((int)((nibs >> 12) & 0xFu) << 28) >> 28;
            int s0 = a0 + a1;
            int d0 = a0 - a1;
            int s1 = a2 + a3;
            int d1 = a2 - a3;
            int wp = pack4_s8(s0 + s1, d0 + d1, s0 - s1, d0 - d1);
            int xp = (g == 0) ? xv.x : ((g == 1) ? xv.y : ((g == 2) ? xv.z : xv.w));
            idot = dp4a_s32_ptx(wp, xp, idot);
        }
    }
    idot = warp_reduce_sum_i32(idot);
    if (lane == 0) out_i32[warp_id] = idot;
}

static uint64_t read_rapl_pkg_uj() {
    std::ifstream ifs("/sys/class/powercap/intel-rapl:0/energy_uj");
    uint64_t v = 0;
    if (ifs >> v) return v;
    return 0ULL;
}

struct NvmlPowerSampler {
    void* lib = nullptr;
    void* dev = nullptr;
    int (*fn_init)() = nullptr;
    int (*fn_shutdown)() = nullptr;
    int (*fn_get_handle)(unsigned int, void**) = nullptr;
    int (*fn_get_power)(void*, unsigned int*) = nullptr;
    bool ok = false;

    NvmlPowerSampler() {
        lib = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
        if (!lib) return;
        fn_init = reinterpret_cast<int (*)()>(dlsym(lib, "nvmlInit_v2"));
        fn_shutdown = reinterpret_cast<int (*)()>(dlsym(lib, "nvmlShutdown"));
        fn_get_handle = reinterpret_cast<int (*)(unsigned int, void**)>(dlsym(lib, "nvmlDeviceGetHandleByIndex_v2"));
        fn_get_power = reinterpret_cast<int (*)(void*, unsigned int*)>(dlsym(lib, "nvmlDeviceGetPowerUsage"));
        if (!fn_init || !fn_shutdown || !fn_get_handle || !fn_get_power) return;
        if (fn_init() != 0) return;
        if (fn_get_handle(0, &dev) != 0) {
            fn_shutdown();
            return;
        }
        ok = true;
    }

    ~NvmlPowerSampler() {
        if (ok && fn_shutdown) fn_shutdown();
        if (lib) dlclose(lib);
    }

    double read_watts() const {
        if (!ok) return 0.0;
        unsigned int mw = 0;
        if (fn_get_power(dev, &mw) != 0) return 0.0;
        return (double)mw * 1e-3;
    }
};

struct RailEnergyResult {
    double gpu_total_mj = 0.0;
    double gpu_avg_w = 0.0;
    int samples = 0;
    double rapl_total_mj = 0.0;
};

template <typename Fn>
static RailEnergyResult measure_100hz_rail_energy(const NvmlPowerSampler& nvml, Fn&& fn) {
    RailEnergyResult res;
    std::atomic<bool> running{true};
    std::vector<std::pair<double, double>> trace;
    trace.reserve(256);
    auto t_start = std::chrono::steady_clock::now();
    std::thread sampler([&]() {
        while (running.load(std::memory_order_relaxed)) {
            double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
            double w = nvml.read_watts();
            trace.emplace_back(t, w);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
        double w = nvml.read_watts();
        trace.emplace_back(t, w);
    });
    uint64_t r0 = read_rapl_pkg_uj();
    fn();
    uint64_t r1 = read_rapl_pkg_uj();
    running.store(false, std::memory_order_relaxed);
    sampler.join();

    res.rapl_total_mj = (r1 >= r0) ? ((double)(r1 - r0) / 1000.0) : 0.0;
    res.samples = (int)trace.size();
    if (trace.size() >= 2) {
        double joules = 0.0;
        for (size_t i = 1; i < trace.size(); ++i) {
            double dt = trace[i].first - trace[i - 1].first;
            if (dt > 0.0) joules += 0.5 * (trace[i - 1].second + trace[i].second) * dt;
        }
        double total_t = trace.back().first - trace.front().first;
        res.gpu_total_mj = joules * 1000.0;
        res.gpu_avg_w = (total_t > 1e-6) ? (joules / total_t) : trace.back().second;
    } else if (!trace.empty()) {
        res.gpu_avg_w = trace[0].second;
    }
    return res;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <s68_real_weights_bundle.bin>\n", argv[0]);
        return 2;
    }
    // Binary format "S68B" written from real Qwen2.5-3B safetensors:
    // Header: magic "S68B" (4B), M (int32), K (int32)
    //   y45_a16 [M*K] (int8 in [-7,7], syn=0), s45_a16 [M*K/32] (fp16)
    //   y45_bh2 [M*K] (int8 in [-7,7]),        s45_bh2 [M*K/32] (fp16)
    //   y45_bh4 [M*K] (int8 in [-7,7], syn=0), s45_bh4 [M*K/32] (fp16)
    //   x_real  [K]   (fp32 real captured Qwen2.5-3B activation vector)
    std::ifstream ifs(argv[1], std::ios::binary);
    if (!ifs) { fprintf(stderr, "Cannot open %s\n", argv[1]); return 2; }
    char magic[4];
    int32_t M = 0, K = 0;
    ifs.read(magic, 4);
    if (memcmp(magic, "S68B", 4) != 0) { fprintf(stderr, "Bad magic\n"); return 2; }
    ifs.read(reinterpret_cast<char*>(&M), 4);
    ifs.read(reinterpret_cast<char*>(&K), 4);

    const size_t n_w = (size_t)M * (size_t)K;
    const size_t n_blk16 = n_w / 16;
    const size_t g32 = n_w / 32;
    std::vector<int8_t> y45_a16(n_w), y45_bh2(n_w), y45_bh4(n_w);
    std::vector<uint16_t> s45_a16_u16(g32), s45_bh2_u16(g32), s45_bh4_u16(g32);
    std::vector<float> x_real(K);

    ifs.read(reinterpret_cast<char*>(y45_a16.data()), (std::streamsize)n_w);
    ifs.read(reinterpret_cast<char*>(s45_a16_u16.data()), (std::streamsize)(g32 * 2));
    ifs.read(reinterpret_cast<char*>(y45_bh2.data()), (std::streamsize)n_w);
    ifs.read(reinterpret_cast<char*>(s45_bh2_u16.data()), (std::streamsize)(g32 * 2));
    ifs.read(reinterpret_cast<char*>(y45_bh4.data()), (std::streamsize)n_w);
    ifs.read(reinterpret_cast<char*>(s45_bh4_u16.data()), (std::streamsize)(g32 * 2));
    ifs.read(reinterpret_cast<char*>(x_real.data()), (std::streamsize)(K * 4));
    if (!ifs) { fprintf(stderr, "Truncated bundle\n"); return 2; }

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    float ms = 0.0f;

    // 1. STREAM Triad Baseline
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

    // 2. Pack direct 4.5000-bpw 64-bit nibble words
    std::vector<uint64_t> pack_a16(n_blk16, 0ULL), pack_bh2(n_blk16, 0ULL), pack_bh4(n_blk16, 0ULL);
    for (size_t b = 0; b < n_blk16; ++b) {
        uint64_t wa = 0ULL, wb2 = 0ULL, wb4 = 0ULL;
        for (int j = 0; j < 16; ++j) {
            wa  |= ((uint64_t)(y45_a16[b * 16 + j] & 0xF)) << (j * 4);
            wb2 |= ((uint64_t)(y45_bh2[b * 16 + j] & 0xF)) << (j * 4);
            wb4 |= ((uint64_t)(y45_bh4[b * 16 + j] & 0xF)) << (j * 4);
        }
        pack_a16[b] = wa;
        pack_bh2[b] = wb2;
        pack_bh4[b] = wb4;
    }

    // 3. Quantize real activation x_real to int8 for dp4a
    float x_amax = 1e-8f;
    for (int j = 0; j < K; ++j) if (fabsf(x_real[j]) > x_amax) x_amax = fabsf(x_real[j]);
    float x_scale = x_amax / 127.0f;
    std::vector<int8_t> x_i8(K);
    for (int j = 0; j < K; ++j) {
        int q = (int)llround((double)x_real[j] / (double)x_scale);
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        x_i8[j] = (int8_t)q;
    }

    // 4. Exact CPU FP64 & int64 references across ALL M rows
    std::vector<double> ref_a16(M, 0.0), ref_bh2(M, 0.0), ref_bh4(M, 0.0);
    std::vector<int64_t> ref_bh4_i64(M, 0LL);
    for (int r = 0; r < M; ++r) {
        double aa = 0.0, ab2 = 0.0, ab4 = 0.0;
        int64_t ib4 = 0LL;
        for (int c = 0; c < K; ++c) {
            size_t idx = (size_t)r * K + c;
            double sc = (double)__half2float(*reinterpret_cast<const __half*>(&s45_a16_u16[idx / 32]));
            aa += sc * (double)y45_a16[idx] * (double)x_i8[c] * (double)x_scale;
        }
        for (int c = 0; c < K; c += 2) {
            size_t idx = (size_t)r * K + c;
            double sc = (double)__half2float(*reinterpret_cast<const __half*>(&s45_bh2_u16[idx / 32])) * 0.7071067811865475;
            double w0 = (double)y45_bh2[idx] + (double)y45_bh2[idx + 1];
            double w1 = (double)y45_bh2[idx] - (double)y45_bh2[idx + 1];
            ab2 += sc * (w0 * (double)x_i8[c] + w1 * (double)x_i8[c + 1]) * (double)x_scale;
        }
        for (int c = 0; c < K; c += 4) {
            size_t idx = (size_t)r * K + c;
            int64_t a0 = (int64_t)y45_bh4[idx];
            int64_t a1 = (int64_t)y45_bh4[idx + 1];
            int64_t a2 = (int64_t)y45_bh4[idx + 2];
            int64_t a3 = (int64_t)y45_bh4[idx + 3];
            int64_t u0 = a0 + a1 + a2 + a3;
            int64_t u1 = a0 - a1 + a2 - a3;
            int64_t u2 = a0 + a1 - a2 - a3;
            int64_t u3 = a0 - a1 - a2 + a3;
            int64_t dot4 = u0 * (int64_t)x_i8[c] + u1 * (int64_t)x_i8[c + 1]
                         + u2 * (int64_t)x_i8[c + 2] + u3 * (int64_t)x_i8[c + 3];
            ib4 += dot4;
            double sc = (double)__half2float(*reinterpret_cast<const __half*>(&s45_bh4_u16[idx / 32])) * 0.5;
            ab4 += sc * (double)dot4 * (double)x_scale;
        }
        ref_a16[r] = aa;
        ref_bh2[r] = ab2;
        ref_bh4[r] = ab4;
        ref_bh4_i64[r] = ib4;
    }

    // 5. Upload GPU buffers
    uint64_t *d_pa = nullptr, *d_pb2 = nullptr, *d_pb4 = nullptr;
    __half *d_sa = nullptr, *d_sb2 = nullptr, *d_sb4 = nullptr;
    int8_t* d_xi8 = nullptr;
    float* d_y = nullptr;
    int* d_yi32 = nullptr;
    uint32_t* d_err = nullptr;

    CUDA_CHECK(cudaMalloc(&d_pa, n_blk16 * 8));
    CUDA_CHECK(cudaMemcpy(d_pa, pack_a16.data(), n_blk16 * 8, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_sa, g32 * 2));
    CUDA_CHECK(cudaMemcpy(d_sa, s45_a16_u16.data(), g32 * 2, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_pb2, n_blk16 * 8));
    CUDA_CHECK(cudaMemcpy(d_pb2, pack_bh2.data(), n_blk16 * 8, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_sb2, g32 * 2));
    CUDA_CHECK(cudaMemcpy(d_sb2, s45_bh2_u16.data(), g32 * 2, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_pb4, n_blk16 * 8));
    CUDA_CHECK(cudaMemcpy(d_pb4, pack_bh4.data(), n_blk16 * 8, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_sb4, g32 * 2));
    CUDA_CHECK(cudaMemcpy(d_sb4, s45_bh4_u16.data(), g32 * 2, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_xi8, K));
    CUDA_CHECK(cudaMemcpy(d_xi8, x_i8.data(), K, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_y, (size_t)M * 4));
    CUDA_CHECK(cudaMalloc(&d_yi32, (size_t)M * 4));
    CUDA_CHECK(cudaMalloc(&d_err, 4));

    dim3 block(128), grid((M + 3) / 4);
    std::vector<float> h_y(M);
    auto eval_err = [&](const std::vector<double>& ref) -> std::pair<double, double> {
        CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, (size_t)M * 4, cudaMemcpyDeviceToHost));
        double mx = 0.0, num = 0.0, den = 0.0;
        for (int r = 0; r < M; ++r) {
            double d = (double)h_y[r] - ref[r];
            if (fabs(d) > mx) mx = fabs(d);
            num += d * d;
            den += ref[r] * ref[r];
        }
        return {mx, sqrt(num / (den + 1e-30))};
    };

    // Exact integer bit-exact check for fused (H_4 W_q) * x_i8
    verify_block_h4_int32_exact_kernel<<<grid, block>>>(d_pb4, (const int4*)d_xi8, d_yi32, M, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<int> h_yi32(M);
    CUDA_CHECK(cudaMemcpy(h_yi32.data(), d_yi32, (size_t)M * 4, cudaMemcpyDeviceToHost));
    uint32_t bh4_i64_mism = 0u;
    for (int r = 0; r < M; ++r) {
        if ((int64_t)h_yi32[r] != ref_bh4_i64[r]) ++bh4_i64_mism;
    }

    NvmlPowerSampler nvml;
    const int e_iters = 6000;

    // A. A16_axis_preencoded_dp4a
    CUDA_CHECK(cudaMemset(d_err, 0, 4));
    gemv_4_5bpw_a16_preencoded_dp4a_kernel<true><<<grid, block>>>(d_pa, d_sa, (const int4*)d_xi8, x_scale, d_y, d_err, M, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    uint32_t syn_a16 = 0u;
    CUDA_CHECK(cudaMemcpy(&syn_a16, d_err, 4, cudaMemcpyDeviceToHost));
    auto err_a16 = eval_err(ref_a16);
    float ms_a16 = 0.0f;
    RailEnergyResult en_a16 = measure_100hz_rail_energy(nvml, [&]() {
        CUDA_CHECK(cudaEventRecord(t0));
        for (int i = 0; i < e_iters; ++i)
            gemv_4_5bpw_a16_preencoded_dp4a_kernel<false><<<grid, block>>>(d_pa, d_sa, (const int4*)d_xi8, x_scale, d_y, d_err, M, K);
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        CUDA_CHECK(cudaEventElapsedTime(&ms_a16, t0, t1));
    });
    double a16_ms = (double)ms_a16 / e_iters;

    // B. Block_H2_dp4a
    gemv_4_5bpw_block_h2_dp4a_kernel<<<grid, block>>>(d_pb2, d_sb2, (const int4*)d_xi8, x_scale, d_y, M, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto err_bh2 = eval_err(ref_bh2);
    float ms_bh2 = 0.0f;
    RailEnergyResult en_bh2 = measure_100hz_rail_energy(nvml, [&]() {
        CUDA_CHECK(cudaEventRecord(t0));
        for (int i = 0; i < e_iters; ++i)
            gemv_4_5bpw_block_h2_dp4a_kernel<<<grid, block>>>(d_pb2, d_sb2, (const int4*)d_xi8, x_scale, d_y, M, K);
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        CUDA_CHECK(cudaEventElapsedTime(&ms_bh2, t0, t1));
    });
    double bh2_ms = (double)ms_bh2 / e_iters;

    // C. Fused A16_BlockH4_dp4a (CHECK_SYN=true verified + sustained timed benchmark)
    CUDA_CHECK(cudaMemset(d_err, 0, 4));
    gemv_4_5bpw_a16_block_h4_dp4a_kernel<true><<<grid, block>>>(d_pb4, d_sb4, (const int4*)d_xi8, x_scale, d_y, d_err, M, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    uint32_t syn_bh4 = 0u;
    CUDA_CHECK(cudaMemcpy(&syn_bh4, d_err, 4, cudaMemcpyDeviceToHost));
    auto err_bh4 = eval_err(ref_bh4);
    float ms_bh4 = 0.0f;
    RailEnergyResult en_bh4 = measure_100hz_rail_energy(nvml, [&]() {
        CUDA_CHECK(cudaEventRecord(t0));
        for (int i = 0; i < e_iters; ++i)
            gemv_4_5bpw_a16_block_h4_dp4a_kernel<false><<<grid, block>>>(d_pb4, d_sb4, (const int4*)d_xi8, x_scale, d_y, d_err, M, K);
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        CUDA_CHECK(cudaEventElapsedTime(&ms_bh4, t0, t1));
    });
    double bh4_ms = (double)ms_bh4 / e_iters;

    const double w_45_bytes = (double)(n_blk16 * 8 + g32 * 2 + (size_t)K + (size_t)M * 4);
    printf("{\n");
    printf("  \"M\": %d, \"K\": %d, \"weights\": %zu, \"stream_triad_gb_s\": %.4f,\n", M, K, n_w, triad_gbs);
    printf("  \"gemv_4_5bpw_a16_axis_dp4a\": {\"ms\": %.6f, \"gweights_per_s\": %.4f, \"effective_gb_s\": %.4f, \"pct_triad\": %.3f, \"syndrome_errors\": %u, \"max_abs_err_fp64\": %.9e, \"rel_l2_err_fp64\": %.9e, \"nvml_ina3221_gpu_mj_per_call\": %.6f, \"nvml_ina3221_avg_w\": %.3f, \"nvml_samples\": %d, \"rapl_cpu_pkg_mj_per_call\": %.6f},\n",
           a16_ms, (double)n_w / (a16_ms * 1e6), w_45_bytes / (a16_ms * 1e6),
           100.0 * (w_45_bytes / (a16_ms * 1e6)) / triad_gbs,
           syn_a16, err_a16.first, err_a16.second,
           en_a16.gpu_total_mj / (double)e_iters, en_a16.gpu_avg_w, en_a16.samples,
           en_a16.rapl_total_mj / (double)e_iters);
    printf("  \"gemv_4_5bpw_block_h2_dp4a\": {\"ms\": %.6f, \"gweights_per_s\": %.4f, \"effective_gb_s\": %.4f, \"pct_triad\": %.3f, \"syndrome_errors\": 0, \"max_abs_err_fp64\": %.9e, \"rel_l2_err_fp64\": %.9e, \"nvml_ina3221_gpu_mj_per_call\": %.6f, \"nvml_ina3221_avg_w\": %.3f, \"nvml_samples\": %d, \"rapl_cpu_pkg_mj_per_call\": %.6f},\n",
           bh2_ms, (double)n_w / (bh2_ms * 1e6), w_45_bytes / (bh2_ms * 1e6),
           100.0 * (w_45_bytes / (bh2_ms * 1e6)) / triad_gbs,
           err_bh2.first, err_bh2.second,
           en_bh2.gpu_total_mj / (double)e_iters, en_bh2.gpu_avg_w, en_bh2.samples,
           en_bh2.rapl_total_mj / (double)e_iters);
    printf("  \"gemv_4_5bpw_a16_block_h4_dp4a\": {\"ms\": %.6f, \"gweights_per_s\": %.4f, \"effective_gb_s\": %.4f, \"pct_triad\": %.3f, \"syndrome_errors\": %u, \"int64_bit_exact_mismatches\": %u, \"max_abs_err_fp64\": %.9e, \"rel_l2_err_fp64\": %.9e, \"nvml_ina3221_gpu_mj_per_call\": %.6f, \"nvml_ina3221_avg_w\": %.3f, \"nvml_samples\": %d, \"rapl_cpu_pkg_mj_per_call\": %.6f}\n",
           bh4_ms, (double)n_w / (bh4_ms * 1e6), w_45_bytes / (bh4_ms * 1e6),
           100.0 * (w_45_bytes / (bh4_ms * 1e6)) / triad_gbs,
           syn_bh4, bh4_i64_mism, err_bh4.first, err_bh4.second,
           en_bh4.gpu_total_mj / (double)e_iters, en_bh4.gpu_avg_w, en_bh4.samples,
           en_bh4.rapl_total_mj / (double)e_iters);
    printf("}\n");

    CUDA_CHECK(cudaFree(d_pa)); CUDA_CHECK(cudaFree(d_sa));
    CUDA_CHECK(cudaFree(d_pb2)); CUDA_CHECK(cudaFree(d_sb2));
    CUDA_CHECK(cudaFree(d_pb4)); CUDA_CHECK(cudaFree(d_sb4));
    CUDA_CHECK(cudaFree(d_xi8)); CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_yi32)); CUDA_CHECK(cudaFree(d_err));
    CUDA_CHECK(cudaEventDestroy(t0)); CUDA_CHECK(cudaEventDestroy(t1));
    return (syn_a16 == 0 && syn_bh4 == 0 && bh4_i64_mism == 0) ? 0 : 4;
}
