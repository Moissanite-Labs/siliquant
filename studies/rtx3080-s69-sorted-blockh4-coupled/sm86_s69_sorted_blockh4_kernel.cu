// Track S69: Bare-Metal sm_86 PTX Zero-Cost Folded vs Shared-RMSNorm O(d) Fused Gather A16_BlockH4 Benchmark
// Zero-spill (ptxas 0 B stack / 0 B spill, 0 PTX ld.local/st.local) on REAL Qwen2.5-3B weights:
//   Mode "DOWN": down_proj stacked2 (4096x11008) — A16_axis vs unpermuted A16_BlockH4 vs
//                offline-folded zero-cost sorted A16_BlockH4 (0 runtime gather ops, bit-exact parity!)
//   Mode "MLP":  gate_proj + up_proj stacked (22016x2048) —
//                (1) unpermuted RMSNorm+quant_i8 + A16_BlockH4 GEMV
//                (2) shared-RMSNorm O(d) fused gather+quant_i8 + contiguous A16_BlockH4 GEMV
//                (3) unpermuted RMSNorm+quant_i8 + naive inline-gather A16_BlockH4 GEMV
//   Plus live 100 Hz trapezoidal NVML INA3221 rail power + intel-rapl:0 CPU mJ/call integration!

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

__device__ __forceinline__ float warp_reduce_max_f32(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float o = __shfl_down_sync(0xFFFFFFFFu, v, off);
        v = (o > v) ? o : v;
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
// 1. STREAM Triad Baseline
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
// 2. Shared-RMSNorm + Quantize-to-int8 (With Optional O(d) Fused Gather!)
//    1 CTA of 256 threads transforms a d=2048 hidden vector ONCE for all q/k/v or gate/up!
// ---------------------------------------------------------------------------
template <bool USE_PERM>
__global__ void rmsnorm_quant_i8_kernel(
    const float* __restrict__ x_in,
    const float* __restrict__ gamma,
    const uint16_t* __restrict__ perm,
    int8_t* __restrict__ x_out_i8,
    float* __restrict__ out_scale,
    float eps,
    int K) {
    __shared__ float sm_red[8];
    __shared__ float sm_inv_rms;
    __shared__ float sm_inv_sc;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int wid = tid >> 5;

    float sum_sq = 0.0f;
    for (int i = tid; i < K; i += blockDim.x) {
        float v = __ldg(x_in + i);
        sum_sq = fmaf(v, v, sum_sq);
    }
    sum_sq = warp_reduce_sum_f32(sum_sq);
    if (lane == 0) sm_red[wid] = sum_sq;
    __syncthreads();
    if (wid == 0) {
        float tot = (lane < 8) ? sm_red[lane] : 0.0f;
        tot = warp_reduce_sum_f32(tot);
        if (lane == 0) sm_inv_rms = rsqrtf(tot / (float)K + eps);
    }
    __syncthreads();
    const float inv_rms = sm_inv_rms;

    float amax = 1e-8f;
    for (int i = tid; i < K; i += blockDim.x) {
        float yv = __ldg(x_in + i) * inv_rms * __ldg(gamma + i);
        float av = fabsf(yv);
        if (av > amax) amax = av;
    }
    amax = warp_reduce_max_f32(amax);
    if (lane == 0) sm_red[wid] = amax;
    __syncthreads();
    if (wid == 0) {
        float mx = (lane < 8) ? sm_red[lane] : 1e-8f;
        mx = warp_reduce_max_f32(mx);
        if (lane == 0) {
            float sc = mx / 127.0f;
            *out_scale = sc;
            sm_inv_sc = 127.0f / mx;
        }
    }
    __syncthreads();
    const float inv_sc = sm_inv_sc;

    for (int j = tid; j < K; j += blockDim.x) {
        int src = USE_PERM ? (int)__ldg(perm + j) : j;
        float yv = __ldg(x_in + src) * inv_rms * __ldg(gamma + src);
        int q = __float2int_rn(yv * inv_sc);
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        x_out_i8[j] = (int8_t)q;
    }
}

// ---------------------------------------------------------------------------
// 3. Direct 4.5000-bpw A16_axis_preencoded_dp4a Baseline
// ---------------------------------------------------------------------------
template <bool CHECK_SYN>
__global__ void gemv_4_5bpw_a16_preencoded_dp4a_kernel(
    const uint64_t* __restrict__ packed,
    const __half* __restrict__ scales,
    const int4* __restrict__ x_i8,
    const float* __restrict__ x_scale_ptr,
    float* __restrict__ y,
    uint32_t* __restrict__ err_count,
    int M, int K) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp_id >= M) return;
    const int n_blk16 = K >> 4;
    const uint64_t* row_p = packed + (size_t)warp_id * n_blk16;
    const __half* row_s = scales + (size_t)warp_id * (K >> 5);
    const float x_scale = __ldg(x_scale_ptr);
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
// 4. Direct 4.5000-bpw Fused Register-Only A16_BlockH4 PTX dp4a GEMV
//    Used identically for unpermuted, zero-cost down_proj-folded, & shared-RMSNorm permuted!
// ---------------------------------------------------------------------------
template <bool CHECK_SYN>
__global__ void gemv_4_5bpw_a16_block_h4_dp4a_kernel(
    const uint64_t* __restrict__ packed,
    const __half* __restrict__ scales,
    const int4* __restrict__ x_i8,
    const float* __restrict__ x_scale_ptr,
    float* __restrict__ y,
    uint32_t* __restrict__ err_count,
    int M, int K) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp_id >= M) return;
    const int n_blk16 = K >> 4;
    const uint64_t* row_p = packed + (size_t)warp_id * n_blk16;
    const __half* row_s = scales + (size_t)warp_id * (K >> 5);
    const float x_half = __ldg(x_scale_ptr) * 0.5f;
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
// 5. Naive Inline Per-GEMV Gather Control for A16_BlockH4 (shows why shared RMSNorm wins!)
// ---------------------------------------------------------------------------
__global__ void gemv_4_5bpw_a16_block_h4_inline_gather_dp4a_kernel(
    const uint64_t* __restrict__ packed,
    const __half* __restrict__ scales,
    const int8_t* __restrict__ x_i8,
    const uint16_t* __restrict__ perm,
    const float* __restrict__ x_scale_ptr,
    float* __restrict__ y,
    int M, int K) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp_id >= M) return;
    const int n_blk16 = K >> 4;
    const uint64_t* row_p = packed + (size_t)warp_id * n_blk16;
    const __half* row_s = scales + (size_t)warp_id * (K >> 5);
    const float x_half = __ldg(x_scale_ptr) * 0.5f;
    float acc = 0.0f;
    for (int b = lane; b < n_blk16; b += 32) {
        uint64_t w64 = __ldg(row_p + b);
        const int base = b << 4;
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
            int c0 = base + (g << 2);
            int x0 = (int)__ldg(x_i8 + __ldg(perm + c0));
            int x1 = (int)__ldg(x_i8 + __ldg(perm + c0 + 1));
            int x2 = (int)__ldg(x_i8 + __ldg(perm + c0 + 2));
            int x3 = (int)__ldg(x_i8 + __ldg(perm + c0 + 3));
            int xp = pack4_s8(x0, x1, x2, x3);
            idot = dp4a_s32_ptx(wp, xp, idot);
        }
        float sc = __half2float(__ldg(row_s + (b >> 1))) * x_half;
        acc = fmaf((float)idot, sc, acc);
    }
    acc = warp_reduce_sum_f32(acc);
    if (lane == 0) y[warp_id] = acc;
}

// ---------------------------------------------------------------------------
// 6. Exact Unscaled int32 Bit-Exact Parity Kernel for Fused (H_4 W_q) * x_i8
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

static std::vector<uint64_t> pack_nibbles(const std::vector<int8_t>& y, size_t n_blk16) {
    std::vector<uint64_t> out(n_blk16, 0ULL);
    for (size_t b = 0; b < n_blk16; ++b) {
        uint64_t w = 0ULL;
        for (int j = 0; j < 16; ++j) w |= ((uint64_t)(y[b * 16 + j] & 0xF)) << (j * 4);
        out[b] = w;
    }
    return out;
}

static int run_down_bench(std::ifstream& ifs, cudaEvent_t t0, cudaEvent_t t1, double triad_gbs) {
    int32_t M = 0, K = 0;
    ifs.read(reinterpret_cast<char*>(&M), 4);
    ifs.read(reinterpret_cast<char*>(&K), 4);
    const size_t n_w = (size_t)M * (size_t)K;
    const size_t n_blk16 = n_w / 16;
    const size_t g32 = n_w / 32;

    std::vector<int8_t> y_ax(n_w), y_bh4(n_w), y_bh4_sort(n_w);
    std::vector<uint16_t> s_ax(g32), s_bh4(g32), s_bh4_sort(g32), perm(K);
    std::vector<float> x_real(K);
    ifs.read(reinterpret_cast<char*>(y_ax.data()), (std::streamsize)n_w);
    ifs.read(reinterpret_cast<char*>(s_ax.data()), (std::streamsize)(g32 * 2));
    ifs.read(reinterpret_cast<char*>(y_bh4.data()), (std::streamsize)n_w);
    ifs.read(reinterpret_cast<char*>(s_bh4.data()), (std::streamsize)(g32 * 2));
    ifs.read(reinterpret_cast<char*>(y_bh4_sort.data()), (std::streamsize)n_w);
    ifs.read(reinterpret_cast<char*>(s_bh4_sort.data()), (std::streamsize)(g32 * 2));
    ifs.read(reinterpret_cast<char*>(perm.data()), (std::streamsize)(K * 2));
    ifs.read(reinterpret_cast<char*>(x_real.data()), (std::streamsize)(K * 4));
    if (!ifs) { fprintf(stderr, "Truncated DOWN bundle\n"); return 2; }

    float x_amax = 1e-8f;
    for (int j = 0; j < K; ++j) if (fabsf(x_real[j]) > x_amax) x_amax = fabsf(x_real[j]);
    float x_scale = x_amax / 127.0f;
    std::vector<int8_t> x_i8(K), x_perm_i8(K);
    for (int j = 0; j < K; ++j) {
        int q = (int)llround((double)x_real[j] / (double)x_scale);
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        x_i8[j] = (int8_t)q;
    }
    for (int j = 0; j < K; ++j) x_perm_i8[j] = x_i8[perm[j]];

    // Exact CPU int64 + FP64 references
    std::vector<double> ref_ax(M, 0.0), ref_bh4(M, 0.0), ref_sort(M, 0.0);
    std::vector<int64_t> ref_sort_i64(M, 0LL);
    for (int r = 0; r < M; ++r) {
        double aa = 0.0, ab = 0.0, as = 0.0;
        int64_t is64 = 0LL;
        for (int c = 0; c < K; ++c) {
            size_t idx = (size_t)r * K + c;
            double sc = (double)__half2float(*reinterpret_cast<const __half*>(&s_ax[idx / 32]));
            aa += sc * (double)y_ax[idx] * (double)x_i8[c] * (double)x_scale;
        }
        for (int c = 0; c < K; c += 4) {
            size_t idx = (size_t)r * K + c;
            int64_t a0 = y_bh4[idx], a1 = y_bh4[idx + 1], a2 = y_bh4[idx + 2], a3 = y_bh4[idx + 3];
            int64_t d4 = (a0 + a1 + a2 + a3) * x_i8[c] + (a0 - a1 + a2 - a3) * x_i8[c + 1]
                       + (a0 + a1 - a2 - a3) * x_i8[c + 2] + (a0 - a1 - a2 + a3) * x_i8[c + 3];
            double sc = (double)__half2float(*reinterpret_cast<const __half*>(&s_bh4[idx / 32])) * 0.5;
            ab += sc * (double)d4 * (double)x_scale;

            int64_t b0 = y_bh4_sort[idx], b1 = y_bh4_sort[idx + 1], b2 = y_bh4_sort[idx + 2], b3 = y_bh4_sort[idx + 3];
            int64_t ds4 = (b0 + b1 + b2 + b3) * x_perm_i8[c] + (b0 - b1 + b2 - b3) * x_perm_i8[c + 1]
                        + (b0 + b1 - b2 - b3) * x_perm_i8[c + 2] + (b0 - b1 - b2 + b3) * x_perm_i8[c + 3];
            is64 += ds4;
            double scs = (double)__half2float(*reinterpret_cast<const __half*>(&s_bh4_sort[idx / 32])) * 0.5;
            as += scs * (double)ds4 * (double)x_scale;
        }
        ref_ax[r] = aa; ref_bh4[r] = ab; ref_sort[r] = as; ref_sort_i64[r] = is64;
    }

    auto pa = pack_nibbles(y_ax, n_blk16);
    auto pb = pack_nibbles(y_bh4, n_blk16);
    auto ps = pack_nibbles(y_bh4_sort, n_blk16);

    uint64_t *d_pa = nullptr, *d_pb = nullptr, *d_ps = nullptr;
    __half *d_sa = nullptr, *d_sb = nullptr, *d_ss = nullptr;
    int8_t *d_xi = nullptr, *d_xp = nullptr;
    float *d_xsc = nullptr, *d_y = nullptr;
    int* d_yi32 = nullptr;
    uint32_t* d_err = nullptr;

    CUDA_CHECK(cudaMalloc(&d_pa, n_blk16 * 8)); CUDA_CHECK(cudaMemcpy(d_pa, pa.data(), n_blk16 * 8, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_sa, g32 * 2)); CUDA_CHECK(cudaMemcpy(d_sa, s_ax.data(), g32 * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_pb, n_blk16 * 8)); CUDA_CHECK(cudaMemcpy(d_pb, pb.data(), n_blk16 * 8, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_sb, g32 * 2)); CUDA_CHECK(cudaMemcpy(d_sb, s_bh4.data(), g32 * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_ps, n_blk16 * 8)); CUDA_CHECK(cudaMemcpy(d_ps, ps.data(), n_blk16 * 8, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_ss, g32 * 2)); CUDA_CHECK(cudaMemcpy(d_ss, s_bh4_sort.data(), g32 * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_xi, K)); CUDA_CHECK(cudaMemcpy(d_xi, x_i8.data(), K, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_xp, K)); CUDA_CHECK(cudaMemcpy(d_xp, x_perm_i8.data(), K, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_xsc, 4)); CUDA_CHECK(cudaMemcpy(d_xsc, &x_scale, 4, cudaMemcpyHostToDevice));
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
            num += d * d; den += ref[r] * ref[r];
        }
        return {mx, sqrt(num / (den + 1e-30))};
    };

    // Exact int32 check on offline-folded zero-cost sorted A16_BlockH4
    verify_block_h4_int32_exact_kernel<<<grid, block>>>(d_ps, (const int4*)d_xp, d_yi32, M, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<int> h_i32(M);
    CUDA_CHECK(cudaMemcpy(h_i32.data(), d_yi32, (size_t)M * 4, cudaMemcpyDeviceToHost));
    uint32_t sort_i64_mism = 0u;
    for (int r = 0; r < M; ++r) if ((int64_t)h_i32[r] != ref_sort_i64[r]) ++sort_i64_mism;

    // Warmup GPU to steady P0 clock & rail power before timing!
    for (int i = 0; i < 3000; ++i)
        gemv_4_5bpw_a16_block_h4_dp4a_kernel<false><<<grid, block>>>(d_pb, d_sb, (const int4*)d_xi, d_xsc, d_y, d_err, M, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    NvmlPowerSampler nvml;
    const int e_iters = 6000;

    // 1. A16_axis
    CUDA_CHECK(cudaMemset(d_err, 0, 4));
    gemv_4_5bpw_a16_preencoded_dp4a_kernel<true><<<grid, block>>>(d_pa, d_sa, (const int4*)d_xi, d_xsc, d_y, d_err, M, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    uint32_t syn_ax = 0u; CUDA_CHECK(cudaMemcpy(&syn_ax, d_err, 4, cudaMemcpyDeviceToHost));
    auto err_ax = eval_err(ref_ax);
    float ms_ax = 0.0f;
    RailEnergyResult en_ax = measure_100hz_rail_energy(nvml, [&]() {
        CUDA_CHECK(cudaEventRecord(t0));
        for (int i = 0; i < e_iters; ++i)
            gemv_4_5bpw_a16_preencoded_dp4a_kernel<false><<<grid, block>>>(d_pa, d_sa, (const int4*)d_xi, d_xsc, d_y, d_err, M, K);
        CUDA_CHECK(cudaEventRecord(t1)); CUDA_CHECK(cudaEventSynchronize(t1));
        CUDA_CHECK(cudaEventElapsedTime(&ms_ax, t0, t1));
    });

    // 2. Unpermuted A16_BlockH4
    CUDA_CHECK(cudaMemset(d_err, 0, 4));
    gemv_4_5bpw_a16_block_h4_dp4a_kernel<true><<<grid, block>>>(d_pb, d_sb, (const int4*)d_xi, d_xsc, d_y, d_err, M, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    uint32_t syn_b4 = 0u; CUDA_CHECK(cudaMemcpy(&syn_b4, d_err, 4, cudaMemcpyDeviceToHost));
    auto err_b4 = eval_err(ref_bh4);
    float ms_b4 = 0.0f;
    RailEnergyResult en_b4 = measure_100hz_rail_energy(nvml, [&]() {
        CUDA_CHECK(cudaEventRecord(t0));
        for (int i = 0; i < e_iters; ++i)
            gemv_4_5bpw_a16_block_h4_dp4a_kernel<false><<<grid, block>>>(d_pb, d_sb, (const int4*)d_xi, d_xsc, d_y, d_err, M, K);
        CUDA_CHECK(cudaEventRecord(t1)); CUDA_CHECK(cudaEventSynchronize(t1));
        CUDA_CHECK(cudaEventElapsedTime(&ms_b4, t0, t1));
    });

    // 3. Offline-folded Zero-Cost Sorted A16_BlockH4
    CUDA_CHECK(cudaMemset(d_err, 0, 4));
    gemv_4_5bpw_a16_block_h4_dp4a_kernel<true><<<grid, block>>>(d_ps, d_ss, (const int4*)d_xp, d_xsc, d_y, d_err, M, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    uint32_t syn_so = 0u; CUDA_CHECK(cudaMemcpy(&syn_so, d_err, 4, cudaMemcpyDeviceToHost));
    auto err_so = eval_err(ref_sort);
    float ms_so = 0.0f;
    RailEnergyResult en_so = measure_100hz_rail_energy(nvml, [&]() {
        CUDA_CHECK(cudaEventRecord(t0));
        for (int i = 0; i < e_iters; ++i)
            gemv_4_5bpw_a16_block_h4_dp4a_kernel<false><<<grid, block>>>(d_ps, d_ss, (const int4*)d_xp, d_xsc, d_y, d_err, M, K);
        CUDA_CHECK(cudaEventRecord(t1)); CUDA_CHECK(cudaEventSynchronize(t1));
        CUDA_CHECK(cudaEventElapsedTime(&ms_so, t0, t1));
    });

    const double w_45_bytes = (double)(n_blk16 * 8 + g32 * 2 + (size_t)K + (size_t)M * 4);
    double ax_ms = (double)ms_ax / e_iters, b4_ms = (double)ms_b4 / e_iters, so_ms = (double)ms_so / e_iters;
    printf("{\n");
    printf("  \"mode\": \"DOWN\", \"M\": %d, \"K\": %d, \"weights\": %zu, \"stream_triad_gb_s\": %.4f,\n", M, K, n_w, triad_gbs);
    printf("  \"a16_axis\": {\"ms\": %.6f, \"gweights_per_s\": %.4f, \"effective_gb_s\": %.4f, \"pct_triad\": %.3f, \"syndrome_errors\": %u, \"rel_l2_err_fp64\": %.9e, \"nvml_ina3221_gpu_mj_per_call\": %.6f, \"nvml_ina3221_avg_w\": %.3f, \"rapl_cpu_pkg_mj_per_call\": %.6f},\n",
           ax_ms, (double)n_w / (ax_ms * 1e6), w_45_bytes / (ax_ms * 1e6), 100.0 * (w_45_bytes / (ax_ms * 1e6)) / triad_gbs,
           syn_ax, err_ax.second, en_ax.gpu_total_mj / e_iters, en_ax.gpu_avg_w, en_ax.rapl_total_mj / e_iters);
    printf("  \"a16_block_h4_unpermuted\": {\"ms\": %.6f, \"gweights_per_s\": %.4f, \"effective_gb_s\": %.4f, \"pct_triad\": %.3f, \"syndrome_errors\": %u, \"rel_l2_err_fp64\": %.9e, \"nvml_ina3221_gpu_mj_per_call\": %.6f, \"nvml_ina3221_avg_w\": %.3f, \"rapl_cpu_pkg_mj_per_call\": %.6f},\n",
           b4_ms, (double)n_w / (b4_ms * 1e6), w_45_bytes / (b4_ms * 1e6), 100.0 * (w_45_bytes / (b4_ms * 1e6)) / triad_gbs,
           syn_b4, err_b4.second, en_b4.gpu_total_mj / e_iters, en_b4.gpu_avg_w, en_b4.rapl_total_mj / e_iters);
    printf("  \"a16_block_h4_zerocost_folded\": {\"ms\": %.6f, \"gweights_per_s\": %.4f, \"effective_gb_s\": %.4f, \"pct_triad\": %.3f, \"syndrome_errors\": %u, \"int64_bit_exact_mismatches\": %u, \"rel_l2_err_fp64\": %.9e, \"nvml_ina3221_gpu_mj_per_call\": %.6f, \"nvml_ina3221_avg_w\": %.3f, \"rapl_cpu_pkg_mj_per_call\": %.6f}\n",
           so_ms, (double)n_w / (so_ms * 1e6), w_45_bytes / (so_ms * 1e6), 100.0 * (w_45_bytes / (so_ms * 1e6)) / triad_gbs,
           syn_so, sort_i64_mism, err_so.second, en_so.gpu_total_mj / e_iters, en_so.gpu_avg_w, en_so.rapl_total_mj / e_iters);
    printf("}\n");

    CUDA_CHECK(cudaFree(d_pa)); CUDA_CHECK(cudaFree(d_sa));
    CUDA_CHECK(cudaFree(d_pb)); CUDA_CHECK(cudaFree(d_sb));
    CUDA_CHECK(cudaFree(d_ps)); CUDA_CHECK(cudaFree(d_ss));
    CUDA_CHECK(cudaFree(d_xi)); CUDA_CHECK(cudaFree(d_xp));
    CUDA_CHECK(cudaFree(d_xsc)); CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_yi32)); CUDA_CHECK(cudaFree(d_err));
    return (syn_ax == 0 && syn_b4 == 0 && syn_so == 0 && sort_i64_mism == 0) ? 0 : 4;
}

static int run_mlp_gather_bench(std::ifstream& ifs, cudaEvent_t t0, cudaEvent_t t1, double triad_gbs) {
    int32_t M = 0, K = 0;
    float eps = 1e-6f;
    ifs.read(reinterpret_cast<char*>(&M), 4);
    ifs.read(reinterpret_cast<char*>(&K), 4);
    ifs.read(reinterpret_cast<char*>(&eps), 4);
    const size_t n_w = (size_t)M * (size_t)K;
    const size_t n_blk16 = n_w / 16;
    const size_t g32 = n_w / 32;

    std::vector<int8_t> y_unp(n_w), y_sort(n_w);
    std::vector<uint16_t> s_unp(g32), s_sort(g32), perm(K);
    std::vector<float> x_raw(K), gamma(K);
    ifs.read(reinterpret_cast<char*>(y_unp.data()), (std::streamsize)n_w);
    ifs.read(reinterpret_cast<char*>(s_unp.data()), (std::streamsize)(g32 * 2));
    ifs.read(reinterpret_cast<char*>(y_sort.data()), (std::streamsize)n_w);
    ifs.read(reinterpret_cast<char*>(s_sort.data()), (std::streamsize)(g32 * 2));
    ifs.read(reinterpret_cast<char*>(perm.data()), (std::streamsize)(K * 2));
    ifs.read(reinterpret_cast<char*>(x_raw.data()), (std::streamsize)(K * 4));
    ifs.read(reinterpret_cast<char*>(gamma.data()), (std::streamsize)(K * 4));
    if (!ifs) { fprintf(stderr, "Truncated MLP bundle\n"); return 2; }

    auto pu = pack_nibbles(y_unp, n_blk16);
    auto ps = pack_nibbles(y_sort, n_blk16);

    uint64_t *d_pu = nullptr, *d_ps = nullptr;
    __half *d_su = nullptr, *d_ss = nullptr;
    uint16_t* d_perm = nullptr;
    float *d_xraw = nullptr, *d_gam = nullptr, *d_xsc = nullptr, *d_y = nullptr;
    int8_t *d_xi_unp = nullptr, *d_xi_perm = nullptr;
    int* d_yi32 = nullptr;
    uint32_t* d_err = nullptr;

    CUDA_CHECK(cudaMalloc(&d_pu, n_blk16 * 8)); CUDA_CHECK(cudaMemcpy(d_pu, pu.data(), n_blk16 * 8, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_su, g32 * 2)); CUDA_CHECK(cudaMemcpy(d_su, s_unp.data(), g32 * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_ps, n_blk16 * 8)); CUDA_CHECK(cudaMemcpy(d_ps, ps.data(), n_blk16 * 8, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_ss, g32 * 2)); CUDA_CHECK(cudaMemcpy(d_ss, s_sort.data(), g32 * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_perm, K * 2)); CUDA_CHECK(cudaMemcpy(d_perm, perm.data(), K * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_xraw, K * 4)); CUDA_CHECK(cudaMemcpy(d_xraw, x_raw.data(), K * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_gam, K * 4)); CUDA_CHECK(cudaMemcpy(d_gam, gamma.data(), K * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_xsc, 4));
    CUDA_CHECK(cudaMalloc(&d_xi_unp, K));
    CUDA_CHECK(cudaMalloc(&d_xi_perm, K));
    CUDA_CHECK(cudaMalloc(&d_y, (size_t)M * 4));
    CUDA_CHECK(cudaMalloc(&d_yi32, (size_t)M * 4));
    CUDA_CHECK(cudaMalloc(&d_err, 4));

    dim3 block(128), grid((M + 3) / 4);

    // Run both RMSNorm variants and verify exact parity between shared-gather & inline-gather outputs!
    rmsnorm_quant_i8_kernel<false><<<1, 256>>>(d_xraw, d_gam, d_perm, d_xi_unp, d_xsc, eps, K);
    rmsnorm_quant_i8_kernel<true><<<1, 256>>>(d_xraw, d_gam, d_perm, d_xi_perm, d_xsc, eps, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int8_t> h_xi_unp(K), h_xi_perm(K);
    CUDA_CHECK(cudaMemcpy(h_xi_unp.data(), d_xi_unp, K, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_xi_perm.data(), d_xi_perm, K, cudaMemcpyDeviceToHost));
    uint32_t norm_perm_mism = 0u;
    for (int j = 0; j < K; ++j) if (h_xi_perm[j] != h_xi_unp[perm[j]]) ++norm_perm_mism;

    // Exact int64 check on shared-RMSNorm permuted GEMV
    verify_block_h4_int32_exact_kernel<<<grid, block>>>(d_ps, (const int4*)d_xi_perm, d_yi32, M, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<int> h_i32(M);
    CUDA_CHECK(cudaMemcpy(h_i32.data(), d_yi32, (size_t)M * 4, cudaMemcpyDeviceToHost));
    uint32_t mlp_i64_mism = 0u;
    for (int r = 0; r < M; ++r) {
        int64_t acc = 0LL;
        for (int c = 0; c < K; c += 4) {
            size_t idx = (size_t)r * K + c;
            int64_t b0 = y_sort[idx], b1 = y_sort[idx + 1], b2 = y_sort[idx + 2], b3 = y_sort[idx + 3];
            acc += (b0 + b1 + b2 + b3) * h_xi_perm[c] + (b0 - b1 + b2 - b3) * h_xi_perm[c + 1]
                 + (b0 + b1 - b2 - b3) * h_xi_perm[c + 2] + (b0 - b1 - b2 + b3) * h_xi_perm[c + 3];
        }
        if ((int64_t)h_i32[r] != acc) ++mlp_i64_mism;
    }

    // Verify shared-RMSNorm + contiguous BlockH4 == unpermuted RMSNorm + inline-gather BlockH4 bit-for-bit!
    CUDA_CHECK(cudaMemset(d_err, 0, 4));
    gemv_4_5bpw_a16_block_h4_dp4a_kernel<true><<<grid, block>>>(d_ps, d_ss, (const int4*)d_xi_perm, d_xsc, d_y, d_err, M, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    uint32_t syn_so = 0u; CUDA_CHECK(cudaMemcpy(&syn_so, d_err, 4, cudaMemcpyDeviceToHost));
    std::vector<float> h_y_shared(M), h_y_inline(M);
    CUDA_CHECK(cudaMemcpy(h_y_shared.data(), d_y, (size_t)M * 4, cudaMemcpyDeviceToHost));

    gemv_4_5bpw_a16_block_h4_inline_gather_dp4a_kernel<<<grid, block>>>(d_ps, d_ss, d_xi_unp, d_perm, d_xsc, d_y, M, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_y_inline.data(), d_y, (size_t)M * 4, cudaMemcpyDeviceToHost));
    double max_shared_vs_inline_diff = 0.0;
    for (int r = 0; r < M; ++r) {
        double d = fabs((double)h_y_shared[r] - (double)h_y_inline[r]);
        if (d > max_shared_vs_inline_diff) max_shared_vs_inline_diff = d;
    }

    // Warmup GPU
    for (int i = 0; i < 3000; ++i) {
        rmsnorm_quant_i8_kernel<false><<<1, 256>>>(d_xraw, d_gam, d_perm, d_xi_unp, d_xsc, eps, K);
        gemv_4_5bpw_a16_block_h4_dp4a_kernel<false><<<grid, block>>>(d_pu, d_su, (const int4*)d_xi_unp, d_xsc, d_y, d_err, M, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    NvmlPowerSampler nvml;
    const int e_iters = 6000;

    // 1. Baseline: Unpermuted RMSNorm + Unpermuted A16_BlockH4 GEMV
    float ms_unp = 0.0f;
    RailEnergyResult en_unp = measure_100hz_rail_energy(nvml, [&]() {
        CUDA_CHECK(cudaEventRecord(t0));
        for (int i = 0; i < e_iters; ++i) {
            rmsnorm_quant_i8_kernel<false><<<1, 256>>>(d_xraw, d_gam, d_perm, d_xi_unp, d_xsc, eps, K);
            gemv_4_5bpw_a16_block_h4_dp4a_kernel<false><<<grid, block>>>(d_pu, d_su, (const int4*)d_xi_unp, d_xsc, d_y, d_err, M, K);
        }
        CUDA_CHECK(cudaEventRecord(t1)); CUDA_CHECK(cudaEventSynchronize(t1));
        CUDA_CHECK(cudaEventElapsedTime(&ms_unp, t0, t1));
    });

    // 2. Shared-RMSNorm O(d) Fused Gather + Sorted Contiguous A16_BlockH4 GEMV
    float ms_sh = 0.0f;
    RailEnergyResult en_sh = measure_100hz_rail_energy(nvml, [&]() {
        CUDA_CHECK(cudaEventRecord(t0));
        for (int i = 0; i < e_iters; ++i) {
            rmsnorm_quant_i8_kernel<true><<<1, 256>>>(d_xraw, d_gam, d_perm, d_xi_perm, d_xsc, eps, K);
            gemv_4_5bpw_a16_block_h4_dp4a_kernel<false><<<grid, block>>>(d_ps, d_ss, (const int4*)d_xi_perm, d_xsc, d_y, d_err, M, K);
        }
        CUDA_CHECK(cudaEventRecord(t1)); CUDA_CHECK(cudaEventSynchronize(t1));
        CUDA_CHECK(cudaEventElapsedTime(&ms_sh, t0, t1));
    });

    // 3. Naive Inline Per-GEMV Gather Control: Unpermuted RMSNorm + Inline-Gather A16_BlockH4 GEMV
    float ms_in = 0.0f;
    RailEnergyResult en_in = measure_100hz_rail_energy(nvml, [&]() {
        CUDA_CHECK(cudaEventRecord(t0));
        for (int i = 0; i < e_iters; ++i) {
            rmsnorm_quant_i8_kernel<false><<<1, 256>>>(d_xraw, d_gam, d_perm, d_xi_unp, d_xsc, eps, K);
            gemv_4_5bpw_a16_block_h4_inline_gather_dp4a_kernel<<<grid, block>>>(d_ps, d_ss, d_xi_unp, d_perm, d_xsc, d_y, M, K);
        }
        CUDA_CHECK(cudaEventRecord(t1)); CUDA_CHECK(cudaEventSynchronize(t1));
        CUDA_CHECK(cudaEventElapsedTime(&ms_in, t0, t1));
    });

    double u_ms = (double)ms_unp / e_iters, sh_ms = (double)ms_sh / e_iters, in_ms = (double)ms_in / e_iters;
    double sh_overhead_pct = 100.0 * (sh_ms - u_ms) / u_ms;
    double in_overhead_pct = 100.0 * (in_ms - u_ms) / u_ms;

    printf("{\n");
    printf("  \"mode\": \"MLP\", \"M\": %d, \"K\": %d, \"weights\": %zu, \"stream_triad_gb_s\": %.4f,\n", M, K, n_w, triad_gbs);
    printf("  \"parity\": {\"norm_perm_mismatches\": %u, \"syndrome_errors\": %u, \"int64_bit_exact_mismatches\": %u, \"max_shared_vs_inline_diff\": %.9e},\n",
           norm_perm_mism, syn_so, mlp_i64_mism, max_shared_vs_inline_diff);
    printf("  \"rmsnorm_unpermuted_plus_blockh4\": {\"ms\": %.6f, \"gweights_per_s\": %.4f, \"nvml_ina3221_gpu_mj_per_call\": %.6f, \"nvml_ina3221_avg_w\": %.3f, \"rapl_cpu_pkg_mj_per_call\": %.6f},\n",
           u_ms, (double)n_w / (u_ms * 1e6), en_unp.gpu_total_mj / e_iters, en_unp.gpu_avg_w, en_unp.rapl_total_mj / e_iters);
    printf("  \"rmsnorm_shared_gather_plus_blockh4\": {\"ms\": %.6f, \"gweights_per_s\": %.4f, \"overhead_pct_vs_unpermuted\": %.4f, \"nvml_ina3221_gpu_mj_per_call\": %.6f, \"nvml_ina3221_avg_w\": %.3f, \"rapl_cpu_pkg_mj_per_call\": %.6f},\n",
           sh_ms, (double)n_w / (sh_ms * 1e6), sh_overhead_pct, en_sh.gpu_total_mj / e_iters, en_sh.gpu_avg_w, en_sh.rapl_total_mj / e_iters);
    printf("  \"rmsnorm_unpermuted_plus_inline_gather_blockh4\": {\"ms\": %.6f, \"gweights_per_s\": %.4f, \"overhead_pct_vs_unpermuted\": %.4f, \"nvml_ina3221_gpu_mj_per_call\": %.6f, \"nvml_ina3221_avg_w\": %.3f, \"rapl_cpu_pkg_mj_per_call\": %.6f}\n",
           in_ms, (double)n_w / (in_ms * 1e6), in_overhead_pct, en_in.gpu_total_mj / e_iters, en_in.gpu_avg_w, en_in.rapl_total_mj / e_iters);
    printf("}\n");

    CUDA_CHECK(cudaFree(d_pu)); CUDA_CHECK(cudaFree(d_su));
    CUDA_CHECK(cudaFree(d_ps)); CUDA_CHECK(cudaFree(d_ss));
    CUDA_CHECK(cudaFree(d_perm)); CUDA_CHECK(cudaFree(d_xraw));
    CUDA_CHECK(cudaFree(d_gam)); CUDA_CHECK(cudaFree(d_xsc));
    CUDA_CHECK(cudaFree(d_xi_unp)); CUDA_CHECK(cudaFree(d_xi_perm));
    CUDA_CHECK(cudaFree(d_y)); CUDA_CHECK(cudaFree(d_yi32));
    CUDA_CHECK(cudaFree(d_err));
    return (norm_perm_mism == 0 && syn_so == 0 && mlp_i64_mism == 0 && max_shared_vs_inline_diff == 0.0) ? 0 : 4;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <s69_bundle.bin>\n", argv[0]);
        return 2;
    }
    std::ifstream ifs(argv[1], std::ios::binary);
    if (!ifs) { fprintf(stderr, "Cannot open %s\n", argv[1]); return 2; }
    char magic[4];
    ifs.read(magic, 4);

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    float ms = 0.0f;

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

    int rc = 2;
    if (memcmp(magic, "S69D", 4) == 0) {
        rc = run_down_bench(ifs, t0, t1, triad_gbs);
    } else if (memcmp(magic, "S69M", 4) == 0) {
        rc = run_mlp_gather_bench(ifs, t0, t1, triad_gbs);
    } else {
        fprintf(stderr, "Unknown bundle magic\n");
    }
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    return rc;
}
