// Bare-Metal Ampere sm_86 PTX/SASS Full 24-Layer Qwen2.5-0.5B A_16 Engine
//
// Features:
//   - Unified 4.500-bpw A_16 [16,11,4] nibble-packed format: 16 B biased u4 nibbles
//     (u = q + 8 in [1,15], preserving exact LSB Extended Hamming [16,11,4] parity)
//     + 2 B fp16 scale per 32-weight group (18 B / 32 w = 4.5000 bpw).
//   - Zero-spill Ampere sm_86 kernels (ptxas -v: 0 B stack, 0 B spill, 0 LDL/STL):
//       1. rmsnorm_q8_kernel: fused RMSNorm + per-32-group symmetric int8 quant + group sums
//       2. a16_gemv_dp4a_kernel<LIVE_VERIFY>: coalesced LDG.128 + LOP3/PRMT + dp4a.u32.s32
//          with optional live 1-LOP3 + 5-POPC Extended Hamming [16,11,4] syndrome verification
//       3. rope_kv_gqa_decode_kernel: fused RoPE (theta=1e6, rotate-half) + KV cache + 14:2 GQA
//       4. silu_mul_q8_kernel: fused SwiGLU (silu(gate)*up) + per-32-group int8 quant
//       5. residual_add_kernel: fast FP32 residual accumulation
//       6. bf16_lm_head_gemv_kernel: tied BF16 embedding/lm_head GEMV (151936 x 896)
//       7. a16_prefill_cp_async_imma_b32_kernel: Ampere cp.async.cg.shared.global (LDGSTS)
//          + mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 (IMMA) batched B=32 prefill
//       8. hampack18b_transcode_to_a16_kernel: cold-disk 18B hampack -> VRAM 4.5bpw A16 SoA
//   - Single C ABI entrypoints (qwen25_gpu_*) for zero-Python-overhead 24-layer decode & prefill.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

static constexpr int QWEN_LAYERS = 24;
static constexpr int QWEN_H = 896;
static constexpr int QWEN_I = 4864;
static constexpr int QWEN_Q_HEADS = 14;
static constexpr int QWEN_KV_HEADS = 2;
static constexpr int QWEN_HEAD_DIM = 64;
static constexpr int QWEN_KV_DIM = QWEN_KV_HEADS * QWEN_HEAD_DIM;  // 128
static constexpr int QWEN_VOCAB = 151936;
static constexpr float QWEN_RMS_EPS = 1.0e-6f;
static constexpr float QWEN_ROPE_THETA = 1000000.0f;
static constexpr int MAX_SEQ_LEN = 512;

// ---------------------------------------------------------------------------
// Inline PTX Primitives (Ampere sm_86)
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint32_t ptx_lop3_xor_and(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t r;
    // (a ^ b) & c  -> truth table 0x28
    asm("lop3.b32 %0, %1, %2, %3, 0x28;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}

__device__ __forceinline__ uint32_t ptx_lop3_and_imm(uint32_t a, uint32_t mask) {
    uint32_t r;
    asm("lop3.b32 %0, %1, %2, 0, 0xC0;" : "=r"(r) : "r"(a), "r"(mask));
    return r;
}

__device__ __forceinline__ uint32_t ptx_prmt(uint32_t a, uint32_t b, uint32_t sel) {
    uint32_t r;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
    return r;
}

__device__ __forceinline__ int32_t ptx_dp4a_u32_s32(uint32_t ua, uint32_t sb, int32_t c) {
    int32_t d;
    asm("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(ua), "r"(sb), "r"(c));
    return d;
}

__device__ __forceinline__ uint32_t a16_syndrome_check_16w(uint32_t w0, uint32_t w1) {
    // 16 weights packed as 4-bit biased nibbles (u = q + 8): LSB parity matches q & 1.
    // Extended Hamming [16,11,4] check matrix rows:
    //   p0 = all 16 bits
    //   s0 = odd indices (1,3,5,7,9,11,13,15) -> nibble positions 1,3,5,7 -> 0x10101010
    //   s1 = bit1 indices (2,3,6,7,10,11,14,15) -> nibble positions 2,3,6,7 -> 0x11001100
    //   s2 = bit2 indices (4,5,6,7,12,13,14,15) -> nibble positions 4,5,6,7 -> 0x11110000
    //   s3 = bit3 indices (8..15) -> all nibbles of w1 -> 0x11111111
    uint32_t px = ptx_lop3_xor_and(w0, w1, 0x11111111u);
    uint32_t p0 = __popc(px) & 1u;
    uint32_t s0 = __popc(px & 0x10101010u) & 1u;
    uint32_t s1 = __popc(px & 0x11001100u) & 1u;
    uint32_t s2 = __popc(px & 0x11110000u) & 1u;
    uint32_t s3 = __popc(w1 & 0x11111111u) & 1u;
    return p0 | s0 | s1 | s2 | s3;
}

__device__ __forceinline__ void cp_async_cg_16(uint32_t smem_addr, const void* gmem_ptr) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(gmem_ptr));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N_WAIT>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N_WAIT));
}

__device__ __forceinline__ void mma_m16n8k32_s8(
    int32_t& d0, int32_t& d1, int32_t& d2, int32_t& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    int32_t c0, int32_t c1, int32_t c2, int32_t c3
) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "r"(c0), "r"(c1), "r"(c2), "r"(c3)
    );
}

// ---------------------------------------------------------------------------
// Kernel 1: Fused RMSNorm + Per-32-Group Symmetric INT8 Activation Quantization
// ---------------------------------------------------------------------------

__global__ void rmsnorm_q8_kernel(
    const float* __restrict__ x_in,
    const float* __restrict__ norm_w,
    int8_t* __restrict__ x_q8,
    float* __restrict__ x_scale,
    int32_t* __restrict__ x_gsum,
    int H,
    float eps
) {
    // 1 block of 256 threads; H <= 4864 (H % 32 == 0)
    __shared__ float s_sq[8];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;

    float sq = 0.0f;
    for (int i = tid; i < H; i += blockDim.x) {
        float v = x_in[i];
        sq += v * v;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, off);
    if (lane == 0) s_sq[warp] = sq;
    __syncthreads();
    if (warp == 0) {
        float tot = (lane < 8) ? s_sq[lane] : 0.0f;
        #pragma unroll
        for (int off = 4; off > 0; off >>= 1) tot += __shfl_xor_sync(0xffffffffu, tot, off);
        if (lane == 0) s_sq[0] = 1.0f / sqrtf(tot / float(H) + eps);
    }
    __syncthreads();
    float inv_rms = s_sq[0];

    // Quantize each 32-element group using one warp per group
    int ng = H >> 5;
    for (int g = warp; g < ng; g += 8) {
        int idx = (g << 5) + lane;
        float val = (x_in[idx] * inv_rms) * norm_w[idx];
        float amax = fabsf(val);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
        float sc = (amax > 1.0e-12f) ? (amax / 127.0f) : 1.0e-12f;
        float inv_sc = 1.0f / sc;
        int32_t qv = __float2int_rn(fminf(127.0f, fmaxf(-127.0f, val * inv_sc)));
        x_q8[idx] = static_cast<int8_t>(qv);
        int32_t gsum = qv;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) gsum += __shfl_xor_sync(0xffffffffu, gsum, off);
        if (lane == 0) {
            x_scale[g] = sc;
            x_gsum[g] = gsum;
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: Pure RMSNorm -> FP32 (for final norm before tied BF16 lm_head)
// ---------------------------------------------------------------------------

__global__ void rmsnorm_fp32_kernel(
    const float* __restrict__ x_in,
    const float* __restrict__ norm_w,
    float* __restrict__ x_out,
    int H,
    float eps
) {
    __shared__ float s_sq[8];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    float sq = 0.0f;
    for (int i = tid; i < H; i += blockDim.x) {
        float v = x_in[i];
        sq += v * v;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, off);
    if (lane == 0) s_sq[warp] = sq;
    __syncthreads();
    if (warp == 0) {
        float tot = (lane < 8) ? s_sq[lane] : 0.0f;
        #pragma unroll
        for (int off = 4; off > 0; off >>= 1) tot += __shfl_xor_sync(0xffffffffu, tot, off);
        if (lane == 0) s_sq[0] = rsqrtf(tot / float(H) + eps);
    }
    __syncthreads();
    float inv_rms = s_sq[0];
    for (int i = tid; i < H; i += blockDim.x) {
        x_out[i] = x_in[i] * inv_rms * norm_w[i];
    }
}

// ---------------------------------------------------------------------------
// Kernel 3: Fused 4.500-bpw A_16 LOP3+PRMT+dp4a.u32.s32 GEMV (+ Optional Live 5-POPC)
// ---------------------------------------------------------------------------
// 4 warps/block (128 threads), 2 rows per warp = 8 rows per block.
// Each lane handles stride-32 groups of 32 weights (16 bytes via LDG.128 uint4!).

template <bool LIVE_VERIFY, bool ADD_RESIDUAL>
__global__ void a16_gemv_dp4a_qwen_kernel(
    const uint4* __restrict__ w_nib,     // [M, K/32] uint4 (16 B / 32 weights)
    const __half* __restrict__ w_scale,  // [M, K/32] fp16 scale
    const uint32_t* __restrict__ x_q8,   // [K/4] packed int8 activations
    const float* __restrict__ x_scale,   // [K/32] activation group scales
    const int32_t* __restrict__ x_gsum,  // [K/32] sum of 32 int8 activations per group
    const float* __restrict__ bias,      // optional [M] FP32 bias (may be nullptr)
    float* __restrict__ y_out,           // [M] FP32 output (accumulated if ADD_RESIDUAL)
    uint32_t* __restrict__ syn_err_cnt,  // optional live syndrome error counter
    int M,
    int K
) {
    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row0 = (blockIdx.x * 4 + warp_id) * 2;
    int row1 = row0 + 1;
    int ng = K >> 5;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    uint32_t local_syn_err = 0u;

    const uint4* w0_ptr = w_nib + (size_t)row0 * (size_t)ng;
    const uint4* w1_ptr = w_nib + (size_t)row1 * (size_t)ng;
    const __half* s0_ptr = w_scale + (size_t)row0 * (size_t)ng;
    const __half* s1_ptr = w_scale + (size_t)row1 * (size_t)ng;

    for (int g = lane; g < ng; g += 32) {
        const uint32_t* xp = x_q8 + (g << 3);
        uint32_t x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
        uint32_t x4 = xp[4], x5 = xp[5], x6 = xp[6], x7 = xp[7];
        float xs = x_scale[g];
        int32_t bias_sub = x_gsum[g] << 3;  // 8 * sum(x_q8)

        if (row0 < M) {
            uint4 wv = __ldg(w0_ptr + g);
            if constexpr (LIVE_VERIFY) {
                local_syn_err += a16_syndrome_check_16w(wv.x, wv.y);
                local_syn_err += a16_syndrome_check_16w(wv.z, wv.w);
            }
            int32_t idot = -bias_sub;
            uint32_t lo, hi;
            lo = ptx_lop3_and_imm(wv.x, 0x0F0F0F0Fu); hi = ptx_lop3_and_imm(wv.x >> 4, 0x0F0F0F0Fu);
            idot = ptx_dp4a_u32_s32(ptx_prmt(lo, hi, 0x5140u), x0, idot);
            idot = ptx_dp4a_u32_s32(ptx_prmt(lo, hi, 0x7362u), x1, idot);
            lo = ptx_lop3_and_imm(wv.y, 0x0F0F0F0Fu); hi = ptx_lop3_and_imm(wv.y >> 4, 0x0F0F0F0Fu);
            idot = ptx_dp4a_u32_s32(ptx_prmt(lo, hi, 0x5140u), x2, idot);
            idot = ptx_dp4a_u32_s32(ptx_prmt(lo, hi, 0x7362u), x3, idot);
            lo = ptx_lop3_and_imm(wv.z, 0x0F0F0F0Fu); hi = ptx_lop3_and_imm(wv.z >> 4, 0x0F0F0F0Fu);
            idot = ptx_dp4a_u32_s32(ptx_prmt(lo, hi, 0x5140u), x4, idot);
            idot = ptx_dp4a_u32_s32(ptx_prmt(lo, hi, 0x7362u), x5, idot);
            lo = ptx_lop3_and_imm(wv.w, 0x0F0F0F0Fu); hi = ptx_lop3_and_imm(wv.w >> 4, 0x0F0F0F0Fu);
            idot = ptx_dp4a_u32_s32(ptx_prmt(lo, hi, 0x5140u), x6, idot);
            idot = ptx_dp4a_u32_s32(ptx_prmt(lo, hi, 0x7362u), x7, idot);
            acc0 += float(idot) * (__half2float(__ldg(s0_ptr + g)) * xs);
        }

        if (row1 < M) {
            uint4 wv = __ldg(w1_ptr + g);
            if constexpr (LIVE_VERIFY) {
                local_syn_err += a16_syndrome_check_16w(wv.x, wv.y);
                local_syn_err += a16_syndrome_check_16w(wv.z, wv.w);
            }
            int32_t idot = -bias_sub;
            uint32_t lo, hi;
            lo = ptx_lop3_and_imm(wv.x, 0x0F0F0F0Fu); hi = ptx_lop3_and_imm(wv.x >> 4, 0x0F0F0F0Fu);
            idot = ptx_dp4a_u32_s32(ptx_prmt(lo, hi, 0x5140u), x0, idot);
            idot = ptx_dp4a_u32_s32(ptx_prmt(lo, hi, 0x7362u), x1, idot);
            lo = ptx_lop3_and_imm(wv.y, 0x0F0F0F0Fu); hi = ptx_lop3_and_imm(wv.y >> 4, 0x0F0F0F0Fu);
            idot = ptx_dp4a_u32_s32(ptx_prmt(lo, hi, 0x5140u), x2, idot);
            idot = ptx_dp4a_u32_s32(ptx_prmt(lo, hi, 0x7362u), x3, idot);
            lo = ptx_lop3_and_imm(wv.z, 0x0F0F0F0Fu); hi = ptx_lop3_and_imm(wv.z >> 4, 0x0F0F0F0Fu);
            idot = ptx_dp4a_u32_s32(ptx_prmt(lo, hi, 0x5140u), x4, idot);
            idot = ptx_dp4a_u32_s32(ptx_prmt(lo, hi, 0x7362u), x5, idot);
            lo = ptx_lop3_and_imm(wv.w, 0x0F0F0F0Fu); hi = ptx_lop3_and_imm(wv.w >> 4, 0x0F0F0F0Fu);
            idot = ptx_dp4a_u32_s32(ptx_prmt(lo, hi, 0x5140u), x6, idot);
            idot = ptx_dp4a_u32_s32(ptx_prmt(lo, hi, 0x7362u), x7, idot);
            acc1 += float(idot) * (__half2float(__ldg(s1_ptr + g)) * xs);
        }
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc0 += __shfl_xor_sync(0xffffffffu, acc0, off);
        acc1 += __shfl_xor_sync(0xffffffffu, acc1, off);
    }
    if constexpr (LIVE_VERIFY) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            local_syn_err += __shfl_xor_sync(0xffffffffu, local_syn_err, off);
        }
        if (lane == 0 && local_syn_err != 0u && syn_err_cnt != nullptr) {
            atomicAdd(syn_err_cnt, local_syn_err);
        }
    }
    if (lane == 0) {
        if (row0 < M) {
            float v0 = acc0 + ((bias != nullptr) ? bias[row0] : 0.0f);
            y_out[row0] = ADD_RESIDUAL ? (y_out[row0] + v0) : v0;
        }
        if (row1 < M) {
            float v1 = acc1 + ((bias != nullptr) ? bias[row1] : 0.0f);
            y_out[row1] = ADD_RESIDUAL ? (y_out[row1] + v1) : v1;
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 4: Fused RoPE (rotate-half, theta=1e6) + KV Cache Append + 14:2 GQA Attention
// ---------------------------------------------------------------------------

__global__ void rope_kv_append_kernel(
    float* __restrict__ q_vec,      // [14 * 64]
    float* __restrict__ k_vec,      // [2 * 64]
    const float* __restrict__ v_vec,// [2 * 64]
    float* __restrict__ k_cache,    // [MAX_SEQ_LEN, 2 * 64]
    float* __restrict__ v_cache,    // [MAX_SEQ_LEN, 2 * 64]
    int pos
) {
    // 1 block of 14 * 32 = 448 threads (1 warp per Q head; first 2 warps also rotate K & store KV)
    int head = threadIdx.x >> 5;
    int j = threadIdx.x & 31;
    float inv_freq = powf(QWEN_ROPE_THETA, -float(2 * j) / float(QWEN_HEAD_DIM));
    float angle = float(pos) * inv_freq;
    float c = cosf(angle);
    float s = sinf(angle);

    int q_base = head * QWEN_HEAD_DIM;
    float q0 = q_vec[q_base + j];
    float q1 = q_vec[q_base + j + 32];
    q_vec[q_base + j]      = q0 * c - q1 * s;
    q_vec[q_base + j + 32] = q1 * c + q0 * s;

    if (head < QWEN_KV_HEADS) {
        int k_base = head * QWEN_HEAD_DIM;
        float k0 = k_vec[k_base + j];
        float k1 = k_vec[k_base + j + 32];
        float kr0 = k0 * c - k1 * s;
        float kr1 = k1 * c + k0 * s;
        int dst = pos * QWEN_KV_DIM + k_base;
        k_cache[dst + j]      = kr0;
        k_cache[dst + j + 32] = kr1;
        v_cache[dst + j]      = v_vec[k_base + j];
        v_cache[dst + j + 32] = v_vec[k_base + j + 32];
    }
}

__global__ void gqa_attention_decode_kernel(
    const float* __restrict__ q_vec,   // [14 * 64]
    const float* __restrict__ k_cache, // [MAX_SEQ_LEN, 128]
    const float* __restrict__ v_cache, // [MAX_SEQ_LEN, 128]
    float* __restrict__ attn_out,      // [14 * 64]
    int seq_len                        // pos + 1
) {
    // Grid: 14 blocks (one per Q head), 64 threads (one per head_dim element)
    int qh = blockIdx.x;
    int d = threadIdx.x;
    int kvh = qh / 7;
    __shared__ float s_scores[MAX_SEQ_LEN];
    __shared__ float s_q[QWEN_HEAD_DIM];
    __shared__ float s_red[2];

    s_q[d] = q_vec[qh * QWEN_HEAD_DIM + d];
    __syncthreads();

    float scale = 0.125f;  // 1 / sqrt(64)
    int lane = d & 31;
    int warp = d >> 5;

    for (int t = 0; t < seq_len; ++t) {
        float prod = s_q[d] * k_cache[t * QWEN_KV_DIM + kvh * QWEN_HEAD_DIM + d];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) prod += __shfl_xor_sync(0xffffffffu, prod, off);
        if (lane == 0) s_red[warp] = prod;
        __syncthreads();
        if (d == 0) s_scores[t] = (s_red[0] + s_red[1]) * scale;
        __syncthreads();
    }

    // Softmax over 0..seq_len-1
    float local_max = -1.0e30f;
    for (int t = d; t < seq_len; t += 64) local_max = fmaxf(local_max, s_scores[t]);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffffu, local_max, off));
    if (lane == 0) s_red[warp] = local_max;
    __syncthreads();
    float gmax = fmaxf(s_red[0], s_red[1]);

    float local_sum = 0.0f;
    for (int t = d; t < seq_len; t += 64) {
        float e = expf(s_scores[t] - gmax);
        s_scores[t] = e;
        local_sum += e;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) local_sum += __shfl_xor_sync(0xffffffffu, local_sum, off);
    if (lane == 0) s_red[warp] = local_sum;
    __syncthreads();
    float inv_sum = 1.0f / (s_red[0] + s_red[1]);

    float acc = 0.0f;
    for (int t = 0; t < seq_len; ++t) {
        acc += (s_scores[t] * inv_sum) * v_cache[t * QWEN_KV_DIM + kvh * QWEN_HEAD_DIM + d];
    }
    attn_out[qh * QWEN_HEAD_DIM + d] = acc;
}

// ---------------------------------------------------------------------------
// Kernel 5: Fused SwiGLU (silu(gate) * up) + Per-32-Group INT8 Quantization
// ---------------------------------------------------------------------------

__global__ void silu_mul_q8_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    int8_t* __restrict__ out_q8,
    float* __restrict__ out_scale,
    int32_t* __restrict__ out_gsum,
    int N
) {
    // 1 warp per 32-element group; gridDim.x = (N / 32 + 7) / 8, blockDim.x = 256
    int g = (blockIdx.x * 8) + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    if (g >= (N >> 5)) return;
    int idx = (g << 5) + lane;
    float gv = gate[idx];
    float sv = gv / (1.0f + expf(-gv));
    float val = sv * up[idx];

    float amax = fabsf(val);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
    float sc = (amax > 1.0e-12f) ? (amax / 127.0f) : 1.0e-12f;
    int32_t qv = __float2int_rn(fminf(127.0f, fmaxf(-127.0f, val * (1.0f / sc))));
    out_q8[idx] = static_cast<int8_t>(qv);
    int32_t gsum = qv;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) gsum += __shfl_xor_sync(0xffffffffu, gsum, off);
    if (lane == 0) {
        out_scale[g] = sc;
        out_gsum[g] = gsum;
    }
}

// ---------------------------------------------------------------------------
// Kernel 6: Quantize Plain FP32 Vector -> Per-32-Group INT8 (for o_proj input)
// ---------------------------------------------------------------------------

__global__ void quantize_fp32_q8_kernel(
    const float* __restrict__ x_in,
    int8_t* __restrict__ x_q8,
    float* __restrict__ x_scale,
    int32_t* __restrict__ x_gsum,
    int N
) {
    int g = (blockIdx.x * 8) + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    if (g >= (N >> 5)) return;
    int idx = (g << 5) + lane;
    float val = x_in[idx];
    float amax = fabsf(val);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
    float sc = (amax > 1.0e-12f) ? (amax / 127.0f) : 1.0e-12f;
    int32_t qv = __float2int_rn(fminf(127.0f, fmaxf(-127.0f, val * (1.0f / sc))));
    x_q8[idx] = static_cast<int8_t>(qv);
    int32_t gsum = qv;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) gsum += __shfl_xor_sync(0xffffffffu, gsum, off);
    if (lane == 0) {
        x_scale[g] = sc;
        x_gsum[g] = gsum;
    }
}

// ---------------------------------------------------------------------------
// Kernel 7: Tied BF16 Embedding Lookup & LM Head GEMV (151936 x 896)
// ---------------------------------------------------------------------------

__global__ void embed_lookup_bf16_kernel(
    const __nv_bfloat16* __restrict__ embed_w,
    float* __restrict__ h_out,
    int token_id,
    int H
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < H) {
        h_out[i] = __bfloat162float(embed_w[(size_t)token_id * (size_t)H + i]);
    }
}

__global__ void bf16_lm_head_gemv_kernel(
    const uint4* __restrict__ embed_w_u4,  // [V, H/8] packed bf16 (8 bf16 per uint4)
    const float* __restrict__ h_in,        // [H]
    float* __restrict__ logits_out,        // [V]
    int V,
    int H
) {
    // 4 warps/block, 1 vocab row per warp
    int row = blockIdx.x * 4 + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    if (row >= V) return;
    int n_u4 = H >> 3;  // 896 / 8 = 112
    const uint4* rp = embed_w_u4 + (size_t)row * (size_t)n_u4;
    float acc = 0.0f;
    for (int i = lane; i < n_u4; i += 32) {
        uint4 v = __ldg(rp + i);
        const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&v);
        int base = i << 3;
        float2 f0 = __bfloat1622float2(b2[0]);
        float2 f1 = __bfloat1622float2(b2[1]);
        float2 f2 = __bfloat1622float2(b2[2]);
        float2 f3 = __bfloat1622float2(b2[3]);
        acc += f0.x * h_in[base + 0] + f0.y * h_in[base + 1]
             + f1.x * h_in[base + 2] + f1.y * h_in[base + 3]
             + f2.x * h_in[base + 4] + f2.y * h_in[base + 5]
             + f3.x * h_in[base + 6] + f3.y * h_in[base + 7];
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
    if (lane == 0) logits_out[row] = acc;
}

__global__ void argmax_stage1_kernel(
    const float* __restrict__ logits,
    float* __restrict__ block_val,
    int32_t* __restrict__ block_idx,
    int V
) {
    __shared__ float s_val[8];
    __shared__ int32_t s_idx[8];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    float best_v = -1.0e38f;
    int32_t best_i = 0;
    for (int i = blockIdx.x * blockDim.x + tid; i < V; i += gridDim.x * blockDim.x) {
        float v = logits[i];
        if (v > best_v) { best_v = v; best_i = i; }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_xor_sync(0xffffffffu, best_v, off);
        int32_t oi = __shfl_xor_sync(0xffffffffu, best_i, off);
        if (ov > best_v || (ov == best_v && oi < best_i)) { best_v = ov; best_i = oi; }
    }
    if (lane == 0) { s_val[warp] = best_v; s_idx[warp] = best_i; }
    __syncthreads();
    if (warp == 0) {
        float bv = (lane < 8) ? s_val[lane] : -1.0e38f;
        int32_t bi = (lane < 8) ? s_idx[lane] : 0;
        #pragma unroll
        for (int off = 4; off > 0; off >>= 1) {
            float ov = __shfl_xor_sync(0xffffffffu, bv, off);
            int32_t oi = __shfl_xor_sync(0xffffffffu, bi, off);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) { block_val[blockIdx.x] = bv; block_idx[blockIdx.x] = bi; }
    }
}

__global__ void argmax_stage2_kernel(
    const float* __restrict__ block_val,
    const int32_t* __restrict__ block_idx,
    int32_t* __restrict__ out_token,
    int N
) {
    __shared__ float s_val[8];
    __shared__ int32_t s_idx[8];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    float best_v = (tid < N) ? block_val[tid] : -1.0e38f;
    int32_t best_i = (tid < N) ? block_idx[tid] : 0;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_xor_sync(0xffffffffu, best_v, off);
        int32_t oi = __shfl_xor_sync(0xffffffffu, best_i, off);
        if (ov > best_v || (ov == best_v && oi < best_i)) { best_v = ov; best_i = oi; }
    }
    if (lane == 0) { s_val[warp] = best_v; s_idx[warp] = best_i; }
    __syncthreads();
    if (warp == 0) {
        float bv = (lane < 8) ? s_val[lane] : -1.0e38f;
        int32_t bi = (lane < 8) ? s_idx[lane] : 0;
        #pragma unroll
        for (int off = 4; off > 0; off >>= 1) {
            float ov = __shfl_xor_sync(0xffffffffu, bv, off);
            int32_t oi = __shfl_xor_sync(0xffffffffu, bi, off);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) *out_token = bi;
    }
}

// ---------------------------------------------------------------------------
// Kernel 8: Ampere cp.async (LDGSTS) + mma.sync.m16n8k32 (IMMA) B=32 Prefill GEMM
// ---------------------------------------------------------------------------
// Computes Y[B=32, M] = X[B=32, K] @ W[M, K]^T (+ optional bias[M]) across any Qwen2.5-0.5B matrix.
// 1 CTA = 128 threads (4 warps) computes a 32-row (M) x 32-token (N=B) output tile (2x m16n8k32/warp)!
// Double-buffered smem with hardware cp.async.cg.shared.global (LDGSTS), branchless SWAR
// LOP3+PRMT nibble-to-s8 unpack (0 byte LDS!), and IMMA.16832.S8.S8!

__device__ __forceinline__ uint32_t u4x4_to_s8x4_swar(uint32_t u4_packed_bytes) {
    // Each byte u in [1..15] -> signed int8 (u - 8) in [-7..+7] via 3-op SWAR!
    uint32_t x = u4_packed_bytes ^ 0x08080808u;
    return x | ((x & 0x08080808u) * 30u);
}

__global__ void a16_prefill_cp_async_imma_b32_kernel(
    const uint4* __restrict__ w_nib,     // [M, K/32] uint4
    const __half* __restrict__ w_scale,  // [M, K/32] fp16
    const int8_t* __restrict__ x_q8,     // [32, K] row-major int8
    const float* __restrict__ x_scale,   // [32, K/32] per-token per-group scale
    const float* __restrict__ bias,      // optional [M]
    float* __restrict__ y_out,           // [32, M] row-major FP32
    int M,
    int K
) {
    // Shared memory double-buffer: 2 stages x (32 x 16B weights + 32 x 32B acts) = 3072 B
    __shared__ __align__(16) uint32_t sm_w[2][32 * 4];
    __shared__ __align__(16) int8_t   sm_x[2][32 * 32];
    __shared__ float sm_ws[2][32];
    __shared__ float sm_xs[2][32];

    int tid = threadIdx.x;
    int warp_id = tid >> 5;   // 0..3 -> each warp computes 32 rows x 8 tokens (2x m16n8k32)
    int lane = tid & 31;
    int m_base = blockIdx.x * 32;
    if (m_base >= M) return;
    int ng = K >> 5;
    int n_base = warp_id * 8;

    int group_id = lane >> 2;  // 0..7
    int tid_in_g = lane & 3;   // 0..3

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;

    auto issue_stage = [&](int stage, int g) {
        if (tid < 32) {
            uint32_t s_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&sm_w[stage][tid * 4]));
            const uint4* g_ptr = w_nib + (size_t)(m_base + tid) * (size_t)ng + g;
            cp_async_cg_16(s_addr, g_ptr);
            sm_ws[stage][tid] = __half2float(__ldg(w_scale + (size_t)(m_base + tid) * (size_t)ng + g));
        } else if (tid < 96) {
            int idx = tid - 32;  // 64 x 16B = 1024B = 32 tokens x 32 bytes
            int tok = idx >> 1;
            int half_k = idx & 1;
            uint32_t s_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&sm_x[stage][tok * 32 + half_k * 16]));
            const void* g_ptr = x_q8 + (size_t)tok * (size_t)K + (g << 5) + half_k * 16;
            cp_async_cg_16(s_addr, g_ptr);
            if (half_k == 0) sm_xs[stage][tok] = x_scale[tok * ng + g];
        }
        cp_async_commit();
    };

    issue_stage(0, 0);

    // tid_in_g in 0..3 -> word_idx = tid_in_g >> 1 (0 or 1), prmt_sel = (tid_in_g & 1) ? 0x7362 : 0x5140
    int w_idx_lo = tid_in_g >> 1;          // 0 or 1 (for k in 0..15)
    int w_idx_hi = w_idx_lo + 2;           // 2 or 3 (for k in 16..31)
    uint32_t prmt_sel = (tid_in_g & 1) ? 0x7362u : 0x5140u;

    auto unpack_s8_4 = [&](uint32_t w32) -> uint32_t {
        uint32_t lo = ptx_lop3_and_imm(w32, 0x0F0F0F0Fu);
        uint32_t hi = ptx_lop3_and_imm(w32 >> 4, 0x0F0F0F0Fu);
        return u4x4_to_s8x4_swar(ptx_prmt(lo, hi, prmt_sel));
    };

    for (int g = 0; g < ng; ++g) {
        int cur = g & 1;
        int nxt = cur ^ 1;
        if (g + 1 < ng) issue_stage(nxt, g + 1);
        if (g + 1 < ng) cp_async_wait<1>();
        else cp_async_wait<0>();
        __syncthreads();

        // Load B operand (shared across both m=0..15 and m=16..31 IMMA tiles!)
        const uint32_t* xb = reinterpret_cast<const uint32_t*>(&sm_x[cur][(n_base + group_id) * 32]);
        uint32_t b0 = xb[tid_in_g];
        uint32_t b1 = xb[tid_in_g + 4];
        float xs0 = sm_xs[cur][n_base + tid_in_g * 2];
        float xs1 = sm_xs[cur][n_base + tid_in_g * 2 + 1];

        // Tile 0: rows [0..15]
        uint32_t a0 = unpack_s8_4(sm_w[cur][group_id * 4 + w_idx_lo]);
        uint32_t a1 = unpack_s8_4(sm_w[cur][(group_id + 8) * 4 + w_idx_lo]);
        uint32_t a2 = unpack_s8_4(sm_w[cur][group_id * 4 + w_idx_hi]);
        uint32_t a3 = unpack_s8_4(sm_w[cur][(group_id + 8) * 4 + w_idx_hi]);
        int32_t d0 = 0, d1 = 0, d2 = 0, d3 = 0;
        mma_m16n8k32_s8(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, 0, 0, 0, 0);
        float ws0 = sm_ws[cur][group_id];
        float ws1 = sm_ws[cur][group_id + 8];
        acc0 += float(d0) * (ws0 * xs0);
        acc1 += float(d1) * (ws0 * xs1);
        acc2 += float(d2) * (ws1 * xs0);
        acc3 += float(d3) * (ws1 * xs1);

        // Tile 1: rows [16..31] reusing b0, b1, xs0, xs1 in registers!
        uint32_t a4 = unpack_s8_4(sm_w[cur][(group_id + 16) * 4 + w_idx_lo]);
        uint32_t a5 = unpack_s8_4(sm_w[cur][(group_id + 24) * 4 + w_idx_lo]);
        uint32_t a6 = unpack_s8_4(sm_w[cur][(group_id + 16) * 4 + w_idx_hi]);
        uint32_t a7 = unpack_s8_4(sm_w[cur][(group_id + 24) * 4 + w_idx_hi]);
        int32_t d4 = 0, d5 = 0, d6 = 0, d7 = 0;
        mma_m16n8k32_s8(d4, d5, d6, d7, a4, a5, a6, a7, b0, b1, 0, 0, 0, 0);
        float ws2 = sm_ws[cur][group_id + 16];
        float ws3 = sm_ws[cur][group_id + 24];
        acc4 += float(d4) * (ws2 * xs0);
        acc5 += float(d5) * (ws2 * xs1);
        acc6 += float(d6) * (ws3 * xs0);
        acc7 += float(d7) * (ws3 * xs1);

        __syncthreads();
    }

    int c0 = n_base + tid_in_g * 2;
    int c1 = c0 + 1;
    int r0 = m_base + group_id;
    int r1 = m_base + group_id + 8;
    int r2 = m_base + group_id + 16;
    int r3 = m_base + group_id + 24;
    float br0 = (bias != nullptr) ? bias[r0] : 0.0f;
    float br1 = (bias != nullptr) ? bias[r1] : 0.0f;
    float br2 = (bias != nullptr) ? bias[r2] : 0.0f;
    float br3 = (bias != nullptr) ? bias[r3] : 0.0f;
    y_out[c0 * M + r0] = acc0 + br0;
    y_out[c1 * M + r0] = acc1 + br0;
    y_out[c0 * M + r1] = acc2 + br1;
    y_out[c1 * M + r1] = acc3 + br1;
    y_out[c0 * M + r2] = acc4 + br2;
    y_out[c1 * M + r2] = acc5 + br2;
    y_out[c0 * M + r3] = acc6 + br3;
    y_out[c1 * M + r3] = acc7 + br3;
}

// ---------------------------------------------------------------------------
// Kernel 9: Cold-Disk 18B Packed Stream -> VRAM SoA Transcode + Live Parity Verify
// ---------------------------------------------------------------------------
// Transcodes interleaved 18-byte cold-disk records ([16B A16 nibbles | 2B fp16 scale])
// into coalesced GPU SoA buffers (w_nib[uint4] + w_scale[__half]) with live 5-POPC verify.

__global__ void hampack18b_aos_to_soa_transcode_kernel(
    const uint8_t* __restrict__ disk_18b,  // [n_groups * 18] AoS bytes
    uint4* __restrict__ soa_nib,           // [n_groups] uint4
    __half* __restrict__ soa_scale,        // [n_groups] fp16
    uint32_t* __restrict__ syn_err_cnt,
    int n_groups
) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n_groups) return;
    const uint8_t* rec = disk_18b + (size_t)g * 18u;
    uint32_t w0, w1, w2, w3;
    uint16_t sc_bits;
    memcpy(&w0, rec + 0, 4);
    memcpy(&w1, rec + 4, 4);
    memcpy(&w2, rec + 8, 4);
    memcpy(&w3, rec + 12, 4);
    memcpy(&sc_bits, rec + 16, 2);

    uint32_t err = a16_syndrome_check_16w(w0, w1) + a16_syndrome_check_16w(w2, w3);
    if (err != 0u && syn_err_cnt != nullptr) atomicAdd(syn_err_cnt, err);

    soa_nib[g] = make_uint4(w0, w1, w2, w3);
    soa_scale[g] = __ushort_as_half(sc_bits);
}

// ---------------------------------------------------------------------------
// Host C-ABI Engine Handle (Zero-Python-Overhead 24-Layer Qwen2.5-0.5B)
// ---------------------------------------------------------------------------

struct QwenLayerGPU {
    uint4*  w_nib[7];    // 0:q, 1:k, 2:v, 3:o, 4:gate, 5:up, 6:down
    __half* w_scale[7];
    float*  bias[3];     // q_bias, k_bias, v_bias
    float*  in_norm;     // [896]
    float*  post_norm;   // [896]
};

struct QwenEngineGPU {
    QwenLayerGPU layers[QWEN_LAYERS];
    __nv_bfloat16* d_embed;  // [151936, 896]
    float* d_final_norm;     // [896]

    // KV caches: [24, MAX_SEQ_LEN, 128]
    float* d_k_cache;
    float* d_v_cache;

    // Scratch activations
    float*   d_h;          // [896]
    float*   d_h_norm;     // [896]
    float*   d_q;          // [896]
    float*   d_k;          // [128]
    float*   d_v;          // [128]
    float*   d_attn_out;   // [896]
    float*   d_gate;       // [4864]
    float*   d_up;         // [4864]
    float*   d_logits;     // [151936]

    int8_t*  d_x_q8;       // [4864]
    float*   d_x_scale;    // [152]
    int32_t* d_x_gsum;     // [152]
    uint32_t* d_syn_err;   // [1]
    float*   d_arg_val;    // [256]
    int32_t* d_arg_idx;    // [256]
    int32_t* d_out_tok;    // [1]

    cudaStream_t stream;
};

static const int PROJ_M[7] = {QWEN_H, QWEN_KV_DIM, QWEN_KV_DIM, QWEN_H, QWEN_I, QWEN_I, QWEN_H};
static const int PROJ_K[7] = {QWEN_H, QWEN_H,      QWEN_H,      QWEN_H, QWEN_H, QWEN_H, QWEN_I};

extern "C" {

void* qwen25_gpu_create() {
    QwenEngineGPU* eng = new QwenEngineGPU();
    memset(eng, 0, sizeof(QwenEngineGPU));
    cudaStreamCreate(&eng->stream);

    // Pin persisting L2 cache window for hot activation scratch (~2 MiB max)
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (prop.persistingL2CacheMaxSize > 0) {
        size_t max_l2 = static_cast<size_t>(prop.persistingL2CacheMaxSize);
        size_t l2_size = (max_l2 < (2u << 20)) ? max_l2 : static_cast<size_t>(2u << 20);
        cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, l2_size);
    }

    cudaMalloc(&eng->d_embed,      (size_t)QWEN_VOCAB * QWEN_H * sizeof(__nv_bfloat16));
    cudaMalloc(&eng->d_final_norm, QWEN_H * sizeof(float));
    cudaMalloc(&eng->d_k_cache,    (size_t)QWEN_LAYERS * MAX_SEQ_LEN * QWEN_KV_DIM * sizeof(float));
    cudaMalloc(&eng->d_v_cache,    (size_t)QWEN_LAYERS * MAX_SEQ_LEN * QWEN_KV_DIM * sizeof(float));

    cudaMalloc(&eng->d_h,          QWEN_H * sizeof(float));
    cudaMalloc(&eng->d_h_norm,     QWEN_H * sizeof(float));
    cudaMalloc(&eng->d_q,          QWEN_H * sizeof(float));
    cudaMalloc(&eng->d_k,          QWEN_KV_DIM * sizeof(float));
    cudaMalloc(&eng->d_v,          QWEN_KV_DIM * sizeof(float));
    cudaMalloc(&eng->d_attn_out,   QWEN_H * sizeof(float));
    cudaMalloc(&eng->d_gate,       QWEN_I * sizeof(float));
    cudaMalloc(&eng->d_up,         QWEN_I * sizeof(float));
    cudaMalloc(&eng->d_logits,     QWEN_VOCAB * sizeof(float));
    cudaMalloc(&eng->d_x_q8,       QWEN_I * sizeof(int8_t));
    cudaMalloc(&eng->d_x_scale,    (QWEN_I / 32) * sizeof(float));
    cudaMalloc(&eng->d_x_gsum,     (QWEN_I / 32) * sizeof(int32_t));
    cudaMalloc(&eng->d_syn_err,    sizeof(uint32_t));
    cudaMalloc(&eng->d_arg_val,    256 * sizeof(float));
    cudaMalloc(&eng->d_arg_idx,    256 * sizeof(int32_t));
    cudaMalloc(&eng->d_out_tok,    sizeof(int32_t));
    cudaMemset(eng->d_syn_err, 0, sizeof(uint32_t));

    // Activate hardware L2 persisting cache policy on hot quantized activation buffer
    cudaStreamAttrValue attr;
    memset(&attr, 0, sizeof(attr));
    attr.accessPolicyWindow.base_ptr = eng->d_x_q8;
    attr.accessPolicyWindow.num_bytes = QWEN_I * sizeof(int8_t);
    attr.accessPolicyWindow.hitRatio = 1.0f;
    attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
    cudaStreamSetAttribute(eng->stream, cudaStreamAttributeAccessPolicyWindow, &attr);

    for (int l = 0; l < QWEN_LAYERS; ++l) {
        for (int p = 0; p < 7; ++p) {
            size_t ng = (size_t)PROJ_M[p] * (size_t)(PROJ_K[p] / 32);
            cudaMalloc(&eng->layers[l].w_nib[p],   ng * sizeof(uint4));
            cudaMalloc(&eng->layers[l].w_scale[p], ng * sizeof(__half));
        }
        cudaMalloc(&eng->layers[l].bias[0], QWEN_H * sizeof(float));
        cudaMalloc(&eng->layers[l].bias[1], QWEN_KV_DIM * sizeof(float));
        cudaMalloc(&eng->layers[l].bias[2], QWEN_KV_DIM * sizeof(float));
        cudaMalloc(&eng->layers[l].in_norm,   QWEN_H * sizeof(float));
        cudaMalloc(&eng->layers[l].post_norm, QWEN_H * sizeof(float));
    }
    return eng;
}

void qwen25_gpu_destroy(void* handle) {
    if (!handle) return;
    QwenEngineGPU* eng = static_cast<QwenEngineGPU*>(handle);
    for (int l = 0; l < QWEN_LAYERS; ++l) {
        for (int p = 0; p < 7; ++p) {
            cudaFree(eng->layers[l].w_nib[p]);
            cudaFree(eng->layers[l].w_scale[p]);
        }
        for (int b = 0; b < 3; ++b) cudaFree(eng->layers[l].bias[b]);
        cudaFree(eng->layers[l].in_norm);
        cudaFree(eng->layers[l].post_norm);
    }
    cudaFree(eng->d_embed);
    cudaFree(eng->d_final_norm);
    cudaFree(eng->d_k_cache);
    cudaFree(eng->d_v_cache);
    cudaFree(eng->d_h);
    cudaFree(eng->d_h_norm);
    cudaFree(eng->d_q);
    cudaFree(eng->d_k);
    cudaFree(eng->d_v);
    cudaFree(eng->d_attn_out);
    cudaFree(eng->d_gate);
    cudaFree(eng->d_up);
    cudaFree(eng->d_logits);
    cudaFree(eng->d_x_q8);
    cudaFree(eng->d_x_scale);
    cudaFree(eng->d_x_gsum);
    cudaFree(eng->d_syn_err);
    cudaFree(eng->d_arg_val);
    cudaFree(eng->d_arg_idx);
    cudaFree(eng->d_out_tok);
    cudaStreamDestroy(eng->stream);
    delete eng;
}

void qwen25_gpu_reset_device() {
    cudaDeviceReset();
}

void qwen25_gpu_load_globals(void* handle, const uint16_t* embed_bf16, const float* final_norm_f32) {
    QwenEngineGPU* eng = static_cast<QwenEngineGPU*>(handle);
    cudaMemcpy(eng->d_embed, embed_bf16, (size_t)QWEN_VOCAB * QWEN_H * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(eng->d_final_norm, final_norm_f32, QWEN_H * sizeof(float), cudaMemcpyHostToDevice);
}

void qwen25_gpu_load_layer_norms_biases(
    void* handle, int layer,
    const float* in_norm, const float* post_norm,
    const float* q_bias, const float* k_bias, const float* v_bias
) {
    QwenEngineGPU* eng = static_cast<QwenEngineGPU*>(handle);
    cudaMemcpy(eng->layers[layer].in_norm,   in_norm,   QWEN_H * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(eng->layers[layer].post_norm, post_norm, QWEN_H * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(eng->layers[layer].bias[0],   q_bias,    QWEN_H * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(eng->layers[layer].bias[1],   k_bias,    QWEN_KV_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(eng->layers[layer].bias[2],   v_bias,    QWEN_KV_DIM * sizeof(float), cudaMemcpyHostToDevice);
}

void qwen25_gpu_load_proj(void* handle, int layer, int proj, const uint8_t* nib_host, const uint16_t* scale_fp16_host) {
    QwenEngineGPU* eng = static_cast<QwenEngineGPU*>(handle);
    size_t ng = (size_t)PROJ_M[proj] * (size_t)(PROJ_K[proj] / 32);
    cudaMemcpy(eng->layers[layer].w_nib[proj],   nib_host,        ng * 16, cudaMemcpyHostToDevice);
    cudaMemcpy(eng->layers[layer].w_scale[proj], scale_fp16_host, ng * 2,  cudaMemcpyHostToDevice);
}

void qwen25_gpu_reset_kv(void* handle) {
    QwenEngineGPU* eng = static_cast<QwenEngineGPU*>(handle);
    cudaMemsetAsync(eng->d_k_cache, 0, (size_t)QWEN_LAYERS * MAX_SEQ_LEN * QWEN_KV_DIM * sizeof(float), eng->stream);
    cudaMemsetAsync(eng->d_v_cache, 0, (size_t)QWEN_LAYERS * MAX_SEQ_LEN * QWEN_KV_DIM * sizeof(float), eng->stream);
    cudaMemsetAsync(eng->d_syn_err, 0, sizeof(uint32_t), eng->stream);
    cudaStreamSynchronize(eng->stream);
}

uint32_t qwen25_gpu_get_syn_errors(void* handle) {
    QwenEngineGPU* eng = static_cast<QwenEngineGPU*>(handle);
    uint32_t out = 0;
    cudaMemcpy(&out, eng->d_syn_err, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    return out;
}

static void run_decode_step_impl(QwenEngineGPU* eng, int token_id, int pos, int live_verify, int compute_logits) {
    cudaStream_t st = eng->stream;
    embed_lookup_bf16_kernel<<<(QWEN_H + 255) / 256, 256, 0, st>>>(eng->d_embed, eng->d_h, token_id, QWEN_H);

    const uint32_t* xq_u32 = reinterpret_cast<const uint32_t*>(eng->d_x_q8);
    for (int l = 0; l < QWEN_LAYERS; ++l) {
        QwenLayerGPU& ly = eng->layers[l];
        rmsnorm_q8_kernel<<<1, 256, 0, st>>>(eng->d_h, ly.in_norm, eng->d_x_q8, eng->d_x_scale, eng->d_x_gsum, QWEN_H, QWEN_RMS_EPS);

        if (live_verify) {
            a16_gemv_dp4a_qwen_kernel<true, false><<<QWEN_H / 8, 128, 0, st>>>(ly.w_nib[0], ly.w_scale[0], xq_u32, eng->d_x_scale, eng->d_x_gsum, ly.bias[0], eng->d_q, eng->d_syn_err, QWEN_H, QWEN_H);
            a16_gemv_dp4a_qwen_kernel<true, false><<<QWEN_KV_DIM / 8, 128, 0, st>>>(ly.w_nib[1], ly.w_scale[1], xq_u32, eng->d_x_scale, eng->d_x_gsum, ly.bias[1], eng->d_k, eng->d_syn_err, QWEN_KV_DIM, QWEN_H);
            a16_gemv_dp4a_qwen_kernel<true, false><<<QWEN_KV_DIM / 8, 128, 0, st>>>(ly.w_nib[2], ly.w_scale[2], xq_u32, eng->d_x_scale, eng->d_x_gsum, ly.bias[2], eng->d_v, eng->d_syn_err, QWEN_KV_DIM, QWEN_H);
        } else {
            a16_gemv_dp4a_qwen_kernel<false, false><<<QWEN_H / 8, 128, 0, st>>>(ly.w_nib[0], ly.w_scale[0], xq_u32, eng->d_x_scale, eng->d_x_gsum, ly.bias[0], eng->d_q, nullptr, QWEN_H, QWEN_H);
            a16_gemv_dp4a_qwen_kernel<false, false><<<QWEN_KV_DIM / 8, 128, 0, st>>>(ly.w_nib[1], ly.w_scale[1], xq_u32, eng->d_x_scale, eng->d_x_gsum, ly.bias[1], eng->d_k, nullptr, QWEN_KV_DIM, QWEN_H);
            a16_gemv_dp4a_qwen_kernel<false, false><<<QWEN_KV_DIM / 8, 128, 0, st>>>(ly.w_nib[2], ly.w_scale[2], xq_u32, eng->d_x_scale, eng->d_x_gsum, ly.bias[2], eng->d_v, nullptr, QWEN_KV_DIM, QWEN_H);
        }

        float* lk = eng->d_k_cache + (size_t)l * MAX_SEQ_LEN * QWEN_KV_DIM;
        float* lv = eng->d_v_cache + (size_t)l * MAX_SEQ_LEN * QWEN_KV_DIM;
        rope_kv_append_kernel<<<1, QWEN_Q_HEADS * 32, 0, st>>>(eng->d_q, eng->d_k, eng->d_v, lk, lv, pos);
        gqa_attention_decode_kernel<<<QWEN_Q_HEADS, QWEN_HEAD_DIM, 0, st>>>(eng->d_q, lk, lv, eng->d_attn_out, pos + 1);

        quantize_fp32_q8_kernel<<<((QWEN_H / 32) + 7) / 8, 256, 0, st>>>(eng->d_attn_out, eng->d_x_q8, eng->d_x_scale, eng->d_x_gsum, QWEN_H);
        if (live_verify) {
            a16_gemv_dp4a_qwen_kernel<true, true><<<QWEN_H / 8, 128, 0, st>>>(ly.w_nib[3], ly.w_scale[3], xq_u32, eng->d_x_scale, eng->d_x_gsum, nullptr, eng->d_h, eng->d_syn_err, QWEN_H, QWEN_H);
        } else {
            a16_gemv_dp4a_qwen_kernel<false, true><<<QWEN_H / 8, 128, 0, st>>>(ly.w_nib[3], ly.w_scale[3], xq_u32, eng->d_x_scale, eng->d_x_gsum, nullptr, eng->d_h, nullptr, QWEN_H, QWEN_H);
        }

        rmsnorm_q8_kernel<<<1, 256, 0, st>>>(eng->d_h, ly.post_norm, eng->d_x_q8, eng->d_x_scale, eng->d_x_gsum, QWEN_H, QWEN_RMS_EPS);
        if (live_verify) {
            a16_gemv_dp4a_qwen_kernel<true, false><<<QWEN_I / 8, 128, 0, st>>>(ly.w_nib[4], ly.w_scale[4], xq_u32, eng->d_x_scale, eng->d_x_gsum, nullptr, eng->d_gate, eng->d_syn_err, QWEN_I, QWEN_H);
            a16_gemv_dp4a_qwen_kernel<true, false><<<QWEN_I / 8, 128, 0, st>>>(ly.w_nib[5], ly.w_scale[5], xq_u32, eng->d_x_scale, eng->d_x_gsum, nullptr, eng->d_up,   eng->d_syn_err, QWEN_I, QWEN_H);
        } else {
            a16_gemv_dp4a_qwen_kernel<false, false><<<QWEN_I / 8, 128, 0, st>>>(ly.w_nib[4], ly.w_scale[4], xq_u32, eng->d_x_scale, eng->d_x_gsum, nullptr, eng->d_gate, nullptr, QWEN_I, QWEN_H);
            a16_gemv_dp4a_qwen_kernel<false, false><<<QWEN_I / 8, 128, 0, st>>>(ly.w_nib[5], ly.w_scale[5], xq_u32, eng->d_x_scale, eng->d_x_gsum, nullptr, eng->d_up,   nullptr, QWEN_I, QWEN_H);
        }

        silu_mul_q8_kernel<<<((QWEN_I / 32) + 7) / 8, 256, 0, st>>>(eng->d_gate, eng->d_up, eng->d_x_q8, eng->d_x_scale, eng->d_x_gsum, QWEN_I);
        if (live_verify) {
            a16_gemv_dp4a_qwen_kernel<true, true><<<QWEN_H / 8, 128, 0, st>>>(ly.w_nib[6], ly.w_scale[6], xq_u32, eng->d_x_scale, eng->d_x_gsum, nullptr, eng->d_h, eng->d_syn_err, QWEN_H, QWEN_I);
        } else {
            a16_gemv_dp4a_qwen_kernel<false, true><<<QWEN_H / 8, 128, 0, st>>>(ly.w_nib[6], ly.w_scale[6], xq_u32, eng->d_x_scale, eng->d_x_gsum, nullptr, eng->d_h, nullptr, QWEN_H, QWEN_I);
        }
    }

    if (compute_logits) {
        rmsnorm_fp32_kernel<<<1, 256, 0, st>>>(eng->d_h, eng->d_final_norm, eng->d_h_norm, QWEN_H, QWEN_RMS_EPS);
        bf16_lm_head_gemv_kernel<<<(QWEN_VOCAB + 3) / 4, 128, 0, st>>>(
            reinterpret_cast<const uint4*>(eng->d_embed), eng->d_h_norm, eng->d_logits, QWEN_VOCAB, QWEN_H
        );
    }
}

void qwen25_gpu_forward_token(void* handle, int token_id, int pos, int live_verify, float* logits_host) {
    QwenEngineGPU* eng = static_cast<QwenEngineGPU*>(handle);
    run_decode_step_impl(eng, token_id, pos, live_verify, logits_host != nullptr ? 1 : 0);
    if (logits_host != nullptr) {
        cudaMemcpyAsync(logits_host, eng->d_logits, QWEN_VOCAB * sizeof(float), cudaMemcpyDeviceToHost, eng->stream);
    }
    cudaStreamSynchronize(eng->stream);
}

// Sustained zero-Python-overhead decode loop on GPU (returns ms/tok; GPU argmax avoids 607KB D2H copy!)
double qwen25_gpu_bench_decode_sustained(
    void* handle,
    int seed_token,
    int n_iters,
    int live_verify,
    int include_lm_head,
    int* out_last_token
) {
    QwenEngineGPU* eng = static_cast<QwenEngineGPU*>(handle);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventRecord(e0, eng->stream);
    int cur_tok = seed_token;
    for (int it = 0; it < n_iters; ++it) {
        int pos = it & 63;  // steady-state KV cache length in [1..64]
        run_decode_step_impl(eng, cur_tok, pos, live_verify, include_lm_head);
        if (include_lm_head) {
            argmax_stage1_kernel<<<256, 256, 0, eng->stream>>>(eng->d_logits, eng->d_arg_val, eng->d_arg_idx, QWEN_VOCAB);
            argmax_stage2_kernel<<<1, 256, 0, eng->stream>>>(eng->d_arg_val, eng->d_arg_idx, eng->d_out_tok, 256);
        }
    }
    if (include_lm_head) {
        cudaMemcpyAsync(&cur_tok, eng->d_out_tok, sizeof(int32_t), cudaMemcpyDeviceToHost, eng->stream);
    }
    cudaEventRecord(e1, eng->stream);
    cudaEventSynchronize(e1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    if (out_last_token) *out_last_token = cur_tok;
    return double(ms) / double(n_iters);
}

// Prompt prefill (returns TTFT ms) + N greedy autoregressive generation tokens on GPU!
double qwen25_gpu_generate_greedy(
    void* handle,
    const int32_t* prompt_tokens,
    int n_prompt,
    int n_gen,
    int live_verify,
    int32_t* out_gen_tokens,
    double* out_ttft_ms
) {
    QwenEngineGPU* eng = static_cast<QwenEngineGPU*>(handle);
    qwen25_gpu_reset_kv(handle);
    cudaEvent_t e0, e_ttft, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e_ttft);
    cudaEventCreate(&e1);

    cudaEventRecord(e0, eng->stream);
    for (int p = 0; p < n_prompt; ++p) {
        int is_last = (p == n_prompt - 1) ? 1 : 0;
        run_decode_step_impl(eng, prompt_tokens[p], p, live_verify, is_last);
    }
    argmax_stage1_kernel<<<256, 256, 0, eng->stream>>>(eng->d_logits, eng->d_arg_val, eng->d_arg_idx, QWEN_VOCAB);
    argmax_stage2_kernel<<<1, 256, 0, eng->stream>>>(eng->d_arg_val, eng->d_arg_idx, eng->d_out_tok, 256);
    int32_t next_tok = 0;
    cudaMemcpyAsync(&next_tok, eng->d_out_tok, sizeof(int32_t), cudaMemcpyDeviceToHost, eng->stream);
    cudaEventRecord(e_ttft, eng->stream);
    cudaEventSynchronize(e_ttft);
    if (n_gen > 0 && out_gen_tokens) out_gen_tokens[0] = next_tok;

    for (int g = 1; g < n_gen; ++g) {
        run_decode_step_impl(eng, next_tok, n_prompt + g - 1, live_verify, 1);
        argmax_stage1_kernel<<<256, 256, 0, eng->stream>>>(eng->d_logits, eng->d_arg_val, eng->d_arg_idx, QWEN_VOCAB);
        argmax_stage2_kernel<<<1, 256, 0, eng->stream>>>(eng->d_arg_val, eng->d_arg_idx, eng->d_out_tok, 256);
        cudaMemcpyAsync(&next_tok, eng->d_out_tok, sizeof(int32_t), cudaMemcpyDeviceToHost, eng->stream);
        cudaStreamSynchronize(eng->stream);
        if (out_gen_tokens) out_gen_tokens[g] = next_tok;
    }
    cudaEventRecord(e1, eng->stream);
    cudaEventSynchronize(e1);
    float ttft_ms = 0.0f, total_ms = 0.0f;
    cudaEventElapsedTime(&ttft_ms, e0, e_ttft);
    cudaEventElapsedTime(&total_ms, e0, e1);
    cudaEventDestroy(e0);
    cudaEventDestroy(e_ttft);
    cudaEventDestroy(e1);
    if (out_ttft_ms) *out_ttft_ms = double(ttft_ms);
    return double(total_ms);
}

// Batched B=32 IMMA+cp.async benchmark across ALL 168 real Qwen2.5-0.5B matrices!
double qwen25_gpu_bench_b32_imma_all168(
    void* handle,
    const int8_t* x_q8_h_host,    // [32, 896]
    const float*  x_sc_h_host,    // [32, 28]
    const int8_t* x_q8_i_host,    // [32, 4864]
    const float*  x_sc_i_host,    // [32, 152]
    float*        sample_y_host,  // [32, 896] output of layer 0 q_proj for accuracy parity check!
    int iters
) {
    QwenEngineGPU* eng = static_cast<QwenEngineGPU*>(handle);
    int8_t *d_xh, *d_xi;
    float *d_sch, *d_sci, *d_y;
    cudaMalloc(&d_xh,  32 * QWEN_H * sizeof(int8_t));
    cudaMalloc(&d_sch, 32 * (QWEN_H / 32) * sizeof(float));
    cudaMalloc(&d_xi,  32 * QWEN_I * sizeof(int8_t));
    cudaMalloc(&d_sci, 32 * (QWEN_I / 32) * sizeof(float));
    cudaMalloc(&d_y,   32 * QWEN_I * sizeof(float));

    cudaMemcpy(d_xh,  x_q8_h_host, 32 * QWEN_H, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sch, x_sc_h_host, 32 * (QWEN_H / 32) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_xi,  x_q8_i_host, 32 * QWEN_I, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sci, x_sc_i_host, 32 * (QWEN_I / 32) * sizeof(float), cudaMemcpyHostToDevice);

    // Compute layer 0 q_proj parity sample
    a16_prefill_cp_async_imma_b32_kernel<<<QWEN_H / 32, 128, 0, eng->stream>>>(
        eng->layers[0].w_nib[0], eng->layers[0].w_scale[0], d_xh, d_sch, eng->layers[0].bias[0], d_y, QWEN_H, QWEN_H
    );
    if (sample_y_host) {
        cudaMemcpyAsync(sample_y_host, d_y, 32 * QWEN_H * sizeof(float), cudaMemcpyDeviceToHost, eng->stream);
    }
    cudaStreamSynchronize(eng->stream);

    // Warmup + timed loop over all 168 real matrices
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventRecord(e0, eng->stream);
    for (int it = 0; it < iters; ++it) {
        for (int l = 0; l < QWEN_LAYERS; ++l) {
            QwenLayerGPU& ly = eng->layers[l];
            for (int p = 0; p < 7; ++p) {
                int M = PROJ_M[p], K = PROJ_K[p];
                const int8_t* xp = (K == QWEN_H) ? d_xh : d_xi;
                const float*  sp = (K == QWEN_H) ? d_sch : d_sci;
                const float*  bp = (p < 3) ? ly.bias[p] : nullptr;
                a16_prefill_cp_async_imma_b32_kernel<<<M / 32, 128, 0, eng->stream>>>(
                    ly.w_nib[p], ly.w_scale[p], xp, sp, bp, d_y, M, K
                );
            }
        }
    }
    cudaEventRecord(e1, eng->stream);
    cudaEventSynchronize(e1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaFree(d_xh); cudaFree(d_sch); cudaFree(d_xi); cudaFree(d_sci); cudaFree(d_y);
    return double(ms) / double(iters);
}

// Cold-disk 18B AoS buffer -> GPU transcode directly into all 168 layer VRAM SoA buffers!
double qwen25_gpu_transcode_18b_all168(void* handle, const uint8_t* disk_18b_host, uint32_t* syn_errors_out) {
    QwenEngineGPU* eng = static_cast<QwenEngineGPU*>(handle);
    size_t total_groups = 0;
    for (int l = 0; l < QWEN_LAYERS; ++l) {
        for (int p = 0; p < 7; ++p) total_groups += (size_t)PROJ_M[p] * (size_t)(PROJ_K[p] / 32);
    }
    size_t total_bytes = total_groups * 18u;  // 201,277,440 bytes
    uint8_t* d_disk = nullptr;
    cudaMalloc(&d_disk, total_bytes);
    cudaMemsetAsync(eng->d_syn_err, 0, sizeof(uint32_t), eng->stream);

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventRecord(e0, eng->stream);

    cudaMemcpyAsync(d_disk, disk_18b_host, total_bytes, cudaMemcpyHostToDevice, eng->stream);
    size_t offset_g = 0;
    for (int l = 0; l < QWEN_LAYERS; ++l) {
        for (int p = 0; p < 7; ++p) {
            int ng = PROJ_M[p] * (PROJ_K[p] / 32);
            hampack18b_aos_to_soa_transcode_kernel<<<(ng + 255) / 256, 256, 0, eng->stream>>>(
                d_disk + offset_g * 18u, eng->layers[l].w_nib[p], eng->layers[l].w_scale[p], eng->d_syn_err, ng
            );
            offset_g += (size_t)ng;
        }
    }

    cudaEventRecord(e1, eng->stream);
    cudaEventSynchronize(e1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, e0, e1);
    if (syn_errors_out) {
        cudaMemcpy(syn_errors_out, eng->d_syn_err, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaFree(d_disk);
    return double(ms);
}

}  // extern "C"
