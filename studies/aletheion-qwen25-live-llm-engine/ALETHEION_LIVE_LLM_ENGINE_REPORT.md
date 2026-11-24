# Real-Weight Full-Model Qwen2.5-0.5B A_16 PTX/SASS vs. AVX-512 Live Silicon Engine Report

**Hardware Platform:** ASUS ZenBook Pro Duo `UX582HS` — Intel Core `i9-11900H` (`8C/16T`, `AVX-512` `VBMI`/`VNNI`/`VPOPCNTDQ`) + NVIDIA GeForce RTX 3080 Laptop (`GA104M`, Ampere `sm_86`, `48` SMs, `8 GiB` `256-bit` GDDR6).  
**Model Under Test (`measured_live_model_weights: true`):** Offline-cached `Qwen/Qwen2.5-0.5B` snapshot `060db6499f32faf8b98477b0a26969ef7d8b9987` (`model.safetensors` SHA-256 `88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`, `24` transformer layers, `168` real linear weight matrices = `357,826,560` weights + `122` preserved `BF16` non-block tensors).  
**Ring-0 Provenance:** Executed inside exclusive `mlos gpu-lease --mode train` + `mlos stamp-run --profile ux582hs-rtx3080-8gb.toml --zone B --require-gpu --strict-anomalies` (`clean_hardware_run: true`, `exit_code: 0`, `anomalies: []`, `graphics_processes: []`, `compute_processes: []`, `linux_kmd_baseline_vram_mib: 2`, `thermal_throttled: false`, `slowdown_active: false`, receipt SHA-256 `208e14b27757d6adef80df9d0378ee813976694d95f99dcd2893b92354add666`).

---

## 1. Exact Algebraic & Packing Invariants (`provenance_tier: "exact"`)

| Invariant | Exact Value |
|---|---|
| Transformer linear matrix inventory (`24` layers $\times$ `7` projections) | `168` matrices (`48` $\times$ `896x896`, `48` $\times$ `128x896`, `48` $\times$ `4864x896`, `24` $\times$ `896x4864`) |
| Exact alignment (`M % 32 == 0` and `K % 128 == 0`) | `true` (zero padding across all `168` real matrices) |
| Total quantized block weights | `357,826,560` (`11,182,080` 32-weight groups / `22,364,160` 16-weight subblocks) |
| Preserved `BF16` non-block tensors (`embed_tokens`/tied `lm_head`, norms, QKV biases) | `122` tensors (`tied_embed_bf16_bytes = 272,269,312` B) |
| Unified `A_16` packed format (`16 B` biased `u4` nibbles + `2 B` `fp16` scale per `32` w) | `18 B / 32 w` = exact `9/2` (`4.5000`) bpw |
| Cold-disk `18 B` linear payload vs. `BF16` linear bytes | `201,277,440` B vs. `715,653,120` B (exact `32/9` = `3.5556x` compression) |
| Exhaustive Extended Hamming `[16,11,4]` enumeration (`65,536` binary vectors) | `2,048` codewords, $d_{\min}=4$, weight enumerator `{0:1, 4:140, 6:448, 8:870, 10:448, 12:140, 16:1}` |
| Exact bitmask equivalence (CUDA `5-POPC` & AVX-512 `vpopcntq` vs. $\mathbb{F}_2$ reference) | `0` mismatches across all `65,536` patterns |
| Exact 3-op SWAR `u4 -> s8` (`x = u ^ 8; x | ((x & 8) * 30)`) over `u in [1..15]` | `15/15` exact two's-complement matches |

---

## 2. ISA & Zero-Spill Silicon Verification (`provenance_tier: "sampled"`)

- **Ampere `sm_86` Bare-Metal Engine (`qwen25_sm86_a16_engine.cu`):**
  - `nvcc -O3 -std=c++17 -arch=sm_86 --ptxas-options=-v` + `cuobjdump -sass`: **`16/16` kernels, `0` compiler warnings, `0` B stack frame, `0` B spill stores, `0` B spill loads, `0` SASS `LDL`, `0` SASS `STL`** (`zero_spill_verified: true`).
  - Core fused GEMV kernels: `43` regs (`live_verify=0`: `LDG.E.128=2, LOP3=18, PRMT=16, IDP.4A=16, SHFL=10, POPC=0`) and `47` regs (`live_verify=1`: `LDG.E.128=2, LOP3=50, PRMT=16, IDP.4A=16, POPC=20, SHFL=15`).
  - Batched `B=32` prefill kernel (`a16_prefill_cp_async_imma_b32_kernel`): `46` regs, `3,584` B smem, `LDGSTS=4` (`cp.async`), `IMMA=2` (`mma.sync.m16n8k32`), `LOP3=48`, `PRMT=8`, `0` spill, `l0_q_proj_rel_l2_vs_fp64 = 2.7e-8`.
  - Cold-disk transcode kernel (`hampack18b_aos_to_soa_transcode_kernel`): `24` regs, `POPC=10`, `0` spill.
- **Intel `i9-11900H` `AVX-512` Engine (`qwen25_avx512_a16_engine.c`):**
  - `gcc -O3 -march=native -mavx512f -mavx512bw -mavx512vl -mavx512vbmi -mavx512vpopcntdq -mavx512vnni -fopenmp`: **`0` warnings**, **`385` `%zmm` register operands**, **`vpermb=8`**, **`vpopcntq=10`**, **`vpdpbusd=4`**, `vfmadd231ps=24` (`avx512_vbmi_vnni_vpopcntdq_verified: true`).

---

## 3. Empirical 4-Arm Latency & Energy (`mJ/tok`) Pareto Frontier (`measured_live_model_weights: true`)

Energy methodology: Synchronized Linux `powercap` `intel-rapl` hardware microjoule counters (`package-0`, `core`, `uncore`, `psys`) paired with `100 Hz` trapezoidal integration of `nvmlDeviceGetPowerUsage` (mW INA3221 rail sensor) over sustained `>= 2.1 s` steady-state windows. (As calibrated pre-run, `GA104M` laptop `nvmlDeviceGetTotalEnergyConsumption` exhibited `>16x` spurious idle counter drift vs. INA3221 rail power and was rejected.) During Arm C (`CUDA_VISIBLE_DEVICES=""`, `OMP_NUM_THREADS=8 OMP_PROC_BIND=spread OMP_PLACES=cores`), parent `cudaDeviceReset()` dropped the dGPU to true idle `P8` (`210.0 MHz`, `13.49–13.74 W`) across all four CPU-only windows.

### 3.1 24-Layer Transformer Backbone Decode (`168` Real Matrices = `357,826,560` Weights)

| Arm | Throughput (`tok/s`) | Latency (`ms/tok`) | GPU (`mJ/tok`) | CPU Pkg (`mJ/tok`) | Combined GPU+CPU (`mJ/tok`) | `psys` (`mJ/tok`) | Live Verify Overhead | Syndrome Errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Arm A:** PyTorch `BF16` (RTX 3080) | `76.921` | `13.00028` | `618.13029` | `815.73337` | `1,433.86366` | `80.35898` | — | — |
| **Arm B1:** `sm_86` PTX `A_16` (`LOP3+PRMT+dp4a`) | **`571.367`** | **`1.75019`** | **`118.08024`** | **`115.15721`** | **`233.23745`** | **`12.80843`** | baseline | — |
| **Arm B2:** `sm_86` PTX `A_16` + Live `5-POPC` | **`547.941`** | **`1.82501`** | `130.65771` | `96.29605` | **`226.95376`** | `12.55737` | `+4.275%` | **`0`** |
| **Arm C1:** `i9-11900H` `AVX-512` `8C` (`GPU P8`) | `60.794` | `16.44899` | `226.65052` *(idle)* | `1,277.95499` | `1,504.60551` | `84.43221` | baseline | — |
| **Arm C2:** `AVX-512` `8C` + Live `vpopcntq` (`GPU P8`) | `58.775` | `17.01412` | `231.23638` *(idle)* | `1,357.69352` | `1,588.92990` | `90.05067` | `+3.436%` | **`0`** |

- **Backbone Speedup & Energy Ratios (Exact $\mathbb{Q}$ certified):** Arm B1 is **`7.428x`** faster than PyTorch `BF16` (`571367/76921`) and **`9.398x`** faster than `AVX-512 8C` (`571367/60794`), delivering **`408.901 GB/s`** `BF16`-equivalent effective bandwidth (`115.003 GB/s` physical) at **`6.148x`** lower combined `mJ/tok` than PyTorch `BF16` and **`5.479x`** lower than `AVX-512` CPU-pkg alone.

### 3.2 Full End-to-End Decode (24 Layers + `151,936`-Vocab Tied `BF16` `lm_head` + Greedy `argmax`)

| Arm | Throughput (`tok/s`) | Latency (`ms/tok`) | GPU (`mJ/tok`) | CPU Pkg (`mJ/tok`) | Combined GPU+CPU (`mJ/tok`) | `psys` (`mJ/tok`) | Live Verify Overhead | Syndrome Errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Arm A:** PyTorch `BF16` (RTX 3080) | `71.576` | `13.97115` | `700.76418` | `834.35847` | `1,535.12265` | `87.15130` | — | — |
| **Arm B1:** `sm_86` PTX `A_16` + `BF16` `lm_head` | **`400.580`** | **`2.49638`** | **`191.40979`** | **`121.37732`** | **`312.78711`** | **`18.12172`** | baseline | — |
| **Arm B2:** `sm_86` PTX `A_16` + Live `5-POPC` | **`388.248`** | **`2.57568`** | `201.01429` | `131.79835` | `332.81264` | `19.04042` | `+3.176%` | **`0`** |
| **Arm C1:** `i9-11900H` `AVX-512` `8C` (`GPU P8`) | `31.697` | `31.54842` | `432.14531` *(idle)* | `2,393.15939` | `2,825.30469` | `159.87605` | baseline | — |
| **Arm C2:** `AVX-512` `8C` + Live `vpopcntq` (`GPU P8`) | `29.926` | `33.41520` | `451.03351` *(idle)* | `2,668.59100` | `3,119.62451` | `178.37388` | `+5.917%` | **`0`** |

- **Pareto Conclusion:** On `UX582HS`, **Arm B1 (`sm_86` PTX `A_16`) strictly Pareto-dominates both Arm A (`BF16` GPU) and Arm C (`AVX-512` CPU-only)** in both throughput (`5.597x` vs. `BF16`, `12.638x` vs. `AVX-512`) and energy per token (`4.908x` lower combined `mJ/tok` than `BF16`; `7.651x` lower than CPU-pkg alone) because its race-to-sleep speedup far exceeds active dGPU power draw (`76.4 W` vs. `75.8 W` 8-core AVX-512 turbo). **Arm C1** remains a practical **`31.70 tok/s`** zero-VRAM CPU-only serving tier when the dGPU is idle (`P8`) or occupied by training.

### 3.3 Batched `B=32` Prefill (`168` Real Linear Matrices) & Arm D Cold-Disk `18 B` Transcode + 128-Token Generation

- **Prefill `B=32` (`Arm A3` vs. `Arm B3`):** PyTorch `BF16` cuBLAS reaches `9,473.749 tok/s` (`3.3776 ms/batch`, `6.78 BF16 TFLOPS`, `14.7487 mJ/tok` combined); custom zero-spill `cp.async` + `IMMA` (`Arm B3`, `32x32` tiled SWAR on-the-fly dequant) reaches **`8,383.469 tok/s`** (`3.81704 ms/batch`, `6.00 INT8 TOPS`, `0.885x` cuBLAS, `15.08536 mJ/tok` combined, `l0_q_proj_rel_l2_vs_fp64 = 2.7e-8`).
- **Arm D Cold-Disk `18 B` -> Hot VRAM Transcode:** Real `201,277,440`-byte `18 B / 32 w` binary (`cold_18b_sha256 = d986dd20062e7c5ea80b3564603271a5fb080ce0b6115bef551bf68ac2004fbf`) read from disk in **`65.896 ms`** (`3.054 GB/s`) + GPU H2D and live `5-POPC` SoA transcode in **`25.8844 ms`** (**`13.824 Gweights/s`**, **`0` syndrome errors**) = **`91.7804 ms`** total cold-to-hot load.
- **Real Technical Prompt (`49` tokens) TTFT, 128-Token Generation & Cross-Hardware Parity:**
  - **TTFT:** **`88.0389 ms`** (`sm_86` `A_16`) vs. `239.2169 ms` (PyTorch `BF16`) -> **`2.717x` faster TTFT**.
  - **128-Token Live-Verified Decode:** **`312.11 tok/s`** (`0` syndrome errors) vs. `69.024 tok/s` (PyTorch `BF16`) -> **`4.522x` faster**.
  - **Cross-Engine Parity (`sm_86` GPU vs. `AVX-512` CPU):** First token identical (`25271`), **`16/16` (`100.0%`) exact autoregressive greedy match**, `logits_cosine = 0.99983232`, `logits_rel_l2 = 0.01834247`, `0` syndrome errors.
  - **Parity vs. `BF16`:** First token top-1 exact match (`25271`), `5/5` top-5 overlap, `logits_cosine = 0.94749924`, **teacher-forced `105/128` (`82.03125%`) exact top-1 match**, `4`-token exact free-running prefix match (aggregate weight `rel_l2 = 0.119513`, max matrix `0.170632`).

---

## 4. Honest Scope Boundaries (`provenance_tier: "anchor"`, `CS-13` `N1`/`N0`)

- **`N1-LIVE-QWEN25-05B-A16` (`N1`):** Full 24-layer `Qwen2.5-0.5B` (`168` real matrices, `357,826,560` weights, `122` preserved `BF16` tensors) runs end-to-end on both bare-metal `sm_86` PTX/SASS and `i9-11900H` `AVX-512` with `0` syndrome errors and `16/16` GPU-vs-CPU token parity under Ring-0 `mlos` Zone-B provenance.
- **`N0-IMMA-PREFILL-BELOW-CUBLAS` (`N0`):** Custom `B=32` `cp.async+IMMA` reaches `0.885x` vendor cuBLAS `BF16` (`8,383.5` vs. `9,473.7 tok/s`), avoiding global dequant materialization but not beating cuBLAS `BF16` at `B=32`.
- **`N0-GA104M-NVML-ENERGY-COUNTER` (`N0`):** `GA104M` `nvmlDeviceGetTotalEnergyConsumption` showed `>16x` idle counter drift and was rejected in favor of `100 Hz` trapezoidal `nvmlDeviceGetPowerUsage` integration + `intel-rapl` microjoule counters.
- **`N0-FREE-RUNNING-GREEDY-DIVERGENCE` (`N0`):** Uncalibrated `4.500-bpw` `A16_axis` achieves `105/128` (`82.03%`) teacher-forced top-1 parity vs. `BF16` but diverges in free-running greedy continuation after token 4; furthermore, the tied `151,936`-vocab `BF16` `lm_head` (`272.27 MB`) exceeds all `168` `A_16` linear matrices combined (`201.28 MB`).

---

## 5. Frozen Artifact Manifest

- **Certificate (`aletheion_live_llm_certificate.json`, `0o600`):**
  - `file_sha256`: `9b6c720c783cac8aaa3793e80ee5df87710c07bfc859f702cdabf428d3727620`
  - `payload_sha256`: `376ce86f49977ad2660ac0a66617ec577ca29252010aee33c0c4cf4ae72fad68`
  - `certificate_sha256`: `4c9d097f837341e5a6e083587c305fb30668dbccee3daa886bb4dc8f9f11fff9`

| File | Lines | Bytes | Mode | SHA-256 |
|---|---:|---:|---:|---|
| `aletheion_live_llm_audit.py` | `823` | `37,822` | `0o644` | `dbc17056531ba16f8e488fba9407144f8341475574ca32cc382ac8059860e5f3` |
| `aletheion_live_llm_bench.py` | `1528` | `68,214` | `0o644` | `d56ae882d5a4dda37236cc1dce4e2ad3e39646ca208abe09524cd36f9a44f1dd` |
| `aletheion_live_llm_certificate.json` | `1` | `8,999` | `0o600` | `9b6c720c783cac8aaa3793e80ee5df87710c07bfc859f702cdabf428d3727620` |
| `aletheion_live_llm_oracle.py` | `559` | `24,324` | `0o644` | `99ec5db3a32cda7d7a047c392fa9c979dbc9d2c1dc9c8e282dde2df630ba457e` |
| `avx512_engine_isa_inspection.json` | `19` | `510` | `0o644` | `61c12076575e06245cfa64a63a6a1a15705b4bbbd049af62fe350d4a427bcb4a` |
| `mlos_zone_b_receipt.json` | `93` | `2,650` | `0o600` | `21cdc5582e591b0fc5935a39c51ad32fea7b97ca76d6513d0f99fe97db3854c2` |
| `qwen25_avx512_a16_engine.c` | `518` | `23,333` | `0o644` | `41ee59d9496089f65bfdd6b71e22d9c05ec27eebd20aa3856bbe7c8e7bec338d` |
| `qwen25_live_engine_empirical_results.json` | `549` | `18,825` | `0o644` | `14b6f2ac10447184ffc1f5a72f82b4d3916f70ff62143ccc0c410a0938b84590` |
| `qwen25_sm86_a16_engine.cu` | `1231` | `53,779` | `0o644` | `43f4ebe46c6edd50f2709e2919b5e553f5dcfeb8331a173faf0b008e147fc452` |
| `sm86_engine_sass_inspection.json` | `384` | `8,701` | `0o644` | `6ad91df7a9b144962a3fc8798be2fb7b02fa56a03eed5b34ffc114eea53433ee` |
| `test_aletheion_live_llm_r5.py` | `846` | `46,360` | `0o644` | `49345f9fe91aee5badc2a667720f76b10b42bddaa818a4d0875291e2c9db9923` |
