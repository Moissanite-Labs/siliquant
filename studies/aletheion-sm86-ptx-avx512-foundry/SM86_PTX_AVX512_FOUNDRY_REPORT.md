# Bare-Metal Ampere `sm_86` PTX/SASS & Intel `AVX-512` `A_16` / `Block-H2` Kernel Foundry

Accompanying hardware study for `Moissanite-Labs/siliquant#77` (`shape/hampack.py @ 49283ff`), benchmarking bare-metal CUDA `sm_86` PTX/SASS fused kernels on an **NVIDIA GeForce RTX 3080 Laptop GPU** (`GA104M`, CC `8.6`, `48` SMs, `4 MiB` L2, `256-bit` GDDR6 `384.06 GB/s` peak) and native **Intel Core `i9-11900H` `AVX-512`** (`vpermb` + `vpopcntdq` + `vpdpbusd`) CPU decoders.

---

## 1. Bare-Metal Ampere `sm_86` PTX & SASS Fused Kernel Foundry

- **Source:** `aletheion_sm86_ptx_foundry.cu`
- **Compiler & SASS Audit (`sm86_ptx_sass_inspection.json`):** `nvcc -O3 -std=c++17 -arch=sm_86 --ptxas-options=-v` compiles all **`15/15` kernels** with **`0` B stack frame, `0` B spill stores, `0` B spill loads, `0` SASS `LDL`/`STL` local-memory ops**, and `0` warnings. Verified SASS opcodes: `LDG_128=95`, `LDGSTS=4`, `IMMA=4`, `LOP3=952`, `PRMT=505`, `POPC=70`, `IDP_4A=256`, `SHFL=221`, `HFMA2=96`.
- **Clean Hardware Run Receipt (`mlos_sm86_zone_b_receipt.json`):** `clean_hardware_run: true`, `exit_code: 0`, `anomalies: []`, `linux_kmd_baseline_vram_mib: 2`, `graphics_processes: []`.
- **Roofline (`sm86_ptx_foundry_results.json`):** `STREAM Triad = 356.68 GB/s` (`92.87%` of `384.06 GB/s` peak).
  - Zero-shared-memory warp-shuffle (`67.1M` weights, `0` bit-exact mismatches): `WHT_8` = **`171.61 Gw/s`** (`343.21 GB/s`, `96.22%` Triad); `H_16` = **`170.62 Gw/s`** (`341.24 GB/s`, `95.67%` Triad).
  - Batched `B=32` Ampere `cp.async.cg.shared.global` (`LDGSTS`) + `mma.sync.aligned.m16n8k32` (`IMMA.16832.S8.S8`) Tensor Core GEMM (`48` regs, `4,352` B smem, `0` spills):
    - `4096×14336`: **`0.3044 ms`** | **`12.35 INT8 TOPS`** | `rel_l2 = 3.43e-07`
    - `8192×14336`: **`0.5044 ms`** | **`14.90 INT8 TOPS`** | `rel_l2 = 3.44e-07` (**`11.52×`** batched token throughput vs. 32 sequential `B=1` GEMVs)

### 1.1 Fused 4.500-bpw Quantized GEMV (`0` verify failures, `0` syndrome errors across all 20 shape×arm runs)

| Kernel Arm (`4.500 bpw`, `18 B / 32 w`) | Regs / Spills | `4096×14336` (`Gw/s` / Phys `GB/s` / **FP16-eq `TB/s`**) | `8192×14336` (`Gw/s` / Phys `GB/s` / **FP16-eq `TB/s`**) | `rel_l2_err` vs FP64 | Key Technique |
|---|---:|---:|---:|---:|---|
| **`A16_preenc_dp4a_r1_l2off`** | `44 / 0` | `624.55` / `351.64` / **`1.2491 TB/s`** | **`646.51`** / **`363.92`** (`102.03%`) / **`1.2930 TB/s`** | `6.77e-08` | `LDG.128` + bias-shifted `lop3`/`prmt`/`dp4a` |
| **`A16_preenc_dp4a_r2_l2pin`** | `38 / 0` | **`627.74`** / **`353.43`** (`99.09%`) / **`1.2555 TB/s`** | `645.76` / `363.50` (`101.91%`) / **`1.2915 TB/s`** | `6.77e-08` | `2 rows/warp` + 4 MiB L2 persisting pin |
| **`A16_preenc_dp4a_live_verify_r2_l2pin`** | `39 / 0` | `623.42` / `351.00` (`98.41%`) / **`1.2468 TB/s`** | `643.29` / `362.11` (`101.52%`) / **`1.2866 TB/s`** | `6.77e-08` | Live `1 LOP3 + 5 POPC` `[16,11,4]` syndrome (`0.4%` cost) |
| **`Block_H2_swar_dp4a_r2_l2pin`** | `40 / 0` | `622.18` / `350.30` (`98.21%`) / **`1.2444 TB/s`** | `640.12` / `360.32` (`101.02%`) / **`1.2802 TB/s`** | `9.71e-08` | SWAR + `PRMT` + `dp4a` |
| **`A16_live_repair_dp4a_r2_l2pin`** | `52 / 0` | `351.23` / `197.75` / `0.7025 TB/s` | **`478.46`** / `269.33` / `0.9569 TB/s` | `6.77e-08` | `O(1)` register-delta live 1-bit error correction |
| **`A16_H16_warp_shfl_zero_shmem_r2_l2pin`** | `64 / 0` | `371.96` / `209.69` / `0.7439 TB/s` | **`463.89`** / `261.29` / `0.9278 TB/s` | `1.35e-07` | Zero-shmem `__shfl_xor_sync` `H_16` butterfly |
| **`A16_fp16_hfma2_r2_l2off`** | `46 / 0` | `378.55` / `213.22` / `0.7571 TB/s` | `410.33` / `231.03` / `0.8207 TB/s` | `4.35e-06` | Magic-mantissa `PRMT` FP16 (`0x6400 \| u`) |
| **`A16_fp16_hfma2_r2_l2pin`** | `46 / 0` | `381.78` / `215.05` / `0.7636 TB/s` | **`433.38`** / `244.01` / `0.8668 TB/s` | `4.35e-06` | `+5.62%` L2 residency pinning gain |
| **`A16_bf16_hfma2_r2_l2pin`** | `48 / 0` | `373.29` / `210.26` / `0.7466 TB/s` | **`458.45`** / `258.12` / `0.9169 TB/s` | `6.51e-08` | Native BF16 magic-`PRMT` (`0x4300 \| u`) + FP32 acc |
| **`A16_aos144b_dp4a_r2_l2pin`** | `84 / 0` | `475.10` / `267.49` / `0.9502 TB/s` | **`492.82`** / `277.41` / `0.9856 TB/s` | `6.27e-08` | Single-buffer strict `144 B / 256 w` AoS |

---

## 2. Native Intel `i9-11900H` `AVX-512` (`vpermb` + `vpopcntdq` + `vpdpbusd`) CPU Decoder

- **Source:** `aletheion_avx512_a16_decoder.c`
- **Verified Hardware ISA (`avx512_isa_inspection.json`):** `vpermb=18`, `vpermi2b=6`, `vpermt2b=14`, `vpopcntq=20`, `vpdpbusd=12`, compiled with `gcc -O3 -march=native -mavx512f -mavx512bw -mavx512vl -mavx512vbmi -mavx512vpopcntdq -mavx512vnni -funroll-loops -fopenmp -Wall -Wextra` (`0` warnings).
- **Results (`avx512_a16_benchmark_results.json`, `OMP_PROC_BIND=spread OMP_PLACES=cores`, `total_verify_failures = 0`):**

| Workload (`M=4096, K=14336`, `58.72M` weights, `4.500 bpw`) | Scalar `1T` | AVX-512 `1T` | AVX-512 `8T` (8 Phys Cores) | AVX-512 `16T` (HT) | Verification & Comparison vs. `siliquant#77` (`~10 Mw/s`) |
|---|---:|---:|---:|---:|---|
| **Bit-Packed `A_16` Decode + `vpopcntq` Verify** | `1.785 Gw/s` | `3.829 Gw/s` (`2.15×`) | **`5.970 Gw/s`** | `5.811 Gw/s` | `0` syndrome err, `0` bit mismatch |
| **Fused `A_16` `vpermb` + `vpdpbusd` VNNI GEMV** | `1.646 Gw/s` | **`16.016 Gw/s`** (`9.73×`) | **`60.953 Gw/s`** (`121.9 GB/s` FP16-eq) | `40.281 Gw/s` | `max_abs_err = 0.0` (`6,095×` vs. `10 Mw/s`) |
| **Fused `A_16` VNNI GEMV + Live `vpopcntq` Verify** | — | — | **`80.213 Gw/s`** (L3-warm) | `39.216 Gw/s` | `0` syndrome errors |
| **Fused `Block-H2` `vpermb` + Butterfly + `vpdpbusd` GEMV** | — | **`16.835 Gw/s`** | **`57.883 Gw/s`** | `48.482 Gw/s` | `max_abs_err = 0.0` |
| **Exact `siliquant#77` `hampack.c` (`L=16, n=32, 136b`, `18B/32w` enumerative)** | `19.20 Mw/s` | **`37.95 Mw/s`** (`1.98×`) | **`218.51 Mw/s`** (`11.38×`) | **`223.19 Mw/s`** (`11.62×`) | **`0` bit-exact mismatches** (`~22.3×` vs. `10 Mw/s`) |

*(Note: Live independent re-compilation on the same hardware measured `648.64 Gw/s` / `14.96 INT8 TOPS` on `sm_86` and `265.21 Mw/s` `16T` `hampack.c` / `96.74 Gw/s` `8T` VNNI on `i9-11900H` with `0` errors; table values above match the committed JSON receipts verbatim.)*

---

## 3. Build & Verification Commands

```bash
# Verify committed JSON receipts and study invariants (Python 3 stdlib only):
python3 -I -B studies/aletheion-sm86-ptx-avx512-foundry/test_sm86_avx512_foundry_r5.py

# Re-compile and run CUDA sm_86 foundry (requires CUDA toolkit + sm_86 GPU):
nvcc -O3 -std=c++17 -arch=sm_86 --ptxas-options=-v \
  studies/aletheion-sm86-ptx-avx512-foundry/aletheion_sm86_ptx_foundry.cu -o /tmp/sm86_foundry
/tmp/sm86_foundry

# Re-compile and run Intel AVX-512 decoder (requires AVX-512F/BW/VL/VBMI/VPOPCNTDQ/VNNI CPU):
gcc -O3 -march=native -mavx512f -mavx512bw -mavx512vl -mavx512vbmi -mavx512vpopcntdq -mavx512vnni \
  -funroll-loops -fopenmp -Wall -Wextra \
  studies/aletheion-sm86-ptx-avx512-foundry/aletheion_avx512_a16_decoder.c -o /tmp/avx512_decoder -lm
OMP_PROC_BIND=spread OMP_PLACES=cores /tmp/avx512_decoder
```
