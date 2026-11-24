# Bare-Metal Ampere `sm_86` PTX/SASS & Intel `AVX-512` `A_16` / `Block-H2` Kernel Foundry

Hardware kernel study accompanying `Moissanite-Labs/siliquant#77` (`shape/hampack.py @ 49283ff`), providing zero-spill Ampere `sm_86` PTX/SASS fused quantized GEMV / batched `cp.async`+`IMMA` Tensor Core GEMM kernels and native Intel `AVX-512` (`vpermb` + `vpopcntdq` + `vpdpbusd`) CPU decoders with verified empirical hardware receipts.

## Highlights

1. **Ampere `sm_86` Zero-Spill PTX/SASS Foundry (`aletheion_sm86_ptx_foundry.cu`):**
   - `15/15` compiled kernels with `0` B stack/spill and `0` SASS `LDL`/`STL` local-memory ops (`sm86_ptx_sass_inspection.json`).
   - Up to **`646.51 Gweights/s`** (`363.92 GB/s` = `102.03%` STREAM Triad, **`1.2930 TB/s` FP16-equiv**) on `8192×14336` (`0` verify failures, `0` syndrome errors), live `1 LOP3 + 5 POPC` verification at `0.4%` cost (`643.29 Gw/s`), zero-shared-memory warp-shuffle `H_16` (`463.89 Gw/s`), and batched `B=32` `cp.async` + `mma.sync.m16n8k32` Tensor Core GEMM (**`14.90 INT8 TOPS`**, `rel_l2 = 3.44e-07`, `11.52×` vs. sequential GEMV).
2. **Intel `i9-11900H` `AVX-512` CPU Decoder (`aletheion_avx512_a16_decoder.c`):**
   - Verified hardware `vpermb`, `vpopcntq`, and `vpdpbusd` emission (`avx512_isa_inspection.json`).
   - Fused VNNI `A_16` GEMV reaches **`60.953 Gw/s`** (`8T`, `max_abs_err = 0.0`, `0` syndrome errors); exact enumerative `18B/32w` `hampack.c` decoding reaches **`223.19 Mw/s`** (`16T`, `0` bit-exact mismatches, `~22.3×` over `~10 Mw/s` baseline).

## Reproduce Receipt Verification

```bash
python3 -I -B studies/aletheion-sm86-ptx-avx512-foundry/test_sm86_avx512_foundry_r5.py
```

See `SM86_PTX_AVX512_FOUNDRY_REPORT.md` for full tables, SASS/ISA counts, and CUDA/GCC compilation commands.
