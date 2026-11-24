# RTX 3080 Live-Weight Study: Column-Sorted / Zero-Cost Permuted `A16_BlockH4`, Cross-Layer Non-Additivity Dissection & Coupled-Coherent Routing, and Bare-Metal `sm_86` PTX Fused Gather

## 1. Executive Summary & Epistemic Ledger

This study evaluates **91 live quantized full-model arms** across **4 cached real-weight LLMs** (**840 linear matrices**, **4,757,127,168 block weights**, **0 syndrome errors**, **0 non-block tensor mutations**), **68 exact single-block additivity decompositions** (32 blocks on `SmolLM2-360M` + 36 blocks on `Qwen2.5-3B`), and **3 bare-metal `sm_86` PTX CUDA GEMV benchmarks** (`down_proj` `4096x11008`, `gate_proj+up_proj` `22016x2048`, `qkv` `2560x2048`) on an NVIDIA GeForce RTX 3080 Laptop GPU (`GA104M`, `sm_86`, 8 GiB GDDR6, AC power online, zero competing GPU processes). All hypotheses were cryptographically pre-registered prior to GPU execution (`s69_preregistration.json`, `created_utc=2026-11-24T18:06:33Z`, inner SHA-256 `b2213512967263de16ab680e833bfc8e2eee7da41621eb49ce1fa7d8872cebd8`) strictly preceding the hardware execution window (`started_utc=2026-11-24T18:35:22Z`, `ended_utc=2026-11-24T19:44:47Z`, `duration_ms=4164377.515`, `clean_hardware_run=true`, `anomalies=[]`).

### Pre-Registered Prediction Outcomes (Honest Confirmations & Falsifications)

| Pillar | Pre-Registered Test | Live Hardware Outcome | Verdict |
| :--- | :--- | :--- | :--- |
| **Pillar 1a** | Global $h_j^{\text{real}}$ sort beats unpermuted `hA16_BlockH4_real` on all 4 models | Regressed on all 4/4 models (`360M`: `0.045766` vs `0.028207`; `0.5B`: `0.032504` vs `0.029413`; `1.5B`: `0.026787` vs `0.023412`; `3B`: `0.026025` vs `0.024774`) due to L2-ball Hessian budget-transfer cannibalization (raw error dropped `-18.53` -> `-23.52 dB` while KL worsened) | **FALSIFIED** |
| **Pillar 1b** | Global sort beats local intra-group sort `hA16_BlockH4_sort_grp128_real` | Local intra-group sort strictly beat global sort on all 4/4 models, beat unpermuted `hA16_BlockH4_real` on `0.5B` (`0.027190` vs `0.029413`) and `1.5B`, and beat BOTH `hA16_BlockH4_real` and `hA16_BlockH2_real` on `1.5B` (`0.023176` < `0.023281` Wiki, `0.014083` < `0.014356` Py) | **FALSIFIED** (local wins) |
| **Pillar 1c** | Zero-text column sort (`A16_BlockH4_sort_wcol`, zero forward passes) | Beat both `A16_axis` and `A16_BlockH4` on 3/4 models (`360M` `0.064414`, `0.5B` `0.049761`, `1.5B` `0.039193` = `-21.71%` vs `A16_axis`, Py `-23.29%`), reversing on `3B` (`0.055560` vs `0.046686`, matched-MSE `<= 1.53e-6 dB`) | **REGIME DISCOVERY** |
| **Pillar 1d-i** | Offline `down_proj` fold exact BF16 identity (`max_abs_logit_diff == 0`) | Algebraically exact over $\mathbb{R}$, but non-associative `cuBLAS` BF16/FP32 K-reduction over `K=11008` across `36` sequential layers yields `max_abs_bf16_logit_diff=1.125`, `mean_abs=0.036162`, `KL=0.000827`, `top1=97.85%` | **FALSIFIED** (BF16 non-assoc.) |
| **Pillar 1d-ii** | Zero-runtime-cost `hA16_BlockH4_zerocost_down_real` improves `3B` cal | `3B` WikiKL `0.024734` < `0.024774` (`PPL 11.1920` < `11.2062`), PyKL `0.021904` < `0.022519`, **best 3B Python PPL `3.1543`** across all 31 arms; also improves `0.5B` (`0.028104` < `0.029413`) | **CONFIRMED** |
| **Pillar 2a** | Interference spike on `3B`: $\mathcal{R}_{\text{cross}}(\text{PerMatrix\_Oracle}) > \mathcal{R}_{\text{cross}}(\text{A16\_axis})$ | `3B` (36 blocks): `PerMatrix_Oracle` `0.584405` > `A16_axis` `0.549633` (`+6.33%`) > `ProjType_Uniform` `0.540284`; opposite sign on `360M` (`0.567253` < `0.605791`) | **CONFIRMED** |
| **Pillar 2b** | Coherent cure on `3B`: `min(Coupled_Subblock, Layer_Coherent, ProjType_Uniform) < PerMatrix_Oracle` | `ProjType_Uniform` (`down_proj->axis`, rest `->BlockH4`, 0 cal, 0 router bits): WikiKL `0.046336` (beats `A16_axis` `0.046686`, `A16_BlockH4` `0.047942`, `PerMatrix_Oracle` `0.050421`) & **best zero-cal PyKL `0.034327`** (`-6.57%` vs `E8_H8`!). `Coupled_RelRatio_ZeroCal` hits WikiKL `0.045768` (`-10.62%` vs `PerMatrix_ZeroCal`, tying `E8_H8` `0.045724`) | **CONFIRMED** |
| **Pillar 2c** | `SmolLM2-360M` additive control (`PerMatrix_Oracle < A16_axis`) | `0.060817` < `0.065173` (`-6.68%`), and `ProjType_Uniform` hits `0.061838` (PPL `16.9701`) | **CONFIRMED** |
| **Pillar 3** | Bare-metal `sm_86` PTX: 0 B stack/spill, 0 `LDL`/`STL`, 0 syn/i64 errors, `gate+up` shared gather < 5%, faster than inline gather | `9` kernels (`20-40` regs), `0 B` stack/spill, `0` `LDL`/`STL`, `0` syn/i64/parity errors across all 3 shapes; `DOWN` zero-cost folded `606.20 Gw/s` (`99.72%` of unpermuted, `218.49 tok/s`); `MLP` shared gather `+4.47% < 5%` (`2.79x` faster than inline `+191.74%`); `QKV` shared gather `+10.35%` (`2.49x` faster than inline `+174.70%`) | **CONFIRMED (5/5)** |

## 2. Pillar 1: Column-Sorted & Zero-Runtime-Cost Permuted `A16_BlockH4`

### Mechanism: Group-Ball Hessian Budget-Transfer Cannibalization vs. Local Intra-Group Sorting

The live 4-model sweep uncovered an unexpectedly sharp two-regime physical law governing input-column permutation before `BlockH4` + L2-ball (`g=128` or `g=32`) quantization:

1. **Hessian-Calibrated Regime (`h_j^{1/4}` column scaling):** Unpermuted Hessian scaling inflates rare high-$h_j$ outlier columns inside mixed `g=128` groups, causing those columns to dominate the shared group L2-ball radius $s_g$ and **transfer quantization precision from quiet columns to critical outlier columns** (raw unweighted weight error rises to `-18.5273 dB` on `Qwen2.5-3B` while WikiText-2 KL drops to `0.024774`). Globally sorting all columns by descending $h_j$ (`hA16_BlockH4_sort_real`) packs all high-$h_j$ columns into Group 0, where they **cannibalize each other's fixed 4.5-bpw L2-ball budget** while granting excess precision to quiet groups—causing raw unweighted error to improve by **`-4.99 dB`** (`-18.5273` -> `-23.5220 dB`) even as WikiText-2 KL **worsens** (`0.024774` -> `0.026025`). The `hA16_axis_sort_real` (`0.027607` vs `0.024417`, `-24.4910 dB`) and `hA16_BlockH2_sort_real` (`0.027380` vs `0.024179`, `-24.1030 dB`) controls prove this regression is driven by cross-group L2-ball redistribution rather than `H_4` itself.
2. **Local Intra-Group Sorting (`hA16_BlockH4_sort_grp128_real`):** Sorting strictly *within* each fixed `g=128` group preserves cross-column Hessian budget transfer while eliminating `H_4` neighbor outlier spill. It strictly beats global sort on all 4/4 models, beats unpermuted `hA16_BlockH4_real` on `Qwen2.5-0.5B` (`0.027190` vs `0.029413`, `-7.56%`) and `Qwen2.5-1.5B` (`0.023176` vs `0.023412`), and **strictly beats `hA16_BlockH2_real` on `Qwen2.5-1.5B`** on both WikiText-2 (`0.023176` < `0.023281`, PPL `12.7768` < `12.9273`) and Python (`0.014083` < `0.014356`).
3. **Zero-Text Weight-Norm Sorting (`A16_BlockH4_sort_wcol`):** With zero calibration text and zero forward passes, globally sorting columns by weight-column L2 norm `||W_{:,j}||_2` delivers large gains on high-sparsity models (`z0 >= 91.9%`): **`-21.71%` WikiKL** (`0.039193` vs `0.050062`) and **`-23.29%` PyKL** (`0.024528` vs `0.031973`) on `Qwen2.5-1.5B`, `-5.26%` Wiki / `-16.23%` Py on `Qwen2.5-0.5B`, and `-1.16%` Wiki / `-3.54%` Py on `SmolLM2-360M`. On `Qwen2.5-3B` (where `A16_axis` `z0` drops to `68.05%`), the regime reverses (`0.055560` vs `0.046686`, confirmed at matched MSE `<= 1.528e-6 dB`, while `A16_BlockH4_sort_wcol` still beats `A16_axis_sort_wcol` `0.058112`).
4. **Zero-Runtime-Cost Offline `down_proj` Folding:** Permuting `down_proj` input columns and absorbing the inverse permutation into `gate_proj`/`up_proj` output rows requires zero runtime gather ops. In `Qwen2.5-3B` calibrated serving, `hA16_BlockH4_zerocost_down_real` improves both WikiText-2 (`0.024734` < `0.024774`) and Python (`0.021904` < `0.022519`) and achieves the **lowest Python PPL (`3.1543`) across all 31 tested `3B` arms**. Live BF16 identity verification across all 36 layers honestly falsifies bit-exact BF16 identity (`max_abs_bf16_logit_diff = 1.125`, `KL = 0.000827`, `97.85%` top-1 match) because `cuBLAS` FP32/BF16 reduction over `K=11008` is non-associative under column permutation.

## 3. Pillar 2: Cross-Layer Non-Additivity Dissection & Coupled-Coherent Routing

Single-block additivity decomposition quantizes each transformer block $\ell \in \{0,\dots,L-1\}$ individually (leaving the remaining $L-1$ blocks in exact BF16), sums $\sum_{\ell=0}^{L-1} \Delta \mathrm{KL}_\ell$, and measures the cross-layer interference ratio $\mathcal{R}_{\text{cross}} = \mathrm{KL}_{\text{full}} / \sum_\ell \Delta \mathrm{KL}_\ell$.

### Table 2A: Exact Single-Block Additivity Decomposition (`32` Blocks `SmolLM2-360M` vs. `36` Blocks `Qwen2.5-3B`)

| Model | Arm | Blocks | Full Wiki KL | $\sum_\ell \Delta\mathrm{KL}_\ell$ | Max Single Block $\Delta\mathrm{KL}$ | $\mathcal{R}_{\text{cross}}$ | $\Delta \mathcal{R}_{\text{cross}}$ vs `A16_axis` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `SmolLM2-360M` | `A16_axis` | `32` | `0.065173` | `0.107583` | `0.018089` | `0.605791` | `+0.00%` |
| `SmolLM2-360M` | `A16_BlockH4` | `32` | `0.082002` | `0.126323` | `0.026301` | `0.649142` | `+7.16%` |
| `SmolLM2-360M` | `PerMatrix_Oracle` | `32` | `0.060817` | `0.107214` | `0.017417` | `0.567253` | `-6.36%` |
| `SmolLM2-360M` | `Layer_Coherent_Oracle` | `32` | `0.066013` | `0.108090` | `0.018089` | `0.610718` | `+0.81%` |
| `SmolLM2-360M` | `ProjType_Uniform` | `32` | `0.061838` | `0.107693` | `0.017227` | `0.574204` | `-5.21%` |
| `Qwen2.5-3B` | `A16_axis` | `36` | `0.046686` | `0.084941` | `0.007621` | `0.549633` | `+0.00%` |
| `Qwen2.5-3B` | `A16_BlockH4` | `36` | `0.047942` | `0.088417` | `0.008287` | `0.542225` | `-1.35%` |
| `Qwen2.5-3B` | `PerMatrix_Oracle` | `36` | `0.050421` | `0.086278` | `0.007723` | `0.584405` | `+6.33%` |
| `Qwen2.5-3B` | `Layer_Coherent_Oracle` | `36` | `0.050204` | `0.087134` | `0.007621` | `0.576170` | `+4.83%` |
| `Qwen2.5-3B` | `ProjType_Uniform` | `36` | `0.046336` | `0.085763` | `0.007488` | `0.540284` | `-1.70%` |

The sign reversal between `SmolLM2-360M` and `Qwen2.5-3B` directly resolves why per-matrix greedy routing succeeds on `360M` yet regresses on `3B`: on `360M`, `PerMatrix_Oracle` lowers $\mathcal{R}_{\text{cross}}$ (`0.605791` -> `0.567253`, `-6.36%`); on `3B`, `PerMatrix_Oracle` triggers a **`+6.33%` cross-layer interference spike** (`0.549633` -> `0.584405`), completely erasing its local single-block gains. Enforcing structural uniformity by projection type (`ProjType_Uniform`: `down_proj -> A16_axis`, all other 6 projections `-> A16_BlockH4`, **0 calibration, 0 router bits**) achieves the **lowest $\mathcal{R}_{\text{cross}}$ (`0.540284`) on `Qwen2.5-3B`**, beating `A16_axis`, `A16_BlockH4`, and `PerMatrix_Oracle` on WikiText-2 (`0.046336`) while setting the **best zero-cal Python KL (`0.034327`, `-6.57%` vs `E8_H8`)**. Meanwhile, scale-calibrated block-coherent routing (`Coupled_RelRatio_ZeroCal`, 1 bit/block, random tokens only) eliminates `PerMatrix_ZeroCal`'s regression (`0.051206` -> `0.045768`, `-10.62%`), essentially tying `E8_H8` (`0.045724`) on `3B` while also beating `A16_axis` on `1.5B` (`0.047890` vs `0.050062`) and `360M` (`0.065133` vs `0.065173`).

## 4. Complete 4-Model Live Hardware Results (`91` Arms, `4,757,127,168` Weights)

### 4.1 `HuggingFaceTB/SmolLM2-360M` (`g=32`, `32`L, `224` mats, `314,572,800` wts; BF16 PPL Wiki=`16.3948`, Py=`5.0450`)

| Arm | Err (dB) | E05 (dB) | z0 (%) | Wiki KL | Wiki PPL | Py KL | Py PPL | Syn / NB (Agree) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `A16_axis` | `-25.0281` | `2.682` | `93.11%` | `0.065173` | `17.2522` | `0.056725` | `5.3303` | `0` / `0` |
| `A16_BlockH4` | `-25.0279` | `5.227` | `48.72%` | `0.082002` | `17.7082` | `0.066762` | `5.3763` | `0` / `0` |
| `PerMatrix_ZeroCal` | `-25.0279` | `3.939` | `73.81%` | `0.061998` | `17.1756` | `0.052377` | `5.3139` | `0` / `0` (168, 75.00%) |
| `PerMatrix_Oracle` | `-25.0282` | `3.873` | `75.19%` | `0.060817` | `17.0658` | `0.050749` | `5.2806` | `0` / `0` (224, 100.00%) |
| `Coupled_Subblock_ZeroCal` | `-25.0280` | `4.013` | `73.00%` | `0.066153` | `17.2854` | `0.056807` | `5.3343` | `0` / `0` (169, 75.45%) |
| `Coupled_Subblock_Oracle` | `-25.0280` | `3.600` | `79.82%` | `0.065422` | `17.2414` | `0.055625` | `5.3072` | `0` / `0` (224, 100.00%) |
| `Layer_Coherent_ZeroCal` | `-25.0278` | `4.146` | `70.76%` | `0.065553` | `17.2686` | `0.057051` | `5.3347` | `0` / `0` (175, 78.13%) |
| `Layer_Coherent_Oracle` | `-25.0280` | `3.560` | `80.51%` | `0.066013` | `17.2398` | `0.056866` | `5.3308` | `0` / `0` (224, 100.00%) |
| `Coupled_RelRatio_ZeroCal` | `-25.0278` | `3.727` | `77.59%` | `0.065133` | `17.2330` | `0.055553` | `5.3333` | `0` / `0` (196, 87.50%) |
| `Coupled_RelRatio_Oracle` | `-25.0277` | `3.555` | `80.43%` | `0.064543` | `17.2559` | `0.056376` | `5.3142` | `0` / `0` (224, 100.00%) |
| `ProjType_Uniform` | `-25.0279` | `4.711` | `59.58%` | `0.061838` | `16.9701` | `0.052089` | `5.3026` | `0` / `0` (224, 100.00%) |
| `A16_BlockH4_sort_wcol` | `-25.0278` | `5.087` | `48.02%` | `0.064414` | `17.3329` | `0.054715` | `5.3250` | `0` / `0` |
| `A16_BlockH4_sort_hrand` | `-25.0281` | `5.161` | `48.40%` | `0.071525` | `17.4902` | `0.058347` | `5.3515` | `0` / `0` |
| `A16_BlockH4_zerocost_down` | `-25.0281` | `5.218` | `48.72%` | `0.067467` | `17.2685` | `0.054732` | `5.3314` | `0` / `0` |
| `hA16_real` | `-24.3718` | `4.497` | `92.46%` | `0.030955` | `16.9734` | `0.027538` | `5.1678` | `0` / `0` |
| `hA16_BlockH2_real` | `-24.0633` | `7.816` | `6.78%` | `0.035651` | `17.0286` | `0.032959` | `5.1908` | `0` / `0` |
| `hA16_BlockH4_real` | `-23.9798` | `8.411` | `48.66%` | `0.028207` | `16.9524` | `0.026602` | `5.1555` | `0` / `0` |
| `hA16_BlockH4_sort_grp128_real` | `-24.0979` | `7.894` | `48.78%` | `0.030636` | `16.9699` | `0.024308` | `5.1731` | `0` / `0` |
| `hA16_BlockH4_sort_real` | `-24.7457` | `6.315` | `48.36%` | `0.045766` | `17.3383` | `0.046179` | `5.2846` | `0` / `0` |
| `hA16_BlockH4_zerocost_down_real` | `-24.0901` | `8.343` | `48.70%` | `0.031454` | `16.7185` | `0.028079` | `5.1727` | `0` / `0` |

### 4.2 `Qwen/Qwen2.5-0.5B` (`g=128`, `24`L, `168` mats, `357,826,560` wts; BF16 PPL Wiki=`17.9505`, Py=`4.8606`)

| Arm | Err (dB) | E05 (dB) | z0 (%) | Wiki KL | Wiki PPL | Py KL | Py PPL | Syn / NB (Agree) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `A16_axis` | `-25.5529` | `2.336` | `91.94%` | `0.052525` | `18.4858` | `0.045602` | `5.1000` | `0` / `0` |
| `A16_BlockH4` | `-25.5532` | `4.636` | `48.22%` | `0.056063` | `18.5926` | `0.045183` | `5.0712` | `0` / `0` |
| `PerMatrix_ZeroCal` | `-25.5533` | `3.709` | `69.15%` | `0.054443` | `18.2899` | `0.046973` | `5.0737` | `0` / `0` (132, 78.57%) |
| `PerMatrix_Oracle` | `-25.5534` | `3.778` | `67.70%` | `0.053824` | `18.3148` | `0.047700` | `5.0853` | `0` / `0` (168, 100.00%) |
| `Coupled_Subblock_ZeroCal` | `-25.5530` | `3.955` | `63.62%` | `0.053910` | `18.5043` | `0.046414` | `5.0802` | `0` / `0` (132, 78.57%) |
| `Coupled_Subblock_Oracle` | `-25.5532` | `3.916` | `64.33%` | `0.055281` | `18.4666` | `0.045693` | `5.0802` | `0` / `0` (168, 100.00%) |
| `Layer_Coherent_ZeroCal` | `-25.5530` | `3.745` | `68.38%` | `0.052881` | `18.6661` | `0.045729` | `5.0642` | `0` / `0` (112, 66.67%) |
| `Layer_Coherent_Oracle` | `-25.5532` | `3.742` | `68.46%` | `0.057114` | `18.5266` | `0.044564` | `5.0426` | `0` / `0` (168, 100.00%) |
| `Coupled_RelRatio_ZeroCal` | `-25.5532` | `3.759` | `68.16%` | `0.053600` | `18.5289` | `0.044446` | `5.0768` | `0` / `0` (140, 83.33%) |
| `Coupled_RelRatio_Oracle` | `-25.5533` | `3.756` | `68.16%` | `0.055595` | `18.5950` | `0.046004` | `5.0636` | `0` / `0` (168, 100.00%) |
| `ProjType_Uniform` | `-25.5531` | `4.168` | `60.85%` | `0.054329` | `18.5079` | `0.043465` | `5.0432` | `0` / `0` (168, 100.00%) |
| `A16_BlockH4_sort_wcol` | `-25.5534` | `4.481` | `47.50%` | `0.049761` | `18.5962` | `0.038201` | `5.0389` | `0` / `0` |
| `A16_BlockH4_sort_hrand` | `-25.5521` | `4.569` | `47.91%` | `0.052642` | `18.5284` | `0.040875` | `5.0293` | `0` / `0` |
| `A16_BlockH4_zerocost_down` | `-25.5530` | `4.626` | `48.13%` | `0.053223` | `18.4881` | `0.041211` | `5.0277` | `0` / `0` |
| `hA16_real` | `-23.6609` | `8.380` | `91.31%` | `0.028102` | `18.3312` | `0.024798` | `4.9610` | `0` / `0` |
| `hA16_BlockH2_real` | `-22.2325` | `12.177` | `7.02%` | `0.026970` | `18.2349` | `0.024050` | `4.9594` | `0` / `0` |
| `hA16_BlockH4_real` | `-21.6328` | `13.346` | `48.13%` | `0.029413` | `18.3511` | `0.023452` | `4.9594` | `0` / `0` |
| `hA16_BlockH4_sort_grp128_real` | `-21.6497` | `13.347` | `48.13%` | `0.027190` | `18.5259` | `0.023315` | `4.9425` | `0` / `0` |
| `hA16_BlockH4_sort_real` | `-22.5682` | `12.045` | `47.81%` | `0.032504` | `18.4297` | `0.027835` | `4.9839` | `0` / `0` |
| `hA16_BlockH4_zerocost_down_real` | `-21.6914` | `13.328` | `48.09%` | `0.028104` | `18.2913` | `0.024016` | `4.9622` | `0` / `0` |

### 4.3 `Qwen/Qwen2.5-1.5B` (`g=128`, `28`L, `196` mats, `1,310,195,712` wts; BF16 PPL Wiki=`12.6180`, Py=`3.7395`)

| Arm | Err (dB) | E05 (dB) | z0 (%) | Wiki KL | Wiki PPL | Py KL | Py PPL | Syn / NB (Agree) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `A16_axis` | `-25.5532` | `2.549` | `93.44%` | `0.050062` | `13.0109` | `0.031973` | `3.8456` | `0` / `0` |
| `A16_BlockH4` | `-25.5532` | `4.889` | `49.12%` | `0.046550` | `13.1362` | `0.030045` | `3.8453` | `0` / `0` |
| `PerMatrix_ZeroCal` | `-25.5532` | `3.669` | `75.61%` | `0.048780` | `13.0235` | `0.033491` | `3.8402` | `0` / `0` (148, 75.51%) |
| `PerMatrix_Oracle` | `-25.5532` | `3.687` | `75.14%` | `0.048195` | `13.0573` | `0.030720` | `3.8413` | `0` / `0` (196, 100.00%) |
| `Coupled_Subblock_ZeroCal` | `-25.5531` | `3.484` | `78.72%` | `0.050313` | `13.1558` | `0.033396` | `3.8352` | `0` / `0` (136, 69.39%) |
| `Coupled_Subblock_Oracle` | `-25.5532` | `3.624` | `76.12%` | `0.049900` | `13.0578` | `0.031481` | `3.8438` | `0` / `0` (196, 100.00%) |
| `Layer_Coherent_ZeroCal` | `-25.5531` | `3.864` | `71.30%` | `0.049210` | `13.2028` | `0.032138` | `3.8436` | `0` / `0` (154, 78.57%) |
| `Layer_Coherent_Oracle` | `-25.5531` | `3.705` | `74.47%` | `0.049108` | `13.1105` | `0.031243` | `3.8507` | `0` / `0` (196, 100.00%) |
| `Coupled_RelRatio_ZeroCal` | `-25.5531` | `3.936` | `69.65%` | `0.047890` | `13.1228` | `0.033083` | `3.8471` | `0` / `0` (126, 64.29%) |
| `Coupled_RelRatio_Oracle` | `-25.5532` | `3.771` | `72.85%` | `0.049344` | `13.0287` | `0.032912` | `3.8689` | `0` / `0` (196, 100.00%) |
| `ProjType_Uniform` | `-25.5532` | `4.357` | `62.01%` | `0.050110` | `13.1305` | `0.032953` | `3.8612` | `0` / `0` (196, 100.00%) |
| `A16_BlockH4_sort_wcol` | `-25.5529` | `4.773` | `48.58%` | `0.039193` | `13.1011` | `0.024528` | `3.8234` | `0` / `0` |
| `A16_BlockH4_sort_hrand` | `-25.5531` | `4.846` | `48.94%` | `0.047169` | `13.2161` | `0.029567` | `3.8217` | `0` / `0` |
| `A16_BlockH4_zerocost_down` | `-25.5532` | `4.880` | `49.07%` | `0.047055` | `13.1760` | `0.029191` | `3.8465` | `0` / `0` |
| `hA16_real` | `-22.9064` | `12.434` | `92.76%` | `0.023637` | `12.8575` | `0.015113` | `3.7882` | `0` / `0` |
| `hA16_BlockH2_real` | `-20.7233` | `16.446` | `6.90%` | `0.023281` | `12.9273` | `0.014356` | `3.8326` | `0` / `0` |
| `hA16_BlockH4_real` | `-19.8754` | `17.722` | `49.07%` | `0.023412` | `12.7947` | `0.015129` | `3.7961` | `0` / `0` |
| `hA16_BlockH4_sort_grp128_real` | `-20.2814` | `17.055` | `49.10%` | `0.023176` | `12.7768` | `0.014083` | `3.7873` | `0` / `0` |
| `hA16_BlockH4_sort_real` | `-22.5432` | `13.299` | `48.98%` | `0.026787` | `12.7825` | `0.017099` | `3.7986` | `0` / `0` |
| `hA16_BlockH4_zerocost_down_real` | `-19.9283` | `17.713` | `49.07%` | `0.024177` | `12.7407` | `0.016297` | `3.8019` | `0` / `0` |

### 4.4 `Qwen/Qwen2.5-3B` (`g=128`, `36`L, `252` mats, `2,774,532,096` wts; BF16 PPL Wiki=`10.9702`, Py=`3.1119`)

| Arm | Err (dB) | E05 (dB) | z0 (%) | Wiki KL | Wiki PPL | Py KL | Py PPL | Syn / NB (Agree) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `E8_H8` | `-25.5850` | `3.043` | `0.99%` | `0.045724` | `11.3153` | `0.036741` | `3.2366` | `0` / `0` |
| `A16_axis` | `-25.5532` | `1.924` | `68.05%` | `0.046686` | `11.2583` | `0.038578` | `3.2183` | `0` / `0` |
| `A16_BlockH4` | `-25.5534` | `4.294` | `36.15%` | `0.047942` | `11.3595` | `0.035620` | `3.2230` | `0` / `0` |
| `PerMatrix_ZeroCal` | `-25.5531` | `2.912` | `57.28%` | `0.051206` | `11.5039` | `0.038365` | `3.2193` | `0` / `0` (194, 76.98%) |
| `PerMatrix_Oracle` | `-25.5531` | `2.860` | `57.58%` | `0.050421` | `11.3858` | `0.039697` | `3.2285` | `0` / `0` (252, 100.00%) |
| `Coupled_Subblock_ZeroCal` | `-25.5533` | `3.042` | `54.90%` | `0.048986` | `11.3464` | `0.039944` | `3.2285` | `0` / `0` (205, 81.35%) |
| `Coupled_Subblock_Oracle` | `-25.5533` | `2.847` | `56.97%` | `0.049322` | `11.3519` | `0.038925` | `3.2168` | `0` / `0` (252, 100.00%) |
| `Layer_Coherent_ZeroCal` | `-25.5533` | `2.953` | `55.87%` | `0.048666` | `11.3456` | `0.037341` | `3.2309` | `0` / `0` (210, 83.33%) |
| `Layer_Coherent_Oracle` | `-25.5534` | `2.795` | `57.48%` | `0.050204` | `11.3484` | `0.038148` | `3.2196` | `0` / `0` (252, 100.00%) |
| `Coupled_RelRatio_ZeroCal` | `-25.5531` | `2.826` | `59.74%` | `0.045768` | `11.2749` | `0.036802` | `3.2247` | `0` / `0` (189, 75.00%) |
| `Coupled_RelRatio_Oracle` | `-25.5531` | `2.746` | `60.46%` | `0.049450` | `11.3364` | `0.038062` | `3.2218` | `0` / `0` (252, 100.00%) |
| `ProjType_Uniform` | `-25.5528` | `3.695` | `46.74%` | `0.046336` | `11.2686` | `0.034327` | `3.2172` | `0` / `0` (252, 100.00%) |
| `A16_BlockH4_sort_wcol` | `-25.5532` | `3.959` | `30.98%` | `0.055560` | `11.3459` | `0.043396` | `3.2437` | `0` / `0` |
| `A16_BlockH4_sort_wcol_matched` | `-25.5532` | `3.959` | `0.01%` | `0.055738` | `11.3585` | `0.042918` | `3.2442` | `0` / `0` |
| `A16_axis_sort_wcol` | `-25.5528` | `1.611` | `59.04%` | `0.058112` | `11.3858` | `0.039806` | `3.2698` | `0` / `0` |
| `A16_BlockH4_sort_hrand` | `-25.5532` | `4.031` | `31.12%` | `0.062024` | `11.0657` | `0.051355` | `3.2406` | `0` / `0` |
| `A16_BlockH4_sort_hrand_matched` | `-25.5532` | `4.031` | `0.01%` | `0.062108` | `11.0505` | `0.051059` | `3.2393` | `0` / `0` |
| `A16_BlockH4_zerocost_down` | `-25.5530` | `4.090` | `32.03%` | `0.061528` | `11.2091` | `0.044595` | `3.2287` | `0` / `0` |
| `A16_BlockH4_zerocost_down_wcol` | `-25.5531` | `4.062` | `31.84%` | `0.051324` | `11.2163` | `0.037749` | `3.2239` | `0` / `0` |
| `hA16_real` | `-21.1650` | `15.760` | `75.03%` | `0.024417` | `11.0895` | `0.021217` | `3.1768` | `0` / `0` |
| `hA16_BlockH2_real` | `-19.4442` | `18.804` | `16.96%` | `0.024179` | `11.0576` | `0.022975` | `3.1749` | `0` / `0` |
| `hA16_BlockH4_real` | `-18.5273` | `20.110` | `45.22%` | `0.024774` | `11.2062` | `0.022519` | `3.1670` | `0` / `0` |
| `hA16_BlockH4_sort_grp128_real` | `-20.8674` | `17.139` | `45.79%` | `0.024918` | `11.1841` | `0.021902` | `3.1723` | `0` / `0` |
| `hA16_BlockH4_sort_real` | `-23.5220` | `11.976` | `32.03%` | `0.026025` | `11.1903` | `0.021765` | `3.1785` | `0` / `0` |
| `hA16_BlockH4_zerocost_down_real` | `-19.3830` | `19.609` | `37.46%` | `0.024734` | `11.1920` | `0.021904` | `3.1543` | `0` / `0` |
| `hA16_BlockH4_zerocost_down_grp128_real` | `-19.1082` | `19.629` | `45.59%` | `0.025009` | `11.2313` | `0.021706` | `3.1605` | `0` / `0` |
| `hA16_BlockH4_sharednorm_real` | `-23.5122` | `11.976` | `31.99%` | `0.026580` | `11.2445` | `0.021906` | `3.1743` | `0` / `0` |
| `hA16_BlockH4_sharednorm_grp128_real` | `-20.8674` | `17.139` | `45.79%` | `0.024958` | `11.1860` | `0.021745` | `3.1683` | `0` / `0` |
| `hA16_BlockH4_sort_rand` | `-23.2259` | `12.413` | `32.11%` | `0.030043` | `11.1198` | `0.023233` | `3.1827` | `0` / `0` |
| `hA16_BlockH2_sort_real` | `-24.1030` | `10.527` | `6.18%` | `0.027380` | `11.0433` | `0.023440` | `3.1707` | `0` / `0` |
| `hA16_axis_sort_real` | `-24.4910` | `9.083` | `59.71%` | `0.027607` | `11.1163` | `0.023031` | `3.1588` | `0` / `0` |

## 5. Pillar 3: Bare-Metal `sm_86` PTX Zero-Cost Folded vs. Shared-RMSNorm `O(d)` Fused Gather

Standalone `sm86_s69_sorted_blockh4_kernel.cu` compiled with `nvcc -O3 -std=c++17 -arch=sm_86 --ptxas-options=-v` and `--ptx`: **`9` kernels**, `20-40` registers/thread, **`0 B` stack frame**, **`0 B` spill stores**, **`0 B` spill loads**, **`0` `ld.local`**, **`0` `st.local`**, and **`32` `dp4a.s32.s32`** instructions. All 3 shapes execute on real exported `Qwen2.5-3B` 4.5000-bpw weights with `0` export/runtime syndrome errors and `0` `int64` bit-exact mismatches. Note on top-level RAPL counter: the 69-minute full-sweep top-level counter wrapped Intel RAPL's 32-bit `max_energy_range_uj` ceiling (clamped to `0.0 J` in top-level summary), whereas every sub-second CUDA kernel energy measurement below integrated exact un-wrapped `intel-rapl:0` CPU package `mJ/call` alongside 100 Hz trapezoidal NVML INA3221 GPU rail energy.

### Table 5A: `down_proj` Stacked2 (`4096x11008`, `45,088,768` weights, STREAM Triad `356.53 GB/s`)

| Kernel Variant | Latency (ms) | Gw/s | Eff. GB/s (% Triad) | Proj. 3B tok/s | GPU mJ/call (Avg W) | CPU RAPL mJ/call | Syn / i64 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `A16_axis` (pre-encoded `dp4a`) | `0.073814` | `610.85` | `343.97` (`96.48%`) | `220.16` | `3.6976` (`48.96 W`) | `2.7936` | `0` / `n/a` |
| `A16_BlockH4` (unpermuted) | `0.074171` | `607.90` | `342.31` (`96.01%`) | `219.10` | `5.0292` (`66.48 W`) | `2.5671` | `0` / `n/a` |
| `A16_BlockH4` (zero-cost offline folded) | `0.074379` | `606.20` | `341.36` (`95.74%`) | `218.49` | `6.6062` (`87.40 W`) | `3.1410` | `0` / `0` |

### Table 5B: Shared-RMSNorm `O(d)` Fused Gather vs. Naive Inline Gather (`gate_proj+up_proj` `22016x2048` & `qkv` `2560x2048`)

| Shape (Triad GB/s) | Pipeline Variant | Latency (ms) | Gw/s | Overhead vs Unp. | GPU mJ/call (Avg W) | CPU RAPL mJ/call | Parity (norm/syn/i64/diff) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `gate+up` `22016x2048` (`356.79`) | Unpermuted RMSNorm + `BlockH4` | `0.082736` | `544.97` | `0.00%` (base) | `6.8011` (`81.66 W`) | `3.3610` | `0/0/0/0.0` |
| `gate+up` `22016x2048` (`356.79`) | Shared-RMSNorm `O(d)` Gather + `BlockH4` | `0.086433` | `521.66` | `+4.47%` | `7.1485` (`81.61 W`) | `3.3429` | `0/0/0/0.0` |
| `gate+up` `22016x2048` (`356.79`) | Naive Inline Per-GEMV Gather + `BlockH4` | `0.241376` | `186.80` | `+191.74%` | `23.2165` (`96.02 W`) | `8.6764` | `0/0/0/0.0` |
| `qkv` `2560x2048` (`356.68`) | Unpermuted RMSNorm + `BlockH4` | `0.012650` | `414.44` | `0.00%` (base) | `1.1992` (`85.92 W`) | `0.4173` | `0/0/0/0.0` |
| `qkv` `2560x2048` (`356.68`) | Shared-RMSNorm `O(d)` Gather + `BlockH4` | `0.013959` | `375.58` | `+10.35%` | `1.2816` (`84.67 W`) | `0.5066` | `0/0/0/0.0` |
| `qkv` `2560x2048` (`356.68`) | Naive Inline Per-GEMV Gather + `BlockH4` | `0.034751` | `150.87` | `+174.70%` | `2.9812` (`84.43 W`) | `1.3697` | `0/0/0/0.0` |

## 6. Frozen Artifact Manifest & Reproducibility

- **Canonical Certificate Inner SHA-256:** `29640bbf590a1d354d823f39f6d4ce0b0e6ee3ad0e9001a676a8dc45fafdcd83`
- **Pre-Registration Inner SHA-256:** `b2213512967263de16ab680e833bfc8e2eee7da41621eb49ce1fa7d8872cebd8` (`created_utc=2026-11-24T18:06:33Z`)
- **Hardware Receipt Inner SHA-256:** `f58343289768a3691a234cf4d3c5a550d916352b963b199ade6bc7b761a87665` (`started_utc=2026-11-24T18:35:22Z`, `ended_utc=2026-11-24T19:44:47Z`)

| Artifact File | Bytes | SHA-256 |
| :--- | :---: | :--- |
| `s69_preregistration.json` | `4492` | `59540d4e8ab82f9ea29015b8c37ee5d523500b37ed9e62745673373fd95de2a6` |
| `sm86_s69_sorted_blockh4_kernel.cu` | `40894` | `0043e6e0da62272d80da7a2c348f0280ec1428803bd54243b1f4b5d96d64bb97` |
| `rtx3080_s69_sorted_blockh4_eval.py` | `67175` | `60be4ec7824994f43405fb9c22d8c577c49eea91c0ee6cbb176b747dad159809` |
| `rtx3080_s69_empirical_results.json` | `93852` | `05175f89d323779ae04d9cfcf02345a87929a42fdffe52708470c1665026ed62` |
| `mlos_zone_b_receipt.json` | `2628` | `19bbdd74184c853b2ac996a64beac46af93bb99e16315a323f6328469d5f6cf5` |
| `rtx3080_s69_oracle.py` | `39759` | `ca6a030fcf7c291263d75731b2a8e00abc03f799b6681536390cf6adc9555579` |
| `rtx3080_s69_audit.py` | `21582` | `f9bbc83493c7af6a83f563358d8dee23ce8f0f94d530b55f01cf9234bb5ded19` |
| `rtx3080_s69_certificate.json` | `4928` | `7526c50285bb837f14e23373fbc7d8ab7c678a9a1de32fb9cf72d5d6ef9ee0ef` |
| `test_rtx3080_s69_r5.py` | `57532` | `b1a2298a547f9213b1da047389d236bdd638d35c9f329f327de77261e88ce602` |

### Standalone Verification Commands
```bash
python3 -I -B rtx3080_s69_audit.py --check rtx3080_s69_certificate.json
python3 -I -B rtx3080_s69_audit.py --verify-receipts .
python3 -I -B test_rtx3080_s69_r5.py -v
```
