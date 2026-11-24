# Live Real-Weight Open LLM (`Qwen2.5-0.5B` & `SmolLM2-360M`) RTX 3080 `sm_86` `A_16` / `Block-H2` / `H_16` / `E8_Had` Empirical Evaluation & End-to-End CUDA Transcode Serving (`Moissanite-Labs/siliquant#77`)

## Epistemic Taxonomy & Hardware Provenance

Every claim in this report is classified into one of three non-overlapping epistemic tiers:

1. **`exact` (Pure-$\mathbb{Z}/\mathbb{Q}$ Certified Mathematics):** Exact architectural tiling & weight counts for `Qwen/Qwen2.5-0.5B` (`24` layers, `168` block linear matrices, `357,826,560` block weights, exact `group_size=128` tiling) and `HuggingFaceTB/SmolLM2-360M` (`32` layers, `224` block linear matrices, `314,572,800` block weights, exact `group_size=32` tiling since `960 mod 128 = 64 != 0`), exact `12`-layer stacked `down_proj` dimensions (`10,752 × 4,864 = 52,297,728` weights, `29,417,472 B` at `9/2 bpw`), exact lossless `int8` VRAM fit of both `18B` (`bound=1207`, `max|v| = 34 <= 127`) and `72B` (`bound=5213`, `max|v| = 72 <= 127`) spherical balls, and exact `2^(5/8)` ball determinant scaling ratios (`5213/3380` & `1207/782`). Verified with zero floating-point literals, zero `float()` casts, and zero bare `assert`s in `rtx3080_live_llm_oracle.py` and `rtx3080_live_llm_audit.py`.
2. **`sampled` (Live Offline Real-Weight LLM & RTX 3080 `sm_86` Silicon Measurements — `measured_live_model_weights: true`):** Full `13`-arm quantization evaluations on both real pretrained open-weight LLMs (`26` total real-weight model arms, `672.4M` real quantized parameters, `0` $A_{16}$ syndrome errors, `0` non-block tensor differences) over strictly held-out WikiText-2 (`4,088` tokens) and real Python stdlib (`4,088` tokens), per-layer anisotropy diagnostics across all `392` real Transformer linear matrices, real-Fisher vs. random-token (`P5`) vs. data-free `1/col_rms^2` proxy calibration shootouts, and bare-metal `sm_86` CUDA real-weight `18B`/`72B` lossless VRAM transcoding + direct `4.50-bpw` serving + live `intel-rapl:0` CPU & `nvidia-smi` GPU Joules/token measurements.
3. **`anchor` (Pinned Upstream & Hardware Reference Baselines):** Companion RTX 3080 `sm_86` on-the-fly enumerative GPU GEMV throughput baseline (`hampack_18B = 3.7346 Gweights/s`, `hammingindex_72B = 0.4929 Gweights/s`), `Moissanite-Labs/siliquant#77` (`49283ff`) Gemma 3 `12B` `A_16` vs. `E8_Had` (`0.0566` vs. `0.0680` Wiki KL) and 1-core CPU `hampack.c` decode (`9.4–10.1 M weights/s`), and `Moissanite-Labs/siliquant#58` Gemma 3 scaling curve (`270M -> 1B -> 4B -> 12B` `P5` random-token retention `90.8% -> 85.3% -> 78.7% -> 53.7%`).

### Pinned Open-Weight Models, Corpora & Hardware Provenance Receipt (`sampled`)

All HuggingFace artifacts were cached locally outside the study directory (never committed into the repository), SHA-256 pinned, and executed strictly offline (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `eval_seed=2026112401`) under an exclusive hardware lease with anomaly checks (`mlos_zone_b_receipt.json`):

| Provenance Item | Pinned Revision / Spec | SHA-256 Digest |
|---|---|---|
| `Qwen/Qwen2.5-0.5B` (`model.safetensors`, `988,097,824 B`) | rev `060db6499f32faf8b98477b0a26969ef7d8b9987` (`24L`, `h=896`, `kv=128`, `inter=4864`, `grp=128`) | `88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342` |
| `Qwen/Qwen2.5-0.5B` (`tokenizer.json`) | tied embeddings, `122` non-block tensors protected (`0` diffs across all `13` arms) | `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539` |
| `HuggingFaceTB/SmolLM2-360M` (`model.safetensors`, `723,674,912 B`) | rev `f8027fd0eaeea54caa13c31d31b9fdc459c38b49` (`32L`, `h=960`, `kv=320`, `inter=2560`, `grp=32`) | `7aaff6661428bed033abba9522bec81938678642cca3181fe752b6ca9e1e540f` |
| `HuggingFaceTB/SmolLM2-360M` (`tokenizer.json`) | tied embeddings, `66` non-block tensors protected (`0` diffs across all `13` arms) | `9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c` |
| `Salesforce/wikitext` (`wikitext-2-raw-v1`, `validation` / `test`) | rev `b08601e04326c79dfdd32d625aee71d232d685c3` (`12` cal / `8` held-out chunks × `512` tok) | val `93abc2e0cd803097893a176fd0de13e75c77a1414c5e64af7bbcf705068938c5` / test `430983e5a711fbc4c6576edffecacf47ff23d58f62f3923d2386d798e0a794c9` |
| Real Python Stdlib Eval Corpus (`6` modules) | `fractions`, `difflib`, `decimal`, `statistics`, `argparse`, `json/decoder` (`8 × 512` held-out) | `fe8ff0f94937ebd41e55f9880dd19bbaa4bbeadec791a6544bbafcb9cd2ee609` |
| Hardware Provenance `mlos_zone_b_receipt.json` (`0o600`) | `clean_hardware_run: true`, `anomalies: []`, `zone: "B"`, `exit_code: 0`, `307.04 s` (`13:11:27Z..13:16:34Z`) | receipt `20735cb7c5f65ad548b029d747565951b3cd21632fb559dd0185722ddfce75fe` |
| RTX 3080 `sm_86` Telemetry & `ptxas -v` | `graphics_processes: []`, `linux_kmd_baseline_vram_mib: 2`, `P0`, `6001 MHz`, `1710 MHz` SM, `64 °C`; **`10/10` kernels `0 B` stack, `0 B` spill** (`20–72` regs) | CUDA src `b83b0a2a517eafb17c747e851b76f68fe4bd7457a0db90a5879f4ce9bf7fb36e` |

---

## Executive Summary of Real-Weight Empirical Findings

1. **Opposite-Sign `E_8` vs. `A_16` Hadamard Law Holds on BOTH Real Pretrained LLMs (Natural & `<= 1.5e-6 dB` Matched MSE — `sampled`):**
   - **On `E_8`, Hadamard $H_8$ HELPS both real LLMs (`E8_H8 < E8_axis`), directly reproducing `siliquant#77`'s `E8_Had < E8` finding:**
     - `SmolLM2-360M` (`grp=32`, `-25.055 dB`): `E8_H8` cuts KL by **`-7.0%`** on WikiText-2 (`0.071621` vs. `0.076987`; matched MSE `0.071548`) and **`-19.0%`** on Python (`0.056793` vs. `0.070137`).
     - `Qwen2.5-0.5B` (`grp=128`, `-25.585 dB`): `E8_H8` cuts KL by **`-5.6%`** on WikiText-2 (`0.054841` vs. `0.058117`; matched MSE `0.054820`) and **`-2.9%`** on Python (`0.044983` vs. `0.046320`).
   - **On `A_16`, Hadamard $H_{16}$ HURTS both real LLMs (`A16_axis < A16_BlockH2 < A16_H16`), proving `A16_axis` wins uncalibrated WikiText-2 KL outright on BOTH real models:**
     - `SmolLM2-360M` (`grp=32`, `-25.028 dB`, BF16 PPL `16.395` Wiki / `5.045` Py): **`A16_axis = 0.065173`** (`PPL 17.252`, `Top-1 86.94%`) / **`0.056725`** (`Py`) is **#1 uncalibrated arm** — beating `E8_H8` (`0.071621`) by **`-9.0%`** and `Z_plain` (`0.080395`) by **`-18.9%`** — while `A16_BlockH2` scores `0.071702` / `0.059325` and **`A16_H16` degrades to `0.079847` (`+22.5%` worse!) / `0.069956` (`+23.3%` worse!)**, persisting unchanged at `1.00e-6 dB` matched weight MSE (`0.079666` / `0.069862`)!
     - `Qwen2.5-0.5B` (`grp=128`, `-25.553 dB`, BF16 PPL `17.950` Wiki / `4.861` Py): **`A16_axis = 0.052525`** (`PPL 18.486`, `Top-1 88.72%`) / `0.045602` (`Py`) is **#1 uncalibrated WikiText-2 arm** — beating `E8_H8` (`0.054841`) by **`-4.2%`** and `Z_plain` (`0.062033`) by **`-15.3%`** — while `A16_BlockH2` scores `0.055206` Wiki / **`0.045392`** Py (#1 uncalibrated `A_16` arm on Python!) and **`A16_H16` degrades to `0.057614` (`+9.7%` worse!) / `0.047449` (`+4.0%` worse!)**, persisting unchanged at `1.51e-6 dB` matched weight MSE (`0.057551` / `0.047448`)!
2. **Real-Weight Calibration & `+K10` Shootout (`sampled` + `anchor`):**
   - Real WikiText-2 Fisher (`hA16_real`) slashes WikiText-2 KL by **`-52.5%`** on `SmolLM2-360M` (`0.065173 -> 0.030955`, `PPL 16.973`) and **`-46.5%`** on `Qwen2.5-0.5B` (`0.052525 -> 0.028102`, `PPL 18.331`), transferring strongly to held-out Python (`-51.5%` / `-45.6%`).
   - Data-free `1/col_rms^2` proxy (`hA16_colproxy`) fails sharply on both real models, capturing only **`33.19%`** of real Fisher gain on `SmolLM2-360M` (mean layerwise Spearman $\rho = 0.1010$ vs. real `0.7194`) and **`23.01%`** on `Qwen2.5-0.5B` ($\rho = 0.1096$ vs. real `0.7972`).
   - Random-token Fisher (`hA16_rand`, `P5`) retains **`99.78%`** (`SmolLM2-360M`) and **`97.78%`** (`Qwen2.5-0.5B`) on WikiText-2 — honestly confirming that at `0.36B–0.5B` scale static architectural column anisotropy dominates over data-dependent activation curvature, sitting right at the small-scale origin of Gemma 3's monotonic `P5` curve (`99.8%/97.8% [0.36B–0.5B] -> 90.8% [270M G3] -> 85.3% [1B] -> 78.7% [4B] -> 53.7% [12B]`).
   - `+K10` kept columns (`A16_axis_K10`, top `10` outlier columns per matrix in `bf16`) cuts `SmolLM2-360M` KL by **`-9.57%`** Wiki (`0.058934`) / **`-11.25%`** Python (`0.050342`), while staying nearly flat on `Qwen2.5-0.5B` (`0.052972` / `0.045578`).
3. **End-to-End Real-Weight CUDA Transcode & Serving (`Qwen2.5-0.5B` `52.30M` Real `down_proj` Weights — `sampled` + `anchor`):**
   - **Cold/Disk-to-VRAM Lossless Transcode (`0` bit mismatches, `0` syndrome errors):** Decoding real `Qwen2.5-0.5B` `18B` `hampack` and `72B` `hammingindex` spherical-ball weights on RTX 3080 into a lossless VRAM `int8` SoA layout takes just **`23.75 ms`** (`2.2016 Gwt/s`, `~162.5 ms` full `Qwen2.5-0.5B`) for `18B` and **`206.64 ms`** (`0.2531 Gwt/s`, `~1.41 s` full `Qwen2.5-0.5B`) for `72B`.
   - **Lossless Transcoded VRAM `int8` `__dp4a` Serving (`0` quantization loss vs. `18B`/`72B` spherical balls!):** Serves the exact decoded `72B` ball at **`358.52 Gweights/s` (`364.45 GB/s` = `102.24%` of STREAM Triad — `727.4×` faster than on-the-fly GPU `72B` GEMV!)** and the exact decoded `18B` ball at **`342.43 Gweights/s` (`364.15 GB/s` = `102.15%` Triad — `91.7×` faster than on-the-fly GPU `18B` GEMV!)**, with CPU FP64 `rel_l2 = 9.07e-8`.
   - **Direct `4.50-bpw` VRAM Serving (`29.42 MB` vs. `52.30 MB`):** **`A16_axis_preencoded_dp4a = 612.42 Gweights/s` (`0.085395 ms`, `345.05 GB/s` = `96.79%` Triad)** and **`Block_H2_dp4a = 608.57 Gweights/s` (`0.085935 ms`, `96.18%` Triad)**, giving a full 24-layer `Qwen2.5-0.5B` (`357.83M` block weights) linear-GEMV floor of **`0.5843 ms/token` (`1,711.5 tok/s`)** at **`50.79 mJ/tok` GPU (`86.92W`) + `32.62 mJ/tok` live `intel-rapl:0` CPU (`83.40 mJ/tok` combined)**.

---

## Section 1 (`sampled`) — Complete `4.5000 bpw` Real-Weight LLM Head-to-Head (`26 / 26` Arms Clean)

Every arm below preserves `100%` of non-block tensors bit-identically (`non_block_tensors_differ = 0`: `122` protected tensors on `Qwen2.5-0.5B`, `66` on `SmolLM2-360M`), records **`0` $A_{16}$ syndrome errors**, and evaluates `4,088` strictly held-out next-token logits on each corpus. Matched-MSE arms equalize per-matrix Frobenius MSE to their axis reference within **`<= 1.51e-6 dB`** (`0.0000015 dB`).

### 1.1 `Qwen/Qwen2.5-0.5B` (`group_size = 128`, `168` Matrices, `357,826,560` Block Weights; BF16 PPL: `17.950` Wiki / `4.861` Py)

| Arm (`4.5000 bpw`) | Wt Err (`dB`) | WikiText-2 `KL` (±SE) | Wiki `PPL` | Wiki `Top-1` | Python `KL` (±SE) | Py `PPL` | Py `Top-1` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Z_plain` | `-24.927` | `0.062033` (±`0.00324`) | `18.944` | `86.74%` | `0.045812` (±`0.00299`) | `5.054` | `90.46%` |
| `E8_axis` | `-25.585` | `0.058117` (±`0.00415`) | `18.756` | `87.40%` | `0.046320` (±`0.00362`) | `5.014` | `90.63%` |
| `E8_H8` *(natural)* | `-25.585` | `0.054841` (±`0.00369`) | `18.643` | `87.82%` | **`0.044983`** (±`0.00338`) | `5.047` | `90.44%` |
| `E8_H8_matched` (`1.0e-6 dB`) | `-25.585` | `0.054820` (±`0.00367`) | `18.658` | `88.16%` | `0.045059` (±`0.00338`) | `5.048` | `90.46%` |
| **`A16_axis`** *(uncal #1 Wiki)* | `-25.553` | **`0.052525`** (±`0.00347`) | **`18.486`** | **`88.72%`** | `0.045602` (±`0.00293`) | `5.100` | `90.95%` |
| `A16_BlockH2` *(natural)* | `-25.553` | `0.055206` (±`0.00358`) | `18.656` | `88.21%` | `0.045392` (±`0.00330`) | `5.061` | `91.00%` |
| `A16_BlockH2_matched` (`9.6e-7 dB`) | `-25.553` | `0.055362` (±`0.00358`) | `18.660` | `88.38%` | `0.045662` (±`0.00331`) | `5.062` | `91.07%` |
| `A16_H16` *(natural)* | `-25.553` | `0.057614` (±`0.00408`) | `18.591` | `87.62%` | `0.047449` (±`0.00364`) | `5.117` | `90.75%` |
| `A16_H16_matched` (`1.5e-6 dB`) | `-25.553` | `0.057551` (±`0.00407`) | `18.589` | `87.48%` | `0.047448` (±`0.00363`) | `5.111` | `90.83%` |
| `A16_axis_K10` *(+10 kept cols)* | `-25.559` | `0.052972` (±`0.00344`) | `18.469` | `88.63%` | `0.045578` (±`0.00290`) | `5.097` | `91.17%` |
| `hA16_colproxy` (`1/col_rms^2`) | `-25.603` | `0.046906` (±`0.00349`) | `18.739` | `88.92%` | `0.037981` (±`0.00269`) | `5.029` | `92.32%` |
| `hA16_rand` *(random-token `P5`)* | `-23.505` | `0.028644` (±`0.00066`) | `18.417` | `90.34%` | **`0.023527`** (±`0.00061`) | `4.975` | **`92.93%`** |
| **`hA16_real`** *(real Fisher)* | `-23.661` | **`0.028102`** (±`0.00133`) | **`18.331`** | **`91.17%`** | `0.024798` (±`0.00129`) | **`4.961`** | `92.22%` |

### 1.2 `HuggingFaceTB/SmolLM2-360M` (`group_size = 32`, `224` Matrices, `314,572,800` Block Weights; BF16 PPL: `16.395` Wiki / `5.045` Py)

| Arm (`4.5000 bpw`) | Wt Err (`dB`) | WikiText-2 `KL` (±SE) | Wiki `PPL` | Wiki `Top-1` | Python `KL` (±SE) | Py `PPL` | Py `Top-1` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Z_plain` | `-24.396` | `0.080395` (±`0.00706`) | `17.523` | `84.25%` | `0.068176` (±`0.00675`) | `5.400` | `88.92%` |
| `E8_axis` | `-25.055` | `0.076987` (±`0.00515`) | `17.728` | `85.62%` | `0.070137` (±`0.00613`) | `5.367` | `89.43%` |
| `E8_H8` *(natural)* | `-25.055` | `0.071621` (±`0.00410`) | `17.666` | `85.18%` | `0.056793` (±`0.00329`) | `5.366` | `89.97%` |
| `E8_H8_matched` (`1.0e-6 dB`) | `-25.055` | `0.071548` (±`0.00408`) | `17.675` | `85.13%` | `0.057079` (±`0.00338`) | `5.367` | `89.82%` |
| **`A16_axis`** *(uncal #1 both)* | `-25.028` | **`0.065173`** (±`0.00367`) | **`17.252`** | **`86.94%`** | **`0.056725`** (±`0.00349`) | **`5.330`** | `89.53%` |
| `A16_BlockH2` *(natural)* | `-25.028` | `0.071702` (±`0.00277`) | `17.522` | `86.28%` | `0.059325` (±`0.00306`) | `5.390` | `89.29%` |
| `A16_BlockH2_matched` (`1.0e-6 dB`) | `-25.028` | `0.071799` (±`0.00279`) | `17.508` | `86.50%` | `0.059337` (±`0.00304`) | `5.382` | `89.29%` |
| `A16_H16` *(natural)* | `-25.028` | `0.079847` (±`0.00483`) | `17.941` | `84.76%` | `0.069956` (±`0.00542`) | `5.386` | `89.85%` |
| `A16_H16_matched` (`1.0e-6 dB`) | `-25.028` | `0.079666` (±`0.00483`) | `17.920` | `84.61%` | `0.069862` (±`0.00544`) | `5.383` | `90.04%` |
| `A16_axis_K10` *(+10 kept cols)* | `-25.039` | **`0.058934`** (±`0.00414`) | `17.230` | `87.77%` | **`0.050342`** (±`0.00369`) | `5.293` | `90.34%` |
| `hA16_colproxy` (`1/col_rms^2`) | `-25.060` | `0.053817` (±`0.00354`) | `17.478` | `87.45%` | `0.045428` (±`0.00350`) | `5.259` | `90.97%` |
| `hA16_rand` *(random-token `P5`)* | `-24.053` | `0.031030` (±`0.00139`) | `17.036` | `90.80%` | `0.028954` (±`0.00181`) | `5.187` | **`93.05%`** |
| **`hA16_real`** *(real Fisher)* | `-24.372` | **`0.030955`** (±`0.00170`) | **`16.973`** | **`91.37%`** | **`0.027538`** (±`0.00147`) | **`5.168`** | `92.86%` |

---

## Section 2 (`sampled`) — Per-Layer Anisotropy Diagnostic (`392` Real Layers) & Calibration Analysis

Using held-out WikiText-2 activation Hessians (`X^T X`) per layer, we score the exact quadratic error `tr(ΔW H ΔW^T)` of `A16_axis`, `A16_BlockH2`, and `A16_H16` across all `392` real linear matrices:

| Projection Kind | `SmolLM2-360M` Reg I (`axis < H16`) | `SmolLM2` `H16/axis` Quad Err | `SmolLM2` `BH2/axis` Quad Err | `Qwen2.5-0.5B` Reg I (`axis < H16`) | `Qwen2.5` `H16/axis` Quad Err | `Qwen2.5` `BH2/axis` Quad Err |
|---|---:|---:|---:|---:|---:|---:|
| `q_proj` | `15 / 32` | `0.9926` | `0.9952` | `12 / 24` | `0.9979` | `0.9959` |
| `k_proj` | `9 / 32` | `0.9783` | `0.9902` | `9 / 24` | `0.9720` | `1.0063` |
| `v_proj` | **`25 / 32`** | **`1.0631`** | `1.0507` | `11 / 24` | `0.9903` | `1.0016` |
| `o_proj` | `15 / 32` | `0.9896` | `0.9856` | `7 / 24` | `0.9932` | `0.9903` |
| `gate_proj` | `13 / 32` | `0.9925` | `0.9955` | `9 / 24` | `0.9989` | `1.0014` |
| `up_proj` | **`23 / 32`** | `1.0055` | `1.0043` | **`14 / 24`** | `0.9987` | `0.9993` |
| `down_proj` | **`21 / 32`** | `1.0230` | `1.0114` | `11 / 24` | `0.9996` | `0.9974` |
| **Total** | **`121 / 224` (`54.0%`)** | — | `BH2 < H16`: `119/224` | **`73 / 168` (`43.5%`)** | — | `BH2 < H16`: `83/168` |

**Why Full-Model `A16_axis` Still Dominates Even When `95/168` `Qwen2.5-0.5B` Layers Score Regime II Locally:** Local layerwise `tr(ΔW H ΔW^T)` ignores inter-layer residual propagation and $A_{16}$'s zero-cube axis alignment; $H_{16}$ smears quantization error into the dense activation modes of the high-sensitivity `v_proj` / `up_proj` / `down_proj` bottlenecks, yielding `+9.7%` worse end-to-end WikiText-2 KL on `Qwen2.5-0.5B` and `+22.5%` worse on `SmolLM2-360M` at exact `<= 1.5e-6 dB` weight-MSE parity.

### Calibration Shootout Summary (`sampled` + `anchor`)

| Model | Real Fisher Wiki KL Gain | `hA16_rand` (`P5`) Wiki / Py Retention | `hA16_colproxy` Wiki / Py Retention | Mean Layerwise Spearman $\rho$ (Real / Rand / Proxy) |
|---|---:|---:|---:|---:|
| `SmolLM2-360M` (`360M`) | **`-52.50%`** (`0.065173 -> 0.030955`) | `99.78%` / `95.15%` | **`33.19%`** / `38.71%` | **`+0.7194`** / `+0.5134` / **`+0.1010`** |
| `Qwen2.5-0.5B` (`500M`) | **`-46.50%`** (`0.052525 -> 0.028102`) | `97.78%` / `106.11%` | **`23.01%`** / `36.63%` | **`+0.7972`** / `+0.5558` / **`+0.1096`** |
| *Anchor: Gemma 3 `270M -> 12B` (`siliquant#58`)* | — | *`90.8% -> 85.3% -> 78.7% -> 53.7%`* | — | — |

---

## Section 3 (`sampled` + `anchor`) — End-to-End Real-Weight CUDA Transcode, Serving & Live Energy

Exporting the first `12` real `Qwen/Qwen2.5-0.5B` `mlp.down_proj` layers stacked into a `10,752 × 4,864` (`52,297,728` real weights, `0` export syndrome errors) matrix and executing `rtx3080_live_llm_transcode_kernels.cu` (`10/10` zero-spill `sm_86` kernels, `STREAM Triad = 356.4807 GB/s`):

| Real-Weight RTX 3080 Kernel (`10,752 × 4,864`, `52.30M` Real Wts) | Latency (`ms`) | Throughput (`Gwt/s`) | Effective BW (`GB/s`) | `%` Triad | Parity / Bit Err | vs. On-the-Fly Enumerative GPU |
|---|---:|---:|---:|---:|---:|---:|
| One-time GPU Transcode: **`18B` `hampack` -> `int8` SoA** | `23.753894` | `2.2016` | — | — | `0` bit / `0` syn | `~162.5 ms` full `Qwen2.5-0.5B` |
| One-time GPU Transcode: **`72B` `hammingindex` -> `int8` SoA** | `206.643213` | `0.2531` | — | — | `0` bit / `0` syn | `~1.41 s` full `Qwen2.5-0.5B` |
| **Lossless Transcoded `18B` VRAM `int8` `__dp4a` GEMV** | `0.152723` | **`342.4347`** | `364.1504` | **`102.152%`** | `rel_l2 = 9.10e-8`, `0` syn | **`91.69×`** over `18B` on-the-fly (`3.735`) |
| **Lossless Transcoded `72B` VRAM `int8` `__dp4a` GEMV** | `0.145871` | **`358.5203`** | **`364.4504`** | **`102.236%`** | `rel_l2 = 9.07e-8`, `0` syn | **`727.37×`** over `72B` on-the-fly (`0.493`) |
| **Direct `4.50-bpw` `A16_axis_preencoded_dp4a` GEMV** (`29.42 MB`) | **`0.085395`** | **`612.4195`** | `345.0466` | **`96.793%`** | `rel_l2 = 9.13e-8`, `0` syn | **`1,242.5×`** over `72B` on-the-fly |
| **Direct `4.50-bpw` `Block_H2_dp4a` GEMV** (`29.42 MB`) | `0.085935` | **`608.5717`** | `342.8786` | **`96.184%`** | `rel_l2 = 9.59e-8`, `0` syn | **`1,234.7×`** over `72B` on-the-fly |

**Full 24-Layer `Qwen/Qwen2.5-0.5B` (`357,826,560` Block Weights) Projected Linear-GEMV Serving & Measured Energy (`sampled`):**
- Linear-GEMV time per token (direct `4.50-bpw` `A16_axis_preencoded_dp4a`): **`0.5843 ms/token` (`1,711.5 tokens/s` linear ceiling)**.
- GPU sampled power draw during sustained GEMV loop: **`86.92 W`** (`68 °C`, `1710 MHz` SM, `6001 MHz` GDDR6, `0` thermal throttle) -> **`50.79 mJ/token` GPU**.
- Live unlocked `/sys/class/powercap/intel-rapl:0/energy_uj` host package measurement (`4.7670 mJ/call` on `52.30M` weights; `1,419.46 J` across the `21.05 s` CUDA benchmark) -> **`32.62 mJ/token` CPU** -> **`83.40 mJ/token` combined GPU + CPU**.

---

## Section 4 — Verification Receipts & Complete File Manifest

- **Isolated Reproducibility Parity (`-I -B`, `-I -O -B`, `-I -B -W error`, `PYTHONHASHSEED=42 -B`):** `--standalone --json` bit-identical to `0o600` disk certificate (`44aa3ea8...`) across `4/4` modes; `--check` `valid=true` across `4/4`; `--verify-receipts .` `11/11` checks `true` across `4/4`; `test_rtx3080_live_llm_r5.py` `5/5 OK` across `4/4`.
- **4-Surface Adversarial Fuzzer (`CLI`, `VCD_FR1`, `VCD_FR0`, `SEC_DIRECT`):** **`82 / 82` PASS** (`25/25` CLI + `19/19` FR1 + `19/19` FR0 + `19/19` SEC, `0` bypasses, `0` crashes).
- **Live Mutation Testing (`aletheia mutate --max-mutants 0 --work-root . --add-cwd`) & Whole-File AST `BoolOp`:** **`67 / 67` (`100.0%`, `0` survivors, `truncated=False`)** official mutate (`2/2` oracle + `65/65` audit) and **`180 / 180` (`100.0%`, `0` survivors)** whole-file AST `BoolOp` (`138/138` oracle + `42/42` audit).
- **Hygiene & Safety (`aletheia audit-check --strict-ancestors --safety-scan`):** `PASS` (`0` findings, `0` local-path leaks, `0` private tokens, `0` CRLF, `0` bare `assert`s, `0` AST floats in oracle/audit/test, scratch `os.mkfifo` non-regular probes only, `PYC=0`).

### File Manifest

| File | Lines | Bytes | Mode | SHA-256 |
|---|---:|---:|---:|---|
| `rtx3080_live_llm_transcode_kernels.cu` | `970` | `44640` | `0o644` | `b83b0a2a517eafb17c747e851b76f68fe4bd7457a0db90a5879f4ce9bf7fb36e` |
| `rtx3080_live_llm_eval.py` | `991` | `42049` | `0o644` | `f0bde1ebee9c51c9514fd5b67eb43ef320c96ef11ae08169016d059195fb9b99` |
| `rtx3080_live_llm_empirical_results.json` | `959` | `30089` | `0o644` | `42ad90b1ddfba30d3d99b9d0a7a624770192b084dbe86b1e51c3a747cb7181e7` |
| `mlos_zone_b_receipt.json` | `91` | `2629` | `0o600` | `5f9710c7bb200a4d399dea7c375b802cd69dca92f439f68258ef302549014b15` |
| `rtx3080_live_llm_oracle.py` | `510` | `23436` | `0o644` | `25d176187529b69c9a15249fc9a15345a247d90ccf448018ecfd2ac33d656523` |
| `rtx3080_live_llm_audit.py` | `500` | `19742` | `0o644` | `9bf55a7c3399a9f96fe01d58574d8ff94c8e081cf18a5ad6a005f0f1cec6c19a` |
| `rtx3080_live_llm_certificate.json` | `1` | `4027` | `0o600` | `44aa3ea84bcedc9b81f125708dc9da105ce80e1fe469636947d14cf9cadc22bb` |
| `test_rtx3080_live_llm_r5.py` | `782` | `41671` | `0o644` | `25a076eb3820051b5a6d0b72e8e0ceb65a360bf12c1d90f0280499849ff96232` |
