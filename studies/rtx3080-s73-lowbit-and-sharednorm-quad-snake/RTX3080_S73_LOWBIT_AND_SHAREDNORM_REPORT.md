# RTX 3080 Live-Weight Study: PTX-Ready Shared-RMSNorm `quad_snake`, Exact Load-Power $p$-Curve, and the `3.0 bpw` Two-Level Stratified `BlockH4` Frontier

## 1. Executive Summary & Epistemic Ledger

This study executes **96 live quantized full-model arms** across **4 cached real-weight LLMs** (**120 transformer blocks**, **840 linear matrices**, **4,757,127,168 block weights**, **0 syndrome errors**, **0 out-of-ball groups**, **0 non-block tensor mutations**) at BOTH exact **`4.500 bpw`** and exact **`3.000 bpw`** (`siliquant#85` theta-series ball bounds) on an NVIDIA GeForce RTX 3080 Laptop GPU (`GA104M`, `sm_86`, 8 GiB GDDR6, AC power online, zero competing GPU processes; the ML-OS receipt also logs a CPU RAPL energy total, which is an uncontrolled laptop reading and is not reported here as an energy result). All hypotheses were cryptographically pre-registered prior to GPU execution (`s73_preregistration.json`, `created_utc=2026-11-24T22:16:22Z`, inner SHA-256 `078207526e3a62e301865f79f0ef2024e0d1b5b2caf2feb9cc5e49734a46265b`) strictly **`+20 s`** before the live hardware execution window (`started_utc=2026-11-24T22:16:42Z`, `ended_utc=2026-11-25T00:14:59Z`, `duration_ms=7097312.632` (ms-precision; the UTC stamps in this sentence are second-precision, consistent with the 7097 s window), `clean_hardware_run=true`, `anomalies=[]`; these are the receipt's own self-reported fields: they record zero competing GPU processes, not controlled conditions, and the laptop host was not under controlled power or thermal conditions).

### Headline Results: Hardware-Executable `4.5 bpw` Records & `3.0 bpw` Distortion-Ratio Amplification

| Model (`g`) | `4.5 bpw` Prior `BlockH2` (Wiki / Py) | `4.5 bpw` New PTX/Zero-Cost Winner (Wiki / Py) | `4.5 bpw` $\Delta$ vs `BlockH2` | `3.0 bpw` `BlockH2` / `sort_real` (Wiki) | `3.0 bpw` Best Stratified `BlockH4` (Wiki) | `3.0 bpw` $\Delta$ vs `BlockH2` / `sort` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `SmolLM2-360M` (`32`) | `0.035651` / `0.032959` | `quad_snake_down_zerocost` (`0.027934` / `0.025417`) | `-21.65%` / `-22.88%` | `0.245958` / `0.400306` | `quad_snake_real` (`0.263070`) | `+6.96%` / `-34.28%` |
| `Qwen2.5-0.5B` (`128`) | `0.026970` / `0.024050` | `quad_lpt_sqrt_sharednorm` (`0.025713` / `0.022710`) | `-4.66%` / `-5.57%` | `0.213362` / `0.307427` | `quad_snake_real` (`0.217828`) | `+2.09%` / `-29.14%` |
| `Qwen2.5-1.5B` (`128`) | `0.023281` / `0.014356` | `quad_snake_down_zerocost` (`0.022783` / `0.015399`) | `-2.14%` / `+7.26%` | `0.186550` / `0.209113` | `quad_lpt_sqrt_real` (`0.177318`) | `-4.95%` / `-15.20%` |
| `Qwen2.5-3B` (`128`) | `0.024179` / `0.022975` | `quad_snake_sharednorm` (`0.024020` / `0.021344`) | `-0.66%` / `-7.10%` | `0.190650` / `0.197239` | `quad_snake_real` (`0.168874`) | `-11.42%` / `-14.38%` |

Note: "PTX/Zero-Cost Winner" refers to the offline `down_proj` fold (zero runtime gathers at inference); it is not a cost claim about the parity/bit-flip check.

### Pre-Registered Prediction Outcomes (Honest Confirmations & Boundaries)

| Pillar | Pre-Registered Hypothesis | Live Hardware Outcome | Verdict |
| :--- | :--- | :--- | :--- |
| **`P_S73_1`** | Hardware-executable `quad_snake_sharednorm` & `quad_lpt_sqrt_sharednorm` vs per-matrix `BlockH2` (`4.5 bpw`) | On **`Qwen2.5-3B`**, PTX-ready `quad_snake_sharednorm` beats previously undefeated per-matrix `BlockH2` on **BOTH** corpora (`0.024020` `-0.66%` Wiki, `0.021344` `-7.10%` Py), and `quad_lpt_sqrt_sharednorm` cures linear-LPT starvation (`0.025409` vs `0.026776`, `-5.11%`). On **`0.5B`**, `quad_lpt_sqrt_real` sets a new all-time record (`0.025153` `-6.74%`, Py `0.022709` `-5.58%`) and PTX-ready `quad_lpt_sqrt_sharednorm` hits `0.025713` (`-4.66%`), while zero-cost `quad_snake_down_zerocost` beats `BlockH2` on `0.5B` (`0.026749`, `-0.82%`), sets the all-time **`1.5B` Wiki record** (`0.022783`, `-2.14%`), and sets the all-time **`360M` record** on both corpora (`0.027934` `-0.97%` vs `BlockH4`, Py `0.025417` `-4.45%`). (Honest boundary: `quad_snake_sharednorm` alone trails `BlockH2` on `0.5B`/`1.5B` where `quad_lpt_sqrt_sharednorm` and `quad_snake_down_zerocost` win.) | **CONFIRMED ON 3B (BOTH CORPORA); PTX/ZERO-COST WINNERS ON 4/4 MODELS** |
| **`P_S73_2`** | Exact 7-point load-power $p$-curve on `Qwen2.5-3B` (`4.5 bpw`) + 4-model starvation telemetry | **`p = 0.375`** shatters the all-time `3B` Python record at **`0.018248`** (`-20.58%` vs `BlockH2`, `-6.51%` vs `p = 0.50`) while beating unpermuted `BlockH4` on Wiki (`0.024678` < `0.024774`). Honest Pareto separation: no single $p > 0$ beats `p = 0.0` (`quad_snake`) on Wiki (`0.023154`, `-4.24%`). Starvation telemetry confirms `0.00%` starved at `p = 0.0` across `4/4` models rising monotonically to `100.00%` at `p = 1.00` on `3B` (Ball-0 2nd-quad rank jumps `1.94G` -> `28.56G`), and exact L2-energy `ball_cv_sqrt` hits its global minimum at `p = 0.50` on all `3/3` Qwen2.5 models! | **CONFIRMED (`p=0.375` PY RECORD; `p=0.0` WIKI CROWN; `p=0.50` MIN `cv_sqrt`)** |
| **`P_S73_3`** | `3.0 bpw` low-bitrate frontier (`siliquant#85` codebooks) & distortion-ratio amplification | `quad_snake_real` beats global `sort_real` on **`4/4`** models (`360M` `-34.28%`, `0.5B` `-29.14%`, `1.5B` `-11.78%`, `3B` `-14.38%`). On **`Qwen2.5-3B`**, `quad_snake_real` sets the all-time `3.0 bpw` record (**`0.168874`**, `-11.42%` vs `BlockH2` — **`2.70x` larger relative margin** than `-4.24%` at `4.5 bpw`!), and PTX-ready `quad_snake_sharednorm` hits **`0.169901`** (`-10.88%` vs `BlockH2`, **`16.58x` larger relative margin** than `-0.66%` at `4.5 bpw`, `PPL=12.081` vs its own BF16 `10.970`). On **`1.5B`**, `quad_lpt_sqrt_real` hits **`0.177318`** (`-4.95%` vs `BlockH2`, `-15.20%` vs `sort`) and `quad_snake_sharednorm` hits `0.183631` (`-1.56%`). Zero-text `quad_snake_rand` crushes `A16_axis` & `E8_H8` on **`4/4`** models (`3B`: `-58.25%` vs `A16_axis`, `-85.92%` vs `E8_H8`), while uncalibrated `3B` `E8_H8` collapses (`2.97x` worse than `A16_axis`). (Honest scale boundary: on small `360M`/`0.5B` at `3.0 bpw`, unpermuted `hA16_BlockH4_real`/`hA16_real` slightly edge `quad_snake`.) | **CONFIRMED AT `>= 1.5B` (`2.70x`–`16.58x` 3B MARGIN AMPLIFICATION; HONEST `<= 0.5B` CROSSOVER)** |

## 2. Exact `Qwen2.5-3B` Load-Power $p$-Curve (`4.5 bpw`) & Ball-0 Starvation Telemetry

| Regime (`p`) | `3B` Quad Spread | `3B` Ball CV (lin) | `3B` Ball CV (sqrt) | `3B` Starved Mats | `3B` Ball-0 2nd Rank / $G$ | `3B` Weight Err (dB) | `3B` WikiKL | `3B` Wiki PPL | `3B` PyKL | `3B` Py $\Delta$ vs `BlockH2` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `p = 0.000` (`quad_snake`) | `0.3379` | `0.8823` | `0.1173` | `0.00%` | `1.94` | `-20.7013` | `0.023154` | `11.0636` | `0.020494` | `-10.80%` |
| `p = 0.125` (`quad_lpt`) | `0.3379` | `0.8725` | `0.1094` | `6.75%` | `2.00` | `-20.7773` | `0.025841` | `11.2438` | `0.022901` | `-0.32%` |
| `p = 0.250` (`quad_lpt`) | `0.3379` | `0.8662` | `0.1032` | `80.56%` | `3.00` | `-20.8117` | `0.026219` | `11.2146` | `0.020832` | `-9.33%` |
| `p = 0.375` (`quad_lpt`) | `0.3379` | `0.8538` | `0.0912` | `92.06%` | `5.28` | `-20.9622` | `0.024678` | `11.1318` | `0.018248` | `-20.58%` |
| `p = 0.500` (`quad_lpt_sqrt`) | `0.3379` | `0.8365` | `0.0746` | `98.41%` | `9.23` | `-21.0174` | `0.025296` | `11.1876` | `0.019520` | `-15.04%` |
| `p = 0.750` (`quad_lpt`) | `0.3379` | `0.7993` | `0.0796` | `100.00%` | `25.25` | `-21.2936` | `0.028536` | `11.1940` | `0.019772` | `-13.94%` |
| `p = 1.000` (`quad_lpt` lin) | `0.3379` | `0.7830` | `0.1077` | `100.00%` | `28.56` | `-21.8744` | `0.026776` | `11.2461` | `0.021951` | `-4.46%` |
| `global_sort` (cannibalized) | `0.3379` | `2.0962` | `0.6106` | `0.00%` | `0.06` | `-23.5220` | `0.026025` | `11.1903` | `0.021765` | `-5.27%` |
| `unpermuted` (`BlockH4`) | `2.2745` | `0.6570` | `0.1440` | `0.00%` | `0.00` | `-18.5273` | `0.024774` | `11.2062` | `0.022519` | `-1.99%` |

## 3. Cross-Bitrate (`3.0 bpw` / `4.5 bpw`) Distortion-Amplification Law Across All `4` Models

| Model (`g`) | `BlockH2` `3.0/4.5` | `BlockH4` `3.0/4.5` | `quad_snake` `3.0/4.5` | `quad_lpt_sqrt` `3.0/4.5` | `quad_snake_sharednorm` `3.0/4.5` | `3.0 bpw` Best Calibrated Arm |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `SmolLM2-360M` (`32`) | `6.90x` | `8.47x` | `8.54x` | `7.81x` | `8.64x` | `b30_hA16_BlockH4_real` (`0.238931`) |
| `Qwen2.5-0.5B` (`128`) | `7.91x` | `7.26x` | `7.24x` | `8.97x` | `7.46x` | `b30_hA16_real` (`0.210827`) |
| `Qwen2.5-1.5B` (`128`) | `8.01x` | `7.74x` | `7.96x` | `7.51x` | `7.74x` | `b30_hA16_BlockH4_quad_lpt_sqrt_real` (`0.177318`) |
| `Qwen2.5-3B` (`128`) | `7.89x` | `7.03x` | `7.29x` | `6.77x` | `7.07x` | `b30_hA16_BlockH4_quad_snake_real` (`0.168874`) |

## 4. Complete 4-Model Live Hardware Results (`96` Arms, `4,757,127,168` Weights)

### 4.1 `HuggingFaceTB/SmolLM2-360M` (`g=32`, `32` Blocks, `224` Matrices, `314,572,800` Weights, BF16 Wiki PPL=`16.3948`, Py PPL=`5.0450`)

| Arm | `bpw` | Bound | Weight Err (dB) | $E_{0.05}$ (dB) | `z0%` | WikiKL | Wiki PPL | PyKL | Py PPL | Syn / OOB / NB |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `b30_A16_BlockH2` | `3.0` | `151` | `-15.9042` | `13.532` | `21.11%` | `0.784254` | `35.3779` | `0.665819` | `10.0792` | `0` / `0` / `0` |
| `b30_A16_BlockH4` | `3.0` | `151` | `-15.9042` | `14.352` | `55.55%` | `0.809444` | `37.1254` | `0.688529` | `10.3302` | `0` / `0` / `0` |
| `b30_A16_BlockH4_quad_snake_wcol` | `3.0` | `151` | `-15.9046` | `14.362` | `55.68%` | `0.856015` | `38.4557` | `0.713601` | `10.4750` | `0` / `0` / `0` |
| `b30_A16_axis` | `3.0` | `151` | `-15.9051` | `8.971` | `96.63%` | `0.542068` | `28.0423` | `0.459771` | `7.7400` | `0` / `0` / `0` |
| `b30_E8_H8` | `3.0` | `96` | `-15.8876` | `9.986` | `0.05%` | `0.541053` | `27.9399` | `0.474658` | `8.3254` | `0` / `0` / `0` |
| `b30_ProjType_Uniform` | `3.0` | `151` | `-15.9043` | `13.497` | `65.50%` | `0.688508` | `33.0518` | `0.585898` | `9.0276` | `0` / `0` / `0` |
| `b30_ProjType_Uniform_QuadSnake_wcol` | `3.0` | `151` | `-15.9047` | `13.509` | `65.62%` | `0.615051` | `29.9324` | `0.517473` | `8.5159` | `0` / `0` / `0` |
| `b30_hA16_BlockH2_real` | `3.0` | `151` | `-15.0502` | `16.248` | `21.79%` | `0.245958` | `20.8872` | `0.224425` | `6.3674` | `0` / `0` / `0` |
| `b30_hA16_BlockH4_quad_lpt_sqrt_real` | `3.0` | `151` | `-15.0713` | `16.497` | `55.70%` | `0.263523` | `21.9202` | `0.243535` | `6.3727` | `0` / `0` / `0` |
| `b30_hA16_BlockH4_quad_snake_rand` | `3.0` | `151` | `-14.5823` | `17.285` | `55.77%` | `0.304907` | `22.3203` | `0.266798` | `6.6103` | `0` / `0` / `0` |
| `b30_hA16_BlockH4_quad_snake_real` | `3.0` | `151` | `-14.9435` | `17.041` | `55.75%` | `0.263070` | `21.4634` | `0.245638` | `6.4748` | `0` / `0` / `0` |
| `b30_hA16_BlockH4_quad_snake_sharednorm` | `3.0` | `151` | `-14.9882` | `17.003` | `55.72%` | `0.264788` | `21.2844` | `0.251711` | `6.4660` | `0` / `0` / `0` |
| `b30_hA16_BlockH4_real` | `3.0` | `151` | `-14.9495` | `17.142` | `55.65%` | `0.238931` | `20.9897` | `0.211970` | `6.3456` | `0` / `0` / `0` |
| `b30_hA16_BlockH4_sort_real` | `3.0` | `151` | `-15.6406` | `15.402` | `55.46%` | `0.400306` | `24.3638` | `0.323040` | `6.9680` | `0` / `0` / `0` |
| `b30_hA16_real` | `3.0` | `151` | `-15.3040` | `12.030` | `96.53%` | `0.241529` | `20.9308` | `0.216980` | `6.2216` | `0` / `0` / `0` |
| `b45_hA16_BlockH2_real` | `4.5` | `1207` | `-24.0633` | `7.816` | `6.78%` | `0.035651` | `17.0286` | `0.032959` | `5.1908` | `0` / `0` / `0` |
| `b45_hA16_BlockH4_quad_lpt_sqrt_real` | `4.5` | `1207` | `-24.1138` | `7.601` | `48.81%` | `0.033746` | `16.8291` | `0.029776` | `5.1788` | `0` / `0` / `0` |
| `b45_hA16_BlockH4_quad_lpt_sqrt_sharednorm` | `4.5` | `1207` | `-24.1634` | `7.554` | `48.70%` | `0.034404` | `16.9025` | `0.029522` | `5.1814` | `0` / `0` / `0` |
| `b45_hA16_BlockH4_quad_snake_down_zerocost` | `4.5` | `1207` | `-23.9794` | `8.412` | `48.67%` | `0.027934` | `16.9263` | `0.025417` | `5.1243` | `0` / `0` / `0` |
| `b45_hA16_BlockH4_quad_snake_real` | `4.5` | `1207` | `-23.7131` | `9.415` | `48.82%` | `0.030821` | `16.9989` | `0.028255` | `5.1477` | `0` / `0` / `0` |
| `b45_hA16_BlockH4_quad_snake_sharednorm` | `4.5` | `1207` | `-23.7593` | `9.382` | `48.70%` | `0.030651` | `17.0099` | `0.028139` | `5.1487` | `0` / `0` / `0` |
| `b45_hA16_BlockH4_real` | `4.5` | `1207` | `-23.9798` | `8.411` | `48.66%` | `0.028207` | `16.9524` | `0.026602` | `5.1555` | `0` / `0` / `0` |

### 4.2 `Qwen/Qwen2.5-0.5B` (`g=128`, `24` Blocks, `168` Matrices, `357,826,560` Weights, BF16 Wiki PPL=`17.9505`, Py PPL=`4.8606`)

| Arm | `bpw` | Bound | Weight Err (dB) | $E_{0.05}$ (dB) | `z0%` | WikiKL | Wiki PPL | PyKL | Py PPL | Syn / OOB / NB |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `b30_A16_BlockH2` | `3.0` | `651` | `-16.4328` | `12.916` | `21.51%` | `0.412372` | `25.5731` | `0.367466` | `6.9166` | `0` / `0` / `0` |
| `b30_A16_BlockH4` | `3.0` | `651` | `-16.4332` | `13.760` | `55.93%` | `0.409209` | `25.3015` | `0.342060` | `6.8374` | `0` / `0` / `0` |
| `b30_A16_BlockH4_quad_snake_wcol` | `3.0` | `651` | `-16.4332` | `13.765` | `56.03%` | `0.416686` | `25.9628` | `0.348639` | `6.8021` | `0` / `0` / `0` |
| `b30_A16_axis` | `3.0` | `651` | `-16.4343` | `8.551` | `96.51%` | `0.439020` | `26.3676` | `0.350304` | `6.9010` | `0` / `0` / `0` |
| `b30_E8_H8` | `3.0` | `420` | `-16.4521` | `9.330` | `0.40%` | `0.406564` | `26.6629` | `0.364221` | `6.9151` | `0` / `0` / `0` |
| `b30_ProjType_Uniform` | `3.0` | `651` | `-16.4334` | `12.940` | `67.68%` | `0.441648` | `26.0948` | `0.365392` | `6.9649` | `0` / `0` / `0` |
| `b30_ProjType_Uniform_QuadSnake_wcol` | `3.0` | `651` | `-16.4334` | `12.945` | `67.79%` | `0.406454` | `25.5599` | `0.362971` | `7.0090` | `0` / `0` / `0` |
| `b30_hA16_BlockH2_real` | `3.0` | `651` | `-13.1190` | `21.321` | `22.29%` | `0.213362` | `21.7120` | `0.192960` | `5.8128` | `0` / `0` / `0` |
| `b30_hA16_BlockH4_quad_lpt_sqrt_real` | `3.0` | `651` | `-12.2697` | `22.793` | `55.87%` | `0.225635` | `22.1232` | `0.207437` | `5.9538` | `0` / `0` / `0` |
| `b30_hA16_BlockH4_quad_snake_rand` | `3.0` | `651` | `-11.9009` | `23.232` | `55.81%` | `0.254159` | `22.7759` | `0.203304` | `5.9177` | `0` / `0` / `0` |
| `b30_hA16_BlockH4_quad_snake_real` | `3.0` | `651` | `-12.1466` | `23.072` | `55.87%` | `0.217828` | `21.3144` | `0.205954` | `5.9059` | `0` / `0` / `0` |
| `b30_hA16_BlockH4_quad_snake_sharednorm` | `3.0` | `651` | `-12.1581` | `23.069` | `55.83%` | `0.223689` | `21.8276` | `0.197736` | `5.8691` | `0` / `0` / `0` |
| `b30_hA16_BlockH4_real` | `3.0` | `651` | `-12.5335` | `22.427` | `55.80%` | `0.213462` | `21.2719` | `0.181139` | `5.8798` | `0` / `0` / `0` |
| `b30_hA16_BlockH4_sort_real` | `3.0` | `651` | `-13.4537` | `21.212` | `55.84%` | `0.307427` | `23.1524` | `0.270246` | `6.2938` | `0` / `0` / `0` |
| `b30_hA16_real` | `3.0` | `651` | `-14.5775` | `16.897` | `96.36%` | `0.210827` | `21.8365` | `0.189808` | `5.8173` | `0` / `0` / `0` |
| `b45_hA16_BlockH2_real` | `4.5` | `5213` | `-22.2325` | `12.177` | `7.02%` | `0.026970` | `18.2349` | `0.024050` | `4.9594` | `0` / `0` / `0` |
| `b45_hA16_BlockH4_quad_lpt_sqrt_real` | `4.5` | `5213` | `-21.3777` | `13.694` | `48.21%` | `0.025153` | `18.4124` | `0.022709` | `4.9896` | `0` / `0` / `0` |
| `b45_hA16_BlockH4_quad_lpt_sqrt_sharednorm` | `4.5` | `5213` | `-21.3986` | `13.685` | `48.14%` | `0.025713` | `18.4326` | `0.022710` | `4.9816` | `0` / `0` / `0` |
| `b45_hA16_BlockH4_quad_snake_down_zerocost` | `4.5` | `5213` | `-21.6333` | `13.346` | `48.12%` | `0.026749` | `18.3058` | `0.023617` | `4.9573` | `0` / `0` / `0` |
| `b45_hA16_BlockH4_quad_snake_real` | `4.5` | `5213` | `-21.2364` | `13.938` | `48.19%` | `0.030080` | `18.3736` | `0.026650` | `4.9707` | `0` / `0` / `0` |
| `b45_hA16_BlockH4_quad_snake_sharednorm` | `4.5` | `5213` | `-21.2571` | `13.929` | `48.11%` | `0.029972` | `18.4456` | `0.026955` | `4.9822` | `0` / `0` / `0` |
| `b45_hA16_BlockH4_real` | `4.5` | `5213` | `-21.6328` | `13.346` | `48.13%` | `0.029413` | `18.3511` | `0.023452` | `4.9594` | `0` / `0` / `0` |

### 4.3 `Qwen/Qwen2.5-1.5B` (`g=128`, `28` Blocks, `196` Matrices, `1,310,195,712` Weights, BF16 Wiki PPL=`12.6180`, Py PPL=`3.7395`)

| Arm | `bpw` | Bound | Weight Err (dB) | $E_{0.05}$ (dB) | `z0%` | WikiKL | Wiki PPL | PyKL | Py PPL | Syn / OOB / NB |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `b30_A16_BlockH2` | `3.0` | `651` | `-16.4331` | `13.160` | `20.82%` | `0.313200` | `16.3603` | `0.209440` | `4.5624` | `0` / `0` / `0` |
| `b30_A16_BlockH4` | `3.0` | `651` | `-16.4329` | `14.014` | `56.18%` | `0.320114` | `16.8356` | `0.216432` | `4.6412` | `0` / `0` / `0` |
| `b30_A16_BlockH4_quad_snake_wcol` | `3.0` | `651` | `-16.4338` | `14.022` | `56.29%` | `0.336141` | `16.3258` | `0.229692` | `4.7239` | `0` / `0` / `0` |
| `b30_A16_axis` | `3.0` | `651` | `-16.4339` | `8.747` | `96.66%` | `0.308467` | `16.7684` | `0.216203` | `4.6911` | `0` / `0` / `0` |
| `b30_E8_H8` | `3.0` | `420` | `-16.4518` | `9.538` | `0.42%` | `0.313192` | `16.5319` | `0.206384` | `4.5880` | `0` / `0` / `0` |
| `b30_ProjType_Uniform` | `3.0` | `651` | `-16.4330` | `13.085` | `67.89%` | `0.302062` | `16.4799` | `0.209854` | `4.5832` | `0` / `0` / `0` |
| `b30_ProjType_Uniform_QuadSnake_wcol` | `3.0` | `651` | `-16.4337` | `13.095` | `68.00%` | `0.296629` | `16.1158` | `0.214510` | `4.6505` | `0` / `0` / `0` |
| `b30_hA16_BlockH2_real` | `3.0` | `651` | `-11.6666` | `25.468` | `21.46%` | `0.186550` | `14.4697` | `0.121586` | `4.2373` | `0` / `0` / `0` |
| `b30_hA16_BlockH4_quad_lpt_sqrt_real` | `3.0` | `651` | `-12.1935` | `24.135` | `56.25%` | `0.177318` | `14.5086` | `0.122736` | `4.2595` | `0` / `0` / `0` |
| `b30_hA16_BlockH4_quad_snake_rand` | `3.0` | `651` | `-11.3482` | `25.521` | `56.24%` | `0.207658` | `15.1836` | `0.134566` | `4.2602` | `0` / `0` / `0` |
| `b30_hA16_BlockH4_quad_snake_real` | `3.0` | `651` | `-11.5373` | `25.351` | `56.25%` | `0.184477` | `14.5961` | `0.129245` | `4.2415` | `0` / `0` / `0` |
| `b30_hA16_BlockH4_quad_snake_sharednorm` | `3.0` | `651` | `-11.5442` | `25.348` | `56.22%` | `0.183631` | `14.5200` | `0.126291` | `4.2431` | `0` / `0` / `0` |
| `b30_hA16_BlockH4_real` | `3.0` | `651` | `-10.7610` | `26.858` | `56.15%` | `0.181148` | `14.4432` | `0.122725` | `4.2580` | `0` / `0` / `0` |
| `b30_hA16_BlockH4_sort_real` | `3.0` | `651` | `-13.4776` | `22.324` | `56.24%` | `0.209113` | `14.7633` | `0.143644` | `4.3202` | `0` / `0` / `0` |
| `b30_hA16_real` | `3.0` | `651` | `-13.7937` | `21.350` | `96.57%` | `0.182789` | `14.4193` | `0.133834` | `4.2518` | `0` / `0` / `0` |
| `b45_hA16_BlockH2_real` | `4.5` | `5213` | `-20.7233` | `16.446` | `6.90%` | `0.023281` | `12.9273` | `0.014356` | `3.8326` | `0` / `0` / `0` |
| `b45_hA16_BlockH4_quad_lpt_sqrt_real` | `4.5` | `5213` | `-21.2276` | `15.189` | `49.24%` | `0.023609` | `12.7669` | `0.014086` | `3.7758` | `0` / `0` / `0` |
| `b45_hA16_BlockH4_quad_lpt_sqrt_sharednorm` | `4.5` | `5213` | `-21.2353` | `15.186` | `49.16%` | `0.024098` | `12.8128` | `0.014576` | `3.7832` | `0` / `0` / `0` |
| `b45_hA16_BlockH4_quad_snake_down_zerocost` | `4.5` | `5213` | `-19.8764` | `17.722` | `49.08%` | `0.022783` | `12.8042` | `0.015399` | `3.8028` | `0` / `0` / `0` |
| `b45_hA16_BlockH4_quad_snake_real` | `4.5` | `5213` | `-20.6250` | `16.292` | `49.24%` | `0.023177` | `12.8822` | `0.015627` | `3.7952` | `0` / `0` / `0` |
| `b45_hA16_BlockH4_quad_snake_sharednorm` | `4.5` | `5213` | `-20.6319` | `16.289` | `49.16%` | `0.023734` | `12.8678` | `0.016980` | `3.8042` | `0` / `0` / `0` |
| `b45_hA16_BlockH4_real` | `4.5` | `5213` | `-19.8754` | `17.722` | `49.07%` | `0.023412` | `12.7947` | `0.015129` | `3.7961` | `0` / `0` / `0` |

### 4.4 `Qwen/Qwen2.5-3B` (`g=128`, `36` Blocks, `252` Matrices, `2,774,532,096` Weights, BF16 Wiki PPL=`10.9702`, Py PPL=`3.1119`)

| Arm | `bpw` | Bound | Weight Err (dB) | $E_{0.05}$ (dB) | `z0%` | WikiKL | Wiki PPL | PyKL | Py PPL | Syn / OOB / NB |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `b30_A16_BlockH2` | `3.0` | `651` | `-16.4368` | `12.492` | `22.96%` | `0.741476` | `20.4281` | `0.423012` | `4.8248` | `0` / `0` / `0` |
| `b30_A16_BlockH4` | `3.0` | `651` | `-16.4350` | `13.355` | `47.63%` | `0.739579` | `20.1676` | `0.598560` | `5.8344` | `0` / `0` / `0` |
| `b30_A16_BlockH4_quad_snake_wcol` | `3.0` | `651` | `-16.4496` | `13.220` | `48.61%` | `0.885835` | `23.2265` | `0.465723` | `5.0244` | `0` / `0` / `0` |
| `b30_A16_axis` | `3.0` | `651` | `-16.4373` | `8.185` | `80.23%` | `0.498083` | `15.9512` | `0.775086` | `6.7969` | `0` / `0` / `0` |
| `b30_E8_H8` | `3.0` | `420` | `-16.4545` | `8.936` | `3.35%` | `1.477182` | `40.7555` | `1.120562` | `9.8709` | `0` / `0` / `0` |
| `b30_ProjType_Uniform` | `3.0` | `651` | `-16.4354` | `12.355` | `55.80%` | `0.716643` | `19.7802` | `0.581394` | `5.7179` | `0` / `0` / `0` |
| `b30_ProjType_Uniform_QuadSnake_wcol` | `3.0` | `651` | `-16.4368` | `12.338` | `55.42%` | `0.873938` | `23.2957` | `0.458408` | `5.0620` | `0` / `0` / `0` |
| `b30_hA16_BlockH2_real` | `3.0` | `651` | `-10.6919` | `27.634` | `32.35%` | `0.190650` | `12.9820` | `0.146893` | `3.5847` | `0` / `0` / `0` |
| `b30_hA16_BlockH4_quad_lpt_p1000_real` | `3.0` | `651` | `-13.2620` | `23.179` | `53.84%` | `0.174586` | `12.0690` | `0.147673` | `3.6339` | `0` / `0` / `0` |
| `b30_hA16_BlockH4_quad_lpt_sqrt_real` | `3.0` | `651` | `-12.3261` | `24.985` | `53.65%` | `0.171152` | `12.2349` | `0.156449` | `3.6316` | `0` / `0` / `0` |
| `b30_hA16_BlockH4_quad_snake_rand` | `3.0` | `651` | `-11.3450` | `26.376` | `52.77%` | `0.207929` | `12.6959` | `0.170879` | `3.6656` | `0` / `0` / `0` |
| `b30_hA16_BlockH4_quad_snake_real` | `3.0` | `651` | `-12.0279` | `25.676` | `53.26%` | `0.168874` | `12.1407` | `0.151010` | `3.5932` | `0` / `0` / `0` |
| `b30_hA16_BlockH4_quad_snake_sharednorm` | `3.0` | `651` | `-12.0389` | `25.674` | `53.25%` | `0.169901` | `12.0812` | `0.146842` | `3.5707` | `0` / `0` / `0` |
| `b30_hA16_BlockH4_real` | `3.0` | `651` | `-9.6956` | `29.000` | `56.07%` | `0.174061` | `12.2236` | `0.135902` | `3.5686` | `0` / `0` / `0` |
| `b30_hA16_BlockH4_sort_real` | `3.0` | `651` | `-14.4710` | `20.952` | `41.32%` | `0.197239` | `12.2649` | `0.155464` | `3.5939` | `0` / `0` / `0` |
| `b30_hA16_real` | `3.0` | `651` | `-12.4253` | `24.557` | `86.00%` | `0.170814` | `12.3049` | `0.131918` | `3.5620` | `0` / `0` / `0` |
| `b45_hA16_BlockH2_real` | `4.5` | `5213` | `-19.4442` | `18.804` | `16.96%` | `0.024179` | `11.0576` | `0.022975` | `3.1749` | `0` / `0` / `0` |
| `b45_hA16_BlockH4_quad_lpt_p0125_real` | `4.5` | `5213` | `-20.7773` | `16.829` | `43.20%` | `0.025841` | `11.2438` | `0.022901` | `3.1871` | `0` / `0` / `0` |
| `b45_hA16_BlockH4_quad_lpt_p0250_real` | `4.5` | `5213` | `-20.8117` | `16.836` | `43.16%` | `0.026219` | `11.2146` | `0.020832` | `3.1744` | `0` / `0` / `0` |
| `b45_hA16_BlockH4_quad_lpt_p0250_sharednorm` | `4.5` | `5213` | `-20.8232` | `16.833` | `43.12%` | `0.026460` | `11.2618` | `0.021400` | `3.1724` | `0` / `0` / `0` |
| `b45_hA16_BlockH4_quad_lpt_p0375_real` | `4.5` | `5213` | `-20.9622` | `16.381` | `43.39%` | `0.024678` | `11.1318` | `0.018248` | `3.1623` | `0` / `0` / `0` |
| `b45_hA16_BlockH4_quad_lpt_p0750_real` | `4.5` | `5213` | `-21.2936` | `15.743` | `43.61%` | `0.028536` | `11.1940` | `0.019772` | `3.1709` | `0` / `0` / `0` |
| `b45_hA16_BlockH4_quad_lpt_p1000_real` | `4.5` | `5213` | `-21.8744` | `14.637` | `43.62%` | `0.026776` | `11.2461` | `0.021951` | `3.1755` | `0` / `0` / `0` |
| `b45_hA16_BlockH4_quad_lpt_sqrt_real` | `4.5` | `5213` | `-21.0174` | `16.250` | `43.53%` | `0.025296` | `11.1876` | `0.019520` | `3.1711` | `0` / `0` / `0` |
| `b45_hA16_BlockH4_quad_lpt_sqrt_sharednorm` | `4.5` | `5213` | `-21.0291` | `16.247` | `43.49%` | `0.025409` | `11.1565` | `0.020597` | `3.1719` | `0` / `0` / `0` |
| `b45_hA16_BlockH4_quad_snake_down_zerocost` | `4.5` | `5213` | `-19.1147` | `19.627` | `45.60%` | `0.025433` | `11.2519` | `0.020392` | `3.1497` | `0` / `0` / `0` |
| `b45_hA16_BlockH4_quad_snake_real` | `4.5` | `5213` | `-20.7013` | `17.014` | `43.15%` | `0.023154` | `11.0636` | `0.020494` | `3.1666` | `0` / `0` / `0` |
| `b45_hA16_BlockH4_quad_snake_sharednorm` | `4.5` | `5213` | `-20.7125` | `17.011` | `43.11%` | `0.024020` | `11.1271` | `0.021344` | `3.1679` | `0` / `0` / `0` |
| `b45_hA16_BlockH4_real` | `4.5` | `5213` | `-18.5273` | `20.110` | `45.22%` | `0.024774` | `11.2062` | `0.022519` | `3.1670` | `0` / `0` / `0` |
| `b45_hA16_BlockH4_sort_real` | `4.5` | `5213` | `-23.5220` | `11.976` | `32.03%` | `0.026025` | `11.1903` | `0.021765` | `3.1785` | `0` / `0` / `0` |

## 5. Cross-Model Ball-0 Starvation & L2-Energy CV Telemetry (`4` Models $\times$ `7` $p$-Values)

| Model (`g`) | Regime | Quad Spread | Ball CV (lin) | Ball CV (sqrt) | Starved Mats | Ball-0 2nd Rank / $G$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `SmolLM2-360M` (`32`) | `p0000_snake` | `0.6582` | `1.8507` | `0.2687` | `0.00%` | `1.97` |
| `SmolLM2-360M` (`32`) | `p0125` | `0.6582` | `1.8429` | `0.2597` | `25.00%` | `2.12` |
| `SmolLM2-360M` (`32`) | `p0250` | `0.6582` | `1.8359` | `0.2484` | `84.82%` | `3.48` |
| `SmolLM2-360M` (`32`) | `p0375` | `0.6582` | `1.8258` | `0.2352` | `98.66%` | `5.63` |
| `SmolLM2-360M` (`32`) | `p0500_sqrt` | `0.6582` | `1.8171` | `0.2261` | `100.00%` | `6.86` |
| `SmolLM2-360M` (`32`) | `p0750` | `0.6582` | `1.8078` | `0.2227` | `100.00%` | `7.72` |
| `SmolLM2-360M` (`32`) | `p1000_lin` | `0.6582` | `1.8039` | `0.2285` | `100.00%` | `7.79` |
| `Qwen2.5-0.5B` (`128`) | `p0000_snake` | `0.4123` | `0.5696` | `0.0679` | `0.00%` | `1.87` |
| `Qwen2.5-0.5B` (`128`) | `p0125` | `0.4123` | `0.5568` | `0.0589` | `1.79%` | `1.89` |
| `Qwen2.5-0.5B` (`128`) | `p0250` | `0.4123` | `0.5526` | `0.0548` | `38.10%` | `2.21` |
| `Qwen2.5-0.5B` (`128`) | `p0375` | `0.4123` | `0.5395` | `0.0434` | `82.14%` | `3.37` |
| `Qwen2.5-0.5B` (`128`) | `p0500_sqrt` | `0.4123` | `0.5207` | `0.0259` | `92.26%` | `5.04` |
| `Qwen2.5-0.5B` (`128`) | `p0750` | `0.4123` | `0.4660` | `0.0573` | `97.62%` | `16.40` |
| `Qwen2.5-0.5B` (`128`) | `p1000_lin` | `0.4123` | `0.4404` | `0.0925` | `99.40%` | `25.02` |
| `Qwen2.5-1.5B` (`128`) | `p0000_snake` | `0.3165` | `0.8157` | `0.1033` | `0.00%` | `1.93` |
| `Qwen2.5-1.5B` (`128`) | `p0125` | `0.3165` | `0.8054` | `0.0951` | `3.06%` | `1.97` |
| `Qwen2.5-1.5B` (`128`) | `p0250` | `0.3165` | `0.7986` | `0.0887` | `78.06%` | `2.89` |
| `Qwen2.5-1.5B` (`128`) | `p0375` | `0.3165` | `0.7845` | `0.0754` | `93.88%` | `4.52` |
| `Qwen2.5-1.5B` (`128`) | `p0500_sqrt` | `0.3165` | `0.7653` | `0.0560` | `96.43%` | `7.94` |
| `Qwen2.5-1.5B` (`128`) | `p0750` | `0.3165` | `0.7243` | `0.0631` | `99.49%` | `24.58` |
| `Qwen2.5-1.5B` (`128`) | `p1000_lin` | `0.3165` | `0.7102` | `0.0892` | `99.49%` | `28.05` |
| `Qwen2.5-3B` (`128`) | `p0000_snake` | `0.3379` | `0.8823` | `0.1173` | `0.00%` | `1.94` |
| `Qwen2.5-3B` (`128`) | `p0125` | `0.3379` | `0.8725` | `0.1094` | `6.75%` | `2.00` |
| `Qwen2.5-3B` (`128`) | `p0250` | `0.3379` | `0.8662` | `0.1032` | `80.56%` | `3.00` |
| `Qwen2.5-3B` (`128`) | `p0375` | `0.3379` | `0.8538` | `0.0912` | `92.06%` | `5.28` |
| `Qwen2.5-3B` (`128`) | `p0500_sqrt` | `0.3379` | `0.8365` | `0.0746` | `98.41%` | `9.23` |
| `Qwen2.5-3B` (`128`) | `p0750` | `0.3379` | `0.7993` | `0.0796` | `100.00%` | `25.25` |
| `Qwen2.5-3B` (`128`) | `p1000_lin` | `0.3379` | `0.7830` | `0.1077` | `100.00%` | `28.56` |

## 6. Cryptographic Artifact Manifest & Verification Commands

- **Certificate Inner SHA-256 (`certificate_sha256`):** `8dfd31ba6d560a0af558c41370ee76b09a1b32d6d0fcc3b40443ef0f1de70bf6`
- **Pre-Registration Inner SHA-256 (`preregistration_sha256`):** `078207526e3a62e301865f79f0ef2024e0d1b5b2caf2feb9cc5e49734a46265b`
- **ML-OS Hardware Receipt Inner SHA-256 (`receipt_sha256`):** `5784d88f7c00c0d341b17fd2a8f5cd89f7a38b87ef0a0c8c376c2455b42658c2`

| File | Mode | Lines | Bytes | SHA-256 |
| :--- | :---: | :---: | :---: | :--- |
| `s73_preregistration.json` | `0o644` | `1` | `5459` | `d9c72972dbb9639e8f10e17af0b1f0904727da5c4362caa95b1392a6c51f9532` |
| `rtx3080_s73_lowbit_eval.py` | `0o644` | `1026` | `42485` | `ddec78ad0145867bca7c44ba40f4b861fbb3c97dfb33688897f814907d6929a9` |
| `rtx3080_s73_empirical_results.json` | `0o644` | `3018` | `97259` | `d7b2ddf000cdb2e23046f69aa9b0446932484f75fd2ec3e706f73456a6a8403c` |
| `mlos_zone_b_receipt.json` | `0o644` | `91` | `2627` | `3d38e9fa5c43b56fff18c6a4743f6186bdfb9e3a1bf83e7297ed7631e6cca9bc` |
| `rtx3080_s73_oracle.py` | `0o644` | `704` | `34984` | `41b6ca843fa826c018ee87f1bc79253a1e9cdf7c552f68cfb6eaa72d4db81404` |
| `rtx3080_s73_audit.py` | `0o644` | `542` | `21613` | `0c14a73d3c5feaf236ab81b57d58d2b392a3a689d1dd4f4f68516bafa0721161` |
| `rtx3080_s73_certificate.json` | `0o644` | `1` | `5202` | `33dfb4153fe974e9ef13120190c14ada6dda63ffbee6caa46047d274f32d36cf` |
| `test_rtx3080_s73_r5.py` | `0o644` | `828` | `52297` | `a19cc343ac89e3ca19cc4cdcfa638f426b988ca35da4a5320ebc3c7fc49191bc` |

```bash
python3 -I -B rtx3080_s73_audit.py --check rtx3080_s73_certificate.json
python3 -I -B rtx3080_s73_audit.py --verify-receipts .
python3 -I -B test_rtx3080_s73_r5.py
```

### Verification Scope, Provenance & Citation Notes

- **In-repo verification** is exactly the three commands above plus the 5-test suite `test_rtx3080_s73_r5.py` (which includes its own token-tamper and receipt-tamper mutation controls). The AUDIT-173 / 82/82 ZBF / 79/79 fast-mutate / 232/232 whole-file AST BoolOp / dual aletheia audit-check counts quoted in the PR body come from external tooling that is not in this repo and cannot be reproduced from it.
- **Provenance:** `mlos_zone_b_receipt.json:51` has `"git_head": null`, and `FROZEN_S73_TOKENS` (`rtx3080_s73_oracle.py:69`) plus the digest pins live in this same study. The checks above therefore prove internal consistency + gate inequalities against the committed `rtx3080_s73_empirical_results.json`, not independent re-execution of the GPU run. Code pin shipped by this study: `rtx3080_s73_lowbit_eval.py` SHA-256 `ddec78ad0145867bca7c44ba40f4b861fbb3c97dfb33688897f814907d6929a9` (git blob `b4ba9bd42ec23f00c6cf7a85d0d48c5d3bae7318`); the study first landed at commit `3065627db68ea888d5ba871f6ac2aebdb8738706` (PR #103 head).
- **Preregistration citation:** `s73_preregistration.json` cites "siliquant#93's 606.20 Gw/s / +4.47% sm_86 PTX kernel". Those figures are recorded on main in `studies/rtx3080-s69-sorted-blockh4-coupled/RTX3080_S69_SORTED_BLOCKH4_REPORT.md` (Table 5A: `down_proj` DOWN offline-fold kernel `606.20 Gw/s`; Table 5B: gate+up shared-RMSNorm gather `+4.47%` vs unpermuted) — isolated-kernel GEMV figures, not decode throughput.
- **Mode column:** git stores all 10 study files as `0644` (git cannot represent `0o600`); the Mode column lists the git-stored mode.
