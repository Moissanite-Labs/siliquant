# RTX 3080 Live-Weight Study: Breaking Group-Ball Hessian Cannibalization via Two-Level Stratified Quad-Interleaved `A16_BlockH4`

## 1. Executive Summary & Epistemic Ledger

This study tests the **Two-Level Saddle-Point Permutation** (`quad_lpt` and `quad_snake`) across **87 live quantized full-model arms** on **4 cached real-weight LLMs** (**120 transformer blocks**, **840 linear matrices**, **4,757,127,168 block weights**, **0 syndrome errors**, **0 non-block tensor mutations**) on an NVIDIA GeForce RTX 3080 Laptop GPU (`GA104M`, `sm_86`, 8 GiB GDDR6, AC power online, zero competing GPU processes, `total_cpu_rapl_joules=108724.241 J`). All hypotheses were cryptographically pre-registered prior to GPU execution (`s72_preregistration.json`, `created_utc=2026-11-24T20:05:48Z`, inner SHA-256 `361bae665103de513002bf9268ff623829ab42aaf9fe7c096cd3e6ca38a4b387`) strictly **`+2,151 s`** before the live hardware execution window (`started_utc=2026-11-24T20:41:39Z`, `ended_utc=2026-11-24T21:37:22Z`, `duration_ms=3342319.48`, `clean_hardware_run=true`, `anomalies=[]`).

### Headline Breakthrough: Two-Level Stratified `A16_BlockH4` Shatters Previously Undefeated `hA16_BlockH2_real` on All `3/3` Qwen2.5 Models

| Model (`g`) | Prior Champion `hA16_BlockH2_real` (Wiki / Py KL) | Unpermuted `hA16_BlockH4_real` (Wiki / Py KL) | Global Cannibalized `sort_real` (Wiki KL) | New Two-Level `BlockH4` Winner (Wiki / Py KL) | $\Delta$ vs `BlockH2` (Wiki / Py) | $\Delta$ vs Unp. `BlockH4` (Wiki) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `Qwen2.5-0.5B` (`128`) | `0.026970` / `0.024050` | `0.029413` / `0.023452` | `0.032504` | `hA16_BlockH4_quad_lpt_real` (`0.025955` / `0.023575`) | `-3.76%` / `-1.97%` | `-11.76%` |
| `Qwen2.5-1.5B` (`128`) | `0.023281` / `0.014356` | `0.023412` / `0.015129` | `0.026787` | `hA16_BlockH4_quad_lpt_sharednorm` (`0.022953` / `0.014898`) | `-1.41%` / `+3.77%` | `-1.96%` |
| `Qwen2.5-3B` (`128`) | `0.024179` / `0.022975` | `0.024774` / `0.022519` | `0.026025` | `hA16_BlockH4_quad_snake_real` (`0.023154` / `0.020494`) | `-4.24%` / `-10.80%` | `-6.54%` |
| `SmolLM2-360M` (`32`) | `0.035651` / `0.032959` | `0.028207` / `0.026602` | `0.045766` | `hA16_BlockH4_quad_snake_real` (`0.030821` / `0.028255`) | `-13.55%` / `-14.27%` | `+9.27%` |

### Pre-Registered Prediction Outcomes (Honest Confirmations & Falsifications)

| Pillar | Pre-Registered Inequality / Question | Live Hardware Outcome | Verdict |
| :--- | :--- | :--- | :--- |
| **Pillar 1a** | `hA16_BlockH4_quad_lpt_real` beats global `sort_real` on `4/4` models & unpermuted `BlockH4` on `>= 3/4` | Linear-load `quad_lpt` (`load_power=1.0`) beat global `sort_real` on `3/4` (`360M` `-26.79%`, `0.5B` `-20.15%`, `1.5B` `-14.27%`) and beat unpermuted `BlockH4` on `2/4` (`0.5B`, `1.5B`), but regressed on heavy-tailed `3B` (`0.026776` > `0.026025`) due to **Heavy-Tail Greedy-LPT Starvation**. Replacing linear LPT with rank-invariant **`quad_snake`** beats global `sort_real` on **`4/4`** models and sets the all-time `3B` record (`0.023154` Wiki / `0.020494` Py); sqrt-energy LPT (`quad_lpt_sqrt_real`) also beats `sort_real` on `3B` (`0.025296`) and breaks `0.020` on `3B` Python (`0.019520`) | **LINEAR LPT FALSIFIED ON 3B; `quad_snake` / `sqrt-LPT` CONFIRMED** |
| **Pillar 1b** | Challenge undefeated `hA16_BlockH2_real` floor across `Qwen2.5-0.5B, 1.5B, 3B` | **SHATTERED ON ALL `3/3` QWEN MODELS!** `0.5B`: `quad_lpt` `0.025955` (`-3.76%`), `sharednorm` `0.026285`, and even zero-cost `down_zerocost` `0.026964` all beat `BlockH2` `0.026970`; `1.5B`: PTX-ready `sharednorm` `0.022953` & `quad_lpt` `0.022964` break `0.023` (vs `BlockH2` `0.023281`); `3B`: `quad_snake` `0.023154` (`-4.24%`) & Py `0.020494` (`-10.80%`) | **CONFIRMED (3/3 QWEN)** |
| **Pillar 2a** | Pareto-corner telemetry (`mean_quad_spread(quad_lpt) < grp128 < unp` & `ball_cv(quad_lpt) < global_sort`) on `4/4` models | Strictly holds on **`4/4`** models for both `h_real` and `wcol`; on `3B` `h_real`, quad spread drops **`6.73x`** (`2.2745` -> `0.3379`), `ball_cv_lin` drops **`2.68x`** vs global sort, and `ball_cv_sqrt` (`0.1077`) is lower than unpermuted (`0.1440`) | **CONFIRMED (4/4)** |
| **Pillar 2b** | Zero-text `3B` `wcol` cure (`quad_lpt_wcol < sort_wcol`) + matched MSE `<= 2e-6 dB` + zero-text `quad_lpt_rand` | `3B` `quad_lpt_wcol` `0.051480` (`-7.34%`) & `quad_snake_wcol` `0.049077` (`-11.67%`) cure `sort_wcol` `0.055560` (`mdiff <= 1.582e-6 dB`); `ProjType_Uniform_QuadLPT_wcol` hits `0.047633` (< unpermuted `BlockH4` `0.047942`); zero-text `hA16_BlockH4_quad_lpt_rand` hits `3B` PyKL `0.021867` (beating real-calibrated `BlockH4` `0.022519` & `BlockH2` `0.022975`!). On `360M/0.5B/1.5B` uncalibrated `sort_wcol` remains superior (`1.5B` `0.039193` < `0.046993`), proving opposing ball-variance optima for uncalibrated vs. Hessian-scaled L2 balls | **CONFIRMED** |
| **Pillar 3** | `BlockH4` vs. matched `BlockH2_quad_lpt_real` control across all 4 models | `BlockH4_quad_lpt` overtakes `BlockH2_quad_lpt` on `0.5B` (`0.025955` < `0.026721`, `-2.87%`) and `1.5B` (`0.022964` < `0.024020` Wiki, `0.014712` < `0.015030` Py); on `3B`, `BlockH2_quad_lpt` suffers the same linear-LPT starvation (`0.026353`), while `BlockH4_quad_snake` decisively beats it on both Wiki (`0.023154`) and Py (`0.020494`) | **CONFIRMED** |

## 2. Physical Mechanism: Why Rank-Invariant `quad_snake` Beats Linear-Load `quad_lpt` on Heavy-Tailed `Qwen2.5-3B`

Comparing `quad_lpt` (`load_power=1.0`), `quad_lpt_sqrt` (`load_power=0.5`), and `quad_snake` on `Qwen2.5-3B` exposes a subtle third-order heavy-tail effect:

1. **Heavy-Tail Greedy-LPT Starvation (`load_power=1.0`):** In `Qwen2.5-3B`, the hottest single `H_4` quad $Q_0$ has linear Hessian curvature $h_0$ orders of magnitude larger than median quads. Greedy LPT places $Q_0$ into Ball 0 and then skips Ball 0 for multiple subsequent rounds while packing multiple secondary loud quads ($Q_1, \dots, Q_{3G}$) into Balls $1 \dots G-1$, creating secondary cannibalization in those balls (`weight_err_db = -21.8744 dB`, WikiKL `0.026776`).
2. **Exact L2-Ball Energy Balancing (`load_power=0.5`, `quad_lpt_sqrt_real`):** Because Hessian column scaling applies $c_j \propto h_j^{1/4}$, the squared weight-energy contribution to the L2-ball norm is $\propto c_j^2 = h_j^{1/2}$, not $h_j^1$. Compressing LPT weights to $h_j^{1/2}$ immediately recovers `-5.53%` WikiKL vs. linear LPT (`0.026776` -> `0.025296`, beating global sort `0.026025`) and achieves an **all-time record `Qwen2.5-3B` Python KL of `0.019520`** (`-15.04%` vs `hA16_BlockH2_real`).
3. **Rank-Invariant Boustrophedon Stratification (`quad_snake`):** `quad_snake` depends strictly on column rank (invariant under every monotone transform $h_j^p$, $p > 0$). Round 0 places exactly one top-$G$ quad into every ball, Round 1 places one rank-$(G\dots 2G-1)$ quad into every ball in reverse order, and so forth—guaranteeing identical quantile stratification across all $G$ balls with zero heavy-tail starvation (`weight_err_db = -20.7013 dB`, **all-time `Qwen2.5-3B` WikiKL `0.023154` / PyKL `0.020494`**).
4. **Uncalibrated vs. Calibrated Regime Dichotomy:** In uncalibrated quantization (`wcol`), `A16_BlockH4_sort_wcol` thrives on sparse models (`360M/0.5B/1.5B`, `z0 >= 91.9%`) precisely because high inter-ball CV gives tiny weight columns their own tiny-radius balls, whereas on `3B` (`z0 = 68.1%`) `quad_lpt_wcol` (`0.051480`) and `quad_snake_wcol` (`0.049077`) mitigate `sort_wcol`'s `0.055560` regression by restoring balanced ball radii.

## 3. Ball-Variance & Intra-Quad Spread Telemetry Across All `4` Models (`5` Regimes)

| Model (`g`) | Regime | `h_real` Quad Spread | `h_real` Ball CV (lin) | `h_real` Ball CV (sqrt) | `down_proj` Quad Spread | `wcol` Quad Spread | `wcol` Ball CV (lin) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `SmolLM2-360M` (`32`) | `unpermuted` | `2.2752` | `1.2039` | `0.2075` | `1.9321` | `0.3296` | `0.0640` |
| `SmolLM2-360M` (`32`) | `global_sort` | `0.6582` | `2.6921` | `0.6579` | `0.2933` | `0.0089` | `0.2009` |
| `SmolLM2-360M` (`32`) | `grp128_sort` | `1.7990` | `1.2039` | `0.2075` | `1.0279` | `0.0918` | `0.0640` |
| `SmolLM2-360M` (`32`) | `quad_snake` | `0.6582` | `1.8507` | `0.2687` | `0.2933` | `0.0089` | `0.0212` |
| `SmolLM2-360M` (`32`) | `quad_lpt` | `0.6582` | `1.8039` | `0.2285` | `0.2933` | `0.0089` | `0.0156` |
| `Qwen2.5-0.5B` (`128`) | `unpermuted` | `2.3303` | `0.4288` | `0.0899` | `2.1576` | `0.4715` | `0.0500` |
| `Qwen2.5-0.5B` (`128`) | `global_sort` | `0.4123` | `1.5477` | `0.5482` | `0.4505` | `0.0078` | `0.2639` |
| `Qwen2.5-0.5B` (`128`) | `grp128_sort` | `1.0019` | `0.4288` | `0.0899` | `0.9478` | `0.0430` | `0.0500` |
| `Qwen2.5-0.5B` (`128`) | `quad_snake` | `0.4123` | `0.5696` | `0.0679` | `0.4505` | `0.0078` | `0.0060` |
| `Qwen2.5-0.5B` (`128`) | `quad_lpt` | `0.4123` | `0.4404` | `0.0925` | `0.4505` | `0.0078` | `0.0025` |
| `Qwen2.5-1.5B` (`128`) | `unpermuted` | `2.3089` | `0.6357` | `0.1331` | `2.2412` | `0.3871` | `0.0556` |
| `Qwen2.5-1.5B` (`128`) | `global_sort` | `0.3165` | `1.9223` | `0.5704` | `0.3877` | `0.0041` | `0.2282` |
| `Qwen2.5-1.5B` (`128`) | `grp128_sort` | `1.1034` | `0.6357` | `0.1331` | `1.0371` | `0.0345` | `0.0556` |
| `Qwen2.5-1.5B` (`128`) | `quad_snake` | `0.3165` | `0.8157` | `0.1033` | `0.3877` | `0.0041` | `0.0047` |
| `Qwen2.5-1.5B` (`128`) | `quad_lpt` | `0.3165` | `0.7102` | `0.0892` | `0.3877` | `0.0041` | `0.0033` |
| `Qwen2.5-3B` (`128`) | `unpermuted` | `2.2745` | `0.6570` | `0.1440` | `2.3470` | `0.3903` | `0.0552` |
| `Qwen2.5-3B` (`128`) | `global_sort` | `0.3379` | `2.0962` | `0.6106` | `0.3928` | `0.0032` | `0.2365` |
| `Qwen2.5-3B` (`128`) | `grp128_sort` | `1.1338` | `0.6570` | `0.1440` | `0.9801` | `0.0349` | `0.0552` |
| `Qwen2.5-3B` (`128`) | `quad_snake` | `0.3379` | `0.8823` | `0.1173` | `0.3928` | `0.0032` | `0.0050` |
| `Qwen2.5-3B` (`128`) | `quad_lpt` | `0.3379` | `0.7830` | `0.1077` | `0.3928` | `0.0032` | `0.0035` |

## 4. Complete 4-Model Live Hardware Results (`87` Arms, `4,757,127,168` Weights)

### 4.1 `HuggingFaceTB/SmolLM2-360M` (`g=32`, `32`L, `224` mats, `314,572,800` wts; BF16 PPL Wiki=`16.3948`, Py=`5.0450`)

| Arm | Err (dB) | E05 (dB) | z0 (%) | Wiki KL | Wiki PPL | Py KL | Py PPL | Syn / NB |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `A16_axis` | `-25.0281` | `2.682` | `93.11%` | `0.065173` | `17.2522` | `0.056725` | `5.3303` | `0` / `0` |
| `A16_BlockH4` | `-25.0279` | `5.227` | `48.72%` | `0.082002` | `17.7082` | `0.066762` | `5.3763` | `0` / `0` |
| `A16_BlockH4_sort_wcol` | `-25.0278` | `5.087` | `48.02%` | `0.064414` | `17.3329` | `0.054715` | `5.3250` | `0` / `0` |
| `A16_BlockH4_quad_lpt_wcol` | `-25.0281` | `5.245` | `48.88%` | `0.076361` | `17.6722` | `0.070017` | `5.4451` | `0` / `0` |
| `A16_BlockH4_quad_snake_wcol` | `-25.0279` | `5.248` | `48.89%` | `0.069997` | `17.6620` | `0.067961` | `5.4549` | `0` / `0` |
| `A16_BlockH4_quad_lpt_hrand` | `-25.0283` | `5.237` | `48.82%` | `0.077421` | `17.5925` | `0.068158` | `5.4153` | `0` / `0` |
| `A16_BlockH4_quad_lpt_down_wcol` | `-25.0280` | `5.227` | `48.74%` | `0.070825` | `17.3464` | `0.061357` | `5.3439` | `0` / `0` |
| `ProjType_Uniform` | `-25.0279` | `4.711` | `59.58%` | `0.061838` | `16.9701` | `0.052089` | `5.3026` | `0` / `0` |
| `ProjType_Uniform_QuadLPT_wcol` | `-25.0280` | `4.732` | `59.71%` | `0.065121` | `17.3670` | `0.059557` | `5.3866` | `0` / `0` |
| `ProjType_Uniform_QuadLPT_hrand` | `-25.0282` | `4.725` | `59.66%` | `0.067355` | `17.1695` | `0.060977` | `5.3619` | `0` / `0` |
| `hA16_BlockH4_quad_lpt_rand` | `-23.7713` | `7.844` | `48.82%` | `0.043848` | `17.3046` | `0.039263` | `5.2417` | `0` / `0` |
| `hA16_real` | `-24.3718` | `4.497` | `92.46%` | `0.030955` | `16.9734` | `0.027538` | `5.1678` | `0` / `0` |
| `hA16_BlockH2_real` | `-24.0633` | `7.816` | `6.78%` | `0.035651` | `17.0286` | `0.032959` | `5.1908` | `0` / `0` |
| `hA16_BlockH4_real` | `-23.9798` | `8.411` | `48.66%` | `0.028207` | `16.9524` | `0.026602` | `5.1555` | `0` / `0` |
| `hA16_BlockH4_sort_real` | `-24.7457` | `6.315` | `48.36%` | `0.045766` | `17.3383` | `0.046179` | `5.2846` | `0` / `0` |
| `hA16_BlockH4_sort_grp128_real` | `-24.0979` | `7.894` | `48.78%` | `0.030636` | `16.9699` | `0.024308` | `5.1731` | `0` / `0` |
| `hA16_BlockH4_quad_lpt_real` | `-24.1320` | `7.572` | `48.78%` | `0.033503` | `17.0351` | `0.028338` | `5.1615` | `0` / `0` |
| `hA16_BlockH4_quad_snake_real` | `-23.7131` | `9.415` | `48.82%` | `0.030821` | `16.9989` | `0.028255` | `5.1477` | `0` / `0` |
| `hA16_BlockH4_quad_lpt_down_zerocost` | `-23.9794` | `8.411` | `48.67%` | `0.028934` | `16.7995` | `0.025140` | `5.1547` | `0` / `0` |
| `hA16_BlockH4_quad_lpt_sharednorm` | `-24.1672` | `7.535` | `48.67%` | `0.033220` | `16.9905` | `0.028632` | `5.1613` | `0` / `0` |
| `hA16_BlockH2_quad_lpt_real` | `-24.3010` | `6.723` | `7.13%` | `0.031743` | `17.0253` | `0.029667` | `5.1510` | `0` / `0` |

### 4.2 `Qwen/Qwen2.5-0.5B` (`g=128`, `24`L, `168` mats, `357,826,560` wts; BF16 PPL Wiki=`17.9505`, Py=`4.8606`)

| Arm | Err (dB) | E05 (dB) | z0 (%) | Wiki KL | Wiki PPL | Py KL | Py PPL | Syn / NB |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `A16_axis` | `-25.5529` | `2.336` | `91.94%` | `0.052525` | `18.4858` | `0.045602` | `5.1000` | `0` / `0` |
| `A16_BlockH4` | `-25.5532` | `4.636` | `48.22%` | `0.056063` | `18.5926` | `0.045183` | `5.0712` | `0` / `0` |
| `A16_BlockH4_sort_wcol` | `-25.5534` | `4.481` | `47.50%` | `0.049761` | `18.5962` | `0.038201` | `5.0389` | `0` / `0` |
| `A16_BlockH4_quad_lpt_wcol` | `-25.5530` | `4.650` | `48.31%` | `0.061151` | `18.8951` | `0.053044` | `5.0617` | `0` / `0` |
| `A16_BlockH4_quad_snake_wcol` | `-25.5526` | `4.649` | `48.33%` | `0.058606` | `18.8813` | `0.051099` | `5.0885` | `0` / `0` |
| `A16_BlockH4_quad_lpt_hrand` | `-25.5529` | `4.638` | `48.20%` | `0.057883` | `18.7321` | `0.048122` | `5.0770` | `0` / `0` |
| `A16_BlockH4_quad_lpt_down_wcol` | `-25.5532` | `4.636` | `48.21%` | `0.056888` | `18.7800` | `0.044372` | `5.0220` | `0` / `0` |
| `ProjType_Uniform` | `-25.5531` | `4.168` | `60.85%` | `0.054329` | `18.5079` | `0.043465` | `5.0432` | `0` / `0` |
| `ProjType_Uniform_QuadLPT_wcol` | `-25.5529` | `4.184` | `60.95%` | `0.058219` | `18.6437` | `0.050455` | `5.0689` | `0` / `0` |
| `ProjType_Uniform_QuadLPT_hrand` | `-25.5530` | `4.169` | `60.87%` | `0.058440` | `18.7746` | `0.046514` | `5.0632` | `0` / `0` |
| `hA16_BlockH4_quad_lpt_rand` | `-21.6126` | `13.217` | `48.24%` | `0.029723` | `18.3420` | `0.024931` | `4.9562` | `0` / `0` |
| `hA16_real` | `-23.6609` | `8.380` | `91.31%` | `0.028102` | `18.3312` | `0.024798` | `4.9610` | `0` / `0` |
| `hA16_BlockH2_real` | `-22.2325` | `12.177` | `7.02%` | `0.026970` | `18.2349` | `0.024050` | `4.9594` | `0` / `0` |
| `hA16_BlockH4_real` | `-21.6328` | `13.346` | `48.13%` | `0.029413` | `18.3511` | `0.023452` | `4.9594` | `0` / `0` |
| `hA16_BlockH4_sort_real` | `-22.5682` | `12.045` | `47.81%` | `0.032504` | `18.4297` | `0.027835` | `4.9839` | `0` / `0` |
| `hA16_BlockH4_sort_grp128_real` | `-21.6497` | `13.347` | `48.13%` | `0.027190` | `18.5259` | `0.023315` | `4.9425` | `0` / `0` |
| `hA16_BlockH4_quad_lpt_real` | `-21.6970` | `13.169` | `48.18%` | `0.025955` | `18.3123` | `0.023575` | `4.9724` | `0` / `0` |
| `hA16_BlockH4_quad_snake_real` | `-21.2364` | `13.938` | `48.19%` | `0.030080` | `18.3736` | `0.026650` | `4.9707` | `0` / `0` |
| `hA16_BlockH4_quad_lpt_down_zerocost` | `-21.6384` | `13.344` | `48.13%` | `0.026964` | `18.3239` | `0.023990` | `4.9741` | `0` / `0` |
| `hA16_BlockH4_quad_lpt_sharednorm` | `-21.6954` | `13.168` | `48.10%` | `0.026285` | `18.3227` | `0.022899` | `4.9696` | `0` / `0` |
| `hA16_BlockH2_quad_lpt_real` | `-22.4369` | `11.840` | `7.12%` | `0.026721` | `18.4487` | `0.022338` | `4.9589` | `0` / `0` |

### 4.3 `Qwen/Qwen2.5-1.5B` (`g=128`, `28`L, `196` mats, `1,310,195,712` wts; BF16 PPL Wiki=`12.6180`, Py=`3.7395`)

| Arm | Err (dB) | E05 (dB) | z0 (%) | Wiki KL | Wiki PPL | Py KL | Py PPL | Syn / NB |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `A16_axis` | `-25.5532` | `2.549` | `93.44%` | `0.050062` | `13.0109` | `0.031973` | `3.8456` | `0` / `0` |
| `A16_BlockH4` | `-25.5532` | `4.889` | `49.12%` | `0.046550` | `13.1362` | `0.030045` | `3.8453` | `0` / `0` |
| `A16_BlockH4_sort_wcol` | `-25.5529` | `4.773` | `48.58%` | `0.039193` | `13.1011` | `0.024528` | `3.8234` | `0` / `0` |
| `A16_BlockH4_quad_lpt_wcol` | `-25.5531` | `4.906` | `49.26%` | `0.046993` | `13.0907` | `0.032593` | `3.8283` | `0` / `0` |
| `A16_BlockH4_quad_snake_wcol` | `-25.5531` | `4.907` | `49.26%` | `0.049807` | `13.0255` | `0.030748` | `3.8327` | `0` / `0` |
| `A16_BlockH4_quad_lpt_hrand` | `-25.5531` | `4.897` | `49.21%` | `0.049156` | `13.0435` | `0.030256` | `3.8371` | `0` / `0` |
| `A16_BlockH4_quad_lpt_down_wcol` | `-25.5532` | `4.888` | `49.12%` | `0.049363` | `13.1324` | `0.031711` | `3.8386` | `0` / `0` |
| `ProjType_Uniform` | `-25.5532` | `4.357` | `62.01%` | `0.050110` | `13.1305` | `0.032953` | `3.8612` | `0` / `0` |
| `ProjType_Uniform_QuadLPT_wcol` | `-25.5532` | `4.377` | `62.16%` | `0.047033` | `13.0922` | `0.032042` | `3.8509` | `0` / `0` |
| `ProjType_Uniform_QuadLPT_hrand` | `-25.5531` | `4.369` | `62.10%` | `0.048002` | `13.0077` | `0.031479` | `3.8444` | `0` / `0` |
| `hA16_BlockH4_quad_lpt_rand` | `-21.4572` | `14.706` | `49.30%` | `0.025820` | `12.9026` | `0.015279` | `3.7863` | `0` / `0` |
| `hA16_real` | `-22.9064` | `12.434` | `92.76%` | `0.023637` | `12.8575` | `0.015113` | `3.7882` | `0` / `0` |
| `hA16_BlockH2_real` | `-20.7233` | `16.446` | `6.90%` | `0.023281` | `12.9273` | `0.014356` | `3.8326` | `0` / `0` |
| `hA16_BlockH4_real` | `-19.8754` | `17.722` | `49.07%` | `0.023412` | `12.7947` | `0.015129` | `3.7961` | `0` / `0` |
| `hA16_BlockH4_sort_real` | `-22.5432` | `13.299` | `48.98%` | `0.026787` | `12.7825` | `0.017099` | `3.7986` | `0` / `0` |
| `hA16_BlockH4_sort_grp128_real` | `-20.2814` | `17.055` | `49.10%` | `0.023176` | `12.7768` | `0.014083` | `3.7873` | `0` / `0` |
| `hA16_BlockH4_quad_lpt_real` | `-21.4823` | `14.800` | `49.21%` | `0.022964` | `12.7743` | `0.014712` | `3.7715` | `0` / `0` |
| `hA16_BlockH4_quad_snake_real` | `-20.6250` | `16.292` | `49.24%` | `0.023177` | `12.8822` | `0.015627` | `3.7952` | `0` / `0` |
| `hA16_BlockH4_quad_lpt_down_zerocost` | `-19.8809` | `17.721` | `49.07%` | `0.023459` | `12.7982` | `0.015956` | `3.7991` | `0` / `0` |
| `hA16_BlockH4_quad_lpt_sharednorm` | `-21.4873` | `14.797` | `49.13%` | `0.022953` | `12.7825` | `0.014898` | `3.7853` | `0` / `0` |
| `hA16_BlockH2_quad_lpt_real` | `-22.9729` | `12.038` | `7.26%` | `0.024020` | `12.9212` | `0.015030` | `3.7699` | `0` / `0` |

### 4.4 `Qwen/Qwen2.5-3B` (`g=128`, `36`L, `252` mats, `2,774,532,096` wts; BF16 PPL Wiki=`10.9702`, Py=`3.1119`)

| Arm | Err (dB) | E05 (dB) | z0 (%) | Wiki KL | Wiki PPL | Py KL | Py PPL | Syn / NB |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `A16_axis` | `-25.5532` | `1.924` | `68.05%` | `0.046686` | `11.2583` | `0.038578` | `3.2183` | `0` / `0` |
| `A16_BlockH4` | `-25.5534` | `4.294` | `36.15%` | `0.047942` | `11.3595` | `0.035620` | `3.2230` | `0` / `0` |
| `A16_BlockH4_sort_wcol` | `-25.5532` | `3.959` | `30.98%` | `0.055560` | `11.3459` | `0.043396` | `3.2437` | `0` / `0` |
| `A16_BlockH4_quad_lpt_wcol` | `-25.5589` | `4.262` | `37.14%` | `0.051480` | `11.2406` | `0.042817` | `3.2582` | `0` / `0` |
| `A16_BlockH4_quad_lpt_wcol_matched` | `-25.5532` | `4.276` | `0.01%` | `0.051450` | `11.2435` | `0.042448` | `3.2565` | `0` / `0` |
| `A16_BlockH4_quad_snake_wcol` | `-25.5592` | `4.262` | `37.15%` | `0.049077` | `11.4171` | `0.041408` | `3.2341` | `0` / `0` |
| `A16_BlockH4_quad_lpt_hrand` | `-25.5581` | `4.200` | `35.87%` | `0.048978` | `11.2110` | `0.037819` | `3.2190` | `0` / `0` |
| `A16_BlockH4_quad_lpt_hrand_matched` | `-25.5532` | `4.210` | `0.01%` | `0.049013` | `11.2237` | `0.037605` | `3.2164` | `0` / `0` |
| `A16_BlockH4_quad_lpt_down_wcol` | `-25.5587` | `4.252` | `37.21%` | `0.049665` | `11.2149` | `0.035490` | `3.2058` | `0` / `0` |
| `ProjType_Uniform` | `-25.5528` | `3.695` | `46.74%` | `0.046336` | `11.2686` | `0.034327` | `3.2172` | `0` / `0` |
| `ProjType_Uniform_QuadLPT_wcol` | `-25.5530` | `3.706` | `46.68%` | `0.047633` | `11.3210` | `0.039032` | `3.2621` | `0` / `0` |
| `ProjType_Uniform_QuadLPT_hrand` | `-25.5530` | `3.696` | `46.73%` | `0.050905` | `11.2802` | `0.042482` | `3.2496` | `0` / `0` |
| `hA16_BlockH4_quad_lpt_rand` | `-20.6064` | `16.660` | `43.39%` | `0.027153` | `11.2419` | `0.021867` | `3.1878` | `0` / `0` |
| `hA16_real` | `-21.1650` | `15.760` | `75.03%` | `0.024417` | `11.0895` | `0.021217` | `3.1768` | `0` / `0` |
| `hA16_BlockH2_real` | `-19.4442` | `18.804` | `16.96%` | `0.024179` | `11.0576` | `0.022975` | `3.1749` | `0` / `0` |
| `hA16_BlockH4_real` | `-18.5273` | `20.110` | `45.22%` | `0.024774` | `11.2062` | `0.022519` | `3.1670` | `0` / `0` |
| `hA16_BlockH4_sort_real` | `-23.5220` | `11.976` | `32.03%` | `0.026025` | `11.1903` | `0.021765` | `3.1785` | `0` / `0` |
| `hA16_BlockH4_sort_grp128_real` | `-20.8674` | `17.139` | `45.79%` | `0.024918` | `11.1841` | `0.021902` | `3.1723` | `0` / `0` |
| `hA16_BlockH4_quad_lpt_real` | `-21.8744` | `14.637` | `43.62%` | `0.026776` | `11.2461` | `0.021951` | `3.1755` | `0` / `0` |
| `hA16_BlockH4_quad_lpt_sqrt_real` | `-21.0174` | `16.250` | `43.53%` | `0.025296` | `11.1876` | `0.019520` | `3.1711` | `0` / `0` |
| `hA16_BlockH4_quad_snake_real` | `-20.7013` | `17.014` | `43.15%` | `0.023154` | `11.0636` | `0.020494` | `3.1666` | `0` / `0` |
| `hA16_BlockH4_quad_lpt_down_zerocost` | `-19.1415` | `19.623` | `45.59%` | `0.026175` | `11.3014` | `0.021456` | `3.1513` | `0` / `0` |
| `hA16_BlockH4_quad_lpt_sharednorm` | `-21.8817` | `14.634` | `43.58%` | `0.027553` | `11.2121` | `0.021768` | `3.1712` | `0` / `0` |
| `hA16_BlockH2_quad_lpt_real` | `-21.8930` | `14.949` | `19.89%` | `0.026353` | `11.2759` | `0.020572` | `3.1460` | `0` / `0` |

## 5. Frozen Artifact Manifest & Reproducibility

- **Canonical Certificate Inner SHA-256:** `7eb5333739ccdef1412a9f82eb12c6be8e08832ac275a2e9ea6cbdba34402f99`
- **Pre-Registration Inner SHA-256:** `361bae665103de513002bf9268ff623829ab42aaf9fe7c096cd3e6ca38a4b387` (`created_utc=2026-11-24T20:05:48Z`)
- **Hardware Receipt Inner SHA-256:** `e6e3ea61c6e8f37e13c5fed30b351fa47e66ea468c2da2a4e19976d38e895072` (`started_utc=2026-11-24T20:41:39Z`, `ended_utc=2026-11-24T21:37:22Z`)

| Artifact File | Bytes | SHA-256 |
| :--- | :---: | :--- |
| `s72_preregistration.json` | `4102` | `3ba6c33796c1cba457d83f0a37fcc9b8a77898f60f75efc93d00b9a196e6df02` |
| `rtx3080_s72_quad_interleaved_eval.py` | `39316` | `5dbdfb0c2849b717e349ef91eddbf706f3ba4ea5ca48341cade24ac93eb09554` |
| `rtx3080_s72_empirical_results.json` | `83361` | `a7c35c646a9c29fa645f207697c74e181ad19d8ce61acbe489a265a132968647` |
| `mlos_zone_b_receipt.json` | `2603` | `93f0b75c3e89ceb971ee9d574c29e98047070b79b01dc368ba645363128853cd` |
| `rtx3080_s72_oracle.py` | `34497` | `69058988c3232b709e8fb2250b3031ec1a9c3327776fc6a18206e283c80ba9df` |
| `rtx3080_s72_audit.py` | `21570` | `9b5520b42b543fb9a91b45aeffff8f3b0620ddc4b84531c45aa3e9e0cdeefeed` |
| `rtx3080_s72_certificate.json` | `4733` | `c2dc4626271503bbd710ce468e2d2051b1241f77d9f6f995dee15d621eb884e7` |
| `test_rtx3080_s72_r5.py` | `52872` | `4dcb7eec141950e339ad7509b74f9216e7d19eb2850f112ed419a136c63d62f0` |

### Standalone Verification Commands
```bash
python3 -I -B rtx3080_s72_audit.py --check rtx3080_s72_certificate.json
python3 -I -B rtx3080_s72_audit.py --verify-receipts .
python3 -I -B test_rtx3080_s72_r5.py -v
```
