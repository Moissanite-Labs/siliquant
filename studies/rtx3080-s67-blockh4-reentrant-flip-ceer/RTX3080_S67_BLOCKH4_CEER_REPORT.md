# Track S67 — Live CEER Cycle 2 Evaluation of `A16_BlockH4` / `A16_BlockH8`, 616-Layer Qwen2.5 Re-Entrant Oscillation Diagnosis, and `siliquant#85` Proposal 4 Zero-Favoring Decoding

**Epistemic Status:** `measured_live_model_weights: true` | **Exact Pillar:** `exact` | **Empirical Pillar:** `sampled` | **Prior Track Links:** `anchor` (`Track S59`, `Track S60`, `Track S63`, `siliquant#85`)
**Hardware Profile:** `ux582hs-rtx3080-8gb` (`NVIDIA GeForce RTX 3080 Laptop GPU`, `GA104M` `sm_86`, `8 GiB` GDDR6, NVIDIA driver `615.78.08`, Linux `7.2.15-1-omarchy`, `x86_64`, Python `3.14.8`, AC online)
**Determinism Zone:** Ring-0 `mlos` **Zone B** (`clean_hardware_run: true`, `anomalies: []`, `exit_code: 0`, `graphics_processes: []`, `compute_processes: []`, `linux_kmd_baseline_vram_mib: 2`)
**Causal Pre-Registration Order:**
- `s67_preregistration.json` locked (`0o600`): **`2026-11-24T15:18:26Z`** (`preregistration_sha256`: `f0b6c25beccd36aa8571d15e9eb9d6a655b3513e83d8f4581cc99759e114d179`)
- Live Zone-B hardware execution started: **`2026-11-24T15:20:36Z`** (strictly `2 min 10 s` after pre-registration freeze)
- Live Zone-B hardware execution ended: **`2026-11-24T16:07:32Z`** (`duration_ms: 2815517.094`, inner harness `elapsed_seconds: 2805.366 s`, CPU RAPL `191,037.879 J`)

---

## 1. Executive Summary & Pre-Registered CEER Cycle-2 Scoreboard

Track S67 executes three pre-registered Cycle-2 experimental tests directly on **`4` live offline-cached BF16 LLMs** (`HuggingFaceTB/SmolLM2-360M`, `Qwen/Qwen2.5-0.5B`, `Qwen/Qwen2.5-1.5B`, `Qwen/Qwen2.5-3B`) spanning **`840` linear weight matrices** and **`4,757,127,168` (`4.757B`) block-quantized weights** across **`29` live quantization arms** at exact `9/2 = 4.5` bits per weight (plus matched-MSE controls exact to `<= 1.667e-6 dB`, `0` syndrome errors, and `0` non-block tensor diffs across all `29/29` arms), scored on identical pinned `4,088`-token held-out WikiText-2 and CPython stdlib splits (`seq_len=512`, `n_eval_chunks=8`).

| Pre-Registered Question | CEER Cycle-2 Verdict | Core Live Result |
|---|---|---|
| **`P_S67_1`**: Live test of `Track S59`'s $\rho_{\text{axis}}(A_{16}, b) = (1, 2, 1, 2, 4)$ prediction via **`A16_BlockH4`** ($\rho=1$, exact dyadic $1/2$) & **`A16_BlockH8`** ($\rho=2$) | **`PARTIAL_PYTHON_WINNER_AND_DYADIC_SIGNATURE`** | **Confirmed:** (1) Exact dyadic zero-retention signature `z0` (`A16_axis 68–93% > BlockH4 36–49% ≈ H16 36–49% >> BlockH2 6.7–8.3% >> BlockH8 0.02–0.78%`) appears verbatim in live weights across all 4 models. (2) **`A16_BlockH4` wins #1 `A_16` Python KL on `0.5B` (`0.045183`) and #1 OVERALL Python KL on `3B` (`0.035620`, matched `0.035523`) — crushing `E8_H8` (`0.036741`, `-3.05%`), `A16_axis` (`0.038578`, `-7.67%`), `A16_H16` (`0.038760`, `-8.10%`), `A16_BlockH8` (`0.039466`, `-9.74%`), and `A16_BlockH2` (`0.040346`, `-11.71%`)!** (3) `A16_BlockH4` beats `A16_BlockH8` on `3/4` models (`SmolLM2`, `0.5B`, `3B`) and beats `A16_axis` on `334/616 = 54.2%` of Qwen2.5 layers. **Falsified / Nuanced:** On WikiText-2, `+2.30..+2.37 dB` small-weight mixing error ($E_{05}$) keeps `A16_axis` ahead of `BlockH4` at `SmolLM2` (`0.065173 vs 0.082002`), `0.5B` (`0.052525 vs 0.056063`), and `3B` (`0.046686 vs 0.047942`), while `E8_H8` (`0.045724`) retains #1 on `3B` WikiText-2 and `BlockH8` (`0.045554 / 0.029673`) edges `BlockH4` (`0.046550 / 0.030045`) on `1.5B`. |
| **`P_S67_2`**: Full `616`-layer (`168+196+252`) structural diagnosis of the `0.5B -> 1.5B -> 3.0B` re-entrant oscillation | **`CONFIRMED_ZERO_CAL_WEIGHT_AND_RAND_DIAG_SEPARATION`** | **Zero-calibration weight-only** statistics hit a **unique global minimum at `Qwen2.5-1.5B`**: `mean_col_outlier_ratio` (`2.6091 -> 2.3373 MIN -> 2.5196`), `mean_small_weight_frac_05` (`0.051059 -> 0.047386 MIN -> 0.059151`), and `down_proj` excess kurtosis (`1.7976 -> 1.2022 MIN -> 2.9241`). Two structural shocks complete the mechanism: (i) at `1.5B`, `down_proj` carries **`76.17%`** of total quadratic output distortion (vs `23.90%` at `0.5B`, `37.44%` at `3B`) with `15/28` layers in Regime II (`BlockH4 < axis`); (ii) at `3B`, `down_proj` kurtosis jumps `2.43×` back to Regime I (`24/36` favoring `axis`) while `gate_proj`/`up_proj` **random-token diagonal concentration explodes `6.78×`** (`25.14 -> 26.81 -> 181.65`), lifting `gate+up` distortion share `4.05×` (`9.36% -> 37.93%`). Block-`{2,4,16}` DC coherence ratios remain flat (`~0.97–1.05`). |
| **`P_S67_3`**: Live test of `siliquant#85` Proposal 4 zero-favoring weighted trellis decoding ($\lambda \in \{0.5, 2.0\}$ on $|w| < 0.05 \cdot \operatorname{rms}(W)$) | **`CONFIRMED_EXACT_OPPOSITE_SIGN`** | Small-weight error $E_{05}$ drops monotonically on **both** models (`SmolLM2`: `2.6824 -> 2.2054 -> 2.0733 dB`, `z0: 93.11% -> 95.09%`; `Qwen2.5-3B`: `1.9239 -> 1.4572 -> 1.3220 dB`, `z0: 68.05% -> 69.31%`) at identical `+0.0021..+0.0046 dB` pooled MSE cost. End-to-end KL has **exact opposite sign**: **`SmolLM2-360M` strictly improves** (`WikiKL 0.065173 -> 0.064840 -> 0.064599`, `-0.88%`; `PyKL 0.056725 -> 0.055975` at $\lambda=0.5$, `-1.32%`), whereas **`Qwen2.5-3B` strictly worsens** (`WikiKL 0.046686 -> 0.047430 / 0.047237`; `PyKL 0.038578 -> 0.038670 / 0.038767`) — directly confirming the 3-model regime triangle live. |

---

## 2. Exact Pillar (`Pillar 1`) & Offline Artifact Pins

### 2.1 Exact $A_{16}$ Block-Hadamard Arithmetic (`Track S59` Basis)
For block-diagonal normalized Hadamard preconditioning $Q_b = I_{16/b} \otimes \frac{1}{\sqrt{b}} H_b$ applied to the $16$-dimensional Barnes-Wall lattice $A_{16}$ ($R = 9/2$ bpw):
- Axis coarsening factor $\rho_{\text{axis}}(A_{16}, b)$ for $b \in \{1, 2, 4, 8, 16\}$ is **$(1, 2, 1, 2, 4)$**.
- Entries of $Q_b$ lie in the dyadic ring $\mathbb{Z}[1/2]$ iff $b \in \{1, 4, 16\}$ (`1, 1/2, 1/4`), whereas $b \in \{2, 8\}$ introduce irrational factors ($1/\sqrt{2}, 1/\sqrt{8}$).
- Thus **`A16_BlockH4`** ($b = 4$) uniquely restores both $\rho_{\text{axis}} = 1$ AND exact dyadic arithmetic $\frac{1}{2} H_4 \in \mathbb{Z}[1/2]^{4 \times 4}$ while keeping cross-block mixing confined to 4-coordinate blocks (non-isometry of $A_{16}$).

### 2.2 Pinned Models & Evaluation Corpora
All four models and both held-out corpora were loaded strictly offline (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`) and verified by SHA-256 inside `rtx3080_s67_audit.py --verify-receipts .`:

| Artifact | Layers / Matrices | Block Weights | Group Size | Pinned Revision / Digest (SHA-256) |
|---|---:|---:|---:|---|
| `HuggingFaceTB/SmolLM2-360M` | `32L` / `224` | `314,572,800` | `32` | rev `f8027fd0eaeea54caa13c31d31b9fdc459c38b49`, safetensors `7aaff6661428bed033abba9522bec81938678642cca3181fe752b6ca9e1e540f` |
| `Qwen/Qwen2.5-0.5B` | `24L` / `168` | `357,826,560` | `128` | rev `060db6499f32faf8b98477b0a26969ef7d8b9987`, safetensors `88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342` |
| `Qwen/Qwen2.5-1.5B` | `28L` / `196` | `1,310,195,712` | `128` | rev `8faed761d45a263340a0528343f099c05c9a4323`, safetensors `a961db72e75d52b18e6b0c9d379e51a26973b233385e0e127fdda7d648aec796` |
| `Qwen/Qwen2.5-3B` | `36L` / `252` | `2,774,532,096` | `128` | rev `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b`, shard1 `f9558df91d3b89b4826e4db37439edb52f1d62a4fd602685013e7ca6b9f60f8f`, shard2 `51410930d5cf19a998fdb17ef0c46e4d9ace72c97a975a3331395a8a500f5edb` |
| **Combined 4-Model Total** | **`120L` / `840`** | **`4,757,127,168`** | — | **29 live arms**, `0` syndrome errors, `0` non-block diffs |
| WikiText-2 (`Salesforce/wikitext`) | `4,088` tokens (`8×512`) | — | — | rev `b08601e04326c79dfdd32d625aee71d232d685c3`, test `430983e5a711fbc4c6576edffecacf47ff23d58f62f3923d2386d798e0a794c9` |
| CPython stdlib (6 modules) | `4,088` tokens (`8×512`) | — | — | `fractions.py`, `difflib.py`, `decimal.py`, `statistics.py`, `argparse.py`, `json/decoder.py`, `fe8ff0f94937ebd41e55f9880dd19bbaa4bbeadec791a6544bbafcb9cd2ee609` |

---

## 3. Complete Live 29-Arm Empirical Tables (`P_S67_1` & `P_S67_3`)

All arms passed `syndrome_errors = 0` and `non_block_tensors_differ = 0`. Matched-MSE arms scale each group to match `A16_axis` exact pooled MSE (`max_matched_mse_db_diff <= 1.667e-6 dB`, well inside the pre-registered `1e-5 dB` tolerance).

### 3.1 `HuggingFaceTB/SmolLM2-360M` (`224` matrices, `314,572,800` wts, `gs=32`, BF16 ref Wiki PPL `16.3948` / Py PPL `5.0450`)

| Arm | Weight Err (dB) | Small-Wt $E_{05}$ (dB) | Exact Zero `z0` (%) | WikiText-2 KL ± SE | Wiki PPL | Python KL ± SE | Python PPL |
|---|---:|---:|---:|---:|---:|---:|---:|
| `A16_axis` | `-25.0281` | `2.6824` | `93.1058%` | `0.065173 ± 0.003671` | `17.2522` | `0.056725 ± 0.003494` | `5.3303` |
| **`A16_zero_lam05`** | `-25.0260` | `2.2054` | `94.4797%` | `0.064840 ± 0.003590` | `17.3309` | **`0.055975 ± 0.003565`** | **`5.3271`** |
| **`A16_zero_lam20`** | `-25.0238` | **`2.0733`** | **`95.0923%`** | **`0.064599 ± 0.003668`** | `17.3189` | `0.056745 ± 0.003629` | `5.3362` |
| `A16_BlockH4` | `-25.0279` | `5.2266` | `48.7226%` | `0.082002 ± 0.005041` | `17.7082` | `0.066762 ± 0.004461` | `5.3763` |
| `A16_BlockH8` | `-25.0277` | `5.1399` | `0.0206%` | `0.085384 ± 0.005583` | `17.8215` | `0.070554 ± 0.005385` | `5.3494` |

### 3.2 `Qwen/Qwen2.5-0.5B` (`168` matrices, `357,826,560` wts, `gs=128`, BF16 ref Wiki PPL `17.9505` / Py PPL `4.8606`)

| Arm | Weight Err (dB) | Small-Wt $E_{05}$ (dB) | Exact Zero `z0` (%) | WikiText-2 KL ± SE | Wiki PPL | Python KL ± SE | Python PPL |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`A16_axis`** | `-25.5529` | **`2.3356`** | **`91.9411%`** | **`0.052525 ± 0.003470`** | **`18.4858`** | `0.045602 ± 0.002930` | `5.0999` |
| `A16_BlockH2` | `-25.5527` | `4.5462` | `6.7119%` | `0.055206 ± 0.003583` | `18.6558` | `0.045392 ± 0.003303` | `5.0606` |
| **`A16_BlockH4`** | `-25.5532` | `4.6357` | `48.2160%` | `0.056063 ± 0.003911` | `18.5926` | **`0.045183 ± 0.003403`** | `5.0712` |
| `A16_BlockH4_matched` | `-25.5529` | `4.6359` | `0.0041%` | `0.056202 ± 0.003922` | `18.6224` | `0.045224 ± 0.003409` | `5.0700` |
| `A16_BlockH8` | `-25.5532` | `4.5508` | `0.2149%` | `0.057186 ± 0.004584` | `18.9092` | `0.049353 ± 0.004597` | `5.0631` |
| `A16_BlockH8_matched` | `-25.5529` | `4.5511` | `0.0022%` | `0.057211 ± 0.004574` | `18.9047` | `0.049408 ± 0.004587` | `5.0704` |
| `A16_H16` | `-25.5527` | `4.6368` | `48.3378%` | `0.057614 ± 0.004075` | `18.5909` | `0.047449 ± 0.003639` | `5.1166` |

### 3.3 `Qwen/Qwen2.5-1.5B` (`196` matrices, `1,310,195,712` wts, `gs=128`, BF16 ref Wiki PPL `12.6180` / Py PPL `3.7395`)

| Arm | Weight Err (dB) | Small-Wt $E_{05}$ (dB) | Exact Zero `z0` (%) | WikiText-2 KL ± SE | Wiki PPL | Python KL ± SE | Python PPL |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`A16_BlockH8`** | `-25.5531` | `4.8093` | `0.1757%` | **`0.045554 ± 0.003729`** | `12.9923` | `0.029673 ± 0.002622` | `3.8380` |
| **`A16_BlockH8_matched`** | `-25.5532` | `4.8092` | `0.0080%` | `0.045661 ± 0.003738` | `13.0028` | **`0.029559 ± 0.002596`** | `3.8396` |
| `A16_BlockH2` | `-25.5532` | `4.8029` | `6.6714%` | `0.045648 ± 0.003010` | **`12.9145`** | `0.030310 ± 0.002611` | `3.8693` |
| `A16_BlockH4` | `-25.5532` | `4.8886` | `49.1237%` | `0.046550 ± 0.003296` | `13.1362` | `0.030045 ± 0.002820` | `3.8453` |
| `A16_BlockH4_matched` | `-25.5532` | `4.8885` | `0.0034%` | `0.046497 ± 0.003288` | `13.1306` | `0.030181 ± 0.002838` | `3.8454` |
| `A16_H16` | `-25.5531` | `4.8906` | `49.2321%` | `0.048376 ± 0.003659` | `13.0069` | `0.030466 ± 0.003148` | **`3.8239`** |
| `A16_axis` | `-25.5532` | **`2.5493`** | **`93.4366%`** | `0.050062 ± 0.004083` | `13.0109` | `0.031973 ± 0.003214` | `3.8456` |

### 3.4 `Qwen/Qwen2.5-3B` (`252` matrices, `2,774,532,096` wts, `gs=128`, BF16 ref Wiki PPL `10.9702` / Py PPL `3.1119`)

| Arm | Weight Err (dB) | Small-Wt $E_{05}$ (dB) | Exact Zero `z0` (%) | WikiText-2 KL ± SE | Wiki PPL | Python KL ± SE | Python PPL |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`E8_H8`** | **`-25.5850`** | `3.0425` | `0.9878%` | **`0.045724 ± 0.002077`** | `11.3153` | `0.036741 ± 0.002207` | `3.2366` |
| `A16_axis` | `-25.5532` | `1.9239` | `68.0512%` | `0.046686 ± 0.002542` | `11.2583` | `0.038578 ± 0.002440` | `3.2182` |
| `A16_zero_lam05` | `-25.5511` | `1.4572` | `68.9271%` | `0.047430 ± 0.002515` | `11.2375` | `0.038670 ± 0.002455` | `3.2171` |
| `A16_zero_lam20` | `-25.5486` | **`1.3220`** | **`69.3103%`** | `0.047237 ± 0.002549` | `11.2401` | `0.038767 ± 0.002586` | **`3.2148`** |
| `A16_H16` | `-25.5536` | `4.2998` | `36.0881%` | `0.047631 ± 0.002316` | **`11.0637`** | `0.038760 ± 0.002482` | `3.2463` |
| **`A16_BlockH4`** | `-25.5534` | `4.2945` | `36.1462%` | `0.047942 ± 0.002618` | `11.3595` | `0.035620 ± 0.002198` | `3.2230` |
| **`A16_BlockH4_matched`** | `-25.5532` | `4.2961` | `0.0056%` | `0.047956 ± 0.002613` | `11.3730` | **`0.035523 ± 0.002210`** | `3.2231` |
| `A16_BlockH8` | `-25.5532` | `4.2096` | `0.7826%` | `0.051088 ± 0.002747` | `11.2374` | `0.039466 ± 0.002616` | `3.2287` |
| `A16_BlockH8_matched` | `-25.5532` | `4.2103` | `0.0485%` | `0.050100 ± 0.002555` | `11.2196` | `0.039449 ± 0.002599` | `3.2283` |
| `A16_BlockH2` | `-25.5540` | `4.1989` | `8.2926%` | `0.051623 ± 0.003021` | `11.3665` | `0.040346 ± 0.002697` | `3.2494` |

*(Notice the matched-MSE parity check: rescaling `A16_BlockH4` by `~1e-5` collapses `z0` from `36.1%–49.1%` to `0.003%–0.006%` without changing $E_{05}$ or KL by more than `0.00014`, proving that the gain of `BlockH4` on `3B` Python and `1.5B` comes from the 4-block Hadamard geometry and not from floating-point zero exactness alone.)*

---

## 4. `P_S67_2` — 616-Layer Structural Diagnosis of the Qwen2.5 Re-Entrant Oscillation

Across all **`616` Qwen2.5 linear matrices** (`168` at `0.5B`, `196` at `1.5B`, `252` at `3.0B`), we recorded zero-calibration weight geometry, zero-calibration random-token activation Hessian geometry (`rand_*`), real-token calibration Hessian geometry (`eval_*`), and per-layer quadratic output distortion shares under `A16_axis`, `A16_BlockH2`, `A16_BlockH4`, `A16_BlockH8`, and `A16_H16`.

### 4.1 Model-Wide Order Parameters Across `0.5B -> 1.5B -> 3.0B`

| Order Parameter (616 Layers Total) | `Qwen2.5-0.5B` (`168L`) | `Qwen2.5-1.5B` (`196L`) | `Qwen2.5-3B` (`252L`) | Pattern |
|---|---:|---:|---:|---|
| **Mean Col-Outlier Ratio** *(zero-cal weight)* | `2.609139` | **`2.337262` (MIN)** | `2.519638` | **V-shaped minimum at `1.5B`** |
| **Mean Small-Wt Fraction `<0.05 rms`** *(zero-cal weight)* | `0.051059` | **`0.047386` (MIN)** | `0.059151` | **V-shaped minimum at `1.5B`** |
| **`down_proj` Excess Kurtosis** *(zero-cal weight)* | `1.797597` | **`1.202150` (MIN)** | `2.924093` | **V-shaped minimum at `1.5B` (`2.43×` jump to `3B`)** |
| **`down_proj` Quad Distortion Share** | `23.90%` | **`76.17%` (PEAK)** | `37.44%` | **Inverted-V peak at `1.5B` (`3.19×` over `0.5B`)** |
| **`down_proj` Layers `BlockH4 < axis`** | `11/24` (`45.8%`) | **`15/28` (`53.6%`)** | `12/36` (`33.3%`) | **Regime II ONLY at `1.5B`; `24/36` Regime I at `3B`** |
| **`gate/up_proj` Rand Diag Conc** *(zero-cal random tokens)* | `25.138076` | `26.809261` | **`181.654144`** | **`6.78×` explosion at `3B` (`181654144/26809261`)** |
| **`gate/up_proj` Real Diag Conc** | `33.150180` | `40.019011` | **`158.597740`** | **`3.96×` explosion at `3B`** |
| **`gate + up` Quad Distortion Share** | `23.02%` | `9.36%` | **`37.93%`** | **`4.05×` jump from `1.5B` to `3B`** |
| All-Layer `BlockH4 < axis` Count | `95/168` (`56.5%`) | `112/196` (`57.1%`) | `127/252` (`50.4%`) | **`334/616 = 54.2%`** overall |
| All-Layer `BlockH4 < BlockH8` Count | `95/168` (`56.5%`) | `106/196` (`54.1%`) | `121/252` (`48.0%`) | **`322/616 = 52.3%`** overall |
| All-Layer `BlockH4 < H16` Count | `87/168` (`51.8%`) | `109/196` (`55.6%`) | `114/252` (`45.2%`) | **`310/616 = 50.3%`** overall |
| Mean Eval DC Ratio (`b=2 / b=4 / b=16`) | `1.001 / 1.010 / 1.002` | `0.998 / 0.994 / 0.998` | `0.999 / 1.004 / 1.011` | Flat (`~1.00 ± 0.01`) — NOT DC coherence |

### 4.2 Per-Projection Breakdown Across All Three Qwen2.5 Scales

| Proj | `0.5B` Kurt / `sm05` / Share / `rDiag` / `bh4<ax` | `1.5B` Kurt / `sm05` / Share / `rDiag` / `bh4<ax` | `3.0B` Kurt / `sm05` / Share / `rDiag` / `bh4<ax` |
|---|---|---|---|
| `down_proj` | `1.80` / `0.0455` / `23.90%` / `474.9` / `11/24` | **`1.20`** / **`0.0432`** / **`76.17%`** / `562.9` / **`15/28`** | **`2.92`** / `0.0592` / `37.44%` / `731.4` / `12/36` |
| `gate_proj` | `1.94` / `0.0445` / `12.60%` / `25.1` / `14/24` | `1.09` / `0.0429` / `5.10%` / `26.8` / `18/28` | `2.14` / **`0.0816`** / **`20.98%`** / **`181.7`** / `25/36` |
| `up_proj` | `1.82` / `0.0430` / `10.42%` / `25.1` / `12/24` | `0.65` / `0.0425` / `4.25%` / `26.8` / `14/28` | `2.70` / **`0.0791`** / **`16.95%`** / **`181.7`** / `15/36` |
| `q_proj` | `5.43` / `0.0624` / `16.82%` / `21.9` / `12/24` | `8.23` / `0.0547` / `4.29%` / `41.1` / `16/28` | `1.98` / `0.0508` / `6.97%` / `58.8` / `16/36` |
| `k_proj` | `3.19` / `0.0564` / `19.28%` / `21.9` / `16/24` | `2.16` / `0.0513` / `4.01%` / `41.1` / `16/28` | `2.00` / `0.0514` / `6.99%` / `58.8` / `17/36` |
| `v_proj` | `1.93` / `0.0524` / `16.03%` / `21.9` / `14/24` | `1.40` / `0.0496` / `5.15%` / `41.1` / `14/28` | `0.92` / `0.0470` / `8.91%` / `58.8` / `21/36` |
| `o_proj` | `6.04` / `0.0532` / `0.96%` / `20.5` / `16/24` | `2.80` / `0.0474` / `1.02%` / `18.6` / `19/28` | `1.80` / `0.0449` / `1.76%` / `12.2` / `21/36` |

**Mechanistic Synthesis:** `Qwen2.5-1.5B` sits in an unusually Gaussian weight basin (`down_proj` kurtosis `1.20`, `up_proj` `0.65`, `gate_proj` `1.09`) where `down_proj` dominates output distortion (`76.17%`), allowing block Hadamard preconditioning (`BlockH8` / `BlockH2` / `BlockH4`) to outperform `A16_axis` across both corpora. At `Qwen2.5-3B`, `down_proj` kurtosis surges `2.43×` (`2.92`) back into Regime I (`mean_bh4_over_axis_quad_err = 1.014969`, `24/36` favoring `axis`), AND `gate/up_proj` small-weight sparsity (`0.0816 / 0.0791`) + random-token diagonal outlier concentration (`181.65`) jump sharply — swinging WikiText-2 back to `E8_H8 < A16_axis < A16_H16 < A16_BlockH4`, while `A16_BlockH4`'s `25/36` `gate_proj` win (`0.987837`) and `21/36` `o_proj`/`v_proj` wins propel it to **#1 overall on Python code (`0.035620`, beating `E8_H8` by `-3.05%`)**.

---

## 5. `P_S67_3` — Live Confirmation of `siliquant#85` Proposal 4 Opposite-Sign Behavior

Applying per-coordinate trellis weighting $1 + \lambda$ to small weights ($|w| < 0.05 \cdot \operatorname{rms}(W)$) cleanly shifts error out of the small-weight tail at a tiny, uniform pooled MSE cost (`+0.0021 dB` at $\lambda = 0.5$, `+0.0043..+0.0046 dB` at $\lambda = 2.0$), with exact opposite KL sign at the two vertices of the regime triangle:

| Model | Regime Vertex | $\lambda=0$ (`A16_axis`) $E_{05}$ / `z0` / WikiKL / PyKL | $\lambda=0.5$ $E_{05}$ / `z0` / WikiKL / PyKL | $\lambda=2.0$ $E_{05}$ / `z0` / WikiKL / PyKL | Verdict |
|---|---|---|---|---|---|
| **`SmolLM2-360M`** | Zero-retention vertex | `2.6824` / `93.11%` / `0.065173` / `0.056725` | `2.2054` / `94.48%` / `0.064840` (**`-0.51%`**) / **`0.055975`** (**`-1.32%`**) | `2.0733` / `95.09%` / **`0.064599`** (**`-0.88%`**) / `0.056745` (`+0.03%`) | **HELPS KL** |
| **`Qwen2.5-3B`** | MSE-dominated vertex | `1.9239` / `68.05%` / **`0.046686`** / **`0.038578`** | `1.4572` / `68.93%` / `0.047430` (`+1.59%`) / `0.038670` (`+0.24%`) | `1.3220` / `69.31%` / `0.047237` (`+1.18%`) / `0.038767` (`+0.49%`) | **HURTS KL** |

---

## 6. Verification & Aletheia R1–R5 Receipts

All verification gates were run after the final edit to the oracle, audit, and test scripts (`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`):

- **Exact 6-Decimal Token Parity (`rtx3080_s67_oracle.py` vs `rtx3080_s67_empirical_results.json`):** `TOKEN_PARITY_MISMATCHES: []` (every rational token matches the live empirical JSON verbatim in exact $\mathbb{Q}$).
- **Certificate & Receipt Replay:** `rtx3080_s67_audit.py --check rtx3080_s67_certificate.json` -> `{"certificate_sha256":"882051fa65f809a2e4c99b1d544f4b3b5c62a43cda40c2067f4b9ed74e59ebca","file_sha256":"c363b80e151a1a4085599d673e5d21df89a15ec2a8090c2485c85265f43d2107","full_replay":true,"valid":true}`; `--verify-receipts .` -> `all_verified: true` (`11/11` receipt checks `true`, including strict causal timestamp order `2026-11-24T15:18:26Z < 2026-11-24T15:20:36Z`).
- **4-Mode `-I` Parity + `umask 077`:** Verified identical standalone JSON bytes vs `0o600` disk certificate (`c363b80e...`), `--check`, `--verify-receipts .`, and `5/5` R5 tests across `-I -B`, `-I -O -B`, `-I -B -W error`, `PYTHONHASHSEED=42 -B`, and `umask 077`.
- **Mutation Testing (`aletheia-fast-mutate --guards require_ok --workers 4 --timeout 180 --max-mutants 0`):**
  - `rtx3080_s67_oracle.py`: **`2/2` (`100.0%`, `survived=0`, `timeout=0`, `truncated=False`)**
  - `rtx3080_s67_audit.py`: **`71/71` (`100.0%`, `survived=0`, `timeout=0`, `truncated=False`)**
  - Combined: **`73/73` (`100.0%`)**
- **Whole-File Exact DFS AST `BoolOp` Mutation (`OP_SWAP` + `SUBCOND_DEL`):**
  - `rtx3080_s67_oracle.py`: **`139/139` (`survs=[]`)**
  - `rtx3080_s67_audit.py`: **`46/46` (`survs=[]`)**
  - Combined: **`185/185` (`100.0%`)**
- **4-Surface v2.1 Zone-B Fuzzing (`aletheia-v21-upgrades`):**
  - `CLI`: **`25/25`**, `FR1`: **`19/19`**, `FR0`: **`19/19`**, `SEC`: **`19/19`** -> **`82/82 PASS`** (`0` bypasses, `0` crashes, `findings=[]`).
- **Dual `aletheia audit-check . --strict-ancestors --safety-scan --scripts rtx3080_s67_audit.py test_rtx3080_s67_r5.py`:** `PASS` on both `aletheia-v21-upgrades` and `aletheia-fast-mutate` (`0` findings, `0` bare `assert`s, `0` AST floats/`float()` in oracle/audit/test, `0` `.pyc`, `0` ELF, `0` HF model weights committed).

---

## 7. Frozen Artifact Manifest (8 Frozen Files)

| File | Lines | Bytes | Mode | SHA-256 |
|---|---:|---:|---|---|
| `s67_preregistration.json` | `1` | `3,193` | `0o600` | `faea457de9801b7fa74f9c38763f96b64460465e4175825a2915fa1a631b05f4` |
| `rtx3080_s67_blockh4_ceer_eval.py` | `887` | `37,424` | `0o644` | `3820e1f616ba505b107bd2fea81289e1b85d8228f383c7a937ce7e5f4e23200e` |
| `rtx3080_s67_empirical_results.json` | `1,448` | `50,810` | `0o644` | `31ee0f744ef674e5e1bc5ca277a4ef47187df155c2f74186f4c461765318db48` |
| `mlos_zone_b_receipt.json` | `91` | `2,628` | `0o600` | `60417f256a3f2166e41fac7138cd901e87f528b44e483d844b5daa4da1e429c3` |
| `rtx3080_s67_oracle.py` | `611` | `29,167` | `0o644` | `6e49c28b1b187a2a7b4a88f31227f42c282430734f1495fc1d4dab1b2d29180b` |
| `rtx3080_s67_audit.py` | `532` | `21,174` | `0o644` | `62393bd5f6b9a9586668c97c763d659c8f22b966b45b0662129a98f53e34a703` |
| `rtx3080_s67_certificate.json` | `1` | `3,999` | `0o600` | `c363b80e151a1a4085599d673e5d21df89a15ec2a8090c2485c85265f43d2107` |
| `test_rtx3080_s67_r5.py` | `865` | `47,324` | `0o644` | `fffa1506398be6da37fcdde69ce150c17e8b0c1ddddca009557b78112fec89c7` |
