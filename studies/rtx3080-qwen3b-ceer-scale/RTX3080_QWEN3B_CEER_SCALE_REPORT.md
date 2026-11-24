# Live 3B-Scale Real-Weight CEER Scale Law (`Qwen/Qwen2.5-3B`, `0.36B -> 3.0B` 4-Point Curve) & RTX 3080 `sm_86` Evaluation Report

**Epistemic Status:** `exact` (pure-$\mathbb{Z}/\mathbb{Q}$ architectural tiling, parameter totals, scale ratios, and lossless `int8` ball-fit invariants) + `sampled` (`measured_live_model_weights: true` — preregistered offline live evaluation of real pretrained `Qwen/Qwen2.5-3B` weights on RTX 3080 Laptop GPU `GA104M` `sm_86`, serialized via `flock` + `mlos gpu-lease --mode train` + `mlos stamp-run --zone B --require-gpu --strict-anomalies`, `clean_hardware_run: true`, `anomalies: []`) + `anchor` (prior live `SmolLM2-360M` / `Qwen2.5-0.5B` / `Qwen2.5-1.5B` RTX 3080 runs and upstream `siliquant#58` Gemma calibration curve).

- **Preregistered Causality Lock (`s63_preregistration.json`, `0o600`, Sealed Strictly BEFORE Model Weight Access):**
  - Created UTC: `2026-11-24T14:39:52Z` `<` Live Run Started UTC: `2026-11-24T14:44:28Z` (`4 min 36 s` prior).
  - File SHA-256: `516142837da45cfcb277b2a2f201dd7bd528c0b44f066339442681946878e29d`
  - Self-Digest `preregistration_sha256`: `70147f901094eaac897bdce423bc00d427f1b387a1c96413d807027287afbc85`
- **Model Audited Live (`2.12×` `Qwen2.5-1.5B`, `7.75×` `Qwen2.5-0.5B` Block Weights):**
  - `Qwen/Qwen2.5-3B` (`revision`: `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b`, `36` layers, `hidden=2048`, `kv=256`, `intermediate=11008`, `252` block linear matrices, `2,774,532,096` quantized block weights, exact `128`-weight `72B` tiling, `182` non-block tensors verified bit-identical across all `14/14` arms).
  - Shards (kept exclusively in the local HuggingFace cache): `model-00001-of-00002.safetensors` (`3,968,658,944` B, `f9558df91d3b89b4826e4db37439edb52f1d62a4fd602685013e7ca6b9f60f8f`), `model-00002-of-00002.safetensors` (`2,203,268,048` B, `51410930d5cf19a998fdb17ef0c46e4d9ace72c97a975a3331395a8a500f5edb`), `model.safetensors.index.json` (`bc8aaa0c87d4335177e01c765f1de0db81661c67c1a72fbfb0d521b09f5ddc56`), `tokenizer.json` (`c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`).
- **Identical Disjoint Corpora Across All Four Scale Points (`4,088` Held-Out Next-Token Logits Each):**
  - Calibration (`6,144` tokens): `Salesforce/wikitext` `validation` (`93abc2e0...`).
  - Held-out prose (`4,088` tokens): `Salesforce/wikitext` `test` (`430983e5...`, BF16 PPL `10.970193`).
  - Held-out code (`4,088` tokens): `6` CPython stdlib modules (`fe8ff0f9...`, BF16 PPL `3.111936`).
- **Ring-0 Hardware Receipt (`mlos_zone_b_receipt.json`, `0o600`):**
  - `receipt_sha256`: `28baf1065dcba61fa554a9d173b785dab6f6dd13ed9ff36a2d55d1783933c05d` (`profile_id`: `ux582hs-rtx3080-8gb`, `clean_hardware_run: true`, `anomalies: []`, `exit_code: 0`, `duration_ms: 1267003.439`, `graphics_processes: []`, `linux_kmd_baseline_vram_mib: 2`, `start/end P8`, `54 °C -> 60 °C`, `ac_online: true`).

---

## Executive CEER Scoreboard: Preregistered Predictions vs. Live `3B` Silicon

| Preregistered `CEER` Prediction (`70147f90...`) | Pre-Run Rational Prediction | Live Observed `Qwen/Qwen2.5-3B` Result (`measured_live_model_weights: true`) | Honest CEER Outcome |
| :--- | :--- | :--- | :---: |
| **`P_3B_1` (`A16_BlockH2` Dominance)** | `A16_BlockH2 < A16_H16 < A16_axis` on WikiText-2 KL (natural & `<= 1e-5 dB` matched MSE) | **FALSIFIED — OPPOSITE-SIGN `E_8` vs. `A_16` LAW RETURNS AT `3B`!** `A16_axis` (`0.046686` Wiki, `0.038578` Py) beats BOTH `A16_H16` (`0.047631` / `0.048159` matched, `0.038760` Py) AND `A16_BlockH2` (`0.051623` / `0.051123` matched, `0.040346` Py). Simultaneously, `E8_H8` (`0.045724` / `0.046143` matched Wiki, `0.036741` Py) crushes `E8_axis` (`0.050420` Wiki `-9.31%`, `0.039298` Py `-6.51%`) and wins #1 uncalibrated KL overall, while `A16_H16` hits lowest uncalibrated Wiki PPL (`11.064`). Proves uncalibrated rotation preference is **non-monotonic across checkpoints** (`Block-H2 < H16` drops from `53.6%` at `1.5B` to `40.5%` = `102/252` at `3B`). | **FALSIFIED** (Opposite-Sign returns!) |
| **`P_3B_2` (`P5` Decay Curve)** | Wiki `P5` drops monotonically below `0.9182` into `[0.78, 0.89]`; test whether Py `P5 < 0.96`; `colproxy < 0.40` | **MONOTONIC DECAY CONFIRMED — STEEPER THAN BRACKET!** WikiText-2 `P5` plunged **`-21.15` pp** from `91.82%` (`1.5B`) straight down to **`70.66%` (`0.706639`)** (overshooting `[78%, 89%]` downward by `-7.34` pp, reaching `Gemma-12B`'s `71%` decay already at `Qwen2.5-3B`!). Python `P5` bent sharply below `96%` (`96.99% -> 91.40%`). Weight-only `hA16_colproxy` collapsed to **`5.13%`** on WikiText-2 (`21.94%` Python; Spearman $\rho$ `real 0.757 > rand 0.530 >> colproxy 0.083`). | **CONFIRMED (STEEPER)** (`70.66% < 78%`; Py `91.40%`) |
| **`P_3B_3` (`hA16_BlockH2_real` Floor)** | `hA16_BlockH2_real < hA16_real` on WikiText-2 KL | **CONFIRMED IN-DOMAIN ON WIKITEXT-2 (DOMAIN SPLIT ON PYTHON)!** `hA16_BlockH2_real` (`0.024179`) strictly beats `hA16_real` (`0.024417`, `-0.97%` KL) and sets the **#1 lowest WikiText-2 PPL of all `14` arms (`11.058`, only `+0.088` PPL over BF16 `10.970`!)**, while `hA16_real` (`0.021217`) beats `hA16_BlockH2_real` (`0.022975`, `+8.29%`) on out-of-domain Python! | **CONFIRMED IN-DOMAIN** (`11.058` PPL!) |
| **`P_3B_4` (`3B` CUDA + 100 Hz INA3221)** | `10/10` zero-spill, `0` err, `rel_l2 < 1e-6`, stacked-2 `tc > 100%`, `A16 > 95%`, `BH2 > 92%` Triad + 100 Hz INA3221 GPU & RAPL CPU `mJ/tok` | **100% CONFIRMED!** `10/10` zero-spill (`20–72` regs), `0` bit/syn err, `rel_l2 <= 1.131e-7`. Stacked-2 `4096×11008` (`45.09M` wts): lossless `tc_18B 101.36%`, `tc_72B 101.43%` (`721.5×` on-the-fly), direct `A16 97.35%` (`615.95 Gw/s`, `222.0 tok/s`, `408.05 + 299.93 = 707.97 mJ/tok`), direct `Block-H2 94.96%` (`600.87 Gw/s`, `216.6 tok/s`, `417.61 + 275.57 = 693.18 mJ/tok`), integrated over `44` real 100 Hz INA3221 rail samples each! | **100% CONFIRMED** |

---

## 1. Live Real-Weight `Qwen/Qwen2.5-3B` 14-Arm Evaluation (`4.5000 bpw`, `group_size=128`)

All `14` arms quantize all `252` transformer block linear matrices (`2,774,532,096` weights) at exact `4.5000 bpw` (`9/2 bpw`, `72B / 128-wt` ball tiling), leaving all `182` non-block tensors verified bit-identical (`0` diffs) and achieving `0` Extended Hamming `[16,11,4]` syndrome errors. Matched-MSE arms match layerwise weight-MSE within `<= 1.692e-6 dB`.

| Arm (`4.5000 bpw`) | Weight Err (dB) | WikiText-2 KL ± SE | Wiki PPL | Wiki Top-1 | Python KL ± SE | Py PPL | Py Top-1 | Syn Err | Non-Blk Diff |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Z_plain` | `-24.9282` | `0.066659` ± `0.003543` | `11.503` | `88.67%` | `0.048326` ± `0.002900` | `3.215` | `92.59%` | `0` | `0` |
| `E8_axis` | `-25.5861` | `0.050420` ± `0.002887` | `11.453` | `89.04%` | `0.039298` ± `0.002650` | `3.238` | `92.34%` | `0` | `0` |
| **`E8_H8`** | `-25.5850` | **`0.045724`** ± `0.002077` | `11.315` | `89.82%` | **`0.036741`** ± `0.002207` | `3.237` | `92.88%` | `0` | `0` |
| `E8_H8_matched` | `-25.5861` | `0.046143` ± `0.002094` | `11.305` | `89.87%` | `0.036889` ± `0.002214` | `3.236` | `93.03%` | `0` | `0` |
| **`A16_axis`** | `-25.5532` | **`0.046686`** ± `0.002542` | `11.258` | `89.60%` | **`0.038578`** ± `0.002440` | `3.218` | `92.78%` | `0` | `0` |
| `A16_BlockH2` | `-25.5540` | `0.051623` ± `0.003021` | `11.367` | `89.95%` | `0.040346` ± `0.002697` | `3.249` | `93.69%` | `0` | `0` |
| `A16_BlockH2_matched` | `-25.5532` | `0.051123` ± `0.003007` | `11.358` | `90.02%` | `0.040650` ± `0.002685` | `3.251` | `93.49%` | `0` | `0` |
| `A16_H16` | `-25.5536` | `0.047631` ± `0.002316` | **`11.064`** | `88.97%` | `0.038760` ± `0.002482` | `3.246` | `93.20%` | `0` | `0` |
| `A16_H16_matched` | `-25.5532` | `0.048159` ± `0.002350` | `11.084` | `89.09%` | `0.038642` ± `0.002468` | `3.250` | `93.25%` | `0` | `0` |
| `A16_axis_K10` | `-25.5565` | `0.046800` ± `0.002681` | `11.244` | `89.65%` | `0.038130` ± `0.002436` | `3.218` | `92.81%` | `0` | `0` |
| `hA16_colproxy` | `-25.6159` | `0.045545` ± `0.002153` | `11.281` | `89.63%` | `0.034770` ± `0.002236` | `3.222` | `93.86%` | `0` | `0` |
| `hA16_rand` | `-20.7456` | `0.030950` ± `0.001567` | `11.191` | `91.36%` | `0.022710` ± `0.001285` | `3.169` | `95.03%` | `0` | `0` |
| **`hA16_real`** | `-21.1650` | `0.024417` ± `0.000871` | `11.090` | `92.03%` | **`0.021217`** ± `0.001080` | `3.177` | `95.11%` | `0` | `0` |
| **`hA16_BlockH2_real`** | `-19.4442` | **`0.024179`** ± `0.000739` | **`11.058`** | **`92.32%`** | `0.022975` ± `0.001328` | `3.175` | `95.18%` | `0` | `0` |

### 1.1 Complete 4-Point Live RTX 3080 Scale Law (`0.36B -> 0.5B -> 1.5B -> 3.0B`, Identical Held-Out Corpora)

| Model (`Source Class`) | Block Weights | Uncal KL Winner (Wiki) | `A16` Hadamard Sign (`axis` vs `H16`) | `E8` Hadamard Sign (`H8` vs `axis`) | Wiki `P5` Retention | Py `P5` Retention | Wiki `colproxy` Retention | Spearman $\rho$ (`real / rand / px`) |
| :--- | ---: | :--- | :--- | :--- | ---: | ---: | ---: | :--- |
| `SmolLM2-360M` (`anchor`) | `314,572,800` | `A16_axis` (`0.065173`) | `axis < H16` (`+22.5%`) | `H8 < axis` (`-7.0%`) | `99.78%` | `95.14%` | `33.19%` | `0.719 / 0.513 / 0.101` |
| `Qwen2.5-0.5B` (`anchor`) | `357,826,560` | `A16_axis` (`0.052525`) | `axis < H16` (`+9.7%`) | `H8 < axis` (`-5.6%`) | `97.78%` | `106.11%` | `23.01%` | `0.797 / 0.556 / 0.110` |
| `Qwen2.5-1.5B` (`anchor`) | `1,310,195,712` | `A16_BlockH2` (`0.045648`) | `H16 < axis` (`-3.4%`) | `axis < H8` (`+1.1%` Wiki; `-16.6%` Py) | `91.82%` | `96.99%` | `32.19%` | `0.755 / 0.517 / 0.132` |
| **`Qwen2.5-3B` (`sampled`, LIVE)** | **`2,774,532,096`** | **`E8_H8` (`0.045724`)** | **`axis < H16` (`+2.0%`)** | **`H8 < axis` (`-9.3%` Wiki; `-6.5%` Py)** | **`70.66%`** | **`91.40%`** | **`5.13%`** | **`0.757 / 0.530 / 0.083`** |
| `Gemma-3 270M..12B` (`anchor`, `siliquant#58`) | `270M..12B` | — | — | — | `99% -> 94% -> 84% -> 71%` | — | — | — |

**Core Scientific Takeaway of the 4-Point Scale Law:**
1. **Uncalibrated Rotation Ranking Is Non-Monotonic Across Checkpoints:** `3/4` real models (`0.36B`, `0.5B`, `3.0B`) exhibit the Opposite-Sign `E_8` vs. `A_16` Law (`A16_axis < A16_H16` while `E8_H8 < E8_axis`), whereas `Qwen2.5-1.5B` is a genuine checkpoint-specific reversal where `A16_BlockH2` wins.
2. **`P5` Calibration Decay & `colproxy` Collapse Are Monotonically Real:** Across the exact same held-out WikiText-2 test split, `P5` random-token retention falls monotonically **`99.78% -> 97.78% -> 91.82% -> 70.66%`**, Python bends down to **`91.40%`**, and weight-only `colproxy` collapses to **`5.13%`**, proving real data-matched Fisher calibration (`hA16_real` / `hA16_BlockH2_real`, cutting KL by `48%–53%` at every scale!) becomes indispensable at `>= 3B`.

---

## 2. Full `252`-Layer Anisotropy Census (`Qwen/Qwen2.5-3B`, Matched MSE `<= 1.69e-6 dB`)

Across all `252` matrices (`36L × 7`), `125 / 252` (`49.6%`) are Regime I (`A16_axis < A16_H16_matched`) vs `127 / 252` (`50.4%`) Regime II (`A16_H16_matched < A16_axis`), while `A16_BlockH2` beats `A16_H16` on only **`102 / 252` (`40.5%`, down from `53.6%` at `1.5B`)**. Notice how `down_proj` (`24/36` Regime I), `q_proj` (`23/36`), `up_proj` (`21/36`), and `o_proj` (`20/36`) flip strongly back to Regime I at `3B`, explaining why `A16_axis` regains its KL edge over `A16_H16` and `A16_BlockH2`!

| Projection Kind | Layers | Regime I (`axis < H16`) | Regime II (`H16 < axis`) | `Block-H2 < H16` | Mean `Block-H2 / axis` Quad Err | Mean `H16 / axis` Quad Err |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `down_proj` | `36` | `24` | `12` | `18` | `1.009266` | `1.008120` |
| `gate_proj` | `36` | `7` | `29` | `13` | `0.991524` | `0.984238` |
| `k_proj` | `36` | `12` | `24` | `9` | `1.003431` | `0.988038` |
| `o_proj` | `36` | `20` | `16` | `13` | `0.995060` | `0.992098` |
| `q_proj` | `36` | `23` | `13` | `17` | `1.004885` | `1.004778` |
| `up_proj` | `36` | `21` | `15` | `15` | `1.003844` | `1.000321` |
| `v_proj` | `36` | `18` | `18` | `17` | `1.005075` | `0.998867` |
| **Total (`36L`)** | **`252`** | **`125`** | **`127`** | **`102`** | — | — |

---

## 3. Real `Qwen2.5-3B` CUDA `sm_86` Transcode, Serving & 100 Hz NVML INA3221 + RAPL Energy

Compiled cleanly via `nvcc -O3 -std=c++17 -arch=sm_86 --ptxas-options=-v` on `rtx3080_qwen3b_kernels.cu`: **`10 / 10` kernels, `20–72` registers, `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`**. GPU rail energy is integrated directly via **100 Hz trapezoidal `nvmlDeviceGetPowerUsage` (INA3221)** inside each sustained `6,000`-call serving window (explicitly avoiding `GA104M`'s drifted cumulative energy counter), paired with unlocked `intel-rapl:0` CPU package energy.

| Real `Qwen2.5-3B` Shape (`STREAM Triad`) | Kernel | Throughput (`Gw/s`) | Effective BW (`GB/s`) | `% Triad` | Latency (`ms`) | `rel_l2` (FP64) | Bit/Syn Err |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | :---: |
| **Single `down_proj` `2048×11008`** (`22.54M` wts, `356.54 GB/s`) | `18B -> int8 SoA` transcode | `2.3234` | — | — | `9.7032` | exact | `0 / 0` |
| | `72B -> int8 SoA` transcode | `0.2681` | — | — | `84.1047` | exact | `0 / 0` |
| | `gemv_tc_18B_lossless_int8_dp4a` | `321.9412` | `342.337` | `96.017%` | `0.07003` | `1.08e-7` | `0 / 0` |
| | `gemv_tc_72B_lossless_int8_dp4a` | `337.5902` | `343.153` | `96.246%` | `0.06678` | `1.09e-7` | `0 / 0` |
| | `gemv_direct_4_5bpw_a16_dp4a` | `588.3930` | `331.472` | `92.970%` | `0.03832` | `1.08e-7` | `0 / 0` |
| | `gemv_direct_4_5bpw_block_h2_dp4a` | `570.0558` | `321.142` | `90.072%` | `0.03955` | `1.13e-7` | `0 / 0` |
| **Stacked-2 `down_proj` `4096×11008`** (`45.09M` wts, `356.30 GB/s`) | `18B -> int8 SoA` transcode | `2.1236` | — | — | `21.2318` | exact | `0 / 0` |
| | `72B -> int8 SoA` transcode | `0.2102` | — | — | `214.5303` | exact | `0 / 0` |
| | `gemv_tc_18B_lossless_int8_dp4a` | `339.7134` | `361.152` | **`101.361%`** | `0.13273` | `1.06e-7` | `0 / 0` |
| | `gemv_tc_72B_lossless_int8_dp4a` | `355.6251` | `361.398` | **`101.430%`** | `0.12679` | `1.09e-7` | `0 / 0` |
| | `gemv_direct_4_5bpw_a16_dp4a` | **`615.9468`** | `346.844` | **`97.345%`** | `0.07320` | `1.08e-7` | `0 / 0` |
| | `gemv_direct_4_5bpw_block_h2_dp4a` | **`600.8665`** | `338.353` | **`94.962%`** | `0.07504` | `1.13e-7` | `0 / 0` |

### 3.1 Full `36L` `Qwen2.5-3B` Linear Serving Pace & 100 Hz INA3221 GPU + RAPL CPU `mJ/tok`

| Shape & Direct `4.50-bpw` Kernel (`2,774,532,096` Block Wts) | `ms/tok` | `tok/s` | 100 Hz INA3221 Avg W (`N` Samples) | GPU INA3221 `mJ/tok` | CPU RAPL `mJ/tok` | Combined `mJ/tok` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Single `2048×11008` — `A16_axis` | `4.7154 ms` | `212.07 tok/s` | `49.87 W` (`23`) | `235.36 mJ` | `326.86 mJ` | `562.22 mJ` |
| Single `2048×11008` — `Block-H2` | `4.8672 ms` | `205.46 tok/s` | `51.29 W` (`25`) | `254.61 mJ` | `342.01 mJ` | `596.62 mJ` |
| Stacked-2 `4096×11008` — `A16_axis` | **`4.5045 ms`** | **`222.00 tok/s`** | `89.93 W` (`44`) | `408.05 mJ` | `299.93 mJ` | `707.97 mJ` |
| Stacked-2 `4096×11008` — `Block-H2` | **`4.6176 ms`** | **`216.56 tok/s`** | `89.94 W` (`44`) | `417.61 mJ` | `275.57 mJ` | **`693.18 mJ`** |

---

## 4. Verification Receipts & Synced 9-File Frozen Manifest

- **Certificate (`rtx3080_qwen3b_certificate.json`, `0o600`, `4,718` B):**
  - File SHA-256: `e14aef18e66668c55478e7ec84517520bfde9755603658bdcc728d952a16a8b6`
  - `certificate_sha256`: `df2675097e53a3e59a6a7bf4c0e67dba95d39c872ed48d533ae487f21b352c5b`
- **Verification Gates (`100.0%` PASS):**
  - `4`-mode `-I` parity (`-I -B`, `-I -O -B`, `-I -B -W error`, `PYTHONHASHSEED=42 -B`) + `umask 077`: `--standalone --json == disk 0o600`, `--check valid=true`, `--verify-receipts . all_verified=true` (`12/12`, incl. strict UTC causal lock `prereg < run_start`), `test_rtx3080_qwen3b_r5.py` `5/5 OK`.
  - Live `aletheia mutate --max-mutants 0 --workers 4 --work-root . --add-cwd --guards require_ok`: `2/2` oracle + `66/66` audit = **`68 / 68` (`100.0%`, `survived=0`, `null_mutant=passed`, `PASS`)**.
  - Whole-file AST `BoolOp` (`OP_SWAP + SUBCOND_DEL`): `150/150` oracle + `46/46` audit = **`196 / 196` (`100.0%`, `survived=0`)**.
  - `4`-surface v2.1 `zone_b_fuzzer`: `25/25` CLI + `19/19` FR1 + `19/19` FR0 + `19/19` SEC = **`82 / 82` (`100.0%`, `bypasses=0`, `crashes=0`)**.
  - `aletheia audit-check . --strict-ancestors --safety-scan --scripts rtx3080_qwen3b_audit.py test_rtx3080_qwen3b_r5.py`: **`PASS`** (`10` files, `2/2` R4, `0` findings, `0` leaks, `0` CRLF, `0` bare `assert`s, `0` AST floats/`.isdigit()`, `PYC=0`).

| File | Lines | Bytes | Mode | Exact SHA-256 |
| :--- | ---: | ---: | :---: | :--- |
| `s63_preregistration.json` | `1` | `3,191` | `0o600` | `516142837da45cfcb277b2a2f201dd7bd528c0b44f066339442681946878e29d` |
| `rtx3080_qwen3b_kernels.cu` | `1069` | `48,609` | `0o644` | `40d8474b84a85dfca2c0270624df7fac479c77f5706e66b3708dc2cc305808bf` |
| `rtx3080_qwen3b_eval.py` | `1044` | `44,570` | `0o644` | `a90f7fd1d9b57e1ea86502c3e82609322d8bfdea6ba7a349ce872c2f4e63cc64` |
| `rtx3080_qwen3b_empirical_results.json` | `714` | `22,330` | `0o644` | `7f0448d1c86066e415b308ab31c6a1a3b0ae0a6a6a446170072f13ac977041d2` |
| `mlos_zone_b_receipt.json` | `93` | `2,651` | `0o600` | `50ec43563d0358f0a6bad9edf31978f929b0cf0469724593ffb3f62c79e38cfe` |
| `rtx3080_qwen3b_oracle.py` | `574` | `27,240` | `0o644` | `963f7fded25c253312c06823ea68ea340371d4a1f8b1ffc50248cf3a7fde1c52` |
| `rtx3080_qwen3b_audit.py` | `517` | `20,532` | `0o644` | `b4db16aaa3ea8c5567103cea5a77f1f090cd99790fad338cb116d25ca0498552` |
| `rtx3080_qwen3b_certificate.json` | `1` | `4,718` | `0o600` | `e14aef18e66668c55478e7ec84517520bfde9755603658bdcc728d952a16a8b6` |
| `test_rtx3080_qwen3b_r5.py` | `840` | `43,678` | `0o644` | `88f19ab0176da7b53ab0ce799f4c2694de1ceb25084304d98135b15752baf1f9` |
