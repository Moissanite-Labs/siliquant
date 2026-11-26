# Live 1.5B-Scale Real-Weight LLM (`Qwen/Qwen2.5-1.5B`) Lattice Quantization & RTX 3080 `sm_86` Scale-Law Report

**Epistemic Status:** `exact` (pure-$\mathbb{Z}/\mathbb{Q}$ architectural tiling, parameter totals, and lossless `int8` ball-fit invariants) + `sampled` (`measured_live_model_weights: true` — offline live evaluation of real pretrained `Qwen/Qwen2.5-1.5B` weights on RTX 3080 Laptop GPU `GA104M` `sm_86`, wrapped in `mlos gpu-lease --mode train` and `mlos stamp-run --zone B --require-gpu --strict-anomalies`, `clean_hardware_run: true`, `anomalies: []`) + `anchor` (prior live `SmolLM2-360M` / `Qwen2.5-0.5B` RTX 3080 runs and upstream `siliquant#58` Gemma calibration curve).

> **Energy figures: uncontrolled, provisional.** Every `mJ/tok` and W figure in this report (row H4 of the hypothesis table, §3 and the §3.1 table, including `289.36 mJ/tok`) comes from one laptop run with no repeats and no reported spread. The GPU figure is a single `nvidia-smi` power reading taken after the run, multiplied by the projected time per token. CPU energy is the `intel-rapl:0` package counter for the whole CPU, so it also counts anything else the CPU was doing; CPU load, other processes and user activity were not recorded. `clean_hardware_run: true` covers the receipt's GPU-side checks and applies to the throughput and latency figures, not to the energy figures. No controlled rerun has been done yet. The raw values are kept as measured.

- **Model Audited Live (`3.66×` Block Weight Scale Jump Over `Qwen2.5-0.5B`):**
  - `Qwen/Qwen2.5-1.5B` (`revision`: `8faed761d45a263340a0528343f099c05c9a4323`, `28` layers, `hidden=1536`, `kv=256`, `intermediate=8960`, `196` block linear matrices, `1,310,195,712` quantized block weights, exact `128`-weight `72B` tiling, `142` non-block tensors verified bit-identical across all `16` arms).
  - `model.safetensors` (`3,087,467,144` B, kept exclusively in the local HuggingFace cache): SHA-256 `a961db72e75d52b18e6b0c9d379e51a26973b233385e0e127fdda7d648aec796`.
  - `tokenizer.json`: SHA-256 `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`.
- **Disjoint Corpora (`4,088` Held-Out Next-Token Logits Each, Zero Eval Peeking):**
  - Calibration (`6,144` tokens): `Salesforce/wikitext` `validation` (`rev b08601e0...`, SHA-256 `93abc2e0cd803097893a176fd0de13e75c77a1414c5e64af7bbcf705068938c5`).
  - Held-out prose (`4,088` next-token logits): `Salesforce/wikitext` `test` (SHA-256 `430983e5a711fbc4c6576edffecacf47ff23d58f62f3923d2386d798e0a794c9`, BF16 PPL `12.617995`).
  - Held-out code (`4,088` next-token logits): `6` CPython stdlib modules (`fractions.py`, `difflib.py`, `decimal.py`, `statistics.py`, `argparse.py`, `json/decoder.py`, SHA-256 `fe8ff0f94937ebd41e55f9880dd19bbaa4bbeadec791a6544bbafcb9cd2ee609`, BF16 PPL `3.739529`).
- **Ring-0 Hardware Receipt (`mlos_zone_b_receipt.json`, `0o600`):**
  - `receipt_sha256`: `08d55dff4bef19e4dc657d2fe44e8732a7f61b806821667e8a28bee8d811b3e5` (`profile_id`: `ux582hs-rtx3080-8gb`, `profile_sha256`: `566c9192d38682a0ec4d652a63a3510aaefb9dc20ef91ae9bdc6d4abbbf5cbe2`).
  - `clean_hardware_run: true`, `anomalies: []`, `exit_code: 0`, `duration_ms: 572885.698`, `graphics_processes: []`, `linux_kmd_baseline_vram_mib: 2`, `pstate: P0`, `6001 MHz` GDDR6, `1710 MHz` SM, `63 °C`, `ac_online: true`.

---

## Executive Summary of Four Pre-Registered Out-of-Sample Hypotheses

| Pre-Registered Question (`1.5B` Live Silicon) | Pre-Run Expectation | Live Observed `Qwen/Qwen2.5-1.5B` Outcome (`measured_live_model_weights: true`) | Verdict |
| :--- | :--- | :--- | :---: |
| **H1. Opposite-sign `E_8` vs `A_16` Hadamard law at `1.5B`** | `A16_axis < A16_H16` while `E8_H8 < E8_axis` (extrapolated from `0.36B`/`0.5B`) | **SCALE PHASE TRANSITION OBSERVED:** On `Qwen2.5-1.5B`, `A16_H16` overtakes `A16_axis` on BOTH WikiText-2 (`0.048376` vs `0.050062`, `-3.37%`; matched `-3.56%`) and Python (`0.030466` vs `0.031973`, `-4.71%`). **`A16_BlockH2` wins overall among all uniform uncalibrated arms on BOTH corpora** (`0.045648` WikiKL `-8.82%` vs `A16_axis`, `-5.64%` vs `A16_H16`; `0.030310` PyKL `-5.20%` vs `A16_axis`). On `E_8`, `E8_H8` still beats `E8_axis` strongly on Python (`0.030131` vs `0.036110`, `-16.56%`) while `E8_axis` slightly edges `E8_H8` on WikiText-2 (`0.052795` vs `0.053354`). | **REVERSED / REFINED** (`A16_BlockH2` wins `1.5B`) |
| **H2. Does `P5` random-token calibration decay emerge `0.5B -> 1.5B`?** | WikiText-2 `P5` retention drops below the `97.78%` (`0.5B`) / `99.78%` (`0.36B`) small-scale plateau | **CONFIRMED ON WIKITEXT-2 (DOMAIN-SELECTIVE):** WikiText-2 `P5` retention drops `-5.96` percentage points from `97.78%` (`Qwen2.5-0.5B`) to **`91.82%`** (`Qwen2.5-1.5B`; `hA16_real 0.023637` `-52.79%` vs `hA16_rand 0.025799`), while Python stays high at **`96.99%`**. Weight-only `hA16_colproxy` remains weak (`32.19%` Wiki / `31.55%` Python; Spearman $\rho = 0.132$ vs `0.755` real). | **CONFIRMED** |
| **H3. Leakage-free calibration-routed hybrid selectors (`A16_hybrid_cal_*`)** | Per-layer routing from disjoint WikiText-2 validation `cal_acts` beats uniform `A16_axis` & `A16_H16` | **NUANCED MIXED OUTCOME:** Out-of-sample `cal -> eval` Regime I/II classification reaches **`138 / 196` (`70.41%`)**. All three hybrid arms beat `A16_axis` on both corpora; **`A16_hybrid_cal_H16` achieves the #1 best uncalibrated Python KL (`0.028573`, PPL `3.819`, `-10.64%` vs `A16_axis`, `-5.73%` vs `A16_BlockH2`)** and `A16_hybrid_cal_BlockH2` hits the lowest uncalibrated WikiText-2 PPL (`12.892`). However, **uniform `A16_BlockH2` still beats every hybrid selector on WikiText-2 KL (`0.045648` vs `0.047239`..`0.049116`)**, and 3-way `cal_best3` overfits `256`-token calibration slices. | **PARTIAL** (wins Python; uniform `Block-H2` wins WikiKL) |
| **H4. Real `1.5B` CUDA `sm_86` transcode, serving & GPU+RAPL `mJ/tok`** | Zero-spill `10/10` kernels, `>100%` Triad lossless transcoded `int8`, `>94%` Triad direct `4.50-bpw` on real `1536×8960` & `6144×8960` `down_proj` | **CONFIRMED:** `10/10` zero-spill kernels (`20–72` regs), `0` bit mismatches, `0` syndrome errors, `rel_l2 <= 1.072e-7`. On stacked-4 `6144×8960` (`55.05M` wts): lossless transcoded `72B` hits `357.86 Gw/s` (`101.90%` Triad, `726.0×` on-the-fly `72B`), direct `A16_axis` hits `617.55 Gw/s` (`97.44%` Triad, `2.1216 ms/tok`, `471.34 tok/s`, `188.08 mJ/tok` GPU + `101.28 mJ/tok` CPU RAPL = `289.36 mJ/tok`, energy uncontrolled, provisional), and direct `Block-H2` hits `596.86 Gw/s` (`94.17%` Triad, only `3.35%` overhead for `#1` uncalibrated WikiKL!). | **CONFIRMED** |

---

## 1. Live Real-Weight `Qwen/Qwen2.5-1.5B` 16-Arm Evaluation (`4.5000 bpw`, `group_size=128`)

All `16` arms quantize all `196` transformer block linear matrices (`1,310,195,712` weights) at exact `4.5000 bpw` (`9/2 bpw`, `72B / 128-wt` ball tiling), leaving all `142` non-block tensors (`embed_tokens`, `lm_head`, norms, biases) verified bit-identical (`0` diffs) and achieving `0` Extended Hamming `[16,11,4]` syndrome errors. Matched-MSE arms scale each layer's quantization residual to match its respective `A16_axis` or `E8_axis` layerwise weight-MSE within `<= 1.236e-6 dB`.

| Arm (`4.5000 bpw`) | Weight Err (dB) | WikiText-2 KL ± SE | Wiki PPL | Wiki Top-1 | Python KL ± SE | Py PPL | Py Top-1 | Syn Err | Non-Blk Diff |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Z_plain` | `-24.9275` | `0.054285` ± `0.004097` | `13.226` | `88.11%` | `0.034835` ± `0.003196` | `3.884` | `93.15%` | `0` | `0` |
| `E8_axis` | `-25.5846` | `0.052795` ± `0.005947` | `13.164` | `89.19%` | `0.036110` ± `0.005262` | `3.864` | `93.20%` | `0` | `0` |
| `E8_H8` | `-25.5848` | `0.053354` ± `0.003931` | `13.216` | `89.80%` | `0.030131` ± `0.003144` | `3.863` | `93.42%` | `0` | `0` |
| `E8_H8_matched` | `-25.5846` | `0.053391` ± `0.003884` | `13.223` | `89.68%` | `0.030327` ± `0.003162` | `3.864` | `93.40%` | `0` | `0` |
| `A16_axis` | `-25.5532` | `0.050062` ± `0.004083` | `13.011` | `89.29%` | `0.031973` ± `0.003214` | `3.846` | `93.76%` | `0` | `0` |
| **`A16_BlockH2`** | `-25.5532` | **`0.045648`** ± `0.003010` | `12.914` | `89.24%` | `0.030310` ± `0.002611` | `3.869` | `92.49%` | `0` | `0` |
| **`A16_BlockH2_matched`** | `-25.5532` | **`0.045695`** ± `0.003013` | `12.921` | `89.14%` | `0.030437` ± `0.002641` | `3.869` | `92.69%` | `0` | `0` |
| `A16_H16` | `-25.5531` | `0.048376` ± `0.003659` | `13.007` | `89.55%` | `0.030466` ± `0.003148` | `3.824` | `93.66%` | `0` | `0` |
| `A16_H16_matched` | `-25.5532` | `0.048281` ± `0.003671` | `12.990` | `89.26%` | `0.030456` ± `0.003137` | `3.825` | `93.66%` | `0` | `0` |
| `A16_hybrid_cal_BlockH2` | `-25.5532` | `0.047239` ± `0.003364` | **`12.892`** | `89.41%` | `0.030387` ± `0.002469` | `3.854` | `92.95%` | `0` | `0` |
| **`A16_hybrid_cal_H16`** | `-25.5532` | `0.048825` ± `0.003633` | `12.958` | `89.14%` | **`0.028573`** ± `0.002553` | **`3.819`** | `93.32%` | `0` | `0` |
| `A16_hybrid_cal_best3` | `-25.5532` | `0.049116` ± `0.003655` | `12.899` | `88.92%` | `0.031378` ± `0.002609` | `3.847` | `92.95%` | `0` | `0` |
| `A16_axis_K10` | `-25.5572` | `0.049865` ± `0.004063` | `12.982` | `89.46%` | `0.031757` ± `0.003219` | `3.844` | `93.71%` | `0` | `0` |
| `hA16_colproxy` | `-25.5858` | `0.041557` ± `0.003703` | `13.069` | `89.80%` | `0.026654` ± `0.002792` | `3.836` | `94.03%` | `0` | `0` |
| `hA16_rand` | `-22.6753` | `0.025799` ± `0.000801` | `12.788` | `91.54%` | `0.015621` ± `0.000846` | `3.776` | `95.03%` | `0` | `0` |
| **`hA16_real`** | `-22.9064` | **`0.023637`** ± `0.000715` | `12.858` | `92.20%` | **`0.015113`** ± `0.000617` | `3.788` | `94.81%` | `0` | `0` |

### 1.1 Cross-Scale Evolution (`0.36B -> 0.5B -> 1.5B` Live RTX 3080 + Upstream Gemma Anchors)

| Model (Source Class) | Block Weights | Uncal `A16` Winner (WikiKL) | `A16_H16 / A16_axis` (WikiKL) | `A16_BlockH2 / A16_axis` (WikiKL) | Wiki `P5` Rand Retention | Py `P5` Rand Retention | Wiki `colproxy` Retention | Spearman $\rho$ (`real / rand / px`) |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `SmolLM2-360M` (`anchor`, live RTX 3080) | `314,572,800` | `A16_axis` (`0.065173`) | `1.2251×` (`+22.5%`) | `1.1002×` (`+10.0%`) | `99.78%` | `95.14%` | `33.19%` | `0.719 / 0.513 / 0.101` |
| `Qwen2.5-0.5B` (`anchor`, live RTX 3080) | `357,826,560` | `A16_axis` (`0.052525`) | `1.0969×` (`+9.7%`) | `1.0510×` (`+5.1%`) | `97.78%` | `106.11%` | `23.01%` | `0.797 / 0.556 / 0.110` |
| **`Qwen2.5-1.5B` (`sampled`, LIVE HERE)** | **`1,310,195,712`** | **`A16_BlockH2` (`0.045648`)** | **`0.9663×` (`-3.4%`)** | **`0.9118×` (`-8.8%`)** | **`91.82%`** | **`96.99%`** | **`32.19%`** | **`0.755 / 0.517 / 0.132`** |
| `Gemma-3 270M -> 1B -> 4B -> 12B` (`anchor`, `siliquant#58`) | `270M..12B` | — | — | — | `99% -> 94% -> 84% -> 71%` | — | — | — |

The monotonic `Qwen2.5-0.5B -> Qwen2.5-1.5B` WikiText-2 `P5` retention drop (`97.78% -> 91.82%`, exact drop `5.9647` percentage points) aligns directly with the `siliquant#58` Gemma scaling trajectory (`99%` at `270M` -> `94%` at `1B` -> `84%` at `4B`), while simultaneously revealing a sharp uncalibrated `A_16` rotation crossover: as hidden width doubles (`896 -> 1536`), pairwise `A16_BlockH2` flips from a `+5.1%` penalty at `0.5B` to an `-8.8%` win at `1.5B`, outperforming BOTH `A16_axis` and full `A16_H16` at matched layerwise weight-MSE!

---

## 2. Full `196`-Layer Anisotropy Census & Leakage-Free Hybrid Routing

Scoring all `196` real `Qwen2.5-1.5B` matrices on held-out `eval_acts` at matched layerwise weight-MSE shows `101 / 196` (`51.5%`) Regime I (`A16_axis < A16_H16_matched`) vs `95 / 196` (`48.5%`) Regime II (`A16_H16_matched < A16_axis`). Crucially, `A16_BlockH2` beats `A16_H16` on **`105 / 196` (`53.6%`)** matrices — capturing the strongest Regime II gains on `o_proj` (`-1.77%`), `k_proj` (`-1.14%`), and `gate_proj` (`-0.28%`) while avoiding full `H16`'s cross-coordinate leakage on Regime I `up_proj` (`20/28` Regime I) and `q_proj`.

| Projection Kind | Layers | Regime I (`axis < H16`) | Regime II (`H16 < axis`) | `Block-H2 < H16` | `cal -> eval` Regime Agree | Mean `Block-H2 / axis` Quad Err | Mean `H16 / axis` Quad Err |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `down_proj` | `28` | `12` | `16` | `16` | `19/28` (`67.9%`) | `1.002081` | `1.000814` |
| `gate_proj` | `28` | `13` | `15` | `16` | `16/28` (`57.1%`) | `0.997156` | `0.999910` |
| `k_proj` | `28` | `13` | `15` | `15` | `21/28` (`75.0%`) | `0.988630` | `0.994153` |
| `o_proj` | `28` | `11` | `17` | `17` | `18/28` (`64.3%`) | `0.982251` | `0.988372` |
| `q_proj` | `28` | `16` | `12` | `12` | `21/28` (`75.0%`) | `1.001660` | `1.002851` |
| `up_proj` | `28` | `20` | `8` | `17` | `19/28` (`67.9%`) | `1.002148` | `1.003773` |
| `v_proj` | `28` | `16` | `12` | `12` | `24/28` (`85.7%`) | `1.008567` | `1.005725` |
| **Total (`28L`)** | **`196`** | **`101`** | **`95`** | **`105`** | **`138/196` (`70.4%`)** | — | — |

- **Disjoint Calibration Routing Counts (`cal_acts`, zero eval peeking):** `94` Regime I (`A16_axis`) / `102` Regime II (rotated); `cal_best3` selected `70` `A16_axis`, `61` `A16_BlockH2`, and `65` `A16_H16`.
- **Practical Engineering Takeaway:** Although `A16_hybrid_cal_H16` sets the uncalibrated Python KL record (`0.028573`), `29.6%` per-layer sign-flip noise between `256`-token calibration slices and held-out prose causes hybrid selectors to trail uniform `A16_BlockH2` on WikiText-2 KL (`0.047239` vs `0.045648`). Uniform **`A16_BlockH2`** is therefore the zero-calibration Pareto optimum at `1.5B` scale.

---

## 3. Real `Qwen2.5-1.5B` CUDA `sm_86` Transcode, Serving & Live Energy/Token

Compiled cleanly via `nvcc -O3 -std=c++17 -arch=sm_86 --ptxas-options=-v` on `rtx3080_qwen15b_kernels.cu`: **`10 / 10` kernels, `20–72` registers, `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`**. Both real `Qwen2.5-1.5B` `down_proj` shapes achieved `0` bit mismatches, `0` syndrome errors, and `rel_l2_err_fp64 <= 1.072e-7 < 1e-6`.

| Real `Qwen2.5-1.5B` Shape (`STREAM Triad`) | Kernel | Throughput (`Gw/s`) | Effective BW (`GB/s`) | `% Triad` | Latency (`ms`) | `rel_l2` (FP64) | Bit/Syn Err |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | :---: |
| **Single `down_proj` `1536×8960`** (`13.76M` wts, `356.34 GB/s`) | `18B -> int8 SoA` transcode | `2.4172` | — | — | `5.6935` | exact | `0 / 0` |
| | `72B -> int8 SoA` transcode | `0.2446` | — | — | `56.2706` | exact | `0 / 0` |
| | `gemv_tc_18B_lossless_int8_dp4a` | `321.9800` | `342.457` | `96.103%` | `0.04274` | `9.74e-8` | `0 / 0` |
| | `gemv_tc_72B_lossless_int8_dp4a` | `337.7282` | `343.376` | `96.361%` | `0.04075` | `1.03e-7` | `0 / 0` |
| | `gemv_direct_4_5bpw_a16_dp4a` | `560.6656` | `315.990` | `88.676%` | `0.02455` | `1.03e-7` | `0 / 0` |
| | `gemv_direct_4_5bpw_block_h2_dp4a` | `543.2052` | `306.149` | `85.914%` | `0.02534` | `1.07e-7` | `0 / 0` |
| **Stacked-4 `down_proj` `6144×8960`** (`55.05M` wts, `356.89 GB/s`) | `18B -> int8 SoA` transcode | `2.1839` | — | — | `25.2078` | exact | `0 / 0` |
| | `72B -> int8 SoA` transcode | `0.2682` | — | — | `205.2944` | exact | `0 / 0` |
| | `gemv_tc_18B_lossless_int8_dp4a` | `341.8699` | `363.445` | **`101.836%`** | `0.16103` | `1.01e-7` | `0 / 0` |
| | `gemv_tc_72B_lossless_int8_dp4a` | `357.8572` | `363.667` | **`101.898%`** | `0.15383` | `1.04e-7` | `0 / 0` |
| | `gemv_direct_4_5bpw_a16_dp4a` | **`617.5481`** | `347.747` | **`97.437%`** | `0.08914` | `1.03e-7` | `0 / 0` |
| | `gemv_direct_4_5bpw_block_h2_dp4a` | **`596.8649`** | `336.100` | **`94.174%`** | `0.09223` | `1.07e-7` | `0 / 0` |

### 3.1 Full `28L` `Qwen2.5-1.5B` Linear Serving Pace & Live GPU + `intel-rapl:0` Energy/Token

| Projection Basis (`1,310,195,712` Block Weights) | Direct `4.50-bpw` `ms/tok` | `tok/s` | GPU Sampled Power | Est. GPU `mJ/tok` | Measured CPU RAPL `mJ/tok` | Combined `mJ/tok` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Single-layer grid (`1536×8960`) | `2.3369 ms` | `427.92 tok/s` | `45.22 W` | `105.67 mJ` | `133.18 mJ` | `238.85 mJ` |
| Saturated stacked-4 grid (`6144×8960`) | **`2.1216 ms`** | **`471.34 tok/s`** | `88.65 W` | `188.08 mJ` | `101.28 mJ` | `289.36 mJ` |

*Power (W) and energy (`mJ/tok`) columns: uncontrolled, provisional (see the note at the top of this report).*

---

## 4. Hardened Verification Receipts & Synced 8-File Frozen Manifest

- **Certificate (`rtx3080_qwen15b_certificate.json`, `0o600`, `4,200` B):**
  - File SHA-256: `ad661dd42dcceb4e0943244b46e996c7f7e234560e5d8cfe2385cb85b84d724a`
  - `certificate_sha256`: `221d2c29730a3c9fb127b0a2a8cce2c0c97b4f335be8925dcf49c6ea1f63b698`
- **Hardening Baked In:** Iterative depth/int/UTF-8-surrogate tree validation, iterative bijective exact-type payload pin (`0` hollow checks), constant-time `hmac.compare_digest` SHA-256 verification, inode-matched `(st_dev, st_ino)` `$PWD` check catching `(OSError, ValueError)`, verified root-owned macOS Darwin prefix normalization, `O_DIRECTORY | O_NOFOLLOW` ancestor walk, `0o600` atomic write with `allow_nan=False`, `st_nlink == 1`, bounded `EEXIST` retry, and scratch `os.mkfifo` non-regular testing.
- **Verification Gates (`100.0%` PASS):**
  - `4`-mode `-I` parity (`-I -B`, `-I -O -B`, `-I -B -W error`, `PYTHONHASHSEED=42 -B`) + `umask 077`: `--standalone --json == disk 0o600`, `--check valid=true`, `--verify-receipts . all_verified=true` (`12/12`), `test_rtx3080_qwen15b_r5.py` `5/5 OK`.
  - Live `aletheia mutate --max-mutants 0 --work-root . --add-cwd --guards require_ok`: `2/2` oracle + `65/65` audit = **`67 / 67` (`100.0%`, `survived=0`, `truncated=False`, `PASS`)**.
  - Whole-file AST `BoolOp` (`OP_SWAP + SUBCOND_DEL`): `146/146` oracle + `42/42` audit = **`188 / 188` (`100.0%`, `survived=0`)**.
  - `4`-surface v2.1 `zone_b_fuzzer`: `25/25` CLI + `19/19` FR1 + `19/19` FR0 + `19/19` SEC = **`82 / 82` (`100.0%`, `bypasses=0`, `crashes=0`)**.
  - `aletheia audit-check . --strict-ancestors --safety-scan --scripts rtx3080_qwen15b_audit.py test_rtx3080_qwen15b_r5.py`: **`PASS`** (`0` findings, `0` leaks, `0` CRLF, `0` bare `assert`s, `0` AST floats/`.isdigit()`, `PYC=0`).

| File | Lines | Bytes | Mode | Exact SHA-256 |
| :--- | ---: | ---: | :---: | :--- |
| `rtx3080_qwen15b_kernels.cu` | `970` | `44,653` | `0o644` | `1381a0ac1831ba40f8f06f719b11758fe7a9d12fad3ed6e5bfd4a9883493de88` |
| `rtx3080_qwen15b_eval.py` | `1024` | `43,530` | `0o644` | `bab3923712d3c3f426417e0d9267345f25b52a4f77cd122aa9b3f46604242880` |
| `rtx3080_qwen15b_empirical_results.json` | `744` | `22,846` | `0o644` | `9155e1b679c7019cf49071ef27bfae7b3e4f77c21d099b2ff9c9f4ddb90cf27b` |
| `mlos_zone_b_receipt.json` | `91` | `2,630` | `0o600` | `7d6b48d9471bbbf82fbbb2e2d6be53c9ea91de81dabcfa90e348afaa1ad563b7` |
| `rtx3080_qwen15b_oracle.py` | `540` | `24,994` | `0o644` | `c26154f821b8192aa7a339597ba2e2c2bc8b851f6affcb9d806010c984a1ae22` |
| `rtx3080_qwen15b_audit.py` | `501` | `19,828` | `0o644` | `1edafccc64b59c26fae0e6cb5b042e3df06dcf09f8f011cfe9baee9e9405f780` |
| `rtx3080_qwen15b_certificate.json` | `1` | `4,200` | `0o600` | `ad661dd42dcceb4e0943244b46e996c7f7e234560e5d8cfe2385cb85b84d724a` |
| `test_rtx3080_qwen15b_r5.py` | `816` | `43,256` | `0o644` | `1e3572c609e719843c1ea975bac4a5a2b5539053ccfd9acf50237eb1e8bceb7b` |
