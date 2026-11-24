# Live Real-Weight Open LLM (`Qwen2.5-0.5B` & `SmolLM2-360M`) RTX 3080 `sm_86` `A_16` / `Block-H2` / `H_16` / `E8_Had` Evaluation & End-to-End CUDA Transcode Serving

Live bare-metal NVIDIA RTX 3080 Laptop (`GA104M`, `sm_86`, `110W` AC `P0`) real-weight LLM quantization evaluation and CUDA transcode-serving study accompanying `Moissanite-Labs/siliquant` (`#58`, `#77`), paired with a pure-$\mathbb{Z}/\mathbb{Q}$ zero-float verifier and deterministic certificate (`measured_live_model_weights: true`, zero model weights committed).

## Highlights

1. **Opposite-Sign `E_8` vs. `A_16` Hadamard Law on Both Real Pretrained LLMs (`672.4M` Real Parameters, `26` Arms, `0` Syndrome Errors):**
   - **On `E_8`, Hadamard $H_8$ helps both real models (`E8_H8 < E8_axis`), reproducing `siliquant#77`:** cuts WikiText-2 KL by `-7.0%` (`0.071621` vs. `0.076987`) and Python KL by `-19.0%` (`0.056793` vs. `0.070137`) on `SmolLM2-360M` (`grp=32`), and cuts WikiText-2 KL by `-5.6%` (`0.054841` vs. `0.058117`) and Python KL by `-2.9%` (`0.044983` vs. `0.046320`) on `Qwen2.5-0.5B` (`grp=128`).
   - **On `A_16`, Hadamard $H_{16}$ hurts both real models (`A16_axis < A16_BlockH2 < A16_H16`), leaving unrotated `A16_axis` #1 uncalibrated on WikiText-2 for both models:**
     - `SmolLM2-360M`: `A16_axis` (`0.065173` Wiki / `0.056725` Py) beats `E8_H8` (`-9.0%`) and `Z_plain` (`-18.9%`), while `A16_H16` degrades to `0.079847` (`+22.5%`) / `0.069956` (`+23.3%`), persisting unchanged at `1.00e-6 dB` matched weight MSE (`0.079666` / `0.069862`).
     - `Qwen2.5-0.5B`: `A16_axis` (`0.052525` Wiki / `0.045602` Py) beats `E8_H8` (`-4.2%`) and `Z_plain` (`-15.3%`), while `A16_H16` degrades to `0.057614` (`+9.7%`) / `0.047449` (`+4.0%`), persisting unchanged at `1.51e-6 dB` matched weight MSE (`0.057551` / `0.047448`).
2. **Real-Weight Calibration & `+K10` Kept-Column Shootout:**
   - Real WikiText-2 Fisher (`hA16_real`) cuts WikiText-2 KL by `-52.5%` on `SmolLM2-360M` (`0.030955`) and `-46.5%` on `Qwen2.5-0.5B` (`0.028102`), transferring strongly to held-out Python (`-51.5%` / `-45.6%`).
   - Data-free `1/col_rms^2` proxy (`hA16_colproxy`) captures only `33.19%` (`SmolLM2-360M`, Spearman `+0.1010` vs. real `+0.7194`) and `23.01%` (`Qwen2.5-0.5B`, Spearman `+0.1096` vs. real `+0.7972`) of real Fisher gain.
   - Random-token Fisher (`hA16_rand`, `P5`) retains `99.78%` (`SmolLM2-360M`) and `97.78%` (`Qwen2.5-0.5B`) at `0.36B–0.5B` scale, sitting at the small-model origin of `siliquant#58`'s Gemma 3 scaling curve (`90.8% -> 85.3% -> 78.7% -> 53.7%`).
   - `+K10` kept columns (`A16_axis_K10`) cuts `SmolLM2-360M` KL by `-9.57%` Wiki / `-11.25%` Python while remaining nearly flat on `Qwen2.5-0.5B`.
3. **End-to-End Real-Weight CUDA Transcode & Serving (`Qwen2.5-0.5B` `52.30M` Real `down_proj` Weights, `10/10` Zero-Spill `sm_86` Kernels):**
   - One-time GPU decode from spherical-ball cold storage to lossless VRAM `int8` SoA takes `23.75 ms` (`18B` `hampack`, `~162.5 ms` full `Qwen2.5-0.5B`) and `206.64 ms` (`72B` `hammingindex`, `~1.41 s` full `Qwen2.5-0.5B`) with `0` bit mismatches and `0` syndrome errors.
   - Lossless transcoded VRAM `int8` `__dp4a` GEMV serves exact `72B` weights at **`358.52 Gweights/s` (`364.45 GB/s` = `102.24%` STREAM Triad — `727.4x` faster than on-the-fly GPU `72B` GEMV)** and exact `18B` weights at **`342.43 Gweights/s` (`102.15%` Triad — `91.7x` faster)** (`rel_l2 <= 9.10e-8`).
   - Direct `4.50-bpw` `A16_axis_preencoded_dp4a` reaches **`612.42 Gweights/s` (`96.79%` Triad)** (`0.5843 ms/token` / `1,711.5 tok/s` full 24-layer linear-GEMV floor at `50.79 mJ/tok` GPU + `32.62 mJ/tok` live `intel-rapl:0` CPU = `83.40 mJ/tok` combined).

## Reproduce Verification (Zero External Dependencies)

```bash
python3 -I -B studies/rtx3080-live-llm-a16-eval/rtx3080_live_llm_audit.py --check studies/rtx3080-live-llm-a16-eval/rtx3080_live_llm_certificate.json
python3 -I -B studies/rtx3080-live-llm-a16-eval/rtx3080_live_llm_audit.py --verify-receipts studies/rtx3080-live-llm-a16-eval
python3 -I -B studies/rtx3080-live-llm-a16-eval/test_rtx3080_live_llm_r5.py
```

## File Manifest

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
| `RTX3080_LIVE_LLM_A16_REPORT.md` | `156` | `21947` | `0o644` | `33e378ac42eb4a83f98842735dabf31624d688d78f6d961a69303d2f279842b0` |
