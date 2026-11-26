# RTX 3080 Live-Weight Study: PTX-Ready Shared-RMSNorm `quad_snake`, Exact Load-Power $p$-Curve, and the `3.0 bpw` Two-Level Stratified `BlockH4` Frontier

Standalone, self-verifying empirical hardware study evaluating **96 live quantized full-model arms** across **4 pretrained open-weight LLMs** (`HuggingFaceTB/SmolLM2-360M`, `Qwen/Qwen2.5-0.5B`, `Qwen/Qwen2.5-1.5B`, `Qwen/Qwen2.5-3B`; **120 transformer blocks**, **840 linear matrices**, **4,757,127,168 block weights**, **0 syndrome errors**, **0 out-of-ball groups**, **0 non-block tensor mutations**) at exact **`4.500 bpw`** and **`3.000 bpw`** (`#85` integer theta-series ball bounds) on an NVIDIA GeForce RTX 3080 Laptop GPU (`sm_86`).

## Headline Findings

1. **PTX-Ready Shared-RMSNorm & Zero-Cost Records (`4.5 bpw`, Executable via `#93` `sm_86` PTX Kernel):**
   - `Qwen/Qwen2.5-3B`: PTX-ready `b45_hA16_BlockH4_quad_snake_sharednorm` beats per-matrix `b45_hA16_BlockH2_real` on both corpora (`WikiKL 0.024020`, `-0.66%`; `PyKL 0.021344`, `-7.10%` vs `0.024179` / `0.022975`), and `quad_lpt_sqrt_sharednorm` cures linear-LPT starvation (`0.025409` vs `0.026776`, `-5.11%`).
   - `Qwen/Qwen2.5-0.5B`: `quad_lpt_sqrt_real` reaches `WikiKL 0.025153` (`-6.74%`) / `PyKL 0.022709` (`-5.58%`); PTX-ready `quad_lpt_sqrt_sharednorm` reaches `0.025713` (`-4.66%`) / `0.022710` (`-5.57%`).
   - `Qwen/Qwen2.5-1.5B` & `HuggingFaceTB/SmolLM2-360M`: zero-runtime-cost `quad_snake_down_zerocost` (offline-folded `down_proj`, zero activation gather) achieves `1.5B` `WikiKL 0.022783` (`-2.14%` vs `BlockH2`) and `360M` `WikiKL 0.027934` / `PyKL 0.025417` (`-0.97%` / `-4.45%` vs unpermuted `BlockH4`).
2. **Exact 7-Point `Qwen2.5-3B` Load-Power $p$-Curve (`4.5 bpw`) & Starvation Telemetry:**
   - `p = 0.375` (`b45_hA16_BlockH4_quad_lpt_p0375_real`) reaches **`PyKL 0.018248`** (`-20.58%` vs `BlockH2`, `-6.51%` vs `p = 0.50` `0.019520`) while beating unpermuted `BlockH4` on WikiText-2 (`0.024678` < `0.024774`).
   - Rank-invariant `quad_snake` (`p = 0.000`) maintains **`0.00%`** Ball-0 starvation across all 4 models and holds the `3B` WikiKL crown (`0.023154`, `-4.24%`), whereas linear LPT (`p = 1.000`) starves **`100.00%`** of `3B` matrices (mean Ball-0 2nd-quad rank jumps `1.94G` -> `28.56G`). Exact L2-energy `ball_cv_sqrt` reaches its global minimum at `p = 0.500` on all `3/3` Qwen2.5 models.
3. **`3.0 bpw` Low-Bitrate Frontier & Distortion-Ratio Amplification (`#85` Exact Bounds `g=128 A16=651/E8=420`, `g=32 A16=151/E8=96`):**
   - `quad_snake_real` beats global cannibalized `sort_real` on all `4/4` models (`360M` `-34.28%`, `0.5B` `-29.14%`, `1.5B` `-11.78%`, `3B` `-14.38%`).
   - On `Qwen/Qwen2.5-3B`, `b30_hA16_BlockH4_quad_snake_real` hits **`WikiKL 0.168874`** (`-11.42%` vs `b30_hA16_BlockH2_real` `0.190650` — **`2.70x`** larger relative margin than `-4.24%` at `4.5 bpw`; `PPL 12.141` vs BF16 `10.970`), and PTX-ready `quad_snake_sharednorm` hits **`0.169901`** (`-10.88%`, **`16.58x`** larger relative margin than `-0.66%` at `4.5 bpw`; `PPL 12.081`), lowering `KL(3.0)/KL(4.5)` blowup from `7.89x` (`BlockH2`) to `7.29x` (`quad_snake`) and `7.07x` (`quad_snake_sharednorm`).
   - On `Qwen/Qwen2.5-1.5B`, `b30_hA16_BlockH4_quad_lpt_sqrt_real` hits **`0.177318`** (`-4.95%` vs `BlockH2`, `-15.20%` vs `sort`, blowup `7.51x` < `8.01x`). Zero-text `quad_snake_rand` beats `A16_axis` & `E8_H8` on all `4/4` models (`3B`: `-58.25%` vs `A16_axis`, `-85.92%` vs `E8_H8`), while uncalibrated `3B` `E8_H8` collapses (`2.97x` worse than `A16_axis`). On small-scale `360M`/`0.5B` at `3.0 bpw`, unpermuted `hA16_BlockH4_real`/`hA16_real` slightly edge `quad_snake`.

## Reproducible Verification (CPU-Only, Zero External Dependencies)

```bash
cd studies/rtx3080-s73-lowbit-and-sharednorm-quad-snake
python3 -I -B rtx3080_s73_audit.py --check rtx3080_s73_certificate.json
python3 -I -B rtx3080_s73_audit.py --verify-receipts .
python3 -I -B test_rtx3080_s73_r5.py
```

Refs #77, #85, #93, #94, #95, #99, #100, #101
