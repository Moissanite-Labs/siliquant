# Two-Level Saddle-Point Theory for Block-Hadamard Quantization: Machine-Checked Lean 4.36.0 Formal Companion

**Project ID:** `CEER-TWO-LEVEL-SADDLE-POINT-LEAN4`  
**Certificate Schema:** `aletheia.ceer_two_level_saddle_point_lean4.certificate.v1`  
**Target Repository:** `Moissanite-Labs/siliquant` (`Refs #77, #85, #87, #89, #90, #91, #92, #93`)  
**Toolchain:** `leanprover/lean4:v4.36.0` (pure `Init`, zero Mathlib, zero external imports, zero compiler warnings)  
**Novelty Tier & Scope:** `N1` (constructive application and unified machine-checked formalization of classical `N0` inequalities to two-scale block-Hadamard quantization, cross-layer interference reversal, and finite-precision SwiGLU column folding)  
**Oracle Class:** `exact` (pure integer and rational arithmetic over $\mathbb{Z}$ and $\mathbb{Q}$, zero IEEE-754 floats on trust paths, zero bare `assert`s, zero mathematical RNG)  
**Hardware Execution Scope:** `measured_live_gpu_in_this_bundle = false` (pure symbolic Lean 4.36.0 theorem proving and pure-$\mathbb{Z}/\mathbb{Q}$ Python certification; empirical constants are read-only anchored to the frozen empirical artifact `rtx3080_s69_empirical_results.json`, SHA-256 `05175f89d323779ae04d9cfcf02345a87929a42fdffe52708470c1665026ed62`, and `rtx3080_s69_certificate.json`, file SHA-256 `7526c50285bb837f14e23373fbc7d8ab7c678a9a1de32fb9cf72d5d6ef9ee0ef`, inner `certificate_sha256` `29640bbf590a1d354d823f39f6d4ce0b0e6ee3ad0e9001a676a8dc45fafdcd83`).

---

## 1. Executive Summary

Empirical measurements in CEER Cycle 4 (`Moissanite-Labs/siliquant#93`, anchored to `05175f89...`) revealed three sharp structural phenomena in INT4/FP4 group-quantized transformer projections with $H_4$ micro-block Hadamard mixing and $g = 128$ macro-ball scaling:

1. **4/4-Model Global-Sort Cannibalization:** Sorting all hidden channels globally by descending Hessian magnitude before contiguous $g = 128$ quantization strictly *worsens* full-model KL divergence relative to unpermuted channels on 4 out of 4 tested models (`SmolLM2-360M`, `Qwen2.5-0.5B`, `Qwen2.5-1.5B`, `Qwen2.5-3B`), and loses to local `grp128` sorting on all 4 models.
2. **Two-Level Opposing Monotonicity (The Saddle Point):** Grouping channels into $H_4$ micro-quads rewards **monotone** homogeneity (packing adjacent magnitudes together to minimize intra-quad spread), whereas assigning quads to $g = 128$ macro-balls rewards **anti-monotone** balance (equalizing ball loads $B_g$ to hit the Cauchy-Schwarz variance floor $G \sum_g B_g^2 \ge (\sum_g B_g)^2$ and maximize the rate-distortion product $\prod_g B_g$). Composing monotone sorting *inside* micro-quads with snake/LPT interleaving *across* macro-balls (**Stratified Quad-Interleaving**, $\text{Sort-into-Quads} \circ \text{Snake-Across-Balls}$) strictly dominates both naive global contiguous sorting and mixed-quad layouts simultaneously for *every* positive trade-off weight pair $(\alpha, \beta) \in \mathbb{Z}_{>0}^2$.
3. **Cross-Layer Interference Ratio ($R_{\text{cross}}$) Reversal & BF16 Non-Associativity Obstruction:** A per-matrix router (`PerMatrix_Oracle`) that reduces the marginal single-block KL sum by $\delta = 2.4187\%$ on `Qwen2.5-3B` still *worsens* full-model KL by $+5.17\%$ because it inflates the cross-layer interference ratio $R_{\text{cross}} = F / S$ by $\gamma = 7.7790\%$, crossing the exact reversal threshold $\gamma > \delta / (1 - \delta) = 2.4787\%$. Meanwhile, offline hidden-channel permutation folding into SwiGLU `down_proj` is algebraically exact over $\mathbb{Z}$ yet strictly non-identical under finite-precision rounded accumulation for every ULP quantum $q \ge 2$ (and concretely in BF16 at scale $[256, 512)$ with $\text{ULP} = 2$), matching the observed 36-layer `Qwen2.5-3B` BF16 fold drift (`KL = 0.000827191572` nats, max logit diff `1.1250`, top-1 agreement `97.8516%`).

This formal companion delivers **21 machine-checked Lean 4.36.0 theorems** across 3 pure-`Init` modules (`0` Mathlib, `0` external imports, `0` warnings on `stdout`/`stderr`, `0` banned axioms across all 21 `#print axioms` checks), paired with a pure-$\mathbb{Z}/\mathbb{Q}$ deterministic certificate oracle (`s72b_lean4_oracle.py`), a fail-closed standalone verifier & R8 static/live Lean 4 auditor (`s72b_lean4_audit.py`), a comprehensive R5 adversarial test suite (`test_s72b_lean4_r5.py`), and the pinned cryptographic certificate (`s72b_certificate.json`).

---

## 2. Prior Literature (`N0`) vs. Companion Contributions (`N1`)

All underlying mathematical inequalities are classical `N0` foundations; this companion claims `N1` novelty strictly for their unified two-scale formulation, exact discrete threshold/phase-boundary derivations, machine-checked Lean 4.36.0 proofs without external libraries, and cryptographically bound cross-verification against frozen CEER Cycle 4 empirical measurements:

- **Cauchy (1821) / Schwarz (1885) / Jensen (1906):** Quadratic variance floor $G \sum_{g=1}^G B_g^2 \ge \left(\sum_{g=1}^G B_g\right)^2$ with equality iff all $B_g$ coincide.
- **Hardy, Littlewood, and Polya (1934):** Rearrangement inequality and Schur-convexity of symmetric convex/concave load functionals.
- **Graham (1969):** Longest-Processing-Time (LPT) list scheduling for makespan and load-variance minimization.
- **Shannon (1959) / Gersho and Gray (1992):** Block rate-distortion geometric-mean bit allocation across fixed-budget quantization groups.
- **Goldberg (1991) / IEEE 754-2008 / BFloat16:** Non-associativity of finite-precision floating-point addition and round-to-nearest-even ULP absorption.
- **Shazeer (2020):** SwiGLU elementwise gated-MLP architecture underlying hidden-channel permutation folding.

---

## 3. Complete 21-Theorem Machine-Checked Lean 4.36.0 Inventory

All 21 theorems compile cleanly under `leanprover/lean4:v4.36.0` (`lake build`: `8/8` jobs, exit code `0`, zero warnings on `stdout` and `stderr`). Every theorem was checked live with `#print axioms`; **5/21 theorems depend on zero axioms (`[]`)**, **14/21 depend only on `[propext, Quot.sound]`**, and **2/21 depend on `[propext, Classical.choice, Quot.sound]`** (`0` `sorryAx`, `0` `Lean.ofReduceBool`, `0` `Lean.trustCompiler`, `0` `Lean.ofReduceNat`).

### Module 1: `S72bFormalTheory/BallBudgetCannibalization.lean` (`7` Theorems)

| # | Theorem Name | Scope | Exact `#print axioms` | Core Formal Statement |
|---:|---|---|---|---|
| 1 | `two_ball_product_and_variance_cannibalization_universal` | Universal ($\forall a,b,c,d \in \mathbb{Z}$) | `[propext, Quot.sound]` | Exact AM-GM product gap $(a+d)(b+c) - (a+b)(c+d) = (a-c)(b-d)$ and variance gap $((a+b)^2+(c+d)^2) - ((a+d)^2+(b+c)^2) = 2(a-c)(b-d)$, with strict inequalities when $c < a \land d < b$. |
| 2 | `two_ball_product_variance_equivalence_universal` | Universal ($\forall B_0,B_1,B_0',B_1' \in \mathbb{Z}$) | `[propext, Quot.sound]` | At fixed total load $B_0 + B_1 = B_0' + B_1'$, maximizing the 2-ball product is algebraically equivalent to minimizing the 2-ball sum of squares ($2(B_0'B_1' - B_0B_1) = (B_0^2+B_1^2) - (B_0'^2+B_1'^2)$). |
| 3 | `g_ball_cauchy_schwarz_variance_floor_universal` | Universal ($\forall \text{loads} : \text{List}\,\mathbb{Z}$) | `[propext, Classical.choice, Quot.sound]` | Arbitrary-$G$ variance floor $(\sum_g B_g)^2 \le G \sum_g B_g^2$ proved inductively over lists of arbitrary length. |
| 4 | `g_ball_variance_floor_equality_iff_balanced_universal` | Universal ($\forall \text{loads} : \text{List}\,\mathbb{Z}$) | `[propext, Classical.choice, Quot.sound]` | Exact equality $G \sum_g B_g^2 = (\sum_g B_g)^2$ holds if and only if every pair of ball loads in the list is equal. |
| 5 | `ball_transfer_imbalance_penalty_universal` | Universal ($\forall B, \delta \in \mathbb{Z}$) | `[propext, Quot.sound]` | Shifting load by $\delta$ away from balanced $(B, B)$ cuts product by $\delta^2$ and inflates squared sum by $2\delta^2$. |
| 6 | `global_sort_worst_among_three_4quad_pairings_universal` | Universal ($Q_0 \ge Q_1 \ge Q_2 \ge Q_3$) | `[propext, Quot.sound]` | Across all 3 equal-cardinality 2-ball pairings of 4 ordered quads, global contiguous sort $(Q_0+Q_1, Q_2+Q_3)$ minimizes product and maximizes variance; snake $(Q_0+Q_3, Q_1+Q_2)$ is optimal. |
| 7 | `s69_four_model_global_sort_cannibalization_witness` | Exact Witness (`decide`) | `[]` (zero axioms) | Verifies exact nanonat inequalities confirming 4/4-model global-sort cannibalization and `grp128 < sort` across `SmolLM2-360M`, `Qwen2.5-0.5B`, `Qwen2.5-1.5B` (WikiText-2 & Python), and `Qwen2.5-3B`. |

### Module 2: `S72bFormalTheory/TwoLevelSaddlePoint.lean` (`7` Theorems)

| # | Theorem Name | Scope | Exact `#print axioms` | Core Formal Statement |
|---:|---|---|---|---|
| 8 | `micro_quad_monotone_dominance_16_universal` | Universal (16 ordered $h_0 \ge \dots \ge h_{15}$) | `[propext, Quot.sound]` | Swapping $h_3 \leftrightarrow h_{12}$ across quads $Q_0, Q_3$ preserves macro ball load $S_0 + S_3$ identically while strictly inflating total quad spread by $2(h_3 - h_{12})$ and shrinking $S_0^2 + S_3^2$. |
| 9 | `macro_ball_snake_beats_global_sort_16_universal` | Universal ($S_0 \ge S_1 \ge S_2 \ge S_3$) | `[propext, Quot.sound]` | Exact macro-ball dominance of `quad_snake` over global contiguous sort with exact product gap $(S_0 - S_2)(S_1 - S_3)$ and variance gap $2(S_0 - S_2)(S_1 - S_3)$. |
| 10 | `macro_ball_snake_optimal_over_all_pairings_universal` | Universal ($S_0 \ge S_1 \ge S_2 \ge S_3$) | `[propext, Quot.sound]` | `quad_snake` achieves the global minimum of macro ball variance and global maximum of macro ball product over all 3 equal-cardinality 4-quad pairings. |
| 11 | `lpt_vs_snake_phase_boundary_universal` | Universal ($S_3 > 0$) | `[propext, Quot.sound]` | Exact algebraic phase boundary between equal-cardinality `quad_snake` and greedy unbalanced `quad_lpt`: $\text{snake\_sq} - \text{lpt\_unbal\_sq} = 2 S_3 (S_0 - (S_1 + S_2))$, so `quad_snake` wins iff $S_0 \le S_1 + S_2$ and unbalanced LPT wins iff $S_0 > S_1 + S_2$. |
| 12 | `two_level_saddle_point_strict_dominance_universal` | Universal ($\forall \alpha, \beta > 0$, 16 ordered $h_i$) | `[propext, Quot.sound]` | For *every* positive trade-off pair $(\alpha, \beta)$ and every non-degenerate 16-channel ladder, Stratified Quad-Interleaving strictly lowers $\alpha \cdot \text{spread} + \beta \cdot \sum B_g^2$ relative to both global sort and mixed-quad layouts simultaneously. |
| 13 | `opposing_monotonicity_sign_inversion_universal` | Universal ($c < a \land d < b$) | `[propext, Quot.sound]` | Proves exact sign inversion (`micro_gain > 0`, `macro_balance_gain < 0`, `micro_gain + macro_balance_gain = 0`) between monotone micro-quad packing and anti-monotone macro-ball packing. |
| 14 | `two_level_16channel_exact_witness` | Exact Witness (`decide`) | `[]` (zero axioms) | Exact 16-channel heavy-tail witness verifying spreads (`33 < 201 < 366`), macro products (`23552 < 60152 < 66392`), and macro squared sums (`174500 < 178580 < 191060 < 264260`). |

### Module 3: `S72bFormalTheory/CrossLayerInterferenceAndFloatObstruction.lean` (`7` Theorems)

| # | Theorem Name | Scope | Exact `#print axioms` | Core Formal Statement |
|---:|---|---|---|---|
| 15 | `rcross_amplification_threshold_universal` | Universal ($\forall D, d, G, g \in \mathbb{Z}$) | `[propext, Quot.sound]` | Exact cross-layer threshold identity $(D-d)(G+g) - DG = g(D-d) - dG$: full-model KL worsens iff $\gamma > \delta / (1 - \delta)$ and improves iff $\gamma < \delta / (1 - \delta)$. |
| 16 | `rcross_full_kl_reversal_universal` | Universal ($S_{\text{route}} > 0$) | `[propext, Quot.sound]` | Exact reversal decomposition $S_{\text{route}}(F_{\text{route}} - F_{\text{base}}) = (F_{\text{route}} S_{\text{base}} - F_{\text{base}} S_{\text{route}}) - F_{\text{route}}(S_{\text{base}} - S_{\text{route}})$ with biconditional characterization of $F_{\text{base}} < F_{\text{route}}$. |
| 17 | `s69_qwen3b_rcross_spike_and_projtype_cure_witness` | Exact Witness (`decide`) | `[]` (zero axioms) | Exact nanonat verification of `PerMatrix_Oracle` reversal vs `A16_BlockH4` and `A16_axis` (`> +6.32%` $R_{\text{cross}}$ spike), `ProjType_Uniform` `-6.57%` Python win over `E8_H8`, and `Coupled_RelRatio_ZeroCal` `-10.62%` cut vs `PerMatrix_ZeroCal` within `+0.10%` of `E8_H8`. |
| 18 | `rounded_fp_addition_non_associative_universal` | Universal ($\forall q \in \mathbb{Z}, q \ge 2$) | `[propext, Quot.sound]` | Exact integer addition satisfies $(q+1)+(q-1) = q+(1+(q-1)) = 2q$, whereas quantum-truncating `roundAdd q` produces $q$ vs $2q$ for every ULP quantum $q \ge 2$. |
| 19 | `swiglu_perm_non_commutation_with_rounded_sum_universal` | Universal ($\forall q \in \mathbb{Z}, q \ge 2$) | `[propext, Quot.sound]` | Exact $\mathbb{Z}$-summation is invariant under cyclic permutation `[q, 1, q - 1] ~> [1, q - 1, q]`, yet `roundFoldLeft q` yields $q \ne 2q$ for every $q \ge 2$. |
| 20 | `bf16_exact_ulp_absorption_witness` | Exact Witness (`decide`) | `[]` (zero axioms) | Exact IEEE-754 BF16 round-to-nearest-even witness at scale $[256, 512)$ ($\text{ULP} = 2$): `bf16(bf16(256, 1), 1) = 256 != 258 = bf16(256, bf16(1, 1))`. |
| 21 | `s69_bf16_36layer_down_fold_obstruction_witness` | Exact Witness (`decide`) | `[]` (zero axioms) | Exact witness for the 36-layer `Qwen2.5-3B` BF16 offline `down_proj` fold check (`kl = 827,191,572` piconats $> 0$, max logit diff $9/8 > 1$, mean diff `36,161,713` nano $> 0$, top-1 agreement `978,516` ppm $< 10^6$). |

---

## 4. Certified Exact Quantitative Witnesses

All rational ratios below are stored canonically in `s72b_certificate.json`; decimal displays are programmatically rounded via `decimal.Decimal(..., ROUND_HALF_UP)` at 60-digit internal precision.

### 4.1 Pillar 1: Group-Ball Budget-Transfer Cannibalization

#### 4-Model Empirical Anchor (`rtx3080_s69_empirical_results.json`, SHA-256 `05175f89...`)

| Model (Eval Split) | `real` (nats) | `grp128` (nats) | `sort` (nats) | Exact `sort / real` ($\mathbb{Q}$) | `sort` vs `real` | Exact `grp128 / sort` ($\mathbb{Q}$) | `grp128` vs `sort` |
|---|---:|---:|---:|---|---:|---|---:|
| `SmolLM2-360M` (WikiText-2) | `0.028207` | `0.030636` | `0.045766` | `22882975/14103443` | `+62.25%` (`1.6225x`) | `6127159/9153190` | `-33.06%` (`0.6694x`) |
| `Qwen2.5-0.5B` (WikiText-2) | `0.029413` | `0.027190` | `0.032504` | `32504237/29412649` | `+10.51%` (`1.1051x`) | `27189940/32504237` | `-16.35%` (`0.8365x`) |
| `Qwen2.5-1.5B` (WikiText-2) | `0.023412` | `0.023176` | `0.026787` | `8928909/7803866` | `+14.42%` (`1.1442x`) | `23176037/26786727` | `-13.48%` (`0.8652x`) |
| `Qwen2.5-3B` (WikiText-2) | `0.024774` | `0.024918` | `0.026025` | `3253180/3096811` | `+5.05%` (`1.0505x`) | `1557351/1626590` | `-4.26%` (`0.9574x`) |

On `Qwen2.5-1.5B`, local `grp128` also beats `h2` on both WikiText-2 (`23,176,037` vs `23,281,449` nanonats; exact ratio `23176037/23281449`, `-0.45%`) and Python (`14,083,191` vs `14,356,267` nanonats; exact ratio `14083191/14356267`, `-1.90%`, while `real = 15,128,765` and `sort = 17,099,112` nanonats).

#### Canonical Exact Structural Witnesses

- **2-Ball Witness $(a, b, c, d) = (100, 80, 20, 10)$:** Global contiguous sort $(180, 30)$ gives product `5,400` and squared sum `33,300`, whereas balanced snake $(110, 100)$ gives product `11,000` (`+5,600`, `2.0370x`, `+103.70%`) and squared sum `22,100` (variance gap `11,200`; global sort inflates squared load by `1.5068x`, `+50.68%`).
- **4-Ball Cauchy-Schwarz Witness $[100, 40, 12, 4]$:** $4 \sum_g B_g^2 = 47,040$ vs $(\sum_g B_g)^2 = 24,336$ (`excess = 22,704`; exact variance ratio `980/507` = `1.9329x`, `+93.29%` above the balanced `[39, 39, 39, 39]` floor where `excess = 0`).

### 4.2 Pillar 2: Two-Level Saddle-Point & LPT-vs-Snake Phase Boundary

| Regime (16-Channel Ladder) | Quad Sums $[S_0, S_1, S_2, S_3]$ | Layout | Macro $\sum B_g^2$ | vs Global Sort | Excess above $S^2 / 2$ |
|---|---|---|---:|---:|---:|
| Heavy-Tail ($S_0 = 376 > S_1+S_2 = 172$) | `[376, 136, 36, 10]` ($S = 558$, $S^2/2 = 155,682$) | `quad_lpt` (unbalanced 1+3) | `174,500` | `-33.97%` | `18,818` |
| Heavy-Tail ($S_0 = 376 > S_1+S_2 = 172$) | `[376, 136, 36, 10]` ($S = 558$, $S^2/2 = 155,682$) | `quad_snake` (balanced 2+2) | `178,580` | `-32.42%` | `22,898` |
| Heavy-Tail ($S_0 = 376 > S_1+S_2 = 172$) | `[376, 136, 36, 10]` ($S = 558$, $S^2/2 = 155,682$) | `quad_alt` (2+2 alternating) | `191,060` | `-27.70%` | `35,378` |
| Heavy-Tail ($S_0 = 376 > S_1+S_2 = 172$) | `[376, 136, 36, 10]` ($S = 558$, $S^2/2 = 155,682$) | `global_sort` (contiguous) | `264,260` | `0.00%` | `108,578` |
| Moderate-Tail ($S_0 = 148 \le S_1+S_2 = 176$) | `[148, 108, 68, 28]` ($S = 352$, $S^2/2 = 61,952$) | `quad_snake` (balanced 2+2) | `61,952` | `-17.12%` | `0` |
| Moderate-Tail ($S_0 = 148 \le S_1+S_2 = 176$) | `[148, 108, 68, 28]` ($S = 352$, $S^2/2 = 61,952$) | `quad_lpt` (unbalanced 1+3) | `63,520` | `-15.03%` | `1,568` |
| Moderate-Tail ($S_0 = 148 \le S_1+S_2 = 176$) | `[148, 108, 68, 28]` ($S = 352$, $S^2/2 = 61,952$) | `global_sort` (contiguous) | `74,752` | `0.00%` | `12,800` |

- **Micro-Scale Spread Ordering (Heavy-Tail Witness):** `spread_mono = 33 < spread_mix_swap = 201 < spread_strided = 366`.
- **Macro-Scale Product Ordering (Heavy-Tail Witness):** `prod_glob = 23,552 < prod_alt = 60,152 < prod_snake = 66,392` (`prod_snake / prod_glob = 8299/2944` = `2.8190x`).
- **Exact LPT-vs-Snake Phase Boundary:** On the heavy-tail ladder, $S_0 = 376 > S_1 + S_2 = 172$, so `quad_lpt` improves ball squared load over `quad_snake` by exact slack $2 \cdot 10 \cdot (376 - 172) = 4,080$. On the moderate-tail ladder, $S_0 = 148 \le S_1 + S_2 = 176$, so equal-cardinality `quad_snake` hits the exact zero-excess Cauchy-Schwarz floor `61,952` and beats the 1+3 split by $2 \cdot 28 \cdot (176 - 148) = 1,568$.
- **Simultaneous Saddle Dominance:** Verified both universally in Lean (`Theorem 12`, $\forall \alpha, \beta > 0$) and explicitly across $(\alpha, \beta) \in \{(1, 1), (10, 1), (1, 10), (1000, 1), (1, 1000)\}$.

### 4.3 Pillar 3: Cross-Layer $R_{\text{cross}}$ Threshold Reversal & BF16 Non-Associativity

#### `Qwen2.5-3B` Cross-Layer Interference Ratio Table (WikiText-2, `05175f89...`)

| Routing Arm | Full-Model KL (nats) | 36-Block Marginal Sum (nats) | Exact $R_{\text{cross}} = F / S$ ($\mathbb{Q}$) | Decimal $R_{\text{cross}}$ |
|---|---:|---:|---|---:|
| `A16_axis` | `0.046686` (`46,686,444` nkl) | `0.084941` (`84,941,117` nkl) | `46686444/84941117` | `0.549633` |
| `A16_BlockH4` | `0.047942` (`47,941,673` nkl) | `0.088417` (`88,416,614` nkl) | `3687821/6801278` | `0.542225` |
| `PerMatrix_Oracle` | `0.050421` (`50,421,289` nkl) | `0.086278` (`86,278,044` nkl) | `50421289/86278044` | `0.584405` |
| `ProjType_Uniform` | `0.046336` (`46,336,085` nkl) | `0.085763` (`85,762,537` nkl) | `46336085/85762537` | `0.540284` |

- **Exact Threshold Reversal (`PerMatrix_Oracle` vs `A16_BlockH4`):**
  - Marginal sum reduction: $\delta = 1069285/44208307$ (`2.4187%`)
  - Exact reversal threshold: $\delta / (1 - \delta) = 1069285/43139022$ (`2.4787%`)
  - Observed $R_{\text{cross}}$ amplification: $\gamma = 12375610552609/159088991251062$ (`7.7790%` $> 2.4787\%$)
  - Net full-model KL multiplier: $(1 - \delta)(1 + \gamma) = 50421289/47941673$ (`1.051722x`, **`+5.17%`** net worsening despite lowering the marginal sum!).
- **`PerMatrix_Oracle` vs `A16_axis`:** $R_{\text{cross}}$ ratio `4282840608239813/4028015069635536` (`1.063263x`, `+6.33%`), full KL ratio `50421289/46686444` (`1.079998x`, `+8.00%`).
- **Uniform & Coupled-Routing Cures:**
  - `ProjType_Uniform` vs `E8_H8` on Python: `34,327,497` vs `36,741,460` nanonats (`-2413963/36741460`, **`-6.57%`**).
  - `Coupled_RelRatio_ZeroCal` vs `PerMatrix_ZeroCal` on WikiText-2: `45,767,877` vs `51,205,992` nanonats (`-1812705/17068664`, **`-10.62%`**).
  - `Coupled_RelRatio_ZeroCal` vs `E8_H8` on WikiText-2: `45,767,877` vs `45,723,724` nanonats (`+44153/45723724`, **`+0.10%`**).
- **Finite-Precision & BF16 Obstruction:**
  - Verified exact non-associativity `roundFoldLeft q [q, 1, q - 1] = q != 2 * q = roundFoldLeft q [1, q - 1, q]` across quanta $q \in \{2, 4, 8, 16, 256\}$ (and proved universally $\forall q \ge 2$).
  - Concrete BF16 ULP=2 (`[256, 512)`): `bf16(bf16(256, 1), 1) = 256`, `bf16(256, bf16(1, 1)) = 258` (exact gap `2`).
  - Empirical 36-layer `Qwen2.5-3B` offline `down_proj` fold (`05175f89...`): `layers_folded = 36`, `exact_identity = false`, `kl_fold_vs_ref = 0.000827191572` nats (`827,191,572` piconats), `max_abs_bf16_logit_diff = 9/8 = 1.1250`, `mean_abs_bf16_logit_diff = 0.036161713`, `top1_agreement = 97.8516%`.

---

## 5. Cryptographic Certificate, Security Hardening & Reproducibility

### 5.1 Pinned Certificate Digests

- **`payload_sha256`:** `58a7345d82b369504c274607c9614aac516bcf46a8fc75f59e977207bca6bbbe`
- **`certificate_sha256`:** `5617abd09a6c294b27dc6ec9f36dda47377f62a7fff015f494e136d0aa2fe1ed`
- **`s72b_certificate.json` file SHA-256:** `223a32a4e2550ed90e3298c235bb9e9421778a353b03568b053c87656f437015` (`6,567` bytes, mode `0o600`, canonical JSON + trailing `\n`, well under `TARGET_MAX_CERT_BYTES = 12,288`)

### 5.2 Hardening Properties of `s72b_lean4_oracle.py` & `s72b_lean4_audit.py`

- **Zero-Float Trust Path:** Iterative strict JSON tree validator rejects `float` and `float` subclasses (`isinstance(cur, float)`), non-exact container/scalar types (`bool` as `int`, `tuple`, `dict`/`list` subclasses), lone UTF-16 surrogates, nesting $> 32$, integers $> 1,000$ pre-parse digits or $> 3,321$ bits, and duplicate JSON keys.
- **Fail-Closed Filesystem I/O:** Atomic `dir_fd` `O_NOFOLLOW` ancestor walk (with root-owned prefix normalization for macOS temp paths), leaf `O_RDONLY | O_NOFOLLOW | O_NONBLOCK`, `os.fstat(fd)` `S_ISREG` + `st_nlink <= 1` check, streaming `65,536`-byte reads capped at `1,048,576` bytes, and durable `0o600` temp-file + `os.fsync` + `os.replace` writes with cryptographically random temporary filenames.
- **R8 Static & Live Lean 4 Auditor (`--verify-lean`):** Comment/string-aware lexer detecting `sorry`, `admit`, `axiom`, `opaque`, `unsafe`, `partial`, `implemented_by`, `extern`, `native_decide`, multi-flag `decide +native` / `decide +revert +native`, `set_option ... debug.skipKernelTC`, `Lean.ofReduceBool`, `Lean.ofReduceNat`, `Lean.trustCompiler`, `mkSorry`, `addDecl`, `Environment.addDecl`, `#eval`, `#reduce`, and `import`; live `lean` kernel execution rejecting any non-zero exit code, any `stderr` output, any `stdout` compiler warning, any missing/extra theorem, or any axiom outside each theorem's pinned allowlist.
- **Field-Complete Secondary Audit (`secondary_audit_payload`):** Enforces exact keysets on every nested object and independently cross-checks every rational ratio, integer witness, empirical anchor hash, module SHA-256, and per-theorem axiom profile without trusting cached summary booleans.

### 5.3 Reproducibility Commands

Run inside the directory containing this report (with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1`):

```bash
# 1. Self-check oracle & certificate parity (R4 3-Mode + PYTHONHASHSEED=42 -B)
python3 -I -B s72b_lean4_audit.py
python3 -I -O -B s72b_lean4_audit.py
python3 -I -B -W error s72b_lean4_audit.py
PYTHONHASHSEED=42 python3 -B s72b_lean4_audit.py

# 2. Verify committed certificate (full recompute + field-complete secondary audit)
python3 -I -B s72b_lean4_audit.py --check s72b_certificate.json --json

# 3. Run R8 static Lean 4 scanner + live Lean 4.36.0 kernel & #print axioms verification
python3 -I -B s72b_lean4_audit.py --verify-lean . --json

# 4. Run full R5 adversarial & live-kernel test suite (R4 3-Mode + PYTHONHASHSEED=42 -B)
python3 -I -B test_s72b_lean4_r5.py -v
python3 -I -O -B test_s72b_lean4_r5.py
python3 -I -B -W error test_s72b_lean4_r5.py
PYTHONHASHSEED=42 python3 -B test_s72b_lean4_r5.py

# 5. Clean scratch Lake build (preserves zero .lake build artifacts in source tree)
python3 -I -B -c "import pathlib, shutil, subprocess, tempfile; d = tempfile.TemporaryDirectory(); p = pathlib.Path(d.name); [shutil.copy2(f, p / f) for f in ('lean-toolchain', 'lakefile.toml')]; shutil.copytree('S72bFormalTheory', p / 'S72bFormalTheory'); subprocess.run(['lake', 'build'], cwd=p, check=True); d.cleanup()"
```

---

## 6. Frozen 9-File Deliverable Manifest

Every executable, formal, test, and certificate artifact in this bundle is pinned below (this report is the 10th file in the directory):

| Relative Path | Lines | Bytes | Mode | SHA-256 |
|---|---:|---:|---|---|
| `lean-toolchain` | `1` | `25` | `0o644` | `582a63f8f9c5128067b688778c1051cf1464c3ca9d57b9325025c051c29b774a` |
| `lakefile.toml` | `11` | `288` | `0o644` | `fbfe16a21ba9cf70dad6317e82924a574b4570dcf591ee2759d07ebcb20e8f4a` |
| `S72bFormalTheory/BallBudgetCannibalization.lean` | `278` | `13339` | `0o644` | `4a16b1092a754a6d5742c630f47ea6b743df4da1542de87b0594fe390904b41f` |
| `S72bFormalTheory/TwoLevelSaddlePoint.lean` | `265` | `14406` | `0o644` | `b4432a684b0e1bf4a37c41770a43228c358cb0d7572e802c382ad94b0373e26b` |
| `S72bFormalTheory/CrossLayerInterferenceAndFloatObstruction.lean` | `194` | `9830` | `0o644` | `ed889b41f96aa912468a69729bf0b9c41de8804c57cdfea969e0ed2218331992` |
| `s72b_lean4_oracle.py` | `660` | `27181` | `0o644` | `468519357c3dfff687b7df43cd7a814ec6c9c2bac7e6e247bfe4fcea6bdc4a28` |
| `s72b_lean4_audit.py` | `1257` | `56235` | `0o644` | `f10541c1dcd7801895c8f1ef660ed08441c734f70418059d13b0debe71d02847` |
| `test_s72b_lean4_r5.py` | `1181` | `66295` | `0o644` | `0e31e95e41e9374f2f1e341c824d873dc2a881d2d4de4b8dc28396a1cfb2bfc4` |
| `s72b_certificate.json` | `1` | `6567` | `0o600` | `223a32a4e2550ed90e3298c235bb9e9421778a353b03568b053c87656f437015` |
