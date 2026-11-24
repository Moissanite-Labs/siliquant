# Heavy-Tail Greedy-LPT Starvation, Load-Power Compression Phase Theory, and Low-Bitrate Pareto Dominance: Machine-Checked Lean 4.36.0 Formal Companion

**Project ID:** `CEER-HEAVY-TAIL-LPT-STARVATION-LEAN4`  
**Certificate Schema:** `aletheia.ceer_heavy_tail_lpt_starvation_lean4.certificate.v1`  
**Target Repository:** `Moissanite-Labs/siliquant` (`Refs #77, #85, #87, #90, #91, #92, #93, #94, #95`)  
**Toolchain:** `leanprover/lean4:v4.36.0` (pure `Init`, zero Mathlib, zero external imports, zero compiler warnings)  
**Novelty Tier & Scope:** `N1` (constructive application and unified machine-checked formalization of classical `N0` scheduling, rearrangement, Pythagorean, rate-distortion, and Pareto inequalities to cardinality-capped heavy-tail greedy-LPT starvation, concave $p$-exponent load compression, three-regime phase trichotomy, low-bitrate gap amplification, and 4/4-model Pareto-corner domination in block-Hadamard quantization)  
**Oracle Class:** `exact` (pure integer and rational arithmetic over $\mathbb{Z}$ and $\mathbb{Q}$, zero IEEE-754 floats on trust paths, zero bare `assert`s, zero mathematical RNG)  
**Hardware Execution Scope:** `measured_live_gpu_in_this_bundle = false` (pure symbolic Lean 4.36.0 theorem proving and pure-$\mathbb{Z}/\mathbb{Q}$ Python certification; empirical constants are read-only anchored to the frozen empirical artifact `rtx3080_s72_empirical_results.json`, SHA-256 `a7c35c646a9c29fa645f207697c74e181ad19d8ce61acbe489a265a132968647`, and `rtx3080_s72_certificate.json`, file SHA-256 `c2dc4626271503bbd710ce468e2d2051b1241f77d9f6f995dee15d621eb884e7`, inner `certificate_sha256` `7eb5333739ccdef1412a9f82eb12c6be8e08832ac275a2e9ea6cbdba34402f99`).

---

## 1. Executive Summary

Empirical measurements in CEER Cycle 5 (`Moissanite-Labs/siliquant#95`, anchored to `a7c35c64...`) uncovered three sharp structural laws governing Stratified Quad-Interleaving (`Sort-into-Quads` $\circ$ `Interleave-Across-Balls`) for INT4/FP4 group-quantized transformer projections with $H_4$ micro-block Hadamard mixing and $g = 128$ macro-ball scaling:

1. **Heavy-Tail Greedy-LPT Starvation & Exact Cures (`Qwen2.5-3B`):** On `Qwen2.5-3B` WikiText-2, linear ($p = 1$) cardinality-capped greedy-LPT (`quad_lpt`, `0.026776` nats) paradoxically *underperforms* naive global contiguous sorting (`0.026025` nats, `+2.89%`, exact `2677639/2602544`). When a single outlier quad satisfies $w_0 > w_1 + w_2$, cardinality-capped greedy-LPT starves Ball 0 until Ball 1 hits its slot cap, forcing Ball 1 to swallow the entire secondary core $(w_1, w_2, w_3)$ while Ball 0 receives only bottom-tail dust $(w_4, w_5)$. Under true $L_2$-ball quantization energy $E_g = \sum_{i \in g} \sqrt{w_i}$, this starvation inversion strictly lowers the macro product $\prod_g E_g$ and inflates $\sum_g E_g^2$ relative to equal-cardinality `quad_snake` (and strictly inflates secondary-ball load above global sort). Both `quad_snake` (`0.023154` nats on WikiText-2, `-4.24%` vs `BlockH2` and `-6.54%` vs `BlockH4`) and concave `quad_lpt_sqrt` ($p = 1/2$: `0.025296` nats on WikiText-2, `-5.53%` vs linear LPT; `0.019520` nats on Python, `-15.04%` vs `BlockH2`, breaking the `0.020` nats barrier) cure the starvation trap.
2. **Concave $p$-Exponent Load Compression & 3-Regime Phase Trichotomy:** Compressing quad weights from linear $w_i = x_i^2$ ($p = 1$) to square-root $x_i$ ($p = 1/2$) strictly shrinks every outlier ratio by $x_0 / x_1 > 1$ and expands the non-starvation threshold from the Euclidean sum of squares $x_1^2 + x_2^2$ to the $L_1$ square $(x_1 + x_2)^2$ by the exact Pythagorean slack $2 x_1 x_2 > 0$ (extending universally to arbitrary $k \ge 2$ secondary quads via $(\sum x_i)^2 \ge \sum x_i^2 + 2 x_1 x_2$). Across outlier steepness, the three policies cleanly partition into **Regime I** (mild tail: `p = 1` LPT beats `p = 0` snake, matching `Qwen2.5-0.5B` `-13.71%` and `Qwen2.5-1.5B` `-0.92%`), **Regime II** (intermediate Pythagorean basin $x_1^2 + x_2^2 < x_0^2 < (x_1 + x_2)^2$: `p = 1/2` LPT strictly beats *both* `p = 1` LPT and `p = 0` snake), and **Regime III** (extreme outlier $x_0 > x_1 + x_2$: *both* `p = 1` and `p = 1/2` starve while `p = 0` `quad_snake` remains unconditionally starvation-free, matching `Qwen2.5-3B` WikiText-2 where snake cuts linear LPT by `-13.53%`).
3. **Low-Bitrate Exponential Gap Amplification & 4/4-Model Pareto-Corner Dominance:** Under high-rate distortion scaling $D(b) = C \cdot 2^{-\alpha b}$, stepping down in bitrate by $\Delta b > 0$ preserves relative ordering and exact ratios $C_1 / C_2$ while amplifying absolute KL separations by $M = 2^{\alpha \Delta b} > 1$ (for $\alpha = 2$, moving from `4.5 bpw` to `3.0 bpw` multiplies the `Qwen2.5-3B` `BlockH2 - quad_snake` gap `8x` from `1,024,654` to `8,197,232` nanonats). Across 4/4 tested models (`SmolLM2-360M`, `Qwen2.5-0.5B`, `Qwen2.5-1.5B`, `Qwen2.5-3B`) and both Hessian (`h_real`) and zero-text weight-column (`wcol`) metrics, Stratified Quad-Interleaving occupies the strict Pareto corner simultaneously reducing micro-quad spread (up to `6.73x` on `h_real` and `120.89x` on `wcol` for `Qwen2.5-3B`) and macro-ball CV (up to `2.68x` on `h_real` and `66.70x` on `wcol`), shattering `BlockH2` on 3/3 Qwen models (`-3.76%`, `-1.41%`, `-4.24%`) and winning zero-text (`quad_lpt_rand` `-2.90%` vs `BlockH4` / `-4.83%` vs `BlockH2` on Python; `quad_snake_wcol` `-11.67%` vs `global_sort_wcol` on WikiText-2).

This formal companion delivers **21 machine-checked Lean 4.36.0 theorems** across 3 pure-`Init` modules (`0` Mathlib, `0` external imports, `0` warnings on `stdout`/`stderr`, `0` banned axioms across all 21 `#print axioms` checks), paired with a pure-$\mathbb{Z}/\mathbb{Q}$ deterministic certificate oracle (`s73b_lean4_oracle.py`), a fail-closed standalone verifier and R9 static/live Lean 4 auditor (`s73b_lean4_audit.py`), a comprehensive R5 adversarial test suite (`test_s73b_lean4_r5.py`), and the pinned cryptographic certificate (`s73b_certificate.json`).

---

## 2. Prior Literature (`N0`) vs. Companion Contributions (`N1`)

All underlying mathematical primitives are classical `N0` foundations; this companion claims `N1` novelty strictly for their unified heavy-tail cardinality-capped scheduling formulation, exact Pythagorean compression basin and three-regime phase trichotomy, low-bitrate gap scaling and Pareto-corner formalization, machine-checked Lean 4.36.0 proofs without external libraries, and cryptographically bound cross-verification against frozen CEER Cycle 5 empirical measurements:

- **Graham (1969):** Longest-Processing-Time (LPT) list scheduling for makespan and load balancing.
- **Coffman, Garey, and Johnson (1978):** Multifit and cardinality-constrained bin-packing / scheduling.
- **Hardy, Littlewood, and Polya (1934):** Rearrangement inequality and power-mean / concave compression.
- **Cauchy (1821) / Schwarz (1885) / Pythagoras:** $L_1$-vs-$L_2$ squared norm gap $(\sum x_i)^2 - \sum x_i^2 = 2 \sum_{i < j} x_i x_j$.
- **Shannon (1959) / Gersho and Gray (1992):** Exponential high-rate distortion scaling $D(b) = C \cdot 2^{-\alpha b}$.
- **Pareto (1896):** Multi-objective dominance over competing micro-spread and macro-variance coordinates.

---

## 3. Complete 21-Theorem Machine-Checked Lean 4.36.0 Inventory

All 21 theorems compile cleanly under `leanprover/lean4:v4.36.0` (`lake build`: `8/8` jobs, exit code `0`, zero warnings on `stdout` and `stderr`). Every theorem was checked live with `#print axioms`; **5/21 theorems depend on zero axioms (`[]`)**, **15/21 depend only on `[propext, Quot.sound]`**, and **1/21 depends on `[propext, Classical.choice, Quot.sound]`** (`0` `sorryAx`, `0` `Lean.ofReduceBool`, `0` `Lean.trustCompiler`, `0` `Lean.ofReduceNat`).

### Module 1: `S73bFormalTheory/HeavyTailLPTStarvation.lean` (`7` Theorems)

| # | Theorem Name | Scope | Exact `#print axioms` | Core Formal Statement |
|---:|---|---|---|---|
| 1 | `single_outlier_greedy_lpt_starvation_universal` | Universal ($\forall \text{sec} : \text{List}\,\mathbb{Z}, w_0, b_1$) | `[propext, Quot.sound]` | Arbitrary-length list induction: whenever all secondary weights are nonnegative and $b_1 + \sum \text{sec} < w_0$, greedy LPT assigns *every* secondary weight to Ball 1 (`lptRun w0 b1 sec = (w0, b1 + sumList sec)`), completely starving Ball 0. |
| 2 | `cardinality_capped_lpt_6quad_starvation_universal` | Universal ($w_0 \ge \dots \ge w_5 > 0$, $w_1 + w_2 < w_0$) | `[propext, Quot.sound]` | Exact 6-quad, 2-ball (`cap = 3`) symbolic execution of `capLpt6`: once $w_1 + w_2 < w_0$, Ball 1 reaches capacity at step 3 with $(w_1, w_2, w_3)$ and Ball 0 is forced to absorb the bottom dust $(w_4, w_5)$, yielding `(w0 + w4 + w5, w1 + w2 + w3)`. |
| 3 | `lpt_starvation_vs_snake_l2_energy_product_variance_universal` | Universal ($\forall x_0, \dots, x_5 \in \mathbb{Z}$) | `[propext, Quot.sound]` | Exact $L_2$-energy product and variance identities between starved linear-LPT `(x0+x4+x5, x1+x2+x3)` and `quad_snake` `(x0+x3+x4, x1+x2+x5)`: `prod_snk - prod_lpt = (x3 - x5) * ((x1 + x2) - (x0 + x4))` and `sq_lpt - sq_snk = 2 * (x3 - x5) * ((x1 + x2) - (x0 + x4))`, strictly favoring snake whenever $x_5 < x_3$ and $x_0 + x_4 < x_1 + x_2$. |
| 4 | `three_ball_cardinality_lpt_secondary_overpack_universal` | Universal ($w_7 \le w_5, w_8 \le w_6$) | `[propext, Quot.sound]` | Exact 9-quad, 3-ball (`cap = 3`) conservation and secondary-overpack identity: starving Ball 0 pushes exact mass $(w_5 - w_7) + (w_6 - w_8) \ge 0$ out of Ball 0 onto the secondary balls $(B_1, B_2)$ relative to 3-ball snake, strictly positive whenever $w_7 < w_5$ or $w_8 < w_6$. |
| 5 | `lpt_starvation_worse_than_global_sort_secondary_load_universal` | Universal ($x_4 < x_1, x_5 < x_2, x_5 < x_3$) | `[propext, Quot.sound]` | Proves the strict secondary-ball load inversion $E_{1,\text{glob}} < E_{1,\text{snk}} < E_{1,\text{lpt}}$ (`x3+x4+x5 < x1+x2+x5 < x1+x2+x3`), explaining why saturated outlier Ball 0 plus overpacked secondary Ball 1 lets starved linear-LPT score worse than naive global contiguous sort. |
| 6 | `heavy_tail_6quad_starvation_exact_witness` | Exact Witness (`decide`) | `[]` (zero axioms) | Exact 6-quad witness $x = [10, 8, 5, 4, 2, 1]$, $w = x^2 = [100, 64, 25, 16, 4, 1]$: `capLpt6` hits balanced linear sum `(105, 105)` yet starved $L_2$ loads `(13, 17)` (`prod = 221 < 224`, `sq = 458 > 452`, `peak = 17 > 16`) lose to snake `(16, 14)` (the companion 3-ball numeric witness `[200, 30, 28, 25, 22, 20, 18, 2, 1]` instantiating Theorem 4 is certified in `s73b_certificate.json` §4.1). |
| 7 | `s72_qwen3b_lpt_starvation_and_cure_witness` | Exact Witness (`decide`) | `[]` (zero axioms) | Verifies exact nanonat empirical ladders on `Qwen2.5-3B` WikiText-2 (`snake < H2 < H4 < lpt_sqrt < sort < lpt_lin`, integer bounds `> 5.5%` sqrt-LPT cure vs linear LPT and `> 4.2%` snake vs `BlockH2`) and Python (`lpt_sqrt < 20,000,000 < snake < rand < H4 < H2`, `> 15.0%` sqrt-LPT vs `BlockH2`; exact rational `%` deltas certified in `s73b_certificate.json` §4.1). |

### Module 2: `S73bFormalTheory/LoadPowerCompressionPhase.lean` (`7` Theorems)

| # | Theorem Name | Scope | Exact `#print axioms` | Core Formal Statement |
|---:|---|---|---|---|
| 8 | `sqrt_load_ratio_compression_universal` | Universal ($x_0, x_1 > 0$) | `[propext, Quot.sound]` | Exact cross-multiplied outlier-ratio compression identity $x_0^2 x_1 - x_0 x_1^2 = x_0 x_1 (x_0 - x_1)$: square-root load ratio $x_0 / x_1$ is strictly smaller than linear ratio $x_0^2 / x_1^2$ if and only if $x_1 < x_0$. |
| 9 | `pythagorean_non_starvation_basin_expansion_universal` | Universal ($x_0, x_1, x_2 > 0$) | `[propext, Quot.sound]` | Exact Pythagorean expansion $(x_1 + x_2)^2 - (x_1^2 + x_2^2) = 2 x_1 x_2 > 0$: for every triple in the open basin $x_1^2 + x_2^2 < x_0^2 < (x_1 + x_2)^2$, linear ($p = 1$) LPT starves while square-root ($p = 1/2$) LPT avoids starvation ($x_0 < x_1 + x_2$). |
| 10 | `general_k_secondary_pythagorean_slack_universal` | Universal ($\forall \text{xs} : \text{List}\,\mathbb{Z}$, $a, b > 0$) | `[propext, Classical.choice, Quot.sound]` | Arbitrary-$k$ list induction proving $\sum x_i^2 \le (\sum x_i)^2$ for all nonnegative lists and strict slack $\sum x_i^2 + 2 a b \le (\sum x_i)^2$ whenever at least two positive secondary quads $a, b$ exist. |
| 11 | `phase_regime1_mild_tail_lpt_beats_snake_universal` | Universal ($\forall s > 0$, scaled `[8,6,6,5,3,1]`) | `[propext, Quot.sound]` | **Regime I (Mild Tail):** For every positive scale $s > 0$, proves the non-starvation decision inequalities $x_0^2 \le x_1^2 + x_2^2 < x_0^2 + x_3^2$ (with `capLpt6 64 36 36 25 9 1 = (90, 81)` executed at $s = 1$ in Theorem 14) and proves linear LPT strictly beats `quad_snake` in $L_2$ load variance (`421s^2 < 425s^2`), product (`210s^2 > 208s^2`), and peak load (`15s < 16s`). |
| 12 | `phase_regime2_intermediate_sqrt_beats_both_universal` | Universal ($\forall s > 0$, scaled `[10,8,5,5,3,1]`) | `[propext, Quot.sound]` | **Regime II (Intermediate Pythagorean Basin):** For every $s > 0$, $p = 1$ starves (`520s^2`), $p = 0$ snake is slightly off-balance (`520s^2`), while $p = 1/2$ `quad_lpt_sqrt` avoids starvation and hits the exact Cauchy-Schwarz floor `(16s, 16s)` (`512s^2 < 520s^2`, `256s^2 > 252s^2`). |
| 13 | `phase_regime3_extreme_tail_snake_unconditionally_starvation_free_universal` | Universal ($x_1 + x_2 < x_0$, $x_5 < x_3$) | `[propext, Quot.sound]` | **Regime III (Extreme Tail):** When $x_0 > x_1 + x_2$, *both* $p = 1$ and $p = 1/2$ trigger full Ball 0 starvation (`capLpt6` places $(x_1, x_2, x_3)$ in Ball 1), whereas `quad_snake` unconditionally assigns $(0, 3, 4)$ vs $(1, 2, 5)$ and strictly lowers secondary-ball load by $x_3 - x_5 > 0$. |
| 14 | `load_power_trichotomy_exact_witness` | Exact Witness (`decide`) | `[]` (zero axioms) | Exact unit-scale ($s = 1$) verification of Regimes I (`421 < 425`), II (`512 < 520`), and III (`[20,8,5,4,2,1]`, `17 > 14`) paired with the empirical cross-scale phase flip across `Qwen2.5-0.5B` (`lpt < snake`, `-13.71%`), `Qwen2.5-1.5B` (`lpt < snake`, `-0.92%`), and `Qwen2.5-3B` (`snake < lpt`, `-13.53%`). |

### Module 3: `S73bFormalTheory/LowBitrateAmplificationAndPareto.lean` (`7` Theorems)

| # | Theorem Name | Scope | Exact `#print axioms` | Core Formal Statement |
|---:|---|---|---|---|
| 15 | `low_bitrate_exponential_gap_amplification_universal` | Universal ($C_1 < C_2$, $M > 1$) | `[propext, Quot.sound]` | Under high-rate distortion scaling $D(b) = C \cdot 2^{-\alpha b}$ with drop multiplier $M = 2^{\alpha \Delta b} > 1$, relative ratio $C_1 / C_2$ is exactly invariant (`(M*C1)*C2 = C1*(M*C2)`), strict ordering is preserved, and the absolute gap expands by $(M - 1)(C_2 - C_1) > 0$. |
| 16 | `two_bitrate_order_and_gap_scaling_universal` | Universal ($C_1 < C_2 < C_3$, $M > 1$) | `[propext, Quot.sound]` | Preserves strict 3-way layout ordering (`M*C1 < M*C2 < M*C3`), strictly amplifies both pairwise absolute gaps, and preserves the exact gap-ratio proportionality $(M C_2 - M C_1)(C_3 - C_2) = (C_2 - C_1)(M C_3 - M C_2)$. |
| 17 | `pareto_corner_strict_domination_universal` | Universal ($\forall \alpha, \beta > 0$, $s_q < s_g < s_u$, $c_q < c_s$) | `[propext, Quot.sound]` | Whenever Stratified Quad-Interleaving strictly lowers micro-quad spread ($s_q < s_g < s_u$) and macro-ball CV ($c_q < c_s$), it strictly dominates global sort, `grp128`, and unpermuted baselines under *every* positive linear trade-off $\alpha \cdot s + \beta \cdot c$. |
| 18 | `blockh4_vs_blockh2_matched_two_level_gain_universal` | Universal ($h_2 \le h_1$) | `[propext, Quot.sound]` | Exact spread decomposition between $H_4$ and paired $H_2$: $(h_0 - h_3) - ((h_0 - h_1) + (h_2 - h_3)) = h_1 - h_2 \ge 0$, vanishing iff the inner gap $h_1 - h_2 = 0$, explaining why monotone quad sorting neutralizes $H_4$'s intrinsic spread handicap against `BlockH2`. |
| 19 | `uncalibrated_vs_calibrated_ball_cv_dichotomy_universal` | Universal ($Q_2 < Q_0$, $Q_3 < Q_1$, $A \in \mathbb{Z}$, $\beta, \gamma > 0$) | `[propext, Quot.sound]` | Under shared intercept $A$, exact macro-variance drop `var_glob - var_snk = 2*(Q0 - Q2)*(Q1 - Q3) > 0` (`var_snk < var_glob`) yields opposite-sign optimality: calibrated Hessian loss (`A + beta * var`) gives `loss_cal_snk < loss_cal_glob`, whereas uncalibrated sparse `wcol` loss (`A - gamma * var`, matching `360M`/`0.5B`/`1.5B` `wcol`) gives `loss_uncal_glob < loss_uncal_snk`. |
| 20 | `s72_four_model_pareto_corner_witness` | Exact Witness (`decide`) | `[]` (zero axioms) | Exact micro-unit ($10^{-6}$) verification that all 4/4 models (`SmolLM2-360M`, `Qwen2.5-0.5B`, `Qwen2.5-1.5B`, `Qwen2.5-3B`) satisfy `quad < grp128 < unp` and `cv_lpt, cv_snk < cv_glob` on *both* `h_real` and `wcol`, plus `Qwen2.5-3B` `> 6.73x` / `> 120.8x` spread and `> 2.67x` / `> 66.7x` CV drops. |
| 21 | `s72_three_qwen_blockh2_shatter_and_zerotext_witness` | Exact Witness (`decide`) | `[]` (zero axioms) | Verifies the 5 exact nanonat empirical KL ladders across `Qwen2.5-0.5B` (`quad_lpt < H2 < H4 < sort`), `Qwen2.5-1.5B` (`sharednorm < lpt < snake < H2 < H4 < sort`), `Qwen2.5-3B` (`snake < H2 < H4 < sort`), `SmolLM2-360M` (`snake < lpt < H2 < sort`), and zero-text `Qwen2.5-3B` (`rand < H4 < H2` on Python; `snake_wcol < lpt_wcol < sort_wcol` on WikiText-2); exact `%` deltas and the `8x` `4.5 -> 3.0 bpw` gap amplification (`1,024,654 -> 8,197,232` nanonats, instantiating Theorems 15–16) are certified in `s73b_certificate.json` §4.3. |

---

## 4. Certified Exact Quantitative Witnesses

All rational ratios below are stored canonically in `s73b_certificate.json`; decimal displays are programmatically rounded via `decimal.Decimal(..., ROUND_HALF_UP)` at 60-digit internal precision.

### 4.1 Pillar 1: Heavy-Tail Greedy-LPT Starvation & Exact Cures

#### `Qwen2.5-3B` Empirical Starvation & Cure Ladders (`rtx3080_s72_empirical_results.json`, SHA-256 `a7c35c64...`)

| Eval Split | Layout | Exact KL (nanonats) | KL (nats) | Comparison Anchor | Exact Ratio / Relative Delta ($\mathbb{Q}$) | Decimal Summary |
|---|---|---:|---:|---|---|---:|
| WikiText-2 | `quad_snake` | `23,154,136` | `0.023154` | vs `BlockH2` (`24,178,790`) | `-512327/12089395` | `-4.24%` |
| WikiText-2 | `quad_snake` | `23,154,136` | `0.023154` | vs `BlockH4` (`24,774,488`) | `-202544/3096811` | `-6.54%` |
| WikiText-2 | `BlockH2` | `24,178,790` | `0.024179` | baseline | — | — |
| WikiText-2 | `BlockH4` | `24,774,488` | `0.024774` | baseline | — | — |
| WikiText-2 | `quad_lpt_sqrt` ($p=1/2$) | `25,296,147` | `0.025296` | vs `linear_lpt` (`26,776,390`) | `-1480243/26776390` | `-5.53%` |
| WikiText-2 | `global_sort` | `26,025,440` | `0.026025` | baseline | — | — |
| WikiText-2 | `linear_lpt` ($p=1$) | `26,776,390` | `0.026776` | vs `global_sort` (`26,025,440`) | `2677639/2602544` | `1.0289x` (`+2.89%`) |
| Python | `quad_lpt_sqrt` ($p=1/2$) | `19,519,743` | `0.019520` | vs `BlockH2` (`22,975,467`) | `-1151908/7658489` | `-15.04%` (`< 0.020`) |
| Python | `quad_snake` | `20,493,620` | `0.020494` | vs `BlockH2` (`22,975,467`) | `-2481847/22975467` | `-10.80%` |
| Python | `quad_lpt_rand` (zero-text) | `21,866,574` | `0.021867` | vs `BlockH4` (`22,518,843`) | `7288858/7506281` | `0.9710x` (`-2.90%`) |
| Python | `BlockH4` | `22,518,843` | `0.022519` | baseline | — | — |
| Python | `BlockH2` | `22,975,467` | `0.022975` | baseline | — | — |

#### Canonical Exact Structural Witnesses

- **6-Quad Starvation Witness ($x = [10, 8, 5, 4, 2, 1]$, $w = x^2 = [100, 64, 25, 16, 4, 1]$):** Because $w_0 = 100 > 89 = w_1 + w_2$, `capLpt6` starves Ball 0 until Ball 1 hits its 3-quad cap, achieving perfectly balanced linear sums `(105, 105)` yet distorted $L_2$-ball energies `(13, 17)` (`prod = 221`, `sq = 458`, `peak = 17`). Equal-cardinality `quad_snake` gives $L_2$ energies `(16, 14)` (`prod = 224 > 221`, `sq = 452 < 458`, `peak = 16 < 17`), with exact gaps `224 - 221 = (4 - 1) * ((8 + 5) - (10 + 2)) = 3` and `458 - 452 = 2 * 3 = 6`.
- **9-Quad, 3-Ball Secondary Overpack Witness ($w = [200, 30, 28, 25, 22, 20, 18, 2, 1]$):** Starved 3-ball cardinality-capped LPT assigns `(w0, w7, w8)` to Ball 0 (`203`) and dumps `(w1..w6)` onto Balls 1–2 (`sec_lpt = 143`), whereas 3-ball snake assigns `(w0, w5, w6)` to Ball 0 (`238`) and lowers secondary-ball mass to `sec_snake = 108`, an exact overpack gap of `(20 - 2) + (18 - 1) = 35`.

### 4.2 Pillar 2: Concave $p$-Exponent Load Compression & 3-Regime Phase Trichotomy

- **Outlier-Ratio Compression ($x_0 = 10, x_1 = 8$):** Linear ($p = 1$) ratio $100 / 64 = 25/16$ (`1.5625`) compresses under square-root ($p = 1/2$) to $10 / 8 = 5/4$ (`1.2500`), an exact compression factor of `5/4` (`1.2500x`).
- **Pythagorean Basin Expansion ($x_1 = 8, x_2 = 5$):** The Euclidean linear non-starvation boundary is $8^2 + 5^2 = 89$, whereas the square-root boundary in $w$-units expands to $(8 + 5)^2 = 169$ — an exact basin expansion slack of $2 \cdot 8 \cdot 5 = 80$ containing $x_0^2 = 100$.

#### Three-Regime Phase Trichotomy (Unit Scale $s = 1$)

| Phase Regime | 6-Quad $L_2$ Ladder $x$ | Starvation Condition | `p = 1` Linear LPT $\sum E_g^2$ | `p = 1/2` Sqrt LPT $\sum E_g^2$ | `p = 0` Snake $\sum E_g^2$ | Strict Winner |
|---|---|---|---:|---:|---:|---|
| **Regime I** (Mild Tail) | `[8, 6, 6, 5, 3, 1]` | $64 \le 36 + 36 = 72$ (none) | `421` (`(14, 15)`) | — | `425` (`(16, 13)`) | **`p = 1` Linear LPT** (`421 < 425`) |
| **Regime II** (Pythagorean Basin) | `[10, 8, 5, 5, 3, 1]` | $89 < 100 < 169$ ($p=1$ only) | `520` (`(14, 18)`) | `512` (`(16, 16)`) | `520` (`(18, 14)`) | **`p = 1/2` Sqrt LPT** (`512 < 520`) |
| **Regime III** (Extreme Outlier) | `[20, 8, 5, 4, 2, 1]` | $20 > 8 + 5 = 13$ (both $p=1, 1/2$) | Ball 1 = `17` | Ball 1 = `17` | Ball 1 = `14` (`[0,3,4]/[1,2,5]`) | **`p = 0` Snake** (unconditionally starvation-free) |

#### Empirical Cross-Scale Phase Flip (WikiText-2, `a7c35c64...`)

| Model | `quad_lpt` ($p=1$, nats) | `quad_snake` ($p=0$, nats) | Exact Ratio ($\mathbb{Q}$) | Relative Delta | Empirical Regime |
|---|---:|---:|---|---:|---|
| `Qwen2.5-0.5B` | `0.025955` (`25,954,645` nkl) | `0.030080` (`30,079,819` nkl) | `25954645/30079819` (`lpt/snake`) | `-13.71%` (`0.8629x`) | Regime I (`lpt` wins) |
| `Qwen2.5-1.5B` | `0.022964` (`22,963,627` nkl) | `0.023177` (`23,176,567` nkl) | `22963627/23176567` (`lpt/snake`) | `-0.92%` (`0.9908x`) | Regime I/II crossover (`lpt` edges `snake`) |
| `Qwen2.5-3B` | `0.026776` (`26,776,390` nkl) | `0.023154` (`23,154,136` nkl) | `11577068/13388195` (`snake/lpt`) | `-13.53%` (`0.8647x`) | Regime III (`snake` wins; `p=1` starves) |

### 4.3 Pillar 3: Low-Bitrate Amplification & 4/4-Model Pareto-Corner Dominance

- **High-Rate `4.5 -> 3.0 bpw` Gap Amplification ($\alpha = 2, \Delta b = 3/2 \implies M = 2^3 = 8$):** On `Qwen2.5-3B` WikiText-2, the `BlockH2 - quad_snake` gap expands exactly `8x` from `1,024,654` nanonats (`0.001025` nats) at `4.5 bpw` to `8,197,232` nanonats (`0.008197` nats) under $\alpha = 2$ scaling at `3.0 bpw`, while the exact relative ratio `11577068/12089395` (`0.9576`, `-4.24%`) remains invariant.

#### 4/4-Model Strict Pareto-Corner Telemetry (`a7c35c64...`, micro-units $10^{-6}$ rounded to 4 decimals)

| Model | Metric | Spread `unp` | Spread `grp128` | Spread `quad` | `quad < grp128 < unp` | `cv_glob` | `cv_lpt` | `cv_snk` | `cv_lpt, cv_snk < cv_glob` |
|---|---|---:|---:|---:|:---:|---:|---:|---:|:---:|
| `SmolLM2-360M` | `h_real` | `2.2752` | `1.7990` | `0.6582` | Yes | `2.6921` | `1.8039` | `1.8507` | Yes |
| `SmolLM2-360M` | `wcol` | `0.3296` | `0.0918` | `0.0089` | Yes | `0.2009` | `0.0156` | `0.0212` | Yes |
| `Qwen2.5-0.5B` | `h_real` | `2.3303` | `1.0019` | `0.4123` | Yes | `1.5477` | `0.4404` | `0.5696` | Yes |
| `Qwen2.5-0.5B` | `wcol` | `0.4715` | `0.0430` | `0.0078` | Yes | `0.2639` | `0.0025` | `0.0060` | Yes |
| `Qwen2.5-1.5B` | `h_real` | `2.3089` | `1.1034` | `0.3165` | Yes | `1.9223` | `0.7102` | `0.8157` | Yes |
| `Qwen2.5-1.5B` | `wcol` | `0.3871` | `0.0345` | `0.0041` | Yes | `0.2282` | `0.0033` | `0.0047` | Yes |
| `Qwen2.5-3B` | `h_real` | `2.2745` | `1.1338` | `0.3379` | Yes | `2.0962` | `0.7830` | `0.8823` | Yes |
| `Qwen2.5-3B` | `wcol` | `0.3903` | `0.0349` | `0.0032` | Yes | `0.2365` | `0.0035` | `0.0050` | Yes |

- **`Qwen2.5-3B` Exact Pareto Drop Factors:** `h_real` micro-quad spread drop `568631/84465` (`6.7321x`), `wcol` micro-quad spread drop `390338/3229` (`120.8851x`), `h_real` ball-CV drop `2096239/782983` (`2.6772x`), and `wcol` ball-CV drop `39422/591` (`66.7039x`).

#### 3/3-Qwen `BlockH2` Shatter, `SmolLM2-360M` Rescue & Zero-Text Wins (`a7c35c64...`)

| Model / Eval Split | Winning Layout (nats) | Baseline (nats) | Exact Ratio ($\mathbb{Q}$) | Relative Delta |
|---|---:|---:|---|---:|
| `Qwen2.5-0.5B` (WikiText-2) | `quad_lpt` `0.025955` | `BlockH2` `0.026970` | `5190929/5393950` | `-3.76%` (`0.9624x`) |
| `Qwen2.5-1.5B` (WikiText-2) | `sharednorm` `0.022953` | `BlockH2` `0.023281` | `7651001/7760483` | `-1.41%` (`0.9859x`) |
| `Qwen2.5-3B` (WikiText-2) | `quad_snake` `0.023154` | `BlockH2` `0.024179` | `11577068/12089395` | `-4.24%` (`0.9576x`) |
| `SmolLM2-360M` (WikiText-2) | `quad_snake` `0.030821` | `global_sort` `0.045766` | `15410384/22882975` | `-32.66%` (`0.6734x`) |
| `Qwen2.5-3B` (Python, zero-text) | `quad_lpt_rand` `0.021867` | `BlockH4` `0.022519` | `7288858/7506281` | `-2.90%` (`0.9710x`) |
| `Qwen2.5-3B` (Python, zero-text) | `quad_lpt_rand` `0.021867` | `BlockH2` `0.022975` | `7288858/7658489` | `-4.83%` (`0.9517x`) |
| `Qwen2.5-3B` (WikiText-2, zero-text `wcol`) | `quad_snake_wcol` `0.049077` | `global_sort_wcol` `0.055560` | `1635893/1852012` | `-11.67%` (`0.8833x`) |

---

## 5. Cryptographic Certificate, Security Hardening & Reproducibility

### 5.1 Pinned Certificate Digests

- **`payload_sha256`:** `8c75be367dcfe591f7d425e17584e0ea67861d7d1da0c9a8d9cf3b7d47638dea`
- **`certificate_sha256`:** `cb0aac952ee0f5c187c6c36a3fcd822256908be0911a97716c7abb876b6f0e5b`
- **`s73b_certificate.json` file SHA-256:** `0890bcafa0893913db6cd46f762acf28a61ee31689af86c96da8d48b61a63766` (`6,461` bytes, mode `0o600`, canonical JSON + trailing `\n`, well under `TARGET_MAX_CERT_BYTES = 12,288`)

### 5.2 Hardening Properties of `s73b_lean4_oracle.py` & `s73b_lean4_audit.py`

- **Zero-Float Trust Path:** Iterative strict JSON tree validator rejects `float` and `float` subclasses (`isinstance(cur, float)`), non-exact container/scalar types (`bool` as `int`, `tuple`, `dict`/`list` subclasses), lone UTF-16 surrogates, nesting $> 32$, integers $> 1,000$ pre-parse digits or $> 3,321$ bits, and duplicate JSON keys.
- **Fail-Closed Filesystem I/O:** Atomic `dir_fd` `O_NOFOLLOW` ancestor walk (with root-owned prefix normalization for macOS `/var`, `/tmp`, `/etc` aliases), leaf `O_RDONLY | O_NOFOLLOW | O_NONBLOCK`, `os.fstat(fd)` `S_ISREG` + `st_nlink <= 1` check, streaming `65,536`-byte reads capped at `1,048,576` bytes, and durable `0o600` temp-file + `os.fsync` + `os.replace` writes with cryptographically random temporary filenames.
- **R9 Static & Live Lean 4 Auditor (`--verify-lean`):** Comment/string/interpolation/char-escape/guillemet-aware lexer (with unary `!r#"...#`/`!s!"..."` entry and spaced/commented `s!`/`f!`/`m!` tracking) enforcing zero-import pure `Init` and rejecting `sorry`, `admit`, `axiom`, `unsafe`, `native_decide`, `sorryAx`, `Lean.Meta.mkSorry`/`mkSorry`, `Lean.ofReduceBool`/`ofReduceBool`, `Lean.ofReduceNat`/`ofReduceNat`, `Lean.trustCompiler`/`trustCompiler`, `partial def`/`instance`/`abbrev`, `debug.skipKernelTC`/`debug.byAsSorry`/`debug.proofAsSorry`/`debug.terminalTacticsAsSorry`, any `import` (`public`/`private`/`meta`), `#eval`/`#eval!`/`#reduce`/`#exit`, `#print` (other than valid `#print axioms <ident>` excluding Lean keywords such as `example`), `run_cmd`, `run_meta`, `initialize`, `builtin_initialize`, balanced `@[...]`/`attribute [...]` blocks containing `extern` or `implemented_by`, multi-flag `decide +native` / `decide (native := ...)` / `decide (config := ...)` and `DecideConfig`/`Decide.Config` bindings with non-false `native`, and unterminated comments, strings, or guillemet identifiers (plus advisory notices on `csimp`, vacuous `False ->`/`True :=` theorems, and `∃ ..., ... ->` signatures); live `lean` kernel execution rejects any non-zero exit code, any `stderr` output, any `stdout` compiler warning, any missing/extra theorem, or any axiom outside each theorem's pinned allowlist.
- **Field-Complete Secondary Audit (`secondary_audit_payload`):** Enforces exact keysets on every nested object and independently recomputes and cross-checks every rational ratio, integer witness, empirical anchor hash, module SHA-256, and per-theorem axiom profile without trusting cached summary booleans.

### 5.3 Reproducibility Commands

Run inside the directory containing this report (with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1`):

```bash
# 1. Self-check oracle & certificate parity (3 isolated modes + PYTHONHASHSEED=42 -B)
python3 -I -B s73b_lean4_audit.py
python3 -I -O -B s73b_lean4_audit.py
python3 -I -B -W error s73b_lean4_audit.py
PYTHONHASHSEED=42 python3 -B s73b_lean4_audit.py

# 2. Verify committed certificate (full recompute + field-complete secondary audit)
python3 -I -B s73b_lean4_audit.py --check s73b_certificate.json --json

# 3. Run R9 static Lean 4 scanner + live Lean 4.36.0 kernel & #print axioms verification
python3 -I -B s73b_lean4_audit.py --verify-lean . --json

# 4. Run full R5 adversarial & live-kernel test suite (3 isolated modes + PYTHONHASHSEED=42 -B)
python3 -I -B test_s73b_lean4_r5.py -v
python3 -I -O -B test_s73b_lean4_r5.py
python3 -I -B -W error test_s73b_lean4_r5.py
PYTHONHASHSEED=42 python3 -B test_s73b_lean4_r5.py

# 5. Clean scratch Lake build (preserves zero .lake build artifacts in source tree)
python3 -I -B -c "import pathlib, shutil, subprocess, tempfile; d = tempfile.TemporaryDirectory(); p = pathlib.Path(d.name); [shutil.copy2(f, p / f) for f in ('lean-toolchain', 'lakefile.toml')]; shutil.copytree('S73bFormalTheory', p / 'S73bFormalTheory'); subprocess.run(['lake', 'build'], cwd=p, check=True); d.cleanup()"
```

---

## 6. Frozen 9-File Deliverable Manifest

Every executable, formal, test, and certificate artifact in this bundle is pinned below (this report is the 10th file in the directory):

| Relative Path | Lines | Bytes | Mode | SHA-256 |
|---|---:|---:|---|---|
| `lean-toolchain` | `1` | `25` | `0o644` | `582a63f8f9c5128067b688778c1051cf1464c3ca9d57b9325025c051c29b774a` |
| `lakefile.toml` | `11` | `282` | `0o644` | `2eb91fdd900281010766fbe8b7d1f0fea3553a2a5bae4d81891cd38221bdfb70` |
| `S73bFormalTheory/HeavyTailLPTStarvation.lean` | `268` | `13479` | `0o644` | `bf9cf6693494b714ba4ecea01fa8bbcbaba182eceefa20fdde9129a947308f15` |
| `S73bFormalTheory/LoadPowerCompressionPhase.lean` | `303` | `16404` | `0o644` | `a0242b19bc467d68c436801db0a53718797b450aaf7f256ff7650a77e8fe3caf` |
| `S73bFormalTheory/LowBitrateAmplificationAndPareto.lean` | `172` | `10234` | `0o644` | `b0ad408f10a7d7f131c3e782581a639fb333a33b07a0c85eef185b914c0135d4` |
| `s73b_lean4_oracle.py` | `638` | `27536` | `0o644` | `ee8bb88c1a1b067592b4acfcfd0d1813bc8005ba7947dd70aa2a68c7f9fe8294` |
| `s73b_lean4_audit.py` | `1250` | `56020` | `0o644` | `e2461cf4db1c63fda1788843b6c315324905719cc9c546f69108c1d31ff3ce8e` |
| `test_s73b_lean4_r5.py` | `1217` | `69276` | `0o644` | `e820164cc0e3a3b858dfc3f182d1f94d59f4184892a00b2985b860a352376c4d` |
| `s73b_certificate.json` | `1` | `6461` | `0o600` | `0890bcafa0893913db6cd46f762acf28a61ee31689af86c96da8d48b61a63766` |
