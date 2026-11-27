# Hadwiger-3D Convex-Covering Operators on `SiliQuant` (`E_8` & `A_16 Block-H4`)

- **Experiment ID:** `EXP-Q47b` (`PREREG-W22-HADWIGER-SILIQUANT`, SHA-256 `c57b8a40d33d2f8822b58c87141cbf69ff3d64a292058faaf21e993db619f74b`)
- **Primary Source:** `arXiv:2611.32334v2` (Aiger, Har-Peled, Zuzic, *Hadwiger's Covering Conjecture in Three Dimensions*) transferred onto `Moissanite-Labs/siliquant` (`d=8 E_8` and `d=16 A_16 Block-H4`)
- **Novelty Classification:** `N1` (pure-`Init` Lean 4.36.0 operator formalization, exact $\mathbb{Z}/\mathbb{Q}$ oracle/verifier, and live RTX 3080 real-weight falsification/transfer audit; foundational geometry and quantization algorithms quarantined as `N0`)
- **Oracle Class:** `exact_and_sampled` (pure-$\mathbb{Z}/\mathbb{Q}$ exact certificates + live RTX 3080 evaluation on real cached `Qwen2.5-0.5B` & `Qwen2.5-1.5B` weights)
- **Energy Disclosure:** `UNCONTROLLED_LAPTOP_NOT_CLAIMED` (zero laptop energy/joule/watt claims)

---

## 1. Executive Verdict (Law 4 Honest Kill-Gate Summary)

| Pillar / Arm | Preregistered Criterion | Measured Outcome | Verdict |
|---|---|---|---|
| **Formal Geometry (`JohnQuantOperators.lean`)** | Pure-`Init` Lean `4.36.0`, `0` `sorry`/custom axioms/`Classical.choice` across $d=3,4,8,16$ mean-width/volume/diameter gaps, `+0 B` folding, $L_2$ optimality, Helly-Chebyshev, & polar-cap split | `24/24` live-verified theorems (`7/24` zero axioms `[]`, `17/24` `[propext]` only; `0` warnings/stderr; `0` `.lake/` residue) | **CONFIRMED** |
| **Arm B1: John/Lowner $S_j$ `+0 B` Folded Diagonal Whitening** | Beat orthogonal-only `BlockH4` + `SharedNorm` by $\ge 2\%$ weighted MSE or $\ge 0.01$ held-out WikiText-2 PPL at `3.0 bpw` | Apples-to-apples `B1c vs A3`: **fails both prongs on both models** (`hMSE` `+6.45%` / `+5.79%` worse, `0/52` layer wins; WikiPPL `-0.2185` / `-0.1364` worse). `B1b vs A2`: `hMSE` `+6.35%` / `+5.66%` worse (`0/52` layers), WikiKL `+12.79%` / `+4.49%` worse, **passes `>= 0.01` WikiPPL on `1.5B` (`+0.0504`)** while failing on `0.5B` (`-0.2196`); unweighted `p999` drops `3.28%` (`0.5B`) & `20.47%` (`1.5B`). `B1d vs A1` (`E_8`): fails `hMSE`, beats WikiPPL (`+0.3823` / `+0.1569`) | **PARTIAL** (`B1c vs A3` FALSIFIED; `1/8` `A_16` prongs passed) |
| **Arm B2: Helly-LP Minimax Chebyshev Gauge Scales & Center** | Must not degrade L2 PPL vs MSE-optimal `QuadLPT` without $\ge 15\%$ reduction in `99.9th`-percentile outlier error | `0/6` `B2` arms reach `>= 15%` `p999` reduction (`B2a` drops `0.10%` / `1.06%` at `3b` and `1.32%` / `2.35%` at `4.5b`; `B2b` worsens `p999` `+103.7%` / `+138.0%`). `B2a` degrades WikiPPL on `0.5B 3b` (`-0.7240`) & `1.5B 4.5b` (`-0.0642`), and `B2b` degrades WikiPPL on both (`-5.4784` / `-2.1231`) without `15%` `p999` (`4` falsification hits; `B2a` non-degraded WikiPPL on `1.5B 3b` [`+0.0031`, PyPPL `-0.0143`] & `0.5B 4.5b` [`+0.1493`]) | **FALSIFIED** (Kill Gate 2 triggered) |
| **Arm B3: Polar-Cap vs Equatorial-Bulk Split (Matched `bpw`)** | Tight geometric sparse-outlier vs `A_16`-bulk partition at strict `<= 3.000` / `<= 4.500 bpw` (`CS-08`) | Payload+scale rates stay `<= 3.000` (`B3b` `2.998 bpw`, `B3a` `2.994 bpw`), **but uncompressed metadata exceeds `3.000`** (`B3b` + `1b`/group selector map = **`3.0053 bpw`**; `B3a` + `16b` `row_ptr` = **`3.0077` / `3.0019 bpw`**). `B3b` achieves lowest `3.0 bpw` `A_16` Python-prefix PPL (`5.8319` / `4.1632` on `fractions.py` only) and cuts `wp999` (`-8.70%` / `-7.98%`), **but has worse WikiPPL (`21.9780 vs 21.2516`; `14.4654 vs 14.4299`) & `hMSE` (`+0.70%` / `+2.17%`) than `A2`**. `B3a` beats `A3` WikiPPL (`+0.1914` / `+0.0939`) at worse `hMSE` | **PARTIAL** |
| **Overall Empirical `EXP-Q47b`** | Combined transfer of Hadwiger-3D operators onto `SiliQuant` | `52/52` live arms verified against raw results (`0` syndrome / `0` OOB / `0` non-block diffs, non-circular `CS-08` parity + metadata accounting); **root geometric obstruction isolated ($L_\infty$-envelope vs $L_2$-ellipsoid mismatch)** | **PARTIAL** (`KG1_PARTIAL_KG2_FALSIFIED_B3_PARTIAL`) |

---

## 2. Exact Geometric Discovery: Why $L_\infty$ John/Helly Operators Fail on $L_2$ Spherical Lattice Balls

Our combined Lean 4 proof and 52-arm live RTX 3080 study identify three exact mathematical mechanisms governing the transfer boundary between `arXiv:2611.32334v2` and `SiliQuant`:

1. **$L_\infty$-Envelope vs $L_2$-Ellipsoid Mismatch (Arm B1 Root Cause):**
   - By Theorem 17 (`lagrange_cauchy_schwarz_john_hessian_universal`), the unique (up to global scalar) diagonal whitening minimizing the Hessian-weighted $L_2$ product objective inside a spherical ball quantizer is $c_j^* \propto h_j^{1/4} / \mathrm{RMS}(W_{:,j})^{1/2}$ (whitening the **$L_2$ covariance ellipsoid**).
   - Hadwiger-3D's halved-difference-body John operator whitens the **$L_\infty$ bounding box** $S_j = \frac{1}{2}(\max_i W_{ij} - \min_i W_{ij})$. On real `Qwen2.5` weights, column half-widths are sharply heavy-tailed (`mean max/median = 3.362, max = 9.255` on `0.5B`; `mean = 3.401, max = 22.449` on `1.5B`).
   - Dividing by $S_j^{1/2}$ therefore over-attenuates high-kurtosis columns: it reduces **unweighted `p999` error** (its native $L_\infty$ objective: `-3.28%` on `0.5B`, **`-20.47%` on `1.5B`**, and **`-20.47%` / `-32.63%`** [`20.4658%` / `32.6273%`] vs `A2` in zero-cal `B1a`), while inflating **Hessian-weighted $L_2$ MSE on `52/52` transformer layers** (`0/24` wins on `0.5B`, `0/28` wins on `1.5B`, `+5.66%` to `+6.45%` worse) and degrading WikiText-2 KL (`+4.33%` to `+12.79%`).
   - **Confounder & multi-prong note:** Comparing `B1b` (no `QuadSnake`) directly against `A3` (with `QuadSnake`) shows positive WikiPPL delta (`+0.3432` on `0.5B`, `+0.1610` on `1.5B`) solely because `QuadSnake` slightly degrades `3.0 bpw` WikiPPL relative to unpermuted `A2` (`21.8143` vs `21.2516` on `0.5B`; `14.5405` vs `14.4299` on `1.5B`). Unpermuted `B1b vs A2` beats the `>= 0.01` WikiPPL threshold on `Qwen2.5-1.5B` (`+0.0504`) while failing on `0.5B` (`-0.2196`) and failing `hMSE`/`PyPPL` on both; `E_8` `B1d vs A1` beats WikiPPL (`+0.3823` / `+0.1569`) while failing `hMSE` on both; and apples-to-apples `B1c vs A3` fails both `hMSE` and PPL on both models (`KG1` verdict = **`PARTIAL`**).
   - **Control asymmetry disclosure:** `A1..A5` apply unpooled `o_proj` `hess_s73` column scales (non-foldable into GQA `v_proj` without `43,008 B` / `86,016 B` runtime scale storage, though declared `extra_runtime_bytes = 0` in the recorded run — patched in `bench.py`), whereas `B1a..B1d` pool `o_proj` across GQA heads for genuine `+0 B` folding and apply `[0.125, 8.0]` geometric clamping (uncoupled `B1e` confirms the `hMSE` loss persists without `qkv`/`gu`/GQA folding).
   - **Verified `+0 B` Folding Physics:** Physically folding shared `d_qkv` into `input_layernorm`, `d_gu` into `post_attention_layernorm`, `d_down` into `up_proj` rows, and GQA-shared `d_o` into `v_proj` rows + bias is algebraically exact at `+0 B` runtime overhead and preserves live forward outputs with small BF16 rounding drift (`prequant_kl = 0.001201` on `0.5B`, `0.001118` on `1.5B`; `98.43%` / `98.04%` top-1 match in BF16).

2. **Low Group Asymmetry & Tail-vs-Bulk Step Inflation (Arm B2 Root Cause):**
   - Real `Qwen2.5` 128-weight groups are nearly origin-symmetric (`mean |M+m|/(M-m) = 0.1104` on `0.5B`, `0.1022` on `1.5B`).
   - Spending 16 bits/group for an FP16 Helly-Chebyshev center $v_g = \frac{1}{2}(\max + \min)$ forces exact `CS-08` bulk reduction `B = 368 -> 352` (`R^2: 651 -> 547`, certified via exact `A_16^8` theta series), which destroys lattice density and **more than doubles `p999` error** (`+103.7%` on `0.5B`, `+138.0%` on `1.5B`).
   - Even at `+0 B` (`B2a` feasible-scale Chebyshev minimax search), Theorem 21's exact scale-inflation tradeoff dominates: stepping scale upward to trim one extreme coordinate widens quantization noise across the other 127 coordinates (`p999` drops only `0.10%`–`1.06%` at `3b` and `1.32%`–`2.35%` at `4.5b`, far below the `15%` kill gate, while `hMSE` rises `+5.2%`–`+5.3%` at `3b` and `+3.9%` at `4.5b`). Under `PREREG:31`, `B2a` degrades WikiPPL on `0.5B 3b` (`-0.7240`) and `1.5B 4.5b` (`-0.0642`) and `B2b` degrades WikiPPL on both scales (`-5.4784` / `-2.1231`), triggering `KG2 = FALSIFIED` (while `B2a` did not degrade WikiPPL at `1.5B 3b` [`+0.0031`, though `PyPPL` & `WikiKL` degraded] or `0.5B 4.5b` [`+0.1493`]).

3. **Polar-Cap Block Splitting Metadata & Residual Scope (Arm B3 Finding):**
   - Allocating `1/19` (`5.263%`) polar-cap sensitivity-aligned blocks to `4.5 bpw` (`R^2 = 5213`) and `18/19` equatorial-bulk blocks to `B = 357` (`R^2 = 577`; payload+scale = `2.998 <= 3.000 bpw`, logged `ball_bound = 651` in the recorded JSON before per-group override — patched in `bench.py`) achieves the **lowest `3.0 bpw` `A_16` Python-prefix PPL on both models** (`5.8319` on `0.5B`, `4.1632` on `1.5B`) and cuts `wp999` by `8.70%` / `7.98%`, **but performs worse than `A2` on WikiPPL (`21.9780 vs 21.2516`; `14.4654 vs 14.4299`) and total `hMSE` (`+0.70%` / `+2.17%`)**. Including an explicit uncompressed `1`-bit/group block-selector bitmap raises `B3b` to **`3.0053 bpw > 3.000`** (`3.005338` / `3.005326`; entropy-coding the `1/19` bitmap would yield `2.9999 bpw`).
   - Entry-level `0.25%` polar-cap CSR (`B3a`, `2.994 bpw` payload+entry excluding `row_ptr`, or **`3.0077` / `3.0019 bpw`** with `16`-bit CSR `row_ptr`) improves WikiText-2 PPL over `A3` on both models (`+0.1914` on `0.5B`, `+0.0939` on `1.5B`), but reducing bulk `R^2` across `99.75%` of entries (`651 -> 577`) still raises total `hMSE` by `+5.41%` / `+6.54%`. In the recorded `s165_empirical_results.json` run, `B3a`/`B3c` used additive residual reconstruction (`wq + sparse_rec`, leaving dense lattice noise on outlier slots); `bench.py` is patched to `torch.where(cap_mask, sparse_rec, wq)`.

---

## 3. Pure-`Init` Lean 4.36.0 Formalization (`JohnQuantOperators.lean`)

Every theorem in [JohnQuantOperators.lean](JohnQuantOperators.lean) (`SHA-256: d447fc897f4c60f118a94784638fe428f573ea062d89f45d35f8bc50bd33a39e`) is live-verified under `leanprover/lean4:v4.36.0` with **zero imports, zero Mathlib, zero `sorry`, zero custom axioms (`7/24` zero axioms `[]`, `17/24` `[propext]` only), zero `Classical.choice`, zero `Quot.sound`, zero `native_decide`, zero warnings, and zero `.lake/` residue**:

| # | Lean 4 Theorem (`JohnQuantOperators.*`) | Scope | Exact Axioms | Role |
|---|---|---|---|---|
| 1 | `mean_width_recurrence_growth_universal` | Universal (`Nat`) | `[propext]` | Exact recurrence $(2k+3)I_{2k+4} = (2k+2)I_{2k+2}$ & strict growth $M_{2k+4} > M_{2k+2}$ |
| 2 | `diameter_sq_gap_cube_over_ball_universal` | Universal (`Nat`) | `[]` | $D^2(Q_d) = 4d = d \cdot D^2(B^d)$ across all $d$ |
| 3 | `volume_ratio_cube_over_cross_growth_universal` | Universal (`Nat`) | `[]` | Strict factorial volume gap growth $d! < (d+1)!$ |
| 4 | `exact_spherical_abs_recurrence_d3_d4_d8_d16` | Exact Witness (`decide`) | `[]` | Exact $\pi$-cleared $I_d^{\mathrm{abs}}$ & $M_d$ across $d=3,4,8,16$ |
| 5 | `exact_mean_width_brackets_d3_d4_d8_d16` | Exact Witness (`decide`) | `[]` | Certified 4-dp rational brackets (`1.5`, `1.6976..7`, `2.3282..3`, `3.2417..8`) |
| 6 | `exact_volume_gaps_d3_d4_d8_d16` | Exact Witness (`decide`) | `[]` | Exact `vol(Q_d)/vol(B_1^d)=d!` and `vol(Q_d)/vol(B^d)` brackets |
| 7 | `john_decomposition_O4_exact_rational` | Exact Witness (`rfl`) | `[propext]` | 8-vector rational John decomposition of $O_4=2B_1^4$ ($\|v_s\|_2^2=1$, $\|v_s\|_1=2$ on $\partial O_4$, $\sum c_s v_s v_s^\top = I_4$) |
| 8 | `john_decomposition_O16_hadamard_exact_rational` | Exact Witness (`rfl`) | `[propext]` | 16-vector rational John decomposition of $O_{16}=4B_1^{16}$ ($\|v_s\|_2^2=1$, $\|v_s\|_1=4$ on $\partial O_{16}$, $\sum v_s v_s^\top = I_{16}$) |
| 9 | `hadwiger_d4_zonotope_and_star_helly_obstruction` | Exact Witness (`decide`) | `[]` | $V_2(Z(O_4))=12+8\sqrt{3} > 24=V_2(Q_4)$ (`48 > 36` / `192 > 144`) & 14-cap wall `7/8 < 1` |
| 10 | `halved_diff_body_translation_and_symmetry_universal` | Universal (`Int`) | `[propext]` | Translation invariance & central symmetry of $S=\frac{1}{2}(K-K)$ |
| 11 | `halved_diff_body_vec_translation_and_symmetry_universal` | Universal (`List Int`) | `[propext]` | Arbitrary-dimension list invariance & central symmetry of $S$ |
| 12 | `folded_diagonal_whitening_zero_byte_invariance_universal` | Universal (`List`) | `[propext]` | Exact `+0 B` cleared-denominator diagonal whitening dot invariance |
| 13 | `folded_diagonal_common_denom_dot_universal` | Universal (`List`) | `[propext]` | Common-denominator factor extraction for folded linear outputs |
| 14 | `rmsnorm_shared_qkv_fold_universal` | Universal (`Int`) | `[propext]` | Simultaneous `+0 B` shared-RMSNorm fold across `Q, K, V` |
| 15 | `blockh4_parseval_orthogonality_universal` | Universal (`Int`) | `[propext]` | Exact rational Block-$H_4$ isometry $\|H_4 v\|_2^2 = 4\|v\|_2^2$ |
| 16 | `john_diagonal_strictly_beats_anisotropy_universal` | Universal (`Int`) | `[propext]` | Strict AM-GM anisotropy gap $a_1^2+a_2^2 > 2a_1 a_2$ for $a_1 \ne a_2$ |
| 17 | `lagrange_cauchy_schwarz_john_hessian_universal` | Universal (`List`) | `[propext]` | Lagrange identity, global $L_2$-ellipsoid whitening optimality & strict collinearity uniqueness ($c_j^* \propto h_j^{1/4}/\mathrm{RMS}_j^{1/2}$) |
| 18 | `helly_chebyshev_center_minimax_optimal_universal` | Universal (`Int`) | `[propext]` | Exact minimax optimality & proved uniqueness of Chebyshev center $c^*=(m+M)/2$ |
| 19 | `helly_asymmetric_origin_penalty_universal` | Universal (`Int`) | `[propext]` | Strict origin-centering penalty $(M-m)+|M+m| > M-m$ |
| 20 | `helly_telescoping_covering_identity_universal` | Universal (`List`) | `[propext]` | Arbitrary-length `CurveVar.lean` Helly telescoping center identity |
| 21 | `helly_safe_scale_zero_oob_and_tail_tradeoff_universal` | Universal (`Nat`) | `[propext]` | Safe gauge zero-OOB sufficiency, scale inflation $s_{\mathrm{bulk}} < s_{\mathrm{safe}}$, & bulk variance inflation |
| 22 | `polar_cap_bulk_contraction_and_crossover_universal` | Universal (`Nat`) | `[]` | Constructive polar-cap bulk range contraction & matched-`bpw` crossover proof |
| 23 | `siliquant_matched_bpw_conservation_universal` | Universal (`Nat`) | `[propext]` | Per-group sidecar & block-split (`B_map` selector) `CS-08` bit-budget conservation |
| 24 | `siliquant_a16_syndrome_and_budget_witness` | Exact Witness (`rfl`) | `[propext]` | `A_16` `[16,11,4]` syndrome checks & matched-`bpw` `g=128` bit identities |

---

## 4. Live RTX 3080 Real-Weight Benchmark Results (`52` Arms, `0` Syndrome / `0` OOB Errors)

Hardware: `NVIDIA GeForce RTX 3080 Laptop GPU` (`8192 MiB` VRAM total; `4 MiB` pre-run; peak allocated `1486.58 MiB` on `0.5B`, `3531.41 MiB` on `1.5B`; `gpu_post.memory_used_mib = 213 MiB` after cleanup in `s165_empirical_results.json`; total runtime `2594.917s`; `energy_claim = UNCONTROLLED_LAPTOP_NOT_CLAIMED`).
Calibration: `wikitext2_val.txt` (`12 x 512` tokens). Held-out Eval: `wikitext2_test.txt` (`8 x 512` tokens = `4,088` scored) and Python stdlib (`8 x 512` tokens = `4,088` scored; **scope disclosure:** because `fractions.py` alone tokenizes to `9,457` Qwen2.5 tokens, the recorded `4,096`-token eval window scored only the leading `4,096` tokens of `fractions.py` without reaching `difflib.py`–`json/decoder.py`, and single-seed `0.003–0.015` PPL deltas carry no multi-seed CIs; `bench.py` is patched via `_tokenize_balanced_python_chunks` to slice across all 6 stdlib files on any future run). Certified `A_16` & `E_8` `g=128` theta bounds (all independently computed & verified in `oracle.py` & `verify.py`): `A_16` `3.0b`: `B=368 -> R^2=651`, `B=357 -> R^2=577`, `B=352 -> R^2=547`; `A_16` `4.5b`: `B=560 -> R^2=5213`, `B=549 -> R^2=4629`, `B=544 -> R^2=4385`; `E_8`: `B=368 -> R^2=420`, `B=560 -> R^2=3380`. Note: `Eff. bpw` values for `B3a`/`B3b`/`B3c` in Tables 4.1–4.2 denote payload+scale/entry rates excluding `16b` CSR `row_ptr` (`3.0077`/`3.0019 bpw`) and `1b`/group block selector map (`3.0053 bpw`); `bench.py` also patches `_a16_closest` (exact 2,048-codeword ML + coordinate-reconstructed syndrome), `out_m` fallback scale update, `torch.where` CSR replacement, and unpooled `o_proj` `extra_runtime_bytes`.

### 4.1 `Qwen/Qwen2.5-0.5B` (`24` layers, `168` linear matrices, `357,826,560` block weights; BF16 ref WikiPPL `17.950473`, PyPPL `4.860631`)

| Arm | Eff. `bpw` | Weighted `hMSE` (`dB`) | `p999` Abs | `wp999` Abs | WikiText-2 PPL (`KL`) | Python PPL (`KL`) | Syn/OOB |
|---|---:|---:|---:|---:|---:|---:|---:|
| `A0_b30_RTN_unwhitened` | `2.932` | `0.082921` (`-10.813`) | `0.02393` | `0.05260` | `224.3252` (`2.561646`) | `82.8516` (`2.821588`) | `0/0` |
| `A1_b30_E8_H8_hess` | `3.000` | `0.015014` (`-18.235`) | `0.01571` | `0.01283` | `22.2092` (`0.222952`) | `5.8108` (`0.197416`) | `0/0` |
| `A2_b30_A16_BlockH4_hess` | `3.000` | `0.015062` (`-18.221`) | `0.01653` | `0.01307` | `21.2516` (`0.213222`) | `5.8807` (`0.181340`) | `0/0` |
| `A3_b30_A16_BlockH4_QuadSnake_SN` | `3.000` | `0.015103` (`-18.210`) | `0.01658` | `0.01325` | `21.8143` (`0.223652`) | `5.8624` (`0.197951`) | `0/0` |
| `A4_b30_A16_BlockH4_QuadLPT_SN` | `3.000` | `0.015053` (`-18.224`) | `0.01655` | `0.01308` | `22.1350` (`0.218938`) | `5.9703` (`0.209169`) | `0/0` |
| `A5_b30_A16_RandOrth_hess` | `3.000` | `0.015029` (`-18.231`) | `0.01846` | `0.01308` | `21.7814` (`0.228860`) | `5.8945` (`0.204683`) | `0/0` |
| `B1a_b30_JohnS_ZeroCal_BlockH4_Folded0B` | `3.000` | `0.028340` (`-15.476`) | `0.01315` | `0.03215` | `31.3146` (`0.589427`) | `8.2647` (`0.523368`) | `0/0` |
| `B1b_b30_JohnHess_BlockH4_Folded0B` | `3.000` | `0.016019` (`-17.954`) | `0.01599` | `0.01446` | `21.4712` (`0.240483`) | `5.9584` (`0.217125`) | `0/0` |
| `B1c_b30_JohnHess_BlockH4_QuadSnake_Folded0B` | `3.000` | `0.016077` (`-17.938`) | `0.01604` | `0.01461` | `22.0329` (`0.248211`) | `6.2159` (`0.238419`) | `0/0` |
| `B1d_b30_JohnHess_E8_H8_Folded0B` | `3.000` | `0.015955` (`-17.971`) | `0.01518` | `0.01417` | `21.8268` (`0.221091`) | `5.8695` (`0.206105`) | `0/0` |
| `B1e_b30_JohnHess_BlockH4_PerMat` | `3.000` | `0.015734` (`-18.032`) | `0.01607` | `0.01380` | `21.6036` (`0.236848`) | `5.8908` (`0.198408`) | `0/0` |
| `B2a_b30_HellyCheb_MinimaxScale_QuadLPT_0B` | `3.000` | `0.015837` (`-18.003`) | `0.01654` | `0.01303` | `22.8590` (`0.239679`) | `5.8821` (`0.204463`) | `0/0` |
| `B2b_b30_HellyCheb_CenterScale_QuadLPT_Matched` | `3.000` | `0.028226` (`-15.494`) | `0.03372` | `0.02266` | `27.6133` (`0.488494`) | `8.2396` (`0.541232`) | `0/0` |
| `B3a_b30_PolarCap_CSR025_BlockH4_Matched` | `2.994` | `0.015920` (`-17.981`) | `0.01621` | `0.01286` | `21.6230` (`0.222298`) | `5.8473` (`0.188851`) | `0/0` |
| `B3b_b30_PolarCap_BlockSplit_1over19_Matched` | `2.998` | `0.015168` (`-18.191`) | `0.01535` | `0.01209` | `21.9780` (`0.222846`) | `5.8319` (`0.190086`) | `0/0` |
| `B3c_b30_JohnHess0B_Plus_PolarCap_CSR025` | `2.994` | `0.017161` (`-17.655`) | `0.01611` | `0.01452` | `21.9556` (`0.251788`) | `6.0409` (`0.215168`) | `0/0` |
| `A1_b45_E8_H8_hess` | `4.500` | `0.001833` (`-27.369`) | `0.00554` | `0.00447` | `18.3405` (`0.025606`) | `4.9711` (`0.023470`) | `0/0` |
| `A2_b45_A16_BlockH4_hess` | `4.500` | `0.001843` (`-27.344`) | `0.00585` | `0.00459` | `18.3606` (`0.029336`) | `4.9620` (`0.023355`) | `0/0` |
| `A3_b45_A16_BlockH4_QuadSnake_SN` | `4.500` | `0.001850` (`-27.328`) | `0.00586` | `0.00464` | `18.4747` (`0.030362`) | `4.9848` (`0.027084`) | `0/0` |
| `A4_b45_A16_BlockH4_QuadLPT_SN` | `4.500` | `0.001845` (`-27.339`) | `0.00588` | `0.00459` | `18.4390` (`0.025685`) | `4.9909` (`0.022728`) | `0/0` |
| `B1b_b45_JohnHess_BlockH4_Folded0B` | `4.500` | `0.001959` (`-27.079`) | `0.00561` | `0.00506` | `18.2232` (`0.027722`) | `5.0054` (`0.024401`) | `0/0` |
| `B1c_b45_JohnHess_BlockH4_QuadSnake_Folded0B` | `4.500` | `0.001966` (`-27.064`) | `0.00564` | `0.00511` | `18.3007` (`0.029728`) | `4.9675` (`0.024824`) | `0/0` |
| `B1d_b45_JohnHess_E8_H8_Folded0B` | `4.500` | `0.001948` (`-27.105`) | `0.00533` | `0.00496` | `18.4007` (`0.027846`) | `4.9298` (`0.024035`) | `0/0` |
| `B2a_b45_HellyCheb_MinimaxScale_QuadLPT_0B` | `4.500` | `0.001917` (`-27.174`) | `0.00580` | `0.00452` | `18.2898` (`0.028537`) | `4.9823` (`0.025565`) | `0/0` |
| `B3a_b45_PolarCap_CSR025_BlockH4_Matched` | `4.494` | `0.001941` (`-27.119`) | `0.00570` | `0.00450` | `18.4681` (`0.028273`) | `4.9655` (`0.024919`) | `0/0` |
| `B3c_b45_JohnHess0B_Plus_PolarCap_CSR025` | `4.494` | `0.002092` (`-26.794`) | `0.00564` | `0.00507` | `18.3388` (`0.030692`) | `4.9510` (`0.025253`) | `0/0` |

### 4.2 `Qwen/Qwen2.5-1.5B` (`28` layers, `196` linear matrices, `1,310,195,712` block weights; BF16 ref WikiPPL `12.617995`, PyPPL `3.739529`)

| Arm | Eff. `bpw` | Weighted `hMSE` (`dB`) | `p999` Abs | `wp999` Abs | WikiText-2 PPL (`KL`) | Python PPL (`KL`) | Syn/OOB |
|---|---:|---:|---:|---:|---:|---:|---:|
| `A0_b30_RTN_unwhitened` | `2.932` | `0.085521` (`-10.679`) | `0.02793` | `0.06942` | `106.1535` (`2.233108`) | `32.3485` (`2.127990`) | `0/0` |
| `A1_b30_E8_H8_hess` | `3.000` | `0.015960` (`-17.970`) | `0.01753` | `0.01622` | `14.7382` (`0.186361`) | `4.3151` (`0.134706`) | `0/0` |
| `A2_b30_A16_BlockH4_hess` | `3.000` | `0.016011` (`-17.956`) | `0.02446` | `0.01654` | `14.4299` (`0.181467`) | `4.2607` (`0.123669`) | `0/0` |
| `A3_b30_A16_BlockH4_QuadSnake_SN` | `3.000` | `0.016060` (`-17.942`) | `0.02117` | `0.01664` | `14.5405` (`0.184347`) | `4.2450` (`0.126568`) | `0/0` |
| `A4_b30_A16_BlockH4_QuadLPT_SN` | `3.000` | `0.016006` (`-17.957`) | `0.01888` | `0.01637` | `14.5341` (`0.184414`) | `4.2407` (`0.120220`) | `0/0` |
| `A5_b30_A16_RandOrth_hess` | `3.000` | `0.015988` (`-17.962`) | `0.03099` | `0.01656` | `14.7273` (`0.185126`) | `4.2741` (`0.126398`) | `0/0` |
| `B1a_b30_JohnS_ZeroCal_BlockH4_Folded0B` | `3.000` | `0.032151` (`-14.928`) | `0.01648` | `0.04349` | `22.7259` (`0.664932`) | `5.0032` (`0.306613`) | `0/0` |
| `B1b_b30_JohnHess_BlockH4_Folded0B` | `3.000` | `0.016917` (`-17.717`) | `0.01946` | `0.01814` | `14.3795` (`0.189617`) | `4.3081` (`0.135811`) | `0/0` |
| `B1c_b30_JohnHess_BlockH4_QuadSnake_Folded0B` | `3.000` | `0.016991` (`-17.698`) | `0.01930` | `0.01825` | `14.6769` (`0.192322`) | `4.2649` (`0.134131`) | `0/0` |
| `B1d_b30_JohnHess_E8_H8_Folded0B` | `3.000` | `0.016867` (`-17.730`) | `0.01766` | `0.01781` | `14.5813` (`0.197774`) | `4.2433` (`0.128245`) | `0/0` |
| `B1e_b30_JohnHess_BlockH4_PerMat` | `3.000` | `0.016704` (`-17.772`) | `0.01929` | `0.01731` | `14.6205` (`0.191701`) | `4.2691` (`0.127680`) | `0/0` |
| `B2a_b30_HellyCheb_MinimaxScale_QuadLPT_0B` | `3.000` | `0.016851` (`-17.734`) | `0.01868` | `0.01637` | `14.5310` (`0.193740`) | `4.2550` (`0.127284`) | `0/0` |
| `B2b_b30_HellyCheb_CenterScale_QuadLPT_Matched` | `3.000` | `0.028673` (`-15.425`) | `0.04493` | `0.02613` | `16.6572` (`0.348742`) | `5.0806` (`0.321383`) | `0/0` |
| `B3a_b30_PolarCap_CSR025_BlockH4_Matched` | `2.994` | `0.017111` (`-17.667`) | `0.02135` | `0.01644` | `14.4465` (`0.189437`) | `4.1967` (`0.125617`) | `0/0` |
| `B3b_b30_PolarCap_BlockSplit_1over19_Matched` | `2.998` | `0.016358` (`-17.863`) | `0.02044` | `0.01531` | `14.4654` (`0.187797`) | `4.1632` (`0.115744`) | `0/0` |
| `B3c_b30_JohnHess0B_Plus_PolarCap_CSR025` | `2.994` | `0.018226` (`-17.393`) | `0.01964` | `0.01826` | `14.6449` (`0.197179`) | `4.3393` (`0.145161`) | `0/0` |
| `A1_b45_E8_H8_hess` | `4.500` | `0.001948` (`-27.104`) | `0.00625` | `0.00567` | `12.6857` (`0.023176`) | `3.7932` (`0.014721`) | `0/0` |
| `A2_b45_A16_BlockH4_hess` | `4.500` | `0.001959` (`-27.079`) | `0.00863` | `0.00579` | `12.8172` (`0.023416`) | `3.7921` (`0.015423`) | `0/0` |
| `A3_b45_A16_BlockH4_QuadSnake_SN` | `4.500` | `0.001967` (`-27.061`) | `0.00751` | `0.00583` | `12.8580` (`0.024018`) | `3.8011` (`0.017081`) | `0/0` |
| `A4_b45_A16_BlockH4_QuadLPT_SN` | `4.500` | `0.001960` (`-27.076`) | `0.00674` | `0.00573` | `12.8052` (`0.023867`) | `3.7839` (`0.014595`) | `0/0` |
| `B1b_b45_JohnHess_BlockH4_Folded0B` | `4.500` | `0.002071` (`-26.839`) | `0.00686` | `0.00635` | `12.8403` (`0.025798`) | `3.8041` (`0.015685`) | `0/0` |
| `B1c_b45_JohnHess_BlockH4_QuadSnake_Folded0B` | `4.500` | `0.002079` (`-26.821`) | `0.00681` | `0.00639` | `12.8401` (`0.024579`) | `3.7970` (`0.015661`) | `0/0` |
| `B1d_b45_JohnHess_E8_H8_Folded0B` | `4.500` | `0.002058` (`-26.866`) | `0.00631` | `0.00622` | `12.9285` (`0.027685`) | `3.7961` (`0.015350`) | `0/0` |
| `B2a_b45_HellyCheb_MinimaxScale_QuadLPT_0B` | `4.500` | `0.002037` (`-26.910`) | `0.00659` | `0.00566` | `12.8695` (`0.026400`) | `3.7727` (`0.015883`) | `0/0` |
| `B3a_b45_PolarCap_CSR025_BlockH4_Matched` | `4.494` | `0.002086` (`-26.806`) | `0.00751` | `0.00575` | `12.7608` (`0.023929`) | `3.8070` (`0.015151`) | `0/0` |
| `B3c_b45_JohnHess0B_Plus_PolarCap_CSR025` | `4.494` | `0.002223` (`-26.531`) | `0.00690` | `0.00639` | `12.7502` (`0.026000`) | `3.7887` (`0.015649`) | `0/0` |

---

## 5. Honest `N0` Prior-Art Quarantine & Engineering Recommendations (Law 4 & Law 5)

- **Quarantined `N0` Prior Art:** Fritz John ellipsoid theorem (1948); Helly's theorem (1923); Hadwiger 3D proof (`arXiv:2611.32334v2`, Aiger-Har-Peled-Zuzic 2026); SmoothQuant / activation-weight diagonal folding (Xiao et al. 2023); `E_8` / `A_16` extended Hamming lattice quantization (`Moissanite-Labs/siliquant`, Conway & Sloane); SqueezeLLM dense-and-sparse quantization (Kim et al. 2023).
- **Actionable Transposition into `SiliQuant`:**
  1. **Do NOT deploy** $L_\infty$ halved-difference-body John $S_j^{-1/2}$ whitening or Helly-Chebyshev group centers/minimax scales inside `A_16`/`E_8` spherical balls (`KG1` = `PARTIAL` with primary `B1c vs A3` falsified & `0/52` layer `hMSE` wins; `KG2` = `FALSIFIED`).
  2. **Test instead** Theorem 17's exact $L_2$-ellipsoid diagonal fold $c_j^* \propto h_j^{1/4} / \mathrm{RMS}(W_{:,j})^{1/2}$ (replacing $L_\infty$ $S_j$ with $L_2$ $\mathrm{RMS}(W_{:,j})$) using the verified `+0 B` RMSNorm + `up_proj` + GQA-`v_proj` folding mechanism.
  3. **Exploratory follow-up only (not promoted at `<= 3.000 bpw` uncompressed):** `B3b` block-level polar-cap splitting (`1/19` `4.5 bpw` + `18/19` `B=357`) reaches `2.998 bpw` payload+scale (and `-8.0%` to `-8.7%` `wp999`) **only** before adding the `1`-bit/group selector bitmap (`3.0053 bpw` uncompressed, requiring entropy coding [`~2.9999 bpw`] or `<= 1/21` split), and trails `A2` on both WikiText-2 PPL and total `hMSE`; any future code-PPL comparison must use balanced multi-file sampling rather than the leading `4,096`-token `fractions.py` slice.

---

## 6. Verification Gates & Artifact Provenance

- **Zero-Shared-Code Parity:** `s165_john_siliquant_oracle.py` and `verify_s165_john_siliquant.py` share `0` code and `0` imports, independently recompute all 4 pillars + `A_16^8`/`E_8^16` theta bounds + raw-results cross-verification + empirical kill gates over exact $\mathbb{Z}/\mathbb{Q}$, and emit bit-identical certificate `78a8d4fa7d91d4aab2e894d06233a50c4fc57ea0690a4ce532aabd78be9ed551` (`payload_sha256: 267418a9070db8c3b1ef9ea789eaf4086b4d70d86067f319bd28aa8e082dff1a`).
- **3x `-I` Determinism:** `-I -B`, `-I -O -B`, and `-I -B -W error` bit-identical across `s165_john_siliquant_oracle.py --json`, `verify_s165_john_siliquant.py --check --json`, and `test_s165_john_siliquant_r5.py` (`7/7 OK x 3`, `0` skips, `0` bare asserts, `0` floats on trust path).
- **Dual `aletheia mutate --max-mutants 0` + AST `BoolOp`:** `236/236` (`100.0%`, `0` survivors, `truncated=False`, `null_mutant=passed`, `--guards require_ok`) on `s165_john_siliquant_oracle.py` and `274/274` (`100.0%`, `0` survivors, `truncated=False`, `null_mutant=passed`, `--guards require`) on `verify_s165_john_siliquant.py` (`510/510 = 100.0%` combined); whole-file AST `BoolOp` (`OP_SWAP` + `SUBCOND_DEL`) `32/32` (`100.0%`, `0` survivors: `7/7` oracle + `25/25` verifier).
- **RW + RO `--live-lean`:** Both live track dir and `chmod -R a-w` read-only temp copy verified under Lean `v4.36.0` (`PASS`, `24/24` theorems: `7/24` zero-axiom `[]` + `17/24` `[propext]`, `0` `Classical.choice`/`Quot.sound`/`sorry`/warnings/stderr, `0` `.lake/` residue).
- **Audit Gate:** `aletheia audit-check --lean-audit --strict-ancestors --safety-scan --strict-unseeded --scripts s165_john_siliquant_oracle.py verify_s165_john_siliquant.py test_s165_john_siliquant_r5.py` `PASS` (`12` bundle files scanned, `4` `.py` scripts clean, `0` findings).

| Artifact | SHA-256 |
|---|---|
| [JohnQuantOperators.lean](JohnQuantOperators.lean) | `d447fc897f4c60f118a94784638fe428f573ea062d89f45d35f8bc50bd33a39e` |
| [lean-toolchain](lean-toolchain) | `582a63f8f9c5128067b688778c1051cf1464c3ca9d57b9325025c051c29b774a` |
| [lakefile.toml](lakefile.toml) | `2b2a9028814ee830a46fdde241f71d3ec8d943225a80b0c4a575ef9390fc55ab` |
| [lake-manifest.json](lake-manifest.json) | `c453fb1b592b20c2fa9c3cb5875f3cbad98ad4923f6a05e54029f2a80396f2c7` |
| [s165_john_siliquant_bench.py](s165_john_siliquant_bench.py) | `936ba0089067caec20e361ecc2e97b05c37aaac156be29164a473a31515843c8` |
| [s165_empirical_results.json](s165_empirical_results.json) | `b2427df13fbf73f186107384451c3c48e28ddacbaf586fd4410436679950712b` |
| [s165_empirical_receipt.json](s165_empirical_receipt.json) | `a430f75e6d9711c42d7054a5642118e6e7991018165b2464b527d239f4f08d4f` |
| [s165_john_siliquant_oracle.py](s165_john_siliquant_oracle.py) | `a8061416dd266a0d1311d1591fce8acec201995777ba3f8fa05c82ce85894076` |
| [verify_s165_john_siliquant.py](verify_s165_john_siliquant.py) | `5ff4661b2380878fc56adafc2131d5d242ec3735287ba266483e7f93efac8c04` |
| [test_s165_john_siliquant_r5.py](test_s165_john_siliquant_r5.py) | `ddf81b00f313ccbdd5888f4ac5a2b3a1e453ce7e1a77f214a6ffe91e8b622268` |
| [s165_john_siliquant_certificate.json](s165_john_siliquant_certificate.json) | `78a8d4fa7d91d4aab2e894d06233a50c4fc57ea0690a4ce532aabd78be9ed551` |
