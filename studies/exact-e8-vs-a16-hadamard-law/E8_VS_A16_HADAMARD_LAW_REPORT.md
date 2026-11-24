# Exact Pure-$\mathbb{Z}/\mathbb{Q}$ and Live Lean 4.36.0 Theory of the Opposite-Sign $E_8$ vs. $A_{16}$ Hadamard Law

- **Schema:** `aletheia.e8_vs_a16_hadamard_law.v1`
- **Novelty Tier (CS-13):** `N1` — Exact pure-$\mathbb{Z}/\mathbb{Q}$ / $\mathbb{Q}(\sqrt{2})$ and kernel-checked `Lean 4.36.0` mathematical proof explaining the opposite-sign Walsh–Hadamard effect on $E_8$ vs. Barnes–Wall $A_{16}$ (`E8_H8 < E8_axis` while `A16_axis < A16_BlockH2 < A16_H16` at matched weight MSE), together with exact falsifiable `Block-H4` predictions and a concrete zero-overhead per-layer selector anchored read-only to real-weight LLM empirical results (`42ad90b1...`).
- **Oracle Class:** `exact` (pure Python 3 stdlib over $\mathbb{Z}$, $\mathbb{Q}$, and $\mathbb{Q}(\sqrt{2})$; zero IEEE-754 floats on trust paths; zero bare `assert`s; zero mathematical RNG; `MAX_INT_BITS = 3321`, `MAX_INT_DIGITS = 1000`, `MAX_JSON_DEPTH = 32`).
- **Scope Disclosures (CS-13):**
  - `measured_live_rtx3080_in_this_track: false` — this study performs exact symbolic/algebraic verification and live `Lean 4.36.0` theorem proving, verifying bit-for-bit read-only parity against the pre-existing RTX 3080 empirical JSON anchor `rtx3080_live_llm_empirical_results.json` (`SHA-256 = 42ad90b1ddfba30d3d99b9d0a7a624770192b084dbe86b1e51c3a747cb7181e7`).
  - `measured_live_block_h4_llm_in_this_track: false` — the `A16_BlockH4` result ($\rho_{\mathrm{axis}} = 1$ restored, non-isometry witness $c_{\mathrm{cross}}$, and $2.08\times$ lower KL on the exact matrix witness) is an exact mathematical prediction for future GPU evaluation under `Moissanite-Labs/siliquant#77`.
- **Prior Literature (`N0`):**
  1. Conway & Sloane (1982, 1999): $E_8 / D_8$ and Barnes–Wall $A_{16} = C_{[16,11,4]} + 2\mathbb{Z}^{16}$ lattice geometry, kissing numbers (`240` and `2,272`), and exact closest-point decoders.
  2. Tseng et al. (2024, QuIP#): $E_8$ lattice quantization paired with Walsh–Hadamard incoherence processing.
  3. `Moissanite-Labs/siliquant#58`, `#77`, `#82`: $4.5$-bpw $A_{16}$ lattice quantization design, `Block-H2` signed `INT8` quantization path, and live RTX 3080 real-weight LLM evaluation (`SmolLM2-360M` + `Qwen2.5-0.5B`, `392` layers, anchor `42ad90b1...`).

---

## 1. Pillar 1 — Lattice-Ray Axis-Anisotropy Inversion ($8\times$ Opposite Sign) & Rational vs. Irrational Alignment

Define the orthogonal block Walsh–Hadamard transform $Q_b = I_{d/b} \otimes \frac{1}{\sqrt{b}} H_b$ ($Q_b^\top = Q_b = Q_b^{-1}$) and the **exact axis-anisotropy ratio** of the rotated lattice $Q_b^\top \Lambda$:

$$\rho_{\mathrm{axis}}(\Lambda, b) := \frac{\min\{\|c e_0\|^2 : c e_0 \in Q_b^\top \Lambda \setminus \{0\}\}}{d_{\min}^2(\Lambda)}.$$

- **Rational vs. Irrational Transforms:** $H_{16}^\top H_{16} = 16 I_{16}$ with $16 = 4^2$, so $Q_{16} = \frac{1}{4} H_{16} \in O_{16}(\mathbb{Q})$ is **exact rational**, and every one of its `16` rows (in both polarities) has exact zero 5-bit syndrome in Extended Hamming $C_{[16,11,4]}$ ($\mathrm{RM}(1,4) \subset C_{[16,11,4]}$, `2,048` codewords, $d_{\min} = 4$, kissing number `2,272`). By contrast, $Q_8 = \frac{1}{\sqrt{8}} H_8 \notin O_8(\mathbb{Q})$ ($\forall k \in \mathbb{N},\; k^2 \neq 8$) and $Q_{\mathrm{BH2}} = I_8 \otimes \frac{1}{\sqrt{2}} H_2 \notin O_{16}(\mathbb{Q})$ ($\forall k \in \mathbb{N},\; k^2 \neq 2$) are **irrational**.
- **Exact $8\times$ Opposite-Sign Contrast (`E_8` $2 \to 1$ vs. `A_16` $1 \to 4$):**
  - On **$E_8$** ($d_{\min}^2 = 2$, `240` roots), coordinate axes are *not* roots: the shortest axis vector is $2 e_j$ (norm² `4`, $\rho_{\mathrm{axis}} = 2$). Under $Q_8$, all `8` Walsh–Hadamard rows $\pm \frac{1}{2} h_j$ are shortest roots of $E_8$ (norm² `2`), rotating onto coordinate axes $\pm\sqrt{2} e_j \in Q_8^\top E_8$ and **halving** axis spacing ($\rho_{\mathrm{axis}}: 2 \to 1$).
  - On **$A_{16}$** ($d_{\min}^2 = 4$, `2,272` kissing vectors), all `32` coordinate axes $\pm 2 e_j$ are **already shortest vectors** ($\rho_{\mathrm{axis}} = 1$). Under $Q_{16}$, the shortest axis vector jumps from $2 e_j$ (norm² `4`) to $4 e_j$ (norm² `16`, $\rho_{\mathrm{axis}} = 4$) — an exact **$4\times$ coarsening**, yielding an exact **$4 \times 2 = 8\times$ opposite-sign contrast** between $E_8$ and $A_{16}$.

| Lattice | $d_{\min}^2$ | Block $b$ | Indicator $\mathbf{1}_b \in C_{[16,11,4]}$ | Shortest Axis Norm² | $\rho_{\mathrm{axis}}$ | Geometric Interpretation |
| :--- | ---: | ---: | :---: | ---: | ---: | :--- |
| $A_{16}$ | `4` | `1` (`A16_axis`) | `False` | `4` | **`1`** | Axes $\pm 2 e_j$ are already shortest vectors |
| $A_{16}$ | `4` | `2` (`A16_BlockH2`) | `False` | `8` | **`2`** | Weight-2 misses $d_{\min}=4$ of $[16,11,4]$ |
| $A_{16}$ | `4` | `4` (`A16_BlockH4`) | `True` | `4` | **`1`** | Weight-4 block codeword restores $\rho_{\mathrm{axis}} = 1$ (non-isometry!) |
| $A_{16}$ | `4` | `8` (`A16_BlockH8`) | `True` | `8` | **`2`** | Weight-8 codeword; $2\times$ coarser than axis |
| $A_{16}$ | `4` | `16` (`A16_H16`) | `True` | `16` | **`4`** | $4\times$ coarser: $2 e_j \to 4 e_j$ |
| $E_8$ | `2` | `1` (`E8_axis`) | — | `4` | **`2`** | Axes $\pm 2 e_j$ are non-roots |
| $E_8$ | `2` | `2` (`E8_BlockH2`) | — | `2` | **`1`** | $\pm(e_0+e_1) \in D_8$ rotates to $\sqrt{2} e_0$ |
| $E_8$ | `2` | `4` (`E8_BlockH4`) | — | `4` | **`2`** | Shortest block-4 axis norm² `4` |
| $E_8$ | `2` | `8` (`E8_H8`) | — | `2` | **`1`** | All 8 Hadamard roots $\frac{1}{2} h_j$ align with axes |

- **`A16_BlockH4` Non-Isometry Witness (`siliquant#77` Prediction):** While $b=4$ restores $\rho_{\mathrm{axis}} = 1$, $Q_4(A_{16}) \neq A_{16}$: the cross-block weight-4 codeword $c_{\mathrm{cross}} = e_0 + e_4 + e_8 + e_{12} \in C_{[16,11,4]}$ maps under $Q_4$ to $(\frac{1}{2})^{16} \notin \mathbb{Z}^{16}$.
- **Coherent Block-DC Concentration & `Block-H2` Signed `INT8` Safety:** Coherent DC concentration scales as `16×` ($H_{16}$) vs. `8×` ($H_8$) vs. `2×` (`Block-H2`), and all `Block-H2` lattice coordinates fit strictly inside signed `INT8` (`78 <= 127` on `[-39, 39]`; 72B ball $\lfloor\sqrt{5213}\rfloor = 72 \le 127$; 18B ball $\lfloor\sqrt{1207}\rfloor = 34 \le 127$).

---

## 2. Pillar 2 — Exact Pure-$\mathbb{Q}/\mathbb{Q}(\sqrt{2})$ Simultaneous Opposite-Sign Witnesses & Regime-Flip Criterion

Running the exact Conway–Sloane $E_8$ decoder and the provably globally optimal 2,048-codeword ML $A_{16}$ decoder natively over $\mathbb{Q}(\sqrt{2})$ (zero floats):

1. **Single-Spike Witness $w = \frac{3}{2} e_0 \in \mathbb{Q}^{16}$:**
   - $A_{16}$: `A16_axis` $= \frac{1}{4}$ $<$ `A16_BlockH2` $= \frac{41}{4} - 6\sqrt{2}$ $<$ `A16_H16` $= \frac{9}{4}$ (`9×` worse; `A16_BlockH4` $= \frac{1}{4}$ matches `A16_axis`).
   - $E_8$ (simultaneous opposite sign): `E8_H8` $= \frac{17}{4} - 3\sqrt{2}$ $<$ `E8_axis` $= \frac{1}{4}$ (`33.97×` lower error!).
2. **Exact Zero-Weight-MSE-Gap Matrix Witness $W = [\frac{3}{2} e_0,\; \frac{3}{8}\mathbf{1}_{16}]$ under PSD $H_1 = \operatorname{diag}(10, 1^{15})$:**

| Arm | Exact Weight MSE $\operatorname{tr}(\Sigma)$ | Exact Natural KL $\operatorname{tr}(H_1 \Sigma)$ | Exact Matched-MSE KL $\operatorname{tr}(H_1 \Sigma)/\operatorname{tr}(\Sigma)$ |
| :--- | :--- | :--- | :--- |
| **`A16_axis`** | `5/2` | `385/64` | **`77/32`** |
| **`A16_BlockH2`** | `41/2 - 12*sqrt2` | `7585/64 - 555/8*sqrt2` | **`185/32`** *(exact midpoint: `77 + 293 = 2 * 185`)* |
| **`A16_H16`** | `5/2` *(exact `0` gap vs `A16_axis`!)* | `1465/64` *(`3.81×` worse)* | **`293/32`** |
| **`A16_BlockH4`** | `1/2` *(`5×` lower)* | `185/64` *(`2.08×` better than `A16_axis`)* | `185/32` |
| **`E8_axis`** | `1/2` | `185/64` | `185/32` |
| **`E8_H8`** | `29/2 - 9*sqrt2` | `3745/64 - 315/8*sqrt2` *(`< E8_axis`)* | `17885/6176 - 2835/3088*sqrt2` *(`< E8_axis`, `3.62×` lower)* |

3. **Exact Trace-Zero Regime I $\leftrightarrow$ Regime II Sign-Reversal Criterion:**
   On the zero-MSE-gap covariance difference $\Delta\Sigma = \Sigma_{\mathrm{H16}} - \Sigma_{\mathrm{axis}} = 2 e_0 e_0^\top - \frac{1}{8}\mathbf{1}_{16}\mathbf{1}_{16}^\top$ ($\operatorname{tr}(\Delta\Sigma) = 0$):
   - Under axis-spiky Hessian $H_1 = \operatorname{diag}(10, 1^{15})$: $\Delta\mathcal{L}(H_1) = \operatorname{tr}(H_1 \Delta\Sigma) = \mathbf{+\frac{135}{8} > 0}$ (**Regime I:** `A16_axis` strictly wins).
   - Under coherent DC Hessian $H_2 = I_{16} + \mathbf{1}_{16}\mathbf{1}_{16}^\top$: $\Delta\mathcal{L}(H_2) = \operatorname{tr}(H_2 \Delta\Sigma) = \mathbf{-30 < 0}$ (**Regime II:** `A16_H16` strictly wins).

---

## 3. Pillar 3 — Quantitative Empirical Bridge (`392` Real LLM Layers, Anchor `42ad90b1...`) & Zero-Overhead Selector

Read-only exact-`Fraction` verification (`--verify-anchor`, zero IEEE-754 floats) confirms exact bit-for-bit parity across both real-weight models (`392` total linear layers) in `rtx3080_live_llm_empirical_results.json` (`42ad90b1...`):

- **End-to-End WikiText-2 KL Ordering (Both Models Confirm Opposite Sign at Matched Weight MSE):**
  - `HuggingFaceTB/SmolLM2-360M` (`224` layers: `121` Regime I, `103` Regime II, `119/224` `Block-H2 < H16`):
    - $E_8$: `E8_H8` (`71621153/1000000000` $\approx 0.071621$) $<$ `E8_axis` (`76987379/1000000000` $\approx 0.076987$).
    - $A_{16}$: `A16_axis` (`32586507/500000000` $\approx 0.065173$) $<$ `A16_BlockH2` (`8962707/125000000` $\approx 0.071702$) $<$ `A16_H16` (`39923259/500000000` $\approx 0.079847$; matched `2489551/31250000` $\approx 0.079666$).
  - `Qwen/Qwen2.5-0.5B` (`168` layers: `73` Regime I, `95` Regime II, `83/168` `Block-H2 < H16`):
    - $E_8$: `E8_H8` (`27420729/500000000` $\approx 0.054841$) $<$ `E8_axis` (`7264601/125000000` $\approx 0.058117$).
    - $A_{16}$: `A16_axis` (`6565661/125000000` $\approx 0.052525$) $<$ `A16_BlockH2` (`55205609/1000000000` $\approx 0.055206$) $<$ `A16_H16` (`11522779/200000000` $\approx 0.057614$; matched `28775733/500000000` $\approx 0.057551$).
- **Combined 392-Layer Projection Breakdown (`194` Regime I / `198` Regime II / `202` `BH2 < H16`) & Zero-Overhead Selector (`select_layer_arm`):**

| Projection Kind | Total Layers | Regime I (`axis < H16`) | Regime II (`H16 < axis`) | `BH2 < H16` | Default Calibration-Free Selector Arm |
| :--- | ---: | ---: | ---: | ---: | :--- |
| `up_proj` | `56` | **`37`** | `19` | `33` | `A16_axis` |
| `v_proj` | `56` | **`36`** | `20` | `30` | `A16_axis` |
| `down_proj` | `56` | **`32`** | `24` | `37` | `A16_axis` |
| `q_proj` | `56` | `27` | `29` | `25` | `A16_axis` |
| `gate_proj` | `56` | `22` | `34` | `23` | `A16_axis` |
| `o_proj` | `56` | `22` | **`34`** | `27` | `A16_candidate_probe` |
| `k_proj` | `56` | `18` | **`38`** *(both models `< 98/100`)* | `27` | `A16_candidate_probe` |

- **Concrete Selector Rule:** (0) Always select `E8_H8` for $E_8$. (1) Without calibration, default to `A16_axis` on `v_proj`/`up_proj`/`down_proj`/`q_proj`/`gate_proj` and flag `k_proj`/`o_proj` as `A16_candidate_probe`. (2) With one-pass Hessian trace $\Delta\mathcal{L}_\ell = \operatorname{tr}(H_\ell(\Sigma_{\mathrm{H16},\ell} - \Sigma_{\mathrm{axis},\ell}))$ and safety margin $\tau \ge 0$, select `A16_H16` iff $\Delta\mathcal{L}_\ell < -\tau$, else `A16_axis`.

---

## 4. Pillar 4 — Live `Lean 4.36.0` Kernel Verification (`12/12` Theorems, Strict Axiom Allowlist)

`E8VsA16HadamardLaw.lean` (`SHA-256 = 8b3d71a139e91290bcbe04ab2a01e740482f088a2febfb905d596331a106eaac`) compiles cleanly with zero imports, zero warnings, and strict `#print axioms` allowlist enforcement (`<= {propext, Classical.choice, Quot.sound}`, verified live against `native_decide`, `decide +native`, and custom `axiom` injections) on live `Lean 4.36.0` (`x86_64-unknown-linux-gnu, commit 611c117bfaf7b7ae1cfcf246ce6fba31e8f509e9, Release`). Theorems 1–9 formalize the lattice, syndrome, Hadamard orthogonality, non-isometry, and DC-concentration geometry directly from definitions in Lean 4; Theorems 10–12 prove the exact scaled-integer error inequalities and trace-zero regime-flip identities whose global optimality over all `2,048` Extended Hamming cosets of $A_{16}$ and over $E_8$ is independently certified over exact $\mathbb{Q}(\sqrt{2})$ in `e8_vs_a16_hadamard_oracle.py` and re-verified by `e8_vs_a16_hadamard_audit.py`:

| # | Theorem (`Aletheia.E8VsA16HadamardLaw.*`) | Live `Lean 4.36.0` `#print axioms` |
|---:|:---|:---|
| 1 | `h8_orthogonal_and_non_square` | `[propext, Quot.sound]` |
| 2 | `h16_orthogonal_and_exact_rational` | `[propext]` |
| 3 | `block_h2_non_square_and_int8_peak_safe` | `[propext, Classical.choice, Quot.sound]` |
| 4 | `h16_all_16_rows_zero_syndrome_in_c16` | `[propext, Classical.choice, Quot.sound]` |
| 5 | `e8_all_8_hadamard_rows_are_shortest_roots` | `[propext, Classical.choice, Quot.sound]` |
| 6 | `a16_axis_vs_hadamard_4x_coarsening` | `[propext]` |
| 7 | `block_h_axis_anisotropy_progression_a16_vs_e8` | `[propext, Classical.choice, Quot.sound]` |
| 8 | `block_h4_non_isometry_on_a16` | `[propext]` |
| 9 | `coherent_dc_concentration_16x_vs_8x_vs_2x` | `[propext, Quot.sound]` |
| 10 | `simultaneous_opposite_sign_single_spike_3_2` | `[propext, Classical.choice, Quot.sound]` |
| 11 | `simultaneous_opposite_sign_zero_mse_gap_matrix_witness` | `[propext, Classical.choice, Quot.sound]` |
| 12 | `regime_flip_zero_mse_gap_criterion` | `[propext, Classical.choice, Quot.sound]` |

---

## 5. Frozen 6-File Manifest & Verification Receipts

| File | Lines | Bytes | Mode | SHA-256 |
| :--- | ---: | ---: | :---: | :--- |
| `E8VsA16HadamardLaw.lean` | `214` | `9,358` | `0o644` | `8b3d71a139e91290bcbe04ab2a01e740482f088a2febfb905d596331a106eaac` |
| `e8_vs_a16_hadamard_oracle.py` | `1,091` | `42,871` | `0o644` | `5c3aeae411df30cfcce62ee2822d5b1923237f0f57e292364343c266dece04c5` |
| `e8_vs_a16_hadamard_audit.py` | `983` | `46,535` | `0o644` | `43d15641d720d84241724470a742aaed243599fd38f1ec52fa6140710b6f1dca` |
| `test_e8_vs_a16_hadamard_r5.py` | `1,184` | `63,518` | `0o644` | `030c781b1f8a58ac584ec53a58659ec9b294f4549668e1cd16b6988cd9db0f43` |
| `e8_vs_a16_hadamard_certificate.json` | `1` | `6,360` | `0o600` | `914a84512d7026b23af0672d4b175128534a7d3c8a0fbc9152b28bc38ea8f1ed` |
| `E8_VS_A16_HADAMARD_LAW_REPORT.md` | *(self)* | *(self)* | `0o644` | *(self)* |

- **Pinned Certificate Digests (`0o600`, `6,360` B `< 10,240` B):**
  - `payload_sha256`: `0f2c38a9b8daf0a1e6615fe76ff4aef73ddcdacb70e567d66d9e7c7c933f9a12`
  - `certificate_sha256`: `b5cafe89c5cc084d5b28163754887c05de08cdab17023b9b0abbfe2ebe349c0d`
- **Verification Summary (`100.0%` PASS):**
  - **R4 3-Mode Isolated Parity (`-I -B`, `-I -O -B`, `-I -B -W error`) + `PYTHONHASHSEED=42 -B` + `umask 077`:** `--standalone` `914a8451...` (bit-identical to disk), `--check e8_vs_a16_hadamard_certificate.json --json` `8774c19d...`, `test_e8_vs_a16_hadamard_r5.py` `12/12 OK` across all modes, `umask 077` `OK`.
  - **Hardened Verification Gates:** Strict `#print axioms` allowlist (`<= {propext, Classical.choice, Quot.sound}`), pre-parse integer-digit ceiling (`MAX_INT_DIGITS = 1000`) and fail-closed typed schema guard on `verify_empirical_anchor_file`.
  - **5-Surface Adversarial Fuzzer (`CLI`, `FR1`, `FR0`, `RT_FR0`, `SEC_DIRECT`):** **`101/101 PASS`** (`25 + 19 + 19 + 19 + 19`, `bypasses = 0`, `crashes = 0`).
  - **Live `aletheia mutate --workers 4 --max-mutants 0`:** `e8_vs_a16_hadamard_oracle.py` **`79/79` (`100.0%`)** + `e8_vs_a16_hadamard_audit.py` **`116/116` (`100.0%`)** = **`195/195` (`100.0%`, `survived = 0`, `truncated = False`)**.
  - **Whole-File DFS AST `BoolOp` (`OP_SWAP + SUBCOND_DEL`):** `e8_vs_a16_hadamard_oracle.py` **`20/20 + 40/40 = 60/60`** + `e8_vs_a16_hadamard_audit.py` **`26/26 + 125/125 = 151/151`** = **`211/211` (`100.0%`, `survived = []`)**.
  - **`aletheia audit-check . --lean-audit --strict-ancestors --safety-scan --scripts e8_vs_a16_hadamard_audit.py test_e8_vs_a16_hadamard_r5.py`:** **`PASS`** (`6` files, `2/2` R4, `1/1` Lean clean, `0` findings, `0` notices, `0` leaks, `0` CRLF, `0` `.pyc`).
