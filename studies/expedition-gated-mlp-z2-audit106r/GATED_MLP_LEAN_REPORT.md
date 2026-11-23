# Track S9 (`AUDIT-106R2`): Pure-$\mathbb{Q}$ Rational Gated-MLP $\mathbb{Z}_2$ Gauge Equivariance Witness (SiLU-like Gate, RMS-like Norm, Muon & Learnable Multipliers), Symmetry-Breaking Controls & `weft-loom#44` / `AUDIT-059` Exact Verification

- **Audit ID:** `AUDIT-106` (revision `AUDIT-106R2`: narrows all claims strictly to the implemented pure-$\mathbb{Q}$ rational surrogate operators, adds full analytical backprop + LRM + separate/fused Muon equivariance verification on the `RMS-like` normalization branch verified by pure-$\mathbb{Q}$ dual numbers, replaces causal-impossibility wording with the exact distributional equivariance statement & symmetry assumptions, and qualifies layer sign-test $p$-values with their independent-layer null assumption)
- **Governance Standard:** Aletheia v2.0.0 (pure Python 3 stdlib `int`/`Fraction`, `0` floats, `0` bare `assert`s, strict Law 4 `N0`/`N1` attribution, strict Law 5 zero-metaphor exact operators)
- **Pinned Upstream Source:** `Moissanite-Labs/weft-loom#44` (`experiment/sign-lineage-2026-11-21` @ `e660cd3de1ebd8fac33c60a3f65f5ccabb5ed9f1`), `experiments/sign-lineage-2026-11-21/real-models/out/weights/weight_lean.json` (`sha256: 1123f698ee72ebb17d31f56f672b6641ad674382234a96226d9cf7a7b7ce00f9`)
- **Novelty Classification (Law 4 Honest Default):** **`N0`** for all classical & upstream primitives (SwiGLU Shazeer 2020; Muon/Newton–Schulz Jordan 2024; Learnable Multipliers `arXiv:2601.04890`; exact hypergeometric/binomial tests; `weft-loom#44` gated-MLP $\mathbb{Z}_2$ binomial/cluster sign-lineage measurement); **`N1` (exact analysis/audit)** strictly for the pure-$\mathbb{Q}$ surrogate witness verifying exact mirror-pair $\mathbb{Z}_2$ equivariance of **separate & fused `gate_up` Muon (odd Newton–Schulz polynomial)**, **Learnable Multipliers (including shared row multipliers)**, and **rational `RMS-like` normalization (both forward and analytical backprop)**, together with the three exact symmetry-breaking witnesses (`B1`–`B3`).

---

## 1. Pillar 1: Exact Pure-$\mathbb{Q}$ Surrogate $\mathbb{Z}_2$ Gauge Equivariance Witness

### 1.1 Explicit Operator Scope (Rational Surrogates vs. Real-Valued Published Operators)
To remain strictly inside exact rational arithmetic ($\mathbb{Q}$, zero floating-point rounding), `gated_mlp_lean_oracle.py` certifies the following **rational surrogate operators**, NOT real-valued transcendental `SiLU` or square-root `RMSNorm`:
- **Gate surrogate (`SiLU-like`, NOT transcendental `SiLU`):** $\phi(g) = \frac{1}{2} g \left(1 + \frac{g}{1 + |g|}\right)$ (non-odd, $\phi(0)=0, \phi'(0)=1/2$; distinct from $\mathrm{SiLU}(g) = g/(1+e^{-g})$).
- **Normalization surrogate (`RMS-like`, NOT square-root `RMSNorm`):** $z_i = \frac{h_i}{1 + \frac{1}{H}\sum_{k=1}^H h_k^2}$ (`hidden_mode="rms_like_sq_denom"`; distinct from $\mathrm{RMSNorm}(h)_i = h_i / \sqrt{\epsilon + \frac{1}{H}\sum_k h_k^2}$, though both depend on $h$ in the denominator strictly through sign-even squares $h_k^2$).

**Honest scope boundary:** This executable fixture verifies exact $\mathbb{Z}_2$ equivariance strictly on the implemented pure-$\mathbb{Q}$ rational surrogates above. It is **not** a real-valued proof or exact verification of Falcon-H1-Tiny's full published real-valued `SiLU`/`RMSNorm` architecture.

### 1.2 Exact Mirror-Pair Equivariance Results (`0/1` Residuals on Both Branches)
For the rational gated MLP block $y = s_d W_d \, \mathcal{N}\!\big(\phi(s_g \, \mathrm{diag}(r) W_g x) \odot (s_u \, \mathrm{diag}(r) W_u x)\big)$ under unit-wise sign reflection $\sigma_D: (W_g, W_u, W_d) \mapsto (W_g, D W_u, W_d D)$ ($D = \mathrm{diag}(\pm 1)$), `audit_z2_conservation_muon_lrm_exact()` verifies **exact `0/1` equivariance residuals** on non-trivial rational weights across **BOTH** the unnormalized branch (`hidden_mode="none"`) AND the rational `RMS-like` normalization branch (`hidden_mode="rms_like_sq_denom"`, using exact analytical chain-rule backprop $\frac{\partial \mathcal{L}}{\partial h_i} = \frac{g_i}{q} - \frac{2 h_i}{H q^2}\langle g, h\rangle$ where $q = 1 + \frac{1}{H}\sum_k h_k^2$, independently cross-checked to `0/1` against pure-$\mathbb{Q}$ dual numbers in `test_gated_mlp_lean_r5.py`):

1. **Forward Output & Loss (both `none` and `rms_like_sq_denom`):** Exact `0/1` max output diff and `0/1` loss diff.
2. **Exact Analytical Backprop Gradients (both `none` and `rms_like_sq_denom`):** $\nabla_{W_g}' = \nabla_{W_g}$, $\nabla_{W_u}' = D \nabla_{W_u}$, $\nabla_{W_d}' = (\nabla_{W_d}) D$ (`0/1` residual).
3. **Learnable Multipliers (`arXiv:2601.04890`, both `none` and `rms_like_sq_denom`):** Even with a **shared row multiplier $r_i$** scaling both `gate_i` and `up_i`, $\nabla_r \mathcal{L}$ is strictly even (`0/1` diff across mirror twins).
4. **Separate & Fused `gate_up_proj` Muon (odd Newton–Schulz polynomial $P(G) = G(c_0 I + c_1 G^\top G + c_2 (G^\top G)^2)$, both `none` and `rms_like_sq_denom`):** Because $S = \mathrm{diag}(I_h, D)$ is orthogonal ($S^\top S = I_{2h}$), $(S G_{gu})^\top (S G_{gu}) = G_{gu}^\top G_{gu}$ and $P(S G_{gu}) = S P(G_{gu})$ identically (`0/1` residual on both separate and fused matrices on both branches).
5. **Observable Sign Flip:** $\langle w_{g,i}, (D W_u)_i \rangle = D_{ii} \langle w_{g,i}, w_{u,i} \rangle$ flips sign strictly on every flipped unit.

### 1.3 Exact Distributional Equivariance Corollary & Explicit Non-Claims
- **Distributional statement and assumptions:** The verified identity pairs each parameter state $\theta$ with its mirror twin $\sigma_D(\theta)$ along exact single-step updates on the rational surrogate witness. Under the symmetry assumptions that initialization, data/minibatch order, stochastic choices (e.g., RNG/dropout), and numerical tie-breaking are invariant in distribution under $\sigma_D$, the induced distribution of trajectories and of $\langle w_{g,i}, w_{u,i} \rangle$ is symmetric about zero.
- **What this witness does NOT prove:** Distributional equivariance does **not** prove that every individual finite training trajectory is symmetric or has zero probability of producing an asymmetric sample. The witness tests one pure-$\mathbb{Q}$ rational surrogate fixture rather than Falcon-H1-Tiny's actual training trajectories; accordingly, it **does not rule out finite-run asymmetry** and **does not identify Falcon-H1-Tiny's unreleased training cause**.

---

## 2. Pillar 2: Exact Pure-$\mathbb{Q}$ Symmetry-Breaking Must-Fail Witnesses

`audit_z2_symmetry_breaking_controls_exact()` verifies three exact mechanisms on the rational fixture that **do** break $\mathbb{Z}_2$ mirror-pair symmetry:
- **`B1` (Hidden-Vector Mean Subtraction / LayerNorm Centering $h - \bar{h}$):** Non-zero forward divergence across mirror twins (`2122564703/5974041600 > 0`).
- **`B2` (Asymmetric Hidden-Unit Penalty $-\alpha \sum_i h_i$):** Breaks mirror-twin symmetry in one exact gradient step (`twin_inner_sum != 0/1`).
- **`B3` (Correlated Initialization $W_u^{(0)} = W_g^{(0)}$ with Zero-Sum Columns and Balanced Row-Sum Signs):** Produces `4/4` (`100%`) positive $\langle w_{g,i}, w_{u,i} \rangle$ while keeping `up_rowsum_positive` at exact `2/4` (`50%`, row sums `+1, -1, +1, -1`), illustrating one synthetic mechanism consistent with an asymmetric `gate_up` inner product alongside near-50% row/column sums (`Falcon-H1-Tiny-R-0.6B`: `gate_up = 78.24%`, `up_rowsum = 50.05%`, `down_colsum = 50.40%`).

---

## 3. Pillar 3: Exact Verification of `weft-loom#44` `weight_lean.json` & `AUDIT-059` Rulings

**Null-model qualification for all layer sign-test $p$-values below:** Every layer sign-test $p$-value reported here is the exact two-sided binomial tail probability strictly **under an independent-layer symmetric ($\Pr = 1/2$) null assumption**. Layers within a single trained model can be correlated through shared training dynamics, initialization, or residual stream coupling and therefore do **not** automatically satisfy the independent-layer null.

- **`Falcon-H1-Tiny-R-0.6B` & `pre-GRPO`:** Identical across all `44 / 44` layers (`70,500 / 90,112 = 78.2360%`; exact two-sided binomial $p = 1 / 8{,}796{,}093{,}022{,}208$ under the independent-layer $\Pr=1/2$ null; `up_rowsum = 50.0499%`, `down_colsum = 50.4028%`).
- **`Falcon-H1-Tiny-90M-Base`:** `12,162 / 18,432 = 65.9831%` across all `24 / 24` layers (exact two-sided binomial $p = 1 / 8{,}388{,}608$ under the independent-layer $\Pr=1/2$ null; `up_rowsum = 49.4303%`, `down_colsum = 49.3924%`).
- **All `5 / 5` SmolLM2 & Qwen3.5 Gated Controls:** Within `0.15%` of `50.0%` (`49.85%–50.10%`), `0/5` all-layer $> 50\%$.
- **`AUDIT-059` Exact Narrowed Statistics:** All-24-layer `Qwen3.5-0.8B` dark anisotropy `21 / 24` (exact two-sided binomial $p = 2325 / 8{,}388{,}608 = 2.7716\times 10^{-4}$ under the independent-layer $\Pr=1/2$ null); L15 stratified exact hypergeometric tail $p < 10^{-9}$ (`3.0805e-10`); L19 individual blocks `1.30e-2` and `2.81e-2` (both $> 0.01$, combined stratified $p = 1.42 \times 10^{-3}$); `Gemma 4 E4B` L10 vs L9 `mean_out` margin `3101 / 237659 = 1.3048%`.

---

## 4. Reproduction Commands

From the repository root (Python 3 stdlib only, zero external packages, zero network calls):

```bash
# 1. Offline standalone verification (recomputes Pillars 1-3 + replays pinned weft-loom#44 summary)
python3 -I -B -W error studies/expedition-gated-mlp-z2-audit106r/gated_mlp_lean_audit.py --standalone --json

# 2. Live recount from a local weft-loom#44 weight_lean.json file
python3 -I -B -W error studies/expedition-gated-mlp-z2-audit106r/gated_mlp_lean_audit.py --live --weight-lean-json <path/to/weight_lean.json> --output studies/expedition-gated-mlp-z2-audit106r/gated_mlp_lean_certificate.json

# 3. Assertion-free R5 test suite (includes pure-Q dual-number exact Jacobian checks for all 3 hidden_mode branches)
python3 -I -O -B -W error studies/expedition-gated-mlp-z2-audit106r/test_gated_mlp_lean_r5.py
```
