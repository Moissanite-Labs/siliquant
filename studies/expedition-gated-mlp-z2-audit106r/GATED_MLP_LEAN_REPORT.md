# Track S9 (`AUDIT-106`): Gated-MLP $\mathbb{Z}_2$ Gauge Conservation (Muon + Learnable Multipliers), Symmetry-Breaking Controls & `weft-loom#44` / `AUDIT-059` Exact Verification

- **Audit ID:** `AUDIT-106` (hardened revision `AUDIT-106R`: strict `exact_stratified_two_block_tail` integer validation, `up`/`down` denominator parity enforcement, Falcon `up`/`down` `~50%` orthogonal-invariant check, unstripped enum strings, and `B3` zero-sum-column wording fix)
- **Governance Standard:** Aletheia v2.0.0 (pure Python 3 stdlib `int`/`Fraction`, `0` floats, `0` bare `assert`s, strict Law 4 `N0`/`N1` attribution, strict Law 5 zero-metaphor exact operators)
- **Pinned Upstream Source:** `Moissanite-Labs/weft-loom#44` (`experiment/sign-lineage-2026-11-21` @ `e660cd3de1ebd8fac33c60a3f65f5ccabb5ed9f1`), `real-models/out/weights/weight_lean.json` (`sha256: 1123f698ee72ebb17d31f56f672b6641ad674382234a96226d9cf7a7b7ce00f9`)
- **Novelty Classification (Law 4 Honest Default):** **`N0`** for all classical & upstream primitives (SwiGLU Shazeer 2020; Muon/Newton–Schulz Jordan 2024; Learnable Multipliers `arXiv:2601.04890`; exact hypergeometric/binomial tests; `weft-loom#44` gated-MLP $\mathbb{Z}_2$ binomial/cluster sign-lineage law); **`N1` (exact analysis/audit)** strictly for the pure-$\mathbb{Q}$ extension proving $\mathbb{Z}_2$ equivariance of **separate & fused `gate_up` Muon (Newton–Schulz)** and **Learnable Multipliers (including shared row multipliers)**, plus the exact symmetry-breaking witnesses narrowing the open Falcon-H1 Tiny weight-lean cause.

---

## 1. Pillar 1: Exact Pure-$\mathbb{Q}$ $\mathbb{Z}_2$ Gauge Conservation No-Go Theorem

For a gated MLP block $y = s_d W_d \big(\phi(s_g \, \mathrm{diag}(r) W_g x) \odot (s_u \, \mathrm{diag}(r) W_u x)\big)$ under unit-wise sign reflection $\sigma_D: (W_g, W_u, W_d) \mapsto (W_g, D W_u, W_d D)$ ($D = \mathrm{diag}(\pm 1)$), `audit_z2_conservation_muon_lrm_exact()` verifies **exact `0/1` equivariance residuals** on non-trivial rational weights across every component of Falcon-H1-Tiny's publicized architecture/optimizer stack:

1. **Forward Output & Loss:** Exact `0/1` max output diff and `0/1` loss diff.
2. **Exact Backprop Gradients:** $\nabla_{W_g}' = \nabla_{W_g}$, $\nabla_{W_u}' = D \nabla_{W_u}$, $\nabla_{W_d}' = (\nabla_{W_d}) D$ (`0/1` residual).
3. **Learnable Multipliers (`arXiv:2601.04890`):** Even with a **shared row multiplier $r_i$** scaling both `gate_i` and `up_i`, $\nabla_r \mathcal{L}$ is strictly even (`0/1` diff across mirror twins).
4. **Separate & Fused `gate_up_proj` Muon (Newton–Schulz Odd Polynomial $P(G) = G(c_0 I + c_1 G^\top G + c_2 (G^\top G)^2)$):** Because $S = \mathrm{diag}(I_h, D)$ is orthogonal ($S^\top S = I_{2h}$), $(S G_{gu})^\top (S G_{gu}) = G_{gu}^\top G_{gu}$ and $P(S G_{gu}) = S P(G_{gu})$ identically (`0/1` residual on both separate and fused matrices!).
5. **RMSNorm on Hidden Vector $h$:** Preserves exact `0/1` forward invariance.
6. **Observable Sign Flip:** $\langle w_{g,i}, (D W_u)_i \rangle = D_{ii} \langle w_{g,i}, w_{u,i} \rangle$ flips sign strictly on every flipped unit.

**Implication:** Trained from independent zero-symmetric initialization, neither Muon (separate or fused), nor Learnable Multipliers, nor anti-curriculum/multi-epoch data mixtures, nor GRPO/logit-only losses can generate Falcon-H1-Tiny's `70,500 / 90,112` (`78.24%`) or `12,162 / 18,432` (`65.98%`) weight lean.

---

## 2. Pillar 2: Exact Pure-$\mathbb{Q}$ Symmetry-Breaking Must-Fail Witnesses

`audit_z2_symmetry_breaking_controls_exact()` proves three exact mechanisms that **do** break $\mathbb{Z}_2$:
- **`B1` (Hidden-Vector Mean Subtraction / LayerNorm Centering $h - \bar{h}$):** Non-zero forward divergence across mirror twins (`> 0`).
- **`B2` (Asymmetric Hidden-Unit Penalty $-\alpha \sum_i h_i$):** Breaks mirror-twin symmetry in one exact gradient step (`twin_inner_sum != 0/1`).
- **`B3` (Correlated Initialization $W_u^{(0)} = W_g^{(0)}$ with Zero-Sum Columns and Balanced Row-Sum Signs):** Produces `4/4` (`100%`) positive $\langle w_{g,i}, w_{u,i} \rangle$ while keeping `up_rowsum_positive` at exact `2/4` (`50%`, row sums `+1, -1, +1, -1`) — matching Falcon-H1-Tiny's empirical orthogonal-invariant signature (`gate_up = 78.24%` while `up_rowsum = 50.05%` and `down_colsum = 50.40%`).

---

## 3. Pillar 3: Exact Verification of `weft-loom#44` `weight_lean.json` & `AUDIT-059` Rulings

- **`Falcon-H1-Tiny-R-0.6B` & `pre-GRPO`:** Identical across all `44 / 44` layers (`70,500 / 90,112 = 78.2360%`; exact two-sided layer sign-test $p = 1 / 8{,}796{,}093{,}022{,}208$; `up_rowsum = 50.0499%`, `down_colsum = 50.4028%`).
- **`Falcon-H1-Tiny-90M-Base`:** `12,162 / 18,432 = 65.9831%` across all `24 / 24` layers ($p = 1 / 8{,}388{,}608$; `up_rowsum = 49.4303%`, `down_colsum = 49.3924%`).
- **All `5 / 5` SmolLM2 & Qwen3.5 Gated Controls:** Within `0.15%` of `50.0%` (`49.85%–50.10%`), `0/5` all-layer $> 50\%$.
- **`AUDIT-059` Exact Narrowed Statistics:** All-24-layer `Qwen3.5-0.8B` dark anisotropy `21 / 24` exact two-sided $p = 2325 / 8{,}388{,}608$ (`2.7716e-4`); L15 stratified exact $p < 10^{-9}$ (`3.0805e-10`); L19 individual blocks `1.30e-2` and `2.81e-2` (both $> 0.01$, combined stratified $p = 1.42 \times 10^{-3}$); `Gemma 4 E4B` L10 vs L9 `mean_out` margin `3101 / 237659 = 1.3048%`.

---

## 4. Reproduction Commands

```bash
# 1. Offline standalone verification (recomputes Pillars 1-3 + replays pinned weft-loom#44 summary)
python3 -I -B -W error symbiosis/stream/track-s9-gated-mlp-lean/gated_mlp_lean_audit.py --standalone --json

# 2. Live recount from a local weft-loom#44 weight_lean.json file
python3 -I -B -W error symbiosis/stream/track-s9-gated-mlp-lean/gated_mlp_lean_audit.py --live --weight-lean-json <path/to/weight_lean.json> --output symbiosis/stream/track-s9-gated-mlp-lean/gated_mlp_lean_certificate.json

# 3. Assertion-free R5 test suite
python3 -I -O -B -W error symbiosis/stream/track-s9-gated-mlp-lean/test_gated_mlp_lean_r5.py
```
