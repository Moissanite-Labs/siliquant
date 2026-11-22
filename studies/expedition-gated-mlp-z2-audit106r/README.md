# Pure-$\mathbb{Q}$ Rational Gated-MLP $\mathbb{Z}_2$ Gauge Equivariance Witness (SiLU-like Gate, RMS-like Norm, Muon & Learnable Multipliers), Symmetry-Breaking Controls & `weft-loom#44` / `AUDIT-059` Exact Verification (`AUDIT-106R2`)

> **Status:** `AUDIT-106R2` revision addressing review feedback on PR `#31` (`honest_novelty_tier: N1` exact pure-$\mathbb{Q}$ rational surrogate witness & audit over `N0` classical and upstream primitives).
> **Runtime Dependencies:** Python 3 standard library only (`int`, `fractions.Fraction`; zero floats on trust paths or JSON, zero bare `assert`s).
> **Context:** Self-contained pure-$\mathbb{Q}$ companion study backing exact recount checks for `Moissanite-Labs/weft-loom#44 @ e660cd3d` and `AUDIT-059`, with explicit rational-surrogate and distributional-equivariance scope boundaries.

## 1. Contents (Zero Edits to Existing `siliquant` Files)

| File | SHA-256 | Role |
|---|---|---|
| [`gated_mlp_lean_oracle.py`](gated_mlp_lean_oracle.py) | `e488473a38b845d7477026f65be8b221537b481a8da88eb403dc6a80bc9777c0` | Exact pure-$\mathbb{Q}$ $\mathbb{Z}_2$ equivariance witness on rational surrogates (`SiLU-like` gate $\phi(g)$, `RMS-like` $h/(1+\mathrm{mean}(h^2))$ with full forward & analytical backprop, separate & fused `gate_up` Muon Newton–Schulz, Learnable Multipliers incl. shared row multipliers), 3 exact symmetry-breaking controls (`B1`–`B3`), exact hypergeometric/binomial/stratified tails for `weft-loom#44` & `AUDIT-059` |
| [`gated_mlp_lean_audit.py`](gated_mlp_lean_audit.py) | `b5b9dcef0cbc8cf906ab563d3d4af2ce8c9a5ec22674dd5aae0ab5ec7231785d` | Dual-mode CLI (`--standalone` 100% offline fixture replay vs. `--live --weight-lean-json <path>` read-only recount with per-layer conservation & denominator parity) |
| [`test_gated_mlp_lean_r5.py`](test_gated_mlp_lean_r5.py) | `1349da3a1e48ad8bb6dca212cebd8b1d34ded09412aa78b47f5d044896fe9ff3` | Assertion-free R5 known-answer, boundary, pure-$\mathbb{Q}$ dual-number exact Jacobian (all 3 `hidden_mode` branches), and mutation-killing suite (`4/4` OK) |
| [`gated_mlp_lean_certificate.json`](gated_mlp_lean_certificate.json) | `e768711cafe87526000b24477cedeacaad30a68baf9118f9ddbc2c358438e47c` | Pinned `--live` canonical certificate recounting `weft-loom#44` `weight_lean.json` (`1123f698...`; inner `certificate_sha256: 97205329...`) |
| [`GATED_MLP_LEAN_REPORT.md`](GATED_MLP_LEAN_REPORT.md) | `a03e734a55459d52dceb1f0c5791868145f5c19c711adf18f06923cc77c2429f` | Technical summary of Pillars 1–3, explicit surrogate vs. published-operator scope, distributional assumptions, and independent-layer null qualification |

## 2. Highlights & Explicit Scope Boundaries

1. **Pillar 1 — Exact Pure-$\mathbb{Q}$ Rational Surrogate $\mathbb{Z}_2$ Equivariance Witness (`0/1` Residuals on Both Branches):**
   - **Surrogate vs. Published Operators:** Certifies exact pure-$\mathbb{Q}$ mirror-pair equivariance on the rational `SiLU-like` gate $\phi(g) = \frac{1}{2}g(1 + g/(1 + |g|))$ (**not** transcendental `SiLU`) and rational `RMS-like` normalization $z_i = h_i / (1 + \frac{1}{H}\sum_k h_k^2)$ (**not** square-root `RMSNorm`). It is **not** a real-valued proof or exact verification of Falcon-H1-Tiny's full published architecture.
   - **Full Forward & Backprop Coverage on Both Branches:** Under $\sigma_D: (W_g, W_u, W_d) \mapsto (W_g, D W_u, W_d D)$ ($D = \mathrm{diag}(\pm 1)$), proves exact `0/1` residuals on **both** `unnormalized_branch` (`none`) and `rms_like_norm_branch` (`rms_like_sq_denom`) across forward output, loss, analytical backprop gradients (independently checked to `0/1` via pure-$\mathbb{Q}$ dual numbers), **Learnable Multipliers** (`arXiv:2601.04890`, incl. shared row multipliers $r_i$), and **separate & fused `gate_up_proj` Muon** (odd Newton–Schulz polynomial $P(S G_{gu}) = S P(G_{gu})$).
   - **Distributional Scope & Non-Claims:** Under the symmetry assumptions that initialization, data/minibatch order, stochastic choices, and tie-breaking are $\sigma_D$-invariant in distribution, the induced trajectory distribution over $\langle w_{g,i}, w_{u,i} \rangle$ is symmetric about zero. This witness tests one rational fixture rather than Falcon training trajectories; it **does not prove every individual trajectory is symmetric**, **does not rule out finite-run asymmetry**, and **does not identify Falcon-H1-Tiny's unreleased training cause**.
2. **Pillar 2 — Three Exact Symmetry-Breaking Must-Fail Controls (`B1`–`B3`):**
   - `B1`: Hidden-vector mean subtraction / LayerNorm centering ($h - \bar{h}$) produces non-zero mirror-twin divergence (`2122564703/5974041600`).
   - `B2`: Asymmetric hidden-unit penalty ($-\alpha \sum_i h_i$) breaks mirror-twin symmetry in one exact gradient step.
   - `B3`: Correlated initialization ($W_u^{(0)} = W_g^{(0)}$) with **zero-sum columns and balanced row-sum signs** gives `4/4` (`100%`) positive $\langle w_{g,i}, w_{u,i} \rangle$ while keeping `up_rowsum_positive = 2/4` (`50%`) — one synthetic example displaying an asymmetric inner product alongside near-50% row/column sums (`Falcon-H1-Tiny-R-0.6B`: `gate_up = 78.2360%`, `up_rowsum = 50.0499%`, `down_colsum = 50.4028%`).
3. **Pillar 3 — Exact Recount of `weft-loom#44` `weight_lean.json` & `AUDIT-059` Statistics:**
   - **Independent-layer null qualification:** Every layer sign-test $p$-value below is the exact two-sided binomial tail probability **strictly under an independent-layer symmetric ($\Pr = 1/2$) null assumption**; correlated layers in a trained model do not automatically satisfy layer independence.
   - `Falcon-H1-Tiny-R-0.6B` & `pre-GRPO`: `70,500 / 90,112 = 78.2360%` across all `44/44` layers (independent-layer binomial $p = 1 / 8{,}796{,}093{,}022{,}208$); `Falcon-H1-Tiny-90M-Base`: `12,162 / 18,432 = 65.9831%` across all `24/24` layers (independent-layer binomial $p = 1 / 8{,}388{,}608$); all `5/5` SmolLM2 & Qwen3.5 controls lie within `49.85%–50.10%`.
   - `AUDIT-059` exact statistics: `Qwen3.5-0.8B` dark anisotropy `21/24` (independent-layer binomial $p = 2325 / 8{,}388{,}608 = 2.7716\times 10^{-4}$), L15 stratified exact hypergeometric $p = 3.0805\times 10^{-10}$, L19 combined stratified $p = 1.42\times 10^{-3}$ (individual blocks `1.30e-2` and `2.81e-2`), `Gemma 4 E4B` L10 vs. L9 `mean_out` margin `3101 / 237659 = 1.3048%`.

## 3. Exact Offline Reproduction (4-Mode Isolated `-I` R4 Parity)

From this directory (Python 3 standard library only, zero external packages, zero network calls):

```bash
# 1. Offline standalone verification (4-mode bit-identical 8dea84eb..., inner ecd8bcc8...):
python3 -I -B gated_mlp_lean_audit.py --standalone --json | sha256sum
python3 -I -O -B gated_mlp_lean_audit.py --standalone --json | sha256sum
python3 -I -B -W error gated_mlp_lean_audit.py --standalone --json | sha256sum
PYTHONHASHSEED=42 python3 -I -B gated_mlp_lean_audit.py --standalone --json | sha256sum

# 2. Assertion-free R5 known-answer, dual-number Jacobian, boundary & mutation-killing suite (4/4 OK):
python3 -I -O -B -W error test_gated_mlp_lean_r5.py
```

**Offline vs. Live Certificate Provenance:** `--standalone --json` and `test_gated_mlp_lean_r5.py` run 100% offline from any clone and match all Pillar 1–3 values identically. [`gated_mlp_lean_certificate.json`](gated_mlp_lean_certificate.json) (`e768711c...`, inner `97205329...`) is the pinned `--live` certificate generated by recounting `Moissanite-Labs/weft-loom#44 @ e660cd3de1ebd8fac33c60a3f65f5ccabb5ed9f1` (`experiments/sign-lineage-2026-11-21/real-models/out/weights/weight_lean.json`, `sha256: 1123f698...`), differing from `--standalone` solely in `mode`/`live_json_recounted`/`diag_source`.

**Verification Receipts (`AUDIT-106R2`):** `4/4` R5 tests OK across all 4 isolated `-I` modes; `aletheia v2.0.0 mutate --max-mutants 0`: **`48/48`** (`39/39` oracle + `9/9` audit, `100.0%`) KILLED (`0` survived, `truncated=False`); exhaustive AST `BoolOp` sweep: **`32/32`** (`26/26` oracle + `6/6` audit, `100.0%`) KILLED (`0` survived); `aletheia v2.0.0 audit-check --strict-ancestors --safety-scan PASS` (`0` bare asserts, `0` leaks, `0` CRLF, `0` safety findings).
