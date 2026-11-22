# Gated-MLP $\mathbb{Z}_2$ Gauge Conservation (Muon + Learnable Multipliers), Symmetry-Breaking Controls & `weft-loom#44` / `AUDIT-059` Exact Verification (`AUDIT-106R`)

> **Status:** Independently `VERIFIED` across 2 model families / 2 continents (`AUDIT-106R`, `honest_novelty_tier: N1` exact analysis/audit over `N0` classical and upstream primitives; human review pending).
> **Seats:** Author `continuum-bridge` · Clean-Room Verifiers: `DeepSeek v4.1 Flash` (East Asia) and `verifier`.
> **Runtime Dependencies:** Python 3 standard library only (`int`, `fractions.Fraction`; zero floats on trust paths or JSON, zero bare `assert`s).
> **Context:** Directly backs our `Moissanite-Labs/siliquant#24` audit rulings and narrows the open Falcon-H1-Tiny gated-MLP weight-lean mechanism (`Moissanite-Labs/weft-loom#44 @ e660cd3d`).

## 1. Contents (Zero Edits to Existing `siliquant` Files)

All 5 core study files below are byte-identical to the frozen `AUDIT-106R` verified package:

| File | SHA-256 | Role |
|---|---|---|
| [`gated_mlp_lean_oracle.py`](gated_mlp_lean_oracle.py) | `154ececbf16849e730f35826ea08c5268ec837bac744b33dc782e092178e5d0c` | Exact pure-$\mathbb{Q}$ $\mathbb{Z}_2$ gauge-conservation prover (separate & fused `gate_up` Muon Newton–Schulz, Learnable Multipliers incl. shared row multipliers, RMSNorm), 3 exact symmetry-breaking witnesses (`B1`–`B3`), exact hypergeometric/binomial/stratified tails for `weft-loom#44` & `AUDIT-059` |
| [`gated_mlp_lean_audit.py`](gated_mlp_lean_audit.py) | `6e9b11db4c8268aa6dfe658e07ca15cde4f35bde6b10b65c17849d7284b6d19e` | Dual-mode CLI (`--standalone` 100% offline fixture replay vs. `--live --weight-lean-json <path>` read-only recount with per-layer conservation & denominator parity) |
| [`test_gated_mlp_lean_r5.py`](test_gated_mlp_lean_r5.py) | `b88b8060101116497c2f234654665452e437a47a09e4ccc6d8137284cbfe0d35` | Assertion-free R5 known-answer, boundary, and mutation-killing suite (`4/4` OK) |
| [`gated_mlp_lean_certificate.json`](gated_mlp_lean_certificate.json) | `1212a46bf1dd44b2ad80c18be70febd634b7cfddf877ceebefa38274a9870142` | Pinned `--live` canonical certificate recounting `weft-loom#44` `weight_lean.json` (`1123f698...`; inner `payload_sha256: 004b70d8...`) |
| [`GATED_MLP_LEAN_REPORT.md`](GATED_MLP_LEAN_REPORT.md) | `36c8a3393f0d21e0555112cdef4898097ff82385a98d332acc02fe099bba936c` | Technical summary of Pillars 1–3, exact statistics, and Law-4/Law-5 boundaries |

## 2. Highlights & Honest Scope Boundaries

1. **Pillar 1 — Exact Pure-$\mathbb{Q}$ $\mathbb{Z}_2$ Gauge Conservation No-Go Theorem (`0/1` Residuals):**
   - Under unit-wise sign reflection $\sigma_D: (W_g, W_u, W_d) \mapsto (W_g, D W_u, W_d D)$ ($D = \mathrm{diag}(\pm 1)$), proves exact `0/1` equivariance residuals across forward output & loss, exact backprop gradients, **Learnable Multipliers** (`arXiv:2601.04890`, including shared row multipliers $r_i$), **separate AND fused `gate_up_proj` Muon** (odd Newton–Schulz polynomial $P(S G_{gu}) = S P(G_{gu})$ for orthogonal $S = \mathrm{diag}(I_h, D)$), and **RMSNorm** on hidden state $h$.
   - **Scope:** Rules out zero-symmetric initialization + Muon (separate/fused) + Learnable Multipliers + RMSNorm + anti-curriculum/logit-only losses as the cause of Falcon-H1-Tiny's positive `gate_up` lean; does not itself inspect Falcon-H1-Tiny's unreleased training code.
2. **Pillar 2 — Three Exact Symmetry-Breaking Must-Fail Witnesses (`B1`–`B3`):**
   - `B1`: Hidden-vector mean subtraction / LayerNorm centering ($h - \bar{h}$) produces non-zero mirror-twin divergence.
   - `B2`: Asymmetric hidden-unit penalty ($-\alpha \sum_i h_i$) breaks mirror-twin symmetry in one exact gradient step.
   - `B3`: Correlated initialization ($W_u^{(0)} = W_g^{(0)}$) with **zero-sum columns and balanced row-sum signs** gives `4/4` (`100%`) positive $\langle w_{g,i}, w_{u,i} \rangle$ while keeping `up_rowsum_positive = 2/4` (`50%`) — matching Falcon-H1-Tiny's orthogonal-invariant signature (`gate_up = 78.2360%`, `up_rowsum = 50.0499%`, `down_colsum = 50.4028%`).
3. **Pillar 3 — Exact Verification of `weft-loom#44` `weight_lean.json` & `AUDIT-059` Statistics:**
   - `Falcon-H1-Tiny-R-0.6B` & `pre-GRPO`: `70,500 / 90,112 = 78.2360%` across all `44/44` layers (exact two-sided layer sign-test $p = 1 / 8{,}796{,}093{,}022{,}208$); `Falcon-H1-Tiny-90M-Base`: `12,162 / 18,432 = 65.9831%` across all `24/24` layers ($p = 1 / 8{,}388{,}608$); all `5/5` SmolLM2 & Qwen3.5 controls lie within `49.85%–50.10%`.
   - `AUDIT-059` exact statistics: `Qwen3.5-0.8B` dark anisotropy `21/24` ($p = 2325 / 8{,}388{,}608 = 2.7716\times 10^{-4}$), L15 stratified $p = 3.0805\times 10^{-10}$, L19 combined stratified $p = 1.42\times 10^{-3}$ (individual blocks `1.30e-2` and `2.81e-2`), `Gemma 4 E4B` L10 vs. L9 `mean_out` margin `3101 / 237659 = 1.3048%`.

## 3. Exact Offline Reproduction (3-Mode Isolated `-I` R4 Parity)

From this directory (Python 3 standard library only, zero external packages, zero network calls):

```bash
# 1. Offline standalone verification (3-mode bit-identical 6a62bcb5..., inner f3cb982d...):
python3 -I -B gated_mlp_lean_audit.py --standalone --json | sha256sum
python3 -I -O -B gated_mlp_lean_audit.py --standalone --json | sha256sum
python3 -I -B -W error gated_mlp_lean_audit.py --standalone --json | sha256sum

# 2. Assertion-free R5 known-answer, boundary & mutation-killing suite (4/4 OK):
python3 -I -O -B -W error test_gated_mlp_lean_r5.py
```

**Offline vs. Live Certificate Provenance:** `--standalone --json` and `test_gated_mlp_lean_r5.py` run 100% offline from any clone of `main` and match all Pillar 1–3 numbers identically. [`gated_mlp_lean_certificate.json`](gated_mlp_lean_certificate.json) (`1212a46b...`, inner `004b70d8...`) is the pinned `--live` certificate generated by recounting `Moissanite-Labs/weft-loom#44 @ e660cd3de1ebd8fac33c60a3f65f5ccabb5ed9f1` (`real-models/out/weights/weight_lean.json`, `sha256: 1123f698...`), differing from `--standalone` solely in `diag_source` (`live_json` vs. `standalone_fixture`).

**External Verification Receipts:** `4/4` R5 tests OK; `aletheia v2.0.0 mutate --max-mutants 0`: `48/48` (`39` oracle + `9` audit, `100.0%`) KILLED (`0` survivors, `truncated=False`); `aletheia v2.0.0 audit-check --strict-ancestors --safety-scan PASS` (`0` defects, `0` leaks, `0` safety findings); corroborated across `DeepSeek v4.1 Flash` (`136` probes) and `verifier` (`29` adversarial probes, `0` bypasses, `0` crashes).
