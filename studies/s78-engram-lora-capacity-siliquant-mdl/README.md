# Engram LoRA Bit-Capacity, Two-Part MDL, and Joint `(bpw, rank)` Bit-Allocation Law (`Lean 4.36.0` Formal Companion)

Pure-`Init` machine-checked `Lean 4.36.0` formalization (`17/17` theorems in `LoraCapacityJointMdl.lean`; `0` external imports, `0` compiler warnings, **literal zero axioms `[]`** on every theorem — not even `propext`, `Quot.sound`, or `Classical.choice`) paired with a deterministic pure-$\mathbb{Z}/\mathbb{Q}$ certified interval oracle, hardened Zone-B CLI verifier, adversarial unit test suite, and cryptographically pinned certificate (`s78_engram_lora_siliquant_certificate.json`).

- **Study ID:** `TRACK-S78-ENGRAM-LORA-CAPACITY-SILIQUANT-MDL`
- **Schema Version:** `aletheia.track_s78.engram_lora_siliquant_mdl.certificate.v1`
- **Oracle Class / Novelty Tier:** `exact` / `N1` (pure `int` / `Fraction` certified outward-rounded interval arithmetic; `0` AST floats, `0` `float()`, `0` bare `assert`s, `0` RNG; analytic `3.0 bpw` KL point explicitly quarantined as `ANALYTIC_S34_EXTRAPOLATION_NOT_LIVE_GPU_MEASURED`)
- **Related PRs:** Refs #99, #100, #103
- **Full Technical Report:** [`REPORT.md`](REPORT.md)

## Key Verified Results

1. **Exact Multiplicative Gauge-Quotient × Marchenko–Pastur Reconciliation:**
   $$C_{\mathrm{MP}}(r) = \frac{1}{2\ln 2}\, P_{\mathrm{raw}}(r)\,\eta_{\mathrm{alg}}(r)\,\kappa(r)$$
   where $P_{\mathrm{raw}}(r) = r(m+n)$, $\eta_{\mathrm{alg}}(r) = 1 - \frac{r}{m+n}$ is the $GL_r(\mathbb{R})$ gauge-quotient efficiency (`dEff + r^2 = pRaw`), and $\kappa(r)$ is the data-aligned Eckart–Young spectral multiplier satisfying $1 < \kappa(0^+) = 1 + \frac{2\sqrt{mn}}{m+n} \le 2$ (saturating `2` on square matrices), strictly decreasing in $r$, and collapsing identically to $\kappa(m) = 1$ at full rank. Pure-$\mathbb{Z}/\mathbb{Q}$ intervals reproduce every Engram benchmark point (`1.401440` bpp on `512 × 1024`, `1.321133` bpp on `1024 × 3584`, `[191493.686, 191493.689]` bits = `23.937 KB` / `0.973987` bpp at `r = 128`, and `0.480898` bpp / $\kappa = 1.000000$ at full rank `r = 512`).
2. **Exact Convex Joint `(bpw, rank)` Two-Part MDL Allocation Law:**
   - Compressing frozen base weights from $b_1 \to b_2 < b_1$ under a fixed total bit budget $B_{\mathrm{tot}} = b_{\mathrm{base}} N_{\mathrm{base}} + b_{\mathrm{LoRA}} P_1 r$ unlocks $\Delta r = \lfloor (b_1 - b_2) N_{\mathrm{base}} / (b_{\mathrm{LoRA}} P_1) \rfloor$ LoRA rank and strictly dominates $(b_1, r_1)$ iff $T < T_{\mathrm{crit}} := \frac{\ln 2 \cdot \Delta C_{\mathrm{LoRA}}(r_1 \to r_2)}{K_{\mathrm{base}}(b_2) - K_{\mathrm{base}}(b_1)}$.
   - On **`Qwen2.5-3B`** inside a `6.0 GiB` weight+adapter envelope, **`4.5 bpw siliquant QuadSnake` (`r = 128`) strictly beats `BF16` (`r = 16`) up to `6.929B` WikiText / `7.828B` Python tokens** while expanding maximum rank headroom **`15.81×`** (`r_max`: `72 -> 1,138` at `16`-bit adapter storage, `18 -> 284` with FP32 AdamW states).
   - On **`Gemma-3-4B-IT`**, `BF16` (`7.2269 GiB`) cannot fit inside a `6.0 GiB` envelope (`r_max = 0`), whereas `4.504 bpw siliquant hZBG128` (`2.9327 GiB`) unlocks `r_max = 884` (`221` AdamW). Under the verified log-convex base KL law $K_{\mathrm{base}}(b) = C \cdot 2^{-\alpha b}$, exact $\ln 2$ cancellation yields unconditional strict convexity $d^2 J / db^2 > 0$ and exact lower-bitrate floor **`b_floor(1B) = 4.3226 bpw`** (`4.5 bpw` stays above the floor, while the quarantined analytic `3.0 bpw` extrapolation crosses below both single-matrix `4.3226` and multi-layer `3.6293` floors).
3. **Formal `Lean 4.36.0` Kernel Certification (`17/17` Zero-Axiom `[]` Theorems):**
   - `LoraCapacityJointMdl.lean` certifies gauge-quotient DOF conservation, full-rank Wilks closure, strict monotonicity of $\eta_{\mathrm{alg}}(r)$, spectral-edge dominance and AM-GM ceiling $\kappa(0^+) \le 2$, finite-spectrum cumulative-average monotonicity, offline-vs-online two-part MDL cancellation, noise memorization no-profit, forgetting/learning conservation, capacity-without-OOD obstruction, joint budget conservation, exact dominance phase boundary, convex KL lower-bitrate floor obstruction, and exact frozen `siliquant` receipt witnesses with **`0` axioms `[]`**.

## Bundle Contents (`6` Frozen Verified Artifacts + This README)

- `LoraCapacityJointMdl.lean` — `17/17` pure-`Init` `Lean 4.36.0` theorems (`0` imports, `0` warnings, `0` axioms `[]`).
- `s78_engram_lora_siliquant_mdl.py` — deterministic pure-$\mathbb{Z}/\mathbb{Q}$ certified interval oracle (`0` floats, `0` bare `assert`s, `0` RNG).
- `verify_s78_engram_lora_mdl.py` — hardened Zone-B CLI verifier (`FR1` full oracle replay, `FR0` independent 4-pillar audit, raising `secondary_audit_payload`, atomic `dir_fd` + `O_NONBLOCK` + `st_nlink == 1` + `$PWD` symlink checks, comment-only Lean static scanner + SHA-256 pin + `lean --stdin`).
- `test_s78_engram_lora_mdl_r5.py` — adversarial Zone-B, live-Lean exploit, I/O security, and pure-$\mathbb{Z}/\mathbb{Q}$ unit test suite (`10` test methods).
- `s78_engram_lora_siliquant_certificate.json` — canonical JSON certificate (`payload_sha256 = 36c8e2fd9a8d493cf94bc9d6f60ee031f65c15365e61673945623a2f62ebbf4e`, `certificate_sha256 = d321dbaeca8884e492a73a638280a06dbbb5901526f874114700a2a743e5b9ed`).
- `REPORT.md` — full technical report, epistemic quarantine, benchmark tables, Lean theorem ledger, and file manifest.

## Quick Verification

Run from inside `studies/s78-engram-lora-capacity-siliquant-mdl/`:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -I -B verify_s78_engram_lora_mdl.py --check s78_engram_lora_siliquant_certificate.json --lean LoraCapacityJointMdl.lean --json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -I -B test_s78_engram_lora_mdl_r5.py -v
lean LoraCapacityJointMdl.lean
```
