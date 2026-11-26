# Exact Operator & Roofline Laws for 2DSD, Dense-and-Sparse CSR vs. `siliquant`, and Fast-Multipole Attention (`Lean 4.36.0` Formal Companion)

Pure-`Init` machine-checked `Lean 4.36.0` formalization (`18` theorems across `TwoDSDAndDenseSparseLaws.lean`; `0` Mathlib, `0` external imports, `0` compiler warnings, `14/18` literal zero-axiom `[]` + `4/18` `[propext]` only, `0` banned axioms) paired with a pure-$\mathbb{Z}/\mathbb{Q}$ deterministic oracle, R9+/R10 static & live Lean 4 verifier, adversarial R5 test suite, and cryptographically pinned certificate (`s79_2dsd_multipole_certificate.json`).

- **Project ID:** `WAVE4B-S79-2DSD-MULTIPOLE-DENSE-SPARSE`
- **Toolchain:** `leanprover/lean4:v4.36.0`
- **Novelty Tier / Oracle Class:** `N1` / `exact` (`measured_live_gpu_in_this_bundle = false`)
- **Related PRs:** Refs #77, #95, #99, #100
- **Full Technical Report:** [`REPORT.md`](REPORT.md)

## Bundle Contents (`8` Frozen Verified Artifacts + This README)

- `lean-toolchain`, `lakefile.toml` — pure-`Init` `Lean 4.36.0` build configuration (`TwoDSDAndDenseSparseLaws`).
- `TwoDSDAndDenseSparseLaws.lean` (`18` theorems) — formal proofs covering (1) `2DSD` lossless multi-candidate cascade verification, 2D speedup dominance over 1D DFlash/LoopSpec, Eq. 6 last-loop KV sharing compression & early-loop drift obstruction; (2) exact matched-bitrate Dense-and-Sparse CSR vs. dense `siliquant` `BlockH4 QuadSnake` crossover cone, high-CSR phase-reversal barrier, simultaneous triple dominance, and exact 5th-power safe sparsity cap (`rho_s <= 1/160 = 0.625%`); and (3) `MultipoleAttention` arbitrary-dimension Lagrange Cauchy–Schwarz logit bound, first-order centroid cancellation, second-order Jensen partition underestimation, exact softmax ratio bracket, $d_k=128$ opening-angle barrier, and exact value-weight covariance residual.
- `s79_2dsd_multipole_oracle.py` — pure Python 3 stdlib $\mathbb{Z}/\mathbb{Q}$ oracle (`0` floats, `0` bare `assert`s, `0` mathematical RNG).
- `verify_s79_2dsd_multipole.py` — standalone CLI verifier, secondary auditor, and R9+/R10 static & live Lean 4 auditor.
- `test_s79_r5.py` — adversarial R5 & live-kernel unit test suite (`14` test methods).
- `s79_2dsd_multipole_certificate.json` — pinned canonical JSON certificate (`certificate_sha256 = 410bb15dfe6e77d6729426a13202bd701328275174072227adbb312f7eb41ba7`).
- `REPORT.md` — full technical report with primary-source anchors and zero-over-attribution Lean 4 theorem table.

## Quick Verification

Run from inside `studies/s79-2dsd-multipole-dense-sparse-laws/`:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -I -B verify_s79_2dsd_multipole.py --check s79_2dsd_multipole_certificate.json --json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -I -B verify_s79_2dsd_multipole.py --verify-lean . --json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -I -B test_s79_r5.py -v
```
