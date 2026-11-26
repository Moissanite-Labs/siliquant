# Exact Operator & Roofline Laws for 2DSD, Dense-and-Sparse CSR vs. `siliquant`, and Fast-Multipole Attention (`Lean 4.36.0` Formal Companion)

Pure-`Init` machine-checked `Lean 4.36.0` formalization (18 kernel-checked declarations across `TwoDSDAndDenseSparseLaws.lean` = 15 universal theorems + 3 `decide` literal-ordering checks; `0` Mathlib, `0` external imports, `0` compiler warnings, `14/18` declarations zero-axiom `[]` + `4/18` `[propext]` only, `0` banned axioms) paired with a pure-$\mathbb{Z}/\mathbb{Q}$ deterministic oracle, R9+/R10 static & live Lean 4 verifier, adversarial R5 test suite, and cryptographically pinned certificate (`s79_2dsd_multipole_certificate.json`).

- **Project ID:** `WAVE4B-S79-2DSD-MULTIPOLE-DENSE-SPARSE`
- **Toolchain:** `leanprover/lean4:v4.36.0`
- **Novelty Tier / Oracle Class:** `N1` / `exact` (`measured_live_gpu_in_this_bundle = false`)
- **Related PRs:** Refs #77, #95, #99, #100
- **Full Technical Report:** [`REPORT.md`](REPORT.md)

## Bundle Contents (`8` Frozen Verified Artifacts + This README)

- `lean-toolchain`, `lakefile.toml` — pure-`Init` `Lean 4.36.0` build configuration (`TwoDSDAndDenseSparseLaws`).
- `TwoDSDAndDenseSparseLaws.lean` (15 universal theorems + 3 literal-ordering checks) — universal proofs covering (1) `2DSD` lossless multi-candidate cascade verification, 2D speedup dominance over 1D DFlash/LoopSpec, Eq. 6 last-loop KV sharing compression & early-loop drift obstruction; (2) exact matched-bitrate Dense-and-Sparse CSR vs. dense `siliquant` `BlockH4 QuadSnake` crossover cone, high-CSR phase-reversal barrier, simultaneous triple dominance, and exact 5th-power safe sparsity cap (`rho_s <= 1/160 = 0.625%`); and (3) `MultipoleAttention` arbitrary-dimension Lagrange Cauchy–Schwarz logit bound, first-order centroid cancellation, second-order Jensen partition underestimation, exact softmax ratio bracket, $d_k=128$ opening-angle barrier, and exact value-weight covariance residual. The 3 remaining `decide` declarations (theorems 16–18) only order quoted external 2DSD / SqueezeLLM / S72 / MultipoleAttention numerals as typed literals — the Lean kernel never sees those sources, so they are literal-ordering checks, not pinned evidence.
- `s79_2dsd_multipole_oracle.py` — pure Python 3 stdlib $\mathbb{Z}/\mathbb{Q}$ oracle (`0` floats, `0` bare `assert`s, `0` mathematical RNG).
- `verify_s79_2dsd_multipole.py` — standalone CLI verifier, secondary auditor, and R9+/R10 static & live Lean 4 auditor.
- `test_s79_r5.py` — adversarial R5 & live-kernel unit test suite (`15` test methods, including a planted-oracle-edit control that must fail closed).
- `s79_2dsd_multipole_certificate.json` — pinned canonical JSON certificate (`certificate_sha256 = e0753699fdfba9f12321e97d82a41503ffa2688da414acf946a9ce805a71b3a7`). The certificate also pins `oracle_module_sha256` (SHA-256 of `s79_2dsd_multipole_oracle.py`); `--check` verifies the on-disk oracle against that pin **before** any full replay and fails closed with `ORACLE_SHA256_MISMATCH` on any oracle edit, so the replay target is fixed instead of self-referential.
- `REPORT.md` — full technical report with primary-source anchors and zero-over-attribution Lean 4 theorem table.

## Anchor & Pinning Status (Offline Verification)

- **Oracle pin (checked):** the certificate pins `oracle_module_sha256` = SHA-256 of `s79_2dsd_multipole_oracle.py`. `verify_s79_2dsd_multipole.py --check` hashes the on-disk oracle module and compares it against that pin **before** the deterministic full replay, failing closed with `ERROR ORACLE_SHA256_MISMATCH` (rc=2) on any oracle edit.
- **External source pins (NOT verified offline):** the `PRIMARY_SOURCE_SHA256` digests in the oracle (2DSD repo `431aade2`, MultipoleAttention `db65e51d`, SqueezeLLM, KVQuant, HF `minseo25/2dsd` snapshots) are pins only — no code in this bundle fetches or checks those external sources, so they are unverified offline. Only `s72_empirical_results_sha256` (`a7c35c646a9c29fa645f207697c74e181ad19d8ce61acbe489a265a132968647`) is verifiable in-repo, against `studies/rtx3080-s72-stratified-quad-blockh4/rtx3080_s72_empirical_results.json`.
- **Qwen2.5-3B anchor labeling:** the S72 `Qwen2.5-3B` nanonats quoted in `REPORT.md` are WikiText-2 mean KL **vs its own BF16** reference (S72 record `reference_bf16`, WikiText-2 BF16 PPL `10.970193`).

## Quick Verification

Run from inside `studies/s79-2dsd-multipole-dense-sparse-laws/`:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -I -B verify_s79_2dsd_multipole.py --check s79_2dsd_multipole_certificate.json --json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -I -B verify_s79_2dsd_multipole.py --verify-lean . --json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -I -B test_s79_r5.py -v
```
