# Heavy-Tail Greedy-LPT Starvation, Load-Power Compression Phase Theory & Low-Bitrate Pareto Dominance (Lean 4.36.0 Formal Companion)

Pure-`Init` machine-checked `Lean 4.36.0` formal theory companion (`21` theorems across `3` modules; `0` Mathlib, `0` external imports, `0` compiler warnings, `0` banned axioms) paired with a pure-$\mathbb{Z}/\mathbb{Q}$ deterministic oracle, R9 static & live Lean 4 auditor, adversarial R5 test suite, and cryptographically pinned certificate (`s73b_certificate.json`).

- **Project ID:** `CEER-HEAVY-TAIL-LPT-STARVATION-LEAN4`
- **Schema:** `aletheia.ceer_heavy_tail_lpt_starvation_lean4.certificate.v1`
- **Toolchain:** `leanprover/lean4:v4.36.0`
- **Novelty Tier / Oracle Class:** `N1` / `exact` (`measured_live_gpu_in_this_bundle = false`)
- **Full Technical Report:** [`S73B_HEAVY_TAIL_LPT_STARVATION_REPORT.md`](S73B_HEAVY_TAIL_LPT_STARVATION_REPORT.md)

## Bundle Contents (`10` Verified Artifacts + This README)

- `lean-toolchain`, `lakefile.toml` — pure-`Init` `Lean 4.36.0` build configuration (`S73bFormalTheory`).
- `S73bFormalTheory/HeavyTailLPTStarvation.lean` (`7` theorems) — arbitrary-length outlier starvation, cardinality-capped 6-quad starvation, $L_2$-energy product/variance inversion vs `quad_snake`, 3-ball secondary overpack, secondary-load inversion vs global sort, plus exact 6-quad and `Qwen2.5-3B` witnesses.
- `S73bFormalTheory/LoadPowerCompressionPhase.lean` (`7` theorems) — concave $p = 1/2$ outlier-ratio compression, 2-quad and arbitrary-$k$ Pythagorean non-starvation basin expansion, universal three-regime phase trichotomy (Regimes I/II/III), and exact unit-scale + cross-model phase-flip witness.
- `S73bFormalTheory/LowBitrateAmplificationAndPareto.lean` (`7` theorems) — high-rate exponential gap amplification, 3-way order/gap-ratio preservation, universal Pareto-corner domination, `BlockH4` vs `BlockH2` spread decomposition, calibrated vs uncalibrated ball-CV dichotomy, and exact 4/4-model Pareto-corner & 3/3-Qwen `BlockH2`-shatter witnesses.
- `s73b_lean4_oracle.py` — pure Python 3 stdlib $\mathbb{Z}/\mathbb{Q}$ oracle (`0` floats, `0` bare `assert`s, `0` mathematical RNG).
- `s73b_lean4_audit.py` — standalone CLI verifier, field-complete secondary auditor, and R9 static/live Lean 4 auditor.
- `test_s73b_lean4_r5.py` — adversarial R5 & live-kernel unit test suite (`14` test methods).
- `s73b_certificate.json` — pinned canonical JSON certificate (`certificate_sha256 = cb0aac952ee0f5c187c6c36a3fcd822256908be0911a97716c7abb876b6f0e5b`).

## Quick Verification

Run from inside `studies/s73b-heavy-tail-lpt-starvation-lean4/`:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -I -B s73b_lean4_audit.py --check s73b_certificate.json --json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -I -B s73b_lean4_audit.py --verify-lean . --json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -I -B test_s73b_lean4_r5.py -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -I -B -c "import pathlib, shutil, subprocess, tempfile; d = tempfile.TemporaryDirectory(); p = pathlib.Path(d.name); [shutil.copy2(f, p / f) for f in ('lean-toolchain', 'lakefile.toml')]; shutil.copytree('S73bFormalTheory', p / 'S73bFormalTheory'); subprocess.run(['lake', 'build'], cwd=p, check=True); d.cleanup()"
```
