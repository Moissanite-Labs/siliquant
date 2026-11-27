# Backprop-Free Evolution Strategies on SiliQuant (A_16 Block-H4) and HoloSphere-V4 (Cayley-SU(2))

Formal Lean 4.36.0 operator kernel, exact pure-Z/Q oracle and zero-shared-code verifier, adversarial R5 test suite, and live RTX 3080 real-weight evaluation transferring backprop-free evolution strategies (`arXiv:2509.24372v3`) onto `SiliQuant` (`A_16 Block-H4`, `[16,11,4]` Extended Hamming SECDED, `+0 B` folded scales, `int8` `Q=64` Al-Kashi FixedRadix error feedback) and `HoloSphere-V4` (`15.11M`, rational Cayley-$SU(2)$ holonomy).

Full technical ledger, SHA-256 manifest, and detailed tables: [`REPORT.md`](REPORT.md).

## Scope and Honest Kill-Gate Summary

- **Preregistration:** [`PREREG_W22_ES_SCALE.json`](PREREG_W22_ES_SCALE.json) (`SHA-256: 1f9db607ce34de1e0cbd472625d411d8c5e9857468a107a0cc24e558a733d3f6`).
- **Energy Disclosure:** `UNCONTROLLED_LAPTOP_NOT_CLAIMED` (zero laptop energy, joule, or watt claims are made or implied; VRAM and wall-clock telemetry only).
- **Overall Certificate Verdict Tag:** `Q51a_FALSIFIED_Q51b_FALSIFIED_Q51c_CONFIRMED`.
- **`EXP-Q51c` (`EvolutionOperators.lean` Formal Kernel):** **CONFIRMED**
  - `21/21` pure-`Init` Lean `4.36.0` theorems live-verified (`0` zero-axiom `[]`, `1` `[propext]`, `20` `[propext, Quot.sound]`; `0` `sorry`, `0` `Classical.choice`, `0` user axioms, `0` compiler warnings, `0` `.lake/` residue).
  - `3x -I` bit-identical (`8515676847d862dd2594fab607a1fac380f87d66c1e06981240891d986cc0ac0`), `366/366` dual `mutate --max-mutants 0` (`183/183` oracle + `183/183` verifier, `0` survivors), `29/29` AST `BoolOp` (`0` survivors), and `audit-check` `PASS`.
- **`EXP-Q51a` (`SiliQuant-ES` on `Qwen2.5-0.5B` `n=7` seeds + full `28/28`-layer `Qwen2.5-1.5B`):** **FALSIFIED**
  - **Systems & Memory Gates PASS:** `0` committed SECDED syndromes and `0` committed OOB groups across `22,364,160` `0.5B` blocks (`3.0` and `4.5 bpw`, `7/7` seeds) and all `81,887,232` `1.5B` blocks (`28/28` layers, `196/196` block linears, `94,633,548` live flips). Full `28/28`-layer `Qwen2.5-1.5B` packed ES executes live antithetic rollouts and full `196`-layer Al-Kashi + `fp16` scale updates at `5069.9 MiB` (`4.95 GiB`) peak allocated and `6060.0 MiB` (`5.92 GiB`) peak reserved (`< 6656 MiB` = `6.5 GiB`, `+0 B` fold invariance error `5.72e-6`, `10,235,904` `fp16` scale updates).
  - **Accuracy Gate FALSIFIED vs Unconfounded `INT4` (`C2`):** While `T1` (`7.37%`, `33/448`) improves `+2.01 pp` over canonical BF16 `C1` (`5.36%`, which suffers `25.56%` ULP stall and `5.86e-3` restoration drift), once matched cleanly across all `168` block linears, `8` antithetic pairs, `int8` `Q=64` EF, and true 4-bit symmetric `qmax=7`, unblocked `INT4` (`C2`) reaches `8.26%` (`37/448`, delta `-0.89 pp`, `p_raw = 0.75`, `p_Holm = 0.75`).
- **`EXP-Q51b` (`HoloSphere-Cayley-ES` on `HoloSphere-V4` `15M`):** **FALSIFIED**
  - **Manifold & SECDED Gates PASS:** `max_su2_shell_err = 4.17e-7` (`< 1e-5` gate) and `0 / 936,960` `A_16` syndromes across `19,859,742` live `w_omega` packed `uint64` flips (`2,045,684` OOB nibble blocks reverted and logged), compared against tangent-projected unclamped `ArmA` Euclidean additive ES (`8.61e12` to `6.76e18` plus `Inf/NaN` and `3/3` WikiText-2 `NaN` collapse, `0.0000` OOD reward).
  - **VRAM & Downstream Gates FALSIFIED:** Peak VRAM is flat at `80.85 MiB` (`< 512 MiB`, `0.17 MiB` warm spread across cells `2..9` vs `Full-Graph Chunk32-BPTT` `3/9` OOMs at `7.47 GiB` and `Checkpointed Chunk32-BPTT` peaking at `7575.37 MiB`), **but** cold-vs-warm total spread `2.20 MiB` (`78.66 -> 80.85 MiB`, `2248 KiB`) exceeds the strict unified `< 1.0 MiB` (`1024 KiB`) spread cap. Downstream `8x` OOD `S_3`/`A_5` exact accuracy gain is `0.00 pp` (`1.56% -> 1.56%`, failing `>= +10 pp`), and `16 KiB` held-out WikiText-2 BPB drift is `+0.0236` (`3.2264 -> 3.2500`, exceeding `<= 0.02`).
- **Scope & Follow-Up Disclosure (`LOW-7`):** In [`s167_es_live_bench.py`](s167_es_live_bench.py) (`:1178`), `ef_res` retains clamped `u_fixed` when `oob_m` reverts `realized = 0` (explaining `4,413,737` `1.5B` and `63,634,691` `0.5B` OOB candidate reverts across `7` seeds), and `3.0 bpw` (`C0_A16_BlockH4_3bpw_Null`, `4.69%`) is evaluated as a null baseline while 7-seed `T1` ES runs `4.5 bpw`. This `LOW-7` `ef_res` OOB windup behavior is now addressed in follow-up `Track S173` (`FT_PIWL` tangential anti-windup).

## Bundle Contents

- `EvolutionOperators.lean`, `lean-toolchain`, `lakefile.toml`, `lake-manifest.json`: pure-`Init` `Lean 4.36.0` formalization (`21` public theorems, zero external dependencies).
- `PREREG_W22_ES_SCALE.json`: SHA-pinned preregistration specification.
- `s167_es_live_bench.py`: live RTX 3080 evaluation script (`Qwen2.5-0.5B`, full `28/28`-layer `Qwen2.5-1.5B`, and `HoloSphere-V4` `15M`) with `--cpu-selftest`.
- `s167_empirical_results.json`, `s167_empirical_receipt.json`: recorded empirical results and integer micro-unit receipt (`WEIGHTS=0`, `BINARY_PDF=0`).
- `s167_es_oracle.py`: exact pure-Z/Q oracle and certificate generator.
- `verify_s167_es.py`: zero-shared-code standalone verifier, static Lean R12/R13b scanner, Zone-B I/O checker, and `--live-lean` runner.
- `test_s167_es_r5.py`: adversarial unit, AST, Zone-B I/O, programmatic verdict, and Lean verifier test suite (`7/7` tests, `0` skips).
- `s167_es_certificate.json`: canonical certificate (`payload_sha256: 9474c1b5ee619c5906b8a807f5361b31089cd4ef1caf3787eb6bbf8bd9cc645a`).
- `REPORT.md`: full experimental and formal report with SHA-256 manifest.

## Quick Verification

Run from inside `studies/es-siliquant-holosphere/`:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -I -B s167_es_oracle.py --json
PYTHONDONTWRITEBYTECODE=1 python3 -I -B verify_s167_es.py --check --json
PYTHONDONTWRITEBYTECODE=1 python3 -I -B verify_s167_es.py --live-lean --check --json
PYTHONDONTWRITEBYTECODE=1 python3 -I -B test_s167_es_r5.py -v
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" python3 -I -B s167_es_live_bench.py --cpu-selftest
lean EvolutionOperators.lean
```
