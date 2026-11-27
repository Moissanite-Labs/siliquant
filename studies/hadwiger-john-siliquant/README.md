# Hadwiger-3D Convex-Covering Operators on SiliQuant (E_8 and A_16 Block-H4)

Formal Lean 4.36.0 operator kernel, exact pure-Z/Q oracle and verifier, and 52-arm live RTX 3080 real-weight evaluation transferring convex-covering operators from Hadwiger's three-dimensional covering conjecture (`arXiv:2611.32334v2`) onto `SiliQuant` spherical lattice balls (`d=8 E_8` and `d=16 A_16 Block-H4`).

Full technical ledger and tables: [`REPORT.md`](REPORT.md).

## Scope and Honest Kill-Gate Summary

- **Energy Disclosure:** `UNCONTROLLED_LAPTOP_NOT_CLAIMED` (zero laptop energy, joule, or watt claims are made or implied).
- **Formal Geometry (`JohnQuantOperators.lean`):** **CONFIRMED** (`24/24` pure-`Init` Lean `4.36.0` theorems; `7/24` zero-axiom `[]`, `17/24` `[propext]` only; `0` `Classical.choice`, `0` `Quot.sound`, `0` `sorry`, `0` compiler warnings, `0` `.lake/` residue).
- **Kill Gate 1 (`KG1` = `PARTIAL` -- John/Lowner `S_j` `+0 B` Folded Diagonal Whitening):**
  - Apples-to-apples `B1c vs A3` (`BlockH4 + QuadSnake`, `3.0 bpw`) fails both weighted `hMSE` (`+6.45%` on `Qwen2.5-0.5B`, `+5.79%` on `Qwen2.5-1.5B`; `0/52` transformer layer wins) and held-out WikiText-2 PPL (`-0.2185` on `0.5B`, `-0.1364` on `1.5B`).
  - Across all `8` `A_16` prongs (`B1b vs A2` and `B1c vs A3` over `hMSE` and WikiPPL on both models), `1/8` passes (`B1b vs A2` passes `>= 0.01` WikiPPL on `1.5B` at `+0.0504` while failing `0.5B` at `-0.2196` and failing `hMSE` / Python PPL on both). `E_8` `B1d vs A1` fails `hMSE` on both models while beating WikiPPL (`+0.3823` / `+0.1569`).
  - Root geometric obstruction: Hadwiger-3D's halved-difference-body $L_\infty$ operator $S_j = \frac{1}{2}(\max_i W_{ij} - \min_i W_{ij})$ cuts unweighted `p999` error (`-3.28%` on `0.5B`, `-20.47%` on `1.5B` in `B1b vs A2`), whereas Theorem 17 proves the Hessian-weighted $L_2$ ball objective requires $L_2$-ellipsoid whitening $c_j^* \propto h_j^{1/4} / \mathrm{RMS}(W_{:,j})^{1/2}$.
- **Kill Gate 2 (`KG2` = `FALSIFIED` -- Helly-LP Minimax Chebyshev Scales and Centers):**
  - `0/6` `B2` arms reach the required `>= 15%` reduction in `99.9th`-percentile (`p999`) outlier error (`B2a` cuts `p999` by `0.10%` / `1.06%` at `3.0 bpw` and `1.32%` / `2.35%` at `4.5 bpw`; `B2b` increases `p999` by `+103.7%` / `+138.0%`).
  - `4` WikiText-2 PPL degradation hits occur without reaching `15%` `p999` reduction (`B2a` on `0.5B 3.0b` `-0.7240` and `1.5B 4.5b` `-0.0642`; `B2b` on both scales `-5.4784` / `-2.1231`), triggering `KG2 = FALSIFIED`.
- **Arm B3 (`B3` = `PARTIAL` -- Polar-Cap vs Equatorial-Bulk Matched-`bpw` Split):**
  - Payload+scale rates stay within `<= 3.000 bpw` (`B3b` `2.998 bpw` via `1/19` `4.5 bpw` + `18/19` `B=357`; `B3a` `2.994 bpw`), **but uncompressed indexing metadata exceeds `3.000 bpw`** (`B3b` with `1`-bit/group selector bitmap = `3.0053 bpw` [`3.005338` / `3.005326`]; `B3a` with `16`-bit CSR `row_ptr` = `3.0077` / `3.0019 bpw`).
  - `B3b` cuts `wp999` (`-8.70%` / `-7.98%`) and achieves the lowest `3.0 bpw` `A_16` Python PPL on the recorded `4,096`-token `fractions.py` slice, **but worsens WikiText-2 PPL (`21.9780 vs 21.2516`; `14.4654 vs 14.4299`) and total `hMSE` (`+0.70%` / `+2.17%`) vs `A2`**. `B3a` beats `A3` WikiText-2 PPL (`+0.1914` / `+0.0939`) at worse total `hMSE` (`+5.41%` / `+6.54%`).
- **Recorded Benchmark Scope Disclosures:** In the recorded `s165_empirical_results.json` run, `A1..A5` declared `extra_runtime_bytes = 0` on unpooled `o_proj` scales, `B3a`/`B3c` used additive CSR residual reconstruction, `B3b` logged `ball_bound = 651` before per-group override, and the `4,096`-token Python evaluation window scored only the leading `4,096` tokens of `fractions.py` at a single seed (each patched in `s165_john_siliquant_bench.py` for future runs and explicitly reconciled in the exact verifier).

## Key Verified Results

1. **Pure-`Init` Lean 4.36.0 Formal Kernel (`JohnQuantOperators.lean`, `24` Theorems):**
   Proves mean-width / volume / diameter gaps across $d=3,4,8,16$, rational John decompositions of $O_4$ and $O_{16}$, the $d=4$ zonotope surface-area and 14-cap star-Helly obstructions (`12 + 8*sqrt(3) > 24` via `48 > 36` and `7/8 < 1`), `+0 B` RMSNorm/QKV/GQA folding invariances, Parseval orthogonality, Lagrange/Cauchy-Schwarz global $L_2$-ellipsoid optimality and uniqueness, Helly-Chebyshev center uniqueness and bulk scale-inflation tradeoff, polar-cap contraction, and `A_16`/`E_8` matched-`bpw` conservation (`0` `sorry`, `0` external imports, `"packages": []`).
2. **Exact Z/Q Oracle, Verifier, and Non-Circular Theta Bounds (`52` Arms):**
   `s165_john_siliquant_oracle.py` and `verify_s165_john_siliquant.py` share zero code and independently verify all 4 pillars, dynamic `A_16` (`B=368->651`, `357->577`, `352->547`; `B=560->5213`, `549->4629`, `544->4385`) and `E_8` (`368->420`, `560->3380`) theta bounds, and exact `Decimal` reconciliation across all `52` live arms (`0` syndrome errors, `0` OOB groups) into bit-identical JSON output (`78a8d4fa7d91d4aab2e894d06233a50c4fc57ea0690a4ce532aabd78be9ed551`).

## Bundle Contents

- `JohnQuantOperators.lean`, `lean-toolchain`, `lakefile.toml`, `lake-manifest.json`: pure-`Init` `Lean 4.36.0` formalization (`24` public theorems, zero external dependencies).
- `s165_john_siliquant_bench.py`: live RTX 3080 evaluation script (`Qwen/Qwen2.5-0.5B` and `Qwen/Qwen2.5-1.5B`).
- `s165_empirical_results.json`, `s165_empirical_receipt.json`: recorded `52`-arm empirical results and integer micro-unit receipt.
- `s165_john_siliquant_oracle.py`: exact pure-Z/Q oracle and certificate generator.
- `verify_s165_john_siliquant.py`: zero-shared-code standalone verifier, static Lean scanner, Zone-B I/O checker, and `--live-lean` runner.
- `test_s165_john_siliquant_r5.py`: adversarial unit, AST, Zone-B I/O, and staged Lean verifier test suite (`7/7` tests, `0` skips).
- `s165_john_siliquant_certificate.json`: canonical certificate (`payload_sha256`: `267418a9070db8c3b1ef9ea789eaf4086b4d70d86067f319bd28aa8e082dff1a`).
- `REPORT.md`: full experimental and formal report with SHA-256 manifest.

## Quick Verification

Run from inside `studies/hadwiger-john-siliquant/`:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -I -B s165_john_siliquant_oracle.py --json
PYTHONDONTWRITEBYTECODE=1 python3 -I -B verify_s165_john_siliquant.py --check --json
PYTHONDONTWRITEBYTECODE=1 python3 -I -B verify_s165_john_siliquant.py --live-lean --check --json
PYTHONDONTWRITEBYTECODE=1 python3 -I -B test_s165_john_siliquant_r5.py -v
```
