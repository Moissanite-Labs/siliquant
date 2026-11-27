# Ternary A_16 Extended-Hamming Sign-Flip Switch vs. Clip-at-Zero Lockout (Lean 4.36.0 & Exact Pure-Z/Q Verification)

Exact algebraic and formal verification of a reversible `{-1, 0, +1}` switch over the `[16, 11, 4]` extended-Hamming `A_16` Block-H4 representation, contrasted with destructive `0/1` row zeroing and clip-at-zero gate absorption (`Model Carving`, `arXiv:2611.27962`).

## Scope & Honest Limits

- **Scope:** Exact algebraic and formal verification only (pure `int` / `Fraction` Python 3 stdlib and pure-`Init` `Lean 4.36.0`; zero floats, zero RNG, zero external packages).
- **Not tested or claimed here:** No model training, no empirical retention benchmarking, and no latency, throughput, power, or energy measurements of any kind are performed or claimed in this bundle.
- **Carried specification/paper witnesses:** Retention fractions (`9921 -> 8333` [`397/2500`], `9396 -> 8322` [`537/5000`], `9558 -> 8729` [`829/10000`], and `6/46 = 3/23`) and pruning counts (`193`, `671`, `141/184`, `33/35`) are carried and checked as exact rational arithmetic witnesses from the paper/specification, not re-derived from model training runs.
- **Model boundaries:** The carving transition model is a cleared abstraction of `Gates.step` (`g' = max(g - u - p, 0)` with `u = 0` once zeroed/dropped from `kept` and `p >= 0`; Adam moment state and slope constants are not modeled). Rollback ordering checks `length -> D^2` in Lean and `length -> SHA-256 -> D^2` in Python. The `-1` switch flips the block scale `s -> -s` on unchanged nibble bytes (`c XOR 0xFFFF` holds in the bipolar sign-code view, not as an in-place nibble rewrite).

## Key Verified Results

1. **`[16, 11, 4]` `A_16` Complement (`verify_s163_ternary_a16_carving.py`, `lean/TernaryA16Carving/Basic.lean`):**
   Exhaustive check across all `65,536` 16-bit vectors with masks `(0xFFFF, 0xAAAA, 0xCCCC, 0xF0F0, 0xFF00)` (weights `16, 8, 8, 8, 8`): `complement(complement(c)) = c`, `syndrome(c XOR 0xFFFF) = syndrome(c)`, bipolar `phi(c XOR 0xFFFF) = -phi(c)` with `||phi||^2 = 16`, and all `2,048` zero-syndrome codewords remain closed under complementation (`syndrome(0x0000) = syndrome(0xFFFF) = 0`).
2. **Block-H4 Parseval, Scale Sign-Flip & Ternary States:**
   Over `2,048` codeword-derived nibble blocks (`s = 3/8`): `sum (H_4 c)^2 = 2,818,048 = 4 * 704,512 = 4 * sum c^2`, `sum w^2 = 99072 = s^2 * sum c^2`, exact negation on `s -> -s` with `0/1` norm drift and `0` nibble bytes changed, and `8` multi-round sequences (lengths `1..64`) with `4/4` even-flip identity returns and `4/4` odd-flip negations. Across `32` blocks x `3` states: `96/96` zero syndromes, `64/64` exact restores to `+1`, and `3/3` invalid requests bounced check-before-assign.
3. **`16 B` Spinor Checkpoint & Check-Before-Assign Rollback:**
   Bijective `<4i` encoding (`pack16B_roundtrip`, `pack16B_unpack16B` in Lean; `4` spinor + `256` byte probes in Python) on `(805306368, -268435456, 536870912, -671088640)` with `D = 1207959552` (`D^2 = 1459166279268040704`). A `19`-op trace bounces `2` invalid switch requests, `2` tampered base assignments, and `3` tampered checkpoints (flipped byte, truncated, non-unit) while keeping every block at syndrome `0` and restoring exact switches `[1] * 8` with `0/1` spinor and weight norm drift.
4. **Collapse & `g = 0` Lockout Duality:**
   Row-zeroing (`up_row_zero`) collapses `64/64` nonzero witness blocks strictly to norm `0` and a single image. Under `Gates.step` with `u = 0` on dropped gates and `p >= 0`, `0/193` locked gates reopen over a `48`-step horizon despite reopening gradients (`193/193` lockout vs. `193/193` ternary recovery; with `hold_open = 1/1000000`, `193/193` reopen).
5. **Lean 4.36.0 Formal Kernel (`36` Public Theorems):**
   `lean/TernaryA16Carving/Basic.lean`, `lean/TernaryA16Carving.lean`, and `lean/AxiomAudit.lean` (`562` LoC total, pure `Init`, `"packages": []`): `#print axioms` certifies `5` axiom-free theorems, `17` `[propext]`, and `14` `[propext, Quot.sound]`, with `0` `Classical.choice`, `0` `sorry`, and `0` compiler warnings.

## Bundle Contents

- `lean/lean-toolchain`, `lean/lakefile.toml`, `lean/lake-manifest.json`: pinned `leanprover/lean4:v4.36.0` pure-`Init` package configuration (`TernaryA16Carving`).
- `lean/TernaryA16Carving.lean`, `lean/TernaryA16Carving/Basic.lean`, `lean/AxiomAudit.lean`: `36` formal theorems and 1:1 `#print axioms` audit target.
- `verify_s163_ternary_a16_carving.py`: standalone exact verifier, Zone-B strict JSON/file I/O handler, static Lean scanner, and staged `--live-lean` runner.
- `test_s163_ternary_a16_r5.py`: `29`-test adversarial unit, AST, Zone-B I/O, type-strict JSON equality, and staged Lean runner suite (`0` skips).
- `s163_ternary_a16_carving_certificate.json`: pinned canonical JSON certificate (`payload_sha256 = b45432ea7bb4993f1848b170d97878983d532f66eceb79032d408763b09538b4`).
- `REPORT_S163_TERNARY_A16_CARVING.md`: full technical verification report and SHA-256 manifest.

## Quick Verification

Run from inside `studies/ternary-a16-carving-lockout/`:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -I -B verify_s163_ternary_a16_carving.py --check s163_ternary_a16_carving_certificate.json --json
PYTHONDONTWRITEBYTECODE=1 python3 -I -B verify_s163_ternary_a16_carving.py --live-lean --check s163_ternary_a16_carving_certificate.json --json
PYTHONDONTWRITEBYTECODE=1 python3 -I -B test_s163_ternary_a16_r5.py -v
```
