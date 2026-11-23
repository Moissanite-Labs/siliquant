# Study (`AUDIT-121AR`): Exact-$\mathbb{Q}$ Conformal / Log-Radial Coxeter-Shell Packing vs. Flat Euclidean Lattice Balls in $\mathbb{R}^8$

- **Scope Boundary (Honest Disclosure):** Strictly additive, self-contained standalone research study under `studies/expedition-conformal-coxeter-audit121ar/`. No files under `shape/`, `harness/`, `lattice/`, `experiments/`, `checks/`, or `.github/` are modified; this package is a standalone exact-arithmetic analysis and certificate verifier, not wired into the core `shape/` or `harness/` runtime.
- **Oracle Class:** `exact` (pure Python 3 standard library over exact $\mathbb{Z}/\mathbb{Q}$; zero floating-point literals or operations on trust/certificate paths; zero bare `assert` statements; zero third-party dependencies).
- **Novelty Class:** `N1` — exact-$\mathbb{Q}$ theta-series complete-shell slack characterization for $E_8$ / $E_8^{\mathrm{Had}}$ and $\mathbb{Z}^8$ balls, 5-family scale-invariant $S^7$ Coxeter angular comparison, exact bulk bit-tax obstruction and scale-inflation crossover threshold for pure uniform-shell conformal packing, zero-extra-bit graded conformal-wing packing inside unused per-block slots, and an honest Law-2b proof of the bulk extended-Hamming parity obstruction (`E5`).
- **Attribution & Cross-PR Context:**
  - $E_8^{\mathrm{Had}} = U E_8$ ($U = H_8 / \sqrt{8}$) construction credited to `Moissanite-Labs/siliquant` PR `#24` (`AUDIT-028`).
  - Builds directly on the exact-$\mathbb{Q}$ flat-Euclidean failure walls (`3.0 bpw` inner cavity / Steiner-quadruple jumps and `4.5 bpw` 1-sparse coordinate outliers) and `Block-H2 == WHT_8` identity established in `Moissanite-Labs/siliquant` PR `#42` (`AUDIT-119BR`).

---

## Manifest

| File | Lines | Bytes | SHA-256 | Role |
|---|---:|---:|---|---|
| [`CONFORMAL_COXETER_REPORT.md`](./CONFORMAL_COXETER_REPORT.md) | `74` | `9,136` | `4238ef9cc19de8d5c57c816ce52d3632e4de323e4aeecde2707824f09d0f1e15` | Full mathematical derivations, theta-series slack tables, $S^7$ Coxeter angular comparison, 5-ensemble distortion law, and Law-4 scope boundaries |
| [`conformal_coxeter_oracle.py`](./conformal_coxeter_oracle.py) | `862` | `37,222` | `323723760c56d932ec9294d637cac7affa635ac9738448665c19bdb966696a25` | Pure-$\mathbb{Z}/\mathbb{Q}$ exact theta-series, $S^7$ Coxeter angular comparison, log-radial ladder, graded conformal-wing packing, and Law-2b bulk parity obstruction oracle |
| [`conformal_coxeter_audit.py`](./conformal_coxeter_audit.py) | `149` | `6,285` | `190d212b951a259e1fe14b149cdb99133719c34948724fa54ed2f687851647f5` | Standalone read-only verifier (`--check`), strict canonical JSON parser, and Zone-B I/O boundary checks |
| [`conformal_coxeter_certificate.json`](./conformal_coxeter_certificate.json) | `1` | `5,602` | `dee8293bd7415d535a3d2fc742ba076c50e360b24ac2d42f50c7a3c3f542dd46` | Sealed canonical JSON certificate (`payload_sha256: 7fa826b2...`, `certificate_sha256: 84b54287...`) |
| [`test_conformal_coxeter_r5.py`](./test_conformal_coxeter_r5.py) | `525` | `28,349` | `cb989dfa4e416a468696d6410a669679305d72b1a5ce9685eb71e196ac1749f5` | Adversarial R5 & Zone-B regression suite (`4` test classes covering mathematical identities, certificate tampers, and strict public-API/CLI guards) |

---

## Quick Reproduction (Zero External Dependencies)

Run directly from `studies/expedition-conformal-coxeter-audit121ar/`:

```bash
# 1. Four-mode isolated parity check on certificate generation (all 4 must match dee8293b...)
python3 -I -B conformal_coxeter_oracle.py --standalone --json | sha256sum
python3 -I -O -B conformal_coxeter_oracle.py --standalone --json | sha256sum
python3 -I -B -W error conformal_coxeter_oracle.py --standalone --json | sha256sum
PYTHONHASHSEED=42 python3 -I -B conformal_coxeter_oracle.py --standalone --json | sha256sum
# Expected SHA-256 (all 4 modes): dee8293bd7415d535a3d2fc742ba076c50e360b24ac2d42f50c7a3c3f542dd46

# 2. Standalone read-only certificate audit across all 3 isolated interpreter modes
python3 -I -B conformal_coxeter_audit.py --check conformal_coxeter_certificate.json
python3 -I -O -B conformal_coxeter_audit.py --check conformal_coxeter_certificate.json
python3 -I -B -W error conformal_coxeter_audit.py --check conformal_coxeter_certificate.json

# 3. Full adversarial R5 + Zone-B unit test suite across all 3 isolated interpreter modes
python3 -I -B test_conformal_coxeter_r5.py
python3 -I -O -B test_conformal_coxeter_r5.py
python3 -I -B -W error test_conformal_coxeter_r5.py
```

---

## Summary of Certified Results

1. **Obstruction #1 (Pure Uniform-Shell Conformal Bulk Bit Tax):** A homogeneous log-radial quantizer (`24` equal-size shells of $E_8^{\mathrm{Had}}$ Shell $m=3$ at $\rho = 5/4$) suffers `1227543017/335544320` (`3.658x`, `+5.63 dB`) worse bulk MSE than flat $E_8^{\mathrm{Had}}$ on typical interior vectors (`E2`), and under shared-scale 16-block (`128`-weight) inflation at `3.0 bpw` ($R^2 = 36$) beats flat $E_8^{\mathrm{Had}}$ only once a heavy-tail outlier exceeds $\|x_{\mathrm{out}}\|_2^2 > 11047887153/83886080 \approx 131.70$.
2. **Exact Dominance Theorem (Graded Conformal-Wing Packing at `0` Extra Per-Block Bits):** Exact integer theta-series enumeration proves complete-shell $E_8$/$E_8^{\mathrm{Had}}$ balls leave `729,007` unused codeword slots (`8.69%`) at `3.0 bpw` ($m \le 18, R^2 = 36$) and `180,292,927` unused slots (`0.52%`) at `4.5 bpw` ($m \le 151, R^2 = 302$). Packing `59,520` graded conformal wing points (`5` inner cavity shells of Shells $m=1,2,3$ = `11,520` pts + `40` outer log-radial shells at $\rho = 9/8$ of Shells $m=1,2$ = `48,000` pts, radial relative MSE `1/868 = -29.39 dB`) fits strictly inside the unused per-block slots (`0` extra per-block bits), cutting inner-cavity error `E1` from `15/8` to `0`, reducing inner Steiner-triple bleed `E4` from `9801/20000` to `1/20000` (`9,801x`), beating flat heavy-tail clip lower bounds `E3` (`24.2x–242.8x`), and maintaining `0` bulk loss on `E2` (`15/8 == 15/8`).
3. **Obstruction #2 (Honest Law-2b Bulk Extended-Hamming Parity Wall `E5`):** On interior asymmetric bulk triple `y = (3, 1, 1, 1/100, 0^4)`, both flat $E_8^{\mathrm{Had}}$ and graded conformal-wing $E_8^{\mathrm{Had}}$ incur the same `9801/20000` (`0.49005`) 4th-coordinate parity jump. Even optimal continuous radial scaling of Shells $m = 1, 2, 3$ onto `y` cannot reduce normalized squared error below `20001/20000`, `30001/20000`, and `35937/80000` (`0.4492125` at $\alpha^* = 367/400$, `35937/4 = 8,984.25x` larger than coordinate-separable $\mathbb{Z}^8$'s total squared error `1/20000` with `0` forced-jump bleed). Conformal radial wings resolve radial-scale walls, not interior coordinate-coupling parity.

---

## Scope Limits (Law 4)

- All theorems are exact algebraic and geometric statements in $\mathbb{R}^8$ proved over $\mathbb{Z}/\mathbb{Q}$ for fixed 8-block and coupled-128D ball budgets.
- End-to-end neural-network KL on real model weights requires integrating graded conformal-wing lookup/indexing into a live quantization harness and is not claimed in this standalone study PR.
