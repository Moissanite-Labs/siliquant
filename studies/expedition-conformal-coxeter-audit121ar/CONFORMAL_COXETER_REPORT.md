# Track S21a (`AUDIT-121A`): Exact-$\mathbb{Q}$ Conformal / Log-Radial Coxeter-Shell Packing vs. Flat Euclidean Lattice Balls in $\mathbb{R}^8$

- **Oracle Class:** `exact` (pure-stdlib $\mathbb{Z}/\mathbb{Q}$, zero floats on all trust/certificate paths, zero bare `assert`s, full Zone-B `strict_cert_io` hardening)
- **Novelty Class:** `N1` (exact-$\mathbb{Q}$ theta-series complete-shell slack characterization, 5-family $S^7$ Coxeter angular comparison, exact bulk bit-tax obstruction & scale-inflation crossover threshold for pure uniform-shell conformal packing, zero-extra-bit graded conformal-wing packing inside unused per-block slots, and honest Law-2b proof of the bulk extended-Hamming parity obstruction)

---

## 1. Executive Summary (Exact Dominance Region vs. Honest Law-2b Obstructions)

`Track S20b` (`AUDIT-119B`) established two flat-Euclidean failure walls of uniform $E_8$ / $E_8^{\mathrm{Had}}$ ball packing in $\mathbb{R}^8$: (1) an empty inner shell $(0, \sqrt{2})$ coupled with Steiner-quadruple parity jumps at low bitrate (`3.0 bpw`), and (2) heavy-tail 1-sparse coordinate outliers (`ffn_down` loud columns) at `4.5 bpw`. This study evaluates **conformal / log-radial Coxeter-shell packing** (geometric radii $r_k = r_0 \rho^k$ paired with scale-invariant $S^7$ Coxeter shells from $\mathbb{Z}^8$, $D_8$, $E_8$, $E_8^{\mathrm{Had}}$, and `Block-H2`) against flat Euclidean balls at strictly matched bitrates (`3.0 bpw` $= 2^{23}$ and `4.5 bpw` $= 2^{35}$ points per 8-block), proving **three exact rational results**:

1. **Obstruction #1 (Pure Uniform-Shell Conformal Bit Tax on Bulk Shells):** A homogeneous log-radial quantizer (`24` equal-size shells of $E_8^{\mathrm{Had}}$ Shell $m=3$ at $\rho = 5/4$) starves the 7 angular dimensions on $S^7$, suffering **`1227543017/335544320` (`3.658x`, `+5.63 dB`) worse bulk MSE** than flat $E_8^{\mathrm{Had}}$ on typical interior vectors (`E2`). Under shared-scale 16-block (`128`-weight) inflation at `3.0 bpw` ($R^2 = 36$), pure uniform conformal beats flat $E_8^{\mathrm{Had}}$ **only** once a heavy-tail outlier exceeds the exact rational squared-norm threshold $\|x_{\mathrm{out}}\|_2^2 > \mathbf{11047887153/83886080 \approx 131.70}$ (`1.91x` ball radius).
2. **Exact Dominance Theorem (Graded Conformal-Wing Packing at `0` Extra Per-Block Bits):** Exact integer theta-series enumeration proves that complete-shell $E_8$/$E_8^{\mathrm{Had}}$ balls leave **`729,007` unused codeword slots (`8.69%`)** at `3.0 bpw` ($m \le 18, R^2 = 36$) and **`180,292,927` unused slots (`0.52%`)** at `4.5 bpw` ($m \le 151, R^2 = 302$). Packing **`59,520` graded conformal wing points** (`5` inner cavity shells of $E_8^{\mathrm{Had}}$ Shells $m=1,2,3$ = `11,520` pts + `40` outer log-radial shells at $\rho = 9/8$ of Shells $m=1,2$ = `48,000` pts, spanning $1.061\times$ to $10.55\times R_{\mathrm{ball}}$ with radial relative MSE `1/868 = -29.39 dB`) fits strictly inside the unused per-block slots (**`0` extra per-block bits**; continuous coupled-128D cardinal ratio `59520/7659601` $\approx 0.00140\text{ bpw}$ at `3.0 bpw` and `59520/34179445441` $< 3.2 \times 10^{-7}\text{ bpw}$ at `4.5 bpw`). It simultaneously achieves:
   - **`E1` (Inner Cavity $0 < \|x\|^2 < 2$):** reduces total error from `15/8` down to **`0`**;
   - **`E4` (Inner Steiner Triple $(1,1,1,1/100,0^4)/\sqrt{2}$):** reduces squared error from `9801/20000` to **`1/20000` (`9,801x` reduction, `0` 4th-coord bleed)** via `(1/2) * Shell_3`;
   - **`E2` (Bulk Typical Shell):** **`0` bulk loss** (`15/8 == 15/8`, all bulk shells $m=0..18$ remain intact);
   - **`E3` (Heavy-Tail 1- and 2-Sparse Outliers beyond $R^2=36$):** achieves `-25.9` to `-31.6 dB` relative error up to $\|x\|^2 = 800$ with **zero group-scale inflation** and **zero `+K10` column indexing**.
3. **Obstruction #2 (Honest Law-2b Bulk Extended-Hamming Parity Wall `E5`):** On an interior asymmetric bulk triple `y = (3, 1, 1, 1/100, 0^4)` ($\|x\|^2 = 11/2 + 1/20000$), **both** flat $E_8^{\mathrm{Had}}$ and graded conformal-wing $E_8^{\mathrm{Had}}$ suffer the exact same **`9801/20000`** (`0.49005`) 4th-coordinate parity jump. Even optimal continuous radial scaling of Shells $m = 1, 2, 3$ onto `y` cannot reduce the normalized squared error below `20001/20000`, `30001/20000`, and **`35937/80000`** (`0.4492125`, attained at $\alpha^* = 367/400$ on Shell-3 vector `(3, 1, 1, 1, 0^4)`, vs `1`, `3/2`, `11/24` on the unperturbed base `y_0 = (3, 1, 1, 0^5)`), which remains **`35937/4` (`8,984.25x`)** larger than coordinate-separable $\mathbb{Z}^8$'s total squared error `1/20000` (`0` forced-jump bleed). Conformal radial wings solve **radial-scale walls**, not **interior coordinate-coupling parity**.

---

## 2. Exact Theta-Series Ball Budgets & $E_8^{\mathrm{Had}}$ Shell Spectrum

| Lattice & Rate | Per-8-Block Budget | Max Shell / $R^2$ | Used Points | Unused Complete-Shell Slots |
|---|---:|---:|---:|---:|
| **$E_8$ / $E_8^{\mathrm{Had}}$ (`3.0 bpw`)** | `8,388,608` ($2^{23}$) | `m = 18` ($R^2 = 36$) | `7,659,601` | **`729,007` (`8.69%`)** |
| **$E_8$ / $E_8^{\mathrm{Had}}$ (`4.5 bpw`)** | `34,359,738,368` ($2^{35}$) | `m = 151` ($R^2 = 302$) | `34,179,445,441` | **`180,292,927` (`0.52%`)** |
| **$\mathbb{Z}^8$ (`3.0 bpw`)** | `8,388,608` ($2^{23}$) | $R^2 = 37$ | `8,069,761` | `318,847` (`3.80%`) |
| **$\mathbb{Z}^8$ (`4.5 bpw`)** | `34,359,738,368` ($2^{35}$) | $R^2 = 302$ | `33,949,336,273` | `410,402,095` (`1.19%`) |

Exact $E_8^{\mathrm{Had}}$ (`Block-H2`) first three shells ($\|x\|^2 = 2m$, matching $240\,\sigma_3(m)$):
- **`m = 1`** (`240` pts): `{wt 1: 16, wt 4: 224}` (all 1-sparse axes $\pm\sqrt{2}\,e_i$ + Hamming quads).
- **`m = 2`** (`2,160` pts): `{wt 2: 112, wt 5: 1,792, wt 8: 256}` (all 2-sparse pairs + all 8-sparse sign patterns).
- **`m = 3`** (`6,720` pts): `{wt 3: 448, wt 4: 896, wt 6: 5,376}` (all 3-sparse triples!).

---

## 3. Exact Scale-Invariant $S^7$ Angular Comparison ($\sin^2\theta$ over $\mathbb{Q}$)

| Coxeter Angular Code on $S^7$ | Size | `k=1` axis | `k=2` pair | `k=3` triple | `k=4` Hamming | `k=4` non-Hamming | `k=8` all-ones |
|---|---:|---:|---:|---:|---:|---:|---:|
| `z8_shell1` | `16` | **`0`** | `1/2` | `2/3` | `3/4` | `3/4` | `7/8` |
| `d8_shell1` | `112` | `1/2` | **`0`** | `1/3` | `1/2` | `1/2` | `3/4` |
| `e8_shell1` | `240` | `1/2` | **`0`** | `1/3` | `1/2` | `1/2` | **`0`** |
| `e8_had_wht8_shell1` | `240` | **`0`** | `1/2` | `1/4` | **`0`** | `7/16` | `1/2` |
| **`block_h2_shell1` (`3x` fewer ops)** | `240` | **`0`** | `1/2` | `1/4` | **`0`** | `7/16` | `1/2` |
| `e8_had` `S1 U S2` | `2,400` | **`0`** | **`0`** | `1/4` | **`0`** | `7/32` | **`0`** |
| `e8_had` `S1 U S2 U S3` | `9,120` | **`0`** | **`0`** | **`0`** | **`0`** | `7/32` | **`0`** |

---

## 4. Exact Log-Radial Ladders & 5-Ensemble Distortion Comparison

Exact log-radial ladder metrics over $\mathbb{Q}$ ($\max\text{rel}^2 = ((\rho-1)/(\rho+1))^2$, $\text{rel MSE} = (\rho-1)^2/(4(1+\rho+\rho^2))$):
- $\rho = 2$: max `1/9`, MSE `1/28` (`-14.47 dB`)
- $\rho = 3/2$: max `1/25`, MSE `1/76` (`-18.81 dB`)
- $\rho = 5/4$: max `1/81`, MSE `1/244` (`-23.87 dB`)
- $\rho = 9/8$: max `1/289`, MSE `1/868` (`-29.39 dB`)
- $\rho = 17/16$: max `1/1089`, MSE `1/3268` (`-35.14 dB`)

| Exact Rational Ensemble (`3.0 bpw`, $R^2=36$) | Flat $E_8^{\mathrm{Had}}$ (`HBG`) | Pure Uniform Conformal (`m=3, M=24`) | Graded Conformal-Wing (`0` extra block bits) | Verdict |
|---|---:|---:|---:|---|
| **`E1` Inner Cavity** (`4` micro vectors) | `15/8` (`1.875`) | — | **`0`** | **Graded dominates** |
| **`E2` Bulk Typical Shell** ($E = 359/8$) | **`15/8`** (`1.875`) | `3682629051/536870912` (`6.859`) | **`15/8`** (`1.875`) | **Uniform fails (`3.66x` tax); Graded ties (`0` loss)** |
| **`E3` Heavy-Tail Outliers** ($\|x\|^2 = 72..800$) | rational clip LB `4.50..182.41` | — | **`0.1385..1.3003`** (`-25.89` to `-31.60 dB`) | **Graded dominates (`24.2x–242.8x` vs LB)** |
| **`E4` Inner Steiner Triple** (`(1,1,1,1/100,0^4)`) | `9801/20000` (`0.49005`) | — | **`1/20000`** (`0.00005`, **`9,801x` cut**) | **Graded eliminates bleed** |
| **`E5` Bulk Steiner Triple** (`(3,1,1,1/100,0^4)`) | `9801/20000` (`0.49005`) | min Shell-3 cont. `35937/80000` (`0.4492`) | `9801/20000` (`0.49005`, **bleed persists**) | **Honest Law-2b Obstruction ($\mathbb{Z}^8$: `0` bleed, `1/20000` total)** |

---

## 5. Honest Scope & Verification Summary (Law 4)

- **Scope:** All claims are exact algebraic/geometric theorems in $\mathbb{R}^8$ over $\mathbb{Z}/\mathbb{Q}$ for fixed 8-block and coupled-128D ball budgets; end-to-end neural-network KL on real model weights requires integrating graded conformal-wing lookup/indexing into a live quantization harness (e.g. `siliquant`) and is not claimed here.
- **Verification:** Reproducible via `python3 -I -B conformal_coxeter_oracle.py --standalone --json` and `python3 -I -B conformal_coxeter_audit.py --check conformal_coxeter_certificate.json`; verified across 4 `-I` modes, `100.0%` live `aletheia v2.0.0 mutate --max-mutants 0`, `100.0%` exhaustive AST `BoolOp` sweep, and `aletheia v2.0.0 audit-check --strict-ancestors --safety-scan PASS`.
