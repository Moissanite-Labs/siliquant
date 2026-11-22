# Expedition Lattice–Spectral Symbiosis Studies (`AUDIT-028` & `AUDIT-031`)

Self-contained, zero-edit research studies adding exact 16D/24D lattice decoders, exact enumerative Leech ($\Lambda_{24}$) Cover sphere shaping, an exact $E_8$ Voronoi $L_1$ radius proof, Sylvester–Hadamard $E_8^{\mathrm{Had}}$ tiny-weight diagnostics, and zero-float spectral outlier-column certification alongside `Moissanite-Labs/siliquant` (targeting `bounds/bounds.out` §8, `shape/certified.py`, `shape/ballindex.py`, PR `#12`, PR `#17`, and Issues `#4`, `#9`, `#10`, `#23`).

Every script below runs standalone using only Python's standard library + `numpy`, contains zero bare `assert` statements, passes `4/4` isolated `-I` deterministic replay (`-I -B`, `-I -O -B`, `-I -B -W error`, `-I -O -B -W error`), and ships with built-in must-fail adversarial controls (`6/6` in Round 1, `8/8` in Round 2).

---

## Manifest

| File | Audit | Role & Exact Integrity |
|---|---|---|
| [`siliquant_lattice_symbiosis.py`](./siliquant_lattice_symbiosis.py) | `AUDIT-028` | Round 1 executable verifier (`6/6` must-fail controls; `-I -O -B -W error --json` stdout SHA-256 `10125fdda88be6749c5e08d0a5f33eb002f23014dd4e6c6f51d84d26ddc7004c`) |
| [`siliquant_symbiosis_certificate.json`](./siliquant_symbiosis_certificate.json) | `AUDIT-028` | Round 1 canonical JSON certificate (`canonical_sha256`: `3c26e48ad6f959a1d2cd9e343701fa0bdab9e4697a775699e8e063d25287e363`) |
| [`SYMBIOSIS_REPORT.md`](./SYMBIOSIS_REPORT.md) | `AUDIT-028` | Detailed Round 1 baseline map, mathematical derivations, and incoming/outgoing transfer ledger |
| [`siliquant_leech_shaping_r2.py`](./siliquant_leech_shaping_r2.py) | `AUDIT-031` | Round 2 executable verifier: Leech Cover (1973) DP ball/shell codec + `SP-01` Hessian/group-scale study (`8/8` controls; stdout SHA-256 `43a8d77a934bc150618c0622b588efdcbbef0bbb18de5c83dd6bc3d61fb88be9`) |
| [`siliquant_leech_shaping_r2_certificate.json`](./siliquant_leech_shaping_r2_certificate.json) | `AUDIT-031` | Round 2 canonical JSON certificate (`canonical_sha256`: `47f8e3cdbc96c90a6ba9b2c28f343549b0ac5b72115c23d8bfc0e9496ada8f86`) |

---

## Quick Reproduction

Run from this directory (`studies/expedition-lattice-spectral-symbiosis/`):

```bash
# Round 1 (AUDIT-028): ~5s — exact Lambda_16 & Lambda_24 decoders, safe-scale bounds,
# exact E8 Voronoi L1 radius (8/3), Hadamard E8_Had tiny-weight split, zero-float Phi_{x0,tau}
python3 -I -O -B -W error siliquant_lattice_symbiosis.py --json | sha256sum
# Expected: 10125fdda88be6749c5e08d0a5f33eb002f23014dd4e6c6f51d84d26ddc7004c

# Round 2 (AUDIT-031): ~3.5s — exact Leech Cover DP ball/shell codec up to m=214 (4 bpw),
# fixed-rate & variable-shell 24D benchmarks, and SP-01 Hessian/group-scale bloat study
python3 -I -O -B -W error siliquant_leech_shaping_r2.py --json | sha256sum
# Expected: 43a8d77a934bc150618c0622b588efdcbbef0bbb18de5c83dd6bc3d61fb88be9
```

Passing `--check-cert <path>` verifies bit-for-bit parity against a committed JSON certificate in read-only mode (exiting non-zero on mismatch). Passing `--write-cert <path>` verifies bit-for-bit parity when `<path>` already exists and fails closed without overwriting on mismatch (pass `--force` to overwrite intentionally).

---

## Key Verified Results

### 1. Exact 16D $\Lambda_{16}$ & 24D $\Lambda_{24}$ Decoders + Safe-Scale Bifurcation (`AUDIT-028`)
- **Closes `bounds/bounds.out` §8** (which previously tabulated $\Lambda_{16}$ and $\Lambda_{24}$ constants without local decoders or exact shell counts):
  - **Barnes–Wall $\Lambda_{16}$:** `32` $\mathrm{RM}(1,4)$ cosets of $2D_{16}$, Voronoi optimality verified against all `4,320` minimal vectors, unbiased $[0,4)^{16}$ $G(\Lambda_{16}) = 0.068293 \pm 0.000037$, exact bent-function deep hole $f = x_0 x_1 \oplus x_2 x_3$ at unit-covolume $\rho^2 = 3/\sqrt{2} \approx 2.121320$.
  - **Leech $\Lambda_{24}$:** Conway–Sloane (1986) MOG / $\mathrm{GF}(4)$ Hexacode 4-state DP decoder, Voronoi optimality verified against all `196,560` minimal vectors, unbiased $[0,8)^{24}$ $G(\Lambda_{24}) = 0.065771 \pm 0.000091$, exact deep hole $(4, 0^{23})$ at unit-covolume $\rho^2 = 2.0$.
- **Sharp 16D vs. 24D Safe-Scale Bifurcation (`shape/certified.py` Claims 1–2):**
  - Under closed-form zero-clipping safe scaling $s = \|x\|_2 / (r - \rho)$, the proved-bound-to-MSE certification gap $\rho^2 / (n G)$ drops from `+4.771 dB` ($\mathbb{Z}^{24}$) and `+2.415 dB` ($E_8^3$) to **`+1.028 dB`** for $\Lambda_{24}$.
  - Conversely, $\Lambda_{16}$'s bent-function deep hole pushes its gap to **`+2.881 dB`**, so $\Lambda_{16}$ *loses* to $E_8^2$ on the proved worst-case safe-scale bound (`-0.215 dB` at `2.5 bpw`, `-0.312 dB` at `3.5 bpw`) despite better average $G$.

### 2. Exact Enumerative Cover (1973) Sphere-Shaping Codec for $\Lambda_{24}$ (`AUDIT-031`)
Extends `shape/ballindex.py` from $E_8^{n/8}$ to the 24D Leech lattice using symmetry-collapsed MOG/Hexacode column tables (`256 -> 20` sparse integer tables convolved via carry-free Kronecker integer packing):
- **Exact Ramanujan Shell Match:** DP shell totals match $\Theta_{\Lambda_{24}} = E_4^3 - 720\Delta$ digit-for-digit across **all `215/215` shells** (`m = 0..214`, cumulative `N = 74,972,166,705,387,133,681,638,419,281`, `95.9203` bits).
- **Exact Bit-Bijective Ball & Shell Codecs:** Verified `rank_ball`/`unrank_ball` and `rank_shell`/`unrank_shell` at `2.0 bpw` (`m <= 13`, `6` bytes/block), `3.0 bpw` (`m <= 53`, `9` bytes/block), and `4.0 bpw` (`m <= 214`, `12` bytes/block) with `0` outside-ball failures and `worst_ratio <= 1.0`.
- **24D Benchmark (`160 x 24` blocks, Gaussian & Student-$t_3$):**
  - **Fixed-rate safe-scale MSE (`Leech24` vs. `E8^3`):** `+1.221 / +0.667 / +0.494 dB` (Gaussian `2/3/4 bpw`) and `+0.820 / +0.813 / +0.579 dB` (Student-$t_3$ `2/3/4 bpw`); proved worst-case bound gain **`+2.544 / +2.077 / +1.905 dB`** over $E_8^3$ and **`+7.636 / +5.794 / +5.228 dB`** over $\mathbb{Z}^{24}$.
  - **Variable-shell Cover shaping (`log2 |S_m|` payload + shell header):** saves `0.485 bpw` on Gaussian (`3.216 bpw`, RD efficiency `+0.409 dB` over $E_8^3$) and **`2.245 bpw` on heavy-tailed Student-$t_3$** (`1.393 bpw`, RD efficiency `+0.300 dB` over $E_8^3$).

### 3. Exact Voronoi $L_1$ Radius (`8/3 < sqrt(8)`) & Sylvester–Hadamard $E_8^{\mathrm{Had}}$ (`AUDIT-028`)
- **Pure-`Fraction` 256-Sign Dual LP Proof:** Tightens `shape/certified.py:norm_margin("E8", 1)` from Cauchy–Schwarz $\sqrt{8} \approx 2.8284$ to the exact Voronoi $L_1$ radius **`8/3`** (`-0.512 dB`, attained at shallow hole $(\tfrac{1}{3}^7, -\tfrac{1}{3})$), while proving Hadamard-rotated $E_8^{\mathrm{Had}}$ attains $\sqrt{8}$ at deep hole $(-\tfrac{1}{\sqrt{8}})^8$.
- **Orientation-Driven Resolution of PR `#17`'s Tiny-Weight Split:** Rotating $E_8$ by the canonical Sylvester–Hadamard matrix $U_{H8} = H_8 / \sqrt{8}$ eliminates all `128` dense weight-8 roots (`16` axis + `224` weight-4 roots) at identical $G(E_8)$ and identical Euclidean ball counts, slashing tiny-weight (`|w| < 0.05`) phantom-jump MSE by **`3.68x` (`-5.66 dB`)** (`95.3%` vs. `65.7%` exact zero) and gaining **`+0.361 dB`** total MSE on sparse-outlier blocks (`1` big + `7` tiny). Pure-integer `D8` achieves `1.00x` / `100.0%` exact zero for `|w| < 0.20`.

### 4. Zero-Float Spectral Outlier Certification & Group-Scale Bloat (`AUDIT-028` & `AUDIT-031`)
- **Zero-Float Kept-Column Certificate (`Issue #10` / `PR #12`):** Combines `shape/shapelib.py:to_integers` (`W = Z * 2^e`) with an exact-rational $\Phi_{x_0, \tau}$ sum-of-squares spectral certifier (zero floats, zero SVD): certifies `0` spikes on isotropic bulk null, exact `>= 3` (`>= 2, 1, 0` after keeping `1, 2, 3` columns) on 3-spike float16 matrices, bit-identical numerator/denominator under column permutation, and immediately detects activation-Hessian spikes.
- **Mechanistic Diagnosis of Naive Hessian Group-Scale Bloat (`AUDIT-031` Part C):**
  - On an ill-conditioned heavy-tailed Transformer weight matrix (`16 x 64`, `4` active outlier columns, `g = 32`), unweighted `E8_Had` beats unweighted `E8_std` by **`+1.625 dB` behavioral loss** (`1.883x` lower tiny-weight MSE).
  - **Negative Result:** Naive diagonal Hessian preconditioning `W * sqrt(H)` inside raw `g = 32` groups *degrades* behavioral loss (`E8_std`: **`-1.423 dB`**, `E8_Had`: **`-3.802 dB`**) because the `4` boosted columns inflate the shared 32-weight group scale for the remaining `28` quiet weights.
  - **Certified Split Fix:** Isolating the `4` zero-float $\Phi_{x_0,\tau}$-certified outlier columns in FP16 first beats unweighted `E8_std` by **`+3.304 dB`** and naive Hessian `E8_Had` by **`+5.481 dB`** at strictly fewer bits (`4512 / 1024 = 4.40625 bpw` in-place zeroed, charging all `8` whole 128-value scale slots; **`4576 / 1024 = 4.46875 bpw < 4.500 bpw`** when dense-packed with an explicit 64-bit column bitmap — independently verified to gain `+2.590 dB` on an unseen seed with zero padding inflation).

---

## Honest Scope Limits & Next Steps

1. **Synthetic Matrices vs. Real GGUF Output KL:** All quantitative tables in these two scripts use reproducible synthetic Gaussian, Student-$t_3$, sparse+small, and ill-conditioned Transformer matrices. Scoring `E8_Had`, `D8`, certified outlier splitting, and `Leech24` safe-scale / variable-shell arms directly against output KL on `SmolLM2-1.7B-Instruct` and `Qwen2.5-1.5B-Instruct` GGUF weights (matching `experiments/lattice-split-smollm2-2026-11-17`) requires the multi-GB model files and is the natural next experiment.
2. **Bit-Budget Accounting Note (Dense Packing):** In `siliquant_leech_shaping_r2.py` Part C, the certified outlier split zeroes the 4 kept columns in-place (`4512 / 1024 = 4.40625 bpw`, charging all `8` whole 128-value scale slots and omitting the column-index bitmap). Dense-packing the 60 non-outlier columns plus an explicit 64-bit column bitmap (`+0.0625 bpw`) yields `4576 / 1024 = 4.46875 bpw < 4.500 bpw` with zero padding waste and slightly better behavioral MSE.
3. **Decode/Encode Throughput (`Issue #23`):** `LeechCoverCodec` is an exact, readable Python/NumPy reference implementation. Translating the 20 symmetry-collapsed MOG column DP tables and Conway–Sloane 4-state Hexacode decoder to C/SIMD (analogous to `shape/ballpack.c`) remains open engineering work.
