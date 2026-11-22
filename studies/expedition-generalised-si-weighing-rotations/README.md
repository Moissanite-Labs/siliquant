# Sparse Circulant Weighing-Matrix Outlier Rotations & `generalised.si` Exact Bridge (`AUDIT-101R2`)

> **Status:** Independently `VERIFIED` (`AUDIT-101R2`, `honest_novelty_tier: N0`, `oracle_class: exact` + `anchor`).
> **Seats:** Author `peer-weaver` · Independent Clean-Room Verifier `verifier`.
> **Runtime Dependencies:** Python 3 standard library only (`int`, `fractions.Fraction`; zero floats on trust paths or JSON, zero bare `assert`s).
> **Scope Boundary (Honest Disclosure):** This package verifies exact algebraic rotation identities, signed-add operation reductions, and combinatorial certificates in pure $\mathbb{Z}$/$\mathbb{Q}$. It does **not** report real-model GGUF output KL or TPU/SIMD wall-clock cycles (those remain tracked in Issue `#25`).

## 1. Exact Pure-$\mathbb{Q}$ Sparse Weighing-Matrix Rotations for `siliquant` (`bounds/`, `shape/`, PR `#24`/`#26`, Issue `#25`)

Extending `PR #24`'s `Hadamard E8_Had` orthogonal outlier-flattening beyond power-of-2 Walsh–Hadamard widths, any proper circulant weighing matrix $W(n, k) \in \{-1, 0, +1\}^{n \times n}$ satisfying $W W^T = k I_n$ defines an exact zero-multiplication orthogonal rotation $Q = \frac{1}{\sqrt{k}} W(n, k)$ with:
- **Exact energy conservation over $\mathbb{Q}$:** $\|W x\|_2^2 = k \|x\|_2^2$ for every $x \in \mathbb{Q}^n$ (zero floating-point multiplications — entries strictly in $\{-1, 0, +1\}$).
- **Exact $t$-sparse outlier flattening bound:** $\frac{\|Q x\|_\infty^2}{\|Q x\|_2^2} = \frac{\|W x\|_\infty^2}{k \|x\|_2^2} \le \frac{\min(t, k)}{k}$ (tight `1/k` for 1-hot coordinate outliers!).
- **Exact signed-add operation reduction:** $k$ signed additions per output coordinate vs. $n$ for a dense $n \times n$ rotation (**exact `n/k` reduction factor** for branchless SIMD/Pallas & TPU v5e dequant kernels!).

### Verified Live Pillar-5 Witnesses (Pinned in [`generalised_si_certificate.json`](generalised_si_certificate.json))

| Verified Matrix (`GSI-R-0005`) | Input Vector | Exact Energy $\|Wx\|_2^2 / \|x\|_2^2$ | Peak Squared Ratio (vs. CS Bound) | Signed-Add Density | Dense Op Reduction |
|---|---|---:|---|---:|---:|
| `CW(120, 36)` class 0 | 1-hot (`100 e_0`) | `360000 / 10000 = 36` | **`1/36`** (`<= 1/36`) | `3/10` (`30%`) | **`10/3×`** (`3.33×`) |
| `CW(120, 36)` class 0 | 2-hot (`100 e_0 + 25 e_1`) | `382500 / 10625 = 36` | **`25/612`** (`< 1/18`) | `3/10` (`30%`) | **`10/3×`** (`3.33×`) |
| `CW(180, 36)` class 0 | 1-hot (`100 e_0`) | `360000 / 10000 = 36` | **`1/36`** (`<= 1/36`) | `1/5` (`20%`) | **`5×`** |
| `CW(98, 16)` (`lj_weight16.json`) | 1-hot (`100 e_0`) | `160000 / 10000 = 16` | **`1/16`** (`<= 1/16`) | `8/49` (`16.3%`) | **`49/8×`** (`6.125×`) |

In addition, [`GENERALISED_SI_CATALOG.md`](GENERALISED_SI_CATALOG.md) (§2.1) and [`generalised_si_catalog.json`](generalised_si_catalog.json) record the strict Law-5 mapping of **`11 / 31`** `generalised.si` domains into `siliquant`:
- **Non-$2^m$ Hadamard / Skew-Hadamard / Paley-Block & Williamson Rotations (`bounds/`, `shape/`, PR `#24`, Issue `#25`):** `H_212` (`Good_53`), `H_220` (`Good_55`), `H_300` (`Williamson_75`), `34/34` T-matrices (`101..359`), `WT47`/`WT83`, plus `vatskishan03/Hadamard` `PALEY_BLOCKS.md` & `PALEY_SKEW.md`.
- **Binary / Type-II Self-Dual Code $\to$ Construction A Lattices (`lattice/`, Issue `#4`):** `self-dual-codes-stabilizer-states` ($\Lambda(C) = (C + 2\mathbb{Z}^n)/\sqrt{2}$, generalizing $E_8$ from `[8,4,4]`), `5g-nr-ldpc-minimum-distances`, `candidate-binary-codes` (`5/5` $\mathbb{F}_2$-rank verified; `d_min` lower bound open).
- **Silicon-Floor Ceilings (`bounds/`, Issue `#25` roofline):** `euv-stochastic-failure-certificates`, `steep-switch-limits`, `placement-lower-bounds` (transparently marked `0` public certificate files on `generalised.si`).

## 2. Exact Offline Reproduction (3-Mode Isolated `-I` R4 Parity)

From this directory:

```bash
# 3-mode isolated determinism check on offline standalone mode (includes CW(7,4) rotation witness, 59511dde...):
python3 -I -B generalised_si_certifier.py --standalone --json | sha256sum
python3 -I -O -B generalised_si_certifier.py --standalone --json | sha256sum
python3 -I -B -W error generalised_si_certifier.py --standalone --json | sha256sum

# R5 mutation-killing & fail-closed unit suite (4/4 OK):
python3 -I -O -B -W error test_generalised_si_r5.py
```

**Offline vs. Live-Intake Note:** `generalised_si_certifier.py --standalone --json` and `test_generalised_si_r5.py` run 100% offline with Python 3 stdlib only. [`generalised_si_certificate.json`](generalised_si_certificate.json) (`695d0df4...`) and [`generalised_si_catalog.json`](generalised_si_catalog.json) (`39c64b58...`) are pinned live-intake certificates computed via `--live --intake-root <dir>` over the 2026-11-22 public `generalised.si` + `vatskishan03/{Hadamard,Quantum}` downloads (every upstream file SHA-256 is pinned inside the JSON for external replay). External `aletheia v2.0.0` mutation receipt: `133/133` (`100.0%`) mutants killed + `119/119` (`100.0%`) AST boolean sub-conditions killed (`0` survivors).
