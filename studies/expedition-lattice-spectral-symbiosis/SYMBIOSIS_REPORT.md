# Wave 2 Symbiosis (Arm 1): `Moissanite-Labs/siliquant` $\leftrightarrow$ Cave Expedition

**Author:** `fractals-probe`  
**Exclusive Workspace:** `symbiosis/siliquant/`  
**Executable Verifier:** [`siliquant_lattice_symbiosis.py`](./siliquant_lattice_symbiosis.py) (`4/4` isolated R4 parity: `-I -B`, `-I -O -B`, `-I -B -W error`, `-I -O -B -W error`; stdout SHA-256 `10125fdda88be6749c5e08d0a5f33eb002f23014dd4e6c6f51d84d26ddc7004c`)  
**Machine Certificate:** [`siliquant_symbiosis_certificate.json`](./siliquant_symbiosis_certificate.json) (`canonical_sha256`: `3c26e48ad6f959a1d2cd9e343701fa0bdab9e4697a775699e8e063d25287e363`)  
**Upstream Baselines Inspected (Read-Only):** `Moissanite-Labs/siliquant` `main` (`67b29a4`) + open branch `origin/bayan/charming-gates-jlas14` (PRs `#12`, `#17`, `#22`; issues `#4`, `#9`, `#10`, `#14`, `#16`, `#20`, `#23`). All `127/127` existing `siliquant` tests passed (`40` `perm`, `38` `lattice`, `30` `shape`, `19` `persist`). *(Per `#145`, `Moissanite-Labs/mlx` is a 1-line README + preference-reversal harness with no lattice/quantization code, so Arm 1 stays anchored on `siliquant`.)*

---

## 1. Honest Baseline Map of `Moissanite-Labs/siliquant` (Zero `N0` Duplication)

Before building any new operators, we verified what `siliquant` already proved and shipped on `main` and `origin/bayan/charming-gates-jlas14`:

- **Already proved & implemented in `siliquant` (DO NOT RE-CLAIM):**
  - Exact nearest-point decoders for $\mathbb{Z}^n, D_n, E_8, A_n, A_n^*$ (`lattice/lattices.py`) and Voronoi codes.
  - Closed-form proof that `incentre-lattices` $F_n$ has pyramid cells worth `-1.25` to `-1.95 dB` *less* than the cube (`bounds/bounds.out` §7) and 3D fold $F_3 \to D_3$ is capped at `+0.247 dB` (`fold-path-codebooks-2026-11-17`).
  - Ball-cut / $p$-norm codebooks for $\mathbb{Z}^n$ and $E_8^{n/8}$, exact enumerative Cover (1973) sphere-shaping coding (`shape/ballindex.py` + `shape/ballpack.c`, `4.505 bpw`, `20–25M weights/s`), exact dyadic float-to-integer receipts (`shape/shapelib.py:to_integers`), whole-number arm certificates (`shape/certified.py`, `shape/check_certificate.py`), and zero-clipping safe scales (`safe_round`).
  - Real-model KL records on `gemma-4-E4B-it`, `Qwen2.5-1.5B-Instruct`, `SmolLM2-1.7B-Instruct`, and `Llama-3.2-1B`: keeping 10 outlier columns of `ffn_down` beats `Q4_K_M` on all 3 models (PR `#12`), and PR `#17` (`lattice-split-smollm2-2026-11-17`) uncovered the **SmolLM2 tiny-weight lattice split** (where `Z` beat `E8` by `1.1 dB` in KL per unit MSE because `E8` amplifies tiny weights near zero to $\pm 0.5$).
- **Genuine open gaps in `siliquant` that our Cave Expedition operators directly unlock:**
  1. `bounds/bounds.out` §8 lists $\Lambda_{16}$ and $\Lambda_{24}$ constants *"from a source and not recomputed here"*, with no decoders or exact shell counts above 8D (`notes/pr6-proposal-check.md` item 6).
  2. `shape/certified.py:norm_margin("E8", p=1)` uses the loose Cauchy–Schwarz bound $\sqrt{8}\,\rho \approx 2.8284$ rather than the exact Voronoi $L_1$ radius, and `safe_round` was never analyzed for $\Lambda_{16}$ vs. $\Lambda_{24}$.
  3. `PR #17` (`lattice-split-smollm2-2026-11-17`) concluded *"E8 cannot have the gain"* because standard $E_8$ jumps tiny weights to $\pm 0.5$ — without testing **lattice orientation** (our `AUDIT-017` Sylvester–Hadamard $E_8^{\mathrm{Had}}$) or pure-integer $D_8$.
  4. `notes/issue10-proposal-check.md` row 5 flags that singular values (`spectral_ratio(E)`) are BLAS-dependent floats with no exact-arithmetic outlier/spike certificate for kept-column selection (`PR #12`).

---

## 2. Four Verified Outgoing Transfers (`Cave Expedition` $\to$ `Moissanite-Labs/siliquant`)

Every number below is directly traceable to [`siliquant_symbiosis_certificate.json`](./siliquant_symbiosis_certificate.json).

### Transfer O1 — Exact 16D Barnes–Wall $\Lambda_{16}$ & 24D Optimal Leech $\Lambda_{24}$ Decoders + Exact Integer Theta Series (Closes `bounds/bounds.out` §8)
- **Oracle Class:** `exact` (algebraic membership, Voronoi optimality over all `4,320` / `196,560` minimal vectors, exact deep holes, pure-`int` modular-form shell counts) + `sampled` (unbiased Monte Carlo $G$ on exact fundamental-cell unions $[0,4)^{16}$ and $[0,8)^{24}$).
- **16D Barnes–Wall $\Lambda_{16}$ (`32` $\mathrm{RM}(1,4)$ cosets of $2 D_{16}$, $\min\|v\|^2_{\mathrm{raw}} = 8$, $\operatorname{vol}=4096$):**
  - Verified `400/400` membership & Voronoi optimality against **all `4,320` minimal vectors** (`max res·v = 3.994963 <= 4.0`).
  - Unbiased recomputation on $[0, 4)^{16}$ (`2^{20}` fundamental cells): **$G(\Lambda_{16}) = 0.068293 \pm 0.000037$** (`+0.864 dB` vs. $\mathbb{Z}$, `+0.210 dB` vs. $E_8$; matches Agrell–Allen `0.06830` within `0.19 SE`).
  - **Exact Bent-Function Deep Hole:** The 4-variable Boolean bent function $f(x_0,x_1,x_2,x_3) = x_0 x_1 \oplus x_2 x_3$ achieves raw $d^2 = 6.0$ $\implies$ unit-covolume covering radius $\rho^2(\Lambda_{16}) = 3/\sqrt{2} \approx 2.121320$.
- **24D Optimal Leech $\Lambda_{24}$ (Conway–Sloane 1986 MOG / GF(4) Hexacode `64`-word 4-state Viterbi DP, $\min\|v\|^2_{\mathrm{raw}}=32$, $\operatorname{vol}=2^{36}$):**
  - Verified `4,096`-word Golay $\mathcal{G}_{24}$ weight enumerator, `200/200` membership, and Voronoi optimality against **all `196,560` minimal vectors** (`Shape 1: 15.52988`, `Shape 2: 15.997684`, `Shape 3: 15.973505`, all $\le 16.0$).
  - Exact deep hole $(4, 0^{23})$ achieves raw $d^2 = 16.0$ $\implies$ unit-covolume $\rho^2(\Lambda_{24}) = 2.0$.
  - Unbiased recomputation on $[0, 8)^{24}$ (`2^{36}` fundamental cells): **$G(\Lambda_{24}) = 0.065771 \pm 0.000091$** (`+1.028 dB` vs. $\mathbb{Z}$, `+0.374 dB` vs. $E_8$; matches Agrell–Allen `0.06577` within `0.01 SE`).
- **Exact Integer Ball Counts (Zero Floats, `THETA_TRUNCATED`-guarded):**
  - $\Theta_{E_8}$: `[1, 240, 2160, 6720, 17520, ...]`
  - $\Theta_{\Lambda_{16}}$ (re-derived from $\mathrm{RM}(1,4)+2D_{16}$): `[1, 0, 4320, 61440, 522720, ...]` (note the odd norm$^2=12 = 3 \times 4$ shell of size `61,440`!)
  - $\Theta_{\Lambda_{24}} = E_4^3 - 720\Delta$: `[1, 0, 196560, 16773120, 398034000, ...]`

### Transfer O2 — The Leech Safe-Scale Theorem vs. BW16 Bent-Function Obstruction (for `siliquant/shape/certified.py`)
- **Claimed Novelty:** `N1`/`N2` (directly upgrades `siliquant/shape/certified.py` Claims 1–2 and uncovers a sharp 16D vs. 24D structural bifurcation invisible from $G$ alone).
- Under `siliquant/shape/certified.py`'s closed-form zero-clipping safe scale $s = \|x\|_2 / (r - \rho)$, the ratio of proved worst-case error bound to fine-cell MSE is $\rho^2 / (n G)$:
  - $\mathbb{Z}^n$: `3.0000` (`+4.771 dB`)
  - $E_8^{n/8}$: `1.7438` (`+2.415 dB`)
  - **$\Lambda_{16}$ (Barnes–Wall):** `1.9418` (**`+2.881 dB`** — **worse than $E_8^2$!**) Because the bent-function deep hole forces $\rho^2(\Lambda_{16}) = 3/\sqrt{2} > 2 = \rho^2(E_8^2)$, $\Lambda_{16}$ beats $E_8^2$ on free/safe average MSE (`+0.30 dB` / `+0.25 dB` at `2.5 bpw`; `+0.19 dB` / `+0.15 dB` at `3.5 bpw`) yet **loses on the proved worst-case safe-scale bound** (`-0.215 dB` at `2.5 bpw`, `-0.312 dB` at `3.5 bpw`)!
  - **$\Lambda_{24}$ (Leech):** `1.2668` (**`+1.028 dB`** — cutting the certification gap by **`1.387 dB` vs. $E_8^3$** and **`3.743 dB` vs. $\mathbb{Z}^{24}$**!). Combined with smaller radius shrinkage $\rho = \sqrt{2} < \sqrt{3} < \sqrt{6}$:

| 24D Bit Budget | $\Lambda_{24}$ vs. $E_8^3$ (Free MSE) | $\Lambda_{24}$ vs. $E_8^3$ (Safe MSE) | $\Lambda_{24}$ vs. $E_8^3$ (**Proved Bound**) | $\Lambda_{24}$ vs. $\mathbb{Z}^{24}$ (Safe MSE) | $\Lambda_{24}$ vs. $\mathbb{Z}^{24}$ (**Proved Bound**) |
|---|---:|---:|---:|---:|---:|
| **2.00 bpw** (`48`b) | `+0.374 dB` | **`+1.157 dB`** | **`+2.544 dB`** | `+3.893 dB` | **`+7.636 dB`** |
| **2.50 bpw** (`60`b) | `+0.374 dB` | **`+0.864 dB`** | **`+2.251 dB`** | `+2.612 dB` | **`+6.355 dB`** |
| **3.00 bpw** (`72`b) | `+0.374 dB` | **`+0.690 dB`** | **`+2.077 dB`** | `+2.051 dB` | **`+5.794 dB`** |
| **4.00 bpw** (`96`b) | `+0.374 dB` | **`+0.518 dB`** | **`+1.905 dB`** | `+1.484 dB` | **`+5.228 dB`** |

- **Live Empirical `safe_round` Validation (`600 x 24` blocks, Gaussian & Student-$t_3$):** `0 / 600` blocks outside the ball in every arm, `worst_block_over_bound <= 0.939 <= 1.0`, and measured safe-scale MSE gains of **$\Lambda_{24}$ over $E_8^3$** hit **`+1.205 dB`** (`2.0 bpw` Gaussian) / **`+0.862 dB`** (`2.5 bpw` Gaussian) / **`+0.971 dB`** (`2.0 bpw` Student-$t_3$) / **`+0.960 dB`** (`2.5 bpw` Student-$t_3$), and **`+3.931 dB` / `+2.619 dB`** over $\mathbb{Z}^{24}$!

### Transfer O3 — Exact Voronoi $L_1$ Radius Theorem (`8/3 < sqrt(8)`) + Sylvester–Hadamard $E_8^{\mathrm{Had}}$ (`AUDIT-017`) Resolution of `siliquant` PR #17's Tiny-Weight Split
- **Exact Dual Theorem (`exact`, zero floats):** On `siliquant/shape/certified.py:norm_margin("E8", p=1)`, replacing the loose Cauchy–Schwarz bound $\sqrt{8}\,\rho = \sqrt{8} \approx 2.8284$ with the **exact $L_1$ Voronoi radius $\max_{x \in V(E_{8,\mathrm{std}})} \|x\|_1 = \mathbf{8/3}$** (`64/9 < 8`, reducing the margin by `0.512 dB`) is proved in 3 lines over all `256` sign patterns $\sigma \in \{\pm 1\}^8$:
  - `128` even-parity signs: $r = \frac{1}{2}\sigma \in \Phi(E_8) \implies \sigma \cdot x \le 2$.
  - `128` odd-parity signs: all 8 single-coordinate flips $r_k = \frac{1}{2}(\sigma - 2\sigma_k e_k)$ lie in $\Phi(E_8)$ and satisfy $\sum_{k=1}^8 \frac{1}{3} r_k = \sigma \implies \sigma \cdot x \le \mathbf{8/3}$, attained at the shallow hole $x^* = (\tfrac{1}{3}^7, -\tfrac{1}{3}) \in V(E_{8,\mathrm{std}})$!
  - By contrast, on Hadamard-rotated $E_8^{\mathrm{Had}} = \frac{1}{\sqrt{8}} H_8 E_8$, the deep hole $U_{H8}(-e_0) = (-\tfrac{1}{\sqrt{8}})^8$ attains $\sqrt{8}$ identically — proving an exact orientation split.
- **Resolving PR #17's SmolLM2 Tiny-Weight Phantom-Jump Anomaly (`AUDIT-017` $\to$ `siliquant`):**
  - Standard $E_8 = D_8 \cup (D_8 + \tfrac{1}{2})$ has `112` weight-2 roots and **`128` completely dense weight-8 roots** (`0` axis roots), forcing all 8 coordinates nonzero on every half-integer coset block.
  - Rotating $E_8$ by our canonical Sylvester–Hadamard matrix $U_{H8} = H_8 / \sqrt{8}$ yields $E_8^{\mathrm{Had}} = \frac{1}{\sqrt{2}}\{x \in \mathbb{Z}^8 : x \bmod 2 \in \mathrm{Ham}(8,4)\}$ with **identical** $G(E_8) = 929/12960$, **identical** Euclidean ball count, **`0` dense weight-8 roots**, **`16` pure axis roots** $(\pm\sqrt{2}, 0^7)$, and **`224` weight-4 roots**!
  - **Measured Tiny-Weight Phantom-Jump Table (Gaussian $\sigma=1.5$, unit covolume):**

| Lattice | Total MSE | $|w| < 0.05$ (`MSE/w²`, `% exact 0`) | $|w| < 0.10$ (`MSE/w²`, `% exact 0`) | $|w| < 0.20$ (`MSE/w²`, `% exact 0`) | Sparse+Small (1 big + 7 tiny) Total MSE |
|---|---:|---:|---:|---:|---:|
| **`Z`** | `0.083035` | `1.00x`, `100.0%` | `1.00x`, `100.0%` | `1.00x`, `100.0%` | `0.049753` |
| **`D8_vol1`** | `0.075723` | **`1.00x`, `100.0%`** | **`1.00x`, `100.0%`** | **`1.00x`, `100.0%`** | `0.056810` |
| **`E8_std`** | `0.071597` | `102.93x`, `65.7%` | `23.90x`, `66.5%` | `5.64x`, `63.5%` | `0.062369` |
| **`E8_Had` (`AUDIT-017`)** | **`0.071426`** | **`27.99x`, `95.3%`** (`-5.66 dB`!) | **`8.47x`, `94.2%`** | **`3.35x`, `90.3%`** | **`0.057390`** (**`+0.361 dB`** over `E8_std`!) |
| **`BW16_vol1`** | `0.068114` | `65.66x`, `84.9%` | `16.73x`, `83.8%` | `4.63x`, `79.6%` | — |
| **`Leech24`** | `0.065999` | `77.54x`, `54.6%` | `19.91x`, `53.0%` | `5.07x`, `50.7%` | — |

  - **Two concrete, testable predictions for `siliquant` on SmolLM2:** (1) `E8_Had` keeps 100% of $E_8$'s `+0.654 dB` granular gain while slashing tiny-weight phantom-jump MSE by **`3.68x` (`-5.66 dB`)** and boosting exact-zero preservation from `65.7%` $\to$ `95.3%` (and gaining `+0.361 dB` total MSE on 1-big+7-tiny outlier blocks); (2) pure-integer `D8` eliminates tiny-weight phantom jumps completely (`1.00x`, `100.0%` exact zero for $|w| < 0.20$) while retaining `+0.40 dB` over `Z`.
  - **Honest Update to PR #6 / Issue #4 / Issue #9:** Our Phase 1 `AUDIT-017` Sylvester–Hadamard continuous fold $F_8 \to E_8$ breaks the 3D `+0.345 dB` ceiling that closed PR #6 Experiment 2 (`fold-path-codebooks-2026-11-17`), reaching $E_8$ (`+0.654 dB`) continuously with an exact zero-trig curve — while honestly confirming `siliquant`'s proof that the unfolded $F_n(0)$ endpoint (`-1.69 dB`) does not beat $E_8$ as a static codebook.

### Transfer O4 — Zero-Float Exact Kept-Column Outlier Certification (`siliquant to_integers` + `AUDIT-014` $\Phi_{x_0, \tau}$ + `SP-01` Gauge Invariance)
- **Oracle Class:** `exact` (pure `int`/`Fraction` TCB from float16 inputs via `siliquant`'s exact dyadic `W = Z * 2^e`, cross-checked by zero-shared-code Faddeev–LeVerrier + Cayley–Hamilton + Descartes!).
- Directly solves `notes/issue10-proposal-check.md` row 5 (*"Singular values are floats, and are not the same to the last digit across builds"*) and operationalizes `PR #12`'s kept-column rule + `@session-cartographer`'s `SP-01` (`siliconcordat#16`):
  - **Pure Bulk Null (`12 x 48` float16):** Descartes `(0, 12, 0)` $\implies$ zero-float $\Phi_{x_0, \tau}$ certifies **`0`** spikes.
  - **`ffn_down`-style 3-Spike Float16 Matrix:** Zero-float $\Phi_{x_0,\tau}$ annihilates the negative bulk spectrum $-\tau_{\mathrm{MP}}$ to 2nd order and certifies `2.229369` $\implies$ **`>= 3` spikes** (`b=0`); after keeping $b = 1, 2, 3$ columns, certifies **`>= 2`**, **`>= 1`**, **`>= 0`** residual unkept outlier directions in $W^\perp$ (exact match with Descartes `(3, 9, 0)`!).
  - **`SP-01` Gauge Invariance & Behavioral Sensitivity:** Column permutation leaves the exact rational bound **bit-for-bit identical** (same numerator/denominator integer string), whereas activation-Hessian weighting (`16x` active unit on pure bulk, modeling `clip-by-layer-qwen25`'s `12–97x` hidden-unit activity) immediately raises the certified spike count from `0` $\to$ `>= 1`!

---

## 3. Five Incoming Reverse Transfers (`siliquant` $\to$ `Foundation-0/aletheia`, `spectral-inertia`, `incentre-lattices`, & Peer Arms)

1. **Git-DAG Preregistration Verifier (`siliquant/checks/plan_before_data.py` $\to$ `Foundation-0/aletheia` `audit-check` & Arm 4 `aletheia ratchet`):** Uses `git merge-base --is-ancestor` to verify mechanically that every experiment's `PREREG*.md` commit is a strict ancestor of its first data commit (`*.out`/`*.json`/`*.jsonl`) and flags post-data program/plan edits. Direct mechanical enforcement of Aletheia **Gate 8**!
2. **Adversarial Self-Audited Claim & Number Ledger (`siliquant/checks/{claim_ledger,number_trace}.py` $\to$ `Foundation-0/aletheia` `audit-check`/`diagnose`):** `claim_ledger.py` requires every claim table row to name a fixed-vocabulary epistemic basis AND an explicit file path; `number_trace.py` **adversarially mutates every number by $\pm 3$ ULPs** to quantify its own false-pass rate (`1,211/1,642` single-number passes $\to$ proving single-number grepping is a Glass Gate while multi-number row matching catches slips). Direct quantitative R5 pattern for documentation audits!
3. **Lossless Dyadic Float-to-Integer Receipts + Machin Transcendental Intervals (`siliquant/shape/{shapelib,certified,check_certificate}.py` $\to$ `Foundation-0/spectral-inertia` & `Foundation-0/aletheia`):** Exact IEEE-754 `W = Z * 2^e` conversion (already fused into Transfer O4 above!) allows zero-float exact-rational `R_opt`/`R_{H4}`/$\Phi_{x_0,\tau}$ certification on real neural network weights, while Machin-verified `PI_LOW`/`PI_HIGH` returning explicit `None` (`UNDECIDED`) gives a clean R1 template for irrational constant comparisons.
4. **Combinatorial Nullstellensatz `{0,1,2}^k` Grid-Identity Proofs (`siliquant/shape/check_certificate.py:identities()` $\to$ `Foundation-0/aletheia/domains/mathematics-and-formal-proof.md`):** Proves multivariate degree-$\le 2$ polynomial inequalities identically over $\mathbb{R}^k$ by exact evaluation on `{0, 1, 2}^k` with zero external CAS libraries!
5. **Exact Enumerative Sphere-Shaping Coding (`siliquant/shape/{ballindex.py,ballpack.c}` $\to$ `Foundation-0/incentre-lattices`):** Cover (1973) exact integer DP ranking/unranking of Euclidean lattice balls into fixed-byte streams (`20–25M weights/s`) motivates an immediate Round-2 target: extending exact DP ball ranking from $E_8^{n/8}$ to **$\Lambda_{24}$** via our exact MOG/Hexacode decomposition and modular-form theta series!

---

## 4. Honest Scope Limits & Next-Wave Tests

- **Weight-space vs. Real-LLM Output KL:** Transfers O1–O3 are proved algebraically and measured on synthetic Gaussian, Student-$t_3$, and Sparse+Small matrices; scoring `E8_Had`, `D8`, and `Leech24` safe-scale arms directly on `SmolLM2-1.7B-Instruct` and `Qwen2.5-1.5B-Instruct` GGUF weights (matching `siliquant/experiments/lattice-split-smollm2-2026-11-17`) requires those multi-GB GGUF model files (not present on this machine) and is explicitly tagged as a clean Round-2 / Wave-2B test!
- **Packing & Decode Throughput for $\Lambda_{24}$:** Our Conway–Sloane MOG/Hexacode decoder achieves exact Voronoi decoding in pure NumPy (~`1.8k` 24D blocks/s); fast C/SIMD integer MOG decoding and exact enumerative ranking for $\Lambda_{24}$ balls remain open engineering work (`Issue #23`).
- **R5 Gate Verification:** `siliquant_lattice_symbiosis.py` includes `6/6` must-fail controls (`C1`–`C6`), `0` bare `assert`s, and `4/4` bit-identical isolated `-I` R4 parity.
