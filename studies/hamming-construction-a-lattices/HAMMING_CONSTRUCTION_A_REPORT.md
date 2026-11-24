# Track S47: Extended Hamming Construction-A Lattices ($A_8, A_{16}, A_{32}$), Exact Syndrome Decoding Optimality, Voronoi Zero-Retention & `INT8`/`INT16`/`INT32`/`f64` Accumulator Bounds

- **Track ID:** `TRACK-S47-HAMMING-CONSTRUCTION-A-LATTICES`
- **Portal Reference:** Portal 52 (`Moissanite-Labs/siliquant#77`, survey commit `a486828`, plan commit `aec439b`)
- **Schema Version:** `aletheia.hamming_construction_a.v1`
- **Honest Scope:** `measured_live_model_weights: false` — all results below are exact pure-$\mathbb{Z}/\mathbb{Q}$ coding-theoretic, lattice, decoder, and IEEE-754 bit-width certifications; zero live LLM weight tensors were evaluated.
- **Honest Novelty Classification:**
  - **`N0` (Classical Prior Art):** Extended Hamming codes (Hamming 1950), Barnes–Wall $\Lambda_{16}$ (Barnes & Wall 1959), MacWilliams weight-enumerator identity (MacWilliams 1963), Construction-A lattices and soft-decision lattice decoding (Conway & Sloane 1986, 1999 Ch. 5/7/20), IEEE Std 754-2008 binary16/binary64 arithmetic, and `Moissanite-Labs/siliquant#77` experimental framing.
  - **`N1` (Exact-$\mathbb{Q}$ Verification Suite & Engineering Bridge):** Self-contained pure-$\mathbb{Z}/\mathbb{Q}$ certifier (`hamming_construction_a_oracle.py`, `hamming_construction_a_audit.py`), explicit naive 1–2 flip Wagner suboptimality witnesses (`A_8` weight-3, `A_16` weight-4, `A_32` weight-3) paired with a certified exact syndrome branch-and-bound decoder (verified against exhaustive brute force on $C_8$/$C_{16}$ and an exact analytical $\ge 3$-flip global-optimality lower bound on $A_{32}$), exact unit-covolume Voronoi dead-zone & discrete-Laplace zero-retention / parity-spill analysis, the sharp $W_{\max} \le 64$ vs. $W_{\max} = 65$ `f64` exact-decode boundary for 128-weight integer groups $\times$ finite `f16` activations, and the exact pure-$\mathbb{Q}$ data-free column-proxy inversion obstruction.
- **Runtime Hygiene:** Pure Python 3 standard library (`int`, `fractions.Fraction`), `0` IEEE-754 `float` literals or `float()` calls on trust paths, `0` bare `assert` statements, `0` `.isdigit()`, `0` `islink`/`is_symlink`, `0` mathematical RNG calls, `0` external writes, LF-only.

---

## 1. File Manifest & Certificate Digests

| File | Lines | Bytes | Mode | SHA-256 |
|---|---:|---:|---|---|
| [`hamming_construction_a_oracle.py`](hamming_construction_a_oracle.py) | `1,063` | `45,189` | `0o644` | `591cb773152c68b631b597f3d236e635b3b9eab235bbf3767685defe300de3d0` |
| [`hamming_construction_a_audit.py`](hamming_construction_a_audit.py) | `977` | `39,865` | `0o644` | `dd6eb32c9dc906e93616e1e0c71ba5d7825df20c331cd879891efb1614b780ee` |
| [`test_hamming_construction_a_r5.py`](test_hamming_construction_a_r5.py) | `792` | `47,341` | `0o644` | `3d31647c400b180a4d309968ab219e641e833df49a398c0dbabccbb6870473fa` |
| [`hamming_construction_a_certificate.json`](hamming_construction_a_certificate.json) | `1` | `7,541` | `0o600` | `51e2c9e2ec5528d6a76c2d840b5b4956acd393ec8565c1810ac8f0db3dcd719e` |

- **`payload_sha256`:** `137650e51c9ce475d4f9bc091eb60cbe989c330f55af9e3817b19e782d69a1e7`
- **`certificate_sha256`:** `e1696a0635ebdc3aefcbe729f2b12ee2c8707b6d4db2885982a453f32f7c5fda`
- **`canonical_sha256` (UTF-8 canonical JSON without trailing LF):** `27eed179796b7890a0682687e288ff3bb6c7497a5479ae0ded3bb39a9d76c954`

---

## 2. Pillar 1 — Exact Extended Hamming Codes & Construction-A Lattices ($A_8, A_{16}, A_{32}$)

For $m \in \{3, 4, 5\}$, $n = 2^m$, $k = 2^m - m - 1$, let $H_n \in \mathbb{F}_2^{(m+1)\times n}$ have row $0$ all-ones and column $j$ equal to $(1, \operatorname{bin}_m(j))^\top$, and let $G_n \in \mathbb{F}_2^{k\times n}$ be its systematic generator matrix satisfying $H_n G_n^\top \equiv 0 \pmod 2$. Every triple of distinct columns $\{i, j, \ell\}$ closes uniquely with $s = i \oplus j \oplus \ell \notin \{i, j, \ell\}$, giving $d_{\min}(C_n) = 4$ and exact weight-4 count $A_4(C_n) = \frac{1}{4}\binom{n}{3} = \frac{n(n-1)(n-2)}{24}$.

Full primal weight enumerators are certified via the exact integer MacWilliams transform from the $2n$-word dual code $C_n^\perp = \mathrm{RM}(1, m)$ (`16`, `32`, `64` dual codewords; `macwilliams_a4_exact: true` for all three). For $C_8$ (`16` codewords) and $C_{16}$ (`2,048` codewords), the primal weight enumerator is additionally cross-checked by exhaustive Gray-code enumeration (`exhaustive_weight_enum_checked: true`, `exhaustive_matches_macwilliams: true`); for $C_{32}$ (`67,108,864` codewords), exhaustive primal enumeration is not run (`exhaustive_weight_enum_checked: false`, `exhaustive_matches_macwilliams: null`) and the exact distribution rests solely on the certified MacWilliams identity.

For the Construction-A scaled lattice $A_n = (C_n + 2\mathbb{Z}^n)/\sqrt{2}$ (computed in the exact integer lattice $L_n = C_n + 2\mathbb{Z}^n$ with squared-norm factor `1/2` and exact Bareiss integer Gram determinants), the certificate records **both** the fundamental-cell covolume $\operatorname{covol}(A_n) = 2^{n/2 - k}$ (`covolume_A_n_q`) and the Conway–Sloane Gram determinant $\det(G(A_n)) = \operatorname{covol}(A_n)^2 = 2^{n - 2k}$ (`det_gram_A_n_q`), together with the scale-invariant Hermite constant $\gamma(\Lambda) = \lambda_1^2 / \operatorname{covol}(\Lambda)^{2/n} = \lambda_1^2 / \det(G)^{1/n}$:

| Lattice | Code $[n, k, d]$ | Code Rate $k/n$ | Codewords $2^k$ | $A_4(C_n)$ | $\lambda_1^2(A_n)$ | Kissing $\tau(A_n) = 2n + 16 A_4$ | Exact $\operatorname{covol}(A_n) = 2^{n/2-k}$ | Exact $\det(G(A_n)) = \operatorname{covol}^2$ | Exact $\gamma(A_n)^{32}$ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`A_8`** | `[8, 4, 4]` | `1/2` | `16` | `14` | `2` | **`240`** ($\cong E_8$) | **`1`** | **`1`** | `2^{32}` (`4,294,967,296`) |
| **`A_16`** | `[16, 11, 4]` | `11/16` | `2,048` | `140` | `2` | **`2,272`** | **`1/8`** | **`1/64`** | `2^{44}` (`17,592,186,044,416`) |
| **`A_32`** | `[32, 26, 4]` | `13/16` | `67,108,864` | `1,240` | `2` | **`19,904`** | **`1/1024`** | **`1/1048576`** | `2^{52}` (`4,503,599,627,370,496`) |

- **Exact $E_8 \cong A_8$ Sylvester–Hadamard Bijection:** $\frac{1}{2} H_8^{\mathrm{Syl}}$ maps all **`240/240`** $E_8$ roots (`112` $D_8$ + `128` spinor) bijectively onto all `240` minimal norm-squared-`4` vectors of $L_8 = C_8 + 2\mathbb{Z}^8$, certifying `siliquant#77`'s rotated-$E_8$ identity algebraically.
- **Honest Barnes–Wall $\Lambda_{16}$ Contrast:** $\tau(A_{16}) = 2,272 < \tau(\Lambda_{16}) = 4,320$ and $\gamma(A_8)^{32} = 2^{32} < \gamma(A_{16})^{32} = 2^{44} < \gamma(\Lambda_{16})^{32} = 2^{48} < \gamma(A_{32})^{32} = 2^{52}$. Packing Hermite constant $\gamma(\Lambda)$ is distinct from quantization normalized second moment $G(\Lambda)$ (evaluated empirically in `siliquant#77`).

---

## 3. Pillar 2 — Exact Syndrome-Guided Decoding vs. Exhaustive Brute Force, Analytical Lower Bound & Naive Wagner Controls

For $x \in \mathbb{Q}^n$, nearest-even $e_j$ and nearest-odd $o_j$ yield per-coordinate flip penalties $\Delta_j = |d_{j,0} - d_{j,1}| \ge 0$ and hard-decision syndrome $s = (s_0, s_{\mathrm{idx}})$. The $O(n)$ 1–2 flip Wagner candidate cost $U$ is **provably globally optimal** whenever $U \le \sum_{t=0}^2 \Delta_{\pi(t)}$ (if $s_0 = 1$) or $U \le \sum_{t=0}^3 \Delta_{\pi(t)}$ (if $s_0 = 0, s_{\mathrm{idx}} \neq 0$). When that sufficiency bound is not met, naive 1–2 flip decoding can be strictly suboptimal, and `decode_construction_a_exact` falls back to exact denominator-cleared integer syndrome branch-and-bound:

- **$A_8$ (`16` deterministic rational vectors):** `16/16` exact matches to `16`-codeword brute force AND `16/16` exact distance matches ($\|x - v_{L_8}\|^2 = 2\|u - v_{E_8}\|^2$) to Conway–Sloane $E_8$ decoding under $x = \frac{1}{2} H_8^{\mathrm{Syl}} u$.
- **$A_{16}$ (`24` deterministic rational vectors including half-integer boundaries):** `24/24` exact matches to `2,048`-codeword Gray-code brute force.
- **$A_{32}$ (`12` deterministic rational vectors):** `12/12` certified valid $L_{32}$ points with squared distance $\le$ Wagner 1–2 bound.
- **Explicit Naive 1–2 Flip Wagner Suboptimality Controls (`3/3`, with explicit proof provenance):**
  1. `A8_weight3_beats_1flip`: naive Wagner `643/400` (`naive_wagner_certified_optimal: false`) vs. exact B&B = **`363/400`** (`exhaustive_bruteforce_executed: true`, `optimal_proof_method: "exhaustive_bruteforce"`).
  2. `A16_weight4_beats_2flip`: naive Wagner `191/100` (`naive_wagner_certified_optimal: false`) vs. exact B&B = **`121/100`** (`exhaustive_bruteforce_executed: true`, `optimal_proof_method: "exhaustive_bruteforce"`).
  3. `A32_weight3_beats_1flip`: naive Wagner `643/400` (`naive_wagner_certified_optimal: false`) vs. exact B&B = **`363/400`** (`exhaustive_bruteforce_executed: false`, `optimal_proof_method: "analytical_ge3_flip_lower_bound"`): since the odd syndrome $s_0 = 1$ requires an odd number of flips from hard decision, every valid codeword either uses the unique 1-flip candidate (`643/400`) or flips $\ge 3$ coordinates at cost at least $\text{base} + \sum_{t=0}^2 \Delta_{\pi(t)} = 243/400 + 3 \cdot 40/400 = 363/400$, which is attained by a verified point in $L_{32}$.

---

## 4. Pillar 3 — Voronoi Zero-Retention Lift, Parity-Spill Control & Accumulator / `f64` Exact-Decode Theorem

Rescaling $A_n$ to unit covolume $\tilde{A}_n = 2^{k/n - 1/2} A_n = 2^{k/n - 1} L_n$ strictly expands the zero-cell packing radius squared $r_{\mathrm{pack}}(\tilde{A}_n)^2 = 2^{2k/n - 2}$ and $\ell_\infty$ zero-cube half-width $\alpha_\infty(\tilde{A}_n) = 2^{k/n - 2}$ as code rate $k/n$ grows (`1/2 -> 11/16 -> 13/16`), verified via exact certified integer-power enclosures on $2^{8/16} \in (1414/1000, 1415/1000)$, $2^{5/16} \in (1241/1000, 1242/1000)$, and $2^{3/16} \in (1138/1000, 1139/1000)$:

| Lattice | $(r_{\mathrm{pack}}^2)^8$ | $\alpha_\infty^{16}$ | Conservative $1/c_n$ Bracket | Discrete-Laplace Zero Retention (`256` exact weights) |
|---|---:|---:|---:|---:|
| **`A_8`** | `1/256` | `1/16777216` | `707/500` *(most generous to $A_8$)* | `152 / 256` = `19/32` (**`0.593750`**) |
| **`A_16`** | `1/32` (`8×`) | `1/2097152` (`8×`) | `621/500` *(strictest to $A_{16}$)* | `222 / 256` = `111/128` (**`0.867188`**) |
| **`A_32`** | `1/8` (`32×`) | `1/524288` (`32×`) | `1139/1000` *(strictest to $A_{32}$)* | `240 / 256` = `15/16` (**`0.937500`**) |

- **Mid-Band Weight-4 Parity-Spill Negative Control (`A_16`):** A single coordinate at `3/4` alongside three `1/3` coordinates snaps to an odd integer and pulls `3` small coordinates off zero (`nonzero_coords_after_decode = 4`), illustrating the exact boundary where minimum codeword weight `4` trades against dead-zone expansion.
- **Integer & WHT Accumulator Bounds (`W_max = 7`, `INT8` activations `[-128, 127]`):**
  - Single-block dot products: `A_8` `7,168`, `A_16` `14,336`, `A_32` `28,672` (all $\le 32,767$, `INT16`-safe).
  - Full `128`-weight group: `114,688 > 32,767` (overflows `INT16`; requires `INT32`).
  - Sylvester WHT on `INT8`: `WHT_8` `1,024`, `WHT_16` `2,048`, `WHT_32` `4,096` (all `INT16`-safe).
- **Sharp `f64` (`53`-bit significand) Exact-Decode Theorem for `siliquant#77` (`128`-weight integer groups $\times$ finite `f16` activations):**
  Every finite binary16 activation is an exact integer multiple of $2^{-24}$ bounded by $65,504 = 2047 \cdot 2^5$ (`1,098,974,756,864` units of $2^{-24}$). Over `128` weights with $|v_i| \le W_{\max}$:
  - **$W_{\max} \le 64$ (Unconditionally Exact):** $128 \cdot 64 \cdot 65,504 \cdot 2^{24} = \mathbf{9,002,801,208,229,888} \le 2^{53} = 9,007,199,254,740,992$ — every partial and total sum is **100% exact in `f64`** for ALL finite `f16` activations (even mixing `65,504` and subnormal `2^{-24}`!).
  - **$W_{\max} = 65$ (Sharp Failure Boundary):** $127 \cdot 65 \cdot 65,504 \cdot 2^{24} + 1 = \mathbf{9,072,036,617,912,321} > 2^{53}$ is odd, so `f64` must drop the $2^{-24}$ subnormal bit.
  - **Full `INT8` ($W_{\max} = 128$):** Remains 100% exact in `f64` whenever activations satisfy $|a_i| \le 32,752$ (`9,002,801,208,229,888 <= 2^53`).

---

## 5. Pillar 4 — Honest Scope & Exact Data-Free Column-Proxy Obstruction

To accompany `Moissanite-Labs/siliquant#77`'s empirical negative finding (`notes/colproxy.out`, showing neither `1/(column RMS)^2` nor RMSNorm gain predicts the real activation importance matrix), Pillar 4 certifies an explicit $2\times 2$ exact rational obstruction: with $W = \operatorname{diag}(1, 4)$ (`weight_col_rms_sq_q = ["1", "16"]`), the data-free proxy `1/(col RMS)^2 = [1, 1/16]` ranks column `0` above column `1` by `16:1`, while true activation covariance $\Sigma_X = \operatorname{diag}(1/64, 16)$ ranks column `1` above column `0` by `1,024:1` — an exact **ranking sign inversion** with total ratio distortion **`16,384`** (`calibration_activations_required = true`).

---

## 6. Verification & Reproducibility Commands

Run inside this directory (`100%` offline, Python 3 stdlib only):

```bash
# 1. Verify committed 0o600 canonical certificate (full replay + cross-multiplied secondary audit)
python3 -I -O -B -W error hamming_construction_a_audit.py --check hamming_construction_a_certificate.json --json

# 2. Standard 3-mode isolated -I parity + PYTHONHASHSEED=42 -B parity (all match 51e2c9e2...)
python3 -I -B hamming_construction_a_audit.py --standalone --json | sha256sum
python3 -I -O -B hamming_construction_a_audit.py --standalone --json | sha256sum
python3 -I -B -W error hamming_construction_a_audit.py --standalone --json | sha256sum
PYTHONHASHSEED=42 python3 -B hamming_construction_a_audit.py --standalone --json | sha256sum

# 3. Assertion-free verification, mutation-killing & Zone-B kernel-race test suite (8/8 PASS)
python3 -I -O -B -W error test_hamming_construction_a_r5.py
```

| Gate | Reproducible Result |
|---|---|
| Standard 3-Mode Isolated (`-I -B`, `-I -O -B`, `-I -B -W error`) + `PYTHONHASHSEED=42 -B` Parity | Bit-identical `51e2c9e2ec5528d6a76c2d840b5b4956acd393ec8565c1810ac8f0db3dcd719e` (`0o600`), `8/8 PASS` |
| Exhaustive Leaf, Zero/Negative-Denom, Shift-DoS & Direct Float Injection Sweep | `244` leaves: `973/973` `full_replay=False` + `973/973` direct `secondary_audit_payload` + `244/244` direct float injections + `472/472` live CLI `--check` (`0` accepted, `0` raw exceptions, `0` `rc=1` tracebacks) |
| Live `aletheia v2.0.0 mutate --max-mutants 0 --guards require_ok` | `35/35` (`oracle`) + `74/74` (`audit`) = **`109/109` (`100.0%`, `0` survivors, `truncated=False`)** |
| Whole-File Exhaustive AST `BoolOp` (`OP_SWAP + SUBCOND_DEL`) | `60/60` (`oracle`, `20` nodes) + `24/24` (`audit`, `8` nodes) = **`84/84` (`100.0%`, `0` survivors)** |
| 4-Surface Adversarial Zone-B Fuzzer | `23/23` CLI + `18/18` FR1 + `18/18` FR0 + `18/18` FR0+Sec = **`77/77 PASS` (`0` bypasses, `0` crashes)** |
| Concurrency & Kernel Races | `16T` distinct + `16T`/`80T` same (`0o600`, real `os.fsync`, `0` `.tmp`), `550` `renameat2(RENAME_EXCHANGE)` swaps & `550` hardlink races (`victim_intact=True`, `0` leaks) |
| `aletheia v2.0.0 audit-check --strict-ancestors --safety-scan` | `PASS` (`5` files, `0` findings, `0` bare `assert`s, `0` CRLF, `0` leaks, `PYC=0`) |
