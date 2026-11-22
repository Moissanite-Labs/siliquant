# AUDIT-116 (`Track S15d`): Attacking the `68%–75%` Stage-1 Inner `E_8` Leaf Unranking Bottleneck — Exact Width Proofs, `O(1)` Bucket Guides, Low-Norm LUT Concentration Bounds & `D_8^+` Leader-Orbit Phase Transition

**Audit ID:** `AUDIT-116` (`Track S15d`)  
**Predecessor Baseline:** `AUDIT-115R` (`Track S15c` — proved exact zero-table `E_8_Had` `WHT_8` isometry and pinpointed Stage-1 inner `E_8` leaf unranking as `67.99%–75.14%` of total CPU cycles)  
**Target Upstream Context:** `Moissanite-Labs/siliquant#23` (BallPack `E_8^4`), `#24` (`closest_E8_had`), `#25` (`EBG32:b136s8` & `EBG32:b88s8`)  
**Novelty Classification:** `N1` (exact pure-$\mathbb{Z}/\mathbb{Q}$ bit-width, `O(1)` top-bit guide, concentration-of-measure LUT hit bound, and `D_8^+` `516.58×` leader phase-transition proofs combined with head-to-head real-silicon C11 + AVX2/FMA benchmarking across 6 Stage-1 variants and 3 realistic weight distributions; honest CPU floor verdict)  
**Overall Verification Status:** `PASS` — **Zero algebraic defects, `34/34` (`100.0%`) mutation score + `7/7` (`100.0%`) AST `BoolOp` kill, zero warnings & zero hardware `div`/`idiv` across `13/13` hot kernels on both GCC 16 & Clang 23, zero mismatches across `8,388,608` decoded weights, bit-identical `z_output_sha256` match to `AUDIT-115`**

---

## 1. OWN_TERMS_FIRST — Exact Questions Tested & Verdicts

Following `AUDIT-115R`'s 4-stage cycle breakdown (`68%–75%` Stage 1 inner `E_8` leaf unranking, `23%–28%` Stage 0 outer splits, `< 4.5%` `WHT_8` + FMA combined), we test four concrete mathematical and microarchitectural questions head-to-head:

1. **Exact Stage-1 Integer Width Reduction:** Can Stage 1 drop `uint64_t` to pure `uint32_t` on `EBG32:b88s8` (`m_max = 48`) and drop `uint128_t` completely to pure `uint64_t` on `EBG32:b136s8` (`m_max = 391`), while compressing `c4` to `uint16_t` and `c2` to `uint8_t`?
   - **Verdict: PROVEN EXACT (`PASS`).** On `88b`, $\max_{m \le 48} S_1(m) = 31,456,320$ (`25` bits) and $W_1(48) = 360,605,761$ (`29` bits $< 2^{32}$) -> **100% pure `uint32_t` Stage 1**. On `136b`, $\max_{m \le 391} S_1(m) = S_1(390) = 16,749,815,040$ (`34` bits) and $W_1(391) = 1,525,056,448,801$ (`41` bits $< 2^{64}$) -> **100% pure `uint64_t` Stage 1, zero `uint128_t`**. Across BOTH regimes, $\max c_4 = 14,976 < 2^{16}$ (`uint16_t`) and $\max c_2 = 24 < 2^8$ (`uint8_t`).
2. **Accelerator `(A)` — Pure Narrow Width + `O(1)` Top-Bit CDF-Prefix Bucket Guide (`b = rem >> sh`):** Does replacing strided checkpoint binary search with an `O(1)` shift-indexed bucket guide (`0` divisions, `0` multiplies to index, `100%` bit-identical to `AUDIT-115`) speed up Stage 1 on real silicon?
   - **Verdict: CLEAR DROP-IN WINNER (`PASS`).** Cuts isolated Stage-1 CPU cycles by **`1.83×` (`gcc`) / `1.78×` (`clang`)** on `88b` (`22.74 -> 12.43` / `23.06 -> 12.95` nom. cyc/w) and **`1.33×` (`gcc`) / `1.53×` (`clang`)** on uniform `136b` (reaching **`1.72×`** on `clang` isotropic Gaussian, `43.99 -> 25.52` cyc/w!), boosting end-to-end fused Adjoint `E_8_Had` GEMV by **`1.39×–1.40×`** on `88b` (`93.5 KB` tables) and **`1.08×–1.15×`** on `136b` (`1.60 MB` tables) with **zero format change**.
3. **Accelerator `(B)` — Branchless Eytzinger `cmov` + Flat `int8[8]` Low-Norm Shell LUT (`m <= 2` `19.2 KB` L1d / `m <= 3` `73.0 KB` L2):** Does a flat low-norm shell LUT + Eytzinger search help under uniform ball sampling or realistic per-block-scaled Gaussian LLM weights?
   - **Verdict: HONESTLY FALSIFIED ON BOTH COUNTS.** Under per-block-scaled `EBG32`, the per-group scale pushes every 32D block toward the ball boundary ($\sum_{b=0}^3 m_b \approx m_{\max}$). On `136b`, `m <= 3` hit rate is **`0.0010%`** (uniform exact), **`0.0000%` (`0 / 524,288`)** on isotropic Gaussian (realistic Hadamard-rotated weights), and **`0.2075%`** even with `3%` `6×` outliers! On `88b`, `m <= 3` hits only `2.42%` (uniform), `2.78%` (isotropic Gaussian), and `14.40%` (heavy `6×` outliers). Meanwhile, Eytzinger `cmov` pointer-chasing yields `1.01×–1.09×` on `88b` and **regresses to `0.76×–0.95×`** on `136b`.
4. **Accelerator `(C)` — `D_8^+` Signed-Leader / Permutation-Orbit Leaf Unranking (`C1` `502`-byte binomial-mask vs `C2` `22`-shape `pshufb`/`psignb` LUT):** Can `D_8^+` orbit unranking beat the `8->4->2` parity tree, and how does its leader table scale from `88b` to `136b`?
   - **Verdict: SHARP $\Theta(m_{\max}^{7/2})$ PHASE TRANSITION PROVED & MEASURED.**
     - At `88b` (`m_max = 48`), there are only **`2,000` leaders** (`955` even + `1,045` odd, max `111` at `m = 47`, `32,000` B L1d-resident!). With the `764,024`-byte universal `22`-shape `pshufb` table (`95,503` permutations), **`(C2) d8_pshufb` achieves `1.71×` (`gcc`) / `1.80×` (`clang`) Stage-1 speedup (`1.36×` / `1.30×` GEMV)** — matching `(A) guide_a` in speed, but at **`8.6×` larger table size** (`806 KB` vs `93.5 KB`) and requiring a non-BallPack intra-shell ordering.
     - At `136b` (`m_max = 391`), the leader count explodes **`516.58×`** to **`1,033,160` leaders** (`409,367` even + `623,793` odd, max `8,749` at `m = 389`, **`19.41 MB`** total — `4.85×` larger than `4 MiB` L3!). On `gcc` uniform (`1.13×`) and on BOTH compilers' Gaussian-Iso (`1.29×`/`1.41×`) and Gaussian-Outlier (`1.08×`/`1.20×`) streams, `(C2) d8_pshufb` trails or loses to `(A) guide_a` while burning `12.1×` more memory (though `clang` hits `1.68×` / `1.34×` GEMV on the tiled 4,096-entry uniform bank whose hot subset stays warm in L3).
     - **`(C1) d8_compact`** (`502`-byte binomial mask, zero `pshufb` LUT) **regresses to `0.56×–0.76×` across every regime**, proving `D_8^+` unranking is only fast when paired with the `764 KB` `pshufb` table.

---

## 2. Law-2b Steelman & Honest CPU Floor Verdict (`N1`)

### Strongest Case for Adopting `(A) guide_a`
- **Zero Bitstream / Encoder Compatibility Risk:** `(A) guide_a` decodes the **exact same BallPack `8->4->2` rank ordering** (`z_output_sha256` bit-identical to `AUDIT-115` on all `8,388,608` weights!), requires zero encoder or checkpoint changes, eliminates every binary-search loop in Stage 1 via a single 1-cycle bit shift (`b = rem >> sh`), and stays compact (`93.5 KB` for `88b`, `1.60 MB` for `136b`).
- **Warp-Friendly GPU/TPU Blueprint:** Replacing variable-iteration binary search with fixed `1`-shift `O(1)` bucket lookup + `~0.6–2.0` expected sequential steps directly eliminates branch divergence across GPU warp lanes while preserving exact BallPack bit-rates (`3.000` bpw `88b`, `4.500` bpw `136b`).

### Honest CPU Reality Check (What Remains Falsified on x86-64 CPU)
- Even after cutting `88b` Stage-1 from `22.74` down to **`12.43` nom. cyc/w** (`1.83×` faster) and `136b` Stage-1 from `41.21–49.52` down to **`25.52–32.35` nom. cyc/w** (`1.33×–1.72×` faster), **Stage 0 (`10–15` cyc/w) + optimized Stage 1 (`12.4–32.4` cyc/w) now establish a hard x86-64 CPU scalar-ALU floor**:
  - `EBG32:b88s8` fastest `1t` Adjoint GEMV (`guide_a`): **`92.818 M w/s/c` (`gcc`, `53.27×` slower than matched Q4)** / **`87.093 M w/s/c` (`clang`, `52.77×` slower)**.
  - `EBG32:b136s8` fastest `1t` Adjoint GEMV: **`34.991 M w/s/c` (`gcc` `guide_a`, `141.30×` slower)** / **`44.514 M w/s/c` (`clang` `d8_pshufb`, `103.24×` slower at `19.4 MB`)** or **`38.226 M w/s/c` (`clang` `guide_a`, `120.22×` slower at `1.60 MB`)**.
- CPU-side BallPack GEMV remains **`~53×` (`88b`) to `~103×–141×` (`136b`) slower** than matched `4.500`-bpw `Q4_K_M`-style AVX2/FMA GEMV. Practical deployment of `EBG32` belongs on **HBM-bandwidth-bound GPUs/TPUs**, not x86-64 CPUs.

---

## 3. Pure-$\mathbb{Z}/\mathbb{Q}$ Exact Proof Summary (`AUDIT-116`, Pillars 1–3)

Verified independently by `e8_stage1_oracle.py`, `e8_stage1_audit.py`, and `test_e8_stage1_r5.py` (pure Python 3 stdlib `int`/`Fraction`, zero floats, zero bare `assert`s, strict fail-closed `type(x) is int` & `type(x) in (tuple, list)` APIs, `hmac.compare_digest` on `payload_sha256`, iterative depth cap `max_depth=64`):

- **Certificate File:** `e8_stage1_certificate.json` (`schema_version: audit116.e8_stage1.v1`, `overall_verified: true`)
- **Inner `payload_sha256`:** `55098d900544b0dd03d4ca292a4f6f82f4a887dc4f6a0cf6d9e19537eab06be7`
- **4-Mode `-I` Parity (`-I -B`, `-I -O -B`, `-I -O -B -W error`, `PYTHONHASHSEED=42 -I -B`):** Bit-identical stdout SHA-256 `093d8f4eb8f1ecf6af266a4b6f356685e01aa43f8c744aaed6a257dd77491066`; `--check e8_stage1_certificate.json` returns `valid: true`.
- **Mutation & AST `BoolOp` Score (`aletheia v2.0.0 mutate --max-mutants 0`):** `13/13` (`100.0%`, `6/6` `BoolOp`) on `e8_stage1_oracle.py` + `21/21` (`100.0%`, `1/1` `BoolOp`) on `e8_stage1_audit.py` = **`34/34` (`100.0%`) KILLED + `7/7` (`100.0%`) AST `BoolOp` KILLED**, `0` survived, `truncated=False`.

| Pillar | Exact Property Verified | Pure-$\mathbb{Z}/\mathbb{Q}$ Receipt |
| :--- | :--- | :--- |
| **P1: `88b` (`m_max=48`) Widths** | $\max S_1(m)$, $W_1(48)$, $\max c_4$, $\max c_2$ | `max S1 = 31,456,320` (`25`b at `m=48`), `W1 = 360,605,761` (`29`b -> **pure `uint32_t`**), `max c4 = 1,872` (`uint16_t`), `max c2 = 16` (`uint8_t`) |
| **P1: `136b` (`m_max=391`) Widths** | $\max S_1(m)$, $W_1(391)$, $\max c_4$, $\max c_2$ | `max S1 = 16,749,815,040` (`34`b at `m=390`), `W1 = 1,525,056,448,801` (`41`b -> **pure `uint64_t`, 0 `u128`**), `max c4 = 14,976` (`uint16_t`), `max c2 = 24` (`uint8_t`) |
| **P2: `O(1)` Top-Bit Guide** | `b = rem >> sh < B`, `guide_cdf[b] <= b << sh` | Verified across all `49/49` `88b` shells; expected extra steps: **`0.6116`** (`88b m=12 B=32`, max span `4`), **`4.3596`** (`136b m=97 B=32`), **`1.9711`** (`136b m=97 B=64`, max span `18`) |
| **P2: Exact Uniform LUT Hit Bound** | $\Pr[m_i \le 2]$, $\Pr[m_i \le 3]$ over exact $E_8^4$ ball | `88b`: **`0.756558%`** (`m<=2`, `19,208` B) / **`2.341009%`** (`m<=3`, `72,968` B); `136b`: **`0.000270%`** (`m<=2`) / **`0.001001%`** (`m<=3`) |
| **P3: `D_8^+` Bijection (`m=0..3`)** | Exhaustive orbit unrank vs $2E_8^{\mathrm{std}}$ | **`9,121` / `9,121`** vectors exact bijection (`0` duplicates, `0` invalid); universal `p(8) = 22` shapes = **`95,503`** perms (`764,024` B `pshufb`), **`502` B** binomial-mask table |
| **P3: `D_8^+` Leader Phase Wall** | Exact orbit sum $\sum |\mathcal{O}(\lambda)| = 240\sigma_3(m)$ | `88b`: **`2,000`** leaders (`955` ev + `1,045` od, max `111` at `m=47`, `32,000` B L1d); `136b`: **`1,033,160`** leaders (`409,367` ev + `623,793` od, max `8,749` at `m=389`, `16.53 MB`) -> **`516.58×` blowup** |

---

## 4. Real-Hardware C11 + AVX2/FMA Head-to-Head Silicon Results (`4096 × 1024`)

### Build, Disassembly, Exactness & Host Disclosure
- **Source & Receipt:** `c-bench/e8_stage1_bench.c` (`source_sha256: 4c56c906ad90218c947d211459b2036c0cd43c3390cb5571529027e6492f68e2`) + `c-bench/run_e8_stage1_bench.py` + `c-bench/stage1_bench_results.json`
- **Compilers & Zero-Div Proof:** Both `gcc (GCC) 16.2.1 20260810` and `clang version 23.1.3` (`-O3 -march=native -Wall -Wextra -Werror -std=c11 -shared -fPIC`) build with **`0` warnings** and **`0` `div`/`idiv` across `13/13` hot exported symbols**.
- **Exactness (`4,194,304` weights = `131,072` blocks = `524,288` leaves per regime):**
  - Variants `0..3` (`baseline`, `guide_a`, `eytz_lut2`, `eytz_lut3`) are bit-identical across all `4` variants and both compilers AND match `AUDIT-115` verbatim (`136b`: `4863b413274293907e2b4792e220a25a0653da58d43814aeca038e48a84a3532`; `88b`: `644aba45ccdb3a4ed850b6bc3b6761ae4393f2d95efad33871433aaaadabde2b`).
  - Variants `4..5` (`d8_compact`, `d8_pshufb`) are bit-identical to each other across both compilers (`136b`: `43f35f9d...`; `88b`: `48f1026e...`), pass exhaustive `9,121`-vector C11 `m=0..3` bijection, and match materialized vs adjoint GEMV (`< 1e-3`) across all `4,096` rows.
- **Host & Contention Disclosure:** `AMD Athlon Silver 3050U` (`x86_64` Zen 1, `2` logical CPUs, `AVX2/FMA/BMI1/BMI2/ADX/F16C`, `L1d=32 KiB/core`, `L2=512 KiB/core`, `L3=4 MiB`, nominal `2300 MHz`, observed boost `3199 MHz`, disclosed host `loadavg = 9.98 9.42 9.31 8/653 578381`). Primary metric is `CLOCK_THREAD_CPUTIME_ID` (`5` trials + `1` warmup) with matched `4.500`-bpw `Q4_K_M`-style AVX2/FMA GEMV measured in the **same run** (`gcc`: `4,944.300` `[4,866.177..5,037.834]` M w/s/c, `0.47` cyc/w, `2.7812 GB/s`; `clang`: `4,595.687` `[4,135.905..4,819.311]` M w/s/c, `0.50` cyc/w, `2.5851 GB/s`).

### Table A: Empirical Low-Norm Shell LUT Hit Rates Across 3 Distributions (`524,288` Leaves Each)

| Regime | Distribution (`524,288` leaves) | Mean Shell $\bar{m}$ | `m <= 2` Hits (`19.2 KB` L1d) | `m <= 3` Hits (`73.0 KB` L2) |
| :--- | :--- | :---: | :---: | :---: |
| **`EBG32:b88s8`** | Uniform `E_8^4` ball (`AUDIT-115` stream) | — | `4,101` (`0.7822%`) | `12,675` (`2.4176%`) |
| **`EBG32:b88s8`** | Per-block-scaled Isotropic Gaussian $\mathcal{N}(0,I_{32})$ | `10.83` | `4,992` (`0.9521%`) | `14,592` (`2.7832%`) |
| **`EBG32:b88s8`** | Per-block-scaled Outlier Gaussian (`3%` `6×`) | `10.83` | `45,440` (`8.6670%`) | `75,520` (`14.4043%`) |
| **`EBG32:b136s8`** | Uniform `E_8^4` ball (`AUDIT-115` stream) | — | `7` (`0.0013%`) | `13` (`0.0025%`) |
| **`EBG32:b136s8`** | Per-block-scaled Isotropic Gaussian $\mathcal{N}(0,I_{32})$ | `86.66` | **`0` (`0.0000%`)** | **`0` (`0.0000%`)** |
| **`EBG32:b136s8`** | Per-block-scaled Outlier Gaussian (`3%` `6×`) | `86.69` | `320` (`0.0610%`) | `1,088` (`0.2075%`) |

### Table B: `EBG32:b88s8` (`3.000` bpw, `m_max = 48`) — Isolated Stage-1 & End-to-End Adjoint GEMV (`4096 × 1024`)

| Variant | Table Bytes | Stage-1 Unif `gcc` (M w/s/c `[min..max]` / cyc / sp) | Stage-1 Unif `clang` (M w/s/c / cyc / sp) | Stage-1 Gauss-Iso / Out (`gcc` \| `clang`) | GEMV `1t` `gcc` (M w/s/c `[min..max]` / cyc / sp / vs Q4) | GEMV `1t` `clang` (M w/s/c / cyc / sp / vs Q4) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `baseline` | `5,490` B | `101.147` `[100.654..101.160]` / `22.74` / `1.00×` | `99.733` / `23.06` / `1.00×` | `1.00×`/`1.00×` \| `1.00×`/`1.00×` | `66.798` `[64.616..66.971]` / `34.43` / `1.00×` / `74.02×` | `62.277` / `36.93` / `1.00×` / `73.79×` |
| **`(A) guide_a`** | **`93,472` B** | **`184.976`** `[183.310..186.058]` / **`12.43`** / **`1.83×`** | `177.595` / `12.95` / **`1.78×`** | **`1.82×`/`1.77×`** \| **`1.82×`/`1.76×`** | **`92.818`** `[92.148..93.851]` / **`24.78`** / **`1.39×`** / **`53.27×`** | **`87.093`** / **`26.41`** / **`1.40×`** / **`52.77×`** |
| `(B) eytz_lut2` | `160,480` B | `103.508` `[100.742..104.398]` / `22.22` / `1.02×` | `105.555` / `21.79` / `1.06×` | `1.03×`/`1.04×` \| `1.09×`/`1.05×` | `64.654` `[63.715..65.828]` / `35.57` / `0.97×` / `76.47×` | `62.601` / `36.74` / `1.01×` / `73.41×` |
| `(B) eytz_lut3` | `214,240` B | `102.232` `[101.504..103.269]` / `22.50` / `1.01×` | `104.859` / `21.93` / `1.05×` | `1.01×`/`1.07×` \| `1.02×`/`1.03×` | `64.124` `[63.633..64.726]` / `35.87` / `0.96×` / `77.11×` | `63.829` / `36.03` / `1.02×` / `72.00×` |
| `(C1) d8_compact` | `42,774` B | `63.601` `[62.756..63.785]` / `36.16` / `0.63×` | `56.275` / `40.87` / `0.56×` | — | `46.940` `[46.373..47.474]` / `49.00` / `0.70×` / `105.33×` | `42.698` / `53.87` / `0.69×` / `107.63×` |
| `(C2) d8_pshufb` | `806,296` B | `172.979` `[170.213..178.049]` / `13.30` / `1.71×` | **`179.213`** / **`12.83`** / **`1.80×`** | `1.69×`/`1.55×` \| `1.67×`/`1.54×` | `90.749` `[87.060..92.973]` / `25.34` / `1.36×` / `54.48×` | `81.262` / `28.30` / `1.30×` / `56.55×` |

*(Fastest `2t` `88b` GEMV = `guide_a`: `gcc` `96.185` `[94.541..96.668]` CPU/c, `23.91` cyc/w, wall `76.326` / `91.705` M w/s; `clang` `92.242` `[91.035..94.028]` CPU/c, `24.93` cyc/w, wall `37.893` / `53.189` M w/s.)*

### Table C: `EBG32:b136s8` (`4.500` bpw, `m_max = 391`) — Isolated Stage-1 & End-to-End Adjoint GEMV (`4096 × 1024`)

| Variant | Table Bytes | Stage-1 Unif `gcc` (M w/s/c `[min..max]` / cyc / sp) | Stage-1 Unif `clang` (M w/s/c / cyc / sp) | Stage-1 Gauss-Iso / Out (`gcc` \| `clang`) | GEMV `1t` `gcc` (M w/s/c `[min..max]` / cyc / sp / vs Q4) | GEMV `1t` `clang` (M w/s/c / cyc / sp / vs Q4) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `baseline` | `685,666` B | `55.818` `[53.026..56.794]` / `41.21` / `1.00×` | `46.449` / `49.52` / `1.00×` | `1.00×`/`1.00×` \| `1.00×`/`1.00×` | `32.366` `[32.172..33.424]` / `71.06` / `1.00×` / `152.76×` | `33.208` / `69.26` / `1.00×` / `138.39×` |
| **`(A) guide_a`** | **`1.60 MB`** | **`73.996`** `[51.517..82.051]` / **`31.08`** / **`1.33×`** | `71.097` / `32.35` / **`1.53×`** | `1.28×`/`1.34×` \| **`1.72×`/`1.55×`** | **`34.991`** `[32.502..38.740]` / **`65.73`** / **`1.08×`** / **`141.30×`** | `38.226` / `60.17` / **`1.15×`** / `120.22×` |
| `(B) eytz_lut2` | `1.41 MB` | `42.974` `[41.331..45.392]` / `53.52` / `0.77×` | `44.086` / `52.17` / `0.95×` | `0.77×`/`0.77×` \| `0.95×`/`0.80×` | `28.667` `[26.679..29.294]` / `80.23` / `0.89×` / `172.47×` | `31.280` / `73.53` / `0.94×` / `146.92×` |
| `(B) eytz_lut3` | `1.47 MB` | `42.440` `[41.428..45.090]` / `54.19` / `0.76×` | `38.527` / `59.70` / `0.83×` | `0.78×`/`0.77×` \| `0.81×`/`0.79×` | `28.055` `[22.595..28.684]` / `81.98` / `0.87×` / `176.23×` | `30.545` / `75.30` / `0.92×` / `150.46×` |
| `(C1) d8_compact` | `18.65 MB` | `32.650` `[31.069..38.885]` / `70.44` / `0.58×` | `30.038` / `76.57` / `0.65×` | — | `21.645` `[20.316..25.179]` / `106.26` / `0.67×` / `228.43×` | `25.309` / `90.88` / `0.76×` / `181.59×` |
| `(C2) d8_pshufb` | **`19.41 MB`** | `63.214` `[56.830..76.243]` / `36.38` / `1.13×` | **`78.150`** / **`29.43`** / **`1.68×`** | `1.29×`/`1.08×` \| `1.41×`/`1.20×` | `33.539` `[28.912..35.853]` / `68.58` / `1.04×` / `147.42×` | **`44.514`** / **`51.67`** / **`1.34×`** / **`103.24×`** |

*(Fastest `2t` `136b` GEMV: `gcc` `guide_a` `44.581` `[39.105..48.251]` CPU/c, `51.59` cyc/w, wall `12.399` / `34.382`; `clang` `d8_pshufb` `52.375` `[47.626..53.746]` CPU/c, `43.91` cyc/w, wall `50.347` / `81.884` M w/s.)*

---

## 5. Reproducibility & File Manifest (`track-s15d-e8-leaf-stage1-attack/`)

All commands run offline with pure Python 3 stdlib (`-I -B`, zero checked-in binaries, zero `__pycache__`):

```bash
# 1. Regenerate & verify pure-Z/Q certificate (4-mode -I deterministic)
python3 -I -B e8_stage1_audit.py --standalone --json
python3 -I -B e8_stage1_audit.py --check e8_stage1_certificate.json

# 2. Run assertion-free R5 test suite
python3 -I -O -B -W error test_e8_stage1_r5.py

# 3. Run deterministic 2-compiler (GCC + Clang) 0-div & 6-variant exactness check
python3 -I -B c-bench/run_e8_stage1_bench.py --json

# 4. Run full 4096x1024 head-to-head benchmark across Uniform / Gaussian-Iso / Gaussian-Outlier
python3 -I -B c-bench/run_e8_stage1_bench.py --bench --write-results c-bench/stage1_bench_results.json
```

| File | Role |
| :--- | :--- |
| `e8_stage1_oracle.py` | Pure-$\mathbb{Z}/\mathbb{Q}$ oracle: bit-widths, `O(1)` bucket guide theorem, exact uniform LUT bounds, `D_8^+` bijection & DP leader count (`13/13` mutate, `6/6` `BoolOp`) |
| `e8_stage1_audit.py` | Standalone + fail-closed Zone-B `--check` CLI (`hmac.compare_digest`, iterative `max_depth=64`, strict types; `21/21` mutate, `1/1` `BoolOp`) |
| `test_e8_stage1_r5.py` | Assertion-free R5 test suite covering all 3 pillars & all `D1`–`D3` fail-closed guards |
| `e8_stage1_certificate.json` | Canonical pinned pure-$\mathbb{Z}/\mathbb{Q}$ receipt (`payload_sha256: 55098d90...`) |
| `c-bench/e8_stage1_bench.c` | Zero-warning C11 + AVX2/FMA implementation of all 6 Stage-1 variants + matched Q4 baseline (`0` `div`/`idiv` across `13/13` hot kernels) |
| `c-bench/run_e8_stage1_bench.py` | Ephemeral `0700` dual-compiler build, `objdump` `0`-div checker, 6-variant exactness proof, & 3-distribution silicon harness |
| `c-bench/stage1_bench_results.json` | Complete `4096 × 1024` empirical silicon receipt (`GCC 16.2.1` & `Clang 23.1.3`) |
| `E8_STAGE1_AUDIT_REPORT.md` | This report |
