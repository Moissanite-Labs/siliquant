# AUDIT-115 (`Track S15c`): Exact `E_8_Had` Zero-Table `WHT_8` Isometry Proof & Real-Hardware C11 + AVX2/FMA Batched SIMD / Fused-GEMV Audit for BallPack `EBG32`

**Audit ID:** `AUDIT-115` (`Track S15c`)  
**Target Upstream Context:** `Moissanite-Labs/siliquant#23` (BallPack tree-structured `E_8^4` indexing), `Moissanite-Labs/siliquant#24` (`closest_E8_had`), and `Moissanite-Labs/siliquant#25` (`EBG32:b136s8` / `EBG32:b88s8` with `E_8_Had`; note that the committed Qwen2.5 `0.020817` KL measurement is on standard `E_8` `EBG32:b136s8`, while `#25` explicitly notes the real-weight `E_8_Had` quality experiment has not yet run)  
**Predecessor Baseline:** `AUDIT-112R` (`Track S15` + `Track S15b`, live at `Moissanite-Labs/siliquant#35`)  
**Novelty Classification:** `N1` (exact algebraic-isometry + activation-adjoint GEMV theorem proof combined with measured x86-64 AVX2/FMA fused-GEMV on this implementation and host against a byte-matched `4.500`-bpw custom symmetric nibble proxy baseline, plus a 4-stage isolated-median cycle breakdown for `E_8_Had` BallPack; no claim of CPU superiority over `llama.cpp` `Q4_K_M` and no end-to-end `E_8_Had` model-quality/KL claim)  
**Overall Verification Status:** `PASS` — **Zero algebraic defects, zero hardware `div`/`idiv` across `8/8` hot kernels, zero mismatches across `8,388,608` decoded weights (`4,194,304` per regime), `41/41` (`100.0%`) mutation score, honest CPU compute-bound verdict on this implementation and host (`98×–220×` slower than the byte-matched `4.500`-bpw custom symmetric nibble AVX2/FMA proxy on Zen-1 CPU; not a production `llama.cpp` `Q4_K_M` latency measurement)**

---

## 1. OWN_TERMS_FIRST — Upstream Claims & Concrete Questions Tested

We evaluate the exact engineering and mathematical questions posed by `siliquant#23`, `#24`, and `#25` on their own terms before applying any adversarial stress test:

1. **Zero-Table `E_8_Had` Indexing (`siliquant#25`):** Can BallPack's existing `E_8^4` tree-unranking tables (`56,850` B compact for `EBG32:b136s8` at `m_max = 391`; `5,490` B L1d-resident for `EBG32:b88s8` at `m_max = 48`) index the Hadamard-rotated `E_8_Had` lattice (`closest_E8_had` in `#24`, proposed for `#25`) with **zero extra lookup-table bytes**, exact bit-invariance across all shells, and a `0`-multiply integer Walsh-Hadamard leaf transform (`WHT_8`)?
   - **Verdict: PROVEN EXACT (`PASS`) — ALGEBRAIC ISOMETRY ONLY (NO MODEL-KL CLAIM).** In exact $\mathbb{Z}/\mathbb{Q}$ arithmetic, $z = (H_8 u)/4$ (`24` add/sub ops, `0` multiplies) maps $2 E_8^{\mathrm{std}}$ bijectively onto Construction-A $\Lambda_{\mathrm{CA}} = 2\mathbb{Z}^8 + \mathcal{H}_8 = \sqrt{2}\, E_8^{\mathrm{Had}}$, where $\mathcal{H}_8 = \mathrm{RM}(1,3)$ is the self-dual `[8,4,4]` extended Hamming code (`1` weight-0, `14` weight-4, `1` weight-8, $d_{\min} = 4$). No coordinate permutation beyond `#24`'s Sylvester–Hadamard $U_{H_8} = H_8/\sqrt{8}$ is needed, and **`0` extra table bytes** are required. An algebraic isometry proof alone does **not** establish end-to-end `E_8_Had` model KL, which requires a dedicated real-weight evaluation run.
2. **Adjoint `WHT_8` Elimination in Fused GEMV:** Can the linear self-adjointness of $H_8$ eliminate per-weight `WHT_8` inside a fused dequant-GEMV kernel by pre-rotating the activation vector $x \mapsto \tilde{x} = (H_8/\sqrt{8})x$ once per GEMV?
   - **Verdict: PROVEN EXACT (`PASS`) — WITH HONEST CPU AMDAHL BOUND.** Exact identity $4\langle z, x \rangle = \langle u, H_8 x \rangle$ holds identically across all `960/960` tested `(u, x)` pairs and reduces `WHT_8` ops/weight from `3` to `3/4096 = 0.000732` (`4,096×` fewer WHT ops on `4096 × 1024`). However, on x86-64 Zen-1 CPU, materialized AVX2 `WHT_8` accounts for only **`1.65%–3.16%`** of the sum of the four isolated stage medians (`1.11–1.26` nominal cycles/weight), so eliminating it yields at most a `~2%–3%` end-to-end CPU speedup on this implementation and host (measured `+3.0%` on `gcc` `136b`, within boost variance on `clang`).
3. **Real-Hardware SIMD / Fused-GEMV Boundary vs. Byte-Matched Symmetric Q4 Proxy:** Once division-free C11 (`0` `div`/`idiv`), AVX2 `WHT_8`, batched SIMD8 unranking, and fused AVX2/FMA GEMV are implemented and benchmarked on real `4096 × 1024` (`4,194,304`-weight, `131,072`-block) matrices under both `GCC` and `Clang`, how close does `EBG32` come to a byte-matched `4.500`-bpw custom symmetric nibble AVX2/FMA proxy baseline (not `llama.cpp` `Q4_K_M`), and how do the four isolated stage medians divide the cycles?
   - **Verdict: DECISIVELY COMPUTE-BOUND ON THIS CPU IMPLEMENTATION AND HOST (`98.12×–219.80×` SLOWER THAN THE BYTE-MATCHED SYMMETRIC Q4 PROXY); STAGE 1 INNER `E_8` LEAF UNRANKING REPRESENTS `67.99%–75.14%` OF THE FOUR ISOLATED STAGE MEDIANS.**

---

## 2. Law-2b Steelman & Honest `N0`/`N1` Scope

### Strongest Case for `E_8_Had` BallPack (`EBG32`)
- **Compression Footprint (Quality Experiment Still Pending for `E_8_Had`):** `EBG32:b136s8` occupies `(136 + 8)/32 = 9/2 = 4.500` bpw (`2,359,296` B for `4096 × 1024`), which is `6.7%` smaller (`16/15 = 1.0667×` fewer weight bytes) than `4.80–4.82`-bpw `Q4_K_M`. `EBG32:b88s8` occupies `(88 + 8)/32 = 3.000` bpw (`1,572,864` B), `1.6000×` smaller than `4.80`-bpw `Q4_K_M` and `5.3333×` smaller than `FP16`. *(Quality boundary: the committed Qwen2.5 `0.020817` KL result in `siliquant` applies to standard `E_8` `EBG32:b136s8`, not `E_8_Had`; Issue `#25` notes the real-weight `E_8_Had` quality experiment has not yet been run, and algebraic isometry alone does not establish end-to-end model KL.)*
- **Zero-Cost Algebraic Drop-In:** Replacing `E_8^std` with `E_8_Had` adds `0` lookup-table bytes, `0` multiplies in decode (`24` integer add/sub ops per `E_8` leaf), `0` extra quantization bits, and `0` per-weight WHT ops when folded into the activation via $H_8 x$.
- **Accelerator Roofline Distinction (Out-of-Scope on This Host):** There are **no GPU or TPU measurements** in this audit. On a bandwidth-bound GPU/TPU (`1–3 TB/s` HBM, hundreds of ALUs per memory byte), `40–66` arithmetic ops/weight could potentially be hidden behind HBM transfers if Stage 1 inner `E_8` unranking is recast as a warp-parallel SRAM lookup / register-shuffle kernel. Our Zen-1 CPU results do **not** falsify GPU/TPU viability; they establish the measured x86-64 AVX2/FMA performance bound **for this implementation on the disclosed host** and identify Stage 1 (`67.99%–75.14%` share of the four isolated stage medians) as the dominant cost center.

### What This Audit Falsifies / Bounds on This CPU Implementation and Host
- On this x86-64 Zen-1 host (`AMD Athlon Silver 3050U`, AVX2/FMA/BMI2/ADX, `2` cores) with this C11 implementation, **neither** AVX2 `WHT_8`, **nor** 8-leaf batched `simd8_avx2` unranking, **nor** activation-adjoint WHT elimination, **nor** `5.49 KB` L1d-resident `88b` tables makes `EBG32` competitive with the byte-matched `4.500`-bpw custom symmetric nibble AVX2/FMA proxy (`q4km_style_gemv_avx2`, a custom symmetric nibble layout rather than `llama.cpp` `Q4_K_M`, from which production `Q4_K_M` latency should not be inferred): `EBG32:b136s8` remains **`177.55×` (`gcc`) to `219.80×` (`clang`) slower** (`0.0196` / `0.0167 GB/s` effective weight bandwidth vs. `3.4752` / `3.6790 GB/s`), and `EBG32:b88s8` remains **`98.12×` (`gcc`) to `104.12×` (`clang`) slower**.
- **Negative SIMD Finding (Disclosed Directly):** `e8had_decode_simd8_avx2` (buffering 2 blocks = `8` `E_8` leaves for transposed 256-bit AVX2 `WHT_8`) is **`-6.8%` to `-20.7%` slower** than single-block `e8had_decode_wht_avx2` on Zen 1 because Zen 1 double-pumps 256-bit AVX2 over 128-bit execution units and the 8-lane pack/unpack/transpose adds register pressure without vectorizing the data-dependent Stage 0/1 tree searches.

---

## 3. Pure-$\mathbb{Z}/\mathbb{Q}$ Exact Proof Summary (Pillars 1–3)

Verified independently with zero shared code from `siliquant#24` or `track-s15` by `e8had_isometry_oracle.py`, `e8had_isometry_audit.py`, and `test_e8had_isometry_r5.py` (pure Python 3 stdlib `int`/`Fraction`, `0` floats, `0` bare `assert`s).

- **Certificate File:** `e8had_isometry_certificate.json` (`schema_version: audit115.e8had_isometry.v1`, `overall_verified: true`)
- **Inner Certificate SHA-256:** `5b56e0cd950e292d8eaee68641198cd5bfd11774921ca6ecfdf0bbf60d1c2b4a`
- **4-Mode `-I` Parity (`-I -B`, `-I -O -B`, `-I -O -B -W error`, `PYTHONHASHSEED=42 -I -B`):** Bit-identical stdout SHA-256 `3344fc066debe71c375888b126425f36e6bc3a154d2ef9e452888145ece7b8da`; `--check e8had_isometry_certificate.json` returns `valid: true`.
- **Mutation Testing (`aletheia v2.0.0 mutate --max-mutants 0`):** `22/22` (`100.0%`) on `e8had_isometry_oracle.py` + `19/19` (`100.0%`) on `e8had_isometry_audit.py` = **`41/41` (`100.0%`) KILLED**, `0` survived, `truncated=False`.

| Pillar | Mathematical / Structural Property Verified | Exact Pure-$\mathbb{Z}/\mathbb{Q}$ Result |
| :--- | :--- | :--- |
| **P1: `H_8` & `WHT_8`** | Sylvester $H_8 = H_2^{\otimes 3}$ symmetry, orthogonality, butterfly equivalence | $H_8 = H_8^T$, $H_8^2 = 8 I_8$; `WHT_8` matches explicit matvec (`24` add/sub ops, `0` multiplies) |
| **P1: `[8,4,4]` Hamming Code** | $\mathcal{H}_8 = \mathrm{RM}(1,3)$ codeword count, weight enumerator, min distance | `16` codewords: `{wt 0: 1, wt 4: 14, wt 8: 1}`, $d_{\min} = 4$, self-dual |
| **P1: Exact Bijection (`m = 0..3`)** | $u \in 2E_8^{\mathrm{std}} \leftrightarrow z = (H_8 u)/4 \in \Lambda_{\mathrm{CA}} = 2\mathbb{Z}^8 + \mathcal{H}_8$, inverse $u = (H_8 z)/2$ | **`9,121` / `9,121` vectors (`9,120` nonzero)** exact bijection & norm isometry $\|z\|^2 = 4m = \|u\|^2/2$ |
| **P1: Shell-by-Shell Counts** | Exact theta series $240\sigma_3(m)$ match for `m = 0, 1, 2, 3` | `m=0: 1`, `m=1: 240` (`16` axis $(\pm 2,0^7)$ + `224` weight-4 $(\pm 1)^4 0^4$), `m=2: 2,160`, `m=3: 6,720` |
| **P2: DP Table Bit-Invariance** | Exact shell counts `m = 0..391` (`392` shells) | `EBG32:b136s8` (`m_max=391`): `W = 85889107931845736051215402649708868585601` (`56,850` B, **`0` extra bytes**); `EBG32:b88s8` (`m_max=48`): `W = 263197133594409826368003841` (`5,490` B, **`0` extra bytes**) |
| **P2: Adjoint GEMV Identity** | $4\langle z, x \rangle = \langle u, H_8 x \rangle$ across `240 × 4 = 960` pairs | `960/960` exact; reduces WHT ops/weight from `3` to `3/4096 = 0.000732` (`4,096×` reduction at `M = 4096`) |
| **P3: `#24` Correspondence** | `closest_E8_had(x) = closest_E8_std(x @ U_H8) @ U_H8`, $U_{H_8} = H_8/\sqrt{8}$ | Exact with `pr24_extra_permutation_needed: false`, integer scale factor squared `= 2` ($E_8^{\mathrm{Had}} = \Lambda_{\mathrm{CA}}/\sqrt{2}$) |

---

## 4. Real-Hardware C11 + AVX2/FMA Implementation & Exactness Proof

- **C11 Kernel Source:** `c-bench/e8had_simd_bench.c` (`source_sha256: d4a7a415a2ae9b5bb91e17c27f64af69f4c497b1245c390b72fd8f5af30066c6`)
- **Harness & Receipt:** `c-bench/run_e8had_simd_bench.py` + `c-bench/simd_bench_results.json` (zero checked-in `.so`/`.o` binaries; shared libraries built ephemerally in a `0700` `TemporaryDirectory`)
- **Compiler Builds (Both `0` Warnings with `-O3 -march=native -Wall -Wextra -Werror -std=c11 -shared -fPIC`):**
  - `gcc_native`: `gcc (GCC) 16.2.1 20260810`, `avx2_enabled_in_binary: true`, **`0` `div`/`idiv` across `8/8` hot symbols** (`e8_decode_std_scalar`, `e8had_decode_wht_scalar`, `e8had_decode_wht_avx2`, `e8had_decode_simd8_avx2`, `e8had_gemv_materialized_avx2`, `e8had_gemv_adjoint_avx2`, `e8had_gemv_adjoint_batch4_avx2`, `q4km_style_gemv_avx2`).
  - `clang_native`: `clang version 23.1.3`, `avx2_enabled_in_binary: true`, **`0` `div`/`idiv` across `8/8` hot symbols**.
- **Full-Matrix Exactness (`4096 × 1024 = 4,194,304` weights = `131,072` blocks = `524,288` `E_8` subblocks per regime, `8,388,608` decoded weights total):**
  - `ebg32_b136s8`: `0` lattice mismatches, `wht_scalar_avx2_simd8_identical: true`, cross-compiler bit-identical `z_output_sha256 = 4863b413274293907e2b4792e220a25a0653da58d43814aeca038e48a84a3532`, `128/128` blocks (`4,096` weights) bit-identical to pure-Python $\mathbb{Z}$ oracle, all `4,096` GEMV rows exact (`gemv_materialized_vs_adjoint_within_1e3: true`, `gemv_adjoint_vs_batch4_within_1e4: true`).
  - `ebg32_b88s8`: `0` lattice mismatches, `wht_scalar_avx2_simd8_identical: true`, cross-compiler bit-identical `z_output_sha256 = 644aba45ccdb3a4ed850b6bc3b6761ae4393f2d95efad33871433aaaadabde2b`, `128/128` blocks (`4,096` weights) bit-identical to pure-Python $\mathbb{Z}$ oracle, all `4,096` GEMV rows exact.

---

## 5. Empirical Silicon Benchmark (`4096 × 1024`, Dual-Clock, `5` Trials + `1` Warmup)

### Host Hardware & Contention Disclosure
- **CPU:** `AMD Athlon Silver 3050U with Radeon Graphics` (`x86_64`, Zen 1, `2` logical CPUs, `AVX2/FMA/BMI1/BMI2/ADX/F16C`, no AVX-512)
- **Cache Hierarchy (`lscpu`):** `L1d: 64 KiB (2 instances)` (`32 KiB/core`), `L2: 1 MiB (2 instances)` (`512 KiB/core`), `L3: 4 MiB (1 instance)`
- **Clocks:** Nominal `2300 MHz` (used for `cycles/weight` column), observed boost `2999 MHz` (turbo frequency scaling accounts for trial-to-trial CPU-time spreads up to `~10–11%` on selected kernels; exact `[min..max]` ranges are quoted for every kernel below)
- **Disclosed Host Load During Benchmark:** `loadavg = 7.81 9.63 9.84 9/621 537991` (multi-process background contention). Accordingly, we report **`CLOCK_THREAD_CPUTIME_ID`** (on-CPU per-core throughput, immune to OS preemption) as the primary architectural metric alongside **`CLOCK_MONOTONIC`** (wall-clock throughput, which reflects scheduler preemption on long `65–140 ms` loops while the `0.65 ms` Q4 proxy kernel completes within a single timeslice).
- **Baseline Proxy & Summary-Receipt Scope:** `q4km_style_gemv_avx2` is a custom symmetric nibble layout matching `EBG32:b136s8`'s `4.500`-bpw footprint (not `llama.cpp` `Q4_K_M`; do not infer production `llama.cpp` `Q4_K_M` latency from it). `c-bench/simd_bench_results.json` records per-kernel medians, extrema (`[min..max]`), and derived ratios rather than raw per-trial sample arrays, so arithmetic readback of stored summaries and ratios is directly reproducible, while independent median or outlier recomputation requires re-running `run_e8had_simd_bench.py --bench`.

### Table A: `gcc_native` (`GCC 16.2.1 -O3 -march=native`) — `4,194,304` Weights (`131,072` Blocks)

| Kernel | Reg. / Bpw | Thr | CPU/Core Med (`[min..max]`, M w/s) | Nom. Cyc/w | Eff. BW CPU (GB/s) | Wall Med / Max (M w/s) | Slowdown vs Byte-Matched Q4 Proxy (`1t` CPU) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `q4km_style_gemv_avx2_1t` | Q4 proxy / `4.500` | 1 | **`6,178.179`** `[6,049.855..6,202.289]` | `0.37` | `3.4752` | `6,084.980` / `6,173.550` | **`1.00×`** (proxy baseline) |
| `q4km_style_gemv_avx2_2t` | Q4 proxy / `4.500` | 2 | `6,132.830` `[6,079.662..6,153.615]` | `0.38` | `3.4497` | `5,472.535` / `5,562.376` | `1.01×` |
| `e8_136b_decode_std_scalar_1t` | `136b` / `4.500` | 1 | `34.758` `[34.430..38.247]` | `66.17` | `0.0196` | `13.570` / `26.030` | `177.75×` |
| `e8had_136b_decode_wht_scalar_1t` | `136b` / `4.500` | 1 | `35.878` `[32.813..36.600]` | `64.11` | `0.0202` | `11.161` / `16.701` | `172.20×` |
| `e8had_136b_decode_wht_avx2_1t` | `136b` / `4.500` | 1 | **`37.692`** `[37.537..38.243]` | `61.02` | `0.0212` | `18.769` / `25.945` | `163.91×` |
| `e8had_136b_decode_simd8_avx2_1t` | `136b` / `4.500` | 1 | `33.411` `[33.247..33.688]` | `68.84` | `0.0188` | `9.976` / `10.887` | `184.91×` |
| `e8had_136b_gemv_materialized_avx2_1t` | `136b` / `4.500` | 1 | `33.789` `[33.159..34.173]` | `68.07` | `0.0190` | `10.896` / `17.870` | `182.84×` |
| `e8had_136b_gemv_adjoint_avx2_1t` | `136b` / `4.500` | 1 | `33.673` `[33.465..34.274]` | `68.30` | `0.0189` | `8.464` / `11.851` | `183.48×` |
| `e8had_136b_gemv_adjoint_batch4_avx2_1t` | `136b` / `4.500` | 1 | **`34.798`** `[33.316..35.457]` | `66.10` | `0.0196` | `11.150` / `14.708` | **`177.55×`** |
| `e8had_136b_gemv_adjoint_batch4_avx2_2t` | `136b` / `4.500` | 2 | `35.992` `[35.668..36.368]` | `63.90` | `0.0202` | `30.670` / `31.625` | `171.65×` |
| `e8_88b_decode_std_scalar_1t` | `88b` / `3.000` | 1 | `65.457` `[64.698..65.895]` | `35.14` | `0.0245` | `10.466` / `18.181` | `94.39×` |
| `e8had_88b_decode_wht_avx2_1t` | `88b` / `3.000` | 1 | **`66.864`** `[65.149..68.487]` | `34.40` | `0.0251` | `25.314` / `32.878` | `92.40×` |
| `e8had_88b_decode_simd8_avx2_1t` | `88b` / `3.000` | 1 | `53.044` `[52.903..53.198]` | `43.36` | `0.0199` | `29.292` / `32.066` | `116.47×` |
| `e8had_88b_gemv_materialized_avx2_1t` | `88b` / `3.000` | 1 | `61.562` `[60.387..62.387]` | `37.36` | `0.0231` | `21.486` / `61.424` | `100.36×` |
| `e8had_88b_gemv_adjoint_avx2_1t` | `88b` / `3.000` | 1 | `62.688` `[61.185..63.592]` | `36.69` | `0.0235` | `11.115` / `18.112` | `98.55×` |
| `e8had_88b_gemv_adjoint_batch4_avx2_1t` | `88b` / `3.000` | 1 | **`62.966`** `[62.450..63.781]` | `36.53` | `0.0236` | `11.478` / `20.226` | **`98.12×`** |
| `e8had_88b_gemv_adjoint_batch4_avx2_2t` | `88b` / `3.000` | 2 | `63.229` `[62.172..63.540]` | `36.38` | `0.0237` | `39.565` / `43.295` | `97.71×` |

### Table B: `clang_native` (`Clang 23.1.3 -O3 -march=native`) — `4,194,304` Weights (`131,072` Blocks)

| Kernel | Reg. / Bpw | Thr | CPU/Core Med (`[min..max]`, M w/s) | Nom. Cyc/w | Eff. BW CPU (GB/s) | Wall Med / Max (M w/s) | Slowdown vs Byte-Matched Q4 Proxy (`1t` CPU) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `q4km_style_gemv_avx2_1t` | Q4 proxy / `4.500` | 1 | **`6,540.509`** `[6,402.247..6,663.337]` | `0.35` | `3.6790` | `6,527.002` / `6,593.076` | **`1.00×`** (proxy baseline) |
| `q4km_style_gemv_avx2_2t` | Q4 proxy / `4.500` | 2 | `6,563.743` `[6,497.458..6,644.968]` | `0.35` | `3.6921` | `499.712` / `5,886.924` | `1.00×` |
| `e8_136b_decode_std_scalar_1t` | `136b` / `4.500` | 1 | `33.351` `[32.633..34.539]` | `68.96` | `0.0188` | `12.158` / `16.040` | `196.11×` |
| `e8had_136b_decode_wht_scalar_1t` | `136b` / `4.500` | 1 | `35.036` `[32.519..35.287]` | `65.65` | `0.0197` | `15.183` / `23.793` | `186.68×` |
| `e8had_136b_decode_wht_avx2_1t` | `136b` / `4.500` | 1 | **`34.458`** `[33.227..35.602]` | `66.75` | `0.0194` | `16.364` / `18.211` | `189.81×` |
| `e8had_136b_decode_simd8_avx2_1t` | `136b` / `4.500` | 1 | `32.105` `[30.947..32.283]` | `71.64` | `0.0181` | `6.341` / `8.125` | `203.73×` |
| `e8had_136b_gemv_materialized_avx2_1t` | `136b` / `4.500` | 1 | `31.219` `[28.158..31.576]` | `73.67` | `0.0176` | `3.858` / `5.055` | `209.50×` |
| `e8had_136b_gemv_adjoint_avx2_1t` | `136b` / `4.500` | 1 | `30.058` `[29.373..31.267]` | `76.52` | `0.0169` | `4.324` / `4.882` | `217.60×` |
| `e8had_136b_gemv_adjoint_batch4_avx2_1t` | `136b` / `4.500` | 1 | **`29.756`** `[27.179..32.723]` | `77.30` | `0.0167` | `2.550` / `4.013` | **`219.80×`** |
| `e8had_136b_gemv_adjoint_batch4_avx2_2t` | `136b` / `4.500` | 2 | `33.896` `[32.571..34.579]` | `67.85` | `0.0191` | `25.457` / `30.797` | `192.96×` |
| `e8_88b_decode_std_scalar_1t` | `88b` / `3.000` | 1 | `64.270` `[63.704..64.478]` | `35.79` | `0.0241` | `7.041` / `8.266` | `101.77×` |
| `e8had_88b_decode_wht_avx2_1t` | `88b` / `3.000` | 1 | `61.288` `[61.006..62.135]` | `37.53` | `0.0230` | `8.267` / `24.657` | `106.72×` |
| `e8had_88b_decode_simd8_avx2_1t` | `88b` / `3.000` | 1 | `57.799` `[56.813..58.433]` | `39.79` | `0.0217` | `3.563` / `5.928` | `113.16×` |
| `e8had_88b_gemv_materialized_avx2_1t` | `88b` / `3.000` | 1 | `61.970` `[61.615..62.571]` | `37.11` | `0.0232` | `4.768` / `22.974` | `105.54×` |
| `e8had_88b_gemv_adjoint_avx2_1t` | `88b` / `3.000` | 1 | `62.062` `[61.568..62.845]` | `37.06` | `0.0233` | `4.448` / `7.117` | `105.39×` |
| `e8had_88b_gemv_adjoint_batch4_avx2_1t` | `88b` / `3.000` | 1 | **`62.818`** `[61.494..63.594]` | `36.61` | `0.0236` | `3.673` / `4.073` | **`104.12×`** |
| `e8had_88b_gemv_adjoint_batch4_avx2_2t` | `88b` / `3.000` | 2 | `64.731` `[63.150..65.827]` | `35.53` | `0.0243` | `43.973` / `53.505` | `101.04×` |

---

## 6. Pinpoint 4-Stage Isolated-Median Cycle Breakdown

Each stage was measured in isolation over the same `4,194,304`-weight (`131,072`-block) matrix using `CLOCK_THREAD_CPUTIME_ID` (`5` trials + `1` warmup). **All stage percentages below are shares of the sum of the four isolated stage medians, not samples of a fused end-to-end run.** *(Specifically, the sum of isolated stage medians is `~60.05` / `68.67` nom. cyc/w for `136b` vs. `68.07` / `73.67` end-to-end fused materialized GEMV because isolated micro-loops hold smaller live register/cache working sets.)*

| Regime & Stage | `gcc_native` (cyc/w / `ns/w` / Share of 4 Isolated Medians) | `clang_native` (cyc/w / `ns/w` / Share of 4 Isolated Medians) | Engineering Meaning |
| :--- | :---: | :---: | :--- |
| **`EBG32:b136s8` — Stage 0:** Outer tree splits (`E_8^4 -> 4 × E_8`) | `14.83` / `6.446` / **`24.69%`** | `15.49` / `6.734` / **`22.56%`** | `128`-bit Barret/reciprocal split + binary search over `m <= 391` |
| **`EBG32:b136s8` — Stage 1:** Inner `4 × E_8` leaf unrank (`8->4->2->LUT`) | **`43.50`** / `18.913` / **`72.45%`** | **`51.60`** / `22.434` / **`75.14%`** | **DOMINANT STAGE (`~3/4` of isolated-median sum)** |
| **`EBG32:b136s8` — Stage 2:** Materialized AVX2 `WHT_8` | `1.26` / `0.548` / `2.10%` | `1.13` / `0.492` / `1.65%` | Essentially free; eliminable via adjoint $H_8 x$ |
| **`EBG32:b136s8` — Stage 3:** AVX2/FMA dot-product | `0.46` / `0.199` / `0.76%` | `0.45` / `0.196` / `0.66%` | Matches Q4 proxy FMA throughput (`0.35–0.37` cyc/w) |
| **`EBG32:b88s8` — Stage 0:** Outer tree splits (`E_8^4 -> 4 × E_8`) | `10.00` / `4.347` / **`27.70%`** | `10.41` / `4.525` / **`27.16%`** | L1d-resident (`m <= 48`), still `> 1/4` of isolated-median sum |
| **`EBG32:b88s8` — Stage 1:** Inner `4 × E_8` leaf unrank (`8->4->2->LUT`) | **`24.54`** / `10.670` / **`67.99%`** | **`26.19`** / `11.388` / **`68.34%`** | **DOMINANT STAGE (`~2/3` of isolated-median sum)** |
| **`EBG32:b88s8` — Stage 2:** Materialized AVX2 `WHT_8` | `1.11` / `0.483` / `3.08%` | `1.21` / `0.526` / `3.16%` | Essentially free; eliminable via adjoint $H_8 x$ |
| **`EBG32:b88s8` — Stage 3:** AVX2/FMA dot-product | `0.44` / `0.193` / `1.23%` | `0.52` / `0.224` / `1.35%` | `< 1.4%` of isolated-median sum |

### Concrete Actionable Roadmap for `siliquant#23/#25`
1. **Do NOT optimize `WHT_8` or the FMA dot-product first:** Together, Stages 2 and 3 account for only **`2.31%–4.51%`** of the four isolated stage medians. Standard `E_8^4` BallPack + exact zero-table `WHT_8` (or activation-adjoint $H_8 x$) already achieves zero table overhead and near-zero WHT cost.
2. **Attack Stage 1 (`67.99%–75.14%` share of the four isolated stage medians) directly:** Every `E_8` leaf currently performs `3` hierarchical `64`-bit split/unrank levels (`8 -> 4 -> 2 -> LUT`). On GPU/TPU (or CPU with a compact precomputed subtable), direct low-shell `E_8` indexing (`m <= 3` holds `9,121` vectors = `~73 KB` uncompressed or `< 16 KB` via `D_4` orbit + sign symmetries!) or replacing `E_8` `8->4->2` unranking with a direct `D_4 × D_4` table lookup collapses Stage 1 by `2×–4×`.
3. **Target HBM-Bandwidth-Bound Accelerators (GPU Warps / Apple Silicon GPU / TPU — Unmeasured Here):** With Stage 1 shortened to `~15–20` warp-parallel integer instructions per `E_8` leaf (`~2–3` instructions/weight across 32 threads), a `4.500`-bpw `EBG32:b136s8` or `3.000`-bpw `EBG32:b88s8` kernel could potentially overlap decode latency with HBM reads — where `1.067×–1.600×` smaller matrix traffic than `4.80`-bpw `Q4_K_M` would need to be verified on actual GPU/TPU silicon.

---

## 7. Reproducibility & File Manifest (`track-s15c-e8had-simd-ballpack/`)

All checks run cleanly offline with pure Python 3 stdlib (`-I -B`, zero network, zero checked-in binaries, zero `__pycache__`):

```bash
# 1. Regenerate & verify pure-Z/Q exact isometry certificate (4-mode -I deterministic)
python3 -I -B e8had_isometry_audit.py --standalone --json
python3 -I -B e8had_isometry_audit.py --check e8had_isometry_certificate.json

# 2. Run assertion-free R5 test suite
python3 -I -O -B -W error test_e8had_isometry_r5.py

# 3. Run deterministic 2-compiler (GCC + Clang) 0-div & exactness verification (verify-only mode)
python3 -I -B c-bench/run_e8had_simd_bench.py --json

# 4. Run full 4096x1024 empirical benchmark + 4-stage breakdown
python3 -I -B c-bench/run_e8had_simd_bench.py --bench --write-results c-bench/simd_bench_results.json
```

| File | Role |
| :--- | :--- |
| `e8had_isometry_oracle.py` | Zero-shared-code pure-$\mathbb{Z}/\mathbb{Q}$ oracle (`H_8`, `WHT_8`, `[8,4,4]` Hamming, `m=0..3` bijection, theta/DP invariance, adjoint GEMV; `22/22` mutate) |
| `e8had_isometry_audit.py` | Standalone + fail-closed Zone-B `--check` CLI (`19/19` mutate, 4-mode `-I` deterministic) |
| `test_e8had_isometry_r5.py` | Assertion-free R5 test suite covering Pillars 1–3, fail-closed tamper checks, and CLI parity |
| `e8had_isometry_certificate.json` | Canonical pinned JSON proof receipt (`5b56e0cd...`) |
| `c-bench/e8had_simd_bench.c` | Zero-warning C11 + explicit AVX2/FMA decode, batched `simd8`, materialized/adjoint/batch4 GEMV, byte-matched custom symmetric Q4 proxy GEMV, and isolated 4-stage timers (`0` `div`/`idiv`) |
| `c-bench/run_e8had_simd_bench.py` | Ephemeral `0700` compiler build, `objdump` `0`-div verifier, exactness checker, and dual-clock benchmark runner |
| `c-bench/simd_bench_results.json` | Pinned `4096 × 1024` empirical summary receipt (medians, extrema, and derived ratios — not raw per-trial arrays; `GCC 16.2.1` & `Clang 23.1.3`) |
| `E8HAD_SIMD_AUDIT_REPORT.md` | This report |
