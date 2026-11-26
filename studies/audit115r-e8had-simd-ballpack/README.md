# Exact `E_8_Had` Zero-Table `WHT_8` Isometry, Adjoint Fused-GEMV & 4-Stage Cycle Breakdown (`AUDIT-115R`)

Companion exact-math proof and real-hardware C11 + AVX2/FMA fused-GEMV audit for `Moissanite-Labs/siliquant#23`, `#24`, and `#25`, extending the packed binary-tree `E_8^4` BallPack codec in Draft PR `#35` (`AUDIT-112R`).

- **Oracle Class:** `exact+anchor` (pure-$\mathbb{Z}/\mathbb{Q}$ exact algebraic oracle + real-hardware `4096 × 1024` C11/AVX2/FMA silicon anchor on the disclosed Zen-1 host against a byte-matched `4.500`-bpw custom symmetric nibble proxy baseline, not `llama.cpp` `Q4_K_M`)
- **Novelty Classification:** `N1` (exact algebraic-isometry + activation-adjoint GEMV proof combined with measured AVX2/FMA fused-GEMV on this implementation and host, byte-matched symmetric Q4 proxy baseline, and 4-stage isolated-median cycle breakdown for `E_8_Had` BallPack; no claim of CPU superiority over `Q4_K_M`, no GPU/TPU measurements, and no end-to-end `E_8_Had` model-quality/KL claim — the committed Qwen2.5 `0.020817` KL figure is on standard `E_8` `EBG32:b136s8`, while `#25` notes the real-weight `E_8_Had` quality run has not yet executed)
- **Hygiene:** Pure Python 3 stdlib (`int` / `fractions.Fraction`, `0` floats on trust paths, `0` bare `assert`s) + pure C11 (`-std=c11 -Wall -Wextra -Werror`, `0` warnings under `GCC 16.2.1` and `Clang 23.1.3`, `0` hardware `div`/`idiv` across `8/8` hot kernels, `0` checked-in binaries).

## Key Verified Results

### 1. Exact Zero-Table `E_8_Had` Isometry (`e8had_isometry_oracle.py`, `e8had_isometry_certificate.json`)
- **Exact Construction-A Bijection:** With the Sylvester–Hadamard matrix $H_8 = H_2^{\otimes 3}$ ($H_8 = H_8^T$, $H_8^2 = 8 I_8$), the integer butterfly map $z = (H_8 u)/4$ (`24` add/sub ops, `0` multiplies, inverse $u = (H_8 z)/2$) maps $2 E_8^{\mathrm{std}}$ bijectively onto Construction-A $\Lambda_{\mathrm{CA}} = 2\mathbb{Z}^8 + \mathcal{H}_8 = \sqrt{2}\, E_8^{\mathrm{Had}}$, where $\mathcal{H}_8 = \mathrm{RM}(1,3)$ is the self-dual `[8,4,4]` extended Hamming code (`16` codewords: `1` weight-0, `14` weight-4, `1` weight-8, $d_{\min} = 4$).
- **Exhaustive Shell Verification (`m = 0..3`):** **`9,121 / 9,121` vectors (`9,120` nonzero)** verified bijectively with exact norm isometry $\|z\|^2 = 4m = \|u\|^2/2$ and exact theta counts $240\,\sigma_3(m)$ (`m=0: 1`, `m=1: 240` [`16` axis + `224` weight-4], `m=2: 2,160`, `m=3: 6,720`).
- **Zero-Table Replacement for `#24` & `#25`:** Matches `#24`'s `closest_E8_had(x) = closest_E8_std(x @ U_H8) @ U_H8` ($U_{H_8} = H_8/\sqrt{8}$) with `pr24_extra_permutation_needed: false` and **`0` extra lookup-table bytes** over the existing `E_8^4` BallPack DP tables across all `392` shells (`EBG32:b136s8` `m_max = 391`, `56,850` B; `EBG32:b88s8` `m_max = 48`, `5,490` B L1d-resident).

### 2. Activation-Adjoint `WHT_8` Elimination (`4,096×` WHT Reduction, With Honest CPU Amdahl Bound)
- **Exact Identity:** $4\langle z, x \rangle = \langle u, H_8 x \rangle$ verified identically across all **`960 / 960`** tested `(u, x)` pairs. Pre-rotating the activation vector once per GEMV reduces `WHT_8` ops/weight from `3` to **`3/4096 = 0.000732`** (**`4,096×` reduction** on a `4096 × 1024` matrix).
- **Honest CPU Amdahl Bound:** On this x86-64 Zen-1 implementation and host, materialized AVX2 `WHT_8` (Stage 2) represents only **`1.65%–3.16%`** of the sum of the four isolated stage medians (`1.11–1.26` nominal cycles/weight), so eliminating per-weight `WHT_8` yields at most `~2%–3%` end-to-end CPU speedup (measured `+3.0%` on `gcc` `136b`, within boost variance on `clang`).

### 3. Real-Hardware AVX2/FMA GEMV & `67.99%–75.14%` Stage-1 Isolated-Median Share (`c-bench/`)
- **Compiler, `0`-Div & Exactness Verification:** Built cleanly (`0` warnings) under `GCC 16.2.1` and `Clang 23.1.3` (`-O3 -march=native -Wall -Wextra -Werror -std=c11`), `objdump -d` proves **`0` hardware `div`/`idiv`** across all `8/8` hot kernels, and full-matrix verification (`4096 × 1024 = 4,194,304` weights = `131,072` blocks = `524,288` `E_8` leaves per regime, **`8,388,608` decoded weights total**) passes with **`0` mismatches**, cross-compiler bit-identical output hashes (`136b`: `4863b413...`; `88b`: `644aba45...`), and `128/128` block parity against the pure-$\mathbb{Z}$ Python oracle.
- **CPU Throughput vs. Byte-Matched `4.500`-bpw Custom Symmetric Nibble Proxy (`AMD Athlon Silver 3050U`, Zen 1, `CLOCK_THREAD_CPUTIME_ID`, `5` trials + `1` warmup; custom symmetric nibble layout, not `llama.cpp` `Q4_K_M` — do not infer production `Q4_K_M` latency; no GPU/TPU measurements):**
  - Byte-matched Q4 proxy (`4.500` bpw, `1t`): **`6,178.179` M w/s** (`0.37` cyc/w, `3.4752 GB/s`, `gcc`) / **`6,540.509` M w/s** (`0.35` cyc/w, `3.6790 GB/s`, `clang`).
  - `EBG32:b136s8` (`4.500` bpw, `adjoint_batch4_avx2_1t`): **`34.798` M w/s** (`66.10` cyc/w, **`177.55×` slower**, `gcc`) / **`29.756` M w/s** (`77.30` cyc/w, **`219.80×` slower**, `clang`).
  - `EBG32:b88s8` (`3.000` bpw, `adjoint_batch4_avx2_1t`): **`62.966` M w/s** (`36.53` cyc/w, **`98.12×` slower**, `gcc`) / **`62.818` M w/s** (`36.61` cyc/w, **`104.12×` slower**, `clang`).
  - **Negative SIMD Finding:** 8-leaf batched `e8had_decode_simd8_avx2` regresses **`-6.8%` to `-20.7%`** relative to single-block `e8had_decode_wht_avx2` on Zen 1 due to double-pumped 256-bit execution and 8-lane pack/unpack overhead without vectorizing the data-dependent tree searches.
- **Pinpoint 4-Stage Isolated-Median Cycle Breakdown (`CLOCK_THREAD_CPUTIME_ID`, nominal `2300 MHz`; percentages are shares of the four isolated stage medians, not samples of a fused end-to-end run; `simd_bench_results.json` stores medians, extrema, and derived summaries rather than raw per-trial arrays):**

| Regime & Stage | `gcc_native` (`cyc/w` / `ns/w` / Share of 4 Isolated Medians) | `clang_native` (`cyc/w` / `ns/w` / Share of 4 Isolated Medians) |
| :--- | :---: | :---: |
| **`EBG32:b136s8` — Stage 0** (Outer `E_8^4 -> 4 × E_8`) | `14.83` / `6.446` / `24.69%` | `15.49` / `6.734` / `22.56%` |
| **`EBG32:b136s8` — Stage 1** (Inner `4 × E_8` leaf `8->4->2->LUT`) | **`43.50`** / `18.913` / **`72.45%`** | **`51.60`** / `22.434` / **`75.14%`** |
| **`EBG32:b136s8` — Stage 2** (Materialized AVX2 `WHT_8`) | `1.26` / `0.548` / `2.10%` | `1.13` / `0.492` / `1.65%` |
| **`EBG32:b136s8` — Stage 3** (AVX2/FMA dot-product) | `0.46` / `0.199` / `0.76%` | `0.45` / `0.196` / `0.66%` |
| **`EBG32:b88s8` — Stage 0** (Outer `E_8^4 -> 4 × E_8`, L1d) | `10.00` / `4.347` / `27.70%` | `10.41` / `4.525` / `27.16%` |
| **`EBG32:b88s8` — Stage 1** (Inner `4 × E_8` leaf `8->4->2->LUT`) | **`24.54`** / `10.670` / **`67.99%`** | **`26.19`** / `11.388` / **`68.34%`** |
| **`EBG32:b88s8` — Stage 2** (Materialized AVX2 `WHT_8`) | `1.11` / `0.483` / `3.08%` | `1.21` / `0.526` / `3.16%` |
| **`EBG32:b88s8` — Stage 3** (AVX2/FMA dot-product) | `0.44` / `0.193` / `1.23%` | `0.52` / `0.224` / `1.35%` |

**Actionable Conclusion:** Stages 2 + 3 combined account for only **`2.31%–4.51%`** of the four isolated stage medians on this implementation and host. Optimizing `E_8_Had` BallPack dequantization means attacking **Stage 1 inner `E_8` leaf unranking (`67.99%–75.14%` of the isolated-median sum)** directly (and evaluating HBM-bandwidth-bound GPUs/TPUs directly — unmeasured here — where `1.067×–1.600×` smaller matrix traffic than `4.80`-bpw `Q4_K_M` could potentially overlap decode arithmetic).

## Offline Reproduction

All checks run offline using pure Python 3 stdlib (`-I -B`, zero network, ephemeral `0700` compiler builds, zero checked-in binaries):

```bash
cd studies/audit115r-e8had-simd-ballpack

# 1. 4-mode -I deterministic pure-Z/Q certificate parity + fail-closed --check
python3 -I -B e8had_isometry_audit.py --standalone --json | sha256sum
python3 -I -O -B e8had_isometry_audit.py --standalone --json | sha256sum
python3 -I -O -B -W error e8had_isometry_audit.py --standalone --json | sha256sum
PYTHONHASHSEED=42 python3 -I -B e8had_isometry_audit.py --standalone --json | sha256sum
# Expected 4-mode SHA-256: 3344fc066debe71c375888b126425f36e6bc3a154d2ef9e452888145ece7b8da
# Inner certificate_sha256: 5b56e0cd950e292d8eaee68641198cd5bfd11774921ca6ecfdf0bbf60d1c2b4a
python3 -I -O -B e8had_isometry_audit.py --check e8had_isometry_certificate.json

# 2. Assertion-free R5 adversarial test suite (41/41 100.0% mutation score: 22/22 oracle + 19/19 audit)
python3 -I -O -B -W error test_e8had_isometry_r5.py -v

# 3. Deterministic 2-compiler (GCC + Clang) 0-div & full 8,388,608-weight exactness verify-only check
python3 -I -B c-bench/run_e8had_simd_bench.py --json

# 4. Re-run full 4096x1024 empirical benchmark + 4-stage breakdown
python3 -I -B c-bench/run_e8had_simd_bench.py --bench --write-results c-bench/simd_bench_results.json
```

## File Manifest

| File | Role |
| :--- | :--- |
| `README.md` | Study index and offline reproduction guide |
| `E8HAD_SIMD_AUDIT_REPORT.md` | Full mathematical proof summary, dual-compiler silicon tables, and 4-stage breakdown |
| `e8had_isometry_oracle.py` | Pure-$\mathbb{Z}/\mathbb{Q}$ oracle (`H_8`, `WHT_8`, `[8,4,4]` Hamming, `m=0..3` bijection, DP invariance, adjoint GEMV; `22/22` mutate) |
| `e8had_isometry_audit.py` | Standalone + fail-closed Zone-B `--check` CLI (`19/19` mutate, 4-mode `-I` deterministic) |
| `test_e8had_isometry_r5.py` | Assertion-free R5 test suite covering Pillars 1–3, fail-closed tamper checks, and CLI parity |
| `e8had_isometry_certificate.json` | Canonical pinned pure-$\mathbb{Z}/\mathbb{Q}$ JSON receipt (`5b56e0cd...`) |
| `c-bench/e8had_simd_bench.c` | Zero-warning C11 + explicit AVX2/FMA decode, batched `simd8`, materialized/adjoint/batch4 GEMV, byte-matched custom symmetric Q4 proxy GEMV, and isolated 4-stage timers (`0` `div`/`idiv`) |
| `c-bench/run_e8had_simd_bench.py` | Ephemeral `0700` dual-compiler build, `objdump` `0`-div verifier, exactness checker, and dual-clock benchmark runner |
| `c-bench/simd_bench_results.json` | Pinned `4096 × 1024` empirical summary receipt (medians, extrema, and derived summaries — not raw per-trial arrays; `GCC 16.2.1` & `Clang 23.1.3`) |
