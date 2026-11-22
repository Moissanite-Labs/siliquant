# Attacking the `68%–75%` Stage-1 Inner `E_8` Leaf Unranking Bottleneck (`AUDIT-116R`)

Companion exact-math proof and real-hardware C11 + AVX2/FMA head-to-head benchmark study for `Moissanite-Labs/siliquant#23`, `#24`, and `#25`, building directly on Draft PR `#35` (`AUDIT-112R`, packed binary-tree `E_8^4` codec) and Draft PR `#36` (`AUDIT-115R`, zero-table `E_8_Had` adjoint GEMV & 4-stage cycle breakdown identifying Stage 1 as `68%–75%` of total cycles).

- **Oracle Class:** `exact+anchor` (pure-$\mathbb{Z}/\mathbb{Q}$ exact algebraic oracle + real-hardware `4096 × 1024` C11/AVX2/FMA silicon anchor across 6 Stage-1 variants, 3 weight distributions, and a matched `4.500`-bpw `Q4_K_M`-style baseline)
- **Novelty Classification:** `N1` (exact bit-width, `O(1)` top-bit bucket guide, concentration-of-measure LUT hit bound, and `D_8^+` `516.58×` leader phase-transition proofs paired with head-to-head real-silicon C11 + AVX2/FMA benchmarking and an honest x86-64 CPU floor verdict)
- **Hygiene:** Pure Python 3 stdlib (`int` / `fractions.Fraction`, `0` floats on trust paths, `0` bare `assert`s) + pure C11 (`-std=c11 -Wall -Wextra -Werror`, `0` warnings under `GCC 16.2.1` and `Clang 23.1.3`, `0` hardware `div`/`idiv` across `13/13` hot kernels, `0` checked-in binaries).

## Key Verified Results

### 1. Exact Pure-$\mathbb{Z}/\mathbb{Q}$ Proofs (`e8_stage1_oracle.py`, `e8_stage1_certificate.json`)
- **Pillar 1 — Exact Stage-1 Integer Width Reduction:**
  - `EBG32:b88s8` (`m_max = 48`): $\max_{m \le 48} S_1(m) = 31,456,320$ (`25` bits at `m = 48`), $W_1(48) = 360,605,761$ (`29` bits $< 2^{32}$) $\implies$ **100% pure `uint32_t` Stage 1**; $\max c_4 = 1,872$ (`uint16_t`), $\max c_2 = 16$ (`uint8_t`).
  - `EBG32:b136s8` (`m_max = 391`): $\max_{m \le 391} S_1(m) = S_1(390) = 16,749,815,040$ (`34` bits), $W_1(391) = 1,525,056,448,801$ (`41` bits $< 2^{64}$) $\implies$ **100% pure `uint64_t` Stage 1, zero `uint128_t`**; $\max c_4 = 14,976$ (`uint16_t`), $\max c_2 = 24$ (`uint8_t`).
- **Pillar 2 — `O(1)` Top-Bit Bucket Guide (`b = rem >> sh`) & Exact Uniform LUT Hit Bounds:**
  - `guide_cdf[b] <= b << sh` proved exact across all `49/49` `88b` shells; expected extra sequential steps: **`0.6116`** (`88b m=12 B=32`, max span `4`), **`4.3596`** (`136b m=97 B=32`, max span `26`), **`1.9711`** (`136b m=97 B=64`, max span `18`).
  - Exact uniform $E_8^4$-ball low-norm LUT hit rates: `88b` **`0.756558%`** (`m <= 2`, `2,401` vecs, `19,208` B L1d) / **`2.341009%`** (`m <= 3`, `9,121` vecs, `72,968` B L2); `136b` **`0.000270%`** (`m <= 2`) / **`0.001001%`** (`m <= 3`).
- **Pillar 3 — `D_8^+` Leader-Orbit Bijection & Sharp `516.58×` Phase Transition ($\Theta(m_{\max}^{7/2})$):**
  - Exhaustive `m = 0..3` bijection verified across **`9,121 / 9,121`** vectors (`0` duplicates, `0` invalid); universal `p(8) = 22` shapes = **`95,503`** permutations (`764,024` B `pshufb` table) and `502` B binomial-mask table.
  - Exact leader counts ($\sum |\mathcal{O}(\lambda)| = 240\,\sigma_3(m)$): `88b` has **`2,000` leaders** (`955` even + `1,045` odd, max `111` at `m = 47`, `32,000` B L1d-resident), whereas `136b` explodes **`516.58×`** to **`1,033,160` leaders** (`409,367` even + `623,793` odd, max `8,749` at `m = 389`, `16,530,560` B leader table / `19.41 MB` total, `4.85×` larger than `4 MiB` L3).

### 2. Real-Hardware C11 + AVX2/FMA Head-to-Head Silicon Results (`c-bench/`, `4096 × 1024`)
- **Compiler, `0`-Div & Exactness Verification:** Built cleanly (`0` warnings) under `GCC 16.2.1` and `Clang 23.1.3` (`-O3 -march=native -Wall -Wextra -Werror -std=c11`), `objdump -d` proves **`0` hardware `div`/`idiv`** across all **`13/13` hot exported symbols**, and full-matrix verification (`4,194,304` weights = `131,072` blocks = `524,288` leaves per regime, **`8,388,608` decoded weights total**) passes with **`0` mismatches** and bit-identical `z_output_sha256` matching `AUDIT-115R` for variants `0..3` (`136b`: `4863b413...`; `88b`: `644aba45...`).
- **Clear Drop-In Winner — `(A) guide_a` (Zero Bitstream / Format Change):**
  - **`EBG32:b88s8` (`93,472` B):** Cuts Stage-1 uniform cycles **`1.83×` (`gcc`: `22.74 -> 12.43` cyc/w, `184.976 M w/s/c`) / `1.78×` (`clang`: `23.06 -> 12.95` cyc/w, `177.595 M w/s/c`)** (Gauss-Iso/Out `1.82×`/`1.77×` `gcc` | `1.82×`/`1.76×` `clang`), lifting `1t` Adjoint GEMV **`1.39×` (`gcc`: `92.818 M w/s/c`, `24.78` cyc/w, `53.27×` vs Q4) / `1.40×` (`clang`: `87.093 M w/s/c`, `26.41` cyc/w, `52.77×` vs Q4)**.
  - **`EBG32:b136s8` (`1.60 MB`):** Cuts Stage-1 uniform cycles **`1.33×` (`gcc`: `41.21 -> 31.08` cyc/w, `73.996 M w/s/c`) / `1.53×` (`clang`: `49.52 -> 32.35` cyc/w, `71.097 M w/s/c`)**, reaching **`1.72×`** on `clang` isotropic Gaussian (`43.99 -> 25.52` cyc/w; Gauss-Out `1.34×`/`1.55×`), lifting `1t` Adjoint GEMV **`1.08×` (`gcc`: `34.991 M w/s/c`, `141.30×` vs Q4) / `1.15×` (`clang`: `38.226 M w/s/c`, `120.22×` vs Q4)**.
- **Honest Falsifications & Negative Findings (`AMD Athlon Silver 3050U`, Zen 1, `CLOCK_THREAD_CPUTIME_ID`):**
  - **`(B) eytz_lut2` / `eytz_lut3` Falsified:** Per-block scaling pushes 32D blocks toward the ball boundary — on `136b`, empirical `m <= 3` hits are **`0 / 524,288` (`0.0000%`)** on isotropic Gaussian and `0.2075%` even with `3%` `6×` outliers (`88b`: `2.78%` Gauss-Iso, `14.40%` outliers). Eytzinger `cmov` gives `1.01×–1.09×` Stage-1 on `88b` (`0.96×–1.02×` GEMV) and **regresses to `0.76×–0.95×`** on `136b`.
  - **`(C1) d8_compact` Falsified:** Regresses to **`0.56×–0.65×` Stage-1** (`0.67×–0.76×` GEMV) across all regimes.
  - **`(C2) d8_pshufb` Trade-Off:** Matches `guide_a` on `88b` (`1.71×`/`1.80×` Stage-1, `1.36×`/`1.30×` GEMV) at `8.6×` larger tables (`806 KB` vs `93.5 KB`) and a non-BallPack intra-shell ordering; at `136b` (`19.41 MB`, `12.1×` larger), it trails `guide_a` on `gcc` uniform (`1.13×`) and both compilers' Gaussian streams (`1.08×–1.41×`), while `clang` reaches `1.68×` Stage-1 / `1.34×` GEMV (`44.514 M w/s/c`, `103.24×` vs Q4) on the tiled 4,096-entry uniform bank.
  - **Honest CPU Floor Verdict:** Even with `guide_a`, x86-64 Zen-1 BallPack GEMV remains **`~53×` (`88b`) to `~103×–141×` (`136b`) slower** than matched `4.500`-bpw `Q4_K_M`-style AVX2/FMA GEMV (`4,944.300` `gcc` / `4,595.687` `clang` M w/s/c). Practical deployment belongs on **HBM-bandwidth-bound GPUs/TPUs**, where `guide_a`'s `1`-shift `O(1)` lookup eliminates warp branch divergence with zero bitstream change.

## Offline Reproduction

All checks run offline with pure Python 3 stdlib (`-I -B`, zero network, ephemeral `0700` compiler builds, zero checked-in binaries):

```bash
cd studies/audit116r-e8-leaf-stage1-attack

# 1. 4-mode -I deterministic pure-Z/Q certificate parity + fail-closed --check
python3 -I -B e8_stage1_audit.py --standalone --json | sha256sum
python3 -I -O -B e8_stage1_audit.py --standalone --json | sha256sum
python3 -I -O -B -W error e8_stage1_audit.py --standalone --json | sha256sum
PYTHONHASHSEED=42 python3 -I -B e8_stage1_audit.py --standalone --json | sha256sum
# Expected 4-mode SHA-256: 093d8f4eb8f1ecf6af266a4b6f356685e01aa43f8c744aaed6a257dd77491066
# Inner payload_sha256:    55098d900544b0dd03d4ca292a4f6f82f4a887dc4f6a0cf6d9e19537eab06be7
python3 -I -O -B e8_stage1_audit.py --check e8_stage1_certificate.json

# 2. Assertion-free R5 adversarial test suite (AUDIT-116R: 40/40 100.0% live mutate [27/27 audit + 13/13 oracle] + 7/7 BoolOp)
python3 -I -O -B -W error test_e8_stage1_r5.py -v

# 3. Deterministic 2-compiler (GCC + Clang) 0-div & full 8,388,608-weight exactness verify-only check
python3 -I -B c-bench/run_e8_stage1_bench.py --json
# Expected 3-mode -I SHA-256: ea0826ef4a817060e6c1dc541075e17206b07e8a1e9ab1e6c42811ca2711e488

# 4. Re-run full 4096x1024 empirical benchmark across Uniform / Gaussian-Iso / Gaussian-Outlier
python3 -I -B c-bench/run_e8_stage1_bench.py --bench --write-results c-bench/stage1_bench_results.json
```

*(Note: `AUDIT-116R` hardened `e8_stage1_audit.py` and `test_e8_stage1_r5.py` with strict 64-char lowercase ASCII hex validation before `hmac.compare_digest` and an explicit `2048`-bit JSON integer cap — raising live `aletheia v2.0.0 mutate --max-mutants 0` coverage from `AUDIT-116`'s `34/34` baseline documented in `E8_STAGE1_AUDIT_REPORT.md` to `40/40` (`100.0%`, `27/27` audit + `13/13` oracle) while keeping the other `6/8` files byte-for-byte unchanged.)*

## File Manifest

| File | Role |
| :--- | :--- |
| `README.md` | Study index and offline reproduction guide |
| `E8_STAGE1_AUDIT_REPORT.md` | Full mathematical proof summary, empirical LUT hit table, and dual-compiler silicon benchmarks |
| `e8_stage1_oracle.py` | Pure-$\mathbb{Z}/\mathbb{Q}$ oracle: bit-widths, `O(1)` bucket guide, exact uniform LUT bounds, `D_8^+` bijection & DP leader wall (`13/13` mutate, `6/6` `BoolOp`) |
| `e8_stage1_audit.py` | Standalone + hardened Zone-B fail-closed `--check` CLI (`hmac.compare_digest` w/ ASCII hex guard, `2048`-bit int cap, iterative `max_depth=64`; `27/27` mutate, `1/1` `BoolOp`) |
| `test_e8_stage1_r5.py` | Assertion-free R5 test suite covering Pillars 1–3, Zone-B tamper/boundary controls, and CLI parity |
| `e8_stage1_certificate.json` | Canonical pinned pure-$\mathbb{Z}/\mathbb{Q}$ JSON receipt (`payload_sha256: 55098d90...`) |
| `c-bench/e8_stage1_bench.c` | Zero-warning C11 + AVX2/FMA implementation of all 6 Stage-1 variants + matched Q4 baseline (`0` `div`/`idiv` across `13/13` hot kernels) |
| `c-bench/run_e8_stage1_bench.py` | Ephemeral `0700` dual-compiler build, `objdump` `0`-div verifier, 6-variant exactness proof, & 3-distribution silicon harness |
| `c-bench/stage1_bench_results.json` | Pinned `4096 × 1024` empirical silicon receipt (`GCC 16.2.1` & `Clang 23.1.3`) |
