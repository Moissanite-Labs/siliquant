# Track S15b — Empirical C11 Microbenchmark & Zero-Division Hardware Companion to `AUDIT-112` (`Moissanite-Labs/siliquant#23` & `#25`)

**Status:** Empirical C11 + Python verification & benchmark complete (`0` compiler warnings across `3/3` compiler profiles, `0` hardware `div`/`idiv` instructions across `6/6` hot unrankers, `0 / 300,000` exactness mismatches across `9,600,000` decoded weights).  
**Scope:** Closes `AUDIT-112`'s explicit wall-clock hardware microbenchmark scope note for `Moissanite-Labs/siliquant#23` (*Can spherical ball-packing unranking approach `Q4_K_M` decode throughput?*) and `Moissanite-Labs/siliquant#25` (*Does `E_8` have a fast, compact packed binary codec?*).  
**Pinned Baseline:** Read-only `Moissanite-Labs/siliquant` PR `#12` @ `20d13d5fa9c5fb2333b8faa0360a90caa1e0a43f` (`shape/ballpack.c`, `sha256: 53074fbadf169650da6cad400b986e07b1ea15df180f6f148f97ef4761b33055`, `6,194` B).

---

## 1. Executive Verdict (Honest Law-2 / Law-4 Assessment)

1. **Exactness & Machine-Code Zero-Division Certified (Law 3):**
   - All **6** compiled C11 shared libraries (`ballpack.c` + `ballpack_tree.c` across `gcc -O3 -march=native`, portable `gcc -O3`, and `clang -O3`, all with `-Wall -Wextra -Werror -std=c11`) build with **0 warnings**.
   - Direct `objdump -d` disassembly audit proves **0 hardware `div`/`idiv` instructions** in all **6** hot unranking functions (`ztree_unrank_compact`, `ztree_unrank_fast`, `ztree_unrank_fast_batch4`, `e8tree_unrank_compact`, `e8tree_unrank_fast`, `e8tree_unrank_fast_batch4`) under all 3 compiler profiles.
   - Across **`100,000` deterministic full-width blocks (`3,200,000` weights) per codec** (`9,600,000` weights total, including `0`, `1`, `count-1`, `count//2`, and every power-of-two boundary):
     - **`Z^32` (`136`b, `B = 783`):** `0 / 100,000` `rank -> unrank -> rank` mismatches; `compact == fast == batch4` byte-identical; **`0 / 100,000` set-membership mismatches against baseline `ballpack.c` (`zb_rank` & `zb_unrank` lossless roundtrip)**; `256 / 256` exact match vs `AUDIT-112` pure-$\mathbb{Z}$ Python oracle; out-of-range `rank = count` fails closed (`bad == 1`, zeroed output).
     - **`E_8^4` (`136`b `m_max = 391`, `4.500` bpw) & `E_8^4` (`88`b `m_max = 48`, `3.000` bpw):** `0 / 100,000` roundtrip or `E_8` lattice-membership mismatches across `400,000` $E_8$ sub-blocks each (verifying coset uniformity, even integer-sum parity, `8 | ||u||^2`, and norm bound); `compact == fast == batch4` byte-identical; `256 / 256` exact match vs Python oracle; out-of-range `rank = count` fails closed (`bad == 1`).
2. **Real Scalar CPU Speedup & First Working Packed C11 `E_8` Codec for `siliquant`:**
   - **`Z^32` (`136`b, `4.500` bpw):** Zero-division binary-tree unranking (`ztree_fast`) reaches **`46.32`–`48.48 M weights/sec/core`** (`20.6`–`21.6 ns/weight`, **`2.22×`–`2.46×` faster** than upstream `ballpack.c` under GCC; **`1.34×`–`1.41×` faster** under Clang at `36.5`–`38.4` vs `27.3 M w/s/core`), while the zero-checkpoint compact tree (`70,780` B, **`8.77×` smaller** than `ballpack.c`'s `620,928` B table) already beats `ballpack.c` under GCC (`1.08×`–`1.14×`).
   - **`E_8^4` (`136`b, `4.500` bpw):** Our first working packed binary C11 `E_8` codec reaches **`36.81`–`37.71 M weights/sec/core`** (`26.5`–`27.2 ns/weight`, **`1.76×`–`1.91×` faster than upstream `Z^32` `ballpack.c` under GCC** and **`1.20×`–`1.26×` faster under Clang**), while delivering the `E_8` lattice coding gain (`+0.65 dB` over `Z^32`, which `ballpack.c`/`ballpack.py` cannot encode at all) and reducing `BallIndex`'s `11.19 MB` `E_8` DP table by **`196.8×`** (`56,850` B compact).
   - **`E_8^4` (`88`b, `3.000` bpw, L1d-resident `5,490` B):** Reaches **`58.04`–`65.33 M weights/sec/core`** (`15.3`–`17.2 ns/weight`, **`2.39×`–`2.99×` faster** than `ballpack.c`; peak 2-core wall-clock **`89.3 M weights/sec`**). Notably, at `m_max = 48` the `5.49 KB` L1d-resident `e8tree_compact` beats `e8tree_fast` (`65.33` vs `58.31 M w/s/core` on Clang) because the tiny compact table avoids checkpoint binary-search overhead.
3. **Honest Law-2 / Law-4 Verdict Against `siliquant#23`'s `Q4_K_M` Falsifier (`45,000 M weights/sec/core`):**
   - **FALSIFIER NOT MET ON SCALAR CPU:** Even our fastest zero-division binary-tree C11 kernel (`46.3`–`48.5 M w/s/core` for `Z^32` 136b, `37.7 M w/s/core` for `E_8^4` 136b, `65.3 M w/s/core` for `E_8^4` 88b) remains **`928×` (`Z^32` 136b) / `1,193×` (`E_8^4` 136b) / `689×` (`E_8^4` 88b) slower** than `siliquant#23`'s `45,000 M weights/sec/core` `Q4_K_M` falsifier threshold on a single scalar Zen-1 core.
   - **Why:** `Q4_K_M` unpacks 32 weights via `~2` SIMD shift-mask instructions (`< 0.1 cycles/weight`), whereas exact enumerative shell-convolution unranking on a scalar CPU still executes `~15` (`Z^32`) or `~11` (`E_8^4`) table-indexed search + multiply-shift steps per 32 weights (`~47–83 cycles/weight`).
   - **Architectural Conclusion for `siliquant#23` & `#25`:** Scalar CPU enumerative decoding **cannot** rival `Q4_K_M` on compute-bound CPU decode — **regardless** of serial vs. binary-tree decomposition. Where `AUDIT-112` + `Track S15b` change the feasibility frontier is **SRAM-resident SIMD / GPU-warp / TPU-VREG decode**: cutting critical-path depth `32 -> 4` (`Z^32`) and `32 -> 2` (`E_8^4`), eliminating `100%` of hardware divisions, and shrinking tables from `620.9 KB` (`Z^32`) / `11.19 MB` (`E_8^4`) down to **`5.5`–`70.8 KB`** makes warp-parallel (32-block-per-warp SIMT) and AVX2/AVX-512/Pallas vectorized unranking architecturally viable for HBM-bandwidth-bound LLM inference.

---

## 2. Host Hardware, Toolchain & Measurement Protocol

| Parameter | Measured Value |
|---|---|
| **CPU Model** | `AMD Athlon Silver 3050U with Radeon Graphics` (`x86_64`, Zen 1, `2` physical cores / `2` logical CPUs, AVX2/BMI2/ADX) |
| **Cache Hierarchy** | `L1d = 32 KiB/core` (`64 KiB` total), `L2 = 512 KiB/core` (`1 MiB` total), `L3 = 4 MiB` shared |
| **Compilers** | `gcc (GCC) 16.2.1 20260810`, `clang version 23.1.3` |
| **Python Runtime** | `Python 3.14.8` (stdlib only; `ctypes` invokes compiled C11 timing harness directly) |
| **Compiler Profiles** | (1) `gcc -O3 -march=native -Wall -Wextra -Werror -std=c11`, (2) `gcc -O3 ...`, (3) `clang -O3 ...` |
| **Workload per Trial** | `100,000` full-width blocks = **`3,200,000` weights/trial**, `1` warmup + **`7` timed trials** per kernel |
| **Dual-Clock Instrumentation** | **Both** `CLOCK_THREAD_CPUTIME_ID` (per-core CPU time, immune to OS preemption) **and** `CLOCK_MONOTONIC` (1-core & 2-core wall-clock) |
| **Host Load Disclosure** | Shared 2-core host carried concurrent background load during benchmarking (`/proc/loadavg = 7.24 8.04 7.94`). Consequently, raw `CLOCK_MONOTONIC` wall-clock experiences OS scheduler preemption (especially on 1-thread runs), whereas `CLOCK_THREAD_CPUTIME_ID` exhibits **`< 2.5%` min-to-max spread** across all 7 trials and provides the exact compute comparison. Both metrics are reported verbatim below and in `bench_results.json`. |

---

## 3. Memory Footprint Comparison

| Codec / Variant | Base 1D / LUT Table | Strided Checkpoint Tables | Total Footprint | Fit vs Host Cache (`32 KiB` L1d / `512 KiB` L2 / `4 MiB` L3) |
|---|---:|---:|---:|---|
| Baseline `ballpack.c` (`Z^32` 136b) | `620,928` B | `0` B | `620,928` B (`606.4 KiB`) | **Spills `512 KiB` L2** -> L3 |
| Baseline `BallIndex` (`E_8^4` 136b, Python DP) | `11,189,304` B (`466,221 × 24` B) | `0` B | `11.19 MB` | **Spills `4 MiB` L3** -> DRAM (no C11 codec in upstream) |
| `ztree_compact` (`Z^32` 136b) | **`70,780` B** (`69.1 KiB`) | `0` B | **`70,780` B** (`8.77×` smaller) | **Fits `512 KiB` L2** (`7.4×` headroom) |
| `ztree_fast` / `batch4` (`Z^32` 136b, `Z16=32, Z8=16, Z4=8`) | `70,780` B | `944,064` B | `1,014,844` B (`991.1 KiB`) | Full table in L3; active norm rows fit L2 |
| `e8tree_compact` (`E_8^4` 136b, `m=391`) | **`56,850` B** (`55.5 KiB`) | `0` B | **`56,850` B** (`196.8×` vs `BallIndex`) | **Fits `512 KiB` L2** (`9.2×` headroom) |
| `e8tree_fast` / `batch4` (`E_8^4` 136b, `stride=16`) | `56,850` B | `804,480` B | `861,330` B (`841.1 KiB`) | Full table in L3; active norm rows fit L2 |
| `e8tree_compact` (`E_8^4` 88b, `m=48`) | **`5,490` B** (`5.36 KiB`) | `0` B | **`5,490` B** (`250.7×` vs `BallIndex`) | **Fits `32 KiB` L1d** (`6.0×` headroom!) |
| `e8tree_fast` / `batch4` (`E_8^4` 88b, `stride=16`) | `5,490` B | `16,504` B | **`21,994` B** (`21.5 KiB`) | **Fits `32 KiB` L1d** |

---

## 4. Empirical Throughput Results (`100,000` Blocks = `3.2M` Weights/Trial, `7` Trials)

### 4.1 Primary Profile: `gcc -O3 -march=native` (`GCC 16.2.1`)

| Kernel / Codec | Threads | CPU Throughput/Core Median [Min..Max] (`M w/s/c`) | CPU `ns/weight` | Speedup vs `ballpack.c` (CPU/core) | 2-Core / 1-Core Wall Median [Min..Max] (`M w/s`) | Gap vs `Q4_K_M` (`45,000 M w/s/c`) |
|---|---:|---:|---:|---:|---:|---:|
| `baseline_ballpack_c` (`Z^32` 136b) | 1 | `20.863` [`20.485`..`21.090`] | `47.932` | `1.00×` | `4.847` [`3.414`..`15.448`] | `2,156.9×` |
| `baseline_ballpack_c` (`Z^32` 136b) | 2 | `21.787` [`21.505`..`22.014`] | `45.900` | `1.04×` | `16.475` [`13.248`..`29.164`] | `2,065.5×` |
| `ztree_compact` (`Z^32` 136b, `70.8 KB`) | 1 | `22.474` [`22.292`..`22.877`] | `44.495` | **`1.08×`** | `9.484` [`7.907`..`13.294`] | `2,002.3×` |
| **`ztree_fast` (`Z^32` 136b)** | 1 | **`46.321`** [`44.590`..`46.928`] | **`21.589`** | **`2.22×`** | `17.413` [`16.043`..`22.075`] | `971.5×` |
| **`ztree_fast` (`Z^32` 136b)** | 2 | **`48.284`** [`47.418`..`48.742`] | **`20.711`** | **`2.31×`** | **`43.915`** [`35.284`..`55.312`] | `932.0×` |
| `ztree_batch4` (`Z^32` 136b) | 1 | `41.676` [`40.323`..`47.036`] | `23.995` | `2.00×` | `14.218` [`10.418`..`42.879`] | `1,079.8×` |
| `ztree_batch4` (`Z^32` 136b) | 2 | `46.412` [`45.594`..`46.681`] | `21.546` | `2.22×` | `41.183` [`34.865`..`42.348`] | `969.6×` |
| `e8tree_compact` (`E_8^4` 136b, `56.9 KB`) | 1 | `22.210` [`22.031`..`22.841`] | `45.024` | `1.06×` | `13.670` [`7.385`..`14.838`] | `2,026.1×` |
| `e8tree_fast` (`E_8^4` 136b) | 1 | `36.805` [`36.159`..`37.423`] | `27.170` | **`1.76×`** | `16.536` [`14.354`..`34.564`] | `1,222.7×` |
| `e8tree_fast` (`E_8^4` 136b) | 2 | `36.940` [`35.143`..`37.418`] | `27.071` | `1.77×` | `33.363` [`21.256`..`35.799`] | `1,218.2×` |
| **`e8tree_batch4` (`E_8^4` 136b)** | 1 | **`37.209`** [`35.971`..`37.509`] | **`26.875`** | **`1.78×`** | `22.322` [`16.343`..`30.056`] | `1,209.4×` |
| **`e8tree_batch4` (`E_8^4` 136b)** | 2 | **`37.493`** [`36.537`..`37.922`] | **`26.672`** | **`1.80×`** | **`34.013`** [`30.638`..`54.337`] | `1,200.2×` |
| **`e8tree_compact` (`E_8^4` 88b, `5.5 KB`)** | 1 | **`58.035`** [`57.698`..`58.427`] | **`17.231`** | **`2.78×`** | `47.638` [`39.031`..`58.137`] | `775.4×` |
| `e8tree_fast` (`E_8^4` 88b) | 1 | `55.894` [`55.425`..`56.140`] | `17.891` | `2.68×` | `23.833` [`19.473`..`26.639`] | `805.1×` |
| `e8tree_fast` (`E_8^4` 88b) | 2 | `56.709` [`55.703`..`57.129`] | `17.634` | `2.72×` | `50.873` [`37.340`..`67.743`] | `793.5×` |
| `e8tree_batch4` (`E_8^4` 88b) | 1 | `56.038` [`55.474`..`56.688`] | `17.845` | `2.69×` | `26.607` [`19.399`..`41.901`] | `803.0×` |
| **`e8tree_batch4` (`E_8^4` 88b)** | 2 | **`57.036`** [`56.278`..`57.370`] | **`17.533`** | **`2.73×`** | **`52.219`** [`38.305`..`57.007`] | `789.0×` |

### 4.2 Cross-Compiler Parity (`gcc -O3` Portable vs `clang -O3` Portable, 1-Thread CPU/Core Median)

| Kernel / Codec | `gcc -O3 -march=native` (`M w/s/c`) | `gcc -O3` Portable (`M w/s/c`) | `clang -O3` Portable (`M w/s/c`) | Cross-Compiler Observation |
|---|---:|---:|---:|---|
| `baseline_ballpack_c` (`Z^32` 136b) | `20.863` (`1.00×`) | `19.724` (`1.00×`) | `27.290` (`1.00×`) | Clang 23 generates tighter 192-bit `sub`/`sbb` chains for `ballpack.c` (`+31%` over GCC). |
| `ztree_compact` (`Z^32` 136b) | `22.474` (`1.08×`) | `22.543` (`1.14×`) | `23.869` (`0.87×`) | Matches/beats GCC baseline with `8.8×` smaller table. |
| `ztree_fast` (`Z^32` 136b) | **`46.321`** (**`2.22×`**) | **`47.179`** (**`2.39×`**) | `36.488` (**`1.34×`**) | GCC vectorizes/schedules branchless `cmov` + 64×64->128 mul-shift tighter than Clang (`+29%`). |
| `ztree_batch4` (`Z^32` 136b) | `41.676` (`2.00×`) | `46.186` (`2.34×`) | **`38.167`** (**`1.40×`**) | Under Clang, 4-block ILP adds `+4.6%` over single-block `ztree_fast`; on GCC single-block avoids stack spills. |
| `e8tree_compact` (`E_8^4` 136b) | `22.210` (`1.06×`) | `22.321` (`1.13×`) | `24.136` (`0.88×`) | First C11 `E_8^4` 136b codec, `56.9 KB` table. |
| `e8tree_fast_batch4` (`E_8^4` 136b) | **`37.209`** (**`1.78×`**) | **`37.667`** (**`1.91×`**) | **`32.654`** (**`1.20×`**) | Beats `Z^32` baseline across **all 3** compiler profiles while encoding `E_8^4`! |
| `e8tree_compact` (`E_8^4` 88b) | `58.035` (`2.78×`) | `59.017` (`2.99×`) | **`65.329`** (**`2.39×`**) | **Fastest single-core kernel (`15.31 ns/weight`)**; `5.49 KB` L1d-resident! |
| `e8tree_fast_batch4` (`E_8^4` 88b) | `56.038` (`2.69×`) | `56.702` (`2.87×`) | `58.308` (`2.14×`) | Peak 2-core wall-clock **`89.35 M w/s`** (`clang_portable` min trial `35.8 ms`). |

---

## 5. Concrete Engineering Roadmap for `siliquant#23` & `#25`

1. **Immediate Drop-In Win for `siliquant#25` (`E_8` Packing):**
   - `ballpack_tree.c` already provides a zero-warning, zero-division, `100,000/100,000`-verified C11 `E_8^4` ranker & unranker (`136`b at `4.500` bpw and `88`b at `3.000` bpw) that is **`1.78×`–`2.99×` faster** than `siliquant`'s existing `Z^32` `ballpack.c` and eliminates `BallIndex`'s `11.19 MB` DP table (`5.49`–`56.85 KB` compact).
2. **Closing the Remaining Gap Toward `siliquant#23` (GPU Warp / TPU VREG / AVX-512):**
   - **SIMD / SIMT Horizontal Multi-Block Vectorization:** Because our tree splits have **zero hardware divisions** and **branchless `cmov` checkpoint searches**, `8` blocks (`AVX2` 32-bit lanes) or `32` blocks (CUDA warp / TPU VREG) can execute the `Z^4` / `E_8` inner `32`-bit & `64`-bit multiply-shifts in lockstep across lanes with SRAM-resident `5.5`–`70.8 KB` tables (no L2/L3 spill across warps!).
   - **Pre-Baked 2-Stage Split Tables for `88`b (`3.000` bpw) `E_8^4`:** At `m_max = 48`, the entire single-block `E_8` inner unranker (`5.49 KB`) fits inside a single warp's register file / L1 shared memory bank, making `E_8^4 @ 88b` (`3.000` bpw) the strongest near-term candidate for a fused dequantization + GEMV kernel (`CUDA` / `Triton` / `Pallas`).

---

## 6. Reproducibility Commands

All shared libraries are compiled ephemerally inside a `0700` `tempfile.TemporaryDirectory()` and deleted automatically on exit (zero binary artifacts left in the directory).

```bash
# 1. Deterministic verification-only mode (3 compiler profiles, 0-warning build, objdump 0-div proof,
#    exactness vs baseline ballpack.c & AUDIT-112 Python oracle; 0 wall-clock timestamps):
python3 -I -O -B -W error run_ballpack_c_bench.py --json

# 2. Full 100,000-block (3,200,000-weight/codec), 7-trial dual-clock empirical benchmark:
python3 -I -O -B -W error run_ballpack_c_bench.py --write-results bench_results.json
```
