# Track S15b — Empirical Scalar-CPU C11 Microbenchmark & Exact Memory-Accounting Companion to `AUDIT-112` (`Moissanite-Labs/siliquant#23` & `#25`)

**Status:** Empirical C11 + Python verification and scalar-CPU microbenchmark complete (`0` compiler warnings across `3/3` compiler profiles, `0` hardware `div`/`idiv` instructions across `6/6` hot unrankers, `0 / 300,000` exactness mismatches across `9,600,000` decoded weights, exact `sizeof`/allocation-tracked C memory accounting).  
**Scope:** Exact research C11 rank/unrank implementation and scalar-CPU microbenchmark harness companion to `AUDIT-112` for `Moissanite-Labs/siliquant#23` and `Moissanite-Labs/siliquant#25`. This study is **not yet wired into a packed-file reader/writer, format integration, or production binding**, and makes **no GPU, TPU, or SIMD performance claims** extrapolated from this scalar CPU harness.  
**Pinned Baseline:** Read-only `Moissanite-Labs/siliquant` PR `#12` @ `20d13d5fa9c5fb2333b8faa0360a90caa1e0a43f` (`shape/ballpack.c`, `sha256: 53074fbadf169650da6cad400b986e07b1ea15df180f6f148f97ef4761b33055`, `6,194` B; `g_ways` table `620,928` B).  
**Measured Tree Implementation:** `c-bench/ballpack_tree.c` (`sha256: 68d804694f83476099975b1d2ac8962ecbb465f3f8ab2d013e52032b3b9abee4`).

---

## 1. Executive Verdict (Honest Law-2 / Law-4 Assessment)

1. **Exactness & Machine-Code Zero-Division Certified (Law 3):**
   - All **6** compiled C11 shared libraries (`ballpack.c` + `ballpack_tree.c` across `gcc -O3 -march=native`, portable `gcc -O3`, and `clang -O3`, all with `-Wall -Wextra -Werror -std=c11`) build with **0 warnings**.
   - Direct `objdump -d` disassembly audit proves **0 hardware `div`/`idiv` instructions** in all **6** hot unranking functions (`ztree_unrank_compact`, `ztree_unrank_fast`, `ztree_unrank_fast_batch4`, `e8tree_unrank_compact`, `e8tree_unrank_fast`, `e8tree_unrank_fast_batch4`) under all 3 compiler profiles.
   - Across **`100,000` deterministic full-width blocks (`3,200,000` weights) per codec** (`9,600,000` weights total, including `0`, `1`, `count-1`, `count//2`, and every power-of-two boundary):
     - **`Z^32` (`136`b, `B = 783`):** `0 / 100,000` `rank -> unrank -> rank` mismatches; `compact == fast == batch4` byte-identical; **`0 / 100,000` set-membership mismatches against baseline `ballpack.c` (`zb_rank` & `zb_unrank` lossless roundtrip)**; `256 / 256` exact match vs `AUDIT-112` pure-$\mathbb{Z}$ Python oracle; out-of-range `rank = count` fails closed (`bad == 1`, zeroed output).
     - **`E_8^4` (`136`b `m_max = 391`, `4.500` bpw) & `E_8^4` (`88`b `m_max = 48`, `3.000` bpw):** `0 / 100,000` roundtrip or `E_8` lattice-membership mismatches across `400,000` $E_8$ sub-blocks each (verifying coset uniformity, even integer-sum parity, `8 | ||u||^2`, and norm bound); `compact == fast == batch4` byte-identical; `256 / 256` exact match vs Python oracle; out-of-range `rank = count` fails closed (`bad == 1`).
2. **Exact C `sizeof`/Allocation Memory Accounting & Scalar-CPU Throughput:**
   - Every `malloc`/`calloc` inside `ztree_init` and `e8tree_init` is routed through `alloc_tracked` and exported via `ztree_footprint_breakdown` / `e8tree_footprint_breakdown` (`c-bench/bench_results.json`, `schema_version = track_s15b.ballpack_c_bench.v2`), separating: (a) **hot unrank working set** (compact vs. fast/batch4, including all multiply-shift reciprocals, `even_c8`, branch totals, and checkpoint offset/length metadata), (b) **rank-only inverse lookup grids**, (c) **init-allocated auxiliary arrays unread at runtime**, and (d) **total heap memory allocated unconditionally at init**.
   - **`Z^32` (`136`b, `B = 783`):**
     - **Memory:** Hot `ztree_unrank_compact` working set is **`172,700` B** (`59,804` B base/LUT + `112,896` B reciprocals; **`3.60×`** smaller than `ballpack.c`'s `620,928` B; adding init-unused `s16` gives `185,244` B = `3.35×`). Hot `ztree_unrank_fast` / `batch4` working set is **`1,135,580` B** (`172,700` B compact + `944,064` B checkpoint data + `18,816` B checkpoint metadata; `1.83×` larger than `ballpack.c`). Unconditional `ztree_init` allocates **`1,161,792` B** heap (+ `400` B `Z32Ctx`), including `12,100` B rank-only inverse grids and `14,112` B unused init arrays (`s16` + `z2_nz_r`).
     - **Scalar CPU (`CLOCK_THREAD_CPUTIME_ID`):** `ztree_fast` reaches **`41.035`–`46.795 M weights/sec/core`** (`21.37`–`24.37 ns/weight`, **`2.23×`–`2.54×`** vs `ballpack.c` `18.441 M w/s/c` on `gcc -O3 -march=native`; `2.01×`–`2.06×` on `gcc -O3`; `1.27×`–`1.35×` on `clang -O3` at `29.33`–`31.22` vs `23.11 M w/s/c`), while `ztree_compact` (`172.7 KB` hot) reaches `21.676 M w/s/c` (`1.18×` GCC native, `0.94×` Clang).
   - **`E_8^4` (`136`b, `m_max = 391`):**
     - **Memory:** Hot `e8tree_unrank_compact` working set is **`153,970` B** (`59,986` B base/LUT/`even_c8`/`p0_tot` + `93,984` B reciprocals; **`72.67×`** smaller than `BallIndex`'s `11,189,304` B DP table). Hot `e8tree_unrank_fast` / `batch4` working set is **`970,970` B** (`+ 798,208` B checkpoint data `+ 18,792` B checkpoint metadata; **`11.52×`** smaller than `BallIndex`). Unconditional `e8tree_init` allocates **`1,026,526` B** heap (+ `656` B `E8Ctx`), including `49,284` B rank-only inverse grids and `6,272` B unused `s_blk2` (**`10.90×`** smaller than `BallIndex`).
     - **Scalar CPU:** `e8tree_fast` / `batch4` reaches **`31.339`–`34.975 M w/s/c`** on `gcc -O3 -march=native` (`1.70×`–`1.90×` vs `Z^32` `ballpack.c`), `29.662`–`29.863 M w/s/c` on `gcc -O3` (`1.63×`–`1.64×`), and `29.739`–`30.124 M w/s/c` on `clang -O3` (`1.29×`–`1.30×`), while `e8tree_compact` reaches `21.33`–`23.58 M w/s/c`.
   - **`E_8^4` (`88`b, `m_max = 48`):**
     - **Memory:** Hot `e8tree_unrank_compact` working set is **`19,122` B** (`7,458` B base/LUT/`even_c8`/`p0_tot` + `11,664` B reciprocals; **`72.00×`** smaller than `BallIndex`'s `1,376,760` B; **fits `32 KiB` L1d**). Hot `e8tree_unrank_fast` / `batch4` working set is **`37,170` B** (`19,122` B compact + `15,720` B checkpoint data + `2,328` B checkpoint metadata; **`37.04×`** smaller than `BallIndex`; **exceeds `32 KiB` (`32,768` B) L1d** and spills into L2). Unconditional `e8tree_init` allocates **`44,038` B** heap (+ `656` B `E8Ctx`), including `6,084` B rank-only inverse grids and `784` B unused `s_blk2` (**`31.26×`** smaller than `BallIndex`).
     - **Scalar CPU:** `e8tree_compact` (`19,122` B, L1d-resident) reaches **`58.106` (`gcc_native`) / `57.891` (`gcc_portable`) / `64.786 M w/s/c` (`clang_portable`, `15.44 ns/weight`)** (`2.80×`–`3.18×` vs `Z^32` `ballpack.c`), beating `e8tree_fast` (`54.791` / `54.828` / `59.513 M w/s/c`) across all 3 compilers because `e8tree_compact` fits in the `32 KiB` L1d cache and avoids checkpoint binary-search overhead while `e8tree_fast` (`37,170` B) exceeds `32 KiB` L1d.
3. **Honest Law-2 / Law-4 Verdict Against `siliquant#23`'s `Q4_K_M` Falsifier (`45,000 M weights/sec/core`):**
   - **FALSIFIER NOT MET ON SCALAR CPU:** Across this scalar CPU harness, the fastest measured kernel (`e8tree_compact` 88b on `clang -O3` at `64.786 M w/s/c`; `ztree_fast` 136b 2t on `gcc_native` at `46.795 M w/s/c`; `e8tree_fast` 136b 2t on `gcc_native` at `34.975 M w/s/c`) remains **`695×` (`E_8^4` 88b) / `962×` (`Z^32` 136b) / `1,287×` (`E_8^4` 136b) slower** than `siliquant#23`'s `45,000 M weights/sec/core` `Q4_K_M` threshold on this Zen-1 host.
   - **Why:** `Q4_K_M` unpacks 32 weights with shift-mask operations, whereas exact enumerative shell-convolution unranking on a scalar CPU still executes table-indexed searches and multiply-shift splits per 32 weights (`15.4`–`46.1 ns/weight` on this host).
   - **Strict Scope Boundary:** This study establishes exact bijectivity, depth reduction (`32 -> 4` for `Z^32`; `32 -> 2` outer steps to 4 independent `E_8` blocks), `0` hardware divisions, exact C11 runtime memory footprints, and scalar-CPU single/dual-core timing on one disclosed Zen-1 machine. Whether vectorized SIMD, GPU-warp, or TPU kernels can close the remaining throughput gap or integrate profitably into `shape/ballpack.py` is **unmeasured here** and requires dedicated target-hardware kernel benchmarks and file-format wiring.

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
| **Host Load Disclosure** | Shared 2-core host carried heavy concurrent background load during benchmarking (`/proc/loadavg = 13.31 13.94 15.60`). Raw `CLOCK_MONOTONIC` wall-clock reflects OS scheduler contention on the shared host, whereas `CLOCK_THREAD_CPUTIME_ID` measures per-thread CPU execution time directly. Both metrics are reported verbatim below and in `c-bench/bench_results.json`; all throughput figures are host-specific to this scalar Zen-1 CPU. |

---

## 3. Exact C11 `sizeof` / Allocation Memory Footprint Breakdown

All byte counts below are measured live from `malloc`/`calloc` calls inside `ztree_init` and `e8tree_init` via `ztree_footprint_breakdown` / `e8tree_footprint_breakdown` (verified identical across all 3 compiler profiles; compiler type sizes: `sizeof(u192)=24`, `sizeof(u128)=16`, `sizeof(Recip192)=32`, `sizeof(Recip128)=32`, `sizeof(Recip64)=16`, `sizeof(Recip32)=8`).

### 3.1 Disjoint Allocation Partition (`ballpack_tree.c`, Built Unconditionally by `*_init`)

| Codec | Hot Compact Base + LUT (`hot_compact_base_bytes`) | Hot Compact Reciprocals (`hot_compact_recip_bytes`) | **Hot Compact Working Set** (`base + recip`) | Fast Extra Ckpt Data (`fast_extra_ckpt_data_bytes`) | Fast Extra Ckpt Meta (`fast_extra_ckpt_meta_bytes`) | **Hot Fast / Batch4 Working Set** | Rank-Only Inverse Grids (`rank_only_inverse_bytes`) | Init-Unused Aux (`init_unused_aux_bytes`) | **Total Heap Allocated at Init** (+ `Ctx` struct) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`Z^32` 136b (`B=783`)** | `59,804` B | `112,896` B | **`172,700` B** (`168.7 KiB`) | `944,064` B | `18,816` B | **`1,135,580` B** (`1,109.0 KiB`) | `12,100` B | `14,112` B | **`1,161,792` B** (+ `400` B) |
| **`E_8^4` 136b (`m=391`)** | `59,986` B | `93,984` B | **`153,970` B** (`150.4 KiB`) | `798,208` B | `18,792` B | **`970,970` B** (`948.2 KiB`) | `49,284` B | `6,272` B | **`1,026,526` B** (+ `656` B) |
| **`E_8^4` 88b (`m=48`)** | `7,458` B | `11,664` B | **`19,122` B** (`18.7 KiB`) | `15,720` B | `2,328` B | **`37,170` B** (`36.3 KiB`) | `6,084` B | `784` B | **`44,038` B** (+ `656` B) |

**Exact Notes on Bucket Membership & Reconciliation with `AUDIT-112` Python Count/LUT Totals:**
- **Why `Z^32` `hot_compact_base_bytes = 59,804` B differs from `AUDIT-112`'s `70,780` B count+LUT figure:** `AUDIT-112` sums theoretical minimal-width 1D count arrays + `Z^2` LUT only (excluding all `112,896` B of reciprocals), counting `s16` (`12,544` B, allocated in `ztree_init` under `init_unused_aux_bytes` alongside `z2_nz_r` `1,568` B, unread by unrank or rank) and sizing `s4` as `uint16_t` (`1,568` B) whereas `ballpack_tree.c` stores `s4` as `uint32_t` (`3,136` B): `59,804 + 12,544 - 1,568 = 70,780` B. Including `s16`, C base + reciprocals = `185,244` B (vs. `70,780 + 112,896 = 183,676` B with `uint16_t` `s4`).
- **Why `E_8^4` `hot_compact_base_bytes` includes `even_c8` and `e8_{even,odd}_p0_tot`:** Both `e8tree_unrank_compact` and `e8tree_unrank_fast` read `even_c8` (`3,136` B at `m=391`, `392` B at `m=48`) and `e8_even_p0_tot`/`e8_odd_p0_tot` (`6,272` B at `m=391`, `784` B at `m=48`). Meanwhile `s_blk2` (`6,272` B at `m=391`, `784` B at `m=48`) is allocated at init under `init_unused_aux_bytes` and unread at runtime. At `m=48`, the unified C `E8Ctx` also uses `u192`/`u128`/`uint64_t` outer arrays (`+1,576` B vs. `m=48`-specialized minimal widths in `AUDIT-112`'s `5,490` B count+LUT floor).
- **Important init allocation note:** In the current research harness, `ztree_init` and `e8tree_init` allocate compact tables, fast checkpoints, and rank-only inverse grids unconditionally in a single context (`total_heap_init_bytes`). A decode-only compact deployment would omit checkpoints, rank inverse grids, and unused auxiliary arrays, retaining only `hot_compact_total_bytes`.

### 3.2 Baseline Comparison Ratios & Cache Fit (`32 KiB` L1d / `512 KiB` L2 / `4 MiB` L3)

| Codec / Variant | Baseline Table | Hot Unrank Working Set | Ratio vs Baseline (Hot Unrank) | Total Init Heap (All Modes + Rank) | Ratio vs Baseline (Total Init) | Fit vs Host Cache (`32 KiB` L1d / `512 KiB` L2 / `4 MiB` L3) |
|---|---:|---:|---:|---:|---:|---|
| Baseline `ballpack.c` (`Z^32` 136b) | `620,928` B (`606.4 KiB`) | `620,928` B | `1.00×` | `620,928` B | `1.00×` | **Spills `512 KiB` L2** -> L3 |
| Baseline `BallIndex` (`E_8^4` 136b, Python DP) | `11,189,304` B (`10.67 MiB`) | `11,189,304` B | `1.00×` | `11,189,304` B | `1.00×` | **Spills `4 MiB` L3** -> DRAM (no C11 implementation in upstream) |
| Baseline `BallIndex` (`E_8^4` 88b, Python DP) | `1,376,760` B (`1.31 MiB`) | `1,376,760` B | `1.00×` | `1,376,760` B | `1.00×` | **Spills `512 KiB` L2** -> L3 |
| `ztree_compact` (`Z^32` 136b) | `620,928` B | **`172,700` B** (`168.7 KiB`) | **`3.60×` smaller** | `1,161,792` B | `0.53×` (`1.87×` larger) | Hot compact **fits `512 KiB` L2** (spills `32 KiB` L1d) |
| `ztree_fast` / `batch4` (`Z^32` 136b) | `620,928` B | **`1,135,580` B** (`1,109.0 KiB`) | **`0.55×`** (`1.83×` larger) | `1,161,792` B | `0.53×` (`1.87×` larger) | **Spills `512 KiB` L2** -> L3 |
| `e8tree_compact` (`E_8^4` 136b, `m=391`) | `11,189,304` B | **`153,970` B** (`150.4 KiB`) | **`72.67×` smaller** | `1,026,526` B | **`10.90×` smaller** | Hot compact **fits `512 KiB` L2** (spills `32 KiB` L1d) |
| `e8tree_fast` / `batch4` (`E_8^4` 136b, `m=391`) | `11,189,304` B | **`970,970` B** (`948.2 KiB`) | **`11.52×` smaller** | `1,026,526` B | **`10.90×` smaller** | **Spills `512 KiB` L2** -> L3 |
| `e8tree_compact` (`E_8^4` 88b, `m=48`) | `1,376,760` B | **`19,122` B** (`18.7 KiB`) | **`72.00×` smaller** | `44,038` B | **`31.26×` smaller** | Hot compact **fits `32 KiB` L1d** (`19,122 <= 32,768` B) |
| `e8tree_fast` / `batch4` (`E_8^4` 88b, `m=48`) | `1,376,760` B | **`37,170` B** (`36.3 KiB`) | **`37.04×` smaller** | `44,038` B | **`31.26×` smaller** | **Does NOT fit `32 KiB` L1d** (`37,170 > 32,768` B -> spills to L2) |

---

## 4. Empirical Scalar-CPU Throughput Results (`100,000` Blocks = `3.2M` Weights/Trial, `7` Trials)

### 4.1 Primary Profile: `gcc -O3 -march=native` (`GCC 16.2.1`)

| Kernel / Variant | Threads | CPU Throughput/Core Median [Min..Max] (`M w/s/c`) | CPU `ns/weight` | Speedup vs `ballpack.c` (CPU/core) | Wall-Clock Median [Min..Max] (`M w/s`, loaded host) | Gap vs `Q4_K_M` (`45,000 M w/s/c`) |
|---|---:|---:|---:|---:|---:|---:|
| `baseline_ballpack_c` (`Z^32` 136b) | 1 | `18.441` [`16.664`..`19.823`] | `54.226` | `1.00×` | `1.192` [`0.936`..`2.139`] | `2,440.2×` |
| `baseline_ballpack_c` (`Z^32` 136b) | 2 | `20.921` [`20.244`..`21.340`] | `47.800` | `1.13×` | `10.514` [`4.402`..`13.053`] | `2,151.0×` |
| `ztree_compact` (`Z^32` 136b, `172.7 KB` hot) | 1 | `21.676` [`21.314`..`22.428`] | `46.133` | `1.18×` | `2.364` [`1.528`..`2.932`] | `2,076.0×` |
| **`ztree_fast` (`Z^32` 136b, `1,135.6 KB` hot)** | 1 | **`41.035`** [`38.979`..`45.345`] | **`24.369`** | **`2.23×`** | `13.115` [`8.215`..`22.674`] | `1,096.6×` |
| **`ztree_fast` (`Z^32` 136b)** | 2 | **`46.795`** [`43.620`..`47.751`] | **`21.370`** | **`2.54×`** | `32.691` [`16.458`..`42.728`] | `961.6×` |
| `ztree_batch4` (`Z^32` 136b) | 1 | `40.778` [`37.308`..`41.929`] | `24.523` | `2.21×` | `10.749` [`8.251`..`20.439`] | `1,103.5×` |
| `ztree_batch4` (`Z^32` 136b) | 2 | `43.983` [`39.127`..`45.765`] | `22.736` | `2.39×` | `29.098` [`10.106`..`42.552`] | `1,023.1×` |
| `e8tree_compact` (`E_8^4` 136b, `154.0 KB` hot) | 1 | `21.970` [`21.259`..`22.475`] | `45.516` | `1.19×` | `2.891` [`2.556`..`7.758`] | `2,048.2×` |
| **`e8tree_fast` (`E_8^4` 136b, `971.0 KB` hot)** | 1 | **`32.404`** [`30.890`..`33.366`] | **`30.861`** | **`1.76×`** | `4.668` [`2.575`..`7.166`] | `1,388.7×` |
| **`e8tree_fast` (`E_8^4` 136b)** | 2 | **`34.975`** [`30.665`..`36.981`] | **`28.592`** | **`1.90×`** | `19.862` [`9.616`..`26.759`] | `1,286.6×` |
| `e8tree_batch4` (`E_8^4` 136b) | 1 | `31.339` [`21.310`..`34.020`] | `31.909` | `1.70×` | `4.860` [`3.000`..`9.320`] | `1,435.9×` |
| `e8tree_batch4` (`E_8^4` 136b) | 2 | `33.984` [`29.028`..`35.521`] | `29.426` | `1.84×` | `16.314` [`5.387`..`20.192`] | `1,324.2×` |
| **`e8tree_compact` (`E_8^4` 88b, `19.1 KB` hot)** | 1 | **`58.106`** [`56.636`..`58.683`] | **`17.210`** | **`3.15×`** | `8.726` [`6.081`..`11.708`] | `774.4×` |
| `e8tree_fast` (`E_8^4` 88b, `37.2 KB` hot) | 1 | `54.791` [`52.200`..`55.919`] | `18.251` | `2.97×` | `5.838` [`2.974`..`9.007`] | `821.3×` |
| `e8tree_fast` (`E_8^4` 88b) | 2 | `56.176` [`55.173`..`57.634`] | `17.801` | `3.05×` | `22.790` [`13.764`..`36.553`] | `801.1×` |
| `e8tree_batch4` (`E_8^4` 88b) | 1 | `54.832` [`49.013`..`56.268`] | `18.237` | `2.97×` | `7.364` [`6.702`..`13.307`] | `820.7×` |
| `e8tree_batch4` (`E_8^4` 88b) | 2 | `57.627` [`56.401`..`57.804`] | `17.353` | `3.12×` | `35.351` [`31.519`..`52.592`] | `780.9×` |

### 4.2 Cross-Compiler Parity (`gcc -O3` Portable vs `clang -O3` Portable, 1-Thread CPU/Core Median)

| Kernel / Variant | `gcc -O3 -march=native` (`M w/s/c`) | `gcc -O3` Portable (`M w/s/c`) | `clang -O3` Portable (`M w/s/c`) | Cross-Compiler Observation |
|---|---:|---:|---:|---|
| `baseline_ballpack_c` (`Z^32` 136b) | `18.441` (`1.00×`) | `18.188` (`1.00×`) | `23.114` (`1.00×`) | Clang 23 generates tighter 192-bit `sub`/`sbb` chains for `ballpack.c` (`+25%` over GCC). |
| `ztree_compact` (`Z^32` 136b) | `21.676` (`1.18×`) | `21.646` (`1.19×`) | `21.771` (`0.94×`) | Beats GCC baseline with `172.7 KB` hot working set (`3.60×` smaller than `620.9 KB`). |
| `ztree_fast` (`Z^32` 136b) | **`41.035`** (**`2.23×`**) | **`36.559`** (**`2.01×`**) | `29.329` (**`1.27×`**) | GCC schedules branchless `cmov` + multiply-shift faster than Clang on this host. |
| `ztree_batch4` (`Z^32` 136b) | `40.778` (`2.21×`) | `37.479` (`2.06×`) | **`31.218`** (**`1.35×`**) | Under Clang, 4-block ILP adds `+6.4%` over single-block `ztree_fast`. |
| `e8tree_compact` (`E_8^4` 136b) | `21.970` (`1.19×`) | `21.327` (`1.17×`) | `23.578` (`1.02×`) | Matches/beats `Z^32` baseline with `154.0 KB` hot working set (`72.67×` vs `BallIndex`). |
| `e8tree_fast` (`E_8^4` 136b) | **`32.404`** (**`1.76×`**) | **`29.863`** (**`1.64×`**) | `29.739` (**`1.29×`**) | Beats `Z^32` baseline across all 3 compiler profiles (`971.0 KB` hot working set). |
| `e8tree_batch4` (`E_8^4` 136b) | `31.339` (`1.70×`) | `29.662` (`1.63×`) | **`30.124`** (**`1.30×`**) | Small Clang gain over single-block `e8tree_fast`. |
| `e8tree_compact` (`E_8^4` 88b) | **`58.106`** (**`3.15×`**) | **`57.891`** (**`3.18×`**) | **`64.786`** (**`2.80×`**) | Fastest single-core kernel (`15.44 ns/weight`); `19,122` B hot set fits `32 KiB` L1d. |
| `e8tree_fast` (`E_8^4` 88b) | `54.791` (`2.97×`) | `54.828` (`3.01×`) | `59.513` (`2.57×`) | Slower than `e8tree_compact` on all 3 compilers; `37,170` B hot set exceeds `32 KiB` L1d. |
| `e8tree_batch4` (`E_8^4` 88b) | `54.832` (`2.97×`) | `52.319` (`2.88×`) | `56.816` (`2.46×`) | Same `37,170` B hot set (> `32 KiB` L1d). |

---

## 5. Remaining Integration & Evaluation Scope (`siliquant#23` & `#25`)

1. **Integration Work Still Needed for `siliquant#25` (`E_8^4` Format Support):**
   - `c-bench/ballpack_tree.c` is an exact, standalone research C11 rank/unrank implementation (tested on `100,000` blocks per codec with `0` mismatches and `0` hardware divisions).
   - It is **not** yet wired into `shape/ballpack.py`, a packed-file reader/writer, scale-code dequantization (`zb_decode`-style float output), or a decode-only initialization mode (splitting out checkpoints and rank-only inverse grids when only compact unranking is desired).
2. **Open Performance Questions for `siliquant#23`:**
   - On scalar CPU, zero-division binary-tree unranking achieves `1.27×`–`3.18×` over `ballpack.c` (`Z^32` 136b) on this Zen-1 host, but remains `~700×`–`1,390×` slower than `siliquant#23`'s `45,000 M w/s/c` `Q4_K_M` falsifier.
   - This harness makes **no performance claims** for SIMD (AVX2/AVX-512/NEON), GPU warps, or TPU kernels; evaluating whether the reduced depth (`32 -> 2..4`), zero-division arithmetic, and compact table footprints (`19.1 KB` for `E_8^4` 88b, `154.0 KB` for `E_8^4` 136b, `172.7 KB` for `Z^32` 136b) translate into competitive accelerator throughput requires dedicated target-hardware kernels and measurements.

---

## 6. Reproducibility Commands

All shared libraries are compiled ephemerally inside a `0700` `tempfile.TemporaryDirectory()` and deleted automatically on exit (zero binary artifacts left in the directory). Pass `--baseline-ballpack-c <path>` pointing to PR `#12`'s `shape/ballpack.c` (`sha256: 53074fbadf169650da6cad400b986e07b1ea15df180f6f148f97ef4761b33055`).

```bash
# 1. Deterministic verification-only mode (3 compiler profiles, 0-warning build, objdump 0-div proof,
#    live C sizeof footprint partition checks, exactness vs ballpack.c & AUDIT-112 Python oracle):
python3 -I -O -B -W error c-bench/run_ballpack_c_bench.py \
  --baseline-ballpack-c <path-to-pr12-shape-ballpack.c> --json

# 2. Full 100,000-block (3,200,000-weight/codec), 7-trial dual-clock scalar-CPU benchmark:
python3 -I -O -B -W error c-bench/run_ballpack_c_bench.py \
  --baseline-ballpack-c <path-to-pr12-shape-ballpack.c> \
  --write-results c-bench/bench_results.json
```
