# Zero-Division Binary-Tree Shell-Convolution Rank/Unrank Study (`AUDIT-112R` + `Track S15b`)

Companion exact-math verification study and empirical scalar-CPU C11 rank/unrank microbenchmark for `Moissanite-Labs/siliquant#23` and `Moissanite-Labs/siliquant#25` (pinned against PR `#12` @ `20d13d5fa9c5fb2333b8faa0360a90caa1e0a43f`).

**Scope:** Pure Python 3 stdlib (`int` / `fractions.Fraction`, zero floats on trust paths, zero bare `assert`s) + an **exact research C11 rank/unrank implementation and scalar-CPU benchmark harness** (`-std=c11 -Wall -Wextra -Werror`, zero warnings under `gcc` and `clang`, zero hardware `div`/`idiv` instructions on every hot unranking path). The C11 implementation is **not yet wired into a packed-file reader/writer, format integration, or production binding**, and makes **no GPU, TPU, or SIMD performance claims** extrapolated from the scalar CPU harness.

## Key Results

1. **Intra-Block Depth Reduction Without Shaping Loss (`AUDIT-112R` Pillar 1 & 3):**
   - Replaces PR `#12` `ballpack.c`'s 32-step sequential 192-bit coordinate chain with an exact **binary-tree shell-convolution bijection** (`32 -> 16 -> 8 -> 4 -> 2 -> LUT`), reducing intra-block critical-path depth **`8.00×` (`32 -> 4`)** and certifying **`4,968/4,968`** universal Granlund–Montgomery / Barrett zero-division reciprocals over the exact `Z^32` ball (`B = 783`, `W_32(783) = 86775110168433561078044359892823273950209`, **`0.00%` shaping loss** vs. `+4.51%` / `+10.40%` / `+17.72%` for naive independent `16D` / `8D` / `4D` sub-blocks).
2. **Exact Research C11 `E_8^4` Rank/Unrank Implementation (`AUDIT-112R` Pillar 2 + `c-bench/`):**
   - Reduced-norm `E_8^4` outer tree + inner `8 -> 4 -> 2` parity-tracked pair LUT cuts critical-path depth **`16.00×` (`32 -> 2` outer splits to 4 independent `E_8` blocks)** at `136`b (`m_max = 391`) and `88`b (`m_max = 48`).
   - Verified **`0 / 300,000` block mismatches (`9,600,000` weights)** across `Z^32 (136b)` (including `0 / 100,000` set-membership mismatches against `ballpack.c:zb_rank/zb_unrank`), `E_8^4 (136b)`, and `E_8^4 (88b)` (`800,000` $E_8$ sub-blocks), plus `256/256` exact parity vs. the pure-$\mathbb{Z}$ Python oracle and `0` `div`/`idiv` instructions in `objdump -d` across all 6 hot unrankers under all 3 compiler profiles.
3. **Exact C11 `sizeof`/Allocation Memory Footprints (`c-bench/bench_results.json`, `c-bench/BENCH_REPORT.md`):**
   - Measured directly from `malloc`/`calloc` allocations inside `ztree_init` and `e8tree_init` (including multiply-shift reciprocals, `even_c8`, branch totals, checkpoint metadata, rank-only inverse grids, and init-unused auxiliary arrays):
     - **`Z^32` (`136`b, vs. `ballpack.c` `620,928` B):** hot `ztree_compact` working set **`172,700` B** (`3.60×` smaller; `59,804` B base/LUT + `112,896` B reciprocals; `AUDIT-112` count+LUT floor alone is `70,780` B); hot `ztree_fast`/`batch4` working set **`1,135,580` B** (`+944,064` B ckpt data `+18,816` B ckpt meta, `1.83×` larger than `ballpack.c`); total unconditional `ztree_init` heap **`1,161,792` B** (incl. `12,100` B rank-only inverse grids and `14,112` B unused init arrays).
     - **`E_8^4` (`136`b `m=391`, vs. `BallIndex` `11,189,304` B):** hot `e8tree_compact` working set **`153,970` B** (`72.67×` smaller; `59,986` B base/LUT/`even_c8`/`p0_tot` + `93,984` B reciprocals; `AUDIT-112` count+LUT floor alone is `56,850` B); hot `e8tree_fast`/`batch4` working set **`970,970` B** (`11.52×` smaller); total unconditional `e8tree_init` heap **`1,026,526` B** (`10.90×` smaller, incl. `49,284` B rank-only inverse grids and `6,272` B unused `s_blk2`).
     - **`E_8^4` (`88`b `m=48`, vs. `BallIndex` `1,376,760` B):** hot `e8tree_compact` working set **`19,122` B** (`72.00×` smaller; **fits `32 KiB` L1d**; `AUDIT-112` count+LUT floor alone is `5,490` B); hot `e8tree_fast`/`batch4` working set **`37,170` B** (`37.04×` smaller; **does NOT fit `32 KiB` L1d** — spills to L2); total unconditional `e8tree_init` heap **`44,038` B** (`31.26×` smaller, incl. `6,084` B rank-only inverse grids and `784` B unused `s_blk2`).
4. **Honest Empirical Scalar-CPU Microbenchmark (`c-bench/BENCH_REPORT.md` & `c-bench/bench_results.json`):**
   - Measured over `100,000` full-width blocks (`3.2M` weights/trial) × `7` trials on an `AMD Athlon Silver 3050U` (`2` Zen-1 cores, `GCC 16.2.1` & `Clang 23.1.3`) with dual-clock instrumentation (`CLOCK_THREAD_CPUTIME_ID` per-core CPU time + `CLOCK_MONOTONIC` wall-clock with disclosed host loadavg; all timings are host-specific to this scalar CPU):
     - Baseline `ballpack.c` (`Z^32` 136b, 1t): `18.441` (`gcc -O3 -march=native`) / `18.188` (`gcc -O3`) / `23.114 M w/s/c` (`clang -O3`).
     - `ztree_fast` / `batch4` (`Z^32` 136b): **`41.04`–`46.80 M w/s/c`** (`2.23×`–`2.54×` GCC native; `2.01×`–`2.06×` GCC portable; `1.27×`–`1.35×` Clang at `29.33`–`31.22 M w/s/c`).
     - `e8tree_fast` / `batch4` (`E_8^4` 136b): **`31.34`–`34.98 M w/s/c`** (`1.70×`–`1.90×` GCC native; `1.63×`–`1.64×` GCC portable; `1.29×`–`1.30×` Clang at `29.74`–`30.12 M w/s/c`).
     - `e8tree_compact` (`E_8^4` 88b, `19.1 KB` hot L1d-resident): **`57.89`–`64.79 M w/s/c`** (`15.44 ns/weight` on Clang, `2.80×`–`3.18×` vs `ballpack.c`), beating `e8tree_fast` (`37.2 KB` hot > `32 KiB` L1d) across all 3 compilers.
   - **Honest Law-2/Law-4 Verdict vs. `#23` `Q4_K_M` Falsifier (`45,000 M w/s/c`):** **NOT MET on scalar CPU** (`695×`–`1,389×` slower than `Q4_K_M` on this single scalar Zen-1 core). Whether non-scalar SIMD/GPU/TPU kernels or packed-file format integration can close that gap is unmeasured here.

## Offline Reproduction

### 1. Exact Pure-$\mathbb{Z}/\mathbb{Q}$ Oracle, 4-Mode `-I` Parity & R5 Suite (`0` External Files)

```bash
cd studies/expedition-ballpack-tree-unrank-audit112r
python3 -I ballpack_tree_audit.py --standalone --json | sha256sum
python3 -I -B ballpack_tree_audit.py --standalone --json | sha256sum
python3 -I -O -B ballpack_tree_audit.py --standalone --json | sha256sum
python3 -I -O -B -W error ballpack_tree_audit.py --standalone --json | sha256sum
# Expected 4-mode SHA-256: ceeec4ec67060f2da071e11aae4642bbd7a0067c3ff5d938b37ec6751111584a

python3 -I -O -B -W error ballpack_tree_audit.py --check ballpack_tree_certificate.json
python3 -I -O -B -W error test_ballpack_tree_r5.py -v
```

### 2. Standalone C11 Zero-Warning Build Check & Live Replay Against PR `#12` (`20d13d5`)

```bash
# Zero-warning C11 build check (compiles to ephemeral /tmp path):
cc -O3 -march=native -Wall -Wextra -Werror -std=c11 -shared -fPIC -pthread \
  -o /tmp/libballpack_tree.so c-bench/ballpack_tree.c && rm -f /tmp/libballpack_tree.so

# Deterministic verification + live C sizeof footprint check against pinned PR #12 shape/ballpack.c:
python3 -I -O -B -W error c-bench/run_ballpack_c_bench.py \
  --baseline-ballpack-c <path-to-pr12-shape-ballpack.c> --json
```

To re-run `--live` against PR `#12` (`shape/{ballindex.py,ballpack.c,ballpack.py}` @ `20d13d5fa9c5fb2333b8faa0360a90caa1e0a43f`), pass `--intake-dir <pr12-checkout>` to `ballpack_tree_audit.py --live`. To re-run the full head-to-head C11 benchmark, pass `--baseline-ballpack-c <path-to-pr12-shape-ballpack.c> --write-results c-bench/bench_results.json` to `c-bench/run_ballpack_c_bench.py`, or inspect the pinned reproducible outputs in [`c-bench/bench_results.json`](c-bench/bench_results.json) and [`c-bench/BENCH_REPORT.md`](c-bench/BENCH_REPORT.md).

See [`BALLPACK_TREE_AUDIT_REPORT.md`](BALLPACK_TREE_AUDIT_REPORT.md) and [`c-bench/BENCH_REPORT.md`](c-bench/BENCH_REPORT.md) for full derivations, exact counts, C11 `sizeof` footprint breakdowns, and scalar-CPU measurements.
