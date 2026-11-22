# Zero-Division Binary-Tree Shell-Convolution Unranking & Packed `E_8^4` C11 Codec (`AUDIT-112R` + `Track S15b`)

Companion exact-math verification study and empirical C11 microbenchmark for `Moissanite-Labs/siliquant#23` and `Moissanite-Labs/siliquant#25` (pinned against PR `#12` @ `20d13d5fa9c5fb2333b8faa0360a90caa1e0a43f`).

Pure Python 3 stdlib (`int` / `fractions.Fraction`, zero floats on trust paths, zero bare `assert`s) + pure C11 (`-std=c11 -Wall -Wextra -Werror`, zero warnings under `gcc` and `clang`, zero hardware `div`/`idiv` instructions on every hot unranking path).

## Key Results

1. **Intra-Block Parallelism Without Shaping Loss (`AUDIT-112R` Pillar 1 & 3):**
   - Replaces PR `#12` `ballpack.c`'s 32-step sequential 192-bit coordinate chain with an exact **binary-tree shell-convolution bijection** (`32 -> 16 -> 8 -> 4 -> 2 -> L1 LUT`), reducing intra-block critical-path depth **`8.00×` (`32 -> 4`)**, shrinking the compact table **`8.77×`** (`620,928` B -> `70,780` B), and certifying **`4,968/4,968`** universal Granlund-Montgomery / Barrett zero-division reciprocals over the exact `Z^32` ball (`B = 783`, `W_32(783) = 86775110168433561078044359892823273950209`, **`0.00%` shaping loss** vs. `+4.51%` / `+10.40%` / `+17.72%` for naive independent `16D` / `8D` / `4D` sub-blocks).
2. **First Working Packed Binary C11 `E_8^4` Codec (`AUDIT-112R` Pillar 2 + `c-bench/`):**
   - Reduced-norm `E_8^4` outer tree + inner `8 -> 4 -> 2` parity-tracked LUT cuts critical-path depth **`16.00×` (`32 -> 2` to 4 independent `E_8` blocks)** and collapses `BallIndex`'s DP table by **`196.8×`** at `136`b (`4.500` bpw, `m_max = 391`: `11.19 MB` -> **`56,850` B**) and **`250.7×`** at `88`b (`3.000` bpw, `m_max = 48`: `1.38 MB` -> **`5,490` B L1d-resident**).
   - Verified **`0 / 300,000` block mismatches (`9,600,000` weights)** across `Z^32 (136b)` (including `0 / 100,000` set-membership mismatches against `ballpack.c:zb_rank/zb_unrank`), `E_8^4 (136b)`, and `E_8^4 (88b)` (`800,000` $E_8$ sub-blocks), plus `256/256` exact parity vs. the pure-$\mathbb{Z}$ Python oracle and `0` `div`/`idiv` instructions in `objdump -d` across all 6 hot unrankers under all 3 compiler profiles.
3. **Honest Empirical C11 Microbenchmark (`c-bench/BENCH_REPORT.md` & `c-bench/bench_results.json`):**
   - Measured over `100,000` full-width blocks (`3.2M` weights/trial) × `7` trials on an `AMD Athlon Silver 3050U` (`2` Zen-1 cores, `GCC 16.2.1` & `Clang 23.1.3`) with dual-clock instrumentation (`CLOCK_THREAD_CPUTIME_ID` per-core CPU time, `< 2.5%` spread, + `CLOCK_MONOTONIC` wall-clock with disclosed host loadavg):
     - Baseline `ballpack.c` (`Z^32` 136b): `20.86` (`gcc -O3 -march=native`) / `19.72` (`gcc -O3`) / `27.29 M w/s/c` (`clang -O3`).
     - `ztree_fast` (`Z^32` 136b): **`46.32`–`48.48 M w/s/c`** (**`2.22×`–`2.46×` GCC**, `1.34×`–`1.41×` Clang).
     - `e8tree_fast` / `batch4` (`E_8^4` 136b): **`36.81`–`37.71 M w/s/c`** (**`1.76×`–`1.91×` faster than `Z^32` `ballpack.c` on GCC**, `1.20×`–`1.26×` Clang).
     - `e8tree_compact` (`E_8^4` 88b, `5.49 KB` L1d): **`58.04`–`65.33 M w/s/c`** (`15.31 ns/weight`, **`2.39×`–`2.99×` faster**; peak 2-core wall `89.35 M w/s`).
   - **Honest Law-2/Law-4 Verdict vs. `#23` `Q4_K_M` Falsifier (`45,000 M w/s/c`):** **NOT MET on scalar CPU** (`689×`–`1,233×` slower than `Q4_K_M` on a single scalar Zen-1 core). The real architectural unlock is eliminating serial depth (`32 -> 2..4`), zeroing hardware divisions, and shrinking tables to SRAM-resident `5.5`–`70.8 KB` for SIMD/GPU-warp/TPU-VREG dequantization — plus delivering an immediate, ready-to-use C11 `E_8^4` packed file codec for `#25`.

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
```

To re-run `--live` against PR `#12` (`shape/{ballindex.py,ballpack.c,ballpack.py}` @ `20d13d5fa9c5fb2333b8faa0360a90caa1e0a43f`), pass `--intake-dir <pr12-checkout>` to `ballpack_tree_audit.py --live`. To re-run the full head-to-head C11 benchmark (`c-bench/run_ballpack_c_bench.py --write-results ...`), place sibling directories `track-s15-ballpack-tree-unrank/` (this study's 5 top-level files) and `intake-siliquant-ballpack/` (PR `#12` `shape/` files) adjacent to `c-bench/`, or inspect the pinned reproducible outputs in [`c-bench/bench_results.json`](c-bench/bench_results.json) and [`c-bench/BENCH_REPORT.md`](c-bench/BENCH_REPORT.md).

See [`BALLPACK_TREE_AUDIT_REPORT.md`](BALLPACK_TREE_AUDIT_REPORT.md) and [`c-bench/BENCH_REPORT.md`](c-bench/BENCH_REPORT.md) for full derivations, exact counts, and hardware measurements.
