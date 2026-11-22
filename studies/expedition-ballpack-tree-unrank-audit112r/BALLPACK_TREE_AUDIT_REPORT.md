# AUDIT-112 — Zero-Overhead Binary-Tree Shell-Convolution Rank/Unrank Study for `Moissanite-Labs/siliquant#23/#25`

- **Audit ID:** `AUDIT-112` (Track S15)
- **Schema Version:** `audit112.ballpack_tree_unrank.v1`
- **Honest Law-4 Novelty Classification:** **`N1`** *(exact engineering/cryptographic-grade pure-$\mathbb{Z}/\mathbb{Q}$ zero-division binary-tree certifier, hardware bit-width/LUT bounds, and `E_8^4` reduced-norm tree rank/unrank decomposition)* over **`N0`** *(classical enumerative sphere shaping Cover 1973, shell-convolution divide-and-conquer coding, Granlund–Montgomery 1994 exact integer reciprocal multiplication, and Jacobi $E_8$ theta series $240\sigma_3(m)$)*
- **Pinned Read-Only Primary Source (`Moissanite-Labs/siliquant#12 @ 20d13d5fa9c5fb2333b8faa0360a90caa1e0a43f`, `0` edits, `0` external writes):**
  - `shape/ballindex.py`: `bec9e571ab77c08a18f2ccfd94c1c3fc5321dde3e1bbb1f1a57a3ce590ab05b7` (`9,746` bytes)
  - `shape/ballpack.c`: `53074fbadf169650da6cad400b986e07b1ea15df180f6f148f97ef4761b33055` (`6,194` bytes)
  - `shape/ballpack.py`: `b804aae680baf9e9842e6bea6971f26ff1bbddad961fdd95dca41366438fa4d0` (`18,607` bytes)
- **Executable Artifacts (100% Pure Python 3 Stdlib, Exact `int`/`Fraction`, `0` Floats, `0` `math.sqrt`, `0` Bare `assert`s, `str`-Key Canonical JSON):**
  - [ballpack_tree_oracle.py](ballpack_tree_oracle.py) — exact pure-$\mathbb{Z}/\mathbb{Q}$ oracle across Pillars 1–3.
  - [ballpack_tree_audit.py](ballpack_tree_audit.py) — dual-mode (`--standalone` default offline replay, `--live` read-only intake recount, `--check <cert.json>` fail-closed Zone-B validator).
  - [test_ballpack_tree_r5.py](test_ballpack_tree_r5.py) — assertion-free `-I -O -B -W error` R5 mutation-killing, AST `BoolOp`-killing, and anti-bypass suite.
  - [ballpack_tree_certificate.json](ballpack_tree_certificate.json) — canonical zero-float live JSON certificate (`live_intake_audited: true`, `overall_verified: true`).

---

## 1. Problem Grounding (`siliquant#23` & `#25`)

On `Moissanite-Labs/siliquant#23` and `#25`, the author of `shape/ballpack.c` (`siliquant#12 @ 20d13d5f`) identified two exact bottlenecks for packed spherical-ball inference:
1. **Within-block serial dependency (`PZ32:b136s8`):** `ballpack.c:unrank_one` unpacks a `136`-bit (`LIMBS = 3`, `192`-bit) block rank across `n = 32` coordinates one weight at a time (`for (int i = 0; i < n; ++i)`), with each `192`-bit subtract/compare loop strictly dependent on the prior coordinate. At `20–25M` weights/s on one CPU core vs. `llama.cpp`'s `Q4_K_M` (`120 × 1.5B / 4 = 45B` weights/s/core), serial decoding lags by **`1,800×` to `2,250×`**, far below the **`6.55%–7.11%`** memory-bandwidth tokens/sec ceiling.
2. **Missing `E_8` C rank/unrank implementation:** `shape/ballpack.py:Format` hardcodes `BallIndex("Z8", ...)` and `shape/ballpack.c` has no `E_8` support; meanwhile `BallIndex("E8", 32, 782)` requires **`466,221`** `192`-bit table entries (**`11,189,304` bytes / `10.67 MiB`**) in its coordinate-by-coordinate DP table.

---

## 2. Pillar 1 — Exact Zero-Overhead Binary-Tree Shell-Convolution Bijection for $\mathbb{Z}^{32}$ (`32 -> 16 -> 8 -> 4 -> 2 -> LUT`)

Using the exact integer shell-convolution identities:
$$W_{2m}(b) = \sum_{r_L = 0}^{b} S_m(r_L)\, W_m(b - r_L), \qquad S_{2m}(r) = \sum_{r_L = 0}^{r} S_m(r_L)\, S_m(r - r_L),$$
we construct a 1-to-1 bijection onto the **exact same spherical integer ball** $\{y \in \mathbb{Z}^{32} : \|y\|_2^2 \le 783\}$:

- **Exact `0.000` Bits Rate Overhead:** Codebook cardinality matches `BallIndex("Z8", 32, 783)` to all 41 decimal digits:
  $$W_{32}(783) = \mathbf{86{,}775{,}110{,}168{,}433{,}561{,}078{,}044{,}359{,}892{,}823{,}273{,}950{,}209} \le 2^{136} < W_{32}(784).$$
- **Critical-Path Depth Collapse (`32 -> 4`, `8.00×` Reduction) & Exact Hardware Bit-Width Drop:** Only **1** root split touches `192`-bit arithmetic; every child drops rapidly to native `uint64`, `uint32`/`uint16`, and **`16` parallel `O(1)` `Z^2` LUT lookups**:

| Tree Level | Node Dim | Parallel Nodes | Max Shell Bits (`HW`) | Max Ball Bits (`HW`) | Max GM Multiplier Bits | Count/LUT Bytes Only (`AUDIT-112`) |
|---|---|---|---|---|---|---|
| Level 0 (Root) | `n = 32` | `1` | `131` (`uint192`) | `136` (`uint192`) | `130` (`uint192`) | `18,840` B (`C_root`) |
| Level 1 | `n = 16` | `2` | `69` (`uint128`) | `75` (`uint128`) | `68` (`uint128`) | `25,088` B (`S_16, W_16`) |
| Level 2 | `n = 8` | `4` | `34` (`uint64`) | `41` (`uint64`) | `34` (`uint64`) | `12,544` B (`S_8, W_8`) |
| Level 3 | `n = 4` | `8` | `14` (`uint16`) | `22` (`uint32`) | `17` (`uint32`) | `4,704` B (`S_4, W_4`) |
| Level 4 (Leaves) | `n = 2` | `16` | `5` (`uint16`) | `12` (`uint16`) | `O(1)` LUT (`0` splits) | `3,136` B + `6,468` B LUT |

- **Exact Table Memory Accounting (Count/LUT Floor vs. Complete C11 Runtime Working Set):** The entire $\mathbb{Z}^2$ ball up to $r \le 783$ contains only **`2,449` integer pairs** (`4,898` bytes as `int8[2]` + `1,570` bytes `uint16` offsets = **`6,468` bytes**). In the pure-$\mathbb{Z}$ Python certificate (`ballpack_tree_certificate.json`), `tree_total_bytes = 70,780` B (`155232/17695 = 8.77×` vs. `ballpack.c`'s `620,928` B) counts **only** minimal-width 1D count arrays + `Z^2` LUT and **excludes** runtime multiply-shift reciprocal tables (`112,896` B in C11), optional strided checkpoints (`944,064` B data + `18,816` B metadata), and rank-only inverse lookup grids (`12,100` B). In the actual `sizeof`-measured C11 implementation (`c-bench/BENCH_REPORT.md`), the hot `ztree_unrank_compact` working set is **`172,700` B** (**`3.60×`** smaller than `620,928` B; `59,804` B read base/LUT + `112,896` B reciprocals), the hot `ztree_unrank_fast`/`batch4` working set is **`1,135,580` B** (`1.83×` larger than `ballpack.c`), and unconditional `ztree_init` heap allocation across all modes + rank LUT is **`1,161,792` B**.
- **Zero-Division Runtime Unranking (Universal Pure-$\mathbb{Z}$ Granlund–Montgomery / Barrett Theorem):** Every static divisor $d \in \{W_m(b), S_m(r)\}$ is precomputed once into `(mult, shift)` satisfying:
  $$m \cdot d \ge 2^s \;\land\; (m \cdot d - 2^s)\, M_{\text{dividend}} < 2^s \implies \lfloor (x \cdot m) / 2^s \rfloor = \lfloor x / d \rfloor \quad \forall\, 0 \le x \le M_{\text{dividend}}.$$
  All **`4,968 / 4,968` (`100.0%`)** static divisors in `Z^32 (B = 783)` are universally certified over $\mathbb{Z}$; runtime `unrank` executes **`0` hardware divisions**.
- **Exhaustive & Full-136-Bit Roundtrip Verification:** Exhaustive bijections verified on `Z^4 (B = 6)` (`233/233` distinct exact roundtrips) and `Z^8 (B = 3)` (`577/577` distinct exact roundtrips), plus `50/50` boundary and SHA-256-seeded full-136-bit ranks on `Z^32 (B = 783)`.

---

## 3. Pillar 2 — Exact `E_8^4` (`n = 32`) Reduced-Norm Binary-Tree Decomposition & Theta-Series Certification

Representing $E_8$ in doubled integer coordinates $u = 2y \in \mathbb{Z}^8$ (zero floats: all coordinates even or all odd, $\sum (u_i // 2) \equiv 0 \pmod 2$), the odd coset satisfies $(2k+1)^2 = 8 \cdot \frac{k(k+1)}{2} + 1$, collapsing its 8-coordinate norm condition $\|u\|_2^2 = 8m$ directly to $\sum_{i=0}^7 \frac{k_i(k_i+1)}{2} = m - 1$:

- **Exact Theta-Series Verification:** Our reduced-norm `8 -> 4+4 -> 2+2 -> pair LUT` decomposition matches Jacobi's exact $E_8$ shell formula $S_{E_8}(2m) = 240\sigma_3(m)$ (`S_{E_8}(0) = 1`) across **all `392 / 392` shells (`m = 0..391`)** with `0` mismatches, and matches `BallIndex("E8", 32, 2m)` identically on both target operating points:

| `E_8^4` Arm | Exact Radius | Exact Codebook Count $W_{E_8^4}(m)$ | Depth to 4 Independent `E_8` Blocks | `E_8` Block Width | Count/LUT Floor Only (`AUDIT-112`) | Exact C11 Hot Compact / Fast / Total Init (`c-bench/`, vs `BallIndex`) | Universal 0-Div Reciprocals |
|---|---|---|---|---|---|---|---|
| **`136`-bit (`4.500` bpw)** | `m = 391` ($\|y\|^2 \le 782$) | `85889107931845736051215402649708868585601` (`136` b) | **Depth `2`** (`32->16->8`) | `<= 41` b (**`uint64`**) | `56,850` B *(excl. reciprocals, `even_c8`, branch totals, ckpts)* | **`153,970` B** (`72.67×`) / **`970,970` B** (`11.52×`) / **`1,026,526` B** (`10.90×`) vs `11,189,304` B | **`3,465 / 3,465`** (`100%`) |
| **`88`-bit (`3.000` bpw)** | `m = 48` ($\|y\|^2 \le 96$) | `263197133594409826368003841` (`88` b) | **Depth `2`** (`32->16->8`) | `<= 29` b (**`uint32`**) | `5,490` B *(excl. reciprocals, `even_c8`, branch totals, ckpts)* | **`19,122` B** (`72.00×`, fits `32 KiB` L1d) / **`37,170` B** (`37.04×`, **exceeds `32 KiB` L1d**) / **`44,038` B** (`31.26×`) vs `1,376,760` B | **`449 / 449`** (`100%`) |

- **Pair LUTs & Exhaustive Controls:** Even/odd pair LUTs hold only `2,449 + 2,456` pairs (`9,810` B) at `136` bits and `293 + 300` pairs (`1,186` B) at `88` bits. Exhaustive bijections verified on `E_8^1 (m <= 1)` (`241/241` distinct) and `E_8^4 (m <= 1)` (`961/961` distinct), plus `50/50` full-136-bit and `44/44` full-88-bit roundtrip proofs.

---

## 4. Pillar 3 — Exact Sub-Block Shaping Tradeoff, `siliquant#23` Bandwidth Ceiling & Honest Cost Model

### 4.1 Why Tree-Convolution Strictly Beats Naive Independent Sub-Blocking at Fixed `136` Bits

| Arm (`136` Bits Total) | Sub-Bound $B_{\text{sub}}$ | Sum $k B_{\text{sub}}$ | Exact Codebook Ratio $W_{32}(783) / W_{\text{sub}}^k$ | Uniform $\mathbb{E}[y_i^2]$ | Exact Shaping Energy Overhead |
|---|---|---|---|---|---|
| **`1 × Z^32` (Tree-Convolution)** | `783` | `783` | `1.000000` | `23.044067` | **`0.00%`** *(zero shaping loss!)* |
| `2 × Z^16` (`68` b each) | `433` | `866` | `1.007305` | `24.083462` | **`+4.51%`** |
| `4 × Z^8` (`34` b each) | `254` | `1,016` | `1.037835` | `25.441487` | **`+10.40%`** |
| `8 × Z^4` (`17` b each) | `162` | `1,296` | `1.017231` | `27.126743` | **`+17.72%`** |

### 4.2 Exact Pure-$\mathbb{Q}$ Verification of `siliquant#23` Bandwidth Ratios & Honest Limitation Boundary

- **Byte Savings (`(Q4 - PZ) / Q4`):** `PZ32` (`4.505 = 901/200` bpw) vs. `Q4_K_M` (`4.80`–`4.82` bpw) saves `59/960 = 6.15%` to `63/964 = 6.54%` (`4.500 = 9/2` bpw saves `1/16 = 6.25%` to `16/241 = 6.64%`), giving an exact memory-bandwidth tokens/sec ceiling gain of **`6.55%–7.11%`** (`59/901` to `16/225`), reproducing `siliquant#23`.
- **Honest Law-4 / Law-5 Scope Boundary:** This pure-$\mathbb{Z}/\mathbb{Q}$ Python certifier proves exact bijectivity, `0.000`-bit rate overhead, critical-path depth (`32 -> 4` for `Z^32`, `depth 2` to 4 independent `uint64`/`uint32` `E_8` blocks), exact register widths, minimal count/LUT byte floors, and universal zero-division reciprocal correctness. It **does NOT measure compiled hardware `weights/sec`** or packed-file format integration; complete `sizeof`-measured C11 runtime footprints and scalar-CPU measurements on one disclosed Zen-1 host are reported in [`c-bench/BENCH_REPORT.md`](c-bench/BENCH_REPORT.md) with no GPU, TPU, or SIMD performance claims extrapolated from the scalar CPU harness.

---

## 5. Reproducibility & Falsification Criteria

```bash
# 1. Offline standalone replay (default; runs intake-free in isolated snapshots)
python3 -I -O -B -W error ballpack_tree_audit.py --standalone --json

# 2. Live read-only recount over staged PR #12 checkout & certificate parity
python3 -I -O -B -W error ballpack_tree_audit.py --live --intake-dir <pr12-checkout> --json | diff -u ballpack_tree_certificate.json -
python3 -I -O -B -W error ballpack_tree_audit.py --check ballpack_tree_certificate.json

# 3. Assertion-free R5 unit, mutation-killing, and Zone-B anti-bypass test suite
python3 -I -O -B -W error test_ballpack_tree_r5.py -v
```

**Falsification Criteria:**
- Any rank $k \in [0, W_n(B) - 1]$ where `rank(unrank(k)) != k`, any unranked vector violating $\|y\|_2^2 \le B$ or $E_8$ coset/sum-parity, or any static divisor failing $m d \ge 2^s \land (m d - 2^s) M_{\text{dividend}} < 2^s$ falsifies Pillars 1 or 2.
- Any discrepancy between `ballpack_tree_audit.py --live --json` and `ballpack_tree_certificate.json`, or any float/non-`str`-key/duplicate-key token inside the JSON certificate, falsifies the audit.
