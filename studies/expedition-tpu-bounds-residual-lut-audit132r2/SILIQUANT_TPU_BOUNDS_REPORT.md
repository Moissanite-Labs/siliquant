# Track S32: Exact-$\mathbb{Z}/\mathbb{Q}$ TPU/Pallas/NEON WCET, Pre-Cast `INT8`/`INT16`/`BF16` Overflow Separation & `O(1)` Branchless-Decode 2-Stage Residual Flat-LUT `E_8` Architecture

- **Track ID:** `TRACK-S32-SILIQUANT-TPU-BOUNDS-RESIDUAL-LUT`
- **Schema:** `aletheia.siliquant_tpu_bounds.certificate.v1`
- **Upstream Context:** `Moissanite-Labs/siliquant#60` (TPU spectral, WCET/VMEM, and pre-cast numerical overflow bounds) and `Moissanite-Labs/siliquant#61` (Cloud TPU MXU vs. on-device Edge/NPU mapping, branchless vector LUTs, and safety/refusal preservation)
- **Arithmetic & Dependencies:** Pure Python 3 standard library over exact $\mathbb{Z}/\mathbb{Q}$ (`int` and `fractions.Fraction`); `0` third-party libraries, `0` AST float literals, `0` JSON floats, `0` bare `assert` statements, `0` mathematical RNG (`secrets.token_hex(8)` is used strictly for Zone-B atomic temp-file collision avoidance in `audit.py`, never in mathematical computation).
- **Certificate Digests (Bit-Identical Across R1 & R2):**
  - `siliquant_tpu_bounds_certificate.json` file SHA-256: `d803101aa6989b566b2c3bf539b2d4f8c74b6d2fb2cb36d7c5b24926a3fa8797`
  - `payload_sha256`: `3dd4fd68dc1ca7cb127d5b8ac8baf4c313cb3862583abd0471c53ce088613f1d`
  - `certificate_sha256`: `e283eac7115c176b775a0724de5c54a98d994bb7a78d728855773ecce7e11631`

---

## 1. Epistemic Taxonomy (Law-4 Honesty: `N0` vs. `N1`)

- **Overall Certificate Novelty Tier:** **`N1`** (exact-$\mathbb{Z}/\mathbb{Q}$ computer-certified hardware dynamic-range separation theorems, TPU roofline/WCET bounds, `O(1)` branchless-decode 2-stage residual flat-LUT `E_8` codec with unconditional no-clip covering contraction, zero-unpack spectral enclosures, and refusal-margin preservation witness).
- **Explicit Law-4 Unmeasured Flags (Certified `false` in Payload):**
  - `measured_pallas_tok_s = false` (TPU v5e roofline ceilings are exact architectural bounds from published hardware specs `819 GB/s` / `197 TFLOP/s` / `394 TOP/s`, NOT measured Pallas kernel throughput).
  - `measured_live_model_kl_residual_lut = false` (2-stage residual flat-LUT `E_8` distortion contraction is proved on exact rational vectors and covering geometry, NOT yet measured on live Gemma-3/Qwen2.5 checkpoints).
  - `measured_live_model_safety_benchmarks = false` (safety/refusal margin preservation is proved algebraically and on an exact `8×8` rational witness, NOT yet measured on live GSM8K/refusal benchmarks).
- **`N0` (Classical Lattice Facts & Published Hardware/Upstream Context — Explicitly Quarantined):**
  - Classical $E_8$ lattice ($R_{\mathrm{cov}} = 1$, Eisenstein $E_4$ theta series $r_{E_8}(m) = 240 \sigma_3(m)$), Conway–Sloane $E_8$ nearest-point decoder, Sylvester $H_8$ and `Block-H2` ($I_4 \otimes H_2$) orthogonal matrices, IEEE-754 `bfloat16` format (`8`-bit significand, exponent `[-126, 127]`), and TPU v5e published specs (`128 MiB` VMEM, `819 GB/s` HBM, `197 TFLOP/s` BF16 MXU, `394 TOP/s` INT8 MXU).
  - Prior upstream table sizes (`5,490 B` / `56,850 B` tree tables; `93,472 B` / `1,603,328 B` `guide_a` tables) and `Block-H2` / `H_8` anisotropy bridges from `siliquant` `#35`, `#36`, `#42`, `#44`, and `#58`–`#59`.
- **`N1` (Exact-$\mathbb{Z}/\mathbb{Q}$ Certified Contributions in Track S32):**
  1. Constructive attainable witness proofs + tight upper impossibilities across all 4 ball regimes proving the sharp `INT8` MXU overflow separation on `EBG32:b136s8` (`Block-H2` peak **`78 <= 127`** vs. `WHT_8` peak **`156 > 127`**), exhaustive `513/513` zero-ULP `BF16` exactness on `[-128, 128]`, and exact DP sharpening of 32-weight super-block dot-product peaks (`13,716` attainable vs. `14,078` Cauchy-Schwarz on `88b`; `40,132` attainable vs. `40,180` CS on `136b`, overflowing `int16` and proving `> 103×` `int32` headroom at $K = 16,384$).
  2. Exact rational VMEM residency fractions, static branchless WCET trip counts (`6` / `9` fixed `select` steps per shell/radius binary-search split node vs. `O(1)` branchless decode in `2` VMEM gathers + `1` mul + `1` FMA; offline outside-ball encode is $\Theta(|\mathrm{LUT}|)$), and exact TPU v5e three-way roofline proving `INT8` MXU doubles the bandwidth-bound batch regime (`67.65 -> 135.30` at `4.50 bpw`; `72.16 -> 144.32` at `4.80 bpw`) and achieves `0%` BF16 gain vs. exact `+1/15` (`+6.67%`) INT8 gain at $B = 73$.
  3. Exact enumeration of all `2,401` `m <= 2` (`32 KiB`) LUT entries (`max |z_i| = 4`, fits signed `int4`/`int8`), tight `m <= 5` (`512 KiB`) bound (`56,881` pts, `max |z_i| = 6`), pure-$\mathbb{Q}$ unconditional no-clip covering contraction theorem (`25/49` for `m <= 2` at `s_f = 5/7 s_c`; `1/4` at `s_f = 1/2 s_c` and `25/144` at `s_f = 5/12 s_c` for `m <= 5`), and exact `0`-mismatch 2-stage encode/decode achieving `22/117` (`0.1880 < 1/4`) fixed-scale and `40/351` (`0.1139 <= 1/8`) optimal-scale distortion contraction.
  4. Zero-unpack exact Frobenius + Bareiss Sylvester positive-definiteness spectral enclosure for both single-stage `E_8^4` (`251/32`, residual `0`) and 2-stage residual `E_8` (`565/64 <= 10.682286...`), plus exact `2×` (`Block-H2`) and `8×` (`H_8`) opposite-sign activation-anisotropy flips in the quadratic KL surrogate $\operatorname{tr}(H\Sigma) = 2 \cdot \mathrm{KL}_{\mathrm{quad}}$ at invariant trace `1/4`.
  5. Exact worst-case refusal-margin shrinkage bound with rank-1 adversary equality, exact Walsh-Hadamard isotropic variance identity (`19/32000 = ||dW||_F^2 / 64`), and `8×8` witness where coarse scalar rounding flips baseline margin `+272/25` to `-32` while 2-stage residual `E_8` (`32 KiB` LUT) reduces MSE by `19/6424` (`0.002957`, `338×`) and preserves margin `+10 > 0`.

---

## 2. Five Certified Pure-$\mathbb{Z}/\mathbb{Q}$ Pillars

### Pillar 1 — Pre-Cast Numerical Dynamic Range, `Block-H2` vs. `WHT_8` `INT8` Separation & Accumulator Width
For doubled $E_8$ lattice vectors $z = 2v \in 2 E_8$ (uniform coordinate parity, $\sum_{i=0}^7 z_i \equiv 0 \pmod 4$, $\|z\|_2^2 = 8m \le 8m_{\max}$), every row of Sylvester $H_8$ has either `0` or `4` negative entries, proving $(H_8 z)_r \equiv \sum z_i \equiv 0 \pmod 4$ identically; uniform coordinate parity proves $((I_4 \otimes H_2) z)_r \equiv 0 \pmod 2$ identically.

| Regime (`m_max`, `W_1` width) | Exact max `\|z_i\|` (`int8 <= 127`) | Exact max `Block-H2` (`<= 127`) | Exact max `WHT_8` (`<= 127`) | CS 8-w / 32-w dot (`\|a_i\| <= 127`) | Exact attainable 32-w `\|\|z\|\|_1` & dot (`sym 127` / `asym 128`) | `int16` (`<= 32,767`) |
|---|---:|---:|---:|---:|---:|---|
| **`EBG32:b88s8`** (`48`, `29b u32`) | **`19`** (`[19,1,1,1,1,-1,-1,-1]`) | **`26`** (`[12,14,2,0^5]`) | **`52`** (`[8,8,6^6]`) | `7,039` / `14,078` | `108` -> **`13,716`** / `13,824` | **SAFE** |
| **`EBG32:b136s8`** (`391`, `41b u64`) | **`55`** (`[55,1,1,1,1,-1,-1,-1]`) | **`78`** (`[38,40,2,0^5]`, **SAFE**) | **`156`** (`[20^6,18,18]`, **OVERFLOW**) | `20,090` / `40,180` | `316` -> **`40,132`** / `40,448` | **OVERFLOW** (needs `int32`) |
| **`1block_E8_3.0bpw`** (`18`, `23b`) | **`12`** (`[12,0^7]`) | **`16`** (`[8,8,0^6]`) | **`32`** (`[4^8]`) | `4,310` / `8,621` | `66` -> **`8,382`** / `8,448` | **SAFE** |
| **`1block_E8_4.5bpw`** (`151`, `35b`) | **`34`** (`[34,2,0^6]`) | **`48`** (`[24,24,0^6]`) | **`96`** (`[12^8]`) | `12,484` / `24,969` | `196` -> **`24,892`** / `25,088` | **SAFE** |

- **Sharp `INT8` Separation (`EBG32:b136s8`):** `WHT_8` achieves `156 > 127` (signed `int8` overflow) while `Block-H2` achieves `78 <= 127`, with `79` ruled out by coordinate parity and `80` ruled out by Cauchy-Schwarz (`80^2 = 6400 > 6256 = 16 \cdot 391`).
- **`BF16` Zero-ULP Exactness:** All `513/513` half-integers $v = z/2 \in [-128, 128]$ (`z in [-256, 256]`) are certified bit-exact in `BF16` (`0` ULP), and the sharp boundary `z = 257` (`v = 128.5`, requiring `9` significand bits) is verified inexact.
- **Hidden-Dim `K = 16,384` (`512` super-blocks) `int32` Headroom on `136b`:** `2,147,483,647 / 20,572,160 = 104.38×` (CS ceiling), `2,147,483,647 / 20,547,584 = 104.51×` (exact attainable symmetric `127`), and `103.69×` (asymmetric `128`) — all strictly `> 100×`.

### Pillar 2 — Static XLA/Pallas WCET, VMEM Residency & TPU v5e Three-Way Roofline
- **VMEM Residency (`128 MiB = 134,217,728 B`):**
  - `tree_88b`: `5,490 B` (`2745/67108864`, `0.0040%`); `tree_136b`: `56,850 B` (`28425/67108864`, `0.0423%`);
  - `guide_a_88b`: `93,472 B` (`2921/4194304`, `0.0696%`); `guide_a_136b`: `1,603,328 B` (`6263/524288`, `1.1945%`, leaving **`98.8054% > 98.8%`** free);
  - `flat_lut_12b_32kib`: `32,768 B` (`1/4096`, `0.0244%`); `flat_lut_16b_512kib`: `524,288 B` (`1/256`, `0.3906%`).
- **Static Branchless Decode WCET:** `EBG32:b88s8` requires `ceil(log2(49)) = 6` fixed `select` steps per shell/radius split node (`W_1(48) = 360,605,761`, `29b` native `uint32` VPU); `EBG32:b136s8` requires `ceil(log2(392)) = 9` fixed `select` steps per shell/radius split node (`W_1(391) = 1,525,056,448,801`, `41b` emulated `uint64`); **2-Stage Residual Flat-LUT `E_8`** executes **online inference decode** in **`O(1)` (`2` fixed VMEM gathers + `1` scale mul + `1` FMA**, `0` branches, `0` divisions, `0` `uint64`), whereas **offline quantization encode** uses Conway–Sloane inside the ball and exact $\Theta(|\mathrm{LUT}|)$ finite-ball search outside the ball.
- **TPU v5e Three-Way Roofline (`819 GB/s`, `197 TFLOP/s` BF16, `394 TOP/s` INT8):**
  - BF16 ridges: $B^*(4.50) = 24625/364$ (`67.65`), $B^*(4.80) = 19700/273$ (`72.16`).
  - INT8 ridges: $B_{\mathrm{INT8}}^*(4.50) = 24625/182$ (`135.30`), $B_{\mathrm{INT8}}^*(4.80) = 39400/273$ (`144.32`) — exact **`2×`** bandwidth-bound batch extension.
  - Break-even Pallas decode rate: **`1,365 G w/s`** (`1,456 G w/s` for `4.50 bpw` HBM saturation).
  - At $B = 73$ (and all $B \ge 73$): BF16 bandwidth gain is **`0`** (`0%`), whereas INT8 MXU remains strictly bandwidth-bound with exact **`1/15`** (`+6.67%`) throughput advantage for `4.50 bpw` over `4.80 bpw`.

### Pillar 3 — `O(1)` Branchless-Decode 2-Stage Residual Flat-LUT `E_8` Codec & Contraction Theorem
- **Decode vs. Encode Complexity Separation:** Online inference **decode** (`decode_2stage_residual_e8`) is strictly **`O(1)` branchless** (`2` fixed VMEM table gathers + `1` scale multiply + `1` FMA, validating the two indexed `2*E_8` ball vectors in `O(1)`). Offline **encode** (`nearest_e8_in_ball_doubled` / `encode_2stage_residual_e8`) validates custom LUT membership in `2*E_8` within shell `m_max`, takes the unconstrained Conway–Sloane nearest point when inside the ball, and performs exact integer $\Theta(|\mathrm{LUT}|)$ projection when outside the ball.
- **Eisenstein $E_4$ Shells `m = 0..8`:** `[1, 240, 2160, 6720, 17520, 30240, 60480, 82560, 140400]` (cumulative `2,401` at `m = 2`, `56,881` at `m = 5`, `340,321` at `m = 8`).
- **Flat LUT Geometry & `int4`/`int8` Headroom:**
  - `12`-bit `32 KiB` LUT (`m <= 2`): `2,401` exact enumerated points (`1,695` slack slots), `max |z_i| = 4` (fits signed `int4` `[-8, 7]` and `int8`).
  - `16`-bit `512 KiB` LUT (`m <= 5`): `56,881` points (`8,655` slack slots), tight `max |z_i| = 6` (fits signed `int4` `[-8, 7]` and `int8`).
- **Unconditional No-Clip Covering Contraction Theorem ($R_{\mathrm{cov}}(E_8) = 1$):**
  - `m <= 2` (`32 KiB`): $(7/5 + 1)^2 = 144/25 < 6 \implies s_f = (5/7) s_c$ guarantees zero clipping over the entire $E_8$ Voronoi cell with worst-case squared contraction $\le 25/49$.
  - `m <= 5` (`512 KiB`): $(2 + 1)^2 = 9 < 12$ and $(12/5 + 1)^2 = 289/25 < 12 \implies s_f = (1/2) s_c$ guarantees unconditional zero-clip squared contraction $\le \mathbf{1/4}$ (`-6.02 dB`), and $s_f = (5/12) s_c$ guarantees $\le \mathbf{25/144} < 1/5$.
- **6-Vector Exact Codec Suite (`m <= 2` `32 KiB` LUT, `0` encode/decode mismatch):**
  - Fixed `s_f = s_c / 3`: aggregate squared-error ratio **`22/117` (`0.1880 < 1/4`)**.
  - Per-vector optimal rational `s_f ∈ {1/2, 1/3, 1/4, 1/6}`: **`6/6`** vectors strictly `<= 1/4` (`bulk_interior: 19/99`, `e8_deep_hole: 0`, `axis_outlier: 1/36`, `block_h2_rotated: 19/144`, `mixed_half_hole: 2/9`, `outer_shell_wing: 4/29`) and aggregate ratio **`40/351` (`0.1139 <= 1/8`)**.

### Pillar 4 — Analytical Spectral / Lipschitz Tension-Drift Enclosure
- **Orthogonal Isometries:** Verified $H_8^\top H_8 = 8 I_8$ and `Block-H2` $B_2^\top B_2 = 2 I_8$.
- **Zero-Unpack Frobenius & Bareiss Sylvester PD Enclosures (`8×8` witness):**
  - Single-stage `W_c`: zero-unpack $\sum_b 2 s_b^2 m_b = \mathbf{251/32}$ matches entrywise $\|W_c\|_F^2$ with `0` residual, and `251/32 · I - W_c^T W_c` passes all `8/8` Bareiss leading principal minors `> 0`.
  - 2-Stage Residual `W_c + W_f`: $\|W_f\|_F^2 = 7/32$, cross inner product $\langle W_c, W_f \rangle = 49/128$, exact identity $\|W_c + W_f\|_F^2 = 565/64 = 251/32 + 7/32 + 2(49/128)$ holds with `0` residual, Cauchy-Schwarz $(49/128)^2 \le (251/32)(7/32)$ holds, rational triangle ceiling $10682286494962767169/10^{18} \ge 565/64$ holds, and Bareiss certifies $\tau_{\mathrm{res}} I - (W_c + W_f)^\top (W_c + W_f) \succ 0$.
- **Exact `Block-H2` vs. `H_8` Activation-Anisotropy Separation (Quadratic KL Surrogate):** Defining the quadratic KL surrogate $J(H, \Sigma) = \operatorname{tr}(H \Sigma) = 2 \cdot \mathrm{KL}_{\mathrm{quad}}$, at identical trace $\operatorname{tr}(\Sigma_e) = \operatorname{tr}(\Sigma_{e,B_2}) = \operatorname{tr}(\Sigma_{e,H_8}) = 1/4$, `Block-H2` produces an exact **`2×`** opposite-sign quadratic KL-surrogate flip (`1/4` vs. `1/8` under $H = e_0 e_0^\top$; `1/8` vs. `1/4` under $H = \frac{1}{2}(e_0+e_1)(e_0+e_1)^\top$) and `H_8` produces an exact **`8×`** opposite-sign flip (`1/4` vs. `1/32` under $H = e_0 e_0^\top$; `1/32` vs. `1/4` under $H = \frac{1}{8}\mathbf{1}\mathbf{1}^\top$).

### Pillar 5 — Safety/Refusal Boundary Margin Preservation
- **Exact Bounds & Identities:** Certifies worst-case shrinkage $(u^\top \Delta W v)^2 \le \|u\|_2^2 \|v\|_2^2 \|\Delta W\|_F^2$, rank-1 adversary tightness equality, Bareiss PD bound $\|\Delta W\|_F^2 I - \Delta W^\top \Delta W \succ 0$, and exact isotropic variance identity over both `64` standard-basis pairs and `64` normalized Walsh-Hadamard pairs (`19/32000 = \|\Delta W\|_F^2 / 64`, residual `0`).
- **Deterministic `8×8` Refusal-Direction Witness:**
  - Baseline refusal margin $u^\top W v = \mathbf{272/25}$ (`+10.8800 > 0`).
  - Coarse scalar $\mathbb{Z}$-grid quantization (`||dW||_F^2 = 1606/125`): margin flips to **`-32 <= 0`** (**refusal boundary broken**).
  - 2-Stage Residual Flat-LUT `E_8` (`m <= 2` `32 KiB` LUT, `s_c = 1, s_f = 1/8`, `||dW||_F^2 = 19/500`): cuts MSE by exact factor $\rho = \mathbf{19/6424}$ (`0.002957`, tightening worst-case/RMS margin drift by $\sqrt{\rho}$), and preserves margin **`+10 > 0`** (**refusal direction survives intact**).
  - Architectural guardrail preserved in payload: `token_embd_quantized_by_block_lut = false` and `token_embd_immutable_row_hash_required = true` (linking `Track S31` / `siliquant#59`).

---

## 3. Audit Architecture & `--check` Verification Scope

- **`siliquant_tpu_bounds_audit.py --check <cert.json>`** enforces fail-closed Zone-B I/O (per-component `O_RDONLY|O_DIRECTORY|O_NOFOLLOW` traversal, leaf `O_NOFOLLOW|O_NONBLOCK` + `S_ISREG` + capped read, CRLF/trailing-LF/UTF-8/canonical-ASCII-JSON checks, bounded `MAX_INT_BITS = 3322` integer cap with zero raw `ValueError`, negative-zero normalization in `format_decimal_trunc`, strict custom-LUT `2*E_8` & ball-shell validation), then executes:
  1. **Full deterministic pure-$\mathbb{Z}/\mathbb{Q}$ replay** (`validate_certificate_payload(cert, full_replay=True)`, enforcing strict exact `bool` on `full_replay`), recomputing the entire certificate and checking exact canonical byte equality plus `payload_sha256` and `certificate_sha256`;
  2. **Secondary algebraic invariant cross-check** (`secondary_algebraic_audit`), fail-closed validating Law-4 flags, exact `4`-regime key set & exact `int` (non-`bool`) types across all integer fields and witness vectors/blocks, `INT8`/`INT16`/`INT32` thresholds, `2×` roofline ridge ratio, `1,365 G w/s` break-even rate, `2,401`/`56,881` LUT counts, exact `6`-vector distortion suite length & integer indices, `<= 1/8` aggregate contraction, Bareiss/anisotropy flags, `8×8` witness dims, and `< 1/100` witness MSE reduction.
- **Honest Scope Note:** `siliquant_tpu_bounds_audit.py` imports canonicalization, parsing, and replay primitives from `siliquant_tpu_bounds_oracle.py`; `--check` is therefore a **full pure-$\mathbb{Z}/\mathbb{Q}$ deterministic replay + secondary invariant cross-check** (not a zero-shared-code independent re-implementation). Under `full_replay=False` paired with `secondary_algebraic_audit`, all top-level schema/digest invariants and audited integer/witness/algebraic invariants are enforced (`18/18` on `S19c`), while full byte-for-byte recomputation requires `full_replay=True`.
- **Zone-B Atomic Write Hardening (`--write`):** Refuses hardlinked targets (`st_nlink > 1`), generates multi-thread-unique temp filenames via `threading.Lock()` + monotonic counter + `os.getpid()` + `threading.get_ident()` + `secrets.token_hex(8)`, creates `0o600` temp files with `O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW|O_NONBLOCK` in the anchored parent `dir_fd`, calls `os.fsync(tmp_fd)`, verifies pre-replace `os.fstat(tmp_fd).st_nlink == 1` (`HARDLINK_FORBIDDEN` with temp cleanup on violation), replaces atomically via `os.replace(..., src_dir_fd=dfd, dst_dir_fd=dfd)`, and calls `os.fsync(dfd)`; rejects deleted CWD (`CWD_UNAVAILABLE`, `rc=2`), lone UTF-16 surrogates, non-ASCII digits, and flag-like path arguments (`--write --json`, `--write=--json`, `--check=--json` -> `rc=2`, `0` files created). Under standard POSIX semantics, a malicious same-UID process with write access to the parent directory remains outside any user-space process boundary.

---

## 4. Verification Receipts (R2 Post-Final-Edit, Strictly Sequential)

| Gate | Scope & Command Class | Result |
|---|---|---|
| **5-Mode Parity** | `-I -B`, `-I -O -B`, `-I -B -W error`, `-I -O -B -W error`, `PYTHONHASHSEED=42 -B` (without `-I`) | **PASS `5/5`**: standalone `d803101a...` ×5 (bit-identical to R1 & disk), `--check --json` `52162e3a...` ×5, `test_r5` `12/12 OK` ×5, zero-arg `OK` |
| **Default-CLI `aletheia v2.0.0 mutate`** | `--max-mutants 0 --work-root . --add-cwd --timeout 45` on both `oracle.py` & `audit.py` | **PASS `136/136` (`100.0%`, `survived=0`, `truncated=False`)**: `oracle.py` `67/67` + `audit.py` `69/69` |
| **Exhaustive AST `BoolOp` (`SWAP+SUBCOND_DEL`)** | All `23` `ast.BoolOp` nodes (`timeout=30`) across `oracle.py` & `audit.py` | **PASS `69/69` (`100.0%`, `surv=[]`)**: `oracle.py` `48/48` (`16` nodes) + `audit.py` `21/21` (`7` nodes) |
| **Zone-B Adversarial Fuzzer (`S19c`)** | CLI + In-Memory (`full_replay=True`) + In-Memory (`full_replay=False + secondary_algebraic_audit`) | **PASS `59/59` (`100.0%`)**: `23/23` CLI + `18/18` full-replay + `18/18` non-replay+secondary (`bypasses=0, crashes=0`) |
| **Live Concurrency, Kernel Races & `procfs`** | `24T×20` & `80T` same-target writes + `renameat2(RENAME_EXCHANGE)` + `os.link`/`os.unlink` + `/proc/kallsyms` | **PASS**: `24T×20` `480/480` OK (`0o600`, `0` errs/`0` temps) + `80T` burst `80/80` OK; `21,892` `renameat2` swaps (`700`r `0` leaks, `700`w `0` clobbers/`0` temps); `8,006` hardlink flips (`600`w `0` clobbers/`0` temps); `/proc/kallsyms` -> `FILE_TOO_LARGE` |
| **Static AST & `aletheia audit-check`** | `audit-check . --strict-ancestors --safety-scan --scripts siliquant_tpu_bounds_audit.py test_siliquant_tpu_bounds_r5.py` | **PASS** (`0` findings, `0` AST floats/`float()`/bare `assert`s/`.isdigit()` across `3/3` `.py`, `0` JSON floats, `PYC=0`, `CRLF=0`, `0` leaks) |

---

## 5. Frozen R2 File Manifest

| File | Lines | Bytes | SHA-256 |
|---|---:|---:|---|
| `siliquant_tpu_bounds_oracle.py` | 1,602 | 68,575 | `82cd7e53bc1ef25b4592fa586b52c83eea276e3abe63d8573d7da2e2b09e694b` |
| `siliquant_tpu_bounds_audit.py` | 545 | 23,788 | `669fbeb3cbc7f5775fb871476ecbd58fa1aecbd266af4c480c16457bc43d53b0` |
| `test_siliquant_tpu_bounds_r5.py` | 965 | 56,340 | `b3456c3e96d81b93b8c4ae73d8cf1f92315ee95a47bf845a929ca6250f39234d` |
| `siliquant_tpu_bounds_certificate.json` | 1 | 11,186 | `d803101aa6989b566b2c3bf539b2d4f8c74b6d2fb2cb36d7c5b24926a3fa8797` |
