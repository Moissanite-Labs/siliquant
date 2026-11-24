# Track S42 (Portals 39–42 Unified): `lookatmyquant` × `Kintsugi` × `gbench` × `MemT-1.0-Pro` Exact-Operator Suite

- **Track ID:** `TRACK-S42-FISHER-WATERFILLING-KINTSUGI-MARGINS`
- **Schema:** `aletheia.fisher_kintsugi_memt.certificate.v1`
- **Arithmetic & Dependencies:** Pure Python 3 standard library over exact $\mathbb{Z}/\mathbb{Q}$ (`int`, `fractions.Fraction`); `0` third-party libraries, `0` AST float literals, `0` `float()` calls, `0` `.isdigit()` calls, `0` bare `assert` statements, `0` `os.path.islink` calls, `0` mathematical RNG across all `3` Python scripts (`fisher_kintsugi_memt_oracle.py`, `fisher_kintsugi_memt_audit.py`, `test_fisher_kintsugi_memt_r5.py`) and `0` JSON floats in `fisher_kintsugi_memt_certificate.json`.
- **Certificate Digests (`0o600`, canonical ASCII JSON + LF):**
  - `fisher_kintsugi_memt_certificate.json` file SHA-256: `ba44be55ca16873671b41f7e0cac32cf5c43361a07cff7c4ff7b04fcc917cee7` (`9,910` B)
  - `payload_sha256`: `332f61cb2d97c9afb2c063c04af42d63928815f700df418c77a5ee6953890484`
  - `certificate_sha256`: `3fac9982cbd9fa232257de9235463a7c778884de1e1fa1e5c8bd534e11abd25b`

---

## 1. Epistemic Taxonomy & `CS-13` Prior-Art Accounting (`N0` vs. `N1`)

- **Overall Certificate Novelty Tier:** **`N1`** (exact pure-$\mathbb{Z}/\mathbb{Q}$ operator theorems, separation witnesses, order-statistic sample-size proofs, and a 4-channel persistence airlock compounding over `N0` empirical benchmarks and classical mathematics).
- **Law-4 Honesty Flags (All Explicitly `False` in Payload):**
  - `measured_live_gemma4_lamq_weights = False` (we certify exact dyadic sensitivity spectra matching the `>= 12x` Gemma 4 E2B scale, not live Gemma 4 checkpoint tensors).
  - `measured_live_gbench_serving_endpoint = False` (we certify exact distribution-free binomial order-statistic bounds at `n in {100, 458, 459, 1099}`, not live server telemetry).
  - `measured_live_memt_554b_weights = False` (we certify exact `Q^{4x4}` `GDN-2` delta-rule capacity and persistent poisoning witnesses, not live `554B` parameter activations).
- **`N0` Prior-Art Anchors (`prior_art_cs13_anchors`, Explicitly Quarantined):**
  1. `ShinMegamiBoson/lookatmyquant` (Portal 41): per-row Fisher bit allocation achieving `12x` lower KL than `Q4_K_M` on Gemma 4 E2B, paired with `Track S30`'s log-convex rate-distortion law $K(b) = C\cdot 2^{-\alpha b}$.
  2. `professorpalmer/kintsugi-method` (Portal 40): task-break logit-margin repair vs. smooth average KL (`CS-20` / `Track S34` Rosen discontinuity).
  3. `professorpalmer/hard-proof` (Portal 40, `CS-13` neighbor): *"Proved is not finished"* discipline — executable repro-only findings paired with adversarial witnesses exposing what a rule or metric leaves open (directly mirrored here by proving what smooth average KL leaves open in Pillar 2 and what the legacy 3-channel airlock leaves open in Pillar 4).
  4. `google-deepmind/gbench` (Portal 39): open-loop Poisson serving-SLO certification requiring `n >= 1,099` for `P99` resolution and emitting explicit `UNCERTIFIED` below budget (`CS-18`).
  5. `MemTensor/MemT-1.0-Pro` (`arXiv:2610.26825`, Portal 42): `GDN-2` rank-1 delta-rule fixed-size dynamic memory (`13M` state) with honest caveat that textual memory still wins detailed long-history retrieval.
  6. Classical mathematics (`N0`): AM-GM inequality, softmax Fisher information metric $\operatorname{diag}(p) - pp^\top$, Gibbs' inequality, binomial order statistics, Sylvester–Hadamard $H_4$, Cauchy–Schwarz, and Parseval's identity.
- **`N1` Certified Contributions in Track S42:**
  1. Exact zero-float dyadic AM-GM Fisher water-filling certifier (`R = AM(w)/GM(w) = 263/16 = 16.4375x > 12x` at `0` extra average bpw) with exact rational budget-preserving perturbation convexity witnesses.
  2. Dual exact-$\mathbb{Q}$ proof of `Kintsugi` / `CS-20` average-KL blindness via BOTH the exact softmax Fisher metric (`1499/176 = 8.5170x` higher average KL yet passing all margins) AND exact dyadic Shannon KL (`21/4 = 5.25x` higher exact bits yet passing all margins, zero Taylor approximation).
  3. Exact pure-integer `gbench` `P99` order-statistic sample-size law certifying the sharp `99%` boundary (`n = 458` FAILS vs `n = 459` PASSES), the `n = 1,099` miss-probability bound `1/62651 < (99/100)^1099 < 1/62650 < 1/60000`, and `> 95%` two-sided order-statistic coverage (`[X_(1081), X_(1096)]` for `4 <= E <= 18` gives `>= 97.8156%`; note `[X_(1081), X_(1095)]` corresponds to `5 <= E <= 18` at `96.8111%`), while keeping `100^1099` strictly internal to preserve the binding `MAX_INT_BITS = 3321` JSON integer cap.
  4. Exact `Q^{4x4}` `GDN-2` rank-capacity impossibility proof (`L = 4` exact `0`-error retrieval vs `L = 5` universal Cauchy–Schwarz lower bound `106/5 > 0` and exact Parseval error `212/5`), persistent rank-1 Channel-2B latent-memory poisoning flipping a benign query (`+2 -> -83/25`) under frozen weights ($\Delta W = 0$, $\Delta\mathrm{KV} = 0$, $\Delta P_{\mathrm{data}} = 0$) even after an orthogonal benign turn, and an executable 4-channel airlock (`evaluate_four_channel_airlock`) extending `Track S31`.

---

## 2. Four Certified Pure-$\mathbb{Z}/\mathbb{Q}$ Mathematical Pillars

### Pillar 1 — Exact AM-GM Fisher Water-Filling vs. Uniform-BPW Rate-Distortion (`lookatmyquant` × `Track S30`)

Under `Track S30`'s log-convex rate-distortion law $K_i(b_i) = w_i \cdot 2^{-\alpha b_i}$ with dyadic row sensitivities $w_i = 2^{e_i}$, $\alpha = 2$, $m = 8$, and average bit budget $\bar b = 4$ bpw (exact mean exponent $\bar e = 0 \implies \operatorname{GM}(w) = 1$), KKT optimality sets $b_i^* = \bar b + (e_i - \bar e)/\alpha$, equalizing every row to $K_i(b_i^*) = K_{\mathrm{eq}} = 1/256$ (`K_Fisher = 1/32`) at **exact `0` extra average bpw**, with universal ratio $R = K_{\mathrm{uniform}}(\bar b) / K_{\mathrm{Fisher}}(\bar b) = \operatorname{AM}(w) / \operatorname{GM}(w) \ge 1$:

| Spectrum ($m=8, \alpha=2, \bar b=4$) | Sensitivity Exponents $e_i$ | Optimal $b_i^*$ (bpw) | Extra bpw | $K_{\mathrm{uniform}}$ | $K_{\mathrm{Fisher}}$ | Exact Ratio $R = \mathrm{AM}/\mathrm{GM}$ |
|---|---|---|---:|---:|---:|---:|
| `uniform_control` | `[0, 0, 0, 0, 0, 0, 0, 0]` | `[4] * 8` | `0` | `1/32` | `1/32` | **`1`** (`1.0000`) |
| `mild_skew` | `[4, 2, 0, 0, 0, 0, -2, -4]` | `[6, 5, 4, 4, 4, 4, 3, 2]` | `0` | `389/4096` | `1/32` | **`389/128`** (`3.0390 > 2`) |
| `heavy_outlier_gemma4_scale` | `[7, -1, -1, -1, -1, -1, -1, -1]` | `[15/2, 7/2, ..., 7/2]` | `0` | `263/512` | `1/32` | **`263/16`** (`16.4375 > 12`) |

- **Strict Convexity Perturbation Witnesses on `heavy_outlier_gemma4_scale` ($\sum \delta_i = 0, \delta \neq 0$, all $b_i^* + \delta_i > 0$):**
  - `pair_half_bit` (`[1/2, -1/2, 0..]`): $K = 17/512$, exact excess `+1/512 > 0` (`17/16` of optimal).
  - `pair_one_bit` (`[1, -1, 0..]`): $K = 41/1024$, exact excess `+9/1024 > 0` (`41/32` of optimal).
  - `multi_row_asymmetric` (`[1, 1/2, -1/2, -1, 0..]`): $K = 43/1024$, exact excess `+11/1024 > 0` (`43/32` of optimal).

### Pillar 2 — Exact `Kintsugi` Task-Break Logit-Margin Inversion vs. Smooth Average KL (`CS-20` Rosen Bridge)

Across a 4-token prompt (`3` wide-margin routine tokens + `1` narrow-margin critical digit/tool-call token), BOTH exact local Fisher geometry AND exact dyadic Shannon KL certify that `Uniform-A` achieves strictly lower average KL than `Kintsugi-B` while suffering a discrete top-1 task-break margin inversion:

| Witness Class | `Uniform-A` Avg KL | `Kintsugi-B` Avg KL | KL Ratio (`B / A`) | `Uniform-A` Critical Margin & Task | `Kintsugi-B` Min Margin & Task |
|---|---:|---:|---:|---|---|
| **2A — Quadratic-Fisher** ($\frac{1}{2}\delta^\top(\operatorname{diag}(p)-pp^\top)\delta$, rank `3`, `3x3` block PD, $F\mathbf{1}=0$) | **`11/512`** | `1499/8192` | **`1499/176`** (`8.5170x`) | **`-1/2 < 0`** (`top1=1`, **FAIL**) | **`+1/4 > 0`** (`top1=0`, **PASS**) |
| **2B — Exact Dyadic Shannon KL** ($\sum p_j \log_2(p_j/q_j) \in \mathbb{Q}$, zero Taylor approx, Gibbs `> 0`) | **`1/16` bit** | `21/64` bit | **`21/4`** (`5.25x`) | **`-1/4 < 0`** (`top1=1`, **FAIL**) | **`+1/8 > 0`** (crit `+1/4`, **PASS**) |

### Pillar 3 — Exact `gbench` Open-Loop `P99` Order-Statistic Sample-Size Law (`CS-18` `UNCERTIFIED` Floor)

For i.i.d. open-loop latencies with exceedance count $E_n = \#\{i : X_i > Q_{0.99}\} \sim \operatorname{Bin}(n, 1/100)$, pure-integer evaluation of $(99^n, 100^n)$ and exact binomial tail sums $\sum_{k=0}^r \binom{n}{k} 99^{n-k} / 100^n$ certifies:

| Cohort $n$ | `100^n` Bits / Digits | `100 * 99^n <= 100^n` (`99%` Floor) | `inv_floor` ($\lfloor (100/99)^n \rfloor$) | $\Pr[E_n = 0]$ (`ppm`) | $\Pr[E_n \le 1]$ (`ppm`) | $\Pr[E_n \ge 5]$ (`ppm`) | Upper-Tail / `gbench` SLO (`CS-18`) |
|---:|---:|:---:|---:|---:|---:|---:|---|
| **`100`** | `665` / `201` | **FAIL** | `2` | `[366032, 366033]` | `[735761, 735762]` | `[3432, 3433]` | `UNCERTIFIED` / `UNCERTIFIED` |
| **`458`** | `3043` / `917` | **FAIL** (`> 1/100`) | `99` | `[10021, 10022]` | `[56381, 56382]` | `[483558, 483559]` | `UNCERTIFIED` / `UNCERTIFIED` |
| **`459`** | `3050` / `919` | **PASS** (`<= 1/100`) | `100` | `[9920, 9921]` | `[55918, 55919]` | `[485446, 485447]` | `CERTIFIED_UPPER_TAIL` / `UNCERTIFIED` |
| **`1099`** | `7302` / `2199` | **PASS** (`< 1/60000`) | **`62650`** | `[15, 16]` | `[193, 194]` | `[985150, 985151]` | `CERTIFIED_UPPER_TAIL` / **`CERTIFIED`** |

- **Two-Sided `n = 1,099` Order-Statistic Bracket:** Because $X_{(1081)} \le Q_{0.99} \le X_{(1096)} \iff 4 \le E_{1099} \le 18$, the exact integer binomial sum $\Pr[4 \le E_{1099} \le 18] \in [978156, 978157] / 10^6$ (`97.8156% > 95%`) certifies distribution-free coverage of $Q_{0.99}$ by **`[X_(1081), X_(1096)]`** (for comparison, the narrower `[X_(1081), X_(1095)]` bracket corresponds to $5 \le E_{1099} \le 18$, with exact coverage `[968111, 968112] / 10^6 = 96.8111%`).
- **Zone-B Binding Bit Cap:** Because $100^{1099}$ requires `7,302` bits (`2,199` digits $> 3,321$ bits), raw $100^{1099}$ is strictly computed internally in Python `int` for exact proofs while ONLY bounded integer brackets (`inv_floor`, `ppm`, bit/digit counts) are serialized into JSON.

### Pillar 4 — Exact `MemT-1.0-Pro` `GDN-2` Rank-Capacity Obstruction & Channel-2B Latent-Memory Poisoning

1. **Exact Transpose-Dual `GDN-2` Recurrence on $S_t \in \mathbb{Q}^{4 \times 4}$:**
   $S_{t+1} = S_t(I_4 - \alpha_t k_t k_t^\top) + \beta_t v_t k_t^\top$ (readout $S k$) is verified identically equal to the transpose of $M_{t+1} = (I_4 - \alpha_t k_t k_t^\top) M_t + \beta_t k_t v_t^\top$ (readout $k^\top M$) at both $L = 4$ and $L = 5$.
2. **Exact Rank-`4` Capacity Obstruction ($L = 4$ vs. $L = 5$):**
   - Writing $L = 4$ orthonormal Sylvester–Hadamard keys $k_0..k_3 = \operatorname{cols}(H_4/2)$ ($\|k_j\|_2^2 = 1$, $K_4^\top K_4 = I_4$) with $v_j = 2 e_j$ achieves **`0/4` squared retrieval error** (`rank(S_4) = 4`).
   - Appending a 5th unit-norm Pythagorean key $k_4 = (3/5, 4/5, 0, 0)^\top$ with $v_4 = (0, 6, 0, 0)^\top$ forces $\operatorname{rank}(K_5) = 4 < 5$ with exact nullspace witness $c = (7/10, -1/10, 7/10, -1/10, -1)$ ($K_5 c \equiv 0$) and inconsistency defect $d = \sum c_j v_j = (7/5, -31/5, 7/5, -1/5) \neq 0$, proving the **universal lower bound** $\sum_{j=0}^4 \|S k_j - v_j\|_2^2 \ge \|d\|_2^2 / \|c\|_2^2 = \mathbf{106/5} > 0$ for **ALL** $S \in \mathbb{Q}^{4 \times 4}$.
   - The `GDN-2` delta-rule state $S_5$ retrieves $k_4$ exactly (`0`), corrupts all `4` earlier keys (`[2597/125, 53/125, 2597/125, 53/125, 0]`), and obeys exact Parseval equality $\sum_{j=0}^4 \|S_5 k_j - v_j\|_2^2 = \|\Delta S\|_F^2 = \mathbf{212/5} \ge 106/5$.
3. **Channel-2B Persistent Latent-Memory Poisoning & 4-Channel Airlock (`Track S31` Extension):**
   - Under frozen weights ($\Delta W = 0$, readout head $w_{\mathrm{cls}} = (1, -1, 0, 0)^\top$), zero KV carryover ($\Delta\mathrm{KV} = 0$), and a 100% benign current query $q_{\mathrm{benign}} = k_0$ ($\Delta P_{\mathrm{data}} = 0$), the single poisoned prior turn $(k_4, v_4)$ leaves a rank-1 perturbation $\Delta S = S_5 - S_4$ (`rank = 1`, $\|\Delta S\|_F^2 = 212/5$, distinct SHA-256 state digest) that flips the benign readout margin from **`+2 > 0` (`SAFE`)** to **`-83/25 < 0` (`UNSAFE`)** — and persists **100% unchanged** (`margin_after_benign_orth_turn = -83/25`) after an intervening benign orthogonal turn $k_{\mathrm{orth}} = (4/5, -3/5, 0, 0)^\top$ ($k_{\mathrm{orth}}^\top k_4 = 0$).
   - `evaluate_four_channel_airlock` certifies that the legacy 3-channel `Track S31` check ($\Delta W, \Delta\mathrm{KV}, \Delta P_{\mathrm{data}}$) returns **`ALLOW`** on the Channel-2B poisoning witness (`legacy_3ch_blind_to_channel2b = True`), whereas the upgraded 4-channel airlock (`Channel-2B: latent_state_hash_match / latent_delta_norm_sq / latent_cross_session_carryover`) returns **`ALLOW`** on the clean record and **`DENY`** on all `4/4` single-channel negative controls (`channel1_weight_tamper`, `channel2a_kv_reuse`, `channel2b_latent_poisoning`, `channel3_data_cot_drop`).

---

## 3. Audit Architecture & Zone-B Security Scope

- **Honest Shared-Code Scope Note:** `fisher_kintsugi_memt_audit.py` imports exact parsing, canonicalization, and standalone replay helpers from `fisher_kintsugi_memt_oracle.py`. `--check` therefore performs **full pure-$\mathbb{Q}$ deterministic replay + field-complete strict-keyset `secondary_audit_payload` + constant-time `hmac.compare_digest` verification** over canonical ASCII LF-only JSON bytes (independent zero-shared-code re-derivation is performed by external swarm verifiers).
- **Zone-B Hardening:**
  - Binding integer cap: `MAX_INT_BITS = 3321` is the binding fail-closed enforcement cap (`len(str((1 << 3321) - 1)) == 1000`), paired with necessary (not sufficient) pre-parse decimal-length cap `MAX_INT_DIGITS = 1000` (integers with `1,000` digits above $2^{3321}-1$, such as $10^{1000}-1$ which has `3,322` bits, fail closed cleanly with `INT_TOO_LARGE` — zero uncaught `ValueError`) across `parse_exact_int_ascii`, `format_exact_rational`, `format_decimal_trunc`, `validate_json_tree_iterative`, `canonical_json_bytes`, and `parse_strict_json_text` (with UTF-8 byte cap `MAX_JSON_BYTES = 1 MiB` and iterative depth cap `MAX_JSON_DEPTH = 32`).
  - Bijective `bool`/`int` separation (`_BOOL_LEAF_KEYS` -> `NON_BOOL_IN_BOOL_FIELD`, `BOOL_IN_NON_BOOL_FIELD`, `BOOL_IN_LIST`) enforced in `validate_certificate_payload(..., full_replay=False)` AND field-complete strict-keyset `secondary_audit_payload` enforced in `verify_certificate_dict(..., full_replay=False)` — including direct re-derivation of exact `GDN-2` `S4`/`S5` canonical SHA-256 digests (`hmac.compare_digest`) and exact `[X_(1081), X_(1096)]` order-statistic rank pinning (`37` re-signed `full_replay=False` tamper kills tested in `test_08`).
  - Atomic per-component `dir_fd` (`O_RDONLY | O_DIRECTORY | O_NOFOLLOW`, zero `os.path.islink`, `$PWD` symlink check, deleted-CWD `CWD_UNAVAILABLE` `rc=2`), `65,536`-B streaming `read_regular_bytes_capped` (`max_bytes <= 64 MiB`, `/proc/kallsyms` capped), durable `0o600` atomic writer (`type(data) is bytes` upfront, target `st_nlink <= 1`, thread-locked monotonic counter + `secrets.token_hex(8)`, 32-attempt `EEXIST` retry, `os.fsync(tmp_fd)`, pre-replace `fstat(tmp_fd).st_nlink == 1`, `os.replace(..., dir_fd)`, `os.fsync(dfd)`), and strict manual CLI parser (`allow_abbrev=False` semantics rejecting `--stand`, `--js`, `--write --json`, `--write=--json`, `--check=--json` with `rc=2` and `0` files created).

---

## 4. Post-Last-Edit Sequential Verification Receipts (`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`)

| Gate | Scope & Command Class | Result |
|---|---|---|
| **5-Mode Parity** | `-I -B`, `-I -O -B`, `-I -B -W error`, `-I -O -B -W error`, `PYTHONHASHSEED=42 -B` (without `-I`) | **PASS `5/5`**: zero-arg `915d8125...` ×5, `--standalone --json` `ba44be55...` (== disk `0o600`) ×5, `--check --json` `da0475c0...` ×5, `test_r5` `12/12 OK` ×5 |
| **Live Default-CLI `aletheia v2.0.0 mutate`** | `--max-mutants 0 --work-root . --add-cwd --timeout 45` on both `oracle.py` & `audit.py` | **PASS `166/166` (`100.0%`, `survived=0`, `truncated=False`)**: `oracle.py` **`102/102`** + `audit.py` **`64/64`** |
| **Exhaustive AST `BoolOp` (`OP_SWAP+SUBCOND_DEL`)** | All `35` `ast.BoolOp` nodes (`timeout=30`) across `oracle.py` & `audit.py` | **PASS `111/111` (`100.0%`, `surv=[]`)**: `oracle.py` **`93/93`** (`29` nodes) + `audit.py` **`18/18`** (`6` nodes) |
| **Live `Track S19c` Zone-B Fuzzer (ALL `4` Surfaces)** | CLI (`--check`) + `validate(full_replay=True)` + `validate(full_replay=False)` + `verify_dict(full_replay=False)` | **PASS `77/77` (`100.0%`, `bypasses=0`, `crashes=0`)**: `23/23` CLI + `18/18` FR1 + `18/18` FR0 + `18/18` FR0+Sec |
| **Live Concurrency & Kernel Races** | `16T×10` distinct + `80T` barrier burst + `renameat2(RENAME_EXCHANGE)` + `os.link`/`os.unlink` | **PASS**: `16T` + `80T` (`0o600`, `0` errs, `0` tmps); `7,618` `renameat2` swaps (`leaks=0, clobbers=0, tmps=0`); `1,292` hardlink flips (`tmps=0`) |
| **Static AST & `aletheia audit-check`** | `audit-check . --strict-ancestors --safety-scan --scripts fisher_kintsugi_memt_audit.py test_fisher_kintsugi_memt_r5.py` | **PASS**: `0` AST floats/`float()`/bare `assert`s/`.isdigit()`/`islink`/`random` across `3/3` `.py`, `0` JSON floats, `2/2` R4, `0` safety/ancestor/leak/CRLF findings, `PYC=0` |

---

## 5. Frozen File Manifest (`5` Files)

| File | Lines | Bytes | Mode | SHA-256 |
|---|---:|---:|:---:|---|
| `fisher_kintsugi_memt_oracle.py` | 1,355 | 58,803 | `644` | `a2b458d3e8e78b5807ec26228acc5ed89491a12fb1fe14498875315c7b046169` |
| `fisher_kintsugi_memt_audit.py` | 1,073 | 45,344 | `644` | `68104af983094015d7eb1a50045e7c9e7b040fdfeeeae3033098699484abfacb` |
| `test_fisher_kintsugi_memt_r5.py` | 963 | 54,832 | `644` | `30cfe8c4ef4a5245060f2d8966ce54188c89feb4d017e9ec85557fa626058aee` |
| `fisher_kintsugi_memt_certificate.json` | 1 | 9,910 | `600` | `ba44be55ca16873671b41f7e0cac32cf5c43361a07cff7c4ff7b04fcc917cee7` |
| `FISHER_KINTSUGI_MEMT_REPORT.md` | — | — | `644` | *(self; verified by `audit-check` below)* |
