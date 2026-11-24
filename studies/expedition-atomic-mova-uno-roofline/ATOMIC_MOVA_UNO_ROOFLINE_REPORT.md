# Track S45 — SSD Expert-Streaming Roofline, Zero-Draft `Uno` vs. External Drafter Budget Crossover & `MoVA` Fisher Quantization Sensitivity

**Track ID:** `TRACK-S45-ATOMIC-ENGINE-MOVA-UNO-ROOFLINE`  
**Certificate Schema:** `aletheia.atomic_mova_uno_roofline.certificate.v1`  
**Scope & Novelty Ledger:** `novelty_tier = N1` (exact-$\mathbb{Z}/\mathbb{Q}$ roofline, budget-crossover, and sparse attention-value Fisher concentration theorems); `published_benchmarks_novelty = N0` (`AtomicBot-ai/atomic-engine` Nov 23 2026 Apple M4 Max `48 GB` Inkling 2 Mini, `Strata SQ-10`, and `K2-Horizon` `arXiv:2611.21003v1`); `oracle_class = exact`; `measured_live_m4_max_hardware_run = false`; `measured_live_k2_horizon_checkpoint_eval = false`; `zero_trust_path_floats = true`; `zero_bare_asserts = true`; `zero_mathematical_rng = true`.

---

## 1. Executive Summary

Track S45 formalizes **four pure-$\mathbb{Z}/\mathbb{Q}$ systems and quantization pillars** (zero floating-point arithmetic, zero bare `assert` statements, zero RNG, zero third-party dependencies) connecting bounded SSD-to-unified-memory expert paging (`AtomicBot-ai/atomic-engine`), lossless speculative decoding (`DSpark`/`DFlash`/`woodcock` vs. zero-draft `K2-Horizon` `Uno`), and sparse Mixture-of-Value Attention (`K2-Horizon` `MoVA` + softplus gating):

1. **Pillar 1 — Published M4 Max Inkling 2 Mini Anchor Ratios (`N0`) & Discrete Zipf-1 SSD Expert-Streaming Roofline (`N1`):**
   - Certifies the exact published ratios (`109.53 / 33.11 = 10953/3311 = 3.3080... > 33/10` at `18 GiB + 100 MiB KV`, TTFT ratio `709/957 < 3/4`, `Strata SQ-10` `112.92 / 109.53 = 3764/3651` and TTFT speedup `709/421`).
   - Certifies the discrete Zipf-1 (`E = 16`, $H_{16} = 2436559/720720$, `M_base = 2 GiB`, $T_{\mathrm{res}} = 9\text{ ms}$, $W_{\mathrm{miss}} = 280\text{ MB}$ at $B_{\mathrm{SSD}} = 7000\text{ MB/s} \implies T_{\mathrm{ssd,max}} = 40\text{ ms}$) roofline $T_{\mathrm{tok}}(C) = T_{\mathrm{res}} + (1 - h(C)) W_{\mathrm{miss}} / B_{\mathrm{SSD}}$, proving the sharp SSD-dominance boundary between `9 GiB` (`C = 7`, $T_{\mathrm{ssd}} = 9.3219\text{ ms} > 9\text{ ms}$) and `10 GiB` (`C = 8`, $T_{\mathrm{ssd}} = 7.8429\text{ ms} < 9\text{ ms}$), and exact reversibility across the live mid-session budget resize **`20 GiB -> 8 GiB -> 20 GiB`** (where SSD latency vanishes to `0` at `20 GiB`, dominates at `8 GiB` with exact slowdown `48760831/21929031` $\approx 2.2235\times$, and returns identically to `9 ms` on regrow).
2. **Pillar 2 — Budget-Crossover Theorem: External Speculative Drafter (`DSpark`) vs. Zero-Draft `K2-Horizon` `Uno` Under Fixed Memory Cap $B$:**
   - Couples the exact renewal-reward speedup $S(k, \alpha, c) = (1 + \sum_{j=1}^k \alpha^j)/(1 + k c)$ (verifying exact closed-form geometric identity) to unified-memory cache displacement $\Delta M_{\mathrm{draft}}$.
   - Comparing external `DSpark`-class ($k=5, \alpha=4/5, c=1/25 \implies S_{\mathrm{ext}} = 3843/1250 = 3.0744\times > 3\times$, $\Delta M_{\mathrm{ext}} = 4\text{ GiB}$) against zero-draft `Uno` ($S_{\mathrm{uno}} = 11/5 = 2.2000\times$, $\Delta M_{\mathrm{uno}} = 0\text{ GiB}$), certifies the **exact rational crossover**:
     $$B^* = \frac{3625767011}{304898880}\text{ GiB} \approx 11.8917\text{ GiB} \qquad (\text{exact residual } = 0).$$
   - Strictly below $B^*$ (`6, 8, 10, 11 GiB`), the external draft model's `4 GiB` footprint evicts hot experts onto NVMe and **zero-draft `Uno` strictly wins** (e.g., `109.93 > 98.37 tok/s` at `8 GiB`; `90.34 > 62.74 tok/s` at `6 GiB`); strictly above $B^*$ (`12, 16, 20, 22 GiB`), **external `DSpark` strictly wins** (`341.60 > 244.44 tok/s` at `22 GiB`).
3. **Pillar 3 — `K2-Horizon` `MoVA` (Mixture-of-Value Attention) & Softplus Gating × AM-GM Fisher Water-Filling & Outlier-Expert Task-Break Hazard:**
   - Certifies `K2-Horizon` `MoVA` active-parameter ratios (`36B-A4B` vs `32B` dense: `4/36 = 1/9`, `4/32 = 1/8`, reported matched-quality active fraction `1/7`) and exact rational softplus gating derivatives $\sigma(z) = r/(1+r) \in \{1/5, 1/3, 1/2, 2/3, 4/5\} \subset (0, 1)$ at $r = e^z \in \{1/4, 1/2, 1, 2, 4\}$ (strictly monotone, zero dead region on $z < 0$).
   - Under the log-convex rate-distortion law $K_i(b_i) = 2^{e_i} \cdot 2^{-\alpha b_i}$ ($\alpha = 2, \bar b = 4\text{ bpw}, m = 8$), dense `V` (`exponents = [1,1,1,1,-1,-1,-1,-1]`) yields AM-GM water-filling gain $R_{\mathrm{dense}} = 5/4 = 1.2500\times$, whereas sparse top-$k$ `MoVA` (`exponents = [7,-1,-1,-1,-1,-1,-1,-1]`, optimal `b* = [15/2, 7/2, ..., 7/2]` at `0` extra average bpw) concentrates per-expert Fisher weight and achieves **$R_{\mathrm{mova}} = 263/16 = 16.4375\times > 12\times$** (`263/20 = 13.15×` stronger leverage than dense `V`), with certified exact convexity excess `+1/512 > 0` under zero-sum bit perturbations.
   - Certifies the sharp outlier-expert task-break hazard: naive uniform `4 bpw` flips the hot `MoVA` expert's critical logit margin from `+1` to **`-1/2 < 0`** (discrete task failure), whereas `0`-extra-bpw Fisher water-filling keeps the post-quantization margin at **`+7/8 > 0`** (task preserved).
4. **Pillar 4 — Honest `N0`/`N1` Scope Ledger & Exact `64/64` Lossless Token-ID Equivalence Gate:**
   - Explicitly partitions `3` published `N0` benchmark sources from `3` `N1` exact-$\mathbb{Z}/\mathbb{Q}$ theorems.
   - Certifies `verify_greedy_token_parity()` over `64`-token integer sequences (`vocab_size = 128,000`): exact **`64/64`** match (`reference_seq_sha256 = b7fd068cf84a5d5a1d3f4be0eddef407d94343b7625ac89d7779a951e86a4474`, `first_mismatch_index = None`) vs fail-closed rejection on a single-token drift at index `63` (`63/64`, `candidate_seq_sha256 = 713ac8403c0769b62728b3bace04108d729ff99bdc6c1f9353b95ce10a8df469`, `exact_match = False`).

---

## 2. Exact Certified Tables (Pure $\mathbb{Z}/\mathbb{Q}$, Toward-Zero Truncated Decimals)

### 2.1 Pillar 1 — Zipf-1 SSD Expert-Streaming Roofline ($E=16, H_{16}=2436559/720720, T_{\mathrm{res}}=9\text{ ms}, T_{\mathrm{ssd,max}}=40\text{ ms}$)

| Budget `B` (GiB) | Cached Experts `C` | Hit Rate `h(C)` (4dp) | `T_ssd` (ms, 4dp) | `T_tok` (ms, 4dp) | `TPS` (2dp) | SSD Dominated | SSD Vanishes |
|---:|---:|---:|---:|---:|---:|:---:|:---:|
| `2` | `0` | `0.0000` | `40.0000` | `49.0000` | `20.40` | `true` | `false` |
| `6` | `4` | `0.6162` | `15.3504` | `24.3504` | `41.06` | `true` | `false` |
| `8` | `6` | `0.7246` | `11.0121` | `20.0121` | `49.96` | `true` | `false` |
| `9` | `7` | `0.7669` | `9.3219` | `18.3219` | `54.57` | `true` | `false` |
| `10` | `8` | `0.8039` | `7.8429` | `16.8429` | `59.37` | `false` | `false` |
| `12` | `10` | `0.8663` | `5.3451` | `14.3451` | `69.71` | `false` | `false` |
| `16` | `14` | `0.9617` | `1.5282` | `10.5282` | `94.98` | `false` | `false` |
| `18` | `16` | `1.0000` | `0.0000` | `9.0000` | `111.11` | `false` | `true` |
| `20` | `16` | `1.0000` | `0.0000` | `9.0000` | `111.11` | `false` | `true` |

### 2.2 Pillar 2 — Budget-Crossover Operating Points ($B^* = 3625767011/304898880 \approx 11.8917\text{ GiB}$)

| Unified Budget `B` (GiB) | External `DSpark` (`S=3843/1250`, $\Delta M=4$ GiB) `TPS` (2dp) | Zero-Draft `Uno` (`S=11/5`, $\Delta M=0$ GiB) `TPS` (2dp) | Strictly Higher Throughput |
|---:|---:|---:|:---|
| `6` | `62.74` | `90.34` | `zero_draft_uno` |
| `8` | `98.37` | `109.93` | `zero_draft_uno` |
| `10` | `126.25` | `130.61` | `zero_draft_uno` |
| `11` | `139.84` | `141.67` | `zero_draft_uno` |
| `12` | `153.62` | `153.36` | `external_dspark` |
| `16` | `214.31` | `208.96` | `external_dspark` |
| `20` | `292.01` | `244.44` | `external_dspark` |
| `22` | `341.60` | `244.44` | `external_dspark` |

### 2.3 Pillar 3 — Dense `V` vs. Sparse `MoVA` Fisher Water-Filling ($\alpha=2, \bar b=4\text{ bpw}, m=8$)

| Projection Architecture | Sensitivity Exponents $e_i$ | Optimal Bit Allocation $b_i^*$ | $K_{\mathrm{uniform}}$ | $K_{\mathrm{Fisher}}$ | AM/GM Gain $R$ |
|:---|:---|:---|---:|---:|---:|
| Dense `V` | `[1, 1, 1, 1, -1, -1, -1, -1]` | `[9/2, 9/2, 9/2, 9/2, 7/2, 7/2, 7/2, 7/2]` | `5/128` | `1/32` | `5/4` (`1.2500x`) |
| Sparse `MoVA` | `[7, -1, -1, -1, -1, -1, -1, -1]` | `[15/2, 7/2, 7/2, 7/2, 7/2, 7/2, 7/2, 7/2]` | `263/512` | `1/32` | `263/16` (`16.4375x`) |

---

## 3. Zone-B Security & Verification Architecture

- **Binding Integer & JSON Bounds:** `MAX_INT_BITS = 3321` (`|n| <= 2^{3321} - 1`) is the binding fail-closed enforcement cap across both parsing (`parse_exact_int_ascii`) and serialization (`_require_bounded_int` / `validate_json_tree_iterative`). `MAX_INT_DIGITS = 1000` serves as a necessary (not sufficient) pre-parse ASCII decimal-length guard: `(1 << 3321) - 1` has exactly `1,000` decimal digits and round-trips cleanly, whereas `10^{1000} - 1` has `1,000` decimal digits and `3,322` bits and fails closed cleanly with `INT_TOO_LARGE` before any raw Python `ValueError` on large integer strings. `MAX_JSON_DEPTH = 32`, `MAX_JSON_BYTES = 1_048_576`, zero floats, zero `NaN`/`Infinity`, zero duplicate keys, and zero lone UTF-16 surrogates are enforced strictly.
- **Race-Free Atomic `dir_fd` I/O (`atomic_mova_uno_audit.py`):** Concrete `pathlib.Path` enforcement, NUL/surrogate refusal, deleted-CWD `CWD_UNAVAILABLE` wrapping, step-by-step `openat(O_RDONLY | O_DIRECTORY | O_NOFOLLOW)` traversal + logical `$PWD` symlink walk (zero `os.path.islink`), `65,536`-B streaming capped reads (`<= 64 MiB`), and atomic `0o600` writes using `threading.Lock()` + monotonic counter + `secrets.token_hex(8)` + 32-attempt `EEXIST` retry, `os.fsync(tmp_fd)` (line `244`), pre-replace `os.fstat(tmp_fd).st_nlink == 1`, target `st_nlink <= 1` (`HARDLINK_FORBIDDEN`), `os.replace(..., src_dir_fd=dfd, dst_dir_fd=dfd)`, and `os.fsync(dfd)` (line `250`).
- **Field-Complete Strict-Nested-Key Secondary Audit (`secondary_audit_payload`):** Enforced under both `full_replay=True` and `full_replay=False` over exact keysets and every algebraic witness across all 4 pillars (`45/45` re-signed semantic & nested-key tampers rejected in `test_07`).
- **Strict Manual CLI Parser:** Zero flag abbreviations (`--st`/`--wr`/`--ch`/`--js` -> `UNKNOWN_FLAG`, `rc=2`), rejecting space- and equals-syntax flag-like/empty paths (`--write --json`, `--write=--json`, `--write ""` -> `rc=2`, `0` files created).

---

## 4. Verification Receipts & Frozen Manifest

All checks executed sequentially with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`:

- **Certificate Digests (`atomic_mova_uno_certificate.json`, mode `0o600`):**
  - `file_sha256`: `651338833ae035bd0cb91709fab1222ca3976e8f87d14bfcc475998410019270`
  - `payload_sha256`: `b9b359fdb95423f5508693b765cee339b9c43cd491677407c8587f951d109127`
  - `certificate_sha256`: `f8b0e4f775226e3e88fdf4863bfb33d7791ad5fac98cfb34507a9d4df31b45a8`
- **5-Mode Parity (`-I -B`, `-I -O -B`, `-I -B -W error`, `-I -O -B -W error`, `PYTHONHASHSEED=42 -B` without `-I`):**
  - `--standalone --json`: bit-identical `651338833ae035bd0cb91709fab1222ca3976e8f87d14bfcc475998410019270` ×5.
  - `--check atomic_mova_uno_certificate.json --json`: `valid = true` ×5.
  - `test_atomic_mova_uno_r5.py`: **`10/10 OK` ×5** (`0` warnings, `0` AST floats, `0` bare asserts, `45/45` re-signed non-replay tampers killed).
- **Live Default-CLI `aletheia v2.0.0 mutate --max-mutants 0 --work-root . --add-cwd`:**
  - `atomic_mova_uno_oracle.py`: **`52/52` (`100.0%`, `survived = 0`, `truncated = False`)**.
  - `atomic_mova_uno_audit.py`: **`76/76` (`100.0%`, `survived = 0`, `truncated = False`)**.
  - Combined: **`128/128` (`100.0%`, `0` survivors)**.
- **Exhaustive `SWAP + SUBCOND_DEL` AST `BoolOp`:**
  - `atomic_mova_uno_oracle.py`: **`48/48`** (`16` nodes, `survived = 0`, `timeouts = 0`).
  - `atomic_mova_uno_audit.py`: **`21/21`** (`7` nodes, `survived = 0`, `timeouts = 0`).
  - Combined: **`69/69` (`100.0%`, `0` survivors)**.
- **Live 4-Surface Adversarial Zone-B Suite:** **`18/18` CLI + `16/16` `full_replay=True` + `16/16` `full_replay=False` (`secondary_audit_payload`) + `16/16` Direct Helper API = `66/66` (`100.0%`, `bypasses = 0`, `crashes = 0`)**.
- **Live Concurrency & Kernel Race Stress:** `16T×25` distinct (`400/400`, `0o600`), `16T×25` same-target (`400/400`, `0o600`), `80T` burst (`80/80`, `0o600`), `renameat2(RENAME_EXCHANGE)` (`196,887` swaps, `0` leaks, `0` clobbers, `victim` intact), `os.link`/`os.unlink` hardlink race (`100,210` flips, `0` unexpected errs), `0` leftover temp files.
- **Static Hygiene (`aletheia v2.0.0 audit-check . --strict-ancestors --safety-scan`):** **`PASS`** (`5` files, `2/2` R4 scripts checked/passed, `0` findings, `PYC = 0`).

### Frozen 5-File Manifest

| File | Lines | Bytes | Mode | SHA-256 |
|:---|---:|---:|:---:|:---|
| `atomic_mova_uno_oracle.py` | 821 | 36,467 | `0644` | `afb2afeea4183f49164f2fff5d6a4457174d79b20d1d1ee07fce295f6a8a6b89` |
| `atomic_mova_uno_audit.py` | 834 | 38,309 | `0644` | `d0a3c3b1b1198030c831781b8a126dddeb75b8a967a9e27349362739665e3ffa` |
| `test_atomic_mova_uno_r5.py` | 826 | 44,557 | `0644` | `1c6a009506eca88b9ced29b3e926e2c87fd81c908fe4c7c8d24eae28209417fc` |
| `atomic_mova_uno_certificate.json` | 1 | 9,567 | `0600` | `651338833ae035bd0cb91709fab1222ca3976e8f87d14bfcc475998410019270` |
| `ATOMIC_MOVA_UNO_ROOFLINE_REPORT.md` | *(this report)* | — | `0644` | *(verified by manifest receipt)* |
