# Wave 2 Round 8 — Arm R8-3 (`AUDIT-074`): Exact Pure-$\mathbb{Q}$ Mathematical, Cryptographic & Compiler-IR Tomography of `loom-fun/unsloth-Inkling-2-Mini-loom`

- **Arm / Owner:** Arm R8-3 (`fractals-probe`), Wave 2 Round 8 (`AUDIT-074`)
- **Pinned External Artifact (Strictly Read-Only, `0` External Writes, `0` Multi-GB Weight Downloads):** `https://huggingface.co/loom-fun/unsloth-Inkling-2-Mini-loom` @ commit `277f099b5003b725e099d48caa1a2c0db1f9b650` (Apache-2.0, `15` siblings; `14` lightweight metadata/config/`.loom` files totaling `5.85 MB` verified byte-for-byte; `12.24 MB` LFS `tokenizer.json` `a45fdc4e...` excluded from heavy downloads; upstream weights at `unsloth/Inkling-2-Mini-GGUF @ ea0c79b2ecd73669e774c8519b79bfabfddbc027`).
- **Oracle Class & Honest Law-4 Novelty Tag:** `exact+anchor` (100% pure Python 3 stdlib `int`/`Fraction`, **ZERO floats** anywhere in code or JSON, **ZERO external deps**, **ZERO bare `assert`s**) | **`N1`** (live compiler-IR bug discovery `LOOM-IR-01..05`, cryptographic lockfile binding audit `LOOM-LOCK-01`/`LOOM-PROV-01`, exact pure-$\mathbb{Q}$ lossless speculative coupling separation `THEOREM S1–S2`, exact `Q8_0 -> Q4_0` proposal-head clipping/argmax-flip proof, exact Strix-Halo MoE coupon-collector roofline & looped-core memory-geometry derivation; all classical/upstream foundations strictly quarantined under `R8-N0-03/08/09/10`).
- **Artifacts (`studies/expedition-inkling-loom-tomography/`):**
  - [inkling_loom_tomography.py](inkling_loom_tomography.py) — Zero-float pure-$\mathbb{Q}$ verifier, Ring-0 lockfile governor, live `.loom` IR anchor checker, speculative coupling oracle, `Q8_0 -> Q4_0` block quantizer, and MoE roofline/memory-geometry engine (`--repo-dir` optional; runs 100% standalone)
  - [test_inkling_loom_r5.py](test_inkling_loom_r5.py) — R1–R5 invariant, boundary, tamper-detection, and `100.0%` mutation-kill test suite
  - [inkling_loom_certificate.json](inkling_loom_certificate.json) — Live-anchored zero-float canonical JSON certificate (`canonical_sha256 = 69e92b234d0881c97c0a45eae3e9815bcc9fad47cb6e3ca92e3b0129de17cae6`)
  - [INKLING_LOOM_AUDIT_REPORT.md](INKLING_LOOM_AUDIT_REPORT.md) — This report

---

## 1. Strict Law-4 `N0` Prior-Art Quarantine Table (Aligned with `ROUND8_PORTALS.md` `R8-N0-03/08/09/10`)

| Quarantine ID | Established Prior Art / External Artifact (`N0`) | Role in `AUDIT-074` |
|---|---|---|
| `R8-N0-03` | Lossless speculative decoding & maximal-coupling rejection sampling (Leviathan–Kalman–Matias 2023; Chen et al. 2023), Medusa/Eagle/DFlash/DSpark parallel drafting, standard SHA-256 lockfiles, Murmur3/LowBias32 counter PRNGs, AMD ROCm/Loom (`ROCm/hrx-system`, `loom-fun/unsloth-Inkling-2-Mini-loom@277f099b`) | Target artifact & classical speculative-sampling baselines |
| `R8-N0-08` | Classical exact `Fraction` arithmetic, coupon-collector occupancy expectation, GGUF `Q8_0`/`Q4_0` block quantization (`llama.cpp`), and convex-combination contraction | Exact algebraic tools |
| `R8-N0-09` | Prior verified expedition studies (`siliconcordat/loom#10`, `Moissanite-Labs/weft-loom#42` `AUDIT-023`, `Moissanite-Labs/siliquant#24/#26` `AUDIT-068R`, `Moissanite-Labs/scry#4` `AUDIT-032`, `Moissanite-Labs/warp-drive#12/#13/#15` `AUDIT-064`, `Foundation-0/aletheia v1.4.0`) | Cross-repository engineering bridges |
| `R8-N0-10` | All `14` read-only pinned files from `loom-fun/unsloth-Inkling-2-Mini-loom@277f099b` (`README.md`, `REPORT.md`, `info.json`, `llm_config.json`, `model.json`, `model.json.lock`, `weights.loom`, `head_weights.loom`, `program/*.loom`, `.gitattributes`) | Verbatim read-only evidence anchors |

*(Naming clarity: **AMD Loom** (`ROCm/hrx-system`, `hrx-loom`) is AMD's Strix-Halo GPU compiler/runtime inspected here; **`siliconcordat/loom`** (`#10`) is Silicon Cordat's hardware-physics & gate-contract verification repository bridged in Section 6.)*

---

## 2. Law-2 Steelman Credit (What AMD Loom & `Inkling-2-Mini-loom` Got Right)

Before detailing verbatim live bugs and structural gaps, four genuine engineering strengths are verified directly from the `.loom` IR and pinned measurements:
1. **Counter-Based Exact-Match Speculative Sampling Is Bit-Exact (`Theorem S1`, `llm.loom:18080-18124, 32520-32746`):** `@g_spec_rows` assigns `position = start + j + 1`, `@g_head_select` hashes `(seed, position)` via a counter-based 32-bit PRNG independent of the draft token, and `@g_spec_accept` accepts draft `j+1` iff `pred[r+j] == cand` (`L32681`). Our exact pure-$\mathbb{Q}$ emulator proves **`32/32` (`100%`) bit-exact trajectory equivalence** against non-speculative decoding across seeds (`T >= 0`).
2. **In-Place Safe Shortconv Replay Isolation for `row < 16` (`llm.loom:21961-22099`):** Speculative spans (`flag bit 4`) suppress recurrent-state publication (`llm.loom:18813`), capturing per-token inputs into `%sc_cap` (`163,577,856 B`) and committing ONLY `consumed` accepted tokens in `@g_ink_sconv_replay_all` (`val0, val1, val2` loaded before write so `reader == writer` is in-place safe — zero speculative taint for `row < 16`).
3. **Genuine Non-Drafted Compute Speedup on Strix Halo:** Even with drafting completely disabled on both engines (`streams4`), Loom delivers **`84.6 vs 38.0 tok/s` (`423/190 = 2.2263x`, `+123%`)**, and prefill delivers **`+653%` to `+996%`** over `llamacpp-rocm/b1383`. Moreover, because `streams4`'s published `379.1 tok/s` roofline assumes `"basis": "shared-experts"`, under realistic independent MoE routing (`E=128, k=8`, `E[U(4)] = 29.123`) the bandwidth ceiling drops to **`185.24 tok/s`**, meaning Loom actually achieves **`45.67%`** of independent-routing bandwidth (not `22%`).
4. **Strict Contraction of `DSpark` Markov-RNN State Gate (`llm.loom:34342-34348`):** `st_new = gate * (st_old - cand) + cand` with `gate = sigmoid(u0) in (0,1)` and `cand = tanh(u1) in (-1,1)` is an exact convex contraction preserving `(-1, 1)` unconditionally.

---

## 3. Module 1 — Cryptographic Lockfile Binding & Provenance Audit (`LOOM-LOCK-01`, `LOOM-PROV-01`)

| Finding ID | Verbatim Anchor | Exact Quantitative / Structural Defect |
|---|---|---|
| **`LOOM-LOCK-01`** | `model.json.lock:1-33`, `model.json:7-26` | `model.json.lock` (`schema_version = 2`, `scope = "assets"`) hashes ONLY `3` passive blobs (`18,044,280,951 B`) and pins **`0 / 9`** executable/policy/config files (**`5,827,121 B`** unpinned: `program/llm.loom` `5,539,325 B`, `program/control.loom` `223,546 B`, `program/service.loom` `26,042 B`, `program/prepare.loom` `16,155 B`, `program/storage.loom` `12,748 B`, `head_weights.loom` `3,098 B`, `weights.loom` `121 B`, `llm_config.json` `4,427 B`, `model.json` `1,659 B`). |
| **`LOOM-LOCK-02`** | `REPORT.md:11`, `control.loom:1`, `llm.loom:1,5`, `prepare.loom:1` | `REPORT.md:11` states `programs_sha256` (`dd38151f...`) hashes `program/*.loom` *"without each file's first-line generator comment"*, YET **`3 / 5`** files (`control.loom:1`, `llm.loom:1`, `prepare.loom:1`) have **executable IR code on line 1** (`// Generated by gen_llm.py` is on `llm.loom:5`), and load-time weight-conversion policies (`head_weights.loom`, `weights.loom`) live outside `program/*.loom`. |
| **`LOOM-PROV-01`** | `README.md:53,63,90,99,114`, `REPORT.md:7,18,23,26,52,74`, `info.json:134,255,277,351,369`, `llm_config.json:203-220` | **Provenance split-brain & dirty kit:** `README.md`/`REPORT.md`/`info.json` claim `hrx-loom unknown`, `loom-skills unknown`, `card_generator unknown`, while `llm_config.json:215-220` records `sdk.hrx_loom = "0.2.0"` (`wheel_sha256: 5148f9e3...`, `hrx_system_sha: 0cd370f2...`) alongside `kit.modified = true` and `dirty_choices = ["moe.decode.rows_per_wg"]`. `REPORT.md:18` asserts *"Every performance value comes from a `devlab-perf` job on an AMD DevLab Dispatch Strix Halo runner"*, contradicted by `README.md:63`, `REPORT.md:23`, and `info.json:255` (`source: "machine"`, *"this GPU has no DevLab runner"*, `commands: []`). Gates `kernel_tests` and `logits` have `❌ no evidence`. |

**Fail-Closed Ring-0 Closure (`verify_loom_lockfile_bundle`):** Rejects the published bundle with **`16` exact defects** (`REJECTED_UNBOUND_LOCKFILE`, `0/9` executables pinned) and promotes only when all `9/9` executable/policy/config digests are SHA-256-bound, SDK/skills provenance is clean hex/non-unknown, `kit_modified == False`, `dirty_choices == []`, and `kernel_tests`/`logits`/methodology gates pass (`PROMOTED_RING0_BOUND`).

---

## 4. Module 2 — Exact Pure-$\mathbb{Q}$ Speculative Coupling (`Theorem S1–S2`) & Live `.loom` IR Bugs (`LOOM-IR-01..05`)

### 4.1 Theorem S2: Exact Acceptance-Rate Deficit (Loom Exact-Match vs. Optimal LKM Maximal Coupling)
Because `program/llm.loom:33323` drafts greedily (`// step_meta: greedy (temp=0)`) and `program/llm.loom:32681` accepts iff `pred == cand`, at $T > 0$ Loom's acceptance probability is strictly capped by $\alpha_{\mathrm{Loom}} = p(\arg\max q) \le \|p\|_\infty$, whereas optimal Leviathan–Kalman–Matias / Chen et al. maximal coupling achieves exact target marginal `p` identically with $\alpha_{\mathrm{LKM}} = \sum_x \min(p_x, q_x) = 1 - \delta_{\mathrm{TV}}(p, q)$:

| Exact Pure-$\mathbb{Q}$ Distribution Pair $(p, q)$ | $\delta_{\mathrm{TV}}(p, q)$ | Optimal $\alpha_{\mathrm{LKM}}$ | Loom Greedy-Match $\alpha_{\mathrm{Loom}}$ | Indep. Match $\alpha_{\mathrm{indep}}$ | Gap $\alpha_{\mathrm{LKM}} - \alpha_{\mathrm{Loom}}$ | Ratio $\alpha_{\mathrm{LKM}} / \alpha_{\mathrm{Loom}}$ |
|---|---:|---:|---:|---:|---:|---:|
| Perfect Drafter $p = q = \mathrm{Uniform}(2)$ | `0/1` | **`1/1`** | `1/2` | `1/2` | `+1/2` | `2/1` (`2x`) |
| Perfect Drafter $p = q = \mathrm{Uniform}(4)$ | `0/1` | **`1/1`** | `1/4` | `1/4` | `+3/4` | `4/1` (`4x`) |
| Perfect Drafter $p = q = \mathrm{Uniform}(8)$ | `0/1` | **`1/1`** | `1/8` | `1/8` | `+7/8` | `8/1` (`8x`) |
| Perfect Drafter $p = q = \mathrm{Uniform}(16)$ | `0/1` | **`1/1`** | `1/16` | `1/16` | `+15/16` | `16/1` (`16x`) |
| Perfect Drafter $p = q = \mathrm{Uniform}(64)$ | `0/1` | **`1/1`** | **`1/64`** | `1/64` | **`+63/64`** | **`64/1` (`64x`)** |
| Skewed $p=(\frac{1}{2},\frac{1}{4},\frac{1}{8},\frac{1}{8}), q=(\frac{2}{5},\frac{3}{10},\frac{3}{20},\frac{3}{20})$ | `1/10` | **`9/10`** | **`1/2`** | `5/16` | **`+2/5` (`+40 pp`)** | **`9/5` (`1.8x`)** |

### 4.2 Five Verbatim Live Compiler-IR Findings (`LOOM-IR-01..05`)

1. **`LOOM-IR-01` [HIGH — Live Silent `top_k` Corruption Bug] (`control.loom:314-367, 2642-2646` + `llm.loom:17996-18002`):**
   - `control.loom:314,348-367` overloads bit 30 (`1073741824 = 1 << 30`) of `top_k` as `"logits": true`; `control.loom:2642,2646` copies it **unmasked** into `packet[7]`; `@g_head_select` (`llm.loom:17996-18002`) loads `%top_k_raw = packet[7]` **without masking `& 1073741823`** (`1073741823` occurs `0` times in all `79,039` lines of `llm.loom`!).
   - Consequently, ANY stochastic request (`temperature > 0`) with `"logits": true` and `1 <= top_k <= 63` (e.g., `top_k = 1` $\to$ `1073741825`, `top_k = 5` $\to$ `1073741829`) evaluates `minsi(top_k_raw, 64) = 64` — **silently overwriting `top_k` with `64`**!
   - **Law-2 Repair:** Insert `%top_k_clean = scalar.andi %top_k_raw, 1073741823 : i32` at `llm.loom:17997` (verified in `simulate_loom_top_k_bit30(..., apply_fix=True)`).
2. **`LOOM-IR-02` [HIGH — Missing Logits Export Under Speculation] (`llm.loom:61800` vs `61809–77073`):**
   - `@g_logits_export` is launched ONLY inside `@llm_advance` (`L61800`) and is **completely omitted from `@llm_verify` (`L61809–77073`)**, leaving host logits readback broken whenever default `draft_depth = 1` speculation is enabled (and `README.md:99` `logits` gate has `❌ no evidence`).
3. **`LOOM-IR-03` [MED — Silent `top_k = 64` Truncation & `scry#4` Tail-Mass Extinction] (`control.loom:259` + `llm.loom:17998-18078`):**
   - `top_k = 0` (standard OpenAI API default) silently clamps to `k = 64` out of `vocab = 200,058` (**`32/100029 ≈ 0.032%`** of vocabulary) and normalizes `top_p` strictly over those `64` logits. On a uniform-head distribution (`head_count = 64, head_mass = 1/2`) with `top_p = 9/10`, Loom keeps `58` tokens (`29/64` true mass) and drops **`35/64 = 54.69%`** of true probability mass (`TV = 35/64`), forcing fixed-support tail extinction under iterated sampling (`Moissanite-Labs/scry#4` `AUDIT-032`).
4. **`LOOM-IR-04` [LOW/PERF — Hardcoded `block_tokens = 8` at `depth = 1`] (`prepare.loom:204-208` + `llm.loom:33318-33320, 78093`):**
   - `prepare.loom:204-205` comments that DFlash blocks should be `[anchor, <mask> x depth]` to stay in int8 matvec, YET `prepare.loom:207` and `g_dsp_prepare:33320` hardcode `block_tokens = 8`. At `depth = 1`, the 3 non-causal DSpark transformer layers compute all `8` slots while the Markov loop (`L78093`, `step = 1..1`) consumes only slot `1` and discards `6/8 = 3/4` (`75%`) of the mask slots. *(Law-2 caveat: because the 3 DSpark layers are bidirectional/non-causal, retaining trained block size `8` preserves the exact trained attention context at slot `1`, at the cost of `4x` token compute.)*
5. **`LOOM-IR-05` [HIGH — Silent Shortconv Replay Drop for `row >= 16`] (`llm.loom:21981-21989` vs `prepare.loom:121`, `storage.loom:31`):**
   - `@g_ink_sconv_replay_all` hardcodes `%c16_r = 16` (`%in_r = index.cmp ult, %row, %c16_r`, matching `%sc_cap` `view<16x8x48x6656xf32>`), whereas `prepare.loom:121` and `storage.loom:31` permit `rows <= 1,048,576` with NO `@require(rows <= 16)` guard! While the default preset (`rows = 4`) is unaffected, ANY deployment with `rows >= 17` and drafting enabled silently skips shortconv replay across all `48` logical layers for rows `16..rows-1` (`simulate_shortconv_replay_all(row_id=16)` leaves stale history `[11/10, 11/5, 33/10]` instead of `[11/5, 33/10, 101/10]`)!

---

## 5. Modules 3 & 4 — `Q8_0 -> Q4_0` Proposal-Head Quantization (`siliquant#24/#26`), MoE Coupon-Collector Roofline (`weft-loom#42`), & Looped-Core Memory Geometry (`warp-drive`)

1. **On-the-Fly `Q8_0 -> Q4_0` LM-Head Requantization (`head_weights.loom:45-71`, `llm.loom:34454-34521`):**
   - Exact tensor geometry: `vocab * hidden / 32 = 200,058 * 2,304 / 32 = 14,404,176` blocks (`Q8_0 = 489,741,984 B`; padded GGUF `201,024` rows `= 492,106,752 B` $\to$ `Q4_0 = 259,275,168 B`; exact compression **`9/17`**, bandwidth saved **`8/17 = 230,466,816 B`**).
   - **Asymmetric Clipping Theorem (`llm.loom:34488-34505`):** Because `d = m / (-8)` and `q = min(15, floor(v/d + 8.5))`, any symmetric block containing `{+M, -M}` (`M > 0`) reconstructs `+M` with `0` error (`q = 0`) while clipping `-M` to `-(7/8)M` (`q = 15`) — an exact **`1/8 = 12.5%`** amplitude loss.
   - **Exact Argmax-Flip Acceptance Collapse Witness:** On exact rational rows `A = (+8, -8, 0...)`, `B = (0, -15/2, 0...)` and activation `x = (0, -1, 0...)`, target `Q8_0` gives `z_A = 8/1 > z_B = 15/2` (`argmax = A`), while `Q4_0` draft head clips `-8 -> -7`, giving `z_A_q4 = 7/1 < z_B_q4 = 15/2` (`argmax = B`) and collapsing greedy acceptance **`1/1 -> 0/1`**.
2. **Exact Pure-$\mathbb{Q}$ Benchmark Forensics & MoE Coupon-Collector Occupancy Theorem:**
   - All `9/9` metrics in `info.json` vs `README.md`/`REPORT.md` verify within `<= 1/20` tok/s (`0` margin or roofline-% mismatches, `3/3` exact `tokens_per_pass`), exposing one IEEE-754 binary64 half-tie rounding artifact on `prefill_2k` (`(1225.0 + 1203.7)/2 = 24287/20 = 1214.35`, reported `1214.3` instead of exact half-to-even `1214.4`, diff `-1/20`).
   - **Apples-to-Oranges Decomposition:** Single-stream `decode` compares Loom **with DFlash depth-1 (`1.602 tok/pass`) + active CPU spin-wait + client timing** vs `llama.cpp` **no drafting + server timing**: `427/136 = 3.1397x (+214%)` reduces to **`53375/27234 = 1.9599x (+96%)`** per verify pass (`26.654 passes/s` vs `13.6`), consistent with drafting-off `streams4` **`423/190 = 2.2263x (+123%)`**.
   - **Exact MoE Coupon-Collector (`E = 128, k = 8`) Explaining `control.loom:4039` Occupancy Gate:**
     - $\mathbb{E}[U(1)] = 8$, $\mathbb{E}[U(2)] = 31/2 = 15.5$ (`1.9375x`), $\mathbb{E}[U(4)] = 14911/512 = 29.1230$, $\mathbb{E}[U(8)] = 1732076671/33554432 = 51.6199$ (`U(8)/U(4) = 1.7725x`).
     - From `REPORT.md:58` (`single_pass = 98.5, ceiling = 112.7, tpp = 1.602`), implied per-verify-pass bandwidth inflation is **`157797/112700 = 1.40015x`** (implied MoE bandwidth share **`42.68%`**), yielding break-even acceptance **`40.02%`** at `1 stream` and **`56.43%`** at `4 streams` **before counting drafter overhead** (`3`-layer `Q8_0` DSpark + `259.28 MB` `Q4_0` LM head) — mathematically explaining why `control.loom:4039-4075` automatically disables DFlash whenever `>= 2` streams are active!
3. **Exact Looped-Core (`n_phys = 24, n_passes = 3 -> 48` Logical Layers) & Single-Byte Memory Geometry:**
   - `6 + 3 * 12 + 6 = 48` logical layers = **`40` SWA** (`5 + 3*10 + 5`) + **`8` full-attention** (`1 + 3*2 + 1` $\implies$ **`16` KV planes** × `2048 B/pos`).
   - Shortconv (`48 * 3 * 6656 * 4 = 3,833,856 B`, `llm.loom:22147`) + `40` SWA rings (`40 * 1040 * 2048 * 2 = 170,393,600 B`, `llm.loom:22148`) = **`174,227,456 B`** (`storage.loom:41` EXACT!).
   - Capture scratch: `16 * 8 * 48 * 6656 * 4 =` **`163,577,856 B`** (`llm.loom:62013` EXACT!).
   - DSpark carry stride: `16,384 + 2 * (1024 * 2048 * 2) =` **`8,404,992 B`** (`storage.loom:78`, `llm_config.json:200` EXACT!); 4-tap residual row `4 * 2304 * 4 =` **`36,864 B`** (`storage.loom:77` EXACT!).

---

## 6. Bidirectional 5-Repository Expedition Bridge

| Own-Org Target Repo | Exact Bridge Established by `AUDIT-074` |
|---|---|
| **`siliconcordat/loom#10`** (`docs/09-gate-contracts.md`, `#10`) | Closes AMD Loom's unpinned lockfile (`LOOM-LOCK-01`) & missing `kernel_tests`/`logits` gate evidence via `verify_loom_lockfile_bundle`, and proves strict convex contraction of `DSpark`'s Markov-RNN state gate (`llm.loom:34342-34348`). |
| **`Moissanite-Labs/weft-loom#42`** (`AUDIT-023`) | Decomposes `README.md`'s `+214%` single-stream decode confound (`DFlash` + active CPU spin + client timing vs no-draft + server timing) to `+96%` per verify pass and `+123%` drafting-off `streams4`, plus `REPORT.md:18` DevLab vs named-machine contradiction. |
| **`Moissanite-Labs/siliquant#24/#26`** (`AUDIT-068R`) | Exact pure-$\mathbb{Q}$ audit of `head_weights.loom` / `@g_prep_q8_0_q4_0` (`9/17` bandwidth ratio, `1/8` asymmetric clipping error, exact argmax-flip acceptance collapse `1 -> 0`). |
| **`Moissanite-Labs/scry#4`** (`AUDIT-032`) | Exact pure-$\mathbb{Q}$ tail-extinction proof (`TV = 35/64 = 54.69%`) for `llm.loom:17998-18078`'s silent `top_k = 64` cap on `200,058`-vocab models + live bit-30 `top_k` corruption bug (`LOOM-IR-01`). |
| **`Moissanite-Labs/warp-drive`** (`#12`/`#13`/`#15`, `AUDIT-064`) | Exact looped-core (`n_passes = 3`, `48` logical layers) memory-geometry audit (`174,227,456 B` / `163,577,856 B` / `8,404,992 B`), speculative shortconv replay isolation + `row >= 16` silent-drop bug (`LOOM-IR-05`), and Ring-0 lockfile contract binding. |

---

## 7. Exact Reproduction Receipts & Falsification Criteria

```bash
# 1. 4-mode isolated -I determinism (standalone JSON + R5 unit tests)
for f in "-I -B" "-I -O -B" "-I -B -W error" "-I -O -B -W error"; do
  PYTHONDONTWRITEBYTECODE=1 python3 $f inkling_loom_tomography.py --json | sha256sum
  PYTHONDONTWRITEBYTECODE=1 python3 $f test_inkling_loom_r5.py
done

# 2. Live read-only mirror verification (against a local read-only mirror of rev 277f099b)
PYTHONDONTWRITEBYTECODE=1 python3 -I -O -B -W error inkling_loom_tomography.py \
  --repo-dir /path/to/inkling-loom --json | sha256sum

# 3. Aletheia v1.4.0 hygiene, strict-ancestors & 100.0% AST mutation audit
python3 -B -W error -m aletheia audit-check . \
  --strict-ancestors --scripts inkling_loom_tomography.py test_inkling_loom_r5.py --json
python3 -B -W error -m aletheia mutate inkling_loom_tomography.py \
  --work-root . --add-cwd --test-cmd "python3 -I -O -B -W error test_inkling_loom_r5.py" --json
```

**Falsification Criteria:** Fails if any of the `14` pinned files from `loom-fun/unsloth-Inkling-2-Mini-loom@277f099b` differs in size or SHA-256; if `llm.loom` masks bit 30 (`1073741823`), launches `@g_logits_export` inside `@llm_verify`, or removes the `row < 16` guard in `@g_ink_sconv_replay_all`; if any exact `Fraction` result (Theorem S1 `32/32`, Theorem S2 `9/10 vs 1/2`, `Q4_0` `1/8` clipping & argmax flip, `TV = 35/64`, MoE `E[U(m)]`, `174,227,456 B` / `163,577,856 B` / `8,404,992 B`, or `9/9` benchmark metrics) fails independent recomputation; if any red-team input bypasses `verify_loom_lockfile_bundle`; if any float or absolute home-directory path appears in code/JSON/report; or if any R4 / `audit-check --strict-ancestors` / `100.0%` `aletheia v1.4.0 mutate` gate fails.
