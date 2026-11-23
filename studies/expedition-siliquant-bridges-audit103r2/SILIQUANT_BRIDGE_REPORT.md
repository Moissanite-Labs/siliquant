# Track S6 (`AUDIT-103`): `Moissanite-Labs/siliquant` Force-Multiplier Audit & Exact Operator Bridges

- **Audit ID:** `AUDIT-103`
- **Governance Standard:** Aletheia v2.0.0 (`OWN_TERMS_FIRST`, Law 2b, Law 4 honest `N0`/`N1` attribution, Law 5 strict zero-metaphor exact operators, strict **Part E vs. Part I** separation)
- **Audited Repository (Strictly Read-Only Local Clone, `0` GitHub Writes / Pushes):** `Moissanite-Labs/siliquant`
  - `origin/main`: `67b29a4e18cd5d93e1f3701a34030f2e089e7312`
  - PR `#12`: `20d13d5fa9c5fb2333b8faa0360a90caa1e0a43f`
  - Draft PR `#24` (`origin/expedition/leech-l1-hadamard-symbiosis`): `d4c54e43bc9f24fd76d57a7467b0c726dd0acd1f`
  - Draft PR `#26` (`origin/feat/magnitude-autotune-quant-bridge`): `7eb484d7d5274482f4d4a73d4012277438bf3c36`
  - Draft PR `#27` (`origin/feat/round8-inkling-loom-speculative-tomography`): `8cf8d2447ab639b065c71c8edab898c66c0dfbeb`
- **Novelty Classification (Law 4 Honest Default):** **`N0`** for all classical mathematical primitives (Conway–Sloane 1982 $E_8$, Rieck 2019 neural persistence, Turrigiano 1998 homeostatic scaling, Barkai–Leibler 1997 integral feedback, al-Bīrūnī/al-Kāshī conditioning & contraction); **`N1` (engineering/audit)** strictly for the branchless pure-integer bitmask $E_8$ decoder + exhaustive 256-point/coset brute-force verification, live `siliquant` PR `#12`/Draft PR `#24`/`#26`/`#27` diagnostic, and pure-$\mathbb{Q}$ cross-web operator bridge certifier.

---

## 1. Live `Moissanite-Labs/siliquant` Diagnostic (PR `#12` & Draft PRs `#24`/`#26`/`#27`)

### 1.1 PR `#12` (`20d13d5`) vs. `@orda-aska`'s `CHANGES_REQUESTED` Review at `3333c42`
At `3333c42`, `@orda-aska` requested changes for **one item only**: `shape/design.out` was stale relative to `shape/design.py` (missing SmolLM2 scored KL rows `0.0609`, `0.0560`, `0.1183`). Commit `20d13d5` updated `shape/design.out`, `shape/ballpack.py`, `shape/test_ballpack.py`, `shape/test_ballformula.py`, `shape/npforward.py`, and `README.md`:
- `shape/` (`75` tests), `perm/` (`43` tests), and `persist/` (`19` tests) execute live inside `siliquant_bridge_audit.py --live` (`live_executed_test_suites: ["shape", "perm", "persist"]`, all passing). `lattice/test_lattices.py` (`38` tests, ~107s, requires `numpy`) was verified separately in a read-only detached checkout (`38/38 OK`), and is explicitly marked `pr12_lattice_tests_executed_in_live_check: false` in the pure-stdlib certificate so no unrun test suite is implied as executed by `--live`.
- `checks/number_trace.py` and `checks/claim_ledger.py` reproduce committed `checks/number_trace.out` and `checks/claim_ledger.out` **byte-for-byte** (`0` diff).
- `checks/plan_before_data.py` exits `0` with all `23` experiment records verified `plan before data = yes` (only the header commit line advances from `head 1e2b65d` to `head 20d13d5`).
- `shape/design.out` now carries the exact `EBG128:b560s16` (`0.0609`), `ZBG128:b560s16` (`0.0560`), and `EBG128:b504s8` (`0.1183`) SmolLM2 KL entries matching `@orda-aska`'s rerun.

### 1.2 Concrete Readiness Diagnostic for Draft PRs `#24`, `#26`, `#27` (Zero Overlap — `studies/` Only)
All three Draft PRs are `MERGEABLE` and touch strictly non-overlapping `studies/` paths (zero modifications to core `siliquant` code). Applying `@orda-aska`'s exact review bar (`bayt-al-hikma#1` + `siliquant#12`):
1. **Draft PR `#24` (`d4c54e4`, `studies/expedition-lattice-spectral-symbiosis/` + `studies/expedition-neuron-homology-unlock/`):**
   - *Gap:* Benchmarks `E8_Had` and zero-float spectral splitting on synthetic matrices only (no real SmolLM2/Gemma/Qwen2.5 GGUF KL run yet, as explicitly tracked by Issue `#25`); `expedition-lattice-spectral-symbiosis/` predates `aletheia v2.0.0` (no dedicated R5 test script, unlike `expedition-neuron-homology-unlock/test_qwen_neuron_map_r5.py`).
   - *Action to make review-ready:* Add explicit **Part E** (exact lattice/spectral proofs) vs. **Part I** (deferred Issue `#25` real-weight runs) labelling, add a standalone R5 mutation-verified test suite for `siliquant_lattice_symbiosis.py` and `siliquant_leech_shaping_r2.py`, verify `aletheia v2.0.0 audit-check --strict-ancestors --safety-scan PASS`, and rebase cleanly once PR `#12` merges.
2. **Draft PR `#26` (`7eb484d`, `studies/expedition-magnitude-autotune-bridge/`, `AUDIT-068R`):**
   - *Strong baseline:* Standalone zero-float certifier + `680`-line `test_magnitude_bridge_r5.py` + `19/19` verbatim anchors to `magnitudedev/magnitude @ d89dfff` (`(3/4)·ULP` rounding-cell theorem).
   - *Action to make review-ready:* Run `aletheia v2.0.0 audit-check --strict-ancestors --safety-scan` + `aletheia mutate --max-mutants 0` + AST conjunct sweep, add explicit `mode`/`live_repo_recounted` distinction, and clearly tag the `(3/4)·ULP` proof as Part E (real-model KL deferred).
3. **Draft PR `#27` (`8cf8d24`, `studies/expedition-inkling-loom-tomography/`, `AUDIT-074R`):**
   - *Strong baseline:* Standalone zero-float certifier + `493`-line `test_inkling_loom_r5.py` + `14/14` verified `Inkling-2-Mini-loom @ 277f099` metadata files. Directly complements `@orda-aska`'s new `weft-loom#46` (`Inkling 2 Mini`).
   - *Action to make review-ready:* Run `aletheia v2.0.0 audit-check --strict-ancestors --safety-scan` + `aletheia mutate --max-mutants 0`, add explicit `mode`/`live_recounted` flag, and cross-link `weft-loom#46` as a non-overlapping metadata tomography.

---

## 2. Pillar 1: Branchless Integer/Bitmask Conway–Sloane $E_8$ Decoder & `siliquant#16` Zero-Codeword Audit

- **Strictly Branchless Integer Architecture (`siliquant#25` Step 2 — Ready for `int64` SIMD/Pallas Lowering):**
  - `branchless_d8_decode_int` & `branchless_e8_decode_int` operate in exact dyadic integers ($x_{\text{scaled}} = 2^4 x$) using **zero data-dependent `if` branches**: arithmetic right-shift sign masks (`e >> 60`), XOR parity reduction (`(q0^...^q7) & 1`), bitwise argmax over 8 lanes, bitwise conditional parity flip, and branchless $D_8$ vs. $D_8 + \tfrac{1}{2}\mathbf{1}$ squared-distance select.
  - **Lane-Width Requirement (`int64` mandatory):** Intermediate squared distances peak at **52 bits** for inputs in $[-2^{24}, +2^{24}]$ and sign extraction uses `>> 60`; SIMD/Pallas lowering **requires `int64` lanes** (`int32` lanes would overflow).
  - **Per-Vector Brute-Force Verification (`48` Audit Vectors × `512` Exhaustive Candidates):** Tested on `48` audit vectors (interior, odd-parity $D_8$, half-integer coset, Voronoi deep holes, boundary ties, negatives, exact zero, plus 32 LCG vectors — **not** exhaustive over all $\mathbb{Z}^8$) against an independent zero-shared-code exact brute-force oracle enumerating all $2^8 = 256$ candidates per coset (`512` exhaustive candidates per vector, proved complete by parity-preserving $\pm 2$ reduction): **`48/48` (`100.0%`) exact squared-distance matches**, **`48/48` valid $E_8$ lattice points**, **`[0]*8 -> [0]*8` (`0/1` error)**.
- **Exact Quantification of `siliquant#16` (`Code.centre` Shift vs. Zero-Preserving $E_8$):**
  - On `siliquant/harness/tq.py`, subtracting `Code.centre` ($\|c\|_2 \approx 0.82$ steps at $q=16$) removes $\mathbf{0}$ from the codebook, driving `Qwen2.5-1.5B` per-row 16-level KL from `0.209` (`VE8Z`) up to `3.658` (`17.5×` worse!).
  - Exact pure-$\mathbb{Q}$ witness (`c = [5,5,5,5,4,4,4,4]/16`, $\|c\|_2 \approx 0.8004$ steps): every near-zero 8-weight block incurs exact floor error **`25/64 = 0.390625`** step$^2$, accumulating **`1575/64 = 24.609375`** step$^2$ across 63 zero blocks of an outlier-dominated `ffn_down` row, vs. **`0/1`** for zero-preserving $E_8$.

---

## 3. Pillar 2: Exact Pure-$\mathbb{Q}$ TPU v5e Roofline Oracle (`siliquant#25` Step 3 & `siliquant#23`)

> **Strict Part E vs. Part I Boundary (`measured_kernel_tok_s_present: false`):** All numbers below are **Part E exact rational hardware-bandwidth arithmetic** at TPU v5e `819,000,000,000 B/s` (weight-only batch-1 roofline; `Qwen2.5-1.5B` uses `1,543,714,304` parameters while `Gemma-4-E4B` uses the nominal rounded `4,000,000,000` placeholder count, not an exact GGUF tensor sum). Real kernel tok/s, dequant instruction latency, KV-cache traffic, and TCO are **Part I** and require a pre-registered benchmark under `experiments/` per `siliquant#23`.

| Format | Exact `bpw` | HBM IO Cut vs. `FP16` | BW Speedup vs. `Q4_K_M` | `Qwen2.5-1.5B` Ceiling (tok/s) | `Gemma-4-E4B` Ceiling (tok/s) |
|---|---:|---:|---:|---:|---:|
| `FP16` | `16/1` (`16.00`) | `0/1` (`0.000%`) | `3/10` (`0.300×`) | `265.269292` | `102.375000` |
| `Q4_K_M` | `24/5` (`4.80`) | `7/10` (`70.000%`) | `1/1` (`1.000×`) | `884.230972` | `341.250000` |
| **`ZBG128:b560s16`** (PR `#12`) | **`9/2` (`4.50`)** | **`23/32` (`71.875%`)** | **`16/15` (`1.066667×`)** | **`943.179704`** | **`364.000000`** |
| **`E8_Had`** (PR `#24`/`#25`) | **`447/100` (`4.47`)** | **`1153/1600` (`72.0625%`)** | **`160/149` (`1.073826×`)** | **`949.509769`** | **`366.442953`** |
| `INT4` (raw floor) | `4/1` (`4.00`) | `3/4` (`75.000%`) | `6/5` (`1.200×`) | `1061.077167` | `409.500000` |

---

## 4. Pillar 3: Track S4 Nature Mechanics → `siliquant` Exact Operator Bridges (Zero Metaphor)

1. **Synaptic Homeostatic Scaling (Turrigiano 1998) + Barkai–Leibler Integral Calibration:**
   - Multiplicative group gain $g = 7/3$ preserves every intra-group weight ratio with **`0/1` max error**, whereas additive shift `1/2` (same operator class as `Code.centre`!) introduces **`5/36`** pairwise ratio distortion.
   - Barkai–Leibler integral overload-rate scale control achieves **`0/1`** deadbeat steady-state error under step disturbance $\Delta u = 1/4$, strictly contrasting proportional-only clipping droop **`1/16 = 0.062500`**.
2. **Exact Kirchhoff–Schur $R_{\mathrm{eff}}$ vs. `siliquant/persist/` MST Neural Persistence:**
   - On two $3 \times 3$ bipartite layer matrices ($W_{\text{tree}}$ with 5 tree edges of weight `5`, vs. $W_{\text{redundant}}$ adding 4 off-tree parallel paths of weight `2`), exact Kruskal MST weights are **identical (`[5, 5, 5, 5, 5]`)** — `persist/` MST persistence is `100%` blind to parallel redundancy.
   - Exact Kirchhoff $R_{\mathrm{eff}}(r_0, c_2)$ strictly separates them by **`3.22×`**: `3/5 = 0.600000` ($W_{\text{tree}}$) vs. **`61/327 = 0.186544`** ($W_{\text{redundant}}$). Cutting bridge `(1, 1)` disconnects $W_{\text{tree}}$ (`SINGULAR_MATRIX`) while $W_{\text{redundant}}$ stays connected at **`73/336 = 0.217262`**.
   - *Honest complexity trade-off:* `persist/` MST is $O((a+b)^2)$ and invariant to off-tree noise; exact $R_{\mathrm{eff}}$ is $O((a+b)^3)$ and complementary specifically for parallel-path redundancy diagnostics.

---

## 5. Pillar 4: Bayt al-Ḥikma (`al-Bīrūnī` & `al-Kāshī`) + `siliconcordat#5/#15/#16` Intersection

- **al-Bīrūnī Dot-Product Cancellation Amplification:** On the matched-$\ell_1$-norm fixture ($\|w_{\text{cancel}}\|_1 = \|w_{\text{align}}\|_1 = 21$) with opposing-sign weights $w = (10, -10, 1)$, $a = (1, 1, 1)$ ($\kappa_{\mathrm{dot}} = \mathbf{21/1}$), uniform perturbation $\varepsilon = 1/4$ causes exact **`3/4 = 75.000%`** output relative error vs. **`1/28 = 3.5714%`** on aligned control $w = (10, 10, 1)$ ($\kappa_{\mathrm{dot}} = 1/1$) — an exact **`21/1`** amplification ratio matching $\kappa_{\mathrm{cancel}} / \kappa_{\mathrm{align}}$. *(Scope note: on general vectors with unequal $\ell_1$ norms, the exact ratio is $(\kappa_{\mathrm{cancel}} / \kappa_{\mathrm{align}}) \cdot (\|w_{\mathrm{align}}\|_1 / \|w_{\mathrm{cancel}}\|_1)$, e.g. `21/2` vs. $\kappa = 10$ for $w = (10, -9, 1)$.)* Directly bounds sensitivity of `ffn_down` outlier rows (`output-budget-gemma-e4b-2026-11-17` Part L) and `WEFT-T3` surgical weight-space edits (`siliconcordat#5/#16`, `||W - W0|| <= R`) under quantized scale/centroid perturbations.
- **al-Kāshī Contractive Scale Iteration:** Lipschitz $L = 1/2$ contracts initial scale error `2/1` to exact **`1/128`** in `8` steps (`(1/2)^8 * 2`), contrasted against must-fail divergent $L = 3/2$ expanding to **`6561/128`**.
- **Replay Integrity & Honest Provenance Scope (`siliconcordat#16` `WEFT-T5`):** `certify_siliquant_bridge_audit_exact` enforces deterministic zero-float re-derivation of all 4 pure-$\mathbb{Q}$ pillars and schema/provenance-tag consistency (`diag_source` vs. `mode`/`live_repo_recounted`) on its input dictionary; actual live git provenance comes strictly from running `siliquant_bridge_audit.py --live` (`compute_live_siliquant_diagnostics` via a zero-write temporary `git clone --shared --no-checkout` + isolated `-I -B` subprocesses).

---

## 6. Reproduction Commands

```bash
# 1. Offline standalone verification (recomputes all 4 exact pure-Q pillars + replays pinned repo diag; --json explicit alias supported)
python3 -I -B -W error symbiosis/stream/track-s6-siliquant/siliquant_bridge_audit.py --standalone --json

# 2. Strictly 100% read-only live audit of local Moissanite-Labs/siliquant clone (temp shared clone, ZERO writes to repo/.git, DIRTY=0)
python3 -I -B -W error symbiosis/stream/track-s6-siliquant/siliquant_bridge_audit.py --live --repo-root publish/siliquant --output symbiosis/stream/track-s6-siliquant/siliquant_bridge_certificate.json

# 3. Assertion-free R5, branchless E8 brute-force, & temporary-git-repo live-path tests
python3 -I -O -B -W error symbiosis/stream/track-s6-siliquant/test_siliquant_bridge_r5.py
```
