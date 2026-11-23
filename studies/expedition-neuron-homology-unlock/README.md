# Expedition Neuron Homology & Layer-10 Exploratory Bridge Study (`AUDIT-059`)

Self-contained, zero-edit research study packaged alongside `Moissanite-Labs/siliquant` (reconciled in-branch with the two-pass outside check in [`Moissanite-Labs/weft-loom#44`](https://github.com/Moissanite-Labs/weft-loom/pull/44) @ `e660cd3d` and cross-linked with [`Moissanite-Labs/siliquant#31`](https://github.com/Moissanite-Labs/siliquant/pull/31) `AUDIT-106R` on Falcon-H1-Tiny exact $\mathbb{Z}_2$ conservation; pairs with [`studies/expedition-lattice-spectral-symbiosis/README.md`](../expedition-lattice-spectral-symbiosis/README.md)).

Every script below runs standalone using **only Python 3's standard library** (zero NumPy, zero external clones required in default standalone mode, zero floats in JSON, zero bare `assert` statements), passes `4/4` isolated `-I` deterministic replay (`-I -B`, `-I -O -B`, `-I -B -W error`, `-I -O -B -W error`), and achieves **`24/24` (`100.0%`)** AST mutation kill rate (`aletheia v2.0.0 mutate`) on its exact statistical & lineage kernel.

---

## Manifest

| File | Audit | Role & Exact Integrity |
|---|---|---|
| [`neuron_homology_exact.py`](./neuron_homology_exact.py) | `AUDIT-059` | Pure-stdlib exact $\mathbb{Q}/\mathbb{Z}$ statistical, $\mathbb{Z}_2$ lineage-equalizer, seeded permutation-null cross-check (with exact feasible `[k_min, k_max]` domain enforcement), and `-O`-immune recounter kernel (`SHA-256`: `74b77624...`, **`24/24` `100.0%` R5 killed**, `9/9` `100.0%` AST `BoolOp` killed) |
| [`qwen_neuron_map_bridge.py`](./qwen_neuron_map_bridge.py) | `AUDIT-059` | Standalone certified replay + live external-extraction bridge (`SHA-256`: `e2398ee6...`; standalone `-I -O -B -W error --json` stdout SHA-256 `d61f2dc4536ff13937caad0c116477479f56bb318ad3a83a28f4bda193efc372`; `standalone_canonical_sha256: 61aeb010...`; live `--live --repos-dir <dir>` reproduces `qwen_neuron_map_certificate.json` byte-for-byte) |
| [`test_qwen_neuron_map_r5.py`](./test_qwen_neuron_map_r5.py) | `AUDIT-059` | R5 refusal-site mutation, feasible-boundary, and independent-checker semantic-tamper test suite (`SHA-256`: `c88f0611...`; `7/7` groups PASS; `-I -O -B -W error` stdout SHA-256 `7ed5ee3594c196eb122aa3cf3090368ab37cbf29cad522d468847d3ce409becd`) |
| [`check_048_audit059_independent.py`](./check_048_audit059_independent.py) | `AUDIT-059` | Published, parameterized, zero-shared-code independent verifier closing `weft-loom#44` `N8` (`SHA-256`: `e3a87063...`; pins `canonical_sha256`, upstream HEAD SHAs, and all `15` page digests while deriving all `part1`–`part5` checks directly from the supplied certificate; standalone `-I -O -B -W error` stdout SHA-256 `d2295af083202a39d2ac4e3d33420846988a8bf77044c5154a162b518bf9561b`; also supports `--live --repos-dir <dir>`) |
| [`qwen_neuron_map_certificate.json`](./qwen_neuron_map_certificate.json) | `AUDIT-059` | Unchanged pinned zero-float canonical JSON certificate from live extraction over 15 static `qwen-neuron-map.pages.dev` pages (`1,504,256` MLP neurons across 13 full models) and both read-only `alphaxiv` clones (`SHA-256`: `7ff59841...`, `canonical_sha256`: `4abe60205f6b6a43762a19711d4a72936ca34e67cf751410a825ef1e7f478289`) |
| [`QWEN_NEURON_MAP_UNLOCK_REPORT.md`](./QWEN_NEURON_MAP_UNLOCK_REPORT.md) | `AUDIT-059` | Full empirical tomography, complete `weft-loom#44` `N1`–`N11` reconciliation ledger, Law-4 `N0`/`N1` quarantine (`R5-N0-10..15`), and tightened model-research unlock playbook |

---

## Quick Reproduction

Run from this directory (`studies/expedition-neuron-homology-unlock/`), with **zero external clones or third-party packages required**:

```bash
# 1. Standalone certified replay (zero external clones, zero floats, <0.1s):
python3 -I -O -B -W error qwen_neuron_map_bridge.py --json | sha256sum
# Expected: d61f2dc4536ff13937caad0c116477479f56bb318ad3a83a28f4bda193efc372

# 2. R5 refusal-site mutation & exact-invariant test suite (7/7 groups PASS):
python3 -I -O -B -W error test_qwen_neuron_map_r5.py | sha256sum
# Expected: 7ed5ee3594c196eb122aa3cf3090368ab37cbf29cad522d468847d3ce409becd

# 3. Published zero-shared-code independent verifier (closes weft-loom#44 N8):
python3 -I -O -B -W error check_048_audit059_independent.py | sha256sum
# Expected: d2295af083202a39d2ac4e3d33420846988a8bf77044c5154a162b518bf9561b
```

Passing `--live --repos-dir <path>` against local read-only snapshots of `qwen-neuron-map.pages.dev`, `alphaxiv/weight-edit-minimization@768d7340`, and `alphaxiv/flint-goalkeeper-neuron@6ece43a9`:
- In `qwen_neuron_map_bridge.py --live --repos-dir <path>`, re-parses all `15` static HTML atlas pages, real float32 `.npy`/`.npz` weight matrices, and raw JSON continuation files directly, reproducing [`qwen_neuron_map_certificate.json`](./qwen_neuron_map_certificate.json) with 0-byte diff.
- In `check_048_audit059_independent.py --live --repos-dir <path> [--siliquant-readme <path>]`, additionally verifies upstream repo HEAD SHAs and clean working trees, AST bare-assert positions (`FG-01`, `WE-01`), SHA-256 digests of all `15` static HTML atlas pages, direct parsing of `q08.html` (`[0, 1, 23]` boundary layers) and both `base_l15`/`base_l19` float32 `.npy` matrices (sub-$2^{-16}$ `bf16` entry counts), raw Flint Goalkeeper JSON row counts, and the optional `siliquant` README quote.

---

## Key Verified Results & Narrowed Scope (`weft-loom#44` Reconciliation)

### 1. Post-Hoc Exploratory Target for Gemma 4 E4B Layer-10 `ffn_down` Clipping (`N6`)
In [`experiments/output-budget-gemma-e4b-2026-11-17/README.md`](../../experiments/output-budget-gemma-e4b-2026-11-17/README.md), clipping `ffn_down` on **Layer 10 alone** accounts for `60/67 = 89.55%` (`0.1560 / (0.1560 + 0.0182)`) of all `ffn_down` clipping KL across 42 layers. On `qwen-neuron-map/g4e4.html`, Layer 10 ranks `#1 / 42` in `mean_out` (`0.235117` vs `#2` `L09` at `0.232089`), is the first layer where `code` share overtakes `prose`, and has `4 / 6` top groups as dark hubs. Per `weft-loom#44` (`N6`), Layer 10 leads Layer 9 by only `1.3048%` (`3101 / 237659`) against a `78.39×` KL jump and full-model Spearman correlation is weak ($\rho = +0.1607$, permutation $p \approx 0.15$), so we treat this strictly as a **post-hoc exploratory target** for a blind activation test—not a causal explanation.

### 2. `Qwen3.5-0.8B` Unique Column-Index Anisotropy & Stratified Exp 8 Join (`N2`, `N3`, `N4`, `N7`, `N9`)
- **All-layer anisotropy (`N7`):** Across `13` full neuron-level models (`1,504,256` neurons), only `Qwen3.5-0.8B` shows strong column stratification (`1849/1191 ≈ 1.552×` vs. max `1.074×` across the other 12 models): **`21 / 24`** layers (`L02..L22`) have `hi > lo` dark fraction while boundary layers `L00`, `L01`, and `L23` reverse (honest all-layer exact two-sided Binomial sign test $p = 2325 / 8{,}388{,}608 = 2.77 \times 10^{-4}$).
- **Stratified Exp 8 join (`N2`, `N3`, `N4`, `N9`):** Joining real `L15` & `L19` `down_proj` tensors (under fixed-point $2^{-16}$ quantization, noting `13.76%` L15 / `12.18%` L19 sub-$2^{-16}$ `bf16` entries) shows `29 / 40` Top-20 $L_\infty$ outliers in `cols [0, 512)` ($p < 10^{-14}$), shared cross-layer dim-`0` residual axis (`cos^2 = 0.7208`), and `Adjacent brackets` enrichment with exact stratified combined $p = 3.08 \times 10^{-10}$ (L15, both blocks $< 0.01$ individually) and $p = 1.42 \times 10^{-3}$ (L19, individual blocks $1.30 \times 10^{-2}$ and $2.81 \times 10^{-2} > 0.01$). Exp 8's selection uses activation deltas on 8 prompts plus weight factor $|u^\top W_0|$, which accounts for the unconditional kurtosis jump once matched (`p ≈ 0.12 / 0.16`). The 1,000-draw seeded permutation null is retained strictly as a code cross-check of the exact hypergeometric tail.

### 3. Single-Run Pythia Dip (`N5`), $\mathbb{Z}_2$ Consistency Identity (`N1`), & `-O`-Immune Recounter (`FG-01`)
- **Pythia (`N5`):** `py0a` (`1k`), `py0b` (`8k`), and `py0c` (`32k`) are checkpoints of **one** run (`py0`: `16.30% -> 4.36% -> 3.77% -> 5.68%`), while `py1` (`5.36%`) and `py2` (`5.69%`) provide `143k` endpoints only.
- **$\mathbb{Z}_2$ equalizer (`N1`):** Setting $(p_1, p_2) = (1, 17/20)$ gives $A_{\mathrm{chance}} = 17/20 = 85\%$ as an arithmetic consistency check motivating margin-centered $\Delta_{\mathrm{lin}}/\kappa$ (complemented by `weft-loom#44`'s exact `Binomial(m, 1/2)` / cluster-law foundation and [`siliquant#31`](https://github.com/Moissanite-Labs/siliquant/pull/31) `AUDIT-106R`).
- **`-O`-immune hygiene (`FG-01`):** `strict_recount_goalkeeper_rows` closes bare-`assert` gaps with `24/24` (`100.0%`) R5 mutation kill rate.

---

## Honest Scope Limits & Prior-Art Quarantine

- **Honest Novelty (`N1`):** All upstream atlases, models, editing methods, and classical statistical tests are strictly quarantined as `N0` (`R5-N0-10..15`). See [`QWEN_NEURON_MAP_UNLOCK_REPORT.md`](./QWEN_NEURON_MAP_UNLOCK_REPORT.md) §4.
- **Hypothesis vs. Live GGUF Measurement:** Scoring column-stratum routing or blind Layer-10 activation checks directly against `llama-perplexity --kl-divergence` on multi-GB GGUF weights remains a live hardware experiment outside this pure-stdlib study.
