# Expedition Neuron Homology & Layer-10 Circuit Fan-Out Unlock Study (`AUDIT-059`)

Self-contained, zero-edit research study packaged alongside `Moissanite-Labs/siliquant` (targeting the open **Gemma 4 E4B Layer-10 `ffn_down` clipping mystery** in [`experiments/output-budget-gemma-e4b-2026-11-17/README.md`](../../experiments/output-budget-gemma-e4b-2026-11-17/README.md) and **column-stratified vs. flat-width `down_proj` quantization routing** alongside [`studies/expedition-lattice-spectral-symbiosis/README.md`](../expedition-lattice-spectral-symbiosis/README.md)).

Every script below runs standalone using **only Python 3's standard library** (zero NumPy, zero external clones required in default standalone certified mode, zero floats in JSON, zero bare `assert` statements), passes `4/4` isolated `-I` deterministic replay (`-I -B`, `-I -O -B`, `-I -B -W error`, `-I -O -B -W error`), and achieves **`24/24` (`100.0%`)** AST mutation kill rate (`aletheia v1.3.0 mutate`) on its exact statistical & lineage kernel.

---

## Manifest

| File | Audit | Role & Exact Integrity |
|---|---|---|
| [`neuron_homology_exact.py`](./neuron_homology_exact.py) | `AUDIT-059` | Pure-stdlib exact $\mathbb{Q}/\mathbb{Z}$ statistical, $\mathbb{Z}_2$ lineage-equalizer, seeded permutation-null, and `-O`-immune recounter kernel (`SHA-256`: `67861e44...`, **`24/24` `100.0%` R5 killed**) |
| [`qwen_neuron_map_bridge.py`](./qwen_neuron_map_bridge.py) | `AUDIT-059` | Standalone certified replay + live external-extraction bridge (`SHA-256`: `46a00c42...`; standalone `-I -O -B -W error --json` stdout SHA-256 `4365ad8b952fbe8e6481ad9e6cd86be1ed9e9f6ba7fed974db1ebbdb9327aa10`; live `--live --repos-dir <dir>` reproduces `qwen_neuron_map_certificate.json` byte-for-byte) |
| [`test_qwen_neuron_map_r5.py`](./test_qwen_neuron_map_r5.py) | `AUDIT-059` | R5 refusal-site mutation & exact-invariant test suite (`SHA-256`: `31fcb274...`; `6/6` groups PASS; `-I -O -B -W error` stdout SHA-256 `953f5ed0b8ea96385f2d32bbcda57ac7b3b2d190d3fe95e2389b5f1d864f2160`) |
| [`qwen_neuron_map_certificate.json`](./qwen_neuron_map_certificate.json) | `AUDIT-059` | Pinned zero-float canonical JSON certificate from live extraction over all 15 static `qwen-neuron-map.pages.dev` pages (`1,504,256` MLP neurons across 13 full models) and both read-only `alphaxiv` clones (`SHA-256`: `7ff59841...`, `canonical_sha256`: `4abe60205f6b6a43762a19711d4a72936ca34e67cf751410a825ef1e7f478289`) |
| [`QWEN_NEURON_MAP_UNLOCK_REPORT.md`](./QWEN_NEURON_MAP_UNLOCK_REPORT.md) | `AUDIT-059` | Full empirical tomography, Law-4 `N0`/`N1` quarantine (`R5-N0-10..15`), and actionable model-research unlock playbook |

---

## Quick Reproduction

Run from this directory (`studies/expedition-neuron-homology-unlock/`), with **zero external clones or third-party packages required**:

```bash
# 1. Standalone certified replay (zero external clones, zero floats, <0.1s):
#    verifies committed certificate canonical_sha256 + zero-float invariant and re-derives
#    every exact hypergeometric/binomial/McNemar/Fisher tail, Z_2 confounder/equalizer,
#    1,000-draw seeded permutation null, Gemma 4 E4B Layer-10 ranking, and prompt-leak control.
python3 -I -O -B -W error qwen_neuron_map_bridge.py --json | sha256sum
# Expected: 4365ad8b952fbe8e6481ad9e6cd86be1ed9e9f6ba7fed974db1ebbdb9327aa10

# 2. R5 refusal-site mutation & exact-invariant test suite (6/6 groups PASS):
python3 -I -O -B -W error test_qwen_neuron_map_r5.py | sha256sum
# Expected: 953f5ed0b8ea96385f2d32bbcda57ac7b3b2d190d3fe95e2389b5f1d864f2160
```

Passing `--live --repos-dir <path>` against local read-only snapshots of `qwen-neuron-map.pages.dev`, `alphaxiv/weight-edit-minimization@768d7340`, and `alphaxiv/flint-goalkeeper-neuron@6ece43a9` re-parses every static HTML atlas, real float32 `.npy`/`.npz` weight matrix, and raw JSON continuation file directly, reproducing [`qwen_neuron_map_certificate.json`](./qwen_neuron_map_certificate.json) with 0-byte diff.

---

## Key Verified Results Bearing Directly on `siliquant`

### 1. Direct Mechanistic Hypothesis for the Gemma 4 E4B Layer-10 `ffn_down` Clipping Mystery
In [`experiments/output-budget-gemma-e4b-2026-11-17/README.md`](../../experiments/output-budget-gemma-e4b-2026-11-17/README.md) (Parts H, L, and M), clipping `ffn_down` on **Layer 10 alone** accounts for **`0.155993` of `0.174156`** (`89.6%`, `60/67` at 4-decimal summary precision `0.1560 / (0.1560 + 0.0182)`) of all `ffn_down` clipping KL across 42 layers, yet Layer 10 ranks only 4th/19th in weight-magnitude outlier statistics and its top 10 changed hidden units carry `0.05%` of the damage.

Auditing all `42` layers (`430,080` MLP neurons) of `qwen-neuron-map/g4e4.html` uncovers three simultaneous circuit-level signatures converging on **Layer 10**:
- **`#1 / 42` in mean outgoing circuit fan-out:** `mean_out = 0.23511` (`#2` is `Layer 09` at `0.23208`; `Layer 41` at `0.23039`; `Layer 06` at `0.22770`; `Layer 22` at `0.22375`).
- **`#1` FIRST layer in the model where `code` activation share overtakes `prose` activation share** — directly matching `siliquant`'s `llama.cpp` C++ evaluation text!
- **`#3 / 42` in active shared-family diversity (`191 / 200` families)**, with **`4 / 6` of its largest neuron groups being high-fan-out Gemma-4-specific dark broadcast hubs (`⧅`)** (`gid=36` `out=0.605`, `gid=3` `out=0.699`).

**Falsifiable `siliquant` Pre-Test:** Compare `ffn_down` clipping/quantization sensitivity on `Layer 09` (`#2` `mean_out = 0.23208`) versus `Layer 12`/`Layer 13` (lower `mean_out = 0.1828/0.1823` despite higher dark fraction `22.69%`), and check Spearman rank correlation of per-layer `mean_out` against `ffn_down` clipping KL across Gemma 4 E4B / Qwen3.5 / SmolLM2.

### 2. `Qwen3.5-0.8B` Unique Column-Index Anisotropy vs. Flat-Width `SmolLM2`
Across all `13` full neuron-level models parsed (`1,504,256` MLP neurons), **ONLY `Qwen3.5-0.8B`** exhibits strong column-index stratification (`hi_to_lo_dark_ratio = 1849/1191 ≈ 1.552×` vs. max `106872/99509 ≈ 1.074×` across all other 12 models):
- **`21 / 21` consecutive middle layers (`L02..L22`)** carry strictly higher dark-neuron shadow (`⧅`) in `cols [2048, 3584)` than in `cols [0, 512)` (peak `106/33 = 3.21×` at `L16`, `209/69 = 3.03×` at `L21`, exact two-sided sign test $p = 1/1{,}048{,}576$).
- Joining real `Qwen3.5-0.8B` `mlp.down_proj` float32 tensors (`L15` & `L19`, `alphaxiv/weight-edit-minimization@768d7340`) shows **`29 / 40` (`72.5%`) of Top-20 $L_\infty$ outlier columns** sit inside `cols [0, 512)` ($1/7$ of width; exact hypergeometric $p < 10^{-14}$), while causal anti-repetition math-rescue neurons (`64+64`; `579` rescues / `5` losses, exact McNemar $p < 10^{-150}$) jump `2.6×–3.5×` in kurtosis (`10.44` vs. `4.03`, `12.18` vs. `3.45`), enrich **`11.63×` (L15)** and **`20.36×` (L19)** on **`Adjacent brackets: }( )( )*(`** (surviving Law-3 stratification inside BOTH `col < 512` and `col >= 512`; `0/1,000` seeded permutation null exceedances), and align across layers (`cos^2 = 0.7208`) on residual outlier dim `0`.
- By contrast, `SmolLM2-{135M, 360M, 1.7B}` is flat across column index with monotonic dark decay (`7.29% -> 5.26% -> 0.90%`).

**Falsifiable `siliquant` Routing Test:** On importance-ordered / width-pruned models (`Qwen3.5-0.8B`), split `down_proj` precision / `E8_Had` protection by column stratum (`cols [0, 512)` vs. `[2048, 3584)` bulk); on flat-width models (`SmolLM2`), route by kurtosis / family mask rather than column index.

### 3. Exact $\mathbb{Z}_2$ Sign-Symmetry Equalizer & `-O`-Immune Verification Hygiene
- Proves in pure $\mathbb{Q}$ (`neuron_homology_exact.py:exact_sign_symmetry_lineage`) why skewed independent activations (`p_1 = 1, p_2 = 17/20`) inflate raw cross-model sign agreement to `17/20 = 85%` at margin-centered excess $\Delta_{\mathrm{lin}} = 0$ and Cohen's $\kappa = 0$, cleanly separating unrelated models from true lineages (`Qwen3.5-0.8B/2B` $\kappa = 43/50$; `4B/9B` $\kappa = 24/25$).
- Ships a `-O`-immune exact Fisher recounter (`strict_recount_goalkeeper_rows`) closing bare-`assert` hygiene gaps (`FG-01`, `WE-01`) with `24/24` (`100.0%`) R5 mutation kill coverage.

---

## Honest Scope Limits & Prior-Art Quarantine

- **Honest Novelty (`N1`):** All upstream atlases, models, editing methods, and classical statistical tests (`qwen-neuron-map.pages.dev`, `alphaxiv/{weight-edit-minimization,flint-goalkeeper-neuron}`, Platonic Representation Hypothesis / CKA, ROME/MEMIT, outlier quantization, Fisher/McNemar/Cohen-$\kappa$) are strictly quarantined as `N0` (`R5-N0-10..15`). See [`QWEN_NEURON_MAP_UNLOCK_REPORT.md`](./QWEN_NEURON_MAP_UNLOCK_REPORT.md) §3.
- **Hypothesis vs. Live GGUF Measurement:** The Gemma 4 E4B Layer-10 circuit fan-out finding and `Qwen3.5-0.8B` column-stratum routing rule are exact cross-dataset structural discoveries yielding preregisterable predictions; scoring those arms directly against `llama-perplexity --kl-divergence` on the multi-GB GGUF weights remains the natural next live hardware experiment in `siliquant`.
