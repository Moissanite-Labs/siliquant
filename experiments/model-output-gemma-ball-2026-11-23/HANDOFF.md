# Gemma Integer-Ball Experiment Handoff Guide

This document is a self-contained operational and scientific handoff for continuing the Gemma integer-ball quantization sweep (`experiments/model-output-gemma-ball-2026-11-23/`) and subsequent controlled experiments in `Moissanite-Labs/siliquant`.

All commit SHAs, file digests, row counts, and metrics below were verified directly against repository branches and raw `results.jsonl` / `results.out` receipts.

---

## 1. Branch, Commit & PR Map

### Key Branches & Commits

| Ref / Branch | Full SHA | Role & Status |
|---|---|---|
| `main` | `716d92e8c250f7eafddfb2e2342483918ef15a78` | Default branch; holds `PREREG.md` (#47), Track S30 tomography (#63), and homepage/site. Does **not** yet hold `run.py` (#69) or raw Gemma `results.jsonl` receipts. |
| `exp/gemma-ball-runner` (HEAD, PR #69) | `ccf045d04ada2170eeafaa37254a764b96552723` | **Canonical runner branch for all new runs.** Preserves exact source GGUF dtypes for `per_layer_token_embd.weight` and all non-block tensors (fixes Gemma 4 E4B stock-arm tensor mismatch). |
| `exp/gemma-ball-runner~1` | `a5f27157f4d7315e2b5108e4e6f24a30d0ecfab3` | Unweighted fallback when an imatrix tensor key is missing/zero-count (fixes Gemma 4 E4B `blk.24.attn_k.weight.in_sum2` crash) + provenance fields `imatrix_fallback_count` / `imatrix_fallback_tensors`. |
| `exp/gemma-ball-runner~2` | `6a9d871eecc5c95f17edf751c7257d7fcd69cf62` | Fixes Issue #58 `RUN-BUG-01` (`hEBG128` / `hHBG128` / `i*BG128` real-imatrix lookup via `arm_needs_real_imat`) and `RUN-BUG-02` (`token_embd.weight` BF16/F16/F32 source dtype preservation). |
| `exp/gemma-ball-runner~3` (legacy) | `ca10cdacd9b1251b989d5fb66578bfe7e8f2554e` | Initial resumable Apple Silicon runner. **All completed and partial rounds currently on `exp/gemma-ball-results` ran on this commit.** Do not use for new runs. |
| `exp/gemma-ball-results` (HEAD) | `4f1d31a89501f857afa2e66434afe431dedeb99c` | Partial `gemma-4-E4B-it` BF16 raw receipts (8/34 rows, stopped on missing imatrix key under `ca10cda`). |
| `exp/gemma-ball-results~1` | `ba5ec5a64c3252ef4c0caa1c23bc33b0594ab506` | Complete `gemma-3-12b-it` BF16 raw receipts (34/34 rows + pilot JSON, runner `ca10cda`). |
| `exp/gemma-ball-results~2` | `86bb1b2a2178bd8b68912c1e7496426f53e9ffa8` | Complete `gemma-4-E4B-it` `Q8_0` bridge raw receipts (2/2 rows, runner `ca10cda`). |
| `exp/gemma-ball-results~3` | `dc10d4bbca8cc049f288aa22e42172aebb493363` | Complete `gemma-3-4b-it` BF16 raw receipts (34/34 rows, runner `ca10cda`). |
| `exp/gemma-ball-results~4` | `90a3247dc9bb70ca3afa0f13627ec4b569c73317` | Complete `gemma-3-270m-it` (34/34) and `gemma-3-1b-it` (34/34) F16 raw receipts (runner `ca10cda`). |
| `devin/1795477739-results-gemma-12b` (PR #75) | `4728f1442914903e2137a941ae5444b355f597c3` | Adds Gemma 3 12B round to `docs/index.html`, `site/results.html`, and `docs/siliquant-results.pdf`. |
| `exp/gemma-experiment-handoff` | *(this branch, off `main`)* | Adds this self-contained `HANDOFF.md` independently of PR #75. |

### Related PRs & Issues

| Item | State | Role |
|---|---|---|
| **PR #47** | Merged (`ea3c81e` -> `9be83f9`) | Plan-before-data preregistration (`experiments/model-output-gemma-ball-2026-11-23/PREREG.md`), defining models, texts, arms, `P1`–`P11` falsifiers, and pilot/zero-codeword gates. |
| **PR #63** | Merged (`3064e48`) | Track S30 pure-$\mathbb{Z}/\mathbb{Q}$ tomography suite (`studies/expedition-gemma-m1-tomography-audit130r2/`) auditing 102 rows across 270M/1B/4B, certifying log-convex rate-distortion law, `Q4_K_M` fallback bands, `RUN-BUG-01..02` patch specs, and pre-registered `PRED-M1-01`. |
| **PR #65** | Open (`c566bfb`) | Track S32 R2 exact-$\mathbb{Z}/\mathbb{Q}$ hardware/lattice bounds (`studies/expedition-tpu-bounds-residual-lut-audit132r2/`): `Block-H2` native `INT8` MXU safety (`max = 78 <= 127` vs `3-stage WHT_8` `156 > 127`), `O(1)` branchless 2-stage residual flat-LUT `E_8` (`32 KiB` L1d / `512 KiB` VMEM), static Pallas WCET, and refusal-margin preservation witness. |
| **PR #66** | Open (`6ea7859`) | Fixes `site/build_results.py` to write `docs/archive/results-2026-11-23.html` + `docs/siliquant-results.pdf` without overwriting `docs/index.html`; updates merged-PR badges. |
| **PR #69** | Open (`ccf045d`) | Merges `exp/gemma-ball-runner` (`run.py` + `test_run.py`) into `main`. |
| **PR #75** | Open (`4728f14`) | Adds Gemma 3 12B BF16 results to homepage/report (`docs/index.html`, `site/results.html`, `docs/siliquant-results.pdf`). Note: open PR #73 (`amp-sumi/homepage-data-fix`) also edits `docs/index.html` and should be ordered/rebased cleanly. |
| **Issue #58** | Open | Live 270M/1B/4B diagnostic brief: `RUN-BUG-01..02` fixes, log-convex iso-KL law (`PRED-M1-03`), `P1`–`P10` readout, and `PRED-M1-01..02`. |
| **Issue #59** | Open | Weight-space misfolding integrity brief: `token_embd.weight` per-row hash locks, `O(N)` Deep Patch Inspection (`DPI`) Row-Energy Concentration Ratio, and zero-cost `ballpack` Broad Weight Inspection (`BWI`). |

---

## 2. Completed Rounds & Verified Receipts

All runs below live on `exp/gemma-ball-results` and executed under runner commit `ca10cdacd9b1251b989d5fb66578bfe7e8f2554e`.

### Evaluation Text Digests (Shared Across Rounds)

| Text Key | Source | Scored Positions | SHA-256 |
|---|---|---:|---|
| `wikitext` | `wikitext-2` test (`wiki.test.raw`, 40 chunks x 512 ctx) | 10,200 | `173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08` |
| `python` | `llama.cpp @ eada89c` `conversion/base.py` (40 chunks) | 10,200 | `e38975e1c68d98ac1664dfd530616eb35c72294382a4dd873d4746b23f27779f` |
| `bridge` (`P7` only) | `eval.txt` (`harness/build_eval_text.py`, `llama.cpp @ 427291b`) | 10,200 | `df5748784801c869b7d1ce54b169af665bfa4c0d3c9839c88d097ffbd33df761` |

### Per-Model SHA-256 Receipt Table

| Model | Status (Rows) | Results Commit | Runner | `model_sha256` | `results.jsonl` SHA-256 | `results.out` SHA-256 | Real / Rand `imatrix_sha256` |
|---|---|---|---|---|---|---|---|
| `gemma-3-270m-it` (F16, 126 mat, 100.27M block wts) | Complete (`34/34`) | `90a3247` | `ca10cda` | `140cb395a3a3dcdb3de66e44c60d7265286c9d49f951c1899ec0d8c1f16b2feb` | `dafe50f93262be5628b97a83b915a0f65aa4eaa52f09b1b9bfd36381c4b34a83` | `4c31bb3b85082db734887b4899bc3e5781c4ec835a20f89ba15cc919c22520b5` | Real: `f68b5c56a61919a9cc5985bc650df9bc32abe9bc2f4871abd79d1fce7c9481f4`<br>Rand: `7680bac70ff15a32af1276541a7ef78c25aea7e6c3efc18d331654f70e1bba36` |
| `gemma-3-1b-it` (F16, 182 mat, 697.76M block wts) | Complete (`34/34`) | `90a3247` | `ca10cda` | `05bd381a5f45611ce53f4fdcc6641cf6cec68c3091d74e8a32ea591f062d3fc5` | `040717b2d9c24d47e64fcc3b41156a375d651f180cd7b9f558ec603c60f7733f` | `aeea9b43446c4f62bb924aaf706f555d47ba498855cc1372546c64de1468429b` | Real: `23e5f2b7328df8d6c3b1ea0c5c250de6fa4d5e2b2a44018bba960720c3e7671e`<br>Rand: `edebd3550cbe494627027d824a516c768944e1f2102606bb03e08005d0094257` |
| `gemma-3-4b-it` (BF16, 238 mat, 3.21B block wts) | Complete (`34/34`, stock `token_embd` mismatch) | `dc10d4b` | `ca10cda` | `7bcfca74279608552011d3c4d11e9298934502ffa410eb21b9047a6276b87ba6` | `5895d850aff01a154a3b57e1c47468c90121e34316fa6e850ae866f6ec4dc23e` | `6ef67975a707ce8a5d93214885fa5b04ccba5bf929a98e32c5f0e0982ff48594` | Real: `19b11624568c4352ef0a43c2aca3599ac4a0af166f7a8102c4db301db8f35b7a`<br>Rand: `25e2cd64848038201ca91d5e272f9da395e41389677a5dcfae151d4f85d044b3` |
| `gemma-4-E4B-it` `Q8_0` bridge (294 total / 258 active mat, 3.95B / 3.89B wts) | Complete (`2/2`, distinct `Q8_0` stratum) | `86bb1b2` | `ca10cda` | `f8854aa4480df62585a279e7ca0a881554fc18a41c59c4f62642d16a2ae47012` | `98f2e3e10de7e732b12476a32a47a6e5c81df16b4b0fddaba082b72b8b857e78` | `f86e87d6990c298ca6cd9c59cb16d0e62bcfdb772f439d138153f8712dc5a8fa` | N/A (uncalibrated Voronoi bridge) |
| `gemma-3-12b-it` (BF16, 336 mat, 10.76B block wts) | Complete (`34/34`, stock `token_embd` mismatch) | `ba5ec5a` | `ca10cda` | `276920b11d80002d45b9b85291e333d9a7798df92b148ba9ae3d0625dd9f82f2` | `fd9f780e714c79e90eeb442aae945c9c6f3046ee10016108b4e543f016b8be03` | `88c0aa689fbe020a7e8c96b00688784b02653287e594a7fdd84a84d56d119d3a` | Real: `9aaa78bb7785aaa02971b681983eb0918f92ba76842f4ac9bee1bea5961ebef8`<br>Rand: `9631381a8238b144664fe275d4b8c3ff9052044f2058559f77756b3cbd3a8aca`<br>Pilot JSON: `280ed9f9c57d5f6c4949a9d76d718e4057cd16af9ff6aef5a21b5e8aa2e86a23` |
| `gemma-4-E4B-it` (BF16, 294 total / 258 active mat, 3.95B / 3.89B wts) | **Partial (`8/34`)**, stopped on missing `blk.24.attn_k.weight.in_sum2` | `4f1d31a` | `ca10cda` | `38e0dba6818d18e2b41062c7b3b9083dcfddc44767078c415ff7e13661911ba5` | `5d364c29005b615eacbe1a6ea27203c2b9773ae45f8de1c4ce6870694bb1205a` | `823b4922fe33eb3b105db3ff958610d0eec4019ebb316fba8ace7e3998840c75` | Real: `b67f51e20abb6bfe4243104514c4164bfc1ba58d147d66df1595140676348505`<br>Pilot JSON: `72b003a85dea5efbaa0f25626b38f41e0af254af9f1e02b3ef45d5e5a941cf30` |

### Verified Headline Results (Primary Arms & Iso-KL Block Savings)

#### `gemma-3-270m-it` (F16 ref: Wiki PPL `50.5941 ± 1.9897`, Python PPL `8.3839 ± 0.7314`; `other_tensors_that_differ == []`)

| Variant | Block bpw | WikiText KL ± SE | Wiki PPL | Wiki Top-1 | Python KL ± SE | Py PPL | Py Top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `file:Q4_0` | 4.5000 | 0.501528 ± 0.007732 | 69.867 | 70.137% | 0.565080 ± 0.029793 | 11.942 | 77.647% |
| `file:Q4_K_M` (fallback band) | 5.4338 | 0.153572 ± 0.002861 | 54.219 | 83.049% | 0.166596 ± 0.010586 | 9.031 | 87.899% |
| `file:Q4_K_M-imat` (fallback band) | 5.4338 | 0.091622 ± 0.001843 | 53.045 | 86.755% | 0.088345 ± 0.005311 | 8.558 | 91.765% |
| `ZBG128:b560s16` | 4.4998 | 0.194755 ± 0.003254 | 55.640 | 80.088% | 0.209988 ± 0.013437 | 8.666 | 85.658% |
| `hZBG128:b560s16` | 4.5178 | 0.076017 ± 0.001915 | 55.469 | 87.922% | 0.100806 ± 0.007263 | 8.744 | 90.868% |
| `hZBG128:b560s16~rand` | 4.5178 | 0.086958 ± 0.001762 | 57.198 | 87.480% | 0.104525 ± 0.007112 | 9.503 | 90.532% |
| `EBG128:b560s16` | 4.4999 | 0.178904 ± 0.003259 | 65.187 | 81.608% | 0.186328 ± 0.011397 | 9.809 | 88.347% |
| `HBG128:b560s16` | 4.4999 | 0.174683 ± 0.003145 | 55.967 | 81.588% | 0.167353 ± 0.009242 | 9.126 | 88.011% |
| `hZBG128:b496s16` / `b624s16` | 4.0179 / 5.0180 | 0.150933 / 0.036582 | 54.995 / 54.626 | 83.029% / 91.588% | 0.182747 / 0.055786 | 9.677 / 8.236 | 88.683% / 92.885% |

- **Iso-KL block savings (`hZBG128` vs `Q4_K_M-imat` at `5.4338` fallback bpw):** WikiText `4.3817` bpw (**19.36%** log-linear) / `4.4137` bpw (**18.77%** linear); Python `4.6294` bpw (**14.80%** log-linear) / `4.6562` bpw (**14.31%** linear). `hZBG128:b560s16` vs `Q4_K_M-imat`: Wiki `-17.03%` (`+5.87` combined SE), Python `+14.10%` (`-1.38` SE).

#### `gemma-3-1b-it` (F16 ref: Wiki PPL `27.1648 ± 0.9855`, Python PPL `6.4675 ± 0.5186`; `other_tensors_that_differ == []`)

| Variant | Block bpw | WikiText KL ± SE | Wiki PPL | Wiki Top-1 | Python KL ± SE | Py PPL | Py Top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `file:Q4_0` | 4.5000 | 0.253345 ± 0.005435 | 31.397 | 78.775% | 0.238356 ± 0.014255 | 7.365 | 85.042% |
| `file:Q4_K_M` (fallback band) | 5.4818 | 0.082847 ± 0.002452 | 28.302 | 87.441% | 0.065954 ± 0.005000 | 6.461 | 92.941% |
| `file:Q4_K_M-imat` (fallback band) | 5.4818 | 0.049941 ± 0.001732 | 28.070 | 90.343% | 0.051476 ± 0.004613 | 6.586 | 93.613% |
| `ZBG128:b560s16` | 4.4998 | 0.104723 ± 0.002984 | 27.606 | 86.000% | 0.099914 ± 0.008882 | 6.737 | 91.989% |
| `hZBG128:b560s16` | 4.5080 | 0.057824 ± 0.002007 | 28.972 | 89.961% | 0.060526 ± 0.005681 | 6.642 | 93.501% |
| `hZBG128:b560s16~rand` | 4.5080 | 0.064742 ± 0.002647 | 29.021 | 89.137% | 0.060636 ± 0.006004 | 6.728 | 92.605% |
| `EBG128:b560s16` | 4.4999 | 0.098952 ± 0.002928 | 29.422 | 85.794% | 0.084463 ± 0.006297 | 6.599 | 92.213% |
| `HBG128:b560s16` | 4.4999 | 0.096326 ± 0.002587 | 26.214 | 86.373% | 0.093926 ± 0.008255 | 6.343 | 90.980% |
| `hZBG128:b496s16` / `b624s16` | 4.0081 / 5.0081 | 0.110038 / 0.030960 | 30.572 / 27.552 | 85.853% / 92.775% | 0.113014 / 0.030943 | 6.840 / 6.659 | 89.524% / 94.902% |

- **Iso-KL block savings (`hZBG128` vs `Q4_K_M-imat` at `5.4818` fallback bpw):** WikiText `4.6253` bpw (**15.62%** log-linear) / `4.6547` bpw (**15.09%** linear); Python `4.6287` bpw (**15.56%** log-linear) / `4.6610` bpw (**14.97%** linear). At fixed `4.5080` bpw, `hZBG128:b560s16` trails `Q4_K_M-imat` (`+15.78%` Wiki, `-2.97` SE; `+17.58%` Py, `-1.24` SE) because `llama-quantize` promotes 6/7 tensor kinds to Q5/Q6/Q8 on small Gemma models (`5.4818` vs `4.81` normal bpw).

#### `gemma-3-4b-it` (BF16 ref: Wiki PPL `15.3252 ± 0.5652`, Python PPL `4.9915 ± 0.3953`; **PROVISIONAL: stock arms have `other_tensors_that_differ == ["token_embd.weight"]`**)

| Variant | Block bpw | WikiText KL ± SE | Wiki PPL | Wiki Top-1 | Python KL ± SE | Py PPL | Py Top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `file:Q4_0` | 4.5000 | 0.224386 ± 0.006350 | 15.679 | 83.471% | 0.105597 ± 0.008433 | 5.070 | 91.429% |
| `file:Q4_K_M` (normal band) | 4.8151 | 0.151295 ± 0.004989 | 15.730 | 86.725% | 0.065083 ± 0.005131 | 5.107 | 93.782% |
| `file:Q4_K_M-imat` (normal band) | 4.8151 | 0.088353 ± 0.003285 | 16.197 | 90.275% | 0.041692 ± 0.003339 | 5.276 | 94.566% |
| `ZBG128:b560s16` | 4.4998 | 0.112601 ± 0.003894 | 15.966 | 88.833% | 0.041972 ± 0.003957 | 5.523 | 94.678% |
| `hZBG128:b560s16` | 4.5040 | 0.071645 ± 0.002942 | 15.719 | 90.853% | 0.031101 ± 0.003135 | 5.250 | 95.294% |
| `hZBG128:b560s16~rand` | 4.5040 | 0.080353 ± 0.003166 | 16.192 | 90.608% | 0.041052 ± 0.004331 | 5.284 | 94.958% |
| `EBG128:b560s16` | 4.4999 | 0.094284 ± 0.003252 | 16.628 | 89.667% | 0.039951 ± 0.003460 | 5.309 | 94.622% |
| `HBG128:b560s16` | 4.4999 | 0.100675 ± 0.003861 | 16.093 | 89.118% | 0.041296 ± 0.005152 | 5.262 | 94.622% |
| `hZBG128:b496s16` / `b624s16` | 4.0042 / 5.0042 | 0.131462 / 0.037855 | 15.890 / 16.492 | 88.333% / 93.294% | 0.073431 / 0.019908 | 5.062 / 5.279 | 93.221% / 96.415% |

- **Headline & iso-KL block savings (`hZBG128:b560s16` vs `Q4_K_M-imat` at `4.8151` bpw):** Wiki `-18.91%` KL (`+3.79` combined SE), Python `-25.40%` KL (`+2.31` SE) at `0.3111` lower block bpw. Iso-KL savings: Wiki `4.3314` bpw (**10.05%** log-linear) / `4.3644` bpw (**9.36%** linear); Python `4.3335` bpw (**10.00%** log-linear) / `4.3790` bpw (**9.06%** linear).
- **Midpoint test (`hZBG128` `b496`/`b624` -> `b560`):** geometric midpoint error `-1.54%` Wiki / `+22.94%` Py vs arithmetic `+18.16%` Wiki / `+50.06%` Py.

#### `gemma-3-12b-it` (BF16 ref: Wiki PPL `8.6517 ± 0.2633`, Python PPL `4.0554 ± 0.2858`; **PROVISIONAL: stock arms have `other_tensors_that_differ == ["token_embd.weight"]`**)

| Variant | Block bpw | WikiText KL ± SE | Wiki PPL | Wiki Top-1 | Python KL ± SE | Py PPL | Py Top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `file:Q4_0` | 4.5000 | 0.123427 ± 0.003243 | 8.546 | 87.216% | 0.081930 ± 0.008479 | 4.182 | 93.389% |
| `file:Q4_K_M` (normal band) | 4.8076 | 0.073699 ± 0.001969 | 8.573 | 89.863% | 0.047703 ± 0.005451 | 4.180 | 93.782% |
| `file:Q4_K_M-imat` (normal band) | 4.8076 | 0.048831 ± 0.001656 | 8.793 | 92.284% | 0.028232 ± 0.002489 | 4.177 | 95.238% |
| `ZBG128:b560s16` | 4.4998 | 0.054484 ± 0.001722 | 8.627 | 91.284% | 0.030215 ± 0.002867 | 4.186 | 94.734% |
| `hZBG128:b560s16` | 4.5026 | 0.037095 ± 0.001513 | 8.744 | 93.245% | 0.023266 ± 0.002128 | 4.154 | 96.022% |
| `hZBG128:b560s16~rand` | 4.5026 | 0.045139 ± 0.001582 | 8.947 | 92.137% | 0.025698 ± 0.002561 | 4.179 | 96.415% |
| `EBG128:b560s16` | 4.4999 | 0.047357 ± 0.001556 | 8.665 | 92.176% | 0.026768 ± 0.002218 | 4.132 | 95.574% |
| `HBG128:b560s16` | 4.4999 | 0.049364 ± 0.001764 | 8.751 | 91.588% | 0.028539 ± 0.002591 | 4.175 | 96.078% |
| `hZBG128:b496s16` / `b624s16` | 4.0027 / 5.0027 | 0.072152 / 0.019691 | 8.903 / 8.703 | 90.676% / 95.000% | 0.051410 / 0.009819 | 4.172 / 4.170 | 94.846% / 97.367% |

- **Headline & iso-KL block savings (`hZBG128:b560s16` vs `Q4_K_M-imat` at `4.8076` bpw):** Wiki `-24.03%` KL (`+5.23` combined SE), Python `-17.59%` KL (`+1.52` SE) at `0.3050` lower block bpw. Iso-KL savings: Wiki `4.2960` bpw (**10.64%** log-linear) / `4.3352` bpw (**9.83%** linear); Python `4.3806` bpw (**8.88%** log-linear) / `4.4143` bpw (**8.18%** linear).
- **Uncalibrated comparison:** `ZBG128:b560s16` has `26.1%` lower KL on WikiText and `36.7%` lower KL on Python than uncalibrated `Q4_K_M` (at `0.3078` lower block bpw).
- **Midpoint test (`hZBG128` `b496`/`b624` -> `b560`):** geometric midpoint error `+1.61%` Wiki / `-3.43%` Py vs arithmetic `+23.79%` Wiki / `+31.58%` Py.
- **Pilot (`pilot-gemma-3-12b-it.json`):** `n_layers=48`, `peak_rss_mb=10070.2`, all 4 zero-codeword checks `0.0` error on exact zeros (`passed=true`).

#### `gemma-4-E4B-it` `Q8_0` Bridge (Distinct `Q8_0` Reference Stratum — NOT BF16 Evidence)

- Reference: `gemma-4-E4B-it-Q8_0.gguf` (`eval.txt`, 10,200 positions, `ppl_base = 7.595148 ± 0.280652`).
- `VE832:16` (`centred=True`, `4.5000` bpw): KL `0.074682 ± 0.002264`, PPL `7.9029`, Top-1 `91.559% ± 0.275%`. **Passes `P7`** (within `±0.002` of committed `0.0747`).
- `VE8Z32:16` (`centred=False`, `4.5000` bpw): KL `0.077252 ± 0.002314`, PPL `7.7097`, Top-1 `91.608% ± 0.275%`.

#### Partial `gemma-4-E4B-it` BF16 Status (`8/34` Rows on `ca10cda`, Commit `4f1d31a`)

- Pilot passed (`pilot-gemma-4-E4B-it.json`: `n_layers=42`, `294` total / `258` active matrices, `peak_rss_mb=5142.44`, 4/4 zero-codeword checks `0.0` error on exact zeros).
- Completed 4 arms (8 rows): `file:Q4_0`, `file:Q4_K_M`, `file:Q4_K_M-imat`, and `ZBG128:b560s16`.
  - Observed uncalibrated `ZBG128:b560s16` (`4.4998` bpw): Wiki KL `0.083364 ± 0.001828` (PPL `52.019`, Top-1 `88.333%`), Python KL `0.039367 ± 0.003180` (PPL `6.042`, Top-1 `94.454%`).
  - **Stock arms are confounded under `ca10cda`**: all 3 stock arms (`Q4_0`, `Q4_K_M`, `Q4_K_M-imat`) had **86 differing non-block tensors** (`token_embd.weight`, `per_layer_token_embd.weight`, and 84 `blk.{0..41}.{inp_gate,proj}.weight`).
- **Why it stopped:** crashed entering `hZBG128:b560s16` with `KeyError: 'blk.24.attn_k.weight.in_sum2'` because Gemma 4 E4B's 18 KV-sharing layers (`blk.24..41` `attn_k`/`attn_v`) are never read during `llama-imatrix` calibration.
- **Resolution:** both bugs are now fixed on `exp/gemma-ball-runner @ ccf045d` (`a5f2715` unweighted fallback + provenance logging; `ccf045d` preserves all non-block source dtypes). **Restart E4B BF16 cleanly in a fresh output directory on `ccf045d`** (do not reuse `ca10cda` stock rows).

---

## 3. Critical Caveats, Confounds & Runner Fixes (MUST READ BEFORE INTERPRETING OR RUNNING)

1. **`token_embd.weight` BF16 -> F16 mismatch on 4B and 12B stock arms (`ca10cda`, fixed in `6a9d871`):**
   - Under `ca10cda`, `quantize_tensor_type_flags()` hardcoded `^token_embd\.weight$=f16`. On F16 models (`270M`, `1B`) this was lossless (`other_tensors_that_differ == []`), but on BF16 models (`4B`, `12B`) it downcast `token_embd.weight` to F16 on stock arms (`other_tensors_that_differ == ["token_embd.weight"]`) while ball arms copied BF16 unchanged.
   - All 4B and 12B ball-vs-stock comparisons are **provisional** until tensor-matched stock reruns land on `ccf045d` (`other_tensors_that_differ == []`).
2. **`per_layer_token_embd.weight` (37.6% of Gemma 4 E4B) and non-block native dtype preservation (fixed in `ccf045d`):**
   - On Gemma 4 E4B, `ca10cda` mutated 86 non-block tensors on stock arms, including `per_layer_token_embd.weight` (37.6% of total model parameters).
   - Commit `ccf045d` inspects every non-block tensor's source `tensor_type` in the GGUF and emits exact `--tensor-type` preservation flags for all of them.
3. **`hEBG128` / `hHBG128` real-imatrix unlock (`RUN-BUG-01`, fixed in `6a9d871`):**
   - `ca10cda` hardcoded `a.startswith("hZBG")` when deciding whether to load/record the real imatrix, blocking `hEBG128:b560s16` and `hHBG128:b560s16`. Commit `6a9d871` replaces this with `arm_needs_real_imat()` matching `WEIGHTED_BALL_RE` and `h`-prefixed `LOUD_BALL_RE`.
4. **Missing imatrix tensor key unweighted fallback + provenance logging (fixed in `a5f2715`):**
   - When an imatrix file lacks `<tensor>.in_sum2` or has `counts == 0` (as in Gemma 4 E4B's 36 unread KV-sharing matrices), `run.py` now falls back to unweighted rounding for that matrix and records `imatrix_fallback_count` and `imatrix_fallback_tensors` in `results.jsonl`.
5. **Operational gotcha — never update `run.py` while multiprocessing workers are live on macOS:**
   - Python on macOS uses the `spawn` start method for `multiprocessing.Pool`. Updating or checking out a new `run.py` mid-sweep causes newly spawned workers to import mismatched code against the parent process. **Always switch runner commits strictly between model jobs.**
6. **Scope & interpretation limits:**
   - **Block-storage savings != whole-model storage or serving speed:** All `bpw` and iso-KL savings in `results.jsonl` measure quantized block matrices only (excluding embeddings, norms, metadata, and packed-format overhead) and do not measure serving throughput or resident RAM.
   - **PPL alone can disagree with KL:** For example, on `gemma-3-12b-it` WikiText, `file:Q4_0` has lower raw PPL (`8.546`) than `file:Q4_K_M-imat` (`8.793`) despite having `2.53x` higher KL (`0.123427` vs `0.048831`). Always lead with mean KL to reference (and top-1 keep rate).
   - **`Q8_0` bridge != BF16 evidence:** Never pool `gemma-4-E4B-it` `Q8_0` bridge rows (`eval.txt`) with 16-bit headline rows (`wikitext` / `python`).
   - **64 GB RAM ceiling:** `gemma-3-27b-it` and `gemma-2-27b-it` (~54 GB weights alone at BF16) exceed a 64 GB host when holding reference + quantized model + activations. **27B models must use a `Q8_0` reference only (never BF16)** and stay in a separate quantized-reference stratum.
   - **Combined SE is not a paired significance test:** `llama-perplexity` reports marginal standard errors; formal paired inference requires per-position paired bootstrap (see `P2-1`).

---

## 4. Exact Environment & Reproducible Run Commands

### Required Tools & Dependencies

- **llama.cpp binaries** (in `PATH` or passed via flags): `llama-cli`, `llama-imatrix`, `llama-perplexity`, `llama-quantize`.
  - *Version pin caveat:* Historical Gemma `results.jsonl` receipts do **not** record the `llama.cpp` build commit (earlier Qwen/SmolLM2/Llama runs pinned `llama.cpp @ eada89c`, while Gemma 4 E4B requires a `llama.cpp` build supporting Gemma 4). Record your exact binary version before running and verify calibration reproducibility via the resulting `imatrix_sha256`.
- **Python 3 virtual environment**: `numpy`, `gguf` (matching `llama.cpp`), `torch` (for `harness/`), and a C compiler (for `shape/ballpack.c`).
- **Datasets** (not committed; verify SHA-256 before running):
  - `<DATA_DIR>/wiki.test.raw` — SHA-256 `173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`
  - `<DATA_DIR>/wiki.train.raw` — WikiText-2 train split for real-text `llama-imatrix`; repo pin from `experiments/imatrix-baseline-2026-11-17/PREREG.md` is SHA-256 `6707892fa3788b5ab9ed78ab5ff37d9fe825f6011a2ad4fcd6a6d467f0e7da57` (*note:* historical Gemma `results.jsonl` receipts do not log `wiki.train.raw` SHA-256 directly—only the resulting per-model `imatrix_sha256` in Section 2).
  - `<DATA_DIR>/base.py` (`conversion/base.py` from `llama.cpp @ eada89c`) — SHA-256 `e38975e1c68d98ac1664dfd530616eb35c72294382a4dd873d4746b23f27779f`
  - `<DATA_DIR>/eval.txt` (`harness/build_eval_text.py` from `llama.cpp @ 427291b`) — SHA-256 `df5748784801c869b7d1ce54b169af665bfa4c0d3c9839c88d097ffbd33df761`

### Pre-Calibration Binary & Digest Check

Before launching a sweep or single-arm comparison, record the exact `llama.cpp` versions and verify all input digests (reference GGUF, evaluation/calibration texts, and `eval.txt` when running `P7` bridge mode). When reusing or validating a model's imatrix, confirm the cached file inside the run's `--out-dir` (`<RUN_OUT_DIR>/cache/<model_id>.wikitext.imatrix.gguf`, e.g. `<OUT_DIR>/<model_id>/cache/<model_id>.wikitext.imatrix.gguf` for full sweeps) matches the Section 2 `imatrix_sha256`:

```bash
llama-imatrix --version && llama-perplexity --version && llama-quantize --version
shasum -a 256 <MODELS_DIR>/<model_file>.gguf <DATA_DIR>/wiki.test.raw <DATA_DIR>/wiki.train.raw <DATA_DIR>/base.py
# For P7 bridge mode, also verify eval.txt:
shasum -a 256 <DATA_DIR>/eval.txt
# After imatrix generation (or on cached imatrix inside the --out-dir passed to run.py), compare against Section 2 imatrix_sha256:
shasum -a 256 <OUT_DIR>/<model_id>/cache/<model_id>.wikitext.imatrix.gguf
```

### Checkout & Runner Pin

```bash
git fetch origin exp/gemma-ball-runner
git checkout --detach ccf045d04ada2170eeafaa37254a764b96552723
test "$(git rev-parse HEAD)" = "ccf045d04ada2170eeafaa37254a764b96552723"
```

### Standard `run.py` CLI Invocations

Use generic paths `<MODELS_DIR>`, `<DATA_DIR>`, `<OUT_DIR>`.

#### 1. Dry-run (plan inspection without executing `llama.cpp`)

```bash
python3 experiments/model-output-gemma-ball-2026-11-23/run.py \
  --model-id gemma-4-E4B-it \
  --model-path <MODELS_DIR>/gemma-4-E4B-it-BF16.gguf \
  --out-dir <OUT_DIR>/gemma-4-E4B-it \
  --wikitext-test <DATA_DIR>/wiki.test.raw \
  --wikitext-train <DATA_DIR>/wiki.train.raw \
  --python-text <DATA_DIR>/base.py \
  --dry-run
```

#### 2. Step 2/3 One-Layer Pilot & Zero-Codeword Gate

```bash
python3 experiments/model-output-gemma-ball-2026-11-23/run.py \
  --model-id gemma-4-E4B-it \
  --model-path <MODELS_DIR>/gemma-4-E4B-it-BF16.gguf \
  --out-dir <OUT_DIR>/gemma-4-E4B-it \
  --pilot-only --pilot-layer 0 --workers 4
```

#### 3. Full 17-Arm Sweep (34 Rows: `wikitext,python`)

```bash
python3 experiments/model-output-gemma-ball-2026-11-23/run.py \
  --model-id gemma-4-E4B-it \
  --model-path <MODELS_DIR>/gemma-4-E4B-it-BF16.gguf \
  --out-dir <OUT_DIR>/gemma-4-E4B-it \
  --texts wikitext,python \
  --wikitext-test <DATA_DIR>/wiki.test.raw \
  --wikitext-train <DATA_DIR>/wiki.train.raw \
  --python-text <DATA_DIR>/base.py \
  --ngl 999 --chunks 40 --imatrix-chunks 100 --ctx 512 --batch 512 --threads 4 --workers 4
```

#### 4. Targeted Single-Arm / Tensor-Matched Stock Reruns (on `ccf045d`)

```bash
# Tensor-matched stock rerun (verify other_tensors_that_differ == [])
python3 experiments/model-output-gemma-ball-2026-11-23/run.py \
  --model-id gemma-3-12b-it \
  --model-path <MODELS_DIR>/gemma-3-12b-it-BF16.gguf \
  --out-dir <OUT_DIR>/gemma-3-12b-it-stock-matched \
  --texts wikitext,python \
  --wikitext-test <DATA_DIR>/wiki.test.raw \
  --wikitext-train <DATA_DIR>/wiki.train.raw \
  --python-text <DATA_DIR>/base.py \
  --arms file:Q4_0,file:Q4_K_M,file:Q4_K_M-imat \
  --ngl 999 --chunks 40 --imatrix-chunks 100 --ctx 512 --batch 512 --threads 4 --workers 4

# PRED-M1-01 weighted E8 / Hadamard-E8 ball arms on 4B (requires >= 6a9d871; use ccf045d)
# Run hEBG128:b496s16 + hEBG128:b560s16 for the direct log-linear iso-KL bracket (see Priority 1b for 4B PRED-M1-01 and 1c for the separate 12B scaling check)
python3 experiments/model-output-gemma-ball-2026-11-23/run.py \
  --model-id gemma-3-4b-it \
  --model-path <MODELS_DIR>/gemma-3-4b-it-BF16.gguf \
  --out-dir <OUT_DIR>/gemma-3-4b-it-hebg \
  --texts wikitext,python \
  --wikitext-test <DATA_DIR>/wiki.test.raw \
  --wikitext-train <DATA_DIR>/wiki.train.raw \
  --python-text <DATA_DIR>/base.py \
  --arms hEBG128:b496s16,hEBG128:b560s16,hHBG128:b560s16 \
  --ngl 999 --chunks 40 --imatrix-chunks 100 --ctx 512 --batch 512 --threads 4 --workers 4
```

#### 5. `P7` Bridge Mode (`Q8_0` Reference on `eval.txt`)

```bash
python3 experiments/model-output-gemma-ball-2026-11-23/run.py \
  --model-id gemma-4-E4B-it \
  --model-path <MODELS_DIR>/gemma-4-E4B-it-Q8_0.gguf \
  --out-dir <OUT_DIR>/gemma-4-E4B-it-bridge \
  --bridge-eval <DATA_DIR>/eval.txt \
  --arms VE832:16,VE8Z32:16,EBG128:b560s16 \
  --ngl 999 --chunks 40 --ctx 512 --batch 512 --threads 4 --workers 4
```

### Verification Gates (Run Before Every Commit / Push)

```bash
# 3-mode Python parity on runner tests (on exp/gemma-ball-runner)
python3 -B -m unittest experiments/model-output-gemma-ball-2026-11-23/test_run.py
python3 -O -B -m unittest experiments/model-output-gemma-ball-2026-11-23/test_run.py
python3 -B -W error -m unittest experiments/model-output-gemma-ball-2026-11-23/test_run.py

# Repository governance checks (3-mode parity)
python3 -B checks/plan_before_data.py && python3 -B checks/claim_ledger.py
python3 -O -B checks/plan_before_data.py && python3 -O -B checks/claim_ledger.py
python3 -B -W error checks/plan_before_data.py && python3 -B -W error checks/claim_ledger.py
```

---

## 5. Prioritized Next-Experiments Queue (Hypotheses, Controls & Falsifiers)

### Priority 1 — Immediate Gemma Closure (Runner `ccf045d`)

| Order | Experiment | Hypothesis / Pre-Registered Prediction | Matched Control & Falsifier |
|---|---|---|---|
| **1a** | Tensor-matched stock reruns on `gemma-3-4b-it` and `gemma-3-12b-it` (`file:Q4_0`, `file:Q4_K_M`, `file:Q4_K_M-imat`) | `PRED-M1-02`: preserving BF16 `token_embd.weight` changes `Q4_K_M-imat` KL by `< 0.002`, confirming 4B and 12B ball wins are genuine block-matrix gains. | Require `other_tensors_that_differ == []` on every stock row. **Falsifier:** if matched `Q4_K_M-imat` drops below `hZBG128:b560s16` or narrows Wiki gap below `2.0` combined SE, retract provisional 4B/12B `P1` win. |
| **1b** | `hEBG128:b496s16`, `hEBG128:b560s16` & `hHBG128:b560s16` on **`gemma-3-4b-it` only** | `PRED-M1-01` (`studies/expedition-gemma-m1-tomography-audit130r2/GEMMA_M1_TOMOGRAPHY_REPORT.md`, registered for **`gemma-3-4b-it` only**): combining $E_8^{16}$ Voronoi packing (`+0.65 dB` over `ZBG128`) with half-strength real imatrix on 4B (`b560 = 4.5040` bpw) is forecast to cut Wiki KL from `0.071645` (`hZBG128:b560s16`) to **`0.061685`** (theoretical $10^{-0.065}$ transfer; Python KL **`0.026777`**; predicted `hEBG128` iso-KL parity rate **`4.2081` bpw** vs stock `file:Q4_K_M-imat` baseline **`b_stock = 4.8151` bpw**, **`12.60%`** matched-KL block savings) or **`0.059990`** Wiki KL (empirical `EBG/ZBG` transfer; predicted `hEBG128` iso-KL parity rate **`4.1852` bpw** vs `b_stock = 4.8151` bpw, **`13.08%`** matched-KL block savings, strictly `> 12%`). | Same 4B real `imatrix_sha256` and block rates (`b496 = 4.0042` bpw, `b560 = 4.5040` bpw) as 4B `hZBG128`, evaluated against 4B tensor-matched `file:Q4_K_M-imat` (`b_stock = 4.8151` bpw, `other_tensors_that_differ == []`). **Primary bracket rule (4B `b496` + `b560`; require monotone `KL(b560) < KL(b496)`, otherwise halt and treat the bracket as anomalous):** (a) **bracketed** (`KL(b560) < KL(stock) < KL(b496)`): compute exact log-linear `hEBG128` parity rate $b_{\text{match}} = b_{496} + (b_{560} - b_{496})\ln(K_{496}/K_{\text{stock}})/\ln(K_{496}/K_{560})$ and block savings $(b_{\text{stock}} - b_{\text{match}})/b_{\text{stock}}$; (b) **both below stock** (`KL(b496) <= KL(stock)`): parity is $\le 4.0042$ bpw (`>= 16.84%` saving vs `b_stock = 4.8151` bpw), so the `> 12.0%` savings gate **passes without interpolation** (add a lower-rate arm only if an exact parity point is desired); (c) **both at or above stock** (`KL(b560) >= KL(stock)`): parity is $\ge 4.5040$ bpw (`<= 6.46%` saving vs `b_stock = 4.8151` bpw), so **`PRED-M1-01` is falsified immediately without interpolation**. **Secondary single-arm diagnostic (only if `b496` is omitted):** use 4B's pinned `hZBG128` (`b496`/`b560`) WikiText log-KL/bpw slope from `dc10d4bbca`, $\alpha_{\text{4B}} = \ln(0.131462/0.071645)/(4.504050 - 4.004156) = 1.214247\,\text{nat/bpw}$, giving diagnostic $b_{\text{match}} = b_{560} - \ln(K_{\text{stock}}/K_{\text{hebg}})/\alpha_{\text{4B}}$. **Falsifier (4B only):** `PRED-M1-01` is falsified if 4B `hEBG128:b560s16` either (i) fails to beat 4B `hZBG128:b560s16` by `>= 2.0` combined SE on WikiText KL or (ii) fails the `> 12.0%` matched-KL block savings gate vs 4B tensor-matched `file:Q4_K_M-imat` (`b_stock = 4.8151` bpw) under the primary bracket rule above. |
| **1c** | `hEBG128:b496s16`, `hEBG128:b560s16` & `hHBG128:b560s16` on **`gemma-3-12b-it`** *(out-of-sample scaling check — not part of `PRED-M1-01`)* | Tests whether the weighted $E_8^{16}$ Voronoi packing gain over `hZBG128` transfers at 12B scale (`b560 = 4.5026` bpw vs 12B `hZBG128:b560s16` Wiki KL `0.037095` and tensor-matched 12B `file:Q4_K_M-imat` baseline `b_stock = 4.8076` bpw; the earlier `10.64%` 12B `hZBG128` figure is provisional under `ca10cda` and must be recomputed against the tensor-matched stock rerun). | Same 12B real `imatrix_sha256` and block rates (`b496 = 4.0027` bpw, `b560 = 4.5026` bpw) against the 12B tensor-matched `ccf045d` `file:Q4_K_M-imat` rerun (`b_stock = 4.8076` bpw, `other_tensors_that_differ == []`). **Baseline alignment:** first recompute 12B `hZBG128` (`b496`/`b560`) log-linear matched-KL block savings against that **same tensor-matched `ccf045d` stock rerun** (do not compare tensor-matched `hEBG128` against the provisional `ca10cda` `10.64%` figure). **Primary bracket rule (12B `b496` + `b560`; require monotone `KL(b560) < KL(b496)`, otherwise halt and treat as anomalous):** (a) bracketed → exact log-linear $b_{\text{match}}$ and savings $(b_{\text{stock}} - b_{\text{match}})/b_{\text{stock}}$; (b) `KL(b496) <= KL(stock)` → parity $\le 4.0027$ bpw (`>= 16.74%` saving vs `b_stock = 4.8076` bpw, passes without interpolation); (c) `KL(b560) >= KL(stock)` → parity $\ge 4.5026$ bpw (`<= 6.34%` saving, fails transfer immediately). **Secondary single-arm diagnostic (only if `b496` is omitted):** use 12B's pinned `hZBG128` (`b496`/`b560`) WikiText log-KL/bpw slope from `ba5ec5a64c`, $\alpha_{\text{12B}} = \ln(0.072152/0.037095)/(4.502556 - 4.002662) = 1.330868\,\text{nat/bpw}$. **Transfer criterion (independent of `PRED-M1-01`):** confirms 12B transfer if 12B `hEBG128:b560s16` beats 12B `hZBG128:b560s16` by `>= 2.0` combined SE on WikiText KL and improves 12B matched-KL block savings above the recomputed tensor-matched 12B `hZBG128` baseline; a 12B miss records non-transfer at 12B without falsifying 4B `PRED-M1-01`. |
| **1d** | Complete `gemma-4-E4B-it` BF16 full sweep (`34/34`) + `EBG128:b560s16` on `gemma-4-E4B-it` `Q8_0` bridge | Full E4B BF16 closes the primary `PREREG.md` model (`P1`–`P10` with both counted-294 and active-258 accounting); bridge `EBG128:b560s16` directly links the 128-weight $E_8$ ball to `VE832:16 = 0.074682`. | Fresh directory on `ccf045d`: require `other_tensors_that_differ == []` on all 3 stock arms and verify `imatrix_fallback_tensors` contains only the 36 unread KV-sharing tensors (`blk.24..41` `attn_k`/`attn_v`). Keep `Q8_0` bridge in its own table. |
| **1e** | `gemma-3-27b-it` (`Q8_0` reference **only**) | Tests whether the `4B -> 12B` block-rate advantage persists at 27B scale without exceeding 64 GB RAM. | **Hard constraint:** never run BF16 27B on 64 GB host. Label all 27B rows explicitly as `Q8_0`-reference stratum; require `other_tensors_that_differ == []`. |

### Priority 2 — Pass 02 Controlled Experiments (Commit Preregistration Before Data)

1. **`P2-1` — Common-Envelope 4B Control:**
   - **Arms:** `hZBG128:b552s16` (primary) plus rate-bracketing `hZBG128:b536s16`, `b544s16`, `b560s16` vs `Q4_K_M-imat` on `gemma-3-4b-it`.
   - **Envelope & controls:** byte-identical non-target tensors verified by hash; count **actual serialized bytes + metadata** (not just nominal block bpw); include both an **unchanged-payload negative control** (must score `KL == 0.0` identically) and an **embedding-only perturbation negative control**; evaluate on held-out text; compute paired inference via **paired bootstrap (seed `20261123`, `10,000` draws)**.
   - **Falsifier:** ball arm fails to beat `Q4_K_M-imat` at equal or lower actual serialized byte count with 95% paired bootstrap CI strictly excluding zero.

2. **`P2-2` — Off-Diagonal Covariance-Aware Scale Selection at Zero Extra Bits:**
   - **Operator:** $C = D^{1/2}\bigl[I + \lambda(L - \operatorname{diag} L)\bigr]D^{1/2}$ with $\lambda = 1/4$, selecting per-group scales without adding any stored metadata bits.
   - **Protocol:** `gemma-3-1b-it` pilot first, then blind confirmation on `gemma-3-4b-it` (no retuning between 1B and 4B).
   - **Nulls & pass gate:** compare against **200 IID nulls** and **200 spectrum-preserving nulls**; require **$p \le 0.025$** and **$\ge 1$ percentage point** improvement above **both** null medians.
   - **Falsifier:** fails either null threshold on blind 4B confirmation.

3. **`P2-3` — Equal-Cost Layer-Upgrade Interaction Test:**
   - **Protocol:** reallocate bits across layers at strictly equal total byte cost; fit/test on `gemma-3-1b-it` first, then transfer the fixed rule to `gemma-3-4b-it`.
   - **Pass gate & falsifier:** require **$\ge 5\%$ held-out KL improvement on both headline texts** (`wikitext` and `python`) on 4B transfer; fail/close if either text misses `5%`.

### Priority 3 — Hardware, Packing & Weight-Space Integrity Fronts

- **Packed-format serving benchmark & downstream evals:** benchmark actual packed files (`shape/ballpack.py` / `shape/ballpack.c` and 128-weight ball packing) for **file bytes, resident RAM, decode tok/s, and latency**, and run downstream language/code benchmarks (e.g., HumanEval, GSM8K, full MMLU) beyond KL/PPL.
- **TPU / Edge `INT8` & flat-LUT kernel benchmarks (PR #65):** benchmark `Block-H2` native `INT8` MXU (`max = 78 <= 127`, `3x` fewer transform ops than 3-stage `WHT_8`) and `O(1)` branchless 2-stage residual flat-LUT `E_8` (`32 KiB` L1d / `512 KiB` VMEM) on real hardware.
- **Geometric & systems fronts (`N1`–`N6`):**
  - `N1`: `BW16` and `Leech24` lattice quantization on real model weights.
  - `N2`: Hermitian-box exact error certificates.
  - `N3`: Two-radius cross-polytope codebooks.
  - `N4`: Permanent / rook-polynomial unranking decoders.
  - `N5`: Loud-column outlier rejection / mixed-precision routing.
  - `N6`: Pallas / NEON / SIMD decoder lowering.
- **Weight-space integrity (Issue #59):**
  - Attach **per-tensor / per-row immutable-zone hash locks** on `token_embd.weight` (and untied `lm_head`) in `ballpack` / GGUF manifests so localized embedding edits cannot hide outside quantized block coverage.
  - Add zero-inference-cost `ballpack` inspection: `O(N)` Deep Patch Inspection (`DPI`) via Row-Energy Concentration Ratio $\mathcal{C}_{\mathrm{row}}(\Delta W) = R \cdot \max_r \|\Delta W_{r,:}\|_2^2 / \|\Delta W\|_F^2$ and Broad Weight Inspection (`BWI`) over per-group scales $t_g$, `share_out_of_range`, and $5\sigma$ loud columns.

---

## 6. Methodology & Verification Standards

Every follow-on study in `Moissanite-Labs/siliquant` must satisfy these rules before results are published or merged:

1. **Plan before data:** Commit a preregistration (`PREREG*.md`) before inspecting or running new target data; pin inputs, metrics, numerical thresholds, seeds, controls, nulls, and stop rules. Any post-hoc change is logged separately as a deviation.
2. **Steelman before falsify:** Fix prototype or harness bugs (such as `RUN-BUG-01..02` or dtype mismatches) before calling a hypothesis falsified, and state explicit numerical falsifiers upfront.
3. **Matched controls & adversarial nulls:** Verify `other_tensors_that_differ == []` (byte-identical non-target tensors), include negative controls (unchanged-payload, embedding-only), and test structural claims against both IID and spectrum-preserving nulls.
4. **Exact SHA-256 provenance:** Never commit model weights or datasets. Record exact SHA-256 hashes for every model file, evaluation text, calibration imatrix, `results.jsonl`, and `results.out`, plus the exact runner commit SHA.
5. **Clean-room verification:** Recompute headline tables, iso-KL interpolations, and certificates independently from raw JSONL/bytes; use exact $\mathbb{Z}/\mathbb{Q}$ arithmetic (no float cutoffs, no bare `assert` on trust paths) for algebraic/bounds claims.
6. **Pre-push verification & public hygiene:**
   - Pass 3-mode Python parity (`python3 -B`, `python3 -O -B`, `python3 -B -W error`) on tests and checkers.
   - Pass `python3 checks/plan_before_data.py` and `python3 checks/claim_ledger.py`.
   - Keep public files, commits, and PR bodies free of personal local paths, secrets, internal working names, or uncommitted binary artifacts.
