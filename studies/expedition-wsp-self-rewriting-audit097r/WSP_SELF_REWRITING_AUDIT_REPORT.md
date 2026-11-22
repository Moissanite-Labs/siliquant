# AUDIT-097 — Exact Pure-Q Audit of Weight-Edit Minimization & Self-Rewriting LLMs (`warp-drive`, `siliquant`, & `aletheia v2.0.0` Operator Bridge)

- **Audit ID:** `AUDIT-097` (Track S1)
- **Schema Version:** `audit097.wsp_self_rewriting.v1`
- **Honest Law-4 Novelty Classification:** **`N1`** (new exact pure-$\mathbb{Q}$ executable operator bridge, live bit-level tensor/screen/quantization recount, and fail-closed verification gates over `N0` primary artifacts)
- **Primary Sources Audited (Strictly Read-Only, Pinned SHA-256):**
  1. `arXiv:2611.21885v2` (*Self-Rewriting LLMs*, `sources/arxiv_2611_21885.pdf`, `7,277,171` B, `sha256: 176d0a9b6068ac63...`)
  2. `alphaXiv/weight-edit-minimization` @ `768d7340142b1882d4bd2740b4723274e4c9278d` (`openresearch.sh/p/EdsyLoFUfY40`, `raw/openresearch_EdsyLoFUfY40.txt`, `16,973` B, `sha256: 4c3d5c355bb98e11...`)
- **Executable Artifacts (100% Pure Python 3 Stdlib, Exact `int`/`Fraction`, `0` Floats, `0` Bare `assert`s):**
  - [wsp_self_rewriting_oracle.py](wsp_self_rewriting_oracle.py) — pure-$\mathbb{Q}$ rank-1/pruning leakage algebra, `siliquant` group-quantization survival oracle, `warp-drive` Stage-2 division-free `FixedRadixCertificate` verifier, paired dual-metric McNemar gate, and 8-gate self-rewriting governance certifier.
  - [wsp_self_rewriting_audit.py](wsp_self_rewriting_audit.py) — dual-mode (`--live` raw `.npz`/`.npy`/`.jsonl` recount + `--standalone` offline fixture replay) runner with fail-closed `validate_audit_certificate()`.
  - [test_wsp_self_rewriting_r5.py](test_wsp_self_rewriting_r5.py) — assertion-free `-I -O -B -W error` R5 mutation-killing, single-conjunct AST-killing, and known-answer live-path test suite.
  - [wsp_self_rewriting_certificate.json](wsp_self_rewriting_certificate.json) — canonical zero-float live JSON certificate (`live_intake_audited: true`, `overall_verified: true`).

---

## 1. Executive Summary & Law-5 Operator Bridges (Zero Metaphor)

Every link across the four repositories is an **exact mathematical operator or machine-checkable trust boundary**, not a metaphorical analogy:

1. **Exact Subspace-Leakage & Safety-Orthogonality Dichotomy (`Exp 8` $\to$ `Moissanite-Labs/warp-drive` Stage-2 `FixedRadixCertificate`):**
   - For a rank-1 edit $\Delta_0 = u v^\top$ whose left direction satisfies safety orthogonality $S^\top u = 0$:
     - **Column pruning (`A` family: `A48_topk48`, `B15_full`)** preserves exact rank-1 ($\|\Delta\|_F^4 - \|\Delta^\top\Delta\|_F^2 = 0$), zero orthogonal leakage ($\|(I-P_u)\Delta\|_F^2 = 0$), and exact safety orthogonality ($S^\top \Delta = 0$). On live `Qwen3.5-0.8B` `Exp 8` `bf16` tensors, `A64_topk64` has **`0.00067%`** off-rank-1 leakage (`0.000034 / 5.015146`) and `A48_topk48` has **`0.00056%`** (`0.000028 / 4.859552`, **`0.83×`** of `A64`), both **PASSING** `warp-drive` Stage-2 bounds.
     - **Row pruning (`ER` family: `ER384_L15`)** remains rank-1 yet rotates $u \mapsto u_R = m_r \odot u$, obeying the **exact closed-form identity** $\|(I-P_u)\Delta_{\mathrm{row}}\|_F^2 = E_R(1-E_R)\|\Delta_0\|_F^2$ (leakage fraction $1 - E_R$) and breaking $S^\top u_R = 0$. On live `ER384_L15` ($E_R = 92.77\%$), measured `bf16` leakage is **`7.232%`** (`0.205191`, **`6,113.75×`** of `A64`), **FAILING** Stage-2 `leakage_energy`.
     - **Entry-wise top-$|\Delta|$ pruning (`D` family: `D4k_entries`)** shatters rank-1 ($\|\Delta\|_F^4 > \|\Delta^\top\Delta\|_F^2$), breaks safety orthogonality ($S^\top \Delta_M \neq 0$), and on live `Exp 8` `D4k_entries` (`8,192` entries) explodes off-direction leakage **`23,696.11×`** (`0.000034 \to 0.795295`, **`21.159%`** of total edit energy outside $\mathrm{span}(u)$), **FAILING** Stage-2 `leakage_energy`.
2. **Hidden Paired Loop-Breaking Regression & Unpaired Validation in `Exp 8`:**
   - Bit-level `bf16` RNE recount proves `edit.npz` modifies **`130,188`** entries (`65,260` in Layer 15 + `64,928` in Layer 19), NOT the `131,072` (`65,536 + 65,536`) reported by `frozen_edit_audit.json` — **`884` entries collapse to exact `bf16` no-ops**, including **all 64 entries of Layer 19 Row 685** where `factors_layer19.npz` has exact `u19[685] == 0`.
   - While `D4k_entries` ties `A64_topk64` on noisy `n=1` accuracy (`149/256` vs `150/256`, exact McNemar `p = 1.000000`), exact pure-$\mathbb{Q}$ paired McNemar on the **causal termination metric (`cut_at_32k`)** reveals a statistically significant regression: `99/256` (`38.7%`) vs `77/256` (`30.1%`, **`+43 / -21`**, exact **`p = 0.008147 <= 0.01`**). `A48_topk48` (`p = 0.000039`), `B15_full` (`p = 0.007084`), `ER384_L15` (`p = 0.000317`), and `ER768_L15` (`p = 0.001294`) also fail the paired truncation gate; only `D8k`/`D16k`/`D32k` and `B15_full_s125` pass.
   - `validate_D4k_math500.jsonl` recounts cleanly to **`1,084 / 2,000` (`54.20%`)** across `500` disjoint MATH-500 IDs (`0` overlap with the `256` MATH-train screens; sample-0 `272/500`), but has **`0` paired `A64` or `G000` MATH-500 control files** (its paired arms did not finish).
3. **$L_0$-vs-$L_2$ Norm Blowup Anti-Correlation (`car-ramp` $\to$ `warp-drive` Stage-2 & `siliconcordat#15` `||W - W0|| <= R`):**
   - Minimizing $L_0$ entry count on `car-ramp` (`171,664` actor params) from `6,073` (`edited_actor`) $\to$ `12` (`greedy12_astra`) $\to$ `10` (`c2_min_10entry`, `0.0058%` $L_0$) requires extreme magnitude spikes (`±876`, `±200`, `±40`) yielding exact $\|\Delta\|_F^2 = 1,767,771.11$, $87,396.95$, and $4,432.56$ — exceeding `Exp 8` (`5.015146` across both `Qwen3.5-0.8B` layers) by **`352,487×`**, **`17,427×`**, and **`884×`**, and **FAILING** `warp-drive` Stage-2 Frobenius and spectral bounds on all 6 checkpoints.
4. **Live `Moissanite-Labs/siliquant` `ffn_down` Quantization-Survival Double Bind (`Part L`, `#10`, `#12`, `#21`, `#26`, `siliconcordat#16` `WEFT-T3`):**
   - **Small dense rank-1 edits (`Exp 8` `down_proj`):** Under symmetric INT4 (`qmax = 7`) at `G = 128` (`ZBG128` group size), **`57,060 / 130,188` (`43.83%`)** of Astra's `L15+L19` edited weights are **completely erased** (`q_edit == q_base`), **`79,380 / 130,188` (`60.97%`)** fall below `siliquant#26`'s **`(3/4)·ULP`** rounding-cell threshold, and **`87,619` unedited base weights flip codes collaterally** (`1.20×` more collateral base corruptions than the `73,128` surviving edited weights!).
   - **Large sparse $L_0$-spike edits (`car-ramp` `C2-min`, `greedy12`):** Spiking `actor_head.W` inflates group max **`60.979×`–`657.963×`**, collapsing **`100%` of non-zero unedited neighbors** (`29/29` and `80/80` in `C2-min`; `24/24` and `90/90` in `greedy12`) directly to quantized code `0`.
5. **Fail-Closed Self-Rewriting Governance Audit (`arXiv:2611.21885v2` $\to$ `warp-drive` Stages 1–4, `siliconcordat#16` `WEFT-T5`, & `aletheia v2.0.0` `R6`/`R7` + Gate-8):**
   - Protocol 1 (`27B` self-edit & `27B` damaged repair) fails **7/8** gates (`G1` unconfined same-user Python process in App. E.3, `G2` model self-submission, `G4` `5`–`8` practice-to-held-out repeats in App. D.3, `G5` `19` unnecessary matrix clips costing `-22.0 pp` in Sec. 3.5, `G6` zero safety canaries, `G7` no unrelated-domain KL latch, `G8` no DeltaNet SSM state-bank flush confirmation).
   - Protocol 2 (`Session 8` uniform `dt_bias + 0.5` across 48 DeltaNet layers) closes `G1` (sealed folder) and `G5`, yet still fails **6/8** gates — including **`G3_cumulative_lineage_reset_mismatch`** because Fig. 1 / App. E.9 explicitly reset every session from base weights rather than composing edits cumulatively.

---

## 2. Live Pure-Q Recount — `Exp 8` (`Qwen3.5-0.8B`) Tensor Leakage & Stage-2 Verdicts

Exact dyadic pure-$\mathbb{Q}$ recount of `data/exp8/{edit.npz, base_l{15,19}_down_proj_f32.npy, factors_layer{15,19}.npz, specs/*.json}` (zero `numpy`, zero floats):

- **Layer 15 (`[1024, 3584]`):** `65,260` non-zero `bf16` entries (`1,024` rows × `64` cols; `276` `bf16` no-op collapses); exact `||u15||^2 * ||v15||^2 = 3.058591`.
- **Layer 19 (`[1024, 3584]`):** `64,928` non-zero `bf16` entries (`1,023` rows × `64` cols; `608` `bf16` no-op collapses, **missing row `685`** where exact `u19[685] == 0`); exact `||u19||^2 * ||v19||^2 = 1.956522`.
- **Cross-Layer Direction Alignment:** exact $\cos^2(u_{15}, u_{19}) = 0.720809$ ($|\cos| \approx 0.8490$), exact $\cos^2(v_{15}, v_{19}) = 0.000759$ ($|\cos| \approx 0.0275$).

| Spec Arm | Active `bf16` Entries | Total $\|\Delta\|_F^2$ | Off-Rank-1 Leakage $\|\Delta_\perp\|_F^2$ | Off-Rank-1 % | Leakage vs `A64` | Stage-2 (`warp-drive`, `R=2^32`) |
|---|---:|---:|---:|---:|---:|---|
| `A64_topk64` | `130,188` | `5.015146` | `0.000034` | `0.001%` | `1.00×` | **PASS** |
| `A48_topk48` | `97,797` | `4.859552` | `0.000028` | `0.001%` | `0.83×` | **PASS** |
| `B15_full` | `65,260` | `3.058608` | `0.000017` | `0.001%` | `0.50×` | **PASS** |
| `ER768_L15` | `49,152` | `3.048401` | `0.010185` | `0.334%` | `303.45×` | **FAIL** (`leakage_energy`) |
| `ER384_L15` | `24,576` | `2.837425` | `0.205191` | `7.232%` | `6,113.75×` | **FAIL** (`leakage_energy`) |
| `D32k_entries` | `65,536` | `4.949101` | `0.060718` | `1.227%` | `1,809.10×` | **FAIL** (`leakage_energy`) |
| `D16k_entries` | `32,768` | `4.688715` | `0.265475` | `5.662%` | `7,909.91×` | **FAIL** (`leakage_energy`) |
| `D8k_entries` | `16,384` | `4.270472` | `0.534346` | `12.513%` | `15,921.02×` | **FAIL** (`leakage_energy`) |
| `D4k_entries` | `8,192` | `3.758719` | `0.795295` | `21.159%` | `23,696.11×` | **FAIL** (`leakage_energy`) |
| `D2k_entries` | `4,096` | `3.210922` | `0.978698` | `30.480%` | `29,160.66×` | **FAIL** (`leakage_energy`) |
| `D1k_entries` | `2,048` | `2.672835` | `1.067572` | `39.942%` | `31,808.69×` | **FAIL** (`leakage_energy`) |

*Policy bounds applied:* `max_frobenius_sq = 10/1`, `max_spectral_norm_sq = 10/1`, `max_leakage_energy = 1/1000`, `max_safety_dot_sq = 1/10000`.

---

## 3. Live Pure-Q Paired Dual-Metric (`accuracy` + `cut_at_32k`) Screening Audit

Exact pure-$\mathbb{Q}$ two-sided McNemar binomial tail test over the `29` JSONLs in `data/screens/` (`256` identical MATH-train IDs; note `results_table.md` lists `31` entries while `ONE_L15E32` and `ENERGY_16` are absent from `data/screens/`, and `candidate_entries.json` lists `27`). Reference arm `A64_topk64`: `150/256` (`58.6%`) accuracy, `77/256` (`30.1%`) cut at `32k` tokens.

| Candidate Arm | Acc (`n/256`) | Acc McNemar `p` vs `A64` | Cut (`n/256`) | Paired Cut `(+more/-less)` | Cut McNemar `p` vs `A64` | Dual Gate (`cut_p > 0.01`, `acc_p > 0.05`) |
|---|---:|---:|---:|---:|---:|---|
| `G000_scale` | `61/256` | `0.000000` | `234/256` | `+157 / -0` | `0.000000` | **FAIL** (acc + cut) |
| `A48_topk48` | `137/256` | `0.078941` | `112/256` | `+53 / -18` | `0.000039` | **FAIL** (`paired_cut_rate_regression`) |
| `B15_full` | `144/256` | `0.429591` | `101/256` | `+49 / -25` | `0.007084` | **FAIL** (`paired_cut_rate_regression`) |
| `B15_full_s125` | `145/256` | `0.551484` | `74/256` | `+30 / -33` | `0.801306` | **PASS** |
| `B19_full` | `111/256` | `0.000000` | `163/256` | `+95 / -9` | `0.000000` | **FAIL** (acc + cut) |
| `ER128_L15` | `129/256` | `0.004601` | `139/256` | `+74 / -12` | `0.000000` | **FAIL** (acc + cut) |
| `ER384_L15` | `142/256` | `0.322236` | `111/256` | `+60 / -26` | `0.000317` | **FAIL** (`paired_cut_rate_regression`) |
| `ER768_L15` | `142/256` | `0.301996` | `105/256` | `+50 / -22` | `0.001294` | **FAIL** (`paired_cut_rate_regression`) |
| `D1k_entries` | `142/256` | `0.291215` | `121/256` | `+59 / -15` | `0.000000` | **FAIL** (`paired_cut_rate_regression`) |
| `D2k_entries` | `139/256` | `0.143865` | `110/256` | `+54 / -21` | `0.000176` | **FAIL** (`paired_cut_rate_regression`) |
| `D4k_entries` | `149/256` | `1.000000` | `99/256` | `+43 / -21` | **`0.008147`** | **FAIL** (`paired_cut_rate_regression`) |
| `D8k_entries` | `151/256` | `1.000000` | `90/256` | `+36 / -23` | `0.117477` | **PASS** |
| `D16k_entries` | `149/256` | `1.000000` | `84/256` | `+34 / -27` | `0.442626` | **PASS** |
| `D32k_entries` | `153/256` | `0.755229` | `73/256` | `+28 / -32` | `0.698883` | **PASS** |

---

## 4. Live `car-ramp` $L_0$-vs-$L_2$ Anti-Correlation & `siliquant` Quantization-Survival Double Bind

### 4.1 `car-ramp` Checkpoints vs. `warp-drive` Stage-2 Norm Bounds (`171,664` Actor Params)

| Checkpoint | Reported Functional $L_0$ | Exact $\|\Delta\|_F^2$ | Exact $\max|\Delta W|$ | Spectral LB $\max|\Delta W|^2$ | Stage-2 Verdict |
|---|---:|---:|---:|---:|---|
| `edited_actor.npz` | `6,073` | `1,767,771.11` | `875.97` | `767,330.72` | **FAIL** (`frobenius_sq`, `spectral_norm_sq`) |
| `joint_free.npz` | `1,939` | `1,767,659.58` | `875.97` | `767,330.72` | **FAIL** (`frobenius_sq`, `spectral_norm_sq`) |
| `greedy12_astra.npz` | `12` | `87,396.95` | `200.00` | `40,000.12` | **FAIL** (`frobenius_sq`, `spectral_norm_sq`) |
| `greedy11_drop02.npz` | `11` | `87,290.09` | `200.00` | `40,000.12` | **FAIL** (`frobenius_sq`, `spectral_norm_sq`) |
| `c2_14entry.npz` | `14` | `4,862.04` | `40.25` | `1,619.99` | **FAIL** (`frobenius_sq`, `spectral_norm_sq`) |
| `c2_min_10entry.npz` | `10` | `4,432.56` | `40.25` | `1,619.99` | **FAIL** (`frobenius_sq`, `spectral_norm_sq`) |

### 4.2 Live `Moissanite-Labs/siliquant` Symmetric-INT4 (`qmax = 7`) Survival & `(3/4)·ULP` Erasure

Direct live pure-integer audit bridging `Exp 8` `down_proj` (`Qwen3.5-0.8B`) and `car-ramp` (`actor_head.W`) to `siliquant`'s `output-budget-gemma-e4b-2026-11-17` Part L (`ffn_down` `0.17%` outlier concentration), `#10` (`qwen-neuron-map`), `#12` (`ZBG128:b560s16`), `#21` (distilled shadow mapping), `#26` (`(3/4)·ULP` rounding-cell erasure), and `siliconcordat#16` (`WEFT-T3`):

| Live Tensor & Group Size | Touched Groups | Edited Weights | Survived Code Flips | Erased Edits | `< (3/4)·ULP` (`siliquant#26`) | Scale-Changed Groups (`max ratio`) | Collateral Unedited Flips (Zeroed) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Exp 8 L15 down_proj` (`G=32`) | `35,765` | `65,260` | `41,673` (`63.86%`) | `23,587` | `31,265` (`47.91%`) | `4,564` (`9.128×`) | `46,168` (`5,470` zeroed) |
| `Exp 8 L15 down_proj` (`G=128`) | `16,371` | `65,260` | `38,352` (`58.77%`) | `26,908` | `37,194` (`56.99%`) | `1,489` (`6.184×`) | `53,250` (`9,675` zeroed) |
| `Exp 8 L19 down_proj` (`G=32`) | `37,561` | `64,928` | `38,255` (`58.92%`) | `26,673` | `36,537` (`56.27%`) | `3,901` (`14.093×`) | `31,558` (`3,026` zeroed) |
| `Exp 8 L19 down_proj` (`G=128`) | `19,325` | `64,928` | `34,776` (`53.56%`) | `30,152` | `42,186` (`64.97%`) | `1,258` (`9.399×`) | `34,369` (`5,288` zeroed) |
| `car-ramp C2-min head` (`G=32`) | `1` | `2` | `2` (`100.00%`) | `0` | `0` (`0.00%`) | `1` (`114.459×`) | **`29 / 29` (`100%` zeroed)** |
| `car-ramp C2-min head` (`G=128`) | `1` | `2` | `2` (`100.00%`) | `0` | `0` (`0.00%`) | `1` (`60.979×`) | **`80 / 80` (`100%` zeroed)** |
| `car-ramp greedy12 head` (`G=32`) | `1` | `1` | `0`* (code `+7→+7`) | `1`* | `0` (`0.00%`) | `1` (`657.963×`) | **`24 / 24` (`100%` zeroed)** |
| `car-ramp greedy12 head` (`G=128`) | `1` | `1` | `1` (`100.00%`) | `0` | `0` (`0.00%`) | `1` (`436.751×`) | **`90 / 90` (`100%` zeroed)** |

*\*In `greedy12` `G=32`, the base weight `0.3040` already occupied max code `+7`; spiking it to `+200.0` leaves the code at `+7` while multiplying the group scale by `657.963×` and collapsing all `24` non-zero unedited neighbors to `0`.*

**Engineering Implication:** Naive post-hoc group quantization (`G=32`/`G=128`) destroys small rank-1 `down_proj` edits (`43.83%` erased at `G=128` across L15+L19, with `87,619` collateral flips) while turning sparse $L_0$ spikes into group-wide weight erasers. Rank-1 edits on `ffn_down` require either out-of-band adapter execution (`warp-drive` Stage 1–4) or outlier-rotated / lattice-calibrated group quantization (`siliquant#12` `ZBG128:b560s16` + Hadamard/weighing-matrix rotation + `#21` shadow mapping).

---

## 5. Self-Rewriting Governance Audit (`arXiv:2611.21885v2`) & Upstream Code Hazards

### 5.1 Eight-Gate Fail-Closed Governance Evaluation

| Session Profile | Accepted | Failed Gates |
|---|---|---|
| `arxiv_2611_21885_proto1_27b_self_edit` | `False` | `G1`, `G2`, `G4` (`8` practice repeats), `G5`, `G6`, `G7`, `G8` |
| `arxiv_2611_21885_proto1_27b_damaged_repair` | `False` | `G1`, `G2`, `G4` (`5` practice repeats), `G5` (`19` extra clips, `-22.0 pp`), `G6`, `G7`, `G8` |
| `arxiv_2611_21885_proto2_session8_dt_bias` | `False` | `G2`, `G3` (base-reset vs cumulative claim), `G4` (`8` repeats), `G6`, `G7`, `G8` |
| `warp_drive_aletheia_v2_compliant_session` | **`True`** | `[]` (`0` failures) |

### 5.2 Upstream Code Hazards Audited in `alphaXiv/weight-edit-minimization @ 768d7340`

1. **`code/st_numpy.py:55-57` (`bf16_bits`):** Performs `u32 + 0x7FFF + lsb` without checking finite input or overflow; finite `0x7F7FFFFF` overflows silently to `0x7F80` (`+inf`). Closed fail-closed in `f32_u32_to_bf16_u16_rne()`.
2. **`weight-edit-minimization.py:88`:** Fetches unpinned `refs/heads/main` tarball with no SHA-256 verification.
3. **`weight-edit-minimization.py:176` (`round3`):** Computes `log10(abs(x).max())`, producing `-inf`/`NaN` on all-zero tensors.

---

## 6. Reproducibility & Falsification Criteria

```bash
# 1. Live bit-level pure-stdlib recount & canonical certificate parity
python3 -I -B -W error wsp_self_rewriting_audit.py --live --json | diff -u wsp_self_rewriting_certificate.json -

# 2. Offline standalone fixture replay (zero external files)
python3 -I -O -B -W error wsp_self_rewriting_audit.py --standalone --json

# 3. Assertion-free R5 unit + known-answer + anti-bypass test suite
python3 -I -O -B -W error test_wsp_self_rewriting_r5.py -v
```

**Falsification Criteria:**
- Any discrepancy between `wsp_self_rewriting_audit.py --live --json` and `wsp_self_rewriting_certificate.json`, or any floating-point value found anywhere in the JSON certificate, falsifies the certificate.
- Any valid column-masked rank-1 edit producing non-zero orthogonal leakage or breaching $S^\top u = 0$, or any row-masked rank-1 edit violating $\|(I-P_u)\Delta_{\mathrm{row}}\|_F^2 = E_R(1-E_R)\|\Delta_0\|_F^2$, falsifies the pruning theorem.
- Any live `Exp 8` paired `MATH-500` control run showing `D4k_entries` matching `A64_topk64` on `cut_at_32k` without increased truncation would overturn the `MATH-train` loop-regression finding (currently falsified on the 256-problem paired screen at `p = 0.008147` and unmeasured on `MATH-500`).
