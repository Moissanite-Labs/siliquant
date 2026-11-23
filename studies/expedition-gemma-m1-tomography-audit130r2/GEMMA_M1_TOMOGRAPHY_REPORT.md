# Track S30 R2 (Portal 26): Live M1 Max `siliquant` Gemma 3 (`270M/1B/4B`) Tomography, Exact Rate-Distortion Convexity, $E_8$/$H_8$ Noise-Covariance Anisotropy & `run.py` `hEBG128` + BF16 Unlock

- **Track ID:** `TRACK-S30-SILIQUANT-GEMMA-M1-TOMOGRAPHY`
- **Schema:** `aletheia.gemma_m1_tomography.certificate.v1`
- **Pinned Upstream Snapshot:** `Moissanite-Labs/siliquant` branch `exp/gemma-ball-results` at commit `dc10d4bbca` (untouched read-only `N0` upstream copies under `pinned_inputs/`)
- **Arithmetic & Dependencies (Certificate Suite):** Pure Python 3 standard library over exact $\mathbb{Z}/\mathbb{Q}$ (`int`, `fractions.Fraction`); `0` third-party libraries, `0` AST float literals, `0` `float()` calls, `0` `.isdigit()` calls, `0` bare `assert` statements, `0` RNG across all `3` Track S30 certificate scripts (`gemma_m1_tomography_oracle.py`, `gemma_m1_tomography_audit.py`, `test_gemma_m1_tomography_r5.py`) and `0` JSON floats in `gemma_m1_tomography_certificate.json`. *(The `4` read-only files under `pinned_inputs/` are unmodified `N0` upstream receipts/scripts.)*
- **Certificate Digests (R2):**
  - `gemma_m1_tomography_certificate.json` file SHA-256: `39190ddaf55a63ef6f4fe8787895dfb4c727cde1fda75fa007cffc386e088b0f`
  - `payload_sha256`: `31bdac20ad99455b9d9a685e0714b2f9d199384938deac65e7beb50caab81943`
  - `certificate_sha256`: `ac95c7a8ef8e3cba75a9cbd606cd0317484a4899cc708e20dea59db5c80461f8`

---

## 1. Epistemic Taxonomy (Law-4 Honesty: `N0` vs. `N1`)

- **Overall Certificate Novelty Tier:** **`N1`** (exact-$\mathbb{Q}$ computer-certified synthesis, operator theorem, rate-distortion enclosure, band-based fallback decoder, falsifiable empirical prediction, and static AST patch verification over live M1 Max receipts).
- **`N0` (Pinned Upstream Measurements, Known Lattice/Information-Theory Facts & Prior Art — Explicitly Quarantined):**
  - Raw `102` M1 Max evaluation rows (`3 models × 34 rows`) in `pinned_inputs/{gemma-3-270m-it,gemma-3-1b-it,gemma-3-4b-it}.jsonl`, the benchmark script `pinned_inputs/run.py`, the preregistered `P1..P10` protocol (`PR #47`), and the upstream linear iso-KL script (`PR #57`, `analyze.py`).
  - Classical $E_8$ lattice definition ($240$ minimal norm-$2$ roots, packing radius$^2 = 1/2$), the Conway–Sloane $E_8$ coset nearest-point decoder, the Sylvester/Walsh Hadamard matrix $H_8$ ($H_8^\top = H_8$, $H_8^2 = 8 I_8$), and the asymptotic $20 \log_{10}(2) \approx 6.0206\,\text{dB/bit}$ Gaussian rate-distortion slope.
  - Prior internal `Track S20b` (`siliquant#42`) framework connecting lattice quantization noise covariance to Hessian-weighted KL divergence.
- **`N1` (Exact-$\mathbb{Q}$ Certified Contributions in Track S30 R2):**
  1. Complete float-free $\mathbb{Q}$ scoring of all `102` raw JSONL rows against `PR #47` `P1..P10` verbatim (including tie-exact Pearson-of-midranks Spearman $\rho^2$), certifying `4B` stock `other_tensors_that_differ = ["token_embd.weight"]` provenance asymmetry (inferentially linked to `RUN-BUG-02` BF16-to-F16 casting) and `P5`'s monotone scale decay on WikiText vs. collapse to `8.46%` on `4B` Python.
  2. Float-free pure-$\mathbb{Q}$ `ln_bounds_exact` and `exp_neg_bounds_exact` rate-distortion convexity proof across all `12/12` `(b496, b560, b624)` triples and two-sided exact Jensen iso-KL enclosure (`width <= 1e-15` Wiki, `width <= 4e-15` Python, both `< 1e-12`) upgrading `PR #57`'s linear upper bound (`9.35%` Wiki / `9.05%` Py) to exact log-linear matched-KL savings (`10.04%` Wiki / `10.00%` Py on `4B`).
  3. Deterministic pure-$\mathbb{Q}$ per-kind `weight_err_db_by_kind` band classifier decoding realized `file:Q4_K_M` tensor types (`270M`/`1B`: `6/7` kinds promoted to `Q8_0`/`Q5`/`Q6_K`, only `attn_output=Q4_K`; `4B`: zero-fallback `5` `Q4_K` + `2` `Q6_K`).
  4. Exact pure-$\mathbb{Z}/\mathbb{Q}$ $E_8$-vs-$Q_H E_8$ ($Q_H = H_8 / \sqrt{8}$, $\det Q_H = \pm 1$) noise-covariance anisotropy operator theorem resolving the `EBG128` vs. `HBG128` MSE-blindness / PPL-vs-KL sign-flip paradox — including $Q_H \notin \mathrm{Aut}(E_8)$, isotropic trace invariance, exact `8×` opposite-sign KL flip at identical $\ell_2$ weight error `1/4`, and isometric single-vector witnesses over $\mathbb{Q}(\sqrt{2})$ — plus exact pure-$\mathbb{Q}$ `PRED-M1-01` forecast for `4B hEBG128:b560s16`.
  5. Exact Python AST audit of `pinned_inputs/run.py` dynamically extracting the `.startswith(<prefix>)` literals at lines `[967, 1035]` (`["hZBG"]`, proving `5` missed weighted ball arms live from the AST) and detecting `RUN-BUG-02` at line `[733]` (`^token_embd\.weight$=f16`), paired with verified replacement functions (`needs_real_imatrix_fixed`, `quantize_tensor_type_flags_fixed`).

---

## 2. Five Certified Pure-$\mathbb{Z}/\mathbb{Q}$ Pillars

### Pillar 1 — Raw-Receipt Provenance & Exact-$\mathbb{Q}$ `PR #47` `P1..P10` Scoring
All `102` rows (`3 × 34`: `17` arms × `2` texts `wikitext` and `python`) are parsed strictly from canonical ASCII JSON tokens into exact `Fraction` values (`0` float intermediates, strictly positive `kl_mean`/`kl_mean_se`/`bits_per_weight`/`ppl`/`ppl_base`) after verifying exact SHA-256 hashes of all `3` pinned JSONL files. Two-sided $2$-SE decisions use exact rational squared-$z$ tests (`z_sq >= 4`, with signed `z_signed_2dp` formatted via pure-integer `isqrt`, emitting `"0.00"` rather than `"-0.00"` on zero truncation).

| Prereg | Comparison / Criterion | `gemma-3-270m-it` | `gemma-3-1b-it` | `gemma-3-4b-it` |
|---|---|---|---|---|
| **P1** | `hZBG128:b560s16 < file:Q4_K_M-imat` ($\ge 2$ SE) | Wiki **HOLD** (`+5.87`), Py **FAIL** (`-1.38`) | Wiki **FAIL** (`-2.97`), Py **FAIL** (`-1.23`) *(confounded: `5.4817 bpw`)* | Wiki **HOLD** (`+3.78`), Py **HOLD** (`+2.31`) *(`4.5040` vs `4.8151 bpw`)* |
| **P2** | `ZBG128:b560s16 > file:Q4_K_M` (Wiki) | **HOLD** (`+9.50` Wiki, `+2.53` Py) | **HOLD** (`+5.66` Wiki, `+3.33` Py) | **FAIL_REVERSED** (`-6.11` Wiki, `-3.56` Py) |
| **P3** | `hZBG < ZBG` on both texts ($\ge 2$ SE) | **HOLD** (`+31.44` / `+7.14`) | **HOLD** (`+13.04` / `+3.73`) | **HOLD** (`+8.39` / `+2.15`) |
| **P4** | `hZBG+K10 < hZBG` on both texts ($\ge 2$ SE) | **FAIL** (`+0.68` / `+0.34`) | **FAIL** (`+0.18` / `-0.02`) | **FAIL** (`0.00` / `-0.07`) |
| **P5** | `hZBG~rand` retains $\ge 70\%$ of real-imatrix KL gain (Wiki) | **HOLD** (`90.78%` Wiki; `96.59%` Py) | **HOLD** (`85.24%` Wiki; `99.72%` Py) | **HOLD** (`78.73%` Wiki; **`8.46%` Py** vs `25.90%` real-imat cut) |
| **P6** | `ZBG+K10 < ZBG` ($\ge 2$ SE or $\ge 5\%$ cut, Wiki) | **FAIL** (`+0.44`, `1.07%`) | **FAIL** (`+0.37`, `1.47%`) | **FAIL** (`+0.59`, `2.86%`) |
| **P7** | `gemma-4-E4B-it` bridge | `NOT_APPLICABLE_GEMMA3` | `NOT_APPLICABLE_GEMMA3` | `NOT_APPLICABLE_GEMMA3` |
| **P8** | `hZBG128:b624s16 < hZBG128:b560s16` ($\ge 2$ SE) & rank #1 | **HOLD** (`+18.71` / `+5.22`, #1) | **HOLD** (`+10.95` / `+4.63`, #1) | **HOLD** (`+9.28` / `+3.01`, #1) |
| **P9** | KL rises at ~4.0 bpw (`ZBG`, `hZBG`, `ZB128@R:16`, `Q4_0`) & `hZBG:b496 < Q4_0` ($\ge 2$ SE) | **HOLD** (`+40.87` / `+11.94`) | **HOLD** (`+22.42` / `+7.26`) | **HOLD** (`+11.77` / `+2.64`) |
| **P10** | Tie-exact Spearman $\rho \ge 9/10$ ($100\operatorname{cov}^2 \ge 81 v_x v_y$) & max top-1 inversion $\le 0.2$ pp | Wiki **FULL HOLD** (`135/136=0.9926`, `0.108` pp); Py $\rho=199/204$, inv `0.784` pp | Wiki $\rho=401/408$, inv `0.206` pp; Py $\rho=33/34$, inv `1.289` pp | Wiki $\rho=203/204=0.9950$, inv `0.539` pp; Py **tied** $\rho^2=49923/55352$ (`0.9496`), inv `0.448` pp |

- **Inferential BF16 Provenance Attribution:** On `270M` and `1B`, all three stock `llama-quantize` arms (`file:Q4_0`, `file:Q4_K_M`, `file:Q4_K_M-imat`) report `other_tensors_that_differ = []`. On `4B`, all three stock arms report `other_tensors_that_differ = ["token_embd.weight"]`; while the raw JSONL rows record the differing tensor name rather than the source dtype directly, pairing this receipt with `RUN-BUG-02` (`^token_embd\.weight$=f16` at `pinned_inputs/run.py:733`) inferentially attributes the mutation to F16 casting of `4B`'s BF16 `token_embd.weight`.
- **Note on Exact Truncation Display:** All decimal strings in the certificate (`*_2dp`, `*_3dp`, `*_4dp`, `*_6dp`) are deterministic toward-zero truncations of certified exact `Fraction` fields (`format_decimal_trunc` or pure-integer `isqrt`), while every inequality/verdict is decided strictly on exact $\mathbb{Z}/\mathbb{Q}$ arithmetic.

### Pillar 2 — Exact Log-Convex Rate-Distortion Theorem & Two-Sided Pure-$\mathbb{Q}$ Iso-KL Enclosure
Using pure-$\mathbb{Q}$ `artanh` series bounds (`ln_bounds_exact`) and alternating Taylor bounds (`exp_neg_bounds_exact`):
- **Strict Convexity (`12/12` triples):** Every `(model, family ∈ {ZBG, hZBG}, text ∈ {wikitext, python})` triple across `(b496, b560, b624)` satisfies strict convexity of $D(b)$: linear interpolation at `b560` overestimates measured KL by **`18.18%` to `27.01%`** on WikiText and **`18.32%` to `50.08%`** on Python.
- **Geometric Interpolation Accuracy:** Log-linear (geometric) interpolation beats linear on **`12/12`** triples and stays within `<= 2.5%` on **`6/6`** WikiText triples (exact magnitude range **`0.186%–2.232%`**, 2dp truncated: `4B ZBG -0.18%`, `1B ZBG +0.88%`, `1B hZBG +0.95%`, `270M ZBG +1.21%`, `4B hZBG -1.52%`, `270M hZBG -2.23%`).
- **Shannon Slope (`-6.02 dB/bit`):** Pure-$\mathbb{Q}$ bounds bracket $20\log_{10}(2) \in [6.0205, 6.0206]$; empirical `ZBG128` weight-error slopes across `(b496, b560, b624)` are `6.0404` (`270M`), `6.0408` (`1B`), and `6.0408` (`4B`) dB/bit — all `3/3` within `0.05` dB/bit.
- **Exact Two-Sided Jensen Iso-KL Bracket on `gemma-3-4b-it` (Upgrading `PR #57` `analyze.py`):** Matching `file:Q4_K_M-imat` (`4.8151 bpw`) with `hZBG128`:
  - **WikiText (`target_kl = 88353/1000000`):** Linear secant upper bound $b_{\text{lin}} = 4.3644$ (`9.35%` savings) vs. exact log-linear enclosure $b_{\log} \in [4.3314, 4.3314]$ (**`width <= 1e-15 < 1e-12`**, **`10.04%`** matched-KL savings).
  - **Python (`target_kl = 10423/250000`):** Linear secant upper bound $b_{\text{lin}} = 4.3789$ (`9.05%` savings) vs. exact log-linear enclosure $b_{\log} \in [4.3335, 4.3335]$ (**`width <= 4e-15 < 1e-12`**, **`10.00%`** matched-KL savings).

### Pillar 3 — Deterministic Pure-$\mathbb{Q}$ Per-Kind `Q4_K_M` Band Classifier
A deterministic pure-$\mathbb{Q}$ band classifier (`classify_llama_quant_kind`) partitions `weight_err_db_by_kind` at exact rational thresholds (`-33.0`, `-27.8`, `-25.0`, `-21.5` dB) along the `~6 dB/bit` `llama.cpp` ladder to infer realized tensor quantization types across all `7` weight kinds (`attn_q`, `attn_k`, `attn_v`, `attn_output`, `ffn_gate`, `ffn_up`, `ffn_down`):
- **`270M` (`5.4338 bpw`) & `1B` (`5.4817 bpw`) `file:Q4_K_M` and `file:Q4_K_M-imat`:** `attn_v = Q8_0`, `{attn_q, attn_k, ffn_gate, ffn_up} = Q5_0_or_Q5_K` (`4` kinds), `ffn_down = Q6_K`, and **only `attn_output = Q4_K`** (`1` kind) — accounting for the `~0.65 bpw` inflation that confounded `1B` P1.
- **`4B` (`4.8151 bpw`) `file:Q4_K_M` and `file:Q4_K_M-imat`:** Zero fallbacks — **`5` `Q4_K`** (`attn_q, attn_k, attn_output, ffn_gate, ffn_up`) + **`2` `Q6_K`** (`attn_v, ffn_down`).
- **Negative Control (`file:Q4_0`, `4.5000 bpw`):** All `7/7` kinds classify cleanly to `Q4_0_or_lower` across all `3` models.

### Pillar 4 — Exact $E_8$-vs-$Q_H E_8$ Noise-Covariance Anisotropy Operator Theorem & `PRED-M1-01`
- **Empirical MSE-Blindness / PPL-vs-KL Paradox:** Across all `21` `(model × 7 weight kinds)` pairs, `max |EBG128 - HBG128| = 666057506077/500000000000000` ($\le 0.00134\,\text{dB}$), while `E8` beats `Z8` (`min(ZBG128 - EBG128) = 32719712394503/50000000000000`, $\ge 0.6543\,\text{dB}$) on `21/21` pairs. Yet `HBG128` achieves strictly lower WikiText PPL on `3/3` models (`55.967 < 65.187`, `26.214 < 29.422` [below `27.165` base], `16.093 < 16.628`), while KL flips from favoring `HBG` on `270M` to favoring `EBG` on `4B` (`+1.26` SE Wiki, `+0.21` SE Py).
- **Exact Pure-$\mathbb{Z}/\mathbb{Q}$ Operator Theorem (Law-5 Bridge to `Track S20b` / `siliquant#42`):**
  1. $H_8^\top = H_8$ and $H_8^2 = 8 I_8$; all `240` minimal $E_8$ roots have exact squared norm `2` and are fixed by the pure-$\mathbb{Q}$ Conway–Sloane decoder `nearest_e8_exact`.
  2. $Q_H = H_8 / \sqrt{8} \notin \mathrm{Aut}(E_8)$: for root $r = (1, 1, 0, \dots, 0)$, $4(Q_H r)_0^2 = 2$, which is not an integer square, so $2(Q_H r)_0 \notin \mathbb{Z}$.
  3. **Isotropic Trace Invariance:** Orthogonality $H_8^\top H_8 = 8 I_8$ implies the universal algebraic identity $\operatorname{tr}(\frac{1}{8} H_8 \Sigma H_8^\top) = \operatorname{tr}(\frac{1}{8} H_8^\top H_8 \Sigma) = \operatorname{tr}(\Sigma)$ for all $8 \times 8$ matrices $\Sigma$ (additionally verified explicitly on a dense generic rational symmetric instance `sigma_gen`) — explaining why isotropic weight MSE (`weight_err_db_by_kind`) is mathematically blind to orthogonal $Q_H$ rotation.
  4. **Exact `8×` Opposite-Sign KL Flip at Identical $\ell_2$ Weight Error:** For axis-aligned error covariance $\Sigma_e = \frac{1}{4} e_1 e_1^\top$ vs. isometric Hadamard-rotated $\Sigma_{e,H} = Q_H \Sigma_e Q_H^\top = \frac{1}{8} H_8 \Sigma_e H_8^\top = \frac{1}{32}\mathbf{1}\mathbf{1}^\top$, both have identical weight MSE $\operatorname{tr}(\Sigma_e) = \operatorname{tr}(\Sigma_{e,H}) = 1/4$, yet:
     - Under axis-aligned activation Hessian $H_{\mathrm{act}}^{(1)} = e_1 e_1^\top$: $\operatorname{tr}(H_{\mathrm{act}}^{(1)}\Sigma_e) = 1/4$ vs. $\operatorname{tr}(H_{\mathrm{act}}^{(1)}\Sigma_{e,H}) = 1/32$ (**$Q_H$ wins by exact `8×`**).
     - Under coherent DC activation Hessian $H_{\mathrm{act}}^{(2)} = \frac{1}{8}\mathbf{1}\mathbf{1}^\top$: $\operatorname{tr}(H_{\mathrm{act}}^{(2)}\Sigma_e) = 1/32$ vs. $\operatorname{tr}(H_{\mathrm{act}}^{(2)}\Sigma_{e,H}) = 1/4$ (**plain $E_8$ wins by exact `8×`**).
  5. **Exact Isometric Single-Vector Witnesses Over $\mathbb{Q}(\sqrt{2})$ ($\det Q_H = \pm 1$, $R_{\mathrm{pack}}(E_8)^2 = 1/2$):**
     - **Deep hole $e_1 = (1, 0^7)$:** $d^2(e_1, E_8) = 1$, while $Q_H e_1 = \frac{1}{4}\sqrt{2}(1^8)$ has unique nearest $E_8$ point $p_1 = (\frac{1}{2})^8$ with exact $d^2(Q_H e_1, E_8) = 3 - 2\sqrt{2} < 1/5 < 1/2 = R_{\mathrm{pack}}(E_8)^2$ (certified purely in $\mathbb{Q}$ via $8 > (14/5)^2 = 196/25$; **$Q_H E_8$ cuts squared error by $> 5\times$**).
     - **Root $r = e_1 + e_2 = (1, 1, 0^6)$:** $d^2(r, E_8) = 0$, while $Q_H r = \frac{1}{2}\sqrt{2}(1,0,1,0,1,0,1,0)$ has unique nearest $E_8$ point $p_2 = (1,0,1,0,1,0,1,0)$ with exact $d^2(Q_H r, E_8) = 6 - 4\sqrt{2} \in (1/3, 1/2)$ (certified purely in $\mathbb{Q}$ via $(11/2)^2 = 121/4 < 32 < (17/3)^2 = 289/9$; **plain $E_8$ wins strictly with `0` error vs. `> 1/3`**).
- **Falsifiable Prediction `PRED-M1-01` (`gemma-3-4b-it`, `hEBG128:b560s16` at `4.5040 bpw`):**
  - Theoretical $10^{-0.065} \in [0.860993, 0.860994]$ transfer from `hZBG128:b560s16`: **`0.061685`** Wiki KL (`12.60%` matched-KL savings vs. `file:Q4_K_M-imat` at `4.2081 bpw`) and **`0.026777`** Python KL.
  - Empirical `EBG/ZBG` transfer on `4B` WikiText: **`0.059990`** Wiki KL (**`13.08%`** matched-KL savings, strictly above `12%`).

### Pillar 5 — Static AST Audit & Verified Patch Spec for `pinned_inputs/run.py`
Parsing `pinned_inputs/run.py` (`afd7ef8a...`) via Python's `ast` module certifies two live bugs blocking `hEBG128` and casting `4B` BF16 embeddings to F16:
- **`RUN-BUG-01` (Hardcoded `startswith("hZBG")` at lines `[967, 1035]`):** Although `WEIGHTED_BALL_RE` already supports all `6` weighted ball families, walking `run_sweep`'s AST dynamically extracts the literal prefix set `["hZBG"]` at lines `[967, 1035]` and evaluates it live against all `8` real-imatrix arms, proving `5` arms are missed (`hEBG128:b560s16`, `hHBG128:b560s16`, `iZBG128:b560s16`, `iEBG128:b560s16`, `iHBG128:b560s16`). Verified replacement `needs_real_imatrix_fixed` handles all `6` weighted prefixes while rejecting `~rand` and unweighted arms.
- **`RUN-BUG-02` (Hardcoded `^token_embd\.weight$=f16` at line `[733]`):** Forces F16 casting of `token_embd.weight` even on BF16 models like `gemma-3-4b-it`. Verified replacement `quantize_tensor_type_flags_fixed` preserves `BF16`/`F16`/`F32` base dtypes and supports Gemma 4 E4B extra tensors.

---

## 3. Audit Architecture & `--check` Verification Scope

- **`gemma_m1_tomography_audit.py --check <cert.json>`** enforces fail-closed Zone-B I/O (per-component `O_RDONLY|O_DIRECTORY|O_NOFOLLOW` traversal, leaf `O_NOFOLLOW|O_NONBLOCK` + `S_ISREG` + capped bounded-chunk read `max_bytes <= 64 MiB`, CRLF/trailing-LF/UTF-8/canonical-ASCII-JSON rejection), then performs:
  1. **Full pure-$\mathbb{Q}$ deterministic replay** (`validate_certificate_payload(..., full_replay=True)`), recomputing the entire certificate from `pinned_inputs/` and checking exact byte-level canonical equality plus `payload_sha256` and `certificate_sha256` (with top-level payload key/metadata/pillar-dict checks enforced even under `full_replay=False`);
  2. **Schema-safe secondary algebraic invariant cross-check** (`secondary_algebraic_audit`), failing closed with `GemmaM1Error` on missing/wrong-type keys and re-validating exact `z_sq >= 4` / `gain_kl` verdict consistency across P1/P2/P3/P4/P6/P8/P9, P5 monotone decay, P10 `spearman_rho_sq >= 81/100` & inversion counts, P2 `12/12` triples & `< 1e-12` Jensen brackets, P3 per-kind fallback counts (`1/4/1/1`, `0/0/2/5`, `0/0/0/0`), P4 `0.002 dB` / `0.65 dB` / `240` roots / `8×` flip / isometric $\mathbb{Q}(\sqrt{2})$ witnesses, and P5 AST line sites `[967, 1035]` / `[733]` & `5` missed arms.
- **Honest Scope Note:** `gemma_m1_tomography_audit.py` imports exact parsing, canonicalization, and oracle replay helpers from `gemma_m1_tomography_oracle.py`; `--check` is therefore a **full deterministic pure-$\mathbb{Q}$ replay + schema-safe secondary algebraic invariant check**, not a zero-shared-code independent re-implementation.
- **Zone-B Atomic Write Hardening (`--write`):** Enforces `type(data) is bytes`, refuses hardlinked targets (`st_nlink > 1`), creates an `O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW|O_NONBLOCK` temp file in the anchored parent `dir_fd`, calls `os.fsync(tmp_fd)`, replaces atomically via `os.replace(..., src_dir_fd=dfd, dst_dir_fd=dfd)`, and calls `os.fsync(dfd)`; rejects lone UTF-16 surrogates, deleted CWD (`CWD_UNAVAILABLE`, `rc=2`), non-canonical/non-ASCII rationals, and flag-like path arguments (`--write --json`, `--write=--json`, `--check=--json` -> `rc=2`, `0` files created).

---

## 4. Verification Receipts (R2)

| Gate | Scope & Command Class | Result |
|---|---|---|
| **5-Mode Parity** | `-I -B`, `-I -O -B`, `-I -B -W error`, `-I -O -B -W error`, `PYTHONHASHSEED=42 -B` | **PASS `5/5`**: standalone `39190dda...` ×5 (bit-identical to disk), `--check --json` `df420d62...` ×5, `test_r5` `13/13 OK` ×5, zero-arg `OK` |
| **Default-CLI `aletheia v2.0.0 mutate`** | `--max-mutants 0 --work-root . --add-cwd --timeout 45` on both `oracle.py` & `audit.py` | **PASS `217/217` (`100.0%`, `survived=0`, `truncated=False`)**: `oracle.py` `121/121` + `audit.py` `96/96` |
| **Exhaustive AST `BoolOp` (`SWAP+SUBCOND_DEL`)** | All `49` `ast.BoolOp` nodes (`timeout=30`) across `oracle.py` & `audit.py` | **PASS `149/149` (`100.0%`, `surv=[]`)**: `oracle.py` `112/112` (`37` nodes) + `audit.py` `37/37` (`12` nodes) |
| **Zone-B Adversarial Fuzzer (`S19c`)** | CLI + In-Memory (`full_replay=True`) + In-Memory (`full_replay=False` + `secondary_algebraic_audit`) | **PASS `41/41` (`100.0%`)**: `23/23` CLI + `18/18` full-replay + `18/18` non-replay+secondary (`bypasses=0, crashes=0`) |
| **Live Concurrent Races & `procfs`** | `renameat2(RENAME_EXCHANGE)` + `os.link`/`os.unlink` + `/proc/kallsyms` | **PASS**: `15,265` swaps (`700` reads `0` leaks, `700` writes `0` clobbers/`0` temps); `957` hardlink flips (`600` writes `0` clobbers/`0` temps); `/proc/kallsyms` -> `FILE_TOO_LARGE` |
| **Static AST & `aletheia audit-check`** | `audit-check . --strict-ancestors --safety-scan --scripts gemma_m1_tomography_audit.py test_gemma_m1_tomography_r5.py` | **PASS** (`9` files, `2/2` R4, `0` findings, `0` AST floats/`float()`/bare `assert`s/`.isdigit()` across `3/3` `.py`, `0` JSON floats, `PYC=0`, `CRLF=0`) |

---

## 5. Frozen R2 File Manifest

| File | Lines | Bytes | SHA-256 |
|---|---:|---:|---|
| `gemma_m1_tomography_oracle.py` | 1,753 | 81,543 | `a791e05d12fcaf448e8b25099e40b5613fb5062540f2e129d054c59804974ada` |
| `gemma_m1_tomography_audit.py` | 592 | 27,877 | `ca1b9dfc70038ded6d19de4b5c048d283ed8b4e8d65d3bea249924df59e210af` |
| `test_gemma_m1_tomography_r5.py` | 930 | 63,096 | `27beb5df48c3270f4784ec699ef6e83cf424035b5e027de61adeedcf32ad9eb8` |
| `gemma_m1_tomography_certificate.json` | 1 | 25,769 | `39190ddaf55a63ef6f4fe8787895dfb4c727cde1fda75fa007cffc386e088b0f` |
| `pinned_inputs/gemma-3-270m-it.jsonl` | 34 | 47,886 | `dafe50f93262be5628b97a83b915a0f65aa4eaa52f09b1b9bfd36381c4b34a83` |
| `pinned_inputs/gemma-3-1b-it.jsonl` | 34 | 47,819 | `040717b2d9c24d47e64fcc3b41156a375d651f180cd7b9f558ec603c60f7733f` |
| `pinned_inputs/gemma-3-4b-it.jsonl` | 34 | 48,063 | `5895d850aff01a154a3b57e1c47468c90121e34316fa6e850ae866f6ec4dc23e` |
| `pinned_inputs/run.py` | 1,104 | 48,297 | `afd7ef8ae34d7986cef9a10825518a61174a44feb325ee0c1684b8b6cded159c` |
