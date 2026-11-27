# Track S167 (`EXP-Q51a` + `EXP-Q51b` + `EXP-Q51c`): Backprop-Free Evolution Strategies on `SiliQuant` (`A_16 Block-H4`) & `HoloSphere-V4` (`Cayley-SU(2)`)

- **Preregistration:** `PREREG-W22-ES-SCALE` (bundled verbatim as [PREREG_W22_ES_SCALE.json](PREREG_W22_ES_SCALE.json), `SHA-256: 1f9db607ce34de1e0cbd472625d411d8c5e9857468a107a0cc24e558a733d3f6`)
- **Primary Portal:** `Portal 199` (`arXiv:2509.24372v3`, `VsonicV/es-at-scale`, Qiu et al., ICML 2026) transferred onto `SiliQuant` (`A_16 Block-H4`, `[16,11,4]` Extended Hamming SECDED, genuine `+0 B` reciprocal RMSNorm/linear folded scales, `int8` `Q=64` Al-Kashi FixedRadix EF) and `HoloSphere-V4` (`15.11M`, `54` `A_16` layers, rational Cayley-$SU(2)$ holonomy)
- **Novelty Classification:** `N1` (pure-`Init` Lean `4.36.0` exact ES operator formalization, zero-shared-code $\mathbb{Z}/\mathbb{Q}$ oracle/verifier with full disk-hashed results reconciliation and programmatic verdict evaluation, and live RTX 3080 real-weight empirical audit; all foundational ES, QES error-feedback, EGGROLL low-rank, and Hadamard orthogonal-ES mechanisms explicitly quarantined as `N0`)
- **Oracle Class:** `exact_and_sampled` (pure-$\mathbb{Z}/\mathbb{Q}$ exact certificates + live RTX 3080 evaluation on real cached `Qwen2.5-0.5B`, full `28/28`-layer `Qwen2.5-1.5B`, and `HoloSphere-V4` `15M` checkpoints; `WEIGHTS=0` inside track)
- **Energy Disclosure:** `UNCONTROLLED_LAPTOP_NOT_CLAIMED` (zero laptop watt/joule/energy claims; VRAM and wall-clock telemetry only)

---

## 1. Executive Verdict (Law 4 Honest Programmatic Kill-Gate Summary)

Every empirical gate boolean and `CONFIRMED`/`PARTIAL`/`FALSIFIED` verdict below is derived strictly programmatically from disk-hashed [s167_empirical_results.json](s167_empirical_results.json) and [s167_empirical_receipt.json](s167_empirical_receipt.json) inside both [s167_es_oracle.py](s167_es_oracle.py) and [verify_s167_es.py](verify_s167_es.py) (overall certificate tag: **`Q51a_FALSIFIED_Q51b_FALSIFIED_Q51c_CONFIRMED`**):

| Experiment / Pillar | Preregistered Gate (`PREREG-W22-ES-SCALE`) | Measured Live Outcome | Law-4 Verdict |
|---|---|---|---|
| **`EXP-Q51c` (`Aletheia-ES-Theory`, `EvolutionOperators.lean`)** | Pure-`Init` Lean `4.36.0` (`0` sorry/user-axiom/Mathlib/`Classical.choice`), `3x -I` bit-identical, RO `--live-lean PASS`, `100%` dual `mutate`, `audit-check PASS` | `21/21` live-verified theorems (`0` `[]`, `1` `[propext]`, `20` `[propext, Quot.sound]`; `0` `Classical.choice`, `0` warnings, `0` `.lake/`); `3x -I` bit-identical (`85156768...`); `366/366` dual `mutate` (`100.0%`, `0` survivors); `29/29` `BoolOp` (`100.0%`); `audit-check PASS` | **CONFIRMED** |
| **`EXP-Q51a` (`SiliQuant-ES` on real `Qwen2.5-0.5B` `n=7` seeds + full `28/28`-layer `1.5B`)** | `0` SECDED syndromes AND VRAM `< 6.5 GiB` (`6656 MiB`) AND `>= +2.0 pp` over unblocked `INT4` ES (`C2`) at matched forward-rollout budget (`p_Holm < 0.05`) | **PASS** `0` committed SECDED syndromes & `0` committed OOB across all `22,364,160` `0.5B` blocks (`3.0` & `4.5 bpw`, `7/7` seeds) AND all `81,887,232` `1.5B` blocks (`28/28` layers, `196/196` linears, `94.63M` live flips); **PASS** VRAM `2898.1 MiB` (`2.83 GiB`, `0.5B`) & `5069.9 MiB` allocated / `6060.0 MiB` reserved (`4.95` / `5.92 GiB < 6.5 GiB`, full `28/28`-layer `1.5B` with active `int8` EF & ES step); **PASS vs `C1`** (`+2.01 pp` over VsonicV BF16 `5.36%`, which suffers `25.56%` ULP stall & `5.86e-3` restore drift); **FAIL vs unconfounded `C2`** (`T1 = 7.37%` vs fair matched-layer antithetic `INT4` `qmax=7` `C2 = 8.26%`, delta **`-0.89 pp`**, `p_raw = 0.75`, `p_Holm = 0.75`; reward `+0.0066`, `p = 0.65625`) | **FALSIFIED** |
| **`EXP-Q51b` (`HoloSphere-Cayley-ES` on real `HoloSphere-V4` `15M`)** | `max_su2_shell_err < 1e-5` AND flat `O(1)` VRAM (`< 512 MiB`, spread `< 1.0 MiB` unified) across `T in {512,2048,8192}`, `R in {1,4,8}` AND `>= +10 pp` `8x` OOD `S_3`/`A_5` with `<= 0.02` WikiText-2 BPB drift | **PASS** `max_su2_shell_err = 4.17e-7` (`24x` below `1e-5` gate) vs tangent-projected unclamped `ArmA` Euclidean blowup (`8.61e12`–`6.76e18` + `Inf/NaN` & `3/3` seeds `NaN` collapse on WT2 BPB, `0.0000` OOD reward) + `0 / 936,960` `A_16` syndromes across `19.86M` live `w_omega` packed `uint64` flips; **MIXED** `O(1)` VRAM: peak `80.85 MiB < 512 MiB` (`0.17 MiB` warm spread across cells `2..9` vs Full `Chunk32-BPTT` `3/9` OOMs at `7.47 GiB` & Checkpointed `Chunk32-BPTT` peaking at `7575.4 MiB = 7.40 GiB`), **but** cold-vs-warm total spread `2.20 MiB` (`78.66 -> 80.85 MiB`) exceeds the strict unified `1.0 MiB` (`1024 KiB`) spread cap; **FAIL** `8x` OOD gain **`0.00 pp`** (`1.56% -> 1.56%`) & `16 KiB` WT2 BPB drift **`+0.0236 > 0.02`** | **FALSIFIED** |

---

## 2. Full-Text Prior-Art Quarantine (`N0` vs `N1`, Law 4)

Every prior mechanism was audited against full-text sources before making any novelty claim:

| Prior Work | Audited Mechanism | Relation / Explicit `N0` Quarantine vs Track S167 `N1` Scope |
|---|---|---|
| **Salimans et al. (2017)** (`arXiv:1703.03864`) | Seed-only NES, mirrored/antithetic sampling, fitness rank shaping | **`N0` foundational origin** of seed-reconstructed antithetic ES. |
| **Choromanski et al. (2018)** (`arXiv:1804.02395`) | Antithetic Orthogonal & Randomized Hadamard ES (`Thm 3.2`: Gaussian orthogonal MSE $\frac{d+2-N}{N}\|g\|_2^2$ vs i.i.d. $\frac{d+1}{N}\|g\|_2^2$, ratio $\frac{d+2-N}{d+1}$) | **`N0` variance-reduction theory**; S167 formalizes cleared-denominator `Block-H4` proofs (noting fixed-radius $H_4$ eliminates $\chi_d^2$ radial variance for exact `0` MSE at $N=d=4$ vs `2/5` for Gaussian orthogonal ES) in pure-`Init` Lean `4.36.0`. |
| **Malladi et al. (2023) `MeZO`** (`arXiv:2305.17333`) | In-place SPSA (`n=1` antithetic pair) with $\Theta(r_{\mathrm{eff}})$ rate | **`N0` memory-efficient ZO fine-tuning** baseline. |
| **Qiu et al. (2026) `es-at-scale`** (`arXiv:2509.24372v3`, `VsonicV/es-at-scale`) | Full-param isotropic BF16 ES ($\sigma=10^{-3}, \alpha=5\times 10^{-4}$), greedy rollouts, $z$-score norm | **`N0` source algorithm**; S167 diagnoses & empirically measures its shared cross-layer seed defect (`WorkerExtension.py:151-155`) and `25.56%` live BF16 ULP stall + `5.86e-3` restoration drift across `7` seeds. |
| **Sarkar et al. (2025/2026) `EGGROLL`** (`arXiv:2511.16652`) | Low-rank perturbations $A B^\top$ ($O(r(m+n))$ memory/compute) | **`N0` low-rank ES throughput mechanism**; does NOT address `A_16 Block-H4` `[16,11,4]` SECDED lattice projection, `+0 B` folded scales, or Cayley-$SU(2)$ Lie-manifold retraction. |
| **Xu, Schweighofer, Miikkulainen, Qiu (2026) `QES`** (`arXiv:2602.03120`, `dibbla/Quantized-Evolution-Strategies`) | Direct `INT4`/`INT8`/`W8A8` ES with accumulated error feedback and stateless seed replay | **`N0` quantized ES error-feedback paradigm**; once matched cleanly on layer set (`168` linears), antithetic pairs (`8` pairs), `int8` `Q=64` EF, and true 4-bit symmetric `qmax=7`, unblocked `INT4` (`8.26%`) is not beaten by `T1` (`7.37%`, `-0.89 pp`), falsifying the `+2.0 pp` accuracy kill gate honestly. |
| **Gan & Isola (2026) `RandOpt`** (`arXiv:2603.12228`) | *Neural Thickets* — ES task experts exist densely around pretrained weights | **`N0` precondition theory**; explains why `HoloSphere-V4 15M` (pretrained on prose/code, not `S_3`/`A_5` group words) yields `0.00 pp` `8x` OOD gain despite `4.17e-7` shell error and `19.86M` SECDED-clean `A_16` flips. |

---

## 3. Pure-`Init` Lean `4.36.0` Formalization (`EvolutionOperators.lean`)

Every theorem in [EvolutionOperators.lean](EvolutionOperators.lean) (`SHA-256: 62c416750d66def4fa3e003363c134d421d300996a234af16841580497490dd3`) is live-verified under `leanprover/lean4:v4.36.0` with **zero imports, zero Mathlib, zero `sorry`, zero user-declared axioms, zero `Classical.choice`, zero `native_decide`, zero warnings, and zero `.lake/` residue** (exact pinned Lean Init kernel axiom split: `0` `[]`, `1` `[propext]`, `20` `[propext, Quot.sound]`):

| # | Exact Lean 4 Theorem (`EvolutionOperators.*`) | Scope | Verified Axioms | Operator Role |
|---|---|---|---|---|
| 1 | `antithetic_poly4_even_cancellation_universal` | Universal (`Int`) | `[propext, Quot.sound]` | $p(\sigma e) - p(-\sigma e) = 2\sigma e(c_1 + c_3 (\sigma e)^2)$ cancels $c_0, c_2, c_4$ identically |
| 2 | `antithetic_poly4_odd_cancellation_universal` | Universal (`Int`) | `[propext, Quot.sound]` | $p(\sigma e) + p(-\sigma e) = 2(c_0 + c_2(\sigma e)^2 + c_4(\sigma e)^4)$ cancels odd terms $c_1, c_3$ |
| 3 | `antithetic_vec_linear_quadratic_cancellation_universal` | Universal (`List Int`) | `[propext, Quot.sound]` | Arbitrary-dim vector $\pm \sigma e$ cancels baseline $c_0$ + even quadratic $q$ (diff) and linear odd term (sum) |
| 4 | `antithetic_variance_strictly_beats_onesided_universal` | Universal (`Int`) | `[propext, Quot.sound]` | Nonzero baseline/even bias $b \ne 0$ adds $2b^2 > 0$ squared error to one-sided ES vs `0` for antithetic |
| 5 | `hadamard_h4_pairwise_orthogonality_and_parseval_universal` | Universal (`Int`) | `[propext, Quot.sound]` | Explicit `h4Row` 6 off-diagonal zeros + `sumSq (h4Apply a b c d) = 4 * sumSq [a, b, c, d]` |
| 6 | `randomized_block_h4_orthogonality_and_parseval_universal` | Universal (`Int`) | `[propext, Quot.sound]` | Explicit `randH4Row` ($s_i^2=1$) pairwise orthogonality & Parseval norm preservation |
| 7 | `orthogonal_h4_es_exact_recovery_zero_mse_universal` | Universal (`Int`) | `[propext, Quot.sound]` | 4-member fixed-radius orthogonal Block-$H_4$ ES recovers $\sum_{k=0}^3 \langle g, u_k\rangle u_k = 4g$ (`MSE = 0`) |
| 8 | `orthogonal_es_zero_crossterm_mse_reduction_universal` | Universal (`List`, `Nat`) | `[propext, Quot.sound]` | Vector pairwise orthogonality `dotProd u v = 0 -> vecCrossSum = 0` + exact Choromanski Thm 3.2 `N*MSE_orth + (N-1)||g||^2 = N*MSE_iid` and ratio `(d + 2 - N)/(d + 1)` |
| 9 | `shared_seed_cross_layer_correlation_obstruction_universal` | Universal (`List Int`, `Int`) | `[propext, Quot.sound]` | Shared seed gives `dotProd e e = sumSq e > 0`, whereas distinct randomized `randH4Row` rows satisfy universal `dotProd = 0` |
| 10 | `heat_kernel_goodhart_spike_attenuation_universal` | Universal (`Nat`) | `[propext, Quot.sound]` | Rank-$r$ narrow spike ($w < W$) strictly attenuates faster: $(w^2/(w^2+\sigma^2))^r < (W^2/(W^2+\sigma^2))^r$ |
| 11 | `discrete_pascal_heat_kernel_plateau_conservation_universal` | Universal (`Nat`) | `[propext, Quot.sound]` | $\sum_{j=0}^k \binom{k}{j} = 2^k$: constant plateaus suffer zero attenuation under discrete diffusion |
| 12 | `discrete_gaussian_center_spike_strict_contraction_universal` | Universal (`Nat`) | `[propext, Quot.sound]` | Central binomial spike recurrence $(2m+2)\binom{2m+2}{m+1}=4(2m+1)\binom{2m}{m}$ & strict contraction $< 4\binom{2m}{m}$ |
| 13 | `bf16_ulp_half_threshold_stall_universal` | Universal (`Int`, `Nat`) | `[propext, Quot.sound]` | Genuine nearest-grid `bf16GridRound` (no hardcoded stall branch): $2|\Delta| < \mathrm{ULP} \implies$ `iterateBf16GridStep ulp delta T (k*ulp) = k*ulp` |
| 14 | `bf16_canonical_es_ulp_stall_witness` | Universal (`Int`, `Nat`) | `[propext, Quot.sound]` | Coupled directly to Thm 13 & gradient factor `gHat`: `|gHat| <= 7` stalls for all `k, t`, while `gHat = 8` advances `(k+1)*2000` |
| 15 | `alkashi_fixedradix_ef_exact_conservation_universal` | Universal (`Int`, `List`) | `[propext, Quot.sound]` | Centered `Q = 2*h` conservation, symmetric `-h <= e_T < h`, advance `w_T >= w_0 + 1`, AND arbitrary projection/clamp/OOB-revert conservation `alkashiProjectedStep` |
| 16 | `cs08_es_vs_grpo_flop_and_hbm_crossover_universal` | Universal (`Nat`) | `[propext, Quot.sound]` | `CS-08` FLOP parity $2PL < 6PL < 8PL$ (`3x`/`4x`) & single-GPU HBM wall $3P < 2NP$ for $N \ge 2$ |
| 17 | `folded_scale_zero_byte_es_invariance_universal` | Universal (`Int`, `List`) | `[propext]` | General factoring AND exact reciprocal cancellation: `p * q = d * d -> sumScaledDot = d^2 * sumUnscaledDot` (`+0 B`) |
| 18 | `cayley_su2_unit_sphere_universal` | Universal (`Int`) | `[propext, Quot.sound]` | Exact Cayley-$SU(2)$ unit norm $(d^2 - \|w\|^2)^2 + 4d^2\|w\|^2 = (d^2 + \|w\|^2)^2$ & $D > 0$ for $d \ne 0$ |
| 19 | `cayley_su2_composition_norm_closure_universal` | Universal (`Q4`) | `[propext, Quot.sound]` | Euler four-square $\|p \cdot q\|^2 = \|p\|^2 \|q\|^2$: arbitrary Cayley ES step chains stay on $S^3$ |
| 20 | `cayley_so3_tangent_orthogonality_and_det_universal` | Universal (`Int`) | `[propext, Quot.sound]` | Exact rational $SO(3)$: $M^\top M = D^2 I_3$ (all `9` entries), $\det M = D^3$, AND $d \ne 0 \implies D^3 > 0$ |
| 21 | `euclidean_additive_es_off_manifold_obstruction_universal` | Universal (`Q4`, `Int`) | `[propext, Quot.sound]` | Naive orthogonal tangent additive Euclidean ES strictly breaks $S^3$: $\|q + \sigma v\|^2 = \|q\|^2 + \sigma^2 \|v\|^2 > \|q\|^2$ |

---

## 4. Live RTX 3080 Real-Weight Empirical Results (`s167_es_live_bench.py`)

- **Hardware & Clean Telemetry:** `NVIDIA GeForce RTX 3080 Laptop GPU` (`8192 MiB` VRAM total; `4 MiB` pre-run; total benchmark runtime `6269.6 s`; in-process `torch.cuda.memory_allocated()` after cleanup = **`0.00 MiB`**; in-process `nvidia-smi` before Python exit = `225 MiB` CUDA driver context; post-exit `nvidia-smi` idle = **`2 MiB`**; `energy_claim = UNCONTROLLED_LAPTOP_NOT_CLAIMED`).
- **Provenance Chain:** `s167_es_live_bench.py` (`75985236...`) executed cleanly after all edits; both [s167_empirical_results.json](s167_empirical_results.json) (`d6edf352...`) and [s167_empirical_receipt.json](s167_empirical_receipt.json) (`d1ee363f...`) embed `bench_script_sha256` and strictly post-date the script; `--cpu-selftest` passed `8/8` PyTorch CPU unit tests (including forced OOB scale-update fallback regression, `+0 B` reciprocal fold invariance, coordinate SECDED check, `A_16` `uint64` pack/unpack roundtrip, and pessimistic tie-rank guard).

### 4.1 `EXP-Q51a`: `SiliQuant-ES` on Real Cached `Qwen2.5-0.5B` (`n=7` Seeds) & Full `28/28`-Layer `Qwen2.5-1.5B`

- **Protocol & Matched Design:** Disjoint verifiable 3-number Countdown (`16` train / `64` held-out test, `0` overlap with `3` demos), strict AST expression verifier, matched forward-pass rollout budget (`10` iters × `16` members × `16` prompts = `2,560` rollouts/seed across **`n = 7` seeds** `{1671..1677}`, powering 2-test Holm-Bonferroni down to floor `2/64 = 0.03125 < 0.05`).
- **Unconfounded `C2` vs `T1` Comparison:** Both `C2` and `T1` optimize the **exact same `168` transformer block linear layers** (`48` RMSNorms frozen during ES, ZERO `norm_f32` shadow dict on GPU), use **`8` antithetic pairs (`16` rollouts/iter)**, the exact same **`int8` `Q=64` Al-Kashi EF**, and identical `sigma_q = alpha_q = 1.5e-3`. `C2` uses true 4-bit symmetric `INT4` (`qmax=7`, `[-7, 7]`, `4.125 bpw` incl. fp16 group scale) with isotropic Gaussian noise; `T1` applies pre-quantization `+0 B` reciprocal RMSNorm/linear scale folding (max fp32 invariance error `1.91e-6`, `0` extra bytes), `A_16 Block-H4` (`4.5 bpw`, `R^2 = 5213`, `[16,11,4]` SECDED), orthogonal Block-`H4` antithetic perturbations, and in-place `+0 B` `fp16` `s_cur` scale updates (`14.28M`–`17.74M` scale updates/seed). `C1` faithfully reproduces canonical `VsonicV/es-at-scale` (BF16, shared cross-layer seed, one-sided Gaussian, `sigma=1e-3, alpha=5e-4`, `216` layers).

| Arm / Baseline (`Qwen2.5-0.5B` & `1.5B`) | Held-Out Exact Acc (Mean) | Per-Seed Test Exact Acc (`1671..1677`) | Mean Test Reward | Peak VRAM (`MiB`) | Key Diagnostics |
|---|---:|---|---:|---:|---|
| `C0_BF16_Null` (unperturbed) | `6.25%` (`4/64`) | — | `0.3184` | `1296.8` | Reference BF16 pretrained |
| `C0_INT4_Null` (`qmax=7`, `[-7, 7]`) | `10.94%` (`7/64`) | — | `0.3466` | — | Unblocked true 4-bit symmetric `INT4` baseline |
| `C0_A16_BlockH4_3bpw_Null` (`3.0 bpw`, `R^2=651`) | `4.69%` (`3/64`) | — | `0.1782` | — | `0` syn, `0` OOB (`1,334,924` OOB scale fallbacks logged) |
| `C0_A16_BlockH4_Null` (`4.5 bpw`, `R^2=5213`, `+0 B` fold) | `7.81%` (`5/64`) | — | `0.3436` | — | `0` syn, `0` OOB (`1,373,379` OOB scale fallbacks logged) |
| `C1_VsonicV_BF16_SharedSeed` (`n=7`) | `5.36%` (`3/56`) | `6.25 / 4.69 / 4.69 / 4.69 / 6.25 / 4.69 / 6.25%` | `0.3239` | `1306.7` | **`25.56%` BF16 ULP stall**, **`5.86e-3`** restore drift |
| `C2_QES_Unblocked_INT4_EF` (`n=7`, unconfounded) | **`8.26%`** (`37/448`) | `9.38 / 6.25 / 10.94 / 3.12 / 7.81 / 12.50 / 7.81%` | `0.3324` | `2807.0` | `4.99M–8.97M` grid flips/seed; `int8` `Q=64` EF |
| `T1_SiliQuant_ES_A16_BlockH4_Folded0B_EF` (`n=7`) | `7.37%` (`33/448`) | `7.81 / 14.06 / 6.25 / 4.69 / 7.81 / 4.69 / 6.25%` | **`0.3390`** | `2898.1` | **`-0.89 pp` vs `C2`** (`p_Holm = 0.75`), **`+2.01 pp` vs `C1`** (`p_Holm = 0.25`); `0` syn / `0` committed OOB (`8.00M–9.60M` OOB candidate groups/seed safely reverted & logged); `+0 B` fold |
| **`Qwen2.5-1.5B` FULL `28/28`-Layer Packed ES Probe** | — | `196/196` block linears, **`81,887,232`** `A_16` blocks | — | **`5069.9`** alloc (`6060.0` resv) | **`4.95 GiB` alloc / `5.92 GiB` resv `< 6.5 GiB`**, resident `4164.1 MiB`, `+0 B` fold err `5.72e-6`, `94,633,548` live flips, `10,235,904` fp16 scale updates, `0` syn, `0` committed OOB (`4,413,737` OOB candidates reverted) |

- **Honest Law-4 Analysis (`EXP-Q51a = FALSIFIED`):**
  1. **Systems & Memory Gate PASS on Full `28/28`-Layer `Qwen2.5-1.5B`:** Streaming layer-by-layer quantization into `A16BlockH4ESLinear` (freeing BF16 block weights so only `y_cur: int8`, `s_cur: fp16`, `ef_res: int8` live on GPU) runs live antithetic orthogonal Block-`H4` rollouts and a full `196`-layer Al-Kashi `int8` EF + `A_16` SECDED + `+0 B` `fp16` scale update at **`5069.9 MiB` (`4.95 GiB`) peak allocated** and **`6060.0 MiB` (`5.92 GiB`) peak reserved** (`< 6.5 GiB` = `6656 MiB`), with **`0 / 81,887,232` SECDED syndromes** and **`0` committed OOB groups**.
  2. **Primary Accuracy Gate FALSIFIED (`-0.89 pp` vs unconfounded `C2`, `p_Holm = 0.75`):** Once `C2` receives fair antithetic pairing, matched `168`-layer scope, `int8` `Q=64` EF, and true 4-bit `qmax=7`, `C2` achieves `8.26%` vs `T1`'s `7.37%` (`-0.89 pp`, `p_raw = 0.75`, `p_Holm = 0.75`; continuous reward `T1 = 0.3390` vs `C2 = 0.3324`, `p = 0.65625`). Meanwhile `T1` does outperform naive `C1` VsonicV BF16 by `+2.01 pp` (`p_raw = 0.125`, `p_Holm = 0.25`). Also note that `~9.1M` candidate groups/seed hit the `R^2 <= 5213` ball-bound revert guard in `T1` (now explicitly logged rather than silently reverted).

### 4.2 `EXP-Q51b`: `HoloSphere-Cayley-ES` on Real `HoloSphere-V4` `15M` Checkpoint

#### Part B1: Like-for-Like `Chunk32` Streaming VRAM (`T in {512, 2048, 8192}`, `R in {1, 4, 8}`)
All three methods execute the exact same `chunk_size = 32` core (`chunk32_sfast_fp32_reference`) with exact `3`-byte causal `CausalDepthwiseByteConv` overlap and `input_vocab_size = 260` (note: `_feature_space_v_ring` operates per `32`-byte streaming window). `Cayley-ES` includes live active ES master/tangent perturbation buffers (`tan_bias`, `os_f32`, `m1_f32`, `m2_f32`, `bb_f32`) in VRAM.

| `(T, R)` Cell | `Cayley-ES` Peak VRAM (`MiB`) | `Cayley-ES` Max $SU(2)$ Shell Err | `Full-Graph Chunk32-BPTT` (`MiB`) | `Full-BPTT` OOM? | `Checkpointed Chunk32-BPTT` (`MiB`) | `Ckpt-BPTT` OOM? |
|---|---:|---:|---:|---|---:|---|
| `T512_R1` | `78.66` | `2.98e-07` | `520.45` | No | `122.04` | No |
| `T512_R4` | `80.79` | `3.58e-07` | `1872.88` | No | `206.44` | No |
| `T512_R8` | `80.79` | `3.58e-07` | `3673.45` | No | `318.98` | No |
| `T2048_R1` | `80.68` | `3.58e-07` | `1846.86` | No | `167.36` | No |
| `T2048_R4` | `80.81` | `3.58e-07` | `7248.54` | No | `251.76` | No |
| `T2048_R8` | `80.81` | `3.58e-07` | `7651.70` (`7.47 GiB`) | **YES (OOM)** | `7481.23` (`7.31 GiB`) | No |
| `T8192_R1` | `80.73` | `3.58e-07` | `7144.44` | No | `348.63` | No |
| `T8192_R4` | `80.85` | `3.58e-07` | `7651.63` (`7.47 GiB`) | **YES (OOM)** | `7575.37` (`7.40 GiB`) | No |
| `T8192_R8` | `80.85` | `3.58e-07` | `7651.67` (`7.47 GiB`) | **YES (OOM)** | `7481.28` (`7.31 GiB`) | No |

- **Honest `O(1)` VRAM Accounting:** `Cayley-ES` stays at `78.66–80.85 MiB` (`< 512 MiB` gate by `6.3x`; warm spread across cells `2..9` is **`0.17 MiB`** across `16x T` and `8x R`), whereas `Full-Graph Chunk32-BPTT` OOMs on `3/9` cells (`7.47 GiB`) and even `Checkpointed Chunk32-BPTT` grows linearly with `T` and reaches `7481–7575 MiB` (`7.31–7.40 GiB`, `93.7x` `Cayley-ES`) at `T2048_R8`, `T8192_R4`, and `T8192_R8`. Because cell `0` (`T512_R1 = 78.66 MiB`) ran cold before PyTorch's first backward workspace allocation (`80.68–80.85 MiB` thereafter), total measured spread is `2.20 MiB` (`2248 KiB`), which honestly fails the strict unified `< 1.0 MiB` (`1024 KiB`) spread cap without post-hoc threshold adjustment.

#### Part B2: Non-Differentiable `S_3` & `A_5` Group-Word ES (`3` Seeds, `20` Iters, `N=16`) + `16 KiB` Held-Out WikiText-2 BPB
Both arms actively perturb and commit live `int8` Al-Kashi EF + `A_16` SECDED updates directly into all `6` `blk.w_omega` packed `uint64` layers (`~6.6M` flips/seed, re-verified by `verify_all_a16_syndromes(strict=True)`). `ArmA` projects additive noise onto the exact tangent space $\langle q, v\rangle = 0$ (matching Theorem 21) and tracks strictly **unclamped** shell error with explicit `Inf`/`NaN` overflow detection and pessimistic tie-rank scoring.

| Arm (`HoloSphere-V4 15M`) | Train Exact Acc (`len 2..4, R=1`) | `8x` OOD Exact Acc (`len 16..24, R=8`) | `8x` OOD Gain (Reward) | `16 KiB` WikiText-2 BPB (Drift) | Unclamped Max $SU(2)$ Shell Err | Live `A_16` `w_omega` Flips & Syndromes |
|---|---:|---:|---:|---:|---:|---:|
| `Arm0_Base` | `15.62%` | `1.56%` (`1/64`) | `0.00 pp` (`0.0898`) | `3.2264` (`0.0000`) | `3.58e-07` | `0` / `0` of `936,960` |
| `ArmA_Euclidean_Additive_ES` | `15.62%` | `0.00%` (`0/64` ×3) | `-1.56 pp` (**`0.0000`**) | **`NaN` collapse (`3/3` seeds)** | **`8.61e12`–`6.76e18` + `Inf/NaN`** (`3/3`) | `20,043,156` flips / `0` syn |
| `ArmC_HoloSphere_Cayley_ES` | `15.62%` | `1.56%` (`0 / 1.56 / 3.12%`) | **`0.00 pp`** (`0.1536`) | `3.2500` (**`+0.0236`**) | **`4.17e-07`** (`0` overflow) | **`19,859,742` flips / `0` syn** |

- **Honest Law-4 Analysis (`EXP-Q51b = FALSIFIED`):**
  - Pre-update population training reward on `ArmC` rises `0.2065 -> 0.2590` (post-step train reward mean `0.3302`), while `ArmA`'s tangent-projected additive ES genuinely overflows `S^3` (`6.76e18` + `Inf/NaN`), collapsing to `0.0000` `8x` OOD reward (tie- and `NaN`-safe!) and `NaN` WikiText-2 BPB on `3/3` seeds.
  - `ArmC` passes the `< 1e-5` $SU(2)$ shell gate (`4.17e-7`) and maintains `0 / 936,960` SECDED syndromes across `19.86M` live `A_16` packed `uint64` flips (`2,045,684` out-of-range nibble 16-blocks safely reverted & logged), **but** fails the `+10 pp` `8x` OOD accuracy gate (`0.00 pp`), narrowly exceeds the `<= 0.02` `16 KiB` WT2 BPB drift cap (`+0.0236`), and registers a `2.20 MiB` cold-vs-warm spread vs the strict `1.0 MiB` spread cap.

---

## 5. Verification Gates & Complete Artifact Provenance

- **Zero-Shared-Code Parity & Disk Reconciliation:** [s167_es_oracle.py](s167_es_oracle.py) and [verify_s167_es.py](verify_s167_es.py) share `0` code and `0` imports, independently compute all `5` mathematical pillars over $\mathbb{Z}/\mathbb{Q}$ (zero hardcoded verifier ratios/booleans), directly read and SHA-256 hash ALL `8` input files from disk (including [PREREG_W22_ES_SCALE.json](PREREG_W22_ES_SCALE.json), [s167_es_live_bench.py](s167_es_live_bench.py), [s167_empirical_results.json](s167_empirical_results.json) parsed via `parse_float=Fraction`, and [s167_empirical_receipt.json](s167_empirical_receipt.json)), reconcile every receipt field against `s167_empirical_results.json`, and emit bit-identical [s167_es_certificate.json](s167_es_certificate.json) (`SHA-256: 8515676847d862dd2594fab607a1fac380f87d66c1e06981240891d986cc0ac0`, `payload_sha256: 9474c1b5ee619c5906b8a807f5361b31089cd4ef1caf3787eb6bbf8bd9cc645a`).
- **`3x -I` Determinism & R5 Suite:** `-I -B`, `-I -O -B`, and `-I -B -W error` bit-identical (`85156768...` ×6) across oracle, verifier, and [test_s167_es_r5.py](test_s167_es_r5.py) (`7/7 OK × 3`, `0` skips, `0` bare asserts, `0` floats; covers disk tamper of `s167_empirical_results.json` `RC=2`, `CONFIRMED`/`PARTIAL`/`FALSIFIED` programmatic verdict branches, `by stop skip`/`variable+include`/`@[default_instance]` Lean scanner refusals, `--live-lean` non-UTF-8 `RC=2 LIVE_LEAN_UTF8`, `allow_abbrev=False` `--ch` & `--json=1` single-line JSON `RC=2`, `--write <symlink>/../victim.json` + hardlink `O_NOFOLLOW` `RC=2`, and pure-Python `s167_es_live_bench.py` unit tests).
- **RW + RO `--live-lean`:** `PASS` bit-identical (`85156768...`) under both RW and `chmod -R a-w` RO copies (`21/21` Lean `4.36.0` theorems, exact version regex & `{0, 1, 20}` axiom split verified, `0` warnings, `0` `Classical.choice`, `0` `.lake/` residue).
- **Dual Uncapped `aletheia mutate --max-mutants 0`:** **`366/366 (100.0%, 0 survivors)`** (`s167_es_oracle.py`: `183/183`, `verify_s167_es.py`: `183/183`).
- **Full AST `BoolOp` (`and <-> or`) Sweep:** **`29/29 (100.0%, 0 survivors)`** (`oracle`: `12/12`, `verify`: `17/17`).
- **`aletheia audit-check --lean-audit --strict-ancestors --safety-scan --strict-unseeded`:** **`PASS`** (`13` files, `0` leaks, `0` CRLF, `0` `__pycache__`, `0` bare asserts, `0` unseeded RNG, `0` JSON errors, `0` Lean findings, `0` ancestor permission findings, `0` safety findings).

### Complete Artifact SHA-256 Manifest (`12/12` Delivered Non-Report Files; `REPORT.md` Self-Excluded)
| File | SHA-256 |
|---|---|
| `EvolutionOperators.lean` | `62c416750d66def4fa3e003363c134d421d300996a234af16841580497490dd3` |
| `lean-toolchain` | `582a63f8f9c5128067b688778c1051cf1464c3ca9d57b9325025c051c29b774a` |
| `lakefile.toml` | `4e16492fd574731d4dc2a998c56d596198dc3083565a768c28dfbd9dad60ece4` |
| `lake-manifest.json` | `5437b4c916b5f0610868c90230b06aac0abd9c11ba0639837e654b03c001c122` |
| `PREREG_W22_ES_SCALE.json` | `1f9db607ce34de1e0cbd472625d411d8c5e9857468a107a0cc24e558a733d3f6` |
| `s167_es_live_bench.py` | `759852360383f08c46eb10fcb3e8aa1c744e11bfc8b7e8a6c9427f19ae8e4439` |
| `s167_empirical_results.json` | `d6edf352c5b8ba76d04550a506812c8c22223cb117a666582fb065f65cde9e67` |
| `s167_empirical_receipt.json` | `d1ee363f31f40cf793bdc74ac56f5378196aa50f4c0b64c75b9cc6d152dd66b4` |
| `s167_es_oracle.py` | `5b32a9c5e93c847dcd0f347923b0a7c04b0fd27864159dee4e60a53311c092cc` |
| `verify_s167_es.py` | `98d57fb66b09b47ca347376f0f7ffc7fc053a5d29ba54cd6969323ceb7aad662` |
| `s167_es_certificate.json` | `8515676847d862dd2594fab607a1fac380f87d66c1e06981240891d986cc0ac0` |
| `test_s167_es_r5.py` | `ad01a19f2a3ed032c8af3684d6d138d97007e026c5c41c1c727faba5d51aab7f` |
