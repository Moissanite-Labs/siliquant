# Track S79: Exact Operator & Roofline Laws for 2DSD, Dense-and-Sparse CSR vs. `siliquant`, and Fast-Multipole Attention

**Project ID:** `WAVE4B-S79-2DSD-MULTIPOLE-DENSE-SPARSE`  
**Novelty Tier:** `N1` (exact algebraic formalization, cross-family synthesis, obstructions, and formula transposition; all foundational algorithms quarantined as `N0`)  
**Oracle Class:** `exact` (pure Python 3 stdlib over $\mathbb{Z}/\mathbb{Q}$, zero IEEE-754 floats on trust paths, zero bare `assert`s, zero mathematical RNG)  
**Lean 4 Kernel:** `leanprover/lean4:v4.36.0` (pure `Init`, zero Mathlib, `18/18` theorems live-verified with `0` stderr/warnings; `14/18` strictly `0` axioms `[]`, `4/18` `Int` ring theorems `[propext]` only; zero `Classical.choice`, `Quot.sound`, `sorryAx`, or `Lean.ofReduceBool`)  
**Live GPU Measured in This Bundle:** `False` (pure formal/exact analytical track; empirical numbers are verbatim pinned anchors from primary sources)

---

## 1. Steelman of the Three Wave-4B Systems (Law 2)

1. **`SqueezeAILab/2DSD` (`commit 431aade2`):** Looped language models (`Ouro-1.4B/2.6B`, `Nanbeige4.2-3B`, `Dory1.0-4B`, plus `minseo25/2dsd-inkling-2-mini-varloop-dspark-drafter` at `541,880,065` params) unlock a second speculative axis (*loop depth* $r \in \{1,\dots,R\}$) orthogonal to token-horizon block drafting (DFlash). By verifying a `main -> frozen -> fork` cascade against recorded early-loop proposals (Paper Eq. 5) and reading uncompressed last-loop KV caches outside a lagged per-loop window $W = 64$ (Paper Eq. 6), 2DSD reaches `5.73x` (`Ouro-1.4B`), `7.64x` (`Ouro-2.6B`), and `4.50x` (`Dory1.0-4B`) greedy speedups over native AR (`1.20x` to `1.63x` over 1D DFlash and up to `3.03x` over 1D LoopSpec).
2. **`SqueezeLLM` / `KVQuant` / `Portal 77` Dense-and-Sparse Quantization:** Hessian/Fisher sensitivity-weighted non-uniform quantization paired with a thin sparse CSR outlier table (`0.45%` @ `32` b/entry = `18/125 = 0.1440` bpw in SqueezeLLM; `0.057%` @ `48` b/entry = `171/6250 = 0.0274` bpw in Portal 77) removes the heavy-tail dynamic-range distortion of extreme weights/KV entries at tiny bitrate overhead.
3. **`MultipoleAttention` (`commit db65e51d`):** Replaces $O(N)$ far-field key-value inner products with hierarchical cluster centroids (`BLOCK = 8192` tokens, Triton `_fwd_centroid_kernel_attn`: `p_tmp = exp2(q·k_bar) * nkeys`, `acc += tl.dot(p_tmp, v_bar)`), reducing long-context attention memory traffic by the cluster compression ratio whenever distant keys are tightly clustered.

---

## 2. Exact Operator Laws & Obstructions (Law 3)

### Pillar 1: 2DSD Cascade Verification, Speedup Dominance, Eq. 6 KV Sharing & `W4B-OBS-1`

- **Lossless Cascade Mass Conservation (Paper Eq. 5):** For any finite cascade of recorded proposal distributions $Q_1, \dots, Q_m$ tested sequentially in fixed order (`main`, `frozen`, `fork`) starting from target mass $P_0 = p_R(\cdot \mid x)$, with acceptance $\min(P_{i-1}(v), Q_i(v))$ and residual $P_i(v) = P_{i-1}(v) - \min(P_{i-1}(v), Q_i(v))$:
  $$\sum_{i=1}^{m} \min(P_{i-1}(v), Q_i(v)) + P_m(v) = P_0(v) \quad \forall v.$$
  In our 3-stage exact witness (`P_0 = [60, 30, 10]`, `Q_main = [40, 40, 20]`, `Q_frozen = [10, 50, 40]`, `Q_fork = [70, 20, 10]`), cumulative accepted mass advances strictly `80 -> 90 -> 100` (`final_residual = 0`), while any permutation `[Q_main, Q_frozen]` vs. `[Q_frozen, Q_main]` yields exact marginal identity `P_0`.
- **2D Speedup Dominance Over 1D DFlash & 1D LoopSpec:** With $R$ loops, block yield $\tau > 1$, and per-wave loop cost $C_{2\mathrm{D}} < C_{1\mathrm{D}}$, $\mathrm{Speedup}_{2\mathrm{D}} = R\tau / C_{2\mathrm{D}}$ strictly dominates both 1D block SD ($R\tau / C_{1\mathrm{D}}$) and 1D depth SD ($R / C_{2\mathrm{D}}$). Exact verified ratios (`Decimal`, `ROUND_HALF_UP`):
  - `Ouro-1.4B` greedy: `573/479` (`1.20x` over DFlash)
  - `Ouro-2.6B` greedy: `382/235` (`1.63x` over DFlash), `191/63` (`3.03x` over LoopSpec)
  - `Dory1.0-4B` greedy: `3/2` (`1.50x` over DFlash)
  - `Ouro-2.6B` sampling: `553/381` (`1.45x` over DFlash); `Dory1.0-4B` sampling: `179/134` (`1.34x` over DFlash)
- **Eq. 6 Last-Loop KV Compression Law & Asymptote:** Reading shared bank $C_j^{(R)}$ for $j \le t - W$ and local loop-$r$ KV inside window $W$ cuts per-layer KV traffic from $R L$ to $L + (R-1)W$, saving exact $(R-1)(L-W)$ slots and rising monotonically toward $R$:
  - `Ouro/Dory` ($R=4, W=64$): `L=2048 -> 128/35 (3.6571x)`, `L=8192 -> 512/131 (3.9084x)`, `L=32768 -> 2048/515 (3.9767x)`
  - `Nanbeige4.2-3B` ($R=2, W=64$): `L=32768 -> 1024/513 (1.9961x)`
- **Obstruction `W4B-OBS-1` (Non-Commutativity vs. CircularEP Contraction):** Whenever early-loop hidden states have nonzero drift $\Delta_h = \|h^{(R)} - h^{(1)}\| > 0$ and query-key coupling $c > 0$, substituting $C_j^{(R)}$ for $C_j^{(1)}$ injects strictly positive cumulative logit drift $c \cdot \Delta_h \cdot (L - W) > 0$. This explains 2DSD Table 3 Hard (32K) training-free drops on `Ouro-1.4B` (`-273/50 = -5.46` pts) and `Ouro-2.6B` (`-713/100 = -7.13` pts), whereas natively contracted `Dory1.0-4B` shifts `+18/25 = +0.72` pts without adaptation. Under a CircularEP contraction factor $\rho = p/q < 1$, residual drift decays geometrically ($\rho = 3/4 \implies \rho^4 = 81/256 = 0.3164$).

### Pillar 2: Sensitivity-Weighted Dense-and-Sparse CSR Crossover vs. `siliquant` `BlockH4 QuadSnake`

- **Exact Matched-Bitrate Crossover Law:** Let dense+CSR add $\Delta b = \rho_s b_{\mathrm{csr}}$ bits/weight, let $M = 2^{2\Delta b} \ge 1$ be the high-rate MSE gain bought by spending $\Delta b$ directly on dense quantization, and let $\gamma \ge 1$ be the dense group-scale shrink gain from excising outliers ($S_{\mathrm{out}}$) off bulk sensitivity mass ($S_{\mathrm{bulk}}$). Dense+CSR strictly beats matched-`bpw` pure dense iff:
  $$M \cdot S_{\mathrm{bulk}} < \gamma (S_{\mathrm{bulk}} + S_{\mathrm{out}}).$$
- **Anchored Regimes & Exact Phase-Reversal Barrier:**
  - `SqueezeLLM` (`0.45% @ 32b -> 18/125 = 0.1440` bpw, `b_eff = 518/125 = 4.1440` bpw; $M=5/4, \gamma=1, S_{\mathrm{out}}/S_{\mathrm{bulk}}=40\%$): distortion ratio `25/28 = 0.8929 < 1` -> **CSR wins**.
  - `Portal 77` (`0.057% @ 48b -> 171/6250 = 0.0274` bpw; $M=104/100$): distortion ratio `26/35 = 0.7429 < 1` -> **CSR wins**.
  - **High-CSR Phase Reversal** ($\rho_s = 1/32, b_{\mathrm{csr}}=32 \implies \Delta b = 1$ bpw, $M = 4, \gamma = 1, S_{\mathrm{out}}/S_{\mathrm{bulk}} = 200\%$): ratio `4/3 = 1.3333 > 1` -> **pure dense wins**!
  - **Simultaneous Triple Dominance (`siliquant` `BlockH4 QuadSnake + CSR`):** Whenever lattice geometry beats uniform scalar ($c_{\mathrm{lat}} < c_{\mathrm{unif}}$, anchored on `S72` `Qwen2.5-3B` nanonats: `23,154,136 < 24,178,790 < 24,774,488`, i.e., `-4.24%` vs `H2` and `-6.54%` vs `H4`) AND CSR sits inside the crossover cone, `BlockH4 QuadSnake + CSR` strictly dominates pure dense lattice, scalar+CSR, and pure dense scalar simultaneously.
  - **Exact Safe Sparsity Cap for `siliquant`:** At $b_{\mathrm{csr}}=32, \gamma=1, S_{\mathrm{out}}/S_{\mathrm{bulk}}=40\%$, exact 5th-power certification $4 \cdot 5^5 = 12{,}500 < 16{,}807 = 7^5$ proves $2^{2 \cdot (1/5)} < 7/5$, locking $\rho_s \le 1/160 = 0.625\%$ (`0.63%`) as a mathematically guaranteed win ceiling!

### Pillar 3: MultipoleAttention Centroid Error, Jensen Underestimation, Truncation Barrier & KV Covariance Residual

- **Arbitrary-Dimension Lagrange Cauchy–Schwarz Bound:** For any query $q$ and far-field cluster radius $\delta = \max_{j \in c} \|k_j - \bar{k}_c\|_2$, $(\langle q, k_j - \bar{k}_c\rangle)^2 \le \|q\|_2^2 \delta^2$, yielding scaled squared logit error $\varepsilon^2 \le \frac{\|q\|_2^2 \delta^2}{d_k}$. In our 2-key witness (`q=[3,4,0,0]`, `k1=[4,2,1,0]`, `k2=[2,2,-1,0]`, `d_k=128`), `36 <= 200` gives actual squared logit spread `9/128 = 0.0703` bounded by `25/64 = 0.3906`.
- **Exact First-Order Cancellation + Second-Order Jensen Underestimation:** Linear errors cancel identically ($\sum_j q^\top (k_j - \bar{k}_c) = 0$), so the leading partition-function bias is strictly positive quadratic convexity: $(2d^2 + 2d\varepsilon + \varepsilon^2) + (2d^2 - 2d\varepsilon + \varepsilon^2) = 4d^2 + 2\varepsilon^2 > 4d^2$ (at $d=4, \varepsilon=1$: mass ratio `33/32 = 1.0313`, meaning centroid replacement `n_c * exp(q·k_bar)` systematically underestimates true cluster softmax mass).
- **Exact Softmax Ratio Bracket & Opening-Angle Truncation Barrier:** Under multiplicative weight perturbation $[a/b, b/a]$ ($a < b$), exact cluster attention mass stays strictly inside $[a W_c / (b W_{\mathrm{near}} + a W_c),\, b W_c / (a W_{\mathrm{near}} + b W_c)]$ (at $a=4,b=5,W_{\mathrm{near}}=10,W_c=6$: `12/37 (0.3243) < 3/8 (0.3750) < 3/7 (0.4286)`). At $d_k = 128$, the $O(1)$ truncation barrier sits at $Q^2 \delta^2 \ge 128$: `(Q=8, delta=1) -> 1/2 = 0.5000 < 1` (safe), whereas `(Q=8, delta=2) -> 2 = 2.0000 >= 1` (violates far-field expansion).
- **Exact Value-Weight Covariance Residual:** Even with zero partition bias, replacing $\sum w_j v_j$ by $(\sum w_j)\bar{v}$ leaves exact 2-key residual $2(w_1 v_1 + w_2 v_2) - (w_1 + w_2)(v_1 + v_2) = (w_1 - w_2)(v_1 - v_2)$ (at `w=(3,1), v=(5,1)`: residual `8 != 0`), proving that `MultipoleAttention`'s `acc += tl.dot(p_tmp, v_bar)` incurs irreducible output error whenever key-query alignment covaries with value norms inside a cluster.

---

## 3. Formal Lean 4 Attribution Table (Zero Over-Attribution)

Every theorem in [TwoDSDAndDenseSparseLaws.lean](TwoDSDAndDenseSparseLaws.lean) (`SHA-256: a570018936e24a9ecffeedbf4991743b59a94a93e3d410076b19cf144ceaf69b`) is verified live by Lean `4.36.0` with zero imports and exact `#print axioms` accounting:

| # | Lean 4 Theorem (`TwoDSDAndDenseSparseLaws.*`) | Type | Exact Axioms | Role |
|---|---|---|---|---|
| 1 | `cascade_single_step_lossless_universal` | Universal (`Nat`) | `[]` | Single-step Eq. 5 mass conservation |
| 2 | `cascade_arbitrary_length_lossless_universal` | Universal (`List Nat`) | `[]` | Arbitrary-length cascade losslessness |
| 3 | `cascade_order_invariance_and_strict_acceptance_gain_universal` | Universal (`Nat`) | `[]` | Marginal order-invariance & strict fork gain |
| 4 | `twodsd_strictly_dominates_1d_and_loopspec_universal` | Universal (`Nat`) | `[]` | Exact rational 2D > 1D DFlash & LoopSpec |
| 5 | `stage2b_fork_rescue_strict_speedup_universal` | Universal (`Nat`) | `[]` | Stage-2b fork net loop-cost reduction |
| 6 | `kv_sharing_bandwidth_compression_and_asymptote_universal` | Universal (`Nat`) | `[]` | Eq. 6 KV savings & monotone asymptote to $R$ |
| 7 | `kv_sharing_non_commutativity_obstruction_circular_ep_universal` | Universal (`Nat`) | `[]` | `W4B-OBS-1` drift & CircularEP geometric decay |
| 8 | `dense_sparse_csr_overhead_crossover_universal` | Universal (`Nat`) | `[]` | Dense+CSR vs. matched-`bpw` pure dense law |
| 9 | `siliquant_blockh4_plus_csr_triple_dominance_universal` | Universal (`Nat`) | `[]` | `BlockH4 QuadSnake + CSR` triple dominance |
| 10 | `csr_overhead_phase_reversal_obstruction_universal` | Universal (`Nat`) | `[]` | High-CSR bitrate phase-reversal barrier |
| 11 | `multipole_logit_cauchy_schwarz_sq_bound_universal` | Universal (`List (Int × Int)`) | `[propext]` | Arbitrary-dim Lagrange Cauchy–Schwarz bound |
| 12 | `multipole_centroid_exact_unbiased_linear_cancellation_universal` | Universal (`List Int`) | `[propext]` | Exact 1st-order centroid logit cancellation |
| 13 | `multipole_two_point_jensen_underestimation_universal` | Universal (`Int`) | `[propext]` | Exact 2nd-order Jensen partition bias |
| 14 | `multipole_softmax_ratio_bracket_universal` | Universal (`Nat`) | `[]` | Exact multiplicative softmax ratio bracket |
| 15 | `multipole_exact_vs_centroid_value_covariance_universal` | Universal (`Int`) | `[propext]` | Exact 2-key value-weight covariance residual |
| 16 | `twodsd_ouro_dory_kv_and_speedup_witness` | Anchored Witness (`decide`) | `[]` | Pinned `2DSD` (`431aade2`) Tables 1–3 & Eq. 6 |
| 17 | `squeezellm_portal77_siliquant_csr_crossover_witness` | Anchored Witness (`decide`) | `[]` | Pinned SqueezeLLM / Portal 77 / `S72` / reversal |
| 18 | `multipole_truncation_barrier_and_covariance_witness` | Anchored Witness (`decide`) | `[]` | Pinned `d_k=128` barrier & nonzero covariance |

*Attribution discipline:* Exact rational ratios (`Fraction`), 5th-power safe CSR cap (`12,500 < 16,807`), and `541,880,065`-param `Inkling-2-Mini` drafter metadata are checked by [s79_2dsd_multipole_oracle.py](s79_2dsd_multipole_oracle.py) and independently re-audited by [verify_s79_2dsd_multipole.py](verify_s79_2dsd_multipole.py); they are NOT claimed as Lean theorems unless listed in rows 1–18 above.

---

## 4. Honest `N0` Prior-Art Quarantine & `N1` Scope (Law 4)

- **Quarantined `N0` Prior Art:** Fast Multipole Method (Greengard & Rokhlin 1987); speculative decoding rejection sampling (Stern et al. 2018; Leviathan et al. 2023; Chen et al. 2023); SqueezeLLM (Kim et al. 2023); KVQuant (Hooper et al. 2024); MultipoleAttention (Hooper et al. 2025); DFlash (Chen et al. 2026); LoopSpec (Cho et al. 2026); 2DSD (SqueezeAILab 2026); CircularEP (Salvatori & Forasassi 2026); classical Lagrange/Cauchy–Schwarz/Jensen/Shannon inequalities.
- **Claimed `N1` Synthesis:** Unified machine-checked Lean 4.36.0 pure-`Init` kernel proofs (`14/18` zero axioms, `4/18` `[propext]`-only) + exact $\mathbb{Z}/\mathbb{Q}$ oracle/auditor connecting (i) `2DSD` Eq. 6 KV non-commutativity to `CircularEP` contraction, (ii) `SqueezeLLM`/`Portal 77`/`Track S74` CSR overhead to `siliquant` `BlockH4 QuadSnake` with an exact `0.625%` break-even cap and phase-reversal obstruction, and (iii) `MultipoleAttention` far-field cluster error to exact Jensen underestimation, opening-angle barrier ($Q^2 \delta^2 \ge d_k$), and irreducible key-value covariance residuals.

---

## 5. Formula Transposition into Our Stack (Law 5)

1. **`siliquant` (`BlockH4 QuadSnake` + CSR Sidecar Gate):** Enforce an exact integer guard before allocating a sparse CSR sidecar on any weight matrix: only enable CSR when measured outlier sensitivity mass satisfies $M \cdot S_{\mathrm{bulk}} < \gamma(S_{\mathrm{bulk}} + S_{\mathrm{out}})$ (and cap $\rho_s \le 1/160 = 0.625\%$ at `32`-bit CSR entries whenever $S_{\mathrm{out}}/S_{\mathrm{bulk}} \le 40\%$, preventing the `1/32` phase reversal where CSR degrades KL divergence vs. pure dense lattice).
2. **`siliquant` + `2DSD` (`Inkling-2-Mini` Looped-LM Serving):** Pair `W = 64` last-loop KV sharing (`3.9767x` KV bandwidth reduction at `L = 32K, R = 4`) with a pre-deployment CircularEP contraction check on early-loop hidden drift $\Delta_h$: if uncontracted ($\rho \approx 1$, `Ouro`-like), trigger lightweight self-distillation before serving; if naturally contracted (`Dory`-like), deploy training-free with zero accuracy regression.
3. **`warp-drive` / Long-Context Multipole Kernel Gate:** Guard far-field centroid substitution (`BLOCK = 8192`) per cluster with the exact integer test $\|q\|_2^2 \delta_c^2 < d_k$ AND key-value covariance split check; clusters breaching either criterion automatically demote to exact near-field attention instead of silently accumulating Jensen underestimation and `(w1-w2)(v1-v2)` bias.
