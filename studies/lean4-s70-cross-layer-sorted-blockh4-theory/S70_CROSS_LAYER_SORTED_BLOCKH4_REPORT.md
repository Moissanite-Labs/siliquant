# CEER Formal Theory Companion: Cross-Layer Non-Additivity, Sorted Block-$H_4$ Optimality, and 3-Model Regime-Triangle Minimax Regret (`Lean 4.36.0`, Pure `Init`)

## 1. Executive Summary & Honest Scope

This self-contained formal-verification bundle proves **21 machine-checked theorems** in pure-`Init` **`Lean 4.36.0`** (zero `Mathlib`, zero external imports, zero compiler warnings on `stdout` or `stderr`, strict `#print axioms` allowlist with zero `sorryAx` / `Lean.ofReduceBool` / `Lean.trustCompiler`) paired with a pure-$\mathbb{Z}/\mathbb{Q}$ Python 3 standard-library oracle (`s70_lean4_oracle.py`), a fail-closed hardened verifier and static Lean 4 scanner (`s70_lean4_audit.py`), a regression/hardening test suite (`test_s70_lean4_r5.py`), and a canonical self-verifying certificate (`s70_certificate.json`).

- **Novelty & Oracle Classification:** `N1` (formal machine-checked synthesis and exact rational verification unifying classical inequalities with frozen empirical LLM quantization anchors), `oracle_class = exact`.
- **Hardware / Measurement Boundary:** `measured_live_gpu_in_this_bundle = false`. Zero GPU execution, zero model training/inference, and zero hardware sweeps occur in this bundle. All empirical LLM KL measurements are pinned read-only by SHA-256 to their upstream frozen artifacts (`49283ff4285f3d4d8a9723940b859b1c43187232`, `c2a15fdbdfd3245b8ab4bf7eb6772f66b1399831026680d5307c2a3be6ccf08b`, and `b192e19f35f78689f6f2ab06ff3c2dbec830a422c006178ba5c78e843be77906`).
- **Canonical Certificate Digests (`s70_certificate.json`, `6,158` B on disk, `6,157` B canonical JSON):**
  - `payload_sha256`: `1aa12bca26cef558f0ebe7aab36087ee69d6adca42aa1413ccddb50670066a76`
  - `certificate_sha256`: `0d463e6227f798a6050765ce57305e804df4fc6608ffa558a5bf7ec17b7e3be5`
  - File SHA-256 (`s70_certificate.json`): `5cdce98df676290d78688da689fe576492487125653e563c667df000980c45e4`

---

## 2. Pillar 1 — Cross-Layer Non-Additivity & Greedy-Layerwise Suboptimality

**Lean 4 Module:** `S70FormalTheory/CrossLayerNonAdditivity.lean` (`259` lines, `7` theorems).

When two quantized layers propagate perturbations $u, v \in \mathbb{Z}^d$ into a composed residual stream, total squared error obeys the exact quadratic interference law
$$\|u + v\|_2^2 = \|u\|_2^2 + \|v\|_2^2 + 2 \langle u, v \rangle, \qquad |\langle u, v \rangle|^2 \le \|u\|_2^2 \|v\|_2^2,$$
with worst-case cross-term envelope $-(\|u\|_2^2 + \|v\|_2^2) \le 2 \langle u, v \rangle \le \|u\|_2^2 + \|v\|_2^2$. Because per-layer selectors optimize $\|u\|_2^2$ and $\|v\|_2^2$ independently while ignoring $\langle u, v \rangle$, a layerwise-optimal arm $B$ can beat a uniform arm $A$ strictly on **every individual layer** yet lose strictly—and by an **unbounded factor** $\ge k$ for every integer $k \ge 1$—on the composed network whenever uniform quantization induces destructive cross-layer cancellation ($\langle u_A, v_A \rangle < 0 < \langle u_B, v_B \rangle$).

- **Exact Parameterized Family (`greedy_layerwise_suboptimality_universal` & `greedy_suboptimality_unbounded_ratio_universal`):** For all integers $0 < b < a$, $d_{1A} = [a, b], d_{2A} = [-a, b]$ vs. $d_{1B} = [a, 0], d_{2B} = [a, 0]$ satisfy $\|d_{1B}\|^2 = a^2 < a^2 + b^2 = \|d_{1A}\|^2$, $\|d_{2B}\|^2 = a^2 < a^2 + b^2 = \|d_{2A}\|^2$, $\langle d_{1A}, d_{2A} \rangle = b^2 - a^2 < 0 < a^2 = \langle d_{1B}, d_{2B} \rangle$, and $\|d_{1A} + d_{2A}\|^2 = 4b^2 < 4a^2 = \|d_{1B} + d_{2B}\|^2$. Setting $(a, b) = (10, 1)$ gives a `100`$\times$ regression (`4` vs. `400`); setting $(a, b) = (k + 1, 1)$ achieves ratio $(k + 1)^2 \ge k$ (`1,002,001` at $k = 1000$).
- **Coupled-Coherence Sufficiency & Sharp Reversal (`coupled_coherence_margin_universal`):** If layerwise arm $B$ saves marginal sum $g$ over $A$, then $B$ beats $A$ globally if $2(c_B - c_A) < g$, whereas $A$ strictly reverses $B$ whenever $g < 2(c_B - c_A)$.

### Frozen `Qwen2.5-3B` WikiText-2 Exact Nanonat Ladder (`qwen3b_wiki_nonadditivity_exact_witness`)

| Quantization / Routing Arm | Exact KL (nanonats, $10^{-9}$) | Delta vs. `A16_axis` (nanonats) | Exact Relative Ratio $\Delta / \text{axis}$ ($\mathbb{Q}$) | Relative Delta vs. `A16_axis` |
| --- | ---: | ---: | --- | ---: |
| `E8_H8` | `45,723,724` | `-962,720` | — | — |
| `A16_axis` (uniform baseline) | `46,686,444` | `0` | `0` | `0.0000%` |
| `A16_BlockH4` (uniform) | `47,941,673` | `+1,255,229` | — | — |
| `Oracle_1bit` (layerwise oracle) | `50,421,289` | `+3,734,845` | `3734845/46686444` | `+7.9998%` |
| `A16_AdaptZeroCal_1bit` | `51,076,282` | `+4,389,838` | `2194919/23343222` | `+9.4028%` |
| `A16_AdaptZeroCal_2bit` | `52,907,251` | `+6,220,807` | `6220807/46686444` | `+13.3247%` |
| `Oracle_2bit` (layerwise oracle) | `53,398,235` | `+6,711,791` | `6711791/46686444` | `+14.3763%` |

Every inequality in the `7`-arm strict ladder (`E8_H8 < A16_axis < A16_BlockH4 < Oracle_1bit < A16_AdaptZeroCal_1bit < A16_AdaptZeroCal_2bit < Oracle_2bit`) is proved with zero axioms (`[]`) in `qwen3b_wiki_nonadditivity_exact_witness`.

---

## 3. Pillar 2 — Zero-Cost SwiGLU Permutation Invariance & Sorted Block-$H_4$ Optimality

**Lean 4 Module:** `S70FormalTheory/SortedBlockH4Optimality.lean` (`402` lines, `7` theorems).

1. **Universal Zero-Cost Hidden-Channel Permutation Invariance (`swiglu_perm_invariance_fin_universal` & `swiglu_matrix_perm_invariance_universal`):** For **every** hidden dimension $d_{\text{hid}} \in \mathbb{N}$, **every** elementwise nonlinearity $\sigma : \mathbb{Z} \to \mathbb{Z}$ (covering `SiLU`, `GeLU`, `ReLU`, or any quantized LUT), **every** bijection $\pi : \mathrm{Fin}\, d_{\text{hid}} \to \mathrm{Fin}\, d_{\text{hid}}$ with two-sided inverse $\mathrm{inv}$, and **every** input vector $x \in \mathbb{Z}^{d_{\text{in}}}$, permuting the rows of `gate_proj` $G$ and `up_proj` $U$ and the columns of `down_proj` $D$ by $\pi$ yields **bit-identical** output on every row $i \in \mathrm{Fin}\, d_{\text{out}}$ at zero online runtime cost. Independent pure-$\mathbb{Z}$ verification on a $(4, 8, 3)$ `SwiGLU` instance with $\sigma(z) = z(z + 1)$ and permutation `(5, 2, 7, 0, 4, 1, 6, 3)` confirms reference output `[1408, -43910, 8694]` and exact max difference `0`.
2. **Exact $H_4$ Orthogonality & Parseval Mixing (`h4_orthogonal_and_parseval_universal`):** Proves $H_4^{\mathsf{T}} H_4 = 4 I_4$ for all $r, c \in \mathrm{Fin}\, 4$ and universal Parseval energy $\|H_4 v\|_2^2 = 4 \|v\|_2^2$ for all $v \in \mathbb{Z}^4$ (concrete witness $v = (11, -7, 3, -5)$: $\|H_4 v\|_2^2 = 4\|v\|_2^2 = 816$).
3. **Global Optimality of Monotone Quad Sorting (`rearrangement_2quad_slack_universal`, `monotone_quad_sq_sum_global_max_universal`, `monotone_quad_spread_global_min_universal`):**
   - Exact 2-quad rearrangement slack: $((a+b)^2 + (c+d)^2) - ((a+d)^2 + (b+c)^2) = 2(a - c)(b - d)$.
   - For **any** $4+4$ partition $(S_A, S_B)$ of sorted channel scales $h_0 \ge \dots \ge h_7$, monotone grouping $(S_{\text{hot}}, S_{\text{quiet}})$ uniquely maximizes $S_A^2 + S_B^2$ with exact non-negative slack $2(S_{\text{hot}} - S_A)(S_A - S_{\text{quiet}})$.
   - For **any** overlapping $4+4$ partition covering $[h_7, h_0]$, total within-quad spread satisfies $(M_1 - m_1) + (M_2 - m_2) \ge h_0 - h_7 = (h_0 - h_3) + (h_4 - h_7) + (h_3 - h_4)$, strictly exceeding monotone spread by at least the inter-quad boundary gap $h_3 - h_4$ whenever $h_3 > h_4$.
   - **Exhaustive $35$-Partition Oracle Check** on $h = (100, 95, 90, 85, 12, 8, 5, 1)$: monotone quad achieves unique global maximum squared sum `137,576` (strictly beating all `34/34` non-monotone partitions) and unique minimum spread `26` vs. `172` minimum non-monotone spread (`6.6154`$\times$ reduction).
4. **Exact `INT8` `dp4a` Butterfly Envelope & Extreme Contrast (`h4_int8_butterfly_and_spill_witness`):** With `4`-bit signed weights in $[-7, 7]$, the exact `H_4` butterfly magnitude is bounded by $4 \times 7 = 28 \le 127$ (`99` units of `INT8` overflow headroom). On a 4-hot/4-quiet profile `(100, 100, 100, 100, 1, 1, 1, 1)`, monotone sorting cuts total quad spread from `198` to `0` while increasing squared quad-sum from `81,608` to `160,016` (`1.9608`$\times$, exact gap `+78,408`).

---

## 4. Pillar 3 — 3-Model Regime-Triangle Obstruction & Exact 5-Cell Minimax-Regret Optimality of $A_{16}$

**Lean 4 Module:** `S70FormalTheory/RegimeTriangleMinimax.lean` (`161` lines, `7` theorems).

1. **Universal 2-Term Simplex Obstruction (`two_term_simplex_obstruction_universal`):** For **every** non-negative 2-term rule $S(d, e) = w_d d + w_e e$ omitting a positively weighted third axis ($w_k > 0$) and **every** residual bound $M \ge 0$, there exist arms $A, B$ where $A$ strictly dominates $B$ on both $d$ and $e$ (so $S(A) \le S(B)$), yet true 3-vertex loss satisfies $L(A) - L(B) \ge M$.
2. **Simplex Convex-Mixture Bound & Unique Minimax Optimality (`simplex_convex_regret_bound_universal` & `simplex_unique_minimax_universal`):** If arm $C^*$ achieves vertex regret $\le R$ at every simplex vertex while every rival exceeds $R$ on at least one vertex, then $C^*$ bounds regret by $(\lambda_1 + \lambda_2 + \lambda_3) R$ over every non-negative mixture and strictly minimizes worst-case regret over the entire simplex.
3. **Construction-A Extended Hamming Family & DP Reconciliation (`hamming_family_exact_rate_and_covering`):** Proves exact rate increase ($1/2 < 11/16 < 13/16$), parity-density decay ($1/2 > 5/16 > 3/16$), determinant doubling ($16 \to 32 \to 64$), `covering_sq(128)` drop ($56 > 44 > 38$), and exact DP table identities ($A_{16}$ two-level `660,776` vs. flat `1,275,648`; $A_{32}$ two-level `2,147,824` vs. flat `2,145,792`, diff `+2,032`).
4. **Frozen `Llama-3.2-1B` Two-Term Obstruction (`llama_a32_two_term_obstruction_exact_witness`):** `A32BG` strictly beats `HBG` on pooled MSE (`-25.627 < -25.585` dB) and small-weight $E_{05}$ (`0.15 < 3.48` dB), yet strictly loses on WikiText-2 KL (`39,540 > 36,038` micro-KL) with exact squared SE-gap `3066001/162194` (`18.9033 > 18`).

### Exact 5-Arm Minimax-Regret Table Across `4.5 bpw` (3 Models) and All `5` Bitrate $\times$ Model Cells

| Lattice Arm | Worst Rel. Regret (`4.5 bpw`, 3 cells, exact $\mathbb{Q}$) | `4.5 bpw` Worst Regret | `4.5 bpw` Rank | Worst Rel. Regret (All `5` cells, exact $\mathbb{Q}$) | All-`5` Worst Regret | All-`5` Rank | Worst Cell (All `5`) |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- |
| **`A16BG`** | **`2157/45800`** | **`4.7096%`** | **`#1`** | **`7383/119615`** | **`6.1723%`** | **`#1`** | `S17_30` |
| `A32BG` | `339/2297` | `14.7584%` | `#3` | `339/2297` | `14.7584%` | `#2` | `L10_45` |
| `ZBG` | `4451/11485` | `38.7549%` | `#5` | `4451/11485` | `38.7549%` | `#3` | `L10_45` |
| `HBG` | `2861/22900` | `12.4934%` | `#2` | `271057/358845` | `75.5360%` | `#4` | `S17_30` |
| `EBG` | `15061/45800` | `32.8843%` | `#4` | `305344/358845` | `85.0908%` | `#5` | `S17_30` |

- **Exact Minimax Advantage:** Across all `5` cells (`Q15_45`, `S17_45`, `L10_45`, `Q15_30`, `S17_30`), **`A16BG`** is the unique minimax-regret optimum by **`2.3911`$\times$** over `#2` (`A32BG`, exact ratio `13516495/5652917`) and achieves rank $\le 2$ in **every single cell** (`[2, 2, 1, 1, 2]`). At `4.5 bpw` alone, **`A16BG`** beats `#2` (`HBG`) by **`2.6528`$\times$** (exact ratio `5722/2157`).

---

## 5. Complete 21-Theorem Lean 4.36.0 Inventory & `#print axioms` Verification

All `3` modules compile standalone with `lean` and as a package with `lake build` (`8/8` jobs, `0` warnings on `stdout`/`stderr`, `0` Mathlib, `0` imports, `0` static R8 findings, `0` anti-vacuity notices).

| # | Module | Theorem | Exact `#print axioms` |
| ---: | --- | --- | --- |
| 1 | `CrossLayerNonAdditivity` | `int_vec_quad_interference_universal` | `[propext, Quot.sound]` |
| 2 | `CrossLayerNonAdditivity` | `int_cauchy_schwarz_universal` | `[propext, Classical.choice, Quot.sound]` |
| 3 | `CrossLayerNonAdditivity` | `interference_envelope_universal` | `[propext, Quot.sound]` |
| 4 | `CrossLayerNonAdditivity` | `greedy_layerwise_suboptimality_universal` | `[propext, Quot.sound]` |
| 5 | `CrossLayerNonAdditivity` | `greedy_suboptimality_unbounded_ratio_universal` | `[propext, Quot.sound]` |
| 6 | `CrossLayerNonAdditivity` | `coupled_coherence_margin_universal` | `[propext, Quot.sound]` |
| 7 | `CrossLayerNonAdditivity` | `qwen3b_wiki_nonadditivity_exact_witness` | `[]` (zero axioms) |
| 8 | `SortedBlockH4Optimality` | `swiglu_perm_invariance_fin_universal` | `[propext, Quot.sound]` |
| 9 | `SortedBlockH4Optimality` | `swiglu_matrix_perm_invariance_universal` | `[propext, Quot.sound]` |
| 10 | `SortedBlockH4Optimality` | `h4_orthogonal_and_parseval_universal` | `[propext, Quot.sound]` |
| 11 | `SortedBlockH4Optimality` | `rearrangement_2quad_slack_universal` | `[propext, Quot.sound]` |
| 12 | `SortedBlockH4Optimality` | `monotone_quad_sq_sum_global_max_universal` | `[propext, Quot.sound]` |
| 13 | `SortedBlockH4Optimality` | `monotone_quad_spread_global_min_universal` | `[propext, Quot.sound]` |
| 14 | `SortedBlockH4Optimality` | `h4_int8_butterfly_and_spill_witness` | `[]` (zero axioms) |
| 15 | `RegimeTriangleMinimax` | `two_term_simplex_obstruction_universal` | `[propext, Quot.sound]` |
| 16 | `RegimeTriangleMinimax` | `simplex_convex_regret_bound_universal` | `[propext, Quot.sound]` |
| 17 | `RegimeTriangleMinimax` | `simplex_unique_minimax_universal` | `[propext, Quot.sound]` |
| 18 | `RegimeTriangleMinimax` | `hamming_family_exact_rate_and_covering` | `[]` (zero axioms) |
| 19 | `RegimeTriangleMinimax` | `llama_a32_two_term_obstruction_exact_witness` | `[]` (zero axioms) |
| 20 | `RegimeTriangleMinimax` | `a16_unique_minimax_regret_45bpw_witness` | `[]` (zero axioms) |
| 21 | `RegimeTriangleMinimax` | `a16_unique_minimax_regret_all5_cells_witness` | `[]` (zero axioms) |

---

## 6. Verification & Adversarial Hardening Receipts

- **Lean 4.36.0 Live Verification:** `s70_lean4_audit.py --verify-lean . --json` passes with `3/3` modules SHA-256 pinned, `21/21` kernel theorems verified, `0` compiler warnings (`stdout` and `stderr` fail-closed), `0` banned axioms, and `0` static R8 findings/notices; clean scratch `lake build` succeeds (`8` jobs, `0` warnings; `.lake/` absent from repository).
- **R4 3-Mode Isolated Parity & Hash-Seed Parity:** `s70_lean4_audit.py --standalone` and `test_s70_lean4_r5.py` (`14/14 OK`) pass bit-identically across `-I -B`, `-I -O -B`, `-I -B -W error`, and `PYTHONHASHSEED=42 -B`, as well as under `umask 077` and simulated `chmod 644 s70_certificate.json` CI checkout parity (while `--write` strictly enforces `0o600`).
- **Adversarial Fuzzing & Mutation Hardening:**
  - 4-surface v2.1 adversarial fuzzer (`CLI 25/25`, `FR1 19/19`, `FR0 19/19`, `SEC_DIRECT 19/19`): **`82/82` clean rejections (`bypasses=0`, `crashes=0`)**.
  - Full `aletheia mutate --guards require_ok --workers 4 --timeout 180 --max-mutants 0`: **`50/50` (`100.0%`)** on `s70_lean4_oracle.py` and **`126/126` (`100.0%`)** on `s70_lean4_audit.py` (`176/176` combined, `SURVIVED=0`, `TIMEOUT=0`).
  - Whole-file baseline-checked DFS AST `BoolOp` (`OP_SWAP + SUBCOND_DEL`): **`18/18` (`100.0%`)** on `s70_lean4_oracle.py` and **`228/228` (`100.0%`)** on `s70_lean4_audit.py` (`246/246` combined, `SURVIVED=0`, `TIMEOUT=0`).
  - Exhaustive re-signed payload leaf/schema tamper sweep (`> 180` mutations) and coordinated multi-field tamper tests: `100%` rejected under both `full_replay=False` and direct `secondary_audit_payload`.

### Reproducibility Commands

```bash
python3 -I -B s70_lean4_audit.py --check s70_certificate.json
python3 -I -B s70_lean4_audit.py --verify-lean . --json
python3 -I -B test_s70_lean4_r5.py
```

---

## 7. Frozen 9-File Artifact Manifest (Excluding This Report)

| File | Lines | Bytes | Mode | SHA-256 |
| --- | ---: | ---: | --- | --- |
| `lean-toolchain` | `1` | `25` | `0o644` | `582a63f8f9c5128067b688778c1051cf1464c3ca9d57b9325025c051c29b774a` |
| `lakefile.toml` | `11` | `264` | `0o644` | `c0a107a13125e50b01675f08fd7fa67dca540874dfdcdcf1e768cd655fa114e3` |
| `S70FormalTheory/CrossLayerNonAdditivity.lean` | `259` | `10,835` | `0o644` | `5824341950d2c1d8723d63e65add314de2ee8a692ecd7e6cc85d093d6aff5a08` |
| `S70FormalTheory/SortedBlockH4Optimality.lean` | `402` | `17,080` | `0o644` | `282226b1ba9d6cf939115dd7b12fabd156413e96f9ff096ff26636bbaf511743` |
| `S70FormalTheory/RegimeTriangleMinimax.lean` | `161` | `8,588` | `0o644` | `e56d15d0d65b9bbca5075409945f0113534f2a6b7707fc7a6b30e41e84db1a5b` |
| `s70_lean4_oracle.py` | `691` | `27,419` | `0o644` | `7c921e60c88761625c08ca3b276278275b20b7d4ca9a4216e20430f491b47ba3` |
| `s70_lean4_audit.py` | `1312` | `58,436` | `0o644` | `30578c4051132ba8878f8b9a399027a24d7f9f45afb6af2fcef6c98b8ece2733` |
| `test_s70_lean4_r5.py` | `1245` | `70,256` | `0o644` | `37ace9be3c154fd343fface7d02d7f3bfac249aa82f5d57e3566b26309d14454` |
| `s70_certificate.json` | `1` | `6,158` | `0o600` | `5cdce98df676290d78688da689fe576492487125653e563c667df000980c45e4` |

## 8. Prior-Literature (`N0`) Attribution

- **Cauchy (1821) / Schwarz (1885):** Inner-product quadratic interference inequality.
- **Hardy, Littlewood, Polya (1934):** Rearrangement inequality and Schur convexity of symmetric quadratic sums.
- **Sylvester (1867) / Walsh (1923):** $H_4$ Walsh–Hadamard orthogonality and Parseval energy conservation.
- **Shazeer (2020):** GLU variants (`SwiGLU`) elementwise gated-MLP structure.
- **Conway & Sloane (1988):** Construction-A Extended Hamming lattices $[2^m, 2^m - m - 1, 4]$.
- **Savage (1951):** Minimax-regret decision rule over finite hypothesis simplices.
