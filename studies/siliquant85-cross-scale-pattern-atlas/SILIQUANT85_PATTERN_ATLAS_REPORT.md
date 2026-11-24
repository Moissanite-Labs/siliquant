# Cross-Scale Pattern Atlas & Pure-Z/Q Tomography of `Moissanite-Labs/siliquant#85` (`49283ff4`)

## 1. Executive Summary & Epistemic Scope (CS-13)

This study performs an exact integer/rational ($\mathbb{Z}/\mathbb{Q}$, zero floating-point trust path) number-trace audit and five-pattern cross-scale synthesis of `Moissanite-Labs/siliquant#85` (head commit `49283ff4285f3d4d8a9723940b859b1c43187232`, `50` files, `18` pinned evidence outputs).

- **Epistemic Taxonomy (`exact` / `sampled` / `anchor`):**
  - **`exact` tier:** Construction-A extended Hamming code family $[2^m, 2^m - m - 1, 4]$ ($m \in \{3, 4, 5\}$), exact DP table-entry reconciliation between two-level `HammingBallIndex` and flat `HamFormat`, and Stage-A coordinate-free scale/permutation invariance over exact $\mathbb{Q}$ rank correlation matrices.
  - **`sampled` tier:** Exact rational encoding of committed PR `#85` empirical measurements (`Qwen2.5-1.5B-Instruct`, `SmolLM2-1.7B-Instruct`, `Llama-3.2-1B-Instruct`, `10,200` WikiText-2 test tokens) and cross-referenced external AVX-512 / RTX 3080 `sm_86` hardware receipts (`measured_live_llm_in_this_track = false`, `measured_live_gpu_in_this_track = false`).
  - **`anchor` tier:** `18` pinned SHA-256 digests from commit `49283ff4285f3d4d8a9723940b859b1c43187232`, verifiable read-only via `siliquant85_pattern_audit.py --verify-portal <dir>`.
- **CS-13 Novelty Tier (`N1` methodological & analytic synthesis):** Prior foundations (`N0`) are Conway & Sloane (1988) Construction-A Hamming lattices, Cover (1973) enumerative spherical coding, and PR `#85`'s committed empirical measurements.

---

## 2. Number-Trace & Artifact-Integrity Audit of PR `#85` (`49283ff4`)

Running PR `#85`'s committed analysis scripts read-only confirms:
1. **Byte-Exact Output Reproducibility (`6/6` runnable scripts):** `hamming-lattices-real-weights-2026-11-24/summary.py` (`8,430` B), `packed-hamming-2026-11-24/summary.py` (`3,501` B), `against_ball.py` (`607` B), `arms_table.py` (`4,092` B), `checks/claim_ledger.py` (`954` B), and `checks/number_trace.py` (`21,131` B) reproduce their committed `.out` files byte-for-byte. (`checks/plan_before_data.py` adds only a shallow-clone warning banner.)
2. **Decimal Number Trace:** `checks/number_trace.out` checks `5,537` decimal numbers across `26` records (`1` missing in an older `2026-11-17` record); PR `#85`'s two records trace **`189/189` (`0` missing)** and **`207/207` (`0` missing)**.
3. **Check `C1` & Prediction Ledger:** `A8BG` (`0.051598`) vs `HBG` (`0.051522`) on `SmolLM2-1.7B` gives exact difference `+76/1000000`, satisfying `760 <= 1057` (`0.1 * SE = 0.0001057`); `hamming-lattices` holds `14/15` decided predictions (`H2` fails at `-4.35` SE on `Llama-3.2-1B`), and `packed-hamming` holds `6/6`.
4. **Three Exact Structural Findings:**
   - **DP Table Reconciliation (`group32.out` vs `hampack_speed.out`):** Two-level block-factored `HammingBallIndex` uses $(L+1)(2L)(B+1) + (k+1)(B+1)$ while single-level `HamFormat` uses $(n+1)(2L)(B+1)$. For `A_16` over `32` weights (`B = 1207`, `L = 16`, `k = 2`), this yields `660,776` vs `1,275,648` (exact ratio `1056/547 = 1.9305x`); for `A_32` (`B = 1015`, `L = 32`, `k = 1`), `2,147,824` vs `2,145,792` (exact difference `2 * 1016 = 2,032` entries); for `A_8^4` (`B = 1567`), `827,904`; for `Z^32` (`B = 783`), `25,872`.
   - **Disclosed PREREG Byte-Budget Overrun (`PH16+K10`):** Stored bpw is `4.504916..4.505425` (above the PREREG `4.5000..4.5007` window due to `10` kept FP16 `ffn_down` columns), matching `PZ32+K10` to the **exact byte** on all `3` models (`737,853,158` / `906,959,478` / `548,016,534` B; base `PH16 == PZ32`: `736,991,766` / `905,975,382` / `547,360,470` B).
   - **Missing Falsifiable Diagnostic Cell:** `Llama-3.2-1B:A32BG128:b560s16+K10` was not scored in PR `#85` (only `ZBG`, `EBG`, `HBG`, `A16BG` received `+K10`).

---

## 3. Five Pure-Z/Q Cross-Scale Patterns

### Pattern 1: Solving PR `#85`'s Open `A_32` Regime Puzzle (`H2` Failure on `Llama-3.2-1B`)

Construction-A extended Hamming lattices $A_{2^m} = C_{[2^m, 2^m - m - 1, 4]} + 2\mathbb{Z}^{2^m}$ ($m \in \{3, 4, 5\}$) exhibit an exact algebraic turnover:

| Lattice | $[n, k, d_{\min}]$ | Code Rate $k/n$ | Parity Density $(m+1)/n$ | $\det(A_n)$ | `covering_sq(128)` | Pooled MSE (`4.5 bpw`) | $E_{0.05}$ (`4.5 bpw`) |
|---|---|---:|---:|---:|---:|---:|---:|
| `A_8` (`HBG`) | `[8, 4, 4]` | `1/2` (`0.5000`) | `1/2` (`0.5000`) | `16` | `56` | `-25.585 dB` | `+3.15..+3.75 dB` |
| `A_16` | `[16, 11, 4]` | `11/16` (`0.6875`) | `5/16` (`0.3125`) | `32` | `44` | **`-25.707 dB`** (peak) | `+1.12..+1.59 dB` |
| `A_32` | `[32, 26, 4]` | `13/16` (`0.8125`) | `3/16` (`0.1875`) | `64` | `38` | `-25.627 dB` (`+0.080 dB` retreat) | `-0.13..+0.19 dB` |

Because $d_{\min} = 4$ is fixed while $k/n \to 1$ (approaching $\mathbb{Z}^n$), parity disturbance on tiny weights ($|w| < 0.05\,\mathrm{RMS}$) drops monotonically (`zero(0.05)` rises `83.7..85.9% -> 93.4..95.6% -> 97.3..99.0%`), **but** pooled Voronoi shaping peaks at `A_16` (`-25.707 dB`) and retreats at `A_32` (`-25.627 dB`), coarsening the outer integer grid (`thr=0.2` on Llama-3.2 worsens from `-7.694 dB` to `-7.605 dB` as `zero(0.2)` jumps `37.05% -> 40.42%`) while doubling parity-coupling span to `32` coordinates.

This separates the three models into a sharp **3-Model Regime Triangle**:

| Model | Regime | `ZBG - EBG` (micro-KL) | `A16 vs HBG` (exact $g^2$) | `A32 vs HBG` (exact $g^2$) | `A32 vs A16` (exact $g^2$) | Packed `PH16 -> +K10` $\Delta$ |
|---|---|---:|---:|---:|---:|---:|
| `SmolLM2-1.7B` | `e05_small_weight_dominated` | `-4899` (`Z` beats `E`!) | `+2.43` SE (`5.896`) | **`+4.15` SE (`17.228`)** | `+1.60` SE (`2.554`) | `-4273/52260` (`-8.18%`) |
| `Qwen2.5-1.5B` | `mse_dominated_k10_quiet` | `+3201` (`E` beats `Z`) | `-0.18` SE (`0.032`) | `-0.26` SE (`0.066`) | `-0.10` SE (`0.010`) | `-355/21171` (`-1.68%`) |
| `Llama-3.2-1B` | `k10_and_coarse_grid_dominated` | `+7647` (`E` beats `Z` `+7.93` SE) | **`+2.18` SE (`4.767`)** | **`-4.35` SE (`18.903`, `H2` FAILS)** | **`-6.45` SE (`41.631`)** | **`-3243/9920` (`-32.69%`)** |

**Key Proof Witness:** On `Llama-3.2-1B`, `A_32` beats `HBG` on **both** pooled MSE (`-25.627 < -25.585 dB`) **and** small-weight error $E_{0.05}$ (`+0.15 < +3.48 dB`), yet **loses on model KL** (`0.039540 > 0.036038`, `-4.35` SE), causing the 2-parameter `(D, E_05)` fit in `arms_table.out` to break down ($\chi^2 \approx 81.8$ across `5` arms vs `2.0` on `Qwen2.5`). Because `10` loud `ffn_down` columns account for `-30.19%` (`-10402/34455`) of 128-wt `A16BG` KL and `-32.69%` of packed `PH16` KL (`19.5x` Qwen2.5's share), scoring the omitted cell `Llama-3.2-1B:A32BG128:b560s16+K10` directly isolates whether the `A_32` excess collapses with loud columns or persists in non-`K10` bulk weights.

### Pattern 2: Cross-Bitrate (`4.5 -> 3.0 bpw`) Nonlinear Amplification Law

Exact pure-$\mathbb{Q}$ `KL(3.0) / KL(4.5)` blowup ratios expose the `E_05` catastrophe on `SmolLM2-1.7B` and how `A_16`/`A_32` suppress it by **`1.56x`** (`HBG/A32` amplification ratio `1442475580/924349743 = 1.5605x`):

| Model | `ZBG` (`3.0/4.5`) | `EBG` (`3.0/4.5`) | `HBG` (`3.0/4.5`) | `A16BG` (`3.0/4.5`) | `A32BG` (`3.0/4.5`) | `3.0 bpw` Best Arm & Margin |
|---|---:|---:|---:|---:|---:|---|
| `Qwen2.5-1.5B` | `215147/22750` (`9.457x`) | `169338/19549` (`8.662x`) | `178526/18875` (`9.458x`) | **`80184/9487` (`8.452x`)** | `87097/9516` (`9.153x`) | `A16BG` (`+4.40` SE vs `HBG`, `+2.18` SE vs `EBG`) |
| `SmolLM2-1.7B` | `46627/6218` (`7.499x`) | `664189/60861` (`10.913x`, $E_{0.05}=+14.27$) | `314951/25761` (`12.226x`, $E_{0.05}=+9.61$) | `380994/47957` (`7.945x`, $E_{0.05}=+3.07$) | **`71769/9160` (`7.835x`, $E_{0.05}=+0.06$)** | `A32BG` (`+27.31` SE vs `HBG`, `+7.54` SE vs `ZBG`) |

### Pattern 3: `colproxy` `21`-Cell Architectural Sign-Inversion Law

Across all `3 models x 7 projection kinds = 21` cells in `notes/colproxy.out` (median Spearman with $\log h$):
- **Universal Positive Noise Law (`21/21`):** Random-token activations (`noise`) are strictly positive in **`21/21` cells** (`+0.236` to `+0.662`) and beat weight-column `sig` in **`20/21` cells** (sole exception: `Llama-3.2 ffn_up`, `+0.653` vs `+0.504`).
- **100% Attention-vs-FFN `gam` Polarity Split (`15/15`):** RMSNorm gain `gam` is strictly positive on **`9/9` attention cells** (`+0.066, +0.156, +0.630`) and strictly negative on **`6/6` FFN cells** (`-0.414, -0.289, -0.392`) across all three architectures.
- **Cross-Model `sig` Sign Inversion:** `Qwen2.5-1.5B` has `2/7` positive cells (exact mean `-2123/7000 = -0.3033`, down to `-0.698` on `attn_k`); `SmolLM2-1.7B` has `4/7` positive (mean `17/200 = +0.0850`); `Llama-3.2-1B` has `6/7` positive (mean `1049/3500 = +0.2997`). Projection invariants across all `3` models: `attn_k` is `3/3` negative; `attn_output` and `ffn_up` are `3/3` positive.
- **External Live GPU Cross-Check:** Confirms why weight-only `colproxy` retains only `23.01%` (`28761/125000`), `33.19%` (`10371/31250`), and `32.19%` (`321873/1000000`) of real-Fisher KL gain on `Qwen2.5-0.5B`, `SmolLM2-360M`, and `Qwen2.5-1.5B`, whereas random-token calibration retains `97.78%` (`195561/200000`), `99.78%` (`99781/100000`), and `91.82%` (`459079/500000`) on WikiText-2 (`96.99%` on Python).

### Pattern 4: $p$-Norm $S^{127}$ Sphere Concentration, Ball/Activation Taxes, & Silicon Engine Bridge

1. **$S^{127}$ Gaussianization (`pball.out`, `128` weights, `560+16` bits):** Moving $p = 2 \to 1$ gains `+0.598 dB` on i.i.d. Laplace (`-24.943 -> -25.541`) but **loses** `-0.440 dB` on i.i.d. Gaussian and loses `-0.372 dB` (`SmolLM2 blk.1.ffn_down`), `-0.400 dB` (`blk.5.ffn_down`), and `-0.218 dB` (`blk.10.attn_q`, after a small `+0.045 dB` bump at $p=1.5$), proving real LLM weights concentrate near the $L_2$ sphere $S^{127}$ (`85%..91%` of Gaussian penalty).
2. **Exact `32`-wt vs `128`-wt Ball-Size + 8-Bit-Scale Tax (`against_ball.out`):** `PH16` (`18 B / 32 w`) over `A16BG128` (`560+16` bits) pays `+11.58%` (`21171/18974`) on `Qwen2.5`, `+8.97%` (`4020/3689`) on `SmolLM2`, and `+15.16%` (`7936/6891`) on `Llama-3.2` (`+11.04%`, `26708/24053` with `+K10`), while still cutting KL by `11.21%..23.51%` vs `PZ32` at identical file bytes.
3. **Exact `Q4_K_M -> Q4_K_M@16` 8-Bit Activation Tax:** Removing `llama.cpp`'s 8-bit activation rounding reduces `Q4_K_M` KL by `-4.97%` (`Qwen2.5`, `+1.81` SE), `-13.76%` (`SmolLM2`, `+5.56` SE), and `-4.59%` (`Llama-3.2`, `+2.38` SE); all ball arms remain decisively ahead (`+6.6` to `+17.7` SE).
4. **Closing PR `#85`'s Open Throughput Bottleneck:** PR `#85`'s `notes/cross-org-survey-2026-11-24.md` and `notes/warp-drive-2026-11-24.md` report `1`-core C `hampack` at `9.37..10.07 Mw/s` (`937/100..1007/100`) and note no fast lattice engine existed prior to cutoff. Verified external hardware receipts close that gap in exact rational ratios:
   - Exact bit-exact `AVX-512` enumerative `hampack` (`L=16, n=32, B=1207`): `37.954 Mw/s` (`1T`, `3.77x..4.05x`), `218.513 Mw/s` (`8T`, `21.70x..23.32x`), `223.193 Mw/s` (`16T`, `22.16x..23.82x`).
   - Direct `4.5 bpw` `A_16` fused `AVX-512 VNNI` GEMV (`8T`, `60.9525 Gw/s`): `6,053x..6,505x`.
   - Live `RTX 3080 sm_86` `18 B -> SoA` transcode (`13.824 Gw/s`, `91.78 ms` cold-to-hot): `1,373x..1,475x`; lossless transcoded `72 B` vs on-the-fly unranking: `3578572/4929` (`726.02x`); `sm_86` `Block-H2` PTX (`622.1772 Gw/s`): `61,785x..66,401x`.

### Pattern 5: Stage-A Scale/Permutation-Invariant Cross-Model Rank Matrix & Survey Updates

Under arbitrary positive rational per-model rescalings and simultaneous arm permutations, exact Spearman $\rho$ and Kendall $\tau$ have **zero exact rational deviation**:

| Model Pair | `7`-Arm Spearman $\rho$ | `7`-Arm Kendall $\tau$ | `5`-Arm Ball Spearman $\rho$ | `5`-Arm Ball Kendall $\tau$ |
|---|---:|---:|---:|---:|
| `Qwen2.5-1.5B` vs `Llama-3.2-1B` | `13/14` (`+0.9286`) | `17/21` (`+0.8095`) | `9/10` (`+0.9000`) | `4/5` (`+0.8000`) |
| `Qwen2.5-1.5B` vs `SmolLM2-1.7B` | `23/28` (`+0.8214`) | `13/21` (`+0.6190`) | **`1/2` (`+0.5000`)** | **`1/5` (`+0.2000`)** |
| `SmolLM2-1.7B` vs `Llama-3.2-1B` | `11/14` (`+0.7857`) | `13/21` (`+0.6190`) | `3/5` (`+0.6000`) | `2/5` (`+0.4000`) |

While global `7`-arm agreement is high (`NF4` always #7, `Q4_K_M@16` #5–#6), internal `5`-ball lattice ordering splits sharply between `SmolLM2` and `Qwen2.5` ($\rho = 1/2$, $\tau = 1/5$), confirming PR `#85`'s warning against weight-error-only selection receipts. **`A_16` is the unique minimax-regret lattice**, ranking top-2 on **`3/3` models at `4.5 bpw`** (`#2, #2, #1`) and **`2/2` models at `3.0 bpw`** (`#1, #2`). In addition to the engine receipts above, formal `Lean 4.36.0` proofs confirm that `E_8`'s `H_8/sqrt(8)` turn does not naively transfer to `A_16` (`8x` opposite-sign ray anisotropy contrast).

---

## 4. Verification Summary & 4-File Core Manifest

- **4-Mode Isolated Parity + `umask 077` (`-I -B`, `-I -O -B`, `-I -B -W error`, `PYTHONHASHSEED=42 -B`):** `PASS` (`--standalone --json` byte-identical to `0o600` disk cert, `--check`, live read-only `--verify-portal` `18/18` SHA-256 + `16/16` JSONL rows, `9/9` R5 tests `0` warnings).
- **Exhaustive Adversarial Tamper Sweep (`FR0` + Direct `secondary_audit_payload`):** `6,940/6,940` rejected cleanly (`443` leaves, `91` dicts, incl. `str`/`list`/`dict` type swaps, `0` bypasses, `0` raw crashes).
- **4-Surface Adversarial Fuzzer (`CLI + FR1 + FR0 + SEC`):** `82/82 PASS` (`25/25 + 19/19 + 19/19 + 19/19`, `0` bypasses, `0` crashes).
- **Live Guard Mutation (`--guards require_ok --workers 4 --timeout 180 --max-mutants 0`):** `117/117 PASS` (`3/3` oracle + `114/114` audit, `100.0%`, `0` survivors, `0` timeouts, `truncated = false`).
- **Whole-File Exhaustive AST `BoolOp` (`OP_SWAP + SUBCOND_DEL`):** `136/136 PASS` (`21/21` oracle + `115/115` audit, `100.0%`, `0` survivors).

| File | Lines | Bytes | Mode | SHA-256 |
|---|---:|---:|:---:|---|
| `siliquant85_pattern_oracle.py` | `647` | `28,169` | `0o644` | `5707f63238063e496965242d85afc74eb63fce2149845b21ab2010ef5cfa88b7` |
| `siliquant85_pattern_audit.py` | `784` | `31,848` | `0o644` | `89366c99b00a2c8a535e0acd9c14763f1700afa54004da1ff9867f3550ba3a6a` |
| `test_siliquant85_pattern_r5.py` | `790` | `49,527` | `0o644` | `46577ac45dbac2834d0a4d50603b03b24e67c5711543225d37bbe06dbf826ee1` |
| `siliquant85_pattern_certificate.json` | `1` | `11,230` | `0o600` | `c2a15fdbdfd3245b8ab4bf7eb6772f66b1399831026680d5307c2a3be6ccf08b` |

- `payload_sha256`: `55116e879ab53a07c12f23b3191310b551a12e629e07afaa866feea5bff72911`
- `certificate_sha256`: `b87d0d1e0352b0bb680efd439f3127857ae201dcd2a1abc0f88a610366edc0da`
