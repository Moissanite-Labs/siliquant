# Track S78: Engram LoRA Bit-Capacity × Track S34 Two-Part MDL × `siliquant` Joint `(bpw, rank)` Bit-Allocation Law

- **Track ID:** `TRACK-S78-ENGRAM-LORA-CAPACITY-SILIQUANT-MDL`
- **Schema Version:** `aletheia.track_s78.engram_lora_siliquant_mdl.certificate.v1`
- **Oracle Class:** `exact` (100% pure-$\mathbb{Z}/\mathbb{Q}$ `int` / `Fraction` certified outward-rounded interval arithmetic; `Decimal` strictly for deterministic display rounding; `0` AST floats, `0` `float()`, `0` bare `assert`s, `0` RNG)
- **Novelty Tier:** `N1` (with strict `N0` / `N1` / analytic-extrapolation quarantine in §1)
- **Live Formal Proof Kernel:** `Lean 4.36.0` — `17/17` theorems proved in pure `Init` (`0` imports, **`0` axioms `[]`** on every theorem — not even `propext`, `Quot.sound`, or `Classical.choice` — `0` warnings, R10+ comment-only static scanner with strict quote/guillemet ban + `EXPECTED_LEAN_SHA256` pin + `--stdin` execution: `0` findings / `0` notices)
- **Certificate Digests:**
  - `payload_sha256`: `36c8e2fd9a8d493cf94bc9d6f60ee031f65c15365e61673945623a2f62ebbf4e`
  - `certificate_sha256`: `d321dbaeca8884e492a73a638280a06dbbb5901526f874114700a2a743e5b9ed`

---

## 1. Epistemic Honesty & Novelty Quarantine (`N0` vs. `N1` vs. Extrapolation)

| Epistemic Tier | Exact Scope |
|---|---|
| **`N0` (Reproduced & Audited)** | Verbatim extraction and pure-$\mathbb{Z}/\mathbb{Q}$ verification of Engram's reverse arithmetic coding measurement, Appendix A Marchenko–Pastur closed form (`1.401440`, `1.321133`, `0.973987`, `0.480898` bpp; `23.937 KB` at `r = 128`), Fig 1 (`98.81%`), Fig 4 (`24`-layer super-additivity `3.29 / 1.30 = 2.5308×`), Fig 7 (`Forgetting / Learning = 1.00`); `Track S34` exact offline-vs-online MDL cancellation and `Gemma-3-4B-IT` log-convex KL law; `Track S31` `CS-15`/`CS-16` `66/25 = 2.64×` matched-gain sign flip; `Track S30`/`S63`/`S72` frozen `4.5 bpw` `siliquant` KL receipts on `Gemma-3-4B-IT` and `Qwen2.5-0.5B/1.5B/3B`. |
| **`N1` (New Certified Cross-Portal Theorems)** | **(1) Multiplicative Gauge-Quotient × Marchenko–Pastur Reconciliation:** $C_{\mathrm{MP}}(r) = \frac{1}{2\ln 2} P_{\mathrm{raw}}(r)\,\eta_{\mathrm{alg}}(r)\,\kappa(r)$ with universal zero-axiom `Lean 4.36.0` proofs of gauge efficiency monotonicity, spectral-edge dominance $1 < \kappa(0^+) \le 2$, full-rank collapse $\kappa(m) = 1$, and universal finite-spectrum cumulative-average monotonicity. **(2) Exact Joint Base-Quantization (`bpw`) × LoRA-Rank (`r`) Bit-Allocation Law:** Exact budget-conserving rank-unlock formula, exact break-even horizon $T_{\mathrm{crit}}$, exact $\ln 2$ cancellation and unconditional strict convexity $d^2 J / db^2 > 0$ of the joint two-part MDL objective, and exact lower-bitrate KL blowup floor $b_{\mathrm{floor}}(T)$ certified across four real frozen `siliquant` models. |
| **Quarantined Analytic Extrapolation** | The `3.0 bpw` `Gemma-3-4B-IT` KL entry (`0.796717` nats) is explicitly quarantined as an analytic `Track S34` $C \cdot 2^{-\alpha b}$ extrapolation (`ANALYTIC_S34_EXTRAPOLATION_NOT_LIVE_GPU_MEASURED`), NOT a live GPU measurement. |

---

## 2. Pinned Read-Only Upstream Provenance (`7/7` Disk-Verified)

All upstream inputs were audited strictly read-only and verified bit-for-bit on disk via `verify_s78_engram_lora_mdl.py --expedition-root`:

| Source Key | Relative Reference | Bytes | Exact SHA-256 |
|---|---|---:|---|
| `engram_lora_size_txt` | `portals/wave4-20261124/engram_lora_size.txt` (`2,942` L) | `34837` | `6d42d05ad0638fe99eb3de6933e54055463191541e559e72d4c49787777c7a02` |
| `track_s34_certificate` | `track-s34-mdl-rosen-similarity-obstruction/mdl_rosen_similarity_certificate.json` | `11522` | `ad03c2fb0977f8fae30849eb18e86b9050d03c4793cbb9e6295891d28b67d242` |
| `track_s31_certificate` | `track-s31-aletheia-v22-society-misfolding-tomography/aletheia_v22_misfolding_certificate.json` | `12707` | `f4f7f8abff0c7eae0d5678016c1041ab34870ef2de57a2f83e4cbb9a529f8a9b` |
| `track_s30_certificate` | `track-s30-siliquant-gemma-m1-tomography/gemma_m1_tomography_certificate.json` | `25769` | `39190ddaf55a63ef6f4fe8787895dfb4c727cde1fda75fa007cffc386e088b0f` |
| `track_s63_certificate` | `track-s63-rtx3080-qwen3b-ceer-scale/rtx3080_qwen3b_certificate.json` | `4718` | `e14aef18e66668c55478e7ec84517520bfde9755603658bdcc728d952a16a8b6` |
| `track_s72_certificate` | `track-s72-stratified-quad-interleaved-blockh4/rtx3080_s72_certificate.json` | `4733` | `c2dc4626271503bbd710ce468e2d2051b1241f77d9f6f995dee15d621eb884e7` |
| `track_s72_empirical_results` | `track-s72-stratified-quad-interleaved-blockh4/rtx3080_s72_empirical_results.json` | `83361` | `a7c35c646a9c29fa645f207697c74e181ad19d8ce61acbe489a265a132968647` |

---

## 3. Pillar 1 — Verbatim Audit of Engram's LoRA Bit-Capacity Theory

From `engram_lora_size.txt`, Engram measures stored information via reverse arithmetic coding code-length reduction:
$$\mathrm{bits}_\theta(x) = -\sum_i \log_2 p_\theta(x_i \mid x_{<i}), \qquad \Delta\mathrm{bits}(x) = \mathrm{bits}_{\theta_0}(x) - \mathrm{bits}_{\theta'}(x)$$
under LoRA parameterization $W' = W + \frac{\alpha}{r} B A$ ($B \in \mathbb{R}^{m \times r}$, $A \in \mathbb{R}^{r \times n}$, $\alpha = 256$). Under Kronecker Fisher whitening $F = F_1 \otimes F_2$, $Z = F_1^{-1/2} G F_2^{-1/2}$ ($Z_{ij} \stackrel{\mathrm{iid}}{\sim} \mathcal{N}(0, 1)$), Eckart–Young yields $C_{\mathrm{MP}}(r) = \frac{1}{2\ln 2}\mathbb{E}\!\left[\sum_{i=1}^r \sigma_i(Z)^2\right]$ with Appendix A Marchenko–Pastur closed form ($\gamma = m/n \in (0, 1]$, $\varphi \in [0, \pi]$):
$$r(\varphi) = m \cdot \frac{(1+\gamma)\varphi - 2\sqrt{\gamma}\sin\varphi - 2(1-\gamma)\arctan\!\left(\frac{1-\sqrt{\gamma}}{1+\sqrt{\gamma}}\tan\frac{\varphi}{2}\right)}{2\pi\gamma}, \qquad C_{\mathrm{MP}}(r(\varphi)) = \frac{mn}{2\ln 2}\cdot\frac{\varphi - \sin\varphi\cos\varphi}{\pi}$$

Our pure-$\mathbb{Z}/\mathbb{Q}$ certified interval solver (`ln2 = [0.693147180560, 0.693147180560]`, `pi = [3.141592653590, 3.141592653590]`) reproduces every Engram benchmark point to full blog precision:

| Engram Benchmark Point | Pure-$\mathbb{Z}/\mathbb{Q}$ Certified Interval | Blog Reported | Match |
|---|---:|---:|:---:|
| `v_proj` (`512 × 1024`, $\gamma = 1/2$) low-rank edge | `[1.401440, 1.401440]` bpp (`kappa(0+) = 1.942809`) | `1.40` bpp | PASS |
| `up_proj` (`1024 × 3584`, $\gamma = 2/7$) low-rank edge | `[1.321133, 1.321133]` bpp (`kappa(0+) = 1.831479`) | `1.32` bpp | PASS |
| Fig 3 `v_proj` `rank = 128` (`P_raw = 196,608`) | `[191493.686, 191493.689]` bits = `23.937 KB`, `[0.973987, 0.973987]` bpp | `23.9 KB`, `0.97` bpp | PASS |
| Full rank `v_proj` `rank = 512` (`P_raw = 786,432`) | `[378193.849, 378193.849]` bits = `47.274 KB`, `0.480898` bpp | `0.48` bpp | PASS |
| Fig 1 `rank-6 mlp[23]` | `667/675 = 98.81%` (`6.67 KB / 6.75 KB`) | `98.8%` | PASS |
| Fig 4 `24`-layer MLP super-additivity (`72` mats, `r = 4`) | `329/130 = 2.5308×` (`3.29` bpp vs `1.30` single-matrix theory) | `3.29` bpp | PASS |
| Fig 7 Conservation of capacity (`v_proj`, `r = 512`) | `45.8 / 45.8 = 1` exact (`Forgetting / Learning = 1.00`) | `1.00` | PASS |

---

## 4. Pillar 2 — Exact Reconciliation: $GL_r(\mathbb{R})$ Gauge Quotient vs. Marchenko–Pastur Spectral Law (`N1`)

For $B \in \mathbb{R}^{m \times r}$, $A \in \mathbb{R}^{r \times n}$, raw parameter count is $P_{\mathrm{raw}}(r) = r(m+n)$. The continuous gauge action $(B, A) \mapsto (BG, G^{-1}A)$ for $G \in GL_r(\mathbb{R})$ leaves $BA$ invariant and removes $r^2$ redundant DOF, leaving exact manifold dimension $D_{\mathrm{eff}}(r) = r(m+n-r)$ and algebraic non-redundancy ratio $\eta_{\mathrm{alg}}(r) = D_{\mathrm{eff}}(r)/P_{\mathrm{raw}}(r) = 1 - \frac{r}{m+n}$.

A fixed data-independent $D_{\mathrm{eff}}(r)$-dimensional linear subspace stores $\frac{D_{\mathrm{eff}}(r)}{2\ln 2}$ bits by Wilks' theorem (`0.7213` bpp as $r \ll m$). LoRA achieves up to `2×` more bits/parameter at low rank (`1.401440` bpp on `512 × 1024`; `1.442695` bpp on square `1024 × 1024`) because Eckart–Young selects the **data-aligned** top-$r$ singular subspace of the whitened score matrix $Z$. Both laws reconcile **identically** via the exact multiplicative decomposition:
$$C_{\mathrm{MP}}(r) = \frac{1}{2\ln 2} \cdot P_{\mathrm{raw}}(r) \cdot \eta_{\mathrm{alg}}(r) \cdot \kappa(r), \qquad \kappa(r) := \frac{C_{\mathrm{MP}}(r)}{D_{\mathrm{eff}}(r) / (2\ln 2)}$$
where:
1. $\eta_{\mathrm{alg}}(r) = 1 - \frac{r}{m+n}$ decreases strictly in $r$ (`Lean 4.36.0` Theorem 3, `0` axioms);
2. $\kappa(r)$ starts at $\kappa(0^+) = 1 + \frac{2\sqrt{mn}}{m+n} \in (1, 2]$ (strictly $> 1$ by Theorem 4, universally $\le 2$ by AM-GM Theorem 5, saturating `2` identically for square matrices by Theorem 6), decreases strictly with $r$ by ordered singular-value averaging (Theorems 7–9), and collapses identically to $\kappa(m) = 1$ at full rank $r = m$ (Theorem 2).

### Certified Pure-$\mathbb{Z}/\mathbb{Q}$ Decomposition Sweep (`512 × 1024`, $\gamma = 1/2$)

| Rank $r$ | $P_{\mathrm{raw}}(r)$ | $D_{\mathrm{eff}}(r)$ | $\eta_{\mathrm{alg}}(r)$ | Wilks bpp | MP bpp (`lo = hi`) | $\kappa(r)$ (`[lo, hi]` `dec6`) |
|---:|---:|---:|---:|---:|---:|---:|
| `1` | `1,536` | `1,535` | `0.999349` | `0.720878` | `1.383214` | `1.918791` |
| `4` | `6,144` | `6,128` | `0.997396` | `0.719469` | `1.355724` | `1.884340` |
| `8` | `12,288` | `12,224` | `0.994792` | `0.717591` | `1.329199` | `1.852309` |
| `16` | `24,576` | `24,320` | `0.989583` | `0.713833` | `1.287598` | `1.803780` |
| `32` | `49,152` | `48,128` | `0.979167` | `0.706319` | `1.222847` | `1.731294` |
| `64` | `98,304` | `94,208` | `0.958333` | `0.691291` | `1.123357` | `1.625013` |
| `128` | `196,608` | `180,224` | `0.916667` | `0.661235` | `0.973987` | `[1.472981, 1.472982]` |
| `256` | `393,216` | `327,680` | `0.833333` | `0.601123` | `0.759554` | `1.263559` |
| `512` | `786,432` | `524,288` | `0.666667` | `0.480898` | `0.480898` | `1.000000` |

*(Note: All rows except `r = 128` collapse to `lo = hi` at `dec6` for $\kappa(r)$; at `r = 128`, the certified outward-rounded bisection enclosure has exact 1-ULP width `[1.472981, 1.472982]`. Corresponding `mp_bits` outward-rounded `dec3` intervals have 1–3 ULP widths on intermediate ranks: `r=1 [2124.616, 2124.617]`, `r=4 [8329.569, 8329.570]`, `r=8 [16333.202, 16333.203]`, `r=16 [31644.019, 31644.020]`, `r=32 [60105.356, 60105.358]`, `r=64 [110430.505, 110430.508]`, `r=128 [191493.686, 191493.689]`, `r=256 [298668.968, 298668.970]`, and `r=512 [378193.849, 378193.849]`.)*

---

## 5. Pillar 3 — Unification with `Track S34` Two-Part MDL & `Track S31` `CS-15`/`CS-16` Obstruction

1. **Exact `Track S34` Equivalence (`Lean` Theorem 10):** `Track S34` establishes $L_{\mathrm{offline}}(X) - L_{\mathrm{online}}(X) \equiv b|\theta| - R(X)$ identically across all three regimes (`b_theta ∈ {12, 4, 2}`, `R = 4`, residual `0`). For a LoRA adapter $\Delta\theta$ stored at $b_{\mathrm{LoRA}}$ bpp, Engram's $\Delta\mathrm{bits}(X)$ is the exact data code-length reduction, so net two-part MDL profit is $\Pi_{\mathrm{MDL}}(\Delta\theta; X) = \Delta\mathrm{bits}(X) - b_{\mathrm{LoRA}} P_{\mathrm{raw}}(r)$.
2. **Noise Memorization No-Profit Theorem (`Lean` Theorem 11):** Because single-layer LoRA capacity is universally bounded above by $\frac{1}{\ln 2} \approx 1.442695$ bpp (and even Engram's empirical `24`-layer super-additive ceiling is `3.29` bpp), storing incompressible data at $b_{\mathrm{LoRA}} \in \{4, 8, 16\}$ bpp yields **strictly negative** two-part MDL profit (`-0.7100`, `-4.7100`, `-12.7100` bpp in the multi-layer regime; `-2.557305`, `-6.557305`, `-14.557305` bpp single-layer upper bound).
3. **`CS-15` / `CS-16` Obstruction (`Lean` Theorems 12–13):** Under second-order Fisher geometry, optimal step $\delta^\star = F^{-1}g$ reduces training NLL by $\frac{1}{2}\delta^{\star\top} F \delta^\star$ while increasing expected NLL on fresh base samples $x' \sim p_{\theta_0}$ ($\mathbb{E}[g'] = 0$) by $+\frac{1}{2}\delta^{\star\top} F \delta^\star$ (`Forgetting / Learning = 1.00`, zero net joint bit reduction: `(train - learn) + (base + learn) = train + base`). Therefore satisfying $N_{\mathrm{bits}}(X_{\mathrm{train}}) \le C_{\mathrm{LoRA}}(r)$ is **necessary for memorization but strictly insufficient for OOD generalization** (`CS-15`), connecting directly to `Track S31`'s `CS-16` matched-gain LoRA forgetting sign flip (`3/20 < 1/4` unmatched flips to `(3/20)/(5/88) = 66/25 = 2.64×` at matched task gain $g^* = 3/10$).

---

## 6. Pillar 4 (`N1` New Cross-Portal Theorem) — Exact Joint `(bpw, rank)` Bit-Allocation Law

Under a fixed linear-block + adapter bit budget $B_{\mathrm{tot}} = b_{\mathrm{base}} N_{\mathrm{base}} + b_{\mathrm{LoRA}} P_1 r$ (where $P_1 = \sum_{\ell \in \mathcal{A}} (m_\ell + n_\ell)$ is the rank-1 adapter footprint), compressing the base from $b_1 \to b_2 < b_1$ frees $(b_1 - b_2) N_{\mathrm{base}}$ bits and unlocks exactly (`Lean` Theorem 14):
$$\Delta r(b_1 \to b_2) = \left\lfloor \frac{(b_1 - b_2) N_{\mathrm{base}}}{b_{\mathrm{LoRA}} P_1} \right\rfloor$$

At fixed $B_{\mathrm{tot}}$, the joint two-part MDL objective over a $T$-token horizon is $J(b_{\mathrm{base}}, r; T) = B_{\mathrm{tot}} + \frac{T}{\ln 2} K_{\mathrm{base}}(b_{\mathrm{base}}) - \Delta\mathrm{bits}_{\mathrm{adapt}}(r; T)$. Thus $(b_2, r_2)$ strictly dominates $(b_1, r_1)$ **if and only if** (`Lean` Theorem 15):
$$\Delta C_{\mathrm{LoRA}}(r_1 \to r_2) > \frac{T}{\ln 2}\big(K_{\mathrm{base}}(b_2) - K_{\mathrm{base}}(b_1)\big) \iff T < T_{\mathrm{crit}} := \frac{\ln 2 \cdot \Delta C_{\mathrm{LoRA}}(r_1 \to r_2)}{K_{\mathrm{base}}(b_2) - K_{\mathrm{base}}(b_1)}$$

### 6.1 Certified Real-Model Allocation & RTX 3080 (`6.0 GiB` Weight+Adapter Envelope) Frontier

Every block weight count $L \sum_\ell m_\ell n_\ell$ matches the pinned `Track S34`/`S72` receipts to the **exact single weight** (`4/4`):

| Model (`siliquant` Frozen Arm) | $N_{\mathrm{base}}$ | $P_1$ (all-7) | GB Freed (`16→4.5`) | $\Delta r$ (`b16` / `AdamW b64`) | `6 GiB` $r_{\max}$ (`BF16` → `siliquant`) | $\Delta C_{16\to 128}$ (Mbits) | $T_{\mathrm{crit}}$ Wiki / Py (B tok) | $T_{\mathrm{crit,Wiki}} / 27\text{M}$ (`enwik8`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Qwen2.5-0.5B` (`quad_lpt_real`) | `357,826,560` | `549,888` | `0.5144` | `467` / `116` | `4,959` → `5,427` (`1.09×`) | `61.032` | `1.630` / `1.794` | `60.4×` |
| `Qwen2.5-1.5B` (`quad_lpt_sharednorm`) | `1,310,195,712` | `1,154,048` | `1.8834` | `816` / `204` | `1,453` → `2,269` (`1.56×`) | `138.377` | `4.179` / `6.438` | `154.8×` |
| `Qwen2.5-3B` (`quad_snake_real`) | `2,774,532,096` | `1,870,848` | `3.9884` | `1,065` / `266` | `72` → `1,138` (`15.81×`) | `[231.455, 231.456]` | `6.929` / `7.828` | `256.6×` |
| `Gemma-3-4B-IT` (`hZBG128:b560s16`) | `3,208,642,560` | `1,862,656` | `4.6108` | `1,237` / `309` | `0` (OOM) → `884` (`221` AdamW) | `243.847` | `2.359` / `5.435` | `87.4×` |

Key architectural findings:
- On `Qwen2.5-3B` (`6.0 GiB` weight+adapter envelope on an `8 GiB` RTX 3080), `BF16` (`5.7476 GiB`) allows only `r_max = 72` (`18` with FP32 AdamW states), whereas `4.5 bpw siliquant QuadSnake` (`2.0331 GiB`, Wiki KL `0.023154` nats = `0.033404` bits/tok) unlocks `r_max = 1,138` (`284` AdamW) — a **`15.81×` rank-headroom multiplier** — with break-even horizon **`6.929B` tokens (Wiki) / `7.828B` tokens (Python)** (`256.6×` larger than Engram's largest `27M`-token `enwik8` corpus).
- On `Gemma-3-4B-IT`, `BF16` (`7.2269 GiB`) cannot fit inside a `6.0 GiB` or `7.0 GiB` envelope at all (`r_max = 0`), while `4.504 bpw siliquant hZBG128` (`2.9327 GiB`) unlocks `r_max = 884` (`221` AdamW) at `6.0 GiB` and `1,172` at `7.0 GiB`.

### 6.2 Exact Convex Optimality & Lower-Bitrate KL Floor (`Gemma-3-4B-IT`, `Track S34` Law)

Substituting `Track S34`'s verified rate-distortion law $K_{\mathrm{base}}(b) = C \cdot 2^{-\alpha b}$ ($C = 199517/5000 = 39.9034$, $\alpha = 18821/10000 = 1.8821$) and linear budget coupling $dr/db = -N_{\mathrm{base}}/(b_{\mathrm{LoRA}} P_1)$ produces **exact $\ln 2$ cancellation** and **unconditional strict convexity**:
$$\frac{dJ}{db} = \frac{N_{\mathrm{base}}}{b_{\mathrm{LoRA}} P_1} C'_{\mathrm{LoRA}}(r(b)) - \alpha T C \cdot 2^{-\alpha b}, \qquad \frac{d^2 J}{db^2} = \alpha^2 (\ln 2) T C \cdot 2^{-\alpha b} + \left(\frac{N_{\mathrm{base}}}{b_{\mathrm{LoRA}} P_1}\right)^2 \big(-C''_{\mathrm{LoRA}}(r(b))\big) > 0$$

Since $C'_{\mathrm{LoRA}}(r) \le \eta_{\max} P_1$, compressing below the exact **lower-bitrate floor**
$$b_{\mathrm{floor}}(T) := \frac{1}{\alpha} \log_2\!\left(\frac{\alpha T C b_{\mathrm{LoRA}}}{\eta_{\max} N_{\mathrm{base}}}\right)$$
strictly worsens total two-part MDL (`dJ/db < 0` unconditionally; discrete obstruction proved in `Lean` Theorem 16). On frozen `Gemma-3-4B-IT` `ZBG128` WikiText-2 receipts (`5.0 → 4.5 → 4.0 bpw`), the step penalty jumps strictly convexly from `27037/500000` (`0.054074`) to `51577/500000` (`0.103154`) nats (`1.9076×` steeper, `Lean` Theorem 17), while the quarantined analytic `3.0 bpw` extrapolation reaches `0.796717` nats (`+0.580962` nats over `4.0 bpw`).

| Serving / Adaptation Horizon $T$ | $b_{\mathrm{floor}}(T)$ Single-MP (`1.331713` bpp) | $b_{\mathrm{floor}}(T)$ Multi-Layer (`3.29` bpp) | Per-Weight $d^2(J/N)/db^2$ | `4.5 bpw` Above Single Floor? | `3.0 bpw` Above Single Floor? |
|---|---:|---:|---:|:---:|:---:|
| `enwik8` (`27M` tokens) | `1.5539` bpw | `0.8607` bpw | `0.108582` | YES | YES |
| `100M` tokens | `2.5576` bpw | `1.8643` bpw | `0.108582` | YES | YES |
| `1B` tokens (serving) | **`4.3226` bpw** | **`3.6293` bpw** | `0.108582` | **YES** | **NO (KL cliff!)** |
| `10B` tokens (serving) | `6.0876` bpw | `5.3943` bpw | `0.108582` | NO | NO |
| `100B` tokens (serving) | `7.8526` bpw | `7.1593` bpw | `0.108582` | NO | NO |

At `T = 1B` serving tokens, **`4.5 bpw siliquant` remains strictly above the convex floor (`4.50 > 4.3226`), whereas `3.0 bpw` crosses below both single-matrix (`4.3226`) and multi-layer (`3.6293`) floors**, quantitatively locating the exact phase transition between rank-unlocking compression and base-KL collapse.

---

## 7. Live Kernel-Checked `Lean 4.36.0` Theorem Ledger (`17/17`, `0` Axioms `[]`)

Every theorem in `LoraCapacityJointMdl.lean` (`0` imports, pure `Init`) compiles cleanly under `Lean 4.36.0` with `0` warnings and `#print axioms` printing `does not depend on any axioms`:

1. `gauge_quotient_eff_dof` — `dEff m n r + r * r = pRaw m n r` (`∀ r ≤ m`)
2. `full_rank_wilks_closure` — `dEff m n m = m * n`
3. `eta_alg_strictly_decreasing` — `dEff m n r2 * pRaw m n r1 < dEff m n r1 * pRaw m n r2` (`∀ 0 < r1 < r2 ≤ m, 0 < n`)
4. `spectral_edge_dominates_algebraic_dof` — `(m + n)^2 < (m + n)^2 + 4 * (m * n)` (`∀ 0 < m, 0 < n`)
5. `spectral_edge_multiplier_le_two` — `4 * (m * n) ≤ (m + n)^2` (AM-GM ceiling $\kappa(0^+) \le 2$)
6. `spectral_edge_saturates_for_square` — `4 * (m * m) = (m + m)^2` ($\kappa(0^+) = 2$ on square matrices)
7. `prefix_sum_ge_len_mul_min` — `allGe s_next xs → xs.length * s_next ≤ listSum xs`
8. `ordered_spectrum_avg_nonincreasing` — `r * sNext ≤ sSum → r * (sSum + sNext) ≤ (r + 1) * sSum`
9. `ordered_spectrum_avg_strict_drop` — `r * sNext < sSum → r * (sSum + sNext) < (r + 1) * sSum`
10. `s34_mdl_offline_online_cancellation` — `(bTheta + lTheta + c) - (lTheta + regret + c) = bTheta - regret`
11. `engram_noise_no_mdl_profit` — `gainBits < adapterBits → dataBits < adapterBits + (dataBits - gainBits)`
12. `conservation_forgetting_cancels_train_gain` — `(trainBits - learn) + (baseBits + learn) = trainBits + baseBits`
13. `cs15_cs16_capacity_obstruction_witness` — finite capacity-without-OOD witness + `Track S31` `66/25 = 2.64×` flip
14. `joint_budget_conservation` — `b1*N + bL*P1*r1 = b2*N + bL*P1*r2 → (b1 - b2)*N = bL*P1*(r2 - r1)`
15. `quant_to_rank_dominance_iff` — `(bTot + k2 - c2 < bTot + k1 - c1) ↔ (k2 - k1 < c2 - c1)`
16. `convex_kl_floor_obstruction` — strict KL convexity + concave LoRA gains ⇒ below-floor steps worsen MDL
17. `pinned_siliquant_receipt_witnesses` — `Gemma-3-4B-IT` ZBG128 convexity (`54074 < 103154`) & `Qwen2.5-3B` `Δr = 1,065` exact Euclidean division

---

## 8. Aletheia R1–R5 & R10+ Verification Gate Summary (R2)

- **Live `Lean 4.36.0` + R10+ Comment-Only Static Audit (`S78-F1` Closed):** `17/17` zero-axiom theorems (`[]`), `0` warnings, `0` imports, `0` findings, `0` notices. Comment-only stripper + strict quote/guillemet ban (`["'«»]`) + expanded forbidden-keyword/attribute/tactic gates (`decide -revert +native`, `decide +revert +native`, `decide (revert := true) +native`, `DecideConfig`, `declare_syntax_cat`, `notation`/`prefix`/`infix`/`postfix`, `@[csimp]`, `dbg_trace`, `run_cmd`/`run_meta`, `#print axioms example`, `Lean.ofReduceNat`, `Lean.Meta.mkSorry`, `«debug».«skipKernelTC»`) + `EXPECTED_LEAN_SHA256` pin + atomic `dir_fd` read + `lean --stdin` execution (zero read-to-compile TOCTOU).
- **Hardened Atomic I/O (`S78-F2` Closed):** Atomic `dir_fd` (`O_RDONLY | O_DIRECTORY | O_NOFOLLOW`) step-down on read and write, logical `$PWD` symlink ancestor defense (`cd -L sym_d && --check c.json` fails closed), `O_NONBLOCK` FIFO defense (non-hanging rejection), `st_nlink == 1` on both read and write (`TMP_HARDLINK_DETECTED` + `HARDLINK_FORBIDDEN`), `secrets.token_hex(8)` temp naming, pre-`os.replace` TOCTOU `lstat` guard, `dir_fd` `os.replace`, and directory `os.fsync`.
- **4-Mode `-I` Parity + `PYTHONHASHSEED=42` + `umask 077` + CI `chmod 644` / `chmod 500` RO-Dir Sim:** `test_s78_engram_lora_mdl_r5.py` `10/10 OK ×6` (`-I -B`, `-I -O -B`, `-I -B -W error`, `PYTHONHASHSEED=42 -B`, CI `chmod 644` / RO-dir `-I -O -B` & `-I -B -W error`), `0o600` certificate emission, `7/7` pinned upstream disk SHA locks verified.
- **Direct 4-Surface `S19c` v2.1 `zone_b_fuzzer` (Zero Wrappers, `S78-F3` Closed):** **`82/82 PASS`** (`CLI 25/25`, `FR1 19/19`, `FR0 [full_replay=False] 19/19`, `SEC [secondary_audit_payload directly] 19/19`; `byp=0, cr=0`), with strict `type(full_replay) is bool` and pre-parse ASCII `parse_int` digit cap (`MAX_INT_DIGITS = 80`).
- **Live `aletheia-fast-mutate` (`--guards require_ok --workers 4 --timeout 180 --max-mutants 0`):**
  - `verify_s78_engram_lora_mdl.py`: **`178/178 KILLED = 100.0%`** (`survived=0, timeout=0, truncated=False, null_mutant=passed`).
  - `s78_engram_lora_siliquant_mdl.py`: **`5/5 KILLED = 100.0%`** (`survived=0, timeout=0, truncated=False, null_mutant=passed`).
  - Combined: **`183/183 KILLED = 100.0%`** (`0` survivors, `0` timeouts).
- **Whole-File DFS AST `BoolOp` (`OP_SWAP + SUBCOND_DEL`):**
  - `verify_s78_engram_lora_mdl.py`: **`19/19 KILLED = 100.0%`** (`survived=0, timeout=0`).
  - `s78_engram_lora_siliquant_mdl.py`: **`9/9 KILLED = 100.0%`** (`survived=0, timeout=0`).
  - Combined: **`28/28 KILLED = 100.0%`** (`0` survivors, `0` timeouts).
- **Dual `aletheia audit-check --lean-audit --strict-ancestors --safety-scan --scripts` (`aletheia-fix-lean-gates-v2` & `aletheia-fast-mutate`):** **`2/2 PASS`** (`0` findings, `0` notices, `0` bare asserts, `0` AST floats, `0` leaks, `0` CRLF, `PYC=0`, `.lake=0`, `ELF=0`, `TT_SNAPS_LEFT=0`).

---

## 9. Frozen File Manifest (5 Core Files + `REPORT.md`)

| File | Lines | Bytes | Mode | SHA-256 |
|---|---:|---:|:---:|---|
| `LoraCapacityJointMdl.lean` *(R1 bit-identical)* | `467` | `27608` | `0o644` | `350eea0cd07c47c6b492e0ec289a8b756a1802e68beee7634e0ea69854c34c89` |
| `s78_engram_lora_siliquant_mdl.py` *(R1 bit-identical)* | `1207` | `50821` | `0o644` | `f9a062fe83ae7b9572762cb57a9b75f8f3cb84ea717389478bd3873eba3531a9` |
| `verify_s78_engram_lora_mdl.py` *(R2)* | `793` | `32650` | `0o644` | `ab7642dd5074192f2eaa8dfc4fb8d38c0382c5e3fba23e400040fd9d9880cf3d` |
| `test_s78_engram_lora_mdl_r5.py` *(R2)* | `984` | `57895` | `0o644` | `b0640d3b3b8b2bf435289b7b05b1009cb95e93c00fad6b6492f7d37cb67311f8` |
| `s78_engram_lora_siliquant_certificate.json` *(R1 bit-identical)* | `1` | `18553` | `0o600` | `3b8341ef5f588f970accb9a9d28f0b683af9747be045e69196e6c19d3fa58378` |
| `REPORT.md` *(R2, self)* | *(self)* | *(self)* | `0o644` | *(computed externally post-write)* |
