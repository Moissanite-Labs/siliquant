# AUDIT-107 — Exact Pure-Q Operator Bridge Audit of `laya` & `vegaml` (`Aletheia v2.1`, `siliquant`, & `warp-drive`)

- **Audit ID:** `AUDIT-107` (Track S10)
- **Schema Version:** `audit107.laya_vegaml.v1`
- **Honest Law-4 Novelty Classification:** **`N1`** (new exact pure-$\mathbb{Q}$ executable certifier, live AST/code-bug recount, and fail-closed operator bridge over **`N0`** classical split-conformal, Wilson score, meet-semilattice, and linear ridge/Wiener operators)
- **Pinned Read-Only Primary Sources (`0` edits, `0` external writes):**
  - `NandhaKishorM/laya` @ `e150aa92a0d4059ac621875b39b620847cf1b88a` (`laya/evidence.py`, `laya/vega.py`, `laya/confidence.py`, `laya/router.py`, `docs/guardrail-limitations.md`)
  - `NandhaKishorM/vegaml` @ `5e7ffad546b006ccc316bb46c02df40e9200713d` (`vegaml/calibrate.py`, `vegaml/band.py`, `vegaml/router.py`, `vegaml/_attach.py`, `vegaml/_universal.py`, `vegaml/_common.py`)
  - `unmerged_prs_snapshot.json` — read-only pinned metadata for `8` Law-1b unmerged PRs (`laya#1524`, `#1519`, `#1517`, `#1502`, `#1390`, `#1388`; `vegaml#57`, `#40`)
- **Executable Artifacts (100% Pure Python 3 Stdlib, Exact `int`/`Fraction`, `0` Floats, `0` Bare `assert`s):**
  - [laya_vegaml_oracle.py](laya_vegaml_oracle.py) — exact pure-$\mathbb{Q}$ oracle across all 4 pillars.
  - [laya_vegaml_audit.py](laya_vegaml_audit.py) — dual-mode (`--standalone` default offline replay + `--live` read-only intake recount) runner with Zone-B-hardened `validate_audit_certificate()`.
  - [test_laya_vegaml_r5.py](test_laya_vegaml_r5.py) — assertion-free `-I -O -B -W error` R5 mutation-killing, AST-conjunct-killing, and anti-bypass test suite.
  - [laya_vegaml_certificate.json](laya_vegaml_certificate.json) — canonical zero-float live JSON certificate (`live_intake_audited: true`, `overall_verified: true`).

---

## 1. Law-5 De-Metaphorization Table (Stage-B Exact Operators Only)

Stripping the *"particle rolling into a Gaussian potential valley under damped Hamiltonian dynamics with a selective-SSM thermostat"* narrative in `vegaml/vegaml/_common.py:3-18, 903-942` leaves exact deterministic operators:

| Upstream Narrative Term (`vegaml` / `laya`) | Exact Stage-B Mathematical Operator | Target Bridge |
|---|---|---|
| *"Particle rolling into potential wells"* (`_common.py:903-942`) | Damped symplectic discrete-time recurrent residual map $(z_{t+1}, p_{t+1}, h_{t+1})$ with `softplus` friction, diagonal exponential decay $e^{-A\Delta t}$, gradient-capped radial-basis attraction, and softmax Boltzmann readout | Non-essential to downstream trust guarantees |
| *"Thermodynamic calibration & conformal set"* (`calibrate.py:113-117, 339-341`) | Exact integer-rank split-conformal LAC quantile $r = \min(\lceil(n+1)(1-\alpha)\rceil, n)$ requiring disjoint parameter/quantile splits | `aletheia#10` Gate 4 / Gate 5 split hygiene |
| *"Unbound particle far from every well"* (`vega.py:48-52, 669-701`) | Out-of-distribution Mahalanobis/radial distance flag (`nearest > threshold`) | Fail-closed union escalation router (`route = "review"`) |
| *"Review band on precision target"* (`band.py:42-60`) | One-sided Wilson lower confidence bound solved via exact sqrt-free integer quadratic cross-multiplication | Zero-float `aletheia v2.1` escalation certification |
| *"Read-only evidence summary"* (`laya/evidence.py:23-82`) | 5-state epistemic meet-semilattice `(INCOMPARABLE < MISSING < UNKNOWN < INSUFFICIENT < PRESENT)` | `aletheia v2.1` `ClaimLedger` fail-closed promotion gate |
| *"Reader port into shared space"* (`_attach.py:92-120`, `_universal.py:19-67`) | Exact closed-form ridge linear map $W = (X_c^\top X_c + n\lambda I)^{-1} X_c^\top Y_c$ | `Moissanite-Labs/siliquant` × `vega-U` (`vegaml#40` GGUF) quantized-reader drift recovery |

---

## 2. Four Exact Pure-$\mathbb{Q}$ Pillars & Live Upstream Code Findings

### Pillar 1 — Exact Split-Conformal LAC & In-Sample Coupling Counterexample (`vegaml/calibrate.py`)
1. **Exact Integer-Rank Quantile & Binary64 Overshoot Witness (`calibrate.py:116`, `_common.py:1153`):**
   - Computing $r = \min(\lceil (n+1)(b-a)/b \rceil, n)$ via exact integer division eliminates IEEE-754 `1.0 - alpha` rounding drift.
   - Pure-integer IEEE-754 binary64 RNE proofs demonstrate exact rank overshoot on `math.ceil((n + 1) * (1.0 - alpha))`:
     - **`n = 149, alpha = 18/100`:** exact $(150)(82/100) = \mathbf{123}$, yet binary64 `150 * (1.0 - 0.18)` overshoots strictly above `123` $\implies$ `ceil = 124` (`+1` overshoot).
     - **`n = 29, alpha = 7/10`:** exact $(30)(3/10) = \mathbf{9}$, yet binary64 `30 * (1.0 - 0.7)` overshoots $\implies$ `ceil = 10` (`+1` overshoot).
2. **Exact Finite-Sample Combinatoric In-Sample Coupling Counterexample (`calibrate.py:339-341`):**
   - Live AST audit confirms `vegaml/calibrate.py:339-341` fits `temperature = _fit_temperature(rows)` and `qhat = _conformal_qhat([1.0 - p[y] for p, y in scaled], alpha)` on the **identical `rows`** (violating exchangeability between calibration scores and fresh test draws).
   - Over all $\binom{6}{3} = \mathbf{20}$ exact finite-sample splits ($N = 6, n_{\mathrm{cal}} = 3, \alpha = 1/4$, nominal coverage target $1 - \alpha = \mathbf{3/4 = 75.00\%}$, `10` balanced candidate calibrators):
     - **Disjoint / fixed calibrator** (independent of $D_{\mathrm{cal}}$): exact average test coverage = **`19/20` (`95.00% >= 75.00%`)** — preserves finite-sample guarantee.
     - **In-sample coupled selector** (minimizing calibration loss on the same $D_{\mathrm{cal}}$ used for $\hat{q}$): exact average test coverage drops to **`1/2` (`50.00% < 75.00%`)** — a **`-25.00 pp`** deficit with **`10/20`** splits suffering **`0%`** test coverage!

### Pillar 2 — Empty-Conformal Non-Trigger Iff Theorem, Sqrt-Free Wilson Bound & Fail-Closed Union Router
1. **Empty-Set Non-Trigger Iff Theorem (`vegaml/band.py:13-17`, `calibrate.py:368`):**
   - For any simplex distribution $p \in \Delta_{k-1} \cap \mathbb{Q}^k$ ($k \ge 2$), $\max_i p_i \ge 1/k$. Hence LAC abstention ($C(x) = \emptyset$) is reachable **if and only if** $\hat{q} < 1 - 1/k$ (witnessed explicitly on the uniform distribution $p_i = 1/k$: `k=2, qhat=1/2` yields `|C|=2` non-triggerable vs `k=2, qhat=2/5` yields `|C|=0` reachable; `k=4, qhat=4/5` yields `|C|=4` non-triggerable).
2. **Sqrt-Free Pure-Integer Squared Wilson Lower Bound (`vegaml/band.py:42-60`):**
   - Exact algebraic equivalence over $\mathbb{Z}$ for $L_{\mathrm{Wilson}}(h, n, z) \ge \tau$:
     $$2\tau_n(n Z_d + Z_n) \le \tau_d(2h Z_d + Z_n) \;\land\; n(n Z_d + Z_n)\tau_n^2 + h^2 Z_d \tau_d^2 \ge n(2h Z_d + Z_n)\tau_n\tau_d$$
   - Reproduces `vegaml/docs/escalation.md:64` (`12/12` at 95% $z = 16449/10000$) via exact integer `isqrt` bracket **`[0.816010, 0.816011]`**: passes $\tau = 816/1000$, rejects $\tau = 817/1000$ (`center_ok=True, quadratic_ok=False`), and rejects $\tau = 95/100$.
3. **Fail-Closed Union Router Closing Live Upstream Gaps (`laya/vega.py:669-701`, `vegaml/band.py:200-216`):**
   - Live AST audit confirms: (a) `vegaml/band.py:Band.route` returns `route="unbanded", review=False` when unconfigured and ignores `unbound`, `abstain`, and `prediction_set`; (b) `laya/vega.py:VegaAgent._route` never reads `unbound` at all, and skips `abstain`/`|pset|>1` whenever a reachable band is fitted.
   - Our fail-closed union operator escalates to `route="review"` iff **any** trigger fires (`band_unconfigured`, `band_unreachable`, `missing_confidence`, `confidence < band_threshold`, `missing_unbound`, `unbound_ood`, `missing_abstain`, `conformal_abstain`, `missing_prediction_set`, `empty_prediction_set`, or `|pset| > max_set`), catching missing metadata, OOD singletons, and band-fitted ambiguous sets that `laya/vega.py` passes as `"auto"`.

### Pillar 3 — 5-State Epistemic Meet-Semilattice $\to$ `Aletheia v2.1` `ClaimLedger` (`laya/laya/evidence.py`)
1. **Exhaustive Lattice Axiom Verification:**
   - Ordered bottom-to-top as `INCOMPARABLE (0) < MISSING (1) < UNKNOWN (2) < INSUFFICIENT (3) < PRESENT (4)` with exact mapping to `ClaimLedger` (`CONTRADICTED_FALSIFIED`, `NOT_SHOWN`, `UNVERIFIED_HYPOTHESIS`, `PARTIAL_FRAGILE`, `ELIGIBLE_FOR_VERIFIED`).
   - Exhaustively verified across all **`5^3 = 125`** state triples: idempotent (`5/5`), commutative (`25/25`), associative (`125/125`), `PRESENT` top identity (`5/5`), and `INCOMPARABLE` absorbing bottom (`5/5`).
2. **Two Live Bugs Confirmed In-Memory on `laya/laya/evidence.py:50-78`:**
   - **Live Bug 1 (`items = -5, issues = []`):** `laya/evidence.py` skips `items == 0`, checks `issues == []`, and returns **`PRESENT`** (`"fitted with no recorded issues on -5 items"`)! Our hardened `inspect_calibration_fail_closed_exact()` rejects `items < 0` as **`UNKNOWN`** (`UNVERIFIED_HYPOTHESIS`).
   - **Live Bug 2 (`items = True, issues = []`):** `laya/evidence.py:50` correctly marks `types["choice"]["state"] = "UNKNOWN"` via its `isinstance(..., bool)` guard, **yet** `laya/evidence.py:76-78` collapses the overall state to **`INSUFFICIENT`** instead of preserving `UNKNOWN` via the semilattice meet! Our fail-closed aggregator preserves **`UNKNOWN`** (and requires bound non-degenerate `checkpoint_sha256` to promote to `ELIGIBLE_FOR_VERIFIED`).

### Pillar 4 — `siliquant` × `vega-U` Quantized-Reader Linear Port Recovery & `laya#781` Guardrail Inversion
1. **Exact Pure-$\mathbb{Q}$ Quantized-Reader Linear Port Theorem (`vegaml/_attach.py:92-120` + `vegaml#40` GGUF):**
   - Given clean reader features $X \in \mathbb{Q}^{8 \times 2}$ mapped to shared-space targets $Y = X W_0 + \mathbf{1} b_0^\top$ and quantized-reader perturbations $\tilde{X} = X A + \mathbf{1} c^\top + E$ with invertible affine drift $(A, c)$ and zero-mean orthogonal rounding noise $E$ ($\mathbf{1}^\top E = 0, X_c^\top E = 0$):
     - **Systematic affine drift ($E = 0, \lambda = 0$):** Un-refitted FP16 port incurs exact residual sum of squares **`3949/100` (`39.49`)**; exact closed-form pure-$\mathbb{Q}$ port refit recovers $W^* = A^{-1} W_0$, $b^* = b_0 - c^\top A^{-1} W_0$ with **EXACT `0/1` residual** and **`0/1` KKT normal-equation error** (100% absorption, zero gradient steps, zero engine retraining).
     - **Affine drift + orthogonal quantization noise ($E \neq 0$):** Un-refitted FP16 port incurs **`2437/50` (`48.74`)**; closed-form port refit drops residual to **`93140/10697` (`8.707`, `5.60×` reduction)** — matching the exact closed-form Wiener errors-in-variables floor $\mathrm{tr}(W_0^\top G_X W_0 - W_0^\top G_X A (A^\top G_X A + G_E)^{-1} A^\top G_X W_0) = \mathbf{93140/10697} > 0$ identically!
   - **Concrete `siliquant` × `vega-U` Experiment Plan:** Since `vegaml#40` runs `vega-U`'s frozen reader from quantized GGUF weights via `llama.cpp`, refitting `port.safetensors` in closed form absorbs per-channel scale/zero-point drift (`0/1` residual), while reducing the remaining irreducible orthogonal floor $G_E = E^\top E$ requires `siliquant`'s outlier-rotated $E_8$ / `ZBG128:b560s16` quantization on `ffn_down` and tapped reader layers.
2. **Exact `laya#781` Markov Cap & Lexical Inversion Theorem (`laya/docs/guardrail-limitations.md:16-60`):**
   - Live exact parsing of `guardrail-limitations.md` confirms embedded attacks `c011` (`pi=182/1000`, `hs=82/100` $\implies$ Markov upper-tail cap `P[2]+P[3] <= 41/100`), `c012` (`pi=301/1000`, `hs=78/100` $\implies$ cap `39/100`), and `c042` (`pi=0`, `hs=51/100` $\implies$ cap `51/200 = 0.255`) are ALL mathematically capped below the `1/2` gate threshold, while benign quoted passage `c038` scores `pi = 619/1000`.
   - Because $\max(\text{c011}, \text{c012}, \text{c042}) = 301/1000 < 619/1000 = \text{c038}$, **no scalar threshold $\theta \in [0, 1]$ can separate embedded injections from benign attack-vocabulary quotations** — proving why System-1 classifiers (`laya`/`vega-U`) must act as escalation routers feeding Zone-B deterministic verifiers rather than replacing them.

---

## 3. Law-1b Unmerged-PR Portal Gems (`8` Pinned PRs Audited Read-Only)

| Repository & PR | State | Pinned Head SHA | Exact Engineering Signal / Bridge |
|---|---|---|---|
| `NandhaKishorM/laya#1524` | `open` | `1d9aaae8b3` | Fused ONNX export verification pinned against the selected PyTorch checkpoint (closes cross-checkpoint verification drift) |
| `NandhaKishorM/laya#1519` | `open` | `216a723efc` | Fixes `laya/vega.py` string-keyed choice-label translation bug (round-tripping caller labels faithfully) |
| `NandhaKishorM/laya#1517` | `open` | `9291f01bfc` | Single-pass eval-overlap leakage checks & signature reuse in `laya/train.py` (aligns with `aletheia#10` Gate 4) |
| `NandhaKishorM/laya#1502` | `closed` | `ac03d3fc00` | Unmerged stdlib HTTP Monid adapter (zero-dependency typed-judgment transport example) |
| `NandhaKishorM/laya#1390` | `open` | `f63252ba2e` | Rust SDK revision pinning & SHA-256 digest checks (parallels `aletheia` R7 provenance pinning) |
| `NandhaKishorM/laya#1388` | `open` | `32097a4e5e` | Rust SDK `predict_tournament` & `cached_embed_fn` pairwise ranking port |
| `NandhaKishorM/vegaml#57` | `open` | `a7d5ccba0e` | Replaces non-deterministic atomic `index_add_` with direct deterministic span slicing in `Perception._pooled_batch` (`aletheia` R4 determinism) |
| `NandhaKishorM/vegaml#40` | `open` | `aa1ad172d8` | Inference-only Rust port loading `vega-U` data-only over **GGUF quantized readers via `llama.cpp`** — direct production target for Pillar 4 |

---

## 4. Reproducibility & Falsification Criteria

```bash
# 1. Offline self-contained replay (default; runs intake-free in air-gapped snapshots)
python3 -I -O -B -W error laya_vegaml_audit.py --standalone --json

# 2. Live read-only recount over staged intake-laya-vegaml/ & canonical certificate parity
python3 -I -O -B -W error laya_vegaml_audit.py --live --json | diff -u laya_vegaml_certificate.json -

# 3. Assertion-free R5 unit, mutation-killing, and Zone-B anti-bypass test suite
python3 -I -O -B -W error test_laya_vegaml_r5.py -v
```

**Falsification Criteria:**
- Any discrepancy between `laya_vegaml_audit.py --live --json` and `laya_vegaml_certificate.json`, or any floating-point / complex / non-JSON value inside the certificate, falsifies the certificate.
- Any valid simplex distribution $p \in \Delta_{k-1}$ yielding $C(x) = \emptyset$ when $\hat{q} \ge 1 - 1/k$, or any invertible affine drift $(A, c)$ yielding non-zero closed-form port residual when $E = 0$, falsifies Pillars 2 or 4.
- Any upstream patch to `vegaml/calibrate.py` splitting temperature and $\hat{q}$ onto disjoint calibration folds, or to `laya/evidence.py` rejecting negative `items` and aggregating via exact lattice meet, supersedes the corresponding live code-bug findings (which remain pinned strictly to `laya @ e150aa92` and `vegaml @ 5e7ffad5`).
