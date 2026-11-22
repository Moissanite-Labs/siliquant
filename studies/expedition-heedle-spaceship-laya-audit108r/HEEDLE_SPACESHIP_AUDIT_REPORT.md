# Track S13 (`AUDIT-108`): Clean-Room Audit of `Moissanite-Labs/heedle` & Pure-$\mathbb{Z}/\mathbb{Q}$ Scaling Bridge (`Spaceship` $\times$ `Laya/VegaML` $\times$ `SiliQuant` / `AUDIT-097`)

- **Audit ID:** `AUDIT-108` (`track-s13-heedle-spaceship-laya`)
- **Schema Version:** `heedle-spaceship-laya-oracle/v1`
- **Certificate `canonical_sha256` (inner body):** `c2403436221bb46d9e228138f00d81bf7d62e55582891ee3795328ed4f9cca09`
- **Certificate File SHA-256 (`heedle_spaceship_certificate.json`, 4-mode `-I` parity):** `07da98398e2ee017a9d779515c5ad827ab04f18bd20617763008cf94d59f95a9`
- **Arithmetic & Runtime Hygiene:** Pure Python 3 stdlib (`int` / `fractions.Fraction`), `0` IEEE-754 `float`s, `0` `math.sqrt`, `0` bare `assert` statements, `0` external writes, LF-only.

---

## 1. Honest Law-4 Novelty Tier & Pinned Read-Only Sources

| Layer | Law-4 Tier | Scope & Primary Attribution |
|---|---|---|
| Classical Gauss–Newton metric $G = J^T J / (nm)$, Gram–Schmidt orthogonal projection onto $\ker(S)$, symmetric INT4 RNE quantization, SHA-256 hash chains, split-conformal LAC, and pinned prior repositories | **`N0`** | `Moissanite-Labs/heedle` (`main @ d5e42392b03fe0edaddf8bdeb35c0c03c9378ca4`, open PR `#8 @ 7de102ecf56ba99e277f7c9d021c2e66eab55e8c`, Issues `#2/#3/#6/#7`); `chuanqisun/spaceship` (`master @ 0499b9930bdc0c471ee62ee1b1c01e1a986fb4ec`, closed-unmerged PR `#5 @ 7633dbee79799944d757e6d6993607174bee88fe`); `NandhaKishorM/laya` & `vegaml`; `Moissanite-Labs/siliquant` & `AUDIT-097`. |
| Executable pure-$\mathbb{Z}/\mathbb{Q}$ unification oracle, column-vs-entry pruning & INT4 survival scaling bridge, `tide/1` exhaustive contrast certifier, `spaceship#5` -> conformal -> Zone-B `snap` router, and clean-room `heedle/draft.py` catches | **`N1`** *(engineering/audit)* | Zero-float, sqrt-free exact rational certifier (`heedle_spaceship_oracle.py`), fail-closed certificate validator, 12 dynamic must-fail controls, and concrete line-level audit findings on `Moissanite-Labs/heedle @ d5e42392`. |

**Pinned Read-Only Intake SHA-256 Digests (`Moissanite-Labs/heedle @ d5e42392b03fe0edaddf8bdeb35c0c03c9378ca4`):**
- `heedle/core.py`: `49b2bf776693f05dfdf361353586f61535622d9ae6fc412d55afe972adb04c8c`
- `heedle/draft.py`: `498b9392da0b5f37412b04f508463d143b0848df09b2d62d9762b5b7974a60d6`
- `heedle/hosts.py`: `796af461d3ee1e5789132e061445ff52c34e5d35707d978fc089d43a94b8346a`
- `docs/protocol.md`: `7c5e583dee69c59436c84b5eceaab5711f0cd676a068de970136a17d5906d3b2`

**Law-5 Scope Boundary:** All claims below are exact rational linear-algebra identities, exact integer quantization transitions, or deterministic finite-domain protocol checks verified offline. This audit does not execute browser WebGPU inference (`spaceship#5`) or train neural weights.

---

## 2. Clean-Room Read-Only Audit of `Moissanite-Labs/heedle @ d5e42392`

1. **Strengths Verified:**
   - `heedle/core.py` separates domain hashing (`WEFT-CBC-V1:` vs. `HEEDLE-V1:`), refuses floats and `bool` in `canonical_json` (`L58-75`), commits to raw little-endian `<f8` moves (`L465-473`), and ports `verify_weight_update_safety` rejecting `NaN`/`Inf` cleanly (`L389-422`).
   - `heedle/draft.py` enforces 18+ labeled rules (`H1–H4`, `E1–E5`, `M1`, `G1–G2`, `R1`, `P1`, `T1`, `C1–C4`, `S1`, `K1–K2`, `U1`), with `tests/test_draft.py` (`L257-263`) asserting every label has at least one tampered test draft.
   - Open PR `#8` (`docs/session-queue.md`, `7de102ec`) cleanly marks its standing as `proposed`, explicitly keeps the Heedle gate unchanged, and claims no new tests.
   - Open Issue `#6` (`tide/1` OKLCH colour-palette host) cleanly separates checkable contrast-floor claims (exhaustive over all `(fg, bg)` pairs) from non-checkable readings ("harmony or feel").
2. **Concrete Clean-Room Findings:**
   - **`HEEDLE-AUDIT-01` [MED] (`heedle/draft.py:156-159`, `T1` Double-Zone Skip):** `near = any(abs(tension - t) < 1e-6 for t in (0.3, 0.7))` disables `T1` completely near `0.7`. At `tension = 0.7` (`zone="knot"`, index `2`), `gesture="release"` (index `0`) passes `T1` instead of restricting tolerance strictly to adjacent `gesture="hold"` (index `1`). Closed in Pillar 4 (`check_t1_gesture_fixed`).
   - **`HEEDLE-AUDIT-02` [LOW] (`heedle/core.py:315-317`, `shaft_move`):** `travel * self.V[:, i] / math.sqrt(self.lam[i])` divides by zero if `i >= rank()` (`lam[i] == 0.0`).
   - **`HEEDLE-AUDIT-03` [LOW] (`heedle/draft.py:68, 116-125, 166`):** `H4` relies on Python dict equality (`{"cost": True} == {"cost": 1}`), which is rescued downstream only because `core.canonical_json` rejects `bool` during `genesis_id` hashing (`malformed draft` rather than `H4`). Likewise, `unpick` events do not forbid stray `move_b64`/`claim`/`gate` fields outside the hashed event body, and float reading comparisons (`_close`) accept `bool` via `float(True) == 1.0`.

---

## 3. Certified Five-Pillar Results (Exact Matches to `heedle_spaceship_certificate.json`)

### Pillar 1 — Exact Rational Gauss–Newton Shafts, Treadle Share & Pinned Shed (`p1_rational_gauss_newton_shafts_and_shed`)

- On diagonal rational Jacobian $J = \mathrm{diag}(4, 2, 1, 1/2)$ ($nm = 4$, $P = 4$), exact eigenvalues of $G = J^T J / 4$ are `["4/1", "1/1", "1/4", "1/16"]`.
- For unit-RMS treadle pattern $c = [1, 1, 1, 1]$ with rational travel $t = 3/5$:
  - $K = 1$: $\rho = 1/4$
  - $K = 2$: $\rho = 1/2$, exact undelivered mean-square residual $\mathrm{MS}(tc - Jd) = t^2(1 - \rho) = 9/50$ (`0/1` discrepancy)
  - $K = 4$: $\rho = 1/1$, undelivered residual `0/1`
- Exact rational Gram–Schmidt projection onto $\ker(S)$ for $S = [[1, 1, 0, 0], [0, 0, 1, -1]]$: `pin_rank = 2`, `open_dimensions = 2`, `projected_move = ["1/1", "-1/1", "2/1", "2/1"]`, `pin_leak_sq = "0/1"`, `idempotent = true`, `pythagorean_exact = true`.
- **Must-Fail Controls 1–2:** Unprojected move `[2, 0, 3, 1]` fails with `ERR_HEDDLE_PIN_LEAK` (`pin_leak_sq = "8/1"`); invalid share `5/4` fails with `ERR_TREADLE_SHARE_INVALID`.

### Pillar 2 — `core.hidden_move` Nullspace Stealth Vector & Zone-B Blinded `snap` (`p2_hidden_move_nullspace_refutation`)

- Rational power iteration on $J_{\mathrm{elsewhere}}^T J_{\mathrm{elsewhere}}$ inside $\ker(J_{\mathrm{known}})$ produces `stealth_move = ["0/1", "0/1", "1/262144", "1/1"]`:
  - **Known-probe visibility:** `known_probe_ms = "0/1"` (invisible to proposer's known probes).
  - **Zone-B blinded probes (`tau_sq = "1/4"`):** `elsewhere_probe_ms = "4398046511105/2199023255552"`, worst witness index `1` changes by `"4/1" > "1/4"`.
- **Must-Fail Controls 3–4:** Zone-B blinded witness snaps the stealth move (`ERR_STEALTH_MOVE_SNAPPED`); forged `snap` pointing at quiet witness index `0` (`"1/1099511627776" <= "1/4"`) is rejected (`ERR_SNAP_WITNESS_DOES_NOT_REFUTE`).

### Pillar 3 — Scaling Bridge to `AUDIT-097` + `siliquant` + `vega-U` (`p3_scaling_bridge_pruning_and_int4_survival`)

1. **Column-Structured Linear-Port vs. Entry-Wise Pruning (`u = [1, -1, 1, -1]`, `v = [2, 1, 3]`, heddle `h = [1, 1, 1, 1]` with $h^T u = 0$):**
   - **Column mask `[[1,0,1], ...]`:** `is_exact_rank_le_1 = true`, `rank1_gram_defect = "0/1"`, `heddle_preserved = true`, `heddle_leak_sq = "0/1"`.
   - **Must-Fail Control 5 (`must_fail_entry_pruning_shatters_heddle`):** Entry-wise scattered mask shatters both rank-1 (`rank1_gram_defect = "294/1"`) and heddle orthogonality (`heddle_leak_sq = "16/1"`), rejected with `ERR_PRUNING_SHATTERED_HEDDLE`.
2. **Exact Symmetric INT4 ($q_{\max} = 7$) Quantization-Survival Floor (`max_base = 28`, $\mathrm{ULP} = 4/1$, $\frac{3}{4}\mathrm{ULP} = 3/1$):**
   - **Surviving cell-center move (`0 -> 3`, $|d| = 3 = \frac{3}{4}\mathrm{ULP}$):** `edited_flips = 1`, `collateral_unedited_flips = 0`.
   - **Must-Fail Control 6 (`must_fail_int4_sub_threshold_erasure`):** Interior move `-1 -> 2` ($|d| = 3 = \frac{3}{4}\mathrm{ULP}$ landing on exact half-ULP tie `2/4 = 1/2`, RNE-rounded to even code `0`) yields `erased_edits = 1`, rejected with `ERR_QUANT_MOVE_ERASED`.
   - **Must-Fail Control 7 (`must_fail_int4_spike_collateral_flip`):** Interior spike `3 -> 56` (`scale_ratio = "2/1"`) survives on the edited weight (`+1 -> +7`) while collaterally flipping `3` unedited neighbor weights (including zeroing `4: +1 -> 0`), rejected with `ERR_QUANT_COLLATERAL_FLIP`.

### Pillar 4 — `heedle` `docs/protocol.md` Hash Chain, `tide/1` Host & `T1` Closure (`p4_heedle_protocol_replay_and_tide1`)

- **Clean Hash-Chained Replay (`pick -> snap -> unpick`):** `events_replayed = 3`, `final_fell_root = ROOT_W0` (`4cfa6d55a790cae5f8f331ccbc1578daecf1b3922590f5624c7d07ea72917b47`), `remaining_budget = 22` (`snap` leaves fell unchanged and spends budget; `unpick` restores fell without refunding budget).
- **`tide/1` Exhaustive Contrast-Floor Host (`4` `(fg, bg)` pairs, floor `9/20`, chroma max `3/10`):**
  - Clean palette: `clean_worst_contrast = "1/2"`, `clean_snapped = false`.
  - Contrast breach: `breach_worst_pair = ["fg_accent", "bg_panel"]`, `breach_worst_contrast = "7/20"`, `breach_snapped = true`.
- **Must-Fail Controls 8–10:** Tampered `tau` rejected at step `0` (`ERR_PROTOCOL_CHAIN_OR_TAU`); `unpick` budget refund rejected at step `1` (`ERR_UNPICK_BUDGET_REFUND`); `gesture="release"` at `tension=7/10` (`zone="knot"`) rejected (`ERR_T1_GESTURE_TOO_LIGHT`, closing `HEEDLE-AUDIT-01`).

### Pillar 5 — `spaceship#5` Circular Realization -> `laya`/`vegaml` Conformal -> Zone-B `snap` (`p5_spaceship_realization_to_zone_b_snap`)

- Closed-unmerged `chuanqisun/spaceship#5` documents explicitly that its same-vector realization score is **not an independent check** because the same steering vector writes and scores the text.
- Disjoint-split pure-$\mathbb{Q}$ conformal LAC calibration (`n = 9`, $\alpha = 1/5$): `rank = 8`, `qhat = "2/5"`, `coverage_lb = "4/5"`.
- Clean routing path: ambiguous System-1 conformal set (`size = 2`) -> `ESCALATE_ZONE_B` -> Zone-B blinded probe `7/10 > 1/2` -> `SNAP_REFUTED` (while clean Zone-B `3/10 <= 1/2` with singleton set -> `VERIFIED`).
- **Must-Fail Controls 11–12:** Promoting same-vector self-realization (`9/10`) to `VERIFIED` without Zone-B is rejected (`ERR_CIRCULAR_REALIZATION_CLAIM`); skipping `ESCALATE_ZONE_B` on an ambiguous conformal set (`size = 2`) is rejected (`ERR_CONFORMAL_ESCALATION_BYPASS`).

---

## 4. Verification & Reproducibility Receipts

Run inside `track-s13-heedle-spaceship-laya/`:

```bash
python3 -I -B -W error heedle_spaceship_certifier.py --check heedle_spaceship_certificate.json
for flags in "-I -B" "-I -O -B" "-I -B -W error" "-I -O -B -W error"; do
  python3 $flags heedle_spaceship_certifier.py --json | sha256sum
done
python3 -I -O -B -W error test_heedle_spaceship_r5.py
python3 -B -W error -m aletheia audit-check . --strict-ancestors --safety-scan --scripts heedle_spaceship_certifier.py test_heedle_spaceship_r5.py
python3 -B -W error -m aletheia safety-scan .
python3 -B -W error -m aletheia mutate heedle_spaceship_oracle.py --work-root . --add-cwd --max-mutants 0 --test-cmd "python3 -I -O -B -W error test_heedle_spaceship_r5.py"
python3 -B -W error -m aletheia mutate heedle_spaceship_certifier.py --work-root . --add-cwd --max-mutants 0 --test-cmd "python3 -I -O -B -W error test_heedle_spaceship_r5.py"
```

| Gate | Exact Result |
|---|---|
| Certificate `--check` | `OK AUDIT-108 canonical_sha256=c2403436221bb46d9e228138f00d81bf7d62e55582891ee3795328ed4f9cca09 pillars=5 controls=12` |
| 4-Mode Isolated `-I` Determinism | Bit-identical `07da98398e2ee017a9d779515c5ad827ab04f18bd20617763008cf94d59f95a9` across `-I -B`, `-I -O -B`, `-I -B -W error`, `-I -O -B -W error` |
| R5 Assertion-Free Suite (`-I -O -B -W error`) | `PASS AUDIT-108 R5_TESTS=8/8` |
| `aletheia mutate --max-mutants 0` | **`93/93`** (`heedle_spaceship_oracle.py`) + **`3/3`** (`heedle_spaceship_certifier.py`) = **`96/96` (`100.0%` killed, `0` survived, `truncated=False`)** |
| Full-AST `BoolOp` (`And`/`Or`) Sub-Clause Sweep | **`109/109` (`100.0%` killed, `0` survived)** across both `.py` modules |
| `aletheia audit-check --strict-ancestors` & `safety-scan` | `PASS` / `CLEAN` (`5` files, `2/2` R4, `0` defects, `0` leaks, `0` safety findings, `PYCACHE=0`) |

### `AUDIT-108R` Hardening Closures

1. **`HEEDLE108-BUG-01` (`replay_heedle_draft_exact`):** Enforces strict `type(...) is int` on `seq` and `remaining_budget` (rejecting `True`/`False`), non-empty `str` on `host.kind`/`host.name`, positive `gate.r_invariant`, valid `gesture in GESTURES` on all event kinds (including `unpick`), sorted distinct non-negative `int` `pins` (`_validate_pins`), non-uniform 64-hex `move_sha256` and `snap.root`, and `witness_change >= 0`.
2. **`HEEDLE108-BUG-02` (`certify_spaceship_realization_escalation`):** Enforces singleton `conformal_set_size == 1` for `VERIFIED` when `zone_b_independent_checked=True` and `zone_b_worst_change <= tau` (non-singleton sets `0` or `>= 2` require `ESCALATE_ZONE_B`).
3. **`HEEDLE108-BUG-03` (`evaluate_tide1_contrast_floor_exact`):** Rejects non-string or empty role keys with `HeedleAuditError("ERR_INVALID_TIDE_ROLE_NAME")` prior to sorting (eliminating raw `TypeError` on mixed key types).
