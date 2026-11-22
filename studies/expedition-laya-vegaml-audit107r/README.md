# `laya` & `vegaml` Exact Pure-$\mathbb{Q}$ Conformal, Wilson, Epistemic Lattice & `siliquant`×`vega-U` Port Recovery Study (`AUDIT-107R`)

Self-contained, pure Python 3 stdlib (`int` / `fractions.Fraction`, zero floats on trust paths or JSON, zero bare `assert`s) study independently `VERIFIED` across two model families (`Muse Code` + `verifier`, `118/118` `aletheia v2.0.0 mutate` killed, `146/146` AST `BoolOp` killed, `0` survivors).

## Scope & Direct Bridges to `siliquant`

- **Pillar 4 — `siliquant` × `vega-U` Quantized-Reader Linear Port Recovery (`vegaml#40` GGUF):**
  - Exact pure-$\mathbb{Q}$ closed-form ridge port refit $W^* = A^{-1} W_0$ absorbs systematic affine quantization drift with **`0/1` residual** and **`0/1` KKT error** (`3949/100 -> 0/1`, zero gradient steps, zero engine retraining).
  - Under affine drift plus zero-mean orthogonal quantization noise $E$, closed-form refit reduces residual **`5.60×`** (`2437/50 -> 93140/10697`), hitting the exact Wiener errors-in-variables floor `93140/10697 > 0` — isolating the irreducible orthogonal noise $G_E = E^\top E$ targeted by `siliquant`'s outlier-rotated $E_8$ / `ZBG128:b560s16` quantization.
- **Pillar 1 — Exact Integer-Rank Split-Conformal LAC & In-Sample Coupling Counterexample (`vegaml/calibrate.py:339-341`):**
  - Pure-integer IEEE-754 binary64 RNE model (`0 / 9,310` mismatches across `n=10..199 × alpha=0.01..0.49`) proves `+1` rank overshoot on `(149, 18/100)` (`123 -> 124`) and `(29, 7/10)` (`9 -> 10`).
  - Exhaustive $\binom{6}{3}=20$ finite-sample split theorem proves disjoint calibration achieves `19/20` (`95.00% >= 75.00%`) vs. in-sample coupling collapse to `1/2` (`50.00% < 75.00%`, `10/20` splits at `0%`).
- **Pillars 2 & 3 — Empty-Set Iff Theorem, Sqrt-Free Wilson Bound, Fail-Closed Union Router & 5-State Epistemic Meet-Semilattice:**
  - Exact iff condition $\hat{q} < 1 - 1/k$, sqrt-free integer Wilson lower bound (`12/12` @ 95% bracket `[0.816010, 0.816011]`), fail-closed union router closing `vegaml/band.py` & `laya/vega.py` gaps, all `125` semilattice triples verified, two live upstream `laya/evidence.py` bugs pinned (`items=-5 -> PRESENT`, `items=True -> INSUFFICIENT`), and `laya#781` Markov-cap lexical inversion proof (`max(c011,c012,c042) = 301/1000 < 619/1000 = c038`).
- **Honest Novelty Standing (Law 4):** `N1` (exact pure-$\mathbb{Q}$ executable certifier, live AST/code-bug audit, and fail-closed bridge) over `N0` classical conformal/Wilson/semilattice/ridge-Wiener primitives.

## Offline Reproduction (4-Mode Isolated `-I` Parity)

The committed [`laya_vegaml_certificate.json`](laya_vegaml_certificate.json) (`sha256: 4a380038bce6992fbd6d6b064d8404574bec4c04c4ef346e97725bea9cca9c60`) was generated in `--live` mode against pinned read-only upstream intake snapshots (`NandhaKishorM/laya @ e150aa92`, `NandhaKishorM/vegaml @ 5e7ffad5`). In this standalone repository snapshot, `--standalone` (default) replays all 4 pure-$\mathbb{Q}$ pillars and certificate validators offline with zero external files (bit-identical `375ad33401f8895c319e49ea4c1f6ba73d0be131943520f6d1a7ec782afd5f2e` across all 4 isolated modes):

```bash
python3 -I laya_vegaml_audit.py --standalone --json | sha256sum
python3 -I -B laya_vegaml_audit.py --standalone --json | sha256sum
python3 -I -O -B laya_vegaml_audit.py --standalone --json | sha256sum
python3 -I -O -B -W error laya_vegaml_audit.py --standalone --json | sha256sum
python3 -I -O -B -W error test_laya_vegaml_r5.py -v
```

See [`LAYA_VEGAML_AUDIT_REPORT.md`](LAYA_VEGAML_AUDIT_REPORT.md) for the full audit tables, Law-1b unmerged-PR pins, and falsification criteria.
