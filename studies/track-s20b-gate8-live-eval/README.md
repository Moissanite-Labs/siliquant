# Track S20b Post-Freeze Live Gate-8 Evaluation (`studies/track-s20b-gate8-live-eval`)

Self-contained pure-$\mathbb{Z}/\mathbb{Q}$ + lossless `Decimal` post-freeze empirical evaluation study scoring the newly committed `Qwen2.5` 3.0-bit rows (`0d843d95` `EBG128:b368s16` and `be63add8` `HBG128:b368s16`, branch snapshot `462fb2f8`) directly against the frozen `evaluate_gate8_rows` predicate merged in [`studies/expedition-e8had-anisotropy-audit119br/`](../expedition-e8had-anisotropy-audit119br/):

- **Exact Gate-8 Verdicts (`9/9` `G8-P1` Clauses Verified in `Fraction` Arithmetic):**
  - **`G8-P1`: `PASS` (`9/9` clauses satisfied):** `EBG128:b368s16` (`kl_u6 = 169338`, `-21.292%` vs `ZBG` `215147`, $z^2 = 100.168$) and `HBG128:b368s16` (`kl_u6 = 178526`, `-17.021%` vs `ZBG`, $z^2 = 63.846$, `z005 = 93.93%`, `err005 = +9.093 dB`), confirming the predicted SmolLM2-to-Qwen2.5 cross-model sign reversal (`+58.3% -> -21.29%`).
  - **`G8-P2`, `G8-P3`, `G8-P4`: `PENDING`** (strictly unrun at snapshot `462fb2f8`).
- **Honest UTC Timestamp Provenance Audit:** Explicitly records exact commit timestamps (`23:15:03Z` oracle finalization vs `23:16:15Z` `0d843d95` and `23:26:03Z` `be63add8`) and marks `externally_cryptographically_precommitted_via_board_115["G8-P1"] = false` while `G8-P2..G8-P4` remain `true`.
- **Zero-Float Trust Path:** Pure Python standard library (`int`, `fractions.Fraction`, `decimal.Decimal`), zero IEEE-754 floats on trust/certificate paths, zero bare `assert`s, Zone-B canonical JSON & non-blocking regular-file I/O hardening.

## Quick Verification

```bash
cd studies/track-s20b-gate8-live-eval

# 1. Zero-skip R5 test suite across all three isolated R4 modes
python3 -I -B test_gate8_live_eval_r5.py
python3 -I -O -B test_gate8_live_eval_r5.py
python3 -I -B -W error test_gate8_live_eval_r5.py

# 2. Fail-closed certificate check & byte-exact standalone replay
python3 -I -B gate8_live_eval_audit.py --check gate8_live_eval_certificate.json --json
python3 -I -B gate8_live_eval_oracle.py --standalone --json | cmp -s - gate8_live_eval_certificate.json
```

*(Note: `--live-s20b <dir>` enforces the exact pre-`#55` frozen Track S20b R2 oracle SHA-256 `dc13922753737ec127cd59817479680d7615b2a326d2664b56d0016fb146dc72`.)*

See [`GATE8_LIVE_EVAL_REPORT.md`](GATE8_LIVE_EVAL_REPORT.md) for the complete 9-clause exact rational table, Steiner-quadruple diagnostics, and UTC provenance audit.
