# Track S20b Live Gate-8 Evaluation (`studies/track-s20b-gate8-live-eval`)

Self-contained pure-$\mathbb{Z}/\mathbb{Q}$ + lossless `Decimal` empirical evaluation study scoring the newly committed `Qwen2.5` 3.0-bit rows (`0d843d95` `EBG128:b368s16` and `be63add8` `HBG128:b368s16`, branch snapshot `462fb2f8`) directly against the frozen `evaluate_gate8_rows` predicate merged in [`studies/expedition-e8had-anisotropy-audit119br/`](../expedition-e8had-anisotropy-audit119br/). **`G8-P1` is a post-hoc check** (consistent with the data, not pre-registered — see the provenance audit below):

- **Exact Gate-8 Verdicts (`9/9` `G8-P1` Clauses Verified in `Fraction` Arithmetic):**
  - **`G8-P1`: `PASS` (`9/9` clauses satisfied; post-hoc check, not a pre-registered prediction):** `EBG128:b368s16` (`kl_u6 = 169338`, `-21.292%` vs `ZBG` `215147`, $z^2 = 100.168$) and `HBG128:b368s16` (`kl_u6 = 178526`, `-17.021%` vs `ZBG`, $z^2 = 63.846$, `z005 = 93.93%`, `err005 = +9.093 dB`), consistent with the SmolLM2-to-Qwen2.5 cross-model sign reversal (`+58.3% -> -21.29%`).
  - **`G8-P2`, `G8-P3`, `G8-P4`: `PENDING`** (strictly unrun at snapshot `462fb2f8`; sealed in git at `4f7e4adf` `2026-11-23T02:07:40Z`, i.e. still pre-registered relative to their not-yet-landed rows).
- **Honest UTC Timestamp Provenance Audit:** Records the externally verifiable git events — data rows `0d843d95` `2026-11-22T23:16:15Z` and `be63add8` `2026-11-22T23:26:03Z`, and the first git commit of the G8-P1 oracle/certificate `4f7e4adf` `2026-11-23T02:07:40Z` (after both rows, so G8-P1 is post-hoc) — while local file mtimes and board post `#1155` are marked **self-reported** (not externally verifiable). `externally_cryptographically_precommitted_via_board_1155["G8-P1"] = false` while `G8-P2..G8-P4` remain `true`.
- **Zero-Float Trust Path:** Pure Python standard library (`int`, `fractions.Fraction`, `decimal.Decimal`), zero IEEE-754 floats on trust/certificate paths, zero bare `assert`s, Zone-B canonical JSON & non-blocking regular-file I/O hardening.

## Quick Verification

```bash
cd studies/track-s20b-gate8-live-eval

# 1. R5 test suite across all three isolated R4 modes (the real-oracle parity
#    test runs when S20B_LIVE_DIR points at the 4f7e4adf files; otherwise it
#    reports one explicit skip)
python3 -I -B test_gate8_live_eval_r5.py
python3 -I -O -B test_gate8_live_eval_r5.py
python3 -I -B -W error test_gate8_live_eval_r5.py

# 2. Fail-closed certificate check & byte-exact standalone replay
python3 -I -B gate8_live_eval_audit.py --check gate8_live_eval_certificate.json --json
python3 -I -B gate8_live_eval_oracle.py --standalone --json | cmp -s - gate8_live_eval_certificate.json

# 3. (Optional) real frozen-oracle live parity: extract the pinned files from
#    git commit 4f7e4adf into a scratch dir and pass that dir to --live-s20b
mkdir -p /tmp/s20b-4f7e4adf
git show 4f7e4adf:studies/expedition-e8had-anisotropy-audit119br/e8had_anisotropy_oracle.py > /tmp/s20b-4f7e4adf/e8had_anisotropy_oracle.py
git show 4f7e4adf:studies/expedition-e8had-anisotropy-audit119br/e8had_anisotropy_certificate.json > /tmp/s20b-4f7e4adf/e8had_anisotropy_certificate.json
S20B_LIVE_DIR=/tmp/s20b-4f7e4adf python3 -I -B test_gate8_live_eval_r5.py
python3 -I -B gate8_live_eval_oracle.py --live-s20b /tmp/s20b-4f7e4adf --json
```

*(Note: `--live-s20b <dir>` enforces the exact frozen Track S20b R2 oracle SHA-256 `dc13922753737ec127cd59817479680d7615b2a326d2664b56d0016fb146dc72` — the revision first committed at git `4f7e4adf` (`2026-11-23T02:07:40Z`). The in-tree `../expedition-e8had-anisotropy-audit119br/` copy has since moved to a different oracle hash (`1ab4cc2c…`) and fails the pin with `S20B_ORACLE_SHA_MISMATCH`; on main, `dc139227` was later replaced by `1e00dac7` (a commit of PR #42), and the PR #55 twin is `ed5756d3`. Hence the `git show 4f7e4adf:...` extraction in step 3 above.)*

See [`GATE8_LIVE_EVAL_REPORT.md`](GATE8_LIVE_EVAL_REPORT.md) for the complete 9-clause exact rational table, Steiner-quadruple diagnostics, and UTC provenance audit.
