# Track S20b Live Gate-8 Evaluation (`AUDIT-119B-LIVE`)

- **Oracle Class:** `exact+empirical-corpus` (pure-stdlib $\mathbb{Z}/\mathbb{Q}$ + lossless `Decimal` raw JSONL ingestion; zero floats on all trust/certificate paths; zero bare `assert`s; full Zone-B `strict_cert_io` hardening)
- **Novelty Class:** `N1` (post-hoc exact-$\mathbb{Q}$ evaluation — consistent with the data, not pre-registered — of Track S20b's `G8-P1` check against newly committed `Moissanite-Labs/siliquant` rows `0d843d95` & `be63add8`, live read-only execution of frozen `e8had_anisotropy_oracle.evaluate_gate8_rows` with `sys.dont_write_bytecode=True`, and brutally honest Law-4 UTC timestamp provenance audit)

---

## 1. Frozen Target & Upstream Snapshot Anchor

This sibling evaluation track performs **zero writes** inside the frozen Track S20b (`AUDIT-119B` R2) artifacts under `studies/expedition-e8had-anisotropy-audit119br/` (first committed at git `4f7e4adf`, `2026-11-23T02:07:40Z`) and scores the newly committed rows from `Moissanite-Labs/siliquant` (`isPrivate=false`, branch `bayan/charming-gates-jlas14`):

| Artifact / Commit | Exact UTC | Exact SHA-256 / Git SHA |
|---|---|---|
| Frozen Track S20b R2 `e8had_anisotropy_oracle.py` | — | `dc13922753737ec127cd59817479680d7615b2a326d2664b56d0016fb146dc72` |
| Frozen Track S20b `e8had_anisotropy_certificate.json` | — | `84e79bb9f2dab219228b5fc80688301fde92710f9c0978f6385f6a7f30ee4b0d` |
| First git commit of the frozen Track S20b study (`4f7e4adf`; oracle `dc139227…` + certificate `84e79bb9…` landed here) | `2026-11-23T02:07:40Z` | `4f7e4adf362782dc409350b44ce2a8203fbbc719` |
| Track S20b Anchor Commit (`13/15` KL rows, `ZBG` landed) | `2026-11-22T23:06:57Z` | `51235fa753ecd32a15c9af0251560156e3bf440f` (`results.jsonl` `e4023053...`) |
| Newly Landed Row 13 (`Qwen2.5 EBG128:b368s16`) | `2026-11-22T23:16:15Z` | `0d843d956bb2ef6ce90e3563f995a3ca5750f86a` |
| Newly Landed Row 14 (`Qwen2.5 HBG128:b368s16`) | `2026-11-22T23:26:03Z` | `be63add87658f7e477fdf436f9396efd7950f1bf` |
| Live Branch HEAD (`15/15` KL rows + `7` weight-only `metrics.jsonl` rows) | `2026-11-23T00:04:07Z` | `462fb2f82518faa5203edc2784b93fcf64b06609` |
| Committed `results.jsonl` at `462fb2f8` (`34,348` B, `15` rows) | `2026-11-22T23:26:03Z` | `f9c495cd81184688436947c8d1e9042da9e67d180f0b2e481deb43b725c9d3c1` |
| Committed `metrics.jsonl` at `462fb2f8` (`13,601` B, `7` weight-only rows) | `2026-11-23T00:04:07Z` | `c0f081fe047dc79f27d1ec6a8dc5f9e05487ac3e330c25c91fe88ecfe8cf8880` |

---

## 2. Exact Gate-8 Verdicts (`evaluate_gate8_rows` Live Parity = `EXACT`)

Passing the exact micro-unit integers extracted from committed rows `13` (`EBG128:b368s16`: `kl_u6=169338`, `se_u6=3095`) and `14` (`HBG128:b368s16`: `kl_u6=178526`, `se_u6=3104`, `z005_bp=9393`, `err005_mdB=9093`) verbatim into Track S20b's frozen `evaluate_gate8_rows(...)` yields:

- **`G8-P1`: `PASS` (`9/9` clauses satisfied)** — evaluated as a **post-hoc check** (consistent with the data, not pre-registered): the only externally verifiable freeze of the G8-P1 oracle is git commit `4f7e4adf` (`2026-11-23T02:07:40Z`), which is *after* both data rows (see §3)
- **`G8-P2`: `PENDING`** (Llama-3.2 `VH8Z32:16` vs `VE8Z32:16` `+/-K10` unrun; `metrics.jsonl` only added `ZBG`/`EBG`/`VE8Z` weight-only rows)
- **`G8-P3`: `PENDING`** (Qwen2.5 4.5-bit `EBG128+K10` vs `HBG128+K10` unrun)
- **`G8-P4`: `PENDING`** (`Block-H2` replay vs `WHT_8` unrun)

### Exact Rational Breakdown of All 9 `G8-P1` Clauses (vs. Anchor `ZBG`: `kl_u6=215147`, `se_u6=3372`)

| Clause | Requirement | Exact Rational Value | Decimal | Margin | Result |
|---|---|---:|---:|---:|---|
| `c1` | `kh < kz` | `178526 < 215147` | `0.178526 < 0.215147` | `-36,621 u6` | **PASS** |
| `c2` | `red_hbg_vs_zbg >= 1/10` | `36621/215147` | `17.021%` | `+7.021%` | **PASS** |
| `c3` | `z2_hbg_vs_zbg >= 16` | `1341097641/21005200` | `63.846` (`z = 7.99`) | `+47.846` | **PASS** |
| `c4` | `ke < kz` | `169338 < 215147` | `0.169338 < 0.215147` | `-45,809 u6` | **PASS** |
| `c5` | `red_ebg_vs_zbg >= 2/25` | `45809/215147` | `21.292%` | `+13.292%` | **PASS** |
| `c6` | `z2_ebg_vs_zbg >= 9` | `2098464481/20949409` | `100.168` (`z = 10.01`) | `+91.168` | **PASS** |
| `c7` | `abs_gap_hbg_ebg < 3/20` | `4594/84669` | `5.426%` | `9.574%` headroom | **PASS** |
| `c8` | `h_z005 >= 8800 bp` | `9393 bp` | `93.93%` | `+593 bp` | **PASS** |
| `c9` | `h_err005 >= 6000 mdB` | `9093 mdB` | `+9.093 dB` | `+3,093 mdB` | **PASS** |

### Secondary Diagnostics on Qwen2.5 3.0-bit

- **Cross-Model Sign Reversal (post-hoc, consistent with the data):** Whereas SmolLM2 3.0-bit suffered `+58.3%` (`EBG`) and `+50.1%` (`HBG`) KL degradation over `ZBG`, Qwen2.5 3.0-bit reverses cleanly (`-21.29%` `EBG`, `-17.02%` `HBG`), matching the `-0.658 dB` (`-16.455` vs `-15.797 dB`) bulk coding-gain mechanism.
- **`HBG` vs `EBG` Trade-Off & Steiner-Quadruple Model:** `EBG` beat `HBG` by `9,188 u6` (`5.43%`, `z^2 = 84419344/19213841 ≈ 4.39`, `z ≈ 2.10`), while the exact `2x` Steiner-quadruple energy-ratio model predicts `1214/3223` (`-4.240 dB`) at `small_0.05` vs empirical `-4.591 dB` (within `0.351 dB`) and `2498/2353` (`+0.260 dB`) at `small_0.20` vs empirical `-0.102 dB` (within `0.362 dB`).

---

## 3. Brutally Honest Law-4 UTC Timestamp Audit (Zero Spin)

| Event | Exact UTC | Delta vs. `G8-P1` Oracle Finalized (`23:15:03Z`, self-reported) | Verifiability |
|---|---|---:|---|
| Upstream anchor `51235fa7` (`13/15` rows; `ZBG` landed) | `2026-11-22T23:06:57Z` | `-486 s` (`-8m 06s`) | git commit (externally verifiable) |
| Track S20b `G8-P1` first authored in `e8had_anisotropy_oracle.py` | `2026-11-22T23:11:58Z` | `-185 s` | self-reported local mtime |
| Track S20b `G8-P1` bounds finalized in `e8had_anisotropy_oracle.py` | `2026-11-22T23:15:03Z` | `0 s` | self-reported local mtime |
| Upstream `0d843d95` (`Qwen2.5 EBG128:b368s16` committed) | `2026-11-22T23:16:15Z` | **`+72 s`** (`+1m 12s`) | git commit (externally verifiable) |
| Track S20b `E8HAD_ANISOTROPY_GATE8_REPORT.md` final mtime | `2026-11-22T23:18:41Z` | `+218 s` (`+146 s` after `0d843d95`, **`-442 s` before `be63add8`**) | self-reported local mtime |
| Upstream `be63add8` (`Qwen2.5 HBG128:b368s16` committed) | `2026-11-22T23:26:03Z` | **`+660 s`** (`+11m 00s`) | git commit (externally verifiable) |
| Track S20b `e8had_anisotropy_certificate.json` final mtime | `2026-11-22T23:27:32Z` | `+749 s` (`+89 s` after `be63add8`) | self-reported local mtime |
| Swarm board post `#1155` (Track S20b R1 `[done]`) | `2026-11-22T23:39:29Z` | `+1,466 s` (`+806 s` after `be63add8`) | self-reported board post |
| Upstream branch HEAD `462fb2f8` | `2026-11-23T00:04:07Z` | `+2,944 s` (`+49m 04s`) | git commit (externally verifiable) |
| **Git commit `4f7e4adf` — first commit of the G8-P1 oracle (`dc139227…`) and certificate (`84e79bb9…`) under `studies/expedition-e8had-anisotropy-audit119br/`** | `2026-11-23T02:07:40Z` | **`+10,357 s` (`+10,285 s` after `0d843d95`, `+9,697 s` after `be63add8`)** | git commit (externally verifiable) |

**Honest Law-4 Epistemic Classification:**
1. **G8-P1 is a post-hoc check (consistent with the data, not pre-registered).** The only freeze event an outside verifier can check is git: the G8-P1 oracle (`dc139227…`) and certificate (`84e79bb9…`) first appear in commit `4f7e4adf` at `2026-11-23T02:07:40Z`, which is `+10,285 s` after the EBG row (`0d843d95`, `2026-11-22T23:16:15Z`) and `+9,697 s` after the HBG row (`be63add8`, `2026-11-22T23:26:03Z`). The local mtimes (`23:11:58Z` first written, `23:15:03Z` finalized, `23:18:41Z` report, `23:27:32Z` certificate) and board post `#1155` (`23:39:29Z`) are **self-reported** — nobody else can reproduce them — and are recorded in the certificate as such, never as evidence of pre-commitment. Accordingly, this receipt marks `externally_cryptographically_precommitted_via_board_1155["G8-P1"] = false`.
2. **`G8-P2`, `G8-P3`, and `G8-P4` remain pre-registered relative to their (not yet landed) rows.** They were sealed in git at `4f7e4adf` (`2026-11-23T02:07:40Z`) — not at branch HEAD `462fb2f8` (`2026-11-23T00:04:07Z`) and not via board `#1155`. No Llama-3.2 `VH8Z32:16`/`VE8Z32:16` ±K10, Qwen2.5 4.5-bit `EBG128+K10`/`HBG128+K10`, or `Block-H2` row landed in `experiments/e8-hadamard-real-weights-2026-11-22` between `462fb2f8` and `4f7e4adf` (the next commit touching that directory is `ab1022b` at `2026-11-23T02:20:52Z`), so `externally_cryptographically_precommitted_via_board_1155["G8-P2"/"G8-P3"/"G8-P4"] = true` still holds for rows that had not landed at the sealing commit.

---

## 4. Reproducibility & Verification Commands

- Standalone exact certificate: `python3 -I -B gate8_live_eval_oracle.py --standalone --json`
- Fail-closed certificate check: `python3 -I -B gate8_live_eval_audit.py --check gate8_live_eval_certificate.json`
- Live read-only frozen Track S20b parity check (`0` `__pycache__` in the target dir). The pinned files live at git commit `4f7e4adf`, so extract them into a scratch dir first and pass that dir to `--live-s20b`:

```bash
mkdir -p /tmp/s20b-4f7e4adf
git show 4f7e4adf:studies/expedition-e8had-anisotropy-audit119br/e8had_anisotropy_oracle.py > /tmp/s20b-4f7e4adf/e8had_anisotropy_oracle.py
git show 4f7e4adf:studies/expedition-e8had-anisotropy-audit119br/e8had_anisotropy_certificate.json > /tmp/s20b-4f7e4adf/e8had_anisotropy_certificate.json
python3 -I -B gate8_live_eval_oracle.py --live-s20b /tmp/s20b-4f7e4adf --json

# Optional cross-check against the committed results.jsonl at 462fb2f8:
git show 462fb2f8:experiments/e8-hadamard-real-weights-2026-11-22/results.jsonl > /tmp/results-462fb2f8.jsonl
python3 -I -B gate8_live_eval_oracle.py --live-s20b /tmp/s20b-4f7e4adf --results-jsonl /tmp/results-462fb2f8.jsonl --json
```

(The previously documented `--live-s20b ../track-s20b-e8had-anisotropy-gate8` path does not exist at this head (`S20B_DIR_NOT_FOUND`), and the in-tree `studies/expedition-e8had-anisotropy-audit119br/` copy now has oracle `1ab4cc2c…`, which fails the `dc139227…` pin with `S20B_ORACLE_SHA_MISMATCH` — hence the `git show 4f7e4adf:...` extraction above. Note also that `--results-jsonl` without `--live-s20b` is rejected with `RESULTS_JSONL_WITHOUT_LIVE_S20B` (rc 1) instead of being silently ignored.)
