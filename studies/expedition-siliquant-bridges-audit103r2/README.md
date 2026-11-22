# Branchless Integer $E_8$ Decoder, TPU v5e Part-E Roofline & Exact Operator Bridges (`AUDIT-103R2`)

> **Status:** Independently `VERIFIED` across 3 independent model families / 2 continents (`AUDIT-103R2`, `honest_novelty_tier: N1` engineering/branchless $E_8$ over `N0` classical prior art, `oracle_class: exact` + `anchor`).
> **Seats:** Author `continuum-bridge` · Clean-Room Verifiers: `verifier`, `Mistral Vibe 3.1.4`, and `Muse Code 1.4.25`.
> **Runtime Dependencies:** Python 3 standard library only (`int`, `fractions.Fraction`; zero floats on trust paths or JSON, zero bare `assert`s).
> **Scope Boundary (Honest Part-E vs. Part-I Disclosure):** All mathematical, lattice, roofline, homeostatic, Kirchhoff, and conditioning results in this package are **Part E exact pure-$\mathbb{Z}$/$\mathbb{Q}$ computations** (`measured_kernel_tok_s_present: false`). Live TPU/SIMD kernel tok/s, real-model GGUF output KL, and dequant instruction latency remain **Part I** and require pre-registered benchmarks under `experiments/` (tracked in Issue `#25` & Issue `#23`).

## 1. Contents (Zero Edits to Existing `siliquant` Files)

All 5 core study files below are byte-identical to the frozen `AUDIT-103R2` verified package:

| File | SHA-256 | Role |
|---|---|---|
| [`siliquant_bridge_oracle.py`](siliquant_bridge_oracle.py) | `cea1b40fc4fa6ba660c56f54aac6ebdcd57d707e114f1627da668f792964c94b` | Branchless `int64` Conway–Sloane $E_8$ decoder + brute-force verifier, `siliquant#16` `Code.centre` floor witness, Part-E TPU v5e roofline oracle, Turrigiano/Barkai–Leibler/Kirchhoff $R_{\mathrm{eff}}$ bridges, al-Bīrūnī/al-Kāshī conditioning & contraction, fail-closed certifier |
| [`siliquant_bridge_audit.py`](siliquant_bridge_audit.py) | `d218ba6728c995e7d91786853ea28685d0d7495b2d22bd071350919165dd586e` | Dual-mode CLI (`--standalone` 100% offline vs. `--live` read-only temp shared clone with zero writes to `.git`) |
| [`test_siliquant_bridge_r5.py`](test_siliquant_bridge_r5.py) | `b3d175ea03d6b441f96e27a0b143a57f5fcdca3e92481b46f1c5b0645ffd61d0` | Assertion-free R5 mutation-killing, 48×512 exhaustive $E_8$ brute-force, & temporary-git-repo live-path suite (`6/6` OK) |
| [`siliquant_bridge_certificate.json`](siliquant_bridge_certificate.json) | `62338fa92540d08e75412c709ed6326082af7a40cc97de0c150eff02506d0eef` | Pinned live-repo canonical certificate (`payload_sha256` inner `6afebf60...`) |
| [`SILIQUANT_BRIDGE_REPORT.md`](SILIQUANT_BRIDGE_REPORT.md) | `a0a93c8aaf389d5a6cb73dbaf92a5e34ae6e3ae40e94393ab29f65dfe9e0d1d8` | Full technical report, PR `#12`/`#24`/`#26`/`#27` diagnostic, and exact operator bridges |

## 2. Highlights & Honest Boundaries

1. **Branchless Pure-Integer Conway–Sloane $E_8$ Decoder (`siliquant#25` Step 2 & `#16`):**
   - Zero data-dependent `if` branches (`branchless_d8_decode_int`, `branchless_e8_decode_int`); arithmetic right-shift sign masks (`>> 60`), XOR parity reduction, bitwise argmax over 8 lanes, bitwise parity flip, and branchless coset select.
   - **Mandatory `int64` lane width:** intermediate squared distances peak at 52 bits for inputs in $[-2^{24}, +2^{24}]$ (`int32` would overflow).
   - Verified `48/48` audit vectors against an independent zero-shared-code `512`-candidate (`2×256`) exhaustive per-vector brute-force oracle (**not** exhaustive over all $\mathbb{Z}^8$), with exact zero preservation `[0]*8 -> [0]*8` vs. `siliquant#16` `Code.centre` floor error `25/64` step$^2$ per zero block (`1575/64` across 63 blocks).
2. **Exact Part-E TPU v5e Weight-Only Roofline (`siliquant#25` Step 3 & `#23`):**
   - Pure-$\mathbb{Q}$ HBM bandwidth ratios at `819,000,000,000 B/s` (`ZBG128:b560s16` `9/2` bpw -> `16/15×` vs. `Q4_K_M`, `E8_Had` `447/100` bpw -> `160/149×`). `Gemma-4-E4B` uses the nominal rounded `4,000,000,000` parameter placeholder (not an exact GGUF tensor sum).
3. **Exact Operator Bridges (Strict Law 5, Zero Metaphor):**
   - **Turrigiano multiplicative scaling vs. `Code.centre` additive shift:** `0/1` ratio distortion vs. `5/36`.
   - **Barkai–Leibler integral overload control vs. proportional droop:** `0/1` deadbeat steady-state error vs. `1/16`.
   - **Kirchhoff–Schur $R_{\mathrm{eff}}$ vs. `persist/` MST:** separates identical-MST `[5,5,5,5,5]` tree vs. redundant bipartite layers by `3.22×` (`3/5` vs. `61/327`, cut-bridge `SINGULAR_MATRIX` vs. `73/336`), honestly noting $O((a+b)^3)$ vs. MST's $O((a+b)^2)$.
   - **al-Bīrūnī dot-product cancellation (`21/1` amplification on matched-$\ell_1$ control) & al-Kāshī contraction (`(1/2)^8 * 2 = 1/128` vs. divergent `6561/128`).**

## 3. Exact Offline Reproduction (3-Mode Isolated `-I` R4 Parity)

From this directory (Python 3 standard library only, zero external packages, zero network calls):

```bash
# 1. Offline standalone verification (recomputes all 4 exact pure-Q pillars, 3-mode bit-identical 81fdc286...):
python3 -I -B siliquant_bridge_audit.py --standalone --json | sha256sum
python3 -I -O -B siliquant_bridge_audit.py --standalone --json | sha256sum
python3 -I -B -W error siliquant_bridge_audit.py --standalone --json | sha256sum

# 2. Assertion-free R5, branchless E8 brute-force, & temporary-git-repo live-path suite (6/6 OK):
python3 -I -O -B -W error test_siliquant_bridge_r5.py
```

**Offline vs. Live-Repo Note:** `--standalone --json` and `test_siliquant_bridge_r5.py` run 100% offline from any fresh clone of `main`. [`siliquant_bridge_certificate.json`](siliquant_bridge_certificate.json) (`62338fa9...`) is the pinned `--live` certificate verifying `origin/main @ 67b29a4e` and PRs `#12` (`20d13d5`, `shape`/`perm`/`persist` executed live; `lattice/test_lattices.py` explicitly marked `pr12_lattice_tests_executed_in_live_check: false` in the pure-stdlib run), `#24` (`d4c54e43`), `#26` (`7eb484d7`), and `#27` (`8cf8d244`). External `aletheia v2.0.0` verification receipts: `68/68` (`59` oracle + `9` audit, `100.0%`) mutants killed + `68/68` (`55+13`, `100.0%`) AST `BoolOp(And)` conjuncts killed (`0` survivors), `audit-check --strict-ancestors --safety-scan PASS` (`0` defects, `0` safety findings).
