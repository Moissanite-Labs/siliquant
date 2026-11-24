# Experimenter record: Gemma closure on ccf045d + 2026-11-24 addendum arms (M1 Max, 2026-11-24)

Session record for the runs committed in `experiments/model-output-gemma-ball-2026-11-23/out/ccf045d/`
on branch `exp/gemma-ball-results-ccf045d`, plus the runner port on `exp/gemma-a16-port`.
Written per `experiments/model-output-gemma-ball-2026-11-23/HANDOFF.md` (main) and
`notes/handoff-addendum-2026-11-24.md` (`bayan/charming-gates-jlas14` @ `2c52ec9`).

## Machine, binaries, runner pins

- M1 Max (MacBookPro18,2), 64 GB RAM, 10 cores, arm64, Darwin 27.
- llama.cpp `0.5.0 (build 11146, commit 7fe450e19)`, AppleClang arm64: `llama-cli`, `llama-quantize`,
  `llama-perplexity`, `llama-imatrix` (version recorded in the run logs).
- Python 3.14 venv with `numpy 2.5.5`, `gguf`; C compiled with `cc -O2 -ffp-contract=off`
  (Apple clang fuses by default; the flag keeps C/NumPy parity, checked by the tests).
- Base rows runner: `ccf045d04ada2170eeafaa37254a764b96552723` (`exp/gemma-ball-runner` HEAD).
- Addendum-arm rows runner: `exp/gemma-a16-port` @ `31bbd3c` (off `ccf045d`).
- All runs: `--ngl 999` (Metal), `--threads 4`, `--workers 4`, `--chunks 40 --ctx 512 --batch 512
  --imatrix-chunks 100`, texts `wikitext,python` (bridge: `eval.txt`). Every row records
  `backend` (`metal`), `ngl` and `threads`.

## Input verification (SHA-256, all match HANDOFF.md section 2)

| input | SHA-256 |
|---|---|
| gemma-3-270m-it-F16.gguf | `140cb395a3a3dcdb3de66e44c60d7265286c9d49f951c1899ec0d8c1f16b2feb` |
| gemma-3-1b-it-f16.gguf | `05bd381a5f45611ce53f4fdcc6641cf6cec68c3091d74e8a32ea591f062d3fc5` |
| gemma-3-4b-it-BF16.gguf | `7bcfca74279608552011d3c4d11e9298934502ffa410eb21b9047a6276b87ba6` |
| gemma-3-12b-it-BF16.gguf | `276920b11d80002d45b9b85291e333d9a7798df92b148ba9ae3d0625dd9f82f2` |
| gemma-4-E4B-it-BF16.gguf | `38e0dba6818d18e2b41062c7b3b9083dcfddc44767078c415ff7e13661911ba5` |
| gemma-4-E4B-it-Q8_0.gguf | `f8854aa4480df62585a279e7ca0a881554fc18a41c59c4f62642d16a2ae47012` |
| wiki.test.raw | `173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08` |
| wiki.train.raw | `6707892fa3788b5ab9ed78ab5ff37d9fe825f6011a2ad4fcd6a6d467f0e7da57` |
| base.py (python text) | `e38975e1c68d98ac1664dfd530616eb35c72294382a4dd873d4746b23f27779f` |
| eval.txt (bridge) | `df5748784801c869b7d1ce54b169af665bfa4c0d3c9839c88d097ffbd33df761` |
| 4B / 12B / E4B real imatrix | `19b11624…` / `9aaa78bb…` / `b67f51e2…` (full values in HANDOFF.md §2; cached imatrix files hashed and matched before reuse) |

## The port (exp/gemma-a16-port @ 31bbd3c)

`run.py` extended per the addendum's "Changes to the queue":

- `HAMMING_BALL_RE` arms `A16BG128:b560s16`, `hA16BG128:b560s16` (also `i…~rand`, `A8/A32`),
  rounded by the new `group_hamming_ball_round` using the vendored `hamming.py` /
  `hamming_dp.c` (copied unchanged from `experiments/hamming-lattices-real-weights-2026-11-24`
  on `bayan/charming-gates-jlas14`): `hamming.closest(X, m)` as the nearest point,
  `hamming.ball_bound(m, n, B)` for the bound and point count, `sqrt(hamming.covering_sq(m, n))`
  as the bisection margin in place of `sqrt(n)/2`. Verified bit-for-bit against the branch's
  `ball_round` on the same inputs; ball-bound invariant checked (bound 5213, max group size 5204
  at b560); C decoder == NumPy decoder bit-for-bit.
- `STOCK16_RE` arms `file:Q4_0@16`, `file:Q4_K_M@16`, `file:Q4_K_M-imat@16`: the stock quantized
  file's block matrices decoded to 16-bit floats inside a copy of the 16-bit model (the
  addendum's activation-rounding control, as `baselines.py` does on the other models).
  Rows carry `source_file_sha256` and `other_tensors_that_differ == []`.
- `backend`, `ngl`, `threads` added to every row.
- `PREREG-ADDENDUM-2026-11-24.md` committed **before** any addendum-arm data (plan before data);
  `checks/plan_before_data.py` passes with the data committed on top.
- Tests: 13 tests in `test_run.py` pass in all 3 modes (`-B`, `-O -B`, `-B -W error`), including
  the new `TestAddendumArms`; the vendored hamming suites (29 tests) pass on this machine.
  macOS `spawn` path validated with a `--metrics-only` A_16 build on gemma-3-270m-it
  (18.6 s, 4 workers, real imatrix `f68b5c56…` loaded, `imatrix_fallback_count == 0`).

## Reproducible run commands (generic paths; out-dirs under `out/ccf045d/`)

```bash
git checkout --detach ccf045d04ada2170eeafaa37254a764b96552723   # base rows
# Priority 1a — tensor-matched stock reruns (fresh out-dir; cached imatrix/kld reused):
python3 experiments/model-output-gemma-ball-2026-11-23/run.py \
  --model-id gemma-3-4b-it --model-path <MODELS_DIR>/gemma-3-4b-it-BF16.gguf \
  --out-dir <OUT_DIR>/gemma-3-4b-it-stock-matched --texts wikitext,python \
  --wikitext-test <DATA_DIR>/wiki.test.raw --wikitext-train <DATA_DIR>/wiki.train.raw \
  --python-text <DATA_DIR>/base.py --arms file:Q4_0,file:Q4_K_M,file:Q4_K_M-imat \
  --ngl 999 --chunks 40 --imatrix-chunks 100 --ctx 512 --batch 512 --threads 4 --workers 4
# (same for gemma-3-12b-it)
# Priority 1b/1c — hEBG/hHBG arms:
#   --arms hEBG128:b496s16,hEBG128:b560s16,hHBG128:b560s16   (4B and 12B)
# Priority 1d — E4B BF16 full sweep (default 17 arms) after --pilot-only --pilot-layer 0, then
#   --bridge-eval <DATA_DIR>/eval.txt --model-path .../gemma-4-E4B-it-Q8_0.gguf
#   --arms VE832:16,VE8Z32:16,EBG128:b560s16
# Addendum arms from exp/gemma-a16-port @ 31bbd3c (same flags):
#   stock-matched dirs += --arms file:Q4_K_M@16,file:Q4_K_M-imat@16
#   heberg dirs        += --arms A16BG128:b560s16,hA16BG128:b560s16
#   E4B sweep dir      += --arms A16BG128:b560s16
```

Operational gotcha hit once: a killed background launch left a zero-header partial
`*-Q4_0.gguf` in the cache, and `ensure_baseline_gguf` trusts existence, so the job crashed in
`compare_file` ("GGUF magic invalid"). Fix applied: delete partial cache files and re-run
(resume logic keys on `(model_id, text, variant)`). Recommendation: validate GGUF magic before
trusting a cached baseline. Runner commits were only switched between model jobs, never while
`Pool` workers were live (HANDOFF.md §3 gotcha 5).

## Results

Raw rows: `experiments/model-output-gemma-ball-2026-11-23/out/ccf045d/*/results.jsonl`
(+ `results.out`), with per-row provenance (`model_sha256`, `imatrix_sha256`, `text_sha256`,
`file_sha256` / `source_file_sha256`, `backend`, `ngl`, `threads`, timestamps).

### 4B stock-matched (ccf045d) + @16 rows
| variant | text | bpw | KL ± SE | PPL | top-1 | other_tensors_that_differ |
|---|---|---:|---:|---:|---:|---|
| `file:Q4_0` | wikitext | 4.5000 | 0.224384 ± 0.006349 | 15.6780 | 83.480 | [] |
| `file:Q4_0` | python | 4.5000 | 0.105576 ± 0.008431 | 5.0700 | 91.429 | [] |
| `file:Q4_K_M` | wikitext | 4.8151 | 0.151277 ± 0.004987 | 15.7290 | 86.735 | [] |
| `file:Q4_K_M` | python | 4.8151 | 0.065077 ± 0.005131 | 5.1072 | 93.725 | [] |
| `file:Q4_K_M-imat` | wikitext | 4.8151 | 0.088341 ± 0.003284 | 16.1957 | 90.255 | [] |
| `file:Q4_K_M-imat` | python | 4.8151 | 0.041738 ± 0.003350 | 5.2770 | 94.566 | [] |
| `file:Q4_K_M@16` | wikitext | 4.8151 | 0.150770 ± 0.004986 | 15.7519 | 86.647 | [] |
| `file:Q4_K_M@16` | python | 4.8151 | 0.063700 ± 0.005028 | 5.1038 | 93.613 | [] |
| `file:Q4_K_M-imat@16` | wikitext | 4.8151 | 0.087541 ± 0.003271 | 16.1469 | 90.284 | [] |
| `file:Q4_K_M-imat@16` | python | 4.8151 | 0.040447 ± 0.003246 | 5.2507 | 94.678 | [] |

### 4B hEBG/hHBG/A16 arms (ccf045d)
| variant | text | bpw | KL ± SE | PPL | top-1 | other_tensors_that_differ |
|---|---|---:|---:|---:|---:|---|
| `hEBG128:b496s16` | wikitext | 4.0035 | 0.125145 ± 0.004552 | 17.0110 | 88.618 | — |
| `hEBG128:b496s16` | python | 4.0035 | 0.063346 ± 0.007478 | 5.2079 | 93.501 | — |
| `hEBG128:b560s16` | wikitext | 4.5042 | 0.060492 ± 0.002623 | 16.2185 | 91.912 | — |
| `hEBG128:b560s16` | python | 4.5042 | 0.026659 ± 0.002246 | 5.1553 | 95.294 | — |
| `hHBG128:b560s16` | wikitext | 4.5042 | 0.215529 ± 0.006597 | 17.4784 | 84.990 | — |
| `hHBG128:b560s16` | python | 4.5042 | 0.128176 ± 0.011700 | 5.8341 | 91.204 | — |
| `A16BG128:b560s16` | wikitext | 4.4997 | 0.097443 ± 0.004050 | 16.1610 | 90.078 | — |
| `A16BG128:b560s16` | python | 4.4997 | 0.045935 ± 0.004297 | 5.4265 | 94.118 | — |
| `hA16BG128:b560s16` | wikitext | 4.5040 | 0.063442 ± 0.002762 | 16.4434 | 92.137 | — |
| `hA16BG128:b560s16` | python | 4.5040 | 0.034266 ± 0.004796 | 5.2824 | 94.678 | — |

### 12B stock-matched (ccf045d) + @16 rows
| variant | text | bpw | KL ± SE | PPL | top-1 | other_tensors_that_differ |
|---|---|---:|---:|---:|---:|---|
| `file:Q4_0` | wikitext | 4.5000 | 0.123430 ± 0.003244 | 8.5454 | 87.216 | [] |
| `file:Q4_0` | python | 4.5000 | 0.081971 ± 0.008488 | 4.1819 | 93.277 | [] |
| `file:Q4_K_M` | wikitext | 4.8076 | 0.073696 ± 0.001969 | 8.5725 | 89.843 | [] |
| `file:Q4_K_M` | python | 4.8076 | 0.047705 ± 0.005441 | 4.1797 | 93.669 | [] |
| `file:Q4_K_M-imat` | wikitext | 4.8076 | 0.048832 ± 0.001657 | 8.7939 | 92.333 | [] |
| `file:Q4_K_M-imat` | python | 4.8076 | 0.028239 ± 0.002487 | 4.1778 | 95.182 | [] |
| `file:Q4_K_M@16` | wikitext | 4.8076 | 0.073281 ± 0.001971 | 8.5781 | 89.951 | [] |
| `file:Q4_K_M@16` | python | 4.8076 | 0.046623 ± 0.005404 | 4.1641 | 93.782 | [] |
| `file:Q4_K_M-imat@16` | wikitext | 4.8076 | 0.048119 ± 0.001637 | 8.7811 | 92.196 | [] |
| `file:Q4_K_M-imat@16` | python | 4.8076 | 0.027851 ± 0.002462 | 4.1800 | 95.406 | [] |

### 12B hEBG/hHBG/A16 arms (ccf045d)
| variant | text | bpw | KL ± SE | PPL | top-1 | other_tensors_that_differ |
|---|---|---:|---:|---:|---:|---|
| `hEBG128:b496s16` | wikitext | 4.0020 | 0.062662 ± 0.002257 | 8.9364 | 90.833 | — |
| `hEBG128:b496s16` | python | 4.0020 | 0.038792 ± 0.003562 | 4.2316 | 94.734 | — |
| `hEBG128:b560s16` | wikitext | 4.5027 | 0.031715 ± 0.001078 | 8.7685 | 93.539 | — |
| `hEBG128:b560s16` | python | 4.5027 | 0.019372 ± 0.001694 | 4.1705 | 95.966 | — |
| `hHBG128:b560s16` | wikitext | 4.5027 | 0.094440 ± 0.002964 | 8.8520 | 88.578 | — |
| `hHBG128:b560s16` | python | 4.5027 | 0.055933 ± 0.005466 | 4.1426 | 94.622 | — |
| `A16BG128:b560s16` | wikitext | 4.4997 | 0.046633 ± 0.001419 | 8.6972 | 91.667 | — |
| `A16BG128:b560s16` | python | 4.4997 | 0.026973 ± 0.002502 | 4.2239 | 95.686 | — |
| `hA16BG128:b560s16` | wikitext | 4.5025 | 0.029509 ± 0.000925 | 8.7857 | 93.598 | — |
| `hA16BG128:b560s16` | python | 4.5025 | 0.021578 ± 0.002579 | 4.1723 | 96.134 | — |

### E4B BF16 sweep headline rows (ccf045d)
| variant | text | bpw | KL ± SE | PPL | top-1 | other_tensors_that_differ |
|---|---|---:|---:|---:|---:|---|
| `file:Q4_0` | wikitext | 4.5000 | 0.219758 ± 0.004194 | 53.8566 | 80.902 | [] |
| `file:Q4_0` | python | 4.5000 | 0.100907 ± 0.008341 | 5.8247 | 91.317 | [] |
| `file:Q4_K_M` | wikitext | 4.8029 | 0.127872 ± 0.002591 | 52.7462 | 85.392 | [] |
| `file:Q4_K_M` | python | 4.8029 | 0.055301 ± 0.003817 | 5.8319 | 93.445 | [] |
| `file:Q4_K_M-imat` | wikitext | 4.8029 | 0.096909 ± 0.002209 | 51.8032 | 87.412 | [] |
| `file:Q4_K_M-imat` | python | 4.8029 | 0.044886 ± 0.003242 | 5.9721 | 94.454 | [] |
| `ZBG128:b560s16` | wikitext | 4.4998 | 0.083364 ± 0.001828 | 52.0190 | 88.333 | — |
| `ZBG128:b560s16` | python | 4.4998 | 0.039367 ± 0.003180 | 6.0424 | 94.454 | — |
| `hZBG128:b560s16` | wikitext | 4.5038 | 0.060614 ± 0.001529 | 50.4007 | 89.990 | — |
| `hZBG128:b560s16` | python | 4.5038 | 0.034911 ± 0.003045 | 5.7822 | 95.350 | — |
| `EBG128:b560s16` | wikitext | 4.4999 | 0.069178 ± 0.001550 | 48.4888 | 89.235 | — |
| `EBG128:b560s16` | python | 4.4999 | 0.031549 ± 0.002734 | 5.8832 | 95.518 | — |
| `HBG128:b560s16` | wikitext | 4.4999 | 0.072232 ± 0.001699 | 51.9329 | 89.324 | — |
| `HBG128:b560s16` | python | 4.4999 | 0.036112 ± 0.002887 | 6.0181 | 94.734 | — |
| `A16BG128:b560s16` | wikitext | 4.4997 | 0.070451 ± 0.001584 | 46.9183 | 88.471 | — |
| `A16BG128:b560s16` | python | 4.4997 | 0.030574 ± 0.002019 | 5.9048 | 95.014 | — |

### E4B Q8_0 bridge (separate stratum)
| variant | text | bpw | KL ± SE | PPL | top-1 | other_tensors_that_differ |
|---|---|---:|---:|---:|---:|---|
| `VE832:16` | bridge | 4.5000 | 0.074682 ± 0.002264 | 7.9029 | 91.559 | — |
| `VE8Z32:16` | bridge | 4.5000 | 0.077252 ± 0.002314 | 7.7097 | 91.608 | — |
| `EBG128:b560s16` | bridge | 4.4999 | 0.038324 ± 0.001272 | 7.9386 | 94.029 | — |

### 4B provisional reference rows (ca10cda)
| variant | text | bpw | KL ± SE | PPL | top-1 | other_tensors_that_differ |
|---|---|---:|---:|---:|---:|---|
| `file:Q4_K_M-imat` | wikitext | 4.8151 | 0.088353 ± 0.003285 | 16.1966 | 90.275 | ['token_embd.weight'] |
| `file:Q4_K_M-imat` | python | 4.8151 | 0.041692 ± 0.003339 | 5.2762 | 94.566 | ['token_embd.weight'] |
| `hZBG128:b560s16` | wikitext | 4.5040 | 0.071645 ± 0.002942 | 15.7187 | 90.853 | — |
| `hZBG128:b560s16` | python | 4.5040 | 0.031101 ± 0.003135 | 5.2497 | 95.294 | — |
| `hZBG128:b496s16` | wikitext | 4.0042 | 0.131462 ± 0.004690 | 15.8903 | 88.333 | — |
| `hZBG128:b496s16` | python | 4.0042 | 0.073431 ± 0.008746 | 5.0620 | 93.221 | — |
| `HBG128:b560s16` | wikitext | 4.4999 | 0.100675 ± 0.003861 | 16.0929 | 89.118 | — |
| `HBG128:b560s16` | python | 4.4999 | 0.041296 ± 0.005152 | 5.2623 | 94.622 | — |

### 12B provisional reference rows (ca10cda)
| variant | text | bpw | KL ± SE | PPL | top-1 | other_tensors_that_differ |
|---|---|---:|---:|---:|---:|---|
| `file:Q4_K_M-imat` | wikitext | 4.8076 | 0.048831 ± 0.001656 | 8.7933 | 92.284 | ['token_embd.weight'] |
| `file:Q4_K_M-imat` | python | 4.8076 | 0.028232 ± 0.002489 | 4.1773 | 95.238 | ['token_embd.weight'] |
| `hZBG128:b560s16` | wikitext | 4.5026 | 0.037095 ± 0.001513 | 8.7437 | 93.245 | — |
| `hZBG128:b560s16` | python | 4.5026 | 0.023266 ± 0.002128 | 4.1541 | 96.022 | — |
| `hZBG128:b496s16` | wikitext | 4.0027 | 0.072152 ± 0.002361 | 8.9027 | 90.676 | — |
| `hZBG128:b496s16` | python | 4.0027 | 0.051410 ± 0.005313 | 4.1717 | 94.846 | — |
| `HBG128:b560s16` | wikitext | 4.4999 | 0.049364 ± 0.001764 | 8.7507 | 91.588 | — |
| `HBG128:b560s16` | python | 4.4999 | 0.028539 ± 0.002591 | 4.1746 | 96.078 | — |

## Derived verdicts (wikitext unless noted)

**PRED-M1-02** (tensor-matched vs provisional ca10cda Q4_K_M-imat, |dKL| < 0.002 gate):
- 4B wikitext: matched 0.088341 vs provisional 0.088353 -> dKL = 0.000012 CONFIRMED
- 4B python: matched 0.041738 vs provisional 0.041692 -> dKL = 0.000046 CONFIRMED
- 12B wikitext: matched 0.048832 vs provisional 0.048831 -> dKL = 0.000001 CONFIRMED
- 12B python: matched 0.028239 vs provisional 0.028232 -> dKL = 0.000007 CONFIRMED

**@16 activation-rounding share on Metal** (KL(file:X) - KL(file:X@16), positive = 16-bit decode better):
- 4B wikitext file:Q4_K_M: 0.151277 -> 0.150770, d = 0.000507
- 4B python file:Q4_K_M: 0.065077 -> 0.063700, d = 0.001377
- 4B wikitext file:Q4_K_M-imat: 0.088341 -> 0.087541, d = 0.000800
- 4B python file:Q4_K_M-imat: 0.041738 -> 0.040447, d = 0.001291
- 12B wikitext file:Q4_K_M: 0.073696 -> 0.073281, d = 0.000415
- 12B python file:Q4_K_M: 0.047705 -> 0.046623, d = 0.001082
- 12B wikitext file:Q4_K_M-imat: 0.048832 -> 0.048119, d = 0.000713
- 12B python file:Q4_K_M-imat: 0.028239 -> 0.027851, d = 0.000388

**PRED-M1-01 (4B)**:
- hEBG b560 KL = 0.060492 ± 0.002623 (forecast 0.061685 theor / 0.059990 empir)
- vs hZBG b560 0.071645: dKL = 0.011153 = 2.83 combined SE (gate >= 2.0: PASS)
- bracket: KL(b560)=0.060492 < KL(stock)=0.088341 < KL(b496)=0.125145 -> b_match = 4.2434 bpw, savings = 11.87% (gate > 12%: FAIL -> falsifier (ii))

**1c 12B transfer**:
- hEBG b560 0.031715 vs hZBG b560 0.037095: dKL = 0.005380 = 2.90 SE (gate >= 2.0: PASS)
- bracket vs matched stock (b_stock=4.8076, KL=0.048832): b_match = 4.1854, savings = 12.94%

**A_16 out-of-family on Gemma (prereg A1: A16 below HBG by >= 2.0 SE on wikitext)**:
- 4B: HBG 0.100675 vs A16 0.097443 -> dKL = 0.003232 = 0.58 SE -> not >= 2.0 (non-transfer)
- 12B: HBG 0.049364 vs A16 0.046633 -> dKL = 0.002731 = 1.21 SE -> not >= 2.0 (non-transfer)

**A2 (hA16 vs A16)**:
- 4B wikitext: A16 0.097443 vs hA16 0.063442 -> hA16 better by 0.034001 = 6.94 SE (gate >= 2.0: PASS)
- 12B wikitext: A16 0.046633 vs hA16 0.029509 -> hA16 better by 0.017124 = 10.11 SE (gate >= 2.0: PASS)
- E4B wikitext: A16 0.070451 vs hZBG b560 0.060614 -> A16 worse by -4.47 SE

## Priority 3 packed-format benchmark on the M1 Max (hampack vs ballpack)

24 block matrices sampled evenly across the 238 of `gemma-3-4b-it-BF16.gguf` (3.21B block
weights total), encoded with `shape/ballpack.Format(32, 136, 16)` (Z^32) and
`shape/hampack.HamFormat(16, 32, 136, 16)` (A_16 x A_16) from `bayan/charming-gates-jlas14`,
decoded by the C coders (`ballpack.c` / `hampack.c`, `cc -O2 -ffp-contract=off`) on one core
(`time.process_time`, best of 3 repeats), uncontended:

| metric | ballpack Z^32 | hampack A16xA16 |
|---|---:|---:|
| file bytes per weight | 0.5625 (4.5 bits) | 0.5625 (4.5 bits) — ratio 1.000 |
| decode ns/weight (1 core) | 27.0 | 41.7 — **1.55x slower** |
| decode count table | 0.62 MB (77,616 entries) | 30.6 MB (1,275,648 entries) |
| readback bit-for-bit | OK | OK |

The Linux cloud container measured 2.5x; the M1 Max's large caches narrow the gap to 1.55x, as
the addendum predicted ("The M1 Max's large caches may change the 2.5 times gap"). Under CPU
contention with the sweep queues the same measurement gave 1.43x (30.0 vs 42.7 ns/w), so treat
1.4-1.6x as the range. Benchmark script: `notes/bench_hampack_ballpack_m1_2026-11-24.py`.

## Verification gates run before committing

- `python3 -B / -O -B / -B -W error -m unittest experiments/model-output-gemma-ball-2026-11-23/test_run.py`
  — 13 tests OK in all 3 modes (on `exp/gemma-a16-port`).
- Vendored hamming suites on `bayan/charming-gates-jlas14`: `test_hamming.py`,
  `shape/test_hammingindex.py`, `shape/test_hampack.py` — 29 tests OK.
- `checks/plan_before_data.py` and `checks/claim_ledger.py` in all 3 modes — clean, before and
  after the results commit on this branch.
- No weights or datasets committed; no local paths, secrets, or internal names in pushed files.

## Not run

- Priority 1e (`gemma-3-27b-it`): no 27B GGUF on this machine; on a 64 GB host it must use a
  `Q8_0` reference only (never BF16) and stay in its own stratum (HANDOFF.md §3).
