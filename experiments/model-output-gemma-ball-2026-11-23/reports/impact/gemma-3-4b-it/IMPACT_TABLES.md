# Descriptive impact calculations: gemma-3-4b-it

Positive KL gain and top-1 Δ favor the candidate. Comparisons are conditional on the inputs; KL SE uses marginal quadrature and does not establish paired significance. Top-1 and PPL deltas are descriptive; this report does not apply an uncertainty screen to them.

## wikitext

Stock comparisons ranked by absolute relative KL effect. PPL columns are percent change from BF16 (negative is lower PPL), candidate / baseline.

| Candidate → baseline | Actual block bpw (candidate / baseline) | KL ratio | KL gain | Gain / SE | ≥2 SE? | Top-1 Δ pp | PPL vs BF16 (%) |
|---|---:|---:|---:|---:|---|---:|---:|
| hZBG128:b624s16 → file:Q4_K_M | 5.0042 / 4.8151 | 0.250 | +0.113440 | +20.89 | yes | +6.569 | +7.61 / +2.64 |
| hZBG128:b560s16 → file:Q4_0 | 4.5040 / 4.5000 | 0.319 | +0.152741 | +21.83 | yes | +7.382 | +2.57 / +2.31 |
| hZBG128:b560s16+K10 → file:Q4_0 | 4.5072 / 4.5000 | 0.319 | +0.152723 | +21.77 | yes | +7.353 | +2.62 / +2.31 |
| hZBG128:b560s16~rand → file:Q4_0 | 4.5040 / 4.5000 | 0.358 | +0.144033 | +20.30 | yes | +7.137 | +5.66 / +2.31 |
| ZBG128:b624s16 → file:Q4_K_M | 5.0000 / 4.8151 | 0.387 | +0.092768 | +16.79 | yes | +5.148 | +4.45 / +2.64 |
| EBG128:b560s16 → file:Q4_0 | 4.4999 / 4.5000 | 0.420 | +0.130102 | +18.24 | yes | +6.196 | +8.50 / +2.31 |
| ZB128@R:32 → file:Q4_K_M | 5.0049 / 4.8151 | 0.423 | +0.087286 | +15.23 | yes | +4.863 | +4.70 / +2.64 |
| hZBG128:b624s16 → file:Q4_K_M-imat | 5.0042 / 4.8151 | 0.428 | +0.050498 | +12.87 | yes | +3.019 | +7.61 / +5.69 |
| HBG128:b560s16 → file:Q4_0 | 4.4999 / 4.5000 | 0.449 | +0.123711 | +16.65 | yes | +5.647 | +5.01 / +2.31 |
| ZBG128:b560s16+K10 → file:Q4_0 | 4.5029 / 4.5000 | 0.487 | +0.115006 | +15.58 | yes | +5.353 | +4.81 / +2.31 |
| ZBG128:b560s16 → file:Q4_0 | 4.4998 / 4.5000 | 0.502 | +0.111785 | +15.01 | yes | +5.362 | +4.18 / +2.31 |
| ZBG128:b624s16 → file:Q4_K_M-imat | 5.0000 / 4.8151 | 0.662 | +0.029826 | +7.35 | yes | +1.598 | +4.45 / +5.69 |
| ZB128@R:32 → file:Q4_K_M-imat | 5.0049 / 4.8151 | 0.724 | +0.024344 | +5.62 | yes | +1.313 | +4.70 / +5.69 |
| VE8Z32:16 → file:Q4_0 | 4.5000 / 4.5000 | 0.774 | +0.050650 | +6.10 | yes | +2.617 | +5.16 / +2.31 |

### Matched KL (interpolation, not measured iso-quality)

| Family / target | Matched block bpw | Saved bpw | Block storage saved | Basis |
|---|---:|---:|---:|---|
| Plain / file:Q4_0 | 3.9999 | ≥0.5001 | ≥11.11% | bound |
| Plain / file:Q4_K_M | 4.3123 | 0.5028 | 10.44% | interpolated |
| Plain / file:Q4_K_M-imat | 4.7241 | 0.0910 | 1.89% | interpolated |
| Real matrix / file:Q4_0 | 4.0042 | ≥0.4958 | ≥11.02% | bound |
| Real matrix / file:Q4_K_M | 4.0042 | ≥0.8109 | ≥16.84% | bound |
| Real matrix / file:Q4_K_M-imat | 4.3644 | 0.4507 | 9.36% | interpolated |

### Calibration contrasts

| Candidate → comparator | Δ block bpw | KL gain | Top-1 Δ pp | PPL Δ |
|---|---:|---:|---:|---:|
| hZBG128:b560s16 → hZBG128:b560s16~rand | +0.0000 | +0.008708 | +0.245 | -0.473 |
| hZBG128:b560s16 → HBG128:b560s16 | +0.0041 | +0.029030 | +1.735 | -0.374 |
| hZBG128:b560s16 → VE8Z32:16 | +0.0040 | +0.102091 | +4.765 | -0.398 |
| hZBG128:b560s16~rand → HBG128:b560s16 | +0.0041 | +0.020322 | +1.490 | +0.099 |
| hZBG128:b560s16~rand → VE8Z32:16 | +0.0040 | +0.093383 | +4.520 | +0.076 |

## python

Stock comparisons ranked by absolute relative KL effect. PPL columns are percent change from BF16 (negative is lower PPL), candidate / baseline.

| Candidate → baseline | Actual block bpw (candidate / baseline) | KL ratio | KL gain | Gain / SE | ≥2 SE? | Top-1 Δ pp | PPL vs BF16 (%) |
|---|---:|---:|---:|---:|---|---:|---:|
| hZBG128:b560s16 → file:Q4_0 | 4.5040 / 4.5000 | 0.295 | +0.074496 | +8.28 | yes | +3.865 | +5.17 / +1.57 |
| hZBG128:b560s16+K10 → file:Q4_0 | 4.5072 / 4.5000 | 0.298 | +0.074149 | +8.19 | yes | +4.033 | +5.14 / +1.57 |
| hZBG128:b624s16 → file:Q4_K_M | 5.0042 / 4.8151 | 0.306 | +0.045175 | +8.21 | yes | +2.633 | +5.75 / +2.31 |
| ZBG128:b624s16 → file:Q4_K_M | 5.0000 / 4.8151 | 0.371 | +0.040955 | +7.40 | yes | +2.128 | +5.97 / +2.31 |
| EBG128:b560s16 → file:Q4_0 | 4.4999 / 4.5000 | 0.378 | +0.065646 | +7.20 | yes | +3.193 | +6.36 / +1.57 |
| hZBG128:b560s16~rand → file:Q4_0 | 4.5040 / 4.5000 | 0.389 | +0.064545 | +6.81 | yes | +3.529 | +5.87 / +1.57 |
| ZBG128:b560s16+K10 → file:Q4_0 | 4.5029 / 4.5000 | 0.389 | +0.064511 | +6.97 | yes | +3.641 | +9.96 / +1.57 |
| HBG128:b560s16 → file:Q4_0 | 4.4999 / 4.5000 | 0.391 | +0.064301 | +6.51 | yes | +3.193 | +5.42 / +1.57 |
| ZBG128:b560s16 → file:Q4_0 | 4.4998 / 4.5000 | 0.397 | +0.063625 | +6.83 | yes | +3.249 | +10.64 / +1.57 |
| hZBG128:b624s16 → file:Q4_K_M-imat | 5.0042 / 4.8151 | 0.478 | +0.021784 | +5.61 | yes | +1.849 | +5.75 / +5.70 |
| ZB128@R:32 → file:Q4_K_M | 5.0049 / 4.8151 | 0.478 | +0.034001 | +5.90 | yes | +1.344 | +6.74 / +2.31 |
| ZBG128:b624s16 → file:Q4_K_M-imat | 5.0000 / 4.8151 | 0.579 | +0.017564 | +4.47 | yes | +1.344 | +5.97 / +5.70 |
| ZB128@R:32 → file:Q4_K_M-imat | 5.0049 / 4.8151 | 0.746 | +0.010610 | +2.50 | yes | +0.560 | +6.74 / +5.70 |
| VE8Z32:16 → file:Q4_0 | 4.5000 / 4.5000 | 0.848 | +0.016049 | +1.39 | no | +0.784 | +8.45 / +1.57 |

### Matched KL (interpolation, not measured iso-quality)

| Family / target | Matched block bpw | Saved bpw | Block storage saved | Basis |
|---|---:|---:|---:|---|
| Plain / file:Q4_0 | 3.9999 | ≥0.5001 | ≥11.11% | bound |
| Plain / file:Q4_K_M | 4.2859 | 0.5292 | 10.99% | interpolated |
| Plain / file:Q4_K_M-imat | 4.5076 | 0.3075 | 6.39% | interpolated |
| Real matrix / file:Q4_0 | 4.0042 | ≥0.4958 | ≥11.02% | bound |
| Real matrix / file:Q4_K_M | 4.1027 | 0.7124 | 14.79% | interpolated |
| Real matrix / file:Q4_K_M-imat | 4.3790 | 0.4361 | 9.06% | interpolated |

### Calibration contrasts

| Candidate → comparator | Δ block bpw | KL gain | Top-1 Δ pp | PPL Δ |
|---|---:|---:|---:|---:|
| hZBG128:b560s16 → hZBG128:b560s16~rand | +0.0000 | +0.009951 | +0.336 | -0.035 |
| hZBG128:b560s16 → HBG128:b560s16 | +0.0041 | +0.010195 | +0.672 | -0.013 |
| hZBG128:b560s16 → VE8Z32:16 | +0.0040 | +0.058447 | +3.081 | -0.163 |
| hZBG128:b560s16~rand → HBG128:b560s16 | +0.0041 | +0.000244 | +0.336 | +0.022 |
| hZBG128:b560s16~rand → VE8Z32:16 | +0.0040 | +0.048496 | +2.745 | -0.129 |

## Illustrative deployment scenarios

Anchor: 0.4361 saved block bpw at the Q4_K_M-imat KL target (smaller of two text estimates). These are assumptions, not a fleet forecast.

| Scenario | Resident GB (base → new) | Footprint reduction | H200 token/s roofline (base → new) | Fleet equivalent reduction | Annual gross opportunity |
|---|---:|---:|---:|---:|---:|
| Low | 27.15 → 27.08 | 0.25% | 559.5 → 563.9 | 0.19% | USD 3,410 |
| Mid | 41.91 → 41.29 | 1.49% | 182.5 → 187.0 | 1.19% | USD 312,595 |
| High | 67.03 → 63.40 | 5.41% | 85.1 → 90.9 | 4.82% | USD 21,104,063 |

Input: runner JSONL. Local SHA-256: 5895d850aff01a154a3b57e1c47468c90121e34316fa6e850ae866f6ec4dc23e
