# Descriptive impact calculations: gemma-3-270m-it

Positive KL gain and top-1 Δ favor the candidate. Comparisons are conditional on the inputs; KL SE uses marginal quadrature and does not establish paired significance. Top-1 and PPL deltas are descriptive; this report does not apply an uncertainty screen to them.

## wikitext

Stock comparisons ranked by absolute relative KL effect. PPL columns are percent change from F16 (negative is lower PPL), candidate / baseline.

| Candidate → baseline | Actual block bpw (candidate / baseline) | KL ratio | KL gain | Gain / SE | ≥2 SE? | Top-1 Δ pp | PPL vs F16 (%) |
|---|---:|---:|---:|---:|---|---:|---:|
| hZBG128:b560s16+K10 → file:Q4_0 | 4.5310 / 4.5000 | 0.148 | +0.427336 | +53.75 | yes | +17.873 | +10.15 / +38.09 |
| hZBG128:b560s16 → file:Q4_0 | 4.5178 / 4.5000 | 0.152 | +0.425511 | +53.42 | yes | +17.785 | +9.64 / +38.09 |
| hZBG128:b560s16~rand → file:Q4_0 | 4.5178 / 4.5000 | 0.173 | +0.414570 | +52.28 | yes | +17.343 | +13.05 / +38.09 |
| hZBG128:b624s16 → file:Q4_K_M | 5.0180 / 5.4338 | 0.238 | +0.116990 | +39.08 | yes | +8.539 | +7.97 / +7.16 |
| HBG128:b560s16 → file:Q4_0 | 4.4999 / 4.5000 | 0.348 | +0.326845 | +39.16 | yes | +11.451 | +10.62 / +38.09 |
| EBG128:b560s16 → file:Q4_0 | 4.4999 / 4.5000 | 0.357 | +0.322624 | +38.45 | yes | +11.471 | +28.84 / +38.09 |
| ZBG128:b560s16+K10 → file:Q4_0 | 4.5130 / 4.5000 | 0.384 | +0.308875 | +36.62 | yes | +10.245 | +13.76 / +38.09 |
| ZBG128:b560s16 → file:Q4_0 | 4.4998 / 4.5000 | 0.388 | +0.306773 | +36.57 | yes | +9.951 | +9.97 / +38.09 |
| hZBG128:b624s16 → file:Q4_K_M-imat | 5.0180 / 5.4338 | 0.399 | +0.055040 | +26.95 | yes | +4.833 | +7.97 / +4.84 |
| ZBG128:b624s16 → file:Q4_K_M | 5.0000 / 5.4338 | 0.638 | +0.055658 | +16.05 | yes | +2.990 | +14.29 / +7.16 |
| ZB128@R:32 → file:Q4_K_M | 5.0198 / 5.4338 | 0.690 | +0.047669 | +13.42 | yes | +3.098 | +15.68 / +7.16 |
| VE8Z32:16 → file:Q4_0 | 4.5000 / 4.5000 | 0.786 | +0.107393 | +10.69 | yes | +3.834 | +42.68 / +38.09 |
| ZB128@R:32 → file:Q4_K_M-imat | 5.0198 / 5.4338 | 1.156 | -0.014281 | -5.10 | yes | -0.608 | +15.68 / +4.84 |
| ZBG128:b624s16 → file:Q4_K_M-imat | 5.0000 / 5.4338 | 1.069 | -0.006292 | -2.34 | yes | -0.716 | +14.29 / +4.84 |

### Matched KL (interpolation, not measured iso-quality)

| Family / target | Matched block bpw | Saved bpw | Block storage saved | Basis |
|---|---:|---:|---:|---|
| Plain / file:Q4_0 | 3.9999 | ≥0.5001 | ≥11.11% | bound |
| Plain / file:Q4_K_M | 4.7125 | 0.7213 | 13.27% | interpolated |
| Plain / file:Q4_K_M-imat | — | — | —% | not reached |
| Real matrix / file:Q4_0 | 4.0179 | ≥0.4821 | ≥10.71% | bound |
| Real matrix / file:Q4_K_M | 4.0179 | ≥1.4159 | ≥26.06% | bound |
| Real matrix / file:Q4_K_M-imat | 4.4137 | 1.0201 | 18.77% | interpolated |

### Calibration contrasts

| Candidate → comparator | Δ block bpw | KL gain | Top-1 Δ pp | PPL Δ |
|---|---:|---:|---:|---:|
| hZBG128:b560s16 → hZBG128:b560s16~rand | -0.0000 | +0.010941 | +0.442 | -1.729 |
| hZBG128:b560s16 → HBG128:b560s16 | +0.0179 | +0.098666 | +6.334 | -0.498 |
| hZBG128:b560s16 → VE8Z32:16 | +0.0178 | +0.318118 | +13.951 | -16.717 |
| hZBG128:b560s16~rand → HBG128:b560s16 | +0.0179 | +0.087725 | +5.892 | +1.231 |
| hZBG128:b560s16~rand → VE8Z32:16 | +0.0178 | +0.307177 | +13.509 | -14.988 |

## python

Stock comparisons ranked by absolute relative KL effect. PPL columns are percent change from F16 (negative is lower PPL), candidate / baseline.

| Candidate → baseline | Actual block bpw (candidate / baseline) | KL ratio | KL gain | Gain / SE | ≥2 SE? | Top-1 Δ pp | PPL vs F16 (%) |
|---|---:|---:|---:|---:|---|---:|---:|
| hZBG128:b560s16+K10 → file:Q4_0 | 4.5310 / 4.5000 | 0.172 | +0.467694 | +15.29 | yes | +13.501 | +5.01 / +42.45 |
| hZBG128:b560s16 → file:Q4_0 | 4.5178 / 4.5000 | 0.178 | +0.464274 | +15.14 | yes | +13.221 | +4.30 / +42.45 |
| hZBG128:b560s16~rand → file:Q4_0 | 4.5178 / 4.5000 | 0.185 | +0.460555 | +15.04 | yes | +12.885 | +13.35 / +42.45 |
| HBG128:b560s16 → file:Q4_0 | 4.4999 / 4.5000 | 0.296 | +0.397727 | +12.75 | yes | +10.364 | +8.85 / +42.45 |
| EBG128:b560s16 → file:Q4_0 | 4.4999 / 4.5000 | 0.330 | +0.378752 | +11.87 | yes | +10.700 | +16.99 / +42.45 |
| hZBG128:b624s16 → file:Q4_K_M | 5.0180 / 5.4338 | 0.335 | +0.110810 | +9.59 | yes | +4.986 | -1.76 / +7.71 |
| ZBG128:b560s16 → file:Q4_0 | 4.4998 / 4.5000 | 0.372 | +0.355092 | +10.86 | yes | +8.011 | +3.36 / +42.45 |
| ZBG128:b560s16+K10 → file:Q4_0 | 4.5130 / 4.5000 | 0.377 | +0.352311 | +10.63 | yes | +7.899 | +4.46 / +42.45 |
| ZBG128:b624s16 → file:Q4_K_M | 5.0000 / 5.4338 | 0.603 | +0.066098 | +5.34 | yes | +2.801 | +13.57 / +7.71 |
| hZBG128:b624s16 → file:Q4_K_M-imat | 5.0180 / 5.4338 | 0.631 | +0.032559 | +4.62 | yes | +1.120 | -1.76 / +2.07 |
| VE8Z32:16 → file:Q4_0 | 4.5000 / 4.5000 | 0.669 | +0.187086 | +5.23 | yes | +4.650 | +32.53 / +42.45 |
| ZB128@R:32 → file:Q4_K_M | 5.0198 / 5.4338 | 0.672 | +0.054591 | +4.27 | yes | +1.961 | +9.89 / +7.71 |
| ZB128@R:32 → file:Q4_K_M-imat | 5.0198 / 5.4338 | 1.268 | -0.023660 | -2.65 | yes | -1.905 | +9.89 / +2.07 |
| ZBG128:b624s16 → file:Q4_K_M-imat | 5.0000 / 5.4338 | 1.138 | -0.012153 | -1.46 | no | -1.065 | +13.57 / +2.07 |

### Matched KL (interpolation, not measured iso-quality)

| Family / target | Matched block bpw | Saved bpw | Block storage saved | Basis |
|---|---:|---:|---:|---|
| Plain / file:Q4_0 | 3.9999 | ≥0.5001 | ≥11.11% | bound |
| Plain / file:Q4_K_M | 4.6980 | 0.7358 | 13.54% | interpolated |
| Plain / file:Q4_K_M-imat | — | — | —% | not reached |
| Real matrix / file:Q4_0 | 4.0179 | ≥0.4821 | ≥10.71% | bound |
| Real matrix / file:Q4_K_M | 4.1164 | 1.3174 | 24.24% | interpolated |
| Real matrix / file:Q4_K_M-imat | 4.6562 | 0.7776 | 14.31% | interpolated |

### Calibration contrasts

| Candidate → comparator | Δ block bpw | KL gain | Top-1 Δ pp | PPL Δ |
|---|---:|---:|---:|---:|
| hZBG128:b560s16 → hZBG128:b560s16~rand | -0.0000 | +0.003719 | +0.336 | -0.759 |
| hZBG128:b560s16 → HBG128:b560s16 | +0.0179 | +0.066547 | +2.857 | -0.381 |
| hZBG128:b560s16 → VE8Z32:16 | +0.0178 | +0.277188 | +8.571 | -2.367 |
| hZBG128:b560s16~rand → HBG128:b560s16 | +0.0179 | +0.062828 | +2.521 | +0.377 |
| hZBG128:b560s16~rand → VE8Z32:16 | +0.0178 | +0.273469 | +8.235 | -1.608 |

## Illustrative deployment scenarios

Anchor: 0.7776 saved block bpw at the Q4_K_M-imat KL target (smaller of two text estimates). These are assumptions, not a fleet forecast.

| Scenario | Resident GB (base → new) | Footprint reduction | H200 token/s roofline (base → new) | Fleet equivalent reduction | Annual gross opportunity |
|---|---:|---:|---:|---:|---:|
| Low | 27.53 → 27.41 | 0.43% | 531.3 → 538.4 | 0.33% | USD 5,773 |
| Mid | 43.69 → 42.57 | 2.55% | 168.9 → 175.8 | 1.96% | USD 515,567 |
| High | 72.17 → 65.71 | 8.96% | 76.7 → 85.5 | 7.74% | USD 33,917,039 |

Input: runner JSONL. Local SHA-256: dafe50f93262be5628b97a83b915a0f65aa4eaa52f09b1b9bfd36381c4b34a83
