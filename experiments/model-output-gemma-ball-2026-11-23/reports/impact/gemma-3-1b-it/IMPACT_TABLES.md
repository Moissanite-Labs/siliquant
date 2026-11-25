# Descriptive impact calculations: gemma-3-1b-it

Positive KL gain and top-1 Δ favor the candidate. Comparisons are conditional on the inputs; KL SE uses marginal quadrature and does not establish paired significance. Top-1 and PPL deltas are descriptive; this report does not apply an uncertainty screen to them.

## wikitext

Stock comparisons ranked by absolute relative KL effect. PPL columns are percent change from F16 (negative is lower PPL), candidate / baseline.

| Candidate → baseline | Actual block bpw (candidate / baseline) | KL ratio | KL gain | Gain / SE | ≥2 SE? | Top-1 Δ pp | PPL vs F16 (%) |
|---|---:|---:|---:|---:|---|---:|---:|
| hZBG128:b560s16+K10 → file:Q4_0 | 4.5129 / 4.5000 | 0.226 | +0.196042 | +33.92 | yes | +11.245 | +6.66 / +15.58 |
| hZBG128:b560s16 → file:Q4_0 | 4.5080 / 4.5000 | 0.228 | +0.195521 | +33.75 | yes | +11.186 | +6.65 / +15.58 |
| hZBG128:b560s16~rand → file:Q4_0 | 4.5080 / 4.5000 | 0.256 | +0.188603 | +31.20 | yes | +10.362 | +6.83 / +15.58 |
| hZBG128:b624s16 → file:Q4_K_M | 5.0081 / 5.4818 | 0.374 | +0.051887 | +18.35 | yes | +5.334 | +1.42 / +4.19 |
| HBG128:b560s16 → file:Q4_0 | 4.4999 / 4.5000 | 0.380 | +0.157019 | +26.09 | yes | +7.598 | -3.50 / +15.58 |
| EBG128:b560s16 → file:Q4_0 | 4.4999 / 4.5000 | 0.391 | +0.154393 | +25.01 | yes | +7.019 | +8.31 / +15.58 |
| ZBG128:b560s16+K10 → file:Q4_0 | 4.5047 / 4.5000 | 0.407 | +0.150167 | +24.34 | yes | +7.215 | +1.01 / +15.58 |
| ZBG128:b560s16 → file:Q4_0 | 4.4998 / 4.5000 | 0.413 | +0.148622 | +23.97 | yes | +7.225 | +1.62 / +15.58 |
| hZBG128:b624s16 → file:Q4_K_M-imat | 5.0081 / 5.4818 | 0.620 | +0.018981 | +8.50 | yes | +2.432 | +1.42 / +3.33 |
| ZBG128:b624s16 → file:Q4_K_M | 5.0000 / 5.4818 | 0.632 | +0.030467 | +10.39 | yes | +2.775 | +4.36 / +4.19 |
| ZB128@R:32 → file:Q4_K_M-imat | 5.0104 / 5.4818 | 1.329 | -0.016420 | -5.91 | yes | -1.392 | +11.01 / +3.33 |
| VE8Z32:16 → file:Q4_0 | 4.5000 / 4.5000 | 0.750 | +0.063232 | +8.79 | yes | +2.284 | +19.54 / +15.58 |
| ZB128@R:32 → file:Q4_K_M | 5.0104 / 5.4818 | 0.801 | +0.016486 | +5.03 | yes | +1.510 | +11.01 / +4.19 |
| ZBG128:b624s16 → file:Q4_K_M-imat | 5.0000 / 5.4818 | 1.049 | -0.002439 | -1.03 | no | -0.127 | +4.36 / +3.33 |

### Matched KL (interpolation, not measured iso-quality)

| Family / target | Matched block bpw | Saved bpw | Block storage saved | Basis |
|---|---:|---:|---:|---|
| Plain / file:Q4_0 | 3.9999 | ≥0.5001 | ≥11.11% | bound |
| Plain / file:Q4_K_M | 4.7088 | 0.7730 | 14.10% | interpolated |
| Plain / file:Q4_K_M-imat | — | — | —% | not reached |
| Real matrix / file:Q4_0 | 4.0081 | ≥0.4919 | ≥10.93% | bound |
| Real matrix / file:Q4_K_M | 4.2684 | 1.2134 | 22.14% | interpolated |
| Real matrix / file:Q4_K_M-imat | 4.6547 | 0.8271 | 15.09% | interpolated |

### Calibration contrasts

| Candidate → comparator | Δ block bpw | KL gain | Top-1 Δ pp | PPL Δ |
|---|---:|---:|---:|---:|
| hZBG128:b560s16 → hZBG128:b560s16~rand | +0.0000 | +0.006918 | +0.824 | -0.049 |
| hZBG128:b560s16 → HBG128:b560s16 | +0.0081 | +0.038502 | +3.588 | +2.758 |
| hZBG128:b560s16 → VE8Z32:16 | +0.0080 | +0.132289 | +8.902 | -3.502 |
| hZBG128:b560s16~rand → HBG128:b560s16 | +0.0081 | +0.031584 | +2.764 | +2.807 |
| hZBG128:b560s16~rand → VE8Z32:16 | +0.0080 | +0.125371 | +8.078 | -3.453 |

## python

Stock comparisons ranked by absolute relative KL effect. PPL columns are percent change from F16 (negative is lower PPL), candidate / baseline.

| Candidate → baseline | Actual block bpw (candidate / baseline) | KL ratio | KL gain | Gain / SE | ≥2 SE? | Top-1 Δ pp | PPL vs F16 (%) |
|---|---:|---:|---:|---:|---|---:|---:|
| hZBG128:b560s16 → file:Q4_0 | 4.5080 / 4.5000 | 0.254 | +0.177830 | +11.59 | yes | +8.459 | +2.70 / +13.88 |
| hZBG128:b560s16~rand → file:Q4_0 | 4.5080 / 4.5000 | 0.254 | +0.177720 | +11.49 | yes | +7.563 | +4.03 / +13.88 |
| hZBG128:b560s16+K10 → file:Q4_0 | 4.5129 / 4.5000 | 0.255 | +0.177648 | +11.59 | yes | +8.403 | +2.41 / +13.88 |
| EBG128:b560s16 → file:Q4_0 | 4.4999 / 4.5000 | 0.354 | +0.153893 | +9.88 | yes | +7.171 | +2.03 / +13.88 |
| HBG128:b560s16 → file:Q4_0 | 4.4999 / 4.5000 | 0.394 | +0.144430 | +8.77 | yes | +5.938 | -1.93 / +13.88 |
| ZBG128:b560s16+K10 → file:Q4_0 | 4.5047 / 4.5000 | 0.412 | +0.140123 | +8.33 | yes | +7.227 | +3.21 / +13.88 |
| ZBG128:b560s16 → file:Q4_0 | 4.4998 / 4.5000 | 0.419 | +0.138442 | +8.24 | yes | +6.947 | +4.17 / +13.88 |
| hZBG128:b624s16 → file:Q4_K_M | 5.0081 / 5.4818 | 0.469 | +0.035011 | +6.05 | yes | +1.961 | +2.97 / -0.11 |
| hZBG128:b624s16 → file:Q4_K_M-imat | 5.0081 / 5.4818 | 0.601 | +0.020533 | +3.76 | yes | +1.289 | +2.97 / +1.84 |
| ZB128@R:32 → file:Q4_K_M-imat | 5.0104 / 5.4818 | 1.318 | -0.016375 | -2.13 | yes | -0.392 | +8.20 / +1.84 |
| ZBG128:b624s16 → file:Q4_K_M | 5.0000 / 5.4818 | 0.740 | +0.017120 | +2.70 | yes | +1.289 | +3.49 / -0.11 |
| VE8Z32:16 → file:Q4_0 | 4.5000 / 4.5000 | 0.742 | +0.061477 | +3.10 | yes | +2.857 | +8.71 / +13.88 |
| ZBG128:b624s16 → file:Q4_K_M-imat | 5.0000 / 5.4818 | 0.949 | +0.002642 | +0.44 | no | +0.617 | +3.49 / +1.84 |
| ZB128@R:32 → file:Q4_K_M | 5.0104 / 5.4818 | 1.029 | -0.001897 | -0.24 | no | +0.280 | +8.20 / -0.11 |

### Matched KL (interpolation, not measured iso-quality)

| Family / target | Matched block bpw | Saved bpw | Block storage saved | Basis |
|---|---:|---:|---:|---|
| Plain / file:Q4_0 | 3.9999 | ≥0.5001 | ≥11.11% | bound |
| Plain / file:Q4_K_M | 4.8323 | 0.6495 | 11.85% | interpolated |
| Plain / file:Q4_K_M-imat | 4.9741 | 0.5077 | 9.26% | interpolated |
| Real matrix / file:Q4_0 | 4.0081 | ≥0.4919 | ≥10.93% | bound |
| Real matrix / file:Q4_K_M | 4.4563 | 1.0255 | 18.71% | interpolated |
| Real matrix / file:Q4_K_M-imat | 4.6610 | 0.8208 | 14.97% | interpolated |

### Calibration contrasts

| Candidate → comparator | Δ block bpw | KL gain | Top-1 Δ pp | PPL Δ |
|---|---:|---:|---:|---:|
| hZBG128:b560s16 → hZBG128:b560s16~rand | +0.0000 | +0.000110 | +0.896 | -0.086 |
| hZBG128:b560s16 → HBG128:b560s16 | +0.0081 | +0.033400 | +2.521 | +0.299 |
| hZBG128:b560s16 → VE8Z32:16 | +0.0080 | +0.116353 | +5.602 | -0.389 |
| hZBG128:b560s16~rand → HBG128:b560s16 | +0.0081 | +0.033290 | +1.625 | +0.385 |
| hZBG128:b560s16~rand → VE8Z32:16 | +0.0080 | +0.116243 | +4.706 | -0.303 |

## Illustrative deployment scenarios

Anchor: 0.8208 saved block bpw at the Q4_K_M-imat KL target (smaller of two text estimates). These are assumptions, not a fleet forecast.

| Scenario | Resident GB (base → new) | Footprint reduction | H200 token/s roofline (base → new) | Fleet equivalent reduction | Annual gross opportunity |
|---|---:|---:|---:|---:|---:|
| Low | 27.56 → 27.43 | 0.46% | 529.3 → 536.7 | 0.35% | USD 6,070 |
| Mid | 43.83 → 42.65 | 2.69% | 167.9 → 175.1 | 2.06% | USD 541,103 |
| High | 72.57 → 65.74 | 9.40% | 76.1 → 85.3 | 8.11% | USD 35,532,162 |

Input: runner JSONL. Local SHA-256: 040717b2d9c24d47e64fcc3b41156a375d651f180cd7b9f558ec603c60f7733f
