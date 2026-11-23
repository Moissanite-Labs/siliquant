# Gemma 3 270M — Runner output transcription

Source: Message from the runner operator, 2026-11-23. The sender reported 34 rows, 17 variants
on both texts, zero errors, and F16 reference PPL 50.594102 on wikitext and
8.38385 on Python. The raw runner files were not available on the remote
branches inspected. The receipt strings below are claims, not verified hashes
of retrieved data. This table carries rounded values and is not runner JSONL.

| Claimed SHA-256 artifact | Receipt |
| --- | --- |
| results.jsonl | `dafe50f93262be5628b97a83b915a0f65aa4eaa52f09b1b9bfd36381c4b34a83` |
| results.out | `4c31bb3b85082db734887b4899bc3e5781c4ec835a20f89ba15cc919c22520b5` |
| gemma-3-270m-it-F16.gguf | `140cb395a3a3dcdb3de66e44c60d7265286c9d49f951c1899ec0d8c1f16b2feb` |
| wikitext | `173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08` |

Columns: variant | text | bpw | mean KL | KL SE | PPL | top-1 kept %

```text
file:Q4_0            | wikitext | 4.5000 | 0.501528 | 0.007732 | 69.867 | 70.137
file:Q4_0            | python   | 4.5000 | 0.565080 | 0.029793 | 11.942 | 77.647
file:Q4_K_M          | wikitext | 5.4338 | 0.153572 | 0.002861 | 54.219 | 83.049
file:Q4_K_M          | python   | 5.4338 | 0.166596 | 0.010586 |  9.031 | 87.899
file:Q4_K_M-imat     | wikitext | 5.4338 | 0.091622 | 0.001843 | 53.045 | 86.755
file:Q4_K_M-imat     | python   | 5.4338 | 0.088345 | 0.005311 |  8.558 | 91.765
ZBG128:b560s16       | wikitext | 4.4998 | 0.194755 | 0.003254 | 55.640 | 80.088
ZBG128:b560s16       | python   | 4.4998 | 0.209988 | 0.013437 |  8.666 | 85.658
hZBG128:b560s16      | wikitext | 4.5178 | 0.076017 | 0.001915 | 55.469 | 87.922
hZBG128:b560s16      | python   | 4.5178 | 0.100806 | 0.007263 |  8.744 | 90.868
hZBG128:b560s16~rand | wikitext | 4.5178 | 0.086958 | 0.001762 | 57.198 | 87.480
hZBG128:b560s16~rand | python   | 4.5178 | 0.104525 | 0.007112 |  9.503 | 90.532
ZBG128:b560s16+K10   | wikitext | 4.5130 | 0.192653 | 0.003369 | 57.555 | 80.382
ZBG128:b560s16+K10   | python   | 4.5130 | 0.212769 | 0.014498 |  8.758 | 85.546
hZBG128:b560s16+K10  | wikitext | 4.5310 | 0.074192 | 0.001850 | 55.729 | 88.010
hZBG128:b560s16+K10  | python   | 4.5310 | 0.097386 | 0.006898 |  8.804 | 91.148
EBG128:b560s16       | wikitext | 4.4999 | 0.178904 | 0.003259 | 65.187 | 81.608
EBG128:b560s16       | python   | 4.4999 | 0.186328 | 0.011397 |  9.809 | 88.347
VE8Z32:16            | wikitext | 4.5000 | 0.394135 | 0.006414 | 72.186 | 73.971
VE8Z32:16            | python   | 4.5000 | 0.377994 | 0.019847 | 11.111 | 82.297
HBG128:b560s16       | wikitext | 4.4999 | 0.174683 | 0.003145 | 55.967 | 81.588
HBG128:b560s16       | python   | 4.4999 | 0.167353 | 0.009242 |  9.126 | 88.011
ZBG128:b496s16       | wikitext | 3.9999 | 0.396725 | 0.005982 | 67.307 | 73.225
ZBG128:b496s16       | python   | 3.9999 | 0.399249 | 0.020692 |  9.859 | 81.457
hZBG128:b496s16      | wikitext | 4.0179 | 0.150933 | 0.003711 | 54.995 | 83.029
hZBG128:b496s16      | python   | 4.0179 | 0.182747 | 0.011705 |  9.677 | 88.683
ZBG128:b624s16       | wikitext | 5.0000 | 0.097914 | 0.001958 | 57.825 | 86.039
ZBG128:b624s16       | python   | 5.0000 | 0.100498 | 0.006400 |  9.522 | 90.700
hZBG128:b624s16      | wikitext | 5.0180 | 0.036582 | 0.000880 | 54.626 | 91.588
hZBG128:b624s16      | python   | 5.0180 | 0.055786 | 0.004639 |  8.236 | 92.885
ZB128@R:16           | wikitext | 4.0198 | 0.457003 | 0.007106 | 82.997 | 72.402
ZB128@R:16           | python   | 4.0198 | 0.472821 | 0.023774 | 12.437 | 80.840
ZB128@R:32           | wikitext | 5.0198 | 0.105903 | 0.002106 | 58.529 | 86.147
ZB128@R:32           | python   | 5.0198 | 0.112005 | 0.007160 |  9.213 | 89.860
```
