# The E8 result on a second model and text, and one ball per group on a model's output

## Question

The second record rounded `gemma-4-E4B-it` and scored its output on one text. With one scale per group of weights set from the largest entry, the E8 Voronoi code gave a closer output than scalar rounding at equal bits. Issue #3 asks whether that holds on a second model and a second text.

Four records of 2026-11-17 then found, on weight error only, that most of E8's gain is the region of its code, and that a ball over a long block is worth more. Nothing of that had been scored on a model.

1. Does E8's win over the scalar rule hold, shrink or disappear on a second model and text?
2. Does one ball per group of weights give a closer output than the E8 Voronoi code, the scalar rule and llama.cpp's own files, at equal bits or fewer?
3. Does weight error say what the output does for these arms?

## Status

`complete`, 2026-11-18. All 32 variants are scored: the 11 of the first plan, the 14 arms of the second and the 7 of its extension. 15 of the 20 predictions hold. One model, one text. No arm has a packed format: bits are counted from the codebook and the rounded weights are scored from 16-bit float copies.

## Preregistration

Three plans, each committed and pushed before its variants were scored.

| Plan | Commit | Covers | Predictions that hold |
|---|---|---|---|
| [PREREG.md](PREREG.md) | `81b83d2` | issue #3 as written: 11 variants | 3 of 6 |
| [PREREG-arms.md](PREREG-arms.md) | `81b83d2` | 14 arms from the weight-error records | 6 of 7 |
| Extension 1 of the second | `7ba09a0`, with 14 variants seen | 7 arms: a codeword at zero, and bits split freely between the point and the scale | 6 of 7 |

The extension was written after the first results. It says which, and its predictions are not blind to them.

## Result in short

1. **As the gemma record ran it, E8's win shrinks with groups of 32 and reverses with one scale per row.** At 4.5 bits the E8 Voronoi code scores a mean KL of 0.0551 against 0.0620 for the scalar rule, a gain of 0.52 dB where gemma had 1.96. Per row it scores 3.658 against 0.632 at 16 levels and 1.041 against 0.129 at 32 levels, while its weight error is 3 dB **lower**.
2. **The cause is that the code has no codeword at zero.** The harness subtracts the mean of its codewords from every codeword, a vector 0.82 steps long. With that left out, the same code scores 0.209 and 0.050 per row and 0.0434 with groups of 32. E8 is then ahead of the scalar rule at all three bit points, by 1.55, 4.81 and 4.13 dB of KL. In weight error the change is within 0.06 dB in the mean.
3. **One ball per group is far ahead of everything else at equal bits.** At 4.5 bits E8 in one ball per group of 128 scores 0.0195 and keeps the first choice on 93.1 percent of positions. llama.cpp's `Q4_0` at the same bits scores 0.0623 and 88.2 percent. Its `Q4_K_M` at 4.81 bits scores 0.0333 and 91.1 percent.
4. **Integer levels with no lattice get nearly all of it.** In the same ball they score 0.0227 at 4.5 bits, 11 errors of the gap ahead of `Q4_K_M`. E8 is worth 0.66 dB of KL over them, which is its cell.
5. **At 4.0 bits one ball per group is under `Q4_0` at 4.5 bits and close to `Q4_K_M` at 4.81**: 0.0371.
6. **For the arms with one ball per group, weight error did say what the output does.** Over nine such arms, from 4.0 to 4.5 bits and with groups of 32 and 128, the gain in KL over the scalar rule is 0.91 to 1.08 times the gain in weight error. Their weight error is known before anything is rounded. For other arms it did not: the Voronoi code with and without a codeword at zero have the same weight error and a factor of 17 between their KL.
7. **A ball over 8 weights is not enough.** E8 in a ball over 8 weights with a scale per 32 scores 0.0462 at 4.5 bits, where one ball over the 32 scores 0.0339. The length of the block is where the gain is, as the weight-error record said.

## The model, the text and the scoring

**Model.** `Qwen2.5-1.5B-Instruct`, the publisher's 16-bit float GGUF, SHA-256 `fc89e330deb3fd8fa560f1c0f35a1e2b8da96d59e13445559ed190307a6f5649`. 28 layers, 196 block matrices, 1.31 billion block weights, rows of 1,536 or 8,960. Only the block matrices are rounded.

**Text.** The wikitext-2 test set, SHA-256 `173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`. 40 passages of 512 tokens, the second half of each scored: 10,200 positions.

**Score.** `llama-perplexity --kl-divergence` of llama.cpp at commit `eada89c`, built for a CPU, 4 threads, against the reference model's saved outputs. The reference perplexity is 9.17. The mean KL divergence, and the share of positions where the first choice stays the same. The standard errors are the tool's. They treat positions as independent and are too small, as the gemma record says.

**Bits.** log2 of the number of codewords per weight, and the bits of the scale. For llama.cpp's files, the bits of the file's block matrices. llama.cpp's files were made with only the block matrices changed and with no importance matrix.

**The arms.**

| Name | What |
|---|---|
| `X32:16`, `XR:16` | the scalar rule of `Q4_0`: one scale per 32 weights or per row, from the largest entry, 16 levels |
| `VE832:16`, `VE8R:16` | the E8 Voronoi code of the harness, as the gemma record ran it |
| `VE8Z...` | the same with a codeword at zero |
| `EB32@32:16`, `ZB32@32:16`, `EC32@32:16` | E8, or integer levels, in one ball (B) or cross-polytope (C) per group of 32, with as many points as 16 levels and a 16-bit scale |
| `EB128@128:16`, `ZB128@128:16` | the same per group of 128 |
| `EB128@R:16`, `ZB128@R:16`, `EC128@R:16`, `EB128@R:32` | a ball or a cross-polytope over 128 weights, one scale per row, with as many points as 16 or 32 levels |
| `EB8@32:16`, `PaB8@32:16`, `SEB8@32:16` | E8 in a region over 8 weights, one scale per 32: a ball, the ball of the 3/2-norm, and a ball at the closed-form scale of the record for issue #10, which never shrinks a block |
| `EBG128:b560s16` | E8 in one ball per group of 128, 560 bits for the point and 16 for the scale |
| `EBG32:b136s8`, `EBG128:b504s8` | the same with an 8-bit scale: the scale is rounded up to a grid of 16 steps per factor of two |
| `ZBG128:b560s16` | integer levels in place of E8 |

In a ball arm each weight is rounded to an integer, or each 8 weights to E8, and the scale of a group is the smallest at which the whole group rounds to a point of the codebook. Zero is a codeword of every ball arm.

## Results

Every number is in `results.out`, printed by `summary.py` from `results.jsonl`.

### Issue #3: the equal-bits table

| Bit point | Scalar rule: mean KL, first choice kept | E8 Voronoi code of the harness | Gap, in errors | Verdict by the plan's rule |
|---|---|---|---|---|
| 4.5 bits, groups of 32 | 0.0620, 88.19% | 0.0551, 89.41% | 2.8 for E8 | **shrinks**: 0.52 dB of KL, gemma 1.96 |
| 4.008 bits, per row | 0.6317, 63.59% | 3.6580, 25.75% | 92 against E8 | **disappears**, and reverses |
| 5.008 bits, per row | 0.1293, 82.94% | 1.0414, 63.60% | 42 against E8 | **disappears**, and reverses |

- `X32:16` scores as llama.cpp's own `Q4_0` file: 0.06204 against 0.06234, and 88.19 against 88.22 percent.
- `Q4_K_M` at 4.812 bits: 0.0333 and 91.11 percent. On gemma E8 at 4.5 bits nearly matched it. Here it has 1.65 times its KL.
- `Q3_K_M` at 3.824 bits: 0.1097 and 83.75 percent.

The same table with a codeword at zero:

| Bit point | Scalar rule | E8 Voronoi code with a codeword at zero | Gap, in errors | KL gain, dB | On gemma, with the centre | Verdict |
|---|---|---|---|---|---|---|
| 4.5 bits, groups of 32 | 0.0620 | 0.0434, 89.55% | 10.2 | 1.55 | 1.96 | holds |
| 4.008 bits, per row | 0.6317 | 0.2088, 77.90% | 50.5 | 4.81 | 3.20 | holds |
| 5.008 bits, per row | 0.1293 | 0.0500, 89.88% | 32.9 | 4.13 | 2.99 | holds |

So the answer to the issue has two parts. The code the gemma record ran does not carry to this model. E8 does.

### Why the code of the harness breaks: no codeword at zero

Three files hold this. None of it was planned.

**`rows.out`: single rows of 21 matrices, weight error.** The harness's code subtracts the mean of its codewords, [0.07, 0.33, 0.03, -0.57, 0.36, 0.14, -0.16, 0.26] steps, 0.82 long. No codeword is zero, and no rebuilt weight is zero: 0.000 of them against 0.19 to 0.65 under the scalar rule. In 4 of the 21 matrices a row has a weight 31 to 48 times the row's rms. There the scale is set by that weight, the scalar rule sends the small weights to zero, and the code puts noise in their place.

| Last layer's `ffn_down` | Matrix error, dB | Worst row, dB | Rows above -3 dB | Share of the error in 10 rows |
|---|---|---|---|---|
| scalar rule, `XR:16` | -9.99 | -1.4 | 18 | 0.221 |
| E8 Voronoi code of the harness, `VE8R:16` | -9.68 | +1.9 | 4 | 0.711 |
| with a codeword at zero, `VE8ZR:16` | -13.19 | -3.7 | 0 | 0.356 |

Row 1421 of that matrix, whose largest weight is 45 times its rms: -8.01 dB under the scalar rule, +1.90 under the code of the harness, -7.53 with a codeword at zero. Over the 21 matrices the code of the harness still has the lower pooled error: -17.80 dB against -15.40 for the scalar rule.

**The output.** With a codeword at zero and nothing else changed:

| | With the centre | With a codeword at zero | Share |
|---|---|---|---|
| per row, 16 levels | 3.6580 | 0.2088 | 0.057 |
| per row, 32 levels | 1.0414 | 0.0500 | 0.048 |
| groups of 32, 16 levels | 0.0551 | 0.0434 | 0.79 |

The extension predicted the first two and predicted that groups of 32 would change by less than 15 percent. They changed by 21.

**`centre.out`: the same change in weight error, on the 59 matrices of the earlier records.** The mean gain of a codeword at zero is between -0.06 and +0.06 dB in every family and cell. The lead of E8 cut by a ball over 8 weights is the same against either form of the code. So the weight-error records of this date stand as written, and the centre is invisible to them.

What this does and does not show. The output test shows that the centre is the cause of the reversal. The row tables show where its weight error goes. That those rows are what the output loses on is likely and not isolated: no arm repairs only them.

### One ball per group

Against the scalar rule at 4.5 bits, `X32:16`, whose weight error is -20.67 dB and mean KL 0.0620:

| Arm | Bits | Weight error, dB | Mean KL | First choice kept | KL gain, dB | Weight error gain, dB | Ratio |
|---|---|---|---|---|---|---|---|
| `EBG128:b560s16` | 4.500 | -25.60 | 0.0195 | 93.08% | 5.02 | 4.93 | 1.02 |
| `EBG32:b136s8` | 4.499 | -25.13 | 0.0208 | 92.65% | 4.74 | 4.46 | 1.06 |
| `ZBG128:b560s16` | 4.500 | -24.94 | 0.0227 | 92.75% | 4.36 | 4.27 | 1.02 |
| `EB32@32:16` | 4.498 | -23.55 | 0.0339 | 90.85% | 2.62 | 2.88 | 0.91 |
| `EC32@32:16` | 4.491 | -23.38 | 0.0341 | 90.94% | 2.59 | 2.71 | 0.96 |
| `ZB32@32:16` | 4.499 | -22.90 | 0.0367 | 90.53% | 2.28 | 2.23 | 1.02 |
| `EB128@128:16` | 4.124 | -23.34 | 0.0331 | 90.90% | 2.72 | 2.66 | 1.02 |
| `ZB128@128:16` | 4.125 | -22.68 | 0.0377 | 90.22% | 2.17 | 2.01 | 1.08 |
| `EBG128:b504s8` | 4.000 | -22.83 | 0.0371 | 90.45% | 2.24 | 2.16 | 1.04 |

Read down the last column. Whatever the bits, the group and the lattice, a decibel of weight error is a decibel of KL for these arms, within a tenth.

**Against llama.cpp's files.**

| Arm | Bits | Mean KL | Against | Bits | Mean KL | Lower by, in errors |
|---|---|---|---|---|---|---|
| `EBG128:b560s16` | 4.500 | 0.0195 | `Q4_K_M` | 4.812 | 0.0333 | 17.0 |
| `EBG32:b136s8` | 4.499 | 0.0208 | `Q4_K_M` | 4.812 | 0.0333 | 15.8 |
| `ZBG128:b560s16` | 4.500 | 0.0227 | `Q4_K_M` | 4.812 | 0.0333 | 11.4 |
| `EB32@32:16` | 4.498 | 0.0339 | `Q4_K_M` | 4.812 | 0.0333 | -0.6, a tie |
| `EB128@128:16` | 4.124 | 0.0331 | `Q4_K_M` | 4.812 | 0.0333 | 0.2, a tie |
| `EB128@128:16` | 4.124 | 0.0331 | `Q4_0` | 4.500 | 0.0623 | 16.6 |
| `EBG128:b504s8` | 4.000 | 0.0371 | `Q3_K_M` | 3.824 | 0.1097 | 33.6 |

**Where the gain at 4.5 bits comes from.**

| Step | From | To | KL gain |
|---|---|---|---|
| The scalar rule to integer levels in a ball, groups of 32 | `X32:16`, 0.0620 | `ZB32@32:16`, 0.0367 | 2.28 dB |
| Integer levels to E8, same ball | `ZB32@32:16`, 0.0367 | `EB32@32:16`, 0.0339 | 0.34 dB |
| 8 bits moved from the scale to the point, same group | `EB32@32:16`, 0.0339 | `EBG32:b136s8`, 0.0208 | 2.12 dB |
| A group of 128 with a 16-bit scale, same bits | `EBG32:b136s8`, 0.0208 | `EBG128:b560s16`, 0.0195 | 0.27 dB |
| E8 to integer levels, group of 128 | `EBG128:b560s16`, 0.0195 | `ZBG128:b560s16`, 0.0227 | -0.66 dB |

The two large steps are the ball and the bits. An arm with 16 levels and a 16-bit scale per 32 weights spends half a bit a weight on the scale. The lattice is worth a third to two thirds of a decibel.

### Regions over 8 weights with one scale per 32

The last three arms of the second plan. Each block of 8 weights is rounded to E8 inside a region, and four blocks share a scale.

| Arm | Bits | Weight error, dB | Mean KL | First choice kept | KL gain over `X32:16`, dB | Weight error gain, dB | Ratio |
|---|---|---|---|---|---|---|---|
| `EB8@32:16`, a ball | 4.494 | -22.49 | 0.0462 | 89.43% | 1.28 | 1.82 | 0.71 |
| `PaB8@32:16`, the ball of the 3/2-norm | 4.500 | -22.62 | 0.0424 | 89.51% | 1.65 | 1.95 | 0.85 |
| `SEB8@32:16`, a ball at the safe scale | 4.494 | -22.10 | 0.0426 | 89.75% | 1.63 | 1.43 | 1.14 |
| for scale: `EB32@32:16`, one ball over the 32 | 4.498 | -23.55 | 0.0339 | 90.85% | 2.62 | 2.88 | 0.91 |

- **All three are behind one ball over the 32 weights**, by 9.7 errors for the ball over 8. They are level with the Voronoi code that has a codeword at zero (0.0434) and ahead of the code of the harness (0.0551).
- **The safe scale did not cost output.** Its weight error is 0.39 dB worse than the fitted scale's and its KL is 2.7 errors lower. The prediction was the other way (A7) and it fails. A gap of 2.7 errors is a tie by this record's own limits, so what the rows say is that the safe scale is no worse.
- The 3/2-norm ball and the safe-scale ball are level with each other: 0.2 errors.

### The size of the group

| Group | Scalar rule | E8 Voronoi code of the harness | E8 in one ball per group | Integer levels in one ball per group |
|---|---|---|---|---|
| 32, at 4.5 bits | 0.0620 | 0.0551 | 0.0339 | 0.0367 |
| 128, at 4.125 bits | 0.1003 | 0.1014 | 0.0331 | 0.0377 |

For the scalar rule and the Voronoi code a larger group costs more than the bits it saves: their KL rises by more than half. For one ball per group it costs nothing here: the arm with groups of 128 has the same KL at 0.37 fewer bits. The gemma record found that finer groups help the output by more than their weight error says. That holds for rules that set the scale from the largest entry. It does not hold for a ball, whose error is the same share of every group.

**One scale per row.** With a ball over 128 weights and one scale per row, E8 scores 0.0577 and integer levels 0.0683 at 4.008 bits. The scalar rule per row scores 0.632 and the Voronoi code with a codeword at zero 0.209. These arms gain 1.5 times their weight-error gain in KL, where the arms with one ball per group gain 1.0 times.

Two more arms per row, scored last. **A cross-polytope in place of the ball**, `EC128@R:16`, scores 0.0487: 5.5 errors under the ball at the same bits, with a weight error 0.14 dB lower. With groups of 32 the two regions tied. **At 32 levels**, `EB128@R:32` scores 0.0163 at 5.008 bits, the lowest KL of the run, and keeps the first choice on 93.9 percent of positions. It is 20.3 errors under `Q4_K_M` at 4.81 bits. The Voronoi code of the harness at the same bits scores 1.041.

### Weight error against the output

Not planned. Spearman's rank correlation between mean weight error and mean KL:

| Variants | How many | Correlation |
|---|---|---|
| all that have both | 30 | 0.829 |
| at 4.49 to 4.50 bits | 12 | 0.937 |
| under 4.2 bits | 11 | 0.982 |

Within one bit point weight error ranks the arms nearly as the output does, with the code of the harness per row as the exception. Across group sizes it does not: `XR:32` and `X32:16` have the same weight error, -20.7 dB, and KL of 0.129 and 0.062.

### The predictions

**The first plan.**

| | Prediction | Result | Holds |
|---|---|---|---|
| P1 | `identity` scores as `unchanged` | mean KL 0.000000 and 100.00 percent for both | yes |
| P2 | `X32:16` scores as llama.cpp's `Q4_0` | KL within 0.47 percent, first choice within 0.03 points | yes |
| P3 | E8 has the lower KL in all three pairs | in 1 of 3 | **no** |
| P4 | Its KL gain is 0.5 to 1.5 times its weight-error gain | 0.30, -2.58 and -2.86 | **no** |
| P5 | The finer group has the lower KL for both rules | 0.0620 under 0.6317, and 0.0551 under 3.6580 | yes |
| P6 | `Q4_K_M` is under `Q4_0`, and `VE832:16` within 1.5 times `Q4_K_M` | 0.0333 under 0.0623. 1.65 times | **no** |

Of the 18 predictions of the gemma record, five have a counterpart here: B1 is P2, C1 is P3, C2 is P4, B3 is P5 and R5 is the first half of P6. Three of the five hold. The other 13 are about arms this run does not have.

**The second plan.**

| | Prediction | Result | Holds |
|---|---|---|---|
| A1 | `EB32@32:16` is under `VE832:16` by 2 errors | 0.0339 against 0.0551, 10.3 errors | yes |
| A2 | Its KL gain over `X32:16` is 0.5 to 1.5 times its weight-error gain | 0.91 | yes |
| A3 | `ZB32@32:16` is under `X32:16` by 2 errors | 0.0367 against 0.0620, 14.6 errors | yes |
| A4 | `EB128@128:16` at 4.124 bits is under `X32:16` at 4.5 | 0.0331 against 0.0620 | yes |
| A5 | `EB128@R:16` is under `VE8R:16` by 2 errors | 0.0577 against 3.6580 | yes |
| A6 | With groups of 128 the scalar rule and the Voronoi code have a higher KL than with 32 | 0.1003 over 0.0620, and 0.1014 over 0.0551 | yes |
| A7 | `SEB8@32:16` is above `EB8@32:16` | 0.0426 against 0.0462: lower, by 2.7 errors | **no** |

A5 is true and says little: `VE8R:16` is the broken arm. Against the code with a codeword at zero, `EB128@R:16` is still lower by 43.7 errors.

**Extension 1.**

| | Prediction | Result | Holds |
|---|---|---|---|
| B1 | With a codeword at zero the Voronoi code per row has under a third of its KL | shares of 0.057 and 0.048 | yes |
| B2 | And is under the scalar rule again, by 2 errors | by 50.5 and 32.9 errors | yes |
| B3 | With groups of 32 the centre changes the KL by under 15 percent | by 21.2 percent | **no** |
| B4 | `EBG128:b560s16` is under `EB32@32:16` by 2 errors | 0.0195 against 0.0339, 15.8 errors | yes |
| B5 | `EBG32:b136s8` is under `EB32@32:16` by 2 errors | 0.0208 against 0.0339, 14.6 errors | yes |
| B6 | `EBG128:b504s8` at 4.0 bits is under `VE832:16` and `Q4_0` at 4.5 | 0.0371 against 0.0551 and 0.0623 | yes |
| B7 | `ZBG128:b560s16` is under `ZB32@32:16` by 2 errors | 0.0227 against 0.0367, 14.9 errors | yes |

## Deviations from the plans

- **The run was started eight times.** A launch on this machine is cut at two hours, the machine restarted once, two starts were to change the order of what was left, and four were to let other records' runs go first. `run.py` skips variants already in `results.jsonl`. `results.start` lists every start with its commit. One variant, `EC32@32:16`, was cut during scoring and built again from the start.
- **The order of the second plan's arms was changed**, by extension 1, before any of the arms concerned was scored. The five arms that decide least were put last. They were scored on 2026-11-18, after the runs of the third model and of the importance-matrix record.
- **`run.py` was changed after the first data**, at `7ba09a0`, to add the arms of the extension, and three times more for the record imatrix-baseline-2026-11-17 (`79fb0cc`, `84b38d0`, `b148352`): that record's arms, and a results file that can hold two models. The code of the planned arms was not changed. `summary.py` was written after 3 variants and changed four times: for one table that no plan asked for, for the order of its tables, for the extension's predictions, and for two labels.
- **Three analyses were not planned**: `rows.py`, `centre.py`, and the rank table. Each is marked where it is used. `rows.py` was written with 13 variants seen and its output was read only after the extension's predictions were committed.
- **A second scoring process was started by mistake** while a variant was being scored, for another model. The machine killed it for lack of memory after five seconds. The variant being scored finished, and its row is `EBG32:b136s8`.
- **Other work shared the machine.** A weight-error study and the programs above ran at low priority beside this run. They change build times and nothing else.

## Limits

- **One model and one text.** No ball arm was scored on gemma, whose file is not on this machine. A third model is in the record model-output-smollm2-2026-11-17: result 3 holds there, and results 2, 4 and 6 do not carry as they stand.
- **Counted bits.** No arm of this repository has a packed file. `shape/ballindex.py` numbers the points of a ball-cut codebook one to one, so the bits can be stored. Reading a block back takes 6 to 11 look-ups a weight in numbers as long as the block's index, each waiting on the one before. Nothing here says how fast that can be made, or that a kernel could compute from it. As things stand the saving is in the file and not in memory.
- **No calibration on either side.** llama.cpp's files were made without an importance matrix, as in the gemma record. With one they score better: 0.0282 for `Q4_K_M` on this model, still behind the ball arms. That comparison is the record imatrix-baseline-2026-11-17.
- **The standard errors are too small.** They treat 10,200 positions of 40 passages as independent. Gaps of 2 or 3 errors should be read as ties. The findings above rest on gaps of 10 errors or more, except E8's win with groups of 32 as the harness has it, which is 2.8.
- **The reference is the 16-bit file**, and rounded weights are stored as 16-bit floats. In the gemma record the reference was an 8-bit file.
- **The backend differs from the gemma run**: CPU here, Metal there. The tool is the same. Model, text, reference and backend all changed, so a difference between the two records cannot be put down to one of them.
- **Published work.** Rounding a block to integer levels in a ball or a cross-polytope with a separate gain is old: pyramid vector quantisation, shell mapping, sphere shaping. For language models a pyramid quantiser is published (arXiv 2410.16926) and a Leech-lattice code with a ball over 24 weights (arXiv 2603.11021). No search was made for these arms against llama.cpp's files, so nothing here is claimed as new.

## Read against the closure rule

Nothing is closed by a certificate. These are measurements on one model.

**What changes in the repository's account.**

- The gemma record's finding, that E8 with one scale per group beats the scalar rule on the output, was obtained with a code that has no codeword at zero. On this model that code fails per row and the same code with a zero codeword does not. Every E8 arm of the gemma record carries the centre. Whether gemma's figures would be better without it is not known: the file is not here.
- The finding of the records of 2026-11-17, that the gain under this rule is in the region and the length of the block more than in the lattice, carries to the output of this model, one for one in decibels.
- "Weight error does not say what the output does" needs a qualifier. For one family of arms on this model it does, to a tenth of a decibel. For others it does not, by a factor of 17.

**What followed.**

1. The third model, `SmolLM2-1.7B-Instruct`: record model-output-smollm2-2026-11-17. One ball per group beats `Q4_K_M` there too. Integer levels are ahead of E8 there, and the harness's code is not broken per row.
2. Issue #14, llama.cpp's files with an importance matrix: record imatrix-baseline-2026-11-17.
3. The codeword at zero is proposed for the harness in issue #16, and not as its default: on the third model it is level per row and a little worse with groups of 32. The gemma arms have not been scored both ways.
4. A packed format for the integer arm: `shape/ballpack.py`, with a plan to score a model read back from the file in `experiments/packed-ball-format-2026-11-18`.

## How to rerun

Needs NumPy, torch, llama.cpp's `gguf` package and `llama-perplexity`, the model file and the text. The program refuses a model file with another SHA-256.

```bash
llama-perplexity -m qwen2.5-1.5b-instruct-fp16.gguf -f wiki.test.raw --chunks 40 -c 512 -t 4 --kl-divergence-base base.kld
```

```bash
python3 experiments/model-output-qwen25-2026-11-17/run.py --model qwen2.5-1.5b-instruct-fp16.gguf --kld base.kld --work DIR --out results.jsonl --bin llama-perplexity X32:16 VE832:16 EBG128:b560s16
```

```bash
python3 experiments/model-output-qwen25-2026-11-17/summary.py > results.out
```

A variant takes 5 minutes to score on 4 CPU cores and up to 12 to build. `rows.py` and `centre.py` say in their first lines how they are run.

## Disclosure

Written by an AI agent session under the owner's direction, on 2026-11-17 and 2026-11-18. The plans, the programs, the run and this record are the session's.
