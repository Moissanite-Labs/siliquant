# Plan: the arms of the later records on a model's output

Written 2026-11-17, before any variant is scored, and committed with [PREREG.md](PREREG.md). That plan covers issue #3 as written. This one covers the arms that the records of this date found on weight error, and that the two check notes (`notes/pr6-proposal-check.md`, `notes/issue10-proposal-check.md`) ask to have scored. Same model, same text, same scoring, same program.

## Question

Four records of this date found, on weight error only:

- most of E8's gain under the rule that clips nothing is the region of its code;
- a ball over a long block is worth 1.2 to 3.7 dB more than the E8 Voronoi code on every real matrix tried, at no more bits;
- integer levels in such a ball beat the E8 Voronoi code with no lattice;
- a region between a ball and a cross-polytope is best over 8 weights, groups of 128 are the best size at 4.5 bits, and a scale with a proved guarantee costs half a decibel.

The second record found that weight error does not say what a model's output does, and that finer groups helped the output by more than their weight error did.

1. Does a ball over a long block give a closer output than the E8 Voronoi code at equal bits?
2. Do integer levels in a ball give a closer output than the scalar rule of `Q4_0` at equal bits?
3. Does the size of a group follow weight error on an output, or not?
4. Do the smaller gains found over 8 weights show on an output at all?

## Variants

All at 16 levels unless marked. Weight errors are means over the 21 matrices of this model that the record ball-dimension-no-clip-rule-2026-11-17 ran, in dB, where that record has the arm.

| Variant | What | Bits | Weight error there |
|---|---|---|---|
| `EB32@32:16` | E8 in one ball per group of 32 | 4.499 | -23.54 |
| `ZB32@32:16` | integer levels in one ball per group of 32 | 4.499 | -22.88 |
| `EC32@32:16` | E8 in one cross-polytope per group of 32 | 4.491 | -23.43 |
| `EB128@128:16` | E8 in one ball per group of 128 | 4.124 | -23.31 |
| `ZB128@128:16` | integer levels in one ball per group of 128 | 4.125 | -22.66 |
| `X128:16` | the scalar rule with groups of 128 | 4.125 | -18.61 |
| `VE8128:16` | the E8 Voronoi code with groups of 128 | 4.125 | -20.87 |
| `EB128@R:16` | E8 in a ball over 128 weights, one scale per row | 4.007 | -21.63 |
| `ZB128@R:16` | integer levels in the same | 4.008 | -20.97 |
| `EC128@R:16` | E8 in a cross-polytope over 128 weights, one scale per row | 4.007 | -21.87 |
| `EB128@R:32` | E8 in a ball over 128 weights, one scale per row, 32 levels | 5.008 | -27.67 |
| `EB8@32:16` | E8 in a ball over 8 weights, groups of 32 | 4.494 | -22.43 |
| `PaB8@32:16` | E8 in a p-norm ball with p = 3/2 over 8 weights, groups of 32 | 4.500 | not run there |
| `SEB8@32:16` | E8 in a ball over 8 weights at the safe scale of the record for issue #10, groups of 32 | 4.494 | not run there |

For comparison, from the first plan: `X32:16` at 4.5 bits, -20.57 dB. `VE832:16` at 4.5 bits, -22.37 dB. `XR:16` and `VE8R:16` at 4.008 bits, -14.80 and -17.89 dB.

They are scored in the order of the table, after the variants of the first plan. If the run is cut short, the table says which were not reached.

## What is known before the run

- Everything in the first plan.
- The weight errors above, and all of the four records.
- **The reason to doubt.** In the gemma record the scalar rule with groups of 32 at 16 levels and with one scale per row at 32 levels scored the same mean KL, 0.117 and 0.118, with weight errors 1.3 dB apart. So a larger group needed 1.3 dB more weight error to tie. The arms with groups of 128 and with a row are the ones this bears on.
- **One ball to a group makes the error of every group the same share of its size.** The scalar rule makes it a share of the square of the group's largest entry. Which of the two an output prefers is not known.
- No variant of this model has been scored. The first plan lists what was seen.

## Predictions

| | Prediction | It fails if |
|---|---|---|
| A1 | `EB32@32:16` has a lower mean KL than `VE832:16`, by at least 2 errors of the gap | it is not lower by that much |
| A2 | The KL gain of `EB32@32:16` over `X32:16` in dB is 0.5 to 1.5 times its gain in mean weight error | it is outside |
| A3 | `ZB32@32:16` has a lower mean KL than `X32:16`, by at least 2 errors | it is not |
| A4 | `EB128@128:16` at 4.124 bits has a lower mean KL than `X32:16` at 4.5 bits | it is not lower |
| A5 | `EB128@R:16` has a lower mean KL than `VE8R:16`, by at least 2 errors | it is not |
| A6 | With groups of 128 both the scalar rule and the E8 Voronoi code have a higher KL than with groups of 32: `X128:16` over `X32:16`, `VE8128:16` over `VE832:16` | either is lower |
| A7 | `SEB8@32:16` has a higher mean KL than `EB8@32:16` | it is lower |

The error of a gap is the root of the sum of the squares of the two standard errors the tool prints, as in the first plan.

Reported without a prediction: `EB128@128:16` against `VE832:16`, which have 0.375 bits between them and where weight error and the size of the group pull apart. `EB8@32:16` and `PaB8@32:16` against `VE832:16`, where the weight errors differ by under half a decibel. The cross-polytope arms against the ball arms. `EB128@R:32` against `VE8R:32`. And for every arm the ratio of its KL gain to its weight-error gain.

## What it would mean

- If A1, A3 and A5 hold, the region found on weight error carries to an output on this model, and the arms are worth a packed format.
- If they fail while the first plan's P3 holds, then a gain in weight error from the lattice carries to the output and a gain from the region does not. That would be a finding about what the output follows, and the next step would be to look at where the error of a ball falls within a group.
- A4 and the unpredicted comparison beside it say whether 0.375 bits can be saved.

## What this run cannot show

- Anything about a second model for these arms. The gemma file is not on this machine.
- The cost of numbering the points of these codebooks. A block of 128 weights at 4 bits has up to 2^512 points.
- Anything below 4 bits.
- The order of near ties. The standard errors treat positions as independent and are understated.

## Extension 1: a codeword at zero, and bits taken from the scale

Written 2026-11-17 with 14 of the 25 variants scored, and before any arm below is scored. It adds seven arms and changes the order of the eleven that are left. `run.py` gains the code for the new arms at the commit that adds this section. The code of the arms already planned is not changed.

### What was seen before this extension

The rows of `results.jsonl` at that commit: all eleven variants of the first plan, and `EB32@32:16`, `ZB32@32:16` and `EC32@32:16` of this one.

- **With groups of 32 the E8 Voronoi code is ahead of the scalar rule by less than on gemma**: mean KL 0.0551 against 0.0620.
- **With one scale per row it is far behind**: 3.658 against 0.632 at 16 levels and 1.041 against 0.129 at 32 levels. Its weight error is 3 dB lower than the scalar rule's on every kind of matrix.
- **One ball per group of 32 is well ahead of both**: 0.0339 for E8 and 0.0367 for integer levels, at 4.5 bits. llama.cpp's `Q4_K_M` scores 0.0333 at 4.81 bits.

**A look at single rows, not planned** (`rows.py`, committed with this section; its output `rows.out` is committed when it finishes). The harness's E8 Voronoi code subtracts the mean of its codewords from every codeword. That vector is 0.82 steps long. So no codeword is zero, and a block of small weights is rebuilt as a point 0.82 steps from zero or further. In a row where one weight is tens of times the rest, the scale is set by that weight, the scalar rule sends the rest to zero, and the Voronoi code puts noise in their place that is larger than they are. Seen before this section was written, on five rows of the last layer's `ffn_down`: row 1421 has a relative error of +1.9 dB under the Voronoi code and -8.0 dB under the scalar rule. `rows.out` will have the same for 21 matrices, and for the Voronoi code with the centre left out. It was running when this section was committed: its numbers were not seen when the predictions below were written.

**Arithmetic, not data.** The error of a code with one ball per group does not depend on the weights: it is G (n + 2) over the bound of the codebook. So the weight error of every new ball arm is known before it is run. A smoke test of the new code on eight matrices of `Qwen2.5-0.5B` gave these to 0.05 dB for the arms with a 16-bit scale. With a scale rounded up to a grid the error came out 0.05 to 0.5 dB lower than the formula, because the block then lies inside the ball and not on its surface.

### The new arms

| Variant | What | Bits | Weight error, dB |
|---|---|---|---|
| `VE8ZR:16` | the E8 Voronoi code with a codeword at zero, one scale per row, 16 levels | 4.008 | as `VE8R:16` in the mean |
| `VE8ZR:32` | the same, 32 levels | 5.008 | as `VE8R:32` in the mean |
| `VE8Z32:16` | the same, groups of 32, 16 levels | 4.5 | as `VE832:16` |
| `EBG128:b560s16` | E8 in one ball per group of 128: 560 bits for the point, 16 for the scale | 4.500 | -25.6 |
| `EBG32:b136s8` | E8 in one ball per group of 32: 136 bits for the point, 8 for the scale | 4.500 | -25.1 |
| `EBG128:b504s8` | E8 in one ball per group of 128: 504 bits for the point, 8 for the scale | 4.000 | -22.8 |
| `ZBG128:b560s16` | integer levels in one ball per group of 128, 560 and 16 bits | 4.500 | -24.9 |

**A codeword at zero.** `VE8Z` is `tq.maxscale_voronoi` with the centre of the code set to zero. Nothing else differs from `VE8`.

**An 8-bit scale.** The scale of a group is rounded up to the largest scale of its matrix divided by a whole power of the 16th root of two. That is 256 steps over 16 factors of two. A scale under the lowest step takes the lowest step. The matrix's largest scale is stored once at 16 bits, and those bits are counted.

**Bits for the point.** The codebook is every point of E8 repeated, or of the integers, whose squared length is at most a bound: the largest bound with no more than 2^560, 2^504 or 2^136 points.

### The order

The plan said the arms are scored in the order of its table. That is changed here, before any of the arms concerned is scored, so that the arms that decide the most come first if the run is cut short. `EB128@128:16` was being built when this was written and stays next. Then:

`VE8ZR:16`, `EBG128:b560s16`, `EBG32:b136s8`, `X128:16`, `VE8128:16`, `ZB128@128:16`, `EB128@R:16`, `ZB128@R:16`, `VE8ZR:32`, `VE8Z32:16`, `EBG128:b504s8`, `ZBG128:b560s16`, `EC128@R:16`, `EB128@R:32`, `EB8@32:16`, `PaB8@32:16`, `SEB8@32:16`.

### Predictions

| | Prediction | It fails if |
|---|---|---|
| B1 | With a codeword at zero the E8 Voronoi code per row is no longer broken: `VE8ZR:16` has less than a third of the mean KL of `VE8R:16`, and `VE8ZR:32` less than a third of `VE8R:32` | either is not so |
| B2 | And E8's lead over the scalar rule returns: `VE8ZR:16` has a lower mean KL than `XR:16`, and `VE8ZR:32` than `XR:32`, each by at least 2 errors of the gap | either is not lower by that much |
| B3 | With groups of 32 the centre does little: the mean KL of `VE8Z32:16` is within 15 percent of `VE832:16` | it is outside |
| B4 | More bits for the point carry to the output: `EBG128:b560s16` has a lower mean KL than `EB32@32:16`, by at least 2 errors | it is not |
| B5 | So do bits taken from the scale: `EBG32:b136s8` has a lower mean KL than `EB32@32:16`, by at least 2 errors | it is not |
| B6 | At 4.0 bits `EBG128:b504s8` has a lower mean KL than the E8 Voronoi code of the gemma record at 4.5 bits, `VE832:16`, and than llama.cpp's `Q4_0` at 4.5 bits | it is not lower than both |
| B7 | `ZBG128:b560s16` has a lower mean KL than `ZB32@32:16`, by at least 2 errors | it is not |

Reported without a prediction: the two best arms at 4.5 bits against llama.cpp's `Q4_K_M` at 4.81 bits. `EBG128:b560s16` against `EBG32:b136s8`, which have the same bits and differ in the size of the group. For each new ball arm the ratio of its KL gain over `X32:16` to its gain in weight error.

### What it would mean

- If B1 and B2 hold, the verdict on issue #3 has two parts: the E8 Voronoi code as the gemma record ran it fails on this model per row, and the cause is the centre of that code, not E8. The gemma record's E8 arms all carry that centre.
- If B1 fails, the rows are not the cause, and the reversal per row is not explained.
- If B4 and B5 hold, the best code found so far at 4.5 bits is not the one with 16 levels and a 16-bit scale, and the arithmetic of the bound says where to look next.
