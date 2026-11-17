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
