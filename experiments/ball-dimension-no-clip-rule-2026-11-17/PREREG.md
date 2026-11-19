# Plan: a ball over more than 8 weights, under the rule that clips nothing

Written 2026-11-17, before the run. The run is `run.py` in this folder at the commit that adds this file.

## Question

Two records of this date found that under the rule that clips nothing most of E8's gain over scalar rounding is the region of the code, that integer levels cut by a ball get most of it with no lattice, and that a ball over 16 or 24 weights would be worth more than one over 8. The last was arithmetic on the data and no codebook was run. `bounds/bounds.out`, section 10, has the same figures for Gaussian weights from integrals.

A ball over more weights needs no new lattice and no new decoder for rounding. Each weight is still rounded to an integer, or each 8 weights to E8. Only the test of whether a block is in the codebook looks at all n weights.

1. Does the gain grow with the number of weights under one ball, as the formula says, on Gaussian and on real weights?
2. With one ball for a whole group of 32 or 128 weights, which is the classical split into a gain and a shape, does it beat the E8 Voronoi code of the second record at no more bits?
3. Does the cross-polytope keep its lead on heavy tails as the blocks grow?

This follows the lead of PR #6 to the end the limits table points at: issue #4 asked whether E8 can be beaten, and the answer so far is by the same lattice with another region. Issue #3 will score a model's output. This run says which arms are worth scoring.

## The codebooks

A block has n = 8, 16, 32, 64 or 128 weights. A codebook is every point of Z^n, or of E8 repeated n/8 times, whose squared length (ball) or sum of sizes (cross-polytope) is at most a bound. The bound is the largest at which the codebook has no more than q^n points.

The sizes are counted exactly. The number of points of a given size over n weights is a coefficient of a power of a polynomial, and the program multiplies the polynomials in whole numbers. It checks that over 8 weights it gets the codebooks of the record region-shapes-no-clip-rule-2026-11-17, and that over 16 weights the count equals a sum over pairs of blocks of 8.

| Arm | q = 16: bound at n = 8, 16, 32, 64, 128 | Bits per weight at those n |
|---|---|---|
| ZB, integers in a ball | 179, 306, 553, 1041, 2010 | 3.9963, 3.9994, 3.9991, 3.9995, 4.0000 |
| EB, E8 in a ball | 178, 304, 552, 1040, 2008 | 3.9942, 3.9959, 3.9985, 3.9991, 3.9994 |
| ZC, integers in a cross-polytope | 29, 53, 101, 195, 382 | 3.9774, 3.9854, 3.9995, 3.9984, 3.9973 |
| EC, E8 in a cross-polytope | 29, 53, 101, 196, 385 | 3.9588, 3.9745, 3.9907, 3.9967, 3.9990 |

At q = 32 the bits are 4.977 to 5.000. No arm has more bits than q levels per weight. These codebooks have no packed format: nothing here numbers their points.

## The rule

As in the second record. The scale of a group is the smallest at which every block of the group rounds to a point of the codebook. It is found per block by bisection, 14 steps, and the group takes the largest. A block that then lies outside the codebook at its group's scale is shrunk toward zero, and the share of such blocks is reported.

The references are the scalar rule X and the E8 Voronoi code VE8, run with the harness functions of the second record in the same cells.

## What is known before the run

**The formula for Gaussian weights.** With fine cells the gain of a region over the scalar rule is 10 log10 of the cube's region part over the region's own (`bounds/bounds.out`, section 10). Computed for this plan on 4 million fresh Gaussian draws, from a seed no study matrix uses:

| Group | n | Ball | Cross-polytope |
|---|---|---|---|
| 32 | 8 | +1.15 | +0.70 |
| 32 | 16 | +1.56 | +1.13 |
| 32 | 32 | +1.89 | +1.49 |
| 128 | 8 | +1.43 | +0.89 |
| 128 | 32 | +2.41 | +1.94 |
| 128 | 128 | +3.01 | +2.59 |
| 2048 | 8 | +1.90 | +1.24 |
| 2048 | 16 | +2.62 | +1.99 |
| 2048 | 32 | +3.25 | +2.70 |
| 2048 | 64 | +3.78 | +3.26 |
| 2048 | 128 | +4.17 | +3.69 |

In dB. Two more terms are known.

- **The cell.** E8 is worth 0.65 dB over the integers.
- **The block on the wall.** One block of each group lies on the surface of the region, where its error is (n + 2)/n times the mean. With g/n blocks to a group that is a factor of 1 + 2/g on the group, whatever n is: 0.26 dB with groups of 32, 0.07 with groups of 128, nothing per row. The scalar rule does not pay it. The E8 Voronoi code does.

So for Gaussian weights the formula puts E8 in one ball per group of 32 at 0.65 + 1.89 - 0.26 = 2.28 dB over the scalar rule, where the E8 Voronoi code measured 1.36. Per row with a ball over 128 weights it gives 0.65 + 4.17 = 4.82 dB, where the E8 Voronoi code measured 2.29.

**On the real matrices** only blocks of 8, 16 and 24 per row have been looked at, by arithmetic and with no rounding: +2.63, +3.66 and +4.23 dB for a ball on the 32 whisper matrices (the record e8-gain-largest-entry-rule-2026-11-17, table 7).

**The formula was high before.** In the region study it was a fair guide to the sign and high by up to 0.3 dB, because a small codebook holds fewer points than its volume. Here the codebooks fill 0.80 to 1.00 of q^n, and more as n grows.

**Published work.** Cutting integer levels by a ball or a cross-polytope over a long block is old in coding: pyramid vector quantisation, and shell mapping and sphere shaping in transmission. For weights of language models a pyramid quantiser is published (arXiv 2410.16926) and so is a code on the Leech lattice with a ball over 24 weights (arXiv 2603.11021). No search was made for this rule with these blocks.

**Seen before writing this plan.** `run.py --smoke` on a 16 by 256 matrix of uniform numbers that is not in the study. The table of codebooks above. The formula table above, on draws that are not study matrices. All of the earlier records. No arm of this run has been run on a matrix of the study, and the Qwen matrices have not been opened except to list their shapes.

## Data and cells

- **G, T.** The six synthetic matrices of the record of the E8 gain: 512 by 2048, Gaussian and Student t with 4 degrees of freedom, from the same seeds.
- **whisper.** The 32 matrices of the first record (SHA-256 of `weights.npz`: `b6c8ee500656e04e8e57c2949a5253f0dda002ba36cd1561846574dbcf01132e`). Rows of 512 or 2,048.
- **qwen.** 21 matrices of `Qwen2.5-1.5B-Instruct`, the seven kinds of block matrix in layers 0, 13 and 27, from the publisher's 16-bit float GGUF file (SHA-256 `fc89e330deb3fd8fa560f1c0f35a1e2b8da96d59e13445559ed190307a6f5649`). Each is cut to its first rows, up to 393,216 weights. Rows of 1,536 or 8,960. This is the model issue #3 will be run on. No codebook has been run on it.

Cells: groups of 32 at q = 16, groups of 128 at q = 16, one group per row at q = 16 and at q = 32. A block has to divide its group, so groups of 32 take n up to 32.

## Measure

Relative squared error of the rounded matrix stored as 16-bit floats, in dB. The gain of an arm is the scalar rule's dB minus the arm's on the same matrix in the same cell. No arm has more bits than the scalar rule or the E8 Voronoi code in its cell, and nothing is added for the bits it has less.

## Predictions

1. **The same codebooks.** Over 8 weights the four arms give the errors of the region study on the 38 matrices they share, to 6 digits, in the three cells they share.
2. **Gaussian data, more weights.** For ZB and for EB the mean gain over X rises with n in every cell. Per row at q = 16 it rises by at least 0.5 dB from n = 8 to 16, and by at least 1.6 dB from n = 8 to 128.
3. **Gaussian data, the formula.** For ZB per row at q = 16 the mean gain is within 0.3 dB of the ball column above at every n.
4. **Gaussian data, one ball per group.** With n equal to the group, EB beats VE8 on all three matrices, by at least 0.6 dB with groups of 32 and by at least 1.3 dB with groups of 128.
5. **Heavy tails.** On T, at every n and in every cell, EC beats EB in the mean.
6. **Real matrices, more weights.** On whisper and on qwen, in every cell, EB at the largest n beats EB at n = 8 in the mean and on at least three quarters of the matrices.
7. **Real matrices, against the best arm so far.** On whisper and on qwen, in every cell, EB at the largest n beats VE8 in the mean by at least 0.5 dB.
8. **Integers alone.** On whisper and on qwen, per row at q = 16, ZB at n = 128 beats VE8 in the mean. So integer levels with no lattice would be ahead of the best arm of the second record.

If 6 and 7 hold there is an arm worth scoring on a model's output, and the plan for issue #3 will name it before that run. If 3 holds and 7 fails, real weights are not round enough for the ball, and the cross-polytope columns say whether its shape is the reason.

Reported without a prediction: every arm in every cell, the cross-polytope on the real matrices, which n is best, the share of blocks shrunk, and the errors at equal bits across group sizes.

## What this run cannot show

- Anything about a model's output. The second record found that finer groups helped the output by more than their weight error did, so an arm with one ball per 128 weights may do worse on an output than its weight error says.
- The cost of numbering the points of these codebooks, in time or in tables. That cost grows with n.
- Whether a ball is the best region over many weights. Two shapes are run.
- Anything at 2 or 3 bits.
