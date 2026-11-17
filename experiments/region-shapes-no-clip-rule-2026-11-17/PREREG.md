# Plan: codebooks cut from a lattice by a ball or a cross-polytope, under the rule that clips nothing

Written 2026-11-17, before the run. The run is `run.py` in this folder at the commit that adds this file.

## Question

The record [e8-gain-largest-entry-rule-2026-11-17](../e8-gain-largest-entry-rule-2026-11-17/README.md) found that under the rule that clips nothing, most of E8's gain over scalar rounding comes from the scale the rule can use, and that the scale depends on the shape of the code's region. It then computed, without running any codebook, what scale other regions would need. On the real matrices a cross-polytope came out ahead of the E8 region, and on Gaussian data a ball did.

That table is arithmetic. This run builds the codebooks and measures them.

1. Do integer levels cut by a ball get the scale part of the gain, with no lattice?
2. Does E8 cut by a ball beat the E8 Voronoi code on Gaussian data?
3. Does E8 cut by a cross-polytope beat the E8 Voronoi code on real weights, at no more bits?

Issue #4 asks whether E8 can be beaten. The third question is a way it might be, by the same lattice with another region.

## The codebooks

A codebook is every point of a lattice inside a region: for a ball, the points of squared length at most a bound, and for a cross-polytope, the points whose entries' sizes add up to at most a bound. The bound is the largest at which the codebook has no more than q^8 points. So an arm never has more bits than the E8 Voronoi code at q levels.

The sizes are counted exactly by `run.py`, which checks its counts against a scan of every point in a box for 20 small regions, and against the known numbers of points of E8 of squared length 2, 4, 6 and 8.

| Arm | Lattice | Region | q = 16: bound, codewords over 2^32, bits | q = 32: bound, codewords over 2^40, bits |
|---|---|---|---|---|
| ZB | integers | ball | squared length 179, 0.9795, 3.9963 | 720, 0.9953, 4.9991 |
| ZC | integers | cross-polytope | sum of sizes 29, 0.8823, 3.9774 | 59, 0.9161, 4.9842 |
| EB | E8 | ball | squared length 178, 0.9682, 3.9942 | 720, 0.9981, 4.9997 |
| EC | E8 | cross-polytope | sum of sizes 29, 0.7959, 3.9588 | 59, 0.8782, 4.9766 |

Bits are per weight, without the 16 bits per group for the scale. These codebooks have no simple index: a packed format would need a way to number their points, which a Voronoi code does not. Nothing here measures that cost.

## The rule

As in the second record. The scale of a group is the smallest at which every block of 8 rounds to a point of the codebook. It is found per block by the bisection of the harness, 14 steps, and the group takes the largest. If a block then lies outside the codebook at its group's scale, it is shrunk toward zero, and the share of such blocks is reported.

The references are not run here. The errors of X, the scalar rule, and VE8, the E8 Voronoi code, are read from `results.json` of the record named above, which ran them with the harness on the same matrices in the same cells. As there, the rounded matrix is stored as 16-bit floats before its error is taken.

## What is known before the run

From the record named above, at q = 16. Scale terms in dB, for regions of the full volume q^8:

| Data | Group | E8 Voronoi region, measured | Ball | Cross-polytope | VE8 over X, measured |
|---|---|---|---|---|---|
| G | 32 | +1.01 | +1.15 | +0.70 | +1.36 |
| G | row | +1.64 | +1.90 | +1.24 | +2.29 |
| T | 32 | +2.66 | +2.78 | +3.68 | +2.89 |
| T | row | +4.42 | +4.15 | +7.48 | +4.61 |
| Real | 32 | +1.64 | +1.78 | +1.86 | +1.77 |
| Real | row | +2.48 | +2.63 | +3.10 | +2.71 |

Two things lower what the arms of this run can reach against that table.

- **The codebooks are smaller than q^8.** EC at q = 16 has 0.80 of the points, which is 0.04 bits and about 0.25 dB.
- **The block on the wall.** One block of every group has a quarter more error than the mean. It costs every arm here what it costs VE8, so it does not change an arm against VE8, and it costs 0.26 dB against X with groups of 32.

So the estimates are rough to about 0.3 dB, and with groups of 32 on the real matrices the sign of EC against VE8 cannot be called: +0.22 from the table, less up to 0.3.

**Seen before writing this plan.** `run.py --smoke` on a 32 by 256 matrix of uniform numbers that is not in the study. The table of codebook sizes above. All of the record named above. No arm of this run has been run on a matrix of the study. The main run of the study `incentre-vortex-codebooks-2026-11-17` was still in progress and none of its results had been read.

## Data and cells

The matrices of the record named above: G and T, 512 by 2048, three seeds each, from the same seeds, and the 32 real matrices from the same file (SHA-256 `b6c8ee500656e04e8e57c2949a5253f0dda002ba36cd1561846574dbcf01132e`).

Three cells, the three bit points of the second record: groups of 32 at q = 16, one group per row at q = 16, one group per row at q = 32.

## Measure

Relative squared error in dB. The gain of an arm over X or over VE8 is the difference in dB on the same matrix in the same cell. The arm has no more bits than either, by 0.001 to 0.04 bits per weight. No correction is made for that, so a gain is understated a little.

## Predictions

1. **Integers in a ball, Gaussian data.** ZB beats X on every matrix of G, by 0.5 to 1.2 dB with groups of 32 and by 1.5 to 2.2 dB per row.
2. **E8 in a ball, Gaussian data.** EB beats VE8 in the mean over G in all three cells, and per row by 0.1 to 0.4 dB.
3. **E8 in a cross-polytope, Gaussian data.** EC loses to VE8 on every matrix of G in all three cells.
4. **E8 in a cross-polytope, heavy tails.** EC beats VE8 on every matrix of T in all three cells, by at least 0.5 dB in the mean with groups of 32 and by at least 1.5 dB per row.
5. **E8 in a cross-polytope, real matrices.** Per row, at q = 16 and at q = 32, EC beats VE8 in the mean over the 32 and on at least 20 of them. With groups of 32 no direction is predicted.
6. **Integers in a cross-polytope, real matrices.** ZC beats X in the mean in all three cells by at least 1 dB. It is behind VE8 in the mean with groups of 32. Per row its mean is within 0.6 dB of VE8's, on either side.

If prediction 5 holds, there is a codebook that beats the best arm of the second record in weight error on one model's weights at no more bits. That is a reason to score it on a model's output, and no more than that. If it fails, the table of other regions in the record named above was not a guide, and the first thing to check is how far the count of a small codebook is from its volume.

Reported without a prediction: every arm against X and VE8 in every cell, the scale terms of the arms, and the share of blocks shrunk.

## What this run cannot show

- Anything about a model's output.
- The cost of numbering the points of these codebooks, in bits wasted or in time. A cross-polytope of integers is the codebook of pyramid vector quantisation, which has a known way to number its points. For E8 in a cross-polytope none is given here.
- Whether the regions tried are the best. Only two shapes are run.
- Whether the gain holds on a language model's weights. The real matrices are from one small speech model.
