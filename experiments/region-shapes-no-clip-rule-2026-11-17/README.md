# Codebooks cut from a lattice by a ball or a cross-polytope, under the rule that clips nothing

## Question

The record [e8-gain-largest-entry-rule-2026-11-17](../e8-gain-largest-entry-rule-2026-11-17/README.md) found that most of E8's gain over scalar rounding under the largest-entry rule comes from the scale, and that the scale depends on the shape of the code's region. It computed what other regions would need and ran none. This run builds four such codebooks and measures them.

1. Do integer levels cut by a ball get the scale part of the gain, with no lattice?
2. Does E8 cut by a ball beat the E8 Voronoi code on Gaussian data?
3. Does E8 cut by a cross-polytope beat the E8 Voronoi code on real weights, at no more bits?

## Status

`complete`. It measures weight error only, on synthetic matrices and the 32 matrices of one small speech model. The codebooks have no packed format and no way to number their points is given.

## Preregistration

[PREREG.md](PREREG.md), committed and pushed at `6964d05` before the run. `results.start` holds the start time and that commit. 5 of 6 predictions hold.

## Result in short

1. **Integer levels in a ball get most of E8's gain.** Against the scalar rule they gain 0.80 dB on Gaussian data with groups of 32 and 1.83 to 1.88 dB per row, where the E8 Voronoi code gains 1.36 and 2.29 to 2.32. On the real matrices they reach 76 to 89 percent of its gain in dB. No lattice is involved.
2. **E8 cut by a ball beats the E8 Voronoi code, by a little, almost everywhere.** By 0.08 to 0.22 dB on Gaussian data, and by 0.13 to 0.20 dB on the real matrices, where it is ahead on 31, 32 and 29 of the 32. On Student t data per row it is behind by 0.3 dB.
3. **E8 cut by a cross-polytope beats it on heavy tails, and on about half the real matrices.** By 0.75 dB on Student t data with groups of 32 and by 2.8 to 2.9 dB per row. On the real matrices per row by 0.40 and 0.49 dB in the mean, ahead on 18 and 20 of the 32. On Gaussian data it loses by 0.6 to 0.7 dB.
4. **Which of the two suits a matrix follows how peaked its blocks are.** The rank correlation between the cross-polytope's gain and the mean peak of a block is 0.77 to 0.91 over the 32 matrices.
5. **Taking the better of the two regions on each matrix beats the E8 Voronoi code on all 32 real matrices in all three cells**, by 0.25 dB in the mean with groups of 32 and by 0.59 and 0.64 dB per row. This table was not planned.

## How it was measured

| | |
|---|---|
| Codebook | every point of the lattice inside the region, with the largest bound at which it has no more than q^8 points |
| ZB, ZC | the integers, cut by a ball and by a cross-polytope |
| EB, EC | E8, cut by a ball and by a cross-polytope |
| Rule | the scale of a group is the smallest at which every block of 8 rounds to a point of the codebook |
| X, VE8 | the scalar rule and the E8 Voronoi code. Not run here: read from `results.json` of the record named above, same matrices and cells |
| Bits | log2 of the number of codewords over 8, and 16 per group. No arm has more bits than X or VE8 |
| Data | G and T, 512 by 2048, three seeds each. The 32 real matrices of the first record |
| Measure | relative squared error of the rounded matrix stored as 16-bit floats, in dB |

A cross-polytope is the set of blocks whose entries' sizes add up to at most a bound. A ball is the set of blocks of at most a given length.

| Arm | q = 16: bound | Codewords over 2^32 | Bits | q = 32: bound | Codewords over 2^40 | Bits |
|---|---|---|---|---|---|---|
| ZB | squared length 179 | 0.9795 | 3.9963 | 720 | 0.9953 | 4.9991 |
| ZC | sum of sizes 29 | 0.8823 | 3.9774 | 59 | 0.9161 | 4.9842 |
| EB | squared length 178 | 0.9682 | 3.9942 | 720 | 0.9981 | 4.9997 |
| EC | sum of sizes 29 | 0.7959 | 3.9588 | 59 | 0.8782 | 4.9766 |

The counts are exact. `run.py` checks them against a scan of every point in a box for 20 small regions, and against the known numbers of points of E8 of squared length 2, 4, 6 and 8.

All numbers below are in [results.out](results.out) or [detail.out](detail.out).

## Results

### Gain over the E8 Voronoi code, in dB

A positive number means the arm has the smaller error. In brackets: the matrices on which it is ahead.

| Data | Group | q | ZB | ZC | EB | EC |
|---|---|---|---|---|---|---|
| G | 32 | 16 | -0.57 (0 of 3) | -1.15 (0) | +0.08 (3) | -0.58 (0) |
| G | row | 16 | -0.46 (0) | -1.29 (0) | +0.20 (3) | -0.69 (0) |
| G | row | 32 | -0.43 (0) | -1.20 (0) | +0.22 (3) | -0.58 (0) |
| T | 32 | 16 | -0.56 (0) | +0.19 (3) | +0.07 (3) | +0.75 (3) |
| T | row | 16 | -0.80 (0) | +2.26 (3) | -0.27 (0) | +2.78 (3) |
| T | row | 32 | -0.96 (0) | +2.27 (3) | -0.30 (0) | +2.88 (3) |
| Real | 32 | 16 | -0.42 (2 of 32) | -0.50 (4) | +0.13 (31) | -0.02 (10) |
| Real | row | 16 | -0.30 (4) | -0.06 (12) | +0.20 (32) | +0.40 (18) |
| Real | row | 32 | -0.39 (3) | -0.03 (12) | +0.17 (29) | +0.49 (20) |

On the real matrices the ranges are wide. EC per row at q = 16 runs from -0.78 to +2.41 dB, and EB from +0.00 to +1.04.

### Gain over the scalar rule, in dB

| Data | Group | q | VE8, from the earlier record | ZB | ZC | EB | EC |
|---|---|---|---|---|---|---|---|
| G | 32 | 16 | +1.36 | +0.80 | +0.22 | +1.44 | +0.79 |
| G | row | 16 | +2.29 | +1.83 | +1.00 | +2.49 | +1.60 |
| G | row | 32 | +2.32 | +1.88 | +1.12 | +2.54 | +1.74 |
| T | 32 | 16 | +2.89 | +2.32 | +3.07 | +2.95 | +3.64 |
| T | row | 16 | +4.61 | +3.81 | +6.87 | +4.34 | +7.39 |
| T | row | 32 | +5.02 | +4.06 | +7.30 | +4.72 | +7.90 |
| Real | 32 | 16 | +1.77 | +1.35 | +1.27 | +1.90 | +1.75 |
| Real | row | 16 | +2.71 | +2.40 | +2.64 | +2.91 | +3.10 |
| Real | row | 32 | +2.92 | +2.53 | +2.89 | +3.10 | +3.41 |

### The predictions

| # | Prediction | Outcome |
|---|---|---|
| 1 | ZB beats X on every matrix of G, by 0.5 to 1.2 dB with groups of 32 and by 1.5 to 2.2 dB per row | Holds: 0.79 to 0.81, and 1.82 to 1.89 |
| 2 | EB beats VE8 in the mean over G in all three cells, and per row by 0.1 to 0.4 dB | Holds: +0.08, +0.20 and +0.22 |
| 3 | EC loses to VE8 on every matrix of G in all three cells | Holds |
| 4 | EC beats VE8 on every matrix of T in all three cells, by at least 0.5 dB in the mean with groups of 32 and at least 1.5 dB per row | Holds: +0.75, +2.78 and +2.88 |
| 5 | On the real matrices per row, at q = 16 and q = 32, EC beats VE8 in the mean and on at least 20 of the 32 | Fails at q = 16 on the count: +0.40 dB in the mean, ahead on 18. Holds at q = 32: +0.49 dB, ahead on 20 |
| 6 | On the real matrices ZC beats X in the mean in all three cells by at least 1 dB, is behind VE8 in the mean with groups of 32, and per row is within 0.6 dB of VE8 in the mean | Holds: +1.27, +2.64 and +2.89 over X. Against VE8: -0.50, -0.06 and -0.03 |

### What the run shows

1. **The earlier record's table of other regions was a fair guide to the sign, and high by up to 0.3 dB.** It gave EC a scale term 0.22 dB above the E8 region's with groups of 32 on the real matrices and 0.62 dB per row. Measured in this run: 0.06 dB below and 0.34 dB above (1.58 against 1.64, and 2.82 against 2.48). The plan named the cause before the run: the codebooks of this run have 0.80 to 0.92 of the q^8 points the table assumed.

2. **Integer levels in a ball** (`detail.out`, table 1). Their gain over the scalar rule is 0.59 of the E8 Voronoi code's with groups of 32 on Gaussian data, 0.80 to 0.81 per row, and 0.76 to 0.89 on the real matrices. What they lack against the E8 Voronoi code, 0.3 to 0.6 dB on G and on the real matrices, is of the size of the 0.65 dB the E8 cell is worth.

3. **The better of two regions** (`detail.out`, table 2, not planned). An encoder can round a matrix both ways and keep the better, at one more bit per matrix.

   | Group | q | EB over VE8 | EC over VE8 | Better of the two: mean (min, max) | Ahead of VE8 on | EC is the better on |
   |---|---|---|---|---|---|---|
   | 32 | 16 | +0.13 | -0.02 | +0.25 (+0.04, +2.06) | 32 of 32 | 9 |
   | row | 16 | +0.20 | +0.40 | +0.59 (+0.05, +2.41) | 32 of 32 | 15 |
   | row | 32 | +0.17 | +0.49 | +0.64 (+0.09, +2.35) | 32 of 32 | 16 |

4. **Which matrices the cross-polytope suits** (`detail.out`, tables 3 and 4, not planned). The peak of a block is its largest entry over its length: 0.650 in the mean for Gaussian blocks, 1 for a single weight. Over the 32 matrices the rank correlation of EC's gain over VE8 with the mean peak is +0.91 with groups of 32 and +0.77 per row. The five matrices where EC does worst per row have a mean peak of 0.651 to 0.662. The matrix where 42 percent of the columns are near zero, on which the E8 Voronoi code gained 0.07 dB over the scalar rule in the earlier record, is among the five where EC does best. With one group per row at q = 16, ZC is 3.18 dB ahead of the E8 Voronoi code there, EC 2.19 and EB 1.04.

5. **A block outside the codebook at its group's scale** was found only for EC, in 47 of its 114 runs, for at most 0.015 percent of the blocks. Those blocks were shrunk as the plan says.

## Deviations from the plan

- **`detail.py` was written after the run.** None of its five tables was planned.
- No other deviation. The run took 102 seconds.

## Limits

- **Weight error only.** Nothing here says a model rounded this way works better.
- **No packed format.** A Voronoi code turns an index into a codeword with one nearest-point step and no table. The codebooks of this run need a way to number the points of a lattice inside a ball or a cross-polytope. For the integers in a cross-polytope that is the known indexing of pyramid vector quantisation. For E8 none is given here, and neither its speed nor its cost in bits is measured.
- **The arms have a little under the bits of the references**, by up to 0.04 bits per weight. Nothing is added to their gains for that.
- **One real model**, a small speech transformer. Whether the weights of a language model have blocks peaked enough for the cross-polytope is not known. The second record found large entries in gemma's `ffn_down`, one in 5,845 beyond 5 standard deviations of its row, which points that way and proves nothing.
- **Two region shapes.** Nothing says either is the best.
- **Three seeds** per synthetic family.
- **The better-of-two table is a choice made by the encoder on each matrix,** so its error is one an encoder can reach. It was still written after the data, and no prediction covers it.
- **The references come from another program.** X and VE8 were rounded by the harness in 32-bit floats and the arms of this run in 64-bit floats. Both are stored as 16-bit floats before the error is taken.

## Read against the closure rule

Nothing is closed by this run. It is a measurement.

For issue #4, which asks whether E8 can be beaten: in these runs the E8 Voronoi code was beaten in weight error, at no more bits, by E8 itself with another region. By 0.1 to 0.2 dB with a ball on nearly every real matrix, and by 0.6 dB per row with the better of two regions. No other lattice in 8 dimensions was needed, and the limits table says none could add more than 0.08 dB by its cell.

## How to rerun

Needs NumPy and the weights file. It reads `results.json` of [e8-gain-largest-entry-rule-2026-11-17](../e8-gain-largest-entry-rule-2026-11-17/README.md), which needs torch to regenerate.

```bash
python3 experiments/region-shapes-no-clip-rule-2026-11-17/run.py --weights weights.npz > results.out
```

```bash
python3 experiments/region-shapes-no-clip-rule-2026-11-17/detail.py > detail.out
```

`run.py --smoke` runs the code path on a small matrix that is not in the study.

## Disclosure

The plan, the programs and this record were written by an AI agent session under the owner's direction.
