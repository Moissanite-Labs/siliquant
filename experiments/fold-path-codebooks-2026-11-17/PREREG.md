# Plan: the fold as a path of codebooks over 3 weights

Written 2026-11-17, before any state of the fold was measured and before any matrix was rounded. The run is `run.py` in this folder at the commit that adds this file. This is the test of issue #9.

## Question

Issue #9 asks: does annealing a codebook along the fold of `Foundation-0/incentre-lattices`, from open packing to close packing, reach a lower weight error than Lloyd's algorithm, at equal bits?

The fold is a stored sequence of states. Each state is a periodic set of points in 3 dimensions: 48 points in a cell that may shear. It starts at the open packing F_3 and ends at a close packing. A state can be used as a codebook for blocks of 3 weights.

1. What is the cell of each state worth, along the whole path?
2. Under the rule that clips nothing, does any state give a lower weight error than a fitted codebook of the same size? Than a ball cut from the integers? Than E8?
3. Does starting Lloyd's algorithm from the fold's end give a better codebook than starting it from a sample of the data?

## What is settled before the run

- **The end point has a ceiling.** The fold ends at the face-centred cubic lattice or at hexagonal close packing. The face-centred cubic cell is worth 0.246 dB over a cube. No cell in 3 dimensions is worth more than 0.345 dB, and the E8 cell is worth 0.654 (`bounds/bounds.out`, sections 2 and 9).
- **The start is known.** F_3 has G = 0.127803, which is 1.857 dB worse than a cube (`bounds/bounds.out`, section 7).
- **The fold moves the cell and not the region.** Every state is spread evenly through space. The records of this date found most of E8's gain under this rule in the region of its code.
- **The formula for the region.** For Gaussian weights with fine cells, a ball over 3 weights is worth this over the scalar rule (`bounds/bounds.out`, section 10, the same integrals for 3 weights, computed for this plan):

| Group | Ball over 3 weights | Ball over 8 | Ball over 24 |
|---|---|---|---|
| 24 | 0.51 | 1.09 | 1.66 |
| 96 | 0.64 | 1.38 | 2.14 |
| row of 1,920 | 0.87 | 1.90 | 3.02 |

So the formula puts the fold's start in a ball at 0.51 - 1.86 = -1.35 dB against the scalar rule with groups of 24, and its end at 0.51 + 0.25 = +0.76. Per row: -0.99 and +1.12.

## The source

`Foundation-0/incentre-lattices` at commit `2d374ff`. Two files:

| File | States | SHA-256 |
|---|---|---|
| `autoresearch-fold/best/path.npz` | 793, ending in hexagonal close packing | `510d0c79016068198b8f5af8b127067a200498557bcdb01e1da27ac4e78bddd5` |
| `results/fold-close-packing/path_fcc.npz` | 1,193, the same 793 and then two slides of layers, ending in the face-centred cubic lattice | `010bdabf40c8c7b23a8809bc3a4d107f3417d4e602b98b0723ef6cc7f2e717b1` |

The source labels both paths as checked at sampled states and not proved to be continuous motions. Nothing here depends on that: each state is used by itself as a set of points.

In a comment on the issue I said the test would also take the source's explicit motion to the cubic end. It is left out. The two stored paths are the fold the issue names.

## Method

**The cell.** For every eighth state, and the seven states below, G by sampling: 200,000 points uniform in the cell, the squared distance to the nearest point of the set, per dimension, at one point per unit of volume. The program first measures four sets whose G is known (the integers, the face-centred and body-centred cubic lattices, and F_3 built from its definition) and stops if any is more than 4 standard errors off.

**Codebooks over 3 weights.** A codebook is every point of a set whose length is at most a bound, about one of its points. The bound is the largest with no more than 4,096 points, which is 4 bits a weight. Cut this way:

| Arm | Set |
|---|---|
| `F0`, `F198`, `F396`, `F594` | states of the fold: the start and three on the way |
| `F792` | the hexagonal end |
| `F992`, `F1192` | half way through the slides, and the face-centred cubic end |
| `Z3B` | the integers |
| `FCC`, `BCC` | the face-centred and body-centred cubic lattices, built directly |
| `Z3C` | the integers, cut by a cross-polytope and not a ball |

**The rule.** One scale per group of weights: the smallest at which every block of the group rounds to a point of the codebook, found as in the record ball-dimension-no-clip-rule-2026-11-17. Groups of 24 and of 96 weights, and one scale per row. A scale costs 16 bits.

**Beside them.** The scalar rule `X` and the E8 Voronoi code `VE8` from the harness at 16 levels. The integers and E8 in a ball over 8 and over 24 weights (`ZB8`, `EB8`, `ZB24`, `EB24`) with the program of that record.

**A fitted codebook.** Lloyd's algorithm, 25 steps, 4,096 points in 3 dimensions, on up to 600,000 blocks of training matrices. Each block is divided by the scale of its group. Two scales: the largest entry of the group (`Le`), and the length of its longest block (`Ll`). Two starts: 4,096 blocks drawn from the training set (`s`), and the codebook `F792` stretched to reach the longest training block (`f`). The second is the nearest thing here to annealing along the fold and then optimising. The table of a fitted codebook is not counted in its bits.

A fitted codebook never rounds a matrix it was fitted to. For the synthetic families it is fitted to two fresh matrices. The 32 real matrices are split in two halves by their order, and each half is rounded with a codebook fitted to the other.

**Data.** As in the record region-shapes-no-clip-rule-2026-11-17, with rows that divide by 3: three Gaussian and three Student-t (4 degrees of freedom) matrices of 512 by 1,920, and the same 32 matrices of `whisper-tiny.en` (SHA-256 begins `b6c8ee50`), whose rows are 384 or 1,536 long.

**Measure.** Squared error over squared size of the matrix, after storing the rounded values as 16-bit floats. Gains are in dB over `X` in the same group, means over the matrices of a family.

## Seen before writing this plan

- The shapes of the two files, the cell of four states, and that the first 793 states of the second file are the first file.
- The source's own analysis of the end states: densities, contacts.
- `run.py --smoke`: the four self-tests of G, and one matrix of uniform numbers that is not in the study rounded with the face-centred cubic codebook. A timing on another such matrix.
- The formula table above.
- All of the earlier records. No state of the fold was measured and no matrix of the study was rounded with any arm over 3 weights.

## Predictions

The issue's own three are I1 to I3, made specific. The others are mine.

| | Prediction | It fails if |
|---|---|---|
| C1 | The start of the fold has a cell 1.86 dB worse than a cube and both ends 0.25 dB better, each within 0.03 dB | any of the three is outside |
| C2 | No sampled state has a cell more than 0.02 dB better than the better of the two ends | one has |
| I1 | The fitted codebook started from a sample, under the better of its two scales, has a lower error than the best state of the fold in all 9 cells of family and group | a state of the fold is lower in any cell |
| I2 | The best state of the fold has a higher error than `ZB8`, `EB8`, `ZB24`, `EB24` and `VE8` in all 9 cells | it is lower than any of them in any cell |
| I2b | It has a lower error than `Z3B`, the ball cut from the integers over the same 3 weights, by 0.15 to 0.30 dB in all 9 cells | it is outside in any cell |
| I3 | The states of the fold differ by their cells: the gain of each state over `F1192`, less 10 log10 of the ratio of their G, is within 0.15 dB | it is outside for any state |
| F1 | On Gaussian matrices `F0` is worse than `X` in all three groups, and `F1192` is better than `X` by the formula's 0.76, 0.89 and 1.12 dB, each within 0.3 dB | either is not so |
| F2 | `FCC` and `F1192` agree within 0.05 dB in every cell | they differ by more in any cell |
| F3 | Starting Lloyd's algorithm from the fold's end changes its error by less than 0.1 dB against a start from a sample, in all 9 cells and under both scales | it differs by 0.1 dB or more anywhere |

I2b says that the issue's second prediction, read as a ball cut over the same 3 weights, is expected to fail by a quarter of a decibel. That is the cell of close packing against the cube, and it is all the fold can give.

The falsifier of the issue stands: any state of the fold with a lower error than the fitted codebook, at equal bits, on any family.

Reported without a prediction: whether G falls at every step of the path. Which state is best in each cell. The fitted codebook against `VE8` and the balls over 8 and 24 weights.

## What it would mean

- If I1, I2 and F3 hold, the fold has no use as a way to find a codebook for weights. Its best state is close packing, a lattice that is already known, and it is worth a quarter of a decibel.
- If F3 fails with the fold's start better, the fold is worth trying as a start for fitted codebooks, in more dimensions than 3.
- If a state of the fold beats the fitted codebook, the issue's falsifier has fired, and a model-output test follows.

## What this run cannot show

- Anything about a model's output.
- Anything for blocks of more than 3 weights. The fold is in 3 dimensions.
- That Lloyd's algorithm found the best codebook. It finds a local one. Two starts are two.
- Anything at other than 4 bits a weight.

## Amendment 1: the rows of the real matrices

Written 2026-11-17 after the first run stopped, and before any matrix was rounded.

**What was wrong.** The plan says the real matrices are of `whisper-tiny.en` with rows of 384 or 1,536. They are the 32 matrices of `whisper-base.en` that every earlier record used (SHA-256 `b6c8ee500656e04e8e57c2949a5253f0dda002ba36cd1561846574dbcf01132e`), and their rows are 512 or 2,048 long. Neither divides by 3. The program checks this and stopped.

**What the first run had done by then.** The self-tests, the cell of every sampled state, and the table of codebooks. Its output is kept as `first-run.out`. So these were seen before this amendment:

- the whole curve of G. The start is at -1.86 dB and both ends near +0.25. The lowest G sampled is state 808, in the slides, at +0.257 dB;
- the sizes of the eleven codebooks, 4,015 to 4,096 points.

No matrix was loaded or rounded. Predictions C1 and C2 are about the curve, so they are no longer blind. They are scored as written, and the record says so.

**The change.** Each real matrix is cut to the first 480 weights of each row if its rows are 512 long, and to the first 2,016 if they are 2,048. Both divide by 96. Nothing else changes. With one scale per row the real matrices have 4.033 or 4.008 bits a weight.

The sampling is seeded, so the second run repeats the first run's numbers for the curve.
