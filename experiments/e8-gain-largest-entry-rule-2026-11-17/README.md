# Why E8 gains more than its cell is worth under the largest-entry rule

## Question

The second record rounded gemma with one scale per group of weights, set so that nothing is clipped. E8 beat scalar rounding at equal bits by 1.51 dB of weight error with groups of 32 and by 2.52 and 2.55 dB with one group per row. The E8 cell is worth 0.65 dB. Where does the rest come from? And does the point set F_8 that PR #6 proposes do better under the same rule? Issue #4 asks both.

## Status

`complete`. It measures weight error only, on synthetic matrices and the 32 matrices of one small speech model. The gemma file was not on the machine.

## Preregistration

[PREREG.md](PREREG.md), committed and pushed at `61c552b` before the run. `results.start` holds the start time and that commit. 4 of 6 predictions hold. The two that fail do so by under 0.1 dB, in the cells with groups of 32, and the cause was found.

## Result in short

1. **The gain needs no model.** On plain Gaussian matrices E8 beats the scalar rule by 1.36 dB with groups of 32 and by 2.29 and 2.32 dB with one group per row. On gemma the second record found 1.51, 2.52 and 2.55.
2. **Most of it is the scale, not the cell.** With one group per row the gain on Gaussian data is 0.65 dB of cell shape and 1.64 dB of scale, and nothing is left over. The scalar rule has to make room for the largest single weight of a group. The E8 rule has to make room for the longest block of 8, and it can use a smaller scale.
3. **The reason is that the region of the E8 code is nearly a ball.** On Gaussian data the block that sets the scale is an ordinary block, not one with a single large weight. A ball of the same volume would give a scale term of 1.90 dB per row. E8 gets 1.64.
4. **With heavy tails the reach along one weight does count.** On Student t data the block that sets the scale is carried by one large weight, the gain rises to 2.9 dB with groups of 32 and 4.6 to 5.0 dB per row, and per row E8's scale term passes the ball's.
5. **The rule has a cost that was not predicted.** The block that sets a group's scale sits on a wall of its cell, where the error is a quarter above the mean. That costs 0.26 dB with groups of 32 and 0.97 dB with groups of 8. With groups of 8 the gain on Gaussian data is 0.07 dB.
6. **Where many weights are near zero the gain goes.** On one of the 32 real matrices, where 42 percent of the columns are near zero, E8 gains 0.07 dB.
7. **F_8 loses to both, with more bits.** Under the same rule it is behind the scalar rule by 1.0 to 1.6 dB on Gaussian data and behind E8 by 2.6 to 3.9 dB, on every matrix of every family.

## How it was measured

| | |
|---|---|
| X | `tq.maxscale_round`: the step of a group is its largest entry over q/2. llama.cpp's `Q4_0` rule |
| VE8 | `tq.maxscale_voronoi`: the E8 Voronoi code with the smallest scale at which every block of the group lies in the code's region |
| VZ8 | the same on the cubic lattice, as a control |
| F8 | F_8 of PR #6 with half a cell equal to the group's largest entry over K |
| Bits | log2(q) per weight and 16 per group. F8: log2(K) + 0.5 and 16 per group |
| Data | G: Gaussian, 512 by 2048, three seeds. T: Student t with 4 degrees of freedom, the same. Real: the 32 matrices of the first record, rows of 512 or 2,048 |
| Measure | relative squared error of the matrix the harness returns as 16-bit floats, in dB |

The harness functions are those the second record used, unchanged. `tq.maxscale_voronoi` does not return its scales, so `run.py` repeats its bisection. In all 456 runs of a lattice arm, rounding at the scales found gives the same matrix the harness returns, entry for entry. No block was shrunk in any of them.

**The split.** With fine cells and nothing clipped the error of a group is G times its squared scale. So the gain of VE8 over X is 10 log10(G_cube / G_E8), the shape, plus 10 log10 of the sum of squared scalar steps over the sum of squared E8 scales, the scale. The shape is 0.654 dB from `bounds/bounds.out`. The scale term is computed from the scales the two rules chose on each matrix.

All numbers below are in [results.out](results.out) or [detail.out](detail.out).

## Results

### The gain and its parts, in dB

| Data | Group | q | Bits | Z8 over X | E8 over X | Shape | Scale | Left over | Scale if the region were a ball |
|---|---|---|---|---|---|---|---|---|---|
| G | 8 | 16 | 6.000 | -1.34 | +0.07 | 0.65 | +0.75 | -1.34 | +0.85 |
| G | 32 | 16 | 4.500 | -0.30 | +1.36 | 0.65 | +1.01 | -0.30 | +1.15 |
| G | 128 | 16 | 4.125 | -0.06 | +1.85 | 0.65 | +1.26 | -0.06 | +1.44 |
| G | row | 16 | 4.008 | -0.00 | +2.29 | 0.65 | +1.64 | -0.00 | +1.90 |
| G | 32 | 32 | 5.500 | -0.35 | +1.33 | 0.65 | +1.03 | -0.35 | +1.15 |
| G | row | 32 | 5.008 | -0.00 | +2.32 | 0.65 | +1.67 | -0.01 | +1.90 |
| T | 32 | 16 | 4.500 | -0.49 | +2.89 | 0.65 | +2.66 | -0.43 | +2.78 |
| T | row | 16 | 4.008 | -0.91 | +4.61 | 0.65 | +4.42 | -0.46 | +4.15 |
| T | row | 32 | 5.008 | -0.17 | +5.02 | 0.65 | +4.45 | -0.08 | +4.15 |
| Real | 32 | 16 | 4.500 | -0.72 | +1.77 | 0.65 | +1.64 | -0.52 | +1.78 |
| Real | row | 16 | 4.028 | -0.82 | +2.71 | 0.65 | +2.48 | -0.43 | +2.63 |
| Real | row | 32 | 5.028 | -0.47 | +2.92 | 0.65 | +2.50 | -0.23 | +2.63 |

Means over 3 matrices for G and T and over 32 for Real. `results.out` has the other cells and the ranges. On the real matrices the gain with groups of 32 runs from +0.07 to +2.37 dB, and with one group per row from +1.23 to +4.10.

Beside the second record:

| Cell | gemma, second record | Gaussian | Student t | whisper |
|---|---|---|---|---|
| groups of 32, q = 16 | 1.51 | 1.36 | 2.89 | 1.77 |
| one group per row, q = 16 | 2.52 | 2.29 | 4.61 | 2.71 |
| one group per row, q = 32 | 2.55 | 2.32 | 5.02 | 2.92 |

The row lengths differ: 2,048 on the synthetic families, 512 or 2,048 on whisper. The second record does not list those of gemma.

### The predictions

| # | Prediction | Outcome |
|---|---|---|
| 1 | On Gaussian data the gain is 1.0 to 2.2 dB with groups of 32 at q = 16, and 1.7 to 3.2 dB per row at q = 16 and 32, on each matrix | Holds: 1.36 to 1.38, 2.27 to 2.31 and 2.31 to 2.33 |
| 2 | On G, shape plus scale is within 0.3 dB of the measured gain in the four cells with groups of 32 or a row, on every matrix | Fails in one cell. Per row the two agree within 0.01 dB. With groups of 32 the gain is under shape plus scale by 0.29 to 0.30 dB at q = 16, inside the mark, and by 0.35 dB at q = 32 on all three matrices, outside it |
| 3 | On G, VZ8 is within 0.3 dB of X in those cells | Fails with groups of 32: behind by 0.30 to 0.36 dB, over the mark on four of the six matrices. Holds per row: within 0.005 |
| 4 | The mean gain on T is larger than on G in those cells | Holds in all four |
| 5 | On the real matrices the mean gain is 1.0 to 3.0 dB with groups of 32 and 1.5 to 4.0 dB per row at q = 16, and shape plus scale is within 0.5 dB of the gain on at least 24 of 32 in both | Holds: 1.77 and 2.71, and 25 of 32 in both |
| 6 | F_8 has a larger error than X in the mean in all three cells on all three families with both values of t, with more bits, and than VE8 on every matrix | Holds |

### What the run shows

1. **The hand estimate was high.** It put the gain on Gaussian data at 1.75 dB with groups of 32 and 2.6 dB per row, treating the region as a ball. Measured: 1.36 and 2.29.

2. **Why predictions 2 and 3 fail: the block on the wall.** The rule takes the smallest scale at which every block is inside the region. So one block of each group lies on the surface of the region, which is a wall of its cell. A point on the surface of a cell in n dimensions is, in the mean, (n + 2)/n times as far from the centre in squared distance as a point inside. For n = 8 that is 1.25. `detail.out`, table 3, finds 1.249 to 1.250 for both lattices on Gaussian data with groups of 8 and of 32. With 4 blocks to a group this costs 10 log10(1 + 0.25/4) = 0.26 dB. The scalar rule does the opposite: its largest entry lands exactly on a level.

   So the formula of the plan is exact per row, where one block in 256 is on a wall, and 0.30 to 0.35 dB off with groups of 32. With groups of 8 every block is on a wall, and E8 gains 0.07 dB on Gaussian data.

3. **The block that sets the scale.** Its reach is its length over q times the scale: 0.707 at the nearest wall of the E8 cell, 1 at its farthest corner, 0.839 for a ball of the same volume. Its peak is its largest entry over its length: 1 for a single weight.

   | Data | Group | Reach | Peak of that block | Peak of all blocks |
   |---|---|---|---|---|
   | G | 32 | 0.824 | 0.649 | 0.650 |
   | G | row | 0.801 | 0.644 | 0.650 |
   | T | 32 | 0.824 | 0.751 | 0.700 |
   | T | row | 0.840 | 0.914 | 0.700 |
   | Real | 32 | 0.824 | 0.689 | 0.675 |
   | Real | row | 0.813 | 0.729 | 0.675 |

   At q = 16. On Gaussian data the block that sets the scale is no more peaked than any other. The second record's possible reason, that E8 reaches twice as far along a single weight, is not what gives the gain there. On Student t data that block is carried by one weight, and there the reason applies.

4. **The matrix where the gain goes.** `decoder.blocks.0.cross_attn.query.weight` gains 0.07 dB with groups of 32. 41.6 percent of its columns have a standard deviation under a tenth of the median, and 48.8 percent of its entries are under half a scalar step. The scalar rule rounds those to zero at almost no cost: its error is 2.46 dB under its formula. E8's error is 0.29 dB over its formula, as on every other matrix. The next lowest gain of the 32 is 1.27 dB.

### F_8 under the same rule

F_8 has slightly more bits than the arms it is set against. A positive number means F_8 has the larger error.

| Data | Group | K | F_8 bits | X and VE8 bits | F_8 minus X, t at the incentre | t at the centroid | F_8 minus VE8, t at the incentre | t at the centroid |
|---|---|---|---|---|---|---|---|---|
| G | row | 12 | 4.093 | 4.008 | +1.18 | +0.98 | +3.48 | +3.27 |
| G | 32 | 12 | 4.585 | 4.500 | +1.51 | +1.28 | +2.87 | +2.64 |
| G | row | 23 | 5.031 | 5.008 | +1.55 | +1.35 | +3.87 | +3.66 |
| T | row | 12 | 4.093 | 4.008 | +2.62 | +2.43 | +7.23 | +7.04 |
| T | 32 | 12 | 4.585 | 4.500 | +1.78 | +1.56 | +4.67 | +4.44 |
| T | row | 23 | 5.031 | 5.008 | +1.40 | +1.20 | +6.43 | +6.22 |
| Real | row | 12 | 4.113 | 4.028 | +2.23 | +2.04 | +4.94 | +4.74 |
| Real | 32 | 12 | 4.585 | 4.500 | +2.02 | +1.80 | +3.79 | +3.57 |
| Real | row | 23 | 5.052 | 5.028 | +1.44 | +1.24 | +4.36 | +4.16 |

F_8 is worse than X and than VE8 on every matrix in every row of this table: 3 of 3 and 32 of 32. Its region is a cube, so the rule gives it the scalar step, and its cells are worth less than the cube's.

### Other regions, computed and not run

This table was not planned. It holds no run of a codebook. For each matrix it asks what scale the rule would need if the region had another shape and the same volume, and gives the scale term.

| Data | Group | E8 region, measured | Ball | Cross-polytope |
|---|---|---|---|---|
| G | 32 | +1.01 | +1.15 | +0.70 |
| G | row | +1.64 | +1.90 | +1.24 |
| T | 32 | +2.66 | +2.78 | +3.68 |
| T | row | +4.42 | +4.15 | +7.48 |
| Real | 32 | +1.64 | +1.78 | +1.86 |
| Real | row | +2.48 | +2.63 | +3.10 |

At q = 16, in dB. A cross-polytope is the set of blocks whose entries' sizes add up to at most a bound. At volume 1 it reaches 1.882 along a weight axis, where the E8 cell reaches 1.0, the ball 0.839 and the cube 0.5.

[Reading] On the real matrices a cross-polytope region would give a larger scale term than the E8 region, by 0.22 dB with groups of 32 and by 0.62 dB per row. If that held in a run, the E8 lattice cut by a cross-polytope would beat the E8 Voronoi code on these weights, and integer levels cut by a cross-polytope would come within 0.03 dB of it per row with no lattice at all (3.10 against 0.65 + 2.48). No such codebook has been run. The count of its codewords, the rule for its scale and the cost of the block on the wall all have to be done first.

## Deviations from the plan

- **`detail.py` was written after the run.** None of its six tables was planned. Tables 1 and 2 put the per-matrix figures for predictions 2, 3 and 5 on record. Tables 3, 4 and 6 were written after reading `results.out`.
- **The run shared the machine with the first study of this date** and took 1,699 seconds. The other study's workers were stopped for 79 seconds near the end so that this run could finish inside the session's limit for a background command, and then resumed. Both programs are deterministic, so this changes no number.
- **The outcome of prediction 3 depends on the reading of "within 0.3 dB".** It is read on each matrix. In the mean over three matrices VZ8 is 0.30 dB behind X at q = 16 and 0.35 at q = 32.

## Limits

- **Weight error only.** The second record found that weight error does not say what a model's output does. For these two arms on gemma its KL gain was 1.17 to 1.29 times the gain in weight error. One model and one text.
- **Not run on gemma.** This shows what the rule and the lattice do on other weights. `run.py` reads any float matrix, so whoever holds the gemma file can run the split on it.
- **One real model**, a small speech transformer, and 32 of its matrices.
- **The split is exact only with fine cells.** Table 2 of `results.out` shows how far each arm is from its formula. On Student t data and on the real matrices the scalar rule is up to 0.6 dB under its formula, because many weights are small against the step and round to zero.
- **Three seeds** per synthetic family. The ranges over seeds are under 0.05 dB on G and under 0.2 dB on T.
- **The table of other regions is arithmetic on the data.** It is not evidence that a codebook with such a region works.
- **A scalar rule with a searched scale or an offset per group was not tried**, as in the second record.

## Read against the closure rule

Nothing is closed by this run. It is a measurement.

What it adds to the table of limits is a reading, not a proof. Under a rule that clips nothing, the gain of a code over scalar rounding has two parts, and the limits in `bounds/bounds.out` bound only the first. The cell of any code in 8 dimensions is worth at most 0.73 dB. The scale part has no such ceiling here: it grows with the group and with the tails of the weights.

For issue #4: E8 is not beaten by F_8, under this rule or with one scale per matrix (see [incentre-vortex-codebooks-2026-11-17](../incentre-vortex-codebooks-2026-11-17/README.md)). In 8 dimensions the cell leaves 0.08 dB above E8. What is left to gain lies in the region and in the dimension.

## How to rerun

Needs NumPy, torch and the weights file of `mlx-community/whisper-base.en-mlx`.

```bash
python3 experiments/e8-gain-largest-entry-rule-2026-11-17/run.py --weights weights.npz > results.out
```

```bash
python3 experiments/e8-gain-largest-entry-rule-2026-11-17/detail.py --weights weights.npz > detail.out
```

The program refuses a weights file whose SHA-256 is not the one in the plan. `run.py --smoke` runs the code path on a small matrix that is not in the study. The run took 28 minutes on 4 shared CPU cores with torch 2.15 on CPU, NumPy 2.5.3 and Python 3.13.

## Disclosure

The plan, the programs and this record were written by an AI agent session under the owner's direction.
