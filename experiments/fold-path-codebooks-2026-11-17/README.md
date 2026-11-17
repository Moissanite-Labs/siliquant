# The fold as a path of codebooks over 3 weights

## Question

Issue #9 asks whether annealing a codebook along the fold of `Foundation-0/incentre-lattices`, from open packing to close packing, reaches a lower weight error than Lloyd's algorithm, at equal bits.

The fold is a stored sequence of states. Each is a periodic set of points in 3 dimensions. It starts at the open packing F_3 and ends at a close packing. A state can be a codebook for blocks of 3 weights.

1. What is the cell of each state worth, along the whole path?
2. Under the rule that clips nothing, does any state beat a fitted codebook of the same size? A ball cut from the integers? E8?
3. Does starting Lloyd's algorithm from the fold's end give a better codebook than starting it from a sample of the data?

## Status

`complete`. Weight error only, at 4 bits a weight, on 6 synthetic matrices and the 32 matrices of one small speech model. The answer to the issue is no, on all three counts.

## Preregistration

[PREREG.md](PREREG.md), committed and pushed at `cae753e` before the run. Amendment 1 at `cda52eb`, after the first run stopped at its own check and before any matrix was rounded. 4 of 9 predictions hold. Two of the four, C1 and C2, are about the cell along the path, which the stopped run had printed before the amendment. They were scored as written and are not blind.

The issue's falsifier was any state of the fold with a lower error than the fitted codebook. It did not fire.

## Result in short

1. **The cell improves almost all the way along the fold, from 1.86 dB worse than a cube to 0.25 dB better.** It passes the cube at state 656 of 1,193, at a density of about 0.55. The two close-packed ends are worth +0.251 and +0.245 dB. No sampled state is better than them by more than the sampling error.
2. **The best state of the fold is its end, and its end is a lattice that is already known.** The face-centred cubic lattice built directly gives the same error as the fold's last state on Gaussian and real matrices.
3. **Every state of the fold loses to a fitted codebook of the same size**, in all 9 cells of family and group, by 0.6 to 8.1 dB in the mean, and on every real matrix.
4. **Every state loses to the E8 Voronoi code and to a ball over 8 or 24 weights**, in all 9 cells. Integer levels in a ball over 8 weights are 0.3 to 2.2 dB ahead of the best state. Over 24 weights they are 0.9 to 5.1 dB ahead.
5. **The fold's end beats integer levels in a ball over the same 3 weights by 0.2 to 0.4 dB.** That is the cell of close packing against the cube. It is the only thing the fold gives, and a longer ball gives several times as much.
6. **Starting Lloyd's algorithm from the fold's end is worse than starting from a sample**, by 0.1 to 4.5 dB. Stretched over the data, up to half of its points lie where no weights are, and Lloyd's algorithm does not move a point that no data chooses.

## How it was measured

**The source.** `Foundation-0/incentre-lattices` at commit `2d374ff`: `autoresearch-fold/best/path.npz` (793 states, ending in hexagonal close packing) and `results/fold-close-packing/path_fcc.npz` (1,193 states: the same 793, then two slides of layers, ending in the face-centred cubic lattice). SHA-256 sums are in the plan. Each state is a cell of three lattice vectors and 48 points. The source labels the paths as checked at sampled states and not proved continuous. Here each state is used by itself.

**The cell.** For every eighth state: G by sampling 200,000 points, at one point per unit of volume. The program first measures four sets whose G is known and stops if any is 4 standard errors off. They were within 1.7.

**The codebooks.** Every point of a set within a bound of one of its points, the bound the largest with no more than 4,096 points. Seven states of the fold, and beside them the integers (`Z3B`), the face-centred and body-centred cubic lattices built directly (`FCC`, `BCC`), and the integers cut by a cross-polytope (`Z3C`). They hold 4,015 to 4,096 points.

**The rule.** One scale per group, the smallest at which every block of the group rounds to a point of the codebook. Groups of 24 and 96 weights, and one scale per row. 16 bits a scale. That is 4.67, 4.17 and 4.03 bits a weight in the mean.

**Beside them.** The scalar rule `X` and the E8 Voronoi code `VE8` at 16 levels. The integers and E8 in a ball over 8 and 24 weights. A fitted codebook of 4,096 points: Lloyd's algorithm, 25 steps, on 600,000 blocks of matrices it does not then round, each block divided by the largest entry of its group (`Le`) or by the length of its group's longest block (`Ll`), started from a sample of the blocks (`s`) or from the fold's hexagonal end (`f`).

**Data.** Three Gaussian and three Student-t matrices of 512 by 1,920, and the 32 matrices of `whisper-base.en` of the earlier records, cut to the first 480 or 2,016 weights of each row (amendment 1).

## Results

Every number is in `results.out`. Gains are dB of weight error over the scalar rule `X` with the same group, means over the matrices of a family.

### The cell along the fold

| State | Density of the source's balls | G | Against the cube, dB |
|---|---|---|---|
| 0, the open packing | 0.2233 | 0.127964 | -1.863 |
| 198 | 0.2452 | 0.115677 | -1.424 |
| 396 | 0.3343 | 0.099261 | -0.760 |
| 594 | 0.5112 | 0.086534 | -0.164 |
| 792, hexagonal close packing | 0.7405 | 0.078656 | +0.251 |
| 992, half way through the slides | 0.7405 | 0.078763 | +0.245 |
| 1192, face-centred cubic | 0.7405 | 0.078768 | +0.245 |

153 states were sampled. The standard error of a state is 0.005 to 0.007 dB. Between sampled states the gain rises on 120 steps and falls on 32. Before state 700 it falls five times, by 0.011 dB at most, which is inside the sampling error of two states. The lowest G sampled is state 808 at +0.257 dB, 0.006 dB above the hexagonal end and inside the error of the two.

So the fold is a path along which the cell gets better, as issue #9 supposed. Its end is where the limits table said it would be: the face-centred cubic cell is worth 0.246 dB, and no cell in 3 dimensions is worth more than 0.345.

### The states as codebooks

| Arm | G, 24 | T, 24 | whisper, 24 | G, 96 | T, 96 | whisper, 96 | G, row | T, row | whisper, row |
|---|---|---|---|---|---|---|---|---|---|
| `F0` | -1.82 | -1.07 | -1.56 | -1.37 | -0.38 | -0.99 | -1.11 | +0.30 | -0.65 |
| `F396` | -0.69 | -0.06 | -0.55 | -0.24 | +0.58 | -0.00 | +0.04 | +1.00 | +0.29 |
| `F792` | +0.31 | +0.90 | +0.42 | +0.76 | +1.48 | +0.92 | +1.05 | +1.66 | +1.14 |
| `F1192` | +0.33 | +0.91 | +0.43 | +0.77 | +1.50 | +0.93 | +1.06 | +1.68 | +1.15 |
| `FCC` | +0.32 | +0.97 | +0.45 | +0.77 | +1.57 | +0.96 | +1.07 | +1.79 | +1.18 |
| `Z3B` | +0.05 | +0.59 | +0.23 | +0.50 | +1.14 | +0.74 | +0.79 | +1.32 | +0.96 |
| `Z3C` | -0.32 | +1.34 | +0.26 | +0.04 | +2.32 | +0.85 | +0.19 | +3.43 | +1.22 |
| `VE8` | +1.20 | +2.63 | +1.56 | +1.78 | +3.62 | +2.25 | +2.29 | +4.64 | +2.69 |
| `ZB8` | +0.63 | +2.06 | +1.13 | +1.24 | +3.04 | +1.88 | +1.83 | +3.83 | +2.39 |
| `EB8` | +1.28 | +2.70 | +1.68 | +1.88 | +3.67 | +2.41 | +2.49 | +4.37 | +2.89 |
| `ZB24` | +1.22 | +3.47 | +2.06 | +2.02 | +4.99 | +3.08 | +2.96 | +6.81 | +3.94 |
| `EB24` | +1.87 | +4.11 | +2.61 | +2.66 | +5.62 | +3.62 | +3.60 | +7.43 | +4.47 |
| fitted, `Les` | +0.71 | +2.80 | +1.25 | +1.52 | +4.69 | +2.40 | +3.43 | +9.81 | +4.16 |
| fitted, `Lls` | +0.89 | +2.99 | +1.48 | +1.60 | +4.85 | +2.58 | +3.44 | +9.78 | +4.25 |
| fitted from the fold, `Lef` | -1.74 | -0.55 | -1.44 | -0.82 | +1.13 | -0.42 | +0.89 | +5.31 | +1.05 |
| fitted from the fold, `Llf` | +0.81 | +2.22 | +1.21 | +1.35 | +3.47 | +1.99 | +2.43 | +6.26 | +3.01 |

Read down the first four rows: the states of the fold get better in the order of the path, in every column.

Read across the rest. The fold's end is ahead of the integers in the same ball by 0.27 or 0.28 dB on Gaussian matrices, 0.32 to 0.36 on Student t and 0.19 or 0.20 on the real ones. Everything else that was run is ahead of it by more than that. Making the ball longer, with no change of lattice, is worth 0.3 to 2.2 dB at 8 weights and 0.9 to 5.1 dB at 24.

### The states differ by their cells

| State | Gain over `F1192`, mean over all matrices and groups | What the cells say | Difference |
|---|---|---|---|
| `F0` | -1.910 | -2.105 | +0.195 |
| `F198` | -1.534 | -1.674 | +0.140 |
| `F396` | -0.925 | -0.993 | +0.067 |
| `F594` | -0.389 | -0.407 | +0.018 |
| `F792` | -0.011 | +0.003 | -0.014 |
| `F992` | -0.001 | -0.001 | -0.000 |

The cell gives the order and most of the size. At the open end it overstates the loss by 0.2 dB. The cells of an open packing are large and uneven, and a codebook of 4,096 points is not fine against them.

### Against a fitted codebook

The better of the two scales, started from a sample, against the best state of the fold. Mean over matrices, with the least and the most.

| Family | Group | Fitted codebook ahead by, dB |
|---|---|---|
| Gaussian | 24 | 0.56 (0.55 to 0.57) |
| Gaussian | 96 | 0.83 (0.82 to 0.85) |
| Gaussian | row | 2.38 (2.35 to 2.44) |
| Student t | 24 | 2.08 (2.06 to 2.12) |
| Student t | 96 | 3.35 (3.30 to 3.43) |
| Student t | row | 8.12 (8.04 to 8.28) |
| whisper | 24 | 1.06 (0.43 to 2.67) |
| whisper | 96 | 1.65 (0.72 to 4.27) |
| whisper | row | 3.10 (1.03 to 6.50) |

Lloyd's algorithm is ahead on every matrix. It gains the most where one scale serves a whole row, because it can put its points where the weights are, and a set spread evenly through a ball cannot.

### Lloyd's algorithm started from the fold

Started from the fold's hexagonal end and run for the same 25 steps, the fitted codebook is worse than from a sample: by 2.3 to 4.5 dB when blocks are divided by the largest entry of their group, and by 0.07 to 3.5 dB when divided by the length of the longest block.

The table of `results.out`, section 6, shows why. Started from a sample, all 4,096 points end in use. Started from the fold's end under the first scale, 1,550 to 2,371 do. The fold's end fills a ball evenly, the weights fill the middle of it, and a point that no block chooses is never moved.

### The predictions

| | Prediction | Result | Holds |
|---|---|---|---|
| C1 | The start is 1.86 dB worse than a cube and both ends 0.25 better, each within 0.03 | -1.863, +0.251 and +0.245 | yes, not blind |
| C2 | No sampled state is more than 0.02 dB better than the better end | The best is 0.006 dB better | yes, not blind |
| I1 | The fitted codebook beats the best state of the fold in all 9 cells | By 0.56 to 8.12 dB | yes |
| I2 | The best state loses to `ZB8`, `EB8`, `ZB24`, `EB24` and `VE8` in all 9 cells | It loses to each in every cell. The least margin is 0.30 dB, to `ZB8` on Gaussian matrices with groups of 24 | yes |
| I2b | It beats `Z3B` by 0.15 to 0.30 dB in all 9 cells | 0.27 or 0.28 on Gaussian, 0.19 or 0.20 on whisper, and 0.32 to 0.36 on Student t | **no**, outside on Student t |
| I3 | Each state's gain over `F1192`, less what the cells say, is within 0.15 dB | 0.195 at `F0`. Within 0.15 at the other five | **no**, at the open end |
| F1 | On Gaussian matrices `F0` is worse than `X`, and `F1192` is ahead by 0.76, 0.89 and 1.12 dB, within 0.3 | `F0` is worse in all three. `F1192` is ahead by 0.33, 0.77 and 1.06 | **no**, with groups of 24 |
| F2 | `FCC` and `F1192` agree within 0.05 dB in every cell | Within 0.03 on Gaussian and whisper. 0.06, 0.07 and 0.11 apart on Student t | **no**, on Student t |
| F3 | Starting Lloyd's algorithm from the fold changes its error by less than 0.1 dB | Worse by up to 4.5 dB. Within 0.1 in 1 of 18 cases | **no** |

What the misses say:

- **F1.** The formula of the plan left out the cost of the block on the wall. A group of 24 weights is 8 blocks of 3, and the largest of them lies on the surface of the ball, where its error is 5/3 of the mean. The record of the E8 gain measured the same cost for the scalar rule. With 32 blocks to a group or more the formula is within 0.12 dB.
- **F2 and I2b.** On Student t data the direction of a lattice against the axes of the weights matters by about 0.1 dB. The fold's last state is the face-centred cubic lattice turned. Built directly, with its vectors along the axes, it does a little better on heavy tails. The plan took the two to be the same codebook.
- **I3.** See the table of the cells above.
- **F3.** The plan expected Lloyd's algorithm to be indifferent to its start. It is not, and the fold is the worse start.

## Deviations from the plan

- **Amendment 1.** The plan named the wrong model and row lengths for the real matrices. The first run stopped at its own check after measuring the cell along the path and before loading a matrix. Its output is `first-run.out`. The real matrices were then cut to rows that divide by 96. The second run repeats the first run's curve number for number where both have it.
- **C1 and C2 are not blind**, for that reason.
- **A motion named in a comment on the issue was not run.** The comment said the test would also take the source's explicit motion to the cubic end. The two stored paths are the fold the issue names.
- The run took 1,665 seconds on one CPU core at low priority.

## Limits

- **Weight error only**, at 4 bits a weight, for blocks of 3. The fold is in 3 dimensions.
- **Lloyd's algorithm finds a local best.** A fitted codebook that is better than every state of the fold shows that the fold does not reach the best. It does not show what the best is.
- **One way of starting from the fold.** Its end was stretched to reach the longest training block. A stretch to a shorter length would leave fewer points unused. It was not tried, and a sample of the data already costs nothing.
- **The table of a fitted codebook is not counted in its bits.** It is 4,096 points of 3 numbers for a whole family of matrices.
- **Sampling.** G along the path has a standard error of 0.005 to 0.007 dB. Whether the cell improves at every single step cannot be read at that grain.
- **One real model**, a small speech model.

## Read against the closure rule

Nothing is closed by a certificate here. It is a measurement, and it answers the issue.

The fold was the one idea of PR #6 with a mechanism that could be tested, and the mechanism is real: the cell does improve along the path. But the path ends at a lattice that has been known since Kepler, worth a quarter of a decibel, under a ceiling of a third. The records of this date found the gain under this rule in the region of the code and in the length of the block, and the fold moves neither. As a start for a fitted codebook it is worse than a sample of the data.

**What follows.** Issue #9 can be closed as answered. Nothing here calls for a test on a model's output.

One thing was seen that the issue did not ask about. With one scale per row, a fitted codebook over 3 weights is 3.4 to 9.8 dB ahead of the scalar rule and ahead of E8 in a ball over 24 weights on Student t data. With a coarse scale, putting the points where the weights are is worth more than any even spread. That is non-uniform rounding, which is old, and no arm of the model-output runs uses it.

## How to rerun

Needs NumPy, SciPy and torch, the two path files of `Foundation-0/incentre-lattices` in one folder, and the weights file of `mlx-community/whisper-base.en-mlx`. The program refuses files with other SHA-256 sums.

```bash
python3 experiments/fold-path-codebooks-2026-11-17/run.py --paths DIR --weights weights.npz > results.out
```

`run.py --smoke` runs the self-tests and the code path on a small matrix that is not in the study.

## Disclosure

Written by an AI agent session under the owner's direction, on 2026-11-17. The plan, the program, the run and this record are the session's.
