# A ball over more than 8 weights, under the rule that clips nothing

## Question

Two records of this date found that under the rule that clips nothing most of E8's gain over scalar rounding is the region of its code, and that integer levels cut by a ball over 8 weights get most of it with no lattice. `bounds/bounds.out`, section 10, says a ball over more weights is worth more. No such codebook had been run.

1. Does the gain grow with the number of weights under one ball, as the formula says, on Gaussian and on real weights?
2. With one ball for a whole group of 32 or 128 weights, does it beat the E8 Voronoi code of the second record at no more bits?
3. Does the cross-polytope keep its lead on heavy tails as the blocks grow?

## Status

`complete`. It measures weight error only: on 6 synthetic matrices, the 32 matrices of one small speech model and 21 matrices of one language model. The codebooks have no packed format. Nothing is scored on a model's output.

## Preregistration

[PREREG.md](PREREG.md), committed and pushed at `5398096` before the run. `results.start` holds the start time and that commit. All 8 predictions hold.

## Result in short

1. **The gain grows with the block, and the formula gives it.** On Gaussian data per row, integer levels in a ball gain 1.83 dB over the scalar rule with blocks of 8 and 4.18 dB with blocks of 128. The formula says 1.90 and 4.17, and is within 0.07 dB at all five block sizes.
2. **E8 with one ball over a long block beats the E8 Voronoi code on every real matrix, in every cell.** At no more bits, in the mean: by 1.2 to 1.3 dB with groups of 32, by 2.5 to 2.9 dB with groups of 128, and by 3.6 to 3.7 dB per row. That is 53 of 53 real matrices in each of four cells.
3. **Integer levels alone, with no lattice, beat it too.** In a ball over 128 weights they are 3.1 dB ahead of the E8 Voronoi code per row on both models, and ahead on every real matrix in every cell.
4. **With one ball to a group the error does not depend on the weights.** E8 in one ball per group of 32 at 16 levels gives -23.54 dB on Gaussian data, -23.54 on Student t data, -23.57 on whisper and -23.54 on qwen. The scale divides each group by its own length, so the error is G (n + 2) over the bound of the codebook, known before anything is rounded.
5. **At 4.125 bits it is ahead of the best arm of the second record at 4.5 bits.** E8 in one ball per group of 128 gives -23.3 dB on every family at 4.125 bits per weight. The E8 Voronoi code with groups of 32 gives -22.3 dB on the real matrices at 4.5 bits, and the scalar rule of `Q4_0` gives -20.5.
6. **On long blocks the ball and the cross-polytope are close on real weights.** The cross-polytope is ahead by 0.2 dB per row and within 0.15 dB either way with groups. On Student t data it stays 0.5 to 1.8 dB ahead.

## How it was measured

| | |
|---|---|
| Block | n = 8, 16, 32, 64 or 128 weights along a row |
| ZB, ZC | every point of Z^n whose squared length (ball) or sum of sizes (cross-polytope) is at most a bound |
| EB, EC | the same for E8 repeated n/8 times |
| Bound | the largest at which the codebook has no more than q^n points, counted exactly |
| Rounding | each weight to an integer, or each 8 weights to E8. Only the test of whether a block is in the codebook looks at all n weights |
| Rule | the scale of a group is the smallest at which every block of the group rounds to a point of the codebook |
| X, VE8 | the scalar rule and the E8 Voronoi code of the second record, run with the harness in the same cells |
| Bits | log2 of the number of codewords over n, and 16 per group. No arm has more bits than X |
| Cells | groups of 32 and of 128 at q = 16, one group per row at q = 16 and 32. A block has to divide its group |
| Data | G and T: 512 by 2048, three seeds each. whisper: the 32 matrices of the first record. qwen: 21 matrices of `Qwen2.5-1.5B-Instruct`, layers 0, 13 and 27, cut to their first rows, up to 393,216 weights each |
| Measure | relative squared error of the rounded matrix stored as 16-bit floats, in dB |

Checks, all in `results.out`:

- Over 8 weights the program gets the codebooks of the record region-shapes-no-clip-rule-2026-11-17, and on the 456 runs the two share the errors are equal to the last digit.
- Over 16 weights the count of a codebook equals a sum over pairs of blocks of 8, for 20 small codebooks.
- No cut codebook has more bits than the scalar rule in its cell. At q = 16 they have 3.9588 to 4.0000 bits per weight, and from n = 32 up at least 3.9907.
- A block lay outside its codebook at its group's scale in 339 of 4,248 runs, all with EC, for at most 0.1 percent of the blocks. Those blocks were shrunk as the plan says.

All numbers below are in [results.out](results.out) or [detail.out](detail.out).

## Results

### Gain over the scalar rule, in dB

Means over the matrices of a family. "Longest" is n = 32 with groups of 32 and n = 128 otherwise.

| Data | Group | q | VE8 | EB, n = 8 | EB, longest | ZB, longest | EC, longest |
|---|---|---|---|---|---|---|---|
| G | 32 | 16 | +1.36 | +1.44 | +2.22 | +1.56 | +1.76 |
| G | 128 | 16 | +1.85 | +1.97 | +3.59 | +2.93 | +3.15 |
| G | row | 16 | +2.29 | +2.49 | +4.84 | +4.18 | +4.34 |
| G | row | 32 | +2.32 | +2.54 | +4.86 | +4.20 | +4.37 |
| T | 32 | 16 | +2.89 | +2.95 | +4.80 | +4.14 | +5.33 |
| T | 128 | 16 | +3.75 | +3.79 | +7.88 | +7.22 | +8.48 |
| T | row | 16 | +4.61 | +4.34 | +11.30 | +10.66 | +13.05 |
| T | row | 32 | +5.02 | +4.72 | +11.70 | +11.05 | +13.46 |
| whisper | 32 | 16 | +1.77 | +1.90 | +3.08 | +2.50 | +3.08 |
| whisper | 128 | 16 | +2.35 | +2.52 | +5.20 | +4.64 | +5.34 |
| whisper | row | 16 | +2.71 | +2.91 | +6.33 | +5.77 | +6.56 |
| whisper | row | 32 | +2.92 | +3.10 | +6.53 | +5.96 | +6.78 |
| qwen | 32 | 16 | +1.80 | +1.86 | +2.97 | +2.31 | +2.86 |
| qwen | 128 | 16 | +2.25 | +2.34 | +4.70 | +4.04 | +4.64 |
| qwen | row | 16 | +3.08 | +3.17 | +6.83 | +6.17 | +7.07 |
| qwen | row | 32 | +3.26 | +3.36 | +6.99 | +6.34 | +7.25 |

`results.out`, table 2, has every block size.

### The block size, against the formula

Integer levels in a ball, one group per row at q = 16, Gaussian data. The formula is the ball column of the plan, from `bounds/bounds.out` section 10 on other draws.

| n | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| Measured gain over X | +1.83 | +2.58 | +3.24 | +3.78 | +4.18 |
| Formula | +1.90 | +2.62 | +3.25 | +3.78 | +4.17 |

E8 adds 0.63 to 0.66 dB to each of these. Its cell is worth 0.65.

### Against the E8 Voronoi code on the real matrices

E8 and the integers at the longest block, against VE8 in the same cell, in dB.

| Data | Group | q | EB over VE8: mean (min, max) | Ahead on | ZB over VE8: mean (min, max) | Ahead on |
|---|---|---|---|---|---|---|
| whisper | 32 | 16 | +1.30 (+0.84, +2.65) | 32 of 32 | +0.73 (+0.20, +3.47) | 32 |
| whisper | 128 | 16 | +2.85 (+1.99, +5.96) | 32 | +2.29 (+1.35, +5.72) | 32 |
| whisper | row | 16 | +3.62 (+2.41, +6.52) | 32 | +3.06 (+1.78, +6.31) | 32 |
| whisper | row | 32 | +3.61 (+2.40, +6.48) | 32 | +3.04 (+1.75, +6.20) | 32 |
| qwen | 32 | 16 | +1.17 (+0.80, +2.88) | 21 of 21 | +0.51 (+0.13, +2.22) | 21 |
| qwen | 128 | 16 | +2.45 (+1.59, +5.38) | 21 | +1.79 (+0.92, +4.72) | 21 |
| qwen | row | 16 | +3.74 (+1.78, +8.14) | 21 | +3.09 (+1.13, +7.46) | 21 |
| qwen | row | 32 | +3.73 (+1.76, +8.08) | 21 | +3.08 (+1.11, +7.41) | 21 |

### The error in dB, by the bits of the cell

The best block size of each arm. Relative squared error in dB.

| Data | Group | q | Bits | X | VE8 | ZB | EB | EC |
|---|---|---|---|---|---|---|---|---|
| whisper | 32 | 16 | 4.500 | -20.49 | -22.27 | -23.00 | -23.57 | -23.57 |
| whisper | 128 | 16 | 4.125 | -18.15 | -20.50 | -22.79 | -23.35 | -23.49 |
| whisper | row | 16 | 4.028 | -15.86 | -18.57 | -21.63 | -22.19 | -22.43 |
| whisper | row | 32 | 5.028 | -21.70 | -24.62 | -27.65 | -28.23 | -28.47 |
| qwen | 32 | 16 | 4.500 | -20.57 | -22.37 | -22.88 | -23.54 | -23.43 |
| qwen | 128 | 16 | 4.125 | -18.61 | -20.87 | -22.66 | -23.31 | -23.25 |
| qwen | row | 16 | 4.009 | -14.80 | -17.89 | -20.97 | -21.63 | -21.87 |
| qwen | row | 32 | 5.009 | -20.68 | -23.94 | -27.01 | -27.67 | -27.93 |

Two readings across rows of this table:

- E8 in one ball per group of 128, at 4.125 bits, is 1.0 dB ahead of the E8 Voronoi code with groups of 32 at 4.5 bits, and 2.8 dB ahead of the scalar rule there, which is the rule of `Q4_0`.
- E8 in a ball of 128 per row, at 4.01 to 4.03 bits, is 0.1 to 0.7 dB behind the E8 Voronoi code with groups of 32 at 4.5 bits.

### One ball to a group

When a block is a whole group, the scale divides the group by its own length. The block then lies on the surface of the ball, and its squared error is (n + 2)/n times the mean of the cell. So the relative error of every group is G (n + 2) over the bound of the codebook, whatever the weights. For E8 with n = 32 at q = 16 that is 0.071682 x 34 / 552, which is -23.55 dB.

| Data | EB, n = 32, groups of 32 | EB, n = 128, groups of 128 | ZB, n = 128, groups of 128 | X, groups of 32 | X, groups of 128 |
|---|---|---|---|---|---|
| G | -23.54 | -23.32 | -22.66 | -21.32 | -19.73 |
| T | -23.54 | -23.31 | -22.66 | -18.74 | -15.44 |
| whisper | -23.57 (-24.16 to -23.53) | -23.35 (-24.11 to -23.30) | -22.79 (-24.96 to -22.65) | -20.49 (-21.44 to -19.10) | -18.15 (-19.24 to -15.39) |
| qwen | -23.54 (-23.55 to -23.53) | -23.31 (-23.33 to -23.31) | -22.66 (-22.67 to -22.64) | -20.57 (-21.54 to -16.85) | -18.61 (-20.48 to -14.01) |

At q = 16, in dB, with the range over the matrices of the real families. This table was not planned.

This is the old split of a block into a gain and a shape, with the shape rounded to a lattice inside a ball. What the rule that clips nothing adds is that the gain is not a free choice: it is the smallest that fits.

### Ball against cross-polytope

EC over EB at the longest block, in dB (`detail.out`, table 3, not planned).

| Data | Groups of 32 | Groups of 128 | Row, q = 16 | Row, q = 32 |
|---|---|---|---|---|
| whisper | +0.01, ahead on 9 of 32 | +0.13, on 10 | +0.23, on 16 | +0.24, on 16 |
| qwen | -0.11, ahead on 4 of 21 | -0.06, on 4 | +0.25, on 6 | +0.26, on 6 |

On Student t data the cross-polytope is ahead by 0.5 to 0.6 dB with groups and by 1.7 to 1.8 dB per row. With blocks of 8 the earlier record had it ahead by 0.7 to 3.2 dB there: a long block makes the ball less exposed to one large weight.

### The predictions

| # | Prediction | Outcome |
|---|---|---|
| 1 | Over 8 weights the four arms give the errors of the region study to 6 digits | Holds: 456 runs, no difference |
| 2 | On G the mean gain of ZB and of EB rises with n in every cell. Per row at q = 16 by at least 0.5 dB from 8 to 16 and at least 1.6 dB from 8 to 128 | Holds: +0.75 and +2.35 for ZB, +0.72 and +2.35 for EB |
| 3 | On G per row at q = 16, ZB is within 0.3 dB of the formula at every n | Holds: within 0.07 |
| 4 | On G with one ball per group, EB beats VE8 on all three matrices, by at least 0.6 dB with groups of 32 and 1.3 dB with groups of 128 | Holds: +0.85 and +1.73 |
| 5 | On T, EC beats EB in the mean at every n in every cell | Holds |
| 6 | On whisper and qwen, EB at the longest block beats EB at n = 8 in the mean and on at least three quarters of the matrices, in every cell | Holds: +1.1 to +3.7 dB, on every matrix |
| 7 | On whisper and qwen, EB at the longest block beats VE8 in the mean by at least 0.5 dB in every cell | Holds: +1.17 to +3.74 |
| 8 | On whisper and qwen, per row at q = 16, ZB at n = 128 beats VE8 in the mean | Holds: +3.06 and +3.09 |

## Deviations from the plan

- **`detail.py` was written after the run.** Its table 1 gives the counts that predictions 6 to 8 need. Tables 2 and 3 were not planned.
- No other deviation. The run took 261 seconds on 4 CPU cores.

## Limits

- **Weight error only.** The second record found that weight error does not say what a model's output does. It also found that finer groups helped the output by more than their weight error did. The arms that do best here use groups of 128 or a row. So this is where an output score is most likely to disagree.
- **No packed format.** A codebook over 128 weights at 4 bits has up to 2^512 points. To store a block its point has to be numbered, and the cost of that grows with the block. Nothing here measures it.
- **Published work.** Rounding a block to integer levels inside a ball or a cross-polytope, with a separate gain, is old: pyramid vector quantisation, and shell mapping and sphere shaping in transmission. For weights of language models a pyramid quantiser is published (arXiv 2410.16926), and a Leech-lattice code with a ball over 24 weights (arXiv 2603.11021). No search was made for this rule with these blocks, so nothing here is claimed as new.
- **Two real models**, one of them the model of issue #3, each a sample of its matrices. The qwen matrices are cut to their first rows.
- **Three seeds** per synthetic family.
- **Two shapes.** The record for issue #10 found a p-norm ball with p between 1 and 2 best over 8 weights. Over long blocks only p = 1 and p = 2 were run.
- **4 and 5 bits only.**

## Read against the closure rule

Nothing is closed by this run. It is a measurement.

It answers issue #4 more fully than the record of the region shapes did. E8 as the second record used it is beaten in weight error at no more bits by E8 with a ball over a long block, on every real matrix tried, and by integer levels with no lattice. The cell of a code in 8 dimensions is capped at 0.73 dB by the limits table. The region is not capped that way: on Gaussian weights it is worth 4.2 dB at 128 weights per row, by a formula that this run confirms to 0.07 dB.

## How to rerun

Needs NumPy, torch, llama.cpp's `gguf` package, the weights file of `mlx-community/whisper-base.en-mlx` and the 16-bit float GGUF file of `Qwen/Qwen2.5-1.5B-Instruct-GGUF`. The program refuses files with other SHA-256 sums.

```bash
python3 experiments/ball-dimension-no-clip-rule-2026-11-17/run.py --weights weights.npz --gguf qwen2.5-1.5b-instruct-fp16.gguf > results.out
```

```bash
python3 experiments/ball-dimension-no-clip-rule-2026-11-17/detail.py > detail.out
```

`run.py --smoke` runs the code path on a small matrix that is not in the study.

## Disclosure

The plan, the programs and this record were written by an AI agent session under the owner's direction.
