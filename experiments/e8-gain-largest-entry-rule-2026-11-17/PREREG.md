# Plan: why E8 gains more than its cell is worth under the largest-entry rule

Written 2026-11-17, before the run. The run is `run.py` in this folder at the commit that adds this file.

## Question

The second record rounded gemma with one scale per group of weights, set so that nothing is clipped. Under that rule E8 beat scalar rounding at equal bits by 1.51 dB of weight error with groups of 32, and by 2.52 and 2.55 dB with one group per row. The shape of the E8 cell is worth 0.65 dB. The record gave a possible reason and did not test it.

Issue #4 asks for the reason, and asks whether a Foundation-0 lattice family beats E8 at those bits.

1. Does the same rule give a gain of that size on Gaussian data, with no model?
2. How much of the gain is the cell, and how much is the scale the rule can use?
3. Does the point set F_8 of PR #6 beat scalar rounding or E8 under this rule?

## The rule, and a way to split the gain

Both arms store log2(q) bits per weight and 16 bits per group.

- **X**, scalar. The step of a group is its largest entry over q/2. This is `tq.maxscale_round`, llama.cpp's `Q4_0` rule.
- **VE8**, the E8 Voronoi code. The scale of a group is the smallest at which every block of 8 lies inside the code's region. This is `tq.maxscale_voronoi`.
- **VZ8**, the same code on the cubic lattice, as a control. Its cells and its region are cubes.

When the cells are fine and nothing is clipped, the error of a group is G times its squared scale, with G = 1/12 for the cube and 0.071682 for E8, both lattices having volume 1. So

    gain of VE8 over X  =  10 log10(G_cube / G_E8)  +  10 log10(sum of d^2 / sum of t^2)

where d is the scalar step of a group and t its E8 scale. The first term is the shape, 0.654 dB. The second is the scale. The run measures the gain and the scale term, and reports what is left over.

`tq.maxscale_voronoi` does not return its scales. `run.py` repeats its bisection to get them and checks, on every matrix and cell, that rounding at those scales gives the same matrix the harness function returns. The harness is not changed.

## What is known before the run

- The E8 cell has its nearest wall at 0.707 and its farthest corner at 1.0, in units where its volume is 1. A ball of that volume has radius 0.839. A cube of that volume reaches 0.5 along a weight axis and 1.414 along its diagonal. The E8 cell reaches 1.0 along a weight axis: (1, 0, ..., 0) is a deep hole of E8.
- So the scalar rule needs room for the largest single entry of a group, and the E8 rule needs room for the longest block, roughly.
- **A hand estimate, made before any run.** For Gaussian weights the largest of 32 entries is about 2.4 standard deviations, and the longest of 4 blocks of 8 is about 3.55. That gives a scale term near 1.1 dB and a gain near 1.75 dB for groups of 32. For rows of 2,048 it gives a scale term near 2.0 dB and a gain near 2.6 dB. The estimate treats the E8 region as a ball. The block that sets the scale will tend to point where the region is thin, so the true figures should be lower.
- For F_8 with fine cells: its cell is worth -1.69 dB against the cube at the incentre and -1.48 dB at the centroid (the plan of `incentre-vortex-codebooks-2026-11-17`, a derivation). Its region is a cube, so the rule gives it the scalar step.

**Seen before writing this plan.** `run.py --smoke` on a 32 by 256 matrix of uniform numbers that is not in the study. `harness/test_tq.py`, which prints one figure: E8 beats scalar rounding at 2 bits on a Gaussian matrix by 0.88 dB with one scale per matrix. No arm has been run on a matrix of the study. The first study of this date had not been run either when this plan was committed.

## Arms and cells

| Group | q | Bits | In the second record |
|---|---|---|---|
| 32 | 16 | 4.5 | yes: 1.51 dB |
| row | 16 | 4 + 16 over the row length | yes: 2.52 dB |
| row | 32 | 5 + 16 over the row length | yes: 2.55 dB |
| 32 | 32 | 5.5 | no |
| 8 | 16 | 6 | no |
| 128 | 16 | 4.125 | no |

**F_8 under the same rule.** Half a cell is the group's largest entry over K, so nothing is clipped. Bits are log2(K) + 0.5 and 16 per group. K is chosen so that F_8 has slightly more bits than the references it is set against:

| Group | K | F_8 bits | Against X and VE8 at |
|---|---|---|---|
| row | 12 | 4.085 + 16 over the row length | q = 16, 4 + 16 over the row length |
| 32 | 12 | 4.585 | q = 16, 4.5 |
| row | 23 | 5.024 + 16 over the row length | q = 32, 5 + 16 over the row length |

It runs with t = 2 - sqrt 2, as proposed, and with t = 8/9, the centroid.

## Data

- **G.** Independent Gaussian entries, 512 by 2048, three seeds, as 32-bit floats.
- **T.** Independent Student t entries with 4 degrees of freedom, 512 by 2048, three seeds.
- **Real.** The 32 matrices of the first record, whole, from the same file (SHA-256 `b6c8ee500656e04e8e57c2949a5253f0dda002ba36cd1561846574dbcf01132e`). Their rows are 512 or 2,048 long. One small speech model.

The gemma file is not on this machine. Nothing here is run on it.

## Measure

Relative squared error of the matrix the harness returns, which it stores as 16-bit floats, as in the second record. Gains are in dB against X on the same matrix at the same q.

For the block that sets each group's E8 scale the run also records its reach, its length over q times the scale, and its peak, its largest entry over its length.

## Predictions

1. **Size on Gaussian data.** The gain of VE8 over X is between 1.0 and 2.2 dB with groups of 32 at q = 16, and between 1.7 and 3.2 dB with one group per row at q = 16 and at q = 32, on each of the three matrices.
2. **The split.** On G, in the four cells with groups of 32 or a row, shape plus scale is within 0.3 dB of the measured gain on every matrix.
3. **The control.** On G, in the same four cells, VZ8 is within 0.3 dB of X.
4. **Heavy tails.** In those four cells the mean gain on T is larger than on G.
5. **Real matrices.** The mean gain over the 32 is between 1.0 and 3.0 dB with groups of 32 at q = 16, and between 1.5 and 4.0 dB with one group per row at q = 16. Shape plus scale is within 0.5 dB of the measured gain on at least 24 of the 32 in both cells.
6. **F_8.** In all three cells and on all three families, F_8 has a larger error than X in the mean, with both values of t, though it has more bits. It has a larger error than VE8 on every matrix.

If prediction 1 fails low, the gain on gemma is not a property of the rule and the lattice alone, and the weights of that model have a part in it. If prediction 2 fails, the split is not a description of the gain and the fine-cell formula is the first thing to check. Table 2 of the output reports how well it holds for each arm.

Reported without a prediction: groups of 8 and of 128, the reach and the peak, and the scale a ball of the same volume would need.

## What this run cannot show

- That the split found here is the split on gemma. It shows what the rule and the lattice do on other weights. The same program can be pointed at the gemma matrices by whoever holds the file.
- Anything about a model's output. In the second record the gain in KL was 1.17 to 1.29 times the gain in weight error for these two arms, on one model and one text.
- Whether a scalar rule with a searched scale or an offset per group closes the gap. The second record lists that as untried, and it stays untried.
- Whether a lattice in more dimensions gains more. No decoder for one is in `lattice/`.
