# Plan: the formula of issue #10, on matrices no codebook has been run on

Written 2026-11-17, before the run. The run is `run.py --set held` in this folder at the commit that adds this file.

`run.py` refuses to start unless this file, `run.py`, `summary.py`, `predictions.py` and `shape/shapelib.py` are committed as they stand and that commit is on the remote. It writes the commit and the SHA-256 of each of the five files to `results.start`. The gate is crossing X2 of the note in PR #5.

## Question

Issue #10 proposes a process in four stages. Probe a matrix of weights with numbers computed from the weights alone. Turn the numbers into a short map with a receipt. Turn the map into knobs: the shape of the codebook's region, the size of a group, the directions kept. Pick the knobs with a formula that gives the error as a sum of parts, then run the pick and compare.

Four of its claims can be measured here, on weight error.

1. Does a formula built from the mechanism give the error of a setting of the knobs, from the weights alone, with nothing rounded?
2. Does the setting the formula picks have the smallest measured error, or close to it?
3. What does the map have to hold? Is "the cloud is round" enough, or the spread of single weights, or does it need where the weights sit?
4. Does the third part, the dropped directions, ever win on real matrices, and does the formula say when?

## The formula

Blocks of 8 weights. A codebook is every point of a lattice inside a region. One scale serves a group of g weights: the smallest at which every block of the group rounds to a point of the codebook. This is the rule of the second record, which clips nothing.

For a lattice with cell constant G, a region of shape K whose unit ball has volume vol(K), and a codebook of N points:

    error / mean square of the weights  =  G * Phi_K(g) * N^(-1/4)

    Phi_K(g)  =  vol(K)^(1/4) * mean over groups of (largest K-norm of a block in the group)^2 / mean square of the weights

Why. The points that round into the codebook fill the cells of its N points, a set of volume N times a cell. Take it to be the shape K at the radius of that volume. Then a group's scale is its largest K-norm over that radius, and with fine cells the error of a weight is G times the squared scale times the cell's volume to the power 1/4. The cell's volume cancels.

In dB the three factors add: the cell (G), the region (Phi), the rate (N). The region part is a number computed from the weights with no rounding. It holds both the shape and the size of issue #10: the shape through K and the size through g.

`shape/shapelib.py` computes it at three levels. Level 3 is the formula of this plan. None of the three has a fitted constant.

| Level | What it adds | Why |
|---|---|---|
| 1 | G times Phi times N^(-1/4), with the exact count N | above |
| 2 | The block on the wall. One block of each group has (8 + 2) / 8 of the mean error. For the scalar rule the largest entry lands on a level and has none | found in the record `e8-gain-largest-entry-rule-2026-11-17` |
| 3 | The cell at zero. A weight under half a step (integers, scalar rule), or a block shorter than the packing radius times the scale (E8), rounds to zero. Its error is its own square | the same record: the scalar rule was up to 0.6 dB under its formula where many weights are near zero |

For the rank arms the formula is the share of the squared weights in the dropped directions, plus what the rounding of the two factors adds to the rest. That sum is in plain units and not in dB.

## The knobs and the arms

| Knob | Settings |
|---|---|
| Cell | the integers Z^8, and E8 |
| Shape | the ball of the p-norm, p = 2/3, 1, 5/4, 3/2, 2, 3. p = 1 is a cross-polytope, p = 2 a ball, large p tends to a cube, p under 1 is a star with arms along the axes. And the cube of the scalar largest-entry rule (X) |
| Size | groups of 32 weights, of 128, and one group per row |
| Kept | every direction, or the 1/8, 1/4 or 1/2 largest directions of the matrix in two rounded factors |

So 12 cut codebooks and X at each group, and three rank arms.

A cut codebook has the largest radius at which it has no more than a cap of points. For p = 1 and p = 2 the codebooks are those of the record `region-shapes-no-clip-rule-2026-11-17`. For the other p the size of an entry is a whole number, floor((k/2)^p 2^20), so that every test of membership and every count is integer arithmetic. `shapelib` counts every codebook exactly. Its tests check the counts against a scan of every point in a box for 36 small regions and against that record's eight codebooks.

The cells. A cap is the most codewords a block may have.

| Group | Cap | Bits per weight with the scales | Use |
|---|---|---|---|
| 32, 128, row | 2^16 | 2.5, 2.125, 2.003 to 2.031 | where the formula stops holding |
| 32 | 2^32 | 4.5 | in the budget |
| 128, row | 2^32 | 4.125, 4.003 to 4.031 | the cells of the earlier records |
| 128 | 2^35 | 4.5 | in the budget |
| row of n | floor(2^(36 - 128/n)) | 4.5 | in the budget |

The three cells "in the budget" have the same total: 4.5 bits per weight with 16 bits for each scale. They are set side by side to test the size knob. X has floor(cap^(1/8)) levels: 4, 16, 20 and 22. No arm has more bits than its cell allows.

The rank arms have the most levels that fit 4.5 bits, with one scale per row of each factor.

## Data

**Seen.** The 38 matrices of the two earlier records of this date: Gaussian and Student t with 4 degrees of freedom from seeds 6000 to 6002 and 7000 to 7002, and the 32 matrices of blocks 0 and 5 of `whisper-base.en`. `run.py --set seen` ran every arm of this plan on them. That is the pilot, in `pilot.json` and `pilot.out`. It was read before this plan was written, and the marks below come from it.

**Held out.** No codebook of any kind had been run on any of these when this plan was committed.

| Family | What | Matrices |
|---|---|---|
| G | Gaussian | 3, seeds 8100 to 8102 |
| L | Laplace | 3, seeds 8200 to 8202 |
| T3 | Student t with 3 degrees of freedom | 3, seeds 8300 to 8302 |
| U | uniform | 3, seeds 8400 to 8402 |
| RS | Gaussian, every row times its own scale, the scale log-normal with sigma 0.5 | 3, seeds 8500 to 8502 |
| BS | Gaussian, four columns in ten times 0.05 | 3, seeds 8600 to 8602 |
| C | every weight is 0.8 times its left neighbour plus noise | 3, seeds 8700 to 8702 |
| K | 64 directions hold all but a thousandth of the squared weights | 3, seeds 8800 to 8802 |
| whisper | blocks 1 to 4 of `mlx-community/whisper-base.en-mlx`, the file of the earlier records, SHA-256 `b6c8ee500656e04e8e57c2949a5253f0dda002ba36cd1561846574dbcf01132e` | 64 |
| qwen | layers 0, 11 and 23 of `Qwen/Qwen2.5-0.5B` at revision `060db6499f32faf8b98477b0a26969ef7d8b9987`: the four attention matrices and the three of the feed-forward block. `model.safetensors`, SHA-256 `88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342` | 21 |

The synthetic matrices are 512 by 2048. The whisper matrices have rows of 512 or 2,048 weights, the qwen matrices of 896 or 4,864. Qwen is a language model, and no record of this repository has touched it. It is stored in a 16-bit float type with 8 bits of mantissa, read here exactly.

No pilot was run on any language model, on any of the six new synthetic families, or at any seed of this table.

## What is known before the run

**From the limits table.** A new section 10 of `bounds/bounds.py` computes the region part for Gaussian weights from integrals. It reproduces what the earlier record measured on three matrices: a ball region is worth +0.86, +1.15, +1.44 and +1.91 dB over the scalar rule with groups of 8, 32, 128 and 2,048.

**A ceiling for round clouds.** If the direction of every block is uniform given the lengths of the blocks of its group, no region has a smaller Phi than the ball.

Proof. Write rho(theta) for how far K reaches in direction theta. Then the K-norm of x is its length over rho(theta), and vol(K) is vol(ball) times the mean of rho^8 over directions. Given the lengths, the mean over directions of the largest of length_j^2 / rho(theta_j)^2 is at least the largest of the means, which is the largest squared length times the mean of rho^-2. And the mean of rho^8 to the power 1/4, times the mean of rho^-2, is at least 1, because a power mean of order 8 is at least the power mean of order -2. The two are equal only if rho is constant.

So on a Gaussian matrix the shape knob has a known best setting, and the best a code in 8 dimensions can do by this formula is the ball's region with the ball's cell: 0.08 dB beyond E8 in a ball. The family RS is round in this sense as well, since a row's scale changes lengths and not directions.

**From the pilot**, on the 38 seen matrices. Residual means measured error over the formula's at level 3, in dB.

| | Gaussian | Student t | whisper, blocks 0 and 5 |
|---|---|---|---|
| Residual from 4 bits up, cut arms: mean size | 0.03 to 0.09 | 0.03 to 0.10 | 0.06 to 0.09 |
| The same at 2 bits: mean | +0.77 to +1.13 | +0.64 to +0.65 | +0.65 to +0.91 |
| Regret of the formula's pick among 13 arms, from 4 bits up: mean per cell | 0.000 | 0.000 to 0.033 | 0.000 to 0.015 |
| The pick is the measured best | 15 of 15 | 12 of 15 | 151 of 160 |
| Regret of always taking E8 in a ball | 0.000 | 0.80 to 3.93 | 0.30 to 0.64 |
| Regret of the pick made on a shuffled copy | 0.000 | 0.000 to 0.033 | 0.07 to 0.15 |
| Measured best group at 4.5 bits | 128 on 3 of 3 | 128 on 3 of 3 | 128 on 29 of 32, 32 on 3 |
| The formula's pick has that group | 3 of 3 | 3 of 3 | 32 of 32 |
| Rank arms behind the best full arm | 18.6 to 22.8 dB | 17.6 to 21.8 dB | 1.9 to 21.8 dB |

Three places where the formula was off in the pilot. They are left as they are, and predicted.

- **At 2 bits it is low** by 0.6 to 1.1 dB in the mean and by up to 3 dB. Its pick still cost 0.08 to 0.12 dB in the mean on whisper. On Student t data with groups of 128 its pick cost 2.4 dB.
- **For the star, p = 2/3, it is low** by 0.16 dB in the mean from 4 bits up.
- **For E8 on a matrix with a lump at zero it is high** by 0.8 dB. One matrix of the 32 has 42 percent of its columns near zero.

On the whisper matrices the measured best shape was p = 5/4 or 3/2 on 22 to 27 of the 32 in every cell from 4 bits up, and the best arm was 0.11 to 0.23 dB ahead of the better of E8 in a ball and E8 in a cross-polytope.

The spectral ratio is the largest singular value of the rounding error over what a matrix of independent entries of the same total square would have. On Gaussian matrices it was 0.996 to 1.000 with groups of 32 and 128. On whisper it rose with the group: 1.34, 1.45 and 1.69 for the scalar rule, 1.24, 1.29 and 1.48 for E8 in a ball.

**Seen before writing this plan.** All of `pilot.out`. The pilot was run twice: the second pass added the spectral ratio and changed no other number. The records of the earlier studies. `bounds/bounds.out` with its new section. Two scratch runs on the seen matrices that came before the pilot: the formula against the results the two earlier records had committed, and Phi for eight values of p, which is how the six shapes were chosen. `run.py --smoke --verbose`, which runs every arm and computes every prediction on ten matrices of 32 by 256 drawn from seed 5: one of each family of the table above, and two more under the names of the real models. Those matrices are not in the study. Its lines were read, and two marks were loosened after it: prediction 2b from -0.2 to -0.05 dB, and prediction 6b from 0.05 to 0.10 dB. The spectra of the held-out matrices were not looked at. One held-out matrix, `model.layers.11.mlp.gate_proj.weight` of qwen, was loaded to time the receipt and the SVD. Its largest entry, its standard deviation and three values of Phi with groups of 32 were printed (cube 26.03, ball 18.29, cross-polytope 19.21). Nothing was rounded.

## Measures

- **Residual**: measured error over the formula's at level 3, in dB, for one arm on one matrix in one cell.
- **Regret**: the measured error of a pick over the smallest measured error among the arms it was picked from, in dB. A pick is the arm with the smallest predicted error.
- Three ways of predicting are set side by side. **Formula**: level 3 on the matrix. **Shuffled**: level 3 on a copy with the same entries in places drawn at random, seed 9100. It knows the spread of single weights and nothing of where they sit. **Round**: always E8 in a ball, the best choice if the cloud were round.
- "From 4 bits up" means the cells with caps 2^32, 2^35 and the row cap.

Errors are relative squared errors of the rounded matrix stored as 16-bit floats, as in the earlier records.

## Predictions

`predictions.py` computes every verdict from `results.json`. The numbers there are the marks.

1. **The formula on real matrices.** Over every arm and cell from 4 bits up on the 85 real matrices, the mean size of the residual is at most 0.12 dB and at least 90 percent of the rows are within 0.3 dB.
2. **The formula on synthetic matrices.** (a) The same two marks hold over the families G, L, T3, U, RS, C and K. (b) On BS the formula is high for E8: the mean residual of the E8 arms is at or under -0.05 dB. The integer arms are within 0.3 dB on at least 90 percent of rows.
3. **At 2 bits the formula is low.** The mean residual of the cut arms is positive in every family, between +0.4 and +1.5 dB on the real matrices, and fewer than half the rows there are within 0.3 dB.
4. **The pick at a fixed group and rate, from 4 bits up.** (a) On the real matrices the mean regret is at most 0.05 dB, the pick is the measured best arm in at least 80 percent of the choices, and the mean rank correlation between predicted and measured error over the 13 arms is at least 0.95. (b) In every synthetic family the mean regret is at most 0.10 dB.
5. **The best shape on real weights lies between the cross-polytope and the ball.** In each of the three cells of the budget the measured best arm has p = 5/4 or 3/2 on more than half of the 85 real matrices, and it is ahead of the better of E8 in a ball and E8 in a cross-polytope by 0.05 to 0.40 dB in the mean.
6. **What the map has to hold.** (a) On the real matrices, in every cell from 4 bits up, the mean regret of "round" is at least 0.2 dB, and the mean regret of "formula" is under that of "shuffled", which is at most that of "round". (b) In G, L, T3 and U, where the entries are independent, the shuffled pick costs within 0.10 dB of the formula's pick in every choice. (c) In RS, E8 in a ball is the measured best arm and the formula's pick in every choice from 4 bits up, and per row the shuffled pick costs at least 0.1 dB in the mean.
7. **The size knob.** (a) Among the 39 arms of the budget the formula's pick has the measured best group on at least 80 percent of the real matrices, its mean regret is at most 0.05 dB, and groups of 128 are the measured best on more than half. (b) On the three Gaussian matrices E8 in a ball with groups of 128 is both the measured best and the formula's pick.
8. **Kept directions.** (a) On the real matrices no rank arm is ahead of the best full arm at 4.5 bits, measured or by the formula, and the measured error of a rank arm is within 1.5 dB of the formula's on at least 90 percent of them. (b) On K the arm that keeps 64 directions beats the best full arm by at least 3 dB on each matrix, and the formula puts it ahead on each.
9. **The ceiling for a round cloud.** On the Gaussian matrices from 4 bits up, no cut arm has a measured error more than 0.05 dB under the floor of a round cloud (the ball's cell, the ball's region from its integral, the arm's own count). E8 in a ball is the measured best arm in every choice and at most 0.45 dB above that floor.
10. **How like noise the rounding error is.** In the three cells of the budget, on the real matrices, the mean spectral ratio rises with the group for the scalar rule and for E8 in a ball, and E8 in a ball is under the scalar rule in each. On the Gaussian matrices with groups of 32 and 128 the mean is within 0.02 of 1 for both.

If 1 and 4 hold, the formula can stand in for a run when a codebook is chosen from this family at 4 bits and above, on weight error. If 3 holds, it cannot at 2 bits as it stands. If 6a holds, the map of issue #10 has to carry the region part itself and not a description of the cloud as round or as a spread of single weights. If 5 fails on qwen, the interior shape is a property of one small speech model.

Reported without a prediction: every arm's gain over X in every cell, levels 1 and 2 of the formula, the probe's numbers for every family (how spiky, how stretched, whether the entries' places matter, the lump at zero), what Shannon's rule would gain from the spread of the mass over directions, and everything by model.

## The receipt

For each matrix `run.py` records the SHA-256 of its bytes and, for each group, the whole numbers from which Phi for the cube, the ball and the cross-polytope follow exactly (`shapelib.exact_terms`). The weights are floats, so each is a whole number times a power of two, and the sums are whole-number arithmetic. `check_receipts.py` replays every receipt from the weights and exits with 1 on any difference. It also changes one digit of a receipt and one entry of a matrix and checks that both are caught.

This is a check and not a prediction. For the other three shapes the comparisons are exact and the sums are 64-bit floats. The singular values are floats.

## What this run cannot show

- Anything about a model's output. The second record found that weight error does not say what the output does. Its three pairs under this rule are the only evidence that a gain in weight error carries over, and they are one model and one text.
- Decode cost, the second limit of issue #10. No codebook here has a packed format, and none is timed. The codebooks with a row cap do not have a whole number of bits per block.
- That the six shapes hold the best region. For one block per group the best region for any cloud has a closed form, which is not computed here.
- Anything about activations. The formula sees weights only.
- Whether the weights of a large language model behave as these do. The second model is small.
