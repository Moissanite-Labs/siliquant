# The formula of issue #10, on matrices no codebook had been run on

## Question

Issue #10 proposes a process: probe a matrix of weights, map it, turn the map into knobs, and pick the knobs with a formula for the error. Four of its claims can be measured on weight error.

1. Does a formula built from the mechanism give the error of a setting of the knobs, from the weights alone, with nothing rounded?
2. Does the setting the formula picks have the smallest measured error?
3. What does the map have to hold?
4. Does dropping directions ever win on real matrices, and does the formula say when?

An extension, asked for by the owner during the run, adds a fifth: which parts of this can be made exact or proved, and what a rule with a proved guarantee costs.

## Status

`complete`. It measures weight error only, on 24 synthetic matrices, 64 matrices of one small speech model and 21 of one small language model. No codebook here has a packed format. Nothing is scored on a model's output.

## Preregistration

[PREREG.md](PREREG.md), committed and pushed at `41e4480` before the run. `run.py` refused to start until that was so, and `results.start` holds the commit and the SHA-256 of the plan and of the four programs. The plan's marks come from a pilot on the 38 matrices the two earlier records of this date had used (`pilot.out`). No codebook had been run on any held-out matrix.

15 of the 16 parts of its ten predictions hold. The one that fails is 8a: a rank arm was predicted never to win on a real matrix, and it wins on two.

[PREREG-extension-1.md](PREREG-extension-1.md), committed and pushed at `8331606`, after the main run had finished and before any of its output was read. 4 of its 5 predictions hold. E4 fails by 0.003 dB on its mark.

`predictions.py` and `safe.py` compute every verdict. None is read off a table by eye.

## Result in short

1. **From 4 bits up the formula gives the error.** On the 85 real matrices the measured error is within 0.3 dB of the formula's in 96.6 percent of 5,525 runs, and 0.085 dB off in the mean. The formula has no fitted constant.
2. **Its pick is the best arm.** Among 13 arms at a fixed group and rate, the arm with the smallest predicted error is the measured best in 96 percent of 425 choices on real matrices. Where it is not, it costs at most 0.04 dB.
3. **At 2 bits the formula is low**, by 0.8 dB in the mean on real matrices and by up to 3 dB. Only 12 percent of the runs are within 0.3 dB.
4. **On real weights the best region is neither the ball nor the cross-polytope.** It is a p-norm ball with p = 5/4 or 3/2 on 64 to 75 of the 85 real matrices, on both models. It is ahead of the better of the two by 0.2 to 0.3 dB in the mean.
5. **The map has to hold the region part itself.** Treating the cloud as round costs 0.4 to 0.7 dB. Knowing only the spread of single weights costs 0.1 to 0.2 dB. On the language model that second description is worse than the first.
6. **Groups of 128 are the best size at 4.5 bits** on 82 of 85 real matrices. The formula picks the measured best group on 84. A group of 32 pays 0.7 to 0.9 dB for its scales, and a row pays 1.0 to 1.2 dB for its largest block.
7. **Dropping directions loses on 83 of 85 real matrices, by 3 to 19 dB, and wins on two by 2 dB.** The formula says which two. In both, an eighth of the directions holds over 99 percent of the squared weights.
8. **The rounding error is not like noise on real matrices,** and less so as the group grows. Its largest singular value is 1.4 to 1.9 times that of noise of the same size on the 85 real matrices, and 1.0 on Gaussian ones.
9. **A rule with a proved guarantee costs about half a decibel.** With a closed-form scale, no block fell outside its codebook in 1,308 runs and no error passed its bound, as the two proofs say. For E8 in a ball the price against the searched rule is 0.40 to 0.46 dB, predicted from the codebook alone to 0.04 dB.
10. **For five arms the pick is exact.** The formula is a fraction times a power of pi, and the pick is whole-number arithmetic with a certificate. A checker that shares no code confirmed all 109.

## How it was measured

| | |
|---|---|
| Codebook | every point of a lattice inside a region, with the largest radius at which it has no more than a cap of points |
| Lattices | the integers Z^8, and E8 |
| Regions | the ball of the p-norm, p = 2/3, 1, 5/4, 3/2, 2, 3. p = 1 is a cross-polytope and p = 2 a ball |
| Rule | the scale of a group is the smallest at which every block of 8 rounds to a point of the codebook, as in the second record |
| X | the scalar largest-entry rule of the harness, at the most levels that fit the cell |
| Rank arms | the 1/8, 1/4 or 1/2 largest directions of the matrix, in two factors rounded by X per row |
| Cells | caps of 2^16 and 2^32 with groups of 32, 128 and a row. And three cells that add up to 4.5 bits per weight with 16 bits per scale: groups of 32 at 2^32, of 128 at 2^35, and a row at what its scale leaves |
| Formula | `shape/shapelib.py`, level 3: cell, region and count, the block on the wall, the cell at zero |
| Data | G, L, T3, U, RS, BS, C, K: 512 by 2048, three seeds each. Blocks 1 to 4 of `whisper-base.en`: 64 matrices. Layers 0, 11 and 23 of `Qwen2.5-0.5B`: 21 matrices |
| Measure | relative squared error of the rounded matrix stored as 16-bit floats, in dB |

The formula, for a lattice with cell constant G, a region of shape K and a codebook of N points:

    error / mean square of the weights  =  G * Phi_K(g) * N^(-1/4)

    Phi_K(g)  =  volume(K)^(1/4) * mean over groups of (largest K-norm of a block in the group)^2 / mean square of the weights

Phi is the region part. It is computed from the weights with nothing rounded. In dB the cell, the region and the rate add.

Residual means measured error over the formula's, in dB. Regret means the measured error of a pick over the smallest measured error among the arms it was picked from.

Under the searched rule a block can still lie outside the codebook at its group's scale. It is then shrunk, as in the earlier records. That happened in 2,081 of 10,464 runs, for at most 2.5 percent of the blocks at 2 bits, 0.54 percent at 4 bits and 0.18 percent in the larger cells of the budget.

Checks. `shapelib` reproduces the eight codebooks of the record `region-shapes-no-clip-rule-2026-11-17` and its errors on a Gaussian matrix to the last digit, and the errors of the scalar rule in `e8-gain-largest-entry-rule-2026-11-17`. A second counter that shares no code with it gives the same count for all 28 codebooks with p = 1 or 2. `check_receipts.py` replayed all 109 receipts from the weights, and caught a changed digit and a changed weight.

All numbers below are in [results.out](results.out), [safe.out](safe.out), [certified.out](certified.out) or [detail.out](detail.out).

## Results

### The formula against the measured error

Measured error over the formula's at level 3, in dB. Mean size, and the share of runs within 0.3 dB.

| Data | At 2 bits: mean | At 4 bits | In the budget cells with caps above 2^32 |
|---|---|---|---|
| whisper, integers | +0.93 | 0.09, 96 % | 0.06, 100 % |
| whisper, E8 | +0.61 | 0.12, 94 % | 0.11, 94 % |
| qwen, integers | +0.98 | 0.09, 95 % | 0.06, 100 % |
| qwen, E8 | +0.72 | 0.05, 100 % | 0.03, 100 % |
| Gaussian, E8 | +0.77 | 0.05, 100 % | 0.03, 100 % |
| Student t, E8 | +0.51 | 0.04, 100 % | 0.04, 100 % |

Where it is off from 4 bits up:

- **The star, p = 2/3.** The formula is low by 0.13 to 0.16 dB in the mean. 63 of the 85 real matrices have some run more than 0.3 dB off. With the star left out, 4 have.
- **A lump at zero, for E8.** Those 4 are the three self-attention matrices that read the stream in decoder block 1 of whisper (key, query, value) and its cross-attention query. Between a half and two thirds of their entries are under a tenth of the root mean square. The formula is high for E8 there by up to 1.6 dB. On the family BS, built to have such a lump, it is high by 0.14 dB.
- **The scalar rule on uniform weights.** The formula is low by 0.75 dB at 16 levels. The rule does clip: the largest entry goes to the lowest level, and a weight within half a step of the other end is pulled back a whole level. That adds 3/q of the error at q levels, which is 0.75, 0.61, 0.56 and 2.43 dB at 16, 20, 22 and 4 levels. Measured: 0.75, 0.61, 0.56 and 2.43 (`detail.out`, table 6, written after the run).

### The pick

Among the 12 cut arms and X, at a fixed group and rate, from 4 bits up.

| Data | Choices | The pick is the measured best | Mean regret | Largest | Rank correlation of predicted and measured error |
|---|---|---|---|---|---|
| whisper | 320 | 307 | 0.000 to 0.001 dB | 0.04 | 0.990 to 0.993 |
| qwen | 105 | 101 | 0.000 to 0.001 dB | 0.01 | 0.992 to 0.998 |
| synthetic | 120 | 108 | 0.000 to 0.24 dB by family and cell | 0.25 | 0.89 to 1.00 |

The synthetic misses are the uniform family, where the formula takes X and E8 with p = 3 is 0.05 to 0.24 dB better with groups of 32 or 128, and one cell of BS.

At 2 bits the pick costs 0.00 to 0.11 dB in the mean on real matrices, and up to 2.1 dB on one. On Student t and uniform data it costs 1.1 to 2.0 dB in four of six cells.

### The shape knob

The measured best arm in the three cells of the budget:

| Model | Group | p = 5/4 or 3/2 is best on | Ahead of the better of E8 in a ball and E8 in a cross-polytope | Ahead of E8 in a ball |
|---|---|---|---|---|
| whisper | 32 | 57 of 64 | 0.29 dB | 0.46 dB |
| whisper | 128 | 49 of 64 | 0.25 | 0.54 |
| whisper | row | 50 of 64 | 0.32 | 0.77 |
| qwen | 32 | 18 of 21 | 0.09 | 0.09 |
| qwen | 128 | 15 of 21 | 0.08 | 0.13 |
| qwen | row | 20 of 21 | 0.16 | 0.30 |

Part of what the inner shapes gain is count and not shape. A codebook with p = 5/4 or 3/2 has 0.9997 or more of its cap, and E8 in a cross-polytope has 0.79 to 0.94 (`results.out`, table 1).

On the synthetic families the best shape is what the family's own geometry says: the ball for Gaussian weights, p = 5/4 for Laplace, the star for Student t with 3 degrees of freedom, and p = 3 or the cube for uniform weights.

### What the map has to hold

Mean regret from 4 bits up, by how the pick is made.

| Model | Formula on the matrix | Formula on a shuffled copy | Always E8 in a ball |
|---|---|---|---|
| whisper | 0.000 to 0.001 dB | 0.04 to 0.10 | 0.46 to 0.77 |
| qwen | 0.000 to 0.001 | 0.25 to 0.41 | 0.09 to 0.31 |

A shuffled copy has the same entries in places drawn at random. It keeps the spread of single weights and loses where they sit.

- On qwen the shuffled pick is worse than always taking the ball. Its rows have their own scales: shuffling raises the region part of the ball per row by 1.75 dB. A shuffled copy looks spikier than the matrix is, and the pick moves to a spikier region.
- The family RS shows the same thing by construction. Each row is Gaussian times its own scale. E8 in a ball is the measured best in 15 of 15 choices, as the proof in the plan says it must be by the formula. The shuffled pick costs 0.71 dB per row.
- In G, L, T3 and U, where the entries are independent, the shuffled pick costs what the formula's pick costs in all 60 choices.

The probe's numbers for each family are in `results.out`, table 8.

### The size knob

At 4.5 bits per weight with the scales, 39 arms: 13 in each of three groups.

| Data | Measured best group | The formula's pick has it | Best arm with groups of 32 is behind by | With a row |
|---|---|---|---|---|
| whisper | 128 on 61 of 64, 32 on 3 | 63 of 64 | 0.74 dB | 0.96 |
| qwen | 128 on 21 of 21 | 21 of 21 | 0.91 | 1.20 |
| Gaussian | 128 on 3 of 3 | 3 of 3 | 1.23 | 0.88 |
| uniform | row on 3 of 3 | 3 of 3 | 2.38 | 0.00 |

The mean regret of the formula's pick among the 39 is 0.0009 dB on the real matrices. For Gaussian weights the limits table gives the same answer from integrals (`bounds/bounds.out`, section 10).

### Kept directions

The rank arms at 4.5 bits, against the best of the 39 full arms.

| Data | Keeps 1/8 | 1/4 | 1/2 | Ahead on |
|---|---|---|---|---|
| whisper | behind by 19.5 dB | 16.0 | 12.2 | 2 of 64, both at 1/4 |
| qwen | 20.0 | 17.6 | 14.0 | 0 of 21 |
| Gaussian | 22.8 | 21.6 | 18.6 | 0 |
| K | ahead by 7.2 | 8.5 | 8.1 | 3 of 3 |

The two are `decoder.blocks.1.attn.key.weight` and `decoder.blocks.1.attn.query.weight` of whisper. The arm that keeps a quarter of the directions is 1.99 and 2.06 dB ahead. In both, an eighth of the directions holds 99.3 and 99.4 percent of the squared weights, and two thirds of the entries are near zero. The formula put the rank arm ahead on the same two and on no other. Its error is within 1.5 dB of the measured one on all 255 rank arms of the real matrices, and within 0.01 dB where the dropped share dominates.

So prediction 8a fails. The knob is live on real matrices, rarely.

By Shannon's rule for a Gaussian source, the spread of the squared weights over the directions of a matrix would be worth 8.5 dB at 4 bits on whisper and 4.9 dB on qwen, in the mean, if the rotation into those directions cost nothing. Dropping does not collect that. A rank arm keeps few directions at many bits. Shannon's rule keeps nearly all, at a number of bits that follows their size: at 4 bits it drops 6 percent of the directions on whisper and 1.5 percent on qwen. No arm here does that, and the rotation would have to be stored.

### The ceiling for a round cloud

On Gaussian matrices from 4 bits up no cut arm is under the floor of a round cloud: the nearest is 0.078 dB above it. E8 in a ball is the measured best in 15 of 15 choices and 0.08 to 0.36 dB above the floor. What is left above E8 in a ball in 8 dimensions is at most 0.08 dB of cell.

### How like noise the rounding error is

The largest singular value of the rounding error over that of a matrix of independent entries of the same total square. Mean over matrices, in the three cells of the budget.

| Data | Arm | Groups of 32 | 128 | Row |
|---|---|---|---|---|
| Gaussian | X | 0.995 | 1.000 | 1.045 |
| Gaussian | E8 in a ball | 1.001 | 0.997 | 1.020 |
| whisper | X | 1.35 | 1.47 | 1.77 |
| whisper | E8 in a ball | 1.26 | 1.33 | 1.59 |
| qwen | X | 1.81 | 1.96 | 2.28 |
| qwen | E8 in a ball | 1.72 | 1.80 | 2.13 |
| RS | X | 2.03 | 2.03 | 2.16 |
| T3 | X | 1.11 | 1.28 | 1.84 |
| T3 | E8 in a ball | 1.69 | 1.86 | 2.82 |

[Reading] The second record found that the largest singular value of the weight change ranked damage to the output best of five measures, and that a finer group helped the output by more than its weight error said. This table gives one reason that would fit. A group's error follows its scale, so the error is gathered where the scales are large, and more so as the group grows. On RS, where every row has its own scale, the ratio is 2. Nothing here tests that reason on an output.

On Student t data E8 in a ball has the more gathered error of the two. On the real matrices it has the less gathered, in every cell.

### The predictions

| # | Prediction | Outcome |
|---|---|---|
| 1 | Real matrices from 4 bits up: mean size of the residual at most 0.12 dB, and 90 percent of runs within 0.3 dB | Holds: 0.085 dB and 96.6 percent. whisper 0.094 and 96.0, qwen 0.059 and 98.5 |
| 2a | The same on the families G, L, T3, U, RS, C, K | Holds: 0.074 dB and 96.4 percent |
| 2b | On BS the formula is high for E8, and the integer arms are within 0.3 dB on 90 percent | Holds: -0.14 dB, and 100 percent |
| 3 | At 2 bits the formula is low: positive mean residual in every family, +0.4 to +1.5 dB on real matrices, under half within 0.3 dB | Holds: +0.79 dB, 12 percent within 0.3, and +0.46 to +1.01 by family |
| 4a | Real matrices from 4 bits up: mean regret at most 0.05 dB, the pick the measured best in 80 percent, rank correlation at least 0.95 | Holds: 0.0003 dB, 96.0 percent, 0.993 |
| 4b | Every synthetic family: mean regret at most 0.10 dB | Holds. The largest is 0.094, on uniform weights |
| 5 | In each cell of the budget the best arm has p = 5/4 or 3/2 on more than half the real matrices, and is 0.05 to 0.40 dB ahead of the better of E8 in a ball and E8 in a cross-polytope | Holds: 75, 64 and 70 of 85, and 0.24, 0.21 and 0.28 dB |
| 6a | Real matrices, every cell from 4 bits up: regret of "round" at least 0.2 dB, and formula under shuffled, which is at most round | Holds for the two models together: 0.000 to 0.001, 0.09 to 0.18, 0.37 to 0.65. On qwen alone the shuffled pick is worse than round in all five cells |
| 6b | G, L, T3, U: the shuffled pick costs within 0.10 dB of the formula's | Holds: the same cost in all 60 |
| 6c | RS: E8 in a ball is the measured best and the formula's pick in every choice, and the shuffled pick costs at least 0.1 dB per row | Holds: 15 of 15, and 0.71 dB |
| 7a | Real matrices at 4.5 bits: the formula's pick has the measured best group on 80 percent, mean regret at most 0.05 dB, and groups of 128 are best on more than half | Holds: 84 of 85, 0.0009 dB, 82 of 85 |
| 7b | Gaussian: E8 in a ball with groups of 128 is the measured best and the formula's pick | Holds: 3 of 3 |
| 8a | Real matrices: no rank arm ahead of the best full arm, measured or by the formula, and the formula within 1.5 dB on 90 percent | **Fails**: 2 of 255 are ahead, by 2 dB, and the formula says so of the same 2. The formula is within 1.5 dB on all 255 |
| 8b | K: the arm that keeps 64 directions is at least 3 dB ahead, and the formula says so | Holds: 7.16 to 7.20 dB, 3 of 3 |
| 9 | Gaussian from 4 bits up: no cut arm more than 0.05 dB under the floor, E8 in a ball best in every choice and at most 0.45 dB above the floor | Holds: nearest +0.078 dB, 15 of 15, at most 0.36 |
| 10 | The spectral ratio rises with the group on real matrices for both arms, E8 in a ball is under X in each, and it is within 0.02 of 1 on Gaussian matrices with groups of 32 and 128 | Holds: 1.47, 1.59, 1.90 and 1.38, 1.44, 1.72. Gaussian: 0.995 to 1.001 |

## Extension 1: what is exact, what is proved, and a rule with a guarantee

Added because the owner asked, during the run, about formulas for the choice of the knobs that can be formally verified. No proof assistant ran in this session: installing one was not permitted. The extension does what needs none. [PREREG-extension-1.md](PREREG-extension-1.md) lists ten statements a proof assistant would cover.

### The pick in exact arithmetic

For the scalar rule and for the integers and E8 cut by a ball or a cross-polytope, the formula at level 2 is a fraction times a power of pi. Its inputs are the whole numbers of the receipt and the exact count of the codebook. So which of two arms has the smaller predicted error is a comparison of whole numbers. `shape/certified.py` makes the pick that way, within the bit budget, and writes a certificate. `shape/check_certificate.py` shares no code with it and recomputes the claim.

| | |
|---|---|
| Certificates, one per held-out matrix: 5 arms at 3 group sizes, 4.5 bits | 109 |
| Confirmed by the checker | 109. A pick moved to another candidate is caught |
| The exact value against the 64-bit floats at level 2 | the same to 7 parts in 10^16 |
| The exact pick is the 64-bit pick at level 2 | 109 of 109 |
| It is the pick of level 3 among the same 15 | 102 of 109 |
| It is the measured best of the 15 | 102 of 109. Mean regret 0.053 dB, largest 1.81 |
| Against the best of all 39 arms | behind by 0.20 dB in the mean. The five arms do not include p = 5/4 and 3/2 |

It differs from the pick of level 3 on the three BS matrices and on the four whisper matrices with a lump at zero: the cell at zero changes the order there, and level 2 does not have it. It is not the measured best on seven: three near ties of 0.01 dB, and those four whisper matrices, by 0.9 to 1.8 dB.

### The safe scale

The rule of the records searches for the smallest scale. The safe scale is a closed form: the largest K-norm of a block of the group over (r - m), with r the radius of the codebook and m the largest K-norm of a vector as long as the lattice's covering radius. Two short proofs in the plan: no block rounds outside the codebook, and the squared error of a block is at most the squared covering radius times the squared scale. The second gives a bound on the error of a matrix that holds for any weights.

| # | Prediction | Outcome |
|---|---|---|
| E1 | In 1,308 runs no block outside the codebook, no block over its bound, no matrix over its bound | Holds. The worst block has 0.9915 of its bound. The bound is 2.48 dB above the measured error for E8 and 4.96 dB for the integers |
| E2 | On real matrices the safe rule's error over the searched rule's is within 0.15 dB of the predicted cost, for each of 12 pairs of codebook and group | Holds, within 0.07. E8 in a ball: 0.40, 0.44 and 0.46 dB against 0.44, 0.46 and 0.48 predicted. The integers in a cross-polytope: 1.15, 1.00 and 0.97 against 1.22, 1.04 and 1.01 |
| E3 | The estimate G s^2, with the cell at zero, is 0.10 dB off in the mean at most and within 0.3 dB in 90 percent of runs on real matrices | Holds: 0.044 dB and 97.6 percent |
| E4 | The arm with the smallest bound is the arm with the smallest estimate, and an E8 arm, on every matrix. On real matrices it is at most 0.05 dB in the mean over the measured best of the 12 | **Fails** on the mark: 109 of 109 and 109 of 109, and 0.053 dB. The pick is the measured best on 79 of 85. On four whisper matrices with a lump at zero the integers in a cross-polytope are best, by up to 1.5 dB |
| E5 | At 4.5 bits the best safe arm is ahead of the best scalar rule on 90 percent of real matrices, by 1.0 to 3.5 dB in the mean | Holds: 85 of 85, by 2.26 dB, from 1.40 to 4.11 |

The best safe arm of a real matrix is 0.5 dB behind the best searched arm with the same four codebooks, and 0.6 to 0.7 dB behind the best of all 39 arms of the main run. The 1,308 runs of the extension took 34 seconds. The main run, whose rule searches for the scale of every block in 14 steps, took 42 minutes for 10,464 runs and its other measurements.

### For any weights

Three inequalities hold for any block of 8 numbers, and so for any matrix. They bound what one region can be worth against another, whatever the weights (`bounds/bounds.out`, section 10): a ball over the cube between -4.53 and +4.50 dB, a cross-polytope over the cube between -6.55 and +11.51, a cross-polytope over a ball between -2.02 and +7.01. The upper end is reached when the block that sets a scale holds one weight. `check_certificate.py` proves the identities behind them by checking every point of a grid, and checks its bounds on pi from Machin's formula in exact fractions.

## What the run shows

1. **The three parts of issue #10 are a cell, a region and a rate, and they add in dB.** The shape and the size of the proposal are one part, not two: the region part holds the shape through the norm and the size through the group. For Gaussian weights the step from groups of 8 to a row costs 4.8 dB with a ball, 5.1 with a cross-polytope and 5.8 with a cube, so the size cannot be priced apart from the shape. The dropped part adds in plain units, not in dB.

2. **The fit to a family is not needed.** The proposal maps a cloud to a tidy family and reads the formula off the family. The region part can be computed on the matrix itself, exactly for three shapes, and it is what the pick needs. A family that treats the weights as independent is worse than no family on the language model.

3. **The formula is a statement about fine cells, and 2 bits is not fine.** The proposal says the formula "can be exact for the tidy families". The region part of a Gaussian cloud is exact, from an integral. The error is not: on Gaussian matrices it is 0.03 to 0.05 dB over the formula for E8 from 4 bits up and 0.77 dB over at 2 bits.

4. **Where the formula was off, it was the formula and not the map.** The proposal says that if measurement and prediction disagree, "the map was wrong". Here the map was exact each time, and the disagreement had a cause in the rule: the wall, the cell at zero, the clip of the scalar rule on its far side, a star's thin arms.

5. **A guarantee is cheap at 4 bits and not at 2.** The safe scale gives up r_eff / (r - m) in scale. At 4.5 bits that is half a decibel for E8 in a ball. At 2 bits the same ratio is 3.7 dB for E8 in a ball and 5.3 dB for the integers, by arithmetic and not run.

## Deviations from the plan

- **The extension.** `shape/certified.py`, `shape/check_certificate.py`, `certify.py`, `safe.py` and the second part of section 10 of the limits table were written while the main run was in progress, after the owner's question. They change no number of the main plan. The extension's plan was committed after the main run finished and before its output was read.
- **`detail.py` was written after the results were read.** None of its six tables was planned. Table 1 names the two matrices of prediction 8a, table 2 splits two predictions by model, table 6 is the arithmetic of the scalar rule's clip.
- **One held-out matrix was loaded before the plan was committed**, to time the receipt and the SVD: `model.layers.11.mlp.gate_proj.weight` of qwen. Three values of its region part with groups of 32 were printed. The plan says so.
- **The pilot was run twice.** The second pass added the spectral ratio and changed no other number.
- **The run took 42 minutes**, about what the pilot's rate gave. The tests of the extension and the limits table ran on the same machine during it. The programs are deterministic, so this changes no number.
- **Prediction 6a is stated for the two models together and holds as stated.** By model it does not hold on qwen. The record reports both.

## Limits

- **Weight error only.** The second record found that weight error does not say what a model's output does. Its three pairs under this rule are the only evidence here that a gain in weight error carries over, and they are one model and one text. The size knob is where this bites: the formula takes groups of 128 on weight error, and the second record's output scores favoured finer groups by more than weight error did.
- **No packed format and no decode cost.** The second hard limit of issue #10 cannot be read here. The codebooks with a row cap do not have a whole number of bits per block: stored one block at a time they need 4.5 bits per weight before the scale, not 4.47 to 4.50 (`results.out`, table 1).
- **Two small models.** 64 matrices of a speech model and 21 of a language model with half a billion weights. The interior shape is weaker on the language model: 0.08 to 0.16 dB over the better of ball and cross-polytope.
- **Six shapes of one family.** For one block per group the best region for a cloud has a closed form. It was not computed.
- **The formula sees weights only.** No activation enters it.
- **Three seeds** per synthetic family.
- **Not proved by a machine.** The extension's proofs are on paper, and its checks are exact arithmetic by a second program. The covering radius of E8 is taken from a book.
- **The margin of the safe scale is a worst case over directions.** A tighter one may exist for the cross-polytope.
- **The exact pick is level 2 and five arms.** Level 3 and the other four shapes are 64-bit floats.

## Read against the closure rule

Three things here are proofs, and they close what they cover.

- **On a round cloud no region beats the ball.** So for Gaussian weights, and for weights turned by a random rotation, the shape knob has a known answer, and E8 in a ball is within 0.08 dB of the best a code in 8 dimensions can do by the formula. What is left there is the size of a block and of a group.
- **For any weights, a region is worth at most 2.5 log10 of the ratio of volumes at equal reach along a weight.** A cross-polytope can never be worth more than 11.51 dB over the cube, nor a ball more than 4.50.
- **At the safe scale nothing is clipped and the error is under its bound.** Measured as well: 1,308 of 1,308.

The rest is measurement. Nothing else is closed. In particular the dropped-directions knob is not closed: it won on two real matrices.

For issue #10: its stage 4 holds on weight error from 4 bits up, with the three parts read as cell, region and rate. Its stage 2 can be built, and was for three shapes. Its claims about a model's output and about decode cost were not tested.

## How to rerun

Needs NumPy, the weights file of `mlx-community/whisper-base.en-mlx` and `model.safetensors` of `Qwen/Qwen2.5-0.5B` at revision `060db64`. No torch.

```bash
python3 -m unittest shape/test_shapelib.py shape/test_certified.py
```

```bash
python3 experiments/shape-formula-held-out-2026-11-17/run.py --set held --whisper weights.npz --qwen model.safetensors
```

```bash
python3 experiments/shape-formula-held-out-2026-11-17/summary.py results > results.out
```

```bash
python3 experiments/shape-formula-held-out-2026-11-17/check_receipts.py results --whisper weights.npz --qwen model.safetensors
```

```bash
python3 experiments/shape-formula-held-out-2026-11-17/safe.py --whisper weights.npz --qwen model.safetensors > safe.out
```

```bash
python3 experiments/shape-formula-held-out-2026-11-17/certify.py results > certified.out
```

```bash
python3 shape/check_certificate.py experiments/shape-formula-held-out-2026-11-17/results.certificates.json
```

`run.py --set held` and `safe.py` refuse to start unless their plan and programs are committed as they stand and pushed, and refuse a weights file whose SHA-256 is not the one in the plan. `run.py --set seen --whisper weights.npz` reruns the pilot. `run.py --smoke` and `safe.py --smoke` run the code paths on small matrices that are not in the study. The main run took 42 minutes on 4 shared CPU cores with NumPy 2.5.3 and Python 3.13, the pilot 8 minutes and the extension 34 seconds.

## Disclosure

The plans, the programs and this record were written by an AI agent session under the owner's direction. A second agent checked the published work cited in `notes/issue10-proposal-check.md`. Nothing here has been reviewed by a person.
