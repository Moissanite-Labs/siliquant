# Issue #10 after the model-output runs

Date: 2026-11-17
Status: one test is still running. Six predictions for a third model were committed before its ball arms were scored. Their verdicts are in the last section but one, and are filled in as the rows land.

[issue10-proposal-check.md](issue10-proposal-check.md) read the proposal of issue #10 when nothing of its family had been scored on a model. It left the proposal's claim about a model's output untested, and its second hard limit, decode cost, unreadable. Five things have landed since: a run on `Qwen2.5-1.5B-Instruct` with 27 scored variants, a study of balls over more than 8 weights, a way to number the points of a ball-cut codebook, the fold study of issue #9, and the first half of an importance-matrix baseline. This note reads the open rows of the check against them and carries out stage 4 of the proposal with what they give.

**Measured** means a figure from a committed output of the record named. **Derived** means arithmetic from the formula. **Predicted** means stated here before the data existed.

## Result

- **For the arm that wins, the proposal's first two stages are not needed.** With one ball over a whole group the region part of the formula reads no weight. The formula is a closed form: the cell, the group's size, the bits, and what the scale adds. On `Qwen2.5-1.5B-Instruct` it gives the measured weight error of all eight arms it covers to 0.02 dB, on 196 matrices and 1.31 billion weights.
- **On that model the output follows it.** One scored arm gives the model's KL per unit of weight error. With it the closed form gives the mean KL of the eight arms within 0.27 dB. Four more arms with one scale per group of 32 or 128 and a codeword at zero follow the same constant within 0.18 dB at their measured weight error. Every one of those rows had been seen, so this is a fit and not a test.
- **It does not follow it with one scale per row, or without a codeword at zero.** Those arms have 0.8 to 4.3 dB and 1.2 to 15.5 dB more KL than the constant gives. The formula does not see either.
- **A test before the data is under way.** Six predictions for `SmolLM2-1.7B-Instruct` were committed and pushed at `49f7a73`, when only its scalar rule and one of llama.cpp's files had a score.
- **Stage 4 can now be carried out under both of its limits.** `shape/design.out` lists 36 settings of the knobs with the error by the formula, the KL it implies on two models, the size of the table that numbers the codebook's points, and the look-ups to read a group back. 30 of the 36 have not been scored on any model.
- **The shape knob is closed in practice for the language model.** Once a region spans the group, the best p-norm region is worth 0.05 dB over the ball in the mean on 21 matrices, by the formula. On the output a cross-polytope and a ball over 32 weights score the same: 0.0341 and 0.0339.
- **By the formula little is left inside this family.** The scored arm at 4.5 bits is 0.25 dB from the same group with a shorter scale, 0.6 dB from what E8 reaches with any group, and 1.5 dB from Shannon's floor for Gaussian weights. No lattice can add more than 0.88 dB to E8.
- **A proved bound carries over.** At the safe scale no group of any matrix has an error over a fixed share of its own squared weights: -22.6 dB for E8 in one ball per group of 128 at 4.5 bits. It costs 0.55 dB against the searched scale.

## Rows of the check that move

| Row | Then | Now | Basis |
|---|---|---|---|
| 4, 7, 10: the probe, the map, the fit to a family | The map has to hold the region part of the matrix | With one ball per group there is nothing to map: the region part is the group's size times a power of the volume of a ball. A probe is needed only for an arm whose scale is shared by several blocks | derived, and measured in `ball-dimension-no-clip-rule-2026-11-17`: -23.54, -23.54, -23.57 and -23.54 dB on four families |
| 11: the shape knob | p = 5/4 or 3/2 is best on real weights, by 0.2 to 0.3 dB, with blocks of 8 | That gain is of blocks of 8 under a scale shared by a group. One ball per group of 32 is 1.1 dB ahead of a ball over 8 weights with the same scale, -23.55 against -22.43 dB on the Qwen matrices, four times what the best shape over 8 weights adds. And with one region per group the best p is worth 0.05 dB on qwen and 0.3 dB on whisper in the mean, the second carried by four matrices with a lump at zero | measured in the Qwen run. Derived in `shape/group_shape.out` |
| 12: the size knob | Groups of 128 on weight error. "The knob most likely to be set wrongly for an output" | Two knobs: the weights under one ball, and the bits of the scale. For one ball per group the group costs the output nothing: 128 weights at 4.125 bits score as 32 weights at 4.5. The caution holds for rules that set the scale from the largest entry, where groups of 128 raise the KL by more than half | measured in the Qwen run |
| 17: the decode budget | Cannot be read: no packed format | Readable in part. `shape/ballindex.py` numbers the points of a ball-cut codebook. A table's size and the look-ups per weight are known for every arm. No kernel computes from it, and nothing is timed | measured in `shape/ballindex.out` and `shape/design.out` |
| 19, 23: a knob's effect on the output | Not tested | On one model, for arms with one scale per group and a codeword at zero: a decibel of weight error is a decibel of KL, within 0.3 dB over twelve arms. A second model is being scored against predictions | measured in the Qwen run, after the fact. Predicted for the third model |
| 20: a bound for any weights | The safe scale, over blocks of 8 | Carried to one ball per group. Queued for the output as `SEB8@32:16` in the Qwen run, over 8 weights, not yet scored | derived |
| 24: issue #9, the fold | The fold moves the cell, and in 3 dimensions a cell is worth at most 0.35 dB | Its record agrees. The cell improves along the fold to +0.25 dB at its end, which is close packing, and every state loses to a ball over 8 or 24 weights | measured in `fold-path-codebooks-2026-11-17` |

Rows that do not move: 9, the Lean route, which was not opened here either. 13, kept directions. And what the check said of 2 bits: no ball arm has been run or scored under 4 bits, and the closed form is a statement about fine cells.

## The closed form

For one ball over a group of n weights, with the scale set from the group's own norm:

    error / squared weights  =  G * n * volume(ball in n dimensions)^(2/n) * 2^(-2 b / n) * f

G is the cell, b the bits of the group's point, and f what the scale adds: 1 + 2/n for the smallest scale that fits, because the group then lies on a wall of its cell, and the mean overshoot of the grid for a scale that is rounded up to one. The first three factors are in `bounds/bounds.out`, section 10, and in the record of balls over more than 8 weights. The fourth is level 2 of `shape/shapelib.py`. `shape/ballformula.py` computes it and reads no weight.

Against the Qwen run, mean over 196 matrices (`shape/output_law.qwen25.out`):

| Arm | Bits | By the closed form | Measured | Mean KL from the anchor | Scored |
|---|---|---|---|---|---|
| `EBG128:b560s16` | 4.500 | -25.60 dB | -25.60 | 0.0200 | 0.0195 |
| `EBG32:b136s8` | 4.499 | -25.15 | -25.13 | 0.0221 | 0.0208 |
| `ZBG128:b560s16` | 4.500 | -24.94 | -24.94 | 0.0232 | 0.0227 |
| `EB32@32:16` | 4.498 | -23.57 | -23.55 | 0.0319 | 0.0339 |
| `EB128@128:16` | 4.124 | -23.34 | -23.34 | 0.0336 | 0.0331 |
| `ZB32@32:16` | 4.499 | -22.91 | -22.90 | 0.0370 | 0.0367 |
| `EBG128:b504s8` | 4.000 | -22.84 | -22.83 | 0.0377 | 0.0371 |
| `ZB128@128:16` | 4.125 | -22.69 | -22.68 | 0.0390 | 0.0377 |

The anchor is the scalar rule with groups of 32, `X32:16`: mean KL 0.0620 at -20.67 dB.

## Where the output follows the weight error, and where not

The same file, for the arms the closed form does not cover, at their measured weight error. The last column is the scored KL over what the anchor's constant gives.

| Arm | What | Above the constant |
|---|---|---|
| llama.cpp `Q4_0` | the scalar rule, groups of 32 | +0.02 dB |
| `EC32@32:16` | E8 in a cross-polytope per group of 32 | +0.11 |
| `X128:16` | the scalar rule, groups of 128 | +0.14 |
| `VE8Z32:16` | the Voronoi code with a codeword at zero, groups of 32 | +0.18 |
| `EB128@R:16`, `ZB128@R:16` | a ball over 128 weights, one scale per row | +0.77, +0.86 |
| llama.cpp `Q3_K_M`, `Q4_K_M` | its mixed files | +0.70, +0.84 |
| `VE832:16`, `VE8128:16` | the Voronoi code with no codeword at zero | +1.22, +2.39 |
| `VE8ZR:16`, `VE8ZR:32` | the same code with one, one scale per row | +2.62, +2.43 |
| `XR:32`, `XR:16` | the scalar rule, one scale per row | +3.26, +4.31 |
| `VE8R:16`, `VE8R:32` | no codeword at zero, one scale per row | +14.9, +15.5 |

[Reading] One line divides the table: whether the error is the same share of every part of a matrix. With one scale per group of up to 128 weights it nearly is, and the constant holds whatever the lattice, the region and the bits. With one scale per row a row with one large weight takes a coarse scale for all of its weights, and the error gathers in such rows. The record of the check measured that gathering on weight error alone, as a spectral ratio that rose from 1.4 to 1.9 with the group. Here it shows as KL. This is one model and one text, and the line was drawn after the rows were read.

The record of the Qwen run says the same of nine arms: their gain in KL is 0.91 to 1.08 times their gain in weight error. What is added here is the constant, the arms outside the nine, and that for eight of them the weight error itself needs no run.

## The third model: predictions and verdicts

`notes/issue10-output-predictions.md`, committed at `49f7a73` at 19:34:39 UTC and pushed two seconds later. The last row of that model on the other branch at that time was llama.cpp's `Q4_0`, committed at 19:33:42. Its ball arms were not scored.

The anchor: `X32:16` of `SmolLM2-1.7B-Instruct`, mean KL 0.1257 at -21.17 dB. That is 16.4 of KL per unit of weight error, where Qwen2.5 has 7.2.

| | Prediction | Verdict |
|---|---|---|
| O1 | The measured weight error of the four ball arms is within 0.05 dB of the closed form | pending |
| O2 | Their mean KL is within 0.5 dB of the anchor's constant times the closed form: 0.0453, 0.0527, 0.0723 and 0.0855 | pending |
| O3 | Their KL over that of `EBG128:b560s16` is within 0.4 dB of the closed form alone: +0.65, +2.03 and +2.76 dB | pending |
| O4 | `VE8Z32:16` is within 0.5 dB of the constant at its measured weight error | pending |
| O5 | `XR:16` and `VE8ZR:16`, one scale per row, are at least 1.5 dB above it | pending |
| O6 | `EBG128:b560s16` has the lowest mean KL of the twelve variants | pending |

`shape/output_law.py` prints the verdicts from that run's rows. When this note was committed none of the four ball arms had a score.

## Stage 4, under both limits

`shape/design.out`: one ball per group of 32, 64 or 128 weights, integer levels or E8, a scale of 16, 8 or 6 bits, at 4.5 and at 4.0 bits per weight. For each, the error by the closed form with the exact count of its codebook, the KL the anchors give on both models, and the cost of the index.

At 4.5 bits:

| Arm | Weight error | Mean KL on Qwen2.5: predicted, scored | Table of the index | Look-ups per weight |
|---|---|---|---|---|
| `EBG128:b568s8` | -25.85 dB | 0.0188, not scored | 621 MB | not built |
| `EBG64:b280s8` | -25.63 | 0.0198, not scored | 73 MB | 12.7 |
| `EBG128:b560s16` | -25.60 | 0.0200, 0.0195 | 561 MB | not built |
| `EBG32:b138s6` | -25.32 | 0.0213, not scored | 9.1 MB | 11.9 |
| `ZBG128:b568s8` | -25.20 | 0.0219, not scored | 34 MB | 9.1 |
| `EBG32:b136s8` | -25.14 | 0.0222, 0.0208 | 7.9 MB | 10.9 |
| `ZBG64:b280s8` | -24.98 | 0.0230, not scored | 4.0 MB | 8.9 |
| `ZBG128:b560s16` | -24.94 | 0.0232, 0.0227 | 31 MB | 8.7 |
| `ZBG32:b138s6` | -24.68 | 0.0247, not scored | 0.51 MB | 8.7 |
| `ZBG32:b136s8` | -24.49 | 0.0258, not scored | 0.44 MB | 8.4 |
| `EBG32:b128s16`, which is `EB32@32:16` | -23.56 | 0.0319, 0.0339 | 5.3 MB | 9.6 |

What the formula says of each knob at 4.5 bits, in dB of weight error:

- **The lattice.** E8 for integer levels: 0.65, at any group and any scale. Its table is 18 times larger.
- **The bits of the scale.** 8 bits for 16: 1.58 with groups of 32, 0.69 with 64, 0.25 with 128. 6 bits for 8: 0.18 with groups of 32, nothing with 64, a loss of 0.10 with 128.
- **The group.** 128 weights for 32, each with its best scale: 0.53. The table is 68 times larger for E8.

So the arm that was scored first, 128 weights with a 16-bit scale, is matched by 64 weights with an 8-bit scale at an eighth of the table: -25.63 against -25.60. And integer levels over 64 weights with an 8-bit scale, at 4 MB, are predicted ahead of integer levels over 128 with a 16-bit scale, at 31 MB.

The pick under a limit on the table, at 4.5 bits:

| Table no larger than | Pick | Weight error | Behind the best arm |
|---|---|---|---|
| 0.5 MB | `ZBG32:b136s8` | -24.49 dB | 1.36 dB |
| 2 MB | `ZBG32:b138s6` | -24.68 | 1.18 |
| 8 MB | `EBG32:b136s8` | -25.14 | 0.71 |
| 32 MB | `EBG32:b138s6` | -25.32 | 0.53 |
| any | `EBG128:b568s8` | -25.85 | |

A table's size is one cost and not the only one. A group's number is as long as its bits, 136 to 570 here, and the look-ups inside a group wait on one another. The check of issue #15 says the same. Which limit binds is for a kernel to say, and none exists.

## For issue #15

Three figures its two workstreams can use, all by the formula and none scored:

- **The arm it calls cheapest, integer levels over 32 weights with an 8-bit scale**: -24.49 dB, so a mean KL of 0.0258 on Qwen2.5 and 0.0584 on SmolLM2 if the constant holds. That would be under `Q4_K_M` with an importance matrix on Qwen2.5, 0.0282 at 4.81 bits, by 0.4 dB.
- **Dropping E8 costs 0.65 dB of KL**, whatever else is chosen. Scored once: 0.66 dB.
- **"Right at the Shannon boundary"**: the scored arm is 1.5 dB above Shannon's floor for Gaussian weights, and 0.6 dB above what E8 can reach with any group and any split of the bits.

## Limits

- **One model fitted, one being tested.** The constant per model is a reading of 27 rows of one run. The third model is the test.
- **The line between the arms that follow the constant and those that do not was drawn after the rows were read.** It has a mechanism behind it and no derivation.
- **Counted bits.** Nothing here has a packed file. The table sizes are of an index written to be exact and not to be fast.
- **Fine cells.** No ball arm has been run under 4 bits. At 2 bits the formula of the check was low by 0.8 dB for blocks of 8.
- **The mean of the matrices' errors in dB is what the constant is fitted to.** On Qwen2.5 the pooled error of the scalar rule differs from it by 0.5 dB, and with the pooled figure the fit is worse.
- **The formula sees weights only.** The importance-matrix baseline shows what reading data gives llama.cpp's files: a sixth off their KL. No ball arm reads data.
- **Published work.** That the granular and the boundary gain separate, and that a ball is the best region for a round source, is in print since 1989 and 1993 (rows 39 to 41 of the check). The closed form is those results under this rule. The constant per model is close to the linearity theorem of HIGGS (arXiv 2411.17525), which the check quotes: for small relative errors the loss in perplexity is a weighted sum of the layers' relative errors. That paper reports its prediction diverging under about 3 bits.

## Sources

- The records `model-output-qwen25-2026-11-17`, `ball-dimension-no-clip-rule-2026-11-17`, `fold-path-codebooks-2026-11-17`, `imatrix-baseline-2026-11-17` and `model-output-smollm2-2026-11-17` on the branch `bayan/charming-gates-jlas14`, and `shape/ballindex.out`. **Checked.**
- Issues #14, #15 and #16 and their comments as they stood at 19:30 UTC on 2026-11-17. **Checked.**

## Disclosure

Written by an AI agent session under the owner's direction. The runs it reads are another session's. Nothing here has been reviewed by a person.
