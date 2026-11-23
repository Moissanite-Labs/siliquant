# Issue #10 after the model-output runs

Date: 2026-11-17
Status: complete. Six predictions for a third model were committed before its ball arms were scored: one holds and five fail, all five for one reason. A second run, planned before its arms, found the reason. Both are under "The third model" and "Why the lattices split".

[issue10-proposal-check.md](issue10-proposal-check.md) read the proposal of issue #10 when nothing of its family had been scored on a model. It left the proposal's claim about a model's output untested, and its second hard limit, decode cost, unreadable. Five things have landed since: a run on `Qwen2.5-1.5B-Instruct` with 27 scored variants, 28 since, a study of balls over more than 8 weights, a way to number the points of a ball-cut codebook, the fold study of issue #9, and the first half of an importance-matrix baseline. This note reads the open rows of the check against them and carries out stage 4 of the proposal with what they give.

**Measured** means a figure from a committed output of the record named. **Derived** means arithmetic from the formula. **Predicted** means stated here before the data existed.

## Result

- **For the arm that wins, the proposal's first two stages are not needed.** With one ball over a whole group the region part of the formula reads no weight. The formula is a closed form: the cell, the group's size, the bits, and what the scale adds. On `Qwen2.5-1.5B-Instruct` it gives the measured weight error of all eight arms it covers to 0.02 dB, on 196 matrices and 1.31 billion weights.
- **On that model the output follows it.** One scored arm gives the model's KL per unit of weight error. With it the closed form gives the mean KL of the eight arms within 0.27 dB. Four more arms with one scale per group of 32 or 128 and a codeword at zero follow the same constant within 0.18 dB at their measured weight error. Every one of those rows had been seen, so this is a fit and not a test.
- **It does not follow it with one scale per row, or without a codeword at zero.** Those arms have 0.8 to 4.3 dB and 1.2 to 15.5 dB more KL than the constant gives. The formula does not see either. One arm scored since is an exception: a cross-polytope over 128 weights with one scale per row is 0.17 dB above the constant.
- **A test before the data: of six predictions one holds and five fail.** Six predictions for `SmolLM2-1.7B-Instruct` were committed and pushed at `49f7a73`, when only its scalar rule and one of llama.cpp's files had a score. The closed form gives the weight error of its four ball arms to 0.035 dB. It does not give their KL. On this model a decibel of weight error is not a decibel of KL.
- **On that model integer levels in a ball score under E8 in the same ball**: 0.0560 against 0.0609, though E8 has 0.66 dB less weight error. The arms with one scale per group sort by lattice: 16.4 to 17.4 of KL per unit of weight error on integer levels, 21.4 to 23.0 on five arms on E8. On Qwen2.5 the same arms were all between 6.8 and 7.7. Three statements made from that split before their rows all held. Nothing measured on the weights tells the two lattices apart on that model.
- **Why: E8's error is noise of its size, and the error of integer levels is better than noise at one place, zero.** A run of this branch's own (`experiments/lattice-split-smollm2-2026-11-17`, plans committed before their arms) put noise in place of each arm's error. For E8 nothing changes. For integer levels the KL rises 1.1 dB, and all of the rise comes from the 7.8 percent of the weights they round to zero. Those weights hold 27 percent of the arm's KL: an error on them costs 4 to 9 times what an error on any other weight costs. The formula counts every squared error alike.
- **Stage 4 can now be carried out under both of its limits.** `shape/design.out` lists 36 settings of the knobs with the error by the formula, the KL it implies on two models, the size of the table that numbers the codebook's points, and the look-ups to read a group back. 30 of the 36 have not been scored on any model.
- **The shape knob is closed in practice for the language model.** Once a region spans the group, the best p-norm region is worth 0.05 dB over the ball in the mean on 21 matrices, by the formula. On the output a cross-polytope and a ball over 32 weights score the same: 0.0341 and 0.0339.
- **By the formula little is left inside this family.** The scored arm at 4.5 bits is 0.25 dB from the same group with a shorter scale, 0.6 dB from what E8 reaches with any group, and 1.5 dB from Shannon's floor for Gaussian weights. No lattice can add more than 0.88 dB to E8.
- **A proved bound carries over.** At the safe scale no group of any matrix has an error over a fixed share of its own squared weights: -22.6 dB for E8 in one ball per group of 128 at 4.5 bits. It costs 0.55 dB against the searched scale.

## Rows of the check that move

| Row | Then | Verdict | Basis |
|---|---|---|---|
| 4, 7, 10: the probe, the map, the fit to a family | The map has to hold the region part of the matrix | With one ball per group there is nothing to map: the region part is the group's size times a power of the volume of a ball. A probe is needed only for an arm whose scale is shared by several blocks. Measured on four families of weights the error of such an arm is the same: -23.54, -23.54, -23.57 and -23.54 dB | derived, measured in `experiments/ball-dimension-no-clip-rule-2026-11-17/results.out` |
| 11: the shape knob | p = 5/4 or 3/2 is best on real weights, by 0.2 to 0.3 dB, with blocks of 8 | That gain is of blocks of 8 under a scale shared by a group. One ball per group of 32 is 1.1 dB ahead of a ball over 8 weights with the same scale, -23.55 against -22.43 dB on the Qwen matrices, four times what the best shape over 8 weights adds. And with one region per group the best p is worth 0.05 dB on qwen and 0.3 dB on whisper in the mean, the second carried by four matrices with a lump at zero | measured in `experiments/model-output-qwen25-2026-11-17/results.jsonl`, derived in `shape/group_shape.out` |
| 12: the size knob | Groups of 128 on weight error. "The knob most likely to be set wrongly for an output" | Two knobs: the weights under one ball, and the bits of the scale. For one ball per group the group costs the output nothing: 128 weights at 4.125 bits score as 32 weights at 4.5. The caution holds for rules that set the scale from the largest entry, where groups of 128 raise the KL by more than half | measured in `experiments/model-output-qwen25-2026-11-17/results.jsonl` |
| 17: the decode budget | Cannot be read: no packed format | Readable in part. `shape/ballindex.py` numbers the points of a ball-cut codebook. A table's size and the look-ups per weight are known for every arm. No kernel computes from it, and nothing is timed | measured in `shape/ballindex.out` and `shape/design.out` |
| 19, 23: a knob's effect on the output | Not tested | On one model, for arms with one scale per group and a codeword at zero: a decibel of weight error is a decibel of KL, within 0.3 dB over twelve arms, read after the fact. On a second model, against predictions made before its rows, it holds inside each lattice and not between the two: one of six predictions holds | measured in `shape/output_law.qwen25.out`, measured in `shape/output_law.smollm2.out` |
| 20: a bound for any weights | The safe scale, over blocks of 8 | Carried to one ball per group. Queued for the output as `SEB8@32:16` in the Qwen run, over 8 weights, not yet scored | derived |
| 24: issue #9, the fold | The fold moves the cell, and in 3 dimensions a cell is worth at most 0.35 dB | Its record agrees. The cell improves along the fold to +0.25 dB at its end, which is close packing, and every state loses to a ball over 8 or 24 weights | measured in `experiments/fold-path-codebooks-2026-11-17/results.out` |

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
| `EC128@R:16` | a cross-polytope over 128 weights, one scale per row. Scored after this note was first written | +0.17 |
| `EB128@R:16`, `ZB128@R:16` | a ball over 128 weights, one scale per row | +0.77, +0.86 |
| llama.cpp `Q3_K_M`, `Q4_K_M` | its mixed files | +0.70, +0.84 |
| `VE832:16`, `VE8128:16` | the Voronoi code with no codeword at zero | +1.22, +2.39 |
| `VE8ZR:16`, `VE8ZR:32` | the same code with one, one scale per row | +2.62, +2.43 |
| `XR:32`, `XR:16` | the scalar rule, one scale per row | +3.26, +4.31 |
| `VE8R:16`, `VE8R:32` | no codeword at zero, one scale per row | +14.9, +15.5 |

One line divides the table: whether a scale serves a group of up to 128 weights or a whole row.

**What the weights say of that line** (`shape/row_spread.qwen25.out`, written after the rows were read, the scalar rule rounded on all 196 matrices). A row of a matrix is one output. For each rule, take every row's squared error over its own squared weights.

| Rule | Weight error of the matrix | Mean of the rows' shares | Root mean square of the rows' shares over their mean | Worst row over the matrix's share | Rows over four times the matrix's share |
|---|---|---|---|---|---|
| `X32:16` | -20.67 dB | -20.70 | 1.02 | 2.4 | none |
| `X128:16` | -18.73 | -18.76 | 1.08 | 4.4 | 0.2 % |
| `XR:16` | -14.91 | -14.86 | 1.35 | 10 | 1.8 % |
| `XR:32` | -20.75 | -20.74 | 1.46 | 15 | 1.7 % |

- **The excess is not in the mean.** The mean of the rows' shares is the matrix's share to 0.05 dB under every rule. A guess made before this was run, that the output follows the mean over the rows and not the matrix, is wrong.
- **It goes with the spread.** With one scale per row the rows' shares are 1.35 to 1.46 times as uneven as an even spread, against 1.02 to 1.08 with groups. `ffn_down` is the most uneven kind, 1.8 to 2.3. With one ball per group every row has the same share by construction.
- **The spread sorts the scalar rule onto the two sides of the line and does not size the excess.** `XR:32` is more uneven than `XR:16` and has the smaller excess, 3.3 against 4.3 dB. And a ball over 128 weights with one scale per row has 1.05 by the formula, less than `X128:16`, with an excess of 0.8 dB against 0.14.

So nothing computed from the weights here predicts how far an arm with one scale per row lies above the constant. This is one model and one text, and the line was drawn after the rows were read.

The record of the Qwen run says the same of nine arms: their gain in KL is 0.91 to 1.08 times their gain in weight error. What is added here is the constant, the arms outside the nine, and that for eight of them the weight error itself needs no run.

## The third model: predictions and verdicts

`notes/issue10-output-predictions.md`, committed at `49f7a73` at 19:34:39 UTC and pushed two seconds later. The last row of that model on the other branch at that time was llama.cpp's `Q4_0`, committed at 19:33:42. Its ball arms were not scored.

The anchor: `X32:16` of `SmolLM2-1.7B-Instruct`, mean KL 0.1257 at -21.17 dB. That is 16.4 of KL per unit of weight error, where Qwen2.5 has 7.2.

| | Prediction | Verdict |
|---|---|---|
| O1 | The measured weight error of the four ball arms is within 0.05 dB of the closed form | **Holds.** +0.013, +0.028, +0.016 and +0.035 dB. |
| O2 | Their mean KL is within 0.5 dB of the anchor's constant times the closed form: 0.0453, 0.0527, 0.0723 and 0.0855 | **Fails.** The three E8 arms are 1.28, 1.49 and 1.41 dB above. `ZBG128:b560s16` is inside, at +0.26. |
| O3 | Their KL over that of `EBG128:b560s16` is within 0.4 dB of the closed form alone: +0.65, +2.03 and +2.76 dB | **Fails.** `ZBG128:b560s16` is 0.36 dB under, where +0.65 was said. The two E8 arms hold: +2.24 against +2.03 and +2.88 against +2.76. |
| O4 | `VE8Z32:16` is within 0.5 dB of the constant at its measured weight error | **Fails.** It is 1.41 dB above. |
| O5 | `XR:16` and `VE8ZR:16`, one scale per row, are at least 1.5 dB above it | **Fails.** `XR:16` is 1.36 dB above. `VE8ZR:16` is 2.36 above, as said. |
| O6 | `EBG128:b560s16` has the lowest mean KL of the twelve variants | **Fails.** `ZBG128:b560s16`, at 0.0560, is under its 0.0609, and is the lowest of the twelve. |

Final: all thirteen rows of the run are in, the last at 21:57 UTC. `shape/output_law.py` prints the figures from that run's rows, and `shape/output_law.smollm2.out` holds its last output. A prediction that names several arms fails when one of them does.

**The first arm is in, and it goes against O2.** `EBG128:b560s16` scored a mean KL of 0.0609 at 19:57 UTC. Its weight error is -25.585 dB, 0.013 dB from the closed form. The anchor gives 0.0453. It is 1.28 dB above that, outside the mark of 0.5. Its gain in KL over the scalar rule is 3.15 dB for a gain of 4.42 dB in weight error: 0.71 of it, where Qwen2.5 had 1.02. So on this model a decibel of weight error is not a decibel of KL between the scalar rule and a ball arm. llama.cpp's `Q4_K_M` is 1.63 dB above the same constant. The arm is still the lowest of the four variants scored, 11 errors of the gap under `Q4_K_M` at 4.82 bits.

Two readings fit one arm, and the next arms tell them apart. If the ball arms share a constant of their own, O3 holds: their KL ratios are the closed form's. If the KL of this model grows more slowly than the weight error, O3 fails for the two arms furthest from this one, by 0.6 and 0.8 dB.

**The second arm is in, and it sides with the first reading.** `EB32@32:16` scored 0.1018 at 20:13 UTC, at a weight error of -23.539 dB, 0.028 dB from the closed form. Over `EBG128:b560s16` that is +2.24 dB of KL, where the closed form alone says +2.03: inside the mark of 0.4. The other reading said +1.44. So far the two ball arms share a constant, 22 to 23 of KL per unit of weight error, and the scalar rule has another, 16.4. Against the scalar rule's constant this arm too is off, by 1.49 dB. On Qwen2.5 the three constants were 7.1, 7.7 and 7.2.

**The third arm is in, and it is the surprise.** `ZBG128:b560s16`, integer levels in the same ball with no lattice, scored 0.0560 at 20:20 UTC. That is under E8 in the same ball, 0.0609, although its weight error is 0.66 dB higher, as the closed form says to 0.02 dB. The closed form put it 0.65 dB above E8 in KL. It is 0.36 dB below: 1.0 dB from the prediction, 8.6 errors of the gap. Against each other the two arms are 2.9 errors apart, which the run's record would call near a tie. So O3 fails on this arm, and O6 fails: the arm with the smallest weight error does not have the lowest KL.

What the three arms show together, on this model:

| Arm | Lattice | Weight error | Mean KL | KL per unit of weight error |
|---|---|---|---|---|
| `X32:16` | integer levels | -21.17 dB | 0.1257 | 16.4 |
| llama.cpp `Q4_0` | integer levels | -21.17 | 0.1271 | 16.6 |
| `ZBG128:b560s16` | integer levels | -24.93 | 0.0560 | 17.4 |
| `EBG128:b560s16` | E8 | -25.59 | 0.0609 | 22.0 |
| `EB32@32:16` | E8 | -23.54 | 0.1018 | 23.0 |
| llama.cpp `Q4_K_M` | its own | -24.56 | 0.0838 | 23.9 |

The line on this model is not between the scalar rule and the ball. It is between integer levels and E8. The three arms on integer levels share a constant within 0.26 dB, the ball among them. The two on E8 share another, 1.2 to 1.5 dB higher. Within each lattice a decibel of weight error is a decibel of KL. Across them it is not: E8's cell, worth 0.65 dB of weight error, costs this model's output more than it gives. On Qwen2.5 the same arms had one constant, and E8 was 0.66 dB ahead.

The formula cannot see this. It has the lattice as a cell constant and nothing else.

**What the lattice reading says of the rows to come.** Written at 20:26 UTC, when the run had seven rows and had been started again at 20:22 for the next six. It is a reading of five arms after the fact, so it is worth something only if it says what comes next. Three statements, each against a prediction already made:

| Arm | The scalar rule's constant says (O2, O4) | The lattice reading says | They differ by |
|---|---|---|---|
| `VE8Z32:16`, E8 with a codeword at zero, groups of 32 | KL within 0.5 dB of 16.4 times its weight error. About 0.090 if its weight error is the -22.6 dB seen on four matrices. | KL 0.9 to 1.9 dB above that: the constant of the two E8 arms, 22.0 to 23.0. About 0.121 to 0.126. So O4 fails. | 1.3 to 1.5 dB |
| `EBG128:b504s8`, E8 at 4.0 bits | KL 0.0855 | E8 against E8, so the closed form holds: 2.76 dB above `EBG128:b560s16`, which is 0.115. Between 0.105 and 0.126. The last arm of O3 holds and O2 fails on it. | 1.3 dB |
| `VE832:16`, E8 with no codeword at zero | Not in the claim | At least 0.9 dB above the scalar rule's law at its weight error: the E8 constant, and whatever the missing zero adds | |

If `VE8Z32:16` lands within 0.5 dB of the scalar rule's law, the lattice reading is wrong and the split is something else about the ball arms on E8.

**The fourth row, and the first of the three statements.** `VE832:16` scored 0.1166 at 20:43 UTC, at a weight error of -22.638 dB. That is 21.4 of KL per unit of weight error, 1.14 dB above the scalar rule's law. The statement was at least 0.9. It holds. Three arms on E8 now have 21.4, 22.0 and 23.0, from two ways of cutting a codebook out of the lattice. Three on integer levels have 16.4, 16.6 and 17.4. On Qwen2.5 this arm lay 1.2 dB above its twin with a codeword at zero. Here it is at the constant of the E8 arms that have one, so the missing zero may cost nothing on this model. `VE8Z32:16` will say.

**The fifth row: one scale per row.** `XR:16` scored 0.4934 at 20:48 UTC, at a weight error of -16.585 dB. `shape/row_spread.smollm2.out` had put its weight error at -16.59 before it was built. It is 1.36 dB above the scalar rule's law. O5 said at least 1.5, so O5 fails on this arm, by 0.14 dB. The direction is as stated and the size is not: on Qwen2.5 the same arm was 4.3 dB above. It also spoils the tidy version of the lattice reading. This arm is on integer levels and has 22.5 of KL per unit of weight error, the figure of the E8 arms. The reading was stated for arms with one scale per group, and it still sorts those. But the two things that raise the constant on this model, E8 and one scale per row, raise it by the same 1.1 to 1.5 dB, and nothing here says they are different things.

**The sixth and seventh rows: E8 with one scale per row.** `VE8R:16` scored 0.3290 at 21:05 UTC and `VE8ZR:16` 0.3376 at 21:24, at weight errors of -19.27 and -19.24 dB. They are 2.29 and 2.36 dB above the scalar rule's law, with 27.8 and 28.3 of KL per unit of weight error. For `VE8ZR:16` O5 said at least 1.5, and that part holds. The two raises add up on this model, near enough: E8 alone is 1.1 to 1.5 dB above the law, one scale per row alone 1.36, and both together 2.3. The codeword at zero does nothing here: with and without it the two arms are 1.2 errors apart. On Qwen2.5 it took the same arm from 3.66 to 0.21. Both arms score under the scalar rule per row, 0.4934, as their weight error says they should.

**The eighth row, and the second of the three statements.** `EBG128:b504s8`, E8 at 4.0 bits with an 8-bit scale, scored 0.1183 at 21:37 UTC. Its weight error is -22.806 dB, 0.035 dB from the closed form: O1 holds on all four ball arms. Its KL is 2.88 dB above that of `EBG128:b560s16`, where the closed form says 2.76. The statement of 20:26 was 0.115, between 0.105 and 0.126. It holds. The scalar rule's constant gives 0.0855 and is 1.41 dB off, as the statement said it would be. Four arms on E8 with one scale per group now have 21.4, 22.0, 22.6 and 23.0 of KL per unit of weight error. Within that lattice a decibel of weight error has been a decibel of KL three times out of three: +2.24 against +2.03, +2.88 against +2.76, and the Voronoi arm at the same constant. At 4.0 bits the arm is 2.8 errors under llama.cpp's `Q4_0` at 4.5 bits.

**The last row, and the third statement.** `VE8Z32:16`, the E8 Voronoi code with a codeword at zero and groups of 32, scored 0.1247 at 21:57 UTC, at a weight error of -22.610 dB. The scalar rule's law gives 0.0901. It is 1.41 dB above, so O4 fails. The statement of 20:26 was 0.9 to 1.9 dB above, about 0.121 to 0.126, with O4 failing. It holds. All three statements made from the lattice reading before their rows have held. Five arms on E8 with one scale per group have 21.4, 22.0, 22.6, 22.7 and 23.0 of KL per unit of weight error: 0.3 dB from end to end, over two codebooks, two group sizes and two rates. The codeword at zero made this arm worse on this model, by 2.8 errors: 0.1247 with it and 0.1166 without. On Qwen2.5 it made it better.

**The count.** Of the six predictions committed before the ball arms were scored, one holds and five fail. The one that holds is about weight error. The five that fail are all about the output, and all for one reason: on this model the scalar rule's constant is not the constant of an arm on E8.

**What the weights alone say of the two lattices** (`shape/lattice_spread.smollm2.out`: all 168 block matrices rounded in full with each lattice in one ball per group of 128. Written after the rows were seen). Nothing that tells them apart.

| Measured on the weights, mean over 168 matrices | Integer levels | E8 | E8 less integer levels |
|---|---|---|---|
| Error of the matrix | -24.928 dB | -25.585 | -0.66 dB |
| Mean over the columns of a column's error over its weight | -24.765 | -25.417 | -0.65 |
| The same in the tenth of the columns with the least weight | | | -0.65 |
| In the hundredth with the least | | | -0.64 |
| In the one column with the least | | | -0.58 |
| Mean over the rows of a row's error over its weight | -24.928 | -25.585 | -0.66 |
| The rounded matrix along the matrix | 0.99836 | 0.99824 | |
| Rounded weights that are zero | 7.9 percent | 3.95 percent | |
| Error on a weight under a quarter of a step, in squared steps | 0.0208 | 0.0720 | |
| Those weights' part of the error (they are 3.95 percent of the weights) | 1.0 percent | 3.9 percent | |

A column of a matrix is one of its inputs. llama.cpp's importance matrix, and any measure that counts each input by how large it is, weights the columns. E8's lead is the same 0.6 to 0.7 dB in the columns with the least weight as in the whole matrix. So no weighting of the columns turns a lead of 0.66 dB into a deficit of 0.36. The columns of this model are also even: the root mean square of their squared weights over the mean is 1.00 to 1.05 by kind of matrix.

The one difference is at zero. Integer levels send a weight under half a step to zero, and E8 sends half of those elsewhere. That moves 3 percent of the error. For it to carry the split, the output would have to count an error on those weights about ten times as heavily as on the rest.

What the weights cannot show is how the errors of the eight weights of a block go together against inputs that go together. E8 rounds eight weights jointly and integer levels round each alone. That needs the model's activations.

**The same measure on Qwen2.5 says the opposite of the output** (`shape/lattice_spread.qwen25.out`, 196 matrices). There the columns are uneven: 1.36 to 1.40 in the three matrices that read the attention's input, 3.03 in `ffn_down`, against 1.00 to 1.05 on SmolLM2. Some columns have almost no weight, down to 0.02 of a step, and in those E8 spends error where integer levels spend none. Counted by column, E8's lead falls from 0.66 dB to 0.17, and in `attn_v` it is 1.43 dB behind. So by the columns E8 should lose ground on Qwen2.5 and keep it on SmolLM2. The scores say the reverse: on Qwen2.5 E8 keeps its 0.66 dB, and on SmolLM2 it loses a full decibel. Error counted by column is not what the output follows on either model.

**From the weights of this model, before its arms with one scale per row are scored** (`shape/row_spread.smollm2.out`, 19:58 UTC). The scalar rule per row has rows 1.30 times as uneven as an even spread at 16 levels and 1.39 at 32, and 1.2 percent of the rows have over four times the matrix's share. With groups of 32 the figure is 1.01. That is the regime of Qwen2.5, where the arms per row lay above the constant. The measured weight error of `X32:16`, -21.166 dB, is reproduced to the last digit shown, and that of `XR:16` comes out at -16.59 dB.

## Why the lattices split: what a rule does to a weight near zero

`experiments/lattice-split-smollm2-2026-11-17`. The plan and each of three extensions were committed and pushed before their arms were scored. The model is run in NumPy by `shape/npforward.py`, since this session has no build of llama.cpp, on the harness's text and on 16 of its 40 passages. Four checks set in the plan hold: its reference perplexity is 0.19 percent from llama.cpp's, its weight errors are the harness's, its KL for three arms is within 6.5 percent, and it has the split, 1.21 dB where the harness has 1.02.

| Arm, at the weight error of the arm named | Mean KL | KL per unit of weight error |
|---|---|---|
| E8 in the ball | 0.0638 | 23.1 |
| Gaussian noise in place of its error, the same size in every group | 0.0622 | 22.5 |
| Integer levels in the ball | 0.0562 | 17.5 |
| Gaussian noise in place of its error | 0.0730 | 22.7 |
| Its own error, the weights of each group trading errors | 0.0737 | 22.9 |
| Noise on the 92 percent of the weights it does not round to zero | 0.0569 | 17.7 |
| Noise on the 7.8 percent it rounds to zero | 0.0706 | 22.0 |

- **Measured: E8's error acts as noise of its size.** 0.11 dB apart, and two draws of noise are 0.14 dB apart.
- **Measured: the error of integer levels is 1.1 dB better than noise, and only at zero.** Away from zero it is noise already. At zero the error on a weight is the weight itself, never larger and never of the other sign. E8 rounds half of those weights to zero and moves the rest to half a step.
- **Measured: the small weights are where the output is lost.** With all else exact, setting them to zero costs 0.0152 and noise of that size on them 0.0319. The integer arm's error on every other weight costs 0.0417. The parts add to 1 percent. Per unit of weight error that is 60, 127 and 14.
- **Measured: no single matrix shows it.** Fed the model's own inputs, the output of a matrix is off by the same amount under each of these arms, to 0.02 dB. The difference is made further on.
- **Measured: for one pattern of error a decibel is a decibel.** The integer arm's error made 0.66 dB smaller has a KL 0.67 dB smaller. The integer arm with 16 more bits a group has 0.76 dB less of both.
- **Measured: the feed-forward matrices carry it.** E8 there and integer levels in attention is 3.3 errors above the integer arm. The reverse is not above it.

Stated before their rows, and how they went:

| | Statement | Verdict |
|---|---|---|
| Plan | My guess: noise agrees with E8 and is above integer levels | Holds |
| Extension 1 | My guess: the gain is all at the weights rounded to zero | Holds |
| Extension 2 | Setting them to zero costs under a third of noise on them | **Fails**: 0.48 of it |
| | The two parts of the arm's error add within 10 percent | Holds: 1.2 percent |
| | Noise on the small weights costs at least twice per unit what noise on all does | Holds: 5.6 times |
| Extension 3 | Three levels at zero beat the same bits spent on every weight, by 2 errors | **Fails**: ahead by 0.20 dB, 0.9 errors |
| | Their KL is 0.043 to 0.045 | **Fails**: 0.04504 |
| | 16 more bits a group give 0.0472 | Holds: 0.04717 |

**What this does to the formula.** It is right about the cell and blind to the place. E8's cell is 0.66 dB better on this model too, and that shows wherever E8 is set against E8. What the formula lacks is not a property of a lattice. It is whether a rule can give a weight near zero an error larger than the weight, and how dear this model holds such an error. Three levels at zero take 0.96 dB off the KL for 0.31 dB off the weight error, so the formula's order of that arm and the one with the same bits everywhere is the wrong way round, by an amount this run cannot tell from nothing.

**What is not known.** Why an error on a small weight is dear here. And whether it is on Qwen2.5, where the two lattices share a figure: the instrument does not handle that model's biases and shared key heads.

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

- **The arm it calls cheapest, integer levels over 32 weights with an 8-bit scale**: -24.49 dB, so a mean KL of 0.0258 on Qwen2.5 if the constant holds there. That would be under `Q4_K_M` with an importance matrix on Qwen2.5, 0.0282 at 4.81 bits, by 0.4 dB. On SmolLM2 the scalar rule's constant holds for the one ball arm on integer levels, within 0.26 dB, and is low by 1.3 to 1.5 dB for the arms on E8. So `shape/design.out`'s figures for that model are near for its `ZBG` rows and low for its `EBG` rows.
- **Dropping E8 costs 0.65 dB of weight error, whatever else is chosen, and of KL on one model of two.** On Qwen2.5 it was scored at 0.66 dB of KL. On SmolLM2 integer levels in the same ball scored 0.36 dB *under* E8. So the case of issue #15 for integer levels, made there on cost, has a score on its side on one model. The lattice-split run says why: integer levels do not win on their cell there. They win at the 7.8 percent of the weights they round to zero, where E8 cannot follow. A code with E8's cell away from zero and integer rounding at zero would have both gains. None is built.
- **"Right at the Shannon boundary"**: the scored arm is 1.5 dB above Shannon's floor for Gaussian weights, and 0.6 dB above what E8 can reach with any group and any split of the bits.

## Limits

- **One model fitted, one tested, and the test went against it.** The constant per model is a reading of 27 rows of one run, and a 28th since. On the third model there is one constant per lattice, read off six arms after the fact.
- **The line between the arms that follow the constant and those that do not was drawn after the rows were read.** It has a mechanism behind it and no derivation.
- **Counted bits.** Nothing here has a packed file. The table sizes are of an index written to be exact and not to be fast.
- **Fine cells.** No ball arm has been run under 4 bits. At 2 bits the formula of the check was low by 0.8 dB for blocks of 8.
- **The mean of the matrices' errors in dB is what the constant is fitted to.** On Qwen2.5 the pooled error of the scalar rule differs from it by 0.5 dB, and with the pooled figure the fit is worse.
- **The lattice-split run is scored in NumPy, on 16 passages of one model.** It agrees with llama.cpp to 0.19 percent in perplexity. Its noise arms are one draw, two for one of them.
- **The formula sees weights only.** The importance-matrix baseline shows what reading data gives llama.cpp's files: a sixth off their KL. No ball arm reads data.
- **Published work.** That the granular and the boundary gain separate, and that a ball is the best region for a round source, is in print since 1989 and 1993 (rows 39 to 41 of the check). The closed form is those results under this rule. The constant per model is close to the linearity theorem of HIGGS (arXiv 2411.17525), which the check quotes: for small relative errors the loss in perplexity is a weighted sum of the layers' relative errors. That paper reports its prediction diverging under about 3 bits.

## Sources

- The records `model-output-qwen25-2026-11-17`, `ball-dimension-no-clip-rule-2026-11-17`, `fold-path-codebooks-2026-11-17`, `imatrix-baseline-2026-11-17` and `model-output-smollm2-2026-11-17` from PR #12, and `shape/ballindex.out`. **Checked.**
- Issues #14, #15 and #16 and their comments as they stood at 19:30 UTC on 2026-11-17. **Checked.**
- The record `lattice-split-smollm2-2026-11-17` on this branch: `results.out`, `layer_error.out`, `build.out`. Its rows are this session's. **Checked.**

## Disclosure

Written by an AI agent session under the owner's direction. The runs of `model-output-*` it reads are another session's. The lattice-split run is its own. Nothing here has been reviewed by a person.
