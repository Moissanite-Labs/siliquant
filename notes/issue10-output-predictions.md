# Issue #10 on a model's output: predictions for the third model, before its ball arms are scored

Written 2026-11-17, about 19:35 UTC. The run of the record `model-output-smollm2-2026-11-17` had scored two variants when this was written: `unchanged` and `X32:16`. Both rows were read. A third, llama.cpp's `Q4_0`, landed while it was being finished: mean KL 0.1271. No ball arm of that model had a score. This note is committed and pushed before any does, and the times of the commits on the two branches say whether that held.

## Why

The check of issue #10 left one claim of its stage 4 untested: that the formula says what a change of a knob does to a model's output. The run on `Qwen2.5-1.5B-Instruct` has since scored 27 variants. Its record says that for the arms with one ball per group, a decibel of weight error is a decibel of KL.

For those arms the formula of issue #10 needs no probe and no map. So it can be stated before a model is touched, and with one scored arm it gives a number for the output of every other. This note states it for a model on which it has not been seen.

## The formula with one ball per group

With one ball over a whole group of n weights, the largest norm of a block in the group is the norm of the group. The region part of the formula is then n times the volume of the unit ball to the power 2/n, for any weights whatever. So

    error / squared weights  =  G * n * volume(ball in n dimensions)^(2/n) * 2^(-2 b / n) * f

with G the cell (1/12 for integer levels, 929/12960 for E8), b the bits of a group's point, and f what the scale adds: 1 + 2/n when the scale is the smallest that fits, since the group then lies on a wall of its cell, and the mean overshoot of a grid when the scale is rounded up to one. `shape/ballformula.py` computes it. It reads no weight and has no fitted constant.

The record `ball-dimension-no-clip-rule-2026-11-17` found the first three factors on weight error, to 0.07 dB. The factor f is level 2 of `shape/shapelib.py`, and for a grid it is new here.

## The claim about the output

For an arm whose error is the same share of every group, the mean KL of a model is one constant times the weight error. The constant belongs to the model and the text. One scored arm gives it.

Expected to hold: one scale per group of 32 or 128 weights, set so that nothing is clipped, with zero among the codewords. Expected to fail: one scale per row, where the error is gathered in the rows with a large weight, and the Voronoi code of the harness, which has no codeword at zero. The record `shape-formula-held-out-2026-11-17` measured the first of those as a spectral ratio that grows with the group. Neither failure is derived. Both are read off the Qwen run.

## What the Qwen run shows of this, after the fact

`shape/output_law.qwen25.out`, from the rows of that record. Not a test: every row was seen.

- **The closed form against the measured weight error**, on 196 matrices and 1.31 billion weights: within 0.02 dB on all eight arms it covers.
- **KL from one anchor.** With `X32:16` as the anchor (mean KL 0.0620 at -20.67 dB), the closed form gives the mean KL of the eight ball arms within 0.27 dB, between 0.94 and 1.06 times.
- **Other arms inside the claim**, with their measured weight error: `EC32@32:16` +0.11 dB, `X128:16` +0.14, `VE8Z32:16` +0.18, llama.cpp's `Q4_0` +0.02.
- **Arms outside it**: one scale per row +0.8 to +4.3 dB above the law, the code with no codeword at zero +1.2 to +15.5, llama.cpp's `Q4_K_M` and `Q3_K_M` +0.8 and +0.7.

## Predictions for `SmolLM2-1.7B-Instruct`

The anchor is its `X32:16`: mean KL 0.12568 at a mean weight error of -21.166 dB. That is 16.4 of KL per unit of weight error, where Qwen2.5 had 7.2.

| Arm | Weight error by the closed form | Mean KL by the anchor | Its KL over that of `EBG128:b560s16` |
|---|---|---|---|
| `EBG128:b560s16` | -25.60 dB | 0.0453 | |
| `ZBG128:b560s16` | -24.94 | 0.0527 | +0.65 dB |
| `EB32@32:16` | -23.57 | 0.0723 | +2.03 |
| `EBG128:b504s8` | -22.84 | 0.0855 | +2.76 |

`shape/output_law.py` computes each verdict from the run's `results.jsonl`.

- **O1.** The measured mean weight error of each of the four is within 0.05 dB of the closed form. The plan of that run already says the four do not depend on the weights, and its author saw them on four matrices. So this is a check of the arithmetic and not a blind prediction.
- **O2.** The mean KL of each of the four is within 0.5 dB of the second column: between 0.89 and 1.12 times it.
- **O3.** The mean KL of the last three over that of `EBG128:b560s16` is within 0.4 dB of the third column. This needs no anchor: it is the closed form alone. The middle one is the cell of E8 and nothing else.
- **O4.** `VE8Z32:16`, the Voronoi code with a codeword at zero and groups of 32, has a mean KL within 0.5 dB of the anchor's law at its measured weight error.
- **O5.** One scale per row is outside the law. `XR:16` and `VE8ZR:16` each have a mean KL at least 1.5 dB above the anchor's law at their measured weight error.
- **O6.** `EBG128:b560s16`, the arm with the smallest predicted error at 4.5 bits or fewer, has the lowest mean KL of the twelve variants, llama.cpp's `Q4_K_M` at 4.82 bits among them.

On the Qwen run all six would hold, with these marks: O2 at most 0.27 dB off, O3 at most 0.37, O4 at 0.18, O5 at 4.3 and 2.6. The marks were set from those figures.

If O2 and O3 hold, stage 4 of issue #10 holds on a model's output for this family of arms: a knob's effect on the KL is known from a closed form before anything is rounded, and one scored arm sets the level. If O2 fails and O3 holds, the law is right inside the family and the scalar rule is the wrong anchor on this model. If O3 fails, a decibel of weight error is not a decibel of KL here.

**Seen before writing this.** The whole record of the Qwen run and its rows. The plan of the third model's run and its first two rows. Nothing else of that model.

## What the formula says is left

At 4.5 bits per weight, E8, from `shape/ballformula.py`. No run.

| | Weight error |
|---|---|
| `EBG128:b560s16`, as scored | -25.60 dB |
| The same group with an 8-bit scale, 568 bits for the point | -25.85 |
| One ball per group of 512, 16-bit scale | -26.07 |
| E8's cell with a ball over ever more weights | -26.21 |
| Shannon's floor for Gaussian weights, which the best cell over ever more weights tends to | -27.09 |

So by the formula the arm that was scored is 0.6 dB from what E8 can reach with any group and any split of the bits, and 1.5 dB from the floor for Gaussian weights. A lattice better than E8 has at most 0.88 dB to gain. If a decibel of weight error stays a decibel of KL, those are the gains in KL as well. What the formula does not see is larger than all of it: on Qwen2.5 the scalar rule has 2.1 to 2.7 times the KL per unit of weight error with one scale per row that it has with one per 32 weights.

The floor for real weights is under the Gaussian one by what their spread is short of a Gaussian's. A region fitted to the weights, the shape knob of issue #10, is what would collect that. It is not computed here.
