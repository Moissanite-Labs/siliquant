# Plan: E8 turned by a Hadamard matrix, on real weights

Written 2026-11-22, before any matrix of a model was rounded by the turned lattice and before anything was scored.

## Question

Two records of this repository disagree about E8 and one explains why.

- On `Qwen2.5-1.5B` E8 in one ball per group of 128 weights beats integer levels in the same ball: mean KL 0.0195 against 0.0227 at 4.5 bits.
- On `SmolLM2-1.7B` it loses: 0.0609 against 0.0560, though its weight error is 0.66 dB lower.
- The record lattice-split-smollm2-2026-11-17 found the reason. Integer levels round the smallest weights to zero, and on that model that is worth more than its size in weight error. E8 as it is usually written cannot do it: in half of its points every coordinate is a whole number and a half, so a weight near zero is moved to plus or minus half a step.

Draft pull request 24 proposes a way out. E8 turned by the Hadamard matrix of order 8, divided by the root of 8, is the same lattice in another position. Its points have whole coordinates, with a rule on which may be odd, so a small weight can stay at zero. A ball has the same number of its points, so the bits are the same. The pull request shows the effect on made-up matrices and says that real weights are the next experiment. Issue 25 lists that as step 1 and as not yet run.

This machine holds the three models and their reference outputs.

**Does turning E8 give it what integer levels have on SmolLM2, and keep what E8 has on Qwen2.5? Is the turned lattice the one arm to use on all three models?**

## What is run

**The models, the text and the score** are those of the earlier records: `Qwen2.5-1.5B-Instruct`, `SmolLM2-1.7B-Instruct` and `Llama-3.2-1B-Instruct`, 16-bit files, the wikitext-2 test text, 10,200 positions, mean KL against the reference's saved outputs, with `llama-perplexity --kl-divergence` of llama.cpp at `eada89c`. No calibration data and no forward pass for any arm.

**Three arms, each in one ball per group of 128 weights with a 16-bit scale.**

| Name | Lattice |
|---|---|
| `ZBG128:bBs16` | integer levels |
| `EBG128:bBs16` | E8 as usually written: whole numbers with an even sum, and the same moved by a half in every coordinate |
| `HBG128:bBs16` | that lattice turned, in every block of 8 consecutive weights of a row, by the Hadamard matrix over the root of 8 |

`B` is 560, which is 4.5 bits a weight with the scale, or 368, which is 3.0.

**The program** is `run.py` of this folder. The first two arms are the ball arm of the record model-output-qwen25-2026-11-17, whose code is not changed. For the third, that code's E8 arm is given the matrix with every block of 8 turned, and what it returns is turned back. The matrix that turns is its own inverse. It is the function `closest_E8_had` of the pull request, inside a ball. `test_turn.py` holds the lattice to the pull request's description of it by a decoder written another way, and holds the first two arms to the base program bit for bit.

**With `+K10`**, the ten loud columns of every `ffn_down` are left as in the reference and counted at 16 bits an entry, by the function of the record loud-columns-2026-11-18. On Llama-3.2 the arms with nothing kept are ruled by two hidden units, so the lattices are compared there with those columns kept.

**Each row also has, from the weights alone,** how many weights the arm stores as zero, and for the weights under 0.05, 0.1 and 0.2 of the root mean square of their matrix: how many there are, how many are stored as zero, and their squared error over their squared size.

**Thirteen rows are scored**, in `results.jsonl`.

| Model | Bits | Variants |
|---|---|---|
| SmolLM2 | 4.5 | `EBG128:b560s16` again, as a check. `HBG128:b560s16` |
| Qwen2.5 | 4.5 | `HBG128:b560s16` |
| Llama-3.2 | 4.5 | `EBG128:b560s16`, `HBG128:b560s16`, `EBG128:b560s16+K10`, `HBG128:b560s16+K10` |
| SmolLM2 | 3.0 | `ZBG128:b368s16`, `EBG128:b368s16`, `HBG128:b368s16` |
| Qwen2.5 | 3.0 | `ZBG128:b368s16`, `EBG128:b368s16`, `HBG128:b368s16` |

**Five more are built and not scored**, in `metrics.jsonl`, for what they do to the small weights: the arms that earlier records scored. `ZBG128:b560s16` and `EBG128:b560s16` on Qwen2.5, `ZBG128:b560s16` on SmolLM2, `ZBG128:b560s16` and `ZBG128:b560s16+K10` on Llama-3.2.

**Tried before this plan**, on made-up matrices only: the 11 tests of `test_turn.py`, which pass. No matrix of a model has been rounded by the turned lattice, by this program or any other.

## What is known beforehand

Scored in earlier records, at 4.5 bits, with no matrix:

| | Qwen2.5 | SmolLM2 | Llama-3.2 | Llama-3.2, loud columns kept |
|---|---|---|---|---|
| integer levels | 0.0227 | 0.0560 | 0.0478 | 0.0302 |
| E8 | 0.0195 | 0.0609 | not scored | not scored |
| `Q4_K_M` at 4.8 bits, no matrix | 0.0333 | 0.0838 | 0.0432 | |

E8 is 4.2 errors under integer levels on Qwen2.5 and 2.9 errors above them on SmolLM2. Its weight error is lower on both, by 0.67 dB and 0.66 dB.

From the lattice-split record, on SmolLM2: integer levels store 7.8 percent of the weights as zero. Noise of the same size on those weights takes away the whole of what integer levels gain over E8. On Qwen2.5 every arm shares one figure of KL per unit of weight error, so nothing is gained at zero there.

From the pull request, on made-up Gaussian blocks: of the weights under 0.05 of a step, E8 stores 65.7 percent as zero and the turned lattice 95.3 percent.

Nothing at 3.0 bits has been scored for a ball arm.

## Predictions

An error is the tool's standard error, and an error of a gap is the root of the sum of the two squares.

| | Prediction | It fails if |
|---|---|---|
| H1 | SmolLM2, 4.5 bits: the turned lattice is under E8 by 2 errors | it is not |
| H2 | SmolLM2, 4.5 bits: the turned lattice is under integer levels by 2 errors | it is not |
| H3 | Qwen2.5, 4.5 bits: the turned lattice is within 2 errors of E8, either way, and under integer levels by 2 errors | either is not so |
| H4 | Llama-3.2, 4.5 bits, loud columns kept: the turned lattice is not above E8 by 2 errors | it is |
| H5 | Llama-3.2, 4.5 bits, loud columns kept: the turned lattice is under integer levels by 2 errors | it is not |
| H6 | One arm for all three. At 4.5 bits, on Qwen2.5, on SmolLM2 and on Llama-3.2 with loud columns kept, the turned lattice is the lowest of the three arms or within 2 errors of the lowest | it is not, on any |
| H7 | Weights only. The turned lattice's pooled weight error is within 0.1 dB of E8's, in all five cells with nothing kept: three models at 4.5 bits and two at 3.0 | it is not, in any |
| H8 | Weights only. At 4.5 bits, of the weights under 0.05 of their matrix's root mean square, the turned lattice stores as zero at least 15 points of a hundred more than E8 does, on all three models | it does not, on any |
| H9 | SmolLM2, 3.0 bits: the turned lattice is under E8 by 2 errors | it is not |

**The check C1.** `EBG128:b560s16` on SmolLM2 is scored again through this program, first. Its mean KL must be within a tenth of a standard error of the committed row's 0.060861. If it is not, the run stops there and nothing else is scored until the cause is found and written down.

My confidence. High in H7: it is the same lattice, and at these bits the error of a lattice hardly depends on how it is turned. Fair in H1, H3, H4, H8 and H9. Low in H2, H5 and H6. H2 needs the turned lattice to keep most of the gain at zero while it stores fewer weights as zero than integer levels do: its coordinates step by 0.71 of an integer step, so only weights under 0.35 of a step can go to zero. H5 needs Llama-3.2 to be like one of the other two models and not a third case, and that model has been a third case before.

The reasoning behind H1 to H5, so that it can be judged apart from the outcomes. If a model gains nothing at zero, as Qwen2.5, the turned lattice scores as E8 does, under integer levels. If a model gains at zero, as SmolLM2, the turned lattice gets E8's lower weight error and part of the gain at zero. Either way it is under integer levels.

**Reported without a prediction.**

- Llama-3.2 with nothing kept: E8 and the turned lattice against integer levels and against `Q4_K_M`. I do not know the sign. There the damage is in two columns, and the lattices differ in what they do to the weights beside a very large one.
- Qwen2.5 at 3.0 bits, all three arms. And at 3.0 bits on both models, the turned lattice against integer levels.
- How far each arm at 4.5 bits is under `Q4_K_M` with no matrix.
- The shares of weights stored as zero, in attention matrices and in feed-forward matrices apart.

## What it would mean

- **H1, H2 and H6 hold**: how E8 is turned was what cost it SmolLM2. The turned lattice is then the arm with no data to start from on all three models, in place of integer levels. It would need a packed format of its own: the format of this branch holds integer levels only.
- **H1 holds and H2 fails**: turning recovers part of what E8 lost and integer levels stay the arm for SmolLM2. No one lattice is best on all three.
- **H1 fails**: the made-up matrices of the pull request do not carry to this model. Either the turned lattice does not keep enough weights at zero in a ball at 4.5 bits, which H8 and the table of small weights will show, or the lattice-split record's account needs another look.
- **H3 fails because the turned lattice is above E8**: turning costs something where nothing is gained at zero, and the choice of lattice depends on the model.

## What this cannot show

- Three models between 1 and 2 billion weights. KL on one text. No task.
- Counted bits. No E8 arm has a file, turned or not.
- No importance matrix. The arms that weight columns were not run with the turned lattice.
- The pull request's own figure of 4.47 bits comes from packing the kept columns apart from the blocks. Here a kept column is set to zero inside its block, as in the loud-columns record, so the arms differ in the lattice only.
- Nothing about the other proposals of the pull request: the Leech lattice, the certificate for which columns to keep, or shells of varying size.
- Blocks of 8 are consecutive weights of a row. Another grouping was not tried.

## Order and time

One process at a time. The check on SmolLM2, then the six other rows at 4.5 bits, then the six at 3.0 bits, then the five weights-only rows. About six minutes a scored row and a minute and a half for one that is only built: an hour and a half.
