# The clip of `ffn_down` on each layer of SmolLM2 alone

## Question

Clipping the large entries of one `ffn_down` at a time gave two different pictures. On gemma one matrix of 42, layer 10, carries 89.6 percent of the damage, and the weights do not find it. On Qwen2.5 two matrices at the two ends carry 70 percent, the weights name them, and the layers do not add up.

Which of the two is a third model like? Or neither?

This is run 3 of issue #20 on a third model.

## Status

`complete`, 2026-11-18. 25 variants: each of the 24 layers alone, and all at once. 2 of 5 predictions hold, and one of the two holds for a reason the plan did not give.

## Preregistration

[PREREG.md](PREREG.md), committed and pushed at `a6598fa` with `run.py`, `summary.py`, `weights_only.jsonl` and `columns.out`, before any variant was scored. The plan states what the weights say first. No program was changed after.

## Result in short

1. **Neither picture: two matrices, one at an end and one a third of the way in.** Layer 1 holds 56.9 percent of the summed damage and layer 7 holds 40.9. No other layer holds 1 percent.
2. **The clip costs most where it removes almost nothing.** In layer 1 it changes 387 entries of 16.8 million and removes a thousandth of the matrix's squared weight. That alone costs a mean KL of 1.74, and the model keeps its first choice 44 percent of the time. Rounding every block matrix of this model to 4.5 bits with llama.cpp's `Q4_0` costs 0.127.
3. **The weight error does not say which.** Over the 24 layers the rank correlation between what the clip removes and what it costs is 0.39. On Qwen2.5 it was 0.77. Layer 0 loses as much to the clip as layers 1 and 7 do, and costs about a hundredth of either.
4. **The activity of the columns it removes does.** In layers 1 and 7 the ten columns that lose most to the clip are 808 and 815 times as active as the mean column. In no other layer is that figure above 242. Its rank correlation with the cost is 0.90. This figure needs calibration text. It was in the plan before any score, without a prediction.
5. **All at once the model is gone**: a mean KL of 4.12, first choice kept 20 percent of the time. I predicted under 0.06.

So three models give three pictures. What they share: one or two matrices carry nearly all of the clip's damage. Where those are, and whether the weights find them, differs each time.

## What was run

**The rule** of the two earlier records: in a matrix, every entry beyond 5 standard deviations of its row is set to that limit. Nothing is rounded. Every other matrix stays as it is in the reference file.

**The variants.** `clip:5@ffn_down#L` for each layer L from 0 to 23, and `clip:5@ffn_down`: all 24 at once.

**The model, the text and the score** are those of the record model-output-smollm2-2026-11-17: `SmolLM2-1.7B-Instruct`, 16-bit file, the wikitext-2 test text, 10,200 positions, mean KL against the reference's saved outputs, with `llama-perplexity --kl-divergence` of llama.cpp at `eada89c`. A row of `ffn_down` has 8,192 entries and there are 2,048 rows.

**The program** is `run.py`. The rule's code is imported unchanged from the Qwen2.5 record.

**Activity.** A column of `ffn_down` is one hidden unit of the feed-forward block. Its activity is the mean squared size of its output on the wikitext-2 training text, from llama.cpp's importance matrix of the record imatrix-baseline-2026-11-17. `columns.out` has it for the ten columns of each layer that lose most to the clip.

## Results

Every number is in `results.out`, printed by `summary.py` from `results.jsonl`, `weights_only.jsonl` and `columns.out`.

### Layer by layer

The sum of the 24 mean KLs is 3.059.

| Layer | Weight error of the clip, dB | Entries clipped | Activity of its ten worst columns over the mean column | Mean KL | Share of the sum | First choice kept |
|---|---|---|---|---|---|---|
| 1 | -29.75 | 387 | 808 | 1.7406 | 56.9% | 44.5% |
| 7 | -28.45 | 533 | 815 | 1.2519 | 40.9% | 50.0% |
| 0 | -29.42 | 757 | 29 | 0.0161 | 0.5% | 94.3% |
| 23 | -31.28 | 996 | 80 | 0.0150 | 0.5% | 94.4% |
| 20 | -36.21 | 666 | 117 | 0.0085 | 0.3% | 96.7% |
| 22 | -31.17 | 430 | 242 | 0.0067 | 0.2% | 96.6% |
| 21 | -37.62 | 432 | 159 | 0.0067 | 0.2% | 96.9% |
| 6 | -33.61 | 584 | 12 | 0.0049 | 0.2% | 97.3% |
| the other 16 | -32.1 to -36.3 | 501 to 1,536 | 4 to 55 | 0.0001 to 0.0019 | 0.3% together | 97.9% to 99.4% |

Every one of the 24 is above zero by 20 of its own standard errors or more.

Layer 7 is 78 times layer 0, the third. So the order of the first two against the rest does not rest on the standard errors.

### What ranks the layers

| Figure of a layer | Needs | Rank correlation with its mean KL, 24 layers |
|---|---|---|
| Weight error of the clip | the weights | 0.389 |
| Activity of the ten columns that lose most to the clip | the weights and calibration text | 0.895 |

By weight error the first three layers are 7, 0 and 1, within 1.3 dB of each other. Two of them are the two that matter and the third costs a hundredth as much. Nothing in that figure tells them apart.

The activity figure does. It also puts layers 20 to 23 next, after a gap, which is where the next costs are. It does not explain layer 0, which is third by cost with a figure of 29.

### All at once

| | Weight error, dB | Mean KL | First choice kept |
|---|---|---|---|
| all 24 layers clipped | -33.69 | 4.1204 | 19.7% |
| the sum of the 24 alone | | 3.0590 | |

The sum is 0.742 of the whole. The clip touches 20,606 entries of 403 million in these 24 matrices.

### The three models side by side

The gemma figures are from the record output-budget-gemma-e4b-2026-11-17, whose reference was an 8-bit file.

| | gemma, 42 layers | Qwen2.5, 28 layers | SmolLM2, 24 layers |
|---|---|---|---|
| Largest share of the sum | 89.6%, layer 10 | 40.6%, layer 1 | 56.9%, layer 1 |
| Second largest | 78.4 times smaller | 29.8%, layer 27 | 40.9%, layer 7 |
| Place of the costliest layer by weight error | fourth | first | third |
| Rank correlation, weight error with KL | 0.753 | 0.766 | 0.389 |
| Mean KL with every layer clipped | 0.164 | 0.593 | 4.120 |
| The sum of the layers alone over that | 1.063 | 0.199 | 0.742 |

### The predictions

| | Prediction | Result | Holds |
|---|---|---|---|
| T1 | With all 24 layers clipped the mean KL is under 0.06 | 4.12 | **no** |
| T2 | Layers 0, 1, 22 and 23 together hold more than half of the sum | 58.1 percent | yes |
| T3 | No single layer holds more than half of the sum | layer 1 holds 56.9 percent | **no** |
| T4 | Spearman's rank correlation between weight error and KL over the 24 layers is at least 0.5 | 0.389 | **no** |
| T5 | The sum is between 0.7 and 1.4 times the KL of all at once | 0.742 | yes |

What I got wrong.

- **T1, by a factor of about 70.** I reasoned from the weight error: the clip removes a thousandth of each matrix here and half of one on Qwen2.5, so it should cost less here. It costs seven times more. The size of the change to the weights said nothing about the size of the change to the output, in either direction.
- **T4 is the same mistake** in another form.
- **T2 holds by its letter and not by its sense.** The four layers at the ends hold 58.1 percent because layer 1 alone holds 56.9. The other three hold 1.2 between them. This model is not "the ends": its second matrix is layer 7.
- **T5 holds for a reason the plan did not give.** The plan said small damage adds. The damage here is not small. With the first choice kept one time in five the KL has little room left to grow, so a ratio near one is not evidence that the layers add.

The plan did say that layer 7 "could be this model's layer 10", with low confidence. It half is.

## Deviations from the plan

None. One start, 25 rows in the planned order, no program changed after the plan.

## Limits

- **One model, one text, one limit, one kind of matrix.** The three models now differ in three ways, so a fourth could differ again.
- **The standard errors are the tool's** and treat positions as independent. The findings above rest on factors of 70 and more.
- **Activity is measured on text of the same source as the test text.** The importance matrix is from the wikitext-2 training text. How well it ranks the layers with other calibration text is not tested.
- **The activity comparison had no prediction.** It was named in the plan as something to report, and the figures were committed before any score. It is one comparison on one model.
- **Why those two matrices.** Not measured. A few hidden units that are hundreds of times as active as the rest, with very large weights into the stream, is what `columns.out` describes. Whether the damage is in those units is not tested here. On gemma it was tested and was not: the ten units with the most change carried 0.05 percent of layer 10's damage.
- **This bears on rules that clip.** The ball arms of this branch and the rule of `Q4_0` clip nothing. Whether those units matter to a rule that rounds is not tested here either.

## Read against the closure rule

Nothing is closed by a certificate. These are 25 measurements on one model.

**What changes in the repository's account.**

1. The Qwen2.5 record said the weights name the matrices, and suggested that a rule that clips could check its own weight error by matrix and leave alone the matrices where it is large. On this model that check finds nothing: the clip removes under two thousandths of the squared weight of every matrix, and the two matrices that cost 1.74 and 1.25 look like layer 0, which costs 0.016. The suggestion does not carry over.
2. "The weights say which matrix" is now true on one model of three. What ranked the layers here needed calibration text.
3. The cost of this clip is not bounded by anything seen so far. It was 0.16 on gemma, 0.59 on Qwen2.5 and 4.12 here, where the rule changes five entries in a hundred thousand.

**What follows.**

1. The plan loud-columns-2026-11-18 tests the two things this record leaves open: whether the damage is in the ten columns, and whether a ball arm at 4.5 bits gains by leaving those columns whole.
2. A rule that clips at a fixed number of standard deviations is not safe on a model it has not been scored on. That is the practical reading of the three records together.

## How to rerun

Needs NumPy, torch, llama.cpp's `gguf` package and `llama-perplexity`, the model file, its reference outputs on the test text, and its importance matrix for `columns.out`.

```bash
python3 experiments/clip-by-layer-smollm2-2026-11-18/run.py --model SmolLM2-1.7B-Instruct-f16.gguf --kld base.kld --text test --work DIR --out results.jsonl --bin llama-perplexity clip:5@ffn_down#1 clip:5@ffn_down
```

```bash
python3 experiments/clip-by-layer-smollm2-2026-11-18/run.py --model SmolLM2-1.7B-Instruct-f16.gguf --columns --imatrix smollm2.imatrix.gguf > columns.out
```

```bash
python3 experiments/clip-by-layer-smollm2-2026-11-18/summary.py > results.out
```

A variant takes five to six minutes to score on 4 CPU cores.

## Disclosure

Written by an AI agent session under the owner's direction, on 2026-11-18. The plan, the programs, the run and this record are the session's. The rule and the question are those of the record output-budget-gemma-e4b-2026-11-17 and of issue #20.
