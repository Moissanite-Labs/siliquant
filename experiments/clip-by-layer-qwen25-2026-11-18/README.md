# The clip of `ffn_down` on each layer of Qwen2.5 alone

## Question

The record output-budget-gemma-e4b-2026-11-17 clipped the large entries of one matrix at a time. On gemma one matrix of 42, the `ffn_down` of layer 10, carries 89.6 percent of the damage, and nothing about its large entries picks it out.

Does Qwen2.5 have such a matrix? And do the weights say which one it is?

This is run 3 of issue #20.

## Status

`complete`, 2026-11-18. 29 variants: each of the 28 layers alone, and all at once. 3 of 6 predictions hold. One analysis, of the weights by column, was not planned and is marked.

## Preregistration

[PREREG.md](PREREG.md), committed and pushed at `4ce900b` with `run.py` and `weights_only.jsonl`, before any variant was scored. The plan states what the weights say first. `summary.py` was committed at `9e078fa`, also before any variant was scored. Neither program was changed after.

## Result in short

1. **Not one matrix: two, at the two ends of the model.** Layer 1 holds 40.6 percent of the summed damage and layer 27, the last, holds 29.8. With their neighbours, layers 2 and 26, that is 83.3 percent in four matrices of 28. On gemma one matrix held 89.6 percent, and it was layer 10 of 42.
2. **Here the weights do say which.** The four layers where the clip changes the weights most are the four where it costs the output most. Over the 28 layers the rank correlation between the two is 0.77. On gemma no figure of the large entries found the matrix.
3. **The weights do not say how much.** Layer 27's matrix loses 11 dB less to the clip than layer 1's, and costs three quarters as much. Layer 2's loses 8 dB more than layer 27's and costs a quarter as much.
4. **The damage does not add up.** All 28 layers clipped at once cost a mean KL of 0.593. The 28 single-layer costs sum to 0.118, a fifth of it. On gemma the sum was 1.06 times the whole.
5. **Not planned: in each of those four matrices the damage sits in about ten hidden units.** Of 8,960 columns, the ten worst hold 70 to 99.5 percent of what the clip removes, and those ten units are 12 to 97 times as active as the mean unit.

So Qwen2.5 has such matrices, at its ends, and one can find them without scoring anything. And clipping 0.03 percent of the entries of one kind of matrix, with nothing rounded, costs about ten times what rounding every block matrix to 4.5 bits does by the rule of `Q4_0` (0.062).

## What was run

**The rule**, as in the gemma record: in a matrix, every entry beyond 5 standard deviations of its row is set to that limit. The standard deviation is of the whole row, with divisor n. Nothing is rounded. The matrix is written as 16-bit floats. Every other matrix stays as it is in the reference file.

**The variants.** `clip:5@ffn_down#L` for each layer L from 0 to 27, and `clip:5@ffn_down`: all 28 at once.

**The model, the text and the score** are those of the record model-output-qwen25-2026-11-17: `Qwen2.5-1.5B-Instruct`, 16-bit file, the wikitext-2 test text, 10,200 positions, mean KL against the reference's saved outputs, with `llama-perplexity --kl-divergence` of llama.cpp at `eada89c`.

**Two differences from the gemma run.** The reference there was an 8-bit file and here it is the 16-bit one. And a row of `ffn_down` has 8,960 entries here with 1,536 rows, where gemma has 10,240 and 2,560. The gemma record's own program refuses any file but gemma's, so the rule is written again in `run.py`, in the same three lines of NumPy.

## Results

Every number is in `results.out`, printed by `summary.py` from `results.jsonl`, or in `columns.out`.

### Layer by layer

The sum of the 28 mean KLs is 0.1177.

| Layer | Weight error of the clip, dB | Entries clipped | Mean KL | Share of the sum | First choice kept |
|---|---|---|---|---|---|
| 1 | -3.19 | 21,171 | 0.0478 | 40.6% | 90.8% |
| 27 | -14.36 | 4,861 | 0.0351 | 29.8% | 90.7% |
| 2 | -6.30 | 13,593 | 0.0093 | 7.9% | 95.2% |
| 26 | -12.76 | 6,874 | 0.0058 | 5.0% | 96.0% |
| 19 | -18.54 | 8,995 | 0.0037 | 3.1% | 97.3% |
| 24 | -25.50 | 2,242 | 0.0029 | 2.5% | 97.7% |
| 21 | -22.60 | 5,178 | 0.0024 | 2.0% | 98.0% |
| 25 | -23.85 | 2,975 | 0.0023 | 2.0% | 97.8% |
| the other 20 | -23.0 to -30.4 | 1,459 to 6,021 | 0.0001 to 0.0009 | 7.1% together | 98.5% to 99.5% |

Every one of the 28 is above zero by 14 of its own standard errors or more. So even the smallest is a measured cost and not noise.

The five layers the plan named from the weights, 1, 2, 26, 27 and 19, hold 86.4 percent of the sum.

### All at once

| | Weight error, dB | Mean KL | First choice kept |
|---|---|---|---|
| all 28 layers clipped | -23.25 | 0.5926 | 63.2% |
| the sum of the 28 alone | | 0.1177 | |

The sum is 0.199 of the whole. Clipping the first layers and the last together costs five times what they cost apart. The plan allowed for this ("if the damage is large the layers will not add") and predicted the other way.

For scale, on the same text: llama.cpp's `Q4_0` scores 0.0623 and its `Q4_K_M` 0.0333. The clip touches 121,152 entries of 385 million in these 28 matrices and rounds nothing.

### Not planned: where the clipped entries sit

`columns.py`, written after the scores were in. Nothing is scored. A column of `ffn_down` is one hidden unit of the feed-forward block. A unit's activity is the mean squared size of its output on the wikitext-2 training text, from llama.cpp's importance matrix of the record imatrix-baseline-2026-11-17.

| Layer | Share of what the clip removes, in its 10 worst columns | In its 100 worst | Activity of those 10 units over the mean unit |
|---|---|---|---|
| 1 | 99.5% | 99.9% | 19.9 |
| 2 | 98.8% | 99.6% | 53.1 |
| 26 | 90.0% | 99.4% | 97.2 |
| 27 | 70.2% | 96.3% | 12.0 |
| the other 24 | 19.8% to 76.1% | 73.5% to 98.0% | 0.6 to 22.2 |

In the four matrices that carry the damage, what the clip removes is the outgoing weights of about ten hidden units, and those units are far more active than the rest. That is a description. It was looked for after the result, on one model, and it is not a test: layer 3's ten worst units are 22 times as active as the mean and its clip costs 0.0003.

### The predictions

| | Prediction | Result | Holds |
|---|---|---|---|
| L1 | With all 28 layers clipped the mean KL is above 0.05 | 0.593 | yes |
| L2 | One layer holds more than half of the sum | the largest holds 40.6 percent | **no** |
| L3 | The layer with the largest KL is layer 1, the one the weights point at | layer 1 | yes |
| L4 | Layers 1, 2, 26, 27 and 19 hold at least 90 percent of the sum | 86.4 percent | **no** |
| L5 | The sum is between 0.7 and 1.4 times the KL of all at once | 0.199 | **no** |
| L6 | Spearman's rank correlation between weight error and KL over the 28 layers is at least 0.5 | 0.766 | yes |

What I got wrong. I expected one matrix, as on gemma, and the weights' loudest. The weights' loudest is the largest, and the last layer is close behind it with a tenth of its weight error. I had given the layers' adding up low confidence and still put the bounds where gemma's figure was.

## Deviations from the plan

- **The run was started twice.** The machine restarted while the first variant was being scored. `results.start` has both starts. No row was lost.
- **`columns.py` was not planned.** It was written with all 29 variants scored, and its output is marked above.
- Nothing else. No program was changed after the plan.

## Limits

- **One model, one text, one limit, one kind of matrix.** Nothing here is about other limits than 5 standard deviations, about the other six kinds, or about another model. gemma and Qwen2.5 differ in the answer, so a third model could differ again.
- **The standard errors are the tool's** and treat positions as independent. The order of layers within 2 errors of each other is not established. The findings above rest on factors of 4 and more.
- **Why the ends.** Not measured. The units found by `columns.py` are a lead: a few very active units with very large outgoing weights. Whether they are the same kind of thing in layer 1 and in layer 27, and what gemma's layer 10 has in their place, is open.
- **Why the damage compounds.** Not measured. The single-layer figures cannot be added to predict a rule that clips everywhere, on this model.
- **This bears on rules that clip.** The ball arms of this branch clip nothing, and `X32`, the rule of `Q4_0`, clips nothing either.

## Read against the closure rule

Nothing is closed by a certificate. These are 29 measurements on one model.

**What changes in the repository's account.** The output-budget record found one matrix on gemma and no figure of the weights that finds it. On Qwen2.5 the damage is at both ends of the model in four matrices, and the plainest figure of the weights, how much the clip changes a matrix, ranks them. So "the weights do not say which matrix" is a statement about gemma, not about models. And "the single-layer costs add up" is a statement about gemma too.

**What follows.**

1. A rule that clips can check its own weight error by matrix before anything is scored, and leave alone the matrices where it is large, or keep the large entries of a few columns apart. On this model that list would have been right. This is a suggestion, not a tested rule.
2. The same 29 variants on a third model would say whether "the ends" or "one layer in the middle" is the usual case.
3. The fourth run of issue #20, the activations that gemma's layer 10 multiplies, has a counterpart here in `columns.py`. Running that program's figures on gemma needs the seat that holds gemma and its importance matrix.

## How to rerun

Needs NumPy, torch, llama.cpp's `gguf` package and `llama-perplexity`, the model file and its reference outputs on the test text.

```bash
python3 experiments/clip-by-layer-qwen25-2026-11-18/run.py --model qwen2.5-1.5b-instruct-fp16.gguf --kld base.kld --text test --work DIR --out results.jsonl --bin llama-perplexity clip:5@ffn_down#1 clip:5@ffn_down
```

```bash
python3 experiments/clip-by-layer-qwen25-2026-11-18/summary.py > results.out
```

```bash
python3 experiments/clip-by-layer-qwen25-2026-11-18/columns.py --model qwen2.5-1.5b-instruct-fp16.gguf --imatrix qwen15.imatrix.gguf > columns.out
```

A variant takes about five and a half minutes to score on 4 CPU cores.

## Disclosure

Written by an AI agent session under the owner's direction, on 2026-11-18. The plan, the programs, the run and this record are the session's. The rule and the question are those of the record output-budget-gemma-e4b-2026-11-17 and of issue #20.
