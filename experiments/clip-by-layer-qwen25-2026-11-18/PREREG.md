# Plan: the clip of `ffn_down` on each layer of Qwen2.5 alone

Written 2026-11-18, before any variant of this plan was scored. This is run 3 of issue #20, claimed there for the seat that holds the Qwen2.5 file.

## Question

The record output-budget-gemma-e4b-2026-11-17 clipped the large entries of one matrix at a time. On gemma one matrix of 42, the `ffn_down` of layer 10, carries 89.6 percent of the damage, and nothing about its large entries picks it out.

Does Qwen2.5 have such a matrix? And do the weights say which one it is?

## What is run

**The rule**, as in that record: in a matrix, every entry beyond 5 standard deviations of its row is set to that limit. The standard deviation is of the whole row, with divisor n. Nothing is rounded. The matrix is written as 16-bit floats. Every other matrix stays as it is in the reference file.

**The variants.** `clip:5@ffn_down#L` for each layer L from 0 to 27: 28 variants, one matrix changed in each. And `clip:5@ffn_down`: all 28 at once. In layer order, then the one with all.

**The model, the text and the score** are those of the record model-output-qwen25-2026-11-17: `Qwen2.5-1.5B-Instruct`, 16-bit file, the wikitext-2 test text, 10,200 positions, mean KL against the reference's saved outputs.

**The program** is `run.py` of this folder. That record's own program refuses any file but gemma's, so the rule is written again here in the same three lines of NumPy.

**Two differences from the gemma run.** The reference there was an 8-bit file and here it is the 16-bit one. And in Qwen2.5 a row of `ffn_down` has 8,960 entries and there are 1,536 rows, where gemma has 10,240 and 2,560.

## What the weights say, before any score

`weights_only.jsonl`, committed with this plan: the rule applied to each layer, nothing scored.

| Layer | Weight error of the clip, dB | Entries clipped | Rows with a clipped entry, of 1,536 | Largest entry, in standard deviations of its row |
|---|---|---|---|---|
| 1 | -3.19 | 21,171 | 1,536 | 81.0 |
| 2 | -6.30 | 13,593 | 1,535 | 56.8 |
| 26 | -12.76 | 6,874 | 1,397 | 71.5 |
| 27 | -14.36 | 4,861 | 670 | 47.6 |
| 19 | -18.54 | 8,995 | 446 | 48.4 |
| the other 23 | -22.6 to -30.4 | 1,459 to 6,021 | 323 to 629 | 29.5 to 68.8 |

In layer 1 the clip removes about half of the matrix's energy, and it touches every row. That is not a few stray entries. It reads like a few hidden units whose outgoing weights are large in every row, which this plan does not check.

So on this model the weights point at a matrix, loudly: layer 1, then layer 2. On gemma they did not.

## Predictions

The sum is the sum of the 28 single-layer mean KLs. An error is the tool's standard error.

| | Prediction | It fails if |
|---|---|---|
| L1 | The clip does damage: with all 28 layers clipped the mean KL is above 0.05 | it is not |
| L2 | One layer holds more than half of the sum | none does |
| L3 | That layer is layer 1, the one the weights point at | the layer with the largest KL is another |
| L4 | The five layers with the largest weight error (1, 2, 26, 27, 19) hold at least 90 percent of the sum | they hold less |
| L5 | The sum is between 0.7 and 1.4 times the KL of all 28 at once | it is outside |
| L6 | Over the 28 layers, Spearman's rank correlation between the clip's weight error and its KL is at least 0.5 | it is lower |

My confidence: high in L1, fair in L2, L4 and L6, and low in L3 and L5. Layer 1's matrix may hold large weights on units that are seldom active, and then clipping them costs little. And if the damage is large the layers will not add.

## What it would mean

- If L2 and L3 hold, Qwen2.5 has such a matrix too, and here the weights name it. A rule that clips would have to leave that matrix alone or keep its large entries apart, and it could know that without scoring anything.
- If L2 holds and L3 fails, it is as on gemma: one matrix, and the weights do not say which.
- If L2 fails, gemma's layer 10 is not a general thing.
- Either way, the ball arms of this branch clip nothing, so this bears on rules that clip and on what the output-budget record found, not on the ball arms.

## What this cannot show

- Why a matrix is the one. The activations it multiplies are not measured here. The importance matrix of this model exists (`qwen15.imatrix.gguf` of the importance-matrix record) and could say, in a later plan.
- Anything about other limits than 5 standard deviations, other kinds of matrix, or other models.
- The order of layers whose KL is within 2 errors of each other.

## Order and time

One scoring process at a time. A variant takes about five and a half minutes to score: about two and a half hours. Rows go to `results.jsonl` as they land. The run is queued behind the packed-format run and the MMLU subset on Qwen2.5.
