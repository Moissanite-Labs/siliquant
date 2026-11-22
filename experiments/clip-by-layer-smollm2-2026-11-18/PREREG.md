# Plan: the clip of `ffn_down` on each layer of SmolLM2 alone

Written 2026-11-18, before any variant of this plan was scored. A third model for the question of run 3 of issue #20.

## Question

Clipping the large entries of one `ffn_down` at a time gave two different pictures. On gemma one matrix of 42, layer 10, carries 89.6 percent of the damage, and the weights do not find it. On Qwen2.5 two matrices at the two ends of the model carry 70 percent, the weights name them, and the layers do not add up (the record clip-by-layer-qwen25-2026-11-18).

Which of the two is a third model like? Or neither?

## What is run

**The rule** of those two records: every entry of a matrix beyond 5 standard deviations of its row is set to that limit. Nothing is rounded. Every other matrix stays as in the reference.

**The variants.** `clip:5@ffn_down#L` for each layer L from 0 to 23, and `clip:5@ffn_down`: all 24 at once. In layer order, then the one with all.

**The model, the text and the score** are those of the record model-output-smollm2-2026-11-17: `SmolLM2-1.7B-Instruct`, 16-bit file, the wikitext-2 test text, 10,200 positions, mean KL against the reference's saved outputs. A row of `ffn_down` has 8,192 entries and there are 2,048 rows.

**The program** is `run.py` of this folder. The rule's code is imported unchanged from the Qwen2.5 record.

## What the weights say, before any score

`weights_only.jsonl` and `columns.out`, committed with this plan. Nothing is scored in either.

- **The clip removes little from any matrix of this model.** Its weight error is -28.5 dB at the worst (layer 7) and -37.6 at the best. On Qwen2.5 it was -3.2 dB in layer 1. It clips 387 to 1,536 entries a layer, where Qwen2.5 had 1,459 to 21,171.
- **The five layers it changes most** are 7, 0, 1, 22 and 23, at -28.5, -29.4, -29.8, -31.2 and -31.3 dB. Four of the five are the first two and the last two layers.
- **By column.** In layers 1 and 7 the ten columns that lose most are 800 times as active as the mean column, by llama.cpp's importance matrix of this model. In layers 20 to 22 they are 117 to 242 times as active. Elsewhere 4 to 80 times.

So the weights point less loudly here, and at two things that disagree in part: weight error says 7, 0, 1, and activity says 1, 7, 22.

## Predictions

The sum is the sum of the 24 single-layer mean KLs. An error is the tool's standard error.

| | Prediction | It fails if |
|---|---|---|
| T1 | The clip costs far less here than on Qwen2.5: with all 24 layers clipped the mean KL is under 0.06, a tenth of Qwen2.5's 0.593 | it is 0.06 or more |
| T2 | The ends: layers 0, 1, 22 and 23 together hold more than half of the sum | they hold half or less |
| T3 | No single layer holds more than half of the sum | one does |
| T4 | The weights say which: Spearman's rank correlation between the clip's weight error and its KL over the 24 layers is at least 0.5 | it is lower |
| T5 | Small damage adds: the sum is between 0.7 and 1.4 times the KL of all 24 at once | it is outside |

My confidence: fair in T1, T4 and T5, low in T2 and T3. Layer 7 is not at an end, and it has both the largest weight error and columns 800 times as active as the mean: it could be this model's layer 10.

**Reported without a prediction.** Whether a layer's rank in KL follows the activity of its worst columns better than it follows its weight error. On Qwen2.5 the columns were looked at after the result. Here they are on the table first, and I do not know which of the two figures the output follows.

## What it would mean

- If T2 holds and T3 holds, this model is like Qwen2.5: the ends, and more than one matrix.
- If T3 fails with layer 7, it is like gemma in having one matrix in the middle, and unlike gemma in that the weights point at it.
- If T5 holds here where it failed on Qwen2.5, the failure to add there goes with the size of the damage, which is what one would expect of small changes.

## What this cannot show

- Why a matrix is the one.
- Anything about other limits than 5 standard deviations, or other kinds of matrix.
- The order of layers whose KL is within 2 errors of each other. With damage this small, many may be.

## Order and time

One scoring process at a time. A variant takes about five minutes: two hours for 25.
