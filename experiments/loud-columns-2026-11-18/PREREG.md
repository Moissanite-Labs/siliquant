# Plan: the loud columns of `ffn_down`

Written 2026-11-18, before any variant of this plan was scored. It follows the two clip-by-layer records of the same day. It was drafted while the last rows of the SmolLM2 one were being scored and committed after that record, so it is written knowing those results.

## Question

Clipping the large entries of one `ffn_down` of SmolLM2 changes 387 entries of 16.8 million in layer 1 and costs a mean KL of 1.74. In layer 7 it changes 533 and costs 1.25. Every other layer costs under 0.02. On Qwen2.5 the clip's damage was in four matrices at the ends, and a look at the weights after the result found most of what the clip removes in about ten columns of each.

A column of `ffn_down` is one hidden unit of the feed-forward block. On gemma the same idea was tested and failed: the ten units with the most change carried 0.05 percent of the damage. Two questions.

**A. Is the clip's damage in those ten columns?** On Qwen2.5 that was a description made after the result. Here it is a test, on both models.

**B. Does it matter to rounding?** If a few hidden units carry that much, a rule that rounds every weight alike spends its error where it costs most. Does a ball arm at 4.5 bits gain by leaving ten columns of each `ffn_down` as they are?

## The loud columns

Of a matrix: the ten columns in which the clip at 5 standard deviations of a row removes the most squared weight. They are found from the weights alone. The limit, 5, is the one the output-budget record chose. The number, ten, is the one the Qwen2.5 record's description used. Neither is tuned here.

## What is run

**The models, the text and the score** are those of the model-output records: `Qwen2.5-1.5B-Instruct` and `SmolLM2-1.7B-Instruct`, 16-bit files, the wikitext-2 test text, 10,200 positions, mean KL against the reference's saved outputs.

**Part A, 8 variants.** The clip of one `ffn_down`, split in two: in its ten loud columns only (`/only10`), and everywhere but in them (`/but10`). On SmolLM2 for layers 1 and 7. On Qwen2.5 for layers 1 and 27. The whole clip of each is already scored in the clip-by-layer records.

**Part B, 6 variants.** The arm is `ZBG128:b560s16`: integer levels in one ball per group of 128 weights, 4.5 bits, no data. It is the arm both models agreed on. A kept column is set to zero before the rest of its matrix is rounded, so it takes no room in any ball, and is put back after. It is counted at 16 bits an entry and 16 bits for its number.

| Variant | Model | What |
|---|---|---|
| `ZBG128:b560s16+K10` | both | the ten loud columns of every `ffn_down` kept |
| `ZBG128:b560s16+K10#1,7` | SmolLM2 | kept in layers 1 and 7 only |
| `ZBG128:b560s16+S#1,7` | SmolLM2 | the `ffn_down` of layers 1 and 7 left whole. Not an arm at equal bits: it is the most that anything done to those two matrices can gain |
| `ZBG128:b560s16+A10` | SmolLM2 | the ten most active columns kept in place of the loud ones, by llama.cpp's importance matrix. This arm reads calibration data |
| `hZBG128:b560s16+K10` | SmolLM2 | the arm with columns weighted by the importance matrix, and the loud columns kept as well |

Already scored, in the model-output and imatrix-baseline records: `ZBG128:b560s16` at 0.0560 on SmolLM2 and 0.0227 on Qwen2.5, `hZBG128:b560s16` at 0.0282 on SmolLM2, and llama.cpp's `Q4_K_M` with an importance matrix at 0.0378 on SmolLM2.

**The program** is `run.py` of this folder. It calls the code of the earlier records for the clip and for the ball arm. A test on single matrices, before this plan: a matrix that keeps no column comes out bit for bit as from the earlier records' program, a kept column comes out equal to the reference, and the two parts of a clip together change exactly the entries the whole clip changes.

## What the weights say, before any score

`picks-smollm2.out`, `picks-qwen25.out` and `weights_only.jsonl`, committed with this plan. Nothing is scored in any of them.

**SmolLM2.**

- **In layers 1 and 7 the ten loud columns are nearly all of the matrix's input.** By llama.cpp's importance matrix they hold 98.5 and 99.0 percent of the input activity of the whole matrix, of 8,192 columns. 7 and 8 of the ten are among the ten most active columns. In no other layer do the loud columns hold more than 29.6 percent, and elsewhere the two picks share 0 to 5 columns.
- **They are a small part of the weights**: 0.18 and 0.22 percent of the matrix's squared weight.
- **The clip in them is a handful of entries.** 41 entries in layer 1 and 37 in layer 7, of 16.8 million each. Everywhere else in those matrices the clip touches 346 and 496.
- **Keeping them does not lower the weight error.** The ball arm's weight error on `ffn_down` is -24.93 dB with nothing kept and -24.94 with ten columns kept in every layer. So if `+K10` gains, it gains by where the error is and not by how much there is.
- **Bits.** `+K10` is 4.5047 bits a weight, `+K10#1,7` is 4.5002, `+A10` is 4.5047, `hZBG128:b560s16+K10` is 4.5096 and `+S#1,7` is 4.7394.

**Qwen2.5.** Another picture.

- **The loud columns are a large part of the weights and a small part of the input.** In layer 1 they hold 58.1 percent of the matrix's squared weight and 2.2 percent of its input activity. In layer 2, 30.3 and 5.9. In layers 26 and 27, 7.0 and 4.0 percent of the weight, 10.9 and 1.3 of the activity. In the other 24 layers they hold under 1 percent of the weight.
- **The two picks mostly differ.** In 17 of 28 layers no loud column is among the ten most active.
- **In layer 1 the clip in the ten loud columns is nearly the whole clip**: 5,097 entries and -3.21 dB, of the whole clip's -3.19. Everywhere else: 16,074 entries and -26.40 dB. In layer 27: 448 entries and -15.90 dB in the ten, 4,413 and -19.62 dB elsewhere.
- **Keeping them lowers the weight error a little.** `ffn_down` goes from -25.08 dB to -25.15, at 4.5051 bits.

**gemma.** Not run here. The output-budget record clipped the ten units with the most change in gemma's layer 10 and found 0.05 percent of that layer's damage in them. So on gemma the answer to question A is already no. The predictions below go the other way for SmolLM2 because of the first line above: there the ten columns are the matrix's input. For Qwen2.5 I have no such reason, and predict from the share of the clip that sits in the ten.

## Predictions

An error is the tool's standard error, and an error of a gap is the root of the sum of the two squares.

| | Prediction | It fails if |
|---|---|---|
| A1 | On SmolLM2, in layers 1 and 7, the clip everywhere but in the ten loud columns costs under a hundredth of the whole clip | either costs a hundredth or more |
| A2 | On SmolLM2, in layers 1 and 7, the clip in the ten loud columns alone costs at least 0.8 of the whole clip | either costs less |
| A3 | On Qwen2.5 layer 1, the clip everywhere but in the ten costs under a tenth of the whole clip | it costs a tenth or more |
| A4 | On Qwen2.5 layer 27, the last, the ten are not the whole story: the clip everywhere but in them costs more than a tenth of the whole clip | it costs a tenth or less |
| B1 | On SmolLM2, `ZBG128:b560s16+K10` is under `ZBG128:b560s16` by 2 errors of the gap, and by at least 20 percent | either is not so |
| B2 | Layers 1 and 7 are where it is: `+K10#1,7` gains at least half of what `+K10` gains | it gains less. Not scored if `+K10` is not under the plain arm by 2 errors |
| B3 | Ten columns are those two matrices: `+K10#1,7` is not above `+S#1,7` by 2 errors | it is |
| B4 | The weights find the units as well as the data does: `+A10` and `+K10` are within 2 errors of each other | they are not |
| B5 | On SmolLM2, `hZBG128:b560s16+K10` is under `hZBG128:b560s16` by 2 errors | it is not |
| B6 | On Qwen2.5 keeping the loud columns changes nothing that shows: `ZBG128:b560s16+K10` is within 2 errors of `ZBG128:b560s16` | it is not |

My confidence: high in A1 and A2. Fair in A3, B1, B2, B3 and B6. Low in A4, B4 and B5.

**Reported without a prediction.** Whether `ZBG128:b560s16+K10`, which reads no data, is level with the arms that read some: `Q4_K_M` with the matrix at 0.0378 and `hZBG128:b560s16` at 0.0282, on SmolLM2. I do not know. It turns on how much of what the importance matrix buys on this model is these two matrices.

## What it would mean

- **A1 and A2 hold**: on SmolLM2 the damage of the clip is the damage to ten hidden units in each of two matrices.
- **B1 holds**: part of what calibration data buys on this model can be had from the weights alone, at five thousandths of a bit a weight.
- **B1 fails with A1 and A2 holding**: those units matter to a rule that clips and not to one that rounds. The clip-by-layer records would then bear on clipping rules only, as their limits say.
- **B6 fails**: the gain is not a property of one model's two matrices.

## What this cannot show

- Why those units are what they are. Nothing here measures an activation.
- Anything about other numbers of columns than ten, other limits than 5, other kinds of matrix, or other bit points.
- Two models. A third could differ, as it did for the clip.
- A kept column is counted at 16 bits. No file is written, and `shape/ballpack.py` has no place for kept columns.

## Order and time

One scoring process at a time. SmolLM2 first: Part A, then Part B. Then Qwen2.5. About five minutes a variant, and under two more to build a ball arm: an hour and a half for 14.

## Extension 1: one column

Written 2026-11-18 after the plan above was committed at `566c318`, and before any variant of this extension was scored. One row of the plan was in when this was written: the clip in the ten loud columns of SmolLM2's layer 1 scores 1.449, which is 0.83 of the whole clip's 1.741.

**What prompted it.** `units-smollm2.out`, made by `units.py` after the plan and before any score. It was not planned and nothing in it is scored. It lists the ten loud columns one by one.

- In layer 1 of SmolLM2 **one column, 4987, holds 98.19 percent of the matrix's input activity**. It is 8,052 times as active as the mean column. The clip touches two of its 2,048 entries: its weights into dimensions 749 and 161 of the stream, at 26.7 and 17.5 standard deviations of their rows.
- In layer 7 one column, 7758, holds 95.68 percent. The clip touches two of its entries, into dimensions 227 and 423.
- The other nine loud columns hold 0.3 percent of the activity between them in layer 1 and 3.3 percent in layer 7.
- The same few dimensions of the stream come up again and again as where the large entries point: 227, 749, 161, 209, 737, 277 and 423.
- Qwen2.5 has nothing like it (`units-qwen25.out`): its most active column of layer 1 holds 3.31 percent. Its loud columns are large columns, up to 61 times the mean column's size, with hundreds of entries beyond the limit each.

So on SmolLM2 "ten columns" may be one. The weights alone do not single that one out among the ten. The importance matrix does.

**Three variants, all on SmolLM2.**

| Variant | What |
|---|---|
| `clip:5@ffn_down#1/act1` | the clip in the most active column of layer 1 only: two entries of 16.8 million, a weight error of -36.9 dB |
| `clip:5@ffn_down#7/act1` | the same in layer 7: two entries, -37.6 dB |
| `ZBG128:b560s16+A1#1,7` | the ball arm with one column kept in each of layers 1 and 7, the most active one. 4.4998 bits a weight, as the plain arm to four places. This arm reads calibration data |

`run.py` and `summary.py` were extended for these, and `weights_only.jsonl` has what each changes. The code of the planned variants is not changed: the test on single matrices was run again and gives the same matrices.

| | Prediction | It fails if |
|---|---|---|
| X1 | The one column is most of the ten: the clip in it alone costs at least 0.8 of the clip in the ten loud columns in layer 1, and at least 0.5 in layer 7 | either is lower |
| X2 | One kept column does what ten do: `ZBG128:b560s16+A1#1,7` is within 2 errors of `ZBG128:b560s16+K10#1,7` | it is not |

My confidence: fair in X1 for layer 1, low for layer 7, where four other loud columns are 30 to 106 times as active as the mean and point into the same dimensions. Fair in X2.

**What it would mean.** If X1 holds, changing two weights of 1.7 billion is enough to wreck this model, and those two can be found with an importance matrix and not, so far, without one. If X2 holds, whatever `+K10#1,7` gains is the gain of one hidden unit in each of two layers.

**Order.** After the 14 variants of the plan.
