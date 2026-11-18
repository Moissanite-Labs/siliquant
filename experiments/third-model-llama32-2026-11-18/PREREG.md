# Plan: a third model for the headline comparisons and for the question of the matrix

Written 2026-11-18, before anything was run on the model. Its file was fetched and its SHA-256 checked. Nothing else.

## Question

The headline comparisons of this branch rest on two models, `Qwen2.5-1.5B-Instruct` and `SmolLM2-1.7B-Instruct`. And the two disagree on what an importance matrix is worth to the ball arm.

- On SmolLM2 the ten loudest dimensions of the stream hold 61.8 percent of what the attention reads. A matrix halves the ball arm's KL, on prose and on Python source, whether it was made from Wikipedia or from noise.
- On Qwen2.5 they hold 11.4 percent. A matrix from Wikipedia helps the ball arm a little on Wikipedia prose and costs it a little on Python source. A matrix from noise does nothing on prose and costs a little on Python.

So the last two records point at a rule: weight the columns where the stream has loud dimensions, and not where it has few. Two models cannot test a rule they suggested.

**Do the comparisons hold on a model of a third family? And does the rule call it right?**

## What is run

**The model.** `Llama-3.2-1B-Instruct`, the 16-bit file `Llama-3.2-1B-Instruct-f16.gguf` of the `bartowski/Llama-3.2-1B-Instruct-GGUF` repository: 2,479,595,360 bytes, SHA-256 starts `1f33ad43`. 16 layers, a stream of 2,048 dimensions. Another maker and another family than the two before.

**Two texts**, both of earlier records: the wikitext-2 test text, and the Python file `conversion/base.py` of llama.cpp at `eada89c`. 10,200 positions each. The reference is the 16-bit model's saved outputs on each.

**Two matrices**, made as in the earlier records: one by `llama-imatrix` from 100 passages of the wikitext-2 training text, and one from 100 passages of a text with no meaning, 80,000 pieces of this model's vocabulary drawn at random by `randtext.py` with seed 1.

**Six variants on each text, twelve in all.** llama.cpp's `Q4_K_M` and the ball arm `ZBG128:b560s16`, each with no matrix, with the wikitext matrix, and with the matrix of no meaning. The weighted ball arm is `hZBG128:b560s16` at the strength 0.25 of the record imatrix-baseline-2026-11-17. The `Q4_K_M` files are made by the `llama-quantize` command of that record.

**The programs** are those of the earlier records. `run_arms.py` and `run_rand.py` of this folder add this model's file to the files those programs accept, and nothing else.

**Before any score** three files are committed: `reference.out` and `matrices.out`, what the tools printed when the reference outputs and the two matrices were made, and `forecast.out`, the table of the record imatrix-gain-parts-2026-11-18 for this model. Nothing is scored in any of them. `forecast.out` has the one figure the rule needs: the share of what the attention reads that sits in its ten most active dimensions, by the wikitext matrix, in the mean over the layers. Call it the share.

## Predictions

An error is the tool's standard error, and an error of a gap is the root of the sum of the two squares.

| | Prediction | It fails if |
|---|---|---|
| L1 | On the wikitext text, with no matrix on either side, the ball arm at 4.5 bits is under `Q4_K_M` by 2 errors | it is not |
| L2 | The same on the Python text | it is not |
| L3 | On the wikitext text, with the wikitext matrix on both sides, the ball arm is under `Q4_K_M` by 2 errors | it is not |
| L4 | On the wikitext text, the ball arm with the matrix of no meaning is under `Q4_K_M` with the wikitext matrix by 2 errors | it is not |
| L5 | The rule. If the share is 40 percent or more: each of the two matrices puts the ball arm under the plain ball arm by 2 errors, on both texts. If the share is 20 percent or less: on the Python text neither matrix puts the ball arm under the plain arm by 2 errors. Between 20 and 40 the rule says nothing and L5 is not scored | the case the share selects does not come out |
| L6 | On the wikitext text the matrix of no meaning buys `Q4_K_M` at least half of what the wikitext matrix buys it | its share of that gain is under half |

My confidence: fair in L1 and L2. The same comparison was 9 to 15 errors on the two earlier models and both texts. Fair in L3, L4 and L6. L5 is the test, and I do not know which case the share will select. The two thresholds are not fitted: they are a wide margin on each side of a gap between 11.4 and 61.8, chosen before this model was looked at.

**Reported without a prediction.** Whether a matrix helps `Q4_K_M` on the Python text. Where this model's forecast puts the gain, by kind of matrix.

## What it would mean

- **L1 to L4 hold**: the headline lines hold on three models of three families.
- **Any of them fails**: that line is a statement about two models, and the pull request must say so.
- **L5 holds**: the share is a figure that can be read from one run of `llama-imatrix`, on noise if need be, and says whether to weight.
- **L5 fails**: the rule was two cases and not a rule.
- **L5 is not scored**: this model sits between the two, and what it does is a third case to describe.

## What this cannot show

- Three models are still three, all between 1 and 2 billion weights.
- The thresholds of L5 are a guess with a wide margin. One model can refute the rule. It cannot place the line.
- KL on two texts. No task.
- The counted arm with groups of 128. No file is written for this model.

## Order and time

One process at a time: the two reference passes, the two matrices, the forecast, the three `Q4_K_M` files, then six variants on the wikitext text and six on the Python text. About an hour and a quarter.
