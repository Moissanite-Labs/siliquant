# Bits split by kind of matrix, fitted on one text and scored on another

## Question

The record output-budget-gemma-e4b-2026-11-17 split the bits over the seven kinds of block matrix by what each kind's rounding costs the output. At 4.5 bits the split scored 31.4 percent under equal bits, on the text it was fitted on.

Does a split fitted on one text hold on a text it never read? On another model?

This is run 1 of issue #20.

## Status

`complete`, 2026-11-18. 30 variants on `Qwen2.5-1.5B-Instruct`: 22 to fit on the wikitext-2 validation text, 8 to test on it and on the test text. All 5 predictions hold.

## Preregistration

[PREREG.md](PREREG.md), committed and pushed at `4ce900b`, before any variant was scored. `summary.py` at `9e078fa`, also before. The split itself, `allocate.out`, was committed at `db8e458`, three seconds before the first variant of Part B started. No program was changed after the plan. `allocate.py` is the gemma record's, unchanged.

## Result in short

1. **The split holds on the text it never read.** At 4.5 bits it is 17.4 percent under equal bits on the validation text, where it was fitted, and 17.9 percent under on the test text. 5.7 errors of the gap on the test text.
2. **At 3.5 bits it holds at four fifths.** 25.8 percent on the text of the fit and 20.4 on the other, 10.1 errors.
3. **The control is a control here.** The same level counts put on the kinds in the order the fit predicts is worst score 2.6 percent above equal bits on the test text, and 7.1 errors above the split. On gemma that arm also beat equal bits.
4. **The gain is smaller than on gemma**: 17 to 18 percent against 31.4. And it is small beside the gain of a ball: on this model and text one ball per group takes the KL at 4.5 bits from 0.0620 to 0.0227, which is 63 percent, with no fit.
5. **What the split does on this model**: 64 levels for `attn_k` and `attn_v`, which are 1.7 percent of the weights between them, 46 for `attn_output`, and 12 for `ffn_gate`. Everything else stays at 16.

So the limit the gemma record stated, that its split was fitted on the scored text, is not what made it work, as far as two halves of one source on another model can say.

## What was run

**The model.** `Qwen2.5-1.5B-Instruct`, the 16-bit file of the record model-output-qwen25-2026-11-17.

**Two texts.** The fit reads the wikitext-2 validation text. The test is on the wikitext-2 test text. 40 passages of 512 tokens of each, the second half scored: 10,200 positions. Both are Wikipedia prose of one source, so this is the mildest held-out text there is.

**The rule.** `X32:Q`, the rule of `Q4_0`: one scale per 32 weights from the largest entry, Q levels, log2(Q) + 0.5 bits.

**Part A, validation text.** Each of the seven kinds rounded alone at 8, 16 and 32 levels with every other matrix left as in the reference, and all kinds at 8 levels: 22 variants.

**The split.** `allocate.py` of the gemma record on those rows. It reads a kind's KL at any level count off its three points, adds the kinds up, and takes the best of 9^7 choices under the budget. The reversed arm is the best arm's level counts in the order it predicts is worst.

**Part B.** On the test text: the best arm at 4.5 bits, the reversed arm, `X32:16`, the best arm at 3.5 bits and `X32:8`. On the validation text: the best arm at 4.5 bits, `X32:16` and the best arm at 3.5 bits.

**The program** is `run.py` of the record clip-by-layer-qwen25-2026-11-18. The score is as in the model-output records.

## Results

Every number is in `results.out`, printed by `summary.py` from `results.jsonl` and `allocate.out`.

### Part A: what each kind costs, on the validation text

| Kind | Share of the block weights | Mean KL at 8 levels | At 16 | At 32 |
|---|---|---|---|---|
| `attn_q` | 5.0% | 0.0075 | 0.0019 | 0.0005 |
| `attn_k` | 0.8% | 0.0139 | 0.0033 | 0.0008 |
| `attn_v` | 0.8% | 0.0229 | 0.0062 | 0.0016 |
| `attn_output` | 5.0% | 0.0461 | 0.0079 | 0.0019 |
| `ffn_gate` | 29.4% | 0.0377 | 0.0094 | 0.0023 |
| `ffn_up` | 29.4% | 0.0465 | 0.0118 | 0.0029 |
| `ffn_down` | 29.4% | 0.0574 | 0.0145 | 0.0036 |

`attn_v` is under one percent of the weights and at 16 levels costs half of what `ffn_up` does with 29 percent. That is what the split spends bits on.

The seven kinds alone at 8 levels add up to 0.232. All at once score 0.273: the sum is 0.85 of the whole. So on this rule the kinds nearly add, where the layers under the clip did not (the record clip-by-layer-qwen25-2026-11-18).

### The split

| Budget | Arm | `attn_q` | `attn_k` | `attn_v` | `attn_output` | `ffn_gate` | `ffn_up` | `ffn_down` | Mean bits | KL it predicts |
|---|---|---|---|---|---|---|---|---|---|---|
| 4.5 | best | 16 | 64 | 64 | 46 | 12 | 16 | 16 | 4.488 | 0.0464 |
| 4.5 | reversed | 46 | 64 | 64 | 16 | 16 | 16 | 12 | 4.488 | 0.0556 |
| 3.5 | best | 8 | 46 | 64 | 22 | 6 | 8 | 8 | 3.498 | 0.1834 |

On gemma the best arm at 4.5 bits was 32, 64, 64, 46, 12, 12, 16.

### Part B: the fit and the test

| Text | Budget | Arm | Bits | Mean KL | First choice kept | Gain over equal bits | Lower than equal bits by, in errors |
|---|---|---|---|---|---|---|---|
| validation, the fit | 4.5 | equal bits, `X32:16` | 4.500 | 0.0572 | 88.6% | | |
| validation, the fit | 4.5 | best | 4.488 | 0.0473 | 89.6% | 17.4% | 6.2 |
| **test** | 4.5 | equal bits, `X32:16` | 4.500 | 0.0620 | 88.2% | | |
| **test** | 4.5 | best | 4.488 | 0.0509 | 88.8% | **17.9%** | 5.7 |
| **test** | 4.5 | reversed | 4.488 | 0.0637 | 87.5% | -2.6% | -0.8 |
| validation, the fit | 3.5 | equal bits, `X32:8` | 3.500 | 0.2734 | 75.9% | | |
| validation, the fit | 3.5 | best | 3.498 | 0.2028 | 79.0% | 25.8% | 12.2 |
| **test** | 3.5 | equal bits, `X32:8` | 3.500 | 0.2790 | 75.0% | | |
| **test** | 3.5 | best | 3.498 | 0.2222 | 78.3% | **20.4%** | 10.1 |

`X32:16` on the test text scores 0.062043 here, the figure of the record model-output-qwen25-2026-11-17 to the last digit, as it should: it is the same rule through the same code.

The fit's own forecast on the validation text was 0.0464 at 4.5 bits, and the arm scored 0.0473. At 3.5 bits it forecast 0.1834 and the arm scored 0.2028: there the kinds do not quite add, as Part A's sum showed.

### The predictions

| | Prediction | Result | Holds |
|---|---|---|---|
| H1 | On the test text the best arm at 4.5 bits is under `X32:16` by 2 errors | 0.0509 against 0.0620, 5.7 errors | yes |
| H2 | Its gain on the test text is at least half its gain on the validation text | 17.9 percent against 17.4 | yes |
| H3 | On the test text the best arm at 3.5 bits is under `X32:8` by 2 errors | 0.2222 against 0.2790, 10.1 errors | yes |
| H4 | On the test text the reversed arm at 4.5 bits is above the best arm by 2 errors | 0.0637 against 0.0509, 7.1 errors | yes |
| H5 | In the best arm at 4.5 bits every attention kind has at least as many levels as every feed-forward kind | the fewest of an attention kind is 16, the most of a feed-forward kind is 16 | yes |

H5 holds at equality: `attn_q` gets 16 levels, no more than `ffn_up` and `ffn_down`.

## Deviations from the plan

- **Part A was started twice.** The machine stopped for 37 minutes after the first variant. `results.start` has both starts. No row was lost.
- Nothing else.

## Limits

- **The mildest held-out text.** Both texts are Wikipedia prose from one source. A split fitted on prose and scored on code, or on dialogue, could behave otherwise. That is not tested.
- **One model.** On gemma the gain was 31.4 percent and the split differed: it gave `attn_q` 32 levels and `ffn_up` 12.
- **One rule.** The split is of level counts under `X32`. Nothing here says what a split would buy under a ball arm.
- **By kind only.** A split by layer could gain more. The clip-by-layer record found the first and last layers of this model to be the costly ones under clipping, which is a hint and not a measurement for this rule.
- **The standard errors are the tool's** and treat positions as independent. The gains above are 5.7 errors and more. The reversed arm against equal bits, 0.8 errors, is a tie.
- **The 64-level choices extrapolate.** Part A measures 8, 16 and 32 levels, and the split uses 46 and 64 for three kinds, as on gemma.

## Read against the closure rule

Nothing is closed by a certificate. These are 30 measurements on one model and two texts.

**What changes in the repository's account.** The gemma record's limit, "the split is fitted on the text it is scored on", can be read with this beside it: on another model, between two halves of one source, a fitted split gave the same gain on the half it never read. The gemma figure itself is not retested.

**What follows.**

1. Run 2 of issue #20 asks whether the gain of a ball and the gain of the split add. It needs no gemma file to ask: a ball arm whose bits differ by kind can be built on this model with the split's level counts turned into bits. The ball gains 63 percent and the split 18, so the most the split could add is small beside what the ball already has.
2. A held-out text of another kind would be the real test of the fit.

## How to rerun

Needs NumPy, torch, llama.cpp's `gguf` package and `llama-perplexity`, the model file and its reference outputs on both texts.

```bash
python3 experiments/bit-split-held-out-qwen25-2026-11-18/run.py --model qwen2.5-1.5b-instruct-fp16.gguf --kld valid.kld --text valid --work DIR --out results.jsonl --bin llama-perplexity X32:8@attn_q X32:16@attn_q X32:32@attn_q
```

```bash
python3 experiments/output-budget-gemma-e4b-2026-11-17/allocate.py --results results.jsonl > allocate.out
```

```bash
python3 experiments/bit-split-held-out-qwen25-2026-11-18/summary.py > results.out
```

A variant takes about five and a half minutes on 4 CPU cores.

## Disclosure

Written by an AI agent session under the owner's direction, on 2026-11-18. The plan, the run and this record are the session's. The rule, the allocation program and the question are those of the record output-budget-gemma-e4b-2026-11-17 and of issue #20.
