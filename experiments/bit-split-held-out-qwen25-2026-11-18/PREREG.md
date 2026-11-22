# Plan: bits split by kind of matrix, fitted on one text and scored on another

Written 2026-11-18, before any variant of this plan was scored. This is run 1 of issue #20, claimed there for the seat that holds the Qwen2.5 file and the wikitext-2 texts.

## Question

The record output-budget-gemma-e4b-2026-11-17 split the bits over the seven kinds of block matrix by what each kind's rounding costs the output. At 4.5 bits the split scored 31.4 percent under equal bits. Its limits say that the split was fitted on the text it was scored on.

Does a split fitted on one text hold on a text it never read? On another model?

## What is run

**The model.** `Qwen2.5-1.5B-Instruct`, the 16-bit file of the record model-output-qwen25-2026-11-17.

**Two texts.** The fit reads the wikitext-2 **validation** text, SHA-256 `4cd0f6876d07a413aa911261ff6d363c72d757d47f0fdd6015702014c89cb9c7`. The test is on the wikitext-2 **test** text, SHA-256 `173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`. 40 passages of 512 tokens of each, the second half scored. The reference outputs on both exist from earlier records. Both texts are Wikipedia prose of one source, so this is the mildest test of a held-out text.

**The rule.** `X32:Q`, the rule of `Q4_0`: one scale per 32 weights from the largest entry, Q levels, log2(Q) + 0.5 bits.

**Part A, on the validation text.** `X32:8@KIND`, `X32:16@KIND` and `X32:32@KIND` for each of the seven kinds, with every other matrix left as in the reference: 21 variants. And `X32:8` on all kinds.

**The split.** `allocate.py` of the output-budget record, unchanged, on the rows of Part A. For each budget, 4.5 and 3.5 bits, it gives a best arm and a reversed arm: the best arm's level counts put on the kinds in the order that it predicts is worst. The output is committed as `allocate.out` before any arm of Part B is scored.

**Part B.**

| Text | Variants |
|---|---|
| test | the best arm at 4.5 bits, the reversed arm at 4.5 bits, `X32:16`, the best arm at 3.5 bits, `X32:8` |
| validation | the best arm at 4.5 bits, `X32:16`, the best arm at 3.5 bits (`X32:8` is in Part A) |

30 variants in all. The program is `run.py` of the record clip-by-layer-qwen25-2026-11-18.

**The gain** of a split is 1 less its mean KL over that of equal bits, on the same text.

## What is known before the run

- On the test text `X32:16` scores 0.0620 (the record model-output-qwen25-2026-11-17). Nothing by kind has been scored on this model, on either text.
- On gemma the best arm at 4.5 bits gave the attention kinds 32 to 64 levels and the feed-forward kinds 12 to 16. Its reversed arm also beat equal bits, by 19.0 percent, so it was a weak control.
- In Qwen2.5 the kinds differ much in size. `attn_k` and `attn_v` are 256 by 1,536. The three feed-forward kinds are 1,536 by 8,960 and hold most of the weights.

## Predictions

An error of a gap is the root of the sum of the squares of the two standard errors.

| | Prediction | It fails if |
|---|---|---|
| H1 | On the test text the best arm at 4.5 bits has a lower mean KL than `X32:16`, by at least 2 errors | it does not |
| H2 | The gain at 4.5 bits on the test text is at least half the gain on the validation text | it is less |
| H3 | On the test text the best arm at 3.5 bits has a lower mean KL than `X32:8`, by at least 2 errors | it does not |
| H4 | On the test text the reversed arm at 4.5 bits has a higher mean KL than the best arm, by at least 2 errors | it does not |
| H5 | In the best arm at 4.5 bits every attention kind has at least as many levels as every feed-forward kind | one has fewer |

H1 and H2 are the answer. My confidence in them is fair to high: the two texts are close, and what makes a kind sensitive should not depend much on the passage.

## What it would mean

- If H1 and H2 hold, the split is not an artefact of fitting on the scored text, on this model and between these two texts. It says nothing about text of another kind.
- If H1 holds and H2 fails, part of the gain on gemma may be fit.
- If H1 fails, the split does not carry even between two halves of one source.
- For scale: on this model one ball per group takes the KL at 4.5 bits from 0.0620 to 0.0227, a gain of 63 percent, with no fit at all. Whether the two gains add is run 2 of the issue. It could be asked on this model too, with a ball arm whose bits differ by kind. That is not in this plan.

## What this cannot show

- Anything about a text that is not Wikipedia prose.
- Whether a finer split, by layer, would gain more.
- Anything about the split under another rule than `X32`.

## Order and time

One scoring process at a time. About six minutes a variant: three hours. Rows go to `results.jsonl` with a label for the text. The run is queued behind the clip-by-layer run.
