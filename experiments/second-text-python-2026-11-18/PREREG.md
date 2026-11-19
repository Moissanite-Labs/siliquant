# Plan: the headline comparisons on a second text

Written 2026-11-18, before any variant of this plan was scored and before the 16-bit models were run on the new text.

## Question

Every KL of the model-output records, the importance-matrix records and the packed-file record is on one text: the wikitext-2 test text, which is Wikipedia prose. The importance matrix those records use was made from the wikitext-2 training text, the same source. Every record says so under its limits.

Do the three headline comparisons hold on a text of another kind?

1. With no matrix on either side, the ball arm at 4.5 bits is under llama.cpp's `Q4_K_M` at 4.8 bits.
2. With the wikitext matrix on both sides, the same.
3. With a matrix made from a text with no meaning, the ball arm is under `Q4_K_M` made with the wikitext matrix.

And one question the record imatrix-gain-parts-2026-11-18 left: on a text that neither matrix has read, is the matrix from noise as good as the matrix from Wikipedia?

## What is run

**The text.** Python source: the file `conversion/base.py` of llama.cpp at commit `eada89c`, as `git show` gives it. 183,074 bytes, SHA-256 starts `05023c54`. The first 40 passages of 512 tokens, the second half of each scored: 10,200 positions, as on the other text. The file is not committed here.

**The reference.** Each 16-bit model's outputs on that text, saved by `llama-perplexity --kl-divergence-base` with the command of the model-output records.

**The models and the score** are those of the model-output records: `SmolLM2-1.7B-Instruct` and `Qwen2.5-1.5B-Instruct`, mean KL against the reference's saved outputs.

**Six variants a model, twelve in all.** Nothing is made new: every file and every matrix is the one an earlier record made and scored on the other text.

| Arm | Matrix | Variant | Scored on wikitext-2 in |
|---|---|---|---|
| llama.cpp `Q4_K_M` | none | `file:…-Q4_K_M.gguf` | the model-output records |
| llama.cpp `Q4_K_M` | wikitext-2 training text | `file:…-Q4_K_M-imat.gguf` | imatrix-baseline-2026-11-17 |
| llama.cpp `Q4_K_M` | no meaning | `file:…-Q4_K_M-rimat.gguf` | imatrix-gain-parts-2026-11-18 |
| the ball arm | none | `ZBG128:b560s16` | the model-output records |
| the ball arm | wikitext-2 training text | `hZBG128:b560s16` | imatrix-baseline-2026-11-17 |
| the ball arm | no meaning | `hZBG128:b560s16~rand` | imatrix-gain-parts-2026-11-18 |

**The programs** are those of the earlier records, unchanged: the model-output program for the files and for the first two ball arms, and the program of imatrix-gain-parts-2026-11-18 for the third. Only the reference outputs they are given differ. A row for a llama.cpp file carries the file's SHA-256, so that it can be seen to be the file scored before. Every row of this folder's `results.jsonl` is on the Python text.

**Before any score**, `reference.out` is committed: what `llama-perplexity` printed for each 16-bit model on the text, its perplexity among it.

## What is known beforehand

On the wikitext-2 test text, mean KL:

| | SmolLM2 | Qwen2.5 |
|---|---|---|
| `Q4_K_M`, no matrix | 0.0838 | 0.0333 |
| `Q4_K_M`, wikitext matrix | 0.0378 | 0.0282 |
| `Q4_K_M`, matrix of no meaning | 0.0399 | 0.0294 |
| the ball arm, no matrix | 0.0560 | 0.0227 |
| the ball arm, wikitext matrix | 0.0282 | 0.0209 |
| the ball arm, matrix of no meaning | 0.0314 | 0.0228 |

The record mmlu-subset-2026-11-18 found that the wikitext matrix's gain carries to questions it never read, counted in answers kept. Nothing in this repository has scored a KL on anything but Wikipedia prose.

## Predictions

An error is the tool's standard error, and an error of a gap is the root of the sum of the two squares.

| | Prediction | It fails if |
|---|---|---|
| C1 | On both models, with no matrix on either side, the ball arm is under `Q4_K_M` by 2 errors | it is not, on either |
| C2 | On both models, with the wikitext matrix on both sides, the ball arm is under `Q4_K_M` by 2 errors | it is not, on either |
| C3 | On both models, the ball arm with the matrix of no meaning is under `Q4_K_M` with the wikitext matrix by 2 errors | it is not, on either |
| C4 | On SmolLM2 the matrix of no meaning buys the ball arm at least half of what the wikitext matrix buys, on this text too | its share is under half |
| C5 | On SmolLM2, on a text neither matrix has read, the wikitext matrix loses its lead over the matrix of no meaning: the ball arm with the wikitext matrix is not under the ball arm with the matrix of no meaning by 2 errors. On the wikitext text it was under by 3.5 | it is under by 2 or more |
| C6 | On SmolLM2 the wikitext matrix still helps `Q4_K_M` on Python: under no matrix by 2 errors | it is not |

My confidence: fair in C1, C2, C3, C4 and C6. The gaps behind them were 6 to 15 errors on the other text, and what the matrix finds on SmolLM2 is a property of the model. Low in C5: I do not know how much of the wikitext matrix's small lead is the text and how much is that it is the better matrix.

**Reported without a prediction.** How much larger or smaller every KL is on Python than on prose. And on Qwen2.5, where the matrix buys little, whether either matrix helps at all on this text.

## What it would mean

- **C1 to C3 hold**: the headline lines of this branch are not a property of Wikipedia prose.
- **Any of them fails**: that line holds on prose and is not established beyond it, and the pull request must say so.
- **C5 holds**: where the text to come is not known, a matrix from noise is as good as one from a text of another kind.
- **C5 fails**: a matrix from real text is better even on a text of another kind, and the noise matrix is the fallback for when there is no text.

## What this cannot show

- One more text, of one kind, from one project. Dialogue, another language, or other code could differ.
- Both models may have read this project's source in training. That bears on how well the reference predicts the text, not on how far a rounded model is from the reference.
- Still KL, not a task.
- The file with blocks of 32 is not scored here: these are the counted arms with groups of 128.

## Order and time

One process at a time. For each model: the reference pass, about five minutes, then its six variants. About an hour and a quarter in all.
