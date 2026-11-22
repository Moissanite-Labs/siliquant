# Plan: the arm with kept columns, as a file

Written 2026-11-19, before any variant of this plan was scored.

## Question

The record third-model-llama32-2026-11-18 found one arm that is ahead of llama.cpp's `Q4_K_M` on all three models with no data and no forward pass: integer levels in one ball per group of weights, with the ten loud columns of every `ffn_down` left whole. That arm was counted. Its bits were those of a codebook, with 16 bits for each kept entry, and no file held it. The packed format of the record packed-ball-format-2026-11-18 had no place for a kept column.

Now it has. Is the file what the count said, and does the model the file holds stay ahead?

## What is new in the format

`shape/ballpack.py` stores, for a matrix, an optional list of kept columns: how many, their numbers as 32-bit integers, and their entries as 16-bit floats row by row. A kept column is zero in every block and is read back as it was stored. A file with no kept columns is byte for byte what it was before. Four tests were added to `shape/test_ballpack.py`, one of them for that last statement, and all 16 pass.

## What is run

**The models, the text and the score** are those of the earlier records: `Qwen2.5-1.5B-Instruct`, `SmolLM2-1.7B-Instruct` and `Llama-3.2-1B-Instruct`, 16-bit files, the wikitext-2 test text, 10,200 positions, mean KL against the reference's saved outputs.

**The file.** `PZ32:b136s8` of the packed-format record: one ball per 32 weights, a number of 136 bits and one byte of scale, 18 bytes a block. With `+K10`, the ten loud columns of every `ffn_down` are kept. A loud column is one of the ten in which a clip at 5 standard deviations of a row removes the most squared weight: the rule of the record loud-columns-2026-11-18, imported unchanged. It reads the weights and nothing else.

**The program** is `run.py` of this folder: the packed-format record's program with the one function that packs a matrix replaced. The file is written, read back from disk, every matrix read back must be bit for bit the one the encoder gave or the program stops, and the model the file holds is scored. Tried without scoring on the first nine block matrices of Llama-3.2: the file is read back exactly with and without kept columns.

**Five variants.**

| Variant | Model | Matrix | Rows in |
|---|---|---|---|
| `PZ32:b136s8+K10` | Qwen2.5 | none | `results.jsonl` |
| `PZ32:b136s8+K10` | SmolLM2 | none | `results.jsonl` |
| `PZ32:b136s8` | Llama-3.2 | none | `results.jsonl` |
| `PZ32:b136s8+K10` | Llama-3.2 | none | `results.jsonl` |
| `hPZ32:b136s8+K10` | Llama-3.2 | the matrix of no meaning of the third-model record | `noise.jsonl` |

Already scored: the file with no matrix and nothing kept on Qwen2.5 and SmolLM2, 0.0245 and 0.0589, in the packed-format record. `Q4_K_M` with no matrix: 0.0333, 0.0838 and 0.0432. `Q4_K_M` with the wikitext matrix: 0.0282, 0.0378 and 0.0359.

## What is known beforehand

The counted arm with groups of 128, no matrix:

| | Qwen2.5 | SmolLM2 | Llama-3.2 |
|---|---|---|---|
| nothing kept | 0.0227 | 0.0560 | 0.0478 |
| loud columns kept | 0.0222 | 0.0528 | 0.0302 |

The file has blocks of 32 and an 8-bit scale where the counted arm has groups of 128 and a 16-bit scale. On the first two models the file scored 1.08 and 1.05 times the counted arm with no matrix.

## Predictions

An error is the tool's standard error, and an error of a gap is the root of the sum of the two squares.

| | Prediction | It fails if |
|---|---|---|
| F1 | On all three models the file with kept columns, made with no data and no forward pass, is under `Q4_K_M` with no matrix by 2 errors | it is not, on any |
| F2 | On Llama-3.2 the file with nothing kept is not under `Q4_K_M` with no matrix by 2 errors: without the columns the file loses there, as the counted arm does | it is under by 2 or more |
| F3 | On Llama-3.2 the kept columns take the file's KL down by at least 20 percent | by less |
| F4 | On Qwen2.5 and SmolLM2 the kept columns do the file no harm: it is not above the file with nothing kept by 2 errors | it is, on either |
| F5 | On Llama-3.2 the file made with the matrix of no meaning and kept columns is under `Q4_K_M` made with the wikitext matrix by 2 errors | it is not |

My confidence: fair in all five. They follow the counted arms, and the file has followed the counted arm within a tenth before. The one I am least sure of is F1 on Llama-3.2, where the count leaves a margin of about 17 errors and a file costs some of it.

**Reported without a prediction.** The bytes of each file and its bits a weight. Whether the file with kept columns and no matrix is under `Q4_K_M` with the wikitext matrix: by the counted arm it would be on Qwen2.5 and Llama-3.2 and not on SmolLM2.

## What it would mean

- **F1 holds**: there is a file of 4.5 bits a weight, made from the weights alone, whose model is closer to the 16-bit model than `Q4_K_M`'s at 4.8 bits, on three models of three families.
- **F1 fails on a model**: the counted arm's lead there does not survive the smaller blocks, and the format's block would have to grow.

## What this cannot show

- Three models between 1 and 2 billion weights. KL on one text here: the Python text is not scored for the file.
- Nothing about speed or memory when the model runs. The file is decoded to floats first, as before.
- No engine reads this file. It is a format with a reader in Python and a decoder in C.

## Order and time

One process at a time: Llama-3.2's three variants, then SmolLM2, then Qwen2.5. About six minutes a variant: half an hour.
