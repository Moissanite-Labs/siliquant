# Plan: a packed file for the integer ball arm, and the model read back from it

Written 2026-11-18, before any model packed this way was scored. The coder is `shape/ballpack.py` with `shape/ballpack.c`, tested in `shape/test_ballpack.py` and committed with this plan.

## Question

Every record that scored a ball arm says the same thing under its limits: the bits are counted from the size of the codebook, no file holds them, and the weights were scored from 16-bit float copies.

1. Can the arm be stored in the bits counted for it, and read back exactly?
2. Does the model read back from the file score as the counted arm did?
3. How long does the file take to decode?

## The format

Integer levels in one ball per block of 32 consecutive weights of a row. For every block, 18 bytes: one byte for the scale and 17 for the number of the block's point among the points of the ball. That is 4.5 bits a weight.

- **The ball** is the largest that holds no more than 2^136 points of Z^32: squared radius 783. It holds 0.996 of 2^136 points.
- **The number** of a point is the one `shape/ballindex.py` gives it: the count of the points before it, reading the entries in order.
- **The scale** of a block is the matrix's largest scale times 2^(-code/16), with the code in one byte. The largest scale is stored once for a matrix as a 16-bit float.
- **With an importance matrix**, every column has a weight, stored as a 16-bit float. Weights are multiplied by it before rounding and divided by it after.
- A weight is read back as float16(entry × scale / column weight).

Why 32 weights and not the 128 of `ZBG128:b560s16`. A number of 136 bits is three machine words and its table of counts is 0.6 megabytes. A number of 560 bits is nine words with a table of 30 megabytes. On Qwen2.5 the E8 arm with blocks of 32 and an 8-bit scale lost 0.27 dB of KL against blocks of 128 (`EBG32:b136s8` against `EBG128:b560s16`).

The file is not a GGUF type and llama.cpp cannot load it. For scoring it is decoded to 16-bit floats. So nothing here saves memory when the model runs.

## What is run

For each variant on each model, by `run.py`:

1. Every block matrix is rounded and packed, and the file is written.
2. The file is read back from disk and every matrix is decoded from it. Each must be, bit for bit, the matrix the encoder said the file would hold. If one is not, the program stops.
3. The numbers of the whole file are decoded once more to 32-bit floats on one core, twice, and the processor time of the faster pass is kept.
4. A copy of the model is written with the decoded matrices and scored as in the model-output records: mean KL and first choice kept against the 16-bit model, wikitext-2 test text, 10,200 positions.

| Variant | What |
|---|---|
| `PZ32:b136s8` | the format, no data |
| `hPZ32:b136s8` | the format with column weights from llama.cpp's importance matrix at strength 0.25, the strength the importance-matrix record chose on validation text |

On `Qwen2.5-1.5B-Instruct` and `SmolLM2-1.7B-Instruct`: four rows. The matrices are those of the importance-matrix record.

The bits of a row are 8 times the bytes of its file over the number of block weights. The file is deleted after it is scored: its size and SHA-256 are in the row.

## What is known before the run

- **The tests.** The C coder gives the numbers of `ballindex.py` on 300 random points and walks a small codebook whole. Round trips are exact on five made-up matrices, with and without column weights.
- **A test on four matrices of Qwen2.5, weight error only, nothing scored.** 41.7 million weights. Stored bits 4.500 without column weights and 4.505 with. Weight error -24.40 dB without, where `ZBG128:b560s16` has -24.94 over the whole model. Read back exactly. 8.3 percent of the weights are stored as zero. Decoded at 22 to 25 million weights a second on one core while another job used all four.
- **The counted arms**, mean KL on Qwen2.5 and SmolLM2: `ZBG128:b560s16` 0.0227 and 0.0560. `hZBG128:b560s16` 0.0209 and 0.0282. llama.cpp `Q4_K_M` 0.0333 and 0.0838 at 4.81 and 4.82 bits, and with the matrix 0.0282 and 0.0378.
- No model packed in this format has been scored.

## Predictions

An error of a gap is the root of the sum of the squares of the two standard errors the tool prints.

| | Prediction | It fails if |
|---|---|---|
| F1 | Every file is read back exactly, and its stored bits are at most 4.501 a weight without column weights and 4.52 with | a matrix differs (the run stops), or a file is larger |
| F2 | On each model `PZ32:b136s8` has a lower mean KL than `Q4_K_M` at 4.8 bits, by at least 2 errors | it does not, on either model |
| F3 | On each model `hPZ32:b136s8` has a lower mean KL than `Q4_K_M` with the matrix, by at least 2 errors | it does not, on either model |
| F4 | On each model the mean KL of `PZ32:b136s8` is no more than 1.25 times that of `ZBG128:b560s16`, and that of `hPZ32:b136s8` no more than 1.25 times that of `hZBG128:b560s16` | one is more, on either model |
| F5 | Each file decodes at 15 million weights a second or more on one core | one is slower |

How I get F4. The block of 32 with an 8-bit scale has about half a decibel more weight error than the block of 128. On Qwen2.5 a decibel of weight error was a decibel of KL for these arms, so the KL should be about 1.13 times as large: near 0.026 and 0.024. On SmolLM2 near 0.063 and 0.032.

F2 and F3 are the point: the lead over llama.cpp's file, at stored bits.

## What it would mean

- If F1 to F3 hold, the limit "counted bits" comes off the headline for this arm: a file of 4.5 bits a weight exists and the model read from it is ahead of `Q4_K_M` at 4.8, with and without the matrix, on these two models and this text.
- If F4 fails, blocks of 32 cost more than their weight error says and the format should use longer blocks, at the price of longer numbers.
- F5 is a floor on speed, not a claim. At 20 million weights a second a core, a model of 1.3 billion block weights takes about a minute to decode on one core. That is a cost at load time. It is about three orders of magnitude from what a kernel that computes from packed weights would need: on this machine llama.cpp's tool goes through the 1.5 billion weights of a `Q4_K_M` file some 120 times a second on four cores.

## What this cannot show

- Anything about memory or speed when the model runs. The weights are decoded to floats first.
- That the coder is fast. It is plain C, one block at a time, with a linear search over symbols.
- Anything about models other than these two, or text other than wikitext-2.
- That the file format is good. It has no alignment, no index of matrices and no versioning beyond a magic word. It exists to count bytes.
