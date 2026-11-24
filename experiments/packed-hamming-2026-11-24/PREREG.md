# Plan: the packed file with A_16 in place of the integers

Committed before any matrix of a model is packed in this format and before anything of it is scored.

## Question

The repository's one file format stores 32 weights in 18 bytes: a one-byte scale code and a 136-bit number of a point of Z^32 in a ball (`shape/ballpack.py`). With ten loud columns of each `ffn_down` kept, it is ahead of `Q4_K_M` on three models with no data (packed-kept-columns-2026-11-19).

The same 18 bytes can number a point of A_16 × A_16 in a ball (`shape/hampack.py`). A_16 = C_16 + 2Z^16, with C_16 the extended Hamming code [16, 11, 4]. Two things are known about it.

- **As a counted arm it scores well** (hamming-lattices-real-weights-2026-11-24). One ball of A_16 per 128 weights scored under integer levels in the same ball on all three models at 4.5 bits.
- **In this block, on made-up matrices, it has less weight error** (`notes/analysis-2026-11-24`). 0.80 dB less than Z^32.

Does that carry to the file, at the same size?

## What is run

`run.py` is the program of packed-kept-columns-2026-11-19 with the format swapped for `HamFormat(16, 32, 136, 16)`. Nothing else changes:

- the same scale code (one byte, 16 steps to a factor of two, from the matrix's largest scale in 16 bits);
- the same choice of kept columns;
- the same read-back from disk, bit for bit, or the program stops;
- the same decode timing on one core;
- the same score: wikitext-2 test text, mean KL against the 16-bit model over 10,200 positions, `llama-perplexity` of llama.cpp at `eada89c`.

The format differs from ballpack in three things:
- the nearest point of A_16, by the dynamic program of the hamming-lattices record, copied into `shape/hampack.c`;
- the bisection's margin;
- the numbering, which walks the entries with the partial syndrome as state.

`shape/test_hampack.py` holds:
- the C coder to a Python walk of the same table;
- both, on small cases, to a full enumeration of the ball in order;
- the counts to `shape/hammingindex.py`'s;
- the matrix and the file to exact round trips.

| Model | Rows |
|---|---|
| Qwen2.5-1.5B-Instruct, SmolLM2-1.7B-Instruct, Llama-3.2-1B-Instruct | `PH16:b136s8` and `PH16:b136s8+K10` |

They are compared with the committed rows of the integer file.

| | `PZ32:b136s8` | `PZ32:b136s8+K10` | `Q4_K_M@16` (hamming-lattices record) |
|---|---|---|---|
| Qwen2.5 | 0.024517 | 0.024719 | 0.031690 |
| SmolLM2 | 0.058855 | 0.058117 | 0.072237 |
| Llama-3.2 | 0.051877 | 0.033987 | 0.041195 |

## Seen before this plan

- **All the 4.5-bit rows of the hamming-lattices record**, including A_16 on all three models. Counted arm, one ball per 128 weights, against integer levels in the same ball:
  - Qwen2.5: 0.018974 against 0.022750;
  - SmolLM2: 0.047957 against 0.055962;
  - Llama-3.2: 0.034455 against 0.047808.
- **Weight error only**, from `notes/analysis-2026-11-24`:
  - `group32.out`: three matrices of SmolLM2 and made-up matrices in this block;
  - `hampack_speed.out`: made-up matrices packed by both formats. A_16 at −25.240 dB against −24.439 for Z^32; decode at 10.2 against 20.8 million weights a second on one core, with the four cores busy.
- No model has been packed in this format. Nothing of it has been scored.

## Predictions

A gap is in errors: the difference of two mean KLs over the root of the sum of their squared standard errors. Under 2 is a tie.

- **F1.** The pooled weight error of every `PH16` file is under that of the `PZ32` file of the same model and kept columns by 0.70 to 0.90 dB.
- **F2.** `PH16:b136s8` is under `PZ32:b136s8` by 2 errors on all three models.
- **F3.** `PH16:b136s8+K10` is under `PZ32:b136s8+K10` by 2 errors on all three models.
- **F4.** The mean KL of `PH16:b136s8` over that of `PZ32:b136s8` is between 0.70 and 0.92 on each model. For ball arms a decibel of weight error has gone with about a decibel of KL, and 0.80 dB is a factor of 0.83. The counted arms at 128 gave 0.83, 0.86 and 0.72.
- **F5.** `PH16:b136s8+K10` is under `Q4_K_M@16` by 2 errors on all three models, and by more errors than `PZ32:b136s8+K10` is.
- **F6.** Every file decodes at 5 million weights a second or more on one core. That is about half the integer file, as on made-up matrices, because the table is 49 times larger. Low confidence on the threshold: the machine's other cores will be idle in this run, which they were not when the figure above was taken.
- **Check.** Every file is read back bit for bit. Its stored bits are 4.5000 to 4.5007 a weight: the blocks' 4.5 and the files' headers, scales and kept columns, as for the integer files.

## What it would mean

- **If F2 and F3 hold:** the file to recommend with no data is A_16 in the same 18 bytes. It would be the first file with a lattice in it.
- **If F1 holds and F2 fails:** the counted arm's gain at 128 does not survive the block of 32 with a one-byte scale.
- **F6** prices what the gain costs in decoding.

## What this cannot show

- No engine reads this file. Like the integer file, it is decoded to 16-bit floats before the model runs.
- One text, three models of 1 to 2 billion weights, no importance matrix.

## Order and time

Started when the run of hamming-lattices-real-weights-2026-11-24 has ended, not before, so the two do not share the four cores:
1. Qwen2.5, both rows;
2. SmolLM2, both rows;
3. Llama-3.2, both rows.

A watcher commits each row as it lands. Packing a model should take 10 to 20 minutes and scoring 4.

## Disclosure

Written by an AI agent session under the owner's direction, on 2026-11-24. The format, its tests, the plan and the program are the session's. Construction A and the extended Hamming codes are textbook (Conway and Sloane, *Sphere Packings, Lattices and Groups*, ch. 5).
