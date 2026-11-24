# The packed file with A_16 in place of the integers

## Question

The repository's one file format (`shape/ballpack.py`, packed-ball-format-2026-11-18) stores 32 weights in 18 bytes:
- a one-byte scale code;
- the 136-bit number of a point of Z^32 in a ball.

With ten loud columns of each `ffn_down` kept, it is ahead of `Q4_K_M` on three models with no data (packed-kept-columns-2026-11-19). The same 18 bytes can number a point of A_16 × A_16 in a ball (`shape/hampack.py`). Does the file then score better, at the same size?

## Status

`complete`, 2026-11-24. 6 files on three models, each read back bit for bit and scored. 6 of 6 predictions hold. The check fails as written, for a reason in the plan and not in the files (see Deviations).

## Preregistration

[PREREG.md](PREREG.md) and `run.py` were committed and pushed at `6209621`, before any model was packed in this format. The run started at `85bea50` (`results.start`), when the run of hamming-lattices-real-weights-2026-11-24 had ended.

`summary.py` was not in the plan. It was written and committed at `99d2712`, after the two rows of Qwen2.5 had landed, and applies the plan's rules as written.

## Result in short

Yes, on all three models. At the same 18 bytes for 32 weights, the file with A_16 × A_16 scores lower than the file with Z^32:

| Model | No columns kept: gap, KL ratio | Ten columns kept: gap, KL ratio |
|---|---|---|
| Qwen2.5-1.5B | +5.2 errors, 0.864 | +6.2 errors, 0.842 |
| SmolLM2-1.7B | +4.6 errors, 0.888 | +7.3 errors, 0.826 |
| Llama-3.2-1B | +12.7 errors, 0.765 | +12.4 errors, 0.786 |

- **Weight error.** It is 0.827 to 0.828 dB lower in every file, as predicted (F1).
- **Against `Q4_K_M@16`, with ten columns kept.** The A_16 file is lower by 15.0, 14.9 and 21.8 errors; the integer file is lower by 8.8, 8.1 and 9.7.
- **The price is time.**
  - It decodes at 9.4 to 10.1 million weights a second on one core: about 100 seconds per billion weights, 2.5 to 3.0 times slower than the integer file.
  - Packing takes 8 to 15 minutes a model on 4 cores, against about one minute.
- **Exactness and size.**
  - Every file reads back bit for bit, and every file has exactly the integer file's byte count.
  - The check on stored bits fails as written for the three files with kept columns. The cause is in the plan, not the files (see Deviations).
- **Not in the plan: the ball of 128 weights is still lower.** The A_16 ball of 128 weights with a 16-bit scale (hamming-lattices-real-weights-2026-11-24) is lower than this file by 3.1 to 6.7 errors at the same 4.5 bits, on the four pairs scored.
  - The blocks of 32 with a one-byte scale give that back.
  - A file with blocks of 128 is the next step (see below).

## What was run

`run.py` is the program of packed-kept-columns-2026-11-19 with `shape/hampack.py`'s `HamFormat(16, 32, 136, 16)` in place of `ballpack.Format(32, 136, 16)`. The steps are unchanged:
- pack every block matrix;
- write the file;
- read it back from disk, and stop unless every matrix is bit for bit the one the encoder gave;
- time the decode on one core;
- score the model the file holds.

The scale code, the choice of kept columns and the score are the integer file's. Models, text and score are those of the earlier records.

## The files

| Model | File | Kept columns | Weight error (dB) | Stored bits a weight | Bytes | Read back exact | Decode, million weights a second, one core | Pack (s) | Mean KL |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B | A_16 | none | -25.236 | 4.50004 | 736991766 | True | 9.64 | 750 | 0.021171 ± 0.000417 |
| Qwen2.5-1.5B | Z^32 | none | -24.409 | 4.50004 | 736991766 | True | 24.15 | 61 | 0.024517 ± 0.000495 |
| Qwen2.5-1.5B | A_16 | ten | -25.238 | 4.50530 | 737853158 | True | 9.87 | 676 | 0.020816 ± 0.000385 |
| Qwen2.5-1.5B | Z^32 | ten | -24.411 | 4.50530 | 737853158 | True | 24.73 | 60 | 0.024719 ± 0.000504 |
| SmolLM2-1.7B | A_16 | none | -25.233 | 4.50003 | 905975382 | True | 9.74 | 879 | 0.052260 ± 0.000958 |
| SmolLM2-1.7B | Z^32 | none | -24.405 | 4.50003 | 905975382 | True | 25.02 | 74 | 0.058855 ± 0.001066 |
| SmolLM2-1.7B | A_16 | ten | -25.236 | 4.50492 | 906959478 | True | 9.77 | 837 | 0.047987 ± 0.000878 |
| SmolLM2-1.7B | Z^32 | ten | -24.408 | 4.50492 | 906959478 | True | 29.09 | 64 | 0.058117 ± 0.001082 |
| Llama-3.2-1B | A_16 | none | -25.234 | 4.50003 | 547360470 | True | 9.37 | 538 | 0.039680 ± 0.000592 |
| Llama-3.2-1B | Z^32 | none | -24.406 | 4.50003 | 547360470 | True | 23.68 | 51 | 0.051877 ± 0.000755 |
| Llama-3.2-1B | A_16 | ten | -25.236 | 4.50542 | 548016534 | True | 10.07 | 491 | 0.026708 ± 0.000338 |
| Llama-3.2-1B | Z^32 | ten | -24.409 | 4.50542 | 548016534 | True | 27.69 | 44 | 0.033987 ± 0.000478 |

## Gaps, in errors (positive: the first is lower), and ratios

- Qwen2.5-1.5B, no columns kept: A_16 file against Z^32 file +5.2; KL ratio 0.864; weight error 0.827 dB lower
- Qwen2.5-1.5B, ten columns kept: A_16 file against Z^32 file +6.2; KL ratio 0.842; weight error 0.827 dB lower
- Qwen2.5-1.5B, ten columns kept: A_16 file against Q4_K_M@16 +15.0; Z^32 file against Q4_K_M@16 +8.8
- SmolLM2-1.7B, no columns kept: A_16 file against Z^32 file +4.6; KL ratio 0.888; weight error 0.828 dB lower
- SmolLM2-1.7B, ten columns kept: A_16 file against Z^32 file +7.3; KL ratio 0.826; weight error 0.828 dB lower
- SmolLM2-1.7B, ten columns kept: A_16 file against Q4_K_M@16 +14.9; Z^32 file against Q4_K_M@16 +8.1
- Llama-3.2-1B, no columns kept: A_16 file against Z^32 file +12.7; KL ratio 0.765; weight error 0.828 dB lower
- Llama-3.2-1B, ten columns kept: A_16 file against Z^32 file +12.4; KL ratio 0.786; weight error 0.827 dB lower
- Llama-3.2-1B, ten columns kept: A_16 file against Q4_K_M@16 +21.8; Z^32 file against Q4_K_M@16 +9.7

## The predictions

| | Prediction | Result | Holds |
|---|---|---|---|
| F1 | weight error of each A_16 file under the Z^32 file's by 0.70 to 0.90 dB | 0.827, 0.827, 0.828, 0.828, 0.828, 0.827 | yes |
| F2 | A_16 file under Z^32 file by 2 errors, no columns kept, three models | +5.2, +4.6, +12.7 | yes |
| F3 | A_16 file under Z^32 file by 2 errors, ten columns kept, three models | +6.2, +7.3, +12.4 | yes |
| F4 | KL ratio of A_16 file to Z^32 file, no columns kept, 0.70 to 0.92 | 0.864, 0.888, 0.765 | yes |
| F5 | A_16 file with columns kept under Q4_K_M@16 by 2, and by more than the Z^32 file | +15.0 against +8.8, +14.9 against +8.1, +21.8 against +9.7 | yes |
| F6 | decode at 5 million weights a second or more, one core | 9.64, 9.87, 9.74, 9.77, 9.37, 10.07 | yes |
| Check | read back bit for bit, 4.5000 to 4.5007 bits a weight stored | 4.50004, 4.50530, 4.50003, 4.50492, 4.50003, 4.50542 | **no** |

6 of 6 decided predictions hold; 0 not yet decided.

The integer files' rows are those of packed-ball-format-2026-11-18 and packed-kept-columns-2026-11-19; `Q4_K_M@16` is that of hamming-lattices-real-weights-2026-11-24 (`baselines.jsonl`). Full output: [results.out](results.out).

## Not in the plan: the file against the ball of 128 weights

`against_ball.py` was written after five of the six rows had landed. It sets each file against the A_16 ball of 128 weights with a 16-bit scale, from hamming-lattices-real-weights-2026-11-24, at the same 4.5 bits a weight. That record did not score the ball with ten columns kept on Qwen2.5 or SmolLM2.

| Model | Kept columns | A_16 file, 32 weights, one-byte scale | A_16 ball, 128 weights, 16-bit scale | File over ball | Ball lower by (errors) |
|---|---|---|---|---|---|
| Qwen2.5-1.5B | none | 0.021171 ± 0.000417 | 0.018974 ± 0.000374 | 1.116 | +3.9 |
| Qwen2.5-1.5B | ten | 0.020816 | not run | | |
| SmolLM2-1.7B | none | 0.052260 ± 0.000958 | 0.047957 ± 0.001019 | 1.090 | +3.1 |
| SmolLM2-1.7B | ten | 0.047987 | not run | | |
| Llama-3.2-1B | none | 0.039680 ± 0.000592 | 0.034455 ± 0.000499 | 1.152 | +6.7 |
| Llama-3.2-1B | ten | 0.026708 ± 0.000338 | 0.024053 ± 0.000307 | 1.110 | +5.8 |

The ball is lower in all four pairs, by 3.1 to 6.7 errors (a KL 0.87 to 0.92 of the file's). A block of 32 with a one-byte scale code costs that much against a ball of 128 with a 16-bit scale. Output: [against_ball.out](against_ball.out).

## Deviations from the plan

- **The check's range of stored bits was wrong for files with kept columns.**
  - The plan gave 4.5000 to 4.5007 bits a weight, "as for the integer files". That fits the files with no columns kept.
  - With ten columns kept, the 16-bit entries of the columns add 0.005 bits: the files store 4.5049 to 4.5054.
  - The integer files with the same columns store the same bytes, to the byte.
  - The rule is left as written and reported as failing for those three files. Every file was read back bit for bit.
- **`summary.py` was written after the first two rows**, as said above.
- **`against_ball.py` was not in the plan.** It was written after five of the six rows had landed, and is reported apart from the predictions.
- No restart of the machine during this run.

## Limits

- No engine reads this file: like the integer file, it is decoded to 16-bit floats before the model runs.
- The decode is 2.5 to 3.0 times slower than the integer file's, on one core and in a C coder written for checking, not speed.
- One text, three models of 1 to 2 billion weights, no importance matrix.

## Read against the plan

The plan's "What it would mean" said: if F2 and F3 hold, the file to recommend with no data is A_16 in the same 18 bytes, and it would be the first file with a lattice in it. Both hold.

- **The recommended file with no data is now `shape/hampack.py`, `HamFormat(16, 32, 136, 16)`, with ten columns of each `ffn_down` kept.**
  - It replaces `shape/ballpack.py`'s integer file, at the same size to the byte.
  - It is lower than the integer file by 6.2 to 12.4 errors on three models, and lower than `Q4_K_M@16` by 14.9 to 21.8 errors.
- **F6 prices it.**
  - Decoding is 2.5 to 3.0 times slower, in a C coder written for checking and not for speed, on one core.
  - No engine reads either file, so this is a cost once per load, not per token.
- **Still open.**
  - **A file with blocks of 128 weights.** It would store the 560-bit number of `shape/hammingindex.py` and a 16-bit scale: 72 bytes for 128 weights, the same 4.5 bits. The arm it would hold is 3.1 to 6.7 errors lower than this file. Its coder needs 560-bit numbers and the two-level numbering, in C.
  - **An engine that multiplies by the file's points directly**, rather than decoding them first. Such an engine would pay the activation rounding that `Q4_K_M` pays on the CPU backend; `Q4_K_M@16` is the fair comparison here only because both are decoded to 16 bits.

## How to rerun

```bash
python3 -m unittest shape/test_hampack.py
python3 experiments/packed-hamming-2026-11-24/run.py --model MODEL-f16.gguf --kld base.kld --work DIR --out results.jsonl --bin llama-perplexity PH16:b136s8 PH16:b136s8+K10
python3 experiments/packed-hamming-2026-11-24/summary.py > results.out
python3 experiments/packed-hamming-2026-11-24/against_ball.py > against_ball.out
```

Packing a model takes 8 to 15 minutes on 4 cores, and scoring takes 4.

## Disclosure

Written by an AI agent session under the owner's direction, on 2026-11-24. The format, its tests, the plan, the program, the run and this record are the session's. Construction A and the extended Hamming codes are textbook. The scoring tool is llama.cpp's.
