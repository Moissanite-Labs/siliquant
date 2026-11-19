# The arm with kept columns, as a file

## Question

The record third-model-llama32-2026-11-18 found one arm that is ahead of llama.cpp's `Q4_K_M` on all three models with no data and no forward pass: integer levels in one ball per group of weights, with the ten loud columns of every `ffn_down` left whole. That arm was counted, and the packed format had no place for a kept column.

Is the file what the count said, and does the model the file holds stay ahead?

## Status

`complete`, 2026-11-19. 5 variants on three models. All 5 predictions hold.

## Preregistration

[PREREG.md](PREREG.md), committed and pushed at `d93fccd` with `run.py`, `summary.py`, the change to `shape/ballpack.py` and its tests, before any variant was scored. No program was changed after.

## Result in short

1. **A file of 4.5 bits a weight, made from the weights alone, holds a model that is closer to the 16-bit model than `Q4_K_M`'s at 4.8 bits, on all three models.** Mean KL 0.0247 against 0.0333 on Qwen2.5, 0.0581 against 0.0838 on SmolLM2 and 0.0340 against 0.0432 on Llama-3.2: 10.2, 13.5 and 11.9 errors. No calibration data is read and the model is never run.
2. **On Llama-3.2 the kept columns are what the file needs.** With nothing kept the file scores 0.0519, 9.0 errors behind `Q4_K_M`. With 160 columns kept it scores 0.0340: 34.5 percent lower.
3. **On the other two models they cost nothing and gain nothing**: 0.0247 against 0.0245 on Qwen2.5 and 0.0581 against 0.0589 on SmolLM2, 0.3 and 0.5 errors.
4. **The cost is five thousandths of a bit.** The files are 4.505 bits a weight, where the file with nothing kept is 4.500. Each was read back from disk bit for bit.
5. **With a matrix made from noise as well, the Llama-3.2 file is under `Q4_K_M` made with real text**: 0.0298 against 0.0359, 9.1 errors. That file needs the model to be run on noise and reads no text.

So the arm that the third model pointed at is now a file on all three. It is not ahead of `Q4_K_M` made with an importance matrix on SmolLM2: there a matrix is worth more than the columns.

## What was run

**The models, the text and the score** are those of the earlier records: `Qwen2.5-1.5B-Instruct`, `SmolLM2-1.7B-Instruct` and `Llama-3.2-1B-Instruct`, 16-bit files, the wikitext-2 test text, 10,200 positions, mean KL against the reference's saved outputs, with `llama-perplexity --kl-divergence` of llama.cpp at `eada89c`.

**The format.** `shape/ballpack.py`, as in the record packed-ball-format-2026-11-18: one ball per 32 weights, a number of 136 bits and one byte of scale, 18 bytes a block. New here: a matrix may carry a list of kept columns, their numbers as 32-bit integers and their entries as 16-bit floats. A kept column is zero in every block and is read back as stored. A file with no kept columns is byte for byte what it was, and a test holds the format to that.

**The rule for the columns.** In every `ffn_down`, the ten columns in which a clip at 5 standard deviations of a row removes the most squared weight. It is the function of the record loud-columns-2026-11-18, imported unchanged, and it reads only the weights.

**The program** is the packed-format record's with the one function that packs a matrix replaced. For each variant the file is written, read back from disk, every matrix read back is compared bit for bit with what the encoder gave, and the model the file holds is scored.

**The variants.** `PZ32:b136s8+K10` on each model. `PZ32:b136s8` on Llama-3.2, which the earlier record did not have. And `hPZ32:b136s8+K10` on Llama-3.2 with the matrix of no meaning of the third-model record, in `noise.jsonl`.

## Results

Every number is in `results.out`, printed by `summary.py` from `results.jsonl`, `noise.jsonl` and the rows of four earlier records.

| Model | File | Bytes | Bits a weight | Mean KL | First choice kept | Under `Q4_K_M` with no matrix, errors | Under `Q4_K_M` with the wikitext matrix, errors |
|---|---|---|---|---|---|---|---|
| Qwen2.5 | nothing kept, earlier record | 736,991,766 | 4.50004 | 0.0245 | 92.4% | 10.5 | 4.6 |
| Qwen2.5 | loud columns kept | 737,853,158 | 4.50530 | 0.0247 | 92.3% | 10.2 | 4.3 |
| SmolLM2 | nothing kept, earlier record | 905,975,382 | 4.50003 | 0.0589 | 87.7% | 13.2 | -16.2 |
| SmolLM2 | loud columns kept | 906,959,478 | 4.50492 | 0.0581 | 87.6% | 13.5 | -15.5 |
| Llama-3.2 | nothing kept | 547,360,470 | 4.50003 | 0.0519 | 88.5% | -9.0 | -17.2 |
| Llama-3.2 | loud columns kept | 548,016,534 | 4.50542 | 0.0340 | 90.3% | 11.9 | 2.7 |
| Llama-3.2 | loud columns kept, and the matrix of no meaning | 548,671,894 | 4.51081 | 0.0298 | 90.9% | 18.3 | 9.1 |

For reference, `Q4_K_M` at 4.81, 4.82 and 4.80 bits: 0.0333, 0.0838 and 0.0432 with no matrix, and 0.0282, 0.0378 and 0.0359 with the wikitext matrix.

The first six files read no data and run no forward pass. The last reads no data and runs the model on 51,200 tokens of noise.

**The file against the count.** The counted arm with groups of 128 and kept columns scored 0.0222, 0.0528 and 0.0302. The file, with blocks of 32 and a byte of scale, scores 0.0247, 0.0581 and 0.0340. That is what the smaller block costs, as it did before the columns were kept.

**Against `Q4_K_M` with a matrix.** Not predicted. The file with kept columns and no matrix is under `Q4_K_M` with the wikitext matrix on Qwen2.5 by 4.3 errors and on Llama-3.2 by 2.7. On SmolLM2 it is 15.5 errors behind. On SmolLM2 what a matrix finds is in the matrices that read the stream, and a kept column of `ffn_down` does not reach it.

### The predictions

| | Prediction | Result | Holds |
|---|---|---|---|
| F1 | On all three models the file with kept columns is under `Q4_K_M` with no matrix by 2 errors | 10.2, 13.5 and 11.9 | yes |
| F2 | On Llama-3.2 the file with nothing kept is not under `Q4_K_M` with no matrix by 2 errors | 9.0 errors behind | yes |
| F3 | On Llama-3.2 the kept columns take the file's KL down by at least 20 percent | 34.5 percent | yes |
| F4 | On Qwen2.5 and SmolLM2 the file with kept columns is not above the file with nothing kept by 2 errors | 0.3 above, 0.5 under | yes |
| F5 | On Llama-3.2 the file with the matrix of no meaning and kept columns is under `Q4_K_M` with the wikitext matrix by 2 errors | 9.1 | yes |

## Deviations from the plan

- **The machine restarted twice during the run**, while the second and the fifth variant were being scored. `results.start` has the three starts. The driver skips what is done, each interrupted variant was built and scored again from the start, and no row was lost.
- Nothing else.

## Limits

- **Three models between 1 and 2 billion weights**, and the kept columns matter on one of them. That they cost nothing on the other two is two cases.
- **One text.** The file was not scored on the Python text. The counted arm with kept columns was, on Llama-3.2 only.
- **KL, not a task.**
- **Ten columns, one limit.** Neither number was tuned, here or before. On Llama-3.2 two columns do what 160 do, so the rule keeps far more than is needed, at a cost that does not show.
- **Nothing about speed or memory when the model runs.** The file is decoded to floats first. No engine reads this file: it has a reader in Python and a decoder in C.
- **The standard errors are the tool's** and treat positions as independent. The findings rest on 9 errors and more, and the two ties are ties.

## Read against the closure rule

Nothing is closed by a certificate. These are 5 measurements on three models, and 16 tests of the format.

**What changes in the repository's account.**

1. The packed-format record ended with "only the integer arm with blocks of 32 has a file". That file loses to `Q4_K_M` on Llama-3.2. The file with kept columns does not, on any of the three.
2. The third-model record said the packed format needs a place for kept columns before its arm has a file. It has one, and the arm has a file.
3. The line that can now be written without a caveat about data: a file of 4.505 bits a weight, made from the weights alone, is ahead of `Q4_K_M` at 4.8 bits with no matrix on three models of three families, by 10 to 14 errors of KL on Wikipedia prose.

**What follows.**

1. For the sprint of issue #15 this is the arm and the file to start from. What it does not have is a reader in an engine.
2. A task benchmark on this file. The record mmlu-subset-2026-11-18 has the program, and none of its rows is for a file.
3. A fourth model, chosen to differ from these three.

## How to rerun

Needs NumPy, torch, a C compiler, llama.cpp's `gguf` package and `llama-perplexity`, the model file and its reference outputs on the test text.

```bash
python3 -m unittest shape/test_ballpack.py
```

```bash
python3 experiments/packed-kept-columns-2026-11-19/run.py --model Llama-3.2-1B-Instruct-f16.gguf --kld wikitext.kld --work DIR --out results.jsonl --bin llama-perplexity 'PZ32:b136s8' 'PZ32:b136s8+K10'
```

```bash
python3 experiments/packed-kept-columns-2026-11-19/run.py --model Llama-3.2-1B-Instruct-f16.gguf --kld wikitext.kld --work DIR --out noise.jsonl --bin llama-perplexity --imatrix llama32.rimatrix.gguf 'hPZ32:b136s8+K10'
```

```bash
python3 experiments/packed-kept-columns-2026-11-19/summary.py > results.out
```

A variant takes four to six minutes on 4 CPU cores: under a minute and a half to pack, the rest to score.

## Disclosure

Written by an AI agent session under the owner's direction, on 2026-11-19. The plan, the change to the format, the run and this record are the session's. The scoring tool and `Q4_K_M` are llama.cpp's.
