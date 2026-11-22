# A packed file for the integer ball arm, and the model read back from it

## Question

Every record that scored a ball arm counted its bits from the size of the codebook and scored the weights from 16-bit float copies. No file held those bits.

1. Can the arm be stored in the bits counted for it, and read back exactly?
2. Does the model read back from the file score as the counted arm did?
3. How long does the file take to decode?

## Status

`complete`, 2026-11-18. Four rows: two variants on two models. All five predictions hold.

## Preregistration

[PREREG.md](PREREG.md), committed and pushed at `4d3995c` with the coder and its tests, before any model packed this way was scored. The plan states a test on four matrices of Qwen2.5 that was run before it, weight error only. `run.py` and `summary.py` are as they were at that commit.

## Result in short

1. **The bits are now the bytes of a file.** Integer levels in one ball per block of 32 weights take 18 bytes a block. The file of `Qwen2.5-1.5B` is 736,991,766 bytes for 1,310,195,712 block weights: 4.50004 bits a weight. Every one of the 364 matrices packed without column weights, and of the 364 packed with them, was read back from disk bit for bit as the encoder said it would be.
2. **The model read back from the file is ahead of llama.cpp's `Q4_K_M` at 4.8 bits, on both models.** Mean KL 0.0245 against 0.0333 on Qwen2.5 and 0.0589 against 0.0838 on SmolLM2: 10.5 and 13.2 errors of the gap.
3. **With the importance matrix on both sides it is ahead too.** 0.0227 against 0.0282 and 0.0309 against 0.0378, at 4.51 bits against 4.81 and 4.82: 7.1 and 6.8 errors.
4. **Blocks of 32 with a one-byte scale cost 5 to 10 percent of KL against the counted arm with blocks of 128.** Their weight error is half a decibel worse. That is the price of a number of 136 bits in place of 560.
5. **A file decodes at 20 to 25 million weights a second on one core.** About a minute for a model. That is a cost when a model is loaded. It is nowhere near what a kernel that computes from packed weights would need.

So for this arm the limit "counted bits" is gone. The limit "no memory is saved when the model runs" stands exactly as it was: the file is decoded to floats.

## What was run

**The format**, `shape/ballpack.py` with `shape/ballpack.c`. Each row of a matrix is cut into blocks of 32 consecutive weights. A block is rounded to a point of Z^32 inside a ball of squared radius 783, the largest ball that holds no more than 2^136 points. It holds 0.996 of 2^136. For a block the file has one byte of scale code and 17 bytes for the number of the point among the points of the ball, in the order `shape/ballindex.py` numbers them. A block's scale is the matrix's largest scale, stored once as a 16-bit float, times 2^(-code/16). With an importance matrix every column also has a weight stored as a 16-bit float. A weight is read back as float16(entry × scale / column weight).

**For each variant on each model**, `run.py`:

1. rounds and packs every block matrix, and writes the file;
2. reads the file back from disk and decodes every matrix from it. The SHA-256 of each decoded matrix must be the one the encoder gave, or the program stops;
3. decodes the numbers of the whole file once more to 32-bit floats on one core, twice, and keeps the processor time of the faster pass;
4. writes a copy of the model with the decoded matrices as 16-bit floats and scores it with `llama-perplexity --kl-divergence` against the reference's saved outputs. The file is then deleted. Its size and SHA-256 are in the row.

| Variant | What |
|---|---|
| `PZ32:b136s8` | the format, no data |
| `hPZ32:b136s8` | the format with column weights: the mean squared size of each input from llama.cpp's importance matrix, over its mean, to the power 0.25. That is the strength the record imatrix-baseline-2026-11-17 chose on validation text |

**Models, text and score** are those of the model-output records: `Qwen2.5-1.5B-Instruct` and `SmolLM2-1.7B-Instruct` from their 16-bit files, the wikitext-2 test text, 10,200 positions, mean KL and first choice kept against the 16-bit model. The importance matrices are those of the importance-matrix record, from the wikitext-2 training text.

**Bits** of a row here are 8 times the bytes of its file over the number of block weights.

## Results

Every number is in `results.out`, printed by `summary.py` from `results.jsonl` and the rows of three other records.

### The files

| Model | Variant | Bytes | Stored bits a weight | Read back exactly | Weights stored as zero | Seconds to pack, 4 cores | Processor seconds to decode, 1 core | Million weights a second |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5 | `PZ32:b136s8` | 736,991,766 | 4.50004 | yes | 8.4% | 61 | 54.2 | 24.1 |
| Qwen2.5 | `hPZ32:b136s8` | 738,009,622 | 4.50626 | yes | 8.8% | 57 | 54.0 | 24.3 |
| SmolLM2 | `PZ32:b136s8` | 905,975,382 | 4.50003 | yes | 8.2% | 74 | 64.4 | 25.0 |
| SmolLM2 | `hPZ32:b136s8` | 906,958,422 | 4.50491 | yes | 8.4% | 83 | 81.8 | 19.7 |

Beyond the 4.5 bits of the blocks, a file without column weights carries 0.00003 to 0.00004 bits a weight: its header, and a name, a shape and a scale for each matrix. With column weights it carries 0.005 to 0.006 more. No block had to be shrunk to fit the ball, in any of the four files.

### The model each file holds

| | Bits | Qwen2.5: mean KL, first choice kept | SmolLM2 |
|---|---|---|---|
| `PZ32:b136s8`, from the file | 4.500 | 0.0245, 92.4% | 0.0589, 87.7% |
| `ZBG128:b560s16`, counted | 4.500 | 0.0227, 92.8% | 0.0560, 88.3% |
| llama.cpp `Q4_K_M` | 4.81, 4.82 | 0.0333, 91.1% | 0.0838, 85.5% |
| llama.cpp `Q4_0` | 4.500 | 0.0623, 88.2% | 0.1271, 82.4% |
| `hPZ32:b136s8`, from the file | 4.506, 4.505 | 0.0227, 92.9% | 0.0309, 91.1% |
| `hZBG128:b560s16`, counted | 4.506, 4.505 | 0.0209, 92.6% | 0.0282, 91.4% |
| llama.cpp `Q4_K_M` with the matrix | 4.81, 4.82 | 0.0282, 91.8% | 0.0378, 89.6% |

| Comparison | Qwen2.5: lower by, in errors | SmolLM2 |
|---|---|---|
| `PZ32:b136s8` against `Q4_K_M` | 10.5 | 13.2 |
| `PZ32:b136s8` against `Q4_0`, the same bits | 22.4 | 25.9 |
| `hPZ32:b136s8` against `Q4_K_M` with the matrix | 7.1 | 6.8 |

### What the shorter block costs

| | Weight error, dB | Mean KL | KL over that of the block of 128 |
|---|---|---|---|
| Qwen2.5, `PZ32:b136s8` against `ZBG128:b560s16` | -24.44 against -24.94 | 0.0245 against 0.0227 | 1.078 |
| Qwen2.5, with the matrix | -23.72 against -24.22 | 0.0227 against 0.0209 | 1.085 |
| SmolLM2, `PZ32:b136s8` against `ZBG128:b560s16` | -24.41 against -24.93 | 0.0589 against 0.0560 | 1.052 |
| SmolLM2, with the matrix | -24.03 against -24.50 | 0.0309 against 0.0282 | 1.097 |

The block of 32 with a one-byte scale has 0.47 to 0.52 dB more weight error. Half a decibel is a factor of 1.12, and the KL rose by 1.05 to 1.10. Each of the four gaps is 1.9 to 3.0 errors, so each alone is close to a tie. All four go the same way.

The arm that differs from this format in one thing only, the 16-bit scale and the column weights kept as 64-bit floats, was not scored. So these ratios are the cost of the block and the scale together with whatever storing the scales costs, which the weight errors say is nothing visible.

### The predictions

| | Prediction | Result | Holds |
|---|---|---|---|
| F1 | Every file is read back exactly, at no more than 4.501 bits without column weights and 4.52 with | exact in all four. 4.50004, 4.50003, 4.50626, 4.50491 | yes |
| F2 | `PZ32:b136s8` is under `Q4_K_M` by 2 errors, on each model | 10.5 and 13.2 errors | yes |
| F3 | `hPZ32:b136s8` is under `Q4_K_M` with the matrix by 2 errors, on each model | 7.1 and 6.8 errors | yes |
| F4 | Each file's KL is no more than 1.25 times that of its counted arm with blocks of 128 | 1.078, 1.085, 1.052, 1.097 | yes |
| F5 | Each file decodes at 15 million weights a second or more on one core | 24.1, 24.3, 25.0, 19.7 | yes |

The plan's own figure for F4 was 1.13, from the weight error. The KL rose by less than that on all four.

## Deviations from the plan

- **The run on Qwen2.5 was started twice, 17 seconds apart.** The driver script was replaced by one with a longer queue. Nothing had been packed.
- **The machine restarted while the fourth variant was being scored.** `hPZ32:b136s8` on SmolLM2 was packed again from the start, and the row is of that second pass. The file of the first pass is gone, so the two cannot be compared. `results.start` has the four starts.
- **The decode of the fourth file is slower than the other three**, 19.7 against 24 to 25 million weights a second. It was timed after the restart. I do not know what else the machine was doing. The timing is of processor time and not of the clock, and the scoring tool was not running, but a watcher that commits rows was.
- No program was changed after the plan.

## Limits

- **Nothing is saved when the model runs.** llama.cpp cannot load the file. It was decoded to 16-bit floats, and the model was scored from those. The saving is on disk and on the wire.
- **The decode is slow against a kernel.** At 20 to 25 million weights a second a core, a model of 1.3 billion block weights takes about a minute on one core. On this machine llama.cpp's tool goes through the 1.5 billion weights of a `Q4_K_M` file some 120 times a second on four cores. A number of 136 bits has to be taken apart one weight at a time, each step waiting on the one before. The coder is plain C with a linear search over symbols, and nothing was done to make it fast.
- **The encoder is NumPy.** It packs a model in one to one and a half minutes on four cores.
- **The file format exists to count bytes.** It has no alignment, no index of matrices and no version beyond a magic word. Only the block matrices are in it: the embeddings, the norms and the output matrix stay in the reference file at 16 bits, as in every arm of these records.
- **Two models of 1.5 and 1.7 billion weights, one text.** With the importance matrix, the calibration text is the training half of the test text's source, as in the importance-matrix record.
- **The standard errors are the tool's** and treat positions as independent. Gaps of 2 or 3 errors are ties. F2 and F3 rest on 6.8 errors or more.
- **Not new.** Numbering the points of a ball is enumerative coding (Cover, 1973), and in transmission it is sphere shaping. `shape/ballindex.py` says so. What is here is a file, a check and a score.

## Read against the closure rule

Nothing is closed by a certificate. "Read back exactly" is a check on these four files: a SHA-256 for every matrix, compared between what the encoder produced and what was decoded from the bytes on disk. The coder is tested against `shape/ballindex.py` on 300 random points and on every point of a small codebook (`shape/test_ballpack.py`). That it numbers every point of the ball of 2^136 points rightly is not proved.

**What changes in the repository's account.** The records of 2026-11-17 end their headline with "counted bits, no packed format". For integer levels in one ball per block of 32 weights that can now read "4.5 bits a weight in a file". For the arms with blocks of 128 and for every E8 arm it still stands: none of them has a file.

**What follows.**

1. A reader for the file inside an inference engine would be the first thing that saves memory. It would have to decode a matrix when it is needed, or compute from numbers of 136 bits. Neither is attempted here, and the decode speed above says the first is a load-time step.
2. Blocks of 128 gain 5 to 10 percent of KL and need numbers of 560 bits and a table of 30 megabytes. Whether that is worth it is a question of the reader, not of the rounding.
3. The MMLU subset of issue #14 scores the counted arms. If it is to bear on the file, the file's arms would be scored there too.

## How to rerun

Needs NumPy, torch, a C compiler, llama.cpp's `gguf` package and `llama-perplexity`, the model files, their reference outputs and their importance matrices.

```bash
python3 -m unittest shape/test_ballpack.py
```

```bash
python3 experiments/packed-ball-format-2026-11-18/run.py --model qwen2.5-1.5b-instruct-fp16.gguf --kld base.kld --work DIR --out results.jsonl --bin llama-perplexity --imatrix qwen15.imatrix.gguf PZ32:b136s8 hPZ32:b136s8
```

```bash
python3 experiments/packed-ball-format-2026-11-18/summary.py > results.out
```

A variant takes one to one and a half minutes to pack, three to six to read back and time, and four to five to score, on 4 CPU cores.

## Disclosure

Written by an AI agent session under the owner's direction, on 2026-11-18. The format, the coder, the plan, the run and this record are the session's.
