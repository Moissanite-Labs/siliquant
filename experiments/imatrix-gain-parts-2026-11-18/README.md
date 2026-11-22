# What the gain of an importance matrix is made of

## Question

An importance matrix halves the KL of the ball arm on SmolLM2 at 4.5 bits: 0.0560 without it, 0.0282 with it. It does the same for llama.cpp's `Q4_K_M`. On Qwen2.5 it buys little. The record loud-columns-2026-11-18 looked for SmolLM2's gain in the columns that a clip must not touch and found a ninth of it there.

**A. Which matrices is the gain in?**

**B. Does it need text?** The matrix holds the mean square of every matrix's inputs over a calibration text. If what it finds is a property of the model, a text with no meaning would find it too.

This belongs to issue #14.

## Status

`complete`, 2026-11-18. 11 variants on two models: 7 of the plan, 2 of extension 1 and 2 of extension 2. 10 of 12 predictions hold. One table was not planned and is marked.

## Preregistration

[PREREG.md](PREREG.md), committed and pushed at `453d907` with `run.py`, `summary.py`, `randtext.py` and the forecast for SmolLM2, before any variant was scored. The forecasts for the matrix of no meaning were committed at `0cdc5f4` and `b10a7de`, each before any variant that uses that matrix was scored. Extension 1 at `a68145f`, before any variant of the plan was scored. Extension 2 at `932781d`, with four rows of the plan in, which it states.

## Result in short

1. **On SmolLM2 two thirds of the gain is in the three matrices through which the attention reads the stream.** With only `attn_q`, `attn_k` and `attn_v` weighted, 18.75 percent of the weights, the ball arm scores 0.0369: that is 68.5 percent of the whole gain. The feed-forward readers, half of the weights, give 9.5 percent. The two writers give 21.9. The three add up to 99.9 percent of the whole.
2. **The reason is in the forecast, made before any score.** What the attention reads on SmolLM2 sits in ten dimensions of the stream of 2,048: they hold 61.8 percent of its mean square. A rule that rounds every column alike spends its error on them.
3. **The matrix does not need a text with a meaning.** A matrix made from 80,000 pieces of the model's own vocabulary drawn at random buys the ball arm 88.6 percent of what the wikitext matrix buys on SmolLM2: 0.0314 against 0.0282. It buys llama.cpp's `Q4_K_M` 95.5 percent: 0.0399 against 0.0378, where no matrix gives 0.0838.
4. **On Qwen2.5 it does the ball arm neither good nor harm**: 0.0228 against 0.0227 with no matrix. It buys `Q4_K_M` 76.5 percent of what the real matrix buys: 0.0294 against 0.0282, where no matrix gives 0.0333.
5. **So with no text read, the ball arm at 4.5 bits is under `Q4_K_M` at 4.8 bits made with real text, on both models**: 0.0314 against 0.0378, 6.6 errors, and 0.0228 against 0.0282, 7.1 errors. This comparison had no prediction.
6. **As a file it is ahead of `Q4_K_M` on Qwen2.5 and level with it on SmolLM2. I predicted ahead on both.** The 18-byte format with the matrix of no meaning scores 0.0358 on SmolLM2, 1.8 errors under `Q4_K_M` with the real matrix, where the prediction asked for 2. On Qwen2.5 it scores 0.0256, 3.3 errors under. But on Qwen2.5 the file made with no matrix at all scores 0.0245: there the meaningless matrix is of no use to the file and may cost it a little.
7. **The meaningless matrix should weight the writers too.** I predicted that leaving them plain would lose nothing. On SmolLM2 it loses 4.5 errors: 0.0357 against 0.0314.

What it does not mean: the model is still run, on 51,200 tokens of noise, to make the matrix. No dataset is needed. Forward passes are.

## What was run

**The models, the text and the score** are those of the model-output records: `SmolLM2-1.7B-Instruct` and `Qwen2.5-1.5B-Instruct`, 16-bit files, the wikitext-2 test text, 10,200 positions, mean KL against the reference's saved outputs, with `llama-perplexity --kl-divergence` of llama.cpp at `eada89c`.

**The two arms.** `ZBG128:b560s16`: integer levels in one ball per group of 128 weights, 4.5 bits, no data. `hZBG128:b560s16`: the same with each column multiplied by (its input's mean square over the mean) to the power 0.25 before rounding and divided after. The power was chosen on validation text in the record imatrix-baseline-2026-11-17.

**Part A, SmolLM2.** The weighted arm on some kinds of matrix and the plain arm on the rest, with the matrix of the wikitext-2 training text: the attention's readers (`attn_q`, `attn_k`, `attn_v`), the feed-forward readers (`ffn_gate`, `ffn_up`), the writers (`attn_output`, `ffn_down`).

**Part B, both models.** `randtext.py` draws 80,000 pieces of the model's own vocabulary at random, each ordinary piece with the same chance, seed 1. `llama-imatrix` reads 100 passages of 512 tokens of that text, by the command of the imatrix-baseline record. The model's perplexity on it is 101,588 for SmolLM2 and 340,141 for Qwen2.5 (`rimatrix.out`): it is noise to the model. Then `hZBG128:b560s16~rand`, the weighted arm with that matrix, and `Q4_K_M-rimat`, llama.cpp's `Q4_K_M` made with that matrix by the `llama-quantize` command of the imatrix-baseline record.

**Extension 1.** The matrix of no meaning on the five reading kinds only, the writers plain.

**Extension 2.** `hPZ32:b136s8` by the program of the record packed-ball-format-2026-11-18, unchanged, given the matrix of no meaning. It writes the file, reads it back from disk, stops if a matrix read back is not bit for bit the one the encoder gave, and scores the model the file holds. Rows in `packed.jsonl`.

**The forecast.** `run.py --forecast` rounds every block matrix by each arm and counts its squared error with each column multiplied by its input's mean square over the wikitext-2 training text. Nothing is scored in it. It is what the output of that matrix would lose if its inputs did not move together.

## Results

Every number is in `results.out`, printed by `summary.py` from `results.jsonl`, `packed.jsonl` and the rows of four earlier records, or in the `forecast` and `guess` tables.

### Part A: which matrices the gain is in, on SmolLM2

| Weighted by the training text's matrix | Share of the weights | Mean KL | Under the plain arm | In errors | Share of the whole gain |
|---|---|---|---|---|---|
| nothing, `ZBG128:b560s16`, earlier record | | 0.0560 | | | |
| the attention's readers | 18.75% | 0.0369 | 34.0% | 14.4 | 0.685 |
| the feed-forward readers | 50% | 0.0533 | 4.7% | 1.8 | 0.095 |
| the writers | 31.25% | 0.0499 | 10.8% | 4.2 | 0.219 |
| every kind, `hZBG128:b560s16`, earlier record | 100% | 0.0282 | 49.6% | 22.7 | 1.000 |

The three shares sum to 0.999. Nothing said a KL must split into parts. Here it does.

What the forecast said, for SmolLM2, before any score:

| Kind | Share of the input in the ten most active columns | What weighting gains, counted by activity, dB |
|---|---|---|
| `attn_q`, `attn_k`, `attn_v` | 61.8% | 3.99, 3.79, 4.51 |
| `ffn_gate`, `ffn_up` | 17.0% | 0.77, 0.88 |
| `attn_output` | 13.7% | 1.02 |
| `ffn_down` | 15.5% | 1.41 |

It put the gain in the attention's readers, and it is there. It gave the two readers of the feed-forward block the least, and they have the least. It does not give sizes: 4 dB on a fifth of the weights became two thirds of the gain.

### Part B: a matrix that read no text

| Model | Arm | Matrix | Bits a weight | Mean KL | First choice kept | Share of the real matrix's gain |
|---|---|---|---|---|---|---|
| SmolLM2 | llama.cpp `Q4_K_M` | none, earlier record | 4.8223 | 0.0838 | 85.5% | |
| SmolLM2 | llama.cpp `Q4_K_M` | wikitext-2 training text, earlier record | 4.8223 | 0.0378 | 89.6% | |
| SmolLM2 | llama.cpp `Q4_K_M` | no meaning | 4.8223 | 0.0399 | 89.7% | 0.955 |
| SmolLM2 | the ball arm | none, earlier record | 4.4998 | 0.0560 | 88.3% | |
| SmolLM2 | the ball arm | wikitext-2 training text, earlier record | 4.5047 | 0.0282 | 91.4% | |
| SmolLM2 | the ball arm | no meaning | 4.5047 | 0.0314 | 90.7% | 0.886 |
| SmolLM2 | the ball arm | no meaning, readers only | 4.5022 | 0.0357 | 90.1% | 0.729 |
| Qwen2.5 | llama.cpp `Q4_K_M` | none, earlier record | 4.8120 | 0.0333 | 91.1% | |
| Qwen2.5 | llama.cpp `Q4_K_M` | wikitext-2 training text, earlier record | 4.8120 | 0.0282 | 91.8% | |
| Qwen2.5 | llama.cpp `Q4_K_M` | no meaning | 4.8120 | 0.0294 | 91.5% | 0.765 |
| Qwen2.5 | the ball arm | none, earlier record | 4.4998 | 0.0227 | 92.8% | |
| Qwen2.5 | the ball arm | wikitext-2 training text, earlier record | 4.5060 | 0.0209 | 92.6% | |
| Qwen2.5 | the ball arm | no meaning | 4.5060 | 0.0228 | 92.8% | -0.045 |
| Qwen2.5 | the ball arm | no meaning, readers only | 4.5024 | 0.0233 | 92.7% | -0.293 |

On Qwen2.5 the ball arm with the matrix of no meaning is 0.1 errors above the arm with no matrix, and the readers-only arm 0.7 errors above. Both are ties.

The matrix of no meaning finds the loud dimensions. On SmolLM2 the ten columns it puts first hold 57.5 percent of the attention's input over the training text, where the training text's own ten hold 61.8. On Qwen2.5 there is little to find: the training text's own ten hold 11.4 percent.

**Where the forecast was wrong.** For SmolLM2 it had the matrix of no meaning as worse than no matrix on `attn_output` and a third as good as the real one on `ffn_down`. Extension 1 was written from that, and E1 failed: with the writers left plain the arm loses 0.0044 of KL. The two arms differ by 0.157 of the whole gain: that is what the meaningless matrix buys on the writers, where the real one buys 0.219. So counting a writer's error by its inputs' mean square over one text does not say what a matrix from another text is worth to it.

### Extension 2: the arm as a file

| Model | Matrix | Bytes of the file | Bits a weight | Mean KL | First choice kept | Under `Q4_K_M` with the real matrix, in errors |
|---|---|---|---|---|---|---|
| SmolLM2 | none, packed-format record | 905,975,382 | 4.50003 | 0.0589 | 87.7% | -16.2 |
| SmolLM2 | wikitext-2 training text, packed-format record | 906,958,422 | 4.50491 | 0.0309 | 91.1% | 6.8 |
| SmolLM2 | no meaning | 906,958,422 | 4.50491 | 0.0358 | 90.9% | 1.8 |
| Qwen2.5 | none, packed-format record | 736,991,766 | 4.50004 | 0.0245 | 92.4% | 4.6 |
| Qwen2.5 | wikitext-2 training text, packed-format record | 738,009,622 | 4.50626 | 0.0227 | 92.9% | 7.1 |
| Qwen2.5 | no meaning | 738,009,622 | 4.50626 | 0.0256 | 92.0% | 3.3 |

Both files were read back from disk bit for bit. `Q4_K_M` with the real matrix scores 0.0378 on SmolLM2 and 0.0282 on Qwen2.5, at 4.82 and 4.81 bits.

- **On SmolLM2 the meaningless matrix is worth having in the file**: 0.0358 against 0.0589 with none, 16.8 errors. It leaves the file 1.8 errors under `Q4_K_M` made with real text, short of the 2 the plan asked for, at 7 percent fewer bits.
- **On Qwen2.5 it is not.** The file made with it is 1.7 errors above the file made with no matrix. The counted arm with groups of 128 showed no such cost, 0.1 errors. So "no harm where there is nothing to find" holds for the counted arm and is not established for the file.
- The file made with no text is under `Q4_K_M` made with real text on Qwen2.5 either way: by 3.3 errors with the meaningless matrix and by 4.6 with none.

### Not planned: guesses from the weights alone

`guess.py`, made after the first rows were in. Nothing is scored in it. It asks whether the model has to be run at all: could the column weights of the reading matrices be guessed from the weights? Three guesses of how loud a dimension of the stream is, each put in the place of the importance matrix and judged by the forecast's count.

What the weighted arm gains over the plain arm, in dB, counted by the training text's activity. Mean over the layers.

| Model | Kind | With the importance matrix | Guess: the norm's gain | Guess: gain times a guess of the stream | Guess: small weights |
|---|---|---|---|---|---|
| SmolLM2 | `attn_q` | 3.99 | 1.15 | 1.90 | 0.25 |
| SmolLM2 | `attn_k` | 3.79 | 1.21 | 1.92 | -0.49 |
| SmolLM2 | `attn_v` | 4.51 | 1.39 | 2.10 | 2.82 |
| SmolLM2 | `ffn_gate` | 0.77 | -0.63 | -0.26 | -0.05 |
| SmolLM2 | `ffn_up` | 0.88 | -0.65 | -0.27 | 0.30 |
| Qwen2.5 | `attn_q` | 0.06 | -0.80 | -1.05 | -0.55 |
| Qwen2.5 | `attn_k` | -0.24 | -0.51 | -0.68 | -0.68 |
| Qwen2.5 | `attn_v` | 0.43 | -0.81 | -0.87 | 0.42 |
| Qwen2.5 | `ffn_gate` | -0.03 | -0.87 | -1.07 | -0.62 |
| Qwen2.5 | `ffn_up` | 0.51 | -1.08 | -1.23 | 0.08 |

The steadiest guess gets half of the importance matrix's forecast on SmolLM2's attention readers, and loses about a decibel to no weighting at all on every reading kind of Qwen2.5. None of the three was scored. By this count the model has to be run: its weights do not say which dimensions of its stream are loud well enough to round by.

### The predictions

| | Prediction | Result | Holds |
|---|---|---|---|
| P1 | The attention's readers' share of the whole gain is over half | 0.685 | yes |
| P2 | The feed-forward readers' share is under a quarter | 0.095 | yes |
| P3 | The writers' share is under a quarter | 0.219 | yes |
| P4 | The three shares sum to between 0.75 and 1.25 | 0.999 | yes |
| R1 | SmolLM2: the matrix of no meaning buys the ball arm at least half of what the real one buys | 0.886 | yes |
| R2 | Qwen2.5: the ball arm with it is not above the plain arm by 2 errors | 0.1 errors above | yes |
| R3 | SmolLM2: it buys `Q4_K_M` at least half of what the real one buys | 0.955 | yes |
| R4 | Qwen2.5: `Q4_K_M` with it is not above `Q4_K_M` with no matrix by 2 errors | 4.4 errors under | yes |
| E1 | SmolLM2: the readers-only arm is not above the arm with every kind weighted by 2 errors | 4.5 errors above | **no** |
| E2 | Qwen2.5: the readers-only arm is not above the plain arm by 2 errors | 0.7 errors above | yes |
| F1 | SmolLM2: the file made with no text is under `Q4_K_M` with the real matrix by 2 errors | 1.8 errors | **no** |
| F2 | Qwen2.5: the same | 3.3 errors | yes |

What I got wrong.

- **E1.** I read the forecast as saying that a meaningless matrix is of no use to the writers. It is of use: less than the real one, and more than nothing.
- **F1.** I took the file to cost 1.10 times the counted arm, as it did with the real matrix, and counted on a margin of 3 errors. With the meaningless matrix the file costs more than that, and the margin is 1.8.
- **My confidence.** I gave low confidence to P2, P4, R2 and R4, and all four hold. The parts add up far more exactly than I had any reason to expect.

## Deviations from the plan

- **The queue stopped for ten minutes on a fault of the program.** The check that skips variants already scored read a `text` field from every row, and the row of a llama.cpp file, written by the earlier record's program, has none. The driver stopped at 20:11 UTC before any Qwen2.5 variant was built. Fixed at `2edd30c` and started again. No row was lost and no variant is built differently. `results.start` has the two extra starts.
- **Extensions 1 and 2 were added after the plan.** `run.py` and `summary.py` were extended for them. The planned variants come out bit for bit as before on the single-matrix test.
- **`summary.py` prints two things more than planned**: the ball arm with the matrix of no meaning against `Q4_K_M` with the real matrix on Qwen2.5 as well as on SmolLM2, and the file made with no matrix beside the two files made with one. Neither has a prediction. It also prints the shares as percentages and the gap of extension 1 as a KL, so that every figure of this record is in `results.out`.
- **`guess.py` was not planned.** Its table is marked above.

## Limits

- **Two models.** One has loud dimensions in its stream and one has few. What a third has is not known. The matrix of no meaning helped where there was something to find and did no harm where there was not, on these two.
- **One text for the score**, of the same source as the real matrix's text. That favours the real matrix. How the two matrices compare on text of another kind is not tested, and it is the comparison where a matrix from noise might do better.
- **One seed, one length, one way of making a meaningless text.** The pieces are the model's own vocabulary.
- **One strength for the ball arm**, 0.25, chosen with the real matrix.
- **The standard errors are the tool's** and treat positions as independent. The gaps the conclusions rest on are 4 errors and more. On Qwen2.5 the ball arm's three figures, 0.0227, 0.0228 and 0.0233, are not told apart.
- **The file on Qwen2.5.** With blocks of 32 the meaningless matrix leans towards harm there, 1.7 errors. Whether it does harm is not settled, and nothing here says how to tell beforehand which model wants the matrix.
- **KL only.** No task was scored with the matrix of no meaning.
- **The forecast is one way of counting**, and it was wrong about the writers.
- **Why SmolLM2's stream has loud dimensions.** Not asked. The loud-columns record found one hidden unit in each of two layers that writes into a few of them.

## Read against the closure rule

Nothing is closed by a certificate. These are 11 measurements on two models.

**What changes in the repository's account.**

1. The imatrix-baseline record said the ball arm with no data keeps its lead over `Q4_K_M` with an importance matrix on one model and loses it on the other. With a matrix that reads no text the counted arm keeps it on both, and the file keeps it on Qwen2.5 and is level on SmolLM2. "No data" in that record meant no forward pass. Here it means no text.
2. On these two models most of what an importance matrix gives `Q4_K_M` does not come from the text: 95.5 and 76.5 percent of it comes from noise.
3. The loud-columns record asked where the rest of SmolLM2's gain is. Two thirds of it is in the attention's three reading matrices.
4. An arm that reads only the weights is not reached. The guesses of `guess.py` get at most half of the forecast on one model and lose on the other.

**What follows.**

1. For the sprint of issue #15: a matrix from noise is worth trying before any dataset is. It needs the model and a quarter of an hour of forward passes on this machine. On SmolLM2 it takes the file from 0.0589 to 0.0358. On Qwen2.5 the file is better without it.
2. A rule for when to use it. The plain candidate: how much of the attention's input the matrix's own first ten columns hold, which is 57.5 percent on SmolLM2 and 8.5 on Qwen2.5 counted by the real text. Not tested, and two models cannot test it.
3. A score on text of another kind would say whether a matrix from noise is the safer one when the text to come is not known.
4. A task benchmark with the matrix of no meaning. The record mmlu-subset-2026-11-18 has the program.

## How to rerun

Needs NumPy, torch, llama.cpp's `gguf` package, `llama-imatrix`, `llama-quantize` and `llama-perplexity`, a model file, its reference outputs on the test text and its importance matrix from the wikitext-2 training text.

```bash
python3 experiments/imatrix-gain-parts-2026-11-18/randtext.py --model SmolLM2-1.7B-Instruct-f16.gguf --pieces 80000 --seed 1 --out rand.txt
llama-imatrix -m SmolLM2-1.7B-Instruct-f16.gguf -f rand.txt --chunks 100 -c 512 -t 4 -o smollm2.rimatrix.gguf
```

```bash
python3 experiments/imatrix-gain-parts-2026-11-18/run.py --model SmolLM2-1.7B-Instruct-f16.gguf --imatrix smollm2.imatrix.gguf --imatrix-rand smollm2.rimatrix.gguf --forecast > forecast-smollm2-rand.out
```

```bash
python3 experiments/imatrix-gain-parts-2026-11-18/run.py --model SmolLM2-1.7B-Instruct-f16.gguf --imatrix smollm2.imatrix.gguf --imatrix-rand smollm2.rimatrix.gguf --kld base.kld --text test --work DIR --out results.jsonl --bin llama-perplexity 'ZBG128:b560s16+h@attn_q,attn_k,attn_v' 'hZBG128:b560s16~rand'
```

```bash
python3 experiments/packed-ball-format-2026-11-18/run.py --model SmolLM2-1.7B-Instruct-f16.gguf --kld base.kld --work DIR --out packed.jsonl --bin llama-perplexity --imatrix smollm2.rimatrix.gguf hPZ32:b136s8
```

```bash
python3 experiments/imatrix-gain-parts-2026-11-18/summary.py > results.out
```

The matrix takes 13 to 15 minutes on 4 CPU cores, a variant about seven.

## Disclosure

Written by an AI agent session under the owner's direction, on 2026-11-18. The plan, the programs, the run and this record are the session's. The question is that of issue #14. The importance matrix, `llama-imatrix` and `Q4_K_M` are llama.cpp's.
