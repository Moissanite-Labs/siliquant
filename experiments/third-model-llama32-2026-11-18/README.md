# A third model: Llama-3.2-1B-Instruct

## Question

The headline comparisons of this branch rest on two models, and the two disagree on what an importance matrix is worth to the ball arm. The last two records pointed at a rule: weight the columns where the stream has loud dimensions, and not where it has few.

Do the comparisons hold on a model of a third family? And does the rule call it right?

## Status

`complete`, 2026-11-19. 19 variants: 12 of the plan, 3 of extension 1 and 4 of extension 2. 10 of 13 predictions hold. The three that fail are the two comparisons with no matrix, and the rule.

## Preregistration

[PREREG.md](PREREG.md), committed and pushed at `b6da101`, before anything was run on the model. `reference.out`, `matrices.out` and `forecast.out` were committed by `636a9de`, before any variant was scored. Extension 1 at `10d0b93` and extension 2 at `f0ddca3`, each before its own variants were scored and with the rows before it in, which each states.

## Result in short

1. **With no matrix on either side the ball arm is behind `Q4_K_M` on this model.** 0.0478 against 0.0432 on the wikitext text, 4.9 errors behind. 0.0456 against 0.0371 on the Python text, 8.3 behind. On the two earlier models it was 9 to 15 errors ahead. So the first headline line of this branch holds on two models of three.
2. **With a matrix it is far ahead, on both texts.** With the wikitext matrix on both sides: 0.0242 against 0.0359, 18.2 errors, and 17.0 on Python. With a matrix from noise on its side against the wikitext matrix on the other: 11.1 and 13.7 errors.
3. **The rule is refuted.** The ten loudest dimensions hold 13.8 percent of what the attention reads here, close to Qwen2.5's 11.4, so the rule said a matrix would not help on Python. It halves the ball arm's KL on both texts, whether it was made from Wikipedia or from noise.
4. **Why the rule looked in the wrong place.** On SmolLM2 the matrix's gain was in the matrices through which the attention reads the stream. Here those give nothing. 92.6 percent of the gain is in the two writers, `attn_output` and `ffn_down`.
5. **And it is two hidden units.** With one column of `ffn_down` left whole in each of layers 0 and 1, 4,096 weights of 973 million, the ball arm with no matrix scores 0.0302 where it scored 0.0478. That is three quarters of what the wikitext matrix buys.
6. **The weights alone find them.** Leaving whole the ten loud columns of every `ffn_down`, the rule of the record loud-columns-2026-11-18, gives the same 0.0302 at 4.505 bits. It reads no data and runs no forward pass. With it the ball arm is under `Q4_K_M` with no matrix by 17.3 errors on the wikitext text and 15.3 on Python: the comparison this model broke is mended.
7. **That rule now stands on three models.** On the wikitext text the ball arm with no data and those columns left whole is under `Q4_K_M` with no matrix by 12.5 errors on Qwen2.5, 16.8 on SmolLM2 and 17.3 here.

So each model has its own place where a rule that rounds every column alike loses. SmolLM2: ten dimensions of the stream, read by the attention. This model: one hidden unit in each of two layers. Qwen2.5: nowhere much. A matrix finds both kinds, from text or from noise. The weights alone find the second kind and not the first.

## What was run

**The model.** `Llama-3.2-1B-Instruct`, the 16-bit file of the `bartowski/Llama-3.2-1B-Instruct-GGUF` repository: 2,479,595,360 bytes, SHA-256 starts `1f33ad43`. 16 layers, 973 million block weights.

**Two texts.** The wikitext-2 test text and the Python file of the record second-text-python-2026-11-18. 10,200 positions each. The 16-bit model's perplexity is 14.3 on the first and 6.4 on the second.

**Two matrices**, by `llama-imatrix` from 100 passages: of the wikitext-2 training text, and of a text of 80,000 random pieces of this model's vocabulary. The model's perplexity on the second is 165,296.

**The plan's twelve variants.** `Q4_K_M` and the ball arm `ZBG128:b560s16`, each with no matrix, the wikitext matrix and the noise matrix, on each text.

**Extension 1.** The weighted arm on the attention's readers only, on the feed-forward readers only and on the writers only, on the wikitext text.

**Extension 2.** The ball arm with no matrix and columns of `ffn_down` left as in the reference, counted at 16 bits an entry: the ten loud columns of every layer (`+K10`), of layers 0 and 1 only (`+K10#0,1`), and one column in each of layers 0 and 1 picked by the matrix (`+A1#0,1`). A loud column is one of the ten in which a clip at 5 standard deviations of a row removes the most squared weight.

**The programs** are those of the earlier records. `run_arms.py`, `run_rand.py` and `run_loud.py` add this model's file to the files those programs accept, and nothing else.

## Results

Every number is in `results.out`, printed by `summary.py` from the three results files, `forecast.out` and the rows of six earlier records, or in `reference.out`, `matrices.out` and `picks.out`.

### The plan: six variants on two texts

| Text | Arm | Matrix | Bits a weight | Mean KL | First choice kept |
|---|---|---|---|---|---|
| wikitext-2 | llama.cpp `Q4_K_M` | none | 4.8023 | 0.0432 | 89.0% |
| wikitext-2 | llama.cpp `Q4_K_M` | wikitext | 4.8023 | 0.0359 | 90.0% |
| wikitext-2 | llama.cpp `Q4_K_M` | noise | 4.8023 | 0.0382 | 89.4% |
| wikitext-2 | the ball arm | none | 4.4998 | 0.0478 | 88.9% |
| wikitext-2 | the ball arm | wikitext | 4.5052 | 0.0242 | 91.5% |
| wikitext-2 | the ball arm | noise | 4.5052 | 0.0284 | 90.5% |
| Python | llama.cpp `Q4_K_M` | none | 4.8023 | 0.0371 | 92.5% |
| Python | llama.cpp `Q4_K_M` | wikitext | 4.8023 | 0.0341 | 92.2% |
| Python | llama.cpp `Q4_K_M` | noise | 4.8023 | 0.0347 | 92.1% |
| Python | the ball arm | none | 4.4998 | 0.0456 | 91.4% |
| Python | the ball arm | wikitext | 4.5052 | 0.0217 | 94.0% |
| Python | the ball arm | noise | 4.5052 | 0.0240 | 93.9% |

The noise matrix buys the ball arm 82.4 percent of what the wikitext matrix buys on the wikitext text, and 90.4 percent on Python. It buys `Q4_K_M` 68.5 and 79.6 percent.

### Three models side by side

How far the ball arm is under `Q4_K_M`, in errors. A minus is behind. The two earlier models are read from the records that scored them.

| Text | Model | No matrix on either side | The wikitext matrix on both sides | Noise for the ball arm, wikitext for `Q4_K_M` |
|---|---|---|---|---|
| wikitext-2 | Qwen2.5 | 11.4 | 9.2 | 7.1 |
| wikitext-2 | SmolLM2 | 14.7 | 10.0 | 6.6 |
| wikitext-2 | Llama-3.2 | **-4.9** | 18.2 | 11.1 |
| Python | Qwen2.5 | 9.3 | 5.0 | 3.6 |
| Python | SmolLM2 | 13.2 | 11.2 | 8.7 |
| Python | Llama-3.2 | **-8.3** | 17.0 | 13.7 |

The two comparisons with a matrix hold in all six cells. The one with no matrix holds in four.

What a matrix is worth to the ball arm, in errors under the plain ball arm:

| Text | Model | The wikitext matrix | The noise matrix |
|---|---|---|---|
| wikitext-2 | Qwen2.5 | 2.3 | -0.1 |
| wikitext-2 | SmolLM2 | 22.7 | 19.8 |
| wikitext-2 | Llama-3.2 | 29.3 | 23.2 |
| Python | Qwen2.5 | -2.2 | -3.0 |
| Python | SmolLM2 | 24.9 | 22.3 |
| Python | Llama-3.2 | 26.8 | 24.0 |

### The rule, and why it failed

The plan fixed the rule before the model was looked at: if the ten loudest dimensions hold 20 percent or less of what the attention reads, neither matrix puts the ball arm under the plain arm by 2 errors on the Python text. The share came out at 13.8. Both matrices put it under by 24 to 27 errors.

Extension 1, on the wikitext text. The whole gain is from 0.0478 to 0.0242.

| Weighted by the wikitext matrix | Mean KL | Share of the whole gain | On SmolLM2 |
|---|---|---|---|
| the attention's readers | 0.0484 | -2.5% | 68.5% |
| the feed-forward readers | 0.0466 | 5.2% | 9.5% |
| the writers | 0.0260 | 92.6% | 21.9% |

The three sum to 95.4 percent. So the share measured the input of matrices that hold none of this model's gain. The rule took one model's place for the place.

### Extension 2: a few columns left whole

`picks.out`, committed before any of these was scored: the ten loud columns of layer 1's `ffn_down` hold 99.0 percent of that matrix's input activity, of layer 0's 51.9 percent, of the last layer's 31.0, and under 5 percent in the other thirteen layers.

| Text | Arm | Reads data | Bits a weight | Columns kept | Mean KL | Under the plain ball arm | Under `Q4_K_M` with no matrix, in errors |
|---|---|---|---|---|---|---|---|
| wikitext-2 | the ball arm, no matrix | no | 4.4998 | 0 | 0.0478 | | -4.9 |
| wikitext-2 | `+K10` | no | 4.5052 | 160 | 0.0302 | 36.8% | 17.3 |
| wikitext-2 | `+K10#0,1` | no | 4.5005 | 20 | 0.0301 | 36.9% | 17.4 |
| wikitext-2 | `+A1#0,1` | yes | 4.4999 | 2 | 0.0302 | 36.7% | 17.3 |
| Python | the ball arm, no matrix | no | 4.4998 | 0 | 0.0456 | | -8.3 |
| Python | `+K10` | no | 4.5052 | 160 | 0.0256 | 43.9% | 15.3 |

Two columns, twenty and a hundred and sixty give the same score. All of it is one hidden unit in each of the first two layers. Keeping them buys 74.6 percent of what the wikitext matrix buys.

The same arm on three models, wikitext text, with no data and no forward pass:

| Model | Bits a weight | Mean KL | Under the plain ball arm, errors | Under `Q4_K_M` with no matrix | Under `Q4_K_M` with the wikitext matrix |
|---|---|---|---|---|---|
| Qwen2.5 | 4.5051 | 0.0222 | 0.6 | 12.5 | 7.0 |
| SmolLM2 | 4.5047 | 0.0528 | 2.2 | 16.8 | -12.2 |
| Llama-3.2 | 4.5052 | 0.0302 | 20.7 | 17.3 | 8.2 |

The rows of the two earlier models are from the record loud-columns-2026-11-18, which found the kept columns worth little there and said so. They cost nothing there either. On this model they are worth 20.7 errors.

### The predictions

| | Prediction | Result | Holds |
|---|---|---|---|
| L1 | Wikitext text, no matrix on either side: the ball arm is under `Q4_K_M` by 2 errors | 4.9 errors behind | **no** |
| L2 | The same on the Python text | 8.3 errors behind | **no** |
| L3 | Wikitext text, the wikitext matrix on both sides: the ball arm is under `Q4_K_M` by 2 errors | 18.2 | yes |
| L4 | Wikitext text: the ball arm with the noise matrix is under `Q4_K_M` with the wikitext matrix by 2 errors | 11.1 | yes |
| L5 | The share is 13.8, so: on the Python text neither matrix puts the ball arm under the plain arm by 2 errors | 26.8 and 24.0 under | **no** |
| L6 | Wikitext text: the noise matrix buys `Q4_K_M` at least half of what the wikitext matrix buys | 68.5 percent | yes |
| X1 | The writers hold the largest of the three shares | 92.6 percent | yes |
| X2 | The three shares sum to between 0.75 and 1.25 | 0.954 | yes |
| Y1 | Wikitext text: `+K10` is under the plain ball arm by 2 errors and by at least 20 percent | 20.7 errors, 36.8 percent | yes |
| Y2 | Wikitext text: `+K10` is under `Q4_K_M` with no matrix by 2 errors | 17.3 | yes |
| Y3 | `+K10#0,1` gains at least half of what `+K10` gains | 1.005 of it | yes |
| Y4 | Python text: `+K10` is under `Q4_K_M` with no matrix by 2 errors | 15.3 | yes |
| Y5 | `+A1#0,1` is within 2 errors of `+K10#0,1` | 0.2 | yes |

What I got wrong.

- **L1 and L2.** I had called them fair, on two models that agreed. A third did not. The line "with no calibration data the ball arm beats `Q4_K_M`" was a statement about two models, and I had written it as a result.
- **L5.** I made a rule from two cases and one figure. The figure was about the attention's input because that was where the first model's gain was.
- **The record loud-columns-2026-11-18 said those columns hardly matter to rounding.** That was right for the two models it had. I wrote it more generally than two models allow.

## Deviations from the plan

- **Two extensions were added after the plan**, each before its own variants were scored. `summary.py` was extended for each, and for three tables that set this model beside the two earlier ones. Those tables are not part of any prediction.
- **`run_loud.py` and `picks.out` were added with extension 2.**
- Nothing else. Four starts, one for each text and each extension. No row was lost and no variant was scored twice.

## Limits

- **Three models, all between 1 and 2 billion weights, and each differs from the other two.** A fourth could differ again. The comparisons that hold on all three are with a matrix, and with the loud columns left whole. Neither is "any model".
- **The kept columns were tested as a remedy on one model.** On the two others they were scored earlier and found to change little. That they cost nothing there is two cases.
- **The ball arm with no matrix and no kept columns is the arm of the first headline line, and it loses here.** Whether it loses on other models of this family or size is not known.
- **The standard errors are the tool's** and treat positions as independent. Everything above rests on 4.9 errors and more.
- **KL on two texts. No task.** The record mmlu-subset-2026-11-18 has no row for this model.
- **A kept column is counted, not stored.** `shape/ballpack.py` has no place for one. No file was written for this model.
- **Why these models have such units.** Not asked. SmolLM2 and this model each have one hidden unit in layer 1 that holds nearly all of its matrix's input. Qwen2.5 has none.

## Read against the closure rule

Nothing is closed by a certificate. These are 19 measurements on one model.

**What changes in the repository's account.**

1. **The first headline line.** "With no calibration data, one ball per group at 4.5 bits beats `Q4_K_M` at 4.8 bits" holds on Qwen2.5 and SmolLM2 and fails on Llama-3.2. What holds on all three with no data and no forward pass is the same arm with ten columns of each `ffn_down` left whole, at 4.505 bits.
2. **The second and third lines hold on all three models and both texts**: with the wikitext matrix on both sides, and with a noise matrix against the wikitext matrix.
3. **The record loud-columns-2026-11-18** concluded that the units a clip must not touch are not the units rounding must protect. On this model they are the same two units. Its conclusion was about SmolLM2 and Qwen2.5.
4. **The records imatrix-gain-parts and second-text-python** pointed at a rule keyed on the stream's loud dimensions. It is refuted. A matrix halves the ball arm's KL on two models of three, for two different reasons, and one figure of the attention's input does not tell them from the third.

**What follows.**

1. The ball arm that should be carried forward keeps the loud columns. The rule reads only the weights, costs five thousandths of a bit, and is neutral, slightly good and very good on the three models.
2. The packed format needs a place for kept columns before this arm has a file.
3. Kept columns and a matrix together: on SmolLM2 they added nothing to each other. On this model it is not scored.
4. A fourth model should be chosen to differ: larger, or of a family none of these is.

## How to rerun

Needs NumPy, torch, llama.cpp's `gguf` package, `llama-imatrix`, `llama-quantize` and `llama-perplexity`, the model file and the two texts.

```bash
llama-perplexity -m Llama-3.2-1B-Instruct-f16.gguf -f wiki.test.raw --chunks 40 -c 512 -t 4 --kl-divergence-base wikitext.kld
llama-imatrix -m Llama-3.2-1B-Instruct-f16.gguf -f wiki.train.raw --chunks 100 -c 512 -t 4 -o llama32.imatrix.gguf
python3 experiments/imatrix-gain-parts-2026-11-18/randtext.py --model Llama-3.2-1B-Instruct-f16.gguf --pieces 80000 --seed 1 --out rand.txt
llama-imatrix -m Llama-3.2-1B-Instruct-f16.gguf -f rand.txt --chunks 100 -c 512 -t 4 -o llama32.rimatrix.gguf
llama-quantize --imatrix llama32.imatrix.gguf --tensor-type '^token_embd\.weight$=f16' Llama-3.2-1B-Instruct-f16.gguf llama32-Q4_K_M-imat.gguf Q4_K_M 4
```

```bash
python3 experiments/third-model-llama32-2026-11-18/run_arms.py --model Llama-3.2-1B-Instruct-f16.gguf --kld wikitext.kld --work DIR --out results-wikitext.jsonl --bin llama-perplexity --imatrix llama32.imatrix.gguf file:llama32-Q4_K_M.gguf ZBG128:b560s16 hZBG128:b560s16
```

```bash
python3 experiments/third-model-llama32-2026-11-18/run_rand.py --model Llama-3.2-1B-Instruct-f16.gguf --imatrix llama32.imatrix.gguf --imatrix-rand llama32.rimatrix.gguf --kld wikitext.kld --text wikitext --work DIR --out results-wikitext.jsonl --bin llama-perplexity 'hZBG128:b560s16~rand' 'ZBG128:b560s16+h@attn_output,ffn_down'
```

```bash
python3 experiments/third-model-llama32-2026-11-18/run_loud.py --model Llama-3.2-1B-Instruct-f16.gguf --imatrix llama32.imatrix.gguf --kld wikitext.kld --text wikitext --work DIR --out results-loud.jsonl --bin llama-perplexity 'ZBG128:b560s16+K10'
```

```bash
python3 experiments/third-model-llama32-2026-11-18/summary.py > results.out
```

A reference pass takes about three minutes on 4 CPU cores, a matrix about ten, a file two, a ball arm four.

## Disclosure

Written by an AI agent session under the owner's direction, on 2026-11-19. The plan, the run and this record are the session's. The programs are those of the earlier records of this branch. The model is Meta's, in a 16-bit file made by a third party.
