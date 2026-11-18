# Plan: what the gain of an importance matrix is made of

Written 2026-11-18, before any variant of this plan was scored. It follows the records imatrix-baseline-2026-11-17 and clip-by-layer-smollm2-2026-11-18, and the run loud-columns-2026-11-18, of which six rows were in when this was written. What they show is stated below.

## Question

An importance matrix halves the KL of the ball arm on SmolLM2 at 4.5 bits: 0.0560 without it, 0.0282 with it. It does the same for llama.cpp's `Q4_K_M`: 0.0838 and 0.0378. On Qwen2.5 it buys little: 0.0227 to 0.0209, and 0.0333 to 0.0282.

The loud-columns run looked for SmolLM2's gain in one place and did not find it there. Leaving whole the ten columns of every `ffn_down` that the clip records point at takes the ball arm from 0.0560 to 0.0528. That is a ninth of what the matrix gains. In layers 1 and 7 alone, the two the clip wrecks, it gives 0.0536.

Two questions about what the matrix buys on SmolLM2.

**A. Which matrices is the gain in?**

**B. Does it need text?** The matrix holds, for every block matrix, the mean square of each of its inputs over a calibration text. If what it finds is a property of the model, a few directions that are loud whatever is read, then a text with no meaning would find it as well. The arm would then need a model to run and nothing to read.

## What is run

**The models, the text and the score** are those of the model-output records: `SmolLM2-1.7B-Instruct` and `Qwen2.5-1.5B-Instruct`, 16-bit files, the wikitext-2 test text, 10,200 positions, mean KL against the reference's saved outputs.

**The two arms**, built by the code of the record model-output-qwen25-2026-11-17. `ZBG128:b560s16`: integer levels in one ball per group of 128 weights, 4.5 bits, no data. `hZBG128:b560s16`: the same with each column multiplied by (its input's mean square over the mean) to the power 0.25 before rounding, and divided after. The power was chosen on validation text in the imatrix-baseline record. It is not tuned here.

**Part A, 3 variants on SmolLM2.** The weighted arm on some kinds of matrix and the plain arm on the rest, with the matrix of the wikitext-2 training text.

| Variant | Weighted | Share of the block weights |
|---|---|---|
| `ZBG128:b560s16+h@attn_q,attn_k,attn_v` | the attention's readers: they read the stream | 18.75% |
| `ZBG128:b560s16+h@ffn_gate,ffn_up` | the feed-forward readers: they read the stream | 50% |
| `ZBG128:b560s16+h@attn_output,ffn_down` | the writers: they read the heads and the hidden units | 31.25% |

Not run on Qwen2.5: the whole gain there is 0.0018, which is 2.3 errors, and its parts could not be told apart.

**Part B, 4 variants, both models.** A matrix that read no text anyone wrote.

- **The text.** `randtext.py`: 80,000 pieces of the model's own vocabulary drawn at random, each ordinary piece with the same chance, seed 1. One text for each model. SHA-256 of the SmolLM2 text starts `d3cdb8cf`, of the Qwen2.5 text `37b8bd64`. The texts are not committed.
- **The matrix.** `llama-imatrix` with the command of the imatrix-baseline record and this text in place of the training text: 100 passages of 512 tokens.
- **The variants.** `hZBG128:b560s16~rand`: the weighted arm with that matrix. And llama.cpp's `Q4_K_M` made with that matrix by the `llama-quantize` command of the imatrix-baseline record, `Q4_K_M-rimat`.

**The program** is `run.py` of this folder. It calls the earlier record's code for both arms. A test on single matrices, before this plan: a matrix of a named kind comes out bit for bit as the weighted arm of that record, and a matrix of another kind as the plain arm.

## What the weights and the matrix say, before any score

`forecast-smollm2.out`, committed with this plan. Nothing is scored in it. For every block matrix and both arms it has the squared error over the squared weight, plain, and with each column counted by its input's mean square over the training text. The second is what the output of that matrix would lose if its inputs did not move together.

| Kind | Share of the input in the 10 most active columns | Error counted by activity, plain arm, dB | Weighted arm, dB | Gain, dB |
|---|---|---|---|---|
| `attn_q` | 61.8% | -24.66 | -28.65 | 3.99 |
| `attn_k` | 61.8% | -26.04 | -29.83 | 3.79 |
| `attn_v` | 61.8% | -20.91 | -25.42 | 4.51 |
| `attn_output` | 13.7% | -25.11 | -26.13 | 1.02 |
| `ffn_gate` | 17.0% | -25.07 | -25.84 | 0.77 |
| `ffn_up` | 17.0% | -24.43 | -25.31 | 0.88 |
| `ffn_down` | 15.5% | -25.65 | -27.06 | 1.41 |

- **What the attention reads is ten dimensions of the stream.** Of 2,048, ten hold 61.8 percent of the input's mean square, in the mean over the 24 layers. What the feed-forward block reads is not like that: 17.0 percent.
- **So the forecast puts the gain in the attention's readers**: about 4 dB there and about 1 dB everywhere else. They are 18.75 percent of the weights.
- **Counted by activity, `attn_v` is the worst-rounded kind under the plain arm**, 4 dB worse than its weight error of -24.93 dB says.
- In `ffn_down` the gain is in a few layers: 3.4 dB in layer 1, 7.0 in layers 7 and 23, 2.3 in layer 22, and 1.4 or less in the other 20.

**The forecast for Part B is not made yet**, because the matrices of the meaningless texts do not exist yet. When each is made, the same table with a third arm, weighted by that matrix and counted by the training text's activity, is written to `forecast-smollm2-rand.out` and `forecast-qwen25-rand.out` and committed before any Part B variant of that model is scored. The predictions below are not changed after.

## Predictions

The whole gain is the KL of the plain arm less the KL of the weighted arm with the training text's matrix: 0.0278 on SmolLM2. A part's share is its own gain over the plain arm, divided by the whole gain.

| | Prediction | It fails if |
|---|---|---|
| P1 | The attention's readers hold most of it: their share is over half | it is half or less |
| P2 | The feed-forward readers hold little: their share is under a quarter | it is a quarter or more |
| P3 | The writers hold little: their share is under a quarter | it is a quarter or more |
| P4 | The parts add: the three shares sum to between 0.75 and 1.25 | the sum is outside |
| R1 | On SmolLM2 the matrix of no meaning buys the ball arm at least half of what the training text's matrix buys | its share is under half |
| R2 | On Qwen2.5 it does no harm: `hZBG128:b560s16~rand` is not above the plain arm by 2 errors | it is |
| R3 | On SmolLM2 it buys `Q4_K_M` at least half of what the training text's matrix buys it | its share is under half |
| R4 | On Qwen2.5 it does `Q4_K_M` no harm: `Q4_K_M-rimat` is not above `Q4_K_M` with no matrix by 2 errors | it is |

My confidence: fair in P1 and P3. Low in P2 and P4: half of the weights are the feed-forward readers, and a small gain on each can add up. Fair in R1 and R3, on the reasoning that directions this loud should not need a meaning to show. Low in R2 and R4: what a meaningless text does to the many columns that are not loud is not known, and llama.cpp uses the matrix at full strength, which was worse than no matrix for the ball arm on Qwen2.5.

**Reported without a prediction.** Whether the ball arm with the matrix of no meaning, at 4.5 bits, is under llama.cpp's `Q4_K_M` at 4.8 bits made with the training text's matrix, on SmolLM2: 0.0378 is the figure to beat.

## What it would mean

- **P1 holds**: what the matrix buys on this model is mostly that the attention's inputs sit in a few dimensions, and a rule that rounds every column alike spends its error there.
- **R1 and R3 hold**: the matrix does not need to read anything. The "calibration data" of this model's gain is the model.
- **R1 fails**: the loud directions depend on what is read, or the gain is in the many quiet columns after all.
- **R2 or R4 fails**: a meaningless matrix is not a safe default, even where it helps.

## What this cannot show

- **A text with no meaning is still the model's own vocabulary**, and the model is still run on it: 51,200 tokens of forward passes. The arm needs no dataset. It is not an arm that reads only the weights.
- One seed, one length, one way of making a meaningless text.
- One strength for the ball arm, chosen with the real matrix.
- Two models. Why SmolLM2 has loud directions and Qwen2.5 has few is not asked here.
- Part A's shares are of a KL, and nothing says a KL must split into parts. P4 asks whether it does.

## Order and time

One process at a time, after the loud-columns run and its extension. For SmolLM2: the meaningless matrix, about 12 minutes. Its forecast, committed. Part A, three variants. Then `hZBG128:b560s16~rand` and `Q4_K_M-rimat`. For Qwen2.5: the matrix, about 16 minutes, its forecast, and the two variants. About seven minutes a variant: an hour and a half in all.

## Extension 1: the matrix of no meaning on the readers only

Written 2026-11-18 after the plan above was committed at `453d907` and after `forecast-smollm2-rand.out` was committed at `0cdc5f4`. No variant of this plan or of this extension was scored when it was written.

**What prompted it.** The forecast. Counted by the training text's activity, the arm weighted by the matrix of no meaning gains, on SmolLM2:

| Kind | Gain with the training text's matrix, dB | With the matrix of no meaning, dB |
|---|---|---|
| `attn_q` | 3.99 | 3.73 |
| `attn_k` | 3.79 | 3.53 |
| `attn_v` | 4.51 | 4.24 |
| `ffn_gate` | 0.77 | 0.56 |
| `ffn_up` | 0.88 | 0.68 |
| `attn_output` | 1.02 | -0.41 |
| `ffn_down` | 1.41 | 0.49 |

- **On the five kinds that read the stream the meaningless matrix does nearly what the real one does.** The ten columns it puts first hold 57.5 percent of the attention's input, where the real matrix's own ten hold 61.8.
- **On the two writers it does not.** For `attn_output` it is worse than no matrix. What the stream carries shows whatever is read. What the heads and the hidden units do depends on what is read.

So an arm that needs no text may do best to weight the readers and leave the writers alone.

**Two variants.** `ZBG128:b560s16+h~rand@attn_q,attn_k,attn_v,ffn_gate,ffn_up` on SmolLM2 and on Qwen2.5: the five reading kinds weighted by the matrix of no meaning, the two writers by the plain arm. `run.py` and `summary.py` were extended for it. The planned variants come out bit for bit as before on the single-matrix test.

| | Prediction | It fails if |
|---|---|---|
| E1 | On SmolLM2, leaving the writers plain loses nothing: the readers-only arm is not above `hZBG128:b560s16~rand` by 2 errors | it is |
| E2 | On Qwen2.5 the readers-only arm is not above the plain arm by 2 errors | it is |

My confidence: fair in both.

**What it would mean.** If E1 holds, the part of an importance matrix that a meaningless text can supply is the part about the stream, and that is the part worth having.

**Order.** After the seven variants of the plan.
