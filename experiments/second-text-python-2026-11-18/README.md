# The headline comparisons on a second text

## Question

Every KL of the model-output records, the importance-matrix records and the packed-file record is on Wikipedia prose, and the importance matrix they use was made from the same source. Do the three headline comparisons hold on a text of another kind?

1. With no matrix on either side, the ball arm at 4.5 bits is under llama.cpp's `Q4_K_M` at 4.8 bits.
2. With the wikitext matrix on both sides, the same.
3. With a matrix made from a text with no meaning, the ball arm is under `Q4_K_M` made with the wikitext matrix.

And on a text that neither matrix has read, is a matrix from noise as good as a matrix from Wikipedia?

## Status

`complete`, 2026-11-18. 12 variants on two models. All 6 predictions hold. One thing the plan did not ask about is the more useful finding, and it is marked.

## Preregistration

[PREREG.md](PREREG.md), committed and pushed at `66acffb` with `summary.py`, before the 16-bit models were run on the new text. `reference.out` was committed at `bb02a8d` for SmolLM2 and at `a3c40e4` for Qwen2.5, each before any variant of that model was scored. No program that builds or scores a variant was written or changed for this record.

## Result in short

1. **All three headline comparisons hold on Python source, on both models.** The ball arm is under `Q4_K_M` by 13.2 and 9.3 errors with no matrix on either side, by 11.2 and 5.0 with the wikitext matrix on both, and by 8.7 and 3.6 with the noise matrix on its side against the wikitext matrix on the other. SmolLM2 first, then Qwen2.5.
2. **On SmolLM2, on a text neither matrix has read, noise is as good as Wikipedia.** For `Q4_K_M` the noise matrix buys 99.3 percent of what the wikitext matrix buys. For the ball arm, 95.1 percent, and the two arms are 1.6 errors apart. On the wikitext text itself they were 3.5 apart.
3. **Not predicted: on Qwen2.5 the ball arm is best with no matrix at all on this text.** 0.0140 with none, 0.0154 with the wikitext matrix, 0.0162 with the noise matrix. The wikitext matrix is 2.2 errors worse than none and the noise matrix 3.0 worse. On prose the wikitext matrix had helped this arm by 2.3 errors. So its small gain on Qwen2.5 was a gain on the kind of text it was made from.
4. **`Q4_K_M` is still helped by a matrix on Qwen2.5**, by less: 2.8 errors for the wikitext matrix, 1.2 for noise.
5. Every KL is lower on Python than on prose, by a factor of 0.56 to 0.74. Both 16-bit models find the Python text easier: a perplexity of 4.1, against 9.4 and 9.2 on the wikitext text.

So the lines of this branch that say "the ball arm is ahead" are not a property of Wikipedia prose. The line that needs care is the one about the matrix: on a model whose stream has loud dimensions it carries over to another kind of text whatever it was made from, and on a model that has few it does not carry over, for this arm.

## What was run

**The text.** Python source: `conversion/base.py` of llama.cpp at commit `eada89c`. 183,074 bytes, SHA-256 starts `05023c54`. The first 40 passages of 512 tokens, the second half of each scored: 10,200 positions.

**The reference.** Each 16-bit model's saved outputs on that text. `reference.out` has what the tool printed.

**The variants.** For each model, llama.cpp's `Q4_K_M` and the ball arm `ZBG128:b560s16`, each with no matrix, with the matrix of the wikitext-2 training text, and with the matrix of no meaning of the record imatrix-gain-parts-2026-11-18. The weighted ball arm is `hZBG128:b560s16`, at the strength 0.25 that the record imatrix-baseline-2026-11-17 chose on wikitext validation text.

**Nothing is made new.** The six `Q4_K_M` files are the files the earlier records scored: each row carries the file's SHA-256, and the six match the earlier rows. The two matrices of each model are the ones those records used. The programs are those records' programs, unchanged.

## Results

Every number is in `results.out`, printed by `summary.py` from `results.jsonl` and the rows of four earlier records, or in `reference.out`.

| Model | Arm | Matrix | Bits a weight | Mean KL on Python | First choice kept | Mean KL on wikitext-2, earlier records |
|---|---|---|---|---|---|---|
| SmolLM2 | llama.cpp `Q4_K_M` | none | 4.8223 | 0.0522 | 91.2% | 0.0838 |
| SmolLM2 | llama.cpp `Q4_K_M` | wikitext-2 training text | 4.8223 | 0.0234 | 94.0% | 0.0378 |
| SmolLM2 | llama.cpp `Q4_K_M` | no meaning | 4.8223 | 0.0236 | 93.8% | 0.0399 |
| SmolLM2 | the ball arm | none | 4.4998 | 0.0364 | 92.5% | 0.0560 |
| SmolLM2 | the ball arm | wikitext-2 training text | 4.5047 | 0.0167 | 94.9% | 0.0282 |
| SmolLM2 | the ball arm | no meaning | 4.5047 | 0.0176 | 94.8% | 0.0314 |
| Qwen2.5 | llama.cpp `Q4_K_M` | none | 4.8120 | 0.0223 | 94.5% | 0.0333 |
| Qwen2.5 | llama.cpp `Q4_K_M` | wikitext-2 training text | 4.8120 | 0.0194 | 94.8% | 0.0282 |
| Qwen2.5 | llama.cpp `Q4_K_M` | no meaning | 4.8120 | 0.0208 | 94.8% | 0.0294 |
| Qwen2.5 | the ball arm | none | 4.4998 | 0.0140 | 95.4% | 0.0227 |
| Qwen2.5 | the ball arm | wikitext-2 training text | 4.5060 | 0.0154 | 95.5% | 0.0209 |
| Qwen2.5 | the ball arm | no meaning | 4.5060 | 0.0162 | 95.4% | 0.0228 |

### The three comparisons

| | SmolLM2 | Qwen2.5 |
|---|---|---|
| No matrix on either side: the ball arm against `Q4_K_M` | 0.0364 against 0.0522, 13.2 errors | 0.0140 against 0.0223, 9.3 errors |
| The wikitext matrix on both sides | 0.0167 against 0.0234, 11.2 errors | 0.0154 against 0.0194, 5.0 errors |
| The noise matrix for the ball arm, the wikitext matrix for `Q4_K_M` | 0.0176 against 0.0234, 8.7 errors | 0.0162 against 0.0194, 3.6 errors |

On Qwen2.5 the best ball arm on this text has no matrix. Against `Q4_K_M` with the wikitext matrix it stands at 0.0140 against 0.0194.

### What a matrix is worth on a text it has not read

| Model | Arm | The wikitext matrix against none, in errors | The noise matrix against none, in errors | Noise buys, of what wikitext buys |
|---|---|---|---|---|
| SmolLM2 | `Q4_K_M` | 26.8 under | 26.6 under | 99.3% |
| SmolLM2 | the ball arm | 24.9 under | 22.3 under | 95.1% |
| Qwen2.5 | `Q4_K_M` | 2.8 under | 1.2 under | 52.6% |
| Qwen2.5 | the ball arm | 2.2 **above** | 3.0 **above** | both lose |

`results.out` prints 157.2 percent in the last cell. It is the ratio of two losses and means nothing: both matrices make that arm worse.

**SmolLM2.** What the matrix finds there is ten loud dimensions of the stream, and the record imatrix-gain-parts-2026-11-18 showed that a meaningless text finds them. Here the same is seen from the other side: on a text of another kind the matrix from Wikipedia is worth as much as it was on Wikipedia, and the matrix from noise is worth the same within 1.6 errors.

**Qwen2.5.** Its stream has few loud dimensions. The wikitext matrix gave the ball arm 2.3 errors on prose. On Python it costs it 2.2. So that gain was the matrix knowing the text, and on another text the weighting spends precision in the wrong columns. llama.cpp's `Q4_K_M` keeps a gain from the same matrix, 2.8 errors where it had 5.5 on prose. The two use the matrix differently, and why one keeps a gain and the other does not is not asked here.

### The predictions

| | Prediction | Result | Holds |
|---|---|---|---|
| C1 | On both models, with no matrix on either side, the ball arm is under `Q4_K_M` by 2 errors | 13.2 and 9.3 | yes |
| C2 | On both models, with the wikitext matrix on both sides, the same | 11.2 and 5.0 | yes |
| C3 | On both models, the ball arm with the matrix of no meaning is under `Q4_K_M` with the wikitext matrix by 2 errors | 8.7 and 3.6 | yes |
| C4 | SmolLM2: the matrix of no meaning buys the ball arm at least half of what the wikitext matrix buys | 95.1 percent | yes |
| C5 | SmolLM2: the ball arm with the wikitext matrix is not under the ball arm with the matrix of no meaning by 2 errors | 1.6 | yes |
| C6 | SmolLM2: the wikitext matrix still helps `Q4_K_M`, under no matrix by 2 errors | 26.8 | yes |

What I did not think to predict. The plan put Qwen2.5's matrices under "reported without a prediction" and asked whether either would help. I expected a tie. For the ball arm both hurt.

## Deviations from the plan

- **`summary.py` prints one comparison more than planned**: the ball arm with the matrix of no meaning against the ball arm with no matrix. It was added when the rows were in, to state the finding about Qwen2.5 from a printed figure.
- Nothing else. Two starts, one for each model, twelve rows in the planned order.

## Limits

- **One more text, of one kind, from one project.** Dialogue, another language or other code could differ. Two texts are not "any text".
- **Two models**, and they differ on the question of the matrix. Which of the two a third model is like is not known.
- **Both models may have read this project's source in training.** That bears on the reference's perplexity, not on how far a rounded model is from the reference.
- **The standard errors are the tool's** and treat positions as independent. The three headline comparisons rest on 3.6 errors and more. The losses on Qwen2.5, 2.2 and 3.0 errors, are at the size this repository treats with care: they say the matrices do not help that arm on this text, and that they lean to harm.
- **The strength of the weighting, 0.25, was chosen on wikitext validation text** by the product of both models' KL. On Qwen2.5 a weaker weighting might not lose on Python. Not tested.
- **The counted arm with groups of 128 only.** The file with blocks of 32 was not scored on this text.
- **KL, not a task.**

## Read against the closure rule

Nothing is closed by a certificate. These are 12 measurements on one new text.

**What changes in the repository's account.**

1. "One text" comes off the limits of the three headline comparisons. They hold on prose and on Python source, on two models.
2. The imatrix-baseline record chose the strength of the weighting on validation text of the same source as the test, and reported a gain on both models. On Qwen2.5 that gain does not carry to another kind of text: there it is a loss. The record's choice was sound for the text it was made on, and its gain on Qwen2.5 should be read as that.
3. The imatrix-gain-parts record asked whether a matrix from noise is the safer one when the text to come is not known. On SmolLM2 it is as good as a matrix from another kind of text. On Qwen2.5 neither is safe for the ball arm, and no matrix is.
4. The same record's file on Qwen2.5 was better with no matrix on prose as well. The two findings agree.

**What follows.**

1. A rule for when to weight at all. Both models now say the same thing: weight where the stream has loud dimensions, and not where it has few. The figure that separates them is in `forecast-*.out` of the imatrix-gain-parts record: the ten loudest dimensions hold 61.8 percent of what the attention reads on SmolLM2 and 11.4 on Qwen2.5. Where the line is between those, two models cannot say.
2. A third model would test that rule and not only add a case.
3. The pull request's second headline line should name this: with the matrix on both sides the ball arm is ahead, and on Qwen2.5 on Python it would be further ahead without it.

## How to rerun

Needs NumPy, torch, llama.cpp's `gguf` package and `llama-perplexity`, the model file, the three `Q4_K_M` files and the two matrices of the earlier records.

```bash
git -C llama.cpp show eada89c:conversion/base.py > base.py.txt
llama-perplexity -m SmolLM2-1.7B-Instruct-f16.gguf -f base.py.txt --chunks 40 -c 512 -t 4 --kl-divergence-base code.kld
```

```bash
python3 experiments/model-output-smollm2-2026-11-17/run.py --model SmolLM2-1.7B-Instruct-f16.gguf --kld code.kld --work DIR --out results.jsonl --bin llama-perplexity --imatrix smollm2.imatrix.gguf file:smollm2-Q4_K_M.gguf file:smollm2-Q4_K_M-imat.gguf file:smollm2-Q4_K_M-rimat.gguf ZBG128:b560s16 hZBG128:b560s16
```

```bash
python3 experiments/imatrix-gain-parts-2026-11-18/run.py --model SmolLM2-1.7B-Instruct-f16.gguf --imatrix smollm2.imatrix.gguf --imatrix-rand smollm2.rimatrix.gguf --kld code.kld --text code --work DIR --out results.jsonl --bin llama-perplexity 'hZBG128:b560s16~rand'
```

```bash
python3 experiments/second-text-python-2026-11-18/summary.py > results.out
```

The reference pass takes about five minutes on 4 CPU cores, a file two to three, a ball arm about seven.

## Disclosure

Written by an AI agent session under the owner's direction, on 2026-11-18. The plan, the run and this record are the session's. The programs are those of the earlier records of this branch. The text is a file of the llama.cpp project.
