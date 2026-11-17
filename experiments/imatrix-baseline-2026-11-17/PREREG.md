# Plan: llama.cpp's files made with an importance matrix, beside the ball arms

Written 2026-11-17, before any importance matrix was computed and before any file made with one was scored. This is the part of issue #14 that the machine at hand can run. The scoring program is the one of the record model-output-qwen25-2026-11-17.

## Question

The runs of this date compare rounding rules that read no data. llama.cpp's `Q4_K_M` was made without an importance matrix, as in the gemma record. With one, it reads a calibration text and weights each weight's error by how much the model uses it.

Does one ball per group stay ahead of `Q4_K_M` when `Q4_K_M` is given an importance matrix?

## What is run

**The importance matrix.** llama.cpp's own tool at commit `eada89c`, on the wikitext-2 **training** text, `wiki.train.raw` of the same archive as the test text, SHA-256 `6707892fa3788b5ab9ed78ab5ff37d9fe825f6011a2ad4fcd6a6d467f0e7da57`. The first 100 passages of 512 tokens:

```bash
llama-imatrix -m MODEL-f16.gguf -f wiki.train.raw --chunks 100 -c 512 -t 4 -o MODEL.imatrix.gguf
```

**The files.** `Q4_K_M` and `Q4_0` made with it, with only the block matrices changed, as the files without it were:

```bash
llama-quantize --imatrix MODEL.imatrix.gguf --tensor-type '^token_embd\.weight$=f16' --tensor-type '^output\.weight$=f16' MODEL-f16.gguf OUT.gguf Q4_K_M 4
```

**The models.** `Qwen2.5-1.5B-Instruct` first. Then `SmolLM2-1.7B-Instruct`, after its own run.

**The score.** As in the two records: mean KL divergence and first-choice agreement against the reference model on the wikitext-2 **test** text, 10,200 positions.

## The calibration text favours the baseline

The matrix is computed on the training half of the text that the test half is drawn from. That is the best case for an importance matrix: same source, same kind of prose. A matrix from other text would help less on this test. So a lead that survives here is not owed to a poor matrix.

## What is known before the run

- **Qwen2.5, mean KL.** `Q4_K_M` 0.0333 at 4.812 bits. `Q4_0` 0.0623 at 4.5 bits. `EBG128:b560s16` 0.0195 at 4.5 bits. `EBG32:b136s8` 0.0208 at 4.5 bits. `EB32@32:16` 0.0339 at 4.498 bits. `EB128@128:16` 0.0331 at 4.124 bits. Standard errors 0.0004 to 0.0016.
- **SmolLM2.** Nothing scored.
- **What an importance matrix is worth is not known to me for these models.** My guess is that it cuts the mean KL of `Q4_K_M` by 10 to 40 percent.
- No importance matrix had been computed for either model, and no file made with one existed.

## Predictions

The error of a gap is the root of the sum of the squares of the two standard errors the tool prints.

| | Prediction | It fails if |
|---|---|---|
| I1 | On each model, `Q4_K_M` made with the matrix has a lower mean KL than `Q4_K_M` made without, by at least 2 errors | it is not, on either model |
| I2 | On each model, `Q4_0` made with the matrix has a lower mean KL than `Q4_0` made without, by at least 2 errors | it is not, on either model |
| I3 | On Qwen2.5, `EBG128:b560s16` at 4.5 bits has a lower mean KL than `Q4_K_M` with the matrix at 4.81 bits, by at least 2 errors | it is not |
| I4 | On Qwen2.5, the tie at 4.12 bits is lost: `Q4_K_M` with the matrix has a lower mean KL than `EB128@128:16`, by at least 2 errors | it is not |
| I5 | On Qwen2.5, `EB32@32:16`, which reads no data, has a lower mean KL than `Q4_0` with the matrix at the same 4.5 bits, by at least 2 errors | it is not |
| I6 | On SmolLM2, `EBG128:b560s16` has a lower mean KL than `Q4_K_M` with the matrix, by at least 2 errors | it is not |

I3 and I6 are the answer to the issue. I4 says what a matrix does to the claim of 0.69 bits saved.

## What it would mean

- If I3 and I6 hold, one ball per group at 4.5 bits is ahead of llama.cpp's usual file even when that file reads data and this code does not, on these two models and this text.
- If I3 fails, the lead of this date is a lead over rounding with no data only, and the records say so.
- Either way, the ball arms could be given the same matrix: a scale or a rounding that weights errors by use. No arm here does. That is a separate study.

## Not in this plan

Issue #14 asks for MMLU, HumanEval and GSM8K. They are not run under this plan.

- The machine has 4 CPU cores and no GPU. GSM8K and HumanEval need the model to write some 200 tokens per problem: about 7 hours per variant for GSM8K, and HumanEval runs code the model wrote.
- At these qualities such a test has little power. The arms differ in mean KL by 0.01 to 0.04 and change the first choice on 7 to 12 percent of positions. A difference of a point of accuracy needs several thousand questions to see.
- llama.cpp's scoring tool has a multiple-choice mode. If a file of MMLU questions in its format can be fetched, a subset will be scored under a plan of its own that states, before anything is run, how small a difference it could detect.

## What this run cannot show

- Anything about a matrix computed from other text, or from more of it.
- Anything about other calibrated methods (GPTQ, AWQ). None is run.
- Anything about accuracy on a task.
- The order of near ties.
