# Plan: a subset of MMLU, for the arms of issue #14

Written 2026-11-18, before any model or variant was scored on any question of MMLU. This is the part of issue #14 that the record imatrix-baseline-2026-11-17 left open: a task benchmark. It says, before anything is run, how small a difference the run could see.

## Question

On the wikitext-2 test text the ball arm given an importance matrix is closer to the 16-bit model than llama.cpp's `Q4_K_M` given the same matrix, on two models. That is a statement about mean KL on prose.

1. Does any arm lose accuracy on MMLU against the 16-bit model, by an amount this run can see?
2. Do the arms keep the 16-bit model's answers in the order their KL says, on questions that are not the text the matrix was computed from?

## What is run

**The questions.** `mmlu-test.bin` of `ikawrakow/validation-datasets-for-llama.cpp`, SHA-256 `88bbb0b99f0ac75dc63b82017d301be2e6b1b922b6a6eb508df41e5ac4396b5c`: 13,943 questions of the MMLU test set, four answers each, in the format of llama.cpp's scoring tool. The file is not committed.

**The subset.** 2,000 questions, drawn by the tool itself with its fixed seed (`--multiple-choice-tasks 2000`). The draw depends on the file and the count only, so every variant of both models answers the same questions in the same order. The first 1,000 of the draw are the draw of 1,000.

**How a question is scored.** As the tool does it, with nothing added: the model reads the question, then each answer as its continuation. The answer with the highest mean log-probability per token is the model's choice. There is no instruction, no worked example and no chat template. So the accuracies here are not the ones published for MMLU, which come from prompts with examples and lettered answers. Chance is 25 percent.

**The tool.** `llama-perplexity` of llama.cpp at commit `eada89c`, with the one-line patch `llama-perplexity-choice.patch` of this folder: beside its running accuracy it prints the answer it chose. Nothing that is computed changes. `run.py` works out the tool's draw from the file on its own, and stops if the answer chosen and the running accuracy ever disagree about a question.

```bash
llama-perplexity-mc -m MODEL.gguf -bf mmlu-test.bin --multiple-choice --multiple-choice-tasks 2000 -t 4
```

**The variants**, on `Qwen2.5-1.5B-Instruct` and then on `SmolLM2-1.7B-Instruct`, in this order on each:

| Variant | Bits | Mean KL on wikitext-2: Qwen2.5, SmolLM2 |
|---|---|---|
| the 16-bit reference | 16 | 0, 0 |
| llama.cpp `Q4_K_M` with the importance matrix | 4.81, 4.82 | 0.0282, 0.0378 |
| `hZBG128:b560s16`, the ball arm with the matrix at strength 0.25 | 4.51 | 0.0209, 0.0282 |
| llama.cpp `Q4_K_M` | 4.81, 4.82 | 0.0333, 0.0838 |
| `ZBG128:b560s16`, the ball arm with no data | 4.50 | 0.0227, 0.0560 |
| llama.cpp `Q4_0` | 4.50 | 0.0623, 0.1271 |

The files and the matrices are those of the records model-output-qwen25-2026-11-17, model-output-smollm2-2026-11-17 and imatrix-baseline-2026-11-17. The two `Q4_0` files were deleted to free disk. They are made again by the command of those records, and `summary.py` stops if a file's SHA-256 is not the one its KL row has. The ball arms are built by the program of the first record, and `summary.py` stops if their bits or weight error differ from their KL rows. The matrix is still the one from the wikitext-2 training text: MMLU is other text for it.

`Q4_0` is in the list as the arm that should show a difference if any does. It has 1.5 to 4.5 times the KL of the others.

## Two measures

**Accuracy**: the share of the questions answered rightly. This is what the issue asks for.

**Same answer as the reference**: the share of the questions where a variant chooses the answer the 16-bit model chooses, right or wrong. This is the measure with more resolution. Rounding cannot be expected to make a model know more: what it does is change answers. A changed answer moves accuracy only when one of the two answers is right, and it can move it either way.

Every comparison is by question. The error of a difference is the standard deviation of the per-question differences over the root of the count.

## How small a difference this can see

Not computed from any score: none exists. From the count alone.

- **Accuracy against the reference.** If a variant and the reference differ in being right on a share d of the questions, the error of the difference in accuracy is at most the root of d over 2,000. For d from 4 to 10 percent that is 0.45 to 0.71 points. A test at 2 errors finds a true loss 4 times in 5 when the loss is 2.8 errors: **1.3 to 2.0 points**. A loss under a point cannot be seen. The realised figure is printed for each variant.
- **Same answer as the reference, between two variants.** The same arithmetic. If on 5 to 15 percent of the questions one of the two keeps the reference's answer and the other does not, the error is 0.5 to 0.9 points, so the run can see a difference of **1.4 to 2.4 points** in how often two variants keep it.
- On 1,000 questions each figure is 1.41 times larger.

My expectation, which nothing here checks, is that at 4.5 to 4.8 bits a loss of accuracy is under a point. So the likely result for accuracy is that no difference is seen, and the record will then state the bound and nothing more.

## Predictions

Errors are paired, as above.

| | Prediction | It fails if |
|---|---|---|
| M1 | On each model the reference is above chance by at least 5 of its own errors | it is not, on either model |
| M2 | On each model none of the four variants other than `Q4_0` is under the reference in accuracy by more than 2.5 points | one is, on either model |
| M3 | On each model `hZBG128:b560s16` is not under `Q4_K_M` with the matrix in accuracy by 2 errors or more | it is, on either model |
| M4 | On each model `ZBG128:b560s16` is not under `Q4_K_M` in accuracy by 2 errors or more | it is, on either model |
| M5 | On each model `hZBG128:b560s16` gives the reference's answer more often than `Q4_0` does, by at least 2 errors | it does not, on either model |
| M6 | On each model `hZBG128:b560s16` does not give the reference's answer less often than `Q4_K_M` with the matrix by 2 errors or more | it does, on either model |
| M7 | On SmolLM2 `Q4_K_M` with the matrix gives the reference's answer more often than `Q4_K_M` without, by at least 2 errors | it does not |
| M8 | On SmolLM2 `hZBG128:b560s16` gives the reference's answer more often than `ZBG128:b560s16`, by at least 2 errors | it does not |

M3 and M6 are the answer to the issue, at the resolution this run has. M5 says whether the second measure can see anything. M7 and M8 ask whether what the matrix bought on wikitext-2 carries to other text: on SmolLM2 it halved the KL of both.

My confidence: high in M1 to M4, fair in M5 and M6, and low in M7 and M8. I do not know how much of the matrix's gain is tied to prose of its own source.

## What it would mean

- If M3 and M6 hold, nothing on this task speaks against the ball arm, within the stated resolution. That is weaker than its lead in KL and the record will say so.
- If M3 or M6 fails, the lead in KL on wikitext-2 does not carry to answers on MMLU, and the third line of the importance-matrix record's answer is limited to that text.
- If M5 fails, this test cannot tell a variant from one with a third of its KL, and nothing should be read from its other rows.
- If M7 or M8 fails while M5 holds, the matrix's gain is at least partly a gain on its own kind of text.

## Order and time

One scoring process at a time. Qwen2.5 first, in the order of the table: if the run is cut short, the rows that answer the issue exist first.

**A rule for the count, fixed now.** The reference of Qwen2.5 is scored first on 2,000 questions and timed. If that takes more than 75 minutes, every other variant is scored on 1,000, the first 1,000 of the same draw, and the reference's row is read on its first 1,000. The figures above for 1,000 then apply.

Rows go to `results.jsonl`, one for each variant, with the answer chosen on every question. A watcher commits each row as it lands.

## A test of the program that reads no test question

Before the first run, `run.py` is tried on 100 questions of `mmlu-validation.bin` (SHA-256 `470af3a74eccacfaf6f43b08aabf510f61e6c92fe20d17241ded934151e225fa`), a different split of MMLU. Its output is not committed and nothing is read from it but whether the program runs, whether its draw is the tool's, and how long a question takes.

## Not in this plan

- **HumanEval and GSM8K.** The importance-matrix plan says why: on 4 CPU cores a variant needs about 7 hours of generation for GSM8K, and HumanEval runs code a model wrote. Neither is run.
- **MMLU as published.** No examples, no lettered answers, no chat template, and a subset.
- Any other arm. The E8 arms are left out: on SmolLM2 they are behind the integer arm, and the integer arm is the one both models agree on.

## What this run cannot show

- A loss of accuracy under about a point.
- Anything about the order of two variants whose difference is under 2 errors.
- Anything about generation, long context, or a chat prompt.
- Anything beyond two models of 1.5 and 1.7 billion weights.
