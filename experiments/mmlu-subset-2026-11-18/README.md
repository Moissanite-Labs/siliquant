# A subset of MMLU, for the arms of issue #14

## Question

On the wikitext-2 test text the ball arm given an importance matrix is closer to the 16-bit model than llama.cpp's `Q4_K_M` given the same matrix, on two models. That is a statement about mean KL on prose. Issue #14 asks for a task benchmark.

1. Does any arm lose accuracy on MMLU against the 16-bit model, by an amount this run can see?
2. Do the arms keep the 16-bit model's answers in the order their KL says, on questions that are not the text the matrix was computed from?

## Status

`complete`, 2026-11-18. Six variants on each of two models, 2,000 questions each. All 8 predictions hold. HumanEval and GSM8K are not run: the plan of the record imatrix-baseline-2026-11-17 says why.

## Preregistration

[PREREG.md](PREREG.md), committed and pushed at `e81eafa` with `run.py`, `summary.py` and the patch to the tool, before any model was scored on any question of MMLU. The plan states, from the count of questions alone, how small a difference the run could see. No program was changed after.

## Result in short

1. **Accuracy does not separate the ball arms from the 16-bit model, on either model.** Their four differences from it are -0.70, -0.50, -0.10 and +0.15 points, none beyond 1.5 errors, and the run can see a loss of 1.1 to 1.3 points for them. So what is established is a bound: none of them loses as much as a point and a half.
2. **How often a variant keeps the 16-bit model's answer follows its KL, in order, through all five variants on both models.** From 90.4 to 92.8 percent on Qwen2.5 and from 87.5 to 93.6 on SmolLM2.
3. **By that measure the ball arm with the matrix is ahead of `Q4_K_M` with the matrix on both models, at 4.5 bits against 4.8**: by 0.5 points on Qwen2.5, which is 0.7 errors and a tie, and by 1.6 points on SmolLM2, 2.3 errors.
4. **What the matrix bought on wikitext-2 carries to MMLU.** On SmolLM2, where it halved the KL, it adds 3.5 points of kept answers to `Q4_K_M` (5.0 errors) and 2.5 to the ball arm (3.7 errors). The matrix was computed from Wikipedia prose and MMLU is other text.
5. **The one loss of accuracy that passes 2 errors is `Q4_K_M` without the matrix on SmolLM2**: 1.45 points, 2.5 errors. It is one of ten comparisons and sits at the edge of what the run can see, so it is a hint and not a finding.

So nothing on this task speaks against the ball arm, and the second measure speaks for it, weakly on one model and clearly enough on the other. That is weaker than its lead in KL, as the plan said it would be.

## What was run

**The questions.** `mmlu-test.bin` of `ikawrakow/validation-datasets-for-llama.cpp`, SHA-256 `88bbb0b99f0ac75dc63b82017d301be2e6b1b922b6a6eb508df41e5ac4396b5c`: 13,943 questions of the MMLU test set, four answers each. 2,000 are drawn by llama.cpp's tool with its fixed seed, so every variant of both models answers the same questions in the same order.

**How a question is scored.** As the tool does it: the model reads the question, then each answer as its continuation, and the answer with the highest mean log-probability per token is its choice. No instruction, no worked example, no chat template. So the accuracies here are not the ones published for MMLU. Chance is 25 percent.

**The tool.** `llama-perplexity` of llama.cpp at `eada89c` with the one-line patch of this folder, which prints the answer chosen beside the running accuracy. `run.py` works out the tool's draw from the file on its own and stops if the answer chosen and the running accuracy ever disagree about a question. They never did, over 24,000 answers.

**The variants**, on `Qwen2.5-1.5B-Instruct` and `SmolLM2-1.7B-Instruct`: the 16-bit reference, llama.cpp's `Q4_0`, `Q4_K_M` and `Q4_K_M` with the importance matrix, and the ball arms `ZBG128:b560s16` (no data) and `hZBG128:b560s16` (the matrix at strength 0.25). They are the variants of the KL records: `summary.py` checks the SHA-256 of each file and the bits and weight error of each built arm against its KL row.

**Two measures.** Accuracy, and the share of questions where a variant chooses the answer the 16-bit model chooses, right or wrong. Every comparison is by question: the error of a difference is the standard deviation of the per-question differences over the root of 2,000.

## Results

Every number is in `results.out`, printed by `summary.py` from `results.jsonl`.

### Qwen2.5-1.5B-Instruct

| Variant | Bits | Mean KL on wikitext-2 | Right of 2,000 | Accuracy | Less the reference's, points | In errors | Smallest loss the run could see | Same answer as the reference |
|---|---|---|---|---|---|---|---|---|
| 16-bit reference | 16 | 0 | 714 | 35.70% | | | | |
| `Q4_0` | 4.50 | 0.0623 | 698 | 34.90% | -0.80 | -1.4 | 1.6 | 90.40% |
| `Q4_K_M` | 4.81 | 0.0333 | 710 | 35.50% | -0.20 | -0.4 | 1.4 | 91.95% |
| `Q4_K_M` with the matrix | 4.81 | 0.0282 | 702 | 35.10% | -0.60 | -1.3 | 1.3 | 92.30% |
| `ZBG128:b560s16` | 4.50 | 0.0227 | 704 | 35.20% | -0.50 | -1.1 | 1.3 | 92.70% |
| `hZBG128:b560s16` | 4.51 | 0.0209 | 700 | 35.00% | -0.70 | -1.5 | 1.3 | 92.80% |

### SmolLM2-1.7B-Instruct

| Variant | Bits | Mean KL on wikitext-2 | Right of 2,000 | Accuracy | Less the reference's, points | In errors | Smallest loss the run could see | Same answer as the reference |
|---|---|---|---|---|---|---|---|---|
| 16-bit reference | 16 | 0 | 712 | 35.60% | | | | |
| `Q4_0` | 4.50 | 0.1271 | 689 | 34.45% | -1.15 | -1.9 | 1.7 | 87.50% |
| `Q4_K_M` | 4.82 | 0.0838 | 683 | 34.15% | -1.45 | -2.5 | 1.6 | 88.50% |
| `Q4_K_M` with the matrix | 4.82 | 0.0378 | 695 | 34.75% | -0.85 | -1.8 | 1.3 | 92.00% |
| `ZBG128:b560s16` | 4.50 | 0.0560 | 715 | 35.75% | +0.15 | +0.3 | 1.3 | 91.05% |
| `hZBG128:b560s16` | 4.51 | 0.0282 | 710 | 35.50% | -0.10 | -0.2 | 1.1 | 93.55% |

On each model, read the KL column against the last column. Sorted by KL, the five variants are sorted by kept answers, with no exception on either model.

### The comparisons

First less second. Accuracy in points, and the share of the reference's answers kept in points, each with its paired error count.

| Comparison | Qwen2.5: accuracy | Kept answers | SmolLM2: accuracy | Kept answers |
|---|---|---|---|---|
| `hZBG128:b560s16` against `Q4_K_M` with the matrix | -0.10, -0.2 errors | +0.50, +0.7 errors | +0.75, +1.5 errors | +1.55, +2.3 errors |
| `ZBG128:b560s16` against `Q4_K_M` | -0.30, -0.5 | +0.75, +1.1 | +1.60, +2.5 | +2.55, +3.1 |
| `hZBG128:b560s16` against `Q4_0`, the same bits | +0.10, +0.2 | +2.40, +3.4 | +1.05, +1.8 | +6.05, +7.9 |
| `ZBG128:b560s16` against `Q4_0`, the same bits | +0.30, +0.5 | +2.30, +3.1 | +1.30, +2.1 | +3.55, +4.3 |
| `Q4_K_M` with the matrix against without | -0.40, -0.8 | +0.35, +0.6 | +0.60, +1.1 | +3.50, +5.0 |
| `hZBG128:b560s16` against `ZBG128:b560s16` | -0.20, -0.4 | +0.10, +0.2 | -0.25, -0.5 | +2.50, +3.7 |

The last two rows are what the matrix is worth. On Qwen2.5, where it cut the KL by 8 to 15 percent, it moves nothing that can be seen. On SmolLM2, where it cut the KL by half, it keeps 2.5 to 3.5 more answers in a hundred.

### The predictions

| | Prediction | Qwen2.5 | SmolLM2 | Holds |
|---|---|---|---|---|
| M1 | The reference is above chance by at least 5 of its own errors | 10.0 errors | 9.9 errors | yes |
| M2 | None of the four variants other than `Q4_0` is under the reference by more than 2.5 points | the most is 0.70 | the most is 1.45 | yes |
| M3 | `hZBG128:b560s16` is not under `Q4_K_M` with the matrix in accuracy by 2 errors | -0.2 | +1.5 | yes |
| M4 | `ZBG128:b560s16` is not under `Q4_K_M` in accuracy by 2 errors | -0.5 | +2.5 | yes |
| M5 | `hZBG128:b560s16` keeps the reference's answer more often than `Q4_0`, by 2 errors | +3.4 | +7.9 | yes |
| M6 | `hZBG128:b560s16` does not keep it less often than `Q4_K_M` with the matrix by 2 errors | +0.7 | +2.3 | yes |
| M7 | On SmolLM2 `Q4_K_M` with the matrix keeps it more often than without, by 2 errors | | +5.0 | yes |
| M8 | On SmolLM2 `hZBG128:b560s16` keeps it more often than `ZBG128:b560s16`, by 2 errors | | +3.7 | yes |

I gave M7 and M8 low confidence: I did not know how much of the matrix's gain was tied to prose of its own source. On this task, not much of it.

M2 to M4 and M6 predict that something will not be seen, and a run with little power makes such predictions easy to keep. M5, M7 and M8 are the ones that could have failed for lack of power, and they are the reason to believe the second measure sees anything.

## Deviations from the plan

- **`results.start` has four start notes for Qwen2.5.** The last two were written when the queue was launched again after the machine had stopped during other records' runs. By then every variant of this record on Qwen2.5 was scored, and nothing was run. No variant of this record was interrupted.
- **The rule for the count.** The reference of Qwen2.5 took 39 minutes on 2,000 questions, under the 75 of the rule, so every variant has 2,000. The reference of SmolLM2 took 54 minutes. The rule was written on the first model only and was applied as written.
- **The order.** The six variants of SmolLM2 were scored after two runs of issue #20, six hours after those of Qwen2.5. The plan fixed the order within this record, and that was kept.
- **The trial of the program** on 100 questions of the validation split passed on a file of llama.cpp's and on a built arm. Its output is not committed, as planned.
- Nothing else. No analysis here is unplanned.

## Limits

- **Accuracy has little power here.** 2,000 questions resolve a loss of 1.1 to 1.7 points. My expectation before the run was that a loss at 4.5 to 4.8 bits is smaller than that, and nothing here contradicts it. Most of this record's statements about accuracy are that nothing was seen.
- **Ten comparisons with the reference.** One passes 2 errors, at 2.5. With ten tries that can happen by chance, so it is not counted as a finding.
- **"Same answer as the reference" is fidelity, not quality.** It says how many answers a variant changes. It cannot say whether a changed answer is better or worse, and rounding is not expected to make a model better.
- **Not MMLU as published.** Bare questions and bare answers scored as continuations, no examples, no lettered choices, no chat template, and a subset. The reference models score 35.7 and 35.6 percent, and those figures are not to be set beside published ones.
- **Two models of 1.5 and 1.7 billion weights, one task.** Nothing here is about generation, long context, code, or arithmetic. HumanEval and GSM8K were not run.
- **The arms are the counted ones**, with blocks of 128. The arm that has a file, of the record packed-ball-format-2026-11-18, was not scored here. Its KL is 5 to 10 percent above theirs.

## Read against the closure rule

Nothing is closed by a certificate. These are twelve measurements of 2,000 answers.

**What changes in the repository's account.** The importance-matrix record ended: "what these do not show: accuracy on a task". For one task there is now a bound and a second measure. The bound: on MMLU scored this way the ball arms lose under a point and a half, on two models. The second measure: the order the KL gives on Wikipedia prose is the order of kept answers on MMLU, for every variant tried, on both models.

**What follows.**

1. A task with more resolution per question would say more than more questions of this one. A generation task needs a machine with a GPU, which this seat does not have.
2. If the kept-answers measure is used again, it could be planned as the main one: it had power where accuracy had none.

## How to rerun

Needs NumPy, torch, llama.cpp's `gguf` package, a build of `llama-perplexity` with `llama-perplexity-choice.patch` applied, the questions file, the model files and their importance matrices.

```bash
python3 experiments/mmlu-subset-2026-11-18/run.py --model qwen2.5-1.5b-instruct-fp16.gguf --data mmlu-test.bin --tasks 2000 --work DIR --out results.jsonl --bin llama-perplexity-mc --imatrix qwen15.imatrix.gguf reference file:qwen15-Q4_K_M-imat.gguf hZBG128:b560s16
```

```bash
python3 experiments/mmlu-subset-2026-11-18/summary.py > results.out
```

On 4 CPU cores a variant scored from a 16-bit copy takes 38 to 59 minutes and one of llama.cpp's files 16 to 26.

## Disclosure

Written by an AI agent session under the owner's direction, on 2026-11-18. The plan, the programs, the run and this record are the session's. The questions file is its maker's, and the scoring is llama.cpp's.
