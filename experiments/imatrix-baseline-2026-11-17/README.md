# llama.cpp's files made with an importance matrix, beside the ball arms

## Question

The model-output records of this date compare rounding rules that read no data. llama.cpp's `Q4_K_M` was made without an importance matrix. Issue #14 asks whether one ball per group of weights stays ahead when `Q4_K_M` is given one.

1. How much does an importance matrix help llama.cpp's files on the two models?
2. Does the ball arm, which reads no data, stay ahead of `Q4_K_M` made with the matrix?
3. If the ball arm is given the same matrix, where does it stand?

## Status

`complete` for what the plan and its two extensions cover: eight runs on the test text and twelve on the validation text, on two models. Mean KL and first-choice agreement on one text. MMLU, HumanEval and GSM8K, which the issue also asks for, are not in this record. No arm has a packed format. Since this was written, an MMLU subset has been scored in the record mmlu-subset-2026-11-18, and the integer arm has a file in the record packed-ball-format-2026-11-18.

## Preregistration

[PREREG.md](PREREG.md), committed and pushed at `86a18b5` before any importance matrix was computed. Extension 1 at `79fb0cc`, with the plan's four files scored on both models and before any arm of the extension was scored. Extension 2 at `b148352`, before anything was scored on the validation text.

5 of the plan's 6 predictions hold. **None of extension 1's 4 holds.** The arm that does best was in that extension as the arm to compare against, with no prediction of its own that it would win. All 3 of extension 2's hold: it chose the strength on text that is not the test text, and the choice is that arm.

## Result in short

1. **The matrix helps llama.cpp's files on both models, and by very different amounts.** On Qwen2.5 it cuts the mean KL of `Q4_K_M` by 15 percent, from 0.0333 to 0.0282. On SmolLM2 by 55 percent, from 0.0838 to 0.0378.
2. **With no data, the ball arm stays ahead of `Q4_K_M` with the matrix on Qwen2.5 and falls behind it on SmolLM2.** Integer levels in one ball per group of 128, at 4.5 bits: 0.0227 against 0.0282 on Qwen2.5, 0.0560 against 0.0378 on SmolLM2.
3. **Given the same matrix at half strength, the ball arm is ahead on both.** `hZBG128:b560s16` at 4.51 bits scores 0.0209 on Qwen2.5 and 0.0282 on SmolLM2. `Q4_K_M` with the matrix at 4.81 and 4.82 bits scores 0.0282 and 0.0378. The gaps are 9.2 and 10.0 errors. **A sweep of five strengths on the validation text picks the same strength**, by a rule fixed before the sweep, so the figures on the test text were not what chose it.
4. **Given the matrix at full strength, the ball arm is worse than at half strength on both models**, and on Qwen2.5 worse than with no matrix at all. Full strength is the measure llama.cpp's quantisers minimise, and it was what I predicted would work.
5. **The tie at 4.12 bits does not survive calibration.** On Qwen2.5 `Q4_K_M` with the matrix is 5.2 errors ahead of `EB128@128:16`.
6. **An E8 arm cannot be weighted this way.** In a test build it broke on two of four matrices. Where an input is nearly unused, integer rounding sends its weights to zero and E8 cannot.

## What was run

**The importance matrix.** llama.cpp's `llama-imatrix` at commit `eada89c` on the first 100 passages of 512 tokens of the wikitext-2 training text, SHA-256 `6707892fa3788b5ab9ed78ab5ff37d9fe825f6011a2ad4fcd6a6d467f0e7da57`. It took 16 minutes for Qwen2.5 and 12 for SmolLM2 on 4 CPU cores.

**llama.cpp's files.** `Q4_K_M` and `Q4_0` made with `llama-quantize --imatrix`, with only the block matrices changed. With the matrix the tool put 3 block matrices of each `Q4_0` file in a 5-bit type, so those files are 4.516 bits.

**The ball arm given the matrix.** Integer levels in one ball per group of 128 weights, 560 bits for the point and 16 for the scale, as `ZBG128:b560s16` of the model-output records. Before rounding, every column of a matrix is multiplied by a weight, and after rounding divided by it. The weight of a column is the mean squared size of its input, read from the same matrix file, over its mean across columns, raised to a power and floored at 0.01.

| Arm | Power | What the rounding then minimises |
|---|---|---|
| `ZBG128:b560s16` | 0 | the squared error of the weights |
| `hZBG128:b560s16` | 1/4 | half way between the two |
| `iZBG128:b560s16` | 1/2 | the squared error of each weight times the mean squared size of its input |

The weights of the columns are stored at 16 bits each and counted: 0.005 or 0.006 bits a weight.

**The score.** As in the model-output records: mean KL divergence and first-choice agreement against the reference model on the wikitext-2 test text, 10,200 positions.

## Results

Every number is in `results.out`, printed by `summary.py` from `results.jsonl` and the results of the two model-output records.

### Both models

| | Bits | Qwen2.5: mean KL, first choice kept | SmolLM2 |
|---|---|---|---|
| `Q4_0` | 4.500 | 0.0623, 88.22% | 0.1271, 82.35% |
| `Q4_0` with the matrix | 4.516 | 0.0519, 88.54% | 0.0860, 85.80% |
| `Q4_K_M` | 4.81, 4.82 | 0.0333, 91.11% | 0.0838, 85.52% |
| `Q4_K_M` with the matrix | 4.81, 4.82 | 0.0282, 91.84% | 0.0378, 89.56% |
| `ZBG128:b560s16`, no data | 4.500 | 0.0227, 92.75% | 0.0560, 88.25% |
| `EBG128:b560s16`, no data | 4.500 | 0.0195, 93.08% | 0.0609, 87.50% |
| `iZBG128:b560s16`, the matrix at full strength | 4.506, 4.505 | 0.0294, 91.56% | 0.0467, 89.54% |
| **`hZBG128:b560s16`, the matrix at half strength** | 4.506, 4.505 | **0.0209, 92.64%** | **0.0282, 91.35%** |

Two rows of `results.jsonl` have the same mean KL to six places, 0.028205: `Q4_K_M` with the matrix on Qwen2.5 and `hZBG128:b560s16` on SmolLM2. They are different runs on different models, and every other field of the two rows differs. It is a coincidence.

### What the matrix is worth to each rule

| | Qwen2.5 | SmolLM2 |
|---|---|---|
| `Q4_K_M`: KL with the matrix against without | 15 percent lower, 5.5 errors | 55 percent lower, 26.6 errors |
| `Q4_0` | 17 percent lower, 5.5 errors | 32 percent lower, 13.8 errors |
| the ball arm at full strength | 29 percent **higher**, 7.2 errors | 17 percent lower, 6.6 errors |
| the ball arm at half strength | 8 percent lower, 2.3 errors | 50 percent lower, 22.7 errors |

On SmolLM2 the matrix is worth about as much to the ball arm at half strength as to `Q4_K_M`. On Qwen2.5 it is worth little to anything.

### The ball arm against `Q4_K_M` with the matrix

| | Qwen2.5: lower by, in errors | SmolLM2 |
|---|---|---|
| `ZBG128:b560s16`, no data | 6.1 | -14.0: it is behind |
| `EBG128:b560s16`, no data | 11.2 | -14.9: it is behind |
| `iZBG128:b560s16`, full strength | -1.3: a tie | -7.4: it is behind |
| `hZBG128:b560s16`, half strength | 9.2 | 10.0 |

The ball arms have 0.31 fewer bits than `Q4_K_M` in every row.

### The strength, chosen on the validation text

Extension 2. The wikitext-2 validation text, which no earlier run had read, scored as the test text is. The matrix is still the one from the training text.

| Strength | Qwen2.5: mean KL | SmolLM2: mean KL | Product of the two, times 10,000 |
|---|---|---|---|
| 0, no weighting | 0.0211 | 0.0501 | 10.592 |
| 0.15 | 0.0186 | 0.0263 | 4.903 |
| 0.25 | 0.0203 | 0.0239 | 4.834 |
| 0.35 | 0.0208 | 0.0255 | 5.303 |
| 0.50 | 0.0274 | 0.0404 | 11.085 |
| `Q4_K_M` with the matrix | 0.0265 | 0.0328 | |

The rule of the plan was one strength for both models, the one with the lowest product. That is 0.25, the half strength of extension 1. Its rows on the test text were scored before the sweep, and they are result 3.

What the table also says:

- **The curve is flat from 0.15 to 0.35 and steep outside it.** Any strength in that range is ahead of `Q4_K_M` with the matrix on both models, on this text too: by 7.7 and 9.9 errors at 0.25.
- **0.15 and 0.25 are not told apart.** Their products differ by 1.4 percent. Qwen2.5 is best at 0.15 and SmolLM2 at 0.25. The rule picked 0.25, and a rule that picked 0.15 would be as good on this evidence.
- **The three strengths that were scored on both texts come in the same order on both**, on each model.

### Why full strength loses

Not tested, so this is a reading and not a finding. At full strength a column whose input is a tenth as large as the mean gets a rounding error ten times as large in its weights. The matrix was measured on 100 passages. An input that is small there is not unused: on the test text it carries more than the matrix says, and its weights have been rounded coarsely. Half strength keeps most of the benefit and risks less. llama.cpp does not weight columns this way either. It uses the matrix when it picks the scale of a block, and still rounds each weight to its nearest level.

The plain weight error shows the trade. On Qwen2.5 it is -24.94 dB with no matrix, -24.22 at half strength and -22.33 at full.

### The predictions

**The plan.**

| | Prediction | Result | Holds |
|---|---|---|---|
| I1 | On each model `Q4_K_M` with the matrix is under `Q4_K_M` without, by 2 errors | by 5.5 and 26.6 errors | yes |
| I2 | The same for `Q4_0` | by 5.5 and 13.8 errors | yes |
| I3 | On Qwen2.5 `EBG128:b560s16` is under `Q4_K_M` with the matrix | by 11.2 errors | yes |
| I4 | On Qwen2.5 `Q4_K_M` with the matrix is under `EB128@128:16` | by 5.2 errors | yes |
| I5 | On Qwen2.5 `EB32@32:16` is under `Q4_0` with the matrix | by 13.7 errors | yes |
| I6 | On SmolLM2 `EBG128:b560s16` is under `Q4_K_M` with the matrix | it is 14.9 errors **above** | **no** |

**Extension 1.**

| | Prediction | Result | Holds |
|---|---|---|---|
| J1 | On each model `iZBG128:b560s16` is under `ZBG128:b560s16`, by 2 errors | 7.2 errors above on Qwen2.5, 6.6 under on SmolLM2 | **no** |
| J2 | On SmolLM2 `iZBG128:b560s16` is under `Q4_K_M` with the matrix | 7.4 errors above | **no** |
| J3 | On Qwen2.5 `iZBG128:b560s16` is under `Q4_K_M` with the matrix | 1.3 errors above | **no** |
| J4 | On each model full strength is no worse than half | worse by 10.3 and 16.4 errors | **no** |

All four of the extension's predictions were about full strength, and all four were wrong. What I expected of full strength is what half strength did.

**Extension 2.**

| | Prediction | Result | Holds |
|---|---|---|---|
| K1 | The strength chosen on the validation text is 0.15, 0.25 or 0.35 | 0.25 | yes |
| K2 | On each model the three strengths scored on the test text come in the same order on the validation text | the same order on both models | yes |
| K3 | At the chosen strength the ball arm is under `Q4_K_M` with the matrix on the test text, by 2 errors, on both models | by 9.2 and 10.0 errors | yes |

## Deviations from the plan

- **Extension 1** was added after the plan's files were scored, and **extension 2** after extension 1's. Each says what had been seen.
- **The program was changed three times after the first data**, each time before the runs that use the change: at `79fb0cc` to add the weighted arms, at `84b38d0` so that its check for rows already scored looks at the model, since this record keeps two models in one file, and at `b148352` to add a weighting of any strength.
- **The E8 arm of the extension was dropped before it was scored**, on the evidence of a test build of four matrices. The extension says so.
- **The runs were started several times** as launches on this machine are cut at two hours. `results.start` lists each start. One line there is a start that scored nothing: every file it was given was already in the results.
- `summary.py` was written with two rows scored and extended for each extension after its rows were in.

## Limits

- **The winning arm was not predicted to win.** Extension 1 tried two strengths on the test text and I had predicted the other. Extension 2 then chose the strength on the validation text by a rule fixed beforehand, and it chose the same one. So the test figures did not choose it. But the validation text is from the same source as the test text and the calibration text, and the sweep says only that the strength transfers within that source.
- **The calibration text is the training half of the test text's source.** That is the best case for any use of the matrix, llama.cpp's and this one. A matrix from other text would help both less. Nothing here says which would lose more.
- **Two models and one text.** The two models disagree on how much a matrix is worth, by a factor of nearly four. A third could disagree again.
- **Mean KL and first-choice agreement, not task accuracy.** The issue asks for MMLU, HumanEval and GSM8K. They are not here. An MMLU subset is in the record mmlu-subset-2026-11-18. The other two were not run.
- **Counted bits.** No ball arm has a packed file. The saving of 0.31 bits is in the file and not in memory.
- **This is the plainest use of the matrix.** Nothing uses second-order statistics of the inputs, as GPTQ does, or a search over the scale with the matrix, as llama.cpp does.
- **The standard errors are too small.** They treat positions as independent.

## Read against the closure rule

Nothing is closed by a certificate. These are measurements.

**The answer to issue #14, for the part that was run.**

- With no calibration on either side, one ball per group at 4.5 bits beats `Q4_K_M` at 4.8 bits on both models.
- When `Q4_K_M` reads data and the ball arm does not, the ball arm wins on Qwen2.5 and loses on SmolLM2.
- When both read the same data, and the ball arm uses it at half strength, the ball arm wins on both, by 9 and 10 errors, with 0.31 fewer bits.

The third line is the one that compares like with like. It rests on an arm that was planned and not predicted, at a strength that a sweep on other text then chose.

**What follows.**

1. A matrix from other text, on both sides, and a test text of another kind. Calibration, validation and test are all one source here.
2. Reading how llama.cpp uses the matrix in its search for a scale. The ball arm has no such search: its scale is the smallest that clips nothing.
3. The benchmarks the issue names, on a machine that can run them.

## How to rerun

Needs what the model-output records need, and the wikitext-2 training text.

```bash
llama-imatrix -m MODEL-f16.gguf -f wiki.train.raw --chunks 100 -c 512 -t 4 -o MODEL.imatrix.gguf
```

```bash
llama-quantize --imatrix MODEL.imatrix.gguf --tensor-type '^token_embd\.weight$=f16' --tensor-type '^output\.weight$=f16' MODEL-f16.gguf OUT.gguf Q4_K_M 4
```

```bash
python3 experiments/model-output-qwen25-2026-11-17/run.py --model MODEL-f16.gguf --kld base.kld --work DIR --out results.jsonl --bin llama-perplexity --imatrix MODEL.imatrix.gguf file:OUT.gguf hZBG128:b560s16 iZBG128:b560s16
```

For SmolLM2 the program is `experiments/model-output-smollm2-2026-11-17/run.py`, with the same arguments.

```bash
python3 experiments/imatrix-baseline-2026-11-17/summary.py > results.out
```

## Disclosure

Written by an AI agent session under the owner's direction, on 2026-11-17. The plan, the programs, the runs and this record are the session's.
