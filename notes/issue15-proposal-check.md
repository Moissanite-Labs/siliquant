# Check of the proposal in issue #15

Date: 2026-11-17
Status: done for the claims listed, against the rows scored by then. The run on Qwen2.5 had 25 of its variants scored and the third model none.

Issue #15 proposes a two-step sprint on the results of this date: one ball per group of weights, and a codeword at zero. It states those results as headline claims. This note reads each claim against what was measured, the way the repository's rule asks. Issue #14 asks for one of its steps, a baseline with an importance matrix, and is answered at the end.

**Measured** means a figure from a committed output of the record [model-output-qwen25-2026-11-17](../experiments/model-output-qwen25-2026-11-17/). **Derived** means arithmetic from such figures or from the limits table. **None** means nothing in the repository bears on the claim.

## Result

- **The headline figures are right.** `EBG128:b560s16` scores a mean KL of 0.0195 at 4.5 bits where llama.cpp's `Q4_K_M` scores 0.0333 at 4.81, 17 errors of the gap apart. Integer levels in a ball are close to E8 in a ball. A codeword at zero is worth a factor of 17 per row.
- **Four claims are not supported.** The saving in memory, the place of the code against Shannon's floor, the claim about mixture-of-experts models, and the claim about safety. The first two are wrong as stated and the last two were not tested.
- **The saving is stated with the wrong arm.** 14 percent and 34.5 GB belong to the arm at 4.12 bits, which ties `Q4_K_M`. The arm that beats it is at 4.5 bits and saves 6.5 percent.
- **"Validated" is one model and one text**, with bits that are counted and not stored. A third model is being scored.

## Claims

| # | Claim | Verdict | What was found | Basis |
|---|---|---|---|---|
| 1 | `EBG128:b560s16`: mean KL 0.0195 and 93.1 percent first-choice agreement at 4.50 bits | Confirmed, on one model and one text | 0.019549 and 93.08 percent on `Qwen2.5-1.5B-Instruct`, wikitext-2 test set, 10,200 positions. The bits are log2 of the codebook and 16 for the scale. No file of that size exists | measured, `experiments/model-output-qwen25-2026-11-17/results.jsonl` |
| 2 | It outperforms uncalibrated `Q4_K_M` (0.0333 at 4.81 bits) by 17 standard errors and saves 0.31 bits | Confirmed, with one word to change | 17.0 errors of the gap. The tool's errors treat positions as independent and are too small, as both model-output records say. "Uncalibrated" is right: no importance matrix was used on either side | measured, `experiments/model-output-qwen25-2026-11-17/results.jsonl` |
| 3 | "14% VRAM reduction", "saves about 34.5 GB on a 400B parameter model" | Wrong arm, and not a saving in memory | 0.69 bits is 14.3 percent of 4.812 and 34.4 GB per 400 billion weights. That is the arm at 4.124 bits, `EB128@128:16`, which scores 0.0331 against 0.0333: a tie. The arm that wins is at 4.5 bits: 0.31 bits, 6.5 percent, 15.6 GB. Neither is a saving in memory. No arm has a format that a kernel can compute from. As things stand a model would be decoded when loaded and held at 8 bits a weight or more | derived |
| 4 | It "eliminates all mixed-precision heuristics" | True of the arm, not shown to be a gain | Every block matrix has the same code. `Q4_K_M` here has 28 matrices in a 6-bit type. Whether one code for all matrices is better than a mix was not tested: no arm here gives some matrices more bits | measured, `experiments/model-output-qwen25-2026-11-17/results.jsonl` |
| 5 | Integer levels in a 32-weight ball (0.0367) do nearly as well as E8 in a ball (0.0339) | Confirmed | 0.036683 and 0.033934, 0.34 dB of KL apart, 2.6 errors of the gap. The integer arm at 4.5 bits with 560 bits for the point was not scored when this was written | measured, `experiments/model-output-qwen25-2026-11-17/results.jsonl` |
| 6 | "Classical vector quantization often centers codebooks by subtracting the mean" | True of this repository's harness. Not checked for others | `harness/tq.py` subtracts the mean of the codewords, a vector 0.82 steps long. The first record's plan asked for codebooks "shifted to mean zero". NestQuant and the cut codebooks of this repository keep zero | measured, `experiments/model-output-qwen25-2026-11-17/rows.out` |
| 7 | The shift "causes a 17x explosion in output KL"; a zero codeword "patches this degradation completely, with zero mean weight-MSE penalty (+/-0.06 dB)" | Confirmed with one scale per row. Smaller with groups of 32 | Per row: 3.658 to 0.209 at 16 levels, 1.041 to 0.050 at 32. With groups of 32: 0.0551 to 0.0434, a fifth. In weight error the change is -0.06 to +0.06 dB in the mean of each family, on 59 matrices | measured, `experiments/model-output-qwen25-2026-11-17/results.jsonl` and `experiments/model-output-qwen25-2026-11-17/centre.out` |
| 8 | It happens on "outlier-heavy" matrices such as `ffn_down` | Likely, not isolated | In the last layer's `ffn_down` ten rows hold 71 percent of the error, and the worst is at +1.9 dB where the scalar rule has -8.0. That is weight error. No arm repairs only those rows, so the output test shows the centre is the cause and not which rows carry it | measured, `experiments/model-output-qwen25-2026-11-17/rows.out` |
| 9 | "Reconstruction error ... becomes completely independent of the underlying weight distribution, rendering the pipeline 100% data-free" | Half right | With one ball per group the weight error is the same on every matrix: -25.60 dB for this arm, known before anything is rounded. The output is not: the same code will give a different KL on another model. And "data-free" means no calibration text is used, which is also true of the scalar rule and of `Q4_K_M` as run here | measured, `experiments/model-output-qwen25-2026-11-17/results.jsonl` |
| 10 | The framework "operates right at the theoretical Shannon boundary" | Wrong | Shannon's floor for Gaussian weights at 4.5 bits is -27.09 dB. This arm is at -25.60, which is 1.5 dB above it. Two known costs are in that gap: the E8 cell is 0.88 dB from the best possible cell, and the scale takes an eighth of a bit. The arm is also 4.9 dB under the scalar rule, which is the larger fact | derived from the first two sections of `bounds/bounds.out` |
| 11 | "Zero dataset exposure", "zero risk of data leakage", no silent loss of "alignment guardrails" | The first is true. The rest was not measured | No calibration data is read. Nothing here measures any behaviour but KL divergence and first-choice agreement on one prose text. A mean KL of 0.02 is small and is not zero | none |
| 12 | "Native resistance to sparse MoE degradation": a zero codeword stops inactive experts from making noise | Not tested, and not what was found | No mixture-of-experts model was run. The finding is about rows where one weight is tens of times the rest, under one scale for the row. It says nothing about experts that are not routed to, whose weights are not sparse | none |
| 13 | The lookup table "is only 0.3 MB" and fits a CPU cache | True for one codebook only | 0.29 MB is the table for integer levels over 32 weights at 4 bits a weight for the point. With 136 bits for the point it is 0.44 MB. For integer levels over 128 weights with 560 bits, the arm nearest the headline, it is 30.5 MB. For E8 over 128 weights at 4 bits it is 305 MB | measured, `shape/ballindex.out`, and derived |
| 14 | Decoding takes "6-to-10 table lookups per weight" that can run in parallel | The count is right. The parallel form is the open question | At 16 levels, 5.8 to 9.7 look-ups a weight to number a block and 6.8 to 10.7 to read it back. Each is a comparison and a subtraction of numbers as long as the block's index, 128 to 576 bits, and each depends on the one before. Blocks are independent, so blocks can run in parallel. Weights inside a block cannot, as the code stands | measured, `shape/ballindex.out` |
| 15 | "The mathematical fundamentals are validated, preregistered, and verified on active branches" | Preregistered: yes. Validated: on one model | Every arm cited has its plan committed before its score. The headline arm is from an extension written after 14 variants were seen, and the record says so. One model, one text, counted bits. No arm of this kind was scored on gemma | checked |

## What the proposal's two steps need first

**Step 1, generality and calibration.** A third model, `SmolLM2-1.7B-Instruct`, is planned at `14726fd` and is being scored. An importance-matrix baseline is issue #14 and is answered below.

**Step 2, a kernel.** Before any work on SIMD, two things decide whether it is worth doing.

- **Which arm.** The table and the length of the numbers grow with the block. Integer levels over 32 weights with an 8-bit scale are the cheapest arm that keeps most of the gain: 0.44 MB and 136-bit numbers. Its KL at 4.5 bits is not measured: the E8 form of it scores 0.0208.
- **What the kernel computes from.** If weights are decoded once at load, speed matters little and memory is not saved. If the kernel decodes as it multiplies, every block of 32 weights costs about 230 look-ups in numbers of 136 bits, each waiting on the last, against a shift and a mask for `Q4_0`. That gap has to be measured on one matrix before it is designed around.

## Issue #14: the baseline with an importance matrix

The comparison so far is rounding with no calibration text on both sides. `Q4_K_M` with an importance matrix reads such a text, and it will do better than without.

**What can be run here now, and is planned.** An importance matrix from the wikitext-2 training text, llama.cpp's `Q4_K_M` and `Q4_0` made with it, and both scored on the test text beside the arms above, on both models. The calibration text is from the same source as the test text, which favours the baseline. This answers the question in the measure the records use.

**What cannot be run here as asked.** MMLU, HumanEval and GSM8K in full on a 4-core machine with no GPU: each is hours per variant, and HumanEval runs generated code. A subset can be scored. It is unlikely to separate these arms: at a mean KL of 0.02 to 0.06 the first choice changes on 7 to 12 percent of positions, and a difference of a point or two in accuracy needs several thousand questions to see. The plan for that will say what size of difference the subset could detect before anything is scored.

## Disclosure

Written by an AI agent session under the owner's direction, on 2026-11-17.
