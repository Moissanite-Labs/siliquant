# Preregistration: integer ball on Gemma

Status: plan before data. This file commits before any Gemma weight passes through the ball. Weights and datasets stay out of git. Committed outputs carry numbers and configuration only.

## Goal

The integer ball has scores on Qwen2.5-1.5B, SmolLM2-1.7B and Llama-3.2-1B. Gemma has an E8 score only.

This run scores the ball on Gemma at 4.5 bits against Q4_K_M, on the models listed below, smallest first.

## Models to test

| # | Model | Size at 16-bit |
| --- | --- | --- |
| 1 | Gemma 3 270M, gemma-3-270m-it | About 0.5 GB |
| 2 | Gemma 3 1B, gemma-3-1b-it | About 2 GB |
| 3 | Gemma 2 2B, gemma-2-2b-it | About 5 GB |
| 4 | Gemma 3 4B, gemma-3-4b-it, the original run model, gemma-4-E4B-it | About 8 GB for the text weights |
| 5 | Gemma 7B, gemma-7b-it | About 16 GB |
| 6 | Gemma 2 9B, gemma-2-9b-it | About 18 GB |
| 7 | Gemma 3 12B, gemma-3-12b-it | About 24 GB |

Gemma 3 27B and Gemma 2 27B stay out of scope at full precision. Their weights alone are about 54 GB at 16-bit, and scoring needs reference plus rounded model plus operating system and activations in memory at once. They enter scope only with a quantized reference.

The 4B model is the primary model for this run. It carries the original Gemma E8 score.

## Model and reference

| Item | Choice |
| --- | --- |
| Model | gemma-4-E4B-it, 7.5B weights, 294 block matrices |
| Primary reference | 16-bit model, matching the ball runs on Qwen, SmolLM2 and Llama |
| Secondary reference | 8-bit file from the original Gemma run, bridge to the E8 figure of 0.075 |
| Headline rule | Headline deltas use the primary reference. Bridge figures sit in a separate table. |

## Texts

Primary text: wikitext-2 test. Second text: Python source. Both texts match the ball runs.

The harness uses the same position count as the ball runs where it allows. The results record states the exact position count for each text.

## Arms

All arms run at 4.5 bits. Continuity cells run at 4.005 bits and 5.005 bits for the Gemma E8 comparison.

| Arm | Definition |
| --- | --- |
| Plain ball | Integer ball with calibration-free scaling |
| Ball, real-text matrix | Half-strength importance matrix derived from real text |
| Ball, data-free matrix | Importance matrix derived from random vocabulary pieces |
| Ball, loud columns | Ten loud ffn_down columns held at full precision. Selection uses Gemma weights alone. |
| Ball, matrix and repair | Real-text matrix plus repair on the loud columns |
| E8 baseline | E8 lattice scored through the same program as the ball arms |
| Hadamard-turned E8 | E8 turned by a Hadamard matrix. Runs if the harness supports the turn on Gemma. Otherwise the results record lists it as a follow-up. |
| Baselines | Q4_K_M and Q4_0 scored on the same texts and the same reference |

## Metrics

Primary metric: mean KL divergence against the reference model. Second metric: top-1 token keep rate.

Error counting follows the ball runs. Deltas against Q4_K_M use the same units as the ball runs.

Each arm reports counted bits, mean KL and top-1 for each text.

## Predictions

Each prediction gets a hold or fail verdict in the results record after scoring.

| ID | Prediction | Verdict |
| --- | --- | --- |
| P1 | Ball with real-text matrix records lower mean KL than Q4_K_M on wikitext-2 at 4.5 bits. |  |
| P2 | Plain ball records higher mean KL than Q4_K_M on wikitext-2 at 4.5 bits. |  |
| P3 | Ball with real-text matrix records lower mean KL than plain ball on both texts at 4.5 bits. |  |
| P4 | Matrix with repair records lower mean KL than matrix alone on both texts at 4.5 bits. |  |
| P5 | Data-free matrix retains at least 70 percent of the real-text matrix gain over plain ball. |  |
| P6 | Repair lowers error concentrated in the ten loud columns. |  |
| P7 | E8 through the same program lands close to the bridge figure after reference alignment. |  |
| P8 | At 5.005 bits, ball with matrix improves on its 4.5-bit KL and holds its rank. |  |
| P9 | At 4.005 bits, KL rises for all arms and ball with matrix stays ahead of Q4_0. |  |
| P10 | Top-1 keep rate follows the KL ranking across arms on both texts. |  |
| P11 | MMLU, if run, stays within the band recorded for the ball runs. |  |

## Sequence

Step 1. This file commits before any Gemma weight passes through the ball.

Step 2. Run a pilot on one Gemma layer. Record runtime and memory. Confirm the full-run estimate from the pilot before the full run starts.

Step 3. Verify the zero codeword in the harness on Gemma. The Qwen run lost its first reading to this fault.

Step 4. Run the arms in order. Plain ball first. Matrix arms next. Repair next. Combined arm last.

Step 5. Score MMLU only after the KL arms land. The results record states the question count and the scoring resolution.

Step 6. Pack the best arm into the file format. Read the packed file back and check it bit for bit.

Step 7. Write README.md, results.out and results.jsonl in the experiment folder. File the work for review.

Experiment folder: experiments/model-output-gemma-ball-2026-11-23.

## Risks

| Risk | Mitigation |
| --- | --- |
| Q4_K_M leads E8 on Gemma at 4.5 bits, 0.069 against 0.075. The ball may trail Q4_K_M as well. | Report the measured result in full. A trail counts as a result. |
| Matrix gain varies by model. The matrix may help little on Gemma. | Plain ball and matrix arms both run, so the matrix contribution reads on its own. |
| Loud columns on Gemma sit in different layers than on Llama. | Derive the columns from Gemma weights. Copying the Llama set would import the wrong repair target. |
| Gemma at 7.5B costs far more time than the 1 to 1.7B models. | The one-layer pilot gates the full run and sets the estimate. |
| This run measures quality only. Decode speed stays as measured in the kernel work. | Keep speed claims tied to kernel measurements. |

## Completion

- This file sits in git history before the first scored output.
- All arms have scores on both texts at 4.5 bits, with continuity cells complete.
- README.md, results.out and results.jsonl sit in the experiment folder with full traceability to committed outputs.
- The review completes. The one-pager carries the Gemma ball row with the result as measured.
