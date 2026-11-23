# Preregistration: integer ball on Gemma

Status: plan before data. This file commits before any Gemma weight passes through the ball. Weights and datasets stay out of git. Committed outputs carry numbers and configuration only.

## Goal

The integer ball has scores on Qwen2.5-1.5B, SmolLM2-1.7B and Llama-3.2-1B. Gemma has an E8 score only.

This run scores the ball on Gemma at 4.5 bits against Q4_K_M, on the models listed below, smallest first.

## Models to test

| # | Model | Size at 16-bit |
| --- | --- | --- |
| 1 | Gemma 3 270M, `gemma-3-270m-it` | About 0.5 GB |
| 2 | Gemma 3 1B, `gemma-3-1b-it` | About 2 GB |
| 3 | Gemma 2 2B, `gemma-2-2b-it` | About 5 GB |
| 4 | Gemma 3 4B, `gemma-3-4b-it` | About 8 GB for the text weights (~8.6 GB with vision tower) |
| 5 | Gemma 4 E4B, `gemma-4-E4B-it` (original run model, primary) | About 15 GB total (~7.9 GB in 294 block matrices, 3.95B weights + ~7.0 GB in token/embedding tables, 3.49B weights; 7.5B total) |
| 6 | Gemma 7B, `gemma-7b-it` | About 16 GB |
| 7 | Gemma 2 9B, `gemma-2-9b-it` | About 18 GB |
| 8 | Gemma 3 12B, `gemma-3-12b-it` | About 24 GB |

Gemma 3 27B and Gemma 2 27B stay out of scope at full precision. Their weights alone are about 54 GB at 16-bit, and scoring needs reference plus rounded model plus operating system and activations in memory at once. They enter scope only with a quantized reference.

`gemma-4-E4B-it` (row 5, ~15 GB at 16-bit) is the primary model for this run. It carries the original Gemma E8 score (`experiments/model-output-gemma-e4b-2026-11-15`). `gemma-3-4b-it` (row 4, ~8 GB text weights) is a separate Gemma 3 model in the sweep.

## Model and reference

| Item | Choice |
| --- | --- |
| Primary model | `gemma-4-E4B-it`, 7.5B total weights (~15 GB at 16-bit: 3.95B weights / ~7.9 GB in 294 block matrices across 42 layers + 3.49B weights / ~7.0 GB in token/embedding tables) |
| Unread matrices | 18 of 42 layers reuse earlier keys/values, so 36 of 294 block matrices (18 `attn_k` + 18 `attn_v`, 55,050,240 weights) are never read by `llama.cpp` (`experiments/model-output-gemma-e4b-2026-11-15/README.md`). The results record reports both counted-all (294 matrices, 3.95B weights) and active-only (258 matrices, 3.90B weights) bit and weight accounting explicitly. |
| Primary reference | 16-bit model (`gemma-4-E4B-it` at F16/BF16), matching the ball runs on Qwen2.5, SmolLM2 and Llama-3.2 |
| Secondary reference (bridge) | `gemma-4-E4B-it-Q8_0.gguf` from `experiments/model-output-gemma-e4b-2026-11-15`, scored on that run's exact `eval.txt` (`harness/build_eval_text.py`, 10,200 positions) as the bridge to `VE832:16 = 0.0747` |
| Headline rule | Headline deltas use the primary 16-bit reference on wikitext-2 and Python source. Bridge figures sit in a separate table and are never mixed with headline text scores. |

## Texts

- Primary text: `wikitext-2` test (`wiki.test.raw`, 40 passages of 512 tokens, second half scored = 10,200 positions), matching the Qwen2.5, SmolLM2 and Llama-3.2 ball runs.
- Second text: Python source (`conversion/base.py` from `llama.cpp` at `eada89c`, 40 passages = 10,200 positions), matching `experiments/second-text-python-2026-11-18`.
- Bridge text (`P7` only): `eval.txt` built by `harness/build_eval_text.py` from `llama.cpp` at `427291b` (40 passages of 512 tokens, second half scored = 10,200 positions), matching `experiments/model-output-gemma-e4b-2026-11-15`.

The harness uses 10,200 scored positions on all three texts where it allows. The results record states the exact position count and SHA-256 for each text.

## Arms

Primary arms run at 4.5 bits using one ball per group of 128 weights with a 16-bit per-group scale (`ZBG128:b560s16`, 560 point bits + 16 scale bits = 4.500 bits/weight; imatrix column scales add `16 / rows` bits, ~0.005 bits/weight).

Continuity cells at ~4.005 and ~5.005 bits pin their group-size and scale rules explicitly so bit-rate scaling is not confounded by a group-size switch:
- **Per-128-group continuity (primary for `P8`/`P9`):** `ZBG128:b496s16` / `hZBG128:b496s16` (496 point bits + 16-bit per-group scale = 4.000 bits plain, ~4.005 bits with imatrix column scales) and `ZBG128:b624s16` / `hZBG128:b624s16` (624 point bits + 16-bit per-group scale = 5.000 bits plain, ~5.005 bits with imatrix column scales).
- **Per-row (`R`) continuity (for direct comparison to `2026-11-15`'s `VE8R:16` = 4.005 bits and `VE8R:32` = 5.005 bits):** `ZB128@R:16` and `ZB128@R:32` (128-weight ball, 16-level or 32-level count, one 16-bit scale per row `R` = `4.000 + 16/cols` and `5.000 + 16/cols` bits), reported in their own continuity rows separate from the per-128-group cells.

| Arm | Definition |
| --- | --- |
| Plain ball | `ZBG128:b560s16`: integer ball over 128 weights, 16-bit per-group scale, calibration-free |
| Ball, real-text matrix | `hZBG128:b560s16`: half-strength (`p = 0.25`) importance matrix from `wikitext-2` training text |
| Ball, data-free matrix | `hZBG128:b560s16~rand`: half-strength (`p = 0.25`) importance matrix from 80,000 random vocabulary pieces |
| Ball, loud columns | `ZBG128:b560s16+K10`: ten loud `ffn_down` columns per layer (most squared weight removed by 5-sigma row clip) held at 16-bit. Selection uses Gemma weights alone. |
| Ball, matrix and repair | `hZBG128:b560s16+K10`: real-text half-strength matrix plus the ten loud `ffn_down` columns held at 16-bit |
| E8 baseline | `EBG128:b560s16` (128-weight E8 ball) and `VE8Z32:16` (`centred=False`, Refs PR `#46` / `#16`) on headline texts; `VE832:16` (`centred=True`) plus `VE8Z32:16` (`centred=False`) on the `P7` bridge text |
| Hadamard-turned E8 | E8 turned by the order-8 Hadamard matrix (`HBG128:b560s16`). Runs if the harness supports the turn on Gemma; otherwise listed as a follow-up. **Harness guard (`PR #41` `SQ41-01`):** never cast `R * t` to `float16` before `turn()` (`harness/tq.py:315` trap, which produces 42%-48% non-zero ghosts on zero-centred Voronoi E8); keep the inverse transform in `float32`/`float64` prior to final `float16` output. |
| Baselines | `Q4_K_M` (with and without imatrix) and `Q4_0` scored on the same texts and the same reference |

## Metrics

Primary metric: mean KL divergence against the reference model (`llama-perplexity --kl-divergence`). Second metric: top-1 token keep rate.

Error counting follows the ball runs (standard errors from `llama-perplexity`). Deltas against `Q4_K_M` use the same units as the ball runs.

Each arm reports counted bits (all 294 block matrices and active 258 block matrices), mean KL and top-1 keep rate for each text.

## Predictions

Each prediction gets a hold or fail verdict in the results record after scoring. Pre-data prior for `P4`/`P6`: committed `experiments/loud-columns-2026-11-18/README.md` found that on Gemma the top-10 `ffn_down` clip units carried only `0.05%` of clip damage, and `+K10` on top of an importance matrix added zero gain on SmolLM2 (`0.0279` vs `0.0282`) and Qwen2.5 (`0.0222` vs `0.0227`, `0.6` SE). `P4` and `P6` are therefore sharp, falsifiable tests with explicit thresholds below.

| ID | Prediction | Verdict |
| --- | --- | --- |
| P1 | Ball with real-text matrix (`hZBG128:b560s16`) records lower mean KL than `Q4_K_M` (same imatrix setting) on `wikitext-2` at 4.5 bits by at least 2.0 SE. |  |
| P2 | Plain ball (`ZBG128:b560s16`) records higher mean KL than `Q4_K_M` on `wikitext-2` at 4.5 bits (test of whether Gemma behaves like Llama-3.2 or like Qwen2.5/SmolLM2). |  |
| P3 | Ball with real-text matrix (`hZBG128:b560s16`) records lower mean KL than plain ball (`ZBG128:b560s16`) on both headline texts at 4.5 bits by at least 2.0 SE. |  |
| P4 | Matrix with repair (`hZBG128:b560s16+K10`) records lower mean KL than matrix alone (`hZBG128:b560s16`) on both headline texts at 4.5 bits by at least 2.0 SE (fails if gain < 2.0 SE on either text, consistent with the `0.05%` Gemma clip prior and SmolLM2/Qwen2.5 null results). |  |
| P5 | Data-free matrix (`hZBG128:b560s16~rand`) retains at least 70 percent of the real-text matrix KL gain over plain ball on `wikitext-2`. |  |
| P6 | Keeping the ten loud `ffn_down` columns (`ZBG128:b560s16+K10`) zeroes weight error in those ten columns (exact 16-bit preservation) and lowers overall mean KL vs plain ball (`ZBG128:b560s16`) by at least 2.0 SE (or >= 5% relative KL reduction) on `wikitext-2`. |  |
| P7 | Bridge check: on `eval.txt` (`harness/build_eval_text.py`, 10,200 positions) against `gemma-4-E4B-it-Q8_0.gguf`, `VE832:16` (`centred=True`, matching `2026-11-15`) lands within +/-0.002 mean KL (+/-3% relative) of the committed `0.0747` figure (`experiments/model-output-gemma-e4b-2026-11-15/results.out`). |  |
| P8 | At ~5.005 bits (`hZBG128:b624s16`, per-128-group scale), ball with matrix improves on its 4.5-bit KL (`hZBG128:b560s16`) by at least 2.0 SE and holds its rank. |  |
| P9 | At ~4.005 bits (`hZBG128:b496s16`, per-128-group scale), KL rises for all arms relative to 4.5 bits and ball with matrix stays ahead of `Q4_0` (4.5 bits) by at least 2.0 SE. |  |
| P10 | Top-1 keep rate follows the KL ranking across arms on both headline texts (Spearman rho >= 0.90, no inversion > 0.2 percentage points). |  |
| P11 | MMLU (2,000 questions from `mmlu-test.bin`, if run) keeps ball-arm accuracy within the band recorded for the ball runs (`experiments/mmlu-subset-2026-11-18/README.md`: -0.70 to +0.15 percentage points vs 16-bit reference, and within 1.5 SE of zero loss), with answer-agreement rate ordered by KL. |  |

## Sequence

Step 1. This file commits before any Gemma weight passes through the ball.

Step 2. Run a pilot on one Gemma layer. Record runtime and peak RAM. Confirm the full-run estimate from the pilot before the full run starts.

Step 3. Verify zero-codeword preservation in the harness on Gemma (cross-referencing PR `#46` `centred: bool = True`/`False`, Refs `#16`): confirm `0` is an exact codeword in all integer/E8 ball arms and in `centred=False` (`VE8Z...`) Voronoi arms, while keeping `centred=True` (`VE832:16`) isolated to the backward-compatible `P7` bridge check. If Hadamard-turned E8 runs, also verify no `float16` pre-`turn()` ghosting (`PR #41` `SQ41-01`, `harness/tq.py:315`).

Step 4. Run the arms in order. Plain ball first. Matrix arms next. Repair next. Combined arm last. Run the `P7` bridge check on `eval.txt` against `gemma-4-E4B-it-Q8_0.gguf` in its own bridge table.

Step 5. Score MMLU only after the KL arms land. The results record states the question count (2,000 from `mmlu-test.bin`) and the scoring resolution.

Step 6. Pack the best 32-weight-block ball arm (`PZ32:b136s8` / `hPZ32:b136s8`) into the file format (`shape/ballpack.py`). Read the packed file back and check it bit for bit.

Step 7. Write README.md, results.out and results.jsonl in the experiment folder. File the work for review.

Experiment folder: `experiments/model-output-gemma-ball-2026-11-23`.

## Risks

| Risk | Mitigation |
| --- | --- |
| `Q4_K_M` leads E8 on Gemma at 4.5 bits, 0.069 against 0.075. The ball may trail `Q4_K_M` as well. | Report the measured result in full. A trail counts as a result. |
| Matrix gain varies by model. The matrix may help little on Gemma. | Plain ball and matrix arms both run, so the matrix contribution reads on its own. |
| Loud columns on Gemma sit in different layers than on Llama, and earlier clip work showed top-10 units carried only `0.05%` of damage. | Derive the columns from Gemma weights and test `P4`/`P6` against the explicit 2.0 SE threshold. |
| `gemma-4-E4B-it` is 7.5B total weights (~15 GB at 16-bit), far larger than the 1 to 1.7B models. | The one-layer pilot gates the full run and confirms runtime and peak RAM on the 64 GB host. |
| This run measures quality only. Decode speed stays as measured in the kernel work. | Keep speed claims tied to kernel measurements. |

## Completion

- This file sits in git history before the first scored output.
- All arms have scores on both headline texts at 4.5 bits, with continuity cells and the `P7` bridge check complete.
- README.md, results.out and results.jsonl sit in the experiment folder with full traceability to committed outputs and explicit counted-vs-active weight accounting.
- The review completes. The one-pager carries the Gemma ball row with the result as measured.
