# Plan: where the bits and the damage go on a real model's output

Written on 2026-11-17 and committed before any variant of this record was built or scored. The numbers under "What is already known" are from the record [model-output-gemma-e4b-2026-11-15](../model-output-gemma-e4b-2026-11-15/README.md) and are the only results this plan was written with.

## Why this run

The first model-output record found that weight error does not say what the output does, and that the output followed how the scale is set. Four records of 2026-11-17 then studied the shape of a code's region and cell, on weight error only. None of them scored a model.

This run goes back to the output with one idea: treat the loss as a budget with parts, and work on the largest part. It asks five questions.

1. Which kind of matrix costs the most output per bit, under the rule that clips nothing?
2. Do bits set by equal gain per bit beat equal bits for every kind?
3. Does keeping the few largest entries apart let a scale per row do the work of a scale per 32 weights?
4. Does a scale stored in 8 or 6 bits cost anything?
5. Does the damage of a clip add up over layers the way the damage of rounding does?

It also computes three measures of a weight change that no record has tried, each with its predicted failures.

## What is measured

Everything is as in the first model-output record. The reference is `gemma-4-E4B-it-Q8_0.gguf`, 8,192,953,472 bytes, SHA-256 `f8854aa4480df62585a279e7ca0a881554fc18a41c59c4f62642d16a2ae47012`. The 294 matrices of the 42 blocks are changed and nothing else. Each changed model is scored by `llama-perplexity --kl-divergence` against the reference's saved outputs on 40 passages of 512 tokens, the second half of each scored: 10,200 positions.

The score is the mean KL divergence from the reference's next-token distribution, and the share of positions where the top token stays the same. "KL" below means that mean.

The text is built by `harness/build_eval_text.py` from the llama.cpp repository at commit `427291b`.

Bits are counted as before: log2 of the levels per weight, plus the bits of each stored scale, plus the bits of anything else stored. Rounded values are written as 16-bit floats. No packed format exists for any arm here and no decoding speed is measured.

## First step: the harness must repeat itself

Before any new variant is scored, three known variants are scored again on this machine with the reference outputs rebuilt.

| Variant | Recorded KL | Passes if |
|---|---|---|
| `unchanged` | 0.000000 | under 0.00001 |
| `X32:16` | 0.117297 | within 0.002 |
| `XR:16` | 0.448262 | within 0.006 |

If any of the three fails, the run stops. The record then says what differs, and every comparison with the earlier record is dropped or redone on the new footing.

## What is already known

| Variant | Bits | KL |
|---|---|---|
| `X32:16`, the `Q4_0` rule | 4.500 | 0.1173 |
| `X32:32` | 5.500 | 0.0313 |
| `XR:16` | 4.005 | 0.4483 |
| `XR:32` | 5.005 | 0.1180 |
| `XR:32` with the step times 0.7 | 5.005 | 0.1011 |
| `VE832:16`, E8 with the same rule | 4.500 | 0.0747 |
| llama.cpp `Q4_K_M` | 4.803 | 0.0689 |
| clip only at 3.45, 5 and 8 standard deviations of the row | | 0.3179, 0.2001, 0.0100 |
| the entries of the clip at 3.45 moved at random | | 0.0954 |
| clip at 5 on `ffn_down` only | | 0.1638 |

Arm S, one scale per matrix, at 4 bits on one kind of matrix at a time: `ffn_down` 1.989, `attn_output` 0.962, `ffn_up` 0.172, `ffn_gate` 0.079, `attn_v` 0.067, `attn_q` 0.058, `attn_k` 0.049. Their sum is 3.376 and the same arm on all kinds at once gives 6.505.

Not known when this was written: the KL of any kind alone under the `X` rules, of any arm with mixed levels, of any arm that keeps entries apart, of any arm with a scale under 16 bits, and of a clip on single layers.

## Part A: one kind at a time under the rule that clips nothing

`X32:Q@KIND` for the seven kinds and Q = 8, 16 and 32. That is 3.5, 4.5 and 5.5 bits on the kind that is rounded, with the rest of the model unchanged. 21 variants. `X32:8` on all kinds at once is scored too.

"KL per weight" of a kind is its KL divided by its number of weights.

| | Prediction | Fails if |
|---|---|---|
| A1 | `ffn_down` has the largest KL of the seven kinds at each Q. | another kind is larger at any Q |
| A2 | Per weight, the two kinds that write into the stream, `ffn_down` and `attn_output`, are the two most sensitive at Q = 16. | either is not among the top two |
| A3 | One more bit divides the KL of a kind by 2.5 to 6. Of the 14 steps (seven kinds, 8 to 16 and 16 to 32), at least 12 lie in that range. | fewer than 12 do |
| A4 | In this mild range the parts add: at Q = 16 the sum of the seven KLs is 0.75 to 1.25 times the KL of `X32:16` on all kinds (0.1173). | the ratio is outside that range |

A4 is the opposite of arm S at 4 bits, where the sum was 0.52 of the whole. The guess is that parts add when each is small.

## Part B: bits by equal gain

A design is at its best when one more unit of resource buys the same gain everywhere. For rounding that means: at the best split of a bit budget, one more bit lowers the KL by the same amount per weight in every kind.

The rule is fixed here and written as `allocate.py` before Part A is scored.

- Levels a kind may get: 4, 6, 8, 12, 16, 22, 32, 46 or 64, with one scale per 32 weights. Bits are log2 of the levels plus 0.5.
- The KL of a kind at a number of bits is read from its three points of Part A: a straight line in the logarithm of KL between them, and the end slopes beyond them.
- The KL of a mixed arm is predicted as the sum over kinds. Part A's prediction A4 is what that rests on.
- The best arm is the choice of levels with the smallest predicted KL whose mean bits per weight, over all 3.95 billion weights, is at most the budget. All 9^7 choices are tried.
- The reversed arm is the choice with the largest predicted KL among those that use the same levels as the best arm, in any order over the kinds, with mean bits between the budget less 0.05 and the budget. If none exists the window is widened by 0.05 at a time, and the record says so.

Budgets: 4.5 bits, against `X32:16`, and 3.5 bits, against `X32:8`.

`allocate.py` prints the two best arms, the two reversed arms and their predicted KLs. That output is committed and pushed before any of the four is scored.

| | Prediction | Fails if |
|---|---|---|
| B1 | At 4.5 bits the best arm has a KL at least 15 percent under that of `X32:16`: 0.0997 or less. | it is above 0.0997 |
| B2 | At 3.5 bits the best arm has a KL at least 15 percent under that of `X32:8`. | it is not |
| B3 | The measured KL of each best arm is 0.7 to 1.3 times the KL the program predicted for it. | either is outside |
| B4 | Each reversed arm has a KL at least 15 percent above the uniform arm at its budget. | either is not |

Not predicted: whether the best arm at 4.5 bits passes `VE832:16` (0.0747). The record reports it.

llama.cpp's `Q4_K_M` already gives some tensors more bits by a fixed rule. The record will quote that rule from the source at commit `427291b` and set its choices beside the ones found here.

## Part C: the largest entries kept apart

The clip probes say a few large entries carry much of the damage. The rule that clips nothing keeps them, and pays with a coarse step for the whole row. A third way keeps them exact and apart.

`XRo:Q~L` is `XR:Q` with this change. In each row, the entries beyond L standard deviations of the row are kept as they are. The scale of the row is set from the largest entry that is left, and the rest are rounded as `XR` does. `X32o:Q~L` is the same with one scale per 32 weights: the kept entries are those beyond L standard deviations of their row, and each group's scale is set from its largest entry that is left.

Bits: those of the plain arm, plus for each kept entry 16 bits for its value and its position in the row, counted as log2 of the row's length rounded up.

Variants: `XRo:16~L` and `XRo:32~L` for L = 3.45, 5 and 8. `XRo:16~2.5`. `X32o:16~3.45`. Eight in all.

| | Prediction | Fails if |
|---|---|---|
| C1 | `XRo:16~3.45` has a KL at most 0.75 of `XR:16`: 0.336 or less. It costs under 0.06 bit more. | either part fails |
| C2 | `XRo:32~3.45` has a KL at most 0.75 of `XR:32`: 0.0885 or less. | it is above |
| C3 | Kept entries do not make a row scale as good as a scale per 32 weights: `XRo:16~3.45` stays above the KL of `X32:16` (0.1173). | it is at or under 0.1173 |
| C4 | At L = 8 nothing changes: `XRo:Q~8` is within 10 percent of `XR:Q` at both Q. | either is not |
| C5 | With a scale per 32 weights the gain is small: `X32o:16~3.45` is 5 to 30 percent under `X32:16`. | it is outside that range |

Keeping large weights apart is published work (SpQR and others). What is new here is the question in C3.

## Part D: the bits of a scale

A scale per 32 weights stored in 16 bits costs 0.5 bit per weight. No cell shape can save more than 0.25 bit per weight at high rates (`bounds/bounds.out`). So the scale is the larger part of the budget.

`X32sB:Q` is `X32:Q` with each group's scale stored in B bits. The row's largest entry R is stored in 16 bits. A group's scale is the smallest of the values R times 2^(-8c/(2^B - 1)), for c from 0 to 2^B - 1, that is at least the group's largest entry. So nothing is clipped, and a group far smaller than its row gets a coarser step than it needs. Bits: log2 of the levels, plus B/32, plus 16 over the row's length.

Variants: `X32s8:16`, `X32s6:16`, `X32s8:18`, `X32s6:20`.

| | Prediction | Fails if |
|---|---|---|
| D1 | An 8-bit scale costs nothing that shows: `X32s8:16`, at 4.25 bits, is within 10 percent of `X32:16`. | it is not |
| D2 | A 6-bit scale costs under 20 percent: `X32s6:16`, at 4.19 bits, is at most 1.2 times `X32:16`. | it is above |
| D3 | Spending the saved bits on levels wins: `X32s8:18`, at 4.42 bits, has a KL at most 0.90 of `X32:16`, and `X32s6:20`, at 4.51 bits, at most 0.80. | either is above |

## Part E: does the damage add up over layers

A clip moves every entry it touches toward zero. If the same thing is done in every layer, the changes may add up in step at the output. Rounding has no preferred direction, so its changes should add like noise.

Six layers are used: 0, 8, 16, 24, 32 and 40. `clip:5@ffn_down` is applied to each of the six alone, and to all six at once. The same is done with `X32:16@ffn_down`. 14 variants.

| | Prediction | Fails if |
|---|---|---|
| E1 | For the clip, the KL of the six at once is at least 1.5 times the sum of the six alone. | it is under 1.5 times |
| E2 | For rounding, the KL of the six at once is 0.75 to 1.25 times the sum of the six alone. | it is outside |

If a single layer's KL is under five times its standard error, the record says so and the prediction it enters is marked as not decided.

## Part F: three more measures of a weight change

No scoring. As in rounds 5 and 6 of the first record, each measure is computed on the 42 matrices of layers 0, 8, 16, 24, 32 and 40, for the 26 variants of round 5 and for `flip:3.45`. Their KLs are known. The measures are not: none has been computed for any variant.

Write E for the change of a matrix W, row by row E_i and W_i.

- **Along the rows.** The sum over rows of (E_i . W_i)^2 / |W_i|^2, over the squared size of W. It is the part of each row's change that lies along the row itself. A change of that kind scales a unit's whole output up or down. **Along the columns** is the same with columns.
- **Tree shift.** The squared distance between the sorted tree sizes of |W + E| and of |W|, over the squared size of the tree sizes of |W|, from `persist/persistlib.py`. It is the measure built on the "neural persistence" of Rieck and others (2019).
- **Squared error and spectral error**, recomputed as the yardsticks.

Each is a mean over the 42 matrices in dB.

| | Prediction | Fails if |
|---|---|---|
| F1 | Along the rows ranks the 26 variants by KL with a rank correlation of at least 0.8, above that of the squared error on the same matrices. | either part fails |
| F2 | Along the rows has `clip:3.45` as worse than `XR:32`, as it is. | it has it as better |
| F3 | Along the rows has `clip:5` as better than `XR:32`, when it is worse. So it is not a certificate either. | it has it as worse |
| F4 | Along the rows has `flip:3.45` as better than `clip:3.45`, as it is. | it does not |
| F5 | Tree shift has both clips as worse than `XR:32`, as they are, and has `XR:32` with the step times 0.7 as worse than `XR:32`, when it is better. | any of the three orderings is the other way |
| F6 | Tree shift ranks the 26 variants worse than the spectral error does: its rank correlation is under the spectral error's on the same matrices. | it is equal or above |

F3 and the second half of F5 are predictions that a measure fails. They are scored like the others.

**A held-out test.** The same four measures are then computed for every variant of Parts B, C and D that changes all kinds. Each measure's rank correlation with KL over those new variants is reported beside its value on the 26. Nothing is predicted for it. The measures are fixed here, before those variants exist.

**The weights alone.** For each of the 294 matrices, with nothing rounded: the normalised persistence (the matrix's own largest weight, the count of neurons, p = 2), and the spread of its rows and of its columns, each as the standard deviation of the logarithm of the row's or column's root mean square.

| | Prediction | Fails if |
|---|---|---|
| F7 | Persistence does not say which kinds are sensitive: over the seven kinds, the rank correlation between mean persistence and KL per weight at Q = 16 in Part A is between -0.75 and 0.75. | it is outside |
| F8 | The two kinds that write into the stream have a larger spread of rows than each of the five that read from it, in the mean over layers. | any reading kind has a larger one |

## One arm that joins the parts

After Parts A to D, one arm is built by a rule fixed here: the levels of the best arm of Part B at a budget of 4.5 bits, found again with 8-bit scales, so that each kind costs log2 of its levels plus 0.25, plus the row scales. Its choice of levels is committed before it is scored.

| | Prediction | Fails if |
|---|---|---|
| G1 | The joined arm, at 4.5 bits or under, has a KL of 0.085 or less: at least 27 percent under the `Q4_0` rule at equal bits. | it is above 0.085 |

Not predicted: whether it passes `VE832:16` (0.0747) or `Q4_K_M` (0.0689 at 4.80 bits).

## Order of work, and what is committed when

1. This plan, committed and pushed. The programs refuse to score anything unless this file is committed and pushed.
2. The new arms, their tests against a plain NumPy version on small matrices, and `allocate.py`. No scoring.
3. The first step: the three known variants.
4. Part A. Its results are committed and pushed.
5. The output of `allocate.py`, committed and pushed. Then Part B.
6. Parts C, D and E, in that order, each committed when it ends.
7. The joined arm's levels, committed. Then its score.
8. Part F.

## Limits known before the run

- One model, one text, one runtime, as before. Issue #3 asks for a second model and a second text, and another session has taken it.
- Part B fits its rule on the same text it is scored on. An allocation fitted on one text may not hold on another. That is not tested here.
- KL does not add in general. Parts B and the joined arm rest on A4. If A4 fails, their predictions are still scored, and the record says what they rested on.
- Standard errors treat positions as independent, which understates them. Differences of a few percent between two variants cannot be ordered.
- No arm here has a packed format. The bits of kept entries and of short scales are counted, not stored.
- No arm uses the model's activations.
- The three measures of Part F are chosen after seeing the KLs of the 26 variants, though before computing any measure on them. Only the held-out test is free of that.
