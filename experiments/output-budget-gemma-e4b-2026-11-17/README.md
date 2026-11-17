# Where the bits and the damage go on a real model's output

## Question

The first model-output record found that weight error does not say what the output does, and that the output followed how the scale is set. The records of 2026-11-17 on main then studied the shape of a code on weight error only. This record goes back to the output of `gemma-4-E4B-it` and treats the loss as a budget with parts.

1. Which kind of matrix costs the most output per bit, under the rule that clips nothing?
2. Do bits set by equal gain per bit beat equal bits for every kind?
3. Does keeping the few largest entries apart let a scale per row do the work of a scale per 32 weights?
4. Does a scale stored in 8 or 6 bits cost anything?
5. Does the damage of a clip add up over layers the way the damage of rounding does?

It also computes three measures of a weight change, one of them built on neural persistence, each with its predicted failures. Two extensions follow one result of question 5 to the single matrix it comes from.

## Status

`complete`. One model, one text, one runtime. No arm has a packed format: bits are counted, and rounded weights are scored from 16-bit float copies. The split of bits in question 2 and in the joined arm is fitted on the text it is scored on.

## Preregistration

| Plan | Commit | Covers | Predictions |
|---|---|---|---|
| [PREREG.md](PREREG.md) | `ec636b1`, pushed before any variant was built | Parts A to F and the joined arm: 56 scored variants | 20 of 27 hold, 5 fail, 2 not decided |
| [PREREG-extension-1.md](PREREG-extension-1.md) | `bdc2b2b`, pushed after the main results and before its own variants | Parts H and K: 37 scored variants | 4 of 4 hold |
| [PREREG-extension-2.md](PREREG-extension-2.md) | `c21dd06`, pushed after extension 1 and before its own work | Part L, weights only, and Part M: 6 scored variants | 1 of 5 holds |

In all, 25 of 36 predictions hold, 9 fail and 2 are not decided. Each extension says which results it was written with, and its predictions are not blind to them. The programs refuse to score unless the plan is committed and pushed, and from Part B on they also check the first step and the committed output of the allocation. [DEVIATIONS.md](DEVIATIONS.md) says which part ran under which check.

## Result in short

1. **Under the rule that clips nothing, no kind of matrix stands out.** Rounded alone to 16 levels with a scale per 32 weights, the seven kinds cost a KL of 0.0085 to 0.0274, a factor of 3.24. In the first record's arm S the factor was 40.7. Predictions A1 and A2 fail: of the two kinds that write into the stream, `attn_output` is first here and `ffn_down` third, by narrow margins.
2. **The gap of the first record came with its arm, and this record does not say which part of the arm.** Arm S has one scale per matrix, searched for least squared error, which clips. The rule here has a scale per 32 weights, set from the largest entry, which clips nothing. Per weight `ffn_down` costs 1.67 times `ffn_gate` here and 25.3 times under arm S. The size of the group and the clipping changed together.
3. **Per weight, the small attention matrices cost most.** `attn_v` costs 31.8 times `ffn_gate` per weight, and `attn_k` is second. They are 3.26 percent of the weights.
4. **In this range the parts add, and a bit is worth a factor between 2.6 and 4.9.** The seven KLs add up to 1.158 times the KL of all kinds at once. One more bit divides the KL of a kind by 2.62 to 4.94 in all 14 steps, 3.65 in the geometric mean.
5. **Bits set by equal gain beat equal bits.** At 4.5 bits the split scores 0.0805 against 0.1173, 31.4 percent lower. At 3.5 bits it scores 0.3164 against 0.4959, 36.2 percent lower. The measured KLs came in 12 and 13 percent under the program's forecasts. The split is fitted on the scored text.
6. **The control arm did not control.** The arm meant to place the same levels the wrong way round also beat equal bits, by 19.0 and 14.4 percent, where the best arm gained 31.4 and 36.2. Prediction B4 fails. The rule that built it left it much of the gain: see Part B.
7. **Keeping 0.17 percent of the weights exact more than halves the KL of a row scale.** 0.2071 against 0.4489 at 16 levels and 0.0520 against 0.1180 at 32, for 0.049 bit. By the slope of the plain rule, the same bits spent on levels would give about 0.42 and 0.11. No arm with those levels was scored. With a scale per 32 weights the gain is 20.6 percent.
8. **A scale does not need 16 bits.** In 8 bits it costs 2.1 percent of KL and saves 0.245 bit. In 6 bits it costs 8.3 percent and saves 0.308. With the saved bits spent on levels, 20 levels and a 6-bit scale score 0.0815 at 4.514 bits, 30.6 percent under the `Q4_0` rule and 0.014 bit over it.
9. **Joined, the parts give 0.0629 at 4.484 bits**, 46.4 percent under the `Q4_0` rule and 0.016 bit under it. That is under llama.cpp's `Q4_K_M` at 4.80 bits (0.0689) and under the first record's E8 arm at 4.5 bits (0.0747). Neither comparison is on equal terms: see Limits.
10. **The damage of a clip adds over layers, and one layer carries nearly all of it.** Six layers clipped at once score 0.978 of the six alone, added up, where E1 expected 1.5 or more. Clipped alone, layer 10's `ffn_down` scores 0.1560. The other 41 layers add up to 0.0182. The damage that the first record's clip found in `ffn_down` is, for this clip, the damage of one matrix.
11. **None of three figures of its large entries picks that matrix out, and its ten units with the most change do not carry the damage.** It has 6.2 percent of the model's large entries of this kind. It is fourth of 42 layers by their number and by the size of the clip's change, and nineteenth by its largest entry. The ten hidden units with the most change carry 0.05 percent of its damage. Predictions M1 to M3 fail.
12. **That one matrix is a third to a half of the gap between two scale rules at 32 levels.** A rule that clips, `R+S:32`, scores 0.3395, and one that does not, `XR:32`, scores 0.1180. Swapping the rule on this one matrix, 0.66 percent of the weights, closes 33.2 percent of the gap one way and opens 55.0 percent of it the other way. M4 expected the first to be over half and fails. M5 expected the second to be and holds.
13. **A weight that nothing reads needs no bits.** With the 55 million unread key and value weights at 0 bits, the joined arm scores the same 0.062882 at 4.397 bits.
14. **None of the three measures is a certificate, and the one built on persistence is the weakest on new arms.** On 16 arms built after the measures were fixed it ranks at 0.462, where squared error ranks at 0.891 and the spectral error at 0.900. Those 16 are all of one family, with a scale set from the largest entry.

## How it was measured

| | |
|---|---|
| Model | `gemma-4-E4B-it-Q8_0.gguf`, SHA-256 in the plan. 294 block matrices, 3,945,267,200 weights. Nothing else is changed. |
| Text and score | As in the first record: 40 passages of 512 tokens built from the llama.cpp repository at commit `427291b`, 10,200 scored positions, `llama-perplexity --kl-divergence` against the reference's saved outputs. "KL" is the mean KL divergence. |
| First step | The reference outputs were rebuilt. `unchanged`, `X32:16` and `XR:16` score 0.000000, 0.117297 and 0.448857, against 0.000000, 0.117297 and 0.448262 in the first record. All three are inside the plan's limits. |
| Bits | log2 of the levels per weight, plus the bits of each stored scale, plus the bits of each kept entry. Over all 3.95 billion block weights. |
| Standard errors | The tool's. They treat positions as independent and are too small. |

The arms, all built on the rule of llama.cpp's `Q4_0`, which sets a scale from the largest entry of a group and clips nothing:

| Name | What |
|---|---|
| `X32:Q`, `XR:Q` | Q levels, one 16-bit scale per 32 weights or per row |
| `X32:Q@KIND` | the same on one kind of matrix, the rest unchanged |
| `X32mix` | each kind of matrix at its own number of levels |
| `XRo:Q~L`, `X32o:Q~L` | entries beyond L standard deviations of their row are kept exact, and the scale is set from the largest entry that is left |
| `X32sB:Q` | the scale of each group stored in B bits, rounded up on a grid, with the row's largest entry in 16 bits |
| `X32s8mix` | the joined arm: `X32mix` with 8-bit scales |
| `clip:5@ffn_down#L` | no rounding. In layer L, entries of `ffn_down` beyond 5 standard deviations of their row are pulled back to that limit |

Every figure below is in [results.out](results.out), written by `summary.py`, or in [detail.out](detail.out), written by `detail.py`.

## Part A: one kind at a time

KL of each kind rounded alone, with the rest of the model unchanged.

| Kind | Weights | 8 levels | 16 levels | 32 levels | Per weight at 16 levels, times 1e9 | The same under arm S of the first record |
|---|---|---|---|---|---|---|
| `attn_v` | 64,225,280 | 0.1058 | 0.0264 | 0.0073 | 0.4105 | 1.0438 |
| `attn_k` | 64,225,280 | 0.0295 | 0.0085 | 0.0029 | 0.1318 | 0.7613 |
| `attn_output` | 256,901,120 | 0.0924 | 0.0274 | 0.0062 | 0.1067 | 3.7460 |
| `attn_q` | 256,901,120 | 0.0505 | 0.0167 | 0.0034 | 0.0652 | 0.2241 |
| `ffn_down` | 1,101,004,800 | 0.0950 | 0.0237 | 0.0064 | 0.0216 | 1.8064 |
| `ffn_up` | 1,101,004,800 | 0.0746 | 0.0190 | 0.0051 | 0.0172 | 0.1558 |
| `ffn_gate` | 1,101,004,800 | 0.0563 | 0.0142 | 0.0054 | 0.0129 | 0.0714 |
| all kinds at once | 3,945,267,200 | 0.4959 | 0.1173 | 0.0313, first record | | |

The first record asked why the two writing kinds were so sensitive, and left it open. Under this rule they are not. Its figures were for arm S: one scale for a whole matrix, searched for least squared error, which clips the largest entries. The rule here differs in both ways at once. Which of the two made the gap is not separated here. Parts H and M show that one `ffn_down` matrix is hurt badly by a clip.

The key and value matrices are counted whole here. 55,050,240 of their 128,450,560 weights, 42.9 percent, are never read by llama.cpp's forward pass. So the weights that are read cost more per weight than the table says. Part K takes this up.

## Part B: bits by equal gain

A design is at its best when one more unit of resource buys the same gain everywhere. `allocate.py` tries all 9^7 choices of levels and takes the one with the least forecast KL inside the budget. The forecast is the sum over kinds of a line through each kind's three points of Part A.

| Arm | Levels: q, k, v, output, gate, up, down | Bits | Forecast | KL | Against equal bits |
|---|---|---|---|---|---|
| equal bits, `X32:16` | 16 for all | 4.500 | | 0.1173 | |
| best at 4.5 bits | 32, 64, 64, 46, 12, 12, 16 | 4.498 | 0.0916 | 0.0805 | 31.4 percent lower |
| reversed at 4.5 bits | 46, 64, 64, 32, 16, 12, 12 | 4.498 | 0.1006 | 0.0950 | 19.0 percent lower |
| equal bits, `X32:8` | 8 for all | 3.500 | | 0.4959 | |
| best at 3.5 bits | 8, 16, 32, 16, 6, 8, 8 | 3.498 | 0.3630 | 0.3164 | 36.2 percent lower |
| reversed at 3.5 bits | 16, 32, 16, 8, 8, 8, 6 | 3.498 | 0.4382 | 0.4243 | 14.4 percent lower |

The best arm takes a little from the two largest kinds and gives a lot to the small ones. Dropping `ffn_gate` and `ffn_up` from 16 levels to 12 saves 0.2316 bit over all weights. Raising `attn_k` and `attn_v` from 16 levels to 64 costs 0.0651.

Turned into bits with the slope of `X32` between 4.5 and 5.5 bits, the gain at 4.5 bits is worth 0.285 bit.

**Why the control failed.** The plan built the reversed arm from the same levels as the best arm, in the order with the worst forecast, inside the same budget. The kinds differ so much in size that every order inside the budget still gives the small kinds many levels.

Its forecast was already under that of equal bits, 0.1006 against a sum of 0.1359 for 16 levels everywhere. The plan did not foresee that. So B4 tests less than it was meant to. The reversed arm kept 19.0 of the best arm's 31.4 percent at 4.5 bits, and 14.4 of its 36.2 at 3.5. The best arm is under the reversed one, at 0.847 and 0.746 of its KL.

The 64 levels of `attn_k` and `attn_v` lie beyond the 32 levels that Part A measured. The forecast there follows the end slope, as the plan says.

llama.cpp's `Q4_K_M` splits bits by a fixed rule. At commit `427291b`, `src/llama-quant.cpp` gives `attn_v` and `ffn_down` a 6-bit type in the first eighth of the layers, the last eighth and every third layer between, and a 4-bit type elsewhere. The split found here gives most to `attn_v` and `attn_k` and nothing extra to `ffn_down`.

## Part C: the largest entries kept apart

| Arm | Entries kept | Bits | KL | Against the plain arm | The same bits spent on levels |
|---|---|---|---|---|---|
| `XR:16` | none | 4.005 | 0.4489 | | |
| `XRo:16~8` | 0.0014 percent | 4.005 | 0.4190 | 0.933 | about 0.4486 |
| `XRo:16~5` | 0.0150 percent | 4.009 | 0.3405 | 0.759 | about 0.4463 |
| `XRo:16~3.45` | 0.1732 percent | 4.054 | 0.2071 | 0.462 | about 0.4205 |
| `XRo:16~2.5` | 1.5387 percent | 4.444 | 0.1063 | 0.237 | about 0.2512 |
| `XR:32`, first record | none | 5.005 | 0.1180 | | |
| `XRo:32~8` | 0.0014 percent | 5.005 | 0.1068 | 0.905 | about 0.1180 |
| `XRo:32~5` | 0.0150 percent | 5.009 | 0.0843 | 0.714 | about 0.1174 |
| `XRo:32~3.45` | 0.1732 percent | 5.054 | 0.0520 | 0.440 | about 0.1106 |
| `X32:16` | none | 4.500 | 0.1173 | | |
| `X32o:16~3.45` | 0.1732 percent | 4.549 | 0.0931 | 0.794 | |

The last column is an estimate: the plain arm's KL divided by 3.752 for each bit.

By that estimate a kept entry buys far more than the same bits spent on levels. No arm with more levels at those bits was scored. The rule that clips nothing pays for each large entry with a coarse step for its whole row. Kept apart, the entry costs 28 to 30 bits once, and the row gets a finer step.

Prediction C3 holds as written: at a limit of 3.45 a row scale with kept entries stays above a scale per 32 weights, 0.2071 against 0.1173. At a limit of 2.5 it does not. `XRo:16~2.5` scores 0.1063 at 4.444 bits, under `X32:16` at 4.5 bits. So with 1.5 percent of the weights kept, a row scale does the work of a scale per 32 weights at the same bits.

Keeping large weights apart is published work (SpQR and others). The clip probes of the first record showed what losing them costs. This part shows what keeping them buys under this rule.

## Part D: the bits of a scale

| Arm | Bits | KL | Against `X32:16` |
|---|---|---|---|
| `X32:16`, 16-bit scale | 4.500 | 0.1173 | |
| `X32s8:16` | 4.255 | 0.1198 | 1.021 |
| `X32s6:16` | 4.192 | 0.1271 | 1.083 |
| `X32s8:18` | 4.425 | 0.0972 | 0.829 |
| `X32s6:20` | 4.514 | 0.0815 | 0.694 |

A scale per 32 weights in 16 bits is 0.5 bit per weight. A 6-bit scale saves 0.308 of that, at a cost of 8.3 percent of KL. llama.cpp's K formats already store scales in 6 bits. The Qwen2.5 record in pull request 12 scores an arm with an 8-bit scale on another model.

## The joined arm

The rule of Part B, run again with 8-bit scales, gives 32, 64, 64, 46, 16, 16 and 16 levels at 4.484 bits and a forecast of 0.0661. It was committed in `allocate_joined.out` before the arm was scored.

| | Bits | KL | Top token kept |
|---|---|---|---|
| The `Q4_0` rule, `X32:16` | 4.500 | 0.1173 | 89.41 % |
| The first record's E8 arm, `VE832:16` | 4.500 | 0.0747 | 91.57 % |
| llama.cpp `Q4_K_M`, first record | 4.803 | 0.0689 | 91.79 % |
| The joined arm | 4.484 | 0.0629 | 92.02 % |

Against the `Q4_0` rule at equal bits the joined arm is 46.4 percent lower, worth 0.471 bit by the same slope. Prediction G1 asked for 0.085 or less.

## Part E: does the damage add up over layers

Six layers of `ffn_down`: 0, 8, 16, 24, 32 and 40.

| | The six alone, added up | All six at once | Ratio |
|---|---|---|---|
| Clip at 5 standard deviations | 0.001673 | 0.001637 | 0.978 |
| Rounding, `X32:16` | 0.005700 | 0.005033 | 0.883 |

The plan guessed that a clip, which moves every entry the same way, would add up in step over layers and give at least 1.5. It gives 0.978. One layer of the clip and two of the rounding have a KL under five times its standard error, so the plan's rule marks both predictions as not decided. The clip's ratio is far from what was predicted all the same.

The six layers are a seventh of the model and carry a hundredth of the clip's damage: the clip on all 42 layers scored 0.1638 in the first record, 97.9 times the six. For rounding the same ratio is 4.16. Extension 1 was written to find out why.

## Part H: the clip on every layer alone

Extension 1. The same clip on each of the other 36 layers alone.

| | KL |
|---|---|
| Layer 10 alone | 0.155993 |
| Layer 21 alone, the next largest | 0.001990 |
| The other 41 layers alone, added up | 0.018163 |
| All 42 alone, added up | 0.174156 |
| All 42 at once, first record | 0.163763 |

The damage adds over all 42 layers too: the ratio is 1.063. Layer 10 holds 89.6 percent of the sum and is 78.4 times the next layer. Seven layers, all among the first fifteen, have a KL under five times its standard error. Together they hold 2.9 percent of the sum, so by the extension's rule the verdicts stand.

The size of the clip's change does not say which layer it is. Layers 20, 21 and 16 are changed more by their clip than layer 10 is, and score 0.0003, 0.0020 and 0.0006. Over all 42 layers the rank correlation between a layer's KL and the size of its change is 0.753, so H3 holds. Without layer 10 it is 0.740. It orders the small layers and says nothing of the one that matters.

## Part K: a closure in the budget

Extension 1. The joined arm with the unread key and value matrices of the last 18 layers zeroed and counted at 0 bits.

| | Bits | KL |
|---|---|---|
| The joined arm | 4.4844 | 0.062882 |
| The same, 55,050,240 unread weights at 0 bits | 4.3971 | 0.062882 |

The score is the same in every printed figure, and 0.0873 bit is saved. The certificate is the first record's: the forward pass never reads those matrices. This run repeats it on the binary named under "How to rerun".

## Part L: is layer 10 marked by its weights

Extension 2, weights only. For the `ffn_down` matrix of each layer, in [large_by_layer.out](large_by_layer.out):

| | Layer 10 | Its place among 42 | The three largest |
|---|---|---|---|
| Entries beyond 5 standard deviations of their row | 11,693 | 4th | layers 21, 20 and 22 |
| Largest entry, in standard deviations of its row | 18.52 | 19th | layers 11, 29 and 12 |
| The clip's squared change over the squared size of the matrix | 0.001817 | 4th | layers 20, 21 and 16 |

The 42 matrices hold 188,362 such entries, the first record's count. Layer 10 has 6.2 percent of them and 89.6 percent of the damage. None of the three figures picks it out. Other figures of the weights were not tried.

Inside layer 10 the change is spread out. It touches 2,427 of 10,240 hidden units and 1,184 of 2,560 dimensions of the stream. Half of it lies in 102 hidden units, or in 51 dimensions. The top hidden unit holds 1.3 percent.

## Part M: parts of layer 10, and one matrix under the other rule

Extension 2. The first four rows clip only some of the entries that the clip of layer 10 touches.

| Variant | KL | Share of layer 10's 0.155993 |
|---|---|---|
| The top hidden unit | 0.000010 | 0.01 percent |
| The top ten hidden units | 0.000071 | 0.05 percent |
| All but the top ten hidden units | 0.155882 | 99.93 percent |
| The top ten dimensions of the stream | 0.001060 | 0.68 percent |

The plan guessed that the hidden units with the most change carry the damage. They do not. The ten units with the most change hold 9.8 percent of the change and 0.05 percent of the damage. Units were ranked by the size of their change, not by their damage, and the units after the tenth were not clipped apart. So where in the other 2,417 units the damage sits is not known.

| Rule on 293 matrices | Rule on layer 10's `ffn_down` | KL |
|---|---|---|
| `R+S:32`, clips | `R+S:32` | 0.3395, first record |
| `R+S:32` | `XR:32` | 0.265947 |
| `XR:32`, clips nothing | `R+S:32` | 0.239950 |
| `XR:32` | `XR:32` | 0.1180, first record |

The gap between the two rules is 0.2215. Freeing this one matrix from the clipping rule closes 33.2 percent of it. Putting this one matrix under the clipping rule opens 55.0 percent of it. The two do not add to the whole, and M4 fails: most of the gap is not this matrix when the rest of the model is already clipped.

So of the gap between these two rules, a third to a half is one matrix of 294. What carries the rest was not measured. The first record's finding that the output followed how the scale is set rests on more arms than these two, arm S among them, and this result is about these two only.

**A reading, not tested here.** Published work reports a few activations in language models that are far larger than the rest, stay nearly the same for every input and act as fixed offsets (Sun and others, 2024).

A matrix that writes such an offset would be hurt by a change that moves many of its entries the same way, and little by noise. That fits the clip against the random move of the first record. One result here sits badly with it: the ten dimensions of the stream with the most change carry 0.68 percent of the damage. Nothing here measures an activation.

## Part F: three more measures of a weight change

Each is a mean over the 42 matrices of six layers, in dB. Rank correlation with KL:

| Measure | The 26 variants of the first record | 16 new arms of Parts B, C and D |
|---|---|---|
| Along the rows | 0.770 | 0.812 |
| Along the columns | 0.675 | 0.806 |
| Tree shift, built on persistence | 0.770 | 0.462 |
| Squared error | 0.603 | 0.891 |
| Spectral error | 0.889 | 0.900 |

The 26 were known when the measures were chosen. The 16 were not built yet.

- **Along the rows** is the part of each row's change that lies along the row. It misses the mark of 0.8 that F1 set. It has the clip at 3.45 as worse than `XR:32`, and the same entries moved at random as better than the clip, both as they are. It has the clip at 5 as better than `XR:32`, which it is not. That failure was predicted.
- **Tree shift** has both clips as worse than `XR:32`, as they are. It has `XR:32` with the step times 0.7 as far worse, when it is better. Both were predicted. On the new arms it is the weakest of the five.
- **Squared error** ranks the 26 at 0.603 and the 16 new arms at 0.891. The 16 are all of one family. Why it does better there was not tested, and two of the 16 have the same squared error and KLs of 0.0805 and 0.0950.

The weights alone, by kind, mean over layers:

| Kind | Persistence, normalised | Spread of the rows | Spread of the columns |
|---|---|---|---|
| `attn_q` | 0.6195 | 0.2614 | 0.1430 |
| `attn_k` | 0.3628 | 0.1754 | 0.1262 |
| `attn_v` | 0.3290 | 0.0353 | 0.1373 |
| `attn_output` | 0.6869 | 0.1617 | 0.1419 |
| `ffn_gate` | 0.7085 | 0.1964 | 0.1061 |
| `ffn_up` | 0.6805 | 0.1123 | 0.1020 |
| `ffn_down` | 0.7882 | 0.1196 | 0.1124 |

Over the seven kinds the rank correlation between persistence and KL per weight is -0.750. That is the edge of the range F7 named, and the plan's wording counts it as inside. Seven points cannot carry more. F8 fails: the writing kinds do not have the larger spread of rows.

## Predictions

| | Prediction | Measured | Verdict |
|---|---|---|---|
| A1 | `ffn_down` has the largest KL of the seven kinds at each number of levels | the largest are `attn_v`, `attn_output`, `attn_v` | fails |
| A2 | Per weight, `ffn_down` and `attn_output` are the two most sensitive at 16 levels | they are `attn_v` and `attn_k` | fails |
| A3 | One more bit divides a kind's KL by 2.5 to 6 in at least 12 of 14 steps | 14 of 14 | holds |
| A4 | The seven KLs add up to 0.75 to 1.25 times the KL of all kinds at once | 1.158 | holds |
| B1 | The best arm at 4.5 bits scores 0.0997 or less | 0.0805 | holds |
| B2 | The best arm at 3.5 bits is at least 15 percent under `X32:8` | 0.3164 against 0.4215 | holds |
| B3 | Each best arm scores 0.7 to 1.3 times its forecast | 0.879 and 0.872 | holds |
| B4 | Each reversed arm is at least 15 percent above equal bits | 0.810 and 0.856 of it | fails |
| C1 | `XRo:16~3.45` scores 0.336 or less for under 0.06 bit more | 0.2071, 0.0494 bit | holds |
| C2 | `XRo:32~3.45` scores 0.0885 or less | 0.0520 | holds |
| C3 | `XRo:16~3.45` stays above `X32:16` | 0.2071 against 0.1173 | holds |
| C4 | At a limit of 8 nothing changes by more than 10 percent | 0.933 and 0.905 | holds |
| C5 | `X32o:16~3.45` is 5 to 30 percent under `X32:16` | 20.6 percent | holds |
| D1 | `X32s8:16` is within 10 percent of `X32:16` | 1.021 | holds |
| D2 | `X32s6:16` is at most 1.2 times `X32:16` | 1.083 | holds |
| D3 | `X32s8:18` is at most 0.90 of `X32:16` and `X32s6:20` at most 0.80 | 0.829 and 0.694 | holds |
| E1 | The clip on six layers at once scores at least 1.5 times the six alone | 0.978 | not decided |
| E2 | Rounding on six layers at once scores 0.75 to 1.25 times the six alone | 0.883 | not decided |
| F1 | Along the rows ranks the 26 at 0.8 or more, above squared error | 0.770, above 0.603 | fails |
| F2 | Along the rows has the clip at 3.45 as worse than `XR:32` | -50.74 against -56.69 dB | holds |
| F3 | Along the rows has the clip at 5 as better than `XR:32`, when it is worse | -59.71 against -56.69 dB | holds |
| F4 | Along the rows has the random move as better than the clip at 3.45 | -57.18 against -50.74 dB | holds |
| F5 | Tree shift has both clips as worse than `XR:32`, and the step times 0.7 as worse too | -26.71, -38.02 and -14.72 against -44.75 dB | holds |
| F6 | Tree shift ranks the 26 worse than the spectral error | 0.770 against 0.889 | holds |
| F7 | Persistence does not say which kinds are sensitive: rank correlation between -0.75 and 0.75 | -0.750 | holds, on the edge |
| F8 | The writing kinds have a larger spread of rows than each reading kind | 0.1196 against 0.2614 for `attn_q` | fails |
| G1 | The joined arm scores 0.085 or less at 4.5 bits or under | 0.0629 at 4.484 bits | holds |
| H1 | The 42 layers alone add up to 0.75 to 1.25 times the clip on all at once | 1.063 | holds |
| H2 | The four largest layers make more than half of the sum | 92.3 percent | holds |
| H3 | A layer's KL goes with the size of its clip's change: rank correlation at least 0.5 | 0.753 | holds |
| K1 | The joined arm with unread weights at 0 bits scores the same to four decimals, under 4.41 bits | 0.062882 at 4.3971 bits | holds |
| M1 | The top ten hidden units of layer 10 carry at least 0.8 of its damage: 0.1248 or more | 0.0001 | fails |
| M2 | The rest carries at most 0.2 of it: 0.0312 or less | 0.1559 | fails |
| M3 | The top hidden unit carries at least half: 0.0780 or more | 0.0000 | fails |
| M4 | `R+S:32` with this matrix under `XR:32` scores 0.2288 or less | 0.2659 | fails |
| M5 | `XR:32` with this matrix under `R+S:32` scores 0.2288 or more | 0.2399 | holds |

F3 and the second half of F5 predict that a measure fails. They hold when it does.

## Deviations from the plan

[DEVIATIONS.md](DEVIATIONS.md) has the full list. The ones that bear on a verdict:

- Part A ran with the first step checked by eye. The check in the program came with commit `2b543c5`, before Part B.
- C4 compares `XRo:32~8` with the `XR:32` of the first record, 0.1180, which was not scored again here.
- The scored draw of the random move in the first record differs from the draw `measures.py` measures. That limits F4.
- A first run of `measures.py` was started while Part A was scoring. It was stopped after one variant and its output deleted. Part F was run again from the start.
- The commit message of the plan says 29 predictions. The plan has 27.
- `detail.py` was written after the results, to print the ratios this record quotes. It scores nothing.
- Programs changed after the first data: `run.py` and `allocate.py` at `2b543c5` for the two checks, `run.py`, `summary.py` and the tests at `e4c70eb` and `566fbbe` for the extensions, and `run.py` at `39ce552` for the fault named below. No arm that had been scored was changed.
- In extension 2 the first launch of Part M stopped on a fault in a check, after its first four variants and before the last two were built. The fix was committed and the part started again. Nothing was scored twice.

## Limits

- **One model, one text, one runtime.** Every variant is scored on the same 10,200 positions.
- **The split of bits is fitted where it is scored.** Part A measures each kind's KL on the text that Part B and the joined arm are then scored on. A split fitted on one text may not hold on another. Nothing here tests that. The other arms are not fitted.
- **The comparison with `Q4_K_M` is not on equal terms.** The joined arm is fitted to this text. The `Q4_K_M` file was made with no importance matrix, which is how the first record ran it and not how it is used at its best.
- **The comparison with the E8 arm is to a code with a known fault.** The Qwen2.5 record in pull request 12 found that the harness's E8 Voronoi code has no codeword at zero, and that this costs it on the output. Every E8 arm of the first gemma record carries that. An E8 arm without it was not scored on gemma.
- **No arm here uses a ball.** The same Qwen2.5 record scores codes cut by one ball per group far under the `Q4_0` rule at equal bits. Whether the parts of this record add to those arms is not known.
- **No packed format.** The bits of kept entries and of short scales are counted and not stored. No decoding speed is measured.
- **Standard errors are too small.** Differences of a few percent between two arms cannot be ordered. C4 at 32 levels, D1 and F7 are near their limits.
- **The measures of Part F were chosen after the KLs of the 26 variants were known.** Only the 16 new arms are free of that, and they are all of one family.
- **Part E's verdicts are not decided by the plan's own rule.** Part H repeats the question on 42 layers, where the rule lets the verdict stand.
- **The extensions were written with results in view.** H3 and M1 to M5 were written after the layers they concern had been seen in part.
- **Parts H, L and M are about one kind of matrix, one limit of the clip and one model.** Why layer 10 is the one is not known. The reading given in Part M was not tested. Part M compares two rules at 32 levels and no others.
- **Two things differ between arm S of the first record and the rule of Part A**: the size of the group that shares a scale, and whether the rule clips. No arm here changes one without the other.

## Read against the closure rule

- **A weight that no path reads needs no bits.** The first record closed 55,050,240 key and value weights by certificate. Part A and the split of bits counted them all the same. Part K applies the closure to the budget and saves 0.0873 bit at the same score.
- **One guess of this record's own plan is ended by a counterexample.** That the hidden units of layer 10 with the most change carry its damage: the ten with the most change carry 0.05 percent of it.
- **No certificate for damage came of the three measures.** Each has an exhibited case it gets wrong. The path "a measure built on persistence orders damage" is not closed by proof. It is not shown, and on the new arms it is the weakest tried.
- **One reading of the first record is ended by a counterexample.** That the writing kinds are the sensitive ones held under arm S. Under the rule that clips nothing, with a scale per 32 weights, it does not hold on the same model and text.

## What follows

- The split of bits fitted on one text and scored on another. The text and model of the Qwen2.5 record would do.
- The ball arms of the Qwen2.5 record on gemma, alone and with bits split by kind. The gemma file is on the machine that ran this record.
- The same split with the weights that are read counted apart from those that are not.
- What makes layer 10's `ffn_down` the one. The first thing to measure is what its large entries multiply: the activations of its hidden units, which llama.cpp's importance matrix tool gives.
- A split of bits by layer as well as by kind, now that one matrix is known to matter far more than its size.
- llama.cpp's files made with an importance matrix, as issue #14 plans for another model.

## Sources

- The first model-output record and its `large_entries.out`, in this repository. Read by the session that wrote this record.
- The Qwen2.5 record, `experiments/model-output-qwen25-2026-11-17` on the branch of pull request 12, at commit `06e3e23`. Read there. It is marked partial.
- llama.cpp at commit `427291b`, `src/llama-quant.cpp`: the rule of `Q4_K_M`. Read. Its pull request 1684: scales of the K formats in 6 bits. Reported by an agent that opened it.
- Dettmers and others, "SpQR", arXiv 2306.03078: large weights kept at higher precision. Reported.
- Sun, Chen, Kolter, Liu, "Massive Activations in Large Language Models", arXiv 2402.17762. Seen as a search summary only.

## How to rerun

Needs NumPy, torch, llama.cpp's `gguf` Python package, `llama-perplexity`, and the model file. Each program refuses to run unless its plan is committed and pushed.

| | |
|---|---|
| Machine | An M1 Max, torch on MPS |
| Python, NumPy, torch | 3.9.6, 2.0.2, 2.8.0 |
| Text and `gguf` package | A checkout of llama.cpp at commit `427291b` |
| Scoring | The installed `llama-perplexity`, which reports version 0.5.0, build 11146, commit `7fe450e19` |
| Seeds | The searched step of `R+S` samples blocks with seed 0, as in the first record. The power iteration of the spectral error uses seed 0. Nothing else is random. |
| Time | A variant scores in 30 to 43 seconds and is built in under a second to 40. Part A, 22 variants, took 17 minutes. |

The first record names commit `427291b` only. Its three scores repeated here, two of them to six decimals, were made with the binary above.

Below, `E` stands for `experiments/output-budget-gemma-e4b-2026-11-17` and `RUN` for the command of the third block.

```bash
python3 harness/build_eval_text.py ~/src/llama.cpp eval.txt
```

```bash
llama-perplexity -m gemma-4-E4B-it-Q8_0.gguf -f eval.txt -c 512 -b 512 --chunks 40 -ngl 999 --kl-divergence-base base.kld
```

```bash
python3 E/run.py --model gemma-4-E4B-it-Q8_0.gguf --kld base.kld --work work --out E/results.jsonl --part first
```

Then, in this order, with each output committed and pushed before the part that reads it:

| Step | Command |
|---|---|
| Part A | `RUN` with `--part A` |
| The split of bits | `python3 E/allocate.py --results E/results.jsonl > E/allocate.out` |
| Part B | `RUN` with `--part B --allocate-out E/allocate.out` |
| Parts C, D, E | `RUN` with `--part C`, then `D`, then `E` |
| The joined arm's levels | `python3 E/allocate.py --results E/results.jsonl --budgets 4.5 --scale-bits 8 > E/allocate_joined.out` |
| The joined arm | `RUN` with `--part G --joined-out E/allocate_joined.out` |
| Parts H and K | `RUN` with `--part H`, then `K` |
| Part L | `python3 E/large_by_layer.py --model gemma-4-E4B-it-Q8_0.gguf` |
| Part M | `RUN` with `--part M` |
| Part F, the 26 variants | `python3 E/measures.py --model gemma-4-E4B-it-Q8_0.gguf --out E/measures.jsonl` |
| Part F, the 16 new arms | the same with `--held-out-from E/results.jsonl` |
| Part F, the weights alone | `python3 E/measures.py --model gemma-4-E4B-it-Q8_0.gguf --weights-only E/weights_only.jsonl` |

```bash
python3 E/summary.py > E/results.out
```

```bash
python3 E/detail.py > E/detail.out
```

The tests need no model:

```bash
python3 E/test_programs.py
```

## Disclosure

The plan, the programs and this record were written by AI agent sessions under the owner's direction. One agent wrote the plan and this record, a second wrote the programs and ran them, and a third read the programs against the plan before most of the runs.
