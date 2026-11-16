# Rounding a real model: what its output follows

This record rounds the weights of a real language model and measures how far its output moves. The model is `gemma-4-E4B-it`, 7.5 billion weights, in an 8-bit GGUF file. Each variant is scored by llama.cpp's own tool against the unrounded model on the same 10,200 positions of text.

The first record in this repo measured the error of rounded weights and nothing else. This one asks whether that error says anything about the output.

Status: `exploratory`. The plan, [PREREG.md](PREREG.md), was committed before the first run and extended five times as results came in. Each extension lists results that were known and was committed before its own variants were scored or its measure computed. 10 of 18 predictions hold.

Findings 2 and 3 below were found along the way. The largest-entry rule and the clip probes were not in the first plan, and only parts of them had a prediction written first.

## Result in short

1. **Weight error does not say what the output does.** Pulling back the 0.17 percent of weights that lie furthest out for their row, with no rounding, changes the weights by -27.5 dB in total and gives a mean KL of 0.318. Rounding every weight to 5 bits with nothing clipped changes them by -21.7 dB, nearly four times as much, and gives 0.118.
2. **In these runs the output followed how the scale is set.** The plain arms of the first record use one scale per matrix, found by least squared error. At 5 bits per weight they keep the model's top token on 62 percent of positions, and on 83 to 86 percent with rows and columns balanced. One scale per row, set from the row's largest entry, keeps it on 90 percent at the same bits.
3. **With that rule the E8 lattice gave a closer output than the scalar rule at equal bits, in all three pairs tested.** At 4.5 bits per weight the mean KL is 0.075 against 0.117 for llama.cpp's `Q4_0`. At 5.005 bits it is 0.059 against 0.118. With one scale per matrix and no balancing the lattice is worse than scalar rounding at 4 and 5 bits.
4. **Two closures score exactly as the unchanged model, and a third within noise.** 2.79 billion weights can be zeroed for this text, and 55 million for any text, with every score unchanged. Moving the weights 1.6 times their own length along exact symmetries gives a mean KL of 0.00005 and keeps the top token on 99.93 percent of positions.
5. **No formula tried here predicts the damage.** Five were tested. The best, the largest singular value of the weight change, has a rank correlation of 0.889 with mean KL over 26 variants, against 0.603 for squared error. It still has the clip probes as better than a rounding that does less damage.

## How it was measured

| | |
|---|---|
| Model | `gemma-4-E4B-it-Q8_0.gguf`, SHA-256 in the plan. 42 blocks, 294 block matrices holding 3.95 billion weights, token tables holding 3.49 billion. |
| What is rounded | The 294 block matrices. Token tables and small tensors stay as they are. |
| Text | Built by `harness/build_eval_text.py` from the llama.cpp repository at commit `427291b`: slices of documentation alternating with slices of source code. 40 passages of 512 tokens, the second half of each scored. |
| Score | `llama-perplexity --kl-divergence`: mean KL divergence from the reference model's next-token distribution, and the share of positions where the top token stays the same. |
| Bits | Counted from the codebook: log2 of the levels per weight, plus 16 bits for each scale. Rounded values are stored as 16-bit floats so that llama.cpp reads what the rounding produced. For llama.cpp's own quantisers the bits are those of the file. |
| Weight error | Squared change over squared size. "Mean" is the mean over matrices of that ratio in dB. "Pooled" sums both over all matrices first. The two differ by under 1 dB for this repo's rounding variants, by up to 2 dB for llama.cpp's quantisers, and by 7 to 14 dB for the clip probes on all matrices. |

Checks of the harness:

- The unchanged file scores a mean KL of 0.000000 against its own saved outputs. The largest KL at any position is 0.000058.
- A copy with the block matrices stored as 16-bit floats scores the same: every printed statistic is equal.
- `X32` at q = 16 is this repo's own code for llama.cpp's `Q4_0` rule. On three matrices it gives the same value as llama.cpp's file on 95 to 96 percent of entries, and the rest differ by at most 0.0004 of the largest weight, one step of a 16-bit float. The scores are the same: mean KL 0.1173 for both, top token kept on 89.41 against 89.40 percent.
- `XR` at q = 32 was scored a second time with nothing else running. Every number was equal. This shows the scores do not depend on load. It is not a repeat of the experiment.

All numbers below are in [results.out](results.out), [checks.out](checks.out) or [large_entries.out](large_entries.out).

## Closures

| Change | Weights | Mean KL | Top token kept |
|---|---|---|---|
| K1. Rows of `per_layer_token_embd` zeroed for the 259,937 tokens not in the text | 2,794,842,624 (37.2 % of the model) | 0.000000 | 99.99 % |
| K2. The same rows of `token_embd` zeroed | 665,438,720 | 1.364 | 95.51 % |
| Keys and values of the last 18 layers zeroed | 55,050,240 | 0.000000 | 99.99 % |
| Random move along three exact symmetries, 4.09 dB larger than the weights moved | 3,633,315,840 moved | 0.000047 | 99.93 % |

The first and third rows score as the unchanged model does on every printed digit.

- **K1.** The table is read by row, one row per token. A row whose token never occurs is on no path from input to output. This closure holds only for inputs made of the 2,207 tokens of the text.
- **K2.** The main token table is read by row too, but the same weights are the output layer. Its rows stay on a path. A shared weight can be closed only if every use of it is off the path.
- **Keys and values.** The last 18 of 42 layers reuse the keys and values of an earlier layer. Their own `attn_k` and `attn_v` are in the file and llama.cpp never reads them. This closure holds for every input.
- **Symmetries.** Three scalings leave the forward pass unchanged: a hidden unit's row of `ffn_up` against its column of `ffn_down`, and an input dimension's columns of the reading matrices against its entry of the gain vector before them. `run.py` documents them in `gauge_layer`.

The move along the symmetries is exact in the forward pass and not in the scores. The copy is stored in 16-bit floats, and one position of 10,200 has a KL of 0.43. Whether the float format causes that was not checked.

## The first grid: one scale per matrix

The arms of the first record: scalar rounding (S) and the E8 Voronoi code (E8), each with and without balancing rows and columns (D+), one scale per matrix found by least squared error.

| Arm | Bits | Weight error, mean | Mean KL | Top token kept |
|---|---|---|---|---|
| S | 5.000 | -21.59 dB | 2.184 | 62.49 % |
| E8 | 5.000 | -22.75 dB | 2.345 | 62.03 % |
| D+S | 5.009 | -23.75 dB | 0.433 | 82.89 % |
| D+E8 | 5.009 | -25.46 dB | 0.275 | 85.99 % |
| S | 4.000 | -17.25 dB | 6.505 | 37.38 % |
| E8 | 4.000 | -18.29 dB | 7.281 | 36.24 % |
| D+S | 4.009 | -18.92 dB | 0.726 | 77.39 % |
| D+E8 | 4.009 | -20.41 dB | 0.421 | 82.48 % |
| D+S | 3.009 | -14.01 dB | 1.635 | 62.87 % |
| D+E8 | 3.009 | -15.26 dB | 0.816 | 75.29 % |
| llama.cpp `Q4_K_M` | 4.803 | -24.48 dB | 0.069 | 91.79 % |
| llama.cpp `Q4_0` | 4.500 | -21.06 dB | 0.117 | 89.40 % |
| llama.cpp `Q3_K_M` | 3.834 | -19.12 dB | 0.188 | 86.89 % |

Without balancing the model is broken at 5 bits, and E8 is worse than S at 4 and 5 bits though its weight error is 1 dB smaller. At 3 bits both keep the top token on about one position in ten, and at 2 bits on under one in a hundred. With balancing E8 is better than S at every rate.

llama.cpp's quantisers are far closer. `Q3_K_M` at 3.83 bits has a larger weight error than D+E8 at 5 bits and a smaller KL. They were run on the same 294 matrices with the command under "How to rerun", and `checks.out` confirms that no other tensor of their files differs from the reference.

## Where the damage comes from

Arm S at 4 bits on one kind of matrix at a time:

| Kind | Weights | Weight error, mean | Mean KL |
|---|---|---|---|
| `ffn_down` | 1,101 million | -16.71 dB | 1.989 |
| `attn_output` | 257 million | -15.29 dB | 0.962 |
| `ffn_up` | 1,101 million | -17.63 dB | 0.172 |
| `ffn_gate` | 1,101 million | -16.67 dB | 0.079 |
| `attn_q` | 257 million | -16.25 dB | 0.058 |
| `attn_v`, `attn_k` | 64 million each | -19.49 and -18.70 dB | 0.067 and 0.049 |

The two kinds that write into the stream carry it. Why they are so much more sensitive is not known. `attn_q` is rounded with nearly the weight error of `attn_output` and gives a sixteenth of the KL.

Scaling at 5 bits, scalar rounding:

| Arm | What shares a scale | Mean KL |
|---|---|---|
| S | the matrix | 2.184 |
| C+S | each column, divided by its standard deviation | 2.899 |
| R+S | each row, divided by its standard deviation | 0.340 |
| D+S | rows and columns | 0.433 |
| `XR` | each row, scale from its largest entry | 0.118 |
| G+S | the matrix, after balancing through the symmetries | 5.507 |
| GF+S | the matrix, after moving the gain vectors into it | 17.846 |

At 5 bits rows help and columns hurt. Balancing through the exact symmetries costs no bits and was predicted to help. The output is far worse.

The cause of that was not examined. In layer 12 three of the four balanced matrices have much heavier tails than before (kurtosis 18 to 33 against about 4, in `checks.out`), and in layer 0 none does. The weight errors of G+S and GF+S are measured against the moved matrices.

## The scale rule

llama.cpp's `Q4_0` sets a scale from the largest entry of each 32 weights. The table sets that rule (X) against the searched step, at two group sizes.

| Group | Rule | Bits at q = 16 and 32 | Mean KL at q = 16 | Mean KL at q = 32 |
|---|---|---|---|---|
| row | searched step (`R+S`) | 4.005 and 5.005 | 0.662 | 0.340 |
| row | largest entry (`XR`) | 4.005 and 5.005 | 0.448 | 0.118 |
| 32 weights | searched step (`B32+S`) | 4.5 and 5.5 | 0.365 | 0.131 |
| 32 weights | largest entry (`X32`) | 4.5 and 5.5 | 0.117 | 0.031 |

The largest-entry rule beats the searched step in all four cells. A finer group helps under both rules at equal levels, and it costs half a bit. `X32` at 4.5 bits scores the same as `XR` at 5.005 bits (0.117 and 0.118).

`XR` at q = 32 with its step multiplied by a factor, which clips a row's entries above that share of its largest:

| Step factor | Weight error, mean | Mean KL |
|---|---|---|
| 1 | -22.39 dB | 0.118 |
| 0.7 | -21.69 dB | 0.101 |
| 0.5 | -16.28 dB | 0.502 |
| 0.35 | -11.05 dB | 1.867 |

At a factor of 0.7 the rule clips 0.86 percent of entries, more than the 0.18 percent that `R+S` clips, and the output is no worse than with nothing clipped. It cuts no entry by more than 30 percent.

The same factor on `X32` at q = 16 does cost. This control was added after the second read: 0.9 gives 0.142, 0.8 gives 0.220 and 0.7 gives 0.416, against 0.117 with no factor. There it clips 4.9, 7.5 and 11.1 percent of entries. Why the factor is free on rows and costly on groups of 32 was not identified.

## A few large weights

Clipping alone, with no rounding. Every entry beyond a limit, counted in standard deviations of its row, is pulled back to the limit.

| Limit | Entries changed | Weight error, pooled | Weight error, mean | Mean KL | Top token kept |
|---|---|---|---|---|---|
| 3.45 | 0.173 % | -27.52 dB | -34.50 dB | 0.318 | 85.04 % |
| 5 | 0.015 % | -33.85 dB | -42.85 dB | 0.200 | 87.12 % |
| 8 | 0.0014 % | -43.24 dB | -56.93 dB | 0.010 | 97.35 % |
| 3.45, each entry moved toward or away from zero at random | 0.173 % | -27.52 dB | -34.50 dB | 0.095 | 91.17 % |

The limit 3.45 came from a look at where the searched step of `R+S` clips. On 42 matrices its limit is 2.22 to 3.92 standard deviations of the row, with a median of 3.27, and it clips 0.18 percent of entries. The probes use one limit on every matrix, so they touch nearly the entries `R+S` clips and not exactly those.

Clipping alone at 3.45 gives 0.318. `R+S`, which clips and rounds, gives 0.340. Rounding alone with nothing clipped (`XR`) gives 0.118. KL does not add, so no share of the damage can be given to either part.

Per unit of pooled squared error, the clip at 3.45 does 10 times the damage of the rounding of `XR`.

The last row moves the same entries by the same amounts in random directions. The mean KL falls to 30 percent. It is one random draw. No run moves other entries by as much, so whether these entries are more sensitive than the rest to a random change was not tested.

Clipping at 5 on one kind of matrix at a time:

| Kind | Entries changed | Weight error, pooled | Mean KL |
|---|---|---|---|
| `ffn_down` | 0.017 % | -32.29 dB | 0.164 |
| `ffn_gate` | 0.015 % | -33.86 dB | 0.014 |
| `ffn_up` | 0.013 % | -33.72 dB | 0.013 |
| `attn_v`, `attn_q`, `attn_k`, `attn_output` | 0.007 to 0.028 % | -32.73 to -42.93 dB | 0.002 to 0.006 |

The sensitive entries are in `ffn_down`, the matrices that write the feed-forward output into the stream. `ffn_up` and `ffn_gate` are changed by a similar amount and give a twelfth of the KL.

`large_entries.out` counts them: 188,362 entries of `ffn_down` lie beyond 5 standard deviations of their row, one in 5,845, with at least 666 in every layer. They are not a handful of weights. They are not evenly spread either. The ten busiest hidden units of a layer hold 2 to 24 percent of its large entries, and the ten busiest dimensions of the stream hold 3 to 23 percent.

## The lattice with the largest-entry rule

`VE8G` is the E8 Voronoi code with one scale per group, the smallest at which no block of the group has to be shrunk. On 42 matrices no block was shrunk in any of the three arms.

| Bits | Scalar, largest-entry rule | Mean KL | E8, same group | Mean KL | Weight error gain, mean | KL gain |
|---|---|---|---|---|---|---|
| 4.500 | `X32` (`Q4_0`) | 0.117 | `VE832` | 0.075 | 1.51 dB | 1.96 dB |
| 5.005 | `XR` | 0.118 | `VE8R` | 0.059 | 2.55 dB | 2.99 dB |
| 4.005 | `XR` | 0.448 | `VE8R` | 0.215 | 2.52 dB | 3.20 dB |

The gain in weight error is larger than the 0.65 dB that the shape of the E8 cell is worth. A possible reason, not tested: the E8 code's region reaches twice as far along a single weight as the cube's does, so a block with one large weight can use a smaller scale.

The scalar side of this comparison is the plain `Q4_0` rule. The step factor above did not improve it at a group of 32. A scalar rule with an offset or a searched scale per group, as llama.cpp's K quantisers have, was not tried at equal bits and might close part of the gap.

At 4.5 bits `VE832` scores 0.075 and `Q4_K_M` at 4.8 bits scores 0.069. At 4.005 bits `VE8R` scores 0.215 and `Q3_K_M` at 3.83 bits scores 0.188.

## Measures that were tried

| Measure | When it was written | What happened |
|---|---|---|
| Squared weight error | the plan | Rank correlation with mean KL over 32 variants: 0.751 pooled, 0.721 as a mean. In 54 of the 496 pairs one variant has the smaller pooled error by at least 1 dB and a mean KL at least twice as large. It has the clip at 3.45 as 5.8 dB better than `XR`, whose KL is 2.7 times smaller. |
| Path-weighted error: each weight counted by the gains and the partner matrix it meets in the forward pass | round 2, before its arms were scored | Ranks the eight arms at 5 bits no better than squared error (0.833 against 0.905). |
| Second path-weighted error, with rows weighted too | after round 2 | An arm built to minimise it was tried on layer 12. On its own measure it was worse than rows-only rounding (`R+S`) on all seven matrices and worse than plain rounding on five (`checks.out`). It was not scored. |
| Spectral error: the largest singular value of the weight change | round 5, before it was computed | Rank correlation 0.889 over 26 variants, against 0.603 for squared error on the same matrices. It has both clip probes as better than `XR`, by 4.6 and 9.6 dB, when both are worse. With the three clip probes left out, a choice made after seeing the result, it is 0.916 over 23. |
| Activation-weighted error: each column counted by the mean square of the input it multiplies, from `llama-imatrix` on 40 passages that are not scored | round 6, before it was computed | Rank correlation 0.763 over 26 variants, below the spectral error. It has the clip probes as better than `XR` by 6.1 and 16.2 dB. |

The spectral and activation-weighted errors were computed on the matrices of six layers, not all 42.

## Predictions

| | Prediction | Verdict |
|---|---|---|
| K1 | Zeroing the unused rows of `per_layer_token_embd` changes nothing. | holds |
| K2 | Zeroing the same rows of `token_embd` moves the output. | holds |
| R1 | E8 has a lower KL than S, with and without balancing, at 3, 4 and 5 bits. | fails: without balancing E8 is worse at 4 and 5 bits |
| R2 | Balancing lowers the KL at 3, 4 and 5 bits. | holds |
| R3 | At 5 bits the KL gain of E8 is 0.5 to 1.5 times its gain in pooled weight error. | fails without balancing (-0.23), holds with it (0.89) |
| R4 | At 2 bits neither S nor E8 keeps the top token on half the positions. | holds: 0.85 and 0.26 percent |
| R5 | `Q4_K_M` beats every arm of the first grid at q = 16. | holds, at 4.80 bits against 4.0 |
| A1 | Rows alone and columns alone each help, and both together help most. | fails: columns hurt, and rows alone beat both |
| A2 | Balancing through the symmetries helps. | fails: 5.507 against 2.184 |
| A3 | The path-weighted error ranks the arms better than squared error. | fails |
| B1 | This repo's copy of the `Q4_0` rule scores as llama.cpp's file. | holds |
| B2 | Clipping is what hurts: the KL is lowest with nothing clipped, and the weight error is lowest with some clipping. | fails on both counts: the KL is lowest at step factor 0.7 and the weight error at 1 |
| B3 | A finer group gives a lower KL under both rules, at both q. | holds, with half a bit more for the finer group |
| C1 | With the largest-entry rule, E8 beats scalar at equal bits in three pairs. | holds |
| C2 | Its KL gain is 0.5 to 1.5 times its gain in mean weight error. | holds: 1.29, 1.17, 1.27 |
| D1 | The random move has under half the KL of the clip. | holds: 0.095 against 0.318. The plan read this as "the direction carries the damage, not the choice of entries". The second half of that was not tested. |
| D2 | The spectral error ranks 26 variants with a rank correlation of at least 0.8, above squared error, and has the clip above `XR`. | fails: 0.889 against 0.603, but the clip is 4.6 dB under `XR` |
| E1 | The activation-weighted error ranks them at 0.8 or more, above the spectral error, and has both clips above `XR`. | fails on all three parts: 0.763 |

## What was not in the plan

- The plan was extended five times. The extensions are sections of [PREREG.md](PREREG.md).
- `flip:3.45` was started a minute before its prediction was committed and was scored four minutes after. The clip probes at 5 and 8 were running when round 5 was committed. Their scores were read after it and are not in its table.
- `results.jsonl` joins the files of the separate runs in the order the runs were started. Up to five ran at once, so the order of its rows is not the order in time.
- The zeroing of unread keys and values, the random move along the symmetries, the clip probes and the step-factor control on `X32` were run with no prediction written.
- The second path-weighted error and the arm that minimises it were written after round 2. The arm was dropped after one layer.
- The limit 3.45 came from a look at four matrices that was not recorded. `checks.py` was written after the second read to put that look, and five others, on record.
- The scale of a matrix is searched on a sample of 131,072 of its blocks, not on all of it as in the first record.
- The first four arms at q = 32 were built a second time without scoring, to add the path-weighted errors that the first pass did not record. Those are the last four rows of `results.jsonl`.
- E1 was computed on the 36 of the 42 matrices that the model reads. The tool records nothing for the others. On those 36 the spectral error has 0.945.
- The lattice A8\* was left out, as the plan says.
- 55 million of the 3.95 billion block weights are never read. They were rounded and counted with the rest, in the bits and in the weight errors.
- Two edits were made to the plan after its commits: the time stated for round 2 was set to the commit's own, and a column of the round 2 table was labelled as pooled.
- The tool prints a perplexity for the reference that differs from the one it prints for the same model run directly (7.595 against 7.887). The cause was not found. Only its KL and top-token lines are used.

## Limits

- One model, one text, one runtime. Every variant is scored on the same 10,200 positions.
- The standard errors treat positions as independent. They come from 40 passages of 255 positions in a row, so the errors are understated. Near ties cannot be ordered: `X32` at q = 16 against `XR` at q = 32, or `VE832` against `Q4_K_M`.
- The clip probes, the random move of the same entries and the move along the symmetries are single draws.
- The reference is an 8-bit file, not the weights as trained.
- No rounding here uses the model's activations. Methods in use do, and llama.cpp's quantisers were run without their importance matrix. Activations were used for one measure only.
- Several comparisons are not at equal bits: llama.cpp's quantisers against the arms, and the two group sizes. Equal-bit comparisons are the three pairs of the lattice table.
- The lattice arms are scored from 16-bit float copies. No packed file format exists for them and no decoding speed was measured. Their scales are counted at 16 bits per group, as for the scalar rule.
- The lattice code runs in 32-bit floats on a GPU. `harness/test_tq.py` checks the searched-step arms against the reference code. The largest-entry rules are checked only by `checks.py` and by the match with llama.cpp's file.
- Why the writing matrices and the large entries of `ffn_down` are so sensitive is not known.

## How to rerun

Needs NumPy, torch, a llama.cpp build with `llama-perplexity`, its `gguf` Python package, and the model file.

```bash
python3 harness/build_eval_text.py ~/src/llama.cpp eval.txt
```

```bash
llama-perplexity -m gemma-4-E4B-it-Q8_0.gguf -f eval.txt -c 512 -b 512 --chunks 40 -ngl 999 --kl-divergence-base base.kld
```

```bash
python3 experiments/model-output-gemma-e4b-2026-11-15/run.py --model gemma-4-E4B-it-Q8_0.gguf --kld base.kld --work work --out results.jsonl unchanged identity XR:32 VE832:16
```

```bash
python3 experiments/model-output-gemma-e4b-2026-11-15/summary.py
```

llama.cpp's own quantisers were made with this command, with `Q4_0`, `Q4_K_M` and `Q3_K_M` as the type, and scored as `file:PATH`:

```bash
llama-quantize --allow-requantize --tensor-type token_embd=q8_0 --tensor-type per_layer_model_proj=f32 --tensor-type inp_gate=f32 --tensor-type '\.proj\.=f32' gemma-4-E4B-it-Q8_0.gguf OUT.gguf Q4_0
```

The activation statistics for round 6:

```bash
llama-imatrix -m gemma-4-E4B-it-Q8_0.gguf -f eval.txt -c 512 -b 512 --chunk 60 --chunks 40 -ngl 999 -o imatrix.gguf
```

The reference outputs take 5.3 GB and each changed copy 12 GB while it is scored. A scalar variant takes about two minutes and an E8 variant about five on an M1 Max.
