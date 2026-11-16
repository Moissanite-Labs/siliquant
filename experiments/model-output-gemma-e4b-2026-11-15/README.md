# Rounding a real model: what its output follows

This record rounds the weights of a real language model and measures how far its output moves. The model is `gemma-4-E4B-it`, 7.5 billion weights, in an 8-bit GGUF file. Each variant is scored by llama.cpp's own tool against the unrounded model on the same 10,200 positions of text.

The first record in this repo measured the error of rounded weights and nothing else. This one asks whether that error says anything about the output.

Status: `exploratory`. The plan, [PREREG.md](PREREG.md), was committed before the first run and extended four times as results came in. Each extension lists what was known and was committed before its variants were scored. 10 of 17 predictions hold.

## Result in short

1. **Weight error does not say what the output does.** Pulling back the largest 0.17 percent of the weights, with no rounding, changes the weights by -34.5 dB and gives a mean KL of 0.318. A rounding of every weight with an error of -22.4 dB gives 0.118. Among 34 rounding variants, 64 of the 561 pairs have the smaller weight error by at least 1 dB and a mean KL at least twice as large.
2. **The output follows how the scale is set.** The arms of the first record use one scale per matrix, found by least squared error. At 5 bits per weight they keep the model's top token on 62 percent of positions, and on 83 to 86 percent with rows and columns balanced. One scale per row, set from the row's largest entry, keeps it on 90 percent at the same bits.
3. **With the scale set that way, the E8 lattice gives a closer output at equal bits.** At 4.5 bits per weight the mean KL is 0.075 against 0.117 for llama.cpp's `Q4_0`. At 5.005 bits it is 0.059 against 0.118. With the first record's scale rule and no balancing the lattice is worse than scalar rounding.
4. **Three closures hold exactly on the model.** 2.79 billion weights can be zeroed with no change in output for this text, and 55 million for any text. The weights can be moved 1.6 times their own length along exact symmetries with the top token kept on 99.93 percent of positions.
5. **No formula tried here predicts the damage.** Squared error and an error weighted along the forward pass do not rank the variants. The largest singular value of the change ranks the variants that round (rank correlation 0.92 over 23 of them) and misses the clipping probes by 4.6 and 9.6 dB.

## How it was measured

| | |
|---|---|
| Model | `gemma-4-E4B-it-Q8_0.gguf`, SHA-256 in the plan. 42 blocks, 294 block matrices holding 3.95 billion weights, token tables holding 3.49 billion. |
| What is rounded | The 294 block matrices. Token tables and small tensors stay as they are. |
| Text | Built by `harness/build_eval_text.py` from the llama.cpp repository at commit `427291b`: slices of documentation alternating with slices of source code. 40 passages of 512 tokens, the second half of each scored. |
| Score | `llama-perplexity --kl-divergence`: mean KL divergence from the reference model's next-token distribution, and the share of positions where the top token stays the same. |
| Bits | Counted from the codebook: log2 of the levels per weight, plus 16 bits for each scale. Rounded values are stored as 16-bit floats so that llama.cpp reads exactly what the rounding produced. |

Three checks of the harness:

- The unchanged model scores a mean KL of 0.000000 against its own saved outputs. The largest KL at any position is 0.000058.
- A copy with the block matrices stored as 16-bit floats scores the same on every passage.
- `X32` at q = 16 is this repo's own code for llama.cpp's `Q4_0` rule. It gives the same values as llama.cpp's file on every entry of three matrices, and the same scores as that file: mean KL 0.1173, top token kept on 89.41 against 89.40 percent.

All numbers below are in [results.out](results.out), printed by `summary.py` from `results.jsonl` and `spectral.jsonl`.

## Closures

| Change | Weights | Mean KL | Top token kept |
|---|---|---|---|
| K1. Rows of `per_layer_token_embd` zeroed for the 259,937 tokens not in the text | 2,794,842,624 (37.2 % of the model) | 0.000000 | 99.99 % |
| K2. The same rows of `token_embd` zeroed | 665,438,720 | 1.364 | 95.51 % |
| Keys and values of the last 18 layers zeroed | 55,050,240 | 0.000000 | 99.99 % |
| Random move along three exact symmetries, 4.09 dB larger than the weights moved | 3,633,315,840 moved | 0.000047 | 99.93 % |

The first and third rows score exactly as the unchanged model does, to every printed digit.

- **K1.** The table is read by row, one row per token. A row whose token never occurs is on no path from input to output. This closure holds only for inputs made of the 2,207 tokens of the text.
- **K2.** The main token table is read by row too, but the same weights are the output layer. Its rows stay on a path. A shared weight can be closed only if every use of it is off the path.
- **Keys and values.** The last 18 of 42 layers reuse the keys and values of an earlier layer. Their own `attn_k` and `attn_v` are in the file and llama.cpp never reads them. This closure holds for every input.
- **Symmetries.** Three scalings leave the forward pass unchanged: a hidden unit's row of `ffn_up` against its column of `ffn_down`, and an input dimension's columns of the reading matrices against its entry of the gain vector before them. `run.py` documents them in `gauge_layer`. The copy is stored in 16-bit floats. One position of 10,200 has a KL of 0.43. Whether the float format causes it was not checked.

## The first grid: one scale per matrix

The arms of the first record: scalar rounding (S) and the E8 Voronoi code (E8), each with and without balancing rows and columns (D+), one scale per matrix found by least squared error.

| Arm | Bits | Weight error, mean | Mean KL | Top token kept |
|---|---|---|---|---|
| S | 5.000 | -21.59 dB | 2.184 | 62.49 % |
| E8 | 5.000 | -22.75 dB | 2.345 | 62.03 % |
| D+S | 5.009 | -23.75 dB | 0.433 | 82.89 % |
| D+E8 | 5.009 | -25.46 dB | 0.275 | 85.99 % |
| S | 4.000 | -17.25 dB | 6.505 | 37.38 % |
| E8 | 4.000 | -18.29 dB | 7.281 | 36.25 % |
| D+S | 4.009 | -18.92 dB | 0.726 | 77.39 % |
| D+E8 | 4.009 | -20.41 dB | 0.421 | 82.48 % |
| D+S | 3.009 | -14.01 dB | 1.635 | 62.87 % |
| D+E8 | 3.009 | -15.26 dB | 0.816 | 75.29 % |
| llama.cpp `Q4_K_M` | 4.803 | -24.48 dB | 0.069 | 91.79 % |
| llama.cpp `Q4_0` | 4.500 | -21.06 dB | 0.117 | 89.40 % |
| llama.cpp `Q3_K_M` | 3.834 | -19.12 dB | 0.188 | 86.89 % |

Without balancing the model is broken at 5 bits, and E8 is worse than S at 4 and 5 bits though its weight error is 1 dB smaller. At 3 bits both keep the top token on about one position in ten, and at 2 bits on under one in a hundred. With balancing E8 is better than S at every rate.

llama.cpp's quantisers are far closer at fewer bits. `Q3_K_M` at 3.83 bits has a larger weight error than D+E8 at 5 bits and a smaller KL.

## Where the damage comes from

Arm S at 4 bits on one kind of matrix at a time:

| Kind | Weights | Mean KL |
|---|---|---|
| `ffn_down` | 1,101 million | 1.989 |
| `attn_output` | 257 million | 0.962 |
| `ffn_up` | 1,101 million | 0.172 |
| `ffn_gate` | 1,101 million | 0.079 |
| `attn_v`, `attn_q`, `attn_k` | 64 to 257 million each | 0.049 to 0.067 |

The two kinds that write into the stream carry it. Their rows are the dimensions of the stream, and those rows differ widely in size.

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

Rows help and columns hurt. Balancing through the exact symmetries costs no bits and was predicted to help. It makes the matrices heavy-tailed and the output far worse.

## The scale rule

llama.cpp's `Q4_0` sets a scale from the largest entry of each 32 weights and clips nothing. The table sets that rule (X) against the searched step, at two group sizes.

| Group | Rule | Bits at q = 16 and 32 | Mean KL at q = 16 | Mean KL at q = 32 |
|---|---|---|---|---|
| row | searched step (`R+S`) | 4.005 and 5.005 | 0.662 | 0.340 |
| row | largest entry (`XR`) | 4.005 and 5.005 | 0.448 | 0.118 |
| 32 weights | searched step (`B32+S`) | 4.5 and 5.5 | 0.365 | 0.131 |
| 32 weights | largest entry (`X32`) | 4.5 and 5.5 | 0.117 | 0.031 |

A finer group helps under both rules. The largest-entry rule beats the searched step in all four cells.

The searched step clips. A sweep of `XR` at q = 32 with its step multiplied by a factor shows that mild clipping is not the cause:

| Step factor | Weight error | Mean KL |
|---|---|---|
| 1 (nothing clipped) | -22.39 dB | 0.118 |
| 0.7 | -21.69 dB | 0.101 |
| 0.5 | -16.28 dB | 0.502 |
| 0.35 | -11.05 dB | 1.867 |

## A few large weights

Clipping alone, with no rounding. Every entry beyond a limit, counted in standard deviations of its row, is pulled back to the limit.

| Limit | Entries changed | Weight error | Mean KL | Top token kept |
|---|---|---|---|---|
| 3.45 (what `R+S` clips at q = 32) | 0.173 % | -34.50 dB | 0.318 | 85.04 % |
| 5 | 0.015 % | -42.85 dB | 0.200 | 87.12 % |
| 8 | 0.0014 % | -56.93 dB | 0.010 | 97.35 % |
| 3.45, each entry moved toward or away from zero at random | 0.173 % | -34.50 dB | 0.095 | 91.17 % |

Clipping at 3.45 accounts for nearly all of the damage of `R+S` (0.318 of 0.340). Per unit of squared error it does 44 times the damage of the rounding noise of `XR`.

The last row moves the same entries by the same amounts in random directions. The damage falls to 30 percent. So both things matter: these entries are sensitive, and pulling them all toward zero is three times worse than moving them at random.

Clipping at 5 on one kind of matrix at a time:

| Kind | Entries changed | Weight error | Mean KL |
|---|---|---|---|
| `ffn_down` | 0.017 % | -33.96 dB | 0.164 |
| `ffn_gate` | 0.015 % | -35.71 dB | 0.014 |
| `ffn_up` | 0.013 % | -36.02 dB | 0.013 |
| `attn_v`, `attn_q`, `attn_k`, `attn_output` | 0.007 to 0.028 % | -38.94 to -56.22 dB | 0.002 to 0.006 |

The sensitive entries are in `ffn_down`, the matrices that write the feed-forward output into the stream. `ffn_up` and `ffn_gate` are changed by a similar amount and give a twelfth of the KL.

[large_entries.out](large_entries.out) counts them: 188,362 entries of `ffn_down` lie beyond 5 standard deviations of their row, one in 5,845. They sit in every layer, and in no layer do the ten busiest hidden units hold more than 24 percent of them. So they are a thin sprinkling, not a handful of special weights.

## The lattice with the working rule

`VE8G` is the E8 Voronoi code with one scale per group, the smallest at which no block of the group has to be shrunk.

| Bits | Scalar, largest-entry rule | Mean KL | E8, same group | Mean KL | Weight error gain | KL gain |
|---|---|---|---|---|---|---|
| 4.500 | `X32` (`Q4_0`) | 0.117 | `VE832` | 0.075 | 1.52 dB | 1.96 dB |
| 5.005 | `XR` | 0.118 | `VE8R` | 0.059 | 2.55 dB | 2.99 dB |
| 4.005 | `XR` | 0.448 | `VE8R` | 0.215 | 2.53 dB | 3.20 dB |

The gain is larger than the 0.65 dB that the shape of the E8 cell is worth. The E8 code's region reaches twice as far along a single weight as the cube's does, so a block with one large weight needs a smaller scale.

At 4.5 bits `VE832` is close to `Q4_K_M` at 4.8 bits (0.075 against 0.069). At 4.005 bits `VE8R` is behind `Q3_K_M` at 3.83 bits (0.215 against 0.188).

## Measures that were tried

| Measure | When it was written | What happened |
|---|---|---|
| Squared weight error | the plan | Rank correlation with mean KL over 34 variants: 0.720. It has clipping at 3.45 as 12 dB better than `XR`, which is 2.7 times closer. |
| Path-weighted error: each weight counted by the gains and the partner matrix it meets in the forward pass | round 2, before its arms were scored | Ranks the eight arms at 5 bits no better than squared error (0.833 against 0.905). |
| Second path-weighted error, with rows weighted too | after round 2 | An arm built to minimise it was tried on one layer. It was worse than plain rounding on its own measure and was not scored. |
| Spectral error: the largest singular value of the weight change | round 5, before it was computed | Ranks the 26 variants better than squared error does (rank correlation 0.889 against 0.603). It has both clip probes as better than `XR`, by 4.6 and 9.6 dB, when both are worse. |

## Predictions

| | Prediction | Verdict |
|---|---|---|
| K1 | Zeroing the unused rows of `per_layer_token_embd` changes nothing. | holds |
| K2 | Zeroing the same rows of `token_embd` moves the output. | holds |
| R1 | E8 has a lower KL than S, with and without balancing, at 3, 4 and 5 bits. | fails: without balancing E8 is worse at 4 and 5 bits |
| R2 | Balancing lowers the KL at 3, 4 and 5 bits. | holds |
| R3 | At 5 bits the KL gain of E8 is 0.5 to 1.5 times its weight error gain. | fails without balancing (-0.23), holds with it (0.89) |
| R4 | At 2 bits neither S nor E8 keeps the top token on half the positions. | holds: 0.85 and 0.26 percent |
| R5 | `Q4_K_M` beats every arm of the first grid at 4 bits. | holds |
| A1 | Rows alone and columns alone each help, and both together help most. | fails: columns hurt, and rows alone beat both |
| A2 | Balancing through the symmetries helps. | fails: 5.507 against 2.184 |
| A3 | The path-weighted error ranks the arms better than squared error. | fails |
| B1 | This repo's copy of the `Q4_0` rule scores as llama.cpp's file. | holds |
| B2 | Clipping is what hurts: the KL is lowest with nothing clipped. | fails: lowest at step factor 0.7 |
| B3 | A finer group gives a lower KL under both rules. | holds |
| C1 | With the largest-entry rule, E8 beats scalar at equal bits in three pairs. | holds |
| C2 | Its KL gain is 0.5 to 1.5 times its weight error gain. | holds: 1.29, 1.17, 1.27 |
| D1 | The direction of the error carries the damage: the random move has under half the KL of the clip. | holds: 0.095 against 0.318 |
| D2 | The spectral error ranks 26 variants with a rank correlation of at least 0.8, above squared error, and has the clip above `XR`. | fails: 0.889 against 0.603, but the clip is 4.6 dB under `XR` |

## What was not in the plan

- The plan was extended four times. The extensions are sections of [PREREG.md](PREREG.md), each committed before its variants were scored. `flip:3.45` was started a minute before its prediction was committed and was scored four minutes after.
- The zeroing of unread keys and values, the random move along the symmetries, and the clip probes were run with no prediction written.
- The second path-weighted error and the arm that minimises it were written after round 2. The arm was dropped after one layer.
- The scale of a matrix is searched on a sample of 131,072 of its blocks, not on all of it as in the first record.
- Up to five runs shared the GPU. `XR` at q = 32 was scored a second time alone. Every number was equal to the first run's.
- The lattice A8\* was left out, as the plan says.
- 55 million of the 3.95 billion block weights are never read. They were rounded and counted with the rest.
- The tool prints a perplexity for the reference that differs from the one it prints for the same model run directly (7.595 against 7.887). Only its KL and top-token lines are used.

## Limits

- One model and one text of 10,200 scored positions. The standard errors are for the mean over those positions and say nothing about other text.
- The reference is an 8-bit file, not the weights as trained.
- Nothing here looks at the model's activations. Methods in use do, and llama.cpp's quantisers were run without their importance matrix.
- The lattice arms are scored from 16-bit float copies. No packed file format exists for them and no decoding speed was measured. Their scales are counted at 16 bits per group, as for the scalar rule.
- The lattice code runs in 32-bit floats on a GPU. Its reconstructions are checked to be codewords of the reference code in `harness/test_tq.py`, for the searched-step arms only.
- The spectral error was computed on 42 of the 294 matrices.
- Which weights the clip probes touch, by row, column and layer, was not examined.
- KL at single positions has a heavy tail. Means are reported with their standard errors, and medians are in `results.out`.

## How to rerun

Needs NumPy, torch, a llama.cpp build with `llama-perplexity`, its `gguf` Python package, and the model file.

```bash
python3 harness/build_eval_text.py ~/src/llama.cpp eval.txt
```

```bash
llama-perplexity -m gemma-4-E4B-it-Q8_0.gguf -f eval.txt -c 512 -b 512 --chunks 40 -ngl 999 --kl-divergence-base base.kld
```

```bash
python3 experiments/model-output-gemma-e4b-2026-11-15/run.py --model gemma-4-E4B-it-Q8_0.gguf --kld base.kld --work work --out results.jsonl identity XR:32 VE832:16
```

```bash
python3 experiments/model-output-gemma-e4b-2026-11-15/summary.py
```

The reference outputs take 5.3 GB and each changed copy 12 GB while it is scored. A scalar variant takes about two minutes and an E8 variant about five on an M1 Max.
