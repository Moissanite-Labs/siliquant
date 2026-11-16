# Plan: does a smaller weight error give a closer output on a real model

Written and committed before the main run. The first record in this repo measured the error of rounded weights and nothing else. This run measures what a real language model's output does.

## What is measured

The reference is `gemma-4-E4B-it` in the 8-bit GGUF file already on the machine (SHA-256 below). It has 7.5 billion weights. The run rounds the 294 matrices inside the 42 blocks (attention and feed-forward, 3.95 billion weights) and leaves the token tables and the small tensors alone.

Each changed model is scored by llama.cpp's own tool, `llama-perplexity --kl-divergence`, against the reference model's saved outputs on the same tokens. The scores are the mean KL divergence from the reference's next-token distribution, and the share of positions where the most likely token stays the same. 40 passages of 512 tokens are read and the second half of each is scored.

The text is built by `harness/build_eval_text.py` from the llama.cpp repository at commit `427291b`: slices of its documentation alternating with slices of its source code. Nothing is downloaded.

## Arms

The arms are those of the first record, at q levels per weight, with one scale per matrix found by the same search.

| Arm | What it does |
|---|---|
| S | scalar rounding to q evenly spaced levels |
| E8 | blocks of 8 weights rounded to the Voronoi code of the E8 lattice, q^8 codewords |
| D+S, D+E8 | the same after rows and columns are balanced by their standard deviations, two 16-bit scale vectors counted in the bits |

The main grid is q = 8, 16 and 32 (3, 4 and 5 bits per weight) for all four arms, and q = 4 for S and E8. A8* is left out of this round: it gained less than half of what E8 gained in the first record and its decoder is about nine times the cost.

Two of llama.cpp's own quantisers, `Q4_K_M` and `Q3_K_M`, are run on the same matrices as a reference point from practice. They use a scale per block of weights and spend more bits than their names say.

## What is already known

Three variants were run while building the harness. The unchanged model scores a KL of zero against its own saved outputs, with noise under 0.0001. A control copy with the block matrices stored as 16-bit floats scores the same. Arm S at q = 16 on the 42 `attn_k` matrices alone gives a mean KL of 0.049 and keeps the top token on 93.2 percent of positions.

## Predictions

| | Prediction | Fails if |
|---|---|---|
| R1 | E8 has a lower mean KL than S, and D+E8 lower than D+S, at q = 8, 16 and 32. | any of the six comparisons goes the other way |
| R2 | Balancing lowers the mean KL: D+S under S and D+E8 under E8 at q = 8, 16 and 32. | any of the six goes the other way |
| R3 | At q = 32 the output follows the weight error: the ratio of mean KL, S over E8, in dB is between 0.5 and 1.5 times the gain in pooled weight error in dB. The same for the balanced pair. | either ratio is outside the range |
| R4 | At q = 4 neither S nor E8 keeps the top token on half the positions. | either keeps it on 50 percent or more |
| R5 | `Q4_K_M` has a lower mean KL than every arm at q = 16. | any arm at q = 16 is lower |

## Closures on the same model

The model has a table, `per_layer_token_embd`, of 2.82 billion weights that is read by row, one row per token. A row whose token never occurs in the input is on no path from the input to the output. The closure rule of the weft-loom notes says such weights can be closed with no effect at all.

The main token table, `token_embd`, is also read by row at the input. The same weights are used a second time as the output layer, so its rows are on a path even when their tokens never occur in the input.

| | Prediction | Fails if |
|---|---|---|
| K1 | Zeroing every row of `per_layer_token_embd` whose token is not in the evaluation text leaves every score equal to the unchanged model's. | mean KL is above 0.0001 or the top token agreement is under 99.99 percent |
| K2 | Zeroing the same rows of `token_embd` moves the output: mean KL above 0.01. | mean KL is 0.01 or less |

## Not part of the predictions

Arm S at q = 16 on one kind of matrix at a time, for the seven kinds, to see where the damage comes from. Whatever the first results point at is run next and labelled as unplanned.

## Limits known before the run

- One model, one text of two kinds, 10,200 scored positions.
- The reference is itself an 8-bit file. Every score is a distance from that file, not from the weights as trained.
- No rounding here looks at the model's activations. Methods in use do.
- Changed tensors are stored as 16-bit floats. The bit rate of an arm is counted from its codebook, not from a file.

## Reference file

`gemma-4-E4B-it-Q8_0.gguf`, 8,192,953,472 bytes, SHA-256 `f8854aa4480df62585a279e7ca0a881554fc18a41c59c4f62642d16a2ae47012`.

## Round 2, added while the first grid was running

Written at 00:46 UTC on 2026-11-16 (the time of the commit that added it), after these results were in and before any variant of this round was scored.

| Variant | Bits | Weight error | Mean KL | Top token kept |
|---|---|---|---|---|
| K1, unused rows of `per_layer_token_embd` zeroed | | | 0.0000 | 99.99 % |
| K2, the same rows of `token_embd` zeroed | | | 1.364 | 95.51 % |
| S at q = 32 | 5.000 | -20.80 dB | 2.184 | 62.49 % |
| E8 at q = 32 | 5.000 | -22.14 dB | 2.345 | 62.03 % |
| D+S at q = 32 | 5.009 | -22.82 dB | 0.433 | 82.89 % |

E8 has the smaller weight error and the larger KL, so R1 has already failed. Balancing cuts the KL five times for 2 dB of weight error. Weight error is not what the output follows, and this round asks what it does follow.

Two variants were scored with no prediction written down. Zeroing the key and value matrices of the 18 layers that reuse an earlier layer's keys and values (55,050,240 weights that llama.cpp never reads) left every score unchanged. Moving the weights by random amounts along three exact symmetries of the forward pass, a move 4.1 dB larger than the weights themselves, gave a mean KL of 0.00005.

### What this round runs

- `R+S` and `C+S` at q = 32: scalar rounding with the rows only, or the columns only, divided by their standard deviations.
- `G+S` and `GF+S` at q = 32: scalar rounding after the matrices are balanced through the exact symmetries, which costs no bits. `G` gives both sides of each symmetry the same size. `GF` moves the gain vectors wholly into the matrices.
- The path-weighted error of every arm: squared error with each weight counted by the gains and the partner matrix it is multiplied with in the forward pass. It is unchanged by the three symmetries, which plain squared error is not. One look was taken before writing this: for arm S on layer 12 it was within 1.5 dB of the plain error on four matrices.
- llama.cpp's `Q4_0` next to the two quantisers already in the plan.

### Predictions

| | Prediction | Fails if |
|---|---|---|
| A1 | R+S and C+S at q = 32 each have a lower mean KL than S, and D+S is lower than both. | any of the four comparisons goes the other way |
| A2 | G+S at q = 32 has a lower mean KL than S. | it is not lower |
| A3 | Over the eight arms at q = 32 (S, E8, D+S, D+E8, R+S, C+S, G+S, GF+S) the path-weighted error ranks the arms by mean KL better than the plain weight error does, with a rank correlation of at least 0.7. | its rank correlation is under 0.7 or not above that of the plain error |

## Round 3, added after round 2's rounding arms were scored

Written after these further results and before any variant of this round was scored.

| Variant | Bits | Weight error, mean | Mean KL | Top token kept |
|---|---|---|---|---|
| D+E8 at q = 32 | 5.009 | -25.46 dB | 0.275 | 85.99 % |
| R+S at q = 32 | 5.005 | -23.20 dB | 0.340 | 84.20 % |
| C+S at q = 32 | 5.004 | -22.02 dB | 2.899 | 57.70 % |
| G+S at q = 32 | 5.000 | -19.32 dB | 5.507 | 35.02 % |
| GF+S at q = 32 | 5.000 | -16.65 dB | 17.846 | 0.63 % |
| S at q = 16 | 4.000 | -17.25 dB | 6.505 | 37.38 % |
| E8 at q = 16 | 4.000 | -18.29 dB | 7.281 | 36.25 % |
| D+S at q = 16 | 4.009 | -18.92 dB | 0.726 | 77.39 % |
| llama.cpp `Q4_0` | 4.500 | -21.06 dB | 0.117 | 89.40 % |

A1 and A2 have failed. Scaling rows helps and scaling columns hurts. Balancing through the symmetries makes the output far worse. `Q4_0` has a larger weight error than three of the arms above and a KL two to three times smaller.

`Q4_0` differs from the arms in two ways. It gives every 32 consecutive weights their own scale, and it sets that scale from the largest entry, so nothing is clipped. The arms search for the scale with the least squared error, which clips the largest weights. This round separates the two.

### What this round runs

- `XG` at q levels: the `Q4_0` rule with one scale per group G, where G is 32 weights (`X32`) or a row (`XR`). `X32` at q = 16 is `Q4_0` itself: on three matrices it gives the same values as llama.cpp's file on every entry.
- `XR` at q = 32 with the step multiplied by 0.7, 0.5 and 0.35. A smaller step is finer and clips every entry above that share of the row's largest.
- `B32+S`: one scale per 32 weights, the block's root mean square, then the searched step with clipping as in arm S.
- `R+S` at q = 16, to complete the grid of two group sizes at q = 16 and 32.

Weight errors were looked at on two or three matrices of layers 12 and 30 before writing this. At q = 16, `X32` has -20.8 to -21.3 dB and `B32+S` has -19.1 and -19.9 dB, so at this group size the rule with no clipping also has the smaller weight error. At q = 32 on rows, `XR` has -20.9 to -22.5 dB against -22.6 to -22.9 dB for `R+S`.

### Predictions

| | Prediction | Fails if |
|---|---|---|
| B1 | `X32` at q = 16 scores as `Q4_0` does: mean KL within 10 percent of 0.1173. | it is outside 0.106 to 0.129 |
| B2 | Clipping is what hurts. Among `XR` at q = 32 with the step multiplied by 1, 0.7, 0.5 and 0.35, the mean KL is lowest with no clipping (1), and the mean weight error is lowest at one of the others. | the lowest mean KL is at 0.7, 0.5 or 0.35, or the lowest weight error is at 1 |
| B3 | A finer scale gives a lower mean KL under both rules, at both q: `X32` under `XR`, and `B32+S` under `R+S`. | any of the four goes the other way |

## Round 4, added while round 3 was running

Written after these further results and before any variant of this round was scored.

| Variant | Bits | Weight error, mean | Mean KL | Top token kept |
|---|---|---|---|---|
| `X32` at q = 16 | 4.500 | -21.06 dB | 0.1173 | 89.41 % |
| `XR` at q = 32 | 5.005 | -22.39 dB | 0.118 | 89.82 % |
| `XR` at q = 32, step times 0.7 | 5.005 | -21.69 dB | 0.101 | 90.78 % |
| llama.cpp `Q4_K_M` | 4.803 | -24.48 dB | 0.069 | 91.79 % |
| llama.cpp `Q3_K_M` | 3.834 | -19.12 dB | 0.188 | 86.89 % |
| S at q = 16 on `attn_output` alone | | -15.29 dB | 0.962 | 76.40 % |
| S at q = 16 on `ffn_down` alone | | -16.71 dB | 1.989 | 63.29 % |
| S at q = 16 on each of the other five kinds alone | | | 0.049 to 0.172 | |

B1 holds: the copy of the `Q4_0` rule scores as llama.cpp's file does. A scale per row with nothing clipped (`XR`, 0.118) is three times closer than a scale per row with the searched step (`R+S`, 0.340). Under plain rounding the damage comes from the two kinds of matrix that write into the stream.

### What this round runs

The lattice was tested in the first grid with a scale found by least squared error, which shrinks 3 to 4 percent of blocks. This round gives it the rule that works for scalar rounding. `VE8G` at q levels is the E8 Voronoi code with one scale per group G, the smallest at which no block of the group is shrunk. `VE832` at q = 16 is the lattice form of `Q4_0`.

Weight errors were looked at on three matrices before writing this. `VE832` at q = 16 has -22.5 to -22.7 dB against -20.8 to -21.3 dB for `X32`. `VE8R` at q = 32 has -24.2 to -25.1 dB against -20.9 to -22.5 dB for `XR`. No block was shrunk.

### Predictions

| | Prediction | Fails if |
|---|---|---|
| C1 | With the scale set this way the lattice gives a closer output at equal bits: `VE832` at q = 16 has a lower mean KL than `X32` at q = 16, and `VE8R` lower than `XR` at q = 32 and at q = 16. | any of the three goes the other way |
| C2 | The gain is in proportion: in each of the three pairs the ratio of mean KL in dB is between 0.5 and 1.5 times the gain in mean weight error in dB. | any ratio is outside the range |

## Round 5, added after the first clipping probe

Written after these further results. The first variant of this round was started a minute before this was committed and had not been scored.

| Variant | Bits | Weight error, mean | Mean KL | Top token kept |
|---|---|---|---|---|
| `VE832` at q = 16 | 4.500 | -22.58 dB | 0.075 | 91.57 % |
| `VE8R` at q = 32 | 5.005 | -24.94 dB | 0.059 | 92.32 % |
| `XR` at q = 32, step times 0.5 and 0.35 | 5.005 | -16.28 and -11.05 dB | 0.502 and 1.867 | |
| `XR` at q = 16 | 4.005 | -16.36 dB | 0.448 | 79.98 % |
| `X32` at q = 32 | 5.500 | -27.13 dB | 0.031 | 94.79 % |
| `B32+S` at q = 16 and 32 | 4.5 and 5.5 | -19.35 and -24.49 dB | 0.365 and 0.131 | |
| `R+S` at q = 16 | 4.005 | -18.47 dB | 0.662 | 78.50 % |
| clip only, beyond 3.45 standard deviations of the row | | -34.50 dB | 0.318 | 85.04 % |

The clip-only probe rounds nothing. It pulls back the 0.17 percent of entries that the searched step of `R+S` clips. Its weight error is 12 dB smaller than that of `XR` at q = 32 and its KL is 2.7 times larger. An error that shrinks the largest weights does far more damage than the same amount of error spread as noise.

### What this round runs

- `flip:3.45`: the same entries as the clip probe, moved by the same amounts, each toward or away from zero at random. The squared error is the same. Only the direction differs.
- A spectral error for 26 variants on the 42 matrices of layers 0, 8, 16, 24, 32 and 40. It is the squared largest singular value of the weight change over the squared largest singular value of the matrix, as a mean in dB.

Noise spreads over all directions and has a small largest singular value for its size. A change with structure does not.

The 26 variants are every rounding variant scored so far at 4 to 5.5 bits with a scale per matrix, row or block, and the clip-only probes. The two symmetry arms are left out, because their error is measured against moved matrices.

### Predictions

| | Prediction | Fails if |
|---|---|---|
| D1 | The direction of the error carries the damage, not the choice of entries: `flip:3.45` has a mean KL under half that of `clip:3.45`. | it is half or more |
| D2 | The spectral error ranks the 26 variants by mean KL with a rank correlation of at least 0.8, above that of the squared error on the same matrices, and it puts `clip:3.45` above `XR` at q = 32. | any of the three parts fails |
