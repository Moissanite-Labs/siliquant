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
