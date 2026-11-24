# Lattices from extended Hamming codes, on real weights, and two baselines scored like the ball arms

## Question

The record e8-hadamard-real-weights-2026-11-22 found E8 turned by the Hadamard matrix of order 8 (E8_Had) the lowest arm at 4.5 bits on three models. E8_Had over the root of 2 is A_8 = C_8 + 2Z^8, with C_8 the extended Hamming code [8, 4, 4].

The longer codes [16, 11, 4] and [32, 26, 4] give lattices A_16 and A_32 the same way. Their cells are worth 0.12 dB more than E8's, and they round more small weights to zero. Do they score better on real models, at 4.5 and at 3.0 bits?

Two baselines are also scored the way the ball arms are:
- `Q4_K_M` decoded to 16 bits, so that llama.cpp no longer rounds the activations to 8 bits;
- bitsandbytes' NF4.

## Status

`complete`, 2026-11-24. 11 scored rows of the new lattices, the check row, and 6 baseline rows. 14 of 15 predictions hold, and the check holds.

## Preregistration

[PREREG.md](PREREG.md) was committed and pushed at `aec439b`, before any matrix of a model was rounded by the new lattices. `summary.py` was committed at `439628d`, before any row was read. Two outputs were committed with the plan, both from made-up data, and the plan quotes them: `cell_check.out` (the lattices' cells, by brute force over all codewords) and `pilot.out` (a 512 × 4096 Laplace matrix).

The run started at `aec439b` and was started again three times after the machine restarted (`results.start`). `run.py`, `baselines.py` and `summary.py` were not changed after.

## Result in short

All figures are mean KL against the 16-bit model on the wikitext-2 test text, 10,200 positions. No calibration data and no forward pass for any arm.

1. **At 4.5 bits A_16 is the arm to use with no data.** It is the lowest arm, or 0.2 errors from the lowest, on all three models:

   | Model | A_16 | E8_Had | A_16 against E8_Had |
   |---|---|---|---|
   | Qwen2.5 | 0.018974 | 0.018875 | 0.2 errors above, a tie |
   | SmolLM2 | 0.047957 | 0.051522 | 2.4 errors under |
   | Llama-3.2 | 0.034455 | 0.036038 | 2.2 errors under |

   - Its weight error is −25.707 dB on all three models, 0.12 dB under E8's.
   - It rounds 93.4 to 95.6 percent of the smallest weights (under 0.05 of their matrix's root mean square) to zero. E8_Had rounds 83.7 to 85.9 percent.
2. **A_32 is not the arm to use.**
   - It is the lowest arm on SmolLM2: 0.045800, 4.2 errors under E8_Had.
   - It is 4.3 errors above E8_Had on Llama-3.2: 0.039540.
   - It rounds the most small weights to zero, 97.3 to 99.0 percent. More zeros did not help everywhere. H2 fails.
3. **At 3.0 bits a lattice closes the gap on SmolLM2.** There integer levels had been far ahead of every E8 arm: 0.419643 against 0.629902 for E8_Had.
   - A_32 scores 0.358845 and A_16 0.380994: 7.5 and 4.7 errors under integer levels, and 27.3 and 24.6 under E8_Had.
   - On Qwen2.5, A_16 scores 0.160368, 2.2 errors under plain E8 (0.169338), which had been the lowest arm. A_32 there scores 0.174194, 1.2 errors above plain E8.
   - So at 3.0 bits A_16 is ahead of every earlier arm on both models: the lowest on Qwen2.5, second to A_32 on SmolLM2.
4. **With ten columns of each `ffn_down` kept, on Llama-3.2:** A_16 scores 0.024053, 2.4 errors under E8_Had with the same columns kept (0.025204). It is the lowest arm with no data that this repository has scored on that model.
5. **The committed `Q4_K_M` rows were scored less favourably than the ball arms.** Decoded to 16 bits and multiplied as the ball arms are, `Q4_K_M` scores:

   | Model | `Q4_K_M@16` | Committed `Q4_K_M` | Gap |
   |---|---|---|---|
   | Qwen2.5 | 0.031690 | 0.033348 | 1.8 errors, a tie |
   | SmolLM2 | 0.072237 | 0.083761 | 5.6 errors |
   | Llama-3.2 | 0.041195 | 0.043179 | 2.4 errors |

   - Every lead of a ball arm over `Q4_K_M` holds against `Q4_K_M@16`. A_16 is ahead by 17.7, 14.3 and 8.9 errors.
   - The ball arms ran on 16-bit activations too. What 8-bit activations cost them is not measured here.
6. **NF4 is far behind.** Blocks of 64 at 4.5 bits score 0.064678, 0.113625 and 0.110765.
   - That is 26.5 to 46.4 errors above E8_Had.
   - It is also above `Q4_K_M@16` on all three models.

## What was run

**Models, text and score** are those of the earlier records:
- `Qwen2.5-1.5B-Instruct`, `SmolLM2-1.7B-Instruct` and `Llama-3.2-1B-Instruct`, 16-bit files;
- mean KL against the reference's saved outputs, by `llama-perplexity --kl-divergence` of llama.cpp at `eada89c`.

**The lattices** are in `hamming.py`.
- **Points.** A_n is the set of whole-number vectors whose parities form a word of the extended Hamming code of length n = 2^m. Coordinate i has the check vector i | n.
- **Nearest point.** Found exactly by a dynamic program over the 2^(m+1) partial syndromes, in C (`hamming_dp.c`). The C code is held bit for bit to a NumPy version, and both to a search over all codewords.
- **Counting the ball.** The points in a ball are counted exactly from the code's weight enumerator (MacWilliams, from the dual code RM(1, m)) and the theta series of the odd and even integers.
- **Tests.** `test_hamming.py` has 16 tests. They check that:
  - A_8 over the root of 2 is E8_Had;
  - its ball bound is E8's doubled;
  - the ball arm of this program equals the base program's `group_ball_round` bit for bit on the integers and on E8.

**The arms.** `A16BG128` and `A32BG128` are A_16 or A_32 repeated over a group of 128 weights, cut by the largest ball with at most 2^560 points (4.5 bits a weight) or 2^368 points (3.0 bits), with one 16-bit scale per group. The scale and the rounding are the base program's. `+K10` keeps the ten loud columns of each `ffn_down`, as in the earlier records.

**The check C1.** `A8BG128:b560s16` on SmolLM2 is E8_Had decoded another way:
- mean KL 0.051598 against the committed 0.051522;
- the difference, 0.000076, is under a tenth of a standard error, 0.000106;
- the weight error is the same to 0.0000 dB.

**The baselines** are in `baselines.py`.
- **`Q4_K_M@16`.** The `Q4_K_M` file the earlier records scored, checked by SHA-256, with every block matrix decoded to 16-bit floats and put in the 16-bit model.
- **`NF4B64`.** bitsandbytes' NF4 with its defaults: blocks of 64, the block's largest size in a 32-bit float, 4.5 bits a weight.

## Results

Every number is in `results.out`, printed by `summary.py` from `results.jsonl`, `baselines.jsonl` and the rows of the earlier records.

**Mean KL at 4.5 bits:**

| Model | Integer levels | E8 | E8_Had | A_16 | A_32 | `Q4_K_M` (4.8 bits) | `Q4_K_M@16` | NF4 |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5 | 0.022750 | 0.019549 | 0.018875 | 0.018974 | 0.019032 | 0.033348 | 0.031690 | 0.064678 |
| SmolLM2 | 0.055962 | 0.060861 | 0.051522 | 0.047957 | 0.045800 | 0.083761 | 0.072237 | 0.113625 |
| Llama-3.2 | 0.047808 | 0.040161 | 0.036038 | 0.034455 | 0.039540 | 0.043179 | 0.041195 | 0.110765 |

**Mean KL at 3.0 bits:**

| Model | Integer levels | E8 | E8_Had | A_16 | A_32 |
|---|---|---|---|---|---|
| Qwen2.5 | 0.215147 | 0.169338 | 0.178526 | 0.160368 | 0.174194 |
| SmolLM2 | 0.419643 | 0.664189 | 0.629902 | 0.380994 | 0.358845 |

**Llama-3.2 with ten columns kept, 4.5 bits:** integer levels 0.030223, E8 0.025378, E8_Had 0.025204, A_16 0.024053.

### The predictions

| | Prediction | Result | Holds |
|---|---|---|---|
| C1 | A_8 reproduces the committed E8_Had row on SmolLM2 | 0.000076 apart, a tenth of an error is 0.000106 | yes |
| H1 | 4.5: A_16 not above E8_Had by 2, three models | −0.2, +2.4, +2.2 | yes |
| H2 | 4.5: A_32 not above E8_Had by 2, three models | −0.3, +4.2, −4.3 | **no** |
| H3 | 4.5, SmolLM2: A_16 under E8_Had by 2 (low confidence) | +2.4 | yes |
| H4 | 4.5: the lowest of five arms is A_16 or A_32 on two models of three | E8_Had, A_32, A_16 | yes |
| H5 | 4.5: weight error of A_16 under E8_Had's by 0.05 dB, of A_32 by 0.02 | 0.120 and 0.039, 0.123 and 0.042, 0.122 and 0.043 | yes |
| H6 | 4.5: smallest weights at zero, A_16 at least E8_Had + 5 points, A_32 at least A_16 | 83.7, 93.4, 97.3; 85.9, 95.6, 99.0; 84.5, 94.3, 98.0 | yes |
| H7 | Llama-3.2 with columns kept: A_16 not above E8_Had by 2 | +2.4 | yes |
| H8 | 3.0, SmolLM2: A_16 and A_32 under E8_Had by 2 | +24.6, +27.3 | yes |
| H9 | 3.0, SmolLM2: A_32 not above integer levels by 2 (low confidence) | +7.5 | yes |
| H10 | 3.0, Qwen2.5: A_16 not above plain E8 by 2 | +2.2 | yes |
| H11 | 3.0: weight error of A_16 and A_32 under integer levels' by 0.4 dB and E8_Had's by 0.03 | A_16 0.794 and 0.137, A_32 0.705 and 0.048 (Qwen2.5); A_16 0.792 and 0.138, A_32 0.705 and 0.051 (SmolLM2) | yes |
| B1 | `Q4_K_M@16` under the committed `Q4_K_M` row, Qwen2.5 and SmolLM2 | +1.8, +5.6 | yes |
| B2 | E8_Had under `Q4_K_M@16` by 2 | +17.4, +12.0 | yes |
| B3 | NF4 above E8_Had by 10 | +33.3, +26.5 | yes |
| B4 | NF4 above `Q4_K_M@16` | +22.7, +16.6 | yes |

Positive gaps: the first arm named is lower. I put low confidence on H3 and H9, and both held. H2, which I thought likely, failed on Llama-3.2.

## Deviations from the plan

- **The machine restarted three times during the run.** Each time this session had gone idle with no task of its own in the background, and the machine was reclaimed.
  - At about 03:01 UTC, during the build of `A32BG128:b560s16` on Qwen2.5. Started again at 05:02.
  - At about 07:35, during the build of `A32BG128:b368s16` on Qwen2.5. Started again at 07:36.
  - Between 07:38 and 08:20, during the same build. Started again at 08:23.

  Each time the program skipped the rows already in the results file and built the interrupted row from the start. No finished row was lost or scored twice. The restarts are noted in `results.start`. From 08:23 the session kept a watch running, so that it would not go idle.
- **The watcher's first commit is mislabelled.** `5a62d0e` says "e8-hadamard" where it should say "hamming-lattices": a slip in the script, fixed after. It was already pushed, so it was not rewritten.
- Nothing else. A build took 8.6 to 20.3 minutes on 4 cores (`build_seconds`), against the 20 to 35 minutes a model the pilot gave.

## Limits

- **Three models of 1 to 2 billion weights, KL on one text, no task, no calibration data.** The Python text and the arms weighted by an importance matrix were not run with the new lattices.
- **Counted bits, no file.** `shape/hammingindex.py`, written during the run, numbers the points of these balls in exactly the counted bits, and `shape/hampack.py` stores A_16 in the packed file's block of 32 weights. No arm scored here was read back from a file. The A_16 file has its own plan, `experiments/packed-hamming-2026-11-24`.
- **Kept columns were run on Llama-3.2 only.** A_32 was run with them on no model.
- **`Q4_K_M@16` removes the 8-bit activations from one side.** The fair comparison for an engine puts both sides on the same activations, and the ball arms on 8-bit activations were not scored.
- The standard errors are the tool's and treat positions as independent. A gap of 2 or 3 errors is a tie.

## Read against the closure rule

Nothing is closed by a certificate. These are 12 builds on three models and 6 baselines.

**What changes in the repository's account.**

1. **The arm to recommend with no data at 4.5 bits is A_16** in one ball per 128 weights. It is the lowest arm, or tied with it, on all three models. E8_Had was that arm before.
2. **At 3.0 bits A_16 is ahead of every earlier arm on both models scored, and A_32 leads on SmolLM2.** On SmolLM2 the integer levels led the lattices by rounding small weights to zero. A lattice that does the same with a better cell takes the lead.
3. **Every comparison of a ball arm with llama.cpp's `Q4_K_M` should be read against `Q4_K_M@16`.** The leads shrink by 0.8 to 5.4 errors and all hold (`notes/analysis-2026-11-24/arms_table.out`).

**What follows.**

1. Score the A_16 file (`experiments/packed-hamming-2026-11-24`, planned and queued).
2. Kept columns with A_16 on Qwen2.5 and SmolLM2.
3. The ball arms with 8-bit activations, for the comparison an engine would see.
4. Why A_32 loses on Llama-3.2 when it wins on SmolLM2.

## How to rerun

Needs NumPy, torch, llama.cpp's `gguf` package and `llama-perplexity`, a C compiler, the model files and their reference outputs.

```bash
python3 -m unittest experiments/hamming-lattices-real-weights-2026-11-24/test_hamming.py
```

```bash
python3 experiments/hamming-lattices-real-weights-2026-11-24/run.py --model MODEL.gguf --kld base.kld --work DIR --out results.jsonl --bin llama-perplexity 'A16BG128:b560s16' 'A32BG128:b560s16'
```

```bash
python3 experiments/hamming-lattices-real-weights-2026-11-24/baselines.py --model MODEL.gguf --q4km MODEL-Q4_K_M.gguf --kld base.kld --work DIR --out baselines.jsonl --bin llama-perplexity 'Q4_K_M@16' NF4B64
```

```bash
python3 experiments/hamming-lattices-real-weights-2026-11-24/summary.py > results.out
```

A scored row takes 10 to 25 minutes on 4 CPU cores.

## Disclosure

Written by an AI agent session under the owner's direction, on 2026-11-24. The plan, the program, the run and this record are the session's.
- The lattices are textbook: Construction A in Conway and Sloane, *Sphere Packings, Lattices and Groups*, ch. 5.
- Their use here was proposed by a survey agent of this session reading Foundation-0's code tools.
- The point about 8-bit activations comes from Moissanite-Labs/ml-os, and NF4 from Moissanite-Labs/hillclimb.
- The scoring tool and `Q4_K_M` are llama.cpp's.
