# E8 turned by a Hadamard matrix, on real weights

## Question

Draft pull request 24 proposes E8 turned by the Hadamard matrix of order 8 over the root of 8, and shows on made-up matrices that it can leave small weights at zero. Issue 25 lists the test on real weights as step 1, not yet run. The record lattice-split-smollm2-2026-11-17 says E8 loses SmolLM2 because it cannot do that.

Does turning E8 give it what integer levels have on SmolLM2, and keep what E8 has on Qwen2.5? Is the turned lattice the one arm for all three models? And does turning help E8 in a Voronoi code, the kind of code a branchless closest-point routine (step 2 of issue 25) speeds up?

## Status

`complete`, 2026-11-23. 15 scored rows and 7 weights-only rows. 11 of 12 predictions hold, and the check holds.

## Preregistration

[PREREG.md](PREREG.md), committed and pushed at `3f21683`, with its second part (the Voronoi arms) at `1aeca0b`, before any matrix of a model was rounded by the turned lattice and before anything was scored. The run started at `7f81554` (`results.start`). `run.py` and `summary.py` were not changed after.

## Result in short

All figures are mean KL against the 16-bit model on the wikitext-2 test text, 10,200 positions, no calibration data and no forward pass for any arm.

1. **E8 turned is the lowest arm, or within 2 errors of the lowest, at 4.5 bits on all three models.** Qwen2.5 0.018875, SmolLM2 0.051522, Llama-3.2 0.036038. E8 as usually written has 0.019549, 0.060861 and 0.040161. Integer levels have 0.022750, 0.055962 and 0.047808.
2. **Turning is what E8 needed on SmolLM2.** There it is 5.4 errors under plain E8 and 3.0 errors under integer levels. I rated the second at low confidence. The weights say why: of the weights under 0.05 of their matrix's root mean square, plain E8 stores 60.7 percent as zero and the turned lattice 85.9, integer levels 99.7.
3. **The weight error does not change.** -25.585 dB for both E8 arms on SmolLM2 and Llama-3.2, -25.587 and -25.586 on Qwen2.5. Turning moves where the error falls, not how much there is.
4. **With no data and no kept columns the turned lattice is ahead of `Q4_K_M` on all three models**: by 18.2, 17.1 and 8.9 errors. Llama-3.2 is the model where the integer arm was behind (4.9 errors). Plain E8 is ahead there too, by 3.4.
5. **With ten columns of each `ffn_down` kept on Llama-3.2, turned and plain E8 are tied** (0.3 errors), both well ahead of integer levels with the same columns, 25.6 and 25.4 errors under `Q4_K_M` against 17.3.
6. **At 3.0 bits there is no single winner.** On Qwen2.5 plain E8 is lowest (0.169338), the turned lattice is 2.1 errors above it, integer levels far behind (0.215147). On SmolLM2 integer levels are far ahead of both E8 arms: 0.419643 against 0.629902 turned and 0.664189 plain, which is 20.2 errors. Not predicted.
7. **In a Voronoi code, turning is not enough, and it is not the same on both models.** On SmolLM2 it helps, 0.110103 against 0.124728, 4.9 errors. On Qwen2.5 it hurts, 0.046466 against 0.043414, 2.2 errors the wrong way (H11 fails). Both stay behind `Q4_K_M`, by 10.7 and 10.6 errors.

## What was run

**Models, text and score** are those of the earlier records: `Qwen2.5-1.5B-Instruct`, `SmolLM2-1.7B-Instruct` and `Llama-3.2-1B-Instruct`, 16-bit files, mean KL against the reference's saved outputs, `llama-perplexity --kl-divergence` of llama.cpp at `eada89c`.

**The lattice.** `turned.py`: the Hadamard matrix of order 8 by doubling, over the root of 8, applied to every block of 8 consecutive weights of a row. It is symmetric and its own inverse. `test_turn.py` holds the lattice to the pull request's description by a decoder written another way: its points times the root of 2 are whole, their remainders form the extended Hamming code, and its 240 shortest points are 16 on an axis and 224 on four coordinates.

**The arms.** Integer levels (`ZBG`), E8 (`EBG`) and E8 turned (`HBG`), each in one ball per group of 128 weights with a 16-bit scale, at 4.5 bits (`b560`) or 3.0 (`b368`). `+K10` keeps ten loud columns of each `ffn_down`, by the function of loud-columns-2026-11-18. The Voronoi arms `VE8Z32:16` and `VH8Z32:16` are the harness's code of E8 with a codeword at zero, 16 points a weight and a scale per 32 weights, plain and turned. The plain arms are the base program's code, and a test holds them to it bit for bit.

**The check C1.** E8 on SmolLM2 scored again through this program: 0.060861, the committed row's value, difference +0.000000 where a tenth of a standard error is 0.000135.

**Weights only.** Each row has the share of weights stored as zero and, for weights under 0.05, 0.1 and 0.2 of their matrix's root mean square, their share, how many are stored as zero and their squared error over their squared size. 7 more arms of earlier records were built without scoring to have the same figures.

## Results

Every number is in `results.out`, printed by `summary.py` from `results.jsonl`, `metrics.jsonl` and the rows of earlier records. The table is at 4.5 bits.

| Model | Integer levels | E8 | E8 turned | `Q4_K_M` at 4.8 bits |
|---|---|---|---|---|
| Qwen2.5 | 0.022750 | 0.019549 | 0.018875 | 0.033348 |
| SmolLM2 | 0.055962 | 0.060861 | 0.051522 | 0.083761 |
| Llama-3.2 | 0.047808 | 0.040161 | 0.036038 | 0.043179 |
| Llama-3.2, ten columns kept | 0.030223 | 0.025378 | 0.025204 | |

### The predictions

| | Prediction | Result | Holds |
|---|---|---|---|
| H1 | SmolLM2 4.5: turned under E8 by 2 errors | 5.4 | yes |
| H2 | SmolLM2 4.5: turned under integer levels by 2 errors | 3.0 | yes |
| H3 | Qwen2.5 4.5: turned within 2 of E8, under integer levels by 2 | 1.1 and 5.2 | yes |
| H4 | Llama-3.2 4.5 with columns kept: turned not above E8 by 2 | 0.3 under | yes |
| H5 | the same: turned under integer levels by 2 | 8.8 | yes |
| H6 | turned lowest or within 2 of the lowest, three cells at 4.5 bits | all three | yes |
| H7 | weight error within 0.1 dB of E8's in five cells | yes | yes |
| H8 | 15 points more of the weights under 0.05 stored as zero than E8, three models | 24.6, 25.2 and 24.8 points | yes |
| H9 | SmolLM2 3.0: turned under E8 by 2 errors | 2.9 | yes |
| H10 | Voronoi, SmolLM2: turned under plain by 2 | 4.9 | yes |
| H11 | Voronoi, Qwen2.5: turned not above plain by 2 | 2.2 above | **no** |
| H12 | Voronoi: turned still not under `Q4_K_M` by 2, both models | 10.7 and 10.6 behind | yes |

I put low confidence on H2 and H6, and both held. H11, which I thought fair, failed.

## Deviations from the plan

- **The run took three hours and forty minutes, not an hour and three quarters.** A turned or Voronoi build takes 6 to 16 minutes (`build_seconds` in each row), more than the minute and a half I planned for. No row differs because of it.
- Nothing else. The machine did not restart during the run.

## Limits

- **Three models between 1 and 2 billion weights, KL on one text, no task, no calibration data.** The Python text and the matrix-weighted arms were not run with the turned lattice.
- **Counted bits, no file.** The packed format of this repository holds integer levels only. No E8 arm, turned or not, has a file or a decoder, so nothing here says how fast the turned lattice decodes or whether its ball can be numbered cheaply.
- **The Voronoi arms are scored on two models**, not on Llama-3.2.
- **Blocks of 8 are consecutive weights of a row.** Another grouping was not tried.
- **The kept columns are set to zero inside their block**, as in the loud-columns record, so the arms differ in the lattice only. The 4.47 bits of the pull request come from packing them apart.
- The standard errors are the tool's and treat positions as independent. A gap of 2 or 3 errors is a tie.

## Read against the closure rule

Nothing is closed by a certificate. These are 22 builds on three models.

**What changes in the repository's account.**

1. The statement that E8 loses SmolLM2 because of what it does to small weights is supported from the other side: turn the lattice so it can keep them at zero and it wins there, with the weight error unchanged.
2. Integer levels at 4.5 bits are no longer the arm to beat on any of the three models. The turned lattice is at least as good on all three, with no data.
3. At 3.0 bits that does not carry over. On SmolLM2 the arm that rounds most weights to zero is far ahead.

**What follows.**

1. A format for the turned lattice. The ball has to be numbered and the numbers decoded. `shape/ballindex.py` numbers the points of an E8 ball, and turning is a change of basis on top of it, so this is engineering and not new mathematics, but nobody has done it.
2. The Voronoi code needs more than turning to reach `Q4_K_M`. Another rule for the scale or kept columns are the candidates, and issue 25 has the step that would use it.
3. Step 1 of issue 25 on Gemma-2B, and the matrix-weighted arms with the turned lattice.
4. Why the turned lattice hurts the Voronoi code on Qwen2.5 and helps it on SmolLM2.

## How to rerun

Needs NumPy, torch, llama.cpp's `gguf` package and `llama-perplexity`, the model files and their reference outputs.

```bash
python3 -m unittest experiments/e8-hadamard-real-weights-2026-11-22/test_turn.py
```

```bash
python3 experiments/e8-hadamard-real-weights-2026-11-22/run.py --model MODEL.gguf --kld base.kld --work DIR --out results.jsonl --bin llama-perplexity 'EBG128:b560s16' 'HBG128:b560s16'
```

```bash
python3 experiments/e8-hadamard-real-weights-2026-11-22/run.py --model MODEL.gguf --metrics-only --out metrics.jsonl 'ZBG128:b560s16'
```

```bash
python3 experiments/e8-hadamard-real-weights-2026-11-22/summary.py > results.out
```

A scored row takes 6 to 20 minutes on 4 CPU cores.

## Disclosure

Written by an AI agent session under the owner's direction, on 2026-11-23. The plan, the program, the run and this record are the session's. The turned lattice is the proposal of draft pull request 24, which this session did not write. The scoring tool and `Q4_K_M` are llama.cpp's.
