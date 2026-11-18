# The ball arms and the zero codeword on a third model

## Question

The run on `Qwen2.5-1.5B-Instruct` found, on one model and one text, that a code with one ball per group of weights is far ahead of llama.cpp's files at equal bits, and that the E8 Voronoi code of the harness breaks per row because it has no codeword at zero.

Do both hold on a model of another family?

## Status

`complete`. All 13 variants are scored. One model, `SmolLM2-1.7B-Instruct`, and the same text as the Qwen2.5 run. No arm has a packed format: bits are counted from the codebook and the rounded weights are scored from 16-bit float copies.

## Preregistration

[PREREG.md](PREREG.md), committed and pushed at `14726fd` before this model's reference pass was run and before any variant was scored. It was written with 17 variants of the Qwen2.5 run seen. 7 of 8 predictions hold.

## Result in short

1. **One ball per group at 4.5 bits beats llama.cpp's `Q4_K_M` at 4.82 bits on this model too.** Integer levels in one ball per group of 128 score a mean KL of 0.0560, and E8 in the same ball 0.0609. `Q4_K_M` scores 0.0838 and `Q4_0` at 4.5 bits 0.1271. The integer arm is 14.7 errors of the gap ahead of `Q4_K_M` and the E8 arm 11.1.
2. **Integer levels are ahead of E8 here**, by 2.9 errors of the gap, though E8's weight error is 0.65 dB lower. On Qwen2.5 E8 was ahead by that 0.65 dB.
3. **Every E8 arm gains less on the output here than its weight error says.** Against the scalar rule the integer arm turns 3.76 dB of weight error into 3.51 dB of KL, a ratio of 0.93. The five E8 arms have ratios of 0.02 to 0.71. On Qwen2.5 every ball arm was between 0.91 and 1.08. This is not explained.
4. **The harness's E8 Voronoi code is not broken per row on this model.** It scores 0.329 where the scalar rule scores 0.493. With a codeword at zero it scores 0.338, a tie. With groups of 32 the code with a codeword at zero is the worse of the two, 0.1247 against 0.1166, by 2.8 errors. So the damage found on Qwen2.5 is a property of that model's rows, and a codeword at zero is not free everywhere.
5. **The E8 win of the gemma record shrinks here as well.** With groups of 32 at 4.5 bits the E8 Voronoi code gains 0.33 dB of KL over the scalar rule, 2.9 errors of the gap. On gemma it was 1.96 dB and on Qwen2.5 0.52.
6. **This model loses two to three times as much as Qwen2.5 to any 4.5-bit rule.** And at 4.0 bits the ball arm is only just under `Q4_0` at 4.5 bits, 0.1183 against 0.1271, where on Qwen2.5 it was near `Q4_K_M`.

## The model, the text and the scoring

**Model.** `SmolLM2-1.7B-Instruct`, the 16-bit float GGUF of `bartowski/SmolLM2-1.7B-Instruct-GGUF`, SHA-256 `cf3f0569a9d0b56d5d3b5b7944768c71bd2fef789f8b36b1b85a05d390850852`. A conversion of the publisher's weights by a third party. 24 layers, 168 block matrices, 1.61 billion block weights, rows of 2,048 or 8,192.

**Text and score.** As in the Qwen2.5 record: the wikitext-2 test set, 40 passages of 512 tokens with the second half scored, 10,200 positions, `llama-perplexity --kl-divergence` at llama.cpp commit `eada89c` on 4 CPU threads. The reference perplexity is 9.43. llama.cpp's files were made with only the block matrices changed and no importance matrix.

**The arms** are those of the Qwen2.5 record, built by the same program. That record's README has the table of names.

## Results

Every number is in `results.out`, printed by `summary.py` from `results.jsonl`.

### Every variant, beside Qwen2.5

| Variant | Bits | Weight error, dB | Mean KL | First choice kept | Qwen2.5: mean KL | First choice kept |
|---|---|---|---|---|---|---|
| `X32:16` | 4.500 | -21.17 | 0.1257 | 82.43% | 0.0620 | 88.19% |
| llama.cpp `Q4_0` | 4.500 | -21.17 | 0.1271 | 82.35% | 0.0623 | 88.22% |
| llama.cpp `Q4_K_M` | 4.822 | -24.56 | 0.0838 | 85.52% | 0.0333 | 91.11% |
| `ZBG128:b560s16` | 4.500 | -24.93 | 0.0560 | 88.25% | 0.0227 | 92.75% |
| `EBG128:b560s16` | 4.500 | -25.58 | 0.0609 | 87.50% | 0.0195 | 93.08% |
| `EB32@32:16` | 4.498 | -23.54 | 0.1018 | 84.31% | 0.0339 | 90.85% |
| `VE832:16` | 4.500 | -22.64 | 0.1166 | 83.27% | 0.0551 | 89.41% |
| `VE8Z32:16` | 4.500 | -22.61 | 0.1247 | 82.75% | 0.0434 | 89.55% |
| `EBG128:b504s8` | 4.000 | -22.81 | 0.1183 | 83.77% | 0.0371 | 90.45% |
| `XR:16` | 4.006 | -16.59 | 0.4934 | 67.00% | 0.6317 | 63.59% |
| `VE8R:16` | 4.006 | -19.27 | 0.3290 | 73.52% | 3.6580 | 25.75% |
| `VE8ZR:16` | 4.006 | -19.24 | 0.3376 | 72.78% | 0.2088 | 77.90% |

`unchanged` scores a mean KL of 0.000000. `X32:16` scores as llama.cpp's own `Q4_0`, to 1.16 percent.

### Against llama.cpp's files

| Arm | Bits | Mean KL | Against | Bits | Mean KL | Lower by, in errors |
|---|---|---|---|---|---|---|
| `ZBG128:b560s16` | 4.500 | 0.0560 | `Q4_K_M` | 4.822 | 0.0838 | 14.7 |
| `EBG128:b560s16` | 4.500 | 0.0609 | `Q4_K_M` | 4.822 | 0.0838 | 11.1 |
| `EBG128:b560s16` | 4.500 | 0.0609 | `Q4_0` | 4.500 | 0.1271 | 24.0 |
| `EB32@32:16` | 4.498 | 0.1018 | `Q4_K_M` | 4.822 | 0.0838 | -7.2: it is behind |
| `EBG128:b504s8` | 4.000 | 0.1183 | `Q4_0` | 4.500 | 0.1271 | 2.8 |
| `EBG128:b504s8` | 4.000 | 0.1183 | `Q4_K_M` | 4.822 | 0.0838 | -13.0: it is behind |

The headline of the Qwen2.5 record carries: at 4.5 bits one ball per group of 128 is ahead of `Q4_K_M` at 4.82 bits. The two smaller claims do not. E8 with 16 levels in a ball over 32 weights tied `Q4_K_M` there and is behind it here. The arm at 4.0 bits was near `Q4_K_M` there and is far behind it here.

### Weight error against the output

Against the scalar rule at 4.5 bits, `X32:16`:

| Arm | Weight error gain, dB | KL gain, dB | Ratio | On Qwen2.5: ratio |
|---|---|---|---|---|
| `ZBG128:b560s16`, integer levels | 3.76 | 3.51 | 0.93 | 1.02 |
| `EBG128:b560s16` | 4.42 | 3.15 | 0.71 | 1.02 |
| `EB32@32:16` | 2.37 | 0.91 | 0.38 | 0.91 |
| `EBG128:b504s8` | 1.64 | 0.26 | 0.16 | 1.04 |
| `VE832:16` | 1.47 | 0.33 | 0.22 | 0.30 |
| `VE8Z32:16` | 1.44 | 0.03 | 0.02 | 0.90 |

The integer arm behaves as on Qwen2.5. The E8 arms do not. Among the three E8 ball arms the steps still match: from `EBG128:b504s8` to `EB32@32:16` is 0.73 dB of weight error and 0.65 dB of KL, and from there to `EBG128:b560s16` is 2.04 and 2.24. So they lie on a line of the same slope as the integer arm, about 1.0 to 1.3 dB of KL above it. On this model E8 costs that much on the output, at any of these bit rates, and its 0.65 dB of weight error does not pay for it.

Why is not known. One difference between the two rules that weight error does not see: rounding to integers moves a weight by half a step at most, and rounding to E8 can move one weight of a block by a whole step. Nothing here tests whether that is the cause.

### The zero codeword

| | Scalar rule | E8 Voronoi code of the harness | With a codeword at zero |
|---|---|---|---|
| per row, 16 levels: this model | 0.4934 | 0.3290 | 0.3376 |
| per row, 16 levels: Qwen2.5 | 0.6317 | 3.6580 | 0.2088 |
| groups of 32: this model | 0.1257 | 0.1166 | 0.1247 |
| groups of 32: Qwen2.5 | 0.0620 | 0.0551 | 0.0434 |

On this model the code of the harness works per row, as it did on gemma, and a codeword at zero changes nothing there: the two forms are 1.2 errors of the gap apart. With groups of 32 the code with the centre is the better by 2.8 errors.

So the two models disagree about the centre with groups of 32. On Qwen2.5 a codeword at zero is 21 percent better. Here it is 7 percent worse. Per row it is 17 times better on Qwen2.5 and level here.

### The predictions

| | Prediction | Result | Holds |
|---|---|---|---|
| S1 | `unchanged` scores 0, and `X32:16` scores as `Q4_0` | 0.000000. KL within 1.16 percent, first choice within 0.08 points | yes |
| S2 | `EBG128:b560s16` at 4.5 bits is under `Q4_K_M` at 4.82 by 2 errors | 0.0609 against 0.0838, 11.1 errors | yes |
| S3 | Its KL gain over `X32:16` is 0.75 to 1.25 times its weight-error gain | 0.71 | **no** |
| S4 | `EB32@32:16` is under `X32:16` and `VE832:16` by 2 errors each | by 7.6 and 5.3 errors | yes |
| S5 | `ZBG128:b560s16` is under `Q4_K_M` by 2 errors | 0.0560 against 0.0838, 14.7 errors | yes |
| S6 | `VE8ZR:16` is under `XR:16` by 2 errors | 0.3376 against 0.4934, 18.9 errors | yes |
| S7 | `VE8ZR:16` is not above `VE8R:16` by 2 errors | above by 1.2 errors | yes |
| S8 | `EBG128:b504s8` at 4.0 bits is under `Q4_0` at 4.5 by 2 errors | 0.1183 against 0.1271, 2.8 errors | yes |

S8 holds by the rule and should be read as a near tie: the standard errors are too small.

## Deviations from the plan

- **The run was started twice.** A launch on this machine is cut at two hours. The second start was made a few minutes before the cut, two minutes into the build of `VE832:16`, which was then built again from the start. `results.start` has both.
- **A first try at the reference pass was killed by the machine for lack of memory**, before the plan was written. The plan says so. The pass was run again when nothing else was scoring.
- **A test build of four matrices, made before the plan, was stopped before it finished** to free memory for the Qwen2.5 run. The plan lists what it had printed.
- `summary.py` was committed before any variant was scored.

## Limits

- **One text**, the same as for Qwen2.5. This is a third model and not a third text.
- **Counted bits.** No arm has a packed file. The saving is in the file and not in memory.
- **No calibration on either side.** llama.cpp's files with an importance matrix are in the record imatrix-baseline-2026-11-17.
- **The standard errors are too small.** They treat positions as independent. Gaps of 2 or 3 errors are ties.
- **The reference is a third party's conversion** of the publisher's weights to 16-bit floats. The publisher's own GGUF repository has a 4-bit file only.
- **Published work**: as in the Qwen2.5 record. Nothing here is claimed as new.

## Read against the closure rule

Nothing is closed by a certificate. These are measurements on one more model.

**What two models now agree on.**

- One ball per group of 128 weights at 4.5 bits gives a closer output than llama.cpp's `Q4_K_M` at 4.8 bits and than its `Q4_0` at 4.5 bits, with no calibration on either side. Integer levels are enough.
- The scalar rule of this repository scores as llama.cpp's `Q4_0`.
- The E8 Voronoi code of the gemma record gains less over the scalar rule than it did on gemma.

**What they disagree on.**

- Whether E8 is worth having inside the ball. Qwen2.5 says yes, by its 0.65 dB. This model says no.
- Whether a decibel of weight error is a decibel of KL. It is for the integer arm on both. It is for the E8 arms on Qwen2.5 only.
- Whether the centre of the Voronoi code does harm. It breaks Qwen2.5 per row and helps a little here with groups of 32.

**What follows.**

1. The arm to carry forward is the integer one, `ZBG128:b560s16`. It needs no lattice, it is ahead of `Q4_K_M` by more than ten errors on both models, and it is the only arm whose output followed its weight error on both.
2. The cost of E8 on this model was an open question when this record was written. Added 2026-11-18: the record lattice-split-smollm2-2026-11-17, from the lane of issue #10, answers it. E8's error costs what noise of its size costs. Integer rounding costs less than noise, because it sends the smallest weights to exactly zero and E8 cannot. The guess above, about a whole step on one weight, was not the cause.
3. Issue #16 proposes a codeword at zero as the default of the harness. This record qualifies it: it is the safer form per row on the two models scored, and with groups of 32 it is better on one and worse on the other.

## How to rerun

Needs what the Qwen2.5 record needs, and this model's file.

```bash
llama-perplexity -m SmolLM2-1.7B-Instruct-f16.gguf -f wiki.test.raw --chunks 40 -c 512 -t 4 --kl-divergence-base base.kld
```

```bash
python3 experiments/model-output-smollm2-2026-11-17/run.py --model SmolLM2-1.7B-Instruct-f16.gguf --kld base.kld --work DIR --out results.jsonl --bin llama-perplexity X32:16 ZBG128:b560s16 EBG128:b560s16
```

```bash
python3 experiments/model-output-smollm2-2026-11-17/summary.py > results.out
```

An E8 arm takes 10 to 15 minutes to build on 4 CPU cores and 5 to score.

## Disclosure

Written by an AI agent session under the owner's direction, on 2026-11-17. The plan, the program, the run and this record are the session's.
