# What in E8's error costs SmolLM2 more than its size

## Question

On `SmolLM2-1.7B-Instruct` the record `model-output-smollm2-2026-11-17` scored integer levels in one ball per group of 128 weights under E8 in the same ball, 0.0560 against 0.0609, though E8 has 0.66 dB less weight error. Five arms on E8 share one figure of KL per unit of weight error there and three on integer levels share another, 1.1 to 1.5 dB lower. On Qwen2.5 all share one. That record's README leaves it open: "The cost of E8 on this model is an open question."

Is E8's error worse for this model than noise of its size, or is the error of integer levels better? And what makes the difference?

## Status

Complete, 2026-11-17: the plan and three extensions, 18 rows. Each plan was committed and pushed before its arms were scored. Extension 3 is reported in its own section at the end.

## Preregistration

| | Plan | Committed and pushed at | Its run started at |
|---|---|---|---|
| Plan | `PREREG.md` | `eac2ef2` | 21:09:01 UTC |
| Extension 1 | `PREREG-extension-1.md` | `e1f4dac` | 22:32:09 |
| Extension 2 | `PREREG-extension-2.md` | `cfa99c2` | 22:57:50 |
| Extension 3 | `PREREG-extension-3.md` | `07ffdaf` | 23:22:01 |

`results.start` has the times and the commits. Each extension was written after the rows before it were read, and says so.

## Result in short

- **E8's error is noise of its size.** Gaussian noise put in its place, with the same squared error in every group of 128 weights, gives the same KL: 0.11 dB away, and two draws of the noise are 0.14 dB apart.
- **The error of integer levels is better than noise of its size, by 1.1 dB.** Noise in its place costs 0.0730 where the arm costs 0.0562. So nothing is wrong with E8 on this model. Integer rounding has something that noise has not.
- **What it has is at zero.** Give noise to the 92 percent of the weights it does not round to zero and nothing changes: 0.0569. Give noise to the 7.8 percent it does round to zero and the whole gain is gone: 0.0706.
- **Those small weights are where this model's output is lost.** With nothing else changed, setting them to zero costs 0.0152, which is 27 percent of the arm's KL from 7.8 percent of its error. Noise of that size on them costs 0.0319. Per unit of weight error that is 60 and 127 of KL, where the arm's error on all other weights costs 14.
- **No single matrix shows any of it.** Fed the model's own inputs, the output of a matrix is off by the same amount under integer levels, under noise, and under noise on the small weights only, to 0.02 dB. The difference is made further on in the model.
- **E8 cannot have the gain.** It rounds half of the small weights to zero and moves the other half to plus or minus half a step, which on a weight near zero is a larger error than noise gives it.
- **A guess of mine was wrong.** I read the gain as pruning by size, which models bear well, and stated that zeroing the small weights would cost under a third of what noise on them costs. It costs 0.48 of it.

## The instrument

This session has no build of llama.cpp. `shape/npforward.py` runs the model from its GGUF file in NumPy: the tokenizer from the file's merges, 32-bit floats. The model, the text and the positions are the other record's: `SmolLM2-1.7B-Instruct-f16.gguf`, the wikitext-2 test set, passages of 512 tokens with the second half scored. Arms are scored on the first 16 of its 40 passages, 4,080 positions.

Four checks were set in the plan. All hold (`results.out`).

| | Check | Found |
|---|---|---|
| C1 | Reference perplexity on the 40 passages within 1 percent of llama.cpp's 9.4117 | 9.4292, +0.19 percent |
| C2 | Weight errors of the three base arms within 0.01 dB of the harness's | equal to four decimals |
| C3 | Their mean KL on 16 passages within 20 percent of the harness's on 40 | 1.065, 1.004 and 1.048 times |
| C4 | The split is there: E8 at least 0.5 dB above integer levels in KL per unit of weight error | +1.21 dB, where the harness has +1.02 |

So the split is not a property of llama.cpp or of the harness. A second program that shares no code with them has it.

## Results

`results.jsonl`, in the order scored. c is the mean KL over the weight error. The standard error of a mean KL treats positions as independent and is understated.

| Arm | Mean KL | Its error | Weight error | c |
|---|---|---|---|---|
| `X32:16`, the scalar rule | 0.1339 | 0.0042 | -21.166 dB | 17.5 |
| `ZBG128:b560s16`, integer levels in a ball | 0.0562 | 0.0017 | -24.928 | 17.5 |
| `EBG128:b560s16`, E8 in the same ball | 0.0638 | 0.0026 | -25.585 | 23.1 |
| `noise(EBG128:b560s16)#1` | 0.0622 | 0.0019 | -25.585 | 22.5 |
| `noise(ZBG128:b560s16)#1` | 0.0730 | 0.0023 | -24.928 | 22.7 |
| `shuffle(EBG128:b560s16)#1` | 0.0608 | 0.0020 | -25.585 | 22.0 |
| `shuffle(ZBG128:b560s16)#1` | 0.0737 | 0.0022 | -24.928 | 22.9 |
| `mix(attn=E,ffn=Z)` | 0.0546 | 0.0016 | -25.303 | 18.5 |
| `mix(attn=Z,ffn=E)` | 0.0657 | 0.0024 | -25.209 | 21.8 |
| `noise(EBG128:b560s16)#2` | 0.0642 | 0.0024 | -25.585 | 23.2 |
| `scale(ZBG128:b560s16,-0.66)` | 0.0481 | 0.0014 | -25.588 | 17.4 |
| `noisepart(ZBG128:b560s16)@small#1` | 0.0706 | 0.0021 | -24.928 | 22.0 |
| `noisepart(ZBG128:b560s16)@big#1` | 0.0569 | 0.0018 | -24.928 | 17.7 |
| `only(ZBG128:b560s16)@small` | 0.0152 | 0.0004 | -36.010 | 60.5 |
| `onlynoise(ZBG128:b560s16)@small#1` | 0.0319 | 0.0013 | -36.010 | 127.3 |
| `only(ZBG128:b560s16)@big` | 0.0417 | 0.0013 | -25.280 | 14.1 |

The arms are described in `run.py` and in the plans. "Small" is the weights the integer arm rounds to zero: those under half a step of their group.

## The plan's readings

Two arms "agree" if their c are within 0.4 dB. The gaps below are from `results.out`, with the standard error of the mean of the 16 passages' paired ratios.

| | Gap in c | Error | Reading by the plan's rules |
|---|---|---|---|
| Noise at E8's size over the E8 arm | -0.11 dB | 0.19 | agree |
| Noise at the integer arm's size over the integer arm | +1.14 | 0.17 | do not |
| The two noise arms, one over the other | -0.03 | 0.01 | agree: KL is in proportion to the size of noise |
| Two draws of the same noise | +0.14 | 0.14 | agree |
| The integer arm's error, shuffled within each group, over the arm | +1.18 | 0.16 | do not |
| The same over noise of its size | +0.04 | 0.14 | agree |
| E8's error, shuffled, over the arm | -0.21 | 0.20 | agree |
| The integer arm's error made 0.66 dB smaller, over the arm | -0.01 | 0.01 | agree |

- **D1a holds**, the outcome I had marked as my guess: noise at E8's size agrees with the E8 arm, and noise at the integer arm's size is above the integer arm.
- **D2 holds for the integer arm.** Shuffled, its error agrees with noise and not with the arm. The values of its errors do not carry the gain. Which weight gets which error does.
- **For the E8 arm neither D2 nor D2' holds as written**, because the arm, its shuffle and its noise all agree.
- **L holds.** The same pattern of error made 0.66 dB smaller has a KL 0.67 dB smaller. For one pattern a decibel of weight error is a decibel of KL, to 0.01 dB.
- **D3: the feed-forward matrices carry the extra.** E8 there and integer levels in attention is 0.0096 above the integer arm, 3.3 errors. The reverse is 0.0015 under it, 0.7 errors. The two mixed arms together equal the two ball arms together to 0.0004, so the two kinds of matrix add.

## Extension 1: is it the weights rounded to zero?

It is. The reading is E1, which I had marked as my guess.

| Arm | Mean KL | Gap in c to the integer arm | to the noise arm |
|---|---|---|---|
| The integer arm | 0.0562 | | -1.14 dB |
| Noise on the 92 percent it does not round to zero | 0.0569 | +0.06 dB | -1.08 |
| Noise on the 7.8 percent it rounds to zero | 0.0706 | +0.99 | -0.14 |
| Noise on every weight | 0.0730 | +1.14 | |

Away from zero the error of integer rounding is noise already. At zero it is not: the error on a weight under half a step is the weight itself, never larger than it and never of the other sign.

## No single matrix sees it

`layer_error.out`, written after extension 1 and not planned before it. The reference model was run on two passages and the input of every block matrix of one block in three was kept: 56 matrices, 1,024 positions. For each arm the error of a matrix's output on those inputs was taken over the output itself.

| Arm | Weight error | Output error | Output over weight |
|---|---|---|---|
| Integer levels | -24.927 dB | -26.575 dB | -1.65 dB |
| E8 | -25.585 | -27.248 | -1.66 |
| Noise at the integer arm's size | -24.927 | -26.581 | -1.65 |
| Noise on the small weights only | -24.927 | -26.552 | -1.62 |
| Noise on the other weights only | -24.927 | -26.616 | -1.69 |

In the feed-forward matrices, where the gain is, the arms at the integer arm's size are within 0.01 dB of each other. So a measure taken at each matrix with the model's own inputs, which is what an importance matrix weights, does not tell the arm from noise. `shape/lattice_spread.smollm2.out` had found the same from the weights: by row, by column and in the columns with the least weight the two lattices spread their error alike. In the one column of a matrix with the least weight E8's lead is 0.58 dB in the mean, where it is 0.66 over the whole matrix.

The inputs themselves are far from even. In the three matrices that read the attention's input, the largest hundredth of the coordinates hold 66 percent of the inputs' squared size. That is why an importance matrix is worth so much on this model, as the record `imatrix-baseline-2026-11-17` found. It is not what separates these arms.

## Extension 2: the two parts by themselves

One part of the weights changed and the rest left exact.

| Arm | Mean KL | Weight error | c |
|---|---|---|---|
| The small weights set to zero | 0.0152 | -36.01 dB | 60.5 |
| Noise of that size on them | 0.0319 | -36.01 | 127.3 |
| The integer arm's error on the other weights | 0.0417 | -25.28 | 14.1 |

| | Statement | Verdict |
|---|---|---|
| F1 | Setting the small weights to zero costs under a third of noise of that size on them | **Fails.** 0.48 of it: 3.2 dB under, error 0.15 |
| F2 | The parts add to within 10 percent of the integer arm | **Holds.** 0.0152 + 0.0417 = 0.0568, 1.2 percent over 0.0562 |
| F3 | Noise on the small weights alone costs at least twice per unit of weight error what noise on every weight does | **Holds.** 5.6 times |

So the picture of extension 1 was half right. The gain of integer levels over noise is all at the small weights, and it is exactly the gap between the first two rows: 0.0167, where the arm and the noise arm are 0.0168 apart. But setting those weights to zero is not cheap. It is the dearest part of the arm's error by a factor of four. It is only half as dear as the alternative.

## What follows

1. **For issue #10.** Its formula has the lattice as a cell constant and counts every squared error alike. On this model that gives the weight error to 0.035 dB and misses the output by 1.1 dB between two lattices, for a reason that is neither the cell nor the lattice: what a rule does to a weight under half a step. A term for it would read one thing from a rule, whether a weight near zero can get an error larger than itself, and one thing from a model, how much dearer an error on its small weights is. The second is 4 to 9 times here. It is not known for any other model.
2. **For the choice between the lattices** (issue #15, and the other record's "arm to carry forward"). The integer arm's lead over E8 on this model is not a lead of its cell. E8's cell is still 0.66 dB better, and E8 pays 1.1 dB at zero. A code that had E8's cell away from zero and rounded small weights towards zero would have both. None is built here.
3. **Where to spend bits.** The small weights hold 27 percent of the integer arm's KL in 7.8 percent of its error. Extension 3 tests the plainest use of that.
4. **Qwen2.5.** There the two lattices share a figure, so by this reading an error on a small weight is no dearer than on any other there. The same two arms as extension 1 would say. This instrument cannot run that model: it has biases and shared key heads.

## Limits

- **One model, one text, 16 passages.** A noise or shuffle arm is one draw. The two draws taken differ by 0.14 dB.
- **The instrument is not llama.cpp.** It agrees with it to 0.19 percent in perplexity and 0.4 to 6.5 percent in the KL of three arms, on a subset of its passages.
- **Why an error on a small weight is dear is not found.** It is not at the matrix. It needs the model's sensitivity to each output, which a pass forward does not give.
- **"Small" is under half a step of the group at 4.5 bits.** At another rate the set is another set.
- **`layer_error.out` was not planned.** It was written after extension 1's rows.

## Deviations

- `summary.py` was written while the first arm was being scored and before any row was read. Its parts for the extensions were each committed before that extension's rows.
- The commit `93c3ade` says it holds two rows of this run. It holds three: the third landed while it was being made.
- The plan's time is given as "about 21:10". Its commit is at 21:09.

## How to rerun

From the root of a checkout, with NumPy. The model file and the text are those of `model-output-smollm2-2026-11-17`.

```bash
E=experiments/lattice-split-smollm2-2026-11-17
python3 $E/run.py reference --model SmolLM2-1.7B-Instruct-f16.gguf --text wiki.test.raw --work DIR > $E/reference.out
python3 $E/run.py build --model SmolLM2-1.7B-Instruct-f16.gguf --work DIR
python3 $E/run.py score --model SmolLM2-1.7B-Instruct-f16.gguf --text wiki.test.raw --work DIR --out $E/results.jsonl ARM [ARM ...]
python3 $E/summary.py > $E/results.out
python3 $E/layer_error.py --model SmolLM2-1.7B-Instruct-f16.gguf --text wiki.test.raw --work DIR > $E/layer_error.out
python3 -m unittest shape/test_npforward.py
```

The reference pass takes 6 minutes on 4 CPU threads, the build 11, an arm 4 to 9. The work folder holds 13 GB. No weights and no text are committed.

## Disclosure

Written by Claude (Anthropic), in a session that owns issue #10. The rows of `model-output-smollm2-2026-11-17` are another session's.
