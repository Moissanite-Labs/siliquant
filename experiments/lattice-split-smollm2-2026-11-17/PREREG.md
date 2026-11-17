# Plan: what in E8's error costs SmolLM2 more than its size

Written 2026-11-17, about 21:10 UTC, before any arm is scored by the program of this folder. The run is `run.py` here at the commit that adds this file.

## Question

The run `model-output-smollm2-2026-11-17` scored integer levels in one ball per group of 128 weights under E8 in the same ball: a mean KL of 0.0560 against 0.0609, though E8 has 0.66 dB less weight error. On that model the arms with one scale per group have 16.4 to 17.4 of KL per unit of weight error on integer levels and 21.4 to 23.0 on E8. On Qwen2.5 the same arms share one figure.

`shape/lattice_spread.py` found nothing in the weights that tells the two errors apart on SmolLM2: the same share by row and by column, the same share along the matrix, no mean, the same spectrum. What differs is how an error depends on its own weight and on the seven beside it.

So: is E8's error worse for this model than noise of its size, or is the error of integer levels better than noise of its size? And which matrices carry the difference?

## The instrument

This session has no build of llama.cpp. The model is run by `shape/npforward.py`, in NumPy: its own tokenizer from the file's merges, 32-bit floats. The harness of the other record is not used and none of its rows is replaced.

**Model**: `SmolLM2-1.7B-Instruct-f16.gguf`, SHA-256 `cf3f0569…0852`, the file of that record. **Text**: the wikitext-2 test set, SHA-256 `173c87a5…dd08`, the file of that record. Passages of 512 tokens, the second half of each scored, as there.

The reference pass is run on that record's 40 passages, for one check. Arms are scored on the first 16 of them, 4,080 positions, to keep an arm to about five minutes.

**Rounded values are stored as 16-bit floats.** The weight error of an arm is the mean over the 168 block matrices of its error in dB, as in that record.

## Arms, in the order they are scored

Base arms, rounded as the harness rounds them. All three have a score in the other record.

| Arm | What it is |
|---|---|
| `X32:16` | the scalar largest-entry rule, groups of 32 |
| `ZBG128:b560s16` | integer levels, one ball per group of 128 |
| `EBG128:b560s16` | E8 in the same ball |

Arms made from a base arm's error, which is the rounded matrix less the weights:

| Arm | What it is | What it keeps of the base arm's error | What it drops |
|---|---|---|---|
| `noise(EBG128:b560s16)#1` | Gaussian noise with the base arm's squared error in every group of 128 | its size, group by group | everything else |
| `noise(ZBG128:b560s16)#1` | the same for integer levels | | |
| `shuffle(EBG128:b560s16)#1` | the weights of each group trade errors at random | its size, and every value of it | which weight has which error, and which errors share a block of eight |
| `shuffle(ZBG128:b560s16)#1` | the same for integer levels | | |
| `mix(attn=E,ffn=Z)` | the four attention matrices of every block from the E8 arm, the three feed-forward ones from the integer arm | | |
| `mix(attn=Z,ffn=E)` | the reverse | | |
| `noise(EBG128:b560s16)#2` | a second draw of the first | | |
| `scale(ZBG128:b560s16,-0.66)` | the integer arm's error made 0.66 dB smaller, weight by weight: the E8 arm's size with the integer arm's pattern | | |

## Seen before this plan

- **The other record's rows**, ten of thirteen: the three base arms at 0.1257, 0.0560 and 0.0609, and its reference perplexity, 9.4117.
- **Of this instrument**: the reference perplexity on the first two passages, 9.387. One matrix built for eight arms with weight errors printed and nothing run through the model, to test the program: the noise and shuffle arms have the base arm's weight error to 0.001 dB, and the scaled arm has -25.589 dB.
- **No arm has been scored by this instrument.**

## Checks of the instrument

| | Holds if |
|---|---|
| C1 | The reference perplexity on the 40 passages is within 1 percent of the harness's 9.4117 |
| C2 | The weight errors of the three base arms are within 0.01 dB of the harness's: -21.166, -24.928 and -25.585 dB |
| C3 | Their mean KL on 16 passages is within 20 percent of the harness's on 40 |
| C4 | The split is there: the E8 arm's KL per unit of weight error is at least 0.5 dB above the integer arm's. In the other record it is 1.02 dB above |

If C1 or C2 fails, the instrument is not that model and nothing after it is read. If C4 fails on these 16 passages, the instrument cannot take the split apart, and that is the result.

## What each outcome would mean

Write c for an arm's mean KL over its weight error. Two arms "agree" if their c are within 0.4 dB.

| | Outcome | Reading |
|---|---|---|
| D1a | Noise at E8's size agrees with the E8 arm, and noise at the integer arm's size is above the integer arm | E8's error acts as noise of its size. The error of integer levels is better than noise. |
| D1b | Noise at the integer arm's size agrees with the integer arm, and noise at E8's size is below the E8 arm | Integer levels act as noise. E8's error is worse than noise of its size. |
| D1c | Neither | Both differ from noise, or the two noise arms do not agree with each other and KL is not in proportion to the size of an error here. |
| D2 | A shuffled arm agrees with the noise of its size and not with its base arm | What matters is which weight gets which error. The values alone do not carry it. |
| D2' | A shuffled arm agrees with its base arm and not with the noise | The values of the errors carry it: their spread, and not their place. |
| D3 | Against the two base arms, the two mixed arms say whether attention or feed-forward carries the extra: `mix(attn=E,ffn=Z)` less the integer arm, and `mix(attn=Z,ffn=E)` less the integer arm, in KL | The larger is where E8 costs. If neither is above zero by 2 errors, it is not in one kind. |
| L | `scale(ZBG128:b560s16,-0.66)` has a KL 0.66 dB under the integer arm's, within 0.2 dB | For one pattern of error a decibel of weight error is a decibel of KL. |

The two draws of noise give the spread from one draw to the next. If they differ by more than 0.4 dB, no reading above that rests on one draw is made.

**A guess, marked as one.** I expect D1a: the three E8 arms of the other record were cut from the lattice in two ways and share a figure, as noise would. I have no mechanism for integer levels being better than noise on this model and not on Qwen2.5.

## Errors

Positions are not independent. The standard error of a mean KL treats them as if they were and is understated. Comparisons between arms are made on the 16 passage means, as a mean of 16 paired log-ratios with its standard error.

## Limits known before the run

- One model, one text, 16 passages. A noise or shuffle arm is one draw, two for one of them.
- The instrument is not llama.cpp. C1 to C3 say how near it is.
- The mixed arms split the matrices one way of many.
- No outcome here says why on Qwen2.5 the two lattices share a figure. That needs the same arms there, and this instrument does not handle that model's biases and shared key heads.
