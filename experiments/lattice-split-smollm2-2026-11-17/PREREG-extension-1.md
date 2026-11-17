# Extension 1 of the plan: is it the weights rounded to zero?

Written 2026-11-17, about 22:32 UTC, after the eleven rows of the plan were read and before either arm below is scored. The program is `run.py` at the commit that adds this file.

## What the eleven rows left open

On this model the error of integer levels is 1.1 dB better than noise of its size, and it loses that when the weights of a group trade errors. So it is in which weight gets which error. An error of integer levels depends on its weight in one plain way: a weight under half a step is rounded to zero, so its error is the weight itself, never larger and never of the other sign. For the rest the error is the distance to the nearest level, which the weight's size says nothing about. `shape/lattice_spread.smollm2.out` has the first kind at 7.9 percent of the weights and 7.8 percent of the error.

## Two arms

Both are the integer arm `ZBG128:b560s16` with part of its error replaced by Gaussian noise of the same squared sum in every group of 128. They are scored after the eleven, on the same 16 passages.

| Arm | Noise on | Kept as the integer arm has it |
|---|---|---|
| `noisepart(ZBG128:b560s16)@small#1` | the weights it rounds to zero | the error of the other 92 percent |
| `noisepart(ZBG128:b560s16)@big#1` | the other 92 percent | the weights it rounds to zero stay zero |

Both have the integer arm's weight error.

## What each outcome would mean

"Agree" is within 0.4 dB in KL per unit of weight error, as in the plan. The integer arm has 17.47 and the noise at its size 22.70: 1.14 dB apart.

| | Outcome | Reading |
|---|---|---|
| E1 | `@big` agrees with the integer arm, and `@small` agrees with the noise arm | The whole gain is that small weights go to zero and nowhere else. |
| E2 | `@small` agrees with the integer arm, and `@big` agrees with the noise arm | The gain is in the larger weights, where an error's size does not depend on its weight. Then something about those errors other than their size and their place in a group matters, and this run has not found it. |
| E3 | Both lie between, more than 0.4 dB from each end | The gain is shared. |

**A guess, marked as one.** E1. Noise on a weight that is nearly zero can make it several times larger or turn its sign, and rounding cannot. The share of the error involved is 8 percent, so E1 would mean the output counts an error on a small weight far more heavily than its size.

## Seen before this extension

The eleven rows of the plan and `results.out`. One matrix built for both arms to test the program, weight error printed and nothing run through the model.

## Limits

One draw of each arm. The two draws of noise in the plan differed by 0.14 dB.
