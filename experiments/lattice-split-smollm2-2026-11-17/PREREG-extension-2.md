# Extension 2 of the plan: what setting the small weights to zero costs by itself

Written 2026-11-17, about 22:58 UTC, after the thirteen rows so far and `layer_error.out` were read, and before any arm below is scored. The program is `run.py` at the commit that adds this file.

## What is known

Extension 1 read E1: the gain of integer levels over noise on this model is in the 7.8 percent of the weights that are rounded to zero. `layer_error.out` then showed that no single matrix sees it: fed the model's own inputs, the output of a matrix is off by the same amount whether those weights are set to zero or given noise of that size. So the difference is made further on.

Setting the smallest weights of a model to zero is pruning by size, which models are known to bear well. If that is what this is, pruning those weights alone should cost little, and noise of the same size on the same weights alone should cost several times as much.

## Three arms

In each, one part of the weights is changed and the rest are exact. "Small" is the weights `ZBG128:b560s16` rounds to zero.

| Arm | Changed | Weight error |
|---|---|---|
| `only(ZBG128:b560s16)@small` | the small weights are set to zero | about -36 dB: 7.8 percent of the integer arm's |
| `onlynoise(ZBG128:b560s16)@small#1` | the small weights get Gaussian noise of the same squared sum in every group | the same |
| `only(ZBG128:b560s16)@big` | the other weights get the integer arm's error | about -25.3 dB |

## Stated before the rows

| | Statement | It fails if |
|---|---|---|
| F1 | Setting the small weights to zero costs under a third of the KL of noise of that size on them | it costs a third or more |
| F2 | The parts add: `only@small` and `only@big` together are within 10 percent of the integer arm's 0.0562 | they are not |
| F3 | Per unit of weight error, noise on the small weights alone costs at least twice what noise on every weight does (22.7) | it costs less than twice |

F3 follows from extension 1 if the parts add: the two rows there put the cost of noise on the small weights at 0.014 to 0.016 above the cost of setting them to zero, on 7.8 percent of the error.

## Limits

One draw of the noise arm. 16 passages. The KL of the first two arms may be near the floor this instrument can see: 32-bit floats, and 16-bit storage of the weights.
