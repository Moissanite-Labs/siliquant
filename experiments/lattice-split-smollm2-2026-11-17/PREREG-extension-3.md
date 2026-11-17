# Extension 3 of the plan: finer levels at zero, against the same bits spent everywhere

Written 2026-11-17, about 23:22 UTC, after the sixteen rows so far were read and before either arm below is built or scored. The program is `run.py` at the commit that adds this file.

## What is known

Extension 2: on this model the parts of the integer arm's error add, and they are not equally dear. Per unit of weight error, its error on the 7.8 percent of the weights it rounds to zero costs 60 of KL, noise on those weights costs 127, and its error on the other 92 percent costs 14. The small weights hold 7.8 percent of the arm's error and 27 percent of its KL: 0.0152 of 0.0562.

So a rule that spends bits where the error is dear should beat one that spends them evenly. The formula of issue #10 counts every squared error alike and cannot say this.

## Two arms

| Arm | What it is | Bits a weight |
|---|---|---|
| `ZBG128:b576s16` | the integer arm with 16 more bits for the point of each group: every step 8.3 percent shorter | 4.625 |
| `fine0(ZBG128:b560s16)` | the integer arm with three levels where it has one at zero: a weight under half a step goes to the nearest of minus a third, zero and a third of a step. Every other weight as in the integer arm | 4.5 and the share of such weights times log2 of 3: about 4.625 |

**The second count is a bound and not a code.** It is what the choice among three would cost if coded at its entropy bound with the three equally likely. The build step prints it from the share of weights rounded to zero. No index for such a codebook is written here.

## Stated before the rows

By the closed form `ZBG128:b576s16` has 0.75 dB less weight error than the integer arm. If its KL per unit of weight error stays 17.5, its KL is 0.0472.

For `fine0`: the error on the small weights falls to a ninth. If the parts add, and what is left of that error costs between 60 and 127 per unit, its KL is 0.0417 for the other weights and 0.0017 to 0.0035 for the small ones: 0.043 to 0.045.

| | Statement | It fails if |
|---|---|---|
| G1 | `fine0(ZBG128:b560s16)` has a lower mean KL than `ZBG128:b576s16`, by at least 2 errors of the paired difference over the 16 passages | it does not |
| G2 | Its mean KL is between 0.043 and 0.045 | it is outside |
| G3 | `ZBG128:b576s16` is within 0.2 dB of 0.0472 | it is not |

If G1 holds, the bits a weight are better spent near zero than everywhere on this model, by an amount the weight error does not show: `fine0` has the larger weight error of the two. If it fails, the cost of the small weights is not collected this simply.

## Limits

One model, 16 passages, an idealised bit count for one of the two arms. Nothing here says the same holds on Qwen2.5, where the two lattices share one figure and this instrument cannot run.
