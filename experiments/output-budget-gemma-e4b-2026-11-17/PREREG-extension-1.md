# Extension 1: where the damage of the clip lives, and a closure in the budget

Written on 2026-11-17 after every part of [PREREG.md](PREREG.md) was scored, and committed before any variant below was built or scored. The results it was written with are in `results.out` at commit `5865ea5`.

## What was seen

Part E clipped the entries of `ffn_down` beyond 5 standard deviations of their row, in six layers, one at a time and all six at once.

| | KL |
|---|---|
| The six layers alone, added up | 0.001673 |
| All six at once | 0.001637 |
| All 42 layers at once, from the first record | 0.163763 |

On the six, the damage adds: the ratio is 0.978. The plan's rule leaves prediction E1 undecided, because one of the six has a KL under five times its standard error. The ratio is far from the 1.5 that E1 predicted.

But the six are a seventh of the layers and carry a hundredth of the damage of all 42. Two readings fit that. A few layers not among the six carry most of the damage. Or the damage grows faster than a sum when many layers are clipped, though it did not with six. This extension tells them apart.

For rounding the same figures are 0.005700, 0.005033 and 0.023730. Seven times the layers give 4.2 times the six. Nothing there needs explaining, and rounding is not run again.

## Part H: the clip on every layer alone

`clip:5@ffn_down` on each of the 36 layers not yet scored alone, one at a time. With the six of Part E that gives all 42.

| | Prediction | Fails if |
|---|---|---|
| H1 | The damage adds over all 42 layers: the sum of the 42 single-layer KLs is 0.75 to 1.25 times the KL of the clip on all layers at once (0.1638). | the ratio is outside that range |
| H2 | A few layers carry it: the four largest single-layer KLs make more than half of the sum. | they make half or less |
| H3 | Those are the layers the clip changes most: over the 42 layers, the rank correlation between a layer's KL and the weight error of its clip is at least 0.5. | it is under 0.5 |

H1 is the first reading above. If it fails with a ratio under 0.75, the second reading stands, and the record says so.

A single layer whose KL is under five times its standard error is listed. H1 and H2 are scored as they are if those layers hold under a tenth of the sum, and are marked not decided otherwise.

## Part K: a closure in the budget

The first record found that the last 18 layers never read their own key and value matrices: 55,050,240 weights that llama.cpp's forward pass does not touch. Zeroing them scored a KL of 0.000000.

Part A and the allocation of Part B counted those weights with the rest. The joined arm gives them 64 levels. A weight that no path reads needs no bits.

One variant: the joined arm of the plan, with the unread key and value matrices zeroed and counted at 0 bits.

| | Prediction | Fails if |
|---|---|---|
| K1 | Its KL equals that of the joined arm (0.062882) to four decimals, and its bits fall from 4.484 to under 4.41. | the KL differs in the fourth decimal, or the bits are 4.41 or more |

The closure holds for llama.cpp's forward pass at the commit the first record names. It says nothing about another runtime.

## Order of work

1. This file, committed and pushed. The program refuses to score a variant of Part H or Part K unless it is.
2. The 36 variants of Part H, then the one of Part K. Results committed and pushed.
3. `summary.py` scores H1 to H3 and K1 with the thresholds above.

## Limits known before the run

- H1 to H3 are about one kind of matrix, one limit and one model.
- The standard errors treat positions as independent, which understates them.
- H3 was written after seeing the six layers of Part E, where the two orders already agree. Only the 36 new layers test it.
