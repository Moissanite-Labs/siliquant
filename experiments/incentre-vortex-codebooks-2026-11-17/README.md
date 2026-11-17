# The codebooks proposed in PR #6 against scalar rounding and E8 at equal bits per weight

## Question

PR #6 holds a research report that proposes five experiments. Three of them name codebooks that can be tested on weight error with no model:

- **F_n and CaB_n.** Round blocks of n weights to the point set F_n of Foundation-0/incentre-lattices, or to F_n with two more points per cell. The report's target is to "surpass E8 and scalar" rounding on heavy-tailed data at 1.5 to 2.5 bits per weight.
- **Vortex codebooks.** Round pairs of weights to 16 or 64 points taken from a relative equilibrium of point vortices.
- **The 880 realisations.** Round blocks of 14 weights to the 880 realisations of the graph L880 from Foundation-0/rigid-realizations.

At equal bits per weight, does any of them give a lower weight error than scalar rounding, and than the E8 code of the first record? Issue #4 asks the same of the first.

## Status

`complete`, with the plan amended once before any result and one sweep added after the results were read. It measures weight error only, with one scale per matrix.

## Preregistration

[PREREG.md](PREREG.md), committed and pushed at `61c552b` before the run.

- **Extension 1**, the 880 realisations, was committed at `fb54260` before it was scored.
- **Amendment 1** made the run smaller. It was committed at `b56c08e` after a first attempt was stopped, before any result existed. See Deviations.
- **`larger_t.py` was not planned.** It was written after the main run was read, and it changes what the record says about the arms with t free.

6 of the 8 predictions of the plan hold, and all 3 of the extension.

## Result in short

1. **As proposed, F_n and CaB_n lose to scalar rounding and to E8 at every rate tried.** On Gaussian data they are behind a scalar mix at equal bits by 0.4 to 1.3 dB and behind an E8 mix by 0.7 to 3.0 dB, on every matrix. On Student t data, the heavy-tailed case the report names, they are behind the scalar mix by 0.0 to 0.8 dB and behind E8 by 1.2 to 3.5 dB. The report's target is met on no family.
2. **The reason is the cell.** The cell of a point of F_n is a pyramid, and with fine cells it is worth 1.2 to 2.0 dB less than a cube. This is derived, and measured to four digits. CaB_7 and CaB_15 are worth 1.04 and 0.68 dB less than a cube.
3. **The report's decoding claim is true and says less than it seems to.** The nearest point of F_n is found by rounding each weight to a grid and then one search for the largest remainder. That holds for any position of the points on the axes. The incentre plays no part in it.
4. **With the position t free, the best codes of the family are not on the incentre lattice at all.** The best t found is 3.2 to 4.0, five to seven times the incentre's 0.586. There the code is K levels per weight with one weight of each block allowed to lie far out: an escape for the block's largest weight. That code beats the scalar mix in 38 of the 40 synthetic cells, by up to 0.7 dB on Gaussian data and 1.7 dB on Student t data. It beats the E8 mix in 8 of the 40, all at 1.3 to 1.9 bits, six of them on Student t data. This was found by a sweep that the plan did not have.
5. **A vortex codebook is about as good as the best scalar quantiser.** 16 points give -9.34 dB on Gaussian data and 64 give -14.56. The best scalar quantisers with 4 and 8 levels give -9.30 and -14.62. E8 gives -10.12 and -15.55.
6. **The 880 realisations make a worse codebook than 880 random points**, by 0.8 dB as they come and by 0.4 dB after the best linear map. At their 0.70 bits per weight the floor for any code on Gaussian weights is -4.21 dB, so the report's milestone of matching 2-bit methods is closed for weight error whatever the codebook.

## How it was measured

| | |
|---|---|
| F8, F16 | F_n with K cells per axis and the points at t = 2 - sqrt 2 = 0.586 on the axes. log2(K) + 4/8 and log2(K) + 5/16 bits per weight |
| CaB7, CaB15 | the same with each cell's centre and one corner added. log2(K) + 4/7 and log2(K) + 5/15 bits |
| F8t and so on | the same with t searched on each matrix |
| V16, V64 | the stable vortex equilibrium of lowest energy found from 200 and 100 random starts. 2 and 3 bits |
| LBG16, LBG64 | points in the plane fitted by Lloyd's algorithm to a separate sample of the family |
| L880 | the 880 realisations as a codebook for blocks of 14, moved to mean zero. L880w: mapped linearly to unit covariance. 0.6987 bits |
| S, E8 | arms S and E8 of the first record, imported from its program |
| Scale | one per matrix, found by the search of the first record |
| Equal bits | a reference at a candidate's bits is a mix of two neighbouring level counts, with error the weighted mean of the two measured errors |
| Data | G: Gaussian, 128 by 1680, three seeds. T: Student t with 4 degrees of freedom, the same. Real: the first 131,040 weights of each of the 32 matrices of the first record |
| Measure | relative squared error in dB. A gap is candidate minus reference: positive means the candidate is worse |

Every arm finds the nearest codeword of its codebook exactly. For F_n and CaB_n the program checks its rule against a search over the whole codebook, on 40 small codebooks in the main run and on 35 more at t from 1.3 to 4.0 in the sweep. The largest difference found is zero.

All numbers below are in [results.out](results.out), [l880.out](l880.out), [larger_t.out](larger_t.out) or `bounds/bounds.out`.

## Results

### The decoding claim, and its proof

The report says F_n has an exact decoder with no search. It does. Let x be in the cube [-1, 1]^n with x_1 >= |x_j| for all j, and let p = 2m + s t e_k be any point of F_n, with 0 < t < 1. For l other than k, (x_l - 2m_l)^2 >= x_l^2, because |x_l| <= 1. On coordinate k the points are 2m + t and 2m - t, and the nearest to x_k is sign(x_k) t, because the midpoints between neighbours are 0 and +-1. So

    |x - p|^2 >= |x|^2 - 2t|x_k| + t^2 >= |x|^2 - 2t x_1 + t^2 = |x - t e_1|^2.

So the cell of the point t e_1 is the pyramid over the facet x_1 = 1. On 200,000 points of the cube for each of n = 2, 3, 4 and 5, the point this rule names was never farther than the nearest point found by a search.

### What the cells are worth

With fine cells and nothing clipped, error is G times the volume per point to the power 2/n. The cube has G = 1/12 = 0.083333 and E8 has 0.071682.

| Point set | G, closed form | G, measured | Against the cube |
|---|---|---|---|
| F_8, t at the incentre | 0.123026 | 0.122996 +- 0.000024 | -1.69 dB |
| F_8, t at the centroid | 0.117284 | 0.117256 +- 0.000023 | -1.48 dB |
| F_16, t at the incentre | 0.110216 | 0.110192 +- 0.000014 | -1.21 dB |
| F_16, t at the centroid | 0.107172 | 0.107149 +- 0.000014 | -1.09 dB |
| CaB_7 | none | 0.105819 +- 0.000017 | -1.04 dB |
| CaB_15 | none | 0.097529 +- 0.000009 | -0.68 dB |

`results.out` has n = 2, 3, 4, 7 and 15 as well. All 14 measured values are within 1.7 standard errors of the closed form. A check after the run with 40 million points gave 0.1 and 0.9 standard errors for n = 8 and 16. The closed form is now section 7 of `bounds/bounds.out`.

The report quotes packing fractions for these point sets: 0.2233, and 0.746865 for the CaB6 packing. A packing fraction says how well balls fit, which matters for a code that has to resist noise. For rounding the figure is G.

### As proposed, at equal bits

Gap to the scalar mix and to the E8 mix in dB, mean over the matrices. Positive means the candidate has the larger error.

| Data | K | F8: bits, scalar, E8 | F16: bits, scalar, E8 | CaB7: bits, scalar, E8 | CaB15: bits, scalar, E8 |
|---|---|---|---|---|---|
| G | 2 | 1.50, +0.91, +1.39 | 1.31, +0.46, +0.76 | 1.57, +0.88, +1.46 | 1.33, +0.42, +0.73 |
| G | 3 | 2.08, +1.03, +1.94 | 1.90, +0.60, +1.38 | 2.16, +0.69, +1.64 | 1.92, +0.44, +1.25 |
| G | 4 | 2.50, +1.08, +2.19 | 2.31, +0.71, +1.74 | 2.57, +0.70, +1.84 | 2.33, +0.46, +1.49 |
| G | 8 | 3.50, +1.21, +2.66 | 3.31, +0.79, +2.20 | 3.57, +0.67, +2.15 | 3.33, +0.39, +1.80 |
| G | 16 | 4.50, +1.30, +3.02 | 4.31, +0.88, +2.55 | 4.57, +0.74, +2.47 | 4.33, +0.42, +2.10 |
| T | 2 | 1.50, +0.59, +1.79 | 1.31, +0.24, +1.27 | 1.57, +0.41, +1.70 | 1.33, +0.12, +1.16 |
| T | 3 | 2.08, +0.47, +2.19 | 1.90, +0.21, +1.77 | 2.16, +0.16, +1.91 | 1.92, +0.01, +1.59 |
| T | 4 | 2.50, +0.62, +2.57 | 2.31, +0.36, +2.19 | 2.57, +0.25, +2.24 | 2.33, +0.07, +1.92 |
| T | 8 | 3.50, +0.71, +3.09 | 3.31, +0.43, +2.76 | 3.57, +0.32, +2.74 | 3.33, +0.12, +2.45 |
| T | 16 | 4.50, +0.77, +3.54 | 4.31, +0.51, +3.21 | 4.57, +0.41, +3.20 | 4.33, +0.21, +2.92 |
| Real | 4 | 2.50, +1.00, +2.24 | 2.31, +0.86, +1.85 | 2.57, +0.32, +1.68 | 2.33, +0.36, +1.36 |
| Real | 16 | 4.50, +1.14, +3.14 | 4.31, +0.71, +2.72 | 4.57, +0.45, +2.56 | 4.33, +0.17, +2.19 |

On G and T every candidate is worse than both mixes on all three matrices, with one exception: CaB15 at K = 3 on T is worse than the scalar mix on two of three, and level with it in the mean. On the real matrices each candidate is worse than the E8 mix on 31 or 32 of 32, and worse than the scalar mix on 28 to 32.

### With t free

**In the main run** t was searched over 0.4 to 1.0. The search ended at 1.0 in most cases: 344 of the 376 rows have t at an end of the grid. So those arms were not at their best, and the amendment had cut the grid at 1.0 where the first plan went to 1.2.

**The sweep** (`larger_t.py`, not planned) searched 1.1 to 4.0. The best t is 3.2 to 4.0 everywhere. The table gives the better of the two searches.

| Data | K | F8t: scalar, E8 | F16t: scalar, E8 | CaB7t: scalar, E8 | CaB15t: scalar, E8 |
|---|---|---|---|---|---|
| G | 2 | -0.18, +0.30 | -0.38, -0.08 | -0.14, +0.44 | -0.40, -0.09 |
| G | 3 | -0.35, +0.57 | -0.41, +0.38 | -0.49, +0.45 | -0.47, +0.34 |
| G | 4 | -0.39, +0.72 | -0.40, +0.63 | -0.55, +0.59 | -0.51, +0.52 |
| G | 8 | -0.25, +1.21 | -0.43, +0.98 | -0.59, +0.89 | -0.66, +0.76 |
| G | 16 | +0.21, +1.93 | -0.09, +1.58 | -0.23, +1.50 | -0.44, +1.23 |
| T | 2 | -1.46, -0.26 | -1.61, -0.58 | -1.66, -0.38 | -1.68, -0.64 |
| T | 3 | -1.56, +0.16 | -1.63, -0.07 | -1.70, +0.05 | -1.72, -0.13 |
| T | 4 | -1.29, +0.65 | -1.41, +0.42 | -1.49, +0.50 | -1.56, +0.28 |
| T | 8 | -0.56, +1.83 | -0.78, +1.54 | -0.83, +1.58 | -1.01, +1.33 |
| T | 16 | +0.02, +2.79 | -0.19, +2.51 | -0.27, +2.52 | -0.44, +2.26 |
| Real | 4 | -0.61, +0.63 | -0.41, +0.58 | -1.11, +0.25 | -0.77, +0.22 |
| Real | 16 | +0.29, +2.28 | -0.05, +1.96 | -0.33, +1.78 | -0.52, +1.50 |

Gaps in dB, mean over the matrices. Negative means the candidate is better.

- Against the scalar mix the candidate is better in 38 of the 40 synthetic cells and in 7 of the 8 real ones.
- Against the E8 mix it is better in 8 of the 40 synthetic cells: F16t and CaB15t at K = 2 on G, all four at K = 2 on T, and F16t and CaB15t at K = 3 on T. On the real matrices it is worse in the mean in all 8, and worse on 27 to 32 of the 32.
- In 30 of the 376 searches the best t was 4.0, the top of this range too: 21 on T at K of 2 to 4 and 9 on the real matrices. So on heavy tails at the lowest bits the gaps above understate the candidate.

**What these codes are.** At t = 3.4 a codeword is K evenly spaced levels on every weight, with one weight of the block moved 1.7 cells further. For the unbounded point set, t and 2 - t give the same points, so nothing new appears inside the grid. What a large t adds is at the edge: one weight per block can be represented beyond the range of the levels. It is an escape for an outlier, paid for with log2(2n)/n bits per weight. Nothing of the incentre lattice is left in it: not the incentre, not the pyramid cells, which exist only for t under 1, and not the packing or rigidity results.

### Vortex codebooks

| Data | V16 | Fitted, 16 | S, q = 4 | E8, q = 4 | V64 | Fitted, 64 | S, q = 8 | E8, q = 8 |
|---|---|---|---|---|---|---|---|---|
| G | -9.34 | -9.67 | -9.23 | -10.12 | -14.56 | -15.22 | -14.25 | -15.55 |
| T | -6.64 | -8.14 | -6.38 | -8.06 | -10.16 | -13.59 | -9.66 | -11.86 |
| Real | -7.38 | | -7.28 | -8.42 | -11.85 | | -11.31 | -12.85 |

Relative squared error in dB, at 2 and 3 bits per weight.

- **16 vortices.** 200 starts gave 3 distinct stable equilibria, and 4 starts did not converge. The lowest energy, -87.325902, was reached from 150 starts: five points on an inner ring and eleven on an outer. Every equilibrium found meets the report's balance equation to 5.0e-12 and its identity for the sum of squared sizes to 3.8e-11.
- **64 vortices.** 100 starts gave 7 distinct stable equilibria, and 26 did not converge.
- V16 is behind E8 by 0.78 dB on G, 1.42 on T and 1.05 on the real matrices, and behind the fitted codebook by 0.33 dB on G and 1.50 on T. V64 is behind E8 by 0.99, 1.70 and 1.00 dB, and behind the fitted codebook by 0.66 and 3.43.
- Against evenly spaced scalar levels V16 is ahead by 0.10, 0.26 and 0.09 dB, and V64 by 0.31, 0.50 and 0.53.
- The equilibrium of lowest energy is not the best codebook among those found. On `G/0` the three run from -9.48 to -9.27 dB.

At high rates no cells in the plane beat hexagons, which are worth 0.17 dB over the cube. E8 is worth 0.65.

### The 880 realisations

The certificate file holds 440 points. Each meets the 17 bar lengths to 4.4e-15. The other 440 are mirror images. Whether these are all the realisations, which is the claim of that repository, is not checked here.

| Data | L880 | L880w | 880 random points | 880 fitted points | Scalar mix | Floor for any code |
|---|---|---|---|---|---|---|
| G | -2.42 | -2.79 | -3.24 | -3.41 | -2.55 | -4.21 |
| T | -2.13 | -2.38 | -2.98 | -3.89 | -1.87 | |
| Real | -2.13 | -2.36 | -2.93 | | -2.00 | |

Relative squared error in dB at 0.6987 bits per weight.

- The realisations are behind 880 random Gaussian points by 0.82, 0.86 and 0.79 dB as they come, and by 0.44, 0.61 and 0.57 dB after the linear map.
- The codebook is far from round: its variance along its widest axis is 19.8 times that along its narrowest. All 17 bars have a length between 1.0003 and 1.0152, so every vertex sits about one unit from its neighbours.
- After the linear map it is ahead of the scalar mix, by 0.24, 0.51 and 0.36 dB. That mix is weak: it sets 30 percent of the weights to zero.
- The best scalar quantiser at 2 bits is at -9.30 dB on Gaussian weights. No code at 0.6987 bits can be under -4.21 dB.

The report also proposes to chain three such blocks into 880^3 codewords for 42 weights. As it describes this, it is a product of three copies of one codebook, which has the error per weight of its factor.

### The predictions

| # | Prediction | Outcome |
|---|---|---|
| 1 | The measured G of F_n is within 4 standard errors of the closed form for seven values of n at two values of t. The G of CaB_7 and CaB_15 is above 1/12 | Holds |
| 2 | The pyramid rule never names a point farther than the nearest found by search | Holds |
| 3 | On G at K of 3 and more, every candidate, as proposed and with t searched, is worse than both mixes on every matrix | Holds for the arms of the plan. The sweep, which was not planned, finds t above 3 at which the candidate beats the scalar mix in 15 of the 16 cells. Against E8 it still loses in all 16 |
| 4 | On G at K = 2 every candidate is worse than the E8 mix on every matrix | Holds for the arms of the plan. In the sweep F16t and CaB15t beat the E8 mix on all three matrices, by 0.08 and 0.09 dB |
| 5 | On T at K of 3 and more every candidate is worse than both mixes in the mean | Fails in part. As proposed all are worse than both. With t searched up to 1.0 the candidate beats the scalar mix in 8 of 16 cells, by up to 0.32 dB, and loses to E8 in all 16. In the sweep it beats the scalar mix in 15 of 16 and the E8 mix in 2 |
| 6 | On the real matrices at K = 4 and 16 every candidate is worse than each mix on at least 24 of 32 | Fails in one cell: CaB7t at K = 4 is worse than the scalar mix on 23. Against E8 it holds in all 16 cells, and in the sweep too |
| 7 | V16 and V64 are worse than E8 on every matrix of G and T, and worse than the fitted codebook in the mean over G | Holds |
| 8 | No row on G is under Shannon's floor by more than 0.05 dB | Holds |
| E1 | Every stored point meets the 17 bar lengths to 1e-6 | Holds: 4.4e-15 |
| E2 | On G, L880 and L880w are each worse than the random codebook on every matrix | Holds |
| E3 | On G no arm of the extension is under Shannon's floor | Holds |

**The hand estimates of the plan** for Gaussian data at K = 2: F_8 about -6.4 dB against -6.7 for the scalar mix, and F_16 about -5.6 against -5.7. Measured with t at the incentre: -5.75 and -5.20, against -6.66 and -5.66. The estimates were right for the scalar mix and 0.4 to 0.7 dB too kind to the candidates. They match what the run found at t = 1.0, -6.38 and -5.62.

**The report's target**, to surpass E8 and scalar rounding on heavy-tailed data, counted as met on a family if some candidate as proposed beat both mixes in the mean at some K. It is met on no family.

## Deviations from the plan

- **The first attempt at the main run was stopped.** It ran for 40 minutes on 4 cores. A timing made then put the whole run at about 19 CPU-hours, more than the session allows a background command. The program wrote its output only at the end, so nothing existed and nothing had been read. The time had been estimated wrongly before the plan was committed.
- **Amendment 1 made the run smaller**, before any result: synthetic matrices of 128 by 1680 in place of 512 by 1680, the first 131,040 weights of each real matrix, K = 4 and 16 only on the real matrices, the reference arms only at the levels the comparisons need, and t searched over 0.4 to 1.0 in place of 0.30 to 1.20. Prediction 6 was restated for the two values of K. The run then took 749 seconds.
- **Cutting the grid for t at 1.0 hid a result.** The search ended at 1.0 nearly everywhere. `larger_t.py` was written after that was seen. Its first version went to 2.0 and its search ended at 2.0 on the heavy-tailed family, so it was widened to 4.0. Only the output of the last version is kept. A session restart stopped one of its runs, and it was run again.
- **The sweep is not preregistered.** Its range was chosen twice after looking. What it found is a lead and not a tested claim.
- **Only 3 and 7 distinct equilibria were found**, where the plan speaks of the 20 lowest.
- **The plan of extension 1 is wrong on two points.** It gives the shortest bar as 1.003, and it is 1.0003. And it says the source check reported that the report's author could not open rigid-realizations. It was the report itself that said so, and the source check that found the repository open.
- **`results.start`** gives the time of the first attempt to within a few seconds. The file was rewritten when the second run started.

## Limits

- **Weight error only.** The second record found that weight error does not say what a model's output does.
- **One scale per matrix, found by least squared error.** The second record found that a model's output is far closer with one scale per group set from the largest entry. Under that rule F_8 at t of 0.586 and 0.889 loses to scalar rounding and to E8 on every matrix ([e8-gain-largest-entry-rule-2026-11-17](../e8-gain-largest-entry-rule-2026-11-17/README.md)). A large t has not been run under that rule.
- **The mixes are weak references on purpose.** A mix is two neighbouring level counts side by side. The scalar arm has evenly spaced levels. The E8 arm uses the simplest rule for blocks out of range, which the first record found to cost 0.19 to 0.26 dB at 2 bits. Beating a mix by a few tenths of a dB at 1.3 to 2 bits is not beating the best scalar or E8 code at those bits.
- **The real matrices are the first 131,040 weights of each**, in rows of 1680 that do not follow the rows of the matrix. One small speech model.
- **Three seeds** per synthetic family, with matrices of 215,040 weights.
- **Vortex equilibria:** only stable ones, which a downhill search reaches. Nothing shows that the lowest energy found is the lowest there is.
- **The 880 realisations** were run with the bar lengths of the certificate. The report also proposes to train the 17 lengths. That was not tried.
- **Nothing is timed.**

## Read against the closure rule

One thing here is proved: the cell of F_n is a pyramid for 0 < t < 1, and its G follows by integration. So F_n as proposed is behind the cube by 1.2 to 2.0 dB with fine cells, in the dimensions listed in `bounds/bounds.out`, and the proposal to beat E8 with it at high rates is closed. At low rates nothing is proved, and the run is a measurement.

For Gaussian weights, Shannon's floor closes the milestone of the 880 realisations for weight error: -4.21 dB at their bits against -9.30 dB for 2-bit scalar rounding.

For issue #4: no Foundation-0 point set tried here beats E8 at 2 bits or more. The family with t free beats the E8 mix only under 2 bits, mostly on heavy-tailed data, with one scale per matrix, and that code is not a Foundation-0 lattice.

## How to rerun

Needs NumPy, the weights file of `mlx-community/whisper-base.en-mlx`, and for the extension `l880_certificate.json` of Foundation-0/rigid-realizations at commit `298926c`. Both programs refuse a file with another SHA-256.

```bash
python3 experiments/incentre-vortex-codebooks-2026-11-17/run.py --weights weights.npz --workers 4 > results.out
```

```bash
python3 experiments/incentre-vortex-codebooks-2026-11-17/l880.py --certificate l880_certificate.json --weights weights.npz > l880.out
```

```bash
python3 experiments/incentre-vortex-codebooks-2026-11-17/larger_t.py --weights weights.npz > larger_t.out
```

`run.py --smoke` runs the code path on a small matrix that is not in the study. The main run took 12 minutes on 4 CPU cores with NumPy 2.5.3 and Python 3.13, the extension 6 minutes, and the sweep about 6.

## Disclosure

The plan, the programs and this record were written by an AI agent session under the owner's direction.
