# Plan: the codebooks proposed in PR #6 against scalar rounding and E8 at equal bits per weight

Written 2026-11-17, before the run. The run is `run.py` in this folder at the commit that adds this file.

## Question

PR #6 holds a research report that proposes five experiments. Two of them name codebooks that can be tested on weight error with no model and no GPU:

- **Its first experiment.** Round blocks of n weights to the point set F_n of Foundation-0/incentre-lattices, or to F_n with two more points per cell, which the report calls CaB-VQ. The report's target is to "surpass E8 and scalar" rounding on heavy-tailed data at 1.5 to 2.5 bits per weight.
- **The codebook half of its fourth experiment.** Round pairs of weights to 16 or 64 points taken from a relative equilibrium of point vortices.

Issue #4 asks the same of the first: does any Foundation-0 lattice family beat E8 at equal bits?

So: at equal bits per weight, does any of these codebooks give a lower weight error than scalar rounding, and than the E8 code of the first record?

## The candidates

**F_n.** Cut the cube [-1, 1]^n into the 2n pyramids over its facets and put a point on the axis of each, at distance t from the centre. Repeat over the lattice 2Z^n. The report takes t = c = 2 - sqrt 2, the incentre of the pyramid. The definition matches `results/rigidity-nd/PROOF.md` of incentre-lattices at commit `2d374ff`. That repository holds no work on rounding. The use as a codebook is the report's.

A finite codebook takes K cells per axis: 2n K^n codewords, log2(K) + log2(2n)/n bits per weight.

**CaB_n.** F_n and, for every cell, its centre and one corner: (2n + 2) K^n codewords. The report picks n = 7 and n = 15, where 2n + 2 is 16 and 32.

**V16 and V64.** N points z_a in the plane with conj(z_a) = sum over b of 1/(z_a - z_b), the report's balance equation. These are the critical points of H = sum |z_a|^2 / 2 - sum over pairs of log |z_a - z_b|. The report does not say which solution to take. `run.py` searches from 200 random starts for N = 16 and 100 for N = 64, keeps the stable ones, and uses the one of lowest energy. The 20 lowest are also scored on one matrix of each synthetic family.

## What is known before the run

**The cell of a point of F_n is its pyramid, for any 0 < t < 1.** This is the report's decoding claim, and it is true. Proof: let x be in the cube with x_1 >= |x_j| for all j, and let p = 2m + s t e_k be any point of F_n. For l other than k, (x_l - 2m_l)^2 >= x_l^2, because |x_l| <= 1. On coordinate k the points are 2m + t and 2m - t, and the nearest to x_k is sign(x_k) t, because the midpoints between neighbours are the integers 0 and +-1. So |x - p|^2 >= |x|^2 - 2t|x_k| + t^2 >= |x|^2 - 2t x_1 + t^2 = |x - t e_1|^2. The run checks this against a search.

So decoding F_n is rounding each weight to a grid and then one search for the largest remainder. The incentre is not needed for it.

**What that cell is worth.** With fine cells and nothing clipped, error is G times the volume per point to the power 2/n. For F_n the cell is a pyramid of height 1 over a cube of side 2. Along its axis the height h has density n h^(n-1), so E[h] = n/(n+1) and E[h^2] = n/(n+2). Each other coordinate is uniform on [-h, h], so its mean square is n / (3(n+2)). The squared error of a block about the point t e_i is

    n/(n+2) - 2t n/(n+1) + t^2 + n(n-1) / (3(n+2)),

and G is that over n (2^n / 2n)^(2/n). It is smallest at t = n/(n+1), the centroid.

| n | G at the incentre | against the cube | G at the centroid | against the cube |
|---|---|---|---|---|
| 2 | 0.114382 | -1.37 dB | 0.111111 | -1.25 dB |
| 3 | 0.127803 | -1.86 dB | 0.120383 | -1.60 dB |
| 4 | 0.130677 | -1.95 dB | 0.122565 | -1.67 dB |
| 7 | 0.125357 | -1.77 dB | 0.119007 | -1.55 dB |
| 8 | 0.123026 | -1.69 dB | 0.117284 | -1.48 dB |
| 15 | 0.111341 | -1.26 dB | 0.108096 | -1.13 dB |
| 16 | 0.110216 | -1.21 dB | 0.107172 | -1.09 dB |

The cube has G = 1/12 = 0.083333 and E8 has 0.071682 (`bounds/bounds.out`). So at high rates F_8 is behind scalar rounding by 1.5 to 1.7 dB and behind E8 by 2.1 to 2.3 dB. This is a derivation. It has not been measured.

**What a codebook in the plane can be worth.** At high rates no cells in the plane beat hexagons, 0.17 dB over the cube. E8 is worth 0.65 dB.

**At low rates these figures do not settle it.** With K = 2 the codebook is one sign per weight, and the moved coordinate can be spent on the weight furthest from its level. That is a correction for one outlier per block, and outliers are what the report says the codebook is for.

**Hand estimates made before any run**, for Gaussian data at K = 2: F_8 at 1.5 bits about -6.4 dB against -6.7 dB for a scalar mix, and F_16 at 1.3125 bits about -5.6 dB against -5.7 dB. Both are rough and were made without the program.

**Seen before writing this plan.** `run.py --smoke` on a 48 by 1680 matrix of uniform numbers that is not in the study, and the search for vortex equilibria on seeds 100 to 239 and 25016 to 25023, 73000 to 73007. For 16 vortices the lowest energy found was -87.325902, five points on an inner ring and eleven on an outer. No arm has been run on a matrix of the study, and no value of G has been measured.

## Arms

| Arm | Codebook | Bits per weight |
|---|---|---|
| F8, F16 | F_n with t = 2 - sqrt 2, as proposed | log2(K) + 4/8, log2(K) + 5/16 |
| CaB7, CaB15 | CaB_n with t = 2 - sqrt 2, as proposed | log2(K) + 4/7, log2(K) + 5/15 |
| F8t, F16t, CaB7t, CaB15t | the same, with t searched on each matrix over 0.30, 0.35, ..., 1.20 and t = 2 - sqrt 2 | the same. The 16 bits for t are not counted |
| V16, V64 | the vortex equilibrium of lowest energy found | 2, 3 |
| LBG16, LBG64 | 16 or 64 points in the plane fitted by Lloyd's algorithm to a separate sample of the family. Synthetic families only | 2, 3 |
| S | q evenly spaced levels, arm S of the first record | log2(q), q = 2 to 24 |
| E8 | the E8 Voronoi code with the shrink rule, arm E8 of the first record | log2(q), q = 2 to 24 |

- K is 2, 3, 4, 8 and 16.
- Every arm finds the nearest codeword of its codebook exactly. `run.py` checks the rule for F_n and CaB_n against a search over the whole codebook on 40 small codebooks before anything else runs.
- Every arm has one free scale per matrix, found by the search of the first record. `run.py` imports that search, the scalar arm and the E8 arm from the first record's program.
- The arms with t searched are there so that a loss is a loss for the family and not for one value of t.

**Equal bits.** The candidates' bits are not whole logarithms. A reference at the candidate's bits is a mix: a share of the blocks rounded at q levels and the rest at q + 1, with the share set so that the mean bits are the candidate's. Its error is taken as the weighted mean of the two measured errors. That is its expected error when the blocks are picked at random. The mix is run as written on two matrices at 2.5 bits as a control.

The mix is a weak reference on purpose. It is a real scheme, and anything smarter at the same bits would do better.

## Data

- **G.** Independent Gaussian entries, 512 by 1680, three seeds.
- **T.** Independent Student t entries with 4 degrees of freedom, 512 by 1680, three seeds. Its kurtosis is infinite. The report asks for a source with kurtosis 200, which is Student t with 4.03 degrees of freedom.
- **Real.** The 32 matrices of the first record, from the same file (SHA-256 of `weights.npz`: `b6c8ee500656e04e8e57c2949a5253f0dda002ba36cd1561846574dbcf01132e`). The file was downloaded for this run from `mlx-community/whisper-base.en-mlx` and its hash checked.

The width 1680 is divisible by 2, 7, 8, 15 and 16, so every arm rounds the same weights. A real matrix is read row by row into rows of 1680, and the last 64 or 256 weights are left out. Blocks of a real matrix therefore do not follow its rows.

## Measure

Relative squared error in dB, as in the first record. The gap of a candidate is its dB minus the dB of a reference mix at the same bits on the same matrix. A positive gap means the candidate is worse.

This measures weights only. The second record found that weight error does not say what a model's output does.

## Predictions

1. **Cells.** For n = 2, 3, 4, 7, 8, 15 and 16, at the incentre and at the centroid, the measured G of F_n is within 4 standard errors of the table above. The measured G of CaB_7 and of CaB_15 is above 1/12.
2. **Decoding.** On 200,000 points of the cube for each of n = 2, 3, 4 and 5, the point the pyramid rule names is never farther than the nearest point found by search, beyond rounding error.
3. **Gaussian data, K of 3 and more.** Every candidate on the incentre lattice, as proposed and with t searched, has a larger error than the scalar mix and than the E8 mix on every one of the three matrices.
4. **Gaussian data, K = 2.** Every candidate has a larger error than the E8 mix on every matrix. Against the scalar mix no direction is predicted. The hand estimates put it within 0.5 dB either way.
5. **Heavy tails, K of 3 and more.** Every candidate has a larger error than both mixes in the mean over the three matrices. At K = 2 nothing is predicted.
6. **Real matrices, K of 3 and more.** Every candidate has a larger error than the E8 mix on at least 24 of the 32, and than the scalar mix on at least 24 of the 32.
7. **Vortex codebooks.** V16 has a larger error than E8 at q = 4, and V64 than E8 at q = 8, on every matrix of G and T. Each has a larger error than the fitted codebook of the same size in the mean over G. Against scalar rounding nothing is predicted.
8. **Floors.** No row on G is under Shannon's floor for its bits by more than 0.05 dB. A violation means a bug, and the run is void until it is found.

The report's target counts as met on a family if some candidate as proposed beats both mixes in the mean over that family at some K. If predictions 3 to 6 hold it is met nowhere at K of 3 and more, and K = 2 is read from the table.

Reported without a prediction: K = 2 on T and on the real matrices, every candidate against the scalar mix at K = 2, the values of t the search picks, the 20 lowest equilibria, and the vortex codebooks on the real matrices.

## What this run cannot show

- Whether any arm gives a better model. That needs a model and a score, as in the second record.
- Anything about the scale rule that the second record found to matter. Every arm here has one scale per matrix, found by least squared error.
- Anything about the report's other proposals: the fold as a training schedule, codebooks from realisations of a graph, the calibration loss, searched level tables, circuits for decoding, and interval certificates. `notes/pr6-proposal-check.md` reads those against the limits.
- Anything about vortex equilibria that a downhill search does not reach. Those are the unstable ones.
- Decoding speed. Nothing here is timed.

## Extension 1: the 880 realisations of a graph as a codebook

Written 2026-11-17 while the main run was in progress. No result of the main run had been read: `results.out` was empty. The program is `l880.py`.

**Why it was added.** The plan above lists "codebooks from realisations of a graph" among the proposals this run cannot show, because the source check had reported that the report's own author could not open that repository. A read of Foundation-0/rigid-realizations at commit `298926c` found the table: `l880_certificate.json` holds 440 certified points, each the coordinates of the 7 free vertices of the 10-vertex graph L880. So the third experiment of PR #6 can be measured in part.

**The proposal.** Take the 880 realisations (the 440 and their mirror images) as a codebook for blocks of 14 weights. That is log2(880)/14 = 0.6987 bits per weight. The report's milestone for it is to "match 2-bit" methods with a table of 12,320 bytes.

**What is known before the run.**

- For Gaussian weights no code at 0.6987 bits has a relative error under 2^(-2 x 0.6987) = -4.21 dB. The best scalar quantiser at 2 bits is at -9.30 dB (`bounds/bounds.out`). So on Gaussian weights the milestone is closed for weight error by Shannon's floor, whatever the codebook, by 5.1 dB. The run cannot change that. It asks only how good this codebook is at its own bits.
- All 17 bars have a length between 1.003 and 1.015. So in every realisation each free vertex sits about one unit from its neighbours, and the 880 points cannot spread through 14 dimensions the way a good codebook does.
- A scalar mix at the same bits, 69.87 percent of the weights at one bit and the rest set to zero, has an expected error on Gaussian data of 0.3013 + 0.6987 x 0.3634 = 0.5552, which is -2.56 dB. This uses the proved figure for the best 2 levels.
- Seen: `l880.py --smoke`, which prints `ok` and no number, and the keys and first row of the certificate file. No arm has been scored on a matrix of the study.

**Arms**, all at 0.6987 bits, each with one free scale per matrix and an exact search over the 880 points:

| Arm | Codebook |
|---|---|
| L880 | the 880 realisations, moved to mean zero |
| L880w | the same, mapped linearly so that the codebook's covariance is the identity. This gives the set its best linear fit to data with no preferred direction |
| random | 880 independent Gaussian points, one fixed draw |
| fitted | 880 points fitted by Lloyd's algorithm to a separate sample of the family. Synthetic families only |
| S-mix | 2 levels on a share of the weights and zero on the rest, as a weighted mean of the two errors |

**Data.** The matrices of the main run, in blocks of 14.

**Predictions.**

- E1. Every one of the 440 stored points meets the 17 bar lengths to within 1e-6. This checks that the file is read correctly. It does not check the certificate.
- E2. On G, L880 and L880w each have a larger error than the random codebook on every one of the three matrices.
- E3. On G, no arm is under Shannon's floor by more than 0.05 dB.

Against the scalar mix nothing is predicted.

**What it cannot show.** Anything about the 17 bar lengths as trained parameters, which the report also proposes: the run uses the lengths of the certificate. Anything about chaining three such blocks. As the report describes it, that is a product of three copies of one codebook, and a product has the error per weight of its factor.

## Amendment 1: a smaller run

Written 2026-11-17 after the first attempt at the main run was stopped. No result existed: the program wrote its output only at the end, `results.out` was empty, and nothing had been read. Extension 1 had not been run either.

**Why.** The first attempt ran for 40 minutes on 4 cores. A timing made then put the whole run at about 19 CPU-hours, more than the session allows a background command. The time had been estimated wrongly before the plan was committed.

**What changes.**

| | In the plan above | In the run |
|---|---|---|
| Synthetic matrices | 512 by 1680 | 128 by 1680, 215,040 weights each, from the same seeds |
| Real matrices | all of each, in rows of 1680 | the first 131,040 weights of each: 78 rows of 1680 |
| K on the real matrices | 2, 3, 4, 8, 16 | 4 and 16 |
| Levels of the reference arms | every q from 2 to 24 | the q the mixes need at the candidates' bits, and 4 and 8 for the codebooks in the plane: 16 values on the synthetic families, 10 on the real matrices |
| Values of t searched | 0.30 to 1.20 in steps of 0.05, and the incentre | 0.4 to 1.0 in steps of 0.1, and the incentre |

The arms, the scale search, the mixes, the measure, the vortex search and the checks are as planned. The program now writes each finished task to a file, so a run that is stopped can be taken up again.

**The predictions** stand as written, with two changes that follow from the table:

- Prediction 6 is about K = 4 and K = 16 on the real matrices, not every K of 3 and more.
- "Reported without a prediction" loses K = 2 on the real matrices, which is no longer run.

Extension 1 uses the matrices of the run, so it too runs on the smaller ones.

**What this costs.** A synthetic matrix of this run has 0.82 of the weights of a matrix of the first record, where the error of a scalar arm varied by 0.010 dB between matrices. The real matrices are a part of each matrix and not the whole.
