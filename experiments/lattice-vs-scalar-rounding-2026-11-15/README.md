# Lattice rounding against scalar rounding at equal bits per weight

## Question

A brief given to the project proposes to balance the rows and columns of a weight matrix and then round blocks of 8 weights to the lattice A8\*, at about 2 bits per weight. It predicts lower weight error than balancing followed by scalar rounding. Is that so, and how do A8\* and E8 compare with scalar rounding where the limits are known?

## Status

`complete`. It measures weight error only.

## Preregistration

[PREREG.md](PREREG.md), committed and pushed at `195f06e` before the run. `results.start` holds the start time and that commit. No arm had been run on a matrix of the study before then. The plan names one figure from the toolkit's tests that had been seen.

## How to rerun

```
python3 experiments/lattice-vs-scalar-rounding-2026-11-15/run.py --weights /path/to/weights.npz --workers 9 > results.out
python3 experiments/lattice-vs-scalar-rounding-2026-11-15/detail.py > detail.out
```

- `weights.npz` is the file of `mlx-community/whisper-base.en-mlx`. The program refuses a file whose SHA-256 is not the one in the plan. Without `--weights` it runs the synthetic families only.
- Seeds are fixed in `run.py`. The run took 15 minutes with 9 workers on a 10-core Apple silicon machine, Python 3.12, NumPy 2.5.4.
- `run.py --smoke` runs the code path on a small matrix that is not in the study.

## Results

All numbers are relative squared error in dB, or differences of it, from `results.out` and `detail.out`. A gain is against scalar rounding on the same matrix at the same number of levels q. Arms are defined in the plan: S is scalar, Z8, A8\* and E8 are lattice codebooks with a simple rule for blocks out of range, and D+ means the matrix was balanced first.

### The four predictions

| # | Prediction | Outcome |
|---|---|---|
| 1 | No arm on Gaussian data is under Shannon's floor, and scalar rounding is not under the scalar floor | The first part holds on all rows. The second fails by a hair: see Deviations |
| 2 | E8 beats scalar rounding on Gaussian data at 2, 3 and 4 bits on every seed | Holds: +0.85, +1.27 and +1.53 dB |
| 3 | E8 is at least as good as A8\* on Gaussian data at every q | Holds |
| 4 | The brief's claim: at 2 bits, balanced A8\* beats balanced scalar rounding by at least 0.2 dB on the family built for balancing, and on the real matrices with a gain on at least 24 of 32 | Split. On the synthetic family the gain is 0.16 dB (0.13 to 0.18), under the mark. On the real matrices it is 0.39 dB with a gain on 31 of 32, over the mark |

### Gain over scalar rounding at 2 bits (q = 4), in dB

| Data | A8\* | E8 | A8\*, exact search | E8, exact search |
|---|---|---|---|---|
| Gaussian | +0.16 | +0.85 | +0.79 (1 matrix) | +1.05 (1) |
| Heavy tails | +0.35 | +1.67 | | |
| Row and column scales, balanced | +0.16 | +0.87 | +0.83 (1) | +1.06 (1) |
| Real, 32 matrices | +0.41 | +1.17 | +1.13 (2) | +1.21 (2) |
| Real, balanced | +0.39 | +1.18 | +1.05 (2) | +1.35 (2) |

On Gaussian data at other rates: A8\* gives -0.08 dB at 1.58 bits, +0.43 at 3 bits and +0.56 at 4. E8 gives +0.57, +1.27 and +1.53.

### What the run shows

1. **E8 beats scalar rounding on every matrix of the run.** The mean gain per family and rate runs from 0.46 to 2.93 dB.
2. **A8\* beats scalar rounding from 2 bits up, by 0.16 to 0.86 dB in the mean of each family.** On the real matrices it loses on at most 2 of 32. At 1.58 bits it loses in three of the six synthetic cells. Wherever A8\* gains in the mean, E8 gains more than twice as much.
3. **The brief's claim is met on the real matrices and missed on the synthetic family.** With exact search the A8\* codebook clears the mark on each of the four matrices tried: 0.57 to 1.68 dB against plain scalar rounding and 0.77 to 1.34 dB with both balanced. So the shortfall comes from the rule for blocks out of range and not from the codebook.
4. **That rule costs each lattice a different amount.** Against exact search at 2 bits it loses 0.60 to 0.75 dB for A8\* and 0.19 to 0.26 dB for E8. With cubic cells (arm Z8) it loses 0.10 to 0.41 dB against plain clipping. Exact search over 65,536 codewords is a table search, so it shows what the codebook can give and not a method.
5. **Balancing helps the real matrices.** Balanced scalar rounding beats plain scalar rounding by 0.76 dB at 2 bits and 1.12 dB at 3, with a gain on 29 and 30 of 32. Its two scale vectors cost 0.04 to 0.06 bits per weight, which is worth 0.24 to 0.38 dB at 6.02 dB per bit.
6. **Balancing and E8 together** beat plain scalar rounding on the real matrices by 1.94 dB at 2 bits (0.99 to 4.34) and 2.76 dB at 3 bits. A search found no published use of this pair.
7. **Much room is left.** On Gaussian data at 2 bits, scalar rounding is at -9.26 dB, E8 at -10.11, E8 with exact search at -10.30, and Shannon's floor at -12.04.

[Reading] E8's gain at 4 bits, 1.53 dB, is more than the 0.65 dB that the shape of its cells is worth at high rates. The range of a Voronoi code is a scaled cell of its lattice, which is rounder than a cube, so less is lost to clipping. The run does not separate the two effects.

### Against the best scalar quantiser

Arm S uses evenly spaced levels. The best scalar quantiser with free levels is a proved floor for any scalar scheme (`bounds/bounds.out`). On Gaussian data E8 is ahead of that floor by 0.58, 0.81, 0.94 and 0.71 dB at 1.58, 2, 3 and 4 bits. A8\* is ahead by 0.12 dB at 2 bits and 0.10 at 3, and behind by 0.07 at 1.58 bits and 0.27 at 4.

## Deviations from the plan

- **The scalar floor check had no tolerance.** At q = 3, arm S is under the scalar floor by 0.005 to 0.012 dB on the three Gaussian seeds. One standard error of the error on a matrix of 262,144 weights is 0.016 dB, and the scale is tuned on the sample. The plan called any violation a bug. This one is sampling error, and the run was kept. At q = 4, 8 and 16 arm S is within 0.06 dB of the best evenly spaced levels.
- **The centre of the codebook is estimated for q = 8 and 16**, from 2^20 random codewords. It is exact for q = 3 and 4. The plan said the codebooks are shifted to mean zero.
- **`detail.py` was written after the run.** Its tables by kind of matrix and by seed were not planned.

## Limits

- **Weight error only.** Nothing here says a model rounded this way works better. That needs activations or a task.
- **One real model**, a small speech transformer, and 32 of its matrices. No language model.
- **One scale per matrix and no calibration data.** Published methods use more of both.
- **The rule for blocks out of range is the simplest one.** The exact-search arms ran on four matrices only.
- **Balanced arms use more bits** than unbalanced ones, by the amount stated above.
- **Three seeds** per synthetic family.
- **The orientation of A8\*** against the weight axes is one fixed choice.

## Read against the closure rule

Nothing is closed by this run. It is a measurement.

What is proved sits in `bounds/bounds.out`: no code passes Shannon's floor, and at high rates A8\* cannot match E8, because their constants are exact and E8's is smaller. At 2 bits the order of the two lattices is measured here and not proved.

## Disclosure

The plan, the programs and this record were written by an AI agent session under the owner's direction. The two toolkits were built by agent sessions to a written specification and their tests were rerun before the plan was committed.
