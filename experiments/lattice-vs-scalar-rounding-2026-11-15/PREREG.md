# Plan: lattice rounding against scalar rounding at equal bits per weight

Written 2026-11-15, before the run. The run is `run.py` in this folder at the commit that adds this file.

## Question

A brief given to the project proposes this pipeline for weights at about 2 bits each: balance the rows and columns of a weight matrix, then round blocks of 8 weights to the lattice A8\* in place of rounding each weight alone. It predicts lower error than balancing followed by scalar rounding.

Does it, at equal bits per weight? And how do A8\* and E8 compare with scalar rounding on plain Gaussian data, where the limits are known?

## What is known before the run

- **A floor for every code.** For independent Gaussian weights, no code at R bits per weight has relative squared error below 2^(-2R). This is Shannon's rate-distortion function. It is -12.04 dB at 2 bits.
- **A floor for scalar rounding.** The best fixed-rate scalar quantiser on Gaussian data (Lloyd-Max) has error 0.1175 with 4 levels, 0.03455 with 8 and 0.00950 with 16. Evenly spaced levels are a special case, so arm S cannot beat these.
- **What lattice shape is worth at high rates.** With fine cells and no clipping, error is proportional to the normalised second moment G of the lattice. G is 1/12 for the cubic lattice, 0.075972 for A8\* and 0.071682 for E8. That is 0.40 dB for A8\* and 0.65 dB for E8. No arrangement of cells in 8 dimensions is worth more than 0.73 dB, and none in any dimension more than 1.53 dB.
- **Published work.** NestQuant rounds weights to E8 with the codebook used here. QuIP# uses an E8 codebook. SINQ balances rows and columns before scalar rounding. A search found no published use of A8\* for weights and no published combination of balancing with a lattice.

At 2 bits the cells are not fine and much of the data is clipped, so the high-rate figures do not settle the question.

Seen before writing this plan: one figure printed by the toolkit's tests (E8 at q = 4 on 512 Gaussian blocks, one scale, codebook not centred: error per dimension 0.0774 with the shrink rule and 0.0760 with exact search). No arm has been run on any matrix of the study.

## Arms

Every arm stores log2(q) bits per weight. Blocks are 8 consecutive weights along a row.

| Arm | Codebook | When a block is outside the codebook's range |
|---|---|---|
| S | q evenly spaced levels per weight | each weight is clipped to the end level |
| Z8 | Voronoi code of the cubic lattice, q^8 points | the block is shrunk toward the codebook's centre |
| A8\* | Voronoi code of A8\*, q^8 points | the same shrink rule |
| E8 | Voronoi code of E8, q^8 points | the same shrink rule |
| A8\*-opt, E8-opt | the same codebooks at q = 4 | exact search for the nearest of the 65,536 codewords |

- A Voronoi code keeps one lattice point from each class of the lattice modulo q times itself. It needs no table. The codebooks are shifted so that their mean is zero.
- Z8 has the same codewords as S. It differs only in the rule for blocks out of range, so S against Z8 measures that rule.
- The exact-search arms show the most the two codebooks can give at q = 4 under any rule. They run on four matrices: `G/0`, `RC/0`, `encoder.blocks.0.attn.query.weight` and `decoder.blocks.5.attn.out.weight`.
- Each arm also runs after balancing, written D+S, D+Z8 and so on. Balancing is 16 rounds of dividing each row and then each column by its standard deviation. The two scale vectors cost 16 bits per entry, which is 0.0625 bits per weight for a 512 by 512 matrix. That cost is reported and not hidden in the comparison.
- Each arm has one free scale per matrix. `run.py` chooses it to minimise the error on that matrix: a grid in half octaves, widened if the best value is at an edge, then ten steps of golden-section search. The shrink rule uses 12 halvings.

## Data

- **G.** Independent Gaussian entries, 512 by 512, three seeds.
- **T.** Independent Student t entries with 4 degrees of freedom, 512 by 512, three seeds. Heavy tails.
- **RC.** A Gaussian matrix with each row and each column multiplied by its own log-normal factor (sigma 0.5), three seeds. This is the case balancing is built for.
- **Real.** The 32 weight matrices of encoder blocks 0 and 5 and decoder blocks 0 and 5 of `mlx-community/whisper-base.en-mlx`, from a copy already on the machine (SHA-256 of `weights.npz`: `b6c8ee500656e04e8e57c2949a5253f0dda002ba36cd1561846574dbcf01132e`). 24 are 512 by 512 and 8 are 2048 by 512 or 512 by 2048. It is a speech model, not a language model. It is the only float transformer on the machine, and nothing is downloaded for this run.

Synthetic matrices run at q = 3, 4, 8 and 16 (1.58, 2, 3 and 4 bits). Real matrices run at q = 4 and 8.

## Measure

Relative squared error, the squared Frobenius norm of the difference over that of the matrix, in dB. The gain of an arm over scalar rounding is the difference in dB on the same matrix at the same q.

This measures weights only. It says nothing about the output of a model.

## Predictions

1. **Floors hold.** No arm on G is under the Shannon floor by more than 0.05 dB, and S on G is not under the scalar floor. A violation means a bug, and the run is void until it is found.
2. **E8 beats scalar rounding on G** at q = 4, 8 and 16, on all three seeds.
3. **E8 is at least as good as A8\* on G** at every q, in the mean over seeds.
4. **The brief's claim.** At q = 4, D+A8\* beats D+S by at least 0.2 dB in the mean over RC, and by at least 0.2 dB in the mean over the 32 real matrices with a gain on at least 24 of them.

The brief's claim counts as refuted on a family if the mean gain there is under 0.2 dB. If the exact-search arm also gains under 0.2 dB on the matrices where it runs, the codebook is at fault and not the shrink rule.

Reported without a prediction: Z8 against S, the exact-search arms against the shrink rule, D+S against S, q = 3, and E8 against scalar rounding on the real matrices.

## What this run cannot show

- Whether any arm gives a better model. That needs activations or a task.
- Anything about methods that use calibration data, or about codes with more than one scale per matrix.
- Anything about language models. The real matrices are from one small speech model.
