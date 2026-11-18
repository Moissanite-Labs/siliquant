# siliquant

Research on rounding model weights to a few bits each (quantisation), and on the mathematics around it. It sits beside [weft](https://github.com/Moissanite-Labs/weft), the software of ML Weft, and [weft-loom](https://github.com/Moissanite-Labs/weft-loom), its research records.

The repository follows one rule from weft-loom: a path is closed only on a proof. So it keeps a table of proved limits, and a proposal is read against that table before anything is run.

## What is here

| Folder | What it holds |
|---|---|
| `bounds/` | Proved floors and ceilings for rounding weights, computed from their formulas: Shannon's floor, the best scalar quantiser, what the shape of a lattice cell is worth. Also lattice constants, covering thickness and codebook counts, the cells of the point set F_n of incentre-lattices, the best lattices known above 8 dimensions, and what the region of a code is worth when one scale serves a group: for Gaussian weights from integrals, and its limits for any weights. Start with `bounds/bounds.out`. |
| `perm/` | Exact counting of matchings that avoid forbidden pairs: derangements, rook and hit polynomials, permanents, bounds on permanents, and the closure of a set of forbidden cells, with a witness for every cell that a second function checks from the board alone. Counts are tested against full enumeration on small cases, and bounds against exact values. `perm/demo.out` holds the figures the notes quote. |
| `lattice/` | Nearest-point decoders for the lattices Z^n, D_n, E_8, A_n and A_n\*, Voronoi codes, scalar reference quantisers, and row and column balancing. Tested against brute force and against known constants. |
| `shape/` | The process of issue #10 as code, for weight error. Codebooks cut from the integers or E8 by the ball of a p-norm, counted exactly. The region part of the error, computed from the weights with nothing rounded, and a formula for the error with no fitted constant. A receipt in whole numbers. For five arms the pick in exact arithmetic with a certificate, and a checker that shares no code with it. A closed-form scale with a proved bound on the error. `shape/ballindex.py` numbers the points of a codebook cut by a ball over any number of weights, so that a block can be stored in the bits of its number; `shape/ballindex.out` gives the sizes of its tables. |
| `harness/` | Tools for experiments on real model files: read weights out of a GGUF file and write a changed copy, round on a GPU, and score a changed model against the original with llama.cpp's own tool. `harness/test_tq.py` checks the GPU code against `lattice/`. |
| `checks/` | Programs that read the repository itself: whether each record's plan was committed before its data, read from git; whether each row of a check note names a file for what it says was recomputed; and a trace of the numbers in the records, which measures how little such a trace can catch. [checks/README.md](checks/README.md) says what each found. |
| `experiments/` | One folder per study, each with its plan, program, output and limits. Index in [experiments/README.md](experiments/README.md). |
| `notes/` | [concept-map.md](notes/concept-map.md): the combinatorics the project owner listed, mapped to code, to the closure rule and to rounding. [brief-check.md](notes/brief-check.md): a source check of the two briefs the repository started from. [pr6-proposal-check.md](notes/pr6-proposal-check.md): a check of the research report in PR #6, claim by claim, and what follows from it. [issue10-proposal-check.md](notes/issue10-proposal-check.md): the same for the proposal of issue #10 and the published work it cites, with what PR #5 has that bears on it. [grafts-followup.md](notes/grafts-followup.md): what became of the fifteen crossings of the grafts note. [issue15-proposal-check.md](notes/issue15-proposal-check.md): the claims of issue #15 read against what was measured, and what issue #14 asks for. |

## How to check it

The counting code needs only Python 3. The rest needs NumPy. The harness also needs torch. The shape toolkit does not.

```bash
python3 -m unittest perm/test_permlib.py
```

```bash
python3 -m unittest lattice/test_lattices.py
```

```bash
python3 -m unittest shape/test_shapelib.py shape/test_certified.py shape/test_ballindex.py
```

```bash
python3 bounds/bounds.py
```

```bash
python3 harness/test_tq.py
```

```bash
python3 checks/plan_before_data.py && python3 checks/claim_ledger.py
```

## How records are kept

The conventions are those of [weft-loom's CONTRIBUTING.md](https://github.com/Moissanite-Labs/weft-loom/blob/main/CONTRIBUTING.md). In short:

- A study has a plan committed before its data, or says "not preregistered".
- Every number in a README can be traced to a committed output file.
- A null or negative result is recorded the same way as a positive one.
- No model weights and no datasets are committed. A record names the file it used by its SHA-256.

## Status

Eleven studies so far, one of them partial. The newest results are at the end of this section.

The first measured weight error only: at 2 bits per weight the lattice E8 beats scalar rounding on every matrix tried, and the lattice the starting briefs proposed, A8\*, gains less than half as much in the mean of each family of data.

The second measured the output of a real model, `gemma-4-E4B-it`. Weight error turned out not to say what the output does. In these runs the output followed how the scale is set more than the size of the error.

With one scale per 32 weights, set from the largest entry, E8 gives a closer output than llama.cpp's `Q4_0` at the same 4.5 bits. With one scale per matrix and no balancing, E8 is worse than scalar rounding at 4 and 5 bits. One model and one text so far.

The other three were run on 2026-11-17 to read the proposals of PR #6 and to answer issue #4. They measure weight error only, on synthetic matrices and one small speech model.

- **The codebooks PR #6 proposes do not beat scalar rounding or E8 as proposed.** The point set of incentre-lattices has cells worth 1.2 to 2.0 dB less than a cube. A vortex codebook is about as good as the best scalar quantiser. The 880 realisations of a graph are a worse codebook than 880 random points.
- **E8's gain under the largest-entry rule needs no model, and most of it is not the lattice.** On Gaussian data it is 1.36 dB with groups of 32 and 2.3 dB per row. Per row, 0.65 dB is the cell and 1.64 dB is the scale: the rule lets E8 use a smaller scale because the region of its code is nearly a ball.
- **The region can be chosen.** Integer levels cut by a ball reach 76 to 89 percent of E8's gain on the real matrices with no lattice. E8 cut by a ball beats the E8 Voronoi code by 0.13 to 0.20 dB on 29 to 32 of 32 real matrices, at no more bits. These codebooks have no packed format yet.

None of the three was scored on a language model.

The sixth, also of 2026-11-17, reads the proposal of issue #10: a formula that gives the error of a rounding from the weights alone and picks the shape of the code's region, the size of a group and the directions kept. It measures weight error on 109 matrices that no codebook had been run on, 85 of them real, from one small speech model and one small language model.

- **From 4 bits up the formula gives the error and picks the best arm.** It has no fitted constant. On the real matrices it is within 0.3 dB of the measured error in 96.6 percent of 5,525 runs, and the arm it picks among 13 is the measured best in 96 percent of 425 choices. At 2 bits it is low by 0.8 dB.
- **The best region on real weights is between the cross-polytope and the ball**, a p-norm ball with p = 5/4 or 3/2, by 0.2 to 0.3 dB. At 4.5 bits groups of 128 weights are the best size on 82 of 85 real matrices. Dropping directions loses on 83 and wins on two, and the formula says which.
- **Two limits are proved.** On a round cloud no region beats the ball. For any weights a region is worth at most 2.5 log10 of a ratio of volumes: 4.50 dB for a ball over the cube and 11.51 for a cross-polytope.
- **A rule with a proved guarantee costs about half a decibel.** With a closed-form scale nothing is clipped and the error is under a bound, by two short proofs and in 1,308 of 1,308 runs. For five arms the pick is whole-number arithmetic with a certificate. No proof assistant has run on any of it.

Nothing in the sixth is scored on a model's output, and its codebooks have no packed format.

The seventh, of the same date, cut codebooks by a ball over 8 to 128 weights. The gain grows with the block as the limits table's formula says, and with one ball per group of weights the error does not depend on the weights. Weight error only.

The eighth answers issue #9: the fold of incentre-lattices as a path of codebooks over 3 weights. The cell does improve along the fold, and its best state is its end, close packing, worth a quarter of a decibel. Every state loses to a fitted codebook, to E8 and to a longer ball.

The ninth scores a second model, `Qwen2.5-1.5B-Instruct`, on a second text, for issue #3. It is partial: 27 of 32 variants.

- **As the gemma record ran it, E8's win shrinks with groups of 32 and reverses with one scale per row.** The cause is in the harness: its E8 Voronoi code subtracts the mean of its codewords, so none of them is zero. With a codeword at zero the same code is ahead of the scalar rule at all three bit points. The change is invisible in mean weight error and worth a factor of 17 in KL per row.
- **One ball per group of weights is far ahead at equal bits.** E8 in one ball per group of 128 scores a mean KL of 0.0195 at 4.5 bits. llama.cpp's `Q4_0` scores 0.0623 at 4.5 bits and its `Q4_K_M` 0.0333 at 4.81. Integer levels with no lattice score 0.0227.
- **For those arms a decibel of weight error is a decibel of KL**, within a tenth, over nine arms from 4.0 to 4.5 bits.

The tenth scores the same arms on a third model, `SmolLM2-1.7B-Instruct`, on the same text.

- **The headline holds.** At 4.5 bits one ball per group of 128 scores 0.0560 for integer levels and 0.0609 for E8, where `Q4_K_M` scores 0.0838 at 4.82 bits and `Q4_0` 0.1271 at 4.5.
- **Two things do not carry.** Integer levels are ahead of E8 on this model, and its E8 arms gain less on the output than their weight error says. And the harness's Voronoi code is not broken per row here: a codeword at zero is level per row and a little worse with groups of 32.
- **So the arm that both models agree on is the plain one**: integer levels in one ball per group of 128 weights.

The eleventh answers issue #14: llama.cpp's files made with an importance matrix, beside the ball arms, on both models.

- **The matrix helps llama.cpp's files, by 15 percent of KL on Qwen2.5 and 55 percent on SmolLM2.** With it, `Q4_K_M` is ahead of the ball arm with no data on SmolLM2 and still behind it on Qwen2.5.
- **Given the same matrix at half strength, the ball arm is ahead on both**: 0.0209 and 0.0282 at 4.5 bits, against 0.0282 and 0.0378 for `Q4_K_M` with the matrix at 4.8 bits. At full strength it does worse. The half-strength arm was planned and not predicted to win. A sweep of five strengths on the validation text then chose that strength by a rule fixed beforehand.

What these do not show: accuracy on a task, a matrix from other text, and anything about a packed format or speed. The bits are counted from the codebook, and the saving is in the file, not in memory.

## Licence

No licence file yet. The owner has not chosen one.

## Disclosure

Most of the code and text here was written by AI agents under the owner's direction, and checked by a second agent before merging.
