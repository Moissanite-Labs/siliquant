# Check of the research report in PR #6

Date: 2026-11-17
Status: done for the claims listed. Three studies were run for it. Each has its own record.

PR #6 adds a saved web page: a report written by a research assistant model for the project owner on 2026-11-16. It proposes four tracks and five experiments that carry mathematics from Foundation-0 into this repository, and a four-week plan.

This note reads the report the way the repository's rule asks: against the table of proved limits, before anything is run. Where a claim could be computed here, it was. The last column of the table says which.

**Checked** means the source was opened or the number computed by the session that wrote this note. **Reported** means it comes from a read-only agent's pass over the sources and was not re-opened here.

## Result

- **The report was written without this repository's records.** It says it could not open `experiments/`. So none of its tracks uses what the two records found: that weight error does not say what the output does, and that the output followed how the scale is set.
- **Three of the five experiments could be measured here in part, on weight error. None meets its target.**
  - The codebook of the first, as proposed, loses to scalar rounding and to E8 at equal bits at every rate tried. Its cells are worth 1.2 to 2.0 dB less than a cube.
  - The codebook of the third is worse than the same number of random points, and its milestone is closed for weight error by Shannon's floor for Gaussian weights.
  - The vortex codebook of the fourth is about as good as the best scalar quantiser and 0.8 to 1.7 dB behind E8.
- **The end point of the second is behind E8 by a proved ceiling.** It is a lattice in 3 dimensions, where no cells are worth more than 0.35 dB. The E8 cell is worth 0.65.
- **Two of its arguments are wrong as stated**: the calibration loss of the second track, and the claim that finding the best level table needs a certified search.
- **Its account of published work is mostly right.** Six of its statements are not in the sources it names or are attributed wrongly (rows 27, 28, 29, 32, 35 and 37), and the nearest prior work to this repository, NestQuant, is missing from it.
- **Following its first track led to three things it does not contain.**
  - Under the rule that clips nothing, most of E8's gain over scalar rounding comes from the shape of the code's region and not from the lattice ([e8-gain-largest-entry-rule-2026-11-17](../experiments/e8-gain-largest-entry-rule-2026-11-17/README.md)).
  - E8 with another region beats the E8 Voronoi code in weight error at no more bits ([region-shapes-no-clip-rule-2026-11-17](../experiments/region-shapes-no-clip-rule-2026-11-17/README.md)).
  - The one part of the first track's codebook that helps is the single moved weight per block, and it helps most when moved far from the incentre, as an escape for an outlier ([incentre-vortex-codebooks-2026-11-17](../experiments/incentre-vortex-codebooks-2026-11-17/README.md)).

All of this is weight error on synthetic matrices and one small speech model. Nothing here was scored on a language model.

## What the report did not have

The report says of itself that it could not open `experiments/` in this repository. Three findings of the records bear on every track:

- **Weight error does not say what the output does.** The first experiment is scored on weight error, and the second on a layer-wise stand-in for it.
- **The output followed how the scale is set.** No track mentions the scale rule.
- **No formula tried predicts the damage.** The largest singular value of the weight change was the best of five and still ranks a clip above a rounding that does less damage. The fourth track proposes a certificate for that singular value.

It also says it could not open rigid-realizations and relative-equilibria. Both clone without credentials (**reported**, and **checked** for rigid-realizations).

## Claims

### First track: the incentre lattice

| # | Claim in the report | Verdict | What was found | Basis |
|---|---|---|---|---|
| 1 | F_n is the set of points +-c e_i with c = 2 - sqrt 2, repeated over 2Z^n | Confirmed | It matches `results/rigidity-nd/PROOF.md` of incentre-lattices. That repository holds no work on rounding weights | source, checked |
| 2 | F_n has an exact decoder with no search, in any dimension | True, and less than it seems | Proved in the record of the first study. It holds for any position of the points on the axes. The decoder is scalar rounding and one search for the largest remainder | proved, and recomputed |
| 3 | This removes the decoding problem of lattice codes | Wrong comparison | It is fast because the cells are close to those of the cubic grid. E8 has had a nearest-point rule with no table since 1982 (`notes/brief-check.md`, claim 3) | derived |
| 4 | F_n is a good codebook, from 1.3 bits per weight up | Wrong as proposed | The cell is a pyramid, worth 1.2 to 2.0 dB less than a cube with fine cells (`bounds/bounds.out`, section 7). At equal bits F_8 and F_16 lose to a scalar mix by 0.2 to 1.3 dB and to an E8 mix by 0.8 to 3.5 dB, on Gaussian and Student t data at every rate from 1.3 to 4.5 bits | derived, and recomputed |
| 5 | Adding the cell centre and a corner (CaB-VQ) removes the corner error. 16 and 32 states at n = 7 and 15. Three candidates are enough to decode | Counts and rule right, quality wrong | The rule is exact. The cells are worth 1.04 and 0.68 dB less than a cube, and as proposed it loses at equal bits on every family | recomputed |
| 6 | Packing fractions 0.2233, and 0.746865 for the CaB6 packing, above close packing in a narrow window | Confirmed, and beside the point | They match `results/cab6-packing/PROOF.md`, which is about balls of two sizes. A packing fraction says how balls fit. For rounding the figure is the second moment of the cell | source, checked |
| 7 | Target: surpass E8 and scalar rounding on heavy-tailed data at 1.5 to 2.5 bits | Not met | On Student t data, as proposed: behind the scalar mix by 0.0 to 0.8 dB and behind the E8 mix by 1.2 to 3.5 dB. With the position of the points free, the best is five to seven times the incentre's. That code beats the scalar mix, and beats the E8 mix only under 2 bits. It was found by a sweep that was not planned | recomputed |
| 8 | The fold of F_3 to close packing as a training schedule that avoids approximate rounding | End point closed by a ceiling. Schedule not tested | The end point is the face-centred cubic lattice, worth 0.25 dB. No cells in 3 dimensions are worth more than 0.35 dB, and the E8 cell is worth 0.65 (`bounds/bounds.out`, section 9). That lattice has an exact nearest-point rule already, in `lattice/`. The repository labels its path to close packing in the larger cell as checked by sampling and not proved | derived, and source checked |
| 9 | The n(n-1)/2 zero modes of F_n give a way to parametrise rotations of blocks | Nothing specific to F_n | By the theorem in `PROOF.md` the zero modes are the turns of each cross-polytope. Those are the rotations of n dimensions, which have n(n-1)/2 parameters for any block | source, checked |

### Second track: realisations of a graph

| # | Claim in the report | Verdict | What was found | Basis |
|---|---|---|---|---|
| 10 | The 880 realisations of L880 are a codebook for 14 weights that fits in 12,320 bytes | The table exists. The codebook is poor | `l880_certificate.json` holds 440 points and each meets the 17 bar lengths to 4e-15. As a codebook they are behind 880 random points by 0.8 to 0.9 dB, and by 0.4 to 0.6 dB after the best linear map | checked, and recomputed |
| 11 | Milestone: match 2-bit methods with this table | Closed for weight error on Gaussian weights | The codebook has 0.70 bits per weight. No code at those bits is under -4.21 dB. The best scalar quantiser at 2 bits is at -9.30 dB | proved, `bounds/bounds.out` |
| 12 | Chaining three blocks gives 880^3 codewords for 42 weights that are not a product | Not a gain | As described it is three copies of one codebook, which has the error per weight of one | derived |
| 13 | The 17 bar lengths can be trained, with a certificate that no realisation is lost | Not evaluated | No construction is given | none |
| 14 | Chains of vertices, each added by two bars, give a codebook with no table | Not evaluated | No encoder is given. Choosing the branches that match a block is a search over all of them | none |
| 15 | A calibration loss that matches distances between outputs along a sparse rigid graph, so that harmless rotations are not penalised | Wrong as stated | The loss is zero for QW with any orthogonal Q. With Q = -1 it is zero and every output is reversed. The output of a layer is read by fixed weights, so a rotation of it is not harmless. The exact symmetries of the model are the three scalings in the second record. Layer-wise methods do not match an N by N table of outputs, which is what the loss is said to replace: they use the second moments of the inputs | derived |
| 16 | All 880 complex realisations of L880 can be real | As that repository states it | Its README labels the result as claimed, pending independent verification. Not checked here beyond the bar lengths of the stored points | reported |

### Third track: equilibria

| # | Claim in the report | Verdict | What was found | Basis |
|---|---|---|---|---|
| 17 | Vortex equilibria make good codebooks for pairs of weights | About as good as the best scalar quantiser | 16 points: -9.34 dB on Gaussian data. 64 points: -14.56. The best scalar quantisers: -9.30 and -14.62. E8: -10.12 and -15.55. A codebook fitted to the data: -9.67 and -15.22 | recomputed |
| 18 | The two identities for the sum and the sum of squared sizes | Confirmed | They hold to 4e-10 on every equilibrium found | recomputed |
| 19 | A changed force law would suit heavy tails | Not evaluated | | none |
| 20 | relative-equilibria holds results to build on | No | Its README says "Status: not started" | checked |
| 21 | The best level table has to be found by a certified search, because the problem has many local optima | Wrong for the plain problem, and the wrong target here | For a density whose logarithm is concave, such as the Gaussian, the best scalar quantiser is the only local optimum. `bounds/bounds.py` computes it. And the second record found that the scale of least squared error gave a worse output than the largest-entry rule in all four cells tried. A certificate that a table has the least squared error certifies what that record found not to be the aim | source reported, and the second record |
| 22 | point-charges certifies 9, 24 and 29 equilibria for 4, 5 and 6 charges, and 44 for 7 | Matches that repository | The line for 6 charges was checked. Issue #4 says "4 charges with 11 balance points". The README gives 9 for 4 charges and nothing supports 11 | reported, one line checked |

### Fourth track: circuits and certificates

| # | Claim in the report | Verdict | What was found | Basis |
|---|---|---|---|---|
| 23 | Circuits from gf2m-circuits and mobius-x1 can decode weights with no table | Not evaluated | A clone of either asked for credentials. The report gives no reason why bit patterns made by such a circuit would be close to weights | none |
| 24 | Codes from gfq-codes lift to lattices with a guaranteed minimum distance | No codes exist yet, and the wrong figure | Its `codes/README.md` says "No code has been added yet". A minimum distance is a packing figure. In more than 8 dimensions the best lattices known are worth 0.10 to 0.37 dB more than E8 by their cells (`bounds/bounds.out`, section 8). A code on the Leech lattice for weights is published | checked, and source |
| 25 | An interval certificate for the largest singular value of the weight change guarantees the error of a deployed model | Certifies a measure that does not rank damage | In the second record that measure ranks a clip as better than a rounding that does less damage. A bound on one layer in the worst case is not a statement about the output of 42 | the second record |

### Published work

| # | Claim in the report | Verdict | What was found | Basis |
|---|---|---|---|---|
| 26 | A survey of August 2026 covers 200 works and 43 transform methods | Confirmed | Jokar, arXiv 2608.25188 | checked |
| 27 | It says activations have kurtosis above 200, which a Hadamard transform brings to about 3 | Not in it | The survey argues that the quantity that matters is the largest value over the root mean square, "not the fourth-moment kurtosis" | checked |
| 28 | It says learned rotations alone nearly match lattice rounding at 4 bits | Not found | | reported |
| 29 | It names two open problems | It names four | The two quoted are among them | checked |
| 30 | QuIP# uses an E8 codebook with a transform and fine-tuning | Confirmed | 65,536 codewords decoded from a table of 256 entries | reported |
| 31 | QTIP uses trellis codes | Confirmed | | reported |
| 32 | SpinQuant's online transforms cost 0.4402 against 0.4489 seconds | Wrongly attributed | The two numbers are from an example in llm-compressor for a model with another method's transform. No hardware is stated | reported |
| 33 | HIGGS is limited to about 1,024 table entries | Confirmed | | reported |
| 34 | GLVQ learns lattices and rounds approximately while training | Confirmed | | reported |
| 35 | LLVQ uses the Leech lattice and its kernel is slower than QTIP's because of its arithmetic | Roughly right | The paper says its implementation is "slightly slower than QTIP". It gives no cause. Its table 3 gives its figures in bits and not in dB, by the agent's reading | checked for the quoted line, the rest reported |
| 36 | SHq uses the sign bit of a block's scale to choose between a level table and its mirror image | Confirmed | Helcig, Killian, Alistarh, arXiv 2611.02966 | checked |
| 37 | A new GPU has an instruction for 3-bit level tables | Roughly right | The instruction is in the vendor's documentation. The report's statement that blocks share a scale of that type is wrong for it | reported |

Not in the report: NestQuant (arXiv 2502.09720), which rounds weights to E8 with a Voronoi code and no table. It is the nearest published work to this repository's E8 arm and is cited in `notes/brief-check.md`.

## The five experiments

| Experiment of the report | Its target | Read against the limits, or measured | Where it stands |
|---|---|---|---|
| 1. Incentre and CaB codebooks | Surpass E8 and scalar rounding on heavy-tailed data at 1.5 to 2.5 bits, with a decoder that needs no table | Measured on weight error | Fails as proposed. With the points moved far out it beats a scalar mix, and an E8 mix under 2 bits. That code is no longer the incentre lattice |
| 2. Rounding along the fold | Lower layer-wise error than rounding to a learned lattice | The end point is in 3 dimensions: at most 0.35 dB by its cell, against 0.65 for E8 | Closed for the cell at the end point. The schedule was not run. Its measure is a layer-wise error. The second record tried a simpler form of one, and it ranked variants worse than the largest singular value did |
| 3. The 880 realisations with trained bar lengths and the rigid loss | Match 2-bit methods with a table of 12,320 bytes | Shannon's floor at its bits, and a measurement | Closed for weight error on Gaussian weights. The codebook is worse than random points. The loss is wrong as stated |
| 4. Vortex codebooks and certified level tables | Lower KL than SHq, with a certificate | The vortex half measured on weight error | The vortex codebook is no better than the best scalar quantiser. The level tables were not run: they need a model. The idea they build on is SHq's, which is published |
| 5. Circuit decoders and a certified error bound | Match the speed of a published kernel, with a bound on the largest singular value | Not runnable here | Not evaluated. The bound is on a measure that does not rank damage |

## What follows

In the order this note would take them.

1. **Issue #3 stays first.** A second model and a second text for the E8 result. It should carry one more arm: E8 cut by a ball. On 29 to 32 of 32 matrices of one model that arm was 0.13 to 0.20 dB ahead of the E8 Voronoi code in weight error. Whether that moves a model's output is not known.
2. **Run the split on gemma.** The record of the E8 gain did not have the file. Its program reads any float matrix. `harness/ggufedit.py` reads the file.
3. **Decide how the points of a cut codebook are numbered.** The arms that beat the E8 Voronoi code have no packed format. A Voronoi code needs no counting and these do. The counts are exact and in the program of the region study. Without a way to number the points they stay measurements.
4. **The block on the wall.** The rule puts one block of every group on a wall of its cell, which costs 0.26 dB with groups of 32. A scale a little above the smallest allowed might get it back. Not tried.
5. **The escape under the rule that clips nothing.** The first track's code with its point far out lets one weight of a block lie beyond the range of the levels, and its points are numbered as simply as a grid's. Under one scale per matrix it beat the scalar mix. Under the rule that clips nothing it has not been run with the point far out.
6. **More dimensions.** By arithmetic on the data, a ball in 24 dimensions would give 1.1 to 1.6 dB more scale than one in 8, and the best lattice known there is worth 0.37 dB more by its cell. That needs a decoder for such a lattice, and one is published.

Not recommended as they stand: the second, third and fifth experiments, the rigid calibration loss and the two certificates. The reasons are in the tables above.

On the pull request itself: the report is a source for this note and not a record. If it is kept, the plain text is enough.

## Sources

- The report: `04_gemini_novel_quantization/page_full_text.txt` on the branch of PR #6, commit `6ecc87d`.
- Foundation-0/incentre-lattices at commit `2d374ff`: `results/rigidity-nd/PROOF.md`, `results/cab6-packing/PROOF.md`, `results/fold-close-packing/PROOF.md`. **Checked.**
- Foundation-0/rigid-realizations at commit `298926c`: `l880_certificate.json`, SHA-256 `583fec3591b377a883afe17104aa25c0835115e47eda2a3c861b8e17721cf92d`. **Checked.**
- Foundation-0/point-charges, relative-equilibria and gfq-codes: their READMEs. **Reported**, with the lines quoted in the tables **checked**.
- Lattice constants: Agrell and Allen, arXiv 2202.09605, table I. **Checked** in the text of the paper.
- The survey: Jokar, "Transforms for LLM Quantization: The Great Inversion and Format Co-Design", arXiv 2608.25188. **Checked** for the lines quoted.
- SHq: Helcig, Killian, Alistarh, arXiv 2611.02966. LLVQ: van der Ouderaa, van Baalen, Whatmough, Nagel, arXiv 2603.11021. **Checked** for the lines quoted.
- QuIP#: arXiv 2402.04396. QTIP: arXiv 2406.11235. SpinQuant: arXiv 2405.16406. HIGGS: arXiv 2411.17525. GLVQ: arXiv 2510.20984. NestQuant: arXiv 2502.09720. The example in llm-compressor. The vendor's documentation of the instruction. **Reported.**
- Uniqueness of the best scalar quantiser for a density whose logarithm is concave: Fleischer (1964), Trushkin (1982), Kieffer (1983), as cited by Dumitrescu and Wu. The originals were not opened. **Reported.**

Not reached: gf2m-circuits and mobius-x1.

## Disclosure

Written by an AI agent session under the owner's direction. The source check was delegated to a second agent, and its findings are marked above.
