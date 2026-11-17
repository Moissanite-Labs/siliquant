# Check of the proposal in issue #10

Date: 2026-11-17
Status: done for the claims listed. One study was run for it, with an extension. It has its own record.

Issue #10 proposes a process for finding the shape of a model's weights, mapping it, and turning the map into a formula that picks how to round. A comment under it lists published work by knob and says what is new. A second comment assigns the issue and asks that anything in PR #5 that bears on it be identified.

This note reads the proposal the way the repository's rule asks: against the table of proved limits, before anything is run. Where a claim could be computed here, it was. The last column of each table says which.

**Proved** means a proof is written out in the plan or the record named. **Measured** means a figure from a committed output of the record [shape-formula-held-out-2026-11-17](../experiments/shape-formula-held-out-2026-11-17/README.md). **Derived** means arithmetic from the formula or the limits table. **Checked** means the source was opened or the line found in its text by the session that wrote this note. **Reported** means it comes from a read-only agent's pass over the sources.

## Result

- **The formula of stage 4 exists and holds on weight error from 4 bits up.** It has no fitted constant. On 85 real matrices that no codebook had been run on, the measured error is within 0.3 dB of it in 96.6 percent of 5,525 runs, and the arm it picks is the measured best in 96 percent of 425 choices.
- **Its parts are not the three the proposal names.** In dB the error is a cell, a region and a rate. The proposal's shape and size are one part, the region, and cannot be priced apart. Its dropped part adds in plain units. The proposal has no knob for the lattice, which is worth 0.65 dB.
- **At 2 bits the formula is low by 0.8 dB in the mean and by up to 3.3 dB on real matrices.** "Exact for the tidy families" holds for the region part of a Gaussian cloud, which is an integral. It does not hold for the error.
- **Two proved limits bound the shape knob.** On a round cloud no region beats the ball, so for Gaussian weights the answer is known before any probe. And for any weights a region is worth at most 2.5 log10 of a ratio of volumes: 4.50 dB for a ball over the cube, 11.51 for a cross-polytope.
- **The map does not need a family.** The number the formula needs is computed on the matrix itself, exactly for three shapes. A description by the spread of single weights picks worse than "round" on the language model.
- **The receipt of stage 2 was built for three shapes**, with a checker that replays it and fails on a changed digit. For five arms the pick itself is whole-number arithmetic with a certificate. No proof assistant ran: the Lean route the proposal names was not opened and not tested.
- **Dropping directions is the weakest knob and is not dead.** It loses on 83 of 85 real matrices by 3 to 19 dB and wins on two by 2 dB. The formula says which.
- **Two of the proposal's claims were not tested.** Nothing was scored on a model's output, and no codebook has a packed format, so the decode budget cannot be read.
- **The comment's account of published work is mostly right. Two of its statements are wrong and seven need a qualification. Of the five things it calls new, three are published.** The split is from 1993. A ball-cut integer grid is plotted in QuIP#. A rule for which region and which budget is published in parts. Two were not found in print: a measured comparison of ball-cut E8 with a single-scale E8 Voronoi code at equal rate, and a replayable receipt. The search was short.

## What the proposal did not use

Three findings of the records bear on it.

- **Weight error does not say what the output does** (second record). Stage 4 minimises weight error. Its loop compares "the measured error against the formula's prediction", which is a check of the formula and not of the output.
- **The output followed how the scale is set** (second record). **And finer groups helped it by more than their weight error did**: this follows from two rows of that record, where groups of 32 at 16 levels and a row at 32 levels have the same mean KL and weight errors 1.3 dB apart. The formula's size knob is settled on weight error: groups of 128 at 4.5 bits. That is the knob most likely to be set wrongly for an output.
- **The scale has a cost on the wall and a block at zero has its own error** (the record of the E8 gain). The comment under the proposal says the split was measured with "nothing left over". It left 0.3 dB with groups of 32 and up to 0.5 dB in the mean on real matrices. Both causes are terms of the formula now.

## Claims

### What the proposal builds on

| # | Claim | Verdict | What was found | Basis |
|---|---|---|---|---|
| 1 | "The error splits into a patch-shape part and a patch-size part" | Right in substance, with other parts | The records split a gain into the cell (0.65 dB for E8) and the scale. In this note's terms: cell, region, rate. "Nothing left over" was true per row on Gaussian data. With groups of 32 the split left 0.30 to 0.35 dB, and 0.23 to 0.52 dB in the mean on real matrices | records, checked |
| 2 | "The ball cut recovers most of the tidy pattern's gain with no lattice" | Confirmed on real matrices | 76 to 89 percent of the E8 Voronoi code's gain on the 32 real matrices, 80 to 81 percent per row on Gaussian data, 59 percent with groups of 32 | record, checked |
| 3 | "The pyramid penalty was proved from a closed form" | Confirmed | The cell of each point is proved to be its pyramid. Its second moment is a closed form, checked by sampling to 1.6 standard errors (`bounds/bounds.out`, section 7) | checked |

### Stage 1: the probe

| # | Claim | Verdict | What was found | Basis |
|---|---|---|---|---|
| 4 | Four questions, each with a number: how spiky, how stretched, whether the coordinates move together, one lump or several | Made concrete, and none of the four is what the formula needs | The formula needs one number per shape and group size: the region part. The four questions are how that number moves. Spiky: per row a cross-polytope region is worth 1.3 dB over the cube on Gaussian weights and 9.3 dB on Student t weights with 3 degrees of freedom. Together: 1.75 dB on the language model, 3.1 dB on a matrix whose rows have their own scales, from a shuffled copy. Stretched: see row 13. Lumps: where the formula's fine cells fail, see row 18 | measured |
| 5 | "The weights are what they are, so the numbers are what they are" | True exactly for three shapes | For the cube, the ball and the cross-polytope the region part is a ratio of whole numbers times a known constant, since a float is a whole number times a power of two. Against 64-bit floats it differs by 6 parts in 10^16. For other shapes the comparisons are exact and the sums are floats. Singular values are floats, and are not the same to the last digit across builds of a linear algebra library. That last was not tested here | measured, and derived |
| 6 | "Any sampling is fixed and seeded" | The probe needs none | Every number is computed on all of a matrix. The one draw is the shuffled copy, seed 9100 | checked |

### Stage 2: the map

| # | Claim | Verdict | What was found | Basis |
|---|---|---|---|---|
| 7 | A short description plus a receipt | Built. The description has to be the region part | Three descriptions were tried as the basis of the pick. The region part of the matrix: regret 0.000 to 0.001 dB. The same on a shuffled copy: 0.04 to 0.10 dB on whisper and 0.25 to 0.41 on qwen. "Round": 0.46 to 0.77 and 0.09 to 0.31 | measured |
| 8 | "The receipt is a replayable computation with exact bounds, the certificate style the Foundation-0 records already use" | Built for three shapes. The style was not compared | A receipt names a matrix by the SHA-256 of its bytes and gives the whole numbers. `check_receipts.py` replayed all 109 and caught a changed digit and a changed weight. The Foundation-0 records were not opened in this session | measured |
| 9 | "The Lean rebuilds in Proven-Commons are the vehicle" | Not evaluated | Proven-Commons was not opened: this session was not given the other repositories. No proof assistant ran: installing one was not permitted. The plan of the extension lists ten statements a proof would cover, from three inequalities on 8 whole numbers to the nearest-point rule of E8 | none |
| 10 | "The fit of a real cloud to one of the tidy families is a judgment the numbers inform" | The fit is not needed, and can mislead | See row 7. A family with independent weights is the shuffled copy at best | measured |

### Stage 3: the variables

| # | Claim | Verdict | What was found | Basis |
|---|---|---|---|---|
| 11 | Patch shape: "ball, diamond, or box, or one continuous knob sliding between them" | Built as the p-norm, and the knob matters between 1 and 2 | Six values of p, with exact counts. On real matrices the best is p = 5/4 or 3/2 on 64 to 75 of 85, by 0.2 to 0.3 dB over the better of ball and cross-polytope. On the language model alone by 0.08 to 0.16 dB. Part of it is that those codebooks fill their cap and a cross-polytope's does not | measured |
| 12 | Patch size: "set per group from its biggest weight" | From its biggest block in the region's norm. The knob is the group | A small group pays for its scale, 16 bits over g weights, and a large one for its largest block. At 4.5 bits groups of 128 are best on 82 of 85 real matrices, and for Gaussian weights by the integrals of the limits table | measured, and derived |
| 13 | Kept directions: "how many of the stretched directions you keep, and which flat ones you drop" | Loses nearly always, as dropping | Keeping the 1/8, 1/4 or 1/2 largest directions in two rounded factors is behind the best full arm by 12 to 20 dB in the mean on real matrices. It is ahead by 2 dB on two, where an eighth of the directions holds over 99 percent. By Shannon's rule the spread over directions would be worth 8.5 dB on whisper and 4.9 on qwen at 4 bits if the rotation were free, by giving each direction its own bits and dropping 6 and 1.5 percent of them. No arm does that. The second record found that balancing columns, the cheap form of it, hurt the output | measured, and derived |
| 14 | "Each knob has a meaning: shape decides the patch part, size the size part" | The two do not separate | For Gaussian weights the step from groups of 8 to a row costs 4.8 dB with a ball, 5.1 with a cross-polytope and 5.8 with a cube (`bounds/bounds.out`, section 10). The size part depends on the shape | derived |

### Stage 4: the formula

| # | Claim | Verdict | What was found | Basis |
|---|---|---|---|---|
| 15 | "The error is a sum of three parts" | In dB, and of other parts | error = G times Phi times N^(-1/4): the cell, the region and the count. The dropped share adds to that in plain units | derived |
| 16 | "The patch-shape part depends on the cloud's spikiness and the shape chosen" | And on where the weights sit | See rows 4 and 7 | measured |
| 17 | "Pick the knobs that make the sum smallest, under two hard limits": bits and decode cost | Bits: done, exactly. Decode cost: cannot be read | Whether a codebook fits a budget is a comparison of whole numbers. No codebook here has a packed format or a timing. The cells with a row cap do not fit a whole number of bits per block | measured, and none |
| 18 | "Run the chosen setup and compare. If they disagree, the map was wrong" | They agree from 4 bits up. Where they did not, it was the formula | The map was exact each time. The causes were in the rule: a block on a wall of its cell, a block at zero, the scalar rule's clip on its far side (3/q of its error on uniform weights, to 0.01 dB), a star's arms, and at 2 bits the cells themselves | measured |
| 19 | "That loop is what makes the formula causal rather than fitted" | Holds for weight error | Nothing is fitted. Over 13 arms the rank correlation of predicted and measured error is 0.99 on real matrices. Whether a change in a knob moves the output as the formula says was not tested | measured |

### Scope

| # | Claim | Verdict | What was found | Basis |
|---|---|---|---|---|
| 20 | "The formula can be exact for the tidy families and bounded for the spiky ones" | The region part is exact for a Gaussian cloud. The error is not. A bound exists for any weights | Gaussian, E8: 0.03 to 0.05 dB over the formula from 4 bits up, 0.77 dB at 2 bits. With the safe scale of the extension the error has a proved bound for any weights, 2.4 dB above the estimate for E8. It held in 1,308 of 1,308 runs | measured, and proved |
| 21 | "The step from a real model's cloud to a family is tested, never proved" | Agreed, and the step can be left out | Row 10 | measured |
| 22 | "The process is deterministic and the core is checkable" | True of what was built | Rows 5 and 8. The programs have no unseeded draw: the synthetic matrices and the shuffled copy come from fixed seeds | checked |
| 23 | "The fit to reality is validated on real matrices and on model output" | Matrices: yes. Output: no | 85 held-out matrices of two models. No model was scored | measured |
| 24 | "Issue #9 is one experiment inside this process, the fold as an annealing schedule for the patch-shape knob" | The fold moves the cell, not the region | In the formula a lattice enters through G alone: its cell's volume cancels. So a path through lattices can gain at most what a cell is worth. In 3 dimensions, where the fold lives, that is 0.35 dB against 0.65 for E8 (`notes/pr6-proposal-check.md`, row 8). The shape knob of this note is the region | derived |

### The comment on published work

The comment under issue #10 lists prior work by knob and then says what is new. A read-only agent opened each source. The lines quoted in rows 25, 26, 27, 33, 34, 36, 37, 39, 40 and 41, and those from HIGGS, were then found again in the texts by the session that wrote this note. Those rows say **checked**. The rest say **reported**.

| # | Statement in the comment | Verdict | What the source says | Basis |
|---|---|---|---|---|
| 25 | QuIP# (ICML 2024): a random Hadamard turn, then blocks of 8 rounded to E8 points | Confirmed, and it is more than the comment says | Its codebook E8P is a ball cut: "227 elements of \|D8^\| with norm <= sqrt 10 plus 29 padding elements", "this ball-shaped 2^16-entry lattice codebook". The paper gives the reason: the transformed weights "follow a roughly ball-shaped Gaussian distribution", and scalar rounding gives a set "shaped like a hypercube rather than a ball" | checked for the lines quoted |
| 26 | QuIP# reaches 2 bits "near-losslessly" | Wrong | The paper does not say so. Its 2-bit claim is "the first time a 2 bit PTQ method has offered similar scaling to higher bitrates". Its table 4 has perplexity 6.19 at 2 bits against 5.12 unrounded on Llama-2 7B, with fine-tuning, which the comment leaves out | checked for the line, the table reported |
| 27 | QTIP (NeurIPS 2024 Spotlight): trellis codes, on the premise that incoherence processing makes the weights about Gaussian | Confirmed | "Incoherence processing makes W approximatelly i.i.d Gaussian, which reduces quantization to Gaussian source coding" | checked for the line quoted |
| 28 | QuIP (2023): the predecessor, with guarantees | Confirmed | Title and venue as stated | reported |
| 29 | QuaRot: random Hadamard turns fused into the weights, end-to-end 4 bits | Confirmed, with a nuance | Fusion is "where possible". Some transforms stay in the forward pass | reported |
| 30 | SpinQuant (ICLR 2025): learned rotations | Confirmed | | reported |
| 31 | SqueezeLLM keeps the giant weights in full precision | Incomplete | The sparse part holds "outliers and sensitive weight values", the second chosen by Fisher information | reported |
| 32 | SVDQuant: a low-rank branch absorbs the outliers, the rest goes to 4 bits | Roughly right | It is for diffusion models, on weights and activations. The outliers are first moved from the activations into the weights | reported |
| 33 | SpQR keeps outlier columns aside | Wrong | "As these outliers are often unstructured, we encode them individually". The method that isolates outlier feature dimensions is LLM.int8() | checked for the line quoted |
| 34 | The OCP MX formats and llama.cpp's K quantisers: a scale per block, and a scale of scales | Confirmed | MX: the scale "is shared across all k elements", with k = 32 in its four formats. llama.cpp: blocks of 16 or 32 inside a super-block of 256, with scales of 4 or 6 bits and one 16-bit scale for them | checked for the MX line, the rest reported |
| 35 | QLoRA's NF4 builds its levels from the quantiles of a round cloud | Loosely | From the quantiles of a normal distribution of one weight. It shapes single levels and not a region | reported |
| 36 | SliceGPT turns the network into its stretched directions and deletes the flat ones, 25 percent of the parameters at 99 percent of performance | Right for two of three models, and not about the weight cloud | 99 percent for Llama-2 70B and OPT 66B, 90 for Phi-2. Its rotation is "the eigenvectors of" the second moments of the **activations**. The stretched directions of issue #10 are those of the weights | checked for the lines quoted |
| 37 | LoRA: updates live in a low-rank subspace, "the same stretched-directions idea" | The paper says the opposite of the weights | "The weight matrices in these layers typically have full-rank." Low rank is a property of the update | checked for the line quoted |
| 38 | Viazovska's 2017 proof for E8 | Right, and beside the point | Posted in 2016, published in 2017. It is a packing theorem. The figure for rounding is the second moment of the cell (`notes/pr6-proposal-check.md`, row 6) | reported |
| 39 | New: "the mechanism split", the shape part against the size part | Published in 1993 | Eyuboglu and Forney: the performance of a lattice quantiser with a bounded codebook "is determined by two separable quantities, the granular gain and the boundary gain, which are determined by the shapes of the granular cells and of the support region". NestQuant restates it for language models | abstract checked, the paper not opened |
| 40 | New: integer levels cut by a ball get most of E8's gain | Plotted in QuIP#, figure 3 | "Each k-bit codebook consists of a d-dimensional base lattice intersected with a ball", with half-integer grids in 1, 4 and 8 dimensions beside D4 and E8, and the remark that more dimensions help the grid "as the resulting codebook is closer in shape to the source distribution". By the agent's reading of the plot the grid in 8 dimensions gets about half of E8P's gain at 2 bits on a Gaussian source | caption checked, the reading of the plot reported |
| 41 | New: E8 cut by a ball beats the E8 Voronoi code | The ball is published as the best region. The head-to-head was not found | NestQuant: "the optimal S under this criterion is rB", a Euclidean ball, and "QuIP# used this approach with Lambda = E8 (same as we do) and S = rB". It keeps the Voronoi region for its decoder and uses a union of scales. LLVQ cuts the Leech lattice by balls. No measurement of ball-cut E8 against a single-scale E8 Voronoi code at equal rate was found. Four searches, so "not found" is weak | checked for the lines quoted |
| 42 | New: "a formula that says which one to use when" | Published in parts | The region: Fischer (1986, 1989) matches it to the source, a pyramid for a Laplace source and an ellipse for a Gaussian. Jeong and Gibson (1993) give "analytical methods for truncating and scaling lattices". The budget: HIGGS picks a setting per layer by a dynamic program over a table of errors, and Radio equalises a marginal return across layers | abstracts reported |
| 43 | New: "a machine-checked measurement and derivation with replayable certificates" | Not found in the work read | No paper in the comment's list ships a replayable receipt of its measurements. This is the part of the proposal with no precedent found | reported |

Not in the comment, and bearing on it:

- **NestQuant** (arXiv 2502.09720), missing from the report of PR #6 as well. It is the nearest published work to this repository's E8 arm.
- **HIGGS** (arXiv 2411.17525), for the link from weight error to output. Its theorem: for small relative errors t_l of the layers, the expected perplexity is that of the model plus a sum of a_l t_l^2, with constants a_l "independent of the compression process". It reports that the prediction "diverges on lower bitwidths", under about 3 bits. The second record of this repository found weight error not to say what the output does when weights are clipped. The two agree if a clip is not a small error.
- **Pyramid vector quantisation for language models** (arXiv 2410.16926): integers cut by a cross-polytope, projected to a sphere, with a separate scale. The arm ZC of the region record is the classical form of it.
- **HyperQuant** (arXiv 2606.23406): a measured split in dB on language-model weights, with 0.49 dB of granular gain for E8 over a plane lattice.

## What PR #5 has that bears on this proposal

The comment that assigned issue #10 asks for this. PR #5 adds `notes/cross-org-grafts.md`: mechanisms from other repositories crossed with this one's concepts. Seven of its crossings meet the proposal. Three were built into the study.

| Crossing in PR #5 | What it is | Where it meets issue #10 | Here |
|---|---|---|---|
| X1, with W7 and O12 | A witness for each claim, re-verified by a checker that can fail | Stage 2: "a receipt that proves the numbers are correct" | Built. `check_receipts.py` replays every receipt from the weights and exits with 1 on a difference. It also changes one digit and one weight and checks that both are caught. PR #5 lists two checkers elsewhere that print a pass whatever happens |
| X2, with O8 | Refuse outputs unless the plan's hash matches and the plan's commit comes before the first output | The loop of stage 4 is a preregistered run | Built. `run.py` refuses the run of the plan unless the plan and four programs are committed as they stand and pushed, and records their hashes |
| X3, with O11 | Generate a table from the data, then compare | Every figure of the record | Built in part. `summary.py`, `safe.py`, `certify.py` and `detail.py` generate the tables, and `predictions.py` and `safe.py` every verdict. The record's prose still quotes them by hand. Another agent then traced it to the outputs and found some twenty slips, none in a result table |
| X5, with W1 | Accept a change only if it stays within a radius of a saved baseline | Stage 4: "compare the measured error against the formula's prediction" | Not built. The formula gives the radius: a rounded matrix whose error is more than, say, 0.3 dB over the formula's is refused. With a mark of 0.3 dB that gate would have passed 96 to 100 percent of the runs of the study from 4 bits up, and 13 to 22 percent at 2 bits (`detail.out`, table 4) |
| X10, with W8 | Rank variants by the largest singular value of the weight change, which ranked damage best in the second record | The formula predicts squared error, the weaker measure | Measured, not built. The record has the ratio of the two for two arms. On real matrices the rounding error is not like noise, and less so as the group grows |
| X13, with O6 | Bits realised over bits nominal | The first hard limit of stage 4 is "total bits" | Reported. The codebook table gives the bits if each block's index is stored alone in whole bits. The cells with a row cap do not fit a whole number of bits per block |
| X7 | A decoder certified on a box of real inputs, not only tested | "The verified core" | Not built. The receipts rest on whole-number arithmetic. The rounding rests on the nearest-point rule of `lattice/`, which is tested against brute force and not certified |

Not relevant here: X4, X6, X8, X9, X11, X12, X14, X15. X11, choosing what to protect by a second-order measure, is where activations would enter. The formula of issue #10 sees weights only.

## A core that can be verified

The owner asked, while the study ran, about formulas for the choice of the knobs that can be formally verified and built into the procedure. What follows is what could be done without a proof assistant, and what one would add.

| | What | How sure | Where |
|---|---|---|---|
| Exact | The receipt. The count of a codebook. For five arms per group size, the formula at level 2 as a fraction times a power of pi. Which of two such arms is ahead. Whether a codebook fits the budget | Whole-number arithmetic, a certificate, and a checker that shares no code and fails on a changed pick. 109 of 109 confirmed | `shape/certified.py`, `shape/check_certificate.py` |
| Proved, on paper | The ball is the best region for a round cloud. Three inequalities for any 8 numbers, and the limits on a region's worth that follow. At the safe scale no block is outside and the error is under a bound | Proofs in the two plans. The identities behind the inequalities are checked at every point of a grid, which proves them. The bounds on pi are checked from Machin's formula in exact fractions | `PREREG.md`, `PREREG-extension-1.md`, `bounds/bounds.out` section 10 |
| Measured | That the formula is close to the error. The shapes with p = 2/3, 5/4, 3/2 and 3. The cell at zero. Anything at 2 bits | The study | the record |

Four rules came out of it for a formula that is meant to be verified.

1. **Every input a whole number computed from the weights.** A float is a whole number times a power of two, and the power cancels in every ratio the formula uses.
2. **The objective a product of fractions and known constants.** Then "this arm is ahead of that one" clears to whole numbers. That holds for the cube, the ball and the cross-polytope, and not for a p-norm with p = 3/2.
3. **No search inside the rule.** The searched scale of the records has no closed form, so nothing can be proved about what it finds. The safe scale is a closed form with a two-line proof that nothing is clipped. It costs 0.40 to 0.46 dB for E8 in a ball at 4.5 bits, and the cost follows from the codebook alone.
4. **A comparison the bounds cannot settle is reported as undecided.** None arose in 109 picks.

What a proof assistant would add is in the extension's plan, in rising order of work: the inequalities, the soundness of the comparison, the two claims of the safe scale, the inequality behind the round-cloud limit (Hölder's with exponents 5 and 5/4), the covering radius of E8, the count of a codebook, and the nearest-point rule of E8. "The formula gives the error" is not a theorem as it stands. It is a statement about fine cells, and the study measures how close it is.

## What follows

In the order this note would take them.

1. **Score the formula's picks on a model.** Issue #3 is the place: its second model and second text, with three more arms at equal bits. E8 in a p-norm ball with p = 3/2, E8 in a ball at the safe scale, and groups of 128 beside groups of 32. The last is the test of the size knob, where weight error and the second record's output scores pull apart. The spectral ratio of this study says which way it might go and proves nothing.
2. **The formula at 2 bits.** It is low by 0.8 dB there. The cause is in the cells and the few shells of a small codebook, not in the map. A formula for that regime needs the codebook's own shells, which `shapelib` already counts.
3. **A way to number the points of a cut codebook.** Still missing, as the note on PR #6 says. Without it the decode budget of stage 4 cannot be read. The count by pairs of halves in `shapelib` is the first half of one: the number of a point is the count of the points before it.
4. **A proof assistant for the ten statements.** This session could not install one. The first seven are short.
5. **Bits per direction, not dropped directions.** Shannon's rule puts 5 to 8 dB on the table at 4 bits if the rotation were free, and it is not free. The second record found the axis-aligned form of it hurt the output. So this needs an output score before any weight-error run.
6. **A gate on the formula.** Refuse a rounding whose error is more than 0.3 dB over the formula's (crossing X5 of PR #5). It is a few lines on top of `shapelib.predict`.

Not recommended as it stands: fitting a cloud to a family before reading the formula off it.

## Sources

- Issue #10 and its two comments, as they stood on 2026-11-17. **Checked.**
- The records of this repository, and `bounds/bounds.out`. **Checked.**
- PR #5, `notes/cross-org-grafts.md` at commit `5b975e6`, read on the branch of that pull request. It is not in this checkout. **Checked.**
- QuIP#: arXiv 2402.04396. QTIP: arXiv 2406.11235. SpQR: arXiv 2306.03078. SliceGPT: arXiv 2401.15024. LoRA: arXiv 2106.09685. NestQuant: arXiv 2502.09720. HIGGS: arXiv 2411.17525. LLVQ: arXiv 2603.11021. HyperQuant: arXiv 2606.23406. Pyramid vector quantisation for language models: arXiv 2410.16926. Radio: arXiv 2505.03031. The OCP MX specification, version 1.0. **Checked** for the lines quoted, in the texts the agent downloaded.
- QuIP: arXiv 2307.13304. QuaRot: arXiv 2404.00456. SpinQuant: arXiv 2405.16406. SqueezeLLM: arXiv 2306.07629. SVDQuant: arXiv 2411.05007. LLM.int8(): arXiv 2208.07339. QLoRA: arXiv 2305.14314. AQLM: arXiv 2401.06118. VPTQ: arXiv 2409.17066. Viazovska: arXiv 1603.04246. llama.cpp's source header and pull request 1684. **Reported.**
- Eyuboglu and Forney, IEEE Transactions on Information Theory 39(1), 1993. Forney and Wei, IEEE Journal on Selected Areas in Communications 7(6), 1989. Fischer, 1986 and 1989. Jeong and Gibson, 1993. Hamkins and Zeger, 2002. Abstracts only, from an index. The first was **checked** there, the rest **reported**. None of the papers was opened.
- The covering radius of E8: Conway and Sloane, Sphere Packings, Lattices and Groups. Not opened. A test samples 200,000 points and finds none further than 1 from the lattice.

Not reached: Proven-Commons and the Foundation-0 records.

## Disclosure

Written by an AI agent session under the owner's direction. The check of published work was delegated to a second agent, and its findings are marked above. A third agent traced the figures of the record and of this note to the outputs, and its findings were applied. Nothing here has been reviewed by a person.
