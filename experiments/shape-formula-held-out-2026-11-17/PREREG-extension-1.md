# Extension 1: a rule with a proved guarantee, and the pick in exact arithmetic

Written 2026-11-17. The main run of [PREREG.md](PREREG.md) had finished and `results.json` existed when this was written. It had not been opened: no line of it, of `results.out` or of any table from it had been read. It is committed, unread, in the commit before this one. `safe.py` refuses to start unless this file, `safe.py`, `shape/certified.py` and `shape/shapelib.py` are committed as they stand and pushed.

`PREREG.md` is unchanged. Its SHA-256 is in `results.start`.

## Why

While the main run was in progress the owner asked about formulas for the choice of the knobs that can be formally verified and built into the procedure. Issue #10 asks for the same thing in its second stage: a core "a machine can check step by step".

No proof assistant ran in this session: installing one was not permitted. So this extension does the two things that need none, and lists what a proof assistant would add.

## What is exact, what is proved, what is measured

| | What | How sure |
|---|---|---|
| Exact | The whole numbers of a receipt. The count of a codebook. For five arms per group size, the formula at level 2 as a fraction times a power of pi. Which of two such arms has the smaller predicted error. Whether a codebook fits the bit budget | Whole-number arithmetic. `shape/certified.py` picks, and `shape/check_certificate.py`, which shares no code with it, confirms each pick and fails on a changed one. Two counters that share no code agree on all 28 codebooks |
| Proved | Three inequalities for any 8 numbers, and from them fixed limits on what one region can be worth against another for any weights (`bounds/bounds.out`, section 10). The ball is the best region for a round cloud (`PREREG.md`). The two claims below | On paper. The identities behind the inequalities are checked at every point of a grid, which proves them for polynomials of their degree. The bounds on pi are checked from Machin's formula in exact fractions |
| Measured | That the formula is close to the error. Everything about the shapes p = 2/3, 5/4, 3/2 and 3, about level 3, and about 2 bits | The main run |

The five arms with an exact formula: the scalar rule, and the integers and E8 cut by a ball or by a cross-polytope.

## The safe scale

The rule of the records searches for the smallest scale at which a block rounds into the codebook. What it finds has no closed form, and a block can still fall outside at its group's scale and be shrunk. A formula with a search inside it is hard to prove anything about.

Let the codebook be every point of a lattice with K-norm at most r, with K a ball or a cross-polytope. Let rho be the covering radius of the lattice, and m the largest K-norm of a vector of length rho: m = rho for the ball and sqrt(8) rho for the cross-polytope. For a group take

    s  =  (largest K-norm of a block of the group) / (r - m).

**Claim 1.** Every block of the group rounds to a point of the codebook. Proof: the nearest lattice point y of x / s is within rho of it. So the K-norm of y is at most that of x / s plus m, which is at most (r - m) + m = r.

**Claim 2.** The squared error of every block is at most rho^2 s^2. Proof: x - s y = s (x / s - y), and x / s - y is no longer than rho.

So for any weights whatever the error of a matrix, as a share of its squared weights, is at most

    bound  =  rho^2 * (blocks per group) * (sum over groups of s^2) / (sum of squared weights),

and with fine cells it is close to the estimate, G s^2 per weight. The two differ by the fixed factor rho^2 / (8 G): 2.42 dB for E8 and 4.77 dB for the integers.

rho is sqrt 2 for the integers in 8 dimensions. For E8 it is 1, taken from Conway and Sloane and not proved here. The claims are for exact arithmetic. The program multiplies the scale by 1 + 10^-9 to cover the rounding of 64-bit floats, and counts the blocks outside instead of assuming there are none.

Against the searched rule the safe scale is larger by r_eff / (r - m), where r_eff is the radius of the region of the same volume as the codebook's cells. And it has no block on a wall, which the searched rule pays 10 log10(1 + 2/g) for. So

    predicted cost  =  20 log10(r_eff / (r - m))  -  10 log10(1 + 2 / g)   dB.

From the codebooks, before any run: 0.4 to 0.5 dB for E8 in a ball, about 0.7 for the integers in a ball and for E8 in a cross-polytope, 1.0 to 1.2 for the integers in a cross-polytope.

## The run

`safe.py`: the 109 held-out matrices of the main plan, four codebooks (the integers and E8, each cut by a ball and by a cross-polytope), the three cells of the budget, rounded at the safe scale. 1,308 runs. The searched rule and the scalar rule are not run again: their errors are read from `results.json`.

`certify.py`: for each matrix, the pick among the five exact arms at three group sizes at level 2, in exact arithmetic, from the receipts in `results.json`. It rounds nothing. It was written and tried on the pilot's receipts before this plan: 38 certificates, all confirmed, the pick the same as the 64-bit pick at level 2 on all 38.

**Seen before writing this.** The pilot, and `certify.py` on the pilot. One scratch run of the safe rule on a Student t matrix of 64 by 2048 from seed 5, which is not in the study, with groups of 128: no block outside, the worst block at 0.76 to 0.87 of its bound, measured cost 0.44 to 1.00 dB against 0.46 to 1.04 predicted, the estimate within 0.04 dB. Nothing of the main run.

## Predictions

`safe.py` computes each verdict.

- **E1. The two claims.** In all 1,308 runs no block lies outside the codebook, no block has more than its bound, and no matrix's measured error, taken after storage as 16-bit floats, is over its bound.
- **E2. What the guarantee costs.** On the 85 real matrices, for each of the 12 pairs of codebook and group, the mean error of the safe rule over that of the searched rule is within 0.15 dB of the predicted cost.
- **E3. The estimate.** On the real matrices the measured error over the estimate, with the cell at zero, has a mean size of at most 0.10 dB and is within 0.3 dB in at least 90 percent of the runs. No wall and no search stand between the formula and the rule here.
- **E4. The pick.** On every matrix the arm with the smallest bound among the 12 is the arm with the smallest estimate, and it is an E8 arm. This follows from the fixed factor between the two and is predicted as a check of the code. On the real matrices that arm's measured error is over the smallest measured among the 12 by at most 0.05 dB in the mean.
- **E5. Against the scalar rule.** At 4.5 bits the best safe arm of a real matrix is ahead of its best scalar rule on at least 90 percent of the 85, by 1.0 to 3.5 dB in the mean.

For `certify.py`, reported and not predicted: on how many held-out matrices the exact pick is the 64-bit pick at level 2 and at level 3, whether every certificate is confirmed, and what the pick costs against the measured best.

If E1 fails, a claim or the code is wrong, and nothing else here stands. If E2 and E3 hold, the procedure of issue #10 has a form in which the scale is a closed form, the guarantee is a theorem, and the price is known in advance: about half a decibel for E8 in a ball.

## For a proof assistant

In rising order of work. None is done here.

1. For 8 whole numbers: (largest size)^2 <= sum of squares <= (sum of sizes)^2 <= 8 * sum of squares, and sum of squares <= 8 * (largest size)^2.
2. The largest over a group and the sum over groups keep those. So the receipt's three numbers are ordered: top_cube <= top_ball <= top_cross <= 8 top_ball, and top_ball <= 8 top_cube.
3. Multiplying every weight by c multiplies each of the receipt's numbers by c^2. So the power of two cancels and the region part does not depend on the unit.
4. The comparison of `check_certificate.py` is sound: if it says one arm is ahead, the real number of the formula is smaller, given the bounds on pi.
5. The pick is a minimiser over the candidates within the budget, and "within the budget" is codewords^d <= 2^n.
6. Claims 1 and 2 above, from the triangle inequality and the covering radius.
7. For positive numbers rho_i and weights w_i that add up to 1: (sum of w_i rho_i^8)^(1/4) * (sum of w_i rho_i^-2) >= 1. It is Hölder's inequality with exponents 5 and 5/4. With it, the ball is the best region for a round cloud.
8. The covering radius of E8 is 1.
9. The count of a codebook by pairs of halves is the number of lattice points in the region.
10. The nearest-point rule of `lattice/` returns a nearest point of E8 for every real input.

Not a theorem as it stands: "the formula gives the error". That is a statement about fine cells, and the main run measures how close it is.

## What this cannot show

- A proof checked by a machine. These are proofs on paper and checks in exact arithmetic.
- Anything about a model's output.
- That the safe rule is the best rule with a guarantee. The margin m is a worst case over directions, and a tighter one may exist for the cross-polytope.
