"""The choice of the knobs in exact arithmetic, with a certificate.

shape/shapelib.py computes the formula of issue #10 in 64-bit floats. For five arms per group
size the same formula is a rational number times a power of pi, and "this arm has the smaller
predicted error than that one" is a comparison of whole numbers. This file computes the pick
that way and writes down what a checker needs to confirm it. shape/check_certificate.py is that
checker. It shares no code with this file.

The arms, for a group of g weights:

    X        the scalar largest-entry rule at q levels          cell 1/12        region a cube
    Z8:p1    the integers cut by a cross-polytope               cell 1/12        region a cross-polytope
    Z8:p2    the integers cut by a ball                         cell 1/12        region a ball
    E8:p1    E8 cut by a cross-polytope                         cell 929/12960   region a cross-polytope
    E8:p2    E8 cut by a ball                                   cell 929/12960   region a ball

The formula at level 2 (cell, region, count, and the block on the wall):

    E  =  G * vol^(1/4) * M * N^(-1/4) * f

    M    =  (top / groups) / (sum_sq / count)      from the receipt: whole numbers
    vol  =  256 for the cube, 256 / 8! for the cross-polytope, pi^4 / 24 for the ball
    N    =  the number of codewords: a whole number, counted exactly
    f    =  (g + 2) / g for a cut codebook, (g - 1) / g for the scalar rule

So E^4 = r * pi^(4b), with r a fraction and b = 1 for a ball and 0 otherwise. Two arms are
compared through E^4. If b is the same on both sides, the comparison is one of fractions. If
not, it needs pi^4 on one side, and pi is known to lie between the two fractions PI_LOW and
PI_HIGH. A comparison that those bounds cannot settle is reported as undecided. It is never
guessed.

What is exact here and what is not:
  exact      the whole numbers of the receipt, the counts, the comparison, the bit budget
  not here   level 3 of the formula (the cell at zero), and the shapes p = 2/3, 5/4, 3/2 and 3,
             whose norms are not rational. For those shapelib's floats stand.
  not proved by a machine: this is exact arithmetic with a checker, not a proof in a proof
             assistant. notes/issue10-proposal-check.md lists the statements a proof would cover.
"""
from __future__ import annotations

from fractions import Fraction

# 3.14159265358979323846 < pi < 3.14159265358979323847
PI_LOW = Fraction(314159265358979323846, 10 ** 20)
PI_HIGH = Fraction(314159265358979323847, 10 ** 20)

G_CUBE = Fraction(1, 12)
G_E8 = Fraction(929, 12960)
ARMS = {"X": (G_CUBE, "cube"), "Z8:p1": (G_CUBE, "cross"), "Z8:p2": (G_CUBE, "ball"), "E8:p1": (G_E8, "cross"), "E8:p2": (G_E8, "ball")}
VOLUME = {"cube": (Fraction(256), 0), "cross": (Fraction(256, 40320), 0), "ball": (Fraction(1, 24), 1)}     # (fraction, power of pi^4)


def fourth_power(arm, size, codewords, terms):
    """E^4 of an arm as (r, b): E^4 = r * pi^(4b). terms are the whole numbers of a receipt for this group size."""
    G, region = ARMS[arm]
    vol, b = VOLUME[region]
    top = int(terms["top_" + region])
    M = Fraction(top, int(terms["groups"])) / Fraction(int(terms["sum_sq"]), int(terms["count"]))
    f = Fraction(size - 1, size) if arm == "X" else Fraction(size + 2, size)
    return G ** 4 * vol * M ** 4 * f ** 4 / int(codewords), b


def compare(a, b):
    """-1 if the first has the smaller E^4, 1 if the second has, 0 if they are equal, None if the bounds on pi cannot say.

    a and b are (r, power) pairs from fourth_power.
    """
    (ra, pa), (rb, pb) = a, b
    if pa == pb:
        return (ra > rb) - (ra < rb)
    if pa < pb:
        c = compare(b, a)
        return None if c is None else -c
    # ra * pi^4 against rb
    if ra * PI_HIGH ** 4 < rb:
        return -1
    if ra * PI_LOW ** 4 > rb:
        return 1
    return None


def within_budget(codewords, size, budget):
    """Whether log2(codewords) / 8 + 16 / size <= budget, exactly. budget is a fraction of bits per weight."""
    e = 8 * (Fraction(budget) - Fraction(16, size))            # codewords <= 2^e
    if e < 0:
        return False
    return int(codewords) ** e.denominator <= 1 << e.numerator


def pick(candidates, budget=None):
    """The candidate with the smallest predicted error, in exact arithmetic.

    candidates: a list of dicts with arm, size, codewords and terms. Those over the budget are
    left out. Returns (index of the pick, the certificate). The certificate lists the inputs
    and the claim, and holds nothing the checker has to trust: it recomputes every number.
    Raises ArithmeticError if a comparison is undecided or if two candidates tie for the lead.
    """
    live = [i for i, c in enumerate(candidates) if budget is None or within_budget(c["codewords"], c["size"], budget)]
    if not live:
        raise ValueError("no candidate is within the budget")
    values = {i: fourth_power(candidates[i]["arm"], candidates[i]["size"], candidates[i]["codewords"], candidates[i]["terms"]) for i in live}
    best = live[0]
    for i in live[1:]:
        c = compare(values[i], values[best])
        if c is None:
            raise ArithmeticError("the bounds on pi do not settle a comparison")
        if c < 0:
            best = i
    for i in live:
        if i != best and compare(values[best], values[i]) != -1:
            raise ArithmeticError("two candidates tie, or a comparison is undecided")
    certificate = {"budget": None if budget is None else [Fraction(budget).numerator, Fraction(budget).denominator],
                   "candidates": [{"arm": c["arm"], "size": int(c["size"]), "codewords": str(int(c["codewords"])),
                                   "terms": {k: str(int(c["terms"][k])) for k in ("count", "groups", "sum_sq", "top_cube", "top_ball", "top_cross")}}
                                  for c in candidates],
                   "pick": best}
    return best, certificate


# ---------------------------------------------------------------------------
# A rule with a guarantee: the safe scale
# ---------------------------------------------------------------------------
#
# The rule of the records searches, block by block, for the smallest scale at which the block
# rounds to a point of the codebook. What it finds has no closed form, and a block can still
# fall outside at its group's scale and be shrunk.
#
# The safe scale is a closed form with a proof. Let the codebook be every point y of a lattice
# with K-norm at most r, for a norm K (the ball, p = 2, or the cross-polytope, p = 1). Let rho
# be the covering radius of the lattice: no point of space is further than rho from the
# lattice. Let m be the largest K-norm of a vector of length rho: m = rho for the ball and
# sqrt(8) rho for the cross-polytope, since a sum of 8 sizes is at most sqrt(8) times the
# length. For a group whose blocks are x_1, ..., x_J take
#
#     s  =  (largest K-norm of a block) / (r - m).
#
# Claim 1. Every block of the group rounds to a point of the codebook. Proof: the nearest
# lattice point y of x / s is within rho of it, so its K-norm is at most that of x / s plus m,
# which is at most (r - m) + m = r.
#
# Claim 2. The squared error of every block is at most rho^2 s^2. Proof: x - s y = s (x / s - y),
# and x / s - y has length at most rho.
#
# So the error of a matrix, as a share of its squared weights, is at most
#
#     bound  =  rho^2 * (blocks per group) * (sum over groups of s^2) / (sum of squared weights),
#
# for any weights whatever. With fine cells the error itself is close to G s^2 per weight: the
# estimate. The two differ by the fixed factor rho^2 / (8 G): 2.42 dB for E8 and 4.77 dB for
# the integers. So within one lattice the arm with the smallest bound is the arm with the
# smallest estimate.
#
# The covering radii: sqrt(2) for the integers in 8 dimensions (every entry is within 1/2 of a
# whole number), and 1 for E8 (Conway and Sloane, Sphere Packings, Lattices and Groups, chapter
# 4, section 8.1; not proved here).

import math as _math

import numpy as _np

COVERING_RADIUS = {"Z8": _math.sqrt(2.0), "E8": 1.0}
SAFETY = 1.0 + 1e-9          # the claims are for exact arithmetic; this covers the rounding of 64-bit floats


def norm_margin(lattice, p):
    """m: the largest K-norm of a vector no longer than the covering radius. p is 1 or 2."""
    if p not in (1, 2):
        raise ValueError("the safe scale is defined for the ball and the cross-polytope")
    return COVERING_RADIUS[lattice] * (1.0 if p == 2 else _math.sqrt(8.0))


def safe_radius(code):
    """r - m for a codebook of shapelib: the K-norm a scaled block may have and still be sure to round inside."""
    return code.shape.radius_as_norm(code.R) - norm_margin(code.lattice, code.shape.p)


def safe_scales(W, code, group):
    """The safe scale of every group: its largest K-norm over (r - m)."""
    W = _np.asarray(W, dtype=_np.float64)
    size = W.shape[1] if group == "R" else int(group)
    top = code.shape.norm(W.reshape(-1, 8)).reshape(-1, size // 8).max(axis=1)
    return _np.maximum(top / safe_radius(code) * SAFETY, 1e-150)      # a group of zeros gets a scale that rounds it to zero


def safe_round(W, code, group):
    """Round a matrix at the safe scale. Returns a dict: the measured error, the bound, the estimate, and the share of blocks outside.

    The share outside is 0 by claim 1. It is counted, not assumed. The rounded matrix is stored
    as 16-bit floats before its error is taken, as everywhere in the records.
    """
    W = _np.asarray(W, dtype=_np.float64)
    m_rows, n = W.shape
    size = n if group == "R" else int(group)
    per = size // 8
    s = safe_scales(W, code, group)
    t = _np.repeat(s, per)
    X = W.reshape(-1, 8)
    Y = code.closest(X / t[:, None])
    outside = float(_np.mean(~code.inside(Y)))
    exact = Y * t[:, None]
    total = float(_np.sum(W ** 2))
    Wh = exact.reshape(m_rows, n).astype(_np.float16).astype(_np.float64)
    rho = COVERING_RADIUS[code.lattice]
    block_err = ((X - exact) ** 2).sum(axis=1)
    b2 = (X ** 2).sum(axis=1)
    if code.lattice == "Z8":
        small = _np.abs(X) < 0.5 * t[:, None]
        with_zero = _np.where(small, X ** 2, (code.G * t ** 2)[:, None]).sum()
    else:
        small = b2 < (_math.sqrt(0.5) * t) ** 2
        with_zero = _np.where(small, b2, 8 * code.G * t ** 2).sum()
    return {"meas": float(_np.sum((W - Wh) ** 2)) / total,
            "meas_before_storage": float(block_err.sum()) / total,
            "bound": rho ** 2 * per * float(_np.sum(s ** 2)) / total,
            "estimate": code.G * size * float(_np.sum(s ** 2)) / total,
            "estimate_with_cell_at_zero": float(with_zero) / total,
            "outside": outside,
            "worst_block_over_its_bound": float(_np.max(block_err / (rho * t) ** 2))}        # t is never zero
