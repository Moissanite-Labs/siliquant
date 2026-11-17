"""The formula of issue #10 for an arm with one ball per group, in closed form, and what it says of the output.

With one ball over a whole group of n weights the region part of the formula does not depend
on the weights. The largest norm of a block in the group is the norm of the group itself, so

    Phi  =  volume(ball in n dimensions)^(2/n) * n

for any weights whatever. The error of a group, as a share of its squared weights, is then

    error  =  G * Phi * 2^(-2 b / n) * f

    G   the cell: 1/12 for integer levels, 929/12960 for E8
    b   the bits of the point of a group (the codebook has about 2^b points)
    f   what the scale adds:
        1 + 2/n            when the scale is the smallest at which the group rounds into the
                           codebook. The group then lies on a wall of its cell, where the mean
                           squared error is (n + 2)/n of the cell's (level 2 of shapelib).
        (2^(2/m) - 1) / (2 ln 2 / m)
                           when the scale is rounded up to a grid of m steps per factor of two.
                           The group is then inside its cell, and the squared scale is larger by
                           the mean of 2^(2u/m) for u between 0 and 1.

Nothing is fitted. No weight is read. The same share holds for every group of every matrix.

predicted_kl() adds one measured number: the mean KL of one arm of the same model, with its
mean weight error. It says the KL of another arm is that KL times the ratio of the two weight
errors. That is a claim about a model's output and not a formula: the record
model-output-qwen25-2026-11-17 measured it for nine arms, and notes/issue10-output-predictions.md
says where it is expected to hold and where not.

Needs only Python 3. Tests: python3 -m unittest shape/test_ballformula.py
"""
from __future__ import annotations

import math
import re

G_CELL = {"Z": 1.0 / 12.0, "E": 929.0 / 12960.0}
SCALE_STEPS = {16: None, 8: 16, 6: 8}       # bits of a scale -> steps of its grid per factor of two, as in the Qwen run


def db(x):
    return 10.0 * math.log10(x)


def log_ball_volume(n):
    """Natural log of the volume of the unit ball in n dimensions."""
    return 0.5 * n * math.log(math.pi) - math.lgamma(0.5 * n + 1.0)


def phi_ball(n):
    """The region part of one ball per group of n weights: n * volume^(2/n). It tends to 2 pi e from below."""
    return n * math.exp(2.0 / n * log_ball_volume(n))


def scale_factor(n, steps):
    """What the scale adds to the error: the wall for an exact scale, the mean overshoot for a grid."""
    if steps is None:
        return 1.0 + 2.0 / n
    return (2.0 ** (2.0 / steps) - 1.0) / (2.0 * math.log(2.0) / steps)


def ball_error(lattice, n, point_bits, steps=None):
    """Error over squared weights of one ball per group of n weights. lattice is "Z" or "E"."""
    return G_CELL[lattice] * phi_ball(n) * 2.0 ** (-2.0 * point_bits / n) * scale_factor(n, steps)


_GROUP = re.compile(r"^([EZ])BG(\d+):b(\d+)s(\d+)$")          # EBG128:b560s16
_LEVELS = re.compile(r"^([EZ])B(\d+)@(\d+):(\d+)$")           # EB32@32:16


def parse(variant):
    """(lattice, n, bits of the point, steps of the scale's grid, bits per weight) for a variant of the Qwen run, or None.

    None for an arm whose error depends on the weights: a scale shared by more than one ball, a
    cross-polytope, the scalar rule, a Voronoi code, a file of llama.cpp.
    """
    m = _GROUP.match(variant)
    if m:
        lattice, n, b, s = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
        return lattice, n, b, SCALE_STEPS[s], (b + s) / n
    m = _LEVELS.match(variant)
    if m and m.group(2) == m.group(3):
        lattice, n, q = m.group(1), int(m.group(2)), int(m.group(4))
        b = n * math.log2(q)
        return lattice, n, b, None, (b + 16) / n
    return None


def predicted_error_db(variant):
    """The formula's weight error of a variant in dB, or None if it depends on the weights."""
    p = parse(variant)
    if p is None:
        return None
    lattice, n, b, steps, _ = p
    return db(ball_error(lattice, n, b, steps))


def predicted_kl(error_db, anchor_kl, anchor_error_db):
    """The mean KL of an arm from its weight error and one scored arm of the same model: the same KL per unit of weight error."""
    return anchor_kl * 10.0 ** ((error_db - anchor_error_db) / 10.0)


def table(total_bits, lattice="E", groups=(8, 16, 32, 64, 128, 256, 512, 1024, 2048), scale_bits=(16, 8, 6)):
    """Rows (n, bits of the scale, bits of the point, error in dB) at a total of bits per weight: what the formula says of each split."""
    rows = []
    for n in groups:
        for s in scale_bits:
            b = total_bits * n - s
            if b <= 0:
                continue
            rows.append((n, s, b, db(ball_error(lattice, n, b, SCALE_STEPS[s]))))
    return rows


def floors(total_bits):
    """Three floors at a total of bits per weight, in dB: E8's cell with a ball over ever more weights, the ball's cell
    with the same, and Shannon's floor for Gaussian weights. The scale's bits are not charged."""
    r = 2.0 ** (-2.0 * total_bits)
    return {"E8 cell, ball over n -> infinity": db(G_CELL["E"] * 2.0 * math.pi * math.e * r),
            "integer cell, ball over n -> infinity": db(G_CELL["Z"] * 2.0 * math.pi * math.e * r),
            "Shannon, Gaussian weights": db(r)}
