"""Checks a certificate of shape/certified.py. Whole-number arithmetic only. Imports nothing from this repository.

A certificate says: among these candidates, within this budget, this one has the smallest
predicted error. check() recomputes the claim from the candidates' whole numbers in its own
way and returns the list of what does not hold. An empty list means the certificate holds.

The quantity compared is the fourth power of the predicted error with the parts common to
every candidate taken out. For a candidate with cell constant gn / gd, region volume
vn / vd * pi^(4b), codewords N, group size g, wall factor fn / fd, and receipt numbers
top, groups, sum_sq, count:

    E^4 * (sum_sq / count)^4  =  (gn^4 * vn * top^4 * fn^4) / (gd^4 * vd * groups^4 * fd^4 * N)  *  pi^(4b)

Both sides of every comparison are multiplied out, so only whole numbers are compared.
pi is taken to lie between 314159265358979323846 / 10^20 and 314159265358979323847 / 10^20.
pi_bounds_hold() confirms that from Machin's formula in exact fractions.

It also checks, each time it is loaded, the identities behind three inequalities that hold
for any 8 numbers, and so for any block of weights (identities()).

This is a checker and not a proof assistant. What it gives is a second computation of the
pick, by other code, from the numbers of the receipt.

Usage:
  python3 shape/check_certificate.py certificates.json
"""
import itertools
import json
import sys

PI_LOW_N, PI_HIGH_N, PI_D = 314159265358979323846, 314159265358979323847, 10 ** 20
CELL = {"X": (1, 12), "Z8:p1": (1, 12), "Z8:p2": (1, 12), "E8:p1": (929, 12960), "E8:p2": (929, 12960)}
REGION = {"X": "cube", "Z8:p1": "cross", "Z8:p2": "ball", "E8:p1": "cross", "E8:p2": "ball"}
VOLUME = {"cube": (256, 1, 0), "cross": (256, 40320, 0), "ball": (1, 24, 1)}     # numerator, denominator, power of pi^4


def value(c):
    """(numerator, denominator, power of pi^4) of the compared quantity for one candidate."""
    gn, gd = CELL[c["arm"]]
    vn, vd, b = VOLUME[REGION[c["arm"]]]
    g = int(c["size"])
    fn, fd = (g - 1, g) if c["arm"] == "X" else (g + 2, g)
    t = c["terms"]
    top, groups, N = int(t["top_" + REGION[c["arm"]]]), int(t["groups"]), int(c["codewords"])
    return gn ** 4 * vn * top ** 4 * fn ** 4, gd ** 4 * vd * groups ** 4 * fd ** 4 * N, b


def strictly_less(x, y):
    """True only if x < y is certain. x and y are (numerator, denominator, power of pi^4), denominators positive."""
    (xn, xd, xb), (yn, yd, yb) = x, y
    if xb == yb:
        return xn * yd < yn * xd
    if xb == 1:                                   # xn/xd * pi^4 < yn/yd, safe with the upper bound on pi
        return xn * PI_HIGH_N ** 4 * yd < yn * xd * PI_D ** 4
    return xn * yd * PI_D ** 4 < yn * PI_LOW_N ** 4 * xd      # xn/xd < yn/yd * pi^4, safe with the lower bound


def in_budget(c, budget):
    """log2(codewords) / 8 + 16 / size <= bn / bd, as codewords^den <= 2^num with whole numbers."""
    bn, bd = budget
    g = int(c["size"])
    num, den = 8 * (bn * g - 16 * bd), bd * g      # the exponent 8 (bn/bd - 16/g) as a fraction, not reduced
    if num < 0:
        return False
    return int(c["codewords"]) ** den <= 2 ** num


def check(cert):
    bad = []
    cands = cert["candidates"]
    for i, c in enumerate(cands):
        t = c["terms"]
        size = int(c["size"])
        if c["arm"] not in CELL:
            bad.append(f"candidate {i}: unknown arm")
            continue
        if int(t["groups"]) * size != int(t["count"]) or size % 8 or int(t["sum_sq"]) <= 0 or int(c["codewords"]) <= 0:
            bad.append(f"candidate {i}: the receipt's numbers do not fit together")
        # the three inequalities that hold for any weights
        if not (int(t["top_cube"]) <= int(t["top_ball"]) <= int(t["top_cross"]) <= 8 * int(t["top_ball"]) and int(t["top_ball"]) <= 8 * int(t["top_cube"])):
            bad.append(f"candidate {i}: the receipt breaks an inequality that holds for any weights")
    if len({(c["terms"]["sum_sq"], c["terms"]["count"]) for c in cands}) != 1:
        bad.append("the candidates are not of one matrix: the squared weights or the count differ")
    if bad:
        return bad
    live = [i for i, c in enumerate(cands) if cert["budget"] is None or in_budget(c, cert["budget"])]
    p = cert["pick"]
    if p not in live:
        return ["the pick is not within the budget"]
    vp = value(cands[p])
    for i in live:
        if i != p and not strictly_less(vp, value(cands[i])):
            bad.append(f"the pick is not certainly ahead of candidate {i} ({cands[i]['arm']}, groups of {cands[i]['size']})")
    return bad


def pi_bounds_hold():
    """Whether 314159265358979323846 / 10^20 < pi < 314159265358979323847 / 10^20, from Machin's formula in exact fractions.

    pi = 16 atan(1/5) - 4 atan(1/239), and atan(1/x) = 1/x - 1/(3 x^3) + 1/(5 x^5) - ...
    The terms shrink and alternate in sign, so a sum that stops after a plus term is above
    atan(1/x) and one that stops after a minus term is below it.
    """
    from fractions import Fraction

    def atan_bounds(x, terms):
        total, sums = Fraction(0), []
        for k in range(terms):
            total += Fraction((-1) ** k, (2 * k + 1) * x ** (2 * k + 1))
            sums.append(total)
        return (sums[-1], sums[-2]) if terms % 2 == 0 else (sums[-2], sums[-1])        # (below, above)

    lo5, hi5 = atan_bounds(5, 40)
    lo239, hi239 = atan_bounds(239, 12)
    low, high = 16 * lo5 - 4 * hi239, 16 * hi5 - 4 * lo239
    return Fraction(PI_LOW_N, PI_D) < low and high < Fraction(PI_HIGH_N, PI_D)


def identities():
    """The identities behind three inequalities for any 8 numbers a_1..a_8. Returns True if all hold.

    1. 8 * sum a_i^2 - (sum a_i)^2  =  sum over pairs i < j of (a_i - a_j)^2.
       The right side is a sum of squares, so (sum of sizes)^2 <= 8 * (squared length).
    2. (sum a_i)^2 - sum a_i^2  =  2 * sum over pairs i < j of a_i a_j.
       With sizes for the a_i the right side is not negative, so squared length <= (sum of sizes)^2.
    3. 8 * m^2 - sum a_i^2  =  sum of (m^2 - a_i^2).
       With m the largest size every term is not negative, so squared length <= 8 * (largest entry)^2.

    Both sides of each are polynomials of degree at most 2 in every variable. Two such polynomials
    that agree at every point of the grid {0, 1, 2}^k are the same polynomial. So checking the
    6,561 points of {0, 1, 2}^8, and of {0, 1, 2}^9 for the third, proves the identities.
    """
    pairs = list(itertools.combinations(range(8), 2))
    for a in itertools.product((0, 1, 2), repeat=8):
        s, s2 = sum(a), sum(x * x for x in a)
        if 8 * s2 - s * s != sum((a[i] - a[j]) ** 2 for i, j in pairs):
            return False
        if s * s - s2 != 2 * sum(a[i] * a[j] for i, j in pairs):
            return False
        for m in (0, 1, 2):
            if 8 * m * m - s2 != sum(m * m - x * x for x in a):
                return False
    return True


def main():
    if not identities():
        print("FAIL: an identity does not hold")
        sys.exit(1)
    if not pi_bounds_hold():
        print("FAIL: the bounds on pi do not hold")
        sys.exit(1)
    certs = json.loads(open(sys.argv[1]).read())
    failed = 0
    for name, cert in certs.items():
        bad = check(cert)
        if bad:
            failed += 1
            print(f"FAIL {name}: {'; '.join(bad[:4])}")
    # a control: a certificate with the pick moved to another candidate must fail
    control = False
    for cert in certs.values():
        if len(cert["candidates"]) > 1:
            forged = dict(cert, pick=(cert["pick"] + 1) % len(cert["candidates"]))
            control = bool(check(forged))
            break
    print(f"identities: hold. bounds on pi: hold. certificates checked: {len(certs)}; failed: {failed}; control (a pick moved to the next candidate): {'caught' if control else 'NOT caught'}")
    sys.exit(0 if (failed == 0 and control) else 1)


if __name__ == "__main__":
    main()
