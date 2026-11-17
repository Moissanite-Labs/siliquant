#!/usr/bin/env python3
"""Proved limits on rounding weights, computed from their formulas.

Sections 1 and 2 hold floors and ceilings that no scheme can pass, and
constants of lattices (those of A_4* and A_8* from a closed form that
section 3 checks). A proposal is read against them before anything is
run: a gain that would have to exceed a ceiling is closed, and a gain inside
the limits has a known largest size. Section 3 checks one constant. Sections
4 to 6 hold covering thickness, codebook counts and a packing figure. Section
7 holds the cells of the point set F_n that PR #6 proposes as a codebook, and
section 8 the best lattices known in more than 8 dimensions, from a source.

Needs NumPy. Usage, from the root of a checkout:
  python3 bounds/bounds.py > bounds/bounds.out
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lattice"))
sys.path.insert(0, str(ROOT / "perm"))
import lattices as L  # noqa: E402
import permlib as P  # noqa: E402


def db(x):
    return 10.0 * math.log10(x)


def phi(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def Phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def cell_error(a, b, c):
    """Integral of (x - c)^2 against the standard normal density over (a, b)."""
    def f(x):
        if math.isinf(x):
            return (1.0 + c * c) * (1.0 if x > 0 else 0.0)
        return (1.0 + c * c) * Phi(x) - x * phi(x) + 2.0 * c * phi(x)
    return f(b) - f(a)


def uniform_error(q, step):
    """Error of q evenly spaced levels on a standard normal source. Mid-rise for even q, mid-tread for odd."""
    if q % 2 == 0:
        levels = [(k + 0.5) * step for k in range(-q // 2, q // 2)]
    else:
        levels = [k * step for k in range(-(q // 2), q // 2 + 1)]
    edges = [-math.inf] + [0.5 * (levels[i] + levels[i + 1]) for i in range(q - 1)] + [math.inf]
    return sum(cell_error(edges[i], edges[i + 1], levels[i]) for i in range(q))


def best_uniform(q):
    lo, hi = 0.01, 4.0
    g = (math.sqrt(5.0) - 1.0) / 2.0
    for _ in range(200):
        a, b = hi - g * (hi - lo), lo + g * (hi - lo)
        if uniform_error(q, a) < uniform_error(q, b):
            hi = b
        else:
            lo = a
    step = 0.5 * (lo + hi)
    return step, uniform_error(q, step)


def ramanujan_q(m):
    """Q(m) = sum over k = 1..m of m! / ((m - k)! m^k)."""
    total, term = 0.0, 1.0
    for k in range(1, m + 1):
        term *= (m - k + 1) / m
        total += term
    return total


def g_a_star(n):
    """Normalised second moment of A_n*. See the note under the table."""
    return (n + 1) ** (1.0 / n) / 12.0 * (1.0 - (ramanujan_q(n + 1) - 1.0) / n)


def ball_volume(n):
    return math.pi ** (n / 2.0) / math.gamma(n / 2.0 + 1.0)


def thickness_a_star(n):
    return math.sqrt(n + 1.0) * ball_volume(n) * (n * (n + 2.0) / (12.0 * (n + 1.0))) ** (n / 2.0)


def thickness_cubic(n):
    return ball_volume(n) * (math.sqrt(n) / 2.0) ** n


def g_pyramid(n, t):
    """Normalised second moment of F_n: the cube [-1, 1]^n cut into the 2n pyramids over its facets,
    with the reproduction point of each pyramid on its axis at distance t from the centre.

    Along the axis the height h has density n h^(n-1) on [0, 1]. Every other coordinate is uniform
    on [-h, h]. The volume per point is 2^n / 2n.
    """
    block = n / (n + 2.0) - 2.0 * t * n / (n + 1.0) + t * t + n * (n - 1.0) / (3.0 * (n + 2.0))
    return block / (n * (2.0 ** n / (2.0 * n)) ** (2.0 / n))


def g_pyramid_measured(n, t, points, seed):
    """The same by Monte Carlo: uniform points of the cube, each sent to the point of the pyramid it lies in."""
    import numpy as np
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=(points, n))
    i = np.argmax(np.abs(x), axis=1)
    rows = np.arange(points)
    e = (x ** 2).sum(axis=1) - x[rows, i] ** 2 + (np.abs(x[rows, i]) - t) ** 2
    scale = n * (2.0 ** n / (2.0 * n)) ** (2.0 / n)
    return float(e.mean()) / scale, float(e.std()) / math.sqrt(points) / scale


def main():
    print("1. Independent Gaussian weights, relative squared error at R bits per weight")
    print("   levels | bits | floor for any code, 2^(-2R) | best scalar quantiser (Lloyd-Max) | best evenly spaced levels | its step in standard deviations")
    for q in (3, 4, 8, 16):
        r = math.log2(q)
        lm = L.lloyd_max_gaussian(q)[1]
        step, ue = best_uniform(q)
        print(f"   {q:>6} | {r:.3f} | {q ** -2.0:.5f} = {db(q ** -2.0):.2f} dB | {lm:.5f} = {db(lm):.2f} dB | {ue:.5f} = {db(ue):.2f} dB | {step:.4f}")
    print("   The first column of numbers is Shannon's rate-distortion function: a proved floor for block codes of any length.")
    print("   The second is the floor for any code that rounds one weight at a time to a fixed number of levels.")
    print("   Both are floors on the expected error. One finite matrix can fall a little under them.")
    print()
    print("2. What the shape of the cells is worth when the cells are fine and nothing is clipped")
    print("   Error is G times the cell volume to the power 2/n. Gain is against the cube, G = 1/12. One bit is 6.02 dB.")
    print("   cells | G | gain in dB | gain in bits per weight")
    rows = [("cube, any dimension", 1.0 / 12.0), ("A_2* (hexagons)", g_a_star(2)), ("A_3*", g_a_star(3)), ("A_4*", g_a_star(4)),
            ("D_4", 13.0 / (120.0 * math.sqrt(2.0))), ("A_8*", g_a_star(8)), ("E_8", 929.0 / 12960.0),
            ("ball in 8 dimensions (no tiling reaches it)", L.sphere_bound_G(8)),
            ("ball in 24 dimensions", L.sphere_bound_G(24)), ("limit of the ball, 1 / (2 pi e)", 1.0 / (2.0 * math.pi * math.e))]
    for name, g in rows:
        gain = db((1.0 / 12.0) / g)
        print(f"   {name:<44} | {g:.6f} | {gain:.3f} | {gain / db(4.0):.4f}")
    print("   The ball rows are ceilings: no partition of space into cells of equal volume has a smaller G in that dimension.")
    print()
    print("3. The closed form used for A_n*, checked")
    print("   G(A_n*) = (n+1)^(1/n) / 12 * (1 - (Q(n+1) - 1) / n), with Q(m) = sum over k = 1..m of m! / ((m-k)! m^k).")
    exact = {1: 1.0 / 12.0, 2: 5.0 / (36.0 * math.sqrt(3.0)), 3: 19.0 / (192.0 * 2.0 ** (1.0 / 3.0))}
    for n, v in exact.items():
        print(f"   n = {n}: formula {g_a_star(n):.9f}, known exact value {v:.9f}, difference {abs(g_a_star(n) - v):.1e}")
    for n in (4, 8):
        est, se = L.mc_normalised_second_moment(L.Astar(n), n_points=4_000_000, seed=20261115)
        print(f"   n = {n}: formula {g_a_star(n):.6f}, measured with the nearest-point decoder {est:.6f} +- {se:.6f} ({(est - g_a_star(n)) / se:+.1f} standard errors)")
    est, se = L.mc_normalised_second_moment(L.E8(), n_points=4_000_000, seed=20261115)
    print(f"   E_8: exact 929/12960 = {929.0 / 12960.0:.6f}, measured {est:.6f} +- {se:.6f} ({(est - 929.0 / 12960.0) / se:+.1f} standard errors)")
    print("   The formula was fitted here to the exact values for n = 1, 2, 3 and then tested at n = 4 and 8. Its source has not been opened.")
    print()
    print("4. Covering thickness: the mean number of balls that cover a point. 1 would be a covering with no overlap")
    print("   n | cubic grid | A_n* | note")
    notes = {2: "A_2* is the thinnest covering of the plane", 3: "A_3* is the thinnest lattice covering", 4: "thinnest lattice covering",
             5: "thinnest lattice covering", 8: "E_8 has 4.0587; the best lattice known has 3.1422, a record and not a bound (Schurmann and Vallentin 2006)"}
    for n in (2, 3, 4, 5, 8):
        print(f"   {n} | {thickness_cubic(n):.4f} | {thickness_a_star(n):.4f} | {notes[n]}")
    print(f"   E_8 computed here: ball volume in 8 dimensions = {ball_volume(8):.4f}")
    print()
    print("5. Exact size of a codebook cut from a lattice by a box, by inclusion-exclusion")
    print("   Vectors of 9 integers with sum zero (the lattice A_8) and every entry between -k and k.")
    print("   k | codewords | bits per weight for 8 weights")
    for k in (1, 2, 3, 4):
        c = P.count_zero_sum_box(9, k)
        print(f"   {k} | {c} | {math.log2(c) / 8.0:.4f}")
    print("   A Voronoi code needs none of this: it has exactly q^n codewords, log2(q) bits per weight.")
    print()
    print("6. Packing ternary weights")
    print(f"   5 ternary weights fit in one byte: 243 of 256 states, {math.log2(243) / 8 * 100:.2f} percent of the byte's 8 bits, {8 / 5:.2f} bits stored per weight against log2(3) = {math.log2(3):.4f}")
    print()
    print("7. The point set F_n of Foundation-0/incentre-lattices as a codebook (proposed in PR #6): what its cells are worth")
    print("   The cube [-1, 1]^n is cut into the 2n pyramids over its facets, with one point on the axis of each at distance t from the centre.")
    print("   The cell of each point is its pyramid for any 0 < t < 1 (proof in experiments/incentre-vortex-codebooks-2026-11-17/README.md).")
    print("   G = [n/(n+2) - 2t n/(n+1) + t^2 + n(n-1)/(3(n+2))] / (n (2^n / 2n)^(2/n)). A negative gain means worse than the cube.")
    print("   n | G at t = 2 - sqrt 2 (the incentre) | gain in dB | G at t = n/(n+1) (the centroid, the best t) | gain in dB | measured at the centroid")
    c = 2.0 - math.sqrt(2.0)
    for n in (2, 3, 4, 8, 16, 24):
        gi, gc = g_pyramid(n, c), g_pyramid(n, n / (n + 1.0))
        est, se = g_pyramid_measured(n, n / (n + 1.0), 2_000_000, 20261117 + n)
        print(f"   {n:>2} | {gi:.6f} | {db((1.0 / 12.0) / gi):+.3f} | {gc:.6f} | {db((1.0 / 12.0) / gc):+.3f} | {est:.6f} +- {se:.6f} ({(est - gc) / se:+.1f} standard errors)")
    print("   F_n is behind the cube in every dimension. Its G tends to 1/12 from above as n grows.")
    print()
    print("8. Lattices in more than 8 dimensions: the best known, from a source and not recomputed here")
    print("   lattice | dimension | G | gain in dB against the cube | gain in dB against E_8")
    e8 = 929.0 / 12960.0
    for name, n, g, digits in (("K_12 (Coxeter-Todd)", 12, 0.070095600, 9), ("Lambda_16 (Barnes-Wall)", 16, 0.06830, 5), ("Lambda_24 (Leech)", 24, 0.06577, 5)):
        print(f"   {name:<24} | {n} | {g:.{digits}f} | {db((1.0 / 12.0) / g):.2f} | {db(e8 / g):.2f}")
    print("   Source: Agrell and Allen, arXiv 2202.09605, table I, with the digits it prints. K_12 is exact there and the other two are numerical.")
    print("   These are records and not ceilings. The ceilings are the ball rows of section 2.")


if __name__ == "__main__":
    main()
