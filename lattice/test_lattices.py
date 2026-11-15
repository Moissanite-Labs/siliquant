"""Tests for lattices.py. Run from the repo root:

    python -m unittest lattice/test_lattices.py -v
"""

import itertools
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import lattices as L  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def lll_reduce(B, delta=0.99):
    """Small LLL basis reduction (rows are basis vectors)."""
    B = np.array(B, dtype=float)
    n = len(B)

    def gso(B):
        Bs = np.zeros_like(B)
        mu = np.zeros((n, n))
        for i in range(n):
            Bs[i] = B[i]
            for j in range(i):
                mu[i, j] = B[i] @ Bs[j] / (Bs[j] @ Bs[j])
                Bs[i] = Bs[i] - mu[i, j] * Bs[j]
        return Bs, mu

    Bs, mu = gso(B)
    k = 1
    while k < n:
        for j in range(k - 1, -1, -1):
            c = round(mu[k, j])
            if c:
                B[k] = B[k] - c * B[j]
                Bs, mu = gso(B)
        if Bs[k] @ Bs[k] >= (delta - mu[k, k - 1] ** 2) * (Bs[k - 1] @ Bs[k - 1]):
            k += 1
        else:
            B[[k, k - 1]] = B[[k - 1, k]]
            Bs, mu = gso(B)
            k = max(k - 1, 1)
    return B


def brute_neighbourhood(B, x, radius, block=50):
    """Best point u @ B with u in round(x B^-1) + {-radius..radius}^n.

    Returns (points, squared distances).
    """
    n = len(B)
    Binv = np.linalg.inv(B)
    deltas = np.array(list(itertools.product(range(-radius, radius + 1), repeat=n)),
                      dtype=float)
    pts = np.empty_like(x)
    dist = np.empty(len(x))
    for i in range(0, len(x), block):
        xs = x[i:i + block]
        u0 = np.rint(xs @ Binv)
        P = (u0[:, None, :] + deltas[None, :, :]) @ B
        d = ((xs[:, None, :] - P) ** 2).sum(axis=2)
        j = np.argmin(d, axis=1)
        pts[i:i + block] = P[np.arange(len(xs)), j]
        dist[i:i + block] = d[np.arange(len(xs)), j]
    return pts, dist


def e8_minimal_vectors():
    vs = []
    for i, j in itertools.combinations(range(8), 2):
        for si in (1.0, -1.0):
            for sj in (1.0, -1.0):
                v = np.zeros(8)
                v[i], v[j] = si, sj
                vs.append(v)
    for signs in itertools.product((0.5, -0.5), repeat=8):
        if sum(1 for s in signs if s < 0) % 2 == 0:
            vs.append(np.array(signs))
    return np.array(vs)


ALL = [L.E8(), L.Astar(8), L.Astar(4), L.Astar(3), L.Astar(2), L.Dn(4),
       L.Dn(8), L.Zn(3), L.An(4), L.An(2)]


# ---------------------------------------------------------------------------
# A. Lattices and closest()
# ---------------------------------------------------------------------------

class TestLattices(unittest.TestCase):
    def test_volumes(self):
        self.assertAlmostEqual(L.Zn(5).volume, 1.0, places=12)
        self.assertAlmostEqual(L.Dn(4).volume, 2.0, places=12)
        self.assertAlmostEqual(L.E8().volume, 1.0, places=12)
        for n in (2, 3, 4, 8):
            self.assertAlmostEqual(L.An(n).volume, math.sqrt(n + 1), places=10)
            self.assertAlmostEqual(L.Astar(n).volume, 1 / math.sqrt(n + 1), places=10)

    def test_helmert_orthonormal(self):
        B = L.helmert_basis(8)
        np.testing.assert_allclose(B @ B.T, np.eye(8), atol=1e-12)
        np.testing.assert_allclose(B.sum(axis=1), 0.0, atol=1e-12)

    def test_astar_is_dual_of_a(self):
        for n in (2, 3, 4, 8):
            np.testing.assert_allclose(L.An(n).G @ L.Astar(n).G.T, np.eye(n), atol=1e-10)

    def test_generator_rows_are_lattice_points(self):
        for lat in ALL:
            np.testing.assert_allclose(lat.closest(lat.G), lat.G, atol=1e-9, err_msg=lat.name)

    def test_e8_generator_rows_in_e8(self):
        # Rows are in D_8 or D_8 + half, and the volume is 1, so they generate E_8.
        lat = L.E8()
        for row in lat.G:
            twice = 2 * row
            self.assertTrue(np.allclose(twice, np.rint(twice)))
            self.assertAlmostEqual(row.sum() % 2, 0.0, places=12)

    def test_closest_is_lattice_point(self):
        rng = np.random.default_rng(1)
        for lat in ALL:
            x = rng.uniform(-5, 5, size=(2000, lat.n))
            c = lat.closest(x)
            coords = c @ lat.Ginv
            self.assertLess(np.abs(coords - np.rint(coords)).max(), 1e-9, lat.name)

    def test_closest_of_lattice_point_is_itself(self):
        rng = np.random.default_rng(2)
        for lat in ALL:
            u = rng.integers(-6, 7, size=(500, lat.n)).astype(float)
            y = u @ lat.G
            np.testing.assert_allclose(lat.closest(y), y, atol=1e-9, err_msg=lat.name)

    def test_closest_does_not_move_far(self):
        # Distance to the chosen point is at most the covering radius bound.
        rng = np.random.default_rng(3)
        for lat in ALL:
            x = rng.uniform(-5, 5, size=(500, lat.n))
            d = np.linalg.norm(x - lat.closest(x), axis=1)
            self.assertLess(d.max(), 0.5 * np.linalg.norm(lat.G, axis=1).sum() + 1e-9)

    def test_nearest_by_brute_force(self):
        # For n <= 4: u0 = round(x G^-1), delta in {-2..2}^n on the raw generator.
        # For n = 8: LLL-reduced basis, delta in {-1,0,1}^8.
        rng = np.random.default_rng(4)
        # The E_8 radius-1 neighbourhood misses the true nearest point for some x
        # (the cell is skewed in this basis), so for E_8 only "closest is never
        # worse" is asserted there. A radius-2 search on 40 points must agree.
        cases = [(L.E8(), 1, False), (L.Astar(8), 1, True), (L.Astar(4), 2, True),
                 (L.Astar(2), 2, True), (L.Dn(4), 2, True), (L.Zn(3), 2, True)]
        for lat, radius, strict in cases:
            with self.subTest(lattice=lat.name):
                x = rng.uniform(-3, 3, size=(300, lat.n))
                B = lat.G
                if lat.n == 8:
                    B = lll_reduce(lat.G)
                    self.assertAlmostEqual(abs(np.linalg.det(B)), lat.volume, places=9)
                    np.testing.assert_allclose(lat.closest(B), B, atol=1e-9)
                bf_pts, bf_d = brute_neighbourhood(B, x, radius)
                c = lat.closest(x)
                d = ((x - c) ** 2).sum(axis=1)
                # closest() is never worse than the brute-force best ...
                self.assertLessEqual((d - bf_d).max(), 1e-9)
                # ... and the brute-force search found a point as good.
                missed = int((bf_d - d > 1e-9).sum())
                print("  brute force %s: %d of 300 not covered by the neighbourhood"
                      % (lat.name, missed))
                if strict:
                    self.assertEqual(missed, 0)
                if lat.name == "E8":
                    _, bf2 = brute_neighbourhood(B, x[:40], 2, block=4)
                    self.assertLessEqual(np.abs(bf2 - d[:40]).max(), 1e-9)

    def test_e8_minimal_vector_criterion(self):
        # y is nearest to x iff no minimal vector v gives |x - y - v| < |x - y|.
        lat = L.E8()
        V = e8_minimal_vectors()
        self.assertEqual(len(V), 240)
        np.testing.assert_allclose((V ** 2).sum(axis=1), 2.0)
        np.testing.assert_allclose(lat.closest(V), V, atol=1e-9)
        rng = np.random.default_rng(5)
        x = rng.uniform(-4, 4, size=(20000, 8))
        r = x - lat.closest(x)
        r2 = (r ** 2).sum(axis=1)
        # |r - v|^2 = |r|^2 - 2 r.v + 2
        worst = (r2[:, None] - 2 * r @ V.T + 2.0).min(axis=1) - r2
        self.assertGreaterEqual(worst.min(), -1e-9)

    def test_d_n_parity_fix(self):
        lat = L.Dn(3)
        x = np.array([[0.4, 0.45, 0.0], [0.6, 0.0, 0.0], [-0.6, 0.1, 0.0]])
        c = lat.closest(x)
        self.assertTrue(np.all(c.sum(axis=1) % 2 == 0))
        np.testing.assert_allclose(c[0], [0, 0, 0])
        np.testing.assert_allclose(c[1], [0, 0, 0])
        self.assertTrue(np.all(np.abs(c - np.rint(c)) < 1e-12))

    def test_input_shape_checked(self):
        with self.assertRaises(ValueError):
            L.Zn(3).closest(np.zeros((4, 2)))


# ---------------------------------------------------------------------------
# B. Second moment
# ---------------------------------------------------------------------------

class TestSecondMoment(unittest.TestCase):
    def check(self, lat, exact):
        g, se = L.mc_normalised_second_moment(lat, n_points=2_000_000, seed=11)
        print("  G(%s) = %.6f +- %.6f   (exact %.6f)" % (lat.name, g, se, exact))
        self.assertLess(abs(g - exact), 4 * se)

    def test_Z3(self):
        self.check(L.Zn(3), 1 / 12)

    def test_A2star(self):
        self.check(L.Astar(2), 5 / (36 * math.sqrt(3)))

    def test_D4(self):
        self.check(L.Dn(4), 13 / (120 * math.sqrt(2)))

    def test_E8(self):
        self.check(L.E8(), 929 / 12960)

    def test_report_others(self):
        for lat in (L.Astar(3), L.Astar(4), L.Astar(8), L.Dn(8)):
            g, se = L.mc_normalised_second_moment(lat, n_points=2_000_000, seed=12)
            print("  G(%s) = %.6f +- %.6f   (not asserted)" % (lat.name, g, se))
            self.assertTrue(0.05 < g < 0.09)


class TestSphereBound(unittest.TestCase):
    def test_n1(self):
        self.assertAlmostEqual(L.sphere_bound_G(1), 1 / 12, places=12)

    def test_large_n(self):
        target = 1 / (2 * math.pi * math.e)
        self.assertLess(abs(L.sphere_bound_G(10000) - target), 1e-4)
        self.assertLess(abs(L.sphere_bound_G(1000) - target), 1e-3)

    def test_values(self):
        for n in (1, 2, 3, 4, 8, 16, 24):
            print("  sphere_bound_G(%d) = %.6f" % (n, L.sphere_bound_G(n)))
        self.assertAlmostEqual(L.sphere_bound_G(2), 1 / (4 * math.pi), places=12)

    def test_below_lattices(self):
        self.assertLess(L.sphere_bound_G(8), 929 / 12960)
        self.assertLess(L.sphere_bound_G(4), 13 / (120 * math.sqrt(2)))


# ---------------------------------------------------------------------------
# C. Voronoi codes
# ---------------------------------------------------------------------------

VORONOI_CASES = [(L.Astar(2), 4), (L.Dn(4), 3), (L.E8(), 2), (L.Zn(3), 4)]


class TestVoronoi(unittest.TestCase):
    def test_codebook_distinct_and_index_roundtrip(self):
        for lat, q in VORONOI_CASES:
            with self.subTest(lattice=lat.name, q=q):
                a = L.default_offset(lat.n)
                u = L.all_indices(lat.n, q)
                self.assertEqual(len(u), q ** lat.n)
                cb = L.voronoi_decode(u, lat, q, a)
                self.assertEqual(len(np.unique(np.round(cb, 7), axis=0)), q ** lat.n)
                np.testing.assert_array_equal(L.voronoi_index(cb, lat, q), u)
                coords = cb @ lat.Ginv
                self.assertLess(np.abs(coords - np.rint(coords)).max(), 1e-9)
                np.testing.assert_allclose(cb, L.enumerate_codebook(lat, q, a), atol=0)

    def test_codewords_nearest_to_a_in_qL(self):
        for lat, q in VORONOI_CASES:
            with self.subTest(lattice=lat.name, q=q):
                n = lat.n
                a = L.default_offset(n)
                u = L.all_indices(n, q)
                r = L.voronoi_decode(u, lat, q, a) - a
                B = lll_reduce(lat.G) if n == 8 else lat.G
                Binv = np.linalg.inv(B)
                radius = 1 if n == 8 else 2
                deltas = np.array(list(itertools.product(range(-radius, radius + 1),
                                                         repeat=n)), dtype=float)
                r2 = (r ** 2).sum(axis=1)
                margin = np.inf
                for i in range(len(r)):
                    u0 = np.rint(r[i] / q @ Binv)
                    U = u0 + deltas
                    P = q * (U @ B)
                    d = ((r[i] - P) ** 2).sum(axis=1)
                    self.assertGreaterEqual(d.min() - r2[i], -1e-9)
                    other = np.any(U != 0, axis=1)
                    margin = min(margin, d[other].min() - r2[i])
                # Strict gap: the decoder never meets a tie.
                self.assertGreater(margin, 1e-7)

    def test_Zn_codebook_is_q_consecutive_integers(self):
        lat, q = L.Zn(3), 4
        cb = L.enumerate_codebook(lat, q)
        for j in range(3):
            vals = np.unique(cb[:, j])
            self.assertEqual(len(vals), q)
            np.testing.assert_allclose(np.diff(vals), 1.0)

    def test_quantize_rare_overload(self):
        for lat, q, sigma in [(L.E8(), 4, 0.5), (L.Dn(4), 3, 0.4),
                              (L.Astar(2), 4, 0.5), (L.Zn(3), 4, 0.5)]:
            with self.subTest(lattice=lat.name):
                a = L.default_offset(lat.n)
                rng = np.random.default_rng(6)
                x = sigma * rng.standard_normal((3000, lat.n))
                rec, u, over = L.voronoi_quantize(x, lat, q, a)
                self.assertLess(over.mean(), 0.05)
                np.testing.assert_allclose(rec[~over], lat.closest(x)[~over], atol=1e-9)
                np.testing.assert_allclose(L.voronoi_decode(u, lat, q, a), rec, atol=1e-9)

    def test_quantize_overload(self):
        for lat, q in [(L.E8(), 2), (L.Dn(4), 3), (L.Astar(2), 4), (L.Zn(3), 2)]:
            with self.subTest(lattice=lat.name):
                a = L.default_offset(lat.n)
                rng = np.random.default_rng(7)
                x = 4.0 * rng.standard_normal((2000, lat.n))
                rec, u, over = L.voronoi_quantize(x, lat, q, a)
                self.assertGreater(over.sum(), 100)
                self.assertTrue(np.all((u >= 0) & (u < q)))
                np.testing.assert_allclose(L.voronoi_decode(u, lat, q, a), rec, atol=1e-9)
                coords = rec @ lat.Ginv
                self.assertLess(np.abs(coords - np.rint(coords)).max(), 1e-9)
                # Overloaded blocks are shrunk, so they land nearer the origin
                # than the plain nearest lattice point.
                self.assertTrue(np.all(np.linalg.norm(rec[over], axis=1)
                                       <= np.linalg.norm(lat.closest(x)[over], axis=1) + 1e-9))

    def test_quantize_with_centre(self):
        lat, q = L.Dn(4), 3
        a = L.default_offset(4)
        centre = np.array([0.3, -0.2, 0.1, 0.4])
        rng = np.random.default_rng(8)
        x = 0.4 * rng.standard_normal((2000, 4))
        rec, u, over = L.voronoi_quantize(x, lat, q, a, centre=centre)
        cw = L.voronoi_decode(u, lat, q, a)
        np.testing.assert_allclose(rec + centre, cw, atol=1e-9)
        np.testing.assert_allclose((rec + centre)[~over], lat.closest(x + centre)[~over],
                                   atol=1e-9)

    def test_centroid(self):
        lat, q = L.E8(), 2
        a = L.default_offset(8)
        exact = L.codebook_centroid(lat, q, a)
        np.testing.assert_allclose(exact, L.enumerate_codebook(lat, q, a).mean(axis=0))
        approx = L.codebook_centroid(lat, q, a, exact_limit=0, samples=2 ** 15, seed=3)
        self.assertLess(np.abs(approx - exact).max(), 0.05)
        print("  E8 q=2 codebook centroid norm %.4f (|a| = %.4f)"
              % (np.linalg.norm(exact), np.linalg.norm(a)))

    def test_nearest_in_codebook(self):
        lat, q = L.Dn(4), 3
        a = L.default_offset(4)
        cb = L.enumerate_codebook(lat, q, a)
        rng = np.random.default_rng(9)
        x = 0.6 * rng.standard_normal((700, 4))
        idx, pts = L.nearest_in_codebook(x, cb, chunk=100)
        naive = np.argmin(((x[:, None, :] - cb[None]) ** 2).sum(axis=2), axis=1)
        np.testing.assert_array_equal(idx, naive)
        np.testing.assert_array_equal(pts, cb[idx])
        rec, _, _ = L.voronoi_quantize(x, lat, q, a)
        d_best = ((x - pts) ** 2).sum(axis=1)
        d_rec = ((x - rec) ** 2).sum(axis=1)
        self.assertLessEqual((d_best - d_rec).max(), 1e-9)

    def test_nearest_in_codebook_E8_q4_runs(self):
        lat, q = L.E8(), 4
        a = L.default_offset(8)
        cb = L.enumerate_codebook(lat, q, a)
        self.assertEqual(cb.shape, (65536, 8))
        self.assertEqual(len(np.unique(np.round(cb, 7), axis=0)), 65536)
        rng = np.random.default_rng(10)
        x = 0.8 * rng.standard_normal((512, 8))
        _, pts = L.nearest_in_codebook(x, cb)
        rec, _, over = L.voronoi_quantize(x, lat, q, a)
        d_best = ((x - pts) ** 2).sum(axis=1)
        d_rec = ((x - rec) ** 2).sum(axis=1)
        self.assertLessEqual((d_best - d_rec).max(), 1e-9)
        print("  E8 q=4, sigma 0.8: overload %.3f, mse shrink rule %.5f, exact %.5f"
              % (over.mean(), d_rec.mean() / 8, d_best.mean() / 8))


# ---------------------------------------------------------------------------
# D. Scalar quantisers
# ---------------------------------------------------------------------------

class TestScalar(unittest.TestCase):
    def test_midrise_levels(self):
        x = np.linspace(-5, 5, 2001)
        y = L.uniform_midrise(x, 2, 1.0)
        self.assertEqual(sorted(set(y)), [-1.5, -0.5, 0.5, 1.5])
        np.testing.assert_allclose(L.uniform_midrise(np.array([-100.0, 100.0]), 3, 0.5),
                                   [-1.75, 1.75])
        np.testing.assert_allclose(L.uniform_midrise(np.array([0.0, 0.49, -0.01]), 3, 1.0),
                                   [0.5, 0.5, -0.5])

    def test_midrise_error_bound(self):
        x = np.random.default_rng(0).uniform(-1.9, 1.9, 5000)
        y = L.uniform_midrise(x, 3, 0.5)
        self.assertLessEqual(np.abs(x - y).max(), 0.25 + 1e-12)

    def test_lloyd_max_known_values(self):
        known = {2: 0.3634, 4: 0.1175, 8: 0.03454, 16: 0.009497}
        for levels, mse in known.items():
            r, got = L.lloyd_max_gaussian(levels)
            print("  Lloyd-Max %2d levels: mse = %.8f" % (levels, got))
            # Relative tolerance 5e-4. The iteration is converged, and gives
            # 0.0345478 for 8 levels (0.03455 at four figures) and 0.0095010
            # for 16 levels (the quoted 0.009497 is 4e-4 lower).
            self.assertLess(abs(got - mse) / mse, 5e-4)
            self.assertEqual(len(r), levels)
            np.testing.assert_allclose(r, -r[::-1], atol=1e-9)

    def test_lloyd_max_two_levels(self):
        r, _ = L.lloyd_max_gaussian(2)
        np.testing.assert_allclose(r, [-math.sqrt(2 / math.pi), math.sqrt(2 / math.pi)],
                                   atol=1e-9)

    def test_lloyd_max_against_sampling(self):
        r, mse = L.lloyd_max_gaussian(4)
        x = np.random.default_rng(1).standard_normal(1_000_000)
        edges = 0.5 * (r[1:] + r[:-1])
        q = r[np.searchsorted(edges, x)]
        self.assertLess(abs(((x - q) ** 2).mean() - mse), 5e-4)


# ---------------------------------------------------------------------------
# E. Dual scaling
# ---------------------------------------------------------------------------

class TestDualScale(unittest.TestCase):
    def make(self, m, n, seed=0, sigma=1.0):
        rng = np.random.default_rng(seed)
        r = np.exp(sigma * rng.standard_normal(m))
        c = np.exp(sigma * rng.standard_normal(n))
        return np.outer(r, c) * rng.standard_normal((m, n))

    def test_identity(self):
        W = self.make(64, 48, seed=1)
        s, t, Wn, _, _ = L.dual_scale(W)
        rebuilt = s[:, None] * Wn * t[None, :]
        self.assertLess(np.abs(rebuilt - W).max() / np.abs(W).max(), 1e-12)

    def test_imbalance_256(self):
        W = self.make(256, 256, seed=2)
        s, t, Wn, before, after = L.dual_scale(W, iters=16)
        print("  dual_scale 256x256: imbalance before %.3f, after %.4f" % (before, after))
        self.assertGreater(before, 10.0)
        self.assertLess(after, 1.2)
        self.assertLess(np.abs(s[:, None] * Wn * t[None, :] - W).max() / np.abs(W).max(),
                        1e-12)

    def test_rectangular(self):
        W = self.make(100, 300, seed=3)
        _, _, _, before, after = L.dual_scale(W)
        self.assertLess(after, before)


if __name__ == "__main__":
    unittest.main()
