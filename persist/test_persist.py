"""Tests of persistlib.py against full enumeration, a second method and the paper's bound.

    python3 -m unittest persist/test_persist.py
"""
from __future__ import annotations

import itertools
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import persistlib as P  # noqa: E402


def spanning_trees(a, b):
    """Every spanning tree of the complete bipartite graph, as tuples of (row, col)."""
    links = [(i, j) for i in range(a) for j in range(b)]
    for subset in itertools.combinations(links, a + b - 1):
        parent = list(range(a + b))

        def find(x):
            while parent[x] != x:
                x = parent[x]
            return x

        ok = True
        for i, j in subset:
            ri, rj = find(i), find(a + j)
            if ri == rj:
                ok = False
                break
            parent[ri] = rj
        if ok:
            yield subset


class TreeTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20261117)

    def test_counts_of_spanning_trees(self):
        # Scoins' formula: a^(b-1) * b^(a-1) spanning trees.
        for a, b in [(2, 2), (2, 3), (3, 3), (2, 4)]:
            self.assertEqual(sum(1 for _ in spanning_trees(a, b)), a ** (b - 1) * b ** (a - 1))

    def test_against_full_enumeration(self):
        # The tree's sorted sizes are at least those of every other spanning tree, place by place.
        for a, b in [(2, 2), (2, 3), (3, 3), (2, 4)]:
            for _ in range(20):
                A = self.rng.random((a, b))
                best = np.sort(P.tree(A)[2])[::-1]
                for t in spanning_trees(a, b):
                    other = np.sort([A[i, j] for i, j in t])[::-1]
                    self.assertTrue((best >= other - 1e-15).all())

    def test_prim_equals_kruskal(self):
        for shape in [(1, 1), (1, 7), (7, 1), (5, 5), (8, 31), (40, 12), (64, 64)]:
            for kind in ("real", "ties"):
                A = self.rng.random(shape)
                if kind == "ties":
                    A = np.floor(A * 4)
                s1 = np.sort(P.tree(A)[2])
                s2 = np.sort(P.tree_kruskal(A)[2])
                self.assertEqual(len(s1), shape[0] + shape[1] - 1)
                np.testing.assert_allclose(s1, s2, rtol=0, atol=0)

    def test_tree_is_spanning(self):
        A = self.rng.random((9, 14))
        r, c, _ = P.tree(A)
        parent = list(range(23))

        def find(x):
            while parent[x] != x:
                x = parent[x]
            return x

        for i, j in zip(r, c):
            ri, rj = find(int(i)), find(9 + int(j))
            self.assertNotEqual(ri, rj)
            parent[ri] = rj
        self.assertEqual(len({find(x) for x in range(23)}), 1)

    def test_largest_link_of_every_vertex_is_in_the_tree(self):
        for shape in [(6, 6), (10, 40), (33, 7)]:
            A = self.rng.random(shape)
            r, c, _ = P.tree(A)
            links = set(zip(r.tolist(), c.tolist()))
            for i in range(shape[0]):
                self.assertIn((i, int(np.argmax(A[i]))), links)
            for j in range(shape[1]):
                self.assertIn((int(np.argmax(A[:, j])), j), links)

    def test_rejects_negative_sizes(self):
        with self.assertRaises(ValueError):
            P.tree(np.array([[1.0, -1.0]]))


class ScoreTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20261118)

    def test_zero_when_all_weights_equal(self):
        W = np.full((5, 8), 0.3)
        self.assertEqual(P.persistence(W), 0.0)
        self.assertEqual(P.persistence(W, essential="min"), 0.0)

    def test_bound_of_theorem_1_holds_with_the_layers_own_largest_weight(self):
        for shape in [(4, 4), (10, 30), (50, 20)]:
            for p in (1, 2, 3):
                W = self.rng.standard_normal(shape)
                for essential in ("none", "min"):
                    self.assertLessEqual(P.persistence(W, p, essential=essential),
                                         P.bound(W, p) * (1 + 1e-12))

    def test_bound_is_reached_only_with_the_last_piece_counted(self):
        # The theorem's example: one link of size b, every other of size a.
        a_size, b_size = 0.2, 1.0
        W = np.full((6, 9), a_size)
        W[2, 5] = b_size
        for p in (1, 2):
            self.assertAlmostEqual(P.persistence(W, p, essential="min"), P.bound(W, p), places=12)
            self.assertLess(P.persistence(W, p, essential="none"), P.bound(W, p))

    def test_last_piece_at_level_zero_passes_the_bound(self):
        W = np.full((6, 9), 0.2)
        W[2, 5] = 1.0
        self.assertGreater(P.persistence(W, essential="zero"), P.bound(W))
        self.assertGreater(P.normalised(W, essential="zero"), 1.0)

    def test_constant_layer_scores_zero(self):
        W = np.full((4, 7), -0.5)
        self.assertEqual(P.bound(W), 0.0)
        self.assertEqual(P.normalised(W), 0.0)
        self.assertEqual(P.mean_normalised([W, W]), 0.0)

    def test_bound_fails_with_the_largest_weight_of_a_network(self):
        W = self.rng.standard_normal((16, 16)) * 0.1
        wmax = 10 * np.abs(W).max()
        self.assertGreater(P.persistence(W, wmax=wmax), P.bound(W, wmax=wmax))

    def test_weights_count_gives_a_smaller_score(self):
        W = self.rng.standard_normal((30, 30))
        ratio = P.normalised(W, count="neurons") / P.normalised(W, count="weights")
        self.assertAlmostEqual(ratio, ((900 - 1) / (60 - 1)) ** 0.5, places=10)

    def test_unchanged_by_reordering_sign_and_one_scale(self):
        W = self.rng.standard_normal((12, 20))
        base = P.persistence(W)
        self.assertAlmostEqual(P.persistence(W[self.rng.permutation(12)]), base, places=12)
        self.assertAlmostEqual(P.persistence(W[:, self.rng.permutation(20)]), base, places=12)
        self.assertAlmostEqual(P.persistence(-W), base, places=12)
        self.assertAlmostEqual(P.persistence(W * self.rng.choice([-1.0, 1.0], W.shape)), base, places=12)
        self.assertAlmostEqual(P.persistence(3.7 * W), base, places=12)

    def test_sign_flip_changes_the_function_and_not_the_score(self):
        d, h, o = 8, 32, 4
        W1 = self.rng.standard_normal((h, d))
        W2 = self.rng.standard_normal((o, h))
        x = self.rng.standard_normal((200, d))
        f = lambda A1, A2: np.maximum(x @ A1.T, 0) @ A2.T  # noqa: E731
        self.assertEqual(P.persistence(-W1), P.persistence(W1))
        self.assertGreater(np.linalg.norm(f(W1, W2) - f(-W1, W2)), np.linalg.norm(f(W1, W2)))

    def test_rescaling_units_keeps_the_function_and_changes_the_score(self):
        d, h, o = 8, 32, 4
        W1 = self.rng.standard_normal((h, d))
        W2 = self.rng.standard_normal((o, h))
        x = self.rng.standard_normal((200, d))
        f = lambda A1, A2: np.maximum(x @ A1.T, 0) @ A2.T  # noqa: E731
        s = np.exp(self.rng.standard_normal(h))
        V1, V2 = W1 * s[:, None], W2 / s[None, :]
        np.testing.assert_allclose(f(V1, V2), f(W1, W2), rtol=1e-10, atol=1e-10)
        before = P.mean_normalised([W1, W2])
        after = P.mean_normalised([V1, V2])
        self.assertGreater(abs(after - before), 0.05)

    def test_maxima_share_and_estimate(self):
        W = self.rng.standard_normal((64, 256))
        self.assertGreater(P.maxima_share(W), 0.5)
        self.assertLessEqual(P.persistence_from_maxima(W), P.persistence(W) * (1 + 1e-12))
        self.assertGreater(P.persistence_from_maxima(W), 0.9 * P.persistence(W))

    def test_sorted_sizes_move_no_more_than_the_largest_entry_change(self):
        for _ in range(50):
            A = np.abs(self.rng.standard_normal((9, 13)))
            B = np.abs(A + 0.3 * self.rng.standard_normal((9, 13)))
            gap = np.abs(A - B).max()
            self.assertLessEqual(P.sorted_distance(A, B, p=np.inf), gap + 1e-12)
            # and after any reordering of the second matrix
            Bp = B[self.rng.permutation(9)][:, self.rng.permutation(13)]
            self.assertAlmostEqual(P.sorted_distance(A, Bp, p=np.inf),
                                   P.sorted_distance(A, B, p=np.inf), places=12)

    def test_sorted_distance_against_frobenius_on_samples(self):
        # Not proved here. Recorded as a check on samples only.
        worst = 0.0
        for _ in range(200):
            A = np.abs(self.rng.standard_normal((7, 11)))
            B = np.abs(A + self.rng.standard_normal((7, 11)) * self.rng.random())
            worst = max(worst, P.sorted_distance(A, B, p=2) / np.linalg.norm(A - B))
        self.assertLessEqual(worst, 1.0)


if __name__ == "__main__":
    unittest.main()
