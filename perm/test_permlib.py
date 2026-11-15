"""Tests for permlib. Run from the repo root:
python3 -m unittest perm/test_permlib.py -v
"""

import itertools
import math
import os
import random
import sys
import unittest
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import permlib as P  # noqa: E402


def random_board(rng, n, lo=0.05, hi=0.95):
    p = rng.uniform(lo, hi)
    return {(i, j) for i in range(n) for j in range(n) if rng.random() < p}


def all_boards(n):
    cells = [(i, j) for i in range(n) for j in range(n)]
    for mask in range(1 << len(cells)):
        yield {cells[t] for t in range(len(cells)) if (mask >> t) & 1}


def board_to_allowed(board, n):
    return [[0 if (i, j) in board else 1 for j in range(n)] for i in range(n)]


class TestDerangements(unittest.TestCase):
    def test_three_ways_agree(self):
        for n in range(41):
            a = P.derangements_inclusion_exclusion(n)
            self.assertEqual(a, P.derangements_two_term(n))
            self.assertEqual(a, P.derangements_alternating(n))
            self.assertEqual(a, P.derangements(n))

    def test_known_values(self):
        known = [1, 0, 1, 2, 9, 44, 265, 1854, 14833, 133496, 1334961]
        self.assertEqual([P.derangements(n) for n in range(11)], known)

    def test_round_n_factorial_over_e(self):
        e = sum(Fraction(1, math.factorial(k)) for k in range(80))
        for n in range(1, 16):
            x = Fraction(math.factorial(n)) / e
            self.assertEqual(P.derangements(n), math.floor(x + Fraction(1, 2)))

    def test_distance_to_inverse_e(self):
        e = sum(Fraction(1, math.factorial(k)) for k in range(80))
        for n in range(16):
            gap = abs(Fraction(P.derangements(n), math.factorial(n)) - 1 / e)
            self.assertLess(gap, Fraction(1, math.factorial(n + 1)))


class TestRencontres(unittest.TestCase):
    def test_sum_is_factorial(self):
        for n in range(0, 15):
            self.assertEqual(sum(P.rencontres(n, k) for k in range(n + 1)),
                             math.factorial(n))

    def test_bruteforce(self):
        for n in range(1, 8):
            counts = [0] * (n + 1)
            for s in itertools.permutations(range(n)):
                counts[sum(1 for i in range(n) if s[i] == i)] += 1
            self.assertEqual(counts, [P.rencontres(n, k) for k in range(n + 1)])


class TestRookNumbers(unittest.TestCase):
    def check(self, board, n, with_brute=True):
        b = P.rook_numbers_bruteforce(board, n) if with_brute else None
        dc = P.rook_numbers_deletion_contraction(board, n)
        dp = P.rook_numbers_dp(board, n)
        self.assertEqual(dc, dp, (n, sorted(board)))
        if with_brute:
            self.assertEqual(b, dp, (n, sorted(board)))

    def test_all_boards_small(self):
        for n in (1, 2, 3):
            for board in all_boards(n):
                self.check(board, n)

    def test_random_n4_to_6(self):
        rng = random.Random(1)
        for _ in range(300):
            n = rng.randint(4, 6)
            self.check(random_board(rng, n), n)

    def test_dc_dp_n10(self):
        rng = random.Random(2)
        for _ in range(50):
            board = random_board(rng, 10, 0.1, 0.4)
            self.check(board, 10, with_brute=False)

    def test_diagonal_is_binomial(self):
        for n in (5, 20, 40):
            board = {(i, i) for i in range(n)}
            self.assertEqual(P.rook_numbers_deletion_contraction(board, n),
                             [math.comb(n, k) for k in range(n + 1)])


class TestFerrers(unittest.TestCase):
    def test_identity(self):
        rng = random.Random(3)
        for _ in range(50):
            n = rng.randint(1, 8)
            heights = sorted(rng.randint(0, n) for _ in range(n))
            for x in range(n + 4):
                lhs, rhs = P.ferrers_check(heights, x)
                self.assertEqual(lhs, rhs, (heights, x))


class TestHitNumbers(unittest.TestCase):
    def test_bruteforce(self):
        rng = random.Random(4)
        for _ in range(60):
            n = rng.randint(1, 7)
            board = random_board(rng, n)
            self.assertEqual(P.hit_numbers(board, n),
                             P.hit_numbers_bruteforce(board, n))

    def test_sum_is_factorial(self):
        rng = random.Random(5)
        for _ in range(60):
            n = rng.randint(1, 9)
            board = random_board(rng, n)
            self.assertEqual(sum(P.hit_numbers(board, n)), math.factorial(n))

    def test_diagonal_gives_rencontres(self):
        for n in range(1, 9):
            board = {(i, i) for i in range(n)}
            self.assertEqual(P.hit_numbers(board, n),
                             [P.rencontres(n, k) for k in range(n + 1)])


class TestPermanents(unittest.TestCase):
    def test_agree_on_integer_matrices(self):
        rng = random.Random(6)
        for n in range(0, 9):
            for _ in range(20 if n < 8 else 6):
                A = [[rng.randint(-3, 3) for _ in range(n)] for _ in range(n)]
                b = P.permanent_bruteforce(A)
                self.assertEqual(b, P.permanent_ryser(A), (n, A))
                self.assertEqual(b, P.permanent_glynn(A), (n, A))

    def test_agree_on_fractions(self):
        rng = random.Random(7)
        for n in range(1, 6):
            for _ in range(10):
                A = [[Fraction(rng.randint(-5, 5), rng.randint(1, 4))
                      for _ in range(n)] for _ in range(n)]
                b = P.permanent_bruteforce(A)
                self.assertEqual(b, P.permanent_ryser(A))
                self.assertEqual(b, P.permanent_glynn(A))

    def test_all_ones_and_derangements(self):
        for n in range(1, 17):
            J = [[1] * n for _ in range(n)]
            JI = [[0 if i == j else 1 for j in range(n)] for i in range(n)]
            self.assertEqual(P.permanent_ryser(J), math.factorial(n))
            self.assertEqual(P.permanent_ryser(JI), P.derangements(n))

    def test_hit_polynomial(self):
        rng = random.Random(8)
        for _ in range(40):
            n = rng.randint(1, 7)
            board = random_board(rng, n)
            h = P.hit_numbers(board, n)
            for x in (0, 1, 2, 3, -1):
                A = [[x if (i, j) in board else 1 for j in range(n)] for i in range(n)]
                self.assertEqual(P.permanent_ryser(A),
                                 sum(h[j] * x ** j for j in range(n + 1)))


class TestMenage(unittest.TestCase):
    def test_known_and_bruteforce(self):
        known = {3: 1, 4: 2, 5: 13, 6: 80, 7: 579, 8: 4738, 9: 43387, 10: 439792}
        for n, v in known.items():
            self.assertEqual(P.menage(n), v)
        for n in range(3, 10):
            self.assertEqual(P.menage(n), P.menage_bruteforce(n))

    def test_small_conventions(self):
        for n in (0, 1, 2):
            self.assertEqual(P.menage(n), P.menage_bruteforce(n))

    def test_hit_numbers_and_rook_numbers(self):
        for n in range(3, 10):
            board = P.menage_board(n)
            self.assertEqual(P.hit_numbers(board, n)[0], P.menage(n))
            r = P.rook_numbers(board, n)
            for k in range(n + 1):
                self.assertEqual(r[k], P.menage_rook_number(n, k))


class TestZeroSumBox(unittest.TestCase):
    def test_bruteforce(self):
        for m in range(1, 7):
            for k in range(0, 4):
                count = sum(1 for v in itertools.product(range(-k, k + 1), repeat=m)
                            if sum(v) == 0)
                self.assertEqual(P.count_zero_sum_box(m, k), count, (m, k))

    def test_known(self):
        self.assertEqual(P.count_zero_sum_box(9, 1), 3139)


class TestMatching(unittest.TestCase):
    def brute(self, A):
        n = len(A)
        matchings = [s for s in itertools.permutations(range(n))
                     if all(A[i][s[i]] for i in range(n))]
        used = {(i, s[i]) for s in matchings for i in range(n)}
        return matchings, used

    def test_dead_cells_and_hall(self):
        rng = random.Random(9)
        seen_dead = 0
        for _ in range(500):
            n = rng.randint(1, 6)
            p = rng.choice((0.3, 0.5, 0.7, 0.9))
            A = [[1 if rng.random() < p else 0 for _ in range(n)] for _ in range(n)]
            matchings, used = self.brute(A)
            perm = P.permanent_bruteforce(A)
            self.assertEqual(perm, len(matchings))
            viol = P.hall_violator(A)
            if perm == 0:
                self.assertIsNotNone(viol)
                nbhd = {c for r in viol for c in range(n) if A[r][c]}
                self.assertLess(len(nbhd), len(viol))
                self.assertEqual(P.dead_cells(A), {})
            else:
                self.assertIsNone(viol)
                allowed = {(i, j) for i in range(n) for j in range(n) if A[i][j]}
                dead = P.dead_cells(A)
                self.assertEqual(set(dead), allowed - used)
                for cell, S in dead.items():
                    self.assertTrue(P.verify_dead_certificate(A, cell, S))
                seen_dead += len(dead)
        self.assertGreater(seen_dead, 0)

    def test_verifier_rejects_bad_certificates(self):
        A = [[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1], [0, 0, 1, 1]]
        self.assertEqual(set(P.dead_cells(A)), {(2, 0), (2, 1)})
        self.assertFalse(P.verify_dead_certificate(A, (2, 0), {2}))
        self.assertFalse(P.verify_dead_certificate(A, (2, 0), {0}))
        self.assertTrue(P.verify_dead_certificate(A, (2, 0), {0, 1}))

    def test_closure(self):
        rng = random.Random(10)
        for _ in range(300):
            n = rng.randint(1, 6)
            b2 = random_board(rng, n, 0.0, 0.7)
            b1 = {c for c in b2 if rng.random() < 0.6}
            c1 = P.closure(b1, n)
            c2 = P.closure(b2, n)
            self.assertTrue(b1 <= c1)
            self.assertTrue(b2 <= c2)
            self.assertEqual(P.closure(c1, n), c1)
            self.assertEqual(P.closure(c2, n), c2)
            self.assertTrue(c1 <= c2)

    def test_closure_no_matching(self):
        board = {(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (1, 2)}
        self.assertEqual(len(P.closure(board, 3)), 9)


class TestBounds(unittest.TestCase):
    def test_bregman_minc(self):
        rng = random.Random(11)
        for _ in range(500):
            n = rng.randint(1, 7)
            p = rng.uniform(0.1, 1.0)
            A = [[1 if rng.random() < p else 0 for _ in range(n)] for _ in range(n)]
            self.assertGreaterEqual(P.bregman_minc_upper(A) * (1 + 1e-12),
                                    P.permanent_ryser(A))

    def test_bregman_equality_block_diagonal(self):
        sizes = (3, 2, 1, 2)
        n = sum(sizes)
        A = [[0] * n for _ in range(n)]
        start = 0
        for s in sizes:
            for i in range(start, start + s):
                for j in range(start, start + s):
                    A[i][j] = 1
            start += s
        exact = P.permanent_ryser(A)
        self.assertEqual(exact, 6 * 2 * 1 * 2)
        self.assertAlmostEqual(P.bregman_minc_upper(A) / exact, 1.0, places=12)

    def test_van_der_waerden(self):
        self.assertEqual(P.van_der_waerden_lower(3), Fraction(6, 27))
        for n in range(1, 8):
            J = [[1] * n for _ in range(n)]
            S = [[Fraction(1, n)] * n for _ in range(n)]
            self.assertEqual(P.permanent_ryser(S), P.van_der_waerden_lower(n))
            self.assertEqual(P.permanent_ryser(J), math.factorial(n))


class TestSinkhorn(unittest.TestCase):
    def test_scaling_identity_and_stochastic(self):
        rng = random.Random(12)
        A = P.random_01_with_matching(rng, 5)
        Pm, r, c = P.sinkhorn(A, 200)
        for i in range(5):
            for j in range(5):
                self.assertAlmostEqual(Pm[i][j], r[i] * A[i][j] * c[j], places=12)
        for i in range(5):
            self.assertAlmostEqual(sum(Pm[i]), 1.0, places=6)

    def test_upper_bound_holds(self):
        rng = random.Random(13)
        for _ in range(150):
            n = rng.randint(2, 7)
            A = P.random_01_with_matching(rng, n)
            exact = P.permanent_ryser(A)
            for iters in (0, 1, 5, 50):
                _, upper = P.sinkhorn_permanent_bounds(A, iters)
                self.assertGreaterEqual(upper * (1 + 1e-9), exact)

    def test_failure_counter_runs(self):
        count = P.sinkhorn_lower_bound_failures(20, 4, 4, 0)
        self.assertTrue(0 <= count <= 20)

    def test_dead_cell_decay(self):
        A = [[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1], [0, 0, 1, 1]]
        self.assertEqual(set(P.dead_cells(A)), {(2, 0), (2, 1)})
        vals = P.sinkhorn_dead_cell_decay(A, (2, 0), [10, 100, 1000])
        self.assertGreater(vals[0], vals[1])
        self.assertGreater(vals[1], vals[2])
        self.assertLess(vals[2], 0.01)


class TestGodsilGutman(unittest.TestCase):
    def test_determinant(self):
        self.assertAlmostEqual(P.determinant_float([[2.0, 1.0], [1.0, 3.0]]), 5.0)
        self.assertAlmostEqual(P.determinant_float([[0.0, 1.0], [1.0, 0.0]]), -1.0)
        self.assertEqual(P.determinant_float([[1.0, 2.0], [2.0, 4.0]]), 0.0)

    def test_estimator_within_five_standard_errors(self):
        rng = random.Random(14)
        for _ in range(8):
            n = rng.randint(2, 6)
            p = rng.uniform(0.4, 1.0)
            A = [[1 if rng.random() < p else 0 for _ in range(n)] for _ in range(n)]
            exact = P.permanent_ryser(A)
            mean, se = P.godsil_gutman(A, 20000, seed=2024)
            self.assertLessEqual(abs(mean - exact), max(5 * se, 1e-9), (A, mean, se, exact))


class TestPoisson(unittest.TestCase):
    def violations(self, boards, n):
        bad = []
        for board in boards:
            if not board:
                continue
            tv = P.exact_tv_to_poisson(board, n)
            sc = P.stein_chen_bound(board, n)
            if sc < tv - 1e-12:
                bad.append((sorted(board), sc, tv))
        return bad

    def test_empty_and_diagonal(self):
        self.assertEqual(P.stein_chen_bound(set(), 5), 0.0)
        self.assertEqual(P.exact_tv_to_poisson(set(), 5), 0.0)
        for n in (5, 10):
            board = {(i, i) for i in range(n)}
            tv = P.exact_tv_to_poisson(board, n)
            self.assertLess(tv, 1.0)
            self.assertGreaterEqual(P.stein_chen_bound(board, n), tv - 1e-12)

    def test_bound_n2_n3(self):
        for n in (2, 3):
            self.assertEqual(self.violations(all_boards(n), n), [])

    def test_bound_n4_all_boards(self):
        self.assertEqual(self.violations(all_boards(4), 4), [])

    def test_bound_random_n5_to_8(self):
        rng = random.Random(15)
        for n in range(5, 9):
            boards = [random_board(rng, n) for _ in range(2000)]
            self.assertEqual(self.violations(boards, n), [])


if __name__ == "__main__":
    unittest.main()
