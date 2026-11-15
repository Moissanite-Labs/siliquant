#!/usr/bin/env python3
"""Figures quoted in the notes that the tests do not print. Pure Python.

Usage, from the root of a checkout:
  python3 perm/demo.py > perm/demo.out
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import permlib as P


def main():
    print("1. Row and column scaling at a cell that no perfect matching uses")
    allowed = [[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1], [0, 0, 1, 1]]
    dead = sorted(P.dead_cells(allowed))
    print(f"   4 by 4 example; cells on no perfect matching: {dead}")
    rounds = [10, 100, 1000, 10000]
    vals = P.sinkhorn_dead_cell_decay(allowed, (2, 0), rounds)
    for k, v in zip(rounds, vals):
        print(f"   after {k:>5} rounds the entry at (2, 0) is {v:.6g}; 1/(2k) is {1 / (2 * k):.6g}")
    print("   One example only.")
    print()
    print("2. Scaling as a lower bound on the permanent: cases where the computed bound exceeds the exact permanent")
    print("   200 random 0/1 matrices with a perfect matching per cell, seed 1")
    total = bad = 0
    for n in (4, 6, 8):
        row = []
        for iters in (4, 16, 64, 1000):
            f = P.sinkhorn_lower_bound_failures(200, n, iters, 1)
            row.append(f)
            total += 200
            bad += f
        print(f"   n = {n}: failures at 4, 16, 64, 1000 rounds: {row}")
    print(f"   {bad} failures in {total} cases. This is a sample, not a proof.")
    print()
    print("3. Godsil-Gutman estimator on the all-ones matrix: standard deviation of one sample over the permanent")
    for n in (4, 6, 8, 10):
        ones = [[1] * n for _ in range(n)]
        mean, sd, _ = P.godsil_gutman_stats(ones, 20000, 7)
        print(f"   n = {n:>2}: {sd / math.factorial(n):.3f}")
    print()
    print("4. Fixed points of a random permutation against the Poisson law of mean 1")
    print("   n | bound from the coupling formula | exact total variation distance | 2^n / (n+1)!")
    for n in (5, 10, 20):
        board = {(i, i) for i in range(n)}
        print(f"   {n} | {P.stein_chen_bound(board, n):.6f} | {P.exact_tv_to_poisson(board, n):.4g} | {2 ** n / math.factorial(n + 1):.4g}")


if __name__ == "__main__":
    main()
