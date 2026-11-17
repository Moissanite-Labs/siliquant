"""Figures quoted in notes/persistence-and-flow-check.md. Random matrices only, no model.

    python3 persist/demo.py > persist/demo.out
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import persistlib as P  # noqa: E402

SEED = 20261117
rng = np.random.default_rng(SEED)
print(f"seed {SEED}, numpy {np.__version__}")

draws = [("gauss", lambda s: rng.standard_normal(s)),
         ("student-t3", lambda s: rng.standard_t(3, s))]

print("\n1. Share of the tree's links that are a row's or a column's largest entry")
for shape in [(64, 64), (256, 256), (128, 512), (256, 1024)]:
    for name, draw in draws:
        print(f"   {shape[0]:4d} x {shape[1]:4d}  {name:10}  {P.maxima_share(draw(shape)):.3f}")

print("\n2. Persistence from the maxima alone against the true value, 256 x 1024")
for name, draw in draws:
    W = draw((256, 1024))
    t, m = P.persistence(W), P.persistence_from_maxima(W)
    print(f"   {name:10}  true {t:.4f}  maxima only {m:.4f}  ratio {m / t:.4f}")

print("\n3. The count in the bound of Theorem 1, 650 x 650, gauss")
W = rng.standard_normal((650, 650))
print(f"   persistence {P.persistence(W):.3f}")
print(f"   divided by the bound with N = neurons (1300):  {P.normalised(W, count='neurons'):.4f}")
print(f"   divided by the bound with N = weights (422500): {P.normalised(W, count='weights'):.4f}")

print("\n4. The largest weight of a network, on a layer whose own is 10 times smaller, 64 x 64")
W = rng.standard_normal((64, 64)) * 0.1
wmax = 10 * np.abs(W).max()
print(f"   own largest weight:      score {P.normalised(W):.3f}")
print(f"   the network's:           score {P.normalised(W, wmax=wmax):.3f}  (the bound allows at most 1)")

print("\n5. The piece that never joins: is the bound of Theorem 1 reached by its own example?")
W = np.full((6, 9), 0.2)
W[2, 5] = 1.0
for essential in ("none", "min", "zero"):
    print(f"   essential={essential:5}  persistence {P.persistence(W, essential=essential):.6f}"
          f"  bound {P.bound(W):.6f}")

print("\n6. Rescaling ReLU units: the same function, another score. 64 -> 256 -> 10")
d, h, o = 64, 256, 10
W1 = rng.standard_normal((h, d)) / np.sqrt(d)
W2 = rng.standard_normal((o, h)) / np.sqrt(h)
x = rng.standard_normal((1000, d))


def f(A1, A2):
    return np.maximum(x @ A1.T, 0) @ A2.T


for sigma in [0.0, 0.25, 0.5, 1.0]:
    s = np.exp(sigma * rng.standard_normal(h))
    V1, V2 = W1 * s[:, None], W2 / s[None, :]
    gap = float(np.abs(f(W1, W2) - f(V1, V2)).max())
    move = float(np.sqrt(((V1 - W1) ** 2).sum() + ((V2 - W2) ** 2).sum())
                 / np.sqrt((W1 ** 2).sum() + (W2 ** 2).sum()))
    print(f"   spread {sigma:4.2f}  move over size {move:5.2f}  largest output gap {gap:.1e}"
          f"  layer scores {P.normalised(V1):.4f} {P.normalised(V2):.4f}"
          f"  mean {P.mean_normalised([V1, V2]):.4f}")

print("\n7. Moves that leave the score as it is and change the function")


def rel(A1, A2):
    return float(np.linalg.norm(f(W1, W2) - f(A1, A2)) / np.linalg.norm(f(W1, W2)))


perm = rng.permutation(h)
print(f"   unchanged                   {P.persistence(W1):.6f} {P.persistence(W2):.6f}")
print(f"   sign of layer 1 flipped     {P.persistence(-W1):.6f} {P.persistence(W2):.6f}"
      f"   output change over output size {rel(-W1, W2):.2f}")
print(f"   rows of layer 1 reordered   {P.persistence(W1[perm]):.6f} {P.persistence(W2):.6f}"
      f"   output change over output size {rel(W1[perm], W2):.2f}")

print("\n8. Sorted tree sizes against the size of the change, 2,000 random pairs, 7 x 11")
worst_inf = worst_2 = 0.0
for _ in range(2000):
    A = np.abs(rng.standard_normal((7, 11)))
    B = np.abs(A + rng.standard_normal((7, 11)) * rng.random())
    worst_inf = max(worst_inf, P.sorted_distance(A, B, p=np.inf) / np.abs(A - B).max())
    worst_2 = max(worst_2, P.sorted_distance(A, B, p=2) / np.linalg.norm(A - B))
print(f"   largest ratio, largest gap of sorted sizes over largest entry change: {worst_inf:.4f}")
print(f"   largest ratio, 2-norm of sorted sizes over Frobenius change:          {worst_2:.4f}")

print("\n9. Time for one tree, 2560 x 10240")
W = rng.standard_normal((2560, 10240)).astype(np.float32)
t0 = time.time()
P.tree(np.abs(W))
print(f"   {time.time() - t0:.1f} s")
