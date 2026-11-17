"""Zero-dimensional persistence of a layer's weight graph.

A layer with a rows and b columns is read as a graph: one vertex for each row,
one for each column, and one link for each weight, carrying its size |w|.
"Neural persistence" (Rieck and others, ICLR 2019, arXiv 1812.09764) lowers a
level from the largest weight down, adds each link when the level reaches its
size, and records the level at which two separate pieces of the graph join.

Those levels are the weights of a maximum spanning tree of the graph: the
a + b - 1 links of largest total size that join every vertex. This module
computes that tree and the scores built on it. NumPy only.

What the paper leaves open, and how this module treats it:

- The largest weight. The paper divides every weight by the largest absolute
  weight of the whole network. `wmax=None` uses the layer's own largest
  weight. Pass the network's to follow the paper.
- The piece that never joins anything. A connected graph ends as one piece,
  which has no joining level. The paper's main text does not say what it is
  given. `essential="none"` leaves it out. `essential="min"` counts it with the
  layer's smallest weight, and with that choice the upper bound of the paper's
  Theorem 1 is reached by the theorem's own example (see the tests).
  `essential="zero"` counts it with level 0.
- The count in the bound. Theorem 1 has the factor (N - 1)^(1/p), and the
  paper says N is "the number of neurons in the layer". `count="neurons"` uses
  a + b. `count="weights"` uses a * b, which is how the formula reads if the
  symbol is taken as a product.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "tree",
    "tree_kruskal",
    "tree_weights",
    "persistence",
    "bound",
    "normalised",
    "mean_normalised",
    "maxima_share",
    "persistence_from_maxima",
    "sorted_distance",
]


def tree(A):
    """Maximum spanning tree of the complete bipartite graph with sizes A >= 0.

    Returns (rows, cols, sizes): the a + b - 1 links in the order they were
    added. Prim's method on the dense matrix, O((a + b)^2) time, O(a + b) memory
    beyond A.
    """
    A = np.asarray(A)
    if A.ndim != 2 or A.size == 0:
        raise ValueError("A must be a non-empty matrix")
    if (A < 0).any():
        raise ValueError("sizes must be non-negative; pass np.abs(W)")
    a, b = A.shape
    neg = -np.inf
    key_row = np.full(a, neg)
    key_col = A[0].astype(float).copy()
    par_row = np.full(a, -1)
    par_col = np.zeros(b, dtype=int)
    in_row = np.zeros(a, dtype=bool)
    in_col = np.zeros(b, dtype=bool)
    in_row[0] = True
    rows = np.empty(a + b - 1, dtype=int)
    cols = np.empty(a + b - 1, dtype=int)
    sizes = np.empty(a + b - 1)
    for k in range(a + b - 1):
        i = int(np.argmax(key_row))
        j = int(np.argmax(key_col))
        if key_col[j] >= key_row[i]:
            rows[k], cols[k], sizes[k] = par_col[j], j, key_col[j]
            in_col[j] = True
            key_col[j] = neg
            better = (~in_row) & (A[:, j] > key_row)
            key_row[better] = A[better, j]
            par_row[better] = j
        else:
            rows[k], cols[k], sizes[k] = i, par_row[i], key_row[i]
            in_row[i] = True
            key_row[i] = neg
            better = (~in_col) & (A[i] > key_col)
            key_col[better] = A[i, better]
            par_col[better] = i
    return rows, cols, sizes


def tree_kruskal(A):
    """The same tree by Kruskal's method with a union-find. Slow reference for tests."""
    A = np.asarray(A)
    a, b = A.shape
    order = np.argsort(-A, axis=None, kind="stable")
    parent = list(range(a + b))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    rows, cols, sizes = [], [], []
    for idx in order:
        i, j = divmod(int(idx), b)
        ri, rj = find(i), find(a + j)
        if ri != rj:
            parent[ri] = rj
            rows.append(i)
            cols.append(j)
            sizes.append(float(A[i, j]))
            if len(sizes) == a + b - 1:
                break
    return np.array(rows), np.array(cols), np.array(sizes)


def tree_weights(W, wmax=None):
    """Sizes of the tree's links divided by wmax, largest first."""
    A = np.abs(np.asarray(W, dtype=float))
    if wmax is None:
        wmax = A.max()
    if wmax <= 0:
        raise ValueError("the largest weight must be positive")
    _, _, sizes = tree(A)
    return np.sort(sizes / wmax)[::-1]


def _lifetimes(W, wmax, essential):
    A = np.abs(np.asarray(W, dtype=float))
    if wmax is None:
        wmax = A.max()
    life = 1.0 - tree_weights(W, wmax)
    if essential == "none":
        return life
    if essential == "min":
        return np.append(life, 1.0 - A.min() / wmax)
    if essential == "zero":
        return np.append(life, 1.0)
    raise ValueError("essential must be 'none', 'min' or 'zero'")


def persistence(W, p=2, wmax=None, essential="none"):
    """p-norm of (1 - size / wmax) over the tree's links."""
    return float(np.linalg.norm(_lifetimes(W, wmax, essential), ord=p))


def bound(W, p=2, wmax=None, count="neurons"):
    """Upper bound of Theorem 1: (largest - smallest) * (N - 1)^(1/p), in units of wmax."""
    A = np.abs(np.asarray(W, dtype=float))
    if wmax is None:
        wmax = A.max()
    a, b = A.shape
    if count == "neurons":
        n = a + b
    elif count == "weights":
        n = a * b
    else:
        raise ValueError("count must be 'neurons' or 'weights'")
    return float((A.max() - A.min()) / wmax * (n - 1) ** (1.0 / p))


def normalised(W, p=2, wmax=None, essential="none", count="neurons"):
    """Persistence divided by the bound. At most 1 when wmax is the layer's own."""
    return persistence(W, p, wmax, essential) / bound(W, p, wmax, count)


def mean_normalised(layers, p=2, scope="layer", essential="none", count="neurons"):
    """Mean of the normalised persistence over layers.

    scope="layer" divides each layer by its own largest weight.
    scope="network" divides every layer by the largest weight of all, as the paper does.
    """
    layers = [np.asarray(W, dtype=float) for W in layers]
    if scope == "network":
        wmax = max(float(np.abs(W).max()) for W in layers)
    elif scope == "layer":
        wmax = None
    else:
        raise ValueError("scope must be 'layer' or 'network'")
    return float(np.mean([normalised(W, p, wmax, essential, count) for W in layers]))


def maxima_share(W):
    """Share of the tree's links that are the largest entry of their row or of their column."""
    A = np.abs(np.asarray(W, dtype=float))
    r, c, s = tree(A)
    return float(np.mean((s == A.max(axis=1)[r]) | (s == A.max(axis=0)[c])))


def persistence_from_maxima(W, p=2, wmax=None):
    """Persistence computed from each row's and each column's largest entry alone.

    One value per vertex gives a + b values. The tree has one fewer, so the
    largest value, which has lifetime 0 when wmax is the layer's own, is dropped.
    """
    A = np.abs(np.asarray(W, dtype=float))
    if wmax is None:
        wmax = A.max()
    m = np.sort(np.concatenate([A.max(axis=1), A.max(axis=0)]) / wmax)[:-1]
    return float(np.linalg.norm(1.0 - m, ord=p))


def sorted_distance(W1, W2, p=2, relative=False):
    """p-norm of the difference between the two matrices' sorted tree sizes.

    relative=False compares raw sizes |w|. relative=True divides each matrix by
    its own largest weight first. The matrices must have the same shape.

    With p = inf and relative=False this is at most the largest entrywise
    difference of |W1| and |W2| (see the tests). It does not change when the
    rows or the columns of either matrix are reordered, so it is at most that
    difference after the best reordering.
    """
    W1 = np.asarray(W1, dtype=float)
    W2 = np.asarray(W2, dtype=float)
    if W1.shape != W2.shape:
        raise ValueError("shapes differ")
    if relative:
        s1, s2 = tree_weights(W1), tree_weights(W2)
    else:
        s1, s2 = tree_weights(W1, wmax=1.0), tree_weights(W2, wmax=1.0)
    return float(np.linalg.norm(s1 - s2, ord=p))
