"""Shape toolkit: probe a matrix of weights, map it, and predict the error of a cut codebook.

Issue #10 proposes a process in four stages: probe, map, variables, formula. This file is
that process as code, for weight error under the rule that clips nothing.

A codebook here is every point y of a lattice (the integers Z^8 or E8) inside a region
    sum over the 8 entries of T[2 |y_i|] <= R,
where T is a table of integers for the shape. For the shape p the table is
    T[k] = floor((k / 2)^p * 2^S),
so the region is the ball of the p-norm: a cross-polytope for p = 1, a ball for p = 2, close
to a cube for large p, and a star with arms along the axes for p under 1. Every test of
whether a point is in a codebook is integer arithmetic, and so is every count of a codebook.
For p = 1 and p = 2 the tables are exact and the codebooks are those of the record
region-shapes-no-clip-rule-2026-11-17.

The rule is that record's: the scale of a group of weights is the smallest at which every
block of 8 rounds to a point of the codebook.

The formula (fine cells, nothing clipped), for a lattice with cell constant G, a shape K
with unit ball of volume vol(K), a codebook of N points and groups of g weights:

    error / mean square of the weights = G * Phi_K(g) * N^(-2/8)
    Phi_K(g) = vol(K)^(2/8) * mean over groups of (largest K-norm of a block in the group)^2
               / mean square of the weights.

G is the cell part, Phi the region part and N the rate. Phi is a number computed from the
weights alone, with no rounding. For p = 1, 2 and for the cube it is a ratio of whole numbers
times a known constant, and exact_terms() returns those whole numbers.

Needs NumPy. Tests: python3 -m unittest shape/test_shapelib.py
"""
from __future__ import annotations

import hashlib
import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lattice"))
import lattices as L  # noqa: E402

N_DIM = 8
S_BITS = 20                       # binary places kept in a table of sizes
G_CELL = {"Z8": 1.0 / 12.0, "E8": 929.0 / 12960.0}
PACKING_RADIUS = {"Z8": 0.5, "E8": math.sqrt(0.5)}
BISECT_STEPS = 14
SHRINK_STEPS = 12
_SENTINEL = 1 << 58               # a size no codebook reaches: eight of them still fit in 64 bits
_E8 = L.E8()


def db(x):
    return 10.0 * math.log10(x)


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------

def iroot(x, b):
    """floor of the b-th root of the whole number x >= 0, exactly."""
    if x < 0 or b < 1:
        raise ValueError("iroot needs x >= 0 and b >= 1")
    if b == 1 or x < 2:
        return x
    if b == 2:
        return math.isqrt(x)
    r = 1 << -(-x.bit_length() // b)          # a power of two at or above the root
    while True:
        s = ((b - 1) * r + x // r ** (b - 1)) // b
        if s >= r:
            return r
        r = s


class Shape:
    """The ball of the p-norm in 8 dimensions, p a positive fraction."""

    def __init__(self, p):
        self.p = Fraction(p)
        if self.p <= 0:
            raise ValueError("p must be positive")
        self.a, self.b = self.p.numerator, self.p.denominator
        if self.b * S_BITS < self.a:
            raise ValueError("p is too large for the table")
        self.name = "p" + (str(self.a) if self.b == 1 else f"{self.a}/{self.b}")
        self._table = [0]

    def size_of_index(self, k):
        """T[k] = floor((k / 2)^p * 2^S) for the entry k / 2, exactly."""
        return iroot(k ** self.a << (self.b * S_BITS - self.a), self.b)

    def table(self, limit):
        """The sizes T[0..K] up to the first above limit, then a sentinel, as 64-bit integers."""
        t = self._table
        while t[-1] <= limit:
            t.append(self.size_of_index(len(t)))
        k = next(i for i, v in enumerate(t) if v > limit)
        return np.array(t[:k] + [_SENTINEL], dtype=np.int64)

    def unit_volume(self, n=N_DIM):
        """Volume of the unit ball of the p-norm in n dimensions: (2 Gamma(1 + 1/p))^n / Gamma(1 + n/p)."""
        p = float(self.p)
        return math.exp(n * (math.log(2.0) + math.lgamma(1.0 + 1.0 / p)) - math.lgamma(1.0 + n / p))

    def norm(self, X):
        """The p-norm of every row of X, in 64-bit floats."""
        p = float(self.p)
        A = np.abs(np.asarray(X, dtype=np.float64))
        if self.p == 1:
            return A.sum(axis=1)
        if self.p == 2:
            return np.sqrt((A * A).sum(axis=1))
        return (A ** p).sum(axis=1) ** (1.0 / p)

    def radius_as_norm(self, R):
        """The p-norm that the size R stands for."""
        x = R / float(1 << S_BITS)
        if self.p == 1:
            return x
        if self.p == 2:
            return math.sqrt(x)
        return x ** (1.0 / float(self.p))


def sizes(shape_table, Y):
    """The size of every row of Y, whose entries are whole numbers or halves. Integer arithmetic."""
    k = np.rint(2.0 * np.abs(Y)).astype(np.int64)
    np.minimum(k, len(shape_table) - 1, out=k)
    return shape_table[k].sum(axis=1)


# ---------------------------------------------------------------------------
# Exact counts of the points of a lattice inside a shape
# ---------------------------------------------------------------------------

def _reduce(key, ways):
    if not len(key):
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    order = np.argsort(key, kind="stable")
    key, ways = key[order], ways[order]
    first = np.concatenate(([True], key[1:] != key[:-1]))
    return key[first], np.add.reduceat(ways, np.nonzero(first)[0])


def _combine(A, B, limit, chunk=1 << 23):
    """All sums of one entry of A and one of B with total size <= limit. Lists are (2 size + parity, ways)."""
    ka, wa = A
    kb, wb = B
    empty = (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
    if not len(ka) or not len(kb):
        return empty
    sb = kb >> 1                                                       # B is sorted by size first
    reach = np.searchsorted(sb, limit - (ka >> 1), side="right")
    done, keys, ways, held = [], [], [], 0

    def flush():
        nonlocal keys, ways, held
        if keys:
            done.append(_reduce(np.concatenate(keys), np.concatenate(ways)))
        keys, ways, held = [], [], 0

    for i in range(len(ka)):
        r = int(reach[i])
        if not r:
            continue
        if held and held + r > chunk:
            flush()
        keys.append((((ka[i] >> 1) + sb[:r]) << 1) | ((ka[i] & 1) ^ (kb[:r] & 1)))
        ways.append(wa[i] * wb[:r])
        held += r
    flush()
    if not done:
        return empty
    return _reduce(np.concatenate([d[0] for d in done]), np.concatenate([d[1] for d in done]))


def _half_lists(shape, limit):
    """For the whole-number entries and for the halves: every vector of 4 entries of size <= limit.

    Returns two lists of (2 size + parity, ways). The parity is that of the sum of the entries for
    whole numbers, and of the sum of their floors for halves, so a vector of 8 is in E8 when its
    two halves have equal parity.
    """
    t = shape.table(limit)
    out = []
    for coset in (0, 1):
        key, ways = [], []
        for k in range(coset, len(t) - 1, 2):
            size = int(t[k])
            if coset == 0:
                key.append((size << 1) | ((k // 2) & 1))
                ways.append(1 if k == 0 else 2)
            else:
                m = k // 2
                key += [(size << 1) | (m & 1), (size << 1) | ((m + 1) & 1)]
                ways += [1, 1]
        one = _reduce(np.array(key, dtype=np.int64), np.array(ways, dtype=np.int64)) if key else (np.zeros(0, np.int64), np.zeros(0, np.int64))
        two = _combine(one, one, limit)
        out.append(_combine(two, two, limit))
    return out


class Counter:
    """Counts of the points of Z8 and of E8 inside a shape, for every size up to a limit."""

    def __init__(self, shape, limit):
        self.shape, self.limit = shape, int(limit)
        self.cosets = []
        for key, ways in _half_lists(shape, self.limit):
            parts = []
            for par in (0, 1):
                sel = (key & 1) == par
                parts.append((key[sel] >> 1, ways[sel], np.cumsum(ways[sel])))
            both_s, both_w = _reduce(key >> 1, ways)
            self.cosets.append({"par": parts, "all": (both_s, both_w, np.cumsum(both_w))})

    @staticmethod
    def _pairs(a, b, R):
        sa, wa, _ = a
        sb, _, cb = b
        if not len(sa) or not len(sb):
            return 0
        j = np.searchsorted(sb, R - sa, side="right")
        ok = j > 0
        return int(np.dot(wa[ok], cb[j[ok] - 1]))

    def count(self, lattice, R):
        """The number of points of the lattice of size <= R, exactly."""
        if R > self.limit:
            raise ValueError("R is above the limit this counter was built for")
        if lattice == "Z8":
            a = self.cosets[0]["all"]
            return self._pairs(a, a, R)
        if lattice == "E8":
            return sum(self._pairs(c["par"][par], c["par"][par], R) for c in self.cosets for par in (0, 1))
        raise ValueError("lattice must be Z8 or E8")

    def largest_size(self, lattice, R):
        """The largest size of a point of the lattice that is <= R."""
        best = 0
        if lattice == "Z8":
            pairs = [(self.cosets[0]["all"], self.cosets[0]["all"])]
        else:
            pairs = [(c["par"][par], c["par"][par]) for c in self.cosets for par in (0, 1)]
        for a, b in pairs:
            sa, sb = a[0], b[0]
            if not len(sa) or not len(sb):
                continue
            j = np.searchsorted(sb, R - sa, side="right")
            ok = j > 0
            if np.any(ok):
                best = max(best, int((sa[ok] + sb[j[ok] - 1]).max()))
        return best


class Codebook:
    """Every point of a lattice inside a shape, with the largest radius at which it has no more than cap points."""

    _cache = {}

    def __new__(cls, lattice, shape, cap):
        key = (lattice, shape.p, int(cap))
        if key in cls._cache:
            return cls._cache[key]
        self = super().__new__(cls)
        cap = int(cap)
        # a first radius from the volume, then a counter that reaches a little past the cap
        guess = (cap / shape.unit_volume()) ** (float(shape.p) / N_DIM) * (1 << S_BITS)
        limit = int(guess * 1.05) + (16 << S_BITS)
        while True:
            counter = Counter(shape, limit)
            if counter.count(lattice, limit) > cap:
                break
            limit = int(limit * 1.2)
        lo, hi = 0, limit                      # count(lo) <= cap < count(hi)
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if counter.count(lattice, mid) <= cap:
                lo = mid
            else:
                hi = mid
        self.lattice, self.shape, self.cap = lattice, shape, cap
        self.R = counter.largest_size(lattice, lo)
        self.N = counter.count(lattice, self.R)
        assert self.N == counter.count(lattice, lo) and self.N <= cap
        self.table = shape.table(self.R)
        self.bits = math.log2(self.N) / N_DIM
        self.G = G_CELL[lattice]
        # the radius of the ball of the p-norm that has the volume of the N cells
        self.r_eff = (self.N / shape.unit_volume()) ** (1.0 / N_DIM)
        cls._cache[key] = self
        return self

    def closest(self, X):
        return np.rint(X) if self.lattice == "Z8" else _E8.closest(X)

    def inside(self, Y):
        return sizes(self.table, Y) <= self.R

    def describe(self):
        return {"lattice": self.lattice, "shape": self.shape.name, "cap": self.cap, "R": self.R,
                "radius_as_norm": self.shape.radius_as_norm(self.R), "codewords": self.N, "bits": self.bits}


# ---------------------------------------------------------------------------
# Rounding under the rule that clips nothing
# ---------------------------------------------------------------------------

def block_scales(X, code):
    """For every block the smallest scale at which it rounds to a point of the codebook, by bisection.

    For p = 1 and p = 2 the bounds and the steps are those of the record
    region-shapes-no-clip-rule-2026-11-17, so its numbers are reproduced. For other p the upper
    bound is checked and widened where it does not hold.
    """
    shape = code.shape
    g = np.maximum(shape.norm(X), 1e-30)
    r = shape.radius_as_norm(code.R)
    margin = {1: 4.0, 2: math.sqrt(2.0)}.get(shape.p, 0.5 * N_DIM ** (1.0 / float(shape.p)))
    lo = g / (r + margin) * 0.99
    hi = g / max(r - margin, 0.25) * 1.01
    if shape.p not in (1, 2):
        for _ in range(60):
            bad = ~code.inside(code.closest(X / hi[:, None]))
            if not np.any(bad):
                break
            hi = np.where(bad, hi * 2.0, hi)
        else:
            raise RuntimeError("no scale found at which a block is inside the codebook")
    for _ in range(BISECT_STEPS):
        mid = np.sqrt(lo * hi)
        ok = code.inside(code.closest(X / mid[:, None]))
        hi = np.where(ok, mid, hi)
        lo = np.where(ok, lo, mid)
    return hi


def round_groups(W, code, group, tb, keep=False):
    """Round a matrix with one scale per group. Returns (relative error, scales of the groups, share of blocks shrunk).

    tb holds the scale of every block, from block_scales. The rounded matrix is stored as 16-bit
    floats before its error is taken, as in the earlier records. With keep, the rounded matrix
    is returned as a fourth value.
    """
    m, n = W.shape
    size = n if group == "R" else int(group)
    X = W.reshape(-1, N_DIM)
    tg = tb.reshape(-1, size // N_DIM).max(axis=1)
    t = np.repeat(tg, size // N_DIM)
    Y = code.closest(X / t[:, None])
    out = ~code.inside(Y)
    if np.any(out):
        xo = X[out] / t[out, None]
        lo, hi = np.zeros(len(xo)), np.ones(len(xo))
        for _ in range(SHRINK_STEPS):
            mid = 0.5 * (lo + hi)
            good = code.inside(code.closest(xo * mid[:, None]))
            lo = np.where(good, mid, lo)
            hi = np.where(good, hi, mid)
        Y[out] = code.closest(xo * lo[:, None])
    Wh = (Y * t[:, None]).reshape(m, n).astype(np.float16).astype(np.float64)
    rel = float(np.sum((W - Wh) ** 2) / np.sum(W ** 2))
    if keep:
        return rel, tg, float(np.mean(out)), Wh
    return rel, tg, float(np.mean(out))


def scalar_rule(W, q, group, keep=False):
    """The scalar largest-entry rule of harness/tq.py (maxscale_round), in NumPy and 32-bit floats.

    Returns (relative error, bits per weight). The same operations in the same precision as the
    harness, so its numbers are reproduced without torch. With keep, the rounded matrix is
    returned as a third value.
    """
    W32 = np.ascontiguousarray(W, dtype=np.float32)
    m, n = W32.shape
    size = n if group == "R" else int(group)
    X = W32.reshape(-1, size)
    j = np.abs(X).argmax(axis=1)[:, None]
    d = np.take_along_axis(X, j, axis=1) / np.float32(-q / 2.0)
    safe = np.where(d == 0, np.float32(1.0), d)
    Rr = np.clip(np.rint(X / safe), -q // 2, q // 2 - 1).astype(np.float32) * d
    Wh = Rr.reshape(m, n).astype(np.float16).astype(np.float64)
    W64 = W32.astype(np.float64)
    rel = float(np.sum((W64 - Wh) ** 2) / np.sum(W64 ** 2))
    if keep:
        return rel, math.log2(q) + 16.0 / size, Wh
    return rel, math.log2(q) + 16.0 / size


def spectral_ratio(E):
    """The largest singular value of E over what a matrix of independent entries of the same total square would have.

    For independent entries of equal variance the largest singular value is close to
    sqrt(variance) * (sqrt(rows) + sqrt(columns)). 1 means the error looks like such noise, and a
    larger number means it is gathered along some direction.
    """
    m, n = E.shape
    fro = math.sqrt(float(np.sum(E ** 2)))
    if fro == 0:
        return 0.0
    return float(np.linalg.norm(E, 2)) * math.sqrt(m * n) / (fro * (math.sqrt(m) + math.sqrt(n)))


# ---------------------------------------------------------------------------
# The probe and the formula
# ---------------------------------------------------------------------------

def phi(W, shape, group):
    """The region part Phi_K(g) of a matrix, from the weights alone, in 64-bit floats."""
    W = np.asarray(W, dtype=np.float64)
    size = W.shape[1] if group == "R" else int(group)
    nb = shape.norm(W.reshape(-1, N_DIM))
    top = nb.reshape(-1, size // N_DIM).max(axis=1)
    return shape.unit_volume() ** (2.0 / N_DIM) * float(np.mean(top ** 2)) / float(np.mean(W ** 2))


def phi_cube(W, group):
    """Phi for the scalar rule: the region is a cube and the norm is the largest entry."""
    W = np.asarray(W, dtype=np.float64)
    size = W.shape[1] if group == "R" else int(group)
    top = np.abs(W).reshape(-1, size).max(axis=1)
    return 4.0 * float(np.mean(top ** 2)) / float(np.mean(W ** 2))


def predict(W, code, group, level=3):
    """The formula's error for a cut codebook on a matrix, as a share of the squared weights. Nothing is rounded.

    level 1: G * Phi * N^(-1/4), with the count N of the codebook.
    level 2: and the block on the wall. The block that sets a group's scale lies on a wall of
             its cell, where the mean squared error is (n + 2) / n times that of the cell.
    level 3: and the cell at zero. A weight (integers) or a block (E8) small enough to lie
             inside the largest ball in the cell at zero rounds to zero, and its error is its
             own square and not the cell's.
    """
    W = np.asarray(W, dtype=np.float64)
    size = W.shape[1] if group == "R" else int(group)
    X = W.reshape(-1, N_DIM)
    per_group = size // N_DIM
    s = code.shape.norm(X).reshape(-1, per_group).max(axis=1) / code.r_eff      # the scale of every group
    total = float(np.sum(W ** 2))
    cell = code.G * s ** 2                                                       # error of one weight in a fine cell
    if level == 1:
        return float(np.sum(cell) * size) / total
    wall = 2.0 * cell                                                            # one block per group has (n + 2) / n of it
    if level == 2:
        return float(np.sum(cell * size + wall)) / total
    sw = np.repeat(s, per_group)
    if code.lattice == "Z8":
        w2 = X ** 2
        small = np.abs(X) < 0.5 * sw[:, None]
        err = np.where(small, w2, (code.G * sw ** 2)[:, None]).sum()
    else:
        b2 = (X ** 2).sum(axis=1)
        small = b2 < (PACKING_RADIUS["E8"] * sw) ** 2
        err = np.where(small, b2, N_DIM * code.G * sw ** 2).sum()
    return float(err + np.sum(wall)) / total


def predict_scalar(W, q, group, level=3):
    """The formula's error for the scalar largest-entry rule at q levels."""
    W = np.asarray(W, dtype=np.float64)
    size = W.shape[1] if group == "R" else int(group)
    X = np.abs(W).reshape(-1, size)
    d = X.max(axis=1) / (q / 2.0)
    total = float(np.sum(W ** 2))
    if level == 1:
        return float(np.sum(d ** 2) * size / 12.0) / total
    if level == 2:
        return float(np.sum(d ** 2) * (size - 1) / 12.0) / total                 # the largest entry lands on a level
    small = X < 0.5 * d[:, None]
    err = np.where(small, X ** 2, (d ** 2 / 12.0)[:, None]).sum() - np.sum(d ** 2) / 12.0
    return float(err) / total


# ---------------------------------------------------------------------------
# Exact terms: the receipt
# ---------------------------------------------------------------------------

def to_integers(W):
    """A matrix of floats as whole numbers times a power of two: W = Z * 2^e, exactly. Z is an array of Python integers."""
    W = np.asarray(W)
    if W.dtype not in (np.float16, np.float32, np.float64):
        raise ValueError("W must be an array of floats")
    if not np.all(np.isfinite(W)):
        raise ValueError("W holds a value that is not finite")
    f, e = np.frexp(W.astype(np.float64))
    mant = np.rint(f * float(1 << 53)).astype(np.int64)            # exact: a double has 53 bits
    e = e.astype(np.int64) - 53
    nz = mant != 0
    if not np.any(nz):
        return np.zeros(W.shape, dtype=object), 0
    low = (mant & -mant).astype(np.float64)                        # the lowest set bit of every mantissa
    tz = np.where(nz, np.log2(np.where(nz, low, 1.0)), 0).astype(np.int64)
    mant = mant >> tz
    e = e + tz
    e_min = int(e[nz].min())
    shift = np.where(nz, e - e_min, 0)
    Z = mant.astype(object) * (np.int64(1) << np.minimum(shift, 62)).astype(object)
    big = shift > 62
    if np.any(big):
        Z[big] = [int(a) << int(b) for a, b in zip(mant[big], shift[big])]
    return Z, e_min


def exact_terms(W, group):
    """Whole numbers from which Phi for the cube, the ball and the cross-polytope follow exactly.

    With W = Z * 2^e:
      mean square of the weights        = sum_sq / count * 4^e
      Phi_cube(g)  = 4 * (top_cube / groups) / (sum_sq / count)
      Phi_ball(g)  = (pi^4 / 24)^(1/4) * (top_ball / groups) / (sum_sq / count)
      Phi_cross(g) = (2^8 / 8!)^(1/4) * (top_cross / groups) / (sum_sq / count)
    The power of two cancels in every ratio.
    """
    Z, e = to_integers(W)
    m, n = Z.shape
    size = n if group == "R" else int(group)
    A = np.abs(Z)
    blocks = A.reshape(-1, N_DIM)
    sq = (blocks * blocks).sum(axis=1)
    l1 = blocks.sum(axis=1)
    per = size // N_DIM
    return {"count": m * n, "groups": (m * n) // size, "exponent": e,
            "sum_sq": int(sq.sum()),
            "top_cube": int((A.reshape(-1, size).max(axis=1) ** 2).sum()),
            "top_ball": int(sq.reshape(-1, per).max(axis=1).sum()),
            "top_cross": int((l1.reshape(-1, per).max(axis=1) ** 2).sum())}


def phi_from_exact(terms):
    """Phi for the cube, the ball and the cross-polytope as (fraction, constant) pairs: Phi = constant * fraction."""
    mean_sq = Fraction(terms["sum_sq"], terms["count"])
    out = {}
    for name, const in (("cube", 4.0), ("ball", (math.pi ** 4 / 24.0) ** 0.25), ("cross", (256.0 / math.factorial(8)) ** 0.25)):
        out[name] = (Fraction(terms["top_" + name], terms["groups"]) / mean_sq, const)
    return out


def matrix_digest(W):
    """SHA-256 of the shape, the type and the bytes of a matrix, so a receipt names what it measured."""
    W = np.ascontiguousarray(W)
    h = hashlib.sha256()
    h.update(f"{W.dtype.str}|{W.shape[0]}x{W.shape[1]}|".encode())
    h.update(W.tobytes())
    return h.hexdigest()
