"""Lattice vector quantisation toolkit (NumPy).

All functions work on batches: x has shape (N, n) and each row is one block.

Lattices are given in an n-dimensional orthonormal coordinate system. A_n
and A_n* live in the zero-sum hyperplane of R^(n+1). We use the Helmert basis
of that hyperplane: row k (k = 1..n) of the (n, n+1) matrix B is
(1, ..., 1, -k, 0, ..., 0) / sqrt(k (k + 1)) with k ones. A point z of the
hyperplane has coordinates y = z @ B.T, and z = y @ B. The public interface
only sees the n-dimensional coordinates y.

Generator matrices G have the basis vectors as rows.
"""

import math

import numpy as np


# ---------------------------------------------------------------------------
# Lattice objects
# ---------------------------------------------------------------------------

class Lattice:
    """A lattice with name, dimension n, generator G, volume and closest()."""

    def __init__(self, name, G, closest_fn):
        self.name = name
        self.G = np.asarray(G, dtype=float)
        self.n = self.G.shape[0]
        self.Ginv = np.linalg.inv(self.G)
        self.volume = abs(np.linalg.det(self.G))
        self._closest = closest_fn

    def closest(self, x):
        """Nearest lattice point to every row of x. Shape (N, n)."""
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.n:
            raise ValueError("x must have shape (N, %d)" % self.n)
        return self._closest(x)

    def coords(self, y):
        """Integer coordinates of lattice points y: y = coords @ G."""
        return np.asarray(y, dtype=float) @ self.Ginv

    def __repr__(self):
        return "Lattice(%s, n=%d, volume=%.6g)" % (self.name, self.n, self.volume)


def _closest_Z(x):
    return np.rint(x)


def _closest_D(x):
    f = np.rint(x)
    odd = (f.sum(axis=1) % 2) != 0
    if np.any(odd):
        err = x - f
        rows = np.nonzero(odd)[0]
        # Coordinate with the largest rounding error, moved the other way.
        j = np.argmax(np.abs(err[rows]), axis=1)
        step = np.where(err[rows, j] >= 0, 1.0, -1.0)
        f[rows, j] += step
    return f


def Zn(n):
    """The integer lattice Z^n."""
    return Lattice("Z%d" % n, np.eye(n), _closest_Z)


def Dn(n):
    """D_n: integer vectors with even coordinate sum. Needs n >= 2.

    The basis is the set of simple roots e_k - e_(k+1) and e_(n-2) + e_(n-1).
    """
    if n < 2:
        raise ValueError("D_n needs n >= 2")
    G = np.zeros((n, n))
    for k in range(n - 1):
        G[k, k] = 1.0
        G[k, k + 1] = -1.0
    G[n - 1, n - 2] = 1.0
    G[n - 1, n - 1] = 1.0
    return Lattice("D%d" % n, G, _closest_D)


def E8():
    """E_8 = D_8 union (D_8 + (1/2, ..., 1/2)), with volume 1."""
    G = np.zeros((8, 8))
    G[0, 0] = 2.0
    for k in range(1, 7):
        G[k, k - 1] = -1.0
        G[k, k] = 1.0
    G[7, :] = 0.5
    half = np.full(8, 0.5)

    def closest(x):
        a = _closest_D(x)
        b = _closest_D(x - half) + half
        da = ((x - a) ** 2).sum(axis=1)
        db = ((x - b) ** 2).sum(axis=1)
        return np.where((db < da)[:, None], b, a)

    return Lattice("E8", G, closest)


def helmert_basis(n):
    """Orthonormal basis (n, n+1) of the zero-sum hyperplane of R^(n+1)."""
    B = np.zeros((n, n + 1))
    for k in range(1, n + 1):
        B[k - 1, :k] = 1.0
        B[k - 1, k] = -float(k)
        B[k - 1] /= math.sqrt(k * (k + 1))
    return B


def _closest_An_ambient(z):
    """Conway-Sloane algorithm for A_n. z has shape (N, n+1), zero sum."""
    m = z.shape[1]
    f = np.rint(z)
    delta = f.sum(axis=1).astype(np.int64)
    d = z - f
    order = np.argsort(d, axis=1)
    rank = np.argsort(order, axis=1)  # rank 0 is the smallest d
    # Sum too high: lower the coordinates that were rounded up the most.
    dec = rank < delta[:, None]
    # Sum too low: raise the coordinates that were rounded down the most.
    inc = rank >= (m + delta)[:, None]
    return f - dec + inc


def _glue_vectors(n):
    """Glue vectors [k], k = 0..n, of A_n*, shape (n+1, n+1)."""
    m = n + 1
    g = np.zeros((m, m))
    for k in range(m):
        g[k, : m - k] = k / m
        g[k, m - k:] = -(m - k) / m
    return g


def An(n):
    """A_n as a lattice of dimension n in Helmert coordinates."""
    B = helmert_basis(n)
    roots = np.zeros((n, n + 1))
    for i in range(n):
        roots[i, i] = 1.0
        roots[i, i + 1] = -1.0
    G = roots @ B.T

    def closest(x):
        return _closest_An_ambient(x @ B) @ B.T

    return Lattice("A%d" % n, G, closest)


def Astar(n):
    """A_n*, the dual of A_n, as a lattice of dimension n in Helmert coordinates.

    closest() tries the n+1 glue cosets of A_n and keeps the nearest.
    The generator is the dual basis of the A_n generator: G* = inv(G_A).T.
    """
    B = helmert_basis(n)
    G_A = An(n).G
    G = np.linalg.inv(G_A).T
    glue = _glue_vectors(n)

    def closest(x):
        z = x @ B
        best = None
        best_d = None
        for g in glue:
            cand = _closest_An_ambient(z - g) + g
            d = ((z - cand) ** 2).sum(axis=1)
            if best is None:
                best, best_d = cand, d
            else:
                better = d < best_d
                best = np.where(better[:, None], cand, best)
                best_d = np.where(better, d, best_d)
        return best @ B.T

    return Lattice("A%d*" % n, G, closest)


# ---------------------------------------------------------------------------
# Second moment
# ---------------------------------------------------------------------------

def mc_normalised_second_moment(lattice, n_points=2_000_000, seed=0,
                                chunk=200_000, cells=4):
    """Monte Carlo estimate of G(L) = E|x - closest(x)|^2 / (n vol^(2/n)).

    x is uniform over the parallelepiped u @ G with u uniform in [0, cells)^n.
    This box is an exact union of cells^n fundamental cells, so the estimate
    has no boundary bias. Returns (estimate, standard error).
    """
    rng = np.random.default_rng(seed)
    n = lattice.n
    total = 0.0
    total_sq = 0.0
    done = 0
    while done < n_points:
        m = min(chunk, n_points - done)
        x = rng.uniform(0.0, cells, size=(m, n)) @ lattice.G
        e = ((x - lattice.closest(x)) ** 2).sum(axis=1)
        total += e.sum()
        total_sq += (e ** 2).sum()
        done += m
    mean = total / done
    var = max(total_sq / done - mean ** 2, 0.0)
    se = math.sqrt(var / done)
    scale = n * lattice.volume ** (2.0 / n)
    return mean / scale, se / scale


def sphere_bound_G(n):
    """Normalised second moment of a ball: Gamma(n/2+1)^(2/n) / ((n+2) pi)."""
    return math.exp((2.0 / n) * math.lgamma(n / 2.0 + 1.0)) / ((n + 2.0) * math.pi)


# ---------------------------------------------------------------------------
# Voronoi codes
# ---------------------------------------------------------------------------

def default_offset(n, seed=1):
    """Fixed generic offset a used to break ties in the Voronoi code.

    A small Gaussian vector (scale 0.01) from a fixed seed. For each coset of
    qL the decoder picks the coset point nearest to a. A tie needs a to lie on
    a bisecting hyperplane, which a generic vector avoids. The tests check
    that the gap to the second-best point is far above rounding error.
    """
    return 0.01 * np.random.default_rng(seed).standard_normal(n)


def _as_offset(a, n):
    return default_offset(n) if a is None else np.asarray(a, dtype=float)


def voronoi_decode(u, lattice, q, a=None):
    """Map integer indices u (N, n), entries 0..q-1, to codewords.

    p = u @ G and the codeword is p - q * closest((p - a) / q). It is the
    point of the coset p + qL that is nearest to a.
    """
    a = _as_offset(a, lattice.n)
    p = np.asarray(u, dtype=float) @ lattice.G
    return p - q * lattice.closest((p - a) / q)


def voronoi_index(y, lattice, q):
    """Index u = round(y G^-1) mod q for lattice points y (N, n)."""
    u = np.rint(np.asarray(y, dtype=float) @ lattice.Ginv).astype(np.int64)
    return np.mod(u, q)


def _roundtrip(w, lattice, q, a):
    y = lattice.closest(w)
    u = voronoi_index(y, lattice, q)
    c = voronoi_decode(u, lattice, q, a)
    ok = np.all(np.abs(c - y) < 1e-7, axis=1)
    return y, u, c, ok


def voronoi_quantize(x, lattice, q, a=None, centre=None, steps=30):
    """Quantise real blocks x to the Voronoi codebook L / qL.

    Returns (reconstruction, indices, overload mask). Blocks whose nearest
    lattice point is not the decoded codeword are shrunk radially. We
    bisect (`steps` halvings) for the largest s in (0, 1] for which the round trip of
    s * x + centre is consistent, and use the codeword found there. The block
    shrinks toward the centre of the codebook, not toward the lattice origin.
    The reconstruction is the codeword minus centre.
    """
    a = _as_offset(a, lattice.n)
    x = np.asarray(x, dtype=float)
    centre = np.zeros(lattice.n) if centre is None else np.asarray(centre, dtype=float)
    w = x + centre
    y, u, c, ok = _roundtrip(w, lattice, q, a)
    over = ~ok
    if np.any(over):
        xo = x[over]
        lo = np.zeros(len(xo))
        hi = np.ones(len(xo))
        for _ in range(steps):
            mid = 0.5 * (lo + hi)
            good = _roundtrip(xo * mid[:, None] + centre, lattice, q, a)[3]
            lo = np.where(good, mid, lo)
            hi = np.where(good, hi, mid)
        _, uo, co, _ = _roundtrip(xo * lo[:, None] + centre, lattice, q, a)
        u = u.copy()
        c = c.copy()
        u[over] = uo
        c[over] = co
    return c - centre, u, over


def all_indices(n, q):
    """All q^n index vectors, shape (q^n, n), first coordinate slowest."""
    return np.indices((q,) * n).reshape(n, -1).T


def enumerate_codebook(lattice, q, a=None, chunk=65536):
    """All q^n codewords, in the order of all_indices(n, q)."""
    n = lattice.n
    if q ** n > 2 ** 20:
        raise ValueError("codebook too large to enumerate")
    u = all_indices(n, q)
    out = np.empty((len(u), n))
    for i in range(0, len(u), chunk):
        out[i:i + chunk] = voronoi_decode(u[i:i + chunk], lattice, q, a)
    return out


def codebook_centroid(lattice, q, a=None, exact_limit=2 ** 20,
                      samples=2 ** 20, seed=0, chunk=65536):
    """Mean of all q^n codewords.

    Exact by enumeration when q^n <= exact_limit, otherwise the mean over
    `samples` uniformly random indices.
    """
    n = lattice.n
    if q ** n <= exact_limit:
        return enumerate_codebook(lattice, q, a, chunk).mean(axis=0)
    rng = np.random.default_rng(seed)
    total = np.zeros(n)
    done = 0
    while done < samples:
        m = min(chunk, samples - done)
        u = rng.integers(0, q, size=(m, n))
        total += voronoi_decode(u, lattice, q, a).sum(axis=0)
        done += m
    return total / samples


def nearest_in_codebook(x, codebook, chunk=256):
    """Exact nearest codeword by brute force, in chunks of rows of x.

    Returns (indices into codebook, codewords).
    """
    x = np.asarray(x, dtype=float)
    cb2 = (codebook ** 2).sum(axis=1)
    idx = np.empty(len(x), dtype=np.int64)
    for i in range(0, len(x), chunk):
        xs = x[i:i + chunk]
        d = cb2[None, :] - 2.0 * xs @ codebook.T
        idx[i:i + chunk] = np.argmin(d, axis=1)
    return idx, codebook[idx]


# ---------------------------------------------------------------------------
# Scalar reference quantisers
# ---------------------------------------------------------------------------

def uniform_midrise(x, bits, step):
    """Mid-rise uniform quantiser with 2^bits levels at (k + 1/2) * step.

    k runs from -2^(bits-1) to 2^(bits-1) - 1. Inputs beyond the ends are
    clipped to the end levels.
    """
    x = np.asarray(x, dtype=float)
    half = 2 ** (bits - 1)
    k = np.clip(np.floor(x / step), -half, half - 1)
    return (k + 0.5) * step


def _phi(t):
    return math.exp(-0.5 * t * t) / math.sqrt(2.0 * math.pi)


def _Phi(t):
    if t == math.inf:
        return 1.0
    if t == -math.inf:
        return 0.0
    return 0.5 * (1.0 + math.erf(t / math.sqrt(2.0)))


def _phi_or_zero(t):
    return 0.0 if math.isinf(t) else _phi(t)


def _t_phi_or_zero(t):
    return 0.0 if math.isinf(t) else t * _phi(t)


def lloyd_max_gaussian(levels, max_iter=200000, tol=1e-13):
    """Lloyd-Max quantiser for the standard normal density.

    Returns (levels as a NumPy array, mean squared error). Cell probabilities
    and centroids use closed forms with math.erf.
    """
    L = int(levels)
    r = list(np.linspace(-2.0, 2.0, L)) if L > 1 else [0.0]
    for _ in range(max_iter):
        edges = [-math.inf] + [0.5 * (r[i] + r[i + 1]) for i in range(L - 1)] + [math.inf]
        new = []
        for i in range(L):
            a, b = edges[i], edges[i + 1]
            p = _Phi(b) - _Phi(a)
            new.append((_phi_or_zero(a) - _phi_or_zero(b)) / p)
        shift = max(abs(new[i] - r[i]) for i in range(L))
        r = new
        if shift < tol:
            break
    edges = [-math.inf] + [0.5 * (r[i] + r[i + 1]) for i in range(L - 1)] + [math.inf]
    mse = 0.0
    for i in range(L):
        a, b = edges[i], edges[i + 1]
        p = _Phi(b) - _Phi(a)
        m1 = _phi_or_zero(a) - _phi_or_zero(b)
        m2 = p + _t_phi_or_zero(a) - _t_phi_or_zero(b)
        mse += m2 - 2.0 * r[i] * m1 + r[i] ** 2 * p
    return np.array(r), mse


# ---------------------------------------------------------------------------
# Dual scaling
# ---------------------------------------------------------------------------

def _imbalance(M):
    s = np.concatenate([M.std(axis=1), M.std(axis=0)])
    return s.max() / s.min()


def dual_scale(W, iters=16):
    """Sinkhorn-style scaling by standard deviations.

    Each iteration divides every row by its standard deviation, then every
    column by its standard deviation. The scales are accumulated, so that
    W = s[:, None] * Wn * t[None, :]. Returns
    (s, t, Wn, imbalance before, imbalance after), where imbalance is the
    largest over the smallest standard deviation among all rows and columns.
    """
    W = np.asarray(W, dtype=float)
    m, n = W.shape
    s = np.ones(m)
    t = np.ones(n)
    Wn = W.copy()
    before = _imbalance(W)
    for _ in range(iters):
        rs = Wn.std(axis=1)
        Wn = Wn / rs[:, None]
        s = s * rs
        cs = Wn.std(axis=0)
        Wn = Wn / cs[None, :]
        t = t * cs
    return s, t, Wn, before, _imbalance(Wn)
