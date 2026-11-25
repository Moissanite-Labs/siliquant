"""Lattices made from extended Hamming codes by Construction A, and what a ball arm needs of them. NumPy only.

For m = 3, 4, 5 the extended Hamming code of length n = 2^m has n - m - 1 information bits and minimum
distance 4. Its Construction A lattice is

    A_n = C + 2 Z^n,

the whole vectors whose remainders on division by 2 form a word of the code. A cell has volume 2^(m+1).
For m = 3 the code is the Reed-Muller code RM(1,3), and A_8 over the root of 2 is E8 turned by the
Hadamard matrix of order 8 over the root of 8 (the turned lattice of e8-hadamard-real-weights-2026-11-22).

The parity checks of the code: coordinate i has the check vector h_i = i + 2^m, the m bits of its
number and a final bit for the overall parity. A word c is in the code when the exclusive-or of h_i
over the i with c_i = 1 is zero.

Nearest point: every coordinate can take its nearest even or its nearest odd whole number; which one is
the parity bit of that coordinate. The bits must form a word of the code. A dynamic program over the
2^(m+1) values of the partial syndrome finds the cheapest choice exactly (Viterbi on the code's syndrome
trellis): n steps, each a minimum of two moves per state.

Counting points: a point over a word of weight w has w odd coordinates and n - w even ones, so the number
of points of each squared length is sum_w A_w [q^r] O(q)^w E(q)^(n-w), with O and E the theta series of
the odd and the even whole numbers, and A_w the weight enumerator of the code, from the MacWilliams
identity and the dual code RM(1, m), whose weights are 0, n/2 (2n - 2 words) and n.
"""
from __future__ import annotations

import math
from math import comb

import numpy as np

CHUNK = 1 << 17          # blocks decoded at a time


def checks(m):
    """The check vector of each coordinate, as whole numbers of m + 1 bits."""
    n = 1 << m
    return np.array([i | n for i in range(n)], dtype=np.int64)


def weight_enumerator(m):
    """A_w for w = 0..n of the extended Hamming code of length 2^m, exactly, by MacWilliams from RM(1, m)."""
    n = 1 << m
    dual = {0: 1, n // 2: 2 * n - 2, n: 1}
    size = 2 * n                                    # words of the dual

    def kraw(k, x):                                 # Krawtchouk K_k(x) for length n
        return sum((-1) ** j * comb(x, j) * comb(n - x, k - j) for j in range(k + 1))

    A = []
    for w in range(n + 1):
        num = sum(b * kraw(w, j) for j, b in dual.items())
        if num % size:
            raise ArithmeticError("MacWilliams sum not divisible")
        A.append(num // size)
    if sum(A) != 1 << (n - m - 1):
        raise ArithmeticError("weight enumerator does not count the code")
    return A


def poly_mul(a, b, limit):
    out = [0] * (min(limit, len(a) + len(b) - 2) + 1)
    for i, x in enumerate(a):
        if x == 0 or i > limit:
            continue
        for j, y in enumerate(b[: limit - i + 1]):
            if y:
                out[i + j] += x * y
    return out


def poly_pow(a, k, limit):
    out = [1]
    base = a[: limit + 1]
    while k:
        if k & 1:
            out = poly_mul(out, base, limit)
        k >>= 1
        if k:
            base = poly_mul(base, base, limit)
    return out


def shells(m, limit):
    """Number of points of A_n of each squared length 0..limit, exactly."""
    n = 1 << m
    O = [0] * (limit + 1)
    E = [0] * (limit + 1)
    k = 0
    while k * k <= limit:
        (E if k % 2 == 0 else O)[k * k] += 1 if k == 0 else 2
        k += 1
    A = weight_enumerator(m)
    total = [0] * (limit + 1)
    Opow = [[1]]
    for _ in range(n):
        Opow.append(poly_mul(Opow[-1], O, limit))
    Epow = [[1]]
    for _ in range(n):
        Epow.append(poly_mul(Epow[-1], E, limit))
    for w, a in enumerate(A):
        if a:
            p = poly_mul(Opow[w], Epow[n - w], limit)
            for r, c in enumerate(p):
                total[r] += a * c
    return total


def ball_bound(m, dims, bits, _cache={}):
    """The largest squared radius at which the ball over `dims` weights (a multiple of 2^m) holds no more
    than 2^bits points of A_n repeated: (bound, number of points)."""
    key = (m, dims, bits)
    if key in _cache:
        return _cache[key]
    n = 1 << m
    if dims % n:
        raise ValueError("the ball does not hold a whole number of blocks")
    cap = 1 << bits
    vol = 2.0 ** ((m + 1) * dims / n)              # volume of a cell of the repeated lattice
    r_unit = math.exp(math.lgamma(dims / 2.0 + 1.0) / dims) / math.sqrt(math.pi)
    limit = int(1.1 * (r_unit * (2.0 ** bits * vol) ** (1.0 / dims)) ** 2) + 4 * dims
    while True:
        total = poly_pow(shells(m, limit), dims // n, limit)
        run, best = 0, None
        for r, c in enumerate(total):
            run += c
            if run > cap:
                break
            best = (r, run)
        if run > cap:
            break
        limit = int(limit * 1.3) + 8
    _cache[key] = best
    return best


_lib = None


def _library():
    """hamming_dp.c compiled to a shared library in a directory of this user's alone (shape/ballpack.py's
    rule), named by the hash of its source; None where no C compiler is here."""
    global _lib
    if _lib is None:
        import ctypes
        import hashlib
        import os
        import subprocess
        import sys
        from pathlib import Path
        here = Path(__file__).resolve().parent
        sys.path.insert(0, str(here.parents[1] / "shape"))
        import ballpack
        src = (here / "hamming_dp.c").read_bytes()
        path = ballpack._build_dir() / f"hamming_dp-{hashlib.sha256(src).hexdigest()[:16]}.so"
        try:
            if not (path.exists() and ballpack._private(path)):
                part = path.with_suffix(f".{os.getpid()}.part")
                subprocess.run(["cc", "-O2", "-ffp-contract=off", "-shared", "-fPIC", "-o", str(part),
                                str(here / "hamming_dp.c"), "-lm"], check=True)
                os.chmod(part, 0o700)
                os.replace(part, path)
            lib = ctypes.CDLL(str(path))
            lib.hamming_closest.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
            lib.hamming_closest.restype = ctypes.c_int
            _lib = lib
        except (OSError, subprocess.CalledProcessError):
            _lib = False
    return _lib or None


def closest(X, m, use_c=True):
    """Nearest point of A_n (n = 2^m) to every block of n consecutive entries of each row of X, exactly.

    Ties go to the even whole number at each coordinate, then to the lower syndrome path; they have
    measure zero for real weights. With use_c the same program in C (hamming_dp.c) is used where it can be
    built; test_hamming.py holds the two to each other bit for bit."""
    n = 1 << m
    X = np.asarray(X, dtype=np.float64)
    shape = X.shape
    lib = _library() if use_c else None
    if lib is not None:
        src = np.ascontiguousarray(X.reshape(-1, n))
        dst = np.empty_like(src)
        if lib.hamming_closest(src.ctypes.data, dst.ctypes.data, len(src), m) != 0:
            raise ValueError("m must be 3, 4 or 5")
        return dst.reshape(shape)
    B = X.reshape(-1, n)
    out = np.empty_like(B)
    h = checks(m)
    states = 2 * n
    idx = np.arange(states)
    for s0 in range(0, len(B), CHUNK):
        x = B[s0:s0 + CHUNK]
        M = len(x)
        e = 2.0 * np.rint(x / 2.0)                  # nearest even whole number
        o = np.where(x >= e, e + 1.0, e - 1.0)      # nearest odd whole number
        de = (x - e) ** 2
        do = (x - o) ** 2
        S = np.full((M, states), np.inf)
        S[:, 0] = 0.0
        take = np.empty((n, M, states), dtype=bool)
        for i in range(n):
            a = S + de[:, i:i + 1]
            b = S[:, idx ^ h[i]] + do[:, i:i + 1]
            t = b < a
            take[i] = t
            S = np.where(t, b, a)
        s = np.zeros(M, dtype=np.int64)
        rows = np.arange(M)
        bit = np.empty((M, n), dtype=bool)
        for i in range(n - 1, -1, -1):
            t = take[i][rows, s]
            bit[:, i] = t
            s = np.where(t, s ^ h[i], s)
        out[s0:s0 + CHUNK] = np.where(bit, o, e)
    return out.reshape(shape)


def covering_sq(m, dims):
    """A bound on the squared distance from any point to A_n repeated over `dims` weights.

    The nearest whole number is within 1/2 at each coordinate; the code's covering radius is 2, so at most
    two coordinates of a block take their second-nearest whole number, which is within 1."""
    n = 1 << m
    return dims / 4.0 + 1.5 * (dims // n)
