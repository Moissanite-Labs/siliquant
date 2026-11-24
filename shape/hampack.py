"""The packed format of ballpack.py with a lattice from an extended Hamming code in place of the integers.

ballpack.py stores a block of n weights as one byte of scale code and the number of a point of Z^n in
a ball. This module stores the point of A_L repeated n / L times, A_L = C_L + 2Z^L with C_L the
extended Hamming code of length L = 8, 16 or 32 (A_8 is E8 turned by H8/sqrt(8), times sqrt(2)). The
file is laid out as ballpack's, with its own magic and L in the header. With n = 32, bits = 136 and
one byte of scale a block still takes 18 bytes: 4.5 bits a weight.

The difference from ballpack, all of it:
  - the nearest point of A_L (hp_closest in hampack.c, the record hamming-lattices-real-weights-
    2026-11-24's dynamic program) in place of rounding each entry;
  - the bisection's margin is the root of A_L's covering bound (n/4 + 1.5 n/L) in place of sqrt(n)/2;
  - the numbering reads the entries in order, as ballpack's, with a state: the partial syndrome of the
    current block. Its table of counts is built here in whole numbers, and ref_rank and ref_unrank walk
    it in Python, so test_hampack.py can hold the C coder to them and both to a full enumeration.

Tests: python3 -m unittest shape/test_hampack.py
"""
from __future__ import annotations

import ctypes
import hashlib
import math
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ballpack  # noqa: E402

MAGIC = b"SQHP1\0"          # then L as 16 bits: a header of 28 bytes, as ballpack's
LIMBS = ballpack.LIMBS
BISECT_STEPS, SHRINK_STEPS = ballpack.BISECT_STEPS, ballpack.SHRINK_STEPS


def checks(L):
    return [j | L for j in range(L)]


def counts(n, L, limit):
    """ways[i][s][r]: ways to finish a point from entry i in state s with squared length at most r, every
    block of L ending in state 0; rows of states other than 0 at the first entry of a block are zero."""
    states, h = 2 * L, checks(L)
    vmax = math.isqrt(limit)
    ev = [(v * v, 1 if v == 0 else 2) for v in range(0, vmax + 1, 2)]
    od = [(v * v, 2) for v in range(1, vmax + 1, 2)]

    def times(series, row):
        out = np.zeros(limit + 1, dtype=object)
        for c, mult in series:
            out[c:] += row[:limit + 1 - c] * mult
        return out

    zero = np.zeros(limit + 1, dtype=object)
    ways = [None] * (n + 1)
    ways[n] = [np.full(limit + 1, 1, dtype=object)] + [zero.copy() for _ in range(states - 1)]
    for i in range(n - 1, -1, -1):
        nxt = ways[i + 1]
        hj = h[i % L]
        ways[i] = [times(ev, nxt[s]) + times(od, nxt[s ^ hj]) for s in range(states)]
        if i % L == 0:
            for s in range(1, states):
                ways[i][s] = zero.copy()
    return ways


class HamFormat(ballpack.Format):
    """A_L repeated over a block of n weights, in a ball, with a number of `bits` bits and a scale code of one byte."""

    def __init__(self, L=16, n=32, bits=136, steps=16):
        if L not in (8, 16, 32) or n % L:
            raise ValueError("L must be 8, 16 or 32 and divide n")
        if bits % 8 or bits > 64 * LIMBS:
            raise ValueError(f"bits must be a multiple of 8 and at most {64 * LIMBS}")
        self.L, self.m = L, L.bit_length() - 1
        self.n, self.bits, self.steps, self.bytes = n, bits, steps, bits // 8
        self.states, self.h = 2 * L, checks(L)
        # the squared radius of a ball of 2^bits points of a lattice of determinant 2^(n(m+1)/L), roughly; then counted
        det = 2.0 ** ((self.m + 1) / L)
        r2 = (2.0 ** (bits / n) * det * math.exp(math.lgamma(n / 2.0 + 1.0) / n) / math.sqrt(math.pi)) ** 2
        limit = int(1.08 * r2) + 2 * n
        while True:
            ways = counts(n, L, limit)
            if ways[0][0][limit] > 1 << bits:
                break
            limit = int(limit * 1.3) + 8
        full = ways[0][0]
        self.bound = max(r for r in range(limit + 1) if full[r] <= 1 << bits)
        self.count = int(full[self.bound])
        if self.bound > 127 * 127:
            raise ValueError("an entry would not fit a byte")
        B = self.bound
        self._ways = [[[int(x) for x in ways[i][s][:B + 1]] for s in range(self.states)] for i in range(n + 1)]
        table = np.zeros((n + 1, self.states, B + 1, LIMBS), dtype=np.uint64)
        mask = (1 << 64) - 1
        for i in range(n + 1):
            for s in range(self.states):
                row = self._ways[i][s]
                for r in range(B + 1):
                    v = row[r]
                    if v:
                        for j in range(LIMBS):
                            table[i, s, r, j] = (v >> (64 * j)) & mask
        self._table = np.ascontiguousarray(table)
        self._lib = _library()
        self._use()
        r = math.sqrt(n / 4.0 + 1.5 * n / L)
        self._margin = r

    def _use(self):
        global _live
        if _live is not self:
            if self._lib.hp_init(self.n, self.L, self.bound, self.bytes, self._table.ctypes.data):
                raise RuntimeError("the C coder did not take the table")
            _live = self

    def table_entries(self):
        return (self.n + 1) * self.states * (self.bound + 1)

    # -- the reference numbering, in Python ------------------------------------------------------

    def _symbols(self, left):
        yield 0
        a = 1
        while a * a <= left:
            yield a
            yield -a
            a += 1

    def ref_rank(self, y):
        y = [int(v) for v in y]
        left, s, total = self.bound, 0, 0
        for i, v in enumerate(y):
            hj = self.h[i % self.L]
            if v * v > left:
                raise ValueError("the point is not in the codebook")
            for u in self._symbols(left):
                if u == v:
                    break
                total += self._ways[i + 1][s ^ hj if u & 1 else s][left - u * u]
            if v & 1:
                s ^= hj
            left -= v * v
            if (i + 1) % self.L == 0 and s:
                raise ValueError("the point is not in the codebook")
        return total

    def ref_unrank(self, k):
        if not 0 <= k < self.count:
            raise ValueError("no point has that number")
        left, s, out = self.bound, 0, []
        for i in range(self.n):
            hj = self.h[i % self.L]
            for u in self._symbols(left):
                t = s ^ hj if u & 1 else s
                w = self._ways[i + 1][t][left - u * u]
                if k < w:
                    break
                k -= w
            else:
                raise AssertionError("the table and the number disagree")
            out.append(u)
            s, left = t, left - u * u
        return np.array(out, dtype=np.int64)

    # -- numbers in C ----------------------------------------------------------------------------

    def rank(self, Y):
        Y = np.ascontiguousarray(Y, dtype=np.int8)
        out = np.empty((len(Y), self.bytes), dtype=np.uint8)
        self._use()
        bad = self._lib.hp_rank(Y.ctypes.data, len(Y), out.ctypes.data)
        if bad:
            raise ValueError(f"{bad} blocks are not points of the codebook")
        return out

    def unrank(self, numbers):
        numbers = np.ascontiguousarray(numbers, dtype=np.uint8)
        Y = np.empty((len(numbers), self.n), dtype=np.int8)
        self._use()
        bad = self._lib.hp_unrank(numbers.ctypes.data, len(numbers), Y.ctypes.data)
        if bad:
            raise ValueError(f"{bad} numbers name no point of the codebook")
        return Y

    def closest(self, X):
        """Nearest point of A_L to each block of L entries of the rows of X (blocks of n are whole blocks of L)."""
        src = np.ascontiguousarray(np.asarray(X, dtype=np.float64).reshape(-1, self.L))
        dst = np.empty_like(src)
        if self._lib.hp_closest(src.ctypes.data, dst.ctypes.data, len(src), self.m) != 0:
            raise ValueError("L must be 8, 16 or 32")
        return dst.reshape(np.shape(X))

    # -- a matrix --------------------------------------------------------------------------------

    def _scales(self, X):
        g = np.maximum(np.sqrt((X ** 2).sum(axis=1)), 1e-30)
        r = math.sqrt(self.bound)
        lo = g / (r + self._margin) * 0.99
        hi = g / max(r - self._margin, 0.05 * r) * 1.01
        for _ in range(BISECT_STEPS):
            mid = np.sqrt(lo * hi)
            ok = (self.closest(X / mid[:, None]) ** 2).sum(axis=1) <= self.bound
            hi = np.where(ok, mid, hi)
            lo = np.where(ok, lo, mid)
        return hi

    def encode(self, W, colscale=None, keep=None):
        """As ballpack.Format.encode, with the nearest point of the lattice in place of rounding each entry."""
        rows, cols = W.shape
        if cols % self.n:
            raise ValueError(f"a block of {self.n} does not tile a row of {cols}")
        X = W.astype(np.float64)
        c16 = None
        if colscale is not None:
            c16 = np.asarray(colscale).astype(np.float16)
            if not np.all(np.isfinite(c16)) or np.any(c16 <= 0):
                raise ValueError("a column weight is not a positive 16-bit float")
            X = X * c16.astype(np.float64)[None, :]
        k32, kept = None, None
        if keep is not None:
            k32 = np.asarray(keep)
            if k32.ndim != 1 or len(k32) == 0 or not np.issubdtype(k32.dtype, np.integer):
                raise ValueError("the kept columns are a list of whole numbers, and not an empty one")
            if k32.min() < 0 or k32.max() >= cols or np.any(np.diff(k32) <= 0):
                raise ValueError("a kept column is outside the matrix, or the columns are not in rising order")
            k32 = k32.astype(np.uint32)
            kept = np.ascontiguousarray(W[:, k32.astype(np.int64)]).astype(np.float16)
            if not np.all(np.isfinite(kept)):
                raise ValueError("an entry of a kept column does not fit a 16-bit float")
            X[:, k32.astype(np.int64)] = 0.0
        X = X.reshape(-1, self.n)
        t = self._scales(X)
        top = np.float16(t.max())
        if float(top) < float(t.max()):
            top = np.nextafter(top, np.float16(np.inf))
        if not np.isfinite(top):
            raise ValueError("the largest scale does not fit a 16-bit float")
        grid = lambda k: float(top) * 2.0 ** (-k / self.steps)
        k = np.clip(np.floor(np.log2(float(top) / t) * self.steps), 0, 255)
        low = (grid(k) < t) & (k > 0)
        k = np.where(low, k - 1, k)
        tq = grid(k)
        Y = self.closest(X / tq[:, None])
        out = (Y ** 2).sum(axis=1) > self.bound
        if np.any(out):
            xo = X[out] / tq[out, None]
            lo, hi = np.zeros(len(xo)), np.ones(len(xo))
            for _ in range(SHRINK_STEPS):
                mid = 0.5 * (lo + hi)
                good = (self.closest(xo * mid[:, None]) ** 2).sum(axis=1) <= self.bound
                lo = np.where(good, mid, lo)
                hi = np.where(good, hi, mid)
            Y[out] = self.closest(xo * lo[:, None])
        Yi = Y.astype(np.int8)
        codes = k.astype(np.uint8)
        records = np.concatenate([codes[:, None], self.rank(Yi)], axis=1)
        stored = {"rows": rows, "cols": cols, "top": top, "colscale": c16, "records": records, "keep": k32, "kept": kept}
        return stored, self.values(Yi, codes, top, c16, rows, cols, k32, kept), float(np.mean(out))

    # -- a file ----------------------------------------------------------------------------------

    def header(self, tensors):
        return MAGIC + struct.pack("<H5I", self.L, self.n, self.bits, self.bound, self.steps, tensors)

    @staticmethod
    def read(path):
        """A file as (HamFormat, list of (name, stored)). The body is read with the checks of ballpack.Format.read."""
        with open(path, "rb") as f:
            blob = f.read()
        if blob[:6] != MAGIC:
            raise ValueError("not a packed file of this format")
        if len(blob) < 28:
            raise ValueError("the file ends inside its header")
        L, n, bits, bound, steps, count = struct.unpack_from("<H5I", blob, 6)
        if L not in (8, 16, 32) or n < L or n % L or n > 4096 or steps < 1:
            raise ValueError("the file's header names a lattice or a block this reader does not know")
        fmt = HamFormat(L, n, bits, steps)
        if fmt.bound != bound:
            raise ValueError("the file's bound is not the one its L, n and bits give")
        return fmt, _body(blob, 28, count, n, fmt.bytes)

    def timing(self, records, top=1.0, repeats=3):
        records = np.ascontiguousarray(records, dtype=np.uint8)
        out = np.empty((len(records), self.n), dtype=np.float32)
        best = None
        for _ in range(repeats):
            self._use()
            t0 = time.process_time()
            bad = self._lib.hp_decode(records.ctypes.data, len(records), float(top), self.steps, out.ctypes.data)
            dt = time.process_time() - t0
            if bad:
                raise ValueError(f"{bad} numbers name no point of the codebook")
            best = dt if best is None else min(best, dt)
        return best, out


def _body(blob, at, count, n, nbytes):
    """The matrices of a file, from byte `at`: ballpack.Format.read's loop, with its checks."""
    tensors = []

    def need(size, what):
        if size < 0 or at + size > len(blob):
            raise ValueError(f"the file ends inside {what} of matrix {len(tensors) + 1} of {count}")

    for _ in range(count):
        need(2, "the name")
        (ln,) = struct.unpack_from("<H", blob, at)
        need(2 + ln, "the name")
        try:
            name = blob[at + 2: at + 2 + ln].decode("utf-8")
        except UnicodeDecodeError:
            raise ValueError(f"the name of matrix {len(tensors) + 1} is not text") from None
        at += 2 + ln
        need(9 + 2, "the header")
        rows, cols, flag = struct.unpack_from("<IIB", blob, at)
        at += 9
        if cols % n:
            raise ValueError(f"matrix {name}: a block of {n} does not tile a row of {cols}")
        top = np.frombuffer(blob, dtype="<f2", count=1, offset=at)[0]
        at += 2
        if not np.isfinite(top) or top <= 0:
            raise ValueError(f"matrix {name}: the largest scale is not a positive finite 16-bit float")
        if flag & ~3:
            raise ValueError("a matrix of the file has a flag this reader does not know")
        c, keep, kept = None, None, None
        if flag & 1:
            need(2 * cols, "the column weights")
            c = np.frombuffer(blob, dtype="<f2", count=cols, offset=at).copy()
            at += 2 * cols
            if not np.all(np.isfinite(c)) or np.any(c <= 0):
                raise ValueError(f"matrix {name}: a column weight is not a positive finite 16-bit float")
        if flag & 2:
            need(4, "the kept columns")
            (count_kept,) = struct.unpack_from("<I", blob, at)
            at += 4
            need(4 * count_kept, "the kept columns")
            keep = np.frombuffer(blob, dtype="<u4", count=count_kept, offset=at).copy()
            at += 4 * count_kept
            if count_kept == 0 or keep.max() >= cols or np.any(np.diff(keep.astype(np.int64)) <= 0):
                raise ValueError("the kept columns of a matrix are not columns of it in rising order")
            need(2 * rows * count_kept, "the kept columns")
            kept = np.frombuffer(blob, dtype="<f2", count=rows * count_kept, offset=at).reshape(rows, count_kept).copy()
            at += 2 * rows * count_kept
            if not np.all(np.isfinite(kept)):
                raise ValueError(f"matrix {name}: an entry of a kept column is not finite")
        blocks, width = rows * cols // n, nbytes + 1
        need(blocks * width, "the blocks")
        records = np.frombuffer(blob, dtype=np.uint8, count=blocks * width, offset=at).reshape(blocks, width).copy()
        at += blocks * width
        tensors.append((name, {"rows": rows, "cols": cols, "top": top, "colscale": c, "records": records, "keep": keep, "kept": kept}))
    if at != len(blob):
        raise ValueError("the file has bytes after its last matrix")
    return tensors


def _library():
    """hampack.c compiled to a shared library in a directory of this user's alone, named by the hash of its source."""
    src = (HERE / "hampack.c").read_bytes()
    path = ballpack._build_dir() / f"hampack-{hashlib.sha256(src).hexdigest()[:16]}.so"
    if not (path.exists() and ballpack._private(path)):
        part = path.with_suffix(f".{os.getpid()}.part")
        subprocess.run(["cc", "-O2", "-ffp-contract=off", "-shared", "-fPIC", "-o", str(part), str(HERE / "hampack.c"), "-lm"], check=True)
        os.chmod(part, 0o700)
        os.replace(part, path)
    lib = ctypes.CDLL(str(path))
    lib.hp_init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    lib.hp_init.restype = ctypes.c_int
    for f in (lib.hp_rank, lib.hp_unrank):
        f.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
        f.restype = ctypes.c_size_t
    lib.hp_decode.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_double, ctypes.c_int, ctypes.c_void_p]
    lib.hp_decode.restype = ctypes.c_size_t
    lib.hp_closest.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    lib.hp_closest.restype = ctypes.c_int
    return lib


_live = None
