"""Number the points of a ball cut from A_n repeated over a group, A_n = C_n + 2Z^n with C_n the extended Hamming code.

The record hamming-lattices-real-weights-2026-11-24 rounds groups of 128 weights to A_16 or A_32
repeated 8 or 4 times, cut by a ball, and counts the bits as log2 of the number of points. This
file numbers the points, so that those bits can be stored: rank() turns a point into a whole
number from 0 to N - 1 and unrank() turns it back. A group then takes ceil(log2 N) bits, at most
one bit more than counted.

The points are integer vectors. A point y is in A_n when the parities y mod 2 form a word of the
extended Hamming code [n, n - m - 1, 4], n = 2^m: the XOR of the check columns h_i = i | n over
the odd entries is zero. That XOR, taken over the entries read so far, is the state of the trellis
that the record's decoder (hamming.py) runs; the same trellis counts here.

Order. A group of k blocks is read block by block. Blocks are compared first by their squared
length t, then by their number among the points of A_n of squared length exactly t, which is the
order of their entries with the symbols 0, 1, -1, 2, -2, ... So the number of a group is a sum,
over its blocks, of the points that agree with it on the blocks before and come earlier in the
block: those with a shorter block, and those with a block of the same length and a lower number.
This is enumerative coding (Cover, 1973), as in ballindex.py, in two levels.

Tables, built once for a lattice and a bound R:
  shell[j][s][t]  ways to finish a block from entry j, with state s, so that the rest has
                  squared length exactly t and the block ends in A_n. shell[0][0][t] is the
                  number of points of A_n of squared length t.
  after[b][r]     ways to fill b blocks with squared length at most r in all.

Exact whole-number arithmetic in Python. Not fast: the point is a checked reference that a
faster coder can be held to. Tests: python3 -m unittest shape/test_hammingindex.py
"""
from __future__ import annotations

import numpy as np


def checks(m):
    """The check columns of the extended Hamming code of length 2^m, as whole numbers of m + 1 bits."""
    n = 1 << m
    return [i | n for i in range(n)]


def _symbols(limit):
    """Integer entries with square at most limit, in the order 0, 1, -1, 2, -2, ...: (entry, square)."""
    out, k = [(0, 0)], 1
    while k * k <= limit:
        out += [(k, k * k), (-k, k * k)]
        k += 1
    return out


class HammingBallIndex:
    """The points of A_n repeated k times (n = 2^m, k = group // n) with squared length at most bound."""

    def __init__(self, m, group, bound):
        n = 1 << m
        if group % n:
            raise ValueError("the group must be a whole number of blocks")
        self.m, self.n, self.k, self.group, self.bound = m, n, group // n, group, int(bound)
        self.h = checks(m)
        self.states = 2 * n
        self.lookups = 0
        R = self.bound
        self._syms = _symbols(R)
        # the squared lengths of the even and of the odd integers, as arrays of counts up to R
        even = np.zeros(R + 1, dtype=object)
        odd = np.zeros(R + 1, dtype=object)
        for v, c in self._syms:
            (odd if v & 1 else even)[c] += 1
        ev = [(c, int(even[c])) for c in range(R + 1) if even[c]]
        od = [(c, int(odd[c])) for c in range(R + 1) if odd[c]]

        def times(series, row):
            out = np.zeros(R + 1, dtype=object)
            for c, mult in series:
                if c == 0:
                    out += row * mult if mult != 1 else row
                else:
                    out[c:] += row[:R + 1 - c] * mult
            return out

        zero = np.zeros(R + 1, dtype=object)
        last = [zero.copy() for _ in range(self.states)]
        last[0][0] = 1
        shell = [None] * (n + 1)
        shell[n] = last
        for j in range(n - 1, -1, -1):
            nxt, hj = shell[j + 1], self.h[j]
            cur = []
            for s in range(self.states):
                cur.append(times(ev, nxt[s]) + times(od, nxt[s ^ hj]))
            shell[j] = cur
        self._shell = [[[int(x) for x in row] for row in level] for level in shell]
        self.block_count = self._shell[0][0]                       # points of A_n of squared length exactly t
        after = [[1] * (R + 1)]
        S = np.array(self.block_count, dtype=object)
        for _ in range(self.k):
            prev = np.array(after[-1], dtype=object)
            new = np.zeros(R + 1, dtype=object)
            for t in range(R + 1):
                if S[t]:
                    new[t:] += S[t] * prev[:R + 1 - t]
            after.append([int(x) for x in new])
        self._after = after
        self.count = after[self.k][R]

    def bits(self):
        """The bits a number takes: the least whole number of bits that holds N - 1."""
        return max(1, (self.count - 1).bit_length())

    def table_entries(self):
        return (self.n + 1) * self.states * (self.bound + 1) + (self.k + 1) * (self.bound + 1)

    def contains(self, y):
        y = np.asarray(y)
        if y.shape != (self.group,) or not np.all(y == np.rint(y)):
            return False
        yi = [int(v) for v in y]
        if sum(v * v for v in yi) > self.bound:
            return False
        for b in range(self.k):
            s = 0
            for j in range(self.n):
                if yi[b * self.n + j] & 1:
                    s ^= self.h[j]
            if s:
                return False
        return True

    # -- one block ------------------------------------------------------------------------------

    def _block_rank(self, blk, t):
        s, left, idx = 0, t, 0
        for j, y in enumerate(blk):
            row = self._shell[j + 1]
            hj = self.h[j]
            for v, c in self._syms:
                if v == y:
                    break
                if c <= left:
                    idx += row[s ^ hj if v & 1 else s][left - c]
                    self.lookups += 1
            if y & 1:
                s ^= hj
            left -= y * y
        return idx

    def _block_unrank(self, idx, t):
        s, left, out = 0, t, []
        for j in range(self.n):
            row = self._shell[j + 1]
            hj = self.h[j]
            for v, c in self._syms:
                if c > left:
                    continue
                ways = row[s ^ hj if v & 1 else s][left - c]
                self.lookups += 1
                if idx < ways:
                    break
                idx -= ways
            else:
                raise AssertionError("the table and the number disagree")
            out.append(v)
            if v & 1:
                s ^= hj
            left -= c
        return out

    # -- a group --------------------------------------------------------------------------------

    def rank(self, y):
        """The number of a point of the codebook, from 0 to N - 1."""
        if not self.contains(y):
            raise ValueError("the point is not in the codebook")
        yi = [int(v) for v in np.asarray(y)]
        S = self.block_count
        left, total = self.bound, 0
        for b in range(self.k):
            blk = yi[b * self.n:(b + 1) * self.n]
            t = sum(v * v for v in blk)
            rest = self._after[self.k - b - 1]
            for u in range(t):
                if S[u]:
                    total += S[u] * rest[left - u]
            total += self._block_rank(blk, t) * rest[left - t]
            left -= t
        return total

    def unrank(self, k):
        """The point of the codebook with the number k, as an array of whole numbers."""
        if not 0 <= k < self.count:
            raise ValueError("no point has that number")
        S = self.block_count
        left, out = self.bound, []
        for b in range(self.k):
            rest = self._after[self.k - b - 1]
            t = 0
            while True:
                if t > left:
                    raise AssertionError("the table and the number disagree")
                ways = S[t] * rest[left - t]
                if k < ways:
                    break
                k -= ways
                t += 1
            per = rest[left - t]
            out += self._block_unrank(k // per, t)
            k %= per
            left -= t
        return np.array(out, dtype=np.int64)
