"""Number the points of a codebook cut from a lattice by a ball over n weights.

The records of 2026-11-17 cut codebooks from the integers Z^n, or from E8 repeated n/8 times,
by a ball: every point whose squared length is at most a bound. They rounded weights to such
codebooks and measured the error, and said each time that the codebooks have no packed
format, because nothing numbered their points. This file numbers them.

rank() turns a point of the codebook into a whole number from 0 to N - 1, where N is the size
of the codebook, and unrank() turns the number back into the point. The two are inverse, so a
block of n weights can be stored in the bits of its number: no more than n log2(q) bits when
the codebook has no more than q^n points.

How. Read the n entries of a point in order. The number of a point is the count of the points
that come before it, and a point comes before it if at the first entry where the two differ
it has the earlier symbol. So the number is a sum, over the entries, of the counts of the ways
to finish a point after an earlier symbol. Those counts depend only on how many entries are
left and how much of the bound is left (and, for E8, on where in a block of 8 the entry is,
which of E8's two halves the block is in, and the parity of the block so far). They are held
in a table, built once. This is enumerative coding (Cover, 1973), and for a ball it is what
transmission calls sphere shaping.

Nothing here is fast. It is whole-number arithmetic in Python, written to be exact and to be
checked. ops() counts the table look-ups a decoder makes, which is the cost that carries over
to any implementation.

Needs NumPy only for the arrays it takes and returns. Tests: python3 -m unittest shape/test_ballindex.py
"""
from __future__ import annotations

import numpy as np


def _symbols(limit, half, unit=4):
    """The entries a coordinate can take with cost at most limit, in the order used for numbering.

    Whole entries in the order 0, 1, -1, 2, -2, ...; half entries 1/2, -1/2, 3/2, -3/2, ...
    Each is returned as (twice the entry, cost). With unit = 4 the cost is the square of twice
    the entry, so that the cost of a half entry is a whole number. With unit = 1, for whole
    entries only, it is the square of the entry.
    """
    out = []
    k = 1 if half else 0
    while k * k * unit // 4 <= limit:
        cost = k * k * unit // 4
        if k == 0:
            out.append((0, 0))
        else:
            out.append((k, cost))
            out.append((-k, cost))
        k += 2
    return out


class BallIndex:
    """The points of Z^n (lattice "Z8") or of E8 repeated n/8 times (lattice "E8") of squared length at most bound."""

    def __init__(self, lattice, n, bound):
        if lattice not in ("Z8", "E8"):
            raise ValueError("lattice must be Z8 or E8")
        if lattice == "E8" and n % 8:
            raise ValueError("E8 needs n to be a multiple of 8")
        self.lattice, self.n, self.bound = lattice, n, bound
        self.unit = 1 if lattice == "Z8" else 4       # for E8 costs are squares of twice the entries
        self.budget = self.unit * bound
        self.lookups = 0
        if lattice == "Z8":
            self._build_integers()
        else:
            self._build_e8()

    # -- the integers ------------------------------------------------------------------------

    def _build_integers(self):
        B = self.budget
        self._whole = _symbols(B, half=False, unit=1)
        # ways[m][r]: points of Z^m with squared length at most r
        row = [1] * (B + 1)
        self._ways = [row]
        for _ in range(self.n):
            new = [0] * (B + 1)
            for _, cost in self._whole:
                for r in range(cost, B + 1):
                    new[r] += row[r - cost]
            row = new
            self._ways.append(row)
        self.count = self._ways[self.n][B]

    # -- E8 ----------------------------------------------------------------------------------

    def _build_e8(self):
        B = self.budget
        self._whole = _symbols(B, half=False)
        self._half = _symbols(B, half=True)
        blocks = self.n // 8
        # after[b][r]: ways to fill b whole blocks with size at most r
        # inside[b][j][c][p][r]: ways to finish the current block from entry j, in half c of E8, with
        # parity p so far, and then fill b more blocks, with size at most r
        self._after = [[1] * (B + 1)]
        self._inside = []
        for b in range(blocks):
            done = self._after[b]
            level = [None] * 9
            level[8] = [[done, [0] * (B + 1)], [done, [0] * (B + 1)]]          # a block ends with an even sum
            for j in range(7, -1, -1):
                level[j] = [[None, None], [None, None]]
                for c, syms in ((0, self._whole), (1, self._half)):
                    for p in (0, 1):
                        new = [0] * (B + 1)
                        for twice, cost in syms:
                            k = twice // 2                                      # the whole part: floor of the entry
                            nxt = level[j + 1][c][p ^ (k & 1)]
                            for r in range(cost, B + 1):
                                new[r] += nxt[r - cost]
                        level[j][c][p] = new
            self._inside.append(level)
            self._after.append([level[0][0][0][r] + level[0][1][0][r] for r in range(B + 1)])
        self.count = self._after[blocks][B]

    # -- shared ------------------------------------------------------------------------------

    def bits(self):
        """The bits a number of a point takes: the least whole number of bits that holds N - 1."""
        return max(1, (self.count - 1).bit_length())

    def table_entries(self):
        """How many whole numbers the table holds."""
        if self.lattice == "Z8":
            return (self.n + 1) * (self.budget + 1)
        return (self.n // 8) * 9 * 4 * (self.budget + 1) + (self.n // 8 + 1) * (self.budget + 1)

    def contains(self, y):
        y2 = np.rint(2.0 * np.asarray(y, dtype=np.float64)).astype(np.int64)
        if not np.allclose(y2 / 2.0, y):
            return False
        if int((y2 * y2).sum()) > 4 * self.bound:
            return False
        if self.lattice == "Z8":
            return bool(np.all(y2 % 2 == 0))
        for b in range(self.n // 8):
            blk = y2[8 * b:8 * b + 8]
            odd = blk % 2 != 0
            if odd.any() and not odd.all():
                return False
            if int(np.floor_divide(blk, 2).sum()) % 2:
                return False
        return True

    def _steps(self):
        """For each entry in turn: (the symbols it may take, a function from (symbol index, size left) to the count of ways to finish).

        A generator that is driven by send(): it is told the symbol chosen and the size left.
        Used by rank and by unrank, so that the two walk the same table the same way.
        """
        if self.lattice == "Z8":
            for i in range(self.n):
                row = self._ways[self.n - i - 1]
                yield self._whole, [row] * len(self._whole)
            return
        blocks = self.n // 8
        for b in range(blocks):
            level = self._inside[blocks - b - 1]
            c = p = None
            for j in range(8):
                if j == 0:
                    syms = self._whole + self._half
                    tables = [level[1][0][(t // 2) & 1] for t, _ in self._whole] + [level[1][1][(t // 2) & 1] for t, _ in self._half]
                else:
                    syms = self._half if c else self._whole
                    tables = [level[j + 1][c][p ^ ((t // 2) & 1)] for t, _ in syms]
                twice = yield syms, tables
                if j == 0:
                    c, p = twice & 1, 0
                p ^= (twice // 2) & 1

    def rank(self, y):
        """The number of a point of the codebook, from 0 to N - 1."""
        if not self.contains(y):
            raise ValueError("the point is not in the codebook")
        y2 = [int(v) for v in np.rint(2.0 * np.asarray(y, dtype=np.float64))]
        left, total = self.budget, 0
        walk = self._steps()
        step = next(walk)
        for i, twice in enumerate(y2):
            syms, tables = step
            for (t, cost), table in zip(syms, tables):
                if t == twice:
                    break
                if cost <= left:
                    total += table[left - cost]
                    self.lookups += 1
            else:
                raise AssertionError("an entry of a point of the codebook is not among its symbols")
            left -= cost
            if i + 1 < self.n:
                step = walk.send(twice) if self.lattice == "E8" else next(walk)
        return total

    def unrank(self, k):
        """The point of the codebook with the number k, as an array of n entries."""
        if not 0 <= k < self.count:
            raise ValueError("no point has that number")
        left, out = self.budget, []
        walk = self._steps()
        step = next(walk)
        for i in range(self.n):
            syms, tables = step
            for (t, cost), table in zip(syms, tables):
                if cost > left:
                    continue
                ways = table[left - cost]
                self.lookups += 1
                if k < ways:
                    break
                k -= ways
            else:
                raise AssertionError("the table and the number disagree")
            out.append(t)
            left -= cost
            if i + 1 < self.n:
                step = walk.send(t) if self.lattice == "E8" else next(walk)
        return np.array(out, dtype=np.float64) / 2.0
