"""Permutations with forbidden positions, and matrix permanents.

Pure Python 3, standard library only. Counts are exact integers or
fractions.Fraction. A board is a set of (row, col) cells of an n x n grid,
0-indexed. A permutation s hits a board B at i if (i, s(i)) is in B.
"""

import itertools
import math
import random
from fractions import Fraction


# ---------------------------------------------------------------------------
# 1. Derangements
# ---------------------------------------------------------------------------

def derangements_inclusion_exclusion(n):
    """D_n = n! * sum_{k=0..n} (-1)^k / k!, summed exactly."""
    total = sum(Fraction((-1) ** k, math.factorial(k)) for k in range(n + 1))
    value = math.factorial(n) * total
    assert value.denominator == 1
    return int(value)


def derangements_two_term(n):
    """D_n = (n-1) (D_{n-1} + D_{n-2}), with D_0 = 1 and D_1 = 0."""
    if n == 0:
        return 1
    a, b = 1, 0
    for m in range(2, n + 1):
        a, b = b, (m - 1) * (a + b)
    return b


def derangements_alternating(n):
    """D_n = n * D_{n-1} + (-1)^n, with D_0 = 1."""
    d = 1
    for m in range(1, n + 1):
        d = m * d + (-1) ** m
    return d


def derangements(n):
    """Number of permutations of n points with no fixed point."""
    return derangements_inclusion_exclusion(n)


def rencontres(n, k):
    """Permutations of n points with exactly k fixed points: C(n,k) D_{n-k}."""
    if k < 0 or k > n:
        return 0
    return math.comb(n, k) * derangements(n - k)


# ---------------------------------------------------------------------------
# Board helpers
# ---------------------------------------------------------------------------

def _check_board(board, n):
    for (i, j) in board:
        if not (0 <= i < n and 0 <= j < n):
            raise ValueError("cell (%r, %r) is outside the %d x %d grid" % (i, j, n, n))


def _pad(poly, n):
    poly = list(poly)
    while len(poly) < n + 1:
        poly.append(0)
    return poly


def _poly_add(p, q):
    size = max(len(p), len(q))
    return tuple((p[i] if i < len(p) else 0) + (q[i] if i < len(q) else 0)
                 for i in range(size))


def _poly_mul(p, q):
    out = [0] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        if a:
            for j, b in enumerate(q):
                out[i + j] += a * b
    return tuple(out)


# ---------------------------------------------------------------------------
# 3. Rook numbers, three independent ways
# ---------------------------------------------------------------------------

def rook_numbers_bruteforce(board, n):
    """r_0..r_n by enumerating non-attacking placements row by row."""
    _check_board(board, n)
    by_row = [[] for _ in range(n)]
    for (i, j) in board:
        by_row[i].append(j)
    counts = [0] * (n + 1)

    def go(row, used, k):
        if row == n:
            counts[k] += 1
            return
        go(row + 1, used, k)
        for j in by_row[row]:
            if not (used >> j) & 1:
                go(row + 1, used | (1 << j), k + 1)

    go(0, 0, 0)
    return counts


def _normalise(rows):
    """Drop empty rows, renumber used columns from 0, sort rows (as bitmasks)."""
    rows = [r for r in rows if r]
    union = 0
    for r in rows:
        union |= r
    if union == (1 << union.bit_length()) - 1:
        return tuple(sorted(rows))
    pos = {}
    for bit in range(union.bit_length()):
        if (union >> bit) & 1:
            pos[bit] = len(pos)
    new = []
    for r in rows:
        m = 0
        for bit, p in pos.items():
            if (r >> bit) & 1:
                m |= 1 << p
        new.append(m)
    return tuple(sorted(new))


def _components(rows):
    """Split rows into groups that share no column with other groups."""
    comps = []
    for r in rows:
        mask, group, keep = r, [r], []
        for m, g in comps:
            if m & mask:
                mask |= m
                group += g
            else:
                keep.append((m, g))
        changed = True
        while changed:
            changed = False
            keep2 = []
            for m, g in keep:
                if m & mask:
                    mask |= m
                    group += g
                    changed = True
                else:
                    keep2.append((m, g))
            keep = keep2
        keep.append((mask, group))
        comps = keep
    return [g for _, g in comps]


_DC_MEMO = {}


def _dc_poly(rows):
    """Rook polynomial (tuple of coefficients) of a normalised board."""
    if not rows:
        return (1,)
    hit = _DC_MEMO.get(rows)
    if hit is not None:
        return hit
    comps = _components(rows)
    if len(comps) > 1:
        poly = (1,)
        for g in comps:
            poly = _poly_mul(poly, _dc_poly(_normalise(g)))
    elif len(rows) == 1:
        poly = (1, bin(rows[0]).count("1"))
    else:
        r = max(range(len(rows)), key=lambda t: bin(rows[t]).count("1"))
        best_j, best_deg = None, -1
        for j in range(rows[r].bit_length()):
            if (rows[r] >> j) & 1:
                deg = sum((x >> j) & 1 for x in rows)
                if deg > best_deg:
                    best_j, best_deg = j, deg
        j = best_j
        bit = 1 << j
        deleted = list(rows)
        deleted[r] = rows[r] & ~bit
        contracted = [x & ~bit for t, x in enumerate(rows) if t != r]
        p_del = _dc_poly(_normalise(deleted))
        p_con = _dc_poly(_normalise(contracted))
        poly = _poly_add(p_del, (0,) + tuple(p_con))
    _DC_MEMO[rows] = poly
    return poly


def rook_numbers_deletion_contraction(board, n):
    """r_0..r_n by R(B) = R(B - c) + x R(B without c's row and column).

    Memoised on a normalised form; boards that split into parts sharing no
    row or column are handled by multiplying the parts.
    """
    _check_board(board, n)
    masks = [0] * n
    for (i, j) in board:
        masks[i] |= 1 << j
    poly = _dc_poly(_normalise(masks))
    return _pad(poly, n)


def rook_numbers_dp(board, n):
    """r_0..r_n by dynamic programming over subsets of used columns."""
    _check_board(board, n)
    by_row = [[] for _ in range(n)]
    for (i, j) in board:
        by_row[i].append(j)
    dp = {0: 1}
    for i in range(n):
        new = dict(dp)
        for mask, v in dp.items():
            for j in by_row[i]:
                if not (mask >> j) & 1:
                    m2 = mask | (1 << j)
                    new[m2] = new.get(m2, 0) + v
        dp = new
    counts = [0] * (n + 1)
    for mask, v in dp.items():
        counts[bin(mask).count("1")] += v
    return counts


def rook_numbers(board, n, method="auto"):
    """r_0..r_n. method: 'brute', 'dc', 'dp' or 'auto' (dp if n <= 10)."""
    if method == "auto":
        method = "dp" if n <= 10 else "dc"
    if method == "brute":
        return rook_numbers_bruteforce(board, n)
    if method == "dc":
        return rook_numbers_deletion_contraction(board, n)
    if method == "dp":
        return rook_numbers_dp(board, n)
    raise ValueError("unknown method %r" % (method,))


# ---------------------------------------------------------------------------
# 4. Ferrers boards
# ---------------------------------------------------------------------------

def ferrers_board(heights):
    """Cells (r, c) with r < heights[c]. Heights are nondecreasing, at most n."""
    return {(r, c) for c, h in enumerate(heights) for r in range(h)}


def falling_factorial(x, m):
    """x (x-1) ... (x-m+1); equal to 1 when m = 0."""
    out = 1
    for t in range(m):
        out *= x - t
    return out


def ferrers_check(heights, x):
    """Return (lhs, rhs) of the Goldman-Joichi-White identity at x.

    lhs = sum_k r_k * x(x-1)...(x-(n-k)+1), rhs = prod_i (x + b_i - i + 1)
    with i running from 1 to n.
    """
    n = len(heights)
    if any(heights[i] > heights[i + 1] for i in range(n - 1)):
        raise ValueError("heights must be nondecreasing")
    r = rook_numbers(ferrers_board(heights), n)
    lhs = sum(r[k] * falling_factorial(x, n - k) for k in range(n + 1))
    rhs = 1
    for i, b in enumerate(heights, start=1):
        rhs *= x + b - i + 1
    return lhs, rhs


# ---------------------------------------------------------------------------
# 5. Hit numbers
# ---------------------------------------------------------------------------

def hit_numbers(board, n, method="auto"):
    """h_0..h_n, h_j = permutations with exactly j hits.

    Uses H(x) = sum_k r_k (n-k)! (x-1)^k, expanded exactly.
    """
    r = rook_numbers(board, n, method)
    h = [0] * (n + 1)
    for k in range(n + 1):
        coef = r[k] * math.factorial(n - k)
        for j in range(k + 1):
            h[j] += coef * math.comb(k, j) * (-1) ** (k - j)
    return h


def hit_numbers_bruteforce(board, n):
    """h_0..h_n by running over all n! permutations."""
    _check_board(board, n)
    h = [0] * (n + 1)
    for s in itertools.permutations(range(n)):
        h[sum(1 for i in range(n) if (i, s[i]) in board)] += 1
    return h


# ---------------------------------------------------------------------------
# 6. Permanents
# ---------------------------------------------------------------------------

def _check_square(A):
    n = len(A)
    for row in A:
        if len(row) != n:
            raise ValueError("matrix must be square")
    return n


def permanent_bruteforce(A):
    """Sum over all permutations of the product of entries."""
    n = _check_square(A)
    total = 0
    for s in itertools.permutations(range(n)):
        p = 1
        for i in range(n):
            p *= A[i][s[i]]
            if not p:
                break
        total += p
    return total


def permanent_ryser(A):
    """Ryser's formula, columns subsets in Gray code order.

    perm(A) = (-1)^n sum_{S nonempty} (-1)^{|S|} prod_i sum_{j in S} a_ij.
    Each step adds or removes one column from the running row sums.
    """
    n = _check_square(A)
    if n == 0:
        return 1
    rowsum = [0 * A[0][0]] * n
    inset = [False] * n
    size = 0
    total = 0
    for t in range(1, 1 << n):
        j = (t & -t).bit_length() - 1
        if inset[j]:
            inset[j] = False
            size -= 1
            rowsum = [rowsum[i] - A[i][j] for i in range(n)]
        else:
            inset[j] = True
            size += 1
            rowsum = [rowsum[i] + A[i][j] for i in range(n)]
        prod = 1
        for v in rowsum:
            prod *= v
            if not prod:
                break
        total += -prod if size % 2 else prod
    return -total if n % 2 else total


def permanent_glynn(A):
    """Glynn's formula, divided by 2^(n-1) at the end.

    perm(A) = 2^-(n-1) sum_{d in {+-1}^n, d_0 = 1} (prod d_k) prod_j sum_i d_i a_ij.
    """
    n = _check_square(A)
    if n == 0:
        return 1
    colsum = [sum(A[i][j] for i in range(n)) for j in range(n)]
    delta = [1] * n
    sign = 1
    total = 0
    for t in range(1 << (n - 1)):
        if t:
            i = (t & -t).bit_length()  # flips row index 1 + ctz(t)
            old = delta[i]
            colsum = [colsum[j] - 2 * old * A[i][j] for j in range(n)]
            delta[i] = -old
            sign = -sign
        prod = 1
        for v in colsum:
            prod *= v
            if not prod:
                break
        total += sign * prod
    d = 1 << (n - 1)
    if isinstance(total, int):
        if total % d == 0:
            return total // d
        return Fraction(total, d)
    return total / d


# ---------------------------------------------------------------------------
# 7. Menage numbers
# ---------------------------------------------------------------------------

def menage_board(n):
    """Cells (i, i) and (i, (i+1) mod n)."""
    return {(i, i) for i in range(n)} | {(i, (i + 1) % n) for i in range(n)}


def menage_bruteforce(n):
    """Permutations s with s(i) != i and s(i) != (i+1) mod n."""
    board = menage_board(n)
    return sum(1 for s in itertools.permutations(range(n))
               if all((i, s[i]) not in board for i in range(n)))


def menage_rook_number(n, k):
    """(2n/(2n-k)) C(2n-k, k); this is r_k of the menage board for n >= 3."""
    return Fraction(2 * n, 2 * n - k) * math.comb(2 * n - k, k)


def menage(n):
    """Touchard's formula, with the convention of the permutation count.

    For n >= 2 the formula sum_k (-1)^k (2n/(2n-k)) C(2n-k,k) (n-k)! gives the
    count (n = 2 gives 0). For n = 1 the formula gives -1 but the count is 0,
    because the two forbidden cells coincide. For n = 0 the count is 1.
    """
    if n == 0:
        return 1
    if n == 1:
        return 0
    total = sum((-1) ** k * menage_rook_number(n, k) * math.factorial(n - k)
                for k in range(n + 1))
    assert total.denominator == 1
    return int(total)


# ---------------------------------------------------------------------------
# 8. Integer vectors in a box with sum zero
# ---------------------------------------------------------------------------

def count_zero_sum_box(m, k):
    """Vectors of length m with entries in [-k, k] and sum 0."""
    if m == 0:
        return 1
    total = 0
    for j in range(m + 1):
        top = m * k - j * (2 * k + 1)
        if top < 0:
            break
        total += (-1) ** j * math.comb(m, j) * math.comb(top + m - 1, m - 1)
    return total


# ---------------------------------------------------------------------------
# 9. Perfect matchings and the closure of a board
# ---------------------------------------------------------------------------

def max_matching(allowed):
    """Maximum matching by augmenting paths. Returns match_row, where
    match_row[r] is the column of row r, or -1 if row r is unmatched."""
    n = len(allowed)
    match_col = [-1] * n
    match_row = [-1] * n

    def augment(r, seen):
        for c in range(n):
            if allowed[r][c] and not seen[c]:
                seen[c] = True
                if match_col[c] == -1 or augment(match_col[c], seen):
                    match_col[c] = r
                    match_row[r] = c
                    return True
        return False

    for r in range(n):
        augment(r, [False] * n)
    return match_row


def _neighbourhood(allowed, rows):
    n = len(allowed)
    return {c for r in rows for c in range(n) if allowed[r][c]}


def hall_violator(allowed):
    """None if a perfect matching exists. Otherwise a set S of rows with
    |N(S)| < |S|, found by alternating paths from an unmatched row."""
    n = len(allowed)
    match_row = max_matching(allowed)
    match_col = [-1] * n
    for r, c in enumerate(match_row):
        if c != -1:
            match_col[c] = r
    free = [r for r in range(n) if match_row[r] == -1]
    if not free:
        return None
    rows = {free[0]}
    stack = [free[0]]
    cols = set()
    while stack:
        r = stack.pop()
        for c in range(n):
            if allowed[r][c] and c not in cols:
                cols.add(c)
                r2 = match_col[c]
                if r2 != -1 and r2 not in rows:
                    rows.add(r2)
                    stack.append(r2)
    return rows


def dead_cells(allowed):
    """Allowed cells on no perfect matching, each with a certificate.

    Returns {} if there is no perfect matching. The certificate of (i, j) is a
    set S of rows with |N(S)| = |S|, j in N(S) and i not in S.
    """
    n = len(allowed)
    match = max_matching(allowed)
    if any(c == -1 for c in match):
        return {}
    adj = [[r2 for r2 in range(n) if r2 != r and allowed[r][match[r2]]]
           for r in range(n)]
    reach = []
    for start in range(n):
        seen = {start}
        stack = [start]
        while stack:
            r = stack.pop()
            for r2 in adj[r]:
                if r2 not in seen:
                    seen.add(r2)
                    stack.append(r2)
        reach.append(seen)
    dead = {}
    for i in range(n):
        for i2 in range(n):
            if i != i2 and allowed[i][match[i2]] and i not in reach[i2]:
                dead[(i, match[i2])] = set(reach[i2])
    return dead


def verify_dead_certificate(allowed, cell, S):
    """Check |N(S)| = |S|, j in N(S), and i not in S. Written independently
    of dead_cells."""
    i, j = cell
    S = set(S)
    if i in S:
        return False
    nbhd = set()
    for r in S:
        for c, v in enumerate(allowed[r]):
            if v:
                nbhd.add(c)
    return len(nbhd) == len(S) and j in nbhd


def closure(board, n):
    """Board united with the dead cells of J - board. If J - board has no
    perfect matching, all n*n cells."""
    allowed = [[0 if (i, j) in board else 1 for j in range(n)] for i in range(n)]
    if hall_violator(allowed) is not None:
        return {(i, j) for i in range(n) for j in range(n)}
    return set(board) | set(dead_cells(allowed))


def matching_through(allowed, cell):
    """A perfect matching that uses the cell, as the list of the column of
    each row. None if there is none."""
    n = len(allowed)
    i, j = cell
    if not allowed[i][j]:
        return None
    forced = [[allowed[r][c] if (r == i) == (c == j) else 0 for c in range(n)]
              for r in range(n)]
    match = max_matching(forced)
    return None if any(c == -1 for c in match) else match


def closure_witnessed(board, n):
    """The closure of a board with a witness for every cell, closed or open.

    closure() returns the cells and drops the reasons. This returns
    (closed, opened). closed maps each cell of the closure to its witness:

      ("board",)           the cell is in the board
      ("dead", S)          the cell is allowed and on no perfect matching of
                           J - board: S is a set of rows with |N(S)| = |S|,
                           j in N(S) and i not in S
      ("no matching", S)   J - board has no perfect matching at all: S is a
                           set of rows with |N(S)| < |S|

    opened maps every other cell to a perfect matching of J - board that uses
    it, as the list of the column of each row. So a cell is closed on a
    certificate and left open on one too. verify_closure() checks both.
    """
    allowed = [[0 if (i, j) in board else 1 for j in range(n)] for i in range(n)]
    cells = [(i, j) for i in range(n) for j in range(n)]
    viol = hall_violator(allowed)
    if viol is not None:
        return {cell: (("board",) if cell in board else ("no matching", frozenset(viol)))
                for cell in cells}, {}
    dead = dead_cells(allowed)
    closed, opened = {}, {}
    for cell in cells:
        if cell in board:
            closed[cell] = ("board",)
        elif cell in dead:
            closed[cell] = ("dead", frozenset(dead[cell]))
        else:
            opened[cell] = matching_through(allowed, cell)
    return closed, opened


def verify_closure(board, n, closed, opened):
    """Check a witnessed closure against the board alone. Returns the list of
    what is wrong, empty if nothing is. Written independently of
    closure_witnessed: it finds no matching and no violator itself."""
    wrong = []
    cells = {(i, j) for i in range(n) for j in range(n)}
    if set(closed) & set(opened):
        wrong.append("a cell is both closed and open")
    if set(closed) | set(opened) != cells:
        wrong.append("the closed and open cells are not all the cells")

    def columns_of(S):
        return {c for r in S for c in range(n) if (r, c) not in board}

    for (i, j), w in closed.items():
        kind = w[0] if isinstance(w, tuple) and w else None
        if kind == "board":
            ok = (i, j) in board
        elif kind in ("dead", "no matching") and len(w) == 2:
            S = set(w[1])
            rows_ok = bool(S) and all(isinstance(r, int) and 0 <= r < n for r in S)
            if not rows_ok:
                ok = False
            elif kind == "dead":
                N = columns_of(S)
                ok = (i, j) not in board and len(N) == len(S) and j in N and i not in S
            else:
                ok = len(columns_of(S)) < len(S)
        else:
            ok = False
        if not ok:
            wrong.append(f"closed cell {(i, j)}: its witness does not hold")
    for (i, j), m in opened.items():
        ok = (isinstance(m, (list, tuple)) and len(m) == n
              and sorted(m) == list(range(n))
              and all((r, m[r]) not in board for r in range(n)) and m[i] == j)
        if not ok:
            wrong.append(f"open cell {(i, j)}: no perfect matching through it is given")
    return wrong


# ---------------------------------------------------------------------------
# 10. Permanent bounds for 0/1 matrices
# ---------------------------------------------------------------------------

def bregman_minc_upper(A):
    """prod_i (r_i!)^(1/r_i) over row sums r_i; 0 if a row is empty."""
    bound = 1.0
    for row in A:
        r = sum(row)
        if r == 0:
            return 0.0
        bound *= math.exp(math.lgamma(r + 1) / r)
    return bound


def van_der_waerden_lower(n):
    """n!/n^n as a Fraction."""
    return Fraction(math.factorial(n), n ** n)


# ---------------------------------------------------------------------------
# 11. Sinkhorn scaling
# ---------------------------------------------------------------------------

def _sinkhorn_step(P, r, c):
    n = len(P)
    for i in range(n):
        s = sum(P[i])
        if s <= 0:
            raise ValueError("row %d has no positive entry" % i)
        P[i] = [v / s for v in P[i]]
        r[i] /= s
    for j in range(n):
        s = sum(P[i][j] for i in range(n))
        if s <= 0:
            raise ValueError("column %d has no positive entry" % j)
        for i in range(n):
            P[i][j] /= s
        c[j] /= s


def sinkhorn(A, iters):
    """Alternate row and column normalisation. Returns P, r, c with
    P = diag(r) A diag(c)."""
    n = len(A)
    P = [[float(v) for v in row] for row in A]
    r = [1.0] * n
    c = [1.0] * n
    for _ in range(iters):
        _sinkhorn_step(P, r, c)
    return P, r, c


def sinkhorn_permanent_bounds(A, iters):
    """Return (lower, upper) from Sinkhorn scaling.

    After a final row normalisation P is row stochastic, so perm(P) <= 1 and
    perm(A) <= 1/(prod r prod c): the upper bound is always valid. The lower
    bound uses perm(P) >= n!/n^n, which holds only if P is exactly doubly
    stochastic, so it is not guaranteed.
    """
    n = len(A)
    P, r, c = sinkhorn(A, iters)
    r = list(r)
    for i in range(n):
        r[i] /= sum(P[i])
    logscale = sum(math.log(v) for v in r) + sum(math.log(v) for v in c)
    upper = math.exp(-logscale)
    lower = float(van_der_waerden_lower(n)) * upper
    return lower, upper


def random_01_with_matching(rng, n):
    """Random 0/1 matrix (density 0.3, 0.5 or 0.7) with a perfect matching."""
    while True:
        p = rng.choice((0.3, 0.5, 0.7))
        A = [[1 if rng.random() < p else 0 for _ in range(n)] for _ in range(n)]
        if all(c != -1 for c in max_matching(A)):
            return A


def sinkhorn_lower_bound_failures(trials, n, iters, seed):
    """Count random 0/1 matrices with a perfect matching for which the
    computed lower bound exceeds the exact permanent (relative slack 1e-12)."""
    rng = random.Random(seed)
    bad = 0
    for _ in range(trials):
        A = random_01_with_matching(rng, n)
        lower, _ = sinkhorn_permanent_bounds(A, iters)
        if lower > permanent_ryser(A) * (1 + 1e-12):
            bad += 1
    return bad


def sinkhorn_dead_cell_decay(allowed, cell, iter_list):
    """Value of P at `cell` after each iteration count in iter_list."""
    n = len(allowed)
    P = [[float(v) for v in row] for row in allowed]
    r = [1.0] * n
    c = [1.0] * n
    out = {}
    wanted = set(iter_list)
    done = 0
    if 0 in wanted:
        out[0] = P[cell[0]][cell[1]]
    for target in range(1, max(iter_list) + 1):
        _sinkhorn_step(P, r, c)
        done = target
        if target in wanted:
            out[target] = P[cell[0]][cell[1]]
    return [out[t] for t in iter_list]


# ---------------------------------------------------------------------------
# 12. Godsil-Gutman estimator
# ---------------------------------------------------------------------------

def determinant_float(M):
    """Determinant by Gaussian elimination with partial pivoting."""
    n = len(M)
    M = [list(row) for row in M]
    det = 1.0
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if M[piv][col] == 0.0:
            return 0.0
        if piv != col:
            M[col], M[piv] = M[piv], M[col]
            det = -det
        det *= M[col][col]
        inv = 1.0 / M[col][col]
        for r in range(col + 1, n):
            f = M[r][col] * inv
            if f:
                rowr, rowc = M[r], M[col]
                for k in range(col + 1, n):
                    rowr[k] -= f * rowc[k]
    return det


def godsil_gutman_stats(A, samples, seed):
    """Return (mean, standard deviation of one sample, standard error)."""
    n = len(A)
    rng = random.Random(seed)
    roots = [[math.sqrt(v) for v in row] for row in A]
    s1 = 0.0
    s2 = 0.0
    for _ in range(samples):
        bits = rng.getrandbits(n * n)
        B = [[roots[i][j] if (bits >> (i * n + j)) & 1 else -roots[i][j]
              for j in range(n)] for i in range(n)]
        d = determinant_float(B)
        d *= d
        s1 += d
        s2 += d * d
    mean = s1 / samples
    var = max(0.0, (s2 - samples * mean * mean) / (samples - 1))
    sd = math.sqrt(var)
    return mean, sd, sd / math.sqrt(samples)


def godsil_gutman(A, samples, seed):
    """Return (mean, standard error) of det(B)^2 over random sign matrices."""
    mean, _, se = godsil_gutman_stats(A, samples, seed)
    return mean, se


# ---------------------------------------------------------------------------
# 13. Poisson approximation for the number of hits
# ---------------------------------------------------------------------------

def exact_tv_to_poisson(board, n):
    """Total variation distance between the law of W and Poisson(|B|/n)."""
    lam = len(board) / n
    h = hit_numbers(board, n)
    nf = math.factorial(n)
    q = math.exp(-lam)
    dist = 0.0
    for k in range(n + 1):
        if k:
            q *= lam / k
        dist += abs(h[k] / nf - q)
    tail = 0.0
    k = n
    while True:
        k += 1
        q *= lam / k
        tail += q
        if k > lam + 10 and q < 1e-18:
            break
    return 0.5 * (dist + tail)


def stein_chen_bound(board, n):
    """((1-e^-lam)/lam) (1/n) sum_{(i,j) in B} [ (r_i+c_j-1)/n
    + (|B|-r_i-c_j+1)/(n(n-1)) ], for n >= 2. Zero for an empty board."""
    size = len(board)
    if size == 0:
        return 0.0
    if n < 2:
        raise ValueError("n must be at least 2")
    rows = {}
    cols = {}
    for (i, j) in board:
        rows[i] = rows.get(i, 0) + 1
        cols[j] = cols.get(j, 0) + 1
    total = Fraction(0)
    for (i, j) in board:
        s = rows[i] + cols[j] - 1
        total += Fraction(s, n) + Fraction(size - s, n * (n - 1))
    lam = size / n
    return (1 - math.exp(-lam)) / lam * float(total) / n
