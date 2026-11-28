"""Pure-Q Sloane Had.16.3 minimax block-16 Hadamard rotation and Sylvester H16 comparison."""

from __future__ import annotations

from fractions import Fraction
import math
from typing import Any, Sequence

from ._validation import require

_HAD_16_3_STRINGS: tuple[str, ...] = (
    "++++++++++++++++",
    "+-+-+-+-+-+-+-+-",
    "++--++--++--++--",
    "+--++--++--++--+",
    "++++----++++----",
    "+-+--+-++-+--+-+",
    "++----++++----++",
    "+--+-++-+--+-++-",
    "++++++++--------",
    "+++-+------+-+++",
    "++-+---+--+-+++-",
    "++---++---+++--+",
    "+-++-+---+--+-++",
    "+-+---++-+-+++--",
    "+--++-+--++--+-+",
    "+---++-+-+++--+-",
)

SLOANE_HAD_16_3_INT: tuple[tuple[int, ...], ...] = tuple(
    tuple(1 if ch == "+" else -1 for ch in row) for row in _HAD_16_3_STRINGS
)

SYLVESTER_H16_INT: tuple[tuple[int, ...], ...] = tuple(
    tuple(1 if ((bin(r & c).count("1") & 1) == 0) else -1 for c in range(16)) for r in range(16)
)


def verify_hadamard16_exact_q(mat: Sequence[Sequence[int]] = SLOANE_HAD_16_3_INT) -> dict[str, Any]:
    """Verify exact rational orthogonality H^T H = 16 I and H H^T = 16 I for a 16x16 +-1 Hadamard matrix."""
    require(len(mat) == 16 and all(len(row) == 16 for row in mat), "HAD16_SHAPE", "expected 16x16 matrix")
    for r in range(16):
        for c in range(16):
            require(mat[r][c] in (-1, 1), "HAD16_ENTRY", f"entry ({r},{c}) must be +-1, got {mat[r][c]}")
    max_hth_err = Fraction(0, 1)
    max_hht_err = Fraction(0, 1)
    for i in range(16):
        for j in range(16):
            target = Fraction(16, 1) if i == j else Fraction(0, 1)
            dot_col = Fraction(sum(mat[k][i] * mat[k][j] for k in range(16)), 1)
            dot_row = Fraction(sum(mat[i][k] * mat[j][k] for k in range(16)), 1)
            max_hth_err = max(max_hth_err, abs(dot_col - target))
            max_hht_err = max(max_hht_err, abs(dot_row - target))
    require(max_hth_err == 0 and max_hht_err == 0, "HAD16_NOT_ORTH", "matrix failed H^T H = 16 I")
    row0_all_one = all(v == 1 for v in mat[0])
    zero_mean_rows = sum(1 for r in range(1, 16) if sum(mat[r]) == 0)
    require(row0_all_one and zero_mean_rows == 15, "HAD16_NORM", "expected normalized first row + 15 zero-sum rows")
    return {
        "order": 16,
        "orthogonality_exact_q": True,
        "max_hth_err_q": str(max_hth_err),
        "max_hht_err_q": str(max_hht_err),
        "zero_sum_rows": zero_mean_rows,
    }


def rotate_block16_had16_3_q(
    vec16: Sequence[Fraction | int],
    *,
    inverse: bool = False,
    mat: Sequence[Sequence[int]] = SLOANE_HAD_16_3_INT,
) -> list[Fraction]:
    """Apply exact pure-Q normalized block-16 Hadamard transform y = (1/4) H x (or H^T x)."""
    require(len(vec16) == 16, "HAD16_VEC_LEN", f"expected 16 coordinates, got {len(vec16)}")
    xs = [Fraction(v) for v in vec16]
    inv_sqrt16 = Fraction(1, 4)
    if not inverse:
        return [inv_sqrt16 * sum(Fraction(mat[r][c]) * xs[c] for c in range(16)) for r in range(16)]
    return [inv_sqrt16 * sum(Fraction(mat[r][c]) * xs[r] for r in range(16)) for c in range(16)]


def minimax_simplex_margin_exact_q(
    mat_16: Sequence[Sequence[int]],
    subset: Sequence[int],
    weights_q: Sequence[Fraction] | None = None,
) -> Fraction:
    """Compute exact pure-Q subset objective -||U_S w||_inf over rows 1..15 for a given simplex point w."""
    t = len(subset)
    require(t >= 1 and len(set(subset)) == t, "HAD16_SUBSET", "subset must be non-empty distinct column indices")
    for idx in subset:
        require(0 <= int(idx) < 16, "HAD16_IDX", f"column index out of range: {idx}")
    if weights_q is None:
        w = [Fraction(1, t)] * t
    else:
        require(len(weights_q) == t, "HAD16_W_LEN", "weights_q length mismatch")
        w = [Fraction(v) for v in weights_q]
        require(all(v >= 0 for v in w) and sum(w) == Fraction(1, 1), "HAD16_SIMPLEX", "weights must lie on simplex")
    inf_norm = Fraction(0, 1)
    for r in range(1, 16):
        val = abs(sum(w[k] * Fraction(mat_16[r][subset[k]]) for k in range(t)))
        if val > inf_norm:
            inf_norm = val
    return -inf_norm


def compare_had16_vs_sylvester_t6_q() -> dict[str, Any]:
    """Compare Sloane Had.16.3 against Sylvester H16 via exact primal & dual rational certificates.

    On the canonical t=6 subset S0 = (0, 1, 2, 4, 8, 15) over rows 1..15:
      - Sylvester H16 achieves ||H_S0 w_unif||_inf = 1/3 at uniform w_unif = [1/6]*6 (valD = -1/3).
      - Sloane Had.16.3 achieves minimax primal ||H_S0 w*||_inf = 3/7 at w* = [1,1,1,1,2,1]/7
        and exact dual certificate y* supported on rows (1,2,3,9,15) with weights
        (+1/7, +1/7, +1/7, +2/7, -2/7), ||y*||_1 = 1, and (H_S0^T y*)_j == 3/7 for all j in S0,
        proving min_{w in Delta_6} ||H_S0 w||_inf == 3/7 (valD = -3/7 < -1/3).
    """
    verify_hadamard16_exact_q(SLOANE_HAD_16_3_INT)
    verify_hadamard16_exact_q(SYLVESTER_H16_INT)
    s0 = (0, 1, 2, 4, 8, 15)
    sylv_vald = minimax_simplex_margin_exact_q(SYLVESTER_H16_INT, s0)
    had3_vald_unif = minimax_simplex_margin_exact_q(SLOANE_HAD_16_3_INT, s0)
    w_star = (
        Fraction(1, 7),
        Fraction(1, 7),
        Fraction(1, 7),
        Fraction(1, 7),
        Fraction(2, 7),
        Fraction(1, 7),
    )
    had3_vald_wstar = minimax_simplex_margin_exact_q(SLOANE_HAD_16_3_INT, s0, weights_q=w_star)

    # Exact dual LP certificate y* on rows 1..15 (||y*||_1 = 1, (H_S0^T y*)_c == 3/7)
    dual_rows = {
        1: Fraction(1, 7),
        2: Fraction(1, 7),
        3: Fraction(1, 7),
        9: Fraction(2, 7),
        15: Fraction(-2, 7),
    }
    l1_dual = sum(abs(v) for v in dual_rows.values())
    dual_margins = [
        sum(yr * Fraction(SLOANE_HAD_16_3_INT[r][col]) for r, yr in dual_rows.items())
        for col in s0
    ]

    require(sylv_vald == Fraction(-1, 3), "SYLV_S0_MISMATCH", f"expected -1/3, got {sylv_vald}")
    require(had3_vald_wstar == Fraction(-3, 7), "HAD16_3_PRIMAL", f"expected -3/7, got {had3_vald_wstar}")
    require(l1_dual == Fraction(1, 1), "HAD16_3_DUAL_L1", f"expected dual L1 == 1, got {l1_dual}")
    require(
        all(m == Fraction(3, 7) for m in dual_margins),
        "HAD16_3_DUAL_GAP",
        f"expected exact 3/7 across all S0 cols, got {[str(m) for m in dual_margins]}",
    )
    require(had3_vald_wstar < sylv_vald, "HAD16_3_MARGIN", "Had.16.3 minimax value did not beat Sylvester on S0")

    # Verify exact rational round-trip on an outlier-spiked 16-vector
    probe = [Fraction(32, 1)] + [Fraction((-1) ** i, 4) for i in range(1, 16)]
    rot_q = rotate_block16_had16_3_q(probe, inverse=False)
    rec_q = rotate_block16_had16_3_q(rot_q, inverse=True)
    require(rec_q == probe, "HAD16_RT_Q", "exact rational Had.16.3 round-trip mismatch")
    max_before = max(abs(v) for v in probe)
    max_after = max(abs(v) for v in rot_q)
    require(max_after < max_before, "HAD16_SPREAD", "expected Had.16.3 to disperse outlier peak")

    return {
        "sylvester_s0_uniform_vald_q": str(sylv_vald),
        "had16_3_s0_uniform_vald_q": str(had3_vald_unif),
        "had16_3_s0_minimax_vald_q": str(had3_vald_wstar),
        "known_minimax_t6_bound_q": str(had3_vald_wstar),
        "dual_certificate_l1_q": str(l1_dual),
        "exact_q_roundtrip_pass": True,
        "outlier_linf_before_q": str(max_before),
        "outlier_linf_after_q": str(max_after),
    }


def apply_had16_3_block_f64(values: Sequence[float]) -> list[float]:
    """Rotate a vector whose length is a multiple of 16 using orthogonal 0.25 * SLOANE_HAD_16_3_INT."""
    n = len(values)
    require(n > 0 and (n % 16) == 0, "HAD16_LEN", f"length must be positive multiple of 16, got {n}")
    out = [0.0] * n
    for base in range(0, n, 16):
        for r in range(16):
            row = SLOANE_HAD_16_3_INT[r]
            acc = 0.0
            for c in range(16):
                acc += float(row[c]) * float(values[base + c])
            out[base + r] = 0.25 * acc
    return out


def apply_had16_3_inverse_f64(values: Sequence[float]) -> list[float]:
    """Invert block-16 Sloane Had.16.3 rotation using orthogonal 0.25 * H^T."""
    n = len(values)
    require(n > 0 and (n % 16) == 0, "HAD16_LEN", f"length must be positive multiple of 16, got {n}")
    out = [0.0] * n
    for base in range(0, n, 16):
        for c in range(16):
            acc = 0.0
            for r in range(16):
                acc += float(SLOANE_HAD_16_3_INT[r][c]) * float(values[base + r])
            out[base + c] = 0.25 * acc
    return out
