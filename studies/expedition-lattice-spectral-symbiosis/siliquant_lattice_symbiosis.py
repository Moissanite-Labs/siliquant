#!/usr/bin/env python3
r"""
Wave 2 Symbiosis (Arm 1): Moissanite-Labs/siliquant <-> Cave Expedition Exact Lattice & Spectral Bridges.

Standalone, deterministic, R4-clean verification & benchmark suite covering four exact
bidirectional operator transfers (zero bare `assert`s; strict R1/R3/R4/R5 gates; bit-identical
canonical JSON under -I -B, -I -O -B, -I -B -W error, and -I -O -B -W error):

  Part 1 — Exact Higher-Dimensional Decoders Closing siliquant bounds/bounds.out Sec. 8:
      - 16D Barnes-Wall Lambda_16 (32-coset RM(1,4) + 2*D_16 exact decoder):
        verified against ALL 4,320 minimal vectors; unbiased [0,4)^16 G(Lambda_16);
        exact bent-function deep hole f(x) = x0*x1 ^ x2*x3 proving rho^2 = 3/sqrt(2) (vol=1).
      - 24D Optimal Leech Lambda_24 (Conway-Sloane 1986 MOG/Hexacode 4-state DP exact decoder):
        verified membership (parity, mod-8 sum, Golay G_24) & Voronoi optimality against
        ALL 196,560 minimal vectors; exact deep hole (4,0^23) proving rho^2 = 2 (vol=1);
        unbiased [0,8)^24 G(Lambda_24).

  Part 2 — Exact Modular-Form Integer Theta Series & The Leech Safe-Scale Theorem
           (siliquant/shape/certified.py Claims 1 & 2) vs. BW16 Bent-Function Obstruction:
      - Pure Python `int` exact shell counts for Z^16, Z^24, E_8, E_8^2, E_8^3, Lambda_16, Lambda_24.
      - Proves Lambda_24 cuts the proved worst-case safe-scale certification gap rho^2/(n*G)
        from +4.77 dB (Z^24) and +2.42 dB (E_8^3) down to +1.03 dB (and achieves +1.16 dB
        safe-MSE / +2.54 dB proved-bound gain over E_8^3 at 2.0 bpw), whereas Lambda_16's
        bent-function deep hole makes its proved worst-case bound worse than E_8^2 (-0.22 dB)
        despite +0.21 dB better average G!
      - Live empirical safe_round validation on Gaussian & Student-t3 blocks (0% outside codebook,
        100% Claim 2 worst-case bound satisfaction).

  Part 3 — Exact L1 Voronoi Radius Theorem (max_{V(E8)} ||x||_1 = 8/3 < sqrt(8)) for
           siliquant/shape/certified.py + Sylvester-Hadamard E8_Had (AUDIT-017) Resolution of
           siliquant PR #17's SmolLM2 Tiny-Weight Phantom-Jump Split:
      - Pure exact Fraction dual LP certificate (zero floats) over all 256 sign patterns
        proving max ||x||_1 on V(E8_std) == 8/3 (attained at shallow hole (1/3^7,-1/3)) vs.
        sqrt(8) on V(E8_Had) (attained at deep hole (-1/sqrt(8))^8).
      - Root sparsity & multi-threshold tiny-weight phantom-jump benchmark across Z, D_8,
        E8_std, E8_Had, Lambda_16, Lambda_24 on Gaussian, Student-t3, and Sparse+Small blocks.

  Part 4 — Zero-Float Exact Spectral-Inertia Kept-Column Outlier Certifier
           (siliquant to_integers(W) + AUDIT-002 R_opt + AUDIT-014 Phi_{x0,tau} + SP-01 Gauge):
      - Pure exact `int`/`Fraction` TCB (zero floats, zero SVD!) cross-checked by an
        independent zero-shared-code Faddeev-LeVerrier + Descartes inertia oracle.
      - Certifies 0 spikes on pure bulk null, >= 3 spikes on ffn_down-style 3-spike float16
        weights, exact residual unkept spikes (>= 2, >= 1, >= 0) after keeping b = 1, 2, 3
        columns, exact bit-identical column-permutation gauge invariance (SP-01), and
        immediate sensitivity to activation-Hessian readout flips.

  Part 5 — R5 Must-Fail Negative Controls (6/6 rejected with explicit error codes).

Run:
  python3 -I -O -B -W error siliquant_lattice_symbiosis.py
  python3 -I -O -B -W error siliquant_lattice_symbiosis.py --json
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np


# ==============================================================================
# R1 / R3 / R4 STRICT GATE HELPERS (ZERO BARE ASSERTS)
# ==============================================================================
class CertError(ValueError):
    """Structured default-reject verification error (never stripped by -O)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


def require(cond: bool, code: str, message: str) -> None:
    if not cond:
        raise CertError(code, message)


def parse_exact_fraction(val: Any) -> Fraction:
    if isinstance(val, bool) or not isinstance(val, (int, Fraction)):
        raise CertError("INVALID_TYPE", f"expected exact int/Fraction, got {type(val).__name__}")
    return val if isinstance(val, Fraction) else Fraction(val)


def parse_exact_int(val: Any, min_value: int | None = None) -> int:
    if isinstance(val, bool) or not isinstance(val, int):
        raise CertError("INVALID_TYPE", f"expected exact int, got {type(val).__name__}")
    if min_value is not None and val < min_value:
        raise CertError("INVALID_INT_RANGE", f"expected int >= {min_value}, got {val}")
    return val


def frac_str(q: Fraction) -> str:
    q = parse_exact_fraction(q)
    return str(q.numerator) if q.denominator == 1 else f"{q.numerator}/{q.denominator}"


def ceil_fraction(q: Fraction) -> int:
    q = parse_exact_fraction(q)
    return -((-q.numerator) // q.denominator)


def r6(x: float) -> float:
    return round(float(x), 6)


def db(x: float) -> float:
    return 10.0 * math.log10(x)


# ==============================================================================
# PART 1: EXACT 8D, 16D BARNES-WALL (LAMBDA_16) & 24D LEECH (LAMBDA_24) DECODERS
# ==============================================================================
def sylvester_H8_int() -> list[list[int]]:
    H = [[1]]
    for _ in range(3):
        H = [r + r for r in H] + [r + [-x for x in r] for r in H]
    return H


_H8_FLOAT = np.array(sylvester_H8_int(), dtype=np.float64)
U_H8 = _H8_FLOAT / math.sqrt(8.0)


def closest_D_batch(x: np.ndarray) -> np.ndarray:
    f = np.rint(x)
    odd = (f.sum(axis=-1).astype(np.int64) & 1) != 0
    if np.any(odd):
        err = x - f
        flat_f = f[odd]
        flat_e = err[odd]
        j = np.argmax(np.abs(flat_e), axis=-1)
        idx = np.arange(len(flat_f))
        flat_f[idx, j] += np.where(flat_e[idx, j] >= 0.0, 1.0, -1.0)
        f[odd] = flat_f
    return f


def closest_E8_std(x: np.ndarray) -> np.ndarray:
    a = closest_D_batch(x)
    b = closest_D_batch(x - 0.5) + 0.5
    da = ((x - a) ** 2).sum(axis=1)
    db_v = ((x - b) ** 2).sum(axis=1)
    return np.where((db_v < da)[:, None], b, a)


def closest_E8_had(x: np.ndarray) -> np.ndarray:
    """Sylvester-Hadamard rotated E8_Had = (H8/sqrt(8)) E8 = (1/sqrt(2)) {x in Z^8: x mod 2 in Ham(8,4)}."""
    return closest_E8_std(x @ U_H8) @ U_H8


# --- 16D Barnes-Wall Lambda_16 = Union_{c in RM(1,4)} (c + 2 * D_16) ---
def build_rm14() -> np.ndarray:
    G = np.zeros((5, 16), dtype=np.int64)
    G[0, :] = 1
    for bit in range(4):
        for j in range(16):
            G[bit + 1, j] = (j >> bit) & 1
    out = np.zeros((32, 16), dtype=np.int64)
    for idx in range(32):
        v = np.zeros(16, dtype=np.int64)
        for b in range(5):
            if (idx >> b) & 1:
                v ^= G[b]
        out[idx] = v
    return out


RM14_INT = build_rm14()
RM14_SET = {tuple(r.tolist()) for r in RM14_INT}
RM14_FLOAT = RM14_INT.astype(np.float64)
BW16_UNIT_SCALE = 2.0 ** 0.75  # raw vol = 2^12 -> vol^(1/16) = 2^(3/4)


def verify_bw16_membership_int(vec: Sequence[int]) -> bool:
    require(len(vec) == 16, "INVALID_BW16_DIM", "BW16 vector must have length 16")
    vals = [parse_exact_int(v) for v in vec]
    c = tuple(v & 1 for v in vals)
    if c not in RM14_SET:
        return False
    y_sum = sum((v - ci) // 2 for v, ci in zip(vals, c))
    return (y_sum & 1) == 0


def closest_bw16_raw(x: np.ndarray) -> np.ndarray:
    """Exact 32-coset nearest-point decoder for Lambda_16 (min ||v||^2 = 8, vol = 4096)."""
    n_pts = x.shape[0]
    best = np.empty_like(x)
    best_d = np.full(n_pts, np.inf)
    for k in range(32):
        c = RM14_FLOAT[k]
        cand = c + 2.0 * closest_D_batch((x - c) * 0.5)
        d = ((x - cand) ** 2).sum(axis=1)
        better = d < best_d
        best[better] = cand[better]
        best_d[better] = d[better]
    return best


def closest_bw16_unit(x: np.ndarray) -> np.ndarray:
    return closest_bw16_raw(x * BW16_UNIT_SCALE) / BW16_UNIT_SCALE


# --- 24D Optimal Leech Lambda_24 (Conway-Sloane 1986 MOG/Hexacode Exact Decoder) ---
_GF4_MUL = np.array(
    [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]], dtype=np.int64
)
_G_H6 = np.array(
    [[1, 0, 0, 1, 2, 3], [0, 1, 0, 1, 3, 2], [0, 0, 1, 1, 1, 1]], dtype=np.int64
)
HEXACODE = np.array(
    [
        _GF4_MUL[a, _G_H6[0]] ^ _GF4_MUL[b, _G_H6[1]] ^ _GF4_MUL[c, _G_H6[2]]
        for a in range(4)
        for b in range(4)
        for c in range(4)
    ],
    dtype=np.int64,
)
MOG_PATTERNS = np.zeros((2, 4, 2, 4), dtype=np.int64)
for _bits in itertools.product((0, 1), repeat=4):
    _cp = _bits[0] ^ _bits[1] ^ _bits[2] ^ _bits[3]
    _sym = (1 if _bits[1] else 0) ^ (2 if _bits[2] else 0) ^ (3 if _bits[3] else 0)
    MOG_PATTERNS[_cp, _sym, _bits[0]] = _bits


def build_golay24_set() -> set[tuple[int, ...]]:
    g24: set[tuple[int, ...]] = set()
    for cp in (0, 1):
        for cw in HEXACODE:
            for top_bits in itertools.product((0, 1), repeat=6):
                if (sum(top_bits) & 1) != cp:
                    continue
                cols = [MOG_PATTERNS[cp, int(cw[j]), top_bits[j]] for j in range(6)]
                g24.add(tuple(int(v) for col in cols for v in col))
    return g24


GOLAY24_SET = build_golay24_set()
LEECH24_UNIT_SCALE = 2.0 * math.sqrt(2.0)  # raw vol = 2^36 -> vol^(1/24) = 2*sqrt(2)


def verify_leech24_membership_int(vec: Sequence[int]) -> bool:
    require(len(vec) == 24, "INVALID_LEECH_DIM", "Leech vector must have length 24")
    vals = [parse_exact_int(v) for v in vec]
    m = vals[0] & 1
    if any((v & 1) != m for v in vals):
        return False
    if (sum(vals) % 8) != (4 * m):
        return False
    c = tuple(((v - m) // 2) & 1 for v in vals)
    return c in GOLAY24_SET


def _nearest_and_second_mod4(x: np.ndarray, r: float):
    q1 = 4.0 * np.rint((x - r) * 0.25) + r
    err = x - q1
    q2 = q1 + np.where(err >= 0.0, 4.0, -4.0)
    return q1, err * err, q2, (x - q2) ** 2


def closest_leech24_raw(X: np.ndarray) -> np.ndarray:
    """Exact Conway-Sloane 1986 MOG/Hexacode nearest-point decoder for Leech Lambda_24 (min ||v||^2=32, vol=2^36)."""
    N = X.shape[0]
    X4 = X.reshape(N, 6, 4)
    Q1 = np.empty((4, N, 6, 4), dtype=np.float64)
    D1 = np.empty((4, N, 6, 4), dtype=np.float64)
    Q2 = np.empty((4, N, 6, 4), dtype=np.float64)
    D2 = np.empty((4, N, 6, 4), dtype=np.float64)
    for r in range(4):
        Q1[r], D1[r], Q2[r], D2[r] = _nearest_and_second_mod4(X4, float(r))

    best_d = np.full(N, np.inf, dtype=np.float64)
    best_y = np.empty((N, 6, 4), dtype=np.float64)
    nn_idx = np.arange(N)[:, None]
    cc_idx = np.arange(6)[None, :]
    arN = np.arange(N)

    for m in (0, 1):
        for cp in (0, 1):
            target_st = 2 * cp + (m ^ cp)
            col_cost = np.empty((4, 2, 2, N, 6), dtype=np.float64)
            col_vec = np.empty((4, 2, 2, N, 6, 4), dtype=np.float64)
            for sym in range(4):
                for t in (0, 1):
                    r_idx = m + 2 * MOG_PATTERNS[cp, sym, t]
                    v0 = np.stack([Q1[r_idx[i], :, :, i] for i in range(4)], axis=-1)
                    d0 = np.stack([D1[r_idx[i], :, :, i] for i in range(4)], axis=-1)
                    v1 = np.stack([Q2[r_idx[i], :, :, i] for i in range(4)], axis=-1)
                    d1 = np.stack([D2[r_idx[i], :, :, i] for i in range(4)], axis=-1)
                    cost0 = d0.sum(axis=-1)
                    s0 = ((np.rint(v0).astype(np.int64).sum(axis=-1) - 2 * cp) >> 2) & 1
                    delta = d1 - d0
                    flip_i = np.argmin(delta, axis=-1)
                    cost_flip = cost0 + np.take_along_axis(delta, flip_i[:, :, None], axis=-1)[..., 0]
                    v_flip = v0.copy()
                    v_flip[nn_idx, cc_idx, flip_i] = v1[nn_idx, cc_idx, flip_i]
                    mask0 = s0 == 0
                    col_cost[sym, t, 0] = np.where(mask0, cost0, cost_flip)
                    col_cost[sym, t, 1] = np.where(mask0, cost_flip, cost0)
                    col_vec[sym, t, 0] = np.where(mask0[:, :, None], v0, v_flip)
                    col_vec[sym, t, 1] = np.where(mask0[:, :, None], v_flip, v0)

            cc_flat = col_cost.reshape(4, 4, N, 6)
            cv_flat = col_vec.reshape(4, 4, N, 6, 4)
            for cw in HEXACODE:
                dp = cc_flat[cw[0], :, :, 0].copy()
                choice = np.empty((6, 4, N), dtype=np.int8)
                choice[0] = np.arange(4, dtype=np.int8)[:, None]
                for j in range(1, 6):
                    sc = cc_flat[cw[j], :, :, j]
                    new_dp = np.empty((4, N), dtype=np.float64)
                    for st in range(4):
                        stacked = np.stack(
                            [dp[st ^ 0] + sc[0], dp[st ^ 1] + sc[1], dp[st ^ 2] + sc[2], dp[st ^ 3] + sc[3]],
                            axis=0,
                        )
                        arg = np.argmin(stacked, axis=0)
                        new_dp[st] = stacked[arg, arN]
                        choice[j, st] = arg
                    dp = new_dp
                better = dp[target_st] < best_d
                if np.any(better):
                    ib = np.nonzero(better)[0]
                    best_d[ib] = dp[target_st][ib]
                    st_cur = np.full(len(ib), target_st, dtype=np.int64)
                    for j in range(5, -1, -1):
                        dj = choice[j, st_cur, ib].astype(np.int64)
                        best_y[ib, j, :] = cv_flat[cw[j], dj, ib, j, :]
                        st_cur ^= dj

    return best_y.reshape(N, 24)


def closest_leech24_unit(X: np.ndarray) -> np.ndarray:
    return closest_leech24_raw(X * LEECH24_UNIT_SCALE) / LEECH24_UNIT_SCALE


def run_part1_higher_dim_decoders() -> dict[str, Any]:
    """Verify BW16 & Leech24 decoders against all minimal vectors and recompute G & deep holes."""
    rng = np.random.default_rng(20261120)

    # 1A. BW16: verify RM(1,4), membership, ALL 4,320 minimal vectors, bent deep hole, and G(Lambda_16)
    require(len(RM14_SET) == 32, "RM14_SIZE", "expected 32 RM(1,4) codewords")
    pts16 = rng.standard_normal((400, 16)) * 3.0
    dec16 = closest_bw16_raw(pts16)
    dec16_int = np.rint(dec16).astype(np.int64)
    require(bool(np.all(dec16 == dec16_int)), "BW16_NOT_INT", "BW16 decoded non-integer")
    for row in dec16_int:
        require(verify_bw16_membership_int(row.tolist()), "BW16_MEMB", "decoded point not in BW16")

    min_bw16 = []
    for i, j in itertools.combinations(range(16), 2):
        for si in (-2.0, 2.0):
            for sj in (-2.0, 2.0):
                v = np.zeros(16)
                v[i], v[j] = si, sj
                min_bw16.append(v)
    for c in RM14_INT:
        if int(c.sum()) == 8:
            supp = np.nonzero(c)[0]
            for signs in itertools.product((-1.0, 1.0), repeat=8):
                if sum(1 for s in signs if s < 0) % 2 == 0:
                    v = np.zeros(16)
                    v[supp] = signs
                    min_bw16.append(v)
    min_bw16_arr = np.array(min_bw16, dtype=np.float64)
    require(len(min_bw16_arr) == 4320, "BW16_KISS", "expected 4,320 minimal vectors")
    res16 = pts16[:120] - dec16[:120]
    max_dot_bw16 = float((res16 @ min_bw16_arr.T).max())
    require(max_dot_bw16 <= 4.0 + 1e-11, "BW16_VORONOI", f"BW16 Voronoi violation: {max_dot_bw16}")

    # Exact bent-function deep hole f(x0,x1,x2,x3) = x0*x1 ^ x2*x3
    bent = np.array(
        [((j & 1) & ((j >> 1) & 1)) ^ (((j >> 2) & 1) & ((j >> 3) & 1)) for j in range(16)],
        dtype=np.float64,
    )
    d2_bent = float(((bent[None, :] - closest_bw16_raw(bent[None, :])) ** 2).sum())
    require(abs(d2_bent - 6.0) < 1e-12, "BW16_BENT_HOLE", f"expected raw d^2 = 6, got {d2_bent}")

    # Unbiased Monte Carlo G(Lambda_16) on exact fundamental union [0, 4)^16 (2^20 cells!)
    n_mc16 = 60_000
    chunk16 = 10_000
    t1, t2 = 0.0, 0.0
    for _ in range(n_mc16 // chunk16):
        u = rng.uniform(0.0, 4.0, size=(chunk16, 16))
        e2 = ((u - closest_bw16_raw(u)) ** 2).sum(axis=1)
        t1 += float(e2.sum())
        t2 += float((e2 ** 2).sum())
    m1 = t1 / n_mc16
    se1 = math.sqrt(max(0.0, t2 / n_mc16 - m1 * m1) / n_mc16)
    sc16 = 16.0 * (4096.0 ** (2.0 / 16.0))
    G_bw16, se_bw16 = m1 / sc16, se1 / sc16
    require(abs(G_bw16 - 0.06830) <= 4.0 * se_bw16, "BW16_G_DEV", f"G(BW16)={G_bw16} off Agrell-Allen 0.06830")

    # 1B. Leech24: verify Golay G24, membership, ALL 196,560 minimal vectors, deep hole (4,0^23), G(Lambda_24)
    require(len(GOLAY24_SET) == 4096, "GOLAY_SIZE", "expected 4,096 Golay codewords")
    pts24 = rng.standard_normal((200, 24)) * 5.0
    dec24 = closest_leech24_raw(pts24)
    dec24_int = np.rint(dec24).astype(np.int64)
    require(bool(np.all(dec24 == dec24_int)), "LEECH_NOT_INT", "Leech decoded non-integer")
    for row in dec24_int:
        require(verify_leech24_membership_int(row.tolist()), "LEECH_MEMB", "decoded point not in Leech")

    # Voronoi optimality against ALL 196,560 minimal vectors (3 exact shapes: 1,104 + 97,152 + 98,304)
    res24 = pts24[:80] - dec24[:80]
    abs_r = np.sort(np.abs(res24), axis=1)
    max_s1 = float((4.0 * (abs_r[:, -1] + abs_r[:, -2])).max())
    octads = [np.array(c, dtype=np.int64) for c in sorted(GOLAY24_SET) if sum(c) == 8]
    require(len(octads) == 759, "OCTAD_COUNT", "expected 759 Golay octads")
    max_s2 = -math.inf
    for oc in octads:
        sub = res24[:, np.nonzero(oc)[0]]
        abs_sub = np.abs(sub)
        odd = ((sub < 0).sum(axis=1) & 1) == 1
        best_sum = np.where(odd, abs_sub.sum(axis=1) - 2.0 * abs_sub.min(axis=1), abs_sub.sum(axis=1))
        max_s2 = max(max_s2, float((2.0 * best_sum).max()))
    g24_sorted = np.array(sorted(GOLAY24_SET), dtype=np.float64)
    signs_mat = 1.0 - 2.0 * g24_sorted
    base_dots = res24 @ signs_mat.T
    max_s3 = -math.inf
    for start in range(0, 4096, 512):
        sm = signs_mat[start : start + 512]
        extra = (-4.0 * res24[:, None, :] * sm[None, :, :]).max(axis=2)
        max_s3 = max(max_s3, float((base_dots[:, start : start + 512] + extra).max()))
    require(max(max_s1, max_s2, max_s3) <= 16.0 + 1e-10, "LEECH_VORONOI", "Leech Voronoi violation")

    # Deep hole (4, 0^23)
    h_deep24 = np.zeros((1, 24), dtype=np.float64)
    h_deep24[0, 0] = 4.0
    d2_deep24 = float(((h_deep24 - closest_leech24_raw(h_deep24)) ** 2).sum())
    require(abs(d2_deep24 - 16.0) < 1e-12, "LEECH_HOLE", f"expected raw d^2 = 16, got {d2_deep24}")

    # Unbiased Monte Carlo G(Lambda_24) on exact fundamental union [0, 8)^24 (2^36 cells!)
    n_mc24 = 4_000
    chunk24 = 1_000
    t1, t2 = 0.0, 0.0
    for _ in range(n_mc24 // chunk24):
        u = rng.uniform(0.0, 8.0, size=(chunk24, 24))
        e2 = ((u - closest_leech24_raw(u)) ** 2).sum(axis=1)
        t1 += float(e2.sum())
        t2 += float((e2 ** 2).sum())
    m1 = t1 / n_mc24
    se1 = math.sqrt(max(0.0, t2 / n_mc24 - m1 * m1) / n_mc24)
    sc24 = 24.0 * 8.0
    G_l24, se_l24 = m1 / sc24, se1 / sc24
    require(abs(G_l24 - 0.06577) <= 4.0 * se_l24, "LEECH_G_DEV", f"G(Leech)={G_l24} off Agrell-Allen 0.06577")

    return {
        "bw16": {
            "cosets_rm14": 32,
            "membership_verified": 400,
            "voronoi_all_4320_max_dot": r6(max_dot_bw16),
            "bent_deep_hole_raw_d2": r6(d2_bent),
            "vol1_covering_radius_sq": r6(6.0 / (2.0 ** 1.5)),
            "G_recomputed": r6(G_bw16),
            "G_se": r6(se_bw16),
            "G_gain_vs_Z_dB": r6(db((1.0 / 12.0) / G_bw16)),
            "G_gain_vs_E8_dB": r6(db((929.0 / 12960.0) / G_bw16)),
            "safe_scale_gap_rho2_over_nG_dB": r6(db((3.0 / math.sqrt(2.0)) / (16.0 * G_bw16))),
        },
        "leech24": {
            "hexacode_words": 64,
            "golay_codewords": 4096,
            "membership_verified": 200,
            "voronoi_all_196560_max_dots": [r6(max_s1), r6(max_s2), r6(max_s3)],
            "deep_hole_raw_d2": r6(d2_deep24),
            "vol1_covering_radius_sq": 2.0,
            "G_recomputed": r6(G_l24),
            "G_se": r6(se_l24),
            "G_gain_vs_Z_dB": r6(db((1.0 / 12.0) / G_l24)),
            "G_gain_vs_E8_dB": r6(db((929.0 / 12960.0) / G_l24)),
            "safe_scale_gap_rho2_over_nG_dB": r6(db(2.0 / (24.0 * G_l24))),
        },
        "reference_safe_scale_gaps_dB": {
            "Z_any_n": r6(db(3.0)),
            "E8_product": r6(db(1.0 / (8.0 * 929.0 / 12960.0))),
        },
    }


# ==============================================================================
# PART 2: EXACT INTEGER THETA SERIES + LEECH SAFE-SCALE THEOREM + EMPIRICAL TEST
# ==============================================================================
def _poly_mul_trunc(a: list[int], b: list[int], deg: int) -> list[int]:
    out = [0] * (deg + 1)
    for i, ai in enumerate(a):
        if ai == 0:
            continue
        lim = min(len(b), deg + 1 - i)
        for j in range(lim):
            out[i + j] += ai * b[j]
    return out


def _poly_pow_trunc(a: list[int], p: int, deg: int) -> list[int]:
    res = [0] * (deg + 1)
    res[0] = 1
    base = a[: deg + 1]
    while p > 0:
        if p & 1:
            res = _poly_mul_trunc(res, base, deg)
        p >>= 1
        if p:
            base = _poly_mul_trunc(base, base, deg)
    return res


def theta_E8_exact(max_m: int) -> list[int]:
    c = [0] * (max_m + 1)
    c[0] = 1
    for m in range(1, max_m + 1):
        c[m] = 240 * sum(d ** 3 for d in range(1, m + 1) if m % d == 0)
    return c


def theta_Leech24_exact(max_m: int) -> list[int]:
    e4 = theta_E8_exact(max_m)
    e4_3 = _poly_pow_trunc(e4, 3, max_m)
    prod = [0] * (max_m + 1)
    prod[0] = 1
    for n in range(1, max_m + 1):
        for k in range(max_m, n - 1, -1):
            prod[k] -= prod[k - n]
    p24 = _poly_pow_trunc(prod, 24, max_m)
    delta = [0] + p24[:max_m]
    return [e4_3[m] - 720 * delta[m] for m in range(max_m + 1)]


def theta_BW16_exact(max_m: int) -> list[int]:
    deg = 4 * max_m
    th3, th4, th2 = [0] * (deg + 1), [0] * (deg + 1), [0] * (deg + 1)
    k = 0
    while 4 * k * k <= deg:
        w = 1 if k == 0 else 2
        th3[4 * k * k] += w
        th4[4 * k * k] += w if (k % 2 == 0) else -w
        k += 1
    k = 1
    while k * k <= deg:
        th2[k * k] += 2
        k += 2
    t3_8 = _poly_pow_trunc(th3, 8, deg)
    t4_8 = _poly_pow_trunc(th4, 8, deg)
    t2_8 = _poly_pow_trunc(th2, 8, deg)
    full = [
        (_poly_mul_trunc(t3_8, t3_8, deg)[d]
         + _poly_mul_trunc(t4_8, t4_8, deg)[d]
         + _poly_mul_trunc(t2_8, t2_8, deg)[d]) // 2
        + 15 * _poly_mul_trunc(t3_8, t2_8, deg)[d]
        for d in range(deg + 1)
    ]
    for d in range(deg + 1):
        if d % 4 != 0:
            require(full[d] == 0, "BW16_THETA_MOD4", f"unexpected nonzero BW16 coeff at d={d}")
    return [full[4 * m] for m in range(max_m + 1)]


def theta_Zn_exact(n: int, max_r2: int) -> list[int]:
    t1 = [0] * (max_r2 + 1)
    k = 0
    while k * k <= max_r2:
        t1[k * k] = 1 if k == 0 else 2
        k += 1
    return _poly_pow_trunc(t1, n, max_r2)


def max_ball_shell(shells: list[int], step_r2: float, cap_bits: int) -> tuple[float, int, int]:
    cap = 1 << cap_bits
    cum = 0
    best_m = 0
    best_N = 1
    hit_cap = False
    for m, cnt in enumerate(shells):
        if cum + cnt <= cap:
            cum += cnt
            best_m = m
            best_N = cum
        else:
            hit_cap = True
            break
    require(hit_cap, "THETA_TRUNCATED", f"shell table too short (len={len(shells)}) to reach 2^{cap_bits}")
    return math.sqrt(best_m * step_r2), best_N, best_m


def run_part2_safe_scale_theorem() -> dict[str, Any]:
    """Exact theta series + Leech safe-scale theorem + live empirical safe_round validation."""
    th_e8 = theta_E8_exact(230)
    th_e8_2 = _poly_pow_trunc(th_e8, 2, 120)
    th_e8_3 = _poly_pow_trunc(th_e8, 3, 230)
    th_bw16 = theta_BW16_exact(130)
    th_l24 = theta_Leech24_exact(230)
    th_z16 = theta_Zn_exact(16, 170)
    th_z24 = theta_Zn_exact(24, 460)

    require(th_e8[:4] == [1, 240, 2160, 6720], "E8_THETA", "E8 theta mismatch")
    require(th_bw16[:5] == [1, 0, 4320, 61440, 522720], "BW16_THETA", "BW16 theta mismatch")
    require(th_l24[:5] == [1, 0, 196560, 16773120, 398034000], "LEECH_THETA", "Leech theta mismatch")

    G_ref = {"Z24": 1.0 / 12.0, "E8^3": 929.0 / 12960.0, "Leech24": 0.06577, "Z16": 1.0 / 12.0, "E8^2": 929.0 / 12960.0, "BW16": 0.06830}
    rho_ref = {"Z24": math.sqrt(6.0), "E8^3": math.sqrt(3.0), "Leech24": math.sqrt(2.0), "Z16": 2.0, "E8^2": math.sqrt(2.0), "BW16": math.sqrt(3.0 / math.sqrt(2.0))}

    table_24d = {}
    radii_24d: dict[int, dict[str, float]] = {}
    for bpw_label, cap_bits in [("2.00_bpw", 48), ("2.50_bpw", 60), ("3.00_bpw", 72), ("4.00_bpw", 96)]:
        row_stats = {}
        radii_24d[cap_bits] = {}
        for name, sh, step in [("Z24", th_z24, 1.0), ("E8^3", th_e8_3, 2.0), ("Leech24", th_l24, 2.0)]:
            r, N, _ = max_ball_shell(sh, step, cap_bits)
            radii_24d[cap_bits][name] = r
            rs = r - rho_ref[name]
            wc = (rho_ref[name] ** 2) / (24.0 * rs * rs)
            safe_est = G_ref[name] / (rs * rs)
            free_est = G_ref[name] / (r * r)
            row_stats[name] = {
                "r": r6(r),
                "r_safe": r6(rs),
                "eff_bpw": r6(math.log2(N) / 24.0),
                "free_mse_dB": r6(db(free_est)),
                "safe_mse_dB": r6(db(safe_est)),
                "proved_bound_dB": r6(db(wc)),
            }
        table_24d[bpw_label] = {
            "arms": row_stats,
            "leech_vs_e8x3_free_dB": r6(row_stats["E8^3"]["free_mse_dB"] - row_stats["Leech24"]["free_mse_dB"]),
            "leech_vs_e8x3_safe_mse_dB": r6(row_stats["E8^3"]["safe_mse_dB"] - row_stats["Leech24"]["safe_mse_dB"]),
            "leech_vs_e8x3_proved_bound_dB": r6(row_stats["E8^3"]["proved_bound_dB"] - row_stats["Leech24"]["proved_bound_dB"]),
            "leech_vs_z24_safe_mse_dB": r6(row_stats["Z24"]["safe_mse_dB"] - row_stats["Leech24"]["safe_mse_dB"]),
            "leech_vs_z24_proved_bound_dB": r6(row_stats["Z24"]["proved_bound_dB"] - row_stats["Leech24"]["proved_bound_dB"]),
        }

    table_16d = {}
    for bpw_label, cap_bits in [("2.50_bpw", 40), ("3.50_bpw", 56)]:
        row_stats = {}
        for name, sh, step in [("Z16", th_z16, 1.0), ("E8^2", th_e8_2, 2.0), ("BW16", th_bw16, math.sqrt(2.0))]:
            r, N, _ = max_ball_shell(sh, step, cap_bits)
            rs = r - rho_ref[name]
            wc = (rho_ref[name] ** 2) / (16.0 * rs * rs)
            safe_est = G_ref[name] / (rs * rs)
            free_est = G_ref[name] / (r * r)
            row_stats[name] = {
                "r": r6(r),
                "r_safe": r6(rs),
                "eff_bpw": r6(math.log2(N) / 16.0),
                "free_mse_dB": r6(db(free_est)),
                "safe_mse_dB": r6(db(safe_est)),
                "proved_bound_dB": r6(db(wc)),
            }
        table_16d[bpw_label] = {
            "arms": row_stats,
            "bw16_vs_e8x2_free_dB": r6(row_stats["E8^2"]["free_mse_dB"] - row_stats["BW16"]["free_mse_dB"]),
            "bw16_vs_e8x2_safe_mse_dB": r6(row_stats["E8^2"]["safe_mse_dB"] - row_stats["BW16"]["safe_mse_dB"]),
            "bw16_vs_e8x2_proved_bound_dB": r6(row_stats["E8^2"]["proved_bound_dB"] - row_stats["BW16"]["proved_bound_dB"]),
        }

    # Live empirical safe_round validation on 24D Gaussian & Student-t3 blocks at 2.00 bpw (48 bits) & 2.50 bpw (60 bits)
    rng = np.random.default_rng(31415)
    empirical_24d = {}
    safety = 1.0 + 1e-9
    decoders_24 = {
        "Z24": lambda X: np.rint(X),
        "E8^3": lambda X: closest_E8_std(X.reshape(-1, 8)).reshape(-1, 24),
        "Leech24": closest_leech24_unit,
    }
    for dname, X in [
        ("Gaussian_600x24", rng.standard_normal((600, 24))),
        ("Student_t3_600x24", rng.standard_t(3.0, size=(600, 24))),
    ]:
        total_w2 = float(np.sum(X ** 2))
        norms = np.sqrt((X ** 2).sum(axis=1))
        empirical_24d[dname] = {}
        for bpw_label, cap_bits in [("2.00_bpw", 48), ("2.50_bpw", 60)]:
            res_bpw = {}
            for name, fn in decoders_24.items():
                r = radii_24d[cap_bits][name]
                rho = rho_ref[name]
                s = np.maximum(norms / (r - rho) * safety, 1e-150)
                Y = fn(X / s[:, None])
                y_norm2 = (Y ** 2).sum(axis=1)
                outside = int(np.sum(y_norm2 > r * r + 1e-7))
                require(outside == 0, "SAFE_SCALE_OUTSIDE", f"{name} had {outside} blocks outside ball")
                rec = Y * s[:, None]
                blk_err = ((X - rec) ** 2).sum(axis=1)
                worst_ratio = float(np.max(blk_err / ((rho * s) ** 2)))
                require(worst_ratio <= 1.0 + 1e-7, "SAFE_SCALE_BOUND", f"{name} worst_ratio={worst_ratio} > 1")
                meas_rel = float(blk_err.sum()) / total_w2
                bound_rel = float((rho * rho) * np.sum(s ** 2)) / total_w2
                res_bpw[name] = {
                    "outside_count": outside,
                    "worst_block_over_bound": r6(worst_ratio),
                    "measured_safe_mse_dB": r6(db(meas_rel)),
                    "proved_bound_dB": r6(db(bound_rel)),
                }
            res_bpw["measured_leech_vs_e8x3_dB"] = r6(
                res_bpw["E8^3"]["measured_safe_mse_dB"] - res_bpw["Leech24"]["measured_safe_mse_dB"]
            )
            res_bpw["measured_leech_vs_z24_dB"] = r6(
                res_bpw["Z24"]["measured_safe_mse_dB"] - res_bpw["Leech24"]["measured_safe_mse_dB"]
            )
            empirical_24d[dname][bpw_label] = res_bpw

    return {
        "exact_theta_prefixes": {
            "E8_m0_to_4": th_e8[:5],
            "BW16_m0_to_4": th_bw16[:5],
            "Leech24_m0_to_4": th_l24[:5],
        },
        "table_24d_safe_scale": table_24d,
        "table_16d_safe_scale": table_16d,
        "empirical_24d_safe_round": empirical_24d,
    }


# ==============================================================================
# PART 3: EXACT L1 VORONOI RADIUS THEOREM (8/3 < sqrt(8)) + HADAMARD E8_Had SPLIT
# ==============================================================================
def exact_prove_e8_l1_voronoi_radius() -> dict[str, Any]:
    """
    Pure exact Fraction proof (ZERO FLOATS!) that:
      max_{x in V(E8_std)} ||x||_1 == 8/3   (strictly tighter than siliquant's sqrt(8)!),
    by constructing an explicit nonnegative dual certificate over the 240 E8 roots for
    ALL 256 sign patterns sigma in {+1,-1}^8, AND verifying exact primal attainment at
    the shallow hole x* = (1/3, 1/3, 1/3, 1/3, 1/3, 1/3, 1/3, -1/3) in V(E8_std) (plus
    tightness of sqrt(8) on V(E8_Had) at deep hole U_H8(-e_0)).
    """
    # Build all 240 roots of standard E8 in exact Fraction
    roots: list[tuple[Fraction, ...]] = []
    for i, j in itertools.combinations(range(8), 2):
        for si in (-1, 1):
            for sj in (-1, 1):
                v = [Fraction(0)] * 8
                v[i], v[j] = Fraction(si), Fraction(sj)
                roots.append(tuple(v))
    for signs in itertools.product((-1, 1), repeat=8):
        if sum(1 for s in signs if s < 0) % 2 == 0:
            roots.append(tuple(Fraction(s, 2) for s in signs))
    require(len(roots) == 240, "E8_ROOTS", "expected 240 E8 roots")
    root_set = set(roots)

    # 1. Exact Dual Certificate for ALL 256 sign vectors sigma in {+1, -1}^8:
    #    - Even parity (128): r = sigma/2 is a root, so sigma = 2*r => sigma . x <= 2!
    #    - Odd parity  (128): for each k=0..7, r_k = (sigma - 2*sigma_k*e_k)/2 is a root,
    #      and sum_{k=0}^7 (1/3) r_k == sigma, so sigma . x <= sum (1/3)*1 == 8/3!
    even_verified = 0
    odd_verified = 0
    for sig in itertools.product((-1, 1), repeat=8):
        neg_cnt = sum(1 for s in sig if s < 0)
        if neg_cnt % 2 == 0:
            r = tuple(Fraction(s, 2) for s in sig)
            require(r in root_set, "DUAL_EVEN_ROOT", "even spinor root missing")
            even_verified += 1
        else:
            acc = [Fraction(0)] * 8
            for k in range(8):
                rk = tuple(Fraction(-sig[j] if j == k else sig[j], 2) for j in range(8))
                require(rk in root_set, "DUAL_ODD_ROOT", "single-flip spinor root missing")
                for j in range(8):
                    acc[j] += Fraction(1, 3) * rk[j]
            require(tuple(acc) == tuple(Fraction(s) for s in sig), "DUAL_ODD_SUM", "dual sum != sigma")
            odd_verified += 1
    require(even_verified == 128 and odd_verified == 128, "DUAL_COUNTS", "expected 128+128")

    # 2. Exact primal attainment witness x* = (1/3^7, -1/3) in V(E8_std)
    x_star = tuple([Fraction(1, 3)] * 7 + [Fraction(-1, 3)])
    for r in roots:
        dot = sum(a * b for a, b in zip(x_star, r))
        require(dot <= Fraction(1), "XSTAR_OUTSIDE", f"x* violates root facet: dot={dot}")
    l1_star = sum(abs(a) for a in x_star)
    norm2_star = sum(a * a for a in x_star)
    require(l1_star == Fraction(8, 3) and norm2_star == Fraction(8, 9), "XSTAR_VAL", "unexpected x* norms")

    # 3. Exact strict inequality 8/3 < sqrt(8) in pure integer arithmetic: (8/3)^2 = 64/9 < 72/9 = 8!
    require(Fraction(8, 3) ** 2 < Fraction(8), "STRICT_INEQ", "(8/3)^2 < 8 failed")

    # 4. Also verify Hadamard E8_Had deep hole y* = U_H8 (-e_0) = (-1/sqrt(8))^8 hits exact sqrt(8)
    #    In scaled coordinates sqrt(8) * E8_Had = H8 @ E8 (exact integers/halves!):
    #    Deep hole of E8_std is h = (-1, 0^7); under H8, H8 @ h = (-1, -1, ..., -1), L1 = 8 -> /sqrt(8) = sqrt(8)!
    H8_int = sylvester_H8_int()
    h_std = [Fraction(-1)] + [Fraction(0)] * 7
    for r in roots:
        require(sum(a * b for a, b in zip(h_std, r)) <= Fraction(1), "HSTD_OUTSIDE", "h_std not in V(E8)")
    h_had_scaled = [sum(Fraction(H8_int[i][j]) * h_std[j] for j in range(8)) for i in range(8)]
    l1_had_sq = sum(abs(v) for v in h_had_scaled) ** 2 / Fraction(8)
    require(l1_had_sq == Fraction(8), "HAD_L1_TIGHT", "expected E8_Had deep-hole L1^2 == 8")

    # 5. Root sparsity profiles of E8_std vs E8_Had (exact integer arithmetic on scaled sqrt(2)*E8_Had = (1/2)*H8*E8!)
    nz_std: dict[int, int] = {}
    nz_had: dict[int, int] = {}
    for r in roots:
        k_std = sum(1 for v in r if v != 0)
        nz_std[k_std] = nz_std.get(k_std, 0) + 1
        rh = [sum(Fraction(H8_int[i][j]) * r[j] for j in range(8)) for i in range(8)]
        k_had = sum(1 for v in rh if v != 0)
        nz_had[k_had] = nz_had.get(k_had, 0) + 1
    require(nz_std == {2: 112, 8: 128}, "NZ_STD", f"unexpected E8_std nnz: {nz_std}")
    require(nz_had == {1: 16, 4: 224}, "NZ_HAD", f"unexpected E8_Had nnz: {nz_had}")

    return {
        "dual_certificates_verified": {"even_signs_le_2": even_verified, "odd_signs_le_8_over_3": odd_verified},
        "exact_V_E8_std_L1_radius": "8/3",
        "exact_V_E8_std_L1_radius_sq": "64/9",
        "siliquant_cauchy_schwarz_margin_sq": "8",
        "margin_reduction_dB": r6(20.0 * math.log10(math.sqrt(8.0) / (8.0 / 3.0))),
        "exact_V_E8_Had_L1_radius_sq": "8",
        "root_nonzero_counts_E8_std": nz_std,
        "root_nonzero_counts_E8_Had": nz_had,
    }


def run_part3_tiny_weight_audit() -> dict[str, Any]:
    """Compare tiny-weight phantom-jump error (PR #17 SmolLM2 mechanism) across Z, D8, E8_std, E8_Had, BW16, Leech24."""
    exact_voronoi = exact_prove_e8_l1_voronoi_radius()
    rng = np.random.default_rng(2026)
    d8_s = 2.0 ** (-1.0 / 8.0)

    # 1,200 blocks of 24 = 28,800 weights (divisible by 8, 16, 24)
    W_gauss = rng.standard_normal((1200, 24)) * 1.5
    W_sparse8 = np.hstack([
        rng.standard_normal((12000, 1)) * 4.0,
        rng.standard_normal((12000, 7)) * 0.22,
    ])

    arms_24 = [
        ("Z", lambda M: np.rint(M)),
        ("D8_vol1", lambda M: closest_D_batch(M.reshape(-1, 8) / d8_s).reshape(M.shape) * d8_s),
        ("E8_std", lambda M: closest_E8_std(M.reshape(-1, 8)).reshape(M.shape)),
        ("E8_Had", lambda M: closest_E8_had(M.reshape(-1, 8)).reshape(M.shape)),
        ("BW16_vol1", lambda M: closest_bw16_unit(M.reshape(-1, 16)).reshape(M.shape)),
        ("Leech24", lambda M: closest_leech24_unit(M)),
    ]
    gauss_results = {}
    for name, fn in arms_24:
        Y = fn(W_gauss)
        E = Y - W_gauss
        entry: dict[str, Any] = {"tot_mse": r6(float(np.mean(E ** 2)))}
        for d in (0.05, 0.10, 0.20, 0.50):
            mask = np.abs(W_gauss) < d
            entry[f"d_lt_{d:.2f}"] = {
                "mse_over_w2": r6(float(np.mean(E[mask] ** 2) / np.mean(W_gauss[mask] ** 2))),
                "exact_zero_pct": r6(float(np.mean(np.abs(Y[mask]) < 1e-12)) * 100.0),
            }
        gauss_results[name] = entry

    # Confirm key improvements quantitatively
    e8s_05 = gauss_results["E8_std"]["d_lt_0.05"]["mse_over_w2"]
    e8h_05 = gauss_results["E8_Had"]["d_lt_0.05"]["mse_over_w2"]
    require(e8h_05 < e8s_05 / 2.5, "E8_HAD_TINY_FAIL", f"expected E8_Had << E8_std at d<0.05: {e8h_05} vs {e8s_05}")
    require(gauss_results["D8_vol1"]["d_lt_0.10"]["exact_zero_pct"] > 99.5, "D8_ZERO_FAIL", "expected D8 > 99.5% exact zero")

    sparse_results = {}
    small_sp = np.abs(W_sparse8) < 0.5
    for name, fn in [
        ("Z8", lambda M: np.rint(M)),
        ("D8_vol1", lambda M: closest_D_batch(M / d8_s) * d8_s),
        ("E8_std", closest_E8_std),
        ("E8_Had", closest_E8_had),
    ]:
        Y = fn(W_sparse8)
        E = Y - W_sparse8
        sparse_results[name] = {
            "tot_mse": r6(float(np.mean(E ** 2))),
            "small_mse_over_w2": r6(float(np.mean(E[small_sp] ** 2) / np.mean(W_sparse8[small_sp] ** 2))),
            "amp_frac_pct": r6(float(np.mean(np.abs(E[small_sp]) > np.abs(W_sparse8[small_sp]) + 1e-12)) * 100.0),
            "exact_zero_pct": r6(float(np.mean(np.abs(Y[small_sp]) < 1e-12)) * 100.0),
        }
    sparse_gain_dB = db(sparse_results["E8_std"]["tot_mse"] / sparse_results["E8_Had"]["tot_mse"])
    require(sparse_gain_dB > 0.25, "E8_HAD_SPARSE_FAIL", f"expected E8_Had > +0.25 dB on Sparse+Small, got {sparse_gain_dB}")

    return {
        "exact_voronoi_L1_theorem": exact_voronoi,
        "gaussian_tiny_weight_audit": gauss_results,
        "e8_had_vs_e8_std_d005_mse_reduction_x": r6(e8s_05 / e8h_05),
        "e8_had_vs_e8_std_d005_reduction_dB": r6(db(e8s_05 / e8h_05)),
        "sparse_plus_small_audit": sparse_results,
        "e8_had_vs_e8_std_sparse_tot_gain_dB": r6(sparse_gain_dB),
    }


# ==============================================================================
# PART 4: ZERO-FLOAT TO_INTEGERS(W) + SPECTRAL-INERTIA KEPT-COLUMN CERTIFIER
# ==============================================================================
def to_integers_dyadic_exact(W: np.ndarray) -> tuple[list[list[int]], int]:
    """Lossless IEEE-754 float -> Z * 2^e (siliquant shapelib.py receipt primitive)."""
    W = np.asarray(W, dtype=np.float64)
    require(W.ndim == 2 and W.shape[0] > 0 and W.shape[1] > 0, "INVALID_W_SHAPE", "W must be 2D non-empty")
    require(bool(np.all(np.isfinite(W))), "NONFINITE_W", "W contains non-finite entries")
    m, n = W.shape
    ratios = [[float(W[i, j]).as_integer_ratio() for j in range(n)] for i in range(m)]
    max_exp = max(d.bit_length() - 1 for row in ratios for _, d in row)
    Z = [[num << (max_exp - (den.bit_length() - 1)) for num, den in row] for row in ratios]
    return Z, -max_exp


def gram_ZZt_exact(Z: list[list[int]]) -> list[list[int]]:
    m = len(Z)
    G = [[0] * m for _ in range(m)]
    for i in range(m):
        zi = Z[i]
        for j in range(i, m):
            s = sum(a * b for a, b in zip(zi, Z[j]))
            G[i][j] = s
            G[j][i] = s
    return G


def exact_moments_shifted(G_int: list[list[int]], tau_rat: Fraction) -> tuple[dict[int, Fraction], list[list[Fraction]]]:
    tau_f = parse_exact_fraction(tau_rat)
    require(tau_f >= 0, "NEGATIVE_TAU", "bulk floor tau must be >= 0")
    n = len(G_int)
    H = [[Fraction(parse_exact_int(G_int[i][j])) - (tau_f if i == j else Fraction(0)) for j in range(n)] for i in range(n)]
    S1 = sum((H[i][i] for i in range(n)), Fraction(0))
    H2 = [[sum((H[i][t] * H[t][j] for t in range(n)), Fraction(0)) for j in range(n)] for i in range(n)]
    S2 = sum((H2[i][i] for i in range(n)), Fraction(0))
    S3 = sum((H2[i][j] * H[j][i] for i in range(n) for j in range(n)), Fraction(0))
    S4 = sum((H2[i][j] * H2[i][j] for i in range(n) for j in range(n)), Fraction(0))
    return {1: S1, 2: S2, 3: S3, 4: S4}, H


def certify_phi_noise_annihilating(
    moments: dict[int, Fraction], x0: Fraction, tau_shift: Fraction, b: int = 0
) -> dict[str, Any]:
    """AUDIT-014 Theorem A3 exact rational quartic SOS bound (zero floats!)."""
    x0_f = parse_exact_fraction(x0)
    tau_f = parse_exact_fraction(tau_shift)
    b_i = parse_exact_int(b, min_value=0)
    require(x0_f > 0, "INVALID_X0", "x0 must be > 0")
    require(tau_f >= 0, "NEGATIVE_TAU", "tau_shift must be >= 0")
    for k in (1, 2, 3, 4):
        require(k in moments, "MISSING_MOMENT", f"missing S_{k}")
        parse_exact_fraction(moments[k])
    d = 3 * x0_f + tau_f
    L = 2 * x0_f * (2 * x0_f + tau_f) / d
    M = (x0_f * x0_f * (x0_f + tau_f) ** 3) / d
    tr_phi = (
        L * tau_f * tau_f * moments[1]
        + (2 * L * tau_f - tau_f * tau_f) * moments[2]
        + (L - 2 * tau_f) * moments[3]
        - moments[4]
    ) / M
    raw = tr_phi - Fraction(b_i)
    bound = raw if raw > 0 else Fraction(0)
    return {"bound": bound, "min_rank_cert": ceil_fraction(bound), "x0": frac_str(x0_f)}


def independent_sym_inertia_descartes(H_raw: list[list[Fraction]]) -> tuple[int, int, int]:
    """Zero-shared-code Faddeev-LeVerrier + Cayley-Hamilton + Descartes exact inertia oracle."""
    n = len(H_raw)
    for i in range(n):
        for j in range(i + 1, n):
            require(H_raw[i][j] == H_raw[j][i], "ORACLE_NOT_SYM", "matrix not symmetric")

    def _mm(A: list[list[Fraction]], B: list[list[Fraction]]) -> list[list[Fraction]]:
        return [[sum((A[r][t] * B[t][c] for t in range(n)), Fraction(0)) for c in range(n)] for r in range(n)]

    coeffs: list[Fraction] = [Fraction(1)]
    B = [row[:] for row in H_raw]
    coeffs.append(-sum((B[i][i] for i in range(n)), Fraction(0)))
    for k in range(2, n + 1):
        c_prev = coeffs[-1]
        shifted = [[B[r][c] + (c_prev if r == c else Fraction(0)) for c in range(n)] for r in range(n)]
        B = _mm(H_raw, shifted)
        coeffs.append(-sum((B[i][i] for i in range(n)), Fraction(0)) / Fraction(k))

    cn = coeffs[-1]
    for r in range(n):
        for c in range(n):
            require(B[r][c] + (cn if r == c else Fraction(0)) == 0, "ORACLE_CH_FAIL", "Cayley-Hamilton nonzero")

    poly = coeffs[:]
    n_zero = 0
    while len(poly) > 1 and poly[-1] == 0:
        poly.pop()
        n_zero += 1

    def _sc(seq: list[Fraction]) -> int:
        nz = [v for v in seq if v != 0]
        return sum(1 for a, b in zip(nz, nz[1:]) if (a > 0) != (b > 0))

    n_pos = _sc(poly)
    n_neg = _sc([c if (idx % 2 == 0) else -c for idx, c in enumerate(poly)])
    require(n_pos + n_neg + n_zero == n, "ORACLE_DESCARTES", "root sum mismatch")
    return n_pos, n_neg, n_zero


def run_part4_zero_float_spectral_outliers() -> dict[str, Any]:
    """Zero-float to_integers(W) + AUDIT-014 Phi_{x0,tau} kept-column spike certification + SP-01 gauge."""
    rng = np.random.default_rng(77)
    m, n = 12, 48
    W_bulk = (rng.standard_normal((m, n)) * 0.25).astype(np.float16).astype(np.float64)
    W_spike = W_bulk.copy()
    for col_idx, amp in [(2, 4.5), (17, 4.0), (35, 3.8)]:
        u = rng.standard_normal(m)
        u /= np.linalg.norm(u)
        W_spike[:, col_idx] += amp * u
    W_spike = W_spike.astype(np.float16).astype(np.float64)
    W_perm = W_spike[:, rng.permutation(n)]

    # SP-01 readout-flip test: weight by activation Hessian H_act = diag(100 on col 2, 1 elsewhere)
    # Comparing unweighted W_bulk vs H_act-weighted W_bulk (a single active unit flip!)
    h_act_sqrt = np.ones(n, dtype=np.float64)
    h_act_sqrt[2] = 16.0  # active hidden unit (like clip-by-layer-qwen25 12-97x activity!)
    W_weighted = (W_bulk * h_act_sqrt[None, :]).astype(np.float16).astype(np.float64)

    def _certify_matrix(W: np.ndarray, b_list: Sequence[int]) -> dict[str, Any]:
        Z, exp2 = to_integers_dyadic_exact(W)
        G_int = gram_ZZt_exact(Z)
        col_sq = sorted(sum(Z[i][j] * Z[i][j] for i in range(m)) for j in range(n))
        med_col_sq = Fraction(col_sq[n // 2])
        # Exact rational Marchenko-Pastur bulk ceiling (m=12, n=48 => (1+sqrt(12/48))^2 = 9/4, x 6/5 margin)
        tau_mp = Fraction(n, m) * med_col_sq * Fraction(9, 4) * Fraction(6, 5)
        moms, H_mat = exact_moments_shifted(G_int, tau_mp)
        n_pos_true, n_neg_true, n_zero_true = independent_sym_inertia_descartes(H_mat)

        by_b: dict[str, Any] = {}
        for b in b_list:
            best_b = {"bound": Fraction(0), "min_rank_cert": 0, "x0": "0"}
            if moms[3] > 0:
                x0_base = moms[4] / moms[3]
                for mult in (Fraction(1, 2), Fraction(3, 4), Fraction(1), Fraction(5, 4)):
                    cand = certify_phi_noise_annihilating(moms, x0_base * mult, tau_mp, b=b)
                    if cand["bound"] > best_b["bound"]:
                        best_b = cand
            require(best_b["bound"] <= Fraction(max(0, n_pos_true - b)), "SOUNDNESS_FAIL", "bound exceeded true n_+(H)-b")
            by_b[f"b={b}"] = {
                "exact_bound_approx": r6(float(best_b["bound"])),
                "certified_min_residual_spikes": best_b["min_rank_cert"],
                "true_remaining_spikes": max(0, n_pos_true - b),
            }
        return {
            "dyadic_exponent": exp2,
            "descartes_inertia_H": [n_pos_true, n_neg_true, n_zero_true],
            "by_kept_columns_b": by_b,
            "exact_bound_b0_str": frac_str(
                max(
                    (
                        certify_phi_noise_annihilating(moms, (moms[4] / moms[3]) * mult, tau_mp, b=0)["bound"]
                        for mult in (Fraction(1, 2), Fraction(3, 4), Fraction(1), Fraction(5, 4))
                    ),
                    default=Fraction(0),
                )
                if moms[3] > 0
                else Fraction(0)
            ),
        }

    res_bulk = _certify_matrix(W_bulk, [0])
    res_spike = _certify_matrix(W_spike, [0, 1, 2, 3])
    res_perm = _certify_matrix(W_perm, [0])
    res_hess = _certify_matrix(W_weighted, [0])

    require(res_bulk["descartes_inertia_H"][0] == 0 and res_bulk["by_kept_columns_b"]["b=0"]["certified_min_residual_spikes"] == 0, "BULK_NULL_FAIL", "bulk null false positive")
    require(res_spike["descartes_inertia_H"][0] == 3 and res_spike["by_kept_columns_b"]["b=0"]["certified_min_residual_spikes"] == 3, "SPIKE3_FAIL", "failed to certify 3 spikes")
    require(res_spike["by_kept_columns_b"]["b=1"]["certified_min_residual_spikes"] == 2, "RESID_B1_FAIL", "expected >=2 residual after b=1")
    require(res_spike["by_kept_columns_b"]["b=2"]["certified_min_residual_spikes"] == 1, "RESID_B2_FAIL", "expected >=1 residual after b=2")
    require(res_spike["by_kept_columns_b"]["b=3"]["certified_min_residual_spikes"] == 0, "RESID_B3_FAIL", "expected 0 after b=3")
    require(res_spike["exact_bound_b0_str"] == res_perm["exact_bound_b0_str"], "SP01_PERM_FAIL", "column permutation broke exact rational bound!")
    require(res_hess["by_kept_columns_b"]["b=0"]["certified_min_residual_spikes"] >= 1, "SP01_HESS_FAIL", "activation-Hessian weighted spike not detected!")

    return {
        "pure_bulk_null": res_bulk,
        "ffn_down_3_spikes": res_spike,
        "ffn_down_3_spikes_col_permuted_SP01": {
            "descartes_inertia_H": res_perm["descartes_inertia_H"],
            "bit_identical_exact_fraction_bound": res_spike["exact_bound_b0_str"] == res_perm["exact_bound_b0_str"],
        },
        "activation_hessian_weighted_bulk_SP01": {
            "descartes_inertia_H": res_hess["descartes_inertia_H"],
            "certified_min_spikes": res_hess["by_kept_columns_b"]["b=0"]["certified_min_residual_spikes"],
        },
    }


# ==============================================================================
# PART 5: R5 MUST-FAIL NEGATIVE CONTROLS (6/6 REJECTED)
# ==============================================================================
def run_part5_r5_controls() -> dict[str, Any]:
    # C1: Bent function is NOT in BW16
    bent_int = [((j & 1) & ((j >> 1) & 1)) ^ (((j >> 2) & 1) & ((j >> 3) & 1)) for j in range(16)]
    require(not verify_bw16_membership_int(bent_int), "C1_FAIL", "bent function falsely accepted in BW16")

    # C2: Deep hole (4, 0^23) is NOT in Leech24
    hole24_int = [4] + [0] * 23
    require(not verify_leech24_membership_int(hole24_int), "C2_FAIL", "deep hole falsely accepted in Leech24")

    # C3-C6: Structured CertError rejections
    checks = [
        ("C3_bool_rejected_as_int", "INVALID_TYPE", lambda: parse_exact_int(True)),
        ("C4_float_rejected_in_spectral_TCB", "INVALID_TYPE", lambda: parse_exact_fraction(1.5)),
        ("C5_negative_tau_rejected", "NEGATIVE_TAU", lambda: certify_phi_noise_annihilating({1: Fraction(1), 2: Fraction(1), 3: Fraction(1), 4: Fraction(1)}, Fraction(1), Fraction(-1), 0)),
        ("C6_nonpositive_x0_rejected", "INVALID_X0", lambda: certify_phi_noise_annihilating({1: Fraction(1), 2: Fraction(1), 3: Fraction(1), 4: Fraction(1)}, Fraction(0), Fraction(1), 0)),
    ]
    out = [
        {"control": "C1_bent_not_in_BW16", "rejected": True},
        {"control": "C2_deep_hole_not_in_Leech24", "rejected": True},
    ]
    for name, exp_code, fn in checks:
        got = None
        try:
            fn()
        except CertError as exc:
            got = exc.code
        require(got == exp_code, "R5_CONTROL_FAIL", f"{name}: expected {exp_code}, got {got}")
        out.append({"control": name, "expected": exp_code, "caught": got, "rejected": True})
    return {"controls_passed": len(out), "controls_total": len(out), "controls": out}


def run_all() -> dict[str, Any]:
    payload = {
        "schema": "aletheia.wave2.siliquant_symbiosis.v1",
        "part1_higher_dim_decoders": run_part1_higher_dim_decoders(),
        "part2_safe_scale_theorem": run_part2_safe_scale_theorem(),
        "part3_voronoi_L1_and_E8_Had": run_part3_tiny_weight_audit(),
        "part4_zero_float_spectral_outliers": run_part4_zero_float_spectral_outliers(),
        "part5_r5_controls": run_part5_r5_controls(),
        "verdict": "ALL_5_PARTS_VERIFIED",
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["canonical_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Wave 2 Arm 1: siliquant <-> Cave Expedition Symbiosis Verifier.")
    parser.add_argument("--json", action="store_true", help="Emit deterministic canonical JSON.")
    parser.add_argument(
        "--check-cert",
        type=str,
        default="",
        help="Verify bit-for-bit parity against an existing committed JSON certificate without modifying it.",
    )
    parser.add_argument(
        "--write-cert",
        type=str,
        default="",
        help="Write canonical JSON certificate to path (if target exists, fails closed on mismatch unless --force).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow --write-cert to overwrite an existing certificate whose contents differ.",
    )
    args = parser.parse_args(argv)
    res = run_all()
    txt = json.dumps(res, indent=2, sort_keys=True)
    expected_bytes = (txt + "\n").encode("utf-8")
    if args.check_cert:
        cert_path = Path(args.check_cert)
        if cert_path.is_symlink() or not cert_path.is_file():
            print(f"CERT_PARITY_ERROR: missing or symlink certificate at {cert_path}", file=sys.stderr)
            return 1
        actual_bytes = cert_path.read_bytes()
        if actual_bytes != expected_bytes:
            print(f"CERT_PARITY_MISMATCH: {cert_path} differs from recomputed certificate", file=sys.stderr)
            return 1
    if args.write_cert:
        cert_path = Path(args.write_cert)
        if cert_path.is_symlink():
            print(f"CERT_WRITE_ERROR: refusing symlink target {cert_path}", file=sys.stderr)
            return 1
        if cert_path.exists():
            if not cert_path.is_file():
                print(f"CERT_WRITE_ERROR: target is not a regular file: {cert_path}", file=sys.stderr)
                return 1
            actual_bytes = cert_path.read_bytes()
            if actual_bytes != expected_bytes and not args.force:
                print(
                    f"CERT_PARITY_MISMATCH: existing {cert_path} differs from recomputed certificate "
                    "(pass --force to overwrite)",
                    file=sys.stderr,
                )
                return 1
        cert_path.write_bytes(expected_bytes)
    if args.json:
        print(txt)
    else:
        p1, p2, p3, p4, p5 = (
            res["part1_higher_dim_decoders"],
            res["part2_safe_scale_theorem"],
            res["part3_voronoi_L1_and_E8_Had"],
            res["part4_zero_float_spectral_outliers"],
            res["part5_r5_controls"],
        )
        print("WAVE 2 SYMBIOSIS (ARM 1): Moissanite-Labs/siliquant <-> Cave Expedition")
        print(f"  [Part 1] BW16 G={p1['bw16']['G_recomputed']} +- {p1['bw16']['G_se']} (4320/4320 Voronoi OK, bent hole rho^2={p1['bw16']['vol1_covering_radius_sq']})")
        print(f"           Leech24 G={p1['leech24']['G_recomputed']} +- {p1['leech24']['G_se']} (196560/196560 Voronoi OK, deep hole rho^2=2.0)")
        t20 = p2["table_24d_safe_scale"]["2.00_bpw"]
        t40 = p2["table_24d_safe_scale"]["4.00_bpw"]
        print(f"  [Part 2] Leech24 vs E8^3 (2.0 bpw): safe_MSE +{t20['leech_vs_e8x3_safe_mse_dB']} dB, PROVED_BOUND +{t20['leech_vs_e8x3_proved_bound_dB']} dB (vs Z24: +{t20['leech_vs_z24_proved_bound_dB']} dB)")
        print(f"           Leech24 vs E8^3 (4.0 bpw): safe_MSE +{t40['leech_vs_e8x3_safe_mse_dB']} dB, PROVED_BOUND +{t40['leech_vs_e8x3_proved_bound_dB']} dB")
        print(f"           BW16 bent-hole contrast (2.5 bpw): free +{p2['table_16d_safe_scale']['2.50_bpw']['bw16_vs_e8x2_free_dB']} dB vs PROVED_BOUND {p2['table_16d_safe_scale']['2.50_bpw']['bw16_vs_e8x2_proved_bound_dB']} dB")
        print(f"  [Part 3] Exact V(E8_std) L1 radius = 8/3 < sqrt(8) (margin -{p3['exact_voronoi_L1_theorem']['margin_reduction_dB']} dB); E8_Had roots = {p3['exact_voronoi_L1_theorem']['root_nonzero_counts_E8_Had']}")
        print(f"           Tiny-weight d<0.05 MSE drop E8_Had vs E8_std: {p3['e8_had_vs_e8_std_d005_mse_reduction_x']}x (-{p3['e8_had_vs_e8_std_d005_reduction_dB']} dB); Sparse+Small gain: +{p3['e8_had_vs_e8_std_sparse_tot_gain_dB']} dB")
        print(f"  [Part 4] Zero-float to_integers+Phi_SOS: bulk_null={p4['pure_bulk_null']['by_kept_columns_b']['b=0']['certified_min_residual_spikes']}, 3_spikes={p4['ffn_down_3_spikes']['by_kept_columns_b']['b=0']['certified_min_residual_spikes']}, SP01_perm_exact={p4['ffn_down_3_spikes_col_permuted_SP01']['bit_identical_exact_fraction_bound']}")
        print(f"  [Part 5] R5 Controls: {p5['controls_passed']}/{p5['controls_total']} rejected")
        print(f"  VERDICT: {res['verdict']} | SHA-256: {res['canonical_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
