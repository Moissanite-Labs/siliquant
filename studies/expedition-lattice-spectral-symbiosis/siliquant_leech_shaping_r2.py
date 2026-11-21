#!/usr/bin/env python3
r"""
Wave 2 Symbiosis — Round 2 (Arm 1):
  (A) Exact Bit-Bijective 24D Leech Lattice (Lambda_24) Cover (1973) DP Sphere-Shaping Codec
      (Fixed-Ball 2.0/3.0/4.0 bpw + Exact Variable-Shell Ranking/Unranking)
  (B) Head-to-Head Rate-Distortion & Safe-Scale Benchmark vs. siliquant E_8^3 & Z^24
  (C) SP-01 Hessian / Activation-Weighted Quantization: E8_std vs. E8_Had + Zero-Float
      to_integers(W) & AUDIT-014 Phi_{x0,tau} Certified Outlier Column Splitting

100% deterministic, zero bare `assert`s, strict R1/R3/R4/R5 gates, bit-identical canonical JSON
across all 4 isolated Python modes (-I -B, -I -O -B, -I -B -W error, -I -O -B -W error).

Key Mathematical Contributions:
  1. Symmetry-Collapsed MOG/Hexacode Cover (1973) DP for 24D Leech Lambda_24:
     Because all three nonzero GF(4) symbols {1, 2, 3} have identical MOG column weight
     distributions, the 6-column DP depends ONLY on suffix zero/nonzero counts (k0, k1)
     rather than all 64 hexacodewords — collapsing 256 tables to 20 sparse integer tables
     and building an EXACT bit-bijective rank/unrank codec up to 4.0 bpw (m <= 214,
     74,972,166,705,387,133,681,638,419,281 = 7.497e28 codewords, 95.92 bits!) in < 1s!
     Every single one of all 215 shell counts m = 0..214 is cross-checked digit-for-digit
     against Ramanujan's modular form E_4(q)^3 - 720 * Delta(q) (to 29 decimal digits!).
  2. Exact Fixed-Ball (48/72/96 bits = 6/9/12 bytes) & Variable-Shell Sphere Shaping:
     Directly compares bit-exact encoded/decoded Lambda_24 vs. siliquant's E_8^3 and Z^24
     at 2.0, 3.0, and 4.0 bpw across out-of-sample Gaussian and heavy-tailed Student-t3
     blocks under free-scale ball shaping, proved safe-scale shaping (certified.py), and
     exact variable-shell shaping.
  3. SP-01 Hessian-Weighted Quantization & Zero-Float Certified Outlier Splitting:
     Evaluates unweighted vs. H_act^{1/2}-preconditioned E8_std vs. E8_Had (AUDIT-017/028)
     + zero-float to_integers(W) & Phi_{x0,tau} certified outlier-column splitting on
     ill-conditioned heavy-tailed Transformer weight matrices, plus an isotropic Gaussian
     null control and exact column-permutation SP-01 gauge-invariance check.

Run:
  python3 -I -O -B -W error siliquant_leech_shaping_r2.py
  python3 -I -O -B -W error siliquant_leech_shaping_r2.py --json
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


def parse_exact_int(val: Any, min_value: int | None = None, max_value: int | None = None) -> int:
    if isinstance(val, bool) or not isinstance(val, int):
        raise CertError("INVALID_TYPE", f"expected exact int, got {type(val).__name__}")
    if min_value is not None and val < min_value:
        raise CertError("INT_OUT_OF_RANGE", f"expected int >= {min_value}, got {val}")
    if max_value is not None and val > max_value:
        raise CertError("INT_OUT_OF_RANGE", f"expected int <= {max_value}, got {val}")
    return val


def parse_exact_fraction(val: Any) -> Fraction:
    if isinstance(val, bool) or not isinstance(val, (int, Fraction)):
        raise CertError("INVALID_TYPE", f"expected exact int/Fraction, got {type(val).__name__}")
    return val if isinstance(val, Fraction) else Fraction(val)


def ceil_fraction(q: Fraction) -> int:
    q = parse_exact_fraction(q)
    return -((-q.numerator) // q.denominator)


def frac_str(q: Fraction) -> str:
    q = parse_exact_fraction(q)
    return str(q.numerator) if q.denominator == 1 else f"{q.numerator}/{q.denominator}"


def r6(x: float) -> float:
    return round(float(x), 6)


def db(x: float) -> float:
    return 10.0 * math.log10(x)


# ==============================================================================
# EXACT INTEGER POLYNOMIAL MULTIPLICATION (CARRY-FREE KRONECKER PACKING)
# ==============================================================================
def poly_mul_exact_nonneg(a: list[int], b: list[int], deg: int) -> list[int]:
    """Exact nonnegative integer polynomial convolution truncated to degree `deg` via carry-free Kronecker packing."""
    la = min(len(a), deg + 1)
    lb = min(len(b), deg + 1)
    max_a = max(a[:la], default=0)
    max_b = max(b[:lb], default=0)
    if max_a == 0 or max_b == 0:
        return [0] * (deg + 1)
    # Upper bound on any output coefficient is min(la, lb) * max_a * max_b
    max_out = min(la, lb) * max_a * max_b
    bits = max_out.bit_length() + 2
    mask = (1 << bits) - 1

    # Pack polynomials into single Python ints (C-speed Karatsuba/Toom-Cook!)
    pa = 0
    for i in range(la - 1, -1, -1):
        pa = (pa << bits) | a[i]
    pb = 0
    for j in range(lb - 1, -1, -1):
        pb = (pb << bits) | b[j]
    prod = pa * pb
    out = [0] * (deg + 1)
    for k in range(deg + 1):
        out[k] = prod & mask
        prod >>= bits
        if prod == 0:
            break
    return out


def poly_mul_signed_trunc(a: list[int], b: list[int], deg: int) -> list[int]:
    """Exact signed integer polynomial convolution truncated to degree `deg`."""
    out = [0] * (deg + 1)
    for i, ai in enumerate(a):
        if ai == 0:
            continue
        lim = min(len(b), deg + 1 - i)
        for j in range(lim):
            out[i + j] += ai * b[j]
    return out


def poly_pow_trunc(a: list[int], p: int, deg: int, nonneg: bool = False) -> list[int]:
    res = [0] * (deg + 1)
    res[0] = 1
    base = a[: deg + 1]
    mul = poly_mul_exact_nonneg if nonneg else poly_mul_signed_trunc
    while p > 0:
        if p & 1:
            res = mul(res, base, deg)
        p >>= 1
        if p:
            base = mul(base, base, deg)
    return res


def theta_E8_exact(max_m: int) -> list[int]:
    c = [0] * (max_m + 1)
    c[0] = 1
    for m in range(1, max_m + 1):
        c[m] = 240 * sum(d ** 3 for d in range(1, m + 1) if m % d == 0)
    return c


def theta_Leech24_ramanujan(max_m: int) -> list[int]:
    """Independent modular-form shell counts N_m = [q^{2m}](E_4(q)^3 - 720 * Delta(q))."""
    e4 = theta_E8_exact(max_m)
    e4_3 = poly_pow_trunc(e4, 3, max_m, nonneg=True)
    prod = [0] * (max_m + 1)
    prod[0] = 1
    for n in range(1, max_m + 1):
        for k in range(max_m, n - 1, -1):
            prod[k] -= prod[k - n]
    p24 = poly_pow_trunc(prod, 24, max_m, nonneg=False)
    delta = [0] + p24[:max_m]
    return [e4_3[m] - 720 * delta[m] for m in range(max_m + 1)]


def theta_Zn_exact(n: int, max_r2: int) -> list[int]:
    t1 = [0] * (max_r2 + 1)
    k = 0
    while k * k <= max_r2:
        t1[k * k] = 1 if k == 0 else 2
        k += 1
    return poly_pow_trunc(t1, n, max_r2, nonneg=True)


# ==============================================================================
# PART A: EXACT MOG/HEXACODE COVER (1973) DP CODEC FOR 24D LEECH LAMBDA_24
# ==============================================================================
_GF4_MUL = ((0, 0, 0, 0), (0, 1, 2, 3), (0, 2, 3, 1), (0, 3, 1, 2))
_G_H6 = ((1, 0, 0, 1, 2, 3), (0, 1, 0, 1, 3, 2), (0, 0, 1, 1, 1, 1))
HEXACODE: tuple[tuple[int, ...], ...] = tuple(
    tuple(
        _GF4_MUL[a][_G_H6[0][j]] ^ _GF4_MUL[b][_G_H6[1][j]] ^ _GF4_MUL[c][_G_H6[2][j]]
        for j in range(6)
    )
    for a in range(4)
    for b in range(4)
    for c in range(4)
)
HEXACODE_INDEX: dict[tuple[int, ...], int] = {cw: idx for idx, cw in enumerate(HEXACODE)}
HEXACODE_NP = np.array(HEXACODE, dtype=np.int64)

MOG_PATTERNS_DICT: dict[tuple[int, int, int], tuple[int, int, int, int]] = {}
MOG_PATTERNS_NP = np.zeros((2, 4, 2, 4), dtype=np.int64)
for _bits in itertools.product((0, 1), repeat=4):
    _cp = _bits[0] ^ _bits[1] ^ _bits[2] ^ _bits[3]
    _sym = (1 if _bits[1] else 0) ^ (2 if _bits[2] else 0) ^ (3 if _bits[3] else 0)
    MOG_PATTERNS_DICT[(_cp, _sym, _bits[0])] = _bits
    MOG_PATTERNS_NP[_cp, _sym, _bits[0]] = _bits


def _symbols_mod4(r: int, max_e: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    k = 0
    while True:
        found = False
        cands = [r] if k == 0 else [4 * k + r, -4 * k + r]
        for v in cands:
            if v * v <= max_e:
                out.append((v, v * v))
                found = True
        if not found:
            break
        k += 1
    out.sort(key=lambda pair: (abs(pair[0]), pair[0] < 0))
    return out


class LeechCoverCodec:
    """
    Exact bit-bijective Cover (1973) enumerative sphere-shaping codec for 24D Leech Lambda_24.
    Supports BOTH:
      - Fixed-ball coding:    rank_ball(v, m_cap)  <-> unrank_ball(idx, m_cap)  (for ANY m_cap <= max_m!)
      - Variable-shell coding: rank_shell(v)       <-> unrank_shell(m, idx)     (for ANY shell m <= max_m!)
    Raw Leech integer coordinates: min ||v||^2 = 32 (m = 2), ||v||^2 = 16 * m, vol = 2^36.
    Unit-covolume scale divides by 2 * sqrt(2) (so unit ||v||^2 = 2 * m, covering radius rho = sqrt(2)).
    """

    def __init__(self, max_shell_m: int) -> None:
        self.max_m = parse_exact_int(max_shell_m, min_value=2)
        self.B_raw = 16 * self.max_m
        self.E_max = {0: 4 * self.max_m, 1: 2 * self.max_m - 3}

        # 1D residue symbols (v, scaled_cost, k_parity)
        self.syms: dict[int, list[tuple[int, int, int]]] = {}
        for r in range(4):
            m = r & 1
            raw_s = _symbols_mod4(r, self.B_raw)
            if m == 0:
                self.syms[r] = [(v, (v * v) >> 2, ((v - r) >> 2) & 1) for v, _ in raw_s]
            else:
                self.syms[r] = [(v, (v * v - 1) >> 3, ((v - r) >> 2) & 1) for v, _ in raw_s]

        # Step 1: 4-coordinate column DP for each (m, cp, sym, t)
        self.col_dp: dict[tuple[int, int, int, int], tuple[list[list[list[int]]], int, list[int]]] = {}
        self.col_exact: dict[tuple[int, int, int, int], list[list[int]]] = {}
        for m in (0, 1):
            Em = self.E_max[m]
            for cp in (0, 1):
                for sym in range(4):
                    for t in (0, 1):
                        b = MOG_PATTERNS_DICT[(cp, sym, t)]
                        r_idx = [m + 2 * bi for bi in b]
                        base_s = (m + ((sum(b) - cp) >> 1)) & 1
                        dp: list[list[list[int]]] = [[[0] * (Em + 1), [0] * (Em + 1)] for _ in range(5)]
                        dp[4][0][0] = 1
                        for pos in range(3, -1, -1):
                            nxt = dp[pos + 1]
                            cur = dp[pos]
                            for _, cost, kp in self.syms[r_idx[pos]]:
                                if cost > Em:
                                    continue
                                for par in (0, 1):
                                    src = nxt[par ^ kp]
                                    dst = cur[par]
                                    for u in range(cost, Em + 1):
                                        val = src[u - cost]
                                        if val:
                                            dst[u] += val
                        self.col_dp[(m, cp, sym, t)] = (dp, base_s, r_idx)
                        ex = [dp[0][base_s], dp[0][base_s ^ 1]]
                        wt_type = 1 if sym else 0
                        if (m, cp, wt_type, t) in self.col_exact:
                            require(
                                self.col_exact[(m, cp, wt_type, t)] == ex,
                                "SYMMETRY_INVARIANT_FAIL",
                                "nonzero GF(4) symbols must share identical column DP counts",
                            )
                        else:
                            self.col_exact[(m, cp, wt_type, t)] = ex

        # Step 2: Symmetry-collapsed 6-column DP via carry-free Kronecker integer multiplication!
        needed_pairs = {
            (sum(1 for x in cw[j:] if x == 0), sum(1 for x in cw[j:] if x != 0))
            for cw in HEXACODE
            for j in range(7)
        }
        self.suf_exact: dict[tuple[int, int, int, int], list[list[int]]] = {}
        self.suf_cum: dict[tuple[int, int, int, int], list[list[int]]] = {}
        for m in (0, 1):
            Em = self.E_max[m]
            for cp in (0, 1):
                base = [[0] * (Em + 1) for _ in range(4)]
                base[0][0] = 1
                self.suf_exact[(m, cp, 0, 0)] = base
                for L in range(1, 7):
                    for k0 in range(L + 1):
                        k1 = L - k0
                        if (k0, k1) not in needed_pairs:
                            continue
                        if k0 > 0 and (m, cp, k0 - 1, k1) in self.suf_exact:
                            prev = self.suf_exact[(m, cp, k0 - 1, k1)]
                            wt = 0
                        else:
                            prev = self.suf_exact[(m, cp, k0, k1 - 1)]
                            wt = 1
                        cur = [[0] * (Em + 1) for _ in range(4)]
                        for t in (0, 1):
                            for s in (0, 1):
                                d_st = 2 * t + s
                                c_poly = self.col_exact[(m, cp, wt, t)][s]
                                for st in range(4):
                                    conv = poly_mul_exact_nonneg(c_poly, prev[st ^ d_st], Em)
                                    dst = cur[st]
                                    for u in range(Em + 1):
                                        if conv[u]:
                                            dst[u] += conv[u]
                        self.suf_exact[(m, cp, k0, k1)] = cur

                for k0, k1 in needed_pairs:
                    ex = self.suf_exact[(m, cp, k0, k1)]
                    cum: list[list[int]] = []
                    for st in range(4):
                        c_arr = [0] * (Em + 1)
                        acc = 0
                        for u, v in enumerate(ex[st]):
                            acc += v
                            c_arr[u] = acc
                        cum.append(c_arr)
                    self.suf_cum[(m, cp, k0, k1)] = cum

        # Step 3: Exact shell counts directly from MOG/Hexacode DP!
        self.dp_shells = [0] * (self.max_m + 1)
        for m in (0, 1):
            Em = self.E_max[m]
            for cp in (0, 1):
                tst = 2 * cp + (m ^ cp)
                for cw in HEXACODE:
                    k0 = sum(1 for x in cw if x == 0)
                    ex_arr = self.suf_exact[(m, cp, k0, 6 - k0)][tst]
                    for u, v in enumerate(ex_arr):
                        if not v:
                            continue
                        raw_e = (4 * u) if m == 0 else (24 + 8 * u)
                        require(raw_e % 16 == 0, "LEECH_NORM_MOD16", f"non-multiple-of-16 raw norm^2: {raw_e}")
                        self.dp_shells[raw_e // 16] += v

        # Cumulative ball counts for every m_cap in 0..max_m
        self.dp_ball_counts = [0] * (self.max_m + 1)
        acc = 0
        for sh_m, cnt in enumerate(self.dp_shells):
            acc += cnt
            self.dp_ball_counts[sh_m] = acc

    def _validate_and_parse_leech(self, v: Sequence[int]) -> tuple[list[int], int, int, tuple[int, ...], int, int]:
        require(isinstance(v, (list, tuple)) and len(v) == 24, "INVALID_LEECH_DIM", "Leech vector must have 24 ints")
        vals = [parse_exact_int(x) for x in v]
        m = vals[0] & 1
        require(all((x & 1) == m for x in vals), "NOT_IN_LEECH", "mixed coordinate parities")
        require((sum(vals) % 8) == (4 * m), "NOT_IN_LEECH", "coordinate sum mod 8 invalid")
        c = [((x - m) >> 1) & 1 for x in vals]
        cols_c = [c[4 * j : 4 * j + 4] for j in range(6)]
        cp = cols_c[0][0] ^ cols_c[0][1] ^ cols_c[0][2] ^ cols_c[0][3]
        for j in range(1, 6):
            cj_p = cols_c[j][0] ^ cols_c[j][1] ^ cols_c[j][2] ^ cols_c[j][3]
            require(cj_p == cp, "NOT_IN_LEECH", f"MOG column {j} parity mismatch")
        top_par = sum(cols_c[j][0] for j in range(6)) & 1
        require(top_par == cp, "NOT_IN_LEECH", "MOG top-row parity mismatch")
        cw = tuple(
            (1 if col[1] else 0) ^ (2 if col[2] else 0) ^ (3 if col[3] else 0)
            for col in cols_c
        )
        require(cw in HEXACODE_INDEX, "NOT_IN_LEECH", "GF(4) projection not in Hexacode H6")
        raw_e = sum(x * x for x in vals)
        require(raw_e % 16 == 0, "NOT_IN_LEECH", "raw norm^2 not divisible by 16")
        shell_m = raw_e // 16
        return vals, m, cp, cw, HEXACODE_INDEX[cw], shell_m

    def _branch_budget(self, m: int, m_cap: int) -> int:
        return (4 * m_cap) if m == 0 else (2 * m_cap - 3)

    def _branches_for_mode(self, m_cap: int, exact_shell: bool) -> list[tuple[int, int, int, int, int]]:
        branches: list[tuple[int, int, int, int, int]] = []
        off = 0
        for m in (0, 1):
            Em = self._branch_budget(m, m_cap)
            if Em < 0:
                continue
            table = self.suf_exact if exact_shell else self.suf_cum
            for cp in (0, 1):
                tst = 2 * cp + (m ^ cp)
                for cw_idx, cw in enumerate(HEXACODE):
                    k0 = sum(1 for x in cw if x == 0)
                    cnt = table[(m, cp, k0, 6 - k0)][tst][Em]
                    if cnt > 0:
                        branches.append((m, cp, cw_idx, off, cnt))
                        off += cnt
        return branches

    def _rank_impl(self, v: Sequence[int], m_cap: int, exact_shell: bool) -> int:
        m_cap = parse_exact_int(m_cap, min_value=0, max_value=self.max_m)
        vals, m, cp, cw, cw_idx, shell_m = self._validate_and_parse_leech(v)
        if exact_shell:
            require(shell_m == m_cap, "SHELL_MISMATCH", f"vector shell {shell_m} != requested {m_cap}")
        else:
            require(shell_m <= m_cap, "OUT_OF_BALL", f"vector shell {shell_m} exceeds cap {m_cap}")

        branches = self._branches_for_mode(m_cap, exact_shell)
        idx = -1
        for bm, bcp, bcw, boff, _ in branches:
            if bm == m and bcp == cp and bcw == cw_idx:
                idx = boff
                break
        require(idx >= 0, "INTERNAL_BRANCH_MISSING", "branch not found for valid Leech vector")

        budget = self._branch_budget(m, m_cap)
        need_st = 2 * cp + (m ^ cp)
        suf_table = self.suf_exact if exact_shell else self.suf_cum

        for j in range(6):
            sym = cw[j]
            suf = cw[j + 1 :]
            k0 = sum(1 for x in suf if x == 0)
            k1 = len(suf) - k0
            suf_arrs = suf_table[(m, cp, k0, k1)]

            col_v = vals[4 * j : 4 * j + 4]
            actual_t = ((col_v[0] - m) >> 1) & 1
            actual_s = ((sum(col_v) - 2 * cp) >> 2) & 1
            actual_col_u = (sum(x * x for x in col_v) >> 2) if m == 0 else ((sum(x * x for x in col_v) - 4) >> 3)

            if actual_t == 1:
                for s0 in (0, 1):
                    ex0 = self.col_exact[(m, cp, 1 if sym else 0, 0)][s0]
                    sc = suf_arrs[need_st ^ s0]
                    for uc in range(min(len(ex0), budget + 1)):
                        vc = ex0[uc]
                        if vc:
                            idx += vc * sc[budget - uc]

            dp_col, base_s, r_idx = self.col_dp[(m, cp, sym, actual_t)]
            par_so_far = base_s
            u_so_far = 0
            for pos in range(4):
                x_act = col_v[pos]
                nxt_dp = dp_col[pos + 1]
                matched = False
                for sym_v, sym_u, sym_kp in self.syms[r_idx[pos]]:
                    if sym_v == x_act:
                        u_so_far += sym_u
                        par_so_far ^= sym_kp
                        matched = True
                        break
                    if u_so_far + sym_u > budget:
                        continue
                    rem_b = budget - (u_so_far + sym_u)
                    for rest_par in (0, 1):
                        s_tot = par_so_far ^ sym_kp ^ rest_par
                        sc = suf_arrs[need_st ^ (2 * actual_t + s_tot)]
                        arr = nxt_dp[rest_par]
                        for ur in range(rem_b + 1):
                            vr = arr[ur]
                            if vr:
                                idx += vr * sc[rem_b - ur]
                require(matched, "SYMBOL_NOT_FOUND", f"symbol {x_act} out of table range")

            budget -= actual_col_u
            need_st ^= 2 * actual_t + actual_s

        require(need_st == 0 and (not exact_shell or budget == 0), "RANK_INVARIANT_FAIL", "final rank state invalid")
        return idx

    def _unrank_impl(self, idx: int, m_cap: int, exact_shell: bool) -> list[int]:
        m_cap = parse_exact_int(m_cap, min_value=0, max_value=self.max_m)
        total = self.dp_shells[m_cap] if exact_shell else self.dp_ball_counts[m_cap]
        require(total > 0, "EMPTY_SHELL", f"shell m={m_cap} has 0 vectors in Lambda_24")
        idx = parse_exact_int(idx, min_value=0, max_value=total - 1)

        branches = self._branches_for_mode(m_cap, exact_shell)
        m = cp = cw_idx = -1
        for bm, bcp, bcw, boff, bcnt in branches:
            if boff <= idx < boff + bcnt:
                m, cp, cw_idx = bm, bcp, bcw
                idx -= boff
                break
        require(m >= 0, "UNRANK_BRANCH_FAIL", "index outside branch table")

        cw = HEXACODE[cw_idx]
        budget = self._branch_budget(m, m_cap)
        need_st = 2 * cp + (m ^ cp)
        suf_table = self.suf_exact if exact_shell else self.suf_cum
        out: list[int] = []

        for j in range(6):
            sym = cw[j]
            suf = cw[j + 1 :]
            k0 = sum(1 for x in suf if x == 0)
            k1 = len(suf) - k0
            suf_arrs = suf_table[(m, cp, k0, k1)]

            cnt_t0 = 0
            for s0 in (0, 1):
                ex0 = self.col_exact[(m, cp, 1 if sym else 0, 0)][s0]
                sc = suf_arrs[need_st ^ s0]
                for uc in range(min(len(ex0), budget + 1)):
                    vc = ex0[uc]
                    if vc:
                        cnt_t0 += vc * sc[budget - uc]
            if idx < cnt_t0:
                t_j = 0
            else:
                idx -= cnt_t0
                t_j = 1

            dp_col, base_s, r_idx = self.col_dp[(m, cp, sym, t_j)]
            par_so_far = base_s
            u_so_far = 0
            for pos in range(4):
                nxt_dp = dp_col[pos + 1]
                picked = False
                for sym_v, sym_u, sym_kp in self.syms[r_idx[pos]]:
                    if u_so_far + sym_u > budget:
                        continue
                    rem_b = budget - (u_so_far + sym_u)
                    ways = 0
                    for rest_par in (0, 1):
                        s_tot = par_so_far ^ sym_kp ^ rest_par
                        sc = suf_arrs[need_st ^ (2 * t_j + s_tot)]
                        arr = nxt_dp[rest_par]
                        for ur in range(rem_b + 1):
                            vr = arr[ur]
                            if vr:
                                ways += vr * sc[rem_b - ur]
                    if idx < ways:
                        out.append(sym_v)
                        u_so_far += sym_u
                        par_so_far ^= sym_kp
                        picked = True
                        break
                    idx -= ways
                require(picked, "UNRANK_SYMBOL_FAIL", "unrank exhausted symbols")

            budget -= u_so_far
            need_st ^= 2 * t_j + par_so_far

        require(idx == 0 and need_st == 0 and (not exact_shell or budget == 0), "UNRANK_STATE_FAIL", "invalid end state")
        return out

    def rank_ball(self, v: Sequence[int], m_cap: int) -> int:
        return self._rank_impl(v, m_cap, exact_shell=False)

    def unrank_ball(self, idx: int, m_cap: int) -> list[int]:
        return self._unrank_impl(idx, m_cap, exact_shell=False)

    def rank_shell(self, v: Sequence[int]) -> tuple[int, int]:
        _, _, _, _, _, shell_m = self._validate_and_parse_leech(v)
        return shell_m, self._rank_impl(v, shell_m, exact_shell=True)

    def unrank_shell(self, shell_m: int, idx: int) -> list[int]:
        return self._unrank_impl(idx, shell_m, exact_shell=True)


# ==============================================================================
# LATTICE DECODERS (Z^n, E_8_std, E_8_Had, LEECH_24)
# ==============================================================================
def _sylvester_H8() -> np.ndarray:
    H = np.array([[1.0]])
    for _ in range(3):
        H = np.block([[H, H], [H, -H]])
    return H


U_H8 = _sylvester_H8() / math.sqrt(8.0)
LEECH_UNIT_SCALE = 2.0 * math.sqrt(2.0)


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
    return np.where((((x - b) ** 2).sum(axis=1) < ((x - a) ** 2).sum(axis=1))[:, None], b, a)


def closest_E8_had(x: np.ndarray) -> np.ndarray:
    return closest_E8_std(x @ U_H8) @ U_H8


def _nearest_and_second_mod4(x: np.ndarray, r: float):
    q1 = 4.0 * np.rint((x - r) * 0.25) + r
    err = x - q1
    q2 = q1 + np.where(err >= 0.0, 4.0, -4.0)
    return q1, err * err, q2, (x - q2) ** 2


def closest_leech24_raw(X: np.ndarray) -> np.ndarray:
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
                    r_idx = m + 2 * MOG_PATTERNS_NP[cp, sym, t]
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
            for cw in HEXACODE_NP:
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
    return closest_leech24_raw(X * LEECH_UNIT_SCALE) / LEECH_UNIT_SCALE


# ==============================================================================
# PART B: EXACT BIT-BIJECTIVE SPHERE-SHAPING BENCHMARK (2.0, 3.0, 4.0 BPW)
# ==============================================================================
def _max_cap_shell(shells: list[int], cap_bits: int) -> tuple[int, int]:
    cap = 1 << cap_bits
    cum = 0
    best_m = 0
    hit = False
    for m, cnt in enumerate(shells):
        if cum + cnt <= cap:
            cum += cnt
            best_m = m
        else:
            hit = True
            break
    require(hit, "THETA_TRUNCATED", f"shell table len={len(shells)} too short for 2^{cap_bits}")
    return best_m, cum


def _bisect_free_ball_scales(X: np.ndarray, fn, r: float, rho: float, steps: int = 14) -> np.ndarray:
    """siliquant shapelib.py no-clip block bisection scale (smallest scale s where ||fn(x/s)||_2 <= r)."""
    g = np.maximum(np.sqrt((X ** 2).sum(axis=1)), 1e-30)
    lo = g / (r + rho) * 0.99
    hi = g / max(r - rho, 0.25) * 1.01
    for _ in range(steps):
        mid = np.sqrt(lo * hi)
        Y = fn(X / mid[:, None])
        ok = ((Y ** 2).sum(axis=1)) <= r * r + 1e-9
        hi = np.where(ok, mid, hi)
        lo = np.where(ok, lo, mid)
    return hi


def _empirical_entropy_bits(shells: Sequence[int]) -> float:
    counts: dict[int, int] = {}
    for s in shells:
        counts[s] = counts.get(s, 0) + 1
    n = len(shells)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def run_parts_a_and_b_leech_shaping() -> tuple[LeechCoverCodec, dict[str, Any]]:
    """Build exact LeechCoverCodec up to 4.0 bpw, verify vs Ramanujan, and benchmark vs E8^3 & Z24."""
    # Exact Ramanujan modular form & E8^3 / Z24 shell counts
    max_m_leech = 216
    ram_leech = theta_Leech24_ramanujan(max_m_leech)
    th_e8_3 = poly_pow_trunc(theta_E8_exact(max_m_leech), 3, max_m_leech, nonneg=True)
    th_z24 = theta_Zn_exact(24, 2 * max_m_leech + 20)

    # Determine exact shell caps for 2.0 (48b), 3.0 (72b), 4.0 (96b) bpw
    caps_bits = {"2.00_bpw": 48, "3.00_bpw": 72, "4.00_bpw": 96}
    m_caps: dict[str, dict[str, tuple[int, int, float]]] = {}
    for label, b_tot in caps_bits.items():
        m_z, n_z = _max_cap_shell(th_z24, b_tot)
        m_e, n_e = _max_cap_shell(th_e8_3, b_tot)
        m_l, n_l = _max_cap_shell(ram_leech, b_tot)
        m_caps[label] = {
            "Z24": (m_z, n_z, math.sqrt(float(m_z))),
            "E8^3": (m_e, n_e, math.sqrt(2.0 * m_e)),
            "Leech24": (m_l, n_l, math.sqrt(2.0 * m_l)),
        }

    # Build LeechCoverCodec once up to 4.0 bpw cap (m = 214)
    m_40 = m_caps["4.00_bpw"]["Leech24"][0]
    codec = LeechCoverCodec(m_40)

    # ZERO-SHARED-CODE CROSS-CHECK: EVERY SINGLE SHELL m = 0..214 matches Ramanujan E4^3 - 720*Delta!
    require(
        codec.dp_shells == ram_leech[: m_40 + 1],
        "RAMANUJAN_MISMATCH",
        "LeechCoverCodec MOG/Hexacode DP shells disagree with Ramanujan E4^3 - 720*Delta!",
    )

    # Bit-bijective byte-packing round-trip tests across 2.0, 3.0, and 4.0 bpw (boundary + random big ints)
    rng = np.random.default_rng(20261121)
    roundtrip_summary = {}
    for label, b_tot in caps_bits.items():
        m_cap = m_caps[label]["Leech24"][0]
        total_cnt = codec.dp_ball_counts[m_cap]
        n_bytes = b_tot // 8
        require(total_cnt <= (1 << b_tot), "BIT_BUDGET_OVERFLOW", f"{label} count exceeds 2^{b_tot}")
        test_indices = [0, 1, total_cnt - 2, total_cnt - 1]
        for _ in range(12):
            hi = int(rng.integers(0, 1 << 30))
            mid = int(rng.integers(0, 1 << 30))
            lo = int(rng.integers(0, 1 << 30))
            raw_big = ((hi << 60) | (mid << 30) | lo) % total_cnt
            test_indices.append(raw_big)
        for idx in test_indices:
            vec = codec.unrank_ball(idx, m_cap)
            # Pack to exact fixed-width bytes and unpack
            packed = idx.to_bytes(n_bytes, byteorder="little", signed=False)
            unpacked_idx = int.from_bytes(packed, byteorder="little", signed=False)
            require(unpacked_idx == idx, "BYTE_PACK_FAIL", "byte pack round-trip failed")
            idx2 = codec.rank_ball(vec, m_cap)
            require(idx2 == idx, "BALL_ROUNDTRIP_FAIL", f"rank_ball(unrank_ball({idx})) == {idx2}")
            # Also verify exact variable-shell round-trip on the exact same vector!
            sh_m, sh_idx = codec.rank_shell(vec)
            vec2 = codec.unrank_shell(sh_m, sh_idx)
            require(vec2 == vec, "SHELL_ROUNDTRIP_FAIL", "unrank_shell(rank_shell(v)) != v")
        roundtrip_summary[label] = {
            "m_cap": m_cap,
            "bytes_per_24d_block": n_bytes,
            "exact_codewords_str": str(total_cnt),
            "eff_bpw": r6(math.log2(total_cnt) / 24.0),
            "ball_and_shell_roundtrips_verified": len(test_indices),
        }

    # Head-to-head out-of-sample benchmark: Gaussian & Student-t3 (160 x 24 = 3,840 weights/dist) at 2.0, 3.0, 4.0 bpw
    decoders_24 = {
        "Z24": lambda X: np.rint(X),
        "E8^3": lambda X: closest_E8_std(X.reshape(-1, 8)).reshape(-1, 24),
        "Leech24": closest_leech24_unit,
    }
    rho_map = {"Z24": math.sqrt(6.0), "E8^3": math.sqrt(3.0), "Leech24": math.sqrt(2.0)}
    datasets = {
        "Gaussian_160x24": rng.standard_normal((160, 24)),
        "Student_t3_160x24": rng.standard_t(3.0, size=(160, 24)),
    }

    fixed_rate_bench: dict[str, Any] = {}
    for dname, X in datasets.items():
        total_w2 = float(np.sum(X ** 2))
        norms = np.sqrt((X ** 2).sum(axis=1))
        fixed_rate_bench[dname] = {}
        for label, b_tot in caps_bits.items():
            n_bytes = b_tot // 8
            row_res: dict[str, Any] = {}
            for arm, fn in decoders_24.items():
                m_cap, _, r = m_caps[label][arm]
                rho = rho_map[arm]

                # 1. Free-scale ball shaping (no-clip bisection, matching siliquant shapelib/ballpack)
                s_free = _bisect_free_ball_scales(X, fn, r, rho, steps=14)
                Y_free = fn(X / s_free[:, None])
                out_free = int(np.sum(((Y_free ** 2).sum(axis=1)) > r * r + 1e-7))
                require(out_free == 0, "FREE_OUTSIDE", f"{dname}/{label}/{arm} free outside={out_free}")
                rec_free = Y_free * s_free[:, None]
                mse_free = float(np.sum((X - rec_free) ** 2)) / total_w2

                # 2. Proved safe-scale ball shaping (matching siliquant certified.py Claims 1 & 2)
                s_safe = np.maximum(norms / (r - rho) * (1.0 + 1e-9), 1e-150)
                Y_safe = fn(X / s_safe[:, None])
                out_safe = int(np.sum(((Y_safe ** 2).sum(axis=1)) > r * r + 1e-7))
                require(out_safe == 0, "SAFE_OUTSIDE", f"{dname}/{label}/{arm} safe outside={out_safe}")
                rec_safe = Y_safe * s_safe[:, None]
                blk_err = ((X - rec_safe) ** 2).sum(axis=1)
                worst_ratio = float(np.max(blk_err / ((rho * s_safe) ** 2)))
                require(worst_ratio <= 1.0 + 1e-7, "SAFE_BOUND_FAIL", f"{arm} worst_ratio={worst_ratio}")
                mse_safe = float(blk_err.sum()) / total_w2
                bound_safe = float((rho * rho) * np.sum(s_safe ** 2)) / total_w2

                # 3. For Leech24: verify bit-exact Cover pack -> bytes -> unpack -> decode on live quantized blocks!
                exact_rt_ok = 0
                if arm == "Leech24":
                    Y_raw_int = np.rint(Y_safe * LEECH_UNIT_SCALE).astype(np.int64)
                    # Check 25 blocks across the batch (100% bit-exact byte round-trip!)
                    for bi in range(25):
                        v_int = Y_raw_int[bi].tolist()
                        code_int = codec.rank_ball(v_int, m_cap)
                        packed = code_int.to_bytes(n_bytes, "little")
                        v_back = codec.unrank_ball(int.from_bytes(packed, "little"), m_cap)
                        require(v_back == v_int, "LIVE_LEECH_PACK_FAIL", f"block {bi} mismatch")
                        exact_rt_ok += 1

                row_res[arm] = {
                    "free_mse_dB": r6(db(mse_free)),
                    "free_sqnr_dB": r6(-db(mse_free)),
                    "safe_mse_dB": r6(db(mse_safe)),
                    "safe_sqnr_dB": r6(-db(mse_safe)),
                    "proved_bound_dB": r6(db(bound_safe)),
                    "safe_outside_count": out_safe,
                    "safe_worst_ratio": r6(worst_ratio),
                }
                if arm == "Leech24":
                    row_res[arm]["live_byte_roundtrips_verified"] = exact_rt_ok

            row_res["leech_vs_e8x3_free_dB"] = r6(row_res["E8^3"]["free_mse_dB"] - row_res["Leech24"]["free_mse_dB"])
            row_res["leech_vs_e8x3_safe_dB"] = r6(row_res["E8^3"]["safe_mse_dB"] - row_res["Leech24"]["safe_mse_dB"])
            row_res["leech_vs_e8x3_bound_dB"] = r6(row_res["E8^3"]["proved_bound_dB"] - row_res["Leech24"]["proved_bound_dB"])
            row_res["leech_vs_z24_free_dB"] = r6(row_res["Z24"]["free_mse_dB"] - row_res["Leech24"]["free_mse_dB"])
            row_res["leech_vs_z24_safe_dB"] = r6(row_res["Z24"]["safe_mse_dB"] - row_res["Leech24"]["safe_mse_dB"])
            row_res["leech_vs_z24_bound_dB"] = r6(row_res["Z24"]["proved_bound_dB"] - row_res["Leech24"]["proved_bound_dB"])
            fixed_rate_bench[dname][label] = row_res

    # Variable-shell Cover shaping benchmark (shared global scale across all 300 blocks -> entropy + log2(N_m) rate!)
    # Compare on Gaussian & Student-t3 (where heavy tails make variable-shell shaping shine vs fixed ball!)
    var_shell_bench: dict[str, Any] = {}
    for dname, X in datasets.items():
        total_w2 = float(np.sum(X ** 2))
        max_norm = float(np.sqrt((X ** 2).sum(axis=1)).max())
        # Pick ONE shared scale per matrix so ALL 300 blocks fit inside m <= 200 (safe, zero clipping!)
        s_global = max_norm / (math.sqrt(2.0 * 180) - math.sqrt(6.0))
        vs_row: dict[str, Any] = {}
        for arm, fn, shells_arr, step_r2 in [
            ("Z24", decoders_24["Z24"], th_z24, 1.0),
            ("E8^3", decoders_24["E8^3"], th_e8_3, 2.0),
            ("Leech24", decoders_24["Leech24"], ram_leech, 2.0),
        ]:
            Y = fn(X / s_global)
            rec = Y * s_global
            mse = float(np.sum((X - rec) ** 2)) / total_w2
            sqnr = -db(mse)
            # Exact shell indices m_i = round(||Y_i||^2 / step_r2)
            shell_ids = np.rint(((Y ** 2).sum(axis=1)) / step_r2).astype(np.int64).tolist()
            h_shell = _empirical_entropy_bits(shell_ids)
            in_shell_bits = [0.0 if shells_arr[m_i] <= 1 else math.log2(shells_arr[m_i]) for m_i in shell_ids]
            mean_var_bpw = (h_shell + float(np.mean(in_shell_bits))) / 24.0
            # Compare against fixed-ball rate needed to hold max(shell_ids) with shared scale!
            m_peak = max(shell_ids)
            fixed_cum = sum(shells_arr[: m_peak + 1])
            fixed_bpw = math.log2(fixed_cum) / 24.0
            # Rate-distortion figure of merit: SQNR - 6.0206 * bpw (dB above 2^(-2R) normalized baseline!)
            rd_var = sqnr - 6.020599913279624 * mean_var_bpw
            rd_fix = sqnr - 6.020599913279624 * fixed_bpw

            live_shell_rt = 0
            if arm == "Leech24":
                Y_raw_int = np.rint(Y * LEECH_UNIT_SCALE).astype(np.int64)
                for bi in range(25):
                    v_int = Y_raw_int[bi].tolist()
                    sm, sidx = codec.rank_shell(v_int)
                    require(sm == shell_ids[bi], "SHELL_ID_MISMATCH", f"shell {sm} != {shell_ids[bi]}")
                    require(codec.unrank_shell(sm, sidx) == v_int, "LIVE_SHELL_RT_FAIL", f"block {bi}")
                    live_shell_rt += 1

            vs_row[arm] = {
                "sqnr_dB": r6(sqnr),
                "var_shell_bpw": r6(mean_var_bpw),
                "fixed_global_bpw": r6(fixed_bpw),
                "var_shell_bit_savings_bpw": r6(fixed_bpw - mean_var_bpw),
                "rd_efficiency_var_dB": r6(rd_var),
                "rd_efficiency_fix_dB": r6(rd_fix),
            }
            if arm == "Leech24":
                vs_row[arm]["live_shell_roundtrips_verified"] = live_shell_rt

        vs_row["leech_vs_e8x3_var_rd_gain_dB"] = r6(
            vs_row["Leech24"]["rd_efficiency_var_dB"] - vs_row["E8^3"]["rd_efficiency_var_dB"]
        )
        vs_row["leech_vs_z24_var_rd_gain_dB"] = r6(
            vs_row["Leech24"]["rd_efficiency_var_dB"] - vs_row["Z24"]["rd_efficiency_var_dB"]
        )
        var_shell_bench[dname] = vs_row

    return codec, {
        "ramanujan_exact_shell_match_count": m_40 + 1,
        "ramanujan_max_shell_verified": m_40,
        "codec_roundtrips": roundtrip_summary,
        "fixed_rate_24d_benchmark": fixed_rate_bench,
        "variable_shell_24d_benchmark": var_shell_bench,
    }


# ==============================================================================
# PART C: SP-01 HESSIAN-WEIGHTED E8_std vs. E8_Had + ZERO-FLOAT OUTLIER SPLITTING
# ==============================================================================
def to_integers_dyadic_exact(W: np.ndarray) -> tuple[list[list[int]], int]:
    """Exact lossless IEEE-754 float -> Z * 2^e (from siliquant shape/shapelib.py)."""
    W = np.asarray(W, dtype=np.float64)
    require(W.ndim == 2 and W.shape[0] > 0 and W.shape[1] > 0, "INVALID_W", "W must be non-empty 2D")
    require(bool(np.all(np.isfinite(W))), "NONFINITE_W", "W contains non-finite values")
    m, n = W.shape
    ratios = [[float(W[i, j]).as_integer_ratio() for j in range(n)] for i in range(m)]
    max_exp = max(d.bit_length() - 1 for row in ratios for _, d in row)
    Z = [[num << (max_exp - (den.bit_length() - 1)) for num, den in row] for row in ratios]
    return Z, -max_exp


def certify_phi_sos_exact(
    moments: dict[int, Fraction], x0: Fraction, tau_shift: Fraction, b: int = 0
) -> dict[str, Any]:
    """AUDIT-014 Theorem A3 exact rational quartic SOS bound (zero floats!)."""
    x0_f = parse_exact_fraction(x0)
    tau_f = parse_exact_fraction(tau_shift)
    b_i = parse_exact_int(b, min_value=0)
    require(x0_f > 0, "INVALID_X0", "x0 must be > 0")
    require(tau_f >= 0, "NEGATIVE_TAU", "tau_shift must be >= 0")
    for k in (1, 2, 3, 4):
        require(k in moments, "MISSING_MOMENT", f"missing moment S_{k}")
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
    return {"bound": bound, "min_rank_cert": ceil_fraction(bound)}


def zero_float_certify_hessian_outliers(W: np.ndarray, h_diag: np.ndarray) -> dict[str, Any]:
    """
    Zero-float exact certification of above-bulk active outlier directions on W_eff = W * diag(sqrt(h_diag)):
      Uses siliquant's lossless dyadic to_integers + exact integer Gram G = Z Z^T + AUDIT-014 Phi_{x0,tau}
      (zero SVD, zero floats in TCB!).
    """
    h_arr = np.asarray(h_diag, dtype=np.float64)
    require(h_arr.ndim == 1 and len(h_arr) == W.shape[1], "H_DIM_MISMATCH", "h_diag length must match W cols")
    require(bool(np.all(np.isfinite(h_arr))) and bool(np.all(h_arr > 0.0)), "INVALID_HESSIAN", "h_diag must be finite > 0")

    W_eff = (W * np.sqrt(h_arr)[None, :]).astype(np.float16).astype(np.float64)
    Z, exp2 = to_integers_dyadic_exact(W_eff)
    m, n = len(Z), len(Z[0])

    # Exact integer Gram G = Z @ Z^T (m x m)
    G = [[0] * m for _ in range(m)]
    for i in range(m):
        zi = Z[i]
        for j in range(i, m):
            s = sum(a * b for a, b in zip(zi, Z[j]))
            G[i][j] = s
            G[j][i] = s

    # Exact integer median column norm^2 -> rational Marchenko-Pastur bulk ceiling
    col_sq = sorted(sum(Z[i][j] * Z[i][j] for i in range(m)) for j in range(n))
    med_col_sq = Fraction(col_sq[n // 2])
    # Exact rational upper bound for (1 + sqrt(m/n))^2: since m/n <= 1, (1 + sqrt(m/n))^2 <= (1 + p/q)^2
    # For m=16, n=64: sqrt(16/64) = 1/2 EXACT => (1 + 1/2)^2 = 9/4! General rational bound: 1 + 2*isqrt_hi + m/n
    mp_factor = Fraction(9, 4) if (4 * m == n) else Fraction(4, 1)
    tau_mp = Fraction(n, m) * med_col_sq * mp_factor * Fraction(5, 4)

    H = [[Fraction(G[i][j]) - (tau_mp if i == j else Fraction(0)) for j in range(m)] for i in range(m)]
    S1 = sum((H[i][i] for i in range(m)), Fraction(0))
    H2 = [[sum((H[i][t] * H[t][j] for t in range(m)), Fraction(0)) for j in range(m)] for i in range(m)]
    S2 = sum((H2[i][i] for i in range(m)), Fraction(0))
    S3 = sum((H2[i][j] * H[j][i] for i in range(m) for j in range(m)), Fraction(0))
    S4 = sum((H2[i][j] * H2[i][j] for i in range(m) for j in range(m)), Fraction(0))
    moms = {1: S1, 2: S2, 3: S3, 4: S4}

    best = {"bound": Fraction(0), "min_rank_cert": 0}
    if S3 > 0:
        x0_base = S4 / S3
        for mult in (Fraction(1, 2), Fraction(3, 4), Fraction(1), Fraction(5, 4)):
            cand = certify_phi_sos_exact(moms, x0_base * mult, tau_mp, b=0)
            if cand["bound"] > best["bound"]:
                best = cand

    return {
        "dyadic_exponent": exp2,
        "certified_min_outlier_spikes": best["min_rank_cert"],
        "exact_bound_str": frac_str(best["bound"]),
        "exact_bound_approx": r6(float(best["bound"])),
    }


def _quantize_matrix_8d_ball(
    W: np.ndarray, fn, r: float, rho: float, group_size: int = 32
) -> tuple[np.ndarray, np.ndarray]:
    """No-clip 8D ball-cut quantization with one scale per `group_size` weights (returns W_hat, scales_per_weight)."""
    m_rows, n_cols = W.shape
    X = W.reshape(-1, 8)
    s_blk = _bisect_free_ball_scales(X, fn, r, rho, steps=14)
    per = group_size // 8
    s_grp = s_blk.reshape(-1, per).max(axis=1)
    s_rep = np.repeat(s_grp, per)
    Y = fn(X / s_rep[:, None])
    out = ((Y ** 2).sum(axis=1)) > r * r + 1e-7
    if np.any(out):
        xo = X[out] / s_rep[out, None]
        lo = np.zeros(len(xo))
        hi = np.ones(len(xo))
        for _ in range(12):
            mid = 0.5 * (lo + hi)
            good = ((fn(xo * mid[:, None]) ** 2).sum(axis=1)) <= r * r + 1e-7
            lo = np.where(good, mid, lo)
            hi = np.where(good, hi, mid)
        Y[out] = fn(xo * lo[:, None])
    W_hat = (Y * s_rep[:, None]).reshape(m_rows, n_cols)
    scales_full = np.repeat(s_rep, 8).reshape(m_rows, n_cols)
    return W_hat, scales_full


def run_part_c_sp01_hessian_transformer() -> dict[str, Any]:
    """SP-01 Hessian-weighted E8_std vs E8_Had + zero-float outlier splitting on Transformer weight matrices."""
    rng = np.random.default_rng(20261122)

    # Exact 8D ball radii for 4.0 bpw (32 bits/8D block -> +0.5 bpw for 16-bit scale/32 weights = 4.5 bpw!)
    th_z8 = theta_Zn_exact(8, 220)
    th_e8 = theta_E8_exact(120)
    m_z8, n_z8 = _max_cap_shell(th_z8, 32)
    m_e8, n_e8 = _max_cap_shell(th_e8, 32)
    r_z8, r_e8 = math.sqrt(float(m_z8)), math.sqrt(2.0 * m_e8)
    # Tighter 3.5 bpw (28 bits/8D) used for kept-column split arm so TOTAL bits <= 4.5 bpw even after keeping 4 cols!
    m_e8_lo, n_e8_lo = _max_cap_shell(th_e8, 28)
    r_e8_lo = math.sqrt(2.0 * m_e8_lo)

    # Matrix dimensions: 16 x 64 (exact 1:4 ratio for fast exact Fraction Gram & realistic ffn_down aspect!)
    m_rows, n_cols = 16, 64

    # Regime 1: Ill-conditioned + heavy-tailed + sparse-tiny Transformer ffn_down
    #   - 4 super-active hidden units (4/64 = 6.25%) with activation variance h_j = 64.0 (8x RMS, like Qwen2.5!)
    #   - Sparse tiny weights (65% entries with |w| ~ 0.03) + heavy Student-t3 bulk + strong active columns!
    h_ill = np.ones(n_cols, dtype=np.float64)
    active_cols = [3, 19, 41, 58]
    for cj in active_cols:
        h_ill[cj] = 64.0

    # Construct W_ill: mixture of tiny weights (modeling SmolLM2 PR #17!) and heavy-tailed bulk + 4 orthogonal active spikes
    is_tiny = rng.random((m_rows, n_cols)) < 0.65
    W_ill = np.where(
        is_tiny,
        rng.standard_normal((m_rows, n_cols)) * 0.03,
        rng.standard_t(3.5, size=(m_rows, n_cols)) * 0.35,
    )
    # 4 orthogonal active-unit directions (comparable variance like Qwen2.5 top ffn_down units)
    Q_act, _ = np.linalg.qr(rng.standard_normal((m_rows, len(active_cols))))
    for idx_a, cj in enumerate(active_cols):
        W_ill[:, cj] = 1.8 * Q_act[:, idx_a] + rng.standard_normal(m_rows) * 0.05
    W_ill = W_ill.astype(np.float16).astype(np.float64)

    # Regime 2: Isotropic Gaussian null control (h_diag = 1, pure Gaussian bulk)
    h_iso = np.ones(n_cols, dtype=np.float64)
    W_iso = (rng.standard_normal((m_rows, n_cols)) * 0.35).astype(np.float16).astype(np.float64)

    # Regime 3: SP-01 column-permuted copy of (W_ill, h_ill)
    perm = rng.permutation(n_cols)
    W_ill_perm = W_ill[:, perm]
    h_ill_perm = h_ill[perm]

    # Zero-float outlier certification on all 3 regimes!
    cert_ill = zero_float_certify_hessian_outliers(W_ill, h_ill)
    cert_iso = zero_float_certify_hessian_outliers(W_iso, h_iso)
    cert_perm = zero_float_certify_hessian_outliers(W_ill_perm, h_ill_perm)

    require(cert_iso["certified_min_outlier_spikes"] == 0, "ISO_NULL_FP", "isotropic null false positive")
    require(cert_ill["certified_min_outlier_spikes"] >= 4, "ILL_SPIKE_MISS", f"expected >=4 spikes, got {cert_ill}")
    require(cert_ill["exact_bound_str"] == cert_perm["exact_bound_str"], "SP01_PERM_MISMATCH", "permutation broke exact bound!")

    def _eval_arms(W: np.ndarray, h_diag: np.ndarray, k_keep: int) -> dict[str, Any]:
        h_sqrt = np.sqrt(h_diag)
        w2_unw = float(np.sum(W ** 2))
        w2_beh = float(np.sum((W * h_sqrt[None, :]) ** 2))

        def _score(W_hat: np.ndarray, scales: np.ndarray, bpw_val: float) -> dict[str, float]:
            E = W - W_hat
            unw_err = float(np.sum(E ** 2)) / w2_unw
            beh_err = float(np.sum((E * h_sqrt[None, :]) ** 2)) / w2_beh
            # Tiny-weight phantom-jump ratio on weights with |W| < 0.05 * scale (SmolLM2 PR #17 metric!)
            tiny = (np.abs(W) > 1e-12) & (np.abs(W) < 0.05 * scales)
            if np.any(tiny):
                tiny_ratio = float(np.mean(E[tiny] ** 2) / np.mean(W[tiny] ** 2))
                tiny_zero_pct = float(np.mean(np.abs(W_hat[tiny]) < 1e-12) * 100.0)
            else:
                tiny_ratio = 1.0
                tiny_zero_pct = 100.0
            return {
                "bpw": r6(bpw_val),
                "unweighted_mse_dB": r6(db(unw_err)),
                "behavioral_hessian_mse_dB": r6(db(beh_err)),
                "tiny_weight_mse_over_w2": r6(tiny_ratio),
                "tiny_weight_exact_zero_pct": r6(tiny_zero_pct),
            }

        bpw_45 = 4.0 + 16.0 / 32.0  # 4.50 bpw (32 bits/8D + 16b scale / 32 weights)
        out: dict[str, Any] = {}

        # 1-3: Unweighted arms (Z8, E8_std, E8_Had)
        Wh, sc = _quantize_matrix_8d_ball(W, lambda X: np.rint(X), r_z8, math.sqrt(2.0))
        out["Z8_ball_unweighted"] = _score(Wh, sc, bpw_45)

        Wh, sc = _quantize_matrix_8d_ball(W, closest_E8_std, r_e8, 1.0)
        out["E8_std_unweighted"] = _score(Wh, sc, bpw_45)

        Wh, sc = _quantize_matrix_8d_ball(W, closest_E8_had, r_e8, 1.0)
        out["E8_Had_unweighted"] = _score(Wh, sc, bpw_45)

        # 4-5: Naive diagonal Hessian preconditioning (W_eff = W * diag(sqrt(h)) inside raw g=32 groups)
        # EXPOSES THE GROUP-SCALE BLOAT FAILURE MODE when k_active < group_size (explaining siliquant's column-scaling drop!)
        W_eff = W * h_sqrt[None, :]
        Wh_eff, sc_eff = _quantize_matrix_8d_ball(W_eff, closest_E8_std, r_e8, 1.0)
        out["E8_std_naive_hessian_bloat"] = _score(Wh_eff / h_sqrt[None, :], sc_eff / h_sqrt[None, :], bpw_45)

        Wh_eff, sc_eff = _quantize_matrix_8d_ball(W_eff, closest_E8_had, r_e8, 1.0)
        out["E8_Had_naive_hessian_bloat"] = _score(Wh_eff / h_sqrt[None, :], sc_eff / h_sqrt[None, :], bpw_45)

        # 6: Certified Outlier-Column Split (zero-float to_integers + Phi_{x0,tau}) + E8_Had on rest at STRICTLY <= 4.50 bpw!
        # Isolates the exact k_keep certified active outlier columns (fp16) BEFORE group scaling so they cannot bloat neighbor scales!
        if k_keep > 0:
            col_scores = np.sum(W_eff ** 2, axis=0)
            top_cols = np.argsort(col_scores)[-k_keep:]
            keep_mask = np.zeros(n_cols, dtype=bool)
            keep_mask[top_cols] = True
            W_eff_rest = W_eff.copy()
            W_eff_rest[:, keep_mask] = 0.0
            Wh_rest_eff, sc_rest = _quantize_matrix_8d_ball(W_eff_rest, closest_E8_had, r_e8_lo, 1.0, group_size=128)
            Wh_split = Wh_rest_eff / h_sqrt[None, :]
            Wh_split[:, keep_mask] = W[:, keep_mask]
            bpw_split = (k_keep * 16.0 + (n_cols - k_keep) * (3.5 + 16.0 / 128.0)) / n_cols
            require(bpw_split < bpw_45, "BIT_BUDGET_EXCEEDED", f"split bpw {bpw_split} >= {bpw_45}")
            out["E8_Had_certified_split_hessian"] = _score(Wh_split, sc_rest / h_sqrt[None, :], bpw_split)
        return out

    arms_ill = _eval_arms(W_ill, h_ill, k_keep=cert_ill["certified_min_outlier_spikes"])
    arms_iso = _eval_arms(W_iso, h_iso, k_keep=cert_iso["certified_min_outlier_spikes"])

    # Verify all three genuine empirical findings:
    #   1. Unweighted E8_Had beats E8_std on ill-conditioned tiny+heavy Transformer behavioral error!
    #   2. Naive unsorted Hessian preconditioning HURTS behavioral error (< 0 dB) due to k < g group-scale bloat!
    #   3. Zero-float Phi_{x0,tau} certified outlier split + E8_Had beats BOTH unweighted E8_std AND naive Hessian at STRICTLY fewer bits!
    #   4. Isotropic H=I null control has 0.000 dB difference and 0 certified spikes!
    e8had_vs_std_unw = arms_ill["E8_std_unweighted"]["behavioral_hessian_mse_dB"] - arms_ill["E8_Had_unweighted"]["behavioral_hessian_mse_dB"]
    naive_bloat_std = arms_ill["E8_std_unweighted"]["behavioral_hessian_mse_dB"] - arms_ill["E8_std_naive_hessian_bloat"]["behavioral_hessian_mse_dB"]
    naive_bloat_had = arms_ill["E8_Had_unweighted"]["behavioral_hessian_mse_dB"] - arms_ill["E8_Had_naive_hessian_bloat"]["behavioral_hessian_mse_dB"]
    split_vs_unw_std = arms_ill["E8_std_unweighted"]["behavioral_hessian_mse_dB"] - arms_ill["E8_Had_certified_split_hessian"]["behavioral_hessian_mse_dB"]
    split_vs_naive_had = arms_ill["E8_Had_naive_hessian_bloat"]["behavioral_hessian_mse_dB"] - arms_ill["E8_Had_certified_split_hessian"]["behavioral_hessian_mse_dB"]
    iso_diff = abs(arms_iso["E8_Had_unweighted"]["behavioral_hessian_mse_dB"] - arms_iso["E8_Had_naive_hessian_bloat"]["behavioral_hessian_mse_dB"])

    require(e8had_vs_std_unw > 0.5, "E8HAD_BEH_LOW", f"expected E8_Had > E8_std by >0.5 dB, got {e8had_vs_std_unw}")
    require(naive_bloat_had < -1.0, "BLOAT_NOT_EXPOSED", f"expected naive Hessian bloat < -1 dB, got {naive_bloat_had}")
    require(split_vs_unw_std > 3.0, "SPLIT_GAIN_LOW", f"expected certified split > +3 dB over E8_std, got {split_vs_unw_std}")
    require(iso_diff < 1e-6, "ISO_NULL_DRIFT", f"isotropic H=I shifted by {iso_diff}")

    return {
        "ball_radii_8d": {"Z8_32b": r6(r_z8), "E8_32b": r6(r_e8), "E8_28b": r6(r_e8_lo)},
        "zero_float_outlier_certificates": {
            "ill_conditioned_transformer": cert_ill,
            "isotropic_gaussian_null": cert_iso,
            "sp01_col_permuted_exact_match": cert_ill["exact_bound_str"] == cert_perm["exact_bound_str"],
        },
        "regime1_ill_conditioned_heavy_tailed": {
            "arms": arms_ill,
            "unweighted_E8_Had_vs_E8_std_behavioral_dB": r6(e8had_vs_std_unw),
            "naive_hessian_group_bloat_E8_std_dB": r6(naive_bloat_std),
            "naive_hessian_group_bloat_E8_Had_dB": r6(naive_bloat_had),
            "certified_split_E8_Had_vs_unweighted_E8_std_dB": r6(split_vs_unw_std),
            "certified_split_E8_Had_vs_naive_hessian_E8_Had_dB": r6(split_vs_naive_had),
            "unweighted_tiny_mse_drop_E8_Had_vs_E8_std_x": r6(
                arms_ill["E8_std_unweighted"]["tiny_weight_mse_over_w2"]
                / max(1e-12, arms_ill["E8_Had_unweighted"]["tiny_weight_mse_over_w2"])
            ),
        },
        "regime2_isotropic_null_control": {
            "arms": arms_iso,
            "hessian_vs_unweighted_diff_dB": r6(iso_diff),
        },
    }


# ==============================================================================
# PART D: R5 MUST-FAIL NEGATIVE CONTROLS (8/8 REJECTED)
# ==============================================================================
def run_part_d_r5_controls(codec: LeechCoverCodec) -> dict[str, Any]:
    # Non-Leech deep hole (4, 0^23)
    deep_hole = [4] + [0] * 23
    # Valid Leech vector on shell m=14 (outside 2.0 bpw cap m=13!): (8, 8, 8, 4, 4, 0^19)?
    # Check: (12, 8, 4, 0^21): norm^2 = 144 + 64 + 16 = 224 = 16 * 14, all === 0 mod 4, sum = 24 === 0 mod 8 -> VALID Leech shell m=14!
    valid_m14 = [12, 8, 4] + [0] * 21
    checks = [
        ("C1_non_leech_rejected_by_rank_ball", "NOT_IN_LEECH", lambda: codec.rank_ball(deep_hole, 13)),
        ("C2_non_leech_rejected_by_rank_shell", "NOT_IN_LEECH", lambda: codec.rank_shell(deep_hole)),
        ("C3_out_of_ball_m14_rejected_at_m13", "OUT_OF_BALL", lambda: codec.rank_ball(valid_m14, 13)),
        ("C4_out_of_range_index_unrank_ball", "INT_OUT_OF_RANGE", lambda: codec.unrank_ball(codec.dp_ball_counts[13], 13)),
        ("C5_empty_shell_m1_rejected_unrank_shell", "EMPTY_SHELL", lambda: codec.unrank_shell(1, 0)),
        ("C6_bool_rejected_in_unrank", "INVALID_TYPE", lambda: codec.unrank_ball(True, 13)),  # type: ignore[arg-type]
        ("C7_float_rejected_in_phi_sos", "INVALID_TYPE", lambda: certify_phi_sos_exact({1: Fraction(1), 2: Fraction(1), 3: Fraction(1), 4: Fraction(1)}, 1.5, Fraction(1))),  # type: ignore[arg-type]
        ("C8_nonpositive_hessian_rejected", "INVALID_HESSIAN", lambda: zero_float_certify_hessian_outliers(np.ones((4, 8)), np.array([-1.0] + [1.0] * 7))),
    ]
    results = []
    for name, exp, fn in checks:
        got = None
        try:
            fn()
        except CertError as exc:
            got = exc.code
        require(got == exp, "R5_FAIL", f"{name}: expected {exp}, got {got}")
        results.append({"control": name, "expected": exp, "caught": got, "rejected": True})
    return {"controls_passed": len(results), "controls_total": len(results), "controls": results}


def run_all() -> dict[str, Any]:
    codec, ab_res = run_parts_a_and_b_leech_shaping()
    c_res = run_part_c_sp01_hessian_transformer()
    d_res = run_part_d_r5_controls(codec)
    payload = {
        "schema": "aletheia.wave2.round2.siliquant_leech_shaping.v1",
        "parts_ab_leech_cover_shaping": ab_res,
        "part_c_sp01_hessian_transformer": c_res,
        "part_d_r5_controls": d_res,
        "verdict": "ALL_ROUND2_PARTS_VERIFIED",
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["canonical_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Wave 2 Round 2 (Arm 1): Leech Cover Sphere Shaping + SP-01 Hessian E8_Had.")
    parser.add_argument("--json", action="store_true", help="Emit canonical deterministic JSON.")
    parser.add_argument("--write-cert", type=str, default="", help="Write canonical JSON certificate to path.")
    args = parser.parse_args(argv)
    res = run_all()
    txt = json.dumps(res, indent=2, sort_keys=True)
    if args.write_cert:
        Path(args.write_cert).write_text(txt + "\n", encoding="utf-8")
    if args.json:
        print(txt)
    else:
        ab = res["parts_ab_leech_cover_shaping"]
        c = res["part_c_sp01_hessian_transformer"]
        d = res["part_d_r5_controls"]
        print("WAVE 2 ROUND 2 (ARM 1): LEECH LAMBDA_24 COVER SPHERE SHAPING + SP-01 HESSIAN E8_Had")
        print(f"  [Part A] MOG/Hexacode Cover DP matched Ramanujan E4^3-720*Delta on ALL {ab['ramanujan_exact_shell_match_count']} shells (m=0..{ab['ramanujan_max_shell_verified']})!")
        for lbl, rt in ab["codec_roundtrips"].items():
            print(f"           {lbl}: m<={rt['m_cap']} ({rt['bytes_per_24d_block']}B, {rt['eff_bpw']} bpw, N={rt['exact_codewords_str']}) -> {rt['ball_and_shell_roundtrips_verified']}/{rt['ball_and_shell_roundtrips_verified']} bit-exact round-trips")
        for dname in ("Gaussian_160x24", "Student_t3_160x24"):
            f2 = ab["fixed_rate_24d_benchmark"][dname]["2.00_bpw"]
            f4 = ab["fixed_rate_24d_benchmark"][dname]["4.00_bpw"]
            vs = ab["variable_shell_24d_benchmark"][dname]
            print(f"  [Part B] {dname}:")
            print(f"           2.0 bpw Leech vs E8^3: free +{f2['leech_vs_e8x3_free_dB']} dB | safe +{f2['leech_vs_e8x3_safe_dB']} dB | bound +{f2['leech_vs_e8x3_bound_dB']} dB (vs Z24 safe +{f2['leech_vs_z24_safe_dB']} dB)")
            print(f"           4.0 bpw Leech vs E8^3: free +{f4['leech_vs_e8x3_free_dB']} dB | safe +{f4['leech_vs_e8x3_safe_dB']} dB | bound +{f4['leech_vs_e8x3_bound_dB']} dB (vs Z24 safe +{f4['leech_vs_z24_safe_dB']} dB)")
            print(f"           Var-shell shaping Leech: {vs['Leech24']['var_shell_bpw']} bpw (saves {vs['Leech24']['var_shell_bit_savings_bpw']} bpw vs fixed global!), RD +{vs['leech_vs_e8x3_var_rd_gain_dB']} dB vs E8^3 / +{vs['leech_vs_z24_var_rd_gain_dB']} dB vs Z24")
        r1 = c["regime1_ill_conditioned_heavy_tailed"]
        print(f"  [Part C] SP-01 Ill-Conditioned Heavy-Tailed Transformer (zero-float cert={c['zero_float_outlier_certificates']['ill_conditioned_transformer']['certified_min_outlier_spikes']} spikes, iso_null={c['zero_float_outlier_certificates']['isotropic_gaussian_null']['certified_min_outlier_spikes']}, SP01_perm={c['zero_float_outlier_certificates']['sp01_col_permuted_exact_match']}):")
        print(f"           Unweighted E8_Had vs E8_std behavioral gain: +{r1['unweighted_E8_Had_vs_E8_std_behavioral_dB']} dB (tiny MSE drop {r1['unweighted_tiny_mse_drop_E8_Had_vs_E8_std_x']}x)")
        print(f"           Naive Hessian precond group-scale bloat (k<g): E8_std {r1['naive_hessian_group_bloat_E8_std_dB']} dB | E8_Had {r1['naive_hessian_group_bloat_E8_Had_dB']} dB")
        print(f"           Certified split + E8_Had (4.398 < 4.500 bpw): +{r1['certified_split_E8_Had_vs_unweighted_E8_std_dB']} dB vs unweighted E8_std | +{r1['certified_split_E8_Had_vs_naive_hessian_E8_Had_dB']} dB vs naive Hessian!")
        print(f"  [Part D] R5 Controls: {d['controls_passed']}/{d['controls_total']} rejected")
        print(f"  VERDICT: {res['verdict']} | SHA-256: {res['canonical_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
