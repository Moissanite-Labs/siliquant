#!/usr/bin/env python3
"""
Track S21a (`AUDIT-121A`): Exact-Q Conformal / Log-Radial Coxeter-Shell Packing
vs. Flat Euclidean Lattice Balls in R^8 (`conformal_coxeter_oracle.py`).

Pure-stdlib exact Z/Q arithmetic (ZERO floats on all trust/certificate paths,
ZERO bare `assert`s, `-O`-safe, full Zone-B `strict_cert_io` hardening):
  1. Exact theta-series ball cardinalities & unused complete-shell slack at
     matched 3.0 bpw (2^23) and 4.5 bpw (2^35) per-8-block budgets for E_8/E_8^Had and Z^8;
  2. Exact scale-invariant S^7 angular code comparison across all 5 Coxeter root
     families (Z^8, D_8, E_8, E_8^Had, Block-H2) plus E_8^Had Shells m=1,2,3;
  3. Exact log-radial ladder distortion over Q (rho in {2, 3/2, 5/4, 9/8, 17/16})
     and exact bulk bit-tax obstruction + 16-block group crossover threshold for
     pure uniform-shell conformal packing;
  4. Graded conformal-wing Coxeter packing (59,520 wing points packed strictly
     inside the 729,007 / 180,292,927 unused complete-shell slots at 0 extra
     per-block bits) closing inner-cavity (E1), inner-Steiner (E4), and heavy-tail
     outlier (E3) walls simultaneously with 0 bulk loss (E2);
  5. Honest Law-2b bulk-Steiner obstruction witness (E5): proves radial wings
     do NOT eliminate interior bulk extended-Hamming parity coupling on (3,1,1,eps,0^4).
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import hmac
import json
from math import isqrt, lcm
import sys
from typing import Any

MAX_JSON_DEPTH = 64
MAX_JSON_BYTES = 2_000_000
MAX_INT_BITS = 2048
MAX_INT_DIGITS = 620
_HEX_LOWER = frozenset("0123456789abcdef")

_H8_CODEWORDS: frozenset[tuple[int, ...]] = frozenset(
    {
        (0, 0, 0, 0, 0, 0, 0, 0),
        (1, 1, 1, 1, 1, 1, 1, 1),
        (1, 1, 1, 1, 0, 0, 0, 0),
        (1, 1, 0, 0, 1, 1, 0, 0),
        (1, 1, 0, 0, 0, 0, 1, 1),
        (1, 0, 1, 0, 1, 0, 1, 0),
        (1, 0, 1, 0, 0, 1, 0, 1),
        (1, 0, 0, 1, 1, 0, 0, 1),
        (1, 0, 0, 1, 0, 1, 1, 0),
        (0, 1, 1, 0, 1, 0, 0, 1),
        (0, 1, 1, 0, 0, 1, 1, 0),
        (0, 1, 0, 1, 1, 0, 1, 0),
        (0, 1, 0, 1, 0, 1, 0, 1),
        (0, 0, 1, 1, 1, 1, 0, 0),
        (0, 0, 1, 1, 0, 0, 1, 1),
        (0, 0, 0, 0, 1, 1, 1, 1),
    }
)


class ConformalError(ValueError):
    """Fail-closed typed validation error (never stripped under `python3 -O`)."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


def _require_valid_utf8_str(s: str, label: str = "str") -> str:
    try:
        s.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ConformalError("INVALID_UTF8", f"{label} contains unpaired UTF-16 surrogate") from exc
    return s


def require_int(val: Any, label: str = "int", *, lo: int | None = None, hi: int | None = None) -> int:
    if type(val) is not int:
        raise ConformalError("INVALID_INT_TYPE", f"{label} must be strict int, got {type(val).__name__}")
    if lo is not None and type(lo) is not int:
        raise ConformalError("INVALID_BOUND_TYPE", f"{label} lower bound must be strict int")
    if hi is not None and type(hi) is not int:
        raise ConformalError("INVALID_BOUND_TYPE", f"{label} upper bound must be strict int")
    blen = val.bit_length()
    if blen > MAX_INT_BITS:
        raise ConformalError("INT_BIT_LIMIT", f"{label} bit_length={blen} exceeds {MAX_INT_BITS}")
    if lo is not None and val < lo:
        raise ConformalError("INT_BELOW_MIN", f"{label} bit_length={blen} below min_bits={lo.bit_length()}")
    if hi is not None and val > hi:
        raise ConformalError("INT_ABOVE_MAX", f"{label} bit_length={blen} above max_bits={hi.bit_length()}")
    return val


def validate_cli_argv(argv: Any) -> list[str] | None:
    if argv is None:
        return None
    if type(argv) not in (list, tuple):
        raise ConformalError("INVALID_ARGV", "argv must be list or tuple of strict str")
    out: list[str] = []
    for item in argv:
        if type(item) is not str:
            raise ConformalError("INVALID_ARGV", "argv entries must be strict str")
        _require_valid_utf8_str(item, "argv")
        out.append(item)
    return out


def require_hex64_ascii(val: Any, label: str = "sha256") -> str:
    if type(val) is not str:
        raise ConformalError("INVALID_DIGEST_TYPE", f"{label} must be strict str")
    if len(val) != 64:
        raise ConformalError("INVALID_DIGEST_LEN", f"{label} must have length 64")
    for ch in val:
        if ch not in _HEX_LOWER:
            raise ConformalError("INVALID_DIGEST_HEX", f"{label} must be lowercase ASCII hex")
    return val


def constant_time_hex64_equal(left: Any, right: Any, label: str = "sha256") -> bool:
    a = require_hex64_ascii(left, f"{label}.left")
    b = require_hex64_ascii(right, f"{label}.right")
    return hmac.compare_digest(a, b)


def _require_frac(val: Any, label: str = "q") -> Fraction:
    if type(val) is not Fraction:
        raise ConformalError("INVALID_FRACTION_TYPE", f"{label} must be strict Fraction")
    nb = val.numerator.bit_length()
    db = val.denominator.bit_length()
    if nb > MAX_INT_BITS or db > MAX_INT_BITS:
        raise ConformalError("FRACTION_BIT_LIMIT", f"{label} bit_lengths=({nb},{db}) exceed {MAX_INT_BITS}")
    return val


def _require_int_vec8(vec: Any, label: str = "vec") -> tuple[int, ...]:
    if type(vec) not in (tuple, list):
        raise ConformalError("INVALID_VEC_TYPE", f"{label} must be strict tuple or list")
    if len(vec) != 8:
        raise ConformalError("INVALID_DIM", f"{label} must have length 8")
    return tuple(require_int(c, f"{label}_i") for c in vec)


def _require_frac_vec8(vec: Any, label: str = "y_q") -> tuple[Fraction, ...]:
    if type(vec) not in (tuple, list):
        raise ConformalError("INVALID_VEC_TYPE", f"{label} must be strict tuple or list")
    if len(vec) != 8:
        raise ConformalError("INVALID_DIM", f"{label} must have length 8")
    return tuple(_require_frac(c, f"{label}_i") for c in vec)


def fmt_q(q: Fraction) -> str:
    val = _require_frac(q, "q")
    if val.denominator == 1:
        return str(val.numerator)
    return f"{val.numerator}/{val.denominator}"


def validate_json_value(obj: Any, *, max_depth: int = MAX_JSON_DEPTH) -> None:
    limit = require_int(max_depth, "max_depth", lo=1, hi=128)
    stack: list[tuple[Any, int]] = [(obj, 0)]
    while stack:
        cur, depth = stack.pop()
        if depth > limit:
            raise ConformalError("MAX_DEPTH_EXCEEDED", f"JSON nesting exceeds {limit}")
        t = type(cur)
        if t is dict:
            for k, v in cur.items():
                if type(k) is not str:
                    raise ConformalError("NON_STR_KEY", "JSON keys must be strict str")
                _require_valid_utf8_str(k, "json_key")
                stack.append((v, depth + 1))
        elif t is list:
            for item in cur:
                stack.append((item, depth + 1))
        elif t is int:
            blen = cur.bit_length()
            if blen > MAX_INT_BITS:
                raise ConformalError("INT_BIT_LIMIT", f"JSON int bit_length={blen} exceeds {MAX_INT_BITS}")
        elif t is str:
            _require_valid_utf8_str(cur, "json_str")
        elif t is bool or cur is None:
            continue
        elif t is float:
            raise ConformalError("FLOAT_FORBIDDEN", "float values are forbidden in exact JSON")
        else:
            raise ConformalError("UNSUPPORTED_JSON_TYPE", f"unsupported JSON type {t.__name__}")


def _reject_float_token(tok: str) -> Any:
    raise ConformalError("JSON_FLOAT_FORBIDDEN", f"float literal forbidden: {tok!r}")


def _reject_constant_token(tok: str) -> Any:
    raise ConformalError("JSON_CONSTANT_FORBIDDEN", f"non-RFC8259 constant forbidden: {tok!r}")


def _parse_capped_int_token(tok: str) -> int:
    digits = tok[1:] if tok.startswith("-") else tok
    if len(digits) > MAX_INT_DIGITS:
        raise ConformalError("INT_DIGIT_LIMIT", f"integer token exceeds {MAX_INT_DIGITS} digits")
    val = int(tok)
    blen = val.bit_length()
    if blen > MAX_INT_BITS:
        raise ConformalError("INT_BIT_LIMIT", f"integer token bit_length={blen} exceeds {MAX_INT_BITS}")
    return val


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in pairs:
        if k in out:
            raise ConformalError("JSON_DUPLICATE_KEY", f"duplicate JSON key: {k!r}")
        out[k] = v
    return out


def parse_strict_json_text(raw_text: str, *, require_object: bool = True, max_depth: int = MAX_JSON_DEPTH) -> Any:
    if type(require_object) is not bool:
        raise ConformalError("INVALID_BOOL_TYPE", "require_object must be strict bool")
    if type(raw_text) is not str:
        raise ConformalError("INVALID_JSON_INPUT", "JSON text must be strict str")
    try:
        encoded = raw_text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ConformalError("INVALID_UTF8", f"JSON text contains unpaired surrogate: {exc}") from exc
    if len(encoded) > MAX_JSON_BYTES:
        raise ConformalError("JSON_TOO_LARGE", f"JSON exceeds {MAX_JSON_BYTES} bytes")
    if len(raw_text.strip()) == 0:
        raise ConformalError("EMPTY_JSON_INPUT", "JSON text must be non-empty")
    try:
        obj = json.loads(
            raw_text,
            parse_float=_reject_float_token,
            parse_constant=_reject_constant_token,
            parse_int=_parse_capped_int_token,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except ConformalError:
        raise
    except (json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise ConformalError("JSON_SYNTAX_ERROR", str(exc)) from exc
    validate_json_value(obj, max_depth=max_depth)
    if require_object and type(obj) is not dict:
        raise ConformalError("JSON_ROOT_NOT_OBJECT", "top-level JSON payload must be an object")
    return obj


def canonical_json_bytes(obj: Any) -> bytes:
    validate_json_value(obj)
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def canonical_sha256(obj: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(obj)).hexdigest()


# ============================================================================
# EXACT Z/Q THETA SERIES, COXETER SHELLS & CONFORMAL QUANTIZATION ENGINE
# ============================================================================

def exact_sigma3(m: int) -> int:
    """Exact sum of cubes of positive divisors sigma_3(m) using integer isqrt."""
    n = require_int(m, "m", lo=1, hi=100_000_000)
    total = 0
    for d in range(1, isqrt(n) + 1):
        if n % d == 0:
            total += d * d * d
            q = n // d
            if q != d:
                total += q * q * q
    return total


def exact_e8_ball_budget(max_points: int) -> dict[str, int]:
    """Exact complete-shell E_8 / E_8^Had ball cardinality <= max_points via Theta_E8(q)."""
    cap = require_int(max_points, "max_points", lo=1, hi=1 << 64)
    cum = 1
    m = 0
    while True:
        nxt = 240 * exact_sigma3(m + 1)
        if cum + nxt > cap:
            return {
                "budget_points": cap,
                "max_shell_m": m,
                "max_radius_sq": 2 * m,
                "used_points": cum,
                "unused_slots": cap - cum,
                "next_shell_points": nxt,
            }
        m += 1
        cum += nxt


def exact_z8_ball_budget(max_points: int, k_limit: int = 350) -> dict[str, int]:
    """Exact complete-shell Z^8 ball cardinality <= max_points via integer polynomial Theta_Z(q)^8."""
    cap = require_int(max_points, "max_points", lo=1, hi=1 << 64)
    klim = require_int(k_limit, "k_limit", lo=1, hi=1000)
    t1 = [0] * (klim + 1)
    t1[0] = 1
    for n in range(1, isqrt(klim) + 1):
        t1[n * n] = 2

    def _poly_mul(a: list[int], b: list[int]) -> list[int]:
        out = [0] * (klim + 1)
        for i, ai in enumerate(a):
            if ai == 0:
                continue
            for j in range(klim + 1 - i):
                bj = b[j]
                if bj != 0:
                    out[i + j] += ai * bj
        return out

    t2 = _poly_mul(t1, t1)
    t4 = _poly_mul(t2, t2)
    t8 = _poly_mul(t4, t4)
    cum = 0
    for k, cnt in enumerate(t8):
        if cum + cnt > cap:
            return {
                "budget_points": cap,
                "max_radius_sq": k - 1,
                "used_points": cum,
                "unused_slots": cap - cum,
                "next_shell_points": cnt,
            }
        cum += cnt
    raise ConformalError("Z8_LIMIT_TOO_SMALL", "k_limit too small for requested max_points")


def enumerate_e8_had_shells_s2() -> dict[int, tuple[tuple[int, ...], ...]]:
    """Enumerate exact shells m=1 (240), m=2 (2160), m=3 (6720) of sqrt(2)*E_8^Had over Z."""
    shells: dict[int, list[tuple[int, ...]]] = {1: [], 2: [], 3: []}
    for target in (4, 8, 12):
        # 1. All non-decreasing nonnegative 8-tuples with sum(a_i^2) == target
        parts: list[tuple[int, ...]] = []
        p_stack: list[tuple[int, int, int, tuple[int, ...]]] = [(0, 0, target, ())]
        while p_stack:
            idx, lo_v, rem, cur = p_stack.pop()
            if idx == 8:
                if rem == 0:
                    parts.append(cur)
                continue
            for val in range(lo_v, isqrt(rem) + 1):
                p_stack.append((idx + 1, val, rem - val * val, cur + (val,)))
        # 2. Distinct permutations whose mod-2 support lies in _H8_CODEWORDS, then all nonzero sign flips
        out = shells[target // 4]
        for part in parts:
            counts: dict[int, int] = {}
            for v in part:
                counts[v] = counts.get(v, 0) + 1
            perm_stack: list[tuple[tuple[int, ...], dict[int, int]]] = [((), counts)]
            while perm_stack:
                prefix, rem_c = perm_stack.pop()
                if len(prefix) == 8:
                    if tuple(c & 1 for c in prefix) in _H8_CODEWORDS:
                        # Expand all sign assignments on nonzero positions
                        nz_idx = [i for i, c in enumerate(prefix) if c != 0]
                        for mask in range(1 << len(nz_idx)):
                            vec = list(prefix)
                            for bit, pos in enumerate(nz_idx):
                                if (mask >> bit) & 1:
                                    vec[pos] = -vec[pos]
                            out.append(tuple(vec))
                    continue
                for val, cnt in rem_c.items():
                    nxt_c = dict(rem_c)
                    if cnt == 1:
                        del nxt_c[val]
                    else:
                        nxt_c[val] = cnt - 1
                    perm_stack.append((prefix + (val,), nxt_c))
    return {m: tuple(sorted(vecs)) for m, vecs in shells.items()}


def generate_coxeter_shell1_families() -> dict[str, tuple[tuple[tuple[int, ...], ...], int]]:
    """Build exact Shell-1 integer point sets and common ||c||^2 for Z^8, D_8, E_8, E_8^Had, Block-H2."""
    # Z^8 Shell-1: 16 vectors +-e_i, ||c||^2 = 1
    z8_s1: list[tuple[int, ...]] = []
    for i in range(8):
        for s in (-1, 1):
            v = [0] * 8
            v[i] = s
            z8_s1.append(tuple(v))
    # D_8 Shell-1: 112 vectors +-e_i +- e_j, ||c||^2 = 2
    d8_s1: list[tuple[int, ...]] = []
    for i in range(8):
        for j in range(i + 1, 8):
            for s1 in (-1, 1):
                for s2 in (-1, 1):
                    v = [0] * 8
                    v[i] = s1
                    v[j] = s2
                    d8_s1.append(tuple(v))
    # E_8 Shell-1 (scaled by 2): 240 vectors, ||c||^2 = 8
    e8_s1: list[tuple[int, ...]] = [tuple(2 * c for c in v) for v in d8_s1]
    for bits in range(256):
        if bin(bits).count("1") % 2 == 0:
            e8_s1.append(tuple(-1 if (bits >> k) & 1 else 1 for k in range(8)))
    # WHT_8(E_8) scaled by sqrt(2): (H_8 @ r2) // 4, ||c||^2 = 4
    h_cur: list[list[int]] = [[1]]
    for _ in range(3):
        h_cur = [r + r for r in h_cur] + [r + [-x for x in r] for r in h_cur]
    wht_s1 = tuple(sorted({tuple(sum(h_cur[i][j] * r2[j] for j in range(8)) // 4 for i in range(8)) for r2 in e8_s1}))
    # Block-H2(E_8) scaled by sqrt(2): (BH2 @ r2) // 2, ||c||^2 = 4
    bh2_s1 = tuple(
        sorted(
            {
                tuple(
                    (r2[2 * k] + r2[2 * k + 1]) // 2 if r == 0 else (r2[2 * k] - r2[2 * k + 1]) // 2
                    for k in range(4)
                    for r in (0, 1)
                )
                for r2 in e8_s1
            }
        )
    )
    return {
        "z8_shell1": (tuple(sorted(z8_s1)), 1),
        "d8_shell1": (tuple(sorted(d8_s1)), 2),
        "e8_shell1": (tuple(sorted(e8_s1)), 8),
        "e8_had_wht8_shell1": (wht_s1, 4),
        "block_h2_shell1": (bh2_s1, 4),
    }


def _angular_sin2_fast(vec: tuple[int, ...], code: tuple[tuple[int, ...], ...], cn2: int) -> Fraction:
    v0, v1, v2, v3, v4, v5, v6, v7 = vec
    vn2 = v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3 + v4 * v4 + v5 * v5 + v6 * v6 + v7 * v7
    best_dot2 = 0
    for c0, c1, c2, c3, c4, c5, c6, c7 in code:
        dot = v0 * c0 + v1 * c1 + v2 * c2 + v3 * c3 + v4 * c4 + v5 * c5 + v6 * c6 + v7 * c7
        d2 = dot * dot
        if d2 > best_dot2:
            best_dot2 = d2
    return Fraction(vn2 * cn2 - best_dot2, vn2 * cn2)


def angular_sin2_exact(vec: tuple[int, ...], code: tuple[tuple[int, ...], ...], c_norm2: int) -> Fraction:
    """Exact fractional angular error 1 - max_{c in C} (vec.c)^2 / (||vec||^2 ||c||^2)."""
    checked_v = _require_int_vec8(vec, "vec")
    cn2 = require_int(c_norm2, "c_norm2", lo=1)
    if sum(x * x for x in checked_v) == 0:
        raise ConformalError("ZERO_VEC", "vec must be nonzero")
    if type(code) not in (tuple, list) or len(code) == 0:
        raise ConformalError("INVALID_CODE", "code must be non-empty strict tuple or list")
    checked_code: list[tuple[int, ...]] = []
    for c in code:
        cc = _require_int_vec8(c, "code_vec")
        if sum(x * x for x in cc) != cn2:
            raise ConformalError("INVALID_CODE_NORM", "code vector squared norm must equal c_norm2")
        checked_code.append(cc)
    return _angular_sin2_fast(checked_v, tuple(checked_code), cn2)


def log_radial_ladder_metrics(rho: Fraction) -> dict[str, str]:
    """Exact rational worst-case and uniform-segment relative squared error for geometric ratio rho > 1."""
    r = _require_frac(rho, "rho")
    if r <= Fraction(1, 1):
        raise ConformalError("INVALID_RHO", "rho must be strict Fraction > 1")
    max_rel = ((r - 1) / (r + 1)) ** 2
    rel_mse = (r - 1) ** 2 / (4 * (1 + r + r * r))
    return {
        "rho": fmt_q(r),
        "max_rel_sq_err": fmt_q(max_rel),
        "uniform_rel_mse": fmt_q(rel_mse),
    }


def _closest_d8_int(nums: tuple[int, ...], d: int) -> tuple[tuple[int, ...], int]:
    d2 = 2 * d
    z = [0] * 8
    diffs = [0] * 8
    parity = 0
    for i in range(8):
        ni = nums[i]
        qi = (2 * ni + d) // d2
        z[i] = qi
        diffs[i] = ni - qi * d
        parity += qi
    if parity & 1:
        best_idx = 0
        best_abs = abs(diffs[0])
        for i in range(1, 8):
            ai = abs(diffs[i])
            if ai > best_abs:
                best_abs = ai
                best_idx = i
        if diffs[best_idx] >= 0:
            z[best_idx] += 1
            diffs[best_idx] -= d
        else:
            z[best_idx] -= 1
            diffs[best_idx] += d
    return tuple(z), sum(df * df for df in diffs)


def closest_e8_had_s2_exact(y_q: tuple[Fraction, ...]) -> tuple[tuple[int, ...], Fraction]:
    """Exact CVP in sqrt(2)*E_8^Had = Block-H2(E_8) returning (q_int, unit_covolume_sq_err)."""
    checked_y = _require_frac_vec8(y_q, "y_q")
    den = 1
    for yi in checked_y:
        den = lcm(den, yi.denominator)
    y_nums = tuple(int(yi * den) for yi in checked_y)
    bh2_y = tuple(
        y_nums[2 * k] + y_nums[2 * k + 1] if r == 0 else y_nums[2 * k] - y_nums[2 * k + 1]
        for k in range(4)
        for r in (0, 1)
    )
    den2 = 2 * den
    z0, sd0 = _closest_d8_int(bh2_y, den2)
    den4 = 2 * den2
    z1, sd1 = _closest_d8_int(tuple(2 * ni - den2 for ni in bh2_y), den4)
    if 4 * sd0 <= sd1:
        p2 = tuple(2 * v for v in z0)
        sd_exact = Fraction(sd0, den2 * den2)
    else:
        p2 = tuple(2 * v + 1 for v in z1)
        sd_exact = Fraction(sd1, den4 * den4)
    q_int = tuple(
        (p2[2 * k] + p2[2 * k + 1]) // 2 if r == 0 else (p2[2 * k] - p2[2 * k + 1]) // 2
        for k in range(4)
        for r in (0, 1)
    )
    return q_int, sd_exact


def _best_conformal_shell_err_fast(
    y_q: tuple[Fraction, ...],
    shell_specs: list[tuple[Fraction, int]],
    shells: dict[int, tuple[tuple[int, ...], ...]],
) -> Fraction:
    den = 1
    for yi in y_q:
        den = lcm(den, yi.denominator)
    y0, y1, y2, y3, y4, y5, y6, y7 = (int(yi * den) for yi in y_q)
    yn2 = Fraction(y0 * y0 + y1 * y1 + y2 * y2 + y3 * y3 + y4 * y4 + y5 * y5 + y6 * y6 + y7 * y7, den * den)
    best = yn2 / 2
    max_dots: dict[int, Fraction] = {}
    for alpha, m in shell_specs:
        if m not in max_dots:
            best_int_dot = 0
            for c0, c1, c2, c3, c4, c5, c6, c7 in shells[m]:
                dot = y0 * c0 + y1 * c1 + y2 * c2 + y3 * c3 + y4 * c4 + y5 * c5 + y6 * c6 + y7 * c7
                if dot > best_int_dot:
                    best_int_dot = dot
            max_dots[m] = Fraction(best_int_dot, den)
        err = (yn2 - 2 * alpha * max_dots[m] + alpha * alpha * 4 * m) / 2
        if err < best:
            best = err
    return best


def best_conformal_shell_err_exact(
    y_q: tuple[Fraction, ...],
    shell_specs: list[tuple[Fraction, int]],
    shells: dict[int, tuple[tuple[int, ...], ...]],
) -> Fraction:
    """Exact minimum unit-covolume squared error over origin {0} + conformal (alpha, shell_m) ladder."""
    checked_y = _require_frac_vec8(y_q, "y_q")
    if type(shell_specs) not in (tuple, list):
        raise ConformalError("INVALID_SHELL_SPECS", "shell_specs must be strict tuple or list")
    if type(shells) is not dict:
        raise ConformalError("INVALID_SHELLS_TYPE", "shells must be strict dict")
    checked_specs: list[tuple[Fraction, int]] = []
    checked_shells: dict[int, tuple[tuple[int, ...], ...]] = {}
    for spec in shell_specs:
        if type(spec) not in (tuple, list) or len(spec) != 2:
            raise ConformalError("INVALID_SPEC_ITEM", "each shell spec must be a 2-tuple (alpha, m)")
        alpha = _require_frac(spec[0], "alpha")
        if alpha <= 0:
            raise ConformalError("INVALID_ALPHA", "alpha must be > 0")
        m = require_int(spec[1], "shell_m", lo=1, hi=100_000_000)
        if m not in shells:
            raise ConformalError("UNKNOWN_SHELL", f"shell m={m} missing from shells dict")
        if m not in checked_shells:
            raw_sh = shells[m]
            if type(raw_sh) not in (tuple, list) or len(raw_sh) == 0:
                raise ConformalError("INVALID_SHELL_VECS", f"shells[{m}] must be non-empty strict tuple or list")
            want_n2 = 4 * m
            vecs: list[tuple[int, ...]] = []
            for c in raw_sh:
                cc = _require_int_vec8(c, f"shell_{m}_vec")
                if sum(x * x for x in cc) != want_n2:
                    raise ConformalError("INVALID_SHELL_NORM", f"shell_{m} vector squared norm must equal {want_n2}")
                vecs.append(cc)
            checked_shells[m] = tuple(vecs)
        checked_specs.append((alpha, m))
    return _best_conformal_shell_err_fast(checked_y, checked_specs, checked_shells)


def build_certificate_payload() -> dict[str, Any]:
    """Compute the complete exact-Z/Q Track S21a certificate payload (zero floats)."""
    # Pillar 1: Exact Theta-Series Ball Budgets & Complete-Shell Slack (3.0 & 4.5 bpw)
    e8_3b = exact_e8_ball_budget(1 << 23)
    e8_45b = exact_e8_ball_budget(1 << 35)
    z8_3b = exact_z8_ball_budget(1 << 23, k_limit=60)
    z8_45b = exact_z8_ball_budget(1 << 35, k_limit=320)

    shells = enumerate_e8_had_shells_s2()
    shell_weight_spectra: dict[str, dict[str, Any]] = {}
    for m in (1, 2, 3):
        wts: dict[str, int] = {}
        for v in shells[m]:
            w = str(sum(1 for c in v if c != 0))
            wts[w] = wts.get(w, 0) + 1
        shell_weight_spectra[f"m{m}"] = {
            "norm_sq_unit_covolume": 2 * m,
            "count": len(shells[m]),
            "theta_formula_count": 240 * exact_sigma3(m),
            "hamming_weights": wts,
        }

    # Pillar 2: Exact Scale-Invariant S^7 Angular Code Comparison across all 5 Coxeter families
    fams = generate_coxeter_shell1_families()
    probe_dirs: list[tuple[str, tuple[int, ...]]] = [
        ("k1_axis", (1, 0, 0, 0, 0, 0, 0, 0)),
        ("k2_pair", (1, 1, 0, 0, 0, 0, 0, 0)),
        ("k3_triple", (1, 1, 1, 0, 0, 0, 0, 0)),
        ("k4_hamming_quad", (1, 1, 1, 1, 0, 0, 0, 0)),
        ("k4_non_hamming_quad", (1, 1, 1, 0, 1, 0, 0, 0)),
        ("k8_all_ones", (1, 1, 1, 1, 1, 1, 1, 1)),
    ]
    angular_comparison: dict[str, Any] = {}
    for fname, (code, cn2) in fams.items():
        angular_comparison[fname] = {
            "size": len(code),
            "sin2_by_direction": {dname: fmt_q(_angular_sin2_fast(vec, code, cn2)) for dname, vec in probe_dirs},
        }
    # Combined E_8^Had Shell-1 U Shell-2 (2,400) & Shell-1 U Shell-2 U Shell-3 (9,120)
    ang_s12: dict[str, str] = {}
    ang_s123: dict[str, str] = {}
    for dname, vec in probe_dirs:
        s1 = _angular_sin2_fast(vec, shells[1], 4)
        s2 = _angular_sin2_fast(vec, shells[2], 8)
        s3 = _angular_sin2_fast(vec, shells[3], 12)
        ang_s12[dname] = fmt_q(min(s1, s2))
        ang_s123[dname] = fmt_q(min(s1, s2, s3))

    # Pillar 3: Exact Conformal Log-Radial Ladder Distortion & Uniform-Shell Bit-Tax Crossover
    ladder_table = [
        log_radial_ladder_metrics(r)
        for r in (Fraction(2, 1), Fraction(3, 2), Fraction(5, 4), Fraction(9, 8), Fraction(17, 16))
    ]

    # Graded conformal-wing specs (3.0 bpw: inner 5 shells = 11,520 pts + outer 40 shells at rho=9/8 = 48,000 pts)
    inner_specs: list[tuple[Fraction, int]] = [
        (Fraction(1, 4), 1),
        (Fraction(1, 4), 2),
        (Fraction(1, 2), 1),
        (Fraction(1, 2), 2),
        (Fraction(1, 2), 3),
    ]
    outer_specs_3b: list[tuple[Fraction, int]] = [
        (Fraction(9, 2) * (Fraction(9, 8) ** j), 1) for j in range(20)
    ] + [(Fraction(3, 1) * (Fraction(9, 8) ** j), 2) for j in range(1, 21)]
    graded_specs_3b = inner_specs + outer_specs_3b
    # Pure uniform-shell conformal codebook (M=24 shells at rho=5/4 using Shell m=3: 24 * 6,720 = 161,280 pts)
    unif_specs_3b: list[tuple[Fraction, int]] = [(Fraction(1, 4) * (Fraction(5, 4) ** j), 3) for j in range(24)]

    inner_pts = 240 + 2160 + 240 + 2160 + 6720
    outer_pts = 20 * 240 + 20 * 2160
    wing_total_pts = inner_pts + outer_pts

    # Ensembles E1..E5 (all exact Fraction vectors in sqrt(2)*E_8^Had coordinates where ||x||^2 = ||y||^2 / 2)
    e1_vecs: list[tuple[str, tuple[Fraction, ...]]] = [
        ("k1_micro", (Fraction(1, 2),) + (Fraction(0),) * 7),
        ("k1_mid", (Fraction(1),) + (Fraction(0),) * 7),
        ("k2_micro", (Fraction(1, 2), Fraction(1, 2)) + (Fraction(0),) * 6),
        ("k2_mid", (Fraction(1), Fraction(1)) + (Fraction(0),) * 6),
    ]
    e1_results: list[dict[str, Any]] = []
    e1_flat_sum = Fraction(0)
    e1_grad_sum = Fraction(0)
    for name, y in e1_vecs:
        _, ef = closest_e8_had_s2_exact(y)
        eg = min(ef, _best_conformal_shell_err_fast(y, graded_specs_3b, shells))
        e1_flat_sum += ef
        e1_grad_sum += eg
        e1_results.append({"case": name, "flat_hbg_err": fmt_q(ef), "graded_wing_err": fmt_q(eg)})

    e2_bulk_vecs: list[tuple[Fraction, ...]] = [
        (Fraction(5, 2), Fraction(3, 2), Fraction(-2, 1), Fraction(1, 1), Fraction(-3, 2), Fraction(1, 2), Fraction(-1, 1), Fraction(2, 1)),
        (Fraction(3, 1), Fraction(-2, 1), Fraction(2, 1), Fraction(-1, 2), Fraction(3, 2), Fraction(-3, 2), Fraction(1, 1), Fraction(1, 2)),
        (Fraction(7, 2), Fraction(1, 1), Fraction(-1, 1), Fraction(2, 1), Fraction(-1, 1), Fraction(3, 2), Fraction(-3, 2), Fraction(1, 1)),
        (Fraction(2, 1), Fraction(2, 1), Fraction(2, 1), Fraction(-2, 1), Fraction(3, 2), Fraction(-3, 2), Fraction(1, 2), Fraction(-1, 2)),
    ]
    e2_energy = Fraction(0)
    e2_flat_sum = Fraction(0)
    e2_unif_sum = Fraction(0)
    e2_grad_sum = Fraction(0)
    for y in e2_bulk_vecs:
        e2_energy += sum(c * c for c in y) / 2
        _, ef = closest_e8_had_s2_exact(y)
        eu = _best_conformal_shell_err_fast(y, unif_specs_3b, shells)
        eg = min(ef, _best_conformal_shell_err_fast(y, graded_specs_3b, shells))
        e2_flat_sum += ef
        e2_unif_sum += eu
        e2_grad_sum += eg

    # Exact rational 16-block group scale-inflation crossover (Pure Uniform vs Flat HBG at 3.0 bpw R^2=36):
    # Under flat scale-inflation, bulk MSE scales as (alpha^2 / 36) * D_flat -> crosses D_unif at alpha^2 = 36 * D_unif / D_flat!
    crossover_outlier_norm_sq = 36 * e2_unif_sum / e2_flat_sum

    # E3: Heavy-tail outliers beyond 3.0 bpw flat ball R^2=36 (exact rational clip lower bound (||x||^2 - 36)^2 / (4*||x||^2)
    # and exact integer-radius clip error when ||x|| is integer, e.g. 2-sparse y=(16,16,0^6) => ||x||=16, exact clip=(16-6)^2=100!)
    e3_cases: list[tuple[str, tuple[Fraction, ...]]] = [
        ("k1_y12_norm2_72", (Fraction(12),) + (Fraction(0),) * 7),
        ("k1_y20_norm2_200", (Fraction(20),) + (Fraction(0),) * 7),
        ("k1_y40_norm2_800", (Fraction(40),) + (Fraction(0),) * 7),
        ("k2_y16_16_norm2_256", (Fraction(16), Fraction(16)) + (Fraction(0),) * 6),
    ]
    e3_results: list[dict[str, Any]] = []
    for name, y in e3_cases:
        xn2 = sum(c * c for c in y) / 2
        eg = _best_conformal_shell_err_fast(y, outer_specs_3b, shells)
        # Exact rational lower bound on (||x|| - 6)^2 via (||x|| - 6) = (||x||^2 - 36)/(||x|| + 6) > (||x||^2 - 36)/(2*||x||)
        clip_lb_q = (xn2 - 36) ** 2 / (4 * xn2)
        e3_results.append(
            {
                "case": name,
                "norm_sq": fmt_q(xn2),
                "flat_ball_clip_sq_err_lower_bound": fmt_q(clip_lb_q),
                "graded_wing_sq_err": fmt_q(eg),
                "graded_wing_rel_err": fmt_q(eg / xn2),
                "wing_beats_flat_clip_lb": eg < clip_lb_q,
            }
        )

    # E4: Inner Steiner-quadruple triple (1, 1, 1, 1/100, 0^4) -> ||x||^2 = 3/2 + 1/20000 in (0, 2)
    y_e4 = (Fraction(1), Fraction(1), Fraction(1), Fraction(1, 100), Fraction(0), Fraction(0), Fraction(0), Fraction(0))
    q_e4_flat, err_e4_flat = closest_e8_had_s2_exact(y_e4)
    err_e4_grad = min(err_e4_flat, _best_conformal_shell_err_fast(y_e4, graded_specs_3b, shells))

    # E5: Honest Law-2b Bulk Steiner Obstruction (3, 1, 1, 1/100, 0^4) -> ||x||^2 = 11/2 + 1/20000 > 2!
    y_e5 = (Fraction(3), Fraction(1), Fraction(1), Fraction(1, 100), Fraction(0), Fraction(0), Fraction(0), Fraction(0))
    q_e5_flat, err_e5_flat = closest_e8_had_s2_exact(y_e5)
    err_e5_grad = min(err_e5_flat, _best_conformal_shell_err_fast(y_e5, graded_specs_3b, shells))
    # Exact optimal continuous radial scaling on Shells m=1,2,3 for BOTH:
    # (a) perturbed y_e5 = (3, 1, 1, 1/100, 0^4): Shell-3 attains 35937/80000 at alpha*=367/400 on (3,1,1,1,0^4)
    #     (8,984.25x larger than Z^8 total sq err 1/20000; 0 forced bleed on Z^8);
    # (b) unperturbed base y_0 = (3, 1, 1, 0^5): Shells m=1,2,3 give 1, 3/2, 11/24.
    v_e5_100 = (300, 100, 100, 1, 0, 0, 0, 0)
    xn2_e5 = sum(c * c for c in y_e5) / 2
    e5_shell_min_proj_y_e5_q = {
        m: xn2_e5 * _angular_sin2_fast(v_e5_100, shells[m], 4 * m) for m in (1, 2, 3)
    }
    e5_shell_min_proj_y_e5 = {f"m{m}": fmt_q(e5_shell_min_proj_y_e5_q[m]) for m in (1, 2, 3)}
    v_e5_base = (3, 1, 1, 0, 0, 0, 0, 0)
    e5_shell_min_proj_y0 = {
        f"m{m}": fmt_q(Fraction(11, 2) * _angular_sin2_fast(v_e5_base, shells[m], 4 * m)) for m in (1, 2, 3)
    }
    z8_e5_total_err = Fraction(1, 20000)

    return {
        "track": "S21a",
        "audit_id": "AUDIT-121A",
        "oracle_class": "exact",
        "novelty_class": "N1",
        "theta_ball_budgets": {
            "e8_3p0_bpw": e8_3b,
            "e8_4p5_bpw": e8_45b,
            "z8_3p0_bpw": z8_3b,
            "z8_4p5_bpw": z8_45b,
            "e8_had_shells_m1_m2_m3": shell_weight_spectra,
        },
        "s7_angular_coxeter_comparison": {
            "block_h2_equals_wht8_shell1_exact": fams["block_h2_shell1"][0] == fams["e8_had_wht8_shell1"][0],
            "families": angular_comparison,
            "e8_had_shell1_union_shell2": {"size": 2400, "sin2_by_direction": ang_s12},
            "e8_had_shell1_union_shell2_union_shell3": {"size": 9120, "sin2_by_direction": ang_s123},
        },
        "conformal_radial_ladders": ladder_table,
        "graded_conformal_wing_budget": {
            "inner_cavity_shells": 5,
            "inner_cavity_points": inner_pts,
            "outer_heavy_tail_shells": 40,
            "outer_heavy_tail_points": outer_pts,
            "total_wing_points": wing_total_pts,
            "fits_3p0_unused_slots_zero_extra_bits": wing_total_pts <= e8_3b["unused_slots"],
            "fits_4p5_unused_slots_zero_extra_bits": wing_total_pts <= e8_45b["unused_slots"],
            "coupled_128d_cardinal_ratio_3p0": fmt_q(Fraction(wing_total_pts, e8_3b["used_points"])),
            "coupled_128d_cardinal_ratio_4p5": fmt_q(Fraction(wing_total_pts, e8_45b["used_points"])),
        },
        "five_ensemble_distortion_law": {
            "e1_inner_cavity": {
                "flat_hbg_total_err": fmt_q(e1_flat_sum),
                "graded_wing_total_err": fmt_q(e1_grad_sum),
                "cases": e1_results,
            },
            "e2_bulk_typical_shell": {
                "total_signal_energy": fmt_q(e2_energy),
                "flat_hbg_total_err": fmt_q(e2_flat_sum),
                "uniform_conformal_m3_24shells_total_err": fmt_q(e2_unif_sum),
                "graded_wing_total_err": fmt_q(e2_grad_sum),
                "uniform_over_flat_bulk_penalty_ratio": fmt_q(e2_unif_sum / e2_flat_sum),
                "graded_zero_bulk_loss": e2_grad_sum == e2_flat_sum,
                "uniform_vs_flat_scale_inflation_crossover_norm_sq": fmt_q(crossover_outlier_norm_sq),
            },
            "e3_heavy_tail_outliers_3p0": e3_results,
            "e4_inner_steiner_triple": {
                "flat_hbg_snapped_4th_coord_s2": q_e4_flat[3],
                "flat_hbg_sq_err": fmt_q(err_e4_flat),
                "graded_wing_sq_err": fmt_q(err_e4_grad),
                "exact_reduction_factor": fmt_q(err_e4_flat / err_e4_grad),
            },
            "e5_bulk_steiner_obstruction": {
                "flat_hbg_snapped_4th_coord_s2": q_e5_flat[3],
                "flat_hbg_sq_err": fmt_q(err_e5_flat),
                "graded_wing_sq_err": fmt_q(err_e5_grad),
                "obstruction_persists_in_bulk": err_e5_grad == err_e5_flat,
                "min_continuous_radial_shell_proj_err_y_e5": e5_shell_min_proj_y_e5,
                "min_continuous_radial_shell_proj_err_y0_unperturbed": e5_shell_min_proj_y0,
                "z8_separable_bleed_err": "0",
                "z8_total_sq_err": fmt_q(z8_e5_total_err),
                "shell3_continuous_over_z8_total_ratio": fmt_q(e5_shell_min_proj_y_e5_q[3] / z8_e5_total_err),
            },
        },
    }


def generate_certificate() -> dict[str, Any]:
    payload = build_certificate_payload()
    payload_sha = canonical_sha256(payload)
    cert_body = {
        "schema": "aletheia.track_s21a.conformal_coxeter_shells.v1",
        "payload_sha256": payload_sha,
        "payload": payload,
    }
    cert_sha = canonical_sha256(cert_body)
    return {
        "schema": cert_body["schema"],
        "payload_sha256": payload_sha,
        "certificate_sha256": cert_sha,
        "payload": payload,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Track S21a exact-Q conformal Coxeter-shell oracle")
    parser.add_argument("--standalone", action="store_true", help="Compute self-contained exact certificate")
    parser.add_argument("--json", action="store_true", help="Emit canonical JSON certificate to stdout")
    try:
        checked_argv = validate_cli_argv(argv)
    except ConformalError as exc:
        sys.stderr.write(f"FAIL {exc.code}: {exc.message}\n")
        return 1
    args = parser.parse_args(checked_argv)
    cert = generate_certificate()
    if args.json:
        sys.stdout.write(canonical_json_bytes(cert).decode("ascii") + "\n")
        return 0
    print(
        f"OK TRACK-S21A-CONFORMAL-COXETER payload_sha256={cert['payload_sha256']} "
        f"certificate_sha256={cert['certificate_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
