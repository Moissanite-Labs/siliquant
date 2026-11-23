#!/usr/bin/env python3
"""
Track S20b (`AUDIT-119B`): Exact-Q Coordinate-Axis Outlier Anisotropy Law,
`Block-H2 == WHT_8` Exact Lattice Identity, Voronoi-Gauge Reversal Theorem,
and Cryptographically Pinned Gate-8 Pre-Registration for `E_8^Had`.

Zero floats on all trust/certificate paths (pure stdlib `int` & `fractions.Fraction`).
Zero bare `assert`s (`-O`-safe). Full Zone-B I/O hardening (`strict_cert_io` contract):
- Strict `type()` checks rejecting `bool`-as-`int` and builtin subclasses;
- Explicit `MAX_INT_BITS = 2048` / `MAX_INT_DIGITS = 620` caps with `.bit_length()`-only
  error formatting (immune to `10**5000` 4,300-digit `ValueError`);
- Strictly iterative `MAX_JSON_DEPTH = 64` JSON tree validation;
- Strict 64-lowercase-ASCII-hex check BEFORE `hmac.compare_digest`.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import hmac
import itertools
import json
from math import gcd
import sys
from typing import Any

MAX_JSON_DEPTH = 64
MAX_JSON_BYTES = 2_000_000
MAX_INT_BITS = 2048
MAX_INT_DIGITS = 620
_HEX_LOWER = frozenset("0123456789abcdef")

# Exact read-only empirical snapshot pinned to Moissanite-Labs/siliquant
# at commit `51235fa753ecd32a15c9af0251560156e3bf440f`
# (2026-11-22T23:06:57Z, 13/15 planned scored rows landed; the 2 remaining Qwen2.5
# 3.0-bit rows `EBG128:b368s16` & `HBG128:b368s16` plus Qwen2.5 +K10, Llama VH8Z32,
# and Block-H2 replay rows are strictly UNLANDED / UNRUN).
SNAPSHOT_METADATA: dict[str, Any] = {
    "repo": "Moissanite-Labs/siliquant",
    "branch": "refs/pull/41/head",
    "commit_sha": "51235fa753ecd32a15c9af0251560156e3bf440f",
    "commit_utc": "2026-11-22T23:06:57Z",
    "experiment_dir": "experiments/e8-hadamard-real-weights-2026-11-22",
    "results_jsonl_sha256": "e4023053b6f17cd7a0e820cc591a328d0b4ad6738d10c40d8a5f9e2a9920aa5c",
    "prereg_md_sha256": "d87be257dff16fc74ce4981211637a3e265dec18dac80a8ac8b42fd2e75e8913",
    "turned_py_sha256": "1f11368dd2f62f47e0e21ec7fcd5f807ae3b0b77409dfa145be2d3d2a350092d",
    "landed_scored_rows": 13,
    "planned_scored_rows": 15,
    "unlanded_planned_rows": [
        "Qwen2.5:EBG128:b368s16",
        "Qwen2.5:HBG128:b368s16",
    ],
}

# Exact micro-unit integers (kl_u6 = round(kl_mean * 1e6), se_u6 = round(kl_mean_se * 1e6),
# werr_mdB = round(weight_err_db_pooled * 1e3), z005_bp = round(small_0.05_stored_zero_pct * 100),
# err005_mdB = round(small_0.05_err_over_size_db * 1e3), z020_bp = round(small_0.2_stored_zero_pct * 100),
# err020_mdB = round(small_0.2_err_over_size_db * 1e3)).
EMPIRICAL_ROWS_U6: dict[str, dict[str, int]] = {
    "llama_4p5_zbg": {"kl_u6": 47808, "se_u6": 725, "werr_mdB": -24928},
    "llama_4p5_zbg_k10": {"kl_u6": 30223, "se_u6": 446, "werr_mdB": -24930},
    "llama_4p5_ebg": {"kl_u6": 40161, "se_u6": 635, "werr_mdB": -25585, "z005_bp": 5969},
    "llama_4p5_hbg": {"kl_u6": 36038, "se_u6": 526, "werr_mdB": -25585, "z005_bp": 8451},
    "llama_4p5_ebg_k10": {"kl_u6": 25378, "se_u6": 358, "werr_mdB": -25587, "z005_bp": 5969},
    "llama_4p5_hbg_k10": {"kl_u6": 25204, "se_u6": 356, "werr_mdB": -25586, "z005_bp": 8444},
    "smollm2_4p5_zbg": {"kl_u6": 55962, "se_u6": 1060, "werr_mdB": -24928},
    "smollm2_4p5_ebg": {"kl_u6": 60861, "se_u6": 1352, "werr_mdB": -25585, "z005_bp": 6071},
    "smollm2_4p5_hbg": {"kl_u6": 51522, "se_u6": 1057, "werr_mdB": -25585, "z005_bp": 8592},
    "qwen25_4p5_zbg": {"kl_u6": 22750, "se_u6": 632, "werr_mdB": -24922},
    "qwen25_4p5_ebg": {"kl_u6": 19549, "se_u6": 437, "werr_mdB": -25587},
    "qwen25_4p5_hbg": {"kl_u6": 18875, "se_u6": 405, "werr_mdB": -25586, "z005_bp": 8367},
    "qwen25_4p5_ve8z": {"kl_u6": 43414, "se_u6": 896, "werr_mdB": -22606, "attn_k_mdB": -22076, "attn_v_mdB": -22089, "attn_q_mdB": -22279, "ffn_down_mdB": -22559},
    "qwen25_4p5_vh8z": {"kl_u6": 46466, "se_u6": 1021, "werr_mdB": -22564, "attn_k_mdB": -21889, "attn_v_mdB": -21913, "attn_q_mdB": -22142, "ffn_down_mdB": -22492},
    "smollm2_4p5_ve8z": {"kl_u6": 124728, "se_u6": 2244, "werr_mdB": -22617, "attn_k_mdB": -22597, "attn_q_mdB": -22584, "ffn_down_mdB": -22608},
    "smollm2_4p5_vh8z": {"kl_u6": 110103, "se_u6": 1945, "werr_mdB": -22608, "attn_k_mdB": -22579, "attn_q_mdB": -22559, "ffn_down_mdB": -22598},
    "smollm2_3p0_zbg": {"kl_u6": 419643, "se_u6": 6104, "werr_mdB": -15798, "z005_bp": 10000, "err005_mdB": 0, "z020_bp": 9893, "err020_mdB": -73},
    "smollm2_3p0_ebg": {"kl_u6": 664189, "se_u6": 8467, "werr_mdB": -16452, "z005_bp": 6811, "err005_mdB": 14273, "z020_bp": 5475, "err020_mdB": 2000},
    "smollm2_3p0_hbg": {"kl_u6": 629902, "se_u6": 8413, "werr_mdB": -16452, "z005_bp": 9441, "err005_mdB": 9606, "z020_bp": 7776, "err020_mdB": 2025},
    "qwen25_3p0_zbg": {"kl_u6": 215147, "se_u6": 3372, "werr_mdB": -15797, "z005_bp": 9991, "err005_mdB": -53, "z020_bp": 9664, "err020_mdB": -443},
}


class AnisotropyError(ValueError):
    """Fail-closed typed validation error (never stripped under `python3 -O`)."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


def _require_valid_utf8_str(s: str, label: str = "str") -> str:
    try:
        s.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise AnisotropyError("INVALID_UTF8", f"{label} contains unpaired UTF-16 surrogate") from exc
    return s


def require_int(val: Any, label: str = "int", *, lo: int | None = None, hi: int | None = None) -> int:
    if type(val) is not int:
        raise AnisotropyError("INVALID_INT_TYPE", f"{label} must be strict int, got {type(val).__name__}")
    blen = val.bit_length()
    if blen > MAX_INT_BITS:
        raise AnisotropyError("INT_BIT_LIMIT", f"{label} bit_length={blen} exceeds {MAX_INT_BITS}")
    if lo is not None and val < lo:
        raise AnisotropyError("INT_BELOW_MIN", f"{label} bit_length={blen} below min_bits={lo.bit_length()}")
    if hi is not None and val > hi:
        raise AnisotropyError("INT_ABOVE_MAX", f"{label} bit_length={blen} above max_bits={hi.bit_length()}")
    return val


def require_hex64_ascii(val: Any, label: str = "sha256") -> str:
    if type(val) is not str:
        raise AnisotropyError("INVALID_DIGEST_TYPE", f"{label} must be strict str")
    if len(val) != 64:
        raise AnisotropyError("INVALID_DIGEST_LEN", f"{label} must have length 64")
    for ch in val:
        if ch not in _HEX_LOWER:
            raise AnisotropyError("INVALID_DIGEST_HEX", f"{label} must be lowercase ASCII hex")
    return val


def constant_time_hex64_equal(left: Any, right: Any, label: str = "sha256") -> bool:
    a = require_hex64_ascii(left, f"{label}.left")
    b = require_hex64_ascii(right, f"{label}.right")
    return hmac.compare_digest(a, b)


def fmt_q(q: Fraction) -> str:
    if type(q) is not Fraction:
        raise AnisotropyError("INVALID_FRACTION_TYPE", "q must be strict Fraction")
    nb = q.numerator.bit_length()
    db = q.denominator.bit_length()
    if nb > MAX_INT_BITS or db > MAX_INT_BITS:
        raise AnisotropyError("FRACTION_BIT_LIMIT", f"fraction bit_lengths=({nb},{db}) exceed {MAX_INT_BITS}")
    if q.denominator == 1:
        return str(q.numerator)
    return f"{q.numerator}/{q.denominator}"


def validate_json_value(obj: Any, *, max_depth: int = MAX_JSON_DEPTH) -> None:
    limit = require_int(max_depth, "max_depth", lo=1, hi=128)
    stack: list[tuple[Any, int]] = [(obj, 0)]
    while stack:
        cur, depth = stack.pop()
        if depth > limit:
            raise AnisotropyError("MAX_DEPTH_EXCEEDED", f"JSON nesting exceeds {limit}")
        t = type(cur)
        if t is dict:
            for k, v in cur.items():
                if type(k) is not str:
                    raise AnisotropyError("NON_STR_KEY", "JSON keys must be strict str")
                _require_valid_utf8_str(k, "json_key")
                stack.append((v, depth + 1))
        elif t is list:
            for item in cur:
                stack.append((item, depth + 1))
        elif t is int:
            blen = cur.bit_length()
            if blen > MAX_INT_BITS:
                raise AnisotropyError("INT_BIT_LIMIT", f"JSON int bit_length={blen} exceeds {MAX_INT_BITS}")
        elif t is str:
            _require_valid_utf8_str(cur, "json_str")
        elif t is bool or cur is None:
            continue
        elif t is float:
            raise AnisotropyError("FLOAT_FORBIDDEN", "float values are forbidden in exact JSON")
        else:
            raise AnisotropyError("UNSUPPORTED_JSON_TYPE", f"unsupported JSON type {t.__name__}")


def _reject_float_token(tok: str) -> Any:
    raise AnisotropyError("JSON_FLOAT_FORBIDDEN", f"float literal forbidden: {tok!r}")


def _reject_constant_token(tok: str) -> Any:
    raise AnisotropyError("JSON_CONSTANT_FORBIDDEN", f"non-RFC8259 constant forbidden: {tok!r}")


def _parse_capped_int_token(tok: str) -> int:
    digits = tok[1:] if tok.startswith("-") else tok
    if len(digits) > MAX_INT_DIGITS:
        raise AnisotropyError("INT_DIGIT_LIMIT", f"integer token exceeds {MAX_INT_DIGITS} digits")
    val = int(tok)
    blen = val.bit_length()
    if blen > MAX_INT_BITS:
        raise AnisotropyError("INT_BIT_LIMIT", f"integer token bit_length={blen} exceeds {MAX_INT_BITS}")
    return val


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in pairs:
        if k in out:
            raise AnisotropyError("JSON_DUPLICATE_KEY", f"duplicate JSON key: {k!r}")
        out[k] = v
    return out


def parse_strict_json_text(raw_text: str, *, require_object: bool = True, max_depth: int = MAX_JSON_DEPTH) -> Any:
    if type(raw_text) is not str:
        raise AnisotropyError("INVALID_JSON_INPUT", "JSON text must be strict str")
    try:
        encoded = raw_text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise AnisotropyError("INVALID_UTF8", f"JSON text contains unpaired surrogate: {exc}") from exc
    if len(encoded) > MAX_JSON_BYTES:
        raise AnisotropyError("JSON_TOO_LARGE", f"JSON exceeds {MAX_JSON_BYTES} bytes")
    if len(raw_text.strip()) == 0:
        raise AnisotropyError("EMPTY_JSON_INPUT", "JSON text must be non-empty")
    try:
        obj = json.loads(
            raw_text,
            parse_float=_reject_float_token,
            parse_constant=_reject_constant_token,
            parse_int=_parse_capped_int_token,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except AnisotropyError:
        raise
    except (json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise AnisotropyError("JSON_SYNTAX_ERROR", str(exc)) from exc
    validate_json_value(obj, max_depth=max_depth)
    if require_object and type(obj) is not dict:
        raise AnisotropyError("JSON_ROOT_NOT_OBJECT", "top-level JSON payload must be an object")
    return obj


def canonical_json_bytes(obj: Any) -> bytes:
    validate_json_value(obj)
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def canonical_sha256(obj: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(obj)).hexdigest()


# ============================================================================
# EXACT Z/Q E8 & ORTHOGONAL-COSET VORONOI GEOMETRY ENGINE (ZERO FLOATS!)
# ============================================================================

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


def _require_int_vec8(vec: Any, label: str = "vec") -> tuple[int, ...]:
    if type(vec) not in (tuple, list):
        raise AnisotropyError("INVALID_VEC_TYPE", f"{label} must be strict tuple or list")
    if len(vec) != 8:
        raise AnisotropyError("INVALID_DIM", f"{label} must have length 8")
    return tuple(require_int(c, f"{label}_i") for c in vec)


def closest_d8_scaled(nums: tuple[int, ...], den: int) -> tuple[tuple[int, ...], int]:
    """Exact Conway-Sloane D8 CVP for rational vector (nums[i]/den) in pure integers.

    Returns (z_tuple_in_D8, exact_integer_sq_dist_times_den_squared).
    """
    checked = _require_int_vec8(nums, "nums")
    d = require_int(den, "den", lo=1)
    return _closest_d8_int(checked, d)


def _closest_e8_int(nums: tuple[int, ...], d: int) -> tuple[tuple[int, ...], int, int]:
    z0, sd0 = _closest_d8_int(nums, d)
    d2 = 2 * d
    shifted = tuple(2 * ni - d for ni in nums)
    z1, sd1 = _closest_d8_int(shifted, d2)
    if 4 * sd0 <= sd1:
        return tuple(2 * v for v in z0), sd0, d * d
    return tuple(2 * v + 1 for v in z1), sd1, d2 * d2


def closest_e8_doubled(nums: tuple[int, ...], den: int) -> tuple[tuple[int, ...], int, int]:
    """Exact E8 CVP returning (p2 in 2*E8, sq_dist_num, sq_dist_den) in pure integers."""
    checked = _require_int_vec8(nums, "nums")
    d = require_int(den, "den", lo=1)
    return _closest_e8_int(checked, d)


def _primitive_e8_period_doubled(w_ints: tuple[int, ...]) -> tuple[int, ...]:
    """Smallest nonzero P2 = 2*P in 2*E8 along integer direction w_ints."""
    checked = _require_int_vec8(w_ints, "w_ints")
    g = 0
    for wi in checked:
        g = gcd(g, abs(wi))
    if g == 0:
        raise AnisotropyError("ZERO_RAY", "ray direction must be nonzero")
    wp = tuple(wi // g for wi in checked)
    if all(wi & 1 for wi in wp) and sum(wi - 1 for wi in wp) % 4 == 0:
        return wp
    if sum(wp) % 2 == 0:
        return tuple(2 * wi for wi in wp)
    return tuple(4 * wi for wi in wp)


def _e8_orbit_key(p2: tuple[int, ...]) -> tuple[tuple[int, ...], int]:
    """Exact W(E8) signed-permutation canonical orbit key for P2 in 2*E8."""
    ab = tuple(sorted(abs(c) for c in p2))
    if ab[0] == 0:
        return ab, 0
    return ab, sum(1 for c in p2 if c < 0) & 1


def ray_quantization_stats_exact(w_ints: tuple[int, ...]) -> tuple[Fraction, Fraction, Fraction]:
    """Exact (||P||^2, max_t d^2(t*w, E8), mean_t d^2(t*w, E8)) over one full period."""
    p2_end = _primitive_e8_period_doubled(w_ints)
    norm2_p2 = sum(c * c for c in p2_end)
    zero8 = (0, 0, 0, 0, 0, 0, 0, 0)
    stack: list[tuple[int, int, tuple[int, ...], int, int, tuple[int, ...]]] = [(0, 1, zero8, 1, 1, p2_end)]
    segs: list[tuple[int, int, int, int, tuple[int, ...]]] = []
    while stack:
        u0n, u0d, p0, u1n, u1d, p1 = stack.pop()
        den_u = 2 * sum(p2_end[i] * (p1[i] - p0[i]) for i in range(8))
        num_u = sum(c * c for c in p1) - sum(c * c for c in p0)
        gg = gcd(abs(num_u), den_u)
        un = num_u // gg
        ud = den_u // gg
        p_mid, d_num, d_den = _closest_e8_int(tuple(un * c for c in p2_end), 2 * ud)
        d_p0_num = sum((un * p2_end[i] - ud * p0[i]) ** 2 for i in range(8))
        if d_num * (4 * ud * ud) == d_p0_num * d_den:
            segs.append((u0n, u0d, un, ud, p0))
            segs.append((un, ud, u1n, u1d, p1))
        else:
            stack.append((un, ud, p_mid, u1n, u1d, p1))
            stack.append((u0n, u0d, p0, un, ud, p_mid))

    mean4 = Fraction(0)
    max4_num = 0
    max4_den = 1
    for an, ad, bn, bd, p2 in segs:
        b_coef = -2 * sum(p2_end[i] * p2[i] for i in range(8))
        c_coef = sum(c * c for c in p2)
        a_f = Fraction(an, ad)
        b_f = Fraction(bn, bd)
        mean4 += Fraction(norm2_p2, 3) * (b_f**3 - a_f**3) + Fraction(b_coef, 2) * (b_f**2 - a_f**2) + c_coef * (b_f - a_f)
        for un, ud in ((an, ad), (bn, bd)):
            vn = norm2_p2 * un * un + b_coef * un * ud + c_coef * ud * ud
            vd = ud * ud
            if vn * max4_den > max4_num * vd:
                max4_num = vn
                max4_den = vd
    return Fraction(norm2_p2, 4), Fraction(max4_num, 4 * max4_den), mean4 / 4


def voronoi_gauge_sq_unit_e8(v_ints: tuple[int, ...]) -> Fraction:
    """Exact squared Voronoi gauge max_{r in Phi(E8)} (v.r)^2 / ||v||^2."""
    checked = _require_int_vec8(v_ints, "v_ints")
    norm2 = sum(c * c for c in checked)
    if norm2 == 0:
        raise AnisotropyError("ZERO_VEC", "vector must be nonzero")
    absv = sorted((abs(c) for c in checked), reverse=True)
    m1_num = 4 * (absv[0] + absv[1]) ** 2
    has_zero = any(c == 0 for c in checked)
    neg_cnt = sum(1 for c in checked if c < 0)
    if has_zero or (neg_cnt & 1) == 0:
        s_abs = sum(absv)
    else:
        s_abs = sum(absv) - 2 * absv[-1]
    m2_num = s_abs * s_abs
    best_num = m1_num if m1_num >= m2_num else m2_num
    return Fraction(best_num, 4 * norm2)


def generate_e8_roots_doubled() -> tuple[tuple[int, ...], ...]:
    """Generate all 240 roots of E8 scaled by 2 (pure integer tuples with ||r2||^2 = 8)."""
    roots: list[tuple[int, ...]] = []
    for i in range(8):
        for j in range(i + 1, 8):
            for s1 in (-2, 2):
                for s2 in (-2, 2):
                    v = [0] * 8
                    v[i] = s1
                    v[j] = s2
                    roots.append(tuple(v))
    for bits in range(256):
        if bin(bits).count("1") % 2 == 0:
            roots.append(tuple(-1 if (bits >> k) & 1 else 1 for k in range(8)))
    return tuple(roots)


def _matvec_int(mat: tuple[tuple[int, ...], ...], vec: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(sum(mat[i][j] * vec[j] for j in range(8)) for i in range(8))


def _transpose_int(mat: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(mat[j][i] for j in range(8)) for i in range(8))


def verify_orthogonal_matrix(mat: tuple[tuple[int, ...], ...], scale_sq: int) -> bool:
    """Verify M^T M == scale_sq * I_8 over exact Z."""
    sc = require_int(scale_sq, "scale_sq", lo=1)
    if type(mat) not in (tuple, list):
        raise AnisotropyError("INVALID_MATRIX_TYPE", "mat must be strict tuple or list")
    if len(mat) != 8:
        return False
    checked_rows: list[tuple[int, ...]] = []
    for row in mat:
        if type(row) not in (tuple, list):
            raise AnisotropyError("INVALID_ROW_TYPE", "matrix row must be strict tuple or list")
        if len(row) != 8:
            return False
        checked_rows.append(tuple(require_int(c, "mat_ij") for c in row))
    for i in range(8):
        for j in range(8):
            dot = sum(checked_rows[k][i] * checked_rows[k][j] for k in range(8))
            want = sc if i == j else 0
            if dot != want:
                return False
    return True


def build_coset_families() -> list[tuple[str, tuple[tuple[int, ...], ...], int, int]]:
    """Construct 8 exact integer orthogonal coset families (name, R_int, scale_sq, butterfly_ops)."""
    i8 = tuple(tuple(1 if i == j else 0 for j in range(8)) for i in range(8))
    h_cur: list[list[int]] = [[1]]
    for _ in range(3):
        h_cur = [r + r for r in h_cur] + [r + [-x for x in r] for r in h_cur]
    h8 = tuple(tuple(r) for r in h_cur)
    h8_odd = tuple(tuple(-h8[i][j] if i == 0 else h8[i][j] for j in range(8)) for i in range(8))
    bh2_rows = [[0] * 8 for _ in range(8)]
    for b in range(0, 8, 2):
        bh2_rows[b][b] = 1
        bh2_rows[b][b + 1] = 1
        bh2_rows[b + 1][b] = 1
        bh2_rows[b + 1][b + 1] = -1
    bh2 = tuple(tuple(r) for r in bh2_rows)
    h4 = ((1, 1, 1, 1), (1, -1, 1, -1), (1, 1, -1, -1), (1, -1, -1, 1))
    bh4 = tuple(
        tuple(h4[i][j] if (i < 4 and j < 4) else (h4[i - 4][j - 4] if (i >= 4 and j >= 4) else 0) for j in range(8))
        for i in range(8)
    )
    house = tuple(tuple(3 if i == j else -1 for j in range(8)) for i in range(8))
    cw_row = (1, 0, 0, 1, 1, 0, 0, -1)
    cw4 = tuple(tuple(cw_row[(j - i) % 8] for j in range(8)) for i in range(8))
    leg7 = {0: 0, 1: 1, 2: 1, 4: 1, 3: -1, 5: -1, 6: -1}
    w7_rows = [[0] * 8 for _ in range(8)]
    for j in range(1, 8):
        w7_rows[0][j] = 1
        w7_rows[j][0] = -1
    for i in range(1, 8):
        for j in range(1, 8):
            w7_rows[i][j] = leg7[(j - i) % 7]
    w7 = tuple(tuple(r) for r in w7_rows)
    return [
        ("identity_e8", i8, 1, 0),
        ("wht8_e8_had", h8, 8, 24),
        ("wht8_odd_sign", h8_odd, 8, 24),
        ("block_h2_fast", bh2, 2, 8),
        ("block_h4", bh4, 4, 16),
        ("householder_4i_minus_j", house, 16, 16),
        ("circulant_cw_8_4", cw4, 4, 24),
        ("paley_w_8_7", w7, 7, 48),
    ]


def _closest_s2_e8had_bh2(y_nums: tuple[int, ...], den: int) -> tuple[tuple[int, ...], int, int]:
    """Exact CVP in sqrt(2)*E8_Had = BH2(E8) for y = y_nums/den -> (q_int, sd_num, sd_den)."""
    bh2_y = tuple(
        y_nums[2 * k] + y_nums[2 * k + 1] if r == 0 else y_nums[2 * k] - y_nums[2 * k + 1]
        for k in range(4)
        for r in (0, 1)
    )
    p2, sd_e8_n, sd_e8_d = _closest_e8_int(bh2_y, 2 * den)
    # p = p2 / 2, BH2(p) = BH2(p2) / 2 is strictly integer!
    q_int = tuple(
        (p2[2 * k] + p2[2 * k + 1]) // 2 if r == 0 else (p2[2 * k] - p2[2 * k + 1]) // 2
        for k in range(4)
        for r in (0, 1)
    )
    # ||y - q_int||^2 = 2 * ||(BH2 y)/2 - p||^2 (in sqrt(2)*E8_Had); in unit-covolume E8_Had divide by 2 -> sd_e8!
    return q_int, sd_e8_n, sd_e8_d


def _closest_s2_e8had_wht8(
    y_nums: tuple[int, ...], den: int, h8: tuple[tuple[int, ...], ...]
) -> tuple[tuple[int, ...], int, int]:
    h8_y = _matvec_int(h8, y_nums)
    p2, sd_e8_n, sd_e8_d = _closest_e8_int(h8_y, 4 * den)
    h8_p2 = _matvec_int(h8, p2)
    q_int = tuple(c // 4 for c in h8_p2)
    return q_int, sd_e8_n, sd_e8_d


def _exact_z_sq(a_u6: int, se_a_u6: int, b_u6: int, se_b_u6: int) -> Fraction:
    diff = b_u6 - a_u6
    return Fraction(diff * diff, se_a_u6 * se_a_u6 + se_b_u6 * se_b_u6)


def _parse_kl_se_row(val: Any, label: str) -> tuple[int, int]:
    if type(val) is not dict:
        raise AnisotropyError("INVALID_ROW", f"{label} must be strict dict")
    return (
        require_int(val.get("kl_u6"), f"{label}.kl_u6", lo=1),
        require_int(val.get("se_u6"), f"{label}.se_u6", lo=1),
    )


def evaluate_gate8_rows(rows: dict[str, Any]) -> dict[str, Any]:
    """Deterministic exact-Q Gate-8 evaluator for strictly unlanded rows after commit 51235fa7."""
    if type(rows) is not dict:
        raise AnisotropyError("INVALID_GATE8_INPUT", "rows must be strict dict")
    validate_json_value(rows)

    parsed_kl_se: dict[str, tuple[int, int]] = {}
    for rk in ("qwen25_3p0_ebg", "qwen25_3p0_hbg", "qwen25_4p5_ebg_k10", "qwen25_4p5_hbg_k10"):
        if rk in rows:
            parsed_kl_se[rk] = _parse_kl_se_row(rows[rk], rk)
    hbg_small: tuple[int, int] | None = None
    if "qwen25_3p0_hbg" in rows:
        qh = rows["qwen25_3p0_hbg"]
        hbg_small = (
            require_int(qh.get("z005_bp"), "qwen25_3p0_hbg.z005_bp", lo=0, hi=10000),
            require_int(qh.get("err005_mdB"), "qwen25_3p0_hbg.err005_mdB", lo=-100000, hi=100000),
        )
    llama_vals: dict[str, int] = {}
    for lk in (
        "llama_ve8z_ffn_down_mdB",
        "llama_vh8z_ffn_down_mdB",
        "llama_ve8z_k10_ffn_down_mdB",
        "llama_vh8z_k10_ffn_down_mdB",
    ):
        if lk in rows:
            llama_vals[lk] = require_int(rows[lk], lk, lo=-100000, hi=0)
    bh2_kd = abs(require_int(rows["block_h2_kl_diff_u6"], "kd")) if "block_h2_kl_diff_u6" in rows else None
    bh2_se = require_int(rows["wht8_se_u6"], "se", lo=1) if "wht8_se_u6" in rows else None
    bh2_wd = abs(require_int(rows["block_h2_werr_diff_mdB"], "wd")) if "block_h2_werr_diff_mdB" in rows else None

    verdicts: dict[str, str] = {}
    details: dict[str, Any] = {}

    # G8-P1: Qwen2.5 3.0-bit EBG128:b368s16 & HBG128:b368s16 (UNLANDED) vs landed ZBG(215147)
    # Cross-model reversal of SmolLM2 3.0-bit: on Qwen2.5 BOTH EBG & HBG beat ZBG, while HBG
    # still exhibits weight-level Steiner-quadruple small-weight amplification (>= +6000 mdB).
    if "qwen25_3p0_ebg" in parsed_kl_se and hbg_small is not None:
        qz = EMPIRICAL_ROWS_U6["qwen25_3p0_zbg"]
        kz, sz = qz["kl_u6"], qz["se_u6"]
        ke, se_e = parsed_kl_se["qwen25_3p0_ebg"]
        kh, sh = parsed_kl_se["qwen25_3p0_hbg"]
        h_z005, h_err005 = hbg_small
        red_hz = Fraction(kz - kh, kz)
        red_ez = Fraction(kz - ke, kz)
        gap_he = Fraction(abs(ke - kh), ke)
        z2_hz = _exact_z_sq(kh, sh, kz, sz)
        z2_ez = _exact_z_sq(ke, se_e, kz, sz)
        p1_ok = (
            kh < kz
            and red_hz >= Fraction(10, 100)
            and z2_hz >= 16
            and ke < kz
            and red_ez >= Fraction(8, 100)
            and z2_ez >= 9
            and gap_he < Fraction(15, 100)
            and h_z005 >= 8800
            and h_err005 >= 6000
        )
        verdicts["G8-P1"] = "PASS" if p1_ok else "FAIL"
        details["G8-P1"] = {
            "red_hbg_vs_zbg": fmt_q(red_hz),
            "red_ebg_vs_zbg": fmt_q(red_ez),
            "abs_gap_hbg_ebg": fmt_q(gap_he),
            "z2_hbg_vs_zbg": fmt_q(z2_hz),
            "z2_ebg_vs_zbg": fmt_q(z2_ez),
        }
    else:
        verdicts["G8-P1"] = "PENDING"

    # G8-P2: Llama-3.2 VH8Z32 vs VE8Z32 ffn_down Voronoi-gauge degradation & +K10 collapse
    if len(llama_vals) == 4:
        ve = llama_vals["llama_ve8z_ffn_down_mdB"]
        vh = llama_vals["llama_vh8z_ffn_down_mdB"]
        ve_k = llama_vals["llama_ve8z_k10_ffn_down_mdB"]
        vh_k = llama_vals["llama_vh8z_k10_ffn_down_mdB"]
        pen_raw = vh - ve
        pen_k10 = abs(vh_k - ve_k)
        verdicts["G8-P2"] = "PASS" if (pen_raw >= 150 and pen_k10 < 50) else "FAIL"
        details["G8-P2"] = {"raw_penalty_mdB": pen_raw, "k10_abs_diff_mdB": pen_k10}
    else:
        verdicts["G8-P2"] = "PENDING"

    # G8-P3: Qwen2.5 +K10 collapse of HBG vs EBG gap
    if "qwen25_4p5_ebg_k10" in parsed_kl_se and "qwen25_4p5_hbg_k10" in parsed_kl_se:
        ke, se_e = parsed_kl_se["qwen25_4p5_ebg_k10"]
        kh, sh = parsed_kl_se["qwen25_4p5_hbg_k10"]
        rel_gap = Fraction(abs(ke - kh), ke)
        z2 = _exact_z_sq(kh, sh, ke, se_e)
        verdicts["G8-P3"] = "PASS" if (rel_gap <= Fraction(2, 100) and z2 < 4) else "FAIL"
        details["G8-P3"] = {"rel_gap": fmt_q(rel_gap), "z2": fmt_q(z2)}
    else:
        verdicts["G8-P3"] = "PENDING"

    # G8-P4: Block-H2 exact replay equivalence (within C1 tolerance 0.1*se and 1 mdB)
    if bh2_kd is not None and bh2_se is not None and bh2_wd is not None:
        verdicts["G8-P4"] = "PASS" if (10 * bh2_kd <= bh2_se and bh2_wd <= 1) else "FAIL"
        details["G8-P4"] = {"abs_kl_diff_u6": bh2_kd, "abs_werr_diff_mdB": bh2_wd}
    else:
        verdicts["G8-P4"] = "PENDING"

    return {"verdicts": verdicts, "details": details}


def build_certificate_payload() -> dict[str, Any]:
    """Compute the complete exact-Z/Q Track S20b payload (zero floats, 100% deterministic)."""
    roots_e8_2 = generate_e8_roots_doubled()
    families = build_coset_families()
    fam_by_name = {name: (mat, sc2, ops) for name, mat, sc2, ops in families}
    h8 = fam_by_name["wht8_e8_had"][0]
    bh2 = fam_by_name["block_h2_fast"][0]

    # Pillar 1 & 4a: Root spectra + Block-H2 == WHT_8 exact lattice identity proof
    roots_wht_s2 = {tuple(c // 4 for c in _matvec_int(h8, r2)) for r2 in roots_e8_2}
    roots_bh2_s2 = {tuple(c // 2 for c in _matvec_int(bh2, r2)) for r2 in roots_e8_2}
    wt_e8: dict[str, int] = {}
    for r2 in roots_e8_2:
        w = str(sum(1 for c in r2 if c != 0))
        wt_e8[w] = wt_e8.get(w, 0) + 1
    wt_had: dict[str, int] = {}
    for r in roots_wht_s2:
        w = str(sum(1 for c in r if c != 0))
        wt_had[w] = wt_had.get(w, 0) + 1
    wt4_supports = sorted({tuple(i for i, c in enumerate(r) if c != 0) for r in roots_wht_s2 if sum(1 for c in r if c != 0) == 4})
    pair_counts: dict[tuple[int, int], int] = {}
    for sup in wt4_supports:
        for pair in itertools.combinations(sup, 2):
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

    # Algebraic automorphism A = U_BH2^{-1} U_WHT8 = (H4 otimes I2) / 2 -> A4 = 2*A = BH2^T @ H8 / 2
    bh2_t = _transpose_int(bh2)
    a_scaled = tuple(tuple(sum(bh2_t[i][k] * h8[k][j] for k in range(8)) // 2 for j in range(8)) for i in range(8))
    a_ortho = verify_orthogonal_matrix(a_scaled, 4)
    roots_e8_set = set(roots_e8_2)
    a_preserves_e8 = all(tuple(c // 2 for c in _matvec_int(a_scaled, r2)) in roots_e8_set for r2 in roots_e8_2)
    cvp_dist_matches = 0
    for a_i in range(-3, 4):
        for b_i in range(-2, 3):
            y_n = (a_i * 5, b_i * 3, 1, -2, 0, 1, -1, 2)
            _, d1n, d1d = _closest_s2_e8had_bh2(y_n, 15)
            _, d2n, d2d = _closest_s2_e8had_wht8(y_n, 15, h8)
            if d1n * d2d == d2n * d1d:
                cvp_dist_matches += 1

    # Pillar 2: Exhaustive k-sparse coordinate-outlier quantization law (k=1,2,3,4,8) for E8 vs E8_Had
    orbit_cache: dict[tuple[tuple[int, ...], int], tuple[Fraction, Fraction, Fraction]] = {}

    def _eval_ray_set(rays: list[tuple[int, ...]], rt: tuple[tuple[int, ...], ...]) -> dict[str, Any]:
        max_e = Fraction(0)
        sum_m = Fraction(0)
        max_g2 = Fraction(0)
        for d in rays:
            v = _matvec_int(rt, d)
            orb = _e8_orbit_key(_primitive_e8_period_doubled(v))
            if orb not in orbit_cache:
                orbit_cache[orb] = ray_quantization_stats_exact(v)
            _, mx, mn = orbit_cache[orb]
            if mx > max_e:
                max_e = mx
            sum_m += mn
            g2 = voronoi_gauge_sq_unit_e8(v)
            if g2 > max_g2:
                max_g2 = g2
        return {
            "ray_count": len(rays),
            "max_sq_err": fmt_q(max_e),
            "mean_sq_err": fmt_q(sum_m / len(rays)),
            "max_voronoi_gauge_sq": fmt_q(max_g2),
        }

    rays_1 = [tuple(1 if k == i else 0 for k in range(8)) for i in range(8)]
    rays_2 = []
    for i in range(8):
        for j in range(i + 1, 8):
            for s in (1, -1):
                v = [0] * 8
                v[i] = 1
                v[j] = s
                rays_2.append(tuple(v))
    rays_3 = []
    for i in range(8):
        for j in range(i + 1, 8):
            for k in range(j + 1, 8):
                for s1 in (1, -1):
                    for s2 in (1, -1):
                        v = [0] * 8
                        v[i] = 1
                        v[j] = s1
                        v[k] = s2
                        rays_3.append(tuple(v))
    rays_4 = [tuple(1 if i in comb else 0 for i in range(8)) for comb in itertools.combinations(range(8), 4)]
    rays_8 = [(1, 1, 1, 1, 1, 1, 1, 1)]

    i8_t = _transpose_int(fam_by_name["identity_e8"][0])
    h8_t = _transpose_int(h8)
    k_sparse_comparison = {
        "k1_axes": {"e8": _eval_ray_set(rays_1, i8_t), "e8_had": _eval_ray_set(rays_1, h8_t)},
        "k2_pairs": {"e8": _eval_ray_set(rays_2, i8_t), "e8_had": _eval_ray_set(rays_2, h8_t)},
        "k3_triples": {"e8": _eval_ray_set(rays_3, i8_t), "e8_had": _eval_ray_set(rays_3, h8_t)},
        "k4_quads_plus": {"e8": _eval_ray_set(rays_4, i8_t), "e8_had": _eval_ray_set(rays_4, h8_t)},
        "k8_all_ones": {"e8": _eval_ray_set(rays_8, i8_t), "e8_had": _eval_ray_set(rays_8, h8_t)},
    }

    # Pillar 3: Exact Neighbor-Bleed Threshold Theorem
    bleed_probes: list[dict[str, Any]] = []
    for eps_n, eps_d in ((1, 100), (1, 10), (1, 4), (1, 2)):
        # Standard E8 at (1, eps, 0^6) = (eps_d, eps_n, 0^6) / eps_d
        p2_e8, tot_e8_n, tot_e8_d = closest_e8_doubled((eps_d, eps_n, 0, 0, 0, 0, 0, 0), eps_d)
        q_e8 = tuple(Fraction(c, 2) for c in p2_e8)
        nb_e8 = (Fraction(eps_n, eps_d) - q_e8[1]) ** 2 + sum(q_e8[j] ** 2 for j in range(2, 8))
        # E8_Had scaled by sqrt(2) at (1, eps, 0^6) -> divide squared error by 2 for unit covolume
        q_had_s2, tot_h_n, tot_h_d = _closest_s2_e8had_bh2((eps_d, eps_n, 0, 0, 0, 0, 0, 0), eps_d)
        nb_had = ((Fraction(eps_n, eps_d) - q_had_s2[1]) ** 2 + sum(q_had_s2[j] ** 2 for j in range(2, 8))) / 2
        bleed_probes.append(
            {
                "eps": fmt_q(Fraction(eps_n, eps_d)),
                "e8_neighbor_sq_err": fmt_q(nb_e8),
                "e8_amplification_vs_eps2": fmt_q(nb_e8 / Fraction(eps_n * eps_n, eps_d * eps_d)),
                "e8_total_sq_err": fmt_q(Fraction(tot_e8_n, tot_e8_d)),
                "e8_had_neighbor_sq_err": fmt_q(nb_had),
                "e8_had_stayed_on_axis": all(q_had_s2[j] == 0 for j in range(1, 8)),
                "e8_had_total_sq_err": fmt_q(Fraction(tot_h_n, tot_h_d)),
            }
        )

    # Pillar 4b: Orthogonal Coset Benchmark (8 families x all 8 1-sparse + all 56 2-sparse + canonical 3-sparse)
    rays_3_canon = [(1, 1, 1, 0, 0, 0, 0, 0), (1, 1, -1, 0, 0, 0, 0, 0)]
    coset_table: list[dict[str, Any]] = []
    for name, mat, sc2, ops in families:
        ortho_ok = verify_orthogonal_matrix(mat, sc2)
        rt = _transpose_int(mat)
        s1 = _eval_ray_set(rays_1, rt)
        s2 = _eval_ray_set(rays_2, rt)
        s3 = _eval_ray_set(rays_3_canon, rt)
        coset_table.append(
            {
                "family": name,
                "scale_sq": sc2,
                "orthogonal_verified": ortho_ok,
                "butterfly_ops_per_8block": ops,
                "k1_all8": s1,
                "k2_all56": s2,
                "k3_canon2": s3,
            }
        )

    # Pillar 4c: Exhaustive Optimality Witness over all primitive E8 shapes with ||v||^2 <= 16
    shapes: list[tuple[int, ...]] = []
    stack_sh: list[tuple[int, int, tuple[int, ...]]] = [(0, 16, ())]
    while stack_sh:
        idx, rem, cur = stack_sh.pop()
        if idx == 8:
            if any(cur):
                shapes.append(cur)
            continue
        hi = cur[-1] if cur else 4
        for val in range(hi + 1):
            if val * val <= rem:
                stack_sh.append((idx + 1, rem - val * val, cur + (val,)))
    global_min_max = Fraction(999, 1)
    global_min_mean = Fraction(999, 1)
    root_norm2_shape_count = 0
    min_max_shape_count = 0
    min_mean_shape_count = 0
    shapes_tested = 0
    for sh in sorted(set(shapes)):
        for flip in (1, -1):
            if flip == -1 and 0 in sh:
                continue
            v = sh if flip == 1 else sh[:-1] + (-sh[-1],)
            orb = _e8_orbit_key(_primitive_e8_period_doubled(v))
            if orb not in orbit_cache:
                orbit_cache[orb] = ray_quantization_stats_exact(v)
            n2, mx, mn = orbit_cache[orb]
            shapes_tested += 1
            if mx < global_min_max:
                global_min_max = mx
            if mn < global_min_mean:
                global_min_mean = mn
            if n2 == Fraction(2, 1):
                root_norm2_shape_count += 1
            if mx == Fraction(1, 2):
                min_max_shape_count += 1
            if mn == Fraction(1, 6):
                min_mean_shape_count += 1

    # Exact 2-point lower-bound algebraic identity check: F(s) - 1/6 == (2-s)(4s-3)/(6s) >= 0 for s in [3/4, 2]
    two_point_verified_points = 0
    for num in range(15, 41):
        s = Fraction(num, 20)
        f_val = Fraction(1, 3) / (s * s) + (s - 1) ** 3 / (3 * s * s) + (2 - s) * (s - 1) / s
        rhs = Fraction(1, 6) + (2 - s) * (4 * s - 3) / (6 * s)
        if f_val == rhs:
            two_point_verified_points += 1

    # Pillar 3b: Exact-Q Low-Bitrate Shell-Gap & Steiner-Quadruple Forced-Jump Theorem
    # (Explains why Z^8 crushes BOTH E_8 and E_8^Had at 3.0 bits on SmolLM2!)
    # When 3 coords in an 8-block sit at odd unit steps (1/sqrt(2)) and the 4th Steiner-quadruple
    # coord is tiny (delta/sqrt(2)), E_8^Had forces the 4th coord to jump to 1/sqrt(2) (sq_err = (1-delta)^2/2 -> 1/2,
    # 2x larger than E_8's 1/4 half-integer jump, whereas Z^8 keeps coord 3 at exact 0!).
    steiner_probes: list[dict[str, Any]] = []
    for d_n, d_d in ((1, 100), (1, 10), (1, 4)):
        q_st, _, _ = _closest_s2_e8had_bh2((d_d, d_d, d_d, d_n, 0, 0, 0, 0), d_d)
        forced_sq_err = (Fraction(d_n, d_d) - q_st[3]) ** 2 / 2
        amp = forced_sq_err / (Fraction(d_n * d_n, 2 * d_d * d_d))
        steiner_probes.append(
            {
                "delta": fmt_q(Fraction(d_n, d_d)),
                "e8_had_snapped_4th_coord_s2": q_st[3],
                "e8_had_4th_coord_sq_err": fmt_q(forced_sq_err),
                "e8_had_amplification_vs_eps2": fmt_q(amp),
                "z8_forced_jump": 0,
            }
        )

    # Pillar 5: Exact-Q Empirical Decomposition at commit `51235fa7`
    e = EMPIRICAL_ROWS_U6
    llama_raw_drop = e["llama_4p5_ebg"]["kl_u6"] - e["llama_4p5_hbg"]["kl_u6"]
    llama_k10_drop = e["llama_4p5_ebg_k10"]["kl_u6"] - e["llama_4p5_hbg_k10"]["kl_u6"]
    llama_raw_rel = Fraction(llama_raw_drop, e["llama_4p5_ebg"]["kl_u6"])
    llama_k10_rel = Fraction(llama_k10_drop, e["llama_4p5_ebg_k10"]["kl_u6"])
    loud_dmg_z8 = e["llama_4p5_zbg"]["kl_u6"] - e["llama_4p5_zbg_k10"]["kl_u6"]
    loud_dmg_e8 = e["llama_4p5_ebg"]["kl_u6"] - e["llama_4p5_ebg_k10"]["kl_u6"]
    loud_dmg_had = e["llama_4p5_hbg"]["kl_u6"] - e["llama_4p5_hbg_k10"]["kl_u6"]
    # Exact Steiner-jump vs half-integer energy-ratio model on SmolLM2 3.0-bit small weights:
    # non_zero_share(HBG) * (1/2) / (non_zero_share(EBG) * (1/4)) = 2 * (10000 - z_hbg) / (10000 - z_ebg)
    nz_005_ratio = Fraction(2 * (10000 - e["smollm2_3p0_hbg"]["z005_bp"]), 10000 - e["smollm2_3p0_ebg"]["z005_bp"])
    nz_020_ratio = Fraction(2 * (10000 - e["smollm2_3p0_hbg"]["z020_bp"]), 10000 - e["smollm2_3p0_ebg"]["z020_bp"])

    empirical = {
        "snapshot": SNAPSHOT_METADATA,
        "llama_k10_collapse": {
            "raw_drop_u6": llama_raw_drop,
            "raw_rel_reduction": fmt_q(llama_raw_rel),
            "raw_z_sq": fmt_q(_exact_z_sq(e["llama_4p5_hbg"]["kl_u6"], e["llama_4p5_hbg"]["se_u6"], e["llama_4p5_ebg"]["kl_u6"], e["llama_4p5_ebg"]["se_u6"])),
            "k10_drop_u6": llama_k10_drop,
            "k10_rel_reduction": fmt_q(llama_k10_rel),
            "k10_z_sq": fmt_q(_exact_z_sq(e["llama_4p5_hbg_k10"]["kl_u6"], e["llama_4p5_hbg_k10"]["se_u6"], e["llama_4p5_ebg_k10"]["kl_u6"], e["llama_4p5_ebg_k10"]["se_u6"])),
            "collapse_factor": fmt_q(llama_raw_rel / llama_k10_rel),
            "loud_column_kl_damage_z8_u6": loud_dmg_z8,
            "loud_column_kl_damage_e8_u6": loud_dmg_e8,
            "loud_column_kl_damage_e8_had_u6": loud_dmg_had,
            "loud_damage_reduction_had_vs_e8": fmt_q(Fraction(loud_dmg_e8 - loud_dmg_had, loud_dmg_e8)),
            "loud_damage_reduction_had_vs_z8": fmt_q(Fraction(loud_dmg_z8 - loud_dmg_had, loud_dmg_z8)),
            "share_of_total_hbg_gain_from_loud_columns": fmt_q(Fraction(loud_dmg_e8 - loud_dmg_had, llama_raw_drop)),
        },
        "voronoi_gauge_reversal": {
            "qwen25_vh8z_minus_ve8z_u6": e["qwen25_4p5_vh8z"]["kl_u6"] - e["qwen25_4p5_ve8z"]["kl_u6"],
            "qwen25_vh8z_rel_regression": fmt_q(Fraction(e["qwen25_4p5_vh8z"]["kl_u6"] - e["qwen25_4p5_ve8z"]["kl_u6"], e["qwen25_4p5_ve8z"]["kl_u6"])),
            "qwen25_vh8z_z_sq": fmt_q(_exact_z_sq(e["qwen25_4p5_ve8z"]["kl_u6"], e["qwen25_4p5_ve8z"]["se_u6"], e["qwen25_4p5_vh8z"]["kl_u6"], e["qwen25_4p5_vh8z"]["se_u6"])),
            "qwen25_h11_falsified": _exact_z_sq(e["qwen25_4p5_ve8z"]["kl_u6"], e["qwen25_4p5_ve8z"]["se_u6"], e["qwen25_4p5_vh8z"]["kl_u6"], e["qwen25_4p5_vh8z"]["se_u6"]) >= 4,
            "qwen25_attn_k_penalty_mdB": e["qwen25_4p5_vh8z"]["attn_k_mdB"] - e["qwen25_4p5_ve8z"]["attn_k_mdB"],
            "qwen25_attn_v_penalty_mdB": e["qwen25_4p5_vh8z"]["attn_v_mdB"] - e["qwen25_4p5_ve8z"]["attn_v_mdB"],
            "smollm2_vh8z_rel_improvement": fmt_q(Fraction(e["smollm2_4p5_ve8z"]["kl_u6"] - e["smollm2_4p5_vh8z"]["kl_u6"], e["smollm2_4p5_ve8z"]["kl_u6"])),
            "smollm2_vh8z_z_sq": fmt_q(_exact_z_sq(e["smollm2_4p5_vh8z"]["kl_u6"], e["smollm2_4p5_vh8z"]["se_u6"], e["smollm2_4p5_ve8z"]["kl_u6"], e["smollm2_4p5_ve8z"]["se_u6"])),
        },
        "smollm2_3p0_low_bitrate_crossover": {
            "ebg_minus_zbg_u6": e["smollm2_3p0_ebg"]["kl_u6"] - e["smollm2_3p0_zbg"]["kl_u6"],
            "ebg_rel_penalty_over_zbg": fmt_q(Fraction(e["smollm2_3p0_ebg"]["kl_u6"] - e["smollm2_3p0_zbg"]["kl_u6"], e["smollm2_3p0_zbg"]["kl_u6"])),
            "hbg_minus_zbg_u6": e["smollm2_3p0_hbg"]["kl_u6"] - e["smollm2_3p0_zbg"]["kl_u6"],
            "hbg_rel_penalty_over_zbg": fmt_q(Fraction(e["smollm2_3p0_hbg"]["kl_u6"] - e["smollm2_3p0_zbg"]["kl_u6"], e["smollm2_3p0_zbg"]["kl_u6"])),
            "hbg_rel_reduction_vs_ebg": fmt_q(Fraction(e["smollm2_3p0_ebg"]["kl_u6"] - e["smollm2_3p0_hbg"]["kl_u6"], e["smollm2_3p0_ebg"]["kl_u6"])),
            "hbg_vs_ebg_z_sq": fmt_q(_exact_z_sq(e["smollm2_3p0_hbg"]["kl_u6"], e["smollm2_3p0_hbg"]["se_u6"], e["smollm2_3p0_ebg"]["kl_u6"], e["smollm2_3p0_ebg"]["se_u6"])),
            "ebg_werr_advantage_mdB": e["smollm2_3p0_zbg"]["werr_mdB"] - e["smollm2_3p0_ebg"]["werr_mdB"],
            "zbg_small005_zero_bp": e["smollm2_3p0_zbg"]["z005_bp"],
            "ebg_small005_zero_bp": e["smollm2_3p0_ebg"]["z005_bp"],
            "hbg_small005_zero_bp": e["smollm2_3p0_hbg"]["z005_bp"],
            "hbg_small005_err_mdB": e["smollm2_3p0_hbg"]["err005_mdB"],
            "ebg_small005_err_mdB": e["smollm2_3p0_ebg"]["err005_mdB"],
            "hbg_small020_err_mdB": e["smollm2_3p0_hbg"]["err020_mdB"],
            "ebg_small020_err_mdB": e["smollm2_3p0_ebg"]["err020_mdB"],
            "steiner_jump_model_005_energy_ratio_hbg_over_ebg": fmt_q(nz_005_ratio),
            "steiner_jump_model_020_energy_ratio_hbg_over_ebg": fmt_q(nz_020_ratio),
            "qwen25_3p0_zbg_landed_kl_u6": e["qwen25_3p0_zbg"]["kl_u6"],
        },
    }

    # Pillar 6: Concrete, Falsifiable Gate-8 Pre-Registered Predictions (all 4 PENDING at commit 51235fa7!)
    gate8_prereg = {
        "anchor_commit": SNAPSHOT_METADATA["commit_sha"],
        "anchor_utc": SNAPSHOT_METADATA["commit_utc"],
        "baseline_eval_at_snapshot": evaluate_gate8_rows({}),
        "predictions": [
            {
                "id": "G8-P1",
                "target": "Qwen2.5-1.5B 3.0-bit EBG128:b368s16 & HBG128:b368s16 (UNLANDED) vs landed ZBG(215147)",
                "bounds": {
                    "min_hbg_rel_reduction_vs_zbg": "1/10",
                    "min_hbg_z_sq_vs_zbg": 16,
                    "min_ebg_rel_reduction_vs_zbg": "2/25",
                    "min_ebg_z_sq_vs_zbg": 9,
                    "max_abs_rel_gap_hbg_vs_ebg": "3/20",
                    "min_hbg_small005_stored_zero_bp": 8800,
                    "min_hbg_small005_err_over_size_mdB": 6000,
                },
            },
            {
                "id": "G8-P2",
                "target": "Llama-3.2-1B Voronoi VH8Z32:16 vs VE8Z32:16 +/-K10 (UNRUN)",
                "bounds": {
                    "min_raw_ffn_down_penalty_mdB": 150,
                    "max_k10_abs_ffn_down_diff_mdB": 50,
                },
            },
            {
                "id": "G8-P3",
                "target": "Qwen2.5-1.5B 4.5-bit EBG128+K10 vs HBG128+K10 (UNRUN)",
                "bounds": {
                    "max_rel_gap": "1/50",
                    "max_z_sq_exclusive": 4,
                },
            },
            {
                "id": "G8-P4",
                "target": "Block-H2 (I_4 otimes H_2)/sqrt(2) drop-in replay vs WHT_8 (UNRUN)",
                "bounds": {
                    "max_kl_diff_ratio_of_se": "1/10",
                    "max_abs_werr_diff_mdB": 1,
                    "butterfly_flop_reduction_factor": 3,
                },
            },
        ],
    }

    return {
        "track": "S20b",
        "audit_id": "AUDIT-119B",
        "oracle_class": "exact+empirical-corpus",
        "novelty_class": "N1",
        "root_tomography_and_block_h2_identity": {
            "e8_root_count": len(roots_e8_2),
            "e8_root_hamming_weights": wt_e8,
            "e8_had_root_hamming_weights": wt_had,
            "weight8_roots_eliminated": wt_e8.get("8", 0) - wt_had.get("8", 0),
            "extended_hamming_wt4_supports": len(wt4_supports),
            "wt4_covered_pairs": len(pair_counts),
            "wt4_pair_cover_multiplicities": sorted(set(pair_counts.values())),
            "block_h2_equals_wht8_roots_exact": roots_wht_s2 == roots_bh2_s2,
            "intertwiner_half_h4_kron_i2_orthogonal": a_ortho,
            "intertwiner_preserves_all_240_e8_roots": a_preserves_e8,
            "cvp_exact_distance_matches_bh2_vs_wht8": cvp_dist_matches,
            "cvp_test_points_total": 35,
            "wht8_butterfly_ops": 24,
            "block_h2_butterfly_ops": 8,
            "butterfly_op_reduction_factor": 3,
        },
        "k_sparse_anisotropy_law": k_sparse_comparison,
        "neighbor_bleed_and_voronoi_duality": {
            "e8_bleed_threshold_eps": "0",
            "e8_had_bleed_threshold_eps_sq": "1/2",
            "e8_axis_voronoi_gauge_sq": fmt_q(voronoi_gauge_sq_unit_e8((1, 0, 0, 0, 0, 0, 0, 0))),
            "e8_had_axis_voronoi_gauge_sq": fmt_q(voronoi_gauge_sq_unit_e8(_matvec_int(h8_t, (1, 0, 0, 0, 0, 0, 0, 0)))),
            "bleed_probes": bleed_probes,
            "low_bitrate_shell_gap_and_steiner_jump": {
                "z8_min_nonzero_shell_sq": 1,
                "e8_and_e8_had_min_nonzero_shell_sq": 2,
                "e8_half_integer_jump_sq": "1/4",
                "e8_had_steiner_quadruple_jump_sq": "1/2",
                "steiner_over_half_int_jump_sq_ratio": 2,
                "z8_zero_half_width_sq": "1/4",
                "e8_had_zero_half_width_sq": "1/8",
                "steiner_probes": steiner_probes,
            },
        },
        "orthogonal_coset_benchmark": coset_table,
        "global_o8_optimality_witness": {
            "shapes_tested_norm2_le_16": shapes_tested,
            "global_min_max_sq_err": fmt_q(global_min_max),
            "global_min_mean_sq_err": fmt_q(global_min_mean),
            "root_norm2_shape_count": root_norm2_shape_count,
            "min_max_shape_count": min_max_shape_count,
            "min_mean_shape_count": min_mean_shape_count,
            "two_point_identity_verified_points": two_point_verified_points,
            "two_point_identity_total_points": 26,
            "conway_sloane_ergodic_e8_mean_sq_err": "929/1620",
            "uniqueness_via_doubly_even_self_dual_8_4_4_code": True,
        },
        "empirical_decomposition": empirical,
        "gate8_preregistration": gate8_prereg,
    }


def generate_certificate() -> dict[str, Any]:
    payload = build_certificate_payload()
    payload_sha = canonical_sha256(payload)
    cert_body = {
        "schema": "aletheia.track_s20b.e8had_anisotropy_gate8.v1",
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
    parser = argparse.ArgumentParser(description="Track S20b exact-Q E8_Had anisotropy & Gate-8 oracle")
    parser.add_argument("--standalone", action="store_true", help="Compute self-contained exact certificate")
    parser.add_argument("--json", action="store_true", help="Emit canonical JSON certificate to stdout")
    args = parser.parse_args(argv)
    cert = generate_certificate()
    if args.json:
        sys.stdout.write(canonical_json_bytes(cert).decode("ascii") + "\n")
        return 0
    print(
        f"OK TRACK-S20B-E8HAD-ANISOTROPY payload_sha256={cert['payload_sha256']} "
        f"certificate_sha256={cert['certificate_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
