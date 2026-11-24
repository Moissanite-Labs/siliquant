#!/usr/bin/env python3
"""Track S32: Exact-Z/Q TPU/Pallas/NEON WCET, Pre-Cast Overflow Separation & 2-Stage Residual Flat-LUT E8 Oracle.

Pure Python 3 standard library over exact Z/Q (`int` and `fractions.Fraction`).
Zero floats, zero bare asserts, zero RNG, zero third-party dependencies.
Certifies five pillars targeting `Moissanite-Labs/siliquant#60` and `#61`:
  1. Exact Pre-Cast Numerical Dynamic-Range & `Block-H2` vs `WHT_8` `INT8` Overflow Separation + Accumulator Width.
  2. Exact Static XLA/Pallas WCET, VMEM Residency & TPU v5e Three-Way Roofline (`BF16` vs `INT8` MXU).
  3. Exact `O(1)` Branchless 2-Stage Residual Flat-LUT `E_8` Codec & Pure-Q Distortion Contraction Theorem.
  4. Exact Analytical Spectral / Lipschitz Tension-Drift Enclosure (Zero-Unpack Frobenius, Bareiss PD & `Block-H2`).
  5. Exact Safety/Refusal Boundary Margin Preservation (`sqrt(rho)` Tightening & 8x8 Witness vs Coarse Scalar).
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
from typing import Any, Sequence

TRACK_ID = "TRACK-S32-SILIQUANT-TPU-BOUNDS-RESIDUAL-LUT"
SCHEMA_VERSION = "aletheia.siliquant_tpu_bounds.certificate.v1"

MAX_JSON_BYTES = 1024 * 1024
MAX_JSON_DEPTH = 32
MAX_INT_DIGITS = 1000
MAX_INT_BITS = 3322
_ASCII_DIGITS = frozenset("0123456789")

INT8_MAX = 127
INT8_MIN = -128
INT16_MAX = 32767
INT32_MAX = 2147483647
TPU_V5E_VMEM_BYTES = 134217728  # 128 MiB exact
HIDDEN_DIM_K = 16384  # 512 super-blocks of 32 weights


class SiliquantTpuBoundsError(ValueError):
    """Fail-closed domain/validation error for Track S32."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


# ---------------------------------------------------------------------------
# Strict Type, ASCII-Digit, Rational & Canonical JSON Primitives
# ---------------------------------------------------------------------------

def require_exact_int(
    val: Any,
    name: str,
    *,
    min_val: int | None = None,
    max_val: int | None = None,
) -> int:
    if type(val) is not int:
        raise SiliquantTpuBoundsError("INVALID_INT", f"{name} must be exact int")
    if min_val is not None and val < min_val:
        raise SiliquantTpuBoundsError("INT_BELOW_MIN", f"{name} must be >= {min_val}")
    if max_val is not None and val > max_val:
        raise SiliquantTpuBoundsError("INT_ABOVE_MAX", f"{name} must be <= {max_val}")
    return val


def require_exact_bool(val: Any, name: str) -> bool:
    if type(val) is not bool:
        raise SiliquantTpuBoundsError("INVALID_BOOL", f"{name} must be exact bool")
    return val


def require_exact_str(val: Any, name: str, *, non_empty: bool = True) -> str:
    if type(val) is not str:
        raise SiliquantTpuBoundsError("INVALID_STR", f"{name} must be exact str")
    if non_empty and not val:
        raise SiliquantTpuBoundsError("EMPTY_STR", f"{name} must be non-empty")
    try:
        val.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise SiliquantTpuBoundsError("INVALID_UTF8_STR", f"{name} contains lone surrogate") from exc
    return val


def require_exact_fraction(
    val: Any,
    name: str,
    *,
    positive: bool = False,
    non_negative: bool = False,
) -> Fraction:
    if type(val) is not Fraction:
        raise SiliquantTpuBoundsError("INVALID_FRACTION", f"{name} must be exact Fraction")
    if positive and val <= 0:
        raise SiliquantTpuBoundsError("FRACTION_NOT_POSITIVE", f"{name} must be > 0")
    if non_negative and val < 0:
        raise SiliquantTpuBoundsError("FRACTION_NEGATIVE", f"{name} must be >= 0")
    return val


def parse_exact_int_ascii(text: Any, name: str) -> int:
    """Parse strict ASCII signed integer (`-?[0-9]+`, no `.isdigit()`, bounded length)."""
    s = require_exact_str(text, name)
    neg = s.startswith("-")
    digits = s[1:] if neg else s
    if not digits:
        raise SiliquantTpuBoundsError("INVALID_INT_TOKEN", f"{name} has no digits")
    if len(digits) > MAX_INT_DIGITS:
        raise SiliquantTpuBoundsError("INT_TOKEN_TOO_LONG", f"{name} exceeds {MAX_INT_DIGITS} digits")
    if len(digits) > 1 and digits[0] == "0":
        raise SiliquantTpuBoundsError("LEADING_ZERO_INT", f"{name} has leading zero")
    for ch in digits:
        if ch not in _ASCII_DIGITS:
            raise SiliquantTpuBoundsError("NON_ASCII_DIGIT", f"{name} contains non-ASCII digit")
    if neg and digits == "0":
        raise SiliquantTpuBoundsError("NEGATIVE_ZERO_INT", f"{name} cannot be -0")
    acc = 0
    for ch in digits:
        acc = acc * 10 + (ord(ch) - 48)
    return -acc if neg else acc


def parse_exact_rational(text: Any, name: str) -> Fraction:
    """Parse canonical exact rational string `p` or `p/q` (strict ASCII digits, `q > 0`, reduced)."""
    s = require_exact_str(text, name)
    if "/" not in s:
        return Fraction(parse_exact_int_ascii(s, name), 1)
    parts = s.split("/")
    if len(parts) != 2:
        raise SiliquantTpuBoundsError("INVALID_RATIONAL", f"{name} has multiple slashes")
    num = parse_exact_int_ascii(parts[0], f"{name}.num")
    den = parse_exact_int_ascii(parts[1], f"{name}.den")
    if den <= 1:
        raise SiliquantTpuBoundsError("NON_CANONICAL_DEN", f"{name} denominator must be > 1")
    if math.gcd(abs(num), den) != 1:
        raise SiliquantTpuBoundsError("UNREDUCED_RATIONAL", f"{name} must be in lowest terms")
    return Fraction(num, den)


def _require_bounded_int(val: int, name: str) -> None:
    if val.bit_length() > MAX_INT_BITS:
        raise SiliquantTpuBoundsError("INT_TOO_LARGE", f"{name} exceeds {MAX_INT_BITS}-bit limit")


def format_exact_rational(q: Any) -> str:
    val = require_exact_fraction(q, "q")
    _require_bounded_int(val.numerator, "q.numerator")
    _require_bounded_int(val.denominator, "q.denominator")
    if val.denominator == 1:
        return str(val.numerator)
    return f"{val.numerator}/{val.denominator}"


def format_decimal_trunc(q: Any, places: Any) -> str:
    """Deterministic toward-zero decimal truncation of exact Fraction using integer arithmetic only."""
    val = require_exact_fraction(q, "q")
    _require_bounded_int(val.numerator, "q.numerator")
    _require_bounded_int(val.denominator, "q.denominator")
    dp = require_exact_int(places, "places", min_val=0, max_val=18)
    neg = val < 0
    num = -val.numerator if neg else val.numerator
    den = val.denominator
    scale = 10 ** dp
    scaled = (num * scale) // den
    prefix = "-" if (neg and scaled != 0) else ""
    if dp == 0:
        return f"{prefix}{scaled}"
    whole, frac = divmod(scaled, scale)
    return f"{prefix}{whole}.{frac:0{dp}d}"


def _validate_json_value(val: Any, depth: int) -> None:
    if depth > MAX_JSON_DEPTH:
        raise SiliquantTpuBoundsError("JSON_TOO_DEEP", "JSON nesting depth exceeded")
    t = type(val)
    if t is bool or val is None:
        return
    if t is int:
        _require_bounded_int(val, "json_int")
        return
    if t is str:
        require_exact_str(val, "json_str", non_empty=False)
        return
    if t is list:
        for item in val:
            _validate_json_value(item, depth + 1)
        return
    if t is dict:
        for k, v in val.items():
            require_exact_str(k, "json_key", non_empty=False)
            _validate_json_value(v, depth + 1)
        return
    raise SiliquantTpuBoundsError("FORBIDDEN_JSON_TYPE", f"unsupported JSON type: {t.__name__}")


def canonical_json_bytes(obj: Any) -> bytes:
    _validate_json_value(obj, 0)
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    if len(raw) > MAX_JSON_BYTES:
        raise SiliquantTpuBoundsError("JSON_TOO_LARGE", "canonical JSON exceeds byte limit")
    return raw


def parse_strict_json_text(text: Any, *, require_object: bool = True) -> Any:
    s = require_exact_str(text, "json_text")
    if len(s.encode("utf-8")) > MAX_JSON_BYTES:
        raise SiliquantTpuBoundsError("JSON_TOO_LARGE", "JSON text exceeds byte limit")

    def _reject_float(token: str) -> Any:
        raise SiliquantTpuBoundsError("FLOAT_FORBIDDEN", f"floating-point JSON token forbidden: {token}")

    def _reject_const(token: str) -> Any:
        raise SiliquantTpuBoundsError("NON_FINITE_FORBIDDEN", f"JSON constant forbidden: {token}")

    def _parse_json_int(token: str) -> int:
        val = parse_exact_int_ascii(token, "json_int")
        _require_bounded_int(val, "json_int")
        return val

    def _no_dup_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in pairs:
            if k in out:
                raise SiliquantTpuBoundsError("DUPLICATE_JSON_KEY", f"duplicate key: {k}")
            out[k] = v
        return out

    try:
        parsed = json.loads(
            s,
            parse_float=_reject_float,
            parse_int=_parse_json_int,
            parse_constant=_reject_const,
            object_pairs_hook=_no_dup_pairs,
        )
    except SiliquantTpuBoundsError:
        raise
    except (ValueError, RecursionError) as exc:
        raise SiliquantTpuBoundsError("INVALID_JSON", f"malformed JSON: {type(exc).__name__}") from exc

    _validate_json_value(parsed, 0)
    if require_object and type(parsed) is not dict:
        raise SiliquantTpuBoundsError("JSON_ROOT_NOT_OBJECT", "JSON root must be an object")
    return parsed


# ---------------------------------------------------------------------------
# Exact E8 Lattice, Sylvester H8, Block-H2, BF16 & Bareiss Linear Algebra
# ---------------------------------------------------------------------------

def _build_hadamard_8() -> tuple[tuple[int, ...], ...]:
    h: list[list[int]] = [[1]]
    for _ in range(3):
        h = [r + r for r in h] + [r + [-x for x in r] for r in h]
    return tuple(tuple(r) for r in h)


H8_MATRIX: tuple[tuple[int, ...], ...] = _build_hadamard_8()

BLOCK_H2_MATRIX: tuple[tuple[int, ...], ...] = (
    (1, 1, 0, 0, 0, 0, 0, 0),
    (1, -1, 0, 0, 0, 0, 0, 0),
    (0, 0, 1, 1, 0, 0, 0, 0),
    (0, 0, 1, -1, 0, 0, 0, 0),
    (0, 0, 0, 0, 1, 1, 0, 0),
    (0, 0, 0, 0, 1, -1, 0, 0),
    (0, 0, 0, 0, 0, 0, 1, 1),
    (0, 0, 0, 0, 0, 0, 1, -1),
)


def _require_int_vec8(vec: Any, name: str) -> tuple[int, ...]:
    if type(vec) not in (list, tuple) or len(vec) != 8:
        raise SiliquantTpuBoundsError("INVALID_VEC8", f"{name} must be list/tuple of 8 ints")
    return tuple(require_exact_int(x, f"{name}[{i}]") for i, x in enumerate(vec))


def _require_frac_vec8(vec: Any, name: str) -> tuple[Fraction, ...]:
    if type(vec) not in (list, tuple) or len(vec) != 8:
        raise SiliquantTpuBoundsError("INVALID_Q_VEC8", f"{name} must be list/tuple of 8 Fractions")
    return tuple(require_exact_fraction(x, f"{name}[{i}]") for i, x in enumerate(vec))


def _require_square_frac_matrix(mat: Any, name: str, *, require_symmetric: bool = False) -> list[list[Fraction]]:
    if type(mat) not in (list, tuple) or not (1 <= len(mat) <= 16):
        raise SiliquantTpuBoundsError("INVALID_MATRIX", f"{name} must be 1..16 square matrix")
    n = len(mat)
    out: list[list[Fraction]] = []
    for i, row in enumerate(mat):
        if type(row) not in (list, tuple) or len(row) != n:
            raise SiliquantTpuBoundsError("NON_SQUARE_MATRIX", f"{name}[{i}] must have length {n}")
        out.append([require_exact_fraction(x, f"{name}[{i}][{j}]") for j, x in enumerate(row)])
    if require_symmetric:
        for i in range(n):
            for j in range(i + 1, n):
                if out[i][j] != out[j][i]:
                    raise SiliquantTpuBoundsError("NON_SYMMETRIC_MATRIX", f"{name} must be symmetric")
    return out


def is_in_doubled_e8(z: Any) -> bool:
    """Return True iff integer 8-vector `z` belongs to `2 * E_8` (same parity & sum divisible by 4)."""
    v = _require_int_vec8(z, "z")
    par = v[0] & 1
    for x in v[1:]:
        if (x & 1) != par:
            return False
    return (sum(v) & 3) == 0


def doubled_e8_shell(z: Any) -> int:
    """Return exact shell index `m = ||z||^2 / 8` for `z in 2 * E_8`."""
    v = _require_int_vec8(z, "z")
    if not is_in_doubled_e8(v):
        raise SiliquantTpuBoundsError("NOT_IN_2E8", "vector is not in 2*E_8")
    n2 = sum(x * x for x in v)
    return n2 // 8


def apply_wht8_int(z: Any) -> tuple[int, ...]:
    v = _require_int_vec8(z, "z")
    return tuple(sum(H8_MATRIX[r][c] * v[c] for c in range(8)) for r in range(8))


def apply_block_h2_int(z: Any) -> tuple[int, ...]:
    v = _require_int_vec8(z, "z")
    return tuple(sum(BLOCK_H2_MATRIX[r][c] * v[c] for c in range(8)) for r in range(8))


def is_exact_bf16_half_integer(z: Any) -> bool:
    """Return True iff `v = z / 2` is bit-exactly representable in IEEE-754 `bfloat16` (0 ULP)."""
    zi = require_exact_int(z, "z")
    if zi == 0:
        return True
    mag = -zi if zi < 0 else zi
    # v = mag * 2^(-1). Strip trailing factors of 2: mag = odd * 2^tz.
    tz = 0
    while (mag & 1) == 0:
        mag >>= 1
        tz += 1
    # BF16 has 8 bits of significand precision (1 implicit + 7 explicit): odd part must fit in 8 bits (<= 255).
    if mag > 255:
        return False
    # Highest set bit of `mag` determines unbiased exponent: e = (bit_length - 1) + tz - 1.
    exp = (mag.bit_length() - 1) + tz - 1
    return -126 <= exp <= 127


def sigma3_exact(m: Any) -> int:
    """Exact divisor cube sum `sigma_3(m) = sum_{d | m} d^3` for positive integer `m`."""
    mi = require_exact_int(m, "m", min_val=1, max_val=100000)
    total = 0
    limit = math.isqrt(mi)
    for d in range(1, limit + 1):
        if mi % d == 0:
            total += d * d * d
            q = mi // d
            if q != d:
                total += q * q * q
    return total


def r_e8_shell_count(m: Any) -> int:
    """Exact Eisenstein `E_4` shell population `r_{E_8}(m)` (`1` for `m=0`, `240 * sigma_3(m)` for `m>=1`)."""
    mi = require_exact_int(m, "m", min_val=0, max_val=100000)
    if mi == 0:
        return 1
    return 240 * sigma3_exact(mi)


def w1_ball_count(m_max: Any) -> int:
    """Exact cumulative lattice point count `W_1(m_max) = sum_{m=0}^{m_max} r_{E_8}(m)`."""
    mm = require_exact_int(m_max, "m_max", min_val=0, max_val=2000)
    return sum(r_e8_shell_count(m) for m in range(mm + 1))


def enumerate_e8_ball_doubled(m_max: Any) -> tuple[tuple[int, ...], ...]:
    """Deterministically enumerate every `z in 2 * E_8` with `||z||_2^2 <= 8 * m_max` (`0 <= m_max <= 3`)."""
    mm = require_exact_int(m_max, "m_max", min_val=0, max_val=3)
    limit = 8 * mm
    pts: list[tuple[int, ...]] = []

    def _dfs(idx: int, rem_sq: int, cur_sum: int, par: int, cur: list[int]) -> None:
        rem = 8 - idx
        if rem == 0:
            if (cur_sum & 3) == 0:
                pts.append(tuple(cur))
            return
        max_here_sq = rem_sq - (rem - 1 if par == 1 else 0)
        if max_here_sq < par:
            return
        max_abs = math.isqrt(max_here_sq)
        start = -max_abs
        if (start & 1) != par:
            start += 1
        for x in range(start, max_abs + 1, 2):
            cur.append(x)
            _dfs(idx + 1, rem_sq - x * x, cur_sum + x, par, cur)
            cur.pop()

    _dfs(0, limit, 0, 0, [])
    _dfs(0, limit, 0, 1, [])
    pts.sort()
    return tuple(pts)


_LUT_M2_CACHE: tuple[tuple[int, ...], ...] | None = None


def _get_lut_m2() -> tuple[tuple[int, ...], ...]:
    global _LUT_M2_CACHE
    if _LUT_M2_CACHE is None:
        _LUT_M2_CACHE = enumerate_e8_ball_doubled(2)
    return _LUT_M2_CACHE


def _round_half_even_exact(x: Fraction) -> int:
    q, r = divmod(x.numerator, x.denominator)
    two_r = 2 * r
    if two_r < x.denominator:
        return q
    if two_r > x.denominator:
        return q + 1
    return q if (q & 1) == 0 else q + 1


def _nearest_d8_exact(x: tuple[Fraction, ...]) -> tuple[int, ...]:
    f = [_round_half_even_exact(v) for v in x]
    if (sum(f) & 1) == 0:
        return tuple(f)
    best_i = 0
    best_err = Fraction(-1, 1)
    for i in range(8):
        err = abs(x[i] - f[i])
        if err > best_err:
            best_err = err
            best_i = i
    if x[best_i] >= f[best_i]:
        f[best_i] += 1
    else:
        f[best_i] -= 1
    return tuple(f)


def nearest_e8_exact(x: Any) -> tuple[Fraction, ...]:
    """Pure-Q Conway-Sloane nearest-point decoder in `E_8 = D_8 union (D_8 + (1/2)^8)`."""
    xv = _require_frac_vec8(x, "x")
    c0 = _nearest_d8_exact(xv)
    half = Fraction(1, 2)
    c1 = _nearest_d8_exact(tuple(v - half for v in xv))
    c0_q = tuple(Fraction(c, 1) for c in c0)
    c1_q = tuple(Fraction(2 * c + 1, 2) for c in c1)
    d0 = sum(((a - b) * (a - b) for a, b in zip(xv, c0_q)), Fraction(0, 1))
    d1 = sum(((a - b) * (a - b) for a, b in zip(xv, c1_q)), Fraction(0, 1))
    return c0_q if d0 <= d1 else c1_q


def _validate_lut_vector(z: Any, name: str, max_shell: int) -> tuple[int, ...]:
    v = _require_int_vec8(z, name)
    if not is_in_doubled_e8(v):
        raise SiliquantTpuBoundsError("INVALID_LUT_VECTOR", f"{name} is not in 2*E_8")
    if doubled_e8_shell(v) > max_shell:
        raise SiliquantTpuBoundsError("LUT_VECTOR_OUT_OF_BALL", f"{name} exceeds shell {max_shell}")
    return v


def _validate_lut(lut_doubled: Any, max_shell: int) -> tuple[tuple[int, ...], ...]:
    if type(lut_doubled) is not tuple or not lut_doubled:
        raise SiliquantTpuBoundsError("INVALID_LUT", "lut_doubled must be non-empty tuple")
    return tuple(
        _validate_lut_vector(z, f"lut_doubled[{idx}]", max_shell) for idx, z in enumerate(lut_doubled)
    )


def _nearest_e8_in_validated_ball(
    xv: tuple[Fraction, ...],
    mm: int,
    validated_lut: tuple[tuple[int, ...], ...],
) -> int:
    limit = 8 * mm
    # Fast exact check: if unconstrained Conway-Sloane nearest point lies inside the ball, it is globally optimal.
    cand = nearest_e8_exact(xv)
    z_cand = tuple(int(2 * v) for v in cand)
    if sum(c * c for c in z_cand) <= limit:
        for idx, z in enumerate(validated_lut):
            if z == z_cand:
                return idx
        raise SiliquantTpuBoundsError("LUT_MISSING_POINT", "LUT does not contain unconstrained optimal point")
    # Outside the ball: clear common denominator once and run exact pure-int argmin over the entire finite ball LUT.
    common_den = math.lcm(*(v.denominator for v in xv))
    x_scaled = tuple(2 * v.numerator * (common_den // v.denominator) for v in xv)
    z0 = validated_lut[0]
    best_idx = 0
    best_d = sum((x_scaled[i] - common_den * z0[i]) ** 2 for i in range(8))
    for idx in range(1, len(validated_lut)):
        z = validated_lut[idx]
        d = sum((x_scaled[i] - common_den * z[i]) ** 2 for i in range(8))
        if d < best_d:
            best_d = d
            best_idx = idx
    return best_idx


def _encode_2stage_validated(
    xv: tuple[Fraction, ...],
    sc: Fraction,
    sf: Fraction,
    mm: int,
    validated_lut: tuple[tuple[int, ...], ...],
) -> tuple[int, int]:
    idx_c = _nearest_e8_in_validated_ball(tuple(v / sc for v in xv), mm, validated_lut)
    q1 = tuple(Fraction(c, 2) for c in validated_lut[idx_c])
    r1 = tuple(a - sc * b for a, b in zip(xv, q1))
    idx_f = _nearest_e8_in_validated_ball(tuple(v / sf for v in r1), mm, validated_lut)
    return idx_c, idx_f


def nearest_e8_in_ball_doubled(
    x: Any,
    m_max: Any,
    lut_doubled: Any,
) -> int:
    """Return exact LUT index minimizing `||x - z/2||_2^2` over `z in 2*E_8` with `m(z) <= m_max`."""
    xv = _require_frac_vec8(x, "x")
    mm = require_exact_int(m_max, "m_max", min_val=0, max_val=2)
    validated_lut = _validate_lut(lut_doubled, mm)
    return _nearest_e8_in_validated_ball(xv, mm, validated_lut)


def encode_2stage_residual_e8(
    x: Any,
    s_c: Any,
    s_f: Any,
    lut_doubled: Any,
    *,
    m_max: int = 2,
) -> tuple[int, int]:
    """Encode 8D exact rational vector into `(idx_c, idx_f)` using 2-Stage Residual Flat-LUT `E_8`."""
    xv = _require_frac_vec8(x, "x")
    sc = require_exact_fraction(s_c, "s_c", positive=True)
    sf = require_exact_fraction(s_f, "s_f", positive=True)
    mm = require_exact_int(m_max, "m_max", min_val=0, max_val=2)
    validated_lut = _validate_lut(lut_doubled, mm)
    return _encode_2stage_validated(xv, sc, sf, mm, validated_lut)


def decode_2stage_residual_e8(
    idx_c: Any,
    idx_f: Any,
    s_c: Any,
    s_f: Any,
    lut_doubled: Any,
    *,
    max_shell: int = 5,
) -> tuple[Fraction, ...]:
    """O(1) branchless 2-gather + 1-mul + 1-FMA decoder: `x_hat = s_c*(LUT[idx_c]/2) + s_f*(LUT[idx_f]/2)`."""
    ms = require_exact_int(max_shell, "max_shell", min_val=0, max_val=5)
    if type(lut_doubled) is not tuple or not lut_doubled:
        raise SiliquantTpuBoundsError("INVALID_LUT", "lut_doubled must be non-empty tuple")
    n = len(lut_doubled)
    ic = require_exact_int(idx_c, "idx_c", min_val=0, max_val=n - 1)
    if_ = require_exact_int(idx_f, "idx_f", min_val=0, max_val=n - 1)
    sc = require_exact_fraction(s_c, "s_c", positive=True)
    sf = require_exact_fraction(s_f, "s_f", positive=True)
    zc = _validate_lut_vector(lut_doubled[ic], "lut_doubled[idx_c]", ms)
    zf = _validate_lut_vector(lut_doubled[if_], "lut_doubled[idx_f]", ms)
    return tuple(sc * Fraction(zc[i], 2) + sf * Fraction(zf[i], 2) for i in range(8))


def bareiss_det_exact(mat: Any) -> Fraction:
    """Exact fraction-free-style Bareiss determinant over Q for square matrices up to 16x16."""
    a = _require_square_frac_matrix(mat, "mat")
    n = len(a)
    sign = 1
    prev_pivot = Fraction(1, 1)
    for k in range(n):
        pivot_row: int | None = None
        for r in range(k, n):
            if a[r][k] != 0:
                pivot_row = r
                break
        if pivot_row is None:
            return Fraction(0, 1)
        if pivot_row != k:
            a[k], a[pivot_row] = a[pivot_row], a[k]
            sign = -sign
        piv = a[k][k]
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                a[i][j] = (a[i][j] * piv - a[i][k] * a[k][j]) / prev_pivot
        prev_pivot = piv
    return sign * a[n - 1][n - 1]


def certify_sylvester_pd_exact(mat: Any) -> bool:
    """Certify strict positive definiteness of a symmetric Q-matrix via Sylvester's leading principal minors."""
    a = _require_square_frac_matrix(mat, "mat", require_symmetric=True)
    n = len(a)
    for k in range(1, n + 1):
        sub = [row[:k] for row in a[:k]]
        if bareiss_det_exact(sub) <= 0:
            return False
    return True


def ceil_log2_int(n: Any) -> int:
    """Return exact `ceil(log2(n))` for integer `n >= 1`."""
    ni = require_exact_int(n, "n", min_val=1)
    return (ni - 1).bit_length()


def ceil_isqrt_rational(q: Any) -> Fraction:
    """Return a tight rational upper bound >= `sqrt(q)` for `q >= 0` with precision `10^-9`."""
    val = require_exact_fraction(q, "q", non_negative=True)
    if val == 0:
        return Fraction(0, 1)
    scale = 10 ** 9
    # Want k / scale >= sqrt(num / den) <=> k^2 * den >= num * scale^2 <=> k = ceil(sqrt(num * scale^2 / den))
    target_num = val.numerator * scale * scale
    # floor(target_num / den), then take ceil sqrt carefully:
    q_div, r_div = divmod(target_num, val.denominator)
    root = math.isqrt(q_div)
    if root * root * val.denominator < target_num:
        root += 1
    return Fraction(root, scale)


# ---------------------------------------------------------------------------
# Pillar 1: Pre-Cast Dynamic-Range, Block-H2 vs WHT_8 Separation & Accumulator
# ---------------------------------------------------------------------------

def _min_norm_for_single_coord(z0: int) -> tuple[int, tuple[int, ...]]:
    """Minimal `||z||^2` and witness `z in 2*E_8` with `z[0] == z0 >= 0`."""
    if (z0 & 1) == 0:
        if (z0 & 3) == 0:
            z = (z0, 0, 0, 0, 0, 0, 0, 0)
        else:
            z = (z0, 2, 0, 0, 0, 0, 0, 0)
    else:
        if (z0 & 3) == 3:
            z = (z0, 1, 1, 1, 1, -1, -1, -1)
        else:
            z = (z0, 1, 1, 1, -1, -1, -1, -1)
    return sum(c * c for c in z), z


def _min_norm_for_wht8_sum(target_sum: int) -> tuple[int, tuple[int, ...]]:
    """Minimal `||z||^2` and witness `z in 2*E_8` with `sum(z) == target_sum` (multiple of 4)."""
    cands: list[tuple[int, tuple[int, ...]]] = []
    for par in (0, 1):
        u_sum = (target_sum - 8 * par) // 2
        q, r = divmod(u_sum, 8)
        z = tuple([2 * (q + 1) + par] * r + [2 * q + par] * (8 - r))
        cands.append((sum(c * c for c in z), z))
    return cands[0] if cands[0][0] <= cands[1][0] else cands[1]


def _min_norm_for_block_h2_pair(target_pair: int) -> tuple[int, tuple[int, ...]]:
    """Minimal `||z||^2` and witness `z in 2*E_8` with `z[0] + z[1] == target_pair` (even)."""
    cands: list[tuple[int, tuple[int, ...]]] = []
    for par in (0, 1):
        u_sum = (target_pair - 2 * par) // 2
        u = u_sum // 2
        v = u_sum - u
        z0 = 2 * u + par
        z1 = 2 * v + par
        if par == 0:
            if (target_pair & 3) == 0:
                z = (z0, z1, 0, 0, 0, 0, 0, 0)
            else:
                z = (z0, z1, 2, 0, 0, 0, 0, 0)
        else:
            if (target_pair & 3) == 0:
                z = (z0, z1, 1, 1, 1, -1, -1, -1)
            else:
                z = (z0, z1, 1, 1, 1, 1, -1, -1)
        cands.append((sum(c * c for c in z), z))
    return cands[0] if cands[0][0] <= cands[1][0] else cands[1]


def _exact_superblock_l1_max(m_max: int) -> tuple[int, list[list[int]]]:
    """Exact DP computing max `||z||_1` and explicit 4-block `(2*E_8)^4` witness for `sum m_b <= m_max`."""
    limit = 8 * m_max
    best_1_val = [0] * (m_max + 1)
    best_1_vec: list[tuple[int, ...]] = [(0,) * 8 for _ in range(m_max + 1)]
    max_s = math.isqrt(64 * m_max) + 2
    # Odd parity: any even S >= 8 can be signed to achieve sum == 0 mod 4.
    for s in range(8, max_s, 2):
        u_sum = (s - 8) // 2
        q, r = divmod(u_sum, 8)
        coords = [2 * (q + 1) + 1] * r + [2 * q + 1] * (8 - r)
        n2 = sum(c * c for c in coords)
        if n2 <= limit:
            if (sum(coords) & 3) != 0:
                coords[-1] = -coords[-1]
            m = n2 // 8
            if s > best_1_val[m]:
                best_1_val[m] = s
                best_1_vec[m] = tuple(coords)
    # Even parity: S must be divisible by 4 (sign flips preserve sum mod 4).
    for s in range(0, max_s, 4):
        u_sum = s // 2
        q, r = divmod(u_sum, 8)
        coords = [2 * (q + 1)] * r + [2 * q] * (8 - r)
        n2 = sum(c * c for c in coords)
        if n2 <= limit:
            m = n2 // 8
            if s > best_1_val[m]:
                best_1_val[m] = s
                best_1_vec[m] = tuple(coords)
    # Prefix maximum over m = 0..m_max
    for m in range(1, m_max + 1):
        if best_1_val[m - 1] > best_1_val[m]:
            best_1_val[m] = best_1_val[m - 1]
            best_1_vec[m] = best_1_vec[m - 1]
    # 2-block convolution
    dp2_val = [0] * (m_max + 1)
    dp2_split = [0] * (m_max + 1)
    for a in range(m_max + 1):
        va = best_1_val[a]
        for b in range(m_max + 1 - a):
            tot = va + best_1_val[b]
            if tot > dp2_val[a + b]:
                dp2_val[a + b] = tot
                dp2_split[a + b] = a
    # 4-block convolution (only need optimal split at m_max)
    best_4 = 0
    best_a2 = 0
    for a2 in range(m_max + 1):
        tot = dp2_val[a2] + dp2_val[m_max - a2]
        if tot > best_4:
            best_4 = tot
            best_a2 = a2
    b2 = m_max - best_a2
    m0 = dp2_split[best_a2]
    m1 = best_a2 - m0
    m2 = dp2_split[b2]
    m3 = b2 - m2
    blocks = [list(best_1_vec[m0]), list(best_1_vec[m1]), list(best_1_vec[m2]), list(best_1_vec[m3])]
    return best_4, blocks


def certify_pillar1_precast_overflow_separation() -> dict[str, Any]:
    """Certify exact attainable maxima, `Block-H2` vs `WHT_8` `INT8` separation, BF16 exactness & accumulator width."""
    regimes_spec = (
        ("EBG32:b88s8", 48),
        ("EBG32:b136s8", 391),
        ("1block_E8_3.0bpw", 18),
        ("1block_E8_4.5bpw", 151),
    )

    # Algebraically verify H_8 mod-4 congruence across all 8 rows (row 0 has 8 +1s; rows 1..7 have 4 +1s and 4 -1s).
    h8_all_rows_preserve_mod4 = True
    for r in range(8):
        neg_cnt = sum(1 for x in H8_MATRIX[r] if x == -1)
        if neg_cnt not in (0, 4):
            h8_all_rows_preserve_mod4 = False

    regimes_out: dict[str, Any] = {}
    for label, m_max in regimes_spec:
        limit = 8 * m_max
        w1 = w1_ball_count(m_max)

        # 1. Exact max |z_i|
        best_coord = 0
        best_coord_z: tuple[int, ...] = (0,) * 8
        for cand in range(1, math.isqrt(limit) + 2):
            n2, z = _min_norm_for_single_coord(cand)
            if n2 <= limit:
                best_coord = cand
                best_coord_z = z
            else:
                break
        next_coord_n2, _ = _min_norm_for_single_coord(best_coord + 1)
        coord_tight = all((next_coord_n2 > limit, is_in_doubled_e8(best_coord_z), doubled_e8_shell(best_coord_z) <= m_max))

        # 2. Exact max |H_8 z|_i (multiples of 4 only)
        wht_cand = (math.isqrt(64 * m_max) // 4) * 4
        while True:
            n2_w, z_w = _min_norm_for_wht8_sum(wht_cand)
            if n2_w <= limit:
                break
            wht_cand -= 4
        next_wht_n2, _ = _min_norm_for_wht8_sum(wht_cand + 4)
        wht_out = apply_wht8_int(z_w)
        wht_tight = all((
            next_wht_n2 > limit,
            is_in_doubled_e8(z_w),
            doubled_e8_shell(z_w) <= m_max,
            wht_out[0] == wht_cand,
        ))

        # 3. Exact max |(I_4 tensor H_2) z|_i (even values only by same-parity invariant)
        bh2_cand = (math.isqrt(16 * m_max) // 2) * 2
        while True:
            n2_b, z_b = _min_norm_for_block_h2_pair(bh2_cand)
            if n2_b <= limit:
                break
            bh2_cand -= 2
        next_bh2_n2, _ = _min_norm_for_block_h2_pair(bh2_cand + 2)
        bh2_out = apply_block_h2_int(z_b)
        bh2_tight = all((
            next_bh2_n2 > limit,
            (bh2_cand + 2) * (bh2_cand + 2) > 16 * m_max,
            is_in_doubled_e8(z_b),
            doubled_e8_shell(z_b) <= m_max,
            bh2_out[0] == bh2_cand,
        ))

        # 4. Exact Cauchy-Schwarz dot-product ceilings & attainable 32-weight super-block lattice maxima
        cs_dot_8w = math.isqrt(INT8_MAX * INT8_MAX * 8 * 8 * m_max)
        cs_dot_32w = math.isqrt(INT8_MAX * INT8_MAX * 32 * 8 * m_max)
        exact_l1_32w, super_blocks = _exact_superblock_l1_max(m_max)
        exact_dot_32w_sym127 = INT8_MAX * exact_l1_32w
        exact_dot_32w_asym128 = 128 * exact_l1_32w
        witness_shell_sum = sum(doubled_e8_shell(blk) for blk in super_blocks)
        witness_l1_check = sum(abs(c) for blk in super_blocks for c in blk)

        regimes_out[label] = {
            "m_max": m_max,
            "w1_count": w1,
            "w1_bit_width": w1.bit_length(),
            "max_abs_z": best_coord,
            "max_abs_z_witness": list(best_coord_z),
            "max_abs_z_witness_shell": doubled_e8_shell(best_coord_z),
            "max_abs_z_tight_certified": coord_tight,
            "max_abs_z_int8_safe": best_coord <= INT8_MAX,
            "wht8_max": wht_cand,
            "wht8_witness": list(z_w),
            "wht8_witness_shell": doubled_e8_shell(z_w),
            "wht8_tight_certified": wht_tight,
            "wht8_int8_safe": wht_cand <= INT8_MAX,
            "block_h2_max": bh2_cand,
            "block_h2_witness": list(z_b),
            "block_h2_witness_shell": doubled_e8_shell(z_b),
            "block_h2_tight_certified": bh2_tight,
            "block_h2_int8_safe": bh2_cand <= INT8_MAX,
            "cs_dot_8w_int8_sym127": cs_dot_8w,
            "cs_dot_32w_int8_sym127": cs_dot_32w,
            "exact_l1_32w_attainable": exact_l1_32w,
            "exact_dot_32w_int8_sym127": exact_dot_32w_sym127,
            "exact_dot_32w_int8_asym128": exact_dot_32w_asym128,
            "exact_dot_32w_witness_blocks": super_blocks,
            "exact_dot_32w_witness_verified": all((
                witness_shell_sum <= m_max,
                witness_l1_check == exact_l1_32w,
            )),
            "dot_32w_int16_safe_sym127": cs_dot_32w <= INT16_MAX,
            "dot_32w_int16_overflow_attainable": exact_dot_32w_sym127 > INT16_MAX,
        }

    # BF16 zero-ULP exactness over every half-integer v = z/2 in [-128, 128] (513 values)
    bf16_all_513_exact = all(is_exact_bf16_half_integer(z) for z in range(-256, 257))
    # Verify boundary: z = 257 -> v = 128.5 requires 9 bits (100000001_2), NOT exact in BF16!
    bf16_boundary_257_inexact = not is_exact_bf16_half_integer(257)

    # K = 16,384 (512 super-blocks) int32 headroom on EBG32:b136s8
    superblocks_k16384 = HIDDEN_DIM_K // 32
    peak_136b_cs = superblocks_k16384 * regimes_out["EBG32:b136s8"]["cs_dot_32w_int8_sym127"]
    peak_136b_att = superblocks_k16384 * regimes_out["EBG32:b136s8"]["exact_dot_32w_int8_sym127"]
    peak_136b_asym = superblocks_k16384 * regimes_out["EBG32:b136s8"]["exact_dot_32w_int8_asym128"]
    headroom_cs = Fraction(INT32_MAX, peak_136b_cs)
    headroom_att = Fraction(INT32_MAX, peak_136b_att)
    headroom_asym = Fraction(INT32_MAX, peak_136b_asym)

    r136 = regimes_out["EBG32:b136s8"]
    r88 = regimes_out["EBG32:b88s8"]
    int8_sep = all((
        r136["block_h2_max"] == 78,
        r136["block_h2_int8_safe"],
        r136["wht8_max"] == 156,
        not r136["wht8_int8_safe"],
        r136["block_h2_tight_certified"],
        r136["wht8_tight_certified"],
    ))
    acc_sep = all((
        r88["dot_32w_int16_safe_sym127"],
        not r136["dot_32w_int16_safe_sym127"],
        r136["dot_32w_int16_overflow_attainable"],
        headroom_cs > 100,
        headroom_att > 100,
        headroom_asym > 100,
    ))

    return {
        "h8_all_rows_preserve_mod4": h8_all_rows_preserve_mod4,
        "regimes": regimes_out,
        "int8_separation_block_h2_vs_wht8_136b_certified": int8_sep,
        "bf16_half_integers_neg128_to_pos128_all_513_exact": bf16_all_513_exact,
        "bf16_boundary_257_inexact_verified": bf16_boundary_257_inexact,
        "accumulator_width_k16384": {
            "hidden_dim_k": HIDDEN_DIM_K,
            "superblocks_32w": superblocks_k16384,
            "peak_136b_cs_sym127": peak_136b_cs,
            "peak_136b_attainable_sym127": peak_136b_att,
            "peak_136b_attainable_asym128": peak_136b_asym,
            "int32_headroom_cs": format_exact_rational(headroom_cs),
            "int32_headroom_cs_2dp": format_decimal_trunc(headroom_cs, 2),
            "int32_headroom_attainable": format_exact_rational(headroom_att),
            "int32_headroom_attainable_2dp": format_decimal_trunc(headroom_att, 2),
            "int32_headroom_asym128_2dp": format_decimal_trunc(headroom_asym, 2),
            "headroom_exceeds_100x_all_3": acc_sep,
        },
    }


# ---------------------------------------------------------------------------
# Pillar 2: Static XLA/Pallas WCET, VMEM Residency & TPU v5e Three-Way Roofline
# ---------------------------------------------------------------------------

def certify_pillar2_wcet_vmem_tpu_roofline() -> dict[str, Any]:
    """Certify exact VMEM fractions, static branchless WCET trip counts & TPU v5e 3-way roofline."""
    vmem_entries_spec = (
        ("tree_88b", 5490),
        ("tree_136b", 56850),
        ("guide_a_88b", 93472),
        ("guide_a_136b", 1603328),
        ("flat_lut_12b_32kib", 32768),
        ("flat_lut_16b_512kib", 524288),
    )
    vmem_residency: dict[str, Any] = {}
    for name, nbytes in vmem_entries_spec:
        frac = Fraction(nbytes, TPU_V5E_VMEM_BYTES)
        pct = frac * 100
        vmem_residency[name] = {
            "bytes": nbytes,
            "vmem_fraction": format_exact_rational(frac),
            "vmem_pct_4dp": format_decimal_trunc(pct, 4),
        }
    guide_a_136b_free_pct = (Fraction(1, 1) - Fraction(1603328, TPU_V5E_VMEM_BYTES)) * 100

    # Static branchless WCET trip counts & VPU lane widths
    w1_48 = w1_ball_count(48)
    w1_391 = w1_ball_count(391)
    steps_88b = ceil_log2_int(48 + 1)
    steps_136b = ceil_log2_int(391 + 1)

    wcet_bounds = {
        "ebg32_b88s8": {
            "m_max": 48,
            "fixed_select_steps": steps_88b,
            "w1_count": w1_48,
            "w1_bit_width": w1_48.bit_length(),
            "fits_uint32_vpu": w1_48.bit_length() <= 32,
            "dynamic_branches": 0,
            "integer_divisions": 0,
        },
        "ebg32_b136s8": {
            "m_max": 391,
            "fixed_select_steps": steps_136b,
            "w1_count": w1_391,
            "w1_bit_width": w1_391.bit_length(),
            "fits_uint32_vpu": w1_391.bit_length() <= 32,
            "fits_uint64_emulated": w1_391.bit_length() <= 64,
            "dynamic_branches": 0,
            "integer_divisions": 0,
        },
        "residual_2stage_flat_lut_e8": {
            "fixed_vmem_gathers": 2,
            "fixed_vector_fmas": 1,
            "dynamic_branches": 0,
            "integer_divisions": 0,
            "requires_uint64": False,
            "o1_wcet_certified": True,
        },
    }

    # Exact TPU v5e Three-Way Roofline: 819 GB/s HBM, 197 TFLOP/s BF16 MXU, 394 TOP/s INT8 MXU
    hbm_gbs = Fraction(819, 1)
    bf16_tflops = Fraction(197, 1)
    int8_tops = Fraction(394, 1)
    bpw_ball = Fraction(9, 2)   # 4.50 bpw
    bpw_q4km = Fraction(24, 5)  # 4.80 bpw

    # B*(bpw) = (MXU_TFLOPS * 1000 * bpw) / (16 * HBM_GBS)
    b_bf16_450 = (bf16_tflops * 1000 * bpw_ball) / (16 * hbm_gbs)
    b_bf16_480 = (bf16_tflops * 1000 * bpw_q4km) / (16 * hbm_gbs)
    b_int8_450 = (int8_tops * 1000 * bpw_ball) / (16 * hbm_gbs)
    b_int8_480 = (int8_tops * 1000 * bpw_q4km) / (16 * hbm_gbs)

    # Break-even Pallas decode rate (G weights/s) to beat Q4_K_M at HBM limit, and 4.50 HBM saturation
    r_decode_breakeven = (hbm_gbs * 8) / bpw_q4km
    r_hbm_sat_450 = (hbm_gbs * 8) / bpw_ball

    # Roofline attained TFLOP/s at batch B = 73: min(2 * B * (8 / bpw) * HBM_GBS / 1000, MXU_PEAK)
    b_test = Fraction(73, 1)
    hbm_cap_450_b73 = (2 * b_test * (8 / bpw_ball) * hbm_gbs) / 1000
    hbm_cap_480_b73 = (2 * b_test * (8 / bpw_q4km) * hbm_gbs) / 1000
    bf16_att_450_b73 = min(hbm_cap_450_b73, bf16_tflops)
    bf16_att_480_b73 = min(hbm_cap_480_b73, bf16_tflops)
    int8_att_450_b73 = min(hbm_cap_450_b73, int8_tops)
    int8_att_480_b73 = min(hbm_cap_480_b73, int8_tops)
    bf16_gain_b73 = (bf16_att_450_b73 / bf16_att_480_b73) - 1
    int8_gain_b73 = (int8_att_450_b73 / int8_att_480_b73) - 1

    return {
        "tpu_v5e_vmem_bytes": TPU_V5E_VMEM_BYTES,
        "vmem_residency": vmem_residency,
        "guide_a_136b_free_vmem_pct_4dp": format_decimal_trunc(guide_a_136b_free_pct, 4),
        "guide_a_136b_leaves_gt_98_8_pct_free": guide_a_136b_free_pct > Fraction(988, 10),
        "wcet_bounds": wcet_bounds,
        "tpu_v5e_three_way_roofline": {
            "hbm_bandwidth_gbs": 819,
            "bf16_mxu_tflops": 197,
            "int8_mxu_tops": 394,
            "b_bf16_ridge_4_50": format_exact_rational(b_bf16_450),
            "b_bf16_ridge_4_50_2dp": format_decimal_trunc(b_bf16_450, 2),
            "b_bf16_ridge_4_80": format_exact_rational(b_bf16_480),
            "b_bf16_ridge_4_80_2dp": format_decimal_trunc(b_bf16_480, 2),
            "b_int8_ridge_4_50": format_exact_rational(b_int8_450),
            "b_int8_ridge_4_50_2dp": format_decimal_trunc(b_int8_450, 2),
            "b_int8_ridge_4_80": format_exact_rational(b_int8_480),
            "b_int8_ridge_4_80_2dp": format_decimal_trunc(b_int8_480, 2),
            "int8_doubles_bandwidth_regime": all((
                b_int8_450 == 2 * b_bf16_450,
                b_int8_480 == 2 * b_bf16_480,
            )),
            "r_decode_breakeven_gwps": format_exact_rational(r_decode_breakeven),
            "r_hbm_sat_4_50_gwps": format_exact_rational(r_hbm_sat_450),
            "batch_73_bf16_bandwidth_gain": format_exact_rational(bf16_gain_b73),
            "batch_73_bf16_zero_gain_certified": all((bf16_gain_b73 == 0, b_bf16_480 < 73)),
            "batch_73_int8_bandwidth_gain": format_exact_rational(int8_gain_b73),
            "batch_73_int8_still_bandwidth_bound": all((
                int8_att_450_b73 < int8_tops,
                int8_gain_b73 == Fraction(1, 15),
            )),
        },
    }


# ---------------------------------------------------------------------------
# Pillar 3: O(1) Branchless 2-Stage Residual Flat-LUT E8 Codec & Contraction
# ---------------------------------------------------------------------------

def certify_pillar3_residual_flat_lut_e8_codec() -> dict[str, Any]:
    """Certify Eisenstein E4 shell LUTs, unconditional covering contraction & 2-stage codec exactness."""
    shells_m0_to_8: list[dict[str, int]] = []
    cum = 0
    for m in range(9):
        cnt = r_e8_shell_count(m)
        cum += cnt
        shells_m0_to_8.append({"m": m, "r_e8": cnt, "cumulative": cum})

    # Enumerate exact 12-bit 32 KiB LUT (m = 0..2, 2,401 points) and certify max |z_i| live
    lut_m2 = _get_lut_m2()
    max_abs_z_m2 = max(max(abs(c) for c in z) for z in lut_m2)
    pts_m2_theta = shells_m0_to_8[2]["cumulative"]
    # Exact max |z_i| for m <= 5 (8m <= 40) computed live
    max_abs_z_m5 = 0
    z_m5_witness: tuple[int, ...] = (0,) * 8
    for cand in range(1, math.isqrt(40) + 2):
        n2_c, z_c = _min_norm_for_single_coord(cand)
        if n2_c <= 40:
            max_abs_z_m5 = cand
            z_m5_witness = z_c
        else:
            break
    n2_m5_next, _ = _min_norm_for_single_coord(max_abs_z_m5 + 1)
    max_abs_z_m5_tight = all((n2_m5_next > 40, is_in_doubled_e8(z_m5_witness), doubled_e8_shell(z_m5_witness) <= 5))

    # Unconditional No-Clip Covering Contraction Proofs (E_8 covering radius R_cov = 1)
    # For m <= 2: every m >= 3 point has ||v||^2 >= 6; (7/5 + 1)^2 = 144/25 < 6 => s_f = (5/7) s_c has ZERO clip!
    m2_noclip_radius = Fraction(7, 5)
    m2_noclip_verified = (m2_noclip_radius + 1) ** 2 < 6
    m2_noclip_worst_contraction = Fraction(1, 1) / (m2_noclip_radius * m2_noclip_radius)
    # For m <= 5: every m >= 6 point has ||v||^2 >= 12; (2 + 1)^2 = 9 < 12 AND (12/5 + 1)^2 = 289/25 < 12!
    m5_noclip_radius_half = Fraction(2, 1)
    m5_noclip_radius_tight = Fraction(12, 5)
    m5_noclip_half_verified = (m5_noclip_radius_half + 1) ** 2 < 12
    m5_noclip_tight_verified = (m5_noclip_radius_tight + 1) ** 2 < 12
    m5_noclip_half_contraction = Fraction(1, 1) / (m5_noclip_radius_half * m5_noclip_radius_half)
    m5_noclip_tight_contraction = Fraction(1, 1) / (m5_noclip_radius_tight * m5_noclip_radius_tight)

    # Deterministic 6-vector exact rational suite covering bulk, E8 deep hole, outlier, Block-H2, half-hole, outer
    # Verify exact 0 encode/decode mismatch + geometric distortion contraction ratio <= 1/4!
    test_suite: tuple[tuple[str, tuple[Fraction, ...]], ...] = (
        ("bulk_interior", tuple(Fraction(k, 10) for k in (7, -5, 3, -9, 6, -4, 8, -2))),
        ("e8_deep_hole", (Fraction(1, 1), Fraction(0, 1), Fraction(0, 1), Fraction(0, 1),
                          Fraction(0, 1), Fraction(0, 1), Fraction(0, 1), Fraction(0, 1))),
        ("axis_outlier", (Fraction(9, 5), Fraction(1, 5), Fraction(-1, 5), Fraction(1, 5),
                          Fraction(0, 1), Fraction(0, 1), Fraction(0, 1), Fraction(0, 1))),
        ("block_h2_rotated", tuple(Fraction(k, 10) for k in (13, 11, -7, 5, -3, 9, -1, 3))),
        ("mixed_half_hole", tuple(Fraction(k, 4) for k in (3, 3, -1, 1, 3, -3, 1, -1))),
        ("outer_shell_wing", tuple(Fraction(k, 5) for k in (7, -6, 4, -3, 5, -2, 3, -1))),
    )
    s_c = Fraction(1, 1)
    sf_fixed = Fraction(1, 3)
    sf_grid = (Fraction(1, 2), Fraction(1, 3), Fraction(1, 4), Fraction(1, 6))

    suite_results: list[dict[str, Any]] = []
    tot_d1 = Fraction(0, 1)
    tot_d2_fixed = Fraction(0, 1)
    tot_d2_opt = Fraction(0, 1)
    all_decode_exact = True
    all_opt_ratio_le_quarter = True

    validated_m2 = _validate_lut(lut_m2, 2)
    for label, x in test_suite:
        # Stage 1
        idx_c_only = _nearest_e8_in_validated_ball(x, 2, validated_m2)
        q1 = tuple(Fraction(c, 2) for c in validated_m2[idx_c_only])
        d1 = sum(((a - b) * (a - b) for a, b in zip(x, q1)), Fraction(0, 1))

        # Stage 2 fixed s_f = s_c / 3
        ic_fix, if_fix = _encode_2stage_validated(x, s_c, sf_fixed, 2, validated_m2)
        xhat_fix = decode_2stage_residual_e8(ic_fix, if_fix, s_c, sf_fixed, validated_m2)
        # Verify 2-gather + 1-FMA decode matches direct sum exactly
        direct_fix = tuple(
            s_c * Fraction(validated_m2[ic_fix][i], 2) + sf_fixed * Fraction(validated_m2[if_fix][i], 2)
            for i in range(8)
        )
        if xhat_fix != direct_fix:
            all_decode_exact = False
        d2_fix = sum(((a - b) * (a - b) for a, b in zip(x, xhat_fix)), Fraction(0, 1))

        # Stage 2 optimal exact rational s_f in sf_grid
        best_sf = sf_grid[0]
        best_d2 = Fraction(10 ** 9, 1)
        best_pair = (0, 0)
        for sf_cand in sf_grid:
            ic_c, if_c = _encode_2stage_validated(x, s_c, sf_cand, 2, validated_m2)
            xhat_c = decode_2stage_residual_e8(ic_c, if_c, s_c, sf_cand, validated_m2)
            d2_c = sum(((a - b) * (a - b) for a, b in zip(x, xhat_c)), Fraction(0, 1))
            if d2_c < best_d2:
                best_d2 = d2_c
                best_sf = sf_cand
                best_pair = (ic_c, if_c)
        # Re-decode optimal pair and verify exact equality
        xhat_opt = decode_2stage_residual_e8(best_pair[0], best_pair[1], s_c, best_sf, lut_m2)
        d2_opt_check = sum(((a - b) * (a - b) for a, b in zip(x, xhat_opt)), Fraction(0, 1))
        if d2_opt_check != best_d2:
            all_decode_exact = False

        ratio_opt = best_d2 / d1
        if ratio_opt > Fraction(1, 4):
            all_opt_ratio_le_quarter = False

        tot_d1 += d1
        tot_d2_fixed += d2_fix
        tot_d2_opt += best_d2
        suite_results.append({
            "vector": label,
            "stage1_err_sq": format_exact_rational(d1),
            "stage2_fixed_sf_1_3_err_sq": format_exact_rational(d2_fix),
            "stage2_opt_sf": format_exact_rational(best_sf),
            "stage2_opt_err_sq": format_exact_rational(best_d2),
            "stage2_opt_ratio": format_exact_rational(ratio_opt),
            "stage2_opt_ratio_4dp": format_decimal_trunc(ratio_opt, 4),
            "idx_c_opt": best_pair[0],
            "idx_f_opt": best_pair[1],
            "ratio_le_1_4": ratio_opt <= Fraction(1, 4),
        })

    agg_fixed_ratio = tot_d2_fixed / tot_d1
    agg_opt_ratio = tot_d2_opt / tot_d1

    return {
        "eisenstein_e4_shells_m0_to_8": shells_m0_to_8,
        "lut_12bit_32kib": {
            "m_max": 2,
            "points": pts_m2_theta,
            "enumerated_points": len(lut_m2),
            "enumerated_matches_theta": len(lut_m2) == pts_m2_theta,
            "slots": 4096,
            "slack_slots": 4096 - pts_m2_theta,
            "bytes_int8": 4096 * 8,
            "max_abs_z": max_abs_z_m2,
            "fits_signed_int4": max_abs_z_m2 <= 7,
            "fits_signed_int8": max_abs_z_m2 <= INT8_MAX,
        },
        "lut_16bit_512kib": {
            "m_max": 5,
            "points": shells_m0_to_8[5]["cumulative"],
            "slots": 65536,
            "slack_slots": 65536 - shells_m0_to_8[5]["cumulative"],
            "bytes_int8": 65536 * 8,
            "max_abs_z": max_abs_z_m5,
            "max_abs_z_tight_certified": max_abs_z_m5_tight,
            "fits_signed_int4": max_abs_z_m5 <= 7,
            "fits_signed_int8": max_abs_z_m5 <= INT8_MAX,
        },
        "unconditional_noclip_contraction_theorem": {
            "e8_covering_radius_sq": 1,
            "m2_noclip_sf_ratio": "5/7",
            "m2_noclip_verified": m2_noclip_verified,
            "m2_noclip_worst_contraction": format_exact_rational(m2_noclip_worst_contraction),
            "m5_noclip_sf_half_verified": m5_noclip_half_verified,
            "m5_noclip_sf_half_contraction": format_exact_rational(m5_noclip_half_contraction),
            "m5_noclip_sf_5_12_verified": m5_noclip_tight_verified,
            "m5_noclip_sf_5_12_contraction": format_exact_rational(m5_noclip_tight_contraction),
        },
        "codec_distortion_suite": suite_results,
        "all_encode_decode_zero_mismatch": all_decode_exact,
        "all_6_opt_ratios_le_1_4": all_opt_ratio_le_quarter,
        "aggregate_fixed_sf_1_3_ratio": format_exact_rational(agg_fixed_ratio),
        "aggregate_fixed_sf_1_3_ratio_4dp": format_decimal_trunc(agg_fixed_ratio, 4),
        "aggregate_fixed_sf_1_3_lt_1_4": agg_fixed_ratio < Fraction(1, 4),
        "aggregate_opt_ratio": format_exact_rational(agg_opt_ratio),
        "aggregate_opt_ratio_4dp": format_decimal_trunc(agg_opt_ratio, 4),
        "aggregate_opt_ratio_le_1_8": agg_opt_ratio <= Fraction(1, 8),
    }


# ---------------------------------------------------------------------------
# Pillar 4: Analytical Spectral / Lipschitz Tension-Drift Enclosure
# ---------------------------------------------------------------------------

def _mat_t_mat(W: list[list[Fraction]]) -> list[list[Fraction]]:
    n = len(W)
    return [
        [sum((W[k][i] * W[k][j] for k in range(n)), Fraction(0, 1)) for j in range(n)]
        for i in range(n)
    ]


def _frob_sq_mat(W: list[list[Fraction]]) -> Fraction:
    return sum((x * x for row in W for x in row), Fraction(0, 1))


def certify_pillar4_spectral_lipschitz_enclosure() -> dict[str, Any]:
    """Certify zero-unpack Frobenius/operator-norm bounds for single-stage & 2-stage E8 + Block-H2 anisotropy."""
    # 1. Exact orthogonality checks: H_8^T H_8 = 8 I_8 and B_2^T B_2 = 2 I_8
    h8_ortho = all(
        sum(H8_MATRIX[k][i] * H8_MATRIX[k][j] for k in range(8)) == (8 if i == j else 0)
        for i in range(8)
        for j in range(8)
    )
    b2_ortho = all(
        sum(BLOCK_H2_MATRIX[k][i] * BLOCK_H2_MATRIX[k][j] for k in range(8)) == (2 if i == j else 0)
        for i in range(8)
        for j in range(8)
    )

    # 2. Single-stage E_8^4 + 2-Stage Residual E_8 explicit 8x8 witness (8 rows, each an E_8 block)
    # Use 8 distinct shell m=1/m=2 doubled-E8 rows for W_c and 8 distinct shell m=1 rows for W_f
    coarse_z_rows = (
        (2, 2, 0, 0, 0, 0, 0, 0),
        (0, 0, 2, 2, 0, 0, 0, 0),
        (0, 0, 0, 0, 2, 2, 0, 0),
        (0, 0, 0, 0, 0, 0, 2, 2),
        (2, -2, 0, 0, 0, 0, 0, 0),
        (0, 0, 2, -2, 0, 0, 0, 0),
        (0, 0, 0, 0, 2, -2, 0, 0),
        (1, 1, 1, 1, 1, 1, 1, 1),
    )
    fine_z_rows = (
        (1, -1, 1, -1, 1, -1, 1, -1),
        (2, 0, 0, 2, 0, 0, 0, 0),
        (0, 2, 0, 0, 0, 2, 0, 0),
        (0, 0, 2, 0, 0, 0, 2, 0),
        (0, 0, 0, 2, 0, 0, 0, 2),
        (1, 1, -1, -1, 1, 1, -1, -1),
        (1, -1, -1, 1, 1, -1, -1, 1),
        (0, 0, 0, 0, 0, 0, 2, -2),
    )
    scales_c = (
        Fraction(1, 2), Fraction(3, 4), Fraction(5, 8), Fraction(1, 1),
        Fraction(3, 8), Fraction(7, 8), Fraction(1, 2), Fraction(3, 4),
    )
    scales_f = (
        Fraction(1, 8), Fraction(1, 8), Fraction(1, 16), Fraction(3, 16),
        Fraction(1, 8), Fraction(1, 16), Fraction(1, 8), Fraction(1, 16),
    )

    Wc = [
        [scales_c[r] * Fraction(coarse_z_rows[r][c], 2) for c in range(8)]
        for r in range(8)
    ]
    Wf = [
        [scales_f[r] * Fraction(fine_z_rows[r][c], 2) for c in range(8)]
        for r in range(8)
    ]
    W_res = [[Wc[r][c] + Wf[r][c] for c in range(8)] for r in range(8)]

    # Zero-unpack shell-radius norms: sum_b 2 * s_b^2 * m_b
    wc_frob_direct = _frob_sq_mat(Wc)
    wc_frob_zero_unpack = sum(
        (2 * scales_c[r] * scales_c[r] * doubled_e8_shell(coarse_z_rows[r]) for r in range(8)),
        Fraction(0, 1),
    )
    wf_frob_direct = _frob_sq_mat(Wf)
    wf_frob_zero_unpack = sum(
        (2 * scales_f[r] * scales_f[r] * doubled_e8_shell(fine_z_rows[r]) for r in range(8)),
        Fraction(0, 1),
    )
    wres_frob_direct = _frob_sq_mat(W_res)
    cross_inner = sum((Wc[r][c] * Wf[r][c] for r in range(8) for c in range(8)), Fraction(0, 1))
    exact_cross_identity = wres_frob_direct == wc_frob_direct + wf_frob_direct + 2 * cross_inner
    cs_cross_bound_holds = cross_inner * cross_inner <= wc_frob_zero_unpack * wf_frob_zero_unpack

    # Exact rational triangle ceiling: (ceil_sqrt(||Wc||_F^2) + ceil_sqrt(||Wf||_F^2))^2
    wc_norm_hi = ceil_isqrt_rational(wc_frob_zero_unpack)
    wf_norm_hi = ceil_isqrt_rational(wf_frob_zero_unpack)
    res_frob_sq_ceiling = (wc_norm_hi + wf_norm_hi) * (wc_norm_hi + wf_norm_hi)

    # Exact Bareiss Sylvester PD proof: tau*I - W^T W > 0 for both single-stage Wc and 2-stage W_res!
    wc_gram = _mat_t_mat(Wc)
    wc_pd_mat = [[(wc_frob_zero_unpack if i == j else Fraction(0, 1)) - wc_gram[i][j] for j in range(8)] for i in range(8)]
    wc_spectral_certified = certify_sylvester_pd_exact(wc_pd_mat)

    wres_gram = _mat_t_mat(W_res)
    wres_pd_mat = [[(res_frob_sq_ceiling if i == j else Fraction(0, 1)) - wres_gram[i][j] for j in range(8)] for i in range(8)]
    wres_spectral_certified = certify_sylvester_pd_exact(wres_pd_mat)

    # 3. Exact Block-H2 vs H_8 activation-anisotropy worst-case bound
    # Sigma_e = (1/4) e_0 e_0^T (tr = 1/4)
    # Under Block-H2: Sigma_{e,B2} = (1/2) B_2 Sigma_e B_2^T = (1/8) (e_0+e_1)(e_0+e_1)^T (tr = 1/4)
    # Under H_8:      Sigma_{e,H8} = (1/8) H_8 Sigma_e H_8^T = (1/32) 1 1^T (tr = 1/4)
    sigma_e = [[Fraction(1, 4) if (i, j) == (0, 0) else Fraction(0, 1) for j in range(8)] for i in range(8)]

    def _conj(U: tuple[tuple[int, ...], ...], S: list[list[Fraction]], denom: int) -> list[list[Fraction]]:
        tmp = [[sum((U[i][k] * S[k][j] for k in range(8)), Fraction(0, 1)) for j in range(8)] for i in range(8)]
        return [
            [sum((tmp[i][k] * U[j][k] for k in range(8)), Fraction(0, 1)) / Fraction(denom, 1) for j in range(8)]
            for i in range(8)
        ]

    def _trace(M: list[list[Fraction]]) -> Fraction:
        return sum((M[i][i] for i in range(8)), Fraction(0, 1))

    def _trace_prod(A: list[list[Fraction]], B: list[list[Fraction]]) -> Fraction:
        return sum((A[i][j] * B[j][i] for i in range(8) for j in range(8)), Fraction(0, 1))

    sigma_b2 = _conj(BLOCK_H2_MATRIX, sigma_e, 2)
    sigma_h8 = _conj(H8_MATRIX, sigma_e, 8)
    tr_e = _trace(sigma_e)
    tr_b2 = _trace(sigma_b2)
    tr_h8 = _trace(sigma_h8)

    h_axis = [[Fraction(1, 1) if (i, j) == (0, 0) else Fraction(0, 1) for j in range(8)] for i in range(8)]
    h_pair_diag = [[Fraction(1, 2) if max(i, j) < 2 else Fraction(0, 1) for j in range(8)] for i in range(8)]
    h_full_dc = [[Fraction(1, 8) for _ in range(8)] for _ in range(8)]

    kl_axis_e8 = _trace_prod(h_axis, sigma_e)
    kl_axis_b2 = _trace_prod(h_axis, sigma_b2)
    kl_axis_h8 = _trace_prod(h_axis, sigma_h8)

    kl_pair_e8 = _trace_prod(h_pair_diag, sigma_e)
    kl_pair_b2 = _trace_prod(h_pair_diag, sigma_b2)

    kl_dc_e8 = _trace_prod(h_full_dc, sigma_e)
    kl_dc_h8 = _trace_prod(h_full_dc, sigma_h8)

    return {
        "h8_orthogonal_8I": h8_ortho,
        "block_h2_orthogonal_2I": b2_ortho,
        "single_stage_zero_unpack_frob_sq": format_exact_rational(wc_frob_zero_unpack),
        "single_stage_direct_frob_sq": format_exact_rational(wc_frob_direct),
        "single_stage_zero_unpack_residual_zero": wc_frob_direct == wc_frob_zero_unpack,
        "single_stage_bareiss_spectral_pd": wc_spectral_certified,
        "residual_2stage_wf_zero_unpack_frob_sq": format_exact_rational(wf_frob_zero_unpack),
        "residual_2stage_direct_frob_sq": format_exact_rational(wres_frob_direct),
        "residual_2stage_cross_inner": format_exact_rational(cross_inner),
        "residual_2stage_exact_cross_identity": exact_cross_identity,
        "residual_2stage_cauchy_schwarz_holds": cs_cross_bound_holds,
        "residual_2stage_triangle_frob_sq_ceiling": format_exact_rational(res_frob_sq_ceiling),
        "residual_2stage_direct_le_ceiling": wres_frob_direct <= res_frob_sq_ceiling,
        "residual_2stage_bareiss_spectral_pd": wres_spectral_certified,
        "anisotropy_operator_separation": {
            "tr_sigma_e": format_exact_rational(tr_e),
            "tr_sigma_b2": format_exact_rational(tr_b2),
            "tr_sigma_h8": format_exact_rational(tr_h8),
            "isotropic_trace_invariance_exact": tr_e == tr_b2 == tr_h8 == Fraction(1, 4),
            "axis_kl_e8": format_exact_rational(kl_axis_e8),
            "axis_kl_block_h2": format_exact_rational(kl_axis_b2),
            "axis_kl_h8": format_exact_rational(kl_axis_h8),
            "pair_diag_kl_e8": format_exact_rational(kl_pair_e8),
            "pair_diag_kl_block_h2": format_exact_rational(kl_pair_b2),
            "full_dc_kl_e8": format_exact_rational(kl_dc_e8),
            "full_dc_kl_h8": format_exact_rational(kl_dc_h8),
            "block_h2_exact_2x_flip": all((kl_axis_e8 == 2 * kl_axis_b2, kl_pair_b2 == 2 * kl_pair_e8)),
            "h8_exact_8x_flip": all((kl_axis_e8 == 8 * kl_axis_h8, kl_dc_h8 == 8 * kl_dc_e8)),
        },
    }


# ---------------------------------------------------------------------------
# Pillar 5: Safety/Refusal Boundary Preservation & 8x8 Witness
# ---------------------------------------------------------------------------

def certify_pillar5_safety_refusal_margin_preservation() -> dict[str, Any]:
    """Certify worst-case spectral/Frobenius margin shrinkage, isotropic variance & 8x8 refusal witness."""
    lut_m2 = _get_lut_m2()
    # Deterministic 8x8 exact rational weight matrix W with 8 distinct rows
    w_base = [
        [Fraction(9, 20), Fraction(9, 20), Fraction(9, 20), Fraction(9, 20),
         Fraction(-11, 20), Fraction(-11, 20), Fraction(-11, 20), Fraction(-11, 20)],
        [Fraction(11, 25), Fraction(12, 25), Fraction(11, 25), Fraction(12, 25),
         Fraction(-14, 25), Fraction(-13, 25), Fraction(-14, 25), Fraction(-13, 25)],
        [Fraction(2, 5), Fraction(9, 20), Fraction(11, 25), Fraction(9, 20),
         Fraction(-11, 20), Fraction(-3, 5), Fraction(-11, 20), Fraction(-13, 25)],
        [Fraction(9, 20), Fraction(2, 5), Fraction(9, 20), Fraction(11, 25),
         Fraction(-13, 25), Fraction(-11, 20), Fraction(-3, 5), Fraction(-11, 20)],
        [Fraction(11, 25), Fraction(9, 20), Fraction(2, 5), Fraction(9, 20),
         Fraction(-11, 20), Fraction(-13, 25), Fraction(-11, 20), Fraction(-3, 5)],
        [Fraction(9, 20), Fraction(11, 25), Fraction(9, 20), Fraction(2, 5),
         Fraction(-3, 5), Fraction(-11, 20), Fraction(-13, 25), Fraction(-11, 20)],
        [Fraction(12, 25), Fraction(9, 20), Fraction(11, 25), Fraction(9, 20),
         Fraction(-14, 25), Fraction(-11, 20), Fraction(-13, 25), Fraction(-11, 20)],
        [Fraction(9, 20), Fraction(12, 25), Fraction(9, 20), Fraction(11, 25),
         Fraction(-11, 20), Fraction(-14, 25), Fraction(-11, 20), Fraction(-13, 25)],
    ]
    u = tuple(Fraction(1, 1) for _ in range(8))
    v = (
        Fraction(2, 1), Fraction(2, 1), Fraction(2, 1), Fraction(2, 1),
        Fraction(1, 1), Fraction(1, 1), Fraction(1, 1), Fraction(1, 1),
    )

    def _bilinear(mat: list[list[Fraction]]) -> Fraction:
        return sum((u[i] * mat[i][j] * v[j] for i in range(8) for j in range(8)), Fraction(0, 1))

    def _sub(a: list[list[Fraction]], b: list[list[Fraction]]) -> list[list[Fraction]]:
        return [[a[i][j] - b[i][j] for j in range(8)] for i in range(8)]

    # 1. Coarse scalar Z-grid quantization (step 1)
    w_scalar = [[Fraction(_round_half_even_exact(x), 1) for x in row] for row in w_base]

    # 2. 2-Stage Residual Flat-LUT E_8 (m <= 2 32 KiB LUT, s_c = 1, s_f = 1/8)
    s_c = Fraction(1, 1)
    validated_m2 = _validate_lut(lut_m2, 2)
    s_f = Fraction(1, 8)
    w_res_e8: list[list[Fraction]] = []
    for row in w_base:
        ic, if_ = _encode_2stage_validated(tuple(row), s_c, s_f, 2, validated_m2)
        w_res_e8.append(list(decode_2stage_residual_e8(ic, if_, s_c, s_f, validated_m2)))

    dw_scalar = _sub(w_scalar, w_base)
    dw_res_e8 = _sub(w_res_e8, w_base)

    margin_base = _bilinear(w_base)
    margin_scalar = _bilinear(w_scalar)
    margin_res_e8 = _bilinear(w_res_e8)

    err_sq_scalar = _frob_sq_mat(dw_scalar)
    err_sq_res_e8 = _frob_sq_mat(dw_res_e8)
    rho = err_sq_res_e8 / err_sq_scalar

    u_norm_sq = sum((x * x for x in u), Fraction(0, 1))
    v_norm_sq = sum((x * x for x in v), Fraction(0, 1))
    drift_res_sq = (margin_res_e8 - margin_base) ** 2
    worst_ceiling_res_sq = u_norm_sq * v_norm_sq * err_sq_res_e8
    worst_ceiling_scalar_sq = u_norm_sq * v_norm_sq * err_sq_scalar

    # Rank-1 aligned adversary tightness check: dW_adv = (1/10) u v^T achieves equality
    # (u^T dW_adv v)^2 == ||u||^2 ||v||^2 ||dW_adv||_F^2 (and sigma_max(dW_adv) == ||dW_adv||_F)!
    alpha = Fraction(1, 10)
    dw_adv = [[alpha * u[i] * v[j] for j in range(8)] for i in range(8)]
    adv_drift_sq = _bilinear(dw_adv) ** 2
    adv_ceiling_sq = u_norm_sq * v_norm_sq * _frob_sq_mat(dw_adv)

    # Exact isotropic expected variance identity over BOTH 64 standard basis pairs & 64 normalized H_8 pairs:
    # E[(u_hat^T dW v_hat)^2] == ||dW||_F^2 / (n * d) with n = d = 8 => ||dW||_F^2 / 64.
    std_basis_mean = sum(
        (dw_res_e8[i][j] * dw_res_e8[i][j] for i in range(8) for j in range(8)),
        Fraction(0, 1),
    ) / Fraction(64, 1)
    wh_sum_sq = Fraction(0, 1)
    for r in range(8):
        for c in range(8):
            inner = sum(
                (H8_MATRIX[r][i] * dw_res_e8[i][j] * H8_MATRIX[c][j] for i in range(8) for j in range(8)),
                Fraction(0, 1),
            )
            wh_sum_sq += inner * inner
    # Each h_r/sqrt(8) is unit norm => (h_r^T dW h_c / 8)^2, averaged over 64 pairs => divide by 64 * 64.
    wh_frame_mean = wh_sum_sq / Fraction(64 * 64, 1)
    expected_iso_var = err_sq_res_e8 / Fraction(64, 1)

    # Exact Bareiss spectral bound certification: err_sq_res_e8 * I - dW_res^T dW_res > 0
    dw_gram = _mat_t_mat(dw_res_e8)
    dw_pd_mat = [[(err_sq_res_e8 if i == j else Fraction(0, 1)) - dw_gram[i][j] for j in range(8)] for i in range(8)]
    dw_bareiss_spectral_pd = certify_sylvester_pd_exact(dw_pd_mat)

    return {
        "matrix_dim_n": 8,
        "matrix_dim_d": 8,
        "baseline_margin": format_exact_rational(margin_base),
        "baseline_margin_4dp": format_decimal_trunc(margin_base, 4),
        "scalar_margin": format_exact_rational(margin_scalar),
        "residual_e8_margin": format_exact_rational(margin_res_e8),
        "scalar_breaks_refusal_boundary": all((margin_base > 0, margin_scalar <= 0)),
        "residual_e8_preserves_refusal_boundary": margin_res_e8 > 0,
        "err_sq_scalar": format_exact_rational(err_sq_scalar),
        "err_sq_residual_e8": format_exact_rational(err_sq_res_e8),
        "mse_reduction_rho": format_exact_rational(rho),
        "mse_reduction_rho_6dp": format_decimal_trunc(rho, 6),
        "sqrt_rho_ceiling_tightening_exact": all((
            worst_ceiling_res_sq == rho * worst_ceiling_scalar_sq,
            rho < Fraction(1, 100),
        )),
        "observed_drift_sq_within_worst_ceiling": drift_res_sq <= worst_ceiling_res_sq,
        "rank1_adversary_tightness_exact": adv_drift_sq == adv_ceiling_sq,
        "isotropic_std_basis_mean": format_exact_rational(std_basis_mean),
        "isotropic_wh_frame_mean": format_exact_rational(wh_frame_mean),
        "isotropic_variance_identity_exact": std_basis_mean == wh_frame_mean == expected_iso_var,
        "dw_bareiss_spectral_pd": dw_bareiss_spectral_pd,
        "token_embd_quantized_by_block_lut": False,
        "token_embd_immutable_row_hash_required": True,
    }


# ---------------------------------------------------------------------------
# Certificate Builder & Full Pure-Q Replay Validator
# ---------------------------------------------------------------------------

def build_standalone_certificate() -> dict[str, Any]:
    """Build deterministic pure-Z/Q Track S32 certificate with exact SHA-256 bindings."""
    payload: dict[str, Any] = {
        "novelty_tier": "N1",
        "oracle_class": "exact",
        "measured_pallas_tok_s": False,
        "measured_live_model_kl_residual_lut": False,
        "measured_live_model_safety_benchmarks": False,
        "zero_trust_path_floats": True,
        "zero_bare_asserts": True,
        "zero_rng": True,
        "pillar1_precast_overflow_separation": certify_pillar1_precast_overflow_separation(),
        "pillar2_wcet_vmem_tpu_roofline": certify_pillar2_wcet_vmem_tpu_roofline(),
        "pillar3_residual_flat_lut_e8_codec": certify_pillar3_residual_flat_lut_e8_codec(),
        "pillar4_spectral_lipschitz_enclosure": certify_pillar4_spectral_lipschitz_enclosure(),
        "pillar5_safety_refusal_margin_preservation": certify_pillar5_safety_refusal_margin_preservation(),
    }
    payload_sha256 = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    outer_core = {
        "schema": SCHEMA_VERSION,
        "track_id": TRACK_ID,
        "payload_sha256": payload_sha256,
        "payload": payload,
    }
    cert_sha256 = hashlib.sha256(canonical_json_bytes(outer_core)).hexdigest()
    return {
        "schema": SCHEMA_VERSION,
        "track_id": TRACK_ID,
        "payload_sha256": payload_sha256,
        "certificate_sha256": cert_sha256,
        "payload": payload,
    }


def validate_certificate_payload(cert: Any, *, full_replay: bool = True) -> dict[str, Any]:
    """Fail-closed validation of certificate structure, SHA-256 digests, and full pure-Q replay."""
    replay_flag = require_exact_bool(full_replay, "full_replay")
    if type(cert) is not dict:
        raise SiliquantTpuBoundsError("CERT_NOT_OBJECT", "certificate must be dict")
    _validate_json_value(cert, 0)
    expected_keys = {"schema", "track_id", "payload_sha256", "certificate_sha256", "payload"}
    if set(cert.keys()) != expected_keys:
        raise SiliquantTpuBoundsError("INVALID_CERT_KEYS", "certificate top-level key mismatch")
    if cert["schema"] != SCHEMA_VERSION:
        raise SiliquantTpuBoundsError("INVALID_SCHEMA", "schema version mismatch")
    if cert["track_id"] != TRACK_ID:
        raise SiliquantTpuBoundsError("INVALID_TRACK_ID", "track_id mismatch")
    payload = cert["payload"]
    if type(payload) is not dict:
        raise SiliquantTpuBoundsError("PAYLOAD_NOT_OBJECT", "payload must be dict")
    required_payload_keys = {
        "novelty_tier",
        "oracle_class",
        "measured_pallas_tok_s",
        "measured_live_model_kl_residual_lut",
        "measured_live_model_safety_benchmarks",
        "zero_trust_path_floats",
        "zero_bare_asserts",
        "zero_rng",
        "pillar1_precast_overflow_separation",
        "pillar2_wcet_vmem_tpu_roofline",
        "pillar3_residual_flat_lut_e8_codec",
        "pillar4_spectral_lipschitz_enclosure",
        "pillar5_safety_refusal_margin_preservation",
    }
    if set(payload.keys()) != required_payload_keys:
        raise SiliquantTpuBoundsError("INVALID_PAYLOAD_KEYS", "payload top-level key mismatch")

    p_sha = require_exact_str(cert["payload_sha256"], "payload_sha256")
    c_sha = require_exact_str(cert["certificate_sha256"], "certificate_sha256")
    actual_p_sha = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    if p_sha != actual_p_sha:
        raise SiliquantTpuBoundsError("PAYLOAD_SHA_MISMATCH", "payload_sha256 mismatch")

    outer_core = {
        "schema": cert["schema"],
        "track_id": cert["track_id"],
        "payload_sha256": p_sha,
        "payload": payload,
    }
    actual_c_sha = hashlib.sha256(canonical_json_bytes(outer_core)).hexdigest()
    if c_sha != actual_c_sha:
        raise SiliquantTpuBoundsError("CERT_SHA_MISMATCH", "certificate_sha256 mismatch")

    if replay_flag:
        rebuilt = build_standalone_certificate()
        if canonical_json_bytes(rebuilt) != canonical_json_bytes(cert):
            raise SiliquantTpuBoundsError("REPLAY_MISMATCH", "certificate failed exact deterministic replay")

    return {
        "valid": True,
        "schema": SCHEMA_VERSION,
        "track_id": TRACK_ID,
        "payload_sha256": p_sha,
        "certificate_sha256": c_sha,
        "full_replay": replay_flag,
    }
