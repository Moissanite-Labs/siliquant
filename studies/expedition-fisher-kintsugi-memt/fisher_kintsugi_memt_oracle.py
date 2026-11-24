#!/usr/bin/env python3
"""Track S42: Portals 39-42 Unified Exact-Operator Suite
(`lookatmyquant` x `Kintsugi` x `gbench` x `MemT-1.0-Pro`).

Pure Python 3 standard library over exact Z/Q (`int` and `fractions.Fraction`).
Zero floats, zero bare asserts, zero `.isdigit()`, zero mathematical RNG.
Certifies four exact mathematical pillars + Zone-B hardening:
  1. Exact AM-GM Fisher Water-Filling vs. Uniform-BPW Rate-Distortion Theorem
     (`lookatmyquant` x `siliquant` Track S30): KKT equalization, exact R = AM(w)/GM(w),
     >= 12x KL reduction at 0.000 extra bpw (`R = 263/16 = 16.4375x`), & strict convexity.
  2. Exact `Kintsugi` Task-Break Logit-Margin Inversion vs. Smooth Average KL
     (`CS-20` Rosen Bridge): both quadratic-Fisher metric AND exact dyadic Shannon KL
     proving Uniform-A achieves strictly lower average KL while inverting critical top-1
     margin whereas Kintsugi-B preserves all task margins.
  3. Exact `gbench` Open-Loop `P99` Order-Statistic Sample-Size Law (`CS-18` Floor):
     pure-integer sharp 99% boundary `n=458` FAILS vs `n=459` PASSES, `n=1099` cohort
     miss probability `< 1/60000` (`1/62651 < P < 1/62650`), & exact binomial order-stat
     brackets (keeping `100**1099` internal so every JSON int respects `MAX_INT_BITS`).
  4. Exact `MemT-1.0-Pro` `GDN-2` Rank-Capacity Obstruction & Channel-2B Persistent
     Latent-Memory Poisoning (`Track S31` Airlock Extension): exact 4/4 Hadamard retrieval,
     rank-4 universal Cauchy-Schwarz lower bound (`106/5 > 0`) + Parseval error (`212/5`)
     at L=5, persistent rank-1 latent poisoning flipping a benign query under frozen weights,
     and 4-channel airlock closing the legacy 3-channel blind spot.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import hmac
import json
import math
from typing import Any, Sequence

TRACK_ID = "TRACK-S42-FISHER-WATERFILLING-KINTSUGI-MARGINS"
SCHEMA_VERSION = "aletheia.fisher_kintsugi_memt.certificate.v1"

MAX_JSON_BYTES = 1_048_576
MAX_JSON_DEPTH = 32
MAX_INT_DIGITS = 1000
MAX_INT_BITS = 3321
_ASCII_DIGITS = frozenset("0123456789")
_HEX_LOWER = frozenset("0123456789abcdef")

_CERT_KEYS = frozenset({"schema", "track_id", "payload_sha256", "certificate_sha256", "payload"})
_PAYLOAD_KEYS = frozenset({
    "novelty_tier",
    "oracle_class",
    "prior_art_cs13_anchors",
    "measured_live_gemma4_lamq_weights",
    "measured_live_gbench_serving_endpoint",
    "measured_live_memt_554b_weights",
    "zero_trust_path_floats",
    "zero_bare_asserts",
    "zero_mathematical_rng",
    "pillar1_fisher_waterfilling_amgm",
    "pillar2_kintsugi_margin_vs_kl",
    "pillar3_gbench_p99_order_statistics",
    "pillar4_memt_gdn2_rank_and_channel2b",
})


class FisherKintsugiMemtError(ValueError):
    """Fail-closed domain/validation error for Track S42."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


# ---------------------------------------------------------------------------
# Strict Type, ASCII-Digit, Rational, Hex & Canonical JSON Primitives
# ---------------------------------------------------------------------------

def require_exact_int(
    val: Any,
    name: str,
    *,
    min_val: int | None = None,
    max_val: int | None = None,
) -> int:
    if type(val) is not int:
        raise FisherKintsugiMemtError("INVALID_INT", f"{name} must be exact int")
    if min_val is not None and val < min_val:
        raise FisherKintsugiMemtError("INT_BELOW_MIN", f"{name} must be >= {min_val}")
    if max_val is not None and val > max_val:
        raise FisherKintsugiMemtError("INT_ABOVE_MAX", f"{name} must be <= {max_val}")
    return val


def require_exact_bool(val: Any, name: str) -> bool:
    if type(val) is not bool:
        raise FisherKintsugiMemtError("INVALID_BOOL", f"{name} must be exact bool")
    return val


def require_exact_str(val: Any, name: str, *, non_empty: bool = True) -> str:
    if type(val) is not str:
        raise FisherKintsugiMemtError("INVALID_STR", f"{name} must be exact str")
    if non_empty and not val:
        raise FisherKintsugiMemtError("EMPTY_STR", f"{name} must be non-empty")
    try:
        val.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise FisherKintsugiMemtError("INVALID_UTF8_STR", f"{name} contains lone surrogate") from exc
    return val


def require_exact_fraction(
    val: Any,
    name: str,
    *,
    positive: bool = False,
    non_negative: bool = False,
) -> Fraction:
    if type(val) is not Fraction:
        raise FisherKintsugiMemtError("INVALID_FRACTION", f"{name} must be exact Fraction")
    if positive and val <= 0:
        raise FisherKintsugiMemtError("FRACTION_NOT_POSITIVE", f"{name} must be > 0")
    if non_negative and val < 0:
        raise FisherKintsugiMemtError("FRACTION_NEGATIVE", f"{name} must be >= 0")
    return val


def require_sha256_hex(val: Any, name: str) -> str:
    s = require_exact_str(val, name)
    if len(s) != 64:
        raise FisherKintsugiMemtError("INVALID_SHA256", f"{name} must be 64 lowercase hex chars")
    for ch in s:
        if ch not in _HEX_LOWER:
            raise FisherKintsugiMemtError("INVALID_SHA256", f"{name} contains non-hex char")
    return s


def _require_bounded_int(val: int, name: str) -> None:
    if val.bit_length() > MAX_INT_BITS:
        raise FisherKintsugiMemtError("INT_TOO_LARGE", f"{name} exceeds {MAX_INT_BITS}-bit limit")


def parse_exact_int_ascii(text: Any, name: str) -> int:
    """Parse strict ASCII signed integer (`-?[0-9]+`, no `.isdigit()`, no `-0`, no leading zeros)."""
    s = require_exact_str(text, name)
    neg = s.startswith("-")
    digits = s[1:] if neg else s
    if not digits:
        raise FisherKintsugiMemtError("INVALID_INT_TOKEN", f"{name} has no digits")
    if len(digits) > MAX_INT_DIGITS:
        raise FisherKintsugiMemtError("INT_TOKEN_TOO_LONG", f"{name} exceeds {MAX_INT_DIGITS} digits")
    if len(digits) > 1 and digits[0] == "0":
        raise FisherKintsugiMemtError("LEADING_ZERO_INT", f"{name} has leading zero")
    for ch in digits:
        if ch not in _ASCII_DIGITS:
            raise FisherKintsugiMemtError("NON_ASCII_DIGIT", f"{name} contains non-ASCII digit")
    if neg and digits == "0":
        raise FisherKintsugiMemtError("NEGATIVE_ZERO_INT", f"{name} cannot be -0")
    acc = 0
    for ch in digits:
        acc = acc * 10 + (ord(ch) - 48)
    res = -acc if neg else acc
    _require_bounded_int(res, name)
    return res


def parse_exact_rational(text: Any, name: str) -> Fraction:
    """Parse canonical exact rational string `p` or `p/q` (strict ASCII digits, `q > 1`, reduced)."""
    s = require_exact_str(text, name)
    if "/" not in s:
        return Fraction(parse_exact_int_ascii(s, name), 1)
    parts = s.split("/")
    if len(parts) != 2:
        raise FisherKintsugiMemtError("INVALID_RATIONAL", f"{name} has multiple slashes")
    num = parse_exact_int_ascii(parts[0], f"{name}.num")
    den = parse_exact_int_ascii(parts[1], f"{name}.den")
    if den <= 1:
        raise FisherKintsugiMemtError("NON_CANONICAL_DEN", f"{name} denominator must be > 1")
    if math.gcd(abs(num), den) != 1:
        raise FisherKintsugiMemtError("UNREDUCED_RATIONAL", f"{name} must be in lowest terms")
    return Fraction(num, den)


def format_exact_rational(q: Any) -> str:
    val = require_exact_fraction(q, "q")
    _require_bounded_int(val.numerator, "q.numerator")
    _require_bounded_int(val.denominator, "q.denominator")
    if val.denominator == 1:
        return str(val.numerator)
    return f"{val.numerator}/{val.denominator}"


def format_decimal_trunc(q: Any, places: Any) -> str:
    """Deterministic toward-zero decimal truncation of exact Fraction; never emits `-0.00` or `-0`."""
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


def validate_json_tree_iterative(obj: Any) -> None:
    """Iterative Zone-B JSON tree validation: depth cap, int bit cap, no floats/cycles/surrogates."""
    stack: list[tuple[Any, int]] = [(obj, 0)]
    while stack:
        val, depth = stack.pop()
        if depth > MAX_JSON_DEPTH:
            raise FisherKintsugiMemtError("JSON_TOO_DEEP", "JSON nesting depth exceeded")
        t = type(val)
        if t is bool or val is None:
            continue
        if t is int:
            _require_bounded_int(val, "json_int")
            continue
        if t is str:
            require_exact_str(val, "json_str", non_empty=False)
            continue
        if t is list:
            for item in val:
                stack.append((item, depth + 1))
            continue
        if t is dict:
            for k, v in val.items():
                require_exact_str(k, "json_key", non_empty=False)
                stack.append((v, depth + 1))
            continue
        raise FisherKintsugiMemtError("FORBIDDEN_JSON_TYPE", f"unsupported JSON type: {t.__name__}")


def canonical_json_bytes(obj: Any) -> bytes:
    validate_json_tree_iterative(obj)
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    if len(raw) > MAX_JSON_BYTES:
        raise FisherKintsugiMemtError("JSON_TOO_LARGE", "canonical JSON exceeds byte limit")
    return raw


def parse_strict_json_text(text: Any, *, require_object: bool = True) -> Any:
    s = require_exact_str(text, "json_text")
    if len(s.encode("utf-8")) > MAX_JSON_BYTES:
        raise FisherKintsugiMemtError("JSON_TOO_LARGE", "JSON text exceeds byte limit")

    def _reject_float(token: str) -> Any:
        raise FisherKintsugiMemtError("FLOAT_FORBIDDEN", f"floating-point JSON token forbidden: {token}")

    def _reject_const(token: str) -> Any:
        raise FisherKintsugiMemtError("NON_FINITE_FORBIDDEN", f"JSON constant forbidden: {token}")

    def _parse_json_int(token: str) -> int:
        return parse_exact_int_ascii(token, "json_int")

    def _no_dup_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in pairs:
            if k in out:
                raise FisherKintsugiMemtError("DUPLICATE_JSON_KEY", f"duplicate key: {k}")
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
    except FisherKintsugiMemtError:
        raise
    except (ValueError, RecursionError) as exc:
        raise FisherKintsugiMemtError("INVALID_JSON", f"malformed JSON: {type(exc).__name__}") from exc

    validate_json_tree_iterative(parsed)
    if require_object and type(parsed) is not dict:
        raise FisherKintsugiMemtError("JSON_ROOT_NOT_OBJECT", "JSON root must be an object")
    return parsed


# ---------------------------------------------------------------------------
# Exact Rational Linear Algebra Primitives (Square Det, Sylvester PD, Rank)
# ---------------------------------------------------------------------------

def _require_rect_frac_matrix(mat: Any, name: str) -> list[list[Fraction]]:
    if type(mat) not in (list, tuple) or not mat or len(mat) > 16:
        raise FisherKintsugiMemtError("INVALID_MATRIX", f"{name} must be non-empty matrix (<= 16 rows)")
    first_row = mat[0]
    if type(first_row) not in (list, tuple) or not first_row or len(first_row) > 16:
        raise FisherKintsugiMemtError("INVALID_MATRIX", f"{name}[0] must be non-empty row (<= 16 cols)")
    cols = len(first_row)
    out: list[list[Fraction]] = []
    for r_idx, row in enumerate(mat):
        if type(row) not in (list, tuple) or len(row) != cols:
            raise FisherKintsugiMemtError("RAGGED_MATRIX", f"{name}[{r_idx}] must have length {cols}")
        out.append([require_exact_fraction(v, f"{name}[{r_idx}][{c_idx}]") for c_idx, v in enumerate(row)])
    return out


def _require_square_frac_matrix(mat: Any, name: str) -> list[list[Fraction]]:
    a = _require_rect_frac_matrix(mat, name)
    if len(a) != len(a[0]):
        raise FisherKintsugiMemtError("NON_SQUARE_MATRIX", f"{name} must be square")
    return a


def dot_exact(u: Any, v: Any) -> Fraction:
    if type(u) not in (list, tuple) or type(v) not in (list, tuple) or not u:
        raise FisherKintsugiMemtError("INVALID_VECTOR", "dot_exact requires non-empty list/tuple vectors")
    if len(u) != len(v):
        raise FisherKintsugiMemtError("VECTOR_LENGTH_MISMATCH", "dot_exact requires equal vector lengths")
    u_f = [require_exact_fraction(a, f"u[{i}]") for i, a in enumerate(u)]
    v_f = [require_exact_fraction(b, f"v[{i}]") for i, b in enumerate(v)]
    return sum((a * b for a, b in zip(u_f, v_f)), Fraction(0, 1))


def mat_vec_exact(mat: Any, vec: Any) -> list[Fraction]:
    a = _require_rect_frac_matrix(mat, "mat")
    if type(vec) not in (list, tuple) or len(vec) != len(a[0]):
        raise FisherKintsugiMemtError("MATRIX_VECTOR_MISMATCH", f"vec must have length {len(a[0])}")
    v_f = [require_exact_fraction(v, f"vec[{j}]") for j, v in enumerate(vec)]
    return [sum((row[j] * v_f[j] for j in range(len(v_f))), Fraction(0, 1)) for row in a]


def bareiss_det_exact(mat: Any) -> Fraction:
    """Exact Bareiss determinant over Q for square matrices up to 16x16."""
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
        pivot = a[k][k]
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                a[i][j] = (a[i][j] * pivot - a[i][k] * a[k][j]) / prev_pivot
            a[i][k] = Fraction(0, 1)
        prev_pivot = pivot
    return a[n - 1][n - 1] * sign


def certify_sylvester_pd_exact(mat: Any) -> bool:
    """Certify exact symmetry and positive-definiteness via all n leading Bareiss principal minors > 0."""
    a = _require_square_frac_matrix(mat, "mat")
    n = len(a)
    for i in range(n):
        for j in range(i + 1, n):
            if a[i][j] != a[j][i]:
                raise FisherKintsugiMemtError("NON_SYMMETRIC_MATRIX", "matrix must be exact symmetric")
    for k in range(1, n + 1):
        sub = [[a[r][c] for c in range(k)] for r in range(k)]
        if bareiss_det_exact(sub) <= 0:
            return False
    return True


def exact_matrix_rank_q(mat: Any) -> int:
    """Exact Gaussian elimination rank over Q for any rectangular matrix up to 16x16."""
    a = _require_rect_frac_matrix(mat, "mat")
    rows = len(a)
    cols = len(a[0])
    rank = 0
    for c in range(cols):
        if rank >= rows:
            break
        piv: int | None = None
        for r in range(rank, rows):
            if a[r][c] != 0:
                piv = r
                break
        if piv is None:
            continue
        if piv != rank:
            a[rank], a[piv] = a[piv], a[rank]
        p_val = a[rank][c]
        for r in range(rank + 1, rows):
            if a[r][c] != 0:
                fac = a[r][c] / p_val
                for j in range(c, cols):
                    a[r][j] = a[r][j] - fac * a[rank][j]
        rank += 1
    return rank


def _pow2_exact(exp: Any, name: str) -> Fraction:
    e = require_exact_int(exp, name, min_val=-256, max_val=256)
    if e >= 0:
        return Fraction(1 << e, 1)
    return Fraction(1, 1 << (-e))


# ---------------------------------------------------------------------------
# Pillar 1: Exact AM-GM Fisher Water-Filling vs. Uniform-BPW Rate-Distortion
# ---------------------------------------------------------------------------

def fisher_waterfilling_analysis(
    sensitivity_exponents: Any,
    alpha: Any,
    b_bar: Any,
    perturbations: Any = (),
) -> dict[str, Any]:
    """Certify exact KKT/AM-GM Fisher water-filling for K_i(b_i) = 2^{e_i - alpha*b_i}."""
    if type(sensitivity_exponents) not in (list, tuple) or not sensitivity_exponents or len(sensitivity_exponents) > 16:
        raise FisherKintsugiMemtError("INVALID_EXPONENTS", "sensitivity_exponents must be non-empty list/tuple (<=16)")
    es = [require_exact_int(e, f"e[{i}]", min_val=-32, max_val=32) for i, e in enumerate(sensitivity_exponents)]
    m = len(es)
    alpha_q = require_exact_fraction(alpha, "alpha", positive=True)
    b_bar_q = require_exact_fraction(b_bar, "b_bar", positive=True)

    e_sum = sum(es)
    if e_sum % m != 0:
        raise FisherKintsugiMemtError("NON_INTEGER_MEAN_EXP", "mean sensitivity exponent must be exact integer")
    e_bar = e_sum // m

    ab_prod = alpha_q * b_bar_q
    if ab_prod.denominator != 1:
        raise FisherKintsugiMemtError("NON_INTEGER_RATE_EXPONENT", "alpha * b_bar must be exact integer")
    ab_int = ab_prod.numerator

    weights = [_pow2_exact(e, "e_i") for e in es]
    am_w = sum(weights, Fraction(0, 1)) / Fraction(m, 1)
    gm_w = _pow2_exact(e_bar, "e_bar")

    b_star = [b_bar_q + Fraction(e - e_bar, 1) / alpha_q for e in es]
    mean_b_star = sum(b_star, Fraction(0, 1)) / Fraction(m, 1)
    all_b_positive = all(b > 0 for b in b_star)

    k_uniform_rows = [_pow2_exact(e - ab_int, "k_unif_exp") for e in es]
    k_uniform_total = sum(k_uniform_rows, Fraction(0, 1))

    k_eq = _pow2_exact(e_bar - ab_int, "k_eq_exp")
    k_fisher_rows: list[Fraction] = []
    for e, b_i in zip(es, b_star):
        exp_i = Fraction(e, 1) - alpha_q * b_i
        k_fisher_rows.append(_pow2_exact(exp_i.numerator, "k_fish_exp"))
    k_fisher_total = sum(k_fisher_rows, Fraction(0, 1))

    improvement_ratio = k_uniform_total / k_fisher_total
    am_over_gm = am_w / gm_w

    if type(perturbations) not in (list, tuple):
        raise FisherKintsugiMemtError("INVALID_PERTURBATIONS", "perturbations must be list/tuple")
    pert_records: list[dict[str, Any]] = []
    for p_idx, item in enumerate(perturbations):
        if type(item) not in (list, tuple) or len(item) != 2:
            raise FisherKintsugiMemtError("INVALID_PERTURBATION", f"perturbation[{p_idx}] must be (label, delta)")
        label = require_exact_str(item[0], f"perturbation[{p_idx}].label")
        delta_raw = item[1]
        if type(delta_raw) not in (list, tuple) or len(delta_raw) != m:
            raise FisherKintsugiMemtError("INVALID_PERTURBATION", f"perturbation[{p_idx}] delta must have length {m}")
        delta = [require_exact_fraction(d, f"delta[{i}]") for i, d in enumerate(delta_raw)]
        if sum(delta, Fraction(0, 1)) != 0:
            raise FisherKintsugiMemtError("PERTURBATION_NOT_ZERO_SUM", f"perturbation {label} must sum to 0")
        if not any(d != 0 for d in delta):
            raise FisherKintsugiMemtError("TRIVIAL_PERTURBATION", f"perturbation {label} must be nonzero")
        b_pert = [b_i + d_i for b_i, d_i in zip(b_star, delta)]
        if not all(bp > 0 for bp in b_pert):
            raise FisherKintsugiMemtError("INFEASIBLE_PERTURBATION", f"perturbation {label} produces non-positive b_i")
        k_pert_rows: list[Fraction] = []
        for e, bp in zip(es, b_pert):
            exp_p = Fraction(e, 1) - alpha_q * bp
            if exp_p.denominator != 1:
                raise FisherKintsugiMemtError("NON_INTEGER_PERT_EXPONENT", f"alpha * delta must be integer in {label}")
            k_pert_rows.append(_pow2_exact(exp_p.numerator, "k_pert_exp"))
        k_pert_total = sum(k_pert_rows, Fraction(0, 1))
        excess = k_pert_total - k_fisher_total
        pert_records.append({
            "label": label,
            "delta": [format_exact_rational(d) for d in delta],
            "k_perturbed_total": format_exact_rational(k_pert_total),
            "excess_kl": format_exact_rational(excess),
            "ratio_to_optimal": format_exact_rational(k_pert_total / k_fisher_total),
            "strictly_worse_than_fisher": excess > 0,
        })

    return {
        "row_count_m": m,
        "alpha": format_exact_rational(alpha_q),
        "b_bar": format_exact_rational(b_bar_q),
        "sensitivity_exponents": es,
        "mean_exponent": e_bar,
        "am_weights": format_exact_rational(am_w),
        "gm_weights": format_exact_rational(gm_w),
        "b_star": [format_exact_rational(b) for b in b_star],
        "extra_average_bpw": format_exact_rational(mean_b_star - b_bar_q),
        "all_b_star_positive": all_b_positive,
        "k_equalized_per_row": format_exact_rational(k_eq),
        "kkt_equalization_exact": all(k_r == k_eq for k_r in k_fisher_rows),
        "k_uniform_total": format_exact_rational(k_uniform_total),
        "k_fisher_total": format_exact_rational(k_fisher_total),
        "improvement_ratio": format_exact_rational(improvement_ratio),
        "improvement_ratio_4dp": format_decimal_trunc(improvement_ratio, 4),
        "am_gm_identity_exact": improvement_ratio == am_over_gm,
        "perturbation_witnesses": pert_records,
        "all_perturbations_strictly_worse": all(r["strictly_worse_than_fisher"] for r in pert_records),
    }


def build_pillar1_fisher_waterfilling() -> dict[str, Any]:
    alpha = Fraction(2, 1)
    b_bar = Fraction(4, 1)
    unif = fisher_waterfilling_analysis([0, 0, 0, 0, 0, 0, 0, 0], alpha, b_bar)
    mild = fisher_waterfilling_analysis([4, 2, 0, 0, 0, 0, -2, -4], alpha, b_bar)
    heavy = fisher_waterfilling_analysis(
        [7, -1, -1, -1, -1, -1, -1, -1],
        alpha,
        b_bar,
        perturbations=[
            (
                "pair_half_bit",
                [Fraction(1, 2), Fraction(-1, 2)] + [Fraction(0, 1)] * 6,
            ),
            (
                "pair_one_bit",
                [Fraction(1, 1), Fraction(-1, 1)] + [Fraction(0, 1)] * 6,
            ),
            (
                "multi_row_asymmetric",
                [Fraction(1, 1), Fraction(1, 2), Fraction(-1, 2), Fraction(-1, 1)] + [Fraction(0, 1)] * 4,
            ),
        ],
    )
    r_unif = parse_exact_rational(unif["improvement_ratio"], "r_unif")
    r_mild = parse_exact_rational(mild["improvement_ratio"], "r_mild")
    r_heavy = parse_exact_rational(heavy["improvement_ratio"], "r_heavy")
    return {
        "uniform_control": unif,
        "mild_skew": mild,
        "heavy_outlier_gemma4_scale": heavy,
        "uniform_ratio_equals_one": r_unif == 1,
        "mild_ratio_exceeds_two": r_mild > 2,
        "heavy_ratio_exceeds_twelve": r_heavy > 12,
        "strict_hierarchy_certified": all((r_unif == 1, r_mild > 2, r_heavy > 12, r_heavy > r_mild > r_unif)),
    }


# ---------------------------------------------------------------------------
# Pillar 2: Exact `Kintsugi` Task-Break Logit-Margin Inversion vs. Average KL
# ---------------------------------------------------------------------------

def _require_prob_simplex(p: Any, name: str) -> list[Fraction]:
    if type(p) not in (list, tuple) or len(p) < 2 or len(p) > 16:
        raise FisherKintsugiMemtError("INVALID_DISTRIBUTION", f"{name} must have length in [2, 16]")
    probs = [require_exact_fraction(v, f"{name}[{i}]", positive=True) for i, v in enumerate(p)]
    if sum(probs, Fraction(0, 1)) != 1:
        raise FisherKintsugiMemtError("DISTRIBUTION_NOT_UNIT_SUM", f"{name} must sum to exact 1")
    return probs


def quadratic_fisher_kl_exact(p: Any, delta: Any) -> Fraction:
    """Exact local softmax Fisher KL proxy: (1/2) delta^T (diag(p) - p p^T) delta."""
    probs = _require_prob_simplex(p, "p")
    if type(delta) not in (list, tuple) or len(delta) != len(probs):
        raise FisherKintsugiMemtError("INVALID_DELTA", f"delta must have length {len(probs)}")
    d_vec = [require_exact_fraction(v, f"delta[{i}]") for i, v in enumerate(delta)]
    second_moment = sum((pi * di * di for pi, di in zip(probs, d_vec)), Fraction(0, 1))
    first_moment = sum((pi * di for pi, di in zip(probs, d_vec)), Fraction(0, 1))
    return (second_moment - first_moment * first_moment) / Fraction(2, 1)


def exact_dyadic_shannon_kl_bits(p: Any, q: Any) -> tuple[Fraction, list[int]]:
    """Exact Shannon KL sum_j p_j log_2(p_j / q_j) in Q for strictly dyadic ratios p_j / q_j = 2^{k_j}."""
    p_vec = _require_prob_simplex(p, "p")
    q_vec = _require_prob_simplex(q, "q")
    if len(p_vec) != len(q_vec):
        raise FisherKintsugiMemtError("DISTRIBUTION_LENGTH_MISMATCH", "p and q must have equal length")
    ks: list[int] = []
    total = Fraction(0, 1)
    for idx, (pj, qj) in enumerate(zip(p_vec, q_vec)):
        ratio = pj / qj
        if ratio >= 1:
            n = ratio.numerator
            if ratio.denominator != 1 or (n & (n - 1)) != 0:
                raise FisherKintsugiMemtError("NON_DYADIC_RATIO", f"p[{idx}]/q[{idx}] is not a power of 2")
            k = n.bit_length() - 1
        else:
            d = ratio.denominator
            if ratio.numerator != 1 or (d & (d - 1)) != 0:
                raise FisherKintsugiMemtError("NON_DYADIC_RATIO", f"p[{idx}]/q[{idx}] is not a power of 2")
            k = -(d.bit_length() - 1)
        ks.append(k)
        total += pj * Fraction(k, 1)
    return total, ks


def _top1_margin_gold0(scores: Sequence[Fraction]) -> tuple[int, Fraction]:
    best_idx = 0
    best_val = scores[0]
    for idx in range(1, len(scores)):
        if scores[idx] > best_val:
            best_val = scores[idx]
            best_idx = idx
    runner_up = scores[1]
    for idx in range(2, len(scores)):
        if scores[idx] > runner_up:
            runner_up = scores[idx]
    return best_idx, scores[0] - runner_up


def build_pillar2_kintsugi_margin_vs_kl() -> dict[str, Any]:
    # Part 2A: Quadratic-Fisher metric witness on 4-token prompt (V=4)
    p_ref = [Fraction(1, 2), Fraction(1, 4), Fraction(1, 8), Fraction(1, 8)]
    f_mat = [
        [(p_ref[i] if i == j else Fraction(0, 1)) - p_ref[i] * p_ref[j] for j in range(4)]
        for i in range(4)
    ]
    ones_4 = [Fraction(1, 1)] * 4
    f_ones = mat_vec_exact(f_mat, ones_4)
    f_shift_invariant = all(x == 0 for x in f_ones)
    f_rank = exact_matrix_rank_q(f_mat)
    f_3x3 = [[f_mat[i][j] for j in range(3)] for i in range(3)]
    f_3x3_pd = certify_sylvester_pd_exact(f_3x3)

    z_routine = [Fraction(4, 1), Fraction(0, 1), Fraction(-2, 1), Fraction(-2, 1)]
    z_critical = [Fraction(1, 2), Fraction(0, 1), Fraction(-2, 1), Fraction(-2, 1)]
    base_tokens = [z_routine, z_routine, z_routine, z_critical]

    zero_d = [Fraction(0, 1)] * 4
    delta_a = [zero_d, zero_d, zero_d, [Fraction(-1, 2), Fraction(1, 2), Fraction(0, 1), Fraction(0, 1)]]
    delta_b = [
        [Fraction(-1, 1), Fraction(0, 1), Fraction(1, 1), Fraction(0, 1)],
        [Fraction(-1, 1), Fraction(0, 1), Fraction(1, 1), Fraction(0, 1)],
        [Fraction(-1, 1), Fraction(0, 1), Fraction(1, 1), Fraction(0, 1)],
        [Fraction(-1, 8), Fraction(1, 8), Fraction(0, 1), Fraction(0, 1)],
    ]

    def _eval_fisher_quantizer(deltas: list[list[Fraction]]) -> dict[str, Any]:
        kls: list[Fraction] = []
        margins: list[Fraction] = []
        top1s: list[int] = []
        for z_base, d_tok in zip(base_tokens, deltas):
            kls.append(quadratic_fisher_kl_exact(p_ref, d_tok))
            z_q = [zb + dt for zb, dt in zip(z_base, d_tok)]
            top_idx, margin = _top1_margin_gold0(z_q)
            top1s.append(top_idx)
            margins.append(margin)
        avg_kl = sum(kls, Fraction(0, 1)) / Fraction(4, 1)
        min_m = min(margins)
        return {
            "per_token_fisher_kl": [format_exact_rational(k) for k in kls],
            "average_fisher_kl": format_exact_rational(avg_kl),
            "per_token_margins": [format_exact_rational(m) for m in margins],
            "per_token_top1": top1s,
            "critical_token_margin": format_exact_rational(margins[3]),
            "min_margin": format_exact_rational(min_m),
            "task_pass": all((all(t == 0 for t in top1s), min_m > 0)),
        }

    f_a = _eval_fisher_quantizer(delta_a)
    f_b = _eval_fisher_quantizer(delta_b)
    avg_fa = parse_exact_rational(f_a["average_fisher_kl"], "avg_fa")
    avg_fb = parse_exact_rational(f_b["average_fisher_kl"], "avg_fb")
    fisher_ratio = avg_fb / avg_fa

    # Part 2B: Exact dyadic Shannon KL witness (ZERO Taylor approximation, pure Z/Q log_2)
    p_rout = [Fraction(3, 4), Fraction(1, 8), Fraction(1, 16), Fraction(1, 16)]
    p_crit = [Fraction(1, 2), Fraction(1, 4), Fraction(1, 4)]
    q_a_crit = [Fraction(1, 4), Fraction(1, 2), Fraction(1, 4)]
    q_b_rout = [Fraction(3, 8), Fraction(1, 4), Fraction(1, 4), Fraction(1, 8)]

    kl_a_rout, ks_a_rout = exact_dyadic_shannon_kl_bits(p_rout, p_rout)
    kl_a_crit, ks_a_crit = exact_dyadic_shannon_kl_bits(p_crit, q_a_crit)
    kl_b_rout, ks_b_rout = exact_dyadic_shannon_kl_bits(p_rout, q_b_rout)
    kl_b_crit, ks_b_crit = exact_dyadic_shannon_kl_bits(p_crit, p_crit)

    avg_shannon_a = (Fraction(3, 1) * kl_a_rout + kl_a_crit) / Fraction(4, 1)
    avg_shannon_b = (Fraction(3, 1) * kl_b_rout + kl_b_crit) / Fraction(4, 1)
    shannon_ratio = avg_shannon_b / avg_shannon_a

    top_a_rout, m_a_rout = _top1_margin_gold0(p_rout)
    top_a_crit, m_a_crit = _top1_margin_gold0(q_a_crit)
    top_b_rout, m_b_rout = _top1_margin_gold0(q_b_rout)
    top_b_crit, m_b_crit = _top1_margin_gold0(p_crit)

    dyadic_a_pass = all((top_a_rout == 0, top_a_crit == 0, m_a_rout > 0, m_a_crit > 0))
    dyadic_b_pass = all((top_b_rout == 0, top_b_crit == 0, m_b_rout > 0, m_b_crit > 0))

    return {
        "quadratic_fisher_witness": {
            "prompt_tokens": 4,
            "vocab_size": 4,
            "fisher_shift_invariant_nullspace": f_shift_invariant,
            "fisher_matrix_rank": f_rank,
            "fisher_3x3_block_pd": f_3x3_pd,
            "uniform_a": f_a,
            "kintsugi_b": f_b,
            "kintsugi_over_uniform_kl_ratio": format_exact_rational(fisher_ratio),
            "kintsugi_over_uniform_kl_ratio_4dp": format_decimal_trunc(fisher_ratio, 4),
            "uniform_lower_avg_kl_yet_breaks_task": all((
                avg_fa < avg_fb,
                not f_a["task_pass"],
                f_b["task_pass"],
            )),
        },
        "exact_dyadic_shannon_witness": {
            "prompt_tokens": 4,
            "uniform_a_routine_kl_bits": format_exact_rational(kl_a_rout),
            "uniform_a_critical_kl_bits": format_exact_rational(kl_a_crit),
            "uniform_a_critical_log2_exponents": ks_a_crit,
            "uniform_a_avg_kl_bits": format_exact_rational(avg_shannon_a),
            "uniform_a_critical_prob_margin": format_exact_rational(m_a_crit),
            "uniform_a_critical_top1": top_a_crit,
            "uniform_a_task_pass": dyadic_a_pass,
            "kintsugi_b_routine_kl_bits": format_exact_rational(kl_b_rout),
            "kintsugi_b_routine_log2_exponents": ks_b_rout,
            "kintsugi_b_critical_kl_bits": format_exact_rational(kl_b_crit),
            "kintsugi_b_avg_kl_bits": format_exact_rational(avg_shannon_b),
            "kintsugi_b_routine_prob_margin": format_exact_rational(m_b_rout),
            "kintsugi_b_critical_prob_margin": format_exact_rational(m_b_crit),
            "kintsugi_b_task_pass": dyadic_b_pass,
            "kintsugi_over_uniform_kl_ratio": format_exact_rational(shannon_ratio),
            "gibbs_strict_positivity_certified": all((
                kl_a_rout == 0,
                kl_b_crit == 0,
                kl_a_crit > 0,
                kl_b_rout > 0,
            )),
            "uniform_lower_avg_kl_yet_breaks_task": all((
                avg_shannon_a < avg_shannon_b,
                not dyadic_a_pass,
                dyadic_b_pass,
            )),
        },
        "cs20_average_kl_blindness_certified": all((
            avg_fa < avg_fb,
            not f_a["task_pass"],
            f_b["task_pass"],
            avg_shannon_a < avg_shannon_b,
            not dyadic_a_pass,
            dyadic_b_pass,
        )),
    }


# ---------------------------------------------------------------------------
# Pillar 3: Exact `gbench` Open-Loop `P99` Order-Statistic Sample-Size Law
# ---------------------------------------------------------------------------

def _ppm_bracket(num: int, den: int) -> tuple[int, int]:
    scaled = num * 1_000_000
    lo = scaled // den
    hi = lo if scaled % den == 0 else lo + 1
    return lo, hi


def _binomial_tail_num(n: int, r: int) -> int:
    """Exact numerator sum_{k=0}^{r} C(n, k) * 99^{n-k} over denominator 100^n."""
    total = 0
    for k in range(r + 1):
        total += math.comb(n, k) * (99 ** (n - k))
    return total


def gbench_p99_cohort_cert(n: Any) -> dict[str, Any]:
    """Exact pure-integer P99 order-statistic certification for cohort size n (CS-18 floor)."""
    n_int = require_exact_int(n, "n", min_val=1, max_val=2000)
    num0 = 99 ** n_int
    den = 100 ** n_int
    inv_floor = den // num0
    inv_ceil = inv_floor if den % num0 == 0 else inv_floor + 1

    p0_lo, p0_hi = _ppm_bracket(num0, den)
    t1_num = _binomial_tail_num(n_int, 1)
    t2_num = _binomial_tail_num(n_int, 2)
    t4_num = _binomial_tail_num(n_int, 4)

    t1_lo, t1_hi = _ppm_bracket(t1_num, den)
    t2_lo, t2_hi = _ppm_bracket(t2_num, den)
    ge1_lo, ge1_hi = _ppm_bracket(den - num0, den)
    ge5_lo, ge5_hi = _ppm_bracket(den - t4_num, den)

    pass_99pct = (100 * num0) <= den
    pass_60k = (60_000 * num0) < den
    gbench_certified = all((n_int >= 1099, pass_60k))

    return {
        "n": n_int,
        "den_bit_length": den.bit_length(),
        "den_decimal_digits": len(str(den)),
        "den_exceeds_max_json_int_bits": den.bit_length() > MAX_INT_BITS,
        "pass_99pct_upper_tail": pass_99pct,
        "miss_prob_below_1_over_60000": pass_60k,
        "reciprocal_miss_prob_floor": inv_floor,
        "reciprocal_miss_prob_ceil": inv_ceil,
        "p_zero_exceedances_ppm_lo": p0_lo,
        "p_zero_exceedances_ppm_hi": p0_hi,
        "p_le_1_exceedance_ppm_lo": t1_lo,
        "p_le_1_exceedance_ppm_hi": t1_hi,
        "p_le_2_exceedances_ppm_lo": t2_lo,
        "p_le_2_exceedances_ppm_hi": t2_hi,
        "p_ge_1_exceedance_ppm_lo": ge1_lo,
        "p_ge_1_exceedance_ppm_hi": ge1_hi,
        "p_ge_5_exceedances_ppm_lo": ge5_lo,
        "p_ge_5_exceedances_ppm_hi": ge5_hi,
        "cs18_upper_tail_status": "CERTIFIED_UPPER_TAIL" if pass_99pct else "UNCERTIFIED",
        "cs18_gbench_slo_status": "CERTIFIED" if gbench_certified else "UNCERTIFIED",
    }


def build_pillar3_gbench_p99_order_statistics() -> dict[str, Any]:
    c100 = gbench_p99_cohort_cert(100)
    c458 = gbench_p99_cohort_cert(458)
    c459 = gbench_p99_cohort_cert(459)
    c1099 = gbench_p99_cohort_cert(1099)

    # Exact two-sided order-statistic coverage at n=1099: P(4 <= E_1099 <= 18) > 95%
    n_g = 1099
    den_g = 100 ** n_g
    mid_num = sum(math.comb(n_g, k) * (99 ** (n_g - k)) for k in range(4, 19))
    cov_lo, cov_hi = _ppm_bracket(mid_num, den_g)
    cov_gt_95pct = (100 * mid_num) > (95 * den_g)

    return {
        "sharp_99pct_threshold_n": 459,
        "gbench_slo_cohort_threshold_n": 1099,
        "cohort_n100": c100,
        "cohort_n458": c458,
        "cohort_n459": c459,
        "cohort_n1099": c1099,
        "sharp_boundary_458_fails_459_passes": all((
            not c458["pass_99pct_upper_tail"],
            c459["pass_99pct_upper_tail"],
        )),
        "n1099_two_sided_order_stat_bracket": {
            "exceedance_range_min": 4,
            "exceedance_range_max": 18,
            "order_stat_lower_rank": 1081,
            "order_stat_upper_rank": 1096,
            "coverage_ppm_lo": cov_lo,
            "coverage_ppm_hi": cov_hi,
            "coverage_exceeds_95pct": cov_gt_95pct,
        },
        "cs18_uncertified_floor_enforced": all((
            c100["cs18_gbench_slo_status"] == "UNCERTIFIED",
            c458["cs18_gbench_slo_status"] == "UNCERTIFIED",
            c459["cs18_gbench_slo_status"] == "UNCERTIFIED",
            c1099["cs18_gbench_slo_status"] == "CERTIFIED",
            c1099["miss_prob_below_1_over_60000"],
            cov_gt_95pct,
        )),
    }


# ---------------------------------------------------------------------------
# Pillar 4: `MemT-1.0-Pro` `GDN-2` Rank-Capacity Obstruction & Channel-2B
# ---------------------------------------------------------------------------

def gdn2_delta_step_exact(
    state: Any,
    key: Any,
    value: Any,
    *,
    alpha: Any = Fraction(1, 1),
    beta: Any = Fraction(1, 1),
) -> list[list[Fraction]]:
    """Exact rational GDN-2 rank-1 delta-rule update: S_{t+1} = S_t (I - alpha k k^T) + beta v k^T."""
    s_mat = _require_square_frac_matrix(state, "state")
    d = len(s_mat)
    if type(key) not in (list, tuple) or len(key) != d:
        raise FisherKintsugiMemtError("INVALID_KEY", f"key must have length {d}")
    if type(value) not in (list, tuple) or len(value) != d:
        raise FisherKintsugiMemtError("INVALID_VALUE", f"value must have length {d}")
    k_vec = [require_exact_fraction(x, f"key[{i}]") for i, x in enumerate(key)]
    v_vec = [require_exact_fraction(x, f"value[{i}]") for i, x in enumerate(value)]
    a_q = require_exact_fraction(alpha, "alpha", non_negative=True)
    b_q = require_exact_fraction(beta, "beta", non_negative=True)
    sk = mat_vec_exact(s_mat, k_vec)
    return [
        [s_mat[i][j] - a_q * sk[i] * k_vec[j] + b_q * v_vec[i] * k_vec[j] for j in range(d)]
        for i in range(d)
    ]


def _gdn2_transpose_step_exact(
    m_mat: list[list[Fraction]],
    key: list[Fraction],
    value: list[Fraction],
) -> list[list[Fraction]]:
    """Exact transpose form M_{t+1} = (I - k k^T) M_t + k v^T matching S_{t+1}^T."""
    d = len(m_mat)
    kt_m = [sum((key[r] * m_mat[r][c] for r in range(d)), Fraction(0, 1)) for c in range(d)]
    return [
        [m_mat[i][j] - key[i] * kt_m[j] + key[i] * value[j] for j in range(d)]
        for i in range(d)
    ]


def _transpose_mat(mat: list[list[Fraction]]) -> list[list[Fraction]]:
    rows = len(mat)
    cols = len(mat[0])
    return [[mat[r][c] for r in range(rows)] for c in range(cols)]


def _vec_diff_norm_sq(u: Sequence[Fraction], v: Sequence[Fraction]) -> Fraction:
    return sum(((a - b) * (a - b) for a, b in zip(u, v)), Fraction(0, 1))


def _mat_frob_norm_sq(a: list[list[Fraction]], b: list[list[Fraction]]) -> Fraction:
    return sum(
        ((a[i][j] - b[i][j]) * (a[i][j] - b[i][j]) for i in range(len(a)) for j in range(len(a[0]))),
        Fraction(0, 1),
    )


def evaluate_four_channel_airlock(record: Any) -> dict[str, Any]:
    """Fail-closed 4-channel persistence & provenance airlock (extending Track S31 with Channel-2B)."""
    if type(record) is not dict:
        raise FisherKintsugiMemtError("INVALID_AIRLOCK_RECORD", "airlock record must be dict")
    req_keys = frozenset({
        "token_embd_hash_match",
        "block_dpi_ratio",
        "max_block_dpi_ratio",
        "kv_cross_model_reuse",
        "kv_residual_evicted_mentions",
        "latent_state_hash_match",
        "latent_delta_norm_sq",
        "max_latent_delta_norm_sq",
        "latent_cross_session_carryover",
        "corpus_provenance_verified",
        "cot_mode_transfer_gap",
        "max_cot_mode_transfer_gap",
    })
    if frozenset(record.keys()) != req_keys:
        raise FisherKintsugiMemtError("INVALID_AIRLOCK_KEYS", "airlock record must have exact 4-channel keys")

    embd_ok = require_exact_bool(record["token_embd_hash_match"], "token_embd_hash_match")
    dpi = parse_exact_rational(record["block_dpi_ratio"], "block_dpi_ratio")
    max_dpi = parse_exact_rational(record["max_block_dpi_ratio"], "max_block_dpi_ratio")
    kv_reuse = require_exact_bool(record["kv_cross_model_reuse"], "kv_cross_model_reuse")
    kv_res = require_exact_int(record["kv_residual_evicted_mentions"], "kv_residual_evicted_mentions", min_val=0)
    lat_hash_ok = require_exact_bool(record["latent_state_hash_match"], "latent_state_hash_match")
    lat_norm = parse_exact_rational(record["latent_delta_norm_sq"], "latent_delta_norm_sq")
    max_lat_norm = parse_exact_rational(record["max_latent_delta_norm_sq"], "max_latent_delta_norm_sq")
    lat_carry = require_exact_bool(record["latent_cross_session_carryover"], "latent_cross_session_carryover")
    corp_ok = require_exact_bool(record["corpus_provenance_verified"], "corpus_provenance_verified")
    cot_gap = parse_exact_rational(record["cot_mode_transfer_gap"], "cot_mode_transfer_gap")
    max_cot_gap = parse_exact_rational(record["max_cot_mode_transfer_gap"], "max_cot_mode_transfer_gap")

    violations_3ch: list[str] = []
    if not embd_ok or dpi > max_dpi:
        violations_3ch.append("CHANNEL_1_WEIGHT_INTEGRITY")
    if kv_reuse or kv_res > 0:
        violations_3ch.append("CHANNEL_2A_KV_PERSISTENCE")
    if not corp_ok or cot_gap > max_cot_gap:
        violations_3ch.append("CHANNEL_3_DATA_PROVENANCE")

    violations_4ch = list(violations_3ch)
    if not lat_hash_ok or lat_norm > max_lat_norm or lat_carry:
        violations_4ch.append("CHANNEL_2B_LATENT_MEMORY_POISONING")

    return {
        "legacy_3ch_decision": "ALLOW" if not violations_3ch else "DENY",
        "airlock_4ch_decision": "ALLOW" if not violations_4ch else "DENY",
        "violations_3ch": violations_3ch,
        "violations_4ch": violations_4ch,
    }


def build_pillar4_memt_gdn2_rank_and_channel2b() -> dict[str, Any]:
    half = Fraction(1, 2)
    keys4 = [
        [half, half, half, half],
        [half, -half, half, -half],
        [half, half, -half, -half],
        [half, -half, -half, half],
    ]
    vals4 = [
        [Fraction(2, 1), Fraction(0, 1), Fraction(0, 1), Fraction(0, 1)],
        [Fraction(0, 1), Fraction(2, 1), Fraction(0, 1), Fraction(0, 1)],
        [Fraction(0, 1), Fraction(0, 1), Fraction(2, 1), Fraction(0, 1)],
        [Fraction(0, 1), Fraction(0, 1), Fraction(0, 1), Fraction(2, 1)],
    ]
    # Certify exact orthonormality K_4^T K_4 = I_4
    gram_is_identity = True
    for i in range(4):
        for j in range(4):
            expected = Fraction(1, 1) if i == j else Fraction(0, 1)
            if dot_exact(keys4[i], keys4[j]) != expected:
                gram_is_identity = False

    s_mat = [[Fraction(0, 1)] * 4 for _ in range(4)]
    m_mat = [[Fraction(0, 1)] * 4 for _ in range(4)]
    for k_vec, v_vec in zip(keys4, vals4):
        s_mat = gdn2_delta_step_exact(s_mat, k_vec, v_vec)
        m_mat = _gdn2_transpose_step_exact(m_mat, k_vec, v_vec)
    transpose_duality_exact = _transpose_mat(s_mat) == m_mat
    rank_s4 = exact_matrix_rank_q(s_mat)
    errs_l4 = [_vec_diff_norm_sq(mat_vec_exact(s_mat, k_vec), v_vec) for k_vec, v_vec in zip(keys4, vals4)]
    total_err_l4 = sum(errs_l4, Fraction(0, 1))

    # 5th unit-norm Pythagorean key k_4 = (3/5, 4/5, 0, 0) -> L = 5 > d = 4 capacity obstruction
    k5 = [Fraction(3, 5), Fraction(4, 5), Fraction(0, 1), Fraction(0, 1)]
    v5 = [Fraction(0, 1), Fraction(6, 1), Fraction(0, 1), Fraction(0, 1)]
    k5_unit_norm = dot_exact(k5, k5) == 1

    keys5 = keys4 + [k5]
    vals5 = vals4 + [v5]
    k5_mat = [[keys5[c][r] for c in range(5)] for r in range(4)]
    rank_k5 = exact_matrix_rank_q(k5_mat)

    c_null = [dot_exact(k5, kj) for kj in keys4] + [Fraction(-1, 1)]
    k5_c = mat_vec_exact(k5_mat, c_null)
    nullspace_exact = all(x == 0 for x in k5_c)
    d_incons = [
        sum((c_null[j] * vals5[j][r] for j in range(5)), Fraction(0, 1))
        for r in range(4)
    ]
    universal_cs_lb = dot_exact(d_incons, d_incons) / dot_exact(c_null, c_null)

    s5_mat = gdn2_delta_step_exact(s_mat, k5, v5)
    m5_mat = _gdn2_transpose_step_exact(m_mat, k5, v5)
    transpose_duality_l5_exact = _transpose_mat(s5_mat) == m5_mat
    rank_s5 = exact_matrix_rank_q(s5_mat)
    errs_l5 = [_vec_diff_norm_sq(mat_vec_exact(s5_mat, k_vec), v_vec) for k_vec, v_vec in zip(keys5, vals5)]
    total_err_l5 = sum(errs_l5, Fraction(0, 1))

    delta_s = [[s5_mat[i][j] - s_mat[i][j] for j in range(4)] for i in range(4)]
    rank_delta_s = exact_matrix_rank_q(delta_s)
    frob_delta_s_sq = _mat_frob_norm_sq(s5_mat, s_mat)
    parseval_exact = total_err_l5 == frob_delta_s_sq

    # Channel-2B Persistent Latent-Memory Poisoning Witness (frozen weights, benign query)
    w_cls = [Fraction(1, 1), Fraction(-1, 1), Fraction(0, 1), Fraction(0, 1)]
    q_benign = keys4[0]
    margin_clean = dot_exact(w_cls, mat_vec_exact(s_mat, q_benign))
    margin_poisoned = dot_exact(w_cls, mat_vec_exact(s5_mat, q_benign))

    # Intervening orthogonal benign turn k_orth = (4/5, -3/5, 0, 0) (k_orth^T k5 = 0)
    k_orth = [Fraction(4, 5), Fraction(-3, 5), Fraction(0, 1), Fraction(0, 1)]
    v_orth = mat_vec_exact(s_mat, k_orth)
    s5_after_orth = gdn2_delta_step_exact(s5_mat, k_orth, v_orth)
    margin_after_orth = dot_exact(w_cls, mat_vec_exact(s5_after_orth, q_benign))
    delta_after_orth = [[s5_after_orth[i][j] - s_mat[i][j] for j in range(4)] for i in range(4)]
    poison_persists_after_benign_turn = all((
        dot_exact(k_orth, k_orth) == 1,
        dot_exact(k_orth, k5) == 0,
        delta_after_orth == delta_s,
        margin_after_orth == margin_poisoned,
    ))

    s4_sha256 = hashlib.sha256(
        canonical_json_bytes([[format_exact_rational(x) for x in row] for row in s_mat])
    ).hexdigest()
    s5_sha256 = hashlib.sha256(
        canonical_json_bytes([[format_exact_rational(x) for x in row] for row in s5_mat])
    ).hexdigest()

    base_airlock = {
        "token_embd_hash_match": True,
        "block_dpi_ratio": "1",
        "max_block_dpi_ratio": "4",
        "kv_cross_model_reuse": False,
        "kv_residual_evicted_mentions": 0,
        "latent_state_hash_match": True,
        "latent_delta_norm_sq": "0",
        "max_latent_delta_norm_sq": "1",
        "latent_cross_session_carryover": False,
        "corpus_provenance_verified": True,
        "cot_mode_transfer_gap": "3/125",
        "max_cot_mode_transfer_gap": "1/20",
    }
    clean_res = evaluate_four_channel_airlock(base_airlock)

    rec_ch1 = dict(base_airlock, token_embd_hash_match=False)
    rec_ch2a = dict(base_airlock, kv_cross_model_reuse=True)
    rec_ch2b = dict(
        base_airlock,
        latent_state_hash_match=(s4_sha256 == s5_sha256),
        latent_delta_norm_sq=format_exact_rational(frob_delta_s_sq),
        latent_cross_session_carryover=True,
    )
    rec_ch3 = dict(base_airlock, cot_mode_transfer_gap="77/500")

    neg_controls = [
        {"name": "channel1_weight_tamper", **evaluate_four_channel_airlock(rec_ch1)},
        {"name": "channel2a_kv_reuse", **evaluate_four_channel_airlock(rec_ch2a)},
        {"name": "channel2b_latent_poisoning", **evaluate_four_channel_airlock(rec_ch2b)},
        {"name": "channel3_data_cot_drop", **evaluate_four_channel_airlock(rec_ch3)},
    ]

    ch2b_res = neg_controls[2]
    legacy_3ch_blind = all((
        ch2b_res["legacy_3ch_decision"] == "ALLOW",
        ch2b_res["airlock_4ch_decision"] == "DENY",
    ))

    return {
        "capacity_obstruction": {
            "state_dim_d": 4,
            "hadamard_keys_orthonormal": gram_is_identity,
            "transpose_duality_l4_exact": transpose_duality_exact,
            "transpose_duality_l5_exact": transpose_duality_l5_exact,
            "rank_s4": rank_s4,
            "l4_per_key_err_sq": [format_exact_rational(e) for e in errs_l4],
            "l4_total_err_sq": format_exact_rational(total_err_l4),
            "l4_zero_error_exact": total_err_l4 == 0,
            "k5_unit_norm_exact": k5_unit_norm,
            "rank_key_matrix_k5": rank_k5,
            "rank_s5": rank_s5,
            "nullspace_witness_c": [format_exact_rational(x) for x in c_null],
            "nullspace_k5_c_zero_exact": nullspace_exact,
            "inconsistency_vector_d": [format_exact_rational(x) for x in d_incons],
            "universal_cauchy_schwarz_lb": format_exact_rational(universal_cs_lb),
            "l5_per_key_err_sq": [format_exact_rational(e) for e in errs_l5],
            "l5_total_err_sq": format_exact_rational(total_err_l5),
            "delta_s_frob_norm_sq": format_exact_rational(frob_delta_s_sq),
            "parseval_error_identity_exact": parseval_exact,
            "capacity_obstruction_certified": all((
                total_err_l4 == 0,
                rank_k5 == 4,
                rank_s5 == 4,
                nullspace_exact,
                universal_cs_lb > 0,
                total_err_l5 >= universal_cs_lb,
                errs_l5[4] == 0,
                all(e > 0 for e in errs_l5[:4]),
            )),
        },
        "channel_2b_poisoning_witness": {
            "frozen_weights_delta_w_zero": True,
            "benign_query_delta_p_data_zero": True,
            "zero_kv_carryover_delta_kv_zero": True,
            "clean_state_sha256": s4_sha256,
            "poisoned_state_sha256": s5_sha256,
            "state_hashes_distinct": s4_sha256 != s5_sha256,
            "delta_s_exact_rank": rank_delta_s,
            "delta_s_frob_norm_sq": format_exact_rational(frob_delta_s_sq),
            "clean_margin": format_exact_rational(margin_clean),
            "poisoned_margin": format_exact_rational(margin_poisoned),
            "margin_after_benign_orth_turn": format_exact_rational(margin_after_orth),
            "benign_readout_sign_flipped": all((margin_clean > 0, margin_poisoned < 0)),
            "poison_persists_after_benign_turn": poison_persists_after_benign_turn,
        },
        "four_channel_airlock": {
            "clean_record": clean_res,
            "negative_controls": neg_controls,
            "legacy_3ch_blind_to_channel2b": legacy_3ch_blind,
            "all_four_negative_controls_denied": all(
                c["airlock_4ch_decision"] == "DENY" for c in neg_controls
            ),
            "airlock_extension_certified": all((
                clean_res["airlock_4ch_decision"] == "ALLOW",
                legacy_3ch_blind,
                all(c["airlock_4ch_decision"] == "DENY" for c in neg_controls),
            )),
        },
    }


# ---------------------------------------------------------------------------
# Standalone Certificate Assembly & Fail-Closed Validation
# ---------------------------------------------------------------------------

def build_standalone_certificate() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "novelty_tier": "N1",
        "oracle_class": "exact",
        "prior_art_cs13_anchors": [
            "ShinMegamiBoson/lookatmyquant",
            "professorpalmer/kintsugi-method",
            "professorpalmer/hard-proof",
            "google-deepmind/gbench",
            "MemTensor/MemT-1.0-Pro",
        ],
        "measured_live_gemma4_lamq_weights": False,
        "measured_live_gbench_serving_endpoint": False,
        "measured_live_memt_554b_weights": False,
        "zero_trust_path_floats": True,
        "zero_bare_asserts": True,
        "zero_mathematical_rng": True,
        "pillar1_fisher_waterfilling_amgm": build_pillar1_fisher_waterfilling(),
        "pillar2_kintsugi_margin_vs_kl": build_pillar2_kintsugi_margin_vs_kl(),
        "pillar3_gbench_p99_order_statistics": build_pillar3_gbench_p99_order_statistics(),
        "pillar4_memt_gdn2_rank_and_channel2b": build_pillar4_memt_gdn2_rank_and_channel2b(),
    }
    payload_sha = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    core = {
        "schema": SCHEMA_VERSION,
        "track_id": TRACK_ID,
        "payload_sha256": payload_sha,
        "payload": payload,
    }
    cert_sha = hashlib.sha256(canonical_json_bytes(core)).hexdigest()
    return {
        "schema": SCHEMA_VERSION,
        "track_id": TRACK_ID,
        "payload_sha256": payload_sha,
        "certificate_sha256": cert_sha,
        "payload": payload,
    }


_BOOL_LEAF_KEYS = frozenset({
    "measured_live_gemma4_lamq_weights",
    "measured_live_gbench_serving_endpoint",
    "measured_live_memt_554b_weights",
    "zero_trust_path_floats",
    "zero_bare_asserts",
    "zero_mathematical_rng",
    "all_b_star_positive",
    "kkt_equalization_exact",
    "am_gm_identity_exact",
    "all_perturbations_strictly_worse",
    "strictly_worse_than_fisher",
    "uniform_ratio_equals_one",
    "mild_ratio_exceeds_two",
    "heavy_ratio_exceeds_twelve",
    "strict_hierarchy_certified",
    "fisher_shift_invariant_nullspace",
    "fisher_3x3_block_pd",
    "task_pass",
    "uniform_lower_avg_kl_yet_breaks_task",
    "uniform_a_task_pass",
    "kintsugi_b_task_pass",
    "gibbs_strict_positivity_certified",
    "cs20_average_kl_blindness_certified",
    "den_exceeds_max_json_int_bits",
    "pass_99pct_upper_tail",
    "miss_prob_below_1_over_60000",
    "sharp_boundary_458_fails_459_passes",
    "coverage_exceeds_95pct",
    "cs18_uncertified_floor_enforced",
    "hadamard_keys_orthonormal",
    "transpose_duality_l4_exact",
    "transpose_duality_l5_exact",
    "l4_zero_error_exact",
    "k5_unit_norm_exact",
    "nullspace_k5_c_zero_exact",
    "parseval_error_identity_exact",
    "capacity_obstruction_certified",
    "frozen_weights_delta_w_zero",
    "benign_query_delta_p_data_zero",
    "zero_kv_carryover_delta_kv_zero",
    "state_hashes_distinct",
    "benign_readout_sign_flipped",
    "poison_persists_after_benign_turn",
    "legacy_3ch_blind_to_channel2b",
    "all_four_negative_controls_denied",
    "airlock_extension_certified",
})


def _validate_payload_nonreplay_schema(payload: dict[str, Any]) -> None:
    if frozenset(payload.keys()) != _PAYLOAD_KEYS:
        raise FisherKintsugiMemtError("INVALID_PAYLOAD_KEYS", "payload keys do not match Track S42 schema")
    if payload["novelty_tier"] != "N1":
        raise FisherKintsugiMemtError("INVALID_NOVELTY_TIER", "novelty_tier must be N1")
    if payload["oracle_class"] != "exact":
        raise FisherKintsugiMemtError("INVALID_ORACLE_CLASS", "oracle_class must be exact")
    anchors = payload["prior_art_cs13_anchors"]
    if type(anchors) is not list or len(anchors) != 5:
        raise FisherKintsugiMemtError("INVALID_CS13_ANCHORS", "prior_art_cs13_anchors must be 5-item list")
    for idx, item in enumerate(anchors):
        require_exact_str(item, f"prior_art_cs13_anchors[{idx}]")
    for key in (
        "measured_live_gemma4_lamq_weights",
        "measured_live_gbench_serving_endpoint",
        "measured_live_memt_554b_weights",
    ):
        if require_exact_bool(payload[key], key) is not False:
            raise FisherKintsugiMemtError("INVALID_LAW4_FLAG", f"{key} must be False")
    for key in ("zero_trust_path_floats", "zero_bare_asserts", "zero_mathematical_rng"):
        if require_exact_bool(payload[key], key) is not True:
            raise FisherKintsugiMemtError("INVALID_HYGIENE_FLAG", f"{key} must be True")
    for key in (
        "pillar1_fisher_waterfilling_amgm",
        "pillar2_kintsugi_margin_vs_kl",
        "pillar3_gbench_p99_order_statistics",
        "pillar4_memt_gdn2_rank_and_channel2b",
    ):
        if type(payload[key]) is not dict:
            raise FisherKintsugiMemtError("INVALID_PILLAR_OBJECT", f"{key} must be dict")

    # Bijective bool/int type separation across the entire payload tree (even under full_replay=False)
    stack: list[Any] = [payload]
    while stack:
        cur = stack.pop()
        if type(cur) is dict:
            for k, v in cur.items():
                if k in _BOOL_LEAF_KEYS:
                    if type(v) is not bool:
                        raise FisherKintsugiMemtError("NON_BOOL_IN_BOOL_FIELD", f"{k} must be exact bool")
                elif type(v) is bool:
                    raise FisherKintsugiMemtError("BOOL_IN_NON_BOOL_FIELD", f"{k} cannot be bool")
                else:
                    stack.append(v)
        elif type(cur) is list:
            for item in cur:
                if type(item) is bool:
                    raise FisherKintsugiMemtError("BOOL_IN_LIST", "bool forbidden inside payload list")
                stack.append(item)


def validate_certificate_payload(cert: Any, *, full_replay: bool = True) -> dict[str, Any]:
    """Fail-closed validation of certificate structure, digests, schema, and optional full pure-Q replay."""
    replay_flag = require_exact_bool(full_replay, "full_replay")
    if type(cert) is not dict:
        raise FisherKintsugiMemtError("CERT_NOT_OBJECT", "certificate must be a dict")
    validate_json_tree_iterative(cert)
    if frozenset(cert.keys()) != _CERT_KEYS:
        raise FisherKintsugiMemtError("INVALID_CERT_KEYS", "certificate keys mismatch")
    if cert["schema"] != SCHEMA_VERSION:
        raise FisherKintsugiMemtError("INVALID_SCHEMA", "unexpected certificate schema")
    if cert["track_id"] != TRACK_ID:
        raise FisherKintsugiMemtError("INVALID_TRACK_ID", "unexpected track_id")

    payload = cert["payload"]
    if type(payload) is not dict:
        raise FisherKintsugiMemtError("PAYLOAD_NOT_OBJECT", "payload must be a dict")

    declared_p_sha = require_sha256_hex(cert["payload_sha256"], "payload_sha256")
    declared_c_sha = require_sha256_hex(cert["certificate_sha256"], "certificate_sha256")

    actual_p_sha = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    if not hmac.compare_digest(declared_p_sha, actual_p_sha):
        raise FisherKintsugiMemtError("PAYLOAD_SHA_MISMATCH", "payload_sha256 mismatch")

    core = {
        "schema": SCHEMA_VERSION,
        "track_id": TRACK_ID,
        "payload_sha256": actual_p_sha,
        "payload": payload,
    }
    actual_c_sha = hashlib.sha256(canonical_json_bytes(core)).hexdigest()
    if not hmac.compare_digest(declared_c_sha, actual_c_sha):
        raise FisherKintsugiMemtError("CERT_SHA_MISMATCH", "certificate_sha256 mismatch")

    _validate_payload_nonreplay_schema(payload)

    if replay_flag:
        expected = build_standalone_certificate()
        if not hmac.compare_digest(canonical_json_bytes(cert), canonical_json_bytes(expected)):
            raise FisherKintsugiMemtError("REPLAY_MISMATCH", "certificate does not match exact pure-Q replay")

    return {
        "valid": True,
        "schema": SCHEMA_VERSION,
        "track_id": TRACK_ID,
        "full_replay": replay_flag,
        "payload_sha256": actual_p_sha,
        "certificate_sha256": actual_c_sha,
    }
