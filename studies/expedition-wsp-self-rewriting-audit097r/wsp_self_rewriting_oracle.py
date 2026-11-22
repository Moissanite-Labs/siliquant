"""Pure-Q exact oracle for weight-edit minimization, rank-1/sparse subspace leakage,
siliquant group-quantization survival ((3/4)-ULP erasure & spike scale distortion),
paired dual-metric screening, and fail-closed self-rewriting governance.

Bridges:
  - alphaXiv/weight-edit-minimization (Exp 8 Qwen3.5-0.8B & Exp 4 car-ramp)
  - arXiv:2611.21885v2 (Self-Rewriting LLMs, Qwen3.5 2B/4B/9B/27B)
  - Moissanite-Labs/warp-drive Stage-2 FixedRadixCertificate & Stages 1-4
  - Moissanite-Labs/siliquant (Part L ffn_down outliers, #10/#12/#21/#26)
  - Foundation-0/aletheia v2.0.0 R6/R7 & Gate-8 invariants

Strict discipline: pure Python 3 stdlib, exact int/Fraction only, zero floats,
zero bare asserts, fail-closed validation on every trust boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import hmac
import json
import math
import re
from typing import Any, Mapping, Sequence

SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z")
BF16_DYADIC_SHIFT = 133
F32_DYADIC_SHIFT = 149
F64_DYADIC_SHIFT = 1074
BF16_EXP_MASK = 0xFF
F32_EXP_MASK = 0xFF
F64_EXP_MASK = 0x7FF


class OracleError(ValueError):
    """Fail-closed validation error for exact weight-edit & governance checks."""


def _require_exact_int(value: Any, name: str, *, allow_zero: bool = True) -> int:
    if type(value) is not int:
        raise OracleError(f"{name} must be an exact int (got {type(value).__name__})")
    if allow_zero:
        if value < 0:
            raise OracleError(f"{name} must be non-negative (got {value})")
    else:
        if value <= 0:
            raise OracleError(f"{name} must be positive (got {value})")
    return value


def _require_exact_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise OracleError(f"{name} must be an exact bool (got {type(value).__name__})")
    return value


def _require_sha256(value: Any, name: str) -> str:
    if type(value) is not str or not SHA256_RE.match(value) or len(set(value)) == 1:
        raise OracleError(f"{name} must be a non-degenerate 64-char lowercase hex sha256")
    return value


def contains_float_recursive(obj: Any) -> bool:
    """Return True if any float or complex exists anywhere inside nested mappings/collections."""
    if isinstance(obj, (float, complex)):
        return True
    if isinstance(obj, Mapping):
        return any(contains_float_recursive(k) or contains_float_recursive(v) for k, v in obj.items())
    if isinstance(obj, (list, tuple, set, frozenset)):
        return any(contains_float_recursive(item) for item in obj)
    return False


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    """Compute canonical SHA-256 over a JSON-serializable mapping."""
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def frac_to_dict(value: Fraction) -> dict[str, int]:
    """Serialize an exact Fraction as a reduced integer pair with zero floats."""
    if not isinstance(value, Fraction):
        raise OracleError(f"expected Fraction, got {type(value).__name__}")
    return {"num": value.numerator, "den": value.denominator}


def format_exact_decimal(value: Fraction, places: int = 6) -> str:
    """Format an exact Fraction to a fixed-decimal string using pure integer arithmetic."""
    if not isinstance(value, Fraction):
        raise OracleError("value must be Fraction")
    _require_exact_int(places, "places", allow_zero=True)
    sign = "-" if value < 0 else ""
    mag = -value if value < 0 else value
    scale = 10 ** places
    scaled = (mag.numerator * scale * 2 + mag.denominator) // (2 * mag.denominator)
    whole, frac = divmod(scaled, scale)
    if places == 0:
        return f"{sign}{whole}"
    return f"{sign}{whole}.{frac:0{places}d}"


def format_exact_percent(value: Fraction, places: int = 3) -> str:
    """Format an exact ratio Fraction as a percentage string via pure integer arithmetic."""
    return format_exact_decimal(value * 100, places=places) + "%"


@dataclass(frozen=True)
class RationalBound:
    """Exact non-negative rational upper bound num / den (den > 0)."""

    num: int
    den: int = 1

    def __post_init__(self) -> None:
        _require_exact_int(self.num, "bound.num", allow_zero=True)
        _require_exact_int(self.den, "bound.den", allow_zero=False)

    def to_dict(self) -> dict[str, int]:
        return {"num": self.num, "den": self.den}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RationalBound:
        if not isinstance(data, Mapping):
            raise OracleError("rational bound must be a mapping")
        if "num" not in data or "den" not in data:
            raise OracleError("rational bound requires 'num' and 'den'")
        if len(data) != 2:
            raise OracleError("rational bound must contain only 'num' and 'den'")
        return cls(num=data["num"], den=data["den"])


def f32_u32_to_scaled_int(bits: int) -> int:
    """Decode IEEE-754 binary32 bits to an exact integer scaled by 2^149 (fail-closed on NaN/Inf)."""
    _require_exact_int(bits, "f32_bits", allow_zero=True)
    if bits >= (1 << 32):
        raise OracleError(f"f32_bits out of uint32 range: {bits}")
    sign = (bits >> 31) & 1
    exp = (bits >> 23) & F32_EXP_MASK
    mant = bits & 0x7FFFFF
    if exp == F32_EXP_MASK:
        raise OracleError(f"non-finite IEEE-754 binary32 bit pattern: 0x{bits:08x}")
    if exp == 0:
        mag = mant
    else:
        mag = ((1 << 23) | mant) << (exp - 1)
    return -mag if sign else mag


def f32_u32_to_fraction(bits: int) -> Fraction:
    """Decode IEEE-754 binary32 bits to an exact Fraction."""
    return Fraction(f32_u32_to_scaled_int(bits), 1 << F32_DYADIC_SHIFT)


def f64_u64_to_scaled_int(bits: int) -> int:
    """Decode IEEE-754 binary64 bits to an exact integer scaled by 2^1074 (fail-closed on NaN/Inf)."""
    _require_exact_int(bits, "f64_bits", allow_zero=True)
    if bits >= (1 << 64):
        raise OracleError(f"f64_bits out of uint64 range: {bits}")
    sign = (bits >> 63) & 1
    exp = (bits >> 52) & F64_EXP_MASK
    mant = bits & 0xFFFFFFFFFFFFF
    if exp == F64_EXP_MASK:
        raise OracleError(f"non-finite IEEE-754 binary64 bit pattern: 0x{bits:016x}")
    if exp == 0:
        mag = mant
    else:
        mag = ((1 << 52) | mant) << (exp - 1)
    return -mag if sign else mag


def f64_u64_to_fraction(bits: int) -> Fraction:
    """Decode IEEE-754 binary64 bits to an exact Fraction."""
    return Fraction(f64_u64_to_scaled_int(bits), 1 << F64_DYADIC_SHIFT)


def bf16_u16_to_scaled_int(bits: int) -> int:
    """Decode bfloat16 bits to an exact integer scaled by 2^133 (fail-closed on NaN/Inf)."""
    _require_exact_int(bits, "bf16_bits", allow_zero=True)
    if bits >= (1 << 16):
        raise OracleError(f"bf16_bits out of uint16 range: {bits}")
    sign = (bits >> 15) & 1
    exp = (bits >> 7) & BF16_EXP_MASK
    mant = bits & 0x7F
    if exp == BF16_EXP_MASK:
        raise OracleError(f"non-finite bfloat16 bit pattern: 0x{bits:04x}")
    if exp == 0:
        mag = mant
    else:
        mag = ((1 << 7) | mant) << (exp - 1)
    return -mag if sign else mag


def bf16_u16_to_fraction(bits: int) -> Fraction:
    """Decode bfloat16 bits to an exact Fraction."""
    return Fraction(bf16_u16_to_scaled_int(bits), 1 << BF16_DYADIC_SHIFT)


def f32_u32_to_bf16_u16_rne(bits: int) -> int:
    """Round finite IEEE-754 binary32 bits to bfloat16 (RNE), rejecting NaN/Inf and overflow-to-Inf.

    Directly closes the silent-infinity hazard in weight-edit-minimization/code/st_numpy.py:55-57
    where 0x7F7FFFFF rounds silently to 0x7F80 (+inf).
    """
    _require_exact_int(bits, "f32_bits", allow_zero=True)
    if bits >= (1 << 32):
        raise OracleError(f"f32_bits out of uint32 range: {bits}")
    exp = (bits >> 23) & F32_EXP_MASK
    if exp == F32_EXP_MASK:
        raise OracleError(f"non-finite IEEE-754 binary32 cannot convert to bf16: 0x{bits:08x}")
    lsb = (bits >> 16) & 1
    rounded = (bits + 0x7FFF + lsb) >> 16
    out = rounded & 0xFFFF
    if ((out >> 7) & BF16_EXP_MASK) == BF16_EXP_MASK:
        raise OracleError(f"binary32 0x{bits:08x} overflows to non-finite bf16 0x{out:04x}")
    return out


def _require_fraction_vector(vec: Sequence[Fraction], name: str) -> list[Fraction]:
    if not isinstance(vec, Sequence) or isinstance(vec, (str, bytes)) or len(vec) == 0:
        raise OracleError(f"{name} must be a non-empty sequence of Fractions")
    out: list[Fraction] = []
    norm_sq = Fraction(0)
    for idx, val in enumerate(vec):
        if not isinstance(val, Fraction):
            raise OracleError(f"{name}[{idx}] must be Fraction (got {type(val).__name__})")
        out.append(val)
        norm_sq += val * val
    if norm_sq == 0:
        raise OracleError(f"{name} must be non-zero")
    return out


def _classify_binary_mask(mask: Sequence[Sequence[int]], rows: int, cols: int) -> tuple[str, list[int], list[int]]:
    if not isinstance(mask, Sequence) or isinstance(mask, (str, bytes)) or len(mask) != rows:
        raise OracleError(f"mask must have {rows} rows")
    active_rows: set[int] = set()
    active_cols: set[int] = set()
    active_count = 0
    for r_idx, row in enumerate(mask):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) != cols:
            raise OracleError(f"mask row {r_idx} must have {cols} columns")
        for c_idx, cell in enumerate(row):
            if type(cell) is not int or cell not in (0, 1):
                raise OracleError(f"mask[{r_idx}][{c_idx}] must be exact int 0 or 1")
            if cell == 1:
                active_rows.add(r_idx)
                active_cols.add(c_idx)
                active_count += 1

    if active_count == 0:
        return "empty_mask", [], []
    if active_count == rows * cols:
        return "full_rank1", sorted(active_rows), sorted(active_cols)

    is_cartesian = active_count == len(active_rows) * len(active_cols)
    if is_cartesian:
        if len(active_rows) == rows:
            return "column_mask", sorted(active_rows), sorted(active_cols)
        if len(active_cols) == cols:
            return "row_mask", sorted(active_rows), sorted(active_cols)
        return "cartesian_subblock", sorted(active_rows), sorted(active_cols)
    return "entry_scattered", sorted(active_rows), sorted(active_cols)


def certify_rank1_vs_pruning_exact(
    u: Sequence[Fraction],
    v: Sequence[Fraction],
    mask: Sequence[Sequence[int]],
    safety_vectors: Sequence[Sequence[Fraction]],
) -> dict[str, Any]:
    """Certify exact pure-Q subspace leakage, rank-1 collapse, and safety orthogonality under pruning."""
    u_vec = _require_fraction_vector(u, "u")
    v_vec = _require_fraction_vector(v, "v")
    rows = len(u_vec)
    cols = len(v_vec)
    regime, active_rows, active_cols = _classify_binary_mask(mask, rows, cols)

    if not isinstance(safety_vectors, Sequence) or isinstance(safety_vectors, (str, bytes)):
        raise OracleError("safety_vectors must be a sequence of Fraction vectors")
    s_vecs: list[list[Fraction]] = []
    for s_idx, s_raw in enumerate(safety_vectors):
        sv = _require_fraction_vector(s_raw, f"safety_vectors[{s_idx}]")
        if len(sv) != rows:
            raise OracleError(f"safety_vectors[{s_idx}] length {len(sv)} != u length {rows}")
        s_vecs.append(sv)

    u_norm_sq = sum(x * x for x in u_vec)
    v_norm_sq = sum(y * y for y in v_vec)
    full_frob_sq = u_norm_sq * v_norm_sq

    delta: list[list[Fraction]] = [
        [u_vec[r] * v_vec[c] if mask[r][c] == 1 else Fraction(0) for c in range(cols)]
        for r in range(rows)
    ]

    frob_sq = Fraction(0)
    col_u_dots: list[Fraction] = [Fraction(0) for _ in range(cols)]
    for r in range(rows):
        ur = u_vec[r]
        for c in range(cols):
            val = delta[r][c]
            frob_sq += val * val
            col_u_dots[c] += ur * val

    par_sq = sum(d * d for d in col_u_dots) / u_norm_sq
    perp_sq = frob_sq - par_sq

    gram_frob_sq = Fraction(0)
    for c1 in range(cols):
        for c2 in range(cols):
            dot = sum(delta[r][c1] * delta[r][c2] for r in range(rows))
            gram_frob_sq += dot * dot
    rank1_gram_defect = frob_sq * frob_sq - gram_frob_sq
    is_exact_rank_le_1 = rank1_gram_defect == 0

    base_max_safety_dot_sq = Fraction(0)
    pruned_max_safety_dot_sq = Fraction(0)
    for sv in s_vecs:
        s_norm_sq = sum(x * x for x in sv)
        su = sum(sv[r] * u_vec[r] for r in range(rows))
        base_dot_sq = (su * su * v_norm_sq) / s_norm_sq
        if base_dot_sq > base_max_safety_dot_sq:
            base_max_safety_dot_sq = base_dot_sq

        col_s_sq = Fraction(0)
        for c in range(cols):
            sd = sum(sv[r] * delta[r][c] for r in range(rows))
            col_s_sq += sd * sd
        pruned_dot_sq = col_s_sq / s_norm_sq
        if pruned_dot_sq > pruned_max_safety_dot_sq:
            pruned_max_safety_dot_sq = pruned_dot_sq

    row_energy_retained = sum(u_vec[r] * u_vec[r] for r in active_rows) / u_norm_sq if active_rows else Fraction(0)
    closed_form_row_leakage_holds = False
    if regime == "row_mask":
        expected_perp = row_energy_retained * (Fraction(1) - row_energy_retained) * full_frob_sq
        closed_form_row_leakage_holds = perp_sq == expected_perp

    leakage_fraction = (perp_sq / frob_sq) if frob_sq > 0 else Fraction(0)
    safety_orthogonality_breached = (base_max_safety_dot_sq == 0) and (pruned_max_safety_dot_sq > 0)

    return {
        "regime": regime,
        "shape": [rows, cols],
        "active_entries": sum(sum(row) for row in mask),
        "full_frobenius_sq": frac_to_dict(full_frob_sq),
        "frobenius_sq": frac_to_dict(frob_sq),
        "parallel_energy_sq": frac_to_dict(par_sq),
        "leakage_energy_sq": frac_to_dict(perp_sq),
        "leakage_fraction": frac_to_dict(leakage_fraction),
        "leakage_percent": format_exact_percent(leakage_fraction, 3),
        "pythagorean_exact": (par_sq + perp_sq == frob_sq) and (perp_sq >= 0),
        "rank1_gram_defect": frac_to_dict(rank1_gram_defect),
        "gram_defect_nonnegative": rank1_gram_defect >= 0,
        "is_exact_rank_le_1": is_exact_rank_le_1,
        "closed_form_row_leakage_holds": closed_form_row_leakage_holds,
        "base_max_safety_dot_sq": frac_to_dict(base_max_safety_dot_sq),
        "pruned_max_safety_dot_sq": frac_to_dict(pruned_max_safety_dot_sq),
        "safety_orthogonality_breached": safety_orthogonality_breached,
    }


def quantize_sym_rne_exact(val_int: int, max_abs_int: int, qmax: int = 7) -> int:
    """Exact pure-integer symmetric round-half-to-even quantizer: clip(rne(val * qmax / max_abs), -qmax, qmax)."""
    if type(val_int) is not int:
        raise OracleError(f"val_int must be exact int (got {type(val_int).__name__})")
    _require_exact_int(max_abs_int, "max_abs_int", allow_zero=True)
    _require_exact_int(qmax, "qmax", allow_zero=False)
    if max_abs_int == 0:
        return 0
    neg = val_int < 0
    num = (-val_int if neg else val_int) * qmax
    q, rem = divmod(num, max_abs_int)
    twice_rem = rem * 2
    if twice_rem > max_abs_int or (twice_rem == max_abs_int and (q & 1) == 1):
        q += 1
    if q > qmax:
        q = qmax
    return -q if neg else q


def certify_group_quantization_survival_exact(
    base_group: Sequence[int],
    edited_group: Sequence[int],
    *,
    qmax: int = 7,
) -> dict[str, Any]:
    """Certify exact pure-Q survival vs (3/4)-ULP erasure & spike-driven collateral zeroing in a weight group.

    Bridges Exp 8 down_proj / car-ramp edits directly to Moissanite-Labs/siliquant
    (Part L ffn_down outlier concentration, #12 ZBG128, #26 (3/4)-ULP rounding-cell erasure).
    """
    _require_exact_int(qmax, "qmax", allow_zero=False)
    if not isinstance(base_group, Sequence) or isinstance(base_group, (str, bytes)) or len(base_group) == 0:
        raise OracleError("base_group must be a non-empty sequence of exact ints")
    if not isinstance(edited_group, Sequence) or isinstance(edited_group, (str, bytes)) or len(edited_group) != len(base_group):
        raise OracleError("edited_group must match base_group length")

    for idx, (bv, ev) in enumerate(zip(base_group, edited_group)):
        if type(bv) is not int or type(ev) is not int:
            raise OracleError(f"group entry {idx} must be exact int")

    max_base = max(abs(x) for x in base_group)
    max_edit = max(abs(x) for x in edited_group)
    scale_ratio = Fraction(max_edit, max_base) if max_base > 0 else Fraction(0, 1)

    edited_count = 0
    survived_code_flips = 0
    erased_edits = 0
    below_half_ulp = 0
    below_three_quarter_ulp = 0
    at_least_one_ulp = 0
    unedited_count = 0
    unedited_nonzero_before = 0
    collateral_unedited_flips = 0
    collateral_unedited_zeroed = 0

    for bv, ev in zip(base_group, edited_group):
        qb = quantize_sym_rne_exact(bv, max_base, qmax)
        qe = quantize_sym_rne_exact(ev, max_edit, qmax)
        if ev != bv:
            edited_count += 1
            d = abs(ev - bv)
            if qe != qb:
                survived_code_flips += 1
            else:
                erased_edits += 1
            if 2 * qmax * d < max_base:
                below_half_ulp += 1
            if 4 * qmax * d < 3 * max_base:
                below_three_quarter_ulp += 1
            if qmax * d >= max_base:
                at_least_one_ulp += 1
        else:
            unedited_count += 1
            if qb != 0:
                unedited_nonzero_before += 1
                if qe == 0:
                    collateral_unedited_zeroed += 1
            if qe != qb:
                collateral_unedited_flips += 1

    survival_rate = Fraction(survived_code_flips, edited_count) if edited_count > 0 else Fraction(0, 1)
    erasure_3q_rate = Fraction(below_three_quarter_ulp, edited_count) if edited_count > 0 else Fraction(0, 1)
    return {
        "group_size": len(base_group),
        "qmax": qmax,
        "max_abs_base_scaled": max_base,
        "max_abs_edit_scaled": max_edit,
        "scale_changed": max_edit != max_base,
        "scale_distortion_ratio": frac_to_dict(scale_ratio),
        "scale_distortion_dec": format_exact_decimal(scale_ratio, 3),
        "edited_count": edited_count,
        "survived_code_flips": survived_code_flips,
        "erased_edits": erased_edits,
        "below_half_ulp": below_half_ulp,
        "below_three_quarter_ulp": below_three_quarter_ulp,
        "at_least_one_ulp": at_least_one_ulp,
        "survival_rate": frac_to_dict(survival_rate),
        "survival_percent": format_exact_percent(survival_rate, 2),
        "below_three_quarter_ulp_percent": format_exact_percent(erasure_3q_rate, 2),
        "unedited_count": unedited_count,
        "unedited_nonzero_before": unedited_nonzero_before,
        "collateral_unedited_flips": collateral_unedited_flips,
        "collateral_unedited_zeroed": collateral_unedited_zeroed,
    }


def make_fixed_radix_certificate(
    *,
    adapter_id: str,
    adapter_sha256: str,
    base_weights_sha256: str,
    radix_scale: int,
    frobenius_sq: Fraction,
    spectral_norm_sq: Fraction,
    leakage_energy: Fraction,
    max_safety_dot_sq: Fraction,
) -> dict[str, Any]:
    """Construct a conservative ceiling-rounded FixedRadixCertificate compatible with warp-drive Stage 2."""
    if type(adapter_id) is not str or not adapter_id.strip():
        raise OracleError("adapter_id must be a non-empty string")
    _require_sha256(adapter_sha256, "adapter_sha256")
    _require_sha256(base_weights_sha256, "base_weights_sha256")
    _require_exact_int(radix_scale, "radix_scale", allow_zero=False)

    def _ceil_scaled(val: Fraction, name: str) -> int:
        if not isinstance(val, Fraction) or val < 0:
            raise OracleError(f"{name} must be a non-negative Fraction")
        scaled = val * radix_scale
        return (scaled.numerator + scaled.denominator - 1) // scaled.denominator

    frob_s = _ceil_scaled(frobenius_sq, "frobenius_sq")
    spec_s = _ceil_scaled(spectral_norm_sq, "spectral_norm_sq")
    leak_s = _ceil_scaled(leakage_energy, "leakage_energy")
    safe_s = _ceil_scaled(max_safety_dot_sq, "max_safety_dot_sq")

    if spec_s > frob_s:
        raise OracleError(f"inconsistent metrics: spectral_norm_sq_scaled ({spec_s}) > frobenius_sq_scaled ({frob_s})")
    if leak_s > frob_s:
        raise OracleError(f"inconsistent metrics: leakage_energy_scaled ({leak_s}) > frobenius_sq_scaled ({frob_s})")
    if safe_s > spec_s:
        raise OracleError(f"inconsistent metrics: max_safety_dot_sq_scaled ({safe_s}) > spectral_norm_sq_scaled ({spec_s})")

    body = {
        "adapter_id": adapter_id,
        "adapter_sha256": adapter_sha256,
        "base_weights_sha256": base_weights_sha256,
        "radix_scale": radix_scale,
        "frobenius_sq_scaled": frob_s,
        "spectral_norm_sq_scaled": spec_s,
        "leakage_energy_scaled": leak_s,
        "max_safety_dot_sq_scaled": safe_s,
    }
    out = dict(body)
    out["cert_sha256"] = canonical_sha256(body)
    return out


def verify_fixed_radix_stage2_division_free(
    cert: Mapping[str, Any],
    bounds: Mapping[str, Any],
    *,
    scale_num: int = 1,
    scale_den: int = 1,
    expected_base_weights_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify a warp-drive Stage-2 FixedRadixCertificate using strictly exact integer cross-multiplication."""
    if not isinstance(cert, Mapping):
        raise OracleError("cert must be a mapping")
    if not isinstance(bounds, Mapping):
        raise OracleError("bounds must be a mapping")
    _require_exact_int(scale_num, "scale_num", allow_zero=False)
    _require_exact_int(scale_den, "scale_den", allow_zero=False)

    req_cert = (
        "adapter_id",
        "adapter_sha256",
        "base_weights_sha256",
        "radix_scale",
        "frobenius_sq_scaled",
        "spectral_norm_sq_scaled",
        "leakage_energy_scaled",
        "max_safety_dot_sq_scaled",
        "cert_sha256",
    )
    for key in req_cert:
        if key not in cert:
            raise OracleError(f"missing certificate field: {key}")
    if len(cert) != len(req_cert):
        raise OracleError("unexpected extra field in certificate")

    req_bound_keys = (
        "max_frobenius_sq",
        "max_spectral_norm_sq",
        "max_leakage_energy",
        "max_safety_dot_sq",
    )
    for bkey in req_bound_keys:
        if bkey not in bounds:
            raise OracleError(f"missing bounds field: {bkey}")
    if len(bounds) != len(req_bound_keys):
        raise OracleError("unexpected extra field in bounds")

    adapter_id = cert["adapter_id"]
    if type(adapter_id) is not str or not adapter_id.strip():
        raise OracleError("cert.adapter_id must be a non-empty string")
    adapter_sha = _require_sha256(cert["adapter_sha256"], "cert.adapter_sha256")
    base_sha = _require_sha256(cert["base_weights_sha256"], "cert.base_weights_sha256")
    cert_sha = _require_sha256(cert["cert_sha256"], "cert.cert_sha256")
    radix_scale = _require_exact_int(cert["radix_scale"], "cert.radix_scale", allow_zero=False)
    frob = _require_exact_int(cert["frobenius_sq_scaled"], "cert.frobenius_sq_scaled", allow_zero=True)
    spec = _require_exact_int(cert["spectral_norm_sq_scaled"], "cert.spectral_norm_sq_scaled", allow_zero=True)
    leak = _require_exact_int(cert["leakage_energy_scaled"], "cert.leakage_energy_scaled", allow_zero=True)
    safe = _require_exact_int(cert["max_safety_dot_sq_scaled"], "cert.max_safety_dot_sq_scaled", allow_zero=True)

    body = {
        "adapter_id": adapter_id,
        "adapter_sha256": adapter_sha,
        "base_weights_sha256": base_sha,
        "radix_scale": radix_scale,
        "frobenius_sq_scaled": frob,
        "spectral_norm_sq_scaled": spec,
        "leakage_energy_scaled": leak,
        "max_safety_dot_sq_scaled": safe,
    }
    expected_digest = canonical_sha256(body)
    if not hmac.compare_digest(cert_sha, expected_digest):
        raise OracleError(f"forged cert_sha256 for {adapter_id}")

    if expected_base_weights_sha256 is not None:
        exp_base = _require_sha256(expected_base_weights_sha256, "expected_base_weights_sha256")
        if not hmac.compare_digest(base_sha, exp_base):
            raise OracleError(f"base_weights_sha256 mismatch for {adapter_id}")

    if spec > frob:
        raise OracleError(f"inconsistent cert for {adapter_id}: spectral > frobenius")
    if leak > frob:
        raise OracleError(f"inconsistent cert for {adapter_id}: leakage > frobenius")
    if safe > spec:
        raise OracleError(f"inconsistent cert for {adapter_id}: safety_dot_sq > spectral")

    req_bounds = (
        ("frobenius_sq", frob, "max_frobenius_sq"),
        ("spectral_norm_sq", spec, "max_spectral_norm_sq"),
        ("leakage_energy", leak, "max_leakage_energy"),
        ("max_safety_dot_sq", safe, "max_safety_dot_sq"),
    )
    violations: list[str] = []
    checks: dict[str, bool] = {}
    for metric_name, val_scaled, bound_key in req_bounds:
        bound = RationalBound.from_dict(bounds[bound_key])
        lhs = val_scaled * (scale_num * scale_num) * bound.den
        rhs = bound.num * radix_scale * (scale_den * scale_den)
        ok = lhs <= rhs
        checks[metric_name] = ok
        if not ok:
            violations.append(metric_name)

    return {
        "adapter_id": adapter_id,
        "cert_sha256": cert_sha,
        "passed": len(violations) == 0,
        "violations": violations,
        "checks": checks,
    }


def exact_mcnemar_two_sided_fraction(n10: int, n01: int) -> Fraction:
    """Compute exact two-sided McNemar binomial tail p-value in pure Q."""
    _require_exact_int(n10, "n10", allow_zero=True)
    _require_exact_int(n01, "n01", allow_zero=True)
    total_disc = n10 + n01
    if total_disc == 0:
        return Fraction(1, 1)
    k = min(n10, n01)
    tail_num = sum(math.comb(total_disc, i) for i in range(k + 1))
    two_sided = Fraction(2 * tail_num, 1 << total_disc)
    return Fraction(1, 1) if two_sided > 1 else two_sided


def certify_paired_dual_screen(
    baseline_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    max_cut_regression_p: Fraction,
    min_acc_drop_p: Fraction,
) -> dict[str, Any]:
    """Audit paired candidate vs baseline on BOTH accuracy and truncation (cut_at_32k)."""
    if not isinstance(max_cut_regression_p, Fraction) or max_cut_regression_p <= 0 or max_cut_regression_p > 1:
        raise OracleError("max_cut_regression_p must be a Fraction in (0, 1]")
    if not isinstance(min_acc_drop_p, Fraction) or min_acc_drop_p <= 0 or min_acc_drop_p > 1:
        raise OracleError("min_acc_drop_p must be a Fraction in (0, 1]")

    def _index_rows(rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, tuple[bool, bool]]:
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or len(rows) == 0:
            raise OracleError(f"{label} must be a non-empty sequence")
        out: dict[str, tuple[bool, bool]] = {}
        for idx, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise OracleError(f"{label}[{idx}] must be a mapping")
            if "id" not in row or "correct" not in row or "finish_reason" not in row:
                raise OracleError(f"{label}[{idx}] missing required keys (id, correct, finish_reason)")
            pid = row["id"]
            if type(pid) is not str or not pid.strip():
                raise OracleError(f"{label}[{idx}].id must be non-empty str")
            if pid in out:
                raise OracleError(f"duplicate problem id in {label}: {pid}")
            correct = _require_exact_bool(row["correct"], f"{label}[{idx}].correct")
            fr = row["finish_reason"]
            if type(fr) is not str or fr not in ("stop", "length"):
                raise OracleError(f"{label}[{idx}].finish_reason must be 'stop' or 'length'")
            out[pid] = (correct, fr == "length")
        return out

    base_map = _index_rows(baseline_rows, "baseline_rows")
    cand_map = _index_rows(candidate_rows, "candidate_rows")
    if set(base_map.keys()) != set(cand_map.keys()):
        raise OracleError("paired screen problem ID sets do not match exactly")

    n_total = len(base_map)
    acc_11 = acc_10 = acc_01 = acc_00 = 0
    cut_11 = cut_10 = cut_01 = cut_00 = 0
    cand_finished = 0
    cand_finished_correct = 0

    for pid in sorted(base_map.keys()):
        b_ok, b_cut = base_map[pid]
        c_ok, c_cut = cand_map[pid]
        if c_ok and b_ok:
            acc_11 += 1
        elif c_ok:
            acc_10 += 1
        elif b_ok:
            acc_01 += 1
        else:
            acc_00 += 1

        if c_cut and b_cut:
            cut_11 += 1
        elif c_cut:
            cut_10 += 1
        elif b_cut:
            cut_01 += 1
        else:
            cut_00 += 1

        if not c_cut:
            cand_finished += 1
            if c_ok:
                cand_finished_correct += 1

    conservation_exact = (
        (acc_11 + acc_10 + acc_01 + acc_00 == n_total)
        and (cut_11 + cut_10 + cut_01 + cut_00 == n_total)
    )
    acc_p = exact_mcnemar_two_sided_fraction(acc_10, acc_01)
    cut_p = exact_mcnemar_two_sided_fraction(cut_10, cut_01)

    acc_regressed = (acc_01 > acc_10) and (acc_p <= min_acc_drop_p)
    cut_regressed = (cut_10 > cut_01) and (cut_p <= max_cut_regression_p)

    failed_gates: list[str] = []
    if acc_regressed:
        failed_gates.append("paired_accuracy_regression")
    if cut_regressed:
        failed_gates.append("paired_cut_rate_regression")

    base_acc = Fraction(acc_11 + acc_01, n_total)
    cand_acc = Fraction(acc_11 + acc_10, n_total)
    base_cut = Fraction(cut_11 + cut_01, n_total)
    cand_cut = Fraction(cut_11 + cut_10, n_total)

    return {
        "n_problems": n_total,
        "baseline_correct": acc_11 + acc_01,
        "candidate_correct": acc_11 + acc_10,
        "baseline_acc": frac_to_dict(base_acc),
        "candidate_acc": frac_to_dict(cand_acc),
        "acc_contingency": {"both_ok": acc_11, "cand_only": acc_10, "base_only": acc_01, "neither": acc_00},
        "acc_mcnemar_p_exact": frac_to_dict(acc_p),
        "acc_mcnemar_p_dec": format_exact_decimal(acc_p, 6),
        "baseline_cut": cut_11 + cut_01,
        "candidate_cut": cut_11 + cut_10,
        "baseline_cut_rate": frac_to_dict(base_cut),
        "candidate_cut_rate": frac_to_dict(cand_cut),
        "cut_contingency": {"both_cut": cut_11, "cand_more_cut": cut_10, "cand_less_cut": cut_01, "neither_cut": cut_00},
        "cut_mcnemar_p_exact": frac_to_dict(cut_p),
        "cut_mcnemar_p_dec": format_exact_decimal(cut_p, 6),
        "candidate_finished": cand_finished,
        "candidate_finished_correct": cand_finished_correct,
        "conservation_exact": conservation_exact,
        "passed_dual_gate": len(failed_gates) == 0,
        "failed_gates": failed_gates,
    }


def certify_self_rewriting_session(
    session: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail-closed self-rewriting governance gate bridging arXiv:2611.21885v2 to warp-drive & aletheia v2.0.0."""
    if not isinstance(session, Mapping):
        raise OracleError("session must be a mapping")
    if not isinstance(policy, Mapping):
        raise OracleError("policy must be a mapping")

    req_session = (
        "session_id",
        "sandbox_confined",
        "external_verifier_decides_submission",
        "base_weights_sha256",
        "parent_weights_sha256",
        "session_index",
        "cumulative_edit_claimed",
        "paired_heldout_control_present",
        "heldout_problem_count",
        "practice_overlap_count",
        "collateral_ablation_checked",
        "unnecessary_tensor_count",
        "safety_canary_passed",
        "safety_canary_total",
        "unrelated_kl_shift",
        "has_recurrent_state_banks",
        "state_banks_flushed_and_confirmed",
    )
    for key in req_session:
        if key not in session:
            raise OracleError(f"missing session field: {key}")
    if len(session) != len(req_session):
        raise OracleError("unexpected extra field in session")

    req_policy = (
        "min_heldout_problems",
        "max_practice_overlap",
        "max_unnecessary_tensors",
        "min_safety_canary_rate",
        "max_unrelated_kl_shift",
    )
    for key in req_policy:
        if key not in policy:
            raise OracleError(f"missing policy field: {key}")
    if len(policy) != len(req_policy):
        raise OracleError("unexpected extra field in policy")

    session_id = session["session_id"]
    if type(session_id) is not str or not session_id.strip():
        raise OracleError("session_id must be a non-empty string")

    sandbox_confined = _require_exact_bool(session["sandbox_confined"], "sandbox_confined")
    ext_verifier = _require_exact_bool(
        session["external_verifier_decides_submission"],
        "external_verifier_decides_submission",
    )
    base_sha = _require_sha256(session["base_weights_sha256"], "base_weights_sha256")
    parent_sha = _require_sha256(session["parent_weights_sha256"], "parent_weights_sha256")
    session_idx = _require_exact_int(session["session_index"], "session_index", allow_zero=False)
    cumulative_claimed = _require_exact_bool(session["cumulative_edit_claimed"], "cumulative_edit_claimed")
    paired_ctrl = _require_exact_bool(session["paired_heldout_control_present"], "paired_heldout_control_present")
    heldout_n = _require_exact_int(session["heldout_problem_count"], "heldout_problem_count", allow_zero=True)
    overlap_n = _require_exact_int(session["practice_overlap_count"], "practice_overlap_count", allow_zero=True)
    if overlap_n > heldout_n:
        raise OracleError("practice_overlap_count cannot exceed heldout_problem_count")
    ablation_checked = _require_exact_bool(session["collateral_ablation_checked"], "collateral_ablation_checked")
    extra_tensors = _require_exact_int(session["unnecessary_tensor_count"], "unnecessary_tensor_count", allow_zero=True)
    canary_passed = _require_exact_int(session["safety_canary_passed"], "safety_canary_passed", allow_zero=True)
    canary_total = _require_exact_int(session["safety_canary_total"], "safety_canary_total", allow_zero=True)
    if canary_passed > canary_total:
        raise OracleError("safety_canary_passed cannot exceed safety_canary_total")

    kl_rat = RationalBound.from_dict(session["unrelated_kl_shift"])
    has_ssm = _require_exact_bool(session["has_recurrent_state_banks"], "has_recurrent_state_banks")
    ssm_flushed = _require_exact_bool(
        session["state_banks_flushed_and_confirmed"],
        "state_banks_flushed_and_confirmed",
    )

    min_heldout = _require_exact_int(policy["min_heldout_problems"], "policy.min_heldout_problems", allow_zero=False)
    max_overlap = _require_exact_int(policy["max_practice_overlap"], "policy.max_practice_overlap", allow_zero=True)
    max_extra = _require_exact_int(policy["max_unnecessary_tensors"], "policy.max_unnecessary_tensors", allow_zero=True)
    min_canary = RationalBound.from_dict(policy["min_safety_canary_rate"])
    if min_canary.num > min_canary.den:
        raise OracleError("policy.min_safety_canary_rate cannot exceed 1")
    max_kl = RationalBound.from_dict(policy["max_unrelated_kl_shift"])

    failed_gates: list[str] = []
    if not sandbox_confined:
        failed_gates.append("G1_sandbox_confinement")
    if not ext_verifier:
        failed_gates.append("G2_external_immutable_verifier")
    if cumulative_claimed and session_idx > 1 and hmac.compare_digest(parent_sha, base_sha):
        failed_gates.append("G3_cumulative_lineage_reset_mismatch")
    if (not paired_ctrl) or (heldout_n < min_heldout) or (overlap_n > max_overlap):
        failed_gates.append("G4_paired_clean_heldout")
    if (not ablation_checked) or (extra_tensors > max_extra):
        failed_gates.append("G5_collateral_ablation_bound")
    if canary_total == 0 or (canary_passed * min_canary.den < min_canary.num * canary_total):
        failed_gates.append("G6_safety_canary_floor")
    if kl_rat.num * max_kl.den > max_kl.num * kl_rat.den:
        failed_gates.append("G7_unrelated_kl_latch")
    if has_ssm and not ssm_flushed:
        failed_gates.append("G8_hybrid_state_bank_flush")

    body = {
        "session_id": session_id,
        "accepted": len(failed_gates) == 0,
        "failed_gates": failed_gates,
        "gate_count_failed": len(failed_gates),
        "gate_count_total": 8,
    }
    out = dict(body)
    out["verdict_sha256"] = canonical_sha256(body)
    return out
