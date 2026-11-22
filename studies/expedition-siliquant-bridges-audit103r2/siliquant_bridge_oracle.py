"""Pure-Q & Branchless-Integer Exact Oracle (v1) for Track S6 (AUDIT-103):
Moissanite-Labs/siliquant Force-Multiplier Audit & Exact Operator Bridges.

Governance & Mathematical Discipline (Aletheia v2.0.0):
- Pure Python 3 standard library only; zero floats anywhere on trust paths.
- Zero bare assert statements; fail closed with typed SiliquantBridgeError.
- Strict Part E (exact integer/rational arithmetic & bounds) vs. Part I
  (unbenchmarked real-hardware kernel / serving claims explicitly flagged False).
- Strict Law-4 N0 prior-art quarantine (Conway-Sloane 1982 E8, Rieck 2019
  neural persistence, Turrigiano 1998, Barkai-Leibler 1997, al-Biruni/al-Kashi)
  and strict Law-5 zero-metaphor operator discipline.
"""

from fractions import Fraction
import hashlib
import itertools
import json
import re
import unicodedata


SCHEMA_VERSION = "aletheia.siliquant_bridge_oracle.v1"
AUDIT_ID = "AUDIT-103"
CANONICAL_DECIMALS = 6

FORBIDDEN_METAPHOR_TOKENS = (
    "miracle",
    "quantumconsciousness",
    "magical",
    "numerology",
    "telepathy",
)

NON_ALNUM_RE = re.compile(r"[^a-z0-9]")

# TPU v5e HBM bandwidth: 819 GB/s = 819,000,000,000 bytes/sec (exact integer!)
TPU_V5E_HBM_BYTES_PER_SEC = 819_000_000_000

# Exact rational bits-per-weight (bpw) for each quantization format
FORMAT_BPW = {
    "FP16": Fraction(16, 1),
    "Q4_K_M": Fraction(24, 5),            # 4.80 bpw
    "ZBG128_b560s16": Fraction(9, 2),     # (560 + 16) / 128 = 4.50 bpw (PR #12 winner)
    "E8_Had": Fraction(447, 100),         # 4.47 bpw (PR #24 / Issue #25)
    "INT4": Fraction(4, 1),               # 4.00 bpw ideal floor
}

# Reference parameter counts (Gemma-4-E4B uses nominal rounded 4B placeholder, not exact GGUF tensor sum)
MODEL_PARAMS = {
    "SmolLM2-1.7B": 1_711_376_384,
    "Qwen2.5-1.5B": 1_543_714_304,
    "Llama-3.2-1B": 1_235_814_400,
    "Gemma-4-E4B": 4_000_000_000,         # nominal rounded 4B placeholder
}


class SiliquantBridgeError(ValueError):
    """Typed fail-closed error carrying a stable machine-readable code."""

    def __init__(self, code, message):
        super().__init__(f"{code}: {message}")
        self.code = code


def verify_no_floats(obj, path="root"):
    if isinstance(obj, (float, complex)):
        raise SiliquantBridgeError("FLOAT_FORBIDDEN", f"float/complex at {path}")
    if isinstance(obj, dict):
        for k, v in obj.items():
            verify_no_floats(k, f"{path}.<key>")
            verify_no_floats(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for idx, item in enumerate(obj):
            verify_no_floats(item, f"{path}[{idx}]")


def require_non_empty_str(val, field):
    if not isinstance(val, str):
        raise SiliquantBridgeError("INVALID_TYPE", f"{field} must be str")
    for ch in val:
        code = ord(ch)
        if code < 32 or code == 127 or unicodedata.category(ch) == "Cf":
            raise SiliquantBridgeError("CONTROL_CHAR", f"{field} contains control/format char")
    stripped = val.strip()
    if not stripped:
        raise SiliquantBridgeError("EMPTY_STRING", f"{field} must be non-empty")
    return stripped


def require_exact_bool(val, field):
    if not isinstance(val, bool):
        raise SiliquantBridgeError("INVALID_BOOL", f"{field} must be bool")
    return val


def require_exact_int(val, field, min_value=None, max_value=None):
    if isinstance(val, bool) or not isinstance(val, int):
        raise SiliquantBridgeError("INVALID_INT", f"{field} must be int")
    if min_value is not None and val < min_value:
        raise SiliquantBridgeError("RANGE_ERROR", f"{field} < {min_value}")
    if max_value is not None and val > max_value:
        raise SiliquantBridgeError("RANGE_ERROR", f"{field} > {max_value}")
    return val


def require_exact_fraction(val, field, min_value=None, max_value=None):
    if isinstance(val, bool):
        raise SiliquantBridgeError("INVALID_FRACTION", f"{field} must not be bool")
    if isinstance(val, int):
        out = Fraction(val, 1)
    elif isinstance(val, Fraction):
        out = val
    elif isinstance(val, (list, tuple)) and len(val) == 2:
        num = require_exact_int(val[0], f"{field}[0]")
        den = require_exact_int(val[1], f"{field}[1]", min_value=1)
        out = Fraction(num, den)
    else:
        raise SiliquantBridgeError("INVALID_FRACTION", f"{field} must be Fraction, int, or [num, den]")
    if min_value is not None and out < min_value:
        raise SiliquantBridgeError("RANGE_ERROR", f"{field} < {min_value}")
    if max_value is not None and out > max_value:
        raise SiliquantBridgeError("RANGE_ERROR", f"{field} > {max_value}")
    return out


def frac_str(val):
    f = require_exact_fraction(val, "frac")
    return f"{f.numerator}/{f.denominator}"


def format_exact_decimal(val, places=CANONICAL_DECIMALS):
    f = require_exact_fraction(val, "decimal_val")
    p = require_exact_int(places, "places", min_value=0, max_value=12)
    neg = f < 0
    mag = -f if neg else f
    scale = 10 ** p
    quot, rem = divmod(mag.numerator * scale, mag.denominator)
    if rem * 2 >= mag.denominator:
        quot += 1
    prefix = "-" if (neg and quot > 0) else ""
    if p == 0:
        return f"{prefix}{quot}"
    whole, frac_part = divmod(quot, scale)
    return f"{prefix}{whole}.{frac_part:0{p}d}"


def canonical_json_bytes(payload):
    verify_no_floats(payload, "canonical_json")
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def canonical_sha256(payload):
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def enforce_law5_label_hygiene(labels):
    """Operator label hygiene filter (NOT a mathematical proof)."""
    verify_no_floats(labels, "labels")
    if not isinstance(labels, (list, tuple)) or len(labels) == 0:
        raise SiliquantBridgeError("INVALID_LABELS", "labels must be non-empty sequence")
    seen = set()
    cleaned = []
    for idx, raw in enumerate(labels):
        name = unicodedata.normalize("NFKC", require_non_empty_str(raw, f"labels[{idx}]")).lower()
        compact = NON_ALNUM_RE.sub("", name)
        if not compact:
            raise SiliquantBridgeError("EMPTY_STRING", f"labels[{idx}] has no alnum chars")
        if compact in seen:
            raise SiliquantBridgeError("DUPLICATE_LABEL", f"duplicate label {name}")
        for bad in FORBIDDEN_METAPHOR_TOKENS:
            if bad in compact:
                raise SiliquantBridgeError("METAPHOR_FORBIDDEN", f"forbidden token in {name}")
        seen.add(compact)
        cleaned.append(name)
    return {
        "label_filter_ok": True,
        "guard_scope": "label_filter_only_not_mathematical_proof",
        "label_count": len(cleaned),
        "labels": cleaned,
    }


# =============================================================================
# Pillar 1: Branchless Integer/Bitmask Conway-Sloane E8 Decoder & Zero-Codeword Check
# (siliquant#25 Step 2 & siliquant#16)
# =============================================================================


def _validate_scaled_vec8(x_scaled, scale_shift, pad=0):
    shift = require_exact_int(scale_shift, "scale_shift", min_value=2, max_value=16)
    if not isinstance(x_scaled, (list, tuple)) or len(x_scaled) != 8:
        raise SiliquantBridgeError("INVALID_VEC8", "x_scaled must be 8-element integer sequence")
    limit = (1 << 24) + pad
    vec = [require_exact_int(v, f"x[{i}]", min_value=-limit, max_value=limit) for i, v in enumerate(x_scaled)]
    return vec, shift


def _branchless_d8_core(vec, shift):
    half = 1 << (shift - 1)

    # 1. Branchless round-to-nearest integer: q_i = floor((x_i + S/2) / S) = (x_i + half) >> shift
    q = [(v + half) >> shift for v in vec]

    # 2. Exact scaled rounding errors e_i = x_i - (q_i << shift) in [-S/2, S/2)
    err = [vec[i] - (q[i] << shift) for i in range(8)]

    # 3. Branchless absolute error |e_i| = (e_i ^ mask) - mask, and flip direction step_i in {+1, -1}
    #    neg_mask = e_i >> 60 (-1 if e_i < 0 else 0); step_i = 1 | neg_mask (+1 if e_i >= 0 else -1)
    abs_err = [0] * 8
    step = [0] * 8
    for i in range(8):
        nm = err[i] >> 60
        abs_err[i] = (err[i] ^ nm) - nm
        step[i] = 1 | nm

    # 4. Branchless XOR parity reduction (0 if sum(q) even, 1 if odd)
    parity = (q[0] ^ q[1] ^ q[2] ^ q[3] ^ q[4] ^ q[5] ^ q[6] ^ q[7]) & 1

    # 5. Branchless argmax of |e_i| across 8 lanes via arithmetic shift mask
    best_val = abs_err[0]
    best_idx = 0
    for i in range(1, 8):
        # gt_mask is -1 iff abs_err[i] > best_val (i.e. best_val - abs_err[i] < 0), else 0
        gt_mask = (best_val - abs_err[i]) >> 60
        best_val = (abs_err[i] & gt_mask) | (best_val & ~gt_mask)
        best_idx = (i & gt_mask) | (best_idx & ~gt_mask)

    # 6. Branchless conditional parity flip on lane best_idx only:
    #    diff = i ^ best_idx; eq_mask = ~((diff | -diff) >> 60) is -1 iff i == best_idx, else 0
    par_mask = -parity
    out = [0] * 8
    for i in range(8):
        diff = i ^ best_idx
        eq_mask = ~((diff | -diff) >> 60)
        flip_mask = par_mask & eq_mask
        out[i] = (q[i] + (step[i] & flip_mask)) << shift
    return out


def branchless_d8_decode_int(x_scaled, scale_shift=4):
    """Strictly branchless integer/bitmask Conway-Sloane D8 nearest-point decoder.

    Inputs represent x_i / 2^scale_shift. Returns scaled integer D8 lattice point
    (multiples of 2^scale_shift whose unscaled coordinate sum is even).
    ZERO data-dependent branches: uses arithmetic right-shift sign masks (>> 60),
    XOR parity reduction, and bitwise argmax/conditional-select suitable for SIMD/Pallas
    lowering on **int64 lanes** (intermediates peak at 52 bits; int32 lanes would overflow).
    """
    vec, shift = _validate_scaled_vec8(x_scaled, scale_shift)
    return _branchless_d8_core(vec, shift)


def branchless_e8_decode_int(x_scaled, scale_shift=4):
    """Strictly branchless parallel D8 / (D8 + 1/2*1) Conway-Sloane E8 decoder.

    Returns scaled integer E8 lattice point y_scaled (representing y_scaled / 2^scale_shift)
    and exact squared scaled distance sum((x_scaled[i] - y_scaled[i])^2).
    Requires **int64 lanes** for SIMD/Pallas lowering (intermediate squared distances peak at
    52 bits for inputs in [-2^24, +2^24] with >> 60 sign masks; int32 lanes would overflow).
    """
    vec, shift = _validate_scaled_vec8(x_scaled, scale_shift)
    half = 1 << (shift - 1)

    cand_a = _branchless_d8_core(vec, shift)
    shifted_in = [v - half for v in vec]
    cand_b_raw = _branchless_d8_core(shifted_in, shift)
    cand_b = [v + half for v in cand_b_raw]

    da = sum((vec[i] - cand_a[i]) * (vec[i] - cand_a[i]) for i in range(8))
    db = sum((vec[i] - cand_b[i]) * (vec[i] - cand_b[i]) for i in range(8))

    # Branchless select: use_b_mask is -1 iff db < da (i.e. db - da < 0), else 0
    use_b_mask = (db - da) >> 60
    best = [(cand_b[i] & use_b_mask) | (cand_a[i] & ~use_b_mask) for i in range(8)]
    best_dist = (db & use_b_mask) | (da & ~use_b_mask)
    return best, best_dist


def brute_force_e8_min_dist_sq(x_scaled, scale_shift=4):
    """Independent zero-shared-code exhaustive brute-force E8 nearest-point oracle.

    Exact theorem: for any x in R^8, every L2-nearest point y in D8 satisfies
    y_i in {floor(x_i), floor(x_i) + 1} (otherwise shifting y_i by +-2 toward x_i
    preserves parity and strictly reduces distance). Thus 2^8 = 256 tuples per coset
    exhaustively covers all exact nearest-point candidates.
    """
    vec, shift = _validate_scaled_vec8(x_scaled, scale_shift)
    half = 1 << (shift - 1)

    # Coset A: integer vectors with even sum in {floor(x_i/S), floor(x_i/S)+1}^8
    near_a = [(v >> shift, (v >> shift) + 1) for v in vec]
    best_d = None
    for tup in itertools.product(*near_a):
        if (sum(tup) & 1) != 0:
            continue
        d = sum((vec[i] - (tup[i] << shift)) ** 2 for i in range(8))
        if best_d is None or d < best_d:
            best_d = d

    # Coset B: half-integer vectors (k + 1/2) with even sum(k)
    near_b = [((v - half) >> shift, ((v - half) >> shift) + 1) for v in vec]
    for tup in itertools.product(*near_b):
        if (sum(tup) & 1) != 0:
            continue
        d = sum((vec[i] - ((tup[i] << shift) + half)) ** 2 for i in range(8))
        if best_d is None or d < best_d:
            best_d = d
    return best_d


def is_valid_e8_point_scaled(y_scaled, scale_shift=4):
    """Exact membership predicate for E8 = D8 union (D8 + 1/2*1) in scaled integers."""
    vec, shift = _validate_scaled_vec8(y_scaled, scale_shift, pad=1 << 16)
    unit = 1 << shift
    half = 1 << (shift - 1)
    if all(v % unit == 0 for v in vec):
        return (sum(v // unit for v in vec) & 1) == 0
    if all((v - half) % unit == 0 for v in vec):
        return (sum((v - half) // unit for v in vec) & 1) == 0
    return False


def audit_branchless_e8_and_zero_codeword_exact(scale_shift=4, centre_shift_scaled=None):
    """Pillar 1: Exhaustive branchless E8 optimality check + exact siliquant#16 Code.centre penalty."""
    shift = require_exact_int(scale_shift, "scale_shift", min_value=2, max_value=12)
    unit = 1 << shift
    half = unit >> 1
    quarter = unit >> 2

    # Structured exhaustive adversarial suite covering interior, odd-parity D8, half-integer coset,
    # deep holes, exact boundary ties, negative coordinates, and exact zero!
    test_vectors = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [unit, 0, 0, 0, 0, 0, 0, 0],
        [-unit, 0, 0, 0, 0, 0, 0, 0],
        [half, half, half, half, half, half, half, half],
        [-half, half, -half, half, -half, half, -half, half],
        [half, half, 0, 0, 0, 0, 0, 0],
        [quarter, quarter, quarter, quarter, quarter, quarter, quarter, quarter],
        [unit + quarter, -quarter, half - 1, -(half - 1), 0, 0, 0, 0],
        [unit - 1, unit - 1, unit - 1, 0, 0, 0, 0, 0],
        [-(unit + 3), 5, -7, 11, -13, 17, -19, 23],
        [half + 1, half - 1, -half + 1, -half - 1, 3, -3, 1, -1],
        [3 * half, -half, half, half, 0, 0, 0, 0],
        [2 * unit + 1, -unit - 1, half, -half, quarter, -quarter, 1, -1],
        [7, 7, 7, 7, 7, 7, 7, 7],
        [9, 9, 9, 9, 9, 9, 9, 9],
        [-9, -7, 5, -3, 1, -1, 3, -5],
    ]
    # Add deterministic pseudo-random lattice sweep via pure-integer LCG (32 extra vectors = 48 total)
    state = 2026103
    for _ in range(32):
        row = []
        for _ in range(8):
            state = (6364136223846793005 * state + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
            val = ((state >> 33) % (4 * unit + 1)) - (2 * unit)
            row.append(val)
        test_vectors.append(row)

    exact_matches = 0
    valid_members = 0
    for vec in test_vectors:
        y, d_branchless = branchless_e8_decode_int(vec, shift)
        d_brute = brute_force_e8_min_dist_sq(vec, shift)
        if is_valid_e8_point_scaled(y, shift):
            valid_members += 1
        if d_branchless == d_brute:
            exact_matches += 1

    # Exact zero-codeword preservation check
    y_zero, d_zero = branchless_e8_decode_int([0] * 8, shift)
    zero_preserved = (y_zero == [0] * 8) and (d_zero == 0)

    # Exact siliquant#16 Code.centre shift penalty:
    # Default centre shift vector with ||c||_2 ~ 0.82 steps: at shift=4 (unit=16), c = (5,5,5,5,4,4,4,4)
    # -> ||c/16||_2^2 = (4*25 + 4*16)/256 = 164/256 = 41/64 = 0.640625 (||c||_2 = 0.8004 steps ~ 0.82!)
    if centre_shift_scaled is None:
        c_vec = [5, 5, 5, 5, 4, 4, 4, 4]
    else:
        c_vec, _ = _validate_scaled_vec8(centre_shift_scaled, shift)

    # On an outlier-dominated row with 1 large block and 63 zero blocks (scale step Delta = 1):
    # Quantizing zero block x=0 with shifted codebook: y = decode(0 + c) - c -> error = ||y||_2^2 = ||decode(c) - c||_2^2
    _, d_centre_zero_sq_scaled = branchless_e8_decode_int(c_vec, shift)
    centre_zero_block_mse = Fraction(d_centre_zero_sq_scaled, unit * unit)
    # 63 zero blocks in a 64-block (512-weight) outlier row: total zero-floor error = 63 * centre_zero_block_mse vs 0/1!
    n_zero_blocks = 63
    shifted_row_floor_sq_err = n_zero_blocks * centre_zero_block_mse

    all_optimal = exact_matches == len(test_vectors) and valid_members == len(test_vectors)
    centre_shift_harms_zeros = shifted_row_floor_sq_err > 0 and d_zero == 0

    return {
        "scale_shift": shift,
        "vectors_tested": len(test_vectors),
        "exact_brute_force_matches": exact_matches,
        "valid_e8_membership_count": valid_members,
        "zero_vector_preserved": zero_preserved,
        "centre_shift_vector_scaled": c_vec,
        "centre_zero_block_sq_err_exact": frac_str(centre_zero_block_mse),
        "centre_zero_block_sq_err_dec6": format_exact_decimal(centre_zero_block_mse),
        "shifted_63_zero_blocks_sq_err_exact": frac_str(shifted_row_floor_sq_err),
        "shifted_63_zero_blocks_sq_err_dec6": format_exact_decimal(shifted_row_floor_sq_err),
        "zero_preserving_63_zero_blocks_sq_err_exact": "0/1",
        "all_branchless_optimal": all_optimal,
        "centre_shift_harms_zero_blocks": centre_shift_harms_zeros,
        "passed": all_optimal and zero_preserved and centre_shift_harms_zeros,
    }


# =============================================================================
# Pillar 2: Exact Pure-Q TPU v5e Roofline Oracle (Part E Exact vs Part I Flagged)
# (siliquant#25 Step 3 & siliquant#23)
# =============================================================================


def audit_tpu_v5e_roofline_exact(
    hbm_bytes_per_sec=TPU_V5E_HBM_BYTES_PER_SEC,
    formats=None,
    models=None,
):
    """Pillar 2: Part-E exact rational bandwidth/bytes-per-token/decode-ceiling arithmetic.

    Note on nominal parameter counts: default Qwen2.5-1.5B uses 1,543,714,304 parameters,
    while Gemma-4-E4B uses the nominal rounded 4,000,000,000 placeholder count (not an exact
    GGUF tensor sum). Measured kernel tok/s is Part I (measured_kernel_tok_s_present=False).
    """
    bw = require_exact_int(hbm_bytes_per_sec, "hbm_bytes_per_sec", min_value=1)
    fmts = FORMAT_BPW if formats is None else formats
    mdls = MODEL_PARAMS if models is None else models
    verify_no_floats({"fmts": fmts, "mdls": mdls}, "pillar2_roofline")
    if not isinstance(fmts, dict) or "FP16" not in fmts or "Q4_K_M" not in fmts or "ZBG128_b560s16" not in fmts or "E8_Had" not in fmts:
        raise SiliquantBridgeError("INVALID_FORMATS", "formats must include FP16, Q4_K_M, ZBG128_b560s16, E8_Had")
    if not isinstance(mdls, dict) or len(mdls) == 0:
        raise SiliquantBridgeError("INVALID_MODELS", "models must be non-empty dict")

    clean_fmts = {}
    for fk, bv in fmts.items():
        name = require_non_empty_str(fk, "format_name")
        if fk != name:
            raise SiliquantBridgeError("INVALID_FORMATS", f"format key has leading/trailing whitespace: {fk!r}")
        bpw = require_exact_fraction(bv, f"bpw[{name}]")
        if bpw <= 0 or bpw > Fraction(32, 1):
            raise SiliquantBridgeError("RANGE_ERROR", f"bpw[{name}] out of (0, 32]")
        clean_fmts[name] = bpw

    clean_mdls = {}
    for mk, pv in mdls.items():
        mname = require_non_empty_str(mk, "model_name")
        if mk != mname:
            raise SiliquantBridgeError("INVALID_MODELS", f"model key has leading/trailing whitespace: {mk!r}")
        clean_mdls[mname] = require_exact_int(pv, f"params[{mname}]", min_value=1)

    fp16_bpw = clean_fmts["FP16"]
    q4km_bpw = clean_fmts["Q4_K_M"]
    zbg_bpw = clean_fmts["ZBG128_b560s16"]
    e8_bpw = clean_fmts["E8_Had"]

    format_ratios = {}
    for fname, bpw in sorted(clean_fmts.items()):
        io_red = Fraction(1, 1) - (bpw / fp16_bpw)
        vs_q4km = q4km_bpw / bpw
        format_ratios[fname] = {
            "bpw_exact": frac_str(bpw),
            "bpw_dec6": format_exact_decimal(bpw),
            "hbm_io_reduction_vs_fp16_exact": frac_str(io_red),
            "hbm_io_reduction_vs_fp16_dec6": format_exact_decimal(io_red),
            "bandwidth_speedup_vs_q4km_exact": frac_str(vs_q4km),
            "bandwidth_speedup_vs_q4km_dec6": format_exact_decimal(vs_q4km),
        }

    per_model = {}
    for mname, params in sorted(clean_mdls.items()):
        m_entry = {}
        for fname, bpw in sorted(clean_fmts.items()):
            bytes_per_tok = Fraction(params, 1) * bpw / Fraction(8, 1)
            tok_s_ceil = Fraction(bw, 1) / bytes_per_tok
            m_entry[fname] = {
                "bytes_per_token_exact": frac_str(bytes_per_tok),
                "bytes_per_token_dec6": format_exact_decimal(bytes_per_tok),
                "decode_tok_s_ceiling_dec6": format_exact_decimal(tok_s_ceil),
            }
        per_model[mname] = m_entry

    # Exact ordering invariants
    zbg_and_e8_beat_q4km_bw = zbg_bpw < q4km_bpw and e8_bpw < q4km_bpw
    e8_and_zbg_cut_hbm_over_70pct = (
        (Fraction(1, 1) - zbg_bpw / fp16_bpw) > Fraction(7, 10)
        and (Fraction(1, 1) - e8_bpw / fp16_bpw) > Fraction(7, 10)
    )

    return {
        "claim_scope": "part_e_exact_arithmetic_roofline_only_not_measured_kernel_tok_s",
        "measured_kernel_tok_s_present": False,
        "hbm_bytes_per_sec": bw,
        "format_ratios": format_ratios,
        "per_model_ceilings": per_model,
        "zbg_and_e8_beat_q4km_bandwidth": zbg_and_e8_beat_q4km_bw,
        "hbm_reduction_exceeds_70pct": e8_and_zbg_cut_hbm_over_70pct,
        "passed": zbg_and_e8_beat_q4km_bw and e8_and_zbg_cut_hbm_over_70pct,
    }


# =============================================================================
# Pillar 3: Track S4 Nature Mechanics -> siliquant Exact Operator Bridges
# (3a Homeostatic Scaling + Integral Calibration; 3b R_eff vs persist/ MST)
# =============================================================================


def _solve_linear_exact(mat_a, rhs_b):
    verify_no_floats({"a": mat_a, "b": rhs_b}, "solve_linear")
    m = len(mat_a)
    aug = [
        [require_exact_fraction(mat_a[i][j], f"A[{i}][{j}]") for j in range(m)]
        + [require_exact_fraction(rhs_b[i], f"b[{i}]")]
        for i in range(m)
    ]
    for col in range(m):
        piv = None
        for r in range(col, m):
            if aug[r][col] != 0:
                piv = r
                break
        if piv is None:
            raise SiliquantBridgeError("SINGULAR_MATRIX", f"singular system at col {col}")
        if piv != col:
            aug[col], aug[piv] = aug[piv], aug[col]
        pv = aug[col][col]
        for c in range(col, m + 1):
            aug[col][c] /= pv
        for r in range(m):
            if r != col:
                fac = aug[r][col]
                for c in range(col, m + 1):
                    aug[r][c] -= fac * aug[col][c]
    return [aug[i][m] for i in range(m)]


def _validate_weight_matrix_exact(weight_matrix):
    verify_no_floats(weight_matrix, "weight_matrix")
    if not isinstance(weight_matrix, (list, tuple)) or len(weight_matrix) == 0:
        raise SiliquantBridgeError("INVALID_MATRIX", "weight_matrix must be non-empty")
    a = len(weight_matrix)
    b = len(weight_matrix[0]) if isinstance(weight_matrix[0], (list, tuple)) else 0
    if b == 0:
        raise SiliquantBridgeError("INVALID_MATRIX", "weight_matrix rows must be non-empty")
    clean = []
    for i in range(a):
        row = weight_matrix[i]
        if not isinstance(row, (list, tuple)) or len(row) != b:
            raise SiliquantBridgeError("INVALID_MATRIX", "weight_matrix must be rectangular")
        clean.append([abs(require_exact_fraction(row[j], f"W[{i}][{j}]")) for j in range(b)])
    return clean, a, b


def bipartite_mst_weights_exact(weight_matrix):
    """Exact pure-Q Kruskal Maximum Spanning Forest weights over positive edges of |W|."""
    clean, a, b = _validate_weight_matrix_exact(weight_matrix)
    edges = []
    for i in range(a):
        for j in range(b):
            if clean[i][j] > 0:
                edges.append((clean[i][j], i, j))
    edges.sort(key=lambda item: (-item[0], item[1], item[2]))
    parent = list(range(a + b))

    def _find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    chosen = []
    for w, i, j in edges:
        ri, rj = _find(i), _find(a + j)
        if ri != rj:
            parent[ri] = rj
            chosen.append(w)
            if len(chosen) == a + b - 1:
                break
    return chosen


def bipartite_effective_resistance_exact(weight_matrix, row_u=0, col_v=0):
    """Exact Kirchhoff effective resistance R_eff(row_u, col_v) on connected component of |W|."""
    clean, a, b = _validate_weight_matrix_exact(weight_matrix)
    ru = require_exact_int(row_u, "row_u", 0, a - 1)
    cv = require_exact_int(col_v, "col_v", 0, b - 1)
    n = a + b
    ground = a + cv
    adj = [[] for _ in range(n)]
    for i in range(a):
        for j in range(b):
            w = clean[i][j]
            if w > 0:
                u = i
                v = a + j
                adj[u].append((v, w))
                adj[v].append((u, w))
    # Restrict Laplacian strictly to the connected component containing ru (GPT6-SQ-02)
    visited = {ru}
    queue = [ru]
    while queue:
        cur = queue.pop()
        for nb, _ in adj[cur]:
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
    if ground not in visited:
        raise SiliquantBridgeError("SINGULAR_MATRIX", "queried endpoints are disconnected")
    free = sorted(v for v in visited if v != ground)
    idx_map = {v: idx for idx, v in enumerate(free)}
    m = len(free)
    lap_red = [[Fraction(0, 1) for _ in range(m)] for _ in range(m)]
    for u in free:
        iu = idx_map[u]
        for nb, w in adj[u]:
            lap_red[iu][iu] += w
            if nb != ground:
                lap_red[iu][idx_map[nb]] -= w
    rhs = [Fraction(1, 1) if v == ru else Fraction(0, 1) for v in free]
    p_red = _solve_linear_exact(lap_red, rhs)
    return p_red[idx_map[ru]]


def _homeostatic_scale_exact(weights, gain):
    return [gain * x for x in weights]


def _proportional_droop_exact(alpha, du, kp):
    return (alpha * du) / (Fraction(1, 1) + alpha * kp)


def audit_nature_to_siliquant_bridges_exact(
    homeo_gain=Fraction(7, 3),
    add_shift=Fraction(1, 2),
    step_u=Fraction(1, 4),
    kp_prop=Fraction(2, 1),
    ki_int=Fraction(2, 1),
    w_base_override=None,
):
    """Pillar 3: Homeostatic ratio invariance + integral scale calibration + R_eff vs persist/ MST."""
    verify_no_floats(
        {"g": homeo_gain, "s": add_shift, "u": step_u, "kp": kp_prop, "ki": ki_int, "wb": w_base_override},
        "pillar3",
    )
    g = require_exact_fraction(homeo_gain, "homeo_gain")
    s_add = require_exact_fraction(add_shift, "add_shift")
    du = require_exact_fraction(step_u, "step_u")
    kp = require_exact_fraction(kp_prop, "kp_prop")
    ki = require_exact_fraction(ki_int, "ki_int")
    if g <= 0 or s_add == 0 or du <= 0 or kp <= 0 or ki <= 0:
        raise SiliquantBridgeError("INVALID_PARAMS", "invalid Pillar 3 parameters")

    # 3a-1: Synaptic homeostatic multiplicative scaling vs additive Code.centre shift
    if w_base_override is None:
        w_base = [Fraction(4, 1), Fraction(2, 1), Fraction(-1, 1), Fraction(1, 2)]
    else:
        if not isinstance(w_base_override, (list, tuple)) or len(w_base_override) != 4:
            raise SiliquantBridgeError("INVALID_PARAMS", "w_base_override must have length 4")
        w_base = [require_exact_fraction(x, "w_base") for x in w_base_override]
    if w_base[0] == 0 or (w_base[0] + s_add) == 0:
        raise SiliquantBridgeError("INVALID_PARAMS", "w_base[0] and w_base[0]+add_shift must be non-zero")
    w_mult = _homeostatic_scale_exact(w_base, g)
    w_shift = [x + s_add for x in w_base]
    ratios_base = [w_base[i] / w_base[0] for i in range(1, 4)]
    ratios_mult = [w_mult[i] / w_mult[0] for i in range(1, 4)]
    ratios_shift = [w_shift[i] / w_shift[0] for i in range(1, 4)]
    mult_ratio_max_err = max(abs(a - b) for a, b in zip(ratios_base, ratios_mult))
    shift_ratio_max_err = max(abs(a - b) for a, b in zip(ratios_base, ratios_shift))
    homeostatic_invariance_exact = mult_ratio_max_err == 0 and shift_ratio_max_err > 0

    # 3a-2: Barkai-Leibler discrete integral scale calibration vs proportional clipping droop
    # Plant r_t - r* = alpha * (du - z_t), integrator z_{t+1} = z_t + ki * (r_t - r*), alpha = 1/2:
    # At deadbeat ki = 1/alpha = 2, step 1 achieves exact 0/1 overload-rate error!
    alpha = Fraction(1, 2)
    z_int = Fraction(0, 1)
    err_integral_deadbeat = Fraction(0, 1)
    for _ in range(2):
        err_integral_deadbeat = alpha * (du - z_int)
        z_int = z_int + ki * err_integral_deadbeat
    prop_droop = _proportional_droop_exact(alpha, du, kp)
    integral_beats_prop_droop = err_integral_deadbeat == 0 and prop_droop > 0

    # 3b: Exact Kirchhoff-Schur R_eff vs siliquant/persist/ MST persistence on 3x3 bipartite layers
    # Both W_tree and W_redundant share the EXACT 5-edge Maximum Spanning Tree of weight [5,5,5,5,5]!
    # Tree edges: (0,0), (0,1), (1,1), (1,2), (2,2) all = 5. Non-tree entries: 0 in W_tree, 2 in W_redundant!
    w_tree = [
        [Fraction(5, 1), Fraction(5, 1), Fraction(0, 1)],
        [Fraction(0, 1), Fraction(5, 1), Fraction(5, 1)],
        [Fraction(0, 1), Fraction(0, 1), Fraction(5, 1)],
    ]
    w_redundant = [
        [Fraction(5, 1), Fraction(5, 1), Fraction(2, 1)],
        [Fraction(2, 1), Fraction(5, 1), Fraction(5, 1)],
        [Fraction(2, 1), Fraction(2, 1), Fraction(5, 1)],
    ]
    mst_tree = bipartite_mst_weights_exact(w_tree)
    mst_red = bipartite_mst_weights_exact(w_redundant)
    mst_blind_to_parallel = mst_tree == mst_red

    reff_tree = bipartite_effective_resistance_exact(w_tree, row_u=0, col_v=2)
    reff_red = bipartite_effective_resistance_exact(w_redundant, row_u=0, col_v=2)
    reff_detects_parallel = reff_red < reff_tree

    # Cutting bridge edge (1, 1) disconnects W_tree (singular!), while W_redundant stays connected with finite R_eff!
    w_tree_cut = [list(row) for row in w_tree]
    w_tree_cut[1][1] = Fraction(0, 1)
    tree_cut_disconnected = False
    try:
        bipartite_effective_resistance_exact(w_tree_cut, row_u=0, col_v=2)
    except SiliquantBridgeError as err:
        if err.code == "SINGULAR_MATRIX":
            tree_cut_disconnected = True

    w_red_cut = [list(row) for row in w_redundant]
    w_red_cut[1][1] = Fraction(0, 1)
    reff_red_cut = bipartite_effective_resistance_exact(w_red_cut, row_u=0, col_v=2)
    redundant_survives_cut = reff_red_cut > reff_red

    passed = (
        homeostatic_invariance_exact
        and integral_beats_prop_droop
        and mst_blind_to_parallel
        and reff_detects_parallel
        and tree_cut_disconnected
        and redundant_survives_cut
    )
    return {
        "homeostatic_mult_ratio_max_err_exact": frac_str(mult_ratio_max_err),
        "additive_shift_ratio_max_err_exact": frac_str(shift_ratio_max_err),
        "integral_calibration_deadbeat_err_exact": frac_str(err_integral_deadbeat),
        "proportional_clipping_droop_exact": frac_str(prop_droop),
        "proportional_clipping_droop_dec6": format_exact_decimal(prop_droop),
        "mst_weights_tree_exact": [frac_str(v) for v in mst_tree],
        "mst_weights_redundant_exact": [frac_str(v) for v in mst_red],
        "mst_blind_to_parallel_redundancy": mst_blind_to_parallel,
        "reff_tree_exact": frac_str(reff_tree),
        "reff_tree_dec6": format_exact_decimal(reff_tree),
        "reff_redundant_exact": frac_str(reff_red),
        "reff_redundant_dec6": format_exact_decimal(reff_red),
        "reff_redundant_after_cut_exact": frac_str(reff_red_cut),
        "reff_redundant_after_cut_dec6": format_exact_decimal(reff_red_cut),
        "tree_cut_disconnected_control": tree_cut_disconnected,
        "redundant_survives_cut": redundant_survives_cut,
        "homeostatic_invariance_exact": homeostatic_invariance_exact,
        "integral_beats_prop_droop": integral_beats_prop_droop,
        "reff_detects_parallel": reff_detects_parallel,
        "passed": passed,
    }


# =============================================================================
# Pillar 4: Bayt al-Hikma (al-Biruni Cancellation & al-Kashi Contraction) -> siliquant
# =============================================================================


def _kashi_iterate_exact(s_star, s_init, lip, n_steps):
    s = s_init
    for _ in range(n_steps):
        s = s_star + lip * (s - s_star)
    return s


def audit_bayt_al_hikma_quant_bounds_exact(
    perturb_eps=Fraction(1, 4),
    contraction_lip=Fraction(1, 2),
    divergent_lip=Fraction(3, 2),
    steps=8,
    w_cancel_override=None,
    delta_cancel_override=None,
    init_s=Fraction(3, 1),
):
    """Pillar 4: Exact al-Biruni dot-product cancellation amplification + al-Kashi scale contraction.

    Exact equality amplification_ratio == kappa_cancel / kappa_align holds on matched-L1-norm
    fixtures (here ||w_cancel||_1 = ||w_align||_1 = 21 with uniform |delta_i| = eps); on
    unmatched-L1-norm vectors, ratio = (kappa_cancel / kappa_align) * (||w_align||_1 / ||w_cancel||_1).
    """
    verify_no_floats(
        {
            "e": perturb_eps,
            "c": contraction_lip,
            "d": divergent_lip,
            "s": steps,
            "wc": w_cancel_override,
            "dc": delta_cancel_override,
            "is": init_s,
        },
        "pillar4_bah",
    )
    eps = require_exact_fraction(perturb_eps, "perturb_eps")
    lip_c = require_exact_fraction(contraction_lip, "contraction_lip")
    lip_d = require_exact_fraction(divergent_lip, "divergent_lip")
    n_steps = require_exact_int(steps, "steps", min_value=2, max_value=40)
    s0 = require_exact_fraction(init_s, "init_s")
    if eps <= 0 or lip_c <= 0 or lip_c >= 1 or lip_d <= 1:
        raise SiliquantBridgeError("INVALID_PARAMS", "invalid Pillar 4 parameters")

    # al-Biruni cancellation condition number: kappa(w, a) = sum(|w_i||a_i|) / |sum(w_i a_i)|
    if w_cancel_override is None:
        w_cancel = [Fraction(10, 1), Fraction(-10, 1), Fraction(1, 1)]
    else:
        if not isinstance(w_cancel_override, (list, tuple)) or len(w_cancel_override) != 3:
            raise SiliquantBridgeError("INVALID_PARAMS", "w_cancel_override must have length 3")
        w_cancel = [require_exact_fraction(x, "w_cancel") for x in w_cancel_override]
    w_align = [Fraction(10, 1), Fraction(10, 1), Fraction(1, 1)]
    act = [Fraction(1, 1), Fraction(1, 1), Fraction(1, 1)]
    # Worst-case sign-aligned perturbation delta_w of magnitude eps on every entry
    if delta_cancel_override is None:
        delta_cancel = [eps, eps, eps]
    else:
        if not isinstance(delta_cancel_override, (list, tuple)) or len(delta_cancel_override) != 3:
            raise SiliquantBridgeError("INVALID_PARAMS", "delta_cancel_override must have length 3")
        delta_cancel = [require_exact_fraction(x, "delta_cancel") for x in delta_cancel_override]
    delta_align = [eps, eps, eps]

    y_cancel = sum(w * a for w, a in zip(w_cancel, act))
    if y_cancel == 0:
        raise SiliquantBridgeError("INVALID_PARAMS", "w_cancel dot act must be non-zero")
    y_align = sum(w * a for w, a in zip(w_align, act))
    kappa_cancel = sum(abs(w * a) for w, a in zip(w_cancel, act)) / abs(y_cancel)
    kappa_align = sum(abs(w * a) for w, a in zip(w_align, act)) / abs(y_align)

    rel_err_cancel = abs(sum(dw * a for dw, a in zip(delta_cancel, act))) / abs(y_cancel)
    rel_err_align = abs(sum(dw * a for dw, a in zip(delta_align, act))) / abs(y_align)
    # Exact upper bound: rel_err <= kappa * (eps / mean_|w|) -> exact amplification ratio equals kappa_cancel / kappa_align!
    amplification_ratio = rel_err_cancel / rel_err_align
    biruni_amplification_exact = amplification_ratio == kappa_cancel / kappa_align and kappa_cancel > kappa_align

    # al-Kashi iterative fixed-point contraction for scale calibration s_{t+1} = s* + L * (s_t - s*)
    s_star = Fraction(1, 1)
    init_err = abs(s0 - s_star)
    s_c = _kashi_iterate_exact(s_star, s0, lip_c, n_steps)
    final_err_c = abs(s_c - s_star)
    expected_err_c = (lip_c ** n_steps) * init_err

    s_d = _kashi_iterate_exact(s_star, s0, lip_d, n_steps)
    final_err_d = abs(s_d - s_star)

    kashi_contraction_exact = final_err_c == expected_err_c and final_err_c < init_err
    kashi_divergence_control = final_err_d > init_err

    return {
        "kappa_cancelling_exact": frac_str(kappa_cancel),
        "kappa_aligned_exact": frac_str(kappa_align),
        "rel_err_cancelling_exact": frac_str(rel_err_cancel),
        "rel_err_cancelling_dec6": format_exact_decimal(rel_err_cancel),
        "rel_err_aligned_exact": frac_str(rel_err_align),
        "rel_err_aligned_dec6": format_exact_decimal(rel_err_align),
        "amplification_ratio_exact": frac_str(amplification_ratio),
        "kashi_init_err_exact": frac_str(init_err),
        "kashi_contracted_err_exact": frac_str(final_err_c),
        "kashi_divergent_err_exact": frac_str(final_err_d),
        "biruni_amplification_exact": biruni_amplification_exact,
        "kashi_contraction_exact": kashi_contraction_exact,
        "kashi_divergence_control": kashi_divergence_control,
        "passed": biruni_amplification_exact and kashi_contraction_exact and kashi_divergence_control,
    }


# =============================================================================
# Pillar 5: Cryptographic Provenance & Live siliquant Repo Verification Gate
# =============================================================================


def audit_siliquant_repo_diagnostics_exact(repo_diag):
    """Validate live Moissanite-Labs/siliquant PR #12 & Draft PR #24/#26/#27 diagnostics."""
    verify_no_floats(repo_diag, "repo_diag")
    if not isinstance(repo_diag, dict):
        raise SiliquantBridgeError("INVALID_DIAG", "repo_diag must be dict")
    req_keys = {
        "diag_source",
        "main_commit",
        "pr12_head_commit",
        "pr12_shape_tests_passed",
        "pr12_perm_tests_passed",
        "pr12_persist_tests_passed",
        "pr12_lattice_tests_executed_in_live_check",
        "pr12_design_out_has_smollm2_fix",
        "pr12_number_trace_exact_match",
        "pr12_claim_ledger_exact_match",
        "pr12_plan_before_data_exit_zero",
        "draft_pr24_head",
        "draft_pr26_head",
        "draft_pr27_head",
    }
    if set(repo_diag.keys()) != req_keys:
        raise SiliquantBridgeError("INVALID_DIAG", "repo_diag keys mismatch")

    diag_src = require_non_empty_str(repo_diag["diag_source"], "diag_source")
    if diag_src not in ("live_git_inspection", "standalone_fixture"):
        raise SiliquantBridgeError("INVALID_DIAG", f"unsupported diag_source {diag_src}")

    commits = {}
    for ck in ("main_commit", "pr12_head_commit", "draft_pr24_head", "draft_pr26_head", "draft_pr27_head"):
        raw_cv = repo_diag[ck]
        cv = require_non_empty_str(raw_cv, ck)
        if raw_cv != cv or not re.fullmatch(r"[0-9a-f]{40}", cv) or len(set(cv)) == 1:
            raise SiliquantBridgeError("INVALID_COMMIT", f"{ck} must be clean non-degenerate 40-char lowercase hex git SHA")
        commits[ck] = cv
    if len(set(commits.values())) != 5:
        raise SiliquantBridgeError("DUPLICATE_COMMIT", "all 5 tracked branch commit SHAs must be pairwise distinct")

    flags = {}
    for fk in (
        "pr12_shape_tests_passed",
        "pr12_perm_tests_passed",
        "pr12_persist_tests_passed",
        "pr12_design_out_has_smollm2_fix",
        "pr12_number_trace_exact_match",
        "pr12_claim_ledger_exact_match",
        "pr12_plan_before_data_exit_zero",
    ):
        flags[fk] = require_exact_bool(repo_diag[fk], fk)
    lattice_exec = require_exact_bool(
        repo_diag["pr12_lattice_tests_executed_in_live_check"],
        "pr12_lattice_tests_executed_in_live_check",
    )

    pr12_all_green = (
        flags["pr12_shape_tests_passed"]
        and flags["pr12_perm_tests_passed"]
        and flags["pr12_persist_tests_passed"]
        and flags["pr12_design_out_has_smollm2_fix"]
        and flags["pr12_number_trace_exact_match"]
        and flags["pr12_claim_ledger_exact_match"]
        and flags["pr12_plan_before_data_exit_zero"]
    )
    out = {"diag_source": diag_src}
    out.update(commits)
    out.update(flags)
    out["pr12_lattice_tests_executed_in_live_check"] = lattice_exec
    out["live_executed_test_suites"] = ["shape", "perm", "persist", "lattice"] if lattice_exec else ["shape", "perm", "persist"]
    out["pr12_all_checks_green"] = pr12_all_green
    out["passed"] = pr12_all_green
    return out


def _strict_equal(a, b):
    """Type-preserving recursive equality (distinguishes True/1, False/0, int/Fraction)."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return set(a.keys()) == set(b.keys()) and all(_strict_equal(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_strict_equal(x, y) for x, y in zip(a, b))
    return a == b


def certify_siliquant_bridge_audit_exact(
    mode,
    live_repo_recounted,
    operator_labels,
    repo_diag_result,
    pillar1_result,
    pillar2_result,
    pillar3_result,
    pillar4_result,
    five_axis_checks,
):
    """Fail-closed cryptographic certificate re-deriving all 5 pillars from inputs.

    Honest provenance scope: this offline function validates schema integrity, exact
    re-derivation of all 4 pure-Q pillars, and consistency between mode, live_repo_recounted,
    and repo_diag_result['diag_source']. Actual live git provenance is established strictly
    by invoking siliquant_bridge_audit.py --live (compute_live_siliquant_diagnostics).
    """
    verify_no_floats(
        {
            "mode": mode,
            "live": live_repo_recounted,
            "labels": operator_labels,
            "diag": repo_diag_result,
            "p1": pillar1_result,
            "p2": pillar2_result,
            "p3": pillar3_result,
            "p4": pillar4_result,
            "axes": five_axis_checks,
        },
        "certify",
    )
    clean_mode = require_non_empty_str(mode, "mode")
    if clean_mode not in ("live_repo_recount", "standalone_fixture_replay"):
        raise SiliquantBridgeError("INVALID_MODE", f"unsupported mode {clean_mode}")
    live_flag = require_exact_bool(live_repo_recounted, "live_repo_recounted")
    if live_flag != (clean_mode == "live_repo_recount"):
        raise SiliquantBridgeError("MODE_MISMATCH", "live_repo_recounted contradicts mode")

    label_guard = enforce_law5_label_hygiene(operator_labels)

    req_axes = (
        "pr12_live_reproducibility",
        "branchless_e8_and_zero_codeword",
        "tpu_v5e_part_e_roofline",
        "nature_homeostasis_and_kirchhoff_vs_mst",
        "bayt_al_hikma_cancellation_and_contraction",
    )
    if not isinstance(five_axis_checks, dict) or set(five_axis_checks.keys()) != set(req_axes):
        raise SiliquantBridgeError("INVALID_AXES", "five_axis_checks keys mismatch")
    clean_axes = {ax: require_exact_bool(five_axis_checks[ax], f"axes.{ax}") for ax in req_axes}

    if (
        not isinstance(repo_diag_result, dict)
        or not isinstance(pillar1_result, dict)
        or not isinstance(pillar2_result, dict)
        or not isinstance(pillar3_result, dict)
        or not isinstance(pillar4_result, dict)
    ):
        raise SiliquantBridgeError("INVALID_PILLAR", "all pillar results must be dicts")

    # Full cryptographic re-derivation of every pillar
    raw_diag = {
        k: repo_diag_result.get(k)
        for k in (
            "diag_source",
            "main_commit",
            "pr12_head_commit",
            "pr12_shape_tests_passed",
            "pr12_perm_tests_passed",
            "pr12_persist_tests_passed",
            "pr12_lattice_tests_executed_in_live_check",
            "pr12_design_out_has_smollm2_fix",
            "pr12_number_trace_exact_match",
            "pr12_claim_ledger_exact_match",
            "pr12_plan_before_data_exit_zero",
            "draft_pr24_head",
            "draft_pr26_head",
            "draft_pr27_head",
        )
    }
    re_diag = audit_siliquant_repo_diagnostics_exact(raw_diag)
    if not _strict_equal(re_diag, repo_diag_result):
        raise SiliquantBridgeError("PILLAR_CONTRADICTION", "repo_diag_result contradicts re-derivation")
    expected_src = "live_git_inspection" if live_flag else "standalone_fixture"
    if re_diag["diag_source"] != expected_src:
        raise SiliquantBridgeError("PROVENANCE_MISMATCH", f"diag_source {re_diag['diag_source']} contradicts mode {clean_mode}")

    re_p1 = audit_branchless_e8_and_zero_codeword_exact()
    if not _strict_equal(re_p1, pillar1_result):
        raise SiliquantBridgeError("PILLAR_CONTRADICTION", "pillar1_result contradicts re-derivation")

    re_p2 = audit_tpu_v5e_roofline_exact()
    if not _strict_equal(re_p2, pillar2_result):
        raise SiliquantBridgeError("PILLAR_CONTRADICTION", "pillar2_result contradicts re-derivation")

    re_p3 = audit_nature_to_siliquant_bridges_exact()
    if not _strict_equal(re_p3, pillar3_result):
        raise SiliquantBridgeError("PILLAR_CONTRADICTION", "pillar3_result contradicts re-derivation")

    re_p4 = audit_bayt_al_hikma_quant_bounds_exact()
    if not _strict_equal(re_p4, pillar4_result):
        raise SiliquantBridgeError("PILLAR_CONTRADICTION", "pillar4_result contradicts re-derivation")

    expected_axes = {
        "pr12_live_reproducibility": re_diag["passed"],
        "branchless_e8_and_zero_codeword": re_p1["passed"],
        "tpu_v5e_part_e_roofline": re_p2["passed"],
        "nature_homeostasis_and_kirchhoff_vs_mst": re_p3["passed"],
        "bayt_al_hikma_cancellation_and_contraction": re_p4["passed"],
    }
    if clean_axes != expected_axes:
        raise SiliquantBridgeError("AXIS_CONTRADICTION", "five_axis_checks contradicts re-derived pillars")

    all_passed = all(clean_axes.values())
    body = {
        "schema_version": SCHEMA_VERSION,
        "audit_id": AUDIT_ID,
        "mode": clean_mode,
        "live_repo_recounted": live_flag,
        "novelty_classification": "N0_prior_art_operators_with_N1_branchless_e8_and_live_siliquant_audit",
        "law5_label_hygiene_guard": label_guard,
        "five_axis_audit_checks": clean_axes,
        "all_five_axes_passed": all_passed,
        "live_siliquant_repo_diagnostics": re_diag,
        "pillar1_branchless_e8_and_zero_codeword": re_p1,
        "pillar2_tpu_v5e_roofline_part_e": re_p2,
        "pillar3_nature_to_siliquant_bridges": re_p3,
        "pillar4_bayt_al_hikma_quant_bounds": re_p4,
        "verdict": "AUDITED_SILIQUANT_FORCE_MULTIPLIER_BRIDGES" if all_passed else "REJECTED_GATE_FAILED",
        "all_gates_passed": all_passed,
    }
    body["certificate_sha256"] = canonical_sha256(body)
    return body
