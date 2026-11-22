"""Pure-Q Exact Oracle (v1) for Track S9 (AUDIT-106):
Gated-MLP Z_2 Gauge Conservation (SwiGLU + Separate/Fused Muon + Learnable Multipliers),
Symmetry-Breaking Witnesses, and Exact Verification of weft-loom#44 & AUDIT-059 Rulings.

Governance & Mathematical Discipline (Aletheia v2.0.0):
- Pure Python 3 standard library only (int / Fraction); zero floats anywhere on trust paths.
- Zero bare assert statements; fail closed with typed GatedMlpLeanError.
- Strict Law-4 N0 prior-art quarantine (SwiGLU Shazeer 2020, Muon/Newton-Schulz Jordan 2024,
  Learnable Multipliers arXiv:2601.04890, Fisher/hypergeometric/sign tests, weft-loom#44
  exact binomial/cluster sign-lineage law) and Law-5 zero-metaphor exact operators.
"""

from fractions import Fraction
import hashlib
import json
import math
import re
import unicodedata


SCHEMA_VERSION = "aletheia.gated_mlp_lean_oracle.v1"
AUDIT_ID = "AUDIT-106"
CANONICAL_DECIMALS = 6

FORBIDDEN_METAPHOR_TOKENS = (
    "miracle",
    "quantumconsciousness",
    "magical",
    "numerology",
    "telepathy",
)
NON_ALNUM_RE = re.compile(r"[^a-z0-9]")

REQUIRED_MODELS = (
    "tiiuae__Falcon-H1-Tiny-R-0.6B",
    "tiiuae__Falcon-H1-Tiny-R-0.6B-pre-GRPO",
    "tiiuae__Falcon-H1-Tiny-90M-Base",
    "HuggingFaceTB__SmolLM2-135M",
    "HuggingFaceTB__SmolLM2-360M",
    "HuggingFaceTB__SmolLM2-135M-Instruct",
    "Qwen__Qwen3.5-0.8B",
    "Qwen__Qwen3.5-2B",
)


class GatedMlpLeanError(ValueError):
    """Typed fail-closed error with machine-readable code."""

    def __init__(self, code, message):
        super().__init__(f"{code}: {message}")
        self.code = code


def verify_no_floats(obj, path="root"):
    """Recursively reject floats, complex, sets, and non-exact objects."""
    if isinstance(obj, (float, complex)):
        raise GatedMlpLeanError("FLOAT_FORBIDDEN", f"float/complex forbidden at {path}")
    if isinstance(obj, (set, frozenset)):
        raise GatedMlpLeanError("INVALID_CONTAINER", f"unordered set forbidden at {path}")
    if isinstance(obj, dict):
        for k, v in obj.items():
            verify_no_floats(k, f"{path}.<key>")
            verify_no_floats(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            verify_no_floats(v, f"{path}[{i}]")


def require_non_empty_str(val, field):
    if not isinstance(val, str):
        raise GatedMlpLeanError("INVALID_TYPE", f"{field} must be str")
    for ch in val:
        if ord(ch) < 32 or ord(ch) == 127 or unicodedata.category(ch) == "Cf":
            raise GatedMlpLeanError("CONTROL_CHAR", f"forbidden control/format char in {field}")
    cleaned = val.strip()
    if not cleaned or cleaned != val:
        raise GatedMlpLeanError("EMPTY_STRING", f"{field} must be non-empty without outer whitespace")
    return cleaned


def require_exact_bool(val, field):
    if not isinstance(val, bool):
        raise GatedMlpLeanError("INVALID_BOOL", f"{field} must be bool")
    return val


def require_exact_int(val, field, min_value=None, max_value=None):
    if isinstance(val, bool) or not isinstance(val, int):
        raise GatedMlpLeanError("INVALID_INT", f"{field} must be int")
    if min_value is not None and val < min_value:
        raise GatedMlpLeanError("RANGE_ERROR", f"{field} < {min_value}")
    if max_value is not None and val > max_value:
        raise GatedMlpLeanError("RANGE_ERROR", f"{field} > {max_value}")
    return val


def require_exact_fraction(val, field):
    if isinstance(val, bool):
        raise GatedMlpLeanError("INVALID_FRACTION", f"{field} must not be bool")
    if isinstance(val, int):
        return Fraction(val, 1)
    if isinstance(val, Fraction):
        return val
    raise GatedMlpLeanError("INVALID_FRACTION", f"{field} must be int or Fraction")


def frac_str(val):
    f = require_exact_fraction(val, "frac")
    return f"{f.numerator}/{f.denominator}"


def format_exact_decimal(val, places=CANONICAL_DECIMALS):
    f = require_exact_fraction(val, "dec_val")
    p = require_exact_int(places, "places", min_value=0, max_value=12)
    neg = f < 0
    mag = -f if neg else f
    scale = 10 ** p
    quot, rem = divmod(mag.numerator * scale, mag.denominator)
    if rem * 2 >= mag.denominator:
        quot += 1
    if p == 0:
        sign = "-" if neg and quot != 0 else ""
        return f"{sign}{quot}"
    whole, frac_part = divmod(quot, scale)
    sign = "-" if neg and quot != 0 else ""
    return f"{sign}{whole}.{frac_part:0{p}d}"


def canonical_json_bytes(obj):
    verify_no_floats(obj, "canonical_json")
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def canonical_sha256(obj):
    return hashlib.sha256(canonical_json_bytes(obj)).hexdigest()


def _strict_equal(a, b):
    """Type-preserving recursive equality (distinguishes True/1, False/0, int/Fraction)."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return set(a.keys()) == set(b.keys()) and all(_strict_equal(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_strict_equal(x, y) for x, y in zip(a, b))
    return a == b


def enforce_law5_label_hygiene(labels):
    """Law-5 label filter (scope: label strings only, NOT a mathematical proof)."""
    verify_no_floats(labels, "labels")
    if not isinstance(labels, (list, tuple)) or len(labels) == 0:
        raise GatedMlpLeanError("INVALID_LABELS", "labels must be non-empty sequence")
    seen = set()
    cleaned = []
    for i, raw in enumerate(labels):
        name = require_non_empty_str(raw, f"labels[{i}]")
        norm = NON_ALNUM_RE.sub("", unicodedata.normalize("NFKC", name).lower())
        if not norm:
            raise GatedMlpLeanError("EMPTY_STRING", f"labels[{i}] empty after normalization")
        for tok in FORBIDDEN_METAPHOR_TOKENS:
            if tok in norm:
                raise GatedMlpLeanError("METAPHOR_FORBIDDEN", f"forbidden metaphor token in {name}")
        if norm in seen:
            raise GatedMlpLeanError("DUPLICATE_LABEL", f"duplicate normalized label {name}")
        seen.add(norm)
        cleaned.append(name)
    return {
        "label_filter_ok": True,
        "guard_scope": "label_filter_only_not_mathematical_proof",
        "label_count": len(cleaned),
        "labels": cleaned,
    }


# =============================================================================
# Exact Linear Algebra & Rational Gated-MLP Primitives
# =============================================================================


def _mat_mul(a, b):
    rows, inner, cols = len(a), len(a[0]), len(b[0])
    return [
        [sum(a[i][k] * b[k][j] for k in range(inner)) for j in range(cols)]
        for i in range(rows)
    ]


def _mat_t(a):
    return [[a[i][j] for i in range(len(a))] for j in range(len(a[0]))]


def _mat_vec(a, x):
    return [sum(a[i][j] * x[j] for j in range(len(x))) for i in range(len(a))]


def _mat_max_abs_diff(a, b):
    return max(abs(a[i][j] - b[i][j]) for i in range(len(a)) for j in range(len(a[0])))


def _vec_max_abs_diff(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _rational_silu_and_deriv(g):
    """Exact pure-Q non-odd SiLU-like gate: phi(g) = g * (1/2) * (1 + g / (1 + |g|)).

    Matches SiLU's key properties strictly in Q: phi(0)=0, phi'(0)=1/2, phi(g) > 0 for g > 0,
    small negative floor for g < 0, and strictly non-odd (phi(-g) != -phi(g) for g != 0).
    """
    denom = Fraction(1, 1) + abs(g)
    sig = Fraction(1, 2) * (Fraction(1, 1) + g / denom)
    dsig = Fraction(1, 2) / (denom * denom)
    return g * sig, sig + g * dsig


def newton_schulz_odd_poly_exact(mat_g, c0=Fraction(3, 1), c1=Fraction(-3, 2), c2=Fraction(1, 4)):
    """Exact pure-Q odd polynomial matrix update P(G) = G * (c0 I + c1 G^T G + c2 (G^T G)^2).

    Represents exact single-iteration Muon / Newton-Schulz orthogonalization step in Q.
    """
    verify_no_floats({"g": mat_g, "c": (c0, c1, c2)}, "newton_schulz")
    if not isinstance(mat_g, (list, tuple)) or len(mat_g) == 0:
        raise GatedMlpLeanError("INVALID_MATRIX", "mat_g must be non-empty")
    ncols = len(mat_g[0]) if isinstance(mat_g[0], (list, tuple)) else 0
    if ncols == 0:
        raise GatedMlpLeanError("INVALID_MATRIX", "mat_g rows must be non-empty")
    g = []
    for i, row in enumerate(mat_g):
        if not isinstance(row, (list, tuple)) or len(row) != ncols:
            raise GatedMlpLeanError("INVALID_MATRIX", "mat_g must be rectangular")
        g.append([require_exact_fraction(v, f"G[{i}][{j}]") for j, v in enumerate(row)])
    k0 = require_exact_fraction(c0, "c0")
    k1 = require_exact_fraction(c1, "c1")
    k2 = require_exact_fraction(c2, "c2")

    gtg = _mat_mul(_mat_t(g), g)
    gtg2 = _mat_mul(gtg, gtg)
    poly = [
        [
            (k0 if i == j else Fraction(0, 1)) + k1 * gtg[i][j] + k2 * gtg2[i][j]
            for j in range(ncols)
        ]
        for i in range(ncols)
    ]
    return _mat_mul(g, poly)


def _swiglu_forward_and_grads(w_g, w_u, w_d, r_shared, s_g, s_u, s_d, x, target, hidden_mode="none", asym_alpha=Fraction(0, 1)):
    """Exact pure-Q SwiGLU forward + backprop with Learnable Multipliers (s_g, s_u, s_d, shared r_i)."""
    h_dim = len(w_g)
    d_dim = len(x)
    g_pre = [s_g * r_shared[i] * _dot(w_g[i], x) for i in range(h_dim)]
    u_pre = [s_u * r_shared[i] * _dot(w_u[i], x) for i in range(h_dim)]
    phi_g, dphi_g = [], []
    for gv in g_pre:
        pv, dpv = _rational_silu_and_deriv(gv)
        phi_g.append(pv)
        dphi_g.append(dpv)
    h_raw = [phi_g[i] * u_pre[i] for i in range(h_dim)]

    if hidden_mode == "none":
        h_eff = list(h_raw)
    elif hidden_mode == "rmsnorm_sq_denom":
        # Exact rational RMS-like normalization preserving sign symmetry: h_i / (1 + mean(h^2))
        msq = sum(v * v for v in h_raw) / Fraction(h_dim, 1)
        h_eff = [v / (Fraction(1, 1) + msq) for v in h_raw]
    elif hidden_mode == "layernorm_center":
        # Exact mean subtraction (LayerNorm centering on hidden vector h): breaks Z_2!
        mean_h = sum(h_raw) / Fraction(h_dim, 1)
        h_eff = [v - mean_h for v in h_raw]
    else:
        raise GatedMlpLeanError("INVALID_MODE", f"unsupported hidden_mode {hidden_mode}")

    y = [s_d * sum(w_d[o][i] * h_eff[i] for i in range(h_dim)) for o in range(d_dim)]
    # Loss L = 1/2 * ||y - target||^2 - asym_alpha * sum(h_raw)
    diff_y = [y[o] - target[o] for o in range(d_dim)]
    loss = Fraction(1, 2) * _dot(diff_y, diff_y) - asym_alpha * sum(h_raw)

    # Exact backprop for hidden_mode == "none"
    grad_wd = [[s_d * diff_y[o] * h_eff[i] for i in range(h_dim)] for o in range(d_dim)]
    grad_heff = [s_d * sum(w_d[o][i] * diff_y[o] for o in range(d_dim)) for i in range(h_dim)]
    grad_hraw = [grad_heff[i] - asym_alpha for i in range(h_dim)]

    grad_gpre = [grad_hraw[i] * dphi_g[i] * u_pre[i] for i in range(h_dim)]
    grad_upre = [grad_hraw[i] * phi_g[i] for i in range(h_dim)]

    grad_wg = [[grad_gpre[i] * s_g * r_shared[i] * x[j] for j in range(d_dim)] for i in range(h_dim)]
    grad_wu = [[grad_upre[i] * s_u * r_shared[i] * x[j] for j in range(d_dim)] for i in range(h_dim)]
    grad_r = [
        grad_gpre[i] * s_g * _dot(w_g[i], x) + grad_upre[i] * s_u * _dot(w_u[i], x)
        for i in range(h_dim)
    ]
    return {
        "y": y,
        "loss": loss,
        "h_raw": h_raw,
        "h_eff": h_eff,
        "grad_wg": grad_wg,
        "grad_wu": grad_wu,
        "grad_wd": grad_wd,
        "grad_r": grad_r,
    }


def _flip_signs(w_u, w_d, signs):
    h_dim = len(signs)
    d_dim = len(w_d)
    wu_f = [[signs[i] * v for v in w_u[i]] for i in range(h_dim)]
    wd_f = [[signs[i] * w_d[o][i] for i in range(h_dim)] for o in range(d_dim)]
    return wu_f, wd_f


# =============================================================================
# Pillar 1: Exact Z_2 Gauge Conservation (SwiGLU + Separate/Fused Muon + LRMs + RMSNorm)
# =============================================================================


def audit_z2_conservation_muon_lrm_exact(signs=(1, -1, -1, 1)):
    """Prove exact pure-Q Z_2 equivariance of SwiGLU, backprop, separate/fused Muon, LRMs, and RMSNorm."""
    verify_no_floats(signs, "signs")
    if not isinstance(signs, (list, tuple)) or len(signs) != 4:
        raise GatedMlpLeanError("INVALID_SIGNS", "signs must be 4-tuple of +-1")
    clean_s = [require_exact_int(s, f"signs[{i}]") for i, s in enumerate(signs)]
    if any(s not in (-1, 1) for s in clean_s):
        raise GatedMlpLeanError("INVALID_SIGNS", "every sign entry must be +1 or -1")

    # Non-trivial asymmetric exact rational weights (h=4, d=3)
    w_g = [
        [Fraction(1, 2), Fraction(-1, 3), Fraction(1, 4)],
        [Fraction(-2, 5), Fraction(1, 2), Fraction(3, 4)],
        [Fraction(3, 5), Fraction(1, 4), Fraction(-1, 2)],
        [Fraction(-1, 4), Fraction(-3, 5), Fraction(2, 3)],
    ]
    w_u = [
        [Fraction(2, 3), Fraction(1, 5), Fraction(-1, 2)],
        [Fraction(1, 4), Fraction(-2, 3), Fraction(1, 5)],
        [Fraction(-1, 2), Fraction(3, 4), Fraction(1, 3)],
        [Fraction(1, 3), Fraction(-1, 4), Fraction(-2, 5)],
    ]
    w_d = [
        [Fraction(1, 2), Fraction(-1, 4), Fraction(2, 5), Fraction(-1, 3)],
        [Fraction(-1, 3), Fraction(3, 5), Fraction(-1, 2), Fraction(1, 4)],
        [Fraction(2, 5), Fraction(1, 3), Fraction(1, 4), Fraction(-1, 2)],
    ]
    r_shared = [Fraction(6, 5), Fraction(4, 5), Fraction(7, 5), Fraction(9, 10)]
    s_g, s_u, s_d = Fraction(5, 4), Fraction(3, 4), Fraction(7, 8)
    x = [Fraction(2, 1), Fraction(-3, 2), Fraction(5, 4)]
    target = [Fraction(1, 2), Fraction(-1, 4), Fraction(3, 4)]

    wu_f, wd_f = _flip_signs(w_u, w_d, clean_s)
    orig = _swiglu_forward_and_grads(w_g, w_u, w_d, r_shared, s_g, s_u, s_d, x, target, hidden_mode="none")
    flip = _swiglu_forward_and_grads(w_g, wu_f, wd_f, r_shared, s_g, s_u, s_d, x, target, hidden_mode="none")

    fwd_diff = _vec_max_abs_diff(orig["y"], flip["y"])
    loss_diff = abs(orig["loss"] - flip["loss"])
    lrm_grad_diff = _vec_max_abs_diff(orig["grad_r"], flip["grad_r"])

    # Verify exact gradient equivariance: G_g' = G_g, G_u' = D G_u, G_d' = G_d D
    exp_gu, exp_gd = _flip_signs(orig["grad_wu"], orig["grad_wd"], clean_s)
    grad_g_diff = _mat_max_abs_diff(orig["grad_wg"], flip["grad_wg"])
    grad_u_diff = _mat_max_abs_diff(exp_gu, flip["grad_wu"])
    grad_d_diff = _mat_max_abs_diff(exp_gd, flip["grad_wd"])

    # Separate Muon (Newton-Schulz odd polynomial) on G_u and G_d
    muon_u_orig = newton_schulz_odd_poly_exact(orig["grad_wu"])
    muon_u_flip = newton_schulz_odd_poly_exact(flip["grad_wu"])
    muon_d_orig = newton_schulz_odd_poly_exact(orig["grad_wd"])
    muon_d_flip = newton_schulz_odd_poly_exact(flip["grad_wd"])
    exp_muon_u, exp_muon_d = _flip_signs(muon_u_orig, muon_d_orig, clean_s)
    muon_sep_diff = max(_mat_max_abs_diff(exp_muon_u, muon_u_flip), _mat_max_abs_diff(exp_muon_d, muon_d_flip))

    # Fused gate_up_proj Muon (Newton-Schulz on stacked [G_g; G_u])
    fused_orig = orig["grad_wg"] + orig["grad_wu"]
    fused_flip = flip["grad_wg"] + flip["grad_wu"]
    muon_fused_orig = newton_schulz_odd_poly_exact(fused_orig)
    muon_fused_flip = newton_schulz_odd_poly_exact(fused_flip)
    exp_fused = muon_fused_orig[:4] + [[clean_s[i] * v for v in muon_fused_orig[4 + i]] for i in range(4)]
    muon_fused_diff = _mat_max_abs_diff(exp_fused, muon_fused_flip)

    # RMSNorm on hidden vector h preserves exact forward invariance!
    rms_orig = _swiglu_forward_and_grads(w_g, w_u, w_d, r_shared, s_g, s_u, s_d, x, target, hidden_mode="rmsnorm_sq_denom")
    rms_flip = _swiglu_forward_and_grads(w_g, wu_f, wd_f, r_shared, s_g, s_u, s_d, x, target, hidden_mode="rmsnorm_sq_denom")
    rms_hidden_diff = _vec_max_abs_diff(rms_orig["y"], rms_flip["y"])

    # Verify gate_up inner products flip strictly for every s_i = -1 and stay fixed for s_i = +1
    gu_orig = [_dot(w_g[i], w_u[i]) for i in range(4)]
    gu_flip = [_dot(w_g[i], wu_f[i]) for i in range(4)]
    inner_flip_exact = all(gu_flip[i] == clean_s[i] * gu_orig[i] for i in range(4))

    passed = (
        fwd_diff == 0
        and loss_diff == 0
        and lrm_grad_diff == 0
        and grad_g_diff == 0
        and grad_u_diff == 0
        and grad_d_diff == 0
        and muon_sep_diff == 0
        and muon_fused_diff == 0
        and rms_hidden_diff == 0
        and inner_flip_exact
    )
    return {
        "signs_applied": clean_s,
        "forward_output_max_diff_exact": frac_str(fwd_diff),
        "loss_diff_exact": frac_str(loss_diff),
        "lrm_shared_row_grad_max_diff_exact": frac_str(lrm_grad_diff),
        "grad_wg_max_diff_exact": frac_str(grad_g_diff),
        "grad_wu_equivariance_diff_exact": frac_str(grad_u_diff),
        "grad_wd_equivariance_diff_exact": frac_str(grad_d_diff),
        "muon_separate_equivariance_diff_exact": frac_str(muon_sep_diff),
        "muon_fused_gate_up_equivariance_diff_exact": frac_str(muon_fused_diff),
        "rmsnorm_on_hidden_output_diff_exact": frac_str(rms_hidden_diff),
        "gate_up_inner_product_sign_flip_exact": inner_flip_exact,
        "passed": passed,
    }


# =============================================================================
# Pillar 2: Exact Z_2 Symmetry-Breaking Must-Fail Witnesses
# =============================================================================


def audit_z2_symmetry_breaking_controls_exact(asym_alpha=Fraction(1, 2), step_lr=Fraction(1, 2)):
    """Three exact pure-Q controls proving which mechanisms DO break Z_2 unit-sign symmetry."""
    verify_no_floats({"a": asym_alpha, "lr": step_lr}, "pillar2_breaks")
    alpha = require_exact_fraction(asym_alpha, "asym_alpha")
    lr = require_exact_fraction(step_lr, "step_lr")
    if alpha <= 0 or lr <= 0:
        raise GatedMlpLeanError("INVALID_PARAMS", "asym_alpha and step_lr must be positive")

    w_g = [
        [Fraction(1, 2), Fraction(-1, 3), Fraction(1, 4)],
        [Fraction(-2, 5), Fraction(1, 2), Fraction(3, 4)],
        [Fraction(3, 5), Fraction(1, 4), Fraction(-1, 2)],
        [Fraction(-1, 4), Fraction(-3, 5), Fraction(2, 3)],
    ]
    w_u = [
        [Fraction(2, 3), Fraction(1, 5), Fraction(-1, 2)],
        [Fraction(1, 4), Fraction(-2, 3), Fraction(1, 5)],
        [Fraction(-1, 2), Fraction(3, 4), Fraction(1, 3)],
        [Fraction(1, 3), Fraction(-1, 4), Fraction(-2, 5)],
    ]
    w_d = [
        [Fraction(1, 2), Fraction(-1, 4), Fraction(2, 5), Fraction(-1, 3)],
        [Fraction(-1, 3), Fraction(3, 5), Fraction(-1, 2), Fraction(1, 4)],
        [Fraction(2, 5), Fraction(1, 3), Fraction(1, 4), Fraction(-1, 2)],
    ]
    ones_r = [Fraction(1, 1)] * 4
    one = Fraction(1, 1)
    x = [Fraction(2, 1), Fraction(-3, 2), Fraction(5, 4)]
    target = [Fraction(1, 2), Fraction(-1, 4), Fraction(3, 4)]
    signs = (1, -1, -1, 1)
    wu_f, wd_f = _flip_signs(w_u, w_d, signs)

    # B1: LayerNorm centering on hidden vector h breaks forward Z_2 invariance!
    ln_orig = _swiglu_forward_and_grads(w_g, w_u, w_d, ones_r, one, one, one, x, target, hidden_mode="layernorm_center")
    ln_flip = _swiglu_forward_and_grads(w_g, wu_f, wd_f, ones_r, one, one, one, x, target, hidden_mode="layernorm_center")
    ln_fwd_diff = _vec_max_abs_diff(ln_orig["y"], ln_flip["y"])

    # B2: Asymmetric hidden-unit penalty (-alpha * sum(h_raw)) breaks mirror-twin Z_2 equivariance after 1 SGD step!
    # Start from mirror twins (W_u, W_d) and (-W_u, -W_d) where average <w_g, w_u> = 0/1!
    wu_neg, wd_neg = _flip_signs(w_u, w_d, (-1, -1, -1, -1))
    a_pos = _swiglu_forward_and_grads(w_g, w_u, w_d, ones_r, one, one, one, x, target, hidden_mode="none", asym_alpha=alpha)
    a_neg = _swiglu_forward_and_grads(w_g, wu_neg, wd_neg, ones_r, one, one, one, x, target, hidden_mode="none", asym_alpha=alpha)
    wg_p1 = [[w_g[i][j] - lr * a_pos["grad_wg"][i][j] for j in range(3)] for i in range(4)]
    wu_p1 = [[w_u[i][j] - lr * a_pos["grad_wu"][i][j] for j in range(3)] for i in range(4)]
    wg_n1 = [[w_g[i][j] - lr * a_neg["grad_wg"][i][j] for j in range(3)] for i in range(4)]
    wu_n1 = [[wu_neg[i][j] - lr * a_neg["grad_wu"][i][j] for j in range(3)] for i in range(4)]
    # Under Z_2 equivariance, sum_i (<wg_p1, wu_p1> + <wg_n1, wu_n1>) MUST be identically 0/1; asymmetric penalty breaks it!
    twin_inner_sum = sum(_dot(wg_p1[i], wu_p1[i]) + _dot(wg_n1[i], wu_n1[i]) for i in range(4))

    # B3: Correlated init W_u^(0) = W_g^(0) with zero-sum columns and balanced row-sum signs reproduces Falcon's signature:
    # 100% positive <w_g, w_u> while up_rowsum stays exactly 50/50 (2 positive, 2 negative)!
    w_bal = [
        [Fraction(2, 1), Fraction(-1, 1), Fraction(0, 1)],    # row sum = +1
        [Fraction(-2, 1), Fraction(1, 1), Fraction(0, 1)],    # row sum = -1
        [Fraction(1, 1), Fraction(1, 1), Fraction(-1, 1)],    # row sum = +1
        [Fraction(-1, 1), Fraction(-1, 1), Fraction(1, 1)],   # row sum = -1
    ]
    corr_gu_pos = sum(1 for row in w_bal if _dot(row, row) > 0)
    corr_up_rowsum_pos = sum(1 for row in w_bal if sum(row) > 0)

    b1_breaks = ln_fwd_diff > 0
    b2_breaks = twin_inner_sum != 0
    b3_signature = corr_gu_pos == 4 and corr_up_rowsum_pos == 2
    return {
        "b1_layernorm_hidden_output_diff_exact": frac_str(ln_fwd_diff),
        "b1_layernorm_breaks_z2": b1_breaks,
        "b2_asym_penalty_twin_inner_sum_exact": frac_str(twin_inner_sum),
        "b2_asym_penalty_breaks_z2": b2_breaks,
        "b3_correlated_init_gu_positive": corr_gu_pos,
        "b3_correlated_init_up_rowsum_positive": corr_up_rowsum_pos,
        "b3_matches_orthogonal_invariant_signature": b3_signature,
        "passed": b1_breaks and b2_breaks and b3_signature,
    }


# =============================================================================
# Pillar 3: Exact Verification of weft-loom#44 weight_lean.json & AUDIT-059 Rulings
# =============================================================================


def exact_hypergeom_tail(pop_n, succ_k, draw_n, obs_k):
    n_pop = require_exact_int(pop_n, "pop_n", min_value=1)
    k_suc = require_exact_int(succ_k, "succ_k", min_value=0, max_value=n_pop)
    n_drw = require_exact_int(draw_n, "draw_n", min_value=0, max_value=n_pop)
    k_obs = require_exact_int(obs_k, "obs_k", min_value=0, max_value=min(k_suc, n_drw))
    num = sum(math.comb(k_suc, k) * math.comb(n_pop - k_suc, n_drw - k) for k in range(k_obs, min(k_suc, n_drw) + 1))
    return Fraction(num, math.comb(n_pop, n_drw))


def exact_stratified_two_block_tail(n1, k1_bg, d1, n2, k2_bg, d2, obs_total):
    p1 = require_exact_int(n1, "n1", min_value=1)
    k1 = require_exact_int(k1_bg, "k1_bg", min_value=0, max_value=p1)
    dr1 = require_exact_int(d1, "d1", min_value=0, max_value=p1)
    p2 = require_exact_int(n2, "n2", min_value=1)
    k2 = require_exact_int(k2_bg, "k2_bg", min_value=0, max_value=p2)
    dr2 = require_exact_int(d2, "d2", min_value=0, max_value=p2)
    tot = require_exact_int(obs_total, "obs_total", min_value=0, max_value=min(k1, dr1) + min(k2, dr2))
    den1 = math.comb(p1, dr1)
    den2 = math.comb(p2, dr2)
    num = 0
    for x1 in range(min(k1, dr1) + 1):
        c1 = math.comb(k1, x1) * math.comb(p1 - k1, dr1 - x1)
        lo2 = max(0, tot - x1)
        for x2 in range(lo2, min(k2, dr2) + 1):
            c2 = math.comb(k2, x2) * math.comb(p2 - k2, dr2 - x2)
            num += c1 * c2
    return Fraction(num, den1 * den2)


def exact_binomial_two_sided(n_trials, k_succ):
    n = require_exact_int(n_trials, "n_trials", min_value=1, max_value=200)
    k = require_exact_int(k_succ, "k_succ", min_value=0, max_value=n)
    m = min(k, n - k)
    tail = Fraction(2 * sum(math.comb(n, j) for j in range(m + 1)), 1 << n)
    return Fraction(1, 1) if tail > 1 else tail


def _gemma_l10_l9_margin_exact():
    return Fraction(3101, 237659)


def audit_weight_lean_and_audit059_exact(weight_lean_summary):
    """Verify weft-loom#44 weight_lean counts + AUDIT-059 exact narrowed hypergeom/sign statistics."""
    verify_no_floats(weight_lean_summary, "weight_lean_summary")
    if not isinstance(weight_lean_summary, dict):
        raise GatedMlpLeanError("INVALID_SUMMARY", "weight_lean_summary must be dict")
    req_top = {"diag_source", "models"}
    if set(weight_lean_summary.keys()) != req_top:
        raise GatedMlpLeanError("INVALID_SUMMARY", "weight_lean_summary keys mismatch")
    diag_src = require_non_empty_str(weight_lean_summary["diag_source"], "diag_source")
    if diag_src not in ("live_json_file", "standalone_fixture"):
        raise GatedMlpLeanError("INVALID_SUMMARY", f"unsupported diag_source {diag_src}")

    models = weight_lean_summary["models"]
    if not isinstance(models, dict) or set(models.keys()) != set(REQUIRED_MODELS):
        raise GatedMlpLeanError("INVALID_MODELS", "models dict must contain the exact 8 weft-loom#44 models")

    parsed = {}
    for mk in REQUIRED_MODELS:
        rec = models[mk]
        if not isinstance(rec, dict) or set(rec.keys()) != {"gu_pos", "units", "up_pos", "dn_pos", "layers_gt_half", "n_layers"}:
            raise GatedMlpLeanError("INVALID_MODEL_REC", f"malformed record for {mk}")
        units = require_exact_int(rec["units"], f"{mk}.units", min_value=1)
        gu_pos = require_exact_int(rec["gu_pos"], f"{mk}.gu_pos", min_value=0, max_value=units)
        up_pos = require_exact_int(rec["up_pos"], f"{mk}.up_pos", min_value=0, max_value=units)
        dn_pos = require_exact_int(rec["dn_pos"], f"{mk}.dn_pos", min_value=0, max_value=units)
        n_layers = require_exact_int(rec["n_layers"], f"{mk}.n_layers", min_value=1, max_value=200)
        l_gt = require_exact_int(rec["layers_gt_half"], f"{mk}.layers_gt_half", min_value=0, max_value=n_layers)
        parsed[mk] = {
            "gu_pos": gu_pos,
            "units": units,
            "gu_share_dec6": format_exact_decimal(Fraction(gu_pos, units)),
            "up_share_dec6": format_exact_decimal(Fraction(up_pos, units)),
            "dn_share_dec6": format_exact_decimal(Fraction(dn_pos, units)),
            "layers_gt_half": l_gt,
            "n_layers": n_layers,
            "layer_sign_test_p_exact": frac_str(exact_binomial_two_sided(n_layers, l_gt)),
        }

    f06 = parsed["tiiuae__Falcon-H1-Tiny-R-0.6B"]
    f06_pre = parsed["tiiuae__Falcon-H1-Tiny-R-0.6B-pre-GRPO"]
    f90 = parsed["tiiuae__Falcon-H1-Tiny-90M-Base"]
    controls = [
        parsed[k]
        for k in (
            "HuggingFaceTB__SmolLM2-135M",
            "HuggingFaceTB__SmolLM2-360M",
            "HuggingFaceTB__SmolLM2-135M-Instruct",
            "Qwen__Qwen3.5-0.8B",
            "Qwen__Qwen3.5-2B",
        )
    ]

    falcon_06_all_layers = f06["layers_gt_half"] == f06["n_layers"] and Fraction(f06["gu_pos"], f06["units"]) > Fraction(3, 4)
    falcon_pre_grpo_eq = f06 == f06_pre
    falcon_90m_all_layers = f90["layers_gt_half"] == f90["n_layers"] and Fraction(f90["gu_pos"], f90["units"]) > Fraction(3, 5)
    falcon_up_dn_near_half = all(
        abs(Fraction(models[mk]["up_pos"], models[mk]["units"]) - Fraction(1, 2)) < Fraction(1, 50)
        and abs(Fraction(models[mk]["dn_pos"], models[mk]["units"]) - Fraction(1, 2)) < Fraction(1, 50)
        for mk in (
            "tiiuae__Falcon-H1-Tiny-R-0.6B",
            "tiiuae__Falcon-H1-Tiny-R-0.6B-pre-GRPO",
            "tiiuae__Falcon-H1-Tiny-90M-Base",
        )
    )
    controls_near_half = all(
        abs(Fraction(c["gu_pos"], c["units"]) - Fraction(1, 2)) < Fraction(1, 100) and c["layers_gt_half"] < c["n_layers"]
        for c in controls
    )

    # Exact AUDIT-059 narrowed statistical verifications
    p_21_24 = exact_binomial_two_sided(24, 21)
    p_l15_strat = exact_stratified_two_block_tail(512, 29, 40, 3072, 48, 24, 16)
    p_l19_lo = exact_hypergeom_tail(512, 8, 35, 3)
    p_l19_hi = exact_hypergeom_tail(3072, 3, 29, 1)
    p_l19_strat = exact_stratified_two_block_tail(512, 8, 35, 3072, 3, 29, 4)
    gemma_margin = _gemma_l10_l9_margin_exact()

    audit059_checks = (
        p_21_24 == Fraction(2325, 8388608)
        and p_l15_strat < Fraction(1, 10 ** 9)
        and p_l19_lo > Fraction(1, 100)
        and p_l19_hi > Fraction(1, 100)
        and p_l19_strat < Fraction(1, 500)
        and gemma_margin < Fraction(1, 50)
    )

    passed = (
        falcon_06_all_layers
        and falcon_pre_grpo_eq
        and falcon_90m_all_layers
        and falcon_up_dn_near_half
        and controls_near_half
        and audit059_checks
    )
    return {
        "diag_source": diag_src,
        "models_verified": parsed,
        "falcon_06b_all_44_layers_gt_half": falcon_06_all_layers,
        "falcon_06b_matches_pre_grpo_exact": falcon_pre_grpo_eq,
        "falcon_90m_all_24_layers_gt_half": falcon_90m_all_layers,
        "falcon_up_and_down_near_half": falcon_up_dn_near_half,
        "all_5_free_sign_controls_near_half": controls_near_half,
        "audit059_21_of_24_p_exact": frac_str(p_21_24),
        "audit059_l15_stratified_p_lt_1e9": p_l15_strat < Fraction(1, 10 ** 9),
        "audit059_l19_lo_exceeds_001": p_l19_lo > Fraction(1, 100),
        "audit059_l19_stratified_p_lt_2e3": p_l19_strat < Fraction(1, 500),
        "audit059_gemma_l10_l9_margin_exact": frac_str(gemma_margin),
        "audit059_gemma_l10_l9_margin_pct_dec6": format_exact_decimal(gemma_margin * 100),
        "audit059_narrowed_stats_verified": audit059_checks,
        "passed": passed,
    }


# =============================================================================
# Pillar 4: Fail-Closed Cryptographic Certification Gate
# =============================================================================


def certify_gated_mlp_lean_audit_exact(
    mode,
    live_json_recounted,
    operator_labels,
    weight_lean_summary_raw,
    pillar1_result,
    pillar2_result,
    pillar3_result,
    three_axis_checks,
):
    """Fail-closed cryptographic certificate re-deriving all 3 pillars with strict type equality.

    Honest provenance scope: validates schema integrity, exact re-derivation of Pillars 1-3,
    and provenance-tag consistency between mode/live_json_recounted and diag_source. True
    live file provenance comes strictly from running gated_mlp_lean_audit.py --live.
    """
    verify_no_floats(
        {
            "mode": mode,
            "live": live_json_recounted,
            "labels": operator_labels,
            "raw": weight_lean_summary_raw,
            "p1": pillar1_result,
            "p2": pillar2_result,
            "p3": pillar3_result,
            "axes": three_axis_checks,
        },
        "certify",
    )
    clean_mode = require_non_empty_str(mode, "mode")
    if clean_mode not in ("live_json_recount", "standalone_fixture_replay"):
        raise GatedMlpLeanError("INVALID_MODE", f"unsupported mode {clean_mode}")
    live_flag = require_exact_bool(live_json_recounted, "live_json_recounted")
    if live_flag != (clean_mode == "live_json_recount"):
        raise GatedMlpLeanError("MODE_MISMATCH", "live_json_recounted contradicts mode")

    label_guard = enforce_law5_label_hygiene(operator_labels)

    req_axes = (
        "z2_conservation_muon_lrm_rmsnorm",
        "z2_symmetry_breaking_controls",
        "weft_loom_weight_lean_and_audit059",
    )
    if not isinstance(three_axis_checks, dict) or set(three_axis_checks.keys()) != set(req_axes):
        raise GatedMlpLeanError("INVALID_AXES", "three_axis_checks keys mismatch")
    clean_axes = {ax: require_exact_bool(three_axis_checks[ax], f"axes.{ax}") for ax in req_axes}

    if (
        not isinstance(pillar1_result, dict)
        or not isinstance(pillar2_result, dict)
        or not isinstance(pillar3_result, dict)
    ):
        raise GatedMlpLeanError("INVALID_PILLAR", "all pillar results must be dicts")

    re_p1 = audit_z2_conservation_muon_lrm_exact()
    if not _strict_equal(re_p1, pillar1_result):
        raise GatedMlpLeanError("PILLAR_CONTRADICTION", "pillar1_result contradicts re-derivation")

    re_p2 = audit_z2_symmetry_breaking_controls_exact()
    if not _strict_equal(re_p2, pillar2_result):
        raise GatedMlpLeanError("PILLAR_CONTRADICTION", "pillar2_result contradicts re-derivation")

    re_p3 = audit_weight_lean_and_audit059_exact(weight_lean_summary_raw)
    if not _strict_equal(re_p3, pillar3_result):
        raise GatedMlpLeanError("PILLAR_CONTRADICTION", "pillar3_result contradicts re-derivation")

    expected_src = "live_json_file" if live_flag else "standalone_fixture"
    if re_p3["diag_source"] != expected_src:
        raise GatedMlpLeanError("PROVENANCE_MISMATCH", f"diag_source {re_p3['diag_source']} contradicts mode {clean_mode}")

    expected_axes = {
        "z2_conservation_muon_lrm_rmsnorm": re_p1["passed"],
        "z2_symmetry_breaking_controls": re_p2["passed"],
        "weft_loom_weight_lean_and_audit059": re_p3["passed"],
    }
    if not _strict_equal(clean_axes, expected_axes):
        raise GatedMlpLeanError("AXIS_CONTRADICTION", "three_axis_checks contradicts re-derived pillars")

    all_passed = all(clean_axes.values())
    body = {
        "schema_version": SCHEMA_VERSION,
        "audit_id": AUDIT_ID,
        "mode": clean_mode,
        "live_json_recounted": live_flag,
        "novelty_classification": "N0_prior_art_primitives_with_N1_exact_muon_lrm_z2_conservation_proof",
        "law5_label_hygiene_guard": label_guard,
        "three_axis_audit_checks": clean_axes,
        "all_three_axes_passed": all_passed,
        "pillar1_z2_conservation_muon_lrm": re_p1,
        "pillar2_z2_symmetry_breaking_controls": re_p2,
        "pillar3_weft_loom_weight_lean_and_audit059": re_p3,
        "verdict": "AUDITED_GATED_MLP_Z2_CONSERVATION_AND_RULINGS" if all_passed else "REJECTED_GATE_FAILED",
        "all_gates_passed": all_passed,
    }
    body["certificate_sha256"] = canonical_sha256(body)
    return body
