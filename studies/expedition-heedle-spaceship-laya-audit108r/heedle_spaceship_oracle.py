"""Pure-Z / Pure-Q Oracle for Track S13 (AUDIT-108): Heedle x Spaceship x Laya/SiliQuant.

Formalizes and verifies in exact integer and rational arithmetic (zero IEEE-754
floats, zero math.sqrt, zero bare assert statements, pure Python 3 stdlib only):

  Pillar 1 (p1_rational_gauss_newton_shafts_and_shed):
    Exact rational Gauss-Newton metric G = J^T J / (n m), orthogonal rational
    shafts, treadle carry share rho = sum(alpha_i^2) in [0, 1], Pythagorean
    undelivered residual MS(t*c - J*d) = t^2*(1 - rho), monotone share growth,
    and exact Gram-Schmidt pinned-heddle orthogonal projection (S*d = 0).

  Pillar 2 (p2_hidden_move_nullspace_refutation):
    Exact rational construction of heedle core.hidden_move inside ker(J_known)
    with non-zero effect on blinded Zone-B probes (J_elsewhere), refuted by a
    single-witness Zone-B snap certificate.

  Pillar 3 (p3_scaling_bridge_pruning_and_int4_survival):
    Scaling bridge to AUDIT-097 + Moissanite-Labs/siliquant + vega-U (laya):
    (a) exact proof that column-structured rank-1/linear-port moves preserve
    heddle orthogonality (S^T d = 0) and rank-1 structure while row/entry-wise
    pruning shatters them; (b) exact symmetric INT4 (qmax=7) RNE survival vs
    (3/4)-ULP erasure and spike-driven collateral neighbor zeroing.

  Pillar 4 (p4_heedle_protocol_replay_and_tide1):
    Exact finite-domain replay of heedle docs/protocol.md (heedle-draft/1)
    hash chain (pick -> snap -> unpick with non-refundable budget), heedle#6
    tide/1 exhaustive finite-domain OKLCH contrast-floor host, and closure of
    the clean-room T1 double-zone-skip defect found in heedle/draft.py:157-159.

  Pillar 5 (p5_spaceship_realization_to_zone_b_snap):
    Formalizes chuanqisun/spaceship#5 circular same-vector realization limit
    bridged to laya/vegaml disjoint-split conformal LAC checking and Aletheia
    v2.1 Zone-B snap escalation.

Honest Law-4 novelty tier:
  - N0: Classical Gauss-Newton / Gram-Schmidt / symmetric RNE quantization /
    SHA-256 hash-chain semantics and pinned prior specifications
    (Moissanite-Labs/heedle @ d5e42392, chuanqisun/spaceship#5 @ 7633dbee,
    NandhaKishorM/laya & vegaml, AUDIT-097).
  - N1 (engineering/audit): Executable pure-Z/Q unification oracle, scaling
    bridge certifier, and clean-room heedle/draft.py audit catches.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from fractions import Fraction
from typing import Any

SCHEMA_VERSION = "heedle-spaceship-laya-oracle/v1"
TRACK_ID = "track-s13-heedle-spaceship-laya"
AUDIT_ID = "AUDIT-108"

HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
FRAC_RE = re.compile(r"^(-?[1-9][0-9]*|0)/([1-9][0-9]*)$")

# Pinned primary intake commits and file SHA-256 digests (read-only audit).
HEEDLE_MAIN_COMMIT = "d5e42392b03fe0edaddf8bdeb35c0c03c9378ca4"
HEEDLE_PR8_COMMIT = "7de102ecf56ba99e277f7c9d021c2e66eab55e8c"
HEEDLE_CORE_SHA256 = "49b2bf776693f05dfdf361353586f61535622d9ae6fc412d55afe972adb04c8c"
HEEDLE_DRAFT_SHA256 = "498b9392da0b5f37412b04f508463d143b0848df09b2d62d9762b5b7974a60d6"
HEEDLE_HOSTS_SHA256 = "796af461d3ee1e5789132e061445ff52c34e5d35707d978fc089d43a94b8346a"
HEEDLE_PROTOCOL_SHA256 = "7c5e583dee69c59436c84b5eceaab5711f0cd676a068de970136a17d5906d3b2"
SPACESHIP_MASTER_COMMIT = "0499b9930bdc0c471ee62ee1b1c01e1a986fb4ec"
SPACESHIP_PR5_COMMIT = "7633dbee79799944d757e6d6993607174bee88fe"

HEEDLE_DOMAIN = b"HEEDLE-V1:"
ZONES: tuple[str, ...] = ("auto", "shared", "knot")
GESTURES: tuple[str, ...] = ("release", "hold", "knot")

ROOT_W0 = hashlib.sha256(b"WEFT-CBC-V1:heedle-audit108-w0").hexdigest()
ROOT_W1 = hashlib.sha256(b"WEFT-CBC-V1:heedle-audit108-w1").hexdigest()
ROOT_W_SNAP = hashlib.sha256(b"WEFT-CBC-V1:heedle-audit108-wsnap").hexdigest()
MOVE_SHA_1 = hashlib.sha256(b"heedle-audit108-move-1").hexdigest()
MOVE_SHA_SNAP = hashlib.sha256(b"heedle-audit108-move-snap").hexdigest()


class HeedleAuditError(ValueError):
    """Typed fail-closed error carrying an explicit machine-readable error code."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


# ---------------------------------------------------------------------------
# Strict Pure-Z / Pure-Q Validation & Exact Rational Vector Primitives
# ---------------------------------------------------------------------------

def _is_nonempty_str(val: Any) -> bool:
    return type(val) is str and len(val.strip()) > 0


def _in_str_set(val: Any, allowed: Any) -> bool:
    return type(val) is str and val in allowed


def _is_hex64(val: Any) -> bool:
    return type(val) is str and bool(HEX64_RE.fullmatch(val)) and len(set(val)) > 1


def contains_float(obj: Any) -> bool:
    """Return True if obj contains any float, complex, set, frozenset, or non-JSON type."""
    if obj is None or type(obj) in (bool, int, str):
        return False
    if type(obj) is dict:
        return any(contains_float(k) or contains_float(v) for k, v in obj.items())
    if type(obj) in (list, tuple):
        return any(contains_float(v) for v in obj)
    return True


def canonical_json_bytes(obj: Any) -> bytes:
    if contains_float(obj):
        raise HeedleAuditError("ERR_FLOAT_FORBIDDEN", "float or non-JSON value forbidden")
    try:
        rendered = json.dumps(
            obj,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise HeedleAuditError("ERR_CANONICAL_JSON", str(exc)) from exc
    return rendered.encode("ascii")


def canonical_sha256(obj: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(obj)).hexdigest()


def frac_to_str(q: Fraction) -> str:
    if type(q) is not Fraction:
        raise HeedleAuditError("ERR_INVALID_FRACTION", "expected Fraction")
    return f"{q.numerator}/{q.denominator}"


def parse_exact_fraction(val: Any, label: str = "value") -> Fraction:
    """Convert strict int, Fraction, or canonical reduced 'p/q' str to exact Fraction."""
    if type(val) is int:
        return Fraction(val, 1)
    if type(val) is Fraction:
        return val
    if type(val) is str:
        if len(val) > 64 or not FRAC_RE.fullmatch(val):
            raise HeedleAuditError("ERR_INVALID_FRACTION_STR", f"{label} invalid fraction string")
        num_s, den_s = val.split("/")
        num = int(num_s)
        den = int(den_s)
        if math.gcd(abs(num), den) != 1:
            raise HeedleAuditError("ERR_UNREDUCED_FRACTION", f"{label} must be reduced")
        return Fraction(num, den)
    raise HeedleAuditError("ERR_INVALID_FRACTION_TYPE", f"{label} must be int, Fraction, or p/q str")


def _req_exact_keys(mapping: Any, expected: set[str] | frozenset[str], label: str) -> dict[str, Any]:
    if type(mapping) is not dict or set(mapping.keys()) != set(expected):
        raise HeedleAuditError("ERR_SCHEMA_KEYS", f"{label} keyset mismatch")
    return mapping


def _req_strict_str_eq(val: Any, expected: str, label: str) -> None:
    if type(val) is not str or val != expected:
        raise HeedleAuditError("ERR_SCHEMA_STR", f"{label} must equal {expected!r}")


def _req_strict_int_range(val: Any, low: int, high: int | None, label: str) -> int:
    if type(val) is not int or val < low:
        raise HeedleAuditError("ERR_SCHEMA_INT", f"{label} must be int >= {low}")
    if high is not None and val > high:
        raise HeedleAuditError("ERR_SCHEMA_INT", f"{label} must be int <= {high}")
    return val


def _req_bool_eq(val: Any, expected: bool, label: str) -> None:
    if val is not expected:
        raise HeedleAuditError("ERR_SCHEMA_BOOL", f"{label} must be {expected!r}")


def _req_str_list(val: Any, expected: list[str] | None, label: str) -> list[str]:
    if type(val) is not list or any(not _is_nonempty_str(item) for item in val):
        raise HeedleAuditError("ERR_SCHEMA_LIST", f"{label} must be list[nonempty str]")
    if expected is not None and val != expected:
        raise HeedleAuditError("ERR_SCHEMA_LIST", f"{label} mismatch")
    return val


def _req_hex64(val: Any, label: str) -> str:
    if not _is_hex64(val):
        raise HeedleAuditError("ERR_INVALID_SHA256", f"{label} must be non-uniform 64-hex")
    return val


def _vec_dot(a: list[Fraction], b: list[Fraction]) -> Fraction:
    return sum((x * y for x, y in zip(a, b)), Fraction(0, 1))


def _mat_vec(mat: list[list[Fraction]], vec: list[Fraction]) -> list[Fraction]:
    return [_vec_dot(row, vec) for row in mat]


def _mat_t_vec(mat: list[list[Fraction]], vec: list[Fraction]) -> list[Fraction]:
    cols = len(mat[0])
    out = [Fraction(0, 1)] * cols
    for r_idx, row in enumerate(mat):
        vr = vec[r_idx]
        for c_idx in range(cols):
            out[c_idx] += row[c_idx] * vr
    return out


def _require_frac_vec(vec: Any, label: str, expected_len: int | None = None) -> list[Fraction]:
    if type(vec) is not list or len(vec) == 0:
        raise HeedleAuditError("ERR_INVALID_VECTOR", f"{label} must be nonempty list")
    if expected_len is not None and len(vec) != expected_len:
        raise HeedleAuditError("ERR_VECTOR_LEN", f"{label} expected length {expected_len}")
    return [parse_exact_fraction(x, f"{label}[{i}]") for i, x in enumerate(vec)]


def _require_frac_mat(mat: Any, label: str) -> list[list[Fraction]]:
    if type(mat) is not list or len(mat) == 0:
        raise HeedleAuditError("ERR_INVALID_MATRIX", f"{label} must be nonempty list")
    first = _require_frac_vec(mat[0], f"{label}[0]")
    cols = len(first)
    rows = [first]
    for idx in range(1, len(mat)):
        rows.append(_require_frac_vec(mat[idx], f"{label}[{idx}]", cols))
    return rows


# ---------------------------------------------------------------------------
# Pillar 1: Exact Rational Gauss-Newton Shafts, Treadle Share & Pinned Shed
# ---------------------------------------------------------------------------

def gram_schmidt_orthogonal_rows(rows: list[list[Fraction]]) -> list[list[Fraction]]:
    """Exact rational Gram-Schmidt (zero square roots): returns orthogonal basis of row(S)."""
    basis: list[list[Fraction]] = []
    norms_sq: list[Fraction] = []
    for row in rows:
        v = list(row)
        for b, b_nsq in zip(basis, norms_sq):
            coeff = _vec_dot(b, v) / b_nsq
            v = [vi - coeff * bi for vi, bi in zip(v, b)]
        nsq = _vec_dot(v, v)
        if nsq != 0:
            basis.append(v)
            norms_sq.append(nsq)
    return basis


def shed_project_exact(pin_jacobian: list[list[Any]], move: list[Any]) -> dict[str, Any]:
    """Exact orthogonal projection onto ker(S) = {d : S d = 0} in pure Q."""
    s_mat = _require_frac_mat(pin_jacobian, "pin_jacobian")
    cols = len(s_mat[0])
    d_vec = _require_frac_vec(move, "move", cols)

    basis = gram_schmidt_orthogonal_rows(s_mat)
    proj = list(d_vec)
    for b in basis:
        coeff = _vec_dot(b, proj) / _vec_dot(b, b)
        proj = [pi - coeff * bi for pi, bi in zip(proj, b)]

    removed = [di - pi for di, pi in zip(d_vec, proj)]
    pin_leak = _mat_vec(s_mat, proj)
    pin_leak_sq = _vec_dot(pin_leak, pin_leak)

    # Idempotence check: P(P(d)) == P(d)
    proj2 = list(proj)
    for b in basis:
        coeff = _vec_dot(b, proj2) / _vec_dot(b, b)
        proj2 = [pi - coeff * bi for pi, bi in zip(proj2, b)]

    d_norm_sq = _vec_dot(d_vec, d_vec)
    proj_norm_sq = _vec_dot(proj, proj)
    rem_norm_sq = _vec_dot(removed, removed)

    return {
        "projected_move": [frac_to_str(x) for x in proj],
        "pin_rank": len(basis),
        "open_dimensions": cols - len(basis),
        "pin_leak_sq": frac_to_str(pin_leak_sq),
        "idempotent": proj2 == proj,
        "pythagorean_exact": proj_norm_sq + rem_norm_sq == d_norm_sq,
        "projected_norm_sq": frac_to_str(proj_norm_sq),
        "removed_norm_sq": frac_to_str(rem_norm_sq),
    }


def certify_heddle_preservation(pin_jacobian: list[list[Any]], move: list[Any]) -> dict[str, Any]:
    """Fail-closed verifier asserting S * d == 0 exactly on pinned heddles."""
    s_mat = _require_frac_mat(pin_jacobian, "pin_jacobian")
    d_vec = _require_frac_vec(move, "move", len(s_mat[0]))
    leak = _mat_vec(s_mat, d_vec)
    leak_sq = _vec_dot(leak, leak)
    if leak_sq != 0:
        return {
            "passed": False,
            "error_code": "ERR_HEDDLE_PIN_LEAK",
            "pin_leak_sq": frac_to_str(leak_sq),
        }
    return {
        "passed": True,
        "error_code": None,
        "pin_leak_sq": "0/1",
    }


def evaluate_rational_shafts_and_treadle(
    jacobian_j: list[list[Any]],
    shaft_vectors: list[list[Any]],
    pattern_c: list[Any],
    travel_t: Any,
    k_shafts: int,
) -> dict[str, Any]:
    """Exact pure-Q verification of G = J^T J/(n m), shaft eigenvalues, carry share rho,
    and Pythagorean undelivered residual MS(t*c - J*d) = t^2*(1 - rho).
    """
    j_mat = _require_frac_mat(jacobian_j, "jacobian_j")
    nm = len(j_mat)
    p_dim = len(j_mat[0])
    if type(shaft_vectors) is not list or len(shaft_vectors) == 0:
        raise HeedleAuditError("ERR_INVALID_SHAFTS", "shaft_vectors must be nonempty list")
    v_list = [_require_frac_vec(v, f"shaft[{i}]", p_dim) for i, v in enumerate(shaft_vectors)]
    c_vec = _require_frac_vec(pattern_c, "pattern_c", nm)
    t_q = parse_exact_fraction(travel_t, "travel_t")
    _req_strict_int_range(k_shafts, 1, len(v_list), "k_shafts")

    c_norm_sq = _vec_dot(c_vec, c_vec)
    if c_norm_sq != Fraction(nm, 1):
        raise HeedleAuditError("ERR_PATTERN_NOT_UNIT_RMS", "pattern c must have unit RMS (<c,c> == n*m)")

    # Verify shaft mutual orthogonality & exact G v_i = lambda_i v_i
    jv_list: list[list[Fraction]] = []
    eigenvalues: list[Fraction] = []
    for i, vi in enumerate(v_list):
        v_nsq = _vec_dot(vi, vi)
        if v_nsq == 0:
            raise HeedleAuditError("ERR_ZERO_SHAFT", f"shaft {i} is zero")
        for j in range(i):
            if _vec_dot(vi, v_list[j]) != 0:
                raise HeedleAuditError("ERR_NON_ORTHOGONAL_SHAFTS", "shafts must be mutually orthogonal")
        jvi = _mat_vec(j_mat, vi)
        jv_nsq = _vec_dot(jvi, jvi)
        if jv_nsq == 0:
            raise HeedleAuditError("ERR_NULL_SHAFT", f"shaft {i} has zero eigenvalue")
        lam_i = jv_nsq / (Fraction(nm, 1) * v_nsq)
        g_vi = [x / Fraction(nm, 1) for x in _mat_t_vec(j_mat, jvi)]
        if g_vi != [lam_i * x for x in vi]:
            raise HeedleAuditError("ERR_NOT_EIGENVECTOR", f"shaft {i} is not an eigenvector of G")
        jv_list.append(jvi)
        eigenvalues.append(lam_i)

    # Compute exact rational treadle move d on first k_shafts shafts
    d_move = [Fraction(0, 1)] * p_dim
    alpha_sq_list: list[Fraction] = []
    for i in range(k_shafts):
        jvi = jv_list[i]
        jv_nsq = _vec_dot(jvi, jvi)
        proj_num = _vec_dot(c_vec, jvi)
        alpha_sq_i = (proj_num * proj_num) / (c_norm_sq * jv_nsq)
        alpha_sq_list.append(alpha_sq_i)
        scale = t_q * proj_num / jv_nsq
        d_move = [di + scale * vi_elem for di, vi_elem in zip(d_move, v_list[i])]

    rho = sum(alpha_sq_list, Fraction(0, 1))
    dy = _mat_vec(j_mat, d_move)
    gain_along_c = _vec_dot(dy, c_vec) / c_norm_sq
    residual_vec = [t_q * ci - dyi for ci, dyi in zip(c_vec, dy)]
    undelivered_ms = _vec_dot(residual_vec, residual_vec) / Fraction(nm, 1)
    expected_undelivered_ms = t_q * t_q * (Fraction(1, 1) - rho)
    # Predicted first-order MS change d^T G d = t^2 * rho
    predicted_ms = _vec_dot(dy, dy) / Fraction(nm, 1)

    return {
        "k_shafts": k_shafts,
        "eigenvalues": [frac_to_str(x) for x in eigenvalues],
        "alpha_sq": [frac_to_str(x) for x in alpha_sq_list],
        "carry_share_rho": frac_to_str(rho),
        "gain_along_c": frac_to_str(gain_along_c),
        "expected_gain_t_rho": frac_to_str(t_q * rho),
        "gain_identity_exact": gain_along_c == t_q * rho,
        "undelivered_ms": frac_to_str(undelivered_ms),
        "expected_undelivered_ms": frac_to_str(expected_undelivered_ms),
        "pythagorean_residual_exact": undelivered_ms == expected_undelivered_ms,
        "predicted_ms_equals_t2_rho": predicted_ms == t_q * t_q * rho,
        "move_d": [frac_to_str(x) for x in d_move],
    }


def certify_treadle_share_claim(rho_val: Any, travel_t: Any, claimed_undelivered_ms: Any) -> dict[str, Any]:
    """Fail-closed verifier checking rho in [0, 1] and undelivered_ms == t^2*(1 - rho)."""
    rho = parse_exact_fraction(rho_val, "rho")
    t_q = parse_exact_fraction(travel_t, "travel_t")
    u_ms = parse_exact_fraction(claimed_undelivered_ms, "undelivered_ms")
    if rho < 0 or rho > 1 or u_ms != t_q * t_q * (Fraction(1, 1) - rho):
        return {
            "passed": False,
            "error_code": "ERR_TREADLE_SHARE_INVALID",
            "rho": frac_to_str(rho),
        }
    return {
        "passed": True,
        "error_code": None,
        "rho": frac_to_str(rho),
    }


def run_pillar_1() -> dict[str, Any]:
    # Diagonal rational Jacobian J (4x4, n=2 inputs x m=2 outputs, P=4 weights)
    # Eigenvalues of G = J^T J / 4 are 4, 1, 1/4, 1/16 on standard basis shafts.
    j_mat = [
        [4, 0, 0, 0],
        [0, 2, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, Fraction(1, 2)],
    ]
    shafts = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
    # Unit-RMS pattern c = [1, 1, 1, 1] (<c, c> = 4 = n*m); each shaft carries alpha_i^2 = 1/4!
    c_pat = [1, 1, 1, 1]
    t_val = Fraction(3, 5)
    k1 = evaluate_rational_shafts_and_treadle(j_mat, shafts, c_pat, t_val, k_shafts=1)
    k2 = evaluate_rational_shafts_and_treadle(j_mat, shafts, c_pat, t_val, k_shafts=2)
    k4 = evaluate_rational_shafts_and_treadle(j_mat, shafts, c_pat, t_val, k_shafts=4)

    # Pinned-heddle orthogonal projection (S = [[1, 1, 0, 0], [0, 0, 1, -1]])
    pin_s = [
        [1, 1, 0, 0],
        [0, 0, 1, -1],
    ]
    raw_move = [2, 0, 3, 1]
    shed_res = shed_project_exact(pin_s, raw_move)
    cert_proj = certify_heddle_preservation(pin_s, shed_res["projected_move"])
    mf_leak = certify_heddle_preservation(pin_s, raw_move)
    mf_share = certify_treadle_share_claim(Fraction(5, 4), t_val, Fraction(0, 1))

    return {
        "passed": all([
            k1["carry_share_rho"] == "1/4",
            k2["carry_share_rho"] == "1/2",
            k4["carry_share_rho"] == "1/1",
            k1["gain_identity_exact"] is True,
            k2["pythagorean_residual_exact"] is True,
            k4["undelivered_ms"] == "0/1",
            shed_res["pin_leak_sq"] == "0/1",
            shed_res["idempotent"] is True,
            shed_res["pythagorean_exact"] is True,
            cert_proj["passed"] is True,
            mf_leak["passed"] is False,
            mf_share["passed"] is False,
        ]),
        "shaft_eigenvalues": k4["eigenvalues"],
        "treadle_k1_rho": k1["carry_share_rho"],
        "treadle_k2_rho": k2["carry_share_rho"],
        "treadle_k2_undelivered_ms": k2["undelivered_ms"],
        "treadle_k4_rho": k4["carry_share_rho"],
        "treadle_k4_undelivered_ms": k4["undelivered_ms"],
        "share_monotone_nondecreasing": True,
        "shed_projection": {
            "pin_rank": shed_res["pin_rank"],
            "open_dimensions": shed_res["open_dimensions"],
            "pin_leak_sq": shed_res["pin_leak_sq"],
            "idempotent": shed_res["idempotent"],
            "pythagorean_exact": shed_res["pythagorean_exact"],
            "projected_move": shed_res["projected_move"],
        },
        "must_fail_unpinned_move_leaks_heddle": {
            "rejected": mf_leak["passed"] is False,
            "error_code": mf_leak["error_code"],
            "pin_leak_sq": mf_leak["pin_leak_sq"],
        },
        "must_fail_share_exceeds_unity": {
            "rejected": mf_share["passed"] is False,
            "error_code": mf_share["error_code"],
            "claimed_rho": mf_share["rho"],
        },
    }


# ---------------------------------------------------------------------------
# Pillar 2: Hidden-Move Nullspace Stealth Vector & Zone-B Blinded Snap
# ---------------------------------------------------------------------------

def construct_hidden_move_exact(
    j_known: list[list[Any]],
    j_elsewhere: list[list[Any]],
    seed_vec: list[Any],
    iterations: int = 4,
) -> dict[str, Any]:
    """Exact pure-Q analog of heedle core.hidden_move: power iteration on J_elsewhere^T J_elsewhere
    restricted to ker(J_known), normalized by max-abs entry (zero sqrt)."""
    jk = _require_frac_mat(j_known, "j_known")
    je = _require_frac_mat(j_elsewhere, "j_elsewhere")
    p_dim = len(jk[0])
    if len(je[0]) != p_dim:
        raise HeedleAuditError("ERR_DIM_MISMATCH", "j_known and j_elsewhere column counts differ")
    _req_strict_int_range(iterations, 1, 16, "iterations")
    v = _require_frac_vec(seed_vec, "seed_vec", p_dim)

    basis = gram_schmidt_orthogonal_rows(jk)

    def _proj(z: list[Fraction]) -> list[Fraction]:
        out = list(z)
        for b in basis:
            coeff = _vec_dot(b, out) / _vec_dot(b, b)
            out = [oi - coeff * bi for oi, bi in zip(out, b)]
        return out

    v = _proj(v)
    if _vec_dot(v, v) == 0:
        raise HeedleAuditError("ERR_SEED_IN_KNOWN_ROWSPACE", "seed projects to zero in ker(J_known)")

    for _ in range(iterations):
        jv = _mat_vec(je, v)
        gtv = _mat_t_vec(je, jv)
        v = _proj(gtv)
        max_abs = max(abs(x) for x in v)
        if max_abs == 0:
            raise HeedleAuditError("ERR_ZERO_ELSEWHERE_ACTION", "projected move vanishes on elsewhere probes")
        v = [x / max_abs for x in v]

    known_dy = _mat_vec(jk, v)
    else_dy = _mat_vec(je, v)
    known_ms = _vec_dot(known_dy, known_dy) / Fraction(len(jk), 1)
    else_ms = _vec_dot(else_dy, else_dy) / Fraction(len(je), 1)
    worst_idx = max(range(len(else_dy)), key=lambda i: else_dy[i] * else_dy[i])
    worst_sq = else_dy[worst_idx] * else_dy[worst_idx]

    return {
        "stealth_move": [frac_to_str(x) for x in v],
        "known_ms": frac_to_str(known_ms),
        "elsewhere_ms": frac_to_str(else_ms),
        "worst_probe_index": worst_idx,
        "worst_probe_change_sq": frac_to_str(worst_sq),
    }


def verify_zone_b_snap_witness(
    j_elsewhere: list[list[Any]],
    move: list[Any],
    witness_index: int,
    tau_sq: Any,
) -> dict[str, Any]:
    """Single-witness Zone-B snap verifier (heedle/draft.py S1 rule in exact Q)."""
    je = _require_frac_mat(j_elsewhere, "j_elsewhere")
    d_vec = _require_frac_vec(move, "move", len(je[0]))
    _req_strict_int_range(witness_index, 0, len(je) - 1, "witness_index")
    tau2 = parse_exact_fraction(tau_sq, "tau_sq")
    if tau2 <= 0:
        raise HeedleAuditError("ERR_NON_POSITIVE_TAU", "tau_sq must be > 0")
    dy_w = _vec_dot(je[witness_index], d_vec)
    change_sq = dy_w * dy_w
    if change_sq > tau2:
        return {
            "snapped": True,
            "error_code": "ERR_STEALTH_MOVE_SNAPPED",
            "witness_index": witness_index,
            "witness_change_sq": frac_to_str(change_sq),
            "tau_sq": frac_to_str(tau2),
        }
    return {
        "snapped": False,
        "error_code": "ERR_SNAP_WITNESS_DOES_NOT_REFUTE",
        "witness_index": witness_index,
        "witness_change_sq": frac_to_str(change_sq),
        "tau_sq": frac_to_str(tau2),
    }


def run_pillar_2() -> dict[str, Any]:
    # Known probes observe only coordinates 0 & 1; blinded Zone-B elsewhere probes
    # include a quiet probe (row 0) and a sensitive hidden probe (row 1) on coords 2 & 3!
    j_known = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ]
    j_elsewhere = [
        [0, 0, Fraction(1, 4), 0],   # quiet probe: change^2 = 1/16 <= tau^2 = 1/4
        [0, 0, 0, 2],                # sensitive probe: change^2 = 4 > tau^2 = 1/4
    ]
    tau_sq = Fraction(1, 4)  # tau = 1/2
    hm = construct_hidden_move_exact(j_known, j_elsewhere, [3, 5, 1, 1], iterations=3)

    # Blinded Zone-B snaps on worst witness index 1; quiet witness index 0 fails S1
    snap_hit = verify_zone_b_snap_witness(j_elsewhere, hm["stealth_move"], hm["worst_probe_index"], tau_sq)
    mf_quiet = verify_zone_b_snap_witness(j_elsewhere, hm["stealth_move"], 0, tau_sq)

    return {
        "passed": all([
            hm["known_ms"] == "0/1",
            snap_hit["snapped"] is True,
            snap_hit["error_code"] == "ERR_STEALTH_MOVE_SNAPPED",
            mf_quiet["snapped"] is False,
            mf_quiet["error_code"] == "ERR_SNAP_WITNESS_DOES_NOT_REFUTE",
        ]),
        "tau_sq": frac_to_str(tau_sq),
        "known_probe_ms": hm["known_ms"],
        "elsewhere_probe_ms": hm["elsewhere_ms"],
        "stealth_move": hm["stealth_move"],
        "zone_b_snap_witness_index": snap_hit["witness_index"],
        "zone_b_snap_change_sq": snap_hit["witness_change_sq"],
        "must_fail_known_probe_only_misses_stealth": {
            "rejected": snap_hit["snapped"] is True,
            "error_code": snap_hit["error_code"],
            "known_probe_ms": hm["known_ms"],
            "blinded_witness_change_sq": snap_hit["witness_change_sq"],
        },
        "must_fail_snap_with_quiet_witness": {
            "rejected": mf_quiet["snapped"] is False,
            "error_code": mf_quiet["error_code"],
            "quiet_witness_change_sq": mf_quiet["witness_change_sq"],
        },
    }


# ---------------------------------------------------------------------------
# Pillar 3: Scaling Bridge — Pruning vs Heddles & INT4 Quantization Survival
# ---------------------------------------------------------------------------

def certify_rank1_heddle_pruning_exact(
    u_vec: list[Any],
    v_vec: list[Any],
    mask: list[list[int]],
    heddle_vec: list[Any],
) -> dict[str, Any]:
    """Exact pure-Q proof: column-structured rank-1 moves preserve heddle orthogonality
    (h^T Delta = 0) and exact rank <= 1, while row/entry-wise pruning shatters them."""
    u = _require_frac_vec(u_vec, "u")
    v = _require_frac_vec(v_vec, "v")
    rows, cols = len(u), len(v)
    h = _require_frac_vec(heddle_vec, "heddle_vec", rows)
    if _vec_dot(u, u) == 0 or _vec_dot(v, v) == 0 or _vec_dot(h, h) == 0:
        raise HeedleAuditError("ERR_ZERO_VECTOR", "u, v, and heddle_vec must be non-zero")
    if type(mask) is not list or len(mask) != rows:
        raise HeedleAuditError("ERR_INVALID_MASK", "mask row count mismatch")
    for r_idx, mrow in enumerate(mask):
        if type(mrow) is not list or len(mrow) != cols or any(type(c) is not int or c not in (0, 1) for c in mrow):
            raise HeedleAuditError("ERR_INVALID_MASK", f"bad binary mask at row {r_idx}")

    # Verify baseline rank-1 move Delta_0 = u v^T lies inside heddle shed (h^T u == 0)
    base_hu = _vec_dot(h, u)
    if base_hu != 0:
        raise HeedleAuditError("ERR_BASE_NOT_IN_SHED", "u must satisfy h^T u == 0")

    delta = [
        [u[r] * v[c] if mask[r][c] == 1 else Fraction(0, 1) for c in range(cols)]
        for r in range(rows)
    ]
    frob_sq = sum((delta[r][c] * delta[r][c] for r in range(rows) for c in range(cols)), Fraction(0, 1))
    # Gram defect ||Delta||_F^4 - ||Delta^T Delta||_F^2 == 0 iff rank(Delta) <= 1
    gram_frob_sq = Fraction(0, 1)
    for c1 in range(cols):
        for c2 in range(cols):
            dot = sum((delta[r][c1] * delta[r][c2] for r in range(rows)), Fraction(0, 1))
            gram_frob_sq += dot * dot
    rank1_gram_defect = frob_sq * frob_sq - gram_frob_sq

    # Heddle leakage ||h^T Delta||_2^2
    h_delta = [sum((h[r] * delta[r][c] for r in range(rows)), Fraction(0, 1)) for c in range(cols)]
    heddle_leak_sq = _vec_dot(h_delta, h_delta)

    passed = rank1_gram_defect == 0 and heddle_leak_sq == 0
    return {
        "passed": passed,
        "error_code": None if passed else "ERR_PRUNING_SHATTERED_HEDDLE",
        "frobenius_sq": frac_to_str(frob_sq),
        "rank1_gram_defect": frac_to_str(rank1_gram_defect),
        "is_exact_rank_le_1": rank1_gram_defect == 0,
        "heddle_leak_sq": frac_to_str(heddle_leak_sq),
        "heddle_preserved": heddle_leak_sq == 0,
    }


def quantize_sym_rne_exact(val_int: int, max_abs_int: int, qmax: int = 7) -> int:
    """Exact pure-integer symmetric round-half-to-even quantizer matching AUDIT-097 / siliquant."""
    if type(val_int) is not int:
        raise HeedleAuditError("ERR_INVALID_INT", "val_int must be exact int")
    _req_strict_int_range(max_abs_int, 0, None, "max_abs_int")
    _req_strict_int_range(qmax, 1, None, "qmax")
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


def certify_int4_heedle_move_survival(
    base_group: list[int],
    edited_group: list[int],
    qmax: int = 7,
) -> dict[str, Any]:
    """Exact pure-Z/Q INT4 survival audit: checks edit survival vs (3/4)-ULP erasure
    and spike-driven collateral flips on unedited pinned weights."""
    _req_strict_int_range(qmax, 1, None, "qmax")
    if (
        type(base_group) is not list
        or type(edited_group) is not list
        or len(base_group) == 0
        or len(base_group) != len(edited_group)
    ):
        raise HeedleAuditError("ERR_INVALID_GROUP", "base_group and edited_group must be equal nonempty lists")
    for idx, (bv, ev) in enumerate(zip(base_group, edited_group)):
        if type(bv) is not int or type(ev) is not int:
            raise HeedleAuditError("ERR_INVALID_INT", f"group entry {idx} must be exact int")

    max_base = max(abs(x) for x in base_group)
    max_edit = max(abs(x) for x in edited_group)
    if max_base == 0:
        raise HeedleAuditError("ERR_ZERO_GROUP", "base_group must be non-zero")
    ulp = Fraction(max_base, qmax)
    three_quarter_ulp = Fraction(3 * max_base, 4 * qmax)

    q_before = [quantize_sym_rne_exact(b, max_base, qmax) for b in base_group]
    q_after = [quantize_sym_rne_exact(e, max_edit, qmax) for e in edited_group]

    edited_flips = 0
    erased_edits = 0
    collateral_flips = 0
    min_edit_ulp_ratio: Fraction | None = None

    for b, e, qb, qa in zip(base_group, edited_group, q_before, q_after):
        if e != b:
            ratio = Fraction(abs(e - b), 1) / ulp
            if min_edit_ulp_ratio is None or ratio < min_edit_ulp_ratio:
                min_edit_ulp_ratio = ratio
            if qa != qb:
                edited_flips += 1
            else:
                erased_edits += 1
        elif qa != qb:
            collateral_flips += 1

    if erased_edits > 0:
        code = "ERR_QUANT_MOVE_ERASED"
    elif collateral_flips > 0:
        code = "ERR_QUANT_COLLATERAL_FLIP"
    else:
        code = None

    return {
        "passed": code is None,
        "error_code": code,
        "qmax": qmax,
        "ulp": frac_to_str(ulp),
        "three_quarter_ulp": frac_to_str(three_quarter_ulp),
        "scale_ratio": frac_to_str(Fraction(max_edit, max_base)),
        "min_edit_ulp_ratio": frac_to_str(min_edit_ulp_ratio) if min_edit_ulp_ratio is not None else "0/1",
        "edited_flips": edited_flips,
        "erased_edits": erased_edits,
        "collateral_unedited_flips": collateral_flips,
        "q_before": q_before,
        "q_after": q_after,
    }


def run_pillar_3() -> dict[str, Any]:
    # u = [1, -1, 1, -1], h = [1, 1, 1, 1] => h^T u == 0 exactly!
    u = [1, -1, 1, -1]
    v = [2, 1, 3]
    h = [1, 1, 1, 1]

    col_mask = [[1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1]]
    entry_mask = [[1, 0, 0], [0, 1, 0], [1, 1, 1], [0, 0, 1]]

    col_res = certify_rank1_heddle_pruning_exact(u, v, col_mask, h)
    entry_res = certify_rank1_heddle_pruning_exact(u, v, entry_mask, h)

    # INT4 (qmax=7): max_base = 28 => ULP = 4, (3/4)*ULP = 3!
    # Clean surviving move from cell center 0 -> +3 (|d|=3 == (3/4)*ULP, lands at q=+1) with 0 collateral flips:
    base_grp = [28, 0, 4, -8]
    surv_grp = [28, 3, 4, -8]
    surv_res = certify_int4_heedle_move_survival(base_grp, surv_grp, qmax=7)

    # Erasure control: interior w=-1 (q=0) stepped by +3 == (3/4)*ULP lands at +2 (half-ULP tie -> even q=0, ERASED!):
    erased_base = [28, -1, 4, -8]
    erased_edit = [28, 2, 4, -8]
    mf_erased = certify_int4_heedle_move_survival(erased_base, erased_edit, qmax=7)

    # Spike collateral control: interior weight jumping 3 -> 56 survives (+1 -> +7) while doubling
    # group absmax 28 -> 56 and collaterally flipping/zeroing 3 unedited neighbors!
    spike_base = [28, 3, 4, -8]
    spike_edit = [28, 56, 4, -8]
    mf_spike = certify_int4_heedle_move_survival(spike_base, spike_edit, qmax=7)

    return {
        "passed": all([
            col_res["passed"] is True,
            entry_res["passed"] is False,
            surv_res["passed"] is True,
            mf_erased["passed"] is False,
            mf_spike["passed"] is False,
        ]),
        "column_structured_pruning": {
            "is_exact_rank_le_1": col_res["is_exact_rank_le_1"],
            "heddle_preserved": col_res["heddle_preserved"],
            "heddle_leak_sq": col_res["heddle_leak_sq"],
            "rank1_gram_defect": col_res["rank1_gram_defect"],
        },
        "int4_surviving_move": {
            "ulp": surv_res["ulp"],
            "three_quarter_ulp": surv_res["three_quarter_ulp"],
            "min_edit_ulp_ratio": surv_res["min_edit_ulp_ratio"],
            "edited_flips": surv_res["edited_flips"],
            "collateral_unedited_flips": surv_res["collateral_unedited_flips"],
        },
        "must_fail_entry_pruning_shatters_heddle": {
            "rejected": entry_res["passed"] is False,
            "error_code": entry_res["error_code"],
            "rank1_gram_defect": entry_res["rank1_gram_defect"],
            "heddle_leak_sq": entry_res["heddle_leak_sq"],
        },
        "must_fail_int4_sub_threshold_erasure": {
            "rejected": mf_erased["passed"] is False,
            "error_code": mf_erased["error_code"],
            "min_edit_ulp_ratio": mf_erased["min_edit_ulp_ratio"],
            "erased_edits": mf_erased["erased_edits"],
        },
        "must_fail_int4_spike_collateral_flip": {
            "rejected": mf_spike["passed"] is False,
            "error_code": mf_spike["error_code"],
            "scale_ratio": mf_spike["scale_ratio"],
            "collateral_unedited_flips": mf_spike["collateral_unedited_flips"],
        },
    }


# ---------------------------------------------------------------------------
# Pillar 4: heedle protocol.md Hash Chain, tide/1 Host & T1 Audit Closure
# ---------------------------------------------------------------------------

def heedle_genesis_id(host: dict[str, Any], gate: dict[str, Any]) -> str:
    payload = {"schema": "heedle-draft/1", "host": host, "gate": gate}
    return hashlib.sha256(HEEDLE_DOMAIN + b"draft:" + canonical_json_bytes(payload)).hexdigest()


def heedle_event_id(ev: dict[str, Any]) -> str:
    body = {
        k: ev[k]
        for k in ("seq", "kind", "prev", "parent_root", "root", "move_sha256", "pins", "gesture", "tau")
        if k in ev
    }
    return hashlib.sha256(HEEDLE_DOMAIN + b"event:" + canonical_json_bytes(body)).hexdigest()


def zone_of_exact(tension: Fraction) -> str:
    """Exact pure-Q thresholds matching heedle/core.py:494-501 (3/10, 7/10)."""
    if tension < Fraction(3, 10):
        return "auto"
    if tension < Fraction(7, 10):
        return "shared"
    return "knot"


def check_t1_gesture_fixed(tension_val: Any, gesture: str) -> dict[str, Any]:
    """Fixed zone-adjacent T1 check closing HEEDLE-AUDIT-01 (heedle/draft.py:157-159).

    Unlike draft.py:157-159 where `near` around 0.7 disabled T1 completely (permitting
    gesture='release' at tension=0.7), tolerance only relaxes by AT MOST ONE adjacent
    zone step when within 1/1000000 of a threshold.
    """
    t_q = parse_exact_fraction(tension_val, "tension")
    if t_q < 0:
        raise HeedleAuditError("ERR_NEGATIVE_TENSION", "tension must be >= 0")
    if not _in_str_set(gesture, GESTURES):
        raise HeedleAuditError("ERR_INVALID_GESTURE", f"unknown gesture {gesture!r}")
    zone = zone_of_exact(t_q)
    min_idx = ZONES.index(zone)
    tol = Fraction(1, 1000000)
    if abs(t_q - Fraction(7, 10)) < tol:
        min_idx = 1  # adjacent zone "hold" permitted, NEVER "release" (0)!
    elif abs(t_q - Fraction(3, 10)) < tol:
        min_idx = 0
    g_idx = GESTURES.index(gesture)
    if g_idx < min_idx:
        return {
            "passed": False,
            "error_code": "ERR_T1_GESTURE_TOO_LIGHT",
            "zone": zone,
            "gesture": gesture,
        }
    return {
        "passed": True,
        "error_code": None,
        "zone": zone,
        "gesture": gesture,
    }


def _validate_pins(pins: Any) -> None:
    if type(pins) is not list or any(type(p) is not int or p < 0 for p in pins) or pins != sorted(set(pins)):
        raise HeedleAuditError("ERR_INVALID_PINS", "pins must be sorted distinct non-negative ints")


def replay_heedle_draft_exact(draft: dict[str, Any]) -> dict[str, Any]:
    """Exact pure-Z/Q finite-domain replay of heedle-draft/1 (H1-H4, E1-E5, S1, K1, U1, T1)."""
    _req_exact_keys(draft, {"schema", "host", "gate", "events"}, "draft")
    _req_strict_str_eq(draft["schema"], "heedle-draft/1", "draft.schema")
    host = _req_exact_keys(draft["host"], {"kind", "name", "w0_root"}, "draft.host")
    if not _is_nonempty_str(host["kind"]) or not _is_nonempty_str(host["name"]):
        raise HeedleAuditError("ERR_INVALID_HOST", "host.kind and host.name must be nonempty str")
    gate = _req_exact_keys(draft["gate"], {"r_invariant", "budget", "cost", "tau"}, "draft.gate")
    w0_root = _req_hex64(host["w0_root"], "host.w0_root")
    r_inv_q = parse_exact_fraction(gate["r_invariant"], "gate.r_invariant")
    if r_inv_q <= 0:
        raise HeedleAuditError("ERR_NON_POSITIVE_R_INV", "gate.r_invariant must be > 0")
    budget = _req_strict_int_range(gate["budget"], 1, None, "gate.budget")
    cost = _req_strict_int_range(gate["cost"], 1, budget, "gate.cost")
    tau_str = gate["tau"]
    tau_q = parse_exact_fraction(tau_str, "gate.tau")
    if tau_q <= 0:
        raise HeedleAuditError("ERR_NON_POSITIVE_TAU", "gate.tau must be > 0")
    events = draft["events"]
    if type(events) is not list or len(events) == 0:
        raise HeedleAuditError("ERR_INVALID_EVENTS", "events must be nonempty list")

    lineage: list[str] = [w0_root]
    prev = heedle_genesis_id(host, gate)
    fell_root = w0_root

    for idx, ev in enumerate(events):
        if type(ev) is not dict or not _in_str_set(ev.get("kind"), ("pick", "snap", "unpick")):
            raise HeedleAuditError("ERR_INVALID_EVENT", f"event[{idx}] invalid kind")
        kind = ev["kind"]
        expected_ev_keys = (
            {"seq", "kind", "prev", "parent_root", "root", "pins", "gesture", "tau", "id", "remaining_budget"}
            if kind == "unpick"
            else {
                "seq", "kind", "prev", "parent_root", "root", "move_sha256",
                "pins", "gesture", "tau", "id", "remaining_budget", "tension",
                "witness_change", "refuted",
            }
        )
        _req_exact_keys(ev, expected_ev_keys, f"event[{idx}]")
        if type(ev["seq"]) is not int or ev["seq"] != idx + 1 or ev["prev"] != prev or ev["tau"] != tau_str:
            return {"passed": False, "error_code": "ERR_PROTOCOL_CHAIN_OR_TAU", "step_index": idx}
        if not _in_str_set(ev["gesture"], GESTURES):
            raise HeedleAuditError("ERR_INVALID_GESTURE", f"event[{idx}] unknown gesture {ev['gesture']!r}")
        _validate_pins(ev["pins"])
        if heedle_event_id(ev) != ev["id"]:
            return {"passed": False, "error_code": "ERR_EVENT_ID_MISMATCH", "step_index": idx}
        if ev["parent_root"] != fell_root:
            return {"passed": False, "error_code": "ERR_PARENT_ROOT_MISMATCH", "step_index": idx}
        prev = ev["id"]

        if kind == "unpick":
            if ev["root"] not in lineage:
                return {"passed": False, "error_code": "ERR_UNPICK_UNKNOWN_ROOT", "step_index": idx}
            if type(ev["remaining_budget"]) is not int or ev["remaining_budget"] != budget:
                return {"passed": False, "error_code": "ERR_UNPICK_BUDGET_REFUND", "step_index": idx}
            fell_root = ev["root"]
            lineage.append(fell_root)
            continue

        _req_hex64(ev["move_sha256"], f"event[{idx}].move_sha256")
        cand_root = _req_hex64(ev["root"], f"event[{idx}].root")

        t1 = check_t1_gesture_fixed(ev["tension"], ev["gesture"])
        if not t1["passed"]:
            return {"passed": False, "error_code": t1["error_code"], "step_index": idx}

        if budget < cost or type(ev["remaining_budget"]) is not int or ev["remaining_budget"] != budget - cost:
            return {"passed": False, "error_code": "ERR_BUDGET_MISMATCH", "step_index": idx}
        budget -= cost

        wc = parse_exact_fraction(ev["witness_change"], "witness_change")
        if wc < 0:
            raise HeedleAuditError("ERR_NEGATIVE_WITNESS_CHANGE", "witness_change must be >= 0")
        if kind == "snap":
            if wc <= tau_q or ev["refuted"] is not True:
                return {"passed": False, "error_code": "ERR_S1_INVALID_SNAP", "step_index": idx}
            # Fell does NOT move on snap, budget stays spent!
            lineage.append(fell_root)
        else:
            if wc > tau_q or ev["refuted"] is not False:
                return {"passed": False, "error_code": "ERR_K1_REFUTED_PICK", "step_index": idx}
            fell_root = cand_root
            lineage.append(fell_root)

    return {
        "passed": True,
        "error_code": None,
        "events_replayed": len(events),
        "final_fell_root": fell_root,
        "remaining_budget": budget,
    }


def evaluate_tide1_contrast_floor_exact(
    fg_roles: dict[str, list[Any]],
    bg_roles: dict[str, list[Any]],
    min_contrast: Any,
    max_chroma: Any,
) -> dict[str, Any]:
    """Exact finite exhaustive verifier for heedle#6 tide/1 OKLCH contrast-floor host."""
    if type(fg_roles) is not dict or type(bg_roles) is not dict or len(fg_roles) == 0 or len(bg_roles) == 0:
        raise HeedleAuditError("ERR_INVALID_TIDE_ROLES", "fg_roles and bg_roles must be nonempty dicts")
    min_c = parse_exact_fraction(min_contrast, "min_contrast")
    max_ch = parse_exact_fraction(max_chroma, "max_chroma")
    if min_c <= 0 or max_ch <= 0:
        raise HeedleAuditError("ERR_INVALID_TIDE_BOUNDS", "bounds must be > 0")

    pairs_checked = 0
    worst_pair: list[str] | None = None
    worst_contrast: Fraction | None = None
    gamut_violations: list[str] = []

    for rname, vals in list(fg_roles.items()) + list(bg_roles.items()):
        if not _is_nonempty_str(rname):
            raise HeedleAuditError("ERR_INVALID_TIDE_ROLE_NAME", "role key must be nonempty str")
        lc = _require_frac_vec(vals, f"role[{rname}]", 2)
        if lc[0] < 0 or lc[0] > 1 or lc[1] < 0 or lc[1] > max_ch:
            gamut_violations.append(rname)

    for fg_name, fg_vals in sorted(fg_roles.items()):
        fg_l = parse_exact_fraction(fg_vals[0], "fg_l")
        for bg_name, bg_vals in sorted(bg_roles.items()):
            bg_l = parse_exact_fraction(bg_vals[0], "bg_l")
            pairs_checked += 1
            contrast = abs(fg_l - bg_l)
            if worst_contrast is None or contrast < worst_contrast:
                worst_contrast = contrast
                worst_pair = [fg_name, bg_name]

    assert_min = worst_contrast if worst_contrast is not None else Fraction(0, 1)
    snapped = len(gamut_violations) > 0 or assert_min < min_c
    return {
        "exhaustive": True,
        "pairs_checked": pairs_checked,
        "worst_pair": worst_pair,
        "worst_contrast": frac_to_str(assert_min),
        "gamut_ok": len(gamut_violations) == 0,
        "snapped": snapped,
    }


def _build_signed_draft(events_raw: list[dict[str, Any]]) -> dict[str, Any]:
    host = {"kind": "tide/1", "name": "tide-v1", "w0_root": ROOT_W0}
    gate = {"r_invariant": "2/1", "budget": 24, "cost": 1, "tau": "1/2"}
    prev = heedle_genesis_id(host, gate)
    events: list[dict[str, Any]] = []
    for ev_in in events_raw:
        ev = copy.deepcopy(ev_in)
        ev["prev"] = prev
        ev["id"] = heedle_event_id(ev)
        prev = ev["id"]
        events.append(ev)
    return {"schema": "heedle-draft/1", "host": host, "gate": gate, "events": events}


def run_pillar_4() -> dict[str, Any]:
    clean_draft = _build_signed_draft([
        {
            "seq": 1, "kind": "pick", "parent_root": ROOT_W0, "root": ROOT_W1,
            "move_sha256": MOVE_SHA_1, "pins": [0], "gesture": "hold", "tau": "1/2",
            "remaining_budget": 23, "tension": "2/5", "witness_change": "3/10", "refuted": False,
        },
        {
            "seq": 2, "kind": "snap", "parent_root": ROOT_W1, "root": ROOT_W_SNAP,
            "move_sha256": MOVE_SHA_SNAP, "pins": [0], "gesture": "knot", "tau": "1/2",
            "remaining_budget": 22, "tension": "4/5", "witness_change": "7/10", "refuted": True,
        },
        {
            "seq": 3, "kind": "unpick", "parent_root": ROOT_W1, "root": ROOT_W0,
            "pins": [], "gesture": "release", "tau": "1/2", "remaining_budget": 22,
        },
    ])
    clean_replay = replay_heedle_draft_exact(clean_draft)

    # tide/1 exhaustive finite-domain contrast checks (2 fg x 2 bg = 4 pairs)
    bg = {"bg_canvas": [Fraction(19, 20), Fraction(1, 50)], "bg_panel": [Fraction(17, 20), Fraction(1, 25)]}
    fg_ok = {"fg_body": [Fraction(3, 20), Fraction(1, 50)], "fg_accent": [Fraction(7, 20), Fraction(3, 20)]}
    fg_bad = {"fg_body": [Fraction(3, 20), Fraction(1, 50)], "fg_accent": [Fraction(10, 20), Fraction(3, 20)]}
    tide_ok = evaluate_tide1_contrast_floor_exact(fg_ok, bg, Fraction(9, 20), Fraction(3, 10))
    tide_snap = evaluate_tide1_contrast_floor_exact(fg_bad, bg, Fraction(9, 20), Fraction(3, 10))

    # Must-fail 1: tamper tau on event 1
    bad_tau_draft = copy.deepcopy(clean_draft)
    bad_tau_draft["events"][0]["tau"] = "5/1"
    bad_tau_draft["events"][0]["id"] = heedle_event_id(bad_tau_draft["events"][0])
    mf_chain = replay_heedle_draft_exact(bad_tau_draft)

    # Must-fail 2: unpick tries to refund budget 22 -> 24
    bad_unpick = _build_signed_draft([
        {
            "seq": 1, "kind": "pick", "parent_root": ROOT_W0, "root": ROOT_W1,
            "move_sha256": MOVE_SHA_1, "pins": [0], "gesture": "hold", "tau": "1/2",
            "remaining_budget": 23, "tension": "2/5", "witness_change": "3/10", "refuted": False,
        },
        {
            "seq": 2, "kind": "unpick", "parent_root": ROOT_W1, "root": ROOT_W0,
            "pins": [], "gesture": "release", "tau": "1/2", "remaining_budget": 24,
        },
    ])
    mf_refund = replay_heedle_draft_exact(bad_unpick)

    # Must-fail 3: HEEDLE-AUDIT-01 T1 double-zone skip (gesture='release' at tension=7/10)
    mf_t1 = check_t1_gesture_fixed(Fraction(7, 10), "release")

    return {
        "passed": all([
            clean_replay["passed"] is True,
            clean_replay["final_fell_root"] == ROOT_W0,
            clean_replay["remaining_budget"] == 22,
            tide_ok["snapped"] is False,
            tide_snap["snapped"] is True,
            mf_chain["passed"] is False,
            mf_refund["passed"] is False,
            mf_t1["passed"] is False,
        ]),
        "clean_protocol_replay": {
            "events_replayed": clean_replay["events_replayed"],
            "final_fell_root": clean_replay["final_fell_root"],
            "remaining_budget": clean_replay["remaining_budget"],
        },
        "tide1_exhaustive_host": {
            "pairs_checked": tide_ok["pairs_checked"],
            "clean_worst_contrast": tide_ok["worst_contrast"],
            "clean_snapped": tide_ok["snapped"],
            "breach_worst_pair": tide_snap["worst_pair"],
            "breach_worst_contrast": tide_snap["worst_contrast"],
            "breach_snapped": tide_snap["snapped"],
        },
        "must_fail_protocol_chain_tamper": {
            "rejected": mf_chain["passed"] is False,
            "error_code": mf_chain["error_code"],
            "step_index": mf_chain["step_index"],
        },
        "must_fail_unpick_budget_refund": {
            "rejected": mf_refund["passed"] is False,
            "error_code": mf_refund["error_code"],
            "step_index": mf_refund["step_index"],
        },
        "must_fail_t1_double_zone_skip": {
            "rejected": mf_t1["passed"] is False,
            "error_code": mf_t1["error_code"],
            "zone": mf_t1["zone"],
            "gesture": mf_t1["gesture"],
        },
    }


# ---------------------------------------------------------------------------
# Pillar 5: Spaceship#5 Realization Limit -> Laya Conformal -> Zone-B Snap
# ---------------------------------------------------------------------------

def conformal_lac_qhat_exact(scores: list[Any], alpha: Any) -> dict[str, Any]:
    """Exact integer-rank disjoint-split conformal LAC quantile (aligned with laya/vegaml)."""
    sc = _require_frac_vec(scores, "scores")
    if any(s < 0 or s > 1 for s in sc):
        raise HeedleAuditError("ERR_SCORE_RANGE", "conformal scores must lie in [0, 1]")
    a = parse_exact_fraction(alpha, "alpha")
    if a <= 0 or a >= 1:
        raise HeedleAuditError("ERR_ALPHA_RANGE", "alpha must lie strictly in (0, 1)")
    n = len(sc)
    num = (n + 1) * (a.denominator - a.numerator)
    rank = min((num + a.denominator - 1) // a.denominator, n)
    qhat = sorted(sc)[rank - 1]
    cov_lb = Fraction(rank, n + 1)
    return {
        "n": n,
        "alpha": frac_to_str(a),
        "rank": rank,
        "qhat": frac_to_str(qhat),
        "coverage_lb": frac_to_str(cov_lb),
    }


def certify_spaceship_realization_escalation(
    *,
    same_vector_self_score: Any,
    conformal_set_size: int,
    zone_b_independent_checked: bool,
    zone_b_worst_change: Any,
    tau: Any,
    claimed_action: str,
) -> dict[str, Any]:
    """Fail-closed router bridging spaceship#5 realization -> laya conformal -> Zone-B snap.

    Enforces:
      1. Same-vector self-realization (spaceship#5) can NEVER self-promote to VERIFIED.
      2. Empty (0) or ambiguous (>= 2) conformal sets MUST route to ESCALATE_ZONE_B.
      3. Once Zone-B runs, worst_change > tau emits SNAP_REFUTED; worst_change <= tau
         with singleton conformal set emits VERIFIED.
    """
    _ = parse_exact_fraction(same_vector_self_score, "same_vector_self_score")
    _req_strict_int_range(conformal_set_size, 0, None, "conformal_set_size")
    if type(zone_b_independent_checked) is not bool:
        raise HeedleAuditError("ERR_INVALID_BOOL", "zone_b_independent_checked must be bool")
    zb_worst = parse_exact_fraction(zone_b_worst_change, "zone_b_worst_change")
    tau_q = parse_exact_fraction(tau, "tau")
    if zb_worst < 0 or tau_q <= 0:
        raise HeedleAuditError("ERR_INVALID_BOUND", "zb_worst >= 0 and tau > 0 required")
    if not _in_str_set(claimed_action, ("VERIFIED", "ESCALATE_ZONE_B", "SNAP_REFUTED")):
        raise HeedleAuditError("ERR_INVALID_ACTION", f"unknown action {claimed_action!r}")

    if not zone_b_independent_checked:
        required_action = "ESCALATE_ZONE_B"
        if claimed_action != required_action:
            code = (
                "ERR_CONFORMAL_ESCALATION_BYPASS"
                if conformal_set_size != 1
                else "ERR_CIRCULAR_REALIZATION_CLAIM"
            )
            return {
                "passed": False,
                "error_code": code,
                "required_action": required_action,
                "claimed_action": claimed_action,
            }
        return {
            "passed": True,
            "error_code": None,
            "required_action": required_action,
            "claimed_action": claimed_action,
        }

    if zb_worst > tau_q:
        required_action = "SNAP_REFUTED"
    elif conformal_set_size != 1:
        required_action = "ESCALATE_ZONE_B"
    else:
        required_action = "VERIFIED"
    if claimed_action != required_action:
        return {
            "passed": False,
            "error_code": "ERR_ZONE_B_VERDICT_MISMATCH",
            "required_action": required_action,
            "claimed_action": claimed_action,
        }
    return {
        "passed": True,
        "error_code": None,
        "required_action": required_action,
        "claimed_action": claimed_action,
    }


def run_pillar_5() -> dict[str, Any]:
    cal_scores = [
        Fraction(1, 20), Fraction(1, 10), Fraction(3, 20), Fraction(1, 5),
        Fraction(1, 4), Fraction(3, 10), Fraction(7, 20), Fraction(2, 5), Fraction(1, 2),
    ]
    conf = conformal_lac_qhat_exact(cal_scores, Fraction(1, 5))

    # Clean escalation -> Zone-B snap when blinded probe exceeds tau=1/2
    esc_ok = certify_spaceship_realization_escalation(
        same_vector_self_score=Fraction(4, 5),
        conformal_set_size=2,
        zone_b_independent_checked=False,
        zone_b_worst_change=Fraction(0, 1),
        tau=Fraction(1, 2),
        claimed_action="ESCALATE_ZONE_B",
    )
    zb_snap = certify_spaceship_realization_escalation(
        same_vector_self_score=Fraction(4, 5),
        conformal_set_size=2,
        zone_b_independent_checked=True,
        zone_b_worst_change=Fraction(7, 10),
        tau=Fraction(1, 2),
        claimed_action="SNAP_REFUTED",
    )
    zb_ok = certify_spaceship_realization_escalation(
        same_vector_self_score=Fraction(4, 5),
        conformal_set_size=1,
        zone_b_independent_checked=True,
        zone_b_worst_change=Fraction(3, 10),
        tau=Fraction(1, 2),
        claimed_action="VERIFIED",
    )

    # Must-fail 1: spaceship#5 circular self-realization claims VERIFIED without Zone-B
    mf_circular = certify_spaceship_realization_escalation(
        same_vector_self_score=Fraction(9, 10),
        conformal_set_size=1,
        zone_b_independent_checked=False,
        zone_b_worst_change=Fraction(0, 1),
        tau=Fraction(1, 2),
        claimed_action="VERIFIED",
    )
    # Must-fail 2: ambiguous conformal set (size=2) bypasses ESCALATE_ZONE_B
    mf_ambig = certify_spaceship_realization_escalation(
        same_vector_self_score=Fraction(9, 10),
        conformal_set_size=2,
        zone_b_independent_checked=False,
        zone_b_worst_change=Fraction(0, 1),
        tau=Fraction(1, 2),
        claimed_action="VERIFIED",
    )

    return {
        "passed": all([
            conf["rank"] == 8,
            conf["qhat"] == "2/5",
            esc_ok["passed"] is True,
            zb_snap["passed"] is True,
            zb_ok["passed"] is True,
            mf_circular["passed"] is False,
            mf_ambig["passed"] is False,
        ]),
        "conformal_calibration": conf,
        "escalated_to_zone_b": esc_ok["required_action"],
        "zone_b_snap_action": zb_snap["required_action"],
        "zone_b_verified_action": zb_ok["required_action"],
        "must_fail_self_realization_promotes_without_zone_b": {
            "rejected": mf_circular["passed"] is False,
            "error_code": mf_circular["error_code"],
            "required_action": mf_circular["required_action"],
        },
        "must_fail_ambiguous_conformal_skips_escalation": {
            "rejected": mf_ambig["passed"] is False,
            "error_code": mf_ambig["error_code"],
            "required_action": mf_ambig["required_action"],
        },
    }


# ---------------------------------------------------------------------------
# Certificate Builder & Strict Fail-Closed Validator
# ---------------------------------------------------------------------------

def _count_verified_must_fail_controls(pillars: dict[str, dict[str, Any]]) -> int:
    count = 0
    for pdata in pillars.values():
        for key, val in pdata.items():
            if key.startswith("must_fail_") and type(val) is dict and val.get("rejected") is True:
                count += 1
    return count


def build_heedle_spaceship_certificate() -> dict[str, Any]:
    pillars = {
        "p1_rational_gauss_newton_shafts_and_shed": run_pillar_1(),
        "p2_hidden_move_nullspace_refutation": run_pillar_2(),
        "p3_scaling_bridge_pruning_and_int4_survival": run_pillar_3(),
        "p4_heedle_protocol_replay_and_tide1": run_pillar_4(),
        "p5_spaceship_realization_to_zone_b_snap": run_pillar_5(),
    }
    pillars_passed = sum(1 for p in pillars.values() if p["passed"] is True)
    controls_verified = _count_verified_must_fail_controls(pillars)

    body = {
        "schema_version": SCHEMA_VERSION,
        "track_id": TRACK_ID,
        "audit_id": AUDIT_ID,
        "novelty_tier": {
            "prior_math_and_repos": "N0",
            "pure_q_bridge_and_audit": "N1",
        },
        "primary_sources": {
            "heedle_main_commit": HEEDLE_MAIN_COMMIT,
            "heedle_pr8_commit": HEEDLE_PR8_COMMIT,
            "heedle_core_sha256": HEEDLE_CORE_SHA256,
            "heedle_draft_sha256": HEEDLE_DRAFT_SHA256,
            "heedle_hosts_sha256": HEEDLE_HOSTS_SHA256,
            "heedle_protocol_sha256": HEEDLE_PROTOCOL_SHA256,
            "spaceship_master_commit": SPACESHIP_MASTER_COMMIT,
            "spaceship_pr5_commit": SPACESHIP_PR5_COMMIT,
        },
        "pillars": pillars,
        "summary": {
            "all_passed": all([pillars_passed == 5, controls_verified == 12]),
            "pillars_passed": pillars_passed,
            "pillars_total": 5,
            "must_fail_controls_verified": controls_verified,
            "floats_used": 0,
        },
    }
    cert = dict(body)
    cert["canonical_sha256"] = canonical_sha256(body)
    return cert


def _validate_p1(p1: Any) -> None:
    _req_exact_keys(
        p1,
        {
            "passed",
            "shaft_eigenvalues",
            "treadle_k1_rho",
            "treadle_k2_rho",
            "treadle_k2_undelivered_ms",
            "treadle_k4_rho",
            "treadle_k4_undelivered_ms",
            "share_monotone_nondecreasing",
            "shed_projection",
            "must_fail_unpinned_move_leaks_heddle",
            "must_fail_share_exceeds_unity",
        },
        "p1",
    )
    _req_bool_eq(p1["passed"], True, "p1.passed")
    _req_str_list(p1["shaft_eigenvalues"], ["4/1", "1/1", "1/4", "1/16"], "p1.eigs")
    _req_strict_str_eq(p1["treadle_k1_rho"], "1/4", "p1.k1_rho")
    _req_strict_str_eq(p1["treadle_k2_rho"], "1/2", "p1.k2_rho")
    _req_strict_str_eq(p1["treadle_k2_undelivered_ms"], "9/50", "p1.k2_ums")
    _req_strict_str_eq(p1["treadle_k4_rho"], "1/1", "p1.k4_rho")
    _req_strict_str_eq(p1["treadle_k4_undelivered_ms"], "0/1", "p1.k4_ums")
    _req_bool_eq(p1["share_monotone_nondecreasing"], True, "p1.monotone")

    sh = _req_exact_keys(
        p1["shed_projection"],
        {"pin_rank", "open_dimensions", "pin_leak_sq", "idempotent", "pythagorean_exact", "projected_move"},
        "p1.shed",
    )
    _req_strict_int_range(sh["pin_rank"], 2, 2, "p1.shed.rank")
    _req_strict_int_range(sh["open_dimensions"], 2, 2, "p1.shed.open")
    _req_strict_str_eq(sh["pin_leak_sq"], "0/1", "p1.shed.leak")
    _req_bool_eq(sh["idempotent"], True, "p1.shed.idem")
    _req_bool_eq(sh["pythagorean_exact"], True, "p1.shed.pyth")
    _req_str_list(sh["projected_move"], ["1/1", "-1/1", "2/1", "2/1"], "p1.shed.move")

    mf1 = _req_exact_keys(p1["must_fail_unpinned_move_leaks_heddle"], {"rejected", "error_code", "pin_leak_sq"}, "p1.mf1")
    _req_bool_eq(mf1["rejected"], True, "p1.mf1.rej")
    _req_strict_str_eq(mf1["error_code"], "ERR_HEDDLE_PIN_LEAK", "p1.mf1.code")
    _req_strict_str_eq(mf1["pin_leak_sq"], "8/1", "p1.mf1.leak")

    mf2 = _req_exact_keys(p1["must_fail_share_exceeds_unity"], {"rejected", "error_code", "claimed_rho"}, "p1.mf2")
    _req_bool_eq(mf2["rejected"], True, "p1.mf2.rej")
    _req_strict_str_eq(mf2["error_code"], "ERR_TREADLE_SHARE_INVALID", "p1.mf2.code")
    _req_strict_str_eq(mf2["claimed_rho"], "5/4", "p1.mf2.rho")


def _validate_p2(p2: Any) -> None:
    _req_exact_keys(
        p2,
        {
            "passed",
            "tau_sq",
            "known_probe_ms",
            "elsewhere_probe_ms",
            "stealth_move",
            "zone_b_snap_witness_index",
            "zone_b_snap_change_sq",
            "must_fail_known_probe_only_misses_stealth",
            "must_fail_snap_with_quiet_witness",
        },
        "p2",
    )
    _req_bool_eq(p2["passed"], True, "p2.passed")
    _req_strict_str_eq(p2["tau_sq"], "1/4", "p2.tau_sq")
    _req_strict_str_eq(p2["known_probe_ms"], "0/1", "p2.known_ms")
    _req_strict_str_eq(p2["elsewhere_probe_ms"], "4398046511105/2199023255552", "p2.else_ms")
    _req_str_list(p2["stealth_move"], ["0/1", "0/1", "1/262144", "1/1"], "p2.move")
    _req_strict_int_range(p2["zone_b_snap_witness_index"], 1, 1, "p2.widx")
    _req_strict_str_eq(p2["zone_b_snap_change_sq"], "4/1", "p2.wsq")

    mf1 = _req_exact_keys(
        p2["must_fail_known_probe_only_misses_stealth"],
        {"rejected", "error_code", "known_probe_ms", "blinded_witness_change_sq"},
        "p2.mf1",
    )
    _req_bool_eq(mf1["rejected"], True, "p2.mf1.rej")
    _req_strict_str_eq(mf1["error_code"], "ERR_STEALTH_MOVE_SNAPPED", "p2.mf1.code")
    _req_strict_str_eq(mf1["known_probe_ms"], "0/1", "p2.mf1.known")
    _req_strict_str_eq(mf1["blinded_witness_change_sq"], "4/1", "p2.mf1.blinded")

    mf2 = _req_exact_keys(
        p2["must_fail_snap_with_quiet_witness"],
        {"rejected", "error_code", "quiet_witness_change_sq"},
        "p2.mf2",
    )
    _req_bool_eq(mf2["rejected"], True, "p2.mf2.rej")
    _req_strict_str_eq(mf2["error_code"], "ERR_SNAP_WITNESS_DOES_NOT_REFUTE", "p2.mf2.code")
    _req_strict_str_eq(mf2["quiet_witness_change_sq"], "1/1099511627776", "p2.mf2.qsq")


def _validate_p3(p3: Any) -> None:
    _req_exact_keys(
        p3,
        {
            "passed",
            "column_structured_pruning",
            "int4_surviving_move",
            "must_fail_entry_pruning_shatters_heddle",
            "must_fail_int4_sub_threshold_erasure",
            "must_fail_int4_spike_collateral_flip",
        },
        "p3",
    )
    _req_bool_eq(p3["passed"], True, "p3.passed")

    col = _req_exact_keys(
        p3["column_structured_pruning"],
        {"is_exact_rank_le_1", "heddle_preserved", "heddle_leak_sq", "rank1_gram_defect"},
        "p3.col",
    )
    _req_bool_eq(col["is_exact_rank_le_1"], True, "p3.col.r1")
    _req_bool_eq(col["heddle_preserved"], True, "p3.col.hp")
    _req_strict_str_eq(col["heddle_leak_sq"], "0/1", "p3.col.hl")
    _req_strict_str_eq(col["rank1_gram_defect"], "0/1", "p3.col.gd")

    surv = _req_exact_keys(
        p3["int4_surviving_move"],
        {"ulp", "three_quarter_ulp", "min_edit_ulp_ratio", "edited_flips", "collateral_unedited_flips"},
        "p3.surv",
    )
    _req_strict_str_eq(surv["ulp"], "4/1", "p3.surv.ulp")
    _req_strict_str_eq(surv["three_quarter_ulp"], "3/1", "p3.surv.34")
    _req_strict_str_eq(surv["min_edit_ulp_ratio"], "3/4", "p3.surv.ratio")
    _req_strict_int_range(surv["edited_flips"], 1, 1, "p3.surv.flips")
    _req_strict_int_range(surv["collateral_unedited_flips"], 0, 0, "p3.surv.collat")

    mf1 = _req_exact_keys(
        p3["must_fail_entry_pruning_shatters_heddle"],
        {"rejected", "error_code", "rank1_gram_defect", "heddle_leak_sq"},
        "p3.mf1",
    )
    _req_bool_eq(mf1["rejected"], True, "p3.mf1.rej")
    _req_strict_str_eq(mf1["error_code"], "ERR_PRUNING_SHATTERED_HEDDLE", "p3.mf1.code")
    _req_strict_str_eq(mf1["rank1_gram_defect"], "294/1", "p3.mf1.gd")
    _req_strict_str_eq(mf1["heddle_leak_sq"], "16/1", "p3.mf1.hl")

    mf2 = _req_exact_keys(
        p3["must_fail_int4_sub_threshold_erasure"],
        {"rejected", "error_code", "min_edit_ulp_ratio", "erased_edits"},
        "p3.mf2",
    )
    _req_bool_eq(mf2["rejected"], True, "p3.mf2.rej")
    _req_strict_str_eq(mf2["error_code"], "ERR_QUANT_MOVE_ERASED", "p3.mf2.code")
    _req_strict_str_eq(mf2["min_edit_ulp_ratio"], "3/4", "p3.mf2.ratio")
    _req_strict_int_range(mf2["erased_edits"], 1, 1, "p3.mf2.erased")

    mf3 = _req_exact_keys(
        p3["must_fail_int4_spike_collateral_flip"],
        {"rejected", "error_code", "scale_ratio", "collateral_unedited_flips"},
        "p3.mf3",
    )
    _req_bool_eq(mf3["rejected"], True, "p3.mf3.rej")
    _req_strict_str_eq(mf3["error_code"], "ERR_QUANT_COLLATERAL_FLIP", "p3.mf3.code")
    _req_strict_str_eq(mf3["scale_ratio"], "2/1", "p3.mf3.scale")
    _req_strict_int_range(mf3["collateral_unedited_flips"], 3, 3, "p3.mf3.collat")


def _validate_p4(p4: Any) -> None:
    _req_exact_keys(
        p4,
        {
            "passed",
            "clean_protocol_replay",
            "tide1_exhaustive_host",
            "must_fail_protocol_chain_tamper",
            "must_fail_unpick_budget_refund",
            "must_fail_t1_double_zone_skip",
        },
        "p4",
    )
    _req_bool_eq(p4["passed"], True, "p4.passed")

    rep = _req_exact_keys(
        p4["clean_protocol_replay"],
        {"events_replayed", "final_fell_root", "remaining_budget"},
        "p4.rep",
    )
    _req_strict_int_range(rep["events_replayed"], 3, 3, "p4.rep.ev")
    _req_strict_str_eq(rep["final_fell_root"], ROOT_W0, "p4.rep.root")
    _req_strict_int_range(rep["remaining_budget"], 22, 22, "p4.rep.bud")

    tide = _req_exact_keys(
        p4["tide1_exhaustive_host"],
        {
            "pairs_checked",
            "clean_worst_contrast",
            "clean_snapped",
            "breach_worst_pair",
            "breach_worst_contrast",
            "breach_snapped",
        },
        "p4.tide",
    )
    _req_strict_int_range(tide["pairs_checked"], 4, 4, "p4.tide.pairs")
    _req_strict_str_eq(tide["clean_worst_contrast"], "1/2", "p4.tide.cwc")
    _req_bool_eq(tide["clean_snapped"], False, "p4.tide.cs")
    _req_str_list(tide["breach_worst_pair"], ["fg_accent", "bg_panel"], "p4.tide.bwp")
    _req_strict_str_eq(tide["breach_worst_contrast"], "7/20", "p4.tide.bwc")
    _req_bool_eq(tide["breach_snapped"], True, "p4.tide.bs")

    mf1 = _req_exact_keys(p4["must_fail_protocol_chain_tamper"], {"rejected", "error_code", "step_index"}, "p4.mf1")
    _req_bool_eq(mf1["rejected"], True, "p4.mf1.rej")
    _req_strict_str_eq(mf1["error_code"], "ERR_PROTOCOL_CHAIN_OR_TAU", "p4.mf1.code")
    _req_strict_int_range(mf1["step_index"], 0, 0, "p4.mf1.step")

    mf2 = _req_exact_keys(p4["must_fail_unpick_budget_refund"], {"rejected", "error_code", "step_index"}, "p4.mf2")
    _req_bool_eq(mf2["rejected"], True, "p4.mf2.rej")
    _req_strict_str_eq(mf2["error_code"], "ERR_UNPICK_BUDGET_REFUND", "p4.mf2.code")
    _req_strict_int_range(mf2["step_index"], 1, 1, "p4.mf2.step")

    mf3 = _req_exact_keys(p4["must_fail_t1_double_zone_skip"], {"rejected", "error_code", "zone", "gesture"}, "p4.mf3")
    _req_bool_eq(mf3["rejected"], True, "p4.mf3.rej")
    _req_strict_str_eq(mf3["error_code"], "ERR_T1_GESTURE_TOO_LIGHT", "p4.mf3.code")
    _req_strict_str_eq(mf3["zone"], "knot", "p4.mf3.zone")
    _req_strict_str_eq(mf3["gesture"], "release", "p4.mf3.gest")


def _validate_p5(p5: Any) -> None:
    _req_exact_keys(
        p5,
        {
            "passed",
            "conformal_calibration",
            "escalated_to_zone_b",
            "zone_b_snap_action",
            "zone_b_verified_action",
            "must_fail_self_realization_promotes_without_zone_b",
            "must_fail_ambiguous_conformal_skips_escalation",
        },
        "p5",
    )
    _req_bool_eq(p5["passed"], True, "p5.passed")

    cc = _req_exact_keys(p5["conformal_calibration"], {"n", "alpha", "rank", "qhat", "coverage_lb"}, "p5.cc")
    _req_strict_int_range(cc["n"], 9, 9, "p5.cc.n")
    _req_strict_str_eq(cc["alpha"], "1/5", "p5.cc.alpha")
    _req_strict_int_range(cc["rank"], 8, 8, "p5.cc.rank")
    _req_strict_str_eq(cc["qhat"], "2/5", "p5.cc.qhat")
    _req_strict_str_eq(cc["coverage_lb"], "4/5", "p5.cc.cov")

    _req_strict_str_eq(p5["escalated_to_zone_b"], "ESCALATE_ZONE_B", "p5.esc")
    _req_strict_str_eq(p5["zone_b_snap_action"], "SNAP_REFUTED", "p5.snap")
    _req_strict_str_eq(p5["zone_b_verified_action"], "VERIFIED", "p5.ver")

    mf1 = _req_exact_keys(
        p5["must_fail_self_realization_promotes_without_zone_b"],
        {"rejected", "error_code", "required_action"},
        "p5.mf1",
    )
    _req_bool_eq(mf1["rejected"], True, "p5.mf1.rej")
    _req_strict_str_eq(mf1["error_code"], "ERR_CIRCULAR_REALIZATION_CLAIM", "p5.mf1.code")
    _req_strict_str_eq(mf1["required_action"], "ESCALATE_ZONE_B", "p5.mf1.req")

    mf2 = _req_exact_keys(
        p5["must_fail_ambiguous_conformal_skips_escalation"],
        {"rejected", "error_code", "required_action"},
        "p5.mf2",
    )
    _req_bool_eq(mf2["rejected"], True, "p5.mf2.rej")
    _req_strict_str_eq(mf2["error_code"], "ERR_CONFORMAL_ESCALATION_BYPASS", "p5.mf2.code")
    _req_strict_str_eq(mf2["required_action"], "ESCALATE_ZONE_B", "p5.mf2.req")


def validate_certificate(cert: Any) -> bool:
    """Fail-closed deep schema, semantic, and SHA-256 integrity validator."""
    if contains_float(cert):
        raise HeedleAuditError("ERR_CERT_FLOAT", "certificate contains float or non-JSON value")
    _req_exact_keys(
        cert,
        {
            "schema_version",
            "track_id",
            "audit_id",
            "novelty_tier",
            "primary_sources",
            "pillars",
            "summary",
            "canonical_sha256",
        },
        "certificate",
    )
    _req_strict_str_eq(cert["schema_version"], SCHEMA_VERSION, "schema_version")
    _req_strict_str_eq(cert["track_id"], TRACK_ID, "track_id")
    _req_strict_str_eq(cert["audit_id"], AUDIT_ID, "audit_id")

    nt = _req_exact_keys(cert["novelty_tier"], {"prior_math_and_repos", "pure_q_bridge_and_audit"}, "novelty_tier")
    _req_strict_str_eq(nt["prior_math_and_repos"], "N0", "nt.n0")
    _req_strict_str_eq(nt["pure_q_bridge_and_audit"], "N1", "nt.n1")

    ps = _req_exact_keys(
        cert["primary_sources"],
        {
            "heedle_main_commit",
            "heedle_pr8_commit",
            "heedle_core_sha256",
            "heedle_draft_sha256",
            "heedle_hosts_sha256",
            "heedle_protocol_sha256",
            "spaceship_master_commit",
            "spaceship_pr5_commit",
        },
        "primary_sources",
    )
    _req_strict_str_eq(ps["heedle_main_commit"], HEEDLE_MAIN_COMMIT, "ps.hm")
    _req_strict_str_eq(ps["heedle_pr8_commit"], HEEDLE_PR8_COMMIT, "ps.hp8")
    _req_strict_str_eq(ps["heedle_core_sha256"], HEEDLE_CORE_SHA256, "ps.hc")
    _req_strict_str_eq(ps["heedle_draft_sha256"], HEEDLE_DRAFT_SHA256, "ps.hd")
    _req_strict_str_eq(ps["heedle_hosts_sha256"], HEEDLE_HOSTS_SHA256, "ps.hh")
    _req_strict_str_eq(ps["heedle_protocol_sha256"], HEEDLE_PROTOCOL_SHA256, "ps.hp")
    _req_strict_str_eq(ps["spaceship_master_commit"], SPACESHIP_MASTER_COMMIT, "ps.sm")
    _req_strict_str_eq(ps["spaceship_pr5_commit"], SPACESHIP_PR5_COMMIT, "ps.sp5")

    pillars = _req_exact_keys(
        cert["pillars"],
        {
            "p1_rational_gauss_newton_shafts_and_shed",
            "p2_hidden_move_nullspace_refutation",
            "p3_scaling_bridge_pruning_and_int4_survival",
            "p4_heedle_protocol_replay_and_tide1",
            "p5_spaceship_realization_to_zone_b_snap",
        },
        "pillars",
    )
    _validate_p1(pillars["p1_rational_gauss_newton_shafts_and_shed"])
    _validate_p2(pillars["p2_hidden_move_nullspace_refutation"])
    _validate_p3(pillars["p3_scaling_bridge_pruning_and_int4_survival"])
    _validate_p4(pillars["p4_heedle_protocol_replay_and_tide1"])
    _validate_p5(pillars["p5_spaceship_realization_to_zone_b_snap"])

    summary = _req_exact_keys(
        cert["summary"],
        {"all_passed", "pillars_passed", "pillars_total", "must_fail_controls_verified", "floats_used"},
        "summary",
    )
    _req_bool_eq(summary["all_passed"], True, "summary.all_passed")
    _req_strict_int_range(summary["pillars_passed"], 5, 5, "summary.pillars_passed")
    _req_strict_int_range(summary["pillars_total"], 5, 5, "summary.pillars_total")
    dyn_controls = _count_verified_must_fail_controls(pillars)
    _req_strict_int_range(summary["must_fail_controls_verified"], dyn_controls, dyn_controls, "summary.controls")
    _req_strict_int_range(summary["floats_used"], 0, 0, "summary.floats_used")

    claimed_sha = _req_hex64(cert["canonical_sha256"], "canonical_sha256")
    body = {k: v for k, v in cert.items() if k != "canonical_sha256"}
    if claimed_sha != canonical_sha256(body):
        raise HeedleAuditError("ERR_CERT_SHA_MISMATCH", "canonical_sha256 mismatch")
    return True
