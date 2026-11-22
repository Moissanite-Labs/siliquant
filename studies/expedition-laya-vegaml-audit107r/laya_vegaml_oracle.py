"""Pure-Q exact oracle for AUDIT-107 (Track S10): Laya & VegaML operator bridges
into Aletheia v2.1, Moissanite-Labs/siliquant, and Moissanite-Labs/warp-drive.

De-metaphorized Stage-B exact operators (Law 5, zero physical metaphor):
  - Pillar 1: Exact integer-rank split-conformal LAC quantile, IEEE-754 ceil overshoot
              witness, and exact finite-sample combinatoric in-sample coupling counterexample.
  - Pillar 2: Exact empty-conformal-set non-trigger iff theorem (qhat < 1 - 1/k),
              sqrt-free pure-integer squared Wilson escalation bound, and fail-closed
              union router closing laya/vega.py's unbound & Band.route's unbanded gaps.
  - Pillar 3: Exact 5-state epistemic meet-semilattice (PRESENT/INSUFFICIENT/UNKNOWN/
              MISSING/INCOMPARABLE) -> Aletheia v2.1 ClaimLedger promotion lattice,
              closing laya/evidence.py's negative-items -> PRESENT bug.
  - Pillar 4: Exact pure-Q closed-form ridge linear port recovery under quantized-reader
              affine drift + orthogonal rounding noise (siliquant x vega-U / vegaml#40 GGUF),
              plus exact laya#781 4-level Markov cap & lexical threshold-impossibility theorem.

Strict discipline: pure Python 3 stdlib, exact int/Fraction only, zero floats,
zero bare asserts, fail-closed validation on every trust boundary.
"""

from __future__ import annotations

from fractions import Fraction
import hashlib
import itertools
import json
import math
import re
from typing import Any, Mapping, Sequence

SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z")
GIT_SHA40_RE = re.compile(r"\A[0-9a-f]{40}\Z")
DECIMAL_STR_RE = re.compile(r"\A-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?\Z")

EVIDENCE_PRESENT = "PRESENT"
EVIDENCE_INSUFFICIENT = "INSUFFICIENT"
EVIDENCE_UNKNOWN = "UNKNOWN"
EVIDENCE_MISSING = "MISSING"
EVIDENCE_INCOMPARABLE = "INCOMPARABLE"

EVIDENCE_STATES: tuple[str, ...] = (
    EVIDENCE_INCOMPARABLE,
    EVIDENCE_MISSING,
    EVIDENCE_UNKNOWN,
    EVIDENCE_INSUFFICIENT,
    EVIDENCE_PRESENT,
)

EVIDENCE_RANK: dict[str, int] = {
    EVIDENCE_INCOMPARABLE: 0,
    EVIDENCE_MISSING: 1,
    EVIDENCE_UNKNOWN: 2,
    EVIDENCE_INSUFFICIENT: 3,
    EVIDENCE_PRESENT: 4,
}

CLAIM_LEDGER_MAP: dict[str, str] = {
    EVIDENCE_PRESENT: "ELIGIBLE_FOR_VERIFIED",
    EVIDENCE_INSUFFICIENT: "PARTIAL_FRAGILE",
    EVIDENCE_UNKNOWN: "UNVERIFIED_HYPOTHESIS",
    EVIDENCE_MISSING: "NOT_SHOWN",
    EVIDENCE_INCOMPARABLE: "CONTRADICTED_FALSIFIED",
}


class LayaVegamlError(ValueError):
    """Fail-closed validation error for exact Laya/VegaML audit primitives."""


def require_exact_int(value: Any, name: str, *, allow_zero: bool = True) -> int:
    if type(value) is not int:
        raise LayaVegamlError(f"{name} must be an exact int (got {type(value).__name__})")
    if allow_zero:
        if value < 0:
            raise LayaVegamlError(f"{name} must be non-negative (got {value})")
    else:
        if value <= 0:
            raise LayaVegamlError(f"{name} must be positive (got {value})")
    return value


def require_exact_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise LayaVegamlError(f"{name} must be an exact bool (got {type(value).__name__})")
    return value


def require_sha256(value: Any, name: str) -> str:
    if type(value) is not str or not SHA256_RE.match(value) or len(set(value)) == 1:
        raise LayaVegamlError(f"{name} must be a non-degenerate 64-char lowercase hex sha256")
    return value


def require_git_sha40(value: Any, name: str) -> str:
    if type(value) is not str or not GIT_SHA40_RE.match(value) or len(set(value)) == 1:
        raise LayaVegamlError(f"{name} must be a non-degenerate 40-char lowercase hex git sha")
    return value


def require_exact_fraction(value: Any, name: str) -> Fraction:
    if type(value) is not Fraction:
        raise LayaVegamlError(f"{name} must be an exact Fraction (got {type(value).__name__})")
    return value


def parse_exact_decimal_to_fraction(text: Any, name: str) -> Fraction:
    """Parse a strict canonical decimal string (no '+', '_', whitespace, '-0', or sci notation) into Fraction."""
    if type(text) is not str or len(text) > 64 or not DECIMAL_STR_RE.match(text):
        raise LayaVegamlError(f"{name} must be a strict decimal string")
    neg = text.startswith("-")
    body = text[1:] if neg else text
    if "." in body:
        whole_s, frac_s = body.split(".", 1)
        scale = 10 ** len(frac_s)
        num = int(whole_s) * scale + int(frac_s)
    else:
        scale = 1
        num = int(body)
    if neg and num == 0:
        raise LayaVegamlError(f"{name} must not be negative zero")
    return Fraction(-num if neg else num, scale)


def contains_float_recursive(obj: Any) -> bool:
    """Return True if any float, complex, bytes, set, frozenset, or non-JSON scalar exists."""
    if obj is None or type(obj) in (bool, int, str):
        return False
    if isinstance(obj, Mapping):
        return any(
            type(k) is not str or contains_float_recursive(v)
            for k, v in obj.items()
        )
    if type(obj) in (list, tuple):
        return any(contains_float_recursive(item) for item in obj)
    return True


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    """Compute canonical SHA-256 over a strict JSON-safe mapping with zero floats."""
    if not isinstance(payload, Mapping) or contains_float_recursive(payload):
        raise LayaVegamlError("canonical_sha256 payload must be a float-free JSON mapping")
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def frac_to_dict(value: Fraction) -> dict[str, int]:
    require_exact_fraction(value, "value")
    return {"num": value.numerator, "den": value.denominator}


def format_exact_decimal(value: Fraction, places: int = 6) -> str:
    require_exact_fraction(value, "value")
    require_exact_int(places, "places", allow_zero=True)
    sign = "-" if value < 0 else ""
    mag = -value if value < 0 else value
    scale = 10 ** places
    scaled = (mag.numerator * scale * 2 + mag.denominator) // (2 * mag.denominator)
    whole, frac = divmod(scaled, scale)
    if places == 0:
        return f"{sign}{whole}"
    return f"{sign}{whole}.{frac:0{places}d}"


def format_exact_percent(value: Fraction, places: int = 2) -> str:
    require_exact_fraction(value, "value")
    return format_exact_decimal(value * 100, places=places) + "%"


def _require_unit_fraction(value: Any, name: str, *, strict_open: bool = False) -> Fraction:
    frac = require_exact_fraction(value, name)
    if strict_open:
        if frac <= 0 or frac >= 1:
            raise LayaVegamlError(f"{name} must lie strictly in (0, 1)")
    else:
        if frac < 0 or frac > 1:
            raise LayaVegamlError(f"{name} must lie in [0, 1]")
    return frac


def _f64_dyadic_rne_fraction(value: Fraction) -> Fraction:
    """Round any positive normal-range Fraction to its exact IEEE-754 binary64 dyadic value (pure ints)."""
    require_exact_fraction(value, "value")
    if value <= 0:
        raise LayaVegamlError("dyadic RNE helper requires strictly positive Fraction")
    num = value.numerator
    den = value.denominator
    exp = num.bit_length() - den.bit_length()
    if exp >= 0:
        if num < (den << exp):
            exp -= 1
    elif (num << (-exp)) < den:
        exp -= 1
    if exp < -1022 or exp > 1023:
        raise LayaVegamlError("value outside normal IEEE-754 binary64 exponent range")
    shift = 52 - exp
    if shift >= 0:
        scaled_num = num << shift
        scaled_den = den
    else:
        scaled_num = num
        scaled_den = den << (-shift)
    q, rem = divmod(scaled_num, scaled_den)
    twice_rem = rem * 2
    if twice_rem > scaled_den or (twice_rem == scaled_den and (q & 1) == 1):
        q += 1
    if q == (1 << 53) and exp == 1023:
        raise LayaVegamlError("value overflows IEEE-754 binary64")
    if shift >= 0:
        return Fraction(q, 1 << shift)
    return Fraction(q << (-shift), 1)


# =============================================================================
# PILLAR 1: Exact Split-Conformal LAC & In-Sample Coupling Counterexample
# =============================================================================

def conformal_lac_qhat_exact(scores: Sequence[Fraction], alpha: Fraction) -> dict[str, Any]:
    """Exact integer-rank split-conformal LAC quantile (zero floating-point 1.0 - alpha drift)."""
    if not isinstance(scores, Sequence) or isinstance(scores, (str, bytes)) or len(scores) == 0:
        raise LayaVegamlError("scores must be a non-empty sequence of Fractions in [0, 1]")
    cleaned = [_require_unit_fraction(s, f"scores[{idx}]") for idx, s in enumerate(scores)]
    a = _require_unit_fraction(alpha, "alpha", strict_open=True)
    n = len(cleaned)
    num = (n + 1) * (a.denominator - a.numerator)
    rank_unclamped = (num + a.denominator - 1) // a.denominator
    rank = n if rank_unclamped > n else rank_unclamped
    ordered = sorted(cleaned)
    qhat = ordered[rank - 1]
    cov_lb = Fraction(rank, n + 1)
    return {
        "n": n,
        "alpha": frac_to_dict(a),
        "rank": rank,
        "qhat": frac_to_dict(qhat),
        "finite_sample_coverage_lb": frac_to_dict(cov_lb),
        "meets_nominal_target": cov_lb >= Fraction(1) - a,
    }


def certify_ieee754_conformal_ceil_overshoot_exact(n: int, alpha: Fraction) -> dict[str, Any]:
    """Pure-integer exact IEEE-754 binary64 proof of math.ceil((n + 1) * (1.0 - alpha)) vs exact integer rank."""
    require_exact_int(n, "n", allow_zero=False)
    a = _require_unit_fraction(alpha, "alpha", strict_open=True)
    exact_prod = Fraction(n + 1, 1) * (Fraction(1, 1) - a)
    exact_rank = min((exact_prod.numerator + exact_prod.denominator - 1) // exact_prod.denominator, n)
    a_f64 = _f64_dyadic_rne_fraction(a)
    one_minus_exact = Fraction(1, 1) - a_f64
    if one_minus_exact == 0:
        f64_ceil = 0
    else:
        one_minus_f64 = _f64_dyadic_rne_fraction(one_minus_exact)
        n1_f64 = _f64_dyadic_rne_fraction(Fraction(n + 1, 1))
        prod_f64 = _f64_dyadic_rne_fraction(n1_f64 * one_minus_f64)
        f64_ceil = min((prod_f64.numerator + prod_f64.denominator - 1) // prod_f64.denominator, n)
    return {
        "n": n,
        "alpha": frac_to_dict(a),
        "exact_product": frac_to_dict(exact_prod),
        "exact_rank": exact_rank,
        "binary64_rank": f64_ceil,
        "overshoot": f64_ceil - exact_rank,
        "overshoot_witnessed": f64_ceil > exact_rank,
    }


def certify_insample_coupling_counterexample_exact() -> dict[str, Any]:
    """Exact combinatoric theorem over all C(6,3)=20 splits proving in-sample parameter+qhat coupling breaks 1 - alpha.

    Directly models vegaml/calibrate.py:339-341 where temperature/model selection and conformal qhat
    are fitted on the identical rows rather than disjoint splits (Aletheia #10 Gate 4/5).
    """
    n_pop = 6
    n_cal = 3
    alpha = Fraction(1, 4)
    target_cov = Fraction(1, 1) - alpha
    rank = ((n_cal + 1) * (alpha.denominator - alpha.numerator) + alpha.denominator - 1) // alpha.denominator

    sym_subsets = [set(comb) for comb in itertools.combinations(range(n_pop), n_cal) if sum(comb) % 2 == 0]
    candidates = [
        [Fraction(1, 10) if i in subset else Fraction(9, 10) for i in range(n_pop)]
        for subset in sym_subsets
    ]

    disjoint_hits = 0
    coupled_hits = 0
    total_test_evals = 0
    zero_coverage_coupled_splits = 0
    n_splits = 0

    for cal in itertools.combinations(range(n_pop), n_cal):
        n_splits += 1
        cal_set = set(cal)
        test = [i for i in range(n_pop) if i not in cal_set]

        q_fixed = sorted(candidates[0][i] for i in cal)[rank - 1]
        disjoint_hits += sum(1 for i in test if candidates[0][i] <= q_fixed)

        best_idx = min(range(len(candidates)), key=lambda m: (sum(candidates[m][i] for i in cal), m))
        q_coupled = sorted(candidates[best_idx][i] for i in cal)[rank - 1]
        c_hits = sum(1 for i in test if candidates[best_idx][i] <= q_coupled)
        coupled_hits += c_hits
        if c_hits == 0:
            zero_coverage_coupled_splits += 1
        total_test_evals += len(test)

    disjoint_cov = Fraction(disjoint_hits, total_test_evals)
    coupled_cov = Fraction(coupled_hits, total_test_evals)
    ceil_149 = certify_ieee754_conformal_ceil_overshoot_exact(149, Fraction(18, 100))
    ceil_29 = certify_ieee754_conformal_ceil_overshoot_exact(29, Fraction(7, 10))
    return {
        "n_population": n_pop,
        "n_calibration": n_cal,
        "n_candidate_calibrators": len(candidates),
        "n_exact_splits": n_splits,
        "alpha": frac_to_dict(alpha),
        "nominal_target_coverage": frac_to_dict(target_cov),
        "disjoint_exact_coverage": frac_to_dict(disjoint_cov),
        "disjoint_exact_coverage_pct": format_exact_percent(disjoint_cov, 2),
        "coupled_exact_coverage": frac_to_dict(coupled_cov),
        "coupled_exact_coverage_pct": format_exact_percent(coupled_cov, 2),
        "coverage_deficit": frac_to_dict(target_cov - coupled_cov),
        "zero_coverage_coupled_splits": zero_coverage_coupled_splits,
        "ieee754_overshoot_n149_a018": ceil_149,
        "ieee754_overshoot_n29_a070": ceil_29,
        "theorem_holds": (
            disjoint_cov >= target_cov
            and coupled_cov < target_cov
            and ceil_149["overshoot_witnessed"]
            and ceil_29["overshoot_witnessed"]
        ),
    }


# =============================================================================
# PILLAR 2: Empty-Conformal Non-Trigger, Squared Wilson Bound & Fail-Closed Router
# =============================================================================

def certify_empty_conformal_non_trigger_exact(k: int, qhat: Fraction) -> dict[str, Any]:
    """Exact iff proof over Q: empty LAC set is reachable iff qhat < 1 - 1/k (vegaml/band.py:13-17)."""
    require_exact_int(k, "k", allow_zero=False)
    if k < 2:
        raise LayaVegamlError("k must be at least 2 options")
    q = _require_unit_fraction(qhat, "qhat")
    boundary = Fraction(k - 1, k)
    cut = Fraction(1, 1) - q
    uniform_p = Fraction(1, k)
    uniform_pset_size = k if uniform_p >= cut else 0
    reachable = q < boundary
    return {
        "k": k,
        "qhat": frac_to_dict(q),
        "empty_set_boundary": frac_to_dict(boundary),
        "probability_cutoff": frac_to_dict(cut),
        "uniform_pset_size": uniform_pset_size,
        "abstention_reachable": reachable,
        "iff_verified": reachable == (uniform_pset_size == 0),
    }


def wilson_lower_meets_target_exact(
    hits: int,
    n: int,
    z: Fraction,
    target: Fraction,
) -> dict[str, Any]:
    """Exact sqrt-free integer cross-multiplication test for Wilson lower bound >= target.

    Algebraic identity: L_Wilson(h, n, z) >= tau iff
      (1) tau <= (h + z^2 / 2) / (n + z^2), AND
      (2) n * (n + z^2) * tau^2 - n * (2*h + z^2) * tau + h^2 >= 0.
    Also provides exact integer-isqrt 6-decimal bracket [lb_floor, lb_ceil].
    """
    require_exact_int(n, "n", allow_zero=False)
    require_exact_int(hits, "hits", allow_zero=True)
    if hits > n:
        raise LayaVegamlError(f"hits ({hits}) cannot exceed n ({n})")
    zf = require_exact_fraction(z, "z")
    if zf <= 0:
        raise LayaVegamlError("z must be a positive Fraction")
    tau = _require_unit_fraction(target, "target", strict_open=True)

    z2 = zf * zf
    zn, zd = z2.numerator, z2.denominator
    tn, td = tau.numerator, tau.denominator

    center_lhs = 2 * tn * (n * zd + zn)
    center_rhs = td * (2 * hits * zd + zn)
    center_ok = center_lhs <= center_rhs

    quad_lhs = n * (n * zd + zn) * tn * tn + hits * hits * zd * td * td
    quad_rhs = n * (2 * hits * zd + zn) * tn * td
    quad_ok = quad_lhs >= quad_rhs
    meets = center_ok and quad_ok

    scale = 1_000_000
    num_c = n * (2 * hits * zd + zn)
    denom = 2 * n * (n * zd + zn)
    radicand = n * zn * (4 * hits * (n - hits) * zd + n * zn)
    rad_scaled = radicand * scale * scale
    root_floor = math.isqrt(rad_scaled)
    root_ceil = root_floor if root_floor * root_floor == rad_scaled else root_floor + 1
    lb_floor_num = max(0, (num_c * scale - root_ceil) // denom)
    lb_ceil_num = max(0, (num_c * scale - root_floor + denom - 1) // denom)

    return {
        "hits": hits,
        "n": n,
        "z": frac_to_dict(zf),
        "target": frac_to_dict(tau),
        "center_ok": center_ok,
        "quadratic_ok": quad_ok,
        "meets_target": meets,
        "lb_floor_6dp": format_exact_decimal(Fraction(lb_floor_num, scale), 6),
        "lb_ceil_6dp": format_exact_decimal(Fraction(lb_ceil_num, scale), 6),
    }


def route_answer_fail_closed_exact(
    answer: Mapping[str, Any],
    *,
    band_threshold: Fraction | None,
    band_reachable: bool,
    max_set: int = 1,
) -> dict[str, Any]:
    """Fail-closed union router: route='review' iff ANY uncertainty/OOD/band trigger fires."""
    if not isinstance(answer, Mapping):
        raise LayaVegamlError("answer must be a mapping")
    require_exact_bool(band_reachable, "band_reachable")
    require_exact_int(max_set, "max_set", allow_zero=False)
    if band_threshold is not None:
        _require_unit_fraction(band_threshold, "band_threshold")

    conf = answer.get("confidence")
    if conf is not None:
        _require_unit_fraction(conf, "answer.confidence")
    unbound = answer.get("unbound")
    if unbound is not None:
        require_exact_bool(unbound, "answer.unbound")
    abstain = answer.get("abstain")
    if abstain is not None:
        require_exact_bool(abstain, "answer.abstain")
    pset = answer.get("prediction_set")
    if pset is not None and (not isinstance(pset, Sequence) or isinstance(pset, (str, bytes))):
        raise LayaVegamlError("answer.prediction_set must be a sequence or None")

    triggers: list[str] = []
    if band_threshold is None:
        triggers.append("band_unconfigured")
    if not band_reachable:
        triggers.append("band_unreachable")
    if conf is None:
        triggers.append("missing_confidence")
    elif band_threshold is not None and conf < band_threshold:
        triggers.append("confidence_below_band")
    if unbound is None:
        triggers.append("missing_unbound")
    elif unbound is True:
        triggers.append("unbound_ood")
    if abstain is None:
        triggers.append("missing_abstain")
    elif abstain is True:
        triggers.append("conformal_abstain")
    if pset is None:
        triggers.append("missing_prediction_set")
    elif len(pset) == 0:
        triggers.append("empty_prediction_set")
    elif len(pset) > max_set:
        triggers.append("conformal_set_ambiguous")

    review = len(triggers) > 0
    return {
        "route": "review" if review else "auto",
        "review": review,
        "triggers": triggers,
    }


def route_laya_vega_default_exact(
    answer: Mapping[str, Any],
    *,
    band_threshold: Fraction | None,
    band_reachable: bool,
) -> dict[str, Any]:
    """Exact pure-Q model of laya/laya/vega.py:669-701 default routing (for gap comparison).

    Models the two documented upstream gaps:
      1. `unbound` is never checked in `_route`, so OOD singletons pass as 'auto'.
      2. When a reachable band is fitted, `abstain` and `|prediction_set| > 1` are skipped.
    """
    if not isinstance(answer, Mapping):
        raise LayaVegamlError("answer must be a mapping")
    require_exact_bool(band_reachable, "band_reachable")
    if band_threshold is not None:
        _require_unit_fraction(band_threshold, "band_threshold")
    conf = answer.get("confidence")
    if conf is not None:
        _require_unit_fraction(conf, "answer.confidence")
    abstain = answer.get("abstain")
    if abstain is not None:
        require_exact_bool(abstain, "answer.abstain")
    pset = answer.get("prediction_set")
    if pset is not None and (not isinstance(pset, Sequence) or isinstance(pset, (str, bytes))):
        raise LayaVegamlError("answer.prediction_set must be a sequence or None")

    if band_threshold is not None and band_reachable and conf is not None:
        review = conf < band_threshold
        return {"route": "review" if review else "auto", "review": review, "mode_used": "band_only"}
    review = bool(abstain) or (pset is not None and len(pset) > 1) or (pset is None and abstain is None)
    return {"route": "review" if review else "auto", "review": review, "mode_used": "conformal_fallback"}


# =============================================================================
# PILLAR 3: 5-State Epistemic Meet-Semilattice -> Aletheia v2.1 ClaimLedger
# =============================================================================

def epistemic_meet_exact(state_a: Any, state_b: Any) -> str:
    """Exact meet operator on the 5-state epistemic lattice (INCOMPARABLE < MISSING < UNKNOWN < INSUFFICIENT < PRESENT)."""
    if type(state_a) is not str or state_a not in EVIDENCE_RANK:
        raise LayaVegamlError(f"invalid epistemic state_a: {state_a!r}")
    if type(state_b) is not str or state_b not in EVIDENCE_RANK:
        raise LayaVegamlError(f"invalid epistemic state_b: {state_b!r}")
    return state_a if EVIDENCE_RANK[state_a] <= EVIDENCE_RANK[state_b] else state_b


def certify_epistemic_lattice_laws_exact() -> dict[str, Any]:
    """Exhaustively certify all semilattice axioms across all 5^3 = 125 state triples."""
    idem_ok = all(epistemic_meet_exact(a, a) == a for a in EVIDENCE_STATES)
    comm_ok = all(
        epistemic_meet_exact(a, b) == epistemic_meet_exact(b, a)
        for a in EVIDENCE_STATES
        for b in EVIDENCE_STATES
    )
    assoc_ok = all(
        epistemic_meet_exact(epistemic_meet_exact(a, b), c) == epistemic_meet_exact(a, epistemic_meet_exact(b, c))
        for a in EVIDENCE_STATES
        for b in EVIDENCE_STATES
        for c in EVIDENCE_STATES
    )
    top_id_ok = all(epistemic_meet_exact(a, EVIDENCE_PRESENT) == a for a in EVIDENCE_STATES)
    bot_abs_ok = all(epistemic_meet_exact(a, EVIDENCE_INCOMPARABLE) == EVIDENCE_INCOMPARABLE for a in EVIDENCE_STATES)
    return {
        "state_count": len(EVIDENCE_STATES),
        "triples_checked": len(EVIDENCE_STATES) ** 3,
        "idempotent": idem_ok,
        "commutative": comm_ok,
        "associative": assoc_ok,
        "top_present_identity": top_id_ok,
        "bottom_incomparable_absorbing": bot_abs_ok,
        "claim_ledger_map": dict(CLAIM_LEDGER_MAP),
        "all_laws_hold": idem_ok and comm_ok and assoc_ok and top_id_ok and bot_abs_ok,
    }


def inspect_calibration_fail_closed_exact(config: Any) -> dict[str, Any]:
    """Hardened pure-Q counterpart of laya/laya/evidence.py:inspect_calibration.

    Fixes two concrete upstream bugs:
      1. Negative `items` (e.g. `items=-5, issues=[]`) is rejected as UNKNOWN (not PRESENT).
      2. Empty calibration block returns MISSING and meet aggregation preserves UNKNOWN instead
         of collapsing UNKNOWN into INSUFFICIENT.
    """
    if not isinstance(config, Mapping):
        raise LayaVegamlError("config must be a mapping")
    training = config.get("training")
    if not isinstance(training, Mapping):
        return {"state": EVIDENCE_UNKNOWN, "types": {}, "claim_ledger_verdict": CLAIM_LEDGER_MAP[EVIDENCE_UNKNOWN]}
    calibration = training.get("laya_train_calibration")
    if not isinstance(calibration, Mapping):
        return {"state": EVIDENCE_UNKNOWN, "types": {}, "claim_ledger_verdict": CLAIM_LEDGER_MAP[EVIDENCE_UNKNOWN]}
    if len(calibration) == 0:
        return {"state": EVIDENCE_MISSING, "types": {}, "claim_ledger_verdict": CLAIM_LEDGER_MAP[EVIDENCE_MISSING]}

    types: dict[str, str] = {}
    overall = EVIDENCE_PRESENT
    for name, entry in calibration.items():
        if type(name) is not str or not name.strip():
            raise LayaVegamlError("calibration type key must be a non-empty str")
        if not isinstance(entry, Mapping):
            st = EVIDENCE_UNKNOWN
        else:
            items = entry.get("items")
            if type(items) is not int or items < 0:
                st = EVIDENCE_UNKNOWN
            elif items == 0:
                st = EVIDENCE_MISSING
            else:
                issues = entry.get("issues")
                if not isinstance(issues, list):
                    st = EVIDENCE_UNKNOWN
                elif len(issues) > 0:
                    st = EVIDENCE_INSUFFICIENT
                else:
                    st = EVIDENCE_PRESENT
        types[name] = st
        overall = epistemic_meet_exact(overall, st)
    return {
        "state": overall,
        "types": types,
        "claim_ledger_verdict": CLAIM_LEDGER_MAP[overall],
    }


def evaluate_evidence_promotion_exact(
    facets: Mapping[str, Any],
    *,
    checkpoint_sha256: Any = None,
) -> dict[str, Any]:
    """Fail-closed Aletheia v2.1 ClaimLedger promotion gate over evidence lattice facets."""
    if not isinstance(facets, Mapping) or len(facets) == 0:
        raise LayaVegamlError("facets must be a non-empty mapping of facet_name -> evidence state")
    if checkpoint_sha256 is not None:
        require_sha256(checkpoint_sha256, "checkpoint_sha256")
    meet_state = EVIDENCE_PRESENT
    for key, val in facets.items():
        if type(key) is not str or not key.strip():
            raise LayaVegamlError("facet name must be a non-empty str")
        meet_state = epistemic_meet_exact(meet_state, val)
    if checkpoint_sha256 is None:
        meet_state = epistemic_meet_exact(meet_state, EVIDENCE_UNKNOWN)
    return {
        "meet_state": meet_state,
        "claim_ledger_verdict": CLAIM_LEDGER_MAP[meet_state],
        "can_promote_to_verified": meet_state == EVIDENCE_PRESENT,
    }


# =============================================================================
# PILLAR 4: siliquant x vega-U Linear Port Recovery & laya#781 Guardrail Proof
# =============================================================================

def _require_fraction_matrix(mat: Any, name: str) -> list[list[Fraction]]:
    if not isinstance(mat, Sequence) or isinstance(mat, (str, bytes)) or len(mat) == 0:
        raise LayaVegamlError(f"{name} must be a non-empty 2D sequence of Fractions")
    out: list[list[Fraction]] = []
    ncols: int | None = None
    for r_idx, row in enumerate(mat):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) == 0:
            raise LayaVegamlError(f"{name}[{r_idx}] must be a non-empty sequence")
        if ncols is None:
            ncols = len(row)
        elif len(row) != ncols:
            raise LayaVegamlError(f"{name} must be rectangular (row {r_idx} width mismatch)")
        out.append([require_exact_fraction(v, f"{name}[{r_idx}][{c_idx}]") for c_idx, v in enumerate(row)])
    return out


def _mat_mul(a: Sequence[Sequence[Fraction]], b: Sequence[Sequence[Fraction]]) -> list[list[Fraction]]:
    rows_a, cols_a = len(a), len(a[0])
    cols_b = len(b[0])
    return [
        [sum(a[i][k] * b[k][j] for k in range(cols_a)) for j in range(cols_b)]
        for i in range(rows_a)
    ]


def _mat_transpose(a: Sequence[Sequence[Fraction]]) -> list[list[Fraction]]:
    rows, cols = len(a), len(a[0])
    return [[a[r][c] for r in range(rows)] for c in range(cols)]


def solve_linear_system_exact(
    a_mat: Sequence[Sequence[Fraction]],
    b_mat: Sequence[Sequence[Fraction]],
) -> list[list[Fraction]]:
    """Exact Gaussian elimination with partial pivoting over Q for A X = B (fail-closed on singular A)."""
    a = _require_fraction_matrix(a_mat, "a_mat")
    b = _require_fraction_matrix(b_mat, "b_mat")
    n = len(a)
    if len(a[0]) != n:
        raise LayaVegamlError("a_mat must be square")
    if len(b) != n:
        raise LayaVegamlError("b_mat row count must match a_mat")
    rhs_cols = len(b[0])

    aug = [list(a[i]) + list(b[i]) for i in range(n)]
    for col in range(n):
        piv = -1
        for r in range(col, n):
            if aug[r][col] != 0:
                piv = r
                break
        if piv == -1:
            raise LayaVegamlError("singular matrix in exact linear solve")
        if piv != col:
            aug[col], aug[piv] = aug[piv], aug[col]
        pval = aug[col][col]
        for j in range(col, n + rhs_cols):
            aug[col][j] /= pval
        for r in range(n):
            if r != col:
                factor = aug[r][col]
                for j in range(col, n + rhs_cols):
                    aug[r][j] -= factor * aug[col][j]
    return [row[n:] for row in aug]


def solve_ridge_port_exact(
    x_mat: Sequence[Sequence[Fraction]],
    y_mat: Sequence[Sequence[Fraction]],
    lam: Fraction = Fraction(0, 1),
) -> dict[str, Any]:
    """Exact pure-Q closed-form ridge linear port X W + 1 b^T -> Y (vegaml/_attach.py:92-119)."""
    x = _require_fraction_matrix(x_mat, "x_mat")
    y = _require_fraction_matrix(y_mat, "y_mat")
    n = len(x)
    if len(y) != n:
        raise LayaVegamlError("x_mat and y_mat must have identical row count")
    lam_f = require_exact_fraction(lam, "lam")
    if lam_f < 0:
        raise LayaVegamlError("lam must be non-negative")

    d_in = len(x[0])
    d_out = len(y[0])
    x_mean = [sum(x[i][j] for i in range(n)) / n for j in range(d_in)]
    y_mean = [sum(y[i][j] for i in range(n)) / n for j in range(d_out)]
    xc = [[x[i][j] - x_mean[j] for j in range(d_in)] for i in range(n)]
    yc = [[y[i][j] - y_mean[j] for j in range(d_out)] for i in range(n)]

    xc_t = _mat_transpose(xc)
    gram = _mat_mul(xc_t, xc)
    cross = _mat_mul(xc_t, yc)
    lhs = [
        [gram[i][j] + (n * lam_f if i == j else Fraction(0)) for j in range(d_in)]
        for i in range(d_in)
    ]
    w = solve_linear_system_exact(lhs, cross)
    xw_mean = [sum(x_mean[k] * w[k][j] for k in range(d_in)) for j in range(d_out)]
    b = [y_mean[j] - xw_mean[j] for j in range(d_out)]

    lhs_w = _mat_mul(lhs, w)
    kkt_sq = sum(
        (lhs_w[i][j] - cross[i][j]) * (lhs_w[i][j] - cross[i][j])
        for i in range(d_in)
        for j in range(d_out)
    )
    pred = [
        [sum(x[i][k] * w[k][j] for k in range(d_in)) + b[j] for j in range(d_out)]
        for i in range(n)
    ]
    res_sq = sum(
        (pred[i][j] - y[i][j]) * (pred[i][j] - y[i][j])
        for i in range(n)
        for j in range(d_out)
    )
    return {
        "W": w,
        "b": b,
        "kkt_residual_sq": kkt_sq,
        "residual_sum_sq": res_sq,
    }


def certify_quantized_reader_port_recovery_exact(
    x_clean: Sequence[Sequence[Fraction]],
    w0: Sequence[Sequence[Fraction]],
    b0: Sequence[Fraction],
    a_drift: Sequence[Sequence[Fraction]],
    c_shift: Sequence[Fraction],
    e_orth: Sequence[Sequence[Fraction]],
) -> dict[str, Any]:
    """Certify siliquant x vega-U quantized-reader linear port recovery & irreducible noise floor over Q."""
    x = _require_fraction_matrix(x_clean, "x_clean")
    w = _require_fraction_matrix(w0, "w0")
    a = _require_fraction_matrix(a_drift, "a_drift")
    e = _require_fraction_matrix(e_orth, "e_orth")
    n, d_in = len(x), len(x[0])
    d_out = len(w[0])
    if len(w) != d_in:
        raise LayaVegamlError("w0 row count must match x_clean width")
    if not isinstance(b0, Sequence) or isinstance(b0, (str, bytes)) or len(b0) != d_out:
        raise LayaVegamlError("b0 length must match w0 output width")
    b0_vec = [require_exact_fraction(v, f"b0[{j}]") for j, v in enumerate(b0)]
    if len(a) != d_in or len(a[0]) != d_in:
        raise LayaVegamlError("a_drift must be square matching x_clean width")
    if not isinstance(c_shift, Sequence) or isinstance(c_shift, (str, bytes)) or len(c_shift) != d_in:
        raise LayaVegamlError("c_shift length must match x_clean width")
    c_vec = [require_exact_fraction(v, f"c_shift[{j}]") for j, v in enumerate(c_shift)]
    if len(e) != n or len(e[0]) != d_in:
        raise LayaVegamlError("e_orth shape must match x_clean")

    x_mean = [sum(x[i][j] for i in range(n)) / n for j in range(d_in)]
    xc = [[x[i][j] - x_mean[j] for j in range(d_in)] for i in range(n)]
    e_mean_sq = sum(sum(e[i][j] for i in range(n)) ** 2 for j in range(d_in))
    if e_mean_sq != 0:
        raise LayaVegamlError("e_orth must be zero-mean (1^T E == 0)")
    xc_e = _mat_mul(_mat_transpose(xc), e)
    if any(xc_e[i][j] != 0 for i in range(d_in) for j in range(d_in)):
        raise LayaVegamlError("e_orth must be orthogonal to centered x_clean (X_c^T E == 0)")

    y = [
        [sum(x[i][k] * w[k][j] for k in range(d_in)) + b0_vec[j] for j in range(d_out)]
        for i in range(n)
    ]

    xa = _mat_mul(x, a)
    x_aff = [[xa[i][j] + c_vec[j] for j in range(d_in)] for i in range(n)]
    unrefit_aff_sq = sum(
        (sum(x_aff[i][k] * w[k][j] for k in range(d_in)) + b0_vec[j] - y[i][j]) ** 2
        for i in range(n)
        for j in range(d_out)
    )
    refit_aff = solve_ridge_port_exact(x_aff, y, lam=Fraction(0))
    a_inv_w0 = solve_linear_system_exact(a, w)
    exact_w_match = refit_aff["W"] == a_inv_w0
    expected_b = [b0_vec[j] - sum(c_vec[k] * a_inv_w0[k][j] for k in range(d_in)) for j in range(d_out)]
    exact_b_match = refit_aff["b"] == expected_b

    x_noisy = [[x_aff[i][j] + e[i][j] for j in range(d_in)] for i in range(n)]
    unrefit_noisy_sq = sum(
        (sum(x_noisy[i][k] * w[k][j] for k in range(d_in)) + b0_vec[j] - y[i][j]) ** 2
        for i in range(n)
        for j in range(d_out)
    )
    refit_noisy = solve_ridge_port_exact(x_noisy, y, lam=Fraction(0))

    gx = _mat_mul(_mat_transpose(xc), xc)
    ge = _mat_mul(_mat_transpose(e), e)
    at_gx_a = _mat_mul(_mat_transpose(a), _mat_mul(gx, a))
    denom_m = [[at_gx_a[i][j] + ge[i][j] for j in range(d_in)] for i in range(d_in)]
    at_gx_w0 = _mat_mul(_mat_transpose(a), _mat_mul(gx, w))
    wiener_w = solve_linear_system_exact(denom_m, at_gx_w0)
    sig_mat = _mat_mul(_mat_transpose(w), _mat_mul(gx, w))
    rec_mat = _mat_mul(_mat_transpose(at_gx_w0), wiener_w)
    wiener_floor_sq = sum(sig_mat[j][j] - rec_mat[j][j] for j in range(d_out))

    ridge_reg = solve_ridge_port_exact(x_noisy, y, lam=Fraction(1, 10))
    return {
        "unrefitted_affine_sq": frac_to_dict(unrefit_aff_sq),
        "refitted_affine_sq": frac_to_dict(refit_aff["residual_sum_sq"]),
        "affine_exact_absorption": (
            refit_aff["residual_sum_sq"] == 0
            and refit_aff["kkt_residual_sq"] == 0
            and exact_w_match
            and exact_b_match
        ),
        "unrefitted_noisy_sq": frac_to_dict(unrefit_noisy_sq),
        "refitted_noisy_sq": frac_to_dict(refit_noisy["residual_sum_sq"]),
        "wiener_floor_sq": frac_to_dict(wiener_floor_sq),
        "wiener_identity_exact": refit_noisy["residual_sum_sq"] == wiener_floor_sq,
        "noise_floor_strictly_positive": wiener_floor_sq > 0,
        "refit_beats_unrefitted": refit_noisy["residual_sum_sq"] < unrefit_noisy_sq,
        "ridge_lam_kkt_exact": ridge_reg["kkt_residual_sq"] == 0,
    }


def certify_guardrail_781_inversion_and_cap_exact(
    embedded_cases: Sequence[Mapping[str, Any]],
    *,
    benign_inversion_score: Fraction,
    default_threshold: Fraction = Fraction(1, 2),
) -> dict[str, Any]:
    """Certify laya/docs/guardrail-limitations.md (#781) 4-level Markov cap & lexical inversion theorem."""
    if not isinstance(embedded_cases, Sequence) or isinstance(embedded_cases, (str, bytes)) or len(embedded_cases) == 0:
        raise LayaVegamlError("embedded_cases must be a non-empty sequence")
    benign_pi = _require_unit_fraction(benign_inversion_score, "benign_inversion_score")
    thr = _require_unit_fraction(default_threshold, "default_threshold", strict_open=True)

    max_attack_pi = Fraction(0, 1)
    max_harm_cap = Fraction(0, 1)
    all_capped_below_threshold = True
    case_summaries: list[dict[str, Any]] = []
    for idx, case in enumerate(embedded_cases):
        if not isinstance(case, Mapping) or "id" not in case or "prompt_injection" not in case or "harm_severity" not in case:
            raise LayaVegamlError(f"embedded_cases[{idx}] missing required keys")
        cid = case["id"]
        if type(cid) is not str or not cid.strip():
            raise LayaVegamlError(f"embedded_cases[{idx}].id must be non-empty str")
        pi = _require_unit_fraction(case["prompt_injection"], f"{cid}.prompt_injection")
        hs = require_exact_fraction(case["harm_severity"], f"{cid}.harm_severity")
        if hs < 0 or hs > 3:
            raise LayaVegamlError(f"{cid}.harm_severity must lie in [0, 3]")
        cap = hs / 2
        if pi > max_attack_pi:
            max_attack_pi = pi
        if cap > max_harm_cap:
            max_harm_cap = cap
        if cap >= thr:
            all_capped_below_threshold = False
        case_summaries.append({
            "id": cid,
            "prompt_injection": frac_to_dict(pi),
            "harm_severity": frac_to_dict(hs),
            "markov_upper_tail_cap": frac_to_dict(cap),
            "cannot_reach_default_threshold": cap < thr and pi < thr,
        })

    p_wit = (Fraction(59, 100), Fraction(0, 1), Fraction(41, 100), Fraction(0, 1))
    s_wit = p_wit[1] + 2 * p_wit[2] + 3 * p_wit[3]
    markov_tight = (p_wit[2] + p_wit[3] == s_wit / 2) and (sum(p_wit) == 1)
    scalar_threshold_impossible = max_attack_pi < benign_pi

    return {
        "case_count": len(case_summaries),
        "cases": case_summaries,
        "max_embedded_attack_pi": frac_to_dict(max_attack_pi),
        "benign_inversion_pi": frac_to_dict(benign_pi),
        "max_harm_upper_tail_cap": frac_to_dict(max_harm_cap),
        "default_threshold": frac_to_dict(thr),
        "all_embedded_harm_capped_below_threshold": all_capped_below_threshold,
        "markov_tightness_verified": markov_tight,
        "scalar_threshold_impossible": scalar_threshold_impossible,
    }
