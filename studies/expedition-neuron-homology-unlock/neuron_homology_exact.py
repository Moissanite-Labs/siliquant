#!/usr/bin/env python3
"""Exact pure-Q/Z statistical, lineage-equalizer, and hardened-recount kernel for AUDIT-059.

Zero third-party dependencies, zero floats on trust paths, zero bare asserts.
Safe under: python3 -I -B, python3 -I -O -B, python3 -I -B -W error, PYTHONHASHSEED=12345.
"""

from __future__ import annotations

from fractions import Fraction
import math
import random
import re
from typing import Any, Mapping, Sequence

GOALKEEPER_RE = re.compile(r"\bgoalkeeper\b", re.IGNORECASE)
GOALKEEPER_CONDITIONS: tuple[str, ...] = (
    "target",
    "baseline",
    "opposite_pole",
    "matched_neuron",
)


class ExactHomologyError(ValueError):
    """Fail-closed domain, type, or invariant error."""


def require_exact_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ExactHomologyError(f"{label} must be an exact int, got {type(value).__name__}")
    return value


def require_exact_fraction(value: Any, label: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise ExactHomologyError(
            f"{label} must be an exact int or Fraction, got {type(value).__name__}"
        )
    return Fraction(value)


def verify_no_floats(obj: Any, path: str = "$") -> None:
    if isinstance(obj, float):
        raise ExactHomologyError(f"float value forbidden at {path}: {obj!r}")
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            verify_no_floats(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for idx, v in enumerate(obj):
            verify_no_floats(v, f"{path}[{idx}]")


def exact_hypergeom_tail(
    population: int,
    successes: int,
    draws: int,
    observed: int,
) -> Fraction:
    """Exact upper-tail hypergeometric probability P(X >= observed) over Q."""
    N = require_exact_int(population, "population")
    K = require_exact_int(successes, "successes")
    n = require_exact_int(draws, "draws")
    k_obs = require_exact_int(observed, "observed")
    if N <= 0:
        raise ExactHomologyError("population must be >= 1")
    k_min = max(0, n + K - N)
    k_max = min(K, n)
    if k_obs < k_min or k_obs > k_max:
        raise ExactHomologyError(f"observed={k_obs} outside feasible [{k_min}, {k_max}]")
    num = sum(math.comb(K, k) * math.comb(N - K, n - k) for k in range(k_obs, k_max + 1))
    den = math.comb(N, n)
    return Fraction(num, den)


def exact_binomial_two_sided_tail(trials: int, discordant_small: int) -> Fraction:
    """Exact two-sided symmetric Binomial(trials, 1/2) tail probability (for sign & McNemar tests)."""
    n = require_exact_int(trials, "trials")
    k = require_exact_int(discordant_small, "discordant_small")
    if n <= 0:
        raise ExactHomologyError("trials must be >= 1")
    if k < 0 or k > n:
        raise ExactHomologyError("discordant_small must lie in [0, trials]")
    m = min(k, n - k)
    tail_sum = sum(math.comb(n, j) for j in range(0, m + 1))
    two_sided = Fraction(2 * tail_sum, 1 << n)
    return Fraction(1) if two_sided > 1 else two_sided


def exact_sign_symmetry_lineage(
    observed_agreement: Fraction | int,
    p_pos_1: Fraction | int,
    p_pos_2: Fraction | int,
) -> dict[str, Any]:
    """Exact rational Z_2 sign-symmetry confounder (⦷) and margin-centered equalizer (⟠).

    For two independent models whose neuron sign marginal probabilities are p1, p2 in [0, 1],
    raw sign agreement under independence equals:
        A_chance = p1*p2 + (1 - p1)*(1 - p2) = 1/2 + (1/2)*(2*p1 - 1)*(2*p2 - 1).
    When activations are non-sign-symmetric (e.g. Pythia p1=1, Falcon-H1 p2=17/20),
    raw agreement inflates to 85% despite zero shared origin. Margin-centered excess
    delta_lin = A_obs - A_chance and Cohen's kappa = delta_lin / (1 - A_chance) vanish
    identically (0/1) for every unrelated pair regardless of activation skewness.
    """
    a_obs = require_exact_fraction(observed_agreement, "observed_agreement")
    p1 = require_exact_fraction(p_pos_1, "p_pos_1")
    p2 = require_exact_fraction(p_pos_2, "p_pos_2")
    if a_obs < 0 or a_obs > 1:
        raise ExactHomologyError("observed_agreement must lie in [0, 1]")
    if p1 < 0 or p1 > 1 or p2 < 0 or p2 > 1:
        raise ExactHomologyError("marginal probabilities p1, p2 must lie in [0, 1]")

    a_chance = p1 * p2 + (1 - p1) * (1 - p2)
    denom = 1 - a_chance
    if denom == 0:
        raise ExactHomologyError("degenerate chance agreement A_chance == 1 (zero variance)")
    delta_lin = a_obs - a_chance
    kappa = delta_lin / denom
    return {
        "observed_agreement": str(a_obs),
        "p_pos_1": str(p1),
        "p_pos_2": str(p2),
        "chance_agreement": str(a_chance),
        "centered_excess": str(delta_lin),
        "cohen_kappa": str(kappa),
        "indicates_shared_origin": delta_lin > 0,
    }


def strict_recount_goalkeeper_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_per_condition: int = 16,
) -> dict[str, Any]:
    """Hardened -O-immune exact-word recounter for alphaxiv/flint-goalkeeper-neuron (closes FG-01).

    Rejects:
      - wrong total row count (must equal 4 * expected_per_condition),
      - unknown or missing condition labels,
      - unbalanced per-condition counts,
      - any prompt leaking the scoring regex `\\bgoalkeeper\\b`,
      - non-string prompt/output fields.
    Returns exact counts and exact one-sided Fisher hypergeometric p-value over Q.
    """
    exp = require_exact_int(expected_per_condition, "expected_per_condition")
    if not isinstance(rows, (list, tuple)):
        raise ExactHomologyError("rows must be a list or tuple")
    expected_total = 4 * exp

    hits: dict[str, int] = {c: 0 for c in GOALKEEPER_CONDITIONS}
    totals: dict[str, int] = {c: 0 for c in GOALKEEPER_CONDITIONS}

    for idx, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ExactHomologyError(f"row {idx} is not a mapping")
        cond = row.get("condition")
        if cond not in totals:
            raise ExactHomologyError(f"row {idx} has unknown condition: {cond!r}")
        prompt = row.get("prompt")
        output = row.get("output")
        if not isinstance(prompt, str) or not isinstance(output, str):
            raise ExactHomologyError(f"row {idx} prompt/output must be str")
        if GOALKEEPER_RE.search(prompt) is not None:
            raise ExactHomologyError(f"row {idx} prompt leaks target word 'goalkeeper'")
        totals[cond] += 1
        if GOALKEEPER_RE.search(output) is not None:
            hits[cond] += 1

    for cond in GOALKEEPER_CONDITIONS:
        if totals[cond] != exp:
            raise ExactHomologyError(
                f"condition {cond!r} count {totals[cond]} != expected {exp}"
            )

    target_hits = hits["target"]
    control_hits = sum(hits[c] for c in GOALKEEPER_CONDITIONS if c != "target")
    total_hits = target_hits + control_hits
    fisher_p = exact_hypergeom_tail(expected_total, total_hits, exp, target_hits)
    return {
        "counts": {c: [hits[c], totals[c]] for c in GOALKEEPER_CONDITIONS},
        "target_hits": target_hits,
        "target_total": exp,
        "control_hits": control_hits,
        "control_total": 3 * exp,
        "prompt_leaks": 0,
        "exact_fisher_p": str(fisher_p),
    }


def permutation_null_enrichment_exact(
    labels: Sequence[int],
    target_label: int,
    sample_size: int,
    observed_hits: int,
    *,
    n_perms: int = 1000,
    seed: int = 20261121,
) -> dict[str, Any]:
    """Deterministic seeded permutation null control (isolated random.Random, zero floats)."""
    n_pop = len(labels)
    t_lbl = require_exact_int(target_label, "target_label")
    k_draw = require_exact_int(sample_size, "sample_size")
    obs = require_exact_int(observed_hits, "observed_hits")
    perms = require_exact_int(n_perms, "n_perms")
    sd = require_exact_int(seed, "seed")
    if k_draw <= 0 or k_draw > n_pop:
        raise ExactHomologyError("sample_size must lie in [1, len(labels)]")
    if obs < 0 or obs > k_draw:
        raise ExactHomologyError("observed_hits must lie in [0, sample_size]")
    if perms <= 0:
        raise ExactHomologyError("n_perms must be >= 1")

    pool = [1 if x == t_lbl else 0 for x in labels]
    total_targets = sum(pool)
    rng = random.Random(sd)
    exceed = 0
    max_null = 0
    sum_null = 0
    for _ in range(perms):
        drawn = sum(rng.sample(pool, k_draw))
        sum_null += drawn
        if drawn > max_null:
            max_null = drawn
        if drawn >= obs:
            exceed += 1
    p_emp = Fraction(exceed + 1, perms + 1)
    return {
        "population": n_pop,
        "target_count": total_targets,
        "sample_size": k_draw,
        "observed_hits": obs,
        "n_perms": perms,
        "seed": sd,
        "exceed_count": exceed,
        "max_null_hits": max_null,
        "mean_null_exact": str(Fraction(sum_null, perms)),
        "empirical_p_exact": str(p_emp),
    }
