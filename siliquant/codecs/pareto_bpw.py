"""Pure-Q Pareto-Lagrangian per-tensor BPW allocator with critical-layer 16-bit pinning."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

from ._validation import require


@dataclass(frozen=True)
class QuantArmSpec:
    """One discrete quantization option for a tensor, expressed in exact rationals."""

    arm_id: str
    bpw_q: Fraction
    distortion_q: Fraction

    def __post_init__(self) -> None:
        require(bool(self.arm_id), "PARETO_ARM_ID", "arm_id must be non-empty")
        require(isinstance(self.bpw_q, Fraction) and self.bpw_q > 0, "PARETO_BPW_Q", "bpw_q must be positive Fraction")
        require(
            isinstance(self.distortion_q, Fraction) and self.distortion_q >= 0,
            "PARETO_DIST_Q",
            "distortion_q must be non-negative Fraction",
        )


@dataclass(frozen=True)
class TensorSensitivityProfile:
    """Sensitivity curve for one weight tensor across discrete quantization bitrates."""

    name: str
    weight_count: int
    arms: tuple[QuantArmSpec, ...]
    min_bpw_q: Fraction | None = None

    def __post_init__(self) -> None:
        require(bool(self.name), "PARETO_NAME", "tensor name must be non-empty")
        require(self.weight_count > 0, "PARETO_WCOUNT", "weight_count must be positive")
        require(len(self.arms) > 0, "PARETO_ARMS", f"tensor {self.name} has zero arms")
        if self.min_bpw_q is not None:
            require(isinstance(self.min_bpw_q, Fraction) and self.min_bpw_q > 0, "PARETO_MIN_BPW", "min_bpw_q must be > 0")


@dataclass(frozen=True)
class AllocationResult:
    """Exact pure-Q allocation output across all model tensors."""

    chosen_arms: dict[str, QuantArmSpec]
    total_weights: int
    total_bits_q: Fraction
    effective_bpw_q: Fraction
    total_distortion_q: Fraction
    lambda_star_q: Fraction
    frontier_sizes: dict[str, int]


def prune_pareto_frontier_q(
    arms: Sequence[QuantArmSpec],
    weight_count: int,
    *,
    min_bpw_q: Fraction | None = None,
) -> list[tuple[QuantArmSpec, Fraction, Fraction]]:
    """Return lower-convex-hull Pareto frontier of `(arm, bits_q, distortion_q)` in pure Q.

    Strictly filters out dominated arms (higher/equal bits with higher/equal distortion)
    and non-convex interior arms whose marginal slope fails to strictly decrease.
    """
    require(weight_count > 0, "PARETO_WCOUNT", "weight_count must be positive")
    eligible = [
        a for a in arms if (min_bpw_q is None or a.bpw_q >= min_bpw_q)
    ]
    require(
        len(eligible) > 0,
        "PARETO_PIN_EMPTY",
        f"no eligible arms satisfy min_bpw_q={min_bpw_q}",
    )

    by_bits: dict[Fraction, QuantArmSpec] = {}
    for arm in eligible:
        bits_q = arm.bpw_q * weight_count
        prev = by_bits.get(bits_q)
        if prev is None or arm.distortion_q < prev.distortion_q or (
            arm.distortion_q == prev.distortion_q and arm.arm_id < prev.arm_id
        ):
            by_bits[bits_q] = arm

    sorted_pts = sorted(
        [(arm, bits_q, arm.distortion_q) for bits_q, arm in by_bits.items()],
        key=lambda item: item[1],
    )

    # Enforce strict monotonicity (lower distortion for more bits)
    monotone: list[tuple[QuantArmSpec, Fraction, Fraction]] = []
    for arm, bits_q, dist_q in sorted_pts:
        if not monotone or dist_q < monotone[-1][2]:
            monotone.append((arm, bits_q, dist_q))

    # Lower convex hull: slopes (d_prev - d_cur) / (b_cur - b_prev) must strictly decrease
    hull: list[tuple[QuantArmSpec, Fraction, Fraction]] = []
    for pt in monotone:
        while len(hull) >= 2:
            _, b1, d1 = hull[-2]
            _, b2, d2 = hull[-1]
            _, b3, d3 = pt
            slope12 = (d1 - d2) / (b2 - b1)
            slope23 = (d2 - d3) / (b3 - b2)
            if slope23 >= slope12:
                hull.pop()
            else:
                break
        hull.append(pt)
    return hull


def _select_for_lambda_q(
    frontiers: Sequence[tuple[TensorSensitivityProfile, list[tuple[QuantArmSpec, Fraction, Fraction]]]],
    lam_q: Fraction,
) -> tuple[list[int], Fraction, Fraction]:
    """Select the optimal index on each tensor's Pareto hull minimizing D_i + lam * B_i."""
    idxs: list[int] = []
    tot_b = Fraction(0, 1)
    tot_d = Fraction(0, 1)
    for _, hull in frontiers:
        best_k = 0
        best_obj = hull[0][2] + lam_q * hull[0][1]
        for k in range(1, len(hull)):
            obj = hull[k][2] + lam_q * hull[k][1]
            if obj < best_obj:
                best_obj = obj
                best_k = k
        idxs.append(best_k)
        tot_b += hull[best_k][1]
        tot_d += hull[best_k][2]
    return idxs, tot_b, tot_d


def solve_pareto_lagrangian_bpw_q(
    profiles: Sequence[TensorSensitivityProfile],
    target_bpw_q: Fraction,
    *,
    pinned_min_bpw_patterns: Sequence[tuple[str, Fraction]] = (),
) -> AllocationResult:
    """Solve exact pure-Q Pareto-Lagrangian bit allocation subject to effective BPW <= target_bpw_q.

    Critical patterns in `pinned_min_bpw_patterns` (e.g. `("message_gate.down_proj", Fraction(16, 1))`)
    enforce fail-closed minimum bitrates before Pareto frontier construction.
    """
    require(len(profiles) > 0, "PARETO_EMPTY", "profiles must be non-empty")
    require(
        isinstance(target_bpw_q, Fraction) and target_bpw_q > 0,
        "PARETO_TARGET",
        "target_bpw_q must be a positive Fraction",
    )

    frontiers: list[tuple[TensorSensitivityProfile, list[tuple[QuantArmSpec, Fraction, Fraction]]]] = []
    total_weights = 0
    for prof in profiles:
        min_bpw = prof.min_bpw_q
        for substr, pin_bpw in pinned_min_bpw_patterns:
            if substr in prof.name:
                min_bpw = pin_bpw if min_bpw is None else max(min_bpw, pin_bpw)
        hull = prune_pareto_frontier_q(prof.arms, prof.weight_count, min_bpw_q=min_bpw)
        frontiers.append((prof, hull))
        total_weights += prof.weight_count

    budget_bits_q = target_bpw_q * total_weights
    min_bits_q = sum(hull[0][1] for _, hull in frontiers)
    require(
        min_bits_q <= budget_bits_q,
        "PARETO_INFEASIBLE",
        f"minimum pinned bits {min_bits_q} exceeds budget {budget_bits_q}",
    )

    # Collect all exact convex breakpoint slopes in strictly sorted order
    slopes_set: set[Fraction] = {Fraction(0, 1)}
    max_slope = Fraction(0, 1)
    for _, hull in frontiers:
        for k in range(len(hull) - 1):
            _, b1, d1 = hull[k]
            _, b2, d2 = hull[k + 1]
            sl = (d1 - d2) / (b2 - b1)
            slopes_set.add(sl)
            if sl > max_slope:
                max_slope = sl
    slopes_set.add(max_slope + 1)
    slopes = sorted(slopes_set)

    # Binary search over exact rational candidate lambdas for smallest lambda meeting budget_bits_q
    lo = 0
    hi = len(slopes) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        _, tb, _ = _select_for_lambda_q(frontiers, slopes[mid])
        if tb <= budget_bits_q:
            hi = mid
        else:
            lo = mid + 1
    lam_star = slopes[lo]
    idxs, tot_b, tot_d = _select_for_lambda_q(frontiers, lam_star)

    # Exact deterministic greedy upgrade on remaining bit slack (highest marginal slope first)
    while True:
        best_choice: tuple[Fraction, int, Fraction, Fraction] | None = None
        for i, (_, hull) in enumerate(frontiers):
            k = idxs[i]
            if k + 1 >= len(hull):
                continue
            db = hull[k + 1][1] - hull[k][1]
            if tot_b + db > budget_bits_q:
                continue
            dd = hull[k][2] - hull[k + 1][2]
            eff = dd / db
            if best_choice is None or eff > best_choice[0]:
                best_choice = (eff, i, db, dd)
        if best_choice is None:
            break
        _, i_best, db_best, dd_best = best_choice
        idxs[i_best] += 1
        tot_b += db_best
        tot_d -= dd_best

    chosen_arms: dict[str, QuantArmSpec] = {}
    frontier_sizes: dict[str, int] = {}
    for (prof, hull), k in zip(frontiers, idxs):
        chosen_arms[prof.name] = hull[k][0]
        frontier_sizes[prof.name] = len(hull)

    return AllocationResult(
        chosen_arms=chosen_arms,
        total_weights=total_weights,
        total_bits_q=tot_b,
        effective_bpw_q=tot_b / total_weights,
        total_distortion_q=tot_d,
        lambda_star_q=lam_star,
        frontier_sizes=frontier_sizes,
    )
