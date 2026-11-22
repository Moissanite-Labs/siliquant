#!/usr/bin/env python3
"""R5 refusal-site mutation & exact-invariant test suite for AUDIT-059.

Targets 100.0% AST mutation kill rate (`aletheia v2.0.0 mutate`) on
`neuron_homology_exact.py` and exercises strict boundaries, negative controls,
and semantic certificate tamper rejection on `check_048_audit059_independent.py`.

Zero third-party deps, zero bare asserts.
Safe under: python3 -I -B, python3 -I -O -B, python3 -I -B -W error, PYTHONHASHSEED=12345.
"""

from __future__ import annotations

import copy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import check_048_audit059_independent as chk  # noqa: E402
import neuron_homology_exact as mod  # noqa: E402


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(f"TEST_FAILURE: {msg}")


def _expect_reject(fn, msg: str) -> None:
    try:
        fn()
    except mod.ExactHomologyError:
        return
    raise RuntimeError(f"MUTATION_ESCAPE — expected ExactHomologyError: {msg}")


def _expect_check_fail(fn, msg: str) -> None:
    try:
        fn()
    except chk.CheckFailure:
        return
    raise RuntimeError(f"SEMANTIC_BYPASS — expected CheckFailure: {msg}")


def _resign_cert(cert: dict) -> dict:
    body = {k: v for k, v in cert.items() if k != "canonical_sha256"}
    out = dict(body)
    out["canonical_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return out


def _make_goalkeeper_rows(per_cond: int = 4, target_hits: int = 4) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for cond in mod.GOALKEEPER_CONDITIONS:
        for i in range(per_cond):
            hit = (cond == "target" and i < target_hits)
            rows.append(
                {
                    "condition": cond,
                    "prompt": f"Write a short neutral note #{i}.",
                    "output": "The goalkeeper saved the shot." if hit else "A quiet park bench.",
                }
            )
    return rows


def run_all_tests() -> dict[str, int]:
    groups = 0

    # 1. Exact scalar & float-guard refusals
    _require(mod.require_exact_int(0, "a") == 0, "int 0")
    _require(mod.require_exact_fraction(3, "a") == Fraction(3), "frac int")
    _require(mod.require_exact_fraction(Fraction(1, 2), "a") == Fraction(1, 2), "frac Q")
    for bad in (1.0, True, False, "1", None):
        _expect_reject(lambda b=bad: mod.require_exact_int(b, "x"), f"int reject {bad!r}")
        _expect_reject(lambda b=bad: mod.require_exact_fraction(b, "x"), f"frac reject {bad!r}")
    mod.verify_no_floats({"a": [1, "3/2", (True, None)], "b": {"c": 0}})
    _expect_reject(lambda: mod.verify_no_floats({"a": [1, {"x": 0.25}]}), "nested float reject")
    _expect_reject(lambda: mod.verify_no_floats((1, 2.0)), "tuple float reject")
    groups += 1

    # 2. Exact hypergeometric tail values & every boundary refusal
    _require(mod.exact_hypergeom_tail(1, 0, 0, 0) == Fraction(1), "N=1 K=0 n=0 k=0")
    _require(mod.exact_hypergeom_tail(1, 1, 1, 1) == Fraction(1), "N=1 K=1 n=1 k=1")
    _require(
        mod.exact_hypergeom_tail(64, 15, 16, 15) == Fraction(1, 9_969_937_491_420),
        "Flint fixed exact Fisher p",
    )
    _require(
        mod.exact_hypergeom_tail(64, 16, 16, 16) == Fraction(1, 488_526_937_079_580),
        "Flint OOD exact Fisher p",
    )
    _require(mod.exact_hypergeom_tail(10, 8, 8, 6) == Fraction(1), "k_obs == k_min")
    _expect_reject(lambda: mod.exact_hypergeom_tail(0, 0, 0, 0), "N==0")
    _expect_reject(lambda: mod.exact_hypergeom_tail(-1, 0, 0, 0), "N<0")
    _expect_reject(lambda: mod.exact_hypergeom_tail(10, -1, 2, 0), "K<0")
    _expect_reject(lambda: mod.exact_hypergeom_tail(10, 11, 2, 0), "K>N")
    _expect_reject(lambda: mod.exact_hypergeom_tail(10, 5, -1, 0), "n<0")
    _expect_reject(lambda: mod.exact_hypergeom_tail(10, 5, 11, 0), "n>N")
    _expect_reject(lambda: mod.exact_hypergeom_tail(10, 8, 8, 5), "k_obs < k_min")
    _expect_reject(lambda: mod.exact_hypergeom_tail(10, 8, 8, 9), "k_obs > k_max")
    groups += 1

    # 3. Exact two-sided Binomial sign / McNemar tail & boundaries
    _require(mod.exact_binomial_two_sided_tail(1, 0) == Fraction(1), "n=1 k=0 clamped to 1")
    _require(mod.exact_binomial_two_sided_tail(1, 1) == Fraction(1), "n=1 k=1 clamped to 1")
    _require(mod.exact_binomial_two_sided_tail(2, 1) == Fraction(1), "n=2 k=1 >1 clamped to 1")
    _require(mod.exact_binomial_two_sided_tail(21, 21) == Fraction(1, 1_048_576), "21/21 sign test")
    _require(mod.exact_binomial_two_sided_tail(21, 0) == Fraction(1, 1_048_576), "0/21 symmetric")
    _require(mod.exact_binomial_two_sided_tail(24, 21) == Fraction(2325, 8_388_608), "21/24 all-layer sign test")
    _require(mod.exact_binomial_two_sided_tail(24, 3) == Fraction(2325, 8_388_608), "3/24 symmetric")
    _expect_reject(lambda: mod.exact_binomial_two_sided_tail(0, 0), "trials==0")
    _expect_reject(lambda: mod.exact_binomial_two_sided_tail(-3, 0), "trials<0")
    _expect_reject(lambda: mod.exact_binomial_two_sided_tail(5, -1), "k<0")
    _expect_reject(lambda: mod.exact_binomial_two_sided_tail(5, 6), "k>n")
    groups += 1

    # 4. Exact Z_2 sign-symmetry confounder & margin-centered equalizer
    pf = mod.exact_sign_symmetry_lineage(Fraction(17, 20), Fraction(1), Fraction(17, 20))
    _require(pf["chance_agreement"] == "17/20", "Pythia-Falcon chance 17/20")
    _require(pf["centered_excess"] == "0", "Pythia-Falcon delta=0")
    _require(pf["cohen_kappa"] == "0", "Pythia-Falcon kappa=0")
    _require(pf["indicates_shared_origin"] is False, "Pythia-Falcon unrelated")
    b0 = mod.exact_sign_symmetry_lineage(0, 0, Fraction(1, 2))
    _require(b0["cohen_kappa"] == "-1", "a_obs=0 boundary")
    b1 = mod.exact_sign_symmetry_lineage(1, 0, Fraction(1, 2))
    _require(b1["cohen_kappa"] == "1" and b1["indicates_shared_origin"] is True, "a_obs=1 boundary")
    q_lin = mod.exact_sign_symmetry_lineage(Fraction(93, 100), Fraction(1, 2), Fraction(1, 2))
    _require(q_lin["cohen_kappa"] == "43/50" and q_lin["indicates_shared_origin"] is True, "Q08/Q2 kappa")
    _expect_reject(lambda: mod.exact_sign_symmetry_lineage(Fraction(-1, 10), Fraction(1, 2), Fraction(1, 2)), "a_obs<0")
    _expect_reject(lambda: mod.exact_sign_symmetry_lineage(Fraction(11, 10), Fraction(1, 2), Fraction(1, 2)), "a_obs>1")
    _expect_reject(lambda: mod.exact_sign_symmetry_lineage(Fraction(1, 2), Fraction(-1, 10), Fraction(1, 2)), "p1<0")
    _expect_reject(lambda: mod.exact_sign_symmetry_lineage(Fraction(1, 2), Fraction(11, 10), Fraction(1, 2)), "p1>1")
    _expect_reject(lambda: mod.exact_sign_symmetry_lineage(Fraction(1, 2), Fraction(1, 2), Fraction(-1, 10)), "p2<0")
    _expect_reject(lambda: mod.exact_sign_symmetry_lineage(Fraction(1, 2), Fraction(1, 2), Fraction(11, 10)), "p2>1")
    _expect_reject(lambda: mod.exact_sign_symmetry_lineage(1, 1, 1), "degenerate A_chance==1 (1,1)")
    _expect_reject(lambda: mod.exact_sign_symmetry_lineage(1, 0, 0), "degenerate A_chance==1 (0,0)")
    groups += 1

    # 5. Hardened -O-immune goalkeeper recounter (FG-01 closure) & every tamper refusal
    good_rows = _make_goalkeeper_rows(per_cond=4, target_hits=4)
    rec = mod.strict_recount_goalkeeper_rows(good_rows, expected_per_condition=4)
    _require(rec["target_hits"] == 4 and rec["control_hits"] == 0, "recount 4/4 vs 0/12")
    _require(rec["exact_fisher_p"] == str(Fraction(1, 1820)), "exact Fisher 1/C(16,4)")
    one_rows = _make_goalkeeper_rows(per_cond=1, target_hits=1)
    _require(mod.strict_recount_goalkeeper_rows(one_rows, expected_per_condition=1)["target_hits"] == 1, "exp=1")
    _expect_reject(lambda: mod.strict_recount_goalkeeper_rows([], expected_per_condition=0), "exp==0 empty")
    _expect_reject(lambda: mod.strict_recount_goalkeeper_rows([], expected_per_condition=-1), "exp<0")

    class _IterWrapper:
        def __init__(self, items):
            self._items = items

        def __iter__(self):
            return iter(self._items)

    _expect_reject(lambda: mod.strict_recount_goalkeeper_rows(_IterWrapper(one_rows), expected_per_condition=1), "non-list iterable")  # type: ignore[arg-type]
    _expect_reject(lambda: mod.strict_recount_goalkeeper_rows(good_rows[:-1], expected_per_condition=4), "short rows")
    bad_nonmap = list(good_rows)
    bad_nonmap[0] = "bad"  # type: ignore[assignment]
    _expect_reject(lambda: mod.strict_recount_goalkeeper_rows(bad_nonmap, expected_per_condition=4), "non-mapping row")
    bad_cond = [dict(r) for r in good_rows]
    bad_cond[0]["condition"] = "extra_unknown"
    _expect_reject(lambda: mod.strict_recount_goalkeeper_rows(bad_cond, expected_per_condition=4), "unknown cond")
    bad_prompt_type = [dict(r) for r in good_rows]
    bad_prompt_type[0]["prompt"] = 123  # type: ignore[assignment]
    _expect_reject(lambda: mod.strict_recount_goalkeeper_rows(bad_prompt_type, expected_per_condition=4), "non-str prompt")
    bad_out_type = [dict(r) for r in good_rows]
    bad_out_type[0]["output"] = None  # type: ignore[assignment]
    _expect_reject(lambda: mod.strict_recount_goalkeeper_rows(bad_out_type, expected_per_condition=4), "non-str output")
    bad_leak = [dict(r) for r in good_rows]
    bad_leak[1]["prompt"] = "Tell me about a Goalkeeper in hockey."
    _expect_reject(lambda: mod.strict_recount_goalkeeper_rows(bad_leak, expected_per_condition=4), "prompt leak")
    bad_unbal = [dict(r) for r in good_rows]
    bad_unbal[0]["condition"] = "baseline"
    _expect_reject(lambda: mod.strict_recount_goalkeeper_rows(bad_unbal, expected_per_condition=4), "unbalanced conds")
    groups += 1

    # 6. Seeded permutation null control boundaries, exact feasibility [k_min, k_max], & determinism
    labels = [1] * 5 + [0] * 45
    p1 = mod.permutation_null_enrichment_exact(labels, 1, 5, 5, n_perms=50, seed=42)
    p2 = mod.permutation_null_enrichment_exact(labels, 1, 5, 5, n_perms=50, seed=42)
    _require(p1 == p2 and p1["exceed_count"] == 0 and p1["empirical_p_exact"] == "1/51", "deterministic null")
    # Feasible exact boundaries: k_min == 0, k_min > 0, k_max < sample_size, sample_size == pop
    p_zero = mod.permutation_null_enrichment_exact([0], 1, 1, 0, n_perms=1, seed=0)
    _require(p_zero["exceed_count"] == 1 and p_zero["empirical_p_exact"] == "1", "edge sample=pop k_min=0")
    p_kmin = mod.permutation_null_enrichment_exact([1, 1, 0], 1, 2, 1, n_perms=3, seed=0)
    _require(p_kmin["exceed_count"] == 3 and p_kmin["empirical_p_exact"] == "1", "edge obs==k_min==1")
    p_kmax = mod.permutation_null_enrichment_exact([1, 0, 0], 1, 2, 1, n_perms=3, seed=0)
    _require(p_kmax["target_count"] == 1 and p_kmax["observed_hits"] == 1, "edge obs==k_max==1 < sample")
    # Negative controls: impossible counts above k_max and below k_min (PR #24 review #3)
    _expect_reject(lambda: mod.permutation_null_enrichment_exact([], 1, 1, 0), "empty labels")
    _expect_reject(lambda: mod.permutation_null_enrichment_exact(labels, 1, 0, 0), "sample_size==0")
    _expect_reject(lambda: mod.permutation_null_enrichment_exact(labels, 1, 51, 0), "sample_size>pop")
    _expect_reject(lambda: mod.permutation_null_enrichment_exact(labels, 1, 5, -1), "obs<0")
    _expect_reject(lambda: mod.permutation_null_enrichment_exact(labels, 1, 5, 6), "obs>sample")
    _expect_reject(lambda: mod.permutation_null_enrichment_exact([0, 0, 0], 1, 2, 1, n_perms=3, seed=0), "obs > total_targets (0 targets)")
    _expect_reject(lambda: mod.permutation_null_enrichment_exact([1, 0, 0], 1, 2, 2, n_perms=3, seed=0), "obs > k_max (1 target, draw 2)")
    _expect_reject(lambda: mod.permutation_null_enrichment_exact([1, 1, 0], 1, 2, 0, n_perms=3, seed=0), "obs < k_min (2 targets in 3, draw 2)")
    _expect_reject(lambda: mod.permutation_null_enrichment_exact([1], 1, 1, 0, n_perms=1, seed=0), "obs < k_min (1 target in 1, draw 1)")
    _expect_reject(lambda: mod.permutation_null_enrichment_exact(labels, 1, 5, 1, n_perms=0), "n_perms==0")
    groups += 1

    # 7. Independent verifier semantic tamper + pinned-hash negative controls (PR #24 review #2)
    base_cert = json.loads(chk.DEFAULT_CERT.read_text(encoding="utf-8"))
    _require(chk.verify_standalone(chk.DEFAULT_CERT)["pinned_identity_verified"] is True, "default cert pinned pass")

    def _tamper(mutator) -> dict:
        d = copy.deepcopy(base_cert)
        mutator(d)
        return _resign_cert(d)

    tampered_cases = [
        ("empty_layer_joins", lambda d: d["part3_q08_weight_edit_empirical_join"].__setitem__("layer_joins", {})),
        ("corrupt_top40_count", lambda d: d["part3_q08_weight_edit_empirical_join"]["top40_linf_outliers_in_col_lt_512"].__setitem__("count", [10, 40])),
        ("corrupt_middle_layers", lambda d: d["part3_q08_weight_edit_empirical_join"]["q08_24layer_column_anisotropy"].__setitem__("middle_layers_L02_to_L22_hi_gt_lo", "10/21")),
        ("corrupt_residual_axis", lambda d: d["part3_q08_weight_edit_empirical_join"]["cross_layer_residual_axis_u15_u19"].__setitem__("top5_dims_L19", [1, 2, 3, 4, 5])),
        ("corrupt_mcnemar", lambda d: d["part3_q08_weight_edit_empirical_join"]["exp8_paired_mcnemar_1152_problems"].__setitem__("rescues", 5)),
        ("corrupt_upstream_head_sha", lambda d: d["part1_provenance_and_hygiene"]["alphaxiv_weight_edit_minimization"].__setitem__("head_sha", "0" * 40)),
        ("corrupt_page_digest", lambda d: d["part1_provenance_and_hygiene"]["qwen_neuron_map_sha256"].__setitem__("q08.html", "0" * 64)),
    ]
    with tempfile.TemporaryDirectory() as td:
        for label, fn in tampered_cases:
            tpath = Path(td) / f"{label}.json"
            tpath.write_text(json.dumps(_tamper(fn)), encoding="utf-8")
            # Must fail semantic verification even with expected_sha256=None AND with default pin!
            _expect_check_fail(lambda p=tpath: chk.verify_standalone(p, expected_sha256=None), f"unpinned semantic {label}")
            _expect_check_fail(lambda p=tpath: chk.verify_standalone(p), f"pinned {label}")
    groups += 1

    return {"groups_passed": groups}


if __name__ == "__main__":
    summary = run_all_tests()
    print(f"AUDIT_059_R5_EXACT_SUITE PASS ({summary['groups_passed']}/7 groups)")
    sys.exit(0)
