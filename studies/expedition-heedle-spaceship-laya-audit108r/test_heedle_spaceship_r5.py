"""R5 Assertion-Free Verification & Mutation-Killing Suite for Track S13 (AUDIT-108).

Runs under `python3 -I -O -B -W error` (zero bare assert statements, zero floats).
Independently verifies all 5 pillars, all 12 must-fail controls, exact pure-Z/Q
linear algebra, pruning vs heddles, symmetric INT4 quantization survival, heedle
protocol replay + tide/1 + T1 audit closure, spaceship#5 -> conformal -> Zone-B
snap routing, fail-closed certificate validation, and CLI modes.
"""
from __future__ import annotations

import copy
import io
import json
import os
import sys
import tempfile
from contextlib import redirect_stdout
from fractions import Fraction

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import heedle_spaceship_certifier as certifier  # noqa: E402
import heedle_spaceship_oracle as oracle  # noqa: E402


class _SubStr(str):
    pass


class _SubInt(int):
    pass


class _SubList(list):
    pass


class _SubDict(dict):
    pass


class _SubFrac(Fraction):
    pass


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(f"TEST_FAILURE: {msg}")


def _expect_err(fn, expected_code: str, msg: str) -> None:
    try:
        fn()
    except oracle.HeedleAuditError as exc:
        if exc.code != expected_code:
            raise RuntimeError(f"{msg}: expected {expected_code}, got {exc.code} ({exc.detail})") from exc
        return
    raise RuntimeError(f"{msg}: expected HeedleAuditError({expected_code}), but succeeded")


def _resign(cert: dict) -> dict:
    body = {k: v for k, v in cert.items() if k != "canonical_sha256"}
    cert["canonical_sha256"] = oracle.canonical_sha256(body)
    return cert


# ---------------------------------------------------------------------------
# Test 1: Pure-Z / Pure-Q Primitives & Helpers
# ---------------------------------------------------------------------------

def test_primitives_and_helpers() -> None:
    _check(oracle._is_nonempty_str("abc"), "nonempty str")
    _check(not oracle._is_nonempty_str("   "), "ws rejected")
    _check(not oracle._is_nonempty_str(_SubStr("abc")), "substr rejected")

    _check(oracle._in_str_set("a", ("a", "b")), "in str set")
    _check(not oracle._in_str_set(_SubStr("a"), ("a", "b")), "substr rejected")
    _check(not oracle._in_str_set("z", ("a", "b")), "missing rejected")
    _check(not oracle._in_str_set([], ("a", "b")), "unhashable rejected")

    _check(oracle._is_hex64(oracle.ROOT_W0), "valid hex64")
    _check(not oracle._is_hex64(_SubStr(oracle.ROOT_W0)), "substr hex64 rejected")
    _check(not oracle._is_hex64("xyz"), "bad hex64 rejected")
    _check(not oracle._is_hex64("0" * 64), "uniform zero hex64 rejected")

    _check(not oracle.contains_float(None), "None ok")
    _check(not oracle.contains_float(True), "bool ok")
    _check(not oracle.contains_float(7), "int ok")
    _check(not oracle.contains_float("s"), "str ok")
    _check(not oracle.contains_float({"k": [1, (2, None, False)]}), "nested ok")
    _check(oracle.contains_float(1.5), "float rejected")
    _check(oracle.contains_float(1j), "complex rejected")
    _check(oracle.contains_float({1}), "set rejected")
    _check(oracle.contains_float(frozenset({1})), "frozenset rejected")
    _check(oracle.contains_float(_SubInt(1)), "subint rejected")
    _check(oracle.contains_float({1.5: "v"}), "float key rejected")
    _check(oracle.contains_float({"k": 1.5}), "float val rejected")
    _check(oracle.contains_float([1, 2.0]), "float list rejected")
    _check(oracle.contains_float((1, 2.0)), "float tuple rejected")

    _expect_err(lambda: oracle.canonical_json_bytes({"x": 1.5}), "ERR_FLOAT_FORBIDDEN", "float json")
    _expect_err(lambda: oracle.canonical_json_bytes({1: "a", "b": 2}), "ERR_CANONICAL_JSON", "mixed key sort")
    _check(oracle.canonical_json_bytes({"b": 1, "a": 2}) == b'{"a":2,"b":1}', "canonical json")

    _check(oracle.frac_to_str(Fraction(-3, 5)) == "-3/5", "frac_to_str")
    _expect_err(lambda: oracle.frac_to_str(_SubFrac(1, 2)), "ERR_INVALID_FRACTION", "subfrac rejected")
    _expect_err(lambda: oracle.frac_to_str(1), "ERR_INVALID_FRACTION", "int frac_to_str rejected")

    _check(oracle.parse_exact_fraction(-2) == Fraction(-2, 1), "neg int")
    _check(oracle.parse_exact_fraction(Fraction(3, 4)) == Fraction(3, 4), "frac")
    _check(oracle.parse_exact_fraction("-3/4") == Fraction(-3, 4), "neg frac str")
    _check(oracle.parse_exact_fraction("0/1") == Fraction(0, 1), "zero frac str")
    _expect_err(lambda: oracle.parse_exact_fraction("1/" + "9" * 64), "ERR_INVALID_FRACTION_STR", "overlong str")
    _expect_err(lambda: oracle.parse_exact_fraction("-0/1"), "ERR_INVALID_FRACTION_STR", "neg zero str")
    _expect_err(lambda: oracle.parse_exact_fraction("01/2"), "ERR_INVALID_FRACTION_STR", "leading zero str")
    _expect_err(lambda: oracle.parse_exact_fraction("2/4"), "ERR_UNREDUCED_FRACTION", "unreduced str")
    _expect_err(lambda: oracle.parse_exact_fraction(1.5), "ERR_INVALID_FRACTION_TYPE", "float frac")
    _expect_err(lambda: oracle.parse_exact_fraction(True), "ERR_INVALID_FRACTION_TYPE", "bool frac")

    _expect_err(lambda: oracle._req_exact_keys(_SubDict({"a": 1}), {"a"}, "k"), "ERR_SCHEMA_KEYS", "subdict")
    _expect_err(lambda: oracle._req_exact_keys({"a": 1, "b": 2}, {"a"}, "k"), "ERR_SCHEMA_KEYS", "extra key")
    _expect_err(lambda: oracle._req_strict_str_eq(_SubStr("a"), "a", "s"), "ERR_SCHEMA_STR", "substr")
    _expect_err(lambda: oracle._req_strict_str_eq("b", "a", "s"), "ERR_SCHEMA_STR", "str diff")
    _expect_err(lambda: oracle._req_strict_int_range(True, 0, 5, "i"), "ERR_SCHEMA_INT", "bool int")
    _expect_err(lambda: oracle._req_strict_int_range(-1, 0, 5, "i"), "ERR_SCHEMA_INT", "below low")
    _expect_err(lambda: oracle._req_strict_int_range(6, 0, 5, "i"), "ERR_SCHEMA_INT", "above high")
    _check(oracle._req_strict_int_range(5, 0, None, "i") == 5, "high None ok")
    _expect_err(lambda: oracle._req_bool_eq(1, True, "b"), "ERR_SCHEMA_BOOL", "int bool")
    _expect_err(lambda: oracle._req_str_list(_SubList(["a"]), ["a"], "l"), "ERR_SCHEMA_LIST", "sublist")
    _expect_err(lambda: oracle._req_str_list([""], None, "l"), "ERR_SCHEMA_LIST", "empty elem")
    _expect_err(lambda: oracle._req_str_list(["a"], ["b"], "l"), "ERR_SCHEMA_LIST", "list diff")
    _check(oracle._req_str_list(["a"], None, "l") == ["a"], "expected None")
    _expect_err(lambda: oracle._req_hex64("0" * 64, "h"), "ERR_INVALID_SHA256", "uniform sha")

    _expect_err(lambda: oracle._require_frac_vec("bad", "v"), "ERR_INVALID_VECTOR", "vec not list")
    _expect_err(lambda: oracle._require_frac_vec([], "v"), "ERR_INVALID_VECTOR", "vec empty")
    _expect_err(lambda: oracle._require_frac_vec([1, 2], "v", 3), "ERR_VECTOR_LEN", "vec wrong len")
    _expect_err(lambda: oracle._require_frac_mat("bad", "m"), "ERR_INVALID_MATRIX", "mat not list")
    _expect_err(lambda: oracle._require_frac_mat([], "m"), "ERR_INVALID_MATRIX", "mat empty")
    _expect_err(lambda: oracle._require_frac_mat([[1, 2], [3]], "m"), "ERR_VECTOR_LEN", "ragged mat")


# ---------------------------------------------------------------------------
# Test 2: Pillar 1 — Gauss-Newton Shafts, Treadle Share & Pinned Shed
# ---------------------------------------------------------------------------

def test_pillar_1() -> None:
    # Linearly dependent pin rows test Gram-Schmidt rank drop
    s_dep = [[1, 1, 0], [2, 2, 0]]
    sh = oracle.shed_project_exact(s_dep, [3, 1, 5])
    _check(sh["pin_rank"] == 1 and sh["open_dimensions"] == 2 and sh["pin_leak_sq"] == "0/1", "dep pin rank 1")
    _check(sh["projected_move"] == ["1/1", "-1/1", "5/1"] and sh["idempotent"] and sh["pythagorean_exact"], "shed proj exact")

    _check(oracle.certify_heddle_preservation(s_dep, [1, -1, 5])["passed"] is True, "heddle clean")
    _check(oracle.certify_heddle_preservation(s_dep, [1, 1, 0])["error_code"] == "ERR_HEDDLE_PIN_LEAK", "heddle leak")

    j_mat = [[2, 0], [0, 1]]
    shafts = [[1, 0], [0, 1]]
    ev = oracle.evaluate_rational_shafts_and_treadle(j_mat, shafts, [1, 1], Fraction(1, 2), 1)
    _check(ev["carry_share_rho"] == "1/2" and ev["gain_identity_exact"] and ev["pythagorean_residual_exact"] and ev["predicted_ms_equals_t2_rho"], "2x2 exact")

    _expect_err(lambda: oracle.evaluate_rational_shafts_and_treadle(j_mat, "bad", [1, 1], 1, 1), "ERR_INVALID_SHAFTS", "shafts not list")
    _expect_err(lambda: oracle.evaluate_rational_shafts_and_treadle(j_mat, [], [1, 1], 1, 1), "ERR_INVALID_SHAFTS", "shafts empty")
    _expect_err(lambda: oracle.evaluate_rational_shafts_and_treadle(j_mat, shafts, [2, 0], 1, 1), "ERR_PATTERN_NOT_UNIT_RMS", "not unit rms")
    _expect_err(lambda: oracle.evaluate_rational_shafts_and_treadle(j_mat, [[0, 0]], [1, 1], 1, 1), "ERR_ZERO_SHAFT", "zero shaft")
    _expect_err(lambda: oracle.evaluate_rational_shafts_and_treadle(j_mat, [[1, 0], [1, 1]], [1, 1], 1, 1), "ERR_NON_ORTHOGONAL_SHAFTS", "non orth")
    _expect_err(lambda: oracle.evaluate_rational_shafts_and_treadle([[1, 0], [0, 0]], [[0, 1]], [1, 1], 1, 1), "ERR_NULL_SHAFT", "null shaft")
    _expect_err(lambda: oracle.evaluate_rational_shafts_and_treadle([[2, 0], [0, 1]], [[1, 1]], [1, 1], 1, 1), "ERR_NOT_EIGENVECTOR", "not eigvec")

    # certify_treadle_share_claim clause isolation
    _check(oracle.certify_treadle_share_claim(Fraction(1, 2), Fraction(2, 1), Fraction(2, 1))["passed"] is True, "share claim ok")
    _check(not oracle.certify_treadle_share_claim(Fraction(-1, 4), Fraction(1, 1), Fraction(5, 4))["passed"], "neg rho rejected")
    _check(not oracle.certify_treadle_share_claim(Fraction(5, 4), Fraction(1, 1), Fraction(-1, 4))["passed"], "rho > 1 rejected")
    _check(not oracle.certify_treadle_share_claim(Fraction(1, 2), Fraction(2, 1), Fraction(1, 1))["passed"], "wrong residual rejected")


# ---------------------------------------------------------------------------
# Test 3: Pillar 2 — Hidden-Move Nullspace Stealth & Zone-B Snap
# ---------------------------------------------------------------------------

def test_pillar_2() -> None:
    jk = [[1, 0, 0]]
    je = [[0, 1, 0], [0, 0, 2]]
    hm = oracle.construct_hidden_move_exact(jk, je, [7, 1, 1], iterations=2)
    _check(hm["known_ms"] == "0/1" and hm["worst_probe_index"] == 1 and hm["worst_probe_change_sq"] == "4/1", "hidden move exact")

    _expect_err(lambda: oracle.construct_hidden_move_exact([[1, 0]], [[1, 0, 0]], [1, 1], 1), "ERR_DIM_MISMATCH", "dim mismatch")
    _expect_err(lambda: oracle.construct_hidden_move_exact([[1, 0]], [[0, 1]], [5, 0], 1), "ERR_SEED_IN_KNOWN_ROWSPACE", "seed in rowspace")
    _expect_err(lambda: oracle.construct_hidden_move_exact([[1, 0]], [[1, 0]], [0, 5], 1), "ERR_ZERO_ELSEWHERE_ACTION", "zero elsewhere")
    _expect_err(lambda: oracle.verify_zone_b_snap_witness(je, [0, 0, 1], 1, 0), "ERR_NON_POSITIVE_TAU", "zero tau")
    # Boundary equality change_sq == tau_sq does NOT snap (strictly > tau)
    eq_w = oracle.verify_zone_b_snap_witness([[0, 1]], [0, 1], 0, 1)
    _check(eq_w["snapped"] is False and eq_w["error_code"] == "ERR_SNAP_WITNESS_DOES_NOT_REFUTE", "eq tau does not snap")


# ---------------------------------------------------------------------------
# Test 4: Pillar 3 — Pruning vs Heddles & Symmetric INT4 Survival
# ---------------------------------------------------------------------------

def test_pillar_3() -> None:
    u, v, h = [1, -1], [2, 3], [1, 1]
    col_ok = oracle.certify_rank1_heddle_pruning_exact(u, v, [[1, 0], [1, 0]], h)
    _check(col_ok["passed"] is True and col_ok["heddle_leak_sq"] == "0/1", "col mask preserves heddle")
    # Row mask preserves rank-1 (gram defect == 0) but STILL shatters heddle (heddle_leak_sq > 0)!
    row_bad = oracle.certify_rank1_heddle_pruning_exact(u, v, [[1, 1], [0, 0]], h)
    _check(not row_bad["passed"] and row_bad["is_exact_rank_le_1"] is True and row_bad["heddle_preserved"] is False, "row mask breaks heddle")
    # Independent control: rank > 1 (gram defect > 0) while heddle_leak_sq == 0!
    r2_safe = oracle.certify_rank1_heddle_pruning_exact(
        [1, -1, 1, -1], [1, 1], [[1, 1], [1, 1], [1, 0], [0, 1]], [1, 1, 0, 0]
    )
    _check(not r2_safe["passed"] and r2_safe["is_exact_rank_le_1"] is False and r2_safe["heddle_preserved"] is True, "rank>1 with zero heddle leak rejected")

    _expect_err(lambda: oracle.certify_rank1_heddle_pruning_exact([0, 0], v, [[1, 1], [1, 1]], h), "ERR_ZERO_VECTOR", "zero u")
    _expect_err(lambda: oracle.certify_rank1_heddle_pruning_exact(u, [0, 0], [[1, 1], [1, 1]], h), "ERR_ZERO_VECTOR", "zero v")
    _expect_err(lambda: oracle.certify_rank1_heddle_pruning_exact(u, v, [[1, 1], [1, 1]], [0, 0]), "ERR_ZERO_VECTOR", "zero h")
    _expect_err(lambda: oracle.certify_rank1_heddle_pruning_exact(u, v, ([1, 1], [1, 1]), h), "ERR_INVALID_MASK", "mask tuple not list")
    _expect_err(lambda: oracle.certify_rank1_heddle_pruning_exact(u, v, [[1, 1]], h), "ERR_INVALID_MASK", "mask row mismatch")
    _expect_err(lambda: oracle.certify_rank1_heddle_pruning_exact(u, v, [(1, 1), [1, 1]], h), "ERR_INVALID_MASK", "mask row tuple not list")
    _expect_err(lambda: oracle.certify_rank1_heddle_pruning_exact(u, v, [[1], [1, 1]], h), "ERR_INVALID_MASK", "mask col mismatch")
    _expect_err(lambda: oracle.certify_rank1_heddle_pruning_exact(u, v, [[True, 0], [1, 1]], h), "ERR_INVALID_MASK", "bool cell")
    _expect_err(lambda: oracle.certify_rank1_heddle_pruning_exact(u, v, [[2, 0], [1, 1]], h), "ERR_INVALID_MASK", "cell 2")
    _expect_err(lambda: oracle.certify_rank1_heddle_pruning_exact([1, 1], v, [[1, 1], [1, 1]], h), "ERR_BASE_NOT_IN_SHED", "u not orth h")

    # Symmetric RNE exact checks (zero max_abs, half-to-even tie on even vs odd, clamp > qmax, negative)
    _check(oracle.quantize_sym_rne_exact(5, 0, 7) == 0, "zero max_abs")
    _check(oracle.quantize_sym_rne_exact(2, 28, 7) == 0, "0.5 ULP tie -> even 0")
    _check(oracle.quantize_sym_rne_exact(6, 28, 7) == 2, "1.5 ULP tie -> even 2")
    _check(oracle.quantize_sym_rne_exact(-6, 28, 7) == -2, "-1.5 ULP tie -> even -2")
    _check(oracle.quantize_sym_rne_exact(100, 28, 7) == 7, "clamp to qmax")
    _expect_err(lambda: oracle.quantize_sym_rne_exact(True, 28, 7), "ERR_INVALID_INT", "bool val_int")

    # Multi-edit survival group (exercises ratio < min_edit_ulp_ratio update + no-edit group)
    multi = oracle.certify_int4_heedle_move_survival([28, 0, 0], [28, 8, 3], qmax=7)
    _check(multi["passed"] is True and multi["min_edit_ulp_ratio"] == "3/4" and multi["edited_flips"] == 2, "multi edit min ratio")
    noop = oracle.certify_int4_heedle_move_survival([28, 4], [28, 4], qmax=7)
    _check(noop["passed"] is True and noop["min_edit_ulp_ratio"] == "0/1", "noop group")

    _expect_err(lambda: oracle.certify_int4_heedle_move_survival((1,), [1]), "ERR_INVALID_GROUP", "base tuple not list")
    _expect_err(lambda: oracle.certify_int4_heedle_move_survival([1], (1,)), "ERR_INVALID_GROUP", "edit tuple not list")
    _expect_err(lambda: oracle.certify_int4_heedle_move_survival([], []), "ERR_INVALID_GROUP", "empty group")
    _expect_err(lambda: oracle.certify_int4_heedle_move_survival([1, 2], [1]), "ERR_INVALID_GROUP", "len mismatch")
    _expect_err(lambda: oracle.certify_int4_heedle_move_survival([False], [0]), "ERR_INVALID_INT", "bool base elem before zero check")
    _expect_err(lambda: oracle.certify_int4_heedle_move_survival([0], [False]), "ERR_INVALID_INT", "bool edit elem before zero check")
    _expect_err(lambda: oracle.certify_int4_heedle_move_survival([0, 0], [0, 0]), "ERR_ZERO_GROUP", "zero base group")


# ---------------------------------------------------------------------------
# Test 5: Pillar 4 — heedle protocol.md Replay, tide/1 & T1 Closure
# ---------------------------------------------------------------------------

def test_pillar_4() -> None:
    # T1 fixed zone-adjacent checks across all zones & tolerance edges
    _check(oracle.check_t1_gesture_fixed(0, "release")["passed"] is True, "tension 0 release ok")
    _check(oracle.check_t1_gesture_fixed(Fraction(1, 5), "release")["passed"] is True, "auto release ok")
    _check(oracle.check_t1_gesture_fixed(Fraction(3, 10), "release")["passed"] is True, "shared boundary release tol ok")
    _check(oracle.check_t1_gesture_fixed(Fraction(1, 2), "release")["passed"] is False, "shared interior release fails")
    _check(oracle.check_t1_gesture_fixed(Fraction(1, 2), "hold")["passed"] is True, "shared interior hold ok")
    _check(oracle.check_t1_gesture_fixed(Fraction(7, 10), "hold")["passed"] is True, "knot boundary hold tol ok")
    _check(oracle.check_t1_gesture_fixed(Fraction(7, 10), "release")["passed"] is False, "knot boundary release FAILS")
    _check(oracle.check_t1_gesture_fixed(Fraction(4, 5), "hold")["passed"] is False, "knot interior hold fails")
    _check(oracle.check_t1_gesture_fixed(Fraction(4, 5), "knot")["passed"] is True, "knot interior knot ok")
    _expect_err(lambda: oracle.check_t1_gesture_fixed(Fraction(-1, 10), "knot"), "ERR_NEGATIVE_TENSION", "neg tension")
    _expect_err(lambda: oracle.check_t1_gesture_fixed(Fraction(1, 2), "shrug"), "ERR_INVALID_GESTURE", "bad gesture")

    # Replay error branches
    base_pick = {
        "seq": 1, "kind": "pick", "parent_root": oracle.ROOT_W0, "root": oracle.ROOT_W1,
        "move_sha256": oracle.MOVE_SHA_1, "pins": [0], "gesture": "hold", "tau": "1/2",
        "remaining_budget": 23, "tension": "2/5", "witness_change": "3/10", "refuted": False,
    }
    d_bad_seq = oracle._build_signed_draft([dict(base_pick, seq=2)])
    _check(oracle.replay_heedle_draft_exact(d_bad_seq)["error_code"] == "ERR_PROTOCOL_CHAIN_OR_TAU", "bad seq")
    d_bool_seq = oracle._build_signed_draft([dict(base_pick, seq=True)])
    _check(oracle.replay_heedle_draft_exact(d_bool_seq)["error_code"] == "ERR_PROTOCOL_CHAIN_OR_TAU", "bool seq rejected")

    # HEEDLE108-BUG-01 hardening checks: host, r_invariant, pins, unpick gesture/bool budget,
    # pick bool budget, move_sha256, snap.root, negative/zero witness_change
    d_bad_hk = oracle._build_signed_draft([base_pick])
    d_bad_hk["host"]["kind"] = ""
    _expect_err(lambda: oracle.replay_heedle_draft_exact(d_bad_hk), "ERR_INVALID_HOST", "empty host.kind")
    d_bad_hn = oracle._build_signed_draft([base_pick])
    d_bad_hn["host"]["name"] = ""
    _expect_err(lambda: oracle.replay_heedle_draft_exact(d_bad_hn), "ERR_INVALID_HOST", "empty host.name")
    d_bad_rinv = oracle._build_signed_draft([base_pick])
    d_bad_rinv["gate"]["r_invariant"] = "0/1"
    _expect_err(lambda: oracle.replay_heedle_draft_exact(d_bad_rinv), "ERR_NON_POSITIVE_R_INV", "zero r_inv")

    _expect_err(lambda: oracle._validate_pins(_SubList([0])), "ERR_INVALID_PINS", "sublist pins")
    _expect_err(lambda: oracle._validate_pins([True]), "ERR_INVALID_PINS", "bool pin")
    _expect_err(lambda: oracle._validate_pins([-1]), "ERR_INVALID_PINS", "neg pin")
    _expect_err(lambda: oracle._validate_pins([2, 1]), "ERR_INVALID_PINS", "unsorted pins")
    _expect_err(lambda: oracle._validate_pins([1, 1]), "ERR_INVALID_PINS", "dup pins")

    d_bad_unp_g = oracle._build_signed_draft([
        base_pick,
        {"seq": 2, "kind": "unpick", "parent_root": oracle.ROOT_W1, "root": oracle.ROOT_W0, "pins": [], "gesture": "shrug", "tau": "1/2", "remaining_budget": 23},
    ])
    _expect_err(lambda: oracle.replay_heedle_draft_exact(d_bad_unp_g), "ERR_INVALID_GESTURE", "unpick bad gesture")

    # Bool remaining_budget (False == 0) rejected on both pick and unpick
    d_bool_pick_b = oracle._build_signed_draft([dict(base_pick, remaining_budget=False)])
    d_bool_pick_b["gate"]["budget"] = 1
    d_bool_pick_b["events"][0]["prev"] = oracle.heedle_genesis_id(d_bool_pick_b["host"], d_bool_pick_b["gate"])
    d_bool_pick_b["events"][0]["id"] = oracle.heedle_event_id(d_bool_pick_b["events"][0])
    _check(oracle.replay_heedle_draft_exact(d_bool_pick_b)["error_code"] == "ERR_BUDGET_MISMATCH", "pick bool budget")

    d_bool_unp_b = oracle._build_signed_draft([
        dict(base_pick, remaining_budget=0),
        {"seq": 2, "kind": "unpick", "parent_root": oracle.ROOT_W1, "root": oracle.ROOT_W0, "pins": [], "gesture": "release", "tau": "1/2", "remaining_budget": False},
    ])
    d_bool_unp_b["gate"]["budget"] = 1
    d_bool_unp_b["events"][0]["prev"] = oracle.heedle_genesis_id(d_bool_unp_b["host"], d_bool_unp_b["gate"])
    d_bool_unp_b["events"][0]["id"] = oracle.heedle_event_id(d_bool_unp_b["events"][0])
    d_bool_unp_b["events"][1]["prev"] = d_bool_unp_b["events"][0]["id"]
    d_bool_unp_b["events"][1]["id"] = oracle.heedle_event_id(d_bool_unp_b["events"][1])
    _check(oracle.replay_heedle_draft_exact(d_bool_unp_b)["error_code"] == "ERR_UNPICK_BUDGET_REFUND", "unpick bool budget")

    d_bad_msha = oracle._build_signed_draft([dict(base_pick, move_sha256="0" * 64)])
    _expect_err(lambda: oracle.replay_heedle_draft_exact(d_bad_msha), "ERR_INVALID_SHA256", "bad move_sha256")
    d_bad_sroot = oracle._build_signed_draft([dict(base_pick, kind="snap", root="0" * 64, witness_change="3/4", refuted=True)])
    _expect_err(lambda: oracle.replay_heedle_draft_exact(d_bad_sroot), "ERR_INVALID_SHA256", "bad snap root")
    d_neg_wc = oracle._build_signed_draft([dict(base_pick, witness_change="-1/1")])
    _expect_err(lambda: oracle.replay_heedle_draft_exact(d_neg_wc), "ERR_NEGATIVE_WITNESS_CHANGE", "neg witness_change")
    d_zero_wc = oracle._build_signed_draft([dict(base_pick, witness_change="0/1")])
    _check(oracle.replay_heedle_draft_exact(d_zero_wc)["passed"] is True, "zero witness_change allowed")

    d_bad_prev = oracle._build_signed_draft([base_pick])
    d_bad_prev["events"][0]["prev"] = oracle.ROOT_W1
    d_bad_prev["events"][0]["id"] = oracle.heedle_event_id(d_bad_prev["events"][0])
    _check(oracle.replay_heedle_draft_exact(d_bad_prev)["error_code"] == "ERR_PROTOCOL_CHAIN_OR_TAU", "bad prev")

    d_bad_id = oracle._build_signed_draft([base_pick])
    d_bad_id["events"][0]["id"] = oracle.ROOT_W1
    _check(oracle.replay_heedle_draft_exact(d_bad_id)["error_code"] == "ERR_EVENT_ID_MISMATCH", "bad event id")

    d_bad_par = oracle._build_signed_draft([dict(base_pick, parent_root=oracle.ROOT_W1)])
    _check(oracle.replay_heedle_draft_exact(d_bad_par)["error_code"] == "ERR_PARENT_ROOT_MISMATCH", "bad parent")

    d_bad_unp = oracle._build_signed_draft([
        base_pick,
        {"seq": 2, "kind": "unpick", "parent_root": oracle.ROOT_W1, "root": oracle.ROOT_W_SNAP, "pins": [], "gesture": "release", "tau": "1/2", "remaining_budget": 23},
    ])
    _check(oracle.replay_heedle_draft_exact(d_bad_unp)["error_code"] == "ERR_UNPICK_UNKNOWN_ROOT", "unpick unknown")

    d_bad_t1 = oracle._build_signed_draft([dict(base_pick, gesture="release", tension="7/10")])
    _check(oracle.replay_heedle_draft_exact(d_bad_t1)["error_code"] == "ERR_T1_GESTURE_TOO_LIGHT", "bad t1")

    d_bad_bud = oracle._build_signed_draft([dict(base_pick, remaining_budget=24)])
    _check(oracle.replay_heedle_draft_exact(d_bad_bud)["error_code"] == "ERR_BUDGET_MISMATCH", "bad budget")

    # Exhaust budget across 2 picks (budget=1, cost=1)
    d_exhaust = oracle._build_signed_draft([
        dict(base_pick, remaining_budget=0),
        dict(base_pick, seq=2, parent_root=oracle.ROOT_W1, root=oracle.ROOT_W_SNAP, remaining_budget=-1),
    ])
    d_exhaust["gate"]["budget"] = 1
    d_exhaust["events"][0]["prev"] = oracle.heedle_genesis_id(d_exhaust["host"], d_exhaust["gate"])
    d_exhaust["events"][0]["id"] = oracle.heedle_event_id(d_exhaust["events"][0])
    d_exhaust["events"][1]["prev"] = d_exhaust["events"][0]["id"]
    d_exhaust["events"][1]["id"] = oracle.heedle_event_id(d_exhaust["events"][1])
    _check(oracle.replay_heedle_draft_exact(d_exhaust)["error_code"] == "ERR_BUDGET_MISMATCH", "budget exhausted")

    d_bad_snap_wc = oracle._build_signed_draft([dict(base_pick, kind="snap", witness_change="1/2", refuted=True)])
    _check(oracle.replay_heedle_draft_exact(d_bad_snap_wc)["error_code"] == "ERR_S1_INVALID_SNAP", "snap wc <= tau")
    d_bad_snap_ref = oracle._build_signed_draft([dict(base_pick, kind="snap", witness_change="3/4", refuted=False)])
    _check(oracle.replay_heedle_draft_exact(d_bad_snap_ref)["error_code"] == "ERR_S1_INVALID_SNAP", "snap refuted False")

    d_bad_pick_wc = oracle._build_signed_draft([dict(base_pick, witness_change="3/4", refuted=False)])
    _check(oracle.replay_heedle_draft_exact(d_bad_pick_wc)["error_code"] == "ERR_K1_REFUTED_PICK", "pick wc > tau")
    d_bad_pick_ref = oracle._build_signed_draft([dict(base_pick, witness_change="1/4", refuted=True)])
    _check(oracle.replay_heedle_draft_exact(d_bad_pick_ref)["error_code"] == "ERR_K1_REFUTED_PICK", "pick refuted True")

    d_zero_tau = oracle._build_signed_draft([base_pick])
    d_zero_tau["gate"]["tau"] = "0/1"
    _expect_err(lambda: oracle.replay_heedle_draft_exact(d_zero_tau), "ERR_NON_POSITIVE_TAU", "zero tau")
    d_no_ev = oracle._build_signed_draft([base_pick])
    d_no_ev["events"] = []
    _expect_err(lambda: oracle.replay_heedle_draft_exact(d_no_ev), "ERR_INVALID_EVENTS", "empty events")
    d_not_list = oracle._build_signed_draft([base_pick])
    d_not_list["events"] = "bad"
    _expect_err(lambda: oracle.replay_heedle_draft_exact(d_not_list), "ERR_INVALID_EVENTS", "events not list")
    d_ev_str = oracle._build_signed_draft([base_pick])
    d_ev_str["events"] = ["bad"]
    _expect_err(lambda: oracle.replay_heedle_draft_exact(d_ev_str), "ERR_INVALID_EVENT", "event not dict")
    d_ev_kind = oracle._build_signed_draft([base_pick])
    d_ev_kind["events"] = [{"kind": "weave"}]
    _expect_err(lambda: oracle.replay_heedle_draft_exact(d_ev_kind), "ERR_INVALID_EVENT", "bad event kind")

    # tide/1 gamut violation & validation errors (keep contrast >= 1/2 so ONLY gamut triggers)
    bg = {"b": [Fraction(9, 10), Fraction(1, 20)]}
    bg_low = {"b": [Fraction(1, 10), Fraction(1, 20)]}
    fg = {"f": [Fraction(1, 10), Fraction(1, 20)]}
    _check(oracle.evaluate_tide1_contrast_floor_exact({"f": [Fraction(-1, 10), 0]}, bg, Fraction(1, 2), Fraction(1, 2))["gamut_ok"] is False, "L < 0 gamut")
    _check(oracle.evaluate_tide1_contrast_floor_exact({"f": [Fraction(11, 10), 0]}, bg_low, Fraction(1, 2), Fraction(1, 2))["snapped"] is True, "L > 1 gamut with high contrast")
    _check(oracle.evaluate_tide1_contrast_floor_exact({"f": [Fraction(1, 10), Fraction(-1, 10)]}, bg, Fraction(1, 2), Fraction(1, 2))["gamut_ok"] is False, "C < 0 gamut")
    _check(oracle.evaluate_tide1_contrast_floor_exact({"f": [Fraction(1, 10), Fraction(6, 10)]}, bg, Fraction(1, 2), Fraction(1, 2))["gamut_ok"] is False, "C > max gamut")
    _expect_err(lambda: oracle.evaluate_tide1_contrast_floor_exact("bad", bg, Fraction(1, 2), Fraction(1, 2)), "ERR_INVALID_TIDE_ROLES", "fg not dict")
    _expect_err(lambda: oracle.evaluate_tide1_contrast_floor_exact(fg, "bad", Fraction(1, 2), Fraction(1, 2)), "ERR_INVALID_TIDE_ROLES", "bg not dict")
    _expect_err(lambda: oracle.evaluate_tide1_contrast_floor_exact({}, bg, Fraction(1, 2), Fraction(1, 2)), "ERR_INVALID_TIDE_ROLES", "fg empty")
    _expect_err(lambda: oracle.evaluate_tide1_contrast_floor_exact(fg, {}, Fraction(1, 2), Fraction(1, 2)), "ERR_INVALID_TIDE_ROLES", "bg empty")
    _expect_err(lambda: oracle.evaluate_tide1_contrast_floor_exact(fg, bg, 0, Fraction(1, 2)), "ERR_INVALID_TIDE_BOUNDS", "min_c <= 0")
    _expect_err(lambda: oracle.evaluate_tide1_contrast_floor_exact(fg, bg, Fraction(1, 2), 0), "ERR_INVALID_TIDE_BOUNDS", "max_ch <= 0")
    # HEEDLE108-BUG-03: non-str and mixed role keys raise typed error instead of TypeError
    _expect_err(lambda: oracle.evaluate_tide1_contrast_floor_exact({1: [Fraction(1, 10), 0]}, bg, Fraction(1, 2), Fraction(1, 2)), "ERR_INVALID_TIDE_ROLE_NAME", "int fg key")
    _expect_err(lambda: oracle.evaluate_tide1_contrast_floor_exact({1: [Fraction(1, 10), 0], "a": [Fraction(1, 10), 0]}, bg, Fraction(1, 2), Fraction(1, 2)), "ERR_INVALID_TIDE_ROLE_NAME", "mixed fg keys")


# ---------------------------------------------------------------------------
# Test 6: Pillar 5 — Spaceship#5 Realization -> Conformal -> Zone-B Snap
# ---------------------------------------------------------------------------

def test_pillar_5() -> None:
    _expect_err(lambda: oracle.conformal_lac_qhat_exact([Fraction(-1, 10)], Fraction(1, 5)), "ERR_SCORE_RANGE", "score < 0")
    _expect_err(lambda: oracle.conformal_lac_qhat_exact([Fraction(11, 10)], Fraction(1, 5)), "ERR_SCORE_RANGE", "score > 1")
    _expect_err(lambda: oracle.conformal_lac_qhat_exact([Fraction(1, 2)], Fraction(0, 1)), "ERR_ALPHA_RANGE", "alpha <= 0")
    _expect_err(lambda: oracle.conformal_lac_qhat_exact([Fraction(1, 2)], Fraction(1, 1)), "ERR_ALPHA_RANGE", "alpha >= 1")
    # Small n where ceil((n+1)*(1-alpha)) > n clamps to n
    clamped = oracle.conformal_lac_qhat_exact([Fraction(1, 5), Fraction(2, 5)], Fraction(1, 10))
    _check(clamped["rank"] == 2 and clamped["qhat"] == "2/5", "conformal rank clamp")

    # Zone-B boundary & mismatch checks
    eq_zb = oracle.certify_spaceship_realization_escalation(
        same_vector_self_score=Fraction(4, 5), conformal_set_size=1, zone_b_independent_checked=True,
        zone_b_worst_change=Fraction(1, 2), tau=Fraction(1, 2), claimed_action="VERIFIED",
    )
    _check(eq_zb["passed"] is True, "zb worst == tau verified")
    bad_zb = oracle.certify_spaceship_realization_escalation(
        same_vector_self_score=Fraction(4, 5), conformal_set_size=1, zone_b_independent_checked=True,
        zone_b_worst_change=Fraction(3, 4), tau=Fraction(1, 2), claimed_action="VERIFIED",
    )
    _check(not bad_zb["passed"] and bad_zb["error_code"] == "ERR_ZONE_B_VERDICT_MISMATCH", "zb mismatch caught")
    # HEEDLE108-BUG-02: non-singleton conformal set (0 or >= 2) cannot VERIFIED even when zb_worst <= tau
    empty_zb = oracle.certify_spaceship_realization_escalation(
        same_vector_self_score=Fraction(4, 5), conformal_set_size=0, zone_b_independent_checked=True,
        zone_b_worst_change=Fraction(1, 10), tau=Fraction(1, 2), claimed_action="VERIFIED",
    )
    _check(not empty_zb["passed"] and empty_zb["required_action"] == "ESCALATE_ZONE_B", "empty conformal cannot VERIFIED")
    ambig_zb = oracle.certify_spaceship_realization_escalation(
        same_vector_self_score=Fraction(4, 5), conformal_set_size=2, zone_b_independent_checked=True,
        zone_b_worst_change=Fraction(1, 10), tau=Fraction(1, 2), claimed_action="ESCALATE_ZONE_B",
    )
    _check(ambig_zb["passed"] is True, "ambiguous conformal stays ESCALATE_ZONE_B")

    _expect_err(
        lambda: oracle.certify_spaceship_realization_escalation(
            same_vector_self_score=0, conformal_set_size=1, zone_b_independent_checked=1,
            zone_b_worst_change=0, tau=1, claimed_action="VERIFIED",
        ),
        "ERR_INVALID_BOOL",
        "int checked flag",
    )
    _expect_err(
        lambda: oracle.certify_spaceship_realization_escalation(
            same_vector_self_score=0, conformal_set_size=1, zone_b_independent_checked=True,
            zone_b_worst_change=-1, tau=1, claimed_action="VERIFIED",
        ),
        "ERR_INVALID_BOUND",
        "neg zb_worst",
    )
    _expect_err(
        lambda: oracle.certify_spaceship_realization_escalation(
            same_vector_self_score=0, conformal_set_size=1, zone_b_independent_checked=True,
            zone_b_worst_change=0, tau=0, claimed_action="VERIFIED",
        ),
        "ERR_INVALID_BOUND",
        "zero tau",
    )
    _expect_err(
        lambda: oracle.certify_spaceship_realization_escalation(
            same_vector_self_score=0, conformal_set_size=1, zone_b_independent_checked=True,
            zone_b_worst_change=0, tau=1, claimed_action="BOGUS",
        ),
        "ERR_INVALID_ACTION",
        "bad action",
    )


# ---------------------------------------------------------------------------
# Test 7: Certificate Validator & Dynamic Control Counter
# ---------------------------------------------------------------------------

def test_certificate_validator() -> None:
    cert = oracle.build_heedle_spaceship_certificate()
    _check(oracle.validate_certificate(cert) is True, "fresh cert validates")
    _check(cert["summary"]["pillars_passed"] == 5 and cert["summary"]["must_fail_controls_verified"] == 12, "5 pillars, 12 controls")

    # Isolate all 3 clauses of _count_verified_must_fail_controls
    cnt = oracle._count_verified_must_fail_controls({
        "p": {
            "other": {"rejected": True},
            "must_fail_not_dict": "bad",
            "must_fail_false": {"rejected": False},
            "must_fail_ok": {"rejected": True},
        }
    })
    _check(cnt == 1, "must_fail counter clause isolation")

    c_flt = copy.deepcopy(cert)
    c_flt["summary"]["floats_used"] = 0.0
    _expect_err(lambda: oracle.validate_certificate(c_flt), "ERR_CERT_FLOAT", "float rejected")

    c_sha = copy.deepcopy(cert)
    c_sha["canonical_sha256"] = oracle.ROOT_W0
    _expect_err(lambda: oracle.validate_certificate(c_sha), "ERR_CERT_SHA_MISMATCH", "sha mismatch")

    c_extra = copy.deepcopy(cert)
    c_extra["pillars"]["p3_scaling_bridge_pruning_and_int4_survival"]["extra"] = 1
    _expect_err(lambda: oracle.validate_certificate(_resign(c_extra)), "ERR_SCHEMA_KEYS", "extra nested key")

    c_sum = copy.deepcopy(cert)
    c_sum["summary"]["must_fail_controls_verified"] = 11
    _expect_err(lambda: oracle.validate_certificate(_resign(c_sum)), "ERR_SCHEMA_INT", "tampered control count")


# ---------------------------------------------------------------------------
# Test 8: CLI Certifier
# ---------------------------------------------------------------------------

def test_certifier_cli() -> None:
    cert = oracle.build_heedle_spaceship_certificate()
    pretty = certifier.format_pretty_json(cert)

    _expect_err(lambda: certifier._load_strict_json('{"a": 1, "a": 2}'), "ERR_CERT_JSON", "dup key")
    _expect_err(lambda: certifier._load_strict_json('{"a": 1.5}'), "ERR_CERT_FLOAT", "float literal")
    _expect_err(lambda: certifier._load_strict_json('{"a": NaN}'), "ERR_CERT_FLOAT", "NaN literal")
    _expect_err(lambda: certifier._load_strict_json("{bad"), "ERR_CERT_JSON", "bad json")
    _expect_err(lambda: certifier._load_strict_json(12345), "ERR_CERT_JSON", "non-str json")

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "cert.json")

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = certifier.main([])
        _check(rc == 0 and buf.getvalue() == pretty, "default prints pretty")

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = certifier.main(["--output", out_path])
        _check(rc == 0 and buf.getvalue() == "", "--output silent")

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = certifier.main(["--output", out_path, "--json"])
        _check(rc == 0 and buf.getvalue() == pretty, "--output --json prints pretty")

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = certifier.main(["--check", out_path])
        _check(
            rc == 0 and f"OK AUDIT-108 canonical_sha256={cert['canonical_sha256']} pillars=5 controls=12\n" == buf.getvalue(),
            "--check summary",
        )

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = certifier.main(["--check", out_path, "--json"])
        _check(rc == 0 and buf.getvalue() == pretty, "--check --json")

        drift_path = os.path.join(tmpdir, "drift.json")
        with open(drift_path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(cert, sort_keys=True) + "\n")
        _expect_err(lambda: certifier.check_certificate_file(drift_path), "ERR_CERT_DRIFT", "byte drift rejected")


def main() -> int:
    tests = [
        test_primitives_and_helpers,
        test_pillar_1,
        test_pillar_2,
        test_pillar_3,
        test_pillar_4,
        test_pillar_5,
        test_certificate_validator,
        test_certifier_cli,
    ]
    for fn in tests:
        fn()
    sys.stdout.write(f"PASS AUDIT-108 R5_TESTS={len(tests)}/{len(tests)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
