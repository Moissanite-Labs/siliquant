#!/usr/bin/env python3
"""Dual-mode pure-Q audit runner for Track S10 (AUDIT-107):
NandhaKishorM/laya & NandhaKishorM/vegaml -> Aletheia v2.1, siliquant & warp-drive bridge.

Modes:
  --standalone  (default) Self-contained exact pure-Q replay with zero external file I/O
  --live        Read-only live recount against staged intake-laya-vegaml/{laya,vegaml} + PR snapshot

Strict discipline: pure Python 3 stdlib, exact int/Fraction only, zero floats, zero bare asserts.
"""

from __future__ import annotations

import argparse
import ast
from fractions import Fraction
import hashlib
import hmac
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from laya_vegaml_oracle import (
    EVIDENCE_INCOMPARABLE,
    EVIDENCE_INSUFFICIENT,
    EVIDENCE_MISSING,
    EVIDENCE_PRESENT,
    EVIDENCE_UNKNOWN,
    LayaVegamlError,
    canonical_sha256,
    certify_empty_conformal_non_trigger_exact,
    certify_epistemic_lattice_laws_exact,
    certify_guardrail_781_inversion_and_cap_exact,
    certify_insample_coupling_counterexample_exact,
    certify_quantized_reader_port_recovery_exact,
    conformal_lac_qhat_exact,
    contains_float_recursive,
    evaluate_evidence_promotion_exact,
    format_exact_decimal,
    inspect_calibration_fail_closed_exact,
    parse_exact_decimal_to_fraction,
    require_exact_int,
    require_git_sha40,
    require_sha256,
    route_answer_fail_closed_exact,
    route_laya_vega_default_exact,
    wilson_lower_meets_target_exact,
)

SCHEMA_VERSION = "audit107.laya_vegaml.v1"

PINNED_INTAKE_FILES: tuple[str, ...] = (
    "laya/laya/evidence.py",
    "laya/laya/vega.py",
    "laya/laya/confidence.py",
    "laya/laya/router.py",
    "laya/docs/guardrail-limitations.md",
    "vegaml/vegaml/calibrate.py",
    "vegaml/vegaml/band.py",
    "vegaml/vegaml/router.py",
    "vegaml/vegaml/_attach.py",
    "vegaml/vegaml/_universal.py",
    "vegaml/vegaml/_common.py",
    "unmerged_prs_snapshot.json",
)

EXPECTED_UNMERGED_PRS: tuple[tuple[str, int], ...] = (
    ("NandhaKishorM/laya", 1524),
    ("NandhaKishorM/laya", 1519),
    ("NandhaKishorM/laya", 1517),
    ("NandhaKishorM/laya", 1502),
    ("NandhaKishorM/laya", 1390),
    ("NandhaKishorM/laya", 1388),
    ("NandhaKishorM/vegaml", 57),
    ("NandhaKishorM/vegaml", 40),
)

VALID_UNMERGED_PR_STATES = frozenset({"open", "closed", "OPEN", "CLOSED"})

PINNED_REPO_HEADS_EXPECTED: dict[str, str] = {
    "laya": "e150aa92a0d4059ac621875b39b620847cf1b88a",
    "vegaml": "5e7ffad546b006ccc316bb46c02df40e9200713d",
}

PINNED_ARTIFACTS_EXPECTED: dict[str, dict[str, Any]] = {
    "laya/docs/guardrail-limitations.md": {
        "bytes": 6791,
        "sha256": "c554f7ecf6a6fa42c8001bb2bb5f3e5678845f246ee8526ee93cfeb1928c3662",
    },
    "laya/laya/confidence.py": {
        "bytes": 17941,
        "sha256": "454ed669ecde141ccb02df2aea9e1a935cdad3ae0a660e5ab8ae38335111ccf9",
    },
    "laya/laya/evidence.py": {
        "bytes": 12794,
        "sha256": "4f2f7d85675ddbfc6c56752233ecd938a009d52959169afe3f9cfd01a36566ae",
    },
    "laya/laya/router.py": {
        "bytes": 117122,
        "sha256": "77ec4e015050d2b2d5965004704977ac36df17465bfdc814806a19dd2d02cb7e",
    },
    "laya/laya/vega.py": {
        "bytes": 66173,
        "sha256": "fd0a5093beca7af784a58ba8d43d35476dcb5276f36c60ad1ee0b5433e132f7d",
    },
    "unmerged_prs_snapshot.json": {
        "bytes": 13388,
        "sha256": "0aeb193d6fdf11d043fb7c3f89002af45826f63251f6081794857cfc2b9d26d2",
    },
    "vegaml/vegaml/_attach.py": {
        "bytes": 28642,
        "sha256": "83c603a6bd62a25d06d639ff6ee1c59defa71877a2944a1ed76a9e7c0f596793",
    },
    "vegaml/vegaml/_common.py": {
        "bytes": 60956,
        "sha256": "a2d48a5b292cc236dde726cceb5b347ca8a8b5eaee1847c5368df7339f58bf1a",
    },
    "vegaml/vegaml/_universal.py": {
        "bytes": 3620,
        "sha256": "1cc978c283acafcfd76e5ca8378ef05b50a766a793d3146de402b7223b65364f",
    },
    "vegaml/vegaml/band.py": {
        "bytes": 13150,
        "sha256": "506a61a936ce9cccf5002bafe804503a63e3fc0c3a47d560ab4156b0d15f55be",
    },
    "vegaml/vegaml/calibrate.py": {
        "bytes": 21576,
        "sha256": "3e92d496097b0f3008def26cbec98f6573d400d37821814f557dee87905ef5ca",
    },
    "vegaml/vegaml/router.py": {
        "bytes": 6369,
        "sha256": "7e880a78150cb25a69a9b6375a34f1b345c15b88ec41e178c3cfcf5a631dfb1d",
    },
}

PINNED_UNMERGED_PRS_EXPECTED: dict[str, Any] = {
    "pr_count": 8,
    "prs": [
        {
            "head_sha": "1d9aaae8b352e58a41baf6b7cb50344c6fb26cf5",
            "merged": False,
            "number": 1524,
            "repo": "NandhaKishorM/laya",
            "state": "open",
            "title": "fix(onnx): verify fused exports against the selected checkpoint",
        },
        {
            "head_sha": "216a723efc7ce974d21d0b5e488e7474ba2a1be3",
            "merged": False,
            "number": 1519,
            "repo": "NandhaKishorM/laya",
            "state": "open",
            "title": "fix(vega): translate engine's string-keyed choice labels back to caller's labels",
        },
        {
            "head_sha": "9291f01bfc3df77642bac1694ebe33f6e7d3d6ba",
            "merged": False,
            "number": 1517,
            "repo": "NandhaKishorM/laya",
            "state": "open",
            "title": "perf(train): single-pass evaluation overlap checks and signature reuse",
        },
        {
            "head_sha": "ac03d3fc007a8484003441f13062b05b853a1dce",
            "merged": False,
            "number": 1502,
            "repo": "NandhaKishorM/laya",
            "state": "closed",
            "title": "Add an opt-in Monid adapter for typed decisions",
        },
        {
            "head_sha": "f63252ba2e2fe64d610828a78566fea1400b41be",
            "merged": False,
            "number": 1390,
            "repo": "NandhaKishorM/laya",
            "state": "open",
            "title": "feat(rust): revision pinning and SHA-256 digest checks",
        },
        {
            "head_sha": "32097a4e5e97aabc999821c98990b946cdc526ea",
            "merged": False,
            "number": 1388,
            "repo": "NandhaKishorM/laya",
            "state": "open",
            "title": "feat(rust): port predict_tournament and cached_embed_fn",
        },
        {
            "head_sha": "a7d5ccba0e28dfe083d636de8c84a8feb8a6cabd",
            "merged": False,
            "number": 57,
            "repo": "NandhaKishorM/vegaml",
            "state": "open",
            "title": "perf(perception): replace dummy scatter index_add with direct span pooling",
        },
        {
            "head_sha": "aa1ad172d8b28745230e3054854dd9e07399deae",
            "merged": False,
            "number": 40,
            "repo": "NandhaKishorM/vegaml",
            "state": "open",
            "title": "Add an inference-only Rust port (rust/)",
        },
    ],
}

STANDALONE_CERT_TOP_KEYS = frozenset({
    "schema_version",
    "audit_id",
    "mode",
    "live_intake_audited",
    "novelty_class",
    "pillar1_split_conformal",
    "pillar2_escalation_band_and_router",
    "pillar3_epistemic_evidence_lattice",
    "pillar4_siliquant_port_and_guardrail781",
    "overall_verified",
    "certificate_sha256",
})

LIVE_CERT_TOP_KEYS = STANDALONE_CERT_TOP_KEYS | frozenset({
    "pinned_repo_heads",
    "pinned_artifacts",
    "live_source_ast_audit",
    "unmerged_pr_gems",
})

_CASE_ROW_RE = re.compile(r"^\|\s*(c011|c012|c042)\s*\|[^|]+\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|$")
_C038_SCORE_RE = re.compile(r"c038[^\n]*\n[^\n]*scores\s+([0-9.]+)")


def sha256_file(path: Path) -> tuple[str, int]:
    if not path.is_file():
        raise LayaVegamlError(f"missing pinned artifact: {path.name}")
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest(), len(data)


def read_git_head_readonly(repo_dir: Path) -> str:
    """Read HEAD commit SHA purely from .git files without invoking git or mutating .git."""
    head_path = repo_dir / ".git" / "HEAD"
    if not head_path.is_file():
        raise LayaVegamlError(f"missing .git/HEAD under {repo_dir.name}")
    text = head_path.read_text(encoding="utf-8").strip()
    if text.startswith("ref: "):
        ref_rel = text[5:].strip()
        ref_path = repo_dir / ".git" / ref_rel
        if ref_path.is_file():
            sha = ref_path.read_text(encoding="utf-8").strip()
            return require_git_sha40(sha, f"{repo_dir.name}.HEAD")
        packed = repo_dir / ".git" / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(encoding="utf-8").splitlines():
                if not line.startswith("#") and " " in line:
                    parts = line.split(" ", 1)
                    if parts[1].strip() == ref_rel:
                        return require_git_sha40(parts[0].strip(), f"{repo_dir.name}.HEAD")
        raise LayaVegamlError(f"unresolvable git ref {ref_rel} under {repo_dir.name}")
    return require_git_sha40(text, f"{repo_dir.name}.HEAD")


def run_pillar1_witnesses() -> dict[str, Any]:
    coupling = certify_insample_coupling_counterexample_exact()
    lac_sample = conformal_lac_qhat_exact(
        [Fraction(i, 20) for i in range(1, 20)],
        Fraction(1, 10),
    )
    ok = (
        coupling["theorem_holds"]
        and lac_sample["rank"] == 18
        and lac_sample["meets_nominal_target"]
    )
    return {
        "pillar_verified": ok,
        "exact_lac_n19_a010": lac_sample,
        "insample_coupling_theorem": coupling,
    }


def run_pillar2_witnesses() -> dict[str, Any]:
    # Empty-set non-trigger iff proof across k=2 (qhat=1/2 non-trigger vs 2/5 triggerable) & k=4
    non_trig_k2 = certify_empty_conformal_non_trigger_exact(2, Fraction(1, 2))
    trig_k2 = certify_empty_conformal_non_trigger_exact(2, Fraction(2, 5))
    non_trig_k4 = certify_empty_conformal_non_trigger_exact(4, Fraction(4, 5))

    # Exact sqrt-free Wilson lower bound matching vegaml/docs/escalation.md:64 (12/12 at 95% z=1.6449 -> 0.816)
    z95 = Fraction(16449, 10000)
    w_12_12_pass816 = wilson_lower_meets_target_exact(12, 12, z95, Fraction(816, 1000))
    w_12_12_fail817 = wilson_lower_meets_target_exact(12, 12, z95, Fraction(817, 1000))
    w_12_12_fail950 = wilson_lower_meets_target_exact(12, 12, z95, Fraction(95, 100))

    # Routing contrast: OOD unbound singleton & band-fitted ambiguous conformal set
    ood_answer = {"confidence": Fraction(9, 10), "unbound": True, "abstain": False, "prediction_set": ["a"]}
    ood_default = route_laya_vega_default_exact(ood_answer, band_threshold=Fraction(4, 5), band_reachable=True)
    ood_fail_closed = route_answer_fail_closed_exact(ood_answer, band_threshold=Fraction(4, 5), band_reachable=True)

    ambig_answer = {"confidence": Fraction(9, 10), "unbound": False, "abstain": False, "prediction_set": ["a", "b"]}
    ambig_default = route_laya_vega_default_exact(ambig_answer, band_threshold=Fraction(4, 5), band_reachable=True)
    ambig_fail_closed = route_answer_fail_closed_exact(ambig_answer, band_threshold=Fraction(4, 5), band_reachable=True)

    clean_answer = {"confidence": Fraction(9, 10), "unbound": False, "abstain": False, "prediction_set": ["a"]}
    clean_fail_closed = route_answer_fail_closed_exact(clean_answer, band_threshold=Fraction(4, 5), band_reachable=True)

    ok = (
        non_trig_k2["iff_verified"]
        and (not non_trig_k2["abstention_reachable"])
        and trig_k2["iff_verified"]
        and trig_k2["abstention_reachable"]
        and non_trig_k4["iff_verified"]
        and (not non_trig_k4["abstention_reachable"])
        and w_12_12_pass816["meets_target"]
        and (not w_12_12_fail817["meets_target"])
        and (not w_12_12_fail950["meets_target"])
        and ood_default["route"] == "auto"
        and ood_fail_closed["route"] == "review"
        and "unbound_ood" in ood_fail_closed["triggers"]
        and ambig_default["route"] == "auto"
        and ambig_fail_closed["route"] == "review"
        and "conformal_set_ambiguous" in ambig_fail_closed["triggers"]
        and clean_fail_closed["route"] == "auto"
    )
    return {
        "pillar_verified": ok,
        "empty_set_non_trigger_k2": non_trig_k2,
        "empty_set_triggerable_k2": trig_k2,
        "empty_set_non_trigger_k4": non_trig_k4,
        "wilson_12_of_12_at_816": w_12_12_pass816,
        "wilson_12_of_12_at_817": w_12_12_fail817,
        "wilson_12_of_12_at_950": w_12_12_fail950,
        "ood_unbound_gap_contrast": {"laya_vega_default": ood_default, "fail_closed_union": ood_fail_closed},
        "band_ambiguous_gap_contrast": {"laya_vega_default": ambig_default, "fail_closed_union": ambig_fail_closed},
        "clean_auto_witness": clean_fail_closed,
    }


def run_pillar3_witnesses() -> dict[str, Any]:
    laws = certify_epistemic_lattice_laws_exact()
    # Negative-items bug witness vs laya/laya/evidence.py:54-75 (items=-5, issues=[])
    neg_cfg = {"training": {"laya_train_calibration": {"choice": {"items": -5, "issues": []}}}}
    bool_cfg = {"training": {"laya_train_calibration": {"choice": {"items": True, "issues": []}}}}
    empty_cfg = {"training": {"laya_train_calibration": {}}}
    clean_cfg = {"training": {"laya_train_calibration": {"choice": {"items": 200, "issues": []}}}}

    neg_res = inspect_calibration_fail_closed_exact(neg_cfg)
    bool_res = inspect_calibration_fail_closed_exact(bool_cfg)
    empty_res = inspect_calibration_fail_closed_exact(empty_cfg)
    clean_res = inspect_calibration_fail_closed_exact(clean_cfg)

    valid_ckpt = hashlib.sha256(b"pinned-vega-u-engine-v1").hexdigest()
    promo_ok = evaluate_evidence_promotion_exact(
        {"calibration": EVIDENCE_PRESENT, "eval_report": EVIDENCE_PRESENT},
        checkpoint_sha256=valid_ckpt,
    )
    promo_no_ckpt = evaluate_evidence_promotion_exact(
        {"calibration": EVIDENCE_PRESENT, "eval_report": EVIDENCE_PRESENT},
        checkpoint_sha256=None,
    )
    promo_conflict = evaluate_evidence_promotion_exact(
        {"calibration": EVIDENCE_PRESENT, "eval_report": EVIDENCE_INCOMPARABLE},
        checkpoint_sha256=valid_ckpt,
    )

    ok = (
        laws["all_laws_hold"]
        and neg_res["state"] == EVIDENCE_UNKNOWN
        and bool_res["state"] == EVIDENCE_UNKNOWN
        and empty_res["state"] == EVIDENCE_MISSING
        and clean_res["state"] == EVIDENCE_PRESENT
        and promo_ok["can_promote_to_verified"]
        and (not promo_no_ckpt["can_promote_to_verified"])
        and promo_no_ckpt["meet_state"] == EVIDENCE_UNKNOWN
        and (not promo_conflict["can_promote_to_verified"])
        and promo_conflict["meet_state"] == EVIDENCE_INCOMPARABLE
    )
    return {
        "pillar_verified": ok,
        "lattice_axioms": laws,
        "negative_items_fail_closed": neg_res,
        "bool_items_fail_closed": bool_res,
        "empty_calibration_fail_closed": empty_res,
        "clean_calibration_present": clean_res,
        "promotion_bound_present": promo_ok,
        "promotion_unbound_blocked": promo_no_ckpt,
        "promotion_incomparable_blocked": promo_conflict,
    }


def run_pillar4_witnesses(
    embedded_cases: list[dict[str, Any]] | None = None,
    benign_c038_pi: Fraction | None = None,
) -> dict[str, Any]:
    # 8x2 clean reader features (centered, orthogonal columns) + 8x2 orthogonal zero-mean quantization noise E
    x_clean = [
        [Fraction(2), Fraction(2)],
        [Fraction(2), Fraction(-2)],
        [Fraction(-2), Fraction(2)],
        [Fraction(-2), Fraction(-2)],
        [Fraction(2), Fraction(2)],
        [Fraction(2), Fraction(-2)],
        [Fraction(-2), Fraction(2)],
        [Fraction(-2), Fraction(-2)],
    ]
    # Orthogonal zero-mean rounding noise: first 4 rows (+1, -1), last 4 rows (-1, +1) -> 1^T E = 0 and X_c^T E = 0!
    e_orth = [
        [Fraction(1, 2), Fraction(-1, 2)],
        [Fraction(1, 2), Fraction(-1, 2)],
        [Fraction(1, 2), Fraction(-1, 2)],
        [Fraction(1, 2), Fraction(-1, 2)],
        [Fraction(-1, 2), Fraction(1, 2)],
        [Fraction(-1, 2), Fraction(1, 2)],
        [Fraction(-1, 2), Fraction(1, 2)],
        [Fraction(-1, 2), Fraction(1, 2)],
    ]
    w0 = [[Fraction(3, 2), Fraction(-1, 2)], [Fraction(1, 4), Fraction(5, 4)]]
    b0 = [Fraction(1, 5), Fraction(-2, 5)]
    # Quantized-reader systematic affine distortion (per-channel scale drift + cross-channel shear + zero-point shift)
    a_drift = [[Fraction(3, 2), Fraction(1, 4)], [Fraction(0, 1), Fraction(4, 5)]]
    c_shift = [Fraction(7, 10), Fraction(-3, 5)]

    port_rec = certify_quantized_reader_port_recovery_exact(x_clean, w0, b0, a_drift, c_shift, e_orth)

    cases = embedded_cases if embedded_cases is not None else [
        {"id": "c011", "prompt_injection": Fraction(182, 1000), "harm_severity": Fraction(82, 100)},
        {"id": "c012", "prompt_injection": Fraction(301, 1000), "harm_severity": Fraction(78, 100)},
        {"id": "c042", "prompt_injection": Fraction(0, 1), "harm_severity": Fraction(51, 100)},
    ]
    benign_pi = benign_c038_pi if benign_c038_pi is not None else Fraction(619, 1000)
    guard781 = certify_guardrail_781_inversion_and_cap_exact(cases, benign_inversion_score=benign_pi)

    ok = (
        port_rec["affine_exact_absorption"]
        and port_rec["wiener_identity_exact"]
        and port_rec["noise_floor_strictly_positive"]
        and port_rec["refit_beats_unrefitted"]
        and port_rec["ridge_lam_kkt_exact"]
        and guard781["all_embedded_harm_capped_below_threshold"]
        and guard781["markov_tightness_verified"]
        and guard781["scalar_threshold_impossible"]
    )
    return {
        "pillar_verified": ok,
        "siliquant_vega_u_port_recovery": port_rec,
        "guardrail_781_inversion_and_cap": guard781,
    }


def _ast_audit_live_sources(intake_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], Fraction]:
    """Read-only AST & live-text verification over staged laya + vegaml files (zero external imports/.pyc)."""
    # 1. vegaml/calibrate.py: confirm _fit_temperature(rows) and _conformal_qhat(..., alpha) share same rows
    cal_tree = ast.parse((intake_dir / "vegaml" / "vegaml" / "calibrate.py").read_text(encoding="utf-8"))
    cal_fn = next((n for n in ast.walk(cal_tree) if isinstance(n, ast.FunctionDef) and n.name == "calibrate"), None)
    qhat_fn = next((n for n in ast.walk(cal_tree) if isinstance(n, ast.FunctionDef) and n.name == "_conformal_qhat"), None)
    if cal_fn is None or qhat_fn is None:
        raise LayaVegamlError("vegaml/calibrate.py missing calibrate or _conformal_qhat")
    cal_src = ast.unparse(cal_fn)
    qhat_src = ast.unparse(qhat_fn)
    calibrate_insample_coupled = ("_fit_temperature(rows)" in cal_src) and ("for p, y in rows" in cal_src) and ("_conformal_qhat(" in cal_src)
    qhat_uses_float_one_minus = "1.0 - alpha" in qhat_src

    # 2. vegaml/band.py: confirm Band.route reads only confidence and never unbound/abstain/prediction_set
    band_tree = ast.parse((intake_dir / "vegaml" / "vegaml" / "band.py").read_text(encoding="utf-8"))
    band_cls = next((n for n in ast.walk(band_tree) if isinstance(n, ast.ClassDef) and n.name == "Band"), None)
    if band_cls is None:
        raise LayaVegamlError("vegaml/band.py missing Band class")
    band_route_fn = next((n for n in band_cls.body if isinstance(n, ast.FunctionDef) and n.name == "route"), None)
    if band_route_fn is None:
        raise LayaVegamlError("vegaml/band.py missing Band.route")
    band_route_src = ast.unparse(band_route_fn)
    band_route_unbanded_not_review = ("'route': 'unbanded'" in band_route_src) and ("'review': False" in band_route_src)
    band_route_ignores_conformal_and_ood = not any(tok in band_route_src for tok in ("unbound", "abstain", "prediction_set"))

    # 3. laya/vega.py: confirm VegaAgent._route never checks unbound
    vega_tree = ast.parse((intake_dir / "laya" / "laya" / "vega.py").read_text(encoding="utf-8"))
    vega_cls = next((n for n in ast.walk(vega_tree) if isinstance(n, ast.ClassDef) and n.name == "VegaAgent"), None)
    if vega_cls is None:
        raise LayaVegamlError("laya/vega.py missing VegaAgent class")
    vega_route_fn = next((n for n in vega_cls.body if isinstance(n, ast.FunctionDef) and n.name == "_route"), None)
    if vega_route_fn is None:
        raise LayaVegamlError("laya/vega.py missing VegaAgent._route")
    vega_route_src = ast.unparse(vega_route_fn)
    laya_vega_route_ignores_unbound = ("unbound" not in vega_route_src) and ("abstain" in vega_route_src)

    # 4. laya/evidence.py: execute read-only in-memory (zero .pyc) to confirm live items=-5 -> PRESENT bug
    ev_src = (intake_dir / "laya" / "laya" / "evidence.py").read_text(encoding="utf-8")
    ev_ns: dict[str, Any] = {}
    exec(compile(ev_src, "laya/laya/evidence.py", "exec"), ev_ns)
    inspect_cal_live = ev_ns.get("inspect_calibration")
    if not callable(inspect_cal_live):
        raise LayaVegamlError("laya/evidence.py missing inspect_calibration")
    live_neg_verdict = inspect_cal_live({"training": {"laya_train_calibration": {"choice": {"items": -5, "issues": []}}}})
    live_bool_verdict = inspect_cal_live({"training": {"laya_train_calibration": {"choice": {"items": True, "issues": []}}}})
    evidence_negative_items_bug_confirmed = live_neg_verdict.get("state") == EVIDENCE_PRESENT
    bool_type_state = (live_bool_verdict.get("types") or {}).get("choice", {}).get("state")
    evidence_bool_guard_confirmed = bool_type_state == EVIDENCE_UNKNOWN
    evidence_unknown_collapsed_to_insufficient = live_bool_verdict.get("state") == EVIDENCE_INSUFFICIENT

    # 5. laya/docs/guardrail-limitations.md: parse live #781 table & c038 score as exact Fractions
    g781_text = (intake_dir / "laya" / "docs" / "guardrail-limitations.md").read_text(encoding="utf-8")
    parsed_cases: list[dict[str, Any]] = []
    for line in g781_text.splitlines():
        m = _CASE_ROW_RE.match(line.strip())
        if m:
            parsed_cases.append({
                "id": m.group(1),
                "jailbreak": parse_exact_decimal_to_fraction(m.group(2), f"{m.group(1)}.jailbreak"),
                "prompt_injection": parse_exact_decimal_to_fraction(m.group(3), f"{m.group(1)}.prompt_injection"),
                "harm_severity": parse_exact_decimal_to_fraction(m.group(4), f"{m.group(1)}.harm_severity"),
            })
    if [c["id"] for c in parsed_cases] != ["c011", "c012", "c042"]:
        raise LayaVegamlError("failed to parse expected c011/c012/c042 rows from guardrail-limitations.md")
    m_c038 = _C038_SCORE_RE.search(g781_text)
    if m_c038 is None:
        raise LayaVegamlError("failed to parse c038 inversion score from guardrail-limitations.md")
    benign_c038 = parse_exact_decimal_to_fraction(m_c038.group(1), "c038.prompt_injection")

    ast_summary = {
        "calibrate_insample_coupled": calibrate_insample_coupled,
        "qhat_uses_float_one_minus_alpha": qhat_uses_float_one_minus,
        "band_route_unbanded_not_review": band_route_unbanded_not_review,
        "band_route_ignores_conformal_and_ood": band_route_ignores_conformal_and_ood,
        "laya_vega_route_ignores_unbound": laya_vega_route_ignores_unbound,
        "evidence_negative_items_bug_confirmed": evidence_negative_items_bug_confirmed,
        "evidence_bool_guard_confirmed": evidence_bool_guard_confirmed,
        "evidence_unknown_collapsed_to_insufficient": evidence_unknown_collapsed_to_insufficient,
        "guardrail_781_parsed_case_ids": [c["id"] for c in parsed_cases],
        "guardrail_781_c038_pi_dec": format_exact_decimal(benign_c038, 3),
    }
    return ast_summary, parsed_cases, benign_c038


def _audit_unmerged_pr_snapshot(intake_dir: Path) -> dict[str, Any]:
    """Validate pinned read-only metadata for the 8 unmerged Law-1b portal PRs."""
    snap_path = intake_dir / "unmerged_prs_snapshot.json"
    if not snap_path.is_file():
        raise LayaVegamlError("missing unmerged_prs_snapshot.json")
    raw = json.loads(snap_path.read_text(encoding="utf-8"), parse_float=str)
    if not isinstance(raw, Mapping) or not isinstance(raw.get("prs"), list):
        raise LayaVegamlError("unmerged_prs_snapshot.json malformed")
    seen_pairs: list[tuple[str, int]] = []
    entries: list[dict[str, Any]] = []
    for idx, pr in enumerate(raw["prs"]):
        if not isinstance(pr, Mapping):
            raise LayaVegamlError(f"pr[{idx}] must be a mapping")
        repo = pr.get("repo")
        num = pr.get("number")
        title = pr.get("title")
        state = pr.get("state")
        merged = pr.get("merged")
        sha = pr.get("head_sha")
        if (
            type(repo) is not str
            or not repo.strip()
            or type(num) is not int
            or num <= 0
            or type(title) is not str
            or not title.strip()
            or type(state) is not str
            or state not in VALID_UNMERGED_PR_STATES
            or type(merged) is not bool
        ):
            raise LayaVegamlError(f"pr[{idx}] missing or malformed metadata")
        require_git_sha40(sha, f"pr[{idx}].head_sha")
        if merged:
            raise LayaVegamlError(f"expected unmerged PR, got merged for {repo}#{num}")
        seen_pairs.append((repo, num))
        entries.append({
            "repo": repo,
            "number": num,
            "state": state,
            "merged": merged,
            "head_sha": sha,
            "title": title,
        })
    if set(seen_pairs) != set(EXPECTED_UNMERGED_PRS) or len(seen_pairs) != len(EXPECTED_UNMERGED_PRS):
        raise LayaVegamlError("unmerged_prs_snapshot.json PR set mismatch")
    return {"pr_count": len(entries), "prs": entries}


def _type_strict_eq(a: Any, b: Any) -> bool:
    """Recursive exact-type and exact-value equality (rejects int 1 == bool True smuggling)."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_type_strict_eq(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_type_strict_eq(x, y) for x, y in zip(a, b))
    return a == b


def validate_audit_certificate(
    cert: Mapping[str, Any],
    *,
    intake_dir: Path | None = None,
) -> dict[str, Any]:
    """Fail-closed Zone-B-hardened certificate validator with exact recomputed pillars, strict key sets, and pinned provenance."""
    if not isinstance(cert, Mapping):
        raise LayaVegamlError("certificate must be a mapping")
    if contains_float_recursive(cert):
        raise LayaVegamlError("certificate contains forbidden float/non-JSON value")
    if "mode" not in cert or cert["mode"] not in ("live_intake_recount", "standalone_fixture_replay"):
        raise LayaVegamlError("certificate mode invalid")

    mode = cert["mode"]
    expected_keys = LIVE_CERT_TOP_KEYS if mode == "live_intake_recount" else STANDALONE_CERT_TOP_KEYS
    if set(cert.keys()) != expected_keys:
        raise LayaVegamlError("certificate top-level key set mismatch")
    if cert["schema_version"] != SCHEMA_VERSION:
        raise LayaVegamlError("certificate schema_version mismatch")
    if cert["audit_id"] != "AUDIT-107":
        raise LayaVegamlError("certificate audit_id mismatch")
    if cert["novelty_class"] != "N1":
        raise LayaVegamlError("certificate novelty_class mismatch")
    if type(cert["live_intake_audited"]) is not bool or cert["live_intake_audited"] != (mode == "live_intake_recount"):
        raise LayaVegamlError("certificate live_intake_audited mismatch")
    if type(cert["overall_verified"]) is not bool or not cert["overall_verified"]:
        raise LayaVegamlError("certificate overall_verified must be exact bool True")

    cert_sha = require_sha256(cert["certificate_sha256"], "certificate_sha256")
    body = {k: v for k, v in cert.items() if k != "certificate_sha256"}
    if not hmac.compare_digest(cert_sha, canonical_sha256(body)):
        raise LayaVegamlError("certificate_sha256 digest mismatch")

    # Exact type-strict recomputation of all 4 pure-Q pillars (zero hollow shells or 1==True smuggling!)
    if not _type_strict_eq(cert["pillar1_split_conformal"], run_pillar1_witnesses()):
        raise LayaVegamlError("pillar1_split_conformal payload mismatch or hollow")
    if not _type_strict_eq(cert["pillar2_escalation_band_and_router"], run_pillar2_witnesses()):
        raise LayaVegamlError("pillar2_escalation_band_and_router payload mismatch or hollow")
    if not _type_strict_eq(cert["pillar3_epistemic_evidence_lattice"], run_pillar3_witnesses()):
        raise LayaVegamlError("pillar3_epistemic_evidence_lattice payload mismatch or hollow")
    if not _type_strict_eq(cert["pillar4_siliquant_port_and_guardrail781"], run_pillar4_witnesses()):
        raise LayaVegamlError("pillar4_siliquant_port_and_guardrail781 payload mismatch or hollow")

    if mode == "live_intake_recount":
        heads = cert["pinned_repo_heads"]
        if not isinstance(heads, Mapping) or set(heads.keys()) != {"laya", "vegaml"}:
            raise LayaVegamlError("live certificate pinned_repo_heads incomplete")
        require_git_sha40(heads["laya"], "pinned_repo_heads.laya")
        require_git_sha40(heads["vegaml"], "pinned_repo_heads.vegaml")

        pa = cert["pinned_artifacts"]
        if not isinstance(pa, Mapping) or set(pa.keys()) != set(PINNED_INTAKE_FILES):
            raise LayaVegamlError("live certificate pinned_artifacts incomplete")
        for rel, meta in pa.items():
            if not isinstance(meta, Mapping) or set(meta.keys()) != {"sha256", "bytes"}:
                raise LayaVegamlError(f"invalid pinned_artifacts entry for {rel}")
            require_sha256(meta["sha256"], f"pinned_artifacts[{rel}].sha256")
            require_exact_int(meta["bytes"], f"pinned_artifacts[{rel}].bytes", allow_zero=False)

        ast_exp = {
            "calibrate_insample_coupled": True,
            "qhat_uses_float_one_minus_alpha": True,
            "band_route_unbanded_not_review": True,
            "band_route_ignores_conformal_and_ood": True,
            "laya_vega_route_ignores_unbound": True,
            "evidence_negative_items_bug_confirmed": True,
            "evidence_bool_guard_confirmed": True,
            "evidence_unknown_collapsed_to_insufficient": True,
            "guardrail_781_parsed_case_ids": ["c011", "c012", "c042"],
            "guardrail_781_c038_pi_dec": "0.619",
        }
        if not _type_strict_eq(cert["live_source_ast_audit"], ast_exp):
            raise LayaVegamlError("live_source_ast_audit payload mismatch or incomplete")

        prs = cert["unmerged_pr_gems"]
        if not isinstance(prs, Mapping) or set(prs.keys()) != {"pr_count", "prs"}:
            raise LayaVegamlError("unmerged_pr_gems key set mismatch")
        if require_exact_int(prs["pr_count"], "unmerged_pr_gems.pr_count") != len(EXPECTED_UNMERGED_PRS):
            raise LayaVegamlError("unmerged_pr_gems pr_count mismatch")
        pr_list = prs["prs"]
        if not isinstance(pr_list, list) or len(pr_list) != len(EXPECTED_UNMERGED_PRS):
            raise LayaVegamlError("unmerged_pr_gems list incomplete")
        req_pr_keys = {"repo", "number", "state", "merged", "head_sha", "title"}
        seen: list[tuple[str, int]] = []
        for idx, rec in enumerate(pr_list):
            if not isinstance(rec, Mapping) or set(rec.keys()) != req_pr_keys:
                raise LayaVegamlError(f"unmerged_pr_gems[{idx}] key set mismatch")
            if rec["merged"] is not False:
                raise LayaVegamlError(f"unmerged_pr_gems[{idx}] must have merged=False")
            require_git_sha40(rec["head_sha"], f"unmerged_pr_gems[{idx}].head_sha")
            if (
                type(rec["repo"]) is not str
                or not rec["repo"].strip()
                or type(rec["number"]) is not int
                or rec["number"] <= 0
                or type(rec["state"]) is not str
                or rec["state"] not in VALID_UNMERGED_PR_STATES
                or type(rec["title"]) is not str
                or not rec["title"].strip()
            ):
                raise LayaVegamlError(f"unmerged_pr_gems[{idx}] field type or value invalid")
            seen.append((rec["repo"], rec["number"]))
        if set(seen) != set(EXPECTED_UNMERGED_PRS):
            raise LayaVegamlError("unmerged_pr_gems PR set mismatch")

        if intake_dir is not None:
            expected_live = run_live_audit(intake_dir)
            if not _type_strict_eq(cert, expected_live):
                raise LayaVegamlError("live certificate provenance mismatch against intake_dir")
        else:
            if not _type_strict_eq(heads, PINNED_REPO_HEADS_EXPECTED):
                raise LayaVegamlError("live certificate pinned_repo_heads provenance mismatch")
            if not _type_strict_eq(pa, PINNED_ARTIFACTS_EXPECTED):
                raise LayaVegamlError("live certificate pinned_artifacts provenance mismatch")
            if not _type_strict_eq(prs, PINNED_UNMERGED_PRS_EXPECTED):
                raise LayaVegamlError("live certificate unmerged_pr_gems provenance mismatch")

    return {"valid": True, "mode": mode, "certificate_sha256": cert_sha}


def run_standalone_audit() -> dict[str, Any]:
    """Self-contained exact pure-Q replay with zero external file I/O."""
    p1 = run_pillar1_witnesses()
    p2 = run_pillar2_witnesses()
    p3 = run_pillar3_witnesses()
    p4 = run_pillar4_witnesses()
    overall = p1["pillar_verified"] and p2["pillar_verified"] and p3["pillar_verified"] and p4["pillar_verified"]
    body = {
        "schema_version": SCHEMA_VERSION,
        "audit_id": "AUDIT-107",
        "mode": "standalone_fixture_replay",
        "live_intake_audited": False,
        "novelty_class": "N1",
        "pillar1_split_conformal": p1,
        "pillar2_escalation_band_and_router": p2,
        "pillar3_epistemic_evidence_lattice": p3,
        "pillar4_siliquant_port_and_guardrail781": p4,
        "overall_verified": overall,
    }
    out = dict(body)
    out["certificate_sha256"] = canonical_sha256(body)
    return out


def run_live_audit(intake_dir: Path) -> dict[str, Any]:
    """Execute complete read-only live audit over staged intake-laya-vegaml/{laya,vegaml} + PR snapshot."""
    if not intake_dir.is_dir():
        raise LayaVegamlError("intake directory does not exist")
    heads = {
        "laya": read_git_head_readonly(intake_dir / "laya"),
        "vegaml": read_git_head_readonly(intake_dir / "vegaml"),
    }
    pinned: dict[str, dict[str, Any]] = {}
    for rel in PINNED_INTAKE_FILES:
        digest, sz = sha256_file(intake_dir / rel)
        pinned[rel] = {"sha256": digest, "bytes": sz}

    ast_summary, parsed_cases, benign_c038 = _ast_audit_live_sources(intake_dir)
    pr_gems = _audit_unmerged_pr_snapshot(intake_dir)
    p1 = run_pillar1_witnesses()
    p2 = run_pillar2_witnesses()
    p3 = run_pillar3_witnesses()
    p4 = run_pillar4_witnesses(embedded_cases=parsed_cases, benign_c038_pi=benign_c038)

    overall = (
        p1["pillar_verified"]
        and p2["pillar_verified"]
        and p3["pillar_verified"]
        and p4["pillar_verified"]
        and ast_summary["calibrate_insample_coupled"]
        and ast_summary["qhat_uses_float_one_minus_alpha"]
        and ast_summary["band_route_unbanded_not_review"]
        and ast_summary["band_route_ignores_conformal_and_ood"]
        and ast_summary["laya_vega_route_ignores_unbound"]
        and ast_summary["evidence_negative_items_bug_confirmed"]
        and ast_summary["evidence_bool_guard_confirmed"]
        and ast_summary["evidence_unknown_collapsed_to_insufficient"]
        and pr_gems["pr_count"] == len(EXPECTED_UNMERGED_PRS)
    )
    body = {
        "schema_version": SCHEMA_VERSION,
        "audit_id": "AUDIT-107",
        "mode": "live_intake_recount",
        "live_intake_audited": True,
        "novelty_class": "N1",
        "pinned_repo_heads": heads,
        "pinned_artifacts": pinned,
        "live_source_ast_audit": ast_summary,
        "unmerged_pr_gems": pr_gems,
        "pillar1_split_conformal": p1,
        "pillar2_escalation_band_and_router": p2,
        "pillar3_epistemic_evidence_lattice": p3,
        "pillar4_siliquant_port_and_guardrail781": p4,
        "overall_verified": overall,
    }
    out = dict(body)
    out["certificate_sha256"] = canonical_sha256(body)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AUDIT-107 pure-Q Laya & VegaML operator bridge runner.")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--standalone", action="store_true", help="Self-contained pure-Q fixture replay (default).")
    mode_group.add_argument("--live", action="store_true", help="Live read-only recount over staged intake-laya-vegaml.")
    parser.add_argument(
        "--intake-dir",
        type=Path,
        default=None,
        help="Path to staged intake-laya-vegaml directory (defaults to ../intake-laya-vegaml).",
    )
    parser.add_argument("--json", action="store_true", help="Emit canonical JSON certificate to stdout.")
    parser.add_argument("--write-cert", type=Path, default=None, help="Write canonical JSON certificate to path.")
    args = parser.parse_args(argv)

    if args.live:
        intake_dir = args.intake_dir if args.intake_dir is not None else (Path(__file__).resolve().parent.parent / "intake-laya-vegaml")
        result = run_live_audit(intake_dir)
    else:
        result = run_standalone_audit()

    rendered = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    if args.write_cert is not None:
        args.write_cert.write_text(rendered, encoding="utf-8")
    if args.json:
        sys.stdout.write(rendered)
    else:
        sys.stdout.write(
            f"AUDIT-107 mode={result['mode']} live={result['live_intake_audited']} "
            f"verified={result['overall_verified']} sha256={result['certificate_sha256'][:16]}...\n"
        )
    return 0 if result["overall_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
