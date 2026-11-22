"""Dual-mode (live_json_recount + standalone_fixture_replay) audit runner for
Track S9 (AUDIT-106R2): Rational Gated-MLP Z_2 Gauge Equivariance Witness (SiLU-like Gate +
RMS-like Norm + Muon + Learnable Multipliers), Symmetry-Breaking Controls, and
weft-loom#44 / AUDIT-059 Exact Rulings Verification.

Pure Python 3 stdlib. Zero floats. Zero bare assert statements.
"""

import argparse
import json
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from gated_mlp_lean_oracle import (
    GatedMlpLeanError,
    REQUIRED_MODELS,
    audit_weight_lean_and_audit059_exact,
    audit_z2_conservation_muon_lrm_exact,
    audit_z2_symmetry_breaking_controls_exact,
    canonical_json_bytes,
    certify_gated_mlp_lean_audit_exact,
    require_exact_int,
)


OPERATOR_LABELS = (
    "rational_silu_like_gated_mlp_z2_gauge_equivariance",
    "muon_newton_schulz_separate_and_fused_equivariance",
    "learnable_multiplier_and_rms_like_norm_forward_and_grad_equivariance",
    "layernorm_asymmetric_penalty_and_correlated_init_breaks",
    "weft_loom_pr44_weight_lean_and_audit059_exact_tails",
)

# Pinned from Moissanite-Labs/weft-loom#44 @ e660cd3de1ebd8fac33c60a3f65f5ccabb5ed9f1
# experiments/sign-lineage-2026-11-21/real-models/out/weights/weight_lean.json
# (sha256: 1123f698ee72ebb17d31f56f672b6641ad674382234a96226d9cf7a7b7ce00f9)
STANDALONE_WEIGHT_LEAN_FIXTURE = {
    "diag_source": "standalone_fixture",
    "models": {
        "tiiuae__Falcon-H1-Tiny-R-0.6B": {"gu_pos": 70500, "units": 90112, "up_pos": 45101, "dn_pos": 45419, "layers_gt_half": 44, "n_layers": 44},
        "tiiuae__Falcon-H1-Tiny-R-0.6B-pre-GRPO": {"gu_pos": 70500, "units": 90112, "up_pos": 45101, "dn_pos": 45419, "layers_gt_half": 44, "n_layers": 44},
        "tiiuae__Falcon-H1-Tiny-90M-Base": {"gu_pos": 12162, "units": 18432, "up_pos": 9111, "dn_pos": 9104, "layers_gt_half": 24, "n_layers": 24},
        "HuggingFaceTB__SmolLM2-135M": {"gu_pos": 22971, "units": 46080, "up_pos": 23103, "dn_pos": 22978, "layers_gt_half": 13, "n_layers": 30},
        "HuggingFaceTB__SmolLM2-360M": {"gu_pos": 41038, "units": 81920, "up_pos": 40995, "dn_pos": 40932, "layers_gt_half": 17, "n_layers": 32},
        "HuggingFaceTB__SmolLM2-135M-Instruct": {"gu_pos": 23060, "units": 46080, "up_pos": 23086, "dn_pos": 23006, "layers_gt_half": 14, "n_layers": 30},
        "Qwen__Qwen3.5-0.8B": {"gu_pos": 42974, "units": 86016, "up_pos": 42777, "dn_pos": 43114, "layers_gt_half": 13, "n_layers": 24},
        "Qwen__Qwen3.5-2B": {"gu_pos": 73680, "units": 147456, "up_pos": 73727, "dn_pos": 73452, "layers_gt_half": 14, "n_layers": 24},
    },
}


def load_live_weight_lean_summary(json_path):
    """Parse weft-loom#44 weight_lean.json strictly read-only with exact per-layer sum conservation."""
    if not os.path.isfile(json_path):
        raise GatedMlpLeanError("FILE_MISSING", f"weight_lean.json not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict):
        raise GatedMlpLeanError("INVALID_JSON", "weight_lean.json root must be dict")
    models_out = {}
    for mk in REQUIRED_MODELS:
        if mk not in raw or not isinstance(raw[mk], dict):
            raise GatedMlpLeanError("MISSING_MODEL", f"missing model {mk} in {json_path}")
        rec = raw[mk]
        per_layer = rec.get("per_layer")
        gu_d = rec.get("gate_up_inner_positive")
        up_d = rec.get("up_rowsum_positive")
        dn_d = rec.get("down_colsum_positive")
        if (
            not isinstance(per_layer, list)
            or not per_layer
            or not isinstance(gu_d, dict)
            or not isinstance(up_d, dict)
            or not isinstance(dn_d, dict)
        ):
            raise GatedMlpLeanError("INVALID_MODEL_REC", f"malformed sections for {mk}")
        tot_u = require_exact_int(gu_d.get("den"), f"{mk}.units", min_value=1)
        up_den = require_exact_int(up_d.get("den"), f"{mk}.up_den", min_value=1)
        dn_den = require_exact_int(dn_d.get("den"), f"{mk}.dn_den", min_value=1)
        if up_den != tot_u or dn_den != tot_u:
            raise GatedMlpLeanError("DEN_MISMATCH", f"up/down denominators contradict units in {mk}")
        tot_gu = require_exact_int(gu_d.get("num"), f"{mk}.gu_pos", min_value=0, max_value=tot_u)
        up_pos = require_exact_int(up_d.get("num"), f"{mk}.up_pos", min_value=0, max_value=tot_u)
        dn_pos = require_exact_int(dn_d.get("num"), f"{mk}.dn_pos", min_value=0, max_value=tot_u)
        sum_gu, sum_u, l_gt = 0, 0, 0
        for idx_l, row in enumerate(per_layer):
            if not isinstance(row, dict):
                raise GatedMlpLeanError("INVALID_MODEL_REC", f"non-dict per_layer[{idx_l}] in {mk}")
            lu = require_exact_int(row.get("units"), f"{mk}.L{idx_l}.units", min_value=1)
            lgu = require_exact_int(row.get("gate_up_inner_positive"), f"{mk}.L{idx_l}.gu", min_value=0, max_value=lu)
            sum_gu += lgu
            sum_u += lu
            if 2 * lgu > lu:
                l_gt += 1
        if sum_gu != tot_gu or sum_u != tot_u:
            raise GatedMlpLeanError("SUM_MISMATCH", f"per_layer sums contradict totals in {mk}")
        models_out[mk] = {
            "gu_pos": tot_gu,
            "units": tot_u,
            "up_pos": up_pos,
            "dn_pos": dn_pos,
            "layers_gt_half": l_gt,
            "n_layers": len(per_layer),
        }
    return {"diag_source": "live_json_file", "models": models_out}


def certify_from_summary(mode, live_json_recounted, summary):
    p1 = audit_z2_conservation_muon_lrm_exact()
    p2 = audit_z2_symmetry_breaking_controls_exact()
    p3 = audit_weight_lean_and_audit059_exact(summary)
    axes = {
        "z2_equivariance_surrogate_muon_lrm_rms_like": p1["passed"],
        "z2_symmetry_breaking_controls": p2["passed"],
        "weft_loom_weight_lean_and_audit059": p3["passed"],
    }
    return certify_gated_mlp_lean_audit_exact(
        mode=mode,
        live_json_recounted=live_json_recounted,
        operator_labels=OPERATOR_LABELS,
        weight_lean_summary_raw=summary,
        pillar1_result=p1,
        pillar2_result=p2,
        pillar3_result=p3,
        three_axis_checks=axes,
    )


def run_audit(live=False, weight_lean_json=None):
    if not live and weight_lean_json is None:
        return certify_from_summary("standalone_fixture_replay", False, STANDALONE_WEIGHT_LEAN_FIXTURE)
    if weight_lean_json is None:
        raise GatedMlpLeanError("PATH_REQUIRED", "--live requires --weight-lean-json <path>")
    summary = load_live_weight_lean_summary(weight_lean_json)
    return certify_from_summary("live_json_recount", True, summary)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Track S9 (AUDIT-106) Gated-MLP Z_2 Conservation Audit")
    parser.add_argument("--standalone", action="store_true", help="Offline fixture replay (default)")
    parser.add_argument("--live", action="store_true", help="Live read-only recount of weft-loom#44 weight_lean.json")
    parser.add_argument("--weight-lean-json", default=None, help="Path to weft-loom#44 weight_lean.json (implies live)")
    parser.add_argument("--json", action="store_true", help="Emit canonical JSON to stdout (default output format)")
    parser.add_argument("--output", default=None, help="Optional path to write canonical JSON certificate")
    args = parser.parse_args(argv)

    use_live = args.live or (args.weight_lean_json is not None)
    cert = run_audit(live=use_live, weight_lean_json=args.weight_lean_json)
    raw = canonical_json_bytes(cert)
    if args.output:
        with open(args.output, "wb") as fh:
            fh.write(raw)
            fh.write(b"\n")
    sys.stdout.write(raw.decode("utf-8") + "\n")
    return 0 if cert["all_gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
