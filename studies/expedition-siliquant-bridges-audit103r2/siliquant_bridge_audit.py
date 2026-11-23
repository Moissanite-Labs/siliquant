"""Dual-mode (live_repo_recount + standalone_fixture_replay) audit runner for
Track S6 (AUDIT-103): Moissanite-Labs/siliquant Force-Multiplier Audit & Exact
Operator Bridges.

Modes:
- Default / --standalone: recomputes ALL 4 pure-Q operator pillars live
  (branchless E8 vs brute-force oracle + Issue #16 centre penalty, TPU v5e
  Part-E roofline, homeostatic scaling + Kirchhoff R_eff vs persist/ MST,
  al-Biruni cancellation + al-Kashi contraction) while replaying the pinned
  live siliquant PR #12 @ 20d13d5 diagnostics (live_repo_recounted=False).
- --live / --siliquant-repo <path>: inspects local Moissanite-Labs/siliquant
  git repo strictly read-only in a temporary detached worktree, verifies git
  commit SHAs and PR #12 checks live (live_repo_recounted=True).

Pure Python 3 stdlib. Zero floats. Zero bare assert statements.
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from siliquant_bridge_oracle import (
    SiliquantBridgeError,
    audit_bayt_al_hikma_quant_bounds_exact,
    audit_branchless_e8_and_zero_codeword_exact,
    audit_nature_to_siliquant_bridges_exact,
    audit_siliquant_repo_diagnostics_exact,
    audit_tpu_v5e_roofline_exact,
    canonical_json_bytes,
    certify_siliquant_bridge_audit_exact,
)


OPERATOR_LABELS = (
    "branchless_integer_conway_sloane_e8_bitmask_decoder",
    "siliquant_issue16_zero_codeword_vs_centre_shift",
    "tpu_v5e_part_e_hbm_roofline_arithmetic",
    "homeostatic_multiplicative_scaling_and_integral_calibration",
    "kirchhoff_effective_resistance_vs_bipartite_mst_persistence",
    "al_biruni_dot_cancellation_and_al_kashi_scale_contraction",
)

STANDALONE_REPO_DIAG_FIXTURE = {
    "diag_source": "standalone_fixture",
    "draft_pr24_head": "d4c54e43bc9f24fd76d57a7467b0c726dd0acd1f",
    "draft_pr26_head": "7eb484d7d5274482f4d4a73d4012277438bf3c36",
    "draft_pr27_head": "8cf8d2447ab639b065c71c8edab898c66c0dfbeb",
    "main_commit": "67b29a4e18cd5d93e1f3701a34030f2e089e7312",
    "pr12_claim_ledger_exact_match": True,
    "pr12_design_out_has_smollm2_fix": True,
    "pr12_head_commit": "20d13d5fa9c5fb2333b8faa0360a90caa1e0a43f",
    "pr12_lattice_tests_executed_in_live_check": False,
    "pr12_number_trace_exact_match": True,
    "pr12_perm_tests_passed": True,
    "pr12_persist_tests_passed": True,
    "pr12_plan_before_data_exit_zero": True,
    "pr12_shape_tests_passed": True,
}


def _git_rev_parse(repo_dir, ref):
    proc = subprocess.run(
        ["git", "-C", repo_dir, "rev-parse", "--verify", ref],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise SiliquantBridgeError("GIT_REF_MISSING", f"cannot resolve {ref} in {repo_dir}")
    return proc.stdout.strip()


def _run_cmd_ok(cmd, cwd):
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
    return proc.returncode == 0, proc.stdout


_DEC_RE = re.compile(r"\d+\.\d+")


def _design_out_has_smollm2_fix(design_txt):
    """Validate exact 11-column shape/design.out SmolLM2 rows with uniqueness & exact KL tokens."""
    lines = design_txt.splitlines()
    required = {
        "EBG128:b560s16": "0.0609",
        "ZBG128:b560s16": "0.0560",
        "EBG128:b504s8": "0.1183",
    }
    seen = {label: 0 for label in required}
    valid = {label: 0 for label in required}
    for ln in lines:
        parts = [p.strip() for p in ln.split("|")]
        lbl = parts[0] if parts else ""
        if lbl not in required:
            continue
        seen[lbl] += 1
        if len(parts) != 11 or not _DEC_RE.fullmatch(parts[1]):
            return False
        cell_tokens = [tok.strip() for tok in parts[4].split(",")]
        if len(cell_tokens) == 2 and _DEC_RE.fullmatch(cell_tokens[0]) and cell_tokens[1] == required[lbl]:
            valid[lbl] += 1
    return all(seen[lbl] == 1 and valid[lbl] == 1 for lbl in required)


def _verify_pr12_worktree(wt_dir):
    shape_ok, _ = _run_cmd_ok([sys.executable, "-I", "-B", "-m", "unittest", "discover", "-s", "shape", "-p", "test_*.py", "-q"], wt_dir)
    perm_ok, _ = _run_cmd_ok([sys.executable, "-I", "-B", "-m", "unittest", "discover", "-s", "perm", "-p", "test_*.py", "-q"], wt_dir)
    persist_ok, _ = _run_cmd_ok([sys.executable, "-I", "-B", "-m", "unittest", "discover", "-s", "persist", "-p", "test_*.py", "-q"], wt_dir)

    with open(os.path.join(wt_dir, "shape", "design.out"), "r", encoding="utf-8") as fh:
        design_has_fix = _design_out_has_smollm2_fix(fh.read())

    nt_ok, nt_out = _run_cmd_ok([sys.executable, "-I", "-B", "checks/number_trace.py"], wt_dir)
    with open(os.path.join(wt_dir, "checks", "number_trace.out"), "r", encoding="utf-8") as fh:
        nt_exp = fh.read()
    nt_exact = nt_ok and bool(nt_exp.strip()) and (nt_out == nt_exp)

    cl_ok, cl_out = _run_cmd_ok([sys.executable, "-I", "-B", "checks/claim_ledger.py"], wt_dir)
    with open(os.path.join(wt_dir, "checks", "claim_ledger.out"), "r", encoding="utf-8") as fh:
        cl_exp = fh.read()
    cl_exact = cl_ok and bool(cl_exp.strip()) and (cl_out == cl_exp)

    pbd_ok, _ = _run_cmd_ok([sys.executable, "-I", "-B", "checks/plan_before_data.py"], wt_dir)
    return {
        "pr12_shape_tests_passed": shape_ok,
        "pr12_perm_tests_passed": perm_ok,
        "pr12_persist_tests_passed": persist_ok,
        "pr12_lattice_tests_executed_in_live_check": False,
        "pr12_design_out_has_smollm2_fix": design_has_fix,
        "pr12_number_trace_exact_match": nt_exact,
        "pr12_claim_ledger_exact_match": cl_exact,
        "pr12_plan_before_data_exit_zero": pbd_ok,
    }


def compute_live_siliquant_diagnostics(repo_dir):
    """Strictly 100% read-only live audit of local Moissanite-Labs/siliquant via temporary shared clone.

    Uses `git clone --shared --no-checkout` + `git checkout --detach` into a temporary
    directory so ZERO bytes are ever written inside `repo_dir` or `repo_dir/.git`.
    """
    if not os.path.isdir(os.path.join(repo_dir, ".git")):
        raise SiliquantBridgeError("REPO_MISSING", f"not a git repository: {repo_dir}")

    main_sha = _git_rev_parse(repo_dir, "origin/main")
    found, parents = _run_cmd_ok(
        ["git", "log", "-1", "--first-parent", "--format=%P",
         "--grep=^Merge pull request #12 ", main_sha],
        repo_dir,
    )
    if not found or len(parents.split()) != 2:
        raise SiliquantBridgeError(
            "GIT_REF_MISSING",
            "cannot resolve the merge for PR #12; fetch the full history of origin/main if the clone is shallow",
        )
    _, pr12_sha = parents.split()
    pr24_sha = _git_rev_parse(repo_dir, "origin/expedition/leech-l1-hadamard-symbiosis")
    pr26_sha = _git_rev_parse(repo_dir, "origin/feat/magnitude-autotune-quant-bridge")
    pr27_sha = _git_rev_parse(repo_dir, "origin/feat/round8-inkling-loom-speculative-tomography")

    with tempfile.TemporaryDirectory(prefix="siliquant_ro_") as tmpdir:
        wt_dir = os.path.join(tmpdir, "pr12")
        clone_p = subprocess.run(
            ["git", "clone", "--shared", "--no-checkout", "-q", repo_dir, wt_dir],
            capture_output=True,
            text=True,
            check=False,
        )
        if clone_p.returncode != 0:
            raise SiliquantBridgeError("CLONE_FAILED", f"git clone --shared failed: {clone_p.stderr.strip()}")
        co_p = subprocess.run(
            ["git", "-C", wt_dir, "checkout", "--detach", "-q", pr12_sha],
            capture_output=True,
            text=True,
            check=False,
        )
        if co_p.returncode != 0:
            raise SiliquantBridgeError("CHECKOUT_FAILED", f"git checkout --detach failed: {co_p.stderr.strip()}")
        checks = _verify_pr12_worktree(wt_dir)

    out = {
        "diag_source": "live_git_inspection",
        "main_commit": main_sha,
        "pr12_head_commit": pr12_sha,
        "draft_pr24_head": pr24_sha,
        "draft_pr26_head": pr26_sha,
        "draft_pr27_head": pr27_sha,
    }
    out.update(checks)
    return out


def certify_from_repo_diag(mode, live_repo_recounted, repo_diag):
    rd = audit_siliquant_repo_diagnostics_exact(repo_diag)
    p1 = audit_branchless_e8_and_zero_codeword_exact()
    p2 = audit_tpu_v5e_roofline_exact()
    p3 = audit_nature_to_siliquant_bridges_exact()
    p4 = audit_bayt_al_hikma_quant_bounds_exact()
    axes = {
        "pr12_live_reproducibility": rd["passed"],
        "branchless_e8_and_zero_codeword": p1["passed"],
        "tpu_v5e_part_e_roofline": p2["passed"],
        "nature_homeostasis_and_kirchhoff_vs_mst": p3["passed"],
        "bayt_al_hikma_cancellation_and_contraction": p4["passed"],
    }
    return certify_siliquant_bridge_audit_exact(
        mode=mode,
        live_repo_recounted=live_repo_recounted,
        operator_labels=OPERATOR_LABELS,
        repo_diag_result=rd,
        pillar1_result=p1,
        pillar2_result=p2,
        pillar3_result=p3,
        pillar4_result=p4,
        five_axis_checks=axes,
    )


def default_siliquant_repo():
    return os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "..", "..", "publish", "siliquant"))


def run_audit(live=False, siliquant_repo=None):
    if not live and siliquant_repo is None:
        return certify_from_repo_diag("standalone_fixture_replay", False, STANDALONE_REPO_DIAG_FIXTURE)
    target = default_siliquant_repo() if siliquant_repo is None else siliquant_repo
    diag = compute_live_siliquant_diagnostics(target)
    return certify_from_repo_diag("live_repo_recount", True, diag)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Track S6 (AUDIT-103) Moissanite-Labs/siliquant Bridge Audit")
    parser.add_argument("--standalone", action="store_true", help="Offline fixture replay for repo diag (default)")
    parser.add_argument("--live", action="store_true", help="Live read-only audit of local siliquant git repo")
    parser.add_argument("--siliquant-repo", "--repo-root", dest="siliquant_repo", default=None, help="Path to local siliquant clone (implies live)")
    parser.add_argument("--json", action="store_true", help="Emit canonical JSON to stdout (default output format)")
    parser.add_argument("--output", default=None, help="Optional path to write canonical JSON certificate")
    args = parser.parse_args(argv)

    use_live = args.live or (args.siliquant_repo is not None)
    cert = run_audit(live=use_live, siliquant_repo=args.siliquant_repo)
    raw = canonical_json_bytes(cert)
    if args.output:
        with open(args.output, "wb") as fh:
            fh.write(raw)
            fh.write(b"\n")
    sys.stdout.write(raw.decode("utf-8") + "\n")
    return 0 if cert["all_gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
