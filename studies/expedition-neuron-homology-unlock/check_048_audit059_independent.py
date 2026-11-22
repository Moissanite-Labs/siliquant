#!/usr/bin/env python3
"""Published, parameterized, zero-shared-code independent verifier for AUDIT-059 (closes N8).

Previously kept in an internal verification script (`check_048_audit059_qwen_neuron_map.py`)
with local paths; now stripped of all host-specific paths, parameterized via CLI flags,
and runnable out of the box in standalone mode with Python 3 standard library only
(zero imports from `neuron_homology_exact.py` or `qwen_neuron_map_bridge.py`, zero bare
`assert` statements, zero floats in JSON output).

Modes:
  1. Standalone mode (default): verifies the committed `qwen_neuron_map_certificate.json`
     integrity, zero-float hygiene, 13-model neuron census consistency, exact rational
     hypergeometric + two-block stratified convolutions (L15 & L19), honest 21/24 all-layer
     sign test, exact Z_2 chance-agreement & Cohen's kappa identities, Gemma 4 E4B L10
     summary counts, Flint Goalkeeper exact Fisher/Bonferroni tails, and `-O` prompt-leak
     rejection via an isolated subprocess check.
  2. Live upstream mode (`--live --repos-dir <dir> [--siliquant-readme <path>]`):
     additionally parses the 15 static HTML atlas pages, real float32 `.npy`/`.npz` weight
     matrices (via pure-stdlib binary parsing), raw Flint Goalkeeper JSON rows, verbatim
     upstream bare-assert positions (FG-01, WE-01), and live `-O` bypass on upstream
     `recount.py`.

Safe under: python3 -I -B, python3 -I -O -B, python3 -I -B -W error, python3 -I -O -B -W error.
"""

from __future__ import annotations

import argparse
import array
import ast
import base64
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import subprocess
import sys
from typing import Any

HERE = Path(__file__).resolve().parent
DEFAULT_CERT = HERE / "qwen_neuron_map_certificate.json"
DEFAULT_SILI_README = (
    HERE.parent.parent
    / "experiments"
    / "output-budget-gemma-e4b-2026-11-17"
    / "README.md"
)


class CheckFailure(RuntimeError):
    """Raised on any independent verification discrepancy."""


def must(cond: bool, msg: str) -> None:
    if not cond:
        raise CheckFailure(msg)


def check_no_floats(obj: Any, path: str = "root") -> None:
    must(not isinstance(obj, float), f"Float found at {path}: {obj!r}")
    if isinstance(obj, dict):
        for k, v in obj.items():
            check_no_floats(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            check_no_floats(v, f"{path}[{i}]")


def indep_hypergeom_tail(pop: int, succ: int, draws: int, obs: int) -> Fraction:
    num = sum(
        math.comb(succ, k) * math.comb(pop - succ, draws - k)
        for k in range(obs, min(succ, draws) + 1)
    )
    return Fraction(num, math.comb(pop, draws))


def indep_stratified_tail(
    n1: int, k1: int, d1: int, n2: int, k2: int, d2: int, obs_total: int
) -> Fraction:
    den = math.comb(n1, d1) * math.comb(n2, d2)
    num = 0
    for x1 in range(max(0, d1 + k1 - n1), min(k1, d1) + 1):
        for x2 in range(max(0, d2 + k2 - n2), min(k2, d2) + 1):
            if x1 + x2 >= obs_total:
                num += (
                    math.comb(k1, x1)
                    * math.comb(n1 - k1, d1 - x1)
                    * math.comb(k2, x2)
                    * math.comb(n2 - k2, d2 - x2)
                )
    return Fraction(num, den)


def indep_binom_two_sided(n: int, k: int) -> Fraction:
    m = min(k, n - k)
    val = Fraction(2 * sum(math.comb(n, j) for j in range(0, m + 1)), 1 << n)
    return Fraction(1) if val > 1 else val


def verify_standalone(cert_path: Path) -> dict[str, Any]:
    must(cert_path.is_file(), f"Missing certificate: {cert_path}")
    cert = json.loads(cert_path.read_text(encoding="utf-8"))
    check_no_floats(cert)

    body = {k: v for k, v in cert.items() if k != "canonical_sha256"}
    recomputed = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    must(recomputed == cert["canonical_sha256"], "canonical_sha256 mismatch")

    p1 = cert["part1_provenance_and_hygiene"]
    must(p1["qwen_neuron_map_fetched_pages"] == 15, "Expected 15 fetched pages")
    must(len(p1["qwen_neuron_map_sha256"]) == 15, "Expected 15 page SHA-256 digests")

    # Part 2: 13-model neuron census, Pythia single-run py0 trajectory + endpoints, SmolLM2 decay, Z_2 algebra
    p2 = cert["part2_atlas_30models_13full_and_z2_confounder"]
    per13 = p2["per_model_13_summary"]
    must(len(per13) == 13, f"Expected 13 models, got {len(per13)}")
    total_neurons = 0
    dark_fracs: dict[str, Fraction] = {}
    for slug, rec in per13.items():
        n = int(rec["neurons"])
        must(int(rec["layers"]) * int(rec["width"]) == n, f"{slug} shape mismatch")
        df = Fraction(int(rec["dark_neurons"]), n)
        must(df == Fraction(rec["dark_fraction_exact"]), f"{slug} dark_fraction mismatch")
        lo_f = Fraction(rec["dark_lo_1_7_exact"])
        hi_f = Fraction(rec["dark_hi_3_7_exact"])
        must(hi_f / lo_f == Fraction(rec["hi_to_lo_dark_ratio_exact"]), f"{slug} hi/lo ratio mismatch")
        total_neurons += n
        dark_fracs[slug] = df
    must(total_neurons == 1_504_256 == p2["total_neurons_parsed"], f"total_neurons={total_neurons}")
    # Pythia: py0a (1k) -> py0b (8k) -> py0c (32k) -> py0 (143k) is one run; py1/py2 are 143k endpoints.
    must(dark_fracs["py0a"] > dark_fracs["py0b"] > dark_fracs["py0c"], "Pythia py0 drop failed")
    must(
        dark_fracs["py0c"] < dark_fracs["py0"]
        and dark_fracs["py0c"] < dark_fracs["py1"]
        and dark_fracs["py0c"] < dark_fracs["py2"],
        "Pythia 143k > 32k failed",
    )
    must(dark_fracs["s135"] > dark_fracs["smol"] > dark_fracs["s17"], "SmolLM2 decay failed")

    # Independent Z_2 chance-agreement identity on rational grid + certificate values
    for p1_num in range(0, 5):
        for p2_num in range(0, 5):
            q1, q2 = Fraction(p1_num, 4), Fraction(p2_num, 4)
            lhs = q1 * q2 + (1 - q1) * (1 - q2)
            rhs = Fraction(1, 2) + Fraction(1, 2) * (2 * q1 - 1) * (2 * q2 - 1)
            must(lhs == rhs, "Z_2 chance identity mismatch")
    lin = p2["z2_sign_symmetry_lineage_proofs"]
    must(Fraction(lin["pythia_vs_falcon_h1_confounder_85pct"]["chance_agreement"]) == Fraction(17, 20), "PF chance")
    must(Fraction(lin["pythia_vs_falcon_h1_confounder_85pct"]["cohen_kappa"]) == 0, "PF kappa")
    must(Fraction(lin["qwen35_08b_vs_2b_true_lineage_93pct"]["cohen_kappa"]) == Fraction(43, 50), "Q08/Q2 kappa")
    must(Fraction(lin["qwen35_4b_vs_9b_true_lineage_98pct"]["cohen_kappa"]) == Fraction(24, 25), "Q4B/Q9B kappa")
    must(Fraction(lin["qwen3_06b_vs_17b_true_lineage_88pct"]["cohen_kappa"]) == Fraction(19, 25), "Q3 kappa")

    # Part 3: exact hypergeometric, two-block stratified convolutions, honest 21/24 sign test, McNemar
    p3 = cert["part3_q08_weight_edit_empirical_join"]
    p_sign_21_21 = indep_binom_two_sided(21, 21)
    p_sign_21_24 = indep_binom_two_sided(24, 21)
    must(p_sign_21_21 == Fraction(1, 1_048_576), "21/21 sign p")
    must(p_sign_21_24 == Fraction(2325, 8_388_608), "21/24 all-layer sign p")

    p_top40 = indep_hypergeom_tail(7168, 1024, 40, 29)
    must(p_top40 < Fraction(1, 10**14), "top-40 Linf tail")

    p_l15_all = indep_hypergeom_tail(3584, 77, 64, 16)
    p_l15_lo = indep_hypergeom_tail(512, 29, 40, 9)
    p_l15_hi = indep_hypergeom_tail(3072, 48, 24, 7)
    p_l15_strat = indep_stratified_tail(512, 29, 40, 3072, 48, 24, 16)
    p_l19_all = indep_hypergeom_tail(3584, 11, 64, 4)
    p_l19_lo = indep_hypergeom_tail(512, 8, 35, 3)
    p_l19_hi = indep_hypergeom_tail(3072, 3, 29, 1)
    p_l19_strat = indep_stratified_tail(512, 8, 35, 3072, 3, 29, 4)
    must(p_l15_all < Fraction(1, 10**13), "L15 adj p")
    must(p_l15_lo < Fraction(1, 100) and p_l15_hi < Fraction(1, 10**7), "L15 both blocks < 0.01")
    must(p_l15_strat < Fraction(1, 10**9), "L15 strat p")
    must(p_l19_all < Fraction(1, 10**4), "L19 adj p")
    must(p_l19_lo > Fraction(1, 100) and p_l19_hi > Fraction(1, 100), "L19 individual blocks > 0.01")
    must(p_l19_strat < Fraction(2, 1000), "L19 strat p")
    must(indep_binom_two_sided(579 + 5, 5) < Fraction(1, 10**150), "Exp 8 McNemar p")

    # Part 4: Gemma 4 E4B Layer-10 exploratory summary
    p4 = cert["part4_gemma4_e4b_layer10_siliquant_bridge"]
    must(p4["rank1_mean_out_layer"] == 10 and p4["rank2_mean_out_layer"] == 9, "L10/L09 rank")
    must(p4["first_code_gt_prose_layer"] == 10, "first_code_gt_prose")
    must(Fraction(1560, 1560 + 182) == Fraction(60, 67) == Fraction(p4["siliquant_cross_reference"]["l10_kl_share_exact"]), "60/67")

    # Part 5: Flint Goalkeeper exact Fisher + Bonferroni-6
    p5 = cert["part5_flint_goalkeeper_hardened_recount"]
    p_fix = indep_hypergeom_tail(64, 15, 16, 15)
    p_ood = indep_hypergeom_tail(64, 16, 16, 16)
    must(p_fix == Fraction(1, 9_969_937_491_420) == Fraction(p5["fixed_replication"]["exact_fisher_p"]), "Flint fixed p")
    must(6 * p_fix == Fraction(1, 1_661_656_248_570) == Fraction(p5["fixed_bonferroni_6_exact"]), "Flint Bonf-6")
    must(p_ood == Fraction(1, 488_526_937_079_580) == Fraction(p5["out_of_domain_preregistered"]["exact_fisher_p"]), "Flint OOD p")

    # Isolated subprocess `-O` negative control against sibling kernel (zero direct import)
    probe_code = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "import neuron_homology_exact as m; "
        "rows = [{'condition': c, 'prompt': ('goalkeeper' if (c=='target' and i==0) else 'neutral'), "
        "'output': 'goalkeeper' if c=='target' else 'bench'} "
        "for c in m.GOALKEEPER_CONDITIONS for i in range(4)]; "
        "m.strict_recount_goalkeeper_rows(rows, expected_per_condition=4)"
    )
    proc = subprocess.run(
        [sys.executable, "-I", "-O", "-B", "-c", probe_code, str(HERE)],
        capture_output=True,
        text=True,
    )
    must(proc.returncode != 0 and "ExactHomologyError" in proc.stderr, "FG-01 -O rejection probe failed")

    return {
        "canonical_sha256": recomputed,
        "total_neurons_verified": total_neurons,
        "q08_all_24_layers_sign_p_exact": str(p_sign_21_24),
        "l15_both_blocks_lt_0_01": True,
        "l15_strat_p_lt_1e_9": True,
        "l19_both_blocks_lt_0_01": False,
        "l19_strat_p_lt_2e_3": True,
        "fg01_dash_O_rejection_verified": True,
    }


def _extract_html_data(html_path: Path) -> dict[str, Any]:
    txt = html_path.read_text(encoding="utf-8")
    m = re.search(r"const DATA\s*=\s*(\{.*?\});\s*\n", txt, re.S)
    must(m is not None, f"const DATA missing in {html_path.name}")
    return json.loads(m.group(1))  # type: ignore[union-attr]


def _load_npy_f32_stdlib(path: Path) -> tuple[tuple[int, ...], array.array]:
    raw = path.read_bytes()
    must(raw[:6] == b"\x93NUMPY", f"not a .npy file: {path.name}")
    major = raw[6]
    hlen = struct.unpack("<H", raw[8:10])[0] if major == 1 else struct.unpack("<I", raw[8:12])[0]
    off = (10 if major == 1 else 12) + hlen
    hdr = ast.literal_eval(raw[(10 if major == 1 else 12) : off].decode("latin1").strip())
    must(hdr["descr"] == "<f4" and not hdr["fortran_order"], f"unexpected dtype/order in {path.name}")
    shape = tuple(int(x) for x in hdr["shape"])
    total = 1
    for s in shape:
        total *= s
    arr = array.array("f")
    arr.frombytes(raw[off : off + 4 * total])
    if sys.byteorder != "little":
        arr.byteswap()
    return shape, arr


def verify_live(cert_path: Path, repos_dir: Path, sili_readme: Path | None) -> dict[str, Any]:
    cert = json.loads(cert_path.read_text(encoding="utf-8"))
    site_dir = repos_dir / "qwen-neuron-map-site"
    we_repo = repos_dir / "alphaxiv-weight-edit-minimization"
    fl_repo = repos_dir / "alphaxiv-flint-goalkeeper-neuron"
    p1 = cert["part1_provenance_and_hygiene"]

    for repo, key in (
        (we_repo, "alphaxiv_weight_edit_minimization"),
        (fl_repo, "alphaxiv_flint_goalkeeper_neuron"),
    ):
        sha = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip()
        must(sha == p1[key]["head_sha"], f"SHA mismatch on {repo.name}")
        must(dirty == "", f"Dirty clone {repo.name}")

    for fname, expected in p1["qwen_neuron_map_sha256"].items():
        actual = hashlib.sha256((site_dir / fname).read_bytes()).hexdigest()
        must(actual == expected, f"Site digest mismatch for {fname}")

    def scan_asserts(repo: Path) -> list[list[Any]]:
        out: list[list[Any]] = []
        for p in sorted(repo.rglob("*.py")):
            tree = ast.parse(p.read_text(encoding="utf-8"), filename=p.name)
            for node in ast.walk(tree):
                if isinstance(node, ast.Assert):
                    out.append([p.relative_to(repo).as_posix(), node.lineno])
        return sorted(out)

    must(scan_asserts(we_repo) == [["weight-edit-minimization.py", 1104]], "WE-01 asserts mismatch")
    must(scan_asserts(fl_repo) == [["recount.py", 39], ["recount.py", 45]], "FG-01 asserts mismatch")

    # Verify all-24-layer Qwen3.5-0.8B counts directly from q08.html (21/24 hi > lo; L00, L01, L23 reverse)
    q08 = _extract_html_data(site_dir / "q08.html")
    n, w, layers = int(q08["n"]), int(q08["width"]), int(q08["layers"])
    ids = struct.unpack(f"<{n}H", base64.b64decode(q08["structures"]["ids"]))
    fam_of = q08["structures"]["familyOf"]
    hi_gt_lo_layers: list[int] = []
    reversed_layers: list[int] = []
    for lay in range(layers):
        base = lay * w
        d_lo = sum(1 for c in range(0, 512) if ids[base + c] == 65535 or fam_of[ids[base + c]] < 0)
        d_hi = sum(1 for c in range(2048, w) if ids[base + c] == 65535 or fam_of[ids[base + c]] < 0)
        if Fraction(d_hi, w - 2048) > Fraction(d_lo, 512):
            hi_gt_lo_layers.append(lay)
        else:
            reversed_layers.append(lay)
    must(len(hi_gt_lo_layers) == 21 and reversed_layers == [0, 1, 23], f"Unexpected reversed layers: {reversed_layers}")

    # Verify sub-2^-16 bf16 entry counts (N4) directly from base_l15 & base_l19 .npy files
    sub_counts: dict[str, int] = {}
    exp8 = we_repo / "data" / "exp8"
    for layer in (15, 19):
        shape, arr = _load_npy_f32_stdlib(exp8 / f"base_l{layer}_down_proj_f32.npy")
        must(shape == (1024, 3584), f"Unexpected L{layer} shape {shape}")
        sub_counts[f"L{layer}_sub_2_neg16"] = sum(1 for v in arr if (v * 65536.0) != int(v * 65536.0))
    must(sub_counts["L15_sub_2_neg16"] == 505_149 and sub_counts["L19_sub_2_neg16"] == 447_042, f"sub_counts={sub_counts}")

    # Verify raw Flint Goalkeeper JSONs
    pat = re.compile(r"\bgoalkeeper\b", re.IGNORECASE)
    for fname, exp_hits in (("fixed-replication-rows.json", 15), ("out-of-domain-rows.json", 16)):
        rows = json.loads((fl_repo / "data" / fname).read_text(encoding="utf-8"))
        must(len(rows) == 64 and sum(1 for r in rows if pat.search(r["prompt"])) == 0, f"{fname} integrity")
        t_hits = sum(1 for r in rows if r["condition"] == "target" and pat.search(r["output"]))
        c_hits = sum(1 for r in rows if r["condition"] != "target" and pat.search(r["output"]))
        must(t_hits == exp_hits and c_hits == 0, f"{fname} counts {(t_hits, c_hits)}")

    # Optional siliquant README cross-check
    sili_checked = False
    if sili_readme is not None and sili_readme.is_file():
        sili_txt = sili_readme.read_text(encoding="utf-8")
        must("layer 10's `ffn_down` scores 0.1560. The other 41 layers add up to 0.0182" in sili_txt, "siliquant L10 quote")
        must("0.05 percent of its damage" in sili_txt, "siliquant top-10 quote")
        sili_checked = True

    return {
        "pages_verified": 15,
        "clones_clean": 2,
        "q08_reversed_boundary_layers": reversed_layers,
        "sub_2_neg16_bf16_entries": sub_counts,
        "siliquant_readme_verbatim_checked": sili_checked,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Parameterized zero-shared-code independent verifier for AUDIT-059."
    )
    parser.add_argument("--cert", type=Path, default=DEFAULT_CERT, help="Path to qwen_neuron_map_certificate.json.")
    parser.add_argument("--live", action="store_true", help="Also verify raw upstream clones under --repos-dir.")
    parser.add_argument("--repos-dir", type=Path, default=None, help="Directory holding read-only upstream clones.")
    parser.add_argument(
        "--siliquant-readme",
        type=Path,
        default=DEFAULT_SILI_README,
        help="Optional path to siliquant output-budget-gemma-e4b-2026-11-17 README.md.",
    )
    args = parser.parse_args(argv)

    out: dict[str, Any] = {
        "check": "check_048_audit059_independent",
        "mode": "live" if args.live else "standalone",
        "standalone": verify_standalone(args.cert),
    }
    if args.live:
        must(args.repos_dir is not None, "--live requires --repos-dir <path>")
        out["live"] = verify_live(args.cert, args.repos_dir, args.siliquant_readme)
    out["verdict"] = "ALL_INDEPENDENT_CHECKS_PASSED"
    check_no_floats(out)
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
