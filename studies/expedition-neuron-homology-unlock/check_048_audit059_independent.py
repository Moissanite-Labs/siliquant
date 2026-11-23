#!/usr/bin/env python3
"""Published, parameterized, zero-shared-code independent verifier for AUDIT-059 (closes N8).

Previously kept in an internal verification script (`check_048_audit059_qwen_neuron_map.py`)
with local paths; now stripped of all host-specific paths, parameterized via CLI flags,
and runnable out of the box in standalone mode with Python 3 standard library only
(zero imports from `neuron_homology_exact.py` or `qwen_neuron_map_bridge.py`, zero bare
`assert` statements, zero floats in JSON output).

Modes:
  1. Standalone mode (default): verifies the committed `qwen_neuron_map_certificate.json`
     against both its pinned `canonical_sha256` identity and full semantic recomputation
     derived directly from the supplied certificate fields: zero-float hygiene, pinned
     upstream repo HEAD SHAs and all 15 page digests, 13-model neuron census consistency,
     exact Z_2 chance-agreement & Cohen's kappa recomputation, exact rational hypergeometric
     and two-block stratified convolutions derived from `part3.layer_joins` (L15 & L19),
     `top40_linf_outliers_in_col_lt_512` cross-check, honest 21/24 all-layer sign test,
     cross-layer residual-axis & McNemar checks, Gemma 4 E4B L10 summary checks, Flint
     Goalkeeper exact Fisher/Bonferroni tails derived from `part5` counts, and `-O`
     prompt-leak rejection via an isolated subprocess check.
  2. Live upstream mode (`--live --repos-dir <dir> [--siliquant-readme <path>]`):
     additionally verifies live upstream repo HEAD SHAs and clean working trees, AST
     bare-assert positions (`FG-01`, `WE-01`), SHA-256 digests of all 15 static HTML atlas
     pages, direct parsing of `q08.html` (all-24-layer `[0, 1, 23]` boundary counts) and
     both `base_l15`/`base_l19` float32 `.npy` matrices (sub-2^-16 bf16 entry counts), raw
     Flint Goalkeeper JSON row counts, and the optional `siliquant` README quote.
     (Full live extraction of all 13 HTML models, `.npz` factors, and Gemma L10 lives in
     `qwen_neuron_map_bridge.py --live`.)

Safe under: python3 -I -B, python3 -I -O -B, python3 -I -B -W error, python3 -I -O -B -W error.
"""

from __future__ import annotations

import argparse
import array
import ast
import base64
from fractions import Fraction
import hashlib
import hmac
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

PINNED_CANONICAL_SHA256 = "4abe60205f6b6a43762a19711d4a72936ca34e67cf751410a825ef1e7f478289"
PINNED_WE_HEAD_SHA = "768d7340142b1882d4bd2740b4723274e4c9278d"
PINNED_FL_HEAD_SHA = "6ece43a96cef46e7013cae4b0423ee60c29c1ab5"
PINNED_PAGE_SHA256: dict[str, str] = {
    "4b.html": "cfc229397e1e98d46ccaa270295bcb02114159bfb2ed79cc77f7f6262220f22d",
    "compare.html": "bb30932da8bc529977c6092cd168394829fbb306541887777dd65e7fb241de8e",
    "g4e4.html": "f1340b4f09edd898c39e566669463ac41450e1a506024456f23868f3955710d3",
    "index.html": "179059c77f9702b3ba65d2a266ecfea92afe3458cc1b7dbe7310595dbd55ec78",
    "py0.html": "4b08cddc5d8a980377670562fe6defb03a9479deb49c95cb6e7d25094ca7e8c3",
    "py0a.html": "4534691a7ae69ade0d1dff044950b60b25f134ea815a99f3ca80e4c71f492774",
    "py0b.html": "48cdb68d70897e36ef04824a430b7d3fd5cbfb4d09f421d063aa4b5941fd8bdd",
    "py0c.html": "b231493347a06d18d187c7d5383553f7e0e815acc592962ceb5070ff35d902a3",
    "py1.html": "443d3fac680a0afa8d63a0f6c3a5b642062ebfe30b345a6d2a061cf660b59bab",
    "py2.html": "c29b46f4393c924227364f31db8e8b68136405bb05c2e2d2814273d61fdaa5b8",
    "q08.html": "1292ab4889313caa3a91976488e8de7b6aa4353e3d99b68bf8f7c4e226c371fa",
    "q2.html": "9c38e3c9102b526418295ceefcbe07afbaecf3314010423e8adaa8bab00633e9",
    "s135.html": "3f06b5d778345180004c608211428d29ecb818967186b4a99ca66a9acea6cba4",
    "s17.html": "51e6f077d4abd5032a378fb3d04dc324c0357196803ae137aa3b59c3d46464b9",
    "smol.html": "139c9af65dce4799ceacce3170d574f0749638147ae60d02b53b85fe859b76ee",
}
_HEX_LOWER = frozenset("0123456789abcdef")


class CheckFailure(RuntimeError):
    """Raised on any independent verification discrepancy."""


def must(cond: bool, msg: str) -> None:
    if not cond:
        raise CheckFailure(msg)


def _require_hex_digest(val: Any, length: int, label: str) -> str:
    must(type(val) is str and len(val) == length and all(c in _HEX_LOWER for c in val), f"invalid {label}")
    return val


def _require_int(val: Any, label: str, *, lo: int | None = None) -> int:
    must(type(val) is int, f"{label} must be strict int")
    if lo is not None:
        must(val >= lo, f"{label} must be >= {lo}")
    return val


def _require_bool(val: Any, label: str) -> bool:
    must(type(val) is bool, f"{label} must be strict bool")
    return val


def _require_pair_int(val: Any, label: str) -> tuple[int, int]:
    must(type(val) is list and len(val) == 2, f"{label} must be 2-element list")
    a = _require_int(val[0], f"{label}[0]", lo=0)
    b = _require_int(val[1], f"{label}[1]", lo=1)
    must(a <= b, f"{label} numerator exceeds denominator")
    return a, b


def check_no_floats(obj: Any, path: str = "root") -> None:
    must(not isinstance(obj, float), f"Float found at {path}: {obj!r}")
    if isinstance(obj, dict):
        for k, v in obj.items():
            check_no_floats(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            check_no_floats(v, f"{path}[{i}]")


def indep_hypergeom_tail(pop: int, succ: int, draws: int, obs: int) -> Fraction:
    must(pop >= 1 and 0 <= succ <= pop and 0 <= draws <= pop, "invalid hypergeometric domain")
    k_min = max(0, draws + succ - pop)
    k_max = min(succ, draws)
    must(k_min <= obs <= k_max, f"observed={obs} outside feasible [{k_min}, {k_max}]")
    num = sum(
        math.comb(succ, k) * math.comb(pop - succ, draws - k)
        for k in range(obs, k_max + 1)
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
    must(n >= 1 and 0 <= k <= n, "invalid binomial domain")
    m = min(k, n - k)
    val = Fraction(2 * sum(math.comb(n, j) for j in range(0, m + 1)), 1 << n)
    return Fraction(1) if val > 1 else val


def verify_certificate_semantics(cert: dict[str, Any]) -> dict[str, Any]:
    """Derive and verify every AUDIT-059 claim directly from the supplied certificate fields."""
    must(type(cert) is dict, "certificate root must be a dict")
    check_no_floats(cert)

    # Part 1: Provenance, upstream HEAD SHAs, bare-assert hygiene, and all 15 page digests
    p1 = cert.get("part1_provenance_and_hygiene")
    must(type(p1) is dict, "missing part1_provenance_and_hygiene")
    we_meta = p1.get("alphaxiv_weight_edit_minimization")
    fl_meta = p1.get("alphaxiv_flint_goalkeeper_neuron")
    must(type(we_meta) is dict and type(fl_meta) is dict, "missing upstream repo provenance")
    we_sha = _require_hex_digest(we_meta.get("head_sha"), 40, "we_head_sha")
    fl_sha = _require_hex_digest(fl_meta.get("head_sha"), 40, "fl_head_sha")
    must(hmac.compare_digest(we_sha, PINNED_WE_HEAD_SHA), "upstream weight-edit head_sha mismatch")
    must(hmac.compare_digest(fl_sha, PINNED_FL_HEAD_SHA), "upstream flint-goalkeeper head_sha mismatch")
    must(_require_bool(we_meta.get("clean_read_only"), "we_clean"), "we repo not clean_read_only")
    must(_require_bool(fl_meta.get("clean_read_only"), "fl_clean"), "fl repo not clean_read_only")
    hyg = p1.get("hygiene_bare_asserts")
    must(type(hyg) is dict, "missing hygiene_bare_asserts")
    must(hyg.get("FG_01_flint_goalkeeper_neuron") == [["recount.py", 39], ["recount.py", 45]], "FG-01 asserts mismatch")
    must(hyg.get("WE_01_weight_edit_minimization") == [["weight-edit-minimization.py", 1104]], "WE-01 asserts mismatch")

    must(_require_int(p1.get("qwen_neuron_map_fetched_pages"), "fetched_pages") == 15, "Expected 15 fetched pages")
    pages = p1.get("qwen_neuron_map_sha256")
    must(type(pages) is dict and set(pages.keys()) == set(PINNED_PAGE_SHA256.keys()), "page digest key set mismatch")
    for fname, exp_dig in PINNED_PAGE_SHA256.items():
        got_dig = _require_hex_digest(pages[fname], 64, f"page_sha256[{fname}]")
        must(hmac.compare_digest(got_dig, exp_dig), f"page digest mismatch for {fname}")

    # Part 2: 13-model neuron census, Pythia single-run py0 trajectory + endpoints, SmolLM2 decay, Z_2 algebra
    p2 = cert.get("part2_atlas_30models_13full_and_z2_confounder")
    must(type(p2) is dict, "missing part2_atlas_30models_13full_and_z2_confounder")
    per13 = p2.get("per_model_13_summary")
    must(type(per13) is dict and len(per13) == 13, "Expected 13 models in per_model_13_summary")
    total_neurons = 0
    dark_fracs: dict[str, Fraction] = {}
    for slug, rec in per13.items():
        must(type(rec) is dict, f"model record {slug} must be dict")
        n = _require_int(rec.get("neurons"), f"{slug}.neurons", lo=1)
        layers_m = _require_int(rec.get("layers"), f"{slug}.layers", lo=1)
        width_m = _require_int(rec.get("width"), f"{slug}.width", lo=1)
        dark_n = _require_int(rec.get("dark_neurons"), f"{slug}.dark_neurons", lo=0)
        must(layers_m * width_m == n, f"{slug} shape mismatch")
        df = Fraction(dark_n, n)
        must(df == Fraction(rec["dark_fraction_exact"]), f"{slug} dark_fraction mismatch")
        lo_f = Fraction(rec["dark_lo_1_7_exact"])
        hi_f = Fraction(rec["dark_hi_3_7_exact"])
        must(lo_f > 0 and hi_f / lo_f == Fraction(rec["hi_to_lo_dark_ratio_exact"]), f"{slug} hi/lo ratio mismatch")
        total_neurons += n
        dark_fracs[slug] = df
    must(
        total_neurons == _require_int(p2.get("total_neurons_parsed"), "total_neurons_parsed") == 1_504_256,
        f"total_neurons={total_neurons}",
    )
    must(dark_fracs["py0a"] > dark_fracs["py0b"] > dark_fracs["py0c"], "Pythia py0 drop failed")
    must(
        dark_fracs["py0c"] < dark_fracs["py0"]
        and dark_fracs["py0c"] < dark_fracs["py1"]
        and dark_fracs["py0c"] < dark_fracs["py2"],
        "Pythia 143k > 32k failed",
    )
    must(dark_fracs["s135"] > dark_fracs["smol"] > dark_fracs["s17"], "SmolLM2 decay failed")

    # Independent Z_2 chance-agreement identity on rational grid + EVERY record in certificate
    for p1_num in range(0, 5):
        for p2_num in range(0, 5):
            q1, q2 = Fraction(p1_num, 4), Fraction(p2_num, 4)
            lhs = q1 * q2 + (1 - q1) * (1 - q2)
            rhs = Fraction(1, 2) + Fraction(1, 2) * (2 * q1 - 1) * (2 * q2 - 1)
            must(lhs == rhs, "Z_2 chance identity mismatch")
    lin = p2.get("z2_sign_symmetry_lineage_proofs")
    expected_lin_keys = {
        "pythia_vs_falcon_h1_confounder_85pct",
        "qwen35_08b_vs_2b_true_lineage_93pct",
        "qwen35_4b_vs_9b_true_lineage_98pct",
        "qwen3_06b_vs_17b_true_lineage_88pct",
        "unrelated_swiglu_null",
    }
    must(type(lin) is dict and set(lin.keys()) == expected_lin_keys, "z2_sign_symmetry_lineage_proofs keys mismatch")
    for k_name, rec in lin.items():
        must(type(rec) is dict, f"lineage record {k_name} must be dict")
        a_obs = Fraction(rec["observed_agreement"])
        q1 = Fraction(rec["p_pos_1"])
        q2 = Fraction(rec["p_pos_2"])
        a_ch = q1 * q2 + (1 - q1) * (1 - q2)
        d_lin = a_obs - a_ch
        kap = d_lin / (1 - a_ch)
        must(a_ch == Fraction(rec["chance_agreement"]), f"{k_name} chance_agreement mismatch")
        must(d_lin == Fraction(rec["centered_excess"]), f"{k_name} centered_excess mismatch")
        must(kap == Fraction(rec["cohen_kappa"]), f"{k_name} cohen_kappa mismatch")
        must(_require_bool(rec.get("indicates_shared_origin"), f"{k_name}.shared") == (d_lin > 0), f"{k_name} shared flag mismatch")
    must(Fraction(lin["pythia_vs_falcon_h1_confounder_85pct"]["cohen_kappa"]) == 0, "PF kappa")
    must(Fraction(lin["qwen35_08b_vs_2b_true_lineage_93pct"]["cohen_kappa"]) == Fraction(43, 50), "Q08/Q2 kappa")
    must(Fraction(lin["qwen35_4b_vs_9b_true_lineage_98pct"]["cohen_kappa"]) == Fraction(24, 25), "Q4B/Q9B kappa")
    must(Fraction(lin["qwen3_06b_vs_17b_true_lineage_88pct"]["cohen_kappa"]) == Fraction(19, 25), "Q3 kappa")

    # Part 3: Derive ALL joins, outliers, sign tests, residual-axis, and McNemar checks directly from `p3`
    p3 = cert.get("part3_q08_weight_edit_empirical_join")
    must(type(p3) is dict, "missing part3_q08_weight_edit_empirical_join")
    q08_width = _require_int(per13["q08"]["width"], "q08.width", lo=1)
    q08_layers = _require_int(per13["q08"]["layers"], "q08.layers", lo=1)

    joins = p3.get("layer_joins")
    must(type(joins) is dict and set(joins.keys()) == {"L15", "L19"}, "layer_joins must contain exactly L15 and L19")
    layer_strat_p: dict[str, Fraction] = {}
    layer_block_lt_001: dict[str, bool] = {}
    top20_sum = 0
    for lname in ("L15", "L19"):
        lj = joins[lname]
        must(type(lj) is dict, f"layer_joins[{lname}] must be dict")
        sel_cnt = _require_int(lj.get("selected_count"), f"{lname}.selected_count", lo=1)
        sel_lt512 = _require_int(lj.get("selected_col_lt_512"), f"{lname}.selected_col_lt_512", lo=0)
        top20_in = _require_int(lj.get("top20_linf_in_col_lt_512"), f"{lname}.top20_linf_in_col_lt_512", lo=0)
        must(top20_in <= 20, f"{lname}.top20_linf_in_col_lt_512 > 20")
        top20_sum += top20_in
        must(_require_int(lj.get("top_row_linf_residual_dim"), f"{lname}.top_row_linf_residual_dim") == 0, f"{lname} top row dim != 0")

        # Adjacent brackets: all + two-block stratified conservation & exact tails
        adj = lj.get("adjacent_brackets")
        must(type(adj) is dict and set(adj.keys()) == {"all", "strat_col_lt_512", "strat_col_ge_512"}, f"{lname} adj keys")
        all_bg_k, all_bg_n = _require_pair_int(adj["all"].get("bg"), f"{lname}.adj.all.bg")
        all_sel_k, all_sel_n = _require_pair_int(adj["all"].get("sel"), f"{lname}.adj.all.sel")
        lo_bg_k, lo_bg_n = _require_pair_int(adj["strat_col_lt_512"].get("bg"), f"{lname}.adj.lo.bg")
        lo_sel_k, lo_sel_n = _require_pair_int(adj["strat_col_lt_512"].get("sel"), f"{lname}.adj.lo.sel")
        hi_bg_k, hi_bg_n = _require_pair_int(adj["strat_col_ge_512"].get("bg"), f"{lname}.adj.hi.bg")
        hi_sel_k, hi_sel_n = _require_pair_int(adj["strat_col_ge_512"].get("sel"), f"{lname}.adj.hi.sel")

        must(all_bg_n == q08_width and lo_bg_n + hi_bg_n == all_bg_n, f"{lname} adj bg width mismatch")
        must(lo_bg_k + hi_bg_k == all_bg_k, f"{lname} adj bg target conservation failed")
        must(all_sel_n == sel_cnt and lo_sel_n + hi_sel_n == all_sel_n and lo_sel_n == sel_lt512, f"{lname} adj sel count mismatch")
        must(lo_sel_k + hi_sel_k == all_sel_k, f"{lname} adj sel hit conservation failed")

        exp_enrich = (all_sel_k * all_bg_n * 100) // (all_sel_n * all_bg_k)
        must(_require_int(adj["all"].get("enrich_x100"), f"{lname}.adj.enrich_x100") == exp_enrich, f"{lname} adj enrich mismatch")

        p_all = indep_hypergeom_tail(all_bg_n, all_bg_k, all_sel_n, all_sel_k)
        p_lo = indep_hypergeom_tail(lo_bg_n, lo_bg_k, lo_sel_n, lo_sel_k)
        p_hi = indep_hypergeom_tail(hi_bg_n, hi_bg_k, hi_sel_n, hi_sel_k)
        p_strat = indep_stratified_tail(lo_bg_n, lo_bg_k, lo_sel_n, hi_bg_n, hi_bg_k, hi_sel_n, all_sel_k)

        must(_require_bool(adj["all"].get("exact_hypergeom_p_lt_1e_4"), f"{lname}.all.lt1e4") == (p_all < Fraction(1, 10**4)), f"{lname} adj all p flag")
        must(_require_bool(adj["strat_col_lt_512"].get("exact_hypergeom_p_lt_1e_2"), f"{lname}.lo.lt1e2") == (p_lo < Fraction(1, 100)), f"{lname} adj lo p flag")
        must(_require_bool(adj["strat_col_ge_512"].get("exact_hypergeom_p_lt_1e_1"), f"{lname}.hi.lt1e1") == (p_hi < Fraction(1, 10)), f"{lname} adj hi p flag")
        layer_strat_p[lname] = p_strat
        layer_block_lt_001[lname] = (p_lo < Fraction(1, 100)) and (p_hi < Fraction(1, 100))

        # Equations/numbers/units enrichment & tail
        eq = lj.get("equations_numbers_units")
        must(type(eq) is dict, f"{lname}.equations_numbers_units missing")
        eq_bg_k, eq_bg_n = _require_pair_int(eq.get("bg"), f"{lname}.eq.bg")
        eq_sel_k, eq_sel_n = _require_pair_int(eq.get("sel"), f"{lname}.eq.sel")
        must(eq_bg_n == q08_width and eq_sel_n == sel_cnt, f"{lname} eq dimensions mismatch")
        eq_enrich = (eq_sel_k * eq_bg_n * 100) // (eq_sel_n * eq_bg_k)
        must(_require_int(eq.get("enrich_x100"), f"{lname}.eq.enrich_x100") == eq_enrich, f"{lname} eq enrich mismatch")
        p_eq = indep_hypergeom_tail(eq_bg_n, eq_bg_k, eq_sel_n, eq_sel_k)
        must(_require_bool(eq.get("exact_hypergeom_p_lt_1e_3"), f"{lname}.eq.lt1e3") == (p_eq < Fraction(1, 1000)), f"{lname} eq p flag")

        # Dark & kurtosis consistency
        _, dk_bg_n = _require_pair_int(lj.get("dark_layer_bg"), f"{lname}.dark_layer_bg")
        _, dk_sel_n = _require_pair_int(lj.get("dark_selected"), f"{lname}.dark_selected")
        must(dk_bg_n == q08_width and dk_sel_n == sel_cnt, f"{lname} dark counts dimension mismatch")
        sel_k = _require_int(lj.get("selected_mean_kurt_x100"), f"{lname}.sel_kurt", lo=1)
        non_k = _require_int(lj.get("nonselected_mean_kurt_x100"), f"{lname}.non_kurt", lo=1)
        must(_require_bool(lj.get("kurtosis_ratio_gt_2_5x"), f"{lname}.kurt_flag") == (sel_k * 10 > non_k * 25), f"{lname} kurtosis ratio flag")

        # L15 seeded permutation null cross-check against L15 counts
        if lname == "L15":
            pnull = lj.get("permutation_null_1000_l15")
            must(type(pnull) is dict, "L15 permutation_null_1000_l15 missing")
            must(_require_int(pnull.get("population"), "pnull.pop") == all_bg_n, "pnull pop mismatch")
            must(_require_int(pnull.get("target_count"), "pnull.targets") == all_bg_k, "pnull target mismatch")
            must(_require_int(pnull.get("sample_size"), "pnull.sample") == all_sel_n, "pnull sample mismatch")
            must(_require_int(pnull.get("observed_hits"), "pnull.obs") == all_sel_k, "pnull obs mismatch")
            n_perms = _require_int(pnull.get("n_perms"), "pnull.n_perms", lo=1)
            exc_cnt = _require_int(pnull.get("exceed_count"), "pnull.exceed", lo=0)
            must(Fraction(exc_cnt + 1, n_perms + 1) == Fraction(pnull["empirical_p_exact"]), "pnull empirical_p mismatch")

    must(layer_block_lt_001["L15"] is True and layer_strat_p["L15"] < Fraction(1, 10**9), "L15 stratified check failed")
    must(layer_block_lt_001["L19"] is False and layer_strat_p["L19"] < Fraction(2, 1000), "L19 stratified check failed")

    # Top-40 L_inf outliers derived directly from p3 & cross-checked with L15+L19 top-20 counts
    top40 = p3.get("top40_linf_outliers_in_col_lt_512")
    must(type(top40) is dict, "missing top40_linf_outliers_in_col_lt_512")
    t40_bg_k, t40_bg_n = _require_pair_int(top40.get("background"), "top40.background")
    t40_obs, t40_draw = _require_pair_int(top40.get("count"), "top40.count")
    must(t40_bg_n == 2 * q08_width and t40_draw == 40 and t40_obs == top20_sum, "top40 counts inconsistent with L15+L19")
    p_top40 = indep_hypergeom_tail(t40_bg_n, t40_bg_k, t40_draw, t40_obs)
    must(_require_bool(top40.get("exact_hypergeom_p_lt_1e_14"), "top40.flag") == (p_top40 < Fraction(1, 10**14)), "top40 p flag mismatch")
    must(p_top40 < Fraction(1, 10**14), "top40 p not < 1e-14")

    # 24-layer column anisotropy derived directly from p3
    aniso = p3.get("q08_24layer_column_anisotropy")
    must(type(aniso) is dict, "missing q08_24layer_column_anisotropy")
    mid_str = aniso.get("middle_layers_L02_to_L22_hi_gt_lo")
    must(type(mid_str) is str and "/" in mid_str, "middle_layers ratio string missing")
    mid_parts = mid_str.split("/")
    must(len(mid_parts) == 2 and all(x.isascii() and x.isdecimal() for x in mid_parts), "invalid middle_layers ratio")
    k_mid, n_mid = int(mid_parts[0]), int(mid_parts[1])
    must(n_mid == 21 and k_mid == 21, f"unexpected middle_layers_L02_to_L22_hi_gt_lo={mid_str!r}")
    p_sign_mid = indep_binom_two_sided(n_mid, k_mid)
    must(p_sign_mid == Fraction(aniso["exact_binomial_sign_p"]), "anisotropy middle sign p mismatch")
    p_sign_21_24 = indep_binom_two_sided(q08_layers, k_mid)
    must(p_sign_21_24 == Fraction(2325, 8_388_608), "21/24 all-layer sign p mismatch")

    # Cross-layer residual axis derived directly from p3
    axis = p3.get("cross_layer_residual_axis_u15_u19")
    must(type(axis) is dict, "missing cross_layer_residual_axis_u15_u19")
    t5_15 = axis.get("top5_dims_L15")
    t5_19 = axis.get("top5_dims_L19")
    must(type(t5_15) is list and len(t5_15) == 5 and all(type(x) is int for x in t5_15), "invalid top5_dims_L15")
    must(type(t5_19) is list and len(t5_19) == 5 and all(type(x) is int for x in t5_19), "invalid top5_dims_L19")
    must(_require_bool(axis.get("identical_top5_set"), "identical_top5_set") == (set(t5_15) == set(t5_19)), "top5 set flag mismatch")
    must(set(t5_15) == set(t5_19) and 0 in t5_15, "L15/L19 top5 residual dims mismatch")
    must(_require_bool(axis.get("dim0_is_rank1_row_linf_both_layers"), "dim0_flag") is True, "dim0_is_rank1 flag false")
    must(_require_bool(axis.get("dot_positive"), "dot_positive") is True, "residual axis dot_positive false")
    must(_require_int(axis.get("cos_sq_x10000"), "cos_sq_x10000", lo=5001) <= 10000, "residual axis cos_sq out of range")

    # Paired McNemar derived directly from p3
    mcn = p3.get("exp8_paired_mcnemar_1152_problems")
    must(type(mcn) is dict, "missing exp8_paired_mcnemar_1152_problems")
    rescues = _require_int(mcn.get("rescues"), "mcnemar.rescues", lo=1)
    losses = _require_int(mcn.get("losses"), "mcnemar.losses", lo=0)
    p_mcn = indep_binom_two_sided(rescues + losses, min(rescues, losses))
    must(_require_bool(mcn.get("exact_mcnemar_p_lt_1e_150"), "mcnemar.flag") == (p_mcn < Fraction(1, 10**150)), "McNemar p flag mismatch")
    must(p_mcn < Fraction(1, 10**150), "McNemar p not < 1e-150")

    # Part 4: Gemma 4 E4B Layer-10 exploratory summary derived directly from p4
    p4 = cert.get("part4_gemma4_e4b_layer10_siliquant_bridge")
    must(type(p4) is dict, "missing part4_gemma4_e4b_layer10_siliquant_bridge")
    must(_require_int(p4.get("layers"), "g4.layers") == per13["g4e4"]["layers"], "g4e4 layers mismatch")
    must(_require_int(p4.get("width"), "g4.width") == per13["g4e4"]["width"], "g4e4 width mismatch")
    must(_require_int(p4.get("total_neurons"), "g4.neurons") == per13["g4e4"]["neurons"], "g4e4 neurons mismatch")
    top5_out = p4.get("top5_layers_by_mean_out")
    must(type(top5_out) is list and len(top5_out) == 5, "top5_layers_by_mean_out invalid")
    for idx in range(4):
        must(
            _require_int(top5_out[idx]["mean_out_x100000"], "mean_out")
            > _require_int(top5_out[idx + 1]["mean_out_x100000"], "mean_out_next"),
            "top5_layers_by_mean_out not strictly descending",
        )
    must(top5_out[0]["layer"] == _require_int(p4.get("rank1_mean_out_layer"), "rank1") == 10, "L10 rank1 mismatch")
    must(top5_out[1]["layer"] == _require_int(p4.get("rank2_mean_out_layer"), "rank2") == 9, "L09 rank2 mismatch")
    must(_require_int(p4.get("first_code_gt_prose_layer"), "first_code_gt_prose") == 10, "first_code_gt_prose")
    top6_grp = p4.get("layer10_top6_groups")
    must(type(top6_grp) is list and len(top6_grp) == 6, "layer10_top6_groups invalid")
    dark_hubs = sum(1 for g in top6_grp if _require_bool(g.get("is_dark_hub"), "is_dark_hub"))
    must(dark_hubs == _require_int(p4.get("layer10_top6_dark_hub_count"), "dark_hub_count"), "L10 dark hub count mismatch")
    must(Fraction(1560, 1560 + 182) == Fraction(60, 67) == Fraction(p4["siliquant_cross_reference"]["l10_kl_share_exact"]), "60/67")

    # Part 5: Flint Goalkeeper exact Fisher + Bonferroni-6 derived directly from p5 counts
    p5 = cert.get("part5_flint_goalkeeper_hardened_recount")
    must(type(p5) is dict, "missing part5_flint_goalkeeper_hardened_recount")

    def _check_flint_split(rec: Any, label: str) -> Fraction:
        must(type(rec) is dict, f"{label} must be dict")
        cnts = rec.get("counts")
        must(type(cnts) is dict and set(cnts.keys()) == {"target", "baseline", "opposite_pole", "matched_neuron"}, f"{label} cond keys")
        t_h, t_n = _require_pair_int(cnts["target"], f"{label}.target")
        c_h = 0
        c_n = 0
        for cname in ("baseline", "opposite_pole", "matched_neuron"):
            h_i, n_i = _require_pair_int(cnts[cname], f"{label}.{cname}")
            must(n_i == t_n, f"{label}.{cname} unbalanced")
            c_h += h_i
            c_n += n_i
        must(t_h == _require_int(rec.get("target_hits"), f"{label}.target_hits"), f"{label} target_hits mismatch")
        must(t_n == _require_int(rec.get("target_total"), f"{label}.target_total"), f"{label} target_total mismatch")
        must(c_h == _require_int(rec.get("control_hits"), f"{label}.control_hits"), f"{label} control_hits mismatch")
        must(c_n == _require_int(rec.get("control_total"), f"{label}.control_total"), f"{label} control_total mismatch")
        must(_require_int(rec.get("prompt_leaks"), f"{label}.prompt_leaks") == 0, f"{label} prompt_leaks != 0")
        p_val = indep_hypergeom_tail(t_n + c_n, t_h + c_h, t_n, t_h)
        must(p_val == Fraction(rec["exact_fisher_p"]), f"{label} exact_fisher_p mismatch")
        return p_val

    p_fix = _check_flint_split(p5.get("fixed_replication"), "fixed_replication")
    p_ood = _check_flint_split(p5.get("out_of_domain_preregistered"), "out_of_domain_preregistered")
    n_runs = _require_int(p5.get("search_ledger_total_runs"), "search_ledger_total_runs", lo=1)
    bonf = Fraction(n_runs) * p_fix
    must(bonf == Fraction(p5["fixed_bonferroni_6_exact"]), "Flint Bonf-6 mismatch")
    must(_require_bool(p5.get("fixed_bonferroni_6_lt_1e_12"), "bonf_flag") == (bonf < Fraction(1, 10**12)), "Flint Bonf flag mismatch")
    must(_require_bool(p5.get("ood_p_lt_1e_14"), "ood_flag") == (p_ood < Fraction(1, 10**14)), "Flint OOD flag mismatch")

    return {
        "total_neurons_verified": total_neurons,
        "q08_all_24_layers_sign_p_exact": str(p_sign_21_24),
        "l15_both_blocks_lt_0_01": layer_block_lt_001["L15"],
        "l15_strat_p_lt_1e_9": layer_strat_p["L15"] < Fraction(1, 10**9),
        "l19_both_blocks_lt_0_01": layer_block_lt_001["L19"],
        "l19_strat_p_lt_2e_3": layer_strat_p["L19"] < Fraction(2, 1000),
    }


def verify_standalone(
    cert_path: Path,
    *,
    expected_sha256: str | None = PINNED_CANONICAL_SHA256,
) -> dict[str, Any]:
    must(cert_path.is_file() and not cert_path.is_symlink(), f"Missing or symlink certificate: {cert_path}")
    cert = json.loads(cert_path.read_text(encoding="utf-8"))
    check_no_floats(cert)

    stored_sha = _require_hex_digest(cert.get("canonical_sha256"), 64, "canonical_sha256")
    body = {k: v for k, v in cert.items() if k != "canonical_sha256"}
    recomputed = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    must(hmac.compare_digest(recomputed, stored_sha), "canonical_sha256 mismatch")

    # Run full semantic recomputation directly from the supplied certificate BEFORE checking pin,
    # so even unpinned semantic verification rejects any destroyed/mutated certificate.
    sem = verify_certificate_semantics(cert)

    if expected_sha256 is not None:
        pinned = _require_hex_digest(expected_sha256, 64, "expected_sha256")
        must(
            hmac.compare_digest(recomputed, pinned),
            f"canonical_sha256 {recomputed} does not match pinned identity {pinned}",
        )

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
        "pinned_identity_verified": expected_sha256 is not None,
        "total_neurons_verified": sem["total_neurons_verified"],
        "q08_all_24_layers_sign_p_exact": sem["q08_all_24_layers_sign_p_exact"],
        "l15_both_blocks_lt_0_01": sem["l15_both_blocks_lt_0_01"],
        "l15_strat_p_lt_1e_9": sem["l15_strat_p_lt_1e_9"],
        "l19_both_blocks_lt_0_01": sem["l19_both_blocks_lt_0_01"],
        "l19_strat_p_lt_2e_3": sem["l19_strat_p_lt_2e_3"],
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
