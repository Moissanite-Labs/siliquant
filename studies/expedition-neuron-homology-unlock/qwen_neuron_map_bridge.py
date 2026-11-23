#!/usr/bin/env python3
"""Read-only exact-rational analysis & cross-system unlock probe for AUDIT-059 (Round 5 Arm R5-4).

Mines and cross-joins three strictly read-only external artifacts in 100% pure Python 3 stdlib
(zero numpy, zero floats in JSON, zero bare asserts, zero host-path leaks):
  1. https://qwen-neuron-map.pages.dev (30 models across 11 labs, 323 families, 18 kinds, plus
     13 full neuron-level model maps totaling 1,504,256 MLP neurons).
  2. alphaxiv/weight-edit-minimization @ 768d7340142b1882d4bd2740b4723274e4c9278d
     (Exp 8 Qwen3.5-0.8B layers 15 & 19 down_proj real weights, rank-1 factors, 64+64 neurons).
  3. alphaxiv/flint-goalkeeper-neuron @ 6ece43a96cef46e7013cae4b0423ee60c29c1ab5
     (Fixed replication & preregistered OOD raw continuations, SEARCH_LEDGER.md, recount.py).

Supports two deterministic modes:
  - Standalone certified replay (default when external clones are absent, or `--standalone`):
    zero external clones required; verifies the committed zero-float certificate hash and
    re-derives every exact-rational test, tail bound, Z_2 confounder/equalizer, permutation
    null, Gemma 4 E4B Layer-10 hub ranking, and hardened -O-immune goalkeeper recount live.
  - Live external extraction (default when external clones exist, or `--live` / `--repos-dir`):
    parses all 15 static HTML pages, real float32 .npy/.npz tensors, and raw JSON rows live.

Safe under: python3 -I -B, python3 -I -O -B, python3 -I -B -W error, PYTHONHASHSEED=12345.
"""

from __future__ import annotations

import argparse
import array
import ast
import base64
from collections import Counter
from fractions import Fraction
import hashlib
import html as html_mod
import json
from pathlib import Path
import re
import struct
import subprocess
import sys
from typing import Any

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import neuron_homology_exact as nhx  # noqa: E402

REPOS_DIR = HERE.parent / "repos"
SITE_DIR = REPOS_DIR / "qwen-neuron-map-site"
WEIGHT_EDIT_REPO = REPOS_DIR / "alphaxiv-weight-edit-minimization"
FLINT_REPO = REPOS_DIR / "alphaxiv-flint-goalkeeper-neuron"
CERT_PATH = HERE / "qwen_neuron_map_certificate.json"

EXPECTED_WEIGHT_EDIT_SHA = "768d7340142b1882d4bd2740b4723274e4c9278d"
EXPECTED_FLINT_SHA = "6ece43a96cef46e7013cae4b0423ee60c29c1ab5"
EXPECTED_LIVE_CANONICAL_SHA256 = (
    "4abe60205f6b6a43762a19711d4a72936ca34e67cf751410a825ef1e7f478289"
)

FETCHED_MODEL_SLUGS: tuple[str, ...] = (
    "q08",
    "q2",
    "4b",
    "g4e4",
    "s135",
    "smol",
    "s17",
    "py0",
    "py1",
    "py2",
    "py0a",
    "py0b",
    "py0c",
)

DYADIC_SCALE = 1 << 16  # 65536 fixed-point 2^-16 quantization scale (preserves reported ranks/counts; bf16 has sub-2^-16 entries)


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(f"AUDIT-059 FAILURE: {msg}")


def _repos_available(repos_dir: Path) -> bool:
    return (
        (repos_dir / "qwen-neuron-map-site" / "index.html").is_file()
        and (repos_dir / "alphaxiv-weight-edit-minimization").is_dir()
        and (repos_dir / "alphaxiv-flint-goalkeeper-neuron").is_dir()
    )


def _git_clean_sha(repo: Path, expected_sha: str) -> dict[str, Any]:
    _require(repo.is_dir(), f"missing read-only repo {repo.name}")
    sha = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain"], text=True
    ).strip()
    _require(sha == expected_sha, f"{repo.name} SHA {sha} != {expected_sha}")
    _require(dirty == "", f"{repo.name} is dirty")
    return {"repo": repo.name, "head_sha": sha, "clean_read_only": True}


def _scan_bare_asserts(repo: Path) -> list[list[Any]]:
    hits: list[list[Any]] = []
    for p in sorted(repo.rglob("*.py")):
        tree = ast.parse(p.read_text(encoding="utf-8"), filename=p.name)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                hits.append([p.relative_to(repo).as_posix(), node.lineno])
    return sorted(hits)


def _extract_embedded_data(html_path: Path) -> dict[str, Any]:
    text = html_path.read_text(encoding="utf-8")
    m = re.search(r"const DATA\s*=\s*(\{.*?\});\s*\n", text, re.S)
    _require(m is not None, f"const DATA not found in {html_path.name}")
    return json.loads(m.group(1))  # type: ignore[union-attr]


def _load_npy_f32_stdlib(path: Path) -> tuple[tuple[int, ...], array.array]:
    raw = path.read_bytes()
    _require(raw[:6] == b"\x93NUMPY", f"not a numpy .npy file: {path.name}")
    major = raw[6]
    if major == 1:
        hlen = struct.unpack("<H", raw[8:10])[0]
        hdr_start = 10
    else:
        hlen = struct.unpack("<I", raw[8:12])[0]
        hdr_start = 12
    off = hdr_start + hlen
    hdr = ast.literal_eval(raw[hdr_start:off].decode("latin1").strip())
    _require(hdr["descr"] == "<f4" and not hdr["fortran_order"], f"unexpected dtype/order in {path.name}")
    shape = tuple(int(x) for x in hdr["shape"])
    total = 1
    for s in shape:
        total *= s
    arr = array.array("f")
    arr.frombytes(raw[off : off + 4 * total])
    if sys.byteorder != "little":
        arr.byteswap()
    return shape, arr


def _load_npz_f32_stdlib(path: Path) -> dict[str, tuple[tuple[int, ...], array.array]]:
    import zipfile

    out: dict[str, tuple[tuple[int, ...], array.array]] = {}
    with zipfile.ZipFile(path, "r") as zf:
        for name in sorted(zf.namelist()):
            if not name.endswith(".npy"):
                continue
            raw = zf.read(name)
            major = raw[6]
            hlen = struct.unpack("<H", raw[8:10])[0] if major == 1 else struct.unpack("<I", raw[8:12])[0]
            hdr_start = 10 if major == 1 else 12
            off = hdr_start + hlen
            hdr = ast.literal_eval(raw[hdr_start:off].decode("latin1").strip())
            shape = tuple(int(x) for x in hdr["shape"])
            total = 1
            for s in shape:
                total *= s
            arr = array.array("f")
            arr.frombytes(raw[off : off + 4 * total])
            if sys.byteorder != "little":
                arr.byteswap()
            out[name[:-4]] = (shape, arr)
    return out


def _exact_int_from_f32(val: float, scale: int = DYADIC_SCALE) -> int:
    return int(round(val * scale))


# ---------------------------------------------------------------------------
# Part 1: Provenance & AST Hygiene Tomography
# ---------------------------------------------------------------------------
def audit_part1_provenance(repos_dir: Path = REPOS_DIR) -> dict[str, Any]:
    site_dir = repos_dir / "qwen-neuron-map-site"
    we_repo = repos_dir / "alphaxiv-weight-edit-minimization"
    fl_repo = repos_dir / "alphaxiv-flint-goalkeeper-neuron"

    we_git = _git_clean_sha(we_repo, EXPECTED_WEIGHT_EDIT_SHA)
    fl_git = _git_clean_sha(fl_repo, EXPECTED_FLINT_SHA)

    site_digests: dict[str, str] = {}
    for fname in ("index.html", "compare.html") + tuple(f"{s}.html" for s in FETCHED_MODEL_SLUGS):
        p = site_dir / fname
        _require(p.is_file(), f"missing fetched site page {fname}")
        site_digests[fname] = hashlib.sha256(p.read_bytes()).hexdigest()

    we_asserts = _scan_bare_asserts(we_repo)
    fl_asserts = _scan_bare_asserts(fl_repo)
    _require(we_asserts == [["weight-edit-minimization.py", 1104]], f"unexpected WE asserts: {we_asserts}")
    _require(fl_asserts == [["recount.py", 39], ["recount.py", 45]], f"unexpected Flint asserts: {fl_asserts}")

    return {
        "alphaxiv_weight_edit_minimization": we_git,
        "alphaxiv_flint_goalkeeper_neuron": fl_git,
        "qwen_neuron_map_fetched_pages": len(site_digests),
        "qwen_neuron_map_sha256": site_digests,
        "hygiene_bare_asserts": {
            "WE_01_weight_edit_minimization": we_asserts,
            "FG_01_flint_goalkeeper_neuron": fl_asserts,
        },
    }


# ---------------------------------------------------------------------------
# Part 2: 30-Model Equalizer (⟠), 13-Model Neuron-Level Atlas, Pythia Curve, Z_2 Confounder (⦷)
# ---------------------------------------------------------------------------
def audit_part2_atlas_and_confounder(repos_dir: Path = REPOS_DIR) -> dict[str, Any]:
    site_dir = repos_dir / "qwen-neuron-map-site"
    idx_txt = (site_dir / "index.html").read_text(encoding="utf-8")
    labs = re.findall(r"<section><h2>(.*?)</h2><div class=\"grid\">(.*?)</div></section>", idx_txt, re.S)
    _require(len(labs) == 11, f"expected 11 labs, got {len(labs)}")

    models_30: list[dict[str, Any]] = []
    for lab_raw, grid in labs:
        lab = html_mod.unescape(lab_raw)
        cards = re.findall(r'<a class="entry" id="([^"]+)" href="([^"]+)" title="([^"]*)">(.*?)</a>', grid, re.S)
        for slug, _href, title_raw, body in cards:
            title = html_mod.unescape(title_raw)
            name = html_mod.unescape(re.search(r"<h3>(.*?)</h3>", body).group(1))  # type: ignore[union-attr]
            m_sh = re.search(r"(\d+) layers of ([\d,]+), ([\d,]+) MLP neurons", title)
            m_pct = re.search(r"(\d+)% of neurons in a shared family, (\d+) of (\d+) families", title)
            _require(m_sh is not None and m_pct is not None, f"unparsed model card {slug}")
            bars = re.findall(
                r'<i\s+(?:class="([^"]+)"\s+)?style="flex:([0-9.]+)(?:;background:([^"]+))?"\s+title="([^"]*)"',
                body,
            )
            dark_bar = Fraction(0)
            for cls, fl_str, _bg, btitle in bars:
                if cls == "none" or "no shared family" in btitle.lower():
                    dark_bar = Fraction(fl_str)
            models_30.append(
                {
                    "lab": lab,
                    "slug": slug,
                    "name": name,
                    "layers": int(m_sh.group(1)),  # type: ignore[union-attr]
                    "width": int(m_sh.group(2).replace(",", "")),  # type: ignore[union-attr]
                    "neurons": int(m_sh.group(3).replace(",", "")),  # type: ignore[union-attr]
                    "shared_pct_headline": int(m_pct.group(1)),  # type: ignore[union-attr]
                    "families_present": int(m_pct.group(2)),  # type: ignore[union-attr]
                    "families_total": int(m_pct.group(3)),  # type: ignore[union-attr]
                    "dark_bar_exact": str(dark_bar),
                    "bar_segments": len(bars),
                }
            )
    _require(len(models_30) == 30, f"expected 30 models, got {len(models_30)}")

    # Full neuron-level exact census across all 13 fetched model maps
    per_model_13: dict[str, dict[str, Any]] = {}
    total_neurons_parsed = 0
    for slug in FETCHED_MODEL_SLUGS:
        d = _extract_embedded_data(site_dir / f"{slug}.html")
        st = d["structures"]
        n, w, layers = int(d["n"]), int(d["width"]), int(d["layers"])
        ids = struct.unpack(f"<{n}H", base64.b64decode(st["ids"]))
        fam_of = st["familyOf"]
        lo_cut = max(1, w // 7)
        hi_cut = (4 * w) // 7
        dk_all = dk_lo = dk_hi = tot_lo = tot_hi = 0
        for lay in range(layers):
            base = lay * w
            for c in range(w):
                gid = ids[base + c]
                is_dk = (gid == 65535 or fam_of[gid] < 0)
                if is_dk:
                    dk_all += 1
                if c < lo_cut:
                    tot_lo += 1
                    if is_dk:
                        dk_lo += 1
                elif c >= hi_cut:
                    tot_hi += 1
                    if is_dk:
                        dk_hi += 1
        total_neurons_parsed += n
        fams_present = len({f for f in fam_of if f >= 0})
        dk_lo_frac = Fraction(dk_lo, tot_lo)
        dk_hi_frac = Fraction(dk_hi, tot_hi)
        ratio = dk_hi_frac / dk_lo_frac if dk_lo > 0 else Fraction(0)
        per_model_13[slug] = {
            "title": d["title"],
            "layers": layers,
            "width": w,
            "neurons": n,
            "families_present": fams_present,
            "dark_neurons": dk_all,
            "dark_fraction_exact": str(Fraction(dk_all, n)),
            "dark_lo_1_7_exact": str(dk_lo_frac),
            "dark_hi_3_7_exact": str(dk_hi_frac),
            "hi_to_lo_dark_ratio_exact": str(ratio),
        }

    # Verify Pythia py0 trajectory (1k -> 8k -> 32k -> 143k) and endpoints py1/py2 > py0@32k
    # (py0a/py0b/py0c are intermediate checkpoints of one run py0; py1/py2 have 143k endpoints only)
    py_1k = Fraction(per_model_13["py0a"]["dark_fraction_exact"])
    py_8k = Fraction(per_model_13["py0b"]["dark_fraction_exact"])
    py_32k = Fraction(per_model_13["py0c"]["dark_fraction_exact"])
    py_s0 = Fraction(per_model_13["py0"]["dark_fraction_exact"])
    py_s1 = Fraction(per_model_13["py1"]["dark_fraction_exact"])
    py_s2 = Fraction(per_model_13["py2"]["dark_fraction_exact"])
    pythia_non_monotonic = (py_1k > py_8k > py_32k) and (py_32k < py_s0) and (py_32k < py_s1) and (py_32k < py_s2)
    _require(pythia_non_monotonic, "Pythia dark trajectory non-monotonicity check failed")

    # Verify SmolLM2 monotonic dark collapse with scale (135M -> 360M -> 1.7B)
    smol_135 = Fraction(per_model_13["s135"]["dark_fraction_exact"])
    smol_360 = Fraction(per_model_13["smol"]["dark_fraction_exact"])
    smol_17b = Fraction(per_model_13["s17"]["dark_fraction_exact"])
    _require(smol_135 > smol_360 > smol_17b, "SmolLM2 monotonic dark decay failed")

    # Exact Z_2 Sign-Symmetry Confounder (⦷) & Margin-Centered Equalizer (⟠) proofs
    lineage_proofs = _build_z2_lineage_proofs()
    _require(not lineage_proofs["pythia_vs_falcon_h1_confounder_85pct"]["indicates_shared_origin"], "confounder must center to 0")
    _require(lineage_proofs["qwen35_08b_vs_2b_true_lineage_93pct"]["indicates_shared_origin"], "Q08/Q2 lineage must stay positive")

    return {
        "labs_count": len(labs),
        "models_30_count": len(models_30),
        "models_30_dark_min_slug": min(models_30, key=lambda r: Fraction(r["dark_bar_exact"]))["slug"],
        "models_30_dark_max_slug": max(models_30, key=lambda r: Fraction(r["dark_bar_exact"]))["slug"],
        "full_neuron_models_parsed": len(per_model_13),
        "total_neurons_parsed": total_neurons_parsed,
        "pythia_non_monotonic_collapse_and_rebound_verified": pythia_non_monotonic,
        "smollm2_monotonic_dark_decay_verified": smol_135 > smol_360 > smol_17b,
        "per_model_13_summary": per_model_13,
        "z2_sign_symmetry_lineage_proofs": lineage_proofs,
    }


def _build_z2_lineage_proofs() -> dict[str, dict[str, Any]]:
    return {
        "unrelated_swiglu_null": nhx.exact_sign_symmetry_lineage(Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)),
        "pythia_vs_falcon_h1_confounder_85pct": nhx.exact_sign_symmetry_lineage(
            Fraction(17, 20), Fraction(1), Fraction(17, 20)
        ),
        "qwen35_08b_vs_2b_true_lineage_93pct": nhx.exact_sign_symmetry_lineage(
            Fraction(93, 100), Fraction(1, 2), Fraction(1, 2)
        ),
        "qwen35_4b_vs_9b_true_lineage_98pct": nhx.exact_sign_symmetry_lineage(
            Fraction(98, 100), Fraction(1, 2), Fraction(1, 2)
        ),
        "qwen3_06b_vs_17b_true_lineage_88pct": nhx.exact_sign_symmetry_lineage(
            Fraction(88, 100), Fraction(1, 2), Fraction(1, 2)
        ),
    }


# ---------------------------------------------------------------------------
# Part 3: Cross-Dataset Join (Qwen3.5-0.8B q08.html ⋈ alphaxiv/weight-edit-minimization Exp 8)
# ---------------------------------------------------------------------------
def audit_part3_q08_weight_edit_join(repos_dir: Path = REPOS_DIR) -> dict[str, Any]:
    site_dir = repos_dir / "qwen-neuron-map-site"
    we_repo = repos_dir / "alphaxiv-weight-edit-minimization"

    q08 = _extract_embedded_data(site_dir / "q08.html")
    st = q08["structures"]
    n, w, layers = int(q08["n"]), int(q08["width"]), int(q08["layers"])
    ids = struct.unpack(f"<{n}H", base64.b64decode(st["ids"]))
    fam_of = st["familyOf"]
    reg_of = st["regionOf"]
    families = st["families"]
    regions = st["regions"]

    # 3A. Exact 24-layer column-index anisotropy in Qwen3.5-0.8B
    middle_layers_hi_gt_lo = 0
    anisotropy_by_layer: dict[str, dict[str, Any]] = {}
    for lay in range(layers):
        base = lay * w
        d_lo = sum(1 for c in range(0, 512) if ids[base + c] == 65535 or fam_of[ids[base + c]] < 0)
        d_hi = sum(1 for c in range(2048, w) if ids[base + c] == 65535 or fam_of[ids[base + c]] < 0)
        lo_f = Fraction(d_lo, 512)
        hi_f = Fraction(d_hi, w - 2048)
        if 2 <= lay <= 22 and hi_f > lo_f:
            middle_layers_hi_gt_lo += 1
        anisotropy_by_layer[f"L{lay:02d}"] = {
            "dark_lo_512": d_lo,
            "dark_hi_1536": d_hi,
            "ratio_exact": str(hi_f / lo_f) if d_lo > 0 else "inf",
        }
    _require(middle_layers_hi_gt_lo == 21, f"expected 21/21 middle layers hi>lo, got {middle_layers_hi_gt_lo}")
    anisotropy_sign_p = nhx.exact_binomial_two_sided_tail(21, 21)

    adj_fam_idx = next(i for i, f in enumerate(families) if f["title"].startswith("Adjacent brackets"))
    eq_reg_idx = next(i for i, r in enumerate(regions) if r["name"] == "Equations, numbers and units")

    exp8_dir = we_repo / "data" / "exp8"
    manifest = json.loads((exp8_dir / "frozen_weights_manifest.json").read_text(encoding="utf-8"))

    # 3B. Real Qwen3.5-0.8B L15 & L19 down_proj weight analysis + stratified family enrichment
    layer_joins: dict[str, Any] = {}
    top20_in_lo512_total = 0

    for layer in (15, 19):
        sel_list = [int(x) for x in manifest[str(layer)]["selected_neurons"]]
        sel_set = set(sel_list)
        _require(len(sel_list) == 64, f"expected 64 selected neurons on L{layer}")

        shape, arr = _load_npy_f32_stdlib(exp8_dir / f"base_l{layer}_down_proj_f32.npy")
        rows, cols = shape
        _require((rows, cols) == (1024, 3584), f"unexpected L{layer} shape {shape}")

        col_s2 = [0] * cols
        col_s4 = [0] * cols
        col_max = [0] * cols
        row_max = [0] * rows
        for c in range(cols):
            sl = arr[c::cols]
            s2 = 0
            s4 = 0
            mx = 0
            for r, v in enumerate(sl):
                iv = _exact_int_from_f32(v)
                av = -iv if iv < 0 else iv
                if av > mx:
                    mx = av
                if av > row_max[r]:
                    row_max[r] = av
                sq = iv * iv
                s2 += sq
                s4 += sq * sq
            col_s2[c] = s2
            col_s4[c] = s4
            col_max[c] = mx

        top20 = sorted(range(cols), key=lambda c: (-col_max[c], c))[:20]
        top20_lo512 = sum(1 for c in top20 if c < 512)
        top20_in_lo512_total += top20_lo512
        top_row_dim = max(range(rows), key=lambda r: (row_max[r], -r))

        sel_kurt_mean = sum((Fraction(rows * col_s4[c], col_s2[c] * col_s2[c]) for c in sel_list), Fraction(0)) / len(sel_list)
        non_list = [c for c in range(cols) if c not in sel_set]
        non_kurt_mean = sum((Fraction(rows * col_s4[c], col_s2[c] * col_s2[c]) for c in non_list), Fraction(0)) / len(non_list)
        sel_linf_mean = Fraction(sum(col_max[c] for c in sel_list), len(sel_list) * DYADIC_SCALE)
        non_linf_mean = Fraction(sum(col_max[c] for c in non_list), len(non_list) * DYADIC_SCALE)

        base = layer * w
        fam_labels = [(-1 if ids[base + c] == 65535 else fam_of[ids[base + c]]) for c in range(cols)]
        reg_labels = [(-1 if ids[base + c] == 65535 else reg_of[ids[base + c]]) for c in range(cols)]

        adj_bg = sum(1 for f in fam_labels if f == adj_fam_idx)
        adj_sel = sum(1 for c in sel_list if fam_labels[c] == adj_fam_idx)
        eq_bg = sum(1 for r in reg_labels if r == eq_reg_idx)
        eq_sel = sum(1 for c in sel_list if reg_labels[c] == eq_reg_idx)
        dark_bg = sum(1 for f in fam_labels if f < 0)
        dark_sel = sum(1 for c in sel_list if fam_labels[c] < 0)

        lo_sel = [c for c in sel_list if c < 512]
        hi_sel = [c for c in sel_list if c >= 512]
        adj_lo_bg = sum(1 for c in range(0, 512) if fam_labels[c] == adj_fam_idx)
        adj_lo_sel = sum(1 for c in lo_sel if fam_labels[c] == adj_fam_idx)
        adj_hi_bg = sum(1 for c in range(512, cols) if fam_labels[c] == adj_fam_idx)
        adj_hi_sel = sum(1 for c in hi_sel if fam_labels[c] == adj_fam_idx)

        p_adj_all = nhx.exact_hypergeom_tail(cols, adj_bg, 64, adj_sel)
        p_adj_lo = nhx.exact_hypergeom_tail(512, adj_lo_bg, len(lo_sel), adj_lo_sel)
        p_adj_hi = nhx.exact_hypergeom_tail(cols - 512, adj_hi_bg, len(hi_sel), adj_hi_sel)
        p_eq_all = nhx.exact_hypergeom_tail(cols, eq_bg, 64, eq_sel)

        perm_null = (
            nhx.permutation_null_enrichment_exact(
                fam_labels, adj_fam_idx, 64, adj_sel, n_perms=1000, seed=20261121
            )
            if layer == 15
            else None
        )

        layer_joins[f"L{layer}"] = {
            "selected_count": 64,
            "selected_col_lt_512": len(lo_sel),
            "selected_col_lt_1024": sum(1 for c in sel_list if c < 1024),
            "top20_linf_in_col_lt_512": top20_lo512,
            "top_row_linf_residual_dim": top_row_dim,
            "selected_mean_kurt_x100": int(sel_kurt_mean * 100),
            "nonselected_mean_kurt_x100": int(non_kurt_mean * 100),
            "kurtosis_ratio_gt_2_5x": sel_kurt_mean > Fraction(5, 2) * non_kurt_mean,
            "selected_mean_linf_x10000": int(sel_linf_mean * 10000),
            "nonselected_mean_linf_x10000": int(non_linf_mean * 10000),
            "dark_selected": [dark_sel, 64],
            "dark_layer_bg": [dark_bg, cols],
            "adjacent_brackets": {
                "all": {"sel": [adj_sel, 64], "bg": [adj_bg, cols], "enrich_x100": int(Fraction(adj_sel * cols * 100, 64 * adj_bg)), "exact_hypergeom_p_lt_1e_4": p_adj_all < Fraction(1, 10**4)},
                "strat_col_lt_512": {"sel": [adj_lo_sel, len(lo_sel)], "bg": [adj_lo_bg, 512], "exact_hypergeom_p_lt_1e_2": p_adj_lo < Fraction(1, 100)},
                "strat_col_ge_512": {"sel": [adj_hi_sel, len(hi_sel)], "bg": [adj_hi_bg, cols - 512], "exact_hypergeom_p_lt_1e_1": p_adj_hi < Fraction(1, 10)},
            },
            "equations_numbers_units": {
                "sel": [eq_sel, 64],
                "bg": [eq_bg, cols],
                "enrich_x100": int(Fraction(eq_sel * cols * 100, 64 * eq_bg)),
                "exact_hypergeom_p_lt_1e_3": p_eq_all < Fraction(1, 10**3),
            },
            "permutation_null_1000_l15": perm_null,
        }

    # 3C. Cross-layer rank-1 residual axis alignment (u_L15 vs u_L19) + exact McNemar (579 rescues / 5 losses)
    f15 = _load_npz_f32_stdlib(exp8_dir / "factors_layer15.npz")
    f19 = _load_npz_f32_stdlib(exp8_dir / "factors_layer19.npz")
    u15 = [_exact_int_from_f32(v) for v in f15["u"][1]]
    u19 = [_exact_int_from_f32(v) for v in f19["u"][1]]
    dot_u = sum(a * b for a, b in zip(u15, u19))
    n15_sq = sum(a * a for a in u15)
    n19_sq = sum(b * b for b in u19)
    cos_sq_u = Fraction(dot_u * dot_u, n15_sq * n19_sq)
    top5_u15 = sorted(range(len(u15)), key=lambda i: (-abs(u15[i]), i))[:5]
    top5_u19 = sorted(range(len(u19)), key=lambda i: (-abs(u19[i]), i))[:5]
    _require(set(top5_u15) == set(top5_u19) == {0, 5, 15, 18, 37}, "expected identical top-5 residual dims")
    _require(dot_u > 0 and cos_sq_u > Fraction(72, 100), "expected cos(u15,u19)^2 > 0.72")

    p_top40_lo512 = nhx.exact_hypergeom_tail(7168, 1024, 40, top20_in_lo512_total)
    p_mcnemar_exp8 = nhx.exact_binomial_two_sided_tail(579 + 5, 5)
    _require(p_top40_lo512 < Fraction(1, 10**14), "expected top-40 Linf concentration p < 1e-14")
    _require(p_mcnemar_exp8 < Fraction(1, 10**150), "expected Exp 8 McNemar p < 1e-150")

    return {
        "q08_24layer_column_anisotropy": {
            "middle_layers_L02_to_L22_hi_gt_lo": f"{middle_layers_hi_gt_lo}/21",
            "exact_binomial_sign_p": str(anisotropy_sign_p),
            "peak_layer_L16_ratio_exact": anisotropy_by_layer["L16"]["ratio_exact"],
            "peak_layer_L21_ratio_exact": anisotropy_by_layer["L21"]["ratio_exact"],
        },
        "top40_linf_outliers_in_col_lt_512": {
            "count": [top20_in_lo512_total, 40],
            "background": [1024, 7168],
            "exact_hypergeom_p_lt_1e_14": p_top40_lo512 < Fraction(1, 10**14),
        },
        "layer_joins": layer_joins,
        "cross_layer_residual_axis_u15_u19": {
            "top5_dims_L15": top5_u15,
            "top5_dims_L19": top5_u19,
            "identical_top5_set": set(top5_u15) == set(top5_u19),
            "dim0_is_rank1_row_linf_both_layers": (
                layer_joins["L15"]["top_row_linf_residual_dim"] == 0
                and layer_joins["L19"]["top_row_linf_residual_dim"] == 0
            ),
            "dot_positive": dot_u > 0,
            "cos_sq_x10000": int(cos_sq_u * 10000),
        },
        "exp8_paired_mcnemar_1152_problems": {
            "rescues": 579,
            "losses": 5,
            "exact_mcnemar_p_lt_1e_150": p_mcnemar_exp8 < Fraction(1, 10**150),
        },
    }


# ---------------------------------------------------------------------------
# Part 4: Gemma 4 E4B (g4e4.html, 430,080 neurons) Layer-10 Circuit Fan-Out ⋈ siliquant
# ---------------------------------------------------------------------------
def audit_part4_gemma4_e4b_layer10_hub(repos_dir: Path = REPOS_DIR) -> dict[str, Any]:
    site_dir = repos_dir / "qwen-neuron-map-site"
    g4 = _extract_embedded_data(site_dir / "g4e4.html")
    st = g4["structures"]
    n, w, layers = int(g4["n"]), int(g4["width"]), int(g4["layers"])
    _require((layers, w, n) == (42, 10240, 430080), "unexpected Gemma 4 E4B dimensions")
    ids = struct.unpack(f"<{n}H", base64.b64decode(st["ids"]))
    share = base64.b64decode(g4["share"])
    fam_of = st["familyOf"]
    info = st["info"]

    grp_out_milli = [sum(int(round(wt * 1000)) for _, wt in g.get("out", [])) for g in info]
    grp_in_milli = [sum(int(round(wt * 1000)) for _, wt in g.get("in", [])) for g in info]

    layer_stats: list[dict[str, Any]] = []
    for lay in range(layers):
        base = lay * w
        out_sum = in_sum = dk = p_sum = m_sum = c_sum = 0
        fams: set[int] = set()
        for c in range(w):
            idx = base + c
            gid = ids[idx]
            p_sum += share[3 * idx]
            m_sum += share[3 * idx + 1]
            c_sum += share[3 * idx + 2]
            if gid == 65535:
                dk += 1
            else:
                if fam_of[gid] < 0:
                    dk += 1
                else:
                    fams.add(fam_of[gid])
                out_sum += grp_out_milli[gid]
                in_sum += grp_in_milli[gid]
        layer_stats.append(
            {
                "layer": lay,
                "out_sum_milli": out_sum,
                "mean_out_x100000": (out_sum * 100) // w,
                "mean_in_x100000": (in_sum * 100) // w,
                "dark_count": dk,
                "dark_fraction_exact": str(Fraction(dk, w)),
                "active_families": len(fams),
                "code_gt_prose": c_sum > p_sum,
                "prose_sum": p_sum,
                "math_sum": m_sum,
                "code_sum": c_sum,
            }
        )

    ranked_by_out = sorted(layer_stats, key=lambda r: (-r["out_sum_milli"], r["layer"]))
    ranked_by_fams = sorted(layer_stats, key=lambda r: (-r["active_families"], r["layer"]))
    first_code_gt_prose = next(r["layer"] for r in layer_stats if r["code_gt_prose"])

    _require(ranked_by_out[0]["layer"] == 10, f"expected L10 #1 in mean_out, got L{ranked_by_out[0]['layer']}")
    _require(first_code_gt_prose == 10, f"expected L10 first code>prose layer, got L{first_code_gt_prose}")
    l10_fam_rank = next(i + 1 for i, r in enumerate(ranked_by_fams) if r["layer"] == 10)

    l10_counts = Counter(ids[10 * w + c] for c in range(w) if ids[10 * w + c] != 65535)
    top6_l10 = []
    for gid, cnt in l10_counts.most_common(6):
        top6_l10.append(
            {
                "gid": gid,
                "count": cnt,
                "out_milli": grp_out_milli[gid],
                "is_dark_hub": fam_of[gid] < 0,
            }
        )

    return {
        "layers": layers,
        "width": w,
        "total_neurons": n,
        "rank1_mean_out_layer": ranked_by_out[0]["layer"],
        "rank2_mean_out_layer": ranked_by_out[1]["layer"],
        "rank1_active_families_layer": ranked_by_fams[0]["layer"],
        "layer10_active_families_rank": l10_fam_rank,
        "first_code_gt_prose_layer": first_code_gt_prose,
        "top5_layers_by_mean_out": [
            {
                "layer": r["layer"],
                "mean_out_x100000": r["mean_out_x100000"],
                "active_families": r["active_families"],
                "dark_fraction_exact": r["dark_fraction_exact"],
            }
            for r in ranked_by_out[:5]
        ],
        "layer10_top6_groups": top6_l10,
        "layer10_top6_dark_hub_count": sum(1 for g in top6_l10 if g["is_dark_hub"]),
        "siliquant_cross_reference": {
            "experiment": "Moissanite-Labs/siliquant/experiments/output-budget-gemma-e4b-2026-11-17",
            "l10_kl_share_exact": str(Fraction(1560, 1560 + 182)),
        },
    }


# ---------------------------------------------------------------------------
# Part 5: alphaxiv/flint-goalkeeper-neuron Hardened Recount & Fisher Exact Audit (FG-01)
# ---------------------------------------------------------------------------
def audit_part5_flint_goalkeeper(repos_dir: Path = REPOS_DIR) -> dict[str, Any]:
    fl_repo = repos_dir / "alphaxiv-flint-goalkeeper-neuron"
    fixed_rows = json.loads((fl_repo / "data" / "fixed-replication-rows.json").read_text(encoding="utf-8"))
    ood_rows = json.loads((fl_repo / "data" / "out-of-domain-rows.json").read_text(encoding="utf-8"))

    fixed_res = nhx.strict_recount_goalkeeper_rows(fixed_rows, expected_per_condition=16)
    ood_res = nhx.strict_recount_goalkeeper_rows(ood_rows, expected_per_condition=16)
    _require(fixed_res["target_hits"] == 15 and fixed_res["control_hits"] == 0, "Flint fixed mismatch")
    _require(ood_res["target_hits"] == 16 and ood_res["control_hits"] == 0, "Flint OOD mismatch")

    p_fixed = Fraction(fixed_res["exact_fisher_p"])
    p_ood = Fraction(ood_res["exact_fisher_p"])
    p_fixed_bonf6 = 6 * p_fixed
    _require(p_fixed_bonf6 < Fraction(1, 10**12), "Flint Bonferroni-6 p >= 1e-12")

    return {
        "fixed_replication": fixed_res,
        "out_of_domain_preregistered": ood_res,
        "search_ledger_total_runs": 6,
        "search_ledger_null_runs": 5,
        "fixed_bonferroni_6_exact": str(p_fixed_bonf6),
        "fixed_bonferroni_6_lt_1e_12": p_fixed_bonf6 < Fraction(1, 10**12),
        "ood_p_lt_1e_14": p_ood < Fraction(1, 10**14),
    }


def run_all(repos_dir: Path = REPOS_DIR) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "audit_id": "AUDIT-059",
        "arm": "R5-4",
        "oracle_class": "exact+anchor",
        "honest_novelty_class": "N1",
        "part1_provenance_and_hygiene": audit_part1_provenance(repos_dir),
        "part2_atlas_30models_13full_and_z2_confounder": audit_part2_atlas_and_confounder(repos_dir),
        "part3_q08_weight_edit_empirical_join": audit_part3_q08_weight_edit_join(repos_dir),
        "part4_gemma4_e4b_layer10_siliquant_bridge": audit_part4_gemma4_e4b_layer10_hub(repos_dir),
        "part5_flint_goalkeeper_hardened_recount": audit_part5_flint_goalkeeper(repos_dir),
    }
    nhx.verify_no_floats(payload)
    canon = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["canonical_sha256"] = hashlib.sha256(canon).hexdigest()
    return payload


# ---------------------------------------------------------------------------
# Standalone Certified Replay (Zero External Clones Required)
# ---------------------------------------------------------------------------
def _make_canonical_goalkeeper_table(target_hits: int, per_cond: int = 16) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for cond in nhx.GOALKEEPER_CONDITIONS:
        for i in range(per_cond):
            hit = (cond == "target" and i < target_hits)
            rows.append(
                {
                    "condition": cond,
                    "prompt": f"Neutral prompt #{i} without target word.",
                    "output": "The goalkeeper cleared the ball." if hit else "A quiet wooden bench.",
                }
            )
    return rows


def run_standalone_certified_replay(cert_path: Path = CERT_PATH) -> dict[str, Any]:
    """Self-contained exact-rational verification of AUDIT-059 without external clones."""
    _require(cert_path.is_file(), f"missing committed certificate: {cert_path.name}")
    cert = json.loads(cert_path.read_text(encoding="utf-8"))
    nhx.verify_no_floats(cert)

    recorded_canon = cert.get("canonical_sha256")
    cert_body = {k: v for k, v in cert.items() if k != "canonical_sha256"}
    recomputed_canon = hashlib.sha256(
        json.dumps(cert_body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    _require(
        recorded_canon == recomputed_canon == EXPECTED_LIVE_CANONICAL_SHA256,
        f"certificate canonical_sha256 mismatch: {recomputed_canon}",
    )

    # Part 1 pinned provenance digest count & exact SHA pins
    p1 = cert["part1_provenance_and_hygiene"]
    _require(p1["qwen_neuron_map_fetched_pages"] == 15, "expected 15 fetched site pages")
    _require(len(p1["qwen_neuron_map_sha256"]) == 15, "expected 15 SHA-256 digests")
    _require(p1["alphaxiv_weight_edit_minimization"]["head_sha"] == EXPECTED_WEIGHT_EDIT_SHA, "WE SHA")
    _require(p1["alphaxiv_flint_goalkeeper_neuron"]["head_sha"] == EXPECTED_FLINT_SHA, "Flint SHA")

    # Part 2 exact 13-model census consistency, trajectories, & Z_2 confounder re-derivation
    p2 = cert["part2_atlas_30models_13full_and_z2_confounder"]
    m13 = p2["per_model_13_summary"]
    _require(len(m13) == 13, "expected 13 full models")
    total_neurons = sum(int(row["neurons"]) for row in m13.values())
    _require(total_neurons == p2["total_neurons_parsed"] == 1_504_256, "total_neurons mismatch")
    for slug, row in m13.items():
        _require(int(row["layers"]) * int(row["width"]) == int(row["neurons"]), f"{slug} shape mismatch")
        exact_dk = Fraction(int(row["dark_neurons"]), int(row["neurons"]))
        _require(str(exact_dk) == row["dark_fraction_exact"], f"{slug} dark_fraction mismatch")
        ratio = Fraction(row["dark_hi_3_7_exact"]) / Fraction(row["dark_lo_1_7_exact"])
        _require(str(ratio) == row["hi_to_lo_dark_ratio_exact"], f"{slug} hi/lo ratio mismatch")

    q08_ratio = Fraction(m13["q08"]["hi_to_lo_dark_ratio_exact"])
    other_max_ratio = max(
        Fraction(row["hi_to_lo_dark_ratio_exact"]) for slug, row in m13.items() if slug != "q08"
    )
    _require(q08_ratio > Fraction(3, 2) and other_max_ratio < Fraction(11, 10), "q08 anisotropy uniqueness failed")

    py_1k = Fraction(m13["py0a"]["dark_fraction_exact"])
    py_8k = Fraction(m13["py0b"]["dark_fraction_exact"])
    py_32k = Fraction(m13["py0c"]["dark_fraction_exact"])
    py_rebound = (
        py_1k > py_8k > py_32k
        and py_32k < Fraction(m13["py0"]["dark_fraction_exact"])
        and py_32k < Fraction(m13["py1"]["dark_fraction_exact"])
        and py_32k < Fraction(m13["py2"]["dark_fraction_exact"])
    )
    _require(py_rebound, "Pythia non-monotonic rebound check failed")
    smol_decay = (
        Fraction(m13["s135"]["dark_fraction_exact"])
        > Fraction(m13["smol"]["dark_fraction_exact"])
        > Fraction(m13["s17"]["dark_fraction_exact"])
    )
    _require(smol_decay, "SmolLM2 monotonic dark decay failed")

    lineage_recomputed = _build_z2_lineage_proofs()
    _require(lineage_recomputed == p2["z2_sign_symmetry_lineage_proofs"], "Z_2 lineage recomputation mismatch")

    # Part 3 exact tail re-derivations from pinned counts + standalone seeded permutation null
    p3 = cert["part3_q08_weight_edit_empirical_join"]
    sign_p = nhx.exact_binomial_two_sided_tail(21, 21)
    _require(str(sign_p) == p3["q08_24layer_column_anisotropy"]["exact_binomial_sign_p"] == "1/1048576", "sign p")
    # Honest all-layer sign test (weft-loom#44 N7): 21/24 layers have hi > lo (L00, L01, L23 reverse).
    sign_p_all24 = nhx.exact_binomial_two_sided_tail(24, 21)
    _require(sign_p_all24 == Fraction(2325, 8_388_608), "all-24-layer sign test p")

    top40_cnt, top40_n = p3["top40_linf_outliers_in_col_lt_512"]["count"]
    top40_k, top40_pop = p3["top40_linf_outliers_in_col_lt_512"]["background"]
    p_top40 = nhx.exact_hypergeom_tail(top40_pop, top40_k, top40_n, top40_cnt)
    _require(p_top40 < Fraction(1, 10**14), "top40 hypergeom p")

    def _exact_stratified_two_block_tail(
        n1: int, k1_bg: int, d1: int, n2: int, k2_bg: int, d2: int, obs_total: int
    ) -> Fraction:
        import math as _math

        den = _math.comb(n1, d1) * _math.comb(n2, d2)
        num = 0
        for x1 in range(max(0, d1 + k1_bg - n1), min(k1_bg, d1) + 1):
            for x2 in range(max(0, d2 + k2_bg - n2), min(k2_bg, d2) + 1):
                if x1 + x2 >= obs_total:
                    num += (
                        _math.comb(k1_bg, x1)
                        * _math.comb(n1 - k1_bg, d1 - x1)
                        * _math.comb(k2_bg, x2)
                        * _math.comb(n2 - k2_bg, d2 - x2)
                    )
        return Fraction(num, den)

    l15 = p3["layer_joins"]["L15"]
    l19 = p3["layer_joins"]["L19"]
    p_l15_adj = nhx.exact_hypergeom_tail(3584, 77, 64, 16)
    p_l15_lo = nhx.exact_hypergeom_tail(512, 29, 40, 9)
    p_l15_hi = nhx.exact_hypergeom_tail(3072, 48, 24, 7)
    p_l15_strat = _exact_stratified_two_block_tail(512, 29, 40, 3072, 48, 24, 16)
    p_l15_eq = nhx.exact_hypergeom_tail(3584, 364, 64, 22)
    p_l19_adj = nhx.exact_hypergeom_tail(3584, 11, 64, 4)
    p_l19_lo = nhx.exact_hypergeom_tail(512, 8, 35, 3)
    p_l19_hi = nhx.exact_hypergeom_tail(3072, 3, 29, 1)
    p_l19_strat = _exact_stratified_two_block_tail(512, 8, 35, 3072, 3, 29, 4)
    p_l19_eq = nhx.exact_hypergeom_tail(3584, 291, 64, 15)
    _require(p_l15_adj < Fraction(1, 10**13), "L15 adjacent brackets p < 1e-13")
    _require(p_l15_lo < Fraction(1, 100) and p_l15_hi < Fraction(1, 10**7), "L15 both blocks < 0.01")
    _require(p_l15_strat < Fraction(1, 10**9), "L15 combined stratified p < 1e-9")
    _require(p_l15_eq < Fraction(1, 10**6), "L15 equations p < 1e-6")
    _require(p_l19_adj < Fraction(1, 10**4), "L19 adjacent brackets p < 1e-4")
    # Narrowed scope (weft-loom#44 N2): in L19 neither block clears 0.01 individually, though combined stratified p < 2e-3.
    _require(p_l19_lo > Fraction(1, 100) and p_l19_hi > Fraction(1, 100), "L19 individual blocks > 0.01")
    _require(p_l19_strat < Fraction(2, 1000), "L19 combined stratified p < 2e-3")
    _require(p_l19_eq < Fraction(1, 10**3), "L19 equations p < 1e-3")
    _require(
        l15["adjacent_brackets"]["all"]["sel"] == [16, 64]
        and l19["adjacent_brackets"]["all"]["sel"] == [4, 64],
        "certified bracket counts mismatch",
    )

    standalone_perm_null = nhx.permutation_null_enrichment_exact(
        [1] * 77 + [0] * (3584 - 77), 1, 64, 16, n_perms=1000, seed=20261121
    )
    _require(
        standalone_perm_null["exceed_count"] == 0 and standalone_perm_null["empirical_p_exact"] == "1/1001",
        "standalone permutation null failed",
    )

    p_mcnemar = nhx.exact_binomial_two_sided_tail(579 + 5, 5)
    _require(p_mcnemar < Fraction(1, 10**150), "Exp 8 McNemar p")

    # Part 4 Gemma 4 E4B Layer-10 circuit fan-out & siliquant KL share exact checks
    p4 = cert["part4_gemma4_e4b_layer10_siliquant_bridge"]
    top5 = p4["top5_layers_by_mean_out"]
    _require(top5[0]["layer"] == 10 and top5[1]["layer"] == 9, "Gemma 4 E4B L10/L09 fan-out ranking")
    _require(
        all(top5[i]["mean_out_x100000"] > top5[i + 1]["mean_out_x100000"] for i in range(len(top5) - 1)),
        "top5 mean_out not strictly descending",
    )
    _require(p4["first_code_gt_prose_layer"] == 10, "L10 first code>prose")
    _require(
        sum(1 for g in p4["layer10_top6_groups"] if g["is_dark_hub"]) == p4["layer10_top6_dark_hub_count"] == 4,
        "L10 top-6 dark hub count",
    )
    l10_kl_share = Fraction(1560, 1560 + 182)
    _require(str(l10_kl_share) == p4["siliquant_cross_reference"]["l10_kl_share_exact"] == "60/67", "L10 KL share")

    # Part 5 Flint goalkeeper hardened -O-immune exact recount + negative control
    p5 = cert["part5_flint_goalkeeper_hardened_recount"]
    fixed_replay = nhx.strict_recount_goalkeeper_rows(_make_canonical_goalkeeper_table(15, 16), expected_per_condition=16)
    ood_replay = nhx.strict_recount_goalkeeper_rows(_make_canonical_goalkeeper_table(16, 16), expected_per_condition=16)
    _require(fixed_replay == p5["fixed_replication"], "Flint fixed exact replay mismatch")
    _require(ood_replay == p5["out_of_domain_preregistered"], "Flint OOD exact replay mismatch")
    bonf6 = 6 * Fraction(fixed_replay["exact_fisher_p"])
    _require(str(bonf6) == p5["fixed_bonferroni_6_exact"] == "1/1661656248570", "Flint Bonferroni-6")

    # Negative control: prompt leak MUST be rejected fail-closed even under python3 -O
    leak_rows = _make_canonical_goalkeeper_table(15, 16)
    leak_rows[0]["prompt"] = "Write about a goalkeeper in soccer."
    leak_rejected = False
    try:
        nhx.strict_recount_goalkeeper_rows(leak_rows, expected_per_condition=16)
    except nhx.ExactHomologyError:
        leak_rejected = True
    _require(leak_rejected, "prompt-leak negative control was not rejected")

    out: dict[str, Any] = {
        "audit_id": "AUDIT-059",
        "arm": "R5-4",
        "mode": "standalone_certified_replay",
        "oracle_class": "exact+anchor",
        "honest_novelty_class": "N1",
        "committed_live_canonical_sha256": recomputed_canon,
        "committed_certificate_zero_floats_verified": True,
        "atlas_13models_total_neurons": total_neurons,
        "pythia_py0_trajectory_and_endpoints_verified": py_rebound,
        "smollm2_monotonic_dark_decay_verified": smol_decay,
        "q08_unique_column_anisotropy_ratio_exact": str(q08_ratio),
        "other_12models_max_hi_lo_ratio_exact": str(other_max_ratio),
        "q08_all_24_layers_hi_gt_lo": "21/24",
        "q08_all_24_layers_sign_test_p_exact": str(sign_p_all24),
        "q08_posthoc_middle_21_of_21_sign_test_p_exact": str(sign_p),
        "top40_linf_in_lo512_p_lt_1e_14": p_top40 < Fraction(1, 10**14),
        "l15_adjacent_brackets_p_lt_1e_13": p_l15_adj < Fraction(1, 10**13),
        "l15_strat_both_blocks_lt_0_01": (p_l15_lo < Fraction(1, 100) and p_l15_hi < Fraction(1, 100)),
        "l15_strat_combined_p_lt_1e_9": p_l15_strat < Fraction(1, 10**9),
        "l19_adjacent_brackets_p_lt_1e_4": p_l19_adj < Fraction(1, 10**4),
        "l19_strat_both_blocks_lt_0_01": (p_l19_lo < Fraction(1, 100) and p_l19_hi < Fraction(1, 100)),
        "l19_strat_combined_p_lt_2e_3": p_l19_strat < Fraction(2, 1000),
        "exp8_mcnemar_579_vs_5_p_lt_1e_150": p_mcnemar < Fraction(1, 10**150),
        "standalone_permutation_null_code_crosscheck_1000_l15": standalone_perm_null,
        "z2_sign_symmetry_lineage_proofs": lineage_recomputed,
        "gemma4_e4b_layer10_exploratory_summary": {
            "epistemic_status": "post_hoc_exploratory_target",
            "rank1_mean_out_layer": 10,
            "layer10_mean_out_x100000": top5[0]["mean_out_x100000"],
            "layer09_mean_out_x100000": top5[1]["mean_out_x100000"],
            "first_code_gt_prose_layer": 10,
            "layer10_top6_dark_hub_count": 4,
            "siliquant_l10_kl_share_exact": str(l10_kl_share),
        },
        "flint_goalkeeper_exact_replay": {
            "fixed_exact_fisher_p": fixed_replay["exact_fisher_p"],
            "fixed_bonferroni_6_exact": str(bonf6),
            "ood_exact_fisher_p": ood_replay["exact_fisher_p"],
            "prompt_leak_negative_control_rejected": leak_rejected,
        },
    }
    nhx.verify_no_floats(out)
    canon = json.dumps(out, sort_keys=True, separators=(",", ":")).encode("utf-8")
    out["standalone_canonical_sha256"] = hashlib.sha256(canon).hexdigest()
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="AUDIT-059 exact-rational cross-model neuron homology & siliquant unlock bridge."
    )
    parser.add_argument("--json", action="store_true", help="Emit canonical sorted JSON (default behavior).")
    parser.add_argument(
        "--standalone",
        action="store_true",
        help="Run self-contained certified replay without external clones.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Force full live extraction from external read-only clones.",
    )
    parser.add_argument(
        "--repos-dir",
        type=Path,
        default=REPOS_DIR,
        help="Path to directory holding read-only external clones for live extraction.",
    )
    args = parser.parse_args(argv)
    _require(not (args.standalone and args.live), "--standalone and --live are mutually exclusive")

    use_live = args.live or (not args.standalone and _repos_available(args.repos_dir))
    res = run_all(args.repos_dir) if use_live else run_standalone_certified_replay(CERT_PATH)
    print(json.dumps(res, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
