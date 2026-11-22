#!/usr/bin/env python3
"""Dual-mode pure-Q audit runner for Track S1 (AUDIT-097):
Self-Rewriting LLMs (arXiv:2611.21885v2) & Weight-Edit Minimization
(alphaXiv/weight-edit-minimization) -> warp-drive, siliquant & aletheia v2.0.0 bridge.

Modes:
  --live        Recounts live staged raw .npz/.npy/.jsonl/PDF artifacts in 100% pure stdlib
  --standalone  Self-contained exact pure-Q fixture replay with zero external file I/O

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
import struct
import sys
from typing import Any, Mapping
import zipfile

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from wsp_self_rewriting_oracle import (
    BF16_DYADIC_SHIFT,
    BF16_EXP_MASK,
    F32_DYADIC_SHIFT,
    F64_DYADIC_SHIFT,
    OracleError,
    _require_exact_int,
    _require_sha256,
    bf16_u16_to_scaled_int,
    canonical_sha256,
    certify_group_quantization_survival_exact,
    certify_paired_dual_screen,
    certify_rank1_vs_pruning_exact,
    certify_self_rewriting_session,
    contains_float_recursive,
    f32_u32_to_bf16_u16_rne,
    f32_u32_to_scaled_int,
    f64_u64_to_scaled_int,
    format_exact_decimal,
    format_exact_percent,
    frac_to_dict,
    make_fixed_radix_certificate,
    quantize_sym_rne_exact,
    verify_fixed_radix_stage2_division_free,
)

SCHEMA_VERSION = "audit097.wsp_self_rewriting.v1"
STAGE2_RADIX_SCALE = 1 << 32
DEFAULT_STAGE2_BOUNDS: dict[str, dict[str, int]] = {
    "max_frobenius_sq": {"num": 10, "den": 1},
    "max_spectral_norm_sq": {"num": 10, "den": 1},
    "max_leakage_energy": {"num": 1, "den": 1000},
    "max_safety_dot_sq": {"num": 1, "den": 10000},
}
DEFAULT_GOVERNANCE_POLICY: dict[str, Any] = {
    "min_heldout_problems": 200,
    "max_practice_overlap": 0,
    "max_unnecessary_tensors": 0,
    "min_safety_canary_rate": {"num": 1, "den": 1},
    "max_unrelated_kl_shift": {"num": 1, "den": 20},
}
CAR_RAMP_ACTOR_MAP: tuple[tuple[str, str], ...] = (
    ("embed.W", "params/Dense_0/kernel.npy"),
    ("embed.b", "params/Dense_0/bias.npy"),
    ("actor0.W", "params/GeneralActorCriticRNN_0/DenseAndActivation_0/Dense_0/kernel.npy"),
    ("actor0.b", "params/GeneralActorCriticRNN_0/DenseAndActivation_0/Dense_0/bias.npy"),
    ("actor1.W", "params/GeneralActorCriticRNN_0/DenseAndActivation_1/Dense_0/kernel.npy"),
    ("actor1.b", "params/GeneralActorCriticRNN_0/DenseAndActivation_1/Dense_0/bias.npy"),
    ("actor2.W", "params/GeneralActorCriticRNN_0/DenseAndActivation_2/Dense_0/kernel.npy"),
    ("actor2.b", "params/GeneralActorCriticRNN_0/DenseAndActivation_2/Dense_0/bias.npy"),
    ("actor3.W", "params/GeneralActorCriticRNN_0/DenseAndActivation_3/Dense_0/kernel.npy"),
    ("actor3.b", "params/GeneralActorCriticRNN_0/DenseAndActivation_3/Dense_0/bias.npy"),
    ("actor4.W", "params/GeneralActorCriticRNN_0/DenseAndActivation_4/Dense_0/kernel.npy"),
    ("actor4.b", "params/GeneralActorCriticRNN_0/DenseAndActivation_4/Dense_0/bias.npy"),
    ("actor_head.W", "params/GeneralActorCriticRNN_0/Dense_0/kernel.npy"),
    ("actor_head.b", "params/GeneralActorCriticRNN_0/Dense_0/bias.npy"),
)
EXP8_AUDITED_SPECS: tuple[str, ...] = (
    "A64_topk64",
    "A48_topk48",
    "B15_full",
    "ER768_L15",
    "ER384_L15",
    "D32k_entries",
    "D16k_entries",
    "D8k_entries",
    "D4k_entries",
    "D2k_entries",
    "D1k_entries",
)
CAR_RAMP_CHECKPOINTS: tuple[tuple[str, str], ...] = (
    ("edited_actor", "edited_actor.npz"),
    ("joint_free", "joint_free.npz"),
    ("greedy12_astra", "greedy12_astra.npz"),
    ("greedy11_drop02", "greedy11_drop02.npz"),
    ("c2_14entry", "c2_14entry.npz"),
    ("c2_min_10entry", "c2_min_10entry.npz"),
)
PINNED_INTAKE_FILES: tuple[str, ...] = (
    "sources/arxiv_2611_21885.pdf",
    "raw/openresearch_EdsyLoFUfY40.txt",
    "repos/weight-edit-minimization/data/exp8/edit.npz",
    "repos/weight-edit-minimization/data/exp8/factors_layer15.npz",
    "repos/weight-edit-minimization/data/exp8/factors_layer19.npz",
    "repos/weight-edit-minimization/data/exp8/frozen_edit_audit.json",
    "repos/weight-edit-minimization/data/exp8/final_choice.json",
    "repos/weight-edit-minimization/data/exp8/validate_D4k_math500.jsonl",
    "repos/weight-edit-minimization/data/car-ramp/base_ppo_final_seed0.npz",
    "repos/weight-edit-minimization/data/car-ramp/edited_actor.npz",
    "repos/weight-edit-minimization/data/car-ramp/greedy12_astra.npz",
    "repos/weight-edit-minimization/data/car-ramp/c2_min_10entry.npz",
)
LIVE_CERT_TOP_KEYS: frozenset[str] = frozenset(
    {
        "schema_version",
        "audit_id",
        "mode",
        "live_intake_audited",
        "novelty_class",
        "stage2_bounds",
        "pinned_artifacts",
        "st_numpy_bf16_overflow_blocked",
        "exact_pruning_theorems",
        "siliquant_quantization_bridge",
        "exp8_tensor_audit",
        "exp8_screen_audit",
        "car_ramp_audit",
        "arxiv_governance_audit",
        "overall_verified",
        "certificate_sha256",
    }
)
STANDALONE_CERT_TOP_KEYS: frozenset[str] = frozenset(
    {
        "schema_version",
        "audit_id",
        "mode",
        "live_intake_audited",
        "novelty_class",
        "stage2_bounds",
        "st_numpy_bf16_overflow_blocked",
        "exact_pruning_theorems",
        "siliquant_quantization_bridge",
        "stage2_fixture_verdicts",
        "paired_dual_screen_fixture",
        "arxiv_governance_audit",
        "overall_verified",
        "certificate_sha256",
    }
)

# Exact 65,536-entry bfloat16 -> scaled integer (2^133) LUT; None marks non-finite exp==0xFF
_BF16_SCALED_LUT: tuple[int | None, ...] = tuple(
    None if ((u >> 7) & BF16_EXP_MASK) == BF16_EXP_MASK else bf16_u16_to_scaled_int(u)
    for u in range(1 << 16)
)


def parse_npy_bytes(raw: bytes) -> tuple[str, tuple[int, ...], bytes]:
    """Parse a NumPy .npy binary buffer in pure stdlib without importing numpy."""
    if not isinstance(raw, (bytes, bytearray)) or len(raw) < 10 or raw[:6] != b"\x93NUMPY":
        raise OracleError("invalid .npy magic header")
    major = raw[6]
    if major == 1:
        hdr_len = struct.unpack("<H", raw[8:10])[0]
        hdr_end = 10 + hdr_len
        hdr_bytes = raw[10:hdr_end]
    elif major in (2, 3):
        if len(raw) <= 12:
            raise OracleError("truncated .npy v2/v3 header")
        hdr_len = struct.unpack("<I", raw[8:12])[0]
        hdr_end = 12 + hdr_len
        hdr_bytes = raw[12:hdr_end]
    else:
        raise OracleError(f"unsupported .npy major version: {major}")
    if hdr_end > len(raw):
        raise OracleError("truncated .npy header payload")
    meta = ast.literal_eval(bytes(hdr_bytes).decode("latin1").strip())
    if not isinstance(meta, dict) or "descr" not in meta or "shape" not in meta:
        raise OracleError("malformed .npy header dictionary")
    shape = tuple(int(dim) for dim in meta["shape"])
    return str(meta["descr"]), shape, bytes(raw[hdr_end:])


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    """Read newline-delimited JSON objects strictly without float coercion."""
    if not path.is_file():
        raise OracleError(f"missing jsonl file: {path.name}")
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        obj = json.loads(stripped, parse_float=str)
        if not isinstance(obj, dict):
            raise OracleError(f"{path.name}:{line_no} is not a JSON object")
        rows.append(obj)
    return rows


def sha256_file(path: Path) -> tuple[str, int]:
    """Return exact (sha256_hex, size_bytes) of a file."""
    if not path.is_file():
        raise OracleError(f"missing pinned artifact: {path.name}")
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest(), len(data)


def run_synthetic_rank1_pruning_theorems() -> dict[str, Any]:
    """Prove exact pure-Q pruning dichotomy (column vs row vs entry) against a safety subspace."""
    u = [Fraction(3, 1), Fraction(4, 1), Fraction(5, 1), Fraction(0, 1)]
    v = [Fraction(2, 1), Fraction(1, 1), Fraction(-2, 1)]
    safety_vectors = [[Fraction(4, 1), Fraction(-3, 1), Fraction(0, 1), Fraction(1, 1)]]

    full_mask = [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]
    col_mask = [[1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1]]
    row_mask = [[1, 1, 1], [0, 0, 0], [1, 1, 1], [0, 0, 0]]
    entry_mask = [[1, 0, 1], [1, 1, 0], [0, 1, 1], [0, 0, 0]]

    full_res = certify_rank1_vs_pruning_exact(u, v, full_mask, safety_vectors)
    col_res = certify_rank1_vs_pruning_exact(u, v, col_mask, safety_vectors)
    row_res = certify_rank1_vs_pruning_exact(u, v, row_mask, safety_vectors)
    entry_res = certify_rank1_vs_pruning_exact(u, v, entry_mask, safety_vectors)

    theorem_holds = (
        full_res["is_exact_rank_le_1"]
        and full_res["leakage_energy_sq"]["num"] == 0
        and not full_res["safety_orthogonality_breached"]
        and col_res["is_exact_rank_le_1"]
        and col_res["leakage_energy_sq"]["num"] == 0
        and not col_res["safety_orthogonality_breached"]
        and row_res["is_exact_rank_le_1"]
        and row_res["closed_form_row_leakage_holds"]
        and row_res["leakage_energy_sq"]["num"] > 0
        and row_res["safety_orthogonality_breached"]
        and (not entry_res["is_exact_rank_le_1"])
        and entry_res["leakage_energy_sq"]["num"] > 0
        and entry_res["safety_orthogonality_breached"]
    )
    return {
        "theorem_holds": theorem_holds,
        "full_rank1": full_res,
        "column_pruned": col_res,
        "row_pruned": row_res,
        "entry_pruned": entry_res,
    }


def run_siliquant_double_bind_witness() -> dict[str, Any]:
    """Exact pure-Q witness for Moissanite-Labs/siliquant ffn_down quantization-survival double bind.

    Regime A (siliquant Part L 0.17% ffn_down outlier + #26 (3/4)-ULP erasure):
      In a G=32 group with max_abs=700 (one 10x outlier; ULP=100 at qmax=7), an edit of +40 on a
      grid-centered base weight 0 sits strictly below (1/2)-ULP and (3/4)-ULP and is 100% ERASED
      (q_edit == q_base == 0). Under outlier-free / Hadamard-rotated max_abs=70 (ULP=10), the
      exact same +40 edit jumps to 4*ULP and survives cleanly (q: 0 -> 4).
    Regime B (sparse L0-spike scale distortion, car-ramp C2-min/greedy12):
      Spiking a single entry in a G=32 group from 70 to 2800 (40x base max; ULP jumps 10 -> 400)
      survives at the spiked coordinate (q=7) while collapsing 31/31 (100%) of unedited base
      neighbors (magnitudes 10..70 < 200 = 0.5*ULP') to quantized code 0.
    """
    base_outlier = [0] + [30 + (i % 4) * 10 for i in range(30)] + [700]
    edit_outlier = [40] + base_outlier[1:]
    regime_a_erased = certify_group_quantization_survival_exact(base_outlier, edit_outlier, qmax=7)

    base_rotated = [0] + [30 + (i % 4) * 10 for i in range(30)] + [70]
    edit_rotated = [40] + base_rotated[1:]
    regime_a_recovered = certify_group_quantization_survival_exact(base_rotated, edit_rotated, qmax=7)

    base_spike = [20, 70] + [20 + (i % 5) * 10 for i in range(30)]
    edit_spike = [2800] + base_spike[1:]
    regime_b_collapse = certify_group_quantization_survival_exact(base_spike, edit_spike, qmax=7)

    double_bind_holds = (
        regime_a_erased["erased_edits"] == 1
        and regime_a_erased["below_three_quarter_ulp"] == 1
        and regime_a_recovered["survived_code_flips"] == 1
        and regime_a_recovered["erased_edits"] == 0
        and regime_b_collapse["survived_code_flips"] == 1
        and regime_b_collapse["collateral_unedited_zeroed"] == 31
    )
    return {
        "double_bind_holds": double_bind_holds,
        "regime_a_outlier_erasure_g32": regime_a_erased,
        "regime_a_outlier_rotated_recovery_g32": regime_a_recovered,
        "regime_b_sparse_spike_collateral_zeroing_g32": regime_b_collapse,
    }


def run_arxiv_governance_audit(base_weights_sha256: str) -> dict[str, Any]:
    """Audit arXiv:2611.21885v2 protocols vs a warp-drive + aletheia v2.0.0 compliant session."""
    alt_parent_sha = hashlib.sha256(b"cumulative-parent-checkpoint-v1").hexdigest()

    sessions: tuple[dict[str, Any], ...] = (
        {
            "session_id": "arxiv_2611_21885_proto1_27b_self_edit",
            "sandbox_confined": False,
            "external_verifier_decides_submission": False,
            "base_weights_sha256": base_weights_sha256,
            "parent_weights_sha256": base_weights_sha256,
            "session_index": 2,
            "cumulative_edit_claimed": False,
            "paired_heldout_control_present": True,
            "heldout_problem_count": 490,
            "practice_overlap_count": 8,
            "collateral_ablation_checked": False,
            "unnecessary_tensor_count": 0,
            "safety_canary_passed": 0,
            "safety_canary_total": 0,
            "unrelated_kl_shift": {"num": 1, "den": 1},
            "has_recurrent_state_banks": True,
            "state_banks_flushed_and_confirmed": False,
        },
        {
            "session_id": "arxiv_2611_21885_proto1_27b_damaged_repair",
            "sandbox_confined": False,
            "external_verifier_decides_submission": False,
            "base_weights_sha256": base_weights_sha256,
            "parent_weights_sha256": base_weights_sha256,
            "session_index": 1,
            "cumulative_edit_claimed": False,
            "paired_heldout_control_present": True,
            "heldout_problem_count": 200,
            "practice_overlap_count": 5,
            "collateral_ablation_checked": False,
            "unnecessary_tensor_count": 19,
            "safety_canary_passed": 0,
            "safety_canary_total": 0,
            "unrelated_kl_shift": {"num": 1, "den": 1},
            "has_recurrent_state_banks": True,
            "state_banks_flushed_and_confirmed": False,
        },
        {
            "session_id": "arxiv_2611_21885_proto2_session8_dt_bias",
            "sandbox_confined": True,
            "external_verifier_decides_submission": False,
            "base_weights_sha256": base_weights_sha256,
            "parent_weights_sha256": base_weights_sha256,
            "session_index": 8,
            "cumulative_edit_claimed": True,
            "paired_heldout_control_present": True,
            "heldout_problem_count": 490,
            "practice_overlap_count": 8,
            "collateral_ablation_checked": True,
            "unnecessary_tensor_count": 0,
            "safety_canary_passed": 0,
            "safety_canary_total": 0,
            "unrelated_kl_shift": {"num": 1, "den": 1},
            "has_recurrent_state_banks": True,
            "state_banks_flushed_and_confirmed": False,
        },
        {
            "session_id": "warp_drive_aletheia_v2_compliant_session",
            "sandbox_confined": True,
            "external_verifier_decides_submission": True,
            "base_weights_sha256": base_weights_sha256,
            "parent_weights_sha256": alt_parent_sha,
            "session_index": 2,
            "cumulative_edit_claimed": True,
            "paired_heldout_control_present": True,
            "heldout_problem_count": 490,
            "practice_overlap_count": 0,
            "collateral_ablation_checked": True,
            "unnecessary_tensor_count": 0,
            "safety_canary_passed": 4,
            "safety_canary_total": 4,
            "unrelated_kl_shift": {"num": 1, "den": 100},
            "has_recurrent_state_banks": True,
            "state_banks_flushed_and_confirmed": True,
        },
    )

    verdicts = {s["session_id"]: certify_self_rewriting_session(s, DEFAULT_GOVERNANCE_POLICY) for s in sessions}
    return {
        "policy": DEFAULT_GOVERNANCE_POLICY,
        "verdicts": verdicts,
    }


def _evaluate_exp8_live_tensors(exp8_dir: Path, base_sha: str) -> dict[str, Any]:
    """Exact bit-level pure-Q audit of Exp 8 edit.npz, factors_layer{15,19}.npz, 11 pruning specs,
    and live Group-32 / Group-128 INT4 quantization survival (siliquant bridge)."""
    edit_path = exp8_dir / "edit.npz"
    if not edit_path.is_file():
        raise OracleError("missing exp8/edit.npz")
    with zipfile.ZipFile(edit_path) as zf:
        edit_raw = {name: parse_npy_bytes(zf.read(name)) for name in zf.namelist()}

    u_scaled: dict[int, list[int]] = {}
    v_scaled: dict[int, list[int]] = {}
    u_norm_sq_int: dict[int, int] = {}
    delta_maps: dict[int, dict[tuple[int, int], int]] = {}
    layer_reports: dict[str, Any] = {}
    siliquant_exp8_survival: dict[str, Any] = {}

    for t_idx, layer in ((0, 15), (1, 19)):
        fac_path = exp8_dir / f"factors_layer{layer}.npz"
        base_path = exp8_dir / f"base_l{layer}_down_proj_f32.npy"
        if not fac_path.is_file() or not base_path.is_file():
            raise OracleError(f"missing Exp 8 layer {layer} tensor inputs")

        with zipfile.ZipFile(fac_path) as zf:
            fdict = {name: parse_npy_bytes(zf.read(name)) for name in zf.namelist()}
        u_ints = [f32_u32_to_scaled_int(u) for (u,) in struct.iter_unpack("<I", fdict["u.npy"][2])]
        v_ints = [f32_u32_to_scaled_int(v) for (v,) in struct.iter_unpack("<I", fdict["v.npy"][2])]
        u_scaled[layer] = u_ints
        v_scaled[layer] = v_ints
        u2_int = sum(x * x for x in u_ints)
        v2_int = sum(y * y for y in v_ints)
        u_norm_sq_int[layer] = u2_int

        _, b_shape, b_body = parse_npy_bytes(base_path.read_bytes())
        if len(b_shape) != 2:
            raise OracleError(f"expected 2D base tensor for layer {layer}")
        n_rows, n_cols = b_shape
        n_total_elems = n_rows * n_cols
        b_u32 = struct.unpack(f"<{n_total_elems}I", b_body)

        idx_descr, _, idx_body = edit_raw[f"t{t_idx}_idx.npy"]
        val_descr, _, val_body = edit_raw[f"t{t_idx}_val.npy"]
        idx_fmt = "<q" if idx_descr == "<i8" else "<i"
        if val_descr != "<u2":
            raise OracleError(f"expected <u2 bf16 values in edit.npz for layer {layer}")
        idxs = [x for (x,) in struct.iter_unpack(idx_fmt, idx_body)]
        vals = [x for (x,) in struct.iter_unpack("<H", val_body)]

        lut = _BF16_SCALED_LUT
        dmap: dict[tuple[int, int], int] = {}
        flat_edit_info: dict[int, tuple[int, int, int]] = {}
        rows_touched: set[int] = set()
        cols_touched: set[int] = set()
        for flat, new_u16 in zip(idxs, vals):
            base_u16 = f32_u32_to_bf16_u16_rne(b_u32[flat])
            if new_u16 != base_u16:
                r, c = divmod(flat, n_cols)
                rows_touched.add(r)
                cols_touched.add(c)
                new_s = lut[new_u16]
                base_s = lut[base_u16]
                if new_s is None:
                    raise OracleError("non-finite bf16 weight in Exp 8 tensor")
                base_int = int(base_s)  # type: ignore[arg-type]
                dmap[(r, c)] = new_s - base_int
                flat_edit_info[flat] = (base_int, new_s, new_u16)
        delta_maps[layer] = dmap

        # Live siliquant Group-32 & Group-128 symmetric INT4 (qmax=7) survival on touched groups
        layer_quant: dict[str, Any] = {}
        for grp_size in (32, 128):
            if n_total_elems % grp_size == 0:
                edits_by_grp: dict[int, list[tuple[int, int, int, int]]] = {}
                for flat, (base_int, new_s, new_u16) in flat_edit_info.items():
                    g_id = flat // grp_size
                    edits_by_grp.setdefault(g_id, []).append((flat - g_id * grp_size, base_int, new_s, new_u16))
                tot_ed = tot_surv = tot_erased = tot_lt_half = tot_lt_3q = tot_ge_1 = 0
                tot_scale_chg = tot_collat_flips = tot_collat_zeroed = 0
                max_scale_rat = Fraction(0, 1)
                for g_id, grp_edits in edits_by_grp.items():
                    start = g_id * grp_size
                    mags = [(((bu & 0x7FFFFFFF) + 0x7FFF + ((bu >> 16) & 1)) >> 16) for bu in b_u32[start : start + grp_size]]
                    max_b_mag = max(mags)
                    if max_b_mag >= 0x7F80:
                        raise OracleError("non-finite base weight in touched quantization group")
                    mb = int(lut[max_b_mag])  # type: ignore[arg-type]
                    e_mags = list(mags)
                    for off, _, _, nu16 in grp_edits:
                        e_mags[off] = nu16 & 0x7FFF
                    me = int(lut[max(e_mags)])  # type: ignore[arg-type]
                    if me != mb:
                        tot_scale_chg += 1
                        if mb > 0:
                            rat = Fraction(me, mb)
                            if rat > max_scale_rat:
                                max_scale_rat = rat
                    edited_offs: set[int] = set()
                    for off, bv, ev, _ in grp_edits:
                        edited_offs.add(off)
                        qb = quantize_sym_rne_exact(bv, mb, 7)
                        qe = quantize_sym_rne_exact(ev, me, 7)
                        tot_ed += 1
                        d_abs = abs(ev - bv)
                        if qe != qb:
                            tot_surv += 1
                        else:
                            tot_erased += 1
                        if 14 * d_abs < mb:
                            tot_lt_half += 1
                        if 28 * d_abs < 3 * mb:
                            tot_lt_3q += 1
                        if 7 * d_abs >= mb:
                            tot_ge_1 += 1
                    if me != mb:
                        for off, mag in enumerate(mags):
                            if off not in edited_offs:
                                bv = int(lut[mag])  # type: ignore[arg-type]
                                qb = quantize_sym_rne_exact(bv, mb, 7)
                                qe = quantize_sym_rne_exact(bv, me, 7)
                                if qe != qb:
                                    tot_collat_flips += 1
                                if qb != 0 and qe == 0:
                                    tot_collat_zeroed += 1
                surv_frac = Fraction(tot_surv, tot_ed) if tot_ed > 0 else Fraction(0, 1)
                lt3q_frac = Fraction(tot_lt_3q, tot_ed) if tot_ed > 0 else Fraction(0, 1)
                layer_quant[f"G{grp_size}"] = {
                    "group_size": grp_size,
                    "touched_groups": len(edits_by_grp),
                    "edited_entries": tot_ed,
                    "survived_code_flips": tot_surv,
                    "erased_edits": tot_erased,
                    "survival_percent": format_exact_percent(surv_frac, 2),
                    "below_half_ulp": tot_lt_half,
                    "below_three_quarter_ulp": tot_lt_3q,
                    "below_three_quarter_ulp_percent": format_exact_percent(lt3q_frac, 2),
                    "at_least_one_ulp": tot_ge_1,
                    "scale_changed_groups": tot_scale_chg,
                    "max_scale_distortion_dec": format_exact_decimal(max_scale_rat, 3),
                    "collateral_unedited_flips": tot_collat_flips,
                    "collateral_unedited_zeroed": tot_collat_zeroed,
                }
        siliquant_exp8_survival[f"layer_{layer}"] = layer_quant

        zero_u_rows = [idx for idx, val in enumerate(u_ints) if val == 0]
        missing_rows = sorted(set(range(n_rows)) - rows_touched)
        rank1_exact_sq = Fraction(u2_int * v2_int, 1 << (4 * F32_DYADIC_SHIFT))
        layer_reports[f"layer_{layer}"] = {
            "shape": [n_rows, n_cols],
            "edit_npz_entries": len(idxs),
            "nonzero_bf16_entries": len(dmap),
            "rows_touched": len(rows_touched),
            "cols_touched": len(cols_touched),
            "missing_rows": missing_rows,
            "exact_zero_u_rows": zero_u_rows,
            "exact_rank1_frob_sq_dec": format_exact_decimal(rank1_exact_sq, 6),
        }

    dot_u = sum(a * b for a, b in zip(u_scaled[15], u_scaled[19]))
    cos2_u = Fraction(dot_u * dot_u, u_norm_sq_int[15] * u_norm_sq_int[19])

    dot_v = sum(a * b for a, b in zip(v_scaled[15], v_scaled[19]))
    v15_sq = sum(x * x for x in v_scaled[15])
    v19_sq = sum(x * x for x in v_scaled[19])
    cos2_v = Fraction(dot_v * dot_v, v15_sq * v19_sq)

    specs_dir = exp8_dir / "specs"
    spec_results: dict[str, Any] = {}
    stage2_verdicts: dict[str, Any] = {}
    a64_leakage_ref = Fraction(0)

    for spec_name in EXP8_AUDITED_SPECS:
        spec_path = specs_dir / f"{spec_name}.json"
        if not spec_path.is_file():
            raise OracleError(f"missing Exp 8 spec: {spec_name}.json")
        spec = json.loads(spec_path.read_text(encoding="utf-8"), parse_float=str)

        tot_entries = 0
        tot_frob = Fraction(0)
        tot_par = Fraction(0)
        tot_perp = Fraction(0)
        per_layer: dict[str, Any] = {}

        for layer, tag in ((15, "15"), (19, "19")):
            dmap = delta_maps[layer]
            u_ints = u_scaled[layer]
            u2_int = u_norm_sq_int[layer]

            if f"entries{tag}" in spec:
                sub = [(r, c, dmap[(r, c)]) for r, c in spec[f"entries{tag}"] if (r, c) in dmap]
            elif f"rows{tag}" in spec:
                rset = set(spec[f"rows{tag}"])
                cset = set(spec[f"cols{tag}"]) if f"cols{tag}" in spec else None
                sub = [(r, c, d) for (r, c), d in dmap.items() if r in rset and (cset is None or c in cset)]
            elif f"cols{tag}" in spec:
                cset = set(spec[f"cols{tag}"])
                sub = [(r, c, d) for (r, c), d in dmap.items() if c in cset]
            else:
                sub = [(r, c, d) for (r, c), d in dmap.items()]

            frob_int = sum(d * d for _, _, d in sub)
            col_dots: dict[int, int] = {}
            for r, c, d in sub:
                col_dots[c] = col_dots.get(c, 0) + u_ints[r] * d
            par_num_int = sum(val * val for val in col_dots.values())

            frob_frac = Fraction(frob_int, 1 << (2 * BF16_DYADIC_SHIFT))
            par_frac = Fraction(par_num_int, u2_int << (2 * BF16_DYADIC_SHIFT)) if frob_int > 0 else Fraction(0)
            perp_frac = frob_frac - par_frac
            off_frac = (perp_frac / frob_frac) if frob_frac > 0 else Fraction(0)

            tot_entries += len(sub)
            tot_frob += frob_frac
            tot_par += par_frac
            tot_perp += perp_frac
            per_layer[f"layer_{layer}"] = {
                "active_entries": len(sub),
                "frob_sq_dec": format_exact_decimal(frob_frac, 6),
                "perp_sq_dec": format_exact_decimal(perp_frac, 6),
                "off_rank1_percent": format_exact_percent(off_frac, 3),
            }

        tot_off = (tot_perp / tot_frob) if tot_frob > 0 else Fraction(0)
        if spec_name == "A64_topk64":
            a64_leakage_ref = tot_perp
        leakage_mult = (tot_perp / a64_leakage_ref) if a64_leakage_ref > 0 else Fraction(0)

        spec_results[spec_name] = {
            "total_active_entries": tot_entries,
            "total_frob_sq_dec": format_exact_decimal(tot_frob, 6),
            "total_par_sq_dec": format_exact_decimal(tot_par, 6),
            "total_perp_sq_dec": format_exact_decimal(tot_perp, 6),
            "off_rank1_percent": format_exact_percent(tot_off, 3),
            "leakage_multiplier_vs_a64_dec": format_exact_decimal(leakage_mult, 2),
            "layers": per_layer,
        }

        adapter_sha = hashlib.sha256(spec_path.read_bytes()).hexdigest()
        cert = make_fixed_radix_certificate(
            adapter_id=f"exp8_{spec_name}",
            adapter_sha256=adapter_sha,
            base_weights_sha256=base_sha,
            radix_scale=STAGE2_RADIX_SCALE,
            frobenius_sq=tot_frob,
            spectral_norm_sq=tot_frob,
            leakage_energy=tot_perp,
            max_safety_dot_sq=Fraction(0),
        )
        stage2_verdicts[spec_name] = verify_fixed_radix_stage2_division_free(
            cert,
            DEFAULT_STAGE2_BOUNDS,
            expected_base_weights_sha256=base_sha,
        )

    frozen_audit = json.loads((exp8_dir / "frozen_edit_audit.json").read_text(encoding="utf-8"), parse_float=str)
    claimed_changed = int(frozen_audit["15"]["changed_values"]) + int(frozen_audit["19"]["changed_values"])
    actual_changed = layer_reports["layer_15"]["nonzero_bf16_entries"] + layer_reports["layer_19"]["nonzero_bf16_entries"]

    return {
        "claimed_changed_values_f32": claimed_changed,
        "actual_nonzero_bf16_entries": actual_changed,
        "bf16_noop_collapse_count": claimed_changed - actual_changed,
        "cos2_u15_u19_dec": format_exact_decimal(cos2_u, 6),
        "cos2_v15_v19_dec": format_exact_decimal(cos2_v, 6),
        "layers": layer_reports,
        "specs": spec_results,
        "stage2_verdicts": stage2_verdicts,
        "siliquant_live_exp8_int4_survival": siliquant_exp8_survival,
    }


def _normalize_screen_rows(raw_rows: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    """Normalize weight-edit-minimization screen JSONL rows into strict oracle schema."""
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(raw_rows):
        if "id" not in row or "correct" not in row or "finish" not in row:
            raise OracleError(f"{label}[{idx}] missing required screen keys (id, correct, finish)")
        c_list = row["correct"]
        f_list = row["finish"]
        if not isinstance(c_list, list) or len(c_list) != 1 or type(c_list[0]) is not int or c_list[0] not in (0, 1):
            raise OracleError(f"{label}[{idx}].correct must be a 1-element list of exact int 0/1")
        if not isinstance(f_list, list) or len(f_list) != 1 or type(f_list[0]) is not str:
            raise OracleError(f"{label}[{idx}].finish must be a 1-element list of str")
        out.append({"id": row["id"], "correct": c_list[0] == 1, "finish_reason": f_list[0]})
    return out


def _evaluate_exp8_live_screens(repo_dir: Path) -> dict[str, Any]:
    """Exact paired McNemar dual-gate audit across Exp 8 screens and MATH-500 validation."""
    screens_dir = repo_dir / "data" / "screens"
    exp8_dir = repo_dir / "data" / "exp8"
    screen_files = sorted(screens_dir.glob("screen_*.jsonl"))
    if len(screen_files) == 0:
        raise OracleError("no screen_*.jsonl files found")

    a64_rows = _normalize_screen_rows(load_jsonl_rows(screens_dir / "screen_A64_topk64.jsonl"), "A64_topk64")
    g000_rows = _normalize_screen_rows(load_jsonl_rows(screens_dir / "screen_G000_scale.jsonl"), "G000_scale")
    a64_ids = {r["id"] for r in a64_rows}

    key_arms = (
        "G000_scale",
        "A48_topk48",
        "B15_full",
        "B15_full_s125",
        "B19_full",
        "ER128_L15",
        "ER384_L15",
        "ER768_L15",
        "D1k_entries",
        "D2k_entries",
        "D4k_entries",
        "D8k_entries",
        "D16k_entries",
        "D32k_entries",
    )
    paired_vs_a64: dict[str, Any] = {}
    for arm in key_arms:
        cand_rows = _normalize_screen_rows(load_jsonl_rows(screens_dir / f"screen_{arm}.jsonl"), arm)
        paired_vs_a64[arm] = certify_paired_dual_screen(
            a64_rows,
            cand_rows,
            max_cut_regression_p=Fraction(1, 100),
            min_acc_drop_p=Fraction(1, 20),
        )

    a64_vs_g000 = certify_paired_dual_screen(
        g000_rows,
        a64_rows,
        max_cut_regression_p=Fraction(1, 100),
        min_acc_drop_p=Fraction(1, 20),
    )

    val_path = exp8_dir / "validate_D4k_math500.jsonl"
    val_rows = load_jsonl_rows(val_path)
    val_ids: set[str] = set()
    total_samples = 0
    val_correct = 0
    sample0_correct = 0
    for idx, row in enumerate(val_rows):
        if "id" not in row or "correct" not in row or type(row["id"]) is not str:
            raise OracleError(f"validate_D4k_math500[{idx}] missing id/correct")
        c_list = row["correct"]
        if not isinstance(c_list, list) or len(c_list) == 0:
            raise OracleError(f"validate_D4k_math500[{idx}].correct must be non-empty list")
        val_ids.add(row["id"])
        for c_val in c_list:
            if type(c_val) is not int or c_val not in (0, 1):
                raise OracleError(f"validate_D4k_math500[{idx}].correct entries must be exact int 0/1")
            total_samples += 1
            val_correct += c_val
        sample0_correct += c_list[0]
    val_acc = Fraction(val_correct, total_samples) if total_samples > 0 else Fraction(0)

    results_json = json.loads((exp8_dir / "results_table.json").read_text(encoding="utf-8"), parse_float=str)
    candidate_entries = json.loads((exp8_dir / "candidate_entries.json").read_text(encoding="utf-8"), parse_float=str)
    paired_math500_controls = sorted(
        p.name for p in exp8_dir.glob("*math500*.jsonl") if p.name != "validate_D4k_math500.jsonl"
    )

    return {
        "screen_jsonl_count": len(screen_files),
        "results_table_json_count": len(results_json),
        "candidate_entries_json_count": len(candidate_entries),
        "screen_problem_count": len(a64_ids),
        "a64_vs_g000": a64_vs_g000,
        "paired_vs_a64": paired_vs_a64,
        "validate_d4k_math500": {
            "problem_rows": len(val_rows),
            "total_samples": total_samples,
            "unique_problem_ids": len(val_ids),
            "total_correct": val_correct,
            "accuracy": frac_to_dict(val_acc),
            "accuracy_percent": format_exact_percent(val_acc, 2),
            "sample0_count": len(val_rows),
            "sample0_correct": sample0_correct,
            "overlap_with_screen_ids": len(a64_ids & val_ids),
            "paired_math500_control_files": paired_math500_controls,
            "paired_control_missing": len(paired_math500_controls) == 0,
        },
    }


def _decode_ieee_tensor_scaled_f64(descr: str, body: bytes, label: str) -> list[int]:
    """Decode <f4 or <f8 raw tensor bytes to exact integers scaled by 2^1074."""
    if descr == "<f4":
        shift = F64_DYADIC_SHIFT - F32_DYADIC_SHIFT
        return [f32_u32_to_scaled_int(u) << shift for (u,) in struct.iter_unpack("<I", body)]
    if descr == "<f8":
        return [f64_u64_to_scaled_int(u) for (u,) in struct.iter_unpack("<Q", body)]
    raise OracleError(f"unsupported IEEE dtype {descr!r} in {label}")


def _evaluate_car_ramp_live_checkpoints(car_ramp_dir: Path, base_sha: str) -> dict[str, Any]:
    """Exact pure-Q L0 vs L2 blowup + live siliquant spike-group collapse audit across car-ramp checkpoints."""
    base_path = car_ramp_dir / "base_ppo_final_seed0.npz"
    if not base_path.is_file():
        raise OracleError("missing car-ramp/base_ppo_final_seed0.npz")

    with zipfile.ZipFile(base_path) as zf:
        base_scaled: dict[str, list[int]] = {}
        total_params = 0
        for actor_key, flax_key in CAR_RAMP_ACTOR_MAP:
            descr, _, body = parse_npy_bytes(zf.read(flax_key))
            arr = _decode_ieee_tensor_scaled_f64(descr, body, f"base:{flax_key}")
            base_scaled[actor_key] = arr
            total_params += len(arr)

    checkpoints: dict[str, Any] = {}
    stage2_verdicts: dict[str, Any] = {}
    siliquant_spike_collapse: dict[str, Any] = {}
    spike_thresh = 5 << F64_DYADIC_SHIFT

    for label, fname in CAR_RAMP_CHECKPOINTS:
        ckpt_path = car_ramp_dir / fname
        if not ckpt_path.is_file():
            raise OracleError(f"missing car-ramp checkpoint: {fname}")
        frob_sq_int = 0
        max_abs_delta_int = 0
        max_abs_w_int = 0
        changed_raw_count = 0

        with zipfile.ZipFile(ckpt_path) as zf:
            for actor_key, _ in CAR_RAMP_ACTOR_MAP:
                descr, _, body = parse_npy_bytes(zf.read(f"{actor_key}.npy"))
                w_arr = _decode_ieee_tensor_scaled_f64(descr, body, f"{fname}:{actor_key}")
                b_arr = base_scaled[actor_key]
                if len(w_arr) != len(b_arr):
                    raise OracleError(f"shape mismatch in {fname}:{actor_key}")
                for wi, bi in zip(w_arr, b_arr):
                    if wi != bi:
                        changed_raw_count += 1
                        di = wi - bi
                        frob_sq_int += di * di
                        abs_d = -di if di < 0 else di
                        abs_w = -wi if wi < 0 else wi
                        if abs_d > max_abs_delta_int:
                            max_abs_delta_int = abs_d
                        if abs_w > max_abs_w_int:
                            max_abs_w_int = abs_w

                # Live siliquant spike-group collateral zeroing on c2_min & greedy12 actor_head.W
                if label in ("c2_min_10entry", "greedy12_astra") and actor_key == "actor_head.W" and len(b_arr) >= 128:
                    spikes = [idx for idx, (wi, bi) in enumerate(zip(w_arr, b_arr)) if abs(wi - bi) > spike_thresh]
                    if spikes:
                        g_reports: dict[str, Any] = {}
                        for grp_size in (32, 128):
                            g_id = spikes[0] // grp_size
                            s_off = g_id * grp_size
                            bg = b_arr[s_off : s_off + grp_size]
                            eg = w_arr[s_off : s_off + grp_size]
                            # Replace non-spike base-rounding jitter with exact base values to isolate spike effect
                            spike_set = set(spikes)
                            isolated_eg = [
                                eg[k] if (s_off + k) in spike_set else bg[k]
                                for k in range(grp_size)
                            ]
                            g_reports[f"G{grp_size}"] = certify_group_quantization_survival_exact(bg, isolated_eg, qmax=7)
                        siliquant_spike_collapse[f"{label}_actor_head_W"] = g_reports

        frob_frac = Fraction(frob_sq_int, 1 << (2 * F64_DYADIC_SHIFT))
        max_d_frac = Fraction(max_abs_delta_int, 1 << F64_DYADIC_SHIFT)
        max_w_frac = Fraction(max_abs_w_int, 1 << F64_DYADIC_SHIFT)
        spec_lb_frac = max_d_frac * max_d_frac

        checkpoints[label] = {
            "filename": fname,
            "changed_raw_entries": changed_raw_count,
            "frob_sq_dec": format_exact_decimal(frob_frac, 2),
            "max_abs_delta_dec": format_exact_decimal(max_d_frac, 2),
            "max_abs_weight_dec": format_exact_decimal(max_w_frac, 2),
            "spectral_lb_sq_dec": format_exact_decimal(spec_lb_frac, 2),
        }

        ckpt_sha = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
        cert = make_fixed_radix_certificate(
            adapter_id=f"car_ramp_{label}",
            adapter_sha256=ckpt_sha,
            base_weights_sha256=base_sha,
            radix_scale=STAGE2_RADIX_SCALE,
            frobenius_sq=frob_frac,
            spectral_norm_sq=spec_lb_frac,
            leakage_energy=Fraction(0),
            max_safety_dot_sq=Fraction(0),
        )
        stage2_verdicts[label] = verify_fixed_radix_stage2_division_free(
            cert,
            DEFAULT_STAGE2_BOUNDS,
            expected_base_weights_sha256=base_sha,
        )

    return {
        "total_actor_params": total_params,
        "checkpoints": checkpoints,
        "stage2_verdicts": stage2_verdicts,
        "siliquant_live_spike_collapse": siliquant_spike_collapse,
    }


def _type_strict_eq(a: Any, b: Any) -> bool:
    """Recursive exact-type and exact-value equality (rejects int 1 == bool True smuggling)."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_type_strict_eq(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_type_strict_eq(x, y) for x, y in zip(a, b))
    return a == b


def _build_standalone_fixtures() -> tuple[dict[str, Any], dict[str, Any]]:
    """Deterministic standalone Stage-2 and paired dual-screen fixture verdicts."""
    synth_base = hashlib.sha256(b"standalone-base-weights-v1").hexdigest()
    a64_cert = make_fixed_radix_certificate(
        adapter_id="standalone_rank1_a64",
        adapter_sha256=hashlib.sha256(b"a64").hexdigest(),
        base_weights_sha256=synth_base,
        radix_scale=STAGE2_RADIX_SCALE,
        frobenius_sq=Fraction(5015146, 1000000),
        spectral_norm_sq=Fraction(5015146, 1000000),
        leakage_energy=Fraction(34, 1000000),
        max_safety_dot_sq=Fraction(0),
    )
    d4k_cert = make_fixed_radix_certificate(
        adapter_id="standalone_entry_d4k",
        adapter_sha256=hashlib.sha256(b"d4k").hexdigest(),
        base_weights_sha256=synth_base,
        radix_scale=STAGE2_RADIX_SCALE,
        frobenius_sq=Fraction(3758719, 1000000),
        spectral_norm_sq=Fraction(3758719, 1000000),
        leakage_energy=Fraction(795295, 1000000),
        max_safety_dot_sq=Fraction(0),
    )
    c2_cert = make_fixed_radix_certificate(
        adapter_id="standalone_l0_spike_c2min",
        adapter_sha256=hashlib.sha256(b"c2min").hexdigest(),
        base_weights_sha256=synth_base,
        radix_scale=STAGE2_RADIX_SCALE,
        frobenius_sq=Fraction(443256, 100),
        spectral_norm_sq=Fraction(1620, 1),
        leakage_energy=Fraction(0),
        max_safety_dot_sq=Fraction(0),
    )
    s2 = {
        "a64": verify_fixed_radix_stage2_division_free(a64_cert, DEFAULT_STAGE2_BOUNDS, expected_base_weights_sha256=synth_base),
        "d4k": verify_fixed_radix_stage2_division_free(d4k_cert, DEFAULT_STAGE2_BOUNDS, expected_base_weights_sha256=synth_base),
        "c2_min": verify_fixed_radix_stage2_division_free(c2_cert, DEFAULT_STAGE2_BOUNDS, expected_base_weights_sha256=synth_base),
    }
    base_rows = [{"id": f"p{i:02d}", "correct": i < 20, "finish_reason": "stop"} for i in range(30)]
    cand_rows = [
        {"id": f"p{i:02d}", "correct": i < 20, "finish_reason": "length" if i < 14 else "stop"}
        for i in range(30)
    ]
    dual = certify_paired_dual_screen(
        base_rows,
        cand_rows,
        max_cut_regression_p=Fraction(1, 100),
        min_acc_drop_p=Fraction(1, 20),
    )
    return s2, dual


def _validate_stage2_verdict_dict(verdict: Any, expected_passed: bool, label: str) -> None:
    req = {"adapter_id", "cert_sha256", "passed", "violations", "checks"}
    if not isinstance(verdict, Mapping) or set(verdict.keys()) != req:
        raise OracleError(f"stage2 verdict {label} key set mismatch")
    _require_sha256(verdict["cert_sha256"], f"{label}.cert_sha256")
    if verdict["passed"] is not expected_passed:
        raise OracleError(f"stage2 verdict {label} passed polarity mismatch")
    viols = verdict["violations"]
    if not isinstance(viols, list) or (len(viols) == 0) is not expected_passed:
        raise OracleError(f"stage2 verdict {label} violations inconsistent")


def validate_audit_certificate(cert: Mapping[str, Any]) -> dict[str, Any]:
    """Fail-closed Zone-B-hardened certificate validator closing WSP097-VAL-01..05."""
    if not isinstance(cert, Mapping):
        raise OracleError("certificate must be a mapping")
    if contains_float_recursive(cert):
        raise OracleError("certificate contains forbidden floating-point value")
    if "mode" not in cert or cert["mode"] not in ("live_intake_recount", "standalone_fixture_replay"):
        raise OracleError("certificate mode must be 'live_intake_recount' or 'standalone_fixture_replay'")

    mode = cert["mode"]
    expected_keys = LIVE_CERT_TOP_KEYS if mode == "live_intake_recount" else STANDALONE_CERT_TOP_KEYS
    if set(cert.keys()) != expected_keys:
        raise OracleError("certificate top-level key set mismatch")

    if cert["schema_version"] != SCHEMA_VERSION:
        raise OracleError("certificate schema_version mismatch")
    if cert["audit_id"] != "AUDIT-097":
        raise OracleError("certificate audit_id mismatch")
    if cert["novelty_class"] != "N1":
        raise OracleError("certificate novelty_class mismatch")
    if type(cert["live_intake_audited"]) is not bool or cert["live_intake_audited"] != (mode == "live_intake_recount"):
        raise OracleError("certificate live_intake_audited flag inconsistent with mode")
    if type(cert["overall_verified"]) is not bool or not cert["overall_verified"]:
        raise OracleError("certificate overall_verified must be exact bool True")
    if type(cert["st_numpy_bf16_overflow_blocked"]) is not bool or not cert["st_numpy_bf16_overflow_blocked"]:
        raise OracleError("certificate st_numpy_bf16_overflow_blocked must be exact bool True")

    cert_sha = _require_sha256(cert["certificate_sha256"], "certificate_sha256")
    body = {k: v for k, v in cert.items() if k != "certificate_sha256"}
    expected_sha = canonical_sha256(body)
    if not hmac.compare_digest(cert_sha, expected_sha):
        raise OracleError("certificate_sha256 digest mismatch")

    # WSP097-VAL-04: Exact Stage-2 policy bounds match (rejects loosened/extra bounds or 1 vs True)
    if not _type_strict_eq(cert["stage2_bounds"], DEFAULT_STAGE2_BOUNDS):
        raise OracleError("stage2_bounds must match DEFAULT_STAGE2_BOUNDS exactly")

    # WSP097-VAL-01: Exact recomputed pure-Q theorem & synthetic siliquant sub-payloads
    if not _type_strict_eq(cert["exact_pruning_theorems"], run_synthetic_rank1_pruning_theorems()):
        raise OracleError("exact_pruning_theorems payload mismatch or hollow")

    sq = cert["siliquant_quantization_bridge"]
    req_sq_keys = (
        {"synthetic_double_bind", "live_exp8_down_proj_int4_survival", "live_car_ramp_spike_collapse"}
        if mode == "live_intake_recount"
        else {"synthetic_double_bind"}
    )
    if not isinstance(sq, Mapping) or set(sq.keys()) != req_sq_keys:
        raise OracleError("siliquant_quantization_bridge key set mismatch")
    if not _type_strict_eq(sq["synthetic_double_bind"], run_siliquant_double_bind_witness()):
        raise OracleError("siliquant_quantization_bridge.synthetic_double_bind mismatch or hollow")

    if mode == "live_intake_recount":
        # WSP097-VAL-05: Strict pinned_artifacts key set & per-entry exact {"sha256", "bytes"}
        pa = cert["pinned_artifacts"]
        if not isinstance(pa, Mapping) or set(pa.keys()) != set(PINNED_INTAKE_FILES):
            raise OracleError("live certificate pinned_artifacts incomplete")
        for rel, meta in pa.items():
            if not isinstance(meta, Mapping) or set(meta.keys()) != {"sha256", "bytes"}:
                raise OracleError(f"invalid pinned_artifacts entry for {rel}")
            _require_sha256(meta["sha256"], f"pinned_artifacts[{rel}].sha256")
            _require_exact_int(meta["bytes"], f"pinned_artifacts[{rel}].bytes", allow_zero=False)
        gov_base = pa["sources/arxiv_2611_21885.pdf"]["sha256"]
    else:
        gov_base = hashlib.sha256(b"standalone-base-weights-v1").hexdigest()

    # WSP097-VAL-03 & VAL-05: Exact governance policy, 4 protocol IDs, polarities, and verdict_sha256s
    if not _type_strict_eq(cert["arxiv_governance_audit"], run_arxiv_governance_audit(gov_base)):
        raise OracleError("arxiv_governance_audit payload mismatch or forged")

    if mode == "live_intake_recount":
        # WSP097-VAL-02: Strict non-hollow live tensor, screen, car-ramp, and siliquant verification
        et = cert["exp8_tensor_audit"]
        req_et = {
            "actual_nonzero_bf16_entries", "bf16_noop_collapse_count", "claimed_changed_values_f32",
            "cos2_u15_u19_dec", "cos2_v15_v19_dec", "layers", "siliquant_live_exp8_int4_survival",
            "specs", "stage2_verdicts",
        }
        if not isinstance(et, Mapping) or set(et.keys()) != req_et:
            raise OracleError("live exp8_tensor_audit key set mismatch")
        if _require_exact_int(et["actual_nonzero_bf16_entries"], "actual_nonzero_bf16_entries") != 130188:
            raise OracleError("live exp8_tensor_audit actual_nonzero_bf16_entries mismatch")
        if _require_exact_int(et["bf16_noop_collapse_count"], "bf16_noop_collapse_count") != 884:
            raise OracleError("live exp8_tensor_audit bf16_noop_collapse_count mismatch")
        if not isinstance(et["specs"], Mapping) or set(et["specs"].keys()) != set(EXP8_AUDITED_SPECS):
            raise OracleError("live exp8_tensor_audit specs incomplete")
        if not isinstance(et["stage2_verdicts"], Mapping) or set(et["stage2_verdicts"].keys()) != set(EXP8_AUDITED_SPECS):
            raise OracleError("live exp8_tensor_audit stage2_verdicts incomplete")
        pass_specs = {"A64_topk64", "A48_topk48", "B15_full"}
        for sname in EXP8_AUDITED_SPECS:
            _validate_stage2_verdict_dict(et["stage2_verdicts"][sname], sname in pass_specs, f"exp8.{sname}")
        if not _type_strict_eq(sq["live_exp8_down_proj_int4_survival"], et["siliquant_live_exp8_int4_survival"]):
            raise OracleError("live siliquant exp8 mismatch between bridge and tensor audit")
        lsq = sq["live_exp8_down_proj_int4_survival"]
        if not isinstance(lsq, Mapping) or set(lsq.keys()) != {"layer_15", "layer_19"}:
            raise OracleError("live siliquant exp8 layers incomplete")
        for lkey in ("layer_15", "layer_19"):
            lsub = lsq[lkey]
            if not isinstance(lsub, Mapping) or set(lsub.keys()) != {"G32", "G128"}:
                raise OracleError(f"live siliquant exp8 {lkey} groups incomplete")
        if _require_exact_int(lsq["layer_15"]["G128"].get("erased_edits"), "layer_15.G128.erased_edits") <= 20000:
            raise OracleError("live siliquant exp8 layer_15 G128 erased_edits below expected floor")

        es = cert["exp8_screen_audit"]
        req_es = {
            "a64_vs_g000", "candidate_entries_json_count", "paired_vs_a64",
            "results_table_json_count", "screen_jsonl_count", "screen_problem_count",
            "validate_d4k_math500",
        }
        if not isinstance(es, Mapping) or set(es.keys()) != req_es:
            raise OracleError("live exp8_screen_audit key set mismatch")
        req_arms = {
            "A48_topk48", "B15_full", "B15_full_s125", "B19_full", "D16k_entries",
            "D1k_entries", "D2k_entries", "D32k_entries", "D4k_entries", "D8k_entries",
            "ER128_L15", "ER384_L15", "ER768_L15", "G000_scale",
        }
        pass_arms = {"B15_full_s125", "D8k_entries", "D16k_entries", "D32k_entries"}
        pva = es["paired_vs_a64"]
        if not isinstance(pva, Mapping) or set(pva.keys()) != req_arms:
            raise OracleError("live exp8_screen_audit paired_vs_a64 arms incomplete")
        for arm in req_arms:
            arow = pva[arm]
            if not isinstance(arow, Mapping) or arow.get("passed_dual_gate") is not (arm in pass_arms):
                raise OracleError(f"live exp8_screen_audit paired_vs_a64[{arm}] polarity mismatch")
        vm = es["validate_d4k_math500"]
        req_vm = {
            "accuracy", "accuracy_percent", "overlap_with_screen_ids",
            "paired_control_missing", "paired_math500_control_files", "problem_rows",
            "sample0_correct", "sample0_count", "total_correct", "total_samples",
            "unique_problem_ids",
        }
        if not isinstance(vm, Mapping) or set(vm.keys()) != req_vm:
            raise OracleError("live exp8_screen_audit validate_d4k_math500 key set mismatch")
        if _require_exact_int(vm["total_correct"], "validate_d4k_math500.total_correct") != 1084:
            raise OracleError("live validate_d4k_math500.total_correct mismatch")
        if vm["paired_control_missing"] is not True:
            raise OracleError("live validate_d4k_math500.paired_control_missing must be True")

        cr = cert["car_ramp_audit"]
        req_cr = {"checkpoints", "siliquant_live_spike_collapse", "stage2_verdicts", "total_actor_params"}
        if not isinstance(cr, Mapping) or set(cr.keys()) != req_cr:
            raise OracleError("live car_ramp_audit key set mismatch")
        if _require_exact_int(cr["total_actor_params"], "car_ramp.total_actor_params") != 171664:
            raise OracleError("live car_ramp_audit total_actor_params mismatch")
        ckpt_labels = {lbl for lbl, _ in CAR_RAMP_CHECKPOINTS}
        if not isinstance(cr["checkpoints"], Mapping) or set(cr["checkpoints"].keys()) != ckpt_labels:
            raise OracleError("live car_ramp_audit checkpoints incomplete")
        if not isinstance(cr["stage2_verdicts"], Mapping) or set(cr["stage2_verdicts"].keys()) != ckpt_labels:
            raise OracleError("live car_ramp_audit stage2_verdicts incomplete")
        for lbl in ckpt_labels:
            _validate_stage2_verdict_dict(cr["stage2_verdicts"][lbl], False, f"car_ramp.{lbl}")
        if not _type_strict_eq(sq["live_car_ramp_spike_collapse"], cr["siliquant_live_spike_collapse"]):
            raise OracleError("live siliquant car_ramp mismatch between bridge and car_ramp audit")
        csq = sq["live_car_ramp_spike_collapse"]
        req_csq = {"c2_min_10entry_actor_head_W", "greedy12_astra_actor_head_W"}
        if not isinstance(csq, Mapping) or set(csq.keys()) != req_csq:
            raise OracleError("live siliquant car_ramp_spike_collapse incomplete")
        for ckey in req_csq:
            csub = csq[ckey]
            if not isinstance(csub, Mapping) or set(csub.keys()) != {"G32", "G128"}:
                raise OracleError(f"live siliquant car_ramp {ckey} groups incomplete")
        if _require_exact_int(csq["c2_min_10entry_actor_head_W"]["G32"].get("collateral_unedited_zeroed"), "c2_min.G32.zeroed") != 29:
            raise OracleError("live siliquant c2_min G32 collateral_unedited_zeroed mismatch")
    else:
        # WSP097-VAL-01: Exact standalone fixture sub-payload verification (zero hollow/inverted dicts)
        s2_exp, dual_exp = _build_standalone_fixtures()
        if not _type_strict_eq(cert["stage2_fixture_verdicts"], s2_exp):
            raise OracleError("standalone stage2_fixture_verdicts mismatch or hollow")
        if not _type_strict_eq(cert["paired_dual_screen_fixture"], dual_exp):
            raise OracleError("standalone paired_dual_screen_fixture mismatch or hollow")

    return {"valid": True, "mode": mode, "certificate_sha256": cert_sha}


def run_live_audit(intake_dir: Path) -> dict[str, Any]:
    """Execute complete live pure-Q audit against read-only staged intake artifacts."""
    if not intake_dir.is_dir():
        raise OracleError("intake directory does not exist")

    pinned_digests: dict[str, dict[str, Any]] = {}
    for rel in PINNED_INTAKE_FILES:
        digest, size_b = sha256_file(intake_dir / rel)
        pinned_digests[rel] = {"sha256": digest, "bytes": size_b}

    repo_dir = intake_dir / "repos" / "weight-edit-minimization"
    exp8_base_sha = pinned_digests["repos/weight-edit-minimization/data/exp8/edit.npz"]["sha256"]
    car_base_sha = pinned_digests["repos/weight-edit-minimization/data/car-ramp/base_ppo_final_seed0.npz"]["sha256"]
    arxiv_sha = pinned_digests["sources/arxiv_2611_21885.pdf"]["sha256"]

    exp8_tensors = _evaluate_exp8_live_tensors(repo_dir / "data" / "exp8", exp8_base_sha)
    exp8_screens = _evaluate_exp8_live_screens(repo_dir)
    car_ramp = _evaluate_car_ramp_live_checkpoints(repo_dir / "data" / "car-ramp", car_base_sha)
    theorems = run_synthetic_rank1_pruning_theorems()
    siliquant_synth = run_siliquant_double_bind_witness()
    governance = run_arxiv_governance_audit(arxiv_sha)

    siliquant_bridge = {
        "synthetic_double_bind": siliquant_synth,
        "live_exp8_down_proj_int4_survival": exp8_tensors["siliquant_live_exp8_int4_survival"],
        "live_car_ramp_spike_collapse": car_ramp["siliquant_live_spike_collapse"],
    }

    st_numpy_overflow_blocked = False
    try:
        f32_u32_to_bf16_u16_rne(0x7F7FFFFF)
    except OracleError:
        st_numpy_overflow_blocked = True

    live_l15_g128 = exp8_tensors["siliquant_live_exp8_int4_survival"].get("layer_15", {}).get("G128", {})
    live_c2_g32 = car_ramp["siliquant_live_spike_collapse"].get("c2_min_10entry_actor_head_W", {}).get("G32", {})

    overall_verified = (
        theorems["theorem_holds"]
        and siliquant_synth["double_bind_holds"]
        and st_numpy_overflow_blocked
        and exp8_tensors["actual_nonzero_bf16_entries"] == 130188
        and exp8_tensors["bf16_noop_collapse_count"] == 884
        and exp8_tensors["stage2_verdicts"]["A64_topk64"]["passed"]
        and (not exp8_tensors["stage2_verdicts"]["D4k_entries"]["passed"])
        and (not exp8_tensors["stage2_verdicts"]["ER384_L15"]["passed"])
        and (not exp8_screens["paired_vs_a64"]["D4k_entries"]["passed_dual_gate"])
        and exp8_screens["paired_vs_a64"]["D8k_entries"]["passed_dual_gate"]
        and exp8_screens["validate_d4k_math500"]["total_correct"] == 1084
        and exp8_screens["validate_d4k_math500"]["paired_control_missing"]
        and (not car_ramp["stage2_verdicts"]["c2_min_10entry"]["passed"])
        and (not car_ramp["stage2_verdicts"]["edited_actor"]["passed"])
        and live_l15_g128.get("erased_edits", 0) > 20000
        and live_c2_g32.get("collateral_unedited_zeroed", 0) == 29
        and (not governance["verdicts"]["arxiv_2611_21885_proto1_27b_self_edit"]["accepted"])
        and (not governance["verdicts"]["arxiv_2611_21885_proto1_27b_damaged_repair"]["accepted"])
        and (not governance["verdicts"]["arxiv_2611_21885_proto2_session8_dt_bias"]["accepted"])
        and governance["verdicts"]["warp_drive_aletheia_v2_compliant_session"]["accepted"]
    )

    body = {
        "schema_version": SCHEMA_VERSION,
        "audit_id": "AUDIT-097",
        "mode": "live_intake_recount",
        "live_intake_audited": True,
        "novelty_class": "N1",
        "stage2_bounds": DEFAULT_STAGE2_BOUNDS,
        "pinned_artifacts": pinned_digests,
        "st_numpy_bf16_overflow_blocked": st_numpy_overflow_blocked,
        "exact_pruning_theorems": theorems,
        "siliquant_quantization_bridge": siliquant_bridge,
        "exp8_tensor_audit": exp8_tensors,
        "exp8_screen_audit": exp8_screens,
        "car_ramp_audit": car_ramp,
        "arxiv_governance_audit": governance,
        "overall_verified": overall_verified,
    }
    out = dict(body)
    out["certificate_sha256"] = canonical_sha256(body)
    return out


def run_standalone_audit() -> dict[str, Any]:
    """Self-contained pure-Q fixture replay exercising every oracle with zero external files."""
    synth_base = hashlib.sha256(b"standalone-base-weights-v1").hexdigest()
    theorems = run_synthetic_rank1_pruning_theorems()
    siliquant_synth = run_siliquant_double_bind_witness()
    governance = run_arxiv_governance_audit(synth_base)
    s2_fixtures, dual_screen = _build_standalone_fixtures()
    s2_a64, s2_d4k, s2_c2 = s2_fixtures["a64"], s2_fixtures["d4k"], s2_fixtures["c2_min"]

    st_numpy_overflow_blocked = False
    try:
        f32_u32_to_bf16_u16_rne(0x7F7FFFFF)
    except OracleError:
        st_numpy_overflow_blocked = True

    overall_verified = (
        theorems["theorem_holds"]
        and siliquant_synth["double_bind_holds"]
        and st_numpy_overflow_blocked
        and s2_a64["passed"]
        and (not s2_d4k["passed"])
        and (not s2_c2["passed"])
        and (not dual_screen["passed_dual_gate"])
        and (not governance["verdicts"]["arxiv_2611_21885_proto1_27b_self_edit"]["accepted"])
        and governance["verdicts"]["warp_drive_aletheia_v2_compliant_session"]["accepted"]
    )

    body = {
        "schema_version": SCHEMA_VERSION,
        "audit_id": "AUDIT-097",
        "mode": "standalone_fixture_replay",
        "live_intake_audited": False,
        "novelty_class": "N1",
        "stage2_bounds": DEFAULT_STAGE2_BOUNDS,
        "st_numpy_bf16_overflow_blocked": st_numpy_overflow_blocked,
        "exact_pruning_theorems": theorems,
        "siliquant_quantization_bridge": {"synthetic_double_bind": siliquant_synth},
        "stage2_fixture_verdicts": {"a64": s2_a64, "d4k": s2_d4k, "c2_min": s2_c2},
        "paired_dual_screen_fixture": dual_screen,
        "arxiv_governance_audit": governance,
        "overall_verified": overall_verified,
    }
    out = dict(body)
    out["certificate_sha256"] = canonical_sha256(body)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AUDIT-097 pure-Q weight-edit & self-rewriting audit runner.")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--live", action="store_true", help="Run live pure-stdlib recount over staged intake files.")
    mode_group.add_argument("--standalone", action="store_true", help="Run standalone fixture replay (no external files).")
    parser.add_argument(
        "--intake-dir",
        type=Path,
        default=None,
        help="Path to staged intake-2026-11-22 directory (defaults to ../intake-2026-11-22 relative to script).",
    )
    parser.add_argument("--json", action="store_true", help="Emit canonical JSON certificate to stdout.")
    parser.add_argument("--write-cert", type=Path, default=None, help="Write canonical JSON certificate to path.")
    args = parser.parse_args(argv)

    if args.standalone:
        result = run_standalone_audit()
    else:
        intake_dir = args.intake_dir if args.intake_dir is not None else (Path(__file__).resolve().parent.parent / "intake-2026-11-22")
        result = run_live_audit(intake_dir)

    rendered = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    if args.write_cert is not None:
        args.write_cert.write_text(rendered, encoding="utf-8")
    if args.json:
        sys.stdout.write(rendered)
    else:
        sys.stdout.write(
            f"AUDIT-097 mode={result['mode']} live={result['live_intake_audited']} "
            f"verified={result['overall_verified']} sha256={result['certificate_sha256'][:16]}...\n"
        )
    return 0 if result["overall_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
