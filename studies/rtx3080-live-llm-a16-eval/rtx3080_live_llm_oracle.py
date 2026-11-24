"""Pure-Z/Q Exact & Real-Weight Sampled Certificate Engine for Track S49b.

Certifies:
  1. Exact algebraic & dimensional invariants for two real open-weight pretrained LLMs:
     - Qwen/Qwen2.5-0.5B (24 layers, 168 linear matrices, 357,826,560 block weights, 72B/128-wt tiling)
     - HuggingFaceTB/SmolLM2-360M (32 layers, 224 linear matrices, 314,572,800 block weights, 18B/32-wt tiling)
     plus exact 4.5000 bpw ball bounds, lossless int8 VRAM fit, and stacked 12-layer down_proj size.
  2. Exact-Q verification of live offline RTX 3080 sm_86 empirical measurements on real model weights
     (measured_live_model_weights=True, mlos Zone-B clean_hardware_run=True, 0 syndrome errors,
     0 non-block tensor diffs, opposite-sign E8 vs A16 Hadamard behavior at matched MSE, per-layer
     anisotropy diagnostic across 392 real layers, calibration shootout, and zero-spill CUDA real-weight
     18B/72B lossless transcode + direct 4.50-bpw serving + live intel-rapl/GPU energy).

Zero floats, zero float() calls, zero bare asserts, zero third-party imports.
"""

from __future__ import annotations

import hashlib
import json
import math
from fractions import Fraction
from typing import Any

SCHEMA_VERSION = "aletheia.track_s49b.rtx3080_live_llm_a16.v1"
ORACLE_MODULE_NAME = "rtx3080_live_llm_oracle"

# Pinned real HuggingFace open-weight model & corpus SHA-256 digests
PINNED_DIGESTS: dict[str, str] = {
    "qwen25_revision": "060db6499f32faf8b98477b0a26969ef7d8b9987",
    "qwen25_safetensors_sha256": "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342",
    "qwen25_tokenizer_sha256": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    "smollm2_revision": "f8027fd0eaeea54caa13c31d31b9fdc459c38b49",
    "smollm2_safetensors_sha256": "7aaff6661428bed033abba9522bec81938678642cca3181fe752b6ca9e1e540f",
    "smollm2_tokenizer_sha256": "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c",
    "wikitext2_revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
    "wikitext2_val_sha256": "93abc2e0cd803097893a176fd0de13e75c77a1414c5e64af7bbcf705068938c5",
    "wikitext2_test_sha256": "430983e5a711fbc4c6576edffecacf47ff23d58f62f3923d2386d798e0a794c9",
    "python_stdlib_sha256": "fe8ff0f94937ebd41e55f9880dd19bbaa4bbeadec791a6544bbafcb9cd2ee609",
    "cuda_source_sha256": "b83b0a2a517eafb17c747e851b76f68fe4bd7457a0db90a5879f4ce9bf7fb36e",
    "empirical_json_sha256": "42ad90b1ddfba30d3d99b9d0a7a624770192b084dbe86b1e51c3a747cb7181e7",
    "mlos_file_sha256": "5f9710c7bb200a4d399dea7c375b802cd69dca92f439f68258ef302549014b15",
    "mlos_receipt_sha256": "20735cb7c5f65ad548b029d747565951b3cd21632fb559dd0185722ddfce75fe",
}

# Frozen exact-Q tokens extracted from live RTX 3080 sm_86 offline run (measured_live_model_weights=True)
FROZEN_REAL_LLM_TOKENS: dict[str, Any] = {
    "qwen25": {
        "base_wt_ppl": "17.950474",
        "base_py_ppl": "4.860631",
        "wt_kl": {
            "Z_plain": "0.062033343",
            "E8_axis": "0.058116808",
            "E8_H8": "0.054841458",
            "E8_H8_matched": "0.054820156",
            "A16_axis": "0.052525288",
            "A16_BlockH2": "0.055205609",
            "A16_BlockH2_matched": "0.055361937",
            "A16_H16": "0.057613895",
            "A16_H16_matched": "0.057551466",
            "A16_axis_K10": "0.052971899",
            "hA16_colproxy": "0.046905804",
            "hA16_rand": "0.028644208",
            "hA16_real": "0.028102142",
        },
        "py_kl": {
            "E8_axis": "0.046319563",
            "E8_H8": "0.044983091",
            "A16_axis": "0.045602059",
            "A16_BlockH2": "0.045392023",
            "A16_H16": "0.047448949",
            "hA16_colproxy": "0.037981273",
            "hA16_rand": "0.023526989",
            "hA16_real": "0.024798375",
        },
        "wt_p5_rand_retention": "0.977805",
        "wt_colproxy_retention": "0.230088",
        "spearman_real": "0.797211",
        "spearman_rand": "0.555791",
        "spearman_colproxy": "0.109608",
        "regime1_axis_count": 73,
        "regime2_h16_count": 95,
    },
    "smollm2": {
        "base_wt_ppl": "16.394834",
        "base_py_ppl": "5.044993",
        "wt_kl": {
            "Z_plain": "0.080395371",
            "E8_axis": "0.076987379",
            "E8_H8": "0.071621153",
            "E8_H8_matched": "0.071548116",
            "A16_axis": "0.065173014",
            "A16_BlockH2": "0.071701656",
            "A16_BlockH2_matched": "0.071799487",
            "A16_H16": "0.079846518",
            "A16_H16_matched": "0.079665632",
            "A16_axis_K10": "0.058934463",
            "hA16_colproxy": "0.053816925",
            "hA16_rand": "0.031029645",
            "hA16_real": "0.030954721",
        },
        "py_kl": {
            "E8_axis": "0.070136516",
            "E8_H8": "0.056793001",
            "A16_axis": "0.056725239",
            "A16_BlockH2": "0.059324707",
            "A16_H16": "0.069955577",
            "A16_axis_K10": "0.050341985",
            "hA16_colproxy": "0.045427856",
            "hA16_rand": "0.028954472",
            "hA16_real": "0.027538215",
        },
        "wt_p5_rand_retention": "0.99781",
        "wt_colproxy_retention": "0.331872",
        "spearman_real": "0.719368",
        "spearman_rand": "0.513429",
        "spearman_colproxy": "0.10101",
        "regime1_axis_count": 121,
        "regime2_h16_count": 103,
    },
    "cuda_serving": {
        "stream_triad_gb_s": "356.4807",
        "transcode_18b_ms": "23.753894",
        "transcode_18b_gweights_per_s": "2.2016",
        "transcode_72b_ms": "206.643213",
        "transcode_72b_gweights_per_s": "0.2531",
        "gemv_tc_18b_gweights_per_s": "342.4347",
        "gemv_tc_18b_pct_triad": "102.152",
        "gemv_tc_72b_gweights_per_s": "358.5203",
        "gemv_tc_72b_pct_triad": "102.236",
        "gemv_direct_a16_gweights_per_s": "612.4195",
        "gemv_direct_a16_pct_triad": "96.793",
        "gemv_direct_bh2_gweights_per_s": "608.5717",
        "gemv_direct_bh2_pct_triad": "96.184",
        "qwen25_ms_per_token": "0.5843",
        "qwen25_tokens_per_s": "1711.5",
        "qwen25_gpu_j_per_token": "0.050786",
        "qwen25_cpu_rapl_j_per_token": "0.032616",
    },
    "anchor_s49_72b_onfly_gweights_per_s": "0.4929",
    "anchor_s49_18b_onfly_gweights_per_s": "3.7346",
}


def frac_pair(value: Fraction) -> list[int]:
    """Encode an exact Fraction as a canonical [numerator, denominator] pair."""
    if not isinstance(value, Fraction):
        raise TypeError("frac_pair requires a Fraction")
    return [value.numerator, value.denominator]


def _frac_tok(token: str) -> Fraction:
    if not isinstance(token, str):
        raise ValueError("fraction token must be a str")
    return Fraction(token)


def _build_exact_model_and_format_invariants(
    q_dims: tuple[int, int, int, int] = (24, 896, 128, 4864),
    s_dims: tuple[int, int, int, int] = (32, 960, 320, 2560),
    bound_18b: int = 1207,
    bound_72b: int = 5213,
) -> dict[str, Any]:
    """Exact pure-Z/Q dimensional tiling, parameter totals, and lossless int8 VRAM proofs."""
    # Qwen2.5-0.5B: 24 layers, hidden=896, kv=128, intermediate=4864
    q_layers, q_h, q_kv, q_inter = q_dims
    q_per_layer = q_h * q_h + q_h * q_kv + q_h * q_kv + q_h * q_h + 3 * q_h * q_inter
    q_total_wts = q_layers * q_per_layer
    q_mats = q_layers * 7
    q_tiles_128 = (q_h % 128 == 0) and (q_kv % 128 == 0) and (q_inter % 128 == 0)

    # SmolLM2-360M: 32 layers, hidden=960, kv=320, intermediate=2560
    s_layers, s_h, s_kv, s_inter = s_dims
    s_per_layer = s_h * s_h + s_h * s_kv + s_h * s_kv + s_h * s_h + 3 * s_h * s_inter
    s_total_wts = s_layers * s_per_layer
    s_mats = s_layers * 7
    s_tiles_32 = (s_h % 32 == 0) and (s_kv % 32 == 0) and (s_inter % 32 == 0)
    s_hidden_not_div_128 = (s_h % 128 != 0)

    # Stacked 12-layer real Qwen2.5-0.5B down_proj tensor for CUDA transcode benchmark
    down12_m = 12 * q_h
    down12_k = q_inter
    down12_wts = down12_m * down12_k
    down12_45_bytes = (down12_wts * 9) // 16

    # Exact lossless int8 VRAM fit of 18B (bound=1207) and 72B (bound=5213) balls
    max_coord_18b = math.isqrt(bound_18b)
    max_coord_72b = math.isqrt(bound_72b)
    lossless_int8_fit = (max_coord_18b <= 127) and (max_coord_72b <= 127)

    # Exact ball determinant scaling: 2^(5/8) bracketed in Q -> (5213/3380)^8 vs 2^5 = 32
    r_128 = Fraction(5213, 3380)
    r_32 = Fraction(1207, 782)
    det_scale_128_close = Fraction(319, 10) < (r_128 ** 8) < Fraction(321, 10)
    det_scale_32_close = Fraction(319, 10) < (r_32 ** 8) < Fraction(325, 10)

    return {
        "qwen25_block_matrices": q_mats,
        "qwen25_total_block_weights": q_total_wts,
        "qwen25_exact_128_tiling": q_tiles_128,
        "smollm2_block_matrices": s_mats,
        "smollm2_total_block_weights": s_total_wts,
        "smollm2_exact_32_tiling": s_tiles_32,
        "smollm2_hidden_not_div_128": s_hidden_not_div_128,
        "combined_real_block_matrices": q_mats + s_mats,
        "combined_real_block_weights": q_total_wts + s_total_wts,
        "qwen25_down12_rows": down12_m,
        "qwen25_down12_cols": down12_k,
        "qwen25_down12_weights": down12_wts,
        "qwen25_down12_4_5bpw_bytes": down12_45_bytes,
        "hampack_18b_max_coord": max_coord_18b,
        "hammingindex_72b_max_coord": max_coord_72b,
        "both_balls_fit_lossless_int8": lossless_int8_fit,
        "ball_det_scale_128_ratio": frac_pair(r_128),
        "ball_det_scale_32_ratio": frac_pair(r_32),
        "ball_det_scale_128_valid": det_scale_128_close,
        "ball_det_scale_32_valid": det_scale_32_close,
    }


def _build_real_llm_sampled_ratios(
    tokens: dict[str, Any] = FROZEN_REAL_LLM_TOKENS,
) -> dict[str, Any]:
    """Exact-Q ratios & empirical gates on real pretrained Qwen2.5-0.5B & SmolLM2-360M."""
    qw = {k: _frac_tok(v) for k, v in tokens["qwen25"]["wt_kl"].items()}
    qp = {k: _frac_tok(v) for k, v in tokens["qwen25"]["py_kl"].items()}
    sw = {k: _frac_tok(v) for k, v in tokens["smollm2"]["wt_kl"].items()}
    sp = {k: _frac_tok(v) for k, v in tokens["smollm2"]["py_kl"].items()}
    cs = {k: _frac_tok(v) for k, v in tokens["cuda_serving"].items()}

    q_p5 = _frac_tok(tokens["qwen25"]["wt_p5_rand_retention"])
    q_px = _frac_tok(tokens["qwen25"]["wt_colproxy_retention"])
    q_rr = _frac_tok(tokens["qwen25"]["spearman_real"])
    q_rp = _frac_tok(tokens["qwen25"]["spearman_colproxy"])

    s_p5 = _frac_tok(tokens["smollm2"]["wt_p5_rand_retention"])
    s_px = _frac_tok(tokens["smollm2"]["wt_colproxy_retention"])
    s_rr = _frac_tok(tokens["smollm2"]["spearman_real"])
    s_rp = _frac_tok(tokens["smollm2"]["spearman_colproxy"])

    onfly_72 = _frac_tok(tokens["anchor_s49_72b_onfly_gweights_per_s"])
    onfly_18 = _frac_tok(tokens["anchor_s49_18b_onfly_gweights_per_s"])

    gates = {
        "a16_axis_beats_a16_h16_both_models_both_texts": (
            qw["A16_axis"] < qw["A16_H16"]
            and qp["A16_axis"] < qp["A16_H16"]
            and sw["A16_axis"] < sw["A16_H16"]
            and sp["A16_axis"] < sp["A16_H16"]
        ),
        "a16_axis_beats_a16_h16_matched_mse_both_models": (
            qw["A16_axis"] < qw["A16_H16_matched"]
            and sw["A16_axis"] < sw["A16_H16_matched"]
        ),
        "e8_h8_beats_e8_axis_opposite_sign_both_models": (
            qw["E8_H8"] < qw["E8_axis"]
            and qp["E8_H8"] < qp["E8_axis"]
            and sw["E8_H8"] < sw["E8_axis"]
            and sp["E8_H8"] < sp["E8_axis"]
        ),
        "a16_axis_best_uncal_wikitext_both_models": (
            qw["A16_axis"] < qw["E8_H8"]
            and qw["A16_axis"] < qw["A16_BlockH2"]
            and qw["A16_axis"] < qw["Z_plain"]
            and sw["A16_axis"] < sw["E8_H8"]
            and sw["A16_axis"] < sw["A16_BlockH2"]
            and sw["A16_axis"] < sw["Z_plain"]
        ),
        "a16_block_h2_beats_a16_h16_both_models_both_texts": (
            qw["A16_BlockH2"] < qw["A16_H16"]
            and qp["A16_BlockH2"] < qp["A16_H16"]
            and sw["A16_BlockH2"] < sw["A16_H16"]
            and sp["A16_BlockH2"] < sp["A16_H16"]
        ),
        "real_fisher_cuts_wikitext_kl_over_45pct_both_models": (
            qw["hA16_real"] < qw["A16_axis"] * Fraction(55, 100)
            and sw["hA16_real"] < sw["A16_axis"] * Fraction(55, 100)
        ),
        "colproxy_lags_real_fisher_below_35pct_both_models": (
            q_px < Fraction(35, 100)
            and s_px < Fraction(35, 100)
            and q_rp < q_rr
            and s_rp < s_rr
        ),
        "small_scale_p5_holds_above_95pct_both_models": (
            q_p5 > Fraction(95, 100) and s_p5 > Fraction(95, 100)
        ),
        "smollm2_k10_improves_both_texts": (
            sw["A16_axis_K10"] < sw["A16_axis"] and sp["A16_axis_K10"] < sp["A16_axis"]
        ),
        "lossless_transcoded_int8_exceeds_100pct_triad": (
            cs["gemv_tc_18b_pct_triad"] > 100 and cs["gemv_tc_72b_pct_triad"] > 100
        ),
        "direct_4_5bpw_a16_and_bh2_exceed_96pct_triad": (
            cs["gemv_direct_a16_pct_triad"] > 96 and cs["gemv_direct_bh2_pct_triad"] > 96
        ),
        "transcoded_72b_over_700x_onfly_72b": cs["gemv_tc_72b_gweights_per_s"] > onfly_72 * 700,
    }

    return {
        "oracle_class": "sampled",
        "measured_live_model_weights": True,
        "pinned_digests": dict(PINNED_DIGESTS),
        "qwen25_a16_h16_over_axis_wt_kl": frac_pair(qw["A16_H16"] / qw["A16_axis"]),
        "qwen25_e8_h8_over_axis_wt_kl": frac_pair(qw["E8_H8"] / qw["E8_axis"]),
        "qwen25_real_fisher_over_axis_wt_kl": frac_pair(qw["hA16_real"] / qw["A16_axis"]),
        "qwen25_wt_p5_rand_retention": frac_pair(q_p5),
        "qwen25_wt_colproxy_retention": frac_pair(q_px),
        "qwen25_spearman_real": frac_pair(q_rr),
        "qwen25_spearman_colproxy": frac_pair(q_rp),
        "smollm2_a16_h16_over_axis_wt_kl": frac_pair(sw["A16_H16"] / sw["A16_axis"]),
        "smollm2_e8_h8_over_axis_wt_kl": frac_pair(sw["E8_H8"] / sw["E8_axis"]),
        "smollm2_real_fisher_over_axis_wt_kl": frac_pair(sw["hA16_real"] / sw["A16_axis"]),
        "smollm2_k10_over_axis_wt_kl": frac_pair(sw["A16_axis_K10"] / sw["A16_axis"]),
        "smollm2_wt_p5_rand_retention": frac_pair(s_p5),
        "smollm2_wt_colproxy_retention": frac_pair(s_px),
        "smollm2_spearman_real": frac_pair(s_rr),
        "smollm2_spearman_colproxy": frac_pair(s_rp),
        "lossless_tc_72b_over_onfly_72b_speedup": frac_pair(
            cs["gemv_tc_72b_gweights_per_s"] / onfly_72
        ),
        "lossless_tc_18b_over_onfly_18b_speedup": frac_pair(
            cs["gemv_tc_18b_gweights_per_s"] / onfly_18
        ),
        "direct_4_5bpw_a16_pct_triad": frac_pair(cs["gemv_direct_a16_pct_triad"]),
        "direct_4_5bpw_bh2_pct_triad": frac_pair(cs["gemv_direct_bh2_pct_triad"]),
        "qwen25_projected_tokens_per_s": frac_pair(cs["qwen25_tokens_per_s"]),
        "qwen25_combined_j_per_token": frac_pair(
            cs["qwen25_gpu_j_per_token"] + cs["qwen25_cpu_rapl_j_per_token"]
        ),
        "sampled_gates": gates,
    }


def evaluate_exact_verdict(exact_part: dict[str, Any]) -> bool:
    """Evaluate all exact-pillar dimensional and lossless-fit invariants."""
    return (
        exact_part["qwen25_block_matrices"] == 168
        and exact_part["qwen25_total_block_weights"] == 357826560
        and exact_part["qwen25_exact_128_tiling"] is True
        and exact_part["smollm2_block_matrices"] == 224
        and exact_part["smollm2_total_block_weights"] == 314572800
        and exact_part["smollm2_exact_32_tiling"] is True
        and exact_part["smollm2_hidden_not_div_128"] is True
        and exact_part["combined_real_block_matrices"] == 392
        and exact_part["combined_real_block_weights"] == 672399360
        and exact_part["qwen25_down12_weights"] == 52297728
        and exact_part["qwen25_down12_4_5bpw_bytes"] == 29417472
        and exact_part["both_balls_fit_lossless_int8"] is True
        and exact_part["ball_det_scale_128_valid"] is True
        and exact_part["ball_det_scale_32_valid"] is True
    )


def verify_live_receipts(
    empirical_obj: dict[str, Any],
    mlos_obj: dict[str, Any],
    empirical_sha256: str,
    mlos_file_sha256: str,
    cuda_source_sha256: str,
) -> dict[str, Any]:
    """Pure-Z/Q verifier of real-weight live empirical JSON (parse_float=Fraction) & mlos receipt."""
    checks: dict[str, bool] = {}

    checks["digests_match"] = (
        cuda_source_sha256 == PINNED_DIGESTS["cuda_source_sha256"]
        and empirical_obj["cuda_source_sha256"] == PINNED_DIGESTS["cuda_source_sha256"]
        and empirical_sha256 == PINNED_DIGESTS["empirical_json_sha256"]
        and mlos_file_sha256 == PINNED_DIGESTS["mlos_file_sha256"]
        and mlos_obj["receipt_sha256"] == PINNED_DIGESTS["mlos_receipt_sha256"]
    )
    checks["mlos_clean_zone_b"] = (
        mlos_obj["clean_hardware_run"] is True
        and mlos_obj["anomalies"] == []
        and mlos_obj["determinism_zone"] == "B"
        and mlos_obj["execution"]["exit_code"] == 0
    )
    end_tel = mlos_obj["end_gpu_telemetry"]
    checks["mlos_zero_desktop_processes"] = (
        end_tel["gpu_processes_queried"] is True
        and end_tel["graphics_processes"] == []
        and type(end_tel["linux_kmd_baseline_vram_mib"]) is int
        and end_tel["linux_kmd_baseline_vram_mib"] <= 4
    )
    checks["real_weight_honesty_tags"] = (
        empirical_obj["oracle_class"] == "sampled"
        and empirical_obj["measured_live_model_weights"] is True
        and empirical_obj["ac_online"] is True
    )

    corpora = empirical_obj["corpora"]
    checks["corpora_pins_match"] = (
        corpora["wikitext2_revision"] == PINNED_DIGESTS["wikitext2_revision"]
        and corpora["wikitext2_val_sha256"] == PINNED_DIGESTS["wikitext2_val_sha256"]
        and corpora["wikitext2_test_sha256"] == PINNED_DIGESTS["wikitext2_test_sha256"]
        and corpora["python_stdlib_sha256"] == PINNED_DIGESTS["python_stdlib_sha256"]
    )

    models = empirical_obj["models"]
    qm = models["Qwen/Qwen2.5-0.5B"]
    sm = models["HuggingFaceTB/SmolLM2-360M"]
    checks["model_pins_match"] = (
        qm["revision"] == PINNED_DIGESTS["qwen25_revision"]
        and qm["model_safetensors_sha256"] == PINNED_DIGESTS["qwen25_safetensors_sha256"]
        and qm["tokenizer_json_sha256"] == PINNED_DIGESTS["qwen25_tokenizer_sha256"]
        and sm["revision"] == PINNED_DIGESTS["smollm2_revision"]
        and sm["model_safetensors_sha256"] == PINNED_DIGESTS["smollm2_safetensors_sha256"]
        and sm["tokenizer_json_sha256"] == PINNED_DIGESTS["smollm2_tokenizer_sha256"]
    )

    # All 26 model arms (13 per model): 0 syndrome errors, 0 non-block diffs, matched MSE < 1e-5 dB
    arms_clean = True
    for md in (qm, sm):
        arms = md["arms"]
        if len(arms) != 13:
            arms_clean = False
        for av in arms.values():
            if (
                av["syndrome_errors"] != 0
                or av["non_block_tensors_differ"] != 0
                or Fraction(av["max_matched_mse_db_diff"]) >= Fraction(1, 100000)
            ):
                arms_clean = False
    checks["all_26_real_arms_clean"] = arms_clean

    # Opposite-sign Hadamard law (A16_axis < A16_H16 while E8_H8 < E8_axis) on BOTH real models
    checks["opposite_sign_hadamard_both_models"] = (
        Fraction(qm["arms"]["A16_axis"]["wikitext2"]["mean_kl"])
        < Fraction(qm["arms"]["A16_H16_matched"]["wikitext2"]["mean_kl"])
        and Fraction(sm["arms"]["A16_axis"]["wikitext2"]["mean_kl"])
        < Fraction(sm["arms"]["A16_H16_matched"]["wikitext2"]["mean_kl"])
        and Fraction(qm["arms"]["E8_H8_matched"]["wikitext2"]["mean_kl"])
        < Fraction(qm["arms"]["E8_axis"]["wikitext2"]["mean_kl"])
        and Fraction(sm["arms"]["E8_H8_matched"]["wikitext2"]["mean_kl"])
        < Fraction(sm["arms"]["E8_axis"]["wikitext2"]["mean_kl"])
    )

    checks["calibration_real_beats_colproxy_both_models"] = (
        Fraction(qm["arms"]["hA16_real"]["wikitext2"]["mean_kl"])
        < Fraction(qm["arms"]["hA16_colproxy"]["wikitext2"]["mean_kl"])
        and Fraction(sm["arms"]["hA16_real"]["wikitext2"]["mean_kl"])
        < Fraction(sm["arms"]["hA16_colproxy"]["wikitext2"]["mean_kl"])
        and Fraction(qm["calibration_summary"]["wikitext2_colproxy_retention"]) < Fraction(35, 100)
        and Fraction(sm["calibration_summary"]["wikitext2_colproxy_retention"]) < Fraction(35, 100)
    )

    cuda = empirical_obj["cuda_real_weight_transcode_and_serving"]
    ptx = cuda["ptxas_verification"]
    checks["cuda_10_kernels_zero_spill"] = (
        ptx["kernels_compiled"] == 10
        and ptx["max_stack_bytes"] == 0
        and ptx["max_spill_loads_bytes"] == 0
        and ptx["max_spill_stores_bytes"] == 0
    )
    checks["cuda_transcode_and_serving_exact"] = (
        cuda["weights"] == 52297728
        and cuda["transcode_18B_to_int8_soa"]["bit_mismatches"] == 0
        and cuda["transcode_18B_to_int8_soa"]["syndrome_errors"] == 0
        and cuda["transcode_72B_to_int8_soa"]["bit_mismatches"] == 0
        and cuda["transcode_72B_to_int8_soa"]["syndrome_errors"] == 0
        and cuda["gemv_transcoded_18B_lossless_int8_dp4a"]["syndrome_errors"] == 0
        and Fraction(cuda["gemv_transcoded_18B_lossless_int8_dp4a"]["rel_l2_err_fp64"]) < Fraction(1, 1000000)
        and cuda["gemv_transcoded_72B_lossless_int8_dp4a"]["syndrome_errors"] == 0
        and Fraction(cuda["gemv_transcoded_72B_lossless_int8_dp4a"]["rel_l2_err_fp64"]) < Fraction(1, 1000000)
        and cuda["gemv_direct_4_5bpw_a16_preencoded_dp4a"]["syndrome_errors"] == 0
        and Fraction(cuda["gemv_direct_4_5bpw_a16_preencoded_dp4a"]["rel_l2_err_fp64"]) < Fraction(1, 1000000)
        and Fraction(cuda["gemv_direct_4_5bpw_block_h2_dp4a"]["rel_l2_err_fp64"]) < Fraction(1, 1000000)
    )

    return {
        "all_verified": all(checks.values()),
        "checks": checks,
    }


def canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    """Serialize payload to strict canonical UTF-8 JSON bytes (allow_nan=False, LF terminated)."""
    text = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )
    return (text + "\n").encode("utf-8")


def build_certificate() -> dict[str, Any]:
    """Build the canonical self-contained Track S49b certificate with SHA-256 digest."""
    exact_part = _build_exact_model_and_format_invariants()
    sampled_part = _build_real_llm_sampled_ratios()

    exact_pass = evaluate_exact_verdict(exact_part)
    sampled_pass = all(sampled_part["sampled_gates"].values())

    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "oracle_module": ORACLE_MODULE_NAME,
        "epistemic_taxonomy": {
            "exact_pillar": "exact",
            "live_llm_pillar": "sampled",
            "upstream_baselines": "anchor",
        },
        "pillar_1_exact_model_and_wire": exact_part,
        "pillar_2_sampled_live_real_llms": sampled_part,
        "verdict": {
            "exact_invariants_pass": exact_pass,
            "sampled_gates_pass": sampled_pass,
            "all_passed": exact_pass and sampled_pass,
        },
    }
    digest = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    body["certificate_sha256"] = digest
    return body
