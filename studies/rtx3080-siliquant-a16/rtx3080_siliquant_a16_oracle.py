"""Pure-Z/Q Exact, Sampled & Anchor Certificate Engine for Track S49.

Certifies:
  1. Exact algebraic invariants of Extended Hamming Construction-A A_16 & A_8,
     Block-H2 vs WHT_8/WHT_16 INT8 dynamic range, exact f64 dot-product bounds
     (L_inf W_max <= 64 and L_2 Euclidean-ball refinements for 18B/72B wire formats),
     and RTX 3080 sm_86 GDDR6 theoretical roofline arithmetic.
  2. Exact-Q verification of frozen live RTX 3080 sm_86 110W AC P0 benchmarks
     (mlos Zone-B receipt, ptxas 0-spill across 24/24 kernels, 7 direct 4.50-bpw
     GEMV arms + exact siliquant#77 18B hampack & 72B hammingindex wire formats,
     PyTorch Tensor Core GEMM vs 55W battery anchor, and 5-seed Portal 59 Regime I,
     Regime II, and Experiment 3B calibration shootouts).

Zero floats, zero float() calls, zero bare asserts, zero third-party imports.
"""

from __future__ import annotations

import hashlib
import json
import math
from fractions import Fraction
from typing import Any

SCHEMA_VERSION = "aletheia.track_s49.rtx3080_siliquant_a16.v1"
ORACLE_MODULE_NAME = "rtx3080_siliquant_a16_oracle"

# Frozen SHA-256 anchors of live RTX 3080 sm_86 AC P0 stamped artifacts
EXPECTED_CUDA_SOURCE_SHA256 = (
    "0bd8df2e79f64ea2bed48c83ee11b6b597b816d458228a0f2420e2b3092f593d"
)
EXPECTED_EMPIRICAL_JSON_SHA256 = (
    "62fff8638cae3158fa3394a4c8be24b8827c7251be5c6ebb1d481bc8acbb13dd"
)
EXPECTED_MLOS_FILE_SHA256 = (
    "d79ccc851ba80e33fb1b0db21c7eb8998eb1a3563d28e6304be8e1e307f68738"
)
EXPECTED_MLOS_RECEIPT_SHA256 = (
    "de1a1b18b65497c629cfa6116eb0be70d4fcdf399e8c2cb2d157b005c051102d"
)

# Exact siliquant#77 (49283ff) enumerative ball cardinalities
EXPECTED_HAMPACK_18B_BOUND = 1207
EXPECTED_HAMPACK_18B_COUNT = 85249697968064323459110312223931592722113
EXPECTED_HAMPACK_18B_LIMBS = (
    "0x7752db132dcdc6c1",
    "0x86be9b4e7b2dc8f8",
    "0x00000000000000fa",
)

EXPECTED_HAMMINGINDEX_72B_BOUND = 5213
EXPECTED_HAMMINGINDEX_72B_LIMBS = (
    "0x4f304e07b16d5101",
    "0x07659db666ff30e3",
    "0xf5e5a80321687252",
    "0x9b09700f620f955f",
    "0x330a2a5480653e4c",
    "0x90af10eaf5f7b462",
    "0x6fb036a8c60c9e08",
    "0x440e7c849462d898",
    "0x0000fa0bf675059d",
)
EXPECTED_HAMMINGINDEX_72B_COUNT = sum(
    int(limb, 16) << (64 * idx) for idx, limb in enumerate(EXPECTED_HAMMINGINDEX_72B_LIMBS)
)

# Exact Extended Hamming [16,11,4] hardware parity masks (matches CUDA __popc)
A16_PARITY_MASKS = (0xFFFF, 0xAAAA, 0xCCCC, 0xF0F0, 0xFF00)
A8_PARITY_MASKS = (0xFF, 0xAA, 0xCC, 0xF0)

# Exact frozen RTX 3080 sm_86 110W AC P0 empirical summary tokens (sampled, exact-Q)
FROZEN_SAMPLED_TOKENS: dict[str, Any] = {
    "stream_triad_gb_s": "356.5679",
    "stream_triad_pct_peak": "92.841",
    "pcie_h2d_gb_s": "12.2908",
    "pcie_d2h_gb_s": "13.1275",
    "fp32_fma_tflops": "17.1444",
    "int8_dp4a_tops": "41.2554",
    "bf16_gemm_tflops": "35.031",
    "fp16_gemm_tflops": "33.722",
    "tf32_gemm_tflops": "17.0938",
    "fp32_strict_gemm_tflops": "7.6297",
    "int8_tc_gemm_tops": "22.5759",
    "standalone_a16_ms": "0.376606",
    "standalone_a16_gweights_per_s": "178.1938",
    "standalone_block_h2_gweights_per_s": "178.2062",
    "standalone_wht8_gweights_per_s": "178.2035",
    "gemv_4096x14336_gweights_per_s": {
        "A16_axis_preencoded_dp4a": "611.2129",
        "Block_H2_dp4a_int8": "609.0706",
        "Block_H2_fp32act": "429.1222",
        "A16_axis": "293.906",
        "A16_axis_dp4a_live_repair": "282.0243",
        "A16_H16": "223.9571",
        "E8_WHT8": "216.9409",
        "hampack_18B_fused_gemv": "3.7346",
        "hammingindex_72B_fused_gemv": "0.4929",
    },
    "gemv_4096x14336_pct_triad": {
        "A16_axis_preencoded_dp4a": "96.511",
        "Block_H2_dp4a_int8": "96.173",
        "Block_H2_fp32act": "67.847",
        "A16_axis": "46.468",
        "A16_axis_dp4a_live_repair": "44.532",
        "A16_H16": "35.409",
        "E8_WHT8": "34.3",
        "hampack_18B_fused_gemv": "0.59",
        "hammingindex_72B_fused_gemv": "0.078",
    },
    "gemv_4096x4096_gweights_per_s": {
        "Block_H2_dp4a_int8": "577.7151",
        "A16_axis_preencoded_dp4a": "576.4954",
        "Block_H2_fp32act": "482.4677",
        "A16_axis_dp4a_live_repair": "292.7806",
        "A16_axis": "279.6859",
        "E8_WHT8": "268.4581",
        "A16_H16": "259.3636",
        "hampack_18B_fused_gemv": "4.162",
        "hammingindex_72B_fused_gemv": "0.5536",
    },
    "hampack_18B_standalone_gweights_per_s_4096x14336": "1.6455",
    "hampack_18B_standalone_gweights_per_s_4096x4096": "1.9107",
    "hammingindex_72B_standalone_gweights_per_s_4096x14336": "0.3037",
    "hammingindex_72B_standalone_gweights_per_s_4096x4096": "0.2944",
    "portal59_regime1_mean_matched_kl": {
        "A16_axis": "0.109015912",
        "A16_BlockH2": "0.129252486",
        "A16_H16": "0.404052282",
        "E8_axis": "0.085557462",
        "E8_BlockH2": "0.100587224",
        "E8_H8": "0.178230395",
    },
    "portal59_regime2_mean_matched_kl": {
        "A16_H16": "0.372072141",
        "A16_BlockH2": "0.77642197",
        "A16_axis": "1.12443943",
        "E8_H8": "0.34446928",
        "E8_BlockH2": "0.573120949",
        "E8_axis": "0.740602885",
    },
    "portal59_calib3b_mean_a16_kl": {
        "hA16_real": "0.002400976",
        "hA16_rand": "0.003238811",
        "plain": "0.003548764",
        "hA16_colproxy": "0.004345282",
    },
    "portal59_calib3b_p5_rand_retention_a16": "0.270094",
    "portal59_calib3b_colproxy_retention_a16": "-0.694101",
    "portal59_calib3b_spearman_real": "0.33454",
    "portal59_calib3b_spearman_colproxy": "-0.181618",
}

# Frozen pinned external anchors (N0 reference measurements)
FROZEN_ANCHOR_TOKENS: dict[str, str] = {
    "battery_55w_stream_triad_gb_s": "356.58",
    "battery_55w_bf16_gemm_tflops": "31.88",
    "battery_55w_a16_spill_gweights_per_s": "147.19",
    "siliquant_pr77_cpu_1core_hampack_gweights_per_s": "0.0101",
    "gemma3_12b_p5_wikitext_retention": "0.5374",
    "gemma3_12b_hbg_vs_ebg_wiki_kl_delta_pct": "4.24",
}


def frac_pair(value: Fraction) -> list[int]:
    """Encode an exact Fraction as a canonical [numerator, denominator] pair."""
    if not isinstance(value, Fraction):
        raise TypeError("frac_pair requires a Fraction")
    return [value.numerator, value.denominator]


def _frac_from_token(token: str) -> Fraction:
    if not isinstance(token, str):
        raise ValueError("fraction token must be a string")
    return Fraction(token)


def _syndrome_from_masks(word: int, masks: tuple[int, ...]) -> int:
    syn = 0
    for bit_idx, mask in enumerate(masks):
        if (word & mask).bit_count() & 1:
            syn |= 1 << bit_idx
    return syn


def _build_extended_hamming_invariants() -> dict[str, Any]:
    """Enumerate all 2^11 [16,11,4] and 2^4 [8,4,4] codewords in pure Z."""
    # Verify 16/16 single-bit flips map bijectively to (p0=1, syn=j) = (j | 16)
    a16_flip_bijection = True
    for j in range(16):
        syn = _syndrome_from_masks(1 << j, A16_PARITY_MASKS)
        expected = 1 | (j << 1)
        if syn != expected:
            a16_flip_bijection = False

    a8_flip_bijection = True
    for j in range(8):
        syn = _syndrome_from_masks(1 << j, A8_PARITY_MASKS)
        expected = 1 | (j << 1)
        if syn != expected:
            a8_flip_bijection = False

    # Enumerate exact weight distribution of [16,11,4] (all 2^16 words)
    a16_weights: dict[int, int] = {}
    a16_codewords = 0
    a16_all_even_weight = True
    for word in range(1 << 16):
        if _syndrome_from_masks(word, A16_PARITY_MASKS) == 0:
            w = word.bit_count()
            a16_weights[w] = a16_weights.get(w, 0) + 1
            a16_codewords += 1
            if w & 1:
                a16_all_even_weight = False

    a16_min_dist = min(w for w in a16_weights if w > 0)
    # Exact kissing number of A_16 = C_16 + 2 Z^16:
    # squared norm 4 vectors = weight-4 codewords * 2^4 signs + 16 axis vectors (±2 e_j)
    a16_kissing = a16_weights[4] * 16 + 16 * 2

    a8_weights: dict[int, int] = {}
    a8_codewords = 0
    for word in range(1 << 8):
        if _syndrome_from_masks(word, A8_PARITY_MASKS) == 0:
            w = word.bit_count()
            a8_weights[w] = a8_weights.get(w, 0) + 1
            a8_codewords += 1
    a8_min_dist = min(w for w in a8_weights if w > 0)
    a8_kissing = a8_weights[4] * 16 + 8 * 2

    return {
        "a16_codeword_count": a16_codewords,
        "a16_min_distance": a16_min_dist,
        "a16_all_even_weight": a16_all_even_weight,
        "a16_weight_enumerator": {str(k): a16_weights[k] for k in sorted(a16_weights)},
        "a16_kissing_number": a16_kissing,
        "a16_single_flip_bijection": a16_flip_bijection,
        "a8_codeword_count": a8_codewords,
        "a8_min_distance": a8_min_dist,
        "a8_kissing_number": a8_kissing,
        "a8_single_flip_bijection": a8_flip_bijection,
    }


def _build_accumulator_and_wire_invariants() -> dict[str, Any]:
    """Exact INT8 dynamic range, f64 accumulator, and 4.50-bpw wire-format bounds."""
    input_peak = 39
    block_h2_peak = 2 * input_peak
    wht4_peak = 4 * input_peak
    wht8_peak = 8 * input_peak
    wht16_peak = 16 * input_peak

    f64_limit = 1 << 53
    f16_max = 65504
    scale_shift = 1 << 24

    # Hypercube L_inf bound for 128-weight groups
    linf_w64_acc = 128 * 64 * f16_max * scale_shift
    linf_w65_acc = 128 * 65 * f16_max * scale_shift

    # Euclidean L_2 ball refinements for siliquant#77 (49283ff)
    max_coord_18b = math.isqrt(EXPECTED_HAMPACK_18B_BOUND)
    max_even_sq_18b = EXPECTED_HAMPACK_18B_BOUND - (EXPECTED_HAMPACK_18B_BOUND & 1)
    max_l1_18b = math.isqrt(32 * max_even_sq_18b)
    l2_acc_18b = max_l1_18b * f16_max * scale_shift
    f64_margin_18b = Fraction(f64_limit, l2_acc_18b)

    max_coord_72b = math.isqrt(EXPECTED_HAMMINGINDEX_72B_BOUND)
    max_even_sq_72b = EXPECTED_HAMMINGINDEX_72B_BOUND - (EXPECTED_HAMMINGINDEX_72B_BOUND & 1)
    max_l1_72b = math.isqrt(128 * max_even_sq_72b)
    l2_acc_72b = max_l1_72b * f16_max * scale_shift
    f64_margin_72b = Fraction(f64_limit, l2_acc_72b)

    # Exact 4.5000 bpw wire formats & RTX 3080 sm_86 GDDR6 roofline
    bpw_18b = Fraction(18 * 8, 32)
    bpw_72b = Fraction(72 * 8, 128)
    bytes_4096x4096 = (4096 * 4096 * 9) // 16
    bytes_4096x14336 = (4096 * 14336 * 9) // 16
    peak_gddr6_gb_s = Fraction(2 * 6001 * 10**6 * 32, 10**9)
    peak_4_5_bpw_gweights_s = peak_gddr6_gb_s / Fraction(9, 16)

    l2_bytes = 4 * 1024 * 1024
    hampack_18b_table_bytes = 33 * 32 * (EXPECTED_HAMPACK_18B_BOUND + 1) * 24
    hi_72b_shell_compressed_bytes = 17 * 16 * (EXPECTED_HAMMINGINDEX_72B_BOUND + 1) * 16
    hi_72b_after_even_bytes = 9 * ((EXPECTED_HAMMINGINDEX_72B_BOUND // 2) + 1) * 72
    hi_72b_total_table_bytes = hi_72b_shell_compressed_bytes + hi_72b_after_even_bytes

    return {
        "int8_input_peak": input_peak,
        "block_h2_unscaled_peak": block_h2_peak,
        "block_h2_int8_safe": block_h2_peak <= 127,
        "wht4_unscaled_peak": wht4_peak,
        "wht8_unscaled_peak": wht8_peak,
        "wht16_unscaled_peak": wht16_peak,
        "wht8_overflows_int8": wht8_peak > 127,
        "wht16_overflows_int8": wht16_peak > 127,
        "f64_exact_integer_limit": f64_limit,
        "linf_w64_exact_safe": linf_w64_acc <= f64_limit,
        "linf_w65_exceeds_f64": linf_w65_acc > f64_limit,
        "hampack_18b_bound": EXPECTED_HAMPACK_18B_BOUND,
        "hampack_18b_max_coord": max_coord_18b,
        "hampack_18b_max_l1": max_l1_18b,
        "hampack_18b_f64_safe": l2_acc_18b <= f64_limit,
        "hampack_18b_f64_margin": frac_pair(f64_margin_18b),
        "hampack_18b_count_bit_length": EXPECTED_HAMPACK_18B_COUNT.bit_length(),
        "hampack_18b_count_fits_136b": EXPECTED_HAMPACK_18B_COUNT <= (1 << 136),
        "hammingindex_72b_bound": EXPECTED_HAMMINGINDEX_72B_BOUND,
        "hammingindex_72b_max_coord": max_coord_72b,
        "hammingindex_72b_max_coord_gt_64": max_coord_72b > 64,
        "hammingindex_72b_max_coord_fits_int8": max_coord_72b <= 127,
        "hammingindex_72b_max_l1": max_l1_72b,
        "hammingindex_72b_l2_ball_f64_safe": l2_acc_72b <= f64_limit,
        "hammingindex_72b_f64_margin": frac_pair(f64_margin_72b),
        "hammingindex_72b_count_bit_length": EXPECTED_HAMMINGINDEX_72B_COUNT.bit_length(),
        "hammingindex_72b_count_fits_560b": EXPECTED_HAMMINGINDEX_72B_COUNT <= (1 << 560),
        "bpw_18b_exact": frac_pair(bpw_18b),
        "bpw_72b_exact": frac_pair(bpw_72b),
        "bpw_exact_match": bpw_18b == bpw_72b == Fraction(9, 2),
        "weight_bytes_4096x4096": bytes_4096x4096,
        "weight_bytes_4096x14336": bytes_4096x14336,
        "rtx3080_theoretical_peak_bw_gb_s": frac_pair(peak_gddr6_gb_s),
        "rtx3080_peak_4_5_bpw_gweights_s": frac_pair(peak_4_5_bpw_gweights_s),
        "rtx3080_l2_cache_bytes": l2_bytes,
        "hampack_18b_table_bytes": hampack_18b_table_bytes,
        "hampack_18b_table_to_l2_ratio": frac_pair(Fraction(hampack_18b_table_bytes, l2_bytes)),
        "hammingindex_72b_compressed_table_bytes": hi_72b_total_table_bytes,
        "hammingindex_72b_table_to_l2_ratio": frac_pair(
            Fraction(hi_72b_total_table_bytes, l2_bytes)
        ),
    }


def _build_sampled_and_anchor_ratios(
    sampled_tokens: dict[str, Any] = FROZEN_SAMPLED_TOKENS,
) -> dict[str, Any]:
    """Exact-Q ratios derived from frozen live RTX 3080 sm_86 AC P0 empirical tokens."""
    bf16_ac = _frac_from_token(sampled_tokens["bf16_gemm_tflops"])
    bf16_bat = _frac_from_token(FROZEN_ANCHOR_TOKENS["battery_55w_bf16_gemm_tflops"])
    sa_a16_nospill = _frac_from_token(sampled_tokens["standalone_a16_gweights_per_s"])
    sa_a16_spill = _frac_from_token(FROZEN_ANCHOR_TOKENS["battery_55w_a16_spill_gweights_per_s"])
    cpu_1core = _frac_from_token(FROZEN_ANCHOR_TOKENS["siliquant_pr77_cpu_1core_hampack_gweights_per_s"])

    gv14 = {
        k: _frac_from_token(v)
        for k, v in sampled_tokens["gemv_4096x14336_gweights_per_s"].items()
    }
    pct14 = {
        k: _frac_from_token(v)
        for k, v in sampled_tokens["gemv_4096x14336_pct_triad"].items()
    }
    r1 = {
        k: _frac_from_token(v)
        for k, v in sampled_tokens["portal59_regime1_mean_matched_kl"].items()
    }
    r2 = {
        k: _frac_from_token(v)
        for k, v in sampled_tokens["portal59_regime2_mean_matched_kl"].items()
    }
    c3 = {
        k: _frac_from_token(v)
        for k, v in sampled_tokens["portal59_calib3b_mean_a16_kl"].items()
    }
    p5_ret = _frac_from_token(sampled_tokens["portal59_calib3b_p5_rand_retention_a16"])
    colproxy_ret = _frac_from_token(
        sampled_tokens["portal59_calib3b_colproxy_retention_a16"]
    )
    rho_real = _frac_from_token(sampled_tokens["portal59_calib3b_spearman_real"])
    rho_proxy = _frac_from_token(sampled_tokens["portal59_calib3b_spearman_colproxy"])

    preenc = gv14["A16_axis_preencoded_dp4a"]
    hp18_sa = _frac_from_token(
        sampled_tokens["hampack_18B_standalone_gweights_per_s_4096x14336"]
    )
    hi72_sa = _frac_from_token(
        sampled_tokens["hammingindex_72B_standalone_gweights_per_s_4096x14336"]
    )

    return {
        "oracle_class": "sampled",
        "measured_live_model_weights": False,
        "provenance_digests": {
            "cuda_source_sha256": EXPECTED_CUDA_SOURCE_SHA256,
            "empirical_json_sha256": EXPECTED_EMPIRICAL_JSON_SHA256,
            "mlos_file_sha256": EXPECTED_MLOS_FILE_SHA256,
            "mlos_receipt_sha256": EXPECTED_MLOS_RECEIPT_SHA256,
        },
        "ac_vs_battery_bf16_gain_ratio": frac_pair(bf16_ac / bf16_bat),
        "zero_spill_standalone_a16_gain_ratio": frac_pair(sa_a16_nospill / sa_a16_spill),
        "gemv_4096x14336_preenc_pct_triad": frac_pair(pct14["A16_axis_preencoded_dp4a"]),
        "gemv_4096x14336_block_h2_dp4a_pct_triad": frac_pair(pct14["Block_H2_dp4a_int8"]),
        "preenc_over_a16_h16_speedup": frac_pair(preenc / gv14["A16_H16"]),
        "preenc_over_e8_wht8_speedup": frac_pair(preenc / gv14["E8_WHT8"]),
        "preenc_over_hampack_18b_gemv_speedup": frac_pair(preenc / gv14["hampack_18B_fused_gemv"]),
        "preenc_over_hammingindex_72b_gemv_speedup": frac_pair(
            preenc / gv14["hammingindex_72B_fused_gemv"]
        ),
        "hampack_18b_gpu_over_cpu_1core_speedup": frac_pair(hp18_sa / cpu_1core),
        "hammingindex_72b_gpu_over_cpu_1core_speedup": frac_pair(hi72_sa / cpu_1core),
        "preenc_gemv_over_cpu_1core_speedup": frac_pair(preenc / cpu_1core),
        "regime1_a16_h16_over_axis_kl_ratio": frac_pair(r1["A16_H16"] / r1["A16_axis"]),
        "regime1_e8_h8_over_axis_kl_ratio": frac_pair(r1["E8_H8"] / r1["E8_axis"]),
        "regime2_a16_h16_over_axis_kl_ratio": frac_pair(r2["A16_H16"] / r2["A16_axis"]),
        "regime2_e8_h8_over_axis_kl_ratio": frac_pair(r2["E8_H8"] / r2["E8_axis"]),
        "calib3b_real_over_plain_kl_ratio": frac_pair(c3["hA16_real"] / c3["plain"]),
        "calib3b_colproxy_over_plain_kl_ratio": frac_pair(c3["hA16_colproxy"] / c3["plain"]),
        "calib3b_p5_rand_retention_a16": frac_pair(p5_ret),
        "calib3b_colproxy_retention_a16": frac_pair(colproxy_ret),
        "calib3b_spearman_real": frac_pair(rho_real),
        "calib3b_spearman_colproxy": frac_pair(rho_proxy),
        "sampled_gates": {
            "ac_bf16_exceeds_battery": bf16_ac > bf16_bat,
            "zero_spill_beats_spill": sa_a16_nospill > sa_a16_spill,
            "preenc_exceeds_95pct_triad": pct14["A16_axis_preencoded_dp4a"] > Fraction(95, 1),
            "block_h2_dp4a_exceeds_95pct_triad": pct14["Block_H2_dp4a_int8"] > Fraction(95, 1),
            "preenc_over_100x_hampack_18b": preenc > gv14["hampack_18B_fused_gemv"] * 100,
            "preenc_over_1000x_hammingindex_72b": (
                preenc > gv14["hammingindex_72B_fused_gemv"] * 1000
            ),
            "regime1_coherent_axis_beats_hadamard": (
                r1["A16_axis"] < r1["A16_H16"] and r1["E8_axis"] < r1["E8_H8"]
            ),
            "regime2_spikes_hadamard_beats_axis": (
                r2["A16_H16"] < r2["A16_axis"] and r2["E8_H8"] < r2["E8_axis"]
            ),
            "calib3b_strict_order_real_rand_plain_proxy": (
                c3["hA16_real"] < c3["hA16_rand"] < c3["plain"] < c3["hA16_colproxy"]
            ),
            "calib3b_p5_below_70pct": p5_ret < Fraction(7, 10),
            "calib3b_colproxy_negative_retention": colproxy_ret < 0,
            "calib3b_spearman_sign_flip": rho_proxy < 0 < rho_real,
        },
    }


def verify_live_receipts(
    empirical_obj: dict[str, Any],
    mlos_obj: dict[str, Any],
    empirical_sha256: str,
    mlos_file_sha256: str,
    cuda_source_sha256: str,
) -> dict[str, Any]:
    """Pure-Z/Q verifier of raw empirical JSON (parsed with parse_float=Fraction) & mlos receipt."""
    checks: dict[str, bool] = {}

    checks["cuda_sha256_match"] = (
        cuda_source_sha256 == EXPECTED_CUDA_SOURCE_SHA256
        and empirical_obj.get("cuda_source_sha256") == EXPECTED_CUDA_SOURCE_SHA256
    )
    checks["empirical_sha256_match"] = empirical_sha256 == EXPECTED_EMPIRICAL_JSON_SHA256
    checks["mlos_file_sha256_match"] = mlos_file_sha256 == EXPECTED_MLOS_FILE_SHA256
    checks["mlos_receipt_sha256_match"] = (
        mlos_obj.get("receipt_sha256") == EXPECTED_MLOS_RECEIPT_SHA256
    )
    checks["mlos_clean_zone_b"] = (
        mlos_obj.get("clean_hardware_run") is True
        and mlos_obj.get("anomalies") == []
        and mlos_obj.get("determinism_zone") == "B"
        and mlos_obj.get("execution", {}).get("exit_code") == 0
    )
    end_tel = mlos_obj.get("end_gpu_telemetry", {})
    checks["mlos_zero_desktop_processes"] = (
        end_tel.get("gpu_processes_queried") is True
        and end_tel.get("graphics_processes") == []
        and isinstance(end_tel.get("linux_kmd_baseline_vram_mib"), int)
        and not isinstance(end_tel.get("linux_kmd_baseline_vram_mib"), bool)
        and end_tel["linux_kmd_baseline_vram_mib"] <= 4
    )
    checks["epistemic_honesty_tags"] = (
        empirical_obj.get("oracle_class") == "sampled"
        and empirical_obj.get("measured_live_model_weights") is False
    )

    cuda = empirical_obj["bare_metal_sm86_cuda"]
    ptxas = cuda["ptxas_verification"]
    checks["ptxas_24_kernels_zero_spill"] = (
        ptxas["kernels_compiled"] == 24
        and ptxas["max_spill_loads_bytes"] == 0
        and ptxas["max_spill_stores_bytes"] == 0
    )

    # Check 0 syndrome errors and < 1e-6 FP64 rel_l2 across all 7 direct GEMV arms x 2 shapes
    direct_ok = True
    for shape_key in ("4096x4096", "4096x14336"):
        arms = cuda["fused_gemv_4_5_bpw"][shape_key]["arms"]
        if len(arms) != 7:
            direct_ok = False
        for arm in arms.values():
            if arm["syndrome_errors"] != 0 or Fraction(arm["rel_l2_err_fp64"]) >= Fraction(1, 1000000):
                direct_ok = False
    checks["direct_gemv_14_arms_verified"] = direct_ok

    # Check exact siliquant#77 enumerative 18B & 72B wire formats
    pr77 = cuda["siliquant_pr77_enumerative_wire_formats"]
    hp18_meta = pr77["hampack_18B_meta"]
    hi72_meta = pr77["hammingindex_72B_meta"]
    enum_ok = (
        hp18_meta["bound"] == EXPECTED_HAMPACK_18B_BOUND
        and tuple(hp18_meta["count_limbs_hex"]) == EXPECTED_HAMPACK_18B_LIMBS
        and hp18_meta["pool_bit_mismatches"] == 0
        and hp18_meta["pool_syndrome_errors"] == 0
        and hi72_meta["bound"] == EXPECTED_HAMMINGINDEX_72B_BOUND
        and tuple(hi72_meta["count_limbs_hex"]) == EXPECTED_HAMMINGINDEX_72B_LIMBS
        and hi72_meta["pool_bit_mismatches"] == 0
        and hi72_meta["pool_syndrome_errors"] == 0
    )
    for shape_key in ("4096x4096", "4096x14336"):
        sh = pr77["shapes"][shape_key]
        for arm_key in ("hampack_18B_fused_gemv", "hammingindex_72B_fused_gemv"):
            if (
                sh[arm_key]["syndrome_errors"] != 0
                or Fraction(sh[arm_key]["rel_l2_err_fp64"]) >= Fraction(1, 1000000)
            ):
                enum_ok = False
        for sa_key in ("hampack_18B_standalone", "hammingindex_72B_standalone"):
            if sh[sa_key]["syndrome_errors"] != 0:
                enum_ok = False
    checks["pr77_enumerative_18b_72b_verified"] = enum_ok

    # Check Portal 59 5/5-seed laws
    p59 = empirical_obj["portal59_empirical_shootout"]
    r1_seeds = p59["regime1_coherent_subspace"]["per_seed"]
    r2_seeds = p59["regime2_axis_aligned_spikes"]["per_seed"]
    c3_seeds = p59["experiment3b_calibration_shootout"]["per_seed"]

    r1_all5 = (
        len(r1_seeds) == 5
        and p59["total_a16_syndrome_errors"] == 0
        and all(
            Fraction(row["matched_mse_max_db_diff"]["A16_family"]) < Fraction(1, 1000000)
            and Fraction(row["matched_mse_max_db_diff"]["E8_family"]) < Fraction(1, 1000000)
            and Fraction(row["held_out_kl_quad_matched_mse"]["A16_axis"])
            < Fraction(row["held_out_kl_quad_matched_mse"]["A16_H16"])
            and Fraction(row["held_out_kl_quad_matched_mse"]["E8_axis"])
            < Fraction(row["held_out_kl_quad_matched_mse"]["E8_H8"])
            for row in r1_seeds
        )
    )
    checks["portal59_regime1_5_of_5_seeds"] = r1_all5

    r2_all5 = len(r2_seeds) == 5 and all(
        Fraction(row["matched_mse_max_db_diff"]["A16_family"]) < Fraction(1, 1000000)
        and Fraction(row["matched_mse_max_db_diff"]["E8_family"]) < Fraction(1, 1000000)
        and Fraction(row["held_out_kl_quad_matched_mse"]["A16_H16"])
        < Fraction(row["held_out_kl_quad_matched_mse"]["A16_axis"])
        and Fraction(row["held_out_kl_quad_matched_mse"]["E8_H8"])
        < Fraction(row["held_out_kl_quad_matched_mse"]["E8_axis"])
        for row in r2_seeds
    )
    checks["portal59_regime2_5_of_5_seeds"] = r2_all5

    c3_all5 = len(c3_seeds) == 5 and all(
        Fraction(row["A16_held_out_kl_quad"]["hA16_real"])
        < Fraction(row["A16_held_out_kl_quad"]["hA16_rand"])
        < Fraction(row["A16_held_out_kl_quad"]["plain"])
        < Fraction(row["A16_held_out_kl_quad"]["hA16_colproxy"])
        and Fraction(row["p5_rand_retention_a16"]) < Fraction(7, 10)
        and Fraction(row["colproxy_retention_a16"]) < 0
        and Fraction(row["spearman_colproxy_vs_true_hessian"])
        < 0
        < Fraction(row["spearman_real_vs_true_hessian"])
        for row in c3_seeds
    )
    checks["portal59_calib3b_5_of_5_seeds"] = c3_all5

    return {
        "all_verified": all(checks.values()),
        "checks": checks,
    }


def canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    """Serialize payload to canonical UTF-8 JSON bytes (LF terminated, sorted keys)."""
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return (text + "\n").encode("utf-8")


def evaluate_exact_verdict(hamming: dict[str, Any], wire_acc: dict[str, Any]) -> bool:
    """Evaluate all exact-pillar algebraic and accumulator invariants."""
    return (
        hamming["a16_codeword_count"] == 2048
        and hamming["a16_min_distance"] == 4
        and hamming["a16_all_even_weight"] is True
        and hamming["a16_kissing_number"] == 2272
        and hamming["a16_single_flip_bijection"] is True
        and hamming["a8_codeword_count"] == 16
        and hamming["a8_min_distance"] == 4
        and hamming["a8_kissing_number"] == 240
        and hamming["a8_single_flip_bijection"] is True
        and wire_acc["block_h2_int8_safe"] is True
        and wire_acc["wht8_overflows_int8"] is True
        and wire_acc["wht16_overflows_int8"] is True
        and wire_acc["linf_w64_exact_safe"] is True
        and wire_acc["linf_w65_exceeds_f64"] is True
        and wire_acc["hampack_18b_f64_safe"] is True
        and wire_acc["hampack_18b_count_fits_136b"] is True
        and wire_acc["hammingindex_72b_max_coord_gt_64"] is True
        and wire_acc["hammingindex_72b_max_coord_fits_int8"] is True
        and wire_acc["hammingindex_72b_l2_ball_f64_safe"] is True
        and wire_acc["hammingindex_72b_count_fits_560b"] is True
        and wire_acc["bpw_exact_match"] is True
    )


def build_certificate() -> dict[str, Any]:
    """Build the canonical self-contained Track S49 certificate with SHA-256 digest."""
    hamming = _build_extended_hamming_invariants()
    wire_acc = _build_accumulator_and_wire_invariants()
    sampled = _build_sampled_and_anchor_ratios()

    exact_all_pass = evaluate_exact_verdict(hamming, wire_acc)
    sampled_all_pass = all(sampled["sampled_gates"].values())

    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "oracle_module": ORACLE_MODULE_NAME,
        "epistemic_taxonomy": {
            "exact_pillar": "exact",
            "gpu_empirical_pillar": "sampled",
            "reference_baselines": "anchor",
        },
        "pillar_1_exact_hamming": hamming,
        "pillar_2_exact_wire_and_accumulator": wire_acc,
        "pillar_3_sampled_rtx3080_and_anchors": sampled,
        "verdict": {
            "exact_invariants_pass": exact_all_pass,
            "sampled_gates_pass": sampled_all_pass,
            "all_passed": exact_all_pass and sampled_all_pass,
        },
    }
    digest = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    body["certificate_sha256"] = digest
    return body
