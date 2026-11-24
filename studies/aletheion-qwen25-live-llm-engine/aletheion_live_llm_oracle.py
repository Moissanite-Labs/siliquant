#!/usr/bin/env python3
"""Pure-Z/Q Oracle for Track S58 Real-Weight Qwen2.5-0.5B A_16 Live LLM Engine.

Strict Aletheia R1-R5 invariants:
  - Pure Python standard library only (zero external dependencies).
  - Zero IEEE-754 float literals or float() calls anywhere in this module.
  - Zero bare assert statements, zero .isdigit() calls, zero RNG calls.
  - Strict separation of exact algebraic derivations (`exact`), real-hardware
    empirical measurements (`sampled`, `measured_live_model_weights: true`),
    and honest scope boundaries (`anchor`, CS-13 N1/N0).
"""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
import hashlib
import json
from typing import Any

CERT_SCHEMA = "aletheia.s58.aletheion_live_llm_engine.certificate.v1"
CERT_TRACK_ID = "TRACK-S58-ALETHEION-LIVE-LLM-ENGINE"

QWEN_LAYERS = 24
QWEN_H = 896
QWEN_I = 4864
QWEN_Q_HEADS = 14
QWEN_KV_HEADS = 2
QWEN_HEAD_DIM = 64
QWEN_KV_DIM = QWEN_KV_HEADS * QWEN_HEAD_DIM
QWEN_VOCAB = 151936
QWEN_NONBLOCK_BF16_TENSORS = 122

PROJ_SHAPES: tuple[tuple[str, int, int], ...] = (
    ("self_attn.q_proj", QWEN_H, QWEN_H),
    ("self_attn.k_proj", QWEN_KV_DIM, QWEN_H),
    ("self_attn.v_proj", QWEN_KV_DIM, QWEN_H),
    ("self_attn.o_proj", QWEN_H, QWEN_H),
    ("mlp.gate_proj", QWEN_I, QWEN_H),
    ("mlp.up_proj", QWEN_I, QWEN_H),
    ("mlp.down_proj", QWEN_H, QWEN_I),
)

# Pinned SHA-256 digests of the real Qwen2.5-0.5B snapshot & Track S58 silicon receipts
PINNED_SHA256: dict[str, str] = {
    "model_safetensors": "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342",
    "config_json": "479dcf0c5286339e41ad3992cd08ae88a467c4187587936248e2b7c96283484b",
    "tokenizer_json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    "cold_18b_payload": "d986dd20062e7c5ea80b3564603271a5fb080ce0b6115bef551bf68ac2004fbf",
    "tech_prompt": "a79058dd23a53971c8c0f7215ef8d4d2591ac3a2cd2b92a0cb37a1a2a6636b8c",
    "qwen25_sm86_a16_engine_cu": "43f4ebe46c6edd50f2709e2919b5e553f5dcfeb8331a173faf0b008e147fc452",
    "qwen25_avx512_a16_engine_c": "41ee59d9496089f65bfdd6b71e22d9c05ec27eebd20aa3856bbe7c8e7bec338d",
    "sm86_engine_sass_inspection_json": "6ad91df7a9b144962a3fc8798be2fb7b02fa56a03eed5b34ffc114eea53433ee",
    "avx512_engine_isa_inspection_json": "61c12076575e06245cfa64a63a6a1a15705b4bbbd049af62fe350d4a427bcb4a",
    "qwen25_live_engine_empirical_results_json": "14b6f2ac10447184ffc1f5a72f82b4d3916f70ff62143ccc0c410a0938b84590",
    "mlos_zone_b_receipt_json": "21cdc5582e591b0fc5935a39c51ad32fea7b97ca76d6513d0f99fe97db3854c2",
    "mlos_receipt_inner_sha256": "208e14b27757d6adef80df9d0378ee813976694d95f99dcd2893b92354add666",
    "mlos_profile_sha256": "566c9192d38682a0ec4d652a63a3510aaefb9dc20ef91ae9bdc6d4abbbf5cbe2",
}

# Exact decimal strings copied verbatim from qwen25_live_engine_empirical_results.json
RAW_SAMPLED_DECIMALS: dict[str, str] = {
    "bb_a_tok_s": "76.921",
    "bb_a_ms": "13.00028",
    "bb_a_gpu_mj": "618.13029",
    "bb_a_cpu_pkg_mj": "815.73337",
    "bb_a_combined_mj": "1433.86366",
    "bb_a_psys_mj": "80.35898",
    "bb_b1_tok_s": "571.367",
    "bb_b1_ms": "1.75019",
    "bb_b1_gpu_mj": "118.08024",
    "bb_b1_cpu_pkg_mj": "115.15721",
    "bb_b1_combined_mj": "233.23745",
    "bb_b1_psys_mj": "12.80843",
    "bb_b1_phys_gb_s": "115.003",
    "bb_b1_bf16_equiv_gb_s": "408.901",
    "bb_b2_tok_s": "547.941",
    "bb_b2_ms": "1.82501",
    "bb_b2_gpu_mj": "130.65771",
    "bb_b2_cpu_pkg_mj": "96.29605",
    "bb_b2_combined_mj": "226.95376",
    "bb_b2_psys_mj": "12.55737",
    "bb_b2_overhead_pct": "4.275",
    "bb_c1_tok_s": "60.794",
    "bb_c1_ms": "16.44899",
    "bb_c1_cpu_pkg_mj": "1277.95499",
    "bb_c1_psys_mj": "84.43221",
    "bb_c2_tok_s": "58.775",
    "bb_c2_ms": "17.01412",
    "bb_c2_cpu_pkg_mj": "1357.69352",
    "bb_c2_psys_mj": "90.05067",
    "bb_c2_overhead_pct": "3.436",
    "e2e_a_tok_s": "71.576",
    "e2e_a_ms": "13.97115",
    "e2e_a_gpu_mj": "700.76418",
    "e2e_a_cpu_pkg_mj": "834.35847",
    "e2e_a_combined_mj": "1535.12265",
    "e2e_a_psys_mj": "87.1513",
    "e2e_b1_tok_s": "400.58",
    "e2e_b1_ms": "2.49638",
    "e2e_b1_gpu_mj": "191.40979",
    "e2e_b1_cpu_pkg_mj": "121.37732",
    "e2e_b1_combined_mj": "312.78711",
    "e2e_b1_psys_mj": "18.12172",
    "e2e_b2_tok_s": "388.248",
    "e2e_b2_ms": "2.57568",
    "e2e_b2_gpu_mj": "201.01429",
    "e2e_b2_cpu_pkg_mj": "131.79835",
    "e2e_b2_combined_mj": "332.81264",
    "e2e_b2_psys_mj": "19.04042",
    "e2e_b2_overhead_pct": "3.176",
    "e2e_c1_tok_s": "31.697",
    "e2e_c1_ms": "31.54842",
    "e2e_c1_cpu_pkg_mj": "2393.15939",
    "e2e_c1_psys_mj": "159.87605",
    "e2e_c2_tok_s": "29.926",
    "e2e_c2_ms": "33.4152",
    "e2e_c2_cpu_pkg_mj": "2668.591",
    "e2e_c2_psys_mj": "178.37388",
    "e2e_c2_overhead_pct": "5.917",
    "pref_a3_tok_s": "9473.749",
    "pref_a3_batch_ms": "3.3776",
    "pref_a3_bf16_tflops": "6.78",
    "pref_a3_combined_mj": "14.7487",
    "pref_b3_tok_s": "8383.469",
    "pref_b3_batch_ms": "3.81704",
    "pref_b3_int8_tops": "6.0",
    "pref_b3_combined_mj": "15.08536",
    "pref_b3_rel_l2_fp64": "2.7e-08",
    "arm_d_cold_read_ms": "65.896",
    "arm_d_cold_read_gb_s": "3.054",
    "arm_d_transcode_ms": "25.8844",
    "arm_d_transcode_gw_s": "13.824",
    "arm_d_total_cold_ms": "91.7804",
    "arm_d_ttft_a16_ms": "88.0389",
    "arm_d_ttft_bf16_ms": "239.2169",
    "arm_d_gen128_a16_tok_s": "312.11",
    "arm_d_gen128_bf16_tok_s": "69.024",
    "arm_d_first_logit_cos": "0.94749924",
    "arm_d_first_logit_rel_l2": "0.324878",
    "arm_d_tf16_mean_rel_l2": "0.307653",
    "arm_c_vs_gpu_logit_cos": "0.99983232",
    "arm_c_vs_gpu_logit_rel_l2": "0.01834247",
    "agg_weight_rel_l2": "0.119513",
    "max_matrix_weight_rel_l2": "0.170632",
}


def dec_to_frac(token: str) -> Fraction:
    """Convert an exact ASCII decimal string to a canonical Fraction without float."""
    if type(token) is not str or not token:
        raise ValueError("dec_to_frac requires a non-empty ASCII decimal string")
    return Fraction(Decimal(token))


def frac_str(q: Fraction) -> str:
    """Serialize a canonical exact Fraction as 'num/den' (or 'num' if den == 1)."""
    if type(q) is not Fraction:
        raise ValueError("frac_str requires a Fraction")
    if q.denominator == 1:
        return str(q.numerator)
    return f"{q.numerator}/{q.denominator}"


def _popcount(x: int) -> int:
    return x.bit_count()


def verify_a16_code_and_masks_exact() -> dict[str, Any]:
    """Exhaustively certify Extended Hamming [16,11,4], CUDA/AVX-512 bitmasks, and SWAR unpack."""
    codewords = 0
    min_dist = 17
    weight_hist: dict[int, int] = {}
    mask_mismatches = 0

    for bits in range(65536):
        # Canonical F_2 check rows on indices 0..15
        p0 = _popcount(bits) & 1
        s0 = _popcount(bits & 0xAAAA) & 1
        s1 = _popcount(bits & 0xCCCC) & 1
        s2 = _popcount(bits & 0xF0F0) & 1
        s3 = _popcount(bits & 0xFF00) & 1
        syn_ref = p0 | s0 | s1 | s2 | s3

        # Pack LSBs into biased nibble words w0 (idx 0..7) and w1 (idx 8..15)
        w0 = 0
        w1 = 0
        for idx in range(8):
            if (bits >> idx) & 1:
                w0 |= 1 << (4 * idx)
            if (bits >> (idx + 8)) & 1:
                w1 |= 1 << (4 * idx)

        # Exact CUDA sm_86 1-LOP3 + 5-POPC check
        px = (w0 ^ w1) & 0x11111111
        cuda_syn = (
            (_popcount(px) & 1)
            | (_popcount(px & 0x10101010) & 1)
            | (_popcount(px & 0x11001100) & 1)
            | (_popcount(px & 0x11110000) & 1)
            | (_popcount(w1 & 0x11111111) & 1)
        )

        # Exact AVX-512 vpopcntq 64-bit lane check
        lane64 = w0 | (w1 << 32)
        avx_syn = (
            (_popcount(lane64 & 0x1111111111111111) & 1)
            | (_popcount(lane64 & 0x1010101010101010) & 1)
            | (_popcount(lane64 & 0x1100110011001100) & 1)
            | (_popcount(lane64 & 0x1111000011110000) & 1)
            | (_popcount(lane64 & 0x1111111100000000) & 1)
        )

        mask_mismatches += int(cuda_syn != syn_ref) + int(avx_syn != syn_ref)

        if syn_ref == 0:
            codewords += 1
            wt = _popcount(bits)
            weight_hist[wt] = weight_hist.get(wt, 0) + 1
            if wt > 0 and wt < min_dist:
                min_dist = wt

    # Exhaustive check of 3-op SWAR u4 -> s8 (u - 8) for all u in [1..15]
    swar_exact_matches = 0
    for u in range(1, 16):
        x = u ^ 8
        s8_u8 = x | ((x & 8) * 30)
        expected_u8 = (u - 8) & 0xFF
        if s8_u8 == expected_u8:
            swar_exact_matches += 1

    return {
        "n_bits": 16,
        "k_info_bits": 11,
        "parity_check_rows": 5,
        "exhaustive_vectors_checked": 65536,
        "codeword_count": codewords,
        "min_nonzero_distance": min_dist,
        "weight_enumerator": {str(k): weight_hist[k] for k in sorted(weight_hist)},
        "cuda_avx512_mask_mismatches": mask_mismatches,
        "swar_u4_to_s8_exact_matches": swar_exact_matches,
        "swar_u4_to_s8_domain_size": 15,
    }


def build_exact_algebraic_section() -> dict[str, Any]:
    """Derive exact Qwen2.5-0.5B dimensions, A_16 packing ratios, and code properties."""
    shape_counts: dict[str, int] = {}
    total_weights = 0
    total_mats = 0
    aligned_32_128 = True
    for _, m_dim, k_dim in PROJ_SHAPES:
        key = f"{m_dim}x{k_dim}"
        shape_counts[key] = shape_counts.get(key, 0) + QWEN_LAYERS
        total_weights += QWEN_LAYERS * m_dim * k_dim
        total_mats += QWEN_LAYERS
        if (m_dim % 32) != 0 or (k_dim % 128) != 0:
            aligned_32_128 = False

    groups_32 = total_weights // 32
    subblocks_16 = total_weights // 16
    cold_18b_bytes = groups_32 * 18
    bf16_linear_bytes = total_weights * 2
    bpw_q = Fraction(18 * 8, 32)
    comp_q = Fraction(bf16_linear_bytes, cold_18b_bytes)
    embed_bf16_bytes = QWEN_VOCAB * QWEN_H * 2
    total_e2e_read_bytes = cold_18b_bytes + embed_bf16_bytes

    return {
        "provenance_tier": "exact",
        "qwen25_05b_inventory": {
            "num_hidden_layers": QWEN_LAYERS,
            "hidden_size": QWEN_H,
            "intermediate_size": QWEN_I,
            "num_attention_heads": QWEN_Q_HEADS,
            "num_key_value_heads": QWEN_KV_HEADS,
            "head_dim": QWEN_HEAD_DIM,
            "vocab_size": QWEN_VOCAB,
            "linear_matrix_count": total_mats,
            "shape_counts": shape_counts,
            "all_dims_aligned_32_128": aligned_32_128,
            "block_weight_count": total_weights,
            "groups_32_count": groups_32,
            "subblocks_16_count": subblocks_16,
            "nonblock_bf16_tensor_count": QWEN_NONBLOCK_BF16_TENSORS,
            "cold_18b_bytes": cold_18b_bytes,
            "bf16_linear_bytes": bf16_linear_bytes,
            "tied_embed_bf16_bytes": embed_bf16_bytes,
            "e2e_decode_weight_read_bytes": total_e2e_read_bytes,
            "bits_per_weight_q": frac_str(bpw_q),
            "compression_ratio_bf16_over_a16_q": frac_str(comp_q),
        },
        "a16_extended_hamming_exact": verify_a16_code_and_masks_exact(),
    }


def build_sampled_empirical_section() -> dict[str, Any]:
    """Encode real-hardware Qwen2.5-0.5B silicon measurements in pure Z/Q."""
    d = {k: dec_to_frac(v) for k, v in RAW_SAMPLED_DECIMALS.items()}

    bb_sp_b1_a = d["bb_b1_tok_s"] / d["bb_a_tok_s"]
    bb_sp_b1_c1 = d["bb_b1_tok_s"] / d["bb_c1_tok_s"]
    bb_en_a_b1 = d["bb_a_combined_mj"] / d["bb_b1_combined_mj"]
    bb_en_c1_b1 = d["bb_c1_cpu_pkg_mj"] / d["bb_b1_combined_mj"]
    bb_pareto_b1_dom = (
        d["bb_b1_tok_s"] > d["bb_a_tok_s"]
        and d["bb_b1_tok_s"] > d["bb_c1_tok_s"]
        and d["bb_b1_combined_mj"] < d["bb_a_combined_mj"]
        and d["bb_b1_combined_mj"] < d["bb_c1_cpu_pkg_mj"]
    )

    e2e_sp_b1_a = d["e2e_b1_tok_s"] / d["e2e_a_tok_s"]
    e2e_sp_b1_c1 = d["e2e_b1_tok_s"] / d["e2e_c1_tok_s"]
    e2e_en_a_b1 = d["e2e_a_combined_mj"] / d["e2e_b1_combined_mj"]
    e2e_en_c1_b1 = d["e2e_c1_cpu_pkg_mj"] / d["e2e_b1_combined_mj"]
    e2e_pareto_b1_dom = (
        d["e2e_b1_tok_s"] > d["e2e_a_tok_s"]
        and d["e2e_b1_tok_s"] > d["e2e_c1_tok_s"]
        and d["e2e_b1_combined_mj"] < d["e2e_a_combined_mj"]
        and d["e2e_b1_combined_mj"] < d["e2e_c1_cpu_pkg_mj"]
    )

    pref_sp_b3_a3 = d["pref_b3_tok_s"] / d["pref_a3_tok_s"]
    ttft_sp = d["arm_d_ttft_bf16_ms"] / d["arm_d_ttft_a16_ms"]
    gen128_sp = d["arm_d_gen128_a16_tok_s"] / d["arm_d_gen128_bf16_tok_s"]
    tf_rate_q = Fraction(105, 128)

    return {
        "provenance_tier": "sampled",
        "measured_live_model_weights": True,
        "model_repo_id": "Qwen/Qwen2.5-0.5B",
        "model_snapshot_id": "060db6499f32faf8b98477b0a26969ef7d8b9987",
        "pinned_sha256": dict(PINNED_SHA256),
        "isa_receipts": {
            "sm86_kernel_count": 16,
            "sm86_compiler_warnings": 0,
            "sm86_total_stack_bytes": 0,
            "sm86_total_spill_stores_bytes": 0,
            "sm86_total_spill_loads_bytes": 0,
            "sm86_total_sass_ldl": 0,
            "sm86_total_sass_stl": 0,
            "sm86_zero_spill_verified": True,
            "sm86_imma_ldgsts_count": 4,
            "sm86_imma_mma_count": 2,
            "avx512_compiler_warnings": 0,
            "avx512_zmm_register_operands": 385,
            "avx512_vpermb_count": 8,
            "avx512_vpopcntq_count": 10,
            "avx512_vpdpbusd_count": 4,
            "avx512_isa_verified": True,
        },
        "ring0_mlos_receipt": {
            "profile_id": "ux582hs-rtx3080-8gb",
            "determinism_zone": "B",
            "clean_hardware_run": True,
            "exit_code": 0,
            "anomaly_count": 0,
            "graphics_process_count": 0,
            "compute_process_count": 0,
            "linux_kmd_baseline_vram_mib": 2,
            "thermal_throttled": False,
            "slowdown_active": False,
            "start_pstate": "P8",
            "end_pstate": "P8",
        },
        "decode_backbone_24layers_q": {
            "arm_a_bf16_tok_s_q": frac_str(d["bb_a_tok_s"]),
            "arm_a_bf16_ms_q": frac_str(d["bb_a_ms"]),
            "arm_a_bf16_gpu_mj_q": frac_str(d["bb_a_gpu_mj"]),
            "arm_a_bf16_cpu_pkg_mj_q": frac_str(d["bb_a_cpu_pkg_mj"]),
            "arm_a_bf16_combined_mj_q": frac_str(d["bb_a_combined_mj"]),
            "arm_a_bf16_psys_mj_q": frac_str(d["bb_a_psys_mj"]),
            "arm_b1_sm86_tok_s_q": frac_str(d["bb_b1_tok_s"]),
            "arm_b1_sm86_ms_q": frac_str(d["bb_b1_ms"]),
            "arm_b1_sm86_gpu_mj_q": frac_str(d["bb_b1_gpu_mj"]),
            "arm_b1_sm86_cpu_pkg_mj_q": frac_str(d["bb_b1_cpu_pkg_mj"]),
            "arm_b1_sm86_combined_mj_q": frac_str(d["bb_b1_combined_mj"]),
            "arm_b1_sm86_psys_mj_q": frac_str(d["bb_b1_psys_mj"]),
            "arm_b1_phys_gb_s_q": frac_str(d["bb_b1_phys_gb_s"]),
            "arm_b1_bf16_equiv_gb_s_q": frac_str(d["bb_b1_bf16_equiv_gb_s"]),
            "arm_b2_sm86_verify_tok_s_q": frac_str(d["bb_b2_tok_s"]),
            "arm_b2_sm86_verify_ms_q": frac_str(d["bb_b2_ms"]),
            "arm_b2_sm86_verify_combined_mj_q": frac_str(d["bb_b2_combined_mj"]),
            "arm_b2_verify_overhead_pct_q": frac_str(d["bb_b2_overhead_pct"]),
            "arm_c1_avx512_8t_tok_s_q": frac_str(d["bb_c1_tok_s"]),
            "arm_c1_avx512_8t_ms_q": frac_str(d["bb_c1_ms"]),
            "arm_c1_avx512_cpu_pkg_mj_q": frac_str(d["bb_c1_cpu_pkg_mj"]),
            "arm_c1_avx512_psys_mj_q": frac_str(d["bb_c1_psys_mj"]),
            "arm_c2_avx512_verify_tok_s_q": frac_str(d["bb_c2_tok_s"]),
            "arm_c2_avx512_verify_ms_q": frac_str(d["bb_c2_ms"]),
            "arm_c2_avx512_cpu_pkg_mj_q": frac_str(d["bb_c2_cpu_pkg_mj"]),
            "arm_c2_verify_overhead_pct_q": frac_str(d["bb_c2_overhead_pct"]),
            "speedup_b1_vs_a_q": frac_str(bb_sp_b1_a),
            "speedup_b1_vs_c1_q": frac_str(bb_sp_b1_c1),
            "energy_ratio_a_over_b1_q": frac_str(bb_en_a_b1),
            "energy_ratio_c1_pkg_over_b1_q": frac_str(bb_en_c1_b1),
            "b1_pareto_dominates_a_and_c1": bb_pareto_b1_dom,
        },
        "decode_full_e2e_with_bf16_lm_head_q": {
            "arm_a_bf16_tok_s_q": frac_str(d["e2e_a_tok_s"]),
            "arm_a_bf16_ms_q": frac_str(d["e2e_a_ms"]),
            "arm_a_bf16_combined_mj_q": frac_str(d["e2e_a_combined_mj"]),
            "arm_a_bf16_psys_mj_q": frac_str(d["e2e_a_psys_mj"]),
            "arm_b1_sm86_tok_s_q": frac_str(d["e2e_b1_tok_s"]),
            "arm_b1_sm86_ms_q": frac_str(d["e2e_b1_ms"]),
            "arm_b1_sm86_combined_mj_q": frac_str(d["e2e_b1_combined_mj"]),
            "arm_b1_sm86_psys_mj_q": frac_str(d["e2e_b1_psys_mj"]),
            "arm_b2_sm86_verify_tok_s_q": frac_str(d["e2e_b2_tok_s"]),
            "arm_b2_sm86_verify_ms_q": frac_str(d["e2e_b2_ms"]),
            "arm_b2_sm86_verify_combined_mj_q": frac_str(d["e2e_b2_combined_mj"]),
            "arm_b2_verify_overhead_pct_q": frac_str(d["e2e_b2_overhead_pct"]),
            "arm_c1_avx512_8t_tok_s_q": frac_str(d["e2e_c1_tok_s"]),
            "arm_c1_avx512_8t_ms_q": frac_str(d["e2e_c1_ms"]),
            "arm_c1_avx512_cpu_pkg_mj_q": frac_str(d["e2e_c1_cpu_pkg_mj"]),
            "arm_c1_avx512_psys_mj_q": frac_str(d["e2e_c1_psys_mj"]),
            "arm_c2_avx512_verify_tok_s_q": frac_str(d["e2e_c2_tok_s"]),
            "arm_c2_avx512_verify_ms_q": frac_str(d["e2e_c2_ms"]),
            "arm_c2_avx512_cpu_pkg_mj_q": frac_str(d["e2e_c2_cpu_pkg_mj"]),
            "arm_c2_verify_overhead_pct_q": frac_str(d["e2e_c2_overhead_pct"]),
            "speedup_b1_vs_a_q": frac_str(e2e_sp_b1_a),
            "speedup_b1_vs_c1_q": frac_str(e2e_sp_b1_c1),
            "energy_ratio_a_over_b1_q": frac_str(e2e_en_a_b1),
            "energy_ratio_c1_pkg_over_b1_q": frac_str(e2e_en_c1_b1),
            "b1_pareto_dominates_a_and_c1": e2e_pareto_b1_dom,
        },
        "prefill_b32_all168_q": {
            "arm_a3_bf16_tok_s_q": frac_str(d["pref_a3_tok_s"]),
            "arm_a3_bf16_batch_ms_q": frac_str(d["pref_a3_batch_ms"]),
            "arm_a3_bf16_tflops_q": frac_str(d["pref_a3_bf16_tflops"]),
            "arm_a3_combined_mj_q": frac_str(d["pref_a3_combined_mj"]),
            "arm_b3_imma_tok_s_q": frac_str(d["pref_b3_tok_s"]),
            "arm_b3_imma_batch_ms_q": frac_str(d["pref_b3_batch_ms"]),
            "arm_b3_int8_tops_q": frac_str(d["pref_b3_int8_tops"]),
            "arm_b3_combined_mj_q": frac_str(d["pref_b3_combined_mj"]),
            "arm_b3_l0_q_proj_rel_l2_vs_fp64_q": frac_str(d["pref_b3_rel_l2_fp64"]),
            "speedup_b3_vs_a3_q": frac_str(pref_sp_b3_a3),
        },
        "arm_d_cold_18b_and_parity_q": {
            "cold_disk_read_ms_q": frac_str(d["arm_d_cold_read_ms"]),
            "cold_disk_read_gb_s_q": frac_str(d["arm_d_cold_read_gb_s"]),
            "gpu_h2d_plus_soa_transcode_ms_q": frac_str(d["arm_d_transcode_ms"]),
            "gpu_transcode_gweights_s_q": frac_str(d["arm_d_transcode_gw_s"]),
            "total_cold_to_hot_vram_ms_q": frac_str(d["arm_d_total_cold_ms"]),
            "prompt_token_count": 49,
            "ttft_a16_sm86_ms_q": frac_str(d["arm_d_ttft_a16_ms"]),
            "ttft_bf16_pytorch_ms_q": frac_str(d["arm_d_ttft_bf16_ms"]),
            "ttft_speedup_a16_vs_bf16_q": frac_str(ttft_sp),
            "gen_128_a16_decode_tok_s_q": frac_str(d["arm_d_gen128_a16_tok_s"]),
            "gen_128_bf16_decode_tok_s_q": frac_str(d["arm_d_gen128_bf16_tok_s"]),
            "gen_128_speedup_a16_vs_bf16_q": frac_str(gen128_sp),
            "first_token_id_shared": 25271,
            "first_token_top1_match_bf16": True,
            "first_token_top5_overlap_bf16": 5,
            "first_token_logits_cosine_vs_bf16_q": frac_str(d["arm_d_first_logit_cos"]),
            "first_token_logits_rel_l2_vs_bf16_q": frac_str(d["arm_d_first_logit_rel_l2"]),
            "teacher_forced_128_top1_matches": 105,
            "teacher_forced_128_total": 128,
            "teacher_forced_128_top1_rate_q": frac_str(tf_rate_q),
            "teacher_forced_first16_mean_rel_l2_q": frac_str(d["arm_d_tf16_mean_rel_l2"]),
            "free_running_128_exact_prefix_match": 4,
            "arm_c_vs_gpu_a16_greedy_16tok_matches": 16,
            "arm_c_vs_gpu_a16_greedy_16tok_total": 16,
            "arm_c_vs_gpu_a16_logits_cosine_q": frac_str(d["arm_c_vs_gpu_logit_cos"]),
            "arm_c_vs_gpu_a16_logits_rel_l2_q": frac_str(d["arm_c_vs_gpu_logit_rel_l2"]),
            "aggregate_weight_rel_l2_q": frac_str(d["agg_weight_rel_l2"]),
            "max_matrix_weight_rel_l2_q": frac_str(d["max_matrix_weight_rel_l2"]),
            "arm_c_all_windows_gpu_idle_p8": True,
            "total_live_syndrome_errors": 0,
        },
    }


def build_anchor_limitations_section() -> dict[str, Any]:
    """Record explicit CS-13 N1/N0 scope citations and honest empirical boundaries."""
    return {
        "provenance_tier": "anchor",
        "cs13_citations": [
            {
                "id": "N1-LIVE-QWEN25-05B-A16",
                "tier": "N1",
                "claim": (
                    "Full 24-layer Qwen2.5-0.5B (168 real linear matrices, 357,826,560 weights, "
                    "122 preserved BF16 tensors) executes end-to-end on bare-metal RTX 3080 sm_86 "
                    "PTX/SASS (400.58 tok/s E2E, 571.37 tok/s backbone) and Intel i9-11900H AVX-512 "
                    "8-core CPU (31.70 tok/s E2E, 60.79 tok/s backbone) with 0 syndrome errors and "
                    "16/16 GPU-vs-CPU greedy token agreement under Ring-0 mlos Zone-B provenance."
                ),
            },
            {
                "id": "N0-IMMA-PREFILL-BELOW-CUBLAS",
                "tier": "N0",
                "claim": (
                    "Custom B=32 cp.async+IMMA A_16 prefill reaches 8,383.47 tok/s (0.885x vendor "
                    "cuBLAS BF16 at 9,473.75 tok/s, 15.09 vs 14.75 mJ/tok) while avoiding global "
                    "dequant materialization; it does not yet exceed vendor cuBLAS BF16 at B=32."
                ),
            },
            {
                "id": "N0-GA104M-NVML-ENERGY-COUNTER",
                "tier": "N0",
                "claim": (
                    "GA104M laptop nvmlDeviceGetTotalEnergyConsumption showed >16x spurious idle "
                    "counter drift vs INA3221 rail power and was rejected; GPU mJ/tok uses 100 Hz "
                    "trapezoidal integration of nvmlDeviceGetPowerUsage (mW) paired with hardware "
                    "intel-rapl microjoule counters."
                ),
            },
            {
                "id": "N0-FREE-RUNNING-GREEDY-DIVERGENCE",
                "tier": "N0",
                "claim": (
                    "Pure 4.500-bpw A16_axis without calibration achieves 105/128 (82.03%) teacher-forced "
                    "top-1 parity vs BF16 and 5/5 top-5 overlap, but free-running greedy generation "
                    "diverges after token 4 and can enter repetition at small 0.5B scale; tied BF16 "
                    "lm_head (272.27 MB) also exceeds all 168 A_16 linear weights combined (201.28 MB)."
                ),
            },
        ],
    }


def build_payload() -> dict[str, Any]:
    return {
        "exact": build_exact_algebraic_section(),
        "sampled": build_sampled_empirical_section(),
        "anchor": build_anchor_limitations_section(),
    }


def canonical_json_bytes(obj: Any) -> bytes:
    return (
        json.dumps(
            obj,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def build_certificate() -> dict[str, Any]:
    payload = build_payload()
    payload_sha = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    body = {
        "schema": CERT_SCHEMA,
        "track_id": CERT_TRACK_ID,
        "payload_sha256": payload_sha,
        "payload": payload,
    }
    cert_sha = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    return {
        "schema": CERT_SCHEMA,
        "track_id": CERT_TRACK_ID,
        "payload_sha256": payload_sha,
        "certificate_sha256": cert_sha,
        "payload": payload,
    }
