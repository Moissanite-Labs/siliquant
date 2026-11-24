"""Pure-Integer / Pure-Rational Oracle for Column-Sorted Block-H4, Coupled Routing, and sm_86 PTX Study.

Zero floats, zero float() calls, zero bare asserts.
Epistemic taxonomy:
  - Pillar 1 (exact): 4-model dimensions (840 matrices, 4,757,127,168 weights, 91 arms, 68 single-block
    decompositions), exact 9/2 bpw layout, exact Block-H4 register butterfly lossless int8_t bound
    (4 * 7 = 28 <= 127), exact dyadic 1/2 scale, exact Layer_Coherent 1-bit/block overhead in Q,
    exact ProjType_Uniform 0-bit per-layer overhead, and exact O(d) vs O(M*d) gather complexity ratio.
  - Pillar 2 (sampled, measured_live_model_weights=True): live RTX 3080 outcomes across all 3 pillars
    (column-sorted & zero-cost folded Block-H4, cross-layer non-additivity dissection & coupled-coherent
    routing, and bare-metal sm_86 PTX zero-cost folded & shared-RMSNorm O(d) fused gather benchmarks).
  - Prior Study Links (anchor): pinned certificate digests of the 5 preceding hardware studies.
"""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from typing import Any

SCHEMA_VERSION = "aletheia.s69.rtx3080_sorted_blockh4_coupled_routing_ptx.v1"

PINNED_DIGESTS: dict[str, str] = {
    "preregistration_file_sha256": "59540d4e8ab82f9ea29015b8c37ee5d523500b37ed9e62745673373fd95de2a6",
    "preregistration_sha256": "b2213512967263de16ab680e833bfc8e2eee7da41621eb49ce1fa7d8872cebd8",
    "cuda_source_sha256": "0043e6e0da62272d80da7a2c348f0280ec1428803bd54243b1f4b5d96d64bb97",
    "empirical_json_sha256": "05175f89d323779ae04d9cfcf02345a87929a42fdffe52708470c1665026ed62",
    "mlos_file_sha256": "19bbdd74184c853b2ac996a64beac46af93bb99e16315a323f6328469d5f6cf5",
    "mlos_receipt_sha256": "f58343289768a3691a234cf4d3c5a550d916352b963b199ade6bc7b761a87665",
    "smollm2_revision": "f8027fd0eaeea54caa13c31d31b9fdc459c38b49",
    "smollm2_safetensors_sha256": "7aaff6661428bed033abba9522bec81938678642cca3181fe752b6ca9e1e540f",
    "qwen05b_revision": "060db6499f32faf8b98477b0a26969ef7d8b9987",
    "qwen05b_safetensors_sha256": "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342",
    "qwen15b_revision": "8faed761d45a263340a0528343f099c05c9a4323",
    "qwen15b_safetensors_sha256": "a961db72e75d52b18e6b0c9d379e51a26973b233385e0e127fdda7d648aec796",
    "qwen3b_revision": "3aab1f1954e9cc14eb9509a215f9e5ca08227a9b",
    "qwen3b_shard1_sha256": "f9558df91d3b89b4826e4db37439edb52f1d62a4fd602685013e7ca6b9f60f8f",
    "qwen3b_shard2_sha256": "51410930d5cf19a998fdb17ef0c46e4d9ace72c97a975a3331395a8a500f5edb",
    "wikitext2_revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
    "wikitext2_val_sha256": "93abc2e0cd803097893a176fd0de13e75c77a1414c5e64af7bbcf705068938c5",
    "wikitext2_test_sha256": "430983e5a711fbc4c6576edffecacf47ff23d58f62f3923d2386d798e0a794c9",
    "python_stdlib_sha256": "fe8ff0f94937ebd41e55f9880dd19bbaa4bbeadec791a6544bbafcb9cd2ee609",
}

PRIOR_STUDY_ANCHORS: dict[str, str] = {
    "s59_certificate_sha256": "b5cafe89c5cc084d5b28163754887c05de08cdab17023b9b0abbfe2ebe349c0d",
    "s60_certificate_sha256": "221d2c29730a3c9fb127b0a2a8cce2c0c97b4f335be8925dcf49c6ea1f63b698",
    "s63_certificate_sha256": "df2675097e53a3e59a6a7bf4c0e67dba95d39c872ed48d533ae487f21b352c5b",
    "s67_certificate_sha256": "882051fa65f809a2e4c99b1d544f4b3b5c62a43cda40c2067f4b9ed74e59ebca",
    "s68_certificate_sha256": "2cb4a47c95ece794ce1d3b6f9c8a3eb01603108ab653b73d868934ae4842c743",
}

MODEL_DIMENSIONS: dict[str, tuple[int, int, int, int]] = {
    "HuggingFaceTB/SmolLM2-360M": (32, 224, 314572800, 20),
    "Qwen/Qwen2.5-0.5B": (24, 168, 357826560, 20),
    "Qwen/Qwen2.5-1.5B": (28, 196, 1310195712, 20),
    "Qwen/Qwen2.5-3B": (36, 252, 2774532096, 31),
}

# Verbatim decimal tokens pinned to rtx3080_s69_empirical_results.json (exact Q parity).
FROZEN_S69_TOKENS: dict[str, Any] = {
    "smollm2": {
        "wt_kl": {
            "A16_axis": "0.065173014",
            "A16_BlockH4": "0.082001576",
            "PerMatrix_ZeroCal": "0.061997583",
            "PerMatrix_Oracle": "0.060817326",
            "Coupled_Subblock_ZeroCal": "0.066152898",
            "Coupled_Subblock_Oracle": "0.065422253",
            "Layer_Coherent_ZeroCal": "0.065553168",
            "Layer_Coherent_Oracle": "0.066012545",
            "Coupled_RelRatio_ZeroCal": "0.065133289",
            "Coupled_RelRatio_Oracle": "0.064542883",
            "ProjType_Uniform": "0.061837779",
            "A16_BlockH4_sort_wcol": "0.064414333",
            "A16_BlockH4_sort_hrand": "0.071524816",
            "A16_BlockH4_zerocost_down": "0.067467402",
            "hA16_real": "0.030954721",
            "hA16_BlockH2_real": "0.03565109",
            "hA16_BlockH4_real": "0.028206886",
            "hA16_BlockH4_sort_grp128_real": "0.030635795",
            "hA16_BlockH4_sort_real": "0.04576595",
            "hA16_BlockH4_zerocost_down_real": "0.031453817",
        },
        "py_kl": {
            "A16_axis": "0.056725239",
            "A16_BlockH4": "0.066761687",
            "PerMatrix_Oracle": "0.05074878",
            "ProjType_Uniform": "0.052088739",
            "A16_BlockH4_sort_wcol": "0.054714696",
            "A16_BlockH4_zerocost_down": "0.054732156",
            "hA16_real": "0.027538215",
            "hA16_BlockH2_real": "0.032958807",
            "hA16_BlockH4_real": "0.026601776",
            "hA16_BlockH4_sort_grp128_real": "0.024308232",
            "hA16_BlockH4_sort_real": "0.046178825",
        },
        "additivity_rcross": {
            "A16_axis": "0.605791",
            "A16_BlockH4": "0.649142",
            "PerMatrix_Oracle": "0.567253",
            "Layer_Coherent_Oracle": "0.610718",
            "ProjType_Uniform": "0.574204",
        },
    },
    "qwen05b": {
        "wt_kl": {
            "A16_axis": "0.052525283",
            "A16_BlockH4": "0.056062964",
            "PerMatrix_ZeroCal": "0.054443338",
            "PerMatrix_Oracle": "0.053823559",
            "Layer_Coherent_ZeroCal": "0.052880589",
            "Coupled_RelRatio_ZeroCal": "0.053600386",
            "ProjType_Uniform": "0.054329276",
            "A16_BlockH4_sort_wcol": "0.049760643",
            "A16_BlockH4_sort_hrand": "0.052642117",
            "A16_BlockH4_zerocost_down": "0.053223329",
            "hA16_real": "0.028102138",
            "hA16_BlockH2_real": "0.02696975",
            "hA16_BlockH4_real": "0.029412649",
            "hA16_BlockH4_sort_grp128_real": "0.02718994",
            "hA16_BlockH4_sort_real": "0.032504237",
            "hA16_BlockH4_zerocost_down_real": "0.028103732",
        },
        "py_kl": {
            "A16_axis": "0.045602058",
            "A16_BlockH4": "0.045183018",
            "ProjType_Uniform": "0.043464967",
            "A16_BlockH4_sort_wcol": "0.038201381",
            "hA16_BlockH2_real": "0.024050317",
            "hA16_BlockH4_real": "0.023451835",
            "hA16_BlockH4_sort_grp128_real": "0.023314575",
            "hA16_BlockH4_sort_real": "0.027835478",
        },
    },
    "qwen15b": {
        "wt_kl": {
            "A16_axis": "0.050062399",
            "A16_BlockH4": "0.046550051",
            "PerMatrix_ZeroCal": "0.048780063",
            "PerMatrix_Oracle": "0.048195276",
            "Coupled_RelRatio_ZeroCal": "0.047889661",
            "ProjType_Uniform": "0.050109638",
            "A16_BlockH4_sort_wcol": "0.039193287",
            "A16_BlockH4_sort_hrand": "0.047169445",
            "A16_BlockH4_zerocost_down": "0.047055179",
            "hA16_real": "0.023636564",
            "hA16_BlockH2_real": "0.023281449",
            "hA16_BlockH4_real": "0.023411598",
            "hA16_BlockH4_sort_grp128_real": "0.023176037",
            "hA16_BlockH4_sort_real": "0.026786727",
            "hA16_BlockH4_zerocost_down_real": "0.024177194",
        },
        "py_kl": {
            "A16_axis": "0.031973308",
            "A16_BlockH4": "0.030045438",
            "A16_BlockH4_sort_wcol": "0.024527654",
            "hA16_real": "0.015112617",
            "hA16_BlockH2_real": "0.014356267",
            "hA16_BlockH4_real": "0.015128765",
            "hA16_BlockH4_sort_grp128_real": "0.014083191",
            "hA16_BlockH4_sort_real": "0.017099112",
        },
        "wt_ppl": {
            "hA16_BlockH2_real": "12.927343",
            "hA16_BlockH4_real": "12.79468",
            "hA16_BlockH4_sort_grp128_real": "12.776768",
        },
    },
    "qwen3b": {
        "bf16_fold": {
            "layers_folded": 36,
            "max_abs_bf16_logit_diff": "1.125",
            "mean_abs_bf16_logit_diff": "0.036161713",
            "kl_fold_vs_ref": "0.000827191572",
            "top1_agreement_pct": "97.8516",
            "exact_identity": False,
        },
        "wt_kl": {
            "E8_H8": "0.045723724",
            "A16_axis": "0.046686444",
            "A16_BlockH4": "0.047941673",
            "PerMatrix_ZeroCal": "0.051205992",
            "PerMatrix_Oracle": "0.050421289",
            "Coupled_Subblock_ZeroCal": "0.048985568",
            "Coupled_Subblock_Oracle": "0.04932156",
            "Layer_Coherent_ZeroCal": "0.048666331",
            "Layer_Coherent_Oracle": "0.050203884",
            "Coupled_RelRatio_ZeroCal": "0.045767877",
            "Coupled_RelRatio_Oracle": "0.049450493",
            "ProjType_Uniform": "0.046336085",
            "A16_BlockH4_sort_wcol": "0.05556036",
            "A16_BlockH4_sort_wcol_matched": "0.055737834",
            "A16_axis_sort_wcol": "0.05811246",
            "A16_BlockH4_sort_hrand": "0.06202423",
            "A16_BlockH4_sort_hrand_matched": "0.062108058",
            "A16_BlockH4_zerocost_down": "0.061528174",
            "A16_BlockH4_zerocost_down_wcol": "0.051323666",
            "hA16_real": "0.024417064",
            "hA16_BlockH2_real": "0.02417879",
            "hA16_BlockH4_real": "0.024774488",
            "hA16_BlockH4_sort_grp128_real": "0.024917616",
            "hA16_BlockH4_sort_real": "0.02602544",
            "hA16_BlockH4_zerocost_down_real": "0.024733987",
            "hA16_BlockH4_zerocost_down_grp128_real": "0.025009076",
            "hA16_BlockH4_sharednorm_real": "0.026580131",
            "hA16_BlockH4_sharednorm_grp128_real": "0.024958235",
            "hA16_BlockH4_sort_rand": "0.030043099",
            "hA16_BlockH2_sort_real": "0.027379935",
            "hA16_axis_sort_real": "0.02760693",
        },
        "py_kl": {
            "E8_H8": "0.03674146",
            "A16_axis": "0.038578245",
            "A16_BlockH4": "0.035619597",
            "PerMatrix_Oracle": "0.039696999",
            "Coupled_RelRatio_ZeroCal": "0.036802163",
            "ProjType_Uniform": "0.034327497",
            "hA16_real": "0.021217129",
            "hA16_BlockH2_real": "0.022975467",
            "hA16_BlockH4_real": "0.022518843",
            "hA16_BlockH4_sort_grp128_real": "0.021902244",
            "hA16_BlockH4_sort_real": "0.021765415",
            "hA16_BlockH4_zerocost_down_real": "0.021903717",
            "hA16_BlockH4_sharednorm_grp128_real": "0.021745405",
        },
        "py_ppl": {
            "hA16_real": "3.176787",
            "hA16_BlockH2_real": "3.174934",
            "hA16_BlockH4_real": "3.166952",
            "hA16_BlockH4_zerocost_down_real": "3.154256",
        },
        "weight_err_db": {
            "hA16_BlockH4_real": "-18.527316",
            "hA16_BlockH4_sort_grp128_real": "-20.867402",
            "hA16_BlockH4_sort_real": "-23.521957",
        },
        "matched_mse_db_diff": {
            "A16_BlockH4_sort_wcol_matched": "1.528e-06",
            "A16_BlockH4_sort_hrand_matched": "1.783e-06",
        },
        "additivity_rcross": {
            "A16_axis": "0.549633",
            "A16_BlockH4": "0.542225",
            "PerMatrix_Oracle": "0.584405",
            "Layer_Coherent_Oracle": "0.57617",
            "ProjType_Uniform": "0.540284",
        },
        "agree_counts": {
            "PerMatrix_ZeroCal": 194,
            "Coupled_Subblock_ZeroCal": 205,
            "Layer_Coherent_ZeroCal": 210,
            "Coupled_RelRatio_ZeroCal": 189,
        },
    },
    "cuda_ptx": {
        "kernels_compiled": 9,
        "min_registers": 20,
        "max_registers": 40,
        "max_stack_bytes": 0,
        "max_spill_stores_bytes": 0,
        "max_spill_loads_bytes": 0,
        "ptx_ldl_count": 0,
        "ptx_stl_count": 0,
        "ptx_dp4a_count": 32,
        "down_triad_gbs": "356.5291",
        "down_axis_gws": "610.8454",
        "down_bh4_unp_gws": "607.902",
        "down_bh4_zerocost_gws": "606.2",
        "down_bh4_zerocost_pct_triad": "95.744",
        "down_bh4_zerocost_rel_l2": "1.048578566e-07",
        "down_zerocost_tok_s": "218.49",
        "mlp_triad_gbs": "356.7879",
        "mlp_unp_ms": "0.082736",
        "mlp_shared_ms": "0.086433",
        "mlp_inline_ms": "0.241376",
        "mlp_shared_overhead_pct": "4.4682",
        "mlp_inline_overhead_pct": "191.7419",
        "mlp_shared_gpu_mj": "7.14852",
        "mlp_inline_gpu_mj": "23.216505",
        "qkv_triad_gbs": "356.6828",
        "qkv_unp_ms": "0.01265",
        "qkv_shared_ms": "0.013959",
        "qkv_inline_ms": "0.034751",
        "qkv_shared_overhead_pct": "10.3462",
        "qkv_inline_overhead_pct": "174.6992",
    },
}


def q_pair(val: Fraction) -> list[int]:
    return [val.numerator, val.denominator]


def _q(token: str) -> Fraction:
    return Fraction(token)


def _reject_floats_iter(root: Any) -> None:
    stack: list[Any] = [root]
    while stack:
        node = stack.pop()
        if isinstance(node, float):
            raise TypeError("float forbidden in canonical JSON")
        if type(node) is dict:
            for k, v in node.items():
                stack.append(k)
                stack.append(v)
        elif type(node) is list:
            stack.extend(node)


def canonical_json_bytes(obj: Any) -> bytes:
    _reject_floats_iter(obj)
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _fraction_to_exact_decimal(val: Fraction) -> str:
    sign = "-" if val.numerator < 0 else ""
    num = -val.numerator if val.numerator < 0 else val.numerator
    den = val.denominator
    c2 = 0
    while den % 2 == 0:
        den //= 2
        c2 += 1
    c5 = 0
    while den % 5 == 0:
        den //= 5
        c5 += 1
    if den != 1:
        raise ValueError("non-terminating rational cannot be serialized as exact decimal")
    scale = c2 if c2 >= c5 else c5
    scaled = num * (5 ** (scale - c5)) * (2 ** (scale - c2))
    if scale == 0:
        return f"{sign}{scaled}"
    digits = str(scaled).rjust(scale + 1, "0")
    frac_part = digits[-scale:].rstrip("0")
    int_part = digits[:-scale]
    return f"{sign}{int_part}.{frac_part}" if frac_part else f"{sign}{int_part}"


def _exact_compact_json_str(obj: Any) -> str:
    t = type(obj)
    if t is dict:
        items = sorted(obj.items(), key=lambda kv: kv[0])
        inner = ",".join(f"{json.dumps(k)}:{_exact_compact_json_str(v)}" for k, v in items)
        return "{" + inner + "}"
    if t is list:
        return "[" + ",".join(_exact_compact_json_str(v) for v in obj) + "]"
    if t is str:
        return json.dumps(obj)
    if t is bool:
        return "true" if obj else "false"
    if t is int:
        return str(obj)
    if obj is None:
        return "null"
    if isinstance(obj, Fraction):
        return _fraction_to_exact_decimal(obj)
    raise TypeError(f"unsupported exact JSON type: {t.__name__}")


def _exact_compact_json_bytes(obj: Any) -> bytes:
    return _exact_compact_json_str(obj).encode("utf-8")


def _is_int_eq(val: Any, target: int) -> bool:
    return type(val) is int and val == target


def _parse_iso_utc_seconds(ts: str) -> int:
    if type(ts) is not str or len(ts) != 20 or not ts.endswith("Z"):
        raise ValueError("timestamp must be YYYY-MM-DDTHH:MM:SSZ")
    if ts[4] != "-" or ts[7] != "-" or ts[10] != "T" or ts[13] != ":" or ts[16] != ":":
        raise ValueError("malformed ISO-8601 UTC timestamp")
    yr = int(ts[0:4])
    mo = int(ts[5:7])
    dy = int(ts[8:10])
    hr = int(ts[11:13])
    mi = int(ts[14:16])
    se = int(ts[17:19])
    if mo < 1 or mo > 12 or dy < 1 or dy > 31 or hr > 23 or mi > 59 or se > 59:
        raise ValueError("timestamp component out of range")
    a = (14 - mo) // 12
    y = yr + 4800 - a
    m = mo + 12 * a - 3
    jdn = dy + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045
    days_since_epoch = jdn - 2440588
    return days_since_epoch * 86400 + hr * 3600 + mi * 60 + se


def _build_exact_pillar() -> dict[str, Any]:
    total_blocks = sum(v[0] for v in MODEL_DIMENSIONS.values())
    total_mats = sum(v[1] for v in MODEL_DIMENSIONS.values())
    total_wts = sum(v[2] for v in MODEL_DIMENSIONS.values())
    total_arms = sum(v[3] for v in MODEL_DIMENSIONS.values())
    additivity_blocks = (
        MODEL_DIMENSIONS["HuggingFaceTB/SmolLM2-360M"][0]
        + MODEL_DIMENSIONS["Qwen/Qwen2.5-3B"][0]
    )

    max_nibble_abs = 7
    max_h4_output_abs = 4 * max_nibble_abs
    int8_max = 127
    bpw_exact = Fraction(16 * 4 + 8, 16)
    qwen3b_layer_coherent_1bit_bpw = Fraction(
        MODEL_DIMENSIONS["Qwen/Qwen2.5-3B"][0],
        MODEL_DIMENSIONS["Qwen/Qwen2.5-3B"][2],
    )
    mlp_gather_ops_shared = 2048
    mlp_gather_ops_inline = 22016 * 2048
    mlp_gather_reduction_exact = Fraction(mlp_gather_ops_inline, mlp_gather_ops_shared)

    return {
        "models_audited": len(MODEL_DIMENSIONS),
        "combined_4model_blocks": total_blocks,
        "combined_4model_matrices": total_mats,
        "combined_4model_weights": total_wts,
        "total_arms_evaluated": total_arms,
        "single_block_additivity_blocks_scored": additivity_blocks,
        "bpw_exact": q_pair(bpw_exact),
        "block_h4_scale_factor_exact": q_pair(Fraction(1, 2)),
        "block_h4_max_butterfly_abs": max_h4_output_abs,
        "block_h4_lossless_int8_dp4a": max_h4_output_abs <= int8_max,
        "down_proj_offline_fold_runtime_gather_ops": 0,
        "projtype_uniform_per_layer_router_bits": 0,
        "qwen3b_layer_coherent_1bit_bpw": q_pair(qwen3b_layer_coherent_1bit_bpw),
        "qwen3b_layer_coherent_below_2e8_bpw": qwen3b_layer_coherent_1bit_bpw < Fraction(2, 100000000),
        "mlp_shared_rmsnorm_gather_reduction_ratio": q_pair(mlp_gather_reduction_exact),
    }


def _build_sampled_pillar() -> dict[str, Any]:
    sm = FROZEN_S69_TOKENS["smollm2"]
    q05 = FROZEN_S69_TOKENS["qwen05b"]
    q15 = FROZEN_S69_TOKENS["qwen15b"]
    q3 = FROZEN_S69_TOKENS["qwen3b"]
    cu = FROZEN_S69_TOKENS["cuda_ptx"]

    sm_ax_wt = _q(sm["wt_kl"]["A16_axis"])
    sm_bh4_wt = _q(sm["wt_kl"]["A16_BlockH4"])
    sm_pm_wt = _q(sm["wt_kl"]["PerMatrix_Oracle"])
    sm_pt_wt = _q(sm["wt_kl"]["ProjType_Uniform"])
    sm_wc_wt = _q(sm["wt_kl"]["A16_BlockH4_sort_wcol"])
    sm_wc_py = _q(sm["py_kl"]["A16_BlockH4_sort_wcol"])
    sm_ax_py = _q(sm["py_kl"]["A16_axis"])
    sm_h4_wt = _q(sm["wt_kl"]["hA16_BlockH4_real"])
    sm_g128_wt = _q(sm["wt_kl"]["hA16_BlockH4_sort_grp128_real"])
    sm_gsort_wt = _q(sm["wt_kl"]["hA16_BlockH4_sort_real"])
    sm_rc_ax = _q(sm["additivity_rcross"]["A16_axis"])
    sm_rc_pm = _q(sm["additivity_rcross"]["PerMatrix_Oracle"])

    q05_ax_wt = _q(q05["wt_kl"]["A16_axis"])
    q05_bh4_wt = _q(q05["wt_kl"]["A16_BlockH4"])
    q05_wc_wt = _q(q05["wt_kl"]["A16_BlockH4_sort_wcol"])
    q05_wc_py = _q(q05["py_kl"]["A16_BlockH4_sort_wcol"])
    q05_ax_py = _q(q05["py_kl"]["A16_axis"])
    q05_h4_wt = _q(q05["wt_kl"]["hA16_BlockH4_real"])
    q05_g128_wt = _q(q05["wt_kl"]["hA16_BlockH4_sort_grp128_real"])
    q05_gsort_wt = _q(q05["wt_kl"]["hA16_BlockH4_sort_real"])
    q05_zc_wt = _q(q05["wt_kl"]["hA16_BlockH4_zerocost_down_real"])

    q15_ax_wt = _q(q15["wt_kl"]["A16_axis"])
    q15_bh4_wt = _q(q15["wt_kl"]["A16_BlockH4"])
    q15_wc_wt = _q(q15["wt_kl"]["A16_BlockH4_sort_wcol"])
    q15_wc_py = _q(q15["py_kl"]["A16_BlockH4_sort_wcol"])
    q15_ax_py = _q(q15["py_kl"]["A16_axis"])
    q15_h2_wt = _q(q15["wt_kl"]["hA16_BlockH2_real"])
    q15_h4_wt = _q(q15["wt_kl"]["hA16_BlockH4_real"])
    q15_g128_wt = _q(q15["wt_kl"]["hA16_BlockH4_sort_grp128_real"])
    q15_gsort_wt = _q(q15["wt_kl"]["hA16_BlockH4_sort_real"])
    q15_h2_py = _q(q15["py_kl"]["hA16_BlockH2_real"])
    q15_h4_py = _q(q15["py_kl"]["hA16_BlockH4_real"])
    q15_g128_py = _q(q15["py_kl"]["hA16_BlockH4_sort_grp128_real"])

    q3_e8_wt = _q(q3["wt_kl"]["E8_H8"])
    q3_ax_wt = _q(q3["wt_kl"]["A16_axis"])
    q3_bh4_wt = _q(q3["wt_kl"]["A16_BlockH4"])
    q3_pm_wt = _q(q3["wt_kl"]["PerMatrix_Oracle"])
    q3_pmz_wt = _q(q3["wt_kl"]["PerMatrix_ZeroCal"])
    q3_sb_wt = _q(q3["wt_kl"]["Coupled_Subblock_Oracle"])
    q3_lc_wt = _q(q3["wt_kl"]["Layer_Coherent_Oracle"])
    q3_rrz_wt = _q(q3["wt_kl"]["Coupled_RelRatio_ZeroCal"])
    q3_pt_wt = _q(q3["wt_kl"]["ProjType_Uniform"])
    q3_wc_wt = _q(q3["wt_kl"]["A16_BlockH4_sort_wcol"])
    q3_axwc_wt = _q(q3["wt_kl"]["A16_axis_sort_wcol"])
    q3_hr_wt = _q(q3["wt_kl"]["hA16_real"])
    q3_h2_wt = _q(q3["wt_kl"]["hA16_BlockH2_real"])
    q3_h4_wt = _q(q3["wt_kl"]["hA16_BlockH4_real"])
    q3_g128_wt = _q(q3["wt_kl"]["hA16_BlockH4_sort_grp128_real"])
    q3_gsort_wt = _q(q3["wt_kl"]["hA16_BlockH4_sort_real"])
    q3_zc_wt = _q(q3["wt_kl"]["hA16_BlockH4_zerocost_down_real"])
    q3_axsort_wt = _q(q3["wt_kl"]["hA16_axis_sort_real"])

    q3_pt_py = _q(q3["py_kl"]["ProjType_Uniform"])
    q3_bh4_py = _q(q3["py_kl"]["A16_BlockH4"])
    q3_e8_py = _q(q3["py_kl"]["E8_H8"])
    q3_ax_py = _q(q3["py_kl"]["A16_axis"])
    q3_zc_py = _q(q3["py_kl"]["hA16_BlockH4_zerocost_down_real"])
    q3_h4_py = _q(q3["py_kl"]["hA16_BlockH4_real"])
    q3_zc_py_ppl = _q(q3["py_ppl"]["hA16_BlockH4_zerocost_down_real"])
    q3_h4_py_ppl = _q(q3["py_ppl"]["hA16_BlockH4_real"])
    q3_h2_py_ppl = _q(q3["py_ppl"]["hA16_BlockH2_real"])

    q3_rc_ax = _q(q3["additivity_rcross"]["A16_axis"])
    q3_rc_b4 = _q(q3["additivity_rcross"]["A16_BlockH4"])
    q3_rc_pm = _q(q3["additivity_rcross"]["PerMatrix_Oracle"])
    q3_rc_pt = _q(q3["additivity_rcross"]["ProjType_Uniform"])

    q3_md_wc = _q(q3["matched_mse_db_diff"]["A16_BlockH4_sort_wcol_matched"])
    q3_md_hr = _q(q3["matched_mse_db_diff"]["A16_BlockH4_sort_hrand_matched"])
    q3_fold_kl = _q(q3["bf16_fold"]["kl_fold_vs_ref"])
    q3_fold_t1 = _q(q3["bf16_fold"]["top1_agreement_pct"])

    d_ax = _q(cu["down_axis_gws"])
    d_unp = _q(cu["down_bh4_unp_gws"])
    d_zc = _q(cu["down_bh4_zerocost_gws"])
    mlp_u = _q(cu["mlp_unp_ms"])
    mlp_s = _q(cu["mlp_shared_ms"])
    mlp_i = _q(cu["mlp_inline_ms"])
    mlp_ov = _q(cu["mlp_shared_overhead_pct"])
    qkv_s = _q(cu["qkv_shared_ms"])
    qkv_i = _q(cu["qkv_inline_ms"])

    gates = {
        "p_s69_1_honest_falsification_global_hreal_sort_loses_to_unpermuted_and_local_grp128_all_4_models": (
            sm_h4_wt < sm_gsort_wt
            and sm_g128_wt < sm_gsort_wt
            and q05_h4_wt < q05_gsort_wt
            and q05_g128_wt < q05_gsort_wt
            and q15_h4_wt < q15_gsort_wt
            and q15_g128_wt < q15_gsort_wt
            and q3_h4_wt < q3_gsort_wt
            and q3_g128_wt < q3_gsort_wt
            and q3_hr_wt < q3_axsort_wt
        ),
        "p_s69_1_local_grp128_beats_unpermuted_blockh4_05b_15b_and_beats_h2_on_15b": (
            q05_g128_wt < q05_h4_wt
            and q05_zc_wt < q05_h4_wt
            and q15_g128_wt < q15_h2_wt
            and q15_h2_wt < q15_h4_wt
            and q15_g128_py < q15_h2_py
            and q15_h2_py < q15_h4_py
        ),
        "p_s69_1_zero_cost_down_improves_3b_cal_and_sets_best_py_ppl": (
            q3_zc_wt < q3_h4_wt
            and q3_zc_py < q3_h4_py
            and q3_zc_py_ppl < q3_h4_py_ppl
            and q3_h4_py_ppl < q3_h2_py_ppl
        ),
        "p_s69_1_zero_text_sort_wcol_beats_axis_and_blockh4_360m_05b_15b_with_3b_reversal": (
            sm_wc_wt < sm_ax_wt
            and sm_wc_wt < sm_bh4_wt
            and sm_wc_py < sm_ax_py
            and q05_wc_wt < q05_ax_wt
            and q05_wc_wt < q05_bh4_wt
            and q05_wc_py < q05_ax_py
            and q15_wc_wt < q15_bh4_wt
            and q15_bh4_wt < q15_ax_wt
            and q15_wc_py < q15_ax_py
            and q3_ax_wt < q3_wc_wt
            and q3_wc_wt < q3_axwc_wt
            and q3_md_wc <= Fraction(2, 1000000)
            and q3_md_hr <= Fraction(2, 1000000)
        ),
        "p_s69_1_honest_bf16_non_associativity_in_36layer_down_fold": (
            q3["bf16_fold"]["layers_folded"] == 36
            and q3["bf16_fold"]["exact_identity"] is False
            and q3_fold_kl < Fraction(1, 1000)
            and q3_fold_t1 > Fraction(97, 1)
        ),
        "p_s69_2_3b_interference_spike_and_smollm2_additive_control_opposite_sign": (
            q3_rc_pm > q3_rc_ax
            and q3_rc_ax > q3_rc_b4
            and q3_rc_b4 > q3_rc_pt
            and sm_pm_wt < sm_pt_wt
            and sm_pt_wt < sm_ax_wt
            and sm_rc_pm < sm_rc_ax
        ),
        "p_s69_2_coherent_cure_projtype_and_relratio_zerocal_beat_oracle_and_axis_on_3b": (
            q3_pt_wt < q3_ax_wt
            and q3_ax_wt < q3_bh4_wt
            and q3_bh4_wt < q3_sb_wt
            and q3_sb_wt < q3_lc_wt
            and q3_lc_wt < q3_pm_wt
            and q3_pt_py < q3_bh4_py
            and q3_bh4_py < q3_e8_py
            and q3_e8_py < q3_ax_py
            and q3_e8_wt < q3_rrz_wt
            and q3_rrz_wt < q3_ax_wt
            and q3_rrz_wt < q3_pmz_wt
        ),
        "p_s69_3_sm86_ptx_zero_spill_bit_exact_zerocost_606gws_and_shared_gather_under_5pct": (
            cu["max_stack_bytes"] == 0
            and cu["max_spill_stores_bytes"] == 0
            and cu["max_spill_loads_bytes"] == 0
            and cu["ptx_ldl_count"] == 0
            and cu["ptx_stl_count"] == 0
            and d_zc >= Fraction(600, 1)
            and d_zc * 100 >= d_unp * 99
            and d_zc * 100 >= d_ax * 99
            and mlp_ov < Fraction(5, 1)
            and mlp_u < mlp_s
            and mlp_s < mlp_i
            and qkv_s < qkv_i
        ),
    }

    return {
        "oracle_class": "sampled",
        "measured_live_model_weights": True,
        "pinned_digests": dict(PINNED_DIGESTS),
        "prior_study_anchors": dict(PRIOR_STUDY_ANCHORS),
        "qwen15b_sort_wcol_over_axis_wt_kl": q_pair(q15_wc_wt / q15_ax_wt),
        "qwen15b_sort_wcol_over_axis_py_kl": q_pair(q15_wc_py / q15_ax_py),
        "qwen15b_grp128_over_h2_wt_kl": q_pair(q15_g128_wt / q15_h2_wt),
        "qwen3b_rcross_oracle_over_axis": q_pair(q3_rc_pm / q3_rc_ax),
        "qwen3b_projtype_over_oracle_wt_kl": q_pair(q3_pt_wt / q3_pm_wt),
        "qwen3b_projtype_over_e8_py_kl": q_pair(q3_pt_py / q3_e8_py),
        "qwen3b_relratio_zerocal_over_e8_wt_kl": q_pair(q3_rrz_wt / q3_e8_wt),
        "qwen3b_down_zerocost_over_unpermuted_gws": q_pair(d_zc / d_unp),
        "qwen3b_down_zerocost_over_axis_gws": q_pair(d_zc / d_ax),
        "qwen3b_mlp_inline_over_shared_ms": q_pair(mlp_i / mlp_s),
        "qwen3b_qkv_inline_over_shared_ms": q_pair(qkv_i / qkv_s),
        "preregistered_pillar_outcomes": {
            "P_S69_1_column_sorted_and_zero_cost_blockh4": "MIXED_GLOBAL_SORT_FALSIFIED_BY_BALL_CANNIBALIZATION_LOCAL_GRP128_ZEROCOST_DOWN_AND_ZERO_TEXT_WCOL_DISCOVERED",
            "P_S69_2_cross_layer_non_additivity_dissection": "CONFIRMED_3OF3_3B_INTERFERENCE_SPIKE_PROJTYPE_UNIFORM_AND_RELRATIO_ZEROCAL_CURE",
            "P_S69_3_sm86_ptx_zerocost_and_shared_gather_pareto": "CONFIRMED_5OF5_ZERO_SPILL_BIT_EXACT_606GWS_SHARED_GATHER_2_79X_FASTER_THAN_INLINE",
        },
        "sampled_gates": gates,
    }


def build_certificate() -> dict[str, Any]:
    p1 = _build_exact_pillar()
    p2 = _build_sampled_pillar()
    exact_ok = (
        p1["combined_4model_blocks"] == 120
        and p1["combined_4model_matrices"] == 840
        and p1["combined_4model_weights"] == 4757127168
        and p1["total_arms_evaluated"] == 91
        and p1["single_block_additivity_blocks_scored"] == 68
        and p1["block_h4_lossless_int8_dp4a"] is True
        and p1["qwen3b_layer_coherent_below_2e8_bpw"] is True
    )
    sampled_ok = all(p2["sampled_gates"].values())
    body = {
        "schema_version": SCHEMA_VERSION,
        "oracle_module": "rtx3080_s69_oracle",
        "epistemic_taxonomy": {
            "exact_pillar": "exact",
            "live_4model_ceer_cycle4_pillar": "sampled",
            "prior_study_links": "anchor",
        },
        "pillar_1_exact_sorted_blockh4_and_routing_arithmetic": p1,
        "pillar_2_sampled_live_ceer_cycle4": p2,
        "verdict": {
            "exact_invariants_pass": exact_ok,
            "sampled_gates_pass": sampled_ok,
            "all_passed": exact_ok and sampled_ok,
        },
    }
    cert_sha = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    cert = dict(body)
    cert["certificate_sha256"] = cert_sha
    return cert


def _to_q(val: Any) -> Fraction:
    if isinstance(val, Fraction):
        return val
    if type(val) is int:
        return Fraction(val)
    if type(val) is str:
        return Fraction(val)
    raise TypeError(f"unsupported rational value type: {type(val).__name__}")


def verify_live_receipts(
    prereg_obj: dict[str, Any],
    emp_obj: dict[str, Any],
    mlos_obj: dict[str, Any],
    prereg_file_sha: str,
    cuda_file_sha: str,
    emp_file_sha: str,
    mlos_file_sha: str,
) -> dict[str, Any]:
    """Verify frozen live receipts against pinned digests, inner hashes, timestamps, 91 arms, additivity, and PTX."""
    prereg_body = {k: v for k, v in prereg_obj.items() if k != "preregistration_sha256"}
    prereg_inner_sha = hashlib.sha256(canonical_json_bytes(prereg_body)).hexdigest()
    mlos_body = {k: v for k, v in mlos_obj.items() if k != "receipt_sha256"}
    mlos_inner_sha = hashlib.sha256(_exact_compact_json_bytes(mlos_body)).hexdigest()

    digests_match = (
        prereg_file_sha == PINNED_DIGESTS["preregistration_file_sha256"]
        and prereg_obj["preregistration_sha256"] == PINNED_DIGESTS["preregistration_sha256"]
        and prereg_inner_sha == prereg_obj["preregistration_sha256"]
        and cuda_file_sha == PINNED_DIGESTS["cuda_source_sha256"]
        and emp_obj["cuda_source_sha256"] == PINNED_DIGESTS["cuda_source_sha256"]
        and emp_file_sha == PINNED_DIGESTS["empirical_json_sha256"]
        and emp_obj["preregistration_file_sha256"] == PINNED_DIGESTS["preregistration_file_sha256"]
        and emp_obj["preregistration_sha256"] == PINNED_DIGESTS["preregistration_sha256"]
        and mlos_file_sha == PINNED_DIGESTS["mlos_file_sha256"]
        and mlos_obj["receipt_sha256"] == PINNED_DIGESTS["mlos_receipt_sha256"]
        and mlos_inner_sha == mlos_obj["receipt_sha256"]
    )

    prereg_ts = _parse_iso_utc_seconds(prereg_obj["created_utc"])
    run_start_ts = _parse_iso_utc_seconds(mlos_obj["execution"]["started_utc"])
    run_end_ts = _parse_iso_utc_seconds(mlos_obj["execution"]["ended_utc"])
    prereg_precedes = prereg_ts < run_start_ts and run_start_ts < run_end_ts

    mlos_clean = (
        mlos_obj["determinism_zone"] == "B"
        and mlos_obj["clean_hardware_run"] is True
        and mlos_obj["anomalies"] == []
        and _is_int_eq(mlos_obj["execution"]["exit_code"], 0)
    )
    mlos_zero_desktop = (
        mlos_obj["start_gpu_telemetry"]["graphics_processes"] == []
        and mlos_obj["end_gpu_telemetry"]["graphics_processes"] == []
        and mlos_obj["start_gpu_telemetry"]["compute_processes"] == []
        and mlos_obj["end_gpu_telemetry"]["compute_processes"] == []
    )
    honesty_ok = (
        emp_obj["measured_live_model_weights"] is True
        and emp_obj["oracle_class"] == "sampled"
        and emp_obj["ac_online"] is True
    )

    mp = emp_obj["model_pins"]
    q3_shards = mp["Qwen/Qwen2.5-3B"]["safetensors_shards"]
    pins_ok = (
        mp["HuggingFaceTB/SmolLM2-360M"]["revision"] == PINNED_DIGESTS["smollm2_revision"]
        and mp["HuggingFaceTB/SmolLM2-360M"]["safetensors_sha256"] == PINNED_DIGESTS["smollm2_safetensors_sha256"]
        and mp["Qwen/Qwen2.5-0.5B"]["revision"] == PINNED_DIGESTS["qwen05b_revision"]
        and mp["Qwen/Qwen2.5-0.5B"]["safetensors_sha256"] == PINNED_DIGESTS["qwen05b_safetensors_sha256"]
        and mp["Qwen/Qwen2.5-1.5B"]["revision"] == PINNED_DIGESTS["qwen15b_revision"]
        and mp["Qwen/Qwen2.5-1.5B"]["safetensors_sha256"] == PINNED_DIGESTS["qwen15b_safetensors_sha256"]
        and mp["Qwen/Qwen2.5-3B"]["revision"] == PINNED_DIGESTS["qwen3b_revision"]
        and len(q3_shards) == 2
        and q3_shards[0]["sha256"] == PINNED_DIGESTS["qwen3b_shard1_sha256"]
        and q3_shards[1]["sha256"] == PINNED_DIGESTS["qwen3b_shard2_sha256"]
    )

    corp = emp_obj["corpora"]
    corpora_ok = (
        corp["wikitext2_revision"] == PINNED_DIGESTS["wikitext2_revision"]
        and corp["wikitext2_val_sha256"] == PINNED_DIGESTS["wikitext2_val_sha256"]
        and corp["wikitext2_test_sha256"] == PINNED_DIGESTS["wikitext2_test_sha256"]
        and corp["python_stdlib_sha256"] == PINNED_DIGESTS["python_stdlib_sha256"]
    )

    models = emp_obj["models"]
    total_arms = 0
    arms_clean = True
    for mname, (exp_blks, exp_mats, exp_wts, exp_arms) in MODEL_DIMENSIONS.items():
        md = models[mname]
        if (
            not _is_int_eq(md["num_hidden_layers"], exp_blks)
            or not _is_int_eq(md["block_linear_matrices"], exp_mats)
            or not _is_int_eq(md["total_block_weights"], exp_wts)
        ):
            arms_clean = False
        if len(md["arms"]) != exp_arms:
            arms_clean = False
        for arm in md["arms"].values():
            total_arms += 1
            if not _is_int_eq(arm["syndrome_errors"], 0) or not _is_int_eq(arm["non_block_tensors_differ"], 0):
                arms_clean = False
    arms_clean = arms_clean and total_arms == 91

    sm_m = models["HuggingFaceTB/SmolLM2-360M"]
    q05_m = models["Qwen/Qwen2.5-0.5B"]
    q15_m = models["Qwen/Qwen2.5-1.5B"]
    q3_m = models["Qwen/Qwen2.5-3B"]
    sm_a, q05_a, q15_a, q3_a = sm_m["arms"], q05_m["arms"], q15_m["arms"], q3_m["arms"]
    q3_bf = q3_m["bf16_down_proj_offline_fold_identity"]

    p1_ok = (
        _to_q(sm_a["hA16_BlockH4_sort_grp128_real"]["wikitext2"]["mean_kl"]) < _to_q(sm_a["hA16_BlockH4_sort_real"]["wikitext2"]["mean_kl"])
        and _to_q(q05_a["hA16_BlockH4_sort_grp128_real"]["wikitext2"]["mean_kl"]) < _to_q(q05_a["hA16_BlockH4_real"]["wikitext2"]["mean_kl"])
        and _to_q(q15_a["hA16_BlockH4_sort_grp128_real"]["wikitext2"]["mean_kl"]) < _to_q(q15_a["hA16_BlockH2_real"]["wikitext2"]["mean_kl"])
        and _to_q(q15_a["A16_BlockH4_sort_wcol"]["wikitext2"]["mean_kl"]) < _to_q(q15_a["A16_axis"]["wikitext2"]["mean_kl"])
        and _to_q(q3_a["hA16_BlockH4_real"]["wikitext2"]["mean_kl"]) < _to_q(q3_a["hA16_BlockH4_sort_real"]["wikitext2"]["mean_kl"])
        and _to_q(q3_a["hA16_BlockH4_zerocost_down_real"]["wikitext2"]["mean_kl"]) < _to_q(q3_a["hA16_BlockH4_real"]["wikitext2"]["mean_kl"])
        and _to_q(q3_a["A16_BlockH4_sort_hrand_matched"]["max_matched_mse_db_diff"]) <= Fraction(2, 1000000)
        and _to_q(q3_a["A16_BlockH4_sort_wcol_matched"]["max_matched_mse_db_diff"]) <= Fraction(2, 1000000)
        and _is_int_eq(q3_bf["layers_folded"], 36)
        and q3_bf["exact_identity"] is False
    )

    sm_ad = sm_m["single_block_additivity_decomposition"]
    q3_ad = q3_m["single_block_additivity_decomposition"]
    p2_ok = (
        _is_int_eq(sm_ad["A16_axis"]["blocks_scored"], 32)
        and _is_int_eq(q3_ad["A16_axis"]["blocks_scored"], 36)
        and _to_q(q3_ad["PerMatrix_Oracle"]["cross_layer_interference_ratio"]) > _to_q(q3_ad["A16_axis"]["cross_layer_interference_ratio"])
        and _to_q(q3_ad["ProjType_Uniform"]["cross_layer_interference_ratio"]) < _to_q(q3_ad["A16_axis"]["cross_layer_interference_ratio"])
        and _to_q(q3_a["ProjType_Uniform"]["wikitext2"]["mean_kl"]) < _to_q(q3_a["A16_axis"]["wikitext2"]["mean_kl"])
        and _to_q(q3_a["ProjType_Uniform"]["python"]["mean_kl"]) < _to_q(q3_a["E8_H8"]["python"]["mean_kl"])
        and _to_q(q3_a["Coupled_RelRatio_ZeroCal"]["wikitext2"]["mean_kl"]) < _to_q(q3_a["A16_axis"]["wikitext2"]["mean_kl"])
        and _to_q(sm_a["PerMatrix_Oracle"]["wikitext2"]["mean_kl"]) < _to_q(sm_a["A16_axis"]["wikitext2"]["mean_kl"])
        and _to_q(sm_ad["PerMatrix_Oracle"]["cross_layer_interference_ratio"]) < _to_q(sm_ad["A16_axis"]["cross_layer_interference_ratio"])
    )

    cu_b = emp_obj["cuda_sm86_ptx_sorted_blockh4_bench"]
    pv = cu_b["ptxas_verification"]
    dn = cu_b["down_proj_stacked2_4096x11008"]
    mlp = cu_b["gate_up_stacked_22016x2048"]
    qkv = cu_b["qkv_stacked_2560x2048"]

    p3_ok = (
        _is_int_eq(pv["max_stack_bytes"], 0)
        and _is_int_eq(pv["max_spill_stores_bytes"], 0)
        and _is_int_eq(pv["max_spill_loads_bytes"], 0)
        and _is_int_eq(pv["ptx_ldl_count"], 0)
        and _is_int_eq(pv["ptx_stl_count"], 0)
        and _is_int_eq(dn["a16_block_h4_zerocost_folded"]["syndrome_errors"], 0)
        and _is_int_eq(dn["a16_block_h4_zerocost_folded"]["int64_bit_exact_mismatches"], 0)
        and _to_q(dn["a16_block_h4_zerocost_folded"]["gweights_per_s"]) >= Fraction(600, 1)
        and _is_int_eq(mlp["parity"]["norm_perm_mismatches"], 0)
        and _is_int_eq(mlp["parity"]["syndrome_errors"], 0)
        and _is_int_eq(mlp["parity"]["int64_bit_exact_mismatches"], 0)
        and _to_q(mlp["parity"]["max_shared_vs_inline_diff"]) == Fraction(0, 1)
        and _to_q(mlp["rmsnorm_shared_gather_plus_blockh4"]["overhead_pct_vs_unpermuted"]) < Fraction(5, 1)
        and _to_q(mlp["rmsnorm_shared_gather_plus_blockh4"]["ms"]) < _to_q(mlp["rmsnorm_unpermuted_plus_inline_gather_blockh4"]["ms"])
        and _is_int_eq(qkv["parity"]["norm_perm_mismatches"], 0)
        and _is_int_eq(qkv["parity"]["syndrome_errors"], 0)
        and _is_int_eq(qkv["parity"]["int64_bit_exact_mismatches"], 0)
        and _to_q(qkv["parity"]["max_shared_vs_inline_diff"]) == Fraction(0, 1)
        and _to_q(qkv["rmsnorm_shared_gather_plus_blockh4"]["ms"]) < _to_q(qkv["rmsnorm_unpermuted_plus_inline_gather_blockh4"]["ms"])
    )

    cem = emp_obj["cuda_export_meta"]
    export_ok = (
        _is_int_eq(cem["down_proj_stacked2_4096x11008"]["export_a16_axis_syndrome_errors"], 0)
        and _is_int_eq(cem["down_proj_stacked2_4096x11008"]["export_a16_block_h4_unpermuted_syndrome_errors"], 0)
        and _is_int_eq(cem["down_proj_stacked2_4096x11008"]["export_a16_block_h4_zerocost_folded_syndrome_errors"], 0)
        and _is_int_eq(cem["gate_up_stacked_22016x2048"]["export_unpermuted_syndrome_errors"], 0)
        and _is_int_eq(cem["gate_up_stacked_22016x2048"]["export_sharednorm_sorted_syndrome_errors"], 0)
        and _is_int_eq(cem["qkv_stacked_2560x2048"]["export_unpermuted_syndrome_errors"], 0)
        and _is_int_eq(cem["qkv_stacked_2560x2048"]["export_sharednorm_sorted_syndrome_errors"], 0)
    )

    checks = {
        "digests_match": digests_match,
        "prereg_strictly_precedes_execution": prereg_precedes,
        "mlos_clean_zone_b": mlos_clean,
        "mlos_zero_desktop_processes": mlos_zero_desktop,
        "real_weight_honesty_tags": honesty_ok,
        "all_4_model_pins_match": pins_ok,
        "corpora_pins_match": corpora_ok,
        "all_91_real_arms_clean": arms_clean,
        "p_s69_1_sorted_blockh4_and_zero_cost_down_findings": p1_ok,
        "p_s69_2_cross_layer_interference_and_coherent_cure": p2_ok,
        "p_s69_3_ptx_zero_spill_bit_exact_and_shared_gather": p3_ok,
        "cuda_export_zero_syndrome_errors": export_ok,
    }
    return {"all_verified": all(checks.values()), "checks": checks}
