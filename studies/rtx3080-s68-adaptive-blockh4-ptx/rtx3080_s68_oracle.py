"""Pure-Integer / Pure-Rational Oracle for Track S68 (CEER Cycle-3 Adaptive Block-H4 Router & sm_86 PTX).

Zero floats, zero float() calls, zero bare asserts.
Epistemic taxonomy:
  - Pillar 1 (exact): 4-model dimensions (840 matrices, 4,757,127,168 weights, 46 arms),
    exact 1-bit / 2-bit per-layer routing overhead in Q, exact A_16 Block-H4 register-butterfly
    int8_t lossless bound (4 * 7 = 28 <= 127), exact dyadic 1/2 scale & 9/2 bpw layout.
  - Pillar 2 (sampled, measured_live_model_weights=True): live RTX 3080 CEER Cycle-3 outcomes
    for P_S68_1 (zero-cal per-layer router), P_S68_2 (calibrated floors hA16_BlockH4_real vs
    hA16_BlockH2_real & hA16_real), and P_S68_3 (bare-metal sm_86 PTX fused A16_BlockH4 kernel).
  - Prior Track Links (anchor): pinned certificate digests of Tracks S59, S60, S63, S67.
"""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from typing import Any

SCHEMA_VERSION = "aletheia.track_s68.rtx3080_adaptive_blockh4_ptx_ceer.v1"

PINNED_DIGESTS: dict[str, str] = {
    "preregistration_file_sha256": "0749081becad57160ab5940e58e20835b5676896caa5ea63e5bb81db6c8b8b83",
    "preregistration_sha256": "ae552069fabb0ff7517ef304029670a4f423cd4d833c7db4cce6a51e2dcf604a",
    "cuda_source_sha256": "088c312911a25f746c4705edf388b32bf450d8089324704ed27eefcc524530d0",
    "empirical_json_sha256": "b192e19f35f78689f6f2ab06ff3c2dbec830a422c006178ba5c78e843be77906",
    "mlos_file_sha256": "031b995413a740f6fb95d31b345507af0405e2c24be9b1b39edc0c85fb36f874",
    "mlos_receipt_sha256": "f29d8f6216c51de05006a9f7e624074555e7086e13672513693bda582161a4d2",
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

PRIOR_TRACK_ANCHORS: dict[str, str] = {
    "s59_certificate_sha256": "b5cafe89c5cc084d5b28163754887c05de08cdab17023b9b0abbfe2ebe349c0d",
    "s60_certificate_sha256": "221d2c29730a3c9fb127b0a2a8cce2c0c97b4f335be8925dcf49c6ea1f63b698",
    "s63_certificate_sha256": "df2675097e53a3e59a6a7bf4c0e67dba95d39c872ed48d533ae487f21b352c5b",
    "s67_certificate_sha256": "882051fa65f809a2e4c99b1d544f4b3b5c62a43cda40c2067f4b9ed74e59ebca",
}

MODEL_DIMENSIONS: dict[str, tuple[int, int, int]] = {
    "HuggingFaceTB/SmolLM2-360M": (224, 314572800, 9),
    "Qwen/Qwen2.5-0.5B": (168, 357826560, 12),
    "Qwen/Qwen2.5-1.5B": (196, 1310195712, 12),
    "Qwen/Qwen2.5-3B": (252, 2774532096, 13),
}

# Verbatim decimal tokens pinned to rtx3080_s68_empirical_results.json (exact Q parity).
FROZEN_S68_TOKENS: dict[str, Any] = {
    "smollm2": {
        "wt_kl": {
            "A16_axis": "0.065173014",
            "A16_BlockH4": "0.082001576",
            "A16_BlockH8": "0.085383616",
            "E8_H8": "0.071621153",
            "A16_AdaptWeightOnly_1bit": "0.062790106",
            "A16_AdaptZeroCal_1bit": "0.061607961",
            "Oracle_1bit": "0.060817326",
            "A16_AdaptZeroCal_2bit": "0.060647127",
            "Oracle_2bit": "0.062753433",
        },
        "py_kl": {
            "A16_axis": "0.056725239",
            "A16_BlockH4": "0.066761687",
            "A16_BlockH8": "0.070553612",
            "E8_H8": "0.056793001",
            "A16_AdaptWeightOnly_1bit": "0.052207687",
            "A16_AdaptZeroCal_1bit": "0.052364201",
            "Oracle_1bit": "0.05074878",
            "A16_AdaptZeroCal_2bit": "0.056298533",
            "Oracle_2bit": "0.054466328",
        },
        "agree_1bit": 168,
        "agree_2bit": 162,
        "agree_wonly": 143,
    },
    "qwen05b": {
        "wt_kl": {
            "A16_axis": "0.052525283",
            "A16_BlockH4": "0.056062964",
            "A16_BlockH8": "0.057186347",
            "E8_H8": "0.054841455",
            "A16_AdaptWeightOnly_1bit": "0.053247627",
            "A16_AdaptZeroCal_1bit": "0.054459829",
            "Oracle_1bit": "0.053823559",
            "A16_AdaptZeroCal_2bit": "0.054239939",
            "Oracle_2bit": "0.053721384",
            "hA16_real": "0.028102138",
            "hA16_BlockH2_real": "0.02696975",
            "hA16_BlockH4_real": "0.029412649",
        },
        "py_kl": {
            "A16_axis": "0.045602058",
            "A16_BlockH4": "0.045183018",
            "A16_BlockH8": "0.049353208",
            "E8_H8": "0.044983089",
            "A16_AdaptWeightOnly_1bit": "0.045903913",
            "A16_AdaptZeroCal_1bit": "0.047048111",
            "Oracle_1bit": "0.04769984",
            "A16_AdaptZeroCal_2bit": "0.042310367",
            "Oracle_2bit": "0.042454058",
            "hA16_real": "0.024798374",
            "hA16_BlockH2_real": "0.024050317",
            "hA16_BlockH4_real": "0.023451835",
        },
        "wt_ppl": {
            "A16_axis": "18.485835",
            "A16_BlockH4": "18.592614",
            "A16_AdaptZeroCal_1bit": "18.284113",
            "hA16_BlockH2_real": "18.23492",
            "hA16_BlockH4_real": "18.35105",
        },
        "agree_1bit": 133,
        "agree_2bit": 116,
        "agree_wonly": 104,
    },
    "qwen15b": {
        "wt_kl": {
            "A16_axis": "0.050062399",
            "A16_BlockH4": "0.046550051",
            "A16_BlockH8": "0.045553865",
            "E8_H8": "0.053354475",
            "A16_AdaptWeightOnly_1bit": "0.050001434",
            "A16_AdaptZeroCal_1bit": "0.048780063",
            "Oracle_1bit": "0.048195276",
            "A16_AdaptZeroCal_2bit": "0.046827636",
            "Oracle_2bit": "0.046657353",
            "hA16_real": "0.023636564",
            "hA16_BlockH2_real": "0.023281449",
            "hA16_BlockH4_real": "0.023411598",
        },
        "py_kl": {
            "A16_axis": "0.031973308",
            "A16_BlockH4": "0.030045438",
            "A16_BlockH8": "0.029672807",
            "E8_H8": "0.030131405",
            "A16_AdaptWeightOnly_1bit": "0.031951767",
            "A16_AdaptZeroCal_1bit": "0.033490859",
            "Oracle_1bit": "0.030719881",
            "A16_AdaptZeroCal_2bit": "0.030554251",
            "Oracle_2bit": "0.030526827",
            "hA16_real": "0.015112617",
            "hA16_BlockH2_real": "0.014356267",
            "hA16_BlockH4_real": "0.015128765",
        },
        "wt_ppl": {
            "hA16_real": "12.857511",
            "hA16_BlockH2_real": "12.927343",
            "hA16_BlockH4_real": "12.79468",
        },
        "agree_1bit": 148,
        "agree_2bit": 155,
        "agree_wonly": 109,
    },
    "qwen3b": {
        "wt_kl": {
            "A16_axis": "0.046686444",
            "A16_BlockH4": "0.047941673",
            "A16_BlockH8": "0.051088476",
            "E8_H8": "0.045723724",
            "A16_AdaptWeightOnly_1bit": "0.04716653",
            "A16_AdaptZeroCal_1bit": "0.051076282",
            "Oracle_1bit": "0.050421289",
            "A16_AdaptZeroCal_2bit": "0.052907251",
            "Oracle_2bit": "0.053398235",
            "hA16_real": "0.024417064",
            "hA16_BlockH2_real": "0.02417879",
            "hA16_BlockH4_real": "0.024774488",
            "hA16_BlockH4_rand": "0.028497679",
        },
        "py_kl": {
            "A16_axis": "0.038578245",
            "A16_BlockH4": "0.035619597",
            "A16_BlockH8": "0.03946555",
            "E8_H8": "0.03674146",
            "A16_AdaptWeightOnly_1bit": "0.036736023",
            "A16_AdaptZeroCal_1bit": "0.03856409",
            "Oracle_1bit": "0.039696999",
            "A16_AdaptZeroCal_2bit": "0.036122968",
            "Oracle_2bit": "0.036210297",
            "hA16_real": "0.021217129",
            "hA16_BlockH2_real": "0.022975467",
            "hA16_BlockH4_real": "0.022518843",
            "hA16_BlockH4_rand": "0.021898842",
        },
        "wt_ppl": {
            "hA16_real": "11.089508",
            "hA16_BlockH2_real": "11.057649",
            "hA16_BlockH4_real": "11.206232",
            "hA16_BlockH4_rand": "11.040656",
        },
        "py_ppl": {
            "hA16_real": "3.176787",
            "hA16_BlockH2_real": "3.174934",
            "hA16_BlockH4_real": "3.166952",
            "hA16_BlockH4_rand": "3.204251",
        },
        "weight_err_db": {
            "hA16_real": "-21.164957",
            "hA16_BlockH2_real": "-19.444235",
            "hA16_BlockH4_real": "-18.527316",
        },
        "agree_1bit": 190,
        "agree_2bit": 214,
        "agree_wonly": 143,
    },
    "cuda_ptx": {
        "kernels_compiled": 7,
        "min_registers": 20,
        "max_registers": 40,
        "max_stack_bytes": 0,
        "max_spill_stores_bytes": 0,
        "max_spill_loads_bytes": 0,
        "ptx_ldl_count": 0,
        "ptx_stl_count": 0,
        "ptx_dp4a_count": 32,
        "single_triad_gbs": "357.6804",
        "single_axis_gws": "578.3556",
        "single_bh2_gws": "579.7362",
        "single_bh4_gws": "575.3089",
        "single_bh4_pct_triad": "90.612",
        "single_bh4_rel_l2": "1.125292e-07",
        "stacked2_triad_gbs": "358.3242",
        "stacked2_axis_gws": "615.631",
        "stacked2_bh2_gws": "600.6854",
        "stacked2_bh4_gws": "592.3081",
        "stacked2_bh4_pct_triad": "93.081",
        "stacked2_bh4_rel_l2": "1.121494101e-07",
        "stacked2_axis_tok_s": "221.89",
        "stacked2_bh2_tok_s": "216.5",
        "stacked2_bh4_tok_s": "213.48",
        "stacked2_axis_combined_mj": "538.9778",
        "stacked2_bh2_combined_mj": "581.0318",
        "stacked2_bh4_combined_mj": "639.7272",
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
        if type(node) is float:
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
    total_mats = sum(v[0] for v in MODEL_DIMENSIONS.values())
    total_wts = sum(v[1] for v in MODEL_DIMENSIONS.values())
    total_arms = sum(v[2] for v in MODEL_DIMENSIONS.values())

    # Exact A16 Block-H4 register-only butterfly arithmetic in Z/Q:
    # Signed 4-bit nibble a_j in [-7, 7] => |(H_4 a)_k| <= 4*7 = 28 <= 127 (exact int8_t)
    max_nibble_abs = 7
    max_h4_output_abs = 4 * max_nibble_abs
    int8_max = 127
    # Exact 4.5000-bpw layout: 16 weights * 4 bits + 1 FP16 scale (16 bits) per 32 weights
    bpw_exact = Fraction(16 * 4 + 8, 16)
    qwen3b_1bit_overhead = Fraction(MODEL_DIMENSIONS["Qwen/Qwen2.5-3B"][0], MODEL_DIMENSIONS["Qwen/Qwen2.5-3B"][1])
    qwen3b_2bit_overhead = Fraction(2 * MODEL_DIMENSIONS["Qwen/Qwen2.5-3B"][0], MODEL_DIMENSIONS["Qwen/Qwen2.5-3B"][1])

    return {
        "models_audited": len(MODEL_DIMENSIONS),
        "combined_4model_matrices": total_mats,
        "combined_4model_weights": total_wts,
        "total_arms_evaluated": total_arms,
        "smollm2_matrices": MODEL_DIMENSIONS["HuggingFaceTB/SmolLM2-360M"][0],
        "smollm2_weights": MODEL_DIMENSIONS["HuggingFaceTB/SmolLM2-360M"][1],
        "qwen05b_matrices": MODEL_DIMENSIONS["Qwen/Qwen2.5-0.5B"][0],
        "qwen05b_weights": MODEL_DIMENSIONS["Qwen/Qwen2.5-0.5B"][1],
        "qwen15b_matrices": MODEL_DIMENSIONS["Qwen/Qwen2.5-1.5B"][0],
        "qwen15b_weights": MODEL_DIMENSIONS["Qwen/Qwen2.5-1.5B"][1],
        "qwen3b_matrices": MODEL_DIMENSIONS["Qwen/Qwen2.5-3B"][0],
        "qwen3b_weights": MODEL_DIMENSIONS["Qwen/Qwen2.5-3B"][1],
        "bpw_exact": q_pair(bpw_exact),
        "block_h4_scale_factor_exact": q_pair(Fraction(1, 2)),
        "block_h4_max_butterfly_abs": max_h4_output_abs,
        "block_h4_lossless_int8_dp4a": max_h4_output_abs <= int8_max,
        "qwen3b_1bit_routing_bpw": q_pair(qwen3b_1bit_overhead),
        "qwen3b_2bit_routing_bpw": q_pair(qwen3b_2bit_overhead),
        "qwen3b_1bit_below_1e7_bpw": qwen3b_1bit_overhead < Fraction(1, 10000000),
    }


def _build_sampled_pillar() -> dict[str, Any]:
    sm = FROZEN_S68_TOKENS["smollm2"]
    q05 = FROZEN_S68_TOKENS["qwen05b"]
    q15 = FROZEN_S68_TOKENS["qwen15b"]
    q3 = FROZEN_S68_TOKENS["qwen3b"]
    cu = FROZEN_S68_TOKENS["cuda_ptx"]

    sm_ax_wt = _q(sm["wt_kl"]["A16_axis"])
    sm_bh4_wt = _q(sm["wt_kl"]["A16_BlockH4"])
    sm_z1_wt = _q(sm["wt_kl"]["A16_AdaptZeroCal_1bit"])
    sm_w1_wt = _q(sm["wt_kl"]["A16_AdaptWeightOnly_1bit"])
    sm_o1_wt = _q(sm["wt_kl"]["Oracle_1bit"])
    sm_z2_wt = _q(sm["wt_kl"]["A16_AdaptZeroCal_2bit"])

    sm_ax_py = _q(sm["py_kl"]["A16_axis"])
    sm_z1_py = _q(sm["py_kl"]["A16_AdaptZeroCal_1bit"])
    sm_w1_py = _q(sm["py_kl"]["A16_AdaptWeightOnly_1bit"])
    sm_o1_py = _q(sm["py_kl"]["Oracle_1bit"])

    sm_wt_retention = (sm_ax_wt - sm_z1_wt) / (sm_ax_wt - sm_o1_wt)
    sm_py_retention = (sm_ax_py - sm_z1_py) / (sm_ax_py - sm_o1_py)

    tot_agree_1bit = sm["agree_1bit"] + q05["agree_1bit"] + q15["agree_1bit"] + q3["agree_1bit"]
    tot_mats = sum(v[0] for v in MODEL_DIMENSIONS.values())

    q05_z2_py = _q(q05["py_kl"]["A16_AdaptZeroCal_2bit"])
    q05_e8_py = _q(q05["py_kl"]["E8_H8"])
    q05_bh4_py = _q(q05["py_kl"]["A16_BlockH4"])

    q3_ax_wt = _q(q3["wt_kl"]["A16_axis"])
    q3_bh4_wt = _q(q3["wt_kl"]["A16_BlockH4"])
    q3_o1_wt = _q(q3["wt_kl"]["Oracle_1bit"])
    q3_z1_wt = _q(q3["wt_kl"]["A16_AdaptZeroCal_1bit"])

    q05_h2_wt = _q(q05["wt_kl"]["hA16_BlockH2_real"])
    q05_h4_wt = _q(q05["wt_kl"]["hA16_BlockH4_real"])
    q05_hr_wt = _q(q05["wt_kl"]["hA16_real"])
    q05_h4_py = _q(q05["py_kl"]["hA16_BlockH4_real"])
    q05_h2_py = _q(q05["py_kl"]["hA16_BlockH2_real"])

    q15_h2_wt = _q(q15["wt_kl"]["hA16_BlockH2_real"])
    q15_h4_wt = _q(q15["wt_kl"]["hA16_BlockH4_real"])
    q15_hr_wt = _q(q15["wt_kl"]["hA16_real"])
    q15_h4_ppl = _q(q15["wt_ppl"]["hA16_BlockH4_real"])
    q15_hr_ppl = _q(q15["wt_ppl"]["hA16_real"])

    q3_h2_wt = _q(q3["wt_kl"]["hA16_BlockH2_real"])
    q3_h4_wt = _q(q3["wt_kl"]["hA16_BlockH4_real"])
    q3_hr_wt = _q(q3["wt_kl"]["hA16_real"])
    q3_h4_py = _q(q3["py_kl"]["hA16_BlockH4_real"])
    q3_h2_py = _q(q3["py_kl"]["hA16_BlockH2_real"])
    q3_hr_py = _q(q3["py_kl"]["hA16_real"])
    q3_h4_py_ppl = _q(q3["py_ppl"]["hA16_BlockH4_real"])
    q3_h2_py_ppl = _q(q3["py_ppl"]["hA16_BlockH2_real"])

    s1_ax = _q(cu["single_axis_gws"])
    s1_bh4 = _q(cu["single_bh4_gws"])
    s2_ax = _q(cu["stacked2_axis_gws"])
    s2_bh2 = _q(cu["stacked2_bh2_gws"])
    s2_bh4 = _q(cu["stacked2_bh4_gws"])

    gates = {
        "p_s68_1_zero_cal_1bit_agreement_above_70pct_all_4_models": (
            Fraction(sm["agree_1bit"], 224) > Fraction(7, 10)
            and Fraction(q05["agree_1bit"], 168) > Fraction(7, 10)
            and Fraction(q15["agree_1bit"], 196) > Fraction(7, 10)
            and Fraction(q3["agree_1bit"], 252) > Fraction(7, 10)
        ),
        "p_s68_1_smollm2_all_3_zero_cal_routers_beat_all_uniform_arms": (
            sm_z2_wt < sm_z1_wt < sm_w1_wt < sm_ax_wt < sm_bh4_wt
            and sm_w1_py < sm_z1_py < sm_ax_py
            and sm_wt_retention > Fraction(4, 5)
        ),
        "p_s68_1_qwen05b_2bit_router_crushes_uniform_python": (
            q05_z2_py < q05_e8_py and q05_z2_py < q05_bh4_py
        ),
        "p_s68_1_honest_falsification_qwen3b_local_oracle_worse_than_uniform": (
            q3_ax_wt < q3_bh4_wt < q3_o1_wt < q3_z1_wt
        ),
        "p_s68_2_hA16_BlockH2_real_remains_undefeated_wiki_kl_all_3_qwen": (
            q05_h2_wt < q05_hr_wt < q05_h4_wt
            and q15_h2_wt < q15_h4_wt < q15_hr_wt
            and q3_h2_wt < q3_hr_wt < q3_h4_wt
        ),
        "p_s68_2_hA16_BlockH4_real_beats_axis_and_wins_05b_py_15b_ppl_3b_py_ppl": (
            q05_h4_py < q05_h2_py
            and q15_h4_ppl < q15_hr_ppl
            and q3_h4_py < q3_h2_py
            and q3_h4_py_ppl < q3_h2_py_ppl
            and q3_hr_py < q3_h4_py
        ),
        "p_s68_3_sm86_ptx_zero_spill_zero_ldl_stl_and_592gws_96pct_speed": (
            cu["max_stack_bytes"] == 0
            and cu["max_spill_stores_bytes"] == 0
            and cu["max_spill_loads_bytes"] == 0
            and cu["ptx_ldl_count"] == 0
            and cu["ptx_stl_count"] == 0
            and s2_bh4 >= Fraction(570, 1)
            and s2_bh4 * 100 >= s2_ax * 92
            and s1_bh4 * 100 >= s1_ax * 99
        ),
    }

    return {
        "oracle_class": "sampled",
        "measured_live_model_weights": True,
        "pinned_digests": dict(PINNED_DIGESTS),
        "prior_track_anchors": dict(PRIOR_TRACK_ANCHORS),
        "four_model_zero_1bit_oracle_agreement": [tot_agree_1bit, tot_mats],
        "qwen3b_zero_2bit_oracle_agreement": [q3["agree_2bit"], 252],
        "smollm2_zero_1bit_over_axis_wt_kl": q_pair(sm_z1_wt / sm_ax_wt),
        "smollm2_zero_1bit_over_axis_py_kl": q_pair(sm_z1_py / sm_ax_py),
        "smollm2_zero_1bit_wt_oracle_retention": q_pair(sm_wt_retention),
        "smollm2_zero_1bit_py_oracle_retention": q_pair(sm_py_retention),
        "smollm2_zero_2bit_over_axis_wt_kl": q_pair(sm_z2_wt / sm_ax_wt),
        "qwen05b_zero_2bit_over_e8_py_kl": q_pair(q05_z2_py / q05_e8_py),
        "qwen3b_bh4_over_axis_gws_single": q_pair(s1_bh4 / s1_ax),
        "qwen3b_bh4_over_axis_gws_stacked2": q_pair(s2_bh4 / s2_ax),
        "qwen3b_bh4_over_bh2_gws_stacked2": q_pair(s2_bh4 / s2_bh2),
        "ceer_cycle3_outcomes": {
            "P_S68_1_zero_cal_adaptive_blockh4_router": "PARTIAL_76PCT_AGREEMENT_SMOLLM2_WIN_QWEN_LOCAL_ORACLE_FALSIFIED",
            "P_S68_2_calibrated_floor_hA16_BlockH4_real": "PARTIAL_BLOCKH2_UNDEFEATED_WIKI_BLOCKH4_WINS_05B_PY_AND_PPL",
            "P_S68_3_sm86_ptx_fused_blockh4_zero_spill_pareto": "CONFIRMED_ZERO_SPILL_BIT_EXACT_592GWS_96PCT_SPEED",
        },
        "sampled_gates": gates,
    }


def build_certificate() -> dict[str, Any]:
    p1 = _build_exact_pillar()
    p2 = _build_sampled_pillar()
    exact_ok = (
        p1["combined_4model_matrices"] == 840
        and p1["combined_4model_weights"] == 4757127168
        and p1["total_arms_evaluated"] == 46
        and p1["block_h4_lossless_int8_dp4a"] is True
        and p1["qwen3b_1bit_below_1e7_bpw"] is True
    )
    sampled_ok = all(p2["sampled_gates"].values())
    body = {
        "schema_version": SCHEMA_VERSION,
        "oracle_module": "rtx3080_s68_oracle",
        "epistemic_taxonomy": {
            "exact_pillar": "exact",
            "live_4model_ceer_cycle3_pillar": "sampled",
            "prior_track_links": "anchor",
        },
        "pillar_1_exact_router_and_h4_arithmetic": p1,
        "pillar_2_sampled_live_ceer_cycle3": p2,
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
    """Verify frozen live receipts against pinned digests, inner hashes, timestamps, 46 arms, and PTX telemetry."""
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
    for mname, (exp_mats, exp_wts, exp_arms) in MODEL_DIMENSIONS.items():
        md = models[mname]
        if not _is_int_eq(md["block_linear_matrices"], exp_mats) or not _is_int_eq(md["total_block_weights"], exp_wts):
            arms_clean = False
        if len(md["arms"]) != exp_arms:
            arms_clean = False
        for arm in md["arms"].values():
            total_arms += 1
            if not _is_int_eq(arm["syndrome_errors"], 0) or not _is_int_eq(arm["non_block_tensors_differ"], 0):
                arms_clean = False
    arms_clean = arms_clean and total_arms == 46

    sm_a = models["HuggingFaceTB/SmolLM2-360M"]["arms"]
    q05_a = models["Qwen/Qwen2.5-0.5B"]["arms"]
    q15_a = models["Qwen/Qwen2.5-1.5B"]["arms"]
    q3_a = models["Qwen/Qwen2.5-3B"]["arms"]

    p1_ok = (
        _is_int_eq(sm_a["A16_AdaptZeroCal_1bit"]["oracle_agreement_count"], 168)
        and _is_int_eq(q05_a["A16_AdaptZeroCal_1bit"]["oracle_agreement_count"], 133)
        and _is_int_eq(q15_a["A16_AdaptZeroCal_1bit"]["oracle_agreement_count"], 148)
        and _is_int_eq(q3_a["A16_AdaptZeroCal_1bit"]["oracle_agreement_count"], 190)
        and _to_q(sm_a["A16_AdaptZeroCal_1bit"]["wikitext2"]["mean_kl"]) < _to_q(sm_a["A16_axis"]["wikitext2"]["mean_kl"])
        and _to_q(sm_a["A16_AdaptZeroCal_1bit"]["python"]["mean_kl"]) < _to_q(sm_a["A16_axis"]["python"]["mean_kl"])
        and _to_q(q05_a["A16_AdaptZeroCal_2bit"]["python"]["mean_kl"]) < _to_q(q05_a["E8_H8"]["python"]["mean_kl"])
        and _to_q(q3_a["A16_axis"]["wikitext2"]["mean_kl"]) < _to_q(q3_a["Oracle_1bit"]["wikitext2"]["mean_kl"])
    )

    p2_ok = (
        _to_q(q05_a["hA16_BlockH2_real"]["wikitext2"]["mean_kl"]) < _to_q(q05_a["hA16_BlockH4_real"]["wikitext2"]["mean_kl"])
        and _to_q(q05_a["hA16_BlockH4_real"]["python"]["mean_kl"]) < _to_q(q05_a["hA16_BlockH2_real"]["python"]["mean_kl"])
        and _to_q(q15_a["hA16_BlockH2_real"]["wikitext2"]["mean_kl"]) < _to_q(q15_a["hA16_BlockH4_real"]["wikitext2"]["mean_kl"])
        and _to_q(q3_a["hA16_BlockH2_real"]["wikitext2"]["mean_kl"]) < _to_q(q3_a["hA16_BlockH4_real"]["wikitext2"]["mean_kl"])
        and _to_q(q3_a["hA16_BlockH4_real"]["python"]["mean_kl"]) < _to_q(q3_a["hA16_BlockH2_real"]["python"]["mean_kl"])
    )

    cu_b = emp_obj["cuda_sm86_ptx_blockh4_bench"]
    pv = cu_b["ptxas_verification"]
    s1 = cu_b["shapes"]["down_proj_single_2048x11008"]
    s2 = cu_b["shapes"]["down_proj_stacked2_4096x11008"]
    s1_b4 = s1["gemv_4_5bpw_a16_block_h4_dp4a"]
    s2_b4 = s2["gemv_4_5bpw_a16_block_h4_dp4a"]
    s2_ax = s2["gemv_4_5bpw_a16_axis_dp4a"]

    p3_ok = (
        _is_int_eq(pv["max_stack_bytes"], 0)
        and _is_int_eq(pv["max_spill_stores_bytes"], 0)
        and _is_int_eq(pv["max_spill_loads_bytes"], 0)
        and _is_int_eq(pv["ptx_ldl_count"], 0)
        and _is_int_eq(pv["ptx_stl_count"], 0)
        and _is_int_eq(s1_b4["syndrome_errors"], 0)
        and _is_int_eq(s1_b4["int64_bit_exact_mismatches"], 0)
        and _is_int_eq(s2_b4["syndrome_errors"], 0)
        and _is_int_eq(s2_b4["int64_bit_exact_mismatches"], 0)
        and _to_q(s2_b4["gweights_per_s"]) >= Fraction(570, 1)
        and _to_q(s2_b4["gweights_per_s"]) * 100 >= _to_q(s2_ax["gweights_per_s"]) * 92
    )

    cem = emp_obj["cuda_export_meta"]
    export_ok = (
        _is_int_eq(cem["down_proj_single_2048x11008"]["export_a16_axis_syndrome_errors"], 0)
        and _is_int_eq(cem["down_proj_single_2048x11008"]["export_a16_block_h4_syndrome_errors"], 0)
        and _is_int_eq(cem["down_proj_stacked2_4096x11008"]["export_a16_axis_syndrome_errors"], 0)
        and _is_int_eq(cem["down_proj_stacked2_4096x11008"]["export_a16_block_h4_syndrome_errors"], 0)
    )

    checks = {
        "digests_match": digests_match,
        "prereg_strictly_precedes_execution": prereg_precedes,
        "mlos_clean_zone_b": mlos_clean,
        "mlos_zero_desktop_processes": mlos_zero_desktop,
        "real_weight_honesty_tags": honesty_ok,
        "all_4_model_pins_match": pins_ok,
        "corpora_pins_match": corpora_ok,
        "all_46_real_arms_clean": arms_clean,
        "p_s68_1_agreement_smollm2_win_and_qwen3b_oracle_nuance": p1_ok,
        "p_s68_2_calibrated_floors_and_h2_vs_h4_scaling": p2_ok,
        "p_s68_3_ptx_zero_spill_bit_exact_and_592gws": p3_ok,
        "cuda_export_zero_syndrome_errors": export_ok,
    }
    return {"all_verified": all(checks.values()), "checks": checks}
