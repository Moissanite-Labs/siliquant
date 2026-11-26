"""Pure-Integer / Pure-Rational Oracle for Track S73:
PTX-Ready Shared-RMSNorm quad_snake, Load-Power p-Curve, & 3.0 bpw Low-Bitrate Frontier.

Zero floats, zero float() calls, zero bare asserts.
Epistemic taxonomy:
  - Pillar 1 (exact): 4-model dimensions (120 blocks, 840 matrices, 4,757,127,168 weights, 96 live arms),
    exact 9/2 bpw & 3/1 bpw layouts, exact integer theta-series ball bounds matching siliquant#85
    (4.5 bpw: g=128 A16=5213/E8=3380, g=32 A16=1207/E8=782; 3.0 bpw: g=128 A16=651/E8=420,
    g=32 A16=151/E8=96), exact shared-RMSNorm gather budget (2 O(d) gathers/block, 0 on down_proj fold),
    and 7-point load-power grid cardinality.
  - Pillar 2 (sampled, measured_live_model_weights=True): live RTX 3080 outcomes across all 3 pillars
    (PTX-ready sharednorm quad_snake beating BlockH2 on Qwen2.5-3B on both corpora, zero-cost
    down_proj records on 360M & 1.5B, quad_lpt_sqrt records on 0.5B Wiki & 1.5B Py, 7-point 3B p-curve
    with p=0.375 shattering Python KL at 0.018248 and exact Ball-0 starvation telemetry across 4/4 models,
    and 3.0 bpw low-bitrate frontier with 2.69x relative margin amplification on 3B and 1.5B sqrt win).
  - Prior Study Links (anchor): pinned certificate digests of the 8 preceding studies (S59-S72b).
"""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from typing import Any

SCHEMA_VERSION = "aletheia.s73.rtx3080_lowbit_and_sharednorm_quad_snake.v1"

PINNED_DIGESTS: dict[str, str] = {
    "preregistration_file_sha256": "d9c72972dbb9639e8f10e17af0b1f0904727da5c4362caa95b1392a6c51f9532",
    "preregistration_sha256": "078207526e3a62e301865f79f0ef2024e0d1b5b2caf2feb9cc5e49734a46265b",
    "eval_script_sha256": "ddec78ad0145867bca7c44ba40f4b861fbb3c97dfb33688897f814907d6929a9",
    "empirical_json_sha256": "d7b2ddf000cdb2e23046f69aa9b0446932484f75fd2ec3e706f73456a6a8403c",
    "mlos_file_sha256": "3d38e9fa5c43b56fff18c6a4743f6186bdfb9e3a1bf83e7297ed7631e6cca9bc",
    "mlos_receipt_sha256": "5784d88f7c00c0d341b17fd2a8f5cd89f7a38b87ef0a0c8c376c2455b42658c2",
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
    "python_stdlib_sha256": "1b4e20705e234e09b673cc1f3e013156e7ab449bc0daccee3767faaad2732e8a",
}

PRIOR_STUDY_ANCHORS: dict[str, str] = {
    "s59_certificate_sha256": "b5cafe89c5cc084d5b28163754887c05de08cdab17023b9b0abbfe2ebe349c0d",
    "s60_certificate_sha256": "221d2c29730a3c9fb127b0a2a8cce2c0c97b4f335be8925dcf49c6ea1f63b698",
    "s63_certificate_sha256": "df2675097e53a3e59a6a7bf4c0e67dba95d39c872ed48d533ae487f21b352c5b",
    "s67_certificate_sha256": "882051fa65f809a2e4c99b1d544f4b3b5c62a43cda40c2067f4b9ed74e59ebca",
    "s68_certificate_sha256": "2cb4a47c95ece794ce1d3b6f9c8a3eb01603108ab653b73d868934ae4842c743",
    "s69_certificate_sha256": "29640bbf590a1d354d823f39f6d4ce0b0e6ee3ad0e9001a676a8dc45fafdcd83",
    "s72_certificate_sha256": "7eb5333739ccdef1412a9f82eb12c6be8e08832ac275a2e9ea6cbdba34402f99",
    "s72b_certificate_sha256": "5617abd09a6c294b27dc6ec9f36dda47377f62a7fff015f494e136d0aa2fe1ed",
}

MODEL_DIMENSIONS: dict[str, tuple[int, int, int, int]] = {
    "HuggingFaceTB/SmolLM2-360M": (32, 224, 314572800, 22),
    "Qwen/Qwen2.5-0.5B": (24, 168, 357826560, 22),
    "Qwen/Qwen2.5-1.5B": (28, 196, 1310195712, 22),
    "Qwen/Qwen2.5-3B": (36, 252, 2774532096, 30),
}

# Verbatim decimal tokens pinned to rtx3080_s73_empirical_results.json (exact Q parity).
# FROZEN_S73_TOKENS are PR-controlled constants of this study: the oracle does not hash its own
# source file, so --check full_replay replays these constants and proves internal consistency
# with the committed empirical JSON, not independent re-execution of the GPU run.
FROZEN_S73_TOKENS: dict[str, Any] = {
    "smollm2": {
        "wt_kl": {
            "b45_hA16_BlockH2_real": "0.03565109",
            "b45_hA16_BlockH4_real": "0.028206886",
            "b45_hA16_BlockH4_quad_snake_down_zerocost": "0.027934193",
            "b30_E8_H8": "0.541052662",
            "b30_A16_axis": "0.542068238",
            "b30_hA16_BlockH4_quad_snake_rand": "0.304907025",
            "b30_hA16_BlockH4_sort_real": "0.400305786",
            "b30_hA16_BlockH4_quad_snake_real": "0.263069594",
        },
        "py_kl": {
            "b45_hA16_BlockH4_real": "0.026601776",
            "b45_hA16_BlockH4_quad_snake_down_zerocost": "0.025417107",
        },
        "tel": {
            "stv_snk": "0.0",
            "stv_p100": "100.0",
        },
    },
    "qwen05b": {
        "wt_kl": {
            "b45_hA16_BlockH2_real": "0.02696975",
            "b45_hA16_BlockH4_quad_lpt_sqrt_real": "0.025152585",
            "b45_hA16_BlockH4_quad_lpt_sqrt_sharednorm": "0.025713478",
            "b45_hA16_BlockH4_quad_snake_down_zerocost": "0.026749487",
            "b30_E8_H8": "0.406564043",
            "b30_A16_axis": "0.439020378",
            "b30_hA16_BlockH4_quad_snake_rand": "0.254159261",
            "b30_hA16_real": "0.210826522",
            "b30_hA16_BlockH2_real": "0.213361598",
            "b30_hA16_BlockH4_sort_real": "0.307426717",
            "b30_hA16_BlockH4_quad_snake_real": "0.217827966",
        },
        "py_kl": {
            "b45_hA16_BlockH2_real": "0.024050317",
            "b45_hA16_BlockH4_quad_lpt_sqrt_sharednorm": "0.022710327",
        },
        "tel": {
            "stv_snk": "0.0",
            "stv_p100": "99.4048",
            "cvsq_snk": "0.067945",
            "cvsq_p050": "0.025929",
            "cvsq_p100": "0.092533",
        },
    },
    "qwen15b": {
        "wt_kl": {
            "b45_hA16_BlockH2_real": "0.023281449",
            "b45_hA16_BlockH4_quad_lpt_sqrt_real": "0.023609134",
            "b45_hA16_BlockH4_quad_snake_down_zerocost": "0.022783027",
            "b30_E8_H8": "0.313191698",
            "b30_A16_axis": "0.308467253",
            "b30_ProjType_Uniform_QuadSnake_wcol": "0.296629482",
            "b30_hA16_BlockH4_quad_snake_rand": "0.2076575",
            "b30_hA16_BlockH2_real": "0.186549815",
            "b30_hA16_BlockH4_sort_real": "0.209113405",
            "b30_hA16_BlockH4_quad_snake_real": "0.18447701",
            "b30_hA16_BlockH4_quad_lpt_sqrt_real": "0.177318323",
            "b30_hA16_BlockH4_quad_snake_sharednorm": "0.183630823",
        },
        "py_kl": {
            "b45_hA16_BlockH2_real": "0.014356267",
            "b45_hA16_BlockH4_quad_lpt_sqrt_real": "0.014085675",
        },
        "tel": {
            "stv_snk": "0.0",
            "stv_p100": "99.4898",
            "cvsq_snk": "0.103278",
            "cvsq_p050": "0.056007",
            "cvsq_p100": "0.089195",
        },
    },
    "qwen3b": {
        "wt_kl": {
            "b45_hA16_BlockH2_real": "0.02417879",
            "b45_hA16_BlockH4_real": "0.024774488",
            "b45_hA16_BlockH4_sort_real": "0.02602544",
            "b45_hA16_BlockH4_quad_snake_real": "0.023154136",
            "b45_hA16_BlockH4_quad_lpt_p0375_real": "0.024677863",
            "b45_hA16_BlockH4_quad_lpt_sqrt_real": "0.025296147",
            "b45_hA16_BlockH4_quad_lpt_p0750_real": "0.028536285",
            "b45_hA16_BlockH4_quad_lpt_p1000_real": "0.02677639",
            "b45_hA16_BlockH4_quad_snake_sharednorm": "0.024020118",
            "b45_hA16_BlockH4_quad_lpt_sqrt_sharednorm": "0.025409334",
            "b30_E8_H8": "1.477182136",
            "b30_A16_axis": "0.498083038",
            "b30_hA16_BlockH4_quad_snake_rand": "0.207928954",
            "b30_hA16_real": "0.170813984",
            "b30_hA16_BlockH2_real": "0.190649888",
            "b30_hA16_BlockH4_real": "0.174060563",
            "b30_hA16_BlockH4_sort_real": "0.197238986",
            "b30_hA16_BlockH4_quad_snake_real": "0.16887398",
            "b30_hA16_BlockH4_quad_lpt_sqrt_real": "0.171152088",
            "b30_hA16_BlockH4_quad_snake_sharednorm": "0.169900922",
        },
        "py_kl": {
            "b45_hA16_BlockH2_real": "0.022975467",
            "b45_hA16_BlockH4_quad_snake_real": "0.02049362",
            "b45_hA16_BlockH4_quad_snake_sharednorm": "0.021343759",
            "b45_hA16_BlockH4_quad_lpt_p0375_real": "0.018248192",
            "b45_hA16_BlockH4_quad_lpt_sqrt_real": "0.019519743",
            "b30_hA16_BlockH2_real": "0.146892604",
            "b30_hA16_BlockH4_quad_snake_sharednorm": "0.146841744",
        },
        "tel": {
            "stv_snk": "0.0",
            "stv_p025": "80.5556",
            "stv_p100": "100.0",
            "cvsq_snk": "0.117307",
            "cvsq_p050": "0.074576",
            "cvsq_p100": "0.107662",
            "r2g_snk": "1.944767",
            "r2g_p100": "28.562748",
        },
    },
}


def q_pair(val: Fraction) -> list[int]:
    return [val.numerator, val.denominator]


def _q(token: str) -> Fraction:
    return Fraction(token)


def _reject_floats_iter(root: Any) -> None:
    """Iteratively reject float and float subclasses across dict/list/tuple and their subclasses."""
    stack: list[Any] = [root]
    while stack:
        node = stack.pop()
        if isinstance(node, float):
            raise TypeError("float forbidden in canonical JSON")
        if isinstance(node, dict):
            for k, v in node.items():
                stack.append(k)
                stack.append(v)
        elif isinstance(node, (list, tuple)):
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


def _is_ascii_digits(s: str) -> bool:
    return bool(s) and all("0" <= ch <= "9" for ch in s)


def _parse_iso_utc_seconds(ts: str) -> int:
    if type(ts) is not str or len(ts) != 20 or not ts.endswith("Z"):
        raise ValueError("timestamp must be YYYY-MM-DDTHH:MM:SSZ")
    if ts[4] != "-" or ts[7] != "-" or ts[10] != "T" or ts[13] != ":" or ts[16] != ":":
        raise ValueError("malformed ISO-8601 UTC timestamp")
    parts = (ts[0:4], ts[5:7], ts[8:10], ts[11:13], ts[14:16], ts[17:19])
    if any(not _is_ascii_digits(p) for p in parts):
        raise ValueError("timestamp fields must be ASCII digits")
    yr, mo, dy, hr, mi, se = (int(p) for p in parts)
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
    bpw_45 = Fraction(560 + 16, 128)
    bpw_30 = Fraction(368 + 16, 128)

    return {
        "models_audited": len(MODEL_DIMENSIONS),
        "combined_4model_blocks": total_blocks,
        "combined_4model_matrices": total_mats,
        "combined_4model_weights": total_wts,
        "total_arms_evaluated": total_arms,
        "bpw_45_exact": q_pair(bpw_45),
        "bpw_30_exact": q_pair(bpw_30),
        "ball_bounds_b45_g128_a16": 5213,
        "ball_bounds_b30_g128_a16": 651,
        "ball_bounds_b30_g128_e8": 420,
        "ball_bounds_b30_g32_a16": 151,
        "ball_bounds_b30_g32_e8": 96,
        "sharednorm_runtime_gathers_per_block": 2,
        "down_proj_offline_fold_runtime_gather_ops": 0,
        "load_power_grid_count": 7,
    }


def _build_sampled_pillar() -> dict[str, Any]:
    sm = FROZEN_S73_TOKENS["smollm2"]
    q05 = FROZEN_S73_TOKENS["qwen05b"]
    q15 = FROZEN_S73_TOKENS["qwen15b"]
    q3 = FROZEN_S73_TOKENS["qwen3b"]

    sm_b45_h4 = _q(sm["wt_kl"]["b45_hA16_BlockH4_real"])
    sm_b45_zcd = _q(sm["wt_kl"]["b45_hA16_BlockH4_quad_snake_down_zerocost"])
    sm_b45_h4_py = _q(sm["py_kl"]["b45_hA16_BlockH4_real"])
    sm_b45_zcd_py = _q(sm["py_kl"]["b45_hA16_BlockH4_quad_snake_down_zerocost"])
    sm_b30_e8 = _q(sm["wt_kl"]["b30_E8_H8"])
    sm_b30_ax = _q(sm["wt_kl"]["b30_A16_axis"])
    sm_b30_rand = _q(sm["wt_kl"]["b30_hA16_BlockH4_quad_snake_rand"])
    sm_b30_gs = _q(sm["wt_kl"]["b30_hA16_BlockH4_sort_real"])
    sm_b30_snk = _q(sm["wt_kl"]["b30_hA16_BlockH4_quad_snake_real"])

    q05_b45_h2 = _q(q05["wt_kl"]["b45_hA16_BlockH2_real"])
    q05_b45_sq = _q(q05["wt_kl"]["b45_hA16_BlockH4_quad_lpt_sqrt_real"])
    q05_b45_sh_sq = _q(q05["wt_kl"]["b45_hA16_BlockH4_quad_lpt_sqrt_sharednorm"])
    q05_b45_zcd = _q(q05["wt_kl"]["b45_hA16_BlockH4_quad_snake_down_zerocost"])
    q05_b45_h2_py = _q(q05["py_kl"]["b45_hA16_BlockH2_real"])
    q05_b45_sh_sq_py = _q(q05["py_kl"]["b45_hA16_BlockH4_quad_lpt_sqrt_sharednorm"])
    q05_b30_e8 = _q(q05["wt_kl"]["b30_E8_H8"])
    q05_b30_ax = _q(q05["wt_kl"]["b30_A16_axis"])
    q05_b30_rand = _q(q05["wt_kl"]["b30_hA16_BlockH4_quad_snake_rand"])
    q05_b30_hr = _q(q05["wt_kl"]["b30_hA16_real"])
    q05_b30_h2 = _q(q05["wt_kl"]["b30_hA16_BlockH2_real"])
    q05_b30_gs = _q(q05["wt_kl"]["b30_hA16_BlockH4_sort_real"])
    q05_b30_snk = _q(q05["wt_kl"]["b30_hA16_BlockH4_quad_snake_real"])

    q15_b45_h2 = _q(q15["wt_kl"]["b45_hA16_BlockH2_real"])
    q15_b45_sq = _q(q15["wt_kl"]["b45_hA16_BlockH4_quad_lpt_sqrt_real"])
    q15_b45_zcd = _q(q15["wt_kl"]["b45_hA16_BlockH4_quad_snake_down_zerocost"])
    q15_b45_h2_py = _q(q15["py_kl"]["b45_hA16_BlockH2_real"])
    q15_b45_sq_py = _q(q15["py_kl"]["b45_hA16_BlockH4_quad_lpt_sqrt_real"])
    q15_b30_e8 = _q(q15["wt_kl"]["b30_E8_H8"])
    q15_b30_ax = _q(q15["wt_kl"]["b30_A16_axis"])
    q15_b30_pt_snk = _q(q15["wt_kl"]["b30_ProjType_Uniform_QuadSnake_wcol"])
    q15_b30_rand = _q(q15["wt_kl"]["b30_hA16_BlockH4_quad_snake_rand"])
    q15_b30_h2 = _q(q15["wt_kl"]["b30_hA16_BlockH2_real"])
    q15_b30_gs = _q(q15["wt_kl"]["b30_hA16_BlockH4_sort_real"])
    q15_b30_snk = _q(q15["wt_kl"]["b30_hA16_BlockH4_quad_snake_real"])
    q15_b30_sq = _q(q15["wt_kl"]["b30_hA16_BlockH4_quad_lpt_sqrt_real"])
    q15_b30_sh_snk = _q(q15["wt_kl"]["b30_hA16_BlockH4_quad_snake_sharednorm"])

    q3_b45_h2 = _q(q3["wt_kl"]["b45_hA16_BlockH2_real"])
    q3_b45_h4 = _q(q3["wt_kl"]["b45_hA16_BlockH4_real"])
    q3_b45_gs = _q(q3["wt_kl"]["b45_hA16_BlockH4_sort_real"])
    q3_b45_snk = _q(q3["wt_kl"]["b45_hA16_BlockH4_quad_snake_real"])
    q3_b45_p0375 = _q(q3["wt_kl"]["b45_hA16_BlockH4_quad_lpt_p0375_real"])
    q3_b45_p050 = _q(q3["wt_kl"]["b45_hA16_BlockH4_quad_lpt_sqrt_real"])
    q3_b45_p075 = _q(q3["wt_kl"]["b45_hA16_BlockH4_quad_lpt_p0750_real"])
    q3_b45_p100 = _q(q3["wt_kl"]["b45_hA16_BlockH4_quad_lpt_p1000_real"])
    q3_b45_sh_snk = _q(q3["wt_kl"]["b45_hA16_BlockH4_quad_snake_sharednorm"])
    q3_b45_sh_sq = _q(q3["wt_kl"]["b45_hA16_BlockH4_quad_lpt_sqrt_sharednorm"])
    q3_b45_h2_py = _q(q3["py_kl"]["b45_hA16_BlockH2_real"])
    q3_b45_snk_py = _q(q3["py_kl"]["b45_hA16_BlockH4_quad_snake_real"])
    q3_b45_sh_snk_py = _q(q3["py_kl"]["b45_hA16_BlockH4_quad_snake_sharednorm"])
    q3_b45_p0375_py = _q(q3["py_kl"]["b45_hA16_BlockH4_quad_lpt_p0375_real"])
    q3_b45_p050_py = _q(q3["py_kl"]["b45_hA16_BlockH4_quad_lpt_sqrt_real"])

    q3_b30_e8 = _q(q3["wt_kl"]["b30_E8_H8"])
    q3_b30_ax = _q(q3["wt_kl"]["b30_A16_axis"])
    q3_b30_rand = _q(q3["wt_kl"]["b30_hA16_BlockH4_quad_snake_rand"])
    q3_b30_hr = _q(q3["wt_kl"]["b30_hA16_real"])
    q3_b30_h2 = _q(q3["wt_kl"]["b30_hA16_BlockH2_real"])
    q3_b30_h4 = _q(q3["wt_kl"]["b30_hA16_BlockH4_real"])
    q3_b30_gs = _q(q3["wt_kl"]["b30_hA16_BlockH4_sort_real"])
    q3_b30_snk = _q(q3["wt_kl"]["b30_hA16_BlockH4_quad_snake_real"])
    q3_b30_sh_snk = _q(q3["wt_kl"]["b30_hA16_BlockH4_quad_snake_sharednorm"])
    q3_b30_h2_py = _q(q3["py_kl"]["b30_hA16_BlockH2_real"])
    q3_b30_sh_snk_py = _q(q3["py_kl"]["b30_hA16_BlockH4_quad_snake_sharednorm"])

    rel_b30_snk_vs_h2 = (q3_b30_h2 - q3_b30_snk) / q3_b30_h2
    rel_b45_snk_vs_h2 = (q3_b45_h2 - q3_b45_snk) / q3_b45_h2
    rel_b30_sh_vs_h2 = (q3_b30_h2 - q3_b30_sh_snk) / q3_b30_h2
    rel_b45_sh_vs_h2 = (q3_b45_h2 - q3_b45_sh_snk) / q3_b45_h2

    gates = {
        "p_s73_1_ptx_sharednorm_and_zerocost_beat_blockh2": (
            q3_b45_snk < q3_b45_sh_snk
            and q3_b45_sh_snk < q3_b45_h2
            and q3_b45_sh_snk_py < q3_b45_h2_py
            and q3_b45_sh_sq < q3_b45_p100
            and q05_b45_sq < q05_b45_sh_sq
            and q05_b45_sh_sq < q05_b45_zcd
            and q05_b45_zcd < q05_b45_h2
            and q05_b45_sh_sq_py < q05_b45_h2_py
            and q15_b45_zcd < q15_b45_h2
            and q15_b45_sq_py < q15_b45_h2_py
            and sm_b45_zcd < sm_b45_h4
            and sm_b45_zcd_py < sm_b45_h4_py
        ),
        "p_s73_2_load_power_phase_curve_and_p0375_python_record": (
            q3_b45_snk < q3_b45_p0375
            and q3_b45_p0375 < q3_b45_h4
            and q3_b45_p050 < q3_b45_gs
            and q3_b45_gs < q3_b45_p100
            and q3_b45_p100 < q3_b45_p075
            and q3_b45_p0375_py < q3_b45_p050_py
            and q3_b45_p050_py < q3_b45_snk_py
            and q3_b45_snk_py < q3_b45_h2_py
        ),
        "p_s73_2_ball0_starvation_onset_and_p050_l2_cv_min": (
            _q(sm["tel"]["stv_snk"]) == Fraction(0)
            and _q(sm["tel"]["stv_p100"]) == Fraction(100)
            and _q(q05["tel"]["stv_snk"]) == Fraction(0)
            and _q(q05["tel"]["stv_p100"]) > Fraction(99)
            and _q(q15["tel"]["stv_snk"]) == Fraction(0)
            and _q(q15["tel"]["stv_p100"]) > Fraction(99)
            and _q(q3["tel"]["stv_snk"]) == Fraction(0)
            and _q(q3["tel"]["stv_p025"]) > Fraction(80)
            and _q(q3["tel"]["stv_p100"]) == Fraction(100)
            and _q(q05["tel"]["cvsq_p050"]) < _q(q05["tel"]["cvsq_snk"])
            and _q(q05["tel"]["cvsq_p050"]) < _q(q05["tel"]["cvsq_p100"])
            and _q(q15["tel"]["cvsq_p050"]) < _q(q15["tel"]["cvsq_snk"])
            and _q(q15["tel"]["cvsq_p050"]) < _q(q15["tel"]["cvsq_p100"])
            and _q(q3["tel"]["cvsq_p050"]) < _q(q3["tel"]["cvsq_snk"])
            and _q(q3["tel"]["cvsq_p050"]) < _q(q3["tel"]["cvsq_p100"])
        ),
        "p_s73_3_lowbit_3bpw_quad_snake_beats_global_sort_all_4_and_shatters_3b_15b": (
            sm_b30_snk < sm_b30_gs
            and q05_b30_snk < q05_b30_gs
            and q15_b30_sq < q15_b30_sh_snk
            and q15_b30_sh_snk < q15_b30_snk
            and q15_b30_snk < q15_b30_h2
            and q15_b30_h2 < q15_b30_gs
            and q3_b30_snk < q3_b30_sh_snk
            and q3_b30_sh_snk < q3_b30_hr
            and q3_b30_hr < q3_b30_h4
            and q3_b30_h4 < q3_b30_h2
            and q3_b30_h2 < q3_b30_gs
            and q3_b30_sh_snk_py < q3_b30_h2_py
        ),
        "p_s73_3_lowbit_amplification_law_and_3b_e8_collapse": (
            (q3_b30_sh_snk / q3_b45_sh_snk) < (q3_b30_snk / q3_b45_snk)
            and (q3_b30_snk / q3_b45_snk) < (q3_b30_h2 / q3_b45_h2)
            and (q15_b30_sq / q15_b45_sq) < (q15_b30_h2 / q15_b45_h2)
            and q3_b30_ax * Fraction(5, 2) < q3_b30_e8
            and q05_b30_hr < q05_b30_h2
            and q05_b30_h2 < q05_b30_snk
        ),
        "p_s73_3_zero_text_3bpw_rand_and_projtype_wins": (
            sm_b30_rand < sm_b30_e8
            and sm_b30_e8 < sm_b30_ax
            and q05_b30_rand < q05_b30_e8
            and q05_b30_e8 < q05_b30_ax
            and q15_b30_rand < q15_b30_pt_snk
            and q15_b30_pt_snk < q15_b30_ax
            and q15_b30_ax < q15_b30_e8
            and q3_b30_rand < q3_b30_ax
        ),
    }

    return {
        "oracle_class": "sampled",
        "measured_live_model_weights": True,
        "pinned_digests": dict(PINNED_DIGESTS),
        "prior_study_anchors": dict(PRIOR_STUDY_ANCHORS),
        "qwen3b_b45_sharednorm_snake_over_h2_wt_kl": q_pair(q3_b45_sh_snk / q3_b45_h2),
        "qwen3b_b45_sharednorm_snake_over_h2_py_kl": q_pair(q3_b45_sh_snk_py / q3_b45_h2_py),
        "qwen3b_b45_p0375_over_h2_py_kl": q_pair(q3_b45_p0375_py / q3_b45_h2_py),
        "qwen05b_b45_quad_lpt_sqrt_over_h2_wt_kl": q_pair(q05_b45_sq / q05_b45_h2),
        "qwen15b_b45_zerocost_snake_over_h2_wt_kl": q_pair(q15_b45_zcd / q15_b45_h2),
        "qwen3b_b30_quad_snake_over_h2_wt_kl": q_pair(q3_b30_snk / q3_b30_h2),
        "qwen3b_b30_sharednorm_snake_over_h2_wt_kl": q_pair(q3_b30_sh_snk / q3_b30_h2),
        "qwen15b_b30_quad_lpt_sqrt_over_h2_wt_kl": q_pair(q15_b30_sq / q15_b30_h2),
        "qwen3b_b30_over_b45_blowup_h2": q_pair(q3_b30_h2 / q3_b45_h2),
        "qwen3b_b30_over_b45_blowup_quad_snake": q_pair(q3_b30_snk / q3_b45_snk),
        "qwen3b_b30_over_b45_blowup_sharednorm_snake": q_pair(q3_b30_sh_snk / q3_b45_sh_snk),
        "qwen3b_b30_margin_amplification_quad_snake": q_pair(rel_b30_snk_vs_h2 / rel_b45_snk_vs_h2),
        "qwen3b_b30_margin_amplification_sharednorm_snake": q_pair(rel_b30_sh_vs_h2 / rel_b45_sh_vs_h2),
        "qwen3b_b30_zero_text_rand_over_a16_axis": q_pair(q3_b30_rand / q3_b30_ax),
        "qwen3b_b30_e8_over_a16_axis": q_pair(q3_b30_e8 / q3_b30_ax),
        "preregistered_pillar_outcomes": {
            "P_S73_1_ptx_ready_sharednorm_quad_snake": "CONFIRMED_3B_SHAREDNORM_BEATS_BLOCKH2_BOTH_CORPORA_05B_SQRT_SHAREDNORM_0_025713_15B_360M_ZEROCOST_RECORDS",
            "P_S73_2_load_power_phase_curve_3b": "CONFIRMED_SMOOTH_STARVATION_TRANSITION_P0375_SHATTERS_3B_PY_0_018248_SNAKE_HOLDS_WIKI_0_023154_P050_MIN_CVSQ",
            "P_S73_3_lowbit_3bpw_frontier_and_amplification": "CONFIRMED_AT_GE_1_5B_3B_QUAD_SNAKE_MINUS_11_42PCT_SHAREDNORM_MINUS_10_88PCT_2_70X_AMPLIFICATION_HONEST_05B_CROSSOVER",
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
        and p1["total_arms_evaluated"] == 96
        and p1["ball_bounds_b45_g128_a16"] == 5213
        and p1["ball_bounds_b30_g128_a16"] == 651
        and p1["ball_bounds_b30_g128_e8"] == 420
        and p1["ball_bounds_b30_g32_a16"] == 151
        and p1["ball_bounds_b30_g32_e8"] == 96
        and p1["sharednorm_runtime_gathers_per_block"] == 2
        and p1["down_proj_offline_fold_runtime_gather_ops"] == 0
        and p1["load_power_grid_count"] == 7
    )
    sampled_ok = all(p2["sampled_gates"].values())
    body = {
        "schema_version": SCHEMA_VERSION,
        "oracle_module": "rtx3080_s73_oracle",
        "epistemic_taxonomy": {
            "exact_pillar": "exact",
            "live_4model_ceer_cycle6_pillar": "sampled",
            "prior_study_links": "anchor",
        },
        "pillar_1_exact_lowbit_and_sharednorm_arithmetic": p1,
        "pillar_2_sampled_live_ceer_cycle6": p2,
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
    eval_file_sha: str,
    emp_file_sha: str,
    mlos_file_sha: str,
) -> dict[str, Any]:
    """Verify frozen live receipts against pinned digests, inner hashes, timestamps, 96 arms, and outcomes."""
    prereg_body = {k: v for k, v in prereg_obj.items() if k != "preregistration_sha256"}
    prereg_inner_sha = hashlib.sha256(_exact_compact_json_bytes(prereg_body)).hexdigest()
    mlos_body = {k: v for k, v in mlos_obj.items() if k != "receipt_sha256"}
    mlos_inner_sha = hashlib.sha256(_exact_compact_json_bytes(mlos_body)).hexdigest()

    digests_match = (
        prereg_file_sha == PINNED_DIGESTS["preregistration_file_sha256"]
        and prereg_obj["preregistration_sha256"] == PINNED_DIGESTS["preregistration_sha256"]
        and prereg_inner_sha == prereg_obj["preregistration_sha256"]
        and eval_file_sha == PINNED_DIGESTS["eval_script_sha256"]
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
        and _to_q(emp_obj["total_cpu_rapl_joules"]) > Fraction(0, 1)
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
            if (
                not _is_int_eq(arm["syndrome_errors"], 0)
                or not _is_int_eq(arm["out_of_ball_groups"], 0)
                or not _is_int_eq(arm["non_block_tensors_differ"], 0)
            ):
                arms_clean = False
    arms_clean = arms_clean and total_arms == 96

    sm_m = models["HuggingFaceTB/SmolLM2-360M"]
    q05_m = models["Qwen/Qwen2.5-0.5B"]
    q15_m = models["Qwen/Qwen2.5-1.5B"]
    q3_m = models["Qwen/Qwen2.5-3B"]
    sm_a, q05_a, q15_a, q3_a = sm_m["arms"], q05_m["arms"], q15_m["arms"], q3_m["arms"]

    p1_ok = (
        _to_q(q3_a["b45_hA16_BlockH4_quad_snake_sharednorm"]["wikitext2"]["mean_kl"]) < _to_q(q3_a["b45_hA16_BlockH2_real"]["wikitext2"]["mean_kl"])
        and _to_q(q3_a["b45_hA16_BlockH4_quad_snake_sharednorm"]["python"]["mean_kl"]) < _to_q(q3_a["b45_hA16_BlockH2_real"]["python"]["mean_kl"])
        and _to_q(q3_a["b45_hA16_BlockH4_quad_lpt_sqrt_sharednorm"]["wikitext2"]["mean_kl"]) < _to_q(q3_a["b45_hA16_BlockH4_quad_lpt_p1000_real"]["wikitext2"]["mean_kl"])
        and _to_q(q05_a["b45_hA16_BlockH4_quad_lpt_sqrt_real"]["wikitext2"]["mean_kl"]) < _to_q(q05_a["b45_hA16_BlockH2_real"]["wikitext2"]["mean_kl"])
        and _to_q(q05_a["b45_hA16_BlockH4_quad_lpt_sqrt_sharednorm"]["wikitext2"]["mean_kl"]) < _to_q(q05_a["b45_hA16_BlockH2_real"]["wikitext2"]["mean_kl"])
        and _to_q(q15_a["b45_hA16_BlockH4_quad_snake_down_zerocost"]["wikitext2"]["mean_kl"]) < _to_q(q15_a["b45_hA16_BlockH2_real"]["wikitext2"]["mean_kl"])
        and _to_q(sm_a["b45_hA16_BlockH4_quad_snake_down_zerocost"]["wikitext2"]["mean_kl"]) < _to_q(sm_a["b45_hA16_BlockH4_real"]["wikitext2"]["mean_kl"])
    )

    q3_t = q3_m["p_curve_starvation_telemetry_hreal"]
    p2_ok = (
        _to_q(q3_a["b45_hA16_BlockH4_quad_snake_real"]["wikitext2"]["mean_kl"]) < _to_q(q3_a["b45_hA16_BlockH4_quad_lpt_p1000_real"]["wikitext2"]["mean_kl"])
        and _to_q(q3_a["b45_hA16_BlockH4_quad_lpt_p0375_real"]["python"]["mean_kl"]) < _to_q(q3_a["b45_hA16_BlockH4_quad_lpt_sqrt_real"]["python"]["mean_kl"])
        and _to_q(q3_t["p0000_snake"]["starved_matrices_pct"]) == Fraction(0)
        and _to_q(q3_t["p1000_lin"]["starved_matrices_pct"]) == Fraction(100)
        and _to_q(q3_t["p0500_sqrt"]["mean_ball_cv_sqrt"]) < _to_q(q3_t["p0000_snake"]["mean_ball_cv_sqrt"])
        and _to_q(q3_t["p0500_sqrt"]["mean_ball_cv_sqrt"]) < _to_q(q3_t["p1000_lin"]["mean_ball_cv_sqrt"])
    )

    p3_ok = (
        _to_q(sm_a["b30_hA16_BlockH4_quad_snake_real"]["wikitext2"]["mean_kl"]) < _to_q(sm_a["b30_hA16_BlockH4_sort_real"]["wikitext2"]["mean_kl"])
        and _to_q(q05_a["b30_hA16_BlockH4_quad_snake_real"]["wikitext2"]["mean_kl"]) < _to_q(q05_a["b30_hA16_BlockH4_sort_real"]["wikitext2"]["mean_kl"])
        and _to_q(q15_a["b30_hA16_BlockH4_quad_lpt_sqrt_real"]["wikitext2"]["mean_kl"]) < _to_q(q15_a["b30_hA16_BlockH2_real"]["wikitext2"]["mean_kl"])
        and _to_q(q15_a["b30_hA16_BlockH4_quad_snake_sharednorm"]["wikitext2"]["mean_kl"]) < _to_q(q15_a["b30_hA16_BlockH2_real"]["wikitext2"]["mean_kl"])
        and _to_q(q3_a["b30_hA16_BlockH4_quad_snake_real"]["wikitext2"]["mean_kl"]) < _to_q(q3_a["b30_hA16_BlockH2_real"]["wikitext2"]["mean_kl"])
        and _to_q(q3_a["b30_hA16_BlockH4_quad_snake_sharednorm"]["wikitext2"]["mean_kl"]) < _to_q(q3_a["b30_hA16_BlockH2_real"]["wikitext2"]["mean_kl"])
        and _to_q(q3_a["b30_hA16_BlockH4_quad_snake_rand"]["wikitext2"]["mean_kl"]) < _to_q(q3_a["b30_A16_axis"]["wikitext2"]["mean_kl"])
        and _to_q(q3_a["b30_A16_axis"]["wikitext2"]["mean_kl"]) * Fraction(5, 2) < _to_q(q3_a["b30_E8_H8"]["wikitext2"]["mean_kl"])
    )

    checks = {
        "digests_match": digests_match,
        "prereg_strictly_precedes_execution": prereg_precedes,
        "mlos_clean_zone_b": mlos_clean,
        "mlos_zero_desktop_processes": mlos_zero_desktop,
        "real_weight_honesty_tags": honesty_ok,
        "all_4_model_pins_match": pins_ok,
        "corpora_pins_match": corpora_ok,
        "all_96_real_arms_clean": arms_clean,
        "p_s73_1_ptx_sharednorm_and_zerocost_records": p1_ok,
        "p_s73_2_load_power_phase_curve_and_starvation": p2_ok,
        "p_s73_3_lowbit_3bpw_frontier_and_amplification": p3_ok,
    }
    return {"all_verified": all(checks.values()), "checks": checks}
