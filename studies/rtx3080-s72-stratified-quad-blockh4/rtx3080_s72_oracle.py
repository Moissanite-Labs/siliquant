"""Pure-Integer / Pure-Rational Oracle for Stratified Quad-Interleaved A16_BlockH4 Study.

Zero floats, zero float() calls, zero bare asserts.
Epistemic taxonomy:
  - Pillar 1 (exact): 4-model dimensions (120 blocks, 840 matrices, 4,757,127,168 weights, 87 arms),
    exact 9/2 bpw layout, exact Block-H4 register butterfly lossless int8_t bound (4 * 7 = 28 <= 127),
    exact dyadic 1/2 scale, exact two-level stratification quad/ball partition invariants (32 quads/ball
    for g=128, 8 quads/ball for g=32), and exact zero-runtime-cost down_proj fold gather count (0).
  - Pillar 2 (sampled, measured_live_model_weights=True): live RTX 3080 outcomes across all 3 pillars
    (two-level stratified quad_lpt & quad_snake calibrated dominance over undefeated BlockH2 across
    all 3 Qwen2.5 models, honest falsification of linear-load LPT on heavy-tailed 3B vs rank-invariant
    quad_snake & sqrt-LPT, zero-text wcol regime dichotomy + 3B cure, exact ball-CV/quad-spread Pareto
    telemetry, and BlockH4 vs BlockH2 matched quad_lpt control).
  - Prior Study Links (anchor): pinned certificate digests of the 6 preceding hardware studies.
"""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from typing import Any

SCHEMA_VERSION = "aletheia.s72.rtx3080_stratified_quad_interleaved_blockh4.v1"

PINNED_DIGESTS: dict[str, str] = {
    "preregistration_file_sha256": "3ba6c33796c1cba457d83f0a37fcc9b8a77898f60f75efc93d00b9a196e6df02",
    "preregistration_sha256": "361bae665103de513002bf9268ff623829ab42aaf9fe7c096cd3e6ca38a4b387",
    "eval_script_sha256": "5dbdfb0c2849b717e349ef91eddbf706f3ba4ea5ca48341cade24ac93eb09554",
    "empirical_json_sha256": "a7c35c646a9c29fa645f207697c74e181ad19d8ce61acbe489a265a132968647",
    "mlos_file_sha256": "93f0b75c3e89ceb971ee9d574c29e98047070b79b01dc368ba645363128853cd",
    "mlos_receipt_sha256": "e6e3ea61c6e8f37e13c5fed30b351fa47e66ea468c2da2a4e19976d38e895072",
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
}

MODEL_DIMENSIONS: dict[str, tuple[int, int, int, int]] = {
    "HuggingFaceTB/SmolLM2-360M": (32, 224, 314572800, 21),
    "Qwen/Qwen2.5-0.5B": (24, 168, 357826560, 21),
    "Qwen/Qwen2.5-1.5B": (28, 196, 1310195712, 21),
    "Qwen/Qwen2.5-3B": (36, 252, 2774532096, 24),
}

# Verbatim decimal tokens pinned to rtx3080_s72_empirical_results.json (exact Q parity).
FROZEN_S72_TOKENS: dict[str, Any] = {
    "smollm2": {
        "wt_kl": {
            "A16_axis": "0.065173014",
            "A16_BlockH4": "0.082001576",
            "A16_BlockH4_sort_wcol": "0.064414333",
            "A16_BlockH4_quad_lpt_wcol": "0.076360925",
            "A16_BlockH4_quad_snake_wcol": "0.06999726",
            "ProjType_Uniform": "0.061837779",
            "hA16_real": "0.030954721",
            "hA16_BlockH2_real": "0.03565109",
            "hA16_BlockH2_quad_lpt_real": "0.031742924",
            "hA16_BlockH4_real": "0.028206886",
            "hA16_BlockH4_sort_real": "0.04576595",
            "hA16_BlockH4_sort_grp128_real": "0.030635795",
            "hA16_BlockH4_quad_lpt_real": "0.03350319",
            "hA16_BlockH4_quad_snake_real": "0.030820768",
            "hA16_BlockH4_quad_lpt_down_zerocost": "0.028933503",
            "hA16_BlockH4_quad_lpt_sharednorm": "0.033220223",
        },
        "wt_ppl": {
            "hA16_BlockH4_real": "16.952402",
            "hA16_BlockH4_quad_lpt_down_zerocost": "16.799468",
        },
        "tel_hreal": {
            "unp_spr": "2.275181",
            "grp_spr": "1.799041",
            "lpt_spr": "0.658205",
            "glob_cv": "2.692134",
            "lpt_cv": "1.803939",
        },
    },
    "qwen05b": {
        "wt_kl": {
            "A16_axis": "0.052525283",
            "A16_BlockH4": "0.056062964",
            "A16_BlockH4_sort_wcol": "0.049760643",
            "A16_BlockH4_quad_lpt_wcol": "0.061150697",
            "hA16_real": "0.028102138",
            "hA16_BlockH2_real": "0.02696975",
            "hA16_BlockH2_quad_lpt_real": "0.026720888",
            "hA16_BlockH4_real": "0.029412649",
            "hA16_BlockH4_sort_real": "0.032504237",
            "hA16_BlockH4_sort_grp128_real": "0.02718994",
            "hA16_BlockH4_quad_lpt_real": "0.025954645",
            "hA16_BlockH4_quad_snake_real": "0.030079819",
            "hA16_BlockH4_quad_lpt_down_zerocost": "0.02696409",
            "hA16_BlockH4_quad_lpt_sharednorm": "0.02628463",
        },
        "py_kl": {
            "hA16_BlockH2_real": "0.024050317",
            "hA16_BlockH4_real": "0.023451835",
            "hA16_BlockH4_quad_lpt_sharednorm": "0.022898961",
            "hA16_BlockH2_quad_lpt_real": "0.022337903",
        },
        "tel_hreal": {
            "unp_spr": "2.330344",
            "grp_spr": "1.001946",
            "lpt_spr": "0.412329",
            "glob_cv": "1.547658",
            "lpt_cv": "0.440352",
        },
    },
    "qwen15b": {
        "wt_kl": {
            "A16_axis": "0.050062399",
            "A16_BlockH4": "0.046550051",
            "A16_BlockH4_sort_wcol": "0.039193287",
            "A16_BlockH4_quad_lpt_wcol": "0.046992542",
            "hA16_real": "0.023636564",
            "hA16_BlockH2_real": "0.023281449",
            "hA16_BlockH2_quad_lpt_real": "0.02402037",
            "hA16_BlockH4_real": "0.023411598",
            "hA16_BlockH4_sort_real": "0.026786727",
            "hA16_BlockH4_sort_grp128_real": "0.023176037",
            "hA16_BlockH4_quad_lpt_real": "0.022963627",
            "hA16_BlockH4_quad_snake_real": "0.023176567",
            "hA16_BlockH4_quad_lpt_down_zerocost": "0.02345904",
            "hA16_BlockH4_quad_lpt_sharednorm": "0.022953003",
        },
        "py_kl": {
            "hA16_BlockH2_quad_lpt_real": "0.015030371",
            "hA16_BlockH4_real": "0.015128765",
            "hA16_BlockH4_quad_lpt_real": "0.014712274",
            "hA16_BlockH4_quad_lpt_sharednorm": "0.014897536",
        },
        "wt_ppl": {
            "hA16_BlockH2_real": "12.927343",
            "hA16_BlockH4_real": "12.79468",
            "hA16_BlockH4_quad_lpt_real": "12.774305",
        },
        "tel_hreal": {
            "unp_spr": "2.308937",
            "grp_spr": "1.10341",
            "lpt_spr": "0.316515",
            "glob_cv": "1.922348",
            "lpt_cv": "0.710218",
        },
    },
    "qwen3b": {
        "wt_kl": {
            "A16_axis": "0.046686444",
            "A16_BlockH4": "0.047941673",
            "A16_BlockH4_sort_wcol": "0.05556036",
            "A16_BlockH4_quad_lpt_wcol": "0.051479628",
            "A16_BlockH4_quad_lpt_wcol_matched": "0.05144985",
            "A16_BlockH4_quad_snake_wcol": "0.04907679",
            "A16_BlockH4_quad_lpt_hrand": "0.048977638",
            "A16_BlockH4_quad_lpt_hrand_matched": "0.049012596",
            "ProjType_Uniform": "0.046336085",
            "ProjType_Uniform_QuadLPT_wcol": "0.047632712",
            "hA16_BlockH4_quad_lpt_rand": "0.027153423",
            "hA16_real": "0.024417064",
            "hA16_BlockH2_real": "0.02417879",
            "hA16_BlockH2_quad_lpt_real": "0.026352707",
            "hA16_BlockH4_real": "0.024774488",
            "hA16_BlockH4_sort_real": "0.02602544",
            "hA16_BlockH4_sort_grp128_real": "0.024917616",
            "hA16_BlockH4_quad_lpt_real": "0.02677639",
            "hA16_BlockH4_quad_lpt_sqrt_real": "0.025296147",
            "hA16_BlockH4_quad_snake_real": "0.023154136",
            "hA16_BlockH4_quad_lpt_down_zerocost": "0.026174985",
            "hA16_BlockH4_quad_lpt_sharednorm": "0.02755345",
        },
        "py_kl": {
            "ProjType_Uniform": "0.034327497",
            "hA16_BlockH4_quad_lpt_rand": "0.021866574",
            "hA16_real": "0.021217129",
            "hA16_BlockH2_real": "0.022975467",
            "hA16_BlockH2_quad_lpt_real": "0.020572193",
            "hA16_BlockH4_real": "0.022518843",
            "hA16_BlockH4_quad_snake_real": "0.02049362",
            "hA16_BlockH4_quad_lpt_sqrt_real": "0.019519743",
        },
        "matched_mse_db_diff": {
            "A16_BlockH4_quad_lpt_wcol_matched": "1.582e-06",
            "A16_BlockH4_quad_lpt_hrand_matched": "1.528e-06",
        },
        "tel_hreal": {
            "unp_spr": "2.274524",
            "grp_spr": "1.133839",
            "lpt_spr": "0.33786",
            "glob_cv": "2.096239",
            "lpt_cv": "0.782983",
            "unp_cv_sqrt": "0.144026",
            "lpt_cv_sqrt": "0.107662",
        },
        "tel_wcol": {
            "unp_spr": "0.390338",
            "lpt_spr": "0.003229",
            "glob_cv": "0.236532",
            "unp_cv": "0.0552",
            "lpt_cv": "0.003546",
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

    max_nibble_abs = 7
    max_h4_output_abs = 4 * max_nibble_abs
    int8_max = 127
    bpw_exact = Fraction(16 * 4 + 8, 16)

    return {
        "models_audited": len(MODEL_DIMENSIONS),
        "combined_4model_blocks": total_blocks,
        "combined_4model_matrices": total_mats,
        "combined_4model_weights": total_wts,
        "total_arms_evaluated": total_arms,
        "bpw_exact": q_pair(bpw_exact),
        "block_h4_scale_factor_exact": q_pair(Fraction(1, 2)),
        "block_h4_max_butterfly_abs": max_h4_output_abs,
        "block_h4_lossless_int8_dp4a": max_h4_output_abs <= int8_max,
        "qwen_quads_per_g128_ball": 128 // 4,
        "smollm2_quads_per_g32_ball": 32 // 4,
        "down_proj_offline_fold_runtime_gather_ops": 0,
    }


def _build_sampled_pillar() -> dict[str, Any]:
    sm = FROZEN_S72_TOKENS["smollm2"]
    q05 = FROZEN_S72_TOKENS["qwen05b"]
    q15 = FROZEN_S72_TOKENS["qwen15b"]
    q3 = FROZEN_S72_TOKENS["qwen3b"]

    sm_h4 = _q(sm["wt_kl"]["hA16_BlockH4_real"])
    sm_gs = _q(sm["wt_kl"]["hA16_BlockH4_sort_real"])
    sm_lpt = _q(sm["wt_kl"]["hA16_BlockH4_quad_lpt_real"])
    sm_snk = _q(sm["wt_kl"]["hA16_BlockH4_quad_snake_real"])
    sm_sh = _q(sm["wt_kl"]["hA16_BlockH4_quad_lpt_sharednorm"])
    sm_h2l = _q(sm["wt_kl"]["hA16_BlockH2_quad_lpt_real"])
    sm_zcd_ppl = _q(sm["wt_ppl"]["hA16_BlockH4_quad_lpt_down_zerocost"])
    sm_h4_ppl = _q(sm["wt_ppl"]["hA16_BlockH4_real"])

    q05_h2 = _q(q05["wt_kl"]["hA16_BlockH2_real"])
    q05_h2l = _q(q05["wt_kl"]["hA16_BlockH2_quad_lpt_real"])
    q05_h4 = _q(q05["wt_kl"]["hA16_BlockH4_real"])
    q05_gs = _q(q05["wt_kl"]["hA16_BlockH4_sort_real"])
    q05_lpt = _q(q05["wt_kl"]["hA16_BlockH4_quad_lpt_real"])
    q05_snk = _q(q05["wt_kl"]["hA16_BlockH4_quad_snake_real"])
    q05_sh = _q(q05["wt_kl"]["hA16_BlockH4_quad_lpt_sharednorm"])
    q05_zcd = _q(q05["wt_kl"]["hA16_BlockH4_quad_lpt_down_zerocost"])
    q05_sh_py = _q(q05["py_kl"]["hA16_BlockH4_quad_lpt_sharednorm"])
    q05_h4_py = _q(q05["py_kl"]["hA16_BlockH4_real"])

    q15_h2 = _q(q15["wt_kl"]["hA16_BlockH2_real"])
    q15_h2l = _q(q15["wt_kl"]["hA16_BlockH2_quad_lpt_real"])
    q15_h4 = _q(q15["wt_kl"]["hA16_BlockH4_real"])
    q15_gs = _q(q15["wt_kl"]["hA16_BlockH4_sort_real"])
    q15_lpt = _q(q15["wt_kl"]["hA16_BlockH4_quad_lpt_real"])
    q15_snk = _q(q15["wt_kl"]["hA16_BlockH4_quad_snake_real"])
    q15_sh = _q(q15["wt_kl"]["hA16_BlockH4_quad_lpt_sharednorm"])
    q15_lpt_py = _q(q15["py_kl"]["hA16_BlockH4_quad_lpt_real"])
    q15_h2l_py = _q(q15["py_kl"]["hA16_BlockH2_quad_lpt_real"])

    q3_ax = _q(q3["wt_kl"]["A16_axis"])
    q3_bh4 = _q(q3["wt_kl"]["A16_BlockH4"])
    q3_swc = _q(q3["wt_kl"]["A16_BlockH4_sort_wcol"])
    q3_lwc = _q(q3["wt_kl"]["A16_BlockH4_quad_lpt_wcol"])
    q3_snwc = _q(q3["wt_kl"]["A16_BlockH4_quad_snake_wcol"])
    q3_ptl = _q(q3["wt_kl"]["ProjType_Uniform_QuadLPT_wcol"])
    q3_lrand_py = _q(q3["py_kl"]["hA16_BlockH4_quad_lpt_rand"])
    q3_hr = _q(q3["wt_kl"]["hA16_real"])
    q3_h2 = _q(q3["wt_kl"]["hA16_BlockH2_real"])
    q3_h2l = _q(q3["wt_kl"]["hA16_BlockH2_quad_lpt_real"])
    q3_h4 = _q(q3["wt_kl"]["hA16_BlockH4_real"])
    q3_gs = _q(q3["wt_kl"]["hA16_BlockH4_sort_real"])
    q3_lpt = _q(q3["wt_kl"]["hA16_BlockH4_quad_lpt_real"])
    q3_sqrt = _q(q3["wt_kl"]["hA16_BlockH4_quad_lpt_sqrt_real"])
    q3_snk = _q(q3["wt_kl"]["hA16_BlockH4_quad_snake_real"])
    q3_snk_py = _q(q3["py_kl"]["hA16_BlockH4_quad_snake_real"])
    q3_sqrt_py = _q(q3["py_kl"]["hA16_BlockH4_quad_lpt_sqrt_real"])
    q3_hr_py = _q(q3["py_kl"]["hA16_real"])
    q3_h4_py = _q(q3["py_kl"]["hA16_BlockH4_real"])
    q3_h2_py = _q(q3["py_kl"]["hA16_BlockH2_real"])
    q3_h2l_py = _q(q3["py_kl"]["hA16_BlockH2_quad_lpt_real"])

    q3_md_wc = _q(q3["matched_mse_db_diff"]["A16_BlockH4_quad_lpt_wcol_matched"])
    q3_md_hr = _q(q3["matched_mse_db_diff"]["A16_BlockH4_quad_lpt_hrand_matched"])

    gates = {
        "p_s72_1_two_level_blockh4_shatters_undefeated_blockh2_on_all_3_qwen_models": (
            q05_lpt < q05_sh
            and q05_sh < q05_zcd
            and q05_zcd < q05_h2
            and q05_h2 < q05_h4
            and q15_sh < q15_lpt
            and q15_lpt < q15_snk
            and q15_snk < q15_h2
            and q15_h2 < q15_h4
            and q3_snk < q3_h2
            and q3_h2 < q3_hr
            and q3_hr < q3_h4
            and q3_snk_py < q3_hr_py
            and q3_hr_py < q3_h4_py
            and q3_h4_py < q3_h2_py
        ),
        "p_s72_1_quad_snake_beats_global_sort_all_4_and_sqrt_lpt_breaks_0_020_3b_python": (
            sm_snk < sm_gs
            and q05_snk < q05_gs
            and q15_snk < q15_gs
            and q3_snk < q3_gs
            and q3_sqrt < q3_gs
            and q3_sqrt_py < Fraction(1, 50)
            and q3_sqrt_py < q3_snk_py
        ),
        "p_s72_1_honest_falsification_linear_lpt_starvation_on_heavy_tailed_3b_and_360m_wins": (
            sm_lpt < sm_gs
            and sm_sh < sm_gs
            and sm_zcd_ppl < sm_h4_ppl
            and sm_h4 < sm_snk
            and q3_gs < q3_lpt
            and q05_sh_py < q05_h4_py
        ),
        "p_s72_2_pareto_corner_quad_spread_and_ball_cv_all_4_models": (
            _q(sm["tel_hreal"]["lpt_spr"]) < _q(sm["tel_hreal"]["grp_spr"])
            and _q(sm["tel_hreal"]["grp_spr"]) < _q(sm["tel_hreal"]["unp_spr"])
            and _q(sm["tel_hreal"]["lpt_cv"]) < _q(sm["tel_hreal"]["glob_cv"])
            and _q(q05["tel_hreal"]["lpt_spr"]) < _q(q05["tel_hreal"]["grp_spr"])
            and _q(q05["tel_hreal"]["grp_spr"]) < _q(q05["tel_hreal"]["unp_spr"])
            and _q(q05["tel_hreal"]["lpt_cv"]) < _q(q05["tel_hreal"]["glob_cv"])
            and _q(q15["tel_hreal"]["lpt_spr"]) < _q(q15["tel_hreal"]["grp_spr"])
            and _q(q15["tel_hreal"]["grp_spr"]) < _q(q15["tel_hreal"]["unp_spr"])
            and _q(q15["tel_hreal"]["lpt_cv"]) < _q(q15["tel_hreal"]["glob_cv"])
            and _q(q3["tel_hreal"]["lpt_spr"]) < _q(q3["tel_hreal"]["grp_spr"])
            and _q(q3["tel_hreal"]["grp_spr"]) < _q(q3["tel_hreal"]["unp_spr"])
            and _q(q3["tel_hreal"]["lpt_cv"]) < _q(q3["tel_hreal"]["glob_cv"])
            and _q(q3["tel_hreal"]["lpt_cv_sqrt"]) < _q(q3["tel_hreal"]["unp_cv_sqrt"])
            and _q(q3["tel_wcol"]["lpt_cv"]) < _q(q3["tel_wcol"]["unp_cv"])
        ),
        "p_s72_2_zero_text_3b_wcol_cure_matched_mse_and_rand_beats_real_blockh4_py": (
            q3_ax < q3_ptl
            and q3_ptl < q3_bh4
            and q3_bh4 < q3_snwc
            and q3_snwc < q3_lwc
            and q3_lwc < q3_swc
            and q3_md_wc <= Fraction(2, 1000000)
            and q3_md_hr <= Fraction(2, 1000000)
            and q3_lrand_py < q3_h4_py
            and _q(q15["wt_kl"]["A16_BlockH4_sort_wcol"]) < _q(q15["wt_kl"]["A16_BlockH4_quad_lpt_wcol"])
        ),
        "p_s72_3_blockh4_overtakes_blockh2_under_matched_two_level_stratification": (
            sm_h2l < sm_gs
            and q05_h2l < q05_gs
            and q15_h2l < q15_gs
            and q3_gs < q3_h2l
            and q05_lpt < q05_h2l
            and q15_lpt < q15_h2l
            and q15_lpt_py < q15_h2l_py
            and q3_snk < q3_h2l
            and q3_snk_py < q3_h2l_py
        ),
    }

    return {
        "oracle_class": "sampled",
        "measured_live_model_weights": True,
        "pinned_digests": dict(PINNED_DIGESTS),
        "prior_study_anchors": dict(PRIOR_STUDY_ANCHORS),
        "qwen05b_quad_lpt_over_h2_wt_kl": q_pair(q05_lpt / q05_h2),
        "qwen05b_quad_lpt_over_unp_blockh4_wt_kl": q_pair(q05_lpt / q05_h4),
        "qwen15b_quad_lpt_sharednorm_over_h2_wt_kl": q_pair(q15_sh / q15_h2),
        "qwen15b_quad_lpt_over_h2_quad_lpt_wt_kl": q_pair(q15_lpt / q15_h2l),
        "qwen3b_quad_snake_over_h2_wt_kl": q_pair(q3_snk / q3_h2),
        "qwen3b_quad_snake_over_unp_blockh4_wt_kl": q_pair(q3_snk / q3_h4),
        "qwen3b_quad_snake_over_h2_py_kl": q_pair(q3_snk_py / q3_h2_py),
        "qwen3b_quad_lpt_sqrt_over_h2_py_kl": q_pair(q3_sqrt_py / q3_h2_py),
        "qwen3b_quad_snake_wcol_over_sort_wcol_wt_kl": q_pair(q3_snwc / q3_swc),
        "qwen3b_quad_spread_reduction_ratio": q_pair(_q(q3["tel_hreal"]["unp_spr"]) / _q(q3["tel_hreal"]["lpt_spr"])),
        "qwen3b_ball_cv_reduction_vs_global_sort": q_pair(_q(q3["tel_hreal"]["glob_cv"]) / _q(q3["tel_hreal"]["lpt_cv"])),
        "preregistered_pillar_outcomes": {
            "P_S72_1_calibrated_two_level_saddle_point_dominance": "LANDMARK_SHATTERS_BLOCKH2_3OF3_QWEN_QUAD_SNAKE_3B_0_023154_QUAD_LPT_05B_15B_LINEAR_LPT_3B_STARVATION_DISCOVERED",
            "P_S72_2_zero_text_quad_lpt_and_ball_variance_pareto": "CONFIRMED_PARETO_CORNER_4OF4_3B_WCOL_CURED_MATCHED_MSE_PASS_UNCAL_VS_CAL_REGIME_DICHOTOMY",
            "P_S72_3_blockh4_vs_blockh2_matched_quad_lpt_synergy": "CONFIRMED_BLOCKH4_OVERTAKES_BLOCKH2_ON_05B_15B_AND_3B_VIA_QUAD_SNAKE",
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
        and p1["total_arms_evaluated"] == 87
        and p1["qwen_quads_per_g128_ball"] == 32
        and p1["smollm2_quads_per_g32_ball"] == 8
        and p1["block_h4_lossless_int8_dp4a"] is True
        and p1["down_proj_offline_fold_runtime_gather_ops"] == 0
    )
    sampled_ok = all(p2["sampled_gates"].values())
    body = {
        "schema_version": SCHEMA_VERSION,
        "oracle_module": "rtx3080_s72_oracle",
        "epistemic_taxonomy": {
            "exact_pillar": "exact",
            "live_4model_ceer_cycle5_pillar": "sampled",
            "prior_study_links": "anchor",
        },
        "pillar_1_exact_two_level_stratification_arithmetic": p1,
        "pillar_2_sampled_live_ceer_cycle5": p2,
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
    """Verify frozen live receipts against pinned digests, inner hashes, timestamps, 87 arms, and telemetry."""
    prereg_body = {k: v for k, v in prereg_obj.items() if k != "preregistration_sha256"}
    prereg_inner_sha = hashlib.sha256(canonical_json_bytes(prereg_body)).hexdigest()
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
            if not _is_int_eq(arm["syndrome_errors"], 0) or not _is_int_eq(arm["non_block_tensors_differ"], 0):
                arms_clean = False
    arms_clean = arms_clean and total_arms == 87

    sm_m = models["HuggingFaceTB/SmolLM2-360M"]
    q05_m = models["Qwen/Qwen2.5-0.5B"]
    q15_m = models["Qwen/Qwen2.5-1.5B"]
    q3_m = models["Qwen/Qwen2.5-3B"]
    sm_a, q05_a, q15_a, q3_a = sm_m["arms"], q05_m["arms"], q15_m["arms"], q3_m["arms"]

    p1_ok = (
        _to_q(sm_a["hA16_BlockH4_quad_snake_real"]["wikitext2"]["mean_kl"]) < _to_q(sm_a["hA16_BlockH4_sort_real"]["wikitext2"]["mean_kl"])
        and _to_q(q05_a["hA16_BlockH4_quad_lpt_real"]["wikitext2"]["mean_kl"]) < _to_q(q05_a["hA16_BlockH2_real"]["wikitext2"]["mean_kl"])
        and _to_q(q05_a["hA16_BlockH4_quad_lpt_sharednorm"]["wikitext2"]["mean_kl"]) < _to_q(q05_a["hA16_BlockH2_real"]["wikitext2"]["mean_kl"])
        and _to_q(q05_a["hA16_BlockH4_quad_lpt_down_zerocost"]["wikitext2"]["mean_kl"]) < _to_q(q05_a["hA16_BlockH2_real"]["wikitext2"]["mean_kl"])
        and _to_q(q15_a["hA16_BlockH4_quad_lpt_sharednorm"]["wikitext2"]["mean_kl"]) < _to_q(q15_a["hA16_BlockH2_real"]["wikitext2"]["mean_kl"])
        and _to_q(q15_a["hA16_BlockH4_quad_lpt_real"]["wikitext2"]["mean_kl"]) < _to_q(q15_a["hA16_BlockH2_real"]["wikitext2"]["mean_kl"])
        and _to_q(q3_a["hA16_BlockH4_quad_snake_real"]["wikitext2"]["mean_kl"]) < _to_q(q3_a["hA16_BlockH2_real"]["wikitext2"]["mean_kl"])
        and _to_q(q3_a["hA16_BlockH4_quad_snake_real"]["python"]["mean_kl"]) < _to_q(q3_a["hA16_real"]["python"]["mean_kl"])
        and _to_q(q3_a["hA16_BlockH4_quad_lpt_sqrt_real"]["python"]["mean_kl"]) < Fraction(1, 50)
        and _to_q(q3_a["hA16_BlockH4_sort_real"]["wikitext2"]["mean_kl"]) < _to_q(q3_a["hA16_BlockH4_quad_lpt_real"]["wikitext2"]["mean_kl"])
    )

    p2_ok = (
        _to_q(sm_m["ball_variance_telemetry_hreal"]["quad_lpt"]["mean_quad_spread"]) < _to_q(sm_m["ball_variance_telemetry_hreal"]["grp128_sort"]["mean_quad_spread"])
        and _to_q(sm_m["ball_variance_telemetry_hreal"]["quad_lpt"]["mean_ball_cv_lin"]) < _to_q(sm_m["ball_variance_telemetry_hreal"]["global_sort"]["mean_ball_cv_lin"])
        and _to_q(q05_m["ball_variance_telemetry_hreal"]["quad_lpt"]["mean_quad_spread"]) < _to_q(q05_m["ball_variance_telemetry_hreal"]["grp128_sort"]["mean_quad_spread"])
        and _to_q(q05_m["ball_variance_telemetry_hreal"]["quad_lpt"]["mean_ball_cv_lin"]) < _to_q(q05_m["ball_variance_telemetry_hreal"]["global_sort"]["mean_ball_cv_lin"])
        and _to_q(q15_m["ball_variance_telemetry_hreal"]["quad_lpt"]["mean_quad_spread"]) < _to_q(q15_m["ball_variance_telemetry_hreal"]["grp128_sort"]["mean_quad_spread"])
        and _to_q(q15_m["ball_variance_telemetry_hreal"]["quad_lpt"]["mean_ball_cv_lin"]) < _to_q(q15_m["ball_variance_telemetry_hreal"]["global_sort"]["mean_ball_cv_lin"])
        and _to_q(q3_m["ball_variance_telemetry_hreal"]["quad_lpt"]["mean_quad_spread"]) < _to_q(q3_m["ball_variance_telemetry_hreal"]["grp128_sort"]["mean_quad_spread"])
        and _to_q(q3_m["ball_variance_telemetry_hreal"]["quad_lpt"]["mean_ball_cv_lin"]) < _to_q(q3_m["ball_variance_telemetry_hreal"]["global_sort"]["mean_ball_cv_lin"])
        and _to_q(q3_a["A16_BlockH4_quad_lpt_wcol"]["wikitext2"]["mean_kl"]) < _to_q(q3_a["A16_BlockH4_sort_wcol"]["wikitext2"]["mean_kl"])
        and _to_q(q3_a["A16_BlockH4_quad_lpt_wcol_matched"]["max_matched_mse_db_diff"]) <= Fraction(2, 1000000)
        and _to_q(q3_a["A16_BlockH4_quad_lpt_hrand_matched"]["max_matched_mse_db_diff"]) <= Fraction(2, 1000000)
        and _to_q(q3_a["hA16_BlockH4_quad_lpt_rand"]["python"]["mean_kl"]) < _to_q(q3_a["hA16_BlockH4_real"]["python"]["mean_kl"])
    )

    p3_ok = (
        _to_q(q05_a["hA16_BlockH4_quad_lpt_real"]["wikitext2"]["mean_kl"]) < _to_q(q05_a["hA16_BlockH2_quad_lpt_real"]["wikitext2"]["mean_kl"])
        and _to_q(q15_a["hA16_BlockH4_quad_lpt_real"]["wikitext2"]["mean_kl"]) < _to_q(q15_a["hA16_BlockH2_quad_lpt_real"]["wikitext2"]["mean_kl"])
        and _to_q(q15_a["hA16_BlockH4_quad_lpt_real"]["python"]["mean_kl"]) < _to_q(q15_a["hA16_BlockH2_quad_lpt_real"]["python"]["mean_kl"])
        and _to_q(q3_a["hA16_BlockH4_quad_snake_real"]["wikitext2"]["mean_kl"]) < _to_q(q3_a["hA16_BlockH2_quad_lpt_real"]["wikitext2"]["mean_kl"])
        and _to_q(q3_a["hA16_BlockH4_quad_snake_real"]["python"]["mean_kl"]) < _to_q(q3_a["hA16_BlockH2_quad_lpt_real"]["python"]["mean_kl"])
    )

    checks = {
        "digests_match": digests_match,
        "prereg_strictly_precedes_execution": prereg_precedes,
        "mlos_clean_zone_b": mlos_clean,
        "mlos_zero_desktop_processes": mlos_zero_desktop,
        "real_weight_honesty_tags": honesty_ok,
        "all_4_model_pins_match": pins_ok,
        "corpora_pins_match": corpora_ok,
        "all_87_real_arms_clean": arms_clean,
        "p_s72_1_two_level_blockh4_dominance_and_3b_quad_snake_record": p1_ok,
        "p_s72_2_pareto_telemetry_and_zero_text_wcol_cure": p2_ok,
        "p_s72_3_blockh4_overtakes_blockh2_under_two_level_stratification": p3_ok,
    }
    return {"all_verified": all(checks.values()), "checks": checks}
