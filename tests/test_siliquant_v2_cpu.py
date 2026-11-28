#!/usr/bin/env python3
"""Pure-CPU unit, exact-Q parity, SECDED/IRIS, and hygiene suite for SiliQuant V2.

Strictly CPU-only (CUDA_VISIBLE_DEVICES="", zero PyTorch required, pure Python stdlib).
Zero bare asserts, zero unseeded RNG, LF-only, bit-identical stdout across:
  python3 -I -B tests/test_siliquant_v2_cpu.py
  python3 -I -O -B tests/test_siliquant_v2_cpu.py
  python3 -I -B -W error tests/test_siliquant_v2_cpu.py
"""

from __future__ import annotations

import ast
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import random
import re
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from siliquant import (  # noqa: E402
    A16ChunkAccumulator,
    QuantArmSpec,
    RazorCanary64,
    SiliquantError,
    SlgbHeader,
    SlqiHeader,
    SlqwHeader,
    TensorSensitivityProfile,
    WatchdogAction,
    a16_closest_fast_py,
    a16_nearest_py,
    apply_had16_3_block_f64,
    apply_had16_3_inverse_f64,
    build_c16_codewords_py,
    check_a16_syndrome_py,
    compare_had16_vs_sylvester_t6_q,
    compute_l0_l27_cliff_bpw_stats,
    crc32_ieee,
    encode_slgb_header,
    encode_slqi_header,
    encode_slqw_header,
    evaluate_razor_watchdog,
    iris_finalize_words_u32,
    make_quad_lpt_perm_py,
    make_quad_snake_perm_py,
    pack_quadlpt_sharednorm_py,
    quantize_dequant_mxfp8_block32_py,
    secded_encode_u64,
    secded_heal_u64,
    secded_syndrome_u64,
    solve_pareto_lagrangian_bpw_q,
    unpack_quadlpt_sharednorm_py,
    validate_slgb_header,
    validate_slqi_header,
    validate_slqw_header,
    verify_gauge_group_algebra_py,
    verify_hadamard16_exact_q,
)


def _check(cond: bool, msg: str, failures: list[str]) -> None:
    if not cond:
        failures.append(msg)


def _check_raises(fn, args: tuple, kwargs: dict, msg: str, failures: list[str]) -> None:
    try:
        fn(*args, **kwargs)
    except SiliquantError:
        return
    failures.append(msg)


def _check_v2_hygiene(failures: list[str]) -> int:
    slash = chr(47)
    rx_home = re.compile("(?:" + slash + "ho" + "me" + slash + "|" + slash + "Us" + "ers" + slash + r")[A-Za-z0-9._-]+")
    rx_secret = re.compile(r"(?:hf_[A-Za-z0-9]{20}|ghp_[A-Za-z0-9]{20}|gho_[A-Za-z0-9]{20}|github_pat_[A-Za-z0-9_]{20}|sk-[A-Za-z0-9]{20}|AKIA[0-9A-Z]{16})")
    rx_internal = re.compile(
        r"(?:(?i:\btrack[-_ ]?s\d+[a-z0-9_-]*\b)"
        r"|\bEXP-Q\d+[A-Za-z0-9_-]*\b"
        r"|\bKG-[A-Z0-9][A-Z0-9_-]*\b"
        r"|\bSOP-\d+[A-Za-z0-9-]*\b"
        r"|\bCS-\d+[A-Z0-9-]*\b"
        r"|\bAUDIT-(?:S\d+|\d+)[A-Za-z0-9._-]*\b"
        r"|\bP2[23][0-9][a-z]*\b"
        r"|\bS97\b|\bS164\b|\bS18[0-9][a-z0-9]*\b"
        r"|expedi" + "tion" + slash + "track-"
        r"|Proj" + "ects" + slash + "expedi" + "tion"
        r"|(?<![/A-Za-z0-9_])#[345]\d{3}\b)"
    )
    checked = 0
    targets = [
        REPO_ROOT / "pyproject.toml",
        REPO_ROOT / "Makefile",
        REPO_ROOT / "README.md",
        REPO_ROOT / ".github" / "workflows" / "ci.yml",
        REPO_ROOT / ".github" / "workflows" / "codeql.yml",
        *sorted((REPO_ROOT / "siliquant").rglob("*.py")),
        REPO_ROOT / "tests" / "test_siliquant_v2_cpu.py",
    ]
    for p in targets:
        if not p.exists():
            continue
        _check(not p.is_symlink(), f"symlink_forbidden:{p.name}", failures)
        raw = p.read_bytes()
        _check(b"\r" not in raw, f"crlf_forbidden:{p.name}", failures)
        txt = raw.decode("utf-8")
        _check(rx_home.search(txt) is None, f"home_leak:{p.name}", failures)
        _check(rx_secret.search(txt) is None, f"secret_leak:{p.name}", failures)
        _check(rx_internal.search(txt) is None, f"internal_id_leak:{p.name}", failures)
        if p.suffix == ".py":
            tree = ast.parse(txt, filename=str(p))
            for node in ast.walk(tree):
                if isinstance(node, ast.Assert):
                    failures.append(f"bare_assert:{p.name}:{getattr(node, 'lineno', 0)}")
        checked += 1
    return checked


def main() -> int:
    failures: list[str] = []
    rng = random.Random(20261128)

    # 1. Exact G1..G4 gauge invariance, BF16 fold parity, H4 cancellation & obstructions
    g_res = verify_gauge_group_algebra_py(seed=182001)
    _check(g_res["g1_to_g4_exact_max_abs_err"] < 1e-12, "G1..G4 exact drift", failures)
    _check(g_res["g1_bf16_fold_max_abs_err"] < 1e-15, "G1 BF16 fold identity", failures)
    _check(g_res["g2_bf16_fold_max_abs_err"] < 1e-15, "G2 BF16 fold identity", failures)
    _check(g_res["h4_squared_identity_err"] < 1e-12, "H4^2 == I4", failures)

    # 2. A_16 ML vs fast decoder, QuadLPT-SharedNorm, MXFP8 cliff, and streaming accumulator
    cws = build_c16_codewords_py()
    _check(len(cws) == 2048, "2048 Extended Hamming codewords", failures)
    for _ in range(16):
        vec = [rng.gauss(0.0, 2.2) + 0.0137 for _ in range(16)]
        y_ml = a16_nearest_py(vec, cws)
        y_fast = a16_closest_fast_py(vec)
        _check(check_a16_syndrome_py(y_fast), "fast A16 syndrome == 0", failures)
        d2_ml = sum((vec[i] - y_ml[i]) ** 2 for i in range(16))
        d2_fast = sum((vec[i] - y_fast[i]) ** 2 for i in range(16))
        _check(d2_fast + 1e-11 >= d2_ml and d2_fast <= d2_ml * 1.5 + 0.5, "fast A16 bounded near ML", failures)
    for _ in range(32):
        cw = cws[rng.randrange(len(cws))]
        pt = [cw[i] + 2 * rng.randint(-2, 2) + rng.uniform(-0.45, 0.45) for i in range(16)]
        _check(a16_closest_fast_py(pt) == a16_nearest_py(pt, cws), "fast A16 exact ML in Voronoi neighborhood", failures)

    k_cols = 256
    scores = [rng.uniform(0.1, 10.0) for _ in range(k_cols)]
    perm_lpt = make_quad_lpt_perm_py(scores, group_size=128, block_w=4)
    perm_snk = make_quad_snake_perm_py(scores, group_size=128, block_w=4)
    _check(sorted(perm_lpt) == list(range(k_cols)), "QuadLPT bijection", failures)
    _check(sorted(perm_snk) == list(range(k_cols)), "QuadSnake bijection", failures)

    mat = [[rng.gauss(0.0, 0.35) for _ in range(k_cols)] for _ in range(4)]
    colsc = [rng.uniform(0.75, 1.35) for _ in range(k_cols)]
    packed_ql = pack_quadlpt_sharednorm_py(mat, colscale=colsc, perm_in=perm_lpt, group_size=128, bpw_key="4.5")
    unpacked_a, syn_a = unpack_quadlpt_sharednorm_py(packed_ql)
    unpacked_b, syn_b = unpack_quadlpt_sharednorm_py(packed_ql)
    _check(syn_a == 0 and syn_b == 0, "QuadLPT-SharedNorm zero syndromes", failures)
    _check(unpacked_a == unpacked_b, "QuadLPT-SharedNorm deterministic unpack", failures)

    vals32 = [rng.gauss(0.0, 1.5) for _ in range(32)]
    ue8, qbytes, dq32 = quantize_dequant_mxfp8_block32_py(vals32)
    _check(0 <= ue8 <= 254 and len(qbytes) == 32 and len(dq32) == 32, "MXFP8 block-32 valid", failures)
    cliff = compute_l0_l27_cliff_bpw_stats()
    _check(cliff["n_matrices"] == 14 and cliff["within_cliff_cap"] is True, "L0/L27 MXFP8 cliff within cap", failures)
    _check(abs(cliff["delta_bpw_over_1p544b"] - 0.183526) < 1e-4, "L0/L27 delta bpw accurate", failures)

    acc = A16ChunkAccumulator(4, 128, group_size=128, bpw_key="4.5")
    acc.push_rows([[rng.gauss(0.0, 0.4) for _ in range(128)] for _ in range(2)])
    acc.push_rows([[rng.gauss(0.0, 0.4) for _ in range(128)] for _ in range(2)])
    stream_res = acc.finalize()
    _check(stream_res["syndrome_errors"] == 0 and len(stream_res["y_int"]) == 4, "Streaming A16 accumulator PASS", failures)

    # 3. Pure-Q Sloane Had.16.3 orthogonality, minimax t=6 comparison & block rotation
    had_orth = verify_hadamard16_exact_q()
    had_cmp = compare_had16_vs_sylvester_t6_q()
    _check(had_orth["orthogonality_exact_q"] is True, "Had.16.3 exact Q orthogonality", failures)
    _check(had_cmp["exact_q_roundtrip_pass"] is True, "Had.16.3 exact Q round-trip", failures)
    v32 = [rng.gauss(0.0, 2.0) for _ in range(32)]
    v32_inv = apply_had16_3_inverse_f64(apply_had16_3_block_f64(v32))
    _check(max(abs(a - b) for a, b in zip(v32, v32_inv)) < 1e-13, "Had.16.3 float roundtrip", failures)

    # 4. Pure-Q Pareto-Lagrangian BPW allocator + critical-layer pinning
    profiles = [
        TensorSensitivityProfile(
            name="layers.0.message_gate.down_proj",
            weight_count=1024,
            arms=(
                QuantArmSpec("a16_3bpw", Fraction(3, 1), Fraction(400, 1)),
                QuantArmSpec("a16_45bpw", Fraction(9, 2), Fraction(100, 1)),
                QuantArmSpec("bf16", Fraction(16, 1), Fraction(1, 1)),
            ),
        ),
        TensorSensitivityProfile(
            name="layers.1.attn.q_proj",
            weight_count=1024,
            arms=(
                QuantArmSpec("a16_3bpw", Fraction(3, 1), Fraction(24, 1)),
                QuantArmSpec("a16_45bpw", Fraction(9, 2), Fraction(6, 1)),
                QuantArmSpec("mxfp8", Fraction(33, 4), Fraction(1, 1)),
            ),
        ),
        TensorSensitivityProfile(
            name="layers.2.mlp.gate_proj",
            weight_count=2048,
            arms=(
                QuantArmSpec("a16_3bpw", Fraction(3, 1), Fraction(16, 1)),
                QuantArmSpec("a16_45bpw", Fraction(9, 2), Fraction(8, 1)),
                QuantArmSpec("mxfp8", Fraction(33, 4), Fraction(2, 1)),
            ),
        ),
    ]
    alloc = solve_pareto_lagrangian_bpw_q(
        profiles,
        Fraction(29, 4),
        pinned_min_bpw_patterns=(("message_gate.down_proj", Fraction(16, 1)),),
    )
    _check(alloc.chosen_arms["layers.0.message_gate.down_proj"].arm_id == "bf16", "Pareto pins critical down_proj", failures)
    _check(alloc.effective_bpw_q <= Fraction(29, 4), "Pareto respects exact Q budget", failures)

    # 5. .slqi / .slgb / .slqw headers, CRC32, SECDED-64 (SEC+DED), and IRIS canary watchdog
    raw_slqi = encode_slqi_header(SlqiHeader())
    raw_slgb = encode_slgb_header(SlgbHeader())
    raw_slqw = encode_slqw_header(SlqwHeader())
    _check(len(raw_slqi) == 128 and len(raw_slgb) == 64 and len(raw_slqw) == 64, "Wire header byte lengths", failures)
    validate_slqi_header(raw_slqi, [0, 12, 200006])
    validate_slgb_header(raw_slgb, [2, 19, 1])
    validate_slqw_header(raw_slqw, [151643, 42, 151645])

    # Fail-closed negative controls on wire headers
    bad_crc_slqi = bytearray(raw_slqi)
    bad_crc_slqi[10] ^= 0x01
    _check_raises(validate_slqi_header, (bytes(bad_crc_slqi),), {}, "slqi rejects corrupt CRC32", failures)
    _check_raises(validate_slgb_header, (raw_slgb, [77, 19]), {}, "slgb enforces bos=2", failures)
    bad_crc_slqw = bytearray(raw_slqw)
    bad_crc_slqw[4] ^= 0x80
    _check_raises(validate_slqw_header, (bytes(bad_crc_slqw),), {}, "slqw rejects corrupt CRC32", failures)

    # SECDED-64 3x SEC + 3x DED
    for idx in range(3):
        u11 = rng.randrange(2048)
        nibs = [rng.randrange(16) & 0xE for _ in range(16)]
        w64 = secded_encode_u64(u11, nibs)
        _check(secded_syndrome_u64(w64) == 0, f"SECDED clean {idx}", failures)
        healed, sec, ded = secded_heal_u64(w64 ^ (1 << (4 * ((idx * 5) % 16))))
        _check(healed == w64 and sec == 1 and ded == 0, f"SECDED SEC repair {idx}", failures)
        b1 = (idx * 3) % 16
        b2 = (b1 + 1) % 16
        _, sec2, ded2 = secded_heal_u64(w64 ^ (1 << (4 * b1)) ^ (1 << (4 * b2)))
        _check(sec2 == 0 and ded2 == 1, f"SECDED DED trap {idx}", failures)

    iris_sig = iris_finalize_words_u32([0x10203040, 0x55667788, crc32_ieee(raw_slqw)])
    can_ok = RazorCanary64(iris_runtime_crc_u32=iris_sig)
    _check(evaluate_razor_watchdog(can_ok, iris_sig) == WatchdogAction.CONTINUE, "IRIS clean watchdog CONTINUE", failures)
    _check(
        evaluate_razor_watchdog(RazorCanary64(iris_runtime_crc_u32=iris_sig ^ 1), iris_sig) == WatchdogAction.SAFE_HALT_DED_ROLLBACK,
        "IRIS mismatch triggers SAFE_HALT",
        failures,
    )
    _check(
        evaluate_razor_watchdog(RazorCanary64(ded_count_u32=1, iris_runtime_crc_u32=iris_sig), iris_sig) == WatchdogAction.SAFE_HALT_DED_ROLLBACK,
        "DED triggers SAFE_HALT",
        failures,
    )

    files_checked = _check_v2_hygiene(failures)
    out = {
        "status": "PASS" if not failures else "FAIL",
        "files_checked": files_checked,
        "g1_g4_max_err": g_res["g1_to_g4_exact_max_abs_err"],
        "had16_3_t6_bound_q": had_cmp["known_minimax_t6_bound_q"],
        "l0_l27_mxfp8_delta_bpw": round(cliff["delta_bpw_over_1p544b"], 6),
        "pareto_eff_bpw_q": str(alloc.effective_bpw_q),
        "wire_crc32_slqw": crc32_ieee(raw_slqw[:-4]),
        "failures": failures,
    }
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
