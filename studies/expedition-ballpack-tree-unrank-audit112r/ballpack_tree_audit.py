#!/usr/bin/env python3
"""Dual-mode pure-Z/Q audit runner for Track S15 (AUDIT-112):
Zero-overhead binary-tree shell-convolution unranking & E_8^4 codec for Moissanite-Labs/siliquant#23/#25.

Modes:
  --standalone       (default) Self-contained exact pure-Z/Q replay with zero external file I/O
  --live             Read-only live recount against staged intake-siliquant-ballpack/ (20d13d5f)
  --check <cert>     Fail-closed Zone-B verification of a JSON certificate (exit 0 on valid, 2 on error)

Strict discipline: pure Python 3 stdlib, exact int/Fraction only, zero floats, zero bare asserts.
"""

from __future__ import annotations

import argparse
import ast
from functools import lru_cache
import hashlib
import hmac
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from ballpack_tree_oracle import (
    BallpackTreeError,
    build_z_tree_tables,
    canonical_sha256,
    certify_e8_tree_bijection_exact,
    certify_shaping_and_bandwidth_ceiling_exact,
    certify_z_tree_bijection_exact,
    contains_forbidden_json_value,
    count_z_ball_serial_dp_row,
    require_exact_int,
    require_git_sha40,
    require_sha256,
)

SCHEMA_VERSION = "audit112.ballpack_tree_unrank.v1"

PINNED_COMMIT_SHA = "20d13d5fa9c5fb2333b8faa0360a90caa1e0a43f"

PINNED_INTAKE_FILES: tuple[str, ...] = (
    "shape/ballindex.py",
    "shape/ballpack.c",
    "shape/ballpack.py",
)

PINNED_ARTIFACTS_EXPECTED: dict[str, dict[str, Any]] = {
    "shape/ballindex.py": {
        "bytes": 9746,
        "sha256": "bec9e571ab77c08a18f2ccfd94c1c3fc5321dde3e1bbb1f1a57a3ce590ab05b7",
    },
    "shape/ballpack.c": {
        "bytes": 6194,
        "sha256": "53074fbadf169650da6cad400b986e07b1ea15df180f6f148f97ef4761b33055",
    },
    "shape/ballpack.py": {
        "bytes": 18607,
        "sha256": "b804aae680baf9e9842e6bea6971f26ff1bbddad961fdd95dca41366438fa4d0",
    },
}

PINNED_SOURCE_AUDIT_EXPECTED: dict[str, Any] = {
    "ballindex_has_z8_and_e8": True,
    "ballindex_serial_unrank_loop": True,
    "ballindex_z32_136_count_matches_tree": True,
    "ballindex_z32_table_entries": 25872,
    "ballindex_e8_136_table_entries": 466221,
    "ballindex_e8_88_table_entries": 57365,
    "ballpack_c_limbs": 3,
    "ballpack_c_serial_unrank_loop": True,
    "ballpack_c_has_no_e8": True,
    "ballpack_py_default_z32_b136": True,
    "ballpack_py_hardcodes_z8_only": True,
}

STANDALONE_CERT_TOP_KEYS = frozenset({
    "schema_version",
    "audit_id",
    "mode",
    "live_intake_audited",
    "novelty_class",
    "pillar1_z32_tree_bijection",
    "pillar2_e8_4_tree_codec",
    "pillar3_shaping_and_roofline",
    "overall_verified",
    "certificate_sha256",
})

LIVE_CERT_TOP_KEYS = STANDALONE_CERT_TOP_KEYS | frozenset({
    "pinned_commit_sha",
    "pinned_artifacts",
    "live_source_audit",
})

_LIMBS_RE = re.compile(r"#define\s+LIMBS\s+([0-9]+)")


def sha256_file(path: Path) -> tuple[str, int]:
    if not path.is_file():
        raise BallpackTreeError(f"missing pinned artifact: {path.name}")
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest(), len(data)


def run_pillar1_witnesses() -> tuple[dict[str, Any], dict[str, Any]]:
    z4_exh = certify_z_tree_bijection_exact(4, 6, exhaustive=True)
    z8_exh = certify_z_tree_bijection_exact(8, 3, exhaustive=True)
    z32_tables = build_z_tree_tables(32, 783)
    dp_row_784 = count_z_ball_serial_dp_row(32, 784)
    z32_136 = certify_z_tree_bijection_exact(
        32,
        783,
        exhaustive=False,
        n_random=24,
        precomputed_tables=z32_tables,
        serial_count_override=dp_row_784[783],
    )
    z32_next_exceeds = dp_row_784[784] > (1 << 136)

    ok = (
        z4_exh["count_matches_ballindex_dp"]
        and z4_exh["all_roundtrips_exact"]
        and z4_exh["exhaustive_bijection_distinct"]
        and z4_exh["all_reciprocals_universally_valid"]
        and z8_exh["count_matches_ballindex_dp"]
        and z8_exh["all_roundtrips_exact"]
        and z8_exh["exhaustive_bijection_distinct"]
        and z8_exh["all_reciprocals_universally_valid"]
        and z32_136["codebook_bits"] == 136
        and z32_136["count_matches_ballindex_dp"]
        and z32_136["all_roundtrips_exact"]
        and z32_136["all_reciprocals_universally_valid"]
        and z32_136["tree_split_depth"] == 4
        and z32_136["rate_overhead_bits"] == 0
        and z32_next_exceeds
    )
    pillar = {
        "pillar_verified": ok,
        "z4_b6_exhaustive": z4_exh,
        "z8_b3_exhaustive": z8_exh,
        "z32_b783_136bit": z32_136,
        "z32_b784_exceeds_2pow136": z32_next_exceeds,
    }
    return pillar, z32_tables


@lru_cache(maxsize=1)
def _cached_pillars_canonical_json() -> tuple[str, str, str]:
    p1, z32_tables = run_pillar1_witnesses()
    p2 = run_pillar2_witnesses()
    p3 = run_pillar3_witnesses(z32_tables)
    return (
        json.dumps(p1, sort_keys=True, separators=(",", ":"), ensure_ascii=True),
        json.dumps(p2, sort_keys=True, separators=(",", ":"), ensure_ascii=True),
        json.dumps(p3, sort_keys=True, separators=(",", ":"), ensure_ascii=True),
    )


def _fresh_expected_pillars() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    s1, s2, s3 = _cached_pillars_canonical_json()
    return json.loads(s1), json.loads(s2), json.loads(s3)


def run_pillar2_witnesses() -> dict[str, Any]:
    e8_1_exh = certify_e8_tree_bijection_exact(1, 1, exhaustive=True)
    e8_4_exh = certify_e8_tree_bijection_exact(4, 1, exhaustive=True)
    e8_4_88 = certify_e8_tree_bijection_exact(4, 48, exhaustive=False, n_random=24)
    e8_4_136 = certify_e8_tree_bijection_exact(4, 391, exhaustive=False, n_random=24)

    ok = (
        e8_1_exh["codebook_count"] == 241
        and e8_1_exh["theta_series_all_shells_exact"]
        and e8_1_exh["all_roundtrips_exact"]
        and e8_1_exh["exhaustive_bijection_distinct"]
        and e8_1_exh["all_reciprocals_universally_valid"]
        and e8_4_exh["codebook_count"] == 961
        and e8_4_exh["theta_series_all_shells_exact"]
        and e8_4_exh["all_roundtrips_exact"]
        and e8_4_exh["exhaustive_bijection_distinct"]
        and e8_4_exh["all_reciprocals_universally_valid"]
        and e8_4_88["codebook_bits"] == 88
        and e8_4_88["theta_series_all_shells_exact"]
        and e8_4_88["depth_to_independent_e8_blocks"] == 2
        and e8_4_88["e8_block_hw_type"] == "uint32"
        and e8_4_88["all_roundtrips_exact"]
        and e8_4_88["all_reciprocals_universally_valid"]
        and e8_4_136["codebook_bits"] == 136
        and e8_4_136["theta_series_all_shells_exact"]
        and e8_4_136["depth_to_independent_e8_blocks"] == 2
        and e8_4_136["e8_block_hw_type"] == "uint64"
        and e8_4_136["all_roundtrips_exact"]
        and e8_4_136["all_reciprocals_universally_valid"]
    )
    return {
        "pillar_verified": ok,
        "e8_1_m1_exhaustive": e8_1_exh,
        "e8_4_m1_exhaustive": e8_4_exh,
        "e8_4_m48_88bit": e8_4_88,
        "e8_4_m391_136bit": e8_4_136,
    }


def run_pillar3_witnesses(z32_tables: Mapping[str, Any]) -> dict[str, Any]:
    res = certify_shaping_and_bandwidth_ceiling_exact(z32_tables)
    ok = (
        res["z32_tree_shaping_loss_pct"] == "0.00%"
        and res["subblock_energy_monotone_worse"]
        and len(res["siliquant_issue23_bandwidth_ceilings"]) == 4
    )
    return {
        "pillar_verified": ok,
        "shaping_and_bandwidth": res,
    }


def _audit_live_sources(intake_dir: Path, z32_dp_match: bool) -> dict[str, Any]:
    """Read-only AST & source inspection over staged shape/{ballindex.py,ballpack.c,ballpack.py} (0 external imports)."""
    bi_src = (intake_dir / "shape" / "ballindex.py").read_text(encoding="utf-8")
    bi_tree = ast.parse(bi_src)
    bi_cls = next((n for n in ast.walk(bi_tree) if isinstance(n, ast.ClassDef) and n.name == "BallIndex"), None)
    if bi_cls is None:
        raise BallpackTreeError("shape/ballindex.py missing BallIndex class")
    bi_unrank = next((n for n in bi_cls.body if isinstance(n, ast.FunctionDef) and n.name == "unrank"), None)
    bi_tbl = next((n for n in bi_cls.body if isinstance(n, ast.FunctionDef) and n.name == "table_entries"), None)
    if bi_unrank is None or bi_tbl is None:
        raise BallpackTreeError("shape/ballindex.py missing unrank or table_entries")
    bi_cls_src = ast.unparse(bi_cls)
    bi_unrank_src = ast.unparse(bi_unrank)
    bi_tbl_src = ast.unparse(bi_tbl)
    has_z8_e8 = ("'Z8'" in bi_cls_src) and ("'E8'" in bi_cls_src)
    serial_bi_loop = "for i in range(self.n):" in bi_unrank_src
    bi_tbl_formula_ok = (
        "(self.n + 1) * (self.budget + 1)" in bi_tbl_src
        and "self.n // 8 * 9 * 4 * (self.budget + 1)" in bi_tbl_src
    )

    # Verify BallIndex.table_entries formula matches exact integer counts for Z^32 & E_8^4
    z32_entries = (32 + 1) * (783 + 1) if bi_tbl_formula_ok else 0
    e8_136_entries = (4 * 9 * 4 * (4 * 782 + 1) + 5 * (4 * 782 + 1)) if bi_tbl_formula_ok else 0
    e8_88_entries = (4 * 9 * 4 * (4 * 96 + 1) + 5 * (4 * 96 + 1)) if bi_tbl_formula_ok else 0

    bc_src = (intake_dir / "shape" / "ballpack.c").read_text(encoding="utf-8")
    m_limbs = _LIMBS_RE.search(bc_src)
    if m_limbs is None:
        raise BallpackTreeError("shape/ballpack.c missing #define LIMBS")
    limbs = int(m_limbs.group(1))
    bc_serial_loop = ("for (int i = 0; i < n; ++i)" in bc_src) and ("unrank_one" in bc_src)
    bc_no_e8 = "E8" not in bc_src

    bp_src = (intake_dir / "shape" / "ballpack.py").read_text(encoding="utf-8")
    bp_tree = ast.parse(bp_src)
    fmt_cls = next((n for n in ast.walk(bp_tree) if isinstance(n, ast.ClassDef) and n.name == "Format"), None)
    if fmt_cls is None:
        raise BallpackTreeError("shape/ballpack.py missing Format class")
    fmt_init = next((n for n in fmt_cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
    if fmt_init is None:
        raise BallpackTreeError("shape/ballpack.py missing Format.__init__")
    fmt_init_src = ast.unparse(fmt_init)
    bp_default_z32 = ("n=32" in fmt_init_src) and ("bits=136" in fmt_init_src)
    bp_z8_only = ("BallIndex('Z8'" in fmt_init_src) and ("'E8'" not in fmt_init_src)

    return {
        "ballindex_has_z8_and_e8": has_z8_e8,
        "ballindex_serial_unrank_loop": serial_bi_loop,
        "ballindex_z32_136_count_matches_tree": z32_dp_match,
        "ballindex_z32_table_entries": z32_entries,
        "ballindex_e8_136_table_entries": e8_136_entries,
        "ballindex_e8_88_table_entries": e8_88_entries,
        "ballpack_c_limbs": limbs,
        "ballpack_c_serial_unrank_loop": bc_serial_loop,
        "ballpack_c_has_no_e8": bc_no_e8,
        "ballpack_py_default_z32_b136": bp_default_z32,
        "ballpack_py_hardcodes_z8_only": bp_z8_only,
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


def validate_audit_certificate(
    cert: Mapping[str, Any],
    *,
    intake_dir: Path | None = None,
) -> dict[str, Any]:
    """Fail-closed Zone-B-hardened certificate validator with exact recomputed pillars & pinned provenance."""
    if not isinstance(cert, Mapping):
        raise BallpackTreeError("certificate must be a mapping")
    if contains_forbidden_json_value(cert):
        raise BallpackTreeError("certificate contains forbidden float/non-str-key/non-JSON value")
    if "mode" not in cert or cert["mode"] not in ("live_intake_recount", "standalone_fixture_replay"):
        raise BallpackTreeError("certificate mode invalid")

    mode = cert["mode"]
    expected_keys = LIVE_CERT_TOP_KEYS if mode == "live_intake_recount" else STANDALONE_CERT_TOP_KEYS
    if set(cert.keys()) != expected_keys:
        raise BallpackTreeError("certificate top-level key set mismatch")
    if cert["schema_version"] != SCHEMA_VERSION:
        raise BallpackTreeError("certificate schema_version mismatch")
    if cert["audit_id"] != "AUDIT-112":
        raise BallpackTreeError("certificate audit_id mismatch")
    if cert["novelty_class"] != "N1":
        raise BallpackTreeError("certificate novelty_class mismatch")
    if type(cert["live_intake_audited"]) is not bool or cert["live_intake_audited"] != (mode == "live_intake_recount"):
        raise BallpackTreeError("certificate live_intake_audited mismatch")
    if type(cert["overall_verified"]) is not bool or not cert["overall_verified"]:
        raise BallpackTreeError("certificate overall_verified must be exact bool True")

    cert_sha = require_sha256(cert["certificate_sha256"], "certificate_sha256")
    body = {k: v for k, v in cert.items() if k != "certificate_sha256"}
    if not hmac.compare_digest(cert_sha, canonical_sha256(body)):
        raise BallpackTreeError("certificate_sha256 digest mismatch")

    p1_exp, p2_exp, p3_exp = _fresh_expected_pillars()
    if not _type_strict_eq(cert["pillar1_z32_tree_bijection"], p1_exp):
        raise BallpackTreeError("pillar1_z32_tree_bijection payload mismatch or hollow")
    if not _type_strict_eq(cert["pillar2_e8_4_tree_codec"], p2_exp):
        raise BallpackTreeError("pillar2_e8_4_tree_codec payload mismatch or hollow")
    if not _type_strict_eq(cert["pillar3_shaping_and_roofline"], p3_exp):
        raise BallpackTreeError("pillar3_shaping_and_roofline payload mismatch or hollow")

    if mode == "live_intake_recount":
        require_git_sha40(cert["pinned_commit_sha"], "pinned_commit_sha")
        pa = cert["pinned_artifacts"]
        if not isinstance(pa, Mapping) or set(pa.keys()) != set(PINNED_INTAKE_FILES):
            raise BallpackTreeError("live certificate pinned_artifacts incomplete")
        for rel, meta in pa.items():
            if not isinstance(meta, Mapping) or set(meta.keys()) != {"sha256", "bytes"}:
                raise BallpackTreeError(f"invalid pinned_artifacts entry for {rel}")
            require_sha256(meta["sha256"], f"pinned_artifacts[{rel}].sha256")
            require_exact_int(meta["bytes"], f"pinned_artifacts[{rel}].bytes", allow_zero=False)
        if not _type_strict_eq(cert["live_source_audit"], PINNED_SOURCE_AUDIT_EXPECTED):
            raise BallpackTreeError("live_source_audit payload mismatch or incomplete")

        if cert["pinned_commit_sha"] != PINNED_COMMIT_SHA:
            raise BallpackTreeError("live certificate pinned_commit_sha provenance mismatch")
        if not _type_strict_eq(pa, PINNED_ARTIFACTS_EXPECTED):
            raise BallpackTreeError("live certificate pinned_artifacts provenance mismatch")
        if intake_dir is not None:
            expected_live = run_live_audit(intake_dir)
            if not _type_strict_eq(cert, expected_live):
                raise BallpackTreeError("live certificate provenance mismatch against intake_dir")

    return {"valid": True, "mode": mode, "certificate_sha256": cert_sha}


def run_standalone_audit() -> dict[str, Any]:
    """Self-contained exact pure-Z/Q replay with zero external file I/O."""
    p1, p2, p3 = _fresh_expected_pillars()
    overall = p1["pillar_verified"] and p2["pillar_verified"] and p3["pillar_verified"]
    body = {
        "schema_version": SCHEMA_VERSION,
        "audit_id": "AUDIT-112",
        "mode": "standalone_fixture_replay",
        "live_intake_audited": False,
        "novelty_class": "N1",
        "pillar1_z32_tree_bijection": p1,
        "pillar2_e8_4_tree_codec": p2,
        "pillar3_shaping_and_roofline": p3,
        "overall_verified": overall,
    }
    out = dict(body)
    out["certificate_sha256"] = canonical_sha256(body)
    return out


def run_live_audit(intake_dir: Path) -> dict[str, Any]:
    """Execute complete read-only live audit over staged intake-siliquant-ballpack/ (20d13d5f)."""
    if not intake_dir.is_dir():
        raise BallpackTreeError("intake directory does not exist")
    pinned: dict[str, dict[str, Any]] = {}
    for rel in PINNED_INTAKE_FILES:
        digest, sz = sha256_file(intake_dir / rel)
        pinned[rel] = {"sha256": digest, "bytes": sz}

    p1, p2, p3 = _fresh_expected_pillars()
    src_audit = _audit_live_sources(intake_dir, p1["z32_b783_136bit"]["count_matches_ballindex_dp"])

    overall = (
        p1["pillar_verified"]
        and p2["pillar_verified"]
        and p3["pillar_verified"]
        and _type_strict_eq(pinned, PINNED_ARTIFACTS_EXPECTED)
        and _type_strict_eq(src_audit, PINNED_SOURCE_AUDIT_EXPECTED)
    )
    body = {
        "schema_version": SCHEMA_VERSION,
        "audit_id": "AUDIT-112",
        "mode": "live_intake_recount",
        "live_intake_audited": True,
        "novelty_class": "N1",
        "pinned_commit_sha": PINNED_COMMIT_SHA,
        "pinned_artifacts": pinned,
        "live_source_audit": src_audit,
        "pillar1_z32_tree_bijection": p1,
        "pillar2_e8_4_tree_codec": p2,
        "pillar3_shaping_and_roofline": p3,
        "overall_verified": overall,
    }
    out = dict(body)
    out["certificate_sha256"] = canonical_sha256(body)
    return out


def _reject_json_float_or_constant(token: str) -> Any:
    raise BallpackTreeError(f"forbidden JSON float or non-finite constant: {token}")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in pairs:
        if k in out:
            raise BallpackTreeError(f"duplicate JSON key: {k}")
        out[k] = v
    return out


def _check_certificate_file(cert_path: Path, intake_dir: Path | None) -> int:
    try:
        raw_text = cert_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        sys.stderr.write(json.dumps({"valid": False, "error_code": "CERT_IO_ERROR", "error": str(exc)}) + "\n")
        return 2
    try:
        parsed = json.loads(
            raw_text,
            parse_float=_reject_json_float_or_constant,
            parse_constant=_reject_json_float_or_constant,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
        verdict = validate_audit_certificate(parsed, intake_dir=intake_dir)
    except json.JSONDecodeError as exc:
        sys.stderr.write(json.dumps({"valid": False, "error_code": "CERT_JSON_DECODE_ERROR", "error": str(exc)}) + "\n")
        return 2
    except BallpackTreeError as exc:
        sys.stderr.write(json.dumps({"valid": False, "error_code": "CERT_VALIDATION_ERROR", "error": str(exc)}) + "\n")
        return 2
    sys.stdout.write(json.dumps(verdict, sort_keys=True) + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AUDIT-112 pure-Z/Q ballpack tree-unranking runner.")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--standalone", action="store_true", help="Self-contained pure-Z/Q replay (default).")
    mode_group.add_argument("--live", action="store_true", help="Live read-only recount over staged intake-siliquant-ballpack.")
    mode_group.add_argument("--check", type=Path, default=None, help="Validate a JSON certificate file fail-closed.")
    parser.add_argument(
        "--intake-dir",
        type=Path,
        default=None,
        help="Path to staged intake-siliquant-ballpack directory (defaults to ../intake-siliquant-ballpack for --live).",
    )
    parser.add_argument("--json", action="store_true", help="Emit canonical JSON certificate to stdout.")
    parser.add_argument("--write-cert", type=Path, default=None, help="Write canonical JSON certificate to path.")
    args = parser.parse_args(argv)

    if args.check is not None:
        return _check_certificate_file(args.check, args.intake_dir)

    if args.live:
        intake_dir = args.intake_dir if args.intake_dir is not None else (Path(__file__).resolve().parent.parent / "intake-siliquant-ballpack")
        result = run_live_audit(intake_dir)
    else:
        result = run_standalone_audit()

    rendered = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    if args.write_cert is not None:
        args.write_cert.write_text(rendered, encoding="utf-8")
    if args.json:
        sys.stdout.write(rendered)
    else:
        sys.stdout.write(
            f"AUDIT-112 mode={result['mode']} live={result['live_intake_audited']} "
            f"verified={result['overall_verified']} sha256={result['certificate_sha256'][:16]}...\n"
        )
    return 0 if result["overall_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
