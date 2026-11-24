#!/usr/bin/env python3
"""Fail-closed Aletheia R1-R5 verifier, atomic cert writer, and portal cross-checker.

Verifies the pure-Z/Q siliquant PR #85 Cross-Scale Pattern Atlas certificate
with iterative JSON-tree validation, zero-float enforcement, independent
secondary semantic checks on both FR1 and FR0, atomic dir_fd I/O with
O_NOFOLLOW/O_NONBLOCK/st_nlink==1, and optional read-only portal SHA/JSONL check.
"""
from __future__ import annotations

import errno
from fractions import Fraction
import hashlib
import json
import os
import stat
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import siliquant85_pattern_oracle as oracle

MAX_JSON_BYTES = 1048576
MAX_JSON_DEPTH = 64
MAX_INT_BITS = 2048
MAX_TEMP_RETRIES = 8
HEX_LOWER = frozenset("0123456789abcdef")
DARWIN_PREFIXES = ("/var", "/tmp", "/etc")


class AuditError(ValueError):
    """Typed fail-closed audit error with a machine-checkable error code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def require_ok(cond: bool, code: str, message: str) -> None:
    """Fail-closed exact-boolean assertion replacement (never stripped by -O)."""
    if type(cond) is not bool or not cond:
        raise AuditError(code, message)


def _is_lower_hex64(val: object) -> bool:
    if type(val) is not str or len(val) != 64:
        return False
    return all(ch in HEX_LOWER for ch in val)


def _validate_json_tree(root: object) -> None:
    """Iteratively validate strict JSON-safe Z/Q tree: no floats, subclasses, None, or surrogates."""
    stack: list[tuple[object, int]] = [(root, 0)]
    while stack:
        node, depth = stack.pop()
        if depth > MAX_JSON_DEPTH:
            raise AuditError("JSON_DEPTH_EXCEEDED", "nesting depth exceeds limit")
        t = type(node)
        if t is dict:
            for k, v in node.items():
                if type(k) is not str:
                    raise AuditError("JSON_KEY_TYPE", "dict key must be exact str")
                try:
                    k.encode("utf-8")
                except UnicodeEncodeError as exc:
                    raise AuditError("JSON_BAD_UNICODE", "dict key has lone surrogate") from exc
                stack.append((v, depth + 1))
        elif t is list:
            for item in node:
                stack.append((item, depth + 1))
        elif t is int:
            if node.bit_length() > MAX_INT_BITS:
                raise AuditError("JSON_INT_OVERFLOW", "integer exceeds MAX_INT_BITS")
        elif t is bool:
            continue
        elif t is str:
            try:
                node.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise AuditError("JSON_BAD_UNICODE", "string has lone surrogate") from exc
        else:
            raise AuditError("JSON_FORBIDDEN_TYPE", f"forbidden type: {t.__name__}")


def canonical_json_bytes(obj: object) -> bytes:
    """Encode validated object to canonical sorted ASCII LF-free JSON bytes."""
    _validate_json_tree(obj)
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_certificate() -> dict[str, object]:
    """Build canonical signed certificate object from oracle payload."""
    payload = oracle.build_payload()
    payload_sha = sha256_hex(canonical_json_bytes(payload))
    envelope: dict[str, object] = {
        "schema_version": oracle.SCHEMA_VERSION,
        "payload_sha256": payload_sha,
        "payload": payload,
    }
    cert_sha = sha256_hex(canonical_json_bytes(envelope))
    envelope["certificate_sha256"] = cert_sha
    return envelope


def _frac_from_pair(pair: object, code: str) -> Fraction:
    if type(pair) is not list or len(pair) != 2:
        raise AuditError(code, "rational pair must be 2-element list")
    num, den = pair[0], pair[1]
    if type(num) is not int or type(den) is not int or den <= 0:
        raise AuditError(code, "rational pair requires int numerator and positive int denominator")
    return Fraction(num, den)


def _require_dict(obj: object, code: str) -> dict[str, object]:
    if type(obj) is not dict:
        raise AuditError(code, "expected exact dict")
    return obj


def _require_keys(obj: object, expected: set[str], code: str) -> dict[str, object]:
    d = _require_dict(obj, code)
    if set(d.keys()) != expected:
        raise AuditError(code, "unexpected dictionary key set")
    return d


def secondary_audit_payload(payload: dict[str, object]) -> None:
    """Independent zero-division-safe semantic verification plus exact canonical bijective check."""
    _validate_json_tree(payload)
    top = _require_keys(payload, {"exact", "sampled", "anchor", "summary"}, "PAYLOAD_SCHEMA")

    # 1. Exact pillar independent invariants
    exact = _require_keys(
        top["exact"],
        {"provenance_tier", "hamming_family", "rate_monotone_increasing", "dp_table_reconciliation", "stage_a_invariance"},
        "EXACT_SCHEMA",
    )
    require_ok(exact["provenance_tier"] == "exact", "EXACT_TIER", "exact provenance_tier mismatch")
    require_ok(exact["rate_monotone_increasing"] is True, "EXACT_HAMMING_MISMATCH", "rate_monotone_increasing must be True")
    hf = exact["hamming_family"]
    require_ok(type(hf) is list, "EXACT_HAMMING_TYPE", "hamming_family must be list")
    require_ok(len(hf) == 3, "EXACT_HAMMING_LEN", "hamming_family must have 3 entries")
    for row, exp_m in zip(hf, (3, 4, 5)):
        r = _require_keys(
            row,
            {"m", "n", "k", "d_min", "code_rate_q", "parity_density_q", "det", "covering_sq_128"},
            "EXACT_HAMMING_MISMATCH",
        )
        n = 1 << exp_m
        k = n - exp_m - 1
        rate = _frac_from_pair(r["code_rate_q"], "EXACT_HAMMING_MISMATCH")
        pden = _frac_from_pair(r["parity_density_q"], "EXACT_HAMMING_MISMATCH")
        require_ok(
            r["m"] == exp_m and r["n"] == n and r["k"] == k and r["d_min"] == 4,
            "EXACT_HAMMING_MISMATCH",
            "Hamming parameters mismatch",
        )
        require_ok(
            rate == Fraction(k, n) and pden == Fraction(exp_m + 1, n),
            "EXACT_HAMMING_MISMATCH",
            "Hamming code rate or parity density mismatch",
        )
        require_ok(r["det"] == (1 << (exp_m + 1)) and r["covering_sq_128"] == 32 + 192 // n, "EXACT_HAMMING_MISMATCH", "det/covering mismatch")

    dp = _require_keys(
        exact["dp_table_reconciliation"],
        {
            "a16_b1207_two_level", "a16_b1207_flat", "a16_flat_over_two_level_q",
            "a32_b1015_two_level", "a32_b1015_flat", "a32_two_level_minus_flat",
            "a8_b1567_flat", "z32_b783_flat",
        },
        "EXACT_DP_TABLE_MISMATCH",
    )
    a16_2l = dp["a16_b1207_two_level"]
    a16_fl = dp["a16_b1207_flat"]
    require_ok(type(a16_2l) is int and a16_2l > 0 and type(a16_fl) is int and a16_fl > 0, "EXACT_DP_BOUNDS", "positive DP sizes required")
    require_ok(
        _frac_from_pair(dp["a16_flat_over_two_level_q"], "EXACT_DP_TABLE_MISMATCH") == Fraction(a16_fl, a16_2l),
        "EXACT_DP_TABLE_MISMATCH",
        "A16 DP ratio mismatch",
    )
    a32_2l = dp["a32_b1015_two_level"]
    a32_fl = dp["a32_b1015_flat"]
    a32_diff = dp["a32_two_level_minus_flat"]
    require_ok(
        type(a32_2l) is int and a32_2l > 0 and type(a32_fl) is int and a32_fl > 0 and type(a32_diff) is int,
        "EXACT_DP_BOUNDS",
        "positive A32 DP sizes required",
    )
    require_ok(
        a32_2l - a32_fl == a32_diff,
        "EXACT_DP_TABLE_MISMATCH",
        "A32 DP difference mismatch",
    )
    sa = _require_keys(
        exact["stage_a_invariance"],
        {"seven_arm_scale_perm_exact_invariant", "five_arm_scale_perm_exact_invariant"},
        "EXACT_STAGE_A_INV_FAIL",
    )
    require_ok(
        sa["seven_arm_scale_perm_exact_invariant"] is True and sa["five_arm_scale_perm_exact_invariant"] is True,
        "EXACT_STAGE_A_INV_FAIL",
        "Stage-A invariance flag false",
    )

    # 2. Sampled pillar independent invariants
    sampled = _require_keys(
        top["sampled"],
        {
            "provenance_tier", "source_commit", "measured_live_llm_in_this_track",
            "measured_live_gpu_in_this_track", "number_trace_audit",
            "pattern_1_regime_triangle", "pattern_2_bitrate_amplification",
            "pattern_3_colproxy_sign_inversion", "pattern_4_sphere_and_tax_decomposition",
            "pattern_5_stage_a_rank_matrix",
        },
        "SAMPLED_SCHEMA",
    )
    require_ok(
        sampled["provenance_tier"] == "sampled"
        and sampled["measured_live_llm_in_this_track"] is False
        and sampled["measured_live_gpu_in_this_track"] is False,
        "SAMPLED_PROVENANCE",
        "sampled provenance flags mismatch",
    )
    nt = _require_dict(sampled["number_trace_audit"], "TRACE_AUDIT_MISMATCH")
    require_ok(
        nt.get("pr85_hamming_not_found") == 0
        and nt.get("pr85_packed_not_found") == 0
        and nt.get("c1_holds") is True,
        "TRACE_AUDIT_MISMATCH",
        "number_trace_audit failed core checks",
    )

    p1 = _require_dict(sampled["pattern_1_regime_triangle"], "PATTERN1_REGIME_MISMATCH")
    require_ok(
        p1.get("a16_peaks_pooled_mse_all_three") is True
        and p1.get("llama_a32_beats_hbg_on_both_d_and_e05_yet_loses_kl") is True,
        "PATTERN1_REGIME_MISMATCH",
        "pattern 1 regime checks failed",
    )
    p1_models = _require_keys(p1.get("models"), set(oracle.MODELS), "PATTERN1_REGIME_MISMATCH")
    llama_m = _require_dict(p1_models["Llama-3.2-1B"], "PATTERN1_REGIME_MISMATCH")
    smol_m = _require_dict(p1_models["SmolLM2-1.7B"], "PATTERN1_REGIME_MISMATCH")
    llama_g = _require_dict(llama_m.get("gap_a32_vs_hbg"), "PATTERN1_REGIME_MISMATCH")
    smol_g = _require_dict(smol_m.get("gap_a32_vs_hbg"), "PATTERN1_REGIME_MISMATCH")
    require_ok(
        llama_g.get("sign") == -1
        and _frac_from_pair(llama_g.get("gap_sq_q"), "PATTERN1_REGIME_MISMATCH") > Fraction(18, 1),
        "PATTERN1_REGIME_MISMATCH",
        "Llama A32 vs HBG failure magnitude mismatch",
    )
    require_ok(
        smol_g.get("sign") == 1
        and _frac_from_pair(smol_g.get("gap_sq_q"), "PATTERN1_REGIME_MISMATCH") > Fraction(17, 1),
        "PATTERN1_REGIME_MISMATCH",
        "SmolLM2 A32 vs HBG win magnitude mismatch",
    )

    p2 = _require_dict(sampled["pattern_2_bitrate_amplification"], "PATTERN2_AMP_MISMATCH")
    require_ok(
        p2.get("smollm2_hamming_breaks_e8_catastrophe") is True
        and _frac_from_pair(p2.get("smollm2_hbg_over_a32_amp_ratio_q"), "PATTERN2_AMP_MISMATCH") > Fraction(3, 2),
        "PATTERN2_AMP_MISMATCH",
        "pattern 2 amplification law mismatch",
    )

    p3 = _require_dict(sampled["pattern_3_colproxy_sign_inversion"], "PATTERN3_COLPROXY_MISMATCH")
    require_ok(
        p3.get("total_cells") == 21
        and p3.get("noise_positive_cells") == 21
        and p3.get("gam_attn_positive_count") == 9
        and p3.get("gam_ffn_negative_count") == 6,
        "PATTERN3_COLPROXY_MISMATCH",
        "pattern 3 colproxy sign invariants mismatch",
    )

    p4 = _require_dict(sampled["pattern_4_sphere_and_tax_decomposition"], "PATTERN4_TAX_SPEED_MISMATCH")
    require_ok(
        p4.get("real_ffn_down_is_gaussian_not_laplace") is True,
        "PATTERN4_TAX_SPEED_MISMATCH",
        "pattern 4 sphere concentration mismatch",
    )
    sb = _require_dict(p4.get("silicon_engine_speed_bridge"), "PATTERN4_TAX_SPEED_MISMATCH")
    require_ok(
        _frac_from_pair(sb.get("avx512_enum_8t_over_pr85_max_q"), "PATTERN4_TAX_SPEED_MISMATCH") > Fraction(20, 1)
        and _frac_from_pair(sb.get("sm86_72b_transcoded_over_onfly_q"), "PATTERN4_TAX_SPEED_MISMATCH") > Fraction(700, 1),
        "PATTERN4_TAX_SPEED_MISMATCH",
        "pattern 4 silicon speedup bridge mismatch",
    )

    p5 = _require_dict(sampled["pattern_5_stage_a_rank_matrix"], "PATTERN5_RANK_MISMATCH")
    require_ok(
        p5.get("a16_minimax_top2_all_models_45") is True
        and p5.get("a16_minimax_top2_all_models_30") is True,
        "PATTERN5_RANK_MISMATCH",
        "pattern 5 minimax top-2 invariant mismatch",
    )

    anchor = _require_keys(
        top["anchor"],
        {"provenance_tier", "pr85_commit", "pinned_file_count", "pinned_portal_sha256", "cs13_novelty_tier", "cs13_prior_literature_n0"},
        "ANCHOR_DIGEST_MISMATCH",
    )
    require_ok(
        anchor["provenance_tier"] == "anchor" and anchor["pinned_file_count"] == len(oracle.PINNED_PORTAL_SHA256),
        "ANCHOR_DIGEST_MISMATCH",
        "anchor metadata mismatch",
    )
    digests = _require_dict(anchor["pinned_portal_sha256"], "ANCHOR_DIGEST_MISMATCH")
    require_ok(
        len(digests) == len(oracle.PINNED_PORTAL_SHA256)
        and all(_is_lower_hex64(v) for v in digests.values()),
        "ANCHOR_DIGEST_MISMATCH",
        "anchor digest format invalid",
    )

    # 3. Exact canonical bijective pin against oracle (eliminates all re-signed leaf/extra-key bypasses)
    expected = oracle.build_payload()
    require_ok(
        canonical_json_bytes(payload) == canonical_json_bytes(expected),
        "PAYLOAD_CANONICAL_MISMATCH",
        "payload diverges from canonical pure-Z/Q oracle",
    )


def verify_certificate_object(obj: object, *, full_recompute: bool = True) -> dict[str, object]:
    """Fail-closed verification of a certificate object under FR1 or FR0."""
    require_ok(type(full_recompute) is bool, "INVALID_BOOL", "full_recompute must be exact bool")
    _validate_json_tree(obj)
    cert = _require_keys(obj, {"schema_version", "payload_sha256", "payload", "certificate_sha256"}, "CERT_SCHEMA")
    require_ok(cert["schema_version"] == oracle.SCHEMA_VERSION, "SCHEMA_MISMATCH", "schema_version mismatch")
    require_ok(_is_lower_hex64(cert["payload_sha256"]), "BAD_PAYLOAD_SHA", "payload_sha256 must be 64 lower hex")
    require_ok(_is_lower_hex64(cert["certificate_sha256"]), "BAD_CERT_SHA", "certificate_sha256 must be 64 lower hex")

    payload = cert["payload"]
    require_ok(type(payload) is dict, "PAYLOAD_NOT_DICT", "payload must be dict")
    actual_payload_sha = sha256_hex(canonical_json_bytes(payload))
    require_ok(actual_payload_sha == cert["payload_sha256"], "PAYLOAD_SHA_MISMATCH", "payload_sha256 mismatch")

    unsigned = {
        "schema_version": cert["schema_version"],
        "payload_sha256": cert["payload_sha256"],
        "payload": payload,
    }
    actual_cert_sha = sha256_hex(canonical_json_bytes(unsigned))
    require_ok(actual_cert_sha == cert["certificate_sha256"], "CERT_SHA_MISMATCH", "certificate_sha256 mismatch")

    secondary_audit_payload(payload)
    if full_recompute:
        rebuilt = build_certificate()
        require_ok(
            cert["certificate_sha256"] == rebuilt["certificate_sha256"],
            "FULL_RECOMPUTE_MISMATCH",
            "full_recompute certificate hash mismatch",
        )
    return cert


def _normalize_darwin_prefix(
    path: str,
    platform_name: str,
    lstat_fn=os.lstat,
    readlink_fn=os.readlink,
) -> str:
    """Normalize root-owned macOS /var, /tmp, /etc symlinks to /private/<prefix>."""
    if platform_name != "darwin":
        return path
    for pref in DARWIN_PREFIXES:
        if path == pref or path.startswith(pref + "/"):
            try:
                st = lstat_fn(pref)
            except OSError:
                return path
            if stat.S_ISLNK(st.st_mode) and st.st_uid == 0:
                try:
                    target = readlink_fn(pref)
                except OSError:
                    return path
                if target == "private" + pref or target == "/private" + pref:
                    return "/private" + path
    return path


def _safe_getcwd() -> str:
    """Resolve current working directory, honoring $PWD only when (st_dev, st_ino) matches."""
    real_cwd = os.getcwd()
    env_pwd = os.environ.get("PWD", "")
    if os.path.isabs(env_pwd):
        try:
            st_env = os.stat(env_pwd)
            st_real = os.stat(real_cwd)
        except (OSError, UnicodeEncodeError):
            return real_cwd
        if st_env.st_dev == st_real.st_dev and st_env.st_ino == st_real.st_ino:
            return env_pwd
    return real_cwd


def _split_clean_path(raw_path: str) -> tuple[list[str], str]:
    if type(raw_path) is not str or not raw_path or "\x00" in raw_path:
        raise AuditError("INVALID_PATH", "path must be non-empty NUL-free str")
    try:
        raw_path.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise AuditError("INVALID_PATH", "path contains lone surrogate") from exc
    if os.path.isabs(raw_path):
        full = raw_path
    else:
        full = os.path.join(_safe_getcwd(), raw_path)
    full = _normalize_darwin_prefix(full, sys.platform)
    parts = [p for p in full.split("/") if p]
    if not parts or any(p in (".", "..") for p in parts):
        raise AuditError("INVALID_PATH", "path must not contain dot segments or be root")
    return parts[:-1], parts[-1]


def _open_dir_chain(dir_parts: list[str], *, create_missing: bool) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        cur_fd = os.open("/", flags)
    except OSError as exc:
        raise AuditError("IO_ERROR", f"cannot open root directory: {exc}") from exc
    try:
        for seg in dir_parts:
            try:
                nxt_fd = os.open(seg, flags, dir_fd=cur_fd)
            except FileNotFoundError as exc:
                if not create_missing:
                    raise AuditError("IO_ERROR", f"directory not found: {seg}") from exc
                try:
                    os.mkdir(seg, 0o700, dir_fd=cur_fd)
                    nxt_fd = os.open(seg, flags, dir_fd=cur_fd)
                except OSError as mk_exc:
                    raise AuditError("IO_ERROR", f"cannot create directory {seg}: {mk_exc}") from mk_exc
            except OSError as exc:
                raise AuditError("IO_ERROR", f"unsafe or unreadable directory segment {seg}: {exc}") from exc
            os.close(cur_fd)
            cur_fd = nxt_fd
        return cur_fd
    except Exception:
        os.close(cur_fd)
        raise


def _read_regular_bytes_in_dir(dir_fd: int, name: str, max_bytes: int) -> bytes:
    oflags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        fd = os.open(name, oflags, dir_fd=dir_fd)
    except OSError as exc:
        raise AuditError("IO_ERROR", f"cannot open file {name}: {exc}") from exc
    try:
        st = os.fstat(fd)
        require_ok(stat.S_ISREG(st.st_mode), "NOT_REGULAR_FILE", f"{name} is not a regular file")
        require_ok(st.st_nlink == 1, "HARDLINK_REJECTED", f"{name} must have st_nlink == 1")
        require_ok(st.st_size <= max_bytes, "FILE_SIZE_EXCEEDS_MAX", f"{name} exceeds byte ceiling")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            total += len(chunk)
            require_ok(total <= max_bytes, "STREAM_READ_EXCEEDS_MAX", f"{name} grew past byte ceiling")
            chunks.append(chunk)
        return b"".join(chunks)
    except OSError as exc:
        raise AuditError("IO_ERROR", f"read failed for {name}: {exc}") from exc
    finally:
        os.close(fd)


def _reject_json_float(token: str) -> object:
    raise AuditError("JSON_FLOAT_FORBIDDEN", f"float literal forbidden: {token}")


def _reject_json_const(token: str) -> object:
    raise AuditError("JSON_CONST_FORBIDDEN", f"non-standard constant forbidden: {token}")


def _no_dup_json_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    out: dict[str, object] = {}
    for k, v in pairs:
        if k in out:
            raise AuditError("JSON_DUPLICATE_KEY", f"duplicate JSON key: {k}")
        out[k] = v
    return out


def _strict_json_loads(raw: bytes) -> object:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AuditError("JSON_BAD_UTF8", "certificate is not valid UTF-8") from exc

    try:
        return json.loads(
            text,
            parse_float=_reject_json_float,
            parse_constant=_reject_json_const,
            object_pairs_hook=_no_dup_json_pairs,
        )
    except AuditError:
        raise
    except (ValueError, RecursionError) as exc:
        raise AuditError("JSON_PARSE_ERROR", f"invalid JSON: {exc}") from exc


def read_certificate_file(path_str: str) -> dict[str, object]:
    """Read, parse strictly, and verify canonical LF-only certificate JSON via safe dir_fd I/O."""
    dir_parts, leaf = _split_clean_path(path_str)
    dir_fd = _open_dir_chain(dir_parts, create_missing=False)
    try:
        raw = _read_regular_bytes_in_dir(dir_fd, leaf, MAX_JSON_BYTES)
    finally:
        os.close(dir_fd)
    obj = _strict_json_loads(raw)
    cert = verify_certificate_object(obj, full_recompute=True)
    require_ok(
        raw == canonical_json_bytes(cert) + b"\n",
        "NON_CANONICAL_BYTES",
        "certificate file bytes must be canonical LF-only JSON",
    )
    return cert


def write_certificate_atomic(path_str: str, cert_obj: dict[str, object]) -> str:
    """Atomically write canonical 0o600 LF-terminated certificate using dir_fd and EEXIST retry."""
    verify_certificate_object(cert_obj, full_recompute=True)
    data = canonical_json_bytes(cert_obj) + b"\n"
    dir_parts, leaf = _split_clean_path(path_str)
    dir_fd = _open_dir_chain(dir_parts, create_missing=True)
    tmp_name = ""
    try:
        # Reject symlink or multi-link destination if it already exists
        try:
            dst_st = os.lstat(leaf, dir_fd=dir_fd)
        except FileNotFoundError:
            dst_st = None
        except OSError as exc:
            raise AuditError("IO_ERROR", f"lstat failed on destination: {exc}") from exc
        if dst_st is not None:
            require_ok(stat.S_ISREG(dst_st.st_mode), "NOT_REGULAR_FILE", "destination must be regular file")
            require_ok(dst_st.st_nlink == 1, "HARDLINK_REJECTED", "destination must have st_nlink == 1")

        pid = os.getpid()
        fd = -1
        for attempt in range(MAX_TEMP_RETRIES):
            cand = f".{leaf}.tmp.{pid}.{attempt}"
            try:
                fd = os.open(cand, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=dir_fd)
                tmp_name = cand
                break
            except OSError as exc:
                if exc.errno == errno.EEXIST:
                    continue
                raise AuditError("TEMP_OPEN_ERROR", f"cannot create temp file: {exc}") from exc
        require_ok(fd != -1, "TEMP_RETRY_EXHAUSTED", "exhausted temporary file retries")
        try:
            os.fchmod(fd, 0o600)
            st = os.fstat(fd)
            require_ok(stat.S_ISREG(st.st_mode) and st.st_nlink == 1, "HARDLINK_REJECTED", "temp file link invariant violated")
            view = memoryview(data)
            written = 0
            while written < len(view):
                n = os.write(fd, view[written:])
                require_ok(n > 0, "IO_ERROR", "short zero-byte write")
                written += n
            os.fsync(fd)
        except OSError as exc:
            raise AuditError("IO_ERROR", f"temp write/fsync failed: {exc}") from exc
        finally:
            os.close(fd)

        try:
            os.replace(tmp_name, leaf, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
            tmp_name = ""
            os.fsync(dir_fd)
        except OSError as exc:
            raise AuditError("IO_ERROR", f"atomic replace failed: {exc}") from exc
        return sha256_hex(data)
    finally:
        if tmp_name:
            try:
                os.unlink(tmp_name, dir_fd=dir_fd)
            except OSError:
                pass
        os.close(dir_fd)


def _decimal_text_to_micro(raw_str: str) -> int:
    """Convert a decimal string (e.g. '0.018974') to exact rounded micro-units without floats."""
    if type(raw_str) is not str:
        raise AuditError("PORTAL_DECIMAL_INVALID", "decimal text must be exact str")
    try:
        val = Fraction(raw_str) * 1000000
    except (ValueError, ZeroDivisionError) as exc:
        raise AuditError("PORTAL_DECIMAL_INVALID", f"invalid decimal text: {raw_str}") from exc
    num, den = val.numerator, val.denominator
    return (2 * num + den) // (2 * den)


def _parse_jsonl_rows(raw: bytes) -> list[dict[str, object]]:
    try:
        text = raw.decode("utf-8")
        out: list[dict[str, object]] = []
        for line in text.splitlines():
            if not line.strip():
                continue
            row = json.loads(
                line,
                parse_float=str,
                parse_constant=_reject_json_const,
                object_pairs_hook=_no_dup_json_pairs,
            )
            if type(row) is not dict or "model" not in row or "variant" not in row or "kl_mean" not in row:
                raise ValueError("missing required JSONL row fields")
            out.append(row)
        return out
    except (UnicodeDecodeError, ValueError, TypeError) as exc:
        raise AuditError("PORTAL_JSONL_SCHEMA", f"malformed portal JSONL row: {exc}") from exc


def verify_portal_readonly(portal_dir: str) -> dict[str, object]:
    """Read-only SHA-256 and raw-JSONL cross-check against a PR #85 portal checkout."""
    base_parts, base_leaf = _split_clean_path(portal_dir)
    base_fd = _open_dir_chain(base_parts + [base_leaf], create_missing=False)
    verified_files = 0
    raw_cache: dict[str, bytes] = {}
    try:
        for rel_path, expected_sha in oracle.PINNED_PORTAL_SHA256.items():
            segs = rel_path.split("/")
            cur = base_fd
            opened_fds: list[int] = []
            try:
                for seg in segs[:-1]:
                    nxt = os.open(seg, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=cur)
                    opened_fds.append(nxt)
                    cur = nxt
                raw = _read_regular_bytes_in_dir(cur, segs[-1], MAX_JSON_BYTES)
            except OSError as exc:
                raise AuditError("PORTAL_IO_ERROR", f"cannot read portal file {rel_path}: {exc}") from exc
            finally:
                for ofd in reversed(opened_fds):
                    os.close(ofd)
            actual_sha = sha256_hex(raw)
            require_ok(actual_sha == expected_sha, "PORTAL_SHA_MISMATCH", f"SHA mismatch for {rel_path}")
            verified_files += 1
            if rel_path.endswith(".jsonl"):
                raw_cache[rel_path] = raw
    finally:
        os.close(base_fd)

    # Cross-check raw JSONL KL micro-values against oracle tables (zero floats via parse_float=str)
    model_map = {
        "qwen2.5-1.5b-instruct-fp16.gguf": "Qwen2.5-1.5B",
        "SmolLM2-1.7B-Instruct-f16.gguf": "SmolLM2-1.7B",
        "Llama-3.2-1B-Instruct-f16.gguf": "Llama-3.2-1B",
    }
    checked_rows = 0
    for row in _parse_jsonl_rows(raw_cache["experiments/hamming-lattices-real-weights-2026-11-24/results.jsonl"]):
        mname = model_map.get(str(row["model"]), "")
        var = str(row["variant"])
        if mname and var in ("A16BG128:b560s16", "A32BG128:b560s16"):
            arm = var.split("128:")[0]
            require_ok(
                _decimal_text_to_micro(str(row["kl_mean"])) == oracle.KL_45_MICRO[mname][arm],
                "PORTAL_VALUE_MISMATCH",
                f"KL mismatch for {mname} {var}",
            )
            checked_rows += 1
        elif mname and var in ("A16BG128:b368s16", "A32BG128:b368s16"):
            arm = var.split("128:")[0]
            require_ok(
                _decimal_text_to_micro(str(row["kl_mean"])) == oracle.KL_30_MICRO[mname][arm],
                "PORTAL_VALUE_MISMATCH",
                f"3.0 bpw KL mismatch for {mname} {var}",
            )
            checked_rows += 1

    for row in _parse_jsonl_rows(raw_cache["experiments/packed-hamming-2026-11-24/results.jsonl"]):
        mname = model_map.get(str(row["model"]), "")
        require_ok(bool(mname), "PORTAL_JSONL_SCHEMA", "unknown packed model name")
        arm = "PH16+K10" if str(row["variant"]).endswith("+K10") else "PH16"
        require_ok(
            _decimal_text_to_micro(str(row["kl_mean"])) == oracle.PACKED_KL_MICRO[mname][arm],
            "PORTAL_VALUE_MISMATCH",
            f"packed KL mismatch for {mname} {arm}",
        )
        checked_rows += 1

    require_ok(checked_rows == 16, "PORTAL_ROW_COUNT", f"expected 16 checked rows, got {checked_rows}")
    return {
        "portal_verified": True,
        "verified_files": verified_files,
        "checked_jsonl_rows": checked_rows,
        "pr85_commit": oracle.PR85_COMMIT,
    }


def _parse_cli_args(argv: object) -> tuple[str, str, bool]:
    if type(argv) not in (list, tuple):
        raise AuditError("INVALID_CLI_ARGS", "argv must be list or tuple of str")
    args: list[str] = []
    for item in argv:
        if type(item) is not str:
            raise AuditError("INVALID_CLI_ARGS", "CLI argument must be exact str")
        args.append(item)

    if not args:
        return ("standalone", "", False)
    if args == ["--standalone"]:
        return ("standalone", "", False)
    if args == ["--standalone", "--json"]:
        return ("standalone", "", True)

    mode_flags = {"--write": "write", "--check": "check", "--verify-portal": "portal"}
    if args[0] in mode_flags:
        mode = mode_flags[args[0]]
        if len(args) < 2 or not args[1] or args[1].startswith("-"):
            raise AuditError("INVALID_CLI_ARGS", f"{args[0]} requires a non-flag path argument")
        target = args[1]
        rest = args[2:]
        if not rest:
            return (mode, target, False)
        if rest == ["--json"] and mode in ("check", "portal"):
            return (mode, target, True)
        raise AuditError("INVALID_CLI_ARGS", "unexpected trailing arguments")

    raise AuditError("INVALID_CLI_ARGS", "unsupported CLI arguments")


def main(argv: object = None) -> int:
    """Fail-closed CLI entry point."""
    try:
        raw_argv = sys.argv[1:] if argv is None else argv
        mode, target, as_json = _parse_cli_args(raw_argv)
        if mode == "standalone":
            cert = build_certificate()
            verify_certificate_object(cert, full_recompute=True)
            if as_json:
                sys.stdout.write(canonical_json_bytes(cert).decode("utf-8") + "\n")
            else:
                sys.stdout.write(f"PASS standalone {cert['certificate_sha256']}\n")
            return 0
        if mode == "write":
            cert = build_certificate()
            file_sha = write_certificate_atomic(target, cert)
            sys.stdout.write(f"WROTE {cert['certificate_sha256']} file_sha256={file_sha}\n")
            return 0
        if mode == "check":
            cert = read_certificate_file(target)
            if as_json:
                sys.stdout.write(canonical_json_bytes(cert).decode("utf-8") + "\n")
            else:
                sys.stdout.write(f"PASS check {cert['certificate_sha256']}\n")
            return 0
        res = verify_portal_readonly(target)
        if as_json:
            sys.stdout.write(canonical_json_bytes(res).decode("utf-8") + "\n")
        else:
            sys.stdout.write(
                f"PASS portal files={res['verified_files']} rows={res['checked_jsonl_rows']}\n"
            )
        return 0
    except AuditError as exc:
        sys.stderr.write(f"FAIL {exc.code}: {exc.message}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
