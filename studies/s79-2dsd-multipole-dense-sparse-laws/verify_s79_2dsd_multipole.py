#!/usr/bin/env python3
"""Hardened Standalone CLI, Pure-Z/Q Secondary Auditor, R9+R10 Static Lean 4 Scanner,
and Live Lean 4.36.0 Kernel + `#print axioms` Verifier for Track S79.

100% pure Python 3 stdlib, zero IEEE-754 floats, zero bare asserts, strict ASCII digit
table, atomic `O_NOFOLLOW` dir_fd traversal, durable `0o600` atomic write, zero
mathematical RNG (CSPRNG tokens used strictly for collision-free temp filenames).
"""
from __future__ import annotations

import errno
from fractions import Fraction
import os
from pathlib import Path
import re
import secrets
import shutil
import stat
import subprocess
import sys
import threading
from typing import Any

_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import s79_2dsd_multipole_oracle as oracle  # noqa: E402

MAX_JSON_BYTES = oracle.MAX_JSON_BYTES
MAX_ALLOWED_CAP_BYTES = 64 * 1024 * 1024
_READ_CHUNK_BYTES = 65_536
_EXACT_PATH_TYPES = (Path, type(Path()))
_DIR_OPEN_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
_TEMP_NAME_LOCK = threading.Lock()
_TEMP_COUNTER = 0

STANDARD_LEAN4_AXIOMS: frozenset[str] = frozenset({"propext"})
_DARWIN_SYSTEM_ROOT_ALIASES: tuple[tuple[str, str], ...] = (
    ("/var", "/private/var"),
    ("/tmp", "/private/tmp"),
    ("/etc", "/private/etc"),
)

# ---------------------------------------------------------------------------
# Full R9 + R10 Static Lean 4 Scanner (includes LEAN4_METAPROGRAMMING ban)
# ---------------------------------------------------------------------------

_LEAN4_IDENT_EXTRA = "_.'"
_LEAN4_CHAR_PREV_IDENT = "_'»!?"
_LEAN4_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")
_LEAN4_SIMPLE_GUILLEMET_INNER_RE = re.compile(r"\A[A-Za-z_][A-Za-z0-9_']*\Z")
_LEAN4_GUILLEMET_IDENT_RE = re.compile(r"«([A-Za-z_][A-Za-z0-9_']*)»")

_LEAN4_HARD_LINE_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bsorry\b"), "LEAN4_SORRY"),
    (re.compile(r"\badmit\b"), "LEAN4_ADMIT"),
    (re.compile(r"(?<!\.)\baxiom\b"), "LEAN4_AXIOM"),
    (re.compile(r"(?<!\.)\bunsafe\b"), "LEAN4_UNSAFE"),
    (re.compile(r"\bnative_decide\b"), "LEAN4_NATIVE_DECIDE"),
    (re.compile(r"\bsorryAx\b"), "LEAN4_SORRY_AX"),
    (re.compile(r"\bLean\.Meta\.mkSorry\b|\bmkSorry\b"), "LEAN4_MK_SORRY"),
    (re.compile(r"\bLean\.ofReduceBool\b|\bofReduceBool\b"), "LEAN4_OF_REDUCE_BOOL"),
    (re.compile(r"\bLean\.ofReduceNat\b|\bofReduceNat\b"), "LEAN4_OF_REDUCE_NAT"),
    (re.compile(r"\bLean\.trustCompiler\b|\btrustCompiler\b"), "LEAN4_TRUST_COMPILER"),
    (
        re.compile(r"(?<!\.)\b(?:syntax|macro|macro_rules|elab|elab_rules|declare_syntax_cat)\b"),
        "LEAN4_METAPROGRAMMING",
    ),
)
_LEAN4_PARTIAL_DECL_RE = re.compile(r"(?<!\.)\bpartial\s+(?:def|instance|abbrev)\b", flags=re.DOTALL)
_LEAN4_SKIP_KERNEL_TC_RE = re.compile(
    r"\bdebug\s*\.\s*(?:skipKernelTC|byAsSorry|proofAsSorry|terminalTacticsAsSorry)\b",
    flags=re.DOTALL,
)
_LEAN4_IMPORT_FULL_RE = re.compile(
    r"(?:^|\n)[ \t]*(?:(?:public|private|meta)\s+)*import\s+([A-Za-z0-9_.]+)",
    flags=re.DOTALL,
)
_LEAN4_PRINT_AXIOMS_KW_BLOCK = (
    "def|theorem|lemma|axiom|inductive|structure|class|instance|opaque|abbrev|example|"
    "namespace|section|end|open|variable|universe|import|set_option|attribute|"
    "local|scoped|private|protected|noncomputable|partial|unsafe|meta|where|by|in|let|have"
)
_LEAN4_EVAL_SPOOF_RE = re.compile(
    r"#(?:eval!?(?![\w'!?])|reduce(?![\w'!?])|exit(?![\w'!?])|"
    r"print(?![\w'!?])(?!\s+axioms[ \t]+(?!(?:"
    + _LEAN4_PRINT_AXIOMS_KW_BLOCK
    + r")(?![\w'!?]))[A-Za-z_]))"
    r"|(?<!\.)\b(?:run_cmd|run_meta|initialize|builtin_initialize)\b",
    flags=re.DOTALL,
)
_LEAN4_ATTR_OPEN_RE = re.compile(r"(?:@|(?<!\.)\battribute\s*)\[", flags=re.DOTALL)
_LEAN4_HARD_ATTR_KW_RE = re.compile(r"\b(extern|implemented_by)\b")
_LEAN4_CSIMP_ATTR_KW_RE = re.compile(r"\b(csimp)\b")
_LEAN4_DECIDE_KW_RE = re.compile(r"\bdecide\b")
_LEAN4_FLAG_TOK_RE = re.compile(r"[+-][A-Za-z0-9_]+")
_LEAN4_OPT_PREFIX_RE = re.compile(r"\s*([A-Za-z0-9_]+)\s*:=")
_LEAN4_CONFIG_NATIVE_RE = re.compile(r"\bnative\b(?:\s*:=\s*([^},]*))?")
_LEAN4_CONFIG_AFTER_COMMA_RE = re.compile(r",\s*(?:\Z|[A-Za-z_][A-Za-z0-9_']*\s*:=)")
_LEAN4_CONFIG_WITH_RE = re.compile(r"\bwith\b")
_LEAN4_DECIDE_CONFIG_BIND_RE = re.compile(
    r"\b(?:DecideConfig|Decide\.Config)\b[ \t)]*:=\s*(?:\(\s*)*\{[^}{]*\bnative\s*:=\s*([^}{,]*)(?:,[^}{]*)?\}"
    r"|\{[^}{]*\bnative\s*:=\s*([^}{,]*)(?:,[^}{]*)?\}\s*:\s*(?:[A-Za-z_][A-Za-z0-9_]*\.)*(?:DecideConfig|Decide\.Config)\b"
    r"|\{[^}{]*\bnative\s*:=\s*([^}{,:]*)(?:,[^}{]*)?:\s*(?:[A-Za-z_][A-Za-z0-9_]*\.)*(?:DecideConfig|Decide\.Config)\b[^}{]*\}",
    flags=re.DOTALL,
)
_LEAN4_VACUOUS_THM_RE = re.compile(
    r"^\s*(?:private\s+|protected\s+)?(?:theorem|lemma)\s+[A-Za-z0-9_.']+\b[^:\n]*:\s*(?:False\s*(?:→|->)|True\s*:=)"
)
_LEAN4_EXISTS_IMP_RE = re.compile(r"∃[^,\n]+,\s*(?!.*(?:∀|∧))[^,\n]*(?:→|->)")
_AXIOM_BLOCK_RE = re.compile(
    r"'([^']+)'\s+(?:(does not depend on any axioms)|depends on axioms:\s*\[([^\]]*)\])",
    flags=re.DOTALL,
)

# Strict schema keysets for field-complete secondary audit
_CERT_KEYS = frozenset({"schema_version", "project_id", "oracle_module_sha256", "payload_sha256", "certificate_sha256", "payload"})
_PAYLOAD_KEYS = frozenset({
    "project_id",
    "pillar1_2dsd_cascade_and_kv_laws",
    "pillar2_dense_sparse_vs_siliquant",
    "pillar3_multipole_error_and_barrier",
    "pillar4_formal_inventory_and_scope",
    "summary",
})
_P1_KEYS = frozenset({
    "cascade_target_mass",
    "cascade_stage_cum_accepted",
    "cascade_final_residual_mass",
    "cascade_order_invariance_verified",
    "ouro14b_greedy_2dsd_over_dflash_q",
    "ouro26b_greedy_2dsd_over_dflash_q",
    "ouro26b_greedy_2dsd_over_loopspec_q",
    "dory4b_greedy_2dsd_over_dflash_q",
    "ouro26b_sampling_2dsd_over_dflash_q",
    "dory4b_sampling_2dsd_over_dflash_q",
    "kv_ratio_r4_w64_l2048_q",
    "kv_ratio_r4_w64_l8192_q",
    "kv_ratio_r4_w64_l32768_q",
    "kv_ratio_r2_w64_l32768_q",
    "ouro14b_hard32k_tf_delta_pct_q",
    "ouro26b_hard32k_tf_delta_pct_q",
    "dory4b_hard32k_tf_delta_pct_q",
    "circular_ep_rho3_4_loop4_decay_q",
    "inkling2_mini_varloop_drafter_params",
})
_P2_KEYS = frozenset({
    "squeezellm_csr_delta_b_q",
    "squeezellm_csr_b_eff_q",
    "squeezellm_csr_over_dense_ratio_q",
    "squeezellm_csr_wins",
    "portal77_sidecar_delta_b_q",
    "portal77_sidecar_over_dense_ratio_q",
    "portal77_sidecar_wins",
    "reversal_rho_1_32_delta_b_q",
    "reversal_csr_over_dense_ratio_q",
    "reversal_pure_dense_wins",
    "s72_qwen3b_snake_nkl",
    "s72_qwen3b_h2_nkl",
    "s72_qwen3b_h4_nkl",
    "s72_qwen3b_snake_vs_h2_rel_q",
    "s72_qwen3b_snake_vs_h4_rel_q",
    "safe_csr_sparsity_cap_at_32b_sOut40pct_q",
    "safe_cap_pow5_lhs",
    "safe_cap_pow5_rhs",
})
_P3_KEYS = frozenset({
    "kernel_block_tokens",
    "witness_q_norm_sq",
    "witness_k_diff_norm_sq",
    "witness_cs_lhs",
    "witness_cs_rhs",
    "witness_centroid_linear_residual",
    "witness_actual_logit_spread_sq_q",
    "witness_cs_logit_bound_sq_q",
    "jensen_d4_eps1_mass_ratio_q",
    "softmax_bracket_lower_q",
    "softmax_bracket_exact_q",
    "softmax_bracket_upper_q",
    "barrier_dk128_q8_delta1_eps_sq_q",
    "barrier_dk128_q8_delta2_eps_sq_q",
    "witness_value_weight_cov_residual",
})
_P4_KEYS = frozenset({
    "lean_toolchain",
    "lean_toolchain_sha256",
    "lakefile_sha256",
    "module_count",
    "theorem_count",
    "zero_axiom_theorem_count",
    "propext_only_theorem_count",
    "universal_theorem_count",
    "literal_ordering_check_count",
    "literal_ordering_checks",
    "modules",
    "primary_source_commits",
    "primary_source_sha256",
    "novelty_tier",
    "oracle_class",
    "max_int_bits_exact_cap",
    "max_int_digits_preparse_ceiling",
    "measured_live_gpu_in_this_bundle",
    "zero_mathlib_imports",
    "zero_trust_path_floats",
    "zero_bare_asserts",
    "zero_mathematical_rng",
    "prior_literature_n0",
})
_MOD_ROW_KEYS = frozenset({"relpath", "namespace", "sha256", "theorem_count", "theorems"})
_SUMMARY_KEYS = frozenset({
    "all_pillars_verified",
    "kernel_checked_declaration_count",
    "universal_theorem_count",
    "literal_ordering_check_count",
    "literal_ordering_checks",
    "zero_axiom_declaration_count",
    "propext_only_declaration_count",
    "declaration_kinds_note",
    "target_max_cert_bytes",
})


# ---------------------------------------------------------------------------
# Hardened R9 + R10 Static Lean 4 Scanner
# ---------------------------------------------------------------------------

def _is_lean4_ident_prev(source: str, pos: int) -> bool:
    """True iff `source[pos]` continues a Lean 4 identifier (distinguishes `foo!r` from unary `!r`)."""
    if pos < 0:
        return False
    ch = source[pos]
    if ch.isalnum() or ch in _LEAN4_IDENT_EXTRA:
        return True
    if ch in "!?":
        for k in range(pos - 1, -1, -1):
            if source[k] not in "!?":
                return source[k].isalnum() or source[k] in "_'"
        return False
    return False


def _is_lean4_native_rhs_false_span(text: str, lo: int, hi: int) -> bool:
    s = text[lo:hi].strip()
    while s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()
    return s in ("false", "Bool.false")


def _is_lean4_safe_config_span(text: str, lo: int, hi: int, match_rparen: dict[int, int]) -> bool:
    while lo < hi:
        sub = text[lo:hi]
        lo += len(sub) - len(sub.lstrip())
        hi -= len(sub) - len(sub.rstrip())
        if match_rparen.get(lo) == hi - 1:
            lo += 1
            hi -= 1
            continue
        break
    span = text[lo:hi]
    if not (span.startswith("{") and span.endswith("}")):
        return False
    body_lo = lo + 1
    body_hi = hi - 1
    body = text[body_lo:body_hi]
    if "{" in body or "}" in body:
        return False
    if _LEAN4_CONFIG_WITH_RE.search(text, body_lo, body_hi) is not None:
        return False
    for m_nv in _LEAN4_CONFIG_NATIVE_RE.finditer(text, body_lo, body_hi):
        if not _is_lean4_native_rhs_false_span(text, m_nv.start(1), m_nv.end(1)):
            return False
        sep = m_nv.end()
        if sep != body_hi and not _LEAN4_CONFIG_AFTER_COMMA_RE.match(text, sep, body_hi):
            return False
    return True


def _scan_lean4_decide_native(cleaned: str) -> list[tuple[int, int]]:
    if "decide" not in cleaned:
        return []
    n = len(cleaned)
    match_rparen: dict[int, int] = {}
    paren_stack: list[int] = []
    for idx, ch in enumerate(cleaned):
        if ch == "(":
            paren_stack.append(idx)
        elif ch == ")" and paren_stack:
            match_rparen[paren_stack.pop()] = idx

    spans: list[tuple[int, int]] = []
    for m_dec in _LEAN4_DECIDE_KW_RE.finditer(cleaned):
        start = m_dec.start()
        pos = m_dec.end()
        native_hit = False
        while True:
            ws = pos
            while ws != n and cleaned[ws].isspace():
                ws += 1
            if ws == n:
                break
            ch = cleaned[ws]
            if ch in "+-":
                m_flag = _LEAN4_FLAG_TOK_RE.match(cleaned, ws)
                if m_flag is None:
                    break
                if m_flag.group(0) == "+native":
                    native_hit = True
                pos = m_flag.end()
                continue
            if ch == "(":
                close_idx = match_rparen.get(ws)
                if close_idx is None:
                    break
                m_pfx = _LEAN4_OPT_PREFIX_RE.match(cleaned, ws + 1, close_idx)
                if m_pfx is None:
                    break
                opt_key = m_pfx.group(1)
                rhs_lo = m_pfx.end()
                if opt_key == "native":
                    if not _is_lean4_native_rhs_false_span(cleaned, rhs_lo, close_idx):
                        native_hit = True
                elif opt_key == "config":
                    if not _is_lean4_safe_config_span(cleaned, rhs_lo, close_idx, match_rparen):
                        native_hit = True
                pos = close_idx + 1
                continue
            break
        if native_hit:
            spans.append((start, pos))
    return spans


def _iter_lean4_attr_blocks(cleaned: str) -> list[tuple[int, int, int]]:
    match_rbrack: dict[int, int] = {}
    brack_stack: list[int] = []
    for idx, ch in enumerate(cleaned):
        if ch == "[":
            brack_stack.append(idx)
        elif ch == "]" and brack_stack:
            match_rbrack[brack_stack.pop()] = idx
    blocks: list[tuple[int, int, int]] = []
    for m_open in _LEAN4_ATTR_OPEN_RE.finditer(cleaned):
        lb = m_open.end() - 1
        rb = match_rbrack.get(lb)
        if rb is not None:
            blocks.append((m_open.start(), lb + 1, rb))
    return blocks


def _match_lean4_char_literal_end(source: str, i: int, n: int) -> int | None:
    if i + 2 < n and source[i + 1] != "\\" and source[i + 1] != "\n" and source[i + 2] == "'":
        return i + 3
    if i + 3 < n and source[i + 1] == "\\":
        if source[i + 3] == "'" and source[i + 2] != "\n":
            return i + 4
        esc = source[i + 2]
        if (
            esc == "x"
            and i + 5 < n
            and source[i + 5] == "'"
            and all(c in _LEAN4_HEX_DIGITS for c in source[i + 3 : i + 5])
        ):
            return i + 6
        if (
            esc == "u"
            and i + 7 < n
            and source[i + 7] == "'"
            and all(c in _LEAN4_HEX_DIGITS for c in source[i + 3 : i + 7])
        ):
            return i + 8
    return None


def strip_lean4_with_report(source: str) -> tuple[str, list[dict[str, Any]]]:
    """R9-hardened Lean 4 comment/string/char/guillemet stripper preserving line count."""
    oracle.require_exact_str(source, "lean_source", non_empty=False)
    out: list[str] = []
    i = 0
    n = len(source)
    depth = 0
    in_str = False
    in_interp = False
    pending_interp = False
    interp_stack: list[list[int]] = []
    raw_hashes = -1
    opened_at = 0
    unterminated: list[dict[str, Any]] = []
    while i < n:
        ch = source[i]
        nxt = source[i + 1] if i + 1 < n else ""
        if depth > 0:
            if ch == "/" and nxt == "-":
                depth += 1
                out.append("  ")
                i += 2
            elif ch == "-" and nxt == "/":
                depth -= 1
                out.append("  ")
                i += 2
            else:
                out.append("\n" if ch == "\n" else " ")
                i += 1
            continue
        if raw_hashes >= 0:
            close_seq = '"' + ("#" * raw_hashes)
            if source.startswith(close_seq, i):
                out.append(" " * len(close_seq))
                i += len(close_seq)
                raw_hashes = -1
            else:
                out.append("\n" if ch == "\n" else " ")
                i += 1
            continue
        if in_str:
            if ch == "\\":
                out.append(" \n" if nxt == "\n" else "  ")
                i += 2
            elif ch == '"':
                in_str = False
                out.append(" ")
                i += 1
            else:
                out.append("\n" if ch == "\n" else " ")
                i += 1
            continue
        if in_interp:
            if ch == "\\":
                out.append(" \n" if nxt == "\n" else "  ")
                i += 2
            elif ch == "{":
                interp_stack.append([1, opened_at])
                in_interp = False
                out.append(" ")
                i += 1
            elif ch == '"':
                in_interp = False
                out.append(" ")
                i += 1
            else:
                out.append("\n" if ch == "\n" else " ")
                i += 1
            continue
        if interp_stack:
            if ch == "{":
                pending_interp = False
                interp_stack[-1][0] += 1
                out.append(ch)
                i += 1
                continue
            if ch == "}":
                pending_interp = False
                interp_stack[-1][0] -= 1
                out.append(ch)
                i += 1
                if interp_stack[-1][0] == 0:
                    _, opened_at = interp_stack.pop()
                    in_interp = True
                continue
        if ch == "/" and nxt == "-":
            depth = 1
            opened_at = i
            out.append("  ")
            i += 2
        elif ch == "-" and nxt == "-":
            nl = source.find("\n", i + 2)
            end_c = n if nl == -1 else nl
            out.append(" " * (end_c - i))
            i = end_c
        elif ch == "r" and not _is_lean4_ident_prev(source, i - 1):
            pending_interp = False
            j = i + 1
            while j < n and source[j] == "#":
                j += 1
            if j < n and source[j] == '"':
                raw_hashes = j - (i + 1)
                opened_at = i
                out.append(" " * (j - i + 1))
                i = j + 1
            else:
                out.append(ch)
                i += 1
        elif ch == "!" and i >= 1 and source[i - 1] in "smf" and not _is_lean4_ident_prev(source, i - 2):
            pending_interp = True
            out.append(ch)
            i += 1
        elif ch == '"':
            if pending_interp:
                in_interp = True
                pending_interp = False
            else:
                in_str = True
            opened_at = i
            out.append(" ")
            i += 1
        elif ch == "«":
            pending_interp = False
            end_g = source.find("»", i + 1)
            if end_g == -1:
                lineno = source.count("\n", 0, i) + 1
                unterminated.append({"line": lineno, "code": "LEAN4_UNTERMINATED_IDENT"})
                out.append("\n" * source.count("\n", i))
                break
            inner = source[i + 1 : end_g]
            if _LEAN4_SIMPLE_GUILLEMET_INNER_RE.match(inner) is not None and inner != "axioms":
                out.append(source[i : end_g + 1])
            else:
                out.append(" " + "".join("\n" if c == "\n" else " " for c in inner) + " ")
            i = end_g + 1
        elif ch == "'" and (i == 0 or not (source[i - 1].isalnum() or source[i - 1] in _LEAN4_CHAR_PREV_IDENT)):
            pending_interp = False
            char_end = _match_lean4_char_literal_end(source, i, n)
            if char_end is not None:
                out.append(" " * (char_end - i))
                i = char_end
            else:
                out.append(ch)
                i += 1
        else:
            if not ch.isspace():
                pending_interp = False
            out.append(ch)
            i += 1
    unclosed_open = depth > 0 or in_str or in_interp or raw_hashes >= 0
    if unclosed_open or interp_stack:
        err_at = opened_at if unclosed_open else interp_stack[-1][1]
        lineno = source.count("\n", 0, err_at) + 1
        unterminated.append(
            {
                "line": lineno,
                "code": "LEAN4_UNTERMINATED_COMMENT" if depth > 0 else "LEAN4_UNTERMINATED_STRING",
            }
        )
    return "".join(out), unterminated


def audit_lean4_source(source: str) -> dict[str, Any]:
    """Full R9+R10 static Lean 4 trust-escape auditor enforcing pure zero-import Init."""
    cleaned, unterminated = strip_lean4_with_report(source)
    cleaned = _LEAN4_GUILLEMET_IDENT_RE.sub(r"\1", cleaned)
    findings: list[dict[str, Any]] = list(unterminated)
    notices: list[dict[str, Any]] = []

    for lineno, line in enumerate(cleaned.splitlines(), start=1):
        for pat, code in _LEAN4_HARD_LINE_RULES:
            if pat.search(line):
                findings.append({"line": lineno, "code": code})
        if _LEAN4_VACUOUS_THM_RE.search(line):
            notices.append({"line": lineno, "code": "LEAN4_VACUOUS_SIGNATURE_NOTICE"})
        if _LEAN4_EXISTS_IMP_RE.search(line):
            notices.append({"line": lineno, "code": "LEAN4_EXISTS_IMPLICATION_NOTICE"})

    for m_imp in _LEAN4_IMPORT_FULL_RE.finditer(cleaned):
        lineno = cleaned[: m_imp.start(1)].count("\n") + 1
        findings.append({"line": lineno, "code": "LEAN4_FORBIDDEN_IMPORT"})

    for m_part in _LEAN4_PARTIAL_DECL_RE.finditer(cleaned):
        lineno = cleaned[: m_part.start()].count("\n") + 1
        findings.append({"line": lineno, "code": "LEAN4_PARTIAL"})

    decide_spans = _scan_lean4_decide_native(cleaned)
    for ds, _ in decide_spans:
        lineno = cleaned[:ds].count("\n") + 1
        findings.append({"line": lineno, "code": "LEAN4_DECIDE_NATIVE"})

    for m_dc in _LEAN4_DECIDE_CONFIG_BIND_RE.finditer(cleaned):
        if any(ds <= m_dc.start() < de for ds, de in decide_spans):
            continue
        if any(grp is not None and not _is_lean4_native_rhs_false_span(grp, 0, len(grp)) for grp in m_dc.groups()):
            lineno = cleaned[: m_dc.start()].count("\n") + 1
            findings.append({"line": lineno, "code": "LEAN4_DECIDE_NATIVE"})

    for m_sk in _LEAN4_SKIP_KERNEL_TC_RE.finditer(cleaned):
        lineno = cleaned[: m_sk.start()].count("\n") + 1
        findings.append({"line": lineno, "code": "LEAN4_SKIP_KERNEL_TC"})

    for m_ev in _LEAN4_EVAL_SPOOF_RE.finditer(cleaned):
        lineno = cleaned[: m_ev.start()].count("\n") + 1
        findings.append({"line": lineno, "code": "LEAN4_EVAL_SPOOF"})

    for attr_start, b_lo, b_hi in _iter_lean4_attr_blocks(cleaned):
        if _LEAN4_HARD_ATTR_KW_RE.search(cleaned, b_lo, b_hi) is not None:
            lineno = cleaned[:attr_start].count("\n") + 1
            findings.append({"line": lineno, "code": "LEAN4_ATTR_TRUST_ESCAPE"})
        if _LEAN4_CSIMP_ATTR_KW_RE.search(cleaned, b_lo, b_hi) is not None:
            lineno = cleaned[:attr_start].count("\n") + 1
            notices.append({"line": lineno, "code": "LEAN4_CSIMP_REWRITE_NOTICE"})

    return {
        "clean": len(findings) == 0,
        "codes": sorted({f["code"] for f in findings}),
        "findings": findings,
        "notices": notices,
    }


# ---------------------------------------------------------------------------
# Darwin Alias Normalization, Atomic dir_fd Traversal & Durable 0o600 Write
# ---------------------------------------------------------------------------

def _coerce_exact_path(path: Any, name: str) -> Path:
    if type(path) is str:
        s = oracle.require_exact_str(path, name)
        if "\x00" in s:
            raise oracle.S79Error("INVALID_PATH", f"{name} must not contain NUL")
        if s.endswith("/"):
            raise oracle.S79Error("INVALID_FILENAME", f"trailing slash refused: {s!r}")
        return Path(s)
    if type(path) in _EXACT_PATH_TYPES:
        raw = oracle.require_exact_str(str(path), name)
        if "\x00" in raw:
            raise oracle.S79Error("INVALID_PATH", f"{name} must not contain NUL")
        return path
    raise oracle.S79Error("INVALID_PATH_TYPE", f"{name} must be str or Path")


def _normalize_darwin_system_prefix(path: Path, *, allow_exact: bool = False) -> Path:
    """Map verified root-owned macOS /var, /tmp, /etc aliases to /private/* without resolving user symlinks."""
    if sys.platform != "darwin" or not path.is_absolute():
        return path
    parts = path.parts
    min_parts = 2 if allow_exact else 3
    if len(parts) < min_parts or ".." in parts:
        return path
    top = "/" + parts[1]
    for alias, canonical in _DARWIN_SYSTEM_ROOT_ALIASES:
        if top != alias:
            continue
        try:
            st = os.lstat(alias)
            if not stat.S_ISLNK(st.st_mode) or st.st_uid != 0:
                return path
            target = os.readlink(alias)
            if target not in (canonical, canonical.lstrip("/")):
                return path
        except OSError:
            return path
        return Path(canonical, *parts[2:])
    return path


def _open_dir_chain_nofollow(parts: tuple[str, ...]) -> int:
    try:
        cur_fd = os.open("/", _DIR_OPEN_FLAGS)
    except OSError as exc:
        raise oracle.S79Error("DIR_OPEN_FAILED", f"cannot open directory root: {exc}") from exc
    for comp in parts:
        if comp in ("", "."):
            continue
        if comp == "..":
            os.close(cur_fd)
            raise oracle.S79Error("PARENT_TRAVERSAL_FORBIDDEN", "path must not contain '..'")
        try:
            next_fd = os.open(comp, _DIR_OPEN_FLAGS, dir_fd=cur_fd)
        except OSError as exc:
            is_sym = exc.errno == errno.ELOOP
            if exc.errno == errno.ENOTDIR:
                try:
                    is_sym = stat.S_ISLNK(os.lstat(comp, dir_fd=cur_fd).st_mode)
                except OSError:
                    is_sym = False
            os.close(cur_fd)
            if is_sym:
                raise oracle.S79Error("SYMLINK_FORBIDDEN", f"symlink component rejected: {comp!r}") from exc
            if exc.errno == errno.ENOTDIR:
                raise oracle.S79Error("NOT_A_DIRECTORY", f"component is not a directory: {comp!r}") from exc
            raise oracle.S79Error("DIR_OPEN_FAILED", f"cannot traverse {comp!r}: {exc}") from exc
        os.close(cur_fd)
        cur_fd = next_fd
    return cur_fd


def _verify_pwd_not_symlink() -> None:
    pwd = os.environ.get("PWD")
    if not pwd or not pwd.startswith("/"):
        return
    oracle.require_exact_str(pwd, "PWD")
    if "\x00" in pwd:
        raise oracle.S79Error("INVALID_PATH", "PWD must not contain NUL")
    try:
        st_dot = os.stat(".")
    except OSError as exc:
        raise oracle.S79Error("CWD_UNAVAILABLE", f"cannot stat current directory: {exc}") from exc
    try:
        st_logical = os.stat(pwd)
    except OSError:
        return
    if (st_logical.st_dev, st_logical.st_ino) != (st_dot.st_dev, st_dot.st_ino):
        return
    norm_pwd = _normalize_darwin_system_prefix(Path(pwd), allow_exact=True)
    parts = tuple(p for p in norm_pwd.parts if p != "/")
    dfd = _open_dir_chain_nofollow(parts)
    try:
        st_pwd = os.fstat(dfd)
    finally:
        os.close(dfd)
    if (st_pwd.st_dev, st_pwd.st_ino) != (st_dot.st_dev, st_dot.st_ino):
        raise oracle.S79Error("PWD_SYMLINK_MISMATCH", "$PWD changed during verification")


def _open_parent_dir_fd(path_obj: Path) -> tuple[int, str]:
    if path_obj.name in ("", ".", ".."):
        raise oracle.S79Error("INVALID_FILENAME", f"invalid target filename: {str(path_obj)!r}")
    if path_obj.is_absolute():
        norm = _normalize_darwin_system_prefix(path_obj)
        parts = tuple(p for p in norm.parts if p != "/")
        return (_open_dir_chain_nofollow(parts[:-1]), parts[-1])
    _verify_pwd_not_symlink()
    try:
        cwd = os.getcwd()
    except OSError as exc:
        raise oracle.S79Error("CWD_UNAVAILABLE", f"current working directory unavailable: {exc}") from exc
    norm_cwd = _normalize_darwin_system_prefix(Path(cwd), allow_exact=True)
    full_parts = tuple(p for p in norm_cwd.parts if p != "/") + path_obj.parts
    return (_open_dir_chain_nofollow(full_parts[:-1]), full_parts[-1])


def read_regular_bytes_capped(path: Any, *, max_bytes: int = MAX_JSON_BYTES) -> bytes:
    """Read regular non-symlink non-hardlinked file via atomic dir_fd + O_NOFOLLOW + O_NONBLOCK."""
    p_obj = _coerce_exact_path(path, "read_path")
    cap = oracle.require_exact_int(max_bytes, "max_bytes", min_val=1, max_val=MAX_ALLOWED_CAP_BYTES)
    dfd, fname = _open_parent_dir_fd(p_obj)
    try:
        flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)
        try:
            fd = os.open(fname, flags, dir_fd=dfd)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise oracle.S79Error("SYMLINK_FORBIDDEN", f"symlink target refused: {fname!r}") from exc
            raise oracle.S79Error("FILE_UNREADABLE", f"cannot open {fname!r}: {exc}") from exc
        try:
            try:
                fst = os.fstat(fd)
            except OSError as exc:
                raise oracle.S79Error("READ_IO_ERROR", f"fstat failed: {exc}") from exc
            if not stat.S_ISREG(fst.st_mode):
                raise oracle.S79Error("NON_REGULAR_FILE", f"opened target not regular: {fname!r}")
            if fst.st_nlink > 1:
                raise oracle.S79Error("HARDLINK_FORBIDDEN", f"opened target hardlinked: {fname!r}")
            chunks: list[bytes] = []
            total = 0
            while True:
                try:
                    buf = os.read(fd, _READ_CHUNK_BYTES)
                except OSError as exc:
                    raise oracle.S79Error("READ_IO_ERROR", f"read failed: {exc}") from exc
                if not buf:
                    break
                total += len(buf)
                if total > cap:
                    raise oracle.S79Error("FILE_TOO_LARGE", f"file exceeds {cap} byte cap")
                chunks.append(buf)
            return b"".join(chunks)
        finally:
            os.close(fd)
    finally:
        os.close(dfd)


def _next_temp_name(base_name: str) -> str:
    global _TEMP_COUNTER
    with _TEMP_NAME_LOCK:
        _TEMP_COUNTER += 1
        seq = _TEMP_COUNTER
    return f".{base_name}.tmp.{os.getpid()}.{seq}.{secrets.token_hex(8)}"


def write_bytes_atomic_0600(path: Any, data: bytes) -> None:
    """Durable 0o600 atomic write via dir_fd, refusing symlinks and multi-link targets."""
    if type(data) is not bytes:
        raise oracle.S79Error("INVALID_BYTES", "data must be exact bytes")
    p_obj = _coerce_exact_path(path, "write_path")
    dfd, fname = _open_parent_dir_fd(p_obj)
    tmp_name: str | None = None
    tmp_fd: int | None = None
    try:
        try:
            lst = os.lstat(fname, dir_fd=dfd)
        except FileNotFoundError:
            lst = None
        except OSError as exc:
            raise oracle.S79Error("TARGET_UNSTATTABLE", f"cannot lstat target {fname!r}: {exc}") from exc
        if lst is not None:
            if stat.S_ISLNK(lst.st_mode):
                raise oracle.S79Error("SYMLINK_FORBIDDEN", f"refusing symlink write target: {fname!r}")
            if not stat.S_ISREG(lst.st_mode):
                raise oracle.S79Error("NON_REGULAR_FILE", f"refusing non-regular write target: {fname!r}")
            if lst.st_nlink > 1:
                raise oracle.S79Error("HARDLINK_FORBIDDEN", f"refusing hardlinked target: {fname!r}")

        create_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        for _ in range(32):
            cand = _next_temp_name(fname)
            try:
                tmp_fd = os.open(cand, create_flags, 0o600, dir_fd=dfd)
                tmp_name = cand
                break
            except FileExistsError:
                continue
            except OSError as exc:
                raise oracle.S79Error("WRITE_IO_ERROR", f"cannot create temp file: {exc}") from exc
        if tmp_fd is None:
            raise oracle.S79Error("TEMP_CREATE_FAILED", "exhausted temp filename retries")

        try:
            view = memoryview(data)
            written = 0
            while written < len(data):
                n = os.write(tmp_fd, view[written:])
                if n <= 0:
                    raise oracle.S79Error("SHORT_WRITE", "os.write returned non-positive count")
                written += n
            os.fchmod(tmp_fd, 0o600)
            os.fsync(tmp_fd)
            st_tmp = os.fstat(tmp_fd)
            if not stat.S_ISREG(st_tmp.st_mode) or st_tmp.st_nlink != 1:
                raise oracle.S79Error("TEMP_TAMPERED", "temp file mode or link count altered")
            os.close(tmp_fd)
            tmp_fd = None

            try:
                lst2 = os.lstat(fname, dir_fd=dfd)
            except FileNotFoundError:
                lst2 = None
            if lst2 is not None:
                if stat.S_ISLNK(lst2.st_mode):
                    raise oracle.S79Error("SYMLINK_FORBIDDEN", f"symlink appeared at target: {fname!r}")
                if not stat.S_ISREG(lst2.st_mode):
                    raise oracle.S79Error("NON_REGULAR_FILE", f"non-regular target appeared: {fname!r}")
                if lst2.st_nlink > 1:
                    raise oracle.S79Error("HARDLINK_FORBIDDEN", f"hardlink appeared at target: {fname!r}")

            os.replace(tmp_name, fname, src_dir_fd=dfd, dst_dir_fd=dfd)
            tmp_name = None
            os.fsync(dfd)
        except oracle.S79Error:
            raise
        except OSError as exc:
            raise oracle.S79Error("WRITE_IO_ERROR", f"atomic write failed: {exc}") from exc
    finally:
        if tmp_fd is not None:
            os.close(tmp_fd)
        if tmp_name is not None:
            try:
                os.unlink(tmp_name, dir_fd=dfd)
            except OSError:
                pass
        os.close(dfd)


# ---------------------------------------------------------------------------
# Strict-Keyset Field-Complete Secondary Auditor & Certificate Verifier
# ---------------------------------------------------------------------------

def _require_dict_exact_keys(obj: Any, expected: frozenset[str], name: str) -> dict[str, Any]:
    if type(obj) is not dict:
        raise oracle.S79Error("INVALID_OBJECT", f"{name} must be a dict")
    if frozenset(obj.keys()) != expected:
        raise oracle.S79Error("KEYSET_MISMATCH", f"{name} keys do not match schema")
    return obj


def _require_exact_int_list(obj: Any, expected: list[int], name: str) -> None:
    if type(obj) is not list or [oracle.require_exact_int(x, name) for x in obj] != expected:
        raise oracle.S79Error("INT_LIST_MISMATCH", f"{name} integer list mismatch")


def _require_exact_str_list(obj: Any, expected: list[str], name: str) -> None:
    if type(obj) is not list or [oracle.require_exact_str(x, name) for x in obj] != expected:
        raise oracle.S79Error("STR_LIST_MISMATCH", f"{name} string list mismatch")


def secondary_audit_payload(payload: dict[str, Any]) -> None:
    """Field-complete strict-keyset independent algebraic cross-check (zero cached summary trust)."""
    oracle.check_json_value_strict(payload)
    p = _require_dict_exact_keys(payload, _PAYLOAD_KEYS, "payload")
    if p["project_id"] != oracle.PROJECT_ID:
        raise oracle.S79Error("PROJECT_ID_MISMATCH", "payload project_id mismatch")

    # Pillar 1: 2DSD Cascade, Speedup Dominance & Eq. 6 KV Sharing
    p1 = _require_dict_exact_keys(p["pillar1_2dsd_cascade_and_kv_laws"], _P1_KEYS, "pillar1")
    _require_exact_int_list(p1["cascade_target_mass"], [60, 30, 10], "cascade_target_mass")
    _require_exact_int_list(p1["cascade_stage_cum_accepted"], [80, 90, 100], "cascade_stage_cum_accepted")
    if (
        oracle.require_exact_int(p1["cascade_final_residual_mass"], "res_mass") != 0
        or oracle.require_exact_bool(p1["cascade_order_invariance_verified"], "ord_inv") is not True
        or oracle.parse_exact_rational(p1["ouro14b_greedy_2dsd_over_dflash_q"], "o14_gd") != Fraction(573, 479)
        or oracle.parse_exact_rational(p1["ouro26b_greedy_2dsd_over_dflash_q"], "o26_gd") != Fraction(764, 470)
        or oracle.parse_exact_rational(p1["ouro26b_greedy_2dsd_over_loopspec_q"], "o26_gl") != Fraction(764, 252)
        or oracle.parse_exact_rational(p1["dory4b_greedy_2dsd_over_dflash_q"], "d4_gd") != Fraction(450, 300)
        or oracle.parse_exact_rational(p1["ouro26b_sampling_2dsd_over_dflash_q"], "o26_sd") != Fraction(553, 381)
        or oracle.parse_exact_rational(p1["dory4b_sampling_2dsd_over_dflash_q"], "d4_sd") != Fraction(358, 268)
        or oracle.parse_exact_rational(p1["kv_ratio_r4_w64_l2048_q"], "kv_2k") != Fraction(4 * 2048, 2048 + 3 * 64)
        or oracle.parse_exact_rational(p1["kv_ratio_r4_w64_l8192_q"], "kv_8k") != Fraction(4 * 8192, 8192 + 3 * 64)
        or oracle.parse_exact_rational(p1["kv_ratio_r4_w64_l32768_q"], "kv_32k") != Fraction(4 * 32768, 32768 + 3 * 64)
        or oracle.parse_exact_rational(p1["kv_ratio_r2_w64_l32768_q"], "kv_r2") != Fraction(2 * 32768, 32768 + 64)
        or oracle.parse_exact_rational(p1["ouro14b_hard32k_tf_delta_pct_q"], "o14_tf") != Fraction(4549 - 5095, 100)
        or oracle.parse_exact_rational(p1["ouro26b_hard32k_tf_delta_pct_q"], "o26_tf") != Fraction(5136 - 5849, 100)
        or oracle.parse_exact_rational(p1["dory4b_hard32k_tf_delta_pct_q"], "d4_tf") != Fraction(7706 - 7634, 100)
        or oracle.parse_exact_rational(p1["circular_ep_rho3_4_loop4_decay_q"], "cep_r4") != Fraction(81, 256)
        or oracle.require_exact_int(p1["inkling2_mini_varloop_drafter_params"], "ink_p") != 541_880_065
    ):
        raise oracle.S79Error("P1_2DSD_MISMATCH", "Pillar 1 2DSD/KV values mismatch")

    # Pillar 2: Dense-and-Sparse vs. siliquant BlockH4 QuadSnake
    p2 = _require_dict_exact_keys(p["pillar2_dense_sparse_vs_siliquant"], _P2_KEYS, "pillar2")
    if (
        oracle.parse_exact_rational(p2["squeezellm_csr_delta_b_q"], "sq_db") != Fraction(45 * 32, 10000)
        or oracle.parse_exact_rational(p2["squeezellm_csr_b_eff_q"], "sq_be") != Fraction(4 * 10000 + 45 * 32, 10000)
        or oracle.parse_exact_rational(p2["squeezellm_csr_over_dense_ratio_q"], "sq_rat") != Fraction(5 * 1000, 4 * 1400)
        or oracle.require_exact_bool(p2["squeezellm_csr_wins"], "sq_w") is not True
        or oracle.parse_exact_rational(p2["portal77_sidecar_delta_b_q"], "p77_db") != Fraction(57 * 48, 100000)
        or oracle.parse_exact_rational(p2["portal77_sidecar_over_dense_ratio_q"], "p77_rat") != Fraction(104 * 1000, 100 * 1400)
        or oracle.require_exact_bool(p2["portal77_sidecar_wins"], "p77_w") is not True
        or oracle.parse_exact_rational(p2["reversal_rho_1_32_delta_b_q"], "rev_db") != Fraction(1, 1)
        or oracle.parse_exact_rational(p2["reversal_csr_over_dense_ratio_q"], "rev_rat") != Fraction(4 * 1000, 3000)
        or oracle.require_exact_bool(p2["reversal_pure_dense_wins"], "rev_w") is not True
        or oracle.require_exact_int(p2["s72_qwen3b_snake_nkl"], "s72_snk") != 23_154_136
        or oracle.require_exact_int(p2["s72_qwen3b_h2_nkl"], "s72_h2") != 24_178_790
        or oracle.require_exact_int(p2["s72_qwen3b_h4_nkl"], "s72_h4") != 24_774_488
        or oracle.parse_exact_rational(p2["s72_qwen3b_snake_vs_h2_rel_q"], "s72_sh2") != Fraction(23_154_136 - 24_178_790, 24_178_790)
        or oracle.parse_exact_rational(p2["s72_qwen3b_snake_vs_h4_rel_q"], "s72_sh4") != Fraction(23_154_136 - 24_774_488, 24_774_488)
        or oracle.parse_exact_rational(p2["safe_csr_sparsity_cap_at_32b_sOut40pct_q"], "safe_rho") != Fraction(1, 160)
        or oracle.require_exact_int(p2["safe_cap_pow5_lhs"], "cap_l") != 12_500
        or oracle.require_exact_int(p2["safe_cap_pow5_rhs"], "cap_r") != 16_807
    ):
        raise oracle.S79Error("P2_DENSE_SPARSE_MISMATCH", "Pillar 2 Dense-and-Sparse values mismatch")

    # Pillar 3: MultipoleAttention Centroid Error, Jensen Bias & Covariance Residual
    p3 = _require_dict_exact_keys(p["pillar3_multipole_error_and_barrier"], _P3_KEYS, "pillar3")
    if (
        oracle.require_exact_int(p3["kernel_block_tokens"], "blk_tok") != 8192
        or oracle.require_exact_int(p3["witness_q_norm_sq"], "q_nsq") != 25
        or oracle.require_exact_int(p3["witness_k_diff_norm_sq"], "kd_nsq") != 8
        or oracle.require_exact_int(p3["witness_cs_lhs"], "cs_l") != 36
        or oracle.require_exact_int(p3["witness_cs_rhs"], "cs_r") != 200
        or oracle.require_exact_int(p3["witness_centroid_linear_residual"], "lin_res") != 0
        or oracle.parse_exact_rational(p3["witness_actual_logit_spread_sq_q"], "act_ls") != Fraction(36, 512)
        or oracle.parse_exact_rational(p3["witness_cs_logit_bound_sq_q"], "cs_lb") != Fraction(200, 512)
        or oracle.parse_exact_rational(p3["jensen_d4_eps1_mass_ratio_q"], "jen_r") != Fraction(33, 32)
        or oracle.parse_exact_rational(p3["softmax_bracket_lower_q"], "sm_lo") != Fraction(12, 37)
        or oracle.parse_exact_rational(p3["softmax_bracket_exact_q"], "sm_ex") != Fraction(3, 8)
        or oracle.parse_exact_rational(p3["softmax_bracket_upper_q"], "sm_hi") != Fraction(3, 7)
        or oracle.parse_exact_rational(p3["barrier_dk128_q8_delta1_eps_sq_q"], "bar_ok") != Fraction(1, 2)
        or oracle.parse_exact_rational(p3["barrier_dk128_q8_delta2_eps_sq_q"], "bar_hi") != Fraction(2, 1)
        or oracle.require_exact_int(p3["witness_value_weight_cov_residual"], "cov_res") != 8
    ):
        raise oracle.S79Error("P3_MULTIPOLE_MISMATCH", "Pillar 3 MultipoleAttention values mismatch")

    # Pillar 4: Formal Lean 4 Inventory & Honest Scope
    p4 = _require_dict_exact_keys(p["pillar4_formal_inventory_and_scope"], _P4_KEYS, "pillar4")
    total_decl, univ_decl, lit_decl = oracle.declaration_counts()
    if (
        p4["lean_toolchain"] != oracle.LEAN_TOOLCHAIN_TEXT
        or not oracle.check_digest_eq(p4["lean_toolchain_sha256"], oracle.LEAN_TOOLCHAIN_SHA256)
        or not oracle.check_digest_eq(p4["lakefile_sha256"], oracle.LAKEFILE_SHA256)
        or oracle.require_exact_int(p4["module_count"], "mod_cnt") != 1
        or oracle.require_exact_int(p4["theorem_count"], "thm_cnt") != total_decl
        or oracle.require_exact_int(p4["zero_axiom_theorem_count"], "zero_ax") != 14
        or oracle.require_exact_int(p4["propext_only_theorem_count"], "propext_ax") != 4
        or oracle.require_exact_int(p4["universal_theorem_count"], "univ_cnt") != univ_decl
        or oracle.require_exact_int(p4["literal_ordering_check_count"], "lit_cnt") != lit_decl
        or oracle.require_exact_int(p4["universal_theorem_count"], "univ_cnt")
        + oracle.require_exact_int(p4["literal_ordering_check_count"], "lit_cnt")
        != total_decl
    ):
        raise oracle.S79Error("P4_HEADER_MISMATCH", "Pillar 4 formal header pins mismatch")
    _require_exact_str_list(
        p4["literal_ordering_checks"], list(oracle.LITERAL_ORDERING_WITNESSES), "p4.literal_ordering_checks"
    )
    mods = p4["modules"]
    if type(mods) is not list or len(mods) != len(oracle.LEAN_MODULE_SPECS):
        raise oracle.S79Error("P4_MODULES_SHAPE", "modules list length mismatch")
    for idx, spec in enumerate(oracle.LEAN_MODULE_SPECS):
        mrow = _require_dict_exact_keys(mods[idx], _MOD_ROW_KEYS, f"modules[{idx}]")
        if (
            mrow["relpath"] != spec["relpath"]
            or mrow["namespace"] != spec["namespace"]
            or not oracle.check_digest_eq(mrow["sha256"], spec["sha256"])
            or oracle.require_exact_int(mrow["theorem_count"], "m_thm_cnt") != len(spec["theorems"])
        ):
            raise oracle.S79Error("P4_MOD_META", f"modules[{idx}] metadata mismatch")
        exp_tmap = {n: list(axs) for n, axs in spec["theorems"]}
        got_tmap = _require_dict_exact_keys(mrow["theorems"], frozenset(exp_tmap.keys()), f"modules[{idx}].theorems")
        for tname, exp_axs in exp_tmap.items():
            _require_exact_str_list(got_tmap[tname], exp_axs, f"modules[{idx}].{tname}")

    commits = _require_dict_exact_keys(
        p4["primary_source_commits"], frozenset(oracle.PRIMARY_SOURCE_COMMITS.keys()), "primary_source_commits"
    )
    for ck, exp_c in oracle.PRIMARY_SOURCE_COMMITS.items():
        if not oracle.check_commit_eq(commits[ck], exp_c):
            raise oracle.S79Error("P4_COMMIT_MISMATCH", f"primary commit {ck} mismatch")

    shas = _require_dict_exact_keys(
        p4["primary_source_sha256"], frozenset(oracle.PRIMARY_SOURCE_SHA256.keys()), "primary_source_sha256"
    )
    for sk, exp_sha in oracle.PRIMARY_SOURCE_SHA256.items():
        if not oracle.check_digest_eq(shas[sk], exp_sha):
            raise oracle.S79Error("P4_SHA_MISMATCH", f"primary snapshot {sk} mismatch")

    if (
        p4["novelty_tier"] != "N1"
        or p4["oracle_class"] != "exact"
        or oracle.require_exact_int(p4["max_int_bits_exact_cap"], "max_bits") != oracle.MAX_INT_BITS
        or oracle.require_exact_int(p4["max_int_digits_preparse_ceiling"], "max_digits") != oracle.MAX_INT_DIGITS
        or oracle.require_exact_bool(p4["measured_live_gpu_in_this_bundle"], "live_gpu") is not False
        or oracle.require_exact_bool(p4["zero_mathlib_imports"], "zero_mathlib") is not True
        or oracle.require_exact_bool(p4["zero_trust_path_floats"], "zero_floats") is not True
        or oracle.require_exact_bool(p4["zero_bare_asserts"], "zero_asserts") is not True
        or oracle.require_exact_bool(p4["zero_mathematical_rng"], "zero_rng") is not True
    ):
        raise oracle.S79Error("P4_SCOPE_MISMATCH", "Pillar 4 honest scope flags mismatch")
    _require_exact_str_list(p4["prior_literature_n0"], list(oracle.PRIOR_LITERATURE_N0), "prior_literature_n0")

    summ = _require_dict_exact_keys(p["summary"], _SUMMARY_KEYS, "summary")
    if (
        oracle.require_exact_bool(summ["all_pillars_verified"], "all_pillars_verified") is not True
        or oracle.require_exact_int(summ["kernel_checked_declaration_count"], "kcd") != total_decl
        or oracle.require_exact_int(summ["universal_theorem_count"], "univ") != univ_decl
        or oracle.require_exact_int(summ["literal_ordering_check_count"], "lit") != lit_decl
        or oracle.require_exact_int(summ["zero_axiom_declaration_count"], "zero_ax") != 14
        or oracle.require_exact_int(summ["propext_only_declaration_count"], "propext_ax") != 4
        or summ["declaration_kinds_note"] != oracle.declaration_kinds_note()
        or oracle.require_exact_int(summ["target_max_cert_bytes"], "target_max_cert_bytes") != oracle.TARGET_MAX_CERT_BYTES
    ):
        raise oracle.S79Error("SUMMARY_MISMATCH", "summary fields mismatch")
    _require_exact_str_list(
        summ["literal_ordering_checks"], list(oracle.LITERAL_ORDERING_WITNESSES), "summary.literal_ordering_checks"
    )


def verify_certificate_dict(cert: Any, *, full_replay: bool = True) -> dict[str, Any]:
    """Verify certificate schema, oracle-module pin, constant-time digests, secondary audit, and optional full replay."""
    replay_flag = oracle.require_exact_bool(full_replay, "full_replay")
    oracle.check_json_value_strict(cert)
    c = _require_dict_exact_keys(cert, _CERT_KEYS, "certificate")
    if c["schema_version"] != oracle.SCHEMA_VERSION:
        raise oracle.S79Error("SCHEMA_MISMATCH", "unexpected schema_version")
    if c["project_id"] != oracle.PROJECT_ID:
        raise oracle.S79Error("PROJECT_ID_MISMATCH", "unexpected project_id")
    decl_o_sha = oracle.require_sha256_hex(c["oracle_module_sha256"], "oracle_module_sha256")
    decl_p_sha = oracle.require_sha256_hex(c["payload_sha256"], "payload_sha256")
    decl_c_sha = oracle.require_sha256_hex(c["certificate_sha256"], "certificate_sha256")
    payload = _require_dict_exact_keys(c["payload"], _PAYLOAD_KEYS, "payload")

    actual_p_sha = oracle.sha256_bytes(oracle.canonical_json_bytes(payload))
    if not oracle.check_digest_eq(decl_p_sha, actual_p_sha):
        raise oracle.S79Error("PAYLOAD_SHA256_MISMATCH", "payload digest mismatch")
    cert_pre = {
        "schema_version": c["schema_version"],
        "project_id": c["project_id"],
        "oracle_module_sha256": decl_o_sha,
        "payload_sha256": decl_p_sha,
        "payload": payload,
    }
    actual_c_sha = oracle.sha256_bytes(oracle.canonical_json_bytes(cert_pre))
    if not oracle.check_digest_eq(decl_c_sha, actual_c_sha):
        raise oracle.S79Error("CERTIFICATE_SHA256_MISMATCH", "certificate digest mismatch")

    # Oracle-module pin: fail closed BEFORE any replay if the on-disk oracle module
    # differs from the module the certificate was built from, so `full_replay` never
    # re-derives the expected certificate from an edited (self-referential) module.
    actual_o_sha = oracle.sha256_bytes(read_regular_bytes_capped(oracle.__file__, max_bytes=MAX_JSON_BYTES))
    if not oracle.check_digest_eq(decl_o_sha, actual_o_sha):
        raise oracle.S79Error(
            "ORACLE_SHA256_MISMATCH",
            "s79_2dsd_multipole_oracle.py SHA-256 does not match the certificate pin; refusing replay",
        )

    secondary_audit_payload(payload)

    if replay_flag:
        expected = oracle.build_certificate()
        if not oracle.check_digest_eq(decl_c_sha, expected["certificate_sha256"]):
            raise oracle.S79Error("FULL_REPLAY_MISMATCH", "certificate disagrees with deterministic replay")

    return {
        "valid": True,
        "schema_version": oracle.SCHEMA_VERSION,
        "project_id": oracle.PROJECT_ID,
        "payload_sha256": actual_p_sha,
        "certificate_sha256": actual_c_sha,
        "full_replay": replay_flag,
        "secondary_audit_passed": True,
    }


# ---------------------------------------------------------------------------
# Live Lean 4.36.0 Kernel + Strict `#print axioms` + Warning-Free Verifier
# ---------------------------------------------------------------------------

def _verify_lean_stdout_axioms(
    stdout_text: str,
    namespace: str,
    expected_theorems: tuple[tuple[str, tuple[str, ...]], ...],
) -> None:
    """Strictly verify `#print axioms` stdout: zero compiler warnings, zero stray lines, exact axiom sets."""
    seen: dict[str, tuple[str, ...]] = {}
    spans: list[tuple[int, int]] = []
    for m_ax in _AXIOM_BLOCK_RE.finditer(stdout_text):
        spans.append((m_ax.start(), m_ax.end()))
        fq_name = m_ax.group(1)
        if fq_name in seen:
            raise oracle.S79Error("LEAN_DUPLICATE_AXIOM_LINE", f"duplicate axiom output for {fq_name!r}")
        if m_ax.group(2) is not None:
            axs: tuple[str, ...] = ()
        else:
            raw_toks = [tok.strip() for tok in m_ax.group(3).split(",") if tok.strip()]
            for ax in raw_toks:
                if ax not in STANDARD_LEAN4_AXIOMS:
                    raise oracle.S79Error("LEAN_BANNED_AXIOM", f"disallowed axiom {ax!r} in {fq_name}")
            axs = tuple(sorted(raw_toks))
        seen[fq_name] = axs

    cursor = 0
    leftover_parts: list[str] = []
    for s_lo, s_hi in spans:
        leftover_parts.append(stdout_text[cursor:s_lo])
        cursor = s_hi
    leftover_parts.append(stdout_text[cursor:])
    if "".join(leftover_parts).strip():
        raise oracle.S79Error("LEAN_UNEXPECTED_STDOUT", "unexpected non-axiom output or warning on lean stdout")

    expected_map = {f"{namespace}.{tname}": exp_axs for tname, exp_axs in expected_theorems}
    if frozenset(seen.keys()) != frozenset(expected_map.keys()):
        raise oracle.S79Error("LEAN_THEOREM_SET_MISMATCH", "printed theorem set does not match pinned manifest")
    for fq_name, exp_axs in expected_map.items():
        if seen[fq_name] != exp_axs:
            raise oracle.S79Error("LEAN_AXIOM_MISMATCH", f"axiom set mismatch for {fq_name!r}")


def verify_lean_modules_live(
    project_dir: Any,
    *,
    lean_bin: str | None = None,
    verify_live_lean: bool = True,
) -> dict[str, Any]:
    """Verify lean-toolchain, lakefile.toml, and TwoDSDAndDenseSparseLaws.lean (SHA-256 + R9/R10 + live Lean)."""
    live_flag = oracle.require_exact_bool(verify_live_lean, "verify_live_lean")
    root = _coerce_exact_path(project_dir, "project_dir")

    tc_bytes = read_regular_bytes_capped(root / "lean-toolchain", max_bytes=4096)
    if not oracle.check_digest_eq(oracle.sha256_bytes(tc_bytes), oracle.LEAN_TOOLCHAIN_SHA256):
        raise oracle.S79Error("LEAN_TOOLCHAIN_SHA_MISMATCH", "lean-toolchain SHA-256 mismatch")

    lf_bytes = read_regular_bytes_capped(root / "lakefile.toml", max_bytes=16384)
    if not oracle.check_digest_eq(oracle.sha256_bytes(lf_bytes), oracle.LAKEFILE_SHA256):
        raise oracle.S79Error("LAKEFILE_SHA_MISMATCH", "lakefile.toml SHA-256 mismatch")

    resolved_lean: str | None = None
    if live_flag:
        if lean_bin is not None:
            resolved_lean = oracle.require_exact_str(lean_bin, "lean_bin")
        else:
            env_lean = os.environ.get("LEAN_BIN")
            resolved_lean = env_lean if env_lean else shutil.which("lean")
        if resolved_lean is None:
            raise oracle.S79Error("LEAN_NOT_FOUND", "lean binary not found for live kernel verification")

    verified_modules: list[str] = []
    total_theorems = 0
    for spec in oracle.LEAN_MODULE_SPECS:
        mod_path = root / spec["relpath"]
        raw = read_regular_bytes_capped(mod_path, max_bytes=MAX_JSON_BYTES)
        file_sha = oracle.sha256_bytes(raw)
        if not oracle.check_digest_eq(file_sha, spec["sha256"]):
            raise oracle.S79Error("LEAN_SHA256_MISMATCH", f"{spec['relpath']} SHA-256 mismatch")
        try:
            src = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise oracle.S79Error("LEAN_INVALID_UTF8", f"{spec['relpath']} is not valid UTF-8") from exc
        scan = audit_lean4_source(src)
        if not scan["clean"] or scan["notices"]:
            raise oracle.S79Error("LEAN_STATIC_AUDIT_FAILED", f"{spec['relpath']} static audit failed")
        if live_flag:
            try:
                proc = subprocess.run(
                    [resolved_lean, str(mod_path)],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise oracle.S79Error("LEAN_EXEC_FAILED", f"cannot invoke lean: {exc}") from exc
            if proc.returncode != 0:
                raise oracle.S79Error("LEAN_KERNEL_FAILED", f"lean exited {proc.returncode} on {spec['relpath']}")
            # Note: Lean 4.36 prints compiler warnings on stdout (stderr stays empty),
            # so the LEAN_UNEXPECTED_STDOUT check in _verify_lean_stdout_axioms is what
            # actually catches warnings; LEAN_STDERR_NONEMPTY alone would not.
            if proc.stderr.strip():
                raise oracle.S79Error("LEAN_STDERR_NONEMPTY", f"lean produced stderr on {spec['relpath']}")
            _verify_lean_stdout_axioms(proc.stdout, spec["namespace"], spec["theorems"])
        verified_modules.append(spec["relpath"])
        total_theorems += len(spec["theorems"])

    return {
        "valid": True,
        "modules_verified": verified_modules,
        "kernel_theorems_verified": total_theorems,
        "live_lean_checked": live_flag,
        "static_r9_r10_clean": True,
        "banned_axioms_found": 0,
    }


# ---------------------------------------------------------------------------
# Strict Manual CLI Parser (Zero Abbreviation, Fail-Closed rc=2)
# ---------------------------------------------------------------------------

def _parse_cli_args(argv: list[str]) -> dict[str, Any]:
    mode = "self_check"
    target_path: str | None = None
    as_json = False
    i = 0
    n = len(argv)
    while i < n:
        arg = argv[i]
        if arg == "--standalone":
            if mode != "self_check":
                raise oracle.S79Error("CLI_CONFLICT", "conflicting CLI mode flags")
            mode = "standalone"
            i += 1
        elif arg == "--json":
            if as_json:
                raise oracle.S79Error("CLI_DUPLICATE", "duplicate --json flag")
            as_json = True
            i += 1
        elif arg in ("--write", "--check", "--verify-lean"):
            if mode != "self_check":
                raise oracle.S79Error("CLI_CONFLICT", "conflicting CLI mode flags")
            if i + 1 >= n:
                raise oracle.S79Error("CLI_MISSING_ARG", f"{arg} requires a path argument")
            cand = argv[i + 1]
            if not cand or cand.startswith("-"):
                raise oracle.S79Error("CLI_INVALID_PATH", f"{arg} path must not be empty or flag-like")
            mode = arg[2:].replace("-", "_")
            target_path = cand
            i += 2
        else:
            raise oracle.S79Error("CLI_UNKNOWN_ARG", f"unknown argument {arg!r}")
    return {"mode": mode, "path": target_path, "json": as_json}


def main(argv: list[str] | tuple[str, ...] | None = None) -> int:
    try:
        if argv is None:
            raw_argv = list(sys.argv[1:])
        else:
            if type(argv) not in (list, tuple):
                raise oracle.S79Error("CLI_INVALID_ARGV", "argv must be list, tuple, or None")
            raw_argv = [oracle.require_exact_str(a, "argv", non_empty=False) for a in argv]
        opts = _parse_cli_args(raw_argv)
        mode = opts["mode"]
        if mode == "self_check":
            cert = oracle.build_certificate()
            res = verify_certificate_dict(cert, full_replay=True)
            if opts["json"]:
                sys.stdout.write(oracle.canonical_json_bytes(res).decode("ascii") + "\n")
            else:
                sys.stdout.write(f"OK {res['schema_version']} {res['certificate_sha256']}\n")
            return 0
        if mode == "standalone":
            cert = oracle.build_certificate()
            verify_certificate_dict(cert, full_replay=True)
            sys.stdout.write(oracle.canonical_json_bytes(cert).decode("ascii") + "\n")
            return 0
        if mode == "write":
            cert = oracle.build_certificate()
            verify_certificate_dict(cert, full_replay=True)
            out_bytes = oracle.canonical_json_bytes(cert) + b"\n"
            write_bytes_atomic_0600(opts["path"], out_bytes)
            if opts["json"]:
                sys.stdout.write(
                    oracle.canonical_json_bytes(
                        {"written": True, "certificate_sha256": cert["certificate_sha256"]}
                    ).decode("ascii")
                    + "\n"
                )
            else:
                sys.stdout.write(f"WROTE {cert['certificate_sha256']}\n")
            return 0
        if mode == "check":
            raw_bytes = read_regular_bytes_capped(opts["path"], max_bytes=MAX_JSON_BYTES)
            try:
                raw_text = raw_bytes.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise oracle.S79Error("INVALID_UTF8", "certificate file is not valid UTF-8") from exc
            parsed = oracle.parse_strict_json_text(raw_text)
            canon = oracle.canonical_json_bytes(parsed)
            if raw_bytes not in (canon, canon + b"\n"):
                raise oracle.S79Error("NON_CANONICAL_BYTES", "certificate file is not canonical JSON")
            res = verify_certificate_dict(parsed, full_replay=True)
            if opts["json"]:
                sys.stdout.write(oracle.canonical_json_bytes(res).decode("ascii") + "\n")
            else:
                sys.stdout.write(f"VALID {res['certificate_sha256']}\n")
            return 0
        res = verify_lean_modules_live(opts["path"])
        if opts["json"]:
            sys.stdout.write(oracle.canonical_json_bytes(res).decode("ascii") + "\n")
        else:
            sys.stdout.write(f"LEAN_VALID {len(res['modules_verified'])} {res['kernel_theorems_verified']}\n")
        return 0
    except oracle.S79Error as exc:
        sys.stderr.write(f"ERROR {exc.code}: {exc.message}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
