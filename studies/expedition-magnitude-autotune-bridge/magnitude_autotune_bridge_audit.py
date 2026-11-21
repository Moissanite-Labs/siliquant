#!/usr/bin/env python3
"""Clean-room zero-float audit and cross-pollination oracle for magnitudedev/magnitude.

Bridges four systems in exact rational arithmetic (pure Python 3 stdlib, zero floats):
  1. magnitudedev/magnitude @ d89dfffcbb2f27251e99163934c04770b453339c (read-only target)
  2. Moissanite-Labs/warp-drive (Stage 1-4 runtime flags, SHA-256 lineage, multi-bank flush)
  3. Moissanite-Labs/siliquant#24 (outlier-split Hessian & lattice quantization admissibility)
  4. Moissanite-Labs/hillclimb#4 (bounded autotuning without Goodhart drift)
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

PINNED_REPO = "magnitudedev/magnitude"
PINNED_COMMIT = "d89dfffcbb2f27251e99163934c04770b453339c"
NOVELTY_CLASS = "N1"

STANDARD_ALPHA = Fraction(8, 1)
DEFAULT_ERROR_RATE = Fraction(1, 20)
LINEAGE_DIVERGENCE_LIMIT = Fraction(1, 4000)
MIN_AUTH_TOKEN_BYTES = 16
SHA256_HEX_LEN = 64
HEX_CHARS = frozenset("0123456789abcdef")
LOOPBACK_HOSTS = frozenset(("127.0.0.1", "::1", "localhost"))
ALLOWED_JIT_SYMBOLS = frozenset(
    ("expf", "logf", "sqrtf", "tanhf", "erff", "powf", "sinf", "cosf")
)

IEEE_FORMATS: dict[str, tuple[int, int, int]] = {
    "F32": (23, -126, -149),
    "F16": (10, -14, -24),
    "BF16": (7, -126, -133),
}

ANCHOR_SPECS: tuple[tuple[str, str, int, str], ...] = (
    (
        "MAG-01a",
        "inference/seismic/runtime/src/native/comparison.rs",
        61,
        "pub maximum_absolute_error: f64,",
    ),
    (
        "MAG-01b",
        "inference/engine/executor/src/native/tuning/precision.rs",
        31,
        "const STANDARD_ALPHA: Limit = Limit::from_finite(8.0);",
    ),
    (
        "MAG-01c",
        "inference/seismic/lang/src/lattice.rs",
        448,
        "pub fn width(&self, bits: u32) -> Option<f64> {",
    ),
    (
        "MAG-02",
        "inference/seismic/runtime/src/native/admission.rs",
        60,
        "pub const GOVERNING_STEP: GoverningStep = GoverningStep::Activation;",
    ),
    (
        "MAG-03a",
        "inference/seismic/measure/src/statistics.rs",
        202,
        "pub fn verdict(&self, rate: f64) -> Option<Verdict> {",
    ),
    (
        "MAG-03b",
        "inference/seismic/runtime/src/native/sampling.rs",
        73,
        "M_7(7) = 255 / 8",
    ),
    (
        "MAG-04a",
        "inference/seismic/runtime/src/artifacts/directory.rs",
        110,
        "Store::open(root, Privacy::Shared)",
    ),
    (
        "MAG-04b",
        "inference/engine/executor/src/kernel_cache.rs",
        421,
        ".take(8)",
    ),
    (
        "MAG-04c",
        "inference/seismic/backends/cpu/src/library.rs",
        20,
        "libc::dlsym(libc::RTLD_DEFAULT",
    ),
    (
        "MAG-04d",
        "inference/seismic/backends/cpu/src/toolchain.rs",
        384,
        "std::mem::transmute::<*const u8, LaunchEntry>",
    ),
    (
        "MAG-04e",
        "inference/seismic/runtime/src/formation.rs",
        542,
        "fn artifact(&self, kept: &Pointer) -> Option<T::Artifact> {",
    ),
    (
        "MAG-05a",
        "inference/service/server/src/main.rs",
        80,
        "auth_token: Option<String>,",
    ),
    (
        "MAG-05b",
        "inference/engine/serving/src/sampling.rs",
        82,
        "RandomState::new()",
    ),
    (
        "MAG-06a",
        "inference/service/models/src/weight_lineage.rs",
        39,
        "const LINEAGE_DIVERGENCE: f64 = 2.5e-4;",
    ),
    (
        "MAG-06b",
        "inference/service/models/src/weight_lineage.rs",
        133,
        "Some(1.0 - correlation(values, other))",
    ),
    (
        "MAG-06c",
        "inference/service/models/src/weight_lineage.rs",
        174,
        '.is_some_and(|rest| rest.ends_with("norm") && !rest.contains(\'.\'))',
    ),
    (
        "MAG-07",
        "inference/ggml/src/lib.rs",
        53,
        "Mxfp4 = 39,",
    ),
    (
        "MAG-08a",
        "inference/engine/scheduler/src/prefix_cache.rs",
        85,
        "pub struct PrefixKey {",
    ),
    (
        "MAG-08b",
        "inference/engine/state/src/lib.rs",
        1039,
        "impl Drop for BankClaim {",
    ),
)


def _rat(value: Fraction) -> dict[str, int]:
    return {"num": value.numerator, "den": value.denominator}


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_exact_rat(value: object) -> bool:
    return isinstance(value, Fraction) or _is_int(value)


def _is_nonempty_str(value: object) -> bool:
    return isinstance(value, str) and len(value.strip()) > 0


def _is_sha256_hex(text: object) -> bool:
    return (
        isinstance(text, str)
        and len(text) == SHA256_HEX_LEN
        and all(ch in HEX_CHARS for ch in text)
    )


def ville_mixture_ratio(rounds: int, wins: int) -> Fraction:
    """Exact pure-Q uniform-Beta mixture likelihood ratio M_n(s).

    Matches inference/seismic/measure/src/statistics.rs:27-53 without floats:
      M_n(s) = sum_{j=0}^{s} C(n+1, j) / ((n + 1) * C(n, s)).
    """
    if (
        not _is_int(rounds)
        or not _is_int(wins)
        or rounds < 0
        or wins < 0
        or wins > rounds
    ):
        raise ValueError("rounds and wins must be ints with 0 <= wins <= rounds")
    numerator = sum(math.comb(rounds + 1, j) for j in range(wins + 1))
    denominator = (rounds + 1) * math.comb(rounds, wins)
    return Fraction(numerator, denominator)


def uncorrected_fwer(num_candidates: int, alpha: Fraction) -> Fraction:
    """Exact family-wise Type-I error 1 - (1 - alpha)^K for K null challengers."""
    if (
        not _is_int(num_candidates)
        or not isinstance(alpha, Fraction)
        or num_candidates < 1
        or alpha <= 0
        or alpha >= 1
    ):
        raise ValueError("num_candidates >= 1 and exact Fraction 0 < alpha < 1 required")
    return Fraction(1, 1) - (Fraction(1, 1) - alpha) ** num_candidates


def magnitude_legacy_verdict(ahead: int, behind: int, alpha: Fraction) -> str:
    """Reproduces Magnitude Tally::verdict (statistics.rs:202-210) at unsplit alpha."""
    if (
        not _is_int(ahead)
        or not _is_int(behind)
        or not isinstance(alpha, Fraction)
        or alpha <= 0
        or alpha >= 1
        or ahead < 0
        or behind < 0
    ):
        raise ValueError("invalid legacy verdict inputs")
    rounds = ahead + behind
    threshold = Fraction(1, 1) / alpha
    if ville_mixture_ratio(rounds, ahead) >= threshold:
        return "leads"
    if ville_mixture_ratio(rounds, behind) >= threshold:
        return "trails"
    return "undecided"


def hardened_supermartingale_gate(
    ahead: int,
    behind: int,
    alpha: Fraction,
    num_candidates: int,
    observed_speedup: Fraction,
    min_speedup: Fraction,
    outside_anchor_ok: bool,
) -> dict[str, Any]:
    """Two-sided Bonferroni pure-Q Ville gate + hillclimb#4 outside-anchor ratification."""
    if (
        not _is_int(ahead)
        or not _is_int(behind)
        or not _is_int(num_candidates)
        or not isinstance(alpha, Fraction)
        or not isinstance(observed_speedup, Fraction)
        or not isinstance(min_speedup, Fraction)
        or not isinstance(outside_anchor_ok, bool)
        or alpha <= 0
        or alpha >= 1
        or num_candidates < 1
        or ahead < 0
        or behind < 0
        or min_speedup <= 0
    ):
        raise ValueError("invalid hardened supermartingale gate inputs")
    rounds = ahead + behind
    threshold = Fraction(2 * num_candidates, 1) / alpha
    lead_ratio = ville_mixture_ratio(rounds, ahead)
    trail_ratio = ville_mixture_ratio(rounds, behind)
    if lead_ratio >= threshold:
        verdict = "leads"
    elif trail_ratio >= threshold:
        verdict = "trails"
    else:
        verdict = "undecided"
    promote = (
        verdict == "leads"
        and observed_speedup >= min_speedup
        and outside_anchor_ok
    )
    return {
        "verdict": verdict,
        "promote": promote,
        "threshold": _rat(threshold),
        "lead_ratio": _rat(lead_ratio),
        "trail_ratio": _rat(trail_ratio),
    }


def ieee_rounding_step(value: Fraction, fmt: str) -> Fraction:
    """Exact IEEE-754 RoundingCells::width w_T(value) matching lattice.rs:448-462.

    For finite IEEE values, width = (gap_below + gap_above) / 2. At exact normal
    powers of two 2^e with e > min_normal_exp, gap_below = ulp / 2 and
    gap_above = ulp, so w_T(2^e) = (3/4) * ulp.
    """
    if not isinstance(value, Fraction) or fmt not in IEEE_FORMATS:
        raise ValueError(f"unsupported value or IEEE format: {fmt}")
    mantissa_bits, min_normal_exp, min_subnormal_exp = IEEE_FORMATS[fmt]
    mag = abs(value)
    if mag == 0:
        return Fraction(2, 1) ** min_subnormal_exp
    exp = mag.numerator.bit_length() - mag.denominator.bit_length()
    pow_exp = Fraction(2, 1) ** exp
    if mag < pow_exp:
        exp -= 1
        pow_exp = Fraction(2, 1) ** exp
    if exp < min_normal_exp:
        return Fraction(2, 1) ** min_subnormal_exp
    ulp = Fraction(2, 1) ** (exp - mantissa_bits)
    if mag == pow_exp and exp > min_normal_exp:
        return Fraction(3, 4) * ulp
    return ulp


def _subset_rms_ratio_sq(
    reference: list[Fraction],
    candidate: list[Fraction],
    indices: list[int],
    step_fmt: str,
) -> Fraction:
    if not indices:
        return Fraction(0, 1)
    err_sq = sum((candidate[i] - reference[i]) ** 2 for i in indices)
    step_power = sum(
        ieee_rounding_step(reference[i], step_fmt) ** 2 for i in indices
    ) / Fraction(12, 1)
    return err_sq / step_power


def _median_magnitude(values: list[Fraction]) -> Fraction:
    ordered = sorted(abs(v) for v in values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / Fraction(2, 1)


def magnitude_rms_admission(
    reference: list[Fraction],
    candidate: list[Fraction],
    subject_fmt: str,
    activation_fmt: str | None,
    alpha: Fraction,
    governing_step: str,
) -> dict[str, Any]:
    """Reproduces Magnitude RMS-only kernel admission (admission.rs:47-69, 155-188)."""
    if (
        not reference
        or len(reference) != len(candidate)
        or not all(isinstance(v, Fraction) for v in reference)
        or not all(isinstance(v, Fraction) for v in candidate)
        or not isinstance(alpha, Fraction)
        or alpha <= 0
        or governing_step not in ("activation", "subject")
    ):
        raise ValueError("invalid admission inputs")
    if governing_step == "activation" and activation_fmt is not None:
        step_fmt = activation_fmt
    else:
        step_fmt = subject_fmt
    indices = list(range(len(reference)))
    ratio_sq = _subset_rms_ratio_sq(reference, candidate, indices, step_fmt)
    admitted = ratio_sq <= alpha * alpha
    return {
        "step_fmt": step_fmt,
        "ratio_sq": _rat(ratio_sq),
        "alpha_sq": _rat(alpha * alpha),
        "admitted": admitted,
    }


def hardened_dual_norm_admission(
    reference: list[Fraction],
    candidate: list[Fraction],
    subject_fmt: str,
    activation_fmt: str | None,
    alpha_rms: Fraction,
    max_ulp_cap: Fraction,
    outlier_ratio_threshold: Fraction,
    is_recurrent_state: bool,
) -> dict[str, Any]:
    """Dual-norm (L2 + L_inf) + siliquant#24 outlier-split + recurrent subject-step gate."""
    if (
        not reference
        or len(reference) != len(candidate)
        or not all(isinstance(v, Fraction) for v in reference)
        or not all(isinstance(v, Fraction) for v in candidate)
        or not isinstance(alpha_rms, Fraction)
        or not isinstance(max_ulp_cap, Fraction)
        or not isinstance(outlier_ratio_threshold, Fraction)
        or not isinstance(is_recurrent_state, bool)
        or alpha_rms <= 0
        or max_ulp_cap <= 0
        or outlier_ratio_threshold <= 1
    ):
        raise ValueError("invalid dual-norm admission inputs")
    if is_recurrent_state or activation_fmt is None:
        step_fmt = subject_fmt
    else:
        step_fmt = activation_fmt
    max_ulp = max(
        abs(c - d) / ieee_rounding_step(d, step_fmt)
        for c, d in zip(candidate, reference)
    )
    med_mag = _median_magnitude(reference)
    cutoff = outlier_ratio_threshold * med_mag
    outlier_idx = [i for i, d in enumerate(reference) if abs(d) > cutoff]
    bulk_idx = [i for i, d in enumerate(reference) if abs(d) <= cutoff]
    bulk_ratio_sq = _subset_rms_ratio_sq(reference, candidate, bulk_idx, step_fmt)
    outlier_ratio_sq = _subset_rms_ratio_sq(reference, candidate, outlier_idx, step_fmt)
    alpha_sq = alpha_rms * alpha_rms
    admitted = (
        max_ulp <= max_ulp_cap
        and bulk_ratio_sq <= alpha_sq
        and outlier_ratio_sq <= alpha_sq
    )
    return {
        "step_fmt": step_fmt,
        "outlier_count": len(outlier_idx),
        "bulk_count": len(bulk_idx),
        "max_ulp": _rat(max_ulp),
        "bulk_ratio_sq": _rat(bulk_ratio_sq),
        "outlier_ratio_sq": _rat(outlier_ratio_sq),
        "admitted": admitted,
    }


def pearson_signed_r_squared(
    xs: list[Fraction], ys: list[Fraction]
) -> tuple[int, Fraction]:
    """Exact pure-Q signed squared Pearson correlation (sign(r), r^2) (zero floats)."""
    n = len(xs)
    if (
        n != len(ys)
        or n < 2
        or not all(isinstance(v, Fraction) for v in xs)
        or not all(isinstance(v, Fraction) for v in ys)
    ):
        raise ValueError("pearson_signed_r_squared requires equal Fraction lists of len >= 2")
    n_rat = Fraction(n, 1)
    mx = sum(xs) / n_rat
    my = sum(ys) / n_rat
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        raise ValueError("pearson_signed_r_squared undefined for constant series")
    sign = 1 if cov > 0 else (-1 if cov < 0 else 0)
    return sign, (cov * cov) / (vx * vy)


def weight_sample_matches(
    xs: list[Fraction],
    ys: list[Fraction],
    max_divergence: Fraction = LINEAGE_DIVERGENCE_LIMIT,
) -> bool:
    """Exact pure-Q check of WeightSample signed 1 - r <= max_divergence (weight_lineage.rs:133)."""
    if not isinstance(max_divergence, Fraction) or max_divergence < 0 or max_divergence >= 1:
        raise ValueError("max_divergence must be exact Fraction with 0 <= d < 1")
    sign, r_sq = pearson_signed_r_squared(xs, ys)
    min_r = Fraction(1, 1) - max_divergence
    return sign > 0 and r_sq >= min_r * min_r


def evaluate_trust_boundary(config: dict[str, Any]) -> dict[str, Any]:
    """Warp-drive <-> Magnitude fail-closed execution, cache, lineage, and bank-flush gate."""
    reasons: list[str] = []
    if not isinstance(config["bind_host"], str) or (
        config["bind_host"] not in LOOPBACK_HOSTS
        and (
            not isinstance(config["auth_token"], str)
            or len(config["auth_token"].strip()) < MIN_AUTH_TOKEN_BYTES
        )
    ):
        reasons.append("BIND_UNAUTHENTICATED_NON_LOOPBACK")
    if not isinstance(config["verifier_mode"], bool) or (
        config["verifier_mode"]
        and (
            not _is_exact_rat(config["temperature"])
            or config["temperature"] != 0
            or not _is_exact_rat(config["top_p"])
            or config["top_p"] != 1
            or not _is_int(config["explicit_seed"])
        )
    ):
        reasons.append("SAMPLING_NON_DETERMINISTIC")
    if not isinstance(config["verifier_mode"], bool) or (
        config["verifier_mode"]
        and not _is_sha256_hex(config["tuning_identity_sha256"])
    ):
        reasons.append("TUNING_IDENTITY_UNPINNED")
    if (
        config["cache_privacy"] != "owner"
        or not _is_int(config["cache_mode"])
        or config["cache_mode"] < 0
        or config["cache_mode"] > 0o777
        or (config["cache_mode"] & 0o077) != 0
        or config["cache_o_nofollow"] is not True
    ):
        reasons.append("CACHE_PRIVACY_NOT_OWNER")
    if (
        not _is_sha256_hex(config["evidence_digest_hex"])
        or config["evidence_hash_verified"] is not True
    ):
        reasons.append("EVIDENCE_DIGEST_UNVERIFIED")
    if (
        config["jit_pointer_authenticated"] is not True
        or not isinstance(config["jit_library_symbols"], (list, tuple))
        or any(
            not isinstance(sym, str) or sym not in ALLOWED_JIT_SYMBOLS
            for sym in config["jit_library_symbols"]
        )
    ):
        reasons.append("JIT_ARTIFACT_UNTRUSTED")
    if (
        not _is_sha256_hex(config["checkpoint_sha256"])
        or config["checkpoint_sha256"] != config["expected_checkpoint_sha256"]
    ):
        reasons.append("CHECKPOINT_SHA256_MISMATCH")
    if not isinstance(config["prefix_cache_reused"], bool) or (
        config["prefix_cache_reused"]
        and (
            not _is_nonempty_str(config["request_tenant_id"])
            or not _is_nonempty_str(config["cached_prefix_tenant_id"])
            or config["request_tenant_id"] != config["cached_prefix_tenant_id"]
        )
    ):
        reasons.append("PREFIX_CACHE_TENANT_MISMATCH")
    if (
        not _is_int(config["tainted_rows"])
        or config["tainted_rows"] != 0
        or not isinstance(config["active_banks"], (list, tuple))
        or not isinstance(config["zeroed_banks"], (list, tuple))
        or len(config["active_banks"]) == 0
        or any(not _is_nonempty_str(bank) for bank in config["active_banks"])
        or set(config["zeroed_banks"]) != set(config["active_banks"])
    ):
        reasons.append("STATE_BANKS_NOT_FLUSHED")
    return {"admitted": len(reasons) == 0, "reasons": reasons}


def _read_git_head_sha(repo_root: Path) -> str:
    head_text = (repo_root / ".git" / "HEAD").read_text(encoding="utf-8").strip()
    if head_text.startswith("ref: "):
        ref_path = repo_root / ".git" / head_text[5:]
        return ref_path.read_text(encoding="utf-8").strip()
    return head_text


def inspect_magnitude_repo(repo_root: Path) -> dict[str, Any]:
    """Read-only anchor verification against pinned magnitudedev/magnitude checkout."""
    head_sha = _read_git_head_sha(repo_root)
    anchors: list[dict[str, Any]] = []
    all_matched = head_sha == PINNED_COMMIT
    for finding_id, rel_path, expected_line, needle in ANCHOR_SPECS:
        file_path = repo_root / rel_path
        lines = file_path.read_text(encoding="utf-8").splitlines()
        line_ok = (
            1 <= expected_line <= len(lines)
            and needle in lines[expected_line - 1]
        )
        all_matched = all_matched and line_ok
        anchors.append(
            {
                "id": finding_id,
                "path": rel_path,
                "expected_line": expected_line,
                "matched": line_ok,
            }
        )
    return {
        "anchor_mode": "live_readonly",
        "pinned_commit": PINNED_COMMIT,
        "observed_commit": head_sha,
        "commit_matches": head_sha == PINNED_COMMIT,
        "anchors_verified": sum(1 for item in anchors if item["matched"]),
        "anchors_total": len(anchors),
        "all_anchors_matched": all_matched,
        "anchors": anchors,
    }


def run_audit_suite(repo_root: Path | None) -> dict[str, Any]:
    """Execute all exact zero-float proofs, counterexamples, and hardened bridge checks."""
    # MAG-01a: localized single-channel spike (+147 BF16 rounding cells) passes RMS-only gate
    ref_spike = [Fraction(3, 2)] * 4096
    cand_spike = list(ref_spike)
    cand_spike[0] = Fraction(3, 2) + Fraction(147, 128)
    mag01a_legacy = magnitude_rms_admission(
        ref_spike, cand_spike, "BF16", "BF16", STANDARD_ALPHA, "activation"
    )
    mag01a_hardened = hardened_dual_norm_admission(
        ref_spike,
        cand_spike,
        "BF16",
        "BF16",
        STANDARD_ALPHA,
        Fraction(16, 1),
        Fraction(16, 1),
        False,
    )

    # MAG-01b: single non-power-of-2 outlier (6144, w_BF16=32 exact per lattice.rs:448-462)
    # masks 100% zeroing of all 4095 normal channels (w_BF16(1)=3/512)
    ref_outlier = [Fraction(6144, 1)] + [Fraction(1, 1)] * 4095
    cand_outlier = [Fraction(6144, 1)] + [Fraction(0, 1)] * 4095
    mag01b_legacy = magnitude_rms_admission(
        ref_outlier, cand_outlier, "BF16", "BF16", STANDARD_ALPHA, "activation"
    )
    mag01b_hardened = hardened_dual_norm_admission(
        ref_outlier,
        cand_outlier,
        "BF16",
        "BF16",
        STANDARD_ALPHA,
        Fraction(16, 1),
        Fraction(16, 1),
        False,
    )

    # MAG-02: F32 recurrent DeltaState at 1.0 off by 16384 F32 ULPs (1/512)
    # admitted under BF16 Activation step (ratio^2 = 4/3 < 64), rejected under F32 Subject
    f32_width_at_one = ieee_rounding_step(Fraction(1, 1), "F32")
    bf16_width_at_one = ieee_rounding_step(Fraction(1, 1), "BF16")
    relaxation_factor = bf16_width_at_one / f32_width_at_one
    ref_recurrent = [Fraction(1, 1)] * 8
    cand_recurrent = [Fraction(1, 1) + Fraction(1, 512)] * 8
    mag02_legacy = magnitude_rms_admission(
        ref_recurrent, cand_recurrent, "F32", "BF16", STANDARD_ALPHA, "activation"
    )
    mag02_hardened = hardened_dual_norm_admission(
        ref_recurrent,
        cand_recurrent,
        "F32",
        "BF16",
        STANDARD_ALPHA,
        Fraction(16, 1),
        Fraction(16, 1),
        True,
    )

    # MAG-03: Ville supermartingale exact rational ladder & multiplicity inflation
    m6_6 = ville_mixture_ratio(6, 6)
    m7_7 = ville_mixture_ratio(7, 7)
    m8_8 = ville_mixture_ratio(8, 8)
    martingale_lhs = (
        ville_mixture_ratio(6, 3) + ville_mixture_ratio(6, 4)
    ) / Fraction(2, 1)
    martingale_rhs = ville_mixture_ratio(5, 3)
    fwer_20 = uncorrected_fwer(20, DEFAULT_ERROR_RATE)
    mag03_legacy_7 = magnitude_legacy_verdict(7, 0, DEFAULT_ERROR_RATE)
    mag03_hardened_7 = hardened_supermartingale_gate(
        7, 0, DEFAULT_ERROR_RATE, 1, Fraction(1, 20), Fraction(1, 50), True
    )
    mag03_hardened_8_anchor_fail = hardened_supermartingale_gate(
        8, 0, DEFAULT_ERROR_RATE, 1, Fraction(1, 20), Fraction(1, 50), False
    )
    mag03_hardened_8_pass = hardened_supermartingale_gate(
        8, 0, DEFAULT_ERROR_RATE, 1, Fraction(1, 20), Fraction(1, 50), True
    )

    # MAG-06: WeightSample signed Pearson (1D *norm.weight in blk.0/blk.1 + output_norm)
    # blind to blk.2..blk.31 + all 2D projections & positive affine rescaling (while rejecting -x)
    sampled_base = [Fraction(i, 10) for i in range(1, 17)]
    sampled_pos_affine = [Fraction(5, 1) * x + Fraction(3, 1) for x in sampled_base]
    sampled_neg_affine = [-x for x in sampled_base]
    pos_sign, pos_r_sq = pearson_signed_r_squared(sampled_base, sampled_pos_affine)
    pos_affine_matches = weight_sample_matches(sampled_base, sampled_pos_affine)
    neg_affine_matches = weight_sample_matches(sampled_base, sampled_neg_affine)
    base_sha = hashlib.sha256(b"blk0..blk31-original").hexdigest()
    tampered_sha = hashlib.sha256(b"blk0..blk31-last-layer-tampered").hexdigest()

    # MAG-04/05/06/08: Trust-boundary contrast (legacy permissive vs warp-drive hardened)
    legacy_risky_config = {
        "bind_host": "0.0.0.0",
        "auth_token": "",
        "verifier_mode": True,
        "temperature": Fraction(1, 1),
        "top_p": Fraction(19, 20),
        "explicit_seed": None,
        "tuning_identity_sha256": "unpinned",
        "cache_privacy": "shared",
        "cache_mode": 0o755,
        "cache_o_nofollow": False,
        "evidence_digest_hex": "0123456789abcdef",
        "evidence_hash_verified": False,
        "jit_pointer_authenticated": False,
        "jit_library_symbols": ["expf", "system"],
        "checkpoint_sha256": tampered_sha,
        "expected_checkpoint_sha256": base_sha,
        "prefix_cache_reused": True,
        "request_tenant_id": "tenant-b",
        "cached_prefix_tenant_id": "tenant-a",
        "tainted_rows": 64,
        "active_banks": ["ConvWindow", "DeltaState", "DeltaTape", "SsmState", "KVPages"],
        "zeroed_banks": ["ConvWindow"],
    }
    hardened_config = {
        "bind_host": "127.0.0.1",
        "auth_token": "0123456789abcdef",
        "verifier_mode": True,
        "temperature": Fraction(0, 1),
        "top_p": Fraction(1, 1),
        "explicit_seed": 42,
        "tuning_identity_sha256": base_sha,
        "cache_privacy": "owner",
        "cache_mode": 0o700,
        "cache_o_nofollow": True,
        "evidence_digest_hex": base_sha,
        "evidence_hash_verified": True,
        "jit_pointer_authenticated": True,
        "jit_library_symbols": ["expf", "sqrtf"],
        "checkpoint_sha256": base_sha,
        "expected_checkpoint_sha256": base_sha,
        "prefix_cache_reused": True,
        "request_tenant_id": "tenant-a",
        "cached_prefix_tenant_id": "tenant-a",
        "tainted_rows": 0,
        "active_banks": ["ConvWindow", "DeltaState", "DeltaTape", "SsmState", "KVPages"],
        "zeroed_banks": ["ConvWindow", "DeltaState", "DeltaTape", "SsmState", "KVPages"],
    }
    trust_legacy = evaluate_trust_boundary(legacy_risky_config)
    trust_hardened = evaluate_trust_boundary(hardened_config)

    if repo_root is None:
        anchor_report: dict[str, Any] = {
            "anchor_mode": "standalone",
            "pinned_commit": PINNED_COMMIT,
            "anchors_verified": 0,
            "anchors_total": len(ANCHOR_SPECS),
            "all_anchors_matched": False,
        }
    else:
        anchor_report = inspect_magnitude_repo(repo_root)

    payload: dict[str, Any] = {
        "schema": "expedition.round7.magnitude_autotune_bridge.v1",
        "target_repo": PINNED_REPO,
        "pinned_commit": PINNED_COMMIT,
        "novelty_class": NOVELTY_CLASS,
        "zero_float_arithmetic": True,
        "anchors": anchor_report,
        "proofs": {
            "mag01a_localized_spike": {
                "legacy_admitted": mag01a_legacy["admitted"],
                "legacy_ratio_sq": mag01a_legacy["ratio_sq"],
                "hardened_admitted": mag01a_hardened["admitted"],
                "hardened_max_ulp": mag01a_hardened["max_ulp"],
            },
            "mag01b_outlier_masking": {
                "outlier_value": 6144,
                "legacy_admitted": mag01b_legacy["admitted"],
                "legacy_ratio_sq": mag01b_legacy["ratio_sq"],
                "hardened_admitted": mag01b_hardened["admitted"],
                "hardened_outlier_count": mag01b_hardened["outlier_count"],
                "hardened_bulk_ratio_sq": mag01b_hardened["bulk_ratio_sq"],
            },
            "mag02_recurrent_step_relaxation": {
                "bf16_over_f32_step_factor": _rat(relaxation_factor),
                "legacy_admitted": mag02_legacy["admitted"],
                "legacy_ratio_sq": mag02_legacy["ratio_sq"],
                "hardened_admitted": mag02_hardened["admitted"],
                "hardened_step_fmt": mag02_hardened["step_fmt"],
                "hardened_bulk_ratio_sq": mag02_hardened["bulk_ratio_sq"],
                "hardened_max_ulp": mag02_hardened["max_ulp"],
            },
            "mag03_ville_supermartingale": {
                "m6_6": _rat(m6_6),
                "m7_7": _rat(m7_7),
                "m8_8": _rat(m8_8),
                "martingale_exact_identity": martingale_lhs == martingale_rhs,
                "uncorrected_fwer_k20_alpha_1_20": _rat(fwer_20),
                "legacy_7_0_verdict": mag03_legacy_7,
                "hardened_7_0_verdict": mag03_hardened_7["verdict"],
                "hardened_8_0_anchor_fail_promote": mag03_hardened_8_anchor_fail["promote"],
                "hardened_8_0_pass_promote": mag03_hardened_8_pass["promote"],
            },
            "mag06_weight_sample_vs_sha256": {
                "pos_affine_sign": pos_sign,
                "pos_affine_r_squared": _rat(pos_r_sq),
                "weight_sample_pos_affine_matches": pos_affine_matches,
                "weight_sample_neg_affine_matches": neg_affine_matches,
                "sha256_catches_tamper": tampered_sha != base_sha,
            },
            "mag04_05_06_08_trust_boundary": {
                "legacy_risky_admitted": trust_legacy["admitted"],
                "legacy_risky_reason_count": len(trust_legacy["reasons"]),
                "legacy_risky_reasons": trust_legacy["reasons"],
                "hardened_admitted": trust_hardened["admitted"],
                "hardened_reasons": trust_hardened["reasons"],
            },
        },
        "cross_pollination_targets": [
            "Moissanite-Labs/warp-drive",
            "Moissanite-Labs/siliquant#24",
            "Moissanite-Labs/hillclimb#4",
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["canonical_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def _default_repo_path() -> Path | None:
    candidate = Path(__file__).resolve().parent.parent / "repos" / "magnitude"
    if (candidate / ".git" / "HEAD").is_file():
        return candidate
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Zero-float audit & cross-pollination oracle for magnitudedev/magnitude"
    )
    parser.add_argument("--standalone", action="store_true", help="Skip sibling repo check")
    parser.add_argument("--repo", type=Path, default=None, help="Read-only magnitude checkout")
    parser.add_argument("--json", action="store_true", help="Emit canonical JSON")
    parser.add_argument("--write-cert", type=Path, default=None, help="Write JSON certificate")
    args = parser.parse_args(argv)

    if args.standalone:
        repo_root = None
    elif args.repo is not None:
        repo_root = args.repo
    else:
        repo_root = _default_repo_path()

    result = run_audit_suite(repo_root)
    rendered = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.write_cert is not None:
        args.write_cert.write_text(rendered, encoding="utf-8")
    if args.json:
        sys.stdout.write(rendered)
    else:
        sys.stdout.write(
            f"magnitude-bridge-audit ({result['anchors']['anchor_mode']}) "
            f"sha256={result['canonical_sha256']}\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
