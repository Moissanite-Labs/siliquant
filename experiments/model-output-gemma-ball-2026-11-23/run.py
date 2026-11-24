#!/usr/bin/env python3
"""Resumable Apple Silicon (Metal) pilot and full-sweep runner for integer ball on Gemma.

Implements the preregistered protocol in experiments/model-output-gemma-ball-2026-11-23/PREREG.md:
  - Step 2/3 one-layer pilot (--pilot-only): single-layer runtime, peak RAM, full-run estimate,
    and zero-codeword preservation checks for ZBG128:b560s16, VE8Z32:16 (centred=False),
    HBG128:b560s16, and HVE8Z32:16 (with FP64 inverse Hadamard before FP16 cast, avoiding the
    PR #41 SQ41-01 harness/tq.py:315 float16-before-turn ghost bug).
  - Full per-model sweep: builds/caches .kld references, wiki.train.raw and rand.txt imatrices
    (supporting both SentencePiece/Gemma and GPT-2 vocabularies), and llama.cpp baselines, then
    runs all PREREG arms in Step 4 order, appending SHA-256 provenance and metrics to
    results.jsonl + human-readable results.out, and skipping completed (model_id, text, variant)
    entries for clean resume.
  - Bridge mode (--bridge-eval): scores VE832:16 (centred=True, matching 2026-11-15) and
    VE8Z32:16 (centred=False) on eval.txt against the 8-bit bridge reference for P7.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import inspect
import json
import math
import os
import platform
import re
import resource
import shutil
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "lattice"))
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT / "shape"))

KINDS = ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down")
BLOCK = re.compile(r"^blk\.(\d+)\.(" + "|".join(KINDS) + r")\.weight$")

# PREREG Step 4 arm order (headline + continuity cells)
DEFAULT_ARMS = (
    "file:Q4_0",
    "file:Q4_K_M",
    "file:Q4_K_M-imat",
    "ZBG128:b560s16",
    "hZBG128:b560s16",
    "hZBG128:b560s16~rand",
    "ZBG128:b560s16+K10",
    "hZBG128:b560s16+K10",
    "EBG128:b560s16",
    "VE8Z32:16",
    "HBG128:b560s16",
    "ZBG128:b496s16",
    "hZBG128:b496s16",
    "ZBG128:b624s16",
    "hZBG128:b624s16",
    "ZB128@R:16",
    "ZB128@R:32",
)

BRIDGE_ARMS = (
    "VE832:16",
    "VE8Z32:16",
)

GROUP_BALL_RE = re.compile(r"^(ZBG|EBG|HBG)(\d+):b(\d+)s(\d+)$")
WEIGHTED_BALL_RE = re.compile(r"^([ih])(ZBG|EBG|HBG)(\d+):b(\d+)s(\d+)(?:~(rand))?$")
LOUD_BALL_RE = re.compile(r"^(h?ZBG)(\d+):b(\d+)s(\d+)\+K(\d+)$")
# 2026-11-24 addendum arms: Construction-A lattices A_n from extended Hamming codes (vendored hamming.py)
# in one ball per group of n weights, optionally with the (half-strength) imatrix column weights.
HAMMING_BALL_RE = re.compile(r"^([ih]?)(A(?:8|16|32))BG(\d+):b(\d+)s(\d+)(?:~(rand))?$")
HAMMING_M = {"A8": 3, "A16": 4, "A32": 5}
# Stock file decoded to 16-bit block weights: the same quantized weights the `file:` arm scores,
# but multiplied by llama.cpp as 16-bit weights instead of 4-bit blocks (activation-rounding control).
STOCK16_RE = re.compile(r"^file:(Q4_0|Q4_K_M|Q4_K_M-imat)@16$")
ROW_BALL_RE = re.compile(r"^(ZB|EB)(\d+)@R:(\d+)$")
VORONOI_RE = re.compile(r"^(H?)VE8(Z?)(\d+|R):(\d+)$")

LOUD_LIMIT_SIGMA = 5.0
SCALE_STEPS = {16: None, 8: 16, 6: 8}
_GGML_TYPE_CODES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
    16: "IQ2_XXS",
    17: "IQ2_XS",
    18: "IQ3_XXS",
    19: "IQ1_S",
    20: "IQ4_NL",
    21: "IQ3_S",
    22: "IQ2_S",
    23: "IQ4_XS",
    24: "I8",
    25: "I16",
    26: "I32",
    27: "I64",
    28: "F64",
    29: "IQ1_M",
    30: "BF16",
    34: "TQ1_0",
    35: "TQ2_0",
    39: "MXFP4",
}
_GGML_TYPE_MAP = {name: name.lower() for name in _GGML_TYPE_CODES.values()}
_BLK_LAYER_RE = re.compile(r"^blk\.(\d+)\.(.+)$")


def arm_needs_real_imat(arm):
    """True when `arm` requires the real-text (wikitext) importance matrix."""
    if arm == "file:Q4_K_M-imat":
        return True
    if arm.endswith("~rand"):
        return False
    if WEIGHTED_BALL_RE.match(arm):
        return True
    if LOUD_BALL_RE.match(arm) and arm.startswith("h"):
        return True
    m = HAMMING_BALL_RE.match(arm)
    if m and m.group(1) in ("i", "h"):
        return True
    mm = STOCK16_RE.match(arm)
    if mm and mm.group(1) == "Q4_K_M-imat":
        return True
    return False

# Worker process globals (populated via Pool initializer or direct call)
_worker_model_path = None
_worker_reader = None
_worker_imatrix_real_path = None
_worker_imatrix_rand_path = None
_worker_imatrix_cache = {}
_bd_mod = None
_ham_mod = None
_bounds_cache = {}
_e8_lattice = None
_e8_offset = None
_e8_centroid_cache = {}


def hadamard8():
    """Normalized order-8 Sylvester Hadamard matrix H_8 / sqrt(8), symmetric orthogonal (H^2 = I)."""
    h = np.array([[1.0]], dtype=np.float64)
    for _ in range(3):
        h = np.block([[h, h], [h, -h]])
    return h / math.sqrt(8.0)


H8 = hadamard8()


def db(x):
    return 10.0 * math.log10(x) if x > 0 else float("-inf")


def peak_rss_mb():
    """Peak resident set size of the current process in MiB (macOS reports bytes, Linux KiB)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return rss / (1024.0 * 1024.0)
    return rss / 1024.0


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def ball_module():
    global _bd_mod
    if _bd_mod is None:
        spec = importlib.util.spec_from_file_location(
            "balldim",
            ROOT / "experiments" / "ball-dimension-no-clip-rule-2026-11-17" / "run.py",
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("failed to load ball-dimension-no-clip-rule-2026-11-17/run.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _bd_mod = mod
    return _bd_mod


def ball_bound(lattice, n, bits):
    """Largest squared-norm bound at which the ball over n weights holds <= 2^bits points."""
    key = (lattice, n, bits)
    if key not in _bounds_cache:
        bd = ball_module()
        cap = 1 << bits
        r_unit = math.exp(math.lgamma(n / 2.0 + 1.0) / n) / math.sqrt(math.pi)
        limit = int(1.08 * (r_unit * 2.0 ** (bits / n)) ** 2) + 2 * n
        step = 2 if lattice == "E8" else 1
        while True:
            total = bd.poly_pow(bd.shells8(lattice, "ball", limit), n // 8, limit)
            run, best = 0, None
            for r, c in enumerate(total):
                run += c
                if run > cap:
                    break
                if r % step == 0:
                    best = (r, run)
            if run > cap:
                break
            limit = int(limit * 1.3) + 8
        if best is None:
            raise RuntimeError(f"could not determine ball bound for {lattice}, n={n}, bits={bits}")
        _bounds_cache[key] = best
    return _bounds_cache[key]


def hamming_module():
    """The vendored hamming.py (Construction-A lattices A_n) next to this runner."""
    global _ham_mod
    if _ham_mod is None:
        spec = importlib.util.spec_from_file_location("gemma_hamming", HERE / "hamming.py")
        if spec is None or spec.loader is None:
            raise RuntimeError("failed to load hamming.py next to run.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _ham_mod = mod
    return _ham_mod


def finish(W, Y, bits, shrunk):
    """Cast rounded float64 matrix Y to float16 once, and compute float64 relative error vs W."""
    Wh = np.asarray(Y, dtype=np.float64).astype(np.float16)
    W64 = np.asarray(W, dtype=np.float64)
    denom = float(np.sum(W64 ** 2))
    rel = float(np.sum((W64 - Wh.astype(np.float64)) ** 2) / denom) if denom > 0 else 0.0
    return Wh, {"rel_err": rel, "bits": float(bits), "overload": float(shrunk)}


def _group_ball_core(W64, lattice, n, block_bits, scale_bits, colscale=None):
    """Core per-group ball quantizer returning (Y_float64, bits, shrunk_fraction)."""
    bd = ball_module()
    rows, cols = W64.shape
    if cols % n != 0:
        raise ValueError(f"group size {n} does not divide row length {cols}")
    bound, points = ball_bound(lattice, n, block_bits)
    extra = 0.0
    work = W64
    if colscale is not None:
        work = work * colscale[None, :]
        extra += 16.0 / rows
    X = work.reshape(-1, n)
    t = bd.block_scales(X, lattice, "ball", bound)
    steps = SCALE_STEPS[scale_bits]
    if steps is not None:
        top = float(t.max())
        k = np.floor(np.log2(top / np.maximum(t, 1e-300)) * steps + 1e-9)
        k = np.clip(k, 0, (1 << scale_bits) - 1)
        t = top * 2.0 ** (-k / steps)
        extra += 16.0 / W64.size
    safe_t = np.where(t > 0, t, 1.0)
    Y = bd.closest(lattice, X / safe_t[:, None])
    out = bd.size_of("ball", Y) > bound + 1e-9
    if np.any(out):
        xo = X[out] / safe_t[out, None]
        lo, hi = np.zeros(len(xo)), np.ones(len(xo))
        for _ in range(bd.SHRINK_STEPS):
            mid = 0.5 * (lo + hi)
            good = bd.size_of("ball", bd.closest(lattice, xo * mid[:, None])) <= bound + 1e-9
            lo = np.where(good, mid, lo)
            hi = np.where(good, hi, mid)
        Y[out] = bd.closest(lattice, xo * lo[:, None])
    Wh64 = (Y * t[:, None]).reshape(rows, cols)
    if colscale is not None:
        Wh64 = Wh64 / colscale[None, :]
    bits = (math.log2(points) + scale_bits) / n + extra
    return Wh64, bits, float(np.mean(out))


def _hamming_block_scales(X, m, bound, margin):
    """bd.block_scales for an A_n lattice: the same bisection, with the lattice's covering margin
    sqrt(covering_sq(m, n)) in place of sqrt(n)/2 and hamming.closest as the nearest point."""
    bd = ball_module()
    ham = hamming_module()
    g = np.maximum(np.sqrt((X ** 2).sum(axis=1)), 1e-30)
    r = math.sqrt(bound)
    lo = g / (r + margin) * 0.99
    hi = g / max(r - margin, 0.05 * r) * 1.01
    for _ in range(bd.BISECT_STEPS):
        mid = np.sqrt(lo * hi)
        ok = (ham.closest(X / mid[:, None], m) ** 2).sum(axis=1) <= bound + 1e-9
        hi = np.where(ok, mid, hi)
        lo = np.where(ok, lo, mid)
    return hi


def group_hamming_ball_round(W, lattice, n, block_bits, scale_bits, colscale=None):
    """One ball per group of n weights of the Construction-A lattice A_{2^m} (hamming.py), one scale per group.

    The same steps as _group_ball_core, with hamming.closest(X, m) as the nearest point,
    hamming.ball_bound(m, n, block_bits) for the bound and the count of points, and
    sqrt(hamming.covering_sq(m, n)) as the bisection's margin in place of sqrt(n)/2.
    """
    bd = ball_module()
    ham = hamming_module()
    m = HAMMING_M[lattice]
    W64 = np.asarray(W, dtype=np.float64)
    rows, cols = W64.shape
    if cols % n != 0:
        raise ValueError(f"group size {n} does not divide row length {cols}")
    bound, points = ham.ball_bound(m, n, block_bits)
    extra = 0.0
    work = W64
    if colscale is not None:
        work = work * colscale[None, :]
        extra += 16.0 / rows
    X = work.reshape(-1, n)
    t = _hamming_block_scales(X, m, bound, math.sqrt(ham.covering_sq(m, n)))
    steps = SCALE_STEPS[scale_bits]
    if steps is not None:
        top = float(t.max())
        k = np.floor(np.log2(top / np.maximum(t, 1e-300)) * steps + 1e-9)
        k = np.clip(k, 0, (1 << scale_bits) - 1)
        t = top * 2.0 ** (-k / steps)
        extra += 16.0 / W64.size
    safe_t = np.where(t > 0, t, 1.0)
    Y = ham.closest(X / safe_t[:, None], m)
    out = (Y ** 2).sum(axis=1) > bound + 1e-9
    if np.any(out):
        xo = X[out] / safe_t[out, None]
        lo, hi = np.zeros(len(xo)), np.ones(len(xo))
        for _ in range(bd.SHRINK_STEPS):
            mid = 0.5 * (lo + hi)
            good = (ham.closest(xo * mid[:, None], m) ** 2).sum(axis=1) <= bound + 1e-9
            lo = np.where(good, mid, lo)
            hi = np.where(good, hi, mid)
        Y[out] = ham.closest(xo * lo[:, None], m)
    Wh64 = (Y * t[:, None]).reshape(rows, cols)
    if colscale is not None:
        Wh64 = Wh64 / colscale[None, :]
    bits = (math.log2(points) + scale_bits) / n + extra
    return finish(W, Wh64, bits, float(np.mean(out)))


def group_ball_round(W, lattice, n, block_bits, scale_bits, colscale=None, hadamard=False):
    """Round W with one ball per group of n weights, optionally under order-8 Hadamard turn.

    When hadamard=True, both the forward turn (X @ H8) and inverse turn (Y @ H8) stay in float64
    so zero blocks remain exact 0.0 before the single final float16 cast (preventing SQ41-01).
    """
    W64 = np.asarray(W, dtype=np.float64)
    rows, cols = W64.shape
    if hadamard:
        if cols % 8 != 0:
            raise ValueError(f"Hadamard-8 turn requires row length divisible by 8, got {cols}")
        W_in = (W64.reshape(-1, 8) @ H8).reshape(rows, cols)
        Y_rot, bits, shrunk = _group_ball_core(W_in, lattice, n, block_bits, scale_bits, colscale=colscale)
        Y64 = (Y_rot.reshape(-1, 8) @ H8).reshape(rows, cols)
        # Preserve exact zero 8-blocks cleanly against sub-ulp floating-point noise
        zero_blocks = np.all(W64.reshape(-1, 8) == 0.0, axis=1)
        if np.any(zero_blocks):
            Y64_blk = Y64.reshape(-1, 8)
            Y64_blk[zero_blocks] = 0.0
            Y64 = Y64_blk.reshape(rows, cols)
        return finish(W, Y64, bits, shrunk)
    Y64, bits, shrunk = _group_ball_core(W64, lattice, n, block_bits, scale_bits, colscale=colscale)
    return finish(W, Y64, bits, shrunk)


def row_ball_round(W, lattice, n, q):
    """Cut ball over n weights with one 16-bit scale per row R (ZB128@R:16 / ZB128@R:32)."""
    bd = ball_module()
    W64 = np.asarray(W, dtype=np.float64)
    rows, cols = W64.shape
    if cols % n != 0:
        raise ValueError(f"block size {n} does not divide row length {cols}")
    bound, points = bd.codebook(lattice, "ball", n, q)
    X = W64.reshape(-1, n)
    tb = bd.block_scales(X, lattice, "ball", bound)
    tg = tb.reshape(rows, cols // n).max(axis=1)
    t = np.repeat(tg, cols // n)
    safe_t = np.where(t > 0, t, 1.0)
    Y = bd.closest(lattice, X / safe_t[:, None])
    out = bd.size_of("ball", Y) > bound + 1e-9
    if np.any(out):
        xo = X[out] / safe_t[out, None]
        lo, hi = np.zeros(len(xo)), np.ones(len(xo))
        for _ in range(bd.SHRINK_STEPS):
            mid = 0.5 * (lo + hi)
            good = bd.size_of("ball", bd.closest(lattice, xo * mid[:, None])) <= bound + 1e-9
            lo = np.where(good, mid, lo)
            hi = np.where(good, hi, mid)
        Y[out] = bd.closest(lattice, xo * lo[:, None])
    bits = math.log2(points) / n + 16.0 / cols
    return finish(W, (Y * t[:, None]).reshape(rows, cols), bits, float(np.mean(out)))


def loud_columns(W, k=10, limit_sigma=LOUD_LIMIT_SIGMA):
    """Top-k columns of W where a row-wise limit_sigma clip removes the most squared weight."""
    W64 = np.asarray(W, dtype=np.float64)
    lim = limit_sigma * W64.std(axis=1, keepdims=True)
    lost = ((W64 - np.clip(W64, -lim, lim)) ** 2).sum(axis=0)
    k_eff = min(int(k), W64.shape[1])
    return np.sort(np.argsort(-lost, kind="stable")[:k_eff])


def _e8_voronoi_numpy(W64, q, group, centred=False, steps=14):
    """Pure-NumPy E8 maxscale Voronoi quantizer returning (Y64, bits, overload).

    Matches harness/tq.py's maxscale_voronoi (and PR #46 centred=True/False) while returning
    float64 so callers applying an inverse Hadamard turn avoid SQ41-01 float16 ghosting.
    """
    global _e8_lattice, _e8_offset
    import lattices as L

    if _e8_lattice is None:
        _e8_lattice = L.E8()
        _e8_offset = L.default_offset(8)
    rows, cols = W64.shape
    size = cols if group == "R" else int(group)
    if size % 8 != 0 or cols % size != 0:
        raise ValueError(f"group size {size} must be a multiple of 8 dividing {cols}")
    if centred:
        if q not in _e8_centroid_cache:
            _e8_centroid_cache[q] = np.asarray(L.codebook_centroid(_e8_lattice, q, _e8_offset), dtype=np.float64)
        centre = _e8_centroid_cache[q]
    else:
        centre = np.zeros(8, dtype=np.float64)
    X = W64.reshape(-1, 8)
    norm = np.maximum(np.linalg.norm(X, axis=1), 1e-20)
    rho, cover = 0.5 ** 0.5, 1.0
    lo = norm / (cover * q) * 0.9
    hi = norm / (rho * q) * 1.1
    for _ in range(steps):
        mid = np.sqrt(lo * hi)
        ok = L._roundtrip(X / mid[:, None] + centre, _e8_lattice, q, _e8_offset)[3]
        hi = np.where(ok, mid, hi)
        lo = np.where(ok, lo, mid)
    t = np.repeat(hi.reshape(-1, size // 8).max(axis=1), size // 8)
    R, _, over = L.voronoi_quantize(X / t[:, None], _e8_lattice, q, _e8_offset, centre=centre)
    Y64 = (R * t[:, None]).reshape(rows, cols)
    bits = math.log2(q) + 16.0 / size
    return Y64, bits, float(np.mean(over))


def voronoi_e8_round(W, q, group, centred=False, hadamard=False):
    """E8 Voronoi code with explicit centred flag (PR #46) and optional FP64 Hadamard turn.

    Uses torch harness/tq.py when torch is installed and hadamard=False (fast on MPS/CPU),
    and always keeps Hadamard inverse transforms in float64 prior to the final float16 cast
    (preventing PR #41 SQ41-01 float16-before-turn ghosting).
    """
    if not isinstance(centred, bool):
        raise ValueError(f"centred must be bool, got {type(centred).__name__}")
    W64 = np.asarray(W, dtype=np.float64)
    rows, cols = W64.shape
    if hadamard:
        if cols % 8 != 0:
            raise ValueError(f"Hadamard-8 turn requires row length divisible by 8, got {cols}")
        W_in = (W64.reshape(-1, 8) @ H8).reshape(rows, cols)
        Y_rot, bits, over = _e8_voronoi_numpy(W_in, q, group, centred=centred)
        Y64 = (Y_rot.reshape(-1, 8) @ H8).reshape(rows, cols)
        if not centred:
            zero_blocks = np.all(W64.reshape(-1, 8) == 0.0, axis=1)
            if np.any(zero_blocks):
                Y64_blk = Y64.reshape(-1, 8)
                Y64_blk[zero_blocks] = 0.0
                Y64 = Y64_blk.reshape(rows, cols)
        return finish(W, Y64, bits, over)

    try:
        import torch
        import tq
        torch.set_num_threads(1)
        sig = inspect.signature(tq.maxscale_voronoi)
        if "centred" in sig.parameters:
            Wh, info = tq.maxscale_voronoi(np.asarray(W, dtype=np.float32), q, group, name="E8", centred=centred)
            return Wh, info
        code = tq.Code("E8", q)
        if centred:
            return tq.maxscale_voronoi(np.asarray(W, dtype=np.float32), q, group, name="E8")
        kept = code.centre
        code.centre = torch.zeros_like(kept)
        try:
            return tq.maxscale_voronoi(np.asarray(W, dtype=np.float32), q, group, name="E8")
        finally:
            code.centre = kept
    except ImportError:
        Y64, bits, over = _e8_voronoi_numpy(W64, q, group, centred=centred)
        return finish(W, Y64, bits, over)


def verify_zero_codeword_matrix(W):
    """Verify exact zero-codeword preservation on synthetic zero+outlier rows and a sample of W.

    Tests ZBG128:b560s16, VE8Z32:16 (centred=False), HBG128:b560s16, and HVE8Z32:16.
    Raises RuntimeError if any zero block develops non-zero ghosts.
    """
    # Construct synthetic test matrix: 4 rows x 256 cols where half the 8-blocks are exact 0
    # and the other half contain a 40-sigma outlier to stress scale selection.
    rng = np.random.default_rng(42)
    synth = np.zeros((4, 256), dtype=np.float32)
    synth[:, 128:] = rng.standard_normal((4, 128)).astype(np.float32) * 0.02
    synth[:, 128] = 1.5  # strong outlier in the second half of each row
    zero_mask = synth == 0.0

    # Also test a slice of the real Gemma matrix with its first 128 columns zeroed
    real_rows = min(8, W.shape[0])
    real_cols = (W.shape[1] // 128) * 128
    if real_cols < 128:
        raise ValueError(f"pilot matrix row width {W.shape[1]} is smaller than 128")
    real_slice = np.asarray(W[:real_rows, :real_cols], dtype=np.float32).copy()
    real_slice[:, :128] = 0.0

    checks = {}
    cases = [
        ("ZBG128:b560s16", lambda M: group_ball_round(M, "Z8", 128, 560, 16, hadamard=False)),
        ("VE8Z32:16", lambda M: voronoi_e8_round(M, 16, 32, centred=False, hadamard=False)),
        ("HBG128:b560s16", lambda M: group_ball_round(M, "E8", 128, 560, 16, hadamard=True)),
        ("HVE8Z32:16", lambda M: voronoi_e8_round(M, 16, 32, centred=False, hadamard=True)),
    ]
    for label, fn in cases:
        Wh_s, _ = fn(synth)
        synth_ghosts = float(np.mean(Wh_s[zero_mask] != 0.0))
        Wh_r, _ = fn(real_slice)
        real_ghosts = float(np.mean(Wh_r[:, :128] != 0.0))
        checks[label] = {
            "synth_zero_ghost_rate": synth_ghosts,
            "real_zero_ghost_rate": real_ghosts,
            "passed": synth_ghosts == 0.0 and real_ghosts == 0.0,
        }
        if synth_ghosts > 0.0 or real_ghosts > 0.0:
            raise RuntimeError(
                f"FAIL zero-codeword check for {label}: "
                f"synth_ghosts={synth_ghosts:.4f}, real_ghosts={real_ghosts:.4f}"
            )
    return checks


# ---------------------------------------------------------------------------
# GGUF & imatrix helpers
# ---------------------------------------------------------------------------

def _load_ggufedit():
    import ggufedit
    return ggufedit


_worker_tensor_dir = None


def _worker_init(model_path, imatrix_real_path, imatrix_rand_path, tensor_dir=None):
    global _worker_model_path, _worker_reader, _worker_imatrix_real_path, _worker_imatrix_rand_path, _worker_imatrix_cache, _worker_tensor_dir
    _worker_model_path = model_path
    _worker_reader = None
    _worker_imatrix_real_path = imatrix_real_path
    _worker_imatrix_rand_path = imatrix_rand_path
    _worker_imatrix_cache = {}
    _worker_tensor_dir = tensor_dir


def _get_reader():
    global _worker_reader
    if _worker_reader is None:
        ggufedit = _load_ggufedit()
        _worker_reader = ggufedit.open_model(_worker_model_path)
    return _worker_reader


def _targets(rd):
    return [t for t in rd.tensors if BLOCK.match(t.name) and len(t.shape) == 2]


def _matrix_of(t):
    ggufedit = _load_ggufedit()
    return np.ascontiguousarray(
        ggufedit.tensor_float(t).reshape(tuple(int(d) for d in reversed(t.shape))),
        dtype=np.float32,
    )


def _column_weights(name, cols, power, which="real"):
    """Return per-column imatrix scales for `name`, or None if `name` has no valid imatrix entry."""
    path = _worker_imatrix_rand_path if which == "rand" else _worker_imatrix_real_path
    if not path:
        raise ValueError(f"imatrix ({which}) is required for weighted arm on {name}")
    if path not in _worker_imatrix_cache:
        _load_ggufedit()
        import gguf
        rd = gguf.GGUFReader(path)
        _worker_imatrix_cache[path] = {
            t.name: np.asarray(t.data, dtype=np.float64).ravel()
            for t in rd.tensors
            if t.name.endswith((".in_sum2", ".counts"))
        }
    im = _worker_imatrix_cache[path]
    in_sum2_key = name + ".in_sum2"
    counts_key = name + ".counts"
    if in_sum2_key not in im or counts_key not in im:
        return None
    counts = im[counts_key]
    if len(counts) == 0 or float(counts[0]) <= 0.0:
        return None
    imp = im[in_sum2_key] / float(counts[0])
    if len(imp) != cols:
        raise ValueError(f"imatrix has {len(imp)} inputs for {name}, expected {cols}")
    mean_imp = float(imp.mean())
    if not math.isfinite(mean_imp) or mean_imp <= 0.0:
        return None
    s = (imp / mean_imp) ** power
    return np.maximum(s, 0.01)


def is_unread_gemma_e4b_tensor(model_id, name, n_layers):
    """True for the 36 unread attn_k/attn_v matrices in layers 24..41 of gemma-4-E4B-it."""
    if "e4b" not in model_id.lower() or n_layers != 42:
        return False
    m = BLOCK.match(name)
    if not m:
        return False
    layer, kind = int(m.group(1)), m.group(2)
    return layer >= 24 and kind in ("attn_k", "attn_v")


def round_matrix_variant(variant, name, W, colscale_fn=None):
    """Dispatch a single matrix W to the requested PREREG variant. Returns (Wh_f16, info_dict)."""
    W64 = np.asarray(W, dtype=np.float64)
    energy = float(np.sum(W64 ** 2))
    m_block = BLOCK.match(name)
    kind = m_block.group(2) if m_block else ""
    imatrix_fallback = False

    if variant == "identity":
        Wh, info = finish(W, W, 16.0, 0.0)
    elif LOUD_BALL_RE.match(variant):
        m = LOUD_BALL_RE.match(variant)
        arm_prefix, n, bbits, sbits, k = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))
        use_imat = arm_prefix.startswith("h")
        cs = colscale_fn(name, W.shape[1], 0.25, "real") if use_imat else None
        imatrix_fallback = use_imat and cs is None
        if kind == "ffn_down":
            cols = loud_columns(W, k=k)
            rest = np.asarray(W, dtype=np.float32).copy()
            rest[:, cols] = 0.0
            Wh, info = group_ball_round(rest, "Z8", n, bbits, sbits, colscale=cs, hadamard=False)
            Wh[:, cols] = np.asarray(W[:, cols], dtype=np.float16)
            rel = float(np.sum((W64 - Wh.astype(np.float64)) ** 2) / energy) if energy > 0 else 0.0
            kept_frac = len(cols) / W.shape[1]
            # Kept columns counted at 16 bits/weight + 16 bits/column index
            bits = info["bits"] * (1.0 - kept_frac) + 16.0 * kept_frac + (16.0 * len(cols)) / W.size
            info = {"rel_err": rel, "bits": bits, "overload": info["overload"], "kept_columns": int(len(cols))}
        else:
            Wh, info = group_ball_round(W, "Z8", n, bbits, sbits, colscale=cs, hadamard=False)
            info["kept_columns"] = 0
    elif HAMMING_BALL_RE.match(variant):
        m = HAMMING_BALL_RE.match(variant)
        prefix, lattice, n, bbits, sbits, src = m.group(1), m.group(2), int(m.group(3)), int(m.group(4)), int(m.group(5)), m.group(6) or "real"
        power = 0.5 if prefix == "i" else 0.25
        cs = colscale_fn(name, W.shape[1], power, src) if prefix in ("i", "h") else None
        imatrix_fallback = prefix in ("i", "h") and cs is None
        Wh, info = group_hamming_ball_round(W, lattice, n, bbits, sbits, colscale=cs)
    elif WEIGHTED_BALL_RE.match(variant):
        m = WEIGHTED_BALL_RE.match(variant)
        prefix, kind_code, n, bbits, sbits, src = m.group(1), m.group(2), int(m.group(3)), int(m.group(4)), int(m.group(5)), m.group(6) or "real"
        power = 0.5 if prefix == "i" else 0.25
        lattice = "Z8" if kind_code == "ZBG" else "E8"
        hadamard = kind_code == "HBG"
        cs = colscale_fn(name, W.shape[1], power, src)
        imatrix_fallback = cs is None
        Wh, info = group_ball_round(W, lattice, n, bbits, sbits, colscale=cs, hadamard=hadamard)
    elif GROUP_BALL_RE.match(variant):
        m = GROUP_BALL_RE.match(variant)
        kind_code, n, bbits, sbits = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
        lattice = "Z8" if kind_code == "ZBG" else "E8"
        hadamard = kind_code == "HBG"
        Wh, info = group_ball_round(W, lattice, n, bbits, sbits, hadamard=hadamard)
    elif ROW_BALL_RE.match(variant):
        m = ROW_BALL_RE.match(variant)
        lattice = "Z8" if m.group(1) == "ZB" else "E8"
        Wh, info = row_ball_round(W, lattice, int(m.group(2)), int(m.group(3)))
    elif VORONOI_RE.match(variant):
        m = VORONOI_RE.match(variant)
        hadamard = bool(m.group(1))
        zero_centre = bool(m.group(2))
        grp = m.group(3) if m.group(3) == "R" else int(m.group(3))
        q = int(m.group(4))
        Wh, info = voronoi_e8_round(W, q, grp, centred=not zero_centre, hadamard=hadamard)
    else:
        raise ValueError(f"unknown variant: {variant}")

    return Wh, {
        "rel_err": float(info["rel_err"]),
        "bits": float(info["bits"]),
        "overload": float(info.get("overload") or 0.0),
        "energy": energy,
        "size": int(W.size),
        "kept_columns": int(info.get("kept_columns", 0)),
        "imatrix_fallback": bool(imatrix_fallback),
    }


def _round_one_worker(task):
    variant, name = task
    t = next(t for t in _get_reader().tensors if t.name == name)
    W = _matrix_of(t)
    Wh, info = round_matrix_variant(variant, name, W, colscale_fn=_column_weights)
    if _worker_tensor_dir:
        # Stream mode: hand the rounded matrix to disk instead of through IPC, so the
        # parent never holds the whole model's replacements in RAM (27B-class models).
        os.makedirs(_worker_tensor_dir, exist_ok=True)
        sidecar = os.path.join(_worker_tensor_dir, re.sub(r"[^A-Za-z0-9_.+-]", "_", name) + ".f16.npy")
        np.save(sidecar, Wh)
        return name, None, info, sidecar
    return name, Wh, info, None


def stats_of(infos, model_id, n_layers):
    """Aggregate counted-all and active-only bits and weight error across block matrices."""
    def _agg(items):
        per_kind, weights, bits, energy, energy_err, over, kept = {}, 0, 0.0, 0.0, 0.0, 0.0, 0
        for name, info in items:
            per_kind.setdefault(BLOCK.match(name).group(2), []).append(info["rel_err"])
            weights += info["size"]
            bits += info["bits"] * info["size"]
            energy += info["energy"]
            energy_err += info["energy"] * info["rel_err"]
            over += (info["overload"] or 0.0) * info["size"]
            kept += info.get("kept_columns", 0)
        every = [v for vs in per_kind.values() for v in vs]
        if not every or weights == 0:
            return {"matrices": 0, "weights": 0, "bits_per_weight": 0.0, "weight_err_db_mean": float("-inf"), "weight_err_db_pooled": float("-inf"), "share_out_of_range": 0.0, "kept_columns": 0}
        return {
            "matrices": len(every),
            "weights": weights,
            "bits_per_weight": bits / weights,
            "weight_err_db_mean": sum(db(v) for v in every) / len(every),
            "weight_err_db_pooled": db(energy_err / energy) if energy > 0 else float("-inf"),
            "weight_err_db_by_kind": {k: sum(db(v) for v in vs) / len(vs) for k, vs in sorted(per_kind.items())},
            "share_out_of_range": over / weights,
            "kept_columns": kept,
        }

    all_items = list(infos.items())
    active_items = [(n, i) for n, i in all_items if not is_unread_gemma_e4b_tensor(model_id, n, n_layers)]
    fallback_tensors = sorted(n for n, i in all_items if i.get("imatrix_fallback"))
    out = _agg(all_items)
    act = _agg(active_items)
    out["matrices_active"] = act["matrices"]
    out["weights_active"] = act["weights"]
    out["bits_per_weight_active"] = act["bits_per_weight"]
    out["weight_err_db_mean_active"] = act["weight_err_db_mean"]
    out["weight_err_db_pooled_active"] = act["weight_err_db_pooled"]
    out["imatrix_fallback_count"] = len(fallback_tensors)
    out["imatrix_fallback_tensors"] = fallback_tensors
    return out


def compare_file(ref_reader, other_path, model_id, n_layers):
    """Compare a llama.cpp quantized GGUF against the reference GGUF."""
    ggufedit = _load_ggufedit()
    other = {t.name: t for t in ggufedit.open_model(other_path).tensors}
    infos, differ = {}, []
    for t in ref_reader.tensors:
        o = other[t.name]
        if BLOCK.match(t.name) and len(t.shape) == 2:
            W = ggufedit.tensor_float(t).astype(np.float64)
            Wh = ggufedit.tensor_float(o).astype(np.float64).reshape(W.shape)
            e = float(np.sum(W ** 2))
            infos[t.name] = {
                "rel_err": float(np.sum((W - Wh) ** 2)) / e if e > 0 else 0.0,
                "bits": 8.0 * int(o.n_bytes) / W.size,
                "overload": 0.0,
                "energy": e,
                "size": int(W.size),
                "kept_columns": 0,
            }
        elif o.tensor_type != t.tensor_type or not np.array_equal(np.asarray(o.data), np.asarray(t.data)):
            differ.append(t.name)
    row = stats_of(infos, model_id, n_layers)
    row["other_tensors_that_differ"] = differ
    return row


# ---------------------------------------------------------------------------
# Data-free random vocabulary text (SentencePiece / Gemma + GPT-2)
# ---------------------------------------------------------------------------

def _gpt2_byte_table():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAC + 1)) + list(range(0xAE, 0xFF + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def build_rand_text(model_path, out_path, pieces_count=80000, seed=1):
    """Generate data-free random vocabulary text for both SentencePiece (Gemma/Llama) and GPT-2 GGUFs."""
    _load_ggufedit()
    import gguf

    rd = gguf.GGUFReader(model_path)
    kind_field = rd.fields.get("tokenizer.ggml.model")
    kind = bytes(kind_field.parts[kind_field.data[0]]).decode("utf-8") if kind_field is not None else "llama"
    tok = rd.fields["tokenizer.ggml.tokens"]
    pieces = [bytes(tok.parts[i]).decode("utf-8", errors="ignore") for i in tok.data]
    typ_field = rd.fields.get("tokenizer.ggml.token_type")
    if typ_field is not None:
        types = [int(typ_field.parts[i][0]) for i in typ_field.data]
    else:
        types = [1] * len(pieces)

    rng = np.random.default_rng(seed)
    if kind == "gpt2":
        table = _gpt2_byte_table()
        ordinary = [i for i, (p, t) in enumerate(zip(pieces, types)) if t == 1 and p and all(ch in table for ch in p)]
        if not ordinary:
            raise RuntimeError("no ordinary GPT-2 tokens found in GGUF vocabulary")
        draw = rng.integers(0, len(ordinary), size=pieces_count)
        raw = b"".join(bytes(table[ch] for ch in pieces[ordinary[i]]) for i in draw)
        text = raw.decode("utf-8", errors="ignore")
    else:
        # SentencePiece ("llama" / Gemma): type == 1 normal pieces, \u2581 represents space
        ordinary = [
            i for i, (p, t) in enumerate(zip(pieces, types))
            if t == 1 and p and not (p.startswith("<") and p.endswith(">"))
        ]
        if not ordinary:
            raise RuntimeError(f"no ordinary SentencePiece tokens found for tokenizer kind={kind!r}")
        draw = rng.integers(0, len(ordinary), size=pieces_count)
        text = "".join(pieces[ordinary[i]].replace("\u2581", " ") for i in draw)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    return sha256_of(out_path)


# ---------------------------------------------------------------------------
# Subprocess & caching helpers for llama.cpp
# ---------------------------------------------------------------------------

def _run_cmd(cmd, label):
    print(f"[{label}] {' '.join(cmd)}", file=sys.stderr, flush=True)
    proc = subprocess.run(cmd, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {proc.returncode}: {' '.join(cmd)}")


_FLOAT_GGML_TYPES = (0, 1, 30)  # F32, F16, BF16


def _valid_gguf(path):
    """A cached GGUF is trusted only if it starts with the GGUF magic (a killed run can
    leave a zero-header partial file behind)."""
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"GGUF"
    except OSError:
        return False


def _source_is_quantized(tensors):
    """True if any source tensor is already quantized: llama-quantize then needs
    --allow-requantize (Q8_0-reference stratum, e.g. gemma-3-27b-it)."""
    return any(int(getattr(t, "tensor_type", 0)) not in _FLOAT_GGML_TYPES for t in tensors)


def ensure_kld(model_path, text_path, kld_path, binary, chunks, ctx, batch, ngl, threads, dry_run=False):
    cmd = [
        binary,
        "-m", str(model_path),
        "-f", str(text_path),
        "-c", str(ctx),
        "-b", str(batch),
        "--chunks", str(chunks),
        "-ngl", str(ngl),
        "-t", str(threads),
        "--kl-divergence-base", str(kld_path),
    ]
    if dry_run or (os.path.exists(kld_path) and _valid_gguf(kld_path)):
        return cmd
    if os.path.exists(kld_path):
        os.remove(kld_path)
    os.makedirs(os.path.dirname(os.path.abspath(kld_path)), exist_ok=True)
    _run_cmd(cmd, f"kld:{os.path.basename(kld_path)}")
    return cmd


def ensure_imatrix(model_path, text_path, imatrix_path, binary, chunks, ctx, batch, ngl, threads, dry_run=False):
    cmd = [
        binary,
        "-m", str(model_path),
        "-f", str(text_path),
        "-c", str(ctx),
        "-b", str(batch),
        "--chunks", str(chunks),
        "-ngl", str(ngl),
        "-t", str(threads),
        "-o", str(imatrix_path),
    ]
    if dry_run or (os.path.exists(imatrix_path) and _valid_gguf(imatrix_path)):
        return cmd
    if os.path.exists(imatrix_path):
        os.remove(imatrix_path)
    os.makedirs(os.path.dirname(os.path.abspath(imatrix_path)), exist_ok=True)
    _run_cmd(cmd, f"imatrix:{os.path.basename(imatrix_path)}")
    return cmd


def _ggml_type_flag(tensor_type, tensor_name="tensor"):
    """Map a GGUF tensor_type (enum, string, or GGML type code) to a llama-quantize --tensor-type value."""
    name = getattr(tensor_type, "name", None)
    if name is not None:
        key = str(name).upper()
    elif isinstance(tensor_type, str):
        key = tensor_type.upper()
    elif isinstance(tensor_type, int) and not isinstance(tensor_type, bool):
        key = _GGML_TYPE_CODES.get(int(tensor_type))
    else:
        key = None
    if key in _GGML_TYPE_MAP:
        return _GGML_TYPE_MAP[key]
    raise ValueError(f"unsupported {tensor_name} tensor_type: {tensor_type!r}")


def _is_block_target(name, shape=None):
    return bool(BLOCK.match(name)) and (shape is None or len(shape) == 2)


def quantize_tensor_type_flags(tensors, token_embd_type=None):
    """Preserve every non-block tensor in its exact source GGUF dtype.

    `tensors` may be an iterable of GGUF tensor objects (with `.name` and `.tensor_type`)
    or plain tensor name strings. For tensor objects, every non-block tensor keeps its
    exact source `tensor_type` (bf16, f16, f32, q8_0, etc.), collapsing uniform
    `blk.<n>.<suffix>` groups into a single anchored regex `^blk\\.\\d+\\.<suffix>$=<type>`.
    When `token_embd_type` is explicitly supplied, it overrides `token_embd.weight` (and
    serves as the fallback dtype for string-only calls, defaulting to `f16` for token
    embedding tensors and `f32` for other non-block string names).
    """
    default_embd_flag = _ggml_type_flag(token_embd_type, "token_embd.weight") if token_embd_type is not None else "f16"
    non_block = []
    seen = set()
    for item in tensors:
        if isinstance(item, str):
            tname = item
            if tname in seen or _is_block_target(tname):
                continue
            seen.add(tname)
            if tname in ("token_embd.weight", "per_layer_token_embd.weight"):
                flag = default_embd_flag
            elif token_embd_type is not None:
                flag = default_embd_flag
            else:
                flag = "f32"
            non_block.append((tname, flag))
        else:
            tname = item.name
            shape = getattr(item, "shape", None)
            if tname in seen or _is_block_target(tname, shape):
                continue
            seen.add(tname)
            if token_embd_type is not None and tname == "token_embd.weight":
                flag = default_embd_flag
            else:
                flag = _ggml_type_flag(getattr(item, "tensor_type", None), tname)
            non_block.append((tname, flag))

    blk_groups = {}
    for tname, flag in non_block:
        m = _BLK_LAYER_RE.match(tname)
        if m:
            blk_groups.setdefault(m.group(2), []).append((tname, flag))

    collapsible = {}
    for suffix, entries in blk_groups.items():
        if len(entries) > 1 and len({f for _, f in entries}) == 1 and not BLOCK.match(f"blk.0.{suffix}"):
            collapsible[suffix] = entries[0][1]

    flags = []
    emitted_suffixes = set()
    for tname, flag in non_block:
        m = _BLK_LAYER_RE.match(tname)
        if m and m.group(2) in collapsible:
            suffix = m.group(2)
            if suffix not in emitted_suffixes:
                flags.extend(["--tensor-type", rf"^blk\.\d+\.{re.escape(suffix)}$={collapsible[suffix]}"])
                emitted_suffixes.add(suffix)
        else:
            flags.extend(["--tensor-type", rf"^{re.escape(tname)}$={flag}"])
    return flags


def ensure_baseline_gguf(model_path, out_path, qtype, binary, tensors, imatrix_path=None, threads=4, dry_run=False, token_embd_type=None):
    cmd = [binary]
    if _source_is_quantized(tensors):
        cmd.append("--allow-requantize")
    if imatrix_path:
        cmd.extend(["--imatrix", str(imatrix_path)])
    cmd.extend(quantize_tensor_type_flags(tensors, token_embd_type=token_embd_type))
    cmd.extend([str(model_path), str(out_path), qtype, str(threads)])
    if dry_run or (os.path.exists(out_path) and _valid_gguf(out_path)):
        return cmd
    if os.path.exists(out_path):
        os.remove(out_path)  # partial/corrupt cache from a killed run
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    _run_cmd(cmd, f"quantize:{qtype}")
    return cmd


# ---------------------------------------------------------------------------
# Summary writer (results.out)
# ---------------------------------------------------------------------------

def render_results_out(jsonl_path, out_path):
    """Generate human-readable summary table from append-only results.jsonl."""
    rows = []
    if os.path.exists(jsonl_path):
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    lines = [
        "# Gemma integer ball results summary",
        "",
        f"Rows: {len(rows)}",
        "",
        "| Model | Text | Variant | Bits (all) | Bits (active) | W-err dB (mean) | Mean KL | SE | Top-1 kept | Build s | Score s |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        kl = f"{r['kl_mean']:.6f}" if "kl_mean" in r else "-"
        kl_se = f"{r['kl_mean_se']:.6f}" if "kl_mean_se" in r else "-"
        top1 = f"{r['same_top']:.4f}" if "same_top" in r else "-"
        bits_all = f"{r.get('bits_per_weight', 0.0):.4f}"
        bits_act = f"{r.get('bits_per_weight_active', r.get('bits_per_weight', 0.0)):.4f}"
        werr = f"{r.get('weight_err_db_mean', float('nan')):.2f}"
        bs = f"{r.get('build_seconds', 0.0):.1f}"
        ss = f"{r.get('score_seconds', 0.0):.1f}"
        lines.append(
            f"| {r.get('model_id', '')} | {r.get('text', '')} | {r.get('variant', '')} | "
            f"{bits_all} | {bits_act} | {werr} | {kl} | {kl_se} | {top1} | {bs} | {ss} |"
        )
    lines.append("")
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Step 2/3 One-layer pilot
# ---------------------------------------------------------------------------

def run_pilot(args):
    ggufedit = _load_ggufedit()
    rd = ggufedit.open_model(args.model_path)
    all_targets = _targets(rd)
    if not all_targets:
        raise RuntimeError(f"no block matrices matching {BLOCK.pattern} found in {args.model_path}")
    layers = sorted({int(BLOCK.match(t.name).group(1)) for t in all_targets})
    n_layers = len(layers)
    pilot_layer = args.pilot_layer if args.pilot_layer in layers else layers[0]
    layer_tensors = [t for t in all_targets if int(BLOCK.match(t.name).group(1)) == pilot_layer]
    active_targets = [t for t in all_targets if not is_unread_gemma_e4b_tensor(args.model_id, t.name, n_layers)]

    # Step 3: zero-codeword verification on first matrix of pilot layer + synthetic outlier matrix
    sample_W = _matrix_of(layer_tensors[0])
    zero_checks = verify_zero_codeword_matrix(sample_W)

    # Step 2: single-layer timing & peak RAM across ZBG128:b560s16, VE8Z32:16, HBG128:b560s16
    pilot_arms = ("ZBG128:b560s16", "VE8Z32:16", "HBG128:b560s16")
    timings = {}
    for arm in pilot_arms:
        t0 = time.time()
        infos = {}
        for t in layer_tensors:
            W = _matrix_of(t)
            _, info = round_matrix_variant(arm, t.name, W)
            infos[t.name] = info
        elapsed = time.time() - t0
        agg = stats_of(infos, args.model_id, n_layers)
        timings[arm] = {
            "layer_seconds": round(elapsed, 3),
            "est_full_model_seconds_1worker": round(elapsed * n_layers, 1),
            "est_full_model_seconds_workers": round(elapsed * n_layers / max(1, args.workers), 1),
            "bits_per_weight": round(agg["bits_per_weight"], 5),
            "weight_err_db_mean": round(agg["weight_err_db_mean"], 3),
        }

    report = {
        "mode": "pilot",
        "model_id": args.model_id,
        "model_path": str(args.model_path),
        "model_sha256": sha256_of(args.model_path),
        "n_layers": n_layers,
        "pilot_layer": pilot_layer,
        "matrices_total": len(all_targets),
        "matrices_active": len(active_targets),
        "weights_total": int(sum(int(np.prod(t.shape)) for t in all_targets)),
        "weights_active": int(sum(int(np.prod(t.shape)) for t in active_targets)),
        "peak_rss_mb": round(peak_rss_mb(), 2),
        "zero_codeword_checks": zero_checks,
        "pilot_timings": timings,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    os.makedirs(args.out_dir, exist_ok=True)
    pilot_path = os.path.join(args.out_dir, f"pilot-{args.model_id}.json")
    with open(pilot_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


# ---------------------------------------------------------------------------
# Full per-model sweep & bridge mode
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Resumable Gemma integer ball pilot & sweep runner (Apple Silicon / Metal).")
    ap.add_argument("--model-id", required=True, help="Logical model id (e.g. gemma-4-E4B-it, gemma-3-4b-it)")
    ap.add_argument("--model-path", required=True, help="Path to 16-bit GGUF (or 8-bit GGUF for --bridge-eval)")
    ap.add_argument("--out-dir", required=True, help="Output directory for results.jsonl, results.out, and cache/")
    ap.add_argument("--texts", default="wikitext,python", help="Comma-separated headline texts: wikitext,python")
    ap.add_argument("--wikitext-test", default="", help="Path to wikitext-2 test text (wiki.test.raw)")
    ap.add_argument("--wikitext-train", default="", help="Path to wikitext-2 train text (wiki.train.raw) for real-text imatrix")
    ap.add_argument("--python-text", default="", help="Path to Python source eval text (conversion/base.py)")
    ap.add_argument("--bridge-eval", default="", help="Path to eval.txt (harness/build_eval_text.py) to run P7 bridge mode")
    ap.add_argument("--arms", default="", help="Comma-separated subset of arms (default: full PREREG list)")
    ap.add_argument("--ngl", type=int, default=999, help="GPU layers for llama.cpp (-ngl, default 999 for Apple Silicon Metal)")
    ap.add_argument("--chunks", type=int, default=40, help="Eval chunks (default 40 -> 10,200 scored positions)")
    ap.add_argument("--imatrix-chunks", type=int, default=100, help="Calibration chunks for llama-imatrix (default 100)")
    ap.add_argument("--ctx", type=int, default=512, help="Context size per chunk (default 512)")
    ap.add_argument("--batch", type=int, default=512, help="Batch size (default 512)")
    ap.add_argument("--threads", type=int, default=4, help="CPU threads for llama.cpp (default 4)")
    ap.add_argument("--workers", type=int, default=4, help="Worker processes for matrix rounding (default 4)")
    ap.add_argument("--stream-arms", action="store_true",
                    help="Write ball-arm GGUFs from per-tensor .npy sidecars (bounded parent RAM for 27B-class models); byte-identical to the default path")
    ap.add_argument("--pilot-only", action="store_true", help="Run Step 2/3 one-layer pilot and zero-codeword verification only")
    ap.add_argument("--pilot-layer", type=int, default=0, help="Layer index for --pilot-only (default 0)")
    ap.add_argument("--metrics-only", action="store_true", help="Compute weight metrics only without calling llama-perplexity")
    ap.add_argument("--dry-run", action="store_true", help="Print planned commands and arms without running llama.cpp or writing results")
    ap.add_argument("--llama-perplexity", default="llama-perplexity", help="Path to llama-perplexity binary")
    ap.add_argument("--llama-imatrix", default="llama-imatrix", help="Path to llama-imatrix binary")
    ap.add_argument("--llama-quantize", default="llama-quantize", help="Path to llama-quantize binary")
    return ap.parse_args(argv)


def run_sweep(args):
    bridge_mode = bool(args.bridge_eval)
    if args.arms:
        arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    else:
        arms = BRIDGE_ARMS if bridge_mode else DEFAULT_ARMS

    if bridge_mode:
        text_specs = [("bridge", args.bridge_eval)]
    else:
        requested = [t.strip() for t in args.texts.split(",") if t.strip()]
        text_map = {"wikitext": args.wikitext_test, "python": args.python_text}
        text_specs = []
        for tname in requested:
            if tname not in text_map:
                raise SystemExit(f"unknown text {tname!r}; expected 'wikitext' or 'python'")
            if not text_map[tname] and not args.dry_run and not args.metrics_only:
                raise SystemExit(f"missing text path for {tname}: pass --{tname}-test / --python-text")
            text_specs.append((tname, text_map[tname]))

    cache_dir = os.path.join(args.out_dir, "cache")
    work_dir = os.path.join(args.out_dir, "work")
    jsonl_path = os.path.join(args.out_dir, "results.jsonl")
    out_path = os.path.join(args.out_dir, "results.out")

    imatrix_real = os.path.join(cache_dir, f"{args.model_id}.wikitext.imatrix.gguf")
    rand_txt = os.path.join(cache_dir, f"{args.model_id}.rand.txt")
    imatrix_rand = os.path.join(cache_dir, f"{args.model_id}.rand.imatrix.gguf")
    baseline_paths = {
        "file:Q4_0": os.path.join(cache_dir, f"{args.model_id}-Q4_0.gguf"),
        "file:Q4_K_M": os.path.join(cache_dir, f"{args.model_id}-Q4_K_M.gguf"),
        "file:Q4_K_M-imat": os.path.join(cache_dir, f"{args.model_id}-Q4_K_M-imat.gguf"),
    }

    if args.dry_run:
        plan = {
            "mode": "bridge" if bridge_mode else "sweep",
            "model_id": args.model_id,
            "model_path": args.model_path,
            "out_dir": args.out_dir,
            "ngl": args.ngl,
            "texts": [{"name": tn, "path": tp, "kld": os.path.join(cache_dir, f"{args.model_id}.{tn}.kld")} for tn, tp in text_specs],
            "imatrix_real": imatrix_real,
            "imatrix_rand": imatrix_rand,
            "arms": list(arms),
            "kld_cmds": [
                ensure_kld(args.model_path, tp or f"<{tn}>", os.path.join(cache_dir, f"{args.model_id}.{tn}.kld"),
                           args.llama_perplexity, args.chunks, args.ctx, args.batch, args.ngl, args.threads, dry_run=True)
                for tn, tp in text_specs
            ],
        }
        print(json.dumps(plan, indent=2), flush=True)
        return plan

    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)

    ggufedit = _load_ggufedit()
    import evalkl

    model_sha = sha256_of(args.model_path)
    rd = ggufedit.open_model(args.model_path)
    all_targets = _targets(rd)
    if not all_targets:
        raise RuntimeError(f"no block matrices found in {args.model_path}")
    names = [t.name for t in all_targets]
    n_layers = len({int(BLOCK.match(n).group(1)) for n in names})
    all_tensors = list(rd.tensors)

    # Step 3 safety gate before scoring: verify zero-codeword preservation on layer 0
    if not bridge_mode:
        verify_zero_codeword_matrix(_matrix_of(all_targets[0]))

    needs_real_imat = any(arm_needs_real_imat(a) for a in arms)
    needs_rand_imat = any(a.endswith("~rand") for a in arms)

    imat_real_sha = None
    if needs_real_imat:
        if not os.path.exists(imatrix_real):
            if not args.wikitext_train:
                raise SystemExit("--wikitext-train is required to build real-text imatrix")
            ensure_imatrix(args.model_path, args.wikitext_train, imatrix_real, args.llama_imatrix,
                           args.imatrix_chunks, args.ctx, args.batch, args.ngl, args.threads)
        imat_real_sha = sha256_of(imatrix_real)

    imat_rand_sha = None
    if needs_rand_imat:
        if not os.path.exists(rand_txt):
            build_rand_text(args.model_path, rand_txt, pieces_count=80000, seed=1)
        if not os.path.exists(imatrix_rand):
            ensure_imatrix(args.model_path, rand_txt, imatrix_rand, args.llama_imatrix,
                           args.imatrix_chunks, args.ctx, args.batch, args.ngl, args.threads)
        imat_rand_sha = sha256_of(imatrix_rand)

    # Ensure quantized baselines if requested
    if "file:Q4_0" in arms:
        ensure_baseline_gguf(args.model_path, baseline_paths["file:Q4_0"], "Q4_0", args.llama_quantize, all_tensors, threads=args.threads)
    if "file:Q4_K_M" in arms:
        ensure_baseline_gguf(args.model_path, baseline_paths["file:Q4_K_M"], "Q4_K_M", args.llama_quantize, all_tensors, threads=args.threads)
    if "file:Q4_K_M-imat" in arms:
        ensure_baseline_gguf(args.model_path, baseline_paths["file:Q4_K_M-imat"], "Q4_K_M", args.llama_quantize, all_tensors, imatrix_path=imatrix_real, threads=args.threads)
    # @16 arms decode the stock quantized file to 16-bit block weights; they need that file built first
    for arm in arms:
        mm = STOCK16_RE.match(arm)
        if not mm:
            continue
        base_arm = "file:" + mm.group(1)
        if base_arm == "file:Q4_K_M-imat":
            ensure_baseline_gguf(args.model_path, baseline_paths[base_arm], "Q4_K_M", args.llama_quantize, all_tensors, imatrix_path=imatrix_real, threads=args.threads)
        else:
            ensure_baseline_gguf(args.model_path, baseline_paths[base_arm], base_arm.split(":", 1)[1], args.llama_quantize, all_tensors, threads=args.threads)

    # Load already-completed (model_id, text, variant) keys for clean resume
    done = set()
    if os.path.exists(jsonl_path):
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    done.add((r.get("model_id"), r.get("text"), r.get("variant")))

    # Ensure .kld references for all requested texts
    kld_info = {}
    for tname, tpath in text_specs:
        kld_path = os.path.join(cache_dir, f"{args.model_id}.{tname}.kld")
        if not args.metrics_only:
            ensure_kld(args.model_path, tpath, kld_path, args.llama_perplexity,
                       args.chunks, args.ctx, args.batch, args.ngl, args.threads)
        kld_info[tname] = {
            "kld_path": kld_path,
            "text_sha256": sha256_of(tpath) if tpath and os.path.exists(tpath) else None,
        }

    # Iterate arms; for each arm, build rounded GGUF once and score on all pending texts before deleting
    for arm in arms:
        pending_texts = [tn for tn, _ in text_specs if (args.model_id, tn, arm) not in done]
        if not pending_texts:
            print(f"[skip] {args.model_id} {arm}: already completed on all texts", file=sys.stderr, flush=True)
            continue

        t0 = time.time()
        base_row = {
            "model_id": args.model_id,
            "model_sha256": model_sha,
            "variant": arm,
            "ngl": args.ngl,
            "backend": "metal" if (platform.system() == "Darwin" and args.ngl > 0) else "cpu",
            "threads": args.threads,
            "bridge": bridge_mode,
            "stream_arms": bool(args.stream_arms),
        }
        if arm_needs_real_imat(arm):
            base_row["imatrix_sha256"] = imat_real_sha
        elif arm.endswith("~rand"):
            base_row["imatrix_sha256"] = imat_rand_sha

        made_temp = False
        if arm in baseline_paths:
            eval_path = baseline_paths[arm]
            base_row.update(compare_file(rd, eval_path, args.model_id, n_layers))
            base_row["file_sha256"] = sha256_of(eval_path)
            base_row["build_seconds"] = round(time.time() - t0, 1)
        elif STOCK16_RE.match(arm):
            # Decode the stock quantized file's block matrices to 16-bit floats in a copy of the
            # 16-bit model: the same weights the `file:` arm scores, multiplied as 16-bit weights.
            mm = STOCK16_RE.match(arm)
            base_arm = "file:" + mm.group(1)
            q_path = baseline_paths[base_arm]
            eval_path = os.path.join(work_dir, re.sub(r"[^A-Za-z0-9_.+-]", "_", f"{args.model_id}_{arm}") + ".gguf")
            made_temp = True
            qmap = {t.name: t for t in ggufedit.open_model(q_path).tensors}
            ref = {t.name: t for t in rd.tensors}
            replace, infos = {}, {}
            for n in names:
                W = _matrix_of(ref[n])
                qt = qmap[n]
                Wh = ggufedit.tensor_float(qt).reshape(W.shape)
                bits = 8.0 * qt.data.nbytes / W.size
                Wh16, _ = finish(W, Wh, bits, 0.0)
                replace[n] = Wh16
                W64 = np.asarray(W, dtype=np.float64)
                energy = float(np.sum(W64 ** 2))
                rel = float(np.sum((W64 - Wh16.astype(np.float64)) ** 2) / energy) if energy > 0 else 0.0
                infos[n] = {"rel_err": rel, "bits": bits, "overload": 0.0, "energy": energy,
                            "size": int(W.size), "kept_columns": 0, "imatrix_fallback": False}
            base_row.update(stats_of(infos, args.model_id, n_layers))
            base_row["source_file_sha256"] = sha256_of(q_path)
            base_row["build_seconds"] = round(time.time() - t0, 1)
            if not args.metrics_only:
                ggufedit.write_copy(rd, eval_path, replace)
                base_row["other_tensors_that_differ"] = compare_file(rd, eval_path, args.model_id, n_layers)["other_tensors_that_differ"]
                base_row["file_sha256"] = sha256_of(eval_path)
            del replace
        else:
            eval_path = os.path.join(work_dir, re.sub(r"[^A-Za-z0-9_.+-]", "_", f"{args.model_id}_{arm}") + ".gguf")
            made_temp = True
            replace, infos, replace_files = {}, {}, {}
            tensor_dir = os.path.join(work_dir, "tensors") if args.stream_arms else None
            if tensor_dir:
                os.makedirs(tensor_dir, exist_ok=True)
            initargs = (args.model_path,
                        imatrix_real if needs_real_imat else None,
                        imatrix_rand if needs_rand_imat else None,
                        tensor_dir)
            if args.workers <= 1:
                _worker_init(*initargs)
                results = [_round_one_worker((arm, n)) for n in names]
            else:
                with Pool(args.workers, initializer=_worker_init, initargs=initargs) as pool:
                    results = list(pool.imap_unordered(_round_one_worker, [(arm, n) for n in names], chunksize=1))
            for name, Wh, info, sidecar in results:
                infos[name] = info
                if sidecar:
                    replace_files[name] = sidecar
                else:
                    replace[name] = Wh
            base_row.update(stats_of(infos, args.model_id, n_layers))
            base_row["build_seconds"] = round(time.time() - t0, 1)
            if not args.metrics_only:
                if args.stream_arms:
                    ggufedit.write_copy_streaming(rd, eval_path, replace_files)
                else:
                    ggufedit.write_copy(rd, eval_path, replace)
            if tensor_dir and os.path.isdir(tensor_dir):
                shutil.rmtree(tensor_dir, ignore_errors=True)
            del replace

        try:
            for tname in pending_texts:
                row = dict(base_row)
                row["text"] = tname
                row["text_sha256"] = kld_info[tname]["text_sha256"]
                row["timestamp_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                if not args.metrics_only:
                    t1 = time.time()
                    scores, _ = evalkl.score(
                        eval_path,
                        kld_info[tname]["kld_path"],
                        binary=args.llama_perplexity,
                        extra=("-ngl", str(args.ngl), "-t", str(args.threads)),
                    )
                    row.update(scores)
                    row["score_seconds"] = round(time.time() - t1, 1)
                with open(jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                done.add((args.model_id, tname, arm))
                render_results_out(jsonl_path, out_path)
                print(json.dumps(row), flush=True)
        finally:
            if made_temp and os.path.exists(eval_path):
                os.remove(eval_path)


def main(argv=None):
    args = parse_args(argv)
    if args.pilot_only:
        return run_pilot(args)
    return run_sweep(args)


if __name__ == "__main__":
    main()
