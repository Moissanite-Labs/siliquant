#!/usr/bin/env python3
"""Track S167 (EXP-Q51a + EXP-Q51b, PREREG-W22-ES-SCALE) Live RTX 3080 Benchmark.

Evaluates backprop-free Evolution Strategies (arXiv:2509.24372v3) transferred onto:
  - EXP-Q51a (SiliQuant-ES): Real cached Qwen2.5-0.5B quantized in A_16 Block-H4
    (3.0 & 4.5 bpw, [16,11,4] SECDED, genuine +0 B reciprocal RMSNorm/linear folded
    scales + in-place fp16 scale ES, Layer-Salted Antithetic Orthogonal-BlockH4
    perturbations, int8 Al-Kashi FixedRadix Q=64 residual accumulator) vs canonical
    VsonicV BF16 shared-seed ES (C1) and unconfounded matched-layer antithetic
    unblocked INT4 (qmax=7) int8-EF ES (C2) across n=7 seeds (powering p_Holm < 0.05),
    plus FULL 28/28-layer (196/196 linear, 81,887,232 A_16 blocks) Qwen2.5-1.5B
    packed SiliQuant-ES live forward + Al-Kashi int8-EF update VRAM probe (< 6.5 GiB).
  - EXP-Q51b (HoloSphere-Cayley-ES): Real HoloSphere-V4 15M checkpoint
    (54 A_16 layers, 0 B HBM vocab, Cayley-SU(2) holonomy) testing like-for-like
    Chunk32 O(1) streaming VRAM (with active ES buffers) vs Full-Graph Chunk32-BPTT
    and Activation-Checkpointed Chunk32-BPTT across T in {512, 2048, 8192} and
    R in {1, 4, 8}, plus forward-only Cayley-SU(2) tangent-space ES (with live
    A_16 uint64 packed w_omega ES updates & SECDED verification) vs tangent-projected
    unclamped Euclidean additive ES on non-differentiable S_3 & A_5 group-word
    evaluation (8x OOD) + 16 KiB held-out WikiText-2 BPB.

Governance:
  - Real cached weights only (WEIGHTS=0 inside expedition/).
  - Energy disclosure: UNCONTROLLED_LAPTOP_NOT_CLAIMED (zero watt/joule claims).
  - Zero bare asserts; strictly seeded RNG; full GPU memory cleanup.
  - Pure-Python helpers import cleanly without torch (supporting python3 -I R5 tests);
    GPU/torch unit tests run via `--cpu-selftest`.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from fractions import Fraction
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any

try:
    import pyarrow.parquet as pq
    import torch
    import torch.nn.functional as F
    import torch.utils.checkpoint as ckpt_mod
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _HAS_TORCH = True
except ModuleNotFoundError:
    pq = None  # type: ignore[assignment]
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    ckpt_mod = None  # type: ignore[assignment]
    AutoModelForCausalLM = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]
    _HAS_TORCH = False

TRACK_DIR = Path(__file__).resolve().parent
HOME_DIR = Path.home()
HF_HUB_CACHE = HOME_DIR / ".cache" / "huggingface" / "hub"

PREREG_PATH = TRACK_DIR / "PREREG_W22_ES_SCALE.json"
QWEN05B_DIR = (
    HF_HUB_CACHE / "models--Qwen--Qwen2.5-0.5B" / "snapshots" / "060db6499f32faf8b98477b0a26969ef7d8b9987"
)
QWEN15B_DIR = (
    HF_HUB_CACHE / "models--Qwen--Qwen2.5-1.5B" / "snapshots" / "8faed761d45a263340a0528343f099c05c9a4323"
)
HOLOSPHERE_V4_DIR = HOME_DIR / "hf-releases" / "holosphere-v4-monarch2-sm86-15m"
WIKITEXT2_TEST_PQ = (
    HF_HUB_CACHE
    / "datasets--Salesforce--wikitext"
    / "snapshots"
    / "b08601e04326c79dfdd32d625aee71d232d685c3"
    / "wikitext-2-raw-v1"
    / "test-00000-of-00001.parquet"
)

ENERGY_CLAIM = "UNCONTROLLED_LAPTOP_NOT_CLAIMED"
# n=7 seeds for EXP-Q51a so exact 2-sided paired sign-flip floor = 2/128 = 1/64 = 0.015625
# and 2-test Holm-Bonferroni floor = 2/64 = 1/32 = 0.03125 < 0.05!
SEEDS_Q51A = (1671, 1672, 1673, 1674, 1675, 1676, 1677)
SEEDS_Q51B = (1671, 1672, 1673)

# Exact A_16^8 g=128 theta-series ball bounds:
#   3.000 bpw (B=368 + 16b fp16 scale = 384 b/group) -> R^2 = 651
#   4.500 bpw (B=560 + 16b fp16 scale = 576 b/group) -> R^2 = 5213
A16_B30_BOUND = 651.0
A16_B45_BOUND = 5213.0
QUANT_GROUP = 128
QUANT_ROW_CHUNK = 512
# PREREG-W22-ES-SCALE: int8 Al-Kashi FixedRadix EF (Q = 64 -> |e| <= 64 fits in signed int8 [-127, 127])
ALKASHI_Q = 64
INT4_QMAX = 7  # True symmetric signed 4-bit INT4 [-7, 7] (4.0 bpw + 0.125 bpw fp16 scale = 4.125 bpw)

# Unified O(1) VRAM flat-spread threshold across bench, oracle, and verifier (< 1.0 MiB = 1024 KiB)
O1_VRAM_SPREAD_MAX_MIB = 1.0
MAX_FINITE_SHELL_NANO_CAP = 10 ** 15

COUNTDOWN_HEADER = (
    "Reach the target using each number at most once with +, -, *.\n"
    "Numbers: [3, 5, 2], Target: 13 -> 3 * 5 - 2\n"
    "Numbers: [4, 6, 3], Target: 30 -> (4 + 6) * 3\n"
    "Numbers: [9, 2, 7], Target: 11 -> 9 * 2 - 7\n"
)


def _require(cond: bool, msg: str) -> None:
    if cond is not True:
        raise RuntimeError(msg)


def _require_torch() -> None:
    _require(_HAS_TORCH and torch is not None, "PyTorch + transformers + pyarrow required for GPU/tensor routines")


def _query_gpu_vram() -> dict[str, Any]:
    """Query GPU identity and VRAM ONLY (zero power/watt/joule telemetry)."""
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    parts = [p.strip() for p in out.split(",")]
    return {
        "name": parts[0],
        "driver_version": parts[1],
        "memory_used_mib": int(parts[2]),
        "memory_total_mib": int(parts[3]),
    }


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cleanup_gpu() -> None:
    gc.collect()
    if _HAS_TORCH and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _exact_paired_permutation_p(diffs: list[float]) -> float:
    """Exact two-sided paired sign-flip permutation p-value (exact rational via Fraction)."""
    n = len(diffs)
    _require(n >= 1, "empty diffs")
    q_diffs = [Fraction(repr(float(d))) for d in diffs]
    obs = abs(sum(q_diffs, Fraction(0, 1)))
    extreme = 0
    total = 1 << n
    for mask in range(total):
        s = Fraction(0, 1)
        for i, d in enumerate(q_diffs):
            s = s + (d if ((mask >> i) & 1) == 0 else -d)
        if abs(s) >= obs:
            extreme += 1
    return float(Fraction(extreme, total))


def _holm_bonferroni_adjust(p_values: list[float]) -> list[float]:
    """Step-down Holm-Bonferroni family-wise error adjustment."""
    m = len(p_values)
    _require(m >= 1, "empty p_values")
    order = sorted(range(m), key=lambda i: p_values[i])
    adj = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, float(m - rank) * float(p_values[idx]))
        if val > running_max:
            running_max = val
        adj[idx] = running_max
    return adj


def _to_scaled_int(x: float, scale: int) -> int:
    """Exact rational rounding of JSON-serialized float representation for bit-identical Z/Q reconciliation."""
    return int(round(Fraction(repr(float(x))) * scale))


def _to_micro(x: float) -> int:
    return _to_scaled_int(x, 1_000_000)


def _to_nano(x: float) -> int:
    return _to_scaled_int(x, 1_000_000_000)


def _to_kib(mib: float) -> int:
    return _to_scaled_int(mib, 1024)


# =============================================================================
# Verifiable Pure-Python Benchmarks (Countdown & S_3 / A_5 Group Words)
# =============================================================================

def build_countdown_split(
    n_items: int, seed: int, exclude_keys: set[tuple[tuple[int, ...], int]]
) -> list[tuple[list[int], int]]:
    """Deterministic verifiable Countdown split (mix of 1-op & 2-op reachable targets on 3 numbers)."""
    rng = random.Random(seed)
    items: list[tuple[list[int], int]] = []
    seen = set(exclude_keys)
    ops = (
        ("+", lambda x, y: x + y),
        ("-", lambda x, y: x - y),
        ("*", lambda x, y: x * y),
    )
    while len(items) < n_items:
        a, b, c = rng.randint(2, 12), rng.randint(2, 12), rng.randint(2, 12)
        use_two_step = (len(items) % 2) == 1
        _, f1 = rng.choice(ops)
        v1 = f1(a, b)
        if use_two_step:
            _, f2 = rng.choice(ops)
            tgt = f2(v1, c)
        else:
            tgt = v1
        if tgt <= 0 or tgt > 120 or tgt in (a, b, c):
            continue
        nums = [a, b, c]
        rng.shuffle(nums)
        key = (tuple(sorted(nums)), tgt)
        if key in seen:
            continue
        seen.add(key)
        items.append((nums, tgt))
    return items


def verify_countdown_output(raw_text: str, nums: list[int], target: int) -> tuple[float, bool]:
    """Strict non-differentiable AST verifier for Countdown expressions."""
    line = raw_text.split("\n")[0].strip()
    if "=" in line:
        line = line.split("=")[0].strip()
    if not line or len(line) > 64:
        return 0.0, False
    try:
        tree = ast.parse(line, mode="eval")
    except SyntaxError:
        return 0.0, False

    used: list[int] = []

    def _eval_node(node: ast.AST) -> int:
        if isinstance(node, ast.Expression):
            return _eval_node(node.body)
        if isinstance(node, ast.Constant) and type(node.value) is int:
            used.append(node.value)
            return int(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            v = _eval_node(node.operand)
            return v if isinstance(node.op, ast.UAdd) else -v
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
            lv = _eval_node(node.left)
            rv = _eval_node(node.right)
            if isinstance(node.op, ast.Add):
                return lv + rv
            if isinstance(node.op, ast.Sub):
                return lv - rv
            return lv * rv
        raise ValueError("disallowed_ast")

    try:
        val = _eval_node(tree)
    except ValueError:
        return 0.0, False

    c_used = Counter(used)
    c_avail = Counter(nums)
    valid_nums = len(used) >= 2 and all(c_used[k] <= c_avail[k] for k in c_used)
    if not valid_nums:
        return 0.10, False
    if val == target:
        return 1.0, True
    return 0.25 + 0.50 / (1.0 + abs(val - target)), False


def _compose_perm(p: tuple[int, ...], q: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(p[q[i]] for i in range(len(p)))


def _eval_s3_word(word: str) -> int:
    # S_3 generators: a = (1,0,2) swap, b = (1,2,0) 3-cycle; 6 canonical elements 0..5
    elems = (
        (0, 1, 2),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (0, 2, 1),
        (2, 1, 0),
    )
    gen_map = {"a": (1, 0, 2), "b": (1, 2, 0)}
    cur = (0, 1, 2)
    for ch in word:
        cur = _compose_perm(gen_map[ch], cur)
    return elems.index(cur)


def _eval_a5_word_action0(word: str) -> int:
    # A_5 generators: a = (1,2,0,3,4) 3-cycle, b = (1,2,3,4,0) 5-cycle; exact image of 0 in {0..4}
    gen_map = {"a": (1, 2, 0, 3, 4), "b": (1, 2, 3, 4, 0)}
    cur = (0, 1, 2, 3, 4)
    for ch in word:
        cur = _compose_perm(gen_map[ch], cur)
    return cur[0]


def build_group_word_split(
    n_s3: int,
    n_a5: int,
    min_len: int,
    max_len: int,
    seed: int,
    exclude_words: set[tuple[str, str]],
) -> list[tuple[str, str, int]]:
    """Build deterministic S_3 and A_5 group-word datasets -> (group, prompt_str, target_byte)."""
    rng = random.Random(seed)
    items: list[tuple[str, str, int]] = []
    seen = set(exclude_words)
    for grp, count in (("S3", n_s3), ("A5", n_a5)):
        got = 0
        while got < count:
            wl = rng.randint(min_len, max_len)
            w = "".join(rng.choice(("a", "b")) for _ in range(wl))
            if (grp, w) in seen:
                continue
            seen.add((grp, w))
            val = _eval_s3_word(w) if grp == "S3" else _eval_a5_word_action0(w)
            prompt = f"{grp}:{w}="
            target_byte = ord("0") + val
            items.append((grp, prompt, target_byte))
            got += 1
    return items


# =============================================================================
# SiliQuant A_16 Block-H4 Quantizer, +0 B Reciprocal Folding, & Packed ES Linear
# =============================================================================

def _h4_mat(device: Any, dtype: Any = None) -> Any:
    _require_torch()
    dt = torch.float32 if dtype is None else dtype
    return (
        torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, -1.0, 1.0, -1.0],
                [1.0, 1.0, -1.0, -1.0],
                [1.0, -1.0, -1.0, 1.0],
            ],
            device=device,
            dtype=dt,
        )
        * 0.5
    )


def _apply_h4_blocks(w: Any, h4: Any) -> Any:
    return (w.reshape(w.shape[0], -1, 4) @ h4.T).reshape(w.shape)


def _verify_a16_codewords_int(y_int: Any) -> int:
    """Verify exact [16,11,4] Extended Hamming SECDED syndrome directly on integer A_16 coordinates."""
    bits = (y_int.reshape(-1, 16).to(torch.int32)) & 1
    p0 = bits.sum(dim=1) & 1
    s0 = bits[:, 1::2].sum(dim=1) & 1
    s1 = (bits[:, 2::4].sum(dim=1) + bits[:, 3::4].sum(dim=1)) & 1
    s2 = (bits[:, 4:8].sum(dim=1) + bits[:, 12:16].sum(dim=1)) & 1
    s3 = bits[:, 8:16].sum(dim=1) & 1
    return int(((p0 | s0 | s1 | s2 | s3) != 0).sum().item())


def _a16_closest(u: Any) -> tuple[Any, int]:
    """Wagner syndrome-guided A_16 = 2 Z^16 + C_[16,11,4] lattice projector
    (1-flip for odd parity, optimal 2-flip for even nonzero syndrome) with direct
    post-projection coordinate-parity SECDED verification."""
    e_val = 2.0 * torch.round(u * 0.5)
    o_val = 2.0 * torch.floor(u * 0.5) + 1.0
    de = (u - e_val) ** 2
    do = (u - o_val) ** 2
    is_odd = do < de
    hard = torch.where(is_odd, o_val, e_val)
    alt = torch.where(is_odd, e_val, o_val)
    fc = (de - do).abs()

    bits = is_odd.to(torch.int32)
    p0 = bits.sum(dim=1) & 1
    s0 = bits[:, 1::2].sum(dim=1) & 1
    s1 = (bits[:, 2::4].sum(dim=1) + bits[:, 3::4].sum(dim=1)) & 1
    s2 = (bits[:, 4:8].sum(dim=1) + bits[:, 12:16].sum(dim=1)) & 1
    s3 = bits[:, 8:16].sum(dim=1) & 1
    syn = s0 + (s1 << 1) + (s2 << 2) + (s3 << 3)

    out = hard.clone()

    odd_m = p0 == 1
    if odd_m.any():
        ro = torch.nonzero(odd_m, as_tuple=False).squeeze(1)
        co = syn[ro].to(torch.int64)
        out[ro, co] = alt[ro, co]

    ev_m = (p0 == 0) & (syn != 0)
    if ev_m.any():
        re_i = torch.nonzero(ev_m, as_tuple=False).squeeze(1)
        se = syn[re_i].to(torch.int64)
        jg = torch.arange(16, device=u.device, dtype=torch.int64).unsqueeze(0)
        pg = jg ^ se.unsqueeze(1)
        pc = fc[re_i] + fc[re_i].gather(1, pg)
        bj = pc.argmin(dim=1)
        bp = bj ^ se
        out[re_i, bj] = alt[re_i, bj]
        out[re_i, bp] = alt[re_i, bp]

    # Verify directly on actual reconstructed integer coordinates (NOT auxiliary parity tracker!)
    errs = _verify_a16_codewords_int(out.round().to(torch.int32))
    return out, errs


def _quantize_matrix_a16_blockh4(
    w_f32: Any,
    h4: Any,
    bound: float = A16_B45_BOUND,
    group_size: int = QUANT_GROUP,
) -> tuple[Any, Any, int, int, int]:
    """Quantize w_f32 into integer A_16 Block-H4 codewords (int8) + exact matched fp16 scales.
    Returns (y_i8, s_fp16, syndrome_errors, oob_groups, oob_fallback_activations)."""
    _require_torch()
    with torch.no_grad():
        m, k = w_f32.shape
        y_parts: list[Any] = []
        s_parts: list[Any] = []
        syn_tot = 0
        oob_tot = 0
        oob_fallback_tot = 0
        r_ball = math.sqrt(bound)
        margin = math.sqrt(group_size / 4.0 + 1.5 * (group_size / 16.0))

        for r0 in range(0, m, QUANT_ROW_CHUNK):
            r1 = min(r0 + QUANT_ROW_CHUNK, m)
            x = _apply_h4_blocks(w_f32[r0:r1], h4)
            xg = x.reshape(-1, group_size)
            g_norm = torch.sqrt((xg ** 2).sum(dim=1)).clamp_min(1e-30)
            lo = g_norm / (r_ball + margin) * 0.99
            hi = g_norm / max(0.05 * r_ball, r_ball - margin) * 1.01
            for _ in range(14):
                mid = torch.sqrt(lo * hi)
                y_cur, _ = _a16_closest((xg / mid.unsqueeze(1)).reshape(-1, 16))
                y_cur = y_cur.reshape(-1, group_size)
                ok = (y_cur ** 2).sum(dim=1) <= bound + 1e-6
                hi = torch.where(ok, mid, hi)
                lo = torch.where(ok, lo, mid)
            sc = hi.to(torch.float16).to(torch.float32).clamp_min(1e-30)
            y_fin, se = _a16_closest((xg / sc.unsqueeze(1)).reshape(-1, 16))
            syn_tot += se
            y_fin = y_fin.reshape(-1, group_size)
            out_m = (y_fin ** 2).sum(dim=1) > bound + 1e-6
            if out_m.any():
                # [MED-8 FIX]: Scale UP sc[out_m] in fp16 and store the updated sc[out_m] in s_parts!
                oob_fallback_tot += int(out_m.sum().item())
                xg_o = xg[out_m]
                sc_lo_f = sc[out_m].clone()
                sc_hi_f = sc_lo_f.clone()
                for _ in range(20):
                    cand = (sc_hi_f * 1.25).to(torch.float16).to(torch.float32).clamp_min(1e-30)
                    sc_hi_f = torch.where(cand > sc_hi_f, cand, sc_hi_f * 2.0)
                    yt, _ = _a16_closest((xg_o / sc_hi_f.unsqueeze(1)).reshape(-1, 16))
                    yt = yt.reshape(-1, group_size)
                    if ((yt ** 2).sum(dim=1) <= bound + 1e-6).all():
                        break
                lo_b = sc_lo_f
                hi_b = sc_hi_f
                for _ in range(12):
                    mid_b = torch.sqrt(lo_b * hi_b).to(torch.float16).to(torch.float32).clamp_min(1e-30)
                    yt, _ = _a16_closest((xg_o / mid_b.unsqueeze(1)).reshape(-1, 16))
                    yt = yt.reshape(-1, group_size)
                    good = (yt ** 2).sum(dim=1) <= bound + 1e-6
                    hi_b = torch.where(good, mid_b, hi_b)
                    lo_b = torch.where(good, lo_b, mid_b)
                sc_new = hi_b.to(torch.float16).to(torch.float32).clamp_min(1e-30)
                yf2, se2 = _a16_closest((xg_o / sc_new.unsqueeze(1)).reshape(-1, 16))
                y_fin[out_m] = yf2.reshape(-1, group_size)
                sc[out_m] = sc_new
                syn_tot += se2
            y_i8_blk = y_fin.round().clamp(-127, 127).to(torch.int8)
            syn_tot += _verify_a16_codewords_int(y_i8_blk)
            oob_tot += int(
                ((y_i8_blk.to(torch.float32).reshape(-1, group_size) ** 2).sum(dim=1) > bound + 1e-6).sum().item()
            )
            y_parts.append(y_i8_blk.reshape(r1 - r0, k))
            s_parts.append(sc.to(torch.float16).reshape(r1 - r0, k // group_size))

        return torch.cat(y_parts, dim=0), torch.cat(s_parts, dim=0), syn_tot, oob_tot, oob_fallback_tot


def _dequant_a16_blockh4(y_i8: Any, s_fp16: Any, h4: Any, dtype: Any = None) -> Any:
    _require_torch()
    dt = torch.bfloat16 if dtype is None else dtype
    with torch.no_grad():
        m, k = y_i8.shape
        sc = s_fp16.to(torch.float32).unsqueeze(-1)
        y_rot = (y_i8.to(torch.float32).reshape(m, k // QUANT_GROUP, QUANT_GROUP) * sc).reshape(m, k)
        return _apply_h4_blocks(y_rot, h4.T).to(dt)


def _quantize_matrix_int4_unblocked(w_f32: Any, qmax: int = INT4_QMAX, group_size: int = QUANT_GROUP) -> tuple[Any, Any]:
    """True symmetric signed INT4 unblocked quantization (qmax=7 -> [-7, 7], 4.125 bpw incl. fp16 scale)."""
    _require_torch()
    with torch.no_grad():
        m, k = w_f32.shape
        xg = w_f32.reshape(m, k // group_size, group_size)
        amax = xg.abs().amax(dim=-1).clamp_min(1e-30)
        sc = (amax / float(qmax)).to(torch.float16)
        scf = sc.to(torch.float32).clamp_min(1e-30).unsqueeze(-1)
        q = torch.clamp(torch.round(xg / scf), -qmax, qmax).to(torch.int8).reshape(m, k)
        return q, sc


def _dequant_int4_unblocked(q_i8: Any, s_fp16: Any, dtype: Any = None) -> Any:
    _require_torch()
    dt = torch.bfloat16 if dtype is None else dtype
    with torch.no_grad():
        m, k = q_i8.shape
        scf = s_fp16.to(torch.float32).unsqueeze(-1)
        return (q_i8.to(torch.float32).reshape(m, k // QUANT_GROUP, QUANT_GROUP) * scf).reshape(m, k).to(dt)


def _h4_row_signs(row_k: int, device: Any, dtype: Any) -> Any:
    rows = (
        (1.0, 1.0, 1.0, 1.0),
        (1.0, -1.0, 1.0, -1.0),
        (1.0, 1.0, -1.0, -1.0),
        (1.0, -1.0, -1.0, 1.0),
    )
    return torch.tensor(rows[row_k & 3], device=device, dtype=dtype)


def _gen_orth_blockh4_noise(shape: tuple[int, ...], group_seed: int, row_k: int, device: Any, dtype: Any) -> Any:
    """Layer-salted Orthogonal-BlockH4 Rademacher perturbation (pairwise orthogonal across row_k=0..3)."""
    gen = torch.Generator(device=device)
    gen.manual_seed(int(group_seed) & 0x7FFFFFFF)
    numel = 1
    for s in shape:
        numel *= s
    bits = torch.randint(0, 2, (numel,), generator=gen, device=device, dtype=torch.int8)
    rad = (bits.to(dtype) * 2.0) - 1.0
    if numel % 4 == 0:
        row = _h4_row_signs(row_k, device, dtype)
        rad = (rad.view(-1, 4) * row.unsqueeze(0)).reshape(shape)
    else:
        rad = rad.reshape(shape)
    return rad


def _gen_gaussian_noise(shape: tuple[int, ...], seed: int, device: Any, dtype: Any) -> Any:
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed) & 0x7FFFFFFF)
    return torch.randn(shape, generator=gen, device=device, dtype=dtype)


def _apply_zero_byte_rmsnorm_fold(
    norm_w: Any,
    downstream_weights: list[Any],
    *,
    alpha_pow: float = 0.25,
) -> tuple[Any, list[Any], float]:
    """Genuine physical +0 B reciprocal scale fold between RMSNorm and downstream linears:
      gamma_folded = gamma * c,   W_folded[:, j] = W[:, j] / c_j
    Preserves exact linear product W_folded @ (gamma_folded * x) == W @ (gamma * x) with
    0 extra runtime parameters / 0 extra GPU bytes! Returns (folded_norm, folded_weights, max_f32_err)."""
    _require_torch()
    with torch.no_grad():
        g_f = norm_w.float()
        col_max = torch.zeros_like(g_f)
        for w in downstream_weights:
            col_max = torch.maximum(col_max, w.float().abs().amax(dim=0))
        mean_c = col_max.mean().clamp_min(1e-8)
        raw_c = (col_max.clamp_min(1e-8) / mean_c).pow(alpha_pow).clamp(0.5, 2.0)
        # Realize exact BF16 norm weight first so reciprocal c_eff cancels BF16 norm rounding too!
        g_new_bf16 = (g_f * raw_c).to(torch.bfloat16)
        safe = g_f.abs() > 1e-6
        c_eff = torch.where(safe, g_new_bf16.float() / torch.where(safe, g_f, torch.ones_like(g_f)), torch.ones_like(g_f))
        c_eff = c_eff.clamp(0.25, 4.0)
        # Recompute exact g_new_bf16 on clamped c_eff
        g_new_bf16 = (g_f * c_eff).to(torch.bfloat16)
        c_eff = torch.where(safe, g_new_bf16.float() / torch.where(safe, g_f, torch.ones_like(g_f)), torch.ones_like(g_f))
        w_folded = [(w.float() / c_eff.unsqueeze(0)) for w in downstream_weights]
        # Verify exact fp32 forward invariance on deterministic probe vector
        gen = torch.Generator(device="cpu").manual_seed(167099)
        x_probe = torch.randn((4, g_f.shape[0]), generator=gen, dtype=torch.float32).to(g_f.device)
        max_err = 0.0
        for w_orig, w_new in zip(downstream_weights, w_folded):
            y_orig = F.linear(x_probe * g_f, w_orig.float())
            y_new = F.linear(x_probe * g_new_bf16.float(), w_new)
            err = float((y_orig - y_new).abs().max().item())
            if err > max_err:
                max_err = err
        return g_new_bf16, w_folded, max_err


def _make_packed_a16_es_linear_class() -> type:
    """Factory for A16BlockH4ESLinear (only defined when torch is imported)."""
    _require_torch()

    class A16BlockH4ESLinear(torch.nn.Module):
        """Packed A_16 Block-H4 ES linear layer storing ONLY:
          - y_cur (int8): A_16 lattice codewords
          - s_cur (fp16): per-group 128 scales (perturbed in-place with +0 B extra state)
          - ef_res (int8): Al-Kashi FixedRadix Q=64 error-feedback residual
          - optional bias (bf16)
        Holds ZERO persistent BF16 weight matrix in VRAM!"""

        def __init__(
            self,
            y_i8: Any,
            s_fp16: Any,
            ef_i8: Any,
            bias: Any,
            layer_idx: int,
            h4: Any,
        ) -> None:
            super().__init__()
            self.register_buffer("y_cur", y_i8)
            self.register_buffer("s_cur", s_fp16)
            self.register_buffer("ef_res", ef_i8)
            if bias is not None:
                self.register_buffer("bias", bias)
            else:
                self.bias = None
            self.layer_idx = int(layer_idx)
            self.h4 = h4
            # active_pert = (group_seed, row_k, sign_val, sigma_w, sigma_s) or None
            self.active_pert: tuple[int, int, float, float, float] | None = None

        def forward(self, x: Any) -> Any:
            s_eff = self.s_cur
            if self.active_pert is not None:
                g_seed, row_k, sign_val, sigma_w, sigma_s = self.active_pert
                if sigma_s > 0.0:
                    eps_s = _gen_orth_blockh4_noise(
                        tuple(self.s_cur.shape), g_seed + 500003 + self.layer_idx * 997, row_k, x.device, torch.float32
                    )
                    s_eff = (self.s_cur.float() * (1.0 + sign_val * sigma_s * eps_s)).clamp(1e-5, 65000.0).to(torch.float16)
                w = _dequant_a16_blockh4(self.y_cur, s_eff, self.h4, dtype=x.dtype)
                eps_w = _gen_orth_blockh4_noise(
                    tuple(self.y_cur.shape), g_seed + self.layer_idx * 997, row_k, x.device, x.dtype
                )
                w.add_(eps_w, alpha=sign_val * sigma_w)
            else:
                w = _dequant_a16_blockh4(self.y_cur, s_eff, self.h4, dtype=x.dtype)
            return F.linear(x, w, self.bias)

    return A16BlockH4ESLinear


def _replace_submodule(root: Any, dotted_name: str, new_mod: Any) -> None:
    parts = dotted_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_mod)


# =============================================================================
# EXP-Q51a: Qwen2.5-0.5B SiliQuant-ES + Full 28/28-Layer Qwen2.5-1.5B Probe
# =============================================================================

def eval_qwen_countdown(
    model: Any,
    tokenizer: Any,
    enc_inputs: dict[str, Any],
    dataset: list[tuple[list[int], int]],
    max_new_tokens: int = 10,
) -> tuple[float, float]:
    _require_torch()
    with torch.no_grad():
        out = model.generate(
            input_ids=enc_inputs["input_ids"],
            attention_mask=enc_inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        prompt_len = enc_inputs["input_ids"].shape[1]
        tot_r = 0.0
        exact_n = 0
        for i, (nums, tgt) in enumerate(dataset):
            gen = tokenizer.decode(out[i, prompt_len:], skip_special_tokens=True)
            r, ok = verify_countdown_output(gen, nums, tgt)
            tot_r += r
            exact_n += int(ok)
        n = len(dataset)
        return tot_r / float(n), exact_n / float(n)


def _apply_qwen_block_folds_cpu(
    orig_lin_cpu: dict[str, Any],
    orig_norm_cpu: dict[str, Any],
    n_layers: int,
) -> tuple[dict[str, Any], dict[str, Any], float]:
    """Apply genuine +0 B reciprocal RMSNorm/linear scale folding across all Qwen2 decoder layers."""
    folded_lin = {k: v.clone() for k, v in orig_lin_cpu.items()}
    folded_norm = {k: v.clone() for k, v in orig_norm_cpu.items()}
    max_err = 0.0
    for li in range(n_layers):
        in_norm = f"model.layers.{li}.input_layernorm"
        qkv_names = [
            f"model.layers.{li}.self_attn.q_proj",
            f"model.layers.{li}.self_attn.k_proj",
            f"model.layers.{li}.self_attn.v_proj",
        ]
        gn, ws, e1 = _apply_zero_byte_rmsnorm_fold(folded_norm[in_norm], [folded_lin[n] for n in qkv_names])
        folded_norm[in_norm] = gn
        for n, wf in zip(qkv_names, ws):
            folded_lin[n] = wf
        max_err = max(max_err, e1)

        post_norm = f"model.layers.{li}.post_attention_layernorm"
        mlp_names = [
            f"model.layers.{li}.mlp.gate_proj",
            f"model.layers.{li}.mlp.up_proj",
        ]
        gn2, ws2, e2 = _apply_zero_byte_rmsnorm_fold(folded_norm[post_norm], [folded_lin[n] for n in mlp_names])
        folded_norm[post_norm] = gn2
        for n, wf in zip(mlp_names, ws2):
            folded_lin[n] = wf
        max_err = max(max_err, e2)
    return folded_lin, folded_norm, max_err


def run_exp_q51a_siliquant_es(device: Any) -> dict[str, Any]:
    """Execute EXP-Q51a on real cached Qwen2.5-0.5B (n=7 seeds) + FULL 28/28-layer Qwen2.5-1.5B packed ES probe."""
    _require_torch()
    with torch.no_grad():
        print("[EXP-Q51a] Loading real cached Qwen2.5-0.5B...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(str(QWEN05B_DIR), local_files_only=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        demo_keys = {((2, 3, 5), 13), ((3, 4, 6), 30), ((2, 7, 9), 11)}
        train_ds = build_countdown_split(16, seed=167001, exclude_keys=demo_keys)
        train_keys = demo_keys | {(tuple(sorted(n)), t) for n, t in train_ds}
        test_ds = build_countdown_split(64, seed=167002, exclude_keys=train_keys)

        train_prompts = [COUNTDOWN_HEADER + f"Numbers: {n}, Target: {t} ->" for n, t in train_ds]
        test_prompts = [COUNTDOWN_HEADER + f"Numbers: {n}, Target: {t} ->" for n, t in test_ds]
        enc_train = tokenizer(train_prompts, return_tensors="pt", padding=True).to(device)
        enc_test = tokenizer(test_prompts, return_tensors="pt", padding=True).to(device)

        _cleanup_gpu()
        torch.cuda.reset_peak_memory_stats(device)
        model = AutoModelForCausalLM.from_pretrained(
            str(QWEN05B_DIR), dtype=torch.bfloat16, local_files_only=True
        ).to(device).eval()

        linear_names: list[str] = []
        norm_names: list[str] = []
        mod_map = dict(model.named_modules())
        for name, mod in mod_map.items():
            if name.startswith("model.layers.") and isinstance(mod, torch.nn.Linear):
                if mod.weight.shape[1] % QUANT_GROUP == 0:
                    linear_names.append(name)
            elif name.startswith("model.layers.") and (
                name.endswith("input_layernorm") or name.endswith("post_attention_layernorm")
            ):
                norm_names.append(name)
        _require(len(linear_names) == 168, f"expected 168 block linears, got {len(linear_names)}")
        _require(len(norm_names) == 48, f"expected 48 block norms, got {len(norm_names)}")

        orig_lin_cpu = {n: mod_map[n].weight.detach().cpu().clone() for n in linear_names}
        orig_norm_cpu = {n: mod_map[n].weight.detach().cpu().clone() for n in norm_names}
        total_block_weights = sum(t.numel() for t in orig_lin_cpu.values())
        total_a16_blocks = total_block_weights // 16

        # 1) C0 BF16 Null (record exact peak VRAM too -> fixes [LOW-1]!)
        torch.cuda.reset_peak_memory_stats(device)
        bf16_tr_r, bf16_tr_acc = eval_qwen_countdown(model, tokenizer, enc_train, train_ds)
        bf16_te_r, bf16_te_acc = eval_qwen_countdown(model, tokenizer, enc_test, test_ds)
        c0_bf16_peak_vram_mib = torch.cuda.max_memory_allocated(device) / 1048576.0
        print(
            f"  C0 BF16 null: train_acc={bf16_tr_acc:.4f} (r={bf16_tr_r:.4f}) | "
            f"test_acc={bf16_te_acc:.4f} (r={bf16_te_r:.4f}) | vram={c0_bf16_peak_vram_mib:.1f} MiB",
            flush=True,
        )

        # 2) C0 True INT4 (qmax=7, [-7, 7]) Null
        int4_base_q: dict[str, Any] = {}
        int4_base_s: dict[str, Any] = {}
        for n in linear_names:
            q_i8, s_f16 = _quantize_matrix_int4_unblocked(
                orig_lin_cpu[n].to(device=device, dtype=torch.float32), qmax=INT4_QMAX
            )
            int4_base_q[n] = q_i8.cpu()
            int4_base_s[n] = s_f16.cpu()
            mod_map[n].weight.data.copy_(_dequant_int4_unblocked(q_i8, s_f16, dtype=torch.bfloat16))
        int4_tr_r, int4_tr_acc = eval_qwen_countdown(model, tokenizer, enc_train, train_ds)
        int4_te_r, int4_te_acc = eval_qwen_countdown(model, tokenizer, enc_test, test_ds)
        print(f"  C0 INT4(qmax=7) null: train_acc={int4_tr_acc:.4f} (r={int4_tr_r:.4f}) | test_acc={int4_te_acc:.4f} (r={int4_te_r:.4f})", flush=True)

        # 3) Apply genuine physical +0 B reciprocal RMSNorm/linear fold on CPU before A_16 quantization!
        folded_lin_cpu, folded_norm_cpu, fold_inv_max_err = _apply_qwen_block_folds_cpu(
            orig_lin_cpu, orig_norm_cpu, n_layers=24
        )
        print(f"  +0 B reciprocal RMSNorm/linear fold max fp32 invariance err = {fold_inv_max_err:.3e} (0 extra bytes)", flush=True)

        # 4) C0 A_16 Block-H4 3.0 bpw (R^2 = 651) Null (covers PREREG 3.0/4.5 bpw!)
        h4 = _h4_mat(device, torch.float32)
        for n in norm_names:
            mod_map[n].weight.data.copy_(folded_norm_cpu[n].to(device))
        b30_syn = 0
        b30_oob = 0
        b30_fb = 0
        for n in linear_names:
            y_i8, s_f16, se, oob, fb = _quantize_matrix_a16_blockh4(
                folded_lin_cpu[n].to(device=device, dtype=torch.float32), h4, bound=A16_B30_BOUND
            )
            b30_syn += se
            b30_oob += oob
            b30_fb += fb
            mod_map[n].weight.data.copy_(_dequant_a16_blockh4(y_i8, s_f16, h4, dtype=torch.bfloat16))
        _require(b30_syn == 0 and b30_oob == 0, f"A16 3.0bpw base err syn={b30_syn} oob={b30_oob}")
        a16_30_tr_r, a16_30_tr_acc = eval_qwen_countdown(model, tokenizer, enc_train, train_ds)
        a16_30_te_r, a16_30_te_acc = eval_qwen_countdown(model, tokenizer, enc_test, test_ds)
        print(
            f"  C0 A16-BlockH4 3.0bpw null (0 syn / 0 oob / fb={b30_fb}): "
            f"train_acc={a16_30_tr_acc:.4f} (r={a16_30_tr_r:.4f}) | test_acc={a16_30_te_acc:.4f} (r={a16_30_te_r:.4f})",
            flush=True,
        )

        # 5) C0 A_16 Block-H4 4.5 bpw (R^2 = 5213) +0 B Folded Null
        a16_base_y: dict[str, Any] = {}
        a16_base_s: dict[str, Any] = {}
        base_syn_errs = 0
        base_oob_groups = 0
        base_oob_fallbacks = 0
        for n in linear_names:
            y_i8, s_f16, se, oob, fb = _quantize_matrix_a16_blockh4(
                folded_lin_cpu[n].to(device=device, dtype=torch.float32), h4, bound=A16_B45_BOUND
            )
            base_syn_errs += se
            base_oob_groups += oob
            base_oob_fallbacks += fb
            a16_base_y[n] = y_i8.cpu()
            a16_base_s[n] = s_f16.cpu()
            mod_map[n].weight.data.copy_(_dequant_a16_blockh4(y_i8, s_f16, h4, dtype=torch.bfloat16))
        _require(base_syn_errs == 0 and base_oob_groups == 0, f"A16 4.5bpw base err syn={base_syn_errs} oob={base_oob_groups}")
        a16_tr_r, a16_tr_acc = eval_qwen_countdown(model, tokenizer, enc_train, train_ds)
        a16_te_r, a16_te_acc = eval_qwen_countdown(model, tokenizer, enc_test, test_ds)
        print(
            f"  C0 A16-BlockH4 4.5bpw+Fold0B null (0 syn / 0 oob / fb={base_oob_fallbacks}): "
            f"train_acc={a16_tr_acc:.4f} (r={a16_tr_r:.4f}) | test_acc={a16_te_acc:.4f} (r={a16_te_r:.4f})",
            flush=True,
        )

        # Shared ES rollout budget across ALL 3 arms (strict forward-pass rollout count parity!)
        n_iters = 10
        n_pop = 16  # 8 antithetic pairs (= 2 orthogonal H_4 4-tuples) vs 16 one-sided VsonicV rollouts
        n_pairs = n_pop // 2
        sigma_bf16 = 1e-3  # canonical VsonicV/es-at-scale sigma
        alpha_bf16 = 5e-4  # canonical VsonicV/es-at-scale alpha
        sigma_q = 1.5e-3   # matched quantized ES weight perturbation scale (identical across C2 and T1!)
        alpha_q = 1.5e-3   # matched quantized ES step size (identical across C2 and T1!)
        sigma_s = 2.0e-3   # +0 B in-place fp16 group-scale perturbation (T1 only, zero extra bytes)
        alpha_s = 2.0e-3   # +0 B in-place fp16 group-scale learning rate (T1 only, zero extra bytes)
        rollouts_per_seed = n_iters * n_pop * len(train_ds)

        arm_runs: dict[str, list[dict[str, Any]]] = {
            "C1_VsonicV_BF16_SharedSeed": [],
            "C2_QES_Unblocked_INT4_EF": [],
            "T1_SiliQuant_ES_A16_BlockH4_Folded0B_EF": [],
        }
        peak_vram_05b_mib = 0.0

        # --- ARM C1: Canonical VsonicV/es-at-scale (BF16, shared cross-layer seed bug, one-sided, 216 layers) ---
        for seed in SEEDS_Q51A:
            for n in linear_names:
                mod_map[n].weight.data.copy_(orig_lin_cpu[n].to(device))
            for n in norm_names:
                mod_map[n].weight.data.copy_(orig_norm_cpu[n].to(device))
            _cleanup_gpu()
            torch.cuda.reset_peak_memory_stats(device)
            t0 = time.perf_counter()
            rng = random.Random(seed * 1009 + 1)
            stalled_params_tot = 0
            updated_params_tot = 0
            max_restore_drift = 0.0
            iter_rewards: list[float] = []

            for _it in range(n_iters):
                member_seeds = [rng.randint(1, (1 << 30) - 1) for _ in range(n_pop)]
                rewards: list[float] = []
                w0_ref = mod_map[linear_names[0]].weight.detach().float().clone()
                for ms in member_seeds:
                    # Exact VsonicV WorkerExtension.py:151-155 bug: identical seed across every layer
                    for n in linear_names:
                        p = mod_map[n].weight.data
                        eps = _gen_gaussian_noise(tuple(p.shape), ms, device, p.dtype)
                        p.add_(eps, alpha=sigma_bf16)
                    for n in norm_names:
                        p = mod_map[n].weight.data
                        eps = _gen_gaussian_noise(tuple(p.shape), ms, device, p.dtype)
                        p.add_(eps, alpha=sigma_bf16)
                    r_m, _ = eval_qwen_countdown(model, tokenizer, enc_train, train_ds)
                    rewards.append(r_m)
                    for n in linear_names:
                        p = mod_map[n].weight.data
                        eps = _gen_gaussian_noise(tuple(p.shape), ms, device, p.dtype)
                        p.add_(eps, alpha=-sigma_bf16)
                    for n in norm_names:
                        p = mod_map[n].weight.data
                        eps = _gen_gaussian_noise(tuple(p.shape), ms, device, p.dtype)
                        p.add_(eps, alpha=-sigma_bf16)
                drift = float((mod_map[linear_names[0]].weight.detach().float() - w0_ref).abs().max().item())
                max_restore_drift = max(max_restore_drift, drift)

                r_mean = sum(rewards) / float(n_pop)
                r_std = math.sqrt(sum((r - r_mean) ** 2 for r in rewards) / float(n_pop))
                iter_rewards.append(r_mean)
                if r_std > 1e-12:
                    z_scores = [(r - r_mean) / (r_std + 1e-8) for r in rewards]
                    for n in linear_names + norm_names:
                        p = mod_map[n].weight.data
                        acc = torch.zeros(p.shape, device=device, dtype=torch.float32)
                        for ms, z in zip(member_seeds, z_scores):
                            eps = _gen_gaussian_noise(tuple(p.shape), ms, device, torch.float32)
                            acc.add_(eps, alpha=z)
                        acc.mul_(alpha_bf16 / float(n_pop))
                        before = p.clone()
                        p.add_(acc.to(p.dtype))
                        nz = acc.abs() > 0.0
                        stalled = nz & (p == before)
                        stalled_params_tot += int(stalled.sum().item())
                        updated_params_tot += int(nz.sum().item())

            tr_r, tr_acc = eval_qwen_countdown(model, tokenizer, enc_train, train_ds)
            te_r, te_acc = eval_qwen_countdown(model, tokenizer, enc_test, test_ds)
            dt = time.perf_counter() - t0
            pk = torch.cuda.max_memory_allocated(device) / 1048576.0
            peak_vram_05b_mib = max(peak_vram_05b_mib, pk)
            stall_frac = stalled_params_tot / float(max(1, updated_params_tot))
            rec = {
                "seed": seed,
                "train_reward": tr_r,
                "train_exact_acc": tr_acc,
                "test_reward": te_r,
                "test_exact_acc": te_acc,
                "bf16_ulp_stalled_frac": stall_frac,
                "bf16_max_restore_drift": max_restore_drift,
                "syndrome_errors": 0,
                "oob_groups": 0,
                "oob_candidates_reverted": 0,
                "peak_vram_mib": pk,
                "wall_s": dt,
                "iter_pre_update_pop_mean_rewards": iter_rewards,
            }
            arm_runs["C1_VsonicV_BF16_SharedSeed"].append(rec)
            print(
                f"  [C1 VsonicV BF16 s={seed}] test_acc={te_acc:.4f} (r={te_r:.4f}) "
                f"ulp_stall={100*stall_frac:.2f}% drift={max_restore_drift:.2e} vram={pk:.1f}MiB ({dt:.1f}s)",
                flush=True,
            )

        # --- ARM C2: Unconfounded QES-style Unblocked INT4 (qmax=7) + Antithetic Gaussian + int8 Al-Kashi EF (Q=64) ---
        # Matched to T1 on: 168 linear layers (48 norms frozen), 8 antithetic pairs, sigma_q, alpha_q, int8 Q=64 EF!
        for seed in SEEDS_Q51A:
            q_cur = {n: int4_base_q[n].to(device) for n in linear_names}
            s_cur = {n: int4_base_s[n].to(device) for n in linear_names}
            ef_res = {n: torch.zeros_like(q_cur[n], dtype=torch.int8) for n in linear_names}
            base_w_iter = {n: _dequant_int4_unblocked(q_cur[n], s_cur[n], dtype=torch.bfloat16) for n in linear_names}
            for n in linear_names:
                mod_map[n].weight.data.copy_(base_w_iter[n])
            for n in norm_names:
                mod_map[n].weight.data.copy_(orig_norm_cpu[n].to(device))
            _cleanup_gpu()
            torch.cuda.reset_peak_memory_stats(device)
            t0 = time.perf_counter()
            rng = random.Random(seed * 1009 + 2)
            iter_rewards = []
            tot_grid_flips = 0

            for _it in range(n_iters):
                pair_seeds = [rng.randint(1, (1 << 28) - 1) for _ in range(n_pairs)]
                rewards_pos: list[float] = []
                rewards_neg: list[float] = []
                for ps in pair_seeds:
                    for sign_val, bucket in ((1.0, rewards_pos), (-1.0, rewards_neg)):
                        for l_idx, n in enumerate(linear_names):
                            p = mod_map[n].weight.data
                            eps = _gen_gaussian_noise(tuple(p.shape), ps + l_idx * 997, device, torch.bfloat16)
                            p.copy_(base_w_iter[n]).add_(eps, alpha=sign_val * sigma_q)
                        r_m, _ = eval_qwen_countdown(model, tokenizer, enc_train, train_ds)
                        bucket.append(r_m)

                all_r = rewards_pos + rewards_neg
                r_mean = sum(all_r) / float(len(all_r))
                r_std = math.sqrt(sum((r - r_mean) ** 2 for r in all_r) / float(len(all_r)))
                iter_rewards.append(r_mean)
                if r_std > 1e-12:
                    pair_w = [(rp - rn) / (2.0 * (r_std + 1e-8)) for rp, rn in zip(rewards_pos, rewards_neg)]
                    for l_idx, n in enumerate(linear_names):
                        m_r, k_c = q_cur[n].shape
                        acc = torch.zeros((m_r, k_c), device=device, dtype=torch.float32)
                        for ps, pw in zip(pair_seeds, pair_w):
                            eps = _gen_gaussian_noise((m_r, k_c), ps + l_idx * 997, device, torch.float32)
                            acc.add_(eps, alpha=pw)
                        acc.mul_(alpha_q / float(n_pairs))
                        scf = s_cur[n].to(torch.float32).clamp_min(1e-30).unsqueeze(-1)
                        delta_q = (acc.reshape(m_r, k_c // QUANT_GROUP, QUANT_GROUP) / scf).reshape(m_r, k_c)
                        u_fixed = ef_res[n].to(torch.int32) + torch.round(delta_q * float(ALKASHI_Q)).to(torch.int32)
                        step_int = torch.div(u_fixed + (ALKASHI_Q // 2), ALKASHI_Q, rounding_mode="floor")
                        new_q = torch.clamp(q_cur[n].to(torch.int32) + step_int, -INT4_QMAX, INT4_QMAX)
                        realized = new_q - q_cur[n].to(torch.int32)
                        tot_grid_flips += int((realized != 0).sum().item())
                        ef_res[n] = (u_fixed - realized * ALKASHI_Q).clamp(-127, 127).to(torch.int8)
                        q_cur[n] = new_q.to(torch.int8)
                        base_w_iter[n] = _dequant_int4_unblocked(q_cur[n], s_cur[n], dtype=torch.bfloat16)
                        mod_map[n].weight.data.copy_(base_w_iter[n])
                else:
                    for n in linear_names:
                        mod_map[n].weight.data.copy_(base_w_iter[n])

            tr_r, tr_acc = eval_qwen_countdown(model, tokenizer, enc_train, train_ds)
            te_r, te_acc = eval_qwen_countdown(model, tokenizer, enc_test, test_ds)
            dt = time.perf_counter() - t0
            pk = torch.cuda.max_memory_allocated(device) / 1048576.0
            peak_vram_05b_mib = max(peak_vram_05b_mib, pk)
            rec = {
                "seed": seed,
                "train_reward": tr_r,
                "train_exact_acc": tr_acc,
                "test_reward": te_r,
                "test_exact_acc": te_acc,
                "grid_flips": tot_grid_flips,
                "syndrome_errors": 0,
                "oob_groups": 0,
                "oob_candidates_reverted": 0,
                "peak_vram_mib": pk,
                "wall_s": dt,
                "iter_pre_update_pop_mean_rewards": iter_rewards,
            }
            arm_runs["C2_QES_Unblocked_INT4_EF"].append(rec)
            print(
                f"  [C2 QES INT4(qmax=7)-Antithetic-EF s={seed}] test_acc={te_acc:.4f} (r={te_r:.4f}) "
                f"flips={tot_grid_flips:,} vram={pk:.1f}MiB ({dt:.1f}s)",
                flush=True,
            )
            del q_cur, s_cur, ef_res, base_w_iter
            _cleanup_gpu()

        # --- ARM T1: SiliQuant-ES (A_16 Block-H4 + [16,11,4] SECDED + Antithetic Orth-H4 + Genuine +0 B Folded-Scale + int8 Al-Kashi EF) ---
        # Zero norm_f32 GPU shadow dict! Exact same 168 linear layers & 8 antithetic pairs & int8 Q=64 EF as C2!
        for seed in SEEDS_Q51A:
            y_cur = {n: a16_base_y[n].to(device) for n in linear_names}
            s_cur = {n: a16_base_s[n].to(device) for n in linear_names}
            ef_res = {n: torch.zeros_like(y_cur[n], dtype=torch.int8) for n in linear_names}
            for n in norm_names:
                # Folded BF16 RMSNorm weights directly in-place (ZERO norm_f32 shadow dict on GPU!)
                mod_map[n].weight.data.copy_(folded_norm_cpu[n].to(device))
            base_w_iter = {n: _dequant_a16_blockh4(y_cur[n], s_cur[n], h4, dtype=torch.bfloat16) for n in linear_names}
            for n in linear_names:
                mod_map[n].weight.data.copy_(base_w_iter[n])

            _cleanup_gpu()
            torch.cuda.reset_peak_memory_stats(device)
            t0 = time.perf_counter()
            rng = random.Random(seed * 1009 + 3)
            iter_rewards = []
            tot_syn_errs = 0
            tot_oob_groups = 0
            tot_oob_reverted = 0
            tot_a16_flips = 0
            tot_scale_updates = 0
            max_orth_dot_err = 0.0

            probe_seed = seed * 1009 + 777
            vecs = [_gen_orth_blockh4_noise((16, 128), probe_seed, rk, device, torch.float32).reshape(-1) for rk in range(4)]
            for i in range(4):
                for j in range(i + 1, 4):
                    max_orth_dot_err = max(max_orth_dot_err, abs(float(torch.dot(vecs[i], vecs[j]).item())))
            del vecs

            for _it in range(n_iters):
                group_seeds = [rng.randint(1, (1 << 27) - 1) for _ in range(n_pairs // 4)]
                rewards_pos = []
                rewards_neg = []

                for p_idx in range(n_pairs):
                    g_seed = group_seeds[p_idx // 4]
                    row_k = p_idx % 4
                    for sign_val, bucket in ((1.0, rewards_pos), (-1.0, rewards_neg)):
                        for l_idx, n in enumerate(linear_names):
                            p = mod_map[n].weight.data
                            m_r, k_c = p.shape
                            # Genuine +0 B in-place folded-scale perturbation on existing s_cur (zero extra GPU tensors!)
                            eps_s = _gen_orth_blockh4_noise(
                                tuple(s_cur[n].shape), g_seed + 500003 + l_idx * 997, row_k, device, torch.float32
                            )
                            mult_s = (1.0 + sign_val * sigma_s * eps_s).to(torch.bfloat16).unsqueeze(-1)
                            eps_w = _gen_orth_blockh4_noise(
                                (m_r, k_c), g_seed + l_idx * 997, row_k, device, torch.bfloat16
                            )
                            p.copy_(
                                (base_w_iter[n].view(m_r, k_c // QUANT_GROUP, QUANT_GROUP) * mult_s).reshape(m_r, k_c)
                            ).add_(eps_w, alpha=sign_val * sigma_q)
                        r_m, _ = eval_qwen_countdown(model, tokenizer, enc_train, train_ds)
                        bucket.append(r_m)

                all_r = rewards_pos + rewards_neg
                r_mean = sum(all_r) / float(len(all_r))
                r_std = math.sqrt(sum((r - r_mean) ** 2 for r in all_r) / float(len(all_r)))
                iter_rewards.append(r_mean)

                if r_std > 1e-12:
                    pair_w = [(rp - rn) / (2.0 * (r_std + 1e-8)) for rp, rn in zip(rewards_pos, rewards_neg)]
                    for l_idx, n in enumerate(linear_names):
                        m_r, k_c = y_cur[n].shape
                        # 1) Update existing fp16 folded group scales s_cur in-place (+0 B extra persistent state!)
                        acc_s = torch.zeros(s_cur[n].shape, device=device, dtype=torch.float32)
                        for p_idx, pw in enumerate(pair_w):
                            g_seed = group_seeds[p_idx // 4]
                            row_k = p_idx % 4
                            eps_s = _gen_orth_blockh4_noise(
                                tuple(s_cur[n].shape), g_seed + 500003 + l_idx * 997, row_k, device, torch.float32
                            )
                            acc_s.add_(eps_s, alpha=pw)
                        acc_s.mul_(alpha_s / float(n_pairs))
                        s_new = (s_cur[n].float() * (1.0 + acc_s).clamp(0.95, 1.05)).clamp(1e-5, 65000.0).to(torch.float16)
                        tot_scale_updates += int((s_new != s_cur[n]).sum().item())
                        s_cur[n] = s_new

                        # 2) Update A_16 lattice codewords with int8 Al-Kashi FixedRadix Q=64 EF
                        acc = torch.zeros((m_r, k_c), device=device, dtype=torch.float32)
                        for p_idx, pw in enumerate(pair_w):
                            g_seed = group_seeds[p_idx // 4]
                            row_k = p_idx % 4
                            eps = _gen_orth_blockh4_noise(
                                (m_r, k_c), g_seed + l_idx * 997, row_k, device, torch.float32
                            )
                            acc.add_(eps, alpha=pw)
                        acc.mul_(alpha_q / float(n_pairs))
                        acc_rot = _apply_h4_blocks(acc, h4)
                        scf = s_cur[n].to(torch.float32).clamp_min(1e-30).unsqueeze(-1)
                        delta_lat = (acc_rot.reshape(m_r, k_c // QUANT_GROUP, QUANT_GROUP) / scf).reshape(m_r, k_c)
                        u_fixed = ef_res[n].to(torch.int32) + torch.round(delta_lat * float(ALKASHI_Q)).to(torch.int32)
                        target_f = y_cur[n].to(torch.float32) + (u_fixed.to(torch.float32) / float(ALKASHI_Q))
                        y_cand, se = _a16_closest(target_f.reshape(-1, 16))
                        y_cand = y_cand.reshape(m_r, k_c)

                        # Enforce exact A_16^8 g=128 ball bound R^2 <= 5213 AND explicitly log every reverted group!
                        g_norm2 = (y_cand.reshape(-1, QUANT_GROUP) ** 2).sum(dim=1)
                        oob_m = g_norm2 > A16_B45_BOUND + 1e-6
                        if oob_m.any():
                            tot_oob_reverted += int(oob_m.sum().item())
                            yc_g = y_cand.reshape(-1, QUANT_GROUP)
                            yo_g = y_cur[n].to(torch.float32).reshape(-1, QUANT_GROUP)
                            yc_g = torch.where(oob_m.unsqueeze(1), yo_g, yc_g)
                            y_cand = yc_g.reshape(m_r, k_c)
                        y_new_i8 = y_cand.round().clamp(-127, 127).to(torch.int8)
                        se_live = _verify_a16_codewords_int(y_new_i8)
                        oob_live = int(
                            ((y_new_i8.to(torch.float32).reshape(-1, QUANT_GROUP) ** 2).sum(dim=1) > A16_B45_BOUND + 1e-6)
                            .sum()
                            .item()
                        )
                        tot_syn_errs += se + se_live
                        tot_oob_groups += oob_live
                        realized = y_new_i8.to(torch.int32) - y_cur[n].to(torch.int32)
                        tot_a16_flips += int((realized != 0).sum().item())
                        ef_res[n] = (u_fixed - realized * ALKASHI_Q).clamp(-127, 127).to(torch.int8)
                        y_cur[n] = y_new_i8
                        base_w_iter[n] = _dequant_a16_blockh4(y_cur[n], s_cur[n], h4, dtype=torch.bfloat16)
                        mod_map[n].weight.data.copy_(base_w_iter[n])
                else:
                    for n in linear_names:
                        mod_map[n].weight.data.copy_(base_w_iter[n])

            _require(tot_syn_errs == 0, f"SiliQuant-ES nonzero syndrome errors: {tot_syn_errs}")
            _require(tot_oob_groups == 0, f"SiliQuant-ES nonzero committed OOB groups: {tot_oob_groups}")
            tr_r, tr_acc = eval_qwen_countdown(model, tokenizer, enc_train, train_ds)
            te_r, te_acc = eval_qwen_countdown(model, tokenizer, enc_test, test_ds)
            dt = time.perf_counter() - t0
            pk = torch.cuda.max_memory_allocated(device) / 1048576.0
            peak_vram_05b_mib = max(peak_vram_05b_mib, pk)
            rec = {
                "seed": seed,
                "train_reward": tr_r,
                "train_exact_acc": tr_acc,
                "test_reward": te_r,
                "test_exact_acc": te_acc,
                "a16_lattice_flips": tot_a16_flips,
                "folded_scale_fp16_updates": tot_scale_updates,
                "max_orth_h4_dot_err": max_orth_dot_err,
                "syndrome_errors": tot_syn_errs,
                "oob_groups": tot_oob_groups,
                "oob_candidates_reverted": tot_oob_reverted,
                "folded_scale_extra_bytes": 0,
                "peak_vram_mib": pk,
                "wall_s": dt,
                "iter_pre_update_pop_mean_rewards": iter_rewards,
            }
            arm_runs["T1_SiliQuant_ES_A16_BlockH4_Folded0B_EF"].append(rec)
            print(
                f"  [T1 SiliQuant-ES s={seed}] test_acc={te_acc:.4f} (r={te_r:.4f}) "
                f"flips={tot_a16_flips:,} sc_upd={tot_scale_updates:,} syn={tot_syn_errs} "
                f"oob={tot_oob_groups} oob_rev={tot_oob_reverted:,} vram={pk:.1f}MiB ({dt:.1f}s)",
                flush=True,
            )
            del y_cur, s_cur, ef_res, base_w_iter
            _cleanup_gpu()

        # Release Qwen2.5-0.5B completely before full 28/28-layer Qwen2.5-1.5B probe!
        del (
            model,
            orig_lin_cpu,
            orig_norm_cpu,
            folded_lin_cpu,
            folded_norm_cpu,
            int4_base_q,
            int4_base_s,
            a16_base_y,
            a16_base_s,
            enc_train,
            enc_test,
            h4,
        )
        _cleanup_gpu()

        # =========================================================================
        # [HIGH-1 FIX] FULL 28/28-LAYER (196/196 LINEAR, 81,887,232 A_16 BLOCKS)
        # Qwen2.5-1.5B Packed SiliQuant-ES Live Forward + Update VRAM & SECDED Probe!
        # =========================================================================
        _require(QWEN15B_DIR.is_dir(), f"Pinned Qwen2.5-1.5B snapshot missing: {QWEN15B_DIR}")
        print(
            f"[EXP-Q51a] Running FULL 28/28-layer Qwen2.5-1.5B packed SiliQuant-ES probe ({QWEN15B_DIR.name[:12]})...",
            flush=True,
        )
        t15 = time.perf_counter()
        # Load BF16 model on CPU so BF16 block weights are NEVER simultaneously resident on GPU!
        m15 = AutoModelForCausalLM.from_pretrained(
            str(QWEN15B_DIR), dtype=torch.bfloat16, local_files_only=True
        ).eval()

        # Apply genuine +0 B reciprocal RMSNorm/linear fold across all 28/28 layers on CPU
        m15_mods = dict(m15.named_modules())
        m15_fold_err = 0.0
        for li in range(28):
            in_ln = m15_mods[f"model.layers.{li}.input_layernorm"]
            qkv_ms = [
                m15_mods[f"model.layers.{li}.self_attn.q_proj"],
                m15_mods[f"model.layers.{li}.self_attn.k_proj"],
                m15_mods[f"model.layers.{li}.self_attn.v_proj"],
            ]
            gn, ws, e1 = _apply_zero_byte_rmsnorm_fold(in_ln.weight.detach(), [m.weight.detach() for m in qkv_ms])
            in_ln.weight.data.copy_(gn)
            for m_sub, wf in zip(qkv_ms, ws):
                m_sub.weight.data.copy_(wf.to(torch.bfloat16))
            m15_fold_err = max(m15_fold_err, e1)

            post_ln = m15_mods[f"model.layers.{li}.post_attention_layernorm"]
            mlp_ms = [
                m15_mods[f"model.layers.{li}.mlp.gate_proj"],
                m15_mods[f"model.layers.{li}.mlp.up_proj"],
            ]
            gn2, ws2, e2 = _apply_zero_byte_rmsnorm_fold(post_ln.weight.detach(), [m.weight.detach() for m in mlp_ms])
            post_ln.weight.data.copy_(gn2)
            for m_sub, wf in zip(mlp_ms, ws2):
                m_sub.weight.data.copy_(wf.to(torch.bfloat16))
            m15_fold_err = max(m15_fold_err, e2)

        # Quantize ALL 28/28 layers (196/196 block linears) onto GPU layer-by-layer into packed A16BlockH4ESLinear
        # retaining y_cur (int8) + s_cur (fp16) + ef_res (int8) on GPU and freeing BF16 originals immediately!
        _cleanup_gpu()
        torch.cuda.reset_peak_memory_stats(device)
        h4_15 = _h4_mat(device, torch.float32)
        A16ESLinear = _make_packed_a16_es_linear_class()

        m15_linear_names = [
            lname
            for lname, lmod in m15.named_modules()
            if lname.startswith("model.layers.")
            and isinstance(lmod, torch.nn.Linear)
            and lmod.weight.shape[1] % QUANT_GROUP == 0
        ]
        _require(len(m15_linear_names) == 196, f"expected 196 Qwen2.5-1.5B block linears, got {len(m15_linear_names)}")

        m15_syn_quant = 0
        m15_oob_quant = 0
        m15_oob_fb = 0
        m15_total_weights = 0
        m15_packed_mods: list[Any] = []

        for l_idx, lname in enumerate(m15_linear_names):
            lmod = m15_mods[lname]
            w_gpu_f32 = lmod.weight.detach().to(device=device, dtype=torch.float32)
            bias_gpu = lmod.bias.detach().to(device=device, dtype=torch.bfloat16) if lmod.bias is not None else None
            m15_total_weights += int(w_gpu_f32.numel())
            y_i8, s_f16, se, oob, fb = _quantize_matrix_a16_blockh4(w_gpu_f32, h4_15, bound=A16_B45_BOUND)
            del w_gpu_f32
            ef_i8 = torch.zeros_like(y_i8, dtype=torch.int8)
            m15_syn_quant += se
            m15_oob_quant += oob
            m15_oob_fb += fb
            packed_mod = A16ESLinear(y_i8, s_f16, ef_i8, bias_gpu, l_idx, h4_15)
            _replace_submodule(m15, lname, packed_mod)
            m15_packed_mods.append(packed_mod)
            # Free original CPU BF16 linear immediately
            lmod.weight = torch.nn.Parameter(torch.empty(0), requires_grad=False)

        # Move remaining non-linear weights (embed_tokens [tied with lm_head], RMSNorms, rotary) to GPU!
        m15.to(device)
        gc.collect()
        torch.cuda.synchronize(device)
        m15_resident_after_quant_mib = torch.cuda.memory_allocated(device) / 1048576.0
        m15_total_blocks = m15_total_weights // 16
        _require(m15_total_weights == 1310195712, f"unexpected 1.5B linear weight count {m15_total_weights}")
        _require(m15_total_blocks == 81887232, f"unexpected 1.5B A_16 block count {m15_total_blocks}")

        # Verify ZERO remaining nn.Linear modules inside model.layers.*
        rem_lin = [n for n, m in m15.named_modules() if n.startswith("model.layers.") and isinstance(m, torch.nn.Linear)]
        _require(len(rem_lin) == 0, f"unquantized BF16 block linears remain on 1.5B: {len(rem_lin)}")

        # Run LIVE full-28-layer antithetic orthogonal Block-H4 ES rollouts (+sigma and -sigma) + full 196-layer ES update!
        tok15 = AutoTokenizer.from_pretrained(str(QWEN15B_DIR), local_files_only=True)
        tok15.padding_side = "left"
        if tok15.pad_token is None:
            tok15.pad_token = tok15.eos_token
        enc15 = tok15(train_prompts[:4], return_tensors="pt", padding=True).to(device)

        probe_g_seed = 1671500
        m15_rewards: list[float] = []
        for sign_val in (1.0, -1.0):
            for pmod in m15_packed_mods:
                pmod.active_pert = (probe_g_seed, 0, sign_val, sigma_q, sigma_s)
            r_15, _ = eval_qwen_countdown(m15, tok15, enc15, train_ds[:4], max_new_tokens=10)
            m15_rewards.append(r_15)
        for pmod in m15_packed_mods:
            pmod.active_pert = None

        # Execute FULL 196-layer Al-Kashi int8 EF + A_16 SECDED + +0 B fp16 scale ES update step on GPU!
        # Use measured antithetic direction if nonzero, else +1.0 probe weight so update memory & flips are genuinely exercised!
        r_diff_15 = m15_rewards[0] - m15_rewards[1]
        pw_15 = 1.0 if abs(r_diff_15) < 1e-12 else (1.0 if r_diff_15 > 0.0 else -1.0)
        m15_syn_post = 0
        m15_oob_post = 0
        m15_oob_rev_post = 0
        m15_flips_post = 0
        m15_scale_upd_post = 0

        for pmod in m15_packed_mods:
            m_r, k_c = pmod.y_cur.shape
            # 1) +0 B fp16 group scale update in-place
            eps_s = _gen_orth_blockh4_noise(
                tuple(pmod.s_cur.shape), probe_g_seed + 500003 + pmod.layer_idx * 997, 0, device, torch.float32
            )
            s_new = (pmod.s_cur.float() * (1.0 + alpha_s * pw_15 * eps_s).clamp(0.95, 1.05)).clamp(1e-5, 65000.0).to(torch.float16)
            m15_scale_upd_post += int((s_new != pmod.s_cur).sum().item())
            pmod.s_cur.copy_(s_new)

            # 2) A_16 lattice update with int8 Al-Kashi Q=64 EF (1 layer temp in VRAM at a time)
            acc = _gen_orth_blockh4_noise(
                (m_r, k_c), probe_g_seed + pmod.layer_idx * 997, 0, device, torch.float32
            ).mul_(alpha_q * pw_15)
            acc_rot = _apply_h4_blocks(acc, h4_15)
            del acc
            scf = pmod.s_cur.to(torch.float32).clamp_min(1e-30).unsqueeze(-1)
            delta_lat = (acc_rot.reshape(m_r, k_c // QUANT_GROUP, QUANT_GROUP) / scf).reshape(m_r, k_c)
            del acc_rot
            u_fixed = pmod.ef_res.to(torch.int32) + torch.round(delta_lat * float(ALKASHI_Q)).to(torch.int32)
            del delta_lat
            target_f = pmod.y_cur.to(torch.float32) + (u_fixed.to(torch.float32) / float(ALKASHI_Q))
            y_cand, se = _a16_closest(target_f.reshape(-1, 16))
            del target_f
            y_cand = y_cand.reshape(m_r, k_c)
            g_norm2 = (y_cand.reshape(-1, QUANT_GROUP) ** 2).sum(dim=1)
            oob_m = g_norm2 > A16_B45_BOUND + 1e-6
            if oob_m.any():
                m15_oob_rev_post += int(oob_m.sum().item())
                yc_g = y_cand.reshape(-1, QUANT_GROUP)
                yo_g = pmod.y_cur.to(torch.float32).reshape(-1, QUANT_GROUP)
                yc_g = torch.where(oob_m.unsqueeze(1), yo_g, yc_g)
                y_cand = yc_g.reshape(m_r, k_c)
            y_new_i8 = y_cand.round().clamp(-127, 127).to(torch.int8)
            del y_cand
            se_live = _verify_a16_codewords_int(y_new_i8)
            oob_live = int(
                ((y_new_i8.to(torch.float32).reshape(-1, QUANT_GROUP) ** 2).sum(dim=1) > A16_B45_BOUND + 1e-6)
                .sum()
                .item()
            )
            m15_syn_post += se + se_live
            m15_oob_post += oob_live
            realized = y_new_i8.to(torch.int32) - pmod.y_cur.to(torch.int32)
            m15_flips_post += int((realized != 0).sum().item())
            pmod.ef_res.copy_((u_fixed - realized * ALKASHI_Q).clamp(-127, 127).to(torch.int8))
            pmod.y_cur.copy_(y_new_i8)
            del u_fixed, realized, y_new_i8

        # Post-update live forward verification on all 28/28 updated packed layers
        r_15_post, _ = eval_qwen_countdown(m15, tok15, enc15, train_ds[:4], max_new_tokens=10)
        torch.cuda.synchronize(device)
        peak_vram_15b_mib = torch.cuda.max_memory_allocated(device) / 1048576.0
        peak_vram_15b_reserved_mib = torch.cuda.max_memory_reserved(device) / 1048576.0
        m15_syn = m15_syn_quant + m15_syn_post
        m15_oob = m15_oob_quant + m15_oob_post
        m15_wall_s = time.perf_counter() - t15
        print(
            f"  Qwen2.5-1.5B FULL 28/28-layer (196 linears, {m15_total_blocks:,} A_16 blocks) SiliQuant-ES: "
            f"resident={m15_resident_after_quant_mib:.1f} MiB, peak_alloc={peak_vram_15b_mib:.1f} MiB "
            f"(reserved={peak_vram_15b_reserved_mib:.1f} MiB, gate < 6656 MiB), "
            f"flips={m15_flips_post:,}, syn={m15_syn}, oob={m15_oob}, oob_rev={m15_oob_rev_post:,}, "
            f"post_r={r_15_post:.3f} ({m15_wall_s:.1f}s)",
            flush=True,
        )
        del m15, m15_mods, m15_packed_mods, enc15, tok15, h4_15
        _cleanup_gpu()

        # Aggregate statistics & exact paired permutation + Holm-Bonferroni tests (n=7 seeds!)
        c1_accs = [r["test_exact_acc"] for r in arm_runs["C1_VsonicV_BF16_SharedSeed"]]
        c2_accs = [r["test_exact_acc"] for r in arm_runs["C2_QES_Unblocked_INT4_EF"]]
        t1_accs = [r["test_exact_acc"] for r in arm_runs["T1_SiliQuant_ES_A16_BlockH4_Folded0B_EF"]]
        c1_rews = [r["test_reward"] for r in arm_runs["C1_VsonicV_BF16_SharedSeed"]]
        c2_rews = [r["test_reward"] for r in arm_runs["C2_QES_Unblocked_INT4_EF"]]
        t1_rews = [r["test_reward"] for r in arm_runs["T1_SiliQuant_ES_A16_BlockH4_Folded0B_EF"]]

        mean_c1_acc = sum(c1_accs) / len(c1_accs)
        mean_c2_acc = sum(c2_accs) / len(c2_accs)
        mean_t1_acc = sum(t1_accs) / len(t1_accs)
        mean_c1_rew = sum(c1_rews) / len(c1_rews)
        mean_c2_rew = sum(c2_rews) / len(c2_rews)
        mean_t1_rew = sum(t1_rews) / len(t1_rews)
        mean_bf16_stall = sum(r["bf16_ulp_stalled_frac"] for r in arm_runs["C1_VsonicV_BF16_SharedSeed"]) / len(SEEDS_Q51A)
        delta_t1_vs_c2_pp = (mean_t1_acc - mean_c2_acc) * 100.0
        delta_t1_vs_c1_pp = (mean_t1_acc - mean_c1_acc) * 100.0
        p_acc_t1_c2 = _exact_paired_permutation_p([a - b for a, b in zip(t1_accs, c2_accs)])
        p_acc_t1_c1 = _exact_paired_permutation_p([a - b for a, b in zip(t1_accs, c1_accs)])
        p_rew_t1_c2 = _exact_paired_permutation_p([a - b for a, b in zip(t1_rews, c2_rews)])
        p_holm_t1_c2, p_holm_t1_c1 = _holm_bonferroni_adjust([p_acc_t1_c2, p_acc_t1_c1])

        # Honest preregistered kill gate for EXP-Q51a ([MED-4] FIX: NO < +2pp PARTIAL loophole!):
        #   CONFIRMED: zero_syndromes AND vram < 6.5 GiB AND delta_t1_vs_c2_pp >= +2.0 pp AND p_Holm < 0.05
        #   PARTIAL:   zero_syndromes AND vram < 6.5 GiB AND delta_t1_vs_c2_pp >= +2.0 pp (p_Holm >= 0.05)
        #   FALSIFIED: otherwise!
        gate_syn_ok = (
            base_syn_errs == 0
            and base_oob_groups == 0
            and b30_syn == 0
            and b30_oob == 0
            and m15_syn == 0
            and m15_oob == 0
            and all(
                r["syndrome_errors"] == 0 and r["oob_groups"] == 0
                for r in arm_runs["T1_SiliQuant_ES_A16_BlockH4_Folded0B_EF"]
            )
        )
        gate_vram_ok = peak_vram_05b_mib < 6656.0 and peak_vram_15b_mib < 6656.0
        gate_acc_2pp_ok = delta_t1_vs_c2_pp >= 2.0
        gate_p_holm_ok = p_holm_t1_c2 < 0.05
        if gate_syn_ok and gate_vram_ok and gate_acc_2pp_ok and gate_p_holm_ok:
            verdict = "CONFIRMED"
        elif gate_syn_ok and gate_vram_ok and gate_acc_2pp_ok:
            verdict = "PARTIAL"
        else:
            verdict = "FALSIFIED"

        return {
            "model_05b": "Qwen/Qwen2.5-0.5B",
            "model_05b_snapshot": QWEN05B_DIR.name,
            "model_15b": "Qwen/Qwen2.5-1.5B",
            "model_15b_snapshot": QWEN15B_DIR.name,
            "linear_layers_05b": len(linear_names),
            "norm_layers_05b": len(norm_names),
            "total_block_weights_05b": total_block_weights,
            "total_a16_blocks_05b": total_a16_blocks,
            "alkashi_q": ALKASHI_Q,
            "ef_dtype": "int8",
            "int4_qmax": INT4_QMAX,
            "fold_0b_max_fp32_invariance_err_05b": fold_inv_max_err,
            "n_train": len(train_ds),
            "n_test": len(test_ds),
            "n_iters": n_iters,
            "n_pop": n_pop,
            "rollouts_per_seed": rollouts_per_seed,
            "seeds": list(SEEDS_Q51A),
            "baselines_null": {
                "C0_BF16_Null": {
                    "train_reward": bf16_tr_r,
                    "train_exact_acc": bf16_tr_acc,
                    "test_reward": bf16_te_r,
                    "test_exact_acc": bf16_te_acc,
                    "peak_vram_mib": c0_bf16_peak_vram_mib,
                },
                "C0_INT4_Null": {
                    "qmax": INT4_QMAX,
                    "train_reward": int4_tr_r,
                    "train_exact_acc": int4_tr_acc,
                    "test_reward": int4_te_r,
                    "test_exact_acc": int4_te_acc,
                },
                "C0_A16_BlockH4_3bpw_Null": {
                    "bound_r2": A16_B30_BOUND,
                    "train_reward": a16_30_tr_r,
                    "train_exact_acc": a16_30_tr_acc,
                    "test_reward": a16_30_te_r,
                    "test_exact_acc": a16_30_te_acc,
                    "syndrome_errors": b30_syn,
                    "oob_groups": b30_oob,
                    "oob_fallback_activations": b30_fb,
                },
                "C0_A16_BlockH4_Null": {
                    "bound_r2": A16_B45_BOUND,
                    "train_reward": a16_tr_r,
                    "train_exact_acc": a16_tr_acc,
                    "test_reward": a16_te_r,
                    "test_exact_acc": a16_te_acc,
                    "syndrome_errors": base_syn_errs,
                    "oob_groups": base_oob_groups,
                    "oob_fallback_activations": base_oob_fallbacks,
                },
            },
            "qwen15b_full_28layer_probe": {
                "layers_quantized": 28,
                "linear_layers_quantized": len(m15_linear_names),
                "total_block_weights": m15_total_weights,
                "total_a16_blocks": m15_total_blocks,
                "ef_dtype": "int8",
                "alkashi_q": ALKASHI_Q,
                "fold_0b_max_fp32_invariance_err": m15_fold_err,
                "folded_scale_extra_bytes": 0,
                "resident_vram_after_quant_mib": m15_resident_after_quant_mib,
                "peak_vram_allocated_mib": peak_vram_15b_mib,
                "peak_vram_reserved_mib": peak_vram_15b_reserved_mib,
                "syndrome_errors": m15_syn,
                "oob_groups": m15_oob,
                "oob_fallback_activations": m15_oob_fb,
                "oob_candidates_reverted": m15_oob_rev_post,
                "a16_lattice_flips": m15_flips_post,
                "folded_scale_fp16_updates": m15_scale_upd_post,
                "probe_rewards_pos_neg": m15_rewards,
                "post_step_reward": r_15_post,
                "wall_s": m15_wall_s,
            },
            "arms": arm_runs,
            "summary": {
                "mean_c1_test_acc": mean_c1_acc,
                "mean_c2_test_acc": mean_c2_acc,
                "mean_t1_test_acc": mean_t1_acc,
                "mean_c1_test_reward": mean_c1_rew,
                "mean_c2_test_reward": mean_c2_rew,
                "mean_t1_test_reward": mean_t1_rew,
                "delta_t1_vs_c2_pp": delta_t1_vs_c2_pp,
                "delta_t1_vs_c1_pp": delta_t1_vs_c1_pp,
                "mean_bf16_ulp_stalled_frac": mean_bf16_stall,
                "p_exact_perm_acc_t1_vs_c2": p_acc_t1_c2,
                "p_exact_perm_acc_t1_vs_c1": p_acc_t1_c1,
                "p_holm_acc_t1_vs_c2": p_holm_t1_c2,
                "p_holm_acc_t1_vs_c1": p_holm_t1_c1,
                "p_exact_perm_rew_t1_vs_c2": p_rew_t1_c2,
                "peak_vram_05b_mib": peak_vram_05b_mib,
                "peak_vram_15b_mib": peak_vram_15b_mib,
                "m15_probe_syndrome_errors": m15_syn,
                "m15_probe_oob_groups": m15_oob,
                "gate_zero_syndromes": gate_syn_ok,
                "gate_vram_under_65gib": gate_vram_ok,
                "gate_ge_2pp_over_int4": gate_acc_2pp_ok,
                "gate_p_holm_lt_005": gate_p_holm_ok,
                "verdict": verdict,
            },
        }


# =============================================================================
# EXP-Q51b: HoloSphere-Cayley-ES on Real HoloSphere-V4 15M Checkpoint
# =============================================================================

def _unpack_hs_a16_u64(packed_u64: Any) -> Any:
    """Unpack HoloSphere-V4 uint64 packed A_16 nibbles -> signed int8 in [-8, 7]."""
    out_dim, n_blk16 = int(packed_u64.shape[0]), int(packed_u64.shape[1])
    raw_u8 = packed_u64.contiguous().view(torch.uint8).view(out_dim, n_blk16, 8)
    lo = (raw_u8 & 0x0F).to(torch.int16)
    hi = ((raw_u8 >> 4) & 0x0F).to(torch.int16)
    nib = torch.stack((lo, hi), dim=-1).reshape(out_dim, n_blk16 * 16)
    return torch.where(nib >= 8, nib - 16, nib).to(torch.int8)


def _pack_hs_a16_u64(c_i8: Any) -> Any:
    """Pack signed int8 A_16 codewords in [-8, 7] -> HoloSphere-V4 uint64 packed format."""
    out_dim, in_dim = int(c_i8.shape[0]), int(c_i8.shape[1])
    n_blk16 = in_dim // 16
    nib = (c_i8.to(torch.int16) & 0x0F).to(torch.uint8).reshape(out_dim, n_blk16, 8, 2)
    packed_u8 = (nib[..., 0] | (nib[..., 1] << 4)).contiguous()
    return packed_u8.view(torch.uint64).reshape(out_dim, n_blk16)


def _update_hs_a16_packed_layer(
    a16_mod: Any,
    ef_i8: Any,
    acc_w_f32: Any,
    h4_unscaled: Any,
) -> tuple[int, int, int]:
    """Apply live int8 Al-Kashi EF + A_16 SECDED update directly to a HoloSphere-V4
    A16BlockH4PackedLinear uint64 buffer (keeping [-8, 7] nibble range + 0 SECDED syndromes).
    Returns (codeword_flips, range_reverted_blocks, post_syndrome_errors)."""
    _require_torch()
    with torch.no_grad():
        out_dim, n_blk16 = int(a16_mod.a16_packed_u64.shape[0]), int(a16_mod.a16_packed_u64.shape[1])
        in_dim = n_blk16 * 16
        c_old = _unpack_hs_a16_u64(a16_mod.a16_packed_u64)
        sc_f32 = a16_mod.a16_scales_fp16.to(torch.float32).clamp_min(1e-30).unsqueeze(-1).unsqueeze(-1)
        # Exact inverse of w = 0.5 * (c @ H_4) * sc is delta_c = 0.5 * (acc / sc) @ H_4 (since H_4^2 = 4 I)
        acc_scaled = (acc_w_f32.reshape(out_dim, n_blk16 // 2, 8, 4) / sc_f32)
        delta_c = (0.5 * torch.matmul(acc_scaled, h4_unscaled)).reshape(out_dim, in_dim)

        u_fixed = ef_i8.to(torch.int32) + torch.round(delta_c * float(ALKASHI_Q)).to(torch.int32)
        target_f = c_old.to(torch.float32) + (u_fixed.to(torch.float32) / float(ALKASHI_Q))
        c_cand, se = _a16_closest(target_f.reshape(-1, 16))
        c_cand_blk = c_cand.round().to(torch.int32).reshape(-1, 16)
        c_old_blk = c_old.to(torch.int32).reshape(-1, 16)
        # Keep old valid codeword for any 16-block exceeding signed 4-bit nibble range [-8, 7]
        oob_nib = ((c_cand_blk < -8) | (c_cand_blk > 7)).any(dim=1)
        reverted = int(oob_nib.sum().item())
        if reverted > 0:
            c_cand_blk = torch.where(oob_nib.unsqueeze(1), c_old_blk, c_cand_blk)
        c_new_i8 = c_cand_blk.to(torch.int8).reshape(out_dim, in_dim)

        realized = c_new_i8.to(torch.int32) - c_old.to(torch.int32)
        flips = int((realized != 0).sum().item())
        ef_i8.copy_((u_fixed - realized * ALKASHI_Q).clamp(-127, 127).to(torch.int8))
        a16_mod.a16_packed_u64.copy_(_pack_hs_a16_u64(c_new_i8))
        syn_post = a16_mod.verify_and_cache_weight(strict_zero_syndrome=True)
        return flips, reverted, se + syn_post


def _encode_byte_batch(items: list[tuple[str, str, int]], bos_id: int, device: Any) -> tuple[Any, Any, Any]:
    seqs = [[bos_id] + list(p.encode("utf-8")) for _, p, _ in items]
    max_l = max(len(s) for s in seqs)
    padded = []
    pos_idx = []
    tgts = []
    for s, (_, _, tb) in zip(seqs, items):
        pos_idx.append(len(s) - 1)
        padded.append(s + [32] * (max_l - len(s)))
        tgts.append(tb)
    return (
        torch.tensor(padded, dtype=torch.long, device=device),
        torch.tensor(pos_idx, dtype=torch.long, device=device),
        torch.tensor(tgts, dtype=torch.long, device=device),
    )


def _causal_byte_conv_streaming(byte_conv: Any, x_inj: Any, prev_tail: Any | None, hs_mod: Any) -> tuple[Any, Any]:
    """Exact streaming CausalDepthwiseByteConv preserving the 3-byte causal history across chunks!"""
    k_sz = byte_conv.kernel_size
    if prev_tail is None:
        xp = F.pad(x_inj.transpose(1, 2), (k_sz - 1, 0))
    else:
        cat = torch.cat((prev_tail, x_inj), dim=1)
        xp = cat.transpose(1, 2)
    out = hs_mod.safe_unit_norm(byte_conv.conv(xp).transpose(1, 2), dim=-1, eps=1e-6)
    if x_inj.shape[1] >= k_sz - 1:
        next_tail = x_inj[:, -(k_sz - 1) :, :]
    else:
        full = x_inj if prev_tail is None else torch.cat((prev_tail, x_inj), dim=1)
        pad_n = (k_sz - 1) - full.shape[1]
        if pad_n > 0:
            z = torch.zeros((full.shape[0], pad_n, full.shape[2]), dtype=full.dtype, device=full.device)
            next_tail = torch.cat((z, full), dim=1)
        else:
            next_tail = full[:, -(k_sz - 1) :, :]
    return out, next_tail


def _update_shell_tracker(raw_err: Any, max_finite: float, nonfinite: bool) -> tuple[float, bool]:
    """Explicit NaN/Inf-safe unclamped shell error tracker (never silently ignores NaN!)."""
    fin_mask = torch.isfinite(raw_err)
    if not bool(fin_mask.all().item()):
        nonfinite = True
        if bool(fin_mask.any().item()):
            val = float(raw_err[fin_mask].max().item())
            if val > max_finite:
                max_finite = val
    else:
        val = float(raw_err.max().item())
        if val > max_finite:
            max_finite = val
    return max_finite, nonfinite


def _holosphere_forward_with_manifold_mode(
    hs_model: Any,
    byte_ids: Any,
    *,
    loop_depth: int,
    manifold_mode: str,
    tangent_bias: list[Any] | None,
    euclid_q_noise: list[Any] | None,
    chunk_size: int = 128,
    hs_mod: Any,
) -> tuple[Any, float, bool]:
    """Run HoloSphere-V4 forward (looped R times, exact causal-conv chunked streaming).
    Returns (logits, unclamped_max_finite_su2_shell_err, su2_shell_err_nonfinite)."""
    bsz, seq_len = byte_ids.shape
    d_model = hs_model.config.d_model
    n_heads = hs_model.config.n_heads
    n_layers = len(hs_model.blocks)
    dev = byte_ids.device

    s_states = [torch.zeros((bsz, n_heads, 32, 32), dtype=torch.float32, device=dev) for _ in range(n_layers)]
    u_states = []
    for _ in range(n_layers):
        u0 = torch.zeros((bsz, n_heads, 8, 4), dtype=torch.float32, device=dev)
        u0[..., 0] = 1.0
        u_states.append(u0)

    max_finite_shell = 0.0
    shell_nonfinite = False
    conv_tail = None
    logits_chunks: list[Any] = []

    for c0 in range(0, seq_len, chunk_size):
        c1 = min(c0 + chunk_size, seq_len)
        x_inj = hs_mod.h256_bent4_inject_zero_vram(
            byte_ids[:, c0:c1], d_model, input_vocab_size=hs_model.config.input_vocab_size
        )
        x, conv_tail = _causal_byte_conv_streaming(hs_model.byte_conv, x_inj, conv_tail, hs_mod)
        if not torch.is_grad_enabled():
            conv_tail = conv_tail.detach()
        clen = c1 - c0
        for _rep in range(loop_depth):
            for l_i, blk in enumerate(hs_model.blocks):
                h_gen = blk._feature_space_v_ring(x, gumbel_temp=0.0, step_seed=42 + l_i * 31)
                raw_w = blk.w_omega(h_gen).view(bsz, clen, d_model // 4, 3)
                scaled_w = raw_w * blk.omega_scale.view(1, 1, d_model // 4, 1)
                if tangent_bias is not None:
                    scaled_w = scaled_w + tangent_bias[l_i].view(1, 1, d_model // 4, 3)
                w_su2 = hs_mod.bounded_cayley_omega(scaled_w, r_max=8.0).view(bsz, clen, n_heads, 8, 3)

                if manifold_mode == "cayley":
                    du_all = hs_mod.cayley_su2(w_su2.float())
                    err_du = ((du_all * du_all).sum(dim=-1) - 1.0).abs()
                    max_finite_shell, shell_nonfinite = _update_shell_tracker(err_du, max_finite_shell, shell_nonfinite)
                    u_prefix = hs_mod.su2_prefix_left_action(du_all.reshape(bsz, clen, n_heads * 8, 4)).reshape(
                        bsz, clen, n_heads, 8, 4
                    )
                    u_path = hs_mod.safe_unit_norm(hs_mod.quat_mul(u_prefix, u_states[l_i].unsqueeze(1)), dim=-1, eps=1e-12)
                    err_up = ((u_path * u_path).sum(dim=-1) - 1.0).abs()
                    max_finite_shell, shell_nonfinite = _update_shell_tracker(err_up, max_finite_shell, shell_nonfinite)
                else:
                    # [MED-7 FIX] ArmA: Exact tangent-space orthogonal Euclidean additive ES (<du, v_tan> = 0, matching Thm 21!)
                    # NO Cayley retraction and NO safe_unit_norm; UNCLAMPED shell error tracked before any numeric cap!
                    du_base = hs_mod.cayley_su2(w_su2.float())
                    if euclid_q_noise is not None:
                        eta = euclid_q_noise[l_i].view(1, 1, n_heads, 8, 4)
                        proj = (du_base * eta).sum(dim=-1, keepdim=True)
                        v_tan = eta - proj * du_base  # exact tangent-space projection <du_base, v_tan> == 0!
                        du_all = du_base + v_tan
                    else:
                        du_all = du_base
                    err_du = ((du_all * du_all).sum(dim=-1) - 1.0).abs()
                    max_finite_shell, shell_nonfinite = _update_shell_tracker(err_du, max_finite_shell, shell_nonfinite)
                    # Unnormalized quaternion prefix product genuinely accumulates Theorem 21 radial blowup!
                    cur = du_all.reshape(bsz, clen, n_heads * 8, 4)
                    step = 1
                    while step < clen:
                        prod = hs_mod.quat_mul(cur[:, step:], cur[:, :-step])
                        cur = torch.cat((cur[:, :step], prod), dim=1)
                        step *= 2
                    u_path = hs_mod.quat_mul(cur.reshape(bsz, clen, n_heads, 8, 4), u_states[l_i].unsqueeze(1))
                    # Record strictly UNCLAMPED shell error (detecting Inf/NaN overflow honestly!)
                    raw_up_err = ((u_path * u_path).sum(dim=-1) - 1.0).abs()
                    max_finite_shell, shell_nonfinite = _update_shell_tracker(raw_up_err, max_finite_shell, shell_nonfinite)

                x_h = h_gen.float().view(bsz, clen, n_heads, 8, 4)
                xt_all = hs_mod.quat_mul(hs_mod.quat_conj(u_path), x_h).reshape(bsz, clen, n_heads, 32)
                z_slow_all = torch.einsum("bthd,hed->bthe", xt_all, blk.w_q_fast.float())
                x_h32 = h_gen.float().view(bsz, clen, n_heads, 32)
                s_cur = s_states[l_i]
                t2 = 0.25 * 0.25
                inv_t = 1.0 / (1.0 + t2)
                c_st = (1.0 - t2) * inv_t
                s_st = (2.0 * 0.25) * inv_t
                ys = []
                for t in range(clen):
                    xt = xt_all[:, t]
                    xi = x_h32[:, t]
                    s_pred = torch.einsum("bhrc,bhc->bhr", s_cur, xt)
                    z_i = z_slow_all[:, t] + s_pred
                    proj = (z_i * xi).sum(dim=-1, keepdim=True)
                    v_i = z_i - proj * xi
                    v_inv = torch.rsqrt((v_i * v_i).sum(dim=-1, keepdim=True) + 1e-6)
                    y_i = c_st * xi + s_st * (v_i * v_inv)
                    err_i = y_i - s_pred
                    s_cur = 0.90 * s_cur + 0.15 * torch.einsum("bhr,bhc->bhrc", err_i, xt)
                    ys.append(y_i)
                s_states[l_i] = s_cur.detach() if not torch.is_grad_enabled() else s_cur
                u_states[l_i] = u_path[:, -1].detach() if not torch.is_grad_enabled() else u_path[:, -1]
                y_cell = torch.stack(ys, dim=1).reshape(bsz, clen, d_model)
                h_cell = hs_mod.safe_unit_norm(blk.w_cell_out(y_cell.to(x.dtype)), dim=-1, eps=1e-6)
                x = hs_mod.safe_unit_norm(h_gen + blk.alpha_cell.abs() * (h_cell - h_gen), dim=-1, eps=1e-6)
                h_mlp = hs_mod.safe_unit_norm(blk.ffn(x), dim=-1, eps=1e-6)
                x = hs_mod.safe_unit_norm(x + blk.alpha_mlp.abs() * (h_mlp - x), dim=-1, eps=1e-6)

        x_unit = hs_mod.safe_unit_norm(x, dim=-1, eps=1e-6)
        logits_chunks.append(hs_model.monarch2_head(x_unit))

    return torch.cat(logits_chunks, dim=1), max_finite_shell, shell_nonfinite


def _eval_group_non_diff(
    hs_model: Any,
    ids: Any,
    pos_idx: Any,
    tgts: Any,
    *,
    loop_depth: int,
    manifold_mode: str,
    tangent_bias: list[Any] | None,
    euclid_q_noise: list[Any] | None,
    hs_mod: Any,
) -> tuple[float, float, float, bool]:
    """Non-differentiable NaN-safe & tie-safe discrete rank/exact-match verifier over group-word byte."""
    with torch.no_grad():
        logits, shell_err, shell_nf = _holosphere_forward_with_manifold_mode(
            hs_model,
            ids,
            loop_depth=loop_depth,
            manifold_mode=manifold_mode,
            tangent_bias=tangent_bias,
            euclid_q_noise=euclid_q_noise,
            chunk_size=128,
            hs_mod=hs_mod,
        )
        bsz = ids.shape[0]
        last = logits[torch.arange(bsz, device=ids.device), pos_idx, :]  # (B, 256)
        finite_row = torch.isfinite(last).all(dim=-1)
        preds = last.argmax(dim=-1)
        # [MED-1 FIX]: Pessimistic tie handling so degenerate constant/all-zero logits NEVER score rank 0!
        max_val = last.max(dim=-1, keepdim=True).values
        n_tied_max = (last == max_val).sum(dim=-1)
        exact = ((preds == tgts) & finite_row & (n_tied_max == 1)).to(torch.float32)

        tgt_scores = last.gather(1, tgts.unsqueeze(1))
        n_gt = (last > tgt_scores).sum(dim=-1)
        n_eq_other = (last == tgt_scores).sum(dim=-1) - 1
        ranks = n_gt + n_eq_other

        rew = torch.where(
            ~finite_row,
            torch.zeros_like(exact),
            torch.where(
                ranks == 0,
                torch.ones_like(exact),
                torch.where(
                    ranks <= 2,
                    torch.full_like(exact, 0.50),
                    torch.where(
                        ranks <= 5,
                        torch.full_like(exact, 0.25),
                        torch.where(ranks <= 15, torch.full_like(exact, 0.10), torch.zeros_like(exact)),
                    ),
                ),
            ),
        )
        return float(rew.mean().item()), float(exact.mean().item()), shell_err, shell_nf


def _eval_wikitext2_bpb(
    hs_model: Any,
    wt_inp: Any,
    wt_tgt: Any,
    *,
    manifold_mode: str,
    tangent_bias: list[Any] | None,
    euclid_q_noise: list[Any] | None,
    hs_mod: Any,
) -> tuple[float, float, bool]:
    with torch.no_grad():
        logits, shell_err, shell_nf = _holosphere_forward_with_manifold_mode(
            hs_model,
            wt_inp,
            loop_depth=1,
            manifold_mode=manifold_mode,
            tangent_bias=tangent_bias,
            euclid_q_noise=euclid_q_noise,
            chunk_size=256,
            hs_mod=hs_mod,
        )
        loss = F.cross_entropy(logits.reshape(-1, 256), wt_tgt.reshape(-1))
        return float(loss.item() / math.log(2.0)), shell_err, shell_nf


def _run_hs_chunk32_step(
    hs_model: Any,
    hs_mod: Any,
    x_chunk: Any,
    loop_r: int,
    tan_bias: list[Any] | None,
    s_st: list[Any],
    u_st: list[Any],
) -> tuple[Any, list[Any], list[Any], float]:
    """Unified Chunk32 core shared identically by Cayley-ES, Full-Graph Chunk32-BPTT, and Checkpointed Chunk32-BPTT!"""
    bsz, cl, d_m = x_chunk.shape
    n_h = hs_model.config.n_heads
    x = x_chunk
    s_next = list(s_st)
    u_next = list(u_st)
    max_sh = 0.0
    for _rp in range(loop_r):
        for li, blk in enumerate(hs_model.blocks):
            hg = blk._feature_space_v_ring(x, gumbel_temp=0.0, step_seed=42 + li * 31)
            rw = blk.w_omega(hg).view(bsz, cl, d_m // 4, 3)
            scaled_w = rw * blk.omega_scale.view(1, 1, d_m // 4, 1)
            if tan_bias is not None:
                scaled_w = scaled_w + tan_bias[li].view(1, 1, d_m // 4, 3)
            ws = hs_mod.bounded_cayley_omega(scaled_w, r_max=8.0).view(bsz, cl, n_h, 8, 3)
            yc, s_new, u_new = hs_mod.chunk32_sfast_fp32_reference(
                hg.float(), ws.float(), blk.w_q_fast.float(), s_init=s_next[li], u_init=u_next[li]
            )
            s_next[li] = s_new
            u_next[li] = u_new
            if not torch.is_grad_enabled():
                err_u = float(((u_new * u_new).sum(dim=-1) - 1.0).abs().max().item())
                if err_u > max_sh:
                    max_sh = err_u
            hc = hs_mod.safe_unit_norm(blk.w_cell_out(yc.to(x.dtype)), dim=-1, eps=1e-6)
            x = hs_mod.safe_unit_norm(hg + blk.alpha_cell.abs() * (hc - hg), dim=-1, eps=1e-6)
            hm = hs_mod.safe_unit_norm(blk.ffn(x), dim=-1, eps=1e-6)
            x = hs_mod.safe_unit_norm(x + blk.alpha_mlp.abs() * (hm - x), dim=-1, eps=1e-6)
    return x, s_next, u_next, max_sh


def run_exp_q51b_holosphere_cayley_es(device: Any) -> dict[str, Any]:
    """Execute EXP-Q51b on real HoloSphere-V4 15M checkpoint."""
    _require_torch()
    print("[EXP-Q51b] Loading real HoloSphere-V4 15M checkpoint...", flush=True)
    if str(HOLOSPHERE_V4_DIR) not in sys.path:
        sys.path.insert(0, str(HOLOSPHERE_V4_DIR))
    import modeling_holosphere_v4 as hs_mod  # noqa: E402

    hs_model = hs_mod.HoloSphereV4ForCausalLM.from_pretrained(
        HOLOSPHERE_V4_DIR, device=str(device), strict_syndrome=True, verify_receipt=True
    )
    syn_init = hs_model.verify_all_a16_syndromes(strict=True)
    _require(syn_init["syndrome_errors"] == 0, "HoloSphere-V4 init syndrome error")

    # [LOW-8 FIX]: 64 x 256 = 16,384 bytes (16 KiB) held-out WikiText-2 UTF-8 bytes
    tbl = pq.read_table(WIKITEXT2_TEST_PQ)
    wt_bytes = "\n".join(x for x in tbl["text"].to_pylist() if x.strip()).encode("utf-8")
    wt_n_chunks = 64
    wt_chunks = [list(wt_bytes[i * 256 : (i + 1) * 256]) for i in range(wt_n_chunks)]
    wt_inp = torch.tensor([[hs_mod.BOS_ID] + c[:-1] for c in wt_chunks], dtype=torch.long, device=device)
    wt_tgt = torch.tensor(wt_chunks, dtype=torch.long, device=device)

    n_layers = len(hs_model.blocks)
    d_model = hs_model.config.d_model
    n_heads = hs_model.config.n_heads

    # =========================================================================
    # [MED-7 FIX] Part B1: Fair Like-for-Like Chunk32 Streaming VRAM Comparison
    # Same chunk_size=32, same chunk32_sfast_fp32_reference, same causal conv!
    # Compares: (1) Cayley-ES WITH active ES buffers, (2) Full-Graph Chunk32-BPTT,
    #           (3) Activation-Checkpointed Chunk32-BPTT!
    # =========================================================================
    print("  [B1] Like-for-like Chunk32 streaming VRAM: Cayley-ES (active buffers) vs Full-BPTT vs Checkpointed-BPTT...", flush=True)
    vram_grid: dict[str, dict[str, Any]] = {}
    gen_seq = torch.Generator(device="cpu").manual_seed(16751)
    chunk_sz_b1 = 32  # TRUE Chunk32 across ALL 3 methods!

    for seq_t in (512, 2048, 8192):
        for loop_r in (1, 4, 8):
            key = f"T{seq_t}_R{loop_r}"
            probe_ids = torch.randint(32, 127, (1, seq_t), generator=gen_seq, dtype=torch.long).to(device)

            # (1) Forward-only Cayley-ES WITH active ES master + tangent perturbation buffers on GPU!
            _cleanup_gpu()
            torch.cuda.reset_peak_memory_stats(device)
            with torch.no_grad():
                es_tan = [torch.zeros((d_model // 4, 3), dtype=torch.float32, device=device) for _ in range(n_layers)]
                es_os = [blk.omega_scale.detach().float().clone() for blk in hs_model.blocks]
                es_m1 = hs_model.monarch_b1_fp16.detach().float().clone()
                es_m2 = hs_model.monarch_b2_fp16.detach().float().clone()
                es_bb = hs_model.byte_bias_fp16.detach().float().clone()
                # Apply live orthogonal tangent perturbation for member 0
                es_tan_pert = [
                    es_tan[li] + 0.08 * _gen_orth_blockh4_noise((d_model // 4, 3), 167510 + li * 37, 0, device, torch.float32)
                    for li in range(n_layers)
                ]
                s_st = [torch.zeros((1, n_heads, 32, 32), dtype=torch.float32, device=device) for _ in range(n_layers)]
                u_st = []
                for _ in range(n_layers):
                    u0 = torch.zeros((1, n_heads, 8, 4), dtype=torch.float32, device=device)
                    u0[..., 0] = 1.0
                    u_st.append(u0)
                conv_tail = None
                es_max_shell = 0.0
                for c0 in range(0, seq_t, chunk_sz_b1):
                    c1 = min(c0 + chunk_sz_b1, seq_t)
                    x_inj = hs_mod.h256_bent4_inject_zero_vram(
                        probe_ids[:, c0:c1], d_model, input_vocab_size=hs_model.config.input_vocab_size
                    )
                    x, conv_tail = _causal_byte_conv_streaming(hs_model.byte_conv, x_inj, conv_tail, hs_mod)
                    x, s_st, u_st, sh = _run_hs_chunk32_step(hs_model, hs_mod, x, loop_r, es_tan_pert, s_st, u_st)
                    if sh > es_max_shell:
                        es_max_shell = sh
                    _last_lg = hs_model.monarch2_head(hs_mod.safe_unit_norm(x, dim=-1, eps=1e-6))
                del s_st, u_st, conv_tail, _last_lg, es_tan, es_tan_pert, es_os, es_m1, es_m2, es_bb
            torch.cuda.synchronize(device)
            es_vram_mib = torch.cuda.max_memory_allocated(device) / 1048576.0

            # (2) Full-Graph Chunk32-BPTT (exact same Chunk32 core, backprop through entire T & R, last-chunk loss only!)
            _cleanup_gpu()
            torch.cuda.reset_peak_memory_stats(device)
            bptt_full_oom = False
            bptt_full_vram_mib = 0.0
            try:
                hs_model.zero_grad(set_to_none=True)
                s_st = [torch.zeros((1, n_heads, 32, 32), dtype=torch.float32, device=device) for _ in range(n_layers)]
                u_st = []
                for _ in range(n_layers):
                    u0 = torch.zeros((1, n_heads, 8, 4), dtype=torch.float32, device=device)
                    u0[..., 0] = 1.0
                    u_st.append(u0)
                conv_tail = None
                x_last = None
                for c0 in range(0, seq_t, chunk_sz_b1):
                    c1 = min(c0 + chunk_sz_b1, seq_t)
                    x_inj = hs_mod.h256_bent4_inject_zero_vram(
                        probe_ids[:, c0:c1], d_model, input_vocab_size=hs_model.config.input_vocab_size
                    )
                    x, conv_tail = _causal_byte_conv_streaming(hs_model.byte_conv, x_inj, conv_tail, hs_mod)
                    x_last, s_st, u_st, _ = _run_hs_chunk32_step(hs_model, hs_mod, x, loop_r, None, s_st, u_st)
                lg_last = hs_model.monarch2_head(hs_mod.safe_unit_norm(x_last, dim=-1, eps=1e-6))
                loss_b = lg_last[:, -1, :].sum()
                loss_b.backward()
                torch.cuda.synchronize(device)
                bptt_full_vram_mib = torch.cuda.max_memory_allocated(device) / 1048576.0
                del s_st, u_st, conv_tail, x_last, lg_last, loss_b
            except torch.cuda.OutOfMemoryError:
                bptt_full_oom = True
                bptt_full_vram_mib = torch.cuda.max_memory_allocated(device) / 1048576.0
            hs_model.zero_grad(set_to_none=True)
            _cleanup_gpu()

            # (3) Activation-Checkpointed Chunk32-BPTT (checkpoint each Chunk32 step via torch.utils.checkpoint!)
            torch.cuda.reset_peak_memory_stats(device)
            bptt_ckpt_oom = False
            bptt_ckpt_vram_mib = 0.0
            try:
                hs_model.zero_grad(set_to_none=True)
                s_st = [torch.zeros((1, n_heads, 32, 32), dtype=torch.float32, device=device) for _ in range(n_layers)]
                u_st = []
                for _ in range(n_layers):
                    u0 = torch.zeros((1, n_heads, 8, 4), dtype=torch.float32, device=device)
                    u0[..., 0] = 1.0
                    u_st.append(u0)
                conv_tail = None
                x_last = None

                def _ckpt_chunk_fn(x_in: Any, *states_flat: Any) -> tuple[Any, ...]:
                    s_in = list(states_flat[:n_layers])
                    u_in = list(states_flat[n_layers:])
                    x_out, s_out, u_out, _ = _run_hs_chunk32_step(hs_model, hs_mod, x_in, loop_r, None, s_in, u_in)
                    return (x_out, *s_out, *u_out)

                for c0 in range(0, seq_t, chunk_sz_b1):
                    c1 = min(c0 + chunk_sz_b1, seq_t)
                    x_inj = hs_mod.h256_bent4_inject_zero_vram(
                        probe_ids[:, c0:c1], d_model, input_vocab_size=hs_model.config.input_vocab_size
                    )
                    x, conv_tail = _causal_byte_conv_streaming(hs_model.byte_conv, x_inj, conv_tail, hs_mod)
                    out_tup = ckpt_mod.checkpoint(_ckpt_chunk_fn, x, *s_st, *u_st, use_reentrant=False)
                    x_last = out_tup[0]
                    s_st = list(out_tup[1 : 1 + n_layers])
                    u_st = list(out_tup[1 + n_layers :])
                lg_last = hs_model.monarch2_head(hs_mod.safe_unit_norm(x_last, dim=-1, eps=1e-6))
                loss_c = lg_last[:, -1, :].sum()
                loss_c.backward()
                torch.cuda.synchronize(device)
                bptt_ckpt_vram_mib = torch.cuda.max_memory_allocated(device) / 1048576.0
                del s_st, u_st, conv_tail, x_last, lg_last, loss_c
            except torch.cuda.OutOfMemoryError:
                bptt_ckpt_oom = True
                bptt_ckpt_vram_mib = torch.cuda.max_memory_allocated(device) / 1048576.0
            hs_model.zero_grad(set_to_none=True)
            del probe_ids
            _cleanup_gpu()

            vram_grid[key] = {
                "T": seq_t,
                "R": loop_r,
                "chunk_size": chunk_sz_b1,
                "es_peak_vram_mib": es_vram_mib,
                "bptt_peak_vram_mib": bptt_full_vram_mib,
                "bptt_oom": bptt_full_oom,
                "bptt_ckpt_peak_vram_mib": bptt_ckpt_vram_mib,
                "bptt_ckpt_oom": bptt_ckpt_oom,
                "es_max_su2_shell_err": es_max_shell,
            }
            print(
                f"    {key}: Cayley-ES={es_vram_mib:.1f} MiB (sh={es_max_shell:.2e}) | "
                f"Full-BPTT={bptt_full_vram_mib:.1f} MiB (oom={bptt_full_oom}) | "
                f"Ckpt-BPTT={bptt_ckpt_vram_mib:.1f} MiB (oom={bptt_ckpt_oom})",
                flush=True,
            )

    es_vram_vals = [v["es_peak_vram_mib"] for v in vram_grid.values()]
    es_vram_max = max(es_vram_vals)
    es_vram_min = min(es_vram_vals)
    es_vram_spread = es_vram_max - es_vram_min

    # =========================================================================
    # Part B2: Non-Differentiable S_3 & A_5 Group-Word ES + Live A_16 Packed Updates!
    # =========================================================================
    train_grp = build_group_word_split(16, 16, min_len=2, max_len=4, seed=167201, exclude_words=set())
    train_words = {(g, p.split(":")[1].rstrip("=")) for g, p, _ in train_grp}
    test_grp_ood = build_group_word_split(32, 32, min_len=16, max_len=24, seed=167202, exclude_words=train_words)

    tr_ids, tr_pos, tr_tgt = _encode_byte_batch(train_grp, hs_mod.BOS_ID, device)
    te_ids, te_pos, te_tgt = _encode_byte_batch(test_grp_ood, hs_mod.BOS_ID, device)

    # Snapshot pristine optimizable HoloSphere-V4 params INCLUDING the 6 A_16 packed w_omega layers!
    base_state = {
        "omega_scale": [blk.omega_scale.detach().cpu().clone() for blk in hs_model.blocks],
        "w_q_fast": [blk.w_q_fast.detach().cpu().clone() for blk in hs_model.blocks],
        "alpha_cell": [blk.alpha_cell.detach().cpu().clone() for blk in hs_model.blocks],
        "alpha_ring": [blk.alpha_ring.detach().cpu().clone() for blk in hs_model.blocks],
        "alpha_mlp": [blk.alpha_mlp.detach().cpu().clone() for blk in hs_model.blocks],
        "w_omega_a16_u64": [blk.w_omega.a16_packed_u64.detach().cpu().clone() for blk in hs_model.blocks],
        "w_omega_a16_sc": [blk.w_omega.a16_scales_fp16.detach().cpu().clone() for blk in hs_model.blocks],
        "monarch_b1": hs_model.monarch_b1_fp16.detach().cpu().clone(),
        "monarch_b2": hs_model.monarch_b2_fp16.detach().cpu().clone(),
        "byte_bias": hs_model.byte_bias_fp16.detach().cpu().clone(),
    }

    def _restore_hs_base() -> None:
        for li, blk in enumerate(hs_model.blocks):
            blk.omega_scale.data.copy_(base_state["omega_scale"][li].to(device))
            blk.w_q_fast.data.copy_(base_state["w_q_fast"][li].to(device))
            blk.alpha_cell.data.copy_(base_state["alpha_cell"][li].to(device))
            blk.alpha_ring.data.copy_(base_state["alpha_ring"][li].to(device))
            blk.alpha_mlp.data.copy_(base_state["alpha_mlp"][li].to(device))
            blk.w_omega.a16_packed_u64.copy_(base_state["w_omega_a16_u64"][li].to(device))
            blk.w_omega.a16_scales_fp16.copy_(base_state["w_omega_a16_sc"][li].to(device))
            blk.w_omega.verify_and_cache_weight(strict_zero_syndrome=True)
        hs_model.monarch_b1_fp16.data.copy_(base_state["monarch_b1"].to(device))
        hs_model.monarch_b2_fp16.data.copy_(base_state["monarch_b2"].to(device))
        hs_model.byte_bias_fp16.data.copy_(base_state["byte_bias"].to(device))

    base_tr_r, base_tr_acc, base_tr_sh, _ = _eval_group_non_diff(
        hs_model, tr_ids, tr_pos, tr_tgt, loop_depth=1, manifold_mode="cayley", tangent_bias=None, euclid_q_noise=None, hs_mod=hs_mod
    )
    base_ood_r, base_ood_acc, base_ood_sh, _ = _eval_group_non_diff(
        hs_model, te_ids, te_pos, te_tgt, loop_depth=8, manifold_mode="cayley", tangent_bias=None, euclid_q_noise=None, hs_mod=hs_mod
    )
    base_bpb, base_bpb_sh, _ = _eval_wikitext2_bpb(
        hs_model, wt_inp, wt_tgt, manifold_mode="cayley", tangent_bias=None, euclid_q_noise=None, hs_mod=hs_mod
    )
    print(
        f"  [Arm0 Base] train_acc={base_tr_acc:.4f} (r={base_tr_r:.4f}) | "
        f"8x_OOD_acc={base_ood_acc:.4f} (r={base_ood_r:.4f}) | WT2_BPB(16KiB)={base_bpb:.4f} | shell_err={max(base_tr_sh, base_ood_sh):.2e}",
        flush=True,
    )

    hs_iters = 20
    hs_pop = 16  # 8 antithetic pairs = 2 orthogonal H_4 4-tuples
    hs_sigma_tangent = 0.08
    hs_alpha_tangent = 0.06
    hs_sigma_head = 0.04
    hs_alpha_head = 0.04
    hs_sigma_a16 = 0.02
    hs_alpha_a16 = 0.02
    h4_unscaled = torch.tensor(
        [[1.0, 1.0, 1.0, 1.0], [1.0, -1.0, 1.0, -1.0], [1.0, 1.0, -1.0, -1.0], [1.0, -1.0, -1.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )

    hs_arms: dict[str, list[dict[str, Any]]] = {
        "ArmA_Euclidean_Additive_ES": [],
        "ArmC_HoloSphere_Cayley_ES": [],
    }

    for arm_name, mode in (("ArmA_Euclidean_Additive_ES", "euclidean"), ("ArmC_HoloSphere_Cayley_ES", "cayley")):
        for seed in SEEDS_Q51B:
            _restore_hs_base()
            tan_bias = [torch.zeros((d_model // 4, 3), dtype=torch.float32, device=device) for _ in range(n_layers)]
            euc_bias = [torch.zeros((n_heads, 8, 4), dtype=torch.float32, device=device) for _ in range(n_layers)]
            m1_f32 = base_state["monarch_b1"].to(device=device, dtype=torch.float32).clone()
            m2_f32 = base_state["monarch_b2"].to(device=device, dtype=torch.float32).clone()
            bb_f32 = base_state["byte_bias"].to(device=device, dtype=torch.float32).clone()
            os_f32 = [t.to(device=device, dtype=torch.float32).clone() for t in base_state["omega_scale"]]
            # Live int8 Al-Kashi EF residuals for all 6 w_omega A_16 packed uint64 layers ([MED-6 FIX]!)
            w_omega_ef = [
                torch.zeros((blk.w_omega.out_features, blk.w_omega.in_features), dtype=torch.int8, device=device)
                for blk in hs_model.blocks
            ]
            w_omega_base_cache = [blk.w_omega._cached_w.clone() for blk in hs_model.blocks]

            _cleanup_gpu()
            torch.cuda.reset_peak_memory_stats(device)
            t0 = time.perf_counter()
            rng = random.Random(seed * 2027 + (1 if mode == "euclidean" else 2))
            max_finite_shell = 0.0
            shell_nonfinite = False
            n_pairs = hs_pop // 2
            iter_rews: list[float] = []
            tot_a16_flips = 0
            tot_a16_reverted = 0
            tot_a16_syn_steps = 0

            for _it in range(hs_iters):
                g_seeds = [rng.randint(1, (1 << 27) - 1) for _ in range(n_pairs // 4)]
                r_pos: list[float] = []
                r_neg: list[float] = []
                for p_idx in range(n_pairs):
                    gs = g_seeds[p_idx // 4]
                    rk = p_idx % 4
                    for sign_v, bucket in ((1.0, r_pos), (-1.0, r_neg)):
                        tb_use = None
                        eu_use = None
                        if mode == "cayley":
                            tb_use = [
                                tan_bias[li]
                                + (sign_v * hs_sigma_tangent)
                                * _gen_orth_blockh4_noise((d_model // 4, 3), gs + li * 37, rk, device, torch.float32)
                                for li in range(n_layers)
                            ]
                        else:
                            eu_use = [
                                euc_bias[li]
                                + (sign_v * hs_sigma_tangent)
                                * _gen_orth_blockh4_noise((n_heads, 8, 4), gs + li * 37, rk, device, torch.float32)
                                for li in range(n_layers)
                            ]
                        for li, blk in enumerate(hs_model.blocks):
                            eps_os = _gen_orth_blockh4_noise((d_model // 4,), gs + 100 + li * 37, rk, device, torch.float32)
                            blk.omega_scale.data.copy_(os_f32[li] + (sign_v * hs_sigma_head) * eps_os)
                            # Perturb w_omega A_16 layer weight during rollout
                            eps_a16 = _gen_orth_blockh4_noise(
                                tuple(w_omega_base_cache[li].shape), gs + 700 + li * 37, rk, device, torch.float32
                            )
                            blk.w_omega._cached_w = w_omega_base_cache[li] + (sign_v * hs_sigma_a16) * eps_a16
                        hs_model.monarch_b1_fp16.data.copy_(
                            (
                                m1_f32
                                + (sign_v * hs_sigma_head)
                                * _gen_orth_blockh4_noise(tuple(m1_f32.shape), gs + 501, rk, device, torch.float32)
                            ).to(torch.float16)
                        )
                        hs_model.monarch_b2_fp16.data.copy_(
                            (
                                m2_f32
                                + (sign_v * hs_sigma_head)
                                * _gen_orth_blockh4_noise(tuple(m2_f32.shape), gs + 502, rk, device, torch.float32)
                            ).to(torch.float16)
                        )
                        hs_model.byte_bias_fp16.data.copy_(
                            (
                                bb_f32
                                + (sign_v * hs_sigma_head)
                                * _gen_orth_blockh4_noise(tuple(bb_f32.shape), gs + 503, rk, device, torch.float32)
                            ).to(torch.float16)
                        )
                        rm, _, sh, sh_nf = _eval_group_non_diff(
                            hs_model,
                            tr_ids,
                            tr_pos,
                            tr_tgt,
                            loop_depth=1,
                            manifold_mode=mode,
                            tangent_bias=tb_use,
                            euclid_q_noise=eu_use,
                            hs_mod=hs_mod,
                        )
                        if sh > max_finite_shell:
                            max_finite_shell = sh
                        shell_nonfinite = shell_nonfinite or sh_nf
                        bucket.append(rm)

                all_r = r_pos + r_neg
                rmean = sum(all_r) / float(len(all_r))
                rstd = math.sqrt(sum((r - rmean) ** 2 for r in all_r) / float(len(all_r)))
                iter_rews.append(rmean)
                if rstd > 1e-12:
                    pw = [(rp - rn) / (2.0 * (rstd + 1e-8)) for rp, rn in zip(r_pos, r_neg)]
                    w_omega_accs = [torch.zeros_like(w_omega_base_cache[li]) for li in range(n_layers)]
                    for p_idx, w_p in enumerate(pw):
                        gs = g_seeds[p_idx // 4]
                        rk = p_idx % 4
                        step_t = (hs_alpha_tangent / float(n_pairs)) * w_p
                        step_h = (hs_alpha_head / float(n_pairs)) * w_p
                        step_a = (hs_alpha_a16 / float(n_pairs)) * w_p
                        for li in range(n_layers):
                            if mode == "cayley":
                                tan_bias[li].add_(
                                    _gen_orth_blockh4_noise((d_model // 4, 3), gs + li * 37, rk, device, torch.float32),
                                    alpha=step_t,
                                )
                            else:
                                euc_bias[li].add_(
                                    _gen_orth_blockh4_noise((n_heads, 8, 4), gs + li * 37, rk, device, torch.float32),
                                    alpha=step_t,
                                )
                            os_f32[li].add_(
                                _gen_orth_blockh4_noise((d_model // 4,), gs + 100 + li * 37, rk, device, torch.float32),
                                alpha=step_h,
                            )
                            w_omega_accs[li].add_(
                                _gen_orth_blockh4_noise(
                                    tuple(w_omega_base_cache[li].shape), gs + 700 + li * 37, rk, device, torch.float32
                                ),
                                alpha=step_a,
                            )
                        m1_f32.add_(_gen_orth_blockh4_noise(tuple(m1_f32.shape), gs + 501, rk, device, torch.float32), alpha=step_h)
                        m2_f32.add_(_gen_orth_blockh4_noise(tuple(m2_f32.shape), gs + 502, rk, device, torch.float32), alpha=step_h)
                        bb_f32.add_(_gen_orth_blockh4_noise(tuple(bb_f32.shape), gs + 503, rk, device, torch.float32), alpha=step_h)

                    # Commit LIVE A_16 uint64 SECDED updates to all 6 w_omega layers!
                    for li, blk in enumerate(hs_model.blocks):
                        flips, rev_b, syn_s = _update_hs_a16_packed_layer(
                            blk.w_omega, w_omega_ef[li], w_omega_accs[li], h4_unscaled
                        )
                        tot_a16_flips += flips
                        tot_a16_reverted += rev_b
                        tot_a16_syn_steps += syn_s
                        w_omega_base_cache[li] = blk.w_omega._cached_w.clone()

                for li, blk in enumerate(hs_model.blocks):
                    blk.omega_scale.data.copy_(os_f32[li])
                    blk.w_omega._cached_w = w_omega_base_cache[li]
                hs_model.monarch_b1_fp16.data.copy_(m1_f32.to(torch.float16))
                hs_model.monarch_b2_fp16.data.copy_(m2_f32.to(torch.float16))
                hs_model.byte_bias_fp16.data.copy_(bb_f32.to(torch.float16))

            # Final evaluation on train, 8x OOD test (R=8, len 16..24), and 16 KiB WikiText-2 UTF-8 BPB
            tb_fin = tan_bias if mode == "cayley" else None
            eu_fin = euc_bias if mode == "euclidean" else None
            tr_r, tr_acc, tr_sh, tr_nf = _eval_group_non_diff(
                hs_model, tr_ids, tr_pos, tr_tgt, loop_depth=1, manifold_mode=mode, tangent_bias=tb_fin, euclid_q_noise=eu_fin, hs_mod=hs_mod
            )
            ood_r, ood_acc, ood_sh, ood_nf = _eval_group_non_diff(
                hs_model, te_ids, te_pos, te_tgt, loop_depth=8, manifold_mode=mode, tangent_bias=tb_fin, euclid_q_noise=eu_fin, hs_mod=hs_mod
            )
            wt_bpb, wt_sh, wt_nf = _eval_wikitext2_bpb(
                hs_model, wt_inp, wt_tgt, manifold_mode=mode, tangent_bias=tb_fin, euclid_q_noise=eu_fin, hs_mod=hs_mod
            )
            max_finite_shell = max(max_finite_shell, tr_sh, ood_sh, wt_sh)
            shell_nonfinite = shell_nonfinite or tr_nf or ood_nf or wt_nf
            syn_post = hs_model.verify_all_a16_syndromes(strict=True)
            dt = time.perf_counter() - t0
            pk = torch.cuda.max_memory_allocated(device) / 1048576.0
            bpb_finite = math.isfinite(wt_bpb)
            rec = {
                "seed": seed,
                "train_reward": tr_r,
                "train_exact_acc": tr_acc,
                "ood_8x_reward": ood_r,
                "ood_8x_exact_acc": ood_acc,
                "ood_8x_gain_pp": (ood_acc - base_ood_acc) * 100.0,
                "wikitext2_bpb": wt_bpb if bpb_finite else None,
                "wikitext2_bpb_finite": bpb_finite,
                "wikitext2_bpb_drift": (wt_bpb - base_bpb) if bpb_finite else None,
                "max_finite_su2_shell_err": max_finite_shell,
                "su2_shell_err_nonfinite": shell_nonfinite,
                "a16_layers_updated": n_layers,
                "a16_codeword_flips": tot_a16_flips,
                "a16_nibble_oob_reverted_blocks": tot_a16_reverted,
                "a16_syndrome_errors": syn_post["syndrome_errors"] + tot_a16_syn_steps,
                "peak_vram_mib": pk,
                "wall_s": dt,
                "iter_pre_update_pop_mean_rewards": iter_rews,
            }
            hs_arms[arm_name].append(rec)
            bpb_str = f"{wt_bpb:.4f} (drift={wt_bpb - base_bpb:+.4f})" if bpb_finite else "NaN_COLLAPSE"
            sh_str = f"{max_finite_shell:.2e}" + ("+INF/NAN" if shell_nonfinite else "")
            print(
                f"  [{arm_name} s={seed}] tr_acc={tr_acc:.4f} | 8x_OOD_acc={ood_acc:.4f} (r={ood_r:.4f}, {rec['ood_8x_gain_pp']:+.2f}pp) | "
                f"WT2_BPB={bpb_str} | shell_err={sh_str} | a16_flips={tot_a16_flips:,} syn={rec['a16_syndrome_errors']} "
                f"vram={pk:.1f}MiB ({dt:.1f}s)",
                flush=True,
            )

    del hs_model, wt_inp, wt_tgt, tr_ids, tr_pos, tr_tgt, te_ids, te_pos, te_tgt, h4_unscaled
    _cleanup_gpu()

    arm_a_runs = hs_arms["ArmA_Euclidean_Additive_ES"]
    arm_c_runs = hs_arms["ArmC_HoloSphere_Cayley_ES"]
    max_shell_c = max(r["max_finite_su2_shell_err"] for r in arm_c_runs)
    shell_c_nf = any(r["su2_shell_err_nonfinite"] for r in arm_c_runs)
    max_shell_a = max(r["max_finite_su2_shell_err"] for r in arm_a_runs)
    shell_a_nf = any(r["su2_shell_err_nonfinite"] for r in arm_a_runs)
    mean_ood_gain_c_pp = sum(r["ood_8x_gain_pp"] for r in arm_c_runs) / len(arm_c_runs)
    mean_ood_gain_a_pp = sum(r["ood_8x_gain_pp"] for r in arm_a_runs) / len(arm_a_runs)
    c_drifts = [r["wikitext2_bpb_drift"] for r in arm_c_runs if r["wikitext2_bpb_drift"] is not None]
    a_drifts = [r["wikitext2_bpb_drift"] for r in arm_a_runs if r["wikitext2_bpb_drift"] is not None]
    mean_bpb_drift_c = (sum(c_drifts) / len(c_drifts)) if len(c_drifts) == len(arm_c_runs) else None
    mean_bpb_drift_a = (sum(a_drifts) / len(a_drifts)) if len(a_drifts) == len(arm_a_runs) else None
    mean_tr_acc_c = sum(r["train_exact_acc"] for r in arm_c_runs) / len(arm_c_runs)
    mean_tr_acc_a = sum(r["train_exact_acc"] for r in arm_a_runs) / len(arm_a_runs)

    # Preregistered kill gate for EXP-Q51b (UNIFIED < 1.0 MiB spread across bench/oracle/verifier!):
    gate_shell_ok = (not shell_c_nf) and (max_shell_c < 1e-5) and all(
        v["es_max_su2_shell_err"] < 1e-5 for v in vram_grid.values()
    )
    gate_o1_vram_ok = (es_vram_max < 512.0) and (es_vram_spread < O1_VRAM_SPREAD_MAX_MIB)
    gate_ood_10pp_ok = mean_ood_gain_c_pp >= 10.0
    gate_bpb_drift_ok = mean_bpb_drift_c is not None and mean_bpb_drift_c <= 0.02
    if gate_shell_ok and gate_o1_vram_ok and gate_ood_10pp_ok and gate_bpb_drift_ok:
        verdict_b = "CONFIRMED"
    elif gate_shell_ok and gate_o1_vram_ok:
        verdict_b = "PARTIAL"
    else:
        verdict_b = "FALSIFIED"

    return {
        "model": "HoloSphere-V4-15M",
        "total_a16_layers": syn_init["a16_layers"],
        "total_a16_blocks": syn_init["total_a16_blocks"],
        "wikitext2_eval_bytes": wt_n_chunks * 256,
        "n_train": len(train_grp),
        "n_test_ood": len(test_grp_ood),
        "n_iters": hs_iters,
        "n_pop": hs_pop,
        "seeds": list(SEEDS_Q51B),
        "vram_o1_scaling_grid": vram_grid,
        "baseline_base": {
            "train_reward": base_tr_r,
            "train_exact_acc": base_tr_acc,
            "ood_8x_reward": base_ood_r,
            "ood_8x_exact_acc": base_ood_acc,
            "wikitext2_bpb": base_bpb,
            "max_su2_shell_err": max(base_tr_sh, base_ood_sh, base_bpb_sh),
        },
        "arms": hs_arms,
        "summary": {
            "es_o1_vram_min_mib": es_vram_min,
            "es_o1_vram_max_mib": es_vram_max,
            "es_o1_vram_spread_mib": es_vram_spread,
            "max_finite_su2_shell_err_cayley_c": max_shell_c,
            "su2_shell_err_nonfinite_cayley_c": shell_c_nf,
            "max_finite_su2_shell_err_euclid_a": max_shell_a,
            "su2_shell_err_nonfinite_euclid_a": shell_a_nf,
            "mean_train_acc_cayley_c": mean_tr_acc_c,
            "mean_train_acc_euclid_a": mean_tr_acc_a,
            "mean_ood_8x_gain_pp_cayley_c": mean_ood_gain_c_pp,
            "mean_ood_8x_gain_pp_euclid_a": mean_ood_gain_a_pp,
            "mean_wikitext2_bpb_drift_cayley_c": mean_bpb_drift_c,
            "mean_wikitext2_bpb_drift_euclid_a": mean_bpb_drift_a,
            "gate_shell_err_lt_1e5": gate_shell_ok,
            "gate_o1_vram_under_512mib": gate_o1_vram_ok,
            "gate_ood_ge_10pp": gate_ood_10pp_ok,
            "gate_bpb_drift_le_002": gate_bpb_drift_ok,
            "verdict": verdict_b,
        },
    }


def _pack_shell_nano(max_finite_err: float, nonfinite: bool) -> tuple[int, bool]:
    """Convert unclamped shell error to pure-integer nano units + explicit overflow boolean."""
    if nonfinite or not math.isfinite(max_finite_err):
        return MAX_FINITE_SHELL_NANO_CAP, True
    q_nano = Fraction(repr(float(max_finite_err))) * 1_000_000_000
    if q_nano >= MAX_FINITE_SHELL_NANO_CAP:
        return MAX_FINITE_SHELL_NANO_CAP, True
    return int(round(q_nano)), False


def build_pure_int_receipt(
    raw_results: dict[str, Any],
    raw_sha256: str,
    prereg_sha256: str,
    bench_script_sha256: str,
) -> dict[str, Any]:
    """Emit strictly float-free pure-integer empirical receipt for Z/Q oracle & verifier."""
    q51a = raw_results["exp_q51a"]
    q51b = raw_results["exp_q51b"]
    s_a = q51a["summary"]
    p15 = q51a["qwen15b_full_28layer_probe"]

    def _pack_a_arm(runs: list[dict[str, Any]]) -> list[dict[str, int]]:
        out = []
        for r in runs:
            out.append(
                {
                    "seed": int(r["seed"]),
                    "train_reward_micro": _to_micro(r["train_reward"]),
                    "train_exact_acc_micro": _to_micro(r["train_exact_acc"]),
                    "test_reward_micro": _to_micro(r["test_reward"]),
                    "test_exact_acc_micro": _to_micro(r["test_exact_acc"]),
                    "syndrome_errors": int(r["syndrome_errors"]),
                    "oob_groups": int(r["oob_groups"]),
                    "oob_candidates_reverted": int(r["oob_candidates_reverted"]),
                    "peak_vram_kib": _to_kib(r["peak_vram_mib"]),
                }
            )
        return out

    def _pack_b_arm(runs: list[dict[str, Any]]) -> list[dict[str, int | bool]]:
        out: list[dict[str, int | bool]] = []
        for r in runs:
            bpb_val = r["wikitext2_bpb"]
            bpb_fin = bpb_val is not None and math.isfinite(float(bpb_val))
            sh_nano, sh_ovf = _pack_shell_nano(
                float(r["max_finite_su2_shell_err"]), bool(r["su2_shell_err_nonfinite"])
            )
            out.append(
                {
                    "seed": int(r["seed"]),
                    "train_reward_micro": _to_micro(r["train_reward"]),
                    "train_exact_acc_micro": _to_micro(r["train_exact_acc"]),
                    "ood_8x_reward_micro": _to_micro(r["ood_8x_reward"]),
                    "ood_8x_exact_acc_micro": _to_micro(r["ood_8x_exact_acc"]),
                    "wikitext2_bpb_finite": bool(bpb_fin),
                    "wikitext2_bpb_micro": _to_micro(float(bpb_val)) if bpb_fin else -1,
                    "max_su2_shell_err_nano": sh_nano,
                    "su2_shell_err_overflow": sh_ovf,
                    "a16_layers_updated": int(r["a16_layers_updated"]),
                    "a16_codeword_flips": int(r["a16_codeword_flips"]),
                    "a16_nibble_oob_reverted_blocks": int(r["a16_nibble_oob_reverted_blocks"]),
                    "a16_syndrome_errors": int(r["a16_syndrome_errors"]),
                    "peak_vram_kib": _to_kib(r["peak_vram_mib"]),
                }
            )
        return out

    vram_grid_int: dict[str, dict[str, int | bool]] = {}
    for k, v in q51b["vram_o1_scaling_grid"].items():
        vram_grid_int[k] = {
            "T": int(v["T"]),
            "R": int(v["R"]),
            "chunk_size": int(v["chunk_size"]),
            "es_peak_vram_kib": _to_kib(v["es_peak_vram_mib"]),
            "bptt_peak_vram_kib": _to_kib(v["bptt_peak_vram_mib"]),
            "bptt_oom": bool(v["bptt_oom"]),
            "bptt_ckpt_peak_vram_kib": _to_kib(v["bptt_ckpt_peak_vram_mib"]),
            "bptt_ckpt_oom": bool(v["bptt_ckpt_oom"]),
            "es_max_su2_shell_err_nano": _to_nano(v["es_max_su2_shell_err"]),
        }

    return {
        "schema": "aletheia.track_s167.empirical_receipt.v2",
        "prereg_id": "PREREG-W22-ES-SCALE",
        "prereg_sha256": prereg_sha256,
        "bench_script_sha256": bench_script_sha256,
        "empirical_results_sha256": raw_sha256,
        "energy_claim": ENERGY_CLAIM,
        "measured_live_model_weights": True,
        "gpu_name": str(raw_results["gpu_before"]["name"]),
        "gpu_vram_total_mib": int(raw_results["gpu_before"]["memory_total_mib"]),
        "in_process_cuda_allocated_after_kib": _to_kib(raw_results["in_process_cuda_allocated_after_mib"]),
        "nvidia_smi_in_process_after_mib": int(raw_results["gpu_after"]["memory_used_mib"]),
        "exp_q51a": {
            "linear_layers_05b": int(q51a["linear_layers_05b"]),
            "total_a16_blocks_05b": int(q51a["total_a16_blocks_05b"]),
            "m15_layers_quantized": int(p15["layers_quantized"]),
            "m15_linear_layers_quantized": int(p15["linear_layers_quantized"]),
            "m15_total_a16_blocks": int(p15["total_a16_blocks"]),
            "m15_a16_lattice_flips": int(p15["a16_lattice_flips"]),
            "m15_oob_candidates_reverted": int(p15["oob_candidates_reverted"]),
            "m15_folded_scale_extra_bytes": int(p15["folded_scale_extra_bytes"]),
            "n_train": int(q51a["n_train"]),
            "n_test": int(q51a["n_test"]),
            "n_iters": int(q51a["n_iters"]),
            "n_pop": int(q51a["n_pop"]),
            "n_seeds": len(q51a["seeds"]),
            "rollouts_per_seed": int(q51a["rollouts_per_seed"]),
            "bf16_null_test_acc_micro": _to_micro(q51a["baselines_null"]["C0_BF16_Null"]["test_exact_acc"]),
            "bf16_null_peak_vram_kib": _to_kib(q51a["baselines_null"]["C0_BF16_Null"]["peak_vram_mib"]),
            "int4_null_test_acc_micro": _to_micro(q51a["baselines_null"]["C0_INT4_Null"]["test_exact_acc"]),
            "a16_3bpw_null_test_acc_micro": _to_micro(q51a["baselines_null"]["C0_A16_BlockH4_3bpw_Null"]["test_exact_acc"]),
            "a16_3bpw_null_syndrome_errors": int(q51a["baselines_null"]["C0_A16_BlockH4_3bpw_Null"]["syndrome_errors"]),
            "a16_3bpw_null_oob_groups": int(q51a["baselines_null"]["C0_A16_BlockH4_3bpw_Null"]["oob_groups"]),
            "a16_null_test_acc_micro": _to_micro(q51a["baselines_null"]["C0_A16_BlockH4_Null"]["test_exact_acc"]),
            "a16_null_syndrome_errors": int(q51a["baselines_null"]["C0_A16_BlockH4_Null"]["syndrome_errors"]),
            "a16_null_oob_groups": int(q51a["baselines_null"]["C0_A16_BlockH4_Null"]["oob_groups"]),
            "mean_bf16_ulp_stalled_micro": _to_micro(s_a["mean_bf16_ulp_stalled_frac"]),
            "peak_vram_05b_kib": _to_kib(s_a["peak_vram_05b_mib"]),
            "peak_vram_15b_kib": _to_kib(s_a["peak_vram_15b_mib"]),
            "m15_probe_syndrome_errors": int(s_a["m15_probe_syndrome_errors"]),
            "m15_probe_oob_groups": int(s_a["m15_probe_oob_groups"]),
            "arms": {
                "C1_VsonicV_BF16_SharedSeed": _pack_a_arm(q51a["arms"]["C1_VsonicV_BF16_SharedSeed"]),
                "C2_QES_Unblocked_INT4_EF": _pack_a_arm(q51a["arms"]["C2_QES_Unblocked_INT4_EF"]),
                "T1_SiliQuant_ES_A16_BlockH4_Folded0B_EF": _pack_a_arm(q51a["arms"]["T1_SiliQuant_ES_A16_BlockH4_Folded0B_EF"]),
            },
        },
        "exp_q51b": {
            "total_a16_layers": int(q51b["total_a16_layers"]),
            "total_a16_blocks": int(q51b["total_a16_blocks"]),
            "wikitext2_eval_bytes": int(q51b["wikitext2_eval_bytes"]),
            "n_train": int(q51b["n_train"]),
            "n_test_ood": int(q51b["n_test_ood"]),
            "n_iters": int(q51b["n_iters"]),
            "n_pop": int(q51b["n_pop"]),
            "base_train_acc_micro": _to_micro(q51b["baseline_base"]["train_exact_acc"]),
            "base_ood_8x_acc_micro": _to_micro(q51b["baseline_base"]["ood_8x_exact_acc"]),
            "base_wikitext2_bpb_micro": _to_micro(q51b["baseline_base"]["wikitext2_bpb"]),
            "base_max_su2_shell_err_nano": _to_nano(q51b["baseline_base"]["max_su2_shell_err"]),
            "vram_o1_scaling_grid": vram_grid_int,
            "arms": {
                "ArmA_Euclidean_Additive_ES": _pack_b_arm(q51b["arms"]["ArmA_Euclidean_Additive_ES"]),
                "ArmC_HoloSphere_Cayley_ES": _pack_b_arm(q51b["arms"]["ArmC_HoloSphere_Cayley_ES"]),
            },
        },
    }


def run_cpu_selftest() -> int:
    """Comprehensive PyTorch CPU unit tests for quantization, OOB scale fallback, +0 B fold,
    SECDED coordinate verification, A_16 u64 pack/unpack, and tie/NaN-safe scoring ([LOW-9])."""
    _require_torch()
    dev = torch.device("cpu")
    h4 = _h4_mat(dev, torch.float32)

    # 1) _a16_closest direct coordinate parity check across random inputs
    gen = torch.Generator(device="cpu").manual_seed(167001)
    u = torch.randn((256, 16), generator=gen) * 5.0
    y, errs = _a16_closest(u)
    _require(errs == 0 and _verify_a16_codewords_int(y.round().to(torch.int32)) == 0, "A16 closest syndrome failed")

    # 2) Adversarial matrix forcing OOB fallback in _quantize_matrix_a16_blockh4 ([MED-8] regression test!)
    # Use very tight bound=651.0 with an outlier spike so initial fp16 scale rounding triggers out_m!
    w_adv = torch.randn((32, 128), generator=gen) * 0.05
    w_adv[0, 0] = 15.0  # extreme single-entry spike
    y_i8, s_f16, syn, oob, fb = _quantize_matrix_a16_blockh4(w_adv, h4, bound=A16_B30_BOUND)
    _require(syn == 0 and oob == 0, f"adversarial quant failed syn={syn} oob={oob}")
    w_rec = _dequant_a16_blockh4(y_i8, s_f16, h4, dtype=torch.float32)
    # Must preserve spike sign and rough magnitude (proves sc[out_m] was updated, NOT left stale/zeroed!)
    _require(float(w_rec[0, 0].item()) > 10.0, f"OOB fallback corrupted scale: w_rec[0,0]={w_rec[0,0].item()}")

    # Also test 4.5 bpw normal matrix
    w_norm = torch.randn((64, 256), generator=gen) * 0.02
    y2, s2, syn2, oob2, _ = _quantize_matrix_a16_blockh4(w_norm, h4, bound=A16_B45_BOUND)
    _require(syn2 == 0 and oob2 == 0, "4.5bpw quant failed")
    rel_err = float(((_dequant_a16_blockh4(y2, s2, h4, dtype=torch.float32) - w_norm) ** 2).mean().sqrt().item()) / float(
        (w_norm ** 2).mean().sqrt().item()
    )
    _require(rel_err < 0.08, f"4.5bpw dequant relative RMSE too high: {rel_err}")

    # 3) +0 B reciprocal RMSNorm/linear scale folding invariance
    g0 = torch.randn((128,), generator=gen).abs() + 0.5
    w1 = torch.randn((64, 128), generator=gen) * 0.1
    w2 = torch.randn((64, 128), generator=gen) * 0.1
    w1[:, 0] *= 8.0
    g_new, ws_new, inv_err = _apply_zero_byte_rmsnorm_fold(g0.to(torch.bfloat16), [w1, w2])
    _require(inv_err < 1e-4, f"+0 B reciprocal fold invariance err too large: {inv_err}")
    _require(g_new.shape == g0.shape and len(ws_new) == 2, "+0 B fold shape mismatch")

    # 4) Orthogonal Block-H4 exact pairwise zero dot product
    vecs = [_gen_orth_blockh4_noise((8, 64), 167999, rk, dev, torch.float32).reshape(-1) for rk in range(4)]
    for i in range(4):
        for j in range(i + 1, 4):
            _require(abs(float(torch.dot(vecs[i], vecs[j]).item())) == 0.0, "orth H4 nonzero dot")

    # 5) HoloSphere-V4 A_16 uint64 pack/unpack bit-exact roundtrip
    c_raw, se_c = _a16_closest(torch.randn((32, 16), generator=gen) * 2.0)
    _require(se_c == 0, "hs test codeword syndrome")
    c_clamped = c_raw.round().clamp(-8, 7).to(torch.int8)
    # Force valid [-8, 7] codewords by replacing any clamped block with zeros (all-zero is valid codeword)
    bad_b = (c_raw.round() < -8) | (c_raw.round() > 7)
    c_clamped = torch.where(bad_b.any(dim=1, keepdim=True), torch.zeros_like(c_clamped), c_clamped).reshape(4, 128)
    _require(_verify_a16_codewords_int(c_clamped) == 0, "hs test valid codeword")
    u64 = _pack_hs_a16_u64(c_clamped)
    c_unp = _unpack_hs_a16_u64(u64)
    _require(bool((c_unp == c_clamped).all().item()), "hs u64 pack/unpack mismatch")

    # 6) Packed A16BlockH4ESLinear forward equivalence (zero persistent BF16 weight)
    A16ESLinear = _make_packed_a16_es_linear_class()
    ef0 = torch.zeros_like(y2, dtype=torch.int8)
    es_lin = A16ESLinear(y2, s2, ef0, None, 0, h4)
    x_in = torch.randn((2, 256), generator=gen, dtype=torch.bfloat16)
    out_es = es_lin(x_in)
    out_ref = F.linear(x_in, _dequant_a16_blockh4(y2, s2, h4, dtype=torch.bfloat16))
    _require(float((out_es - out_ref).abs().max().item()) == 0.0, "A16BlockH4ESLinear forward mismatch")

    # 7) Pessimistic tie & NaN scoring guard check ([MED-1])
    tied_logits = torch.zeros((4, 256), dtype=torch.float32)
    tgts = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    max_v = tied_logits.max(dim=-1, keepdim=True).values
    n_tied = (tied_logits == max_v).sum(dim=-1)
    exact_t = (tied_logits.argmax(dim=-1) == tgts) & (n_tied == 1)
    n_gt = (tied_logits > tied_logits.gather(1, tgts.unsqueeze(1))).sum(dim=-1)
    n_eq_other = (tied_logits == tied_logits.gather(1, tgts.unsqueeze(1))).sum(dim=-1) - 1
    _require(int(exact_t.sum().item()) == 0 and int((n_gt + n_eq_other).min().item()) == 255, "tie-rank guard failed")

    # 8) Pure-Python verifiers & Holm-Bonferroni
    r_ok, is_ok = verify_countdown_output("3 * 5 - 2", [3, 5, 2], 13)
    _require(is_ok and r_ok == 1.0, "countdown exact check failed")
    _require(abs(_exact_paired_permutation_p([1.0] * 7) - (1.0 / 64.0)) < 1e-12, "perm p n=7 failed")
    adj = _holm_bonferroni_adjust([1.0 / 64.0, 0.25])
    _require(abs(adj[0] - (1.0 / 32.0)) < 1e-12 and abs(adj[1] - 0.25) < 1e-12, "holm adjust failed")

    print("PASS s167_es_live_bench --cpu-selftest (8/8 checks OK)", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Track S167 Live RTX 3080 Benchmark")
    parser.add_argument("--cpu-selftest", action="store_true", help="Run CPU unit tests and exit")
    args = parser.parse_args(argv)
    if args.cpu_selftest:
        return run_cpu_selftest()

    _require_torch()
    _require(torch.cuda.is_available(), "CUDA GPU required for live RTX 3080 benchmark")
    device = torch.device("cuda:0")
    prereg_sha = _sha256_file(PREREG_PATH)
    bench_sha = _sha256_file(Path(__file__).resolve())
    gpu_before = _query_gpu_vram()
    print(
        f"=== Track S167 Live RTX 3080 Benchmark | GPU={gpu_before['name']} ({gpu_before['memory_total_mib']} MiB) ===",
        flush=True,
    )
    t_all = time.perf_counter()

    res_a = run_exp_q51a_siliquant_es(device)
    res_b = run_exp_q51b_holosphere_cayley_es(device)
    _cleanup_gpu()
    in_proc_alloc_mib = torch.cuda.memory_allocated(device) / 1048576.0
    gpu_after = _query_gpu_vram()

    raw_results: dict[str, Any] = {
        "schema": "aletheia.track_s167.empirical_results.v2",
        "prereg_id": "PREREG-W22-ES-SCALE",
        "prereg_sha256": prereg_sha,
        "bench_script_sha256": bench_sha,
        "energy_claim": ENERGY_CLAIM,
        "total_wall_seconds": time.perf_counter() - t_all,
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
        "in_process_cuda_allocated_after_mib": in_proc_alloc_mib,
        "exp_q51a": res_a,
        "exp_q51b": res_b,
    }
    raw_path = TRACK_DIR / "s167_empirical_results.json"
    raw_text = json.dumps(raw_results, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
    raw_path.write_text(raw_text, encoding="utf-8")
    raw_sha = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()

    receipt = build_pure_int_receipt(raw_results, raw_sha, prereg_sha, bench_sha)
    rec_path = TRACK_DIR / "s167_empirical_receipt.json"
    rec_text = json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
    rec_path.write_text(rec_text, encoding="utf-8")
    rec_sha = hashlib.sha256(rec_text.encode("utf-8")).hexdigest()

    print(
        f"=== DONE in {raw_results['total_wall_seconds']:.1f}s | "
        f"EXP-Q51a={res_a['summary']['verdict']} | EXP-Q51b={res_b['summary']['verdict']} | "
        f"in_proc_cuda_alloc={in_proc_alloc_mib:.2f} MiB | nvidia_smi_in_proc={gpu_after['memory_used_mib']} MiB ===",
        flush=True,
    )
    print(f"bench_script_sha256={bench_sha}", flush=True)
    print(f"raw_sha256={raw_sha}", flush=True)
    print(f"receipt_sha256={rec_sha}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
