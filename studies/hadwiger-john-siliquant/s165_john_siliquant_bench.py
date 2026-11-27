#!/usr/bin/env python3
"""Track S165 (EXP-Q47b, PREREG-W22-HADWIGER-SILIQUANT) Live RTX 3080 Benchmark.

Transfers dimension-independent convex-covering operators from arXiv:2611.32334v2
(Hadwiger 3D) directly onto Moissanite-Labs/siliquant (d=8 E_8 and d=16 A_16 Block-H4)
using real cached Qwen2.5-0.5B and Qwen2.5-1.5B weights and held-out WikiText-2 + Python
evaluation corpora.

Strict governance:
  - Real cached weights from ~/.cache/huggingface/ only (WEIGHTS=0 inside expedition/).
  - Exact CS-08 matched bit-budget accounting (3.000 bpw and 4.500 bpw).
  - Zero uncontrolled laptop energy/J/W claims (UNCONTROLLED_LAPTOP_NOT_CLAIMED).
  - Zero bare asserts; strictly seeded RNG; full GPU memory cleanup between models.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

EVAL_SEED = 20261126165
SEQ_LEN = 512
N_EVAL_CHUNKS = 8
N_REAL_CAL_CHUNKS = 12
QUANT_ROW_CHUNK = 512
LOGIT_TOK_CHUNK = 128

# Exact integer theta-series ball bounds for g=128 (certified in JohnQuantOperators & oracle):
#   3.000 bpw (384 bits/group): B=368 + 16-bit fp16 scale -> A16=651, E8=420
#   4.500 bpw (576 bits/group): B=560 + 16-bit fp16 scale -> A16=5213, E8=3380
#   Matched-bpw reduced bulk bounds for A16 g=128:
#     B=357 (-11 bits/group for 0.25% @ 32b CSR or 1/19 polar-cap 4.5b blocks): R^2 = 577
#     B=352 (-16 bits/group for fp16 Helly-Chebyshev center v_g):              R^2 = 547
#     B=549 (-11 bits/group at 4.5 bpw for 0.25% @ 32b CSR):                   R^2 = 4629
#     B=544 (-16 bits/group at 4.5 bpw for fp16 Helly-Chebyshev center v_g):   R^2 = 4385
BALL_BOUNDS: dict[str, dict[str, float]] = {
    "3.0": {"A16": 651.0, "E8": 420.0},
    "4.5": {"A16": 5213.0, "E8": 3380.0},
}
A16_BULK_BOUND_CSR = {"3.0": 577.0, "4.5": 4629.0}
A16_BULK_BOUND_CHEB_CENTER = {"3.0": 547.0, "4.5": 4385.0}

STDLIB_PY_MODULES = (
    "fractions.py",
    "difflib.py",
    "decimal.py",
    "statistics.py",
    "argparse.py",
    "json/decoder.py",
)


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def _query_gpu_memory_info() -> dict[str, Any]:
    """Query GPU identity and VRAM usage ONLY (zero power/energy/watt telemetry)."""
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


def _load_python_eval_text() -> tuple[list[str], str]:
    stdlib_dir = Path(json.__file__).resolve().parent.parent
    chunks: list[str] = []
    h = hashlib.sha256()
    for rel in STDLIB_PY_MODULES:
        raw = (stdlib_dir / rel).read_bytes()
        h.update(rel.encode("utf-8") + b"\0" + raw)
        chunks.append(raw.decode("utf-8", errors="ignore"))
    return chunks, h.hexdigest()


def _tokenize_chunks(tokenizer: Any, text: str, n_chunks: int, seq_len: int) -> torch.Tensor:
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    need = n_chunks * seq_len
    if ids.numel() < need:
        reps = (need // ids.numel()) + 1
        ids = ids.repeat(reps)
    return ids[:need].reshape(n_chunks, seq_len).contiguous()


def _tokenize_balanced_python_chunks(
    tokenizer: Any, module_texts: list[str], n_chunks: int, seq_len: int
) -> torch.Tensor:
    """Take a balanced token slice from every stdlib module so fractions.py cannot starve the corpus."""
    need = n_chunks * seq_len
    per_mod = need // len(module_texts)
    rem = need % len(module_texts)
    parts: list[torch.Tensor] = []
    for idx, txt in enumerate(module_texts):
        ids = tokenizer(txt, return_tensors="pt", add_special_tokens=False).input_ids[0]
        take = per_mod + (1 if idx < rem else 0)
        if ids.numel() < take:
            reps = (take // ids.numel()) + 1
            ids = ids.repeat(reps)
        parts.append(ids[:take])
    return torch.cat(parts, dim=0).reshape(n_chunks, seq_len).contiguous()


def _sylvester_hadamard(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    h = torch.tensor([[1.0]], device=device, dtype=dtype)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    return h / math.sqrt(float(n))


def _block_hadamard(d: int, b: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    hb = _sylvester_hadamard(b, device, dtype)
    return torch.block_diag(*([hb] * (d // b)))


def _seeded_random_orth(d: int, seed: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    mat = torch.randn((d, d), generator=gen, dtype=torch.float64)
    q, r = torch.linalg.qr(mat)
    signs = torch.sign(torch.diag(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    q = q * signs.unsqueeze(0)
    return q.to(device=device, dtype=dtype)


def _apply_block_rot(w: torch.Tensor, q_rot: torch.Tensor) -> torch.Tensor:
    d = q_rot.shape[0]
    return (w.reshape(w.shape[0], -1, d) @ q_rot.T).reshape(w.shape)


_C16_11_4_CACHE: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}


def _get_c16_11_4_codewords(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return exact (2048, 16) parity table for extended Hamming code C_[16, 11, 4]."""
    key = (device, dtype)
    if key in _C16_11_4_CACHE:
        return _C16_11_4_CACHE[key]
    info_cols = (3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15)
    rows: list[list[int]] = []
    for msg in range(2048):
        bits = [0] * 16
        for b_idx, col in enumerate(info_cols):
            bits[col] = (msg >> b_idx) & 1
        bits[1] = sum(bits[j] for j in range(16) if (j & 1) and j != 1) & 1
        bits[2] = sum(bits[j] for j in range(16) if (j & 2) and j != 2) & 1
        bits[4] = sum(bits[j] for j in range(16) if (j & 4) and j != 4) & 1
        bits[8] = sum(bits[j] for j in range(16) if (j & 8) and j != 8) & 1
        bits[0] = sum(bits[1:]) & 1
        rows.append(bits)
    tbl = torch.tensor(rows, device=device, dtype=dtype)
    _C16_11_4_CACHE[key] = tbl
    return tbl


def _a16_closest(u: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Exact ML nearest lattice point in A_16 = 2 Z^16 + C_[16,11,4] with reconstructed syndrome check."""
    e_val = 2.0 * torch.round(u * 0.5)
    o_val = 2.0 * torch.floor(u * 0.5) + 1.0
    de = (u - e_val) ** 2
    do = (u - o_val) ** 2
    diff = do - de

    cw = _get_c16_11_4_codewords(u.device, u.dtype)
    n_rows = u.shape[0]
    best_idx_parts: list[torch.Tensor] = []
    for r0 in range(0, n_rows, 16384):
        r1 = min(r0 + 16384, n_rows)
        best_idx_parts.append((diff[r0:r1] @ cw.T).argmin(dim=1))
    best_idx = torch.cat(best_idx_parts, dim=0)
    best_cw = cw[best_idx] > 0.5
    out = torch.where(best_cw, o_val, e_val)

    rec_int = torch.round(out).to(torch.int64)
    non_int = ((out - rec_int.to(out.dtype)).abs() > 1e-4).any(dim=1)
    rec_b = rec_int & 1
    vp0 = rec_b.sum(dim=1) & 1
    vs0 = rec_b[:, 1::2].sum(dim=1) & 1
    vs1 = (rec_b[:, 2::4].sum(dim=1) + rec_b[:, 3::4].sum(dim=1)) & 1
    vs2 = (rec_b[:, 4:8].sum(dim=1) + rec_b[:, 12:16].sum(dim=1)) & 1
    vs3 = rec_b[:, 8:16].sum(dim=1) & 1
    errs = int((non_int | ((vp0 | vs0 | vs1 | vs2 | vs3) != 0)).sum().item())
    return out, errs


def _e8_closest(u8: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Exact nearest lattice point in E_8 = D_8 U (D_8 + (1/2)^8) with live parity check."""
    f1 = torch.round(u8)
    odd1 = (torch.sum(f1.to(torch.int64), dim=1) & 1) != 0
    if odd1.any():
        diff1 = u8 - f1
        idx1 = diff1.abs().argmax(dim=1)
        step1 = torch.where(diff1.gather(1, idx1.unsqueeze(1)).squeeze(1) >= 0, 1.0, -1.0)
        f1 = f1.clone()
        rows1 = torch.nonzero(odd1, as_tuple=False).squeeze(1)
        f1[rows1, idx1[rows1]] += step1[rows1]

    u_s = u8 - 0.5
    f2i = torch.round(u_s)
    odd2 = (torch.sum(f2i.to(torch.int64), dim=1) & 1) != 0
    if odd2.any():
        diff2 = u_s - f2i
        idx2 = diff2.abs().argmax(dim=1)
        step2 = torch.where(diff2.gather(1, idx2.unsqueeze(1)).squeeze(1) >= 0, 1.0, -1.0)
        f2i = f2i.clone()
        rows2 = torch.nonzero(odd2, as_tuple=False).squeeze(1)
        f2i[rows2, idx2[rows2]] += step2[rows2]
    f2 = f2i + 0.5

    d1 = ((u8 - f1) ** 2).sum(dim=1)
    d2 = ((u8 - f2) ** 2).sum(dim=1)
    out = torch.where((d1 <= d2).unsqueeze(1), f1, f2)

    doubled = torch.round(out * 2.0).to(torch.int64)
    non_half = ((out * 2.0 - doubled.to(out.dtype)).abs() > 1e-4).any(dim=1)
    par0 = doubled[:, 0:1] & 1
    mixed_par = ((doubled & 1) != par0).any(dim=1)
    bad_sum = (doubled.sum(dim=1) & 3) != 0
    errs = int((non_half | mixed_par | bad_sum).sum().item())
    return out, errs


def _quantize_rtn_chunk(
    w_chunk: torch.Tensor,
    bpw_key: str,
    group_size: int,
) -> tuple[torch.Tensor, int, int]:
    """Unwhitened symmetric RTN matched strictly <= 3.000 bpw (qmax=3, 2.932 bpw) or <= 4.500 bpw (qmax=9, 4.373 bpw)."""
    qmax = 3.0 if bpw_key == "3.0" else 9.0
    xg = w_chunk.reshape(-1, group_size)
    amax = xg.abs().amax(dim=1).clamp_min(1e-30)
    scale = (amax / qmax).to(torch.float16).to(torch.float32).clamp_min(1e-30)
    q = torch.clamp(torch.round(xg / scale.unsqueeze(1)), -qmax, qmax)
    wq = (q * scale.unsqueeze(1)).reshape(w_chunk.shape)
    return wq, 0, 0


def _quantize_ball_chunk(
    w_chunk: torch.Tensor,
    bound: float,
    lattice: str,
    group_size: int,
    q_rot: torch.Tensor | None = None,
    colscale: torch.Tensor | None = None,
    cheb_minimax_scale: bool = False,
    cheb_center: bool = False,
    per_group_bounds: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int, int]:
    """Quantize a weight row chunk onto A_16 or E_8 ball with optional Helly-Chebyshev operators."""
    x = w_chunk.clone()
    if colscale is not None:
        x = x * colscale.unsqueeze(0)

    # Optional Helly-Chebyshev asymmetric center v_g = 0.5*(max + min) before rotation
    centers: torch.Tensor | None = None
    if cheb_center:
        xg_pre = x.reshape(-1, group_size)
        v_raw = 0.5 * (xg_pre.amax(dim=1) + xg_pre.amin(dim=1))
        centers = v_raw.to(torch.float16).to(torch.float32)
        x = (xg_pre - centers.unsqueeze(1)).reshape(w_chunk.shape)

    if q_rot is not None:
        x = _apply_block_rot(x, q_rot)

    xg = x.reshape(-1, group_size)
    n_groups = xg.shape[0]
    if per_group_bounds is not None:
        b_vec = per_group_bounds.to(device=w_chunk.device, dtype=w_chunk.dtype)
    else:
        b_vec = torch.full((n_groups,), float(bound), device=w_chunk.device, dtype=w_chunk.dtype)

    r_ball = torch.sqrt(b_vec)
    blk_dim = 16.0 if lattice == "A16" else 8.0
    margin = math.sqrt(group_size / 4.0 + 1.5 * (group_size / blk_dim))

    g_norm = torch.sqrt((xg ** 2).sum(dim=1)).clamp_min(1e-30)
    lo = g_norm / (r_ball + margin) * 0.99
    hi = g_norm / torch.clamp(r_ball - margin, min=0.05 * r_ball) * 1.01

    for _ in range(16):
        mid = torch.sqrt(lo * hi)
        scaled = xg / mid.unsqueeze(1)
        if lattice == "A16":
            y_cur, _ = _a16_closest(scaled.reshape(-1, 16))
        else:
            y_cur, _ = _e8_closest(scaled.reshape(-1, 8))
        y_cur = y_cur.reshape(-1, group_size)
        ok = (y_cur ** 2).sum(dim=1) <= b_vec + 1e-6
        hi = torch.where(ok, mid, hi)
        lo = torch.where(ok, lo, mid)

    scaled_f = xg / hi.unsqueeze(1)
    if lattice == "A16":
        y_fin, syn_errs = _a16_closest(scaled_f.reshape(-1, 16))
    else:
        y_fin, syn_errs = _e8_closest(scaled_f.reshape(-1, 8))
    y_fin = y_fin.reshape(-1, group_size)

    scales_best = hi.to(torch.float16).to(torch.float32).clamp_min(1e-30)

    out_m = (y_fin ** 2).sum(dim=1) > b_vec + 1e-6
    fallback_count = int(out_m.sum().item())
    if out_m.any():
        xo = scaled_f[out_m]
        bo = b_vec[out_m]
        s_lo = torch.zeros(xo.shape[0], device=w_chunk.device, dtype=w_chunk.dtype)
        s_hi = torch.ones(xo.shape[0], device=w_chunk.device, dtype=w_chunk.dtype)
        for _ in range(14):
            sm = 0.5 * (s_lo + s_hi)
            if lattice == "A16":
                yt, _ = _a16_closest((xo * sm.unsqueeze(1)).reshape(-1, 16))
            else:
                yt, _ = _e8_closest((xo * sm.unsqueeze(1)).reshape(-1, 8))
            yt = yt.reshape(-1, group_size)
            good = (yt ** 2).sum(dim=1) <= bo + 1e-6
            s_lo = torch.where(good, sm, s_lo)
            s_hi = torch.where(good, s_hi, sm)
        s_raw = (hi[out_m] / s_lo.clamp_min(1e-12))
        s_f16 = s_raw.to(torch.float16)
        s_up = torch.nextafter(s_f16, torch.full_like(s_f16, float("inf")))
        s_adj = torch.where(s_f16.to(torch.float32) < s_raw, s_up.to(torch.float32), s_f16.to(torch.float32)).clamp_min(1e-30)
        if lattice == "A16":
            yf2, se2 = _a16_closest((xg[out_m] / s_adj.unsqueeze(1)).reshape(-1, 16))
        else:
            yf2, se2 = _e8_closest((xg[out_m] / s_adj.unsqueeze(1)).reshape(-1, 8))
        y_fin[out_m] = yf2.reshape(-1, group_size)
        scales_best[out_m] = s_adj
        syn_errs += se2

    # Optional exact +0 B Helly-LP minimax Chebyshev gauge-scale refinement:
    # Over strictly feasible scales s >= s_min (still inside exact ball ||Y||^2 <= b_vec),
    # choose the exact scale minimizing max_j |x_j - s * y_j| (L_inf minimax error).
    if cheb_minimax_scale:
        # Evaluate baseline L_inf error in original (unrotated) weighted coordinate space
        def _linf_err(y_cand: torch.Tensor, s_cand: torch.Tensor) -> torch.Tensor:
            rec = (y_cand * s_cand.unsqueeze(1)).reshape(w_chunk.shape)
            if q_rot is not None:
                rec = _apply_block_rot(rec, q_rot.T)
            rec_g = rec.reshape(-1, group_size)
            orig_g = (w_chunk * colscale.unsqueeze(0) if colscale is not None else w_chunk).reshape(-1, group_size)
            if centers is not None:
                orig_g = orig_g - centers.unsqueeze(1)
            return (rec_g - orig_g).abs().amax(dim=1)

        best_linf = _linf_err(y_fin, scales_best)
        # Candidate multipliers around [0.96, 1.40] (only accepting strictly feasible ball points!)
        for mult in (0.96, 0.98, 1.02, 1.05, 1.09, 1.14, 1.20, 1.28, 1.38):
            s_try = (hi * mult).to(torch.float16).to(torch.float32).clamp_min(1e-30)
            if lattice == "A16":
                y_try, se_try = _a16_closest((xg / s_try.unsqueeze(1)).reshape(-1, 16))
            else:
                y_try, se_try = _e8_closest((xg / s_try.unsqueeze(1)).reshape(-1, 8))
            syn_errs += se_try
            y_try = y_try.reshape(-1, group_size)
            feas = (y_try ** 2).sum(dim=1) <= b_vec + 1e-6
            cand_linf = _linf_err(y_try, s_try)
            better = feas & (cand_linf < best_linf)
            if better.any():
                y_fin = torch.where(better.unsqueeze(1), y_try, y_fin)
                scales_best = torch.where(better, s_try, scales_best)
                best_linf = torch.where(better, cand_linf, best_linf)

    oob_groups = int(((y_fin ** 2).sum(dim=1) > b_vec + 1e-6).sum().item())
    wq = (y_fin * scales_best.unsqueeze(1)).reshape(w_chunk.shape)
    if q_rot is not None:
        wq = _apply_block_rot(wq, q_rot.T)
    if centers is not None:
        wq = (wq.reshape(-1, group_size) + centers.unsqueeze(1)).reshape(w_chunk.shape)
    if colscale is not None:
        wq = wq / colscale.unsqueeze(0)
    return wq, syn_errs, oob_groups, fallback_count


def _quantize_matrix(
    w_f32: torch.Tensor,
    bound: float,
    lattice: str,
    group_size: int,
    q_rot: torch.Tensor | None = None,
    colscale: torch.Tensor | None = None,
    col_perm: torch.Tensor | None = None,
    cheb_minimax_scale: bool = False,
    cheb_center: bool = False,
    rtn_bpw_key: str | None = None,
    group_bounds_full: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int, int, int]:
    wf_use = w_f32
    cs_use = colscale
    inv_perm: torch.Tensor | None = None
    if col_perm is not None:
        wf_use = w_f32[:, col_perm]
        if colscale is not None:
            cs_use = colscale[col_perm]
        inv_perm = torch.empty_like(col_perm)
        inv_perm[col_perm] = torch.arange(col_perm.numel(), device=col_perm.device, dtype=col_perm.dtype)

    m = wf_use.shape[0]
    g_per_row = wf_use.shape[1] // group_size
    wq_parts: list[torch.Tensor] = []
    syn_tot = 0
    oob_tot = 0
    fb_tot = 0
    for r0 in range(0, m, QUANT_ROW_CHUNK):
        r1 = min(r0 + QUANT_ROW_CHUNK, m)
        if rtn_bpw_key is not None:
            wq_c, syn_c, oob_c = _quantize_rtn_chunk(wf_use[r0:r1], rtn_bpw_key, group_size)
            fb_c = 0
        else:
            gb_slice = None
            if group_bounds_full is not None:
                gb_slice = group_bounds_full[r0 * g_per_row : r1 * g_per_row]
            wq_c, syn_c, oob_c, fb_c = _quantize_ball_chunk(
                wf_use[r0:r1],
                bound,
                lattice,
                group_size,
                q_rot=q_rot,
                colscale=cs_use,
                cheb_minimax_scale=cheb_minimax_scale,
                cheb_center=cheb_center,
                per_group_bounds=gb_slice,
            )
        wq_parts.append(wq_c)
        syn_tot += syn_c
        oob_tot += oob_c
        fb_tot += fb_c
    wq = torch.cat(wq_parts, dim=0)
    if inv_perm is not None:
        wq = wq[:, inv_perm]
    return wq, syn_tot, oob_tot, fb_tot


# ---------------------------------------------------------------------------
# Permutation Constructors (Matching Track S73 Exactly)
# ---------------------------------------------------------------------------
def _make_quad_snake_perm(metric: torch.Tensor, group_size: int, block_w: int = 4) -> torch.Tensor:
    k = int(metric.numel())
    n_balls = k // group_size
    b_per_ball = group_size // block_w
    sorted_idx = torch.argsort(metric, descending=True, stable=True).cpu().tolist()
    blocks = [sorted_idx[q * block_w : (q + 1) * block_w] for q in range(k // block_w)]
    balls: list[list[int]] = [[] for _ in range(n_balls)]
    for r in range(b_per_ball):
        for t in range(n_balls):
            q = r * n_balls + t
            b = t if (r % 2 == 0) else (n_balls - 1 - t)
            balls[b].extend(blocks[q])
    flat = [c for ball in balls for c in ball]
    return torch.tensor(flat, dtype=torch.int64)


def _make_quad_lpt_perm(metric: torch.Tensor, group_size: int, block_w: int = 4, load_power: float = 0.5) -> torch.Tensor:
    k = int(metric.numel())
    n_balls = k // group_size
    b_per_ball = group_size // block_w
    m_cpu = metric.detach().to(torch.float64).cpu()
    sorted_idx = torch.argsort(m_cpu, descending=True, stable=True).tolist()
    vals = m_cpu.clamp_min(0.0).pow(load_power).tolist()
    blocks = [sorted_idx[q * block_w : (q + 1) * block_w] for q in range(k // block_w)]
    block_loads = [sum(vals[j] for j in blk) for blk in blocks]
    balls: list[list[int]] = [[] for _ in range(n_balls)]
    ball_loads = [0.0] * n_balls
    ball_counts = [0] * n_balls
    for q, blk in enumerate(blocks):
        best_b = -1
        best_load = float("inf")
        for b in range(n_balls):
            if ball_counts[b] < b_per_ball and ball_loads[b] < best_load:
                best_load = ball_loads[b]
                best_b = b
        balls[best_b].extend(blk)
        ball_loads[best_b] += block_loads[q]
        ball_counts[best_b] += 1
    flat = [c for ball in balls for c in ball]
    return torch.tensor(flat, dtype=torch.int64)


# ---------------------------------------------------------------------------
# John/Lowner Halved-Difference-Body +0 B Diagonal Folding & Polar-Cap Split
# ---------------------------------------------------------------------------
def _geom_norm_clamp(vec: torch.Tensor, lo: float = 0.125, hi: float = 8.0) -> torch.Tensor:
    v = vec.to(torch.float64).clamp_min(1e-12)
    gm = torch.exp(torch.log(v).mean())
    return (v / gm).clamp(min=lo, max=hi).to(torch.float32)


def _compute_john_scales_for_model(
    orig_weights_cpu: dict[str, torch.Tensor],
    h_cal_real: dict[str, torch.Tensor],
    n_blocks: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    use_hessian: bool,
) -> dict[str, torch.Tensor]:
    """Compute strictly +0 B foldable John/Lowner halved-difference-body diagonal scales.

    For each layer:
      - q_proj, k_proj, v_proj share ONE scale d_qkv folded at +0 B into input_layernorm.weight.
      - gate_proj, up_proj share ONE scale d_gu folded at +0 B into post_attention_layernorm.weight.
      - down_proj scale d_down folds at +0 B into up_proj.weight rows.
      - o_proj scale d_o is GQA-group-shared across repeated query heads so it folds at +0 B
        into v_proj.weight rows and v_proj.bias.
    """
    scales: dict[str, torch.Tensor] = {}
    gqa_rep = n_heads // n_kv_heads
    for l_i in range(n_blocks):
        # 1) Shared {q, k, v} -> input_layernorm
        q_name = f"model.layers.{l_i}.self_attn.q_proj"
        k_name = f"model.layers.{l_i}.self_attn.k_proj"
        v_name = f"model.layers.{l_i}.self_attn.v_proj"
        w_qkv = torch.cat(
            [orig_weights_cpu[q_name].float(), orig_weights_cpu[k_name].float(), orig_weights_cpu[v_name].float()],
            dim=0,
        )
        s_qkv = 0.5 * (w_qkv.amax(dim=0) - w_qkv.amin(dim=0)).clamp_min(1e-6)
        if use_hessian:
            hv = h_cal_real[q_name].float().clamp_min(1e-12)
            raw = (hv / hv.mean()).pow(0.25) / s_qkv.pow(0.5)
        else:
            raw = 1.0 / s_qkv.pow(0.5)
        d_qkv = _geom_norm_clamp(raw)
        scales[q_name] = d_qkv
        scales[k_name] = d_qkv
        scales[v_name] = d_qkv
        del w_qkv

        # 2) Shared {gate, up} -> post_attention_layernorm
        g_name = f"model.layers.{l_i}.mlp.gate_proj"
        u_name = f"model.layers.{l_i}.mlp.up_proj"
        w_gu = torch.cat([orig_weights_cpu[g_name].float(), orig_weights_cpu[u_name].float()], dim=0)
        s_gu = 0.5 * (w_gu.amax(dim=0) - w_gu.amin(dim=0)).clamp_min(1e-6)
        if use_hessian:
            hv = h_cal_real[g_name].float().clamp_min(1e-12)
            raw = (hv / hv.mean()).pow(0.25) / s_gu.pow(0.5)
        else:
            raw = 1.0 / s_gu.pow(0.5)
        d_gu = _geom_norm_clamp(raw)
        scales[g_name] = d_gu
        scales[u_name] = d_gu
        del w_gu

        # 3) down_proj -> folds at +0 B into up_proj output rows
        d_name = f"model.layers.{l_i}.mlp.down_proj"
        w_d = orig_weights_cpu[d_name].float()
        s_d = 0.5 * (w_d.amax(dim=0) - w_d.amin(dim=0)).clamp_min(1e-6)
        if use_hessian:
            hv = h_cal_real[d_name].float().clamp_min(1e-12)
            raw = (hv / hv.mean()).pow(0.25) / s_d.pow(0.5)
        else:
            raw = 1.0 / s_d.pow(0.5)
        scales[d_name] = _geom_norm_clamp(raw)
        del w_d

        # 4) o_proj -> GQA-shared across query heads -> folds at +0 B into v_proj rows & bias
        o_name = f"model.layers.{l_i}.self_attn.o_proj"
        w_o = orig_weights_cpu[o_name].float()
        s_o = 0.5 * (w_o.amax(dim=0) - w_o.amin(dim=0)).clamp_min(1e-6)
        # Pool across GQA-repeated query heads so one scale per (kv_head, head_dim) folds into v_proj
        s_o_gqa = s_o.reshape(n_kv_heads, gqa_rep, head_dim).mean(dim=1)
        if use_hessian:
            hv = h_cal_real[o_name].float().clamp_min(1e-12)
            hv_gqa = hv.reshape(n_kv_heads, gqa_rep, head_dim).mean(dim=1)
            raw_gqa = (hv_gqa / hv_gqa.mean()).pow(0.25) / s_o_gqa.pow(0.5)
        else:
            raw_gqa = 1.0 / s_o_gqa.pow(0.5)
        d_o_gqa = _geom_norm_clamp(raw_gqa.reshape(-1)).reshape(n_kv_heads, 1, head_dim)
        d_o = d_o_gqa.expand(n_kv_heads, gqa_rep, head_dim).reshape(-1).contiguous()
        scales[o_name] = d_o
        del w_o
    return scales


@torch.no_grad()
def _verify_physical_zero_byte_folding_invariance(
    model: torch.nn.Module,
    orig_weights_cpu: dict[str, torch.Tensor],
    orig_norms_cpu: dict[str, torch.Tensor],
    orig_vbias_cpu: dict[str, torch.Tensor],
    john_scales: dict[str, torch.Tensor],
    wt_eval_chunks: torch.Tensor,
    ref_wt_lp: list[torch.Tensor],
    n_blocks: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    device: torch.device,
) -> dict[str, float]:
    """Physically absorb inverse diagonal scales into RMSNorm, up_proj rows, and v_proj rows/bias
    and verify exact pre-quantization forward invariance (+0 B runtime overhead!)."""
    gqa_rep = n_heads // n_kv_heads
    mod_map = dict(model.named_modules())
    for l_i in range(n_blocks):
        in_norm = mod_map[f"model.layers.{l_i}.input_layernorm"]
        post_norm = mod_map[f"model.layers.{l_i}.post_attention_layernorm"]
        q_m = mod_map[f"model.layers.{l_i}.self_attn.q_proj"]
        k_m = mod_map[f"model.layers.{l_i}.self_attn.k_proj"]
        v_m = mod_map[f"model.layers.{l_i}.self_attn.v_proj"]
        o_m = mod_map[f"model.layers.{l_i}.self_attn.o_proj"]
        g_m = mod_map[f"model.layers.{l_i}.mlp.gate_proj"]
        u_m = mod_map[f"model.layers.{l_i}.mlp.up_proj"]
        d_m = mod_map[f"model.layers.{l_i}.mlp.down_proj"]

        d_qkv = john_scales[f"model.layers.{l_i}.self_attn.q_proj"].to(device)
        d_gu = john_scales[f"model.layers.{l_i}.mlp.gate_proj"].to(device)
        d_down = john_scales[f"model.layers.{l_i}.mlp.down_proj"].to(device)
        d_o = john_scales[f"model.layers.{l_i}.self_attn.o_proj"].to(device)
        d_v_out = d_o.reshape(n_kv_heads, gqa_rep, head_dim)[:, 0, :].reshape(-1)

        in_norm.weight.copy_((orig_norms_cpu[f"model.layers.{l_i}.input_layernorm"].to(device).float() / d_qkv).to(torch.bfloat16))
        post_norm.weight.copy_((orig_norms_cpu[f"model.layers.{l_i}.post_attention_layernorm"].to(device).float() / d_gu).to(torch.bfloat16))

        q_m.weight.copy_((orig_weights_cpu[f"model.layers.{l_i}.self_attn.q_proj"].to(device).float() * d_qkv.unsqueeze(0)).to(torch.bfloat16))
        k_m.weight.copy_((orig_weights_cpu[f"model.layers.{l_i}.self_attn.k_proj"].to(device).float() * d_qkv.unsqueeze(0)).to(torch.bfloat16))
        v_w = orig_weights_cpu[f"model.layers.{l_i}.self_attn.v_proj"].to(device).float() * d_qkv.unsqueeze(0) / d_v_out.unsqueeze(1)
        v_m.weight.copy_(v_w.to(torch.bfloat16))
        v_b_key = f"model.layers.{l_i}.self_attn.v_proj.bias"
        if v_b_key in orig_vbias_cpu and v_m.bias is not None:
            v_m.bias.copy_((orig_vbias_cpu[v_b_key].to(device).float() / d_v_out).to(torch.bfloat16))
        o_m.weight.copy_((orig_weights_cpu[f"model.layers.{l_i}.self_attn.o_proj"].to(device).float() * d_o.unsqueeze(0)).to(torch.bfloat16))

        g_m.weight.copy_((orig_weights_cpu[f"model.layers.{l_i}.mlp.gate_proj"].to(device).float() * d_gu.unsqueeze(0)).to(torch.bfloat16))
        u_w = orig_weights_cpu[f"model.layers.{l_i}.mlp.up_proj"].to(device).float() * d_gu.unsqueeze(0) / d_down.unsqueeze(1)
        u_m.weight.copy_(u_w.to(torch.bfloat16))
        d_m.weight.copy_((orig_weights_cpu[f"model.layers.{l_i}.mlp.down_proj"].to(device).float() * d_down.unsqueeze(0)).to(torch.bfloat16))

    scores = _score_against_reference(model, wt_eval_chunks[:2], ref_wt_lp[:2], device)

    # Restore original norms and v_proj biases cleanly
    for l_i in range(n_blocks):
        mod_map[f"model.layers.{l_i}.input_layernorm"].weight.copy_(
            orig_norms_cpu[f"model.layers.{l_i}.input_layernorm"].to(device)
        )
        mod_map[f"model.layers.{l_i}.post_attention_layernorm"].weight.copy_(
            orig_norms_cpu[f"model.layers.{l_i}.post_attention_layernorm"].to(device)
        )
        v_b_key = f"model.layers.{l_i}.self_attn.v_proj.bias"
        if v_b_key in orig_vbias_cpu:
            mod_map[f"model.layers.{l_i}.self_attn.v_proj"].bias.copy_(orig_vbias_cpu[v_b_key].to(device))
    for n, w_cpu in orig_weights_cpu.items():
        mod_map[n].weight.copy_(w_cpu.to(device))
    torch.cuda.empty_cache()
    return {
        "prequant_fold_mean_kl": scores["mean_kl"],
        "prequant_fold_top1_match_pct": scores["top1_match_pct"],
        "extra_runtime_bytes": 0,
    }


def _leading_polar_direction(wf: torch.Tensor, h_sqrt: torch.Tensor, n_iters: int = 10) -> torch.Tensor:
    """Deterministic power iteration for top right singular vector w* of W diag(h^{1/2})."""
    v = h_sqrt / h_sqrt.norm().clamp_min(1e-30)
    for _ in range(n_iters):
        u = wf @ (v * h_sqrt)
        v_new = (wf.T @ u) * h_sqrt
        v = v_new / v_new.norm().clamp_min(1e-30)
    return v


def _collect_block_linears(model: torch.nn.Module) -> list[tuple[str, torch.nn.Linear]]:
    return [
        (name, mod)
        for name, mod in model.named_modules()
        if isinstance(mod, torch.nn.Linear) and name.startswith("model.layers.")
    ]


def _proj_kind(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def _layer_idx(name: str) -> int:
    return int(name.split(".")[2])


def _snapshot_protected_non_block_tensors(model: torch.nn.Module) -> dict[str, bytes]:
    """Snapshot embeddings, lm_head, and final model.norm (must never be altered)."""
    snaps: dict[str, bytes] = {}
    for name, param in model.named_parameters():
        if not name.startswith("model.layers."):
            snaps[name] = hashlib.sha256(param.detach().cpu().view(torch.uint8).numpy().tobytes()).digest()
    return snaps


def _count_protected_diffs(model: torch.nn.Module, snaps: dict[str, bytes]) -> int:
    diffs = 0
    for name, param in model.named_parameters():
        if name in snaps:
            cur = hashlib.sha256(param.detach().cpu().view(torch.uint8).numpy().tobytes()).digest()
            if cur != snaps[name]:
                diffs += 1
    return diffs


@torch.no_grad()
def _collect_reference_logits(
    model: torch.nn.Module, chunks: torch.Tensor, device: torch.device
) -> tuple[list[torch.Tensor], float]:
    ref_log_probs: list[torch.Tensor] = []
    total_nll = 0.0
    total_toks = 0
    for i in range(chunks.shape[0]):
        inp = chunks[i : i + 1].to(device)
        logits_bf16 = model(inp, use_cache=False).logits[:, :-1, :]
        targets = inp[:, 1:]
        seq_t = logits_bf16.shape[1]
        lp_slices: list[torch.Tensor] = []
        for s0 in range(0, seq_t, LOGIT_TOK_CHUNK):
            s1 = min(s0 + LOGIT_TOK_CHUNK, seq_t)
            lp = F.log_softmax(logits_bf16[:, s0:s1, :].to(torch.float32), dim=-1)
            nll = F.nll_loss(lp.reshape(-1, lp.shape[-1]), targets[:, s0:s1].reshape(-1), reduction="sum")
            total_nll += float(nll.item())
            total_toks += int(targets[:, s0:s1].numel())
            lp_slices.append(lp.cpu())
            del lp
        ref_log_probs.append(torch.cat(lp_slices, dim=1))
        del inp, logits_bf16, lp_slices
    torch.cuda.empty_cache()
    return ref_log_probs, math.exp(total_nll / total_toks)


@torch.no_grad()
def _score_against_reference(
    model: torch.nn.Module,
    chunks: torch.Tensor,
    ref_log_probs: list[torch.Tensor],
    device: torch.device,
) -> dict[str, float]:
    kl_list: list[float] = []
    total_nll = 0.0
    total_toks = 0
    same_top1 = 0
    for i in range(chunks.shape[0]):
        inp = chunks[i : i + 1].to(device)
        logits_bf16 = model(inp, use_cache=False).logits[:, :-1, :]
        targets = inp[:, 1:]
        ref_cpu = ref_log_probs[i]
        seq_t = logits_bf16.shape[1]
        for s0 in range(0, seq_t, LOGIT_TOK_CHUNK):
            s1 = min(s0 + LOGIT_TOK_CHUNK, seq_t)
            log_q = F.log_softmax(logits_bf16[:, s0:s1, :].to(torch.float32), dim=-1)
            log_p = ref_cpu[:, s0:s1, :].to(device)
            p_ref = torch.exp(log_p)
            tok_kl = (p_ref * (log_p - log_q)).sum(dim=-1).reshape(-1)
            kl_list.extend(tok_kl.cpu().tolist())
            nll = F.nll_loss(log_q.reshape(-1, log_q.shape[-1]), targets[:, s0:s1].reshape(-1), reduction="sum")
            total_nll += float(nll.item())
            total_toks += int(targets[:, s0:s1].numel())
            same_top1 += int((log_q.argmax(dim=-1) == log_p.argmax(dim=-1)).sum().item())
            del log_q, log_p, p_ref, tok_kl
        del inp, logits_bf16
    torch.cuda.empty_cache()
    kl_t = torch.tensor(kl_list, dtype=torch.float64)
    return {
        "mean_kl": round(float(kl_t.mean().item()), 9),
        "p95_kl": round(float(torch.quantile(kl_t, 0.95).item()), 9),
        "ppl": round(math.exp(total_nll / total_toks), 6),
        "top1_match_pct": round(100.0 * float(same_top1) / float(total_toks), 4),
        "tokens_scored": total_toks,
    }


@torch.no_grad()
def _capture_hessian_diag(
    model: torch.nn.Module,
    linears: list[tuple[str, torch.nn.Linear]],
    chunks: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    sum_sq: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    hooks = []
    for name, mod in linears:
        sum_sq[name] = torch.zeros(mod.in_features, device=device, dtype=torch.float64)
        counts[name] = 0

        def _make_hook(lname: str):
            def _hook(_m, inputs):
                x = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
                sum_sq[lname] += (x.to(torch.float64) ** 2).sum(dim=0)
                counts[lname] += int(x.shape[0])
            return _hook

        hooks.append(mod.register_forward_pre_hook(_make_hook(name)))

    for i in range(chunks.shape[0]):
        model(chunks[i : i + 1].to(device), use_cache=False)

    for hk in hooks:
        hk.remove()

    out = {n: (sum_sq[n] / max(counts[n], 1)).to(torch.float32).cpu() for n in sum_sq}
    del sum_sq
    torch.cuda.empty_cache()
    return out


def _eval_model(
    repo_id: str,
    local_dir: str,
    group_size: int,
    wt_val_text: str,
    wt_test_text: str,
    py_test_text: list[str],
    device: torch.device,
    include_b45: bool = True,
) -> dict[str, Any]:
    print(f"\n=== MODEL: {repo_id} (g={group_size}) ===", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(local_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        local_dir, local_files_only=True, dtype=torch.bfloat16
    ).to(device)
    torch.cuda.reset_peak_memory_stats(device)
    model.eval()

    linears = _collect_block_linears(model)
    n_blocks = max(_layer_idx(n) for n, _ in linears) + 1
    n_heads = int(model.config.num_attention_heads)
    n_kv_heads = int(model.config.num_key_value_heads)
    head_dim = int(model.config.hidden_size) // n_heads
    protected_snaps = _snapshot_protected_non_block_tensors(model)

    orig_weights_cpu = {n: m.weight.detach().cpu().clone() for n, m in linears}
    orig_norms_cpu: dict[str, torch.Tensor] = {}
    orig_vbias_cpu: dict[str, torch.Tensor] = {}
    mod_map = dict(model.named_modules())
    for l_i in range(n_blocks):
        for nk in (f"model.layers.{l_i}.input_layernorm", f"model.layers.{l_i}.post_attention_layernorm"):
            orig_norms_cpu[nk] = mod_map[nk].weight.detach().cpu().clone()
        vm = mod_map[f"model.layers.{l_i}.self_attn.v_proj"]
        if vm.bias is not None:
            orig_vbias_cpu[f"model.layers.{l_i}.self_attn.v_proj.bias"] = vm.bias.detach().cpu().clone()

    total_block_weights = sum(int(m.weight.numel()) for _, m in linears)

    wt_eval_chunks = _tokenize_chunks(tokenizer, wt_test_text, N_EVAL_CHUNKS, SEQ_LEN)
    py_eval_chunks = _tokenize_balanced_python_chunks(tokenizer, py_test_text, N_EVAL_CHUNKS, SEQ_LEN)
    wt_cal_chunks = _tokenize_chunks(tokenizer, wt_val_text, N_REAL_CAL_CHUNKS, SEQ_LEN)

    ref_wt_lp, ppl_base_wt = _collect_reference_logits(model, wt_eval_chunks, device)
    ref_py_lp, ppl_base_py = _collect_reference_logits(model, py_eval_chunks, device)
    print(
        f"  BASE BF16 PPL: Wiki={ppl_base_wt:.6f} Py={ppl_base_py:.6f} "
        f"({n_blocks}L, {len(linears)} mats, {total_block_weights:,} wts)",
        flush=True,
    )

    h_cal_real = _capture_hessian_diag(model, linears, wt_cal_chunks, device)

    # Measure raw column anisotropy of halved difference bodies S_j = (max - min)/2
    aniso_ratios: list[float] = []
    asym_ratios: list[float] = []
    for n, _ in linears:
        wf = orig_weights_cpu[n].float()
        cmax = wf.amax(dim=0)
        cmin = wf.amin(dim=0)
        half_w = 0.5 * (cmax - cmin).clamp_min(1e-8)
        aniso_ratios.append(float((half_w.max() / half_w.median().clamp_min(1e-8)).item()))
        # Group-level asymmetry ratio |M+m|/(M-m)
        wg = wf.reshape(-1, group_size)
        gm = wg.amin(dim=1)
        gM = wg.amax(dim=1)
        asym_ratios.append(float(((gM + gm).abs() / (gM - gm).clamp_min(1e-8)).mean().item()))
        del wf, wg
    geom_telemetry = {
        "mean_col_john_halfwidth_max_over_median": round(sum(aniso_ratios) / len(aniso_ratios), 6),
        "max_col_john_halfwidth_max_over_median": round(max(aniso_ratios), 6),
        "mean_group_asymmetry_ratio": round(sum(asym_ratios) / len(asym_ratios), 6),
    }
    print(
        f"  [GEOM] mean John max/med={geom_telemetry['mean_col_john_halfwidth_max_over_median']:.3f} "
        f"max={geom_telemetry['max_col_john_halfwidth_max_over_median']:.3f} "
        f"mean group asym={geom_telemetry['mean_group_asymmetry_ratio']:.4f}",
        flush=True,
    )

    # Precompute strictly +0 B foldable John/Lowner diagonal scales & verify physical fold invariance
    john_pure_scales = _compute_john_scales_for_model(
        orig_weights_cpu, h_cal_real, n_blocks, n_heads, n_kv_heads, head_dim, use_hessian=False
    )
    john_hess_scales = _compute_john_scales_for_model(
        orig_weights_cpu, h_cal_real, n_blocks, n_heads, n_kv_heads, head_dim, use_hessian=True
    )
    fold_invariance = _verify_physical_zero_byte_folding_invariance(
        model,
        orig_weights_cpu,
        orig_norms_cpu,
        orig_vbias_cpu,
        john_hess_scales,
        wt_eval_chunks,
        ref_wt_lp,
        n_blocks,
        n_heads,
        n_kv_heads,
        head_dim,
        device,
    )
    print(
        f"  [+0B FOLD CHECK] prequant_kl={fold_invariance['prequant_fold_mean_kl']:.9f} "
        f"top1={fold_invariance['prequant_fold_top1_match_pct']:.4f}% extra_B=0",
        flush=True,
    )

    # Precompute Track S73 SharedNorm permutations (QuadSnake & QuadLPT p=0.5)
    snake_perms = {n: _make_quad_snake_perm(h_cal_real[n], group_size, block_w=4) for n, _ in linears}
    lpt_perms = {n: _make_quad_lpt_perm(h_cal_real[n], group_size, block_w=4, load_power=0.5) for n, _ in linears}

    def _make_sharednorm_perm(pdict: dict[str, torch.Tensor]):
        def _fn(n: str) -> torch.Tensor | None:
            kd = _proj_kind(n)
            l_i = _layer_idx(n)
            if kd == "down_proj":
                return pdict[n]
            if kd in ("q_proj", "k_proj", "v_proj"):
                return pdict[f"model.layers.{l_i}.self_attn.q_proj"]
            if kd in ("gate_proj", "up_proj"):
                return pdict[f"model.layers.{l_i}.mlp.gate_proj"]
            return None
        return _fn

    H8_MAT = _sylvester_hadamard(8, device, torch.float32)
    BH4_16 = _block_hadamard(16, 4, device, torch.float32)
    RAND_ORTH_16 = _seeded_random_orth(16, EVAL_SEED + 16, device, torch.float32)

    arms_out: dict[str, Any] = {}

    def _run_arm(
        arm_name: str,
        bpw_key: str,
        lattice: str,
        q_rot: torch.Tensor | None,
        scale_mode: str = "hess_s73",
        perm_fn: Any = None,
        cheb_minimax_scale: bool = False,
        cheb_center: bool = False,
        polar_mode: str = "none",
        rtn_mode: bool = False,
        effective_bpw: float | None = None,
        extra_runtime_bytes: int = 0,
    ) -> dict[str, Any]:
        tot_err = tot_w = tot_hmse = tot_hw = sm_err = sm_w = 0.0
        syn_t = oob_t = fb_t = 0
        max_abs_err = 0.0
        max_w_abs_err = 0.0
        p999_per_mat: list[float] = []
        wp999_per_mat: list[float] = []
        layer_hmse = [0.0] * n_blocks
        layer_hw = [0.0] * n_blocks
        csr_kept_entries = 0
        polar_hi_groups = 0
        total_groups = 0
        total_mat_rows = 0

        base_bound = BALL_BOUNDS[bpw_key][lattice]
        if cheb_center:
            _require(lattice == "A16", "cheb_center matched bound is defined for A16")
            use_bound = A16_BULK_BOUND_CHEB_CENTER[bpw_key]
        elif polar_mode in ("entry_csr_025", "block_polar_1over19"):
            _require(lattice == "A16", "polar bulk matched bound is defined for A16")
            use_bound = A16_BULK_BOUND_CSR[bpw_key]
        else:
            use_bound = base_bound

        with torch.no_grad():
            for n, m in linears:
                wf = orig_weights_cpu[n].to(device=device, dtype=torch.float32)
                hv = h_cal_real[n].to(device=device, dtype=torch.float32).clamp_min(1e-12)
                h_norm = hv / hv.mean()
                h_sqrt = torch.sqrt(h_norm)
                l_i = _layer_idx(n)

                # Column scale selection
                if scale_mode == "none":
                    cs = None
                elif scale_mode == "hess_s73":
                    cs = h_norm.clamp_min(1e-8) ** 0.25
                elif scale_mode == "john_pure_folded0b":
                    cs = john_pure_scales[n].to(device=device, dtype=torch.float32)
                elif scale_mode == "john_hess_folded0b":
                    cs = john_hess_scales[n].to(device=device, dtype=torch.float32)
                elif scale_mode == "john_hess_permat":
                    s_col = 0.5 * (wf.amax(dim=0) - wf.amin(dim=0)).clamp_min(1e-6)
                    cs = _geom_norm_clamp(h_norm.clamp_min(1e-8).pow(0.25) / s_col.pow(0.5)).to(device)
                else:
                    raise RuntimeError(f"Unknown scale_mode: {scale_mode}")

                cperm = perm_fn(n) if callable(perm_fn) else None
                if cperm is not None:
                    cperm = cperm.to(device)

                n_mat_groups = wf.numel() // group_size
                total_groups += n_mat_groups
                total_mat_rows += int(wf.shape[0])

                # Optional Polar-Cap / Equatorial-Bulk split (Arm B3)
                wf_dense = wf
                cap_mask: torch.Tensor | None = None
                sparse_rec: torch.Tensor | None = None
                group_bounds_full: torch.Tensor | None = None

                if polar_mode == "entry_csr_025":
                    # Top 0.25% polar-cap sensitivity outliers excised into exact FP16+16b CSR
                    # Polar projection weight combines |W_ij| * sqrt(h_j) with leading polar direction alignment
                    w_polar = _leading_polar_direction(wf, h_sqrt, n_iters=8)
                    # Polar-weighted sensitivity score: |W_ij| * sqrt(h_j) * (1 + |w*_j| * sqrt(n))
                    n_cols = wf.shape[1]
                    polar_boost = 1.0 + 0.5 * w_polar.abs() * math.sqrt(float(n_cols))
                    score = (wf.abs() * h_sqrt.unsqueeze(0) * polar_boost.unsqueeze(0)).reshape(-1)
                    k_cap = max(1, int(wf.numel() * 0.0025))
                    # Exact top-k threshold via kthvalue
                    thresh = torch.kthvalue(score, score.numel() - k_cap + 1).values
                    cap_mask = (score >= thresh).reshape(wf.shape)
                    # Trim ties to exact <= k_cap for strict CS-08 bit budget
                    if int(cap_mask.sum().item()) > k_cap:
                        flat_idx = torch.nonzero(cap_mask.reshape(-1), as_tuple=False).squeeze(1)
                        vals_f = score[flat_idx]
                        _, top_sub = torch.topk(vals_f, k_cap)
                        cap_mask = torch.zeros_like(cap_mask.reshape(-1))
                        cap_mask[flat_idx[top_sub]] = True
                        cap_mask = cap_mask.reshape(wf.shape)
                    csr_kept_entries += int(cap_mask.sum().item())
                    sparse_rec = torch.where(cap_mask, wf.to(torch.float16).to(torch.float32), torch.zeros_like(wf))
                    wf_dense = torch.where(cap_mask, torch.zeros_like(wf), wf)
                    del score, w_polar, polar_boost
                elif polar_mode == "block_polar_1over19":
                    # Exact 1/19 (5.263%) polar-cap outlier blocks at 4.5 bpw (R^2=5213, 576 bits)
                    # and 18/19 equatorial-bulk blocks at B_dense=357 (R^2=577, 373 bits):
                    # payload+scale = (576 + 18*373)/19 = 383.684 <= 384 bits (2.998 bpw);
                    # with explicit 1-bit/group selector map = 384.684 bits = 3.0053 bpw!
                    _require(bpw_key == "3.0" and lattice == "A16", "block_polar_1over19 is defined for 3.0 bpw A16")
                    w_polar = _leading_polar_direction(wf, h_sqrt, n_iters=8)
                    wf_p = wf if cperm is None else wf[:, cperm]
                    hs_p = h_sqrt if cperm is None else h_sqrt[cperm]
                    wp_p = w_polar if cperm is None else w_polar[cperm]
                    wg = (wf_p * hs_p.unsqueeze(0)).reshape(-1, group_size)
                    wp_g = wp_p.reshape(1, -1, group_size).expand(wf_p.shape[0], -1, -1).reshape(-1, group_size)
                    # Score each block by polar alignment energy + block L2 sensitivity energy
                    polar_proj = (wg * wp_g).sum(dim=1).abs()
                    blk_e = (wg ** 2).sum(dim=1)
                    blk_score = blk_e * (1.0 + polar_proj / (torch.sqrt(blk_e).clamp_min(1e-12) * wp_g.norm(dim=1).clamp_min(1e-12)))
                    k_hi = n_mat_groups // 19
                    group_bounds_full = torch.full((n_mat_groups,), 577.0, device=device, dtype=torch.float32)
                    if k_hi > 0:
                        _, hi_idx = torch.topk(blk_score, k_hi)
                        group_bounds_full[hi_idx] = 5213.0
                        polar_hi_groups += k_hi
                    del w_polar, wg, wp_g, polar_proj, blk_e, blk_score

                wq, errs, oob, fb_c = _quantize_matrix(
                    wf_dense,
                    use_bound,
                    lattice,
                    group_size,
                    q_rot=q_rot,
                    colscale=cs,
                    col_perm=cperm,
                    cheb_minimax_scale=cheb_minimax_scale,
                    cheb_center=cheb_center,
                    rtn_bpw_key=bpw_key if rtn_mode else None,
                    group_bounds_full=group_bounds_full,
                )
                if sparse_rec is not None and cap_mask is not None:
                    wq = torch.where(cap_mask, sparse_rec, wq)
                    del sparse_rec, cap_mask

                dw = wq - wf
                w_sq = float((wf ** 2).sum().item())
                cur_mse = float((dw ** 2).sum().item())
                cur_hmse = float(((dw ** 2) * h_norm.unsqueeze(0)).sum().item())
                cur_hw = float(((wf ** 2) * h_norm.unsqueeze(0)).sum().item())

                abs_dw = dw.abs().reshape(-1)
                w_abs_dw = (dw.abs() * h_sqrt.unsqueeze(0)).reshape(-1)
                mat_max = float(abs_dw.max().item())
                mat_wmax = float(w_abs_dw.max().item())
                if mat_max > max_abs_err:
                    max_abs_err = mat_max
                if mat_wmax > max_w_abs_err:
                    max_w_abs_err = mat_wmax
                k999 = max(1, int(round(0.999 * abs_dw.numel())))
                p999_per_mat.append(float(torch.kthvalue(abs_dw, k999).values.item()))
                wp999_per_mat.append(float(torch.kthvalue(w_abs_dw, k999).values.item()))

                rms = math.sqrt(w_sq / float(wf.numel()) + 1e-30)
                sm_mask = wf.abs() < (0.05 * rms)
                sc = int(sm_mask.sum().item())
                tot_err += cur_mse
                tot_w += w_sq
                tot_hmse += cur_hmse
                tot_hw += cur_hw
                layer_hmse[l_i] += cur_hmse
                layer_hw[l_i] += cur_hw
                sm_err += float((dw[sm_mask] ** 2).sum().item()) if sc > 0 else 0.0
                sm_w += float((wf[sm_mask] ** 2).sum().item()) if sc > 0 else 0.0
                syn_t += errs
                oob_t += oob
                fb_t += fb_c
                m.weight.copy_(wq.to(torch.bfloat16))
                del wf, wf_dense, wq, dw, abs_dw, w_abs_dw, sm_mask, hv, h_norm, h_sqrt, cs, cperm, group_bounds_full

        nb_diffs = _count_protected_diffs(model, protected_snaps)
        _require(nb_diffs == 0, f"Protected non-block tensors modified in arm {arm_name}!")
        wt_scores = _score_against_reference(model, wt_eval_chunks, ref_wt_lp, device)
        py_scores = _score_against_reference(model, py_eval_chunks, ref_py_lp, device)

        w_err_db = 10.0 * math.log10(max(tot_err, 1e-30) / max(tot_w, 1e-30))
        h_mse_rel = tot_hmse / max(tot_hw, 1e-30)
        h_err_db = 10.0 * math.log10(max(h_mse_rel, 1e-30))
        e05_db = 10.0 * math.log10(max(sm_err, 1e-30) / max(sm_w, 1e-30))
        mean_p999 = sum(p999_per_mat) / len(p999_per_mat)
        mean_wp999 = sum(wp999_per_mat) / len(wp999_per_mat)

        eff_bpw_meta: float | None = None
        if effective_bpw is None:
            if rtn_mode:
                eff_bpw = 2.932355 if bpw_key == "3.0" else 4.372928
                eff_bpw_meta = eff_bpw
            elif polar_mode == "entry_csr_025":
                base_bits = 357.0 if bpw_key == "3.0" else 549.0
                eff_bpw = (base_bits + 16.0 + 32.0 * (csr_kept_entries / total_block_weights) * 128.0) / 128.0
                # Include 16-bit CSR row_ptr per matrix row
                eff_bpw_meta = eff_bpw + (16.0 * total_mat_rows) / total_block_weights
            elif polar_mode == "block_polar_1over19":
                frac_hi = polar_hi_groups / max(total_groups, 1)
                eff_bpw = (frac_hi * 576.0 + (1.0 - frac_hi) * 373.0) / 128.0
                # Include explicit 1-bit/group selector bitmap
                eff_bpw_meta = eff_bpw + 1.0 / 128.0
            else:
                eff_bpw = float(bpw_key)
                eff_bpw_meta = eff_bpw
        else:
            eff_bpw = effective_bpw
            eff_bpw_meta = eff_bpw

        if extra_runtime_bytes == 0 and scale_mode in ("hess_s73", "john_hess_permat"):
            # Unpooled o_proj (for hess_s73) or all per-matrix column scales (for john_hess_permat)
            # cannot fold into upstream projections at +0 B without GQA pooling / shared norm.
            extra_bytes_actual = sum(
                int(m.in_features) * 2
                for n, m in linears
                if scale_mode == "john_hess_permat" or _proj_kind(n) == "o_proj"
            )
        else:
            extra_bytes_actual = extra_runtime_bytes

        row = {
            "arm": arm_name,
            "bpw_target": bpw_key,
            "effective_bpw": round(eff_bpw, 6),
            "effective_bpw_with_metadata": round(eff_bpw_meta, 6),
            "extra_runtime_bytes": extra_bytes_actual,
            "fallback_rescaled_groups": fb_t,
            "lattice": "RTN" if rtn_mode else lattice,
            "group_size": group_size,
            "ball_bound": 0 if rtn_mode else int(use_bound),
            "weight_err_db": round(w_err_db, 6),
            "weighted_mse_rel": round(h_mse_rel, 9),
            "weighted_mse_db": round(h_err_db, 6),
            "small_weight_e05_db": round(e05_db, 6),
            "mean_p999_abs_err": round(mean_p999, 9),
            "mean_p999_weighted_abs_err": round(mean_wp999, 9),
            "max_abs_err": round(max_abs_err, 9),
            "max_weighted_abs_err": round(max_w_abs_err, 9),
            "syndrome_errors": syn_t,
            "out_of_ball_groups": oob_t,
            "protected_non_block_diffs": nb_diffs,
            "csr_entries": csr_kept_entries,
            "polar_hi_groups": polar_hi_groups,
            "layer_weighted_mse_rel": [round(layer_hmse[i] / max(layer_hw[i], 1e-30), 9) for i in range(n_blocks)],
            "wikitext2": wt_scores,
            "python": py_scores,
        }
        print(
            f"  [{arm_name:44s}] {eff_bpw:5.3f}b hMSE={h_mse_rel:.6f} ({h_err_db:7.3f}dB) "
            f"p999={mean_p999:.5f} wp999={mean_wp999:.5f} WikiPPL={wt_scores['ppl']:7.4f} "
            f"WikiKL={wt_scores['mean_kl']:.6f} PyPPL={py_scores['ppl']:6.4f} syn={syn_t} oob={oob_t}",
            flush=True,
        )
        return row

    sn_snake = _make_sharednorm_perm(snake_perms)
    sn_lpt = _make_sharednorm_perm(lpt_perms)

    # =========================================================================
    # 3.0 BPW ARMS (Baselines A0-A5, John B1a-B1d, Helly B2a-B2b, Polar B3a-B3c)
    # =========================================================================
    arms_out["A0_b30_RTN_unwhitened"] = _run_arm("A0_b30_RTN_unwhitened", "3.0", "A16", None, scale_mode="none", rtn_mode=True)
    arms_out["A1_b30_E8_H8_hess"] = _run_arm("A1_b30_E8_H8_hess", "3.0", "E8", H8_MAT, scale_mode="hess_s73")
    arms_out["A2_b30_A16_BlockH4_hess"] = _run_arm("A2_b30_A16_BlockH4_hess", "3.0", "A16", BH4_16, scale_mode="hess_s73")
    arms_out["A3_b30_A16_BlockH4_QuadSnake_SN"] = _run_arm(
        "A3_b30_A16_BlockH4_QuadSnake_SN", "3.0", "A16", BH4_16, scale_mode="hess_s73", perm_fn=sn_snake
    )
    arms_out["A4_b30_A16_BlockH4_QuadLPT_SN"] = _run_arm(
        "A4_b30_A16_BlockH4_QuadLPT_SN", "3.0", "A16", BH4_16, scale_mode="hess_s73", perm_fn=sn_lpt
    )
    arms_out["A5_b30_A16_RandOrth_hess"] = _run_arm(
        "A5_b30_A16_RandOrth_hess", "3.0", "A16", RAND_ORTH_16, scale_mode="hess_s73"
    )

    # Arm B1: John/Lowner halved-difference-body diagonal whitening (+0 B folded!)
    arms_out["B1a_b30_JohnS_ZeroCal_BlockH4_Folded0B"] = _run_arm(
        "B1a_b30_JohnS_ZeroCal_BlockH4_Folded0B", "3.0", "A16", BH4_16, scale_mode="john_pure_folded0b", extra_runtime_bytes=0
    )
    arms_out["B1b_b30_JohnHess_BlockH4_Folded0B"] = _run_arm(
        "B1b_b30_JohnHess_BlockH4_Folded0B", "3.0", "A16", BH4_16, scale_mode="john_hess_folded0b", extra_runtime_bytes=0
    )
    arms_out["B1c_b30_JohnHess_BlockH4_QuadSnake_Folded0B"] = _run_arm(
        "B1c_b30_JohnHess_BlockH4_QuadSnake_Folded0B",
        "3.0",
        "A16",
        BH4_16,
        scale_mode="john_hess_folded0b",
        perm_fn=sn_snake,
        extra_runtime_bytes=0,
    )
    arms_out["B1d_b30_JohnHess_E8_H8_Folded0B"] = _run_arm(
        "B1d_b30_JohnHess_E8_H8_Folded0B", "3.0", "E8", H8_MAT, scale_mode="john_hess_folded0b", extra_runtime_bytes=0
    )
    arms_out["B1e_b30_JohnHess_BlockH4_PerMat"] = _run_arm(
        "B1e_b30_JohnHess_BlockH4_PerMat", "3.0", "A16", BH4_16, scale_mode="john_hess_permat"
    )

    # Arm B2: Exact Helly-LP Minimax Chebyshev Gauge Scales (0 B minimax & matched-bpw center)
    arms_out["B2a_b30_HellyCheb_MinimaxScale_QuadLPT_0B"] = _run_arm(
        "B2a_b30_HellyCheb_MinimaxScale_QuadLPT_0B",
        "3.0",
        "A16",
        BH4_16,
        scale_mode="hess_s73",
        perm_fn=sn_lpt,
        cheb_minimax_scale=True,
        extra_runtime_bytes=0,
    )
    arms_out["B2b_b30_HellyCheb_CenterScale_QuadLPT_Matched"] = _run_arm(
        "B2b_b30_HellyCheb_CenterScale_QuadLPT_Matched",
        "3.0",
        "A16",
        BH4_16,
        scale_mode="hess_s73",
        perm_fn=sn_lpt,
        cheb_minimax_scale=True,
        cheb_center=True,
        effective_bpw=3.0,
    )

    # Arm B3: Optimal-Projection Polar-Cap vs Equatorial-Bulk Split (exact matched <= 3.000 bpw)
    arms_out["B3a_b30_PolarCap_CSR025_BlockH4_Matched"] = _run_arm(
        "B3a_b30_PolarCap_CSR025_BlockH4_Matched",
        "3.0",
        "A16",
        BH4_16,
        scale_mode="hess_s73",
        perm_fn=sn_snake,
        polar_mode="entry_csr_025",
    )
    arms_out["B3b_b30_PolarCap_BlockSplit_1over19_Matched"] = _run_arm(
        "B3b_b30_PolarCap_BlockSplit_1over19_Matched",
        "3.0",
        "A16",
        BH4_16,
        scale_mode="hess_s73",
        perm_fn=sn_snake,
        polar_mode="block_polar_1over19",
    )
    arms_out["B3c_b30_JohnHess0B_Plus_PolarCap_CSR025"] = _run_arm(
        "B3c_b30_JohnHess0B_Plus_PolarCap_CSR025",
        "3.0",
        "A16",
        BH4_16,
        scale_mode="john_hess_folded0b",
        perm_fn=sn_snake,
        polar_mode="entry_csr_025",
    )

    # =========================================================================
    # 4.5 BPW ARMS (Matched Frontier Comparison)
    # =========================================================================
    if include_b45:
        arms_out["A1_b45_E8_H8_hess"] = _run_arm("A1_b45_E8_H8_hess", "4.5", "E8", H8_MAT, scale_mode="hess_s73")
        arms_out["A2_b45_A16_BlockH4_hess"] = _run_arm("A2_b45_A16_BlockH4_hess", "4.5", "A16", BH4_16, scale_mode="hess_s73")
        arms_out["A3_b45_A16_BlockH4_QuadSnake_SN"] = _run_arm(
            "A3_b45_A16_BlockH4_QuadSnake_SN", "4.5", "A16", BH4_16, scale_mode="hess_s73", perm_fn=sn_snake
        )
        arms_out["A4_b45_A16_BlockH4_QuadLPT_SN"] = _run_arm(
            "A4_b45_A16_BlockH4_QuadLPT_SN", "4.5", "A16", BH4_16, scale_mode="hess_s73", perm_fn=sn_lpt
        )
        arms_out["B1b_b45_JohnHess_BlockH4_Folded0B"] = _run_arm(
            "B1b_b45_JohnHess_BlockH4_Folded0B", "4.5", "A16", BH4_16, scale_mode="john_hess_folded0b", extra_runtime_bytes=0
        )
        arms_out["B1c_b45_JohnHess_BlockH4_QuadSnake_Folded0B"] = _run_arm(
            "B1c_b45_JohnHess_BlockH4_QuadSnake_Folded0B",
            "4.5",
            "A16",
            BH4_16,
            scale_mode="john_hess_folded0b",
            perm_fn=sn_snake,
            extra_runtime_bytes=0,
        )
        arms_out["B1d_b45_JohnHess_E8_H8_Folded0B"] = _run_arm(
            "B1d_b45_JohnHess_E8_H8_Folded0B", "4.5", "E8", H8_MAT, scale_mode="john_hess_folded0b", extra_runtime_bytes=0
        )
        arms_out["B2a_b45_HellyCheb_MinimaxScale_QuadLPT_0B"] = _run_arm(
            "B2a_b45_HellyCheb_MinimaxScale_QuadLPT_0B",
            "4.5",
            "A16",
            BH4_16,
            scale_mode="hess_s73",
            perm_fn=sn_lpt,
            cheb_minimax_scale=True,
            extra_runtime_bytes=0,
        )
        arms_out["B3a_b45_PolarCap_CSR025_BlockH4_Matched"] = _run_arm(
            "B3a_b45_PolarCap_CSR025_BlockH4_Matched",
            "4.5",
            "A16",
            BH4_16,
            scale_mode="hess_s73",
            perm_fn=sn_snake,
            polar_mode="entry_csr_025",
        )
        arms_out["B3c_b45_JohnHess0B_Plus_PolarCap_CSR025"] = _run_arm(
            "B3c_b45_JohnHess0B_Plus_PolarCap_CSR025",
            "4.5",
            "A16",
            BH4_16,
            scale_mode="john_hess_folded0b",
            perm_fn=sn_snake,
            polar_mode="entry_csr_025",
        )

    peak_vram_mib = round(float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0), 2)
    print(f"  [PEAK VRAM] {peak_vram_mib:.2f} MiB", flush=True)

    out = {
        "repo_id": repo_id,
        "group_size": group_size,
        "num_hidden_layers": n_blocks,
        "block_linear_matrices": len(linears),
        "total_block_weights": total_block_weights,
        "protected_non_block_tensors": len(protected_snaps),
        "peak_vram_mib": peak_vram_mib,
        "geometry_telemetry": geom_telemetry,
        "physical_zero_byte_fold_check": fold_invariance,
        "reference_bf16": {
            "wikitext2_ppl": round(ppl_base_wt, 6),
            "python_ppl": round(ppl_base_py, 6),
        },
        "arms": arms_out,
    }
    del model, ref_wt_lp, ref_py_lp, orig_weights_cpu, orig_norms_cpu, orig_vbias_cpu
    del h_cal_real, john_pure_scales, john_hess_scales, snake_perms, lpt_perms
    gc.collect()
    torch.cuda.empty_cache()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Track S165 EXP-Q47b Real-Weight RTX 3080 Benchmark")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "s165_empirical_results.json",
    )
    parser.add_argument("--skip-15b", action="store_true")
    args = parser.parse_args()

    _require(torch.cuda.is_available(), "CUDA GPU is required for live RTX 3080 evaluation")
    torch.manual_seed(EVAL_SEED)
    device = torch.device("cuda:0")

    prereg_path = Path(__file__).resolve().parent.parent / "verification" / "prereg" / "PREREG_W22_HADWIGER_SILIQUANT_JOHN.json"
    prereg_raw = prereg_path.read_bytes()
    prereg_sha256 = hashlib.sha256(prereg_raw).hexdigest()

    hf_cache = Path.home() / ".cache" / "huggingface"
    s49b_cache = hf_cache / "s49b_eval"
    s60_cache = hf_cache / "s60_eval"
    s49b_pins = json.loads((s49b_cache / "pins.json").read_text(encoding="utf-8"))
    q15_pin = json.loads((s60_cache / "qwen15b_pin.json").read_text(encoding="utf-8"))

    wt_val_text = (s49b_cache / "wikitext2_val.txt").read_text(encoding="utf-8")
    wt_test_text = (s49b_cache / "wikitext2_test.txt").read_text(encoding="utf-8")
    py_test_text, py_sha256 = _load_python_eval_text()

    t_start = time.monotonic()
    gpu_pre = _query_gpu_memory_info()

    models_out: dict[str, Any] = {}
    models_out["Qwen/Qwen2.5-0.5B"] = _eval_model(
        "Qwen/Qwen2.5-0.5B",
        s49b_pins["Qwen/Qwen2.5-0.5B"]["local_dir"],
        128,
        wt_val_text,
        wt_test_text,
        py_test_text,
        device=device,
        include_b45=True,
    )
    if not args.skip_15b:
        models_out["Qwen/Qwen2.5-1.5B"] = _eval_model(
            "Qwen/Qwen2.5-1.5B",
            q15_pin["local_dir"],
            128,
            wt_val_text,
            wt_test_text,
            py_test_text,
            device=device,
            include_b45=True,
        )

    gc.collect()
    torch.cuda.empty_cache()
    gpu_post = _query_gpu_memory_info()
    elapsed = round(time.monotonic() - t_start, 3)

    payload = {
        "schema": "aletheia.track_s165.hadwiger_john_siliquant_empirical.v1",
        "track": "S165",
        "experiment_id": "EXP-Q47b",
        "prereg_id": "PREREG-W22-HADWIGER-SILIQUANT",
        "prereg_sha256": prereg_sha256,
        "oracle_class": "sampled",
        "measured_live_model_weights": True,
        "energy_claim": "UNCONTROLLED_LAPTOP_NOT_CLAIMED",
        "eval_seed": EVAL_SEED,
        "ball_bounds": {
            "3.0": {"A16": 651, "E8": 420, "A16_csr_bulk": 577, "A16_cheb_center_bulk": 547},
            "4.5": {"A16": 5213, "E8": 3380, "A16_csr_bulk": 4629, "A16_cheb_center_bulk": 4385},
        },
        "corpora": {
            "wikitext2_repo": s49b_pins["wikitext2"]["repo_id"],
            "wikitext2_revision": s49b_pins["wikitext2"]["revision"],
            "wikitext2_val_sha256": s49b_pins["wikitext2"]["val_sha256"],
            "wikitext2_test_sha256": s49b_pins["wikitext2"]["test_sha256"],
            "python_stdlib_modules": list(STDLIB_PY_MODULES),
            "python_stdlib_sha256": py_sha256,
            "seq_len": SEQ_LEN,
            "n_eval_chunks": N_EVAL_CHUNKS,
            "n_real_cal_chunks": N_REAL_CAL_CHUNKS,
        },
        "model_pins": {
            "Qwen/Qwen2.5-0.5B": {
                "revision": s49b_pins["Qwen/Qwen2.5-0.5B"]["revision"],
                "safetensors_sha256": s49b_pins["Qwen/Qwen2.5-0.5B"]["model_safetensors_sha256"],
                "tokenizer_sha256": s49b_pins["Qwen/Qwen2.5-0.5B"]["tokenizer_json_sha256"],
            },
            "Qwen/Qwen2.5-1.5B": {
                "revision": q15_pin["revision"],
                "safetensors_sha256": q15_pin["safetensors"][0]["sha256"],
                "tokenizer_sha256": q15_pin["tokenizer_json_sha256"],
            },
        },
        "gpu_pre": gpu_pre,
        "gpu_post": gpu_post,
        "elapsed_seconds": elapsed,
        "models": models_out,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nWROTE {args.output.name} ({args.output.stat().st_size} bytes, elapsed={elapsed}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
