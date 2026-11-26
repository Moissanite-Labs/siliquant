"""Track S73 CEER Cycle-6 Live RTX 3080 Evaluator:
PTX-Ready Shared-RMSNorm quad_snake, Exact Load-Power p-Curve, & 3.0 bpw Low-Bitrate Frontier.

Evaluates 96 live quantized full-model arms across all 4 cached real-weight LLMs
(SmolLM2-360M g=32, Qwen2.5-0.5B/1.5B/3B g=128; 120 transformer blocks, 840 linear matrices,
4,757,127,168 block weights, 0 syndrome errors, 0 out-of-ball groups, 0 non-block diffs):
  - P_S73_1 (4.5 bpw): PTX-ready shared-RMSNorm quad_snake_sharednorm & quad_lpt_sqrt_sharednorm
    plus zero-runtime-cost quad_snake_down_zerocost vs per-matrix hA16_BlockH2_real & BlockH4.
  - P_S73_2 (4.5 bpw, Qwen2.5-3B): Exact 7-point load-power exponent phase curve
    p in {0.0 (snake), 0.125, 0.25, 0.375, 0.50, 0.75, 1.00} + sharednorm at p=0.25 +
    4-model exact ball-starvation telemetry.
  - P_S73_3 (3.0 bpw exact siliquant#85 codebooks: g=128 B=368 A16=651/E8=420; g=32 B=88
    A16=151/E8=96): 15-16 live arms per model across uncalibrated/zero-text & calibrated regimes,
    directly measuring the 3.0 bpw vs 4.5 bpw distortion-amplification law.
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

EVAL_SEED = 2026112473
SEQ_LEN = 512
N_EVAL_CHUNKS = 8
N_RAND_CHUNKS = 8
N_REAL_CAL_CHUNKS = 12
QUANT_ROW_CHUNK = 512
LOGIT_TOK_CHUNK = 128

# Exact integer theta-series ball bounds matching Moissanite-Labs/siliquant#85
# g=128 uses 16-bit scale: B=560 -> 4.500 bpw, B=368 -> 3.000 bpw
# g=32  uses  8-bit scale: B=136 -> 4.500 bpw, B=88  -> 3.000 bpw
BALL_BOUNDS: dict[str, dict[int, dict[str, float]]] = {
    "4.5": {
        128: {"A16": 5213.0, "E8": 3380.0},
        32: {"A16": 1207.0, "E8": 782.0},
    },
    "3.0": {
        128: {"A16": 651.0, "E8": 420.0},
        32: {"A16": 151.0, "E8": 96.0},
    },
}

LOAD_POWER_GRID = (
    ("p0000_snake", 0.0),
    ("p0125", 0.125),
    ("p0250", 0.25),
    ("p0375", 0.375),
    ("p0500_sqrt", 0.5),
    ("p0750", 0.75),
    ("p1000_lin", 1.0),
)

STDLIB_PY_MODULES = (
    "fractions.py",
    "difflib.py",
    "decimal.py",
    "statistics.py",
    "argparse.py",
    "json/decoder.py",
)


def _read_rapl_pkg_uj() -> int:
    try:
        return int(Path("/sys/class/powercap/intel-rapl:0/energy_uj").read_text(encoding="utf-8").strip())
    except OSError:
        return 0


def _read_rapl_max_range_uj() -> int:
    try:
        return int(Path("/sys/class/powercap/intel-rapl:0/max_energy_range_uj").read_text(encoding="utf-8").strip())
    except OSError:
        return 0


def _rapl_delta_joules(start_uj: int, end_uj: int, max_range_uj: int) -> float:
    if end_uj >= start_uj:
        diff = end_uj - start_uj
    elif max_range_uj > 0:
        diff = (max_range_uj - start_uj) + end_uj
    else:
        diff = 0
    return round(diff / 1e6, 3)


def _query_nvidia_smi() -> dict[str, Any]:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,pstate,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    parts = [p.strip() for p in out.split(",")]
    return {
        "name": parts[0],
        "driver_version": parts[1],
        "pstate": parts[2],
        "temperature_c": int(parts[3]),
        "power_draw_w": float(parts[4]),
        "clocks_sm_mhz": int(parts[5]),
        "clocks_mem_mhz": int(parts[6]),
        "memory_used_mib": int(parts[7]),
        "memory_total_mib": int(parts[8]),
    }


def _load_python_eval_text() -> tuple[str, str]:
    stdlib_dir = Path(json.__file__).resolve().parent.parent
    chunks: list[str] = []
    h = hashlib.sha256()
    for rel in STDLIB_PY_MODULES:
        raw = (stdlib_dir / rel).read_bytes()
        h.update(rel.encode("utf-8") + b"\0" + raw)
        chunks.append(raw.decode("utf-8", errors="ignore"))
    return "\n\n".join(chunks), h.hexdigest()


def _tokenize_chunks(tokenizer: Any, text: str, n_chunks: int, seq_len: int) -> torch.Tensor:
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    need = n_chunks * seq_len
    if ids.numel() < need:
        reps = (need // ids.numel()) + 1
        ids = ids.repeat(reps)
    return ids[:need].reshape(n_chunks, seq_len).contiguous()


def _sylvester_hadamard(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    h = torch.tensor([[1.0]], device=device, dtype=dtype)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    return h / math.sqrt(float(n))


def _block_hadamard(d: int, b: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    hb = _sylvester_hadamard(b, device, dtype)
    return torch.block_diag(*([hb] * (d // b)))


def _apply_block_rot(w: torch.Tensor, q_rot: torch.Tensor) -> torch.Tensor:
    d = q_rot.shape[0]
    return (w.reshape(w.shape[0], -1, d) @ q_rot.T).reshape(w.shape)


def _a16_closest(u: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Exact nearest lattice point in A_16 = 2 Z^16 + C_{[16,11,4]} with live syndrome check."""
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
    out_b = bits.clone()

    odd_m = p0 == 1
    if odd_m.any():
        ro = torch.nonzero(odd_m, as_tuple=False).squeeze(1)
        co = syn[ro].to(torch.int64)
        out[ro, co] = alt[ro, co]
        out_b[ro, co] ^= 1

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
        out_b[re_i, bj] ^= 1
        out_b[re_i, bp] ^= 1

    vp0 = out_b.sum(dim=1) & 1
    vs0 = out_b[:, 1::2].sum(dim=1) & 1
    vs1 = (out_b[:, 2::4].sum(dim=1) + out_b[:, 3::4].sum(dim=1)) & 1
    vs2 = (out_b[:, 4:8].sum(dim=1) + out_b[:, 12:16].sum(dim=1)) & 1
    vs3 = out_b[:, 8:16].sum(dim=1) & 1
    errs = int(((vp0 | vs0 | vs1 | vs2 | vs3) != 0).sum().item())
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

    # Verify strict E_8 membership: 2*out is integer, all entries same parity, sum(out) is even integer
    doubled = torch.round(out * 2.0).to(torch.int64)
    non_half = ((out * 2.0 - doubled.to(out.dtype)).abs() > 1e-4).any(dim=1)
    par0 = doubled[:, 0:1] & 1
    mixed_par = ((doubled & 1) != par0).any(dim=1)
    bad_sum = (doubled.sum(dim=1) & 3) != 0
    errs = int((non_half | mixed_par | bad_sum).sum().item())
    return out, errs


def _quantize_ball_chunk(
    w_chunk: torch.Tensor,
    bpw_key: str,
    lattice: str,
    group_size: int,
    q_rot: torch.Tensor | None = None,
    colscale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int, int]:
    X = w_chunk.clone()
    if colscale is not None:
        X = X * colscale.unsqueeze(0)
    if q_rot is not None:
        X = _apply_block_rot(X, q_rot)

    Xg = X.reshape(-1, group_size)
    bound = BALL_BOUNDS[bpw_key][group_size][lattice]
    r_ball = math.sqrt(bound)
    blk_dim = 16.0 if lattice == "A16" else 8.0
    margin = math.sqrt(group_size / 4.0 + 1.5 * (group_size / blk_dim))

    g_norm = torch.sqrt((Xg ** 2).sum(dim=1)).clamp_min(1e-30)
    lo = g_norm / (r_ball + margin) * 0.99
    hi = g_norm / max(r_ball - margin, 0.05 * r_ball) * 1.01

    for _ in range(16):
        mid = torch.sqrt(lo * hi)
        scaled = Xg / mid.unsqueeze(1)
        if lattice == "A16":
            Y, _ = _a16_closest(scaled.reshape(-1, 16))
        else:
            Y, _ = _e8_closest(scaled.reshape(-1, 8))
        Y = Y.reshape(-1, group_size)
        ok = (Y ** 2).sum(dim=1) <= bound + 1e-6
        hi = torch.where(ok, mid, hi)
        lo = torch.where(ok, lo, mid)

    scaled_f = Xg / hi.unsqueeze(1)
    if lattice == "A16":
        Y_fin, syn_errs = _a16_closest(scaled_f.reshape(-1, 16))
    else:
        Y_fin, syn_errs = _e8_closest(scaled_f.reshape(-1, 8))
    Y_fin = Y_fin.reshape(-1, group_size)

    out_m = (Y_fin ** 2).sum(dim=1) > bound + 1e-6
    if out_m.any():
        xo = scaled_f[out_m]
        s_lo = torch.zeros(xo.shape[0], device=w_chunk.device, dtype=w_chunk.dtype)
        s_hi = torch.ones(xo.shape[0], device=w_chunk.device, dtype=w_chunk.dtype)
        for _ in range(14):
            sm = 0.5 * (s_lo + s_hi)
            if lattice == "A16":
                yt, _ = _a16_closest((xo * sm.unsqueeze(1)).reshape(-1, 16))
            else:
                yt, _ = _e8_closest((xo * sm.unsqueeze(1)).reshape(-1, 8))
            yt = yt.reshape(-1, group_size)
            good = (yt ** 2).sum(dim=1) <= bound + 1e-6
            s_lo = torch.where(good, sm, s_lo)
            s_hi = torch.where(good, s_hi, sm)
        if lattice == "A16":
            yf2, se2 = _a16_closest((xo * s_lo.unsqueeze(1)).reshape(-1, 16))
        else:
            yf2, se2 = _e8_closest((xo * s_lo.unsqueeze(1)).reshape(-1, 8))
        Y_fin[out_m] = yf2.reshape(-1, group_size)
        syn_errs += se2

    oob_groups = int(((Y_fin ** 2).sum(dim=1) > bound + 1e-6).sum().item())
    scales_out = hi.to(torch.float16).to(torch.float32)
    Wq = (Y_fin * scales_out.unsqueeze(1)).reshape(w_chunk.shape)
    if q_rot is not None:
        Wq = _apply_block_rot(Wq, q_rot.T)
    if colscale is not None:
        Wq = Wq / colscale.unsqueeze(0)
    return Wq, syn_errs, oob_groups


def _quantize_ball_matrix(
    w_f32: torch.Tensor,
    bpw_key: str,
    lattice: str,
    group_size: int,
    q_rot: torch.Tensor | None = None,
    colscale: torch.Tensor | None = None,
    col_perm: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int, int]:
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
    if m <= QUANT_ROW_CHUNK:
        wq, syn_tot, oob_tot = _quantize_ball_chunk(wf_use, bpw_key, lattice, group_size, q_rot, cs_use)
    else:
        wq_parts: list[torch.Tensor] = []
        syn_tot = 0
        oob_tot = 0
        for r0 in range(0, m, QUANT_ROW_CHUNK):
            r1 = min(r0 + QUANT_ROW_CHUNK, m)
            wq_c, syn_c, oob_c = _quantize_ball_chunk(wf_use[r0:r1], bpw_key, lattice, group_size, q_rot, cs_use)
            wq_parts.append(wq_c)
            syn_tot += syn_c
            oob_tot += oob_c
        wq = torch.cat(wq_parts, dim=0)

    if inv_perm is not None:
        wq = wq[:, inv_perm]
    return wq, syn_tot, oob_tot


# ---------------------------------------------------------------------------
# Permutation Constructors (All Strictly Bijective Over {0, ..., K-1})
# ---------------------------------------------------------------------------
def _make_desc_perm(metric: torch.Tensor) -> torch.Tensor:
    return torch.argsort(metric, descending=True, stable=True)


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


def _make_quad_lpt_perm(
    metric: torch.Tensor,
    group_size: int,
    block_w: int = 4,
    load_power: float = 1.0,
) -> tuple[torch.Tensor, int]:
    """Returns (permutation, rank of 2nd quad placed into Ball 0)."""
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
    ball0_second_rank = -1

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
        if best_b == 0 and ball_counts[0] == 2:
            ball0_second_rank = q

    flat = [c for ball in balls for c in ball]
    return torch.tensor(flat, dtype=torch.int64), ball0_second_rank


def _compute_p_curve_telemetry(
    h_map: dict[str, torch.Tensor], group_size: int
) -> dict[str, dict[str, float]]:
    """Measure intra-quad spread, ball CV (lin & sqrt), and Ball-0 starvation across p-curve."""
    regimes = ("unpermuted", "global_sort") + tuple(lbl for lbl, _ in LOAD_POWER_GRID)
    accum: dict[str, dict[str, list[float]]] = {
        r: {"quad_spread": [], "ball_cv_lin": [], "ball_cv_sqrt": [], "starved": [], "b0_r2_ratio": []}
        for r in regimes
    }
    for _name, hv in h_map.items():
        h_f = hv.detach().to(torch.float64).cpu().clamp_min(1e-12)
        h_norm = h_f / h_f.mean()
        h_sqrt = torch.sqrt(h_norm)
        k = int(h_norm.numel())
        n_balls = k // group_size

        entries: list[tuple[str, torch.Tensor, int]] = [
            ("unpermuted", torch.arange(k, dtype=torch.int64), -1),
            ("global_sort", _make_desc_perm(h_f), 1),
        ]
        for lbl, p_val in LOAD_POWER_GRID:
            if p_val == 0.0:
                entries.append((lbl, _make_quad_snake_perm(h_f, group_size, block_w=4), 2 * n_balls - 1))
            else:
                perm_p, r2_q = _make_quad_lpt_perm(h_f, group_size, block_w=4, load_power=p_val)
                entries.append((lbl, perm_p, r2_q))

        for rname, perm, r2_q in entries:
            hp = h_norm[perm]
            hsp = h_sqrt[perm]
            quads = hp.reshape(-1, 4)
            q_spread = float((quads.amax(dim=1) - quads.amin(dim=1)).mean().item())
            b_lin = hp.reshape(n_balls, group_size).sum(dim=1)
            cv_lin = float((b_lin.std(unbiased=False) / b_lin.mean().clamp_min(1e-12)).item())
            b_sq = hsp.reshape(n_balls, group_size).sum(dim=1)
            cv_sq = float((b_sq.std(unbiased=False) / b_sq.mean().clamp_min(1e-12)).item())
            starved_flag = 1.0 if (r2_q >= 2 * n_balls) else 0.0
            r2_ratio = float(r2_q) / float(n_balls) if r2_q >= 0 else 0.0

            accum[rname]["quad_spread"].append(q_spread)
            accum[rname]["ball_cv_lin"].append(cv_lin)
            accum[rname]["ball_cv_sqrt"].append(cv_sq)
            accum[rname]["starved"].append(starved_flag)
            accum[rname]["b0_r2_ratio"].append(r2_ratio)

    out: dict[str, dict[str, float]] = {}
    for rname in regimes:
        d = accum[rname]
        n_m = len(d["quad_spread"])
        out[rname] = {
            "mean_quad_spread": round(sum(d["quad_spread"]) / n_m, 6),
            "mean_ball_cv_lin": round(sum(d["ball_cv_lin"]) / n_m, 6),
            "mean_ball_cv_sqrt": round(sum(d["ball_cv_sqrt"]) / n_m, 6),
            "starved_matrices_pct": round(100.0 * sum(d["starved"]) / n_m, 4),
            "ball0_mean_second_quad_rank_over_G": round(sum(d["b0_r2_ratio"]) / n_m, 6),
        }
    return out


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


def _snapshot_non_block_tensors(model: torch.nn.Module) -> dict[str, bytes]:
    snaps: dict[str, bytes] = {}
    for name, param in model.named_parameters():
        if not name.startswith("model.layers.") or not name.endswith("_proj.weight"):
            snaps[name] = hashlib.sha256(param.detach().cpu().view(torch.uint8).numpy().tobytes()).digest()
    return snaps


def _count_non_block_diffs(model: torch.nn.Module, snaps: dict[str, bytes]) -> int:
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
    py_test_text: str,
    is_qwen3b: bool,
    device: torch.device,
) -> dict[str, Any]:
    print(f"\n=== MODEL: {repo_id} (g={group_size}) ===", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(local_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        local_dir, local_files_only=True, dtype=torch.bfloat16
    ).to(device)
    model.eval()

    linears = _collect_block_linears(model)
    n_blocks = max(_layer_idx(n) for n, _ in linears) + 1
    non_block_snaps = _snapshot_non_block_tensors(model)
    orig_weights_cpu = {n: m.weight.detach().cpu().clone() for n, m in linears}
    total_block_weights = sum(int(m.weight.numel()) for _, m in linears)

    wt_eval_chunks = _tokenize_chunks(tokenizer, wt_test_text, N_EVAL_CHUNKS, SEQ_LEN)
    py_eval_chunks = _tokenize_chunks(tokenizer, py_test_text, N_EVAL_CHUNKS, SEQ_LEN)
    wt_cal_chunks = _tokenize_chunks(tokenizer, wt_val_text, N_REAL_CAL_CHUNKS, SEQ_LEN)

    ref_wt_lp, ppl_base_wt = _collect_reference_logits(model, wt_eval_chunks, device)
    ref_py_lp, ppl_base_py = _collect_reference_logits(model, py_eval_chunks, device)
    print(
        f"  BASE BF16 PPL: Wiki={ppl_base_wt:.6f} Py={ppl_base_py:.6f} "
        f"({n_blocks}L, {len(linears)} mats, {total_block_weights:,} wts)",
        flush=True,
    )

    rng_gen = torch.Generator().manual_seed(EVAL_SEED)
    vocab_sz = int(model.config.vocab_size)
    rand_chunks = torch.randint(0, vocab_sz, (N_RAND_CHUNKS, SEQ_LEN), generator=rng_gen)
    h_rand = _capture_hessian_diag(model, linears, rand_chunks, device)
    h_cal_real = _capture_hessian_diag(model, linears, wt_cal_chunks, device)

    wcol_norm: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for n, _ in linears:
            wf = orig_weights_cpu[n].to(dtype=torch.float32)
            wcol_norm[n] = (wf ** 2).mean(dim=0)
            del wf

    # Exact p-curve & Ball-0 starvation telemetry (CPU, zero GPU overhead)
    p_telemetry = _compute_p_curve_telemetry(h_cal_real, group_size)
    print(
        f"  [STARVATION h_real] snake={p_telemetry['p0000_snake']['starved_matrices_pct']:.1f}% | "
        f"p025={p_telemetry['p0250']['starved_matrices_pct']:.1f}% | "
        f"p050={p_telemetry['p0500_sqrt']['starved_matrices_pct']:.1f}% | "
        f"p100={p_telemetry['p1000_lin']['starved_matrices_pct']:.1f}%",
        flush=True,
    )

    # Precompute permutation caches on CPU (strictly deterministic)
    perms_cache: dict[str, dict[str, torch.Tensor]] = {
        "wcol_quad_snake": {n: _make_quad_snake_perm(wcol_norm[n], group_size, block_w=4) for n, _ in linears},
        "hrand_quad_snake": {n: _make_quad_snake_perm(h_rand[n], group_size, block_w=4) for n, _ in linears},
        "hreal_global": {n: _make_desc_perm(h_cal_real[n]) for n, _ in linears},
        "hreal_quad_snake": {n: _make_quad_snake_perm(h_cal_real[n], group_size, block_w=4) for n, _ in linears},
        "hreal_quad_lpt_p0500": {
            n: _make_quad_lpt_perm(h_cal_real[n], group_size, block_w=4, load_power=0.5)[0] for n, _ in linears
        },
    }
    if is_qwen3b:
        for lbl, p_val in (("p0125", 0.125), ("p0250", 0.25), ("p0375", 0.375), ("p0750", 0.75), ("p1000", 1.0)):
            perms_cache[f"hreal_quad_lpt_{lbl}"] = {
                n: _make_quad_lpt_perm(h_cal_real[n], group_size, block_w=4, load_power=p_val)[0]
                for n, _ in linears
            }

    H8_MAT = _sylvester_hadamard(8, device, torch.float32)
    BH2_16 = _block_hadamard(16, 2, device, torch.float32)
    BH4_16 = _block_hadamard(16, 4, device, torch.float32)

    arms_out: dict[str, Any] = {}

    def _run_arm(
        arm_name: str,
        bpw_key: str,
        lattice: str,
        q_rot_fn: Any,
        cal_src: dict[str, torch.Tensor] | None = None,
        perm_fn: Any = None,
    ) -> dict[str, Any]:
        tot_err = tot_w = sm_err = sm_w = 0.0
        sm_z = sm_c = syn_t = oob_t = 0
        with torch.no_grad():
            for n, m in linears:
                wf = orig_weights_cpu[n].to(device=device, dtype=torch.float32)
                w_sq = float((wf ** 2).sum().item())
                cs = None
                if cal_src is not None:
                    hv = cal_src[n].to(device=device, dtype=torch.float32).clamp_min(1e-12)
                    cs = (hv / hv.mean()).clamp_min(1e-8) ** 0.25
                    del hv
                q_rot = q_rot_fn(n) if callable(q_rot_fn) else q_rot_fn
                cperm = perm_fn(n) if callable(perm_fn) else None
                if cperm is not None:
                    cperm = cperm.to(device)

                wq, errs, oob = _quantize_ball_matrix(
                    wf, bpw_key, lattice, group_size, q_rot=q_rot, colscale=cs, col_perm=cperm
                )
                dw = wq - wf
                cur_mse = float((dw ** 2).sum().item())

                rms = math.sqrt(w_sq / float(wf.numel()) + 1e-30)
                sm_mask = wf.abs() < (0.05 * rms)
                sc = int(sm_mask.sum().item())
                tot_err += cur_mse
                tot_w += w_sq
                sm_err += float((dw[sm_mask] ** 2).sum().item()) if sc > 0 else 0.0
                sm_w += float((wf[sm_mask] ** 2).sum().item()) if sc > 0 else 0.0
                sm_z += int((wq[sm_mask].abs() < 1e-12).sum().item()) if sc > 0 else 0
                sm_c += sc
                syn_t += errs
                oob_t += oob
                m.weight.copy_(wq.to(torch.bfloat16))
                del wf, wq, dw, sm_mask, cs, cperm

        nb_diffs = _count_non_block_diffs(model, non_block_snaps)
        wt_scores = _score_against_reference(model, wt_eval_chunks, ref_wt_lp, device)
        py_scores = _score_against_reference(model, py_eval_chunks, ref_py_lp, device)
        w_err_db = 10.0 * math.log10(max(tot_err, 1e-30) / max(tot_w, 1e-30))
        e05_db = 10.0 * math.log10(max(sm_err, 1e-30) / max(sm_w, 1e-30))
        z0_pct = 100.0 * float(sm_z) / max(sm_c, 1)
        row = {
            "arm": arm_name,
            "bpw": bpw_key,
            "lattice": lattice,
            "group_size": group_size,
            "ball_bound": int(BALL_BOUNDS[bpw_key][group_size][lattice]),
            "weight_err_db": round(w_err_db, 6),
            "small_weight_e05_db": round(e05_db, 6),
            "small_weight_zero_pct": round(z0_pct, 4),
            "syndrome_errors": syn_t,
            "out_of_ball_groups": oob_t,
            "non_block_tensors_differ": nb_diffs,
            "wikitext2": wt_scores,
            "python": py_scores,
        }
        print(
            f"  [{arm_name:42s}] {bpw_key}b dB={w_err_db:8.4f} E05={e05_db:6.3f} z0={z0_pct:5.1f}% "
            f"WikiKL={wt_scores['mean_kl']:.6f} PPL={wt_scores['ppl']:.3f} "
            f"PyKL={py_scores['mean_kl']:.6f} syn={syn_t} oob={oob_t} nb={nb_diffs}",
            flush=True,
        )
        return row

    def _projtype_rot(n: str) -> torch.Tensor | None:
        return None if _proj_kind(n) == "down_proj" else BH4_16

    def _make_sharednorm_fn(perm_key: str):
        pdict = perms_cache[perm_key]
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

    # =========================================================================
    # PILLAR 1 (4.5 bpw): PTX-Ready Shared-RMSNorm quad_snake & quad_lpt_sqrt
    # =========================================================================
    arms_out["b45_hA16_BlockH2_real"] = _run_arm("b45_hA16_BlockH2_real", "4.5", "A16", BH2_16, cal_src=h_cal_real)
    arms_out["b45_hA16_BlockH4_real"] = _run_arm("b45_hA16_BlockH4_real", "4.5", "A16", BH4_16, cal_src=h_cal_real)
    arms_out["b45_hA16_BlockH4_quad_snake_real"] = _run_arm(
        "b45_hA16_BlockH4_quad_snake_real", "4.5", "A16", BH4_16, cal_src=h_cal_real,
        perm_fn=lambda n: perms_cache["hreal_quad_snake"][n],
    )
    arms_out["b45_hA16_BlockH4_quad_lpt_sqrt_real"] = _run_arm(
        "b45_hA16_BlockH4_quad_lpt_sqrt_real", "4.5", "A16", BH4_16, cal_src=h_cal_real,
        perm_fn=lambda n: perms_cache["hreal_quad_lpt_p0500"][n],
    )
    arms_out["b45_hA16_BlockH4_quad_snake_sharednorm"] = _run_arm(
        "b45_hA16_BlockH4_quad_snake_sharednorm", "4.5", "A16", BH4_16, cal_src=h_cal_real,
        perm_fn=_make_sharednorm_fn("hreal_quad_snake"),
    )
    arms_out["b45_hA16_BlockH4_quad_lpt_sqrt_sharednorm"] = _run_arm(
        "b45_hA16_BlockH4_quad_lpt_sqrt_sharednorm", "4.5", "A16", BH4_16, cal_src=h_cal_real,
        perm_fn=_make_sharednorm_fn("hreal_quad_lpt_p0500"),
    )
    arms_out["b45_hA16_BlockH4_quad_snake_down_zerocost"] = _run_arm(
        "b45_hA16_BlockH4_quad_snake_down_zerocost", "4.5", "A16", BH4_16, cal_src=h_cal_real,
        perm_fn=lambda n: perms_cache["hreal_quad_snake"][n] if _proj_kind(n) == "down_proj" else None,
    )

    # =========================================================================
    # PILLAR 2 (4.5 bpw, Qwen2.5-3B): Exact 7-Point Load-Power Phase Curve
    # =========================================================================
    if is_qwen3b:
        arms_out["b45_hA16_BlockH4_sort_real"] = _run_arm(
            "b45_hA16_BlockH4_sort_real", "4.5", "A16", BH4_16, cal_src=h_cal_real,
            perm_fn=lambda n: perms_cache["hreal_global"][n],
        )
        for lbl in ("p0125", "p0250", "p0375", "p0750", "p1000"):
            arm_nm = f"b45_hA16_BlockH4_quad_lpt_{lbl}_real"
            pk = f"hreal_quad_lpt_{lbl}"
            arms_out[arm_nm] = _run_arm(
                arm_nm, "4.5", "A16", BH4_16, cal_src=h_cal_real,
                perm_fn=lambda n, key=pk: perms_cache[key][n],
            )
        arms_out["b45_hA16_BlockH4_quad_lpt_p0250_sharednorm"] = _run_arm(
            "b45_hA16_BlockH4_quad_lpt_p0250_sharednorm", "4.5", "A16", BH4_16, cal_src=h_cal_real,
            perm_fn=_make_sharednorm_fn("hreal_quad_lpt_p0250"),
        )

    # =========================================================================
    # PILLAR 3 (3.0 bpw exact siliquant#85 codebooks): Low-Bitrate Frontier
    # =========================================================================
    arms_out["b30_E8_H8"] = _run_arm("b30_E8_H8", "3.0", "E8", H8_MAT)
    arms_out["b30_A16_axis"] = _run_arm("b30_A16_axis", "3.0", "A16", None)
    arms_out["b30_A16_BlockH2"] = _run_arm("b30_A16_BlockH2", "3.0", "A16", BH2_16)
    arms_out["b30_A16_BlockH4"] = _run_arm("b30_A16_BlockH4", "3.0", "A16", BH4_16)
    arms_out["b30_ProjType_Uniform"] = _run_arm("b30_ProjType_Uniform", "3.0", "A16", _projtype_rot)
    arms_out["b30_A16_BlockH4_quad_snake_wcol"] = _run_arm(
        "b30_A16_BlockH4_quad_snake_wcol", "3.0", "A16", BH4_16,
        perm_fn=lambda n: perms_cache["wcol_quad_snake"][n],
    )
    arms_out["b30_ProjType_Uniform_QuadSnake_wcol"] = _run_arm(
        "b30_ProjType_Uniform_QuadSnake_wcol", "3.0", "A16", _projtype_rot,
        perm_fn=lambda n: None if _proj_kind(n) == "down_proj" else perms_cache["wcol_quad_snake"][n],
    )
    arms_out["b30_hA16_BlockH4_quad_snake_rand"] = _run_arm(
        "b30_hA16_BlockH4_quad_snake_rand", "3.0", "A16", BH4_16, cal_src=h_rand,
        perm_fn=lambda n: perms_cache["hrand_quad_snake"][n],
    )
    arms_out["b30_hA16_real"] = _run_arm("b30_hA16_real", "3.0", "A16", None, cal_src=h_cal_real)
    arms_out["b30_hA16_BlockH2_real"] = _run_arm("b30_hA16_BlockH2_real", "3.0", "A16", BH2_16, cal_src=h_cal_real)
    arms_out["b30_hA16_BlockH4_real"] = _run_arm("b30_hA16_BlockH4_real", "3.0", "A16", BH4_16, cal_src=h_cal_real)
    arms_out["b30_hA16_BlockH4_sort_real"] = _run_arm(
        "b30_hA16_BlockH4_sort_real", "3.0", "A16", BH4_16, cal_src=h_cal_real,
        perm_fn=lambda n: perms_cache["hreal_global"][n],
    )
    arms_out["b30_hA16_BlockH4_quad_snake_real"] = _run_arm(
        "b30_hA16_BlockH4_quad_snake_real", "3.0", "A16", BH4_16, cal_src=h_cal_real,
        perm_fn=lambda n: perms_cache["hreal_quad_snake"][n],
    )
    arms_out["b30_hA16_BlockH4_quad_lpt_sqrt_real"] = _run_arm(
        "b30_hA16_BlockH4_quad_lpt_sqrt_real", "3.0", "A16", BH4_16, cal_src=h_cal_real,
        perm_fn=lambda n: perms_cache["hreal_quad_lpt_p0500"][n],
    )
    arms_out["b30_hA16_BlockH4_quad_snake_sharednorm"] = _run_arm(
        "b30_hA16_BlockH4_quad_snake_sharednorm", "3.0", "A16", BH4_16, cal_src=h_cal_real,
        perm_fn=_make_sharednorm_fn("hreal_quad_snake"),
    )
    if is_qwen3b:
        arms_out["b30_hA16_BlockH4_quad_lpt_p1000_real"] = _run_arm(
            "b30_hA16_BlockH4_quad_lpt_p1000_real", "3.0", "A16", BH4_16, cal_src=h_cal_real,
            perm_fn=lambda n: perms_cache["hreal_quad_lpt_p1000"][n],
        )

    out = {
        "repo_id": repo_id,
        "group_size": group_size,
        "num_hidden_layers": n_blocks,
        "block_linear_matrices": len(linears),
        "total_block_weights": total_block_weights,
        "non_block_tensors_protected": len(non_block_snaps),
        "reference_bf16": {
            "wikitext2_ppl": round(ppl_base_wt, 6),
            "python_ppl": round(ppl_base_py, 6),
        },
        "p_curve_starvation_telemetry_hreal": p_telemetry,
        "arms": arms_out,
    }
    del model, ref_wt_lp, ref_py_lp, orig_weights_cpu, h_rand, h_cal_real, wcol_norm, perms_cache
    gc.collect()
    torch.cuda.empty_cache()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Track S73 CEER Cycle-6 Live RTX 3080 Runner")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "rtx3080_s73_empirical_results.json",
    )
    args = parser.parse_args()

    torch.manual_seed(EVAL_SEED)
    device = torch.device("cuda:0")
    here = Path(__file__).resolve().parent

    prereg_raw = (here / "s73_preregistration.json").read_bytes()
    prereg_file_sha256 = hashlib.sha256(prereg_raw).hexdigest()
    prereg_obj = json.loads(prereg_raw.decode("utf-8"))

    s49b_cache = Path.home() / ".cache" / "huggingface" / "s49b_eval"
    s60_cache = Path.home() / ".cache" / "huggingface" / "s60_eval"
    s63_cache = Path.home() / ".cache" / "huggingface" / "s63_eval"
    s49b_pins = json.loads((s49b_cache / "pins.json").read_text(encoding="utf-8"))
    q15_pin = json.loads((s60_cache / "qwen15b_pin.json").read_text(encoding="utf-8"))
    q3_pin = json.loads((s63_cache / "qwen3b_pin.json").read_text(encoding="utf-8"))

    wt_val_text = (s49b_cache / "wikitext2_val.txt").read_text(encoding="utf-8")
    wt_test_text = (s49b_cache / "wikitext2_test.txt").read_text(encoding="utf-8")
    py_test_text, py_sha256 = _load_python_eval_text()

    ac_pre = Path("/sys/class/power_supply/AC0/online").read_text(encoding="utf-8").strip()
    if ac_pre != "1":
        raise RuntimeError("AC power must be online before sustained GPU sweeps")

    t_start = time.monotonic()
    smi_pre = _query_nvidia_smi()
    rapl_max = _read_rapl_max_range_uj()
    rapl_start = _read_rapl_pkg_uj()

    models_out: dict[str, Any] = {}
    models_out["HuggingFaceTB/SmolLM2-360M"] = _eval_model(
        "HuggingFaceTB/SmolLM2-360M",
        s49b_pins["HuggingFaceTB/SmolLM2-360M"]["local_dir"],
        32, wt_val_text, wt_test_text, py_test_text, is_qwen3b=False, device=device,
    )
    models_out["Qwen/Qwen2.5-0.5B"] = _eval_model(
        "Qwen/Qwen2.5-0.5B",
        s49b_pins["Qwen/Qwen2.5-0.5B"]["local_dir"],
        128, wt_val_text, wt_test_text, py_test_text, is_qwen3b=False, device=device,
    )
    models_out["Qwen/Qwen2.5-1.5B"] = _eval_model(
        "Qwen/Qwen2.5-1.5B",
        q15_pin["local_dir"],
        128, wt_val_text, wt_test_text, py_test_text, is_qwen3b=False, device=device,
    )
    models_out["Qwen/Qwen2.5-3B"] = _eval_model(
        "Qwen/Qwen2.5-3B",
        q3_pin["local_dir"],
        128, wt_val_text, wt_test_text, py_test_text, is_qwen3b=True, device=device,
    )

    rapl_end = _read_rapl_pkg_uj()
    smi_post = _query_nvidia_smi()
    ac_post = Path("/sys/class/power_supply/AC0/online").read_text(encoding="utf-8").strip()

    payload = {
        "schema": "aletheia.track_s73.lowbit_and_sharednorm_quad_snake.v1",
        "oracle_class": "sampled",
        "measured_live_model_weights": True,
        "preregistration_file_sha256": prereg_file_sha256,
        "preregistration_sha256": prereg_obj["preregistration_sha256"],
        "eval_seed": EVAL_SEED,
        "ac_online": ac_pre == "1" and ac_post == "1",
        "ball_bounds": {
            bpw: {str(g): {lat: int(v) for lat, v in lm.items()} for g, lm in gm.items()}
            for bpw, gm in BALL_BOUNDS.items()
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
            "n_rand_cal_chunks": N_RAND_CHUNKS,
            "n_real_cal_chunks": N_REAL_CAL_CHUNKS,
        },
        "model_pins": {
            "HuggingFaceTB/SmolLM2-360M": {
                "revision": s49b_pins["HuggingFaceTB/SmolLM2-360M"]["revision"],
                "safetensors_sha256": s49b_pins["HuggingFaceTB/SmolLM2-360M"]["model_safetensors_sha256"],
                "tokenizer_sha256": s49b_pins["HuggingFaceTB/SmolLM2-360M"]["tokenizer_json_sha256"],
            },
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
            "Qwen/Qwen2.5-3B": {
                "revision": q3_pin["revision"],
                "safetensors_shards": q3_pin["safetensors"],
                "model_index_sha256": q3_pin["model_index_sha256"],
                "tokenizer_sha256": q3_pin["tokenizer_json_sha256"],
            },
        },
        "models": models_out,
        "total_cpu_rapl_joules": _rapl_delta_joules(rapl_start, rapl_end, rapl_max),
        "nvidia_smi_pre": smi_pre,
        "nvidia_smi_post": smi_post,
        "elapsed_seconds": round(time.monotonic() - t_start, 3),
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nWROTE {args.output.name} ({args.output.stat().st_size} bytes, elapsed={payload['elapsed_seconds']}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
