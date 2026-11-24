"""Track S72 CEER Cycle-5 Live RTX 3080 Evaluator: Two-Level Stratified Quad-Interleaved A16_BlockH4.

Tests the exact two-level saddle-point permutation across all 4 cached real-weight models
(SmolLM2-360M, Qwen2.5-0.5B, Qwen2.5-1.5B, Qwen2.5-3B; 840 matrices, 4,757,127,168 weights):
  - Stage 1 (micro H_4 quad / H_2 pair homogeneity): global descending sort into monotone blocks
  - Stage 2 (macro g=128/g=32 ball balance): LPT greedy load-balancing (quad_lpt) & boustrophedon
    snake interleaving (quad_snake) across all G = K/g balls
  - Direct ball-variance (CV_ball) & intra-quad spread telemetry across 5 permutation regimes
  - Zero-runtime-cost down_proj offline fold & PTX-ready shared-RMSNorm variants
  - Zero-text weight-norm (wcol) & random-token (hrand) permutations + ProjType_Uniform combos
  - Matched-MSE (<= 2e-6 dB) controls & matched BlockH2_quad_lpt control
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

EVAL_SEED = 2026112472
SEQ_LEN = 512
N_EVAL_CHUNKS = 8
N_RAND_CHUNKS = 8
N_REAL_CAL_CHUNKS = 12
QUANT_ROW_CHUNK = 512
LOGIT_TOK_CHUNK = 128

BALL_BOUNDS = {
    128: {"A16": 5213.0, "E8": 3380.0, "Z": 3380.0},
    32: {"A16": 1207.0, "E8": 782.0, "Z": 782.0},
}

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
    """Exact nearest lattice point in A_16 = 2 Z^16 + C_{[16,11,4]}."""
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


def _quantize_ball_chunk(
    w_chunk: torch.Tensor,
    group_size: int,
    q_rot: torch.Tensor | None = None,
    colscale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    X = w_chunk.clone()
    if colscale is not None:
        X = X * colscale.unsqueeze(0)
    if q_rot is not None:
        X = _apply_block_rot(X, q_rot)

    Xg = X.reshape(-1, group_size)
    bound = BALL_BOUNDS[group_size]["A16"]
    r_ball = math.sqrt(bound)
    margin = math.sqrt(group_size / 4.0 + 1.5 * (group_size / 16.0))

    g_norm = torch.sqrt((Xg ** 2).sum(dim=1)).clamp_min(1e-30)
    lo = g_norm / (r_ball + margin) * 0.99
    hi = g_norm / max(r_ball - margin, 0.05 * r_ball) * 1.01

    for _ in range(16):
        mid = torch.sqrt(lo * hi)
        scaled = Xg / mid.unsqueeze(1)
        Y, _ = _a16_closest(scaled.reshape(-1, 16))
        Y = Y.reshape(-1, group_size)
        ok = (Y ** 2).sum(dim=1) <= bound
        hi = torch.where(ok, mid, hi)
        lo = torch.where(ok, lo, mid)

    scaled_f = Xg / hi.unsqueeze(1)
    Y_fin, syn_errs = _a16_closest(scaled_f.reshape(-1, 16))
    Y_fin = Y_fin.reshape(-1, group_size)

    out_m = (Y_fin ** 2).sum(dim=1) > bound
    if out_m.any():
        xo = scaled_f[out_m]
        s_lo = torch.zeros(xo.shape[0], device=w_chunk.device, dtype=w_chunk.dtype)
        s_hi = torch.ones(xo.shape[0], device=w_chunk.device, dtype=w_chunk.dtype)
        for _ in range(12):
            sm = 0.5 * (s_lo + s_hi)
            yt, _ = _a16_closest((xo * sm.unsqueeze(1)).reshape(-1, 16))
            yt = yt.reshape(-1, group_size)
            good = (yt ** 2).sum(dim=1) <= bound
            s_lo = torch.where(good, sm, s_lo)
            s_hi = torch.where(good, s_hi, sm)
        yf2, se2 = _a16_closest((xo * s_lo.unsqueeze(1)).reshape(-1, 16))
        Y_fin[out_m] = yf2.reshape(-1, group_size)
        syn_errs += se2

    scales_out = hi.to(torch.float16).to(torch.float32)
    Wq = (Y_fin * scales_out.unsqueeze(1)).reshape(w_chunk.shape)
    if q_rot is not None:
        Wq = _apply_block_rot(Wq, q_rot.T)
    if colscale is not None:
        Wq = Wq / colscale.unsqueeze(0)
    return Wq, syn_errs


def _quantize_ball_matrix(
    w_f32: torch.Tensor,
    group_size: int,
    q_rot: torch.Tensor | None = None,
    colscale: torch.Tensor | None = None,
    col_perm: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
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
        wq, syn_tot = _quantize_ball_chunk(wf_use, group_size, q_rot, cs_use)
    else:
        wq_parts: list[torch.Tensor] = []
        syn_tot = 0
        for r0 in range(0, m, QUANT_ROW_CHUNK):
            r1 = min(r0 + QUANT_ROW_CHUNK, m)
            wq_c, syn_c = _quantize_ball_chunk(wf_use[r0:r1], group_size, q_rot, cs_use)
            wq_parts.append(wq_c)
            syn_tot += syn_c
        wq = torch.cat(wq_parts, dim=0)

    if inv_perm is not None:
        wq = wq[:, inv_perm]
    return wq, syn_tot


# ---------------------------------------------------------------------------
# Permutation Constructors (All Strictly Bijective Over {0, ..., K-1})
# ---------------------------------------------------------------------------
def _make_desc_perm(metric: torch.Tensor, local_group: int | None = None) -> torch.Tensor:
    k = metric.numel()
    if local_group is None:
        return torch.argsort(metric, descending=True, stable=True)
    n_g = k // local_group
    parts = []
    for g in range(n_g):
        base = g * local_group
        parts.append(torch.argsort(metric[base : base + local_group], descending=True, stable=True) + base)
    return torch.cat(parts, dim=0)


def _make_quad_snake_perm(metric: torch.Tensor, group_size: int, block_w: int = 4) -> torch.Tensor:
    """Stage 1: global descending sort into monotone blocks of width block_w.
    Stage 2: boustrophedon snake distribution across G = K / group_size balls.
    """
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
) -> torch.Tensor:
    """Stage 1: global descending sort into monotone blocks of width block_w.
    Stage 2: Longest-Processing-Time (LPT) greedy load balancing across G = K / group_size balls
    with exact capacity constraint (group_size / block_w blocks per ball).
    """
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


def _compute_regime_telemetry(
    h_map: dict[str, torch.Tensor], group_size: int
) -> dict[str, dict[str, float]]:
    """Measure mean normalized intra-quad spread & inter-ball CV across 5 permutation regimes."""
    regimes = ("unpermuted", "global_sort", "grp128_sort", "quad_snake", "quad_lpt")
    accum: dict[str, dict[str, list[float]]] = {
        r: {"quad_spread": [], "ball_cv_lin": [], "ball_cv_sqrt": [], "down_quad_spread": [], "down_ball_cv_lin": []}
        for r in regimes
    }
    for name, hv in h_map.items():
        h_f = hv.detach().to(torch.float64).cpu().clamp_min(1e-12)
        h_norm = h_f / h_f.mean()
        h_sqrt = torch.sqrt(h_norm)
        k = int(h_norm.numel())
        n_balls = k // group_size
        is_down = name.endswith(".down_proj")

        perms = {
            "unpermuted": torch.arange(k, dtype=torch.int64),
            "global_sort": _make_desc_perm(h_f),
            "grp128_sort": _make_desc_perm(h_f, local_group=group_size),
            "quad_snake": _make_quad_snake_perm(h_f, group_size, block_w=4),
            "quad_lpt": _make_quad_lpt_perm(h_f, group_size, block_w=4, load_power=1.0),
        }
        for rname, perm in perms.items():
            hp = h_norm[perm]
            hsp = h_sqrt[perm]
            quads = hp.reshape(-1, 4)
            q_spread = float((quads.amax(dim=1) - quads.amin(dim=1)).mean().item())

            b_lin = hp.reshape(n_balls, group_size).sum(dim=1)
            cv_lin = float((b_lin.std(unbiased=False) / b_lin.mean().clamp_min(1e-12)).item())

            b_sq = hsp.reshape(n_balls, group_size).sum(dim=1)
            cv_sq = float((b_sq.std(unbiased=False) / b_sq.mean().clamp_min(1e-12)).item())

            accum[rname]["quad_spread"].append(q_spread)
            accum[rname]["ball_cv_lin"].append(cv_lin)
            accum[rname]["ball_cv_sqrt"].append(cv_sq)
            if is_down:
                accum[rname]["down_quad_spread"].append(q_spread)
                accum[rname]["down_ball_cv_lin"].append(cv_lin)

    out: dict[str, dict[str, float]] = {}
    for rname in regimes:
        d = accum[rname]
        out[rname] = {
            "mean_quad_spread": round(sum(d["quad_spread"]) / len(d["quad_spread"]), 6),
            "mean_ball_cv_lin": round(sum(d["ball_cv_lin"]) / len(d["ball_cv_lin"]), 6),
            "mean_ball_cv_sqrt": round(sum(d["ball_cv_sqrt"]) / len(d["ball_cv_sqrt"]), 6),
            "down_proj_mean_quad_spread": round(sum(d["down_quad_spread"]) / len(d["down_quad_spread"]), 6),
            "down_proj_mean_ball_cv_lin": round(sum(d["down_ball_cv_lin"]) / len(d["down_ball_cv_lin"]), 6),
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
    print(f"  BASE BF16 PPL: Wiki={ppl_base_wt:.6f} Py={ppl_base_py:.6f} ({n_blocks}L, {len(linears)} mats, {total_block_weights:,} wts)", flush=True)

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

    # Compute exact Ball-Variance & Intra-Quad Spread Telemetry Diagnostic (P_S72_2)
    telemetry_real = _compute_regime_telemetry(h_cal_real, group_size)
    telemetry_wcol = _compute_regime_telemetry(wcol_norm, group_size)
    print(
        f"  [TELEMETRY h_real] unp=(spr={telemetry_real['unpermuted']['mean_quad_spread']:.4f}, cv={telemetry_real['unpermuted']['mean_ball_cv_lin']:.4f}) | "
        f"glob=(spr={telemetry_real['global_sort']['mean_quad_spread']:.4f}, cv={telemetry_real['global_sort']['mean_ball_cv_lin']:.4f}) | "
        f"grp=(spr={telemetry_real['grp128_sort']['mean_quad_spread']:.4f}, cv={telemetry_real['grp128_sort']['mean_ball_cv_lin']:.4f}) | "
        f"lpt=(spr={telemetry_real['quad_lpt']['mean_quad_spread']:.4f}, cv={telemetry_real['quad_lpt']['mean_ball_cv_lin']:.4f})",
        flush=True,
    )

    # Precompute permutation caches on CPU (strictly deterministic & fast!)
    perms_cache: dict[str, dict[str, torch.Tensor]] = {
        "wcol_global": {n: _make_desc_perm(wcol_norm[n]) for n, _ in linears},
        "wcol_quad_lpt": {n: _make_quad_lpt_perm(wcol_norm[n], group_size, block_w=4) for n, _ in linears},
        "wcol_quad_snake": {n: _make_quad_snake_perm(wcol_norm[n], group_size, block_w=4) for n, _ in linears},
        "hrand_quad_lpt": {n: _make_quad_lpt_perm(h_rand[n], group_size, block_w=4) for n, _ in linears},
        "hreal_global": {n: _make_desc_perm(h_cal_real[n]) for n, _ in linears},
        "hreal_grp": {n: _make_desc_perm(h_cal_real[n], local_group=group_size) for n, _ in linears},
        "hreal_quad_lpt": {n: _make_quad_lpt_perm(h_cal_real[n], group_size, block_w=4) for n, _ in linears},
        "hreal_quad_snake": {n: _make_quad_snake_perm(h_cal_real[n], group_size, block_w=4) for n, _ in linears},
        "hreal_pair_lpt": {n: _make_quad_lpt_perm(h_cal_real[n], group_size, block_w=2) for n, _ in linears},
    }
    if is_qwen3b:
        perms_cache["hreal_quad_lpt_sqrt"] = {
            n: _make_quad_lpt_perm(h_cal_real[n], group_size, block_w=4, load_power=0.5) for n, _ in linears
        }

    BH2_16 = _block_hadamard(16, 2, device, torch.float32)
    BH4_16 = _block_hadamard(16, 4, device, torch.float32)

    arms_out: dict[str, Any] = {}
    axis_layer_mse: dict[str, float] = {}

    def _run_arm(
        arm_name: str,
        q_rot_fn: Any,
        cal_src: dict[str, torch.Tensor] | None = None,
        perm_fn: Any = None,
        match_ref: dict[str, float] | None = None,
        record_mse: bool = False,
    ) -> dict[str, Any]:
        tot_err = tot_w = sm_err = sm_w = 0.0
        sm_z = sm_c = syn_t = 0
        max_mdiff = 0.0
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

                wq, errs = _quantize_ball_matrix(wf, group_size, q_rot=q_rot, colscale=cs, col_perm=cperm)
                dw = wq - wf
                cur_mse = float((dw ** 2).sum().item())
                if match_ref is not None:
                    tgt = match_ref[n]
                    if cur_mse > 0.0:
                        alpha = math.sqrt(tgt / cur_mse)
                        dw = dw * alpha
                        wq = wf + dw
                        matched_mse = float((dw ** 2).sum().item())
                        db_d = abs(
                            10.0 * math.log10(max(matched_mse, 1e-30) / max(w_sq, 1e-30))
                            - 10.0 * math.log10(max(tgt, 1e-30) / max(w_sq, 1e-30))
                        )
                        if db_d > max_mdiff:
                            max_mdiff = db_d
                        cur_mse = matched_mse

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
                if record_mse:
                    axis_layer_mse[n] = cur_mse
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
            "group_size": group_size,
            "weight_err_db": round(w_err_db, 6),
            "small_weight_e05_db": round(e05_db, 6),
            "small_weight_zero_pct": round(z0_pct, 4),
            "syndrome_errors": syn_t,
            "non_block_tensors_differ": nb_diffs,
            "max_matched_mse_db_diff": round(max_mdiff, 9),
            "wikitext2": wt_scores,
            "python": py_scores,
        }
        print(
            f"  [{arm_name:36s}] dB={w_err_db:8.4f} E05={e05_db:6.3f} z0={z0_pct:5.1f}% "
            f"WikiKL={wt_scores['mean_kl']:.6f} PPL={wt_scores['ppl']:.3f} "
            f"PyKL={py_scores['mean_kl']:.6f} syn={syn_t} nb={nb_diffs}",
            flush=True,
        )
        return row

    def _projtype_rot(n: str) -> torch.Tensor | None:
        return None if _proj_kind(n) == "down_proj" else BH4_16

    def _sharednorm_quad_lpt(n: str) -> torch.Tensor | None:
        kd = _proj_kind(n)
        l_i = _layer_idx(n)
        if kd == "down_proj":
            return perms_cache["hreal_quad_lpt"][n]
        if kd in ("q_proj", "k_proj", "v_proj"):
            return perms_cache["hreal_quad_lpt"][f"model.layers.{l_i}.self_attn.q_proj"]
        if kd in ("gate_proj", "up_proj"):
            return perms_cache["hreal_quad_lpt"][f"model.layers.{l_i}.mlp.gate_proj"]
        return None

    # --- PART A: Zero-Text / Zero-Cal Baselines & Two-Level Stratified Arms (P_S72_2) ---
    arms_out["A16_axis"] = _run_arm("A16_axis", None, record_mse=True)
    arms_out["A16_BlockH4"] = _run_arm("A16_BlockH4", BH4_16)
    arms_out["A16_BlockH4_sort_wcol"] = _run_arm("A16_BlockH4_sort_wcol", BH4_16, perm_fn=lambda n: perms_cache["wcol_global"][n])
    arms_out["A16_BlockH4_quad_lpt_wcol"] = _run_arm("A16_BlockH4_quad_lpt_wcol", BH4_16, perm_fn=lambda n: perms_cache["wcol_quad_lpt"][n])
    arms_out["A16_BlockH4_quad_snake_wcol"] = _run_arm("A16_BlockH4_quad_snake_wcol", BH4_16, perm_fn=lambda n: perms_cache["wcol_quad_snake"][n])
    arms_out["A16_BlockH4_quad_lpt_hrand"] = _run_arm("A16_BlockH4_quad_lpt_hrand", BH4_16, perm_fn=lambda n: perms_cache["hrand_quad_lpt"][n])
    arms_out["A16_BlockH4_quad_lpt_down_wcol"] = _run_arm(
        "A16_BlockH4_quad_lpt_down_wcol", BH4_16,
        perm_fn=lambda n: perms_cache["wcol_quad_lpt"][n] if _proj_kind(n) == "down_proj" else None,
    )
    arms_out["ProjType_Uniform"] = _run_arm("ProjType_Uniform", _projtype_rot)
    arms_out["ProjType_Uniform_QuadLPT_wcol"] = _run_arm(
        "ProjType_Uniform_QuadLPT_wcol", _projtype_rot,
        perm_fn=lambda n: None if _proj_kind(n) == "down_proj" else perms_cache["wcol_quad_lpt"][n],
    )
    arms_out["ProjType_Uniform_QuadLPT_hrand"] = _run_arm(
        "ProjType_Uniform_QuadLPT_hrand", _projtype_rot,
        perm_fn=lambda n: None if _proj_kind(n) == "down_proj" else perms_cache["hrand_quad_lpt"][n],
    )
    arms_out["hA16_BlockH4_quad_lpt_rand"] = _run_arm(
        "hA16_BlockH4_quad_lpt_rand", BH4_16, cal_src=h_rand, perm_fn=lambda n: perms_cache["hrand_quad_lpt"][n]
    )

    if is_qwen3b:
        arms_out["A16_BlockH4_quad_lpt_wcol_matched"] = _run_arm(
            "A16_BlockH4_quad_lpt_wcol_matched", BH4_16,
            perm_fn=lambda n: perms_cache["wcol_quad_lpt"][n], match_ref=axis_layer_mse,
        )
        arms_out["A16_BlockH4_quad_lpt_hrand_matched"] = _run_arm(
            "A16_BlockH4_quad_lpt_hrand_matched", BH4_16,
            perm_fn=lambda n: perms_cache["hrand_quad_lpt"][n], match_ref=axis_layer_mse,
        )

    # --- PART B: Calibrated Two-Level Saddle-Point Dominance & BlockH2 Matched Control (P_S72_1 & P_S72_3) ---
    arms_out["hA16_real"] = _run_arm("hA16_real", None, cal_src=h_cal_real)
    arms_out["hA16_BlockH2_real"] = _run_arm("hA16_BlockH2_real", BH2_16, cal_src=h_cal_real)
    arms_out["hA16_BlockH4_real"] = _run_arm("hA16_BlockH4_real", BH4_16, cal_src=h_cal_real)
    arms_out["hA16_BlockH4_sort_real"] = _run_arm(
        "hA16_BlockH4_sort_real", BH4_16, cal_src=h_cal_real, perm_fn=lambda n: perms_cache["hreal_global"][n]
    )
    arms_out["hA16_BlockH4_sort_grp128_real"] = _run_arm(
        "hA16_BlockH4_sort_grp128_real", BH4_16, cal_src=h_cal_real, perm_fn=lambda n: perms_cache["hreal_grp"][n]
    )
    arms_out["hA16_BlockH4_quad_lpt_real"] = _run_arm(
        "hA16_BlockH4_quad_lpt_real", BH4_16, cal_src=h_cal_real, perm_fn=lambda n: perms_cache["hreal_quad_lpt"][n]
    )
    arms_out["hA16_BlockH4_quad_snake_real"] = _run_arm(
        "hA16_BlockH4_quad_snake_real", BH4_16, cal_src=h_cal_real, perm_fn=lambda n: perms_cache["hreal_quad_snake"][n]
    )
    arms_out["hA16_BlockH4_quad_lpt_down_zerocost"] = _run_arm(
        "hA16_BlockH4_quad_lpt_down_zerocost", BH4_16, cal_src=h_cal_real,
        perm_fn=lambda n: perms_cache["hreal_quad_lpt"][n] if _proj_kind(n) == "down_proj" else None,
    )
    arms_out["hA16_BlockH4_quad_lpt_sharednorm"] = _run_arm(
        "hA16_BlockH4_quad_lpt_sharednorm", BH4_16, cal_src=h_cal_real, perm_fn=_sharednorm_quad_lpt
    )
    arms_out["hA16_BlockH2_quad_lpt_real"] = _run_arm(
        "hA16_BlockH2_quad_lpt_real", BH2_16, cal_src=h_cal_real, perm_fn=lambda n: perms_cache["hreal_pair_lpt"][n]
    )
    if is_qwen3b:
        arms_out["hA16_BlockH4_quad_lpt_sqrt_real"] = _run_arm(
            "hA16_BlockH4_quad_lpt_sqrt_real", BH4_16, cal_src=h_cal_real,
            perm_fn=lambda n: perms_cache["hreal_quad_lpt_sqrt"][n],
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
        "ball_variance_telemetry_hreal": telemetry_real,
        "ball_variance_telemetry_wcol": telemetry_wcol,
        "arms": arms_out,
    }
    del model, ref_wt_lp, ref_py_lp, orig_weights_cpu, h_rand, h_cal_real, wcol_norm, perms_cache
    gc.collect()
    torch.cuda.empty_cache()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Track S72 CEER Cycle-5 Stratified Quad-Interleaved Block-H4 Runner")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "rtx3080_s72_empirical_results.json",
    )
    args = parser.parse_args()

    torch.manual_seed(EVAL_SEED)
    device = torch.device("cuda:0")
    here = Path(__file__).resolve().parent

    prereg_raw = (here / "s72_preregistration.json").read_bytes()
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
        "schema": "aletheia.track_s72.stratified_quad_interleaved_blockh4.v1",
        "oracle_class": "sampled",
        "measured_live_model_weights": True,
        "preregistration_file_sha256": prereg_file_sha256,
        "preregistration_sha256": prereg_obj["preregistration_sha256"],
        "eval_seed": EVAL_SEED,
        "ac_online": ac_pre == "1" and ac_post == "1",
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
