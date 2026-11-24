"""Track S69: CEER Cycle-4 Column-Sorted / Zero-Runtime-Cost Permuted A16_BlockH4,
Cross-Layer Non-Additivity Dissection & Coupled-Coherent Routing, and sm_86 PTX Serving Benchmark.

Offline evaluation using ONLY already-cached real open-weight pretrained models (zero downloads):
  - HuggingFaceTB/SmolLM2-360M (32L, 224 mats, g=32) + full 32-block single-layer additivity
  - Qwen/Qwen2.5-0.5B          (24L, 168 mats, g=128)
  - Qwen/Qwen2.5-1.5B          (28L, 196 mats, g=128)
  - Qwen/Qwen2.5-3B            (36L, 252 mats, g=128) + exact BF16 fold check + 36-block additivity + sm_86 PTX bench
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import re
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

EVAL_SEED = 2026112469
SEQ_LEN = 512
N_EVAL_CHUNKS = 8      # 4,088 held-out next-token logits per corpus
N_RAND_CHUNKS = 8      # 4,096 zero-data uniform random tokens
N_REAL_CAL_CHUNKS = 12 # 6,144 disjoint WikiText-2 val tokens
QUANT_ROW_CHUNK = 512  # VRAM-safe row chunking on 8 GiB RTX 3080
LOGIT_TOK_CHUNK = 128  # VRAM-safe token chunking for FP32 vocab KL/PPL

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


def _query_nvidia_smi() -> dict[str, str]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,pstate,clocks.current.sm,clocks.current.memory,power.draw,temperature.gpu,memory.used,clocks_event_reasons.active",
        "--format=csv,noheader,nounits",
    ]
    out = subprocess.check_output(cmd, text=True).strip().splitlines()[0]
    cols = [c.strip() for c in out.split(",")]
    keys = [
        "name",
        "driver_version",
        "pstate",
        "sm_clock_mhz",
        "mem_clock_mhz",
        "power_draw_w",
        "temp_c",
        "memory_used_mib",
        "throttle_hex",
    ]
    return dict(zip(keys, cols))


def _load_python_eval_text() -> tuple[str, str]:
    stdlib_dir = Path(json.__file__).resolve().parent.parent
    parts: list[str] = []
    for rel in STDLIB_PY_MODULES:
        p = stdlib_dir / rel
        if p.exists():
            parts.append(p.read_text(encoding="utf-8"))
    text = "\n\n".join(parts)
    return text, hashlib.sha256(text.encode("utf-8")).hexdigest()


def _tokenize_chunks(tokenizer: Any, text: str, n_chunks: int, seq_len: int) -> torch.Tensor:
    need = n_chunks * seq_len
    ids = tokenizer(text[: need * 8], add_special_tokens=False, return_attention_mask=False)["input_ids"]
    if len(ids) < need:
        ids = tokenizer(text, add_special_tokens=False, return_attention_mask=False)["input_ids"]
    if len(ids) < need:
        raise RuntimeError(f"Corpus too short: {len(ids)} < {need} tokens")
    return torch.tensor(ids[:need], dtype=torch.long).reshape(n_chunks, seq_len)


def _sylvester_hadamard(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    h = torch.tensor([[1.0]], device=device, dtype=dtype)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    return h / math.sqrt(float(n))


def _block_hadamard(total_n: int, block_b: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    hb = _sylvester_hadamard(block_b, device, dtype)
    return torch.kron(torch.eye(total_n // block_b, device=device, dtype=dtype), hb)


def _apply_block_rot(w: torch.Tensor, q_mat: torch.Tensor) -> torch.Tensor:
    b = q_mat.shape[0]
    m, k = w.shape
    return (w.reshape(m, k // b, b) @ q_mat.T).reshape(m, k)


def _a16_closest(u16: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Exact GPU-vectorized O(B*16) Extended Hamming [16,11,4] soft decoder."""
    e_val = 2.0 * torch.round(u16 * 0.5)
    o_val = 2.0 * torch.floor(u16 * 0.5) + 1.0
    de = (u16 - e_val) ** 2
    do = (u16 - o_val) ** 2
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
        r_o = torch.nonzero(odd_m, as_tuple=False).squeeze(1)
        c_o = syn[r_o].to(torch.int64)
        out[r_o, c_o] = alt[r_o, c_o]
        out_b[r_o, c_o] ^= 1

    ev_m = (p0 == 0) & (syn != 0)
    if ev_m.any():
        r_e = torch.nonzero(ev_m, as_tuple=False).squeeze(1)
        se = syn[r_e].to(torch.int64)
        jg = torch.arange(16, device=u16.device, dtype=torch.int64).unsqueeze(0)
        pg = jg ^ se.unsqueeze(1)
        pc = fc[r_e] + fc[r_e].gather(1, pg)
        bj = pc.argmin(dim=1)
        bp = bj ^ se
        out[r_e, bj] = alt[r_e, bj]
        out[r_e, bp] = alt[r_e, bp]
        out_b[r_e, bj] ^= 1
        out_b[r_e, bp] ^= 1

    vp0 = out_b.sum(dim=1) & 1
    vs0 = out_b[:, 1::2].sum(dim=1) & 1
    vs1 = (out_b[:, 2::4].sum(dim=1) + out_b[:, 3::4].sum(dim=1)) & 1
    vs2 = (out_b[:, 4:8].sum(dim=1) + out_b[:, 12:16].sum(dim=1)) & 1
    vs3 = out_b[:, 8:16].sum(dim=1) & 1
    bad_syn = (vp0 | vs0 | vs1 | vs2 | vs3) != 0
    return out, int(bad_syn.sum().item())


def _e8_closest(u8: torch.Tensor) -> torch.Tensor:
    f1 = torch.round(u8)
    odd1 = (torch.sum(f1.to(torch.int64), dim=1) & 1) != 0
    if odd1.any():
        diff1 = u8 - f1
        idx1 = diff1.abs().argmax(dim=1)
        step1 = torch.where(diff1.gather(1, idx1.unsqueeze(1)).squeeze(1) >= 0, 1.0, -1.0)
        f1_adj = f1.clone()
        rows1 = torch.nonzero(odd1, as_tuple=False).squeeze(1)
        f1_adj[rows1, idx1[rows1]] += step1[rows1]
        f1 = f1_adj
    u_s = u8 - 0.5
    f2i = torch.round(u_s)
    odd2 = (torch.sum(f2i.to(torch.int64), dim=1) & 1) != 0
    if odd2.any():
        diff2 = u_s - f2i
        idx2 = diff2.abs().argmax(dim=1)
        step2 = torch.where(diff2.gather(1, idx2.unsqueeze(1)).squeeze(1) >= 0, 1.0, -1.0)
        f2_adj = f2i.clone()
        rows2 = torch.nonzero(odd2, as_tuple=False).squeeze(1)
        f2_adj[rows2, idx2[rows2]] += step2[rows2]
        f2i = f2_adj
    f2 = f2i + 0.5
    d1 = ((u8 - f1) ** 2).sum(dim=1)
    d2 = ((u8 - f2) ** 2).sum(dim=1)
    return torch.where((d1 <= d2).unsqueeze(1), f1, f2)


def _quantize_ball_chunk(
    w_chunk: torch.Tensor,
    lattice: str,
    group_size: int,
    q_rot: torch.Tensor | None,
    colscale: torch.Tensor | None,
) -> tuple[torch.Tensor, int]:
    X = w_chunk.clone()
    if colscale is not None:
        X = X * colscale.unsqueeze(0)
    if q_rot is not None:
        X = _apply_block_rot(X, q_rot)

    Xg = X.reshape(-1, group_size)
    bound = BALL_BOUNDS[group_size][lattice]
    r_ball = math.sqrt(bound)
    if lattice == "A16":
        margin = math.sqrt(group_size / 4.0 + 1.5 * (group_size / 16.0))
    elif lattice == "E8":
        margin = math.sqrt(group_size / 4.0 + 1.5 * (group_size / 8.0))
    else:
        margin = math.sqrt(group_size) * 0.5

    g_norm = torch.sqrt((Xg ** 2).sum(dim=1)).clamp_min(1e-30)
    lo = g_norm / (r_ball + margin) * 0.99
    hi = g_norm / max(r_ball - margin, 0.05 * r_ball) * 1.01

    for _ in range(16):
        mid = torch.sqrt(lo * hi)
        scaled = Xg / mid.unsqueeze(1)
        if lattice == "A16":
            Y, _ = _a16_closest(scaled.reshape(-1, 16))
            Y = Y.reshape(-1, group_size)
        elif lattice == "E8":
            Y = _e8_closest(scaled.reshape(-1, 8)).reshape(-1, group_size)
        else:
            Y = torch.round(scaled)
        ok = (Y ** 2).sum(dim=1) <= bound
        hi = torch.where(ok, mid, hi)
        lo = torch.where(ok, lo, mid)

    scaled_f = Xg / hi.unsqueeze(1)
    if lattice == "A16":
        Y_fin, syn_errs = _a16_closest(scaled_f.reshape(-1, 16))
        Y_fin = Y_fin.reshape(-1, group_size)
    elif lattice == "E8":
        Y_fin = _e8_closest(scaled_f.reshape(-1, 8)).reshape(-1, group_size)
        syn_errs = 0
    else:
        Y_fin = torch.round(scaled_f)
        syn_errs = 0

    out_m = (Y_fin ** 2).sum(dim=1) > bound
    if out_m.any():
        xo = scaled_f[out_m]
        s_lo = torch.zeros(xo.shape[0], device=w_chunk.device, dtype=w_chunk.dtype)
        s_hi = torch.ones(xo.shape[0], device=w_chunk.device, dtype=w_chunk.dtype)
        for _ in range(12):
            sm = 0.5 * (s_lo + s_hi)
            if lattice == "A16":
                yt, _ = _a16_closest((xo * sm.unsqueeze(1)).reshape(-1, 16))
                yt = yt.reshape(-1, group_size)
            elif lattice == "E8":
                yt = _e8_closest((xo * sm.unsqueeze(1)).reshape(-1, 8)).reshape(-1, group_size)
            else:
                yt = torch.round(xo * sm.unsqueeze(1))
            good = (yt ** 2).sum(dim=1) <= bound
            s_lo = torch.where(good, sm, s_lo)
            s_hi = torch.where(good, s_hi, sm)
        if lattice == "A16":
            yf2, se2 = _a16_closest((xo * s_lo.unsqueeze(1)).reshape(-1, 16))
            Y_fin[out_m] = yf2.reshape(-1, group_size)
            syn_errs += se2
        elif lattice == "E8":
            Y_fin[out_m] = _e8_closest((xo * s_lo.unsqueeze(1)).reshape(-1, 8)).reshape(-1, group_size)
        else:
            Y_fin[out_m] = torch.round(xo * s_lo.unsqueeze(1))

    scales_out = hi.to(torch.float16).to(torch.float32)
    Wq = (Y_fin * scales_out.unsqueeze(1)).reshape(w_chunk.shape)
    if q_rot is not None:
        Wq = _apply_block_rot(Wq, q_rot.T)
    if colscale is not None:
        Wq = Wq / colscale.unsqueeze(0)
    return Wq, syn_errs


def _quantize_ball_matrix(
    w_f32: torch.Tensor,
    lattice: str,
    group_size: int,
    q_rot: torch.Tensor | None = None,
    colscale: torch.Tensor | None = None,
    col_perm: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    """Row-chunked exact L2-ball quantizer on GPU with optional exact bijective column permutation."""
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
        wq, syn_tot = _quantize_ball_chunk(wf_use, lattice, group_size, q_rot, cs_use)
    else:
        wq_parts: list[torch.Tensor] = []
        syn_tot = 0
        for r0 in range(0, m, QUANT_ROW_CHUNK):
            r1 = min(r0 + QUANT_ROW_CHUNK, m)
            wq_c, syn_c = _quantize_ball_chunk(wf_use[r0:r1], lattice, group_size, q_rot, cs_use)
            wq_parts.append(wq_c)
            syn_tot += syn_c
        wq = torch.cat(wq_parts, dim=0)

    if inv_perm is not None:
        wq = wq[:, inv_perm]
    return wq, syn_tot


def _make_desc_perm(metric: torch.Tensor, local_group: int | None = None) -> torch.Tensor:
    """Return exact bijective permutation sorting columns descending globally or within each local_group."""
    k = metric.numel()
    if local_group is None:
        return torch.argsort(metric, descending=True, stable=True)
    n_g = k // local_group
    parts = []
    for g in range(n_g):
        base = g * local_group
        sub = torch.argsort(metric[base : base + local_group], descending=True, stable=True) + base
        parts.append(sub)
    return torch.cat(parts, dim=0)


def _quantize_direct_4_5bpw_int4(w_f32: torch.Tensor, mode: str) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Direct 4.5000-bpw bounded [-7,7] per-32-weight Extended Hamming A_16 quantizer for sm_86 PTX."""
    X = w_f32
    if mode == "a16_block_h4":
        h4 = _sylvester_hadamard(4, w_f32.device, w_f32.dtype)
        X = (w_f32.reshape(w_f32.shape[0], -1, 4) @ h4.T).reshape(w_f32.shape)
    Xg = X.reshape(-1, 32)
    sc = (Xg.abs().amax(dim=1, keepdim=True) / 7.0).clamp_min(1e-8).to(torch.float16).to(torch.float32)
    u16 = (Xg / sc).reshape(-1, 16)
    e_val = (2.0 * torch.round(u16 * 0.5)).clamp(-6.0, 6.0)
    o_val = (2.0 * torch.floor(u16 * 0.5) + 1.0).clamp(-7.0, 7.0)
    de, do = (u16 - e_val) ** 2, (u16 - o_val) ** 2
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
        jg = torch.arange(16, device=w_f32.device, dtype=torch.int64).unsqueeze(0)
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
    return out.reshape(w_f32.shape).to(torch.int8), sc.squeeze(1).to(torch.float16), errs


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

    arr = np.asarray(kl_list, dtype=np.float64)
    mean_kl = float(arr.mean())
    se_kl = float(arr.std(ddof=1) / math.sqrt(len(arr)))
    return {
        "mean_kl": round(mean_kl, 9),
        "se_kl": round(se_kl, 9),
        "ppl": round(math.exp(total_nll / total_toks), 6),
        "top1_pct": round(100.0 * same_top1 / total_toks, 4),
        "tokens_scored": total_toks,
    }


@torch.no_grad()
def _capture_block8_cov_and_diag(
    model: torch.nn.Module,
    linears: list[tuple[str, torch.nn.Linear]],
    chunks: torch.Tensor,
    device: torch.device,
    capture_vec_layers: set[str] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, np.ndarray]]:
    cov8_sum: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    captured_vecs: dict[str, np.ndarray] = {}
    hooks = []

    for name, mod in linears:
        k8 = mod.in_features // 8
        cov8_sum[name] = torch.zeros((k8, 8, 8), device=device, dtype=torch.float64)
        counts[name] = 0

        def _make_hook(lname: str):
            def _hook(_m, inputs):
                x = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).to(torch.float32)
                if capture_vec_layers and lname in capture_vec_layers and lname not in captured_vecs:
                    captured_vecs[lname] = x[0].cpu().numpy().copy()
                t_toks, k_dim = x.shape
                xb = x.reshape(t_toks, k_dim // 8, 8).permute(1, 0, 2)
                cov8_sum[lname] += torch.bmm(xb.transpose(1, 2), xb).to(torch.float64)
                counts[lname] += int(t_toks)
            return _hook

        hooks.append(mod.register_forward_pre_hook(_make_hook(name)))

    for i in range(chunks.shape[0]):
        model(chunks[i : i + 1].to(device), use_cache=False)

    for hk in hooks:
        hk.remove()

    cov8_cpu: dict[str, torch.Tensor] = {}
    diag_cpu: dict[str, torch.Tensor] = {}
    for name, _ in linears:
        c8 = (cov8_sum[name] / max(counts[name], 1)).to(torch.float32).cpu()
        cov8_cpu[name] = c8
        diag_cpu[name] = torch.diagonal(c8, dim1=1, dim2=2).reshape(-1).clone()
    del cov8_sum
    torch.cuda.empty_cache()
    return cov8_cpu, diag_cpu, captured_vecs


@torch.no_grad()
def _capture_hessian_diag_only(
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


def _quad8_error(dw: torch.Tensor, cov8_gpu: torch.Tensor) -> float:
    m, k = dw.shape
    dw_g = dw.reshape(m, k // 8, 8).permute(1, 0, 2)
    prod = torch.bmm(dw_g, cov8_gpu)
    return float(0.5 * (prod * dw_g).sum().item())


@torch.no_grad()
def _verify_down_proj_offline_fold_bf16_exact(
    model: torch.nn.Module,
    linears: list[tuple[str, torch.nn.Linear]],
    orig_weights_cpu: dict[str, torch.Tensor],
    h_cal_real: dict[str, torch.Tensor],
    sample_chunk: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    """Prove exact BF16 identity when folding down_proj input permutation offline into gate_proj & up_proj rows."""
    inp = sample_chunk.to(device)
    logits_ref_cpu = model(inp, use_cache=False).logits.detach().cpu()

    mod_map = dict(linears)
    n_layers = max(_layer_idx(n) for n, _ in linears) + 1
    for l_idx in range(n_layers):
        dn = f"model.layers.{l_idx}.mlp.down_proj"
        gn = f"model.layers.{l_idx}.mlp.gate_proj"
        un = f"model.layers.{l_idx}.mlp.up_proj"
        perm = _make_desc_perm(h_cal_real[dn]).to(device)
        mod_map[dn].weight.copy_(orig_weights_cpu[dn].to(device)[:, perm])
        mod_map[gn].weight.copy_(orig_weights_cpu[gn].to(device)[perm, :])
        mod_map[un].weight.copy_(orig_weights_cpu[un].to(device)[perm, :])

    logits_fold_cpu = model(inp, use_cache=False).logits.detach().cpu()

    # Restore exact original BF16 weights before CPU comparison
    for l_idx in range(n_layers):
        for suffix in ("down_proj", "gate_proj", "up_proj"):
            nm = f"model.layers.{l_idx}.mlp.{suffix}"
            mod_map[nm].weight.copy_(orig_weights_cpu[nm].to(device))

    del inp
    torch.cuda.empty_cache()

    lr = logits_ref_cpu.to(torch.float32)
    lf = logits_fold_cpu.to(torch.float32)
    diff = (lf - lr).abs()
    max_diff = float(diff.max().item())
    mean_diff = float(diff.mean().item())
    top1_agree = float((lf.argmax(dim=-1) == lr.argmax(dim=-1)).to(torch.float32).mean().item()) * 100.0
    lp_ref = F.log_softmax(lr, dim=-1)
    lp_fold = F.log_softmax(lf, dim=-1)
    kl_fold = float((torch.exp(lp_ref) * (lp_ref - lp_fold)).sum(dim=-1).mean().item())
    del logits_ref_cpu, logits_fold_cpu, lr, lf, diff, lp_ref, lp_fold
    return {
        "layers_folded": n_layers,
        "max_abs_bf16_logit_diff": round(max_diff, 9),
        "mean_abs_bf16_logit_diff": round(mean_diff, 9),
        "kl_fold_vs_ref": round(kl_fold, 12),
        "top1_agreement_pct": round(top1_agree, 4),
        "exact_identity": max_diff == 0.0,
    }


def _eval_model(
    repo_id: str,
    local_dir: str,
    group_size: int,
    wt_val_text: str,
    wt_test_text: str,
    py_test_text: str,
    is_qwen3b: bool,
    run_additivity_decomp: bool,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
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

    # Zero-data random tokens (fixed EVAL_SEED, zero WikiText/Python tokens!)
    rng_gen = torch.Generator().manual_seed(EVAL_SEED)
    vocab_sz = int(model.config.vocab_size)
    rand_chunks = torch.randint(0, vocab_sz, (N_RAND_CHUNKS, SEQ_LEN), generator=rng_gen)
    cov8_rand, h_rand, _ = _capture_block8_cov_and_diag(model, linears, rand_chunks, device)

    cap_layers = {"model.layers.18.mlp.down_proj", "model.layers.0.mlp.gate_proj"} if is_qwen3b else None
    cov8_eval, _, captured_vecs = _capture_block8_cov_and_diag(
        model, linears, wt_eval_chunks[:4], device, capture_vec_layers=cap_layers
    )
    h_cal_real = _capture_hessian_diag_only(model, linears, wt_cal_chunks, device)

    # On Qwen2.5-3B, capture pre-post_attention_layernorm & pre-input_layernorm raw inputs for shared-RMSNorm CUDA bench
    x_pre_postln_vec: np.ndarray | None = None
    postln_gamma_vec: np.ndarray | None = None
    postln_eps = 1e-6
    x_pre_inln_vec: np.ndarray | None = None
    inln_gamma_vec: np.ndarray | None = None
    inln_eps = 1e-6
    if is_qwen3b:
        ln0 = model.model.layers[0].post_attention_layernorm
        postln_gamma_vec = ln0.weight.detach().to(torch.float32).cpu().numpy().copy()
        postln_eps = float(getattr(ln0, "variance_epsilon", 1e-6))
        inln0 = model.model.layers[0].input_layernorm
        inln_gamma_vec = inln0.weight.detach().to(torch.float32).cpu().numpy().copy()
        inln_eps = float(getattr(inln0, "variance_epsilon", 1e-6))
        cap_ln: list[np.ndarray] = []
        cap_inln: list[np.ndarray] = []

        def _ln_hook(_m, inputs):
            if not cap_ln:
                cap_ln.append(inputs[0].detach().reshape(-1, inputs[0].shape[-1])[0].to(torch.float32).cpu().numpy().copy())

        def _inln_hook(_m, inputs):
            if not cap_inln:
                cap_inln.append(inputs[0].detach().reshape(-1, inputs[0].shape[-1])[0].to(torch.float32).cpu().numpy().copy())

        hk_ln = ln0.register_forward_pre_hook(_ln_hook)
        hk_inln = inln0.register_forward_pre_hook(_inln_hook)
        with torch.no_grad():
            model(wt_eval_chunks[:1].to(device), use_cache=False)
        hk_ln.remove()
        hk_inln.remove()
        x_pre_postln_vec = cap_ln[0]
        x_pre_inln_vec = cap_inln[0]

    # Exact BF16 offline down_proj folding verification on Qwen2.5-3B
    bf16_fold_receipt: dict[str, Any] | None = None
    if is_qwen3b:
        bf16_fold_receipt = _verify_down_proj_offline_fold_bf16_exact(
            model, linears, orig_weights_cpu, h_cal_real, wt_eval_chunks[:1], device
        )
        print(f"  BF16 OFFLINE FOLD CHECK: {bf16_fold_receipt}", flush=True)

    # Precompute weight column L2 norms (zero-text, zero forward passes)
    wcol_norm: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for n, _ in linears:
            wf = orig_weights_cpu[n].to(dtype=torch.float32)
            wcol_norm[n] = (wf ** 2).mean(dim=0)
            del wf

    H8 = _sylvester_hadamard(8, device, torch.float32)
    BH2_16 = _block_hadamard(16, 2, device, torch.float32)
    BH4_16 = _block_hadamard(16, 4, device, torch.float32)

    arms_out: dict[str, Any] = {}
    layer_stats: dict[str, dict[str, tuple[float, float, float, float, int, int, int]]] = {}
    q_rand: dict[str, dict[str, float]] = {}
    q_eval: dict[str, dict[str, float]] = {}
    axis_layer_mse: dict[str, float] = {}
    current_arm: dict[str, str] = {}

    def _summarize(
        arm_name: str,
        lattice_tag: str,
        stats_map: dict[str, tuple[float, float, float, float, int, int, int]],
        max_match_db: float = 0.0,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        tot_err = sum(v[0] for v in stats_map.values())
        tot_w = sum(v[1] for v in stats_map.values())
        sm_err = sum(v[2] for v in stats_map.values())
        sm_w = sum(v[3] for v in stats_map.values())
        sm_z = sum(v[4] for v in stats_map.values())
        sm_c = sum(v[5] for v in stats_map.values())
        syn_t = sum(v[6] for v in stats_map.values())
        nb_diffs = _count_non_block_diffs(model, non_block_snaps)
        wt_scores = _score_against_reference(model, wt_eval_chunks, ref_wt_lp, device)
        py_scores = _score_against_reference(model, py_eval_chunks, ref_py_lp, device)
        w_err_db = 10.0 * math.log10(max(tot_err, 1e-30) / max(tot_w, 1e-30))
        e05_db = 10.0 * math.log10(max(sm_err, 1e-30) / max(sm_w, 1e-30))
        z0_pct = 100.0 * float(sm_z) / max(sm_c, 1)
        row: dict[str, Any] = {
            "arm": arm_name,
            "lattice": lattice_tag,
            "group_size": group_size,
            "weight_err_db": round(w_err_db, 6),
            "small_weight_e05_db": round(e05_db, 6),
            "small_weight_zero_pct": round(z0_pct, 4),
            "syndrome_errors": syn_t,
            "non_block_tensors_differ": nb_diffs,
            "max_matched_mse_db_diff": round(max_match_db, 9),
            "wikitext2": wt_scores,
            "python": py_scores,
        }
        if extra:
            row.update(extra)
        print(
            f"  [{arm_name:30s}] dB={w_err_db:8.4f} E05={e05_db:6.3f} z0={z0_pct:5.1f}% "
            f"WikiKL={wt_scores['mean_kl']:.6f} PPL={wt_scores['ppl']:.3f} "
            f"PyKL={py_scores['mean_kl']:.6f} syn={syn_t} nb={nb_diffs}",
            flush=True,
        )
        return row

    def _run_uniform_arm(
        arm_name: str,
        lattice: str,
        q_rot: torch.Tensor | None,
        cal_src: dict[str, torch.Tensor] | None = None,
        perm_mode: str | None = None,
        match_ref: dict[str, float] | None = None,
        record_quad: bool = False,
        record_mse: bool = False,
    ) -> dict[str, Any]:
        st_map: dict[str, tuple[float, float, float, float, int, int, int]] = {}
        qr_map: dict[str, float] = {}
        qe_map: dict[str, float] = {}
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

                cperm: torch.Tensor | None = None
                kd = _proj_kind(n)
                if perm_mode == "wcol":
                    cperm = _make_desc_perm(wcol_norm[n].to(device))
                elif perm_mode == "hrand":
                    cperm = _make_desc_perm(h_rand[n].to(device))
                elif perm_mode == "hreal_global":
                    cperm = _make_desc_perm(h_cal_real[n].to(device))
                elif perm_mode == "hreal_grp":
                    cperm = _make_desc_perm(h_cal_real[n].to(device), local_group=group_size)
                elif perm_mode == "zerocost_down_hrand":
                    if kd == "down_proj":
                        cperm = _make_desc_perm(h_rand[n].to(device))
                elif perm_mode == "zerocost_down_wcol":
                    if kd == "down_proj":
                        cperm = _make_desc_perm(wcol_norm[n].to(device))
                elif perm_mode == "zerocost_down_hreal":
                    if kd == "down_proj":
                        cperm = _make_desc_perm(h_cal_real[n].to(device))
                elif perm_mode == "zerocost_down_hreal_grp":
                    if kd == "down_proj":
                        cperm = _make_desc_perm(h_cal_real[n].to(device), local_group=group_size)
                elif perm_mode == "sharednorm_hreal":
                    l_i = _layer_idx(n)
                    if kd == "down_proj":
                        cperm = _make_desc_perm(h_cal_real[n].to(device))
                    elif kd in ("q_proj", "k_proj", "v_proj"):
                        cperm = _make_desc_perm(h_cal_real[f"model.layers.{l_i}.self_attn.q_proj"].to(device))
                    elif kd in ("gate_proj", "up_proj"):
                        cperm = _make_desc_perm(h_cal_real[f"model.layers.{l_i}.mlp.gate_proj"].to(device))
                elif perm_mode == "sharednorm_hreal_grp":
                    l_i = _layer_idx(n)
                    if kd == "down_proj":
                        cperm = _make_desc_perm(h_cal_real[n].to(device), local_group=group_size)
                    elif kd in ("q_proj", "k_proj", "v_proj"):
                        cperm = _make_desc_perm(h_cal_real[f"model.layers.{l_i}.self_attn.q_proj"].to(device), local_group=group_size)
                    elif kd in ("gate_proj", "up_proj"):
                        cperm = _make_desc_perm(h_cal_real[f"model.layers.{l_i}.mlp.gate_proj"].to(device), local_group=group_size)

                wq, errs = _quantize_ball_matrix(wf, lattice, group_size, q_rot=q_rot, colscale=cs, col_perm=cperm)
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
                sm_cnt = int(sm_mask.sum().item())
                sm_e = float((dw[sm_mask] ** 2).sum().item()) if sm_cnt > 0 else 0.0
                sm_w = float((wf[sm_mask] ** 2).sum().item()) if sm_cnt > 0 else 0.0
                sm_z = int((wq[sm_mask].abs() < 1e-12).sum().item()) if sm_cnt > 0 else 0
                st_map[n] = (cur_mse, w_sq, sm_e, sm_w, sm_z, sm_cnt, errs)
                if record_mse:
                    axis_layer_mse[n] = cur_mse
                if record_quad:
                    c8r = cov8_rand[n].to(device)
                    qr_map[n] = _quad8_error(dw, c8r)
                    del c8r
                    c8e = cov8_eval[n].to(device)
                    qe_map[n] = _quad8_error(dw, c8e)
                    del c8e
                m.weight.copy_(wq.to(torch.bfloat16))
                current_arm[n] = arm_name
                del wf, wq, dw, sm_mask, cs, cperm

        if record_quad:
            layer_stats[arm_name] = st_map
            q_rand[arm_name] = qr_map
            q_eval[arm_name] = qe_map
        return _summarize(arm_name, lattice, st_map, max_match_db=max_mdiff)

    # --- PART A: Base Uncalibrated Arms (ends on A16_axis for delta-routing!) ---
    if is_qwen3b:
        arms_out["E8_H8"] = _run_uniform_arm("E8_H8", "E8", H8)
    arms_out["A16_BlockH4"] = _run_uniform_arm("A16_BlockH4", "A16", BH4_16, record_quad=True)
    arms_out["A16_axis"] = _run_uniform_arm("A16_axis", "A16", None, record_quad=True, record_mse=True)

    # --- PART B: Coupled-Coherent Routing Ablations (P_S69_2b, fast delta-updates!) ---
    all_names = [n for n, _ in linears]
    cands = ("A16_axis", "A16_BlockH4")

    def _build_routes(q_src: dict[str, dict[str, float]]) -> dict[str, dict[str, str]]:
        # 1. Per-matrix greedy
        r_pm = {n: min(cands, key=lambda a: q_src[a][n]) for n in all_names}
        # 2. Coupled subblock ({q,k,v,o} shared, {gate,up,down} shared)
        r_sb: dict[str, str] = {}
        # 3. Layer coherent (all 7 projections in block l share 1 arm!)
        r_lc: dict[str, str] = {}
        # 4. Scale-calibrated relative-ratio layer coherent (sum of Q(m, a)/Q(m, axis))
        r_rr: dict[str, str] = {}
        for l_i in range(n_blocks):
            blk = [n for n in all_names if _layer_idx(n) == l_i]
            attn_m = [n for n in blk if _proj_kind(n) in ("q_proj", "k_proj", "v_proj", "o_proj")]
            mlp_m = [n for n in blk if _proj_kind(n) in ("gate_proj", "up_proj", "down_proj")]
            c_attn = min(cands, key=lambda a: sum(q_src[a][n] for n in attn_m))
            c_mlp = min(cands, key=lambda a: sum(q_src[a][n] for n in mlp_m))
            for n in attn_m:
                r_sb[n] = c_attn
            for n in mlp_m:
                r_sb[n] = c_mlp
            c_blk = min(cands, key=lambda a: sum(q_src[a][n] for n in blk))
            c_rel = min(cands, key=lambda a: sum(q_src[a][n] / max(q_src["A16_axis"][n], 1e-30) for n in blk))
            for n in blk:
                r_lc[n] = c_blk
                r_rr[n] = c_rel
        return {"pm": r_pm, "sb": r_sb, "lc": r_lc, "rr": r_rr}

    rz = _build_routes(q_rand)
    ro = _build_routes(q_eval)
    r_projtype = {n: ("A16_axis" if _proj_kind(n) == "down_proj" else "A16_BlockH4") for n in all_names}

    def _run_routed(arm_name: str, rmap: dict[str, str], is_oracle: bool, oracle_ref: dict[str, str] | None) -> dict[str, Any]:
        with torch.no_grad():
            for n, m in linears:
                tgt = rmap[n]
                if current_arm.get(n) != tgt:
                    q_rot = BH4_16 if tgt == "A16_BlockH4" else None
                    wf = orig_weights_cpu[n].to(device=device, dtype=torch.float32)
                    wq, _ = _quantize_ball_matrix(wf, "A16", group_size, q_rot=q_rot)
                    m.weight.copy_(wq.to(torch.bfloat16))
                    current_arm[n] = tgt
                    del wf, wq
        st_map = {n: layer_stats[rmap[n]][n] for n in all_names}
        cnts: dict[str, int] = {}
        for n in all_names:
            cnts[rmap[n]] = cnts.get(rmap[n], 0) + 1
        ag = sum(1 for n in all_names if rmap[n] == oracle_ref[n]) if oracle_ref is not None else len(all_names)
        return _summarize(
            arm_name,
            "routed_1bit",
            st_map,
            extra={
                "is_oracle_ceiling": is_oracle,
                "chosen_arm_counts": cnts,
                "oracle_agreement_count": ag,
                "oracle_agreement_pct": round(100.0 * ag / len(all_names), 4),
            },
        )

    arms_out["PerMatrix_ZeroCal"] = _run_routed("PerMatrix_ZeroCal", rz["pm"], False, ro["pm"])
    arms_out["PerMatrix_Oracle"] = _run_routed("PerMatrix_Oracle", ro["pm"], True, ro["pm"])
    arms_out["Coupled_Subblock_ZeroCal"] = _run_routed("Coupled_Subblock_ZeroCal", rz["sb"], False, ro["sb"])
    arms_out["Coupled_Subblock_Oracle"] = _run_routed("Coupled_Subblock_Oracle", ro["sb"], True, ro["sb"])
    arms_out["Layer_Coherent_ZeroCal"] = _run_routed("Layer_Coherent_ZeroCal", rz["lc"], False, ro["lc"])
    arms_out["Layer_Coherent_Oracle"] = _run_routed("Layer_Coherent_Oracle", ro["lc"], True, ro["lc"])
    arms_out["Coupled_RelRatio_ZeroCal"] = _run_routed("Coupled_RelRatio_ZeroCal", rz["rr"], False, ro["rr"])
    arms_out["Coupled_RelRatio_Oracle"] = _run_routed("Coupled_RelRatio_Oracle", ro["rr"], True, ro["rr"])
    arms_out["ProjType_Uniform"] = _run_routed("ProjType_Uniform", r_projtype, False, None)

    # --- PART C: Single-Block Additivity Decomposition (P_S69_2a, SmolLM2-360M & Qwen2.5-3B) ---
    additivity_out: dict[str, Any] = {}
    if run_additivity_decomp:
        print(f"  Running {n_blocks}-block single-layer additivity decomposition...", flush=True)
        # Restore full exact BF16 model first
        mod_map = dict(linears)
        with torch.no_grad():
            for n, m in linears:
                m.weight.copy_(orig_weights_cpu[n].to(device))
                current_arm[n] = "BF16"

        decomp_arms = ("A16_axis", "A16_BlockH4", "PerMatrix_Oracle", "Layer_Coherent_Oracle", "ProjType_Uniform")
        per_block_dkl: dict[str, list[float]] = {a: [] for a in decomp_arms}

        for l_i in range(n_blocks):
            blk = [n for n in all_names if _layer_idx(n) == l_i]
            # Quantize this single block ONCE for A16_axis and ONCE for A16_BlockH4 (stored on CPU to preserve 8 GiB VRAM!)
            q_blk: dict[str, dict[str, torch.Tensor]] = {"A16_axis": {}, "A16_BlockH4": {}}
            with torch.no_grad():
                for n in blk:
                    wf = orig_weights_cpu[n].to(device=device, dtype=torch.float32)
                    wq_ax, _ = _quantize_ball_matrix(wf, "A16", group_size, q_rot=None)
                    wq_b4, _ = _quantize_ball_matrix(wf, "A16", group_size, q_rot=BH4_16)
                    q_blk["A16_axis"][n] = wq_ax.to(torch.bfloat16).cpu()
                    q_blk["A16_BlockH4"][n] = wq_b4.to(torch.bfloat16).cpu()
                    del wf, wq_ax, wq_b4
                torch.cuda.empty_cache()

                # Score single-block WikiText-2 KL for uniform axis & BlockH4
                for base_a in ("A16_axis", "A16_BlockH4"):
                    for n in blk:
                        mod_map[n].weight.copy_(q_blk[base_a][n].to(device))
                    sc = _score_against_reference(model, wt_eval_chunks, ref_wt_lp, device)
                    per_block_dkl[base_a].append(sc["mean_kl"])

                # Layer_Coherent_Oracle chooses 1 uniform arm per block -> exact reuse!
                lc_choice = ro["lc"][blk[0]]
                per_block_dkl["Layer_Coherent_Oracle"].append(per_block_dkl[lc_choice][-1])

                # PerMatrix_Oracle: reuse if all 7 chose same arm, else score!
                pm_choices = {ro["pm"][n] for n in blk}
                if len(pm_choices) == 1:
                    only_c = next(iter(pm_choices))
                    per_block_dkl["PerMatrix_Oracle"].append(per_block_dkl[only_c][-1])
                else:
                    for n in blk:
                        mod_map[n].weight.copy_(q_blk[ro["pm"][n]][n].to(device))
                    sc = _score_against_reference(model, wt_eval_chunks, ref_wt_lp, device)
                    per_block_dkl["PerMatrix_Oracle"].append(sc["mean_kl"])

                # ProjType_Uniform (down_proj=axis, rest=BlockH4)
                for n in blk:
                    mod_map[n].weight.copy_(q_blk[r_projtype[n]][n].to(device))
                sc_pt = _score_against_reference(model, wt_eval_chunks, ref_wt_lp, device)
                per_block_dkl["ProjType_Uniform"].append(sc_pt["mean_kl"])

                # Restore block l_i back to exact BF16
                for n in blk:
                    mod_map[n].weight.copy_(orig_weights_cpu[n].to(device))
                del q_blk

        for a_nm in decomp_arms:
            sum_dkl = sum(per_block_dkl[a_nm])
            full_kl = arms_out[a_nm]["wikitext2"]["mean_kl"]
            r_cross = full_kl / max(sum_dkl, 1e-30)
            additivity_out[a_nm] = {
                "full_wikitext2_kl": round(full_kl, 9),
                "sum_single_block_dkl": round(sum_dkl, 9),
                "cross_layer_interference_ratio": round(r_cross, 6),
                "max_single_block_dkl": round(max(per_block_dkl[a_nm]), 9),
                "blocks_scored": n_blocks,
            }
            print(
                f"  [ADDITIVITY {a_nm:22s}] full={full_kl:.6f} sum_dKL={sum_dkl:.6f} R_cross={r_cross:.4f}",
                flush=True,
            )

    # --- PART D: Column-Sorted & Zero-Runtime-Cost Block-H4 Arms (P_S69_1) ---
    arms_out["A16_BlockH4_sort_wcol"] = _run_uniform_arm("A16_BlockH4_sort_wcol", "A16", BH4_16, perm_mode="wcol")
    arms_out["A16_BlockH4_sort_hrand"] = _run_uniform_arm("A16_BlockH4_sort_hrand", "A16", BH4_16, perm_mode="hrand")
    if is_qwen3b:
        arms_out["A16_axis_sort_wcol"] = _run_uniform_arm("A16_axis_sort_wcol", "A16", None, perm_mode="wcol")
        arms_out["A16_BlockH4_sort_wcol_matched"] = _run_uniform_arm(
            "A16_BlockH4_sort_wcol_matched", "A16", BH4_16, perm_mode="wcol", match_ref=axis_layer_mse
        )
        arms_out["A16_BlockH4_zerocost_down_wcol"] = _run_uniform_arm(
            "A16_BlockH4_zerocost_down_wcol", "A16", BH4_16, perm_mode="zerocost_down_wcol"
        )
        arms_out["A16_BlockH4_sort_hrand_matched"] = _run_uniform_arm(
            "A16_BlockH4_sort_hrand_matched", "A16", BH4_16, perm_mode="hrand", match_ref=axis_layer_mse
        )
    arms_out["A16_BlockH4_zerocost_down"] = _run_uniform_arm(
        "A16_BlockH4_zerocost_down", "A16", BH4_16, perm_mode="zerocost_down_hrand"
    )

    # Calibrated floors & column-sorted interventions
    arms_out["hA16_real"] = _run_uniform_arm("hA16_real", "A16", None, cal_src=h_cal_real)
    arms_out["hA16_BlockH2_real"] = _run_uniform_arm("hA16_BlockH2_real", "A16", BH2_16, cal_src=h_cal_real)
    arms_out["hA16_BlockH4_real"] = _run_uniform_arm("hA16_BlockH4_real", "A16", BH4_16, cal_src=h_cal_real)
    arms_out["hA16_BlockH4_sort_grp128_real"] = _run_uniform_arm(
        "hA16_BlockH4_sort_grp128_real", "A16", BH4_16, cal_src=h_cal_real, perm_mode="hreal_grp"
    )
    arms_out["hA16_BlockH4_sort_real"] = _run_uniform_arm(
        "hA16_BlockH4_sort_real", "A16", BH4_16, cal_src=h_cal_real, perm_mode="hreal_global"
    )
    arms_out["hA16_BlockH4_zerocost_down_real"] = _run_uniform_arm(
        "hA16_BlockH4_zerocost_down_real", "A16", BH4_16, cal_src=h_cal_real, perm_mode="zerocost_down_hreal"
    )

    if is_qwen3b:
        arms_out["hA16_BlockH4_zerocost_down_grp128_real"] = _run_uniform_arm(
            "hA16_BlockH4_zerocost_down_grp128_real", "A16", BH4_16, cal_src=h_cal_real, perm_mode="zerocost_down_hreal_grp"
        )
        arms_out["hA16_BlockH4_sharednorm_real"] = _run_uniform_arm(
            "hA16_BlockH4_sharednorm_real", "A16", BH4_16, cal_src=h_cal_real, perm_mode="sharednorm_hreal"
        )
        arms_out["hA16_BlockH4_sharednorm_grp128_real"] = _run_uniform_arm(
            "hA16_BlockH4_sharednorm_grp128_real", "A16", BH4_16, cal_src=h_cal_real, perm_mode="sharednorm_hreal_grp"
        )
        arms_out["hA16_BlockH4_sort_rand"] = _run_uniform_arm(
            "hA16_BlockH4_sort_rand", "A16", BH4_16, cal_src=h_rand, perm_mode="hrand"
        )
        arms_out["hA16_BlockH2_sort_real"] = _run_uniform_arm(
            "hA16_BlockH2_sort_real", "A16", BH2_16, cal_src=h_cal_real, perm_mode="hreal_global"
        )
        arms_out["hA16_axis_sort_real"] = _run_uniform_arm(
            "hA16_axis_sort_real", "A16", None, cal_src=h_cal_real, perm_mode="hreal_global"
        )

    cuda_export_pkg: dict[str, Any] | None = None
    if is_qwen3b:
        down_layers = [(n, orig_weights_cpu[n].clone()) for n, _ in linears if _proj_kind(n) == "down_proj"][:2]
        gate_up_layers = [
            ("model.layers.0.mlp.gate_proj", orig_weights_cpu["model.layers.0.mlp.gate_proj"].clone()),
            ("model.layers.0.mlp.up_proj", orig_weights_cpu["model.layers.0.mlp.up_proj"].clone()),
        ]
        qkv_layers = [
            ("model.layers.0.self_attn.q_proj", orig_weights_cpu["model.layers.0.self_attn.q_proj"].clone()),
            ("model.layers.0.self_attn.k_proj", orig_weights_cpu["model.layers.0.self_attn.k_proj"].clone()),
            ("model.layers.0.self_attn.v_proj", orig_weights_cpu["model.layers.0.self_attn.v_proj"].clone()),
        ]
        cuda_export_pkg = {
            "down_layers": down_layers,
            "down_perm": _make_desc_perm(h_cal_real["model.layers.0.mlp.down_proj"]).cpu().numpy().astype(np.uint16),
            "down_x_real": captured_vecs["model.layers.18.mlp.down_proj"],
            "gate_up_layers": gate_up_layers,
            "gate_up_perm": _make_desc_perm(h_cal_real["model.layers.0.mlp.gate_proj"]).cpu().numpy().astype(np.uint16),
            "x_pre_postln": x_pre_postln_vec,
            "postln_gamma": postln_gamma_vec,
            "postln_eps": postln_eps,
            "qkv_layers": qkv_layers,
            "qkv_perm": _make_desc_perm(h_cal_real["model.layers.0.self_attn.q_proj"]).cpu().numpy().astype(np.uint16),
            "x_pre_inln": x_pre_inln_vec,
            "inln_gamma": inln_gamma_vec,
            "inln_eps": inln_eps,
        }

    out: dict[str, Any] = {
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
        "arms": arms_out,
    }
    if bf16_fold_receipt is not None:
        out["bf16_down_proj_offline_fold_identity"] = bf16_fold_receipt
    if additivity_out:
        out["single_block_additivity_decomposition"] = additivity_out

    del model, ref_wt_lp, ref_py_lp, orig_weights_cpu, cov8_rand, cov8_eval, h_rand, h_cal_real, wcol_norm
    gc.collect()
    torch.cuda.empty_cache()
    return out, cuda_export_pkg


def _run_cuda_s69_bench(
    cu_path: Path,
    pkg: dict[str, Any],
    total_3b_weights: int,
    nvcc: str,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="s69_cuda_") as td:
        td_p = Path(td)
        p_down = td_p / "down2_4096x11008.bin"
        p_mlp = td_p / "gate_up_22016x2048.bin"

        # Export DOWN bundle
        perm_d = torch.from_numpy(pkg["down_perm"].astype(np.int64)).to(device)
        ya_p, sa_p, yb_p, sb_p, ys_p, ss_p = [], [], [], [], [], []
        syn_d_ax = syn_d_b4 = syn_d_so = 0
        with torch.no_grad():
            for _, wc in pkg["down_layers"]:
                wf = wc.to(device=device, dtype=torch.float32)
                ya, sa, ea = _quantize_direct_4_5bpw_int4(wf, "a16_axis")
                yb, sb, eb = _quantize_direct_4_5bpw_int4(wf, "a16_block_h4")
                ys, ss, es = _quantize_direct_4_5bpw_int4(wf[:, perm_d], "a16_block_h4")
                syn_d_ax += ea; syn_d_b4 += eb; syn_d_so += es
                ya_p.append(ya.cpu()); sa_p.append(sa.cpu())
                yb_p.append(yb.cpu()); sb_p.append(sb.cpu())
                ys_p.append(ys.cpu()); ss_p.append(ss.cpu())
                del wf, ya, sa, yb, sb, ys, ss
        ya_c = torch.cat(ya_p, dim=0); sa_c = torch.cat(sa_p, dim=0)
        yb_c = torch.cat(yb_p, dim=0); sb_c = torch.cat(sb_p, dim=0)
        ys_c = torch.cat(ys_p, dim=0); ss_c = torch.cat(ss_p, dim=0)
        M_d, K_d = ya_c.shape
        with open(p_down, "wb") as bf:
            bf.write(b"S69D")
            bf.write(struct.pack("<ii", M_d, K_d))
            bf.write(ya_c.numpy().tobytes()); bf.write(sa_c.view(torch.uint16).numpy().tobytes())
            bf.write(yb_c.numpy().tobytes()); bf.write(sb_c.view(torch.uint16).numpy().tobytes())
            bf.write(ys_c.numpy().tobytes()); bf.write(ss_c.view(torch.uint16).numpy().tobytes())
            bf.write(pkg["down_perm"].tobytes())
            bf.write(pkg["down_x_real"].astype(np.float32).tobytes())

        # Export MLP gate_proj+up_proj stacked bundle (22016x2048) & QKV stacked bundle (2560x2048)
        def _export_sharednorm_bundle(
            out_path: Path, layers_list: list[tuple[str, torch.Tensor]], perm_np: np.ndarray,
            x_pre: np.ndarray, gamma_v: np.ndarray, eps_v: float,
        ) -> dict[str, int]:
            perm_t = torch.from_numpy(perm_np.astype(np.int64)).to(device)
            yu_p, su_p, ym_p, sm_p = [], [], [], []
            syn_u = syn_s = 0
            with torch.no_grad():
                for _, wc in layers_list:
                    wf = wc.to(device=device, dtype=torch.float32)
                    yu, su, eu = _quantize_direct_4_5bpw_int4(wf, "a16_block_h4")
                    ym, sm, em = _quantize_direct_4_5bpw_int4(wf[:, perm_t], "a16_block_h4")
                    syn_u += eu; syn_s += em
                    yu_p.append(yu.cpu()); su_p.append(su.cpu())
                    ym_p.append(ym.cpu()); sm_p.append(sm.cpu())
                    del wf, yu, su, ym, sm
            yu_c = torch.cat(yu_p, dim=0); su_c = torch.cat(su_p, dim=0)
            ym_c = torch.cat(ym_p, dim=0); sm_c = torch.cat(sm_p, dim=0)
            M_s, K_s = yu_c.shape
            with open(out_path, "wb") as bf:
                bf.write(b"S69M")
                bf.write(struct.pack("<iif", M_s, K_s, float(eps_v)))
                bf.write(yu_c.numpy().tobytes()); bf.write(su_c.view(torch.uint16).numpy().tobytes())
                bf.write(ym_c.numpy().tobytes()); bf.write(sm_c.view(torch.uint16).numpy().tobytes())
                bf.write(perm_np.tobytes())
                bf.write(x_pre.astype(np.float32).tobytes())
                bf.write(gamma_v.astype(np.float32).tobytes())
            return {
                "M": M_s, "K": K_s, "weights": M_s * K_s,
                "export_unpermuted_syndrome_errors": syn_u,
                "export_sharednorm_sorted_syndrome_errors": syn_s,
            }

        p_qkv = td_p / "qkv_2560x2048.bin"
        meta_mlp = _export_sharednorm_bundle(
            p_mlp, pkg["gate_up_layers"], pkg["gate_up_perm"],
            pkg["x_pre_postln"], pkg["postln_gamma"], pkg["postln_eps"],
        )
        meta_qkv = _export_sharednorm_bundle(
            p_qkv, pkg["qkv_layers"], pkg["qkv_perm"],
            pkg["x_pre_inln"], pkg["inln_gamma"], pkg["inln_eps"],
        )

        torch.cuda.empty_cache()
        export_meta = {
            "down_proj_stacked2_4096x11008": {
                "M": M_d, "K": K_d, "weights": M_d * K_d,
                "export_a16_axis_syndrome_errors": syn_d_ax,
                "export_a16_block_h4_unpermuted_syndrome_errors": syn_d_b4,
                "export_a16_block_h4_zerocost_folded_syndrome_errors": syn_d_so,
            },
            "gate_up_stacked_22016x2048": meta_mlp,
            "qkv_stacked_2560x2048": meta_qkv,
        }

        bin_path = td_p / "s69_bench"
        ptx_path = td_p / "s69_bench.ptx"
        comp = subprocess.run(
            [nvcc, "-O3", "-std=c++17", "-arch=sm_86", "--ptxas-options=-v", str(cu_path), "-ldl", "-lpthread", "-o", str(bin_path)],
            check=True, capture_output=True, text=True,
        )
        subprocess.run(
            [nvcc, "-O3", "-std=c++17", "-arch=sm_86", "--ptx", str(cu_path), "-o", str(ptx_path)],
            check=True, capture_output=True, text=True,
        )
        ptx_text = ptx_path.read_text(encoding="utf-8")
        ptx_ldl = len(re.findall(r"\bld\.local\b", ptx_text))
        ptx_stl = len(re.findall(r"\bst\.local\b", ptx_text))
        ptx_dp4a = len(re.findall(r"\bdp4a\.s32\.s32\b", ptx_text))
        ptxas_text = comp.stderr + "\n" + comp.stdout
        spills = [
            (int(m.group(1)), int(m.group(2)), int(m.group(3)))
            for m in re.finditer(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", ptxas_text)
        ]
        regs = [int(m.group(1)) for m in re.finditer(r"Used (\d+) registers", ptxas_text)]

        res_down = json.loads(subprocess.run([str(bin_path), str(p_down)], check=True, capture_output=True, text=True).stdout)
        res_mlp = json.loads(subprocess.run([str(bin_path), str(p_mlp)], check=True, capture_output=True, text=True).stdout)
        res_qkv = json.loads(subprocess.run([str(bin_path), str(p_qkv)], check=True, capture_output=True, text=True).stdout)

        # Projected full 3B serving from DOWN 4096x11008
        sf = float(total_3b_weights) / float(res_down["weights"])
        def _proj(d: dict[str, Any]) -> dict[str, float]:
            ms_t = d["ms"] * sf
            g_mj = d["nvml_ina3221_gpu_mj_per_call"] * sf
            c_mj = d["rapl_cpu_pkg_mj_per_call"] * sf
            return {
                "ms_per_token": round(ms_t, 4),
                "tokens_per_s": round(1000.0 / ms_t, 2),
                "nvml_ina3221_gpu_mj_per_token": round(g_mj, 4),
                "rapl_cpu_mj_per_token": round(c_mj, 4),
                "combined_mj_per_token": round(g_mj + c_mj, 4),
            }
        res_down["full_qwen25_3b_projected_serving"] = {
            "a16_axis": _proj(res_down["a16_axis"]),
            "a16_block_h4_unpermuted": _proj(res_down["a16_block_h4_unpermuted"]),
            "a16_block_h4_zerocost_folded": _proj(res_down["a16_block_h4_zerocost_folded"]),
            "zerocost_over_unpermuted_speed_ratio": round(
                res_down["a16_block_h4_zerocost_folded"]["gweights_per_s"]
                / res_down["a16_block_h4_unpermuted"]["gweights_per_s"], 6
            ),
            "zerocost_over_axis_speed_ratio": round(
                res_down["a16_block_h4_zerocost_folded"]["gweights_per_s"]
                / res_down["a16_axis"]["gweights_per_s"], 6
            ),
        }
        print(
            f"  [CUDA DOWN] Triad={res_down['stream_triad_gb_s']:.2f} | "
            f"axis={res_down['a16_axis']['gweights_per_s']:.2f} | "
            f"bh4={res_down['a16_block_h4_unpermuted']['gweights_per_s']:.2f} | "
            f"zerocost_folded={res_down['a16_block_h4_zerocost_folded']['gweights_per_s']:.2f} Gw/s "
            f"(syn={res_down['a16_block_h4_zerocost_folded']['syndrome_errors']} "
            f"i64={res_down['a16_block_h4_zerocost_folded']['int64_bit_exact_mismatches']})",
            flush=True,
        )
        print(
            f"  [CUDA MLP 22016x2048] unp={res_mlp['rmsnorm_unpermuted_plus_blockh4']['ms']:.5f}ms | "
            f"shared_gather={res_mlp['rmsnorm_shared_gather_plus_blockh4']['ms']:.5f}ms "
            f"({res_mlp['rmsnorm_shared_gather_plus_blockh4']['overhead_pct_vs_unpermuted']:+.2f}%) | "
            f"inline_gather={res_mlp['rmsnorm_unpermuted_plus_inline_gather_blockh4']['ms']:.5f}ms "
            f"({res_mlp['rmsnorm_unpermuted_plus_inline_gather_blockh4']['overhead_pct_vs_unpermuted']:+.2f}%)",
            flush=True,
        )
        print(
            f"  [CUDA QKV 2560x2048] unp={res_qkv['rmsnorm_unpermuted_plus_blockh4']['ms']:.5f}ms | "
            f"shared_gather={res_qkv['rmsnorm_shared_gather_plus_blockh4']['ms']:.5f}ms "
            f"({res_qkv['rmsnorm_shared_gather_plus_blockh4']['overhead_pct_vs_unpermuted']:+.2f}%) | "
            f"inline_gather={res_qkv['rmsnorm_unpermuted_plus_inline_gather_blockh4']['ms']:.5f}ms "
            f"({res_qkv['rmsnorm_unpermuted_plus_inline_gather_blockh4']['overhead_pct_vs_unpermuted']:+.2f}%)",
            flush=True,
        )

        bench_out = {
            "ptxas_verification": {
                "kernels_compiled": len(spills),
                "min_registers": min(regs) if regs else 0,
                "max_registers": max(regs) if regs else 0,
                "max_stack_bytes": max((s[0] for s in spills), default=0),
                "max_spill_stores_bytes": max((s[1] for s in spills), default=0),
                "max_spill_loads_bytes": max((s[2] for s in spills), default=0),
                "ptx_ldl_count": ptx_ldl,
                "ptx_stl_count": ptx_stl,
                "ptx_dp4a_count": ptx_dp4a,
            },
            "down_proj_stacked2_4096x11008": res_down,
            "gate_up_stacked_22016x2048": res_mlp,
            "qkv_stacked_2560x2048": res_qkv,
        }
        return export_meta, bench_out


def main() -> int:
    parser = argparse.ArgumentParser(description="Track S69 CEER Cycle-4 Sorted Block-H4 & Coupled Routing Runner")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "rtx3080_s69_empirical_results.json",
    )
    parser.add_argument("--nvcc", type=str, default="/opt/cuda/bin/nvcc")
    args = parser.parse_args()

    torch.manual_seed(EVAL_SEED)
    device = torch.device("cuda:0")
    here = Path(__file__).resolve().parent
    cu_path = here / "sm86_s69_sorted_blockh4_kernel.cu"
    cu_sha256 = hashlib.sha256(cu_path.read_bytes()).hexdigest()

    prereg_raw = (here / "s69_preregistration.json").read_bytes()
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
    rapl_start = _read_rapl_pkg_uj()

    models_out: dict[str, Any] = {}
    models_out["HuggingFaceTB/SmolLM2-360M"], _ = _eval_model(
        "HuggingFaceTB/SmolLM2-360M",
        s49b_pins["HuggingFaceTB/SmolLM2-360M"]["local_dir"],
        32,
        wt_val_text,
        wt_test_text,
        py_test_text,
        is_qwen3b=False,
        run_additivity_decomp=True,
        device=device,
    )
    models_out["Qwen/Qwen2.5-0.5B"], _ = _eval_model(
        "Qwen/Qwen2.5-0.5B",
        s49b_pins["Qwen/Qwen2.5-0.5B"]["local_dir"],
        128,
        wt_val_text,
        wt_test_text,
        py_test_text,
        is_qwen3b=False,
        run_additivity_decomp=False,
        device=device,
    )
    models_out["Qwen/Qwen2.5-1.5B"], _ = _eval_model(
        "Qwen/Qwen2.5-1.5B",
        q15_pin["local_dir"],
        128,
        wt_val_text,
        wt_test_text,
        py_test_text,
        is_qwen3b=False,
        run_additivity_decomp=False,
        device=device,
    )
    models_out["Qwen/Qwen2.5-3B"], cuda_pkg = _eval_model(
        "Qwen/Qwen2.5-3B",
        q3_pin["local_dir"],
        128,
        wt_val_text,
        wt_test_text,
        py_test_text,
        is_qwen3b=True,
        run_additivity_decomp=True,
        device=device,
    )

    if cuda_pkg is None:
        raise RuntimeError("Missing Qwen2.5-3B CUDA export package")

    total_3b_weights = int(models_out["Qwen/Qwen2.5-3B"]["total_block_weights"])
    cuda_export_meta, cuda_res = _run_cuda_s69_bench(cu_path, cuda_pkg, total_3b_weights, args.nvcc, device)
    del cuda_pkg
    gc.collect()
    torch.cuda.empty_cache()

    rapl_end = _read_rapl_pkg_uj()
    smi_post = _query_nvidia_smi()
    ac_post = Path("/sys/class/power_supply/AC0/online").read_text(encoding="utf-8").strip()

    payload = {
        "schema": "aletheia.track_s69.sorted_blockh4_and_coupled_routing.v1",
        "oracle_class": "sampled",
        "measured_live_model_weights": True,
        "preregistration_file_sha256": prereg_file_sha256,
        "preregistration_sha256": prereg_obj["preregistration_sha256"],
        "eval_seed": EVAL_SEED,
        "ac_online": ac_pre == "1" and ac_post == "1",
        "cuda_source_file": cu_path.name,
        "cuda_source_sha256": cu_sha256,
        "corpora": {
            "wikitext2_repo": s49b_pins["wikitext2"]["repo_id"],
            "wikitext2_revision": s49b_pins["wikitext2"]["revision"],
            "wikitext2_val_sha256": s49b_pins["wikitext2"]["val_sha256"],
            "wikitext2_test_sha256": s49b_pins["wikitext2"]["test_sha256"],
            "python_stdlib_modules": list(STDLIB_PY_MODULES),
            "python_stdlib_sha256": py_sha256,
            "seq_len": SEQ_LEN,
            "n_eval_chunks": N_EVAL_CHUNKS,
            "n_rand_router_chunks": N_RAND_CHUNKS,
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
        "cuda_export_meta": cuda_export_meta,
        "cuda_sm86_ptx_sorted_blockh4_bench": cuda_res,
        "total_cpu_rapl_joules": round(max(rapl_end - rapl_start, 0) / 1e6, 3),
        "nvidia_smi_pre": smi_pre,
        "nvidia_smi_post": smi_post,
        "elapsed_seconds": round(time.monotonic() - t_start, 3),
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nWROTE {args.output.name} ({args.output.stat().st_size} bytes, elapsed={payload['elapsed_seconds']}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
