"""Track S68: CEER Cycle-3 Zero-Calibration Per-Layer Adaptive Block-H4 Router,
Calibrated hA16_BlockH4_real Floor, and Bare-Metal sm_86 PTX Fused Block-H4 Serving Pareto.

Offline evaluation using ONLY already-cached real open-weight pretrained models (zero downloads):
  - HuggingFaceTB/SmolLM2-360M (32L, 224 mats, g=32)
  - Qwen/Qwen2.5-0.5B          (24L, 168 mats, g=128)
  - Qwen/Qwen2.5-1.5B          (28L, 196 mats, g=128)
  - Qwen/Qwen2.5-3B            (36L, 252 mats, g=128) + live sm_86 PTX fused Block-H4 kernel bench
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

EVAL_SEED = 2026112468
SEQ_LEN = 512
N_EVAL_CHUNKS = 8      # 4,088 held-out next-token logits per corpus
N_RAND_CHUNKS = 8      # 4,096 zero-data uniform random tokens for router block-8 covariance
N_REAL_CAL_CHUNKS = 12 # 6,144 disjoint WikiText-2 val tokens for P_S68_2 calibrated floors
QUANT_ROW_CHUNK = 512  # VRAM-safe row chunking for 3B on 8 GiB RTX 3080
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
) -> tuple[torch.Tensor, int]:
    """Row-chunked exact L2-ball quantizer on GPU (FP32, < 180 MB peak VRAM)."""
    m = w_f32.shape[0]
    if m <= QUANT_ROW_CHUNK:
        return _quantize_ball_chunk(w_f32, lattice, group_size, q_rot, colscale)
    wq_parts: list[torch.Tensor] = []
    syn_tot = 0
    for r0 in range(0, m, QUANT_ROW_CHUNK):
        r1 = min(r0 + QUANT_ROW_CHUNK, m)
        wq_c, syn_c = _quantize_ball_chunk(w_f32[r0:r1], lattice, group_size, q_rot, colscale)
        wq_parts.append(wq_c)
        syn_tot += syn_c
    return torch.cat(wq_parts, dim=0), syn_tot


def _quantize_direct_4_5bpw_int4(w_f32: torch.Tensor, mode: str) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Direct 4.5000-bpw bounded [-7,7] per-32-weight group quantizer for sm_86 PTX serving."""
    X = w_f32
    if mode == "block_h2":
        h2 = _sylvester_hadamard(2, w_f32.device, w_f32.dtype)
        X = (w_f32.reshape(w_f32.shape[0], -1, 2) @ h2.T).reshape(w_f32.shape)
    elif mode == "a16_block_h4":
        h4 = _sylvester_hadamard(4, w_f32.device, w_f32.dtype)
        X = (w_f32.reshape(w_f32.shape[0], -1, 4) @ h4.T).reshape(w_f32.shape)

    Xg = X.reshape(-1, 32)
    sc = (Xg.abs().amax(dim=1, keepdim=True) / 7.0).clamp_min(1e-8).to(torch.float16).to(torch.float32)
    u = Xg / sc

    if mode in ("a16_axis", "a16_block_h4"):
        u16 = u.reshape(-1, 16)
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

    q = torch.round(u).clamp(-7.0, 7.0).reshape(w_f32.shape).to(torch.int8)
    return q, sc.squeeze(1).to(torch.float16), 0


def _collect_block_linears(model: torch.nn.Module) -> list[tuple[str, torch.nn.Linear]]:
    return [
        (name, mod)
        for name, mod in model.named_modules()
        if isinstance(mod, torch.nn.Linear) and name.startswith("model.layers.")
    ]


def _proj_kind(name: str) -> str:
    return name.rsplit(".", 1)[-1]


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
    capture_vec_layer: str | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], np.ndarray | None]:
    """Capture exact intra-block-8 covariance [K//8, 8, 8] + diagonal Hessian [K] across all tokens."""
    cov8_sum: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    captured_vec: list[np.ndarray] = []
    hooks = []

    for name, mod in linears:
        k8 = mod.in_features // 8
        cov8_sum[name] = torch.zeros((k8, 8, 8), device=device, dtype=torch.float64)
        counts[name] = 0

        def _make_hook(lname: str):
            def _hook(_m, inputs):
                x = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).to(torch.float32)
                if capture_vec_layer is not None and lname == capture_vec_layer and not captured_vec:
                    captured_vec.append(x[0].cpu().numpy().copy())
                t_toks, k_dim = x.shape
                # [K//8, T, 8] -> bmm([K//8, 8, T], [K//8, T, 8]) -> [K//8, 8, 8]
                xb = x.reshape(t_toks, k_dim // 8, 8).permute(1, 0, 2)
                c8 = torch.bmm(xb.transpose(1, 2), xb).to(torch.float64)
                cov8_sum[lname] += c8
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
    vec_out = captured_vec[0] if captured_vec else None
    return cov8_cpu, diag_cpu, vec_out


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
    """Exact intra-block-8 quadratic output error: 0.5 * tr(dW C^(8) dW^T)."""
    m, k = dw.shape
    # [K//8, M, 8] @ [K//8, 8, 8] -> [K//8, M, 8]
    dw_g = dw.reshape(m, k // 8, 8).permute(1, 0, 2)
    prod = torch.bmm(dw_g, cov8_gpu)
    return float(0.5 * (prod * dw_g).sum().item())


def _eval_model(
    repo_id: str,
    local_dir: str,
    group_size: int,
    wt_val_text: str,
    wt_test_text: str,
    py_test_text: str,
    run_calibrated_floors: bool,
    run_rand_cal_blockh4: bool,
    export_cuda_shapes: bool,
    device: torch.device,
) -> tuple[dict[str, Any], list[tuple[str, torch.Tensor]] | None, np.ndarray | None]:
    print(f"\n=== MODEL: {repo_id} (g={group_size}) ===", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(local_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        local_dir, local_files_only=True, dtype=torch.bfloat16
    ).to(device)
    model.eval()

    linears = _collect_block_linears(model)
    non_block_snaps = _snapshot_non_block_tensors(model)
    orig_weights_cpu = {n: m.weight.detach().cpu().clone() for n, m in linears}
    total_block_weights = sum(int(m.weight.numel()) for _, m in linears)

    wt_eval_chunks = _tokenize_chunks(tokenizer, wt_test_text, N_EVAL_CHUNKS, SEQ_LEN)
    py_eval_chunks = _tokenize_chunks(tokenizer, py_test_text, N_EVAL_CHUNKS, SEQ_LEN)

    ref_wt_lp, ppl_base_wt = _collect_reference_logits(model, wt_eval_chunks, device)
    ref_py_lp, ppl_base_py = _collect_reference_logits(model, py_eval_chunks, device)
    print(f"  BASE BF16 PPL: Wiki={ppl_base_wt:.6f} Py={ppl_base_py:.6f} ({len(linears)} mats, {total_block_weights:,} wts)", flush=True)

    # Zero-data random tokens (fixed EVAL_SEED, zero WikiText/Python tokens!)
    rng_gen = torch.Generator().manual_seed(EVAL_SEED)
    vocab_sz = int(model.config.vocab_size)
    rand_chunks = torch.randint(0, vocab_sz, (N_RAND_CHUNKS, SEQ_LEN), generator=rng_gen)
    cov8_rand, h_rand, _ = _capture_block8_cov_and_diag(model, linears, rand_chunks, device)

    # Eval block-8 cov ONLY for explicit oracle ceilings (+ optional layer 18 down_proj real act vec)
    cap_layer = "model.layers.18.mlp.down_proj" if export_cuda_shapes else None
    cov8_eval, _, x_real_vec = _capture_block8_cov_and_diag(
        model, linears, wt_eval_chunks[:4], device, capture_vec_layer=cap_layer
    )

    h_cal_real: dict[str, torch.Tensor] | None = None
    if run_calibrated_floors:
        wt_cal_chunks = _tokenize_chunks(tokenizer, wt_val_text, N_REAL_CAL_CHUNKS, SEQ_LEN)
        h_cal_real = _capture_hessian_diag_only(model, linears, wt_cal_chunks, device)

    # Rotation matrices
    H8 = _sylvester_hadamard(8, device, torch.float32)
    BH2_16 = _block_hadamard(16, 2, device, torch.float32)
    BH4_16 = _block_hadamard(16, 4, device, torch.float32)
    BH8_16 = _block_hadamard(16, 8, device, torch.float32)

    BASE_SPECS: dict[str, tuple[str, torch.Tensor | None]] = {
        "E8_H8": ("E8", H8),
        "A16_BlockH8": ("A16", BH8_16),
        "A16_BlockH4": ("A16", BH4_16),
        "A16_axis": ("A16", None),
    }

    # Precompute weight-only colproxy metric vector per layer (zero forward passes)
    wonly_colproxy: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for n, _ in linears:
            wf = orig_weights_cpu[n].to(dtype=torch.float32)
            wonly_colproxy[n] = 1.0 / (wf ** 2).mean(dim=0).clamp_min(1e-12)
            del wf

    arms_out: dict[str, Any] = {}
    layer_stats: dict[str, dict[str, tuple[float, float, float, float, int, int, int]]] = {}
    q_rand: dict[str, dict[str, float]] = {}
    q_wonly: dict[str, dict[str, float]] = {}
    q_eval: dict[str, dict[str, float]] = {}
    current_arm: dict[str, str] = {}

    def _summarize_from_stats(
        arm_name: str,
        lattice_tag: str,
        stats_map: dict[str, tuple[float, float, float, float, int, int, int]],
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
            "wikitext2": wt_scores,
            "python": py_scores,
        }
        if extra:
            row.update(extra)
        print(
            f"  [{arm_name:24s}] dB={w_err_db:8.4f} E05={e05_db:6.3f} z0={z0_pct:5.1f}% "
            f"WikiKL={wt_scores['mean_kl']:.6f} PPL={wt_scores['ppl']:.3f} "
            f"PyKL={py_scores['mean_kl']:.6f} syn={syn_t} nb={nb_diffs}",
            flush=True,
        )
        return row

    # 1. Run all 4 base uniform arms (ends with model holding A16_axis!)
    for arm_name in ("E8_H8", "A16_BlockH8", "A16_BlockH4", "A16_axis"):
        lat, q_rot = BASE_SPECS[arm_name]
        st_map: dict[str, tuple[float, float, float, float, int, int, int]] = {}
        qr_map: dict[str, float] = {}
        qw_map: dict[str, float] = {}
        qe_map: dict[str, float] = {}
        with torch.no_grad():
            for n, m in linears:
                wf = orig_weights_cpu[n].to(device=device, dtype=torch.float32)
                w_sq = float((wf ** 2).sum().item())
                wq, errs = _quantize_ball_matrix(wf, lat, group_size, q_rot=q_rot)
                dw = wq - wf
                cur_mse = float((dw ** 2).sum().item())
                rms = math.sqrt(w_sq / float(wf.numel()) + 1e-30)
                sm_mask = wf.abs() < (0.05 * rms)
                sm_cnt = int(sm_mask.sum().item())
                sm_e = float((dw[sm_mask] ** 2).sum().item()) if sm_cnt > 0 else 0.0
                sm_w = float((wf[sm_mask] ** 2).sum().item()) if sm_cnt > 0 else 0.0
                sm_z = int((wq[sm_mask].abs() < 1e-12).sum().item()) if sm_cnt > 0 else 0
                st_map[n] = (cur_mse, w_sq, sm_e, sm_w, sm_z, sm_cnt, errs)

                c8r = cov8_rand[n].to(device)
                qr_map[n] = _quad8_error(dw, c8r)
                del c8r
                c8e = cov8_eval[n].to(device)
                qe_map[n] = _quad8_error(dw, c8e)
                del c8e
                wp_v = wonly_colproxy[n].to(device)
                qw_map[n] = float(((dw ** 2).sum(dim=0) * wp_v).sum().item())
                del wp_v

                m.weight.copy_(wq.to(torch.bfloat16))
                current_arm[n] = arm_name
                del wf, wq, dw, sm_mask

        layer_stats[arm_name] = st_map
        q_rand[arm_name] = qr_map
        q_wonly[arm_name] = qw_map
        q_eval[arm_name] = qe_map
        arms_out[arm_name] = _summarize_from_stats(arm_name, lat, st_map)

    # 2. Compute zero-hyperparameter routing maps
    all_names = [n for n, _ in linears]
    cands_1bit = ("A16_axis", "A16_BlockH4")
    cands_2bit = ("A16_axis", "A16_BlockH4", "A16_BlockH8", "E8_H8")

    route_wonly_1 = {n: min(cands_1bit, key=lambda a: q_wonly[a][n]) for n in all_names}
    route_zero_1 = {n: min(cands_1bit, key=lambda a: q_rand[a][n]) for n in all_names}
    route_oracle_1 = {n: min(cands_1bit, key=lambda a: q_eval[a][n]) for n in all_names}
    route_zero_2 = {n: min(cands_2bit, key=lambda a: q_rand[a][n]) for n in all_names}
    route_oracle_2 = {n: min(cands_2bit, key=lambda a: q_eval[a][n]) for n in all_names}

    def _run_routed_arm(
        arm_name: str,
        route_map: dict[str, str],
        bits_per_layer: int,
        oracle_ref_map: dict[str, str] | None,
        is_oracle: bool,
    ) -> dict[str, Any]:
        # Delta-update ONLY layers whose chosen arm differs from currently loaded weight!
        updated = 0
        with torch.no_grad():
            for n, m in linears:
                tgt = route_map[n]
                if current_arm.get(n) != tgt:
                    lat, q_rot = BASE_SPECS[tgt]
                    wf = orig_weights_cpu[n].to(device=device, dtype=torch.float32)
                    wq, _ = _quantize_ball_matrix(wf, lat, group_size, q_rot=q_rot)
                    m.weight.copy_(wq.to(torch.bfloat16))
                    current_arm[n] = tgt
                    updated += 1
                    del wf, wq

        st_map = {n: layer_stats[route_map[n]][n] for n in all_names}
        counts: dict[str, int] = {}
        by_kind_counts: dict[str, dict[str, int]] = {}
        for n in all_names:
            c = route_map[n]
            counts[c] = counts.get(c, 0) + 1
            kd = _proj_kind(n)
            by_kind_counts.setdefault(kd, {})
            by_kind_counts[kd][c] = by_kind_counts[kd].get(c, 0) + 1

        agree_cnt = (
            sum(1 for n in all_names if route_map[n] == oracle_ref_map[n])
            if oracle_ref_map is not None
            else len(all_names)
        )
        extra = {
            "is_oracle_ceiling": is_oracle,
            "bits_per_layer": bits_per_layer,
            "total_routing_bits": bits_per_layer * len(all_names),
            "routing_bpw_overhead": round((bits_per_layer * len(all_names)) / float(total_block_weights), 12),
            "layers_updated_delta": updated,
            "chosen_arm_counts": counts,
            "chosen_by_projection_kind": by_kind_counts,
            "oracle_agreement_count": agree_cnt,
            "oracle_agreement_pct": round(100.0 * agree_cnt / len(all_names), 4),
        }
        return _summarize_from_stats(arm_name, f"routed_{bits_per_layer}bit", st_map, extra=extra)

    arms_out["A16_AdaptWeightOnly_1bit"] = _run_routed_arm(
        "A16_AdaptWeightOnly_1bit", route_wonly_1, 1, route_oracle_1, is_oracle=False
    )
    arms_out["A16_AdaptZeroCal_1bit"] = _run_routed_arm(
        "A16_AdaptZeroCal_1bit", route_zero_1, 1, route_oracle_1, is_oracle=False
    )
    arms_out["Oracle_1bit"] = _run_routed_arm(
        "Oracle_1bit", route_oracle_1, 1, route_oracle_1, is_oracle=True
    )
    arms_out["A16_AdaptZeroCal_2bit"] = _run_routed_arm(
        "A16_AdaptZeroCal_2bit", route_zero_2, 2, route_oracle_2, is_oracle=False
    )
    arms_out["Oracle_2bit"] = _run_routed_arm(
        "Oracle_2bit", route_oracle_2, 2, route_oracle_2, is_oracle=True
    )

    # 3. Calibrated Floors (P_S68_2) on Qwen2.5 scales
    if run_calibrated_floors and h_cal_real is not None:
        cal_specs: list[tuple[str, torch.Tensor | None, dict[str, torch.Tensor]]] = [
            ("hA16_real", None, h_cal_real),
            ("hA16_BlockH2_real", BH2_16, h_cal_real),
            ("hA16_BlockH4_real", BH4_16, h_cal_real),
        ]
        if run_rand_cal_blockh4:
            cal_specs.append(("hA16_BlockH4_rand", BH4_16, h_rand))

        for arm_name, q_rot, h_src in cal_specs:
            st_map = {}
            with torch.no_grad():
                for n, m in linears:
                    wf = orig_weights_cpu[n].to(device=device, dtype=torch.float32)
                    w_sq = float((wf ** 2).sum().item())
                    hv = h_src[n].to(device=device, dtype=torch.float32).clamp_min(1e-12)
                    cs = (hv / hv.mean()).clamp_min(1e-8) ** 0.25
                    wq, errs = _quantize_ball_matrix(wf, "A16", group_size, q_rot=q_rot, colscale=cs)
                    dw = wq - wf
                    cur_mse = float((dw ** 2).sum().item())
                    rms = math.sqrt(w_sq / float(wf.numel()) + 1e-30)
                    sm_mask = wf.abs() < (0.05 * rms)
                    sm_cnt = int(sm_mask.sum().item())
                    sm_e = float((dw[sm_mask] ** 2).sum().item()) if sm_cnt > 0 else 0.0
                    sm_w = float((wf[sm_mask] ** 2).sum().item()) if sm_cnt > 0 else 0.0
                    sm_z = int((wq[sm_mask].abs() < 1e-12).sum().item()) if sm_cnt > 0 else 0
                    st_map[n] = (cur_mse, w_sq, sm_e, sm_w, sm_z, sm_cnt, errs)
                    m.weight.copy_(wq.to(torch.bfloat16))
                    current_arm[n] = arm_name
                    del wf, hv, cs, wq, dw, sm_mask
            arms_out[arm_name] = _summarize_from_stats(arm_name, "A16", st_map)

    down_layers_export: list[tuple[str, torch.Tensor]] | None = None
    if export_cuda_shapes:
        down_layers_export = [(n, orig_weights_cpu[n].clone()) for n, _ in linears if _proj_kind(n) == "down_proj"][:2]

    out = {
        "repo_id": repo_id,
        "group_size": group_size,
        "block_linear_matrices": len(linears),
        "total_block_weights": total_block_weights,
        "non_block_tensors_protected": len(non_block_snaps),
        "reference_bf16": {
            "wikitext2_ppl": round(ppl_base_wt, 6),
            "python_ppl": round(ppl_base_py, 6),
        },
        "arms": arms_out,
    }

    del model, ref_wt_lp, ref_py_lp, orig_weights_cpu, cov8_rand, cov8_eval, h_rand, h_cal_real, wonly_colproxy
    gc.collect()
    torch.cuda.empty_cache()
    return out, down_layers_export, x_real_vec


def _export_cuda_bundle(
    layers_cpu: list[tuple[str, torch.Tensor]],
    x_real_vec: np.ndarray,
    bundle_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    ya_parts, sa_parts = [], []
    yb2_parts, sb2_parts = [], []
    yb4_parts, sb4_parts = [], []
    syn_a = 0
    syn_b4 = 0
    with torch.no_grad():
        for _, w_cpu in layers_cpu:
            wf = w_cpu.to(device=device, dtype=torch.float32)
            ya, sa, ea = _quantize_direct_4_5bpw_int4(wf, "a16_axis")
            yb2, sb2, _ = _quantize_direct_4_5bpw_int4(wf, "block_h2")
            yb4, sb4, eb4 = _quantize_direct_4_5bpw_int4(wf, "a16_block_h4")
            syn_a += ea
            syn_b4 += eb4
            ya_parts.append(ya.cpu())
            sa_parts.append(sa.cpu())
            yb2_parts.append(yb2.cpu())
            sb2_parts.append(sb2.cpu())
            yb4_parts.append(yb4.cpu())
            sb4_parts.append(sb4.cpu())
            del wf, ya, sa, yb2, sb2, yb4, sb4
    torch.cuda.empty_cache()

    ya_cat = torch.cat(ya_parts, dim=0)
    sa_cat = torch.cat(sa_parts, dim=0)
    yb2_cat = torch.cat(yb2_parts, dim=0)
    sb2_cat = torch.cat(sb2_parts, dim=0)
    yb4_cat = torch.cat(yb4_parts, dim=0)
    sb4_cat = torch.cat(sb4_parts, dim=0)
    M_s, K_s = ya_cat.shape

    with open(bundle_path, "wb") as bf:
        bf.write(b"S68B")
        bf.write(struct.pack("<ii", M_s, K_s))
        bf.write(ya_cat.numpy().tobytes())
        bf.write(sa_cat.view(torch.uint16).numpy().tobytes())
        bf.write(yb2_cat.numpy().tobytes())
        bf.write(sb2_cat.view(torch.uint16).numpy().tobytes())
        bf.write(yb4_cat.numpy().tobytes())
        bf.write(sb4_cat.view(torch.uint16).numpy().tobytes())
        bf.write(x_real_vec.astype(np.float32).tobytes())

    return {
        "source_layers": [n for n, _ in layers_cpu],
        "M": M_s,
        "K": K_s,
        "weights": M_s * K_s,
        "export_a16_axis_syndrome_errors": syn_a,
        "export_a16_block_h4_syndrome_errors": syn_b4,
    }


def _run_cuda_bench(
    cu_path: Path,
    bundles: dict[str, Path],
    total_model_block_weights: int,
    nvcc: str,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="s68_cuda_") as td:
        bin_path = Path(td) / "s68_blockh4_bench"
        ptx_path = Path(td) / "s68_blockh4.ptx"
        comp = subprocess.run(
            [
                nvcc, "-O3", "-std=c++17", "-arch=sm_86", "--ptxas-options=-v",
                str(cu_path), "-ldl", "-lpthread", "-o", str(bin_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [
                nvcc, "-O3", "-std=c++17", "-arch=sm_86", "--ptx",
                str(cu_path), "-o", str(ptx_path),
            ],
            check=True,
            capture_output=True,
            text=True,
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

        out_shapes: dict[str, Any] = {}
        for shape_label, bpath in bundles.items():
            smi_before = _query_nvidia_smi()
            rapl_before = _read_rapl_pkg_uj()
            t0 = time.monotonic()
            run = subprocess.run([str(bin_path), str(bpath)], check=True, capture_output=True, text=True)
            elapsed_s = time.monotonic() - t0
            rapl_after = _read_rapl_pkg_uj()
            smi_after = _query_nvidia_smi()
            parsed = json.loads(run.stdout)

            scale_factor = float(total_model_block_weights) / float(parsed["weights"])
            da = parsed["gemv_4_5bpw_a16_axis_dp4a"]
            db2 = parsed["gemv_4_5bpw_block_h2_dp4a"]
            db4 = parsed["gemv_4_5bpw_a16_block_h4_dp4a"]

            def _proj_serving(d: dict[str, Any]) -> dict[str, float]:
                ms_tok = d["ms"] * scale_factor
                gpu_mj_tok = d["nvml_ina3221_gpu_mj_per_call"] * scale_factor
                cpu_mj_tok = d["rapl_cpu_pkg_mj_per_call"] * scale_factor
                return {
                    "ms_per_token": round(ms_tok, 4),
                    "tokens_per_s": round(1000.0 / ms_tok, 2),
                    "nvml_ina3221_gpu_mj_per_token": round(gpu_mj_tok, 4),
                    "rapl_cpu_mj_per_token": round(cpu_mj_tok, 4),
                    "combined_mj_per_token": round(gpu_mj_tok + cpu_mj_tok, 4),
                }

            parsed["full_qwen25_3b_projected_serving"] = {
                "total_block_weights": int(total_model_block_weights),
                "a16_axis": _proj_serving(da),
                "block_h2": _proj_serving(db2),
                "a16_block_h4": _proj_serving(db4),
                "block_h4_over_axis_speed_ratio": round(db4["gweights_per_s"] / da["gweights_per_s"], 6),
                "block_h4_over_bh2_speed_ratio": round(db4["gweights_per_s"] / db2["gweights_per_s"], 6),
                "bench_total_cpu_rapl_j": round(max(rapl_after - rapl_before, 0) / 1e6, 4),
                "bench_elapsed_s": round(elapsed_s, 3),
            }
            parsed["nvidia_smi_before"] = smi_before
            parsed["nvidia_smi_after"] = smi_after
            out_shapes[shape_label] = parsed
            print(
                f"  [CUDA {shape_label}] Triad={parsed['stream_triad_gb_s']:.2f} GB/s | "
                f"A16_axis={da['gweights_per_s']:.2f} Gw/s | BlockH2={db2['gweights_per_s']:.2f} Gw/s | "
                f"A16_BlockH4={db4['gweights_per_s']:.2f} Gw/s (syn={db4['syndrome_errors']} i64_mism={db4['int64_bit_exact_mismatches']})",
                flush=True,
            )

        return {
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
            "shapes": out_shapes,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Track S68 CEER Cycle-3 Adaptive Router & sm_86 PTX BlockH4 Runner")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "rtx3080_s68_empirical_results.json",
    )
    parser.add_argument("--nvcc", type=str, default="/opt/cuda/bin/nvcc")
    args = parser.parse_args()

    torch.manual_seed(EVAL_SEED)
    device = torch.device("cuda:0")
    here = Path(__file__).resolve().parent
    cu_path = here / "sm86_a16_blockh4_kernel.cu"
    cu_sha256 = hashlib.sha256(cu_path.read_bytes()).hexdigest()

    prereg_raw = (here / "s68_preregistration.json").read_bytes()
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

    t_start = time.monotonic()
    smi_pre = _query_nvidia_smi()
    rapl_start = _read_rapl_pkg_uj()

    models_out: dict[str, Any] = {}
    models_out["HuggingFaceTB/SmolLM2-360M"], _, _ = _eval_model(
        "HuggingFaceTB/SmolLM2-360M",
        s49b_pins["HuggingFaceTB/SmolLM2-360M"]["local_dir"],
        32,
        wt_val_text,
        wt_test_text,
        py_test_text,
        run_calibrated_floors=False,
        run_rand_cal_blockh4=False,
        export_cuda_shapes=False,
        device=device,
    )
    models_out["Qwen/Qwen2.5-0.5B"], _, _ = _eval_model(
        "Qwen/Qwen2.5-0.5B",
        s49b_pins["Qwen/Qwen2.5-0.5B"]["local_dir"],
        128,
        wt_val_text,
        wt_test_text,
        py_test_text,
        run_calibrated_floors=True,
        run_rand_cal_blockh4=False,
        export_cuda_shapes=False,
        device=device,
    )
    models_out["Qwen/Qwen2.5-1.5B"], _, _ = _eval_model(
        "Qwen/Qwen2.5-1.5B",
        q15_pin["local_dir"],
        128,
        wt_val_text,
        wt_test_text,
        py_test_text,
        run_calibrated_floors=True,
        run_rand_cal_blockh4=False,
        export_cuda_shapes=False,
        device=device,
    )
    models_out["Qwen/Qwen2.5-3B"], down_layers_2, x_real_vec = _eval_model(
        "Qwen/Qwen2.5-3B",
        q3_pin["local_dir"],
        128,
        wt_val_text,
        wt_test_text,
        py_test_text,
        run_calibrated_floors=True,
        run_rand_cal_blockh4=True,
        export_cuda_shapes=True,
        device=device,
    )

    if down_layers_2 is None or x_real_vec is None:
        raise RuntimeError("Missing Qwen2.5-3B down_proj layers or real activation vector for CUDA bench")

    total_3b_weights = int(models_out["Qwen/Qwen2.5-3B"]["total_block_weights"])
    with tempfile.TemporaryDirectory(prefix="s68_bundle_") as btd:
        p_single = Path(btd) / "qwen3b_down1_2048x11008.bin"
        p_stack2 = Path(btd) / "qwen3b_down2_4096x11008.bin"
        meta_single = _export_cuda_bundle(down_layers_2[:1], x_real_vec, p_single, device)
        meta_stack2 = _export_cuda_bundle(down_layers_2, x_real_vec, p_stack2, device)
        del down_layers_2
        gc.collect()
        torch.cuda.empty_cache()
        cuda_res = _run_cuda_bench(
            cu_path,
            {"down_proj_single_2048x11008": p_single, "down_proj_stacked2_4096x11008": p_stack2},
            total_3b_weights,
            args.nvcc,
        )

    rapl_end = _read_rapl_pkg_uj()
    smi_post = _query_nvidia_smi()
    ac_online = Path("/sys/class/power_supply/AC0/online").read_text(encoding="utf-8").strip()

    payload = {
        "schema": "aletheia.track_s68.zero_cal_adaptive_blockh4_ptx.v1",
        "oracle_class": "sampled",
        "measured_live_model_weights": True,
        "preregistration_file_sha256": prereg_file_sha256,
        "preregistration_sha256": prereg_obj["preregistration_sha256"],
        "eval_seed": EVAL_SEED,
        "ac_online": ac_online == "1",
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
        "cuda_export_meta": {
            "down_proj_single_2048x11008": meta_single,
            "down_proj_stacked2_4096x11008": meta_stack2,
        },
        "cuda_sm86_ptx_blockh4_bench": cuda_res,
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
