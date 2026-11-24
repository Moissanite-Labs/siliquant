"""Track S60: Live 1.5B-Scale Real-Weight Open LLM (Qwen/Qwen2.5-1.5B) RTX 3080 Evaluation.

Measures REAL downloaded pretrained open-weight Qwen/Qwen2.5-1.5B (~1.54B params, 28 layers,
196 block linear matrices, exact 128-wt 72B tiling, measured_live_model_weights=True):
  1. Opposite-sign E_8 vs A_16 Hadamard law at 3x scale jump (natural & <= 1e-5 dB matched MSE)
     over 4,088 held-out WikiText-2 test + 4,088 held-out Python stdlib tokens.
  2. Per-layer Regime I vs Regime II anisotropy census across all 196 real matrices (q/k/v/o/gate/up/down)
     comparing disjoint calibration-split (WikiText-2 validation) vs held-out eval-split routing.
  3. Leakage-free calibration-routed hybrid selectors (A16_hybrid_cal_BlockH2, A16_hybrid_cal_H16,
     A16_hybrid_cal_best3) evaluated on strictly held-out WikiText-2 test & Python code.
  4. P5 random-token calibration decay trajectory (hA16_real vs hA16_rand vs hA16_colproxy & +K10).
  5. Real Qwen2.5-1.5B down_proj (1536x8960 single & 6144x8960 4-layer stacked) CUDA sm_86 lossless
     18B/72B transcode + direct 4.50-bpw serving + live nvidia-smi GPU & intel-rapl:0 CPU mJ/token.
"""

from __future__ import annotations

import argparse
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

EVAL_SEED = 2026112460
SEQ_LEN = 512
N_EVAL_CHUNKS = 8      # 4,096 held-out tokens (4,088 next-token logits) per corpus
N_CAL_CHUNKS = 12      # 6,144 disjoint calibration tokens (WikiText-2 validation)
K_KEEP_DOWN = 10       # siliquant#77 top-10 kept down_proj columns

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


def _block_h2_matrix(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    h2 = _sylvester_hadamard(2, device, dtype)
    return torch.kron(torch.eye(n // 2, device=device, dtype=dtype), h2)


def _apply_block_rot(w: torch.Tensor, q_mat: torch.Tensor) -> torch.Tensor:
    b = q_mat.shape[0]
    m, k = w.shape
    return (w.reshape(m, k // b, b) @ q_mat.T).reshape(m, k)


def _a16_closest(u16: torch.Tensor) -> tuple[torch.Tensor, int]:
    """GPU-vectorized compact O(B*16) Extended Hamming [16,11,4] Construction-A soft decoder."""
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


def _quantize_ball_matrix(
    w_f32: torch.Tensor,
    lattice: str,
    group_size: int,
    q_rot: torch.Tensor | None = None,
    colscale: torch.Tensor | None = None,
    keep_cols: torch.Tensor | None = None,
    return_int_and_scales: bool = False,
) -> tuple[torch.Tensor, int] | tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]:
    """Exact siliquant L2-ball quantizer on GPU (FP32)."""
    X = w_f32.clone()
    if colscale is not None:
        X = X * colscale.unsqueeze(0)
    kept_vals = None
    if keep_cols is not None:
        kept_vals = w_f32[:, keep_cols].clone()
        X[:, keep_cols] = 0.0
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
        s_lo = torch.zeros(xo.shape[0], device=w_f32.device, dtype=w_f32.dtype)
        s_hi = torch.ones(xo.shape[0], device=w_f32.device, dtype=w_f32.dtype)
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
    Wq = (Y_fin * scales_out.unsqueeze(1)).reshape(w_f32.shape)
    if q_rot is not None:
        Wq = _apply_block_rot(Wq, q_rot.T)
    if colscale is not None:
        Wq = Wq / colscale.unsqueeze(0)
    if keep_cols is not None and kept_vals is not None:
        Wq[:, keep_cols] = kept_vals

    if return_int_and_scales:
        return Wq, syn_errs, Y_fin.reshape(w_f32.shape), scales_out
    return Wq, syn_errs


def _quantize_direct_4_5bpw_int4(w_f32: torch.Tensor, block_h2: bool) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Direct 4.50-bpw bounded [-7,7] per-32-weight group quantizer for GPU SoA __dp4a serving."""
    X = w_f32
    if block_h2:
        h2 = _block_h2_matrix(2, w_f32.device, w_f32.dtype)
        X = (w_f32.reshape(w_f32.shape[0], -1, 2) @ h2.T).reshape(w_f32.shape)
    Xg = X.reshape(-1, 32)
    sc = (Xg.abs().amax(dim=1, keepdim=True) / 7.0).clamp_min(1e-8).to(torch.float16).to(torch.float32)
    u = Xg / sc
    if not block_h2:
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
        logits = model(inp).logits[:, :-1, :].to(torch.float32)
        targets = inp[:, 1:]
        log_p = F.log_softmax(logits, dim=-1)
        nll = F.nll_loss(log_p.reshape(-1, log_p.shape[-1]), targets.reshape(-1), reduction="sum")
        total_nll += float(nll.item())
        total_toks += int(targets.numel())
        ref_log_probs.append(log_p.cpu())
        del inp, logits, log_p
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
        logits = model(inp).logits[:, :-1, :].to(torch.float32)
        targets = inp[:, 1:]
        log_q = F.log_softmax(logits, dim=-1)
        log_p = ref_log_probs[i].to(device)
        p_ref = torch.exp(log_p)
        tok_kl = (p_ref * (log_p - log_q)).sum(dim=-1).reshape(-1)
        kl_list.extend(tok_kl.cpu().tolist())
        nll = F.nll_loss(log_q.reshape(-1, log_q.shape[-1]), targets.reshape(-1), reduction="sum")
        total_nll += float(nll.item())
        total_toks += int(targets.numel())
        same_top1 += int((log_q.argmax(dim=-1) == log_p.argmax(dim=-1)).sum().item())
        del inp, logits, log_q, log_p, p_ref, tok_kl

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
def _capture_hessian_diag(
    model: torch.nn.Module,
    linears: list[tuple[str, torch.nn.Linear]],
    chunks: torch.Tensor,
    device: torch.device,
    capture_sample_layers: set[str] | None = None,
    max_sample_chunks: int = 2,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    sum_sq: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    samples: dict[str, list[torch.Tensor]] = {}
    hooks = []

    for name, mod in linears:
        in_f = mod.in_features
        sum_sq[name] = torch.zeros(in_f, device=device, dtype=torch.float64)
        counts[name] = 0
        if capture_sample_layers and name in capture_sample_layers:
            samples[name] = []

        def _make_hook(lname: str):
            def _hook(_m, inputs):
                x = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
                sum_sq[lname] += (x.to(torch.float64) ** 2).sum(dim=0)
                counts[lname] += int(x.shape[0])
                if lname in samples and len(samples[lname]) < max_sample_chunks:
                    samples[lname].append(x[:128].to(torch.float32).cpu())
            return _hook

        hooks.append(mod.register_forward_pre_hook(_make_hook(name)))

    for i in range(chunks.shape[0]):
        model(chunks[i : i + 1].to(device))

    for hk in hooks:
        hk.remove()

    h_diag = {n: (sum_sq[n] / max(counts[n], 1)).to(torch.float32) for n in sum_sq}
    sample_out = {n: torch.cat(v, dim=0) for n, v in samples.items() if v}
    return h_diag, sample_out


def _spearman_np(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    den = math.sqrt(float((rx * rx).sum() * (ry * ry).sum()))
    return float((rx * ry).sum() / den) if den > 0.0 else 0.0


def _export_cuda_bundle(
    layers_cpu: list[tuple[str, torch.Tensor]],
    x_real_vec: np.ndarray,
    bundle_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    y18_parts, s18_parts = [], []
    y72_parts, s72_parts = [], []
    y45a_parts, s45a_parts = [], []
    y45b_parts, s45b_parts = [], []
    export_syn = 0
    with torch.no_grad():
        for _, w_cpu in layers_cpu:
            wf = w_cpu.to(device=device, dtype=torch.float32)
            _, syn18, y18_f, s18_f = _quantize_ball_matrix(wf, "A16", 32, return_int_and_scales=True)
            _, syn72, y72_f, s72_f = _quantize_ball_matrix(wf, "A16", 128, return_int_and_scales=True)
            y45_a, s45_a, syn45 = _quantize_direct_4_5bpw_int4(wf, block_h2=False)
            y45_b, s45_b, _ = _quantize_direct_4_5bpw_int4(wf, block_h2=True)
            export_syn += syn18 + syn72 + syn45
            y18_parts.append(y18_f.to(torch.int8).cpu())
            s18_parts.append(s18_f.cpu())
            y72_parts.append(y72_f.to(torch.int8).cpu())
            s72_parts.append(s72_f.to(torch.float16).cpu())
            y45a_parts.append(y45_a.cpu())
            s45a_parts.append(s45_a.cpu())
            y45b_parts.append(y45_b.cpu())
            s45b_parts.append(s45_b.cpu())
            del wf, y18_f, s18_f, y72_f, s72_f, y45_a, s45_a, y45_b, s45_b
    torch.cuda.empty_cache()

    y18_cat = torch.cat(y18_parts, dim=0)
    s18_cat = torch.cat(s18_parts, dim=0)
    y72_cat = torch.cat(y72_parts, dim=0)
    s72_cat = torch.cat(s72_parts, dim=0)
    y45a_cat = torch.cat(y45a_parts, dim=0)
    s45a_cat = torch.cat(s45a_parts, dim=0)
    y45b_cat = torch.cat(y45b_parts, dim=0)
    s45b_cat = torch.cat(s45b_parts, dim=0)
    M_s, K_s = y18_cat.shape
    top18 = float(s18_cat.max().item()) * 1.001
    codes18 = torch.clamp(torch.floor(torch.log2(top18 / s18_cat.clamp_min(1e-12)) * 16.0), 0, 255).to(torch.uint8)

    with open(bundle_path, "wb") as bf:
        bf.write(b"S60B")
        bf.write(struct.pack("<iif", M_s, K_s, top18))
        bf.write(y18_cat.numpy().tobytes())
        bf.write(codes18.numpy().tobytes())
        bf.write(y72_cat.numpy().tobytes())
        bf.write(s72_cat.view(torch.uint16).numpy().tobytes())
        bf.write(y45a_cat.numpy().tobytes())
        bf.write(s45a_cat.view(torch.uint16).numpy().tobytes())
        bf.write(y45b_cat.numpy().tobytes())
        bf.write(s45b_cat.view(torch.uint16).numpy().tobytes())
        bf.write(x_real_vec.astype(np.float32).tobytes())

    return {
        "source_layers": [n for n, _ in layers_cpu],
        "M": M_s,
        "K": K_s,
        "weights": M_s * K_s,
        "export_syndrome_errors": export_syn,
    }


def _run_cuda_bench(
    cu_path: Path,
    bundles: dict[str, Path],
    total_model_block_weights: int,
    nvcc: str,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="s60_cuda_") as td:
        bin_path = Path(td) / "s60_qwen15b_bench"
        comp = subprocess.run(
            [nvcc, "-O3", "-std=c++17", "-arch=sm_86", "--ptxas-options=-v", str(cu_path), "-o", str(bin_path)],
            check=True,
            capture_output=True,
            text=True,
        )
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
            full_tok_ms = parsed["gemv_direct_4_5bpw_a16_preencoded_dp4a"]["ms"] * scale_factor
            gpu_w = float(smi_after["power_draw_w"])
            gpu_j_per_tok = gpu_w * (full_tok_ms / 1000.0)
            cpu_rapl_j_per_tok = (
                parsed["gemv_direct_4_5bpw_a16_preencoded_dp4a"]["rapl_cpu_pkg_mj_per_call"] * scale_factor
            ) / 1000.0
            parsed["full_qwen25_1_5b_projected_serving"] = {
                "total_block_weights": int(total_model_block_weights),
                "direct_4_5bpw_ms_per_token": round(full_tok_ms, 4),
                "direct_4_5bpw_tokens_per_s": round(1000.0 / full_tok_ms, 2),
                "gpu_power_draw_w_sampled": smi_after["power_draw_w"],
                "estimated_gpu_j_per_token": round(gpu_j_per_tok, 6),
                "measured_cpu_rapl_j_per_token": round(cpu_rapl_j_per_tok, 6),
                "bench_total_cpu_rapl_j": round(max(rapl_after - rapl_before, 0) / 1e6, 4),
                "bench_elapsed_s": round(elapsed_s, 3),
            }
            parsed["nvidia_smi_before"] = smi_before
            parsed["nvidia_smi_after"] = smi_after
            out_shapes[shape_label] = parsed

        return {
            "ptxas_verification": {
                "kernels_compiled": len(spills),
                "min_registers": min(regs) if regs else 0,
                "max_registers": max(regs) if regs else 0,
                "max_stack_bytes": max((s[0] for s in spills), default=0),
                "max_spill_stores_bytes": max((s[1] for s in spills), default=0),
                "max_spill_loads_bytes": max((s[2] for s in spills), default=0),
            },
            "shapes": out_shapes,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Track S60 Live Qwen2.5-1.5B RTX 3080 Scale Law Runner")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "rtx3080_qwen15b_empirical_results.json",
    )
    parser.add_argument("--nvcc", type=str, default="/opt/cuda/bin/nvcc")
    args = parser.parse_args()

    torch.manual_seed(EVAL_SEED)
    device = torch.device("cuda:0")
    here = Path(__file__).resolve().parent
    cu_path = here / "rtx3080_qwen15b_kernels.cu"
    cu_sha256 = hashlib.sha256(cu_path.read_bytes()).hexdigest()

    s49b_cache = Path.home() / ".cache" / "huggingface" / "s49b_eval"
    s60_cache = Path.home() / ".cache" / "huggingface" / "s60_eval"
    corpora_pins = json.loads((s49b_cache / "pins.json").read_text(encoding="utf-8"))
    q15_pin = json.loads((s60_cache / "qwen15b_pin.json").read_text(encoding="utf-8"))

    wt_val_text = (s49b_cache / "wikitext2_val.txt").read_text(encoding="utf-8")
    wt_test_text = (s49b_cache / "wikitext2_test.txt").read_text(encoding="utf-8")
    py_test_text, py_sha256 = _load_python_eval_text()

    t_start = time.monotonic()
    smi_pre = _query_nvidia_smi()
    rapl_start = _read_rapl_pkg_uj()

    local_dir = q15_pin["local_dir"]
    tokenizer = AutoTokenizer.from_pretrained(local_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        local_dir, local_files_only=True, dtype=torch.bfloat16
    ).to(device)
    model.eval()

    linears = _collect_block_linears(model)
    non_block_snaps = _snapshot_non_block_tensors(model)
    orig_weights_cpu = {n: m.weight.detach().cpu().clone() for n, m in linears}
    total_block_weights = sum(int(m.weight.numel()) for _, m in linears)
    group_size = 128

    wt_eval_chunks = _tokenize_chunks(tokenizer, wt_test_text, N_EVAL_CHUNKS, SEQ_LEN)
    py_eval_chunks = _tokenize_chunks(tokenizer, py_test_text, N_EVAL_CHUNKS, SEQ_LEN)
    wt_cal_chunks = _tokenize_chunks(tokenizer, wt_val_text, N_CAL_CHUNKS, SEQ_LEN)
    rng_gen = torch.Generator().manual_seed(EVAL_SEED)
    vocab_sz = int(model.config.vocab_size)
    rand_cal_chunks = torch.randint(0, vocab_sz, (N_CAL_CHUNKS, SEQ_LEN), generator=rng_gen)

    # Reference BF16 logits & PPL
    ref_wt_lp, ppl_base_wt = _collect_reference_logits(model, wt_eval_chunks, device)
    ref_py_lp, ppl_base_py = _collect_reference_logits(model, py_eval_chunks, device)

    # Capture calibration Hessians + disjoint calibration activation slices (for leakage-free routing!)
    all_layer_names = {n for n, _ in linears}
    h_cal_real, cal_acts = _capture_hessian_diag(
        model, linears, wt_cal_chunks, device, capture_sample_layers=all_layer_names, max_sample_chunks=2
    )
    h_cal_rand, _ = _capture_hessian_diag(model, linears, rand_cal_chunks, device)
    h_eval_true, eval_acts = _capture_hessian_diag(
        model, linears, wt_eval_chunks[:2], device, capture_sample_layers=all_layer_names, max_sample_chunks=2
    )

    H16 = _sylvester_hadamard(16, device, torch.float32)
    H8 = _sylvester_hadamard(8, device, torch.float32)
    BH2_16 = _block_h2_matrix(16, device, torch.float32)

    def run_arm(
        arm_name: str,
        lattice: str,
        q_rot: torch.Tensor | None = None,
        per_layer_rot: dict[str, torch.Tensor | None] | None = None,
        cal_mode: str | None = None,
        keep_down_k: int = 0,
        match_mse_ref: dict[str, float] | None = None,
        record_layer_mse: bool = False,
        record_layer_quad: bool = False,
    ) -> tuple[dict[str, Any], dict[str, float], dict[str, float], dict[str, float]]:
        total_err_sq = 0.0
        total_w_sq = 0.0
        total_syn_errs = 0
        max_matched_db_diff = 0.0
        layer_mse_map: dict[str, float] = {}
        layer_quad_cal: dict[str, float] = {}
        layer_quad_eval: dict[str, float] = {}

        with torch.no_grad():
            for n, m in linears:
                wf = orig_weights_cpu[n].to(device=device, dtype=torch.float32)
                w_sq = float((wf ** 2).sum().item())
                total_w_sq += w_sq

                cs_vec = None
                if cal_mode == "real":
                    h = h_cal_real[n].clamp_min(1e-12)
                    cs_vec = (h / h.mean()).clamp_min(1e-8) ** 0.25
                elif cal_mode == "rand":
                    h = h_cal_rand[n].clamp_min(1e-12)
                    cs_vec = (h / h.mean()).clamp_min(1e-8) ** 0.25
                elif cal_mode == "colproxy":
                    h = 1.0 / (wf ** 2).mean(dim=0).clamp_min(1e-12)
                    cs_vec = (h / h.mean()).clamp_min(1e-8) ** 0.25

                keep_idx = None
                if keep_down_k > 0 and _proj_kind(n) == "down_proj":
                    col_e = (wf ** 2).sum(dim=0)
                    keep_idx = torch.topk(col_e, k=keep_down_k).indices.sort().values

                rot_use = per_layer_rot[n] if per_layer_rot is not None else q_rot
                wq, errs = _quantize_ball_matrix(
                    wf, lattice, group_size, q_rot=rot_use, colscale=cs_vec, keep_cols=keep_idx
                )
                total_syn_errs += errs
                dw = wq - wf
                cur_mse = float((dw ** 2).sum().item())

                if match_mse_ref is not None:
                    tgt = match_mse_ref[n]
                    if cur_mse > 0.0:
                        alpha = math.sqrt(tgt / cur_mse)
                        dw = dw * alpha
                        wq = wf + dw
                        matched_mse = float((dw ** 2).sum().item())
                        db_diff = abs(
                            10.0 * math.log10(max(matched_mse, 1e-30) / max(w_sq, 1e-30))
                            - 10.0 * math.log10(max(tgt, 1e-30) / max(w_sq, 1e-30))
                        )
                        if db_diff > max_matched_db_diff:
                            max_matched_db_diff = db_diff
                        cur_mse = matched_mse

                total_err_sq += cur_mse
                if record_layer_mse:
                    layer_mse_map[n] = cur_mse
                if record_layer_quad:
                    xc = cal_acts[n].to(device=device, dtype=torch.float32)
                    layer_quad_cal[n] = float(0.5 * ((dw @ xc.T) ** 2).mean().item())
                    xe = eval_acts[n].to(device=device, dtype=torch.float32)
                    layer_quad_eval[n] = float(0.5 * ((dw @ xe.T) ** 2).mean().item())
                    del xc, xe

                m.weight.copy_(wq.to(torch.bfloat16))
                del wf, wq, dw

        non_blk_diffs = _count_non_block_diffs(model, non_block_snaps)
        wt_scores = _score_against_reference(model, wt_eval_chunks, ref_wt_lp, device)
        py_scores = _score_against_reference(model, py_eval_chunks, ref_py_lp, device)
        w_err_db = 10.0 * math.log10(max(total_err_sq, 1e-30) / max(total_w_sq, 1e-30))

        row = {
            "arm": arm_name,
            "lattice": lattice,
            "group_size": group_size,
            "weight_err_db": round(w_err_db, 6),
            "syndrome_errors": total_syn_errs,
            "non_block_tensors_differ": non_blk_diffs,
            "max_matched_mse_db_diff": round(max_matched_db_diff, 9),
            "wikitext2": wt_scores,
            "python": py_scores,
        }
        print(
            f"  [{arm_name:24s}] dB={row['weight_err_db']:8.4f} "
            f"WikiKL={wt_scores['mean_kl']:.6f} PPL={wt_scores['ppl']:.3f} "
            f"PyKL={py_scores['mean_kl']:.6f} syn={total_syn_errs} nb={non_blk_diffs}",
            flush=True,
        )
        return row, layer_mse_map, layer_quad_cal, layer_quad_eval

    arms_out: dict[str, Any] = {}
    # 1. Natural 4.5000 bpw arms (recording natural cal/eval quad errors for leakage-free routing!)
    arms_out["Z_plain"], _, _, _ = run_arm("Z_plain", "Z")
    arms_out["E8_axis"], e8_mse, _, _ = run_arm("E8_axis", "E8", record_layer_mse=True)
    arms_out["E8_H8"], _, _, _ = run_arm("E8_H8", "E8", q_rot=H8)
    arms_out["A16_axis"], a16_mse, qcal_ax, qeval_ax = run_arm(
        "A16_axis", "A16", record_layer_mse=True, record_layer_quad=True
    )
    arms_out["A16_BlockH2"], _, qcal_bh2_nat, qeval_bh2_nat = run_arm(
        "A16_BlockH2", "A16", q_rot=BH2_16, record_layer_quad=True
    )
    arms_out["A16_H16"], _, qcal_h16_nat, qeval_h16_nat = run_arm(
        "A16_H16", "A16", q_rot=H16, record_layer_quad=True
    )

    # 2. Matched Layerwise Weight-MSE arms (<= 1e-5 dB per layer!)
    arms_out["E8_H8_matched"], _, _, _ = run_arm("E8_H8_matched", "E8", q_rot=H8, match_mse_ref=e8_mse)
    arms_out["A16_BlockH2_matched"], _, qcal_bh2_m, qeval_bh2_m = run_arm(
        "A16_BlockH2_matched", "A16", q_rot=BH2_16, match_mse_ref=a16_mse, record_layer_quad=True
    )
    arms_out["A16_H16_matched"], _, qcal_h16_m, qeval_h16_m = run_arm(
        "A16_H16_matched", "A16", q_rot=H16, match_mse_ref=a16_mse, record_layer_quad=True
    )

    # 3. Leakage-Free Calibration-Routed Per-Layer Hybrid Selectors (strictly from cal_acts!)
    rot_hyb_bh2: dict[str, torch.Tensor | None] = {}
    rot_hyb_h16: dict[str, torch.Tensor | None] = {}
    rot_hyb_best3: dict[str, torch.Tensor | None] = {}
    cal_reg1_cnt = 0
    cal_reg2_cnt = 0
    best3_counts = {"A16_axis": 0, "A16_BlockH2": 0, "A16_H16": 0}
    for n, _ in linears:
        is_reg2_cal = qcal_h16_nat[n] < qcal_ax[n]
        if is_reg2_cal:
            cal_reg2_cnt += 1
            rot_hyb_bh2[n] = BH2_16
            rot_hyb_h16[n] = H16
        else:
            cal_reg1_cnt += 1
            rot_hyb_bh2[n] = None
            rot_hyb_h16[n] = None
        # Best-of-3 strictly on calibration split
        cands = [(qcal_ax[n], "A16_axis", None), (qcal_bh2_nat[n], "A16_BlockH2", BH2_16), (qcal_h16_nat[n], "A16_H16", H16)]
        cands.sort(key=lambda x: x[0])
        best3_counts[cands[0][1]] += 1
        rot_hyb_best3[n] = cands[0][2]

    arms_out["A16_hybrid_cal_BlockH2"], _, _, _ = run_arm(
        "A16_hybrid_cal_BlockH2", "A16", per_layer_rot=rot_hyb_bh2
    )
    arms_out["A16_hybrid_cal_H16"], _, _, _ = run_arm(
        "A16_hybrid_cal_H16", "A16", per_layer_rot=rot_hyb_h16
    )
    arms_out["A16_hybrid_cal_best3"], _, _, _ = run_arm(
        "A16_hybrid_cal_best3", "A16", per_layer_rot=rot_hyb_best3
    )

    # 4. Calibration & +K10 Shootout
    arms_out["A16_axis_K10"], _, _, _ = run_arm("A16_axis_K10", "A16", keep_down_k=K_KEEP_DOWN)
    arms_out["hA16_colproxy"], _, _, _ = run_arm("hA16_colproxy", "A16", cal_mode="colproxy")
    arms_out["hA16_rand"], _, _, _ = run_arm("hA16_rand", "A16", cal_mode="rand")
    arms_out["hA16_real"], _, _, _ = run_arm("hA16_real", "A16", cal_mode="real")

    # Per-layer anisotropy census across all 196 real matrices (matched-MSE & natural + cal->eval accuracy!)
    kinds = sorted({_proj_kind(n) for n, _ in linears})
    per_kind_diag: dict[str, Any] = {}
    tot_reg1_eval = 0
    tot_reg2_eval = 0
    tot_bh2_beats_h16_eval = 0
    tot_cal_eval_agree = 0
    for kd in kinds:
        k_names = [n for n, _ in linears if _proj_kind(n) == kd]
        reg1 = sum(1 for n in k_names if qeval_ax[n] < qeval_h16_m[n])
        reg2 = len(k_names) - reg1
        bh2_win = sum(1 for n in k_names if qeval_bh2_m[n] < qeval_h16_m[n])
        agree = sum(
            1 for n in k_names if (qcal_ax[n] < qcal_h16_nat[n]) == (qeval_ax[n] < qeval_h16_nat[n])
        )
        r_h16 = [qeval_h16_m[n] / max(qeval_ax[n], 1e-30) for n in k_names]
        r_bh2 = [qeval_bh2_m[n] / max(qeval_ax[n], 1e-30) for n in k_names]
        tot_reg1_eval += reg1
        tot_reg2_eval += reg2
        tot_bh2_beats_h16_eval += bh2_win
        tot_cal_eval_agree += agree
        per_kind_diag[kd] = {
            "layers": len(k_names),
            "regime1_axis_beats_h16": reg1,
            "regime2_h16_beats_axis": reg2,
            "block_h2_beats_h16": bh2_win,
            "cal_to_eval_regime_agree": agree,
            "mean_h16_over_axis_quad_err": round(sum(r_h16) / len(r_h16), 6),
            "mean_bh2_over_axis_quad_err": round(sum(r_bh2) / len(r_bh2), 6),
        }

    # Layerwise Spearman rho
    rho_real, rho_rand, rho_proxy = [], [], []
    for n, _ in linears:
        ht = h_eval_true[n].cpu().numpy()
        hr = h_cal_real[n].cpu().numpy()
        hrn = h_cal_rand[n].cpu().numpy()
        wf = orig_weights_cpu[n].to(torch.float32)
        hp = (1.0 / (wf ** 2).mean(dim=0).clamp_min(1e-12)).numpy()
        rho_real.append(_spearman_np(hr, ht))
        rho_rand.append(_spearman_np(hrn, ht))
        rho_proxy.append(_spearman_np(hp, ht))

    def _ret(plain_kl: float, real_kl: float, test_kl: float) -> float:
        denom = plain_kl - real_kl
        return round((plain_kl - test_kl) / denom, 6) if abs(denom) > 1e-12 else 0.0

    k_wt_pl = arms_out["A16_axis"]["wikitext2"]["mean_kl"]
    k_wt_re = arms_out["hA16_real"]["wikitext2"]["mean_kl"]
    k_wt_rn = arms_out["hA16_rand"]["wikitext2"]["mean_kl"]
    k_wt_px = arms_out["hA16_colproxy"]["wikitext2"]["mean_kl"]
    k_py_pl = arms_out["A16_axis"]["python"]["mean_kl"]
    k_py_re = arms_out["hA16_real"]["python"]["mean_kl"]
    k_py_rn = arms_out["hA16_rand"]["python"]["mean_kl"]
    k_py_px = arms_out["hA16_colproxy"]["python"]["mean_kl"]

    # Free model VRAM before exporting real Qwen2.5-1.5B down_proj tensors & running CUDA bench
    x_real_vec = eval_acts["model.layers.14.mlp.down_proj"][0].to(torch.float32).cpu().numpy()
    down_layers_4 = [(n, orig_weights_cpu[n]) for n, _ in linears if _proj_kind(n) == "down_proj"][:4]
    down_layer_1 = down_layers_4[:1]
    del model, ref_wt_lp, ref_py_lp, cal_acts, eval_acts
    torch.cuda.empty_cache()

    with tempfile.TemporaryDirectory(prefix="s60_bundle_") as btd:
        p_single = Path(btd) / "qwen15b_down1_1536x8960.bin"
        p_stack4 = Path(btd) / "qwen15b_down4_6144x8960.bin"
        meta_single = _export_cuda_bundle(down_layer_1, x_real_vec, p_single, device)
        meta_stack4 = _export_cuda_bundle(down_layers_4, x_real_vec, p_stack4, device)
        cuda_res = _run_cuda_bench(
            cu_path,
            {"down_proj_single_1536x8960": p_single, "down_proj_stacked4_6144x8960": p_stack4},
            total_block_weights,
            args.nvcc,
        )

    rapl_end = _read_rapl_pkg_uj()
    smi_post = _query_nvidia_smi()
    ac_online = Path("/sys/class/power_supply/AC0/online").read_text(encoding="utf-8").strip()

    st_file = q15_pin["safetensors"][0]
    cfg = q15_pin["config"]
    payload = {
        "schema": "aletheia.track_s60.qwen15b_scale_law.v1",
        "oracle_class": "sampled",
        "measured_live_model_weights": True,
        "eval_seed": EVAL_SEED,
        "ac_online": ac_online == "1",
        "cuda_source_file": cu_path.name,
        "cuda_source_sha256": cu_sha256,
        "corpora": {
            "wikitext2_repo": corpora_pins["wikitext2"]["repo_id"],
            "wikitext2_revision": corpora_pins["wikitext2"]["revision"],
            "wikitext2_val_sha256": corpora_pins["wikitext2"]["val_sha256"],
            "wikitext2_test_sha256": corpora_pins["wikitext2"]["test_sha256"],
            "python_stdlib_modules": list(STDLIB_PY_MODULES),
            "python_stdlib_sha256": py_sha256,
            "seq_len": SEQ_LEN,
            "n_eval_chunks": N_EVAL_CHUNKS,
            "n_cal_chunks": N_CAL_CHUNKS,
        },
        "model": {
            "repo_id": q15_pin["repo_id"],
            "revision": q15_pin["revision"],
            "model_safetensors_name": st_file["name"],
            "model_safetensors_bytes": st_file["bytes"],
            "model_safetensors_sha256": st_file["sha256"],
            "tokenizer_json_sha256": q15_pin["tokenizer_json_sha256"],
            "num_hidden_layers": cfg["num_hidden_layers"],
            "hidden_size": cfg["hidden_size"],
            "intermediate_size": cfg["intermediate_size"],
            "num_attention_heads": cfg["num_attention_heads"],
            "num_key_value_heads": cfg["num_key_value_heads"],
            "tie_word_embeddings": cfg["tie_word_embeddings"],
            "block_linear_matrices": len(linears),
            "total_block_weights": total_block_weights,
            "group_size": group_size,
            "bpw_exact": "9/2",
            "reference_bf16": {
                "wikitext2_ppl": round(ppl_base_wt, 6),
                "python_ppl": round(ppl_base_py, 6),
                "non_block_tensors_protected": len(non_block_snaps),
            },
            "arms": arms_out,
            "hybrid_selector_routing": {
                "cal_predicted_regime1_axis": cal_reg1_cnt,
                "cal_predicted_regime2_rotated": cal_reg2_cnt,
                "cal_best3_counts": best3_counts,
                "cal_to_eval_regime_agree_count": tot_cal_eval_agree,
                "cal_to_eval_regime_accuracy": round(tot_cal_eval_agree / len(linears), 6),
            },
            "calibration_summary": {
                "wikitext2_p5_rand_retention": _ret(k_wt_pl, k_wt_re, k_wt_rn),
                "wikitext2_colproxy_retention": _ret(k_wt_pl, k_wt_re, k_wt_px),
                "python_p5_rand_retention": _ret(k_py_pl, k_py_re, k_py_rn),
                "python_colproxy_retention": _ret(k_py_pl, k_py_re, k_py_px),
                "mean_layerwise_spearman_real_vs_eval": round(sum(rho_real) / len(rho_real), 6),
                "mean_layerwise_spearman_rand_vs_eval": round(sum(rho_rand) / len(rho_rand), 6),
                "mean_layerwise_spearman_colproxy_vs_eval": round(sum(rho_proxy) / len(rho_proxy), 6),
            },
            "per_layer_anisotropy_diagnostic": {
                "total_layers_scored": len(linears),
                "regime1_axis_beats_h16_count": tot_reg1_eval,
                "regime2_h16_beats_axis_count": tot_reg2_eval,
                "block_h2_beats_h16_count": tot_bh2_beats_h16_eval,
                "by_projection_kind": per_kind_diag,
            },
            "cuda_export_meta": {
                "down_proj_single_1536x8960": meta_single,
                "down_proj_stacked4_6144x8960": meta_stack4,
            },
        },
        "cuda_real_weight_transcode_and_serving": cuda_res,
        "total_cpu_rapl_joules": round(max(rapl_end - rapl_start, 0) / 1e6, 3),
        "nvidia_smi_pre": smi_pre,
        "nvidia_smi_post": smi_post,
        "elapsed_seconds": round(time.monotonic() - t_start, 3),
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"WROTE {args.output.name} ({args.output.stat().st_size} bytes, elapsed={payload['elapsed_seconds']}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
