"""Track S67: CEER Cycle-2 Live Evaluation of Track S59's A16_BlockH4 / A16_BlockH8 Predictions,
616-Layer Qwen2.5 Re-Entrant Oscillation Diagnosis, and siliquant#85 Proposal 4 Zero-Favoring Decoding.

Offline evaluation using ONLY already-cached real open-weight pretrained models (zero downloads):
  - HuggingFaceTB/SmolLM2-360M (32L, 224 mats, g=32): A16_axis, A16_BlockH4, A16_BlockH8, Proposal 4 (lam=0.5, 2.0)
  - Qwen/Qwen2.5-0.5B (24L, 168 mats, g=128): A16_axis, A16_BlockH2, A16_BlockH4(+matched), A16_BlockH8(+matched), A16_H16 + 168-layer geometry
  - Qwen/Qwen2.5-1.5B (28L, 196 mats, g=128): A16_axis, A16_BlockH2, A16_BlockH4(+matched), A16_BlockH8(+matched), A16_H16 + 196-layer geometry
  - Qwen/Qwen2.5-3B   (36L, 252 mats, g=128): E8_H8, A16_axis, A16_BlockH2, A16_BlockH4(+matched), A16_BlockH8(+matched), A16_H16, Proposal 4 (lam=0.5, 2.0) + 252-layer geometry
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

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

EVAL_SEED = 2026112467
SEQ_LEN = 512
N_EVAL_CHUNKS = 8      # 4,088 held-out next-token logits per corpus
N_CAL_CHUNKS = 12      # 6,144 tokens for random-token activation geometry
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


def _a16_closest(u16: torch.Tensor, coord_wt: torch.Tensor | None = None) -> tuple[torch.Tensor, int]:
    """Exact GPU-vectorized O(B*16) Extended Hamming [16,11,4] soft decoder (optional Proposal-4 coord weights)."""
    e_val = 2.0 * torch.round(u16 * 0.5)
    o_val = 2.0 * torch.floor(u16 * 0.5) + 1.0
    de = (u16 - e_val) ** 2
    do = (u16 - o_val) ** 2
    if coord_wt is not None:
        de = de * coord_wt
        do = do * coord_wt
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
    cw_chunk: torch.Tensor | None,
) -> tuple[torch.Tensor, int]:
    X = w_chunk.clone()
    if q_rot is not None:
        X = _apply_block_rot(X, q_rot)

    Xg = X.reshape(-1, group_size)
    cw16 = cw_chunk.reshape(-1, 16) if cw_chunk is not None else None
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
            Y, _ = _a16_closest(scaled.reshape(-1, 16), cw16)
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
        Y_fin, syn_errs = _a16_closest(scaled_f.reshape(-1, 16), cw16)
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
        cwo = cw_chunk.reshape(-1, group_size)[out_m].reshape(-1, 16) if cw_chunk is not None else None
        s_lo = torch.zeros(xo.shape[0], device=w_chunk.device, dtype=w_chunk.dtype)
        s_hi = torch.ones(xo.shape[0], device=w_chunk.device, dtype=w_chunk.dtype)
        for _ in range(12):
            sm = 0.5 * (s_lo + s_hi)
            if lattice == "A16":
                yt, _ = _a16_closest((xo * sm.unsqueeze(1)).reshape(-1, 16), cwo)
                yt = yt.reshape(-1, group_size)
            elif lattice == "E8":
                yt = _e8_closest((xo * sm.unsqueeze(1)).reshape(-1, 8)).reshape(-1, group_size)
            else:
                yt = torch.round(xo * sm.unsqueeze(1))
            good = (yt ** 2).sum(dim=1) <= bound
            s_lo = torch.where(good, sm, s_lo)
            s_hi = torch.where(good, s_hi, sm)
        if lattice == "A16":
            yf2, se2 = _a16_closest((xo * s_lo.unsqueeze(1)).reshape(-1, 16), cwo)
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
    return Wq, syn_errs


def _quantize_ball_matrix(
    w_f32: torch.Tensor,
    lattice: str,
    group_size: int,
    q_rot: torch.Tensor | None = None,
    zero_lambda: float = 0.0,
) -> tuple[torch.Tensor, int]:
    """Row-chunked exact L2-ball quantizer on GPU (FP32, < 180 MB peak VRAM)."""
    cw_full = None
    if zero_lambda > 0.0 and lattice == "A16" and q_rot is None:
        rms = math.sqrt(float((w_f32 ** 2).mean().item()) + 1e-30)
        cw_full = torch.where(w_f32.abs() < (0.05 * rms), 1.0 + zero_lambda, 1.0)

    m = w_f32.shape[0]
    if m <= QUANT_ROW_CHUNK:
        return _quantize_ball_chunk(w_f32, lattice, group_size, q_rot, cw_full)
    wq_parts: list[torch.Tensor] = []
    syn_tot = 0
    for r0 in range(0, m, QUANT_ROW_CHUNK):
        r1 = min(r0 + QUANT_ROW_CHUNK, m)
        cwc = cw_full[r0:r1] if cw_full is not None else None
        wq_c, syn_c = _quantize_ball_chunk(w_f32[r0:r1], lattice, group_size, q_rot, cwc)
        wq_parts.append(wq_c)
        syn_tot += syn_c
    return torch.cat(wq_parts, dim=0), syn_tot


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
def _capture_activation_geometry(
    model: torch.nn.Module,
    linears: list[tuple[str, torch.nn.Linear]],
    chunks: torch.Tensor,
    device: torch.device,
    capture_sample_acts: bool = False,
) -> tuple[dict[str, dict[str, float]], dict[str, torch.Tensor]]:
    """Capture dimensionless diagonal concentration & block-{2,4,16} DC coherence for every layer."""
    sum_sq: dict[str, torch.Tensor] = {}
    dc_num: dict[str, dict[int, float]] = {}
    dc_den: dict[str, float] = {}
    samples: dict[str, list[torch.Tensor]] = {}
    hooks = []

    for name, mod in linears:
        in_f = mod.in_features
        sum_sq[name] = torch.zeros(in_f, device=device, dtype=torch.float64)
        dc_num[name] = {2: 0.0, 4: 0.0, 16: 0.0}
        dc_den[name] = 0.0
        if capture_sample_acts:
            samples[name] = []

        def _make_hook(lname: str):
            def _hook(_m, inputs):
                x = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).to(torch.float32)
                sum_sq[lname] += (x.to(torch.float64) ** 2).sum(dim=0)
                t_toks, k_dim = x.shape
                dc_den[lname] += float((x ** 2).sum().item())
                for b in (2, 4, 16):
                    xb = x.reshape(t_toks, k_dim // b, b)
                    # Sum of squared block averages * b == energy in block DC subspace
                    dc_num[lname][b] += float(((xb.sum(dim=2) ** 2).sum().item()) / float(b))
                if capture_sample_acts and len(samples[lname]) < 2:
                    samples[lname].append(x[:128].cpu())
            return _hook

        hooks.append(mod.register_forward_pre_hook(_make_hook(name)))

    for i in range(chunks.shape[0]):
        model(chunks[i : i + 1].to(device), use_cache=False)

    for hk in hooks:
        hk.remove()

    geom: dict[str, dict[str, float]] = {}
    for name, mod in linears:
        h = sum_sq[name].clamp_min(0.0)
        h_sum = float(h.sum().item()) + 1e-30
        h_sq_sum = float((h ** 2).sum().item())
        diag_conc = float(mod.in_features) * h_sq_sum / (h_sum * h_sum)
        den = dc_den[name] + 1e-30
        # Normalized DC coherence ratio relative to isotropic baseline (1/b): 1.0 = isotropic, >1 = coherent DC
        geom[name] = {
            "diag_conc": diag_conc,
            "dc_ratio_b2": (dc_num[name][2] / den) * 2.0,
            "dc_ratio_b4": (dc_num[name][4] / den) * 4.0,
            "dc_ratio_b16": (dc_num[name][16] / den) * 16.0,
        }
    sample_out = {n: torch.cat(v, dim=0) for n, v in samples.items() if v}
    torch.cuda.empty_cache()
    return geom, sample_out


def _eval_model(
    repo_id: str,
    local_dir: str,
    group_size: int,
    wt_test_text: str,
    py_test_text: str,
    arm_specs: list[tuple[str, str, int, float, str | None]],
    compute_full_geometry: bool,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate a single cached model cleanly and free VRAM/RAM completely before returning."""
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

    # Precompute rotation matrices
    H8 = _sylvester_hadamard(8, device, torch.float32)
    ROT_A16 = {
        1: None,
        2: _block_hadamard(16, 2, device, torch.float32),
        4: _block_hadamard(16, 4, device, torch.float32),
        8: _block_hadamard(16, 8, device, torch.float32),
        16: _sylvester_hadamard(16, device, torch.float32),
    }

    # Capture activation geometry & eval activation slices for Qwen2.5 structural diagnosis
    rand_geom: dict[str, dict[str, float]] = {}
    eval_geom: dict[str, dict[str, float]] = {}
    eval_acts: dict[str, torch.Tensor] = {}
    weight_geom: dict[str, dict[str, float]] = {}

    if compute_full_geometry:
        rng_gen = torch.Generator().manual_seed(EVAL_SEED)
        vocab_sz = int(model.config.vocab_size)
        rand_chunks = torch.randint(0, vocab_sz, (N_CAL_CHUNKS, SEQ_LEN), generator=rng_gen)
        rand_geom, _ = _capture_activation_geometry(model, linears, rand_chunks[:4], device, capture_sample_acts=False)
        eval_geom, eval_acts = _capture_activation_geometry(model, linears, wt_eval_chunks[:2], device, capture_sample_acts=True)

        with torch.no_grad():
            for n, _ in linears:
                wf = orig_weights_cpu[n].to(device=device, dtype=torch.float32)
                w2 = wf ** 2
                m2 = float(w2.mean().item()) + 1e-30
                m4 = float((w2 ** 2).mean().item())
                kurt = m4 / (m2 * m2) - 3.0
                rms = math.sqrt(m2)
                sm_frac = float((wf.abs() < 0.05 * rms).to(torch.float32).mean().item())
                col_e = w2.mean(dim=0)
                col_out = float(col_e.max().item()) / (float(col_e.mean().item()) + 1e-30)
                b4 = w2.reshape(-1, 4)
                b4_conc = float((b4.amax(dim=1) / (b4.mean(dim=1) + 1e-30)).mean().item())
                weight_geom[n] = {
                    "weight_kurtosis": kurt,
                    "small_weight_frac_05": sm_frac,
                    "col_outlier_ratio": col_out,
                    "block4_energy_conc": b4_conc,
                }
                del wf, w2, b4

    arms_out: dict[str, Any] = {}
    layer_mse_store: dict[str, dict[str, float]] = {}
    layer_quad_store: dict[str, dict[str, float]] = {}

    for arm_name, lattice, block_b, z_lam, match_ref_name in arm_specs:
        q_rot = H8 if (lattice == "E8" and block_b == 8) else (ROT_A16[block_b] if lattice == "A16" else None)
        match_ref = layer_mse_store[match_ref_name] if match_ref_name is not None else None
        rec_mse = arm_name == "A16_axis"
        rec_quad = compute_full_geometry and arm_name in (
            "A16_axis", "A16_BlockH2", "A16_BlockH4", "A16_BlockH4_matched", "A16_BlockH8", "A16_BlockH8_matched", "A16_H16"
        )

        total_err_sq = 0.0
        total_w_sq = 0.0
        sm_err_sq = 0.0
        sm_w_sq = 0.0
        sm_zero_cnt = 0
        sm_tot_cnt = 0
        total_syn_errs = 0
        max_matched_db_diff = 0.0
        cur_mse_map: dict[str, float] = {}
        cur_quad_map: dict[str, float] = {}

        with torch.no_grad():
            for n, m in linears:
                wf = orig_weights_cpu[n].to(device=device, dtype=torch.float32)
                w_sq = float((wf ** 2).sum().item())
                total_w_sq += w_sq

                wq, errs = _quantize_ball_matrix(wf, lattice, group_size, q_rot=q_rot, zero_lambda=z_lam)
                total_syn_errs += errs
                dw = wq - wf
                cur_mse = float((dw ** 2).sum().item())

                if match_ref is not None:
                    tgt = match_ref[n]
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
                rms = math.sqrt(w_sq / float(wf.numel()) + 1e-30)
                sm_mask = wf.abs() < (0.05 * rms)
                sm_cnt = int(sm_mask.sum().item())
                if sm_cnt > 0:
                    sm_err_sq += float((dw[sm_mask] ** 2).sum().item())
                    sm_w_sq += float((wf[sm_mask] ** 2).sum().item())
                    sm_zero_cnt += int((wq[sm_mask].abs() < 1e-12).sum().item())
                    sm_tot_cnt += sm_cnt

                if rec_mse:
                    cur_mse_map[n] = cur_mse
                if rec_quad:
                    xe = eval_acts[n].to(device=device, dtype=torch.float32)
                    cur_quad_map[n] = float(0.5 * ((dw @ xe.T) ** 2).mean().item())
                    del xe

                m.weight.copy_(wq.to(torch.bfloat16))
                del wf, wq, dw, sm_mask

        if rec_mse:
            layer_mse_store[arm_name] = cur_mse_map
        if rec_quad:
            layer_quad_store[arm_name] = cur_quad_map

        non_blk_diffs = _count_non_block_diffs(model, non_block_snaps)
        wt_scores = _score_against_reference(model, wt_eval_chunks, ref_wt_lp, device)
        py_scores = _score_against_reference(model, py_eval_chunks, ref_py_lp, device)
        w_err_db = 10.0 * math.log10(max(total_err_sq, 1e-30) / max(total_w_sq, 1e-30))
        e05_db = 10.0 * math.log10(max(sm_err_sq, 1e-30) / max(sm_w_sq, 1e-30))
        sm_zero_pct = 100.0 * float(sm_zero_cnt) / max(sm_tot_cnt, 1)

        arms_out[arm_name] = {
            "arm": arm_name,
            "lattice": lattice,
            "block_b": block_b,
            "zero_lambda": z_lam,
            "group_size": group_size,
            "weight_err_db": round(w_err_db, 6),
            "small_weight_e05_db": round(e05_db, 6),
            "small_weight_zero_pct": round(sm_zero_pct, 4),
            "syndrome_errors": total_syn_errs,
            "non_block_tensors_differ": non_blk_diffs,
            "max_matched_mse_db_diff": round(max_matched_db_diff, 9),
            "wikitext2": wt_scores,
            "python": py_scores,
        }
        print(
            f"  [{arm_name:22s}] dB={w_err_db:8.4f} E05={e05_db:7.3f}dB z0={sm_zero_pct:5.1f}% "
            f"WikiKL={wt_scores['mean_kl']:.6f} PPL={wt_scores['ppl']:.3f} "
            f"PyKL={py_scores['mean_kl']:.6f} syn={total_syn_errs} nb={non_blk_diffs}",
            flush=True,
        )

    # Summarize structural geometry & per-projection distortion shares
    geom_summary: dict[str, Any] = {}
    if compute_full_geometry:
        kinds = sorted({_proj_kind(n) for n, _ in linears})
        by_kind: dict[str, Any] = {}
        ax_q = layer_quad_store["A16_axis"]
        bh2_q = layer_quad_store["A16_BlockH2"]
        bh4_q = layer_quad_store["A16_BlockH4_matched"]
        bh8_q = layer_quad_store["A16_BlockH8_matched"]
        h16_q = layer_quad_store["A16_H16"]
        tot_ax_q = sum(ax_q.values()) + 1e-30

        for kd in kinds:
            kn = [n for n, _ in linears if _proj_kind(n) == kd]
            by_kind[kd] = {
                "layers": len(kn),
                "mean_weight_kurtosis": round(sum(weight_geom[n]["weight_kurtosis"] for n in kn) / len(kn), 6),
                "mean_small_weight_frac_05": round(sum(weight_geom[n]["small_weight_frac_05"] for n in kn) / len(kn), 6),
                "mean_col_outlier_ratio": round(sum(weight_geom[n]["col_outlier_ratio"] for n in kn) / len(kn), 6),
                "mean_block4_energy_conc": round(sum(weight_geom[n]["block4_energy_conc"] for n in kn) / len(kn), 6),
                "rand_mean_diag_conc": round(sum(rand_geom[n]["diag_conc"] for n in kn) / len(kn), 6),
                "rand_mean_dc_ratio_b2": round(sum(rand_geom[n]["dc_ratio_b2"] for n in kn) / len(kn), 6),
                "rand_mean_dc_ratio_b4": round(sum(rand_geom[n]["dc_ratio_b4"] for n in kn) / len(kn), 6),
                "rand_mean_dc_ratio_b16": round(sum(rand_geom[n]["dc_ratio_b16"] for n in kn) / len(kn), 6),
                "eval_mean_diag_conc": round(sum(eval_geom[n]["diag_conc"] for n in kn) / len(kn), 6),
                "eval_mean_dc_ratio_b2": round(sum(eval_geom[n]["dc_ratio_b2"] for n in kn) / len(kn), 6),
                "eval_mean_dc_ratio_b4": round(sum(eval_geom[n]["dc_ratio_b4"] for n in kn) / len(kn), 6),
                "eval_mean_dc_ratio_b16": round(sum(eval_geom[n]["dc_ratio_b16"] for n in kn) / len(kn), 6),
                "a16_axis_quad_err_share": round(sum(ax_q[n] for n in kn) / tot_ax_q, 6),
                "bh4_beats_axis_count": sum(1 for n in kn if bh4_q[n] < ax_q[n]),
                "bh4_beats_h16_count": sum(1 for n in kn if bh4_q[n] < h16_q[n]),
                "bh4_beats_bh2_count": sum(1 for n in kn if bh4_q[n] < bh2_q[n]),
                "bh4_beats_bh8_count": sum(1 for n in kn if bh4_q[n] < bh8_q[n]),
                "mean_bh4_over_axis_quad_err": round(sum(bh4_q[n] / max(ax_q[n], 1e-30) for n in kn) / len(kn), 6),
                "mean_bh8_over_axis_quad_err": round(sum(bh8_q[n] / max(ax_q[n], 1e-30) for n in kn) / len(kn), 6),
            }

        all_n = [n for n, _ in linears]
        n_l = len(all_n)
        geom_summary = {
            "total_layers": n_l,
            "mean_weight_kurtosis": round(sum(weight_geom[n]["weight_kurtosis"] for n in all_n) / n_l, 6),
            "mean_small_weight_frac_05": round(sum(weight_geom[n]["small_weight_frac_05"] for n in all_n) / n_l, 6),
            "mean_col_outlier_ratio": round(sum(weight_geom[n]["col_outlier_ratio"] for n in all_n) / n_l, 6),
            "mean_block4_energy_conc": round(sum(weight_geom[n]["block4_energy_conc"] for n in all_n) / n_l, 6),
            "rand_mean_diag_conc": round(sum(rand_geom[n]["diag_conc"] for n in all_n) / n_l, 6),
            "rand_mean_dc_ratio_b2": round(sum(rand_geom[n]["dc_ratio_b2"] for n in all_n) / n_l, 6),
            "rand_mean_dc_ratio_b4": round(sum(rand_geom[n]["dc_ratio_b4"] for n in all_n) / n_l, 6),
            "rand_mean_dc_ratio_b16": round(sum(rand_geom[n]["dc_ratio_b16"] for n in all_n) / n_l, 6),
            "eval_mean_diag_conc": round(sum(eval_geom[n]["diag_conc"] for n in all_n) / n_l, 6),
            "eval_mean_dc_ratio_b2": round(sum(eval_geom[n]["dc_ratio_b2"] for n in all_n) / n_l, 6),
            "eval_mean_dc_ratio_b4": round(sum(eval_geom[n]["dc_ratio_b4"] for n in all_n) / n_l, 6),
            "eval_mean_dc_ratio_b16": round(sum(eval_geom[n]["dc_ratio_b16"] for n in all_n) / n_l, 6),
            "bh4_beats_axis_count": sum(1 for n in all_n if bh4_q[n] < ax_q[n]),
            "bh4_beats_h16_count": sum(1 for n in all_n if bh4_q[n] < h16_q[n]),
            "bh4_beats_bh2_count": sum(1 for n in all_n if bh4_q[n] < bh2_q[n]),
            "bh4_beats_bh8_count": sum(1 for n in all_n if bh4_q[n] < bh8_q[n]),
            "by_projection_kind": by_kind,
        }

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
        "structural_geometry": geom_summary,
    }

    del model, ref_wt_lp, ref_py_lp, orig_weights_cpu, eval_acts
    gc.collect()
    torch.cuda.empty_cache()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Track S67 CEER Cycle-2 Block-H4 & Re-Entrant Flip Runner")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "rtx3080_s67_empirical_results.json",
    )
    args = parser.parse_args()

    torch.manual_seed(EVAL_SEED)
    device = torch.device("cuda:0")
    here = Path(__file__).resolve().parent

    prereg_raw = (here / "s67_preregistration.json").read_bytes()
    prereg_file_sha256 = hashlib.sha256(prereg_raw).hexdigest()
    prereg_obj = json.loads(prereg_raw.decode("utf-8"))

    s49b_cache = Path.home() / ".cache" / "huggingface" / "s49b_eval"
    s60_cache = Path.home() / ".cache" / "huggingface" / "s60_eval"
    s63_cache = Path.home() / ".cache" / "huggingface" / "s63_eval"
    s49b_pins = json.loads((s49b_cache / "pins.json").read_text(encoding="utf-8"))
    q15_pin = json.loads((s60_cache / "qwen15b_pin.json").read_text(encoding="utf-8"))
    q3_pin = json.loads((s63_cache / "qwen3b_pin.json").read_text(encoding="utf-8"))

    wt_test_text = (s49b_cache / "wikitext2_test.txt").read_text(encoding="utf-8")
    py_test_text, py_sha256 = _load_python_eval_text()

    t_start = time.monotonic()
    smi_pre = _query_nvidia_smi()
    rapl_start = _read_rapl_pkg_uj()

    smol_arms = [
        ("A16_axis", "A16", 1, 0.0, None),
        ("A16_BlockH4", "A16", 4, 0.0, None),
        ("A16_BlockH8", "A16", 8, 0.0, None),
        ("A16_zero_lam05", "A16", 1, 0.5, None),
        ("A16_zero_lam20", "A16", 1, 2.0, None),
    ]
    qwen_block_arms = [
        ("A16_axis", "A16", 1, 0.0, None),
        ("A16_BlockH2", "A16", 2, 0.0, None),
        ("A16_BlockH4", "A16", 4, 0.0, None),
        ("A16_BlockH4_matched", "A16", 4, 0.0, "A16_axis"),
        ("A16_BlockH8", "A16", 8, 0.0, None),
        ("A16_BlockH8_matched", "A16", 8, 0.0, "A16_axis"),
        ("A16_H16", "A16", 16, 0.0, None),
    ]
    qwen3b_arms = [
        ("E8_H8", "E8", 8, 0.0, None),
        *qwen_block_arms,
        ("A16_zero_lam05", "A16", 1, 0.5, None),
        ("A16_zero_lam20", "A16", 1, 2.0, None),
    ]

    models_out: dict[str, Any] = {}
    models_out["HuggingFaceTB/SmolLM2-360M"] = _eval_model(
        "HuggingFaceTB/SmolLM2-360M",
        s49b_pins["HuggingFaceTB/SmolLM2-360M"]["local_dir"],
        32,
        wt_test_text,
        py_test_text,
        smol_arms,
        compute_full_geometry=False,
        device=device,
    )
    models_out["Qwen/Qwen2.5-0.5B"] = _eval_model(
        "Qwen/Qwen2.5-0.5B",
        s49b_pins["Qwen/Qwen2.5-0.5B"]["local_dir"],
        128,
        wt_test_text,
        py_test_text,
        qwen_block_arms,
        compute_full_geometry=True,
        device=device,
    )
    models_out["Qwen/Qwen2.5-1.5B"] = _eval_model(
        "Qwen/Qwen2.5-1.5B",
        q15_pin["local_dir"],
        128,
        wt_test_text,
        py_test_text,
        qwen_block_arms,
        compute_full_geometry=True,
        device=device,
    )
    models_out["Qwen/Qwen2.5-3B"] = _eval_model(
        "Qwen/Qwen2.5-3B",
        q3_pin["local_dir"],
        128,
        wt_test_text,
        py_test_text,
        qwen3b_arms,
        compute_full_geometry=True,
        device=device,
    )

    rapl_end = _read_rapl_pkg_uj()
    smi_post = _query_nvidia_smi()
    ac_online = Path("/sys/class/power_supply/AC0/online").read_text(encoding="utf-8").strip()

    payload = {
        "schema": "aletheia.track_s67.blockh4_reentrant_ceer.v1",
        "oracle_class": "sampled",
        "measured_live_model_weights": True,
        "preregistration_file_sha256": prereg_file_sha256,
        "preregistration_sha256": prereg_obj["preregistration_sha256"],
        "eval_seed": EVAL_SEED,
        "ac_online": ac_online == "1",
        "corpora": {
            "wikitext2_repo": s49b_pins["wikitext2"]["repo_id"],
            "wikitext2_revision": s49b_pins["wikitext2"]["revision"],
            "wikitext2_val_sha256": s49b_pins["wikitext2"]["val_sha256"],
            "wikitext2_test_sha256": s49b_pins["wikitext2"]["test_sha256"],
            "python_stdlib_modules": list(STDLIB_PY_MODULES),
            "python_stdlib_sha256": py_sha256,
            "seq_len": SEQ_LEN,
            "n_eval_chunks": N_EVAL_CHUNKS,
            "n_cal_chunks": N_CAL_CHUNKS,
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
