#!/usr/bin/env python3
"""Live RTX 3080 (sm_86) 110W AC CUDA/PyTorch Empirical Runner.

Executes three live Ampere sm_86 empirical stages (run under `mlos gpu-lease --mode train`
and stamped via `mlos stamp-run --zone B`):
  1. AC 110W P0 Roofline Calibration (bare-metal CUDA STREAM Triad / FP32 / INT8 dp4a +
     PyTorch BF16 / FP16 / TF32 / FP32 / INT8 4096x4096 Tensor-Core GEMM + nvidia-smi telemetry).
  2. Bare-Metal 4-Way Fused 4.50-bpw Quantized GEMV Shootout at 4096x4096 & 4096x14336
     (A16_axis, A16_axis_dp4a_live_repair, A16_axis_preencoded_dp4a, A16_H16,
     Block_H2_fp32act, Block_H2_dp4a_int8, E8_WHT8) with ptxas 0-spill verification,
     0 syndrome errors, and CPU FP64 parity.
  3. 5-Seed Anisotropy & Calibration Anisotropic-Hessian & Calibration Shootout on RTX 3080 GPU
     (synthetic heavy-tailed Transformer layers with strict train/held-out split;
     explicitly tagged `sampled`, `measured_live_model_weights: false`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


SEEDS = [2026112401, 2026112402, 2026112403, 2026112404, 2026112405]


def _query_nvidia_smi() -> dict[str, Any]:
    fields = [
        "name",
        "driver_version",
        "pstate",
        "power.draw",
        "power.limit",
        "clocks.current.sm",
        "clocks.current.memory",
        "memory.used",
        "memory.total",
        "temperature.gpu",
        "clocks_event_reasons.active",
    ]
    proc = subprocess.run(
        ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    parts = [p.strip() for p in proc.stdout.strip().splitlines()[0].split(",")]
    out = dict(zip(fields, parts))
    # Also check active graphics/compute processes
    proc_apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    out["compute_apps_lines"] = [ln.strip() for ln in proc_apps.stdout.splitlines() if ln.strip()]
    return out


def _run_cuda_kernels(cu_path: Path, nvcc_bin: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="s49_cuda_") as td:
        bin_path = Path(td) / "rtx3080_s49_kernels"
        comp = subprocess.run(
            [
                nvcc_bin,
                "-O3",
                "-std=c++17",
                "-arch=sm_86",
                "--ptxas-options=-v",
                str(cu_path),
                "-o",
                str(bin_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        ptxas_log = comp.stderr + "\n" + comp.stdout
        spills = re.findall(r"(\d+) bytes spill stores, (\d+) bytes spill loads", ptxas_log)
        max_spill_stores = max((int(a) for a, _ in spills), default=-1)
        max_spill_loads = max((int(b) for _, b in spills), default=-1)
        reg_counts = [int(x) for x in re.findall(r"Used (\d+) registers", ptxas_log)]

        run = subprocess.run([str(bin_path)], capture_output=True, text=True, check=True)
        data = json.loads(run.stdout)
        data["ptxas_verification"] = {
            "kernels_compiled": len(spills),
            "max_spill_stores_bytes": max_spill_stores,
            "max_spill_loads_bytes": max_spill_loads,
            "min_registers": min(reg_counts) if reg_counts else 0,
            "max_registers": max(reg_counts) if reg_counts else 0,
        }
        return data


def _bench_pytorch_gemm(n: int = 4096, warmup: int = 10, iters: int = 40) -> dict[str, Any]:
    device = torch.device("cuda:0")
    results: dict[str, Any] = {"matrix_dim": n, "warmup": warmup, "iters": iters, "dtypes": {}}
    flops = 2.0 * float(n) * float(n) * float(n)

    def _time_fn(fn) -> tuple[float, dict[str, Any]]:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize(device)
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(iters):
            fn()
        t1.record()
        torch.cuda.synchronize(device)
        smi_under_load = _query_nvidia_smi()
        ms = t0.elapsed_time(t1) / float(iters)
        return ms, smi_under_load

    gen = torch.Generator(device=device).manual_seed(20261124)

    # 1. BF16 Tensor Core GEMM
    a_bf = torch.randn((n, n), device=device, dtype=torch.bfloat16, generator=gen)
    b_bf = torch.randn((n, n), device=device, dtype=torch.bfloat16, generator=gen)
    ms, smi_bf = _time_fn(lambda: torch.matmul(a_bf, b_bf))
    results["dtypes"]["bf16"] = {
        "ms": round(ms, 6),
        "tflops": round(flops / (ms * 1e9), 4),
        "power_draw_w": smi_bf["power.draw"],
        "sm_clock_mhz": smi_bf["clocks.current.sm"],
        "mem_clock_mhz": smi_bf["clocks.current.memory"],
        "temp_c": smi_bf["temperature.gpu"],
        "throttle_hex": smi_bf["clocks_event_reasons.active"],
    }
    del a_bf, b_bf

    # 2. FP16 Tensor Core GEMM
    a_f16 = torch.randn((n, n), device=device, dtype=torch.float16, generator=gen)
    b_f16 = torch.randn((n, n), device=device, dtype=torch.float16, generator=gen)
    ms, smi_f16 = _time_fn(lambda: torch.matmul(a_f16, b_f16))
    results["dtypes"]["fp16"] = {
        "ms": round(ms, 6),
        "tflops": round(flops / (ms * 1e9), 4),
        "power_draw_w": smi_f16["power.draw"],
        "sm_clock_mhz": smi_f16["clocks.current.sm"],
        "temp_c": smi_f16["temperature.gpu"],
        "throttle_hex": smi_f16["clocks_event_reasons.active"],
    }
    del a_f16, b_f16

    # 3. TF32 Tensor Core GEMM
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    a_f32 = torch.randn((n, n), device=device, dtype=torch.float32, generator=gen)
    b_f32 = torch.randn((n, n), device=device, dtype=torch.float32, generator=gen)
    ms, smi_tf32 = _time_fn(lambda: torch.matmul(a_f32, b_f32))
    results["dtypes"]["tf32"] = {
        "ms": round(ms, 6),
        "tflops": round(flops / (ms * 1e9), 4),
        "power_draw_w": smi_tf32["power.draw"],
        "sm_clock_mhz": smi_tf32["clocks.current.sm"],
        "temp_c": smi_tf32["temperature.gpu"],
        "throttle_hex": smi_tf32["clocks_event_reasons.active"],
    }

    # 4. Pure FP32 (TF32 disabled)
    torch.backends.cuda.matmul.allow_tf32 = False
    ms, smi_f32 = _time_fn(lambda: torch.matmul(a_f32, b_f32))
    results["dtypes"]["fp32_strict"] = {
        "ms": round(ms, 6),
        "tflops": round(flops / (ms * 1e9), 4),
        "power_draw_w": smi_f32["power.draw"],
        "sm_clock_mhz": smi_f32["clocks.current.sm"],
        "temp_c": smi_f32["temperature.gpu"],
        "throttle_hex": smi_f32["clocks_event_reasons.active"],
    }
    torch.backends.cuda.matmul.allow_tf32 = prev_tf32
    del a_f32, b_f32

    # 5. INT8 Tensor Core GEMM via torch._int_mm (IMMA sm_86)
    if hasattr(torch, "_int_mm"):
        a_i8 = torch.randint(-64, 64, (n, n), device=device, dtype=torch.int8, generator=gen)
        b_i8 = torch.randint(-64, 64, (n, n), device=device, dtype=torch.int8, generator=gen)
        ms, smi_i8 = _time_fn(lambda: torch._int_mm(a_i8, b_i8))
        results["dtypes"]["int8_tensor_core"] = {
            "ms": round(ms, 6),
            "tops": round(flops / (ms * 1e9), 4),
            "power_draw_w": smi_i8["power.draw"],
            "sm_clock_mhz": smi_i8["clocks.current.sm"],
            "temp_c": smi_i8["temperature.gpu"],
            "throttle_hex": smi_i8["clocks_event_reasons.active"],
        }
        del a_i8, b_i8

    torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------------
# Stage 3: GPU-Vectorized Exact Lattice Quantizers & Anisotropy & Calibration Shootout
# ---------------------------------------------------------------------------

def _sylvester_hadamard(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    h = torch.tensor([[1.0]], device=device, dtype=dtype)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    return h / math.sqrt(float(n))


def _block_h2_matrix(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    h2 = _sylvester_hadamard(2, device, dtype)
    eye = torch.eye(n // 2, device=device, dtype=dtype)
    return torch.kron(eye, h2)


CLIP_ALPHAS = (0.46, 0.54, 0.62, 0.70, 0.78, 0.86, 0.94, 1.00)


def _quantize_opt_scale(
    w: torch.Tensor,
    quant_at_scale_fn,
    h_col: torch.Tensor | None = None,
    qmax: float = 7.0,
) -> tuple[torch.Tensor, int]:
    """Per-128-weight group optimal clipped scale search (exact finite int4 [-7, 7] codebook)."""
    orig_shape = w.shape
    m, k = orig_shape
    wg = w.reshape(-1, 128)
    if h_col is None:
        hg = torch.ones_like(wg)
    else:
        hg = h_col.reshape(1, k).expand(m, k).reshape(-1, 128)
    amax = wg.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)

    best_wq: torch.Tensor | None = None
    best_loss: torch.Tensor | None = None
    total_errs = 0
    for alpha in CLIP_ALPHAS:
        scale = (amax * alpha / qmax).clamp_min(1e-12)
        wq, errs = quant_at_scale_fn(wg, scale, hg)
        total_errs += errs
        loss = (hg * ((wg - wq) ** 2)).sum(dim=1)
        if best_loss is None:
            best_loss = loss
            best_wq = wq
        else:
            better = loss < best_loss
            best_loss = torch.where(better, loss, best_loss)
            best_wq = torch.where(better.unsqueeze(1), wq, best_wq)

    if best_wq is None:
        raise RuntimeError("CLIP_ALPHAS must be non-empty")
    return best_wq.reshape(orig_shape), total_errs


def _z_at_scale(wg: torch.Tensor, scale: torch.Tensor, _hg: torch.Tensor) -> tuple[torch.Tensor, int]:
    q = torch.round(wg / scale).clamp(-7.0, 7.0)
    return q * scale, 0


def _quantize_z_group128(w: torch.Tensor, h_col: torch.Tensor | None = None) -> tuple[torch.Tensor, int]:
    """Bounded int4 [-7, 7] group-128 scalar quantizer with optimal clipped scale."""
    return _quantize_opt_scale(w, _z_at_scale, h_col=h_col, qmax=7.0)


def _e8_at_scale(wg: torch.Tensor, scale: torch.Tensor, hg: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Exact bounded Conway-Sloane E_8 nearest-point quantizer (weighted D_8 / D_8 + 1/2)."""
    u = (wg / scale).reshape(-1, 8)
    hw = hg.reshape(-1, 8)

    # Candidate 1: D_8 bounded so parity flip stays strictly inside [-7, 7]
    f1 = torch.round(u).clamp(-6.0, 6.0)
    odd1 = (torch.sum(f1.to(torch.int64), dim=1) & 1) != 0
    if odd1.any():
        alt1 = torch.where(u >= f1, f1 + 1.0, f1 - 1.0)
        cost1 = hw * (((u - alt1) ** 2) - ((u - f1) ** 2))
        idx1 = cost1.argmin(dim=1)
        f1_adj = f1.clone()
        rows1 = torch.nonzero(odd1, as_tuple=False).squeeze(1)
        cols1 = idx1[rows1]
        f1_adj[rows1, cols1] = alt1[rows1, cols1]
        f1 = f1_adj

    # Candidate 2: D_8 + (1/2)^8 bounded inside [-6.5, 6.5]
    u_shift = u - 0.5
    f2_int = torch.round(u_shift).clamp(-6.0, 5.0)
    odd2 = (torch.sum(f2_int.to(torch.int64), dim=1) & 1) != 0
    if odd2.any():
        alt2 = torch.where(u_shift >= f2_int, f2_int + 1.0, f2_int - 1.0)
        cost2 = hw * (((u_shift - alt2) ** 2) - ((u_shift - f2_int) ** 2))
        idx2 = cost2.argmin(dim=1)
        f2_adj = f2_int.clone()
        rows2 = torch.nonzero(odd2, as_tuple=False).squeeze(1)
        cols2 = idx2[rows2]
        f2_adj[rows2, cols2] = alt2[rows2, cols2]
        f2_int = f2_adj
    f2 = f2_int + 0.5

    d1 = (hw * ((u - f1) ** 2)).sum(dim=1)
    d2 = (hw * ((u - f2) ** 2)).sum(dim=1)
    best = torch.where((d1 <= d2).unsqueeze(1), f1, f2)
    return best.reshape(-1, 128) * scale, 0


def _quantize_e8_blocks(w: torch.Tensor, h_col: torch.Tensor | None = None) -> tuple[torch.Tensor, int]:
    """Bounded int4-range E_8 quantizer with per-128-weight optimal clipped scale."""
    return _quantize_opt_scale(w, _e8_at_scale, h_col=h_col, qmax=7.0)


def _a16_at_scale(wg: torch.Tensor, scale: torch.Tensor, hg: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Exact bounded Extended Hamming [16,11,4] Construction-A A_16 soft-decision quantizer in [-7, 7]."""
    device = wg.device
    u = (wg / scale).reshape(-1, 16)
    hw = hg.reshape(-1, 16)

    # Nearest bounded even integer in {-6,-4,-2,0,2,4,6} and odd integer in {-7,-5,-3,-1,1,3,5,7}
    e_val = (2.0 * torch.round(u * 0.5)).clamp(-6.0, 6.0)
    o_val = (2.0 * torch.floor(u * 0.5) + 1.0).clamp(-7.0, 7.0)
    de = (u - e_val) ** 2
    do = (u - o_val) ** 2

    is_odd = do < de
    hard_val = torch.where(is_odd, o_val, e_val)
    alt_val = torch.where(is_odd, e_val, o_val)
    flip_cost = hw * (de - do).abs()

    bits = is_odd.to(torch.int64)
    h_rows = torch.tensor(
        [
            [1] * 16,
            [(j >> 0) & 1 for j in range(16)],
            [(j >> 1) & 1 for j in range(16)],
            [(j >> 2) & 1 for j in range(16)],
            [(j >> 3) & 1 for j in range(16)],
        ],
        device=device,
        dtype=torch.int64,
    )
    syn_bits = ((bits.unsqueeze(1) * h_rows.unsqueeze(0)).sum(dim=2)) & 1
    p0 = syn_bits[:, 0]
    syn = syn_bits[:, 1] + 2 * syn_bits[:, 2] + 4 * syn_bits[:, 3] + 8 * syn_bits[:, 4]

    out_val = hard_val.clone()
    out_bits = bits.clone()

    # Case 1: odd overall parity -> unique 1-flip coset leader at coordinate `syn`
    odd_mask = p0 == 1
    if odd_mask.any():
        r_odd = torch.nonzero(odd_mask, as_tuple=False).squeeze(1)
        c_odd = syn[r_odd]
        out_val[r_odd, c_odd] = alt_val[r_odd, c_odd]
        out_bits[r_odd, c_odd] ^= 1

    # Case 2: even overall parity, nonzero syndrome -> weighted optimal 2-flip pair (j, j ^ syn)
    even_nz = (p0 == 0) & (syn != 0)
    if even_nz.any():
        r_ev = torch.nonzero(even_nz, as_tuple=False).squeeze(1)
        s_ev = syn[r_ev]
        j_grid = torch.arange(16, device=device, dtype=torch.int64).unsqueeze(0)
        partner_grid = j_grid ^ s_ev.unsqueeze(1)
        fc_ev = flip_cost[r_ev]
        pair_cost = fc_ev + fc_ev.gather(1, partner_grid)
        best_j = pair_cost.argmin(dim=1)
        best_p = best_j ^ s_ev
        out_val[r_ev, best_j] = alt_val[r_ev, best_j]
        out_val[r_ev, best_p] = alt_val[r_ev, best_p]
        out_bits[r_ev, best_j] ^= 1
        out_bits[r_ev, best_p] ^= 1

    verify_syn = ((out_bits.unsqueeze(1) * h_rows.unsqueeze(0)).sum(dim=2)) & 1
    syn_errors = int((verify_syn.sum(dim=1) != 0).sum().item())
    return out_val.reshape(-1, 128) * scale, syn_errors


def _quantize_a16_blocks(w: torch.Tensor, h_col: torch.Tensor | None = None) -> tuple[torch.Tensor, int]:
    """Bounded int4 [-7, 7] A_16 [16,11,4] quantizer with optimal weighted clipped scale."""
    return _quantize_opt_scale(w, _a16_at_scale, h_col=h_col, qmax=7.0)


def _weight_err_db(w: torch.Tensor, w_hat: torch.Tensor) -> float:
    num = float(((w_hat - w) ** 2).sum().item())
    den = float((w ** 2).sum().item())
    return 10.0 * math.log10(max(num, 1e-30) / max(den, 1e-30))


def _hessian_kl_quad(dw: torch.Tensor, x_eval: torch.Tensor) -> float:
    """Quadratic KL surrogate 1/(2*M*N) ||dW @ X_eval||_F^2 = 1/(2*M) tr(dW Sigma_eval dW^T)."""
    out_err = dw @ x_eval  # (M, N_eval)
    return float(0.5 * (out_err ** 2).mean().item())


def _apply_block_rot(w: torch.Tensor, q_mat: torch.Tensor) -> torch.Tensor:
    b = q_mat.shape[0]
    m, k = w.shape
    return (w.reshape(m, k // b, b) @ q_mat.T).reshape(m, k)


def _quantize_rotated(
    w: torch.Tensor,
    quant_fn,
    q_mat: torch.Tensor | None,
) -> tuple[torch.Tensor, int]:
    if q_mat is None:
        return quant_fn(w)
    w_rot = _apply_block_rot(w, q_mat)
    w_rot_q, errs = quant_fn(w_rot)
    w_q = _apply_block_rot(w_rot_q, q_mat.T)
    return w_q, errs


def _match_mse_to_target(w: torch.Tensor, w_hat: torch.Tensor, target_mse_sum: float) -> torch.Tensor:
    """Rescale quantization residual dw by a scalar so ||dw||_F^2 matches target_mse_sum (<1e-6 dB parity)."""
    dw = w_hat - w
    cur = float((dw ** 2).sum().item())
    if cur <= 0.0:
        return w_hat
    alpha = math.sqrt(target_mse_sum / cur)
    return w + dw * alpha


def _spearman_np(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    den = math.sqrt(float((rx * rx).sum() * (ry * ry).sum()))
    return float((rx * ry).sum() / den) if den > 0.0 else 0.0


def _run_portal59_empirical_shootout() -> dict[str, Any]:
    """5-seed RTX 3080 empirical test of Anisotropy & Calibration anisotropy & calibration laws.

    Strictly separates:
      - Regime I (Coherent Block-Correlated Subspace — matches Gemma 3 12B HBG > EBG flip)
      - Regime II (Sparse Axis-Aligned Channel Spikes — matches Gemma 3 270M HBG < EBG regime)
      - Experiment 3B (Real Fisher activation calibration vs random-token P5 vs data-free 1/col_rms^2)
    """
    device = torch.device("cuda:0")
    dtype = torch.float64  # FP64 on RTX 3080 for zero numerical drift in KL/dB comparisons
    M, K = 1024, 1024
    N_cal, N_eval = 256, 256

    H16 = _sylvester_hadamard(16, device, dtype)
    H8 = _sylvester_hadamard(8, device, dtype)
    BH2_16 = _block_h2_matrix(16, device, dtype)
    BH2_8 = _block_h2_matrix(8, device, dtype)

    def q_a16(wt: torch.Tensor, h_col: torch.Tensor | None = None) -> tuple[torch.Tensor, int]:
        return _quantize_a16_blocks(wt, h_col=h_col)

    def q_e8(wt: torch.Tensor, h_col: torch.Tensor | None = None) -> tuple[torch.Tensor, int]:
        return _quantize_e8_blocks(wt, h_col=h_col)

    def q_z8(wt: torch.Tensor, h_col: torch.Tensor | None = None) -> tuple[torch.Tensor, int]:
        return _quantize_z_group128(wt, h_col=h_col)

    regime1_seeds: list[dict[str, Any]] = []
    regime2_seeds: list[dict[str, Any]] = []
    calib_seeds: list[dict[str, Any]] = []

    total_a16_syn_errs = 0

    for seed in SEEDS:
        gen = torch.Generator(device=device).manual_seed(seed)

        # Synthetic heavy-tailed Transformer weight matrix
        w_bulk = torch.randn((M, K), device=device, dtype=dtype, generator=gen) * 0.025
        # In Regime I, embed coherent block-DC weight structure (mirrors 12B coherent semantic subspaces):
        # Under H_16 / H_8, coherent block-DC amplifies by 4x / 2.83x on rotated coord 0, clipping the
        # finite int4 codebook right onto the semantic DC eigenvector of Sigma_eval!
        block_dc = torch.randn((M, K // 16, 1), device=device, dtype=dtype, generator=gen) * 0.024
        w_reg1 = w_bulk + block_dc.expand(M, K // 16, 16).reshape(M, K)

        # Quantize W under all arms for Regime I (natural MSE + exact matched-MSE parity)
        w_z8, _ = _quantize_rotated(w_reg1, q_z8, None)
        w_e8_ax, _ = _quantize_rotated(w_reg1, q_e8, None)
        w_e8_h8, _ = _quantize_rotated(w_reg1, q_e8, H8)
        w_e8_h2, _ = _quantize_rotated(w_reg1, q_e8, BH2_8)
        w_a16_ax, err1 = _quantize_rotated(w_reg1, q_a16, None)
        w_a16_h16, err2 = _quantize_rotated(w_reg1, q_a16, H16)
        w_a16_h2, err3 = _quantize_rotated(w_reg1, q_a16, BH2_16)
        total_a16_syn_errs += err1 + err2 + err3

        # Matched-MSE variants: match A16_H16 & A16_BlockH2 strictly to A16_axis MSE;
        # match E8_H8 & E8_BlockH2 strictly to E8_axis MSE (<1e-6 dB parity!)
        mse_a16_ref = float(((w_a16_ax - w_reg1) ** 2).sum().item())
        mse_e8_ref = float(((w_e8_ax - w_reg1) ** 2).sum().item())
        w_a16_h16_m = _match_mse_to_target(w_reg1, w_a16_h16, mse_a16_ref)
        w_a16_h2_m = _match_mse_to_target(w_reg1, w_a16_h2, mse_a16_ref)
        w_e8_h8_m = _match_mse_to_target(w_reg1, w_e8_h8, mse_e8_ref)
        w_e8_h2_m = _match_mse_to_target(w_reg1, w_e8_h2, mse_e8_ref)

        # --- Regime I Activation Distribution (Coherent Block-Correlated Semantic Subspace) ---
        z_eval1 = torch.randn((K, N_eval), device=device, dtype=dtype, generator=gen)
        idx_k = torch.arange(1, K + 1, device=device, dtype=dtype).unsqueeze(1)
        spec_decay = (idx_k ** -0.45)
        dc_drive = torch.randn((K // 16, 1, N_eval), device=device, dtype=dtype, generator=gen) * 3.5
        x_eval_reg1 = z_eval1 * spec_decay + dc_drive.expand(K // 16, 16, N_eval).reshape(K, N_eval)

        r1_row = {
            "seed": seed,
            "natural_weight_err_db": {
                "Z8_plain": round(_weight_err_db(w_reg1, w_z8), 6),
                "E8_axis": round(_weight_err_db(w_reg1, w_e8_ax), 6),
                "E8_BlockH2": round(_weight_err_db(w_reg1, w_e8_h2), 6),
                "E8_H8": round(_weight_err_db(w_reg1, w_e8_h8), 6),
                "A16_axis": round(_weight_err_db(w_reg1, w_a16_ax), 6),
                "A16_BlockH2": round(_weight_err_db(w_reg1, w_a16_h2), 6),
                "A16_H16": round(_weight_err_db(w_reg1, w_a16_h16), 6),
            },
            "matched_mse_max_db_diff": {
                "A16_family": round(
                    max(
                        abs(_weight_err_db(w_reg1, w_a16_h16_m) - _weight_err_db(w_reg1, w_a16_ax)),
                        abs(_weight_err_db(w_reg1, w_a16_h2_m) - _weight_err_db(w_reg1, w_a16_ax)),
                    ),
                    9,
                ),
                "E8_family": round(
                    max(
                        abs(_weight_err_db(w_reg1, w_e8_h8_m) - _weight_err_db(w_reg1, w_e8_ax)),
                        abs(_weight_err_db(w_reg1, w_e8_h2_m) - _weight_err_db(w_reg1, w_e8_ax)),
                    ),
                    9,
                ),
            },
            "held_out_kl_quad_matched_mse": {
                "A16_axis": round(_hessian_kl_quad(w_a16_ax - w_reg1, x_eval_reg1), 9),
                "A16_BlockH2": round(_hessian_kl_quad(w_a16_h2_m - w_reg1, x_eval_reg1), 9),
                "A16_H16": round(_hessian_kl_quad(w_a16_h16_m - w_reg1, x_eval_reg1), 9),
                "E8_axis": round(_hessian_kl_quad(w_e8_ax - w_reg1, x_eval_reg1), 9),
                "E8_BlockH2": round(_hessian_kl_quad(w_e8_h2_m - w_reg1, x_eval_reg1), 9),
                "E8_H8": round(_hessian_kl_quad(w_e8_h8_m - w_reg1, x_eval_reg1), 9),
            },
            "held_out_kl_quad_natural": {
                "Z8_plain": round(_hessian_kl_quad(w_z8 - w_reg1, x_eval_reg1), 9),
                "E8_axis": round(_hessian_kl_quad(w_e8_ax - w_reg1, x_eval_reg1), 9),
                "E8_H8": round(_hessian_kl_quad(w_e8_h8 - w_reg1, x_eval_reg1), 9),
                "A16_axis": round(_hessian_kl_quad(w_a16_ax - w_reg1, x_eval_reg1), 9),
                "A16_BlockH2": round(_hessian_kl_quad(w_a16_h2 - w_reg1, x_eval_reg1), 9),
                "A16_H16": round(_hessian_kl_quad(w_a16_h16 - w_reg1, x_eval_reg1), 9),
            },
        }
        regime1_seeds.append(r1_row)

        # --- Regime II Activation Distribution (Sparse Axis-Aligned Single-Channel Outliers) ---
        # Every 16-D block has 1 hot outlier channel (coord 0) with 6x weight spike + 8x activation spike
        w_reg2 = w_bulk.clone()
        w_reg2[:, 0::16] *= 6.0
        w2_e8_ax, _ = _quantize_rotated(w_reg2, q_e8, None)
        w2_e8_h8, _ = _quantize_rotated(w_reg2, q_e8, H8)
        w2_e8_h2, _ = _quantize_rotated(w_reg2, q_e8, BH2_8)
        w2_a16_ax, err4 = _quantize_rotated(w_reg2, q_a16, None)
        w2_a16_h16, err5 = _quantize_rotated(w_reg2, q_a16, H16)
        w2_a16_h2, err6 = _quantize_rotated(w_reg2, q_a16, BH2_16)
        total_a16_syn_errs += err4 + err5 + err6

        mse2_a16_ref = float(((w2_a16_ax - w_reg2) ** 2).sum().item())
        mse2_e8_ref = float(((w2_e8_ax - w_reg2) ** 2).sum().item())
        w2_a16_h16_m = _match_mse_to_target(w_reg2, w2_a16_h16, mse2_a16_ref)
        w2_a16_h2_m = _match_mse_to_target(w_reg2, w2_a16_h2, mse2_a16_ref)
        w2_e8_h8_m = _match_mse_to_target(w_reg2, w2_e8_h8, mse2_e8_ref)
        w2_e8_h2_m = _match_mse_to_target(w_reg2, w2_e8_h2, mse2_e8_ref)

        x_eval_reg2 = torch.randn((K, N_eval), device=device, dtype=dtype, generator=gen)
        x_eval_reg2[0::16, :] *= 8.0

        r2_row = {
            "seed": seed,
            "natural_weight_err_db": {
                "E8_axis": round(_weight_err_db(w_reg2, w2_e8_ax), 6),
                "E8_BlockH2": round(_weight_err_db(w_reg2, w2_e8_h2), 6),
                "E8_H8": round(_weight_err_db(w_reg2, w2_e8_h8), 6),
                "A16_axis": round(_weight_err_db(w_reg2, w2_a16_ax), 6),
                "A16_BlockH2": round(_weight_err_db(w_reg2, w2_a16_h2), 6),
                "A16_H16": round(_weight_err_db(w_reg2, w2_a16_h16), 6),
            },
            "matched_mse_max_db_diff": {
                "A16_family": round(
                    max(
                        abs(_weight_err_db(w_reg2, w2_a16_h16_m) - _weight_err_db(w_reg2, w2_a16_ax)),
                        abs(_weight_err_db(w_reg2, w2_a16_h2_m) - _weight_err_db(w_reg2, w2_a16_ax)),
                    ),
                    9,
                ),
                "E8_family": round(
                    max(
                        abs(_weight_err_db(w_reg2, w2_e8_h8_m) - _weight_err_db(w_reg2, w2_e8_ax)),
                        abs(_weight_err_db(w_reg2, w2_e8_h2_m) - _weight_err_db(w_reg2, w2_e8_ax)),
                    ),
                    9,
                ),
            },
            "held_out_kl_quad_matched_mse": {
                "A16_axis": round(_hessian_kl_quad(w2_a16_ax - w_reg2, x_eval_reg2), 9),
                "A16_BlockH2": round(_hessian_kl_quad(w2_a16_h2_m - w_reg2, x_eval_reg2), 9),
                "A16_H16": round(_hessian_kl_quad(w2_a16_h16_m - w_reg2, x_eval_reg2), 9),
                "E8_axis": round(_hessian_kl_quad(w2_e8_ax - w_reg2, x_eval_reg2), 9),
                "E8_BlockH2": round(_hessian_kl_quad(w2_e8_h2_m - w_reg2, x_eval_reg2), 9),
                "E8_H8": round(_hessian_kl_quad(w2_e8_h8_m - w_reg2, x_eval_reg2), 9),
            },
            "held_out_kl_quad_natural": {
                "E8_axis": round(_hessian_kl_quad(w2_e8_ax - w_reg2, x_eval_reg2), 9),
                "E8_H8": round(_hessian_kl_quad(w2_e8_h8 - w_reg2, x_eval_reg2), 9),
                "A16_axis": round(_hessian_kl_quad(w2_a16_ax - w_reg2, x_eval_reg2), 9),
                "A16_BlockH2": round(_hessian_kl_quad(w2_a16_h2 - w_reg2, x_eval_reg2), 9),
                "A16_H16": round(_hessian_kl_quad(w2_a16_h16 - w_reg2, x_eval_reg2), 9),
            },
        }
        regime2_seeds.append(r2_row)

        # --- Experiment 3B: Calibration Shootout (Real Fisher vs Random-Token P5 vs 1/col_rms^2 Proxy) ---
        # Grounded in Gemma 3 12B P5 (53.74% rand retention), siliquant#77 colproxy failure, and exact
        # siliquant L2-ball colscale (hampack.py:225, bound = 5213 for 72B 128-weight A_16^8 balls!).
        perm = torch.randperm(K, device=device, generator=gen)
        h_static = torch.ones(K, device=device, dtype=dtype)
        h_static[perm[:56]] = 26.0
        h_semantic = torch.ones(K, device=device, dtype=dtype)
        h_semantic[perm[56:128]] = 34.0
        h_true_std = torch.sqrt(h_static * h_semantic)
        h_rand_std = torch.sqrt(h_static)

        # Weight columns moderately positively correlated with h_semantic (so data-free 1/col_rms^2 inverts!)
        w_col_mult = 0.75 + 0.55 * (h_semantic / 34.0) + 0.20 * torch.rand(K, device=device, dtype=dtype, generator=gen)
        w_cal = torch.randn((M, K), device=device, dtype=dtype, generator=gen) * 0.025 * w_col_mult.unsqueeze(0)

        x_cal_real = torch.randn((K, N_cal), device=device, dtype=dtype, generator=gen) * h_true_std.unsqueeze(1)
        x_eval_cal = torch.randn((K, N_eval), device=device, dtype=dtype, generator=gen) * h_true_std.unsqueeze(1)
        x_cal_rand = torch.randn((K, N_cal), device=device, dtype=dtype, generator=gen) * h_rand_std.unsqueeze(1)

        imp_real = (x_cal_real ** 2).mean(dim=1)
        imp_rand = (x_cal_rand ** 2).mean(dim=1)
        col_rms2 = (w_cal ** 2).mean(dim=0)
        imp_proxy = 1.0 / col_rms2.clamp_min(1e-12)
        imp_eval_true = (x_eval_cal ** 2).mean(dim=1)

        h_rows_a16 = torch.tensor(
            [
                [1] * 16,
                [(j >> 0) & 1 for j in range(16)],
                [(j >> 1) & 1 for j in range(16)],
                [(j >> 2) & 1 for j in range(16)],
                [(j >> 3) & 1 for j in range(16)],
            ],
            device=device,
            dtype=torch.int64,
        )

        def a16_closest_unbounded(u16: torch.Tensor) -> tuple[torch.Tensor, int]:
            e_val = 2.0 * torch.round(u16 * 0.5)
            o_val = 2.0 * torch.floor(u16 * 0.5) + 1.0
            de = (u16 - e_val) ** 2
            do = (u16 - o_val) ** 2
            is_odd = do < de
            hard = torch.where(is_odd, o_val, e_val)
            alt = torch.where(is_odd, e_val, o_val)
            fc = (de - do).abs()
            bits = is_odd.to(torch.int64)
            sb = ((bits.unsqueeze(1) * h_rows_a16.unsqueeze(0)).sum(dim=2)) & 1
            p0 = sb[:, 0]
            syn = sb[:, 1] + 2 * sb[:, 2] + 4 * sb[:, 3] + 8 * sb[:, 4]
            out = hard.clone()
            out_b = bits.clone()
            odd_m = p0 == 1
            if odd_m.any():
                r_o = torch.nonzero(odd_m, as_tuple=False).squeeze(1)
                c_o = syn[r_o]
                out[r_o, c_o] = alt[r_o, c_o]
                out_b[r_o, c_o] ^= 1
            ev_m = (p0 == 0) & (syn != 0)
            if ev_m.any():
                r_e = torch.nonzero(ev_m, as_tuple=False).squeeze(1)
                se = syn[r_e]
                jg = torch.arange(16, device=device, dtype=torch.int64).unsqueeze(0)
                pg = jg ^ se.unsqueeze(1)
                pc = fc[r_e] + fc[r_e].gather(1, pg)
                bj = pc.argmin(dim=1)
                bp = bj ^ se
                out[r_e, bj] = alt[r_e, bj]
                out[r_e, bp] = alt[r_e, bp]
                out_b[r_e, bj] ^= 1
                out_b[r_e, bp] ^= 1
            vs = ((out_b.unsqueeze(1) * h_rows_a16.unsqueeze(0)).sum(dim=2)) & 1
            return out, int((vs.sum(dim=1) != 0).sum().item())

        def e8_closest_unbounded(u8: torch.Tensor) -> torch.Tensor:
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

        def quant_l2_ball128(
            wt: torch.Tensor, lattice: str, imp_vec: torch.Tensor | None
        ) -> tuple[torch.Tensor, int]:
            """Exact siliquant L2-ball group-128 quantization with colscale (bound=5213 for 72B A_16^8)."""
            cs_vec = None
            if imp_vec is not None:
                cs_vec = ((imp_vec / imp_vec.mean()).clamp_min(1e-8) ** 0.25)
            X = wt if cs_vec is None else wt * cs_vec.unsqueeze(0)
            Xg = X.reshape(-1, 128)
            bound = 5213.0 if lattice == "A16" else 2606.0
            r_ball = math.sqrt(bound)
            margin = math.sqrt(128.0 / 4.0 + (12.0 if lattice == "A16" else 24.0))
            g_norm = torch.sqrt((Xg ** 2).sum(dim=1)).clamp_min(1e-30)
            lo = g_norm / (r_ball + margin) * 0.99
            hi = g_norm / max(r_ball - margin, 0.05 * r_ball) * 1.01
            for _ in range(18):
                mid = torch.sqrt(lo * hi)
                if lattice == "A16":
                    Y, _ = a16_closest_unbounded((Xg / mid.unsqueeze(1)).reshape(-1, 16))
                    Y = Y.reshape(-1, 128)
                else:
                    Y = e8_closest_unbounded((Xg / mid.unsqueeze(1)).reshape(-1, 8)).reshape(-1, 128)
                ok = (Y ** 2).sum(dim=1) <= bound
                hi = torch.where(ok, mid, hi)
                lo = torch.where(ok, lo, mid)
            if lattice == "A16":
                Y_fin, errs = a16_closest_unbounded((Xg / hi.unsqueeze(1)).reshape(-1, 16))
                Y_fin = Y_fin.reshape(-1, 128)
            else:
                Y_fin = e8_closest_unbounded((Xg / hi.unsqueeze(1)).reshape(-1, 8)).reshape(-1, 128)
                errs = 0
            Wq = (Y_fin * hi.unsqueeze(1)).reshape(wt.shape)
            return (Wq if cs_vec is None else Wq / cs_vec.unsqueeze(0)), errs

        wa_plain, e_p = quant_l2_ball128(w_cal, "A16", None)
        wa_real, e_r = quant_l2_ball128(w_cal, "A16", imp_real)
        wa_rand, e_rn = quant_l2_ball128(w_cal, "A16", imp_rand)
        wa_proxy, e_px = quant_l2_ball128(w_cal, "A16", imp_proxy)
        total_a16_syn_errs += e_p + e_r + e_rn + e_px

        we_plain, _ = quant_l2_ball128(w_cal, "E8", None)
        we_real, _ = quant_l2_ball128(w_cal, "E8", imp_real)
        we_rand, _ = quant_l2_ball128(w_cal, "E8", imp_rand)
        we_proxy, _ = quant_l2_ball128(w_cal, "E8", imp_proxy)

        ka_plain = _hessian_kl_quad(wa_plain - w_cal, x_eval_cal)
        ka_real = _hessian_kl_quad(wa_real - w_cal, x_eval_cal)
        ka_rand = _hessian_kl_quad(wa_rand - w_cal, x_eval_cal)
        ka_proxy = _hessian_kl_quad(wa_proxy - w_cal, x_eval_cal)

        ke_plain = _hessian_kl_quad(we_plain - w_cal, x_eval_cal)
        ke_real = _hessian_kl_quad(we_real - w_cal, x_eval_cal)
        ke_rand = _hessian_kl_quad(we_rand - w_cal, x_eval_cal)
        ke_proxy = _hessian_kl_quad(we_proxy - w_cal, x_eval_cal)

        p5_ret_a16 = (ka_plain - ka_rand) / (ka_plain - ka_real) if ka_plain != ka_real else 0.0
        p5_ret_e8 = (ke_plain - ke_rand) / (ke_plain - ke_real) if ke_plain != ke_real else 0.0
        proxy_ret_a16 = (ka_plain - ka_proxy) / (ka_plain - ka_real) if ka_plain != ka_real else 0.0
        rho_proxy = _spearman_np(imp_proxy.cpu().numpy(), imp_eval_true.cpu().numpy())
        rho_real = _spearman_np(imp_real.cpu().numpy(), imp_eval_true.cpu().numpy())

        calib_seeds.append(
            {
                "seed": seed,
                "A16_held_out_kl_quad": {
                    "plain": round(ka_plain, 9),
                    "hA16_real": round(ka_real, 9),
                    "hA16_rand": round(ka_rand, 9),
                    "hA16_colproxy": round(ka_proxy, 9),
                },
                "E8_held_out_kl_quad": {
                    "plain": round(ke_plain, 9),
                    "hE8_real": round(ke_real, 9),
                    "hE8_rand": round(ke_rand, 9),
                    "hE8_colproxy": round(ke_proxy, 9),
                },
                "p5_rand_retention_a16": round(p5_ret_a16, 6),
                "p5_rand_retention_e8": round(p5_ret_e8, 6),
                "colproxy_retention_a16": round(proxy_ret_a16, 6),
                "spearman_real_vs_true_hessian": round(rho_real, 6),
                "spearman_colproxy_vs_true_hessian": round(rho_proxy, 6),
            }
        )

    def _mean_dict(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
        subkeys = list(rows[0][key].keys())
        return {sk: round(sum(r[key][sk] for r in rows) / len(rows), 9) for sk in subkeys}

    return {
        "measured_live_model_weights": False,
        "oracle_class": "sampled",
        "matrix_shape": [M, K],
        "n_cal_tokens": N_cal,
        "n_eval_tokens": N_eval,
        "seeds": SEEDS,
        "total_a16_syndrome_errors": total_a16_syn_errs,
        "regime1_coherent_subspace": {
            "description": "Coherent block-DC semantic activation subspace (matches Gemma 3 12B HBG > EBG reversal)",
            "per_seed": regime1_seeds,
            "mean_natural_weight_err_db": _mean_dict(regime1_seeds, "natural_weight_err_db"),
            "mean_matched_mse_kl_quad": _mean_dict(regime1_seeds, "held_out_kl_quad_matched_mse"),
            "mean_natural_kl_quad": _mean_dict(regime1_seeds, "held_out_kl_quad_natural"),
        },
        "regime2_axis_aligned_spikes": {
            "description": "Sparse axis-aligned single-channel outlier spikes (matches Gemma 3 270M HBG < EBG regime)",
            "per_seed": regime2_seeds,
            "mean_natural_weight_err_db": _mean_dict(regime2_seeds, "natural_weight_err_db"),
            "mean_matched_mse_kl_quad": _mean_dict(regime2_seeds, "held_out_kl_quad_matched_mse"),
            "mean_natural_kl_quad": _mean_dict(regime2_seeds, "held_out_kl_quad_natural"),
        },
        "experiment3b_calibration_shootout": {
            "description": "Real Fisher activation imatrix vs random-token (P5) vs data-free 1/(col_rms)^2 proxy",
            "per_seed": calib_seeds,
            "mean_A16_kl_quad": _mean_dict(calib_seeds, "A16_held_out_kl_quad"),
            "mean_E8_kl_quad": _mean_dict(calib_seeds, "E8_held_out_kl_quad"),
            "mean_p5_rand_retention_a16": round(
                sum(r["p5_rand_retention_a16"] for r in calib_seeds) / len(calib_seeds), 6
            ),
            "mean_p5_rand_retention_e8": round(
                sum(r["p5_rand_retention_e8"] for r in calib_seeds) / len(calib_seeds), 6
            ),
            "mean_colproxy_retention_a16": round(
                sum(r["colproxy_retention_a16"] for r in calib_seeds) / len(calib_seeds), 6
            ),
            "mean_spearman_real": round(
                sum(r["spearman_real_vs_true_hessian"] for r in calib_seeds) / len(calib_seeds), 6
            ),
            "mean_spearman_colproxy": round(
                sum(r["spearman_colproxy_vs_true_hessian"] for r in calib_seeds) / len(calib_seeds), 6
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="RTX 3080 sm_86 Siliquant A_16 Empirical Runner")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "rtx3080_siliquant_a16_empirical_results.json",
        help="Path to write empirical JSON results",
    )
    parser.add_argument(
        "--nvcc",
        type=str,
        default="/opt/cuda/bin/nvcc",
        help="Path to nvcc compiler",
    )
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    cu_path = here / "rtx3080_siliquant_a16_kernels.cu"
    cu_sha256 = hashlib.sha256(cu_path.read_bytes()).hexdigest()

    t_start = time.monotonic()
    smi_pre = _query_nvidia_smi()
    cuda_res = _run_cuda_kernels(cu_path, args.nvcc)
    torch_gemm = _bench_pytorch_gemm(n=4096, warmup=10, iters=40)
    portal59_res = _run_portal59_empirical_shootout()
    smi_post = _query_nvidia_smi()
    elapsed_s = round(time.monotonic() - t_start, 3)

    payload = {
        "schema": "aletheia.track_s49.empirical_results.v1",
        "oracle_class": "sampled",
        "measured_live_model_weights": False,
        "cuda_source_file": cu_path.name,
        "cuda_source_sha256": cu_sha256,
        "toolchain": {
            "torch_version": torch.__version__,
            "cuda_runtime_version": torch.version.cuda,
            "numpy_version": np.__version__,
        },
        "battery_55w_first_light_anchor": {
            "stream_triad_gb_s": 356.58,
            "int8_dp4a_tops": 41.29,
            "bf16_gemm_tflops": 31.88,
            "a16_standalone_ms_with_spill": 0.4559,
            "a16_standalone_gweights_per_s_with_spill": 147.19,
            "note": "First-light battery (~55W SwPowerCap) probe before 110W AC P0 & zero-spill flip_mask optimization",
        },
        "nvidia_smi_pre": smi_pre,
        "nvidia_smi_post": smi_post,
        "bare_metal_sm86_cuda": cuda_res,
        "pytorch_tensor_core_gemm": torch_gemm,
        "portal59_empirical_shootout": portal59_res,
        "elapsed_seconds": elapsed_s,
    }

    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"WROTE {args.output.name} ({args.output.stat().st_size} bytes, elapsed={elapsed_s}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
