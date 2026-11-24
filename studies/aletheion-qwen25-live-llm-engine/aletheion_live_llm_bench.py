#!/usr/bin/env python3
"""Real-Weight Full-Model Qwen2.5-0.5B A_16 PTX/SASS vs AVX-512 Live Benchmark.

Executes all 4 empirical arms on the real offline-cached Qwen2.5-0.5B weights
(24 layers, 168 real linear matrices = 357,826,560 weights + 122 preserved BF16
non-block tensors):
  - Arm A: PyTorch BF16 baseline on RTX 3080 (sm_86)
  - Arm B: Bare-metal sm_86 PTX/SASS A16_axis (LOP3+PRMT+dp4a pure & live 5-POPC
           verify + cp.async/IMMA B=32 prefill across all 168 real matrices)
  - Arm C: Intel i9-11900H AVX-512 8-core CPU-only A16_axis (vpermb+vpdpbusd
           pure & live vpopcntq verify in a clean zero-CUDA-context phase)
  - Arm D: Cold-disk 18 B / 32 w (201,277,440 B) hampack -> GPU SoA transcode +
           TTFT + 128-token technical prompt generation parity vs BF16 & AVX-512

Captures synchronized hardware telemetry:
  - Intel RAPL microjoule counters (package-0, core, uncore, psys)
  - NVIDIA NVML 100 Hz trapezoidal integration of hardware mW rail power
    (plus temperature, SM/mem clocks, and P-state)
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import math
import os
import pathlib
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

import numpy as np

QWEN_LAYERS = 24
QWEN_H = 896
QWEN_I = 4864
QWEN_KV_DIM = 128
QWEN_VOCAB = 151936
PROJ_NAMES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
PROJ_SHAPES = (
    (QWEN_H, QWEN_H),
    (QWEN_KV_DIM, QWEN_H),
    (QWEN_KV_DIM, QWEN_H),
    (QWEN_H, QWEN_H),
    (QWEN_I, QWEN_H),
    (QWEN_I, QWEN_H),
    (QWEN_H, QWEN_I),
)
TOTAL_LINEAR_MATRICES = 168
TOTAL_BLOCK_WEIGHTS = 357_826_560
TOTAL_GROUPS_32 = 11_182_080
TOTAL_SUBBLOCKS_16 = 22_364_160
TOTAL_COLD_18B_BYTES = 201_277_440
TOTAL_NONBLOCK_BF16_TENSORS = 122

TECH_PROMPT = (
    "Extended Hamming [16,11,4] lattice quantization packs 32 neural network "
    "weights into 18 bytes while preserving hardware parity detection. On "
    "Ampere GPUs, fusing nibble unpacking with dp4a instructions"
)


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes | memoryview) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Hardware Energy & Power Telemetry (Intel RAPL + NVML 100 Hz Power Integral)
# ---------------------------------------------------------------------------

RAPL_DOMAINS = {
    "package_0": pathlib.Path("/sys/class/powercap/intel-rapl:0"),
    "core": pathlib.Path("/sys/class/powercap/intel-rapl:0:0"),
    "uncore": pathlib.Path("/sys/class/powercap/intel-rapl:0:1"),
    "psys": pathlib.Path("/sys/class/powercap/intel-rapl:1"),
}


def read_rapl_snapshot() -> dict[str, tuple[int, int]]:
    snap: dict[str, tuple[int, int]] = {}
    for name, dom in RAPL_DOMAINS.items():
        e_file = dom / "energy_uj"
        m_file = dom / "max_energy_range_uj"
        if e_file.is_file() and m_file.is_file():
            try:
                e_uj = int(e_file.read_text(encoding="utf-8").strip())
                m_uj = int(m_file.read_text(encoding="utf-8").strip())
                snap[name] = (e_uj, m_uj)
            except OSError:
                pass
    return snap


def rapl_delta_mj(
    start: dict[str, tuple[int, int]],
    end: dict[str, tuple[int, int]],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, (e0, m0) in start.items():
        if name not in end:
            continue
        e1, _ = end[name]
        diff_uj = (e1 - e0) if e1 >= e0 else (m0 - e0 + e1)
        out[name] = round(diff_uj / 1000.0, 4)
    return out


class NvmlPowerSampler:
    """100 Hz NVML hardware rail power sampler with trapezoidal energy integration.

    Note: GA104M laptop NVML nvmlDeviceGetTotalEnergyConsumption exhibits >16x
    spurious idle counter drift relative to the INA3221 rail power sensor, so
    energy is computed by trapezoidal integration of nvmlDeviceGetPowerUsage (mW).
    """

    def __init__(self, device_index: int = 0, interval_s: float = 0.010) -> None:
        self.device_index = device_index
        self.interval_s = interval_s
        self._nvml = ctypes.CDLL("libnvidia-ml.so.1")
        rc = self._nvml.nvmlInit_v2()
        if rc != 0:
            raise RuntimeError(f"nvmlInit_v2 failed with rc={rc}")
        self._handle = ctypes.c_void_p()
        rc = self._nvml.nvmlDeviceGetHandleByIndex_v2(
            ctypes.c_uint(device_index), ctypes.byref(self._handle)
        )
        if rc != 0:
            raise RuntimeError(f"nvmlDeviceGetHandleByIndex_v2 failed with rc={rc}")
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[tuple[float, float, int, int, int]] = []

    def _read_once(self) -> tuple[float, float, int, int, int]:
        t = time.perf_counter()
        p_mw = ctypes.c_uint(0)
        temp_c = ctypes.c_uint(0)
        pstate = ctypes.c_uint(0)
        sm_mhz = ctypes.c_uint(0)
        self._nvml.nvmlDeviceGetPowerUsage(self._handle, ctypes.byref(p_mw))
        self._nvml.nvmlDeviceGetTemperature(self._handle, ctypes.c_uint(0), ctypes.byref(temp_c))
        self._nvml.nvmlDeviceGetPerformanceState(self._handle, ctypes.byref(pstate))
        self._nvml.nvmlDeviceGetClockInfo(self._handle, ctypes.c_uint(0), ctypes.byref(sm_mhz))
        return (t, float(p_mw.value) / 1000.0, int(temp_c.value), int(pstate.value), int(sm_mhz.value))

    def snapshot_once(self) -> dict[str, Any]:
        _, p_w, temp_c, pstate, sm_mhz = self._read_once()
        return {
            "power_w": round(p_w, 3),
            "temp_c": temp_c,
            "pstate": f"P{pstate}",
            "sm_clock_mhz": sm_mhz,
        }

    def start(self) -> None:
        self._samples = [self._read_once()]
        self._stop.clear()

        def _loop() -> None:
            while not self._stop.is_set():
                time.sleep(self.interval_s)
                self._samples.append(self._read_once())

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._samples.append(self._read_once())

        energy_j = 0.0
        for i in range(1, len(self._samples)):
            t0, p0, _, _, _ = self._samples[i - 1]
            t1, p1, _, _, _ = self._samples[i]
            dt = max(0.0, t1 - t0)
            energy_j += 0.5 * (p0 + p1) * dt

        powers = [s[1] for s in self._samples]
        temps = [s[2] for s in self._samples]
        pstates = [s[3] for s in self._samples]
        clocks = [s[4] for s in self._samples]
        dt_total = max(1e-9, self._samples[-1][0] - self._samples[0][0])
        return {
            "samples": len(self._samples),
            "window_s": round(dt_total, 4),
            "gpu_energy_mj": round(energy_j * 1000.0, 3),
            "gpu_mean_power_w": round(energy_j / dt_total, 3),
            "gpu_peak_power_w": round(max(powers), 3),
            "gpu_max_temp_c": max(temps),
            "gpu_min_pstate": f"P{min(pstates)}",
            "gpu_max_pstate": f"P{max(pstates)}",
            "gpu_mean_sm_mhz": round(sum(clocks) / len(clocks), 1),
        }

    def close(self) -> None:
        try:
            self._nvml.nvmlShutdown()
        except Exception:
            pass


def measure_window_energy(
    nvml: NvmlPowerSampler,
    fn,
    total_units: int,
    unit_label: str,
) -> dict[str, Any]:
    """Run fn() inside a synchronized wall-clock + RAPL + NVML energy window."""
    rapl0 = read_rapl_snapshot()
    nvml.start()
    t0 = time.perf_counter()
    fn_ret = fn()
    t1 = time.perf_counter()
    gpu_stats = nvml.stop()
    rapl1 = read_rapl_snapshot()

    wall_s = max(1e-9, t1 - t0)
    rapl_mj = rapl_delta_mj(rapl0, rapl1)
    gpu_mj = float(gpu_stats["gpu_energy_mj"])
    pkg_mj = float(rapl_mj.get("package_0", 0.0))
    core_mj = float(rapl_mj.get("core", 0.0))
    psys_mj = float(rapl_mj.get("psys", 0.0))
    combined_gpu_cpu_mj = gpu_mj + pkg_mj

    return {
        "units": total_units,
        "unit_label": unit_label,
        "wall_s": round(wall_s, 5),
        "ms_per_unit": round((wall_s * 1000.0) / total_units, 5),
        "units_per_s": round(total_units / wall_s, 3),
        "gpu_energy_mj_total": round(gpu_mj, 3),
        "gpu_mj_per_unit": round(gpu_mj / total_units, 5),
        "rapl_energy_mj_total": rapl_mj,
        "cpu_pkg_mj_per_unit": round(pkg_mj / total_units, 5),
        "cpu_core_mj_per_unit": round(core_mj / total_units, 5),
        "psys_mj_per_unit": round(psys_mj / total_units, 5),
        "combined_gpu_plus_cpu_pkg_mj_per_unit": round(combined_gpu_cpu_mj / total_units, 5),
        "gpu_telemetry": gpu_stats,
        "fn_result": fn_ret,
    }


# ---------------------------------------------------------------------------
# Exact Vectorized Nearest-Codeword A_16 [16,11,4] Quantizer (PyTorch GPU)
# ---------------------------------------------------------------------------

def quantize_matrix_a16_exact_gpu(w_f32_gpu):
    """Quantize a real [M, K] weight matrix to exact 4.500-bpw A_16 [16,11,4].

    Per 32-weight group: fp16 scale s = max(|w|) / 7.0 (floored at 1e-8).
    Per 16-weight subblock: nearest-codeword Extended Hamming [16,11,4]
    projection in [-7, +7] minimizing exact squared error (0 syndrome errors).
    Returns:
      nib_u8_np   : [M, K//2] uint8 packed biased u4 nibbles (u = q + 8 in [1, 15])
      scale_u16_np: [M, K//32] uint16 IEEE fp16 bit patterns
      w_deq_gpu   : [M, K] float32 dequantized weights (for exact rel_l2 audit)
      syn_errors  : int (verified 0)
    """
    import torch

    M, K = w_f32_gpu.shape
    ng = K // 32
    w_grp = w_f32_gpu.view(M, ng, 32)
    amax = w_grp.abs().amax(dim=-1)
    scale_fp16 = torch.clamp(amax / 7.0, min=1e-7).to(torch.float16)
    scale_f32 = scale_fp16.to(torch.float32)

    # Normalized target in [-7, +7], reshaped into 16-weight subblocks [N16, 16]
    x = torch.clamp(w_grp / scale_f32.unsqueeze(-1), -7.0, 7.0).view(-1, 16)
    n16 = x.shape[0]
    q = torch.round(x).to(torch.int32).clamp(-7, 7)

    # Best alternative parity-flipped integer q_alt in [-7, +7] for each coord:
    # Step toward residual (x - q) if within [-7, +7], else step inward.
    res = x - q.to(torch.float32)
    step = torch.where(res >= 0.0, torch.ones_like(q), -torch.ones_like(q))
    step = torch.where((q == 7) & (step > 0), -torch.ones_like(q), step)
    step = torch.where((q == -7) & (step < 0), torch.ones_like(q), step)
    q_alt = q + step  # guaranteed in [-7, +7] and opposite LSB!

    # Squared-error penalty delta_e[n, i] >= 0 for flipping coord i
    de = (q_alt.to(torch.float32) - x).square() - (q.to(torch.float32) - x).square()

    # Compute 5-bit Extended Hamming [16,11,4] syndrome on LSBs b = q & 1
    b = q & 1
    idx_ar = torch.arange(16, device=x.device, dtype=torch.int32)
    p0 = b.sum(dim=-1) & 1
    s0 = (b * ((idx_ar >> 0) & 1)).sum(dim=-1) & 1
    s1 = (b * ((idx_ar >> 1) & 1)).sum(dim=-1) & 1
    s2 = (b * ((idx_ar >> 2) & 1)).sum(dim=-1) & 1
    s3 = (b * ((idx_ar >> 3) & 1)).sum(dim=-1) & 1
    s_idx = s0 | (s1 << 1) | (s2 << 2) | (s3 << 3)  # in [0..15]

    row_ar = torch.arange(n16, device=x.device, dtype=torch.long)

    # Case 1: p0 == 1 -> flip single coordinate s_idx (distance-1 exact repair)
    odd_mask = p0 == 1
    if odd_mask.any():
        r_odd = row_ar[odd_mask]
        c_odd = s_idx[odd_mask].to(torch.long)
        q[r_odd, c_odd] = q_alt[r_odd, c_odd]

    # Case 2: p0 == 0 and s_idx != 0 -> flip minimum-cost pair (j, j ^ s_idx)
    even_mask = (p0 == 0) & (s_idx != 0)
    if even_mask.any():
        r_ev = row_ar[even_mask]
        s_ev = s_idx[even_mask].to(torch.long)  # [Ne]
        de_ev = de[r_ev]  # [Ne, 16]
        j_all = torch.arange(16, device=x.device, dtype=torch.long).unsqueeze(0)  # [1, 16]
        k_all = j_all ^ s_ev.unsqueeze(1)  # [Ne, 16]
        pair_cost = de_ev + torch.gather(de_ev, 1, k_all)  # [Ne, 16]
        best_j = pair_cost.argmin(dim=-1)  # [Ne]
        best_k = best_j ^ s_ev
        q[r_ev, best_j] = q_alt[r_ev, best_j]
        q[r_ev, best_k] = q_alt[r_ev, best_k]

    # Verify ZERO syndrome errors across every single 16-weight subblock!
    b2 = q & 1
    chk_p0 = b2.sum(dim=-1) & 1
    chk_s0 = (b2 * ((idx_ar >> 0) & 1)).sum(dim=-1) & 1
    chk_s1 = (b2 * ((idx_ar >> 1) & 1)).sum(dim=-1) & 1
    chk_s2 = (b2 * ((idx_ar >> 2) & 1)).sum(dim=-1) & 1
    chk_s3 = (b2 * ((idx_ar >> 3) & 1)).sum(dim=-1) & 1
    syn_err = int((chk_p0 | chk_s0 | chk_s1 | chk_s2 | chk_s3).sum().item())
    if syn_err != 0:
        raise RuntimeError(f"A_16 syndrome verification failed: {syn_err} errors")

    # Pack biased u4 nibbles: u = q + 8 in [1, 15]; low nibble = even, high nibble = odd
    u = (q + 8).view(M, K).to(torch.uint8)
    packed = (u[:, 0::2] & 0x0F) | ((u[:, 1::2] & 0x0F) << 4)
    w_deq = (q.view(M, ng, 32).to(torch.float32) * scale_f32.unsqueeze(-1)).view(M, K)

    nib_u8_np = packed.cpu().numpy().copy()
    scale_u16_np = scale_fp16.view(torch.uint16).cpu().numpy().copy()
    return nib_u8_np, scale_u16_np, w_deq, syn_err


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    a64 = a.astype(np.float64, copy=False)
    b64 = b.astype(np.float64, copy=False)
    denom = float(np.linalg.norm(b64))
    if denom == 0.0:
        return 0.0
    return float(np.linalg.norm(a64 - b64) / denom)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a64 = a.astype(np.float64, copy=False)
    b64 = b.astype(np.float64, copy=False)
    na = float(np.linalg.norm(a64))
    nb = float(np.linalg.norm(b64))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a64, b64) / (na * nb))


# ---------------------------------------------------------------------------
# C-ABI Loaders
# ---------------------------------------------------------------------------

def load_sm86_lib(so_path: pathlib.Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(so_path))
    lib.qwen25_gpu_create.restype = ctypes.c_void_p
    lib.qwen25_gpu_destroy.argtypes = [ctypes.c_void_p]
    lib.qwen25_gpu_load_globals.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.qwen25_gpu_load_layer_norms_biases.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.qwen25_gpu_load_proj.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint16),
    ]
    lib.qwen25_gpu_reset_kv.argtypes = [ctypes.c_void_p]
    lib.qwen25_gpu_get_syn_errors.argtypes = [ctypes.c_void_p]
    lib.qwen25_gpu_get_syn_errors.restype = ctypes.c_uint32
    lib.qwen25_gpu_forward_token.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.qwen25_gpu_bench_decode_sustained.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.qwen25_gpu_bench_decode_sustained.restype = ctypes.c_double
    lib.qwen25_gpu_generate_greedy.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.qwen25_gpu_generate_greedy.restype = ctypes.c_double
    lib.qwen25_gpu_bench_b32_imma_all168.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int8),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int8),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
    ]
    lib.qwen25_gpu_bench_b32_imma_all168.restype = ctypes.c_double
    lib.qwen25_gpu_transcode_18b_all168.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint32),
    ]
    lib.qwen25_gpu_transcode_18b_all168.restype = ctypes.c_double
    return lib


def load_avx512_lib(so_path: pathlib.Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(so_path))
    lib.qwen25_cpu_create.argtypes = [ctypes.c_int]
    lib.qwen25_cpu_create.restype = ctypes.c_void_p
    lib.qwen25_cpu_destroy.argtypes = [ctypes.c_void_p]
    lib.qwen25_cpu_load_globals.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.qwen25_cpu_load_layer_norms_biases.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.qwen25_cpu_load_proj.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint16),
    ]
    lib.qwen25_cpu_reset_kv.argtypes = [ctypes.c_void_p]
    lib.qwen25_cpu_get_syn_errors.argtypes = [ctypes.c_void_p]
    lib.qwen25_cpu_get_syn_errors.restype = ctypes.c_uint64
    lib.qwen25_cpu_forward_token.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.qwen25_cpu_bench_decode_sustained.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.qwen25_cpu_bench_decode_sustained.restype = ctypes.c_int
    return lib


def wait_for_gpu_idle(nvml: NvmlPowerSampler, max_wait_s: float = 5.0) -> dict[str, Any]:
    t0 = time.perf_counter()
    snap = nvml.snapshot_once()
    while (time.perf_counter() - t0) < max_wait_s:
        if snap["pstate"] == "P8" and snap["power_w"] <= 20.0:
            break
        time.sleep(0.25)
        snap = nvml.snapshot_once()
    return snap


# ---------------------------------------------------------------------------
# Phase C: Clean Zero-CUDA-Context AVX-512 8-Core CPU-Only Subprocess
# ---------------------------------------------------------------------------

def run_arm_c_subprocess(
    model_dir: pathlib.Path,
    avx512_so: pathlib.Path,
    disk_18b_path: pathlib.Path,
    expected_disk_sha: str,
    parity_dir: pathlib.Path,
    num_threads: int,
    out_json: pathlib.Path,
) -> None:
    from safetensors import safe_open

    raw_18b = disk_18b_path.read_bytes()
    actual_sha = sha256_bytes(raw_18b)
    if len(raw_18b) != TOTAL_COLD_18B_BYTES or actual_sha != expected_disk_sha:
        raise RuntimeError(
            f"Arm C 18B handoff mismatch: len={len(raw_18b)} sha={actual_sha}"
        )

    lib = load_avx512_lib(avx512_so)
    eng = lib.qwen25_cpu_create(num_threads)
    nvml = NvmlPowerSampler()
    try:
        # Load preserved BF16 globals & layer norms/biases directly from safetensors on CPU
        sf_path = model_dir / "model.safetensors"
        with safe_open(str(sf_path), framework="pt", device="cpu") as sf:
            import torch

            emb_u16 = (
                sf.get_tensor("model.embed_tokens.weight")
                .contiguous()
                .view(torch.uint16)
                .numpy()
            )
            fn_f32 = sf.get_tensor("model.norm.weight").to(torch.float32).contiguous().numpy()
            lib.qwen25_cpu_load_globals(
                eng,
                emb_u16.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                fn_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            )
            for l in range(QWEN_LAYERS):
                in_n = sf.get_tensor(f"model.layers.{l}.input_layernorm.weight").to(torch.float32).contiguous().numpy()
                po_n = sf.get_tensor(f"model.layers.{l}.post_attention_layernorm.weight").to(torch.float32).contiguous().numpy()
                qb = sf.get_tensor(f"model.layers.{l}.self_attn.q_proj.bias").to(torch.float32).contiguous().numpy()
                kb = sf.get_tensor(f"model.layers.{l}.self_attn.k_proj.bias").to(torch.float32).contiguous().numpy()
                vb = sf.get_tensor(f"model.layers.{l}.self_attn.v_proj.bias").to(torch.float32).contiguous().numpy()
                lib.qwen25_cpu_load_layer_norms_biases(
                    eng,
                    l,
                    in_n.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    po_n.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    qb.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    kb.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    vb.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                )

        # Unpack the exact same 18B records into CPU engine buffers
        arr_18b = np.frombuffer(raw_18b, dtype=np.uint8).reshape(TOTAL_GROUPS_32, 18)
        offset = 0
        for l in range(QWEN_LAYERS):
            for p, (M, K) in enumerate(PROJ_SHAPES):
                ng = M * (K // 32)
                sl = arr_18b[offset : offset + ng]
                nib = np.ascontiguousarray(sl[:, :16])
                sc = np.ascontiguousarray(sl[:, 16:18]).view(np.uint16).reshape(-1)
                lib.qwen25_cpu_load_proj(
                    eng,
                    l,
                    p,
                    nib.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                    sc.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                )
                offset += ng

        gpu_idle_before = wait_for_gpu_idle(nvml, max_wait_s=12.0)

        # Parity check vs GPU A16 & BF16 on prompt tokens
        meta = json.loads((parity_dir / "parity_meta.json").read_text(encoding="utf-8"))
        prompt_ids: list[int] = meta["prompt_ids"]
        bf16_logits_first = np.load(parity_dir / "bf16_logits_first.npy")
        gpu_a16_logits_first = np.load(parity_dir / "gpu_a16_logits_first.npy")
        gpu_a16_gen_tokens: list[int] = meta["gpu_a16_gen_tokens"]

        lib.qwen25_cpu_reset_kv(eng)
        cpu_logits = np.empty(QWEN_VOCAB, dtype=np.float32)
        for pos, tid in enumerate(prompt_ids):
            out_ptr = (
                cpu_logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                if pos == len(prompt_ids) - 1
                else None
            )
            lib.qwen25_cpu_forward_token(eng, int(tid), int(pos), 1, out_ptr)

        syn_after_prompt = int(lib.qwen25_cpu_get_syn_errors(eng))
        cpu_first_tok = int(np.argmax(cpu_logits))
        gpu_first_tok = int(np.argmax(gpu_a16_logits_first))
        bf16_first_tok = int(np.argmax(bf16_logits_first))
        cpu_vs_gpu_rel_l2 = rel_l2(cpu_logits, gpu_a16_logits_first)
        cpu_vs_gpu_cos = cosine_sim(cpu_logits, gpu_a16_logits_first)
        cpu_vs_bf16_rel_l2 = rel_l2(cpu_logits, bf16_logits_first)
        cpu_vs_bf16_cos = cosine_sim(cpu_logits, bf16_logits_first)

        # Verify 16 greedy continuation tokens match GPU A16!
        cpu_gen_16 = [cpu_first_tok]
        cur = cpu_first_tok
        for g in range(1, 16):
            lib.qwen25_cpu_forward_token(
                eng,
                int(cur),
                len(prompt_ids) + g - 1,
                1,
                cpu_logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            )
            cur = int(np.argmax(cpu_logits))
            cpu_gen_16.append(cur)
        cpu_vs_gpu_16tok_match = sum(
            1 for a, b in zip(cpu_gen_16, gpu_a16_gen_tokens[:16]) if a == b
        )

        # Calibrate iteration counts to achieve >= 2.5 s sustained windows
        lib.qwen25_cpu_reset_kv(eng)
        # Warmup 2 steps
        lib.qwen25_cpu_bench_decode_sustained(eng, int(prompt_ids[0]), 2, 0, 0)
        t_cal0 = time.perf_counter()
        lib.qwen25_cpu_bench_decode_sustained(eng, int(prompt_ids[0]), 5, 0, 0)
        t_bb = max(1e-4, (time.perf_counter() - t_cal0) / 5.0)
        n_bb = max(40, min(400, int(math.ceil(2.6 / t_bb))))

        t_cal1 = time.perf_counter()
        lib.qwen25_cpu_bench_decode_sustained(eng, int(prompt_ids[0]), 5, 0, 1)
        t_e2e = max(1e-4, (time.perf_counter() - t_cal1) / 5.0)
        n_e2e = max(30, min(300, int(math.ceil(2.6 / t_e2e))))

        # C1: Pure AVX-512 VNNI (live_verify=0)
        lib.qwen25_cpu_reset_kv(eng)
        c1_backbone = measure_window_energy(
            nvml,
            lambda: lib.qwen25_cpu_bench_decode_sustained(eng, int(prompt_ids[0]), n_bb, 0, 0),
            n_bb,
            "tok",
        )
        lib.qwen25_cpu_reset_kv(eng)
        c1_e2e = measure_window_energy(
            nvml,
            lambda: lib.qwen25_cpu_bench_decode_sustained(eng, int(prompt_ids[0]), n_e2e, 0, 1),
            n_e2e,
            "tok",
        )

        # C2: AVX-512 VNNI + Live vpopcntq [16,11,4] Parity Verify (live_verify=1)
        lib.qwen25_cpu_reset_kv(eng)
        c2_backbone = measure_window_energy(
            nvml,
            lambda: lib.qwen25_cpu_bench_decode_sustained(eng, int(prompt_ids[0]), n_bb, 1, 0),
            n_bb,
            "tok",
        )
        syn_c2_bb = int(lib.qwen25_cpu_get_syn_errors(eng))

        lib.qwen25_cpu_reset_kv(eng)
        c2_e2e = measure_window_energy(
            nvml,
            lambda: lib.qwen25_cpu_bench_decode_sustained(eng, int(prompt_ids[0]), n_e2e, 1, 1),
            n_e2e,
            "tok",
        )
        syn_c2_e2e = int(lib.qwen25_cpu_get_syn_errors(eng))

        out_doc = {
            "num_threads": num_threads,
            "omp_proc_bind": os.environ.get("OMP_PROC_BIND", ""),
            "omp_places": os.environ.get("OMP_PLACES", ""),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "gpu_idle_before_arm_c": gpu_idle_before,
            "parity_vs_gpu_a16": {
                "first_token_cpu": cpu_first_tok,
                "first_token_gpu_a16": gpu_first_tok,
                "first_token_bf16": bf16_first_tok,
                "top1_match_gpu_a16": bool(cpu_first_tok == gpu_first_tok),
                "top1_match_bf16": bool(cpu_first_tok == bf16_first_tok),
                "logits_rel_l2_vs_gpu_a16": round(cpu_vs_gpu_rel_l2, 8),
                "logits_cosine_vs_gpu_a16": round(cpu_vs_gpu_cos, 8),
                "logits_rel_l2_vs_bf16": round(cpu_vs_bf16_rel_l2, 6),
                "logits_cosine_vs_bf16": round(cpu_vs_bf16_cos, 8),
                "greedy_16tok_match_vs_gpu_a16": cpu_vs_gpu_16tok_match,
                "greedy_16tok_total": 16,
                "syn_errors_prompt_plus_16": syn_after_prompt,
            },
            "c1_pure_vnni": {
                "backbone_24layers": c1_backbone,
                "full_e2e_with_bf16_lm_head": c1_e2e,
            },
            "c2_live_vpopcntq_verify": {
                "backbone_24layers": c2_backbone,
                "full_e2e_with_bf16_lm_head": c2_e2e,
                "syn_errors_backbone": syn_c2_bb,
                "syn_errors_e2e": syn_c2_e2e,
                "verify_overhead_backbone_pct": round(
                    100.0 * (c2_backbone["ms_per_unit"] - c1_backbone["ms_per_unit"]) / c1_backbone["ms_per_unit"],
                    3,
                ),
                "verify_overhead_e2e_pct": round(
                    100.0 * (c2_e2e["ms_per_unit"] - c1_e2e["ms_per_unit"]) / c1_e2e["ms_per_unit"],
                    3,
                ),
            },
        }
        out_json.write_text(json.dumps(out_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    finally:
        nvml.close()
        lib.qwen25_cpu_destroy(eng)


# ---------------------------------------------------------------------------
# Main Orchestrator (Arms A, D, B on GPU -> Clean Subprocess Arm C on CPU)
# ---------------------------------------------------------------------------

def run_all_arms(
    model_dir: pathlib.Path,
    sm86_so: pathlib.Path,
    avx512_so: pathlib.Path,
    out_json: pathlib.Path,
    cpu_threads: int,
) -> None:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.cache_utils import DynamicCache

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available")

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    cc_major, cc_minor = torch.cuda.get_device_capability(0)

    sf_path = model_dir / "model.safetensors"
    cfg_path = model_dir / "config.json"
    tok_path = model_dir / "tokenizer.json"
    model_safetensors_sha = sha256_file(sf_path)
    config_sha = sha256_file(cfg_path)
    tokenizer_sha = sha256_file(tok_path)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    prompt_ids = tokenizer.encode(TECH_PROMPT, add_special_tokens=False)
    n_prompt = len(prompt_ids)
    n_gen = 128

    nvml = NvmlPowerSampler()
    tmp_cold_18b: pathlib.Path | None = None
    tmp_parity_dir: pathlib.Path | None = None
    tmp_shm_18b: pathlib.Path | None = None
    tmp_arm_c_json: pathlib.Path | None = None

    try:
        # ===================================================================
        # ARM A: PyTorch BF16 Baseline on RTX 3080 (sm_86)
        # ===================================================================
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            dtype=torch.bfloat16,
            local_files_only=True,
        ).to(device)
        model.eval()

        # Verify exact tensor inventory
        sd = model.state_dict()
        lin_keys: list[str] = []
        for l in range(QWEN_LAYERS):
            for pn in PROJ_NAMES:
                lin_keys.append(f"model.layers.{l}.{pn}.weight")
        non_lin_keys = sorted(k for k in sd.keys() if k not in set(lin_keys) and k != "lm_head.weight")
        if len(lin_keys) != TOTAL_LINEAR_MATRICES:
            raise RuntimeError(f"Expected {TOTAL_LINEAR_MATRICES} linear matrices, got {len(lin_keys)}")
        total_w = sum(int(sd[k].numel()) for k in lin_keys)
        if total_w != TOTAL_BLOCK_WEIGHTS:
            raise RuntimeError(f"Expected {TOTAL_BLOCK_WEIGHTS} weights, got {total_w}")
        if len(non_lin_keys) != TOTAL_NONBLOCK_BF16_TENSORS:
            raise RuntimeError(f"Expected {TOTAL_NONBLOCK_BF16_TENSORS} non-block tensors, got {len(non_lin_keys)}")

        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            # Exact BF16 prompt prefill + 128-token greedy generation + TTFT
            torch.cuda.synchronize()
            t_ttft0 = time.perf_counter()
            cache_bf16 = DynamicCache()
            out_pref = model(input_ids=prompt_tensor, past_key_values=cache_bf16, use_cache=True)
            bf16_logits_first = out_pref.logits[0, -1, :].to(torch.float32).cpu().numpy()
            next_tok = int(torch.argmax(out_pref.logits[0, -1, :]).item())
            torch.cuda.synchronize()
            bf16_ttft_ms = (time.perf_counter() - t_ttft0) * 1000.0

            bf16_gen_tokens = [next_tok]
            bf16_tf_logits_sample: list[np.ndarray] = [bf16_logits_first]
            cur_id = torch.tensor([[next_tok]], dtype=torch.long, device=device)
            t_gen0 = time.perf_counter()
            for g in range(1, n_gen):
                out_g = model(input_ids=cur_id, past_key_values=cache_bf16, use_cache=True)
                lg = out_g.logits[0, -1, :]
                if g < 16:
                    bf16_tf_logits_sample.append(lg.to(torch.float32).cpu().numpy())
                next_tok = int(torch.argmax(lg).item())
                bf16_gen_tokens.append(next_tok)
                cur_id = torch.tensor([[next_tok]], dtype=torch.long, device=device)
            torch.cuda.synchronize()
            bf16_gen_127_ms = (time.perf_counter() - t_gen0) * 1000.0
            bf16_gen_text = tokenizer.decode(bf16_gen_tokens, skip_special_tokens=True)

            # Sustained Arm A1 (24-layer backbone decode without lm_head) & A2 (full E2E with lm_head)
            seed_tok = torch.tensor([[prompt_ids[0]]], dtype=torch.long, device=device)

            def _bf16_decode_loop(n_iters: int, include_lm_head: bool) -> int:
                # Fresh KV cache of length 32 recycled cleanly each 32 steps
                last_t = int(prompt_ids[0])
                kv = DynamicCache()
                for it in range(n_iters):
                    if (it & 31) == 0:
                        kv = DynamicCache()
                    if include_lm_head:
                        o = model(input_ids=seed_tok, past_key_values=kv, use_cache=True)
                        last_t = int(torch.argmax(o.logits[0, -1, :]).item())
                    else:
                        _ = model.model(input_ids=seed_tok, past_key_values=kv, use_cache=True)
                torch.cuda.synchronize()
                return last_t

            # Warmup + calibrate >= 2.5 s
            _bf16_decode_loop(10, False)
            t0 = time.perf_counter()
            _bf16_decode_loop(15, False)
            dt_a1 = max(1e-4, (time.perf_counter() - t0) / 15.0)
            n_a1 = max(60, min(600, int(math.ceil(2.6 / dt_a1))))

            _bf16_decode_loop(10, True)
            t0 = time.perf_counter()
            _bf16_decode_loop(15, True)
            dt_a2 = max(1e-4, (time.perf_counter() - t0) / 15.0)
            n_a2 = max(60, min(600, int(math.ceil(2.6 / dt_a2))))

            a1_backbone = measure_window_energy(
                nvml, lambda: _bf16_decode_loop(n_a1, False), n_a1, "tok"
            )
            a2_e2e = measure_window_energy(
                nvml, lambda: _bf16_decode_loop(n_a2, True), n_a2, "tok"
            )

            # Arm A3: B=32 PyTorch BF16 linear prefill across all 168 real matrices
            torch.manual_seed(58)
            x_bf16_h = torch.randn(32, QWEN_H, dtype=torch.bfloat16, device=device) * 0.25
            x_bf16_i = torch.randn(32, QWEN_I, dtype=torch.bfloat16, device=device) * 0.25
            lin_mods = []
            for l in range(QWEN_LAYERS):
                ly = model.model.layers[l]
                lin_mods.extend([
                    (ly.self_attn.q_proj, QWEN_H),
                    (ly.self_attn.k_proj, QWEN_H),
                    (ly.self_attn.v_proj, QWEN_H),
                    (ly.self_attn.o_proj, QWEN_H),
                    (ly.mlp.gate_proj, QWEN_H),
                    (ly.mlp.up_proj, QWEN_H),
                    (ly.mlp.down_proj, QWEN_I),
                ])

            def _bf16_b32_all168(iters: int) -> int:
                for _ in range(iters):
                    for mod, kdim in lin_mods:
                        _ = mod(x_bf16_h if kdim == QWEN_H else x_bf16_i)
                torch.cuda.synchronize()
                return iters

            _bf16_b32_all168(5)
            t0 = time.perf_counter()
            _bf16_b32_all168(10)
            dt_a3 = max(1e-4, (time.perf_counter() - t0) / 10.0)
            n_a3 = max(40, min(2000, int(math.ceil(2.6 / dt_a3))))
            a3_b32 = measure_window_energy(
                nvml, lambda: _bf16_b32_all168(n_a3), n_a3 * 32, "tok"
            )
            a3_batch_ms = a3_b32["ms_per_unit"] * 32.0
            a3_tflops = (2.0 * 32.0 * TOTAL_BLOCK_WEIGHTS) / (a3_batch_ms * 1e9)
            a3_b32["ms_per_b32_batch"] = round(a3_batch_ms, 5)
            a3_b32["bf16_tflops"] = round(a3_tflops, 3)

            # ===============================================================
            # Quantize ALL 168 Real Linear Matrices to Exact 4.500-bpw A16_axis
            # ===============================================================
            t_q0 = time.perf_counter()
            disk_18b_chunks: list[np.ndarray] = []
            w_err_sq_sum = 0.0
            w_ref_sq_sum = 0.0
            max_mat_rel_l2 = 0.0
            total_quant_syn_err = 0
            l0_q_w_deq_f64: np.ndarray | None = None

            for l in range(QWEN_LAYERS):
                for p, pn in enumerate(PROJ_NAMES):
                    key = f"model.layers.{l}.{pn}.weight"
                    wf = sd[key].to(torch.float32)
                    nib_np, sc_np, w_deq, syn_e = quantize_matrix_a16_exact_gpu(wf)
                    total_quant_syn_err += syn_e
                    diff_sq = float((w_deq - wf).square().sum().item())
                    ref_sq = float(wf.square().sum().item())
                    w_err_sq_sum += diff_sq
                    w_ref_sq_sum += ref_sq
                    mat_rl2 = math.sqrt(diff_sq / max(1e-30, ref_sq))
                    if mat_rl2 > max_mat_rel_l2:
                        max_mat_rel_l2 = mat_rl2
                    if l == 0 and p == 0:
                        l0_q_w_deq_f64 = w_deq.to(torch.float64).cpu().numpy()

                    ng = nib_np.shape[0] * (nib_np.shape[1] // 16)
                    rec = np.empty((ng, 18), dtype=np.uint8)
                    rec[:, :16] = nib_np.reshape(ng, 16)
                    rec[:, 16:18] = sc_np.view(np.uint8).reshape(ng, 2)
                    disk_18b_chunks.append(rec)

            quant_wall_s = time.perf_counter() - t_q0
            disk_18b_arr = np.concatenate(disk_18b_chunks, axis=0)
            disk_18b_bytes = disk_18b_arr.tobytes()
            if len(disk_18b_bytes) != TOTAL_COLD_18B_BYTES:
                raise RuntimeError(f"Cold 18B size mismatch: {len(disk_18b_bytes)}")
            disk_18b_sha = sha256_bytes(disk_18b_bytes)
            agg_weight_rel_l2 = math.sqrt(w_err_sq_sum / max(1e-30, w_ref_sq_sum))

            # Extract preserved BF16 globals & per-layer norms/biases to CPU numpy
            embed_bf16_u16 = sd["model.embed_tokens.weight"].cpu().contiguous().view(torch.uint16).numpy()
            final_norm_f32 = sd["model.norm.weight"].to(torch.float32).cpu().contiguous().numpy()
            layer_nb: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
            for l in range(QWEN_LAYERS):
                layer_nb.append((
                    sd[f"model.layers.{l}.input_layernorm.weight"].to(torch.float32).cpu().contiguous().numpy(),
                    sd[f"model.layers.{l}.post_attention_layernorm.weight"].to(torch.float32).cpu().contiguous().numpy(),
                    sd[f"model.layers.{l}.self_attn.q_proj.bias"].to(torch.float32).cpu().contiguous().numpy(),
                    sd[f"model.layers.{l}.self_attn.k_proj.bias"].to(torch.float32).cpu().contiguous().numpy(),
                    sd[f"model.layers.{l}.self_attn.v_proj.bias"].to(torch.float32).cpu().contiguous().numpy(),
                ))

            # Prepare deterministic B=32 INT8 activations + exact FP64 reference for Layer 0 q_proj
            rng = np.random.default_rng(58)
            x_f_h = rng.standard_normal((32, QWEN_H)).astype(np.float32) * 0.25
            x_f_i = rng.standard_normal((32, QWEN_I)).astype(np.float32) * 0.25

            def _quant_b32_act(xf: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
                B, K = xf.shape
                ng_k = K // 32
                xg = xf.reshape(B, ng_k, 32)
                am = np.max(np.abs(xg), axis=-1)
                sc = np.maximum(am / 127.0, 1e-12).astype(np.float32)
                q8 = np.clip(np.rint(xg / sc[:, :, None]), -127, 127).astype(np.int8)
                x_deq = (q8.astype(np.float64) * sc.astype(np.float64)[:, :, None]).reshape(B, K)
                return np.ascontiguousarray(q8.reshape(B, K)), np.ascontiguousarray(sc), x_deq

            x_q8_h, x_sc_h, x_deq_h_f64 = _quant_b32_act(x_f_h)
            x_q8_i, x_sc_i, _ = _quant_b32_act(x_f_i)
            l0_q_bias_f64 = layer_nb[0][2].astype(np.float64)
            imma_ref_y_f64 = x_deq_h_f64 @ l0_q_w_deq_f64.T + l0_q_bias_f64[None, :]

        # Release PyTorch BF16 model from GPU VRAM cleanly before loading bare-metal sm_86 engine!
        del model, sd, lin_mods, cache_bf16, out_pref, x_bf16_h, x_bf16_i
        gc.collect()
        torch.cuda.empty_cache()

        # ===================================================================
        # ARM D: Cold-Disk 18B hampack -> Hot VRAM Transcode + TTFT + 128-Tok
        # ===================================================================
        fd_cold, cold_path_str = tempfile.mkstemp(prefix="s58_qwen25_18b_", suffix=".bin", dir="/tmp")
        tmp_cold_18b = pathlib.Path(cold_path_str)
        os.chmod(tmp_cold_18b, 0o600)
        with os.fdopen(fd_cold, "wb") as fh:
            fh.write(disk_18b_bytes)
            fh.flush()
            os.fsync(fh.fileno())

        # Cold-disk read from /tmp into host memory
        t_rd0 = time.perf_counter()
        cold_read_buf = tmp_cold_18b.read_bytes()
        disk_read_ms = (time.perf_counter() - t_rd0) * 1000.0
        if sha256_bytes(cold_read_buf) != disk_18b_sha:
            raise RuntimeError("Cold-disk 18B readback SHA mismatch")
        cold_read_np = np.frombuffer(cold_read_buf, dtype=np.uint8)

        # Immediately remove /tmp cold file once read back (keep /dev/shm copy only for Arm C handoff)
        tmp_cold_18b.unlink()
        tmp_cold_18b = None

        lib_gpu = load_sm86_lib(sm86_so)
        eng_gpu = lib_gpu.qwen25_gpu_create()
        try:
            lib_gpu.qwen25_gpu_load_globals(
                eng_gpu,
                embed_bf16_u16.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                final_norm_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            )
            for l in range(QWEN_LAYERS):
                in_n, po_n, qb, kb, vb = layer_nb[l]
                lib_gpu.qwen25_gpu_load_layer_norms_biases(
                    eng_gpu,
                    l,
                    in_n.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    po_n.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    qb.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    kb.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    vb.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                )

            # GPU H2D + 18B AoS -> SoA transcode with live 5-POPC syndrome verify!
            syn_transcode = ctypes.c_uint32(0)
            transcode_ms = float(
                lib_gpu.qwen25_gpu_transcode_18b_all168(
                    eng_gpu,
                    cold_read_np.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                    ctypes.byref(syn_transcode),
                )
            )
            if syn_transcode.value != 0:
                raise RuntimeError(f"GPU transcode syndrome error count={syn_transcode.value}")

            # Single-token exact logit parity after prompt prefill (vs BF16)
            lib_gpu.qwen25_gpu_reset_kv(eng_gpu)
            gpu_logits_first = np.empty(QWEN_VOCAB, dtype=np.float32)
            for pos, tid in enumerate(prompt_ids):
                out_ptr = (
                    gpu_logits_first.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                    if pos == n_prompt - 1
                    else None
                )
                lib_gpu.qwen25_gpu_forward_token(eng_gpu, int(tid), int(pos), 1, out_ptr)
            syn_first = int(lib_gpu.qwen25_gpu_get_syn_errors(eng_gpu))

            gpu_first_tok = int(np.argmax(gpu_logits_first))
            bf16_first_tok = int(np.argmax(bf16_logits_first))
            top5_gpu = set(np.argsort(gpu_logits_first)[-5:].tolist())
            top5_bf16 = set(np.argsort(bf16_logits_first)[-5:].tolist())
            first_logit_rel_l2 = rel_l2(gpu_logits_first, bf16_logits_first)
            first_logit_cos = cosine_sim(gpu_logits_first, bf16_logits_first)

            # Teacher-forced 128-token parity vs BF16 (feeding BF16's token sequence)
            lib_gpu.qwen25_gpu_reset_kv(eng_gpu)
            for pos, tid in enumerate(prompt_ids[:-1]):
                lib_gpu.qwen25_gpu_forward_token(eng_gpu, int(tid), int(pos), 1, None)
            tf_top1_matches = 0
            tf_rel_l2_list: list[float] = []
            tf_buf = np.empty(QWEN_VOCAB, dtype=np.float32)
            # Step 0 predicts bf16_gen_tokens[0] from prompt_ids[-1]
            tf_inputs = [prompt_ids[-1]] + bf16_gen_tokens[: n_gen - 1]
            for step_idx, in_tok in enumerate(tf_inputs):
                lib_gpu.qwen25_gpu_forward_token(
                    eng_gpu,
                    int(in_tok),
                    n_prompt - 1 + step_idx,
                    1,
                    tf_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                )
                pred_tok = int(np.argmax(tf_buf))
                if pred_tok == bf16_gen_tokens[step_idx]:
                    tf_top1_matches += 1
                if step_idx < len(bf16_tf_logits_sample):
                    tf_rel_l2_list.append(rel_l2(tf_buf, bf16_tf_logits_sample[step_idx]))
            syn_tf_128 = int(lib_gpu.qwen25_gpu_get_syn_errors(eng_gpu))

            # Free-running 128-token greedy generation on GPU (live_verify=1!) + exact TTFT
            prompt_i32 = np.asarray(prompt_ids, dtype=np.int32)
            gpu_gen_out = np.empty(n_gen, dtype=np.int32)
            ttft_ms_c = ctypes.c_double(0.0)
            total_gen_ms = float(
                lib_gpu.qwen25_gpu_generate_greedy(
                    eng_gpu,
                    prompt_i32.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                    n_prompt,
                    n_gen,
                    1,
                    gpu_gen_out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                    ctypes.byref(ttft_ms_c),
                )
            )
            syn_gen_128 = int(lib_gpu.qwen25_gpu_get_syn_errors(eng_gpu))
            gpu_gen_tokens = [int(x) for x in gpu_gen_out.tolist()]
            gpu_gen_text = tokenizer.decode(gpu_gen_tokens, skip_special_tokens=True)
            free_prefix_match = 0
            for a, b in zip(gpu_gen_tokens, bf16_gen_tokens):
                if a == b:
                    free_prefix_match += 1
                else:
                    break
            free_pos_match = sum(1 for a, b in zip(gpu_gen_tokens, bf16_gen_tokens) if a == b)
            gen_decode_127_ms = max(1e-6, total_gen_ms - float(ttft_ms_c.value))

            arm_d = {
                "cold_18b_bytes": TOTAL_COLD_18B_BYTES,
                "cold_18b_sha256": disk_18b_sha,
                "cold_disk_read_ms": round(disk_read_ms, 4),
                "cold_disk_read_gb_s": round((TOTAL_COLD_18B_BYTES / 1e9) / (disk_read_ms / 1000.0), 3),
                "gpu_h2d_plus_soa_transcode_ms": round(transcode_ms, 4),
                "gpu_transcode_gweights_s": round((TOTAL_BLOCK_WEIGHTS / 1e9) / (transcode_ms / 1000.0), 3),
                "total_cold_to_hot_vram_ms": round(disk_read_ms + transcode_ms, 4),
                "transcode_syn_errors": int(syn_transcode.value),
                "prompt_token_count": n_prompt,
                "prompt_sha256": sha256_bytes(TECH_PROMPT.encode("utf-8")),
                "ttft_a16_sm86_ms": round(float(ttft_ms_c.value), 4),
                "ttft_bf16_pytorch_ms": round(bf16_ttft_ms, 4),
                "gen_128_total_ms": round(total_gen_ms, 4),
                "gen_128_decode_tok_s": round(127.0 / (gen_decode_127_ms / 1000.0), 3),
                "bf16_gen_128_decode_tok_s": round(127.0 / (bf16_gen_127_ms / 1000.0), 3),
                "gen_128_live_verify_syn_errors": syn_gen_128,
                "parity_vs_bf16": {
                    "first_token_a16": gpu_first_tok,
                    "first_token_bf16": bf16_first_tok,
                    "first_token_top1_match": bool(gpu_first_tok == bf16_first_tok),
                    "first_token_top5_overlap": len(top5_gpu & top5_bf16),
                    "first_token_logits_rel_l2": round(first_logit_rel_l2, 6),
                    "first_token_logits_cosine": round(first_logit_cos, 8),
                    "teacher_forced_128_top1_matches": tf_top1_matches,
                    "teacher_forced_128_total": n_gen,
                    "teacher_forced_128_top1_rate": round(tf_top1_matches / float(n_gen), 6),
                    "teacher_forced_first16_mean_logit_rel_l2": round(
                        sum(tf_rel_l2_list) / max(1, len(tf_rel_l2_list)), 6
                    ),
                    "teacher_forced_syn_errors": syn_tf_128,
                    "free_running_128_exact_prefix_match": free_prefix_match,
                    "free_running_128_positional_matches": free_pos_match,
                    "syn_errors_first": syn_first,
                },
                "bf16_generated_preview": bf16_gen_text[:240],
                "a16_sm86_generated_preview": gpu_gen_text[:240],
            }

            # ===============================================================
            # ARM B: RTX 3080 sm_86 PTX/SASS A16_axis Sustained Benchmarks
            # ===============================================================
            last_tok_c = ctypes.c_int(0)

            # Warmup + calibrate >= 2.5 s windows
            lib_gpu.qwen25_gpu_reset_kv(eng_gpu)
            lib_gpu.qwen25_gpu_bench_decode_sustained(eng_gpu, int(prompt_ids[0]), 20, 0, 0, ctypes.byref(last_tok_c))
            cal_bb_ms = float(
                lib_gpu.qwen25_gpu_bench_decode_sustained(eng_gpu, int(prompt_ids[0]), 60, 0, 0, ctypes.byref(last_tok_c))
            )
            n_b_bb = max(300, min(6000, int(math.ceil(2600.0 / max(0.2, cal_bb_ms)))))

            lib_gpu.qwen25_gpu_bench_decode_sustained(eng_gpu, int(prompt_ids[0]), 20, 0, 1, ctypes.byref(last_tok_c))
            cal_e2e_ms = float(
                lib_gpu.qwen25_gpu_bench_decode_sustained(eng_gpu, int(prompt_ids[0]), 40, 0, 1, ctypes.byref(last_tok_c))
            )
            n_b_e2e = max(200, min(4000, int(math.ceil(2600.0 / max(0.3, cal_e2e_ms)))))

            # B1: Pure LOP3+PRMT+dp4a (live_verify=0)
            lib_gpu.qwen25_gpu_reset_kv(eng_gpu)
            b1_backbone = measure_window_energy(
                nvml,
                lambda: float(
                    lib_gpu.qwen25_gpu_bench_decode_sustained(
                        eng_gpu, int(prompt_ids[0]), n_b_bb, 0, 0, ctypes.byref(last_tok_c)
                    )
                ),
                n_b_bb,
                "tok",
            )
            b1_bb_gpu_ms = float(b1_backbone["fn_result"])
            b1_backbone["gpu_event_ms_per_tok"] = round(b1_bb_gpu_ms, 5)
            b1_backbone["gpu_event_tok_s"] = round(1000.0 / b1_bb_gpu_ms, 3)
            b1_backbone["physical_gb_s"] = round((TOTAL_COLD_18B_BYTES / 1e9) / (b1_bb_gpu_ms / 1000.0), 3)
            b1_backbone["bf16_equiv_effective_gb_s"] = round(
                ((TOTAL_BLOCK_WEIGHTS * 2) / 1e9) / (b1_bb_gpu_ms / 1000.0), 3
            )

            lib_gpu.qwen25_gpu_reset_kv(eng_gpu)
            b1_e2e = measure_window_energy(
                nvml,
                lambda: float(
                    lib_gpu.qwen25_gpu_bench_decode_sustained(
                        eng_gpu, int(prompt_ids[0]), n_b_e2e, 0, 1, ctypes.byref(last_tok_c)
                    )
                ),
                n_b_e2e,
                "tok",
            )
            b1_e2e_gpu_ms = float(b1_e2e["fn_result"])
            b1_e2e["gpu_event_ms_per_tok"] = round(b1_e2e_gpu_ms, 5)
            b1_e2e["gpu_event_tok_s"] = round(1000.0 / b1_e2e_gpu_ms, 3)

            # B2: Live 5-POPC [16,11,4] Parity Verify (live_verify=1)
            lib_gpu.qwen25_gpu_reset_kv(eng_gpu)
            b2_backbone = measure_window_energy(
                nvml,
                lambda: float(
                    lib_gpu.qwen25_gpu_bench_decode_sustained(
                        eng_gpu, int(prompt_ids[0]), n_b_bb, 1, 0, ctypes.byref(last_tok_c)
                    )
                ),
                n_b_bb,
                "tok",
            )
            syn_b2_bb = int(lib_gpu.qwen25_gpu_get_syn_errors(eng_gpu))
            b2_bb_gpu_ms = float(b2_backbone["fn_result"])
            b2_backbone["gpu_event_ms_per_tok"] = round(b2_bb_gpu_ms, 5)
            b2_backbone["gpu_event_tok_s"] = round(1000.0 / b2_bb_gpu_ms, 3)
            b2_backbone["syn_errors"] = syn_b2_bb

            lib_gpu.qwen25_gpu_reset_kv(eng_gpu)
            b2_e2e = measure_window_energy(
                nvml,
                lambda: float(
                    lib_gpu.qwen25_gpu_bench_decode_sustained(
                        eng_gpu, int(prompt_ids[0]), n_b_e2e, 1, 1, ctypes.byref(last_tok_c)
                    )
                ),
                n_b_e2e,
                "tok",
            )
            syn_b2_e2e = int(lib_gpu.qwen25_gpu_get_syn_errors(eng_gpu))
            b2_e2e_gpu_ms = float(b2_e2e["fn_result"])
            b2_e2e["gpu_event_ms_per_tok"] = round(b2_e2e_gpu_ms, 5)
            b2_e2e["gpu_event_tok_s"] = round(1000.0 / b2_e2e_gpu_ms, 3)
            b2_e2e["syn_errors"] = syn_b2_e2e

            # B3: Batched B=32 cp.async + IMMA across all 168 real matrices!
            sample_y = np.empty((32, QWEN_H), dtype=np.float32)
            cal_imma_ms = float(
                lib_gpu.qwen25_gpu_bench_b32_imma_all168(
                    eng_gpu,
                    x_q8_h.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
                    x_sc_h.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    x_q8_i.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
                    x_sc_i.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    sample_y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    10,
                )
            )
            imma_rel_l2 = rel_l2(sample_y, imma_ref_y_f64)
            imma_max_abs = float(np.max(np.abs(sample_y.astype(np.float64) - imma_ref_y_f64)))
            if imma_rel_l2 > 1e-4:
                raise RuntimeError(f"IMMA B=32 accuracy check failed: rel_l2={imma_rel_l2}")

            n_b3 = max(40, min(2000, int(math.ceil(2600.0 / max(0.5, cal_imma_ms)))))
            b3_imma = measure_window_energy(
                nvml,
                lambda: float(
                    lib_gpu.qwen25_gpu_bench_b32_imma_all168(
                        eng_gpu,
                        x_q8_h.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
                        x_sc_h.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        x_q8_i.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
                        x_sc_i.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        None,
                        n_b3,
                    )
                ),
                n_b3 * 32,
                "tok",
            )
            b3_batch_ms = float(b3_imma["fn_result"])
            b3_imma["gpu_event_ms_per_b32_batch"] = round(b3_batch_ms, 5)
            b3_imma["gpu_event_prefill_tok_s"] = round(32000.0 / b3_batch_ms, 3)
            b3_imma["int8_tops"] = round((2.0 * 32.0 * TOTAL_BLOCK_WEIGHTS) / (b3_batch_ms * 1e9), 3)
            b3_imma["l0_q_proj_rel_l2_vs_fp64"] = round(imma_rel_l2, 9)
            b3_imma["l0_q_proj_max_abs_vs_fp64"] = round(imma_max_abs, 9)
        finally:
            lib_gpu.qwen25_gpu_destroy(eng_gpu)

        # Free CUDA resources & destroy primary context so RTX 3080 drops to true idle P8 before Arm C!
        del cold_read_np, cold_read_buf
        gc.collect()
        torch.cuda.empty_cache()
        lib_gpu.qwen25_gpu_reset_device()

        # ===================================================================
        # ARM C: Clean Zero-CUDA Subprocess on Intel i9-11900H AVX-512 (8C)
        # ===================================================================
        shm_base = pathlib.Path("/dev/shm") if pathlib.Path("/dev/shm").is_dir() else pathlib.Path("/tmp")
        fd_shm, shm_path_str = tempfile.mkstemp(prefix="s58_armc_18b_", suffix=".bin", dir=str(shm_base))
        tmp_shm_18b = pathlib.Path(shm_path_str)
        os.chmod(tmp_shm_18b, 0o600)
        with os.fdopen(fd_shm, "wb") as fh:
            fh.write(disk_18b_bytes)

        tmp_parity_dir = pathlib.Path(tempfile.mkdtemp(prefix="s58_armc_par_", dir=str(shm_base)))
        os.chmod(tmp_parity_dir, 0o700)
        np.save(tmp_parity_dir / "bf16_logits_first.npy", bf16_logits_first)
        np.save(tmp_parity_dir / "gpu_a16_logits_first.npy", gpu_logits_first)
        (tmp_parity_dir / "parity_meta.json").write_text(
            json.dumps({"prompt_ids": prompt_ids, "gpu_a16_gen_tokens": gpu_gen_tokens}),
            encoding="utf-8",
        )
        tmp_arm_c_json = tmp_parity_dir / "arm_c_out.json"

        # Wait briefly for GPU to drop to P8 before launching Arm C
        nvml.close()
        time.sleep(1.5)

        child_env = dict(os.environ)
        child_env["CUDA_VISIBLE_DEVICES"] = ""
        child_env["OMP_NUM_THREADS"] = str(cpu_threads)
        child_env["OMP_PROC_BIND"] = "spread"
        child_env["OMP_PLACES"] = "cores"
        child_env["PYTHONDONTWRITEBYTECODE"] = "1"

        cmd = [
            sys.executable,
            "-B",
            str(pathlib.Path(__file__).resolve()),
            "--phase",
            "arm_c",
            "--model-dir",
            str(model_dir),
            "--avx512-so",
            str(avx512_so),
            "--disk-18b-path",
            str(tmp_shm_18b),
            "--expected-disk-sha",
            disk_18b_sha,
            "--parity-dir",
            str(tmp_parity_dir),
            "--cpu-threads",
            str(cpu_threads),
            "--out-json",
            str(tmp_arm_c_json),
        ]
        res = subprocess.run(cmd, env=child_env, capture_output=True, text=True, check=False)
        if res.returncode != 0:
            raise RuntimeError(
                f"Arm C subprocess failed rc={res.returncode}\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
            )
        arm_c = json.loads(tmp_arm_c_json.read_text(encoding="utf-8"))

        # ===================================================================
        # Compute Exact Latency-Energy (mJ/tok) Pareto Summary
        # ===================================================================
        c1_bb = arm_c["c1_pure_vnni"]["backbone_24layers"]
        c1_e2e = arm_c["c1_pure_vnni"]["full_e2e_with_bf16_lm_head"]
        c2_bb = arm_c["c2_live_vpopcntq_verify"]["backbone_24layers"]
        c2_e2e = arm_c["c2_live_vpopcntq_verify"]["full_e2e_with_bf16_lm_head"]

        pareto = {
            "decode_backbone_24layers": {
                "arm_a_bf16_gpu_tok_s": a1_backbone["units_per_s"],
                "arm_b1_a16_sm86_tok_s": b1_backbone["gpu_event_tok_s"],
                "arm_b2_a16_sm86_live_verify_tok_s": b2_backbone["gpu_event_tok_s"],
                "arm_c1_a16_avx512_8t_tok_s": c1_bb["units_per_s"],
                "arm_c2_a16_avx512_8t_live_verify_tok_s": c2_bb["units_per_s"],
                "speedup_b1_vs_a_bf16": round(b1_backbone["gpu_event_tok_s"] / a1_backbone["units_per_s"], 3),
                "speedup_b1_vs_c1_avx512": round(b1_backbone["gpu_event_tok_s"] / c1_bb["units_per_s"], 3),
                "arm_a_bf16_combined_mj_per_tok": a1_backbone["combined_gpu_plus_cpu_pkg_mj_per_unit"],
                "arm_b1_a16_sm86_combined_mj_per_tok": b1_backbone["combined_gpu_plus_cpu_pkg_mj_per_unit"],
                "arm_b2_a16_sm86_verify_combined_mj_per_tok": b2_backbone["combined_gpu_plus_cpu_pkg_mj_per_unit"],
                "arm_c1_a16_avx512_cpu_pkg_mj_per_tok": c1_bb["cpu_pkg_mj_per_unit"],
                "arm_c1_a16_avx512_psys_mj_per_tok": c1_bb["psys_mj_per_unit"],
                "arm_b1_a16_sm86_psys_mj_per_tok": b1_backbone["psys_mj_per_unit"],
                "energy_ratio_a_bf16_over_b1_a16_combined": round(
                    a1_backbone["combined_gpu_plus_cpu_pkg_mj_per_unit"]
                    / max(1e-9, b1_backbone["combined_gpu_plus_cpu_pkg_mj_per_unit"]),
                    3,
                ),
                "energy_ratio_c1_cpu_pkg_over_b1_combined": round(
                    c1_bb["cpu_pkg_mj_per_unit"]
                    / max(1e-9, b1_backbone["combined_gpu_plus_cpu_pkg_mj_per_unit"]),
                    3,
                ),
            },
            "decode_full_e2e_with_bf16_lm_head": {
                "arm_a_bf16_gpu_tok_s": a2_e2e["units_per_s"],
                "arm_b1_a16_sm86_tok_s": b1_e2e["gpu_event_tok_s"],
                "arm_b2_a16_sm86_live_verify_tok_s": b2_e2e["gpu_event_tok_s"],
                "arm_c1_a16_avx512_8t_tok_s": c1_e2e["units_per_s"],
                "arm_c2_a16_avx512_8t_live_verify_tok_s": c2_e2e["units_per_s"],
                "speedup_b1_vs_a_bf16": round(b1_e2e["gpu_event_tok_s"] / a2_e2e["units_per_s"], 3),
                "speedup_b1_vs_c1_avx512": round(b1_e2e["gpu_event_tok_s"] / c1_e2e["units_per_s"], 3),
                "arm_a_bf16_combined_mj_per_tok": a2_e2e["combined_gpu_plus_cpu_pkg_mj_per_unit"],
                "arm_b1_a16_sm86_combined_mj_per_tok": b1_e2e["combined_gpu_plus_cpu_pkg_mj_per_unit"],
                "arm_b2_a16_sm86_verify_combined_mj_per_tok": b2_e2e["combined_gpu_plus_cpu_pkg_mj_per_unit"],
                "arm_c1_a16_avx512_cpu_pkg_mj_per_tok": c1_e2e["cpu_pkg_mj_per_unit"],
                "arm_c1_a16_avx512_psys_mj_per_tok": c1_e2e["psys_mj_per_unit"],
                "arm_b1_a16_sm86_psys_mj_per_tok": b1_e2e["psys_mj_per_unit"],
                "energy_ratio_a_bf16_over_b1_a16_combined": round(
                    a2_e2e["combined_gpu_plus_cpu_pkg_mj_per_unit"]
                    / max(1e-9, b1_e2e["combined_gpu_plus_cpu_pkg_mj_per_unit"]),
                    3,
                ),
            },
            "prefill_b32_all168_linear": {
                "arm_a3_bf16_pytorch_tok_s": a3_b32["units_per_s"],
                "arm_b3_a16_imma_tok_s": b3_imma["gpu_event_prefill_tok_s"],
                "speedup_b3_vs_a3": round(b3_imma["gpu_event_prefill_tok_s"] / a3_b32["units_per_s"], 3),
                "arm_a3_bf16_combined_mj_per_tok": a3_b32["combined_gpu_plus_cpu_pkg_mj_per_unit"],
                "arm_b3_a16_imma_combined_mj_per_tok": b3_imma["combined_gpu_plus_cpu_pkg_mj_per_unit"],
            },
        }

        doc = {
            "schema_version": "s58.qwen25_live_llm_empirical.v1",
            "provenance_tier": "sampled",
            "measured_live_model_weights": True,
            "model_identity": {
                "repo_id": "Qwen/Qwen2.5-0.5B",
                "snapshot_id": model_dir.name,
                "model_safetensors_sha256": model_safetensors_sha,
                "config_json_sha256": config_sha,
                "tokenizer_json_sha256": tokenizer_sha,
                "num_hidden_layers": QWEN_LAYERS,
                "hidden_size": QWEN_H,
                "intermediate_size": QWEN_I,
                "num_attention_heads": 14,
                "num_key_value_heads": 2,
                "vocab_size": QWEN_VOCAB,
                "linear_matrix_count": TOTAL_LINEAR_MATRICES,
                "block_weight_count": TOTAL_BLOCK_WEIGHTS,
                "groups_32_count": TOTAL_GROUPS_32,
                "subblocks_16_count": TOTAL_SUBBLOCKS_16,
                "nonblock_bf16_tensor_count": TOTAL_NONBLOCK_BF16_TENSORS,
                "cold_18b_bytes": TOTAL_COLD_18B_BYTES,
                "bf16_linear_bytes": TOTAL_BLOCK_WEIGHTS * 2,
                "quantization_wall_s": round(quant_wall_s, 3),
                "quantization_syndrome_errors": total_quant_syn_err,
                "aggregate_weight_rel_l2": round(agg_weight_rel_l2, 6),
                "max_matrix_weight_rel_l2": round(max_mat_rel_l2, 6),
            },
            "hardware_telemetry_method": {
                "gpu_name": gpu_name,
                "compute_capability": f"{cc_major}.{cc_minor}",
                "gpu_energy_method": "nvmlDeviceGetPowerUsage_100hz_trapezoidal_mj",
                "nvml_total_energy_counter_note": (
                    "Rejected GA104M nvmlDeviceGetTotalEnergyConsumption due to >16x "
                    "idle counter drift vs INA3221 rail nvmlDeviceGetPowerUsage."
                ),
                "cpu_energy_method": "linux_powercap_intel_rapl_microjoules",
            },
            "arm_a_pytorch_bf16_gpu": {
                "a1_backbone_24layers": a1_backbone,
                "a2_full_e2e_with_bf16_lm_head": a2_e2e,
                "a3_prefill_b32_all168": a3_b32,
            },
            "arm_b_sm86_ptx_a16_gpu": {
                "b1_pure_dp4a": {
                    "backbone_24layers": b1_backbone,
                    "full_e2e_with_bf16_lm_head": b1_e2e,
                },
                "b2_live_5popc_verify": {
                    "backbone_24layers": b2_backbone,
                    "full_e2e_with_bf16_lm_head": b2_e2e,
                    "verify_overhead_backbone_pct": round(
                        100.0 * (b2_bb_gpu_ms - b1_bb_gpu_ms) / b1_bb_gpu_ms, 3
                    ),
                    "verify_overhead_e2e_pct": round(
                        100.0 * (b2_e2e_gpu_ms - b1_e2e_gpu_ms) / b1_e2e_gpu_ms, 3
                    ),
                },
                "b3_prefill_b32_cp_async_imma": b3_imma,
            },
            "arm_c_avx512_cpu_only_a16": arm_c,
            "arm_d_cold_18b_transcode_and_128tok_gen": arm_d,
            "pareto_latency_energy_summary": pareto,
        }
        out_json.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(pareto, indent=2, sort_keys=True))
    finally:
        nvml.close()
        if tmp_cold_18b is not None and tmp_cold_18b.exists():
            tmp_cold_18b.unlink()
        if tmp_shm_18b is not None and tmp_shm_18b.exists():
            tmp_shm_18b.unlink()
        if tmp_parity_dir is not None and tmp_parity_dir.exists():
            for child in tmp_parity_dir.iterdir():
                child.unlink()
            tmp_parity_dir.rmdir()


def main() -> int:
    ap = argparse.ArgumentParser(description="Track S58 Real-Weight Qwen2.5-0.5B A_16 Live Benchmark")
    ap.add_argument("--phase", choices=("all", "arm_c"), default="all")
    ap.add_argument("--model-dir", type=pathlib.Path, required=True)
    ap.add_argument("--sm86-so", type=pathlib.Path)
    ap.add_argument("--avx512-so", type=pathlib.Path, required=True)
    ap.add_argument("--out-json", type=pathlib.Path, required=True)
    ap.add_argument("--cpu-threads", type=int, default=8)
    ap.add_argument("--disk-18b-path", type=pathlib.Path)
    ap.add_argument("--expected-disk-sha", type=str, default="")
    ap.add_argument("--parity-dir", type=pathlib.Path)
    args = ap.parse_args()

    if args.phase == "arm_c":
        if args.disk_18b_path is None or args.parity_dir is None:
            raise SystemExit("--disk-18b-path and --parity-dir required for --phase arm_c")
        run_arm_c_subprocess(
            model_dir=args.model_dir,
            avx512_so=args.avx512_so,
            disk_18b_path=args.disk_18b_path,
            expected_disk_sha=args.expected_disk_sha,
            parity_dir=args.parity_dir,
            num_threads=args.cpu_threads,
            out_json=args.out_json,
        )
        return 0

    if args.sm86_so is None:
        raise SystemExit("--sm86-so required for --phase all")
    run_all_arms(
        model_dir=args.model_dir,
        sm86_so=args.sm86_so,
        avx512_so=args.avx512_so,
        out_json=args.out_json,
        cpu_threads=args.cpu_threads,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
