"""Drop-in Hugging Face CausalLM & bit-packed `A_16` lattice decoder (`siliquant-hf-v1`).

Zero extra dependencies beyond `torch` and `transformers`.
Supports `AutoModelForCausalLM.from_pretrained(repo_or_dir, trust_remote_code=True, ...)`
on CPU, CUDA, MPS, and ROCm with:
  - Exact Extended Hamming [16, 11, 4] lattice decomposition `y = 2*k + c`,
    where `c in C_{[16, 11, 4]}` is stored as 11 systematic information bits per 16-block
    (0.6875 bpw) and `k` is packed at per-layer minimum-byte width `k_bits in {2, 3, 4}`
    plus sparse exact CSR overflow `(qov_rowptr, qov_col, qov_val)`.
    Note: because `qham11` stores the 11 systematic bits and reconstructs the 5 parity checks
    algebraically, `verify_syndrome=True` enforces codec/lattice consistency on unpack, while
    end-to-end payload tamper integrity is provided by `siliquant_verification_receipt.json`
    SHA-256 verification.
  - Exact integer bit-pattern storage for FP16 group scales (`scales_fp16_bits: int16`)
    and optional FP32 column scales (`colscale_fp32_bits: int32`), making packed layers
    completely immune to `.to(dtype)` / `from_pretrained(..., dtype=torch.bfloat16)`
    mantissa truncation (`max_abs_roundtrip_diff == 0.0` unconditionally).
  - Exact inverse block Hadamard (`axis`, `block_h2`, `block_h4`, `h16`), optional
    column scale, and optional input permutation (`perm_in`).
  - Full `.forward()`, `.generate()`, KV-cache, tied/untied embeddings, and optional
    cached vs on-the-fly dequantization.
"""

from __future__ import annotations

import inspect
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationMixin, LlamaModel, PreTrainedModel, Qwen2Model
from transformers.modeling_outputs import CausalLMOutputWithPast

try:
    from .configuration_siliquant import (
        SUPPORTED_K_BITS,
        SUPPORTED_ROTATIONS,
        SiliquantConfig,
    )
except ImportError:
    from configuration_siliquant import (
        SUPPORTED_K_BITS,
        SUPPORTED_ROTATIONS,
        SiliquantConfig,
    )

# Compatibility detection between transformers 5.x (`cache=`) and 4.x (`past_key_values=`).
_CAUSAL_LM_OUTPUT_PARAMS = inspect.signature(CausalLMOutputWithPast).parameters
_USES_CACHE_FIELD = "cache" in _CAUSAL_LM_OUTPUT_PARAMS
_QWEN2_FWD_PARAMS = inspect.signature(Qwen2Model.forward).parameters
_MODEL_USES_CACHE_ARG = "cache" in _QWEN2_FWD_PARAMS

# Extended Hamming [16, 11, 4] information bit positions (parity bits sit at {0, 1, 2, 4, 8}).
A16_INFO_POSITIONS: tuple[int, ...] = (3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15)
_INT_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


def _validate_perm_in(perm_in: torch.Tensor, in_features: int) -> torch.Tensor:
    """Validate that `perm_in` is a 1D integer bijection on `[0, in_features)` and return `long`."""
    if not isinstance(perm_in, torch.Tensor):
        raise ValueError(f"perm_in must be a torch.Tensor, got {type(perm_in).__name__}")
    if perm_in.ndim != 1 or int(perm_in.shape[0]) != int(in_features):
        raise ValueError(
            f"perm_in shape mismatch: got {tuple(perm_in.shape)}, expected ({in_features},)"
        )
    if perm_in.dtype not in _INT_DTYPES:
        raise ValueError(f"perm_in must have an integer dtype, got {perm_in.dtype}")
    p = perm_in.to(torch.long)
    if in_features == 0:
        return p
    if not bool(((p >= 0) & (p < in_features)).all().item()):
        raise ValueError(f"perm_in contains out-of-range indices outside [0, {in_features})")
    expected = torch.arange(in_features, device=p.device, dtype=torch.long)
    if not torch.equal(torch.sort(p).values, expected):
        raise ValueError("perm_in is not a valid bijection on 0..in_features-1 (duplicate indices)")
    return p


def check_a16_syndrome_errors(y_int: torch.Tensor) -> int:
    """Count 16-element blocks violating Extended Hamming [16, 11, 4] parity checks."""
    if y_int.numel() == 0:
        return 0
    if y_int.shape[-1] % 16 != 0:
        raise ValueError(f"Last dimension must be divisible by 16, got {y_int.shape[-1]}")
    bits = (y_int.reshape(-1, 16) & 1).to(torch.int32)
    p0 = bits.sum(dim=1) & 1
    s0 = bits[:, 1::2].sum(dim=1) & 1
    s1 = (bits[:, 2::4].sum(dim=1) + bits[:, 3::4].sum(dim=1)) & 1
    s2 = (bits[:, 4:8].sum(dim=1) + bits[:, 12:16].sum(dim=1)) & 1
    s3 = bits[:, 8:16].sum(dim=1) & 1
    return int(((p0 | s0 | s1 | s2 | s3) != 0).sum().item())


def _pack_hamming11(bits16: torch.Tensor) -> torch.Tensor:
    """Pack `N_blk x 16` valid [16, 11, 4] parity bits into 11 bits/block (`uint8`)."""
    n_blk = int(bits16.shape[0])
    dev = bits16.device
    info_idx = torch.tensor(A16_INFO_POSITIONS, device=dev, dtype=torch.long)
    info = bits16.index_select(1, info_idx).to(torch.int64)  # [N_blk, 11]
    shifts = torch.arange(11, device=dev, dtype=torch.int64)
    u11 = (info << shifts.unsqueeze(0)).sum(dim=1)  # [N_blk], values in [0, 2047]

    rem = n_blk % 8
    if rem != 0:
        pad = torch.zeros(8 - rem, device=dev, dtype=torch.int64)
        u11 = torch.cat([u11, pad], dim=0)
    grp = u11.reshape(-1, 8)  # [G, 8]
    lo44 = grp[:, 0] | (grp[:, 1] << 11) | (grp[:, 2] << 22) | (grp[:, 3] << 33)
    hi44 = grp[:, 4] | (grp[:, 5] << 11) | (grp[:, 6] << 22) | (grp[:, 7] << 33)

    out = torch.empty((grp.shape[0], 11), device=dev, dtype=torch.uint8)
    out[:, 0] = (lo44 & 0xFF).to(torch.uint8)
    out[:, 1] = ((lo44 >> 8) & 0xFF).to(torch.uint8)
    out[:, 2] = ((lo44 >> 16) & 0xFF).to(torch.uint8)
    out[:, 3] = ((lo44 >> 24) & 0xFF).to(torch.uint8)
    out[:, 4] = ((lo44 >> 32) & 0xFF).to(torch.uint8)
    out[:, 5] = (((lo44 >> 40) & 0x0F) | ((hi44 & 0x0F) << 4)).to(torch.uint8)
    out[:, 6] = ((hi44 >> 4) & 0xFF).to(torch.uint8)
    out[:, 7] = ((hi44 >> 12) & 0xFF).to(torch.uint8)
    out[:, 8] = ((hi44 >> 20) & 0xFF).to(torch.uint8)
    out[:, 9] = ((hi44 >> 28) & 0xFF).to(torch.uint8)
    out[:, 10] = ((hi44 >> 36) & 0xFF).to(torch.uint8)
    return out.reshape(-1)


def _unpack_hamming11(qham11: torch.Tensor, n_blk: int) -> torch.Tensor:
    """Unpack 11-bit Hamming info stream back into exact `[n_blk, 16]` `int16` codewords."""
    if qham11.dtype != torch.uint8:
        raise ValueError(f"qham11 must have dtype uint8, got {qham11.dtype}")
    if qham11.ndim != 1:
        raise ValueError(f"qham11 must be 1D, got shape {tuple(qham11.shape)}")
    n_groups = (n_blk + 7) // 8
    expected_bytes = n_groups * 11
    if int(qham11.numel()) != expected_bytes:
        raise ValueError(
            f"qham11 byte length mismatch: got {qham11.numel()}, expected {expected_bytes}"
        )
    dev = qham11.device
    raw = qham11.reshape(n_groups, 11).to(torch.int64)
    lo44 = (
        raw[:, 0]
        | (raw[:, 1] << 8)
        | (raw[:, 2] << 16)
        | (raw[:, 3] << 24)
        | (raw[:, 4] << 32)
        | ((raw[:, 5] & 0x0F) << 40)
    )
    hi44 = (
        ((raw[:, 5] >> 4) & 0x0F)
        | (raw[:, 6] << 4)
        | (raw[:, 7] << 12)
        | (raw[:, 8] << 20)
        | (raw[:, 9] << 28)
        | (raw[:, 10] << 36)
    )
    u11 = torch.stack(
        [
            lo44 & 0x7FF,
            (lo44 >> 11) & 0x7FF,
            (lo44 >> 22) & 0x7FF,
            (lo44 >> 33) & 0x7FF,
            hi44 & 0x7FF,
            (hi44 >> 11) & 0x7FF,
            (hi44 >> 22) & 0x7FF,
            (hi44 >> 33) & 0x7FF,
        ],
        dim=1,
    ).reshape(-1)

    if u11.shape[0] > n_blk:
        if bool((u11[n_blk:] != 0).any().item()):
            raise ValueError("Non-zero padding bits detected in qham11 stream")
        u11 = u11[:n_blk]

    # Unpack 11 info bits into positions (3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15).
    b3 = (u11 >> 0) & 1
    b5 = (u11 >> 1) & 1
    b6 = (u11 >> 2) & 1
    b7 = (u11 >> 3) & 1
    b9 = (u11 >> 4) & 1
    b10 = (u11 >> 5) & 1
    b11 = (u11 >> 6) & 1
    b12 = (u11 >> 7) & 1
    b13 = (u11 >> 8) & 1
    b14 = (u11 >> 9) & 1
    b15 = (u11 >> 10) & 1

    # Reconstruct the 5 parity check bits {1, 2, 4, 8, 0} by exact XOR.
    b1 = b3 ^ b5 ^ b7 ^ b9 ^ b11 ^ b13 ^ b15
    b2 = b3 ^ b6 ^ b7 ^ b10 ^ b11 ^ b14 ^ b15
    b4 = b5 ^ b6 ^ b7 ^ b12 ^ b13 ^ b14 ^ b15
    b8 = b9 ^ b10 ^ b11 ^ b12 ^ b13 ^ b14 ^ b15
    b0 = b1 ^ b2 ^ b3 ^ b4 ^ b5 ^ b6 ^ b7 ^ b8 ^ b9 ^ b10 ^ b11 ^ b12 ^ b13 ^ b14 ^ b15

    return torch.stack(
        [b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15],
        dim=1,
    ).to(torch.int16)


def _pack_signed_quotient(k_int: torch.Tensor, k_bits: int) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Pack integer quotient `k [M, K]` into `k_bits in {2, 3, 4}` + sparse CSR overflow."""
    if k_bits not in SUPPORTED_K_BITS:
        raise ValueError(f"Unsupported k_bits={k_bits}; expected one of {SUPPORTED_K_BITS}")
    if k_int.ndim != 2:
        raise ValueError(f"Expected 2D quotient tensor, got shape {tuple(k_int.shape)}")
    m, k_dim = int(k_int.shape[0]), int(k_int.shape[1])
    if k_dim % 16 != 0:
        raise ValueError(f"in_features K={k_dim} must be a multiple of 16")

    lo_val = -(1 << (k_bits - 1))
    hi_val = (1 << (k_bits - 1)) - 1
    ov_mask = (k_int < lo_val) | (k_int > hi_val)
    k_base = torch.where(ov_mask, torch.zeros_like(k_int), k_int)
    u = (k_base - lo_val).to(torch.int32)

    if k_bits == 4:
        qquot = (u[:, 0::2] | (u[:, 1::2] << 4)).to(torch.uint8)
    elif k_bits == 2:
        qquot = (
            u[:, 0::4] | (u[:, 1::4] << 2) | (u[:, 2::4] << 4) | (u[:, 3::4] << 6)
        ).to(torch.uint8)
    else:  # k_bits == 3: 8 values -> 24 bits = 3 bytes
        ug = u.reshape(m, k_dim // 8, 8)
        v24 = (
            ug[:, :, 0]
            | (ug[:, :, 1] << 3)
            | (ug[:, :, 2] << 6)
            | (ug[:, :, 3] << 9)
            | (ug[:, :, 4] << 12)
            | (ug[:, :, 5] << 15)
            | (ug[:, :, 6] << 18)
            | (ug[:, :, 7] << 21)
        )
        b0 = (v24 & 0xFF).to(torch.uint8)
        b1 = ((v24 >> 8) & 0xFF).to(torch.uint8)
        b2 = ((v24 >> 16) & 0xFF).to(torch.uint8)
        qquot = torch.stack([b0, b1, b2], dim=2).reshape(m, (k_dim // 8) * 3)

    n_ov = int(ov_mask.sum().item())
    if n_ov == 0:
        return qquot, None, None, None

    row_counts = ov_mask.sum(dim=1, dtype=torch.int32)
    rowptr = torch.zeros(m + 1, device=k_int.device, dtype=torch.int32)
    rowptr[1:] = torch.cumsum(row_counts, dim=0, dtype=torch.int32)
    nz = torch.nonzero(ov_mask, as_tuple=False)
    ov_cols_i64 = nz[:, 1]
    ov_vals_i32 = k_int[nz[:, 0], nz[:, 1]].to(torch.int32)

    col_dtype = torch.int16 if k_dim <= 32767 else torch.int32
    qov_col = ov_cols_i64.to(col_dtype)

    max_abs_val = int(ov_vals_i32.abs().max().item())
    if max_abs_val <= 127:
        qov_val = ov_vals_i32.to(torch.int8)
    elif max_abs_val <= 32767:
        qov_val = ov_vals_i32.to(torch.int16)
    else:
        raise ValueError(f"Overflow quotient magnitude {max_abs_val} exceeds int16 range")

    return qquot, rowptr, qov_col, qov_val


def _unpack_signed_quotient(
    qquot: torch.Tensor,
    m: int,
    k_dim: int,
    k_bits: int,
    qov_rowptr: torch.Tensor | None = None,
    qov_col: torch.Tensor | None = None,
    qov_val: torch.Tensor | None = None,
) -> torch.Tensor:
    """Unpack `qquot` + optional CSR overflow back into exact `[M, K]` `int16` quotients."""
    if k_bits not in SUPPORTED_K_BITS:
        raise ValueError(f"Unsupported k_bits={k_bits}")
    if m <= 0 or k_dim <= 0 or (k_dim % 16) != 0:
        raise ValueError(f"Invalid dimensions (m={m}, k_dim={k_dim}); require m>0, k_dim>0, k_dim%16==0")
    if qquot.dtype != torch.uint8:
        raise ValueError(f"qquot must have dtype uint8, got {qquot.dtype}")
    if qquot.ndim != 2:
        raise ValueError(f"qquot must be 2D, got shape {tuple(qquot.shape)}")

    csr_flags = (qov_rowptr is not None, qov_col is not None, qov_val is not None)
    if any(csr_flags) and not all(csr_flags):
        raise ValueError(
            "Partial CSR overflow buffers provided; qov_rowptr, qov_col, and qov_val must be all present or all None"
        )

    lo_val = -(1 << (k_bits - 1))
    dev = qquot.device

    if k_bits == 4:
        expected = (m, k_dim // 2)
        if tuple(qquot.shape) != expected:
            raise ValueError(f"qquot shape mismatch: got {tuple(qquot.shape)}, expected {expected}")
        raw = qquot.to(torch.int16)
        u = torch.empty((m, k_dim), device=dev, dtype=torch.int16)
        u[:, 0::2] = raw & 0x0F
        u[:, 1::2] = (raw >> 4) & 0x0F
    elif k_bits == 2:
        expected = (m, k_dim // 4)
        if tuple(qquot.shape) != expected:
            raise ValueError(f"qquot shape mismatch: got {tuple(qquot.shape)}, expected {expected}")
        raw = qquot.to(torch.int16)
        u = torch.empty((m, k_dim), device=dev, dtype=torch.int16)
        u[:, 0::4] = raw & 0x03
        u[:, 1::4] = (raw >> 2) & 0x03
        u[:, 2::4] = (raw >> 4) & 0x03
        u[:, 3::4] = (raw >> 6) & 0x03
    else:  # k_bits == 3
        expected = (m, (k_dim // 8) * 3)
        if tuple(qquot.shape) != expected:
            raise ValueError(f"qquot shape mismatch: got {tuple(qquot.shape)}, expected {expected}")
        raw = qquot.reshape(m, k_dim // 8, 3).to(torch.int32)
        v24 = raw[:, :, 0] | (raw[:, :, 1] << 8) | (raw[:, :, 2] << 16)
        u = torch.stack(
            [(v24 >> (3 * j)) & 0x07 for j in range(8)],
            dim=2,
        ).reshape(m, k_dim).to(torch.int16)

    k_int = u + lo_val
    if all(csr_flags):
        if qov_rowptr.dtype not in _INT_DTYPES or qov_col.dtype not in _INT_DTYPES or qov_val.dtype not in _INT_DTYPES:
            raise ValueError("CSR overflow buffers (qov_rowptr, qov_col, qov_val) must have integer dtypes")
        if qov_rowptr.ndim != 1 or int(qov_rowptr.shape[0]) != m + 1:
            raise ValueError(
                f"qov_rowptr shape mismatch: got {tuple(qov_rowptr.shape)}, expected ({m + 1},)"
            )
        if qov_col.ndim != 1 or qov_val.ndim != 1:
            raise ValueError(
                f"qov_col and qov_val must be 1D, got {tuple(qov_col.shape)} and {tuple(qov_val.shape)}"
            )
        if tuple(qov_col.shape) != tuple(qov_val.shape):
            raise ValueError(
                f"qov_col shape {tuple(qov_col.shape)} != qov_val shape {tuple(qov_val.shape)}"
            )
        rp64 = qov_rowptr.to(torch.long)
        if int(rp64[0].item()) != 0:
            raise ValueError(f"qov_rowptr[0] must be 0, got {int(rp64[0].item())}")
        counts = rp64[1:] - rp64[:-1]
        if bool((counts < 0).any().item()):
            raise ValueError("Non-monotonic qov_rowptr detected (negative row count)")
        n_ov = int(qov_col.numel())
        if int(rp64[-1].item()) != n_ov:
            raise ValueError(
                f"qov_rowptr[-1] ({int(rp64[-1].item())}) != qov_col length ({n_ov})"
            )
        if n_ov > 0:
            cols = qov_col.to(torch.long)
            if not bool(((cols >= 0) & (cols < k_dim)).all().item()):
                raise ValueError(f"qov_col contains out-of-range column index outside [0, {k_dim})")
            rows = torch.repeat_interleave(torch.arange(m, device=dev, dtype=torch.long), counts)
            flat_idx = rows * k_dim + cols
            if n_ov > 1:
                sorted_flat = torch.sort(flat_idx).values
                if bool((sorted_flat[1:] == sorted_flat[:-1]).any().item()):
                    raise ValueError("Duplicate (row, col) entry detected in CSR overflow buffers")
            vals_i32 = qov_val.to(torch.int32)
            if bool(((vals_i32 < -32768) | (vals_i32 > 32767)).any().item()):
                raise ValueError("qov_val contains value outside int16 range [-32768, 32767]")
            k_int[rows, cols] = vals_i32.to(torch.int16)
    return k_int


def pack_a16_tensor(
    y_int: torch.Tensor,
    scales_fp16: torch.Tensor,
    *,
    group_size: int,
    rotation: str = "axis",
    colscale_f32: torch.Tensor | None = None,
    perm_in: torch.Tensor | None = None,
    k_bits: int | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Losslessly pack an exact `A_16` integer lattice matrix + metadata into integer buffers.

    Fails closed (`ValueError`) if `y_int` has any Extended Hamming [16, 11, 4] syndrome error.
    When `k_bits is None`, automatically chooses `k_bits in {2, 3, 4}` minimizing exact byte size.
    """
    if rotation not in SUPPORTED_ROTATIONS:
        raise ValueError(f"Unsupported rotation={rotation!r}; expected {SUPPORTED_ROTATIONS}")
    if y_int.ndim != 2:
        raise ValueError(f"Expected 2D y_int tensor, got shape {tuple(y_int.shape)}")
    m, k_dim = int(y_int.shape[0]), int(y_int.shape[1])
    if k_dim % 16 != 0 or k_dim % group_size != 0:
        raise ValueError(
            f"in_features K={k_dim} must be divisible by 16 and group_size={group_size}"
        )

    y_i32 = y_int.to(torch.int32)
    syn_errs = check_a16_syndrome_errors(y_i32)
    if syn_errs != 0:
        raise ValueError(
            f"Refusing to pack invalid A_16 lattice tensor: {syn_errs} Extended Hamming [16,11,4] syndrome errors"
        )

    c_bits = (y_i32 & 1).reshape(-1, 16)
    k_int = (y_i32 - (y_i32 & 1)) >> 1

    # Select exact minimum-byte k_bits if not fixed.
    if k_bits is None:
        best_kb = 4
        best_bytes = -1
        col_bytes = 2 if k_dim <= 32767 else 4
        for kb in SUPPORTED_K_BITS:
            lo_v = -(1 << (kb - 1))
            hi_v = (1 << (kb - 1)) - 1
            ov_m = (k_int < lo_v) | (k_int > hi_v)
            n_ov = int(ov_m.sum().item())
            base_bytes = m * (k_dim * kb // 8)
            if n_ov > 0:
                max_v = int(k_int[ov_m].abs().max().item())
                val_bytes = 1 if max_v <= 127 else 2
                ov_bytes = 4 * (m + 1) + n_ov * (col_bytes + val_bytes)
            else:
                ov_bytes = 0
            tot_bytes = base_bytes + ov_bytes
            if best_bytes < 0 or tot_bytes < best_bytes:
                best_bytes = tot_bytes
                best_kb = kb
        chosen_kb = best_kb
    else:
        chosen_kb = int(k_bits)

    qham11 = _pack_hamming11(c_bits)
    qquot, qov_rowptr, qov_col, qov_val = _pack_signed_quotient(k_int, chosen_kb)

    sc_f16 = scales_fp16.reshape(m, k_dim // group_size).to(torch.float16).contiguous()
    if not bool(torch.isfinite(sc_f16).all().item()):
        raise ValueError("Refusing to pack non-finite scales_fp16 tensor")
    scales_fp16_bits = sc_f16.view(torch.int16).clone()

    buffers: dict[str, torch.Tensor] = {
        "qquot": qquot.contiguous(),
        "qham11": qham11.contiguous(),
        "scales_fp16_bits": scales_fp16_bits.contiguous(),
    }
    n_overflow = 0
    ov_col_dtype = "int16"
    ov_val_dtype = "int8"
    if qov_rowptr is not None and qov_col is not None and qov_val is not None:
        n_overflow = int(qov_col.numel())
        ov_col_dtype = "int16" if qov_col.dtype == torch.int16 else "int32"
        ov_val_dtype = "int8" if qov_val.dtype == torch.int8 else "int16"
        buffers["qov_rowptr"] = qov_rowptr.contiguous()
        buffers["qov_col"] = qov_col.contiguous()
        buffers["qov_val"] = qov_val.contiguous()

    has_colscale = colscale_f32 is not None
    if colscale_f32 is not None:
        if colscale_f32.ndim != 1 or int(colscale_f32.shape[0]) != k_dim:
            raise ValueError(
                f"colscale_f32 shape mismatch: got {tuple(colscale_f32.shape)}, expected ({k_dim},)"
            )
        cs = colscale_f32.to(torch.float32).contiguous()
        if not bool(torch.isfinite(cs).all().item()) or bool((cs == 0).any().item()):
            raise ValueError("Refusing to pack non-finite or zero colscale_f32 values")
        buffers["colscale_fp32_bits"] = cs.view(torch.int32).clone().contiguous()

    has_perm_in = perm_in is not None
    if perm_in is not None:
        valid_p = _validate_perm_in(perm_in, k_dim)
        buffers["perm_in"] = valid_p.to(torch.int32).contiguous()

    layer_meta: dict[str, Any] = {
        "in_features": k_dim,
        "out_features": m,
        "group_size": int(group_size),
        "lattice": "A16",
        "rotation": str(rotation),
        "k_bits": int(chosen_kb),
        "n_overflow": int(n_overflow),
        "ov_col_dtype": ov_col_dtype,
        "ov_val_dtype": ov_val_dtype,
        "has_colscale": bool(has_colscale),
        "has_perm_in": bool(has_perm_in),
    }
    return buffers, layer_meta


def _sylvester_hadamard_f32(d: int, device: torch.device) -> torch.Tensor:
    """Return normalized Sylvester Hadamard matrix `H_d / sqrt(d)` in `float32`."""
    h = torch.tensor([[1.0]], device=device, dtype=torch.float32)
    while int(h.shape[0]) < d:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    return h / math.sqrt(float(d))


def unpack_a16_integers(
    qquot: torch.Tensor,
    qham11: torch.Tensor,
    *,
    out_features: int,
    in_features: int,
    k_bits: int,
    qov_rowptr: torch.Tensor | None = None,
    qov_col: torch.Tensor | None = None,
    qov_val: torch.Tensor | None = None,
    verify_syndrome: bool = True,
) -> tuple[torch.Tensor, int]:
    """Unpack exact integer lattice matrix `y_int = 2*k + c` (`int32 [M, K]`, no int16 wrap)."""
    if out_features <= 0 or in_features <= 0 or (in_features % 16) != 0:
        raise ValueError(
            f"Invalid dimensions (out_features={out_features}, in_features={in_features}); "
            "require positive dims with in_features % 16 == 0"
        )
    n_blk = out_features * (in_features // 16)
    c_bits = _unpack_hamming11(qham11, n_blk).reshape(out_features, in_features)
    k_int = _unpack_signed_quotient(
        qquot, out_features, in_features, k_bits, qov_rowptr, qov_col, qov_val
    )
    y_int = (k_int.to(torch.int32) << 1) + c_bits.to(torch.int32)
    syn_errs = check_a16_syndrome_errors(y_int) if verify_syndrome else 0
    if verify_syndrome and syn_errs != 0:
        raise ValueError(f"Decoded A_16 tensor failed syndrome check ({syn_errs} errors)")
    return y_int, syn_errs


def dequantize_a16_weight(
    qquot: torch.Tensor,
    qham11: torch.Tensor,
    scales_fp16_bits: torch.Tensor,
    *,
    out_features: int,
    in_features: int,
    group_size: int,
    rotation: str,
    k_bits: int,
    qov_rowptr: torch.Tensor | None = None,
    qov_col: torch.Tensor | None = None,
    qov_val: torch.Tensor | None = None,
    colscale_fp32_bits: torch.Tensor | None = None,
    perm_in: torch.Tensor | None = None,
    verify_syndrome: bool = True,
    target_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, int]:
    """Exact bit-for-bit dequantization of a packed `A_16` weight matrix."""
    if group_size <= 0 or (in_features % group_size) != 0:
        raise ValueError(f"Invalid group_size={group_size} for in_features={in_features}")
    if scales_fp16_bits.dtype != torch.int16:
        raise ValueError(f"scales_fp16_bits must have dtype int16, got {scales_fp16_bits.dtype}")
    expected_sc_shape = (out_features, in_features // group_size)
    if tuple(scales_fp16_bits.shape) != expected_sc_shape:
        raise ValueError(
            f"scales_fp16_bits shape mismatch: got {tuple(scales_fp16_bits.shape)}, expected {expected_sc_shape}"
        )

    y_int, syn_errs = unpack_a16_integers(
        qquot,
        qham11,
        out_features=out_features,
        in_features=in_features,
        k_bits=k_bits,
        qov_rowptr=qov_rowptr,
        qov_col=qov_col,
        qov_val=qov_val,
        verify_syndrome=verify_syndrome,
    )
    scales_f32 = scales_fp16_bits.view(torch.float16).to(torch.float32)
    if not bool(torch.isfinite(scales_f32).all().item()):
        raise ValueError("Non-finite value detected in scales_fp16_bits")
    wq = (
        y_int.to(torch.float32).reshape(-1, group_size) * scales_f32.reshape(-1, 1)
    ).reshape(out_features, in_features)

    if rotation != "axis":
        if rotation == "block_h2":
            rot_d = 2
        elif rotation == "block_h4":
            rot_d = 4
        elif rotation == "h16":
            rot_d = 16
        else:
            raise ValueError(f"Unsupported rotation={rotation!r}")
        q_mat = _sylvester_hadamard_f32(rot_d, wq.device)
        # Sylvester Hadamard is symmetric (q_mat.T == q_mat); write q_mat.T explicitly for exact parity.
        wq = (wq.reshape(out_features, -1, rot_d) @ q_mat.T).reshape(out_features, in_features)

    if colscale_fp32_bits is not None:
        if colscale_fp32_bits.dtype != torch.int32:
            raise ValueError(
                f"colscale_fp32_bits must have dtype int32, got {colscale_fp32_bits.dtype}"
            )
        if colscale_fp32_bits.ndim != 1 or int(colscale_fp32_bits.shape[0]) != in_features:
            raise ValueError(
                f"colscale_fp32_bits shape mismatch: got {tuple(colscale_fp32_bits.shape)}, expected ({in_features},)"
            )
        cs = colscale_fp32_bits.view(torch.float32)
        if not bool(torch.isfinite(cs).all().item()) or bool((cs == 0).any().item()):
            raise ValueError("Non-finite or zero value detected in colscale_fp32_bits")
        wq = wq / cs.unsqueeze(0)

    if perm_in is not None:
        p = _validate_perm_in(perm_in, in_features)
        inv_perm = torch.empty(in_features, device=wq.device, dtype=torch.long)
        inv_perm[p] = torch.arange(in_features, device=wq.device, dtype=torch.long)
        wq = wq.index_select(1, inv_perm)

    if target_dtype is not None and wq.dtype != target_dtype:
        wq = wq.to(target_dtype)
    return wq, syn_errs


class SiliquantLinear(nn.Module):
    """Drop-in bit-packed `A_16` quantized linear projection with exact round-trip decode."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        group_size: int = 128,
        rotation: str = "block_h4",
        k_bits: int = 4,
        n_overflow: int = 0,
        ov_col_dtype: str = "int16",
        ov_val_dtype: str = "int8",
        has_colscale: bool = False,
        has_perm_in: bool = False,
        has_bias: bool = False,
        cache_dequantized: bool = True,
        verify_syndrome_on_load: bool = True,
    ) -> None:
        super().__init__()
        if rotation not in SUPPORTED_ROTATIONS:
            raise ValueError(f"Unsupported rotation={rotation!r}")
        if k_bits not in SUPPORTED_K_BITS:
            raise ValueError(f"Unsupported k_bits={k_bits}")
        if in_features % 16 != 0 or in_features % group_size != 0:
            raise ValueError(
                f"in_features={in_features} must be divisible by 16 and group_size={group_size}"
            )

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.group_size = int(group_size)
        self.rotation = str(rotation)
        self.k_bits = int(k_bits)
        self.n_overflow = int(n_overflow)
        self.cache_dequantized = bool(cache_dequantized)
        self.verify_syndrome_on_load = bool(verify_syndrome_on_load)
        self._syndrome_verified = False
        self._cached_weight: torch.Tensor | None = None
        self._cached_key: tuple[torch.device, torch.dtype] | None = None

        qquot_cols = (self.in_features * self.k_bits) // 8
        n_blk = self.out_features * (self.in_features // 16)
        qham_bytes = ((n_blk + 7) // 8) * 11

        self.register_buffer("qquot", torch.zeros((self.out_features, qquot_cols), dtype=torch.uint8))
        self.register_buffer("qham11", torch.zeros((qham_bytes,), dtype=torch.uint8))
        self.register_buffer(
            "scales_fp16_bits",
            torch.zeros((self.out_features, self.in_features // self.group_size), dtype=torch.int16),
        )

        if self.n_overflow > 0:
            col_dt = torch.int16 if ov_col_dtype == "int16" else torch.int32
            val_dt = torch.int8 if ov_val_dtype == "int8" else torch.int16
            self.register_buffer("qov_rowptr", torch.zeros((self.out_features + 1,), dtype=torch.int32))
            self.register_buffer("qov_col", torch.zeros((self.n_overflow,), dtype=col_dt))
            self.register_buffer("qov_val", torch.zeros((self.n_overflow,), dtype=val_dt))
        else:
            self.register_buffer("qov_rowptr", None)
            self.register_buffer("qov_col", None)
            self.register_buffer("qov_val", None)

        if has_colscale:
            self.register_buffer("colscale_fp32_bits", torch.zeros((self.in_features,), dtype=torch.int32))
        else:
            self.register_buffer("colscale_fp32_bits", None)

        if has_perm_in:
            self.register_buffer("perm_in", torch.zeros((self.in_features,), dtype=torch.int32))
        else:
            self.register_buffer("perm_in", None)

        if has_bias:
            self.bias = nn.Parameter(torch.zeros((self.out_features,)))
        else:
            self.register_parameter("bias", None)

    def clear_cache(self) -> None:
        """Drop any cached dequantized weight tensor."""
        self._cached_weight = None
        self._cached_key = None

    def _apply(self, fn: Any, recurse: bool = True) -> "SiliquantLinear":
        self.clear_cache()
        return super()._apply(fn, recurse=recurse)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        self.clear_cache()
        self._syndrome_verified = False
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def verify_syndrome(self) -> int:
        """Verify Extended Hamming [16, 11, 4] syndrome across 100% of blocks; return error count."""
        _, syn_errs = unpack_a16_integers(
            self.qquot,
            self.qham11,
            out_features=self.out_features,
            in_features=self.in_features,
            k_bits=self.k_bits,
            qov_rowptr=self.qov_rowptr,
            qov_col=self.qov_col,
            qov_val=self.qov_val,
            verify_syndrome=True,
        )
        self._syndrome_verified = syn_errs == 0
        return syn_errs

    def dequantize(
        self,
        dtype: torch.dtype | None = None,
        *,
        verify_syndrome: bool = False,
    ) -> torch.Tensor:
        """Unpack and reconstruct the exact weight matrix `[out_features, in_features]`."""
        wq, _ = dequantize_a16_weight(
            self.qquot,
            self.qham11,
            self.scales_fp16_bits,
            out_features=self.out_features,
            in_features=self.in_features,
            group_size=self.group_size,
            rotation=self.rotation,
            k_bits=self.k_bits,
            qov_rowptr=self.qov_rowptr,
            qov_col=self.qov_col,
            qov_val=self.qov_val,
            colscale_fp32_bits=self.colscale_fp32_bits,
            perm_in=self.perm_in,
            verify_syndrome=verify_syndrome,
            target_dtype=dtype,
        )
        if verify_syndrome:
            self._syndrome_verified = True
        return wq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        key = (x.device, x.dtype)
        need_verify = self.verify_syndrome_on_load and not self._syndrome_verified
        if self.cache_dequantized and self._cached_weight is not None and self._cached_key == key:
            w = self._cached_weight
        else:
            w = self.dequantize(dtype=x.dtype, verify_syndrome=need_verify)
            if self.cache_dequantized:
                self._cached_weight = w
                self._cached_key = key
        bias = self.bias.to(dtype=x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)

    def packed_bytes(self) -> int:
        """Return exact resident byte count of packed buffers + optional bias."""
        total = 0
        for buf in self.buffers():
            if buf is not None:
                total += int(buf.numel()) * int(buf.element_size())
        if self.bias is not None:
            total += int(self.bias.numel()) * int(self.bias.element_size())
        return total

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"g={self.group_size}, rot={self.rotation}, k_bits={self.k_bits}, "
            f"n_ov={self.n_overflow}, bias={self.bias is not None}"
        )


def _replace_verified_block_linear(
    root: nn.Module,
    dotted_name: str,
    new_module: SiliquantLinear,
) -> None:
    """Safely replace an existing `nn.Linear` under `model.layers.` with matching dimensions."""
    if not isinstance(dotted_name, str) or not dotted_name.startswith("model.layers."):
        raise ValueError(
            f"Refusing to replace non-layer module {dotted_name!r}; must start with 'model.layers.'"
        )
    parts = dotted_name.split(".")
    if any((not p) or ("__" in p) for p in parts):
        raise ValueError(f"Invalid module path {dotted_name!r}")
    try:
        existing = root.get_submodule(dotted_name)
    except AttributeError as exc:
        raise ValueError(f"Target module {dotted_name!r} does not exist in base architecture") from exc
    if not isinstance(existing, nn.Linear):
        raise ValueError(
            f"Refusing to replace {dotted_name!r} ({type(existing).__name__}); expected nn.Linear"
        )
    if existing.in_features != new_module.in_features or existing.out_features != new_module.out_features:
        raise ValueError(
            f"Dimension mismatch replacing {dotted_name!r}: existing "
            f"({existing.out_features}, {existing.in_features}) != "
            f"({new_module.out_features}, {new_module.in_features})"
        )
    parent = root.get_submodule(".".join(parts[:-1]))
    setattr(parent, parts[-1], new_module)


class SiliquantForCausalLM(PreTrainedModel, GenerationMixin):
    """Drop-in Hugging Face CausalLM wrapping Qwen2 / Llama decoders with `SiliquantLinear`."""

    config_class = SiliquantConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _supports_sdpa = True
    _no_split_modules = ["Qwen2DecoderLayer", "LlamaDecoderLayer"]
    _tied_weights_keys = (
        {"lm_head.weight": "model.embed_tokens.weight"}
        if _USES_CACHE_FIELD
        else ["lm_head.weight"]
    )

    def __init__(self, config: SiliquantConfig) -> None:
        super().__init__(config)
        base_cfg = config.to_base_config()
        if config.base_arch == "qwen2":
            self.model: nn.Module = Qwen2Model(base_cfg)
        elif config.base_arch == "llama":
            self.model = LlamaModel(base_cfg)
        else:
            raise ValueError(f"Unsupported base_arch={config.base_arch!r}")

        self.vocab_size = int(base_cfg.vocab_size)
        self.lm_head = nn.Linear(int(base_cfg.hidden_size), self.vocab_size, bias=False)

        # Replace specified transformer linear projections with bit-packed SiliquantLinear.
        for layer_name, meta in config.siliquant_layer_map.items():
            qlinear = SiliquantLinear(
                in_features=int(meta["in_features"]),
                out_features=int(meta["out_features"]),
                group_size=int(meta.get("group_size", config.siliquant_default_group_size)),
                rotation=str(meta.get("rotation", "axis")),
                k_bits=int(meta.get("k_bits", 4)),
                n_overflow=int(meta.get("n_overflow", 0)),
                ov_col_dtype=str(meta.get("ov_col_dtype", "int16")),
                ov_val_dtype=str(meta.get("ov_val_dtype", "int8")),
                has_colscale=bool(meta.get("has_colscale", False)),
                has_perm_in=bool(meta.get("has_perm_in", False)),
                has_bias=bool(meta.get("has_bias", False)),
                cache_dequantized=bool(config.siliquant_cache_dequantized),
                verify_syndrome_on_load=bool(config.siliquant_verify_syndrome_on_load),
            )
            _replace_verified_block_linear(self, layer_name, qlinear)

        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, SiliquantLinear):
            return
        std = float(getattr(self.config, "initializer_range", 0.02))
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.lm_head = new_embeddings

    def set_decoder(self, decoder: nn.Module) -> None:
        self.model = decoder

    def get_decoder(self) -> nn.Module:
        return self.model

    def clear_dequant_cache(self) -> None:
        """Clear cached dequantized weights across all `SiliquantLinear` layers."""
        for mod in self.modules():
            if isinstance(mod, SiliquantLinear):
                mod.clear_cache()

    def verify_all_syndromes(self) -> dict[str, int]:
        """Verify Extended Hamming [16, 11, 4] syndrome on every quantized layer."""
        total_errors = 0
        total_blocks = 0
        layer_count = 0
        for mod in self.modules():
            if isinstance(mod, SiliquantLinear):
                errs = mod.verify_syndrome()
                total_errors += errs
                total_blocks += mod.out_features * (mod.in_features // 16)
                layer_count += 1
        return {
            "quantized_layers": layer_count,
            "total_a16_blocks": total_blocks,
            "total_syndrome_errors": total_errors,
        }

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        legacy_pkv = kwargs.pop("past_key_values", None)
        active_cache = cache if cache is not None else legacy_pkv
        model_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "inputs_embeds": inputs_embeds,
            "use_cache": use_cache,
            **kwargs,
        }
        if _MODEL_USES_CACHE_ARG:
            model_kwargs["cache"] = active_cache
        else:
            model_kwargs["past_key_values"] = active_cache

        outputs = self.model(**model_kwargs)
        hidden_states = outputs.last_hidden_state
        if isinstance(logits_to_keep, int):
            if logits_to_keep > 0:
                hidden_states = hidden_states[:, -logits_to_keep:, :]
        else:
            hidden_states = hidden_states[:, logits_to_keep, :]

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            if hasattr(self, "loss_function"):
                loss = self.loss_function(
                    logits=logits, labels=labels, vocab_size=self.vocab_size, **kwargs
                )
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, self.vocab_size), shift_labels.view(-1)
                )

        if _USES_CACHE_FIELD:
            out_cache = getattr(outputs, "cache", None)
            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                cache=out_cache,
                hidden_states=getattr(outputs, "hidden_states", None),
                attentions=getattr(outputs, "attentions", None),
            )
        out_cache = getattr(outputs, "past_key_values", None)
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=out_cache,
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
        )


SiliquantForCausalLM.register_for_auto_class("AutoModelForCausalLM")
