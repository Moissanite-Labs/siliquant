#!/usr/bin/env python3
"""Lossless bit-packed Hugging Face exporter for `siliquant-hf-v1` lattice models.

Quantizes transformer block linear projections (`model.layers.*.self_attn.*_proj` and
`model.layers.*.mlp.*_proj`) using exact Extended Hamming [16, 11, 4] `A_16` L2-ball
quantization (matching Moissanite-Labs/siliquant PRs #83-#95 bit-for-bit), packs them
losslessly (`0.0` max abs round-trip diff, `0` syndrome errors) into `model.safetensors`,
preserves untouched BF16 embeddings/lm_head/layernorms/biases, emits exact upstream HF
YAML license metadata, generates a publication-grade Model Card (`README.md`), and writes
a SHA-256 verification receipt (`siliquant_verification_receipt.json`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors.torch import save_file

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from configuration_siliquant import (
    SILIQUANT_FORMAT_VERSION,
    SUPPORTED_ROTATIONS,
    SiliquantConfig,
)
from modeling_siliquant import (
    _validate_perm_in,
    check_a16_syndrome_errors,
    dequantize_a16_weight,
    pack_a16_tensor,
)

# Exact theta-series L2-ball bounds matching Moissanite-Labs/siliquant#85 / Tracks S68-S73.
BALL_BOUNDS: dict[str, dict[int, float]] = {
    "4.5": {128: 5213.0, 32: 1207.0},
    "3.0": {128: 651.0, 32: 151.0},
}

QUANT_ROW_CHUNK = 512

# Exact upstream HF YAML license metadata (strictly inherited; NEVER relabel Qwen2.5-3B as apache-2.0).
UPSTREAM_LICENSE_REGISTRY: dict[str, dict[str, str | None]] = {
    "HuggingFaceTB/SmolLM2-360M": {
        "license": "apache-2.0",
        "license_name": None,
        "license_link": "https://huggingface.co/HuggingFaceTB/SmolLM2-360M/blob/main/LICENSE",
        "license_summary": "Apache License 2.0 (inherited from HuggingFaceTB/SmolLM2-360M)",
    },
    "Qwen/Qwen2.5-0.5B": {
        "license": "apache-2.0",
        "license_name": None,
        "license_link": "https://huggingface.co/Qwen/Qwen2.5-0.5B/blob/main/LICENSE",
        "license_summary": "Apache License 2.0 (inherited from Qwen/Qwen2.5-0.5B)",
    },
    "Qwen/Qwen2.5-1.5B": {
        "license": "apache-2.0",
        "license_name": None,
        "license_link": "https://huggingface.co/Qwen/Qwen2.5-1.5B/blob/main/LICENSE",
        "license_summary": "Apache License 2.0 (inherited from Qwen/Qwen2.5-1.5B)",
    },
    "Qwen/Qwen2.5-3B": {
        "license": "other",
        "license_name": "qwen-research",
        "license_link": "https://huggingface.co/Qwen/Qwen2.5-3B/blob/main/LICENSE",
        "license_summary": "Qwen Research License (inherited from Qwen/Qwen2.5-3B)",
    },
}

TOKENIZER_AND_META_FILES: tuple[str, ...] = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "generation_config.json",
    "LICENSE",
    "LICENSE.txt",
)

SILIQUANT_PR_REFERENCES: tuple[tuple[str, str], ...] = (
    ("#83", "https://github.com/Moissanite-Labs/siliquant/pull/83"),
    ("#85", "https://github.com/Moissanite-Labs/siliquant/pull/85"),
    ("#91", "https://github.com/Moissanite-Labs/siliquant/pull/91"),
    ("#93", "https://github.com/Moissanite-Labs/siliquant/pull/93"),
    ("#95", "https://github.com/Moissanite-Labs/siliquant/pull/95"),
)


def resolve_upstream_license(base_model_id: str, override_license: str | None = None) -> dict[str, str | None]:
    """Return exact HF YAML license metadata for `base_model_id` (fail-closed on conflicting overrides)."""
    norm = base_model_id.strip()
    is_registered = False
    if norm in UPSTREAM_LICENSE_REGISTRY:
        spec = dict(UPSTREAM_LICENSE_REGISTRY[norm])
        is_registered = True
    elif "qwen2.5-3b" in norm.lower():
        spec = dict(UPSTREAM_LICENSE_REGISTRY["Qwen/Qwen2.5-3B"])
        is_registered = True
    elif "qwen2.5-1.5b" in norm.lower():
        spec = dict(UPSTREAM_LICENSE_REGISTRY["Qwen/Qwen2.5-1.5B"])
        is_registered = True
    elif "qwen2.5-0.5b" in norm.lower():
        spec = dict(UPSTREAM_LICENSE_REGISTRY["Qwen/Qwen2.5-0.5B"])
        is_registered = True
    elif "smollm2-360m" in norm.lower():
        spec = dict(UPSTREAM_LICENSE_REGISTRY["HuggingFaceTB/SmolLM2-360M"])
        is_registered = True
    else:
        lic = override_license if override_license else "other"
        spec = {
            "license": lic,
            "license_name": None,
            "license_link": None,
            "license_summary": f"Inherited upstream license ({lic}) from {norm}",
        }

    # Hard compliance guard: registered upstream models cannot be silently relabeled.
    if is_registered and override_license is not None and override_license.strip() != spec["license"]:
        raise ValueError(
            f"Compliance violation: conflicting override_license={override_license!r} for "
            f"registered model {base_model_id!r} (pinned upstream license={spec['license']!r})"
        )
    if "qwen2.5-3b" in norm.lower() and spec.get("license") != "other":
        raise ValueError("Compliance violation: Qwen2.5-3B must use license: other (qwen-research)")
    return spec


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sylvester_hadamard(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    h = torch.tensor([[1.0]], device=device, dtype=dtype)
    while int(h.shape[0]) < n:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    return h / math.sqrt(float(n))


def _rotation_matrix(rotation: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
    if rotation == "axis":
        return None
    if rotation == "block_h2":
        return _sylvester_hadamard(2, device, dtype)
    if rotation == "block_h4":
        return _sylvester_hadamard(4, device, dtype)
    if rotation == "h16":
        return _sylvester_hadamard(16, device, dtype)
    raise ValueError(f"Unsupported rotation={rotation!r}; expected {SUPPORTED_ROTATIONS}")


def _apply_block_rot(w: torch.Tensor, q_mat: torch.Tensor) -> torch.Tensor:
    d = int(q_mat.shape[0])
    m, k = int(w.shape[0]), int(w.shape[1])
    return (w.reshape(m, k // d, d) @ q_mat.T).reshape(m, k)


def _a16_closest(u16: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Exact vectorized O(B*16) Extended Hamming [16, 11, 4] nearest-lattice-point decoder."""
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
    if bool(odd_m.any().item()):
        r_o = torch.nonzero(odd_m, as_tuple=False).squeeze(1)
        c_o = syn[r_o].to(torch.int64)
        out[r_o, c_o] = alt[r_o, c_o]
        out_b[r_o, c_o] ^= 1

    ev_m = (p0 == 0) & (syn != 0)
    if bool(ev_m.any().item()):
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
    errs = int(((vp0 | vs0 | vs1 | vs2 | vs3) != 0).sum().item())
    return out, errs


def _quantize_a16_ball_chunk(
    w_chunk: torch.Tensor,
    group_size: int,
    bpw_key: str,
    q_rot: torch.Tensor | None = None,
    colscale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Exact A_16 L2-ball quantizer returning `(y_int [M,K], scales_fp16 [M,K//g], wq_f32, syn_errs)`."""
    x = w_chunk.clone()
    if colscale is not None:
        x = x * colscale.unsqueeze(0)
    if q_rot is not None:
        x = _apply_block_rot(x, q_rot)

    xg = x.reshape(-1, group_size)
    bound = BALL_BOUNDS[bpw_key][group_size]
    r_ball = math.sqrt(bound)
    margin = math.sqrt(group_size / 4.0 + 1.5 * (group_size / 16.0))

    g_norm = torch.sqrt((xg ** 2).sum(dim=1)).clamp_min(1e-30)
    lo = g_norm / (r_ball + margin) * 0.99
    hi = g_norm / max(r_ball - margin, 0.05 * r_ball) * 1.01

    for _ in range(16):
        mid = torch.sqrt(lo * hi)
        scaled = xg / mid.unsqueeze(1)
        y_try, _ = _a16_closest(scaled.reshape(-1, 16))
        y_try = y_try.reshape(-1, group_size)
        inside = (y_try ** 2).sum(dim=1) <= bound
        hi = torch.where(inside, mid, hi)
        lo = torch.where(inside, lo, mid)

    scaled_f = xg / hi.unsqueeze(1)
    y_fin, syn_errs = _a16_closest(scaled_f.reshape(-1, 16))
    y_fin = y_fin.reshape(-1, group_size)

    out_m = (y_fin ** 2).sum(dim=1) > bound
    if bool(out_m.any().item()):
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
        y_fin[out_m] = yf2.reshape(-1, group_size)
        syn_errs += se2

    m_rows, k_cols = int(w_chunk.shape[0]), int(w_chunk.shape[1])
    scales_fp16 = hi.to(torch.float16).reshape(m_rows, k_cols // group_size)
    scales_f32 = scales_fp16.to(torch.float32).reshape(-1, 1)
    y_int = y_fin.round().to(torch.int16).reshape(m_rows, k_cols)
    wq = (y_int.to(torch.float32).reshape(-1, group_size) * scales_f32).reshape(m_rows, k_cols)
    if q_rot is not None:
        wq = _apply_block_rot(wq, q_rot.T)
    if colscale is not None:
        wq = wq / colscale.unsqueeze(0)
    return y_int, scales_fp16, wq, syn_errs


def quantize_a16_matrix(
    w_f32: torch.Tensor,
    *,
    group_size: int,
    bpw_key: str = "4.5",
    rotation: str = "block_h4",
    colscale: torch.Tensor | None = None,
    perm_in: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Row-chunked exact A_16 L2-ball quantizer returning `(y_int, scales_fp16, wq_f32, syn_errs)`."""
    if bpw_key not in BALL_BOUNDS or group_size not in BALL_BOUNDS[bpw_key]:
        raise ValueError(f"Unsupported (bpw_key={bpw_key!r}, group_size={group_size})")
    q_rot = _rotation_matrix(rotation, w_f32.device, torch.float32)

    w_work = w_f32
    cs_work = colscale
    if perm_in is not None:
        p_idx = _validate_perm_in(perm_in, int(w_f32.shape[1])).to(device=w_f32.device)
        w_work = w_work.index_select(1, p_idx)
        if cs_work is not None:
            cs_work = cs_work.index_select(0, p_idx)

    m = int(w_work.shape[0])
    if m <= QUANT_ROW_CHUNK:
        y_int, sc_f16, wq, syn_tot = _quantize_a16_ball_chunk(
            w_work, group_size, bpw_key, q_rot, cs_work
        )
    else:
        y_parts: list[torch.Tensor] = []
        sc_parts: list[torch.Tensor] = []
        wq_parts: list[torch.Tensor] = []
        syn_tot = 0
        for r0 in range(0, m, QUANT_ROW_CHUNK):
            r1 = min(r0 + QUANT_ROW_CHUNK, m)
            yi, sci, wqi, syn_i = _quantize_a16_ball_chunk(
                w_work[r0:r1], group_size, bpw_key, q_rot, cs_work
            )
            y_parts.append(yi)
            sc_parts.append(sci)
            wq_parts.append(wqi)
            syn_tot += syn_i
        y_int = torch.cat(y_parts, dim=0)
        sc_f16 = torch.cat(sc_parts, dim=0)
        wq = torch.cat(wq_parts, dim=0)

    if perm_in is not None:
        p_idx = perm_in.to(device=w_f32.device, dtype=torch.long)
        inv_p = torch.empty_like(p_idx)
        inv_p[p_idx] = torch.arange(p_idx.shape[0], device=w_f32.device, dtype=torch.long)
        wq = wq.index_select(1, inv_p)

    return y_int, sc_f16, wq, syn_tot


def make_quad_snake_perm(score: torch.Tensor, group_size: int, block_w: int = 4) -> torch.Tensor:
    """Stratified boustrophedon (snake) quad-interleaved permutation (Track S72)."""
    if score.ndim != 1:
        raise ValueError(f"score must be 1D, got shape {tuple(score.shape)}")
    k = int(score.shape[0])
    if k <= 0 or group_size <= 0 or block_w <= 0 or (k % group_size) != 0 or (group_size % block_w) != 0:
        raise ValueError(
            f"Invalid permutation geometry (k={k}, group_size={group_size}, block_w={block_w}); "
            "require positive values with k % group_size == 0 and group_size % block_w == 0"
        )
    n_groups = k // group_size
    n_quads = k // block_w
    qpg = group_size // block_w
    idx = torch.argsort(score.to(torch.float64), descending=True, stable=True)
    quads = idx.reshape(n_quads, block_w)
    buckets: list[list[torch.Tensor]] = [[] for _ in range(n_groups)]
    for t in range(qpg):
        order = range(n_groups) if (t % 2 == 0) else range(n_groups - 1, -1, -1)
        for pos, g in enumerate(order):
            buckets[g].append(quads[t * n_groups + pos])
    return torch.cat([torch.cat(b, dim=0) for b in buckets], dim=0).to(torch.int32)


def make_quad_lpt_perm(
    score: torch.Tensor,
    group_size: int,
    block_w: int = 4,
    load_power: float = 1.0,
) -> torch.Tensor:
    """Stratified greedy LPT quad-interleaved permutation (Track S72)."""
    if score.ndim != 1:
        raise ValueError(f"score must be 1D, got shape {tuple(score.shape)}")
    k = int(score.shape[0])
    if k <= 0 or group_size <= 0 or block_w <= 0 or (k % group_size) != 0 or (group_size % block_w) != 0:
        raise ValueError(
            f"Invalid permutation geometry (k={k}, group_size={group_size}, block_w={block_w}); "
            "require positive values with k % group_size == 0 and group_size % block_w == 0"
        )
    n_groups = k // group_size
    n_quads = k // block_w
    qpg = group_size // block_w
    s64 = score.to(torch.float64).clamp_min(1e-30)
    idx = torch.argsort(s64, descending=True, stable=True)
    quads = idx.reshape(n_quads, block_w)
    q_loads = (s64[quads].sum(dim=1) ** float(load_power)).tolist()

    loads = [0.0] * n_groups
    counts = [0] * n_groups
    buckets: list[list[torch.Tensor]] = [[] for _ in range(n_groups)]
    for q in range(n_quads):
        best_g = -1
        best_load = float("inf")
        for g in range(n_groups):
            if counts[g] < qpg and loads[g] < best_load:
                best_load = loads[g]
                best_g = g
        if best_g < 0:
            raise ValueError("LPT bucket capacity exceeded")
        buckets[best_g].append(quads[q])
        loads[best_g] += q_loads[q]
        counts[best_g] += 1
    return torch.cat([torch.cat(b, dim=0) for b in buckets], dim=0).to(torch.int32)


def _select_layer_rotation(layer_name: str, recipe_rotation: str) -> str:
    if recipe_rotation in SUPPORTED_ROTATIONS:
        return recipe_rotation
    if recipe_rotation == "projtype_uniform":
        # Track S69 ProjType_Uniform: Block-H4 on MLP projections, axis on attention projections.
        proj = layer_name.rsplit(".", 1)[-1]
        return "block_h4" if proj in ("gate_proj", "up_proj", "down_proj") else "axis"
    raise ValueError(f"Unsupported rotation policy {recipe_rotation!r}")


def _collect_block_linears(model: nn.Module) -> list[tuple[str, nn.Linear]]:
    return [
        (name, mod)
        for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear) and name.startswith("model.layers.")
    ]


def _build_model_card(
    *,
    repo_name: str,
    base_model_id: str,
    license_spec: dict[str, str | None],
    recipe: str,
    bpw_key: str,
    group_size: int,
    stats: dict[str, Any],
    benchmarks: dict[str, Any] | None = None,
) -> str:
    yaml_lines = ["---"]
    yaml_lines.append(f"license: {license_spec['license']}")
    if license_spec.get("license_name"):
        yaml_lines.append(f"license_name: {license_spec['license_name']}")
    if license_spec.get("license_link"):
        yaml_lines.append(f"license_link: {license_spec['license_link']}")
    yaml_lines.append(f"base_model: {base_model_id}")
    yaml_lines.append("library_name: transformers")
    yaml_lines.append("pipeline_tag: text-generation")
    yaml_lines.append("tags:")
    for tag in (
        "siliquant",
        "lattice-quantization",
        "a16",
        "extended-hamming",
        "block-hadamard",
        "quantization",
        "pytorch",
        "safetensors",
    ):
        yaml_lines.append(f"  - {tag}")
    yaml_lines.append("---")

    pr_links = ", ".join(f"[{num}]({url})" for num, url in SILIQUANT_PR_REFERENCES)
    bf16_mb = stats["equivalent_bf16_total_bytes"] / (1024.0 * 1024.0)
    packed_mb = stats["model_safetensors_bytes"] / (1024.0 * 1024.0)
    ratio = stats["total_compression_ratio_vs_bf16"]

    lines = [
        *yaml_lines,
        "",
        f"# {repo_name}",
        "",
        f"Bit-packed **`A_16` Extended Hamming `[16, 11, 4]` lattice** quantization of "
        f"[`{base_model_id}`](https://huggingface.co/{base_model_id}) in **`{SILIQUANT_FORMAT_VERSION}`** "
        f"format (`recipe={recipe}`, target `{bpw_key} bpw`, group size `g={group_size}`).",
        "",
        "## Quickstart (`transformers` Drop-In)",
        "",
        "Requires only standard `torch`, `transformers`, and `safetensors` — **zero custom C++/CUDA builds required**.",
        "",
        "```python",
        "import torch",
        "from transformers import AutoModelForCausalLM, AutoTokenizer",
        "",
        f'repo_id = "{repo_name}"',
        "device = \"cuda\" if torch.cuda.is_available() else \"cpu\"",
        "tokenizer = AutoTokenizer.from_pretrained(repo_id)",
        "model = AutoModelForCausalLM.from_pretrained(",
        "    repo_id,",
        "    trust_remote_code=True,",
        '    dtype="auto",',
        "    # Optional (requires `accelerate`): device_map=\"auto\",",
        ").to(device)",
        "",
        'inputs = tokenizer("The exact identity of the Hadamard matrix is", return_tensors="pt").to(device)',
        "with torch.no_grad():",
        "    out = model.generate(**inputs, max_new_tokens=48, do_sample=False)",
        "print(tokenizer.decode(out[0], skip_special_tokens=True))",
        "```",
        "",
        "## Compression & Exact Verification Summary",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Base Model | `{base_model_id}` |",
        f"| Format Version | `{SILIQUANT_FORMAT_VERSION}` |",
        f"| Quantization Recipe | `{recipe}` (`g={group_size}`, target `{bpw_key} bpw`) |",
        f"| Quantized Linear Layers | `{stats['quantized_layers']}` (`{stats['total_quantized_weights']:,}` weights) |",
        f"| Untouched Precision | `embed_tokens`, `lm_head`, RMSNorms, biases preserved in `bfloat16` |",
        f"| Effective Packed Weight Bits/Weight | `{stats['effective_packed_bpw']:.4f} bpw` |",
        f"| Checkpoint Size (`model.safetensors`) | `{packed_mb:.2f} MiB` vs `{bf16_mb:.2f} MiB` BF16 (`{ratio:.2f}x` smaller) |",
        f"| Extended Hamming `[16, 11, 4]` Unpack Codec Check | `{stats['total_syndrome_errors']}` / `{stats['total_a16_blocks']:,}` blocks (`0` by construction; SHA-256 receipt enforces file integrity) |",
        f"| Max Abs Round-Trip Unpack Diff | `{stats['max_abs_roundtrip_diff']:.1e}` (exact bit-for-bit) |",
        "",
    ]

    if benchmarks:
        lines.extend([
            "## Certified Held-Out Quality Benchmarks",
            "",
            "| Corpus / Metric | Value |",
            "| --- | --- |",
        ])
        for k, v in sorted(benchmarks.items()):
            lines.append(f"| `{k}` | `{v}` |")
        lines.append("")

    lines.extend([
        "## Mathematical Format (`siliquant-hf-v1`)",
        "",
        "Every 16-element weight vector `y in A_16 = 2 Z^16 + C_{[16, 11, 4]}` is stored losslessly via:",
        "- `qham11`: 11 systematic information bits per 16-block (`0.6875 bpw`), algebraically reconstructing the 5 parity checks at `{0, 1, 2, 4, 8}` (`verify_syndrome=True` checks codec/lattice consistency; end-to-end payload tamper integrity across `qquot`, `qham11`, and scales is verified via `siliquant_verification_receipt.json` SHA-256 hashes).",
        "- `qquot`: per-layer minimum-byte packed signed quotient `k = (y - c) >> 1` (`k_bits in {2, 3, 4}`) plus compact CSR overflow `(qov_rowptr, qov_col, qov_val)`.",
        "- `scales_fp16_bits` (`int16`) & optional `colscale_fp32_bits` (`int32`): exact IEEE-754 bit patterns immune to `.to(bfloat16)` mantissa truncation.",
        "",
        "## Standalone CLI & Reference Kernels",
        "",
        "- `python siliquant_dequant.py verify .` — verifies every file SHA-256 in `siliquant_verification_receipt.json` (rejecting tampered, missing, unlisted, or symlinked files; Hugging Face download metadata — .gitattributes, .cache/huggingface/ — and __pycache__ are ignored) and checks `A_16` codec consistency across 100% of blocks.",
        "- `python siliquant_dequant.py unpack . --out unpacked.safetensors --dtype bfloat16` — exports dense BF16 weights for downstream engines.",
        "- `kernels/sm86_a16_blockh4.cu` — standalone reference zero-spill Ampere `sm_86` PTX fused fixed-4-bit nibble-packed `A_16` + Block-H4 `dp4a.s32.s32` GEMV kernel (distinct from the lossless `qham11`+`qquot`+CSR archive container).",
        "- `kernels/avx512_a16_blockh4.c` — standalone reference Intel AVX-512 (`vpermb` + `vpopcntq` + `vpdpbusd`) CPU fixed-4-bit nibble-packed `A_16` + Block-H4 GEMV decoder.",
        "",
        "## Upstream Research & Code",
        "",
        f"Developed on [Moissanite-Labs/siliquant](https://github.com/Moissanite-Labs/siliquant) (PRs {pr_links}).",
        "",
        "## License",
        "",
        f"{license_spec['license_summary']}.",
    ])
    if license_spec.get("license_link"):
        lines.append(f"See upstream license: {license_spec['license_link']}")
    lines.append("")
    return "\n".join(lines)


def _guard_output_dir(output_dir: Path) -> None:
    """Enforce repo WEIGHTS=0 policy: reject exporting binary weights inside source trees or git worktrees."""
    resolved_path = output_dir.resolve()
    resolved = str(resolved_path).replace("\\", "/")
    if "/Projects/expedition/" in resolved or resolved.endswith("/Projects/expedition"):
        raise ValueError(
            "Refusing to export model weights inside Projects/expedition/ (WEIGHTS=0 policy); "
            "export to an external release directory or temporary directory."
        )

    # Also block exporting inside the enclosing siliquant checkout (even in a git-less tarball).
    script_dir = Path(__file__).resolve().parent
    candidate_roots: list[Path] = [script_dir]
    if script_dir.parent.name == "studies":
        candidate_roots.append(script_dir.parent.parent)
    for root in candidate_roots:
        try:
            resolved_path.relative_to(root)
        except ValueError:
            pass
        else:
            raise ValueError(
                f"Refusing to export model weights inside siliquant source tree ({root.name}, WEIGHTS=0 policy); "
                "export to an external release directory or temporary directory."
            )

    # Block exporting binary weights into any Git working tree or checkout containing repo checks.
    for anc in (resolved_path, *resolved_path.parents):
        if (anc / ".git").exists() or (anc / "checks" / "claim_ledger.py").is_file():
            raise ValueError(
                f"Refusing to export model weights inside git/source worktree ({anc.name}, WEIGHTS=0 policy); "
                "export to a clean external release directory or temporary directory."
            )


@torch.no_grad()
def export_model_to_siliquant_hf(
    model: nn.Module,
    output_dir: Path,
    *,
    base_model_id: str,
    repo_name: str,
    group_size: int = 128,
    bpw_key: str = "4.5",
    rotation: str = "block_h4",
    perm_mode: str = "none",
    recipe_name: str | None = None,
    hessian_diag: dict[str, torch.Tensor] | None = None,
    use_colscale: bool = False,
    source_dir: Path | None = None,
    benchmarks: dict[str, Any] | None = None,
    cache_dequantized: bool = True,
    override_license: str | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Quantize an in-memory Qwen2/Llama CausalLM and export a complete `siliquant-hf-v1` repo."""
    _guard_output_dir(output_dir)
    if bpw_key not in BALL_BOUNDS or group_size not in BALL_BOUNDS[bpw_key]:
        raise ValueError(f"Unsupported (bpw_key={bpw_key!r}, group_size={group_size})")
    if perm_mode not in (
        "none",
        "quad_lpt_wcol",
        "quad_snake_wcol",
        "quad_lpt_down_wcol",
        "quad_lpt_hreal",
        "quad_snake_hreal",
        "quad_lpt_sqrt_hreal",
    ):
        raise ValueError(f"Unsupported perm_mode={perm_mode!r}")

    license_spec = resolve_upstream_license(base_model_id, override_license=override_license)
    compute_dev = device if device is not None else next(model.parameters()).device
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "kernels").mkdir(parents=True, exist_ok=True)

    # Collect block linears and snapshot original weights on CPU so model is not unexpectedly mutated.
    linears = _collect_block_linears(model)
    if not linears:
        raise ValueError("No transformer block nn.Linear layers found under 'model.layers.'")

    work_weights: dict[str, torch.Tensor] = {
        name: mod.weight.detach().to(device="cpu", dtype=torch.float32).clone()
        for name, mod in linears
    }
    work_biases: dict[str, torch.Tensor | None] = {
        name: (
            mod.bias.detach().to(device="cpu", dtype=torch.bfloat16).clone()
            if mod.bias is not None
            else None
        )
        for name, mod in linears
    }

    # Optional zero-runtime-cost offline fold on MLP down_proj (Track S69/S72):
    # Permute gate_proj & up_proj output rows and down_proj input cols identically BEFORE quantization.
    if perm_mode == "quad_lpt_down_wcol":
        layer_indices = sorted({int(n.split(".")[2]) for n, _ in linears})
        for li in layer_indices:
            down_k = f"model.layers.{li}.mlp.down_proj"
            gate_k = f"model.layers.{li}.mlp.gate_proj"
            up_k = f"model.layers.{li}.mlp.up_proj"
            if down_k in work_weights and gate_k in work_weights and up_k in work_weights:
                w_down = work_weights[down_k]
                score = (w_down ** 2).mean(dim=0)
                perm = make_quad_lpt_perm(score, group_size, block_w=4).to(torch.long)
                work_weights[down_k] = w_down.index_select(1, perm)
                work_weights[gate_k] = work_weights[gate_k].index_select(0, perm)
                work_weights[up_k] = work_weights[up_k].index_select(0, perm)
                if work_biases[gate_k] is not None:
                    work_biases[gate_k] = work_biases[gate_k].index_select(0, perm)
                if work_biases[up_k] is not None:
                    work_biases[up_k] = work_biases[up_k].index_select(0, perm)

    safetensors_dict: dict[str, torch.Tensor] = {}
    layer_map: dict[str, dict[str, Any]] = {}
    layer_receipts: dict[str, dict[str, Any]] = {}
    ref_dequant_f32: dict[str, torch.Tensor] = {}

    total_quant_weights = 0
    total_packed_bytes = 0
    total_bf16_quant_bytes = 0
    total_syn_errs = 0
    total_a16_blocks = 0
    max_abs_roundtrip_diff = 0.0

    for name, _ in linears:
        w_f32 = work_weights[name].to(device=compute_dev, dtype=torch.float32)
        m_rows, k_cols = int(w_f32.shape[0]), int(w_f32.shape[1])
        layer_rot = _select_layer_rotation(name, rotation)

        # Build optional colscale and explicit input permutation.
        colscale_t: torch.Tensor | None = None
        if use_colscale:
            if hessian_diag is None or name not in hessian_diag:
                raise ValueError(f"use_colscale=True requires hessian_diag entry for {name}")
            hv = hessian_diag[name].to(device=compute_dev, dtype=torch.float32).clamp_min(1e-12)
            colscale_t = (hv / hv.mean()).clamp_min(1e-8) ** 0.25

        perm_t: torch.Tensor | None = None
        if perm_mode == "quad_lpt_wcol":
            score = (w_f32 ** 2).mean(dim=0).cpu()
            perm_t = make_quad_lpt_perm(score, group_size, block_w=4)
        elif perm_mode == "quad_snake_wcol":
            score = (w_f32 ** 2).mean(dim=0).cpu()
            perm_t = make_quad_snake_perm(score, group_size, block_w=4)
        elif perm_mode in ("quad_lpt_hreal", "quad_snake_hreal", "quad_lpt_sqrt_hreal"):
            if hessian_diag is None or name not in hessian_diag:
                raise ValueError(f"perm_mode={perm_mode!r} requires hessian_diag entry for {name}")
            h_score = hessian_diag[name].cpu()
            if perm_mode == "quad_lpt_hreal":
                perm_t = make_quad_lpt_perm(h_score, group_size, block_w=4, load_power=1.0)
            elif perm_mode == "quad_lpt_sqrt_hreal":
                perm_t = make_quad_lpt_perm(h_score, group_size, block_w=4, load_power=0.5)
            else:
                perm_t = make_quad_snake_perm(h_score, group_size, block_w=4)

        y_int, scales_fp16, wq_ref, syn_errs = quantize_a16_matrix(
            w_f32,
            group_size=group_size,
            bpw_key=bpw_key,
            rotation=layer_rot,
            colscale=colscale_t,
            perm_in=perm_t,
        )
        if syn_errs != 0 or check_a16_syndrome_errors(y_int) != 0:
            raise RuntimeError(f"Quantizer produced non-zero A_16 syndrome errors on {name}")

        # Store permuted colscale if both colscale & perm_in are active so dequant divides in permuted space.
        cs_to_store: torch.Tensor | None = None
        if colscale_t is not None:
            cs_cpu = colscale_t.detach().cpu().to(torch.float32)
            cs_to_store = cs_cpu.index_select(0, perm_t.to(torch.long)) if perm_t is not None else cs_cpu

        buffers, meta = pack_a16_tensor(
            y_int.detach().cpu(),
            scales_fp16.detach().cpu(),
            group_size=group_size,
            rotation=layer_rot,
            colscale_f32=cs_to_store,
            perm_in=perm_t.cpu() if perm_t is not None else None,
        )
        meta["has_bias"] = work_biases[name] is not None

        # Immediate lossless round-trip verification against wq_ref (must be exact 0.0!).
        wq_rt, rt_syn = dequantize_a16_weight(
            buffers["qquot"],
            buffers["qham11"],
            buffers["scales_fp16_bits"],
            out_features=m_rows,
            in_features=k_cols,
            group_size=group_size,
            rotation=layer_rot,
            k_bits=meta["k_bits"],
            qov_rowptr=buffers.get("qov_rowptr"),
            qov_col=buffers.get("qov_col"),
            qov_val=buffers.get("qov_val"),
            colscale_fp32_bits=buffers.get("colscale_fp32_bits"),
            perm_in=buffers.get("perm_in"),
            verify_syndrome=True,
            target_dtype=torch.float32,
        )
        if rt_syn != 0:
            raise RuntimeError(f"Round-trip syndrome error on {name}: {rt_syn}")
        diff = float((wq_rt - wq_ref.detach().cpu()).abs().max().item())
        if diff > max_abs_roundtrip_diff:
            max_abs_roundtrip_diff = diff
        if diff != 0.0:
            raise RuntimeError(
                f"Lossless round-trip violated on {name}: max_abs_diff={diff:.3e} != 0.0"
            )

        ref_dequant_f32[name] = wq_rt
        layer_map[name] = meta
        layer_bytes = 0
        for bkey, btensor in buffers.items():
            safetensors_dict[f"{name}.{bkey}"] = btensor.contiguous()
            layer_bytes += int(btensor.numel()) * int(btensor.element_size())
        if work_biases[name] is not None:
            safetensors_dict[f"{name}.bias"] = work_biases[name].contiguous()

        total_quant_weights += m_rows * k_cols
        total_packed_bytes += layer_bytes
        total_bf16_quant_bytes += m_rows * k_cols * 2
        total_syn_errs += rt_syn
        total_a16_blocks += m_rows * (k_cols // 16)
        layer_receipts[name] = {
            "shape": [m_rows, k_cols],
            "rotation": layer_rot,
            "k_bits": meta["k_bits"],
            "n_overflow": meta["n_overflow"],
            "packed_bytes": layer_bytes,
            "bf16_bytes": m_rows * k_cols * 2,
            "effective_bpw": round(8.0 * layer_bytes / float(m_rows * k_cols), 4),
            "syndrome_errors": rt_syn,
            "max_abs_roundtrip_diff": diff,
        }

    # Copy all untouched non-block parameters (embed_tokens, lm_head if untied, layernorms) in BF16.
    quant_layer_set = set(layer_map.keys())
    tie_emb = bool(getattr(model.config, "tie_word_embeddings", False))
    non_block_bf16_bytes = 0
    for pname, param in model.named_parameters():
        # Skip quantized weights & already-saved quantized biases.
        if pname.endswith(".weight") and pname[: -len(".weight")] in quant_layer_set:
            continue
        if pname.endswith(".bias") and pname[: -len(".bias")] in quant_layer_set:
            continue
        if tie_emb and pname == "lm_head.weight":
            # Tied to model.embed_tokens.weight; omit duplicate tensor exactly like upstream HF checkpoints.
            continue
        t_bf16 = param.detach().to(device="cpu", dtype=torch.bfloat16).clone().contiguous()
        safetensors_dict[pname] = t_bf16
        non_block_bf16_bytes += int(t_bf16.numel()) * int(t_bf16.element_size())

    # Persist model.safetensors.
    recipe = recipe_name or f"A16_{rotation}_{perm_mode}_{bpw_key}bpw"
    st_path = output_dir / "model.safetensors"
    save_file(
        safetensors_dict,
        str(st_path),
        metadata={
            "format": "pt",
            "siliquant_format_version": SILIQUANT_FORMAT_VERSION,
            "siliquant_recipe": recipe,
            "base_model": base_model_id,
        },
    )

    # Save SiliquantConfig -> config.json with auto_map.
    sq_config = SiliquantConfig.from_base_config(
        model.config,
        siliquant_bpw_target=float(bpw_key),
        siliquant_default_group_size=int(group_size),
        siliquant_recipe=recipe,
        siliquant_layer_map=layer_map,
        siliquant_cache_dequantized=cache_dequantized,
        siliquant_verify_syndrome_on_load=True,
        base_model_name_or_path=base_model_id,
        dtype_str="bfloat16",
    )
    sq_config.save_pretrained(str(output_dir))

    # Copy self-contained remote-code Python files, CLI tool, and reference kernels.
    for fname in ("configuration_siliquant.py", "modeling_siliquant.py", "siliquant_dequant.py"):
        shutil.copy2(SCRIPT_DIR / fname, output_dir / fname)
    for kname in ("sm86_a16_blockh4.cu", "avx512_a16_blockh4.c"):
        src_k = SCRIPT_DIR / "kernels" / kname
        if src_k.is_file():
            shutil.copy2(src_k, output_dir / "kernels" / kname)

    # Copy upstream tokenizer, generation_config, and LICENSE if source_dir is supplied.
    copied_upstream: list[str] = []
    if source_dir is not None and source_dir.is_dir():
        for fname in TOKENIZER_AND_META_FILES:
            fp = source_dir / fname
            if fp.is_file():
                shutil.copy2(fp, output_dir / fname)
                copied_upstream.append(fname)

    # Ensure upstream license pointer file exists if upstream repo did not include a local LICENSE file.
    if not (output_dir / "LICENSE").is_file() and not (output_dir / "LICENSE.txt").is_file():
        notice_lines = [
            f"Upstream Base Model: {base_model_id}",
            f"License: {license_spec['license_summary']}",
        ]
        if license_spec.get("license_link"):
            notice_lines.append(f"Upstream License URL: {license_spec['license_link']}")
        (output_dir / "NOTICE_UPSTREAM_LICENSE.txt").write_text(
            "\n".join(notice_lines) + "\n", encoding="utf-8"
        )

    st_bytes = st_path.stat().st_size
    equiv_bf16_total = total_bf16_quant_bytes + non_block_bf16_bytes
    eff_bpw = 8.0 * float(total_packed_bytes) / float(total_quant_weights)
    stats: dict[str, Any] = {
        "format_version": SILIQUANT_FORMAT_VERSION,
        "base_model_id": base_model_id,
        "base_arch": sq_config.base_arch,
        "recipe": recipe,
        "bpw_target": float(bpw_key),
        "group_size": int(group_size),
        "rotation_policy": rotation,
        "perm_mode": perm_mode,
        "use_colscale": bool(use_colscale),
        "tie_word_embeddings": tie_emb,
        "quantized_layers": len(layer_map),
        "total_quantized_weights": total_quant_weights,
        "total_a16_blocks": total_a16_blocks,
        "total_syndrome_errors": total_syn_errs,
        "max_abs_roundtrip_diff": max_abs_roundtrip_diff,
        "packed_quant_bytes": total_packed_bytes,
        "equivalent_bf16_quant_bytes": total_bf16_quant_bytes,
        "non_block_bf16_bytes": non_block_bf16_bytes,
        "equivalent_bf16_total_bytes": equiv_bf16_total,
        "model_safetensors_bytes": st_bytes,
        "effective_packed_bpw": round(eff_bpw, 4),
        "quant_compression_ratio_vs_bf16": round(
            float(total_bf16_quant_bytes) / float(total_packed_bytes), 4
        ),
        "total_compression_ratio_vs_bf16": round(
            float(equiv_bf16_total) / float(st_bytes), 4
        ),
        "copied_upstream_files": copied_upstream,
        "license_metadata": license_spec,
    }

    readme_text = _build_model_card(
        repo_name=repo_name,
        base_model_id=base_model_id,
        license_spec=license_spec,
        recipe=recipe,
        bpw_key=bpw_key,
        group_size=group_size,
        stats=stats,
        benchmarks=benchmarks,
    )
    (output_dir / "README.md").write_text(readme_text, encoding="utf-8")

    # Compute SHA-256 of EVERY file in output_dir (excluding the receipt itself) and write receipt.
    file_sha256: dict[str, str] = {}
    for fp in sorted(output_dir.rglob("*")):
        if fp.is_file() and fp.name != "siliquant_verification_receipt.json":
            rel = fp.relative_to(output_dir).as_posix()
            file_sha256[rel] = _sha256_file(fp)

    receipt: dict[str, Any] = {
        "schema": "siliquant_hf_verification_receipt.v1",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repo_name": repo_name,
        "summary": stats,
        "benchmarks": benchmarks or {},
        "file_sha256": file_sha256,
        "layers": layer_receipts,
    }
    (output_dir / "siliquant_verification_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    return {
        "receipt": receipt,
        "ref_dequant_f32": ref_dequant_f32,
        "work_biases": work_biases,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export a Hugging Face Qwen2/Llama model to lossless bit-packed siliquant-hf-v1"
    )
    parser.add_argument("--source-dir", type=Path, required=True, help="Local HF model snapshot dir")
    parser.add_argument("--output-dir", type=Path, required=True, help="Target export dir (outside expedition/)")
    parser.add_argument("--base-model-id", type=str, required=True, help="Upstream HF repo ID")
    parser.add_argument("--repo-name", type=str, required=True, help="Target HF repo name")
    parser.add_argument("--group-size", type=int, choices=(32, 128), default=128)
    parser.add_argument("--bpw", type=str, choices=("4.5", "3.0"), default="4.5")
    parser.add_argument(
        "--rotation",
        type=str,
        choices=("axis", "block_h2", "block_h4", "h16", "projtype_uniform"),
        default="block_h4",
    )
    parser.add_argument(
        "--perm-mode",
        type=str,
        choices=("none", "quad_lpt_wcol", "quad_snake_wcol", "quad_lpt_down_wcol"),
        default="quad_lpt_down_wcol",
    )
    parser.add_argument("--recipe-name", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--benchmarks-json", type=Path, default=None)
    args = parser.parse_args(argv)

    from transformers import AutoModelForCausalLM

    benchmarks = None
    if args.benchmarks_json is not None:
        benchmarks = json.loads(args.benchmarks_json.read_text(encoding="utf-8"))

    dev = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.source_dir), local_files_only=True, dtype=torch.bfloat16
    )
    model.eval()
    result = export_model_to_siliquant_hf(
        model,
        args.output_dir,
        base_model_id=args.base_model_id,
        repo_name=args.repo_name,
        group_size=args.group_size,
        bpw_key=args.bpw,
        rotation=args.rotation,
        perm_mode=args.perm_mode,
        recipe_name=args.recipe_name,
        source_dir=args.source_dir,
        benchmarks=benchmarks,
        device=dev,
    )
    print(json.dumps(result["receipt"]["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
