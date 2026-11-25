#!/usr/bin/env python3
"""Standalone CLI inspector, SHA-256 receipt verifier, and unpacker for `siliquant-hf-v1` models.

Uses only `torch`, `transformers`, and `safetensors` (zero custom C++/CUDA builds required).
Note: `qham11` stores the 11 systematic bits of Extended Hamming [16, 11, 4], so syndrome
checking verifies codec/lattice consistency while end-to-end payload tamper integrity across
`qquot`, `qham11`, scales, and metadata is enforced via `siliquant_verification_receipt.json`
SHA-256 manifest verification (rejecting missing, tampered, unlisted, symlinked, or path-
traversal entries).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from modeling_siliquant import dequantize_a16_weight, unpack_a16_integers


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# Download/runtime artifacts created outside the export itself: `huggingface_hub.snapshot_download`
# writes a top-level `.gitattributes` and `.cache/huggingface/**` metadata into `local_dir`, and
# Python writes `__pycache__/**/*.pyc`. These (and only these) are dropped from the on-disk set
# before the unlisted-file check; symlinks anywhere (including inside these paths) still fail
# closed in the rglob enumeration below. This file is copied standalone into HF repos, so the
# helper is duplicated in verify_siliquant_hf.py rather than imported.
IGNORED_EXPORT_ARTIFACT_TOP_FILE = ".gitattributes"
IGNORED_EXPORT_ARTIFACT_CACHE_PREFIX = ".cache/huggingface/"


def _is_ignored_export_artifact(rel_posix: str) -> bool:
    """True only for top-level `.gitattributes`, anything under `.cache/huggingface/`, and `__pycache__` `*.pyc` paths."""
    if rel_posix == IGNORED_EXPORT_ARTIFACT_TOP_FILE:
        return True
    if rel_posix.startswith(IGNORED_EXPORT_ARTIFACT_CACHE_PREFIX):
        return True
    return "__pycache__" in rel_posix.split("/") and rel_posix.endswith(".pyc")


def _load_config_and_tensors(model_dir: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    cfg_path = model_dir / "config.json"
    st_path = model_dir / "model.safetensors"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing config.json in {model_dir}")
    if not st_path.is_file():
        raise FileNotFoundError(f"Missing model.safetensors in {model_dir}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if cfg.get("siliquant_format_version") != "siliquant-hf-v1":
        raise ValueError(
            f"Unsupported siliquant_format_version={cfg.get('siliquant_format_version')!r}"
        )
    tensors = load_file(str(st_path), device="cpu")
    return cfg, tensors


def inspect_model(model_dir: Path) -> dict[str, Any]:
    """Inspect bit-packed model layout, effective bpw, and verify 100% of A_16 syndromes."""
    cfg, tensors = _load_config_and_tensors(model_dir)
    layer_map: dict[str, dict[str, Any]] = cfg.get("siliquant_layer_map", {})

    total_quant_weights = 0
    total_packed_bytes = 0
    total_bf16_bytes = 0
    total_syndrome_errors = 0
    total_a16_blocks = 0
    k_bits_hist: dict[str, int] = {}
    rot_hist: dict[str, int] = {}

    for layer_name, meta in layer_map.items():
        m = int(meta["out_features"])
        k = int(meta["in_features"])
        k_bits = int(meta["k_bits"])
        rot = str(meta["rotation"])
        n_ov = int(meta.get("n_overflow", 0))

        qquot = tensors[f"{layer_name}.qquot"]
        qham11 = tensors[f"{layer_name}.qham11"]
        scales_bits = tensors[f"{layer_name}.scales_fp16_bits"]
        rowptr = tensors.get(f"{layer_name}.qov_rowptr") if n_ov > 0 else None
        qov_col = tensors.get(f"{layer_name}.qov_col") if n_ov > 0 else None
        qov_val = tensors.get(f"{layer_name}.qov_val") if n_ov > 0 else None
        colscale_bits = tensors.get(f"{layer_name}.colscale_fp32_bits")
        perm_in = tensors.get(f"{layer_name}.perm_in")

        _, syn_errs = unpack_a16_integers(
            qquot,
            qham11,
            out_features=m,
            in_features=k,
            k_bits=k_bits,
            qov_rowptr=rowptr,
            qov_col=qov_col,
            qov_val=qov_val,
            verify_syndrome=True,
        )
        total_syndrome_errors += syn_errs
        total_a16_blocks += m * (k // 16)
        total_quant_weights += m * k
        total_bf16_bytes += m * k * 2

        layer_bytes = 0
        for t in (qquot, qham11, scales_bits, rowptr, qov_col, qov_val, colscale_bits, perm_in):
            if t is not None:
                layer_bytes += int(t.numel()) * int(t.element_size())
        total_packed_bytes += layer_bytes

        kb_key = str(k_bits)
        k_bits_hist[kb_key] = k_bits_hist.get(kb_key, 0) + 1
        rot_hist[rot] = rot_hist.get(rot, 0) + 1

    st_bytes = (model_dir / "model.safetensors").stat().st_size
    eff_bpw = (8.0 * total_packed_bytes / total_quant_weights) if total_quant_weights > 0 else 0.0
    compression_ratio = (total_bf16_bytes / total_packed_bytes) if total_packed_bytes > 0 else 0.0

    return {
        "format_version": cfg["siliquant_format_version"],
        "base_arch": cfg["base_arch"],
        "recipe": cfg.get("siliquant_recipe"),
        "quantized_layers": len(layer_map),
        "total_quantized_weights": total_quant_weights,
        "total_a16_blocks": total_a16_blocks,
        "total_syndrome_errors": total_syndrome_errors,
        "packed_quant_bytes": total_packed_bytes,
        "equivalent_bf16_quant_bytes": total_bf16_bytes,
        "effective_packed_bpw": round(eff_bpw, 4),
        "quant_compression_ratio_vs_bf16": round(compression_ratio, 4),
        "model_safetensors_bytes": st_bytes,
        "k_bits_histogram": k_bits_hist,
        "rotation_histogram": rot_hist,
    }


def verify_receipt_and_syndromes(model_dir: Path) -> dict[str, Any]:
    """Verify exact SHA-256 receipt manifest (the tamper/integrity guarantee) and unpack codec consistency.

    Note: `total_syndrome_errors` on unpack is `0` by construction because `_unpack_hamming11`
    rebuilds the 5 parity bits from the 11 stored info bits; tamper detection across `model.safetensors`,
    `qquot`, `qham11`, scales, and metadata is enforced by the SHA-256 manifest check below.
    """
    root_resolved = model_dir.resolve()
    receipt_path = model_dir / "siliquant_verification_receipt.json"
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise FileNotFoundError(f"Missing or symlinked siliquant_verification_receipt.json in {model_dir}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    file_hashes: dict[str, str] = receipt.get("file_sha256", {})
    if not isinstance(file_hashes, dict) or not file_hashes:
        raise ValueError("Receipt contains empty file_sha256 manifest")

    # Reject symlinks anywhere under model_dir and enumerate exact relative file set.
    actual_files: set[str] = set()
    for fp in model_dir.rglob("*"):
        if fp.is_symlink():
            raise ValueError(f"Symlink detected in export directory: {fp.relative_to(model_dir).as_posix()}")
        if fp.is_file() and fp.name != "siliquant_verification_receipt.json":
            actual_files.add(fp.relative_to(model_dir).as_posix())

    # Validate every manifest key is a safe relative path staying strictly inside model_dir.
    manifest_files: set[str] = set()
    for rel_path in file_hashes:
        if not isinstance(rel_path, str) or not rel_path:
            raise ValueError(f"Invalid manifest entry key: {rel_path!r}")
        rel_obj = Path(rel_path)
        if rel_obj.is_absolute() or ".." in rel_obj.parts or "\\" in rel_path:
            raise ValueError(f"Unsafe path traversal entry in receipt manifest: {rel_path!r}")
        resolved_target = (model_dir / rel_obj).resolve()
        try:
            resolved_target.relative_to(root_resolved)
        except ValueError as exc:
            raise ValueError(f"Receipt entry escapes export directory: {rel_path!r}") from exc
        manifest_files.add(rel_path)

    # Drop ONLY the download/runtime artifacts (top-level .gitattributes, .cache/huggingface/**,
    # __pycache__/**/*.pyc) from the on-disk set before the unlisted-file check; every symlink was
    # already rejected during enumeration above, so symlinks inside ignored paths still fail.
    actual_files = {rel for rel in actual_files if not _is_ignored_export_artifact(rel)}

    missing_from_disk = sorted(manifest_files - actual_files)
    unlisted_on_disk = sorted(actual_files - manifest_files)
    if missing_from_disk:
        raise ValueError(f"Files listed in receipt missing on disk: {missing_from_disk}")
    if unlisted_on_disk:
        raise ValueError(f"Unlisted files present in export directory: {unlisted_on_disk}")

    mismatches: list[str] = []
    for rel_path, expected_sha in sorted(file_hashes.items()):
        fp = model_dir / rel_path
        if fp.is_symlink() or not fp.is_file():
            mismatches.append(f"INVALID:{rel_path}")
            continue
        actual_sha = _sha256_file(fp)
        if actual_sha != expected_sha:
            mismatches.append(f"SHA_MISMATCH:{rel_path}")

    if mismatches:
        raise ValueError(f"Receipt verification failed: {mismatches}")

    summary = inspect_model(model_dir)
    if summary["total_syndrome_errors"] != 0:
        raise ValueError(
            f"Non-zero A_16 syndrome errors detected: {summary['total_syndrome_errors']}"
        )
    summary["receipt_files_verified"] = len(file_hashes)
    summary["status"] = "PASS"
    return summary


def unpack_to_safetensors(model_dir: Path, out_path: Path, dtype_name: str = "bfloat16") -> dict[str, Any]:
    """Unpack a `siliquant-hf-v1` checkpoint into a standard dense HF safetensors file."""
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_name not in dtype_map:
        raise ValueError(f"Unsupported dtype={dtype_name!r}; expected {tuple(dtype_map)}")
    target_dtype = dtype_map[dtype_name]

    cfg, tensors = _load_config_and_tensors(model_dir)
    layer_map: dict[str, dict[str, Any]] = cfg.get("siliquant_layer_map", {})

    consumed_keys: set[str] = set()
    out_tensors: dict[str, torch.Tensor] = {}
    total_syn_errs = 0

    for layer_name, meta in layer_map.items():
        m = int(meta["out_features"])
        k = int(meta["in_features"])
        g = int(meta["group_size"])
        rot = str(meta["rotation"])
        k_bits = int(meta["k_bits"])
        n_ov = int(meta.get("n_overflow", 0))

        keys = [f"{layer_name}.qquot", f"{layer_name}.qham11", f"{layer_name}.scales_fp16_bits"]
        rowptr = qov_col = qov_val = None
        if n_ov > 0:
            rowptr = tensors[f"{layer_name}.qov_rowptr"]
            qov_col = tensors[f"{layer_name}.qov_col"]
            qov_val = tensors[f"{layer_name}.qov_val"]
            keys.extend([f"{layer_name}.qov_rowptr", f"{layer_name}.qov_col", f"{layer_name}.qov_val"])
        colscale_bits = tensors.get(f"{layer_name}.colscale_fp32_bits")
        if colscale_bits is not None:
            keys.append(f"{layer_name}.colscale_fp32_bits")
        perm_in = tensors.get(f"{layer_name}.perm_in")
        if perm_in is not None:
            keys.append(f"{layer_name}.perm_in")

        wq, syn_errs = dequantize_a16_weight(
            tensors[f"{layer_name}.qquot"],
            tensors[f"{layer_name}.qham11"],
            tensors[f"{layer_name}.scales_fp16_bits"],
            out_features=m,
            in_features=k,
            group_size=g,
            rotation=rot,
            k_bits=k_bits,
            qov_rowptr=rowptr,
            qov_col=qov_col,
            qov_val=qov_val,
            colscale_fp32_bits=colscale_bits,
            perm_in=perm_in,
            verify_syndrome=True,
            target_dtype=target_dtype,
        )
        total_syn_errs += syn_errs
        out_tensors[f"{layer_name}.weight"] = wq.contiguous()
        consumed_keys.update(keys)

    for key, tensor in tensors.items():
        if key not in consumed_keys:
            out_tensors[key] = tensor.contiguous()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(out_tensors, str(out_path), metadata={"format": "pt", "unpacked_from": "siliquant-hf-v1"})
    return {
        "unpacked_layers": len(layer_map),
        "total_tensors": len(out_tensors),
        "total_syndrome_errors": total_syn_errs,
        "dtype": dtype_name,
        "output_bytes": out_path.stat().st_size,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Siliquant HF v1 inspector & unpacker CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ins = sub.add_parser("inspect", help="Inspect compression & verify A_16 syndromes")
    p_ins.add_argument("model_dir", type=Path)

    p_ver = sub.add_parser("verify", help="Verify SHA-256 receipt and A_16 syndromes")
    p_ver.add_argument("model_dir", type=Path)

    p_unp = sub.add_parser("unpack", help="Unpack packed weights to standard safetensors")
    p_unp.add_argument("model_dir", type=Path)
    p_unp.add_argument("--out", type=Path, required=True)
    p_unp.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")

    args = parser.parse_args(argv)
    if args.cmd == "inspect":
        res = inspect_model(args.model_dir)
    elif args.cmd == "verify":
        res = verify_receipt_and_syndromes(args.model_dir)
    else:
        res = unpack_to_safetensors(args.model_dir, args.out, args.dtype)
    print(json.dumps(res, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
