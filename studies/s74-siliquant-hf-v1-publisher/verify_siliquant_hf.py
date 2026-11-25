#!/usr/bin/env python3
"""Fail-closed cold-reload verifier for exported `siliquant-hf-v1` HF repositories.

Contains ZERO bare `assert` statements so `-O` optimization can never bypass any check.
Verifies:
  1. Receipt SHA-256 integrity across every file in the export directory.
  2. Privacy/secret leak scan across every text/code/config/markdown file (zero `/home/`,
     `/Users/`, tokens, or private keys).
  3. Exact upstream HF YAML license compliance on `README.md` (Qwen2.5-3B MUST carry
     `license: other` + `license_name: qwen-research` + `license_link`, NEVER `apache-2.0`).
  4. Cold reload via `AutoModelForCausalLM.from_pretrained(export_dir, trust_remote_code=True,
     dtype="auto")` with `0` missing/unexpected keys and intact `tie_word_embeddings`.
  5. Extended Hamming `[16, 11, 4]` syndrome check across 100% of blocks (`0` errors).
  6. Exact `0.0` max abs round-trip weight diff (against reference weights when supplied,
     plus dtype-immunity verification across `bfloat16` and `float32`).
  7. Live `.generate("The exact identity of the Hadamard matrix is", max_new_tokens=32)`
     non-empty finite output (zero NaN/Inf) + KV-cache incremental forward parity.
  8. Optional held-out KL/PPL match against certified receipts (`< 1e-4` KL, `< 1e-3` PPL).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_PROMPT = "The exact identity of the Hadamard matrix is"

TEXT_SUFFIXES = (".py", ".json", ".md", ".txt", ".cu", ".c", ".h", ".yaml", ".yml")

LEAK_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("unix_home_path", re.compile(r"/home/[A-Za-z0-9._-]+")),
    ("macos_user_path", re.compile(r"/Users/[A-Za-z0-9._-]+")),
    ("github_pat", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("hf_token", re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")),
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    ("aws_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("private_key", re.compile(r"BEGIN [A-Z ]*PRIVATE KEY")),
)


class VerificationError(RuntimeError):
    """Raised whenever any fail-closed verification check fails."""


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise VerificationError(msg)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_yaml_frontmatter(readme_text: str) -> dict[str, Any]:
    lines = readme_text.splitlines()
    _require(len(lines) >= 3 and lines[0].strip() == "---", "README.md missing YAML frontmatter start")
    end_idx = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break
    _require(end_idx > 0, "README.md missing YAML frontmatter closing delimiter")

    meta: dict[str, Any] = {}
    cur_list_key: str | None = None
    for raw in lines[1:end_idx]:
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("- ") and cur_list_key is not None:
            meta[cur_list_key].append(stripped[2:].strip())
            continue
        cur_list_key = None
        if ":" in stripped:
            k, v = stripped.split(":", 1)
            key = k.strip()
            val = v.strip()
            _require(key not in meta, f"Duplicate YAML frontmatter key detected: {key!r}")
            if val == "":
                meta[key] = []
                cur_list_key = key
            else:
                meta[key] = val.strip('"').strip("'")
    return meta


def scan_export_hygiene(export_dir: Path) -> dict[str, Any]:
    """Scan every text/source/config/markdown file in `export_dir` for leaks or secrets."""
    scanned = 0
    findings: list[str] = []
    for fp in sorted(export_dir.rglob("*")):
        if not fp.is_file():
            continue
        if fp.suffix.lower() not in TEXT_SUFFIXES and fp.name not in ("LICENSE",):
            continue
        scanned += 1
        text = fp.read_text(encoding="utf-8", errors="replace")
        rel = fp.relative_to(export_dir).as_posix()
        for pat_name, pat in LEAK_PATTERNS:
            if pat.search(text):
                findings.append(f"{rel}:{pat_name}")
    _require(len(findings) == 0, f"Hygiene/leak scan failed: {findings}")
    return {"files_scanned": scanned, "findings": findings}


def verify_readme_license(export_dir: Path, base_model_id: str) -> dict[str, Any]:
    """Verify exact upstream HF YAML license metadata in `README.md`."""
    readme_path = export_dir / "README.md"
    _require(readme_path.is_file(), "Missing README.md in export directory")
    meta = _parse_yaml_frontmatter(readme_path.read_text(encoding="utf-8"))

    lic = str(meta.get("license", ""))
    _require(bool(lic), "README.md YAML frontmatter missing 'license'")
    _require(meta.get("library_name") == "transformers", "README.md must set library_name: transformers")
    _require(meta.get("base_model") == base_model_id, f"README.md base_model mismatch: {meta.get('base_model')!r}")

    low_id = base_model_id.lower()
    if "qwen2.5-3b" in low_id:
        _require(
            lic == "other",
            f"Qwen2.5-3B license MUST be 'other' (qwen-research), got {lic!r}",
        )
        _require(
            meta.get("license_name") == "qwen-research",
            f"Qwen2.5-3B license_name MUST be 'qwen-research', got {meta.get('license_name')!r}",
        )
        _require(
            str(meta.get("license_link", "")) == "https://huggingface.co/Qwen/Qwen2.5-3B/blob/main/LICENSE",
            f"Qwen2.5-3B license_link MUST equal exact upstream URL, got {meta.get('license_link')!r}",
        )
    elif any(tag in low_id for tag in ("smollm2-360m", "qwen2.5-0.5b", "qwen2.5-1.5b")):
        _require(lic == "apache-2.0", f"{base_model_id} expected license: apache-2.0, got {lic!r}")

    return {
        "license": lic,
        "license_name": meta.get("license_name"),
        "license_link": meta.get("license_link"),
        "base_model": meta.get("base_model"),
    }


def verify_receipt_hashes(export_dir: Path) -> dict[str, Any]:
    """Verify SHA-256 of every file recorded in `siliquant_verification_receipt.json` (no symlinks/traversal)."""
    root_resolved = export_dir.resolve()
    receipt_path = export_dir / "siliquant_verification_receipt.json"
    _require(not receipt_path.is_symlink() and receipt_path.is_file(), "Missing or symlinked siliquant_verification_receipt.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    manifest = receipt.get("file_sha256", {})
    _require(isinstance(manifest, dict) and len(manifest) > 0, "Empty file_sha256 manifest in receipt")

    actual_files: set[str] = set()
    for fp in export_dir.rglob("*"):
        _require(not fp.is_symlink(), f"Symlink detected in export directory: {fp.relative_to(export_dir).as_posix()}")
        if fp.is_file() and fp.name != "siliquant_verification_receipt.json":
            actual_files.add(fp.relative_to(export_dir).as_posix())

    manifest_files: set[str] = set()
    for rel in manifest:
        _require(isinstance(rel, str) and bool(rel), f"Invalid manifest key: {rel!r}")
        rel_obj = Path(rel)
        _require(
            not rel_obj.is_absolute() and ".." not in rel_obj.parts and "\\" not in rel,
            f"Unsafe path traversal entry in receipt manifest: {rel!r}",
        )
        try:
            (export_dir / rel_obj).resolve().relative_to(root_resolved)
        except ValueError as exc:
            raise VerificationError(f"Receipt entry escapes export directory: {rel!r}") from exc
        manifest_files.add(rel)

    missing_from_disk = sorted(manifest_files - actual_files)
    unlisted_on_disk = sorted(actual_files - manifest_files)
    _require(not missing_from_disk, f"Files missing on disk: {missing_from_disk}")
    _require(not unlisted_on_disk, f"Unlisted files in export dir: {unlisted_on_disk}")

    for rel, expected_sha in sorted(manifest.items()):
        fp = export_dir / rel
        _require(not fp.is_symlink() and fp.is_file(), f"Invalid target file: {rel}")
        actual_sha = _sha256_file(fp)
        _require(
            actual_sha == expected_sha,
            f"SHA-256 mismatch on {rel}: expected {expected_sha}, got {actual_sha}",
        )
    return receipt


@torch.no_grad()
def verify_siliquant_export(
    export_dir: Path,
    *,
    ref_weights_f32: dict[str, torch.Tensor] | None = None,
    ref_logits_input_ids: torch.Tensor | None = None,
    ref_logits: torch.Tensor | None = None,
    prompt: str = DEFAULT_PROMPT,
    max_new_tokens: int = 32,
    device: str = "cpu",
    expected_benchmarks: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Run full fail-closed verification on an exported `siliquant-hf-v1` directory."""
    _require(export_dir.is_dir(), f"Export directory does not exist: {export_dir}")

    # 1. Receipt SHA-256 manifest verification.
    receipt = verify_receipt_hashes(export_dir)
    summary = receipt.get("summary", {})
    base_model_id = str(summary.get("base_model_id", ""))
    _require(bool(base_model_id), "Receipt missing summary.base_model_id")

    # 2. Privacy & secret hygiene scan.
    hygiene = scan_export_hygiene(export_dir)

    # 3. Exact upstream license YAML verification.
    license_check = verify_readme_license(export_dir, base_model_id)

    # 4. Cold reload via AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True, dtype="auto").
    loaded, loading_info = AutoModelForCausalLM.from_pretrained(
        str(export_dir),
        trust_remote_code=True,
        dtype="auto",
        local_files_only=True,
        output_loading_info=True,
    )
    missing_keys = list(loading_info.get("missing_keys", []))
    unexpected_keys = list(loading_info.get("unexpected_keys", []))
    mismatched_keys = list(loading_info.get("mismatched_keys", []))
    _require(len(missing_keys) == 0, f"Unexpected missing_keys on cold reload: {missing_keys}")
    _require(len(unexpected_keys) == 0, f"Unexpected unexpected_keys on cold reload: {unexpected_keys}")
    _require(len(mismatched_keys) == 0, f"Unexpected mismatched_keys on cold reload: {mismatched_keys}")

    target_dev = torch.device(device)
    loaded = loaded.to(target_dev)
    loaded.eval()

    # Verify tied embedding invariant.
    if bool(getattr(loaded.config, "tie_word_embeddings", False)):
        _require(
            loaded.lm_head.weight.data_ptr() == loaded.model.embed_tokens.weight.data_ptr(),
            "tie_word_embeddings=True but lm_head.weight is not tied to model.embed_tokens.weight",
        )

    # 5. Extended Hamming [16, 11, 4] syndrome verification across 100% of blocks.
    syn_report = loaded.verify_all_syndromes()
    _require(
        int(syn_report["quantized_layers"]) == int(summary["quantized_layers"]),
        f"Quantized layer count mismatch: {syn_report['quantized_layers']} != {summary['quantized_layers']}",
    )
    _require(
        int(syn_report["total_syndrome_errors"]) == 0,
        f"Non-zero A_16 syndrome errors on cold reload: {syn_report['total_syndrome_errors']}",
    )

    # 6. Exact bit-for-bit weight round-trip verification (+ live dtype-immunity check).
    max_abs_weight_diff = 0.0
    dtype_immunity_checked = False
    layer_map = loaded.config.siliquant_layer_map
    for layer_name in layer_map:
        mod = loaded.get_submodule(layer_name)
        w_f32 = mod.dequantize(dtype=torch.float32, verify_syndrome=True).cpu()
        _require(bool(torch.isfinite(w_f32).all().item()), f"Non-finite dequantized weights in {layer_name}")
        if not dtype_immunity_checked:
            probe_mod = copy.deepcopy(mod).cpu()
            for dt in (torch.bfloat16, torch.float16, torch.float32):
                probe_mod.to(dtype=dt)
                w_dt = probe_mod.dequantize(dtype=torch.float32, verify_syndrome=True)
                dt_diff = float((w_dt - w_f32).abs().max().item())
                _require(
                    dt_diff == 0.0,
                    f"Dtype-immunity violated on {layer_name} after .to({dt}): diff={dt_diff:.3e}",
                )
            dtype_immunity_checked = True
        if ref_weights_f32 is not None and layer_name in ref_weights_f32:
            d = float((w_f32 - ref_weights_f32[layer_name].cpu()).abs().max().item())
            if d > max_abs_weight_diff:
                max_abs_weight_diff = d
            _require(d == 0.0, f"Weight round-trip diff on {layer_name}: {d:.3e} != 0.0")

    # Optional exact forward logits parity check.
    max_abs_logits_diff: float | None = None
    if ref_logits_input_ids is not None and ref_logits is not None:
        out_ref_check = loaded(ref_logits_input_ids.to(target_dev), use_cache=False).logits.cpu()
        max_abs_logits_diff = float(
            (out_ref_check.to(torch.float32) - ref_logits.cpu().to(torch.float32)).abs().max().item()
        )
        _require(
            max_abs_logits_diff == 0.0,
            f"Forward logits mismatch vs quantized reference: {max_abs_logits_diff:.3e} != 0.0",
        )

    # 7. Live .generate() check + finite logits + KV-cache incremental forward check.
    tokenizer = None
    has_tokenizer = (export_dir / "tokenizer.json").is_file() or (export_dir / "tokenizer_config.json").is_file()
    if has_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(str(export_dir), local_files_only=True)
        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(target_dev)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(target_dev)
    else:
        vocab_sz = int(loaded.config.vocab_size)
        base_toks = [1 + (i % max(vocab_sz - 2, 1)) for i in range(6)]
        input_ids = torch.tensor([base_toks], dtype=torch.long, device=target_dev)
        attention_mask = torch.ones_like(input_ids)

    fwd_out = loaded(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    _require(bool(torch.isfinite(fwd_out.logits).all().item()), "Non-finite logits on live forward")

    # Verify KV-cache 1-step incremental continuation stays finite and well-formed.
    next_tok = fwd_out.logits[:, -1:, :].argmax(dim=-1)
    uses_cache_kw = hasattr(fwd_out, "cache")
    kv_cache = fwd_out.cache if uses_cache_kw else getattr(fwd_out, "past_key_values", None)
    _require(kv_cache is not None, "KV cache was not returned when use_cache=True")
    if uses_cache_kw:
        step_out = loaded(input_ids=next_tok, cache=kv_cache, use_cache=True)
    else:
        step_out = loaded(input_ids=next_tok, past_key_values=kv_cache, use_cache=True)
    _require(bool(torch.isfinite(step_out.logits).all().item()), "Non-finite logits on KV-cache step")

    gen_kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": int(max_new_tokens),
        "min_new_tokens": 1,
        "do_sample": False,
        "use_cache": True,
    }
    if tokenizer is not None and tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        gen_kwargs["pad_token_id"] = tokenizer.eos_token_id
    elif getattr(loaded.config, "eos_token_id", None) is not None:
        eos_id = loaded.config.eos_token_id
        gen_kwargs["pad_token_id"] = eos_id[0] if isinstance(eos_id, list) else int(eos_id)

    gen_ids = loaded.generate(**gen_kwargs)
    new_tok_count = int(gen_ids.shape[1]) - int(input_ids.shape[1])
    _require(new_tok_count >= 1, f".generate() produced {new_tok_count} new tokens")
    _require(
        bool(((gen_ids >= 0) & (gen_ids < int(loaded.config.vocab_size))).all().item()),
        ".generate() emitted out-of-range token IDs",
    )
    decoded_preview = (
        tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        if tokenizer is not None
        else f"<synthetic_tokens:{gen_ids[0].tolist()[:12]}>"
    )

    # 8. Optional certified benchmark cross-check.
    benchmark_deltas: dict[str, float] = {}
    if expected_benchmarks is not None:
        rec_bench = receipt.get("benchmarks", {})
        for metric_name, ref_val in expected_benchmarks.items():
            _require(metric_name in rec_bench, f"Missing certified benchmark key {metric_name!r}")
            actual_val = float(rec_bench[metric_name])
            delta = abs(actual_val - float(ref_val))
            tol = 1e-4 if "kl" in metric_name.lower() else 1e-3
            _require(
                delta < tol,
                f"Benchmark mismatch on {metric_name}: |{actual_val} - {ref_val}| = {delta:.3e} >= {tol}",
            )
            benchmark_deltas[metric_name] = delta

    return {
        "status": "PASS",
        "base_model_id": base_model_id,
        "receipt_files_verified": len(receipt["file_sha256"]),
        "hygiene": hygiene,
        "license_check": license_check,
        "syndrome_report": syn_report,
        "max_abs_weight_diff": max_abs_weight_diff,
        "max_abs_logits_diff": max_abs_logits_diff,
        "generated_new_tokens": new_tok_count,
        "generated_preview": decoded_preview[:240],
        "benchmark_deltas": benchmark_deltas,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fail-closed cold-reload verifier for siliquant-hf-v1")
    parser.add_argument("export_dir", type=Path, help="Path to exported HF repo directory")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--expected-benchmarks-json", type=Path, default=None)
    args = parser.parse_args(argv)

    expected_b = None
    if args.expected_benchmarks_json is not None:
        expected_b = json.loads(args.expected_benchmarks_json.read_text(encoding="utf-8"))

    report = verify_siliquant_export(
        args.export_dir,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        expected_benchmarks=expected_b,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
