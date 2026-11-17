#!/usr/bin/env python3
"""The arms of the run on Qwen2.5, on a third model: SmolLM2-1.7B-Instruct.

The plan is PREREG.md in this folder. This file adds nothing of its own. It loads the program
of the record model-output-qwen25-2026-11-17, adds this model's file to the files that program
accepts, and runs it. So every arm is built and scored by the same code as there.

Usage, with NumPy and torch, llama.cpp's gguf package on GGUF_PY and llama-perplexity:
  python3 experiments/model-output-smollm2-2026-11-17/run.py --model SmolLM2-1.7B-Instruct-f16.gguf \
      --kld smollm2-base.kld --work DIR --out experiments/model-output-smollm2-2026-11-17/results.jsonl \
      --bin llama-perplexity VARIANT [VARIANT ...]
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHA256 = "cf3f0569a9d0b56d5d3b5b7944768c71bd2fef789f8b36b1b85a05d390850852"

spec = importlib.util.spec_from_file_location("qwen_run", HERE.parent / "model-output-qwen25-2026-11-17" / "run.py")
qwen_run = importlib.util.module_from_spec(spec)
sys.modules["qwen_run"] = qwen_run          # so that the worker processes find its functions by name
spec.loader.exec_module(qwen_run)
qwen_run.MODELS[SHA256] = "SmolLM2-1.7B-Instruct-f16.gguf"

if __name__ == "__main__":
    qwen_run.main()
