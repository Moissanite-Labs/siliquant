#!/usr/bin/env python3
"""The arms of the model-output records on a third model: Llama-3.2-1B-Instruct.

The plan is PREREG.md in this folder. This file adds nothing of its own. It loads the program of the record
model-output-qwen25-2026-11-17, adds this model's file to the files that program accepts, and runs it. So every
llama.cpp file is compared and scored, and every ball arm is built and scored, by the same code as there.

Usage, with NumPy and torch, llama.cpp's gguf package on GGUF_PY and llama-perplexity:
  python3 experiments/third-model-llama32-2026-11-18/run_arms.py --model Llama-3.2-1B-Instruct-f16.gguf --kld base.kld --work DIR \
      --out results-wikitext.jsonl --bin llama-perplexity --imatrix llama32.imatrix.gguf file:llama32-Q4_K_M.gguf ZBG128:b560s16 hZBG128:b560s16
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHA256 = "1f33ad43d2b85b908ff06fe7002b69806a57359b9b2617ca27d7bdea428ae146"
NAME = "Llama-3.2-1B-Instruct-f16.gguf"

spec = importlib.util.spec_from_file_location("qwen_run", HERE.parent / "model-output-qwen25-2026-11-17" / "run.py")
qwen_run = importlib.util.module_from_spec(spec)
sys.modules["qwen_run"] = qwen_run          # so that the worker processes find its functions by name
spec.loader.exec_module(qwen_run)
qwen_run.MODELS[SHA256] = NAME

if __name__ == "__main__":
    qwen_run.main()
