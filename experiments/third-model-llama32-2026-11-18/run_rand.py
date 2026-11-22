#!/usr/bin/env python3
"""The program of the record imatrix-gain-parts-2026-11-18 on a third model: Llama-3.2-1B-Instruct.

The plan is PREREG.md in this folder. This file adds nothing of its own. It loads that record's program, adds this
model's file to the files it accepts, and runs it: for the forecast, and for the ball arm with the matrix of no meaning.

Usage:
  python3 experiments/third-model-llama32-2026-11-18/run_rand.py --model Llama-3.2-1B-Instruct-f16.gguf --imatrix llama32.imatrix.gguf \
      --imatrix-rand llama32.rimatrix.gguf --forecast > forecast.out
  ... --kld base.kld --text wikitext --work DIR --out results-wikitext.jsonl --bin llama-perplexity 'hZBG128:b560s16~rand'
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHA256 = "1f33ad43d2b85b908ff06fe7002b69806a57359b9b2617ca27d7bdea428ae146"
NAME = "Llama-3.2-1B-Instruct-f16.gguf"

spec = importlib.util.spec_from_file_location("gain_run", HERE.parent / "imatrix-gain-parts-2026-11-18" / "run.py")
gain_run = importlib.util.module_from_spec(spec)
sys.modules["gain_run"] = gain_run          # so that the worker processes find its functions by name
spec.loader.exec_module(gain_run)
gain_run.Q.MODELS[SHA256] = NAME
gain_run.ACCEPTED = gain_run.ACCEPTED + (NAME,)

if __name__ == "__main__":
    gain_run.main()
