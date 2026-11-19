#!/usr/bin/env python3
"""The program of the record loud-columns-2026-11-18 on a third model: Llama-3.2-1B-Instruct.

The plan is extension 2 of PREREG.md in this folder. This file adds nothing of its own. It loads that record's
program, adds this model's file to the files it accepts, and runs it: for the table of loud and active columns,
and for the ball arm with columns of ffn_down left whole.

Usage:
  python3 experiments/third-model-llama32-2026-11-18/run_loud.py --model Llama-3.2-1B-Instruct-f16.gguf --imatrix llama32.imatrix.gguf --picks > picks.out
  ... --kld base.kld --text wikitext --work DIR --out results-loud.jsonl --bin llama-perplexity 'ZBG128:b560s16+K10'
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHA256 = "1f33ad43d2b85b908ff06fe7002b69806a57359b9b2617ca27d7bdea428ae146"
NAME = "Llama-3.2-1B-Instruct-f16.gguf"

spec = importlib.util.spec_from_file_location("loud_run", HERE.parent / "loud-columns-2026-11-18" / "run.py")
loud_run = importlib.util.module_from_spec(spec)
sys.modules["loud_run"] = loud_run          # so that the worker processes find its functions by name
spec.loader.exec_module(loud_run)
loud_run.Q.MODELS[SHA256] = NAME
loud_run.ACCEPTED = loud_run.ACCEPTED + (NAME,)

if __name__ == "__main__":
    loud_run.main()
