#!/usr/bin/env python3
"""The run program of this record: that of clip-by-layer-qwen25-2026-11-18, which holds the rules X32:Q@KIND and X32mix.

The plan is PREREG.md in this folder. The split itself comes from allocate.py of the record
output-budget-gemma-e4b-2026-11-17, used unchanged:

  python3 experiments/bit-split-held-out-qwen25-2026-11-18/run.py --model qwen2.5-1.5b-instruct-fp16.gguf --kld valid.kld --text valid \
      --work DIR --out experiments/bit-split-held-out-qwen25-2026-11-18/results.jsonl X32:8@attn_q X32:16@attn_q ...
  python3 experiments/output-budget-gemma-e4b-2026-11-17/allocate.py --results PART-A-ROWS.jsonl > experiments/bit-split-held-out-qwen25-2026-11-18/allocate.out
"""
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("clip_run", HERE.parent / "clip-by-layer-qwen25-2026-11-18" / "run.py")
R = importlib.util.module_from_spec(spec)
sys.modules["clip_run"] = R
spec.loader.exec_module(R)

if __name__ == "__main__":
    R.main()
