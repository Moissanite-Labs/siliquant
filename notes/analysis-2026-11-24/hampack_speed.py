#!/usr/bin/env python3
"""Decode speed on one core: the packed integer format (ballpack) against the same block with A_16, A_32 or A_8 (hampack).

A Laplace matrix of 1024 x 4096 is packed by each format, and the blocks are decoded to 32-bit floats
by each coder's whole decode (number to point to scaled floats), best of 5. Also the weight error of
each file. Run from the root: python3 notes/analysis-2026-11-24/hampack_speed.py
"""
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "shape"))
import ballpack  # noqa: E402
from hampack import HamFormat  # noqa: E402

W = np.random.default_rng(0).laplace(size=(1024, 4096)).astype(np.float32)
print("| Format | Bound | Table entries | Weight error (dB) | Decode, million weights a second |")
print("|---|---|---|---|---|")
for name, f in (("Z^32 (ballpack)", ballpack.Format(32, 136)), ("A_16 x A_16", HamFormat(16, 32, 136)),
                ("A_32", HamFormat(32, 32, 136)), ("A_8^4 (E8 turned)", HamFormat(8, 32, 136))):
    stored, Wh, _ = f.encode(W)
    e = 10 * math.log10(float(np.sum((W - Wh.astype(np.float64)) ** 2) / np.sum(W.astype(np.float64) ** 2)))
    sec, _ = f.timing(stored["records"], float(stored["top"]), repeats=5)
    entries = f.table_entries() if hasattr(f, "states") else (f.n + 1) * (f.bound + 1)
    print(f"| {name} | {f.bound} | {entries} | {e:.3f} | {W.size / sec / 1e6:.1f} |", flush=True)
