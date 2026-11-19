#!/usr/bin/env python3
"""Not in either plan: what the centre of the harness's E8 Voronoi code costs in weight error.

The harness subtracts the mean of its codewords from every codeword, so none is zero. The run
on Qwen2.5 found that this breaks the code on a model's output when one scale serves a row.
The records of 2026-11-17 used that code as their reference for weight error. This program
asks what the centre did to those comparisons.

It rounds the 59 matrices of the record ball-dimension-no-clip-rule-2026-11-17 (6 synthetic,
32 of whisper, 21 of Qwen2.5) with the code as the harness has it (VE8) and with a codeword
at zero (VE8Z), in three of that record's cells, and sets both beside that record's stored
errors for E8 cut by a ball over 8 weights (EB8) and for the scalar rule (X).

Weight error only. It checks first that its VE8 is the stored VE8 of that record.

Usage, from the root of a checkout, with llama.cpp's gguf package on GGUF_PY:
  python3 experiments/model-output-qwen25-2026-11-17/centre.py --weights weights.npz --gguf qwen2.5-1.5b-instruct-fp16.gguf > centre.out
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "harness"))
if os.environ.get("GGUF_PY"):
    sys.path.insert(0, os.environ["GGUF_PY"])

import torch  # noqa: E402

torch.set_num_threads(1)
import tq  # noqa: E402

BALL = ROOT / "experiments" / "ball-dimension-no-clip-rule-2026-11-17"
CELLS = (("32", 16), ("R", 16), ("R", 32))


def db(x):
    return 10.0 * math.log10(x)


def voronoi(W, q, group, zero):
    code = tq.Code("E8", q)
    kept = code.centre
    if zero:
        code.centre = torch.zeros_like(kept)
    try:
        return tq.maxscale_voronoi(W, q, "R" if group == "R" else int(group), name="E8")[1]["rel_err"]
    finally:
        code.centre = kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--gguf", required=True)
    args = ap.parse_args()
    spec = importlib.util.spec_from_file_location("ball_dimension_run", BALL / "run.py")
    bd = importlib.util.module_from_spec(spec)
    sys.modules["ball_dimension_run"] = bd
    spec.loader.exec_module(bd)
    bd._weights_path, bd._gguf_path = args.weights, args.gguf
    stored = {}
    for r in json.load(open(BALL / "results.json")):
        stored[(r["matrix"], r["arm"], r["n"], r["group"], r["q"])] = r["rel_err"]
    names = bd.matrix_names()
    out = {}
    worst = 0.0
    for name in names:
        W = bd.matrix(name)
        for group, q in CELLS:
            v = voronoi(W, q, group, zero=False)
            z = voronoi(W, q, group, zero=True)
            out[(name, group, q)] = (v, z)
            worst = max(worst, abs(db(v / stored[(name, "VE8", 8, group, q)])))
    print("The centre of the harness's E8 Voronoi code, in weight error, on the 59 matrices of the record ball-dimension-no-clip-rule-2026-11-17")
    print(f"check: this program's VE8 against that record's stored VE8, largest difference over {len(out)} runs: {worst:.4f} dB")
    print()
    fam = lambda n: n.split("/", 1)[0]
    fams = ["G", "T", "whisper", "qwen"]
    print("1. A codeword at zero against the code as the harness has it: gain of VE8Z over VE8 in dB of weight error")
    print("family | cell | matrices | mean | least | most | matrices where VE8Z is better")
    for f in fams:
        ms = [n for n in names if fam(n) == f]
        for group, q in CELLS:
            g = [db(out[(n, group, q)][0] / out[(n, group, q)][1]) for n in ms]
            print(f"{f} | {group}:{q} | {len(ms)} | {np.mean(g):+.3f} | {np.min(g):+.3f} | {np.max(g):+.3f} | {sum(x > 0 for x in g)}")
    print()
    print("2. E8 cut by a ball over 8 weights (EB8, stored in that record) against each: gain of EB8 in dB")
    print("family | cell | EB8 over VE8: mean | matrices where EB8 is better | EB8 over VE8Z: mean | least | most | matrices where EB8 is better")
    for f in fams:
        ms = [n for n in names if fam(n) == f]
        for group, q in CELLS:
            a = [db(out[(n, group, q)][0] / stored[(n, "EB", 8, group, q)]) for n in ms]
            b = [db(out[(n, group, q)][1] / stored[(n, "EB", 8, group, q)]) for n in ms]
            print(f"{f} | {group}:{q} | {np.mean(a):+.3f} | {sum(x > 0 for x in a)} of {len(ms)} | {np.mean(b):+.3f} | {np.min(b):+.3f} | {np.max(b):+.3f} | {sum(x > 0 for x in b)} of {len(ms)}")
    print()
    print("3. Each over the scalar rule X (stored in that record): mean gain in dB")
    print("family | cell | VE8 | VE8Z | EB8")
    for f in fams:
        ms = [n for n in names if fam(n) == f]
        for group, q in CELLS:
            x = [stored[(n, "X", 1, group, q)] for n in ms]
            print(f"{f} | {group}:{q} | {np.mean([db(xx / out[(n, group, q)][0]) for xx, n in zip(x, ms)]):+.2f} | "
                  f"{np.mean([db(xx / out[(n, group, q)][1]) for xx, n in zip(x, ms)]):+.2f} | "
                  f"{np.mean([db(xx / stored[(n, 'EB', 8, group, q)]) for xx, n in zip(x, ms)]):+.2f}")


if __name__ == "__main__":
    main()
