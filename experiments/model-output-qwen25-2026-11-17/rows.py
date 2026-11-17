#!/usr/bin/env python3
"""Not in either plan: why the E8 Voronoi code with one scale per row breaks this model.

Written after 13 variants were scored. The E8 Voronoi code of the harness had a lower weight
error than the scalar rule on every kind of matrix and a mean KL six times as high with one
scale per row. This program looks at single rows, on 21 matrices of the model (layers 0, 13
and 27, all seven kinds), whole.

It runs three rules with one scale per row at 16 levels: the scalar rule (XR:16), the E8
Voronoi code of the harness (VE8R:16), and the same code with a codeword at zero (VE8ZR:16).
The harness subtracts the mean of its codewords, "the centre", from every codeword, so that
its points are E8 moved by a fixed vector and none of them is zero. The third rule leaves the
centre out.

It scores nothing on the model's output. It is weight error, row by row.

Usage, from the root of a checkout, with llama.cpp's gguf package on GGUF_PY:
  python3 experiments/model-output-qwen25-2026-11-17/rows.py --model qwen2.5-1.5b-instruct-fp16.gguf > rows.out
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT / "lattice"))
if os.environ.get("GGUF_PY"):
    sys.path.insert(0, os.environ["GGUF_PY"])

import torch  # noqa: E402

torch.set_num_threads(1)
import ggufedit  # noqa: E402
import lattices as L  # noqa: E402
import tq  # noqa: E402

LAYERS = (0, 13, 27)
KINDS = ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down")
Q = 16


def db(x):
    return 10.0 * math.log10(max(x, 1e-30))


def rules(W):
    """name -> rebuilt matrix, for the three rules."""
    out = {"XR:16": np.asarray(tq.maxscale_round(W, Q, "R")[0], dtype=np.float64),
           "VE8R:16": np.asarray(tq.maxscale_voronoi(W, Q, "R", name="E8")[0], dtype=np.float64)}
    code = tq.Code("E8", Q)
    kept = code.centre
    code.centre = torch.zeros_like(kept)
    try:
        out["VE8ZR:16"] = np.asarray(tq.maxscale_voronoi(W, Q, "R", name="E8")[0], dtype=np.float64)
    finally:
        code.centre = kept
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    rd = ggufedit.open_model(args.model)

    def matrix(name):
        t = next(t for t in rd.tensors if t.name == name)
        return np.ascontiguousarray(ggufedit.tensor_float(t).reshape(tuple(int(d) for d in reversed(t.shape))), dtype=np.float32)

    print("1. The centre that the harness's E8 Voronoi code subtracts from its codewords, in steps of the lattice")
    for q in (16, 32):
        c = np.asarray(L.codebook_centroid(L.E8(), q), dtype=np.float64)
        print(f"   {q} levels: {np.round(c, 4).tolist()}, length {np.linalg.norm(c):.4f}")
    print("   E8's nearest points to zero are 1.4142 long, so its cell at zero holds every block shorter than 0.7071.")
    print("   With the centre taken off, no codeword is zero: the codeword nearest to a block of zeros is 0.82 steps away.")

    print()
    print("2. Error of single rows, one scale per row, 16 levels. Relative error of a row is its squared error over its squared length, in dB.")
    print("matrix | rows by row length | largest entry over the row's rms: median, largest | rule | matrix error | row error: median | 99th percentile | worst | "
          "rows with error above -3 dB | share of the matrix's error in its 10 worst rows | share of rebuilt weights that are zero")
    worst_of = {}
    total = {}
    for layer in LAYERS:
        for kind in KINDS:
            name = f"blk.{layer}.{kind}.weight"
            W = matrix(name)
            W64 = W.astype(np.float64)
            rn = (W64 ** 2).sum(axis=1)
            peak = np.abs(W64).max(axis=1) / np.sqrt(rn / W.shape[1])
            for rule, Wh in rules(W).items():
                E = Wh - W64
                re = (E ** 2).sum(axis=1)
                rel = re / rn
                q50, q99 = np.quantile(rel, [0.5, 0.99])
                i = int(rel.argmax())
                worst_of[(name, rule)] = (i, float(rel[i]), float(peak[i]))
                t = total.setdefault(rule, [0.0, 0.0, 0, 0])
                t[0] += float(re.sum())
                t[1] += float(rn.sum())
                t[2] += int((rel > 0.5).sum())
                t[3] += W.shape[0]
                print(f"{name} | {W.shape[0]} by {W.shape[1]} | {np.median(peak):.1f}, {peak.max():.1f} | {rule} | {db(re.sum() / rn.sum()):.2f} | {db(q50):.1f} | {db(q99):.1f} | "
                      f"{db(rel.max()):.1f} | {int((rel > 0.5).sum())} | {np.sort(re)[-10:].sum() / re.sum():.3f} | {float((Wh == 0).mean()):.3f}")
    print()
    print("3. Over the 21 matrices")
    print("rule | error pooled over all weights, dB | rows with error above -3 dB | of rows")
    for rule, (e, n, bad, rows) in total.items():
        print(f"{rule} | {db(e / n):.2f} | {bad} | {rows}")

    print()
    print("4. The worst row of the E8 Voronoi code in each matrix where it has a row above -3 dB, under all three rules")
    print("matrix | row | its largest entry over its rms | XR:16, dB | VE8R:16, dB | VE8ZR:16, dB")
    for layer in LAYERS:
        for kind in KINDS:
            name = f"blk.{layer}.{kind}.weight"
            i, rel, peak = worst_of[(name, "VE8R:16")]
            if rel <= 0.5:
                continue
            W = matrix(name)[i:i + 1]
            w = W.astype(np.float64)
            errs = {rule: db(((Wh - w) ** 2).sum() / (w ** 2).sum()) for rule, Wh in rules(W).items()}
            print(f"{name} | {i} | {peak:.1f} | {errs['XR:16']:.2f} | {errs['VE8R:16']:.2f} | {errs['VE8ZR:16']:.2f}")


if __name__ == "__main__":
    main()
