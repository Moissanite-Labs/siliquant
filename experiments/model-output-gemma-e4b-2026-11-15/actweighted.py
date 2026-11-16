#!/usr/bin/env python3
"""Activation-weighted error of the variants on a fixed subset of matrices (round 6 of PREREG.md).

`llama-imatrix` writes, for each block matrix, the sum of squares of every input it multiplies
and the number of positions seen. For each variant and each of the 42 block matrices of layers
0, 8, 16, 24, 32 and 40 the matrix is rounded as the variant does, and the squared change is
summed with column j counted by the mean square of input j, over the same sum for the matrix.
Matrices the model never reads have no entry in the file and are left out.

Usage, with NumPy and torch and llama.cpp's gguf package:
  llama-imatrix -m REF.gguf -f eval.txt -c 512 -b 512 --chunk 60 --chunks 40 -ngl 999 -o imatrix.gguf
  python3 actweighted.py --model REF.gguf --imatrix imatrix.gguf --out actweighted.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "harness"))
sys.path.insert(0, str(HERE))
import ggufedit  # noqa: E402
import run  # noqa: E402
import spectral  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--imatrix", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    by = {t.name: t for t in ggufedit.open_model(args.model).tensors}
    im = {t.name: t for t in ggufedit.open_model(args.imatrix).tensors}
    names = [f"blk.{il}.{k}.weight" for il in spectral.LAYERS for k in run.KINDS]
    mats, act = {}, {}
    for name in names:
        if name + ".in_sum2" not in im:
            continue
        mats[name] = ggufedit.tensor_float(by[name])
        act[name] = ggufedit.tensor_float(im[name + ".in_sum2"]).astype(np.float64) / float(ggufedit.tensor_float(im[name + ".counts"])[0])
        if len(act[name]) != mats[name].shape[1]:
            raise SystemExit(f"{name}: {len(act[name])} inputs in the file, {mats[name].shape[1]} columns in the matrix")
    for v in spectral.VARIANTS + spectral.EXTRA:
        weighted, plain = [], []
        for index, name in enumerate(names):
            if name not in mats:
                continue
            W = mats[name]
            Wh = spectral.round_one(W, v, index)
            E2 = (W.astype(np.float64) - Wh.astype(np.float64)) ** 2
            W2 = W.astype(np.float64) ** 2
            a = act[name]
            weighted.append(10.0 * math.log10(max(float((E2.sum(axis=0) * a).sum()), 1e-300) / float((W2.sum(axis=0) * a).sum())))
            plain.append(10.0 * math.log10(max(float(E2.sum()), 1e-300) / float(W2.sum())))
        row = {"variant": v, "matrices": len(weighted), "act_err_db_mean": sum(weighted) / len(weighted),
               "squared_err_db_mean": sum(plain) / len(plain)}
        with open(args.out, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
