#!/usr/bin/env python3
"""Not planned: where the clipped entries of a layer's ffn_down sit, by column, and how active those columns' inputs are.

Written after the 29 variants were scored, to describe the matrices the scores point at. Nothing is scored here.
A column of ffn_down is one hidden unit of the feed-forward block: its entries are that unit's weights into the
stream. The activity of a unit is the mean squared size of its output over the calibration text, from llama.cpp's
importance matrix of the record imatrix-baseline-2026-11-17 (wikitext-2 training text).

Usage: GGUF_PY=... python3 experiments/clip-by-layer-qwen25-2026-11-18/columns.py --model qwen2.5-1.5b-instruct-fp16.gguf --imatrix qwen15.imatrix.gguf > columns.out
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("clip_run", HERE / "run.py")
R = importlib.util.module_from_spec(spec)
sys.modules["clip_run"] = R
spec.loader.exec_module(R)
Q = R.Q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--imatrix", required=True)
    args = ap.parse_args()
    Q._model_path, Q._imatrix_path = args.model, args.imatrix
    tensors = {t.name: t for t in Q.reader().tensors}
    print("The clip of ffn_down at 5 standard deviations of a row, by column. Not planned, nothing scored.")
    print("layer | entries clipped | columns with a clipped entry, of 8960 | columns that hold half of the clipped entries | share of the clip's squared error in its 10 worst columns, percent | "
          "in its 100 worst | share of the layer's input activity in those 10 columns, percent | activity of the 10 over the mean column")
    for layer in range(28):
        name = f"blk.{layer}.ffn_down.weight"
        W = Q.matrix_of(tensors[name]).astype(np.float64)
        lim = 5.0 * W.std(axis=1, keepdims=True)
        beyond = np.abs(W) > lim
        err = (W - np.clip(W, -lim, lim)) ** 2
        per_col = beyond.sum(axis=0)
        err_col = err.sum(axis=0)
        order = np.argsort(-per_col)
        half = int(np.searchsorted(np.cumsum(per_col[order]), 0.5 * per_col.sum()) + 1)
        worst = np.argsort(-err_col)
        act = Q.column_weights(name, W.shape[1], 1.0)            # each input's mean squared size over the mean
        print(f"{layer} | {int(per_col.sum())} | {int((per_col > 0).sum())} | {half} | {100 * err_col[worst[:10]].sum() / err_col.sum():.1f} | "
              f"{100 * err_col[worst[:100]].sum() / err_col.sum():.1f} | {100 * act[worst[:10]].sum() / act.sum():.2f} | {act[worst[:10]].mean():.2f}")


if __name__ == "__main__":
    main()
