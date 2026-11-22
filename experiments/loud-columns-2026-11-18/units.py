#!/usr/bin/env python3
"""Not planned: the ten loud columns of an ffn_down one by one, and the matrix's most active columns. Nothing scored.

Written after the plan was committed and before any variant was scored, to see what the ten columns are.
It is what extension 1 of PREREG.md was written from.

Usage: GGUF_PY=... python3 experiments/loud-columns-2026-11-18/units.py --model M.gguf --imatrix M.imatrix.gguf --layers 1,7 > units.out
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("loud_run", HERE / "run.py")
L = importlib.util.module_from_spec(spec)
sys.modules["loud_run"] = L
spec.loader.exec_module(L)
Q = L.Q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--imatrix", required=True)
    ap.add_argument("--layers", required=True)
    args = ap.parse_args()
    Q._model_path, Q._imatrix_path = args.model, args.imatrix
    if Q.MODELS.get(Q.sha256_of(args.model)) not in L.ACCEPTED:
        raise SystemExit("FAIL: the model file is not one the plan names")
    print("The ten loud columns of an ffn_down, one by one. Not planned, nothing scored.")
    print("An entry is measured in standard deviations of its row. A row of ffn_down is one dimension of the stream.")
    for layer in (int(x) for x in args.layers.split(",")):
        name = f"blk.{layer}.ffn_down.weight"
        W = Q.matrix_of(L.tensor(name)).astype(np.float64)
        act = Q.column_weights(name, W.shape[1], 1.0)
        Z = np.abs(W) / W.std(axis=1, keepdims=True)
        mean_norm = np.sqrt((W ** 2).sum(axis=0).mean())
        top = np.argsort(-act, kind="stable")[:5]
        print()
        print(f"layer {layer}: the five most active columns and their share of the matrix's input activity, percent: "
              + ", ".join(f"column {int(c)} {100 * act[c] / act.sum():.2f}" for c in top))
        print("loud column | activity over the mean column | share of the matrix's input activity, percent | its size over the mean column's | entries beyond 5 | "
              "its three largest entries: dimension of the stream and size")
        for c in L.loud(W):
            big = np.argsort(-Z[:, c], kind="stable")[:3]
            print(f"{int(c)} | {act[c]:.1f} | {100 * act[c] / act.sum():.2f} | {np.linalg.norm(W[:, c]) / mean_norm:.2f} | {int((Z[:, c] > 5).sum())} | "
                  + ", ".join(f"{int(r)} at {Z[r, c]:.1f}" for r in big))


if __name__ == "__main__":
    main()
