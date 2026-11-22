#!/usr/bin/env python3
"""Not planned: is a very active hidden unit marked by the size of its own input weights? Nothing scored.

A hidden unit of the feed-forward block reads the stream through one row of ffn_gate and one row of ffn_up.
For every layer this prints the units whose two rows are far larger than the layer's usual: the product of the
two row norms, over the median product of the layer. Beside each: its share of the input activity of ffn_down,
by llama.cpp's importance matrix, and whether it is one of the ten loud columns.

Written after the plan was committed, while its variants were being scored. It is a description on two models.

Usage: GGUF_PY=... python3 experiments/loud-columns-2026-11-18/inputs.py --model M.gguf --imatrix M.imatrix.gguf > inputs.out
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
    args = ap.parse_args()
    Q._model_path, Q._imatrix_path = args.model, args.imatrix
    if Q.MODELS.get(Q.sha256_of(args.model)) not in L.ACCEPTED:
        raise SystemExit("FAIL: the model file is not one the plan names")
    names = {t.name for t in Q.reader().tensors}
    layers = sorted(int(Q.BLOCK.match(n).group(1)) for n in names if Q.BLOCK.match(n) and Q.BLOCK.match(n).group(2) == "ffn_down")
    print("Hidden units by the size of their input weights: the norm of the unit's row of ffn_gate times the norm of its row of ffn_up, over the layer's median. Not planned, nothing scored.")
    print("layer | units over 5 times the median | over 10 times | the three largest: unit, times the median, share of the input activity of ffn_down in percent, is it a loud column | "
          "the most active unit: unit, its share in percent, its place by this figure")
    for layer in layers:
        D = Q.matrix_of(L.tensor(f"blk.{layer}.ffn_down.weight")).astype(np.float64)
        G = Q.matrix_of(L.tensor(f"blk.{layer}.ffn_gate.weight")).astype(np.float64)
        U = Q.matrix_of(L.tensor(f"blk.{layer}.ffn_up.weight")).astype(np.float64)
        act = Q.column_weights(f"blk.{layer}.ffn_down.weight", D.shape[1], 1.0)
        p = np.linalg.norm(G, axis=1) * np.linalg.norm(U, axis=1)
        p = p / np.median(p)
        loud = set(L.loud(D).tolist())
        top = np.argsort(-p, kind="stable")[:3]
        j = int(np.argmax(act))
        print(f"{layer} | {int((p > 5).sum())} | {int((p > 10).sum())} | "
              + "; ".join(f"{int(u)}, {p[u]:.1f}, {100 * act[u] / act.sum():.2f}, {'yes' if int(u) in loud else 'no'}" for u in top)
              + f" | {j}, {100 * act[j] / act.sum():.2f}, {int((p > p[j]).sum()) + 1}")


if __name__ == "__main__":
    main()
