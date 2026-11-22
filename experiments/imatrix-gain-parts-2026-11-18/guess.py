#!/usr/bin/env python3
"""Not planned: can the column weights of the reading matrices be guessed from the weights alone? Nothing scored.

Three guesses of how loud each dimension of the stream is where a matrix reads it:
  gain          the square of the gain that the block's norm gives the dimension
  gain x stream that, times a guess of the stream's energy in the dimension: the mean square of the token embeddings
                plus the squared size of the rows of every attn_output and ffn_down before this point
  small weights one over the squared size of the matrix's own column: a dimension the matrix reads with small weights
Each guess is put in the place of the importance matrix in the weighted arm, hZBG128:b560s16, at the same power 0.25.
The arm is judged as forecast-*.out judges it: by its squared error with each column counted by its input's mean
square over the wikitext-2 training text, against the plain arm.

Written after the first rows of the plan were in. It is a forecast and not a score.

Usage: GGUF_PY=... python3 experiments/imatrix-gain-parts-2026-11-18/guess.py --model M.gguf --imatrix M.imatrix.gguf > guess.out
"""
import argparse
import importlib.util
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("gain_run", HERE / "run.py")
G = importlib.util.module_from_spec(spec)
sys.modules["gain_run"] = G
spec.loader.exec_module(G)
Q = G.Q
READERS = ("attn_q", "attn_k", "attn_v", "ffn_gate", "ffn_up")
GUESSES = ("the importance matrix", "gain", "gain x stream", "small weights")
_stream = {}


def tensor(name):
    return next(t for t in Q.reader().tensors if t.name == name)


def vector(name):
    return np.asarray(Q.ggufedit.tensor_float(tensor(name)), dtype=np.float64).ravel()


def one(name):
    layer, kind = int(Q.BLOCK.match(name).group(1)), Q.BLOCK.match(name).group(2)
    block = "attn" if kind.startswith("attn") else "ffn"
    W = Q.matrix_of(tensor(name))
    W64 = W.astype(np.float64)
    act = Q.column_weights(name, W.shape[1], 1.0)
    gain = vector(f"blk.{layer}.{block}_norm.weight") ** 2
    guess = {"the importance matrix": act, "gain": gain, "gain x stream": gain * _stream[(layer, block)], "small weights": 1.0 / (W64 ** 2).sum(axis=0)}
    weight = (W64 ** 2).sum(axis=0)
    counted = lambda Wh: float((((W64 - Wh.astype(np.float64)) ** 2).sum(axis=0) * act).sum() / (weight * act).sum())
    out = {"plain": counted(Q.group_ball_round(W, "Z8", 128, 560, 16)[0])}
    for k in GUESSES:
        s = np.maximum((guess[k] / guess[k].mean()) ** 0.25, 0.01)
        out[k] = counted(Q.group_ball_round(W, "Z8", 128, 560, 16, colscale=s)[0])
        out["first ten of " + k] = float(act[np.argsort(-guess[k], kind="stable")[:10]].sum() / act.sum())
    return name, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--imatrix", required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    Q._model_path = args.model
    G._paths["real"] = args.imatrix
    G.use_matrix("real")
    if Q.MODELS.get(Q.sha256_of(args.model)) not in G.ACCEPTED:
        raise SystemExit("FAIL: the model file is not one the plan names")
    names = [t.name for t in Q.targets(Q.reader())]
    layers = sorted({int(Q.BLOCK.match(n).group(1)) for n in names})
    E = Q.matrix_of(tensor("token_embd.weight")).astype(np.float64)
    seen = (E ** 2).mean(axis=0)
    for l in layers:
        _stream[(l, "attn")] = seen.copy()
        seen = seen + (Q.matrix_of(tensor(f"blk.{l}.attn_output.weight")).astype(np.float64) ** 2).sum(axis=1)
        _stream[(l, "ffn")] = seen.copy()
        seen = seen + (Q.matrix_of(tensor(f"blk.{l}.ffn_down.weight")).astype(np.float64) ** 2).sum(axis=1)
    with Pool(args.workers) as pool:
        rows = dict(pool.imap_unordered(one, [n for n in names if Q.BLOCK.match(n).group(2) in READERS], chunksize=1))
    db = lambda x: 10.0 * math.log10(x)
    print("Column weights of the reading matrices guessed from the weights alone, in the place of the importance matrix. Not planned, nothing scored.")
    print("gain: what the weighted arm gains over the plain arm, in dB, with each column counted by its input's mean square over the wikitext-2 training text. Mean over the layers, and the worst layer.")
    print("first ten: the share of the input's mean square in the ten columns the guess puts first, in percent, mean over the layers.")
    print("kind | " + " | ".join(f"{k}: gain | worst layer | first ten" for k in GUESSES))
    for kind in READERS:
        rs = [rows[f"blk.{l}.{kind}.weight"] for l in layers]
        cells = []
        for k in GUESSES:
            g = [db(r["plain"]) - db(r[k]) for r in rs]
            cells.append(f"{np.mean(g):.2f} | {min(g):.2f} | {100 * np.mean([r['first ten of ' + k] for r in rs]):.1f}")
        print(f"{kind} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
