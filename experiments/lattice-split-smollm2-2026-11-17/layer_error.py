#!/usr/bin/env python3
"""What an arm's error does to the output of each matrix, fed the model's own inputs. Nothing is scored.

Written after every row of results.jsonl was read, to ask where the gain of integer levels over
noise comes from. The reference model is run once on two passages and the input of every block
matrix is kept. For a matrix W, an arm's rounded matrix R and the inputs X (one row per position):

  weight error   |R - W|^2 / |W|^2
  output error   |X (R - W)^T|^2 / |X W^T|^2

If the inputs were white the two would be equal. "lift" is the second over the first, in dB: how
much more (or less) an arm's error shows in the matrix's output than its size says. Two arms with
the same weight error and different lifts differ in how their error meets the inputs.

Usage, from the root of a checkout, with the work folder of run.py (its base arms built):
  python3 layer_error.py --model M.gguf --text wiki.test.raw --work DIR > layer_error.out
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("split_run", HERE / "run.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
NF = R.NF

ARMS = ("ZBG128:b560s16", "EBG128:b560s16", "noise(ZBG128:b560s16)#1", "shuffle(ZBG128:b560s16)#1",
        "noisepart(ZBG128:b560s16)@small#1", "noisepart(ZBG128:b560s16)@big#1")
EVERY = 3                      # one block in every three
PASSAGES = 2


def db(x):
    return 10.0 * math.log10(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--work", required=True)
    args = ap.parse_args()
    model = NF.Model(args.model)
    ids = R.load_text(model, args.text)
    P = NF.passages(ids, R.LENGTH, PASSAGES)
    names = [n for n in model.names() if int(n.split(".")[1]) % EVERY == 0]
    keep = {}

    def tap(name, x):
        if name in names:
            keep[name] = x

    model.forward(P, tap=tap)
    arms = {a: R.Changed(model, args.work, a) for a in ARMS}
    rows = []
    for name in names:
        X = keep[name]
        W = model.rd.matrix(name)
        ref = float(((X @ W.T).astype(np.float64) ** 2).sum())
        w2 = float((W.astype(np.float64) ** 2).sum())
        # the inputs' own unevenness: how much of their squared size the largest coordinates hold
        a = (X.astype(np.float64) ** 2).mean(axis=0)
        top = float(np.sort(a)[::-1][:max(1, len(a) // 100)].sum() / a.sum())
        got = {}
        for arm, ch in arms.items():
            E = np.asarray(ch[name], dtype=np.float32) - W
            got[arm] = (float((E.astype(np.float64) ** 2).sum()) / w2, float(((X @ E.T).astype(np.float64) ** 2).sum()) / ref)
        rows.append((name, top, got))
        print(f"# {name} done", file=sys.stderr, flush=True)
    kinds = NF.KINDS
    print(f"Blocks: one in every {EVERY}. Matrices: {len(rows)}. Positions: {P.size}. lift = output error over weight error, dB.")
    print("The inputs: the part of their squared size in the largest hundredth of the coordinates, by kind: "
          + ", ".join(f"{k} {np.mean([t for n, t, _ in rows if n.split('.')[2] == k]):.3f}" for k in kinds))
    print()
    print("arm | weight error, dB | output error, dB | lift, dB: all | " + " | ".join(kinds))
    for arm in ARMS:
        w = np.mean([db(g[arm][0]) for _, _, g in rows])
        o = np.mean([db(g[arm][1]) for _, _, g in rows])
        by = [np.mean([db(g[arm][1]) - db(g[arm][0]) for n, _, g in rows if n.split(".")[2] == k]) for k in kinds]
        print(f"{arm} | {w:.3f} | {o:.3f} | {o - w:+.2f} | " + " | ".join(f"{v:+.2f}" for v in by))
    print()
    base = ARMS[0]
    print(f"Output error of an arm over that of {base}, dB, mean over the matrices: all | " + " | ".join(kinds) + " | feed-forward matrices | attention matrices")
    for arm in ARMS[1:]:
        d = lambda sel: np.mean([db(g[arm][1]) - db(g[base][1]) for n, _, g in rows if sel(n.split(".")[2])])     # noqa: E731
        print(f"{arm} | {d(lambda k: True):+.2f} | " + " | ".join(f"{d(lambda k, kk=kk: k == kk):+.2f}" for kk in kinds)
              + f" | {d(lambda k: k.startswith('ffn')):+.2f} | {d(lambda k: k.startswith('attn')):+.2f}")


if __name__ == "__main__":
    main()
