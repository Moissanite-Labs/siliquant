#!/usr/bin/env python3
"""Where the large entries of the feed-forward output matrices sit. Nothing is scored here.

An entry is large when it lies beyond 5 standard deviations of its row. For each layer the
program counts them and says how concentrated they are in columns (hidden units) and in rows
(dimensions of the stream).

Usage, with NumPy and llama.cpp's gguf package:
  python3 large_entries.py REF.gguf > large_entries.out
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "harness"))
import ggufedit  # noqa: E402

LIMIT = 5.0


def main():
    reader = ggufedit.open_model(sys.argv[1])
    by = {t.name: t for t in reader.tensors}
    layers = sum(1 for n in by if n.endswith(".ffn_down.weight"))
    total, size, per_layer, top_cols = 0, 0, [], []
    print(f"ffn_down entries beyond {LIMIT:g} standard deviations of their row")
    print("layer | entries | share of the matrix, % | share of them in the 10 busiest columns, % | in the 10 busiest rows, % | largest entry, in standard deviations of its row")
    for il in range(layers):
        W = ggufedit.tensor_float(by[f"blk.{il}.ffn_down.weight"])
        Z = np.abs(W) / W.std(axis=1, keepdims=True)
        M = Z > LIMIT
        n = int(M.sum())
        cols = np.sort(M.sum(axis=0))[::-1]
        rows = np.sort(M.sum(axis=1))[::-1]
        total += n
        size += W.size
        per_layer.append((n, il))
        top_cols.append(100.0 * cols[:10].sum() / max(n, 1))
        print(f"{il} | {n} | {100.0 * n / W.size:.4f} | {top_cols[-1]:.1f} | {100.0 * rows[:10].sum() / max(n, 1):.1f} | {Z.max():.1f}")
    print(f"total: {total} of {size} entries, one in {size / total:.0f}")
    five = sorted(per_layer, reverse=True)[:5]
    print(f"the five layers with the most ({', '.join(str(il) for _, il in five)}) hold {100.0 * sum(n for n, _ in five) / total:.1f} % of them")
    print(f"fewest in a layer: {min(per_layer)[0]}. Largest share held by the 10 busiest columns of a layer: {max(top_cols):.1f} %")


if __name__ == "__main__":
    main()
