#!/usr/bin/env python3
"""Part L of PREREG-extension-2.md: the large entries of ffn_down, layer by layer. Weights only, nothing scored.

Axes. A tensor is read as NumPy gives it (ggufedit.tensor_float): shape (rows, columns). For ffn_down
that is (2560, 10240): a row is one dimension of the stream (the output side of the matrix), a column
is one hidden unit (the input side, of size d_ff). The first record's clip takes "rows" the same way:
the limit of an entry is 5 standard deviations of the 10240 entries of its row (divisor n), so a
stream dimension has a limit and a hidden unit does not. The output states this and the shape.

For each of the 42 layers: the number of entries beyond 5 standard deviations of their row, the
largest entry in standard deviations of its row, and the squared size of the clip's change over the
squared size of the matrix. For layer 10 also the change split by hidden unit and by dimension of
the stream: the top 20 of each with their share, and how many hold half and nine tenths of it.

Writes large_by_layer.out (text) and large_by_layer.json (the ranked unit and dimension indices
that Part M reads). Refuses to run unless PREREG-extension-2.md and this program are committed and pushed.

Usage: python3 large_by_layer.py --model REF.gguf
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "harness"))
spec = importlib.util.spec_from_file_location("ob_run", HERE / "run.py")
ob_run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ob_run)
import ggufedit  # noqa: E402

LIMIT = 5.0
LAYER = 10


def clip_change(W, limit=LIMIT):
    """W minus its clip at `limit` standard deviations of each row (float64), and the row standard deviations."""
    W = W.astype(np.float64)
    sd = W.std(axis=1, keepdims=True)
    lim = limit * sd
    return W - np.clip(W, -lim, lim), sd


def ranked(share, top=20):
    order = np.argsort(-share, kind="stable")
    cum = np.cumsum(share[order])
    return order, {"half": int(np.searchsorted(cum, 0.5 * cum[-1]) + 1), "nine_tenths": int(np.searchsorted(cum, 0.9 * cum[-1]) + 1),
                   "top": [{"index": int(i), "share": float(share[i] / share.sum())} for i in order[:top]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    ob_run.plan_gate()
    ext = ob_run.committed_and_pushed(ROOT / ob_run.EXTENSION2)
    reader = ggufedit.open_model(args.model)
    by = {t.name: t for t in reader.tensors}
    lines = [f"Part L, from the plan commit {ext['commit']}",
             "Axes: the tensor is read as NumPy gives it, shape (rows, columns). For ffn_down that is (stream dimensions, hidden units).",
             "A row is one dimension of the stream (the output side); the clip limit is 5 standard deviations of the entries of a row,",
             "taken across the hidden units, as in the first record. A column is one hidden unit (the input side, size d_ff).", ""]
    layers = []
    out = {}
    for il in range(42):
        t = by[f"blk.{il}.ffn_down.weight"]
        W = ggufedit.tensor_float(t)
        E, sd = clip_change(W)
        n = int(np.sum(E != 0))
        row = {"layer": il, "shape": list(W.shape), "n_beyond": n, "share_of_entries": n / W.size,
               "max_sd": float(np.max(np.abs(W) / sd)), "change_over_size": float(np.sum(E * E) / np.sum(W.astype(np.float64) ** 2))}
        layers.append(row)
        if il == LAYER:
            sq = E * E
            units, dims = sq.sum(axis=0), sq.sum(axis=1)
            uo, us = ranked(units)
            do, ds = ranked(dims)
            out = {"layer": il, "shape": list(W.shape), "units_ranked": [int(i) for i in uo[units[uo] > 0]],
                   "dims_ranked": [int(i) for i in do[dims[do] > 0]], "units": us, "dims": ds,
                   "units_with_change": int(np.sum(units > 0)), "dims_with_change": int(np.sum(dims > 0)),
                   "axis_note": "rows = dimensions of the stream (2560); columns = hidden units (10240); the clip limit is per row"}
    lines.append("layer | shape | entries beyond 5 sd | share of entries | largest entry, in sd of its row | clip's squared change over squared size of the matrix")
    for r in layers:
        lines.append(f"{r['layer']} | {r['shape'][0]}x{r['shape'][1]} | {r['n_beyond']} | {r['share_of_entries']:.2e} | {r['max_sd']:.2f} | {r['change_over_size']:.3e}")
    lines += ["", f"Layer {LAYER}: the clip's squared change by hidden unit (columns) and by dimension of the stream (rows)"]
    for key, name in (("units", "hidden units"), ("dims", "dimensions of the stream")):
        r = out[key]
        lines.append(f"   {name}: {out[key + '_with_change']} have any change; {r['half']} hold half of it and {r['nine_tenths']} hold nine tenths")
        lines.append("   top 20 (index, share): " + ", ".join(f"{x['index']} {100 * x['share']:.2f} %" for x in r["top"]))
    Path(HERE / "large_by_layer.out").write_text("\n".join(lines) + "\n")
    Path(HERE / "large_by_layer.json").write_text(json.dumps({"plan_commit": ext["commit"], "layers": layers, "layer10": out}, indent=None) + "\n")
    print("\n".join(lines[-8:]))


if __name__ == "__main__":
    main()
