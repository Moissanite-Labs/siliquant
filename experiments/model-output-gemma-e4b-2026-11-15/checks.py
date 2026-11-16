#!/usr/bin/env python3
"""Small checks behind statements of the write-up that no scored variant records. Nothing is scored here.

Written after the second read of the record, which found these statements had no committed evidence.

Usage, with NumPy and torch and llama.cpp's gguf package:
  python3 checks.py REF.gguf [Q4_0.gguf Q4_K_M.gguf Q3_K_M.gguf] > checks.out
The optional files are llama.cpp's own quantised copies of the reference (see README.md for the command).
"""
from __future__ import annotations

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
import tq  # noqa: E402


def db(x):
    return 10.0 * math.log10(x)


def kurt(W):
    W = W.astype(np.float64)
    return float(((W - W.mean()) ** 4).mean() / W.var() ** 2)


def main():
    reader = ggufedit.open_model(sys.argv[1])
    by = {t.name: t for t in reader.tensors}
    live = run.live_kv_layers(reader)
    names = [f"blk.{il}.{k}.weight" for il in spectral.LAYERS for k in run.KINDS]
    mats = {n: ggufedit.tensor_float(by[n]) for n in names}
    total = sum(W.size for W in mats.values())

    print(f"1. Where the searched step of R+S clips at q = 32, on the {len(names)} matrices of layers {spectral.LAYERS}")
    lims, clipped = [], 0.0
    for n, W in mats.items():
        info = tq.round_matrix(W, "R+S", 32)[1]
        lims.append(16.0 * info["scale"])          # mid-rise, 32 levels: entries beyond 16 steps are clipped
        clipped += info["overload"] * W.size
    print(f"   clip limit in standard deviations of the row: least {min(lims):.2f}, median {np.median(lims):.2f}, most {max(lims):.2f}")
    print(f"   share of entries clipped: {100.0 * clipped / total:.4f} %")
    print("   The clip probes use one limit, 3.45, on every matrix. So they touch nearly the entries R+S clips, not exactly those.")
    print()

    print("2. What XR at q = 32 clips when its step is multiplied by 0.7")
    clipped, med, p99 = 0.0, [], []
    for n, W in mats.items():
        clipped += tq.maxscale_round(W, 32, "R", shrink=0.7)[1]["overload"] * W.size
        z = np.abs(W).max(axis=1) / W.std(axis=1)
        med.append(float(np.median(z)))
        p99.append(float(np.quantile(z, 0.99)))
    print(f"   share of entries clipped: {100.0 * clipped / total:.4f} %")
    print(f"   largest entry of a row, in standard deviations of the row: median over rows {min(med):.1f} to {max(med):.1f} by matrix, "
          f"99th percentile {min(p99):.1f} to {max(p99):.1f}")
    print(f"   so the limit, 0.7 of the row's largest entry, is {0.7 * min(med):.1f} to {0.7 * max(med):.1f} standard deviations for the median row "
          "and higher for rows with a large entry. No entry is cut by more than 30 percent.")
    print()

    print("3. Blocks shrunk by the lattice arms with one scale per group")
    for arm, q, group in (("VE832", 16, "32"), ("VE8R", 32, "R"), ("VE8R", 16, "R")):
        shrunk = sum(tq.maxscale_voronoi(W, q, group, "E8")[1]["overload"] * W.size for W in mats.values())
        print(f"   {arm} at q = {q}: share of blocks shrunk {100.0 * shrunk / total:.6f} %")
    print()

    print("4. Balancing through the symmetries (arm G): kurtosis of the matrices before and after")
    print("   layer | matrix | before | after")
    for il in (0, 12, 30, 41):
        out = run.gauge_layer(by, il, live, None)
        for k in ("ffn_up", "ffn_gate", "ffn_down", "attn_q"):
            name = f"blk.{il}.{k}.weight"
            print(f"   {il} | {k} | {kurt(ggufedit.tensor_float(by[name])):.1f} | {kurt(out[name]):.1f}")
    print("   A normal distribution has kurtosis 3. Whether the heavier tails cause the damage of arm G was not tested.")
    print()

    print("5. The arm built to minimise the second path-weighted error (P+S), layer 12, q = 32")
    print("   matrix | S: squared error, second path-weighted error | R+S | P+S     (all in dB)")
    pw = run.path2_weights(reader, by, 12)
    for k in run.KINDS:
        name = f"blk.12.{k}.weight"
        W = ggufedit.tensor_float(by[name])
        rw, cw = pw[k]
        cells = []
        for arm in ("S", "R+S", "P+S"):
            Wh, info = tq.round_matrix(W, arm, 32, row_w=rw, col_w=cw) if arm == "P+S" else tq.round_matrix(W, arm, 32)
            cells.append(f"{db(info['rel_err']):.2f}, {db(run.path2_err(reader, name, W, Wh, by)):.2f}")
        print(f"   {k} | " + " | ".join(cells))
    print("   P+S is worse than S on its own measure on all seven matrices, so it was not scored on the model.")
    print()

    if len(sys.argv) >= 3:
        print("6. This repo's code for the Q4_0 rule against llama.cpp's Q4_0 file")
        other = {t.name: t for t in ggufedit.open_model(sys.argv[2]).tensors}
        for name in ("blk.12.ffn_up.weight", "blk.12.attn_output.weight", "blk.30.ffn_down.weight"):
            W = ggufedit.tensor_float(by[name])
            mine = tq.maxscale_round(W, 16, "32")[0].astype(np.float64)
            theirs = ggufedit.tensor_float(other[name]).astype(np.float64)
            diff = np.abs(mine - theirs)
            print(f"   {name}: equal on {100.0 * np.mean(diff == 0):.2f} % of entries, largest difference {diff.max() / np.abs(theirs).max():.1e} of the largest weight")
        print("   The differences are of the size of one step of a 16-bit float: llama.cpp stores the scale as a 16-bit float and")
        print("   multiplies, this code multiplies and then stores the product as a 16-bit float.")
        print()
        print("7. Which tensors llama.cpp's quantisers changed")
        for path in sys.argv[2:]:
            o = {t.name: t for t in ggufedit.open_model(path).tensors}
            changed = [t.name for t in reader.tensors if o[t.name].tensor_type != t.tensor_type or not np.array_equal(np.asarray(o[t.name].data), np.asarray(t.data))]
            outside = [n for n in changed if not run.BLOCK.match(n)]
            kinds = {}
            for t in o.values():
                if run.BLOCK.match(t.name):
                    kinds[t.tensor_type.name] = kinds.get(t.tensor_type.name, 0) + 1
            print(f"   {Path(path).name}: {len(changed)} tensors differ from the reference, {len(outside)} of them outside the 294 block matrices {outside[:4]}. "
                  f"Block matrices by storage type: {dict(sorted(kinds.items()))}")


if __name__ == "__main__":
    main()
