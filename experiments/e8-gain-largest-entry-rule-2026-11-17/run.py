#!/usr/bin/env python3
"""Why E8 gains more than its cell is worth under the largest-entry rule.

The plan is PREREG.md in this folder. This file is the only program of the run.

The second record found that E8 with one scale per group, set so that nothing
is clipped, beats scalar rounding under the same rule by 1.5 to 2.5 dB of
weight error on gemma, when the shape of the E8 cell is worth 0.65 dB. This
program runs the same two rules, with the harness functions the second record
used, on synthetic matrices and on the matrices of the first record, and
splits the gain into the part the cell accounts for and the part the scale
accounts for. It also runs the point set F_8 of PR #6 under the same rule.

Usage, from the root of a checkout, with NumPy and torch:
  python3 experiments/e8-gain-largest-entry-rule-2026-11-17/run.py --smoke
  python3 experiments/e8-gain-largest-entry-rule-2026-11-17/run.py --weights /path/to/weights.npz > results.out
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "harness"))
import tq  # noqa: E402

_spec = importlib.util.spec_from_file_location("codebooks", ROOT / "experiments" / "incentre-vortex-codebooks-2026-11-17" / "run.py")
cb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cb)

WEIGHTS_SHA256 = "b6c8ee500656e04e8e57c2949a5253f0dda002ba36cd1561846574dbcf01132e"
REAL_PATTERN = re.compile(r"^(encoder|decoder)\.blocks\.(0|5)\..*weight$")
REAL_COUNT = 32
G_CUBE = 1.0 / 12.0
G_E8 = 929.0 / 12960.0
R_BALL8 = math.gamma(5.0) ** (1.0 / 8.0) / math.sqrt(math.pi)     # radius of the ball of volume 1 in 8 dimensions
CELLS = ((8, 16), (32, 16), (128, 16), ("R", 16), (32, 32), ("R", 32))
F8_CELLS = (("R", 12, 16), (32, 12, 16), ("R", 23, 32))           # (group, K for F_8, q of the references it is set against)
BISECT_STEPS = 14


def db(x):
    return 10.0 * math.log10(x)


def synthetic(kind, seed):
    rng = np.random.default_rng({"G": 6000, "T": 7000}[kind] + seed)
    W = rng.normal(size=(512, 2048)) if kind == "G" else rng.standard_t(4, size=(512, 2048))
    return W.astype(np.float32)


def lattice_scales(W, q, group, name):
    """The scale of every block and of every group under the rule of tq.maxscale_voronoi.

    The bisection is copied from that function, which does not return its scales. run() checks
    that rounding at these scales gives the matrix that function returns.
    """
    code = tq.Code(name, q)
    m, n = W.shape
    size = n if group == "R" else int(group)
    X = torch.from_numpy(np.ascontiguousarray(W)).to(tq.DEV).reshape(-1, 8)
    norm = torch.clamp(torch.linalg.vector_norm(X, dim=1), min=1e-20)
    rho, cover = {"E8": (0.5 ** 0.5, 1.0), "Z8": (0.5, 2.0 ** 0.5)}[name]
    lo = norm / (cover * q) * 0.9
    hi = norm / (rho * q) * 1.1
    for _ in range(BISECT_STEPS):
        mid = torch.sqrt(lo * hi)
        ok = code.roundtrip(X / mid[:, None] + code.centre)[1]
        hi = torch.where(ok, mid, hi)
        lo = torch.where(ok, lo, mid)
    per_block = hi.reshape(-1, size // 8)
    tg, which = per_block.max(dim=1)
    t = tg[:, None].expand(-1, size // 8).reshape(-1)
    R, _ = code.quantize(X / t[:, None])
    Wh = (R * t[:, None]).reshape(m, n).to(torch.float16).cpu().numpy()
    return per_block.double().cpu().numpy(), tg.double().cpu().numpy(), which.cpu().numpy(), Wh


def f8_largest_entry(W, K, group, t):
    """F_8 with one scale per group and nothing clipped: half a cell is the group's largest entry over K.

    The codebook and its exact nearest-codeword rule are those of the first study of this date.
    Stored as 16-bit floats, as the harness stores the other arms.
    """
    m, n = W.shape
    size = n if group == "R" else int(group)
    X = W.astype(np.float64).reshape(-1, size)
    h = np.abs(X).max(axis=1, keepdims=True) / K
    h = np.where(h == 0, 1.0, h)
    rec = cb.f_round((X / h).reshape(-1, 8), K, t).reshape(-1, size) * h
    Wh = rec.reshape(m, n).astype(np.float16)
    W64 = W.astype(np.float64)
    rel = float(np.sum((W64 - Wh.astype(np.float64)) ** 2) / np.sum(W64 ** 2))
    return rel, math.log2(K) + 0.5 + 16.0 / size


def run(name, W):
    """Every cell of the plan on one matrix. Returns rows."""
    W = np.ascontiguousarray(W, dtype=np.float32)
    m, n = W.shape
    W64 = W.astype(np.float64)
    total = float(np.sum(W64 ** 2))
    rows = []
    for group, q in CELLS:
        size = n if group == "R" else int(group)
        Xg = np.abs(W64).reshape(-1, size)
        d = Xg.max(axis=1) / (q / 2.0)                       # the scalar step of each group
        _, ix = tq.maxscale_round(W, q, group)
        row = {"matrix": name, "group": str(group), "size": size, "q": q, "bits": ix["bits"], "X": ix["rel_err"],
               "X_model": float(np.sum(d ** 2) * size * G_CUBE / total)}
        blocks = W64.reshape(-1, size // 8, 8)
        bnorm = np.sqrt((blocks ** 2).sum(axis=2))
        ball = bnorm.max(axis=1) / (R_BALL8 * q)             # the scale a ball of the same volume would need
        row["scale_term_ball"] = db(float(np.sum(d ** 2)) / float(np.sum(ball ** 2)))
        for lat, G in (("Z8", G_CUBE), ("E8", G_E8)):
            Wh_ref, iv = tq.maxscale_voronoi(W, q, group, name=lat)
            per_block, tg, which, Wh = lattice_scales(W, q, group, lat)
            row[f"V{lat}"] = iv["rel_err"]
            row[f"V{lat}_same_as_harness"] = bool(np.array_equal(Wh, Wh_ref))
            row[f"V{lat}_shrunk"] = iv["overload"]
            row[f"V{lat}_model"] = float(np.sum(tg ** 2) * size * G / total)
            row[f"V{lat}_scale_term"] = db(float(np.sum(d ** 2)) / float(np.sum(tg ** 2)))
            if lat == "E8":
                g = np.arange(len(tg))
                b = blocks[g, which]                           # the block that sets its group's scale
                bn = bnorm[g, which]
                row["E8_reach"] = float(np.mean(bn / (q * tg)))            # 0.707 is the nearest wall of the cell, 1 its farthest corner
                row["E8_peak"] = float(np.mean(np.abs(b).max(axis=1) / bn))
                row["E8_same_block"] = float(np.mean(which == np.argmax(np.abs(blocks).max(axis=2), axis=1)))
                row["all_peak"] = float(np.mean(np.abs(blocks).max(axis=2) / np.maximum(bnorm, 1e-30)))
        rows.append(row)
    for group, K, q in F8_CELLS:
        size = n if group == "R" else int(group)
        for t_name, t in (("incentre", cb.C_IN), ("centroid", 8.0 / 9.0)):
            rel, bits = f8_largest_entry(W, K, group, t)
            rows.append({"matrix": name, "group": str(group), "size": size, "q": q, "arm": "F8", "K": K, "t": t_name, "bits": bits, "F8": rel})
    return rows


def summarise(rows, out):
    main = [r for r in rows if "arm" not in r]
    f8 = [r for r in rows if r.get("arm") == "F8"]
    fams = [f for f in ("G", "T", "real") if any(r["matrix"].split("/")[0] == f for r in main)]
    shape = db(G_CUBE / G_E8)
    out.append("0. Checks")
    bad = [(r["matrix"], r["group"], r["q"], lat) for r in main for lat in ("Z8", "E8") if not r[f"V{lat}_same_as_harness"]]
    out.append(f"   cells where rounding at the scales found here does not give the matrix the harness returns: {len(bad)} of {2 * len(main)}" + (f" {bad[:6]}" if bad else ""))
    shr = [(r["matrix"], r["group"], r["q"], lat) for r in main for lat in ("Z8", "E8") if r[f"V{lat}_shrunk"] > 0]
    out.append(f"   cells where a lattice arm shrank a block: {len(shr)}" + (f" {shr[:6]}" if shr else ""))
    out.append("")
    out.append("1. Gain in weight error over the scalar largest-entry rule (X) at equal bits, in dB, and its two parts")
    out.append(f"   shape = what the E8 cell is worth against the cube, {shape:.3f} dB, from bounds/bounds.out")
    out.append("   scale = 10 log10 of (sum of squared scalar steps over sum of squared E8 scales), from the scales the two rules chose")
    out.append("   family | group | q | bits | X dB | Z8 over X | E8 over X: mean (min, max) | shape | scale | shape + scale | left over: mean (min, max) | scale if the region were a ball | matrices")
    for f in fams:
        for group, q in CELLS:
            rs = [r for r in main if r["matrix"].split("/")[0] == f and r["group"] == str(group) and r["q"] == q]
            if not rs:
                continue
            gain = [db(r["X"] / r["VE8"]) for r in rs]
            gz = [db(r["X"] / r["VZ8"]) for r in rs]
            sc = [r["VE8_scale_term"] for r in rs]
            left = [g - shape - s for g, s in zip(gain, sc)]
            sizes = sorted({r["size"] for r in rs})
            gl = str(group) if group != "R" else "row (" + "/".join(str(s) for s in sizes) + ")"
            out.append(f"   {f:>4} | {gl:>14} | {q} | {np.mean([r['bits'] for r in rs]):.3f} | {np.mean([db(r['X']) for r in rs]):.2f} | {np.mean(gz):+.2f} | "
                       f"{np.mean(gain):+.2f} ({min(gain):+.2f}, {max(gain):+.2f}) | {shape:.2f} | {np.mean(sc):+.2f} | {shape + np.mean(sc):+.2f} | "
                       f"{np.mean(left):+.2f} ({min(left):+.2f}, {max(left):+.2f}) | {np.mean([r['scale_term_ball'] for r in rs]):+.2f} | {len(rs)}")
    out.append("")
    out.append("2. How well the fine-cell formula gives each arm's error: measured error over formula, in dB (0 means the formula is exact)")
    out.append("   family | group | q | X | Z8 | E8")
    for f in fams:
        for group, q in CELLS:
            rs = [r for r in main if r["matrix"].split("/")[0] == f and r["group"] == str(group) and r["q"] == q]
            if rs:
                out.append(f"   {f:>4} | {str(group):>3} | {q} | {np.mean([db(r['X'] / r['X_model']) for r in rs]):+.2f} | "
                           f"{np.mean([db(r['VZ8'] / r['VZ8_model']) for r in rs]):+.2f} | {np.mean([db(r['VE8'] / r['VE8_model']) for r in rs]):+.2f}")
    out.append("")
    out.append("3. The block that sets its group's E8 scale")
    out.append("   reach = its length over q times the scale: 0.707 at the nearest wall of the E8 cell, 1 at its farthest corner, 0.839 for a ball of the same volume")
    out.append("   peak = its largest entry over its length: 1 for a single weight, 0.354 for eight equal weights")
    out.append("   family | group | q | reach | peak of that block | peak of all blocks | share of groups where it also holds the largest entry")
    for f in fams:
        for group, q in CELLS:
            rs = [r for r in main if r["matrix"].split("/")[0] == f and r["group"] == str(group) and r["q"] == q]
            if rs:
                out.append(f"   {f:>4} | {str(group):>3} | {q} | {np.mean([r['E8_reach'] for r in rs]):.3f} | {np.mean([r['E8_peak'] for r in rs]):.3f} | "
                           f"{np.mean([r['all_peak'] for r in rs]):.3f} | {np.mean([r['E8_same_block'] for r in rs]):.2f}")
    out.append("")
    out.append("4. F_8 of PR #6 under the same rule, against X and E8 at the nearest level count with no more bits")
    out.append("   A positive number means F_8 has the larger error, in dB.")
    out.append("   family | group | K | t | F8 bits | X and E8 bits | F8 dB | F8 minus X: mean (min, max) | worse than X | F8 minus E8: mean (min, max) | worse than E8 | matrices")
    key = {(r["matrix"], r["group"], r["q"]): r for r in main}
    for f in fams:
        for group, K, q in F8_CELLS:
            for t_name in ("incentre", "centroid"):
                rs = [r for r in f8 if r["matrix"].split("/")[0] == f and r["group"] == str(group) and r["K"] == K and r["t"] == t_name]
                if not rs:
                    continue
                gx = [db(r["F8"] / key[(r["matrix"], r["group"], q)]["X"]) for r in rs]
                ge = [db(r["F8"] / key[(r["matrix"], r["group"], q)]["VE8"]) for r in rs]
                out.append(f"   {f:>4} | {str(group):>3} | {K} | {t_name} | {np.mean([r['bits'] for r in rs]):.3f} | {np.mean([key[(r['matrix'], r['group'], q)]['bits'] for r in rs]):.3f} | "
                           f"{np.mean([db(r['F8']) for r in rs]):.2f} | {np.mean(gx):+.2f} ({min(gx):+.2f}, {max(gx):+.2f}) | {sum(g > 0 for g in gx)} | "
                           f"{np.mean(ge):+.2f} ({min(ge):+.2f}, {max(ge):+.2f}) | {sum(g > 0 for g in ge)} | {len(rs)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="run the code path on a matrix that is not in the study and print nothing but ok")
    ap.add_argument("--weights", help="path to weights.npz of mlx-community/whisper-base.en-mlx")
    args = ap.parse_args()
    t0 = time.time()
    if args.smoke:
        W = np.random.default_rng(7).uniform(-1, 1, size=(32, 256)).astype(np.float32)
        rows = run("smoke", W)
        out = []
        summarise([dict(r, matrix="G/smoke") for r in rows], out)
        ok = all(r["VE8_same_as_harness"] and r["VZ8_same_as_harness"] and 0 < r["VE8"] < 1 for r in rows if "arm" not in r) and all(0 < r["F8"] < 1 for r in rows if "arm" in r)
        print("ok" if ok else "FAIL", f"({len(rows)} rows, {len(out)} lines, {time.time() - t0:.0f} s)")
        sys.exit(0 if ok else 1)
    rows = []
    for kind in ("G", "T"):
        for seed in range(3):
            rows += run(f"{kind}/{seed}", synthetic(kind, seed))
    if args.weights:
        raw = Path(args.weights).read_bytes()
        if hashlib.sha256(raw).hexdigest() != WEIGHTS_SHA256:
            raise SystemExit("FAIL: the weights file does not have the SHA-256 named in the plan")
        z = np.load(args.weights)
        keys = sorted(k for k in z.files if REAL_PATTERN.match(k) and z[k].ndim == 2)
        if len(keys) != REAL_COUNT:
            raise SystemExit(f"FAIL: {len(keys)} matrices match the plan's pattern, expected {REAL_COUNT}")
        for k in keys:
            rows += run("real/" + k, z[k])
    (HERE / "results.json").write_text(json.dumps(rows, indent=1) + "\n")
    out = []
    summarise(rows, out)
    out.append("")
    out.append(f"rows: {len(rows)}; matrices: {len({r['matrix'] for r in rows})}; real weights used: {'yes' if args.weights else 'no'}; torch device: {tq.DEV}; seconds: {time.time() - t0:.0f}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
