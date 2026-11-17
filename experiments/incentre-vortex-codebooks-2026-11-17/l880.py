#!/usr/bin/env python3
"""Extension 1 of the plan: the 880 realisations of the graph L880 as a codebook for blocks of 14 weights.

PR #6 proposes this as "Laman-VQ". The plan is the section "Extension 1" of PREREG.md.
The realisations are read from the certificate file of Foundation-0/rigid-realizations,
which is not committed here. The program refuses a file with another SHA-256.

Usage, from the root of a checkout, with NumPy:
  python3 experiments/incentre-vortex-codebooks-2026-11-17/l880.py --certificate /path/to/l880_certificate.json --smoke
  python3 experiments/incentre-vortex-codebooks-2026-11-17/l880.py --certificate /path/to/l880_certificate.json --weights /path/to/weights.npz > l880.out
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("codebooks", HERE / "run.py")
main_run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(main_run)
L, first, db = main_run.L, main_run.first, main_run.db

CERT_SHA256 = "583fec3591b377a883afe17104aa25c0835115e47eda2a3c861b8e17721cf92d"
N_CODE = 880
DIM = 14
BITS = math.log2(N_CODE) / DIM


def read_certificate(path):
    """The 880 codewords, and how far the 440 stored points are from meeting the 17 bar lengths.

    The file holds 440 box centres: the coordinates of the 7 free vertices, as integers over
    2^48. The other 440 realisations are their mirror images, which the file does not store.
    A mirror image is taken here as (x, -y) for every free vertex: the reflection in the line
    through the two pinned vertices that lie on the x axis.
    """
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != CERT_SHA256:
        raise SystemExit("FAIL: the certificate file does not have the SHA-256 named in the plan")
    d = json.loads(raw)
    scale = 2.0 ** d["center_scale_exp2"]
    half = np.array(d["centers_num"], dtype=np.float64) / scale              # (440, 14)
    free = d["free_vertices"]
    pinned = {int(k): (float(Fraction(v[0])), float(Fraction(v[1]))) for k, v in d["pinned_vertices"].items()}
    lengths = [float(Fraction(s)) for s in d["edge_lengths"]]
    worst = 0.0
    for row in half:
        pos = dict(pinned)
        for i, v in enumerate(free):
            pos[v] = (row[2 * i], row[2 * i + 1])
        for (a, b), want in zip(d["edges"], lengths):
            got = math.hypot(pos[a][0] - pos[b][0], pos[a][1] - pos[b][1])
            worst = max(worst, abs(got - want))
    mirror = half.copy()
    mirror[:, 1::2] *= -1.0
    info = {"stored": len(half), "free_vertices": len(free), "bars": len(lengths), "worst_bar_error": worst,
            "shortest_bar": min(lengths), "longest_bar": max(lengths),
            "claimed_real": d["certified_real_realizations_SE2"], "claimed_complex": d["complex_realizations_SE2"]}
    return np.concatenate([half, mirror]), info


def whiten(cb):
    """The codebook moved to mean zero and mapped linearly so that its covariance is the identity."""
    c = cb - cb.mean(axis=0)
    lam, V = np.linalg.eigh(c.T @ c / len(c))
    return c @ (V / np.sqrt(lam)) @ V.T, lam


def lloyd(kind, rounds=40):
    """880 points in 14 dimensions fitted by Lloyd's algorithm to a sample of the family that no study matrix uses."""
    rng = np.random.default_rng({"G": 7100, "T": 8100}[kind])
    x = rng.normal(size=(2 ** 17, DIM)) if kind == "G" else rng.standard_t(4, size=(2 ** 17, DIM))
    cb = x[rng.choice(len(x), size=N_CODE, replace=False)].copy()
    for _ in range(rounds):
        idx = L.nearest_in_codebook(x, cb, chunk=2048)[0]
        sums = np.zeros_like(cb)
        np.add.at(sums, idx, x)
        counts = np.bincount(idx, minlength=N_CODE)
        cb = np.where(counts[:, None] > 0, sums / np.maximum(counts, 1)[:, None], cb)
    return cb


def score(W, cb, t0):
    total = float(np.sum(W ** 2))
    X = W.reshape(-1, DIM)
    return first.search_scale(lambda h: float(np.sum((X - L.nearest_in_codebook(X / h, cb, chunk=2048)[1] * h) ** 2)) / total, t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--certificate", required=True, help="path to l880_certificate.json of Foundation-0/rigid-realizations")
    ap.add_argument("--weights", help="path to weights.npz of mlx-community/whisper-base.en-mlx")
    ap.add_argument("--smoke", action="store_true", help="run the code path on a matrix that is not in the study and print nothing but ok")
    args = ap.parse_args()
    t_start = time.time()
    cb, info = read_certificate(args.certificate)
    centred = cb - cb.mean(axis=0)
    centred = centred / math.sqrt(float((centred ** 2).mean()))
    white, lam = whiten(cb)
    random_cb = np.random.default_rng(880).normal(size=(N_CODE, DIM))
    books = {"L880": centred, "L880w": white, "random": random_cb}
    names = ["smoke"] if args.smoke else [f"{k}/{s}" for k in ("G", "T") for s in range(3)]
    if args.weights and not args.smoke:
        main_run._weights_path = args.weights
        names += main_run.real_names()
    fitted = {} if args.smoke else {k: lloyd(k) for k in ("G", "T")}
    rows = []
    for name in names:
        W = main_run.matrix(name)
        sigma = float(W.std())
        total = float(np.sum(W ** 2))
        kind = name.split("/")[0]
        for arm, book in list(books.items()) + ([("fitted", fitted[kind])] if kind in fitted else []):
            tt, e, edge = score(W, book, sigma)
            rows.append({"matrix": name, "arm": arm, "bits": BITS, "rel_err": e, "scale": tt, "at_edge": edge})
        X = W.reshape(-1, 8)
        e2 = first.search_scale(lambda step: float(np.sum((X - first.scalar_round(X, 2, step)) ** 2)) / total, 2.0 * sigma)[1]
        rows.append({"matrix": name, "arm": "S-mix", "bits": BITS, "rel_err": (1.0 - BITS) * 1.0 + BITS * e2, "one_bit": e2})
    if args.smoke:
        ok = info["worst_bar_error"] < 1e-6 and all(0.0 < r["rel_err"] < 1.5 for r in rows) and cb.shape == (N_CODE, DIM)
        print("ok" if ok else "FAIL", f"({len(rows)} rows, {time.time() - t_start:.0f} s)")
        sys.exit(0 if ok else 1)
    (HERE / "l880.json").write_text(json.dumps({"certificate": info, "covariance_eigenvalues": lam.tolist(), "rows": rows}, indent=1) + "\n")
    by = {(r["matrix"], r["arm"]): r for r in rows}
    fams = [f for f in ("G", "T", "real") if any(m.split("/")[0] == f for m in names)]
    out = ["1. The file"]
    out.append(f"   {info['stored']} stored points, {info['free_vertices']} free vertices, {info['bars']} bars of length {info['shortest_bar']:.4f} to {info['longest_bar']:.4f}.")
    out.append(f"   The file states {info['claimed_real']} real realisations of {info['claimed_complex']} complex ones. That claim is not checked here.")
    out.append(f"   Largest difference between a bar of a stored point and its stated length: {info['worst_bar_error']:.1e}")
    out.append(f"   Codebook: {N_CODE} points in {DIM} dimensions, {BITS:.4f} bits per weight. Shannon's floor for Gaussian weights at those bits: {db(4.0 ** -BITS):.2f} dB.")
    out.append(f"   Variances of the centred codebook along its principal axes, largest over smallest: {lam.max() / lam.min():.1f}. Smallest over the mean: {lam.min() / lam.mean():.4f}.")
    out.append("")
    out.append("2. Relative squared error in dB at equal bits, mean over the matrices of a family (min, max)")
    out.append("   L880 = the realisations, moved to mean zero, one free scale. L880w = the same, mapped linearly to unit covariance.")
    out.append("   random = 880 independent Gaussian points. fitted = 880 points fitted to a separate sample of the family.")
    out.append("   S-mix = a share of the weights rounded to 2 levels and the rest set to zero, so that the mean bits are equal.")
    out.append("   family | L880 | L880w | random | fitted | S-mix | L880 minus random: mean (min, max) | L880w minus random: mean (min, max) | L880w minus S-mix: mean (min, max) | matrices")
    for f in fams:
        ms = [m for m in names if m.split("/")[0] == f]

        def col(arm):
            v = [db(by[(m, arm)]["rel_err"]) for m in ms if (m, arm) in by]
            return f"{np.mean(v):.2f} ({min(v):.2f}, {max(v):.2f})" if v else "-"

        def gap(a, b):
            v = [db(by[(m, a)]["rel_err"] / by[(m, b)]["rel_err"]) for m in ms]
            return f"{np.mean(v):+.2f} ({min(v):+.2f}, {max(v):+.2f})"

        out.append(f"   {f:>4} | {col('L880')} | {col('L880w')} | {col('random')} | {col('fitted')} | {col('S-mix')} | {gap('L880', 'random')} | {gap('L880w', 'random')} | {gap('L880w', 'S-mix')} | {len(ms)}")
    out.append("")
    low = [(r["matrix"], r["arm"]) for r in rows if r["matrix"].startswith("G/") and db(r["rel_err"]) < db(4.0 ** -r["bits"]) - 0.05]
    out.append(f"3. Rows on the Gaussian family under Shannon's floor by more than 0.05 dB: {len(low)}" + (f" {low}" if low else ""))
    edges = [(r["matrix"], r["arm"]) for r in rows if r.get("at_edge")]
    out.append(f"   scale searches that ended at an edge of their range: {len(edges)}" + (f" {edges[:8]}" if edges else ""))
    out.append(f"   rows: {len(rows)}; matrices: {len(names)}; seconds: {time.time() - t_start:.0f}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
