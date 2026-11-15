#!/usr/bin/env python3
"""Lattice rounding against scalar rounding at equal bits per weight.

The plan is PREREG.md in this folder. Arms, data, the scale search and the
summary are fixed there. This file is the only program of the run.

Usage, from the root of a checkout, with NumPy:
  python3 experiments/lattice-vs-scalar-rounding-2026-11-15/run.py --smoke
  python3 experiments/lattice-vs-scalar-rounding-2026-11-15/run.py --weights /path/to/weights.npz --workers 8
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "lattice"))
import lattices as L  # noqa: E402

WEIGHTS_SHA256 = "b6c8ee500656e04e8e57c2949a5253f0dda002ba36cd1561846574dbcf01132e"
REAL_PATTERN = re.compile(r"^(encoder|decoder)\.blocks\.(0|5)\..*weight$")
REAL_COUNT = 32
Q_SYNTH = (3, 4, 8, 16)
Q_REAL = (4, 8)
SHRINK_STEPS = 12
OPT_MATRICES = ("G/0", "RC/0", "real/encoder.blocks.0.attn.query.weight", "real/decoder.blocks.5.attn.out.weight")

_cache = {}


def lattice(name):
    if name not in _cache:
        _cache[name] = {"Z8": L.Zn(8), "A8*": L.Astar(8), "E8": L.E8()}[name]
    return _cache[name]


def centroid(name, q):
    key = ("c", name, q)
    if key not in _cache:
        _cache[key] = L.codebook_centroid(lattice(name), q)
    return _cache[key]


def codebook(name, q):
    key = ("cb", name, q)
    if key not in _cache:
        _cache[key] = L.enumerate_codebook(lattice(name), q) - centroid(name, q)
    return _cache[key]


def synthetic(kind, seed):
    """The three synthetic families of the plan, 512 x 512."""
    rng = np.random.default_rng({"G": 1000, "T": 2000, "RC": 3000}[kind] + seed)
    if kind == "G":
        return rng.normal(size=(512, 512))
    if kind == "T":
        return rng.standard_t(4, size=(512, 512))
    r = np.exp(0.5 * rng.normal(size=512))
    c = np.exp(0.5 * rng.normal(size=512))
    return r[:, None] * rng.normal(size=(512, 512)) * c[None, :]


def scalar_round(x, q, step):
    """q evenly spaced levels with clipping at the ends. Mid-rise for even q, mid-tread for odd q."""
    if q % 2 == 0:
        k = np.clip(np.floor(x / step), -q // 2, q // 2 - 1)
        return (k + 0.5) * step
    return np.clip(np.rint(x / step), -(q // 2), q // 2) * step


def search_scale(err, t0):
    """Scale that minimises err(t). Coarse grid in half octaves, widened at the edges, then golden section on log t."""
    ks = list(range(-6, 7))
    vals = {k: err(t0 * 2.0 ** (k / 2.0)) for k in ks}
    for _ in range(3):
        best = min(vals, key=vals.get)
        if best == min(vals):
            new = range(min(vals) - 4, min(vals))
        elif best == max(vals):
            new = range(max(vals) + 1, max(vals) + 5)
        else:
            break
        for k in new:
            vals[k] = err(t0 * 2.0 ** (k / 2.0))
    best = min(vals, key=vals.get)
    at_edge = best in (min(vals), max(vals))
    lo, hi = math.log(t0) + (best - 1) * math.log(2) / 2, math.log(t0) + (best + 1) * math.log(2) / 2
    g = (math.sqrt(5) - 1) / 2
    a, b = hi - g * (hi - lo), lo + g * (hi - lo)
    fa, fb = err(math.exp(a)), err(math.exp(b))
    t_best, e_best = t0 * 2.0 ** (best / 2.0), vals[best]
    for _ in range(10):
        if fa < e_best:
            t_best, e_best = math.exp(a), fa
        if fb < e_best:
            t_best, e_best = math.exp(b), fb
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - g * (hi - lo)
            fa = err(math.exp(a))
        else:
            lo, a, fa = a, b, fb
            b = lo + g * (hi - lo)
            fb = err(math.exp(b))
    return t_best, e_best, at_edge


def run_one(task):
    """All arms for one matrix, one q and one setting of dual scaling."""
    name, W, q, dual, do_opt = task
    W = np.asarray(W, dtype=np.float64)
    m, n = W.shape
    total = float(np.sum(W ** 2))
    if dual:
        s, t, Wn, _, imb = L.dual_scale(W, iters=16)
        back = s[:, None] * t[None, :]
        side = 16.0 * (m + n) / (m * n)
    else:
        Wn, back, side, imb = W, None, 0.0, None
    X = Wn.reshape(-1, 8)
    sigma = float(Wn.std())

    def rel_err(Xh):
        Wh = Xh.reshape(m, n)
        if back is not None:
            Wh = Wh * back
        return float(np.sum((W - Wh) ** 2)) / total

    rows = []
    prefix = "D+" if dual else ""
    t0 = 4.0 * sigma / q
    ts, es, edge = search_scale(lambda step: rel_err(scalar_round(X, q, step)), t0)
    rows.append({"matrix": name, "q": q, "arm": prefix + "S", "rel_err": es, "scale": ts, "at_edge": edge,
                 "overload": float(np.mean(np.abs(X / ts) > q / 2.0)), "bits": math.log2(q) + side})
    found = {}
    for lat in ("Z8", "A8*", "E8"):
        Lt, c = lattice(lat), centroid(lat, q)
        over = {}

        def err(tt, Lt=Lt, c=c, over=over):
            rec, _, ov = L.voronoi_quantize(X / tt, Lt, q, centre=c, steps=SHRINK_STEPS)
            over[tt] = float(np.mean(ov))
            return rel_err(rec * tt)

        tl, el, edge = search_scale(err, t0 / Lt.volume ** (1.0 / 8.0))
        found[lat] = tl
        rows.append({"matrix": name, "q": q, "arm": prefix + lat, "rel_err": el, "scale": tl, "at_edge": edge,
                     "overload": over[tl], "bits": math.log2(q) + side})
    if do_opt and q == 4:
        for lat in ("A8*", "E8"):
            cb = codebook(lat, q)
            best = None
            for k in range(-4, 5):
                tt = found[lat] * 2.0 ** (k / 8.0)
                e = rel_err(L.nearest_in_codebook(X / tt, cb)[1] * tt)
                if best is None or e < best[0]:
                    best = (e, tt, abs(k) == 4)
            rows.append({"matrix": name, "q": q, "arm": prefix + lat + "-opt", "rel_err": best[0], "scale": best[1],
                         "at_edge": best[2], "overload": None, "bits": math.log2(q) + side})
    if imb is not None:
        for r in rows:
            r["imbalance_after"] = imb
    return rows


def tasks_for(args):
    out = []
    if args.smoke:
        W = np.random.default_rng(7).uniform(-1, 1, size=(64, 64))
        return [("smoke", W, q, dual, False) for q in (3, 4) for dual in (False, True)]
    for kind in ("G", "T", "RC"):
        for seed in range(3):
            name = f"{kind}/{seed}"
            for q in Q_SYNTH:
                for dual in (False, True):
                    out.append((name, synthetic(kind, seed), q, dual, name in OPT_MATRICES))
    if args.weights:
        raw = Path(args.weights).read_bytes()
        if hashlib.sha256(raw).hexdigest() != WEIGHTS_SHA256:
            raise SystemExit("FAIL: the weights file does not have the SHA-256 named in the plan")
        z = np.load(args.weights)
        keys = sorted(k for k in z.files if REAL_PATTERN.match(k) and z[k].ndim == 2)
        if len(keys) != REAL_COUNT:
            raise SystemExit(f"FAIL: {len(keys)} matrices match the plan's pattern, expected {REAL_COUNT}")
        for k in keys:
            name = "real/" + k
            for q in Q_REAL:
                for dual in (False, True):
                    out.append((name, z[k].astype(np.float64), q, dual, name in OPT_MATRICES))
    return out


def db(x):
    return 10.0 * math.log10(x)


def summarise(rows, out):
    by = {}
    for r in rows:
        by[(r["matrix"], r["q"], r["arm"])] = r
    families = sorted({m.split("/")[0] for m, _, _ in by})
    qs = sorted({q for _, q, _ in by})
    arms = ["S", "Z8", "A8*", "E8", "A8*-opt", "E8-opt"]
    out.append("Floors on the Gaussian family: Shannon 2^(-2R) for any code; the best fixed-rate scalar quantiser (Lloyd-Max) for arm S")
    for q in qs:
        lm = L.lloyd_max_gaussian(q)[1]
        out.append(f"  q = {q:>2}: R = {math.log2(q):.3f} bits, Shannon floor {db(q ** -2.0):.2f} dB, scalar floor {db(lm):.2f} dB")
    out.append("")
    out.append("Relative squared error in dB, mean over the matrices of a family (lower is better)")
    out.append("family | q | " + " | ".join(f"{a:>8}" for a in arms) + " | " + " | ".join(f"{'D+' + a:>10}" for a in arms))
    for fam in families:
        mats = sorted({m for m, _, _ in by if m.split("/")[0] == fam})
        for q in qs:
            cells = []
            for a in arms + ["D+" + a for a in arms]:
                v = [db(by[(m, q, a)]["rel_err"]) for m in mats if (m, q, a) in by]
                cells.append(f"{np.mean(v):.2f} ({len(v)})" if v else "-")
            if any(c != "-" for c in cells):
                out.append(f"{fam:>6} | {q:>2} | " + " | ".join(f"{c:>8}" for c in cells[:6]) + " | " + " | ".join(f"{c:>10}" for c in cells[6:]))
    out.append("")
    out.append("Gain over scalar rounding in dB (positive means the arm is better): arm against S, and D+arm against D+S")
    out.append("family | q | pair | mean | min | max | matrices with a gain | matrices")
    for fam in families:
        mats = sorted({m for m, _, _ in by if m.split("/")[0] == fam})
        for q in qs:
            for a, base in [(x, "S") for x in arms[1:]] + [("D+" + x, "D+S") for x in arms[1:]] + [("D+S", "S")]:
                g = [db(by[(m, q, base)]["rel_err"] / by[(m, q, a)]["rel_err"]) for m in mats if (m, q, a) in by and (m, q, base) in by]
                if g:
                    out.append(f"{fam:>6} | {q:>2} | {a + ' over ' + base:>20} | {np.mean(g):+.3f} | {min(g):+.3f} | {max(g):+.3f} | {sum(x > 0 for x in g)} | {len(g)}")
    out.append("")
    edges = [(r["matrix"], r["q"], r["arm"]) for r in rows if r["at_edge"]]
    out.append(f"scale searches that ended at an edge of their range: {len(edges)}" + (f" {edges[:8]}" if edges else ""))
    low = [(r["matrix"], r["q"], r["arm"], db(r["rel_err"])) for r in rows
           if r["matrix"].startswith("G/") and db(r["rel_err"]) < db(r["q"] ** -2.0) - 0.05]
    out.append(f"rows on the Gaussian family under the Shannon floor by more than 0.05 dB: {len(low)}" + (f" {low}" if low else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="run the code path on a matrix that is not in the study and print nothing but ok")
    ap.add_argument("--weights", help="path to weights.npz of mlx-community/whisper-base.en-mlx")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    t0 = time.time()
    tasks = tasks_for(args)
    with Pool(args.workers) as pool:
        rows = [r for part in pool.imap(run_one, tasks, chunksize=1) for r in part]
    if args.smoke:
        ok = all(0.0 < r["rel_err"] < 1.0 for r in rows) and len(rows) == 4 * 4
        print("ok" if ok else "FAIL", f"({time.time() - t0:.0f} s)")
        sys.exit(0 if ok else 1)
    (HERE / "results.json").write_text(json.dumps(rows, indent=1) + "\n")
    out = []
    summarise(rows, out)
    out.append(f"rows: {len(rows)}; matrices: {len({r['matrix'] for r in rows})}; real weights used: {'yes' if args.weights else 'no'}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
