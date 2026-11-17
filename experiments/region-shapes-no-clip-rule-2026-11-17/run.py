#!/usr/bin/env python3
"""Codebooks cut from a lattice by a ball or a cross-polytope, under the rule that clips nothing.

The plan is PREREG.md in this folder. This file is the only program of the run.

A codebook here is every point y of a lattice (the integers Z^8 or E8) whose size is at most
a radius: its length for a ball, the sum of the sizes of its entries for a cross-polytope.
The radius is the largest at which the codebook has no more than q^8 points, so an arm never
has more bits than the E8 Voronoi code at q levels. The scale of a group is the smallest at
which every block of the group rounds to a point of the codebook, as in the second record.

The scalar rule X and the E8 Voronoi code VE8 are not run here. Their errors are read from
the record e8-gain-largest-entry-rule-2026-11-17, which ran them with the harness on the same
matrices.

Usage, from the root of a checkout, with NumPy:
  python3 experiments/region-shapes-no-clip-rule-2026-11-17/run.py --smoke
  python3 experiments/region-shapes-no-clip-rule-2026-11-17/run.py --weights /path/to/weights.npz > results.out
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "lattice"))
import lattices as L  # noqa: E402

PRIOR = ROOT / "experiments" / "e8-gain-largest-entry-rule-2026-11-17" / "results.json"
WEIGHTS_SHA256 = "b6c8ee500656e04e8e57c2949a5253f0dda002ba36cd1561846574dbcf01132e"
REAL_PATTERN = re.compile(r"^(encoder|decoder)\.blocks\.(0|5)\..*weight$")
REAL_COUNT = 32
CELLS = (("32", 16), ("R", 16), ("R", 32))
ARMS = (("ZB", "Z8", "ball"), ("ZC", "Z8", "cross"), ("EB", "E8", "ball"), ("EC", "E8", "cross"))
BISECT_STEPS = 14
SHRINK_STEPS = 12
_E8 = L.E8()
_count_cache = {}


def db(x):
    return 10.0 * math.log10(x)


# ---------------------------------------------------------------------------
# Exact counts of lattice points in a region
# ---------------------------------------------------------------------------

def _power(per_coordinate, limit):
    """Counts of vectors of 8 coordinates by (total size, parity of the sum).

    per_coordinate is a list of (size, parity, ways) for one coordinate. Returns table[s][p],
    the number of vectors of total size s <= limit whose integer parts add up to parity p.
    """
    table = [[0, 0] for _ in range(limit + 1)]
    table[0][0] = 1
    for _ in range(8):
        new = [[0, 0] for _ in range(limit + 1)]
        for s in range(limit + 1):
            for p in (0, 1):
                c = table[s][p]
                if c:
                    for size, par, ways in per_coordinate:
                        if s + size <= limit:
                            new[s + size][p ^ par] += c * ways
        table = new
    return table


def counts(lattice, shape, limit):
    """N[r] for r = 0..limit: the number of points of the lattice inside the region of radius r, exactly.

    shape "cross": the sizes of the entries add up to at most r.
    shape "ball": the squared length is at most r.
    E8 is the integer vectors with an even sum together with the vectors of half-integers
    whose sum is even.
    """
    key = (lattice, shape, limit)
    if key in _count_cache:
        return _count_cache[key]
    if shape == "cross":
        whole = [(0, 0, 1)] + [(a, a % 2, 2) for a in range(1, limit + 1)]
        # entry k + 1/2 has size a + 1/2 with k = a or k = -a - 1, and the eight halves add up to 4
        half = [(a, a % 2, 1) for a in range(0, limit + 1)] + [(a, (a + 1) % 2, 1) for a in range(0, limit + 1)]
        half_limit = limit - 4

        def half_index(r):
            return r - 4
    else:
        top = math.isqrt(limit)
        whole = [(0, 0, 1)] + [(a * a, a % 2, 2) for a in range(1, top + 1)]
        # in quarter units the squared size of k + 1/2 is (2k + 1)^2, for k = a or k = -a - 1
        half = []
        a = 0
        while (2 * a + 1) ** 2 <= 4 * limit:
            half.append(((2 * a + 1) ** 2, a % 2, 1))
            half.append(((2 * a + 1) ** 2, (a + 1) % 2, 1))
            a += 1
        half_limit = 4 * limit

        def half_index(r):
            return 4 * r
    t = _power(whole, limit)
    out = []
    run_all = run_even = 0
    if lattice == "E8":
        h = _power(half, half_limit) if half_limit >= 0 else []
        run_half = 0
        done = -1
    for r in range(limit + 1):
        run_all += t[r][0] + t[r][1]
        run_even += t[r][0]
        if lattice == "Z8":
            out.append(run_all)
        else:
            upto = min(half_index(r), half_limit)
            while done < upto:
                done += 1
                run_half += h[done][0]
            out.append(run_even + run_half)
    _count_cache[key] = out
    return out


def count(lattice, shape, radius):
    return counts(lattice, shape, max(radius, 8))[radius]


def radius_for(lattice, shape, q):
    """The largest radius at which the codebook has at most q^8 points, and its size."""
    cap = q ** 8
    table = counts(lattice, shape, 2 * q + 8 if shape == "cross" else q * q)
    step = 2 if (lattice == "E8" and shape == "ball") else 1      # squared lengths in E8 are even
    best = max(r for r in range(0, len(table), step) if table[r] <= cap)
    assert best + step < len(table), "the table of counts is too short"
    return best, table[best]


def scan(lattice, shape, radius, box):
    """The same count by listing every point in a box. For the self-test."""
    vals = np.arange(-box, box + 1, dtype=np.float64)
    total = 0
    for offset in ((0.0,) if lattice == "Z8" else (0.0, 0.5)):
        v = vals + offset if offset == 0.0 else vals[:-1] + 0.5
        grids = np.stack(np.meshgrid(*([v] * 4), indexing="ij"), axis=-1).reshape(-1, 4)
        size = np.abs(grids).sum(axis=1) if shape == "cross" else (grids ** 2).sum(axis=1)
        sums = grids.sum(axis=1)
        for a in np.unique(size):
            if a > radius:
                continue
            for par in (0, 1):
                na = int(np.sum((size == a) & (np.round(sums - 4 * offset) % 2 == par)))
                if not na:
                    continue
                rest = size <= radius - a
                if lattice == "E8":
                    rest &= (np.round(sums - 4 * offset) % 2 == par)   # the two halves have sums of equal parity
                total += na * int(np.sum(rest))
    return total


def selftest():
    out = []
    ok = True
    for lattice, shape, radii, box in (("Z8", "cross", (0, 1, 2, 3), 3), ("Z8", "ball", (0, 1, 2, 3, 4, 5), 3),
                                       ("E8", "cross", (0, 2, 4, 5, 6), 6), ("E8", "ball", (0, 2, 4, 6, 8), 3)):
        for r in radii:
            ok &= count(lattice, shape, r) == scan(lattice, shape, r, box)
    out.append((ok, "the counts equal a scan of every point in a box, for 20 small regions"))
    shells = [count("E8", "ball", 2 * m) for m in range(5)]
    out.append((shells == [1, 241, 2401, 9121, 26641], f"E8 has 1, 240, 2160, 6720 and 17520 points of squared length 0, 2, 4, 6 and 8: running totals {shells}"))
    return out


# ---------------------------------------------------------------------------
# Rounding under the rule that clips nothing
# ---------------------------------------------------------------------------

def closest(lattice, x):
    return np.rint(x) if lattice == "Z8" else _E8.closest(x)


def size_of(shape, y):
    return np.abs(y).sum(axis=1) if shape == "cross" else (y ** 2).sum(axis=1)


def block_scales(X, lattice, shape, radius):
    """For every block the smallest scale at which it rounds to a point of the codebook, by bisection.

    Two bounds hold for any block: rounding moves a point by at most 4 in the sum of sizes and
    by at most sqrt 2 in length (the cubic lattice is the worse of the two).
    """
    if shape == "cross":
        g, r, margin = np.abs(X).sum(axis=1), float(radius), 4.0
    else:
        g, r, margin = np.sqrt((X ** 2).sum(axis=1)), math.sqrt(radius), math.sqrt(2.0)
    g = np.maximum(g, 1e-30)
    lo = g / (r + margin) * 0.99
    hi = g / max(r - margin, 0.25) * 1.01
    for _ in range(BISECT_STEPS):
        mid = np.sqrt(lo * hi)
        ok = size_of(shape, closest(lattice, X / mid[:, None])) <= radius + 1e-9
        hi = np.where(ok, mid, hi)
        lo = np.where(ok, lo, mid)
    return hi


def round_groups(W, lattice, shape, radius, group, tb):
    """Round a matrix with one scale per group. Returns (relative error, scales of the groups, share of blocks shrunk)."""
    m, n = W.shape
    size = n if group == "R" else int(group)
    X = W.reshape(-1, 8)
    tg = tb.reshape(-1, size // 8).max(axis=1)
    t = np.repeat(tg, size // 8)
    Y = closest(lattice, X / t[:, None])
    out = size_of(shape, Y) > radius + 1e-9
    if np.any(out):
        xo = X[out] / t[out, None]
        lo, hi = np.zeros(len(xo)), np.ones(len(xo))
        for _ in range(SHRINK_STEPS):
            mid = 0.5 * (lo + hi)
            good = size_of(shape, closest(lattice, xo * mid[:, None])) <= radius + 1e-9
            lo = np.where(good, mid, lo)
            hi = np.where(good, hi, mid)
        Y[out] = closest(lattice, xo * lo[:, None])
    Wh = (Y * t[:, None]).reshape(m, n).astype(np.float16).astype(np.float64)
    return float(np.sum((W - Wh) ** 2) / np.sum(W ** 2)), tg, float(np.mean(out))


def run(name, W, prior):
    W = np.ascontiguousarray(W, dtype=np.float32).astype(np.float64)
    m, n = W.shape
    X = W.reshape(-1, 8)
    rows = []
    for q in sorted({q for _, q in CELLS}):
        for arm, lattice, shape in ARMS:
            radius, codewords = radius_for(lattice, shape, q)
            tb = block_scales(X, lattice, shape, radius)
            for group, qq in CELLS:
                if qq != q:
                    continue
                size = n if group == "R" else int(group)
                rel, tg, shrunk = round_groups(W, lattice, shape, radius, group, tb)
                d = np.abs(W).reshape(-1, size).max(axis=1) / (q / 2.0)
                ref = prior.get((name, group, q))
                rows.append({"matrix": name, "arm": arm, "group": group, "size": size, "q": q, "radius": radius, "codewords": codewords,
                             "bits": math.log2(codewords) / 8.0 + 16.0 / size, "rel_err": rel, "shrunk": shrunk,
                             "scale_term": db(float(np.sum(d ** 2)) / float(np.sum(tg ** 2))),
                             "X": ref["X"] if ref else None, "VE8": ref["VE8"] if ref else None, "ref_bits": ref["bits"] if ref else None})
    return rows


def summarise(rows, checks, out):
    fams = [f for f in ("G", "T", "real") if any(r["matrix"].split("/")[0] == f for r in rows)]
    out.append("0. Checks")
    for ok, text in checks:
        out.append(f"   {'ok  ' if ok else 'FAIL'} {text}")
    shr = [(r["matrix"], r["arm"], r["group"], r["q"], r["shrunk"]) for r in rows if r["shrunk"] > 0]
    out.append(f"   runs where a block lay outside the codebook at its group's scale and was shrunk: {len(shr)} of {len(rows)}" + (f", largest share of blocks {max(s[4] for s in shr):.2e}" if shr else ""))
    out.append("")
    out.append("1. The codebooks")
    out.append("   arm | lattice | region | q | radius | codewords | codewords over q^8 | bits per weight without the scale")
    seen = set()
    for r in rows:
        k = (r["arm"], r["q"])
        if k in seen:
            continue
        seen.add(k)
        lattice, shape = {a: (l, s) for a, l, s in ARMS}[r["arm"]]
        rad = f"sum of sizes <= {r['radius']}" if shape == "cross" else f"squared length <= {r['radius']}"
        out.append(f"   {r['arm']} | {lattice} | {shape} | {r['q']} | {rad} | {r['codewords']} | {r['codewords'] / r['q'] ** 8:.4f} | {math.log2(r['codewords']) / 8.0:.4f}")
    out.append("")
    out.append("2. Error against the two arms of the second record at the same q, in dB. A positive number means the arm of this run has the smaller error.")
    out.append("   No arm of this run has more bits than X or VE8.")
    out.append("   family | group | q | arm | bits | X and VE8 bits | arm dB | over X: mean (min, max) | over VE8: mean (min, max) | ahead of VE8 on | scale term | matrices")
    for f in fams:
        for group, q in CELLS:
            for arm, _, _ in ARMS:
                rs = [r for r in rows if r["matrix"].split("/")[0] == f and r["group"] == group and r["q"] == q and r["arm"] == arm and r["X"]]
                if not rs:
                    continue
                gx = [db(r["X"] / r["rel_err"]) for r in rs]
                ge = [db(r["VE8"] / r["rel_err"]) for r in rs]
                out.append(f"   {f:>4} | {group:>2} | {q} | {arm} | {np.mean([r['bits'] for r in rs]):.3f} | {np.mean([r['ref_bits'] for r in rs]):.3f} | {np.mean([db(r['rel_err']) for r in rs]):.2f} | "
                           f"{np.mean(gx):+.2f} ({min(gx):+.2f}, {max(gx):+.2f}) | {np.mean(ge):+.2f} ({min(ge):+.2f}, {max(ge):+.2f}) | {sum(g > 0 for g in ge)} | {np.mean([r['scale_term'] for r in rs]):+.2f} | {len(rs)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="run the code path on a matrix that is not in the study and print nothing but ok")
    ap.add_argument("--weights", help="path to weights.npz of mlx-community/whisper-base.en-mlx")
    args = ap.parse_args()
    t0 = time.time()
    checks = selftest()
    if args.smoke:
        W = np.random.default_rng(7).uniform(-1, 1, size=(32, 256))
        rows = run("smoke", W, {})
        out = []
        summarise([dict(r, matrix="G/smoke") for r in rows], checks, out)
        ok = all(c[0] for c in checks) and all(0 < r["rel_err"] < 1 and r["codewords"] <= r["q"] ** 8 for r in rows)
        print("ok" if ok else "FAIL", f"({len(rows)} rows, {len(out)} lines, {time.time() - t0:.0f} s)")
        sys.exit(0 if ok else 1)
    if not all(c[0] for c in checks):
        raise SystemExit("FAIL: a check of the code failed: " + "; ".join(t for ok, t in checks if not ok))
    prior = {(r["matrix"], r["group"], r["q"]): r for r in json.loads(PRIOR.read_text()) if "arm" not in r}
    spec_path = ROOT / "experiments" / "e8-gain-largest-entry-rule-2026-11-17" / "run.py"
    src = spec_path.read_text()
    rows = []
    for kind, base in (("G", 6000), ("T", 7000)):
        assert f'"{kind}": {base}' in src, "the seeds of the second study of this date have changed"
        for seed in range(3):
            rng = np.random.default_rng(base + seed)
            W = rng.normal(size=(512, 2048)) if kind == "G" else rng.standard_t(4, size=(512, 2048))
            rows += run(f"{kind}/{seed}", W.astype(np.float32), prior)
    if args.weights:
        raw = Path(args.weights).read_bytes()
        if hashlib.sha256(raw).hexdigest() != WEIGHTS_SHA256:
            raise SystemExit("FAIL: the weights file does not have the SHA-256 named in the plan")
        z = np.load(args.weights)
        keys = sorted(k for k in z.files if REAL_PATTERN.match(k) and z[k].ndim == 2)
        if len(keys) != REAL_COUNT:
            raise SystemExit(f"FAIL: {len(keys)} matrices match the plan's pattern, expected {REAL_COUNT}")
        for k in keys:
            rows += run("real/" + k, z[k], prior)
    (HERE / "results.json").write_text(json.dumps(rows, indent=1) + "\n")
    out = []
    summarise(rows, checks, out)
    out.append("")
    out.append(f"rows: {len(rows)}; matrices: {len({r['matrix'] for r in rows})}; real weights used: {'yes' if args.weights else 'no'}; seconds: {time.time() - t0:.0f}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
