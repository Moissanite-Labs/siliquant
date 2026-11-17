#!/usr/bin/env python3
"""Codebooks proposed in PR #6 against scalar rounding and E8 at equal bits per weight.

The plan is PREREG.md in this folder. Arms, data, the scale search and the
summary are fixed there. This file is the only program of the run.

The candidates are the point sets F_n and "CaB_n" built on the incentre
lattice of Foundation-0/incentre-lattices, and codebooks of 16 and 64 points
in the plane taken from relative equilibria of point vortices. The references
are the two arms of the first record, scalar rounding (S) and the E8 Voronoi
code with the shrink rule (E8), at 2 to 24 levels per weight. A reference at
a fractional number of bits is a mix of two neighbouring level counts.

Usage, from the root of a checkout, with NumPy:
  python3 experiments/incentre-vortex-codebooks-2026-11-17/run.py --smoke
  python3 experiments/incentre-vortex-codebooks-2026-11-17/run.py --weights /path/to/weights.npz --workers 4 > results.out
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "lattice"))
import lattices as L  # noqa: E402

# The scalar arm, the scale search and the E8 arm are those of the first record, imported and not copied.
_spec = importlib.util.spec_from_file_location("first_record", ROOT / "experiments" / "lattice-vs-scalar-rounding-2026-11-15" / "run.py")
first = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(first)

WEIGHTS_SHA256 = first.WEIGHTS_SHA256
REAL_PATTERN = first.REAL_PATTERN
REAL_COUNT = first.REAL_COUNT
SHRINK_STEPS = first.SHRINK_STEPS

C_IN = 2.0 - math.sqrt(2.0)          # the incentre coordinate of the proposal
WIDTH = 1680                          # divisible by 2, 7, 8, 15 and 16, so every block size tiles a row
Q_REF = tuple(range(2, 25))           # levels per weight of the two reference arms
K_CELLS = (2, 3, 4, 8, 16)            # cells per axis of the candidates
T_GRID = tuple(round(0.30 + 0.05 * i, 2) for i in range(19))   # 0.30 .. 1.20, for the arms with t searched
FAMILIES = (("F8", 8, False), ("F16", 16, False), ("CaB7", 7, True), ("CaB15", 15, True))
UNBOUNDED = 2 ** 20 + 1               # so many cells per axis that none is ever clipped
VORTEX_STARTS = {16: 200, 64: 100}
VORTEX_KEEP = 20
_weights_path = None
_cache = {}


def db(x):
    return 10.0 * math.log10(x)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def matrix(name):
    """A matrix of the study as an array of rows of 1680 weights."""
    if name == "smoke":
        return np.random.default_rng(7).uniform(-1, 1, size=(48, WIDTH))
    kind, rest = name.split("/", 1)
    if kind == "G":
        return np.random.default_rng(4000 + int(rest)).normal(size=(512, WIDTH))
    if kind == "T":
        return np.random.default_rng(5000 + int(rest)).standard_t(4, size=(512, WIDTH))
    z = np.load(_weights_path)
    w = z[rest].astype(np.float64).reshape(-1)
    return w[: len(w) // WIDTH * WIDTH].reshape(-1, WIDTH)


def real_names():
    raw = Path(_weights_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != WEIGHTS_SHA256:
        raise SystemExit("FAIL: the weights file does not have the SHA-256 named in the plan")
    z = np.load(_weights_path)
    keys = sorted(k for k in z.files if REAL_PATTERN.match(k) and z[k].ndim == 2)
    if len(keys) != REAL_COUNT:
        raise SystemExit(f"FAIL: {len(keys)} matrices match the plan's pattern, expected {REAL_COUNT}")
    return ["real/" + k for k in keys]


# ---------------------------------------------------------------------------
# The candidates built on the incentre lattice
# ---------------------------------------------------------------------------

def f_round(X, K, t, cab=False, centred=True):
    """Nearest codeword of the finite F_n or CaB_n codebook, found exactly. X is (N, n).

    Units: one cell is 2 wide. The cell centres on each axis are the K values 2k - (K - 1),
    k = 0..K-1. A codeword of F_n is a vector of cell centres with exactly one coordinate moved
    by +t or -t. CaB_n adds, for every vector of cell centres, that vector itself and that
    vector plus (1, ..., 1). So F_n has 2n K^n codewords and CaB_n has (2n + 2) K^n.

    Exactness: the squared distance to a codeword is a sum over coordinates. For a fixed choice
    of the moved coordinate i and sign s, every coordinate is free to take its nearest allowed
    value, so the best codeword for (i, s) is found coordinate by coordinate, and the best of
    the 2n choices is the one that lowers the unmoved error the most. For CaB_n the best centre
    vector and the best corner vector are found the same way and the nearest of the three is
    kept. selftest() checks this against a search over the whole codebook.

    The corners make the mean of the CaB_n codebook 1 / (2n + 2) on every coordinate. With
    centred=True the codebook is shifted so that its mean is zero.
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[1]
    off = K - 1.0
    mu = 1.0 / (2 * n + 2) if (cab and centred) else 0.0
    if mu:
        X = X + mu

    def centre(v):
        return 2.0 * np.clip(np.rint((v + off) / 2.0), 0, K - 1) - off

    C0 = centre(X)
    E0 = (X - C0) ** 2
    Cp = centre(X - t) + t
    Cm = centre(X + t) - t
    Rp = E0 - (X - Cp) ** 2
    Rm = E0 - (X - Cm) ** 2
    plus = Rp >= Rm
    R = np.where(plus, Rp, Rm)
    i = np.argmax(R, axis=1)
    rows = np.arange(len(X))
    rec = C0.copy()
    rec[rows, i] = np.where(plus[rows, i], Cp[rows, i], Cm[rows, i])
    if cab:
        e0 = E0.sum(axis=1)
        eF = e0 - R[rows, i]
        Cc = centre(X - 1.0) + 1.0
        ec = ((X - Cc) ** 2).sum(axis=1)
        use0 = (e0 <= eF) & (e0 <= ec)
        usec = (ec < eF) & (ec < e0)
        rec[use0] = C0[use0]
        rec[usec] = Cc[usec]
    return rec - mu


def f_codebook(n, K, t, cab=False):
    """Every codeword of the finite codebook, for the self-test. Small n and K only."""
    centres = 2.0 * np.arange(K) - (K - 1.0)
    grid = np.stack(np.meshgrid(*([centres] * n), indexing="ij"), axis=-1).reshape(-1, n)
    out = []
    for i in range(n):
        for s in (1.0, -1.0):
            c = grid.copy()
            c[:, i] += s * t
            out.append(c)
    if cab:
        out.append(grid.copy())
        out.append(grid + 1.0)
    return np.concatenate(out)


def f_bits(n, K, cab):
    return math.log2(K) + math.log2(2 * n + 2 if cab else 2 * n) / n


def g_closed_form(n, t):
    """Normalised second moment of F_n with the reproduction point at t e_i. Derived in PREREG.md."""
    block = n / (n + 2.0) - 2.0 * t * n / (n + 1.0) + t * t + n * (n - 1.0) / (3.0 * (n + 2.0))
    return block / (n * (2.0 ** n / (2.0 * n)) ** (2.0 / n))


def g_measured(n, t, cab, points, seed, chunk=250_000):
    """Monte Carlo value of the normalised second moment of the unbounded point set. Returns (G, standard error)."""
    rng = np.random.default_rng(seed)
    total = total_sq = 0.0
    done = 0
    while done < points:
        m = min(chunk, points - done)
        x = rng.uniform(-1.0, 1.0, size=(m, n))
        e = ((x - f_round(x, UNBOUNDED, t, cab=cab, centred=False)) ** 2).sum(axis=1)
        total += e.sum()
        total_sq += (e ** 2).sum()
        done += m
    mean = total / done
    se = math.sqrt(max(total_sq / done - mean ** 2, 0.0) / done)
    scale = n * (2.0 ** n / (2 * n + 2 if cab else 2 * n)) ** (2.0 / n)
    return mean / scale, se / scale


def selftest():
    """Checks of the code that use no matrix of the study. Returns a list of (ok, text)."""
    out = []
    rng = np.random.default_rng(11)
    worst = 0.0
    cases = 0
    for n, K, cab in ((2, 2, False), (2, 3, False), (3, 2, False), (3, 3, False), (4, 2, False), (3, 4, False),
                      (2, 3, True), (3, 2, True), (3, 3, True), (4, 2, True)):
        for t in (C_IN, 0.35, n / (n + 1.0), 1.15):
            cb = f_codebook(n, K, t, cab)
            x = rng.normal(size=(4000, n)) * rng.choice([0.4, 1.0, 2.5, 6.0], size=(4000, 1))
            fast = ((x - f_round(x, K, t, cab=cab, centred=False)) ** 2).sum(axis=1)
            slow = ((x - L.nearest_in_codebook(x, cb)[1]) ** 2).sum(axis=1)
            worst = max(worst, float(np.max(np.abs(fast - slow))))
            cases += 1
    out.append((worst < 1e-9, f"the fast rule finds the nearest codeword of the whole codebook: {cases} codebooks, 4000 blocks each, largest difference in squared distance {worst:.1e}"))
    for n, K, cab in ((3, 3, False), (3, 3, True)):
        cb = f_codebook(n, K, C_IN, cab)
        mean = cb.mean(axis=0)
        want = 1.0 / (2 * n + 2) if cab else 0.0
        out.append((float(np.max(np.abs(mean - want))) < 1e-12, f"mean of the {'CaB' if cab else 'F'}_{n} codebook at K = {K} is {want:.4f} on every coordinate, and it has {len(cb)} codewords ({(2 * n + (2 if cab else 0)) * K ** n} expected)"))
    return out


def voronoi_claim(n, points, seed):
    """PR #6: the cell of the point c e_i of the unbounded F_n is the pyramid over facet i.

    For x uniform in the cube [-1, 1)^n, compares the point the pyramid rule names (the axis
    of the largest |x_i|, with its sign) with a search over every point of F_n whose cell
    centre has coordinates -2, 0 or 2. Returns the largest amount by which the rule's point
    is farther than the nearest point found.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=(points, n))
    i = np.argmax(np.abs(x), axis=1)
    rule = np.zeros_like(x)
    rows = np.arange(points)
    rule[rows, i] = np.sign(x[rows, i]) * C_IN
    d_rule = ((x - rule) ** 2).sum(axis=1)
    cb = f_codebook(n, 3, C_IN)
    d_best = ((x - L.nearest_in_codebook(x, cb)[1]) ** 2).sum(axis=1)
    return float(np.max(d_rule - d_best))


# ---------------------------------------------------------------------------
# Codebooks in the plane
# ---------------------------------------------------------------------------

def vortex_energy(z):
    d = np.abs(z[:, None] - z[None, :])
    np.fill_diagonal(d, 1.0)
    return 0.5 * float(np.sum(np.abs(z) ** 2)) - 0.5 * float(np.sum(np.log(d)))


def vortex_gradient(z):
    d = z[:, None] - z[None, :]
    np.fill_diagonal(d, 1.0)
    inv = 1.0 / np.conj(d)
    np.fill_diagonal(inv, 0.0)
    return z - inv.sum(axis=1)


def vortex_hessian(z):
    N = len(z)
    p = np.stack([z.real, z.imag], axis=1)
    d = p[:, None, :] - p[None, :, :]
    r2 = (d ** 2).sum(axis=2)
    np.fill_diagonal(r2, 1.0)
    M = (2.0 * d[:, :, :, None] * d[:, :, None, :] - r2[:, :, None, None] * np.eye(2)) / (r2 ** 2)[:, :, None, None]
    M[np.arange(N), np.arange(N)] = 0.0
    H4 = -M
    H4[np.arange(N), np.arange(N)] = np.eye(2) + M.sum(axis=1)
    return H4.transpose(0, 2, 1, 3).reshape(2 * N, 2 * N)


def vortex_minimum(N, seed):
    """One stable relative equilibrium of N equal point vortices, from a random start.

    The energy is H = sum |z_a|^2 / 2 - sum over pairs of log |z_a - z_b|. Its critical points
    are the solutions of conj(z_a) = sum over b of 1 / (z_a - z_b), the balance equation that
    PR #6 quotes. A turn of the whole arrangement leaves H unchanged, so that direction is left
    out. The search is Newton's method with every eigenvalue of the Hessian replaced by its
    size, so that it runs downhill, with a halving step. At a critical point that still has a
    downhill direction it steps off along it. Returns (z, energy, residual of the balance
    equation, smallest eigenvalue of the Hessian apart from the turn) or None when the search
    does not end at a minimum.
    """
    rng = np.random.default_rng(seed)
    z = (rng.normal(size=N) + 1j * rng.normal(size=N)) * math.sqrt(N / 2.0)
    h = vortex_energy(z)

    def turn(zz):
        r = np.stack([-zz.imag, zz.real], axis=1).reshape(-1)
        return r / np.linalg.norm(r)

    def state(zz):
        g = vortex_gradient(zz)
        gv = np.stack([g.real, g.imag], axis=1).reshape(-1)
        r = turn(zz)
        Hm = vortex_hessian(zz)
        Hm = Hm - np.outer(r, r @ Hm) - np.outer(Hm @ r, r) + np.outer(r, r) * (r @ Hm @ r) + np.outer(r, r)
        lam, V = np.linalg.eigh(Hm)          # the turn now has eigenvalue 1 and is orthogonal to the rest
        return float(np.max(np.abs(g))), gv - r * (r @ gv), lam, V

    def moved(zz, dv, alpha=1.0):
        step = (alpha * dv).reshape(-1, 2)
        return zz + step[:, 0] + 1j * step[:, 1]

    low = None
    for _ in range(800):
        gmax, gv, lam, V = state(z)
        low = float(lam.min())
        if gmax < 1e-10 and low > -1e-8:
            break
        if gmax < 1e-5 and low > -1e-8:
            # Close to a minimum the energy no longer resolves a step, so take Newton's step and keep it if the gradient falls.
            # A direction that is nearly flat (one ring free to turn inside another) is damped, as in the search above.
            z2 = moved(z, -V @ ((V.T @ gv) / np.maximum(np.abs(lam), 1e-6)))
            if float(np.max(np.abs(vortex_gradient(z2)))) < gmax:
                z, h = z2, vortex_energy(z2)
                continue
            break
        if gmax < 1e-7:
            tries = [0.05 * V[:, 0], -0.05 * V[:, 0]]      # a critical point with a downhill direction: step off along it
        else:
            tries = [-V @ ((V.T @ gv) / np.maximum(np.abs(lam), 1e-6))]
        done = False
        for dv in tries:
            alpha = 1.0
            while alpha > 1e-10 and not done:
                z2 = moved(z, dv, alpha)
                h2 = vortex_energy(z2)
                if np.isfinite(h2) and h2 < h:
                    z, h, done = z2, h2, True
                alpha *= 0.5
            if done:
                break
        if not done:
            break
    res = float(np.max(np.abs(vortex_gradient(z))))
    if not np.isfinite(res) or res > 1e-9 or low is None or low < -1e-7:
        return None
    return z, vortex_energy(z), res, low


def vortex_minima(N):
    """The distinct stable equilibria found from the starts of the plan, lowest energy first."""
    key = ("vortex", N)
    if key not in _cache:
        found = {}
        failed = 0
        for s in range(VORTEX_STARTS[N]):
            r = vortex_minimum(N, 9000 + 1000 * N + s)
            if r is None:
                failed += 1
                continue
            k = round(r[1], 6)
            if k not in found:
                found[k] = {"z": r[0], "energy": r[1], "residual": r[2], "min_eig": r[3], "hits": 0}
            found[k]["hits"] += 1
        _cache[key] = ([found[k] for k in sorted(found)], failed)
    return _cache[key]


def points_of(z):
    return np.stack([z.real, z.imag], axis=1)


def lloyd_codebook(N, kind):
    """A codebook of N points in the plane fitted by Lloyd's algorithm to a sample of the family.

    The sample is 2^18 pairs drawn with a seed that no matrix of the study uses. Four starts of
    80 rounds, and the one with the least error on the sample is kept. It is the best codebook
    this search finds and not a proved optimum.
    """
    key = ("lloyd", N, kind)
    if key not in _cache:
        rng = np.random.default_rng({"G": 7000, "T": 8000}[kind] + N)
        x = rng.normal(size=(2 ** 18, 2)) if kind == "G" else rng.standard_t(4, size=(2 ** 18, 2))
        best = None
        for _ in range(4):
            cb = x[rng.choice(len(x), size=N, replace=False)].copy()
            for _ in range(80):
                idx = L.nearest_in_codebook(x, cb, chunk=8192)[0]
                for j in range(N):
                    sel = idx == j
                    if sel.any():
                        cb[j] = x[sel].mean(axis=0)
            err = float(((x - L.nearest_in_codebook(x, cb, chunk=8192)[1]) ** 2).sum() / (x ** 2).sum())
            if best is None or err < best[0]:
                best = (err, cb)
        _cache[key] = best[1]
    return _cache[key]


# ---------------------------------------------------------------------------
# One task: all rows of one kind for one matrix
# ---------------------------------------------------------------------------

def task_reference(args):
    """Arms S and E8 of the first record at q levels on one matrix."""
    name, q = args
    W = matrix(name)
    total = float(np.sum(W ** 2))
    X = W.reshape(-1, 8)
    sigma = float(W.std())
    t0 = 4.0 * sigma / q
    rows = []
    ts, es, edge = first.search_scale(lambda step: float(np.sum((X - first.scalar_round(X, q, step)) ** 2)) / total, t0)
    rows.append({"matrix": name, "arm": "S", "param": q, "bits": math.log2(q), "rel_err": es, "scale": ts, "at_edge": edge})
    Lt, c = first.lattice("E8"), first.centroid("E8", q)

    def err(tt):
        rec = L.voronoi_quantize(X / tt, Lt, q, centre=c, steps=SHRINK_STEPS)[0]
        return float(np.sum((X - rec * tt) ** 2)) / total

    tl, el, edge = first.search_scale(err, t0)
    rows.append({"matrix": name, "arm": "E8", "param": q, "bits": math.log2(q), "rel_err": el, "scale": tl, "at_edge": edge})
    return rows


def task_candidates(args):
    """The four families on the incentre lattice, as proposed (t = 2 - sqrt 2) and with t searched, on one matrix."""
    name, = args
    W = matrix(name)
    total = float(np.sum(W ** 2))
    sigma = float(W.std())
    rows = []
    for fam, n, cab in FAMILIES:
        X = W.reshape(-1, n)
        for K in K_CELLS:
            t0 = 2.0 * sigma / K
            best = None
            for t in (C_IN,) + T_GRID:
                tt, e, edge = first.search_scale(lambda h, t=t: float(np.sum((X - f_round(X / h, K, t, cab=cab) * h) ** 2)) / total, t0)
                if t == C_IN:
                    rows.append({"matrix": name, "arm": fam, "param": K, "bits": f_bits(n, K, cab), "rel_err": e, "scale": tt, "at_edge": edge, "t": t})
                if best is None or e < best[0]:
                    best = (e, tt, edge, t)
            rows.append({"matrix": name, "arm": fam + "t", "param": K, "bits": f_bits(n, K, cab), "rel_err": best[0], "scale": best[1], "at_edge": best[2], "t": best[3]})
    return rows


def plane_arm(W, cb, t0):
    total = float(np.sum(W ** 2))
    X = W.reshape(-1, 2)
    return first.search_scale(lambda h: float(np.sum((X - L.nearest_in_codebook(X / h, cb, chunk=16384)[1] * h) ** 2)) / total, t0)


def task_plane(args):
    """Codebooks of 16 and 64 points in the plane on one matrix."""
    name, every = args
    W = matrix(name)
    sigma = float(W.std())
    kind = name.split("/")[0]
    rows = []
    for N in (16, 64):
        minima, _ = vortex_minima(N)
        use = minima[:VORTEX_KEEP] if every else minima[:1]
        for rank, m in enumerate(use):
            cb = points_of(m["z"])
            cb = cb / math.sqrt(float((cb ** 2).sum()) / (2 * N))     # unit mean square per coordinate
            tt, e, edge = plane_arm(W, cb, sigma)
            rows.append({"matrix": name, "arm": f"V{N}", "param": rank, "bits": math.log2(N) / 2.0, "rel_err": e, "scale": tt, "at_edge": edge, "energy": m["energy"]})
        if kind in ("G", "T"):
            tt, e, edge = plane_arm(W, lloyd_codebook(N, kind), 1.0)
            rows.append({"matrix": name, "arm": f"LBG{N}", "param": 0, "bits": math.log2(N) / 2.0, "rel_err": e, "scale": tt, "at_edge": edge})
    return rows


def task_mix_control(args):
    """A mix of two level counts run as written, to check the weighted mean that stands in for it."""
    name, bits = args
    W = matrix(name)
    X = W.reshape(-1, 8)
    q_lo = int(math.floor(2.0 ** bits + 1e-9))
    q_hi = q_lo + 1
    p = (bits - math.log2(q_lo)) / (math.log2(q_hi) - math.log2(q_lo))
    hi = np.random.default_rng(99).random(len(X)) < p
    total = float(np.sum(W ** 2))
    sigma = float(W.std())
    rows = []
    for arm in ("S", "E8"):
        sq = 0.0
        for q, part in ((q_lo, X[~hi]), (q_hi, X[hi])):
            t0 = 4.0 * sigma / q
            if arm == "S":
                e = first.search_scale(lambda step: float(np.sum((part - first.scalar_round(part, q, step)) ** 2)), t0)[1]
            else:
                Lt, c = first.lattice("E8"), first.centroid("E8", q)
                e = first.search_scale(lambda tt: float(np.sum((part - L.voronoi_quantize(part / tt, Lt, q, centre=c, steps=SHRINK_STEPS)[0] * tt) ** 2)), t0)[1]
            sq += e
        rows.append({"matrix": name, "arm": arm + "-mix-run", "param": bits, "bits": float(np.mean(np.where(hi, math.log2(q_hi), math.log2(q_lo)))),
                     "rel_err": sq / total, "share_hi": float(hi.mean()), "q_lo": q_lo})
    return rows


def run_task(task):
    kind, args = task
    return {"reference": task_reference, "candidates": task_candidates, "plane": task_plane, "mix": task_mix_control}[kind](args)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def mix_error(curve, bits):
    """Error of the mix of two neighbouring level counts whose mean bits equal `bits`. curve maps q to error."""
    q_lo = int(math.floor(2.0 ** bits + 1e-9))
    if abs(math.log2(q_lo) - bits) < 1e-9:
        return curve[q_lo]
    q_hi = q_lo + 1
    p = (bits - math.log2(q_lo)) / (math.log2(q_hi) - math.log2(q_lo))
    return (1.0 - p) * curve[q_lo] + p * curve[q_hi]


def summarise(rows, checks, cells, claim, out):
    by = {}
    for r in rows:
        by[(r["matrix"], r["arm"], r["param"])] = r
    mats = sorted({r["matrix"] for r in rows})
    fams = [f for f in ("G", "T", "real") if any(m.split("/")[0] == f for m in mats)]

    def curve(m, arm):
        return {q: by[(m, arm, q)]["rel_err"] for q in Q_REF}

    out.append("0. Checks of the code, on no matrix of the study")
    for ok, text in checks:
        out.append(f"   {'ok  ' if ok else 'FAIL'} {text}")
    out.append("")
    out.append("1. What the cells are worth when they are fine and nothing is clipped: G against the cube's 1/12")
    out.append("   point set | reproduction point t | points per cube | G from the closed form | G measured | standard errors apart | gain in dB against the cube (negative is worse)")
    for c in cells:
        closed = "-" if c["closed"] is None else f"{c['closed']:.6f}"
        apart = "-" if c["closed"] is None else f"{(c['measured'] - c['closed']) / c['se']:+.1f}"
        out.append(f"   {c['name']:<6} | {c['t_name']:<22} | {c['points']:>3} | {closed:>8} | {c['measured']:.6f} +- {c['se']:.6f} | {apart:>5} | {db((1.0 / 12.0) / c['measured']):+.3f}")
    out.append("")
    out.append("2. The decoding claim of PR #6: the cell of each point of F_n is the pyramid over its facet")
    for n, points, gap in claim:
        out.append(f"   n = {n}: {points} points of the cube, the point named by the pyramid rule is farther than the nearest point found by at most {gap:.1e}")
    out.append("")
    out.append("3. The two reference arms, relative squared error in dB, mean over the matrices of a family")
    out.append("   q | bits | " + " | ".join(f"S on {f} | E8 on {f}" for f in fams))
    for q in Q_REF:
        cellsq = []
        for f in fams:
            ms = [m for m in mats if m.split("/")[0] == f]
            cellsq.append(f"{np.mean([db(by[(m, 'S', q)]['rel_err']) for m in ms]):.2f} | {np.mean([db(by[(m, 'E8', q)]['rel_err']) for m in ms]):.2f}")
        out.append(f"   {q:>2} | {math.log2(q):.3f} | " + " | ".join(cellsq))
    out.append("")
    out.append("4. The candidates at equal bits. A reference at the candidate's bits is the mix of two neighbouring level counts.")
    out.append("   Gap = candidate minus reference in dB. A positive gap means the candidate has the larger error.")
    out.append("   family | arm | K | bits | t | candidate dB | S mix dB | E8 mix dB | gap to S mix: mean (min, max) | worse than S mix | gap to E8 mix: mean (min, max) | worse than E8 mix | matrices")
    for f in fams:
        ms = [m for m in mats if m.split("/")[0] == f]
        for fam, n, cab in FAMILIES:
            for arm in (fam, fam + "t"):
                for K in K_CELLS:
                    if (ms[0], arm, K) not in by:
                        continue
                    bits = f_bits(n, K, cab)
                    cd = [db(by[(m, arm, K)]["rel_err"]) for m in ms]
                    sd = [db(mix_error(curve(m, "S"), bits)) for m in ms]
                    ed = [db(mix_error(curve(m, "E8"), bits)) for m in ms]
                    gs = [a - b for a, b in zip(cd, sd)]
                    ge = [a - b for a, b in zip(cd, ed)]
                    ts = sorted({by[(m, arm, K)]["t"] for m in ms})
                    tt = f"{ts[0]:.3f}" if len(ts) == 1 else f"{ts[0]:.2f}-{ts[-1]:.2f}"
                    out.append(f"   {f:>4} | {arm:<6} | {K:>2} | {bits:.4f} | {tt:>9} | {np.mean(cd):.2f} | {np.mean(sd):.2f} | {np.mean(ed):.2f} | "
                               f"{np.mean(gs):+.2f} ({min(gs):+.2f}, {max(gs):+.2f}) | {sum(g > 0 for g in gs)} | {np.mean(ge):+.2f} ({min(ge):+.2f}, {max(ge):+.2f}) | {sum(g > 0 for g in ge)} | {len(ms)}")
    out.append("")
    out.append("5. Codebooks of 16 and 64 points in the plane, at 2 and 3 bits per weight, relative squared error in dB")
    for N in (16, 64):
        minima, failed = vortex_minima(N)
        q = int(round(math.sqrt(N)))
        out.append(f"   {N} vortices: {VORTEX_STARTS[N]} starts, {failed} did not converge to a stable equilibrium, {len(minima)} distinct stable equilibria found.")
        out.append(f"   Lowest energy {minima[0]['energy']:.6f}, reached from {minima[0]['hits']} starts. Largest residual of the balance equation over all of them {max(m['residual'] for m in minima):.1e}. "
                   f"Sum of |z|^2 minus N(N-1)/2, largest size {max(abs(float(np.sum(np.abs(m['z']) ** 2)) - N * (N - 1) / 2.0) for m in minima):.1e}.")
        out.append(f"   family | V{N}, lowest energy | Lloyd codebook | S at q = {q} | E8 at q = {q} | V{N} minus E8: mean (min, max) | V{N} minus S: mean (min, max) | V{N} minus Lloyd: mean | matrices")
        for f in fams:
            ms = [m for m in mats if m.split("/")[0] == f]
            v = [db(by[(m, f"V{N}", 0)]["rel_err"]) for m in ms]
            s = [db(by[(m, "S", q)]["rel_err"]) for m in ms]
            e = [db(by[(m, "E8", q)]["rel_err"]) for m in ms]
            lb = [db(by[(m, f"LBG{N}", 0)]["rel_err"]) for m in ms if (m, f"LBG{N}", 0) in by]
            ge = [a - b for a, b in zip(v, e)]
            gs = [a - b for a, b in zip(v, s)]
            out.append(f"   {f:>4} | {np.mean(v):.2f} | {(f'{np.mean(lb):.2f}' if lb else '-'):>6} | {np.mean(s):.2f} | {np.mean(e):.2f} | {np.mean(ge):+.2f} ({min(ge):+.2f}, {max(ge):+.2f}) | "
                       f"{np.mean(gs):+.2f} ({min(gs):+.2f}, {max(gs):+.2f}) | {(f'{np.mean(v) - np.mean(lb):+.2f}' if lb else '-'):>6} | {len(ms)}")
        for m in [x for x in mats if x in ("G/0", "T/0")]:
            alle = [db(by[(m, f"V{N}", k)]["rel_err"]) for k in range(VORTEX_KEEP) if (m, f"V{N}", k) in by]
            out.append(f"   the {len(alle)} lowest equilibria on {m}: best {min(alle):.2f} dB, worst {max(alle):.2f} dB")
    out.append("")
    out.append("6. Controls")
    for r in [r for r in rows if r["arm"].endswith("-mix-run")]:
        arm = r["arm"].split("-")[0]
        formula = mix_error(curve(r["matrix"], arm), r["param"])
        out.append(f"   {arm} mix at {r['param']:.4f} bits on {r['matrix']}, run as written: {db(r['rel_err']):.3f} dB at {r['bits']:.4f} bits. The weighted mean used in table 4: {db(formula):.3f} dB.")
    low = [(r["matrix"], r["arm"], r["param"]) for r in rows if r["matrix"].startswith("G/") and db(r["rel_err"]) < -db(4.0) * r["bits"] - 0.05]
    out.append(f"   rows on the Gaussian family under Shannon's floor by more than 0.05 dB: {len(low)}" + (f" {low}" if low else ""))
    edges = [(r["matrix"], r["arm"], r["param"]) for r in rows if r.get("at_edge")]
    out.append(f"   scale searches that ended at an edge of their range: {len(edges)}" + (f" {edges[:8]}" if edges else ""))
    grid_edge = [(r["matrix"], r["arm"], r["param"], r["t"]) for r in rows if r["arm"].endswith("t") and r.get("t") in (T_GRID[0], T_GRID[-1])]
    out.append(f"   searches for t that ended at an end of its grid: {len(grid_edge)}" + (f" {grid_edge[:8]}" if grid_edge else ""))


def main():
    global _weights_path
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="run the code path on a matrix that is not in the study and print nothing but ok")
    ap.add_argument("--weights", help="path to weights.npz of mlx-community/whisper-base.en-mlx")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    t_start = time.time()
    checks = selftest()
    if args.smoke:
        global Q_REF, K_CELLS, VORTEX_STARTS
        Q_REF, K_CELLS, VORTEX_STARTS = (2, 3, 4, 5, 6, 8), (2, 3), {16: 6, 64: 2}
        tasks = [("reference", ("smoke", q)) for q in Q_REF] + [("candidates", ("smoke",)), ("plane", ("smoke", False)), ("mix", ("smoke", 2.5))]
        rows = [r for t in tasks for r in run_task(t)]
        cells = [{"name": "F3", "t_name": "incentre", "points": 6, "closed": g_closed_form(3, C_IN), **dict(zip(("measured", "se"), g_measured(3, C_IN, False, 200_000, 1)))}]
        out = []
        summarise(rows, checks, cells, [(3, 2000, voronoi_claim(3, 2000, 5))], out)
        ok = all(c[0] for c in checks) and all(0.0 < r["rel_err"] < 1.5 for r in rows)
        print("ok" if ok else "FAIL", f"({len(rows)} rows, {len(out)} lines, {time.time() - t_start:.0f} s)")
        sys.exit(0 if ok else 1)
    if not all(c[0] for c in checks):
        raise SystemExit("FAIL: a check of the code failed: " + "; ".join(t for ok, t in checks if not ok))
    names = [f"{k}/{s}" for k in ("G", "T") for s in range(3)]
    if args.weights:
        _weights_path = args.weights
        names += real_names()
    for N in (16, 64):
        vortex_minima(N)                 # found once here, then inherited by the workers
    for kind in ("G", "T"):
        for N in (16, 64):
            lloyd_codebook(N, kind)
    tasks = [("reference", (m, q)) for m in names for q in Q_REF]
    tasks += [("candidates", (m,)) for m in names]
    tasks += [("plane", (m, m in ("G/0", "T/0"))) for m in names]
    tasks += [("mix", (m, 2.5)) for m in ("G/0", "T/0")]
    with Pool(args.workers) as pool:
        rows = [r for part in pool.imap(run_task, tasks, chunksize=1) for r in part]
    cells = []
    for n in (2, 3, 4, 7, 8, 15, 16):
        for t_name, t in (("incentre, 2 - sqrt 2", C_IN), ("centroid, n / (n + 1)", n / (n + 1.0))):
            g, se = g_measured(n, t, False, 4_000_000, 20261117 + n)
            cells.append({"name": f"F{n}", "t_name": t_name, "points": 2 * n, "closed": g_closed_form(n, t), "measured": g, "se": se})
    for n in (7, 15):
        g, se = g_measured(n, C_IN, True, 4_000_000, 20261117 + n)
        cells.append({"name": f"CaB{n}", "t_name": "incentre, 2 - sqrt 2", "points": 2 * n + 2, "closed": None, "measured": g, "se": se})
    claim = [(n, 200_000, voronoi_claim(n, 200_000, 31 + n)) for n in (2, 3, 4, 5)]
    (HERE / "results.json").write_text(json.dumps({"rows": rows, "cells": cells, "voronoi_claim": claim,
                                                   "vortex": {str(N): [{"energy": m["energy"], "hits": m["hits"], "residual": m["residual"], "min_eig": m["min_eig"],
                                                                        "points": points_of(m["z"]).tolist()} for m in vortex_minima(N)[0]] for N in (16, 64)}}, indent=1) + "\n")
    out = []
    summarise(rows, checks, cells, claim, out)
    out.append(f"   rows: {len(rows)}; matrices: {len(names)}; real weights used: {'yes' if args.weights else 'no'}; seconds: {time.time() - t_start:.0f}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
