"""The rounding arms on a GPU, in 32-bit floats, for whole models.

The reference code is lattice/lattices.py (NumPy, 64-bit). This file repeats
the Z^8 and E_8 decoders, the Voronoi code with the shrink rule, scalar
rounding and the balancing of rows and columns in torch, so that a few
billion weights take minutes. harness/test_tq.py checks it against the
reference.

Differences from the first record's program, all for speed:
  - the scale of a matrix is searched on a fixed random sample of its blocks,
    and the whole matrix is then rounded once at the scale found;
  - sums over the sample are in 32-bit floats. The error reported for a
    matrix is computed afterwards in 64-bit floats on the values written to
    the model file.
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lattice"))
import lattices as L  # noqa: E402

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SHRINK_STEPS = 12
SAMPLE_BLOCKS = 131072


def closest_Z(x):
    return torch.round(x)


def closest_D(x):
    f = torch.round(x)
    odd = torch.remainder(f.sum(dim=1), 2.0) != 0
    err = x - f
    j = err.abs().argmax(dim=1, keepdim=True)
    step = torch.where(err.gather(1, j) >= 0, 1.0, -1.0) * odd[:, None]
    return f.scatter_add(1, j, step)


def closest_E8(x):
    a = closest_D(x)
    b = closest_D(x - 0.5) + 0.5
    da = ((x - a) ** 2).sum(dim=1)
    db = ((x - b) ** 2).sum(dim=1)
    return torch.where((db < da)[:, None], b, a)


class Code:
    """The Voronoi code L / qL with the centre and the offset of the reference code."""

    _cache = {}

    def __new__(cls, name, q):
        key = (name, q)
        if key not in cls._cache:
            self = super().__new__(cls)
            lat = {"Z8": L.Zn(8), "E8": L.E8()}[name]
            f32 = dict(dtype=torch.float32, device=DEV)
            self.name, self.q, self.volume = name, q, lat.volume
            self.G = torch.tensor(lat.G, **f32)
            self.Ginv = torch.tensor(lat.Ginv, **f32)
            self.a = torch.tensor(L.default_offset(8), **f32)
            self.centre = torch.tensor(L.codebook_centroid(lat, q), **f32)
            self.closest = {"Z8": closest_Z, "E8": closest_E8}[name]
            cls._cache[key] = self
        return cls._cache[key]

    def roundtrip(self, w):
        y = self.closest(w)
        u = torch.remainder(torch.round(y @ self.Ginv), float(self.q))
        p = u @ self.G
        c = p - self.q * self.closest((p - self.a) / self.q)
        return c, ((c - y).abs() < 1e-3).all(dim=1)

    def quantize(self, x, steps=SHRINK_STEPS):
        """Blocks x (N, 8) to codewords minus the centre. Returns (reconstruction, share of blocks shrunk)."""
        c, ok = self.roundtrip(x + self.centre)
        over = ~ok
        n_over = int(over.sum())
        if n_over:
            xo = x[over]
            lo = torch.zeros(n_over, device=x.device)
            hi = torch.ones(n_over, device=x.device)
            for _ in range(steps):
                mid = 0.5 * (lo + hi)
                good = self.roundtrip(xo * mid[:, None] + self.centre)[1]
                lo = torch.where(good, mid, lo)
                hi = torch.where(good, hi, mid)
            c[over] = self.roundtrip(xo * lo[:, None] + self.centre)[0]
        return c - self.centre, n_over / len(x)


def scalar_round(x, q, step):
    """q evenly spaced levels with clipping at the ends. Mid-rise for even q, mid-tread for odd q."""
    if q % 2 == 0:
        return (torch.clamp(torch.floor(x / step), -q // 2, q // 2 - 1) + 0.5) * step
    return torch.clamp(torch.round(x / step), -(q // 2), q // 2) * step


def dual_scale(W, iters=16):
    """Rows, then columns, divided by their standard deviations, `iters` times. W = s[:, None] * Wn * t[None, :]."""
    s = torch.ones(W.shape[0], device=W.device)
    t = torch.ones(W.shape[1], device=W.device)
    Wn = W.clone()
    for _ in range(iters):
        rs = Wn.std(dim=1, correction=0)
        Wn = Wn / rs[:, None]
        s = s * rs
        cs = Wn.std(dim=0, correction=0)
        Wn = Wn / cs[None, :]
        t = t * cs
    return s, t, Wn


def search_scale(err, t0):
    """Scale that minimises err(t): half-octave grid widened at the edges, then golden section on log t.

    The same search as the first record's program.
    """
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
    return t_best


def maxscale_round(W, q, group, shrink=1.0):
    """Scalar rounding with one scale per group of weights and no search: llama.cpp's Q4_0 rule at q levels.

    group is a number of consecutive weights in a row, "R" for a whole row or "M" for the whole
    matrix. In each group the entry of largest size, with its sign, is mapped exactly to level
    -q/2, the step is that entry divided by -q/2, and the other entries are rounded to levels
    -q/2 .. q/2 - 1. Nothing is clipped except entries of the opposite sign above (q/2 - 1) steps.
    With shrink below 1 the step is multiplied by it, so that entries above shrink times the
    largest are clipped. The scale of a group is counted at 16 bits.
    Returns (the rounded matrix as float16, a dict).
    """
    m, n = W.shape
    Wt = torch.from_numpy(np.ascontiguousarray(W)).to(DEV)
    size = n if group == "R" else (m * n if group == "M" else int(group))
    X = Wt.reshape(-1, size)
    j = X.abs().argmax(dim=1, keepdim=True)
    d = X.gather(1, j) * shrink / (-q / 2.0)
    safe = torch.where(d == 0, torch.ones_like(d), d)
    R = torch.clamp(torch.round(X / safe), -q // 2, q // 2 - 1) * d
    Wh = R.reshape(m, n).to(torch.float16).cpu().numpy()
    W64 = W.astype(np.float64)
    rel = float(np.sum((W64 - Wh.astype(np.float64)) ** 2) / np.sum(W64 ** 2))
    over = float((((X / safe) > (q // 2 - 1) + 0.5) | ((X / safe) < -(q // 2) - 0.5)).float().mean())
    return Wh, {"rel_err": rel, "scale": None, "overload": over, "bits": math.log2(q) + 16.0 / size}


def _finish(W, Wh, over, bits, extra=None):
    W64 = W.astype(np.float64)
    rel = float(np.sum((W64 - Wh.astype(np.float64)) ** 2) / np.sum(W64 ** 2))
    info = {"rel_err": rel, "scale": None, "overload": over, "bits": bits}
    info.update(extra or {})
    return Wh, info


def maxscale_round_kept(W, q, group, limit):
    """maxscale_round with the largest entries of each row kept as they are (arms XRo and X32o of the plan).

    In each row, the entries whose size is above `limit` standard deviations of the row (the
    standard deviation of the whole row, with divisor n) are kept exactly. The scale of a group
    (a row for group "R", else `group` consecutive weights) is set from the group's largest
    entry that is not kept; the entries that are not kept are rounded as maxscale_round does, at
    q levels. The result is the rounded matrix as float16, with the kept entries written as
    float16 of their own value.
    Bits per weight: log2(q) + 16 / group size, plus for each kept entry 16 bits for its value
    and ceil(log2(row length)) bits for its position, shared over all the weights. The info dict
    also holds "kept", the number of kept entries.
    """
    if q % 2:
        raise ValueError("even level counts only")
    m, n = W.shape
    size = n if group == "R" else int(group)
    lim = float(limit) * W.astype(np.float64).std(axis=1, keepdims=True)
    kept_np = np.abs(W.astype(np.float64)) > lim
    Wt = torch.from_numpy(np.ascontiguousarray(W)).to(DEV)
    kept = torch.from_numpy(kept_np).to(DEV)
    X = Wt.reshape(-1, size)
    K = kept.reshape(-1, size)
    rest = torch.where(K, torch.full_like(X, -1.0), X.abs())
    j = rest.argmax(dim=1, keepdim=True)
    d = X.gather(1, j) / (-q / 2.0)
    safe = torch.where(d == 0, torch.ones_like(d), d)
    R = torch.clamp(torch.round(X / safe), -q // 2, q // 2 - 1) * d
    R = torch.where(K, X, R)
    Wh = R.reshape(m, n).to(torch.float16).cpu().numpy()
    ratio = X / safe
    clipped = ((ratio > (q // 2 - 1) + 0.5) | (ratio < -(q // 2) - 0.5)) & ~K
    n_kept = int(kept_np.sum())
    bits = math.log2(q) + 16.0 / size + n_kept * (16.0 + math.ceil(math.log2(n))) / (m * n)
    return _finish(W, Wh, float(clipped.float().mean()), bits, {"kept": n_kept})


def sbit_scales(a, row_max16, bits):
    """The stored scale of each group, for the arms X32sB: a (largest size in each group), row_max16 (the row's largest, as stored).

    The scale is the smallest of R * 2^(-8 c / (2^B - 1)), c = 0 .. 2^B - 1, that is at least a.
    NumPy float64 in, float64 out.
    """
    cmax = 2 ** bits - 1
    a = np.asarray(a, dtype=np.float64)
    R = np.asarray(row_max16, dtype=np.float64)

    def val(c):
        return R * 2.0 ** (-8.0 * c / cmax)

    with np.errstate(divide="ignore"):
        c = np.floor(cmax * np.log2(R / np.where(a > 0, a, 1.0)) / 8.0)
    c = np.where(a > 0, c, cmax)
    c = np.clip(c, 0, cmax)
    for _ in range(2):
        c = np.where(val(c) < a, c - 1, c)
        c = np.where((c < cmax) & (val(np.minimum(c + 1, cmax)) >= a), c + 1, c)
    c = np.clip(c, 0, cmax)
    return val(c)


def maxscale_round_sbit(W, q, group, bits):
    """maxscale_round with each group's scale stored in `bits` bits (arms X32sB of the plan).

    The row's largest entry R is stored in 16 bits (as a float16 that is not below it, so that
    nothing is clipped). A group's scale s is the smallest of R * 2^(-8 c / (2^B - 1)), c from 0
    to 2^B - 1, that is at least the group's largest entry. The step is s / (q / 2), and the
    entries are mapped as in maxscale_round, with the sign of the group's largest entry deciding
    which side the end level is on. Bits per weight: log2(q) + B / group size + 16 / row length.
    The info dict also holds "scale_ok", true when every group's scale is at least its largest entry.
    """
    if q % 2:
        raise ValueError("even level counts only")
    m, n = W.shape
    size = int(group)
    Wt = torch.from_numpy(np.ascontiguousarray(W)).to(DEV)
    X = Wt.reshape(-1, size)
    j = X.abs().argmax(dim=1, keepdim=True)
    top = X.gather(1, j)
    a = top.abs().reshape(-1).cpu().numpy().astype(np.float64)
    rowmax = np.abs(W).max(axis=1)
    r16 = rowmax.astype(np.float16)
    r16 = np.where(r16.astype(np.float32) < rowmax, np.nextafter(r16, np.float16(np.inf)), r16).astype(np.float64)
    Rg = np.repeat(r16, n // size)
    s = sbit_scales(a, Rg, bits)
    st = torch.from_numpy(s.astype(np.float32)).to(DEV)[:, None]
    d = torch.sign(top) * st / (-q / 2.0)
    safe = torch.where(d == 0, torch.ones_like(d), d)
    R = torch.clamp(torch.round(X / safe), -q // 2, q // 2 - 1) * d
    Wh = R.reshape(m, n).to(torch.float16).cpu().numpy()
    over = float((((X / safe) > (q // 2 - 1) + 0.5) | ((X / safe) < -(q // 2) - 0.5)).float().mean())
    return _finish(W, Wh, over, math.log2(q) + bits / size + 16.0 / n, {"scale_ok": bool(np.all(s >= a))})


def maxscale_voronoi(W, q, group, name="E8", steps=14):
    """A Voronoi code with one scale per group of weights and nothing shrunk: the lattice form of maxscale_round.

    Each block of 8 has a smallest scale at which it lies inside the code's region (its nearest
    lattice point decodes to itself). It is found by bisection between two bounds that hold for
    any block: the region contains the ball of the lattice's packing radius times q and lies
    inside the ball of its covering radius times q. A group takes the largest of its blocks'
    scales, so no block of the group is shrunk. group is a number of consecutive weights in a row
    (a multiple of 8) or "R" for a row. The scale is counted at 16 bits per group.
    """
    code = Code(name, q)
    m, n = W.shape
    Wt = torch.from_numpy(np.ascontiguousarray(W)).to(DEV)
    size = n if group == "R" else int(group)
    X = Wt.reshape(-1, 8)
    norm = torch.clamp(torch.linalg.vector_norm(X, dim=1), min=1e-20)
    rho, cover = {"E8": (0.5 ** 0.5, 1.0), "Z8": (0.5, 2.0 ** 0.5)}[name]
    lo = norm / (cover * q) * 0.9        # too small: the block is outside
    hi = norm / (rho * q) * 1.1          # large enough: the block is inside
    for _ in range(steps):
        mid = torch.sqrt(lo * hi)
        ok = code.roundtrip(X / mid[:, None] + code.centre)[1]
        hi = torch.where(ok, mid, hi)
        lo = torch.where(ok, lo, mid)
    t = hi.reshape(-1, size // 8).max(dim=1, keepdim=True).values.expand(-1, size // 8).reshape(-1)
    R, over = code.quantize(X / t[:, None])
    Wh = (R * t[:, None]).reshape(m, n).to(torch.float16).cpu().numpy()
    W64 = W.astype(np.float64)
    rel = float(np.sum((W64 - Wh.astype(np.float64)) ** 2) / np.sum(W64 ** 2))
    return Wh, {"rel_err": rel, "scale": None, "overload": over, "bits": math.log2(q) + 16.0 / size}


def path_scales(Wt, row_w, col_w, iters=4):
    """Row scales d and column scales e for rounding under a weighted error with weights row_w[i] * col_w[j].

    A uniform quantiser's noise grows with the size of the matrix it is given. Scaling rows by d
    and columns by e before rounding, and back after, leaves noise in entry (i, j) of size
    proportional to 1 / (d_i e_j). The weighted noise is smallest when d_i^2 is proportional to
    sqrt(row_w_i) / a_i, with a_i the size of row i of the column-scaled matrix, and likewise for
    e. The two conditions are solved in turn, `iters` times.
    """
    m, n = Wt.shape
    W2 = Wt.cpu().double() ** 2
    rw = torch.ones(m, dtype=torch.float64) if row_w is None else torch.as_tensor(row_w, dtype=torch.float64)
    cw = torch.ones(n, dtype=torch.float64) if col_w is None else torch.as_tensor(col_w, dtype=torch.float64)
    rw = torch.clamp(rw / rw.mean(), min=1e-12)
    cw = torch.clamp(cw / cw.mean(), min=1e-12)
    d2 = torch.ones(m, dtype=torch.float64)
    e2 = torch.ones(n, dtype=torch.float64)
    tiny = 1e-30
    for _ in range(iters):
        a = torch.sqrt(torch.clamp(W2 @ e2, min=tiny) / n)
        d2 = torch.sqrt(rw) / a
        d2 = d2 / torch.exp(torch.log(d2).mean())
        b = torch.sqrt(torch.clamp(d2 @ W2, min=tiny) / m)
        e2 = torch.sqrt(cw) / b
        e2 = e2 / torch.exp(torch.log(e2).mean())
    return torch.sqrt(d2).float().to(Wt.device), torch.sqrt(e2).float().to(Wt.device), rw, cw


def round_matrix(W, arm, q, seed=0, sample=SAMPLE_BLOCKS, row_w=None, col_w=None, row_power=None):
    """Round one matrix (NumPy float32, entries a multiple of 8) with one arm at q levels per weight.

    Arms: "S" scalar, "Z8", "E8", each with an optional prefix: "D+" rows and columns balanced,
    "R+" rows only (each row divided by its standard deviation), "C+" columns only. The scale
    vectors are counted at 16 bits per entry. "P+" scales rows and columns for the weighted error
    with weights row_w[i] * col_w[j] (see path_scales) and searches the step on that weighted error.
    With row_power p and prefix "R+", rows are divided by their standard deviation to the power p.
    "B32+" gives each block of 32 consecutive weights in a row its own scale (its root mean square).
    Returns (the rounded matrix as float16, a dict of what happened).
    """
    prefix, base = (arm[:2], arm[2:]) if arm[:2] in ("D+", "R+", "C+", "P+") else ("", arm)
    blk = re.match(r"^B(\d+)\+(.*)$", arm)
    if blk:
        prefix, base = "B+", blk.group(2)
    dual = prefix != ""
    m, n = W.shape
    Wt = torch.from_numpy(np.ascontiguousarray(W)).to(DEV)
    weight = None
    if prefix == "B+":
        # One scale per block of consecutive weights in a row: the block's root mean square.
        size = int(blk.group(1))
        s = torch.sqrt((Wt.reshape(m, n // size, size) ** 2).mean(dim=2))
        s = torch.where(s == 0, torch.ones_like(s), s).repeat_interleave(size, dim=1)
        Wn = Wt / s
        back = s.reshape(-1, 8)
        side = 16.0 / size
    elif prefix == "P+":
        d, e, rw, cw = path_scales(Wt, row_w, col_w)
        Wn = Wt * d[:, None] * e[None, :]
        back = (1.0 / (d[:, None] * e[None, :])).reshape(-1, 8)
        weight = (torch.sqrt(rw).float().to(DEV)[:, None] * torch.sqrt(cw).float().to(DEV)[None, :]).reshape(-1, 8)
        side = 16.0 * (m + n) / (m * n)
    elif prefix == "R+" and row_power is not None:
        s = Wt.std(dim=1, correction=0) ** row_power
        Wn = Wt / s[:, None]
        back = (s[:, None] * torch.ones(n, device=DEV)[None, :]).reshape(-1, 8)
        side = 16.0 * m / (m * n)
    elif prefix == "D+":
        s, t, Wn = dual_scale(Wt)
        back = (s[:, None] * t[None, :]).reshape(-1, 8)
        side = 16.0 * (m + n) / (m * n)
    elif prefix == "R+":
        s = Wt.std(dim=1, correction=0)
        Wn = Wt / s[:, None]
        back = (s[:, None] * torch.ones(n, device=DEV)[None, :]).reshape(-1, 8)
        side = 16.0 * m / (m * n)
    elif prefix == "C+":
        t = Wt.std(dim=0, correction=0)
        Wn = Wt / t[None, :]
        back = (torch.ones(m, device=DEV)[:, None] * t[None, :]).reshape(-1, 8)
        side = 16.0 * n / (m * n)
    else:
        Wn, back, side = Wt, None, 0.0
    X = Wn.reshape(-1, 8)
    N = len(X)
    if N > sample:
        idx = torch.from_numpy(np.random.default_rng(seed).choice(N, size=sample, replace=False)).to(DEV)
        Xs, Ws = X[idx], Wt.reshape(-1, 8)[idx]
        Bs = back[idx] if dual else None
        Vs = weight[idx] if weight is not None else None
    else:
        Xs, Ws, Bs, Vs = X, Wt.reshape(-1, 8), back, weight
    if Vs is not None:
        Ws = Ws * Vs
        Bs = Bs * Vs
    total_s = float((Ws ** 2).sum())
    sigma = float(Wn.std())
    t0 = 4.0 * sigma / q

    if base == "S":
        def rec(x, tt):
            return scalar_round(x, q, tt), None
    else:
        code = Code(base, q)
        t0 = t0 / code.volume ** (1.0 / 8.0)

        def rec(x, tt):
            r, over = code.quantize(x / tt)
            return r * tt, over

    def err(tt):
        r = rec(Xs, tt)[0]
        if dual:
            r = r * Bs
        return float(((Ws - r) ** 2).sum()) / total_s

    tt = search_scale(err, t0)
    R, over = rec(X, tt)
    if dual:
        R = R * back
    if base == "S":
        over = float((X.abs() / tt > q / 2.0).float().mean())
    Wh = R.reshape(m, n).to(torch.float16).cpu().numpy()
    W64 = W.astype(np.float64)
    rel = float(np.sum((W64 - Wh.astype(np.float64)) ** 2) / np.sum(W64 ** 2))
    return Wh, {"rel_err": rel, "scale": tt, "overload": over, "bits": math.log2(q) + side}
