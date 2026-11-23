#!/usr/bin/env python3
"""Checks of harness/tq.py against the reference code in lattice/lattices.py.

Needs NumPy and torch. Usage, from the root of a checkout:
  python3 harness/test_tq.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "lattice"))
import lattices as L  # noqa: E402
import tq  # noqa: E402


def t(x):
    return torch.tensor(x, dtype=torch.float32, device=tq.DEV)


def main():
    rng = np.random.default_rng(11)
    fails = 0

    def check(name, ok, detail=""):
        nonlocal fails
        fails += not ok
        print(("ok   " if ok else "FAIL ") + name + (" " + detail if detail else ""))

    # 1. The decoders return the same lattice point as the reference, away from ties.
    x = rng.normal(scale=2.0, size=(200_000, 8)).astype(np.float32)
    for name, ref, fn in (("D8", L.Dn(8), tq.closest_D), ("E8", L.E8(), tq.closest_E8)):
        a = ref.closest(x.astype(np.float64))
        b = fn(t(x)).cpu().numpy().astype(np.float64)
        same = np.all(a == b, axis=1)
        # Where they differ, both must be equally near to x within float32 rounding.
        da = ((x - a) ** 2).sum(axis=1)
        db = ((x - b) ** 2).sum(axis=1)
        check(f"{name} decoder matches the reference", bool(np.all(same | (np.abs(da - db) < 1e-4))),
              f"({int((~same).sum())} of {len(x)} differ, all ties)")

    # 2. Every reconstruction is a codeword: decoding its index gives it back.
    for name in ("Z8", "E8"):
        ref = {"Z8": L.Zn(8), "E8": L.E8()}[name]
        for q in (4, 8, 16):
            code = tq.Code(name, q)
            xs = rng.normal(scale=q / 3.0, size=(100_000, 8)).astype(np.float32)
            r, over = code.quantize(t(xs))
            c = (r + code.centre).cpu().numpy().astype(np.float64)
            u = L.voronoi_index(c, ref, q)
            back = L.voronoi_decode(u, ref, q)
            check(f"{name} q={q}: reconstructions are codewords of the reference code",
                  bool(np.max(np.abs(back - c)) < 1e-3), f"(share shrunk {over:.3f})")

    # 2b. Issue #16: centred=False keeps 0 as an exact codeword without mutating the centred=True cache entry.
    c_on = tq.Code("E8", 16, centred=True)
    c_off = tq.Code("E8", 16, centred=False)
    r_zero_off, ov_off = c_off.quantize(t(np.zeros((8, 8), dtype=np.float32)))
    r_zero_on, _ = c_on.quantize(t(np.zeros((8, 8), dtype=np.float32)))
    check("Issue #16: centred=False preserves exact zero codeword and isolates cache",
          c_on is not c_off and c_on is tq.Code("E8", 16)
          and bool(np.all(c_off.centre.cpu().numpy() == 0.0))
          and bool(np.all(r_zero_off.cpu().numpy() == 0.0)) and ov_off == 0.0
          and float(np.linalg.norm(r_zero_on.cpu().numpy()[0])) > 0.75)

    # 3. Same error as the reference quantiser on the same blocks at the same scale.
    for name in ("Z8", "E8"):
        ref = {"Z8": L.Zn(8), "E8": L.E8()}[name]
        for q in (4, 16):
            code = tq.Code(name, q)
            xs = rng.standard_t(5, size=(60_000, 8)).astype(np.float32) * q / 5.0
            centre = L.codebook_centroid(ref, q)
            rn = L.voronoi_quantize(xs.astype(np.float64), ref, q, centre=centre, steps=tq.SHRINK_STEPS)[0]
            rt = code.quantize(t(xs))[0].cpu().numpy().astype(np.float64)
            en, et = np.mean((xs - rn) ** 2), np.mean((xs - rt) ** 2)
            check(f"{name} q={q}: error equals the reference", abs(et / en - 1.0) < 2e-3, f"(ratio {et / en:.6f})")

    # 4. Scalar rounding and the balancing equal the reference.
    xs = rng.normal(size=(4096, 8)).astype(np.float32)
    for q in (3, 4, 16):
        half = q // 2
        if q % 2 == 0:
            ref = (np.clip(np.floor(xs / 0.37), -half, half - 1) + 0.5) * 0.37
        else:
            ref = np.clip(np.rint(xs / 0.37), -half, half) * 0.37
        check(f"scalar q={q} equals the reference", bool(np.max(np.abs(tq.scalar_round(t(xs), q, 0.37).cpu().numpy() - ref)) < 1e-5))
    W = (np.exp(rng.normal(size=(96, 1))) * rng.normal(size=(96, 160)) * np.exp(rng.normal(size=(1, 160)))).astype(np.float32)
    s, tt, Wn = tq.dual_scale(t(W))
    sr, tr, Wr, _, _ = L.dual_scale(W.astype(np.float64))
    check("balancing equals the reference", bool(np.max(np.abs(Wn.cpu().numpy() - Wr)) < 5e-3)
          and bool(np.max(np.abs((s[:, None] * Wn * tt[None, :]).cpu().numpy() - W) / np.abs(W).max()) < 1e-4))

    # 5. round_matrix: the reported error is the error of what it returns, and E8 beats scalar on a Gaussian matrix.
    W = rng.normal(size=(512, 1024)).astype(np.float32)
    out = {}
    for arm in ("S", "E8", "D+S", "D+E8"):
        Wh, info = tq.round_matrix(W, arm, 4)
        rel = float(np.sum((W.astype(np.float64) - Wh.astype(np.float64)) ** 2) / np.sum(W.astype(np.float64) ** 2))
        out[arm] = info["rel_err"]
        check(f"round_matrix {arm}: reported error is the error of the returned matrix", abs(rel - info["rel_err"]) < 1e-12,
              f"({10 * np.log10(rel):.2f} dB)")
    check("E8 beats scalar at 2 bits on a Gaussian matrix by 0.4 to 1.0 dB", 0.4 < 10 * np.log10(out["S"] / out["E8"]) < 1.0,
          f"({10 * np.log10(out['S'] / out['E8']):.2f} dB)")
    # 6. The arms of the output-budget plan, against plain NumPy loops on small matrices.
    def np_round(group_vals, q, d):
        """Levels -q/2 .. q/2 - 1 times d, for one group."""
        if d == 0:
            return np.zeros_like(group_vals)
        return np.clip(np.rint(group_vals / d), -(q // 2), q // 2 - 1) * d

    def close16(a, b):
        a, b = a.astype(np.float64), b.astype(np.float64)
        return bool(np.all(np.abs(a - b) <= 2.5e-3 * np.abs(b) + 1e-6))

    def heavy(m, n):
        return (rng.standard_t(3, size=(m, n)) * np.exp(rng.normal(size=(m, 1)))).astype(np.float32)

    # 6a. Plain largest-entry rule, as the base the new arms change.
    W = heavy(6, 64)
    for q in (8, 16):
        ref = np.zeros(W.shape)
        for i in range(6):
            for g in range(0, 64, 32):
                x = W[i, g:g + 32].astype(np.float64)
                k = int(np.argmax(np.abs(x)))
                ref[i, g:g + 32] = np_round(x, q, x[k] / (-q / 2.0))
        check(f"maxscale_round q={q} equals the loop", close16(tq.maxscale_round(W, q, 32)[0], ref))

    # 6b. Kept entries (XRo, X32o).
    for group in ("R", 32):
        for q, lim in ((16, 1.5), (32, 2.5), (16, 0.5)):
            W = heavy(6, 64)
            Wh, info = tq.maxscale_round_kept(W, q, group, lim)
            size = 64 if group == "R" else 32
            ref = np.zeros(W.shape)
            kept = 0
            for i in range(6):
                sd = W[i].astype(np.float64).std()
                for g in range(0, 64, size):
                    x = W[i, g:g + size].astype(np.float64)
                    keep = np.abs(W[i].astype(np.float64))[g:g + size] > lim * sd
                    kept += int(keep.sum())
                    left = np.where(keep, -1.0, np.abs(x))
                    k = int(np.argmax(left))
                    r = np_round(x, q, x[k] / (-q / 2.0)) if not keep.all() else x
                    ref[i, g:g + size] = np.where(keep, x, r)
            expect_bits = math.log2(q) + 16.0 / size + kept * (16 + math.ceil(math.log2(64))) / W.size
            check(f"kept entries group={group} q={q} L={lim}: matrix, count and bits equal the loop",
                  close16(Wh, ref) and info["kept"] == kept and kept > 0 and abs(info["bits"] - expect_bits) < 1e-9,
                  f"(kept {kept})")
            kept_mask = np.zeros(W.shape, dtype=bool)
            for i in range(6):
                kept_mask[i] = np.abs(W[i].astype(np.float64)) > lim * W[i].astype(np.float64).std()
            check(f"kept entries group={group} q={q} L={lim}: kept entries are written as they are",
                  bool(np.all(Wh[kept_mask] == W[kept_mask].astype(np.float16))))
    # With a limit nothing reaches, the rule is the plain one.
    W = heavy(6, 64)
    check("kept entries with an unreachable limit equal the plain rule",
          np.array_equal(tq.maxscale_round_kept(W, 16, 32, 1e9)[0], tq.maxscale_round(W, 16, 32)[0]))

    # 6c. A scale stored in B bits (X32sB).
    for bits, q in ((8, 16), (6, 16), (8, 18), (6, 20), (3, 8)):
        W = heavy(8, 96)
        Wh, info = tq.maxscale_round_sbit(W, q, 32, bits)
        cmax = 2 ** bits - 1
        ref = np.zeros(W.shape)
        ok_scale, worst_ratio = True, 0.0
        steps, signs = np.zeros(W.shape), np.zeros(W.shape)
        for i in range(8):
            R = np.float64(np.float16(np.abs(W[i]).max()))
            if R < np.abs(W[i]).max():
                R = np.float64(np.nextafter(np.float16(np.abs(W[i]).max()), np.float16(np.inf)))
            for g in range(0, 96, 32):
                x = W[i, g:g + 32].astype(np.float64)
                a = np.abs(x).max()
                cands = [R * 2.0 ** (-8.0 * c / cmax) for c in range(cmax + 1)]
                s = min(v for v in cands if v >= a)           # the smallest stored value that is at least a
                ok_scale = ok_scale and s >= a
                worst_ratio = max(worst_ratio, a / s)
                k = int(np.argmax(np.abs(x)))
                sign = 1.0 if x[k] > 0 else -1.0
                ref[i, g:g + 32] = np_round(x, q, sign * s / (-q / 2.0))
                steps[i, g:g + 32] = s / (q / 2.0)
                signs[i, g:g + 32] = sign
        expect_bits = math.log2(q) + bits / 32.0 + 16.0 / 96
        check(f"X32s{bits}:{q}: matrix and bits equal the loop, scales at least the largest entries",
              close16(Wh, ref) and ok_scale and info["scale_ok"] and abs(info["bits"] - expect_bits) < 1e-12,
              f"(largest entry over its scale at most {worst_ratio:.4f})")
        # No entry is clipped: every error is at most half a step (the step is scale / (q / 2)),
        # except an entry of the sign opposite to the group's largest, within the last step of the other end.
        err = np.abs(Wh.astype(np.float64) - W.astype(np.float64))
        half_step = steps / 2.0
        opposite_edge = (np.sign(W) != signs) & (np.abs(W) > (q / 2.0 - 0.5) * steps)
        check(f"X32s{bits}:{q}: no entry is clipped",
              bool(np.all((err <= half_step * 1.001 + 2.5e-3 * np.abs(W) + 1e-6) | opposite_edge)))
    # The largest entry of a group far below its row keeps the step of the smallest stored scale.
    W = np.ones((2, 64), dtype=np.float32) * 0.001
    W[0, 0] = 1.0
    Wh, info = tq.maxscale_round_sbit(W, 16, 32, 6)
    check("X32s6: a group far below its row is not clipped", bool(np.all(np.abs(Wh[0, 32:] - 0.001) < 0.001)) and info["scale_ok"])

    print("FAILED" if fails else "all checks pass")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
