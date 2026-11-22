#!/usr/bin/env python3
"""Stage 4 of issue #10 for one ball per group: every setting of the knobs at a budget of bits, with what a decoder costs.

Issue #10 asks for the knobs that make the error smallest "under two hard limits: total bits no
more than the budget, and decode cost no more than the budget". For an arm with one ball per
group both sides can now be written down before anything is rounded:

  the error    shape/ballformula.py, with the exact count of the codebook. It reads no weight.
  the output   one scored arm of a model gives its mean KL per unit of weight error, and the
               claim of notes/issue10-output-predictions.md is that the rest follow from it.
  the decoder  shape/ballindex.py numbers the points of a codebook. Its table has a known
               number of entries, each a whole number as long as a group's index, and reading
               a group back takes a measured number of look-ups per weight.

The knobs: the lattice (integer levels or E8), the weights under one ball (32, 64 or 128), and
the bits of the scale (16, 8 or 6: the last two round the scale up to a grid).

It also gives, for each setting, the bound of shape/certified.py carried to one ball per group:
at the safe scale, the largest norm of the group over (r - rho), no group of any matrix has an
error over rho^2 / (r - rho)^2 of its squared weights. rho is the covering radius of the lattice
over n weights: sqrt(n) / 2 for the integers and sqrt(n / 8) for E8.

Usage, from the root of a checkout, with NumPy:
  python3 shape/design.py > shape/design.out
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shape"))
sys.path.insert(0, str(ROOT / "lattice"))
import ballformula as F  # noqa: E402
import ballindex as BI  # noqa: E402
import lattices as L  # noqa: E402

_spec = importlib.util.spec_from_file_location("balldim", ROOT / "experiments" / "ball-dimension-no-clip-rule-2026-11-17" / "run.py")
balldim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(balldim)

RUNS = (("Qwen2.5-1.5B", ROOT / "experiments" / "model-output-qwen25-2026-11-17" / "results.jsonl"),
        ("SmolLM2-1.7B", ROOT / "experiments" / "model-output-smollm2-2026-11-17" / "results.jsonl"))
BUDGETS = (4.5, 4.0)
GROUPS = (32, 64, 128)
SCALE_BITS = (16, 8, 6)
BLOCKS = 40                       # groups numbered and read back to count look-ups
MAX_ENTRIES = 6_000_000           # tables above this are sized from their formula and not built


def ball_bound(lattice, n, bits):
    """The largest bound at which the ball over n weights holds no more than 2^bits points: (bound, points). As in the Qwen run."""
    cap = 1 << bits
    r_unit = math.exp(math.lgamma(n / 2.0 + 1.0) / n) / math.sqrt(math.pi)
    limit = int(1.08 * (r_unit * 2.0 ** (bits / n)) ** 2) + 2 * n
    step = 2 if lattice == "E8" else 1
    while True:
        total = balldim.poly_pow(balldim.shells8(lattice, "ball", limit), n // 8, limit)
        run, best = 0, None
        for r, c in enumerate(total):
            run += c
            if run > cap:
                break
            if r % step == 0:
                best = (r, run)
        if run > cap:
            break
        limit = int(limit * 1.3) + 8
    return best


def table_entries(lattice, n, bound):
    """The entries of the table of shape/ballindex.py, from its own formula."""
    if lattice == "Z8":
        return (n + 1) * (bound + 1)
    budget = 4 * bound
    return (n // 8) * 9 * 4 * (budget + 1) + (n // 8 + 1) * (budget + 1)


def lookups(lattice, n, bound, e8):
    """Look-ups per weight to read a group back, counted on Gaussian groups rounded into the codebook."""
    ix = BI.BallIndex(lattice, n, bound)
    rng = np.random.default_rng(4000 + n + bound)
    ys = []
    for _ in range(BLOCKS):
        x = rng.normal(size=n)
        scale = math.sqrt(float((x ** 2).sum()) / bound)
        while True:
            y = np.rint(x / scale) if lattice == "Z8" else e8.closest((x / scale).reshape(-1, 8)).reshape(-1)
            if ix.contains(y):
                break
            scale *= 1.01
        ys.append(y)
    ks = [ix.rank(y) for y in ys]
    ix.lookups = 0
    ok = all(np.array_equal(ix.unrank(k), y) for y, k in zip(ys, ks))
    return ix.lookups / (BLOCKS * n), ok


def anchors():
    out = {}
    for name, path in RUNS:
        if not path.exists():
            continue
        rows = {}
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r["variant"]] = r
        if "X32:16" in rows:
            out[name] = (rows["X32:16"]["kl_mean"], rows["X32:16"]["weight_err_db_mean"], rows)
    return out


def name_of(lattice, n, point_bits, scale_bits):
    return f"{lattice[0]}BG{n}:b{point_bits}s{scale_bits}"


def main():
    e8 = L.E8()
    anc = anchors()
    out = []
    P = out.append
    P("One ball per group: every setting of the knobs at a budget, by the closed form of shape/ballformula.py. Nothing is rounded but the sample groups that count look-ups.")
    for name, (kl, err, _) in anc.items():
        P(f"Anchor of {name}: X32:16, mean KL {kl:.5f} at a mean weight error of {err:.3f} dB.")
    P("")
    rows = []
    for budget in BUDGETS:
        for lattice in ("Z8", "E8"):
            for n in GROUPS:
                for s in SCALE_BITS:
                    b = int(round(budget * n)) - s
                    bound, points = ball_bound(lattice, n, b)
                    steps = F.SCALE_STEPS[s]
                    err = F.G_CELL[lattice[0]] * F.phi_ball(n) * points ** (-2.0 / n) * F.scale_factor(n, steps)
                    r = math.sqrt(bound)
                    rho = math.sqrt(n) / 2.0 if lattice == "Z8" else math.sqrt(n / 8.0)
                    r_eff = math.exp((math.log(points) - F.log_ball_volume(n)) / n)
                    entries = table_entries(lattice, n, bound)
                    width = points.bit_length()
                    t0 = time.time()
                    per_weight, ok = lookups(lattice, n, bound, e8) if entries <= MAX_ENTRIES else (None, None)
                    rows.append({"budget": budget, "name": name_of(lattice, n, b, s), "lattice": lattice, "n": n, "scale_bits": s, "point_bits": b,
                                 "bits": (math.log2(points) + s) / n, "bound": bound, "err_db": F.db(err),
                                 "safe_bound_db": F.db(rho ** 2 / (r - rho) ** 2 * (1.0 if steps is None else 2.0 ** (2.0 / steps))),
                                 "safe_cost_db": 20.0 * math.log10(r_eff / (r - rho)) - (F.db(1.0 + 2.0 / n) if steps is None else 0.0),
                                 "entries": entries, "width": width, "megabytes": entries * math.ceil(width / 8) / 1e6,
                                 "lookups": per_weight, "read_back_ok": ok, "seconds": time.time() - t0})
    for budget in BUDGETS:
        P(f"Budget {budget} bits per weight with the scale")
        P("arm | bits | weight error by the formula, dB | " + " | ".join(f"mean KL on {m}: predicted, scored" for m in anc)
          + " | bits of a group's number | table entries | table, megabytes | look-ups per weight to read back | at the safe scale: no group over, dB | what the safe scale costs, dB")
        for r in sorted((x for x in rows if x["budget"] == budget), key=lambda x: x["err_db"]):
            cells = []
            for m, (kl, err, scored) in anc.items():
                got = scored.get(r["name"])
                cells.append(f"{F.predicted_kl(r['err_db'], kl, err):.4f}, " + (f"{got['kl_mean']:.4f}" if got else "-"))
            look = "-" if r["lookups"] is None else f"{r['lookups']:.1f}"
            P(f"{r['name']} | {r['bits']:.4f} | {r['err_db']:.2f} | " + " | ".join(cells)
              + f" | {r['width']} | {r['entries']} | {r['megabytes']:.2f} | {look} | {r['safe_bound_db']:.2f} | {r['safe_cost_db']:+.2f}")
        P("")
    P("The pick under both limits: the smallest error by the formula among the arms whose table is no larger than a limit")
    P("budget | table no larger than | arm | weight error, dB | table, megabytes | behind the best arm of the budget, dB")
    for budget in BUDGETS:
        mine = [x for x in rows if x["budget"] == budget]
        best = min(x["err_db"] for x in mine)
        for limit in (0.5, 2.0, 8.0, 32.0, float("inf")):
            ok = [x for x in mine if x["megabytes"] <= limit]
            if not ok:
                P(f"{budget} | {limit} MB | none | - | - | -")
                continue
            p = min(ok, key=lambda x: x["err_db"])
            P(f"{budget} | {'any' if math.isinf(limit) else f'{limit} MB'} | {p['name']} | {p['err_db']:.2f} | {p['megabytes']:.2f} | {p['err_db'] - best:+.2f}")
    P("")
    P("What each knob is worth by the formula, at 4.5 bits, in dB of weight error (and of KL, where the claim holds)")
    by = {x["name"]: x for x in rows if x["budget"] == 4.5}
    def gap(a, b):
        return by[a]["err_db"] - by[b]["err_db"]
    P(f"E8 for integer levels, same group and scale: {gap('EBG128:b560s16', 'ZBG128:b560s16'):+.2f} (128, 16-bit scale), {gap('EBG32:b136s8', 'ZBG32:b136s8'):+.2f} (32, 8-bit scale)")
    P(f"a scale of 8 bits for one of 16: {gap('EBG32:b136s8', 'EBG32:b128s16'):+.2f} (32), {gap('EBG64:b280s8', 'EBG64:b272s16'):+.2f} (64), {gap('EBG128:b568s8', 'EBG128:b560s16'):+.2f} (128)")
    P(f"a scale of 6 bits for one of 8: {gap('EBG32:b138s6', 'EBG32:b136s8'):+.2f} (32), {gap('EBG64:b282s6', 'EBG64:b280s8'):+.2f} (64), {gap('EBG128:b570s6', 'EBG128:b568s8'):+.2f} (128)")
    P(f"a group of 128 for one of 32, each with its best scale: {min(by[k]['err_db'] for k in by if k.startswith('EBG128')) - min(by[k]['err_db'] for k in by if k.startswith('EBG32')):+.2f} for E8, "
      f"{min(by[k]['err_db'] for k in by if k.startswith('ZBG128')) - min(by[k]['err_db'] for k in by if k.startswith('ZBG32')):+.2f} for integer levels")
    fl = F.floors(4.5)
    P("Floors at 4.5 bits, the scale not charged: " + "; ".join(f"{k} {v:.2f} dB" for k, v in fl.items()))
    P("")
    bad = [x["name"] for x in rows if x["read_back_ok"] is False]
    P(f"groups read back equal to what was numbered: {'all' if not bad else 'NOT for ' + ', '.join(bad)}. Tables not built, sized from their formula: "
      + (", ".join(x["name"] for x in rows if x["lookups"] is None) or "none"))
    print("\n".join(out))


if __name__ == "__main__":
    main()
