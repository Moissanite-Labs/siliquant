#!/usr/bin/env python3
"""Larger values of t for the candidates on the incentre lattice. Written after the main run was read. Not planned.

The main run searched t over 0.4 to 1.0 and the search ended at 1.0 in most cases. This program
searches t over 1.1 to 2.0 in steps of 0.1 and 2.2 to 4.0 in steps of 0.2 with the same code, and
sets the best of all the values tried against the reference mixes of the main run. It reads
results.json of the main run. A first version stopped at 2.0, and on the heavy-tailed family the
search then ended at 2.0, so the range was widened once more.

For the unbounded point set, t and 2 - t give the same points: 2m + t e_i is 2(m + e_i) - (2 - t) e_i.
The two differ only at the edge of a finite codebook, where a larger t reaches further out.

Usage, from the root of a checkout:
  python3 experiments/incentre-vortex-codebooks-2026-11-17/larger_t.py --weights /path/to/weights.npz > larger_t.out
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("codebooks", HERE / "run.py")
run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run)
run.T_GRID = tuple(round(1.1 + 0.1 * i, 1) for i in range(10)) + tuple(round(2.2 + 0.2 * i, 1) for i in range(10))     # 1.1 .. 2.0, 2.2 .. 4.0


def work(task):
    return run.run_task(task)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    main_rows = json.loads((HERE / "results.json").read_text())["rows"]
    by = {(r["matrix"], r["arm"], r["param"]): r for r in main_rows}
    names = sorted({r["matrix"] for r in main_rows}, key=lambda m: (("G", "T", "real").index(m.split("/")[0]), m))
    if any(m.startswith("real/") for m in names):
        if not args.weights:
            raise SystemExit("the main run used the real matrices: give --weights")
        run._weights_path = args.weights
        run.real_names()                       # checks the SHA-256
    tasks = [("candidates", (m, f[0])) for m in names for f in run.FAMILIES]
    with Pool(args.workers) as pool:
        new = [r for part in pool.imap(work, tasks, chunksize=1) for r in part if r["arm"].endswith("t")]
    (HERE / "larger_t.json").write_text(json.dumps(new, indent=1) + "\n")
    big = {(r["matrix"], r["arm"], r["param"]): r for r in new}

    def curve(m, arm):
        return {q: by[(m, arm, q)]["rel_err"] for q in run.Q_FOR[m.split("/")[0]]}

    rng = np.random.default_rng(12)
    worst, cases = 0.0, 0
    for n, K, cab in ((2, 2, False), (2, 3, False), (3, 2, False), (3, 3, False), (2, 3, True), (3, 2, True), (3, 3, True)):
        for t in (1.3, 1.7, 2.0, 3.4, 4.0):
            cb = run.f_codebook(n, K, t, cab)
            x = rng.normal(size=(4000, n)) * rng.choice([0.4, 1.0, 2.5, 6.0], size=(4000, 1))
            fast = ((x - run.f_round(x, K, t, cab=cab, centred=False)) ** 2).sum(axis=1)
            slow = ((x - run.L.nearest_in_codebook(x, cb)[1]) ** 2).sum(axis=1)
            worst = max(worst, float(np.max(np.abs(fast - slow))))
            cases += 1
    print(f"Check: for t from 1.3 to 4.0 the fast rule finds the nearest codeword of the whole codebook: {cases} codebooks, 4000 blocks each, largest difference in squared distance {worst:.1e}")
    if worst > 1e-9:
        raise SystemExit("FAIL: the fast rule is not exact at these values of t")
    print()
    print("Values of t from 1.1 to 4.0, and the better of the two searches (0.4 to 1.0 in the main run, 1.1 to 4.0 here) against the mixes of the main run.")
    print("Gap = candidate minus reference in dB. A positive gap means the candidate has the larger error.")
    print("family | arm | K | bits | best t in 1.1..4.0 | its dB | dB at the best t in 0.4..1.0 | matrices where the larger t is better | better of the two: gap to S mix: mean (min, max) | worse than S mix | gap to E8 mix: mean (min, max) | worse than E8 mix | matrices")
    for f in ("G", "T", "real"):
        ms = [m for m in names if m.split("/")[0] == f]
        for fam, n, cab in run.FAMILIES:
            arm = fam + "t"
            for K in run.K_FOR[f]:
                bits = run.f_bits(n, K, cab)
                small = [by[(m, arm, K)]["rel_err"] for m in ms]
                large = [big[(m, arm, K)]["rel_err"] for m in ms]
                best = [min(a, b) for a, b in zip(small, large)]
                gs = [run.db(e / run.mix_error(curve(m, "S"), bits)) for m, e in zip(ms, best)]
                ge = [run.db(e / run.mix_error(curve(m, "E8"), bits)) for m, e in zip(ms, best)]
                ts = sorted({big[(m, arm, K)]["t"] for m in ms})
                tt = f"{ts[0]:.1f}" if len(ts) == 1 else f"{ts[0]:.1f}-{ts[-1]:.1f}"
                print(f"{f:>4} | {arm:<6} | {K:>2} | {bits:.4f} | {tt:>7} | {np.mean([run.db(e) for e in large]):.2f} | {np.mean([run.db(e) for e in small]):.2f} | {sum(b < a for a, b in zip(small, large))} | "
                      f"{np.mean(gs):+.2f} ({min(gs):+.2f}, {max(gs):+.2f}) | {sum(g > 0 for g in gs)} | {np.mean(ge):+.2f} ({min(ge):+.2f}, {max(ge):+.2f}) | {sum(g > 0 for g in ge)} | {len(ms)}")
    top = [(r["matrix"], r["arm"], r["param"]) for r in new if r["t"] == run.T_GRID[-1]]
    print(f"searches that ended at t = {run.T_GRID[-1]}: {len(top)} of {len(new)}")


if __name__ == "__main__":
    main()
