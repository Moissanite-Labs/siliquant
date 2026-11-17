#!/usr/bin/env python3
"""Extension 1: a rule with a proved guarantee, on the held-out matrices.

The plan is PREREG-extension-1.md in this folder. The rule and its two claims are in
shape/certified.py: a closed-form scale at which no block can fall outside the codebook, and a
bound on the error that holds for any weights. This program rounds every held-out matrix by
that rule with four codebooks (the integers and E8, cut by a ball and by a cross-polytope) in
the three cells of the budget, counts what the claims say cannot happen, and sets the rule
beside the searched rule of the main run, whose errors it reads from results.json.

It refuses to start unless the extension's plan and programs are committed as they stand.

Usage, from the root of a checkout, with NumPy:
  python3 experiments/shape-formula-held-out-2026-11-17/safe.py --whisper weights.npz --qwen model.safetensors > safe.out
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import get_context
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "shape"))
import run  # noqa: E402
import certified as C  # noqa: E402
S = run.S

ARMS = (("Z8", 1), ("Z8", 2), ("E8", 1), ("E8", 2))
REAL = ("whisper", "qwen")
GATED = ("experiments/shape-formula-held-out-2026-11-17/PREREG-extension-1.md", "experiments/shape-formula-held-out-2026-11-17/safe.py",
         "shape/certified.py", "shape/shapelib.py")


def db(x):
    return 10.0 * math.log10(x)


def one_matrix(task):
    W = np.ascontiguousarray(run.load(task), dtype=np.float32).astype(np.float64)
    n = W.shape[1]
    rows = []
    for g, cap, rate, budget in run.cells_for(n):
        if not budget:
            continue
        gg = g if g == "R" else int(g)
        size = n if g == "R" else int(g)
        for lattice, p in ARMS:
            code = S.Codebook(lattice, S.Shape(p), cap)
            r = C.safe_round(W, code, gg)
            rows.append({"matrix": task["matrix"], "family": task["family"], "arm": f"{lattice}:{code.shape.name}", "lattice": lattice,
                         "group": g, "size": size, "codewords": code.N, "bits": code.bits + 16.0 / size,
                         "r_eff": code.r_eff, "safe_radius": C.safe_radius(code),
                         "predicted_cost_db": 20.0 * math.log10(code.r_eff / C.safe_radius(code)) - db(1.0 + 2.0 / size), **r})
    return rows


def verdicts(rows, main):
    """The five predictions of the extension. main maps (matrix, arm, group) to a row of the main run's budget cells."""
    out = []

    def add(tag, holds, text):
        out.append(f"{tag} | {'holds' if holds else 'FAILS'} | {text}")

    real = [r for r in rows if r["family"] in REAL]
    # E1: what the two claims say cannot happen
    outside = sum(r["outside"] > 0 for r in rows)
    block = max(r["worst_block_over_its_bound"] for r in rows)
    over = sum(r["meas"] > r["bound"] for r in rows)
    add("E1", outside == 0 and block <= 1.0 + 1e-9 and over == 0,
        f"{len(rows)} runs: a block outside the codebook in {outside} (claim 1 says none). The worst block has {block:.4f} of its bound (claim 2 says at most 1). "
        f"The matrix's error is over its bound in {over} (claim 2 says none). The bound is above the measured error by {np.mean([db(r['bound'] / r['meas']) for r in rows if r['lattice'] == 'E8']):.2f} dB in the mean for E8 "
        f"and {np.mean([db(r['bound'] / r['meas']) for r in rows if r['lattice'] == 'Z8']):.2f} dB for the integers")
    # E2: what the guarantee costs against the searched rule
    parts, ok = [], True
    for arm in ("Z8:p1", "Z8:p2", "E8:p1", "E8:p2"):
        for g in ("32", "128", "R"):
            rs = [r for r in real if r["arm"] == arm and r["group"] == g]
            cost = float(np.mean([db(r["meas"] / main[(r["matrix"], arm, g)]["meas"]) for r in rs]))
            pred = float(np.mean([r["predicted_cost_db"] for r in rs]))
            ok &= abs(cost - pred) <= 0.15
            parts.append(f"{arm}@{g}: {cost:+.2f} against {pred:+.2f}")
    add("E2", ok, "real matrices, error of the safe rule over that of the searched rule, in dB, measured against predicted (mark: within 0.15 in each of 12). " + "; ".join(parts))
    # E3: the estimate
    res = np.array([db(r["meas"] / r["estimate_with_cell_at_zero"]) for r in real])
    add("E3", float(np.abs(res).mean()) <= 0.10 and float(np.mean(np.abs(res) <= 0.3)) >= 0.90,
        f"real matrices, measured error over the estimate: mean size {np.abs(res).mean():.3f} dB (mark 0.10), within 0.3 dB on {np.mean(np.abs(res) <= 0.3):.3f} (mark 0.90), "
        f"mean {res.mean():+.3f}, min {res.min():+.2f}, max {res.max():+.2f}, {len(res)} rows")
    # E4: the pick by the bound and by the estimate
    by = defaultdict(list)
    for r in rows:
        by[r["matrix"]].append(r)
    same = e8 = 0
    regret = []
    for m, rs in by.items():
        b = min(rs, key=lambda r: r["bound"])
        e = min(rs, key=lambda r: r["estimate"])
        same += (b["arm"], b["group"]) == (e["arm"], e["group"])
        e8 += b["lattice"] == "E8"
        if rs[0]["family"] in REAL:
            regret.append(db(b["meas"] / min(r["meas"] for r in rs)))
    add("E4", same == len(by) and e8 == len(by) and float(np.mean(regret)) <= 0.05,
        f"{len(by)} matrices, 12 safe arms each: the arm with the smallest bound is the arm with the smallest estimate on {same}, and an E8 arm on {e8} (predicted: all, both). "
        f"On the real matrices its error is over the smallest measured among the 12 by {np.mean(regret):.4f} dB in the mean (mark 0.05), at most {max(regret):.2f}")
    # E5: against the scalar rule in the same budget
    gains, ahead = [], 0
    for m, rs in by.items():
        if rs[0]["family"] not in REAL:
            continue
        x = min(main[(m, "X", g)]["meas"] for g in ("32", "128", "R"))
        s = min(r["meas"] for r in rs)
        gains.append(db(x / s))
        ahead += s < x
    add("E5", ahead / len(gains) >= 0.90 and 1.0 <= float(np.mean(gains)) <= 3.5,
        f"real matrices at 4.5 bits: the best safe arm is ahead of the best scalar rule on {ahead} of {len(gains)} (mark 0.90), by {np.mean(gains):.2f} dB in the mean (predicted 1.0 to 3.5), "
        f"from {min(gains):+.2f} to {max(gains):+.2f}")
    return out


def summarise(rows, main, meta):
    out = []
    P = out.append
    P(f"Extension 1. Matrices: {meta['matrices']}. Runs: {len(rows)}. Seconds: {meta['seconds']}. Commit of the plan: {meta['gate']['commit']}.")
    P("")
    P("1. The safe rule beside the searched rule of the main run, in the three cells of the budget. dB.")
    P("   cost = error of the safe rule over that of the searched rule with the same codebook. gain over X = the scalar rule's error in the same cell over the safe rule's.")
    P("   family | arm | group | safe rule dB | cost: mean (min, max) | predicted cost | gain over X: mean | measured over estimate: mean (min, max) | bound over measured: mean | matrices")
    fams = []
    for r in rows:
        if r["family"] not in fams:
            fams.append(r["family"])
    for f in fams:
        for arm in ("Z8:p1", "Z8:p2", "E8:p1", "E8:p2"):
            for g in ("32", "128", "R"):
                rs = [r for r in rows if r["family"] == f and r["arm"] == arm and r["group"] == g]
                if not rs:
                    continue
                cost = [db(r["meas"] / main[(r["matrix"], arm, g)]["meas"]) for r in rs]
                gx = [db(main[(r["matrix"], "X", g)]["meas"] / r["meas"]) for r in rs]
                est = [db(r["meas"] / r["estimate_with_cell_at_zero"]) for r in rs]
                P(f"   {f:>7} | {arm} | {g:>3} | {np.mean([db(r['meas']) for r in rs]):.2f} | {np.mean(cost):+.2f} ({min(cost):+.2f}, {max(cost):+.2f}) | {np.mean([r['predicted_cost_db'] for r in rs]):+.2f} | "
                  f"{np.mean(gx):+.2f} | {np.mean(est):+.2f} ({min(est):+.2f}, {max(est):+.2f}) | {np.mean([db(r['bound'] / r['meas']) for r in rs]):.2f} | {len(rs)}")
    P("")
    P("2. The best safe arm of each matrix among the 12, and what the guarantee costs at the level of the pick")
    P("   family | best safe arm, measured: arm@group and count | behind the best searched arm with the same four codebooks: mean (min, max) | behind the best of all 39 arms of the main run: mean | ahead of the best scalar rule: mean | matrices")
    by = defaultdict(list)
    for r in rows:
        by[r["matrix"]].append(r)
    for f in fams:
        best, behind4, behind39, overx = [], [], [], []
        for m, rs in by.items():
            if rs[0]["family"] != f:
                continue
            b = min(rs, key=lambda r: r["meas"])
            best.append(f"{b['arm']}@{b['group']}")
            four = min(main[(m, r["arm"], r["group"])]["meas"] for r in rs)
            every = min(v["meas"] for k, v in main.items() if k[0] == m)
            behind4.append(db(b["meas"] / four))
            behind39.append(db(b["meas"] / every))
            overx.append(db(min(main[(m, "X", g)]["meas"] for g in ("32", "128", "R")) / b["meas"]))
        P(f"   {f:>7} | " + ", ".join(f"{a} {n}" for a, n in Counter(best).most_common()) + f" | {np.mean(behind4):+.2f} ({min(behind4):+.2f}, {max(behind4):+.2f}) | {np.mean(behind39):+.2f} | {np.mean(overx):+.2f} | {len(best)}")
    P("")
    P("3. The predictions of the extension")
    for line in verdicts(rows, main):
        P("   " + line)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--whisper")
    ap.add_argument("--qwen")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--smoke", action="store_true", help="run the code path on one small matrix that is not in the study and print nothing but ok")
    args = ap.parse_args()
    t0 = time.time()
    if args.smoke:
        rows = one_matrix({"matrix": "smoke/T3", "family": "smoke"})
        ok = all(r["outside"] == 0 and r["worst_block_over_its_bound"] <= 1 + 1e-9 and r["meas"] <= r["bound"] for r in rows) and len(rows) == 12
        print("ok" if ok else "FAIL", f"({len(rows)} runs, {time.time() - t0:.0f} s)")
        sys.exit(0 if ok else 1)
    if not (args.whisper and args.qwen):
        raise SystemExit("FAIL: the extension needs both weight files")
    run.GATED = GATED
    gate = run.plan_gate()
    if not gate["commit_is_on_the_remote"]:
        raise SystemExit("FAIL: the commit that holds the extension's plan is not on the remote yet. Push it first.")
    run._files["whisper"], run._files["qwen"] = args.whisper, args.qwen
    tasks = run.tasks_for("held", args.whisper, args.qwen)
    for cap in run.all_caps({2048, 512, 896, 4864}):
        for lattice, p in ARMS:
            S.Codebook(lattice, S.Shape(p), cap)
    with get_context("fork").Pool(args.workers) as pool:
        results = pool.map(one_matrix, tasks, chunksize=1)
    rows = [r for rs in results for r in rs]
    meta = {"gate": gate, "matrices": len(tasks), "seconds": round(time.time() - t0)}
    (HERE / "safe.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=1) + "\n")
    data = json.loads((HERE / "results.json").read_text())
    main_rows = {(r["matrix"], r["arm"], r["group"]): r for r in data["rows"] if r["kind"] in ("cut", "scalar") and r["budget"]}
    print("\n".join(summarise(rows, main_rows, meta)))


if __name__ == "__main__":
    main()
