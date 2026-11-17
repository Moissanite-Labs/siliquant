#!/usr/bin/env python3
"""The pick of the budget in exact arithmetic, with a certificate, from the receipts of pilot.json or results.json.

Not in the plan. Written while the run of the plan was in progress, after the owner asked for
formulas that can be verified. It reads the receipts the run wrote and rounds nothing.

For the five arms whose formula is a fraction times a power of pi (the scalar rule, and the
integers and E8 cut by a ball or a cross-polytope), at the three group sizes of the budget,
shape/certified.py picks the arm with the smallest predicted error at level 2 in whole-number
arithmetic. shape/check_certificate.py, which shares no code with it, confirms each pick.

The chain: check_receipts.py replays the receipts from the weights. This program picks from the
receipts. check_certificate.py confirms the picks from the same receipts.

Usage, from the root of a checkout, with NumPy:
  python3 experiments/shape-formula-held-out-2026-11-17/certify.py results > certified.out
  python3 shape/check_certificate.py experiments/shape-formula-held-out-2026-11-17/results.certificates.json
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "shape"))
import certified as C  # noqa: E402
import check_certificate as K  # noqa: E402

BUDGET = Fraction(9, 2)
EXACT_ARMS = ("X", "Z8:p1", "Z8:p2", "E8:p1", "E8:p2")


def db(x):
    return 10.0 * math.log10(x)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "results"
    data = json.loads((HERE / f"{name}.json").read_text())
    rows = data["rows"]
    probe = {r["matrix"]: r for r in rows if r["kind"] == "probe"}
    arms = [r for r in rows if r["kind"] in ("cut", "scalar") and r["budget"]]
    by_matrix = {}
    for r in arms:
        by_matrix.setdefault(r["matrix"], []).append(r)
    certs, lines = {}, {}
    for mname, rs in by_matrix.items():
        receipt = probe[mname]["receipt"]
        exact = [r for r in rs if r["arm"] in EXACT_ARMS]
        cands = [{"arm": r["arm"], "size": r["size"], "codewords": (r["q"] ** 8 if r["arm"] == "X" else r["codewords"]),
                  "terms": {k: int(v) for k, v in receipt[r["group"]].items()}} for r in exact]
        best, cert = C.pick(cands, budget=BUDGET)
        certs[mname] = cert
        meas = [r["meas"] for r in exact]
        f2 = int(np.argmin([r["L2"] for r in exact]))
        f3 = int(np.argmin([r["L3"] for r in exact]))
        r, b = C.fourth_power(cands[best]["arm"], cands[best]["size"], cands[best]["codewords"], cands[best]["terms"])
        value = (float(r) * math.pi ** (4 * b)) ** 0.25
        lines[mname] = {"family": probe[mname]["family"], "pick": f"{exact[best]['arm']}@{exact[best]['group']}",
                        "same_as_float_level_2": best == f2, "same_as_float_level_3": best == f3,
                        "exact_over_float_level_2": value / exact[best]["L2"],
                        "regret_among_exact_arms": db(meas[best] / min(meas)),
                        "regret_among_all_arms": db(meas[best] / min(x["meas"] for x in rs)),
                        "is_measured_best_of_exact_arms": meas[best] == min(meas),
                        "measured_over_exact_formula": db(meas[best] / value)}
    (HERE / f"{name}.certificates.json").write_text(json.dumps(certs, indent=1) + "\n")
    failed = {m: K.check(c) for m, c in certs.items()}
    failed = {m: b for m, b in failed.items() if b}
    out = []
    P = out.append
    P(f"Set: {data['meta']['set']}. The pick among {len(EXACT_ARMS)} arms at 3 group sizes, 4.5 bits per weight with the scales, in exact arithmetic at level 2.")
    P(f"certificates written: {len(certs)}. Confirmed by shape/check_certificate.py: {len(certs) - len(failed)}. Identities for any 8 numbers: {'hold' if K.identities() else 'FAIL'}. Bounds on pi: {'hold' if K.pi_bounds_hold() else 'FAIL'}.")
    worst = max(abs(v["exact_over_float_level_2"] - 1.0) for v in lines.values())
    P(f"the exact value of the pick against the 64-bit floats of shapelib at level 2: largest relative difference {worst:.1e}")
    P("")
    P("family | matrices | the pick: arm@group and count | same as the float pick at level 2 | at level 3 | the pick is the measured best of the 15 | regret among the 15: mean, max | regret among all 39 arms: mean, max | measured over the exact formula for the pick: mean (min, max)")
    fams = []
    for v in lines.values():
        if v["family"] not in fams:
            fams.append(v["family"])
    for f in fams + ["all"]:
        vs = [v for v in lines.values() if f == "all" or v["family"] == f]
        picks = ", ".join(f"{a} {n}" for a, n in Counter(v["pick"] for v in vs).most_common())
        r15 = [v["regret_among_exact_arms"] for v in vs]
        r39 = [v["regret_among_all_arms"] for v in vs]
        res = [v["measured_over_exact_formula"] for v in vs]
        P(f"{f:>7} | {len(vs)} | {picks} | {sum(v['same_as_float_level_2'] for v in vs)} | {sum(v['same_as_float_level_3'] for v in vs)} | {sum(v['is_measured_best_of_exact_arms'] for v in vs)} | "
          f"{np.mean(r15):.3f}, {max(r15):.2f} | {np.mean(r39):.3f}, {max(r39):.2f} | {np.mean(res):+.2f} ({min(res):+.2f}, {max(res):+.2f})")
    print("\n".join(out))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
