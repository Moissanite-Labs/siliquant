#!/usr/bin/env python3
"""Tables and a verdict on each prediction of PREREG.md, from the committed result files. Standard library only.

Reads, from this folder: results.jsonl, allocate.out, allocate_joined.out, measures.jsonl, weights_only.jsonl.
Reads, from the first model-output record: results.jsonl (KL of the 26 variants of round 5 and of the
llama.cpp anchors) and spectral.jsonl. A prediction whose files are missing is printed as NOT RUN.
Each threshold in the verdict table is the plan's, word for word, with the measured value beside it.

Usage, from the root of a checkout:
  python3 experiments/output-budget-gemma-e4b-2026-11-17/summary.py > experiments/output-budget-gemma-e4b-2026-11-17/results.out
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
EARLIER = HERE.parents[0] / "model-output-gemma-e4b-2026-11-15"
KINDS = ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down")
WRITERS = ("ffn_down", "attn_output")
LAYERS6 = (0, 8, 16, 24, 32, 40)


def _spearman():
    spec = importlib.util.spec_from_file_location("earlier_summary", EARLIER / "summary.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.spearman        # the rank correlation of the first record's summary


spearman = _spearman()


def read_jsonl(path):
    p = Path(path)
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()] if p.exists() else []


def scored(path):
    rows = {}
    for r in read_jsonl(path):
        if "kl_mean" in r and r["variant"] not in rows:       # the first scored line of a variant is the one used
            rows[r["variant"]] = r
    return rows


ROWS = scored(HERE / "results.jsonl")
OLD = scored(EARLIER / "results.jsonl")
VERDICTS = []


def kl(v, rows=None):
    return (rows or ROWS)[v]["kl_mean"]


def have(*vs):
    return all(v in ROWS for v in vs)


def verdict(name, prediction, threshold, measured, ok, note=""):
    """ok True holds, False fails, None not run or not decided."""
    VERDICTS.append((name, prediction, threshold, measured, ok, note))


HEAD = "   variant | bits per weight (changed) | bits per weight (all block weights) | weight error, mean dB | mean KL +- se | median KL | top token kept, % | perplexity | kept entries"


def line(v, name=None):
    r = ROWS[v]
    f = lambda x, nd=3: "-" if x is None else f"{x:.{nd}f}"
    return (f"   {name or v} | {f(r.get('bits_per_weight_changed'))} | {f(r.get('bits_per_weight_all'))} | {f(r.get('weight_err_db_mean'), 2)} | "
            f"{r['kl_mean']:.6f} +- {r['kl_mean_se']:.6f} | {r['kl_median']:.6f} | {r['same_top']:.2f} | {r['ppl']:.3f} | {r.get('kept_entries', '-')}")


def table(vs):
    print(HEAD)
    for v in vs:
        if v in ROWS:
            print(line(v))
        else:
            print(f"   {v} | not scored")


def main():
    # ----- start
    starts = read_jsonl(HERE / "results.start")
    if starts:
        s = starts[0]
        print(f"Run started {s['started']}, plan commit {s['plan_commit']}, programs at {s['head']} (on the remote: {s['commit_is_on_the_remote']}), model SHA-256 {s['model_sha256']}")
        print(f"   {len(starts)} invocation(s) of run.py recorded in results.start")
    print()

    # ----- first step
    print("First step: the harness repeats itself")
    first = (("unchanged", 0.0, 0.00001, "under 0.00001"), ("X32:16", 0.117297, 0.002, "within 0.002"), ("XR:16", 0.448262, 0.006, "within 0.006"))
    ok_first = True
    for v, rec, lim, text in first:
        if v in ROWS:
            x = kl(v)
            good = x < lim if v == "unchanged" else abs(x - rec) <= lim
            ok_first = ok_first and good
            print(f"   {v}: recorded {rec:.6f}, now {x:.6f}, difference {x - rec:+.6f}, passes if {text}: {'PASS' if good else 'FAIL'}")
        else:
            ok_first = False
            print(f"   {v}: not scored")
    print(f"   first step: {'PASS' if ok_first else 'NOT PASSED'}")
    print()

    # ----- part A
    print("Part A: one kind at a time under the rule that clips nothing (X32:Q@KIND)")
    table([f"X32:{q}@{k}" for k in KINDS for q in (8, 16, 32)] + ["X32:8", "X32:16", "X32:32"])
    a_ok = all(f"X32:{q}@{k}" in ROWS for k in KINDS for q in (8, 16, 32))
    if a_ok:
        print("   per weight (KL divided by the kind's weights, times 1e9), by kind at Q = 16:")
        per = {k: kl(f"X32:16@{k}") / ROWS[f"X32:16@{k}"]["weights_changed"] for k in KINDS}
        for k in sorted(KINDS, key=lambda k: -per[k]):
            print(f"   {k} | {ROWS[f'X32:16@{k}']['weights_changed']} weights | {per[k] * 1e9:.4f}")
        top = {q: max(KINDS, key=lambda k: kl(f"X32:{q}@{k}")) for q in (8, 16, 32)}
        verdict("A1", "ffn_down has the largest KL of the seven kinds at each Q", "another kind is larger at any Q fails",
                "largest at Q = 8, 16, 32: " + ", ".join(top[q] for q in (8, 16, 32)), all(top[q] == "ffn_down" for q in (8, 16, 32)))
        two = sorted(KINDS, key=lambda k: -per[k])[:2]
        verdict("A2", "per weight, ffn_down and attn_output are the two most sensitive at Q = 16", "either is not among the top two fails",
                "top two per weight: " + ", ".join(two), set(two) == set(WRITERS))
        ratios = [(k, lo, kl(f"X32:{lo}@{k}") / kl(f"X32:{2 * lo}@{k}")) for k in KINDS for lo in (8, 16)]
        inside = sum(2.5 <= r <= 6.0 for _, _, r in ratios)
        print("   KL divided by the KL one bit up (8 to 16, 16 to 32): " + "; ".join(f"{k} {lo}->{2 * lo} {r:.2f}" for k, lo, r in ratios))
        verdict("A3", "one more bit divides the KL of a kind by 2.5 to 6; at least 12 of the 14 steps lie in that range", "fewer than 12 fails",
                f"{inside} of 14 in 2.5 to 6", inside >= 12)
        if "X32:16" in ROWS:
            total = sum(kl(f"X32:16@{k}") for k in KINDS)
            print(f"   at Q = 16 the seven KLs add up to {total:.6f}; X32:16 on all kinds is {kl('X32:16'):.6f}")
            verdict("A4", "at Q = 16 the sum of the seven KLs is 0.75 to 1.25 times the KL of X32:16 on all kinds (0.1173)", "ratio outside that range fails",
                    f"ratio {total / kl('X32:16'):.3f}", 0.75 <= total / kl("X32:16") <= 1.25)
        else:
            verdict("A4", "sum of the seven KLs at Q = 16 is 0.75 to 1.25 times the KL of X32:16", "ratio outside fails", "X32:16 not scored", None)
    else:
        for n in ("A1", "A2", "A3", "A4"):
            verdict(n, "Part A", "-", "Part A not complete", None)
    print()

    # ----- part B
    print("Part B: bits by equal gain (the arms of allocate.out)")
    alloc = {}
    for ln in (HERE / "allocate.out").read_text().splitlines() if (HERE / "allocate.out").exists() else []:
        if ln.startswith("JSON "):
            r = json.loads(ln[5:])
            alloc[(r["budget"], r["role"])] = r
    if alloc:
        print("   arm | levels | mean bits | predicted KL | measured KL +- se | measured over predicted | top token kept, %")
        for key in sorted(alloc, key=lambda k: (-k[0], k[1])):
            r = alloc[key]
            lv = " ".join(f"{k}={r['levels'][k]}" for k in KINDS)
            if r["variant"] in ROWS:
                m = ROWS[r["variant"]]
                print(f"   {key[1]} at {key[0]} | {lv} | {m['bits_per_weight_all']:.4f} | {r['predicted_kl']:.6f} | {m['kl_mean']:.6f} +- {m['kl_mean_se']:.6f} | {m['kl_mean'] / r['predicted_kl']:.3f} | {m['same_top']:.2f}")
            else:
                print(f"   {key[1]} at {key[0]} | {lv} | {r['mean_bits']:.4f} | {r['predicted_kl']:.6f} | not scored")
        print(f"   VE832:16 (4.500 bits, first record): {kl('VE832:16', OLD):.4f}" if "VE832:16" in OLD else "")
        q4 = next((v for v in OLD if v.startswith("file:") and "Q4_K_M" in v), None)
        if q4:
            print(f"   llama.cpp Q4_K_M ({OLD[q4]['bits_per_weight']:.3f} bits, first record): {kl(q4, OLD):.4f}")
    b4, b3 = alloc.get((4.5, "best")), alloc.get((3.5, "best"))
    r4, r3 = alloc.get((4.5, "reversed")), alloc.get((3.5, "reversed"))
    scored_b = all(x is not None and x["variant"] in ROWS for x in (b4, b3, r4, r3))
    if b4 and b4["variant"] in ROWS:
        verdict("B1", "at 4.5 bits the best arm has a KL of 0.0997 or less (15 percent under X32:16)", "it is above 0.0997 fails",
                f"{kl(b4['variant']):.4f}", kl(b4["variant"]) <= 0.0997)
    else:
        verdict("B1", "best arm at 4.5 bits, KL 0.0997 or less", "above 0.0997 fails", "not scored", None)
    if b3 and b3["variant"] in ROWS and have("X32:8"):
        verdict("B2", "at 3.5 bits the best arm has a KL at least 15 percent under that of X32:8", "it is not fails",
                f"{kl(b3['variant']):.4f} against 0.85 x X32:8 = {0.85 * kl('X32:8'):.4f}", kl(b3["variant"]) <= 0.85 * kl("X32:8"))
    else:
        verdict("B2", "best arm at 3.5 bits at least 15 percent under X32:8", "not 15 percent under fails", "not scored", None)
    if b4 and b3 and b4["variant"] in ROWS and b3["variant"] in ROWS:
        rr = [kl(b["variant"]) / b["predicted_kl"] for b in (b4, b3)]
        verdict("B3", "the measured KL of each best arm is 0.7 to 1.3 times the KL the program predicted", "either outside fails",
                f"{rr[0]:.3f} at 4.5 bits, {rr[1]:.3f} at 3.5 bits", all(0.7 <= x <= 1.3 for x in rr))
    else:
        verdict("B3", "measured over predicted KL of each best arm, 0.7 to 1.3", "either outside fails", "not scored", None)
    if r4 and r3 and r4["variant"] in ROWS and r3["variant"] in ROWS and have("X32:16", "X32:8"):
        rr = [kl(r4["variant"]) / kl("X32:16"), kl(r3["variant"]) / kl("X32:8")]
        verdict("B4", "each reversed arm has a KL at least 15 percent above the uniform arm at its budget", "either is not fails",
                f"{rr[0]:.3f} x X32:16 at 4.5 bits, {rr[1]:.3f} x X32:8 at 3.5 bits (need 1.15 or more)", all(x >= 1.15 for x in rr))
    else:
        verdict("B4", "each reversed arm at least 15 percent above the uniform arm", "either is not fails", "not scored", None)
    print()

    # ----- part C
    print("Part C: the largest entries kept apart")
    cv = [f"XRo:{q}~{L}" for q in (16, 32) for L in ("3.45", "5", "8")] + ["XRo:16~2.5", "X32o:16~3.45"]
    table(["XR:16", "X32:16"] + cv)
    for v in cv:
        if v in ROWS:
            r = ROWS[v]
            print(f"   {v}: kept {r['kept_entries']} entries, {100 * r['share_kept']:.4f} % of the weights; bits per weight {r['bits_per_weight']:.4f}")
    xr32 = OLD.get("XR:32")
    if have("XRo:16~3.45", "XR:16"):
        d_bits = ROWS["XRo:16~3.45"]["bits_per_weight"] - ROWS["XR:16"]["bits_per_weight"]
        verdict("C1", "XRo:16~3.45 has a KL of 0.336 or less and costs under 0.06 bit more than XR:16", "either part fails",
                f"KL {kl('XRo:16~3.45'):.4f}, {d_bits:+.4f} bit", kl("XRo:16~3.45") <= 0.336 and d_bits < 0.06)
        verdict("C3", "XRo:16~3.45 stays above the KL of X32:16 (0.1173)", "at or under 0.1173 fails",
                f"{kl('XRo:16~3.45'):.4f}", kl("XRo:16~3.45") > 0.1173)
    else:
        verdict("C1", "XRo:16~3.45 KL 0.336 or less, under 0.06 bit more", "either part fails", "not scored", None)
        verdict("C3", "XRo:16~3.45 above 0.1173", "at or under fails", "not scored", None)
    if have("XRo:32~3.45"):
        verdict("C2", "XRo:32~3.45 has a KL of 0.0885 or less", "it is above fails", f"{kl('XRo:32~3.45'):.4f}", kl("XRo:32~3.45") <= 0.0885)
    else:
        verdict("C2", "XRo:32~3.45 KL 0.0885 or less", "above fails", "not scored", None)
    if have("XRo:16~8", "XR:16", "XRo:32~8") and xr32:
        r16, r32 = kl("XRo:16~8") / kl("XR:16"), kl("XRo:32~8") / kl("XR:32", OLD)
        verdict("C4", "XRo:Q~8 is within 10 percent of XR:Q at both Q", "either is not fails",
                f"Q = 16: {r16:.3f} (XR:16 of this record); Q = 32: {r32:.3f} (XR:32 {kl('XR:32', OLD):.4f} from the first record, not scored again)",
                abs(r16 - 1) <= 0.10 and abs(r32 - 1) <= 0.10)
    else:
        verdict("C4", "XRo:Q~8 within 10 percent of XR:Q at both Q", "either is not fails", "not scored", None)
    if have("X32o:16~3.45", "X32:16"):
        x = kl("X32o:16~3.45") / kl("X32:16")
        verdict("C5", "X32o:16~3.45 is 5 to 30 percent under X32:16", "outside that range fails", f"ratio {x:.3f} (0.70 to 0.95 holds)", 0.70 <= x <= 0.95)
    else:
        verdict("C5", "X32o:16~3.45 5 to 30 percent under X32:16", "outside fails", "not scored", None)
    print()

    # ----- part D
    print("Part D: the bits of a scale")
    table(["X32:16", "X32s8:16", "X32s6:16", "X32s8:18", "X32s6:20"])
    if have("X32:16", "X32s8:16"):
        x = kl("X32s8:16") / kl("X32:16")
        verdict("D1", "X32s8:16, at 4.25 bits, is within 10 percent of X32:16", "it is not fails", f"ratio {x:.3f}", abs(x - 1) <= 0.10)
    else:
        verdict("D1", "X32s8:16 within 10 percent of X32:16", "it is not fails", "not scored", None)
    if have("X32:16", "X32s6:16"):
        x = kl("X32s6:16") / kl("X32:16")
        verdict("D2", "X32s6:16, at 4.19 bits, is at most 1.2 times X32:16", "it is above fails", f"ratio {x:.3f}", x <= 1.2)
    else:
        verdict("D2", "X32s6:16 at most 1.2 times X32:16", "above fails", "not scored", None)
    if have("X32:16", "X32s8:18", "X32s6:20"):
        x, y = kl("X32s8:18") / kl("X32:16"), kl("X32s6:20") / kl("X32:16")
        verdict("D3", "X32s8:18, at 4.42 bits, has a KL at most 0.90 of X32:16, and X32s6:20, at 4.51 bits, at most 0.80", "either is above fails",
                f"X32s8:18 {x:.3f}, X32s6:20 {y:.3f}", x <= 0.90 and y <= 0.80)
    else:
        verdict("D3", "X32s8:18 at most 0.90 and X32s6:20 at most 0.80 of X32:16", "either above fails", "not scored", None)
    print()

    # ----- part E
    print("Part E: does the damage add up over layers (ffn_down, layers 0, 8, 16, 24, 32, 40)")
    for arm, label, name in (("clip:5", "clip at 5 standard deviations", "E1"), ("X32:16", "rounding X32:16", "E2")):
        singles = [f"{arm}@ffn_down#{il}" for il in LAYERS6]
        allv = f"{arm}@ffn_down#" + ",".join(str(i) for i in LAYERS6)
        table(singles + [allv])
        want = "at least 1.5 times the sum of the six alone" if name == "E1" else "0.75 to 1.25 times the sum of the six alone"
        if have(*singles, allv):
            tot = sum(kl(v) for v in singles)
            weak = [v for v in singles if kl(v) < 5 * ROWS[v]["kl_mean_se"]]
            x = kl(allv) / tot
            print(f"   {label}: the six alone add up to {tot:.6f}; all six at once {kl(allv):.6f}; ratio {x:.3f}")
            if weak:
                print(f"   single layers with a KL under five times its standard error: {', '.join(v.split('#')[1] for v in weak)}")
            ok = x >= 1.5 if name == "E1" else 0.75 <= x <= 1.25
            verdict(name, f"for {'the clip' if name == 'E1' else 'rounding'}, the KL of the six at once is {want}",
                    "under 1.5 times fails" if name == "E1" else "outside that range fails",
                    f"ratio {x:.3f}" + (f"; layers {', '.join(v.split('#')[1] for v in weak)} have a KL under 5 x its standard error" if weak else ""),
                    None if weak else ok, "not decided: a single layer is under five times its standard error" if weak else "")
        else:
            verdict(name, f"{label}: six at once against the sum of the six alone", want, "not scored", None)
    print()

    # ----- part F
    print("Part F: three more measures of a weight change")
    meas = {r["variant"]: r for r in read_jsonl(HERE / "measures.jsonl")}
    old_spec = [r["variant"] for r in read_jsonl(EARLIER / "spectral.jsonl") if r["variant"] != "flip:3.45"]
    v26 = [v for v in old_spec if v in meas and v in OLD]
    if len(v26) == 26:
        print("   on the 42 matrices of layers 0, 8, 16, 24, 32 and 40 (mean dB); mean KL from the first record")
        print("   variant | along the rows | along the columns | tree shift | squared error | spectral error | mean KL")
        for v in sorted(v26, key=lambda v: kl(v, OLD)):
            m = meas[v]
            print(f"   {v} | {m['along_rows_db_mean']:.2f} | {m['along_cols_db_mean']:.2f} | {m['tree_shift_db_mean']:.2f} | {m['squared_err_db_mean']:.2f} | {m['spectral_err_db_mean']:.2f} | {kl(v, OLD):.4f}")
        if "flip:3.45" in meas:
            m = meas["flip:3.45"]
            print(f"   flip:3.45 (not among the 26) | {m['along_rows_db_mean']:.2f} | {m['along_cols_db_mean']:.2f} | {m['tree_shift_db_mean']:.2f} | {m['squared_err_db_mean']:.2f} | {m['spectral_err_db_mean']:.2f} | {kl('flip:3.45', OLD):.4f}")
        ks = [kl(v, OLD) for v in v26]
        corr = {key: spearman([meas[v][key] for v in v26], ks) for key in ("along_rows_db_mean", "along_cols_db_mean", "tree_shift_db_mean", "squared_err_db_mean", "spectral_err_db_mean")}
        corr_read = {key: spearman([meas[v][key + "_read"] for v in v26], ks) for key in corr}
        print("   rank correlation with mean KL over the 26 variants, all 42 matrices: " + ", ".join(f"{k.replace('_db_mean', '')} {x:+.3f}" for k, x in corr.items()))
        print("   the same on the 36 matrices the model reads: " + ", ".join(f"{k.replace('_db_mean', '')} {x:+.3f}" for k, x in corr_read.items()))
        ar = lambda v: meas[v]["along_rows_db_mean"]
        tr = lambda v: meas[v]["tree_shift_db_mean"]
        verdict("F1", "along the rows ranks the 26 variants by KL with a rank correlation of at least 0.8, above that of the squared error on the same matrices",
                "either part fails", f"along the rows {corr['along_rows_db_mean']:+.3f}, squared error {corr['squared_err_db_mean']:+.3f}",
                corr["along_rows_db_mean"] >= 0.8 and corr["along_rows_db_mean"] > corr["squared_err_db_mean"])
        if "clip:3.45" in meas and "flip:3.45" in meas:
            verdict("F2", "along the rows has clip:3.45 as worse than XR:32", "it has it as better fails",
                    f"clip:3.45 {ar('clip:3.45'):.2f} dB, XR:32 {ar('XR:32'):.2f} dB", ar("clip:3.45") > ar("XR:32"))
            verdict("F3", "along the rows has clip:5 as better than XR:32 (a predicted failure of the measure)", "it has it as worse fails",
                    f"clip:5 {ar('clip:5'):.2f} dB, XR:32 {ar('XR:32'):.2f} dB", ar("clip:5") < ar("XR:32"))
            verdict("F4", "along the rows has flip:3.45 as better than clip:3.45", "it does not fails",
                    f"flip:3.45 {ar('flip:3.45'):.2f} dB, clip:3.45 {ar('clip:3.45'):.2f} dB", ar("flip:3.45") < ar("clip:3.45"))
        else:
            for n in ("F2", "F3", "F4"):
                verdict(n, "needs flip:3.45", "-", "not measured", None)
        verdict("F5", "tree shift has both clips as worse than XR:32, and XR:32~0.7 as worse than XR:32 (a predicted failure)",
                "any of the three orderings is the other way fails",
                f"clip:3.45 {tr('clip:3.45'):.2f}, clip:5 {tr('clip:5'):.2f}, XR:32~0.7 {tr('XR:32~0.7'):.2f}, XR:32 {tr('XR:32'):.2f} dB",
                tr("clip:3.45") > tr("XR:32") and tr("clip:5") > tr("XR:32") and tr("XR:32~0.7") > tr("XR:32"))
        verdict("F6", "tree shift ranks the 26 variants worse than the spectral error does", "equal or above fails",
                f"tree shift {corr['tree_shift_db_mean']:+.3f}, spectral error {corr['spectral_err_db_mean']:+.3f}",
                corr["tree_shift_db_mean"] < corr["spectral_err_db_mean"])
        extra = [v for v in meas if v not in old_spec and v != "flip:3.45" and v in ROWS]
        if extra:
            print(f"   held-out test: the variants of Parts B, C and D that change all kinds ({len(extra)} measured), rank correlation with their mean KL, beside the value on the 26")
            ke = [kl(v) for v in extra]
            for key in corr:
                print(f"   {key.replace('_db_mean', '')} | held out {spearman([meas[v][key] for v in extra], ke):+.3f} | on the 26 {corr[key]:+.3f}")
            print("   variant | along the rows | along the columns | tree shift | squared error | spectral error | mean KL")
            for v in sorted(extra, key=kl):
                m = meas[v]
                print(f"   {v} | {m['along_rows_db_mean']:.2f} | {m['along_cols_db_mean']:.2f} | {m['tree_shift_db_mean']:.2f} | {m['squared_err_db_mean']:.2f} | {m['spectral_err_db_mean']:.2f} | {kl(v):.4f}")
        else:
            print("   held-out test: not measured")
    else:
        print(f"   measures.jsonl has {len(v26)} of the 26 variants")
        for n in ("F1", "F2", "F3", "F4", "F5", "F6"):
            verdict(n, "Part F measures", "-", "not measured", None)
    wo = read_jsonl(HERE / "weights_only.jsonl")
    if len(wo) == 294:
        print("   the weights alone, mean over the layers of each kind: persistence (normalised), spread of the rows, spread of the columns")
        mean = lambda kind, key: sum(r[key] for r in wo if r["kind"] == kind) / sum(1 for r in wo if r["kind"] == kind)
        for k in KINDS:
            print(f"   {k} | {mean(k, 'persistence_normalised'):.4f} | {mean(k, 'row_spread'):.4f} | {mean(k, 'col_spread'):.4f} | zero rows {sum(r['zero_rows'] for r in wo if r['kind'] == k)}, zero columns {sum(r['zero_cols'] for r in wo if r['kind'] == k)}")
        if a_ok:
            per = [kl(f"X32:16@{k}") / ROWS[f"X32:16@{k}"]["weights_changed"] for k in KINDS]
            rho = spearman([mean(k, "persistence_normalised") for k in KINDS], per)
            verdict("F7", "over the seven kinds the rank correlation between mean persistence and KL per weight at Q = 16 is between -0.75 and 0.75",
                    "it is outside fails", f"{rho:+.3f}", -0.75 <= rho <= 0.75)
        else:
            verdict("F7", "persistence against KL per weight", "-", "Part A not complete", None)
        w_min = min(mean(k, "row_spread") for k in WRITERS)
        readers = [k for k in KINDS if k not in WRITERS]
        r_max = max(mean(k, "row_spread") for k in readers)
        verdict("F8", "the two kinds that write into the stream have a larger spread of rows than each of the five that read from it, in the mean over layers",
                "any reading kind has a larger one fails", f"smaller writer {w_min:.4f}, largest reader {r_max:.4f} ({max(readers, key=lambda k: mean(k, 'row_spread'))})",
                w_min > r_max)
    else:
        print(f"   weights_only.jsonl has {len(wo)} of 294 lines")
        verdict("F7", "persistence against KL per weight", "-", "not measured", None)
        verdict("F8", "spread of rows of the writers", "-", "not measured", None)
    print()

    # ----- joined arm
    print("The arm that joins the parts")
    joined = None
    for ln in (HERE / "allocate_joined.out").read_text().splitlines() if (HERE / "allocate_joined.out").exists() else []:
        if ln.startswith("JSON "):
            joined = json.loads(ln[5:])
    if joined:
        print("   levels: " + " ".join(f"{k}={joined['levels'][k]}" for k in KINDS) + f"; predicted mean bits {joined['mean_bits']:.4f}, predicted KL {joined['predicted_kl']:.6f}")
    if joined and joined["variant"] in ROWS:
        r = ROWS[joined["variant"]]
        table([joined["variant"]])
        verdict("G1", "the joined arm, at 4.5 bits or under, has a KL of 0.085 or less", "it is above 0.085 fails",
                f"{r['kl_mean']:.4f} at {r['bits_per_weight_all']:.4f} bits",
                r["kl_mean"] <= 0.085 and r["bits_per_weight_all"] <= 4.5 + 1e-9)
        if "VE832:16" in OLD:
            print(f"   not predicted: against VE832:16 (0.0747) {kl('VE832:16', OLD):.4f}; ", end="")
            q4 = next((v for v in OLD if v.startswith("file:") and "Q4_K_M" in v), None)
            print(f"against llama.cpp Q4_K_M {kl(q4, OLD):.4f}" if q4 else "")
    else:
        verdict("G1", "the joined arm at 4.5 bits or under, KL 0.085 or less", "above 0.085 fails", "not scored", None)
    print()

    # ----- extension 1
    print("Extension 1, Part H: the clip at 5 standard deviations on ffn_down, one layer at a time (PREREG-extension-1.md)")
    h_names = [f"clip:5@ffn_down#{il}" for il in range(42)]
    if all(v in ROWS for v in h_names):
        print("   layer | weight error of the clip, dB | KL | standard error | KL over five standard errors")
        for il, v in enumerate(h_names):
            r = ROWS[v]
            print(f"   {il} | {r['weight_err_db_mean']:.2f} | {r['kl_mean']:.6f} | {r['kl_mean_se']:.6f} | {'yes' if r['kl_mean'] >= 5 * r['kl_mean_se'] else 'NO'}")
        kls = [kl(v) for v in h_names]
        total = sum(kls)
        joint = OLD["clip:5@ffn_down"]["kl_mean"] if "clip:5@ffn_down" in OLD else None
        weak = [il for il, v in enumerate(h_names) if kl(v) < 5 * ROWS[v]["kl_mean_se"]]
        weak_share = sum(kls[il] for il in weak) / total
        small = weak_share < 0.1
        print(f"   sum of the 42 single-layer KLs {total:.6f}; the clip on all layers at once (first record) {joint:.6f}; ratio {total / joint:.3f}")
        top4 = sum(sorted(kls, reverse=True)[:4])
        print(f"   four largest layers: {', '.join(str(il) for il in sorted(range(42), key=lambda i: -kls[i])[:4])}, {top4:.6f} = {100 * top4 / total:.1f} % of the sum")
        print(f"   layers with a KL under five times its standard error: {', '.join(map(str, weak)) or 'none'}; together {100 * weak_share:.2f} % of the sum")
        ratio = total / joint
        not_dec = bool(weak) and not small
        verdict("H1", "the sum of the 42 single-layer KLs is 0.75 to 1.25 times the KL of the clip on all layers at once", "the ratio is outside that range fails",
                f"ratio {ratio:.3f}" + (f"; layers {', '.join(map(str, weak))} are under 5 x their standard error and hold {100 * weak_share:.1f} % of the sum" if weak else ""),
                None if not_dec else 0.75 <= ratio <= 1.25, "not decided" if not_dec else "")
        verdict("H2", "a few layers carry it: the four largest single-layer KLs make more than half of the sum", "they make half or less fails",
                f"{100 * top4 / total:.1f} % of the sum" + (f"; layers {', '.join(map(str, weak))} under 5 x their standard error hold {100 * weak_share:.1f} %" if weak else ""),
                None if not_dec else top4 > 0.5 * total, "not decided" if not_dec else "")
        rho = spearman(kls, [ROWS[v]["weight_err_db_mean"] for v in h_names])
        verdict("H3", "over the 42 layers, the rank correlation between a layer's KL and the weight error of its clip (dB, nearer zero is a larger change) is at least 0.5",
                "it is under 0.5 fails", f"{rho:+.3f}", rho >= 0.5)
    else:
        print(f"   {sum(v in ROWS for v in h_names)} of 42 layers scored")
        for n in ("H1", "H2", "H3"):
            verdict(n, "Part H", "-", "not scored", None)
    print()
    print("Extension 1, Part K: the joined arm with the unread keys and values zeroed and counted at 0 bits")
    kv = PART_K0 = "X32s8mix:attn_q=32,attn_k=64,attn_v=64,attn_output=46,ffn_gate=16,ffn_up=16,ffn_down=16"
    if kv + "+deadkv" in ROWS and kv in ROWS:
        table([kv, kv + "+deadkv"])
        r = ROWS[kv + "+deadkv"]
        print(f"   weights zeroed {r['weights_zeroed']} in {r['tensors_zeroed']} tensors")
        same = round(r["kl_mean"], 4) == round(kl(kv), 4)
        verdict("K1", "the KL equals that of the joined arm to four decimals, and the bits fall from 4.484 to under 4.41", "the KL differs in the fourth decimal, or the bits are 4.41 or more fails",
                f"KL {r['kl_mean']:.6f} against {kl(kv):.6f}; bits {r['bits_per_weight_all']:.4f} against {ROWS[kv]['bits_per_weight_all']:.4f}; weights zeroed {r['weights_zeroed']}",
                same and r["bits_per_weight_all"] < 4.41 and r["weights_zeroed"] == 55050240)
    else:
        verdict("K1", "Part K", "-", "not scored", None)
    print()

    # ----- extension 2
    print("Extension 2, Part L: the large entries of ffn_down by layer (weights only, large_by_layer.json)")
    lj = HERE / "large_by_layer.json"
    if lj.exists():
        d = json.loads(lj.read_text())
        print("   axes: " + d["layer10"]["axis_note"])
        print("   layer | entries beyond 5 sd | largest entry, in sd of its row | clip's squared change over squared size of the matrix")
        for r in d["layers"]:
            print(f"   {r['layer']} | {r['n_beyond']} | {r['max_sd']:.2f} | {r['change_over_size']:.3e}")
        l10 = d["layer10"]
        for key, name in (("units", "hidden units (columns)"), ("dims", "dimensions of the stream (rows)")):
            r = l10[key]
            print(f"   layer 10, {name}: {l10[key + '_with_change']} have any change; {r['half']} hold half of the squared change, {r['nine_tenths']} hold nine tenths; "
                  f"the top one holds {100 * r['top'][0]['share']:.1f} %, the top ten {100 * sum(x['share'] for x in r['top'][:10]):.1f} %")
            print("   top 20: " + ", ".join(f"{x['index']} {100 * x['share']:.2f} %" for x in r["top"]))
    else:
        print("   not computed")
    print()
    print("Extension 2, Part M: parts of layer 10's ffn_down, and one matrix under the other rule")
    table(PART_M0 := ["clip:5@ffn_down#10", "clip:5@ffn_down#10:u1", "clip:5@ffn_down#10:u10", "clip:5@ffn_down#10:not10", "clip:5@ffn_down#10:d10",
                      "R+S:32|ffn_down#10=XR:32", "XR:32|ffn_down#10=R+S:32"])
    for v in ("R+S:32", "XR:32"):
        if v in OLD:
            print(f"   {v} on all matrices (first record): KL {kl(v, OLD):.4f}, bits {OLD[v]['bits_per_weight']:.3f}")
    for v in ("R+S:32|ffn_down#10=XR:32", "XR:32|ffn_down#10=R+S:32"):
        if v in ROWS and "arm_check" in ROWS[v]:
            print(f"   {v}: the base arm's weight errors match the first record's by kind, largest difference {ROWS[v]['arm_check']['max_abs_difference_db']:.6f} dB")
    v10 = "clip:5@ffn_down#10"
    mv = dict(zip(("m1", "m2", "m3", "m4", "m5", "m6"), PART_M0[1:]))
    if "clip:5@ffn_down#10:d10" in ROWS and "clip:5@ffn_down#10:u10" in ROWS:
        print(f"   variant 4 (no prediction) against variant 2: ten stream dimensions {kl(mv['m4']):.4f}, ten hidden units {kl(mv['m2']):.4f}")
    mid = (kl("R+S:32", OLD) + kl("XR:32", OLD)) / 2 if "R+S:32" in OLD and "XR:32" in OLD else None
    if v10 in ROWS and mid is not None:
        print(f"   layer 10 alone {kl(v10):.6f}; 0.8 of it {0.8 * kl(v10):.4f}, 0.2 of it {0.2 * kl(v10):.4f}, 0.5 of it {0.5 * kl(v10):.4f}; midpoint of the two rules {mid:.4f}")
    for name, var, pred, thr, ok_fn, fail in (
            ("M1", mv["m2"], "ten hidden units carry it: variant 2 scores at least 0.1248 (0.8 of layer 10's KL)", "0.1248", lambda x: x >= 0.1248, "it scores less"),
            ("M2", mv["m3"], "the rest carries little: variant 3 scores at most 0.0312 (0.2 of layer 10's KL)", "0.0312", lambda x: x <= 0.0312, "it scores more"),
            ("M3", mv["m1"], "one hidden unit carries half: variant 1 scores at least 0.0780", "0.0780", lambda x: x >= 0.0780, "it scores less"),
            ("M4", mv["m5"], "variant 5 (R+S:32 everywhere, XR:32 on layer 10) scores at most 0.2288, the midpoint of the two rules", "0.2288", lambda x: x <= 0.2288, "it scores more"),
            ("M5", mv["m6"], "variant 6 (XR:32 everywhere, R+S:32 on layer 10) scores at least 0.2288", "0.2288", lambda x: x >= 0.2288, "it scores less")):
        if var in ROWS:
            verdict(name, pred, fail + " fails", f"{kl(var):.4f} (needs {thr})", ok_fn(kl(var)))
        else:
            verdict(name, pred, fail + " fails", "not scored", None)
    print()

    # ----- verdict
    print("Verdicts")
    print("   id | prediction | fails if | measured | verdict")
    for name, pred, thr, meas_s, ok, note in VERDICTS:
        word = "NOT RUN" if ok is None and not note else ("NOT DECIDED" if ok is None else ("HOLDS" if ok else "FAILS"))
        print(f"   {name} | {pred} | {thr} | {meas_s} | {word}")
    decided = [ok for *_, ok, _ in VERDICTS if ok is not None]
    print(f"   {sum(1 for ok in decided if ok)} of {len(VERDICTS)} predictions hold; {sum(1 for ok in decided if not ok)} fail; "
          f"{sum(1 for *_, ok, _ in VERDICTS if ok is None)} not decided or not run")


if __name__ == "__main__":
    main()
