#!/usr/bin/env python3
"""Tables and verdicts for the MMLU subset, from results.jsonl of this folder.

Every comparison is by question: two variants answered the same questions, so the error of a
difference is the standard deviation of the per-question differences over the root of the count.
Scores the predictions M1 to M8 of PREREG.md by the rules written there. The mean KL of each
variant is copied from the records that measured it, for the reader.

Usage, from the root of a checkout:
  python3 experiments/mmlu-subset-2026-11-18/summary.py > experiments/mmlu-subset-2026-11-18/results.out
  python3 experiments/mmlu-subset-2026-11-18/summary.py --first 1000     # only the first 1,000 questions of the draw
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
MODELS = (
    ("Qwen2.5-1.5B-Instruct", "qwen2.5-1.5b", "qwen15", EXP / "model-output-qwen25-2026-11-17" / "results.jsonl"),
    ("SmolLM2-1.7B-Instruct", "smollm2", "smollm2", EXP / "model-output-smollm2-2026-11-17" / "results.jsonl"),
)
REF = "16-bit reference"
ORDER = (REF, "Q4_0", "Q4_K_M", "Q4_K_M with the matrix", "ZBG128:b560s16", "hZBG128:b560s16")
PAIRS = (
    ("hZBG128:b560s16", "Q4_K_M with the matrix"),
    ("ZBG128:b560s16", "Q4_K_M"),
    ("hZBG128:b560s16", "Q4_0"),
    ("ZBG128:b560s16", "Q4_0"),
    ("Q4_K_M with the matrix", "Q4_K_M"),
    ("hZBG128:b560s16", "ZBG128:b560s16"),
)
CHANCE = 25.0


def load(path, model, data=None):
    """variant -> row, for the rows of one model (a part of its file's name), and of one questions file if one is named."""
    rows = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if model in r["model"].lower() and (data is None or r["data"] == data):
                    rows[r["variant"]] = r
    return rows


def mean_se(xs):
    """Mean and its standard error, in percent, of a list of per-question values."""
    n = len(xs)
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return 100.0 * m, 100.0 * math.sqrt(var / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=0, help="use only the first N questions of the draw")
    ap.add_argument("--results", default=str(HERE / "results.jsonl"), help="another results file, for tests of this program")
    args = ap.parse_args()
    print("A subset of MMLU in llama.cpp's multiple-choice mode: the same questions for every variant, compared question by question")
    print("An error is a standard error over questions. An error of a difference is taken from the per-question differences.")
    verdict = {}
    seen = {}
    for title, key, short, base_path in MODELS:
        mine = load(Path(args.results), key, "mmlu-test.bin")
        base = load(base_path, key)
        imat = load(EXP / "imatrix-baseline-2026-11-17" / "results.jsonl", key)
        names = {REF: "reference", "Q4_0": f"file:{short}-Q4_0.gguf", "Q4_K_M": f"file:{short}-Q4_K_M.gguf",
                 "Q4_K_M with the matrix": f"file:{short}-Q4_K_M-imat.gguf", "ZBG128:b560s16": "ZBG128:b560s16", "hZBG128:b560s16": "hZBG128:b560s16"}
        other = {REF: None, "Q4_0": base.get(names["Q4_0"]), "Q4_K_M": base.get(names["Q4_K_M"]),
                 "Q4_K_M with the matrix": imat.get(names["Q4_K_M with the matrix"]), "ZBG128:b560s16": base.get("ZBG128:b560s16"),
                 "hZBG128:b560s16": imat.get("hZBG128:b560s16")}
        rows = {a: mine.get(names[a]) for a in ORDER}
        print()
        print(title)
        scored = [a for a in ORDER if rows[a]]
        print(f"variants scored: {len(scored)} of {len(ORDER)}")
        if not scored:
            continue
        n = min(len(rows[a]["chosen"]) for a in scored)
        if args.first:
            n = min(n, args.first)
        answer = rows[scored[0]]["answer"][:n]
        if any(rows[a]["answer"][:n] != answer or len(rows[a]["answer"]) != len(rows[a]["chosen"]) for a in scored):
            raise SystemExit("FAIL: the variants did not answer the same questions")
        right = {a: [int(c == t) for c, t in zip(rows[a]["chosen"][:n], answer)] for a in scored}
        same = {a: [int(c == d) for c, d in zip(rows[a]["chosen"][:n], rows[REF]["chosen"][:n])] for a in scored} if REF in scored else {}
        for a in scored:
            if n == len(rows[a]["chosen"]) and sum(right[a]) != rows[a]["right"]:
                raise SystemExit(f"FAIL: {a}: the answers chosen do not add up to the count the tool gave")
            o = other[a]
            if o and "bits_per_weight" in rows[a] and (abs(o["bits_per_weight"] - rows[a]["bits_per_weight"]) > 1e-9 or abs(o["weight_err_db_mean"] - rows[a]["weight_err_db_mean"]) > 1e-6):
                raise SystemExit(f"FAIL: {a}: the arm built here is not the arm the KL record scored")
            if o and "file_sha256" in rows[a] and o["file_sha256"] != rows[a]["file_sha256"]:
                raise SystemExit(f"FAIL: {a}: the file scored here is not the file the KL record scored")
        print(f"questions: {n}, the tool's draw from the {rows[scored[0]]['questions_in_file']} of {rows[scored[0]]['data']}. Chance is {CHANCE:.0f} percent.")
        print("variant | bits per weight | mean KL on wikitext-2, from its record | right | accuracy, percent | its error | accuracy less the reference's, points | its error | in errors | "
              "questions where one of the two is right and the other is not, percent | smallest loss this could see, points | same answer as the reference, percent | its error")
        for a in scored:
            acc, se = mean_se(right[a])
            o = other[a]
            head = f"{a} | {o['bits_per_weight']:.3f} | {o['kl_mean']:.4f}" if o else f"{a} | 16 | 0"
            if a == REF or REF not in scored:
                print(f"{head} | {sum(right[a])} | {acc:.2f} | {se:.2f} | - | - | - | - | - | - | -")
                continue
            d, dse = mean_se([x - y for x, y in zip(right[a], right[REF])])
            disc = 100.0 * sum(x != y for x, y in zip(right[a], right[REF])) / n
            s, sse = mean_se(same[a])
            print(f"{head} | {sum(right[a])} | {acc:.2f} | {se:.2f} | {d:+.2f} | {dse:.2f} | {d / dse if dse else 0:+.1f} | {disc:.1f} | {2.8 * dse:.1f} | {s:.2f} | {sse:.2f}")
        print("   The smallest loss this could see is 2.8 errors of the difference: found 4 times in 5 by a test at 2 errors.")
        print("comparison | accuracy of the first less the second, points | its error | in errors | same answer as the reference, first less second, points | its error | in errors")
        gaps = {}
        for a, b in PAIRS:
            if a in scored and b in scored and REF in scored:
                d, dse = mean_se([x - y for x, y in zip(right[a], right[b])])
                s, sse = mean_se([x - y for x, y in zip(same[a], same[b])])
                gaps[(a, b)] = (d, d / dse if dse else 0.0, s, s / sse if sse else 0.0)
                print(f"{a} against {b} | {d:+.2f} | {dse:.2f} | {gaps[(a, b)][1]:+.1f} | {s:+.2f} | {sse:.2f} | {gaps[(a, b)][3]:+.1f}")
        if REF in scored:
            acc, se = mean_se(right[REF])
            lost = {a: mean_se([x - y for x, y in zip(right[a], right[REF])])[0] for a in scored if a != REF}
            seen[short] = {"ref": (acc, se), "lost": lost, "gaps": gaps}

    def both(test, need):
        """yes if the test holds on every model that has what it needs, NO if it fails on one, blank until all are in."""
        have = [m for m in ("qwen15", "smollm2") if m in seen and need(seen[m])]
        if any(not test(seen[m]) for m in have):
            return "NO"
        return "yes" if len(have) == 2 else "not all scored"

    four = ("Q4_K_M", "Q4_K_M with the matrix", "ZBG128:b560s16", "hZBG128:b560s16")
    pair = lambda a, b: (lambda s: (a, b) in s["gaps"])
    verdict["M1"] = both(lambda s: (s["ref"][0] - CHANCE) / s["ref"][1] >= 5, lambda s: True)
    verdict["M2"] = both(lambda s: all(s["lost"][a] >= -2.5 for a in four), lambda s: all(a in s["lost"] for a in four))
    verdict["M3"] = both(lambda s: s["gaps"][("hZBG128:b560s16", "Q4_K_M with the matrix")][1] > -2, pair("hZBG128:b560s16", "Q4_K_M with the matrix"))
    verdict["M4"] = both(lambda s: s["gaps"][("ZBG128:b560s16", "Q4_K_M")][1] > -2, pair("ZBG128:b560s16", "Q4_K_M"))
    verdict["M5"] = both(lambda s: s["gaps"][("hZBG128:b560s16", "Q4_0")][3] >= 2, pair("hZBG128:b560s16", "Q4_0"))
    verdict["M6"] = both(lambda s: s["gaps"][("hZBG128:b560s16", "Q4_K_M with the matrix")][3] > -2, pair("hZBG128:b560s16", "Q4_K_M with the matrix"))
    sm = seen.get("smollm2", {"gaps": {}})["gaps"]
    for name, key in (("M7", ("Q4_K_M with the matrix", "Q4_K_M")), ("M8", ("hZBG128:b560s16", "ZBG128:b560s16"))):
        verdict[name] = ("yes" if sm[key][3] >= 2 else "NO") if key in sm else "not scored"
    print()
    print("The predictions of PREREG.md")
    print("prediction | holds")
    for k in sorted(verdict):
        print(f"{k} | {verdict[k]}")
    print(f"hold: {sum(v == 'yes' for v in verdict.values())} of {len(verdict)}")


if __name__ == "__main__":
    main()
