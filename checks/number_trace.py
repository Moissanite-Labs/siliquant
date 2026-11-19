#!/usr/bin/env python3
"""Can every number in a record be found in a committed output? A first pass by machine.

The repository's rule: every number in a README can be traced to a committed output file.
The records are checked against it by a second reader. This program does the dull part of
that reading (crossing X3 of notes/cross-org-grafts.md). For each record under experiments/
it takes every number with a decimal point in the README and looks for a number in the
record's own output files (*.out, *.json, *.jsonl) that rounds to it. Failing that it looks
for the decibels of a stored value, then in the plan of the record, then in the outputs of
the other records and the limits table.

That first pass is weak, and the program measures how weak: an output file holds thousands of
numbers, so a wrong number is often found too. Table 1 gives, beside each count, how many
numbers are still found after being changed by 3 in their last place.

The second pass is stricter. A row of a table in a README is mostly copied from one or two
rows of an output table. So for every line of a README with three numbers or more it looks
for the two lines of the record's *.out files that together hold the most of them, and lists
the lines it cannot cover that way. Table 2 also measures this pass: it changes one number of
each line by 3 in its last place and counts how often the change is caught.

What it cannot do, and says:
  - A number that is not found may be right: a difference or a ratio of two outputs that no
    table prints, a row put together from several tables, or a figure from a source. Those
    have to be read by a person. The lists are where to start.
  - A number that is found may still be in the wrong place.
It never changes a record.

Usage, from the root of a checkout:
  python3 checks/number_trace.py > checks/number_trace.out
"""
from __future__ import annotations

import bisect
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NUMBER = re.compile(r"(?<![\w.,/])[-+−]?\d+(?:,\d{3})*\.\d+(?![\w]|\.\d)")
ANY_NUMBER = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
SKIP_LINE = re.compile(r"arXiv|sha-?256|SHA-?256|doi", re.I)


def strip_code(text):
    """The text with fenced code and inline code blanked out, line numbers kept."""
    out, fenced = [], False
    for line in text.split("\n"):
        if line.strip().startswith("```"):
            fenced = not fenced
            out.append("")
            continue
        out.append("" if fenced else re.sub(r"`[^`]*`", " ", line))
    return out


def numbers_in_json(obj, into):
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        if math.isfinite(obj):
            into.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            numbers_in_json(v, into)
    elif isinstance(obj, list):
        for v in obj:
            numbers_in_json(v, into)


def pool_of(files):
    """Every number in the given files, as two sorted lists of sizes: the numbers, and the decibels of the stored positive ones."""
    direct, stored = [], []
    for f in files:
        text = f.read_text(errors="replace")
        if f.suffix == ".json":
            try:
                numbers_in_json(json.loads(text), stored)
                continue
            except json.JSONDecodeError:
                pass
        if f.suffix == ".jsonl":
            ok = True
            for line in text.splitlines():
                if line.strip():
                    try:
                        numbers_in_json(json.loads(line), stored)
                    except json.JSONDecodeError:
                        ok = False
            if ok:
                continue
        for m in ANY_NUMBER.finditer(text):
            try:
                direct.append(float(m.group(0)))
            except ValueError:
                pass
    decibels = [10.0 * math.log10(v) for v in stored if 0.0 < v < 1e6]
    return sorted(abs(v) for v in direct + stored), sorted(abs(v) for v in decibels)


def found(pool, value, decimals):
    """True if some number of the pool rounds to the value at that many decimals, sign aside."""
    half = 0.5 * 10.0 ** -decimals * (1.0 + 1e-9)
    i = bisect.bisect_left(pool, abs(value) - half)
    return i < len(pool) and pool[i] <= abs(value) + half


def lines_of(files):
    """The numbers on every line of the given text files, as a list of sorted lists of sizes."""
    out = []
    for f in files:
        for line in f.read_text(errors="replace").splitlines():
            vals = []
            for m in ANY_NUMBER.finditer(line):
                try:
                    vals.append(abs(float(m.group(0))))
                except ValueError:
                    pass
            if vals:
                out.append(sorted(vals))
    return out


def cover(row, lines):
    """How many of the numbers of a README line are held by the best two output lines together."""
    if not lines:
        return 0
    hits = [[found(vals, v, d) for v, d in row] for vals in lines]
    first = max(range(len(lines)), key=lambda i: sum(hits[i]))
    got = hits[first]
    if all(got):
        return len(row)
    best = sum(got)
    for h in hits:
        both = sum(a or b for a, b in zip(got, h))
        if both > best:
            best = both
    return best


def tokens(readme):
    out = []
    for no, line in enumerate(strip_code(readme.read_text()), start=1):
        if SKIP_LINE.search(line):
            continue
        for m in NUMBER.finditer(line):
            raw = m.group(0)
            text = raw.replace("−", "-").replace(",", "")
            decimals = len(text.split(".")[1])
            out.append((no, raw, float(text), decimals, line[max(0, m.start() - 45):m.end() + 25].strip()))
    return out


def main():
    folders = sorted(p for p in (ROOT / "experiments").iterdir() if p.is_dir() and (p / "README.md").exists())
    outputs = {f: sorted(x for x in f.iterdir() if x.suffix in (".out", ".json", ".jsonl")) for f in folders}
    pools = {f: pool_of(outputs[f]) for f in folders}
    bounds = pool_of([ROOT / "bounds" / "bounds.out"])
    print("Numbers with a decimal point in each record's README, looked for in committed outputs")
    print("own = the record's own *.out, *.json and *.jsonl. dB = ten times the log of a value stored in its json. plan = the record's PREREG files.")
    print("elsewhere = the outputs of the other records and bounds/bounds.out. A number is found if some number rounds to it, sign aside.")
    print()
    print("1. One number at a time")
    print("record | numbers | found in own outputs | only as dB of a stored value | only in the plan | only elsewhere | not found | of those, with two decimals or more | "
          "numbers with two decimals or more | still found in own outputs when changed by 3 in the last place")
    misses = {}
    total = [0, 0]
    for f in folders:
        own, own_db = pools[f]
        plan = pool_of(sorted(f.glob("PREREG*.md")))[0]
        others = sorted(v for g in folders if g != f for v in pools[g][0]) + []
        others = sorted(others + bounds[0])
        counts = [0, 0, 0, 0, 0]
        miss = []
        toks = tokens(f / "README.md")
        for no, raw, value, decimals, context in toks:
            if found(own, value, decimals):
                counts[0] += 1
            elif found(own_db, value, decimals):
                counts[1] += 1
            elif found(plan, value, decimals):
                counts[2] += 1
            elif found(others, value, decimals):
                counts[3] += 1
            else:
                counts[4] += 1
                if decimals >= 2:
                    miss.append((no, raw, context))
        misses[f.name] = miss
        total[0] += len(toks)
        total[1] += counts[4]
        fine = [(v, d) for _, _, v, d, _ in toks if d >= 2]
        wrong = sum(found(own, v + 3.0 * 10.0 ** -d, d) for v, d in fine)
        print(f"{f.name} | {len(toks)} | {counts[0]} | {counts[1]} | {counts[2]} | {counts[3]} | {counts[4]} | {len(miss)} | {len(fine)} | {wrong}")
    print()
    print(f"in all: {total[0]} numbers, {total[1]} not found")
    print()
    print("Not found, with two decimals or more: the line of the README, the number, and where it stands")
    for name, miss in misses.items():
        if miss:
            print(f"  {name}")
            for no, raw, context in miss:
                print(f"    line {no}: {raw}   ...{context}...")
    print()
    print("2. A line at a time: lines of a README with three numbers or more, against the best two lines of the record's *.out files")
    print("record | lines | all numbers on two output lines | not | a number changed by 3 in the last place is caught, of the lines that were covered")
    uncovered = {}
    for f in folders:
        lines = lines_of([x for x in outputs[f] if x.suffix == ".out"])
        rows = {}
        for no, raw, value, decimals, context in tokens(f / "README.md"):
            rows.setdefault(no, []).append((raw, value, decimals))
        rows = {no: r for no, r in rows.items() if len(r) >= 3}
        full, caught, tried, bad = 0, 0, 0, []
        for no, r in sorted(rows.items()):
            row = [(v, d) for _, v, d in r]
            got = cover(row, lines)
            if got == len(row):
                full += 1
                changed = [(row[0][0] + 3.0 * 10.0 ** -row[0][1], row[0][1])] + row[1:]
                tried += 1
                caught += cover(changed, lines) < len(row)
            else:
                bad.append((no, got, len(row), " ".join(raw for raw, _, _ in r)))
        uncovered[f.name] = bad
        print(f"{f.name} | {len(rows)} | {full} | {len(bad)} | {caught} of {tried}")
    print()
    print("Lines not covered by two output lines: the line of the README, how many of its numbers the best two lines hold, and its numbers")
    for name, bad in uncovered.items():
        if bad:
            print(f"  {name}")
            for no, got, n, raws in bad:
                print(f"    line {no}: {got} of {n}   {raws[:150]}")
    sys.exit(0)


if __name__ == "__main__":
    main()
