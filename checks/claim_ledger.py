#!/usr/bin/env python3
"""Do the check notes say what each verdict rests on, in words from a fixed list?

The notes under notes/ that check a proposal claim by claim have a table with a Basis column.
This program reads those tables (crossing X4 of notes/cross-org-grafts.md) and applies two
rules.

  1. Every part of a Basis is one of a fixed list of kinds: source, search, read, reported,
     checked, derived, proved, recomputed, record, none. Anything else is listed.
  2. A row whose basis is "recomputed", "measured" or "proved" names, in backticks, a file
     of this repository that exists. A folder does not count: the reader has to be able
     to open the output or the program. A row that rests on a record ("the first
     experiment", "the second record") is taken as naming that record. So is a row of a
     note that says once, above its tables, which record a word of the Basis means
     ("**Measured** means a figure from a committed output of the record ...").

It reads the Markdown as it stands. It keeps no second copy of the claims. It exits with 1
if any row breaks a rule.

It cannot tell whether the file named holds the number claimed. That is the second reader's
work, and checks/number_trace.py shows how little a machine does of it.

Usage, from the root of a checkout:
  python3 checks/claim_ledger.py > checks/claim_ledger.out
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KINDS = (
    ("recomputed", re.compile(r"\brecomputed\b|\bmeasured\b")),
    ("proved", re.compile(r"\bproved\b")),
    ("derived", re.compile(r"\bderived\b")),
    ("checked", re.compile(r"\bchecked\b")),
    ("reported", re.compile(r"\breported\b")),
    ("record", re.compile(r"\brecords?\b|\bexperiments?\b")),
    ("source", re.compile(r"\bsource\b")),
    ("search", re.compile(r"\bsearch\b")),
    ("read", re.compile(r"\bread of\b")),
    ("none", re.compile(r"^none$")),
)
NEEDS_FILE = ("recomputed", "proved")
TICKED = re.compile(r"`([^`]+)`")
RECORD = re.compile(r"\b(first|second|third|fourth|fifth|sixth) (experiment|record)\b|experiments/[\w.-]+")


def kinds_of(basis):
    """The kinds a Basis cell names, and the parts of it that name none."""
    found, unknown = [], []
    for part in re.split(r",\s*(?:and\s+)?|\s+and\s+", basis.strip()):
        part = part.strip()
        if not part:
            continue
        hits = [k for k, pat in KINDS if pat.search(part)]
        if hits:
            found += [h for h in hits if h not in found]
        elif not TICKED.search(part) and not re.fullmatch(r"(one|the) [\w ]+|the rest", part):      # a file named, or "one example"
            unknown.append(part)
    return found, unknown


_BY_NAME = {}


def files_named(text):
    """The files and folders a text names in backticks, as (name, is a file here, is a folder here). A bare file name counts if one file of the tree has it."""
    if not _BY_NAME:
        for p in ROOT.rglob("*"):
            if p.is_file() and ".git" not in p.parts:
                _BY_NAME.setdefault(p.name, []).append(p)
    out = []
    for raw in TICKED.findall(text):
        cand = raw.strip().split(" ")[0].rstrip(".,;:")
        if "/" in cand:
            p = ROOT / cand
            out.append((cand, p.is_file(), p.is_dir()))
        elif re.search(r"\.(py|out|json|jsonl|md)$", cand):
            out.append((cand, len(_BY_NAME.get(cand, [])) == 1, False))
    return out


LEGEND = re.compile(r"\*\*(Measured|Recomputed|Proved)\*\* means ([^*]*?experiments/[\w.-]+)")


def legend_of(text):
    """The words of a Basis that the note ties to a record once, above its tables."""
    return {m.group(1).lower() for m in LEGEND.finditer(text)}


def tables(path):
    """Rows of every table of the note that has a Basis column, as (line number, cells by header)."""
    header = None
    for no, line in enumerate(path.read_text().split("\n"), start=1):
        if not line.startswith("|"):
            header = None
            continue
        cells = [c.strip() for c in re.split(r"(?<!\\)\|", line.strip().strip("|"))]      # a pipe written \| is text
        if "Basis" in cells and "Verdict" in cells:
            header = cells
            continue
        if header is None or set(line.strip()) <= set("|-: "):
            continue
        if len(cells) != len(header):
            yield no, None
            continue
        yield no, dict(zip(header, cells))


def main():
    notes = sorted(p for p in (ROOT / "notes").glob("*.md") if "| Basis |" in p.read_text())
    print("The Basis column of the check notes, read against two rules")
    print("kinds: " + ", ".join(k for k, _ in KINDS) + ". A recomputed or proved row has to name a file of this repository that exists, or a record.")
    print()
    print("note | rows | of them recomputed or proved | name a file that exists | name a record only | name neither | parts of a Basis outside the list | rows that could not be read")
    problems = []
    count = {}
    for note in notes:
        rows = need = with_file = with_record = 0
        legend = legend_of(note.read_text())
        neither, outside, unread = [], [], []
        for no, row in tables(note):
            if row is None:
                unread.append(no)
                continue
            rows += 1
            found, unknown = kinds_of(row["Basis"])
            for k in found:
                count[k] = count.get(k, 0) + 1
            for part in unknown:
                outside.append((no, row.get("#", "?"), part))
            if any(k in found for k in NEEDS_FILE):
                need += 1
                text = " ".join(row.values())
                named = files_named(text)
                if any(is_file for _, is_file, _ in named):
                    with_file += 1
                elif RECORD.search(text) or any(re.search(rf"\b{word}\b", row["Basis"]) for word in legend):
                    with_record += 1
                else:
                    folders = [c for c, _, is_dir in named if is_dir]
                    missing = [c for c, is_file, is_dir in named if not is_file and not is_dir]
                    why = "names only a folder: " + ", ".join(folders) if folders else ("names a file that is not there: " + ", ".join(missing) if missing else "names no file")
                    neither.append((no, row.get("#", "?"), row["Basis"], why))
        print(f"{note.relative_to(ROOT)} | {rows} | {need} | {with_file} | {with_record} | {len(neither)} | {len(outside)} | {len(unread)}")
        problems += [(note, "recomputed or proved, and " + why, no, num, basis) for no, num, basis, why in neither]
        problems += [(note, "a part of the Basis is outside the list", no, num, part) for no, num, part in outside]
        problems += [(note, "the row does not have the columns of its header", no, "?", "") for no in unread]
    print()
    print("How often each kind is named, over all rows: " + ", ".join(f"{k} {count.get(k, 0)}" for k, _ in KINDS))
    print()
    print("Rows that break a rule")
    for note, what, no, num, detail in problems:
        print(f"  {note.relative_to(ROOT)} line {no}, claim {num}: {what}. Basis: {detail}")
    if not problems:
        print("  none")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
