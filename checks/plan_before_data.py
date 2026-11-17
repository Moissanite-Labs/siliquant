#!/usr/bin/env python3
"""Was each record's plan committed before its data? Read from the history of the repository.

The repository's rule: a study has a plan committed before its data, or says it was not
preregistered. A record says so in words. This program reads it off git instead (crossing X2
of notes/cross-org-grafts.md). For every folder under experiments/ it finds

  - the commits that added or changed a plan (PREREG*.md),
  - the commits that added a file of data (*.out, *.json, *.jsonl),
  - the commits that changed a program after the first data was committed,

and reports, for each record: whether any data was committed before the first plan, whether
the first data after the plan descends from it, each later change to a plan with whether new
data followed it (a new file of data, or rows added to one), and whether the record's README
names each such change by its commit.

Data committed in the same commit as the first plan is counted apart: it is what the plan was
written with, a pilot for instance, and the plan has to say so.

It cannot show what was read or run before a commit. A plan committed first proves only that
the plan existed first. It exits with 1 if any record has data from before its first plan, or
first data that does not descend from it.

Usage, from the root of a checkout with full history:
  python3 checks/plan_before_data.py > checks/plan_before_data.out
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = re.compile(r"\.(out|json|jsonl)$")
PLAN = re.compile(r"(^|/)PREREG[^/]*\.md$")


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True, capture_output=True, text=True).stdout


def is_ancestor(a, b):
    """True if commit a is an ancestor of commit b and not b itself."""
    if a == b:
        return False
    return subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", a, b]).returncode == 0


def history(folder):
    """Commits that touched the folder, oldest first, as (hash, date, subject, [(status, path)])."""
    raw = git("log", "--reverse", "--topo-order", "--no-merges", "--name-status", "--format=@%H|%cI|%s", "--", folder)
    out = []
    for line in raw.splitlines():
        if line.startswith("@"):
            h, date, subject = line[1:].split("|", 2)
            out.append((h, date, subject, []))
        elif line.strip():
            parts = line.split("\t")
            out[-1][3].append((parts[0][0], parts[-1]))
    return out


def main():
    folders = sorted(p for p in (ROOT / "experiments").iterdir() if p.is_dir())
    shallow = git("rev-parse", "--is-shallow-repository").strip() == "true"
    print("Plans and data in the order git has them, for every record under experiments/")
    print(f"head {git('rev-parse', '--short', 'HEAD').strip()}" + (" (a shallow clone: the history may be cut short)" if shallow else ""))
    print()
    print("record | first plan | data files committed with it | first data after it | plan before data | plan changed before first data | plan changed after first data | of those, followed by new data | named in the README | programs changed after first data")
    bad = 0
    notes = []
    for folder in folders:
        rel = str(folder.relative_to(ROOT))
        hist = history(rel)
        plans = [(h, d, s, [p for st, p in files if PLAN.search(p)]) for h, d, s, files in hist]
        plans = [x for x in plans if x[3]]
        data_adds = [(h, d, s, [p for st, p in files if st == "A" and DATA.search(p)]) for h, d, s, files in hist]
        data_adds = [x for x in data_adds if x[3]]
        data_changes = [h for h, d, s, files in hist if any(st in "AM" and DATA.search(p) for st, p in files)]      # rows appended to a file count too
        if not plans:
            print(f"{folder.name} | none | - | {data_adds[0][0][:7] if data_adds else 'none'} | no plan file | - | - | - | - | -")
            notes.append(f"{folder.name}: no PREREG file. The record has to say it was not preregistered.")
            continue
        first_plan = plans[0][0]
        with_plan = [f for h, d, s, files in data_adds if h == first_plan for f in files]
        earlier = [x for x in data_adds if is_ancestor(x[0], first_plan)]
        data_adds = [x for x in data_adds if x[0] != first_plan and not is_ancestor(x[0], first_plan)]
        first_data = data_adds[0][0] if data_adds else None
        before = not earlier and (first_data is None or is_ancestor(first_plan, first_data))
        if not before:
            bad += 1
        early = [x for x in plans[1:] if first_data is None or (x[0] != first_data and not is_ancestor(first_data, x[0]))]
        later = [x for x in plans[1:] if first_data and (x[0] == first_data or is_ancestor(first_data, x[0]))]
        followed = [x for x in later if any(is_ancestor(x[0], h) for h in data_changes)]
        readme = (folder / "README.md").read_text() if (folder / "README.md").exists() else ""
        named = [x for x in later if x[0][:7] in readme]
        prog = []
        if first_data:
            for h, d, s, files in hist:
                if h != first_data and is_ancestor(first_data, h):
                    prog += [(h, p) for st, p in files if st == "M" and p.endswith(".py")]
        print(f"{folder.name} | {first_plan[:7]} | {len(with_plan)} | {first_data[:7] if first_data else 'none yet'} | {'yes' if before else 'NO'} | {len(early)} | {len(later)} | {len(followed)} | {len(named)} of {len(later)} | {len(prog)}")
        if with_plan:
            notes.append(f"{folder.name}: {len(with_plan)} data files were committed with the first plan: {', '.join(sorted(Path(f).name for f in with_plan))}")
        for h, d, s, files in early:
            notes.append(f"{folder.name}: plan changed at {h[:7]} ({d[:16]}), before the first data was committed. {s[:90]}")
        for h, d, s, files in later:
            tag = "new data followed" if any(x[0] == h for x in followed) else "no data added or changed after it"
            said = "named in the README" if h[:7] in readme else "not named in the README by its commit"
            notes.append(f"{folder.name}: plan changed at {h[:7]} ({d[:16]}), {tag}, {said}. {s[:90]}")
        for h, p in prog:
            notes.append(f"{folder.name}: program {Path(p).name} changed at {h[:7]} after the first data")
    print()
    print("Data committed with a plan, each change to a plan, and each program changed after the first data")
    for n in notes:
        print("  " + n)
    if not notes:
        print("  none")
    print()
    print(f"records with data from before their first plan, or first data that does not descend from it: {bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
