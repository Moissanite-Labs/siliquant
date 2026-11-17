# Checks on the records themselves

Three programs that read the repository, not weights. They are crossings X2, X3 and X4 of [notes/cross-org-grafts.md](../notes/cross-org-grafts.md), three of the five that note ranked first. They only read. None changes a record. (Crossing X1, a witness for every cell of a closure, is in `perm/`.)

| File | What it answers | Output |
|---|---|---|
| `plan_before_data.py` | Was each record's plan committed before its data? Read from git. | `plan_before_data.out` |
| `number_trace.py` | Can each number in a record's README be found in a committed output? | `number_trace.out` |
| `claim_ledger.py` | Does each row of a check note say what its verdict rests on, in words from a fixed list, with a file to open? | `claim_ledger.out` |

Run from the root of a checkout with full history:

```bash
python3 checks/plan_before_data.py > checks/plan_before_data.out
python3 checks/number_trace.py > checks/number_trace.out
python3 checks/claim_ledger.py > checks/claim_ledger.out
```

The first two outputs are of the tree at the commit that added them. The model-output run on Qwen2.5 had a plan and no data yet.

## What the first one found

It works. It exits with 1 if a record has data from before its first plan.

- **All seven records with data have their plan before it.** In each, the first commit of data descends from the first commit of the plan. The eighth has a plan and no data yet.
- **The gemma record changed its plan twice after its first data was committed**, at `e2ff142` (round 6) and `7961368` (fixes after a second read). New data followed the first of the two. Its README said "extended five times" and named neither commit. It names them now.
- **The plan of that record was changed six times before its first data was committed.** Its rounds 1 to 5 were committed together at `eac9d89`. So git shows that every plan came before the commit of the data, and cannot show that the plan of round 2 came before the results of round 2 were seen. The record says it did.
- **The record on issue #10 has three pilot files committed with its plan.** Its plan says so.
- Four records changed a program after their first data. The output lists each change.

What it cannot show: what was read or run before a commit. A plan committed first proves only that the plan existed first.

## What the second one found

**It does not do the job, and it measures by how much.**

It was built the plain way: take every number with a decimal point in a README, and look for a number in the record's outputs that rounds to it. Of 1,912 numbers, 2 are not found. That reads well and means little. An output file holds thousands of numbers, so a wrong number is found too. The program changes each number with two decimals or more by 3 in its last place and looks again: **1,211 of 1,642 changed numbers are still found**, 26 percent in the gemma record and 99 percent in the record on issue #10.

A second pass is stricter. It takes each line of a README with three numbers or more and asks for two lines of the record's `.out` files that hold them all. 202 of 331 lines pass. For those it changes one number by 3 in its last place: **the change is caught on 102 of 202**. The 129 lines that do not pass are mostly rows put together from more than two lines of output, or differences that no table prints.

So matching numbers after the fact cannot carry the rule that every number traces to an output. It catches a number that is nowhere at all, and about half of the slips within a table row.

What it is good for: the two lists in its output are where a second reader should start. The one number found nowhere, 0.746865, is a figure from a source and the record says so.

## What the third one found

It reads the tables of the three check notes as they stand: 105 rows. It keeps no second copy of the claims. It exits with 1 if a row breaks a rule.

- **Every part of every Basis is one of ten kinds.** Nothing had to be changed for that.
- **Eleven rows said "recomputed" or "proved" and named no file to open.** Eight were in the note on PR #6 and three in the note on the briefs. One of them is the row the grafts note found: claim 10 of the briefs cites 1.26/n and named only the folder `perm/`. The figure is in `perm/demo.out`, section 4. Each of the eleven rows names its file now.
- **The note on issue #10 passes by its legend.** It says once, above its tables, that "measured" means a committed output of one record. 14 of its rows rest on that sentence and name no file of their own. The program counts them apart.

What it cannot tell: whether the file named holds the number claimed.

## What follows

- The form that would carry the rule is the one X3 names: the tables of a README printed by a program from the outputs, and copied. Six records have such a program (`detail.py`, `summary.py`) for some of their tables. A record that printed all of them could be checked by comparing text, with no chance matches. That is a change to how a record is written, not a check that can be added afterwards.
- `plan_before_data.py` could run on every pull request. No workflow is added here.

## Disclosure

Written by an AI agent session under the owner's direction, on 2026-11-17.
