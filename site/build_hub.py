#!/usr/bin/env python3
"""Build the Siliquant research hub.

Reads experiment records in this repo and writes docs/data.json and
docs/hub.html (rendered from site/hub_template.html). The curated
results report remains the Pages homepage at docs/index.html.

Rules:
- The builder never uses the network. Every number in the output is
  parsed from a source file in this repo. Nothing is invented. A
  missing or unparsable record exits non-zero.
- Headline arms are found by parsing variant names in results.jsonl,
  never by hardcoding KL values. If a named arm is absent, exit
  non-zero.
- The Hadamard-turned E8 numbers come from the merged main record and
  are cross-checked against its results.out table.
- The Gemma 3 matched-KL table is parsed from the PR #57 impact
  reports and cross-checked against the same table in
  docs/index.html; any disagreement exits non-zero.

Usage:
    python3 site/build_hub.py [--root DIR] [--check]
    --root DIR   read the repo at DIR and write docs/ under DIR
                 (default: the repo containing this script).
    --check      rebuild in memory and verify that docs/data.json and
                 docs/hub.html are byte-for-byte current; do not write.
"""

import argparse
import html
import json
import os
import re
import sys
from html.parser import HTMLParser

REPO = "Moissanite-Labs/siliquant"
GITHUB = "https://github.com/" + REPO

EM_DASH = "—"
MINUS = "−"  # U+2212, used by docs/index.html in the saved-% cells

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMPACT_DIR = os.path.join(
    "experiments/model-output-gemma-ball-2026-11-23/reports/impact")
IMPACT_MODELS = (("270m", "270M"), ("1b", "1B"), ("4b", "4B"))
IMPACT_TEXTS = (("wikitext", "WikiText"), ("python", "Python"))


def fail(msg):
    print("build_hub: ERROR: " + msg, file=sys.stderr)
    sys.exit(1)


def read_text(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except OSError as exc:
        fail("cannot read %s: %s" % (path, exc))


def read_bytes(path):
    try:
        with open(path, "rb") as fh:
            return fh.read()
    except OSError as exc:
        fail("cannot read %s: %s" % (path, exc))


def parse_jsonl(path):
    """Parse a results.jsonl file. Every non-empty line must be valid JSON
    with a string variant. kl_mean, when present, must be numeric."""
    rows = []
    for lineno, line in enumerate(read_text(path).splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            fail("%s line %d is not valid JSON: %s" % (path, lineno, exc))
        if not isinstance(row, dict) or not isinstance(row.get("variant"), str):
            fail("%s line %d has no string variant: %r" % (path, lineno, line[:120]))
        if "kl_mean" in row and row["kl_mean"] is not None:
            if not isinstance(row["kl_mean"], (int, float)):
                fail("%s line %d: kl_mean is not a number" % (path, lineno))
        rows.append(row)
    if not rows:
        fail("%s has no rows" % path)
    return rows


def first_numeric(rows, variant):
    """First row with this variant name and a numeric kl_mean."""
    for row in rows:
        if row["variant"] == variant and isinstance(row.get("kl_mean"), (int, float)):
            return row
    fail("variant %r not found with a numeric kl_mean" % variant)


def parse_results_out_tables(path):
    """Parse the per-model summary tables in a results.out file.

    Returns {(model, variant): kl_mean} using the model headings that
    precede each table. Used to cross-check results.jsonl rows that
    carry no model field of their own.
    """
    out = {}
    model = None
    for line in read_text(path).splitlines():
        if line.startswith("Qwen2.5-1.5B-Instruct"):
            model = "Qwen2.5-1.5B-Instruct"
            continue
        if line.startswith("SmolLM2-1.7B-Instruct"):
            model = "SmolLM2-1.7B-Instruct"
            continue
        if model and " | " in line:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 5 and parts[0] and parts[0] != "variant":
                try:
                    out[(model, parts[0])] = float(parts[3])
                except ValueError:
                    pass
    return out


def headline_ball():
    """Integer ball arm hZBG128:b560s16 on three models, wikitext text."""
    arms = []
    # Qwen2.5 and SmolLM2: imatrix-baseline results.jsonl, cross-checked
    # against the per-model tables in its results.out.
    folder = os.path.join(ROOT, "experiments/imatrix-baseline-2026-11-17")
    rows = parse_jsonl(os.path.join(folder, "results.jsonl"))
    tables = parse_results_out_tables(os.path.join(folder, "results.out"))
    ball_rows = [r for r in rows if r["variant"] == "hZBG128:b560s16"]
    if len(ball_rows) != 2:
        fail("imatrix-baseline: expected 2 hZBG128:b560s16 rows, found %d" % len(ball_rows))
    for model, row in (("Qwen2.5-1.5B-Instruct", ball_rows[0]),
                       ("SmolLM2-1.7B-Instruct", ball_rows[1])):
        expected = tables.get((model, "hZBG128:b560s16"))
        if expected is None:
            fail("imatrix-baseline results.out has no hZBG128 row for %s" % model)
        if abs(row["kl_mean"] - expected) > 1e-6:
            fail("imatrix-baseline hZBG128 KL %r disagrees with results.out %r for %s"
                 % (row["kl_mean"], expected, model))
        arms.append({"model": model, "variant": "hZBG128:b560s16",
                     "kl_mean": row["kl_mean"],
                     "bits_per_weight": row.get("bits_per_weight")})
    # Q4_K_M baselines from the per-model output records.
    qwen = parse_jsonl(os.path.join(
        ROOT, "experiments/model-output-qwen25-2026-11-17/results.jsonl"))
    smol = parse_jsonl(os.path.join(
        ROOT, "experiments/model-output-smollm2-2026-11-17/results.jsonl"))
    arms[0]["q4km_kl_mean"] = first_numeric(qwen, "file:qwen15-Q4_K_M.gguf")["kl_mean"]
    arms[1]["q4km_kl_mean"] = first_numeric(smol, "file:smollm2-Q4_K_M.gguf")["kl_mean"]
    # Llama-3.2: third-model wikitext record, which names its model per row.
    llama = parse_jsonl(os.path.join(
        ROOT, "experiments/third-model-llama32-2026-11-18/results-wikitext.jsonl"))
    ball = first_numeric(llama, "hZBG128:b560s16")
    q4km = first_numeric(llama, "file:llama32-Q4_K_M.gguf")
    if "Llama" not in str(ball.get("model", "")):
        fail("third-model hZBG128 row does not name a Llama model")
    arms.append({"model": "Llama-3.2-1B-Instruct", "variant": "hZBG128:b560s16",
                 "kl_mean": ball["kl_mean"],
                 "bits_per_weight": ball.get("bits_per_weight"),
                 "q4km_kl_mean": q4km["kl_mean"]})
    return arms


def headline_gemma_e8():
    """E8 on Gemma at 4.5 bits, with the scalar arm and Q4_K_M beside it."""
    rows = parse_jsonl(os.path.join(
        ROOT, "experiments/model-output-gemma-e4b-2026-11-15/results.jsonl"))
    e8 = first_numeric(rows, "VE832:16")
    scalar = first_numeric(rows, "X32:16")
    q4km = first_numeric(rows, "file:Q4_K_M.gguf")
    return {"model": "gemma-4-E4B-it", "variant": "VE832:16",
            "kl_mean": e8["kl_mean"], "bits_per_weight": e8.get("bits_per_weight"),
            "scalar_kl_mean": scalar["kl_mean"],
            "scalar_variant": "X32:16",
            "q4km_kl_mean": q4km["kl_mean"]}


def headline_packed_file():
    """Packed ball file PZ32:b136s8, cross-checked against its README table."""
    rows = parse_jsonl(os.path.join(
        ROOT, "experiments/packed-ball-format-2026-11-18/results.jsonl"))
    pz = [r for r in rows if r["variant"] == "PZ32:b136s8"]
    if len(pz) != 2:
        fail("packed-ball-format: expected 2 PZ32:b136s8 rows, found %d" % len(pz))
    readme = read_text(os.path.join(
        ROOT, "experiments/packed-ball-format-2026-11-18/README.md"))
    m = re.search(r"\| `PZ32:b136s8`, from the file \| 4\.500 \| ([\d.]+), [\d.]+% \| ([\d.]+), [\d.]+% \|",
                  readme)
    if not m:
        fail("packed-ball-format README table row for PZ32:b136s8 not found")
    for row, expected in ((pz[0], float(m.group(1))), (pz[1], float(m.group(2)))):
        if round(row["kl_mean"], 4) != expected:
            fail("packed-ball-format PZ32 KL %r does not match README %r"
                 % (row["kl_mean"], expected))
    return [{"model": "Qwen2.5-1.5B-Instruct", "variant": "PZ32:b136s8",
             "kl_mean": pz[0]["kl_mean"], "bits_per_weight": pz[0].get("bits_per_weight")},
            {"model": "SmolLM2-1.7B-Instruct", "variant": "PZ32:b136s8",
             "kl_mean": pz[1]["kl_mean"], "bits_per_weight": pz[1].get("bits_per_weight")}]


def experiments_index():
    """Every experiments/*/ folder with title and record presence."""
    base = os.path.join(ROOT, "experiments")
    index = []
    for name in sorted(os.listdir(base)):
        folder = os.path.join(base, name)
        if not os.path.isdir(folder):
            continue
        readme_path = os.path.join(folder, "README.md")
        has_readme = os.path.exists(readme_path)
        title = (read_text(readme_path).splitlines()[0].lstrip("# ").strip()
                 if has_readme else name + " (no README)")
        has_results = (os.path.exists(os.path.join(folder, "results.jsonl"))
                       or os.path.exists(os.path.join(folder, "results.json")))
        prereg = sorted(f for f in os.listdir(folder)
                        if f == "PREREG.md" or f.startswith("PREREG"))
        entry = {"name": name, "title": title,
                 "url": GITHUB + "/tree/main/experiments/" + name,
                 "has_readme": has_readme,
                 "has_prereg": bool(prereg),
                 "prereg_files": prereg,
                 "has_results": has_results,
                 "has_results_out": os.path.exists(os.path.join(folder, "results.out")),
                 "has_results_jsonl": os.path.exists(os.path.join(folder, "results.jsonl"))}
        # Any results.jsonl present must parse.
        if entry["has_results_jsonl"]:
            parse_jsonl(os.path.join(folder, "results.jsonl"))
        index.append(entry)
    if not index:
        fail("no experiment folders found under experiments/")
    return index


def headline_hadamard():
    """Hadamard-turned E8 at 4.5 bits, from the merged record on main.

    The KL values are read from results.jsonl and cross-checked against
    the per-model summary table in results.out. Q4_K_M comes from that
    same table. No KL value is written into this script.
    """
    folder = os.path.join(ROOT, "experiments/e8-hadamard-real-weights-2026-11-22")
    rows = parse_jsonl(os.path.join(folder, "results.jsonl"))
    out_text = read_text(os.path.join(folder, "results.out"))
    models = [("Qwen2.5-1.5B-Instruct", "Qwen2.5-1.5B", "qwen2.5-1.5b"),
              ("SmolLM2-1.7B-Instruct", "SmolLM2-1.7B", "smollm2-1.7b"),
              ("Llama-3.2-1B-Instruct", "Llama-3.2-1B", "llama-3.2-1b")]

    def out_kl(prefix, bits, arm, variant):
        needle = prefix + " | " + bits + " | none | " + arm + " |"
        matches = [ln for ln in out_text.splitlines()
                   if ln.startswith(needle)
                   and len(ln.split("|")) > 7
                   and ln.split("|")[4].strip() == variant]
        if len(matches) != 1:
            fail("turned-E8 results.out has %d rows for %r with variant %r"
                 % (len(matches), needle, variant))
        cells = [c.strip() for c in matches[0].split("|")]
        try:
            return float(cells[7])
        except (ValueError, IndexError):
            fail("turned-E8 results.out row is unparsable: %r" % matches[0])

    arms = []
    for name, prefix, model_key in models:
        found = [r for r in rows if r["variant"] == "HBG128:b560s16"
                 and model_key in str(r.get("model", "")).lower()]
        if len(found) != 1:
            fail("turned-E8 results.jsonl has %d HBG128:b560s16 rows for %s"
                 % (len(found), name))
        kl = found[0]["kl_mean"]
        expected = out_kl(prefix, "4.5", "E8 turned", "HBG128:b560s16")
        if abs(kl - expected) > 1e-6:
            fail("turned-E8 KL %r disagrees with results.out %r for %s"
                 % (kl, expected, name))
        arms.append({"model": name, "variant": "HBG128:b560s16",
                     "kl_mean": kl,
                     "q4km_kl_mean": out_kl(
                         prefix, "4.8", "llama.cpp Q4_K_M, no matrix", "file")})
    return arms


def md_table_rows(line):
    """Split a markdown table row into stripped cells, or None."""
    if not line.lstrip().startswith("|"):
        return None
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    if all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
        return None
    return cells


def gemma_matched_kl():
    """Matched-KL block-storage rows from the PR #57 impact reports.

    For each Gemma 3 model (270m, 1b, 4b) and each text section
    (wikitext, python) of reports/impact/gemma-3-*-it/IMPACT_TABLES.md,
    take the 'Real matrix / file:Q4_K_M-imat' row of the Matched KL
    table, plus the Q4_K_M-imat actual block bpw from the first stock
    table row whose candidate is 'hZBG128:b624s16 -> file:Q4_K_M-imat'.
    Any missing or unparsable row exits non-zero.
    """
    out = []
    for slug, label in IMPACT_MODELS:
        rel = os.path.join(IMPACT_DIR, "gemma-3-%s-it" % slug, "IMPACT_TABLES.md")
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            fail("missing impact report: %s" % rel)
        text = None
        matched = None
        stock = {}
        for line in read_text(path).splitlines():
            heading = re.match(r"^##\s+(\S+)", line)
            if heading:
                text = heading.group(1)
                matched = None
                continue
            if re.match(r"^###\s+Matched KL", line):
                matched = True
                continue
            if re.match(r"^###", line):
                matched = False
                continue
            cells = md_table_rows(line)
            if cells is None or text is None:
                continue
            if matched:
                if cells[0] == "Real matrix / file:Q4_K_M-imat":
                    try:
                        matched_bpw = float(cells[1])
                        pct = cells[3]
                        if pct.endswith("%"):
                            pct = pct[:-1].strip()
                        saved_pct = float(pct)
                        basis = cells[4]
                    except (ValueError, IndexError):
                        fail("%s: Matched KL row unparsable: %r" % (rel, line))
                    if text not in stock:
                        fail("%s: no stock Q4_K_M-imat bpw row before the Matched KL "
                             "table in section %r" % (rel, text))
                    out.append({"model": label, "text": dict(IMPACT_TEXTS)[text],
                                "matched_bpw": matched_bpw,
                                "stock_bpw": stock[text],
                                "saved_pct": saved_pct,
                                "basis": basis,
                                "source": rel})
            else:
                if cells[0] == "hZBG128:b624s16 → file:Q4_K_M-imat" and text not in stock:
                    bpw_cell = cells[1] if len(cells) > 1 else ""
                    parts = [p.strip() for p in bpw_cell.split("/")]
                    try:
                        stock[text] = float(parts[1])
                    except (ValueError, IndexError):
                        fail("%s: stock bpw cell unparsable: %r" % (rel, line))
    expected = len(IMPACT_MODELS) * len(IMPACT_TEXTS)
    if len(out) != expected:
        fail("gemma matched-KL: expected %d rows, parsed %d" % (expected, len(out)))
    return out


def clean_study_title(heading):
    """Clean a studies README '# ' heading for plain-text display: remove
    backticks, remove '$' signs, replace \\mathbb{X} and \\mathfrak{X} with
    X, replace \\times with x, and collapse whitespace."""
    text = heading.replace("`", "").replace("$", "")
    text = re.sub(r"\\mathbb\{([^{}]*)\}", r"\1", text)
    text = text.replace("\\times", "x")
    text = re.sub(r"\\mathfrak\{([^{}]*)\}", r"\1", text)
    return " ".join(text.split())


def studies_index():
    """Every studies/*/ folder with a README.md, titled by its first '# ' heading."""
    base = os.path.join(ROOT, "studies")
    out = []
    for name in sorted(os.listdir(base)):
        folder = os.path.join(base, name)
        readme_path = os.path.join(folder, "README.md")
        if not (os.path.isdir(folder) and os.path.exists(readme_path)):
            continue
        title = None
        for line in read_text(readme_path).splitlines():
            if line.startswith("# "):
                title = clean_study_title(line[2:].strip())
                break
        if not title:
            title = name
        out.append({"name": name, "title": title,
                    "url": GITHUB + "/tree/main/studies/" + name})
    if not out:
        fail("no study folders with a README found under studies/")
    return out


class IndexGemmaTableParser(HTMLParser):
    """Collect the tbody rows of the table inside the h4#gemma-impact block."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.in_block = False
        self.in_table = False
        self.in_row = False
        self.in_cell = False
        self.cell_tag = None
        self.rows = []
        self.row = None
        self.cell = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "h4" and attrs.get("id") == "gemma-impact":
            self.in_block = True
        elif self.in_block and tag == "table":
            self.in_table = True
        elif self.in_table and tag == "tr":
            self.in_row = True
            self.row = []
        elif self.in_row and tag in ("td", "th"):
            self.in_cell = True
            self.cell_tag = tag
            self.cell = []

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.in_cell:
            self.row.append((self.cell_tag, "".join(self.cell).strip()))
            self.in_cell = False
        elif tag == "tr" and self.in_row:
            self.rows.append(self.row)
            self.in_row = False
        elif tag == "table" and self.in_table:
            self.in_table = False
            self.in_block = False

    def handle_data(self, data):
        if self.in_cell:
            self.cell.append(data)


def index_gemma_rows():
    """The six (model, text, matched, stock, saved) rows of the #gemma-impact table."""
    path = os.path.join(ROOT, "docs/index.html")
    if not os.path.exists(path):
        fail("missing %s" % path)
    parser = IndexGemmaTableParser()
    parser.feed(read_text(path))
    rows = []
    for row in parser.rows:
        cells = [text for tag, text in row if tag == "td"]
        if len(cells) != 4:
            continue
        if "·" not in cells[0]:
            continue
        model, text = [p.strip() for p in cells[0].split("·", 1)]
        rows.append((model, text, cells[1], cells[2], cells[3]))
    if len(rows) != 6:
        fail("docs/index.html: expected 6 rows in the #gemma-impact table, found %d"
             % len(rows))
    return rows


def cross_check_index(kl_rows):
    """Fail unless the #gemma-impact table in docs/index.html equals the
    parsed IMPACT_TABLES rows (bpw as 4-decimal strings, percent as
    2-decimal; the page uses U+2212 for the saved percent)."""
    page_rows = index_gemma_rows()
    for entry, (model, text, matched, stock, saved) in zip(kl_rows, page_rows):
        where = "%s %s" % (entry["model"], entry["text"])
        if model != entry["model"] or text != entry["text"]:
            fail("docs/index.html #gemma-impact row order mismatch at %s: page has "
                 "%s · %s" % (where, model, text))
        if matched != "%.4f" % entry["matched_bpw"]:
            fail("docs/index.html #gemma-impact %s: matched bpw %r != parsed %s"
                 % (where, matched, "%.4f" % entry["matched_bpw"]))
        if stock != "%.4f" % entry["stock_bpw"]:
            fail("docs/index.html #gemma-impact %s: stock bpw %r != parsed %s"
                 % (where, stock, "%.4f" % entry["stock_bpw"]))
        saved_clean = saved.replace("%", "").strip()
        if not saved_clean.startswith(MINUS):
            fail("docs/index.html #gemma-impact %s: saved cell %r does not use U+2212"
                 % (where, saved))
        digits = saved_clean[len(MINUS):]
        if digits != "%.2f" % entry["saved_pct"]:
            fail("docs/index.html #gemma-impact %s: saved %% %r != parsed %s"
                 % (where, digits, "%.2f" % entry["saved_pct"]))


def build_data():
    data = {"generated_from": "site/build_hub.py",
            "headline_gemma_e8": headline_gemma_e8(),
            "headline_integer_ball": headline_ball(),
            "headline_packed_file": headline_packed_file(),
            "headline_hadamard": headline_hadamard(),
            "experiments": experiments_index(),
            "gemma_matched_kl": gemma_matched_kl(),
            "studies": studies_index()}
    data["hadamard_source"] = {
        "record": "experiments/e8-hadamard-real-weights-2026-11-22",
        "url": GITHUB + "/tree/main/experiments/e8-hadamard-real-weights-2026-11-22",
        "pr": 41, "status": "merged main record, landed through PR #41"}
    return data


def render(data):
    template = read_text(os.path.join(ROOT, "site/hub_template.html"))

    def esc(value):
        return html.escape(str(value), quote=True)

    def row(cells, header=False):
        tag = "th" if header else "td"
        return "<tr>" + "".join("<%s>%s</%s>" % (tag, c, tag) for c in cells) + "</tr>"

    gemma = data["headline_gemma_e8"]
    gemma_rows = row(["Gemma E8, VE832:16, 4.5 bits", esc("%.6g" % gemma["kl_mean"]),
                      esc("%.6g" % gemma["q4km_kl_mean"]), esc("%.6g" % gemma["scalar_kl_mean"])])
    ball_rows = "".join(
        row([esc(a["model"]), esc("%.6g" % a["kl_mean"]),
             esc("%.6g" % a["q4km_kl_mean"]), esc("%.3f" % a["bits_per_weight"])])
        for a in data["headline_integer_ball"])
    packed_rows = "".join(
        row([esc(a["model"]), esc("%.6g" % a["kl_mean"]), esc("%.3f" % a["bits_per_weight"])])
        for a in data["headline_packed_file"])
    had_rows = "".join(
        row([esc(a["model"]), esc("%.6g" % a["kl_mean"]), esc("%.6g" % a["q4km_kl_mean"])])
        for a in data["headline_hadamard"])
    exp_rows = "".join(
        row(['<a href="%s">%s</a>' % (e["url"], esc(e["name"])),
             esc(e["title"]),
             "yes" if e["has_readme"] else "no",
             "yes" if e["has_prereg"] else "no",
             "yes" if e["has_results"] else "no",
             "yes" if e["has_results_out"] else "no"])
        for e in data["experiments"])
    kl_rows = "".join(
        row([esc("%s · %s" % (r["model"], r["text"])),
             esc("%.4f" % r["matched_bpw"]),
             esc("%.4f" % r["stock_bpw"]),
             esc("%s%.2f%%" % (MINUS, r["saved_pct"])),
             esc(r["basis"]),
             '<a href="%s">%s</a>' % (GITHUB + "/blob/main/" + r["source"],
                                      esc("IMPACT_TABLES.md"))])
        for r in data["gemma_matched_kl"])
    study_rows = "".join(
        row(['<a href="%s">%s</a>' % (s["url"], esc(s["name"])), esc(s["title"])])
        for s in data["studies"])

    rendered = (template
                .replace("{{GEMMA_ROWS}}", gemma_rows)
                .replace("{{BALL_ROWS}}", ball_rows)
                .replace("{{PACKED_ROWS}}", packed_rows)
                .replace("{{HADAMARD_ROWS}}", had_rows)
                .replace("{{EXPERIMENT_ROWS}}", exp_rows)
                .replace("{{EXPERIMENT_COUNT}}", esc(len(data["experiments"])))
                .replace("{{GEMMA_KL_ROWS}}", kl_rows)
                .replace("{{STUDIES_ROWS}}", study_rows)
                .replace("{{STUDIES_COUNT}}", esc(len(data["studies"]))))
    if EM_DASH in rendered or "stated plainly" in rendered.lower():
        fail("rendered hub contains an em dash or the banned phrase")
    if "{{" in rendered:
        fail("rendered hub still contains a {{ placeholder")
    for token in ("{{GEMMA_ROWS}}", "{{BALL_ROWS}}", "{{PACKED_ROWS}}",
                  "{{HADAMARD_ROWS}}",
                  "{{EXPERIMENT_ROWS}}", "{{EXPERIMENT_COUNT}}",
                  "{{GEMMA_KL_ROWS}}", "{{STUDIES_ROWS}}", "{{STUDIES_COUNT}}"):
        if token in rendered:
            fail("template placeholder %s was not replaced" % token)
    return rendered


def dump_data(data):
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main():
    global ROOT
    ap = argparse.ArgumentParser(description="Build the Siliquant research hub.")
    ap.add_argument("--root", default=None,
                    help="repo root to read and write under (default: this script's repo)")
    ap.add_argument("--check", action="store_true",
                    help="verify docs/data.json and docs/hub.html are current, do not write")
    args = ap.parse_args()
    if args.root:
        ROOT = os.path.abspath(args.root)

    data = build_data()
    cross_check_index(data["gemma_matched_kl"])
    fresh_json = dump_data(data)
    fresh_html = render(data)

    if args.check:
        for name, fresh in (("docs/data.json", fresh_json), ("docs/hub.html", fresh_html)):
            stored = read_bytes(os.path.join(ROOT, name)).decode("utf-8")
            if stored != fresh:
                fail("--check: %s is stale; run python3 site/build_hub.py" % name)
        print("check ok: docs/data.json and docs/hub.html match the sources")
        return

    out_dir = os.path.join(ROOT, "docs")
    os.makedirs(out_dir, exist_ok=True)
    data_path = os.path.join(out_dir, "data.json")
    html_path = os.path.join(out_dir, "hub.html")
    with open(data_path, "w", encoding="utf-8") as fh:
        fh.write(fresh_json)
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(fresh_html)
    print("wrote %s (%d experiments, %d studies, %d matched-KL rows)"
          % (data_path, len(data["experiments"]), len(data["studies"]),
             len(data["gemma_matched_kl"])))
    print("wrote %s" % html_path)


if __name__ == "__main__":
    main()
