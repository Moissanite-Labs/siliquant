#!/usr/bin/env python3
"""Build the Siliquant research hub.

Reads experiment records on main and writes docs/data.json and
docs/index.html (rendered from scripts/hub_template.html).

Rules:
- Every number in the output is parsed from a source file in this repo
  or from a live GitHub PR body fetched at build time. Nothing is
  invented. A missing or unparsable record exits non-zero.
- Headline arms are found by parsing variant names in results.jsonl,
  never by hardcoding KL values. If a named arm is absent, exit non-zero.
- The Hadamard-turned E8 numbers come only from open PR #41's body and
  are labelled as claims from that open PR, never as merged main data.
- Roofline figures come from the bridge report's Pillar 2 table and are
  labelled as ceilings, never as measured throughput.

Usage:
    python3 scripts/build_site.py [--prs-json FILE] [--pr41-body FILE]
                                  [--fetch-prs] [--check]
    --fetch-prs   fetch the open PR list and PR #41's body from the
                  GitHub API (needs GITHUB_TOKEN in the environment).
    --check       verify docs/data.json against the source files
                  instead of writing output.
"""

import argparse
import html
import json
import os
import re
import sys
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = "Moissanite-Labs/siliquant"
GITHUB = "https://github.com/" + REPO
API = "https://api.github.com/repos/" + REPO

EM_DASH = "\u2014"


def fail(msg):
    print("build_site: ERROR: " + msg, file=sys.stderr)
    sys.exit(1)


def read_text(path):
    try:
        with open(path, encoding="utf-8") as fh:
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
    folder = os.path.join(REPO_ROOT, "experiments/imatrix-baseline-2026-11-17")
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
        REPO_ROOT, "experiments/model-output-qwen25-2026-11-17/results.jsonl"))
    smol = parse_jsonl(os.path.join(
        REPO_ROOT, "experiments/model-output-smollm2-2026-11-17/results.jsonl"))
    arms[0]["q4km_kl_mean"] = first_numeric(qwen, "file:qwen15-Q4_K_M.gguf")["kl_mean"]
    arms[1]["q4km_kl_mean"] = first_numeric(smol, "file:smollm2-Q4_K_M.gguf")["kl_mean"]
    # Llama-3.2: third-model wikitext record, which names its model per row.
    llama = parse_jsonl(os.path.join(
        REPO_ROOT, "experiments/third-model-llama32-2026-11-18/results-wikitext.jsonl"))
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
        REPO_ROOT, "experiments/model-output-gemma-e4b-2026-11-15/results.jsonl"))
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
        REPO_ROOT, "experiments/packed-ball-format-2026-11-18/results.jsonl"))
    pz = [r for r in rows if r["variant"] == "PZ32:b136s8"]
    if len(pz) != 2:
        fail("packed-ball-format: expected 2 PZ32:b136s8 rows, found %d" % len(pz))
    readme = read_text(os.path.join(
        REPO_ROOT, "experiments/packed-ball-format-2026-11-18/README.md"))
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


def roofline():
    """Pillar 2 roofline ceilings from the bridge report's table."""
    path = os.path.join(
        REPO_ROOT, "studies/expedition-siliquant-bridges-audit103r2/SILIQUANT_BRIDGE_REPORT.md")
    text = read_text(path)
    if "measured_kernel_tok_s_present: false" not in text:
        fail("bridge report does not state measured_kernel_tok_s_present: false")
    if "819,000,000,000 B/s" not in text:
        fail("bridge report does not state the 819 GB/s TPU v5e bandwidth")
    rows = {}
    for line in text.splitlines():
        cells = [re.sub(r"[`*]", "", c).strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 6:
            _name = cells[0].split(" (")[0]
            if cells[0].startswith("INT4"):
                _name = "INT4 (raw floor)"
            if _name not in ("FP16", "Q4_K_M", "ZBG128:b560s16",
                             "E8_Had", "INT4 (raw floor)"):
                continue
            name = _name
            try:
                rows[name] = {"bpw": float(cells[1].split("(")[-1].rstrip(")")),
                                  "qwen_ceiling_tok_s": float(cells[4]),
                                  "gemma_ceiling_tok_s": float(cells[5])}
            except ValueError:
                fail("bridge report roofline row unparsable: %r" % line)
    for name in ("FP16", "Q4_K_M", "ZBG128:b560s16", "E8_Had", "INT4 (raw floor)"):
        if name not in rows:
            fail("bridge report roofline table is missing the %s row" % name)
    return {"source": "studies/expedition-siliquant-bridges-audit103r2/SILIQUANT_BRIDGE_REPORT.md",
            "bandwidth_bytes_per_sec": 819000000000,
            "basis": "Part E exact rational arithmetic, TPU v5e, weight-only batch-1. "
                     "No kernel was measured.",
            "formats": rows}


def experiments_index():
    """Every experiments/*/ folder with title and record presence."""
    base = os.path.join(REPO_ROOT, "experiments")
    index = []
    for name in sorted(os.listdir(base)):
        folder = os.path.join(base, name)
        if not os.path.isdir(folder):
            continue
        readme_path = os.path.join(folder, "README.md")
        if not os.path.exists(readme_path):
            fail("experiments/%s has no README.md" % name)
        title = read_text(readme_path).splitlines()[0].lstrip("# ").strip()
        has_results = (os.path.exists(os.path.join(folder, "results.jsonl"))
                       or os.path.exists(os.path.join(folder, "results.json")))
        prereg = sorted(f for f in os.listdir(folder)
                        if f == "PREREG.md" or f.startswith("PREREG"))
        entry = {"name": name, "title": title,
                 "url": GITHUB + "/tree/main/experiments/" + name,
                 "has_readme": True,
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


def github_api(path):
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        fail("--fetch-prs needs GITHUB_TOKEN in the environment")
    req = urllib.request.Request(API + path,
                                 headers={"Authorization": "Bearer " + token,
                                          "Accept": "application/vnd.github+json",
                                          "User-Agent": "siliquant-hub-builder"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        fail("GitHub API request %s failed: %s" % (path, exc))


def parse_pr41_claims(body):
    """Hadamard-turned E8 KL numbers, claimed in open PR #41's body."""
    if not body:
        fail("PR #41 body is empty, cannot verify the Hadamard claims")
    m = re.search(
        r"KL 0\.(\d+), 0\.(\d+) and 0\.(\d+) \(Qwen2\.5, SmolLM2, Llama-3\.2\) "
        r"against 0\.(\d+), 0\.(\d+) and 0\.(\d+)", body)
    if not m:
        fail("PR #41 body does not contain the Hadamard KL claim sentence")
    nums = [float("0." + g) for g in m.groups()]
    models = ["Qwen2.5-1.5B-Instruct", "SmolLM2-1.7B-Instruct", "Llama-3.2-1B-Instruct"]
    return [{"model": model, "variant": "E8 turned by Hadamard (open PR #41 claim)",
             "kl_mean": kl, "q4km_kl_mean": q4}
            for model, kl, q4 in zip(models, nums[:3], nums[3:])]


def open_prs(prs_json=None, pr41_body=None, fetch=False):
    """Live open PR list plus the PR #41 Hadamard claims."""
    if fetch:
        raw = github_api("/pulls?state=open&per_page=100")
        prs = [{"number": p["number"], "title": p["title"],
                "url": p["html_url"],
                "body": p.get("body") or ""} for p in raw]
        body41 = next((p["body"] for p in prs if p["number"] == 41), None)
        if body41 is None:
            fail("PR #41 is not in the live open PR list")
    else:
        if not prs_json:
            fail("need --prs-json or --fetch-prs for the live open PR list")
        prs = json.loads(read_text(prs_json))
        body41 = pr41_body if pr41_body is not None else None
        if body41 is None:
            fail("need --pr41-body (or --fetch-prs) for the PR #41 claims")
    claims = parse_pr41_claims(body41)
    listed = [{"number": p["number"], "title": p["title"],
               "url": p.get("url") or p.get("html_url", ""),
               "summary": (p.get("body") or "").split("\n\n")[0][:280]}
              for p in prs]
    return listed, claims


def build_data(prs_json=None, pr41_body=None, fetch=False):
    data = {"generated_from": "scripts/build_site.py",
            "headline_gemma_e8": headline_gemma_e8(),
            "headline_integer_ball": headline_ball(),
            "headline_packed_file": headline_packed_file(),
            "roofline": roofline(),
            "experiments": experiments_index()}
    prs, claims = open_prs(prs_json, pr41_body, fetch)
    data["open_prs"] = prs
    data["hadamard_open_pr41_claims"] = claims
    data["hadamard_source"] = {"pr": 41, "url": GITHUB + "/pull/41",
                               "status": "open PR claim, not merged main data"}
    return data


def render(data):
    template = read_text(os.path.join(REPO_ROOT, "scripts/hub_template.html"))

    def row(cells, header=False):
        tag = "th" if header else "td"
        return "<tr>" + "".join("<%s>%s</%s>" % (tag, c, tag) for c in cells) + "</tr>"

    gemma = data["headline_gemma_e8"]
    gemma_rows = row(["Gemma E8, VE832:16, 4.5 bits", "%.6g" % gemma["kl_mean"],
                      "%.6g" % gemma["q4km_kl_mean"], "%.6g" % gemma["scalar_kl_mean"]])
    ball_rows = "".join(
        row([html.escape(a["model"]), "%.6g" % a["kl_mean"],
             "%.6g" % a["q4km_kl_mean"], "%.3f" % a["bits_per_weight"]])
        for a in data["headline_integer_ball"])
    packed_rows = "".join(
        row([html.escape(a["model"]), "%.6g" % a["kl_mean"], "%.3f" % a["bits_per_weight"]])
        for a in data["headline_packed_file"])
    had_rows = "".join(
        row([html.escape(a["model"]), "%.6g" % a["kl_mean"], "%.6g" % a["q4km_kl_mean"]])
        for a in data["hadamard_open_pr41_claims"])
    roof = data["roofline"]["formats"]
    roof_rows = "".join(
        row([html.escape(name), "%.2f" % roof[name]["bpw"],
             "%.1f" % roof[name]["qwen_ceiling_tok_s"],
             "%.1f" % roof[name]["gemma_ceiling_tok_s"]])
        for name in ("FP16", "Q4_K_M", "ZBG128:b560s16", "E8_Had", "INT4 (raw floor)"))
    exp_rows = "".join(
        row(['<a href="%s">%s</a>' % (e["url"], html.escape(e["name"])),
             html.escape(e["title"]),
             "yes" if e["has_prereg"] else "no",
             "yes" if e["has_results"] else "no",
             "yes" if e["has_results_out"] else "no"])
        for e in data["experiments"])
    pr_rows = "".join(
        row(['<a href="%s">#%d</a>' % (p["url"], p["number"]),
             html.escape(p["title"]),
             html.escape(p["summary"])])
        for p in data["open_prs"])

    rendered = (template
                .replace("{{GEMMA_ROWS}}", gemma_rows)
                .replace("{{BALL_ROWS}}", ball_rows)
                .replace("{{PACKED_ROWS}}", packed_rows)
                .replace("{{HADAMARD_ROWS}}", had_rows)
                .replace("{{ROOFLINE_ROWS}}", roof_rows)
                .replace("{{EXPERIMENT_ROWS}}", exp_rows)
                .replace("{{OPEN_PR_ROWS}}", pr_rows)
                .replace("{{EXPERIMENT_COUNT}}", str(len(data["experiments"])))
                .replace("{{OPEN_PR_COUNT}}", str(len(data["open_prs"])))
                .replace("{{DATA_JSON}}", html.escape(json.dumps(data, indent=2))))
    if EM_DASH in rendered or "stated plainly" in rendered.lower():
        fail("rendered hub contains an em dash or the banned phrase")
    for token in ("{{GEMMA_ROWS}}", "{{BALL_ROWS}}", "{{PACKED_ROWS}}",
                  "{{HADAMARD_ROWS}}", "{{ROOFLINE_ROWS}}", "{{EXPERIMENT_ROWS}}",
                  "{{OPEN_PR_ROWS}}", "{{EXPERIMENT_COUNT}}", "{{OPEN_PR_COUNT}}",
                  "{{DATA_JSON}}"):
        if token in rendered:
            fail("template placeholder %s was not replaced" % token)
    return rendered


def main():
    ap = argparse.ArgumentParser(description="Build the Siliquant research hub.")
    ap.add_argument("--prs-json", default=None,
                    help="JSON file with the live open PR list (from gh pr list)")
    ap.add_argument("--pr41-body", default=None,
                    help="file holding the live body text of PR #41")
    ap.add_argument("--fetch-prs", action="store_true",
                    help="fetch the open PR list and PR #41 body from the GitHub API")
    ap.add_argument("--check", action="store_true",
                    help="verify docs/data.json against the sources, do not write")
    args = ap.parse_args()

    pr41_body = read_text(args.pr41_body) if args.pr41_body else None
    if args.check:
        stored_path = os.path.join(REPO_ROOT, "docs/data.json")
        try:
            stored = json.loads(read_text(stored_path))
        except json.JSONDecodeError as exc:
            fail("docs/data.json does not parse: %s" % exc)
        fresh = build_data(args.prs_json, pr41_body, args.fetch_prs)
        for key in ("headline_gemma_e8", "headline_integer_ball",
                    "headline_packed_file", "roofline", "experiments"):
            if stored.get(key) != fresh.get(key):
                fail("--check: docs/data.json section %r differs from the sources" % key)
        if args.prs_json or args.fetch_prs:
            if [p["number"] for p in stored.get("open_prs", [])] != \
               [p["number"] for p in fresh.get("open_prs", [])]:
                fail("--check: open PR list in docs/data.json is stale")
            if stored.get("hadamard_open_pr41_claims") != fresh.get("hadamard_open_pr41_claims"):
                fail("--check: PR #41 Hadamard claims in docs/data.json differ from the live PR body")
        print("check ok: docs/data.json matches the sources")
        return

    data = build_data(args.prs_json, pr41_body, args.fetch_prs)
    rendered = render(data)
    data_path = os.path.join(REPO_ROOT, "docs/data.json")
    html_path = os.path.join(REPO_ROOT, "docs/index.html")
    with open(data_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(rendered)
    print("wrote %s (%d experiments, %d open PRs)"
          % (data_path, len(data["experiments"]), len(data["open_prs"])))
    print("wrote %s" % html_path)


if __name__ == "__main__":
    main()
