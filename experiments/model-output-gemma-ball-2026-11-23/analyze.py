"""Validate Gemma rows and quantify stock-baseline and deployment impact."""

import argparse
from fractions import Fraction as Q
from hashlib import sha256
from math import sqrt
import json
from pathlib import Path
import re


PRIMARY = "gemma-4-E4B-it"
MODELS = (PRIMARY, "gemma-3-270m-it", "gemma-3-1b-it", "gemma-2-2b-it",
          "gemma-3-4b-it", "gemma-7b-it", "gemma-2-9b-it", "gemma-3-12b-it")
TEXTS = ("wikitext", "python")
PLAIN = "ZBG128:b560s16"
MATRIX = "h" + PLAIN
RAND = MATRIX + "~rand"
LOUD = PLAIN + "+K10"
REPAIR = MATRIX + "+K10"
PLAIN4, MATRIX4 = "ZBG128:b496s16", "hZBG128:b496s16"
PLAIN5, MATRIX5 = "ZBG128:b624s16", "hZBG128:b624s16"
Q4, Q4KM, Q4IMAT = "file:Q4_0", "file:Q4_K_M", "file:Q4_K_M-imat"
HADAMARD = "HBG128:b560s16"
COHORT = (Q4, Q4KM, Q4IMAT, PLAIN, MATRIX, RAND, LOUD, REPAIR,
          "EBG128:b560s16", "VE8Z32:16", HADAMARD)
CONTINUITY = (PLAIN4, MATRIX4, PLAIN5, MATRIX5, "ZB128@R:16", "ZB128@R:32")
BRIDGE = ("VE832:16", "VE8Z32:16")
REAL_IMAT = (Q4IMAT, MATRIX, REPAIR, MATRIX4, MATRIX5)
NEG_INF = object()
DB_FIELDS = ("weight_err_db_mean", "weight_err_db_pooled",
             "weight_err_db_mean_active", "weight_err_db_pooled_active")
UINT_FIELDS = ("matrices", "weights", "matrices_active", "weights_active", "kept_columns")
NUM_FIELDS = ("bits_per_weight", "bits_per_weight_active", "build_seconds",
              "share_out_of_range", "score_seconds", "kl_mean", "kl_mean_se", "same_top",
              "same_top_se", "ppl", "ppl_se", "ppl_base", "ppl_base_se", "ln_ppl_ratio",
              "ln_ppl_ratio_se", "kl_median", "kl_p99", "kl_max", "rms_dp", "rms_dp_se")


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def object_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON field: {key!r}")
        result[key] = value
    return result


def decimal(token):
    mantissa, _, exponent = token.lower().partition("e")
    require(len(mantissa) <= 256 and (not exponent or
            (len(exponent) <= 5 and abs(int(exponent)) <= 512)), "decimal outside runner range")
    return Q(token)


def constant(token):
    require(token == "-Infinity", f"illegal JSON constant: {token}")
    return NEG_INF


def read_json(text):
    return json.loads(text, parse_float=decimal, parse_constant=constant,
                      object_pairs_hook=object_pairs)


def is_sha(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def number(value):
    return type(value) in (int, Q)


def uint(value):
    return type(value) is int and value >= 0


def no_infinity(value):
    if isinstance(value, dict):
        return all(no_infinity(item) for item in value.values())
    if isinstance(value, list):
        return all(no_infinity(item) for item in value)
    return value is not NEG_INF


def validate_row(row):
    require(isinstance(row, dict), "row must be a JSON object")
    require(row.get("model_id") in MODELS, "unknown or missing model_id")
    text, arm = row.get("text"), row.get("variant")
    require(type(row.get("bridge")) is bool and
            text in (("bridge",) if row["bridge"] else TEXTS), "text/bridge mismatch")
    require(arm in (BRIDGE if row["bridge"] else COHORT + CONTINUITY),
            "variant outside preregistered text/bridge scope")
    require(not row["bridge"] or row["model_id"] == PRIMARY, "bridge requires primary model")
    require(is_sha(row.get("model_sha256")), "invalid model_sha256")
    require(is_sha(row.get("text_sha256")) or
            ("text_sha256" in row and row["text_sha256"] is None and "kl_mean" not in row),
            "invalid text_sha256")
    if arm in REAL_IMAT + (RAND,):
        require(is_sha(row.get("imatrix_sha256")), "missing or invalid imatrix_sha256")
    if arm.startswith("file:"):
        require(is_sha(row.get("file_sha256")), "missing or invalid file_sha256")
    for name in ("imatrix_sha256", "file_sha256"):
        if name in row:
            require(is_sha(row[name]), f"invalid {name}")
    require(type(row.get("ngl")) is int, "invalid ngl")
    require(isinstance(row.get("timestamp_utc"), str) and
            re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?\+00:00",
                         row["timestamp_utc"]), "invalid timestamp_utc")
    for name in UINT_FIELDS:
        require(uint(row.get(name)), f"invalid or missing {name}")
    require(0 <= row["matrices_active"] <= row["matrices"] and
            0 <= row["weights_active"] <= row["weights"], "active counts exceed total")
    for name in ("bits_per_weight", "bits_per_weight_active", "build_seconds", "share_out_of_range"):
        require(name in row, f"missing {name}")
    for name in NUM_FIELDS:
        if name in row:
            require(number(row[name]), f"invalid {name}")
            if name.endswith(("_se", "_seconds")) or name in (
                    "kl_mean", "bits_per_weight", "bits_per_weight_active", "share_out_of_range"):
                require(row[name] >= 0, f"negative {name}")
    for name in DB_FIELDS:
        require(name in row and (row[name] is NEG_INF or number(row[name])), f"invalid {name}")
    if "weight_err_db_by_kind" in row:
        by_kind = row["weight_err_db_by_kind"]
        require(isinstance(by_kind, dict) and all(v is NEG_INF or number(v) for v in by_kind.values()),
                "invalid weight_err_db_by_kind")
    if arm.startswith("file:") or "other_tensors_that_differ" in row:
        names = row.get("other_tensors_that_differ")
        require(isinstance(names, list) and all(isinstance(n, str) for n in names),
                "invalid other_tensors_that_differ")
    for name, value in row.items():
        if name not in DB_FIELDS + ("weight_err_db_by_kind",):
            require(no_infinity(value), f"illegal infinity in {name}")
    require(("kl_mean" in row) == ("kl_mean_se" in row), "KL and KL SE must appear together")
    require("kl_mean" in row or not any(k in row for k in ("same_top", "same_top_se", "score_seconds")),
            "score metadata without KL")
    require("same_top_se" not in row or "same_top" in row, "top-1 SE without top-1")
    require("same_top" not in row or 0 <= row["same_top"] <= 100, "top-1 outside [0,100]")


def load_rows(inputs):
    rows, provenance = {}, {}
    for raw in inputs:
        for lineno, line in enumerate(raw.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = read_json(line)
                validate_row(row)
                model, text, arm = row["model_id"], row["text"], row["variant"]
                require((model, text, arm) not in rows, "duplicate (model_id, text, variant)")
                pins = [(("model", model, row["bridge"]), row["model_sha256"]),
                        (("text", model, text), row["text_sha256"])]
                if arm in REAL_IMAT + (RAND,):
                    pins.append((("imatrix", model, arm == RAND), row["imatrix_sha256"]))
                if arm.startswith("file:"):
                    pins.append((("baseline", model, arm), row["file_sha256"]))
                for key, value in pins:
                    if value is not None:
                        require(provenance.setdefault(key, value) == value,
                                f"conflicting {key[0]} hash for {model}")
                rows[model, text, arm] = row
            except (ValueError, RecursionError) as exc:
                raise ValueError(f"JSONL line {lineno}: {exc}") from exc
    return rows


def load_transcribed_table(raw, model):
    matches = re.findall(r"(?ms)^```text\n(.*?)^```$", raw.decode("utf-8"))
    require(len(matches) == 1, "expected one fenced text table in transcription")
    rows = {}
    for line in matches[0].splitlines():
        parts = [field.strip() for field in line.split("|")]
        require(len(parts) == 7, "expected variant, text, bits, KL, KL SE, PPL and top-1")
        arm, text, *tokens = parts
        require(text in TEXTS and arm in COHORT + CONTINUITY, "invalid transcribed headline cell")
        key = model, text, arm
        require(key not in rows, "duplicate transcribed headline cell")
        require(all(re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", token) for token in tokens),
                "invalid transcribed numeric field")
        bits, kl, se, ppl, top = map(decimal, tokens)
        require(bits > 0 and ppl > 0 and 0 <= top <= 100, "invalid transcribed bits, PPL or top-1")
        rows[key] = dict(model_id=model, bits_per_weight=bits, kl_mean=kl, kl_mean_se=se,
                         ppl=ppl, same_top=top)
    require(len(rows) > 0, "empty transcription")
    return rows


FAMILIES = (("Plain", (PLAIN4, PLAIN, PLAIN5)), ("Real matrix", (MATRIX4, MATRIX, MATRIX5)))
CASES = (("Low", 7, 70, 25, 100, 25, 2), ("Mid", 27, 85, 50, 1000, 50, 3),
         ("High", 70, 95, 100, 10000, 75, 5))


def select(rows, model, text):
    return {arm: row for (m, t, arm), row in rows.items()
            if m == model and t == text and "kl_mean" in row}


def matched_rate(points, target):
    points = sorted(points)
    require(len(points) >= 2, "at least two rates required")
    require(all(x0 < x1 and y0 > y1 for (x0, y0), (x1, y1) in zip(points, points[1:])),
            "iso-quality needs strictly ordered, decreasing observed rates/KL")
    if target > points[0][1]:
        return ("bound", points[0][0])
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if y0 >= target >= y1:
            return ("interpolated", x0 + (x1 - x0) * Q(y0 - target, y0 - y1))
    return ("not reached", None)


def comparison(candidate, baseline, f16):
    gain = Q(baseline["kl_mean"]) - candidate["kl_mean"]
    base_kl = baseline["kl_mean"]
    variance = Q(candidate["kl_mean_se"]) ** 2 + Q(baseline["kl_mean_se"]) ** 2
    def ppl(row):
        return 100 * (Q(row["ppl"], f16) - 1) if "ppl" in row and f16 else None
    return {
        "gain": gain, "ratio": Q(candidate["kl_mean"], base_kl) if base_kl else None,
        "se_squared": variance, "screen": gain != 0 and gain ** 2 >= 4 * variance,
        "keep": (Q(candidate["same_top"]) - baseline["same_top"]
                 if "same_top" in candidate and "same_top" in baseline else None),
        "ppl": (ppl(candidate), ppl(baseline)),
    }


def scenario(baseline_bpw, anchor, specification):
    label, billions, coverage, transfer, fleet, eligible, hourly = specification
    params = billions * 10**9
    covered = Q(coverage, 100)
    new_bpw = baseline_bpw - anchor * Q(transfer, 100)
    before = params * (baseline_bpw * covered + 16 * (1 - covered)) / 8
    after = params * (new_bpw * covered + 16 * (1 - covered)) / 8
    reserve = 20 * 10**9
    extra_traffic = before / 5
    bandwidth = 48 * 10**11
    before_tps = bandwidth / (before + extra_traffic)
    after_tps = bandwidth / (after + extra_traffic)
    fleet_fraction = Q(eligible, 100) * (1 - before_tps / after_tps)
    return (label, before + reserve, after + reserve, (before - after) / (before + reserve),
            before_tps, after_tps, fleet_fraction, 8760 * hourly * fleet * fleet_fraction)


def fmt(value, digits=3, signed=False):
    return "—" if value is None else format(float(value), f"{'+' if signed else ''}.{digits}f")


def impact(rows, model, f16):
    lines = [f"# Descriptive impact calculations: {model}", "",
             "Positive KL gain and top-1 Δ favor the candidate. Comparisons are conditional on the inputs; "
             "KL SE uses marginal quadrature and does not establish paired significance. Top-1 and PPL have no SE here.", ""]
    anchors = []
    for text in TEXTS:
        cells = select(rows, model, text)
        lines.extend([f"## {text}", "",
                      "Stock comparisons ranked by absolute relative KL effect. PPL columns are percent change "
                      "from F16 (negative is lower PPL), candidate / baseline.", "",
                      "| Candidate → baseline | Actual block bpw (candidate / baseline) | KL ratio | KL gain | Gain / SE | ≥2 SE? | Top-1 Δ pp | PPL vs F16 (%) |",
                      "|---|---:|---:|---:|---:|---|---:|---:|"])
        pairs = [(a, Q4) for a in COHORT if not a.startswith("file:")] + [
            (a, b) for a in (PLAIN5, MATRIX5, "ZB128@R:32") for b in (Q4KM, Q4IMAT)]
        stock = [(a, b, comparison(cells[a], cells[b], f16.get(text)))
                 for a, b in pairs if a in cells and b in cells]
        def effect(item):
            ratio = item[2]["ratio"]
            return abs(1 - ratio) if ratio is not None else -1
        for a, b, c in sorted(stock, key=effect, reverse=True):
            variance = c["se_squared"]
            se = fmt(float(c["gain"]) / sqrt(variance), 2, True) if variance else (
                "undefined (zero SE)" if c["gain"] else "—")
            lines.append(f"| {a} → {b} | {fmt(cells[a]['bits_per_weight'], 4)} / "
                         f"{fmt(cells[b]['bits_per_weight'], 4)} | {fmt(c['ratio'])} | "
                         f"{fmt(c['gain'], 6, True)} | {se} | {'yes' if c['screen'] else 'no'} | "
                         f"{fmt(c['keep'], 3, True)} | {' / '.join(fmt(v, 2, True) for v in c['ppl'])} |")
        lines.extend(["", "### Matched KL (interpolation, not measured iso-quality)",
                      "", "| Family / target | Matched block bpw | Saved bpw | Block storage saved | Basis |",
                      "|---|---:|---:|---:|---|"])
        for family, arms in FAMILIES:
            for baseline in (Q4, Q4KM, Q4IMAT):
                if baseline not in cells or not all(arm in cells for arm in arms):
                    continue
                points = [(Q(cells[arm]["bits_per_weight"]), Q(cells[arm]["kl_mean"])) for arm in arms]
                kind, rate = matched_rate(points, cells[baseline]["kl_mean"])
                old = Q(cells[baseline]["bits_per_weight"])
                saved = old - rate if rate is not None else None
                if family == "Real matrix" and baseline == Q4IMAT and saved is not None:
                    anchors.append((old, saved))
                lines.append(f"| {family} / {baseline} | {fmt(rate, 4)} | "
                             f"{'≥' if kind == 'bound' else ''}{fmt(saved, 4)} | "
                             f"{'≥' if kind == 'bound' else ''}{fmt(100 * saved / old if saved is not None else None, 2)}% | {kind} |")
        lines.extend(["", "### Calibration contrasts",
                      "", "| Candidate → comparator | Δ block bpw | KL gain | Top-1 Δ pp | PPL Δ |",
                      "|---|---:|---:|---:|---:|"])
        for a, b in [(MATRIX, RAND)] + [(a, b) for a in (MATRIX, RAND)
                                       for b in (HADAMARD, "VE8Z32:16")]:
            if a not in cells or b not in cells:
                continue
            x, y = cells[a], cells[b]
            c = comparison(x, y, f16.get(text))
            ppl = Q(x["ppl"]) - y["ppl"] if "ppl" in x and "ppl" in y else None
            lines.append(f"| {a} → {b} | {fmt(Q(x['bits_per_weight']) - y['bits_per_weight'], 4, True)} | "
                         f"{fmt(c['gain'], 6, True)} | {fmt(c['keep'], 3, True)} | {fmt(ppl, 3, True)} |")
        lines.append("")
    if len(anchors) == len(TEXTS) and len({old for old, _ in anchors}) == 1:
        old = anchors[0][0]
        anchor = min(saved for _, saved in anchors)
        if anchor > 0:
            lines.extend(["## Illustrative deployment scenarios", "",
                          f"Anchor: {fmt(anchor, 4)} saved block bpw at the Q4_K_M-imat KL target "
                          "(smaller of two text estimates). These are assumptions, not a fleet forecast.", "",
                          "| Scenario | Resident GB (base → new) | Footprint reduction | H200 token/s roofline (base → new) | Fleet equivalent reduction | Annual gross opportunity |",
                          "|---|---:|---:|---:|---:|---:|"])
            for case in CASES:
                label, before, after, memory, start, end, fleet, dollars = scenario(old, anchor, case)
                lines.append(f"| {label} | {fmt(before / 10**9, 2)} → {fmt(after / 10**9, 2)} | "
                             f"{fmt(100 * memory, 2)}% | {fmt(start, 1)} → {fmt(end, 1)} | "
                             f"{fmt(100 * fleet, 2)}% | USD {float(dollars):,.0f} |")
            lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path, help="runner JSONL files")
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--transcribed-table", action="store_true", help="one rounded message table, not raw JSONL")
    parser.add_argument("--f16-wikitext", type=decimal)
    parser.add_argument("--f16-python", type=decimal)
    args = parser.parse_args(argv)
    try:
        raw = [path.read_bytes() for path in args.results]
        require(not args.transcribed_table or len(raw) == 1, "provide a single transcription")
        f16 = {"wikitext": args.f16_wikitext, "python": args.f16_python}
        require(all(value is None or value > 0 for value in f16.values()), "F16 PPL must be positive")
        rows = load_transcribed_table(raw[0], args.model) if args.transcribed_table else load_rows(raw)
        require(any(m == args.model and t in TEXTS for m, t, _ in rows), "no headline cells for model")
        report = impact(rows, args.model, f16)
    except (ValueError, OSError, RecursionError) as exc:
        parser.exit(2, f"gemma impact: {exc}\n")
    print(report)
    source = "transcription (raw receipts unverified)" if args.transcribed_table else "runner JSONL"
    print(f"Input: {source}. Local SHA-256: " + ", ".join(sha256(blob).hexdigest() for blob in raw))


if __name__ == "__main__":
    main()
