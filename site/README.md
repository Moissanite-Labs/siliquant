# Results report

`results.html` is the source for the archived GitHub Pages report and the PDF. Edit the report there, keeping the evidence status and commit-pinned links with each result. The earlier codebook study and its PDF remain in `docs/`.

Build both editions with Python 3 and a recent Chrome or Chromium:

```sh
python3 site/build_results.py
# If the browser has a different executable name:
python3 site/build_results.py --chrome chromium
```

Commit `docs/archive/results-2026-11-24.html` and `docs/siliquant-results.pdf` together. The builder leaves `docs/index.html` unchanged. Open the archived HTML or serve `docs/` locally with `python3 -m http.server --directory docs`. The PDF uses the same report body with a print layout; browser versions and fonts can change the rendering. The existing GitHub Pages source is `main:/docs`.

## Research hub

`docs/hub.html` is a generated research hub: the measured headline arms, the Gemma 3 matched-KL savings table (PR #57), and indexes of `experiments/` and `studies/`, rendered from `site/hub_template.html`. `docs/data.json` carries the same data as JSON. The hub page loads nothing from outside: no CDN scripts, no embedded JSON.

```sh
python3 site/build_hub.py             # write docs/data.json and docs/hub.html
python3 site/build_hub.py --check     # verify both are current, write nothing
python3 -I -B site/test_build_hub.py  # builder tests (stdlib unittest, no network)
```

The builder reads only files in this repo and never uses the network. It also cross-checks the `#gemma-impact` table in `docs/index.html` against the parsed impact reports. A PR that changes experiment records, the Gemma impact tables, the studies list, or that table in `docs/index.html` must regenerate the hub with `python3 site/build_hub.py`; `--check` (run in CI by `.github/workflows/site-hub-check.yml`) fails until it does.
