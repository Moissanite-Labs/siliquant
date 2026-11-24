# Results report

`results.html` is the source for the archived GitHub Pages report and the PDF. Edit the report there, keeping the evidence status and commit-pinned links with each result. The earlier codebook study and its PDF remain in `docs/`.

Build both editions with Python 3 and a recent Chrome or Chromium:

```sh
python3 site/build_results.py
# If the browser has a different executable name:
python3 site/build_results.py --chrome chromium
```

Commit `docs/archive/results-2026-11-24.html` and `docs/siliquant-results.pdf` together. The builder leaves `docs/index.html` unchanged. Open the archived HTML or serve `docs/` locally with `python3 -m http.server --directory docs`. The PDF uses the same report body with a print layout; browser versions and fonts can change the rendering. The existing GitHub Pages source is `main:/docs`.
