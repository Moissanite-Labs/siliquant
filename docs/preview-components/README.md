# Preview components

This folder holds the source for the Siliquant link card and the X post images. Its parts (headline, metrics strip, Hamming square, syndrome, background) can also be built into the site later.

These are source files for images. `docs/index.html` does not load them. GitHub Pages will serve this folder as static files, which does no harm.

## Outputs

`./finalize.sh` writes everything to `renders/`. That folder is git-ignored and can be rebuilt.

| File | Size | Use |
|---|---|---|
| `renders/og-a16-v2.png` | 1200×630, about 76 KB | Link card for `og:image` / `twitter:image` (summary_large_image). The site copy lives at `docs/assets/og-a16-v2.png`. |
| `renders/x-post-1600x900.png` | 16:9, about 110 KB | Image attached to the X announcement post. The only layout that names the three KL models |
| `renders/x-post-1080x1350.png` | 4:5, about 116 KB | Optional portrait variant |
| `renders/*.raw.png` | | Uncompressed renders, before pngquant |
| `renders/*.measure.json` | | Contrast per text box, measured on rendered pixels, plus layout checks |
| `renders/previews/` | | 506 px X-timeline size, X 2:1 crop, WhatsApp centre crop |
| `renders/*.html` | | The generated pages that get screenshotted |
| `renders/components-demo.html` | | Each component on its own |

If `--install-og` is passed, the script also copies the card to `../assets/og-a16-v2.png`. `docs/assets/` is the only asset source for the site, so the PNG the site uses lives there and nowhere else.

## Dependencies

You need four things:
- **Node 20 or later.**
- **Chrome or Chromium.** The script uses `$CHROME_PATH` if it is set, then `/usr/bin/google-chrome`. Otherwise it uses Playwright's own Chromium, which you install with `npx playwright-core install chromium`.
- **Fonts.** Inter and JetBrains Mono must be installed as system fonts. Both are OFL fonts from Google Fonts, and the site does not ship copies. Newsreader Display is loaded from the site's own `../assets/fonts/newsreader-display-500.woff2`; there is no second copy here. `render.mjs` stops with an error if any of the three would fall back to another font. Set `ALLOW_FALLBACK_FONTS=1` to render anyway, but the result will not match.
- **pngquant (optional).** It compresses the images. Without it, `renders/*.png` is the uncompressed render, and the 300 KB card limit still holds.

Install once (this needs the network):
```
cd docs/preview-components
npm ci                                   # playwright-core 1.65.1, pngjs 7.0.0 (pinned in package-lock.json)
npx playwright-core install chromium     # only if there is no system Chrome
```
After that the build runs offline. Pages load only `file://` URLs.

## Rerun
```
cd docs/preview-components
./finalize.sh                 # build, render, check, compress, previews
./finalize.sh --install-og    # the same, then copy the card to docs/assets/og-a16-v2.png
```
The script exits non-zero if any of these fail:
- text contrast is below 4.5:1;
- a text box overflows;
- text is outside the 64 px inset;
- on the card, key text is outside x 285–915 or the X 2:1 crop;
- the card is 300 KB or larger.

Reference build: Chrome 156 with playwright-core 1.65.1, pngquant 2.18. It reproduces the approved images byte for byte. Other Chrome versions can differ by a few anti-aliased pixels.

## Files
- `src/tokens.css`: the site's design tokens, copied verbatim from `docs/index.html` `:root` and `[data-theme="dark"]`, as `.dark` and `.light` classes. It also has the Newsreader Display `@font-face`. Use no other colours.
- `src/components.css`: the component classes, listed below.
- `src/components.mjs`: HTML builders: `page()`, `h1()`, `rows()`, `square()`, `synBits()`, and `METRICS_CSS`.
- `src/block.mjs`: the data for one real A16 block, taken from the page's `#chainFig`: Qwen2.5-1.5B release, layer 0 k_proj, row 0, block 0. It contains `Y`, `C`, `PARITY` (positions 0, 1, 2, 4, 8), `FLIP` (5), and `syndrome()`, which is the same code as the page's and as `check_a16_syndrome_errors`. It throws if the clean block doesn't check to 0 or if the flip doesn't locate to 5.
- `src/claims.mjs`: the claim strings, with a claim label for each number. These are the only numbers allowed on the images. They match the page's results strip, and each one cites its public evidence in the repo.
- `src/build.mjs`: the three image layouts and the demo, written to `renders/*.html`.
- `render.mjs`: render, measure and check. Run it as `node render.mjs <name> <w> <h>`.
- `previews.mjs`: makes the downscales and crops.
- `finalize.sh`: runs the whole pipeline.

## Reusing the components in the site
Components need an ancestor with the class `dark` or `light`. In the site, map that to `[data-theme]`, or alias the tokens, because the site's own variable names are identical. Elements with the class `t` are text, and `data-m` names a box for the contrast check. You can drop both in the site.

- **Headline (`.hl`).** Newsreader Display 500 with the same gradient as `.hero-title`: ink to accent, built from tokens through `--hl`. The site already has this on `.hero-title`, including the `@supports` solid fallback and the forced-colors rule, so reuse that rule there. Here `.hl` is the image-only copy.
- **Metrics strip (`.rows > .row > .big + .txt > .rest + .scope`, with `METRICS_CSS`).**
  - Layout: `.rows` is one grid and each `.row` is `display: contents`, so the big-number column lines up across all rows.
  - Numbers use `--win` in mono, and claims use `--text-primary`. Scope lines use `--text-muted` in Inter, at 5.8:1 or better on `--bg-page`.
  - Models line: `rows()` requires `o.models`, the font size in px of an extra line under the Quantization scope that names the three models (Qwen2.5-1.5B-Instruct · SmolLM2-1.7B-Instruct · Llama-3.2-1B-Instruct). Only the 1600×900 post image sets it (14 px). The og card and the portrait pass `0` because the line does not fit them, so they do not name the models.
  - Keep the scope line next to its number. The bit-flip number must always carry its qualifier (lowest bit of each 4-bit code; costs 3.1% tok/s). "0 extra bytes" describes the engine's packed layout only. The Hugging Face files rely on SHA-256 checks instead.
- **Hamming square (`.sq` + `.cell`, from `square({ showY })`).**
  - Layout: a 4×4 grid with index i = 4·row + col, sized by `--cell` and `--gap`.
  - Cell states: `.one` means the low bit is 1 (accent fill), `.par` is a parity position (accent outline), and `.x` is the flipped bit (fail colours). `.ix` is the corner index and `.y` is the lattice integer.
  - The crosshair: add `<div class="cross-r">` and `<div class="cross-c">` as the first children of `.sq` and position them for your `--cell`. Row 1: top = cell + gap − pad, height = cell + 2·pad. Column 1: left uses the same formula.
  - Give `.sq` `width: max-content` (or `inline-grid`) when its parent is wider than the square, so the crosshair spans only the square.
  - To make it interactive, as the page's Flip button does, re-render with a different `FLIP` and run `syndrome()` on the edited bits.
- **Syndrome (`.sb.ok` / `.sb.bad`, from `synBits(size)`).** These are the five bits `[p0 · s3 s2 s1 s0]`, where p0 = 1 means a single flip. The position is s3·8 + s2·4 + s1·2 + s0. Write the result as "▲ caught at 5" in `--fail`. Never call it correction: only detection runs live.
- **Background (`.glow` + `.field`).** `.glow` is an accent-soft radial glow. `.field` is a 24 px integer-lattice dot field in `--border-strong`, masked clear in the middle. For the site, keep the field behind a hero only, and add `aria-hidden="true"`.

## Claim rules
- Every number must come from `src/claims.mjs`, and must match the page's results strip and its public evidence: the page's Limits section (`#limits`), `experiments/hamming-lattices-real-weights-2026-11-24/README.md`, and `studies/aletheion-qwen25-live-llm-engine/ALETHEION_LIVE_LLM_ENGINE_REPORT.md`.
- The images do not show the model count. The page does: the results strip shows "6 models" with "4.5 bpw target" (`docs/index.html:1911`: `<span class="kpi-big">6 models</span><span class="sr-only">, </span><span class="kpi-rest">4.5 bpw target</span>`), the hero says "Six models are released with their full records." (line 1886), and the `#bench` table has one row for each model. To add it to an image, add a row to `ROWS` with its scope ("4.5 bpw target", and the strip's scope line "360M to 3B base models, embeddings in BF16; three Apache-2.0, three Qwen Research", `docs/index.html:1912`), and keep the ZeroCal "Transformers load test pending" note wherever the models are listed.
- Never write these: "free", "zero overhead", "error-correcting", or anything comparing against FP16, cuBLAS or llama.cpp.
