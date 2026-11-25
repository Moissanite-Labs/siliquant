// Reusable HTML builders. Each returns a string; style comes from tokens.css + components.css.
// Elements with class "t" are text (hidden in the contrast-measurement pass); data-m names a measured box.
// Boxes whose data-m starts with "w-" are motif, not claims: they may sit outside the og centre crop.
import { HEADLINE, ROWS } from './claims.mjs';
export { square, synBits, POS, SYN, FLIP, PARITY, Y, C, CF, syndrome } from './block.mjs';

// Full standalone page at a fixed pixel size. cssBase is relative to the HTML file's folder.
export const page = (theme, w, h, css, body, cssBase = '../src/') => `<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><link rel="stylesheet" href="${cssBase}tokens.css"><link rel="stylesheet" href="${cssBase}components.css"><style>
body{width:${w}px;height:${h}px} ${METRICS_CSS} ${css}</style></head><body class="${theme}">${body}
<script>document.fonts.ready.then(()=>{document.body.dataset.ready='1'})</script></body></html>`;

// Metrics strip layout: one shared grid so the big-number column lines up across rows.
export const METRICS_CSS = '.rows{display:grid;grid-template-columns:auto 1fr} .row{display:contents} .row>*{border-bottom:1px solid var(--border-hairline);display:flex;flex-direction:column;justify-content:center} .row>.big{flex-direction:row;align-items:center;justify-content:flex-start}';

// Headline: Newsreader Display 500 with the site's .hero-title gradient (class "hl").
export const h1 = (size, m = 'title') => `<h1 class="hl t" data-m="${m}" style="font-size:${size}px">${HEADLINE[0]}<br>${HEADLINE[1]}</h1>`;

// Metrics strip rows: big number | claim + scope line. Wrap in <div class="rows">.
// o.models = true adds the row's `models` line (where the row has one) under its scope line.
export const rows = (o) => ROWS.map((r, i) => `<div class="row"><div class="big t" data-m="big${i}" style="font-size:${o.big}px">${r.big}</div><div class="txt"><div class="rest t" data-m="rest${i}" style="font-size:${o.rest}px">${r.rest}</div><div class="scope t" data-m="scope${i}" style="font-size:${o.scope}px">${r.scope}</div>${o.models && r.models ? `<div class="scope models t" data-m="models${i}" style="font-size:${o.models}px">${r.models}</div>` : ''}</div></div>`).join('');
