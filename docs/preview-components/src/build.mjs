// Builds the three approved preview layouts into renders/*.html. Run via ../finalize.sh.
//   og-a16-v3.html        1200x630  link card (summary_large_image) -> docs/assets/og-a16-v3.png
//   x-post-1600x900.html  1600x900  image attached to the X announcement post
//   x-post-1080x1350.html 1080x1350 optional portrait variant
//   components-demo.html  each component on its own, for reuse in the site
import fs from 'node:fs';
import { page, h1, rows, square, synBits, POS } from './components.mjs';
import { EYEBROW, URL_TXT } from './claims.mjs';
const dir = new URL('../renders/', import.meta.url);
fs.mkdirSync(dir, { recursive: true });
const out = (n, h) => fs.writeFileSync(new URL(n, dir), h);

/* ---------- Link card (og:image / twitter:image), 1200x630. Key text stays in x 285–915 (WhatsApp centre crop). ---------- */
const ogB = theme => page(theme, 1200, 630, `
.stack{position:absolute;left:301px;width:598px;top:0;bottom:0;display:flex;flex-direction:column;justify-content:center}
.eyebrow{font-size:15px;margin-bottom:14px}
.stack .scope{white-space:nowrap}
.rows{margin-top:28px;border-top:1px solid var(--border-hairline)}
.row>*{padding:11px 0}.row>.big{padding-right:24px}
.txt .scope{margin-top:4px}
.url{font-size:14px;margin-top:16px}
.wl{position:absolute;left:72px;top:50%;transform:translateY(-50%);--cell:40px;--gap:6px}
.wr{position:absolute;left:946px;top:50%;transform:translateY(-50%);width:190px}
.wl .cap,.wr .cap{font-family:var(--font-mono);font-size:12px;color:var(--text-muted);letter-spacing:.06em;text-transform:uppercase;font-weight:600}
.wl .cap{margin-bottom:12px} .wl .note{margin-top:12px;font-family:var(--font-mono);font-size:13px;color:var(--fail);font-weight:600}
.cell{font-size:17px;padding-top:6px}
.cross-r{left:-5px;right:-5px;top:41px;height:50px}.cross-c{top:-5px;bottom:-5px;left:41px;width:50px}
.syn{display:flex;gap:5px;align-items:center;margin:12px 0 12px}.sdot{color:var(--text-muted);font-family:var(--font-mono);font-weight:700;padding:0 2px}
.wr .eq{font-family:var(--font-mono);font-size:16px;color:var(--text-secondary);line-height:1.5}
.wr .eq b{color:var(--fail)}
`, `<div class="glow"></div><div class="field"></div>
<div class="wl"><div class="cap t" data-m="w-cap">A<sub>16</sub> block · low bits</div>
  <div class="sq"><div class="cross-r"></div><div class="cross-c"></div>${square()}</div>
  <div class="note t" data-m="w-note">bit 5 flipped</div></div>
<div class="wr"><div class="cap t" data-m="w-syn">Syndrome</div><div class="syn">${synBits(28)}</div>
  <div class="eq t" data-m="w-eq">0101₂ = ${POS}<br><b>▲ caught at ${POS}</b></div></div>
<div class="stack"><div class="eyebrow t" data-m="eyebrow">${EYEBROW}</div>${h1(60)}
<div class="rows">${rows({ big: 34, rest: 17, scope: 14, models: 0 /* og card: no models line, no room in the centre 630 */ })}</div><div class="url t" data-m="url">${URL_TXT}</div></div>`);
out('og-a16-v3.html', ogB('dark'));

/* ---------- Post image 1600x900 (direction B expanded) ---------- */
const post = (theme) => page(theme, 1600, 900, `
.left{position:absolute;left:96px;top:0;bottom:0;width:900px;display:flex;flex-direction:column;justify-content:center}
.eyebrow{font-size:19px;margin-bottom:20px}
.rows{margin-top:44px;border-top:1px solid var(--border-hairline)}
.row>*{padding:17px 0}.row>.big{padding-right:40px}.left .scope{white-space:nowrap}
.txt .scope{margin-top:6px}.txt .models{margin-top:3px}.url{font-size:18px;margin-top:26px}
.right{position:absolute;left:1072px;top:0;bottom:0;width:440px;display:flex;flex-direction:column;justify-content:center;--cell:92px;--gap:11px}
.cap{font-family:var(--font-mono);font-size:15px;color:var(--text-muted);letter-spacing:.06em;text-transform:uppercase;font-weight:600;margin-bottom:8px}
.src{font-family:var(--font-mono);font-size:14px;color:var(--text-muted);margin-bottom:18px}
.cell{font-size:34px;border-radius:10px}
.cross-r{left:-8px;right:-8px;top:95px;height:108px;border-radius:14px}.cross-c{top:-8px;bottom:-8px;left:95px;width:108px;border-radius:14px}
.foot{margin-top:26px;display:flex;align-items:center;gap:6px;white-space:nowrap;font-family:var(--font-mono);font-size:18px;color:var(--text-secondary)}
.foot .syl{font-size:14px;letter-spacing:.06em;text-transform:uppercase;font-weight:600;color:var(--text-muted);margin-right:8px}
.sdot{color:var(--text-muted);font-weight:700;padding:0 2px}.foot b{color:var(--fail);margin-left:12px}
.legend{margin-top:14px;font-family:var(--font-mono);font-size:13.5px;color:var(--text-muted);line-height:1.55}
.legend .acc{font-weight:700}
`, `<div class="glow" style="background:radial-gradient(ellipse 50% 60% at 78% 50%, var(--accent-soft), transparent 70%)"></div><div class="field"></div>
<div class="left"><div class="eyebrow t" data-m="eyebrow">${EYEBROW}</div>${h1(88)}
<div class="rows">${rows({ big: 50, rest: 23, scope: 17.5, models: 14 /* the only layout with the models line */ })}</div><div class="url t" data-m="url">${URL_TXT}</div></div>
<div class="right"><div class="cap t" data-m="w-cap">One A<sub>16</sub> block, one flipped bit</div>
<div class="src t" data-m="w-src">Qwen2.5-1.5B release · layer 0 k_proj · row 0</div>
<div class="sq"><div class="cross-r"></div><div class="cross-c"></div>${square({ showY: true })}</div>
<div class="foot"><span class="syl t" data-m="w-syl">syndrome</span>${synBits(28)}<b class="t" data-m="w-caught">▲ caught at ${POS}</b></div>
<div class="legend t" data-m="w-legend"><span class="acc">p</span> parity bit · big digit: low bit c · small: y</div></div>`);
out('x-post-1600x900.html', post('dark'));

/* ---------- Portrait 1080x1350 ---------- */
out('x-post-1080x1350.html', page('dark', 1080, 1350, `
.wrap{position:absolute;left:96px;right:96px;top:0;bottom:0;display:flex;flex-direction:column;justify-content:center}
.eyebrow{font-size:19px;margin-bottom:20px}
.mid{display:flex;align-items:center;gap:48px;margin:56px 0 48px;--cell:100px;--gap:12px}
.cell{font-size:36px;border-radius:10px}
.cross-r{left:-8px;right:-8px;top:104px;height:116px;border-radius:14px}.cross-c{top:-8px;bottom:-8px;left:104px;width:116px;border-radius:14px}
.side{font-family:var(--font-mono);color:var(--text-secondary);font-size:22px;line-height:1.5}
.side .cap{font-size:14px;letter-spacing:.06em;text-transform:uppercase;font-weight:600;color:var(--text-muted);margin-bottom:10px}
.syn{display:flex;gap:4px;align-items:center;margin:4px 0 14px}.sdot{color:var(--text-muted);font-weight:700;padding:0 2px}
.side b{color:var(--fail)} .side .src{font-size:16px;color:var(--text-muted);margin-top:16px}
.rows{border-top:1px solid var(--border-hairline)}
.row>*{padding:18px 0}.row>.big{padding-right:40px}
.txt .scope{margin-top:6px}.url{font-size:18px;margin-top:26px}
`, `<div class="glow"></div><div class="field"></div>
<div class="wrap"><div class="eyebrow t" data-m="eyebrow">${EYEBROW}</div>${h1(88)}
<div class="mid"><div class="sq"><div class="cross-r"></div><div class="cross-c"></div>${square({ showY: true })}</div>
<div class="side"><div class="cap t" data-m="w-cap">Syndrome</div><div class="syn">${synBits(32)}</div>
<div class="t" data-m="w-eq">0101₂ = ${POS}<br><b>▲ caught at ${POS}</b></div>
<div class="src t" data-m="w-src">one real A<sub>16</sub> block,<br>Qwen2.5-1.5B release,<br>layer 0 k_proj, row 0</div></div></div>
<div class="rows">${rows({ big: 44, rest: 21, scope: 16, models: 0 /* portrait: no models line, kept as approved */ })}</div><div class="url t" data-m="url">${URL_TXT}</div></div>`));
/* ---------- Component demo (not an image; open in a browser) ---------- */
out('components-demo.html', `<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>Siliquant preview components</title>
<link rel="stylesheet" href="../src/tokens.css"><link rel="stylesheet" href="../src/components.css"><style>
body{overflow:auto;padding:48px;font-family:var(--font-mono);color:var(--text-muted)} section{position:relative;margin:0 0 56px} h2{font-size:13px;letter-spacing:.06em;text-transform:uppercase;margin:0 0 16px}
.rows{display:grid;grid-template-columns:auto 1fr;max-width:760px;border-top:1px solid var(--border-hairline)} .row{display:contents} .row>*{border-bottom:1px solid var(--border-hairline);padding:14px 0;display:flex;flex-direction:column;justify-content:center} .row>.big{padding-right:32px}
.sq{--cell:72px;--gap:9px;width:max-content}.cell{font-size:28px}.cross-r{left:-6px;right:-6px;top:75px;height:84px;border-radius:12px}.cross-c{top:-6px;bottom:-6px;left:75px;width:84px;border-radius:12px}
.syn{display:flex;gap:5px;align-items:center}.sdot{font-weight:700;padding:0 2px}
.swatch{position:relative;height:180px;border:1px solid var(--border-hairline);border-radius:8px;overflow:hidden}
</style></head><body class="dark">
<section><h2>Headline (.hl)</h2>${h1(64)}</section>
<section><h2>Metrics strip (.rows / .row / .big / .rest / .scope)</h2><div class="rows">${rows({ big: 40, rest: 19, scope: 15, models: 0 })}</div></section>
<section><h2>Hamming square (.sq / .cell / .cross-r / .cross-c)</h2><div class="sq"><div class="cross-r"></div><div class="cross-c"></div>${square({ showY: true })}</div></section>
<section><h2>Syndrome (.sb)</h2><div class="syn">${synBits(28)}<span style="margin-left:12px;color:var(--fail)">▲ caught at ${POS}</span></div></section>
<section><h2>Background (.glow + .field)</h2><div class="swatch"><div class="glow"></div><div class="field"></div></div></section>
<script>document.fonts.ready.then(()=>{document.body.dataset.ready='1'})</script></body></html>`);
console.log('built renders/*.html');
