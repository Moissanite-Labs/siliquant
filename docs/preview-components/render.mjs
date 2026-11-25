// node render.mjs <name> <w> <h>
// Renders renders/<name>.html to renders/<name>.raw.png, measures text contrast on the rendered pixels,
// checks layout (64 px inset; for 1200x630 also the WhatsApp centre and X 2:1 crops), writes renders/<name>.measure.json.
// Contrast: per text box, the text core (95th-pct luminance if text is lighter than its background, else 5th)
// against the median of the same box rendered with text hidden (body.bgonly).
import { chromium } from 'playwright-core';
import { PNG } from 'pngjs';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const here = path.dirname(fileURLToPath(import.meta.url));
const [, , name, W, H] = process.argv;
const w = +W, h = +H;
const R = path.join(here, 'renders');
const chrome = process.env.CHROME_PATH || (fs.existsSync('/usr/bin/google-chrome') ? '/usr/bin/google-chrome' : undefined);
const b = await chromium.launch(chrome ? { executablePath: chrome } : {});
const p = await b.newPage({ viewport: { width: w, height: h }, deviceScaleFactor: 1 });
await p.goto('file://' + path.join(R, name + '.html'));
await p.waitForSelector('body[data-ready="1"]');
// Inter and JetBrains Mono are system fonts (the site ships no copies). Fail if they would fall back.
const missing = await p.evaluate(() => {
  const c = document.createElement('canvas').getContext('2d'); const s = 'Lattice 0 extra bytes 400 tok/s';
  const wd = f => { c.font = f; return c.measureText(s).width; };
  const out = [];
  if (wd('20px "JetBrains Mono", monospace') === wd('20px monospace') && wd('20px "JetBrains Mono", serif') === wd('20px serif')) out.push('JetBrains Mono');
  if (wd('20px "Inter", sans-serif') === wd('20px sans-serif') && wd('20px "Inter", serif') === wd('20px serif')) out.push('Inter');
  if (![...document.fonts].some(f => f.family.includes('Newsreader Display') && f.status === 'loaded')) out.push('Newsreader Display (docs/assets/fonts/newsreader-display-500.woff2)');
  return out;
});
if (missing.length && !process.env.ALLOW_FALLBACK_FONTS) { await b.close(); console.error('Missing fonts: ' + missing.join(', ') + '. See README "Dependencies". Set ALLOW_FALLBACK_FONTS=1 to render anyway.'); process.exit(2); }
const raw = path.join(R, name + '.raw.png'), bgp = path.join(R, '.' + name + '.bg.png');
await p.screenshot({ path: raw });
const boxes = await p.evaluate(() => [...document.querySelectorAll('[data-m]')].map(e => { const r = e.getBoundingClientRect(); return { m: e.dataset.m, x0: Math.floor(r.left), y0: Math.floor(r.top), x1: Math.ceil(r.right), y1: Math.ceil(r.bottom) }; }));
const overflow = await p.evaluate(() => [...document.querySelectorAll('.big,.rest,.scope,h1')].filter(e => e.scrollWidth > e.clientWidth + 1).map(e => e.dataset.m));
await p.evaluate(() => document.body.classList.add('bgonly'));
await p.screenshot({ path: bgp });
await b.close();
const A = PNG.sync.read(fs.readFileSync(raw)), B = PNG.sync.read(fs.readFileSync(bgp));
fs.unlinkSync(bgp);
const lum = (r, g, bb) => { const f = x => { x /= 255; return x <= 0.03928 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4; }; return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(bb); };
const text = [];
for (const bx of boxes) {
  const fg = [], bg = [];
  for (let y = Math.max(0, bx.y0); y < Math.min(h, bx.y1); y++) for (let x = Math.max(0, bx.x0); x < Math.min(w, bx.x1); x++) {
    const i = (y * w + x) * 4;
    const la = lum(A.data[i], A.data[i + 1], A.data[i + 2]), lb = lum(B.data[i], B.data[i + 1], B.data[i + 2]);
    bg.push(lb); if (Math.abs(la - lb) > 0.02) fg.push(la);
  }
  if (!fg.length) { text.push({ el: bx.m, box: [bx.x0, bx.y0, bx.x1, bx.y1], contrast: null }); continue; }
  fg.sort((a, b) => a - b); bg.sort((a, b) => a - b);
  const g = bg[Math.floor(bg.length / 2)];
  const lighter = fg[Math.floor(fg.length / 2)] > g;
  const f = lighter ? fg[Math.floor(fg.length * 0.95)] : fg[Math.floor(fg.length * 0.05)];
  text.push({ el: bx.m, box: [bx.x0, bx.y0, bx.x1, bx.y1], contrast: +(((Math.max(f, g) + 0.05) / (Math.min(f, g) + 0.05)).toFixed(2)) });
}
const key = boxes.filter(b => !b.m.startsWith('w-'));
const res = { name, size: [w, h], missingFonts: missing, overflow,
  inset64: boxes.every(b => b.x0 >= 64 && b.x1 <= w - 64 && b.y0 >= 64 && b.y1 <= h - 64),
  ...(w === 1200 && h === 630 ? { keyInCentre630: key.every(b => b.x0 >= 285 && b.x1 <= 915), clearOfX2to1: boxes.every(b => b.y0 >= 15 && b.y1 <= 615) } : {}),
  minContrast: Math.min(...text.filter(t => t.contrast).map(t => t.contrast)), text };
fs.writeFileSync(path.join(R, name + '.measure.json'), JSON.stringify(res, null, 2));
const ok = res.inset64 && !overflow.length && res.minContrast >= 4.5 && res.keyInCentre630 !== false && res.clearOfX2to1 !== false;
console.log(`${name}: minContrast ${res.minContrast}, inset64 ${res.inset64}` + (w === 1200 ? `, centre630 ${res.keyInCentre630}, x2to1 ${res.clearOfX2to1}` : '') + `, overflow [${overflow}] -> ${ok ? 'PASS' : 'FAIL'}`);
if (!ok) process.exit(1);
