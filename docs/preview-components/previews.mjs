// Timeline-size and crop previews into renders/previews/ (Chrome downscale; no Python needed).
import { chromium } from 'playwright-core';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const R = path.join(path.dirname(fileURLToPath(import.meta.url)), 'renders');
fs.mkdirSync(path.join(R, 'previews'), { recursive: true });
const chrome = process.env.CHROME_PATH || (fs.existsSync('/usr/bin/google-chrome') ? '/usr/bin/google-chrome' : undefined);
const b = await chromium.launch(chrome ? { executablePath: chrome } : {});
// [source png, output, crop x, y, w, h, output width]
const jobs = [
  ['og-a16-v3.png', 'og-506w.png', 0, 0, 1200, 630, 506],                 // X card in the timeline
  ['og-a16-v3.png', 'og-x-2to1-506w.png', 0, 15, 1200, 600, 506],         // X 2:1 crop
  ['og-a16-v3.png', 'og-whatsapp-centre-300.png', 285, 0, 630, 630, 300], // WhatsApp square crop
  ['x-post-1600x900.png', 'x-post-1600x900-506w.png', 0, 0, 1600, 900, 506],
  ['x-post-1080x1350.png', 'x-post-1080x1350-380w.png', 0, 0, 1080, 1350, 380],
];
for (const [src, out, x, y, cw, ch, ow] of jobs) {
  const oh = Math.round(ch * ow / cw), s = ow / cw;
  const p = await b.newPage({ viewport: { width: ow, height: oh } });
  const data = fs.readFileSync(path.join(R, src)).toString('base64');
  await p.setContent(`<body style="margin:0;overflow:hidden"><img src="data:image/png;base64,${data}" style="display:block;transform-origin:0 0;transform:translate(${-x * s}px,${-y * s}px) scale(${s})">`);
  await p.waitForFunction(() => document.images[0].complete);
  await p.screenshot({ path: path.join(R, 'previews', out) });
  await p.close();
}
await b.close();
console.log('previews: renders/previews/');
