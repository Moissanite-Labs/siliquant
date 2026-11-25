// One real A16 block from the page's chain figure (docs/index.html #chainFig data-block):
// Qwen2.5-1.5B release, layer 0 k_proj, row 0, block 0.
export const Y = [-28,-9,31,14,-4,-5,-4,1,5,6,-8,3,-1,4,-3,12];
export const C = [0,1,1,0,0,1,0,1,1,0,0,1,1,0,1,0];
export const PARITY = [0,1,2,4,8];          // A16_INFO_POSITIONS complement
export const FLIP = 5;
// Same as syndrome() in docs/index.html (check_a16_syndrome_errors): [p0, s3, s2, s1, s0]
export function syndrome(c) {
  const s = a => a.reduce((x, y) => x + y, 0) & 1;
  return [s(c), s(c.slice(8)), s(c.slice(4, 8).concat(c.slice(12))), s(c.filter((_, i) => i % 4 > 1)), s(c.filter((_, i) => i % 2))];
}
export const CF = C.map((b, i) => i === FLIP ? b ^ 1 : b);
export const SYN = syndrome(CF);           // -> [1,0,1,0,1]
export const POS = SYN[1]*8 + SYN[2]*4 + SYN[3]*2 + SYN[4];   // -> 5
if (syndrome(C).some(v => v) || POS !== FLIP) throw new Error('block data does not check');
const neg = n => String(n).replace('-', '\u2212');
export function square({ showY = false, showIx = true } = {}) {
  return CF.map((b, i) => {
    const cls = ['cell', b ? 'one' : '', PARITY.includes(i) ? 'par' : '', i === FLIP ? 'x' : ''].join(' ');
    const y = i === FLIP ? (Y[i] ^ 1) : Y[i];
    return `<div class="${cls}"><span class="ix t">${PARITY.includes(i) ? 'p' + i : i}</span><span class="t">${b}</span>${showY ? `<span class="y t">${neg(y)}</span>` : ''}</div>`;
  }).join('');
}
export function synBits(size) {
  const st = `style="width:${size}px;height:${Math.round(size*1.2)}px;font-size:${Math.round(size*0.62)}px"`;
  const [p, ...r] = SYN;
  return `<span class="sb ${p ? 'bad' : 'ok'}" ${st}><span class="t">${p}</span></span><span class="t sdot">·</span>` + r.map(v => `<span class="sb ${v ? 'bad' : 'ok'}" ${st}><span class="t">${v}</span></span>`).join('');
}
