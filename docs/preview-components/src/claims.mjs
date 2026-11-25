// Claim strings for the preview images. Every number here is ruled in CLAIMS-RULING (grk-tensor, 24 Nov 2026)
// and matches the results strip on docs/index.html. Change a number only after a new ruling.
//
// Labels: CITED = source path and value; ESTIMATE = arithmetic on cited values.
//   Kernel        400 tok/s, 5.6× vs PyTorch BF16   CITED (PR #88 engine report: 400.580 tok/s, 5.60×). Qwen2.5-0.5B, batch 1, laptop RTX 3080 only.
//   Quantization  16–40% lower KL than plain Q4_K_M  CITED KL values, ESTIMATE %. A16 4.5 bpw vs Q4_K_M 4.8, no calibration, three 1–2B Instruct models, not the released models.
//   Bit flips     0 extra bytes                     CITED. Must carry the qualifier: lowest bit of each 4-bit weight code; costs 3.1% tok/s.
// Deliberately absent: the Release pillar / model count (left off the 25 Nov images), 4.9× energy, kernel Gw/s,
// and any FP16 / cuBLAS / llama.cpp comparison. Never write "free", "zero overhead" or "error-correcting".
const STRIP = [
  { pillar: 'Kernel', big: '400 tok/s', rest: '5.6× faster than PyTorch BF16' },
  { pillar: 'Quantization', big: '16–40%', rest: 'lower KL than plain Q4_K_M' },
  { pillar: 'Bit flips', lead: 'Built-in bit-flip detection,', big: '0 extra bytes', rest: '' },
];
const by = p => STRIP.find(s => s.pillar === p);
const k = by('Kernel'), q = by('Quantization'), b = by('Bit flips');
export const HEADLINE = ['Lattice quantization,', 'from silicon to model'];
export const EYEBROW = 'Siliquant · Working Paper · 24 Nov 2026';
export const URL_TXT = 'moissanite-labs.github.io/siliquant';
export const ROWS = [
  { lab: k.pillar, big: k.big.replace(' ', '\u202F'), rest: k.rest, scope: 'Qwen2.5-0.5B · batch 1 · laptop RTX 3080' },
  { lab: q.pillar, big: q.big, rest: q.rest, scope: 'A16 at 4.5 bpw vs Q4_K_M at 4.8 · no calibration' },
  { lab: b.pillar, big: b.big, rest: b.lead.replace(/,$/, ''), scope: 'lowest bit of each 4-bit code · costs 3.1% tok/s' },
];
