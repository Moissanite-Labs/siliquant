// Claim strings for the preview images. Each number matches the results strip on docs/index.html
// and carries the scope stated there; the page's Limits section (#limits) gives what each one does not cover.
// Change a number only when the page and its public evidence change with it.
//
// Labels: CITED = source path and value; ESTIMATE = arithmetic on cited values.
//   Kernel        400 tok/s, 5.6× vs PyTorch BF16   CITED: studies/aletheion-qwen25-live-llm-engine/ALETHEION_LIVE_LLM_ENGINE_REPORT.md
//                 (400.580 tok/s, 5.60×). Qwen2.5-0.5B, batch 1, laptop RTX 3080 only.
//   Quantization  16–40% lower KL than plain Q4_K_M  CITED KL values, ESTIMATE %: experiments/hamming-lattices-real-weights-2026-11-24/README.md
//                 and results.out. A16 4.5 bpw vs Q4_K_M 4.8, no calibration, on Qwen2.5-1.5B-Instruct, SmolLM2-1.7B-Instruct
//                 and Llama-3.2-1B-Instruct (16-bit GGUF, WikiText-2). Not the released models.
//   Bit flips     0 extra bytes                     CITED: studies/aletheion-qwen25-live-llm-engine/qwen25_sm86_a16_engine.cu
//                 (a16_syndrome_check_16w) and the engine report. A storage claim about the engine's packed layout
//                 (18 B per 32 weights) only: the Hugging Face files store 11 of the 16 low bits and rely on SHA-256
//                 checks instead. Must carry the qualifier: lowest bit of each 4-bit weight code; costs 3.1% tok/s.
// Not on the images: the model count (the page's strip and #bench carry it), 4.9× lower energy than PyTorch BF16
// (same engine report and setup as the 5.6×), kernel Gw/s, and any FP16 / cuBLAS / llama.cpp comparison.
// Never write "free", "zero overhead" or "error-correcting".
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
  { lab: q.pillar, big: q.big, rest: q.rest, scope: 'A16 at 4.5 bpw vs Q4_K_M at 4.8 · no calibration',
    // The three models the range covers. Drawn only where a layout passes a non-zero o.models to rows():
    // today that is the 1600x900 post image alone. The og card and the portrait pass 0 (no room).
    models: 'Qwen2.5-1.5B-Instruct · SmolLM2-1.7B-Instruct · Llama-3.2-1B-Instruct' },
  { lab: b.pillar, big: b.big, rest: b.lead.replace(/,$/, ''), scope: 'lowest bit of each 4-bit code · costs 3.1% tok/s' },
];
