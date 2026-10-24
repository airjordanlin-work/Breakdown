// Push-to-talk audio helpers: the backend wants 16 kHz mono 16-bit PCM
// (see backend/app/stt.py). The microphone runs at the device rate, usually
// 48 kHz on iPhone, so it is averaged down here.

export const TARGET_RATE = 16000;

// Average-downsample float samples to 16 kHz. Averaging each window is a
// crude low-pass, which is enough for speech to text. Audio arrives one
// chunk at a time, so window edges are kept in absolute sample positions:
// a ratio like 44.1 kHz / 16 kHz = 2.75 then loses nothing at chunk
// boundaries. Returns a function that takes the next chunk and gives back
// the 16 kHz samples ready so far.
export function createDownsampler(inRate) {
  if (inRate === TARGET_RATE) return (input) => Float32Array.from(input);
  if (inRate < TARGET_RATE) throw new Error(`mic rate ${inRate} is below 16 kHz`);
  const ratio = inRate / TARGET_RATE;
  let buf = new Float32Array(0);   // input not yet fully used
  let base = 0;                    // absolute index of buf[0]
  let made = 0;                    // output samples produced so far
  return (input) => {
    const joined = new Float32Array(buf.length + input.length);
    joined.set(buf);
    joined.set(input, buf.length);
    const out = [];
    for (;;) {
      const start = Math.floor(made * ratio) - base;
      const end = Math.floor((made + 1) * ratio) - base;
      if (end > joined.length) break;
      let sum = 0;
      for (let j = start; j < end; j++) sum += joined[j];
      out.push(end > start ? sum / (end - start) : 0);
      made += 1;
    }
    const keepFrom = Math.floor(made * ratio) - base;
    buf = joined.slice(keepFrom);
    base += keepFrom;
    return Float32Array.from(out);
  };
}

// Float samples in [-1, 1] to little-endian 16-bit PCM bytes.
export function floatToPcm16(samples) {
  const out = new Uint8Array(samples.length * 2);
  const view = new DataView(out.buffer);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return out;
}

export function bytesToBase64(bytes) {
  let bin = "";
  const step = 0x8000;
  for (let i = 0; i < bytes.length; i += step) {
    bin += String.fromCharCode.apply(null, bytes.subarray(i, i + step));
  }
  return btoa(bin);
}

// Joins Float32Array pieces into one.
export function concatFloat(parts) {
  const total = parts.reduce((n, p) => n + p.length, 0);
  const out = new Float32Array(total);
  let off = 0;
  for (const p of parts) { out.set(p, off); off += p.length; }
  return out;
}
