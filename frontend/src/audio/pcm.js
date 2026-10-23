// Push-to-talk audio helpers: the backend wants 16 kHz mono 16-bit PCM
// (see backend/app/stt.py). The microphone runs at the device rate, usually
// 48 kHz on iPhone, so it is averaged down here.

export const TARGET_RATE = 16000;

// Average-downsample float samples from inRate to 16 kHz. Averaging each
// window is a crude low-pass, which is enough for speech to text.
export function downsampleTo16k(input, inRate) {
  if (inRate === TARGET_RATE) return Float32Array.from(input);
  if (inRate < TARGET_RATE) throw new Error(`mic rate ${inRate} is below 16 kHz`);
  const ratio = inRate / TARGET_RATE;
  const outLen = Math.floor(input.length / ratio);
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const start = Math.floor(i * ratio);
    const end = Math.min(input.length, Math.floor((i + 1) * ratio));
    let sum = 0;
    for (let j = start; j < end; j++) sum += input[j];
    out[i] = end > start ? sum / (end - start) : 0;
  }
  return out;
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
