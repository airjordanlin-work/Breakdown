// Audio worklet for push-to-talk: hands each block of mic samples to the page.
// The page downsamples to 16 kHz and sends it (src/audio/pcm.js).
class PcmCapture extends AudioWorkletProcessor {
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch && ch.length) this.port.postMessage(ch.slice(0));
    return true;
  }
}
registerProcessor("pcm-capture", PcmCapture);
