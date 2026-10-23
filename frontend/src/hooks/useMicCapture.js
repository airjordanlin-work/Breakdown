import { useCallback, useMemo, useRef } from "react";
import { TARGET_RATE, bytesToBase64, concatFloat, downsampleTo16k, floatToPcm16 } from "../audio/pcm";

// Push-to-talk microphone capture. start() opens the mic (the first time,
// the browser asks for permission; it has to be called from a tap), and
// calls onChunk(base64) about every SEND_MS with 16 kHz mono PCM16.
// stop() sends what is left and closes the mic, so the recording indicator
// goes away between turns. Nothing is stored: chunks go straight to onChunk.
const SEND_MS = 250;
export const MAX_HOLD_MS = 8000;   // the backend cuts clips at 8 s too

export function useMicCapture(onChunk) {
  const state = useRef(null);   // { stream, ctx, node, source, parts, rate, timer }
  const wanted = useRef(0);     // bumps on every start and stop, so a slow start can tell it was cancelled
  const onChunkRef = useRef(onChunk);
  onChunkRef.current = onChunk;

  const flush = useCallback(() => {
    const s = state.current;
    if (!s || !s.parts.length) return;
    const raw = concatFloat(s.parts);
    s.parts.length = 0;   // same array the capture node pushes into
    const pcm = floatToPcm16(downsampleTo16k(raw, s.rate));
    if (pcm.length) onChunkRef.current?.(bytesToBase64(pcm));
  }, []);

  const stop = useCallback(() => {
    wanted.current += 1;
    const s = state.current;
    if (!s) return;
    flush();
    state.current = null;
    clearInterval(s.timer);
    try { s.node.disconnect(); } catch { /* already */ }
    try { s.source.disconnect(); } catch { /* already */ }
    s.stream.getTracks().forEach((t) => t.stop());
    s.ctx.close().catch(() => {});
  }, [flush]);

  // Resolves once the mic is recording. Throws if the mic is blocked or missing.
  const start = useCallback(async () => {
    if (state.current) return;
    const ticket = ++wanted.current;
    if (!navigator.mediaDevices?.getUserMedia) throw new Error("This browser has no microphone access");
    // Made before the first await, while still inside the tap, so iOS lets it run.
    const Ctx = window.AudioContext || window.webkitAudioContext;
    const ctx = new Ctx();
    const resumed = ctx.resume().catch(() => {});
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      });
    } catch (e) {
      ctx.close().catch(() => {});
      throw e;
    }
    await resumed;
    if (ctx.sampleRate < TARGET_RATE) {
      stream.getTracks().forEach((t) => t.stop());
      ctx.close().catch(() => {});
      throw new Error(`Microphone rate ${ctx.sampleRate} Hz is too low`);
    }
    const source = ctx.createMediaStreamSource(stream);
    const parts = [];
    let node;
    if (ctx.audioWorklet) {
      await ctx.audioWorklet.addModule(`${import.meta.env.BASE_URL}pcm-worklet.js`);
      node = new AudioWorkletNode(ctx, "pcm-capture");
      node.port.onmessage = (e) => parts.push(e.data);
    } else {
      // Older browsers: the deprecated ScriptProcessor does the same job.
      node = ctx.createScriptProcessor(4096, 1, 1);
      node.onaudioprocess = (e) => parts.push(Float32Array.from(e.inputBuffer.getChannelData(0)));
    }
    source.connect(node);
    // A worklet only runs while connected to the output; it writes silence there.
    node.connect(ctx.destination);
    state.current = { stream, ctx, node, source, parts, rate: ctx.sampleRate, timer: null };
    if (ticket !== wanted.current) {
      // Released before the mic opened: close it right away.
      wanted.current -= 1;
      stop();
      return;
    }
    state.current.timer = setInterval(flush, SEND_MS);
  }, [flush]);

  // Stable object: callers list it in effect deps, and a new one each render would stop the mic.
  return useMemo(() => ({ start, stop, active: () => state.current !== null }), [start, stop]);
}
