import { useRef, useState, useCallback, useEffect } from "react";

// Binary coach audio frame (see backend/app/audio_frames.py):
//   [2 bytes big-endian header length][JSON header][audio payload]
export function decodeAudioChunk(buf) {
  const n = new DataView(buf).getUint16(0, false);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 2, n)));
  return { header, payload: buf.slice(2 + n) };
}

// Plays coach audio streamed from the backend through Web Audio.
// iOS Safari only allows sound after a user gesture, so call unlock() from a
// tap handler before the first cue. Until then, audio is dropped (captions
// still show).
export function useCoachAudio() {
  const ctxRef     = useRef(null);
  const sourceRef  = useRef(null);
  const [unlocked, setUnlocked] = useState(false);

  const unlock = useCallback(async () => {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return;
    if (!ctxRef.current) ctxRef.current = new Ctx();
    const ctx = ctxRef.current;
    if (ctx.state !== "running") await ctx.resume();
    // A silent one-sample buffer completes the unlock on older iOS versions.
    const src = ctx.createBufferSource();
    src.buffer = ctx.createBuffer(1, 1, 22050);
    src.connect(ctx.destination);
    src.start(0);
    setUnlocked(ctx.state === "running");
  }, []);

  const playChunk = useCallback(async (buf) => {
    const ctx = ctxRef.current;
    if (!ctx || ctx.state !== "running") return;
    let header, payload;
    try {
      ({ header, payload } = decodeAudioChunk(buf));
    } catch (e) {
      console.warn("Bad coach audio frame", e);
      return;
    }
    if (header.type !== "coach_audio_chunk" || header.codec !== "wav") return;
    try {
      const audio = await ctx.decodeAudioData(payload);
      // Newest cue wins: cues are short and already rate-limited server-side.
      try { sourceRef.current?.stop(); } catch { /* already stopped */ }
      const src = ctx.createBufferSource();
      src.buffer = audio;
      src.connect(ctx.destination);
      src.start(0);
      sourceRef.current = src;
    } catch (e) {
      console.warn("Could not play coach audio", e);
    }
  }, []);

  useEffect(() => () => { ctxRef.current?.close(); }, []);

  return { unlocked, unlock, playChunk };
}
