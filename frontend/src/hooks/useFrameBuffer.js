import { useRef, useCallback } from "react";

// The pictures the phone sent for pose estimation, kept in this tab's memory
// for a few seconds so the last attempt can be replayed. Nothing here is
// uploaded or stored: it is cleared as it ages out and when the page closes.
// A frame is { t, b64 } where t is its capture time in seconds, the same
// number the backend reports for a missed moment.
export const KEEP_SECONDS = 15;
export const MAX_FRAMES = 240;

// Frames of the buffer with t in [from, to], oldest first.
export function framesBetween(frames, from, to) {
  return frames.filter((f) => f.t >= from && f.t <= to);
}

export function useFrameBuffer() {
  const frames = useRef([]);

  const add = useCallback((b64, t) => {
    const list = frames.current;
    list.push({ t, b64 });
    while (list.length > MAX_FRAMES || (list.length && list[0].t < t - KEEP_SECONDS)) list.shift();
  }, []);

  const between = useCallback((from, to) => framesBetween(frames.current, from, to), []);
  const clear = useCallback(() => { frames.current = []; }, []);

  return { add, between, clear };
}
