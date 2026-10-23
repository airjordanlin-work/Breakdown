import { useEffect, useRef, useState } from "react";

// Plays the frames around a missed moment from the phone's own memory, slowly,
// stops on the miss and circles the joint. The pictures are mirrored like the
// live view, and the backend's x is already in that mirrored picture.
export const REPLAY_SPEED = 0.5;
export const RING = "#ff2d2d";

// The frame closest to time t, or null if none is within `within` seconds.
export function nearestFrame(frames, t, within = 0.3) {
  let best = null;
  for (const f of frames) {
    if (best === null || Math.abs(f.t - t) < Math.abs(best.t - t)) best = f;
  }
  return best && Math.abs(best.t - t) <= within ? best : null;
}

export default function ReplayOverlay({ frames, fix, onClose }) {
  const canvasRef = useRef(null);
  const [paused, setPaused] = useState(false);
  const [run, setRun] = useState(0);   // bump to play again

  useEffect(() => {
    const target = nearestFrame(frames, fix.t);
    if (!target) return undefined;
    const clip = frames.filter((f) => f.t <= target.t);
    const images = clip.map((f) => {
      const img = new Image();
      img.src = `data:image/jpeg;base64,${f.b64}`;
      return img;
    });
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    let raf = 0;
    let stopped = false;
    setPaused(false);

    const draw = (i, ring) => {
      const img = images[i];
      if (!img.complete || !img.naturalWidth) return;
      if (canvas.width !== img.naturalWidth) { canvas.width = img.naturalWidth; canvas.height = img.naturalHeight; }
      ctx.save();
      ctx.scale(-1, 1);
      ctx.drawImage(img, -canvas.width, 0, canvas.width, canvas.height);
      ctx.restore();
      if (ring) {
        ctx.beginPath();
        ctx.arc(fix.x * canvas.width, fix.y * canvas.height, canvas.width * 0.07, 0, Math.PI * 2);
        ctx.lineWidth = Math.max(3, canvas.width * 0.012);
        ctx.strokeStyle = RING;
        ctx.stroke();
      }
    };

    const start = performance.now();
    const t0 = clip[0].t;
    const tick = () => {
      if (stopped) return;
      const elapsed = ((performance.now() - start) / 1000) * REPLAY_SPEED;
      let i = 0;
      while (i + 1 < clip.length && clip[i + 1].t - t0 <= elapsed) i += 1;
      const last = i === clip.length - 1;
      draw(i, last);
      if (last) { setPaused(true); return; }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => { stopped = true; cancelAnimationFrame(raf); };
  }, [frames, fix, run]);

  return (
    <div role="dialog" aria-label="Replay of the mistake" style={{
      position: "absolute", inset: 0, zIndex: 6, background: "rgba(6,6,8,0.96)",
      display: "flex", flexDirection: "column", padding: 12, gap: 8,
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center",
        fontSize: 11, letterSpacing: "0.2em", color: "#ffc14c" }}>
        <span>REPLAY · HALF SPEED · {paused ? "PAUSED AT THE MISTAKE" : "PLAYING"}</span>
        <span style={{ display: "flex", gap: 8 }}>
          <button type="button" onClick={() => setRun((n) => n + 1)} style={btn("#ffc14c")}>AGAIN</button>
          <button type="button" onClick={onClose} style={btn("#aaa")}>CLOSE</button>
        </span>
      </div>
      <canvas ref={canvasRef} style={{ flex: 1, minHeight: 0, width: "100%", objectFit: "contain" }} />
      <div style={{ fontFamily: "'Anton',sans-serif", fontSize: 20, letterSpacing: "0.05em",
        color: RING, textAlign: "center" }}>
        {`${fix.fix} ${fix.joint}`.toUpperCase()}
      </div>
    </div>
  );
}

const btn = (color) => ({
  background: "transparent", border: `1px solid ${color}`, color,
  padding: "4px 12px", fontFamily: "'Anton',sans-serif",
  fontSize: 10, letterSpacing: "0.15em", cursor: "pointer",
});
