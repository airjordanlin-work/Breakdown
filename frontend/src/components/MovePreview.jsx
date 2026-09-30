import { useEffect, useRef, useState } from "react";
import { drawFigure, drawSkeleton, lerpPose } from "../render/figure";

const API = "http://localhost:8000";
const cache = new Map();          // move id -> preview data, so switching back is instant
const TRAIL = 4;                  // afterimages drawn behind the current pose
const TRAIL_GAP = 3;              // frames between afterimages
const SPEEDS = [0.25, 0.5, 0.75, 1];
const DEFAULT_SPEED = 0.5;

const prefersReducedMotion = () =>
  window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

function bounds(frames) {
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const f of frames) for (const [x, y] of f) {
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
  }
  return { minX, maxX, minY, maxY };
}

export default function MovePreview({ moveId, title }) {
  const canvasRef = useRef(null);
  const [data, setData] = useState(() => cache.get(moveId) || null);
  const [error, setError] = useState(null);
  const [playing, setPlaying] = useState(() => !prefersReducedMotion());
  const [speed, setSpeed] = useState(DEFAULT_SPEED);
  const [view, setView] = useState("figure");
  const posRef = useRef(0);       // fractional frame position, for smooth slow motion

  useEffect(() => {
    setError(null);
    posRef.current = 0;
    if (cache.has(moveId)) { setData(cache.get(moveId)); return; }
    setData(null);
    let cancelled = false;
    fetch(`${API}/moves/${moveId}/preview`)
      .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
      .then((d) => { cache.set(moveId, d); if (!cancelled) setData(d); })
      .catch(() => { if (!cancelled) setError("Preview unavailable for this move."); });
    return () => { cancelled = true; };
  }, [moveId]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !data) return;
    const ctx = canvas.getContext("2d");
    const { frames, bones, fps } = data;
    const n = frames.length;
    const b = bounds(frames);
    let raf, last = performance.now();

    // Pose at a fractional frame position, blended between neighbors so
    // 0.25x stays smooth instead of stepping through 60 frames.
    const poseAt = (pos) => {
      const p = ((pos % n) + n) % n;
      const i = Math.floor(p);
      return lerpPose(frames[i], frames[(i + 1) % n], p - i);
    };

    const render = () => {
      const dpr = window.devicePixelRatio || 1;
      const W = canvas.clientWidth, H = canvas.clientHeight;
      if (canvas.width !== W * dpr) { canvas.width = W * dpr; canvas.height = H * dpr; }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);

      const pad = 0.16;
      const unit = Math.min(W * (1 - 2 * pad) / (b.maxX - b.minX || 1),
                            H * (1 - 2 * pad) / (b.maxY - b.minY || 1));
      const offX = W / 2 - ((b.minX + b.maxX) / 2) * unit;
      const offY = H * (1 - pad) - b.maxY * unit;
      const map = ([x, y]) => [x * unit + offX, y * unit + offY];

      const floorY = b.maxY * unit + offY + unit * 0.3;
      ctx.strokeStyle = "#26262c";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(W * 0.12, floorY); ctx.lineTo(W * 0.88, floorY);
      ctx.stroke();

      const pos = posRef.current;
      for (let k = TRAIL; k >= 1; k--) {
        drawSkeleton(ctx, poseAt(pos - k * TRAIL_GAP), bones, map, "#ff2d2d", 3,
                     0.05 + (TRAIL - k) * 0.04);
      }
      const pose = poseAt(pos);
      if (view === "figure") {
        drawFigure(ctx, pose, map, unit);
      } else {
        drawSkeleton(ctx, pose, bones, map, "#eee9e0", 5, 1);
        const [hx, hy] = map(pose[0]);
        ctx.fillStyle = "#eee9e0";
        ctx.beginPath(); ctx.arc(hx, hy, 9, 0, Math.PI * 2); ctx.fill();
      }
    };

    const tick = (now) => {
      posRef.current += ((now - last) / 1000) * fps * speed;
      last = now;
      render();
      raf = requestAnimationFrame(tick);
    };

    render();
    if (playing) raf = requestAnimationFrame(tick);
    const onResize = () => render();
    window.addEventListener("resize", onResize);
    return () => { cancelAnimationFrame(raf); window.removeEventListener("resize", onResize); };
  }, [data, playing, speed, view]);

  return (
    <figure className="stage" aria-label={`Animated preview of ${title}`}>
      <div className="stage-title" aria-hidden="true">{title}</div>
      <canvas ref={canvasRef} className="stage-canvas" />
      {!data && !error && <p className="stage-note">Loading preview</p>}
      {error && <p className="stage-note">{error}</p>}
      <figcaption className="stage-controls">
        <button type="button" onClick={() => setPlaying((p) => !p)} disabled={!data}>
          {playing ? "Pause" : "Play"}
        </button>
        <div className="segmented" role="group" aria-label="Playback speed">
          {SPEEDS.map((s) => (
            <button key={s} type="button" aria-pressed={speed === s}
              onClick={() => setSpeed(s)} disabled={!data}>
              {s}x
            </button>
          ))}
        </div>
        <div className="segmented" role="group" aria-label="Display style">
          {["figure", "skeleton"].map((v) => (
            <button key={v} type="button" aria-pressed={view === v}
              onClick={() => setView(v)} disabled={!data}>
              {v === "figure" ? "Figure" : "Skeleton"}
            </button>
          ))}
        </div>
      </figcaption>
    </figure>
  );
}
