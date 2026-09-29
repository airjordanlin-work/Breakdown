import { useEffect, useRef, useState } from "react";

const API = "http://localhost:8000";
const cache = new Map();          // move id -> preview data, so switching back is instant
const TRAIL = 4;                  // afterimages drawn behind the current pose
const TRAIL_GAP = 3;              // frames between afterimages

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

function drawPose(ctx, pts, bones, map, color, width, alpha) {
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.lineCap = "round";
  ctx.beginPath();
  for (const [a, b] of bones) {
    const [ax, ay] = map(pts[a]);
    const [bx, by] = map(pts[b]);
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
  }
  ctx.stroke();
}

export default function MovePreview({ moveId, title }) {
  const canvasRef = useRef(null);
  const [data, setData] = useState(() => cache.get(moveId) || null);
  const [error, setError] = useState(null);
  const [playing, setPlaying] = useState(() => !prefersReducedMotion());
  const [speed, setSpeed] = useState(1);
  const frameRef = useRef(0);

  useEffect(() => {
    setError(null);
    frameRef.current = 0;
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
    const b = bounds(frames);
    let raf, last = performance.now(), acc = 0;

    const render = () => {
      const dpr = window.devicePixelRatio || 1;
      const W = canvas.clientWidth, H = canvas.clientHeight;
      if (canvas.width !== W * dpr) { canvas.width = W * dpr; canvas.height = H * dpr; }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);

      const pad = 0.14;
      const scale = Math.min(W * (1 - 2 * pad) / (b.maxX - b.minX || 1),
                             H * (1 - 2 * pad) / (b.maxY - b.minY || 1));
      const offX = W / 2 - ((b.minX + b.maxX) / 2) * scale;
      const offY = H * (1 - pad) - b.maxY * scale;
      const map = ([x, y]) => [x * scale + offX, y * scale + offY];

      // floor line at the lowest point the move ever reaches
      const floorY = b.maxY * scale + offY + 6;
      ctx.globalAlpha = 1;
      ctx.strokeStyle = "#26262c";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(W * 0.12, floorY); ctx.lineTo(W * 0.88, floorY);
      ctx.stroke();

      const i = frameRef.current;
      for (let k = TRAIL; k >= 1; k--) {
        const j = (i - k * TRAIL_GAP + frames.length * TRAIL_GAP * TRAIL) % frames.length;
        drawPose(ctx, frames[j], bones, map, "#ff2d2d", 4, 0.06 + (TRAIL - k) * 0.05);
      }
      drawPose(ctx, frames[i], bones, map, "#eee9e0", 5, 1);

      const [hx, hy] = map(frames[i][0]);
      ctx.globalAlpha = 1;
      ctx.fillStyle = "#eee9e0";
      ctx.beginPath(); ctx.arc(hx, hy, 9, 0, Math.PI * 2); ctx.fill();
    };

    const tick = (now) => {
      acc += (now - last) * speed;
      last = now;
      const step = 1000 / fps;
      while (acc >= step) { frameRef.current = (frameRef.current + 1) % frames.length; acc -= step; }
      render();
      raf = requestAnimationFrame(tick);
    };

    render();
    if (playing) raf = requestAnimationFrame(tick);
    const onResize = () => render();
    window.addEventListener("resize", onResize);
    return () => { cancelAnimationFrame(raf); window.removeEventListener("resize", onResize); };
  }, [data, playing, speed]);

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
        <button type="button" onClick={() => setSpeed((s) => (s === 1 ? 0.5 : 1))}
          aria-pressed={speed === 0.5} disabled={!data}>
          {speed === 1 ? "Slow motion" : "Normal speed"}
        </button>
      </figcaption>
    </figure>
  );
}
