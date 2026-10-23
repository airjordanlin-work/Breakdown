import { useRef, useEffect } from "react";

const CONNECTIONS = [
  [11,12],[11,13],[13,15],[12,14],[14,16],
  [11,23],[12,24],[23,24],
  [23,25],[25,27],[24,26],[26,28],
];

const JOINT_COLOR = (v) => v >= 0.7 ? "#4cff4c" : v >= 0.4 ? "#ffc14c" : "#ff4c4c";
const LINE_COLOR  = (va, vb) => {
  const avg = (va + vb) / 2;
  return avg >= 0.7 ? "rgba(255,255,255,0.9)"
       : avg >= 0.4 ? "rgba(255,255,255,0.45)"
       :               "rgba(255,255,255,0.12)";
};

function drawGraffitiBackground(ctx, W, H) {
  // large faded BREAK text watermark
  ctx.save();
  ctx.translate(W * 0.5, H * 0.5);
  ctx.rotate(-0.12);
  ctx.font = `bold ${Math.floor(W * 0.18)}px 'Arial Black', Arial`;
  ctx.fillStyle = "rgba(255,45,45,0.03)";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  const words = ["BREAK", "B-BOY", "DANCE", "STYLE"];
  const spacing = H * 0.28;
  words.forEach((w, i) => {
    ctx.fillText(w, 0, (i - 1.5) * spacing);
  });
  ctx.restore();

  // corner tags
  ctx.save();
  ctx.font = `bold ${Math.floor(W * 0.06)}px 'Arial Black', Arial`;
  ctx.fillStyle = "rgba(255,45,45,0.04)";
  ctx.textAlign = "left";
  ctx.fillText("BR", 20, 80);
  ctx.textAlign = "right";
  ctx.fillText("EAK", W - 20, H - 40);
  ctx.restore();

  // subtle grid lines — street court feel
  ctx.save();
  ctx.strokeStyle = "rgba(255,255,255,0.02)";
  ctx.lineWidth = 1;
  const gridSize = Math.floor(W / 8);
  for (let x = 0; x < W; x += gridSize) {
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
  }
  for (let y = 0; y < H; y += gridSize) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
  }
  ctx.restore();
}

function drawSkeleton(ctx, W, H, landmarks, visibility) {
  if (!landmarks || landmarks.length === 0) return;
  for (const [a, b] of CONNECTIONS) {
    if (a >= landmarks.length || b >= landmarks.length) continue;
    ctx.beginPath();
    ctx.moveTo(landmarks[a][0] * W, landmarks[a][1] * H);
    ctx.lineTo(landmarks[b][0] * W, landmarks[b][1] * H);
    ctx.strokeStyle = LINE_COLOR(visibility[a]||0, visibility[b]||0);
    ctx.lineWidth = 2.5;
    ctx.stroke();
  }
  const mapped = new Set(CONNECTIONS.flat());
  for (const i of mapped) {
    if (i >= landmarks.length) continue;
    const x = landmarks[i][0] * W, y = landmarks[i][1] * H;
    ctx.beginPath(); ctx.arc(x, y, 7, 0, Math.PI*2);
    ctx.fillStyle = "rgba(0,0,0,0.7)"; ctx.fill();
    ctx.beginPath(); ctx.arc(x, y, 5, 0, Math.PI*2);
    ctx.fillStyle = JOINT_COLOR(visibility[i]||0); ctx.fill();
  }
}

function drawGhost(ctx, W, H, ghostBones, anchor, aligned) {
  if (!ghostBones || ghostBones.length === 0 || !anchor) return;
  const { hip_cx, hip_cy, scale } = anchor;
  if (!scale || scale <= 0) return;

  const toPixel = (bx, by) => [
    hip_cx * W + bx * scale * W,
    hip_cy * H + by * scale * W,
  ];

  const color = aligned ? "rgba(0,220,60," : "rgba(40,170,170,";
  const alpha = aligned ? 0.55 : 0.28;

  for (const [x1,y1,x2,y2] of ghostBones) {
    const [px1,py1] = toPixel(x1,y1);
    const [px2,py2] = toPixel(x2,y2);
    if (!aligned) ctx.setLineDash([8,5]);
    ctx.beginPath(); ctx.moveTo(px1,py1); ctx.lineTo(px2,py2);
    ctx.strokeStyle = `${color}${alpha})`; ctx.lineWidth = 2; ctx.stroke();
    ctx.setLineDash([]);
  }

  const seen = new Set();
  for (const [x1,y1,x2,y2] of ghostBones) {
    for (const [bx,by] of [[x1,y1],[x2,y2]]) {
      const k = `${bx},${by}`;
      if (seen.has(k)) continue; seen.add(k);
      const [px,py] = toPixel(bx,by);
      ctx.beginPath(); ctx.arc(px,py,7,0,Math.PI*2);
      ctx.fillStyle="rgba(0,0,0,0.6)"; ctx.fill();
      ctx.beginPath(); ctx.arc(px,py,4,0,Math.PI*2);
      ctx.fillStyle=`${color}0.75)`; ctx.fill();
    }
  }

  // head
  const [hx,hy] = toPixel(0,-1.90);
  const hr = Math.max(scale * W * 0.10, 10);
  ctx.beginPath(); ctx.arc(hx,hy,hr,0,Math.PI*2);
  if (aligned) { ctx.fillStyle=`${color}0.35)`; ctx.fill(); }
  ctx.strokeStyle=`${color}${alpha})`; ctx.lineWidth=2; ctx.stroke();

  // label
  const label = aligned ? "TARGET  MATCHED" : "TARGET POSE";
  const lcolor = aligned ? "#4cff4c" : "#4ccfcf";
  ctx.save();
  ctx.font = "bold 11px 'Arial Black', Arial";
  ctx.fillStyle = lcolor;
  ctx.textAlign = "center";
  const [lx, ly] = toPixel(0, -2.25);
  ctx.fillText(label, Math.min(Math.max(lx, 80), W-80), Math.max(ly, 20));
  ctx.restore();
}

function drawMoveName(ctx, W, H, moveName) {
  if (!moveName) return;
  const display = moveName.replace(/_/g," ").toUpperCase();
  ctx.save();
  ctx.font = `bold ${Math.floor(W * 0.028)}px 'Arial Black', Arial`;
  ctx.fillStyle = "rgba(255,45,45,0.5)";
  ctx.textAlign = "left";
  ctx.fillText(`MOVE: ${display}`, 16, H - 28);
  ctx.restore();
}

function drawNoBody(ctx, W, H) {
  ctx.save();
  ctx.font = `bold ${Math.floor(W * 0.025)}px 'Arial Black', Arial`;
  ctx.fillStyle = "rgba(80,80,80,0.7)";
  ctx.textAlign = "center";
  ctx.fillText("STAND IN FRAME", W/2, H/2 - 12);
  ctx.font = `${Math.floor(W * 0.018)}px Arial`;
  ctx.fillStyle = "rgba(55,55,55,0.7)";
  ctx.fillText("Full body must be visible", W/2, H/2 + 14);
  ctx.restore();
}

function drawBufferBar(ctx, W, H, fillRatio) {
  const bx=0, by=H-5, bw=W, bh=5;
  ctx.fillStyle = "rgba(255,255,255,0.06)"; ctx.fillRect(bx,by,bw,bh);
  const fill = bw * Math.min(fillRatio||0, 1.0);
  ctx.fillStyle = (fillRatio||0) >= 1 ? "#4cff4c" : "#ff2d2d";
  ctx.fillRect(bx, by, fill, bh);
}

export default function SkeletonCanvas({
  landmarks, visibility, ghostBones, ghostAnchor,
  aligned, fillRatio, moveName, width=640, height=480, mirror=false, clear=false,
}) {
  const canvasRef    = useRef(null);
  const containerRef = useRef(null);

  // resize canvas to match container
  useEffect(() => {
    const canvas    = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;
    const ro = new ResizeObserver(() => {
      canvas.width  = container.clientWidth;
      canvas.height = container.clientHeight;
    });
    ro.observe(container);
    canvas.width  = container.clientWidth;
    canvas.height = container.clientHeight;
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const W = canvas.width, H = canvas.height;
    if (!W || !H) return;

    if (clear) {
      ctx.clearRect(0,0,W,H);   // a camera picture sits underneath
    } else {
      ctx.fillStyle = "#060608"; ctx.fillRect(0,0,W,H);
      drawGraffitiBackground(ctx, W, H);
    }

    const hasBody = landmarks && landmarks.length > 0;
    if (!hasBody) {
      drawNoBody(ctx, W, H);
    } else {
      // Mirror view (second screen): flip the body but never the text.
      ctx.save();
      if (mirror) { ctx.translate(W, 0); ctx.scale(-1, 1); }
      drawGhost(ctx, W, H, ghostBones, ghostAnchor, aligned);
      drawSkeleton(ctx, W, H, landmarks, visibility||[]);
      ctx.restore();
    }

    drawMoveName(ctx, W, H, moveName);
    drawBufferBar(ctx, W, H, fillRatio);
  }, [landmarks, visibility, ghostBones, ghostAnchor, aligned, fillRatio, moveName, mirror, clear]);

  return (
    <div ref={containerRef} style={{ width:"100%", height:"100%", position:"relative" }}>
      <canvas
        ref={canvasRef}
        style={{ display:"block", width:"100%", height:"100%" }}
      />
    </div>
  );
}