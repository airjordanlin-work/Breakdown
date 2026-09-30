// Draws a pose from the move preview as either a plain skeleton or a
// character figure.
//
// Joint order matches backend/app/move_catalog.py PREVIEW_JOINTS + neck:
//   0 nose, 1 L shoulder, 2 R shoulder, 3 L elbow, 4 R elbow, 5 L wrist,
//   6 R wrist, 7 L hip, 8 R hip, 9 L knee, 10 R knee, 11 L ankle,
//   12 R ankle, 13 neck.   Each joint is [x, y, depth].
//
// The figure is built from named body parts (PARTS below). Each part has a
// look; to swap in mascot artwork later, replace a part's draw function
// with one that draws an image rotated along the same two joints.

const J = { NOSE: 0, LSH: 1, RSH: 2, LEL: 3, REL: 4, LWR: 5, RWR: 6,
            LHIP: 7, RHIP: 8, LKN: 9, RKN: 10, LAN: 11, RAN: 12, NECK: 13 };

export const LOOK = {
  skin:   "#eee9e0",
  hoodie: "#ff2d2d",
  pants:  "#2e2e36",
  shoes:  "#f7f5f0",
  soles:  "#16161a",
  beanie: "#16161a",
  outline:"#060608",
};

// Thickness in hip-width units (the data's unit of length).
const PARTS = [
  { name: "L thigh",   a: J.LHIP, b: J.LKN, w: 0.62, color: "pants" },
  { name: "R thigh",   a: J.RHIP, b: J.RKN, w: 0.62, color: "pants" },
  { name: "L shin",    a: J.LKN,  b: J.LAN, w: 0.48, color: "pants" },
  { name: "R shin",    a: J.RKN,  b: J.RAN, w: 0.48, color: "pants" },
  { name: "L upper arm", a: J.LSH, b: J.LEL, w: 0.44, color: "hoodie" },
  { name: "R upper arm", a: J.RSH, b: J.REL, w: 0.44, color: "hoodie" },
  { name: "L forearm", a: J.LEL,  b: J.LWR, w: 0.38, color: "hoodie" },
  { name: "R forearm", a: J.REL,  b: J.RWR, w: 0.38, color: "hoodie" },
];

export function lerpPose(a, b, t) {
  return a.map((p, i) => p.map((v, k) => v + (b[i][k] - v) * t));
}

export function drawSkeleton(ctx, pose, bones, map, color, width, alpha) {
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.lineCap = "round";
  ctx.beginPath();
  for (const [a, b] of bones) {
    const [ax, ay] = map(pose[a]);
    const [bx, by] = map(pose[b]);
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
  }
  ctx.stroke();
  ctx.globalAlpha = 1;
}

// Darken a hex color: far limbs are drawn slightly darker to read as behind.
function shade(hex, amount) {
  const n = parseInt(hex.slice(1), 16);
  const f = (c) => Math.max(0, Math.round(c * (1 - amount)));
  const r = f(n >> 16), g = f((n >> 8) & 255), b = f(n & 255);
  return `rgb(${r},${g},${b})`;
}

function capsule(ctx, p, q, width, color) {
  ctx.strokeStyle = LOOK.outline;
  ctx.lineCap = "round";
  ctx.lineWidth = width + 3;
  ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.lineTo(q[0], q[1]); ctx.stroke();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.lineTo(q[0], q[1]); ctx.stroke();
}

function dot(ctx, p, r, color) {
  ctx.fillStyle = LOOK.outline;
  ctx.beginPath(); ctx.arc(p[0], p[1], r + 1.5, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = color;
  ctx.beginPath(); ctx.arc(p[0], p[1], r, 0, Math.PI * 2); ctx.fill();
}

function shoe(ctx, knee, ankle, unit, color) {
  // Sneaker: a rounded block past the ankle, perpendicular-ish to the shin.
  const dx = ankle[0] - knee[0], dy = ankle[1] - knee[1];
  const len = Math.hypot(dx, dy) || 1;
  const ux = dx / len, uy = dy / len;
  const cx = ankle[0] + ux * unit * 0.12, cy = ankle[1] + uy * unit * 0.12;
  ctx.save();
  ctx.translate(cx, cy);
  ctx.rotate(Math.atan2(uy, ux) - Math.PI / 2);
  const w = unit * 0.62, h = unit * 0.3;
  ctx.fillStyle = LOOK.outline;
  ctx.beginPath(); ctx.roundRect(-w / 2 - 1.5, -h / 2 - 1.5, w + 3, h + 3, h / 2); ctx.fill();
  ctx.fillStyle = color;
  ctx.beginPath(); ctx.roundRect(-w / 2, -h / 2, w, h, h / 2); ctx.fill();
  ctx.fillStyle = LOOK.soles;
  ctx.fillRect(-w / 2 + h / 3, h / 2 - h * 0.28, w - (2 * h) / 3, h * 0.28);
  ctx.restore();
}

export function drawFigure(ctx, pose, map, unit) {
  const P = pose.map(map);
  const depth = (i) => pose[i][2];
  const nearSide = (depth(J.LSH) + depth(J.LHIP)) >= (depth(J.RSH) + depth(J.RHIP)) ? "L" : "R";
  const isFar = (name) => !name.startsWith(nearSide);

  // Far-side limbs first, then torso and head, then near-side limbs.
  const far = PARTS.filter((p) => isFar(p.name));
  const near = PARTS.filter((p) => !isFar(p.name));

  const drawPart = (part, farSide) => {
    const base = LOOK[part.color];
    capsule(ctx, P[part.a], P[part.b], part.w * unit, farSide ? shade(base, 0.35) : base);
  };

  far.forEach((p) => drawPart(p, true));
  shoe(ctx, P[nearSide === "L" ? J.RKN : J.LKN], P[nearSide === "L" ? J.RAN : J.LAN], unit, shade(LOOK.shoes, 0.25));
  dot(ctx, P[nearSide === "L" ? J.RWR : J.LWR], unit * 0.2, shade(LOOK.skin, 0.25));

  // Torso: hoodie body from shoulders to hips.
  const torso = [P[J.LSH], P[J.RSH], P[J.RHIP], P[J.LHIP]];
  ctx.lineJoin = "round";
  ctx.fillStyle = LOOK.hoodie;
  ctx.strokeStyle = LOOK.outline;
  ctx.lineWidth = unit * 0.34 + 3;
  ctx.beginPath();
  torso.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  ctx.closePath();
  ctx.stroke();
  ctx.strokeStyle = LOOK.hoodie;
  ctx.lineWidth = unit * 0.34;
  ctx.stroke();
  ctx.fill();
  // waistband
  capsule(ctx, P[J.LHIP], P[J.RHIP], unit * 0.34, shade(LOOK.hoodie, 0.25));

  // Neck and head, with a beanie on the side away from the neck.
  capsule(ctx, P[J.NECK], P[J.NOSE], unit * 0.3, LOOK.skin);
  const head = P[J.NOSE], neck = P[J.NECK];
  const r = unit * 0.5;
  dot(ctx, head, r, LOOK.skin);
  const up = Math.atan2(head[1] - neck[1], head[0] - neck[0]);
  ctx.fillStyle = LOOK.beanie;
  ctx.beginPath();
  ctx.arc(head[0], head[1], r + 1, up - Math.PI / 2.1, up + Math.PI / 2.1);
  ctx.closePath();
  ctx.fill();

  near.forEach((p) => drawPart(p, false));
  shoe(ctx, P[nearSide === "L" ? J.LKN : J.RKN], P[nearSide === "L" ? J.LAN : J.RAN], unit, LOOK.shoes);
  dot(ctx, P[nearSide === "L" ? J.LWR : J.RWR], unit * 0.2, LOOK.skin);
}
