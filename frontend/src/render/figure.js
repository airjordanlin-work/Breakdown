// Draws a pose from the move preview as either a plain skeleton or a
// character figure.
//
// Joint order matches backend/app/move_catalog.py PREVIEW_JOINTS + neck:
//   0 nose, 1 L shoulder, 2 R shoulder, 3 L elbow, 4 R elbow, 5 L wrist,
//   6 R wrist, 7 L hip, 8 R hip, 9 L knee, 10 R knee, 11 L ankle,
//   12 R ankle, 13 neck.   Each joint is [x, y, depth].
//
// The figure is built from named body parts (PARTS below). Limbs taper from
// one joint to the next, the torso narrows toward the waist, and the far side
// is drawn darker so overlapping limbs read. To swap in mascot artwork later,
// replace a part's draw function with one that draws an image rotated along
// the same two joints.

const J = { NOSE: 0, LSH: 1, RSH: 2, LEL: 3, REL: 4, LWR: 5, RWR: 6,
            LHIP: 7, RHIP: 8, LKN: 9, RKN: 10, LAN: 11, RAN: 12, NECK: 13 };

export const LOOK = {
  skin:   "#d9b79b",
  hoodie: "#44566b",
  trim:   "#e0a640",
  pants:  "#232833",
  shoes:  "#efece4",
  soles:  "#15171c",
  beanie: "#171a20",
  outline:"#07080a",
};

// Radii are in hip-width units (the data's unit of length): a at the first
// joint, b at the second.
const PARTS = [
  { name: "L thigh",     a: J.LHIP, b: J.LKN, ra: 0.36, rb: 0.27, color: "pants" },
  { name: "R thigh",     a: J.RHIP, b: J.RKN, ra: 0.36, rb: 0.27, color: "pants" },
  { name: "L shin",      a: J.LKN,  b: J.LAN, ra: 0.26, rb: 0.17, color: "pants" },
  { name: "R shin",      a: J.RKN,  b: J.RAN, ra: 0.26, rb: 0.17, color: "pants" },
  { name: "L upper arm", a: J.LSH,  b: J.LEL, ra: 0.25, rb: 0.2,  color: "hoodie" },
  { name: "R upper arm", a: J.RSH,  b: J.REL, ra: 0.25, rb: 0.2,  color: "hoodie" },
  { name: "L forearm",   a: J.LEL,  b: J.LWR, ra: 0.2,  rb: 0.15, color: "hoodie" },
  { name: "R forearm",   a: J.REL,  b: J.RWR, ra: 0.2,  rb: 0.15, color: "hoodie" },
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

// A limb segment that tapers from radius r1 at p to r2 at q: two end caps
// joined by the tangent trapezoid, with a thin outline.
function taper(ctx, p, q, r1, r2, color, outlineStart = true) {
  const dx = q[0] - p[0], dy = q[1] - p[1];
  const len = Math.hypot(dx, dy) || 1;
  const nx = -dy / len, ny = dx / len;
  const path = (grow, startCap = true) => {
    ctx.beginPath();
    if (startCap) ctx.arc(p[0], p[1], r1 + grow, 0, Math.PI * 2);
    ctx.moveTo(q[0] + r2 + grow, q[1]);
    ctx.arc(q[0], q[1], r2 + grow, 0, Math.PI * 2);
    ctx.moveTo(p[0] + nx * (r1 + grow), p[1] + ny * (r1 + grow));
    ctx.lineTo(q[0] + nx * (r2 + grow), q[1] + ny * (r2 + grow));
    ctx.lineTo(q[0] - nx * (r2 + grow), q[1] - ny * (r2 + grow));
    ctx.lineTo(p[0] - nx * (r1 + grow), p[1] - ny * (r1 + grow));
    ctx.closePath();
  };
  ctx.fillStyle = LOOK.outline;
  path(1.4, outlineStart); ctx.fill();
  ctx.fillStyle = color;
  path(0); ctx.fill();
}

function dot(ctx, p, r, color) {
  ctx.fillStyle = LOOK.outline;
  ctx.beginPath(); ctx.arc(p[0], p[1], r + 1.4, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = color;
  ctx.beginPath(); ctx.arc(p[0], p[1], r, 0, Math.PI * 2); ctx.fill();
}

function shoe(ctx, knee, ankle, unit, color) {
  // Sneaker: a rounded block past the ankle with a dark sole and a trim stripe.
  const dx = ankle[0] - knee[0], dy = ankle[1] - knee[1];
  const len = Math.hypot(dx, dy) || 1;
  const ux = dx / len, uy = dy / len;
  const cx = ankle[0] + ux * unit * 0.1, cy = ankle[1] + uy * unit * 0.1;
  ctx.save();
  ctx.translate(cx, cy);
  ctx.rotate(Math.atan2(uy, ux) - Math.PI / 2);
  const w = unit * 0.6, h = unit * 0.3;
  ctx.fillStyle = LOOK.outline;
  ctx.beginPath(); ctx.roundRect(-w / 2 - 1.4, -h / 2 - 1.4, w + 2.8, h + 2.8, h / 2); ctx.fill();
  ctx.fillStyle = color;
  ctx.beginPath(); ctx.roundRect(-w / 2, -h / 2, w, h, h / 2); ctx.fill();
  ctx.fillStyle = LOOK.trim;
  ctx.fillRect(-w * 0.1, -h / 2 + h * 0.12, w * 0.16, h * 0.34);
  ctx.fillStyle = LOOK.soles;
  ctx.fillRect(-w / 2 + h / 3, h / 2 - h * 0.26, w - (2 * h) / 3, h * 0.26);
  ctx.restore();
}

// Rounded polygon: corners are eased with quadratic curves.
function roundedPoly(ctx, pts, radius) {
  const n = pts.length;
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const prev = pts[(i + n - 1) % n], cur = pts[i], next = pts[(i + 1) % n];
    const d1 = Math.hypot(cur[0] - prev[0], cur[1] - prev[1]) || 1;
    const d2 = Math.hypot(next[0] - cur[0], next[1] - cur[1]) || 1;
    const r1 = Math.min(radius, d1 / 2), r2 = Math.min(radius, d2 / 2);
    const a = [cur[0] + (prev[0] - cur[0]) * (r1 / d1), cur[1] + (prev[1] - cur[1]) * (r1 / d1)];
    const b = [cur[0] + (next[0] - cur[0]) * (r2 / d2), cur[1] + (next[1] - cur[1]) * (r2 / d2)];
    if (i === 0) ctx.moveTo(a[0], a[1]); else ctx.lineTo(a[0], a[1]);
    ctx.quadraticCurveTo(cur[0], cur[1], b[0], b[1]);
  }
  ctx.closePath();
}

export function drawFigure(ctx, pose, map, unit) {
  const P = pose.map(map);
  const depth = (i) => pose[i][2];
  const nearSide = (depth(J.LSH) + depth(J.LHIP)) >= (depth(J.RSH) + depth(J.RHIP)) ? "L" : "R";
  const isFar = (name) => !name.startsWith(nearSide);
  const farSuffix = nearSide === "L" ? "R" : "L";
  const J_ = (side, l, r) => (side === "L" ? l : r);

  // Far legs and arms first, then the near leg, then the torso (its hem covers
  // the hip ends of the legs), then the head and the near arm.
  const far = PARTS.filter((p) => isFar(p.name));
  const near = PARTS.filter((p) => !isFar(p.name));
  const isLeg = (p) => p.color === "pants";

  const drawPart = (part, farSide) => {
    const base = LOOK[part.color];
    // Hip and shoulder ends blend into the torso, so only the far end is outlined.
    const rootEnd = part.a === J.LHIP || part.a === J.RHIP || part.a === J.LSH || part.a === J.RSH;
    taper(ctx, P[part.a], P[part.b], part.ra * unit, part.rb * unit,
          farSide ? shade(base, 0.35) : base, !rootEnd);
  };

  ctx.lineJoin = "round";
  far.forEach((p) => drawPart(p, true));
  shoe(ctx, P[J_(farSuffix, J.LKN, J.RKN)], P[J_(farSuffix, J.LAN, J.RAN)], unit, shade(LOOK.shoes, 0.3));
  dot(ctx, P[J_(farSuffix, J.LWR, J.RWR)], unit * 0.15, shade(LOOK.skin, 0.3));
  near.filter(isLeg).forEach((p) => drawPart(p, false));
  shoe(ctx, P[J_(nearSide, J.LKN, J.RKN)], P[J_(nearSide, J.LAN, J.RAN)], unit, LOOK.shoes);

  // Torso: hoodie body, wide at the shoulders and narrower at the waist.
  const lsh = P[J.LSH], rsh = P[J.RSH], lhp = P[J.LHIP], rhp = P[J.RHIP];
  const mix = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
  const waistL = mix(lhp, rhp, -0.05), waistR = mix(lhp, rhp, 1.05);
  const shL = mix(lsh, rsh, -0.1), shR = mix(lsh, rsh, 1.1);
  const torso = [shL, shR, waistR, waistL];
  const pad = unit * 0.14;
  const hoodShade = nearSide === "L" ? 0.18 : 0;
  roundedPoly(ctx, torso, unit * 0.3);
  ctx.lineWidth = pad * 2 + 2.8; ctx.strokeStyle = LOOK.outline; ctx.stroke();
  ctx.lineWidth = pad * 2; ctx.strokeStyle = LOOK.hoodie; ctx.stroke();
  ctx.fillStyle = LOOK.hoodie; ctx.fill();
  // side shading toward the far shoulder
  const g = ctx.createLinearGradient(shL[0], shL[1], shR[0], shR[1]);
  g.addColorStop(0, hoodShade ? shade(LOOK.hoodie, 0.28) : "rgba(0,0,0,0)");
  g.addColorStop(0.6, "rgba(0,0,0,0)");
  g.addColorStop(1, hoodShade ? "rgba(0,0,0,0)" : shade(LOOK.hoodie, 0.28));
  ctx.fillStyle = g; ctx.fill();
  // hem band at the waist
  taper(ctx, waistL, waistR, unit * 0.14, unit * 0.14, shade(LOOK.hoodie, 0.3));
  // Neck, hood collar, head and beanie.
  const head = P[J.NOSE], neck = P[J.NECK];
  taper(ctx, neck, head, unit * 0.17, unit * 0.15, shade(LOOK.skin, 0.1));
  ctx.fillStyle = shade(LOOK.hoodie, 0.22);
  ctx.beginPath(); ctx.arc(neck[0], neck[1], unit * 0.27, 0, Math.PI * 2); ctx.fill();
  const r = unit * 0.46;
  const up = Math.atan2(head[1] - neck[1], head[0] - neck[0]);
  const hc = [head[0] + Math.cos(up) * r * 0.25, head[1] + Math.sin(up) * r * 0.25];
  dot(ctx, hc, r, LOOK.skin);
  ctx.fillStyle = LOOK.beanie;
  ctx.beginPath();
  ctx.arc(hc[0], hc[1], r + 0.5, up - Math.PI * 0.5, up + Math.PI * 0.5);
  ctx.closePath();
  ctx.fill();
  // beanie cuff, a trim-colored band across the forehead
  ctx.save();
  ctx.translate(hc[0], hc[1]);
  ctx.rotate(up - Math.PI / 2);
  ctx.fillStyle = LOOK.trim;
  ctx.fillRect(-r, -r * 0.1, r * 2, r * 0.22);
  ctx.restore();

  near.filter((p) => !isLeg(p)).forEach((p) => drawPart(p, false));
  dot(ctx, P[J_(nearSide, J.LWR, J.RWR)], unit * 0.15, LOOK.skin);
}
