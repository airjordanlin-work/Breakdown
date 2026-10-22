// Draws a pose from the move preview as either a plain skeleton or a flat
// silhouette figure.
//
// Joint order matches backend/app/move_catalog.py PREVIEW_JOINTS + neck:
//   0 nose, 1 L shoulder, 2 R shoulder, 3 L elbow, 4 R elbow, 5 L wrist,
//   6 R wrist, 7 L hip, 8 R hip, 9 L knee, 10 R knee, 11 L ankle,
//   12 R ankle, 13 neck.   Each joint is [x, y, depth].
//
// The silhouette is one flat color with no face or clothes, so it reads as
// clean and neutral. Limbs taper from joint to joint, the torso narrows toward
// the waist, and the limbs on the far side of the body are a shade darker so
// overlapping limbs still read. To swap in artwork later, replace the shape
// drawing with images rotated along the same joint pairs.

const J = { NOSE: 0, LSH: 1, RSH: 2, LEL: 3, REL: 4, LWR: 5, RWR: 6,
            LHIP: 7, RHIP: 8, LKN: 9, RKN: 10, LAN: 11, RAN: 12, NECK: 13 };

export const LOOK = {
  body: "#eee9e0",
  far:  "#a9a59d",
};

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

// A limb segment that tapers from radius r1 at p to r2 at q. With startCap
// false the p end is flat, for ends that tuck under the torso.
function limb(ctx, p, q, r1, r2, color, startCap = true) {
  const dx = q[0] - p[0], dy = q[1] - p[1];
  const len = Math.hypot(dx, dy) || 1;
  const nx = -dy / len, ny = dx / len;
  ctx.fillStyle = color;
  // Three separate fills: in one path the arcs and the trapezoid can wind in
  // opposite directions and leave hairline gaps where they overlap.
  if (startCap) { ctx.beginPath(); ctx.arc(p[0], p[1], r1, 0, Math.PI * 2); ctx.fill(); }
  ctx.beginPath(); ctx.arc(q[0], q[1], r2, 0, Math.PI * 2); ctx.fill();
  ctx.beginPath();
  ctx.moveTo(p[0] + nx * r1, p[1] + ny * r1);
  ctx.lineTo(q[0] + nx * r2, q[1] + ny * r2);
  ctx.lineTo(q[0] - nx * r2, q[1] - ny * r2);
  ctx.lineTo(p[0] - nx * r1, p[1] - ny * r1);
  ctx.closePath();
  ctx.fill();
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
  const nearIsLeft = (depth(J.LSH) + depth(J.LHIP)) >= (depth(J.RSH) + depth(J.RHIP));

  // Radii are in hip-width units (the data's unit of length).
  const leg = (left, color) => {
    const [hip, kn, an] = left ? [J.LHIP, J.LKN, J.LAN] : [J.RHIP, J.RKN, J.RAN];
    limb(ctx, P[hip], P[kn], unit * 0.31, unit * 0.24, color, false);   // hip end tucks under the torso
    limb(ctx, P[kn], P[an], unit * 0.24, unit * 0.15, color);
    // foot: a short taper pointing on from the knee-to-ankle direction
    const dx = P[an][0] - P[kn][0], dy = P[an][1] - P[kn][1];
    const len = Math.hypot(dx, dy) || 1;
    const tip = [P[an][0] + (dx / len) * unit * 0.28, P[an][1] + (dy / len) * unit * 0.28];
    limb(ctx, P[an], tip, unit * 0.15, unit * 0.11, color);
  };
  const arm = (left, color) => {
    const [sh, el, wr] = left ? [J.LSH, J.LEL, J.LWR] : [J.RSH, J.REL, J.RWR];
    limb(ctx, P[sh], P[el], unit * 0.22, unit * 0.17, color);
    limb(ctx, P[el], P[wr], unit * 0.17, unit * 0.12, color);
  };

  // Far side first, then near leg, then torso and head, then the near arm.
  const farLeft = !nearIsLeft;
  leg(farLeft, LOOK.far);
  arm(farLeft, LOOK.far);
  leg(nearIsLeft, LOOK.body);

  // Torso: wider at the shoulders, narrower at the waist.
  const mix = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
  const torso = [
    mix(P[J.LSH], P[J.RSH], -0.08), mix(P[J.LSH], P[J.RSH], 1.08),
    mix(P[J.LHIP], P[J.RHIP], 1.0), mix(P[J.LHIP], P[J.RHIP], 0.0),
  ];
  const pad = unit * 0.13;
  ctx.fillStyle = LOOK.body;
  ctx.strokeStyle = LOOK.body;
  ctx.lineJoin = "round";
  ctx.lineWidth = pad * 2;
  roundedPoly(ctx, torso, unit * 0.3);
  ctx.stroke();
  ctx.fill();

  // Neck and head.
  const head = P[J.NOSE], neck = P[J.NECK];
  limb(ctx, neck, head, unit * 0.15, unit * 0.14, LOOK.body);
  const up = Math.atan2(head[1] - neck[1], head[0] - neck[0]);
  ctx.fillStyle = LOOK.body;
  ctx.beginPath();
  ctx.arc(head[0] + Math.cos(up) * unit * 0.1, head[1] + Math.sin(up) * unit * 0.1, unit * 0.42, 0, Math.PI * 2);
  ctx.fill();

  arm(nearIsLeft, LOOK.body);
}
