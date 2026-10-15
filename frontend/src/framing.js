// "Frame the whole body" check (FR-3). Uses the same 0.4 visibility cutoff as
// the backend's _diagnose_visibility, on the landmarks the pose model needs
// to score a move: head, shoulders, hips and ankles.

const SEEN = 0.4;
const NOSE = 0, L_SHOULDER = 11, R_SHOULDER = 12, L_HIP = 23, R_HIP = 24, L_ANKLE = 27, R_ANKLE = 28;

// Frames in a row needed to call the dancer framed, so one lucky frame doesn't
// dismiss the guide.
export const READY_FRAMES = 10;
// Once framed, the guide only returns after the dancer has been gone (no pose,
// or no torso) this long. Time based, so it doesn't depend on the frame rate.
// Floorwork and headspins hide the head and feet briefly, which is not "lost".
export const LOST_MS = 1500;

const seen = (vis, ...ids) => ids.every((i) => (vis?.[i] ?? 0) >= SEEN);
const torsoSeen = (vis) => seen(vis, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP);

// What to tell the dancer, or null when the whole body is in frame.
export function framingProblem(vis) {
  if (!vis || vis.length === 0) return "STEP INTO FRAME";
  if (!torsoSeen(vis)) return "STEP BACK, TORSO NOT IN FRAME";
  if (!seen(vis, L_ANKLE, R_ANKLE)) return "STEP BACK, FEET NOT IN FRAME";
  if (!seen(vis, NOSE)) return "STEP BACK, HEAD NOT IN FRAME";
  return null;
}

export const INITIAL_FRAMING = { ready: false, good: 0, lostSince: null, problem: "STEP INTO FRAME" };

// Fold one frame's visibility into the framing state. Pass the frame's own
// visibility (live_visibility), not the held one, or a dancer who left the
// frame is never noticed. `now` is in milliseconds.
export function nextFraming(state, vis, now = Date.now()) {
  const problem = framingProblem(vis);
  const good = problem ? 0 : state.good + 1;
  if (!state.ready) {
    return { ready: good >= READY_FRAMES, good, lostSince: null, problem };
  }
  const gone = !vis || vis.length === 0 || !torsoSeen(vis);
  const lostSince = gone ? (state.lostSince ?? now) : null;
  const ready = !(lostSince !== null && now - lostSince >= LOST_MS);
  return { ready, good: ready ? good : 0, lostSince: ready ? lostSince : null, problem };
}
