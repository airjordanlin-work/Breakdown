// "Frame the whole body" check (FR-3). Uses the same 0.4 visibility cutoff as
// the backend's _diagnose_visibility, on the landmarks the pose model needs
// to score a move: head, shoulders, hips and ankles.

const SEEN = 0.4;
const NOSE = 0, L_SHOULDER = 11, R_SHOULDER = 12, L_HIP = 23, R_HIP = 24, L_ANKLE = 27, R_ANKLE = 28;

// Frames in a row needed to call the dancer framed, and lost, so one noisy
// frame doesn't flicker the guide.
export const READY_FRAMES = 10;
export const LOST_FRAMES  = 30;

const seen = (vis, ...ids) => ids.every((i) => (vis?.[i] ?? 0) >= SEEN);

// What to tell the dancer, or null when the whole body is in frame.
export function framingProblem(vis) {
  if (!vis || vis.length === 0) return "STEP INTO FRAME";
  if (!seen(vis, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)) return "STEP BACK, TORSO NOT IN FRAME";
  if (!seen(vis, L_ANKLE, R_ANKLE)) return "STEP BACK, FEET NOT IN FRAME";
  if (!seen(vis, NOSE)) return "STEP BACK, HEAD NOT IN FRAME";
  return null;
}

export const INITIAL_FRAMING = { ready: false, good: 0, bad: 0, problem: "STEP INTO FRAME" };

// Fold one frame's visibility into the framing state.
export function nextFraming(state, vis) {
  const problem = framingProblem(vis);
  const good = problem ? 0 : state.good + 1;
  const bad  = problem ? state.bad + 1 : 0;
  let ready = state.ready;
  if (!ready && good >= READY_FRAMES) ready = true;
  if (ready && bad >= LOST_FRAMES) ready = false;
  return { ready, good, bad, problem };
}
