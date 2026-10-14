# Plan: move-specific cues and a slow demo on request

Status labels: EXISTS means it is in code on this branch, PROPOSED means it is not built.

Jordan's ask: a real coach watching toprock says things like "twist your hips more" or "extend that leg out", and when you ask, shows you the move slowly. The app should too.

## Can the phone show a demo and still watch the dancer?

Yes. The camera keeps streaming while the screen shows something else; the session page already hides the camera feed and draws a skeleton instead. The demo is an overlay on that same page. The dancer watches it, then dances. While it plays the coach stays quiet, so it does not correct a dancer who is just watching. No second device is needed.

## The constraint

Every body claim must come from measured data (CLAUDE.md rule 1, spec 3.6). Until this branch the scorer only measured elbow and knee angles at four keyframes, so "twist your hips" could never be said truthfully, and the grounding guard refused it. So each new cue needs a new measured feature first, with its target taken from the move's own AIST++ reference clip.

## Which reference is toprock?

AIST++ labels its break clips by number, not by move name. The catalog measures each clip (`backend/app/move_catalog.py`):

- BR 01: no floor time, no inversion. A standing, stepping move, so it is the toprock candidate.
- BR 09: hips near the floor for most of the clip, no inversion. The footwork candidate.

Jordan should confirm both by watching the library preview. Once confirmed, add `display_name` to their meta files.

## Features per move

All features are measured on x and y only (the plane the phone camera sees and DTW compares), over the whole rep window, and compared as live / reference. They only flag "not enough"; doing more than the reference is not called wrong.

| Move | Feature | How it is measured | Spoken fix | Status |
| --- | --- | --- | --- | --- |
| Toprock (BR 01) | `hip_turn` | 1 minus narrowest / widest apparent hip width over the rep (10th and 90th percentile). Hips turning away from the camera look narrower. | "Twist your hips more" | EXISTS |
| Toprock (BR 01) | `leg_reach` | 90th percentile of how far a foot gets out to the side of the hips, in torso lengths. | "Extend that leg out" | EXISTS |
| Footwork (BR 09) | `leg_reach` | Same as above, for the sweeping leg. | "Kick that leg out wider" | PROPOSED (one meta line, needs a check on real footwork first) |
| Footwork (BR 09) | `hips_low` | Hip height above the lowest body point, in hip widths, reusing the catalog's floor measure. | "Get your hips lower" | PROPOSED |
| Any | shoulder and hip separation | Needs depth. MediaPipe depth from one phone camera is too rough. | none | Not planned |

Tiers: at least 75% of the reference is Perfect, at least 50% is Close, below that is Miss. These are starting values to tune on the phone, not measured. A feature counts toward a clean rep like any joint.

## From feature to spoken cue

1. The scorer adds one result per feature to the rep: `{"joint": "hips", "fix": "twist", "tier": ...}` (EXISTS). `diff` is how far short of the reference the rep fell, in percent, so it can be ranked with joint angle misses.
2. The live cue picks the worst Miss, else the worst Close, as before. The prompt says "their hips need to twist more through the rep"; the templated fallback says "Twist your hips more" (EXISTS).
3. The grounding guard already refuses a body part that is not in the facts. It now also refuses "twist" or "rotate" unless a hip twist was measured (EXISTS). So the model cannot say "twist your hips" on a move with no hip feature.
4. The debrief and the focus chip use the same `fix` value, so "Next one, twist your hips" works with no extra code (EXISTS).

## Demo on request

1. **Button first (EXISTS).** A SHOW ME button on the session page sends `demo_request`. The server answers with `demo`: the move id, half speed, two plays, and which preview joints to ring. The client plays the move's reference clip (the same frames as the move library preview) in an overlay with that joint ringed in yellow.
2. **What the coach says (EXISTS).** One templated line, for example "Here's BR 01 at half speed. Watch the hips. See how far they turn." The joint is the session's measured focus joint, so the spoken line and the ring always match and only name something the scorer flagged. With no focus yet it says "Watch the whole move." It is templated, not generated, because it must name exactly what the ring shows.
3. **Quiet while watching (EXISTS).** Live cues pause for the length of the demo.
4. **Voice request (PROPOSED).** "Show me" by voice waits for push-to-talk and speech to text (FR-11, M3). The STT text maps to the same `demo_request`.
5. **Text chat (PROPOSED).** FR-15 text chat can route "show me" to the same request.
6. **Later (PROPOSED).** Side-by-side: the demo figure next to the dancer's last rep, both slowed down, with the focus joint ringed on each.

## AIST++ license check (needed before sharing the app publicly)

The demo animates the same converted AIST++ keypoints the move library preview already shows, and those files are already committed. Before a public demo or app store build, confirm the AIST++ annotation license terms on the [AIST++ site](https://google.github.io/aichoreographer/) and add the required credit to the README. The source AIST Dance Video Database videos have their own terms; this app does not use or ship the videos.

## Limits of this slice

- The hip twist proxy assumes the dancer roughly faces the phone. A dancer turned side-on all the time looks "square".
- Feature cut-offs and the demo speed are design choices, not measured.
- No benchmark yet. No number about cue accuracy goes in the README until a script in `scripts/` produces it (CLAUDE.md rule 2).
- Left and right in highlights follow the existing joint names, which come from a mirrored camera image. That is unchanged from before.

## Manual device test (Jordan)

1. Start a session on BR 01 on the iPhone. Stand still: nothing should be said.
2. Toprock with square hips: after a few reps the cue or focus chip should say hips and twist.
3. Toprock with small steps: it should say leg and reach.
4. Tap SHOW ME: the overlay plays BR 01 at half speed with the focus joint ringed, the line is spoken, and no live cue talks over it. It closes on its own after two plays.
5. Note whether the cut-offs feel too strict or too loose.

## Next steps

1. Device test above, then tune the cut-offs.
2. Confirm toprock and footwork clip names, add `display_name`.
3. Add `leg_reach` and `hips_low` to the footwork move.
4. Extend the benchmark: feature values on labeled good and bad reps, committed under `docs/benchmarks/`.
5. Voice "show me" with push-to-talk (M3).
