# Plan: a coach that talks like a person

Status labels: EXISTS means it is in code on this branch, PROPOSED means it is not built.

The goal is a coach that holds a conversation, not one that plays a few stock lines. That happens in three layers, and each layer must only say what the app measured.

## Where this sits in the spec

- Live cues and audio to the phone are M1 (FR-10). EXISTS, with PR 31 (quiet when idle) and PR 32 (clean reps against real joint targets).
- Debrief after an attempt is FR-14 (P1). The spec lists it under the conversational coach (section 3.1) and does not tie it to a milestone. It needs no new hardware access, so it fits now.
- Push-to-talk and speech-to-text are M3 (FR-11, FR-12). Barge-in is M4 (FR-13). Text chat fallback is FR-15.
- CLAUDE.md rule 11: grounding and tool-calling logic for the coach is an Opus task from M3 on. This thread already runs on Opus.

## 1. Live cues (while moving)

What it should sound like: a coach standing next to you. Short (8 words max, spec 3.1), varied, spoken, never a number or an angle.

What it may say, from the `facts` block for that rep only:

- the move name
- the one joint to fix and which way (straighten or bend), from the scorer
- whether that joint was also the problem last rep ("still that left elbow")
- a joint that was off last rep and landed this rep ("there it is")
- the current clean streak, as the exact number

What makes it feel human: the model sees its own last few lines and is told not to repeat them, can call back to the previous rep, and gets the streak so it can hype a run. Anything it says goes through the grounding guard; a failure falls back to a templated cue built from the same facts.

## 2. Debrief (when the dancer stops)

Trigger: the dancer was moving, at least one rep was scored, and then they have been still or out of frame for 2 s (a starting value to tune on the phone).

What it says: 3 to 5 sentences, about 35 to 55 words (spec 3.2): praise, explain the fix and why it matters, then an uplift, spoken, from that attempt's measured reps only: reps, clean reps, best streak, which joint was off most and which way, which joints landed on every rep. It may end with one short question. No points, no grade, no comparison to past sessions (nothing from past sessions is stored yet).

Fallback: a templated debrief from the same numbers when the model is missing, slow, too long or fails the guard.

## 3. Open conversation (the dancer asks)

The user asks things like "why do I keep missing the elbow?" or "how am I doing?". The answer cites the session's measured data, or says plainly that it is not measured ("I can't see your feet from here", "I don't have last week's session").

What it needs that does not exist yet (all PROPOSED):

1. **Text chat input** (FR-15). A text box on the session page that sends `text_message {text}` over the existing socket. Cheapest way to test the whole conversation loop before any audio input. Can be built in the cloud.
2. **Conversation memory within a session** (spec 3.5 short-term). A list of the last N coach and user turns plus the attempt history from the debrief layer, passed into each prompt. Cloud-buildable.
3. **Tools for the model** (spec 3.6): `get_recent_attempts`, `get_move_info`, later `get_progress`. Results become facts the guard accepts.
4. **Push-to-talk and speech-to-text on the phone** (FR-11, M3). Hold a button, capture mic audio, send it, get text back. STT choice is still OPEN in the spec decision log. Recommended first try: browser speech recognition in iOS Safari for zero cost and no server work, with a hosted STT as the fallback if accuracy with music is poor. Needs Jordan's iPhone and a real mic, so the device test is local only.
5. **Turn-taking** (M4): queue speech until a pause, barge-in stops TTS. Needed before voice chat feels natural.
6. **Cross-session memory** (FR-17, M5): needed before "you're cleaner than last week" can ever be said truthfully.

## Recommended order

1. Grounded LLM live cues and stop-time debrief, with the guard and tests. **This PR.**
2. Text chat fallback with in-session memory and the same guard (FR-15). Cloud-buildable, testable with a fake model.
3. Measure: extend `benchmark_pipeline.py coach` with a debrief stage and token counts, and label cues for the grounding rate (NFR-11, M2).
4. Push-to-talk and STT (M3), device test on the iPhone.
5. Barge-in and speech queueing (M4).
6. Personas and cross-session memory (M5).

## Cost and latency per turn

No figure goes in docs until a committed benchmark produces it (CLAUDE.md rule 2).

- Live cue: one Haiku call, small prompt, 40 output tokens max. Latency is already measured by `benchmark_pipeline.py coach` (spec NFR-1).
- Debrief: one Haiku call, larger facts block, 120 output tokens max, plus a longer TTS synth. Not measured yet; add a `debrief` stage to the benchmark.
- Open conversation: one call per user turn plus STT. Not measured; NFR-2 sets the voice-to-voice target.
- Cost per 10-minute session (NFR-6) should be computed by the benchmark from logged input and output token counts times the published price, not estimated by hand.

## Which model for which part

- Live cues: Claude Haiku. Latency matters most and the job is tiny.
- Debrief: Claude Haiku. Short, structured, grounded summary.
- Open conversation: Claude Haiku by default, matching the spec decision log. Try a stronger model only if answers are poor, and measure the latency cost first.
- STT: OPEN (see above). TTS: Kokoro stays for now (PR 30 fixes voice loading).

## Defaults picked (change any of these)

- Voice and tone: friendly, plain, a little hype on streaks, no forced slang. Same Kokoro voice as now.
- Debrief length: up to 5 sentences, about 55 words (changed from 3 sentences, 25 words after Jordan asked for coaching that explains and uplifts).
- Pause before a debrief: 2 s of stillness or no body.
- Cost limit: none set in code. Live cues keep their cooldown (5 s with joint data), and there is at most one debrief per attempt.
