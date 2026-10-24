# Push-to-talk and the voice orchestrator (plan)

Status: PROPOSED. Nothing here is built. Implements FR-11, FR-12, FR-15 and starts FR-13. Milestones M3 and M4 in the spec.

## Goal

The dancer holds a button, says something ("how was that?", "what is wrong with my elbow?"), lets go, and hears a short, grounded answer from the coach. Random or abusive talk must not be able to run up the model bill.

## Decisions to make

| Decision | Recommendation | Why |
|---|---|---|
| Order of work | Text chat first, then push-to-talk | Both use the same turn path. Text chat is fully testable in the cloud and needs no mic. |
| Speech to text | Behind a small interface. Default to local faster-whisper (base.en) on the Mac. Hosted STT as a switchable alternative. | Free per turn, no audio leaves the machine (NFR-7). The cost is latency on a 2019 Intel CPU, so it must be measured before we promise anything. |
| Audio capture | Record 16 kHz mono PCM in the browser (AudioWorklet) and send raw chunks | iPhone Safari records MediaRecorder audio as AAC in mp4 and Chrome as webm/opus. Raw PCM avoids ffmpeg and codec differences on the server. |
| Coach model | Haiku for replies, same as cues and debriefs | Already in use, cheapest. Reconsider only if replies are poor. |
| Spend limits | Enforced on the server, on by default | The client cannot be trusted to limit itself. |

## Turn flow

1. Phone: tap-and-hold button. On the first press the page asks for the mic (needs a user gesture, HTTPS already provided by the tunnel). Echo cancellation and noise suppression on.
2. Phone sends `ptt_start`, then PCM chunks, then `ptt_end`. Hard stop at 8 seconds on both sides.
3. Backend states: idle, listening, transcribing, thinking, speaking. One user turn at a time.
4. STT turns the clip into text. The audio is dropped immediately (rule 7). Only the transcript is kept, in memory, for the last few turns.
5. Gate (see spend limits). If it fails, send a templated reply with no model call.
6. Coach builds the prompt from the facts block (scores, last rep summary, the move) and any tool results, plus the last 4 turns. Haiku answers in one or two sentences. The existing grounding guard checks the answer. A failed guard falls back to a templated line (rule 1, rule 8).
7. Reply is sent as `transcript` (what we heard), `coach_cue` with mode `reply` and the TTS audio, using the existing audio streaming and caption path (NFR-17).
8. A user turn takes priority over any queued cue or debrief. Live cues are already off by default, so the queue is short.

## Protocol additions (spec section 8, to be added with the code)

- Phone to backend: `ptt_start`, binary PCM chunks, `ptt_end`; `chat {text}` for the text box.
- Backend to phone and viewers: `transcript {text, turn_id}`, `coach_cue {mode:"reply"}`, `chat_limit {reason}` when a turn is refused.

## Spend limits (the cost worry)

All server side, all configurable in `.env.example`, all logged per turn (NFR-18).

1. Clip length: max 8 s, min 0.4 s. Longer or shorter clips are dropped before STT.
2. Rate: at most 6 user turns per minute and a per-session cap (default 60). Over the limit gets a templated line, no model call.
3. Junk filter before the model: empty or under 2 words, no letters, a repeat of the previous transcript, or STT confidence too low gets a templated "say that again?".
4. Small prompts: facts block plus last 4 turns, `max_tokens` capped (about 120), one sentence of output preferred.
5. Scope: the system prompt keeps the coach on dancing, the move and the scores. Off-topic gets a short redirect.
6. Budget: a per-day token budget (input plus output), counted from the API usage fields. When it is spent the coach answers with templates only and says so. It fails closed.
7. Logs: per turn, STT ms, LLM ms, TTS ms, tokens in and out, gate result. No transcripts in logs.

NFR-6 asks for under $0.10 per 10-minute session. We should compute that from logged tokens and the current Haiku price, not from memory of the price. The benchmark script below produces the number.

## Work breakdown (one concern per PR)

1. **Orchestrator core and text chat** (cloud). State machine, turn gate, limits, budget counters, `chat` message, reply path with grounding guard and templated fallbacks, simple chat box in the UI. Pytest covers the gate, the limits, the budget fail-closed, and priority over cues. Implements FR-15 and the reply half of FR-11/FR-12.
2. **STT interface and local engine** (cloud for code, Mac for speed). `stt.py` with a fake engine for tests and a faster-whisper engine. Tests use synthetic audio from Kokoro for plumbing; real accuracy needs your Mac.
3. **Push-to-talk in the frontend** (cloud for code, phone for the test). Hold button, AudioWorklet capture to 16 kHz PCM, `ptt_*` messages, "listening" state, caption of what was heard. Needs your iPhone.
4. **Benchmark script** (cloud for the harness, Mac for numbers). `scripts/benchmark_pipeline.py voice`: STT ms by clip length, LLM ms, TTS ms, tokens per turn, cost per 10-minute session. Output committed under `docs/benchmarks/` (rule 2). No numbers go in the README until that output exists.
5. **Barge-in (M4, later, separate plan).** Stopping coach audio when the user presses the button. Stop-time target under 300 ms (NFR-3).

## What an unattended run can finish

PRs 1, 2 (code and fake-engine tests), 3 (code and build), and the harness for 4. It stops there because rule 10 applies: the real mic, iPhone Safari behaviour, real STT speed and accuracy with music, and the cost number all need your hardware. The PR descriptions will list the exact manual steps.

## Risks

- Intel CPU may make local Whisper too slow for NFR-2 (voice to voice under 1.5 s median). Mitigation: smaller model, or a hosted STT, chosen from the benchmark.
- Music and floor noise will hurt STT (NFR-14). Hold-to-talk limits the exposure; we measure and publish word error rate.
- iOS Safari mic permission and audio session quirks can interrupt coach playback after recording. Needs a phone test.
- A cap that is too tight feels broken. Defaults are starting guesses to tune after real use.

## Questions for Jordan

1. Local Whisper first, or a hosted STT from the start?
2. Daily token budget to start with, and per-session turn cap? (The defaults above are guesses.)
3. Should the text chat box ship in the main UI, or stay behind a debug flag?
