# Test automation before deploy

Status: layers 1 to 4 EXISTS in this PR. Layers 5 and 6 are PROPOSED.

Goal: stop hand-testing the things a computer can check, so an iPhone run only covers what needs a real phone.

## What Jordan was checking by hand

Live coach speech and debrief, SHOW ME, whether the top bar fits in portrait, push-to-talk, the result card. Most of that is layout and message flow, which a browser test can check. Audio output, camera quality and pose accuracy cannot be checked off a phone.

## Layers

| # | Layer | Status | Run it | Needs |
|---|-------|--------|--------|-------|
| 1 | Backend unit and API tests (`backend/tests`, pytest, FastAPI TestClient, model/TTS faked) | EXISTS | `pytest` | nothing |
| 2 | Browser tests with iPhone 14 emulation: top bar fits (portrait and landscape), SHOW ME opens and closes the demo and sends `demo_request` / `demo_end`, LIVE CUES toggle, caption, result card, "voice off" and "no audio" notes, hold-to-talk sends `ptt_start` / `ptt_end`, typed box when speech to text is off | EXISTS | `cd frontend && npm run test:e2e` | Chromium; WebKit with `PW_WEBKIT=1` |
| 3 | Smoke script against a running app or tunnel: page, `/health`, `/moves`, `/stt/status`, auth gate, session start, WebSocket `voice_status`, SHOW ME answered, session end | EXISTS | `SMOKE_TOKEN=... python backend/scripts/smoke_deployed.py https://<tunnel>` | `websockets` (in requirements.txt) |
| 4 | GitHub Actions: pytest, smoke against a local backend, then browser tests in Chromium and WebKit | EXISTS | on every PR | nothing |
| 5 | Replay a recorded landmark clip through the full backend socket and assert the result card (catches scoring regressions end to end) | PROPOSED | | needs a committed clip with a person in it |
| 6 | Pre-deploy gate script: `./dev.sh --tunnel`, then smoke, then print "ready to test on phone" | PROPOSED | | |

### How layer 2 works

The backend is real for REST (`/moves`, `/session/start`, previews). The session WebSocket is faked inside each test (`page.routeWebSocket`), so the test controls exactly what the coach "says" and sees exactly what the phone sends. That means no MediaPipe, API key, Kokoro or Whisper is needed and tests are fast and deterministic. Camera and mic are Chromium's fake devices, so `getUserMedia` succeeds.

Limits of layer 2:
- The "iphone-chromium" project is Chromium with an iPhone viewport, touch and user agent. It checks layout and logic, not Safari behavior.
- The "iphone-webkit" project (on in CI) uses the real WebKit engine. It is closer to Safari but is still not iOS Safari: no safe-area insets from a real notch, no iOS audio session, no iOS permission prompts.
- The fake camera shows a test pattern, not a person, so pose is never detected. Tests never assert on pose.

### Why not mock Anthropic and TTS more

Layer 1 already fakes the model and TTS (see `test_chat.py`, `test_tts_streaming.py`, `test_grounding.py`). No new backend mocking is needed for this work.

## Run it locally (2019 Intel Mac)

```bash
cd frontend && npm install
npx playwright install chromium webkit     # one time, about 300 MB
BACKEND_PYTHON=../backend/.venv/bin/python npm run test:e2e
PW_WEBKIT=1 BACKEND_PYTHON=../backend/.venv/bin/python npm run test:e2e   # adds WebKit
```

Playwright starts the backend on :8000 and a production build on :4173 itself. If you already run `./dev.sh`, stop it first or the ports clash (local runs reuse a server that is already up on those ports).

Before sending a link to the phone, with the tunnel up:

```bash
SMOKE_TOKEN=<your ACCESS_TOKEN> python backend/scripts/smoke_deployed.py https://<name>.trycloudflare.com
```

## What cannot be automated: manual iPhone checklist

Run after the automated layers pass. Takes about five minutes.

1. Open the tunnel link in Safari. Allow camera. The picture appears and the framing guide shows.
2. Tap TAP FOR SOUND if shown. Start a rep, stop. The coach debrief is spoken out loud (autoplay works on iOS only after a tap, and only a real phone proves it).
3. Audio comes out of the speaker, not the earpiece, also after the mic has been used.
4. Hold HOLD TO TALK, ask a question, release. The coach hears and answers. The recording indicator turns off after release.
5. Long-press the talk button: no iOS text-selection or context menu appears.
6. Pose quality: the skeleton follows you at an acceptable frame rate in your real room and lighting.
7. Portrait and landscape: top bar looks right with the real notch and home indicator.
8. SHOW ME demo is readable at arm's length.

Everything here needs a real iPhone, a real microphone or a real person on camera, so it stays in a local thread or in Jordan's hands (CLAUDE.md rule 10).

## What this does not give you

- No claim about speed, accuracy or latency. Those need `benchmark_pipeline.py` and committed outputs.
- Layer 2 passing does not mean sound works on iOS.
