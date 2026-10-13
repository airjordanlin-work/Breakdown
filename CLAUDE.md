# CLAUDE.md

Breakdown is a real-time, phone-first, voice-conversational breakdance coach. Read `docs/PRODUCT_SPEC.md` before starting any non-trivial task. It defines the goal, requirements, data model and milestones. Work on milestones in order (M1 first) unless told otherwise.

## Stack

- Backend: Python 3.12, FastAPI + WebSocket (`backend/api/server.py`), core logic in `backend/app/` (pose_estimator, buffer, dtw_engine, scorer, coach, voice, move_catalog, imu_protocol).
- Frontend: React + Vite in `frontend/`.
- Coach LLM: Claude Haiku (`claude-haiku-4-5-20251001`).
- TTS: Kokoro ONNX. Currently plays on the server via `afplay`, which does not reach a phone. Streaming audio to the client is the first fix (M1).
- Hardware (optional, later): ESP32-S3 + 2 MPU-6050 over BLE, 32-byte packet defined in `imu_protocol.py`.

## Commands

```bash
# run everything for dev
./dev.sh

# backend only
cd backend && uvicorn api.server:app --reload --port 8000

# frontend only
cd frontend && npm install && npm run dev

# tests
pytest

# benchmarks (write CSVs to backend/recordings/)
python scripts/benchmark_pipeline.py dtw
python scripts/benchmark_pipeline.py coach --n 5
python scripts/benchmark_pipeline.py pose --video <clip> --with-dtw
```

Use the project venv (`source .venv/bin/activate`) or `python3`.

## Rules

1. **Grounded facts only.** The coach may state body or score facts only from the `facts` block or tool results. Do not let prompts invite the model to invent measurements.
2. **Never invent metrics.** Do not write a number into the README, docs or resume text unless a script in `scripts/` produced it and the output is committed under `docs/benchmarks/`.
3. **Do not claim unbuilt features** in the README or docs. Mark status EXISTS or PROPOSED.
4. **Small PRs.** One concern per branch. Every code change ends with the branch name and a conventional commit message (`feat:`, `fix:`, `test:`, `docs:`, `refactor:`), for example `feat/stream-tts-to-client` and `feat: stream TTS audio chunks to the client over WebSocket`.
5. **Tests with changes.** Add or update Pytest tests for backend changes. Do not merge with failing tests. CI must stay green.
6. **No secrets.** Never commit `.env`, API keys or recordings. `backend/recordings/` stays git-ignored. Use `.env.example` for new settings.
7. **Privacy default.** Do not persist raw video or audio. Landmarks and transcripts only, and only when the spec says so.
8. **Degrade, don't crash.** Every external call (LLM, STT, TTS) needs a timeout and a templated fallback.
9. **No em-dashes** in docs and README text. Keep prose plain and direct.
10. **Stop and say so** when a task needs something you cannot do in the cloud: a Mac with a webcam, an iPhone, BLE hardware, `afplay`, or a real microphone. Prepare the code and tests, list the exact manual steps for Jordan, and stop. Do not fake results.

## Definition of done

- Code runs, `pytest` passes locally, CI green.
- Behavior matches the requirement ID in the spec (mention it in the PR, for example "implements FR-10").
- Docs updated if behavior or protocol changed (spec section 8 for protocol).
- Any new measurable claim has a benchmark script and committed output.
- PR description says what was verified and what still needs a manual device test.

## Local-only work (needs Jordan's machine)

Real pose fps, phone camera over HTTPS tunnel (ngrok or cloudflared), audio playback on iPhone Safari, microphone and push-to-talk, BLE/IMU. Run these in a local thread, not a cloud thread.