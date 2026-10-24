"""Push-to-talk: clip handling, speech to text, and the turn that follows (FR-11)."""

import asyncio
import base64
import json
import threading
from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient

from api import server
from api.server import CoachingSession, app
from app.chat import CoachChat
from app.coach import AICoach
from app.orchestrator import LINES, MAX_CHUNK_BYTES, VoiceOrchestrator
from app.stt import (BYTES_PER_S, MAX_CLIP_BYTES, FakeSTT, NoSTT, Transcript, WhisperSTT,
                     clip_seconds, pcm16_to_float)
from app.turn_gate import TokenBudget, TurnGate


def pcm(seconds: float) -> bytes:
    return b"\x01\x00" * int(seconds * BYTES_PER_S // 2)


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def chunks(data: bytes, size: int = 8000):
    for i in range(0, len(data), size):
        yield b64(data[i:i + size])


def make(stt=None, available=True):
    sent, spoken = [], []
    stt = stt or FakeSTT()

    async def emit(msg):
        sent.append(msg)

    async def speak(text):
        spoken.append(text)

    async def transcribe(clip):
        return stt.transcribe(clip)

    chat = CoachChat(llm=None, gate=TurnGate(per_minute=100), budget=TokenBudget(daily=10_000))
    chat._available = False
    facts = lambda: {"move": "Cross step", "session": {"reps": 3, "clean_reps": 2, "clean_streak": 0, "best_streak": 2},
                     "focus": {"joint": "left elbow", "fix": "straighten"}}
    orch = VoiceOrchestrator(emit, speak, facts, chat=chat, transcribe=transcribe, stt_available=available)
    return orch, sent, spoken, stt


def run(coro):
    return asyncio.run(coro)


# ---- audio helpers ---------------------------------------------------------

def test_pcm_conversion_and_length():
    data = np.array([0, 16384, -32768], dtype="<i2").tobytes()
    out = pcm16_to_float(data)
    assert out.dtype == np.float32 and np.allclose(out, [0.0, 0.5, -1.0])
    assert pcm16_to_float(data + b"\x01").shape == (3,)   # odd byte dropped
    assert clip_seconds(pcm(2.0)) == 2.0


# ---- a full turn -------------------------------------------------------------

def test_hold_talk_release_answers_what_was_heard():
    orch, sent, spoken, stt = make(FakeSTT("what should I fix"))

    async def go():
        assert await orch.ptt_start()
        for c in chunks(pcm(1.5)):
            orch.ptt_audio(c)
        return await orch.ptt_end()
    reply = run(go())
    assert reply.source == "quick" and "left elbow" in spoken[-1]
    assert stt.clips == [len(pcm(1.5))]
    assert {"type": "transcript", "text": "what should I fix", "source": "voice"} in sent
    states = [m["state"] for m in sent if m["type"] == "voice_state"]
    assert states == ["listening", "transcribing", "thinking", "speaking", "idle"]
    assert not orch.busy and not orch.listening


def test_audio_is_dropped_after_the_turn():
    orch, *_ = make()

    async def go():
        await orch.ptt_start()
        orch.ptt_audio(b64(pcm(1.0)[:MAX_CHUNK_BYTES]))
        await orch.ptt_end()
    run(go())
    assert orch._clip is None


def test_too_short_clip_never_reaches_speech_to_text():
    orch, sent, spoken, stt = make()

    async def go():
        await orch.ptt_start()
        orch.ptt_audio(b64(pcm(0.2)))
        return await orch.ptt_end()
    assert run(go()) is None
    assert stt.clips == [] and spoken == [LINES["too_short"]]
    assert not orch.busy


def test_clip_is_cut_at_the_maximum_length():
    orch, sent, spoken, stt = make()

    async def go():
        await orch.ptt_start()
        for c in chunks(pcm(12.0), 32000):
            orch.ptt_audio(c)
        await orch.ptt_end()
    run(go())
    assert stt.clips == [MAX_CLIP_BYTES]


def test_oversized_or_bad_chunks_are_ignored():
    orch, sent, spoken, stt = make()

    async def go():
        await orch.ptt_start()
        orch.ptt_audio(b64(b"\x00" * (MAX_CHUNK_BYTES + 2)))
        orch.ptt_audio("not base64 !!")
        orch.ptt_audio(None)
        orch.ptt_audio(b64(pcm(1.0)))
        await orch.ptt_end()
    run(go())
    assert stt.clips == [len(pcm(1.0))]


def test_silence_or_low_confidence_asks_again_without_the_model():
    for heard in (FakeSTT(""), FakeSTT("mumble", confidence=0.1)):
        orch, sent, spoken, _ = make(heard)

        async def go():
            await orch.ptt_start()
            orch.ptt_audio(b64(pcm(1.0)))
            return await orch.ptt_end()
        assert run(go()) is None
        assert spoken == [LINES["not_heard"]]
        assert not any(m["type"] == "transcript" for m in sent)


def test_speech_to_text_failure_degrades_to_a_spoken_line():
    orch, sent, spoken, _ = make(FakeSTT(fail=True))

    async def go():
        await orch.ptt_start()
        orch.ptt_audio(b64(pcm(1.0)))
        return await orch.ptt_end()
    assert run(go()) is None
    assert spoken == [LINES["stt_failed"]] and not orch.busy


def test_without_speech_to_text_the_button_says_to_type():
    orch, sent, spoken, _ = make(available=False)

    async def go():
        started = await orch.ptt_start()
        await asyncio.sleep(0.01)   # the line is spoken off the receive loop
        return started
    assert run(go()) is False
    assert spoken == [LINES["stt_off"]] and not orch.busy
    assert {"type": "stt_status", "available": False, "state": "off"} in sent


def test_a_press_while_the_model_loads_says_so_without_waiting_for_it():
    stt = FakeSTT()
    stt.loading = True
    orch, sent, spoken, _ = make(stt)
    orch._stt_status = stt.status

    async def go():
        started = await orch.ptt_start()
        heard_before_speaking = list(spoken)
        await asyncio.sleep(0.01)
        return started, heard_before_speaking
    started, before = run(go())
    assert started is False and before == []
    assert spoken == [LINES["stt_loading"]] and stt.clips == [] and not orch.busy
    assert {"type": "stt_status", "available": True, "state": "loading"} in sent

    stt.loading = False   # status is read on every press, so it is never stale
    assert run(orch.ptt_start()) is True


def test_closing_drops_the_clip_and_skips_the_model():
    orch, sent, spoken, stt = make(FakeSTT("talk to me about practice"))

    async def go():
        await orch.ptt_start()
        orch.ptt_audio(b64(pcm(1.0)))
        await orch.close()
        late = await orch.ptt_end()
        typed = await orch.handle_text("talk to me about practice")
        return late, typed
    assert run(go()) == (None, None)
    assert stt.clips == [] and spoken == [] and not orch.busy and not orch.listening
    assert orch._listen_timer is None


def test_cancel_drops_the_clip_and_says_nothing():
    orch, sent, spoken, stt = make()

    async def go():
        await orch.ptt_start()
        orch.ptt_audio(b64(pcm(1.0)))
        await orch.ptt_cancel()
    run(go())
    assert spoken == [] and stt.clips == [] and not orch.busy and not orch.listening


def test_a_clip_with_no_release_is_ended_by_the_timer(monkeypatch):
    import app.orchestrator as orch_mod
    monkeypatch.setattr(orch_mod, "MAX_CLIP_S", 0.01)
    monkeypatch.setattr(orch_mod, "LISTEN_GRACE_S", 0.01)
    orch, sent, spoken, stt = make(FakeSTT("how did I do"))

    async def go():
        await orch.ptt_start()
        orch.ptt_audio(b64(pcm(1.0)))
        await asyncio.sleep(0.1)
    run(go())
    assert stt.clips and spoken and not orch.busy


def test_typing_while_listening_is_refused():
    orch, sent, spoken, _ = make()

    async def go():
        await orch.ptt_start()
        second = await orch.handle_text("hello")
        await orch.ptt_cancel()
        return second
    assert run(go()) is None
    assert any(m.get("busy") for m in sent)


# ---- the whisper engine, with a stand-in model --------------------------------

class FakeModel:
    def __init__(self, segments):
        self.segments = segments
        self.calls = []

    def transcribe(self, audio, **kw):
        self.calls.append((audio, kw))
        return iter(self.segments), None


def seg(text, no_speech=0.1, logprob=-0.3):
    return SimpleNamespace(text=text, no_speech_prob=no_speech, avg_logprob=logprob)


def test_whisper_joins_speech_and_drops_silent_segments():
    engine = WhisperSTT()
    engine.available = True
    engine._model = FakeModel([seg(" how did"), seg(" I do"), seg(" [music]", no_speech=0.9)])
    out = engine.transcribe(pcm(1.0))
    assert out.text == "how did I do"
    assert 0 < out.confidence < 1
    audio, kw = engine._model.calls[0]
    assert audio.dtype == np.float32 and kw["language"] == "en"


def test_whisper_all_silence_gives_empty_text():
    engine = WhisperSTT()
    engine.available = True
    engine._model = FakeModel([seg(" thanks for watching", no_speech=0.95)])
    out = engine.transcribe(pcm(1.0))
    assert out.text == "" and out.confidence is None


def test_whisper_reports_loading_and_refuses_clips_meanwhile():
    engine = WhisperSTT()
    engine.available = True
    engine._loading = True
    assert engine.status() == "loading"
    try:
        engine.transcribe(pcm(1.0))
        assert False, "should refuse while loading"
    except RuntimeError as e:
        assert "loading" in str(e)
    engine._loading = False
    engine.available = False
    assert engine.status() == "off"


def test_a_clip_queued_behind_a_stuck_one_fails_fast(monkeypatch):
    import time as _time

    class Stuck(FakeSTT):
        def transcribe(self, pcm):
            _time.sleep(0.3)
            return super().transcribe(pcm)

    monkeypatch.setattr(server, "get_stt", lambda: Stuck("how did I do"))
    monkeypatch.setattr(server, "STT_TIMEOUT_S", 0.05)
    monkeypatch.setattr(server, "_stt_inflight", None)
    session = make_session()

    async def go():
        orch = server.make_orchestrator(session, websocket=None, viewers=None)
        first = asyncio.get_running_loop().time()
        for _ in range(2):
            await orch.ptt_start()
            orch.ptt_audio(b64(pcm(1.0)))
            await orch.ptt_end()
        return asyncio.get_running_loop().time() - first, orch
    took, orch = run(go())
    assert took < 0.25   # the second clip did not wait for the first to finish
    assert not orch.busy


def test_whisper_unavailable_says_why():
    engine = NoSTT("speech to text is off")
    assert not engine.available
    try:
        engine.transcribe(b"")
    except RuntimeError as e:
        assert "off" in str(e)


# ---- over the socket -------------------------------------------------------------

def make_session():
    s = CoachingSession.__new__(CoachingSession)
    s._lock = threading.Lock()
    s.closed = False
    s.voice = None
    s.ai_coach = AICoach(llm=lambda *a: None)
    s.speak_lock = asyncio.Lock()
    s.coach_tasks = set()
    s.last_stats = {"reps": 2, "clean_reps": 1, "focus_joint": "right knee", "focus_fix": "bend"}
    s.last_summary, s.last_move_name = None, "Cross step"
    s.chat = CoachChat(gate=TurnGate(), budget=TokenBudget(daily=10_000))
    s.chat._available = False
    return s


def test_socket_push_to_talk_round_trip(monkeypatch):
    stt = FakeSTT("what should I fix")
    monkeypatch.setattr(server, "get_stt", lambda: stt)
    server._sessions["sess-ptt"] = make_session()
    try:
        with TestClient(app).websocket_connect("/ws/sess-ptt") as ws:
            ws.receive_text()   # voice status
            ws.send_text(json.dumps({"type": "ptt_start"}))
            assert json.loads(ws.receive_text()) == {"type": "voice_state", "state": "listening"}
            for c in chunks(pcm(1.0)):
                ws.send_text(json.dumps({"type": "ptt_audio", "data": c}))
            ws.send_text(json.dumps({"type": "ptt_end"}))
            msgs = []
            while not msgs or msgs[-1] != {"type": "voice_state", "state": "idle"}:
                msgs.append(json.loads(ws.receive_text()))
        cue = next(m for m in msgs if m["type"] == "coach_cue")
        assert cue["mode"] == "reply" and "right knee" in cue["cue"]
        assert {"type": "transcript", "text": "what should I fix", "source": "voice"} in msgs
        assert stt.clips == [len(pcm(1.0))]
    finally:
        server._sessions.pop("sess-ptt", None)


def test_stt_status_endpoint(monkeypatch):
    monkeypatch.setattr(server, "get_stt", lambda: NoSTT("speech to text is off (STT_ENGINE=off)"))
    body = TestClient(app).get("/stt/status").json()
    assert body == {"available": False, "state": "off", "reason": "speech to text is off (STT_ENGINE=off)"}
