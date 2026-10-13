"""Coach audio is streamed to the client instead of played on the server (FR-10)."""

import asyncio
import io
import json
import threading
import time

import numpy as np
import pytest
import soundfile as sf

from app.audio_frames import (
    AUDIO_CHUNK_TYPE,
    AudioFrameError,
    decode_audio_chunk,
    encode_audio_chunk,
)
from app.voice import VoiceCoach
from api import server


class FakeKokoro:
    """Stands in for kokoro_onnx.Kokoro: half a second of a 440 Hz tone."""

    def __init__(self, delay: float = 0.0):
        self.delay = delay
        self.calls = []

    def create(self, text, voice, speed):
        self.calls.append((text, voice, speed))
        time.sleep(self.delay)
        sr = 24000
        t = np.arange(sr // 2) / sr
        return (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), sr


def make_voice(kokoro=None) -> VoiceCoach:
    # Skip __init__ so the test never loads the real Kokoro model files.
    voice = VoiceCoach.__new__(VoiceCoach)
    voice._gender = "female"
    voice._kokoro = kokoro
    voice._available = kokoro is not None
    voice._synth_lock = threading.Lock()
    return voice


class FakeSocket:
    def __init__(self):
        self.sent = []   # (kind, data) in send order

    async def send_text(self, data):
        self.sent.append(("text", data))

    async def send_bytes(self, data):
        self.sent.append(("bytes", data))


# ── frame format ─────────────────────────────────────────────

def test_audio_chunk_round_trip():
    frame = encode_audio_chunk("abc123", b"RIFFdata", seq=2, codec="wav", final=False)
    header, payload = decode_audio_chunk(frame)
    assert header == {
        "type": AUDIO_CHUNK_TYPE, "turn_id": "abc123", "seq": 2, "codec": "wav", "final": False,
    }
    assert payload == b"RIFFdata"


def test_audio_chunk_header_length_is_big_endian_prefix():
    frame = encode_audio_chunk("t", b"")
    n = int.from_bytes(frame[:2], "big")
    assert json.loads(frame[2:2 + n])["turn_id"] == "t"
    assert len(frame) == 2 + n


def _frame_with_header(raw: bytes) -> bytes:
    return len(raw).to_bytes(2, "big") + raw + b"audio"


@pytest.mark.parametrize("frame", [
    b"", b"\x00", b"\x00\x10{}", b"\x00\x02\xff\xfe",
    _frame_with_header(b"null"),
    _frame_with_header(b"[1,2]"),
    _frame_with_header(b'{"type":"coach_audio_chunk"}'),
    _frame_with_header(b'{"type":"other","turn_id":"t","seq":0,"codec":"wav","final":true}'),
    _frame_with_header(b'{"type":"coach_audio_chunk","turn_id":"t","seq":"0","codec":"wav","final":true}'),
])
def test_decode_rejects_malformed_frames(frame):
    with pytest.raises(AudioFrameError):
        decode_audio_chunk(frame)


# ── synthesis ────────────────────────────────────────────────

def test_synthesize_wav_returns_playable_wav():
    kokoro = FakeKokoro()
    wav = make_voice(kokoro).synthesize_wav("Lock that left arm out")
    assert wav[:4] == b"RIFF"
    samples, sr = sf.read(io.BytesIO(wav))
    assert sr == 24000
    assert len(samples) == 12000
    assert kokoro.calls == [("Lock that left arm out", "af_bella", 1.0)]


def test_synthesize_wav_without_model_returns_none():
    assert make_voice(None).synthesize_wav("hello") is None


def test_synthesize_wav_survives_kokoro_error():
    class Broken:
        def create(self, *a, **k):
            raise RuntimeError("onnx blew up")
    assert make_voice(Broken()).synthesize_wav("hello") is None


def test_synthesize_wav_never_calls_afplay(monkeypatch):
    import subprocess
    def boom(*a, **k):
        raise AssertionError("server path must not play audio locally")
    monkeypatch.setattr(subprocess, "run", boom)
    assert make_voice(FakeKokoro()).synthesize_wav("hello") is not None


# ── server delivery ──────────────────────────────────────────

def test_send_coach_cue_sends_caption_then_audio_with_same_turn_id():
    ws = FakeSocket()
    sent_audio = asyncio.run(server.send_coach_cue(ws, make_voice(FakeKokoro()), "Stay low"))
    assert sent_audio is True
    assert [kind for kind, _ in ws.sent] == ["text", "bytes"]
    caption = json.loads(ws.sent[0][1])
    header, payload = decode_audio_chunk(ws.sent[1][1])
    assert caption["type"] == "coach_cue" and caption["cue"] == "Stay low"
    assert header["turn_id"] == caption["turn_id"]
    assert header["codec"] == "wav" and header["final"] is True
    assert payload[:4] == b"RIFF"


def _caption_then_skip(ws) -> tuple[dict, dict]:
    """Assert the cue went out as caption + coach_audio_skipped, and return both."""
    assert [kind for kind, _ in ws.sent] == ["text", "text"]
    caption, skipped = (json.loads(data) for _, data in ws.sent)
    assert caption["type"] == "coach_cue" and caption["cue"] == "Stay low"
    assert skipped["type"] == "coach_audio_skipped"
    assert skipped["turn_id"] == caption["turn_id"]
    return caption, skipped


def test_send_coach_cue_without_tts_still_sends_caption():
    ws = FakeSocket()
    voice = make_voice(None)
    voice.unavailable_reason = "Kokoro model file missing: kokoro-v1.0.onnx"
    assert asyncio.run(server.send_coach_cue(ws, voice, "Stay low")) is False
    _, skipped = _caption_then_skip(ws)
    assert skipped["reason"] == "Kokoro model file missing: kokoro-v1.0.onnx"


def test_send_coach_cue_with_no_voice_object_still_sends_caption():
    ws = FakeSocket()
    assert asyncio.run(server.send_coach_cue(ws, None, "Stay low")) is False
    _, skipped = _caption_then_skip(ws)
    assert skipped["reason"] == "no voice for this session"


def test_send_coach_cue_reports_synth_failure():
    class Broken:
        def create(self, *a, **k):
            raise RuntimeError("onnx blew up")
    ws = FakeSocket()
    assert asyncio.run(server.send_coach_cue(ws, make_voice(Broken()), "Stay low")) is False
    _, skipped = _caption_then_skip(ws)
    assert "failed" in skipped["reason"]


def test_voice_status_reports_why_audio_is_off():
    voice = make_voice(None)
    voice.unavailable_reason = "kokoro-onnx is not installed"
    assert server.voice_status(voice) == {
        "type": "voice_status", "available": False, "reason": "kokoro-onnx is not installed",
    }
    assert server.voice_status(make_voice(FakeKokoro()))["available"] is True


def test_warm_up_voice_logs_timing(capsys):
    kokoro = FakeKokoro()
    server.warm_up_voice(make_voice(kokoro))
    assert len(kokoro.calls) == 1
    assert "warm-up synth took" in capsys.readouterr().out


def test_warm_up_voice_logs_loudly_when_audio_is_off(capsys):
    voice = make_voice(None)
    voice.unavailable_reason = "Kokoro voices file missing: voices-v1.0.bin"
    server.warm_up_voice(voice)
    out = capsys.readouterr().out
    assert "COACH AUDIO OFF" in out and "voices-v1.0.bin" in out


def test_send_coach_cue_times_out_to_text_only():
    ws = FakeSocket()
    slow = make_voice(FakeKokoro(delay=0.5))
    start = time.perf_counter()
    assert asyncio.run(server.send_coach_cue(ws, slow, "Stay low", timeout=0.05)) is False
    assert time.perf_counter() - start < 0.4
    _, skipped = _caption_then_skip(ws)
    assert "timed out" in skipped["reason"]
