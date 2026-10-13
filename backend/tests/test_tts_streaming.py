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


@pytest.mark.parametrize("frame", [b"", b"\x00", b"\x00\x10{}", b"\x00\x02\xff\xfe"])
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


def test_send_coach_cue_without_tts_still_sends_caption():
    ws = FakeSocket()
    assert asyncio.run(server.send_coach_cue(ws, make_voice(None), "Stay low")) is False
    assert [kind for kind, _ in ws.sent] == ["text"]


def test_send_coach_cue_with_no_voice_object_still_sends_caption():
    ws = FakeSocket()
    assert asyncio.run(server.send_coach_cue(ws, None, "Stay low")) is False
    assert [kind for kind, _ in ws.sent] == ["text"]


def test_send_coach_cue_times_out_to_text_only():
    ws = FakeSocket()
    slow = make_voice(FakeKokoro(delay=0.5))
    start = time.perf_counter()
    assert asyncio.run(server.send_coach_cue(ws, slow, "Stay low", timeout=0.05)) is False
    assert time.perf_counter() - start < 0.4
    assert [kind for kind, _ in ws.sent] == ["text"]
