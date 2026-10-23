"""The voice benchmark harness (scripts/benchmark_pipeline.py stt and chat), run on stand-ins.

These check the plumbing only. Real numbers come from running the script on
the demo machine; nothing measured here is a result.
"""

import csv
import importlib.util
import io
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "benchmark_pipeline.py"
spec = importlib.util.spec_from_file_location("benchmark_pipeline", SCRIPT)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def wav_bytes(seconds=1.0, rate=48000, channels=1):
    t = np.arange(int(seconds * rate)) / rate
    tone = (np.sin(2 * np.pi * 220 * t) * 8000).astype("<i2")
    if channels == 2:
        tone = np.repeat(tone, 2)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(tone.tobytes())
    return buf.getvalue()


def test_word_error_rate():
    assert bench.word_error_rate("how did I do", "How did I do?") == 0.0
    assert bench.word_error_rate("how did I do", "how I do") == 0.25
    assert bench.word_error_rate("what should I fix", "what should I fix it") == 0.25
    assert bench.word_error_rate("", "") == 0.0


@pytest.mark.parametrize("rate,channels", [(48000, 1), (44100, 2), (16000, 1)])
def test_wav_is_converted_to_what_the_phone_sends(rate, channels):
    pcm = bench.wav_to_pcm16k(wav_bytes(1.0, rate, channels))
    assert abs(len(pcm) - 32000) <= 2


def test_stt_stage_runs_on_a_stand_in_engine(tmp_path, monkeypatch):
    (tmp_path / "a.wav").write_bytes(wav_bytes(1.0))
    (tmp_path / "b.wav").write_bytes(wav_bytes(2.0, 44100))
    clips = tmp_path / "clips.csv"
    clips.write_text("path,text\na.wav,how did I do\nb.wav,what should I fix\n")

    class Engine:
        def __init__(self, model):
            self.available, self.unavailable_reason = True, None

        def warm_up(self):
            pass

        def transcribe(self, pcm):
            return SimpleNamespace(text="how did I do", ms=12.0, confidence=0.8)

    import app.stt
    monkeypatch.setattr(app.stt, "WhisperSTT", Engine)
    args = SimpleNamespace(clips=str(clips), synth=False, models=["tiny.en"], repeat=1, out_dir=str(tmp_path))
    bench.bench_stt(args)
    out = next(tmp_path.glob("bench_stt_*.csv"))
    rows = list(csv.DictReader(out.open()))
    assert [r["wer"] for r in rows] == ["0.000", "0.750"]   # only "I" matches
    assert rows[1]["clip_s"] == "2.00"


def test_chat_stage_runs_on_a_stand_in_model(tmp_path, monkeypatch):
    import app.chat as chat_mod

    async def fake(system, messages, max_tokens, timeout):
        return "Keep working on that left elbow next round.", 400, 15

    monkeypatch.setattr(chat_mod, "ANTHROPIC_API_KEY", "test")
    monkeypatch.setattr(chat_mod, "anthropic_chat", fake)
    bench.bench_chat(SimpleNamespace(n=1, out_dir=str(tmp_path)))
    out = next(tmp_path.glob("bench_chat_*.csv"))
    rows = list(csv.DictReader(out.open()))
    sources = {r["question"]: r["source"] for r in rows}
    assert sources["hello"] == "quick" and sources["what should I fix"] == "quick"
    assert sources["why does a straight elbow matter for this move"] == "model"
    assert len(rows) == len(bench.CHAT_QUESTIONS)
