"""A failed Kokoro load names the installed kokoro-onnx version and the fix."""

import sys
import types

from app import voice
from app.voice import VoiceCoach, describe_load_failure


def _exploding_kokoro_module(exc):
    mod = types.ModuleType("kokoro_onnx")

    class Kokoro:
        def __init__(self, *args, **kwargs):
            raise exc

    mod.Kokoro = Kokoro
    return mod


def _load_with(monkeypatch, tmp_path, version, exc):
    model, voices = tmp_path / "m.onnx", tmp_path / "v.bin"
    model.write_bytes(b"x")
    voices.write_bytes(b"x")
    monkeypatch.setattr(voice, "MODEL_PATH", model)
    monkeypatch.setattr(voice, "VOICES_PATH", voices)
    monkeypatch.setattr(voice, "_kokoro_version", lambda: version)
    monkeypatch.setitem(sys.modules, "kokoro_onnx", _exploding_kokoro_module(exc))
    return VoiceCoach()


def test_old_kokoro_onnx_gets_upgrade_hint(monkeypatch, tmp_path, capsys):
    exc = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")
    coach = _load_with(monkeypatch, tmp_path, "0.3.3", exc)
    out = capsys.readouterr().out
    assert not coach.available
    assert "0.3.3" in coach.unavailable_reason
    assert "too old" in out and "requirements-tts.txt" in out


def test_current_kokoro_onnx_decode_error_points_at_voices_file(monkeypatch, tmp_path, capsys):
    exc = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")
    _load_with(monkeypatch, tmp_path, "0.4.7", exc)
    out = capsys.readouterr().out
    assert "0.4.7" in out and "Zip archive" in out


def test_other_errors_still_report_version(monkeypatch):
    monkeypatch.setattr(voice, "_kokoro_version", lambda: None)
    assert "unknown" in describe_load_failure(RuntimeError("boom"))
