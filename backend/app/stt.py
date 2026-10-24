"""Speech to text for push-to-talk (FR-11).

The phone sends 16 kHz mono 16-bit PCM while the dancer holds the button.
Here that clip becomes text. The audio is never written anywhere and the
caller drops it as soon as this returns (CLAUDE.md rule 7).

Engines:
- ``WhisperSTT``: faster-whisper running on this machine (default model
  ``base.en``, int8 on the CPU). Free per turn, nothing leaves the machine.
  Optional: install with ``pip install -r backend/requirements-voice.txt``.
  The first use downloads the model, so the first turn is slow.
- ``FakeSTT``: for tests.
- No engine: when faster-whisper is not installed or the model fails to load,
  ``available`` is False and push-to-talk says so instead of crashing.

Choose with ``STT_ENGINE`` (whisper | off) and ``WHISPER_MODEL`` in .env.
How fast this is on a given machine is not known until
``scripts/benchmark_pipeline.py voice`` is run there.
"""

from __future__ import annotations

import math
import os
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

SAMPLE_RATE  = 16_000
BYTES_PER_S  = SAMPLE_RATE * 2          # 16-bit mono
MIN_CLIP_S   = 0.4
MAX_CLIP_S   = 8.0
MAX_CLIP_BYTES = int(MAX_CLIP_S * BYTES_PER_S)

STT_ENGINE    = os.getenv("STT_ENGINE", "whisper").strip().lower()
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base.en").strip()
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "int8").strip()
try:
    STT_TIMEOUT_S = float(os.getenv("STT_TIMEOUT_S", "") or 10.0)
except ValueError:
    STT_TIMEOUT_S = 10.0

# Biases Whisper toward words a dancer is likely to say. Not a fact source.
VOCAB_PROMPT = ("Breaking practice with a coach. Toprock, cross step, footwork, freeze, "
                "elbow, knee, hips, shoulder, ankle, wrist, rep, streak.")
NO_SPEECH_LIMIT = 0.6   # Whisper's own no-speech probability above this means silence


@dataclass
class Transcript:
    text: str
    confidence: Optional[float] = None   # 0 to 1, rough; None when the engine gives none
    ms: float = 0.0


def clip_seconds(pcm: bytes) -> float:
    return len(pcm) / BYTES_PER_S


def pcm16_to_float(pcm: bytes) -> np.ndarray:
    """16-bit little-endian PCM to float32 in [-1, 1], what Whisper takes."""
    if len(pcm) % 2:
        pcm = pcm[:-1]
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


class FakeSTT:
    """Returns a fixed transcript. For tests."""

    def __init__(self, text: str = "how did I do", confidence: float = 0.9, fail: bool = False) -> None:
        self.text, self.confidence, self.fail = text, confidence, fail
        self.available = True
        self.unavailable_reason: Optional[str] = None
        self.clips: list[int] = []

    def transcribe(self, pcm: bytes) -> Transcript:
        self.clips.append(len(pcm))
        if self.fail:
            raise RuntimeError("fake STT failure")
        return Transcript(self.text, self.confidence)


class NoSTT:
    available = False

    def __init__(self, reason: str) -> None:
        self.unavailable_reason = reason

    def transcribe(self, pcm: bytes) -> Transcript:
        raise RuntimeError(self.unavailable_reason)


class WhisperSTT:
    """faster-whisper on the CPU. The model loads on first use and is shared."""

    def __init__(self, model: str = WHISPER_MODEL, compute_type: str = WHISPER_COMPUTE) -> None:
        self.model_name, self.compute_type = model, compute_type
        self._model = None
        self._lock = threading.Lock()
        self.available = True
        self.unavailable_reason: Optional[str] = None
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            self.available = False
            self.unavailable_reason = "faster-whisper is not installed (pip install -r backend/requirements-voice.txt)"

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            start = time.perf_counter()
            self._model = WhisperModel(self.model_name, device="cpu", compute_type=self.compute_type)
            print(f"STT: loaded whisper {self.model_name} ({self.compute_type}) in {time.perf_counter() - start:.1f} s")
        return self._model

    def warm_up(self) -> None:
        """Load the model now (first use downloads it), so the first turn is not the slow one."""
        if not self.available:
            return
        try:
            with self._lock:
                self._load()
        except Exception as e:
            self.available = False
            self.unavailable_reason = f"whisper model failed to load: {e}"
            print(f"STT: {self.unavailable_reason}")

    def transcribe(self, pcm: bytes) -> Transcript:
        if not self.available:
            raise RuntimeError(self.unavailable_reason)
        start = time.perf_counter()
        with self._lock:   # one clip at a time; the model is not shared across threads
            model = self._load()
            segments, _info = model.transcribe(
                pcm16_to_float(pcm), language="en", beam_size=1,
                condition_on_previous_text=False, initial_prompt=VOCAB_PROMPT,
            )
            segments = list(segments)
        spoken = [s for s in segments if s.no_speech_prob < NO_SPEECH_LIMIT]
        text = " ".join(s.text.strip() for s in spoken).strip()
        conf = None
        if spoken:
            conf = round(math.exp(sum(s.avg_logprob for s in spoken) / len(spoken)), 3)
        return Transcript(text, conf, (time.perf_counter() - start) * 1000)


_engine = None
_engine_lock = threading.Lock()


def get_stt():
    """The process-wide engine, built once from STT_ENGINE."""
    global _engine
    with _engine_lock:
        if _engine is None:
            if STT_ENGINE in ("off", "none", ""):
                _engine = NoSTT("speech to text is off (STT_ENGINE=off)")
            else:
                _engine = WhisperSTT()
                if not _engine.available:
                    print(f"STT: push-to-talk off ({_engine.unavailable_reason})")
        return _engine
