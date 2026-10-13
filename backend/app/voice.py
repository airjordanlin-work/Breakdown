"""Voice coaching module — async TTS using Kokoro ONNX (local, free, no API)."""

from __future__ import annotations

import io
import random
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from app.scorer import ScoreResult, PERFECT, MISS, CLOSE

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH    = _PROJECT_ROOT / "kokoro-v1.0.onnx"
VOICES_PATH   = _PROJECT_ROOT / "voices-v1.0.bin"

VOICE_MAP = {
    "female": "af_bella",
    "male":   "am_michael",
}

PERFECT_PHRASES = [
    "Yeah, that's it! Perfect form",
    "Nailed it, great position",
    "Perfect, keep that up",
    "That's the one, hold it",
]

CLOSE_PHRASES = [
    "Almost there, just open up your {joint} a little more",
    "Getting closer, adjust your {joint} slightly",
    "Nearly perfect, tweak that {joint}",
]

EXTEND_PHRASES = [
    "Straighten your {joint} out more",
    "Open up that {joint}",
    "Get that {joint} fully extended",
]

BEND_PHRASES = [
    "Bring your {joint} in a bit",
    "Bend that {joint} more",
    "Tuck your {joint} in",
]

NO_BODY_PHRASES = [
    "Hey, step back so I can see your full body",
    "Step back a bit, I need to see your hips and shoulders",
    "Move back so your full body is in frame",
]


class VoiceCoach:
    """Async voice coach using Kokoro TTS — runs fully local, no API required."""

    def __init__(self, cooldown: float = 4.0, gender: str = "female") -> None:
        self._cooldown           = cooldown
        self._gender             = gender
        self._last_spoken: float = 0.0
        self._thread: threading.Thread | None = None
        self._lock               = threading.Lock()
        self._synth_lock         = threading.Lock()   # Kokoro is not shared across threads
        self._kokoro             = None
        self._available          = False
        # Why TTS is off, shown in the server log and the client UI. None when ready.
        self.unavailable_reason: str | None = "not loaded"
        self._recent: list[str]  = []   # tracks recently used phrases to avoid repetition

        try:
            from kokoro_onnx import Kokoro
            if not MODEL_PATH.exists():
                self.unavailable_reason = f"Kokoro model file missing: {MODEL_PATH.name}"
                print(f"VoiceCoach: model not found at {MODEL_PATH}")
                print("  Run: curl -L -o kokoro-v1.0.onnx https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx")
                return
            if not VOICES_PATH.exists():
                self.unavailable_reason = f"Kokoro voices file missing: {VOICES_PATH.name}"
                print(f"VoiceCoach: voices not found at {VOICES_PATH}")
                print("  Run: curl -L -o voices-v1.0.bin https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin")
                return

            self._kokoro    = Kokoro(str(MODEL_PATH), str(VOICES_PATH))
            self._available = True
            self.unavailable_reason = None
            print(f"VoiceCoach: ready — {gender} voice ({VOICE_MAP[gender]})")

        except ImportError:
            self.unavailable_reason = "kokoro-onnx is not installed"
            print("VoiceCoach: kokoro-onnx not installed — run: pip install kokoro-onnx soundfile")
        except Exception as e:
            self.unavailable_reason = f"Kokoro failed to load: {e}"
            print(f"VoiceCoach: failed to load — {e}")

    def _pick(self, phrases: list[str]) -> str:
        """Pick a phrase not recently used to avoid repetition."""
        available = [p for p in phrases if p not in self._recent]
        if not available:
            # all phrases used — reset and pick from full list
            available = phrases
            self._recent = []
        choice = random.choice(available)
        self._recent.append(choice)
        # keep recent list to last 3 entries
        if len(self._recent) > 3:
            self._recent.pop(0)
        return choice

    def set_gender(self, gender: str) -> None:
        """Switch between female (Bella) and male (Michael) at runtime."""
        if gender not in VOICE_MAP:
            print(f"VoiceCoach: unknown gender '{gender}' — use 'female' or 'male'")
            return
        self._gender = gender
        print(f"VoiceCoach: switched to {gender} ({VOICE_MAP[gender]})")

    @property
    def available(self) -> bool:
        return self._available

    def synthesize_wav(self, text: str) -> bytes | None:
        """Render text to 16-bit mono WAV bytes, or None if TTS is unavailable.

        This is the phone path: the server sends these bytes to the browser
        instead of playing them on the server. Blocking, so call it from a
        worker thread.
        """
        if not self._available or not text:
            return None
        try:
            import soundfile as sf
            with self._synth_lock:
                samples, sample_rate = self._kokoro.create(
                    text, voice=VOICE_MAP[self._gender], speed=1.0
                )
            buf = io.BytesIO()
            sf.write(buf, samples, sample_rate, format="WAV", subtype="PCM_16")
            return buf.getvalue()
        except Exception as e:
            print(f"VoiceCoach synth error: {e}")
            return None

    def _speak_async(self, text: str) -> None:
        """Generate and play audio locally on a daemon thread (desktop CLI only).

        Uses macOS afplay, so it never reaches a phone. The web server uses
        synthesize_wav() and streams the audio to the client instead.
        """
        def run() -> None:
            wav = self.synthesize_wav(text)
            if wav is None:
                return
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
                    f.write(wav)
                    f.flush()
                    subprocess.run(["afplay", f.name], timeout=10)
            except Exception as e:
                print(f"VoiceCoach error: {e}")

        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(target=run, daemon=True)
            self._thread.start()

    def _can_speak(self) -> bool:
        if not self._available:
            return False
        now = time.time()
        if now - self._last_spoken < self._cooldown:
            return False
        self._last_spoken = now
        return True

    def speak(self, text: str) -> None:
        """Speak arbitrary text if cooldown has elapsed."""
        if not self._can_speak():
            return
        print(f"VoiceCoach: '{text}'")
        self._speak_async(text)

    def feedback_from_score(self, score_result: ScoreResult) -> None:
        """Speak joint-specific feedback based on the latest ScoreResult."""
        if not self._can_speak() or score_result is None:
            return

        misses   = [r for r in score_result.results if r.tier == MISS]
        closes   = [r for r in score_result.results if r.tier == CLOSE]
        perfects = [r for r in score_result.results if r.tier == PERFECT]

        if not misses and not closes and perfects:
            self._speak_async(self._pick(PERFECT_PHRASES))
        elif not misses and closes:
            joint = closes[0].joint_name.replace("_", " ")
            self._speak_async(self._pick(CLOSE_PHRASES).format(joint=joint))
        elif misses:
            joint  = misses[0].joint_name.replace("_", " ")
            actual = misses[0].actual_angle
            target = misses[0].target_angle
            if actual < target:
                self._speak_async(self._pick(EXTEND_PHRASES).format(joint=joint))
            else:
                self._speak_async(self._pick(BEND_PHRASES).format(joint=joint))

    def feedback_no_body(self) -> None:
        if not self._can_speak():
            return
        self._speak_async(self._pick(NO_BODY_PHRASES))

    def feedback_aligned(self) -> None:
        if not self._can_speak():
            return
        self._speak_async("Nice, hold that")

    def feedback_buffer_ready(self) -> None:
        if not self._can_speak():
            return
        self._speak_async("Okay, go ahead")