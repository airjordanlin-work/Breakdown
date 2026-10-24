"""Turn-taking between the dancer and the coach (spec 3.1, M3).

One user turn at a time. A turn is a typed question (``chat``) or a
push-to-talk clip (``ptt_start``, ``ptt_audio`` chunks, ``ptt_end``). While a
turn is in progress:

- live cues stay quiet (the dancer is talking or listening);
- a second turn is refused with a short ``voice_state`` message instead of
  queueing up more model calls;
- the phone gets ``voice_state`` updates (listening, transcribing, thinking,
  speaking, idle) so it can show what the coach is doing.

Push-to-talk audio is kept in memory only until speech to text returns, then
dropped (CLAUDE.md rule 7). A clip is cut at MAX_CLIP_S, and one that never
gets a ``ptt_end`` is ended by a timer so the coach cannot get stuck listening.

The orchestrator does not know about WebSockets. The server passes in
``emit`` (send a JSON message to the phone and viewers), ``speak`` (caption
plus TTS audio) and ``facts`` (the measured values a reply may use), which
keeps it testable without a socket.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import time
from typing import Awaitable, Callable, Optional

from app.chat import CoachChat, Reply
from app.stt import MAX_CLIP_BYTES, MAX_CLIP_S, MIN_CLIP_S, Transcript, clip_seconds

Emit = Callable[[dict], Awaitable[None]]
Speak = Callable[[str], Awaitable[None]]
Facts = Callable[[], dict]
Transcribe = Callable[[bytes], Awaitable[Transcript]]

MAX_CHUNK_BYTES = 64_000     # one ptt_audio message; bigger ones are ignored
LISTEN_GRACE_S  = 2.0        # after MAX_CLIP_S with no ptt_end, end the clip anyway
MIN_CONFIDENCE  = 0.3        # starting guess, not measured: below this, ask again

LINES = {
    "too_short": "Hold the button while you talk.",
    "not_heard": "I didn't catch that. Say it again?",
    "stt_failed": "I couldn't hear that one. Try again, or type it.",
    "stt_off": "I can't hear yet on this setup. Type your question instead.",
}


class VoiceOrchestrator:
    """Runs user turns for one session."""

    def __init__(
        self,
        emit: Emit,
        speak: Speak,
        facts: Facts,
        chat: Optional[CoachChat] = None,
        on_busy: Optional[Callable[[bool], None]] = None,
        transcribe: Optional[Transcribe] = None,
        stt_available: bool = False,
    ) -> None:
        self._emit = emit
        self._speak = speak
        self._facts = facts
        self.chat = chat or CoachChat()
        self._on_busy = on_busy or (lambda busy: None)
        self._transcribe = transcribe
        self.stt_available = stt_available and transcribe is not None
        self.state = "idle"
        self.busy = False
        self.last_reply: Optional[Reply] = None
        self._clip: Optional[bytearray] = None
        self._listen_timer: Optional[asyncio.Task] = None

    async def _set_state(self, state: str) -> None:
        self.state = state
        await self._emit({"type": "voice_state", "state": state})

    def _claim(self) -> bool:
        if self.busy:
            return False
        self.busy = True
        self._on_busy(True)
        return True

    async def _release(self) -> None:
        self.busy = False
        self._on_busy(False)
        await self._set_state("idle")

    async def handle_text(self, text: str, source: str = "text") -> Optional[Reply]:
        """Answer one typed (or already transcribed) turn. None if another turn is running."""
        if not self._claim():
            await self._emit({"type": "voice_state", "state": self.state, "busy": True})
            return None
        try:
            return await self._answer(text, source)
        finally:
            await self._release()

    # ---- push-to-talk ---------------------------------------------------

    @property
    def listening(self) -> bool:
        return self._clip is not None

    async def ptt_start(self) -> bool:
        """The dancer pressed the button. False if a turn is already running."""
        if not self._claim():
            await self._emit({"type": "voice_state", "state": self.state, "busy": True})
            return False
        if not self.stt_available:
            try:
                await self._set_state("speaking")
                await self._speak(LINES["stt_off"])
            finally:
                await self._release()
            return False
        self._clip = bytearray()
        await self._set_state("listening")
        self._listen_timer = asyncio.create_task(self._end_after(MAX_CLIP_S + LISTEN_GRACE_S))
        return True

    async def _end_after(self, seconds: float) -> None:
        await asyncio.sleep(seconds)
        if self.listening:
            print("Turn: push-to-talk clip never ended, ending it now")
            await self.ptt_end()

    def ptt_audio(self, b64: str) -> None:
        """One chunk of 16 kHz mono PCM16, base64. Ignored unless listening."""
        if self._clip is None or not isinstance(b64, str):
            return
        try:
            chunk = base64.b64decode(b64, validate=True)
        except (binascii.Error, ValueError):
            return
        if len(chunk) > MAX_CHUNK_BYTES:
            return
        room = MAX_CLIP_BYTES - len(self._clip)
        if room > 0:
            self._clip.extend(chunk[:room])

    def _take_clip(self) -> Optional[bytes]:
        clip, self._clip = self._clip, None
        if self._listen_timer is not None and self._listen_timer is not asyncio.current_task():
            self._listen_timer.cancel()
        self._listen_timer = None
        return bytes(clip) if clip is not None else None

    async def ptt_cancel(self) -> None:
        """The dancer slid off the button: drop the clip, say nothing."""
        if self._take_clip() is not None:
            await self._release()

    async def ptt_end(self) -> Optional[Reply]:
        """The dancer let go: transcribe, drop the audio, answer."""
        pcm = self._take_clip()
        if pcm is None:
            return None
        try:
            if clip_seconds(pcm) < MIN_CLIP_S:
                await self._say(LINES["too_short"])
                return None
            await self._set_state("transcribing")
            start = time.perf_counter()
            try:
                heard = await self._transcribe(pcm)
            except Exception as e:
                print(f"Turn: speech to text failed ({type(e).__name__})")
                await self._say(LINES["stt_failed"])
                return None
            finally:
                del pcm   # the audio is not kept past this point
            stt_ms = (time.perf_counter() - start) * 1000
            text = (heard.text or "").strip()
            if not text or (heard.confidence is not None and heard.confidence < MIN_CONFIDENCE):
                print(f"Turn: input=voice answer=not_heard stt_ms={stt_ms:.0f}")
                await self._say(LINES["not_heard"])
                return None
            return await self._answer(text, "voice", stt_ms)
        finally:
            await self._release()

    async def _say(self, line: str) -> None:
        await self._set_state("speaking")
        await self._speak(line)

    # ---- shared ------------------------------------------------------------

    async def _answer(self, text: str, source: str, stt_ms: Optional[float] = None) -> Reply:
        await self._emit({"type": "transcript", "text": text, "source": source})
        await self._set_state("thinking")
        reply = await self.chat.answer(text, self._facts())
        self.last_reply = reply
        log_turn(reply, source, stt_ms)
        await self._set_state("speaking")
        await self._speak(reply.text)
        return reply


def log_turn(reply: Reply, source: str, stt_ms: Optional[float] = None) -> None:
    """One log line per turn (NFR-18). No transcript and no reply text: those stay out of logs."""
    parts = [f"input={source}", f"answer={reply.source}"]
    if reply.reason:
        parts.append(f"reason={reply.reason}")
    if stt_ms is not None:
        parts.append(f"stt_ms={stt_ms:.0f}")
    if reply.llm_ms is not None:
        parts.append(f"llm_ms={reply.llm_ms:.0f}")
    if reply.tokens_in or reply.tokens_out:
        parts.append(f"tokens_in={reply.tokens_in} tokens_out={reply.tokens_out}")
    print("Turn: " + " ".join(parts))
