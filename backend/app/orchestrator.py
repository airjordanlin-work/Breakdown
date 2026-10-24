"""Turn-taking between the dancer and the coach (spec 3.1, M3).

One user turn at a time. A turn is a typed question (``chat``) or, later, a
push-to-talk clip. While a turn is in progress:

- live cues stay quiet (the dancer is talking or listening);
- a second turn is refused with a short ``voice_state`` message instead of
  queueing up more model calls;
- the phone gets ``voice_state`` updates (thinking, speaking, idle) so it can
  show what the coach is doing.

The orchestrator does not know about WebSockets. The server passes in
``emit`` (send a JSON message to the phone and viewers), ``speak`` (caption
plus TTS audio) and ``facts`` (the measured values a reply may use), which
keeps it testable without a socket.
"""

from __future__ import annotations

from typing import Awaitable, Callable, Optional

from app.chat import CoachChat, Reply

Emit = Callable[[dict], Awaitable[None]]
Speak = Callable[[str], Awaitable[None]]
Facts = Callable[[], dict]


class VoiceOrchestrator:
    """Runs user turns for one session."""

    def __init__(
        self,
        emit: Emit,
        speak: Speak,
        facts: Facts,
        chat: Optional[CoachChat] = None,
        on_busy: Optional[Callable[[bool], None]] = None,
    ) -> None:
        self._emit = emit
        self._speak = speak
        self._facts = facts
        self.chat = chat or CoachChat()
        self._on_busy = on_busy or (lambda busy: None)
        self.state = "idle"
        self.busy = False
        self.last_reply: Optional[Reply] = None

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

    async def _answer(self, text: str, source: str) -> Reply:
        await self._emit({"type": "transcript", "text": text, "source": source})
        await self._set_state("thinking")
        reply = await self.chat.answer(text, self._facts())
        self.last_reply = reply
        log_turn(reply, source)
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
