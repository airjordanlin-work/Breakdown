"""Spend limits for user turns (talking or typing to the coach).

Every user turn passes through here before anything costs money. All limits
are enforced on the server; the phone is never trusted to limit itself.

- ``screen_text``: junk that never reaches the model (empty, one word with no
  question in it, no letters, the same thing twice in a row).
- ``TurnGate``: per-session turn rate and per-session cap.
- ``TokenBudget``: a per-day token budget shared by every session in this
  process. It fails closed: once spent, only templated replies are sent until
  the day changes. It lives in memory, so a backend restart resets it.

The defaults are starting guesses (not measured) and can be changed in .env.
"""

from __future__ import annotations

import os
import re
import threading
import time
from collections import deque
from datetime import date
from typing import Callable, Optional


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


CHAT_TURNS_PER_MIN     = _env_int("CHAT_TURNS_PER_MIN", 4)
CHAT_TURNS_PER_SESSION = _env_int("CHAT_TURNS_PER_SESSION", 30)
CHAT_DAILY_TOKENS      = _env_int("CHAT_DAILY_TOKENS", 40_000)
MAX_TEXT_CHARS         = 300   # longer typed text is cut, never sent whole

_LETTER = re.compile(r"[a-zA-Z]")


def normalize(text: str) -> str:
    """Lowercase, collapse spaces, drop punctuation at the ends."""
    return re.sub(r"\s+", " ", (text or "").strip().lower()).strip(" .,!?")


def screen_text(text: str, previous: Optional[str] = None) -> Optional[str]:
    """Why ``text`` should not become a turn, or None if it may.

    Reasons: "empty", "no_words", "repeat".
    """
    said = normalize(text)
    if not said:
        return "empty"
    if not _LETTER.search(said):
        return "no_words"
    if previous is not None and said == normalize(previous):
        return "repeat"
    return None


class TurnGate:
    """Per-session limits.

    Every turn counts toward the per-minute rate (even free ones, since each
    reply is still spoken). Only turns that reach the model count toward the
    per-session cap.
    """

    def __init__(
        self,
        per_minute: int = CHAT_TURNS_PER_MIN,
        per_session: int = CHAT_TURNS_PER_SESSION,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.per_minute = per_minute
        self.per_session = per_session
        self._clock = clock
        self._recent: deque[float] = deque()
        self.model_turns = 0

    def admit_turn(self) -> Optional[str]:
        """Count one turn toward the rate, or return "rate" if the last minute is full."""
        now = self._clock()
        while self._recent and now - self._recent[0] >= 60.0:
            self._recent.popleft()
        if len(self._recent) >= self.per_minute:
            return "rate"
        self._recent.append(now)
        return None

    def admit_model(self) -> Optional[str]:
        """Count one model call, or return "session_cap" when this session has used them all."""
        if self.model_turns >= self.per_session:
            return "session_cap"
        self.model_turns += 1
        return None


class TokenBudget:
    """Tokens (input plus output) the coach may spend on replies per day."""

    def __init__(self, daily: int = CHAT_DAILY_TOKENS, today: Callable[[], date] = date.today) -> None:
        self.daily = daily
        self._today = today
        self._day = today()
        self.spent = 0
        self._lock = threading.Lock()

    def _roll(self) -> None:
        day = self._today()
        if day != self._day:
            self._day, self.spent = day, 0

    @property
    def remaining(self) -> int:
        with self._lock:
            self._roll()
            return max(0, self.daily - self.spent)

    def can_spend(self, estimate: int) -> bool:
        """True if a call expected to use ``estimate`` tokens fits what is left today."""
        return self.remaining >= max(0, estimate)

    def spend(self, tokens: int) -> None:
        with self._lock:
            self._roll()
            self.spent += max(0, int(tokens))


# One budget for the whole process: every session draws on the same day's tokens.
DAILY_BUDGET = TokenBudget()
