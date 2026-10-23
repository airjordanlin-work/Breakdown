"""Answers to the dancer's own questions, typed or spoken (FR-11, FR-12, FR-15).

A user turn costs money only when it has to. In order:

1. Junk (empty, no words) gets a templated "say that again".
2. The per-session rate limit (``turn_gate.TurnGate``) is checked.
3. The same question about the same facts gets the same answer again, free.
4. Common questions ("hi", "how did I do", "what should I fix", "what went
   well", "streak", "help", "thanks") are answered straight from the facts,
   with no model call.
5. Anything else goes to Claude Haiku, if the session cap and the daily token
   budget allow it, with only the facts block and the last few turns.
6. The model's answer must pass the grounding guard; if it fails, or the call
   fails or times out, a templated answer from the same facts is used.

Every reply is built from ``facts``: measured values from the scorer. Nothing
here invents a body or score fact (CLAUDE.md rule 1).
"""

from __future__ import annotations

import json
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

import httpx

from app.coach import ANTHROPIC_API_KEY, ANTHROPIC_URL, FIX_PHRASES, MODEL
from app.grounding import ungrounded_claims
from app.turn_gate import DAILY_BUDGET, MAX_TEXT_CHARS, TokenBudget, TurnGate, normalize, screen_text

CHAT_MAX_TOKENS = 120        # reply length cap sent to the model
CHAT_MAX_WORDS  = 45         # longer replies are dropped for a templated one
CHAT_TIMEOUT_S  = 6.0
HISTORY_TURNS   = 4          # user/coach pairs sent with each model call

# (system, messages, max_tokens, timeout) -> (text, input_tokens, output_tokens), or None on failure.
# Tests pass a fake; production uses anthropic_chat.
ChatLLM = Callable[[str, list, int, float], Awaitable[Optional[tuple[str, int, int]]]]


CHAT_SYSTEM_PROMPT = """You are a warm, plain-spoken breaking coach. The dancer asked you something
between rounds of practice. Answer like a real coach standing next to them.
Rules:
- 1 to 3 short spoken sentences, at most 40 words. No lists, no emoji, no quotes.
- Use ONLY the FACTS block for anything about their body or results. Every body part,
  side and number you say must be in FACTS. Say numbers exactly as in FACTS or not at all.
- If they ask about something FACTS does not cover (a body part, music, timing, how they
  look, past sessions), say plainly that you did not measure that, then point to what you
  did measure.
- Never mention angles, degrees, points, grades or percentages. Never compare with
  earlier sessions or days, and never say they are improving. You have no history.
- Do not talk about the beat, rhythm, tempo, timing, pace, speed or music.
- Stay on breaking and this practice. If they ask about anything else, answer in one short
  friendly line that brings them back to practice.
- At most one question back to them.
"""


@dataclass
class Reply:
    text: str
    source: str              # quick | cache | model | fallback | limit | screen
    reason: Optional[str] = None
    tokens_in: int = 0
    tokens_out: int = 0
    llm_ms: Optional[float] = None
    extra: dict = field(default_factory=dict)


def chat_facts(move_name: str, stats: Optional[dict], last_attempt: Optional[dict]) -> dict:
    """The facts a reply may use: session tallies, the focus joint, and the last attempt's card.

    ``stats`` is the frame payload's stats dict; ``last_attempt`` is the latest
    rep_summary message. Replay positions and times are left out: they are
    not things to say, and their numbers would loosen the number check.
    """
    stats = stats or {}
    facts: dict = {"move": move_name or "this move"}
    facts["session"] = {
        k: int(stats.get(k) or 0) for k in ("reps", "clean_reps", "clean_streak", "best_streak")
    }
    if stats.get("focus_joint"):
        facts["focus"] = {"joint": stats["focus_joint"], "fix": stats.get("focus_fix") or "straighten"}
    if last_attempt:
        fix = last_attempt.get("fix") or None
        if fix:
            fix = {k: fix[k] for k in ("joint", "fix", "times_off", "checkpoint", "of") if k in fix}
        facts["last_attempt"] = {
            "reps": int(last_attempt.get("reps") or 0),
            "clean_reps": int(last_attempt.get("clean_reps") or 0),
            "best_streak": int(last_attempt.get("best_streak") or 0),
            "went_well": list(last_attempt.get("went_well") or []),
            "fix": fix,
        }
    return facts


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def _fix_line(fix: Optional[dict]) -> Optional[str]:
    if not fix or not fix.get("joint"):
        return None
    phrase = FIX_PHRASES.get(fix.get("fix") or "straighten", fix.get("fix") or "straighten")
    return f"Work on your {fix['joint']}: {phrase} it." if phrase in ("straighten", "bend") \
        else f"Work on your {fix['joint']}, they need to {phrase}."


def _the_fix(facts: dict) -> Optional[dict]:
    last = facts.get("last_attempt") or {}
    return last.get("fix") or facts.get("focus")


def summary_line(facts: dict) -> str:
    """How it went, from the last attempt if there is one, else the session."""
    last = facts.get("last_attempt")
    if last and last.get("reps"):
        line = f"That attempt was {_plural(last['reps'], 'rep')}, {last['clean_reps']} clean."
    else:
        s = facts.get("session") or {}
        if not s.get("reps"):
            return "No reps scored yet. Do a full rep, then stop, and I'll tell you how it went."
        line = f"So far {_plural(s['reps'], 'rep')}, {s['clean_reps']} clean."
    fix = _fix_line(_the_fix(facts))
    return f"{line} {fix}" if fix else f"{line} Nothing to fix right now, keep it going."


_INTENTS: list[tuple[str, re.Pattern]] = [
    ("greet",  re.compile(r"^(hi|hey|hello|yo|sup|hiya|what'?s up|good (morning|evening|afternoon))\b( (coach|there|man|bro))?$")),
    ("thanks", re.compile(r"^(thanks|thank you|thx|cheers|appreciate it)\b")),
    ("help",   re.compile(r"\b(help|what can (you|i) (do|ask)|how does this work)\b")),
    ("well",   re.compile(r"\bwhat (went|did i do) (well|right|good)\b|\bwhat was good\b")),
    ("fix",    re.compile(r"\b(what('?s| is) wrong|what (should|do|can) i (fix|work on|change|improve)|what to fix|what am i doing wrong|fix)\b")),
    ("streak", re.compile(r"\bstreak\b")),
    ("how",    re.compile(r"\b(how (did|was|'?d) (i|that|it)|how am i doing|how many|my score|score|how'?d i do|was that (good|clean))\b")),
    ("demo",   re.compile(r"\b(show me|demo)\b")),
]


def quick_answer(text: str, facts: dict) -> Optional[str]:
    """A reply straight from the facts for a common question, or None."""
    said = normalize(text)
    for intent, pattern in _INTENTS:
        if not pattern.search(said):
            continue
        if intent == "greet":
            s = facts.get("session") or {}
            if s.get("reps"):
                return f"Hey! {_plural(s['reps'], 'rep')} so far, {s['clean_reps']} clean. Ask me what to fix."
            return "Hey! I'm watching. Do a few reps, stop, and ask me how it went."
        if intent == "thanks":
            return "Any time. Go again when you're ready."
        if intent == "help":
            return "Ask me how that went, what to fix, or what went well. I answer from what I measured."
        if intent == "well":
            went = (facts.get("last_attempt") or {}).get("went_well") or []
            if went:
                return f"Your {' and '.join(went[:3])} landed every rep. Nice."
            return "Nothing landed every rep yet. " + (_fix_line(_the_fix(facts)) or "Do a full rep and I'll tell you.")
        if intent == "fix":
            return _fix_line(_the_fix(facts)) or "Nothing flagged yet. Do a full rep and I'll tell you."
        if intent == "streak":
            s = facts.get("session") or {}
            return f"Clean streak is {s.get('clean_streak', 0)}, best is {s.get('best_streak', 0)}."
        if intent == "how":
            return summary_line(facts)
        if intent == "demo":
            return "Tap SHOW ME and I'll play the move slowly."
    return None


_LIMIT_LINES = {
    "rate": "Give me a second. Keep dancing and ask me again.",
    "session_cap": "That's all my questions for this session. The card shows what to fix.",
    "daily_budget": "I'm out of coach time for today, so short answers only. ",
    "screen": "I didn't catch that. Say it again?",
}


def _estimate_tokens(system: str, messages: list) -> int:
    """Rough input size, erring high (about 3 characters per token)."""
    return (len(system) + len(json.dumps(messages))) // 3 + 1


async def anthropic_chat(system: str, messages: list, max_tokens: int, timeout: float) -> Optional[tuple[str, int, int]]:
    """One Messages API call with usage. None on timeout or any error."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                ANTHROPIC_URL,
                headers={
                    "x-api-key":         ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type":      "application/json",
                },
                json={"model": MODEL, "max_tokens": max_tokens, "system": system, "messages": messages},
            )
            resp.raise_for_status()
            body = resp.json()
            usage = body.get("usage") or {}
            return body["content"][0]["text"], int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))
    except httpx.TimeoutException:
        print("CoachChat: timeout, using a templated answer")
        return None
    except Exception as e:
        print(f"CoachChat: error ({e}), using a templated answer")
        return None


class CoachChat:
    """One session's conversation with the coach: limits, quick answers, Haiku, guard."""

    def __init__(
        self,
        llm: Optional[ChatLLM] = None,
        gate: Optional[TurnGate] = None,
        budget: Optional[TokenBudget] = None,
    ) -> None:
        self._llm: ChatLLM = llm or anthropic_chat
        self._available = llm is not None or bool(ANTHROPIC_API_KEY)
        self.gate = gate or TurnGate()
        self.budget = budget or DAILY_BUDGET
        self.history: deque[tuple[str, str]] = deque(maxlen=HISTORY_TURNS)
        self._last: Optional[tuple[str, str, Reply]] = None   # (question, facts json, reply)
        self.refused = 0

    async def answer(self, text: str, facts: dict) -> Reply:
        text = (text or "")[:MAX_TEXT_CHARS]
        if screen_text(text) is not None:
            return Reply(_LIMIT_LINES["screen"], "screen", screen_text(text))

        why = self.gate.admit_turn()
        if why:
            return Reply(_LIMIT_LINES[why], "limit", why)

        key = (normalize(text), json.dumps(facts, sort_keys=True))
        if self._last and self._last[:2] == key:
            return Reply(self._last[2].text, "cache")

        quick = quick_answer(text, facts)
        if quick:
            return self._done(key, text, Reply(quick, "quick"))

        if not self._available:
            return self._done(key, text, Reply(summary_line(facts), "fallback", "no_api_key"))
        why = self.gate.admit_model()
        if why:
            return Reply(_LIMIT_LINES[why], "limit", why)

        messages = self._messages(text, facts)
        estimate = _estimate_tokens(CHAT_SYSTEM_PROMPT, messages) + CHAT_MAX_TOKENS
        if not self.budget.can_spend(estimate):
            return Reply(_LIMIT_LINES["daily_budget"] + summary_line(facts), "limit", "daily_budget")

        start = time.perf_counter()
        result = await self._llm(CHAT_SYSTEM_PROMPT, messages, CHAT_MAX_TOKENS, CHAT_TIMEOUT_S)
        ms = (time.perf_counter() - start) * 1000
        if result is None:
            # A failed call may still have been billed; count the estimate to stay safe.
            self.budget.spend(estimate)
            return self._done(key, text, Reply(summary_line(facts), "fallback", "llm_failed", llm_ms=ms))
        reply_text, t_in, t_out = result
        self.budget.spend(t_in + t_out)
        reply = Reply(reply_text.strip().strip('"').strip(), "model", tokens_in=t_in, tokens_out=t_out, llm_ms=ms)
        reason = self._reject(reply.text, facts)
        if reason:
            self.refused += 1
            print(f"CoachChat: refused model answer ({reason})")
            reply = Reply(summary_line(facts), "fallback", "guard", tokens_in=t_in, tokens_out=t_out, llm_ms=ms)
        return self._done(key, text, reply)

    def _reject(self, text: str, facts: dict) -> Optional[str]:
        if not text:
            return "empty"
        if len(text.split()) > CHAT_MAX_WORDS:
            return "too long"
        if text.count("?") > 1:
            return "too many questions"
        flags = ungrounded_claims(text, facts)
        return "; ".join(flags) if flags else None

    def _messages(self, text: str, facts: dict) -> list:
        messages: list = []
        for q, a in self.history:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})
        messages.append({
            "role": "user",
            "content": "FACTS (measured this session):\n" + json.dumps(facts, indent=1) + "\n\nDancer: " + text,
        })
        return messages

    def _done(self, key: tuple[str, str], text: str, reply: Reply) -> Reply:
        self.history.append((text, reply.text))
        self._last = (key[0], key[1], reply)
        return reply
