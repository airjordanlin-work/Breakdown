"""
Real-time AI coaching module.

Uses Claude Haiku to generate short, natural coaching cues from live pose
data, and a short spoken debrief when the dancer stops after an attempt.
The coach only speaks while a pose is tracked and the dancer is moving (or
right after they stop, for the debrief), and a cooldown keeps it from
feeling spammy.

Every turn is built from a ``facts`` dict of measured values. The model's
reply is checked by ``grounding.ungrounded_claims`` against those facts; a
reply that names an unmeasured joint, an unknown number, or a past session
is dropped. When the API is missing, fails, or is refused by the guard, a
templated line built from the same facts is used.
"""

from __future__ import annotations

import json
import os
import random
import time
from collections import deque
from typing import Awaitable, Callable, Optional, Sequence

import httpx
from dotenv import load_dotenv

from app.attempts import attempt_facts
from app.grounding import ungrounded_claims

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_URL     = "https://api.anthropic.com/v1/messages"
MODEL             = "claude-haiku-4-5-20251001"
MAX_TOKENS        = 40       # short cues only
MAX_CUE_WORDS     = 8        # longer replies are dropped for a templated cue
COOLDOWN_SECONDS  = 5.0      # between cues that carry a joint correction or praise
COOLDOWN_ENCOURAGE = 15.0    # moving, but no joint data yet: speak less
RECENT_CUES       = 4        # never repeat one of the last few cues
TIMEOUT_SECONDS   = 4.0      # max wait for API response
DEBRIEF_MAX_TOKENS = 120
DEBRIEF_MAX_WORDS = 25       # spec 3.2: live and debrief replies stay under 25 words
DEBRIEF_TIMEOUT_SECONDS = 6.0
STREAK_HYPE_MIN   = 2        # only mention a clean streak from this length up

# (system, prompt, max_tokens, timeout) -> reply text, or None on any failure.
# Tests pass a fake; production uses anthropic_complete.
LLMClient = Callable[[str, str, int, float], Awaitable[Optional[str]]]


SYSTEM_PROMPT = """You are a friendly breaking coach talking to a dancer while they practice.
Talk like a real person standing next to them: warm, short, plain words.
Rules:
- ONE short phrase, at most 6 words. No quotes, no emoji.
- Only name a body part if the message names it, and use the exact side given.
- Never say numbers, angles, degrees or scores.
- No gym talk ("engage your core") and no forced slang.
Good: "Straighten that left elbow", "Yes, that's it", "Keep it flowing"
Grounding (never break this):
- Only state what the message says was measured. If a body part, side or
  count is not in the message, do not mention it.
- Never compare with earlier sessions. You have no data about them.
"""

DEBRIEF_SYSTEM_PROMPT = """You are a friendly breaking coach. The dancer just stopped after an attempt
and you give them a quick spoken debrief, like a real coach would between rounds.
Rules:
- 1 to 3 short spoken sentences, under 25 words in total. No lists, no emoji, no quotes.
- Use ONLY the FACTS block. Every body part, side and number you say must be in FACTS.
- Say numbers exactly as they appear in FACTS, or not at all.
- Never mention angles, degrees, points, grades or percentages.
- Never compare with earlier sessions or days. You have no data about them.
- Lead with something that went well if FACTS shows it, then the one thing to work on.
- You may end with one short question. At most one question.
"""

# Wearable sensor names -> how a coach would say it
_LIMB_WORDS = {"wrist": "base arm", "leg": "legs"}

# Templated cues: used when the API key is missing, the call fails or times
# out, or the model repeats itself. Same facts as the prompt, nothing more.
_FIX_TEMPLATES = {
    "straighten": ["Straighten your {joint}", "Open up that {joint}", "Reach through your {joint}"],
    "bend":       ["Bend your {joint} more", "Bring your {joint} in", "Soften that {joint}"],
    # whole-rep features (app/move_features.py)
    "twist":      ["Twist your {joint} more", "Turn those {joint} more", "More twist in the {joint}"],
    "reach":      ["Extend that {joint} out", "Reach your {joint} out more", "Kick that {joint} out wider"],
}
# How the prompt says each fix: "their hips need to twist more".
FIX_PHRASES = {
    "straighten": "straighten",
    "bend": "bend",
    "twist": "twist more through the rep",
    "reach": "reach out further to the side",
}
_CLOSE_TEMPLATES = ["Almost, {fix} your {joint}", "Close, {fix} your {joint} a bit", "Nearly, {fix} your {joint}"]
_PERFECT_TEMPLATES = ["Yes, that's it", "Clean, keep going", "Just like that", "That's the one"]
_ENCOURAGE_TEMPLATES = ["Keep it moving", "Nice, stay with it", "Good, keep flowing", "Stay loose"]
_FREEZE_CLEAN_TEMPLATES = ["Solid hold", "Nice freeze", "Locked in, nice"]
_FREEZE_SHAKY_TEMPLATES = ["Steady your {limb}", "Lock your {limb} in", "Settle your {limb}"]


def correction_focus(score_result: Optional[dict]) -> Optional[dict]:
    """The one joint result a cue should be about: worst Miss, else worst Close.

    Returns {"tier": "Miss" | "Close" | "Perfect", "joint", "fix"}, or None when
    there is no joint data. "Perfect" means every checked joint matched.
    """
    results = (score_result or {}).get("results") or []
    if not results:
        return None
    for tier in ("Miss", "Close"):
        hits = [r for r in results if r.get("tier") == tier]
        if hits:
            r = max(hits, key=lambda r: abs(r.get("diff", 0)))
            return {"tier": tier, "joint": r["joint"], "fix": r.get("fix") or "straighten"}
    return {"tier": "Perfect", "joint": None, "fix": None}


def live_facts(
    move_name: str,
    score_result: Optional[dict],
    previous_result: Optional[dict] = None,
    clean_streak: int = 0,
    move_tips: Sequence[str] = (),
) -> dict:
    """What a live cue may talk about: this rep's worst joint, the rep before, the streak."""
    facts: dict = {"move": move_name or "this move"}
    focus = correction_focus(score_result)
    if focus is None:
        if move_tips:
            facts["tips"] = list(move_tips)
        return facts
    facts["focus"] = focus
    prev = correction_focus(previous_result)
    if prev is not None and focus["joint"] and prev.get("joint") == focus["joint"]:
        facts["same_joint_as_last_rep"] = True
    if previous_result:
        now_perfect = {r["joint"] for r in score_result.get("results", []) if r.get("tier") == "Perfect"}
        was_off = [r["joint"] for r in previous_result.get("results", []) if r.get("tier") != "Perfect"]
        fixed = [j for j in was_off if j in now_perfect]
        if fixed:
            facts["fixed_since_last_rep"] = fixed
    if clean_streak >= STREAK_HYPE_MIN:
        facts["clean_streak"] = int(clean_streak)
    return facts


def freeze_facts(freeze: dict) -> dict:
    """What a freeze cue may talk about, from the wearable sensors."""
    return {
        "held_seconds": round(float(freeze.get("duration_s", 0)), 1),
        "stability": freeze.get("stability", 0),
        "rating": freeze.get("tier", ""),
        "shakiest": _LIMB_WORDS.get(freeze.get("shakiest") or "", "body"),
    }


def _build_freeze_prompt(freeze: dict, last_cue: Optional[str]) -> str:
    """Prompt for a cue right after a freeze, based on wearable IMU data.

    Only facts the sensors actually measured go into the prompt, so the cue
    can't invent feedback the data doesn't support.
    """
    tier      = freeze.get("tier", "")
    duration  = freeze.get("duration_s", 0)
    stability = freeze.get("stability", 0)
    limb      = _LIMB_WORDS.get(freeze.get("shakiest") or "", "body")

    lines = [
        "The user just finished a freeze, measured by motion sensors on their body.",
        f"Held for {duration:.1f} seconds. Stability {stability}/100, rated \"{tier}\".",
    ]
    if tier in ("rock solid", "steady"):
        lines.append("It was clean. Give a short hype cue that acknowledges the hold.")
    else:
        lines.append(f"It was wobbly, mostly in the {limb}.")
        lines.append(f"Give a short correction cue about stabilizing the {limb}.")
    lines.append("Only mention the hold time if you use the exact number above.")
    if last_cue:
        lines.append(f'Do NOT repeat: "{last_cue}"')
    lines.append("ONE phrase, max 10 words, sound like a real bboy coach.")
    return "\n".join(lines)


def _build_prompt(
    move_name: str,
    score_result: Optional[dict],
    recent_cues: Sequence[str] = (),
    move_tips: Sequence[str] = (),
    facts: Optional[dict] = None,
) -> str:
    """Prompt for a live cue while the dancer is moving through a move.

    Only called when a pose is tracked and the dancer is moving. The joint and
    the direction come from the scorer; nothing else about the body is given.
    """
    lines = [f'The dancer is practicing the move "{move_name or "this move"}" right now.']
    focus = correction_focus(score_result)
    if focus is None:
        lines.append("No joint measurements yet. Give short encouragement to keep going.")
        if move_tips:
            tips = "; ".join(move_tips)
            lines.append(f"Written tips for this move (you may use one): {tips}")
        else:
            lines.append("Do not name any body part.")
    elif focus["tier"] == "Miss":
        phrase = FIX_PHRASES.get(focus["fix"], focus["fix"])
        lines.append(f"Measured: their {focus['joint']} needs to {phrase}.")
        lines.append(f"Give one clear correction about the {focus['joint']}.")
    elif focus["tier"] == "Close":
        phrase = FIX_PHRASES.get(focus["fix"], focus["fix"])
        lines.append(f"Measured: their {focus['joint']} is close, it needs to {phrase} a little.")
        lines.append(f"Give one small, encouraging refinement about the {focus['joint']}.")
    else:
        lines.append("Measured: every checked joint matched the reference.")
        lines.append("Give short praise. Do not name any body part.")
    facts = facts or {}
    if facts.get("same_joint_as_last_rep"):
        lines.append("Same joint was off on the rep before too, so you can say it is still that joint.")
    if facts.get("fixed_since_last_rep"):
        fixed = ", ".join(facts["fixed_since_last_rep"])
        lines.append(f"Their {fixed} was off last rep and landed this rep. You can call that out.")
    if facts.get("clean_streak"):
        lines.append(f"They are on {facts['clean_streak']} clean reps in a row. You may hype that, with that exact number.")
    if recent_cues:
        quoted = ", ".join(f'"{c}"' for c in recent_cues)
        lines.append(f"Do NOT repeat or closely reword any of: {quoted}")
    lines.append("ONE phrase, max 6 words.")
    return "\n".join(lines)


def _pick(options: Sequence[str], recent: Sequence[str]) -> str:
    fresh = [o for o in options if o not in recent]
    if not fresh and recent:
        # every option was used lately: at least never say the last one again
        fresh = [o for o in options if o != recent[-1]]
    return random.choice(fresh or list(options))


def fallback_cue(
    score_result: Optional[dict],
    recent_cues: Sequence[str] = (),
    freeze: Optional[dict] = None,
) -> str:
    """Templated cue built from the same facts the prompt would get."""
    if freeze:
        if freeze.get("tier") in ("rock solid", "steady"):
            return _pick(_FREEZE_CLEAN_TEMPLATES, recent_cues)
        limb = _LIMB_WORDS.get(freeze.get("shakiest") or "", "body")
        return _pick(_FREEZE_SHAKY_TEMPLATES, recent_cues).format(limb=limb)

    focus = correction_focus(score_result)
    if focus is None:
        return _pick(_ENCOURAGE_TEMPLATES, recent_cues)
    if focus["tier"] == "Perfect":
        return _pick(_PERFECT_TEMPLATES, recent_cues)
    if focus["tier"] == "Miss":
        templates = _FIX_TEMPLATES.get(focus["fix"], _FIX_TEMPLATES["straighten"])
        options = [t.format(joint=focus["joint"]) for t in templates]
    else:
        options = [t.format(joint=focus["joint"], fix=focus["fix"]) for t in _CLOSE_TEMPLATES]
    return _pick(options, recent_cues)


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def fallback_debrief(facts: dict) -> str:
    """Templated debrief from the same facts the model would get."""
    reps, clean = facts.get("reps", 0), facts.get("clean_reps", 0)
    if clean == reps:
        parts = [f"All {_plural(reps, 'rep')} clean, nice."] if reps > 1 else ["Clean rep, nice."]
    else:
        parts = [f"That's {_plural(reps, 'rep')}, {clean} clean."]
    best = facts.get("best_streak", 0)
    if best >= STREAK_HYPE_MIN and best < reps:
        parts.append(f"Best run was {best} in a row.")
    off = facts.get("off_most")
    if off:
        parts.append(f"Next one, {off['fix']} your {off['joint']}.")
    else:
        parts.append("Keep that going.")
    return " ".join(parts)


def _build_debrief_prompt(facts: dict, recent: Sequence[str] = ()) -> str:
    lines = [
        f'The dancer just stopped after an attempt at "{facts.get("move", "this move")}".',
        "FACTS (measured on this attempt; clean means no joint missed its target):",
        json.dumps(facts, indent=1),
    ]
    if recent:
        quoted = ", ".join(f'"{c}"' for c in recent)
        lines.append(f"Your recent lines, do not repeat them: {quoted}")
    lines.append("Give the debrief now.")
    return "\n".join(lines)


def _clean_reply(text: str) -> str:
    return text.strip().strip('"').strip()


async def anthropic_complete(system: str, prompt: str, max_tokens: int, timeout: float) -> Optional[str]:
    """One Messages API call. None on timeout or any error."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                ANTHROPIC_URL,
                headers={
                    "x-api-key":         ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type":      "application/json",
                },
                json={
                    "model":      MODEL,
                    "max_tokens": max_tokens,
                    "system":     system,
                    "messages":   [{"role": "user", "content": prompt}],
                },
            )
            resp.raise_for_status()
            return resp.json()["content"][0]["text"]
    except httpx.TimeoutException:
        print("AICoach: timeout, using a templated line")
        return None
    except Exception as e:
        print(f"AICoach: error ({e}), using a templated line")
        return None


class AICoach:
    """Live cues and stop-time debriefs from Claude Haiku, with templated fallback."""

    def __init__(self, llm: Optional[LLMClient] = None) -> None:
        self._last_called: float  = 0.0
        self._last_cue: Optional[str] = None
        self._recent: deque[str]  = deque(maxlen=RECENT_CUES)
        self._inflight: int       = 0
        self._llm: LLMClient      = llm or anthropic_complete
        self._available: bool     = llm is not None or bool(ANTHROPIC_API_KEY)
        self.refused: int         = 0   # replies dropped by the grounding guard
        self._quiet_until: float  = 0.0  # no live cues before this time (a demo is playing)

        if not self._available:
            print("AICoach: ANTHROPIC_API_KEY not set — using templated cues")
            print("  Add ANTHROPIC_API_KEY=sk-ant-... to your .env file")
        else:
            print(f"AICoach: ready — model={MODEL} cooldown={COOLDOWN_SECONDS}s")

    @property
    def _pending(self) -> bool:
        return self._inflight > 0

    def hold(self, seconds: float) -> None:
        """No live cues for ``seconds``, e.g. while the dancer watches a demo."""
        self._quiet_until = max(self._quiet_until, time.time() + max(0.0, seconds))

    def release(self) -> None:
        """End a demo hold early, e.g. when the dancer closes the demo."""
        self._quiet_until = 0.0

    @property
    def held(self) -> bool:
        """True while a demo hold is active: nothing live or debrief should speak."""
        return time.time() < self._quiet_until

    def can_fire_now(self) -> bool:
        """For event-driven cues (like a finished freeze) that skip the cooldown."""
        return not self._pending and not self.held

    def should_fire(
        self,
        *,
        tracked: bool,
        moving: bool,
        fill_ratio: float,
        has_score: bool = False,
    ) -> bool:
        """Speak only while a pose is tracked this frame and the dancer is moving.

        No body in frame, a half-full buffer, or standing still all mean
        silence; the on-screen guide handles framing.
        """
        if self._pending or not (tracked and moving and fill_ratio >= 1.0):
            return False
        if time.time() < self._quiet_until:
            return False
        cooldown = COOLDOWN_SECONDS if has_score else COOLDOWN_ENCOURAGE
        return time.time() - self._last_called >= cooldown

    def _remember(self, cue: str) -> str:
        self._last_cue = cue
        self._recent.append(cue)
        return cue

    def _grounded(self, text: Optional[str], facts: dict) -> Optional[str]:
        """``text`` if the guard finds nothing unmeasured in it, else None."""
        if not text:
            return None
        flags = ungrounded_claims(text, facts)
        if flags:
            self.refused += 1
            print(f"AICoach: refused '{text}' ({'; '.join(flags)})")
            return None
        return text

    async def get_cue(
        self,
        move_name: str,
        score_result: Optional[dict],
        freeze: Optional[dict] = None,
        move_tips: Optional[Sequence[str]] = None,
        previous_result: Optional[dict] = None,
        clean_streak: int = 0,
    ) -> Optional[str]:
        """Return a short coaching cue: from Claude Haiku, or templated if that fails."""
        self._inflight   += 1
        self._last_called = time.time()
        recent = list(self._recent)
        try:
            if freeze:
                facts = freeze_facts(freeze)
            else:
                facts = live_facts(move_name, score_result, previous_result, clean_streak, move_tips or ())
            cue = None
            if self._available:
                cue = await self._ask_model(move_name, score_result, freeze, recent, move_tips or (), facts)
                cue = self._grounded(cue, facts)
            if not cue or cue.lower() in (c.lower() for c in recent):
                cue = fallback_cue(score_result, recent, freeze)
            print(f"AICoach: '{cue}'")
            return self._remember(cue)
        finally:
            self._inflight -= 1

    async def _ask_model(
        self,
        move_name: str,
        score_result: Optional[dict],
        freeze: Optional[dict],
        recent: list[str],
        move_tips: Sequence[str],
        facts: Optional[dict] = None,
    ) -> Optional[str]:
        """One Haiku call. None on timeout, error, or a reply that is too long."""
        if freeze:
            prompt = _build_freeze_prompt(freeze, self._last_cue)
        else:
            prompt = _build_prompt(move_name, score_result, recent, move_tips, facts)

        reply = await self._llm(SYSTEM_PROMPT, prompt, MAX_TOKENS, TIMEOUT_SECONDS)
        cue = _clean_reply(reply or "").rstrip(".")
        limit = 10 if freeze else MAX_CUE_WORDS
        if not cue or len(cue.split()) > limit:
            return None
        return cue

    async def get_debrief(
        self,
        move_name: str,
        reps: list[dict],
        session: Optional[dict] = None,
    ) -> Optional[str]:
        """Spoken debrief for an attempt that just ended. None if no reps were scored."""
        if not reps:
            return None
        facts = attempt_facts(move_name, reps, session)
        self._inflight += 1
        self._last_called = time.time()
        recent = list(self._recent)
        try:
            text = None
            if self._available:
                reply = await self._llm(
                    DEBRIEF_SYSTEM_PROMPT, _build_debrief_prompt(facts, recent),
                    DEBRIEF_MAX_TOKENS, DEBRIEF_TIMEOUT_SECONDS,
                )
                text = _clean_reply(reply or "")
                if len(text.split()) > DEBRIEF_MAX_WORDS or text.count("?") > 1:
                    print(f"AICoach: debrief too long or too many questions, using template: '{text}'")
                    text = None
                text = self._grounded(text, facts)
            if not text:
                text = fallback_debrief(facts)
            print(f"AICoach debrief: '{text}'")
            return self._remember(text)
        finally:
            self._inflight -= 1
