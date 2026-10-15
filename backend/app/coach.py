"""
Real-time AI coaching module.

Uses Claude Haiku to generate short, natural coaching cues from live pose
data. The coach only speaks while a pose is tracked and the dancer is
moving, and a cooldown keeps it from feeling spammy. When the API is
missing or fails, a templated cue built from the same facts is used.
"""

from __future__ import annotations

import os
import random
import time
from collections import deque
from typing import Optional, Sequence

import httpx
from dotenv import load_dotenv

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


SYSTEM_PROMPT = """You are a friendly breaking coach talking to a dancer while they practice.
Talk like a real person standing next to them: warm, short, plain words.
Rules:
- ONE short phrase, at most 6 words. No quotes, no emoji.
- Only name a body part if the message names it, and use the exact side given.
- Never say numbers, angles, degrees or scores.
- No gym talk ("engage your core") and no forced slang.
Good: "Straighten that left elbow", "Yes, that's it", "Keep it flowing"
"""

# Wearable sensor names -> how a coach would say it
_LIMB_WORDS = {"wrist": "base arm", "leg": "legs"}

# Templated cues: used when the API key is missing, the call fails or times
# out, or the model repeats itself. Same facts as the prompt, nothing more.
_FIX_TEMPLATES = {
    "straighten": ["Straighten your {joint}", "Open up that {joint}", "Reach through your {joint}"],
    "bend":       ["Bend your {joint} more", "Bring your {joint} in", "Soften that {joint}"],
}
_CLOSE_TEMPLATES = ["Almost, {fix} your {joint}", "Close, {fix} that {joint}", "Nearly, {fix} your {joint}"]
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
        lines.append(f"Measured: their {focus['joint']} needs to {focus['fix']}.")
        lines.append(f"Give one clear correction about the {focus['joint']}.")
    elif focus["tier"] == "Close":
        lines.append(f"Measured: their {focus['joint']} is close, it needs to {focus['fix']} a little.")
        lines.append(f"Give one small, encouraging refinement about the {focus['joint']}.")
    else:
        lines.append("Measured: every checked joint matched the reference.")
        lines.append("Give short praise. Do not name any body part.")
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
        options = [t.format(joint=focus["joint"]) for t in _FIX_TEMPLATES[focus["fix"]]]
    else:
        options = [t.format(joint=focus["joint"], fix=focus["fix"]) for t in _CLOSE_TEMPLATES]
    return _pick(options, recent_cues)


class AICoach:
    """Generates real-time coaching cues using Claude Haiku, with templated fallback."""

    def __init__(self) -> None:
        self._last_called: float  = 0.0
        self._last_cue: Optional[str] = None
        self._recent: deque[str]  = deque(maxlen=RECENT_CUES)
        self._pending: bool       = False
        self._available: bool     = bool(ANTHROPIC_API_KEY)

        if not self._available:
            print("AICoach: ANTHROPIC_API_KEY not set — using templated cues")
            print("  Add ANTHROPIC_API_KEY=sk-ant-... to your .env file")
        else:
            print(f"AICoach: ready — model={MODEL} cooldown={COOLDOWN_SECONDS}s")

    def can_fire_now(self) -> bool:
        """For event-driven cues (like a finished freeze) that skip the cooldown."""
        return not self._pending

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
        cooldown = COOLDOWN_SECONDS if has_score else COOLDOWN_ENCOURAGE
        return time.time() - self._last_called >= cooldown

    def _remember(self, cue: str) -> str:
        self._last_cue = cue
        self._recent.append(cue)
        return cue

    async def get_cue(
        self,
        move_name: str,
        score_result: Optional[dict],
        freeze: Optional[dict] = None,
        move_tips: Optional[Sequence[str]] = None,
    ) -> Optional[str]:
        """Return a short coaching cue: from Claude Haiku, or templated if that fails."""
        self._pending     = True
        self._last_called = time.time()
        recent = list(self._recent)
        try:
            cue = await self._ask_model(move_name, score_result, freeze, recent, move_tips or ()) if self._available else None
            if not cue or cue.lower() in (c.lower() for c in recent):
                cue = fallback_cue(score_result, recent, freeze)
            print(f"AICoach: '{cue}'")
            return self._remember(cue)
        finally:
            self._pending = False

    async def _ask_model(
        self,
        move_name: str,
        score_result: Optional[dict],
        freeze: Optional[dict],
        recent: list[str],
        move_tips: Sequence[str],
    ) -> Optional[str]:
        """One Haiku call. None on timeout, error, or a reply that is too long."""
        if freeze:
            prompt = _build_freeze_prompt(freeze, self._last_cue)
        else:
            prompt = _build_prompt(move_name, score_result, recent, move_tips)

        try:
            async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
                resp = await client.post(
                    ANTHROPIC_URL,
                    headers={
                        "x-api-key":         ANTHROPIC_API_KEY,
                        "anthropic-version": "2023-06-01",
                        "content-type":      "application/json",
                    },
                    json={
                        "model":      MODEL,
                        "max_tokens": MAX_TOKENS,
                        "system":     SYSTEM_PROMPT,
                        "messages":   [{"role": "user", "content": prompt}],
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                cue  = data["content"][0]["text"].strip().strip('"').rstrip(".")
        except httpx.TimeoutException:
            print("AICoach: timeout — using a templated cue")
            return None
        except Exception as e:
            print(f"AICoach: error — {e}; using a templated cue")
            return None

        limit = 10 if freeze else MAX_CUE_WORDS
        if not cue or len(cue.split()) > limit:
            return None
        return cue
