"""
Real-time AI coaching module.

Uses Claude Haiku to generate natural, context-aware coaching cues
based on live pose data. A cooldown gate ensures it only fires every
4-6 seconds so it never feels spammy.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_URL     = "https://api.anthropic.com/v1/messages"
MODEL             = "claude-haiku-4-5-20251001"
MAX_TOKENS        = 60       # short cues only
COOLDOWN_SECONDS  = 5.0   # when aligned
COOLDOWN_NO_BODY  = 12.0  # when not in frame — less frequent
TIMEOUT_SECONDS   = 4.0      # max wait for API response


SYSTEM_PROMPT = """You are an experienced bboy (breakdancer) coach giving real-time feedback.
You speak like a real street dance coach — direct, encouraging, authentic.
Rules:
- ONE short phrase only, maximum 10 words
- Never say "I" or "you should"  
- Sound like a bboy coach, not a fitness instructor
- Use breakdance vocabulary: freeze, lock, pop, flow, toprock, footwork, stance
- If correcting a joint, name it specifically: "left elbow", "right arm"
- Never give generic gym/fitness advice like "feet shoulder-width apart"
Good examples:
"Lock that left arm out"
"Stay low, keep the energy"
"Right elbow needs to open up"
"Nice freeze, hold it"
"Find the rhythm, stay loose"
"That's it, keep flowing"
Bad examples (never say these):
"Feet shoulder-width apart"
"Chest up, weight forward"
"Engage your core"
"""

# Wearable sensor names -> how a coach would say it
_LIMB_WORDS = {"wrist": "base arm", "leg": "legs"}


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
    aligned: bool,
    fill_ratio: float,
    score_result: Optional[dict],
    last_cue: Optional[str],
) -> str:
    lines = [f'The user is practicing breakdancing, working on move: "{move_name or "br_01"}"']

    if fill_ratio < 1.0:
        lines.append("The camera is still detecting the user's position.")
        lines.append("Give a very short breakdance-specific cue to help them get ready.")
        lines.append("Examples: 'Get low, spread your weight', 'Arms out, stay loose', 'Find your stance'")
        if last_cue:
            lines.append(f'Do NOT repeat: "{last_cue}"')
        lines.append("ONE phrase, max 8 words, sound like a bboy coach not a fitness instructor.")
        return "\n".join(lines)

    if not aligned:
        lines.append("The user is attempting the move but not yet aligned with the reference.")
        lines.append("Give an encouraging breakdance-specific cue.")
        lines.append("Examples: 'Keep moving, find the rhythm', 'Stay low', 'Lock it in'")
        if last_cue:
            lines.append(f'Do NOT repeat: "{last_cue}"')
        lines.append("ONE phrase, max 8 words.")
        return "\n".join(lines)

    lines.append("The user is aligned with the reference move.")

    if score_result and score_result.get("results"):
        results  = score_result["results"]
        misses   = [r for r in results if r["tier"] == "Miss"]
        closes   = [r for r in results if r["tier"] == "Close"]
        perfects = [r for r in results if r["tier"] == "Perfect"]

        if misses:
            r = misses[0]
            joint     = r["joint"]
            diff      = r["diff"]
            direction = "extend" if diff > 0 else "bend"
            lines.append(f"Joint issue: {joint} is {diff:.0f} degrees off — needs to {direction}")
            lines.append("Give a specific joint correction cue.")
        elif closes:
            r = closes[0]
            lines.append(f"Almost there: {r['joint']} is {r['diff']:.0f} degrees off")
            lines.append("Give an encouraging refinement cue.")
        elif perfects:
            lines.append("All joints perfect.")
            lines.append("Give a short hype/encouragement cue.")
    else:
        lines.append("No keyframe data yet.")
        lines.append("Give a general encouragement cue.")

    if last_cue:
        lines.append(f'Do NOT repeat: "{last_cue}"')

    lines.append("ONE phrase, max 10 words, sound like a real bboy coach.")
    return "\n".join(lines)


class AICoach:
    """Generates real-time coaching cues using Claude Haiku."""

    def __init__(self) -> None:
        self._last_called: float  = 0.0
        self._last_cue: Optional[str] = None
        self._pending: bool       = False
        self._available: bool     = bool(ANTHROPIC_API_KEY)

        if not self._available:
            print("AICoach: ANTHROPIC_API_KEY not set — AI coaching disabled")
            print("  Add ANTHROPIC_API_KEY=sk-ant-... to your .env file")
        else:
            print(f"AICoach: ready — model={MODEL} cooldown={COOLDOWN_SECONDS}s")

    def can_fire_now(self) -> bool:
        """For event-driven cues (like a finished freeze) that skip the cooldown."""
        return self._available and not self._pending

    def should_fire(self, aligned: bool = False, fill_ratio: float = 0) -> bool:
        if not self._available or self._pending:
            return False
        cooldown = COOLDOWN_SECONDS if (aligned and fill_ratio >= 1.0) else COOLDOWN_NO_BODY
        return time.time() - self._last_called >= cooldown

    async def get_cue(
        self,
        move_name: str,
        aligned: bool,
        fill_ratio: float,
        score_result: Optional[dict],
        freeze: Optional[dict] = None,
    ) -> Optional[str]:
        """Call Claude Haiku and return a coaching cue string, or None on failure."""
        if not self._available:
            return None

        self._pending    = True
        self._last_called = time.time()

        if freeze:
            prompt = _build_freeze_prompt(freeze, self._last_cue)
        else:
            prompt = _build_prompt(
                move_name   = move_name,
                aligned     = aligned,
                fill_ratio  = fill_ratio,
                score_result= score_result,
                last_cue    = self._last_cue,
            )

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
                cue  = data["content"][0]["text"].strip().strip('"')

                # sanity check — ignore if too long or empty
                if not cue or len(cue.split()) > 14:
                    return None

                self._last_cue = cue
                print(f"AICoach: '{cue}'")
                return cue

        except httpx.TimeoutException:
            print("AICoach: timeout — skipping this cue")
            return None
        except Exception as e:
            print(f"AICoach: error — {e}")
            return None
        finally:
            self._pending = False