"""Grounding guard for coach output (spec 3.6).

The coach may only state body or score facts that are in the ``facts`` dict
it was given for that turn. ``ungrounded_claims`` is a cheap heuristic run on
every model reply before it is spoken. Any flag means the reply is dropped
and a templated line built from the same facts is used instead.

What it checks:
- body parts: every body part named must appear in the facts, with the same
  side when a side is given ("left elbow" is refused when only the right
  elbow was measured). "arm" and "leg" count as grounded when a joint on that
  limb was measured.
- numbers: digits and the words two to twenty must equal a number in the
  facts. "one" is skipped because it is mostly a pronoun ("that's the one").
- banned claims: angles, degrees, points, grades, percentages, and any
  comparison to earlier sessions, since nothing from past sessions is stored.

It is a heuristic, not proof. It errs on the side of refusing.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable

_BODY_PARTS = (
    "elbow", "knee", "hip", "shoulder", "wrist", "ankle", "arm", "leg", "hand",
    "foot", "feet", "back", "head", "chest", "core", "neck", "spine", "heel",
    "toe", "glute", "torso", "waist", "thigh", "shin", "body",
)
_BODY_RE = re.compile(
    r"\b(?:(left|right)\s+)?(" + "|".join(_BODY_PARTS) + r")s?\b", re.IGNORECASE
)
# A generic limb word is grounded when a joint on that limb was measured.
_LIMB_OF = {
    "arm":  ("elbow", "shoulder", "wrist", "arm"),
    "hand": ("wrist", "hand"),
    "leg":  ("knee", "hip", "ankle", "leg"),
    "foot": ("ankle", "foot"),
    "feet": ("ankle", "foot", "feet"),
}
# "body" is too generic to be a claim on its own ("keep your body loose").
_GENERIC = {"body"}

_NUMBER_WORDS = {
    "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
}
_NUMBER_RE = re.compile(
    r"\b(\d+(?:\.\d+)?|" + "|".join(_NUMBER_WORDS) + r")\b", re.IGNORECASE
)

_BANNED = re.compile(
    r"\b(degrees?|angles?|percent|points?|grades?|calories|"
    r"last (?:time|session|week|night)|yesterday|than before|"
    r"improv\w*|personal best)\b|%|°",
    re.IGNORECASE,
)


def _walk(value: Any) -> Iterable[Any]:
    if isinstance(value, dict):
        for v in value.values():
            yield from _walk(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _walk(v)
    else:
        yield value


def _allowed_numbers(facts: dict) -> set[float]:
    nums: set[float] = set()
    for v in _walk(facts):
        if isinstance(v, bool):
            continue
        if isinstance(v, int):
            nums.add(float(v))
        elif isinstance(v, float):
            nums.add(round(v, 1))
            nums.add(float(round(v)))
    return nums


def _facts_text(facts: dict) -> str:
    """Every string in the facts, lowercased, underscores as spaces."""
    text = json.dumps(facts, ensure_ascii=False).lower().replace("_", " ")
    return text


def ungrounded_claims(text: str, facts: dict) -> list[str]:
    """Return a list of reasons ``text`` says something the facts do not support.

    An empty list means the text passed.
    """
    flags: list[str] = []
    said = text or ""
    # The move name may contain digits ("BR 01"); saying it is not a claim.
    move = facts.get("move")
    if isinstance(move, str) and move:
        said = re.sub(re.escape(move), " ", said, flags=re.IGNORECASE)

    for m in _BANNED.finditer(said):
        flags.append(f"banned claim: {m.group(0)!r}")

    allowed_text = _facts_text(facts)

    def in_facts(phrase: str) -> bool:
        # whole words only, so "hip" is not grounded by "ship" or "arm" by "warm"
        return re.search(rf"\b{re.escape(phrase)}s?\b", allowed_text) is not None

    for m in _BODY_RE.finditer(said):
        side, part = (m.group(1) or "").lower(), m.group(2).lower()
        if part in _GENERIC:
            continue
        candidates = _LIMB_OF.get(part, (part,))
        if side:
            if not in_facts(f"{side} {part}") and not any(in_facts(f"{side} {j}") for j in candidates):
                flags.append(f"body part not in facts: {m.group(0)!r}")
        elif not any(in_facts(p) for p in candidates):
            flags.append(f"body part not in facts: {m.group(0)!r}")

    allowed = _allowed_numbers(facts)
    for m in _NUMBER_RE.finditer(said):
        token = m.group(1).lower()
        value = float(_NUMBER_WORDS[token]) if token in _NUMBER_WORDS else float(token)
        if value not in allowed and round(value, 1) not in allowed:
            flags.append(f"number not in facts: {m.group(0)!r}")
    return flags
