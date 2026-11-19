"""Per-move coaching knowledge (docs/COACHING_KNOWLEDGE.md).

A small markdown file per move in ``backend/knowledge/`` holds what a good
coach would know about the move: key points, common mistakes, fixes, cue
wording, priorities. It is loaded as a REFERENCE block for the debrief and
chat prompts so the coach can explain why a fix matters and word it well.

It is guidance, not data. It must never supply a number or a fact about this
dancer's body: those come only from the facts block (CLAUDE.md rule 1). Two
things enforce that:
- ``lint`` rejects files with digits or measurement words, and a test runs it
  over every file.
- Replies still pass ``grounding.ungrounded_claims`` against the facts, so a
  reply that leans on the reference to name an unmeasured joint is refused and
  the templated line is used.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "knowledge"
SECTIONS = ("Key points", "Common mistakes", "Fixes", "Cue wording", "Priorities")
MAX_BLOCK_CHARS = 1500   # keeps the extra prompt tokens small

_BANNED = re.compile(r"\d|\b(degrees?|angles?|percent|score[sd]?)\b|%|°", re.IGNORECASE)


def _parse(text: str) -> dict:
    """Front matter (move, status) plus the bullet lists under each ## heading."""
    meta: dict = {}
    body = text
    if text.startswith("---"):
        _, _, rest = text.partition("---")
        head, _, body = rest.partition("---")
        for line in head.strip().splitlines():
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()
    sections: dict[str, list[str]] = {}
    current: Optional[str] = None
    for line in body.splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
            sections[current] = []
        elif current and line.startswith("- "):
            sections[current].append(line[2:].strip())
    return {"move": meta.get("move", ""), "status": meta.get("status", ""), "sections": sections}


def load_all(directory: Path = KNOWLEDGE_DIR) -> dict[str, dict]:
    """Every knowledge file, keyed by lowercase move name."""
    found: dict[str, dict] = {}
    if not directory.is_dir():
        return found
    for path in sorted(directory.glob("*.md")):
        entry = _parse(path.read_text(encoding="utf-8"))
        if entry["move"]:
            entry["file"] = path.name
            found[entry["move"].lower()] = entry
    return found


def lint(path: Path) -> list[str]:
    """Problems in one knowledge file. Empty means it is safe to load."""
    text = path.read_text(encoding="utf-8")
    entry = _parse(text)
    problems = []
    if not entry["move"]:
        problems.append("missing 'move:' in front matter")
    for name in SECTIONS:
        if not entry["sections"].get(name):
            problems.append(f"missing or empty section '{name}'")
    body = "\n".join(f"- {b}" for items in entry["sections"].values() for b in items)
    for m in _BANNED.finditer(body):
        problems.append(f"measurement word or digit in a bullet: {m.group(0)!r}")
    return problems


def knowledge_block(move_name: Optional[str], directory: Path = KNOWLEDGE_DIR) -> str:
    """The REFERENCE text for a move, or "" when there is none."""
    entry = load_all(directory).get((move_name or "").lower())
    if not entry:
        return ""
    lines = [
        "REFERENCE (general coaching knowledge for this move, NOT measured on this dancer).",
        "Use it only to explain why a measured fix matters and to word it well. Never state",
        "anything from it as something you saw or measured, and never name a body part that",
        "FACTS does not list.",
    ]
    for name in ("Key points", "Common mistakes", "Fixes", "Cue wording"):
        items = entry["sections"].get(name) or []
        if items:
            lines.append(f"{name}: " + " ".join(items))
    block = "\n".join(lines)
    return block[:MAX_BLOCK_CHARS]
