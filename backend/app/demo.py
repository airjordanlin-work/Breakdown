"""Slow demo of the selected move, on request (spec 3.1 "show me").

When the dancer asks for a demo, the client plays the move's reference
sequence as a slow animated figure (the same frames as the move library
preview) and the coach says one line about what to watch. The joint to watch
is the session's measured focus joint, so the spoken line and the highlight
only name a body part the scorer actually flagged.

The line is templated, not generated: it is said while the dancer is looking
at the screen, and it must name exactly what the highlight shows.
"""

from __future__ import annotations

from typing import Optional

from app.move_catalog import PREVIEW_JOINTS, L_EL, R_EL, L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN

DEMO_SPEED = 0.5    # playback speed of the reference clip
DEMO_LOOPS = 2      # times the clip plays before the coach starts talking again

# Preview joint indices (positions in PREVIEW_JOINTS) to highlight per body part.
_HIGHLIGHT_LANDMARKS = {
    "hips":        [L_HIP, R_HIP],
    "leg":         [L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN],
    "left elbow":  [L_EL],
    "right elbow": [R_EL],
    "left knee":   [L_KN],
    "right knee":  [R_KN],
}

_WATCH = {
    "twist":      "See how far they turn.",
    "reach":      "See how far it goes out.",
    "straighten": "See how straight it gets.",
    "bend":       "See how much it bends.",
}


def highlight_for(body_part: Optional[str]) -> list[int]:
    """Preview joint indices for a body part name, [] if unknown or None."""
    landmarks = _HIGHLIGHT_LANDMARKS.get((body_part or "").replace("_", " ").lower(), [])
    return [PREVIEW_JOINTS.index(i) for i in landmarks]


def demo_facts(move_name: str, focus_joint: Optional[str], focus_fix: Optional[str]) -> dict:
    """What the demo line may talk about: the move, and the measured focus joint if any."""
    facts: dict = {"move": move_name}
    if focus_joint and highlight_for(focus_joint):
        facts["focus"] = {"joint": focus_joint.replace("_", " "), "fix": focus_fix or ""}
    return facts


def demo_line(facts: dict) -> str:
    """One spoken line for the demo, built only from ``facts``."""
    move = facts.get("move") or "the move"
    line = f"Here's {move} at half speed."
    focus = facts.get("focus")
    if not focus:
        return f"{line} Watch the whole move."
    watch = _WATCH.get(focus.get("fix", ""), "")
    return f"{line} Watch the {focus['joint']}. {watch}".strip()


def demo_message(move_id: str, move_name: str, focus_joint: Optional[str],
                 focus_fix: Optional[str], clip_seconds: float) -> tuple[dict, str, float]:
    """(client message, spoken line, seconds the coach stays quiet) for one demo."""
    facts = demo_facts(move_name, focus_joint, focus_fix)
    focus = facts.get("focus") or {}
    message = {
        "type": "demo",
        "move_id": move_id,
        "move_name": move_name,
        "speed": DEMO_SPEED,
        "loops": DEMO_LOOPS,
        "seconds": clip_seconds,
        "focus": focus.get("joint"),
        "highlight": highlight_for(focus.get("joint")),
    }
    quiet_s = clip_seconds / DEMO_SPEED * DEMO_LOOPS
    return message, demo_line(facts), quiet_s
