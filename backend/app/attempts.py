"""Attempts: a stretch of moving with scored reps, ended by a pause.

The debrief (spec 3.1, FR-14) runs once per attempt. An attempt starts with
the first scored rep after the last debrief and ends when the dancer has not
been moving for DEBRIEF_PAUSE_S (standing still or out of frame both count).

``attempt_facts`` turns the attempt's reps into the facts block the debrief
is allowed to talk about. Every value is a direct tally of scorer results.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

# Starting value, not measured. Tune with a device test.
DEBRIEF_PAUSE_S = 2.0


def rep_is_clean(rep: dict) -> bool:
    return all(r.get("tier") != "Miss" for r in rep.get("results", []))


class AttemptTracker:
    """Collects scored reps and says when the dancer has stopped."""

    def __init__(self, pause_s: float = DEBRIEF_PAUSE_S) -> None:
        self.pause_s = pause_s
        self.reps: list[dict] = []
        self._last_moving_at: Optional[float] = None

    def add_rep(self, rep: dict) -> None:
        """``rep`` is the frame payload's score_result: {"results": [{joint, tier, fix}, ...]}."""
        self.reps.append(rep)

    def on_frame(self, moving: bool, t: float) -> Optional[list[dict]]:
        """Call once per frame. Returns the finished attempt's reps once, when it ends."""
        if moving:
            self._last_moving_at = t
            return None
        if not self.reps or self._last_moving_at is None:
            return None
        if t - self._last_moving_at < self.pause_s:
            return None
        reps, self.reps = self.reps, []
        return reps

    def reset(self) -> None:
        self.reps = []
        self._last_moving_at = None


def attempt_facts(move_name: str, reps: list[dict], session: Optional[dict] = None) -> dict:
    """Facts block for a debrief: only counts taken from the attempt's scored reps.

    ``session`` may carry session totals from the scorer (reps, clean_reps,
    best_streak); they are copied as is.
    """
    clean = [rep_is_clean(r) for r in reps]
    best = run = 0
    for ok in clean:
        run = run + 1 if ok else 0
        best = max(best, run)

    off = Counter()
    latest_fix: dict[str, str] = {}
    joints_seen: list[str] = []
    landed_all: set[str] = set()
    for i, rep in enumerate(reps):
        names = set()
        for r in rep.get("results", []):
            name = r.get("joint")
            if not name:
                continue
            names.add(name)
            if name not in joints_seen:
                joints_seen.append(name)
            if r.get("tier") != "Perfect":
                off[name] += 1
                latest_fix[name] = r.get("fix") or "straighten"
        landed_all = names if i == 0 else landed_all & names

    off_most = None
    if off:
        joint, times = max(off.items(), key=lambda kv: (kv[1], -joints_seen.index(kv[0])))
        off_most = {"joint": joint, "fix": latest_fix[joint], "times_off": times}

    facts = {
        "move": move_name,
        "reps": len(reps),
        "clean_reps": sum(clean),
        "best_streak": best,
        "off_most": off_most,
        "landed_every_rep": [j for j in joints_seen if j in landed_all and off[j] == 0],
    }
    if session:
        facts["session"] = {k: session[k] for k in ("reps", "clean_reps", "best_streak") if k in session}
    return facts


def attempt_summary(move_name: str, reps: list[dict]) -> dict:
    """The result card for a finished attempt: what landed, what to fix, and where.

    Built only from the scorer's results (no model). ``fix.checkpoint`` is the
    keyframe checkpoint where the focus joint was off most often, as "k of n"
    through the move; it is omitted for whole-rep features, which have none.
    """
    facts = attempt_facts(move_name, reps)
    fix = None
    off = facts["off_most"]
    if off:
        fix = {"joint": off["joint"], "fix": off["fix"], "times_off": off["times_off"]}
        where = Counter()
        of = 0
        for rep in reps:
            for m in rep.get("moments", []):
                if m.get("joint") == off["joint"]:
                    where[m["checkpoint"]] += 1
                    of = m.get("of", of)
        if where:
            checkpoint, _ = max(where.items(), key=lambda kv: (kv[1], -kv[0]))
            fix["checkpoint"] = checkpoint
            fix["of"] = of
    return {
        "type": "rep_summary",
        "move": move_name,
        "reps": facts["reps"],
        "clean_reps": facts["clean_reps"],
        "best_streak": facts["best_streak"],
        "went_well": facts["landed_every_rep"],
        "fix": fix,
    }
