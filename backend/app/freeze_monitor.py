"""Live freeze detection for one coaching session, fusing IMU and camera.

The IMU decides WHEN you're holding still and HOW steady you are.
The camera can veto: if it clearly saw you standing upright for most of the
hold, that was a pause, not a freeze.

The camera only gets a veto when it can actually see. During real freezes
the camera often loses the torso (limbs overlap, body inverted), which is
the whole reason the IMUs exist. So "camera can't tell" never blocks a
freeze; only "camera clearly saw you standing" does.
"""

from __future__ import annotations

import time
from typing import Optional

from app.imu_metrics import FreezeResult, FreezeTracker
from app.imu_store import ImuStore
from app import posture

UPRIGHT_VETO_RATIO = 0.6   # veto if >60% of confident camera frames saw upright


class FreezeMonitor:
    def __init__(self, store: ImuStore, tracker: Optional[FreezeTracker] = None) -> None:
        self.store = store
        self.tracker = tracker or FreezeTracker()
        self._cursor: Optional[float] = None
        self._upright = 0
        self._down = 0
        self.completed = 0
        self.vetoed = 0
        self.last_posture = posture.UNKNOWN

    def _reset_votes(self) -> None:
        self._upright = self._down = 0

    def _looks_upright(self) -> bool:
        confident = self._upright + self._down
        return bool(confident) and self._upright / confident > UPRIGHT_VETO_RATIO

    def on_frame(self, raw_landmarks=None, visibility=None,
                 now: Optional[float] = None) -> dict:
        """Call once per camera frame. Returns the freeze payload for the UI."""
        now = time.time() if now is None else now
        if self._cursor is None:
            self._cursor = now - 0.5   # don't replay stale buffer on session start

        new = self.store.since(self._cursor)
        if new:
            self._cursor = new[-1].host_time
        result = self.tracker.update(new)

        confirmed: Optional[FreezeResult] = None
        if result is not None:
            if self._looks_upright():
                self.vetoed += 1           # camera saw a standing pause
            else:
                confirmed = result
                self.completed += 1
            self._reset_votes()

        vote = posture.classify(raw_landmarks, visibility)
        self.last_posture = vote
        if self.tracker.holding:
            if vote == posture.UPRIGHT:
                self._upright += 1
            elif vote == posture.DOWN:
                self._down += 1
        elif result is None:
            self._reset_votes()

        # Hide the live meter while the camera says you're just standing.
        live = None if self._looks_upright() else self.tracker.current()
        return {
            "live": live,
            "result": confirmed.to_dict() if confirmed else None,
            "count": self.completed,
            # debug: what the camera thinks right now, and how many holds it blocked
            "posture": vote,
            "holding": self.tracker.holding,
            "vetoed": self.vetoed,
        }
