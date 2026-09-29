"""Freeze stability from IMU data.

A freeze is a held pose after movement. While it's held, the gyros should
read close to zero; any wobble shows up as small rotations. This module finds
holds in the IMU stream and scores how steady they were.

Why wobble is measured as spread (standard deviation), not raw magnitude:
every gyro has a small constant offset (about 1 to 2 deg/s on these sensors).
Standard deviation ignores a constant offset and only measures how much the
reading moves around, so no calibration step is needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from app.imu_store import TimedImuSample

# Starting thresholds. Tune these with recorded sessions
# (scripts/analyze_recording.py) rather than guessing.
STILL_DPS = 30.0      # below this on every sensor = holding still
MOTION_DPS = 80.0     # above this = real movement (not just a twitch)
MIN_HOLD_S = 0.5      # shorter holds aren't counted as freezes
LOOKBACK_S = 2.0      # a hold only counts if you were moving this recently
SETTLE_S = 0.25       # ignore this much at each end of a hold when scoring:
                      # the body is still slowing down / starting to move

# Wobble (deg/s of spread) to tier boundaries
TIERS = [(4.0, "rock solid"), (8.0, "steady"), (15.0, "shaky")]
UNSTABLE = "unstable"


def _gyro_mag(sample) -> float:
    gx, gy, gz = sample.gyro_dps
    return math.sqrt(gx * gx + gy * gy + gz * gz)


def _spread(values: Sequence[Sequence[float]]) -> float:
    """Combined standard deviation across x, y, z (deg/s)."""
    n = len(values)
    if n < 2:
        return 0.0
    total = 0.0
    for axis in range(3):
        mean = sum(v[axis] for v in values) / n
        total += sum((v[axis] - mean) ** 2 for v in values) / n
    return math.sqrt(total)


def _tier(wobble: float) -> str:
    for limit, name in TIERS:
        if wobble < limit:
            return name
    return UNSTABLE


def _stability(wobble: float) -> int:
    """0 to 100, where 100 means no measurable wobble."""
    return max(0, min(100, round(100 - wobble * 4)))


@dataclass(frozen=True)
class FreezeResult:
    start_time: float
    duration_s: float
    wobble_dps: float
    wrist_wobble_dps: Optional[float]
    leg_wobble_dps: Optional[float]
    stability: int              # 0 to 100
    tier: str                   # rock solid / steady / shaky / unstable
    shakiest: Optional[str]     # "wrist" or "leg", whichever wobbled more

    def to_dict(self) -> dict:
        return {
            "duration_s": round(self.duration_s, 2),
            "wobble_dps": round(self.wobble_dps, 1),
            "wrist_wobble_dps": None if self.wrist_wobble_dps is None else round(self.wrist_wobble_dps, 1),
            "leg_wobble_dps": None if self.leg_wobble_dps is None else round(self.leg_wobble_dps, 1),
            "stability": self.stability,
            "tier": self.tier,
            "shakiest": self.shakiest,
        }


def _trim_edges(samples: Sequence[TimedImuSample], settle_s: float) -> Sequence[TimedImuSample]:
    """Drop the settling-in and settling-out parts of a hold.

    A hold is detected the moment rotation drops under STILL_DPS, but the body
    is still decelerating for a fraction of a second after that. Scoring those
    edges would grade the landing into the freeze, not the freeze itself.
    Falls back to the full hold if trimming would leave too little.
    """
    t0, t1 = samples[0].host_time + settle_s, samples[-1].host_time - settle_s
    core = [s for s in samples if t0 <= s.host_time <= t1]
    return core if len(core) >= 10 else samples


def score_hold(samples: Sequence[TimedImuSample], settle_s: float = SETTLE_S) -> FreezeResult:
    duration = samples[-1].host_time - samples[0].host_time
    start = samples[0].host_time
    samples = _trim_edges(samples, settle_s)
    wrist = [s.packet.wrist.gyro_dps for s in samples if s.packet.wrist]
    leg = [s.packet.leg.gyro_dps for s in samples if s.packet.leg]
    wrist_w = _spread(wrist) if len(wrist) >= 2 else None
    leg_w = _spread(leg) if len(leg) >= 2 else None

    parts = [w for w in (wrist_w, leg_w) if w is not None]
    wobble = max(parts) if parts else 0.0   # the shakiest limb sets the grade

    shakiest = None
    if wrist_w is not None and leg_w is not None:
        shakiest = "wrist" if wrist_w >= leg_w else "leg"
    elif parts:
        shakiest = "wrist" if wrist_w is not None else "leg"

    return FreezeResult(
        start_time=start,
        duration_s=duration,
        wobble_dps=wobble,
        wrist_wobble_dps=wrist_w,
        leg_wobble_dps=leg_w,
        stability=_stability(wobble),
        tier=_tier(wobble),
        shakiest=shakiest,
    )


class FreezeTracker:
    """Feed IMU samples in time order; get a FreezeResult when a hold ends.

    A hold starts when every sensor drops below STILL_DPS shortly after real
    movement, so standing around before you start dancing doesn't count.
    """

    def __init__(self, still_dps=STILL_DPS, motion_dps=MOTION_DPS,
                 min_hold_s=MIN_HOLD_S, lookback_s=LOOKBACK_S) -> None:
        self.still_dps = still_dps
        self.motion_dps = motion_dps
        self.min_hold_s = min_hold_s
        self.lookback_s = lookback_s
        self._hold: list[TimedImuSample] = []
        self._last_motion: Optional[float] = None
        self.last_result: Optional[FreezeResult] = None

    def _peak(self, s: TimedImuSample) -> Optional[float]:
        mags = [_gyro_mag(x) for x in (s.packet.wrist, s.packet.leg) if x is not None]
        return max(mags) if mags else None

    def update(self, samples: Iterable[TimedImuSample]) -> Optional[FreezeResult]:
        """Process new samples. Returns a result if a freeze just finished."""
        finished = None
        for s in samples:
            peak = self._peak(s)
            if peak is None:
                continue  # no sensor data in this packet
            if peak < self.still_dps:
                if self._hold:
                    self._hold.append(s)
                elif self._last_motion is not None and s.host_time - self._last_motion <= self.lookback_s:
                    self._hold.append(s)
            else:
                if peak >= self.motion_dps:
                    self._last_motion = s.host_time
                result = self._end_hold()
                if result:
                    finished = result
        return finished

    def _end_hold(self) -> Optional[FreezeResult]:
        hold, self._hold = self._hold, []
        if len(hold) < 2 or hold[-1].host_time - hold[0].host_time < self.min_hold_s:
            return None
        self.last_result = score_hold(hold)
        return self.last_result

    def current(self) -> Optional[dict]:
        """Live view of a hold in progress, for the UI. None if not holding."""
        if len(self._hold) < 2:
            return None
        live = score_hold(self._hold)
        if live.duration_s < self.min_hold_s:
            return None
        return live.to_dict()