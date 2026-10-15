"""Is the dancer actually moving? Used to keep the coach quiet while idle.

Motion is measured on the normalized landmarks in the pose buffer (hip-center
origin, torso-width units), over the joints the references contain. For each
joint we take how far it travelled (bounding box diagonal) across the last
few frames, then the median over joints, so one jittery wrist can't make a
still dancer look like they're moving.

MOVING_THRESHOLD is a starting value, not a measured one. In the AIST++
references every 15-frame window of a move is above it, and a held T-pose
mostly sits below it, but live MediaPipe jitter on a phone has not been
measured yet. Tune it with a device test.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from app.dtw_engine import COMPARE_JOINTS

MOTION_SPAN = 15          # recent buffer frames to look at
MOVING_THRESHOLD = 0.15   # median joint travel, in torso widths


def motion_level(frames: Sequence[np.ndarray], span: int = MOTION_SPAN) -> Optional[float]:
    """Median joint travel over the last ``span`` frames, or None if too few frames."""
    if len(frames) < span:
        return None
    window = np.stack([np.asarray(f, dtype=np.float32).reshape(33, 3) for f in list(frames)[-span:]])
    xy = window[:, COMPARE_JOINTS, :2]
    travel = np.linalg.norm(xy.max(axis=0) - xy.min(axis=0), axis=-1)
    return float(np.median(travel))


def is_moving(frames: Sequence[np.ndarray], threshold: float = MOVING_THRESHOLD) -> bool:
    level = motion_level(frames)
    return level is not None and level >= threshold
