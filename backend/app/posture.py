"""Coarse body posture from camera landmarks, used to confirm freezes.

Only answers one question: is the torso clearly upright (standing), or not?
Freezes put the torso sideways or upside down, so a hold where the camera
confidently sees an upright torso is a standing pause, not a freeze.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 11, 12, 23, 24
UPRIGHT_MAX_DEG = 40.0   # torso within 40 degrees of vertical counts as upright
MIN_VISIBILITY = 0.5

UPRIGHT = "upright"
DOWN = "down"            # sideways or inverted: freeze territory
UNKNOWN = "unknown"      # camera can't see the torso well enough to say


def torso_angle_deg(raw_landmarks: np.ndarray) -> float:
    """Angle between the hip-to-shoulder line and straight up, in degrees.

    0 = standing, 90 = lying sideways, 180 = fully inverted. Uses image
    coordinates, where y increases downward. Scale-free: depends only on
    direction, not on distance from the camera or body size.
    """
    shoulders = (raw_landmarks[L_SHOULDER, :2] + raw_landmarks[R_SHOULDER, :2]) / 2
    hips = (raw_landmarks[L_HIP, :2] + raw_landmarks[R_HIP, :2]) / 2
    dx, dy = shoulders - hips
    return math.degrees(math.atan2(abs(dx), -dy))


def classify(raw_landmarks: Optional[np.ndarray], visibility: Optional[np.ndarray]) -> str:
    if raw_landmarks is None or visibility is None:
        return UNKNOWN
    if any(visibility[i] < MIN_VISIBILITY for i in (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)):
        return UNKNOWN
    return UPRIGHT if torso_angle_deg(raw_landmarks) < UPRIGHT_MAX_DEG else DOWN
