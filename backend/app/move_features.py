"""Whole-rep body features: how much the hips turn, how far a leg reaches out.

Keyframe joint angles only see elbows and knees at a few moments. A coach also
watches things that play out over the whole rep, like the hip twist in toprock
or a leg kicking out. Each feature here is measured the same way on the live
rep window and on the move's reference sequence, then compared as a ratio
(live / reference), so the target always comes from the reference dancer.

Everything is measured on x and y only, the plane the camera sees and DTW
compares. Features only flag "not enough": doing more than the reference is
not called wrong.

A move opts in by listing features in its meta file:
    "features": ["hip_turn", "leg_reach"]

Tier cut-offs are starting values, not measured. Tune them with a device test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

L_SH, R_SH, L_HIP, R_HIP, L_AN, R_AN = 11, 12, 23, 24, 27, 28

PERFECT_RATIO = 0.75   # live reaches at least this share of the reference: Perfect
CLOSE_RATIO = 0.5      # at least this share: Close; below: Miss
MIN_REFERENCE = 1e-3   # a reference value this small means the move has no such motion


def hip_turn(seq: np.ndarray) -> float:
    """How far the hips twist during the rep, 0 (square the whole time) to 1.

    As the hips turn away from the camera their apparent width shrinks, so
    1 - narrow / wide over the rep tracks the twist without needing depth.
    Percentiles drop single noisy frames.
    """
    width = np.abs(seq[:, L_HIP, 0] - seq[:, R_HIP, 0])
    wide = float(np.percentile(width, 90))
    if wide <= 0:
        return 0.0
    return float(1.0 - np.percentile(width, 10) / wide)


def leg_reach(seq: np.ndarray) -> float:
    """How far a foot gets out to the side of the hips, in torso lengths."""
    mid_hip = (seq[:, L_HIP, :2] + seq[:, R_HIP, :2]) / 2
    mid_sh = (seq[:, L_SH, :2] + seq[:, R_SH, :2]) / 2
    torso = float(np.median(np.linalg.norm(mid_sh - mid_hip, axis=1)))
    if torso <= 0:
        return 0.0
    out = np.abs(seq[:, [L_AN, R_AN], 0] - mid_hip[:, None, 0]).max(axis=1)
    return float(np.percentile(out, 90) / torso)


@dataclass(frozen=True)
class FeatureSpec:
    measure: Callable[[np.ndarray], float]
    body_part: str    # what the coach names, and what the grounding guard checks
    fix: str          # which way to move when the dancer does too little


FEATURES: dict[str, FeatureSpec] = {
    "hip_turn":  FeatureSpec(hip_turn, "hips", "twist"),
    "leg_reach": FeatureSpec(leg_reach, "leg", "reach"),
}


@dataclass(frozen=True)
class FeatureResult:
    name: str
    body_part: str
    fix: str
    live: float
    reference: float
    ratio: float
    tier: str


def tier_for(ratio: float) -> str:
    if ratio >= PERFECT_RATIO:
        return "Perfect"
    if ratio >= CLOSE_RATIO:
        return "Close"
    return "Miss"


def compare(
    names: Sequence[str],
    live: np.ndarray,
    reference: np.ndarray,
) -> list[FeatureResult]:
    """Measure each named feature on both sequences, (T, 33, 3) each.

    Unknown names, and features the reference move barely shows, are skipped.
    """
    results = []
    for name in names:
        spec: Optional[FeatureSpec] = FEATURES.get(name)
        if spec is None:
            continue
        ref = spec.measure(reference)
        if not np.isfinite(ref) or ref < MIN_REFERENCE:
            continue
        got = spec.measure(live)
        if not np.isfinite(got):
            continue
        ratio = got / ref
        results.append(FeatureResult(name, spec.body_part, spec.fix, got, ref, ratio, tier_for(ratio)))
    return results
