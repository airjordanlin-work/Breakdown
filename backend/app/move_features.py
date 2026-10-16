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
    """How far the hips twist against the shoulders during the rep, 0 to 1.

    As the hips turn away from the camera their apparent width shrinks. The
    live window is rescaled every frame by hip width, which would divide that
    signal away, so each frame measures hip width as a share of shoulder
    width. A ratio does not change under uniform rescaling, so the live and
    reference paths agree. 1 - narrow / wide over the rep, with percentiles to
    drop single noisy frames.
    """
    hip_w = np.abs(seq[:, L_HIP, 0] - seq[:, R_HIP, 0])
    sh_w = np.abs(seq[:, L_SH, 0] - seq[:, R_SH, 0])
    ok = sh_w > 1e-6
    if ok.sum() < 2:
        return 0.0
    share = hip_w[ok] / sh_w[ok]
    wide = float(np.percentile(share, 90))
    if wide <= 0:
        return 0.0
    return float(1.0 - np.percentile(share, 10) / wide)


def leg_reach(seq: np.ndarray) -> float:
    """How far a foot gets out to the side of the hips, in torso lengths.

    Measured per frame against that frame's own torso length, so per-frame
    rescaling of the live window cancels out.
    """
    mid_hip = (seq[:, L_HIP, :2] + seq[:, R_HIP, :2]) / 2
    mid_sh = (seq[:, L_SH, :2] + seq[:, R_SH, :2]) / 2
    torso = np.linalg.norm(mid_sh - mid_hip, axis=1)
    ok = torso > 1e-6
    if not ok.any():
        return 0.0
    out = np.abs(seq[:, [L_AN, R_AN], 0] - mid_hip[:, None, 0]).max(axis=1)
    return float(np.percentile(out[ok] / torso[ok], 90))


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
