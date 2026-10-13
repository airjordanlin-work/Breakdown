"""Scoring engine — keyframe thresholding, rep tallies, and session stats.

A rep is one move-length window of live motion that DTW aligned to the
reference. score_rep() checks each keyframe on the live frame DTW matched to
it. When a keyframe joint has no target_angle in the meta file, the target is
measured from the reference move itself at that frame, so the check matches
what the reference dancer actually did.

Angles are measured on x and y only, the same plane DTW compares: MediaPipe's
single-camera depth is too rough to compare against measured AIST++ depth.
"""

from __future__ import annotations

import json
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from app.dtw_engine import DTWResult
from app.pose_estimator import PoseFrame

PERFECT = "Perfect"
CLOSE = "Close"
MISS = "Miss"

PERFECT_POINTS = 100.0
CLOSE_POINTS = 40.0
MISS_POINTS = 0.0

DEFAULT_THRESHOLD_DEG = 15.0
DEFAULT_KEYFRAME_OFFSET = 10
FOCUS_REPS = 5   # the focus joint is picked from this many recent reps


@dataclass(frozen=True)
class KeyframeResult:
    """Per-joint score outcome at a keyframe moment."""

    joint_name: str
    actual_angle: float
    target_angle: float
    diff: float
    tier: str
    points_earned: float


@dataclass(frozen=True)
class ScoreResult:
    """Score outcome for one scoring attempt."""

    results: list[KeyframeResult]
    points_this_attempt: float
    combo_multiplier_applied: float
    move_name: str


@dataclass(frozen=True)
class SessionStats:
    """Running totals for the current session."""

    total_points: float
    total_attempts: int
    perfects: int
    closes: int
    misses: int
    streak: int
    grade: str
    # Rep view: plain tallies of scored reps, no points.
    reps: int = 0
    clean_reps: int = 0          # reps with no joint in the Miss tier
    clean_streak: int = 0
    best_streak: int = 0
    focus_joint: Optional[str] = None   # joint most often off in recent reps
    focus_fix: Optional[str] = None     # "straighten" or "bend"


@dataclass(frozen=True)
class _JointSpec:
    """Parsed joint target from reference metadata."""

    joint_name: str
    joint_triplet: tuple[int, int, int]
    target_angle: Optional[float]   # None: measure it from the reference move
    threshold: float


@dataclass(frozen=True)
class _KeyframeSpec:
    """Parsed keyframe entry from reference metadata."""

    frame: int
    joints: list[_JointSpec]


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at vertex b formed by segments ba and bc, in degrees."""
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def _score_tier(diff: float, threshold: float) -> tuple[str, float]:
    """Map absolute angular difference to a tier and base points."""
    if diff <= threshold:
        return PERFECT, PERFECT_POINTS
    if diff <= 2.0 * threshold:
        return CLOSE, CLOSE_POINTS
    return MISS, 0.0


def _parse_triplet(value: Any) -> tuple[int, int, int]:
    """Validate and parse a three-index landmark triplet."""
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"joint_triplet must have exactly 3 indices, got: {value!r}")
    triplet = tuple(int(i) for i in value)
    if any(i < 0 or i > 32 for i in triplet):
        raise ValueError(f"joint_triplet indices must be in [0, 32], got: {triplet}")
    return triplet  # type: ignore[return-value]


def _parse_joint(raw: dict[str, Any]) -> _JointSpec:
    """Parse one joint entry from keyframe metadata."""
    triplet = raw.get("joint_triplet", raw.get("landmarks"))
    target = raw.get("target_angle", raw.get("target_angle_deg"))
    threshold = raw.get("threshold", raw.get("threshold_deg", DEFAULT_THRESHOLD_DEG))
    if triplet is None:
        raise ValueError(f"Joint entry missing joint_triplet: {raw}")
    return _JointSpec(
        joint_name=str(raw.get("joint_name", "joint")),
        joint_triplet=_parse_triplet(triplet),
        target_angle=None if target is None else float(target),
        threshold=float(threshold),
    )


def _load_keyframes(reference_dir: Path, move_name: str) -> list[_KeyframeSpec]:
    """Load keyframe specs from ``{move_name}_meta.json``."""
    meta_path = reference_dir / f"{move_name}_meta.json"
    if not meta_path.exists():
        return []

    with meta_path.open(encoding="utf-8") as f:
        meta = json.load(f)

    raw_keyframes = meta.get("keyframes", [])
    if not isinstance(raw_keyframes, list):
        raise ValueError(f"'keyframes' must be a list in {meta_path}")

    specs: list[_KeyframeSpec] = []
    for entry in raw_keyframes:
        if not isinstance(entry, dict):
            continue
        frame = int(entry.get("frame", -1))
        joints_raw = entry.get("joints", [])
        if frame < 0 or not isinstance(joints_raw, list):
            continue
        joints = [_parse_joint(j) for j in joints_raw if isinstance(j, dict)]
        if joints:
            specs.append(_KeyframeSpec(frame=frame, joints=joints))
    return specs


def _reference_frame_index(dtw_result: DTWResult, live_frame_index: int) -> Optional[int]:
    """Map a live frame index to a reference frame index using the DTW warping path."""
    if not dtw_result.live_path or not dtw_result.reference_path:
        return None
    if len(dtw_result.live_path) != len(dtw_result.reference_path):
        return None

    matches = [
        ref
        for live, ref in zip(dtw_result.live_path, dtw_result.reference_path)
        if live == live_frame_index
    ]
    if not matches:
        return None
    return int(np.median(np.asarray(matches, dtype=np.float32)))


def _live_frame_index(dtw_result: DTWResult, reference_index: int) -> Optional[int]:
    """Live frame DTW matched to a reference frame (median if several), or None."""
    if len(dtw_result.live_path) != len(dtw_result.reference_path):
        return None
    matches = [
        live
        for live, ref in zip(dtw_result.live_path, dtw_result.reference_path)
        if ref == reference_index
    ]
    if not matches:
        return None
    return int(np.median(np.asarray(matches, dtype=np.float32)))


def _angle_xy(landmarks: np.ndarray, triplet: tuple[int, int, int]) -> float:
    i, j, k = triplet
    return joint_angle(landmarks[i, :2], landmarks[j, :2], landmarks[k, :2])


def _match_keyframe(
    keyframes: list[_KeyframeSpec],
    reference_index: Optional[int],
    *,
    max_offset: int = DEFAULT_KEYFRAME_OFFSET,
) -> Optional[_KeyframeSpec]:
    """Return the nearest keyframe if within ``max_offset`` frames of the reference index."""
    if reference_index is None or not keyframes:
        return None
    nearest = min(keyframes, key=lambda k: abs(k.frame - reference_index))
    if abs(nearest.frame - reference_index) > max_offset:
        return None
    return nearest


def _attempt_tier(results: list[KeyframeResult]) -> str:
    """Collapse per-joint tiers into one attempt tier."""
    if not results:
        return MISS
    tiers = {r.tier for r in results}
    if MISS in tiers:
        return MISS
    if tiers == {PERFECT}:
        return PERFECT
    return CLOSE


class Scorer:
    """Score live poses at keyframe moments using DTW alignment and reference metadata."""

    def __init__(self, reference_dir: Path) -> None:
        self.reference_dir = Path(reference_dir)
        self.total_points: float = 0.0
        self.total_attempts: int = 0
        self.perfects: int = 0
        self.closes: int = 0
        self.misses: int = 0
        self.streak: int = 0
        self.reps: int = 0
        self.clean_reps: int = 0
        self.clean_streak: int = 0
        self.best_streak: int = 0
        self._recent_reps: deque[list[KeyframeResult]] = deque(maxlen=FOCUS_REPS)
        self._references: dict[str, Optional[np.ndarray]] = {}

    def _reference(self, move_name: str) -> Optional[np.ndarray]:
        """Reference landmarks (T, 33, 3) for a move, cached; None if missing."""
        if move_name not in self._references:
            path = self.reference_dir / f"{move_name}.npy"
            seq = None
            if path.exists():
                seq = np.load(path).astype(np.float32).reshape(-1, 33, 3)
            self._references[move_name] = seq
        return self._references[move_name]

    def _combo_multiplier(self) -> float:
        """Return combo multiplier from current perfect streak."""
        if self.streak >= 5:
            return 2.0
        if self.streak >= 3:
            return 1.5
        return 1.0

    def _update_streak(self, tier: str) -> None:
        """Perfect increments streak, Miss resets it, Close leaves it unchanged."""
        if tier == PERFECT:
            self.streak += 1
        elif tier == MISS:
            self.streak = 0

    def grade(self) -> str:
        """Return session letter grade from perfect-rate thresholds."""
        if self.total_attempts <= 0:
            return "D"
        perfect_rate = self.perfects / self.total_attempts
        if perfect_rate >= 0.90:
            return "S"
        if perfect_rate >= 0.75:
            return "A"
        if perfect_rate >= 0.60:
            return "B"
        if perfect_rate >= 0.40:
            return "C"
        return "D"

    def focus(self) -> tuple[Optional[str], Optional[str]]:
        """Joint off most often over the last few reps, and which way to fix it."""
        off = [r for rep in self._recent_reps for r in rep if r.tier != PERFECT]
        if not off:
            return None, None
        counts = Counter(r.joint_name for r in off)
        worst = max(counts, key=lambda j: (counts[j], sum(r.diff for r in off if r.joint_name == j)))
        latest = [r for r in off if r.joint_name == worst][-1]
        fix = "straighten" if latest.actual_angle < latest.target_angle else "bend"
        return worst, fix

    def stats(self) -> SessionStats:
        """Return an immutable snapshot of current session totals."""
        focus_joint, focus_fix = self.focus()
        return SessionStats(
            total_points=self.total_points,
            total_attempts=self.total_attempts,
            perfects=self.perfects,
            closes=self.closes,
            misses=self.misses,
            streak=self.streak,
            grade=self.grade(),
            reps=self.reps,
            clean_reps=self.clean_reps,
            clean_streak=self.clean_streak,
            best_streak=self.best_streak,
            focus_joint=focus_joint,
            focus_fix=focus_fix,
        )

    def reset(self) -> None:
        """Clear all session counters and combo streak."""
        self.total_points = 0.0
        self.total_attempts = 0
        self.perfects = 0
        self.closes = 0
        self.misses = 0
        self.streak = 0
        self.reps = 0
        self.clean_reps = 0
        self.clean_streak = 0
        self.best_streak = 0
        self._recent_reps.clear()

    def score_rep(self, dtw_result: DTWResult, window: Sequence[np.ndarray]) -> Optional[ScoreResult]:
        """Score one rep: every keyframe, on the live frame DTW aligned to it.

        ``window`` is the live landmark window DTW compared, (33, 3) per frame.
        Each joint's result is its average over the keyframes, so one rep gives
        one line per joint. None if not aligned or nothing could be checked.
        """
        if not dtw_result.aligned:
            return None
        keyframes = _load_keyframes(self.reference_dir, dtw_result.move_name)
        if not keyframes:
            return None
        reference = self._reference(dtw_result.move_name)

        samples: dict[str, list[tuple[float, float, float]]] = {}
        thresholds: dict[str, float] = {}
        for kf in keyframes:
            live_idx = _live_frame_index(dtw_result, kf.frame)
            if live_idx is None or live_idx >= len(window):
                continue
            live = np.asarray(window[live_idx], dtype=np.float32).reshape(33, 3)
            for spec in kf.joints:
                target = spec.target_angle
                if target is None:
                    if reference is None or kf.frame >= len(reference):
                        continue
                    target = _angle_xy(reference[kf.frame], spec.joint_triplet)
                actual = _angle_xy(live, spec.joint_triplet)
                samples.setdefault(spec.joint_name, []).append((actual, target, actual - target))
                thresholds[spec.joint_name] = spec.threshold

        if not samples:
            return None
        joint_results = []
        for name, vals in samples.items():
            actual = float(np.mean([v[0] for v in vals]))
            target = float(np.mean([v[1] for v in vals]))
            diff = float(np.mean([abs(v[2]) for v in vals]))
            tier, points = _score_tier(diff, thresholds[name])
            joint_results.append(KeyframeResult(name, actual, target, diff, tier, points))
        return self._record(joint_results, dtw_result.move_name)

    def _record(self, joint_results: list[KeyframeResult], move_name: str) -> ScoreResult:
        """Update every tally for one scored attempt and return its result."""
        attempt_tier = _attempt_tier(joint_results)
        self._update_streak(attempt_tier)
        combo = self._combo_multiplier()
        raw_points = float(sum(r.points_earned for r in joint_results))
        scored_points = raw_points * combo

        self.total_attempts += 1
        self.total_points += scored_points
        if attempt_tier == PERFECT:
            self.perfects += 1
        elif attempt_tier == CLOSE:
            self.closes += 1
        else:
            self.misses += 1

        self.reps += 1
        self._recent_reps.append(joint_results)
        if attempt_tier == MISS:
            self.clean_streak = 0
        else:
            self.clean_reps += 1
            self.clean_streak += 1
            self.best_streak = max(self.best_streak, self.clean_streak)

        return ScoreResult(
            results=joint_results,
            points_this_attempt=scored_points,
            combo_multiplier_applied=combo,
            move_name=move_name,
        )

    def score(self, dtw_result: DTWResult, pose_frame: PoseFrame) -> Optional[ScoreResult]:
        """Score the current pose at a matched keyframe; None if not aligned or no keyframes."""
        if not dtw_result.aligned:
            return None

        keyframes = _load_keyframes(self.reference_dir, dtw_result.move_name)
        if not keyframes:
            return None

        live_index = max(dtw_result.live_frames - 1, 0)
        ref_index = _reference_frame_index(dtw_result, live_index)
        matched = _match_keyframe(keyframes, ref_index)
        if matched is None:
            return None

        landmarks = pose_frame.landmarks
        if landmarks.ndim != 2 or landmarks.shape[0] < 33 or landmarks.shape[1] < 2:
            raise ValueError(f"Expected landmarks shape (33, >=2), got {landmarks.shape}")

        reference = self._reference(dtw_result.move_name)
        joint_results: list[KeyframeResult] = []
        for spec in matched.joints:
            i, j, k = spec.joint_triplet
            target = spec.target_angle
            if target is None:
                if reference is None or matched.frame >= len(reference):
                    continue
                target = _angle_xy(reference[matched.frame], spec.joint_triplet)
            actual = joint_angle(landmarks[i], landmarks[j], landmarks[k])
            diff = abs(actual - target)
            tier, points = _score_tier(diff, spec.threshold)
            joint_results.append(
                KeyframeResult(
                    joint_name=spec.joint_name,
                    actual_angle=actual,
                    target_angle=target,
                    diff=diff,
                    tier=tier,
                    points_earned=points,
                )
            )
        if not joint_results:
            return None
        return self._record(joint_results, dtw_result.move_name)
