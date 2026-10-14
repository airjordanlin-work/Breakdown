"""Tests for keyframe scoring, combo logic, and session stats."""

import json

import numpy as np
import pytest
from pathlib import Path

from app.dtw_engine import DTWResult
from app.pose_estimator import PoseFrame
from app.scorer import (
    CLOSE,
    CLOSE_POINTS,
    MISS,
    MISS_POINTS,
    PERFECT,
    PERFECT_POINTS,
    Scorer,
    SessionStats,
    _score_tier,
    joint_angle,
)


def _make_meta(tmp_path: Path, *, frame: int = 5, target_angle: float = 90.0, threshold: float = 5.0) -> Path:
    """Write minimal windmill metadata for scorer tests."""
    meta = {
        "name": "windmill",
        "keyframes": [
            {
                "frame": frame,
                "joints": [
                    {
                        "joint_name": "left_elbow",
                        "joint_triplet": [11, 13, 15],
                        "target_angle": target_angle,
                        "threshold": threshold,
                    }
                ],
            }
        ],
    }
    with (tmp_path / "windmill_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f)
    return tmp_path


def _make_pose_90_elbow() -> PoseFrame:
    """Landmarks with ~90° at left elbow."""
    landmarks = np.zeros((33, 3), dtype=np.float32)
    landmarks[11] = [1.0, 0.0, 0.0]
    landmarks[13] = [0.0, 0.0, 0.0]
    landmarks[15] = [0.0, 1.0, 0.0]
    visibility = np.ones(33, dtype=np.float32)
    return PoseFrame(landmarks=landmarks, visibility=visibility)


def _make_pose_180_elbow() -> PoseFrame:
    """Landmarks with ~180° at left elbow (miss vs 90° target)."""
    landmarks = np.zeros((33, 3), dtype=np.float32)
    landmarks[11] = [1.0, 0.0, 0.0]
    landmarks[13] = [0.0, 0.0, 0.0]
    landmarks[15] = [-1.0, 0.0, 0.0]
    visibility = np.ones(33, dtype=np.float32)
    return PoseFrame(landmarks=landmarks, visibility=visibility)


def _make_dtw_result(
    *,
    aligned: bool = True,
    live_frames: int = 10,
    live_index: int = 9,
    ref_index: int = 5,
    move_name: str = "windmill",
) -> DTWResult:
    """DTWResult with warping path that maps the live tail frame to a keyframe."""
    live_path = tuple([live_index] * 12)
    ref_path = tuple([ref_index] * 12)
    return DTWResult(
        aligned=aligned,
        distance=1.0,
        normalized_distance=0.1,
        move_name=move_name,
        threshold=100.0,
        live_frames=live_frames,
        reference_frames=10,
        live_path=live_path,
        reference_path=ref_path,
    )


def test_joint_angle_right_angle() -> None:
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 0.0])
    c = np.array([0.0, 1.0, 0.0])
    assert joint_angle(a, b, c) == pytest.approx(90.0, abs=0.1)


def test_joint_angle_straight_line() -> None:
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 0.0])
    c = np.array([-1.0, 0.0, 0.0])
    assert joint_angle(a, b, c) == pytest.approx(180.0, abs=0.1)


def test_score_tier_perfect() -> None:
    tier, points = _score_tier(5.0, 10.0)
    assert tier == PERFECT
    assert points == PERFECT_POINTS


def test_score_tier_close() -> None:
    tier, points = _score_tier(15.0, 10.0)
    assert tier == CLOSE
    assert points == CLOSE_POINTS


def test_score_tier_miss() -> None:
    tier, points = _score_tier(25.0, 10.0)
    assert tier == MISS
    assert points == MISS_POINTS


def test_combo_multiplier_thresholds(tmp_path: Path) -> None:
    scorer = Scorer(tmp_path)
    scorer.streak = 0
    assert scorer._combo_multiplier() == 1.0
    scorer.streak = 3
    assert scorer._combo_multiplier() == 1.5
    scorer.streak = 5
    assert scorer._combo_multiplier() == 2.0


def test_update_streak_rules(tmp_path: Path) -> None:
    scorer = Scorer(tmp_path)
    scorer._update_streak(PERFECT)
    assert scorer.streak == 1
    scorer._update_streak(CLOSE)
    assert scorer.streak == 1
    scorer._update_streak(MISS)
    assert scorer.streak == 0


def test_grade_thresholds(tmp_path: Path) -> None:
    scorer = Scorer(tmp_path)
    assert scorer.grade() == "D"

    scorer.total_attempts = 10
    scorer.perfects = 9
    assert scorer.grade() == "S"

    scorer.perfects = 8
    assert scorer.grade() == "A"

    scorer.perfects = 6
    assert scorer.grade() == "B"

    scorer.perfects = 4
    assert scorer.grade() == "C"

    scorer.perfects = 3
    assert scorer.grade() == "D"


def test_stats_and_reset(tmp_path: Path) -> None:
    scorer = Scorer(tmp_path)
    stats = scorer.stats()
    assert isinstance(stats, SessionStats)
    assert stats.total_points == 0.0
    assert stats.total_attempts == 0

    scorer.total_points = 200.0
    scorer.total_attempts = 2
    scorer.perfects = 1
    scorer.closes = 1
    scorer.misses = 0
    scorer.streak = 2
    scorer.reset()

    stats = scorer.stats()
    assert stats.total_points == 0.0
    assert stats.total_attempts == 0
    assert stats.perfects == 0
    assert stats.closes == 0
    assert stats.misses == 0
    assert stats.streak == 0


def test_score_returns_none_when_not_aligned(tmp_path: Path) -> None:
    _make_meta(tmp_path)
    scorer = Scorer(tmp_path)
    dtw = _make_dtw_result(aligned=False)
    result = scorer.score(dtw, _make_pose_90_elbow())
    assert result is None
    assert scorer.total_attempts == 0


def test_score_returns_none_without_keyframes(tmp_path: Path) -> None:
    scorer = Scorer(tmp_path)
    dtw = _make_dtw_result()
    result = scorer.score(dtw, _make_pose_90_elbow())
    assert result is None


def test_score_perfect_awards_points(tmp_path: Path) -> None:
    _make_meta(tmp_path, target_angle=90.0, threshold=5.0)
    scorer = Scorer(tmp_path)
    result = scorer.score(_make_dtw_result(), _make_pose_90_elbow())

    assert result is not None
    assert result.move_name == "windmill"
    assert len(result.results) == 1
    assert result.results[0].tier == PERFECT
    assert result.points_this_attempt == pytest.approx(100.0)
    assert scorer.perfects == 1
    assert scorer.streak == 1


def test_score_miss_resets_streak(tmp_path: Path) -> None:
    _make_meta(tmp_path, target_angle=90.0, threshold=1.0)
    scorer = Scorer(tmp_path)
    scorer.streak = 4

    result = scorer.score(_make_dtw_result(), _make_pose_180_elbow())
    assert result is not None
    assert result.results[0].tier == MISS
    assert scorer.streak == 0
    assert scorer.misses == 1


def test_score_combo_on_third_perfect(tmp_path: Path) -> None:
    _make_meta(tmp_path, target_angle=90.0, threshold=5.0)
    scorer = Scorer(tmp_path)
    dtw = _make_dtw_result()
    pose = _make_pose_90_elbow()

    r1 = scorer.score(dtw, pose)
    r2 = scorer.score(dtw, pose)
    r3 = scorer.score(dtw, pose)

    assert r1 is not None and r2 is not None and r3 is not None
    assert r1.points_this_attempt == pytest.approx(100.0)
    assert r2.points_this_attempt == pytest.approx(100.0)  # streak 2 → still 1x
    assert r3.points_this_attempt == pytest.approx(150.0)  # streak 3 → 1.5x
    assert r3.combo_multiplier_applied == pytest.approx(1.5)
    assert scorer.streak == 3


# ------------------------------------------------------------ reps

from app.dtw_engine import compare_to_move, load_reference_library
from app.scorer import KeyframeResult

REF_DIR = Path(__file__).resolve().parents[1] / "reference_moves"


def _br01():
    return next(m for m in load_reference_library(REF_DIR) if m.name == "br_01")


def test_reference_dancer_scores_a_clean_rep():
    """Doing the move exactly like the reference must score every joint Perfect."""
    move = _br01()
    window = list(move.sequence.reshape(-1, 33, 3))
    dtw = compare_to_move(move.sequence, move)
    result = Scorer(REF_DIR).score_rep(dtw, window)
    assert result is not None
    # br_01's meta also asks for the whole-rep hip twist and leg reach
    assert {r.joint_name for r in result.results} == {
        "left_elbow", "right_elbow", "left_knee", "right_knee", "hips", "leg"}
    assert all(r.tier == PERFECT and r.diff < 1e-3 for r in result.results)


def test_targets_come_from_the_reference_not_a_constant():
    move = _br01()
    dtw = compare_to_move(move.sequence, move)
    result = Scorer(REF_DIR).score_rep(dtw, list(move.sequence.reshape(-1, 33, 3)))
    targets = {r.joint_name: r.target_angle for r in result.results}
    assert targets["left_elbow"] != pytest.approx(160.0, abs=1.0)


def test_bent_elbows_are_flagged_with_a_direction():
    move = _br01()
    window = [f.copy() for f in move.sequence.reshape(-1, 33, 3)]
    for f in window:   # fold the left wrist back onto the shoulder: elbow fully bent
        f[15] = f[11]
    dtw = compare_to_move(move.sequence, move)
    scorer = Scorer(REF_DIR)
    result = scorer.score_rep(dtw, window)
    left = next(r for r in result.results if r.joint_name == "left_elbow")
    assert left.tier == MISS
    assert left.actual_angle < left.target_angle
    assert scorer.stats().focus_joint == "left_elbow"
    assert scorer.stats().focus_fix == "straighten"


def _rep(tier):
    return [KeyframeResult("left_knee", 100.0, 120.0, 20.0, tier, 0.0)]


def test_clean_rep_streaks(tmp_path: Path):
    scorer = Scorer(tmp_path)
    for tier in (PERFECT, CLOSE, PERFECT, MISS, PERFECT):
        scorer._record(_rep(tier), "br_01")
    st = scorer.stats()
    assert st.reps == 5
    assert st.clean_reps == 4
    assert st.best_streak == 3
    assert st.clean_streak == 1
    assert st.focus_joint == "left_knee" and st.focus_fix == "straighten"
    scorer.reset()
    assert scorer.stats().reps == 0 and scorer.stats().best_streak == 0


def test_not_aligned_rep_is_not_scored():
    move = _br01()
    dtw = compare_to_move(move.sequence, move)
    from dataclasses import replace
    assert Scorer(REF_DIR).score_rep(replace(dtw, aligned=False), []) is None
