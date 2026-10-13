"""Motion detection that keeps the coach quiet while the dancer is idle."""

import numpy as np

from app.activity import MOTION_SPAN, MOVING_THRESHOLD, is_moving, motion_level


def _pose(offset: float = 0.0) -> np.ndarray:
    base = np.linspace(-1, 1, 33 * 3, dtype=np.float32).reshape(33, 3)
    return base + offset


def test_too_few_frames_is_not_moving():
    frames = [_pose(i * 0.5) for i in range(MOTION_SPAN - 1)]
    assert motion_level(frames) is None
    assert not is_moving(frames)


def test_still_pose_with_small_jitter_is_not_moving():
    rng = np.random.default_rng(0)
    frames = [_pose() + rng.normal(0, 0.01, (33, 3)).astype(np.float32) for _ in range(30)]
    assert motion_level(frames) < MOVING_THRESHOLD
    assert not is_moving(frames)


def test_dancing_pose_is_moving():
    frames = [_pose(0.1 * np.sin(i / 2)) for i in range(30)]
    assert is_moving(frames)


def test_only_recent_frames_count():
    # moved a lot earlier, then stood still for the whole span
    frames = [_pose(i * 0.3) for i in range(20)] + [_pose(10.0)] * MOTION_SPAN
    assert not is_moving(frames)
