"""Tests for the camera posture check used to confirm freezes."""

import numpy as np

from app import posture


def body(shoulder_xy, hip_xy, vis=1.0):
    raw = np.zeros((33, 3), dtype=np.float32)
    for i in (11, 12):
        raw[i, :2] = shoulder_xy
    for i in (23, 24):
        raw[i, :2] = hip_xy
    return raw, np.full(33, vis, dtype=np.float32)


def test_standing_is_upright():
    # image y grows downward, so shoulders above hips = smaller y
    assert posture.classify(*body((0.5, 0.3), (0.5, 0.6))) == posture.UPRIGHT


def test_sideways_is_down():
    assert posture.classify(*body((0.3, 0.7), (0.6, 0.7))) == posture.DOWN


def test_inverted_is_down():
    assert posture.classify(*body((0.5, 0.8), (0.5, 0.5))) == posture.DOWN


def test_angle_ignores_distance_from_camera():
    near = posture.torso_angle_deg(body((0.5, 0.1), (0.5, 0.9))[0])
    far = posture.torso_angle_deg(body((0.5, 0.45), (0.5, 0.55))[0])
    assert abs(near - far) < 1e-6


def test_low_visibility_is_unknown():
    assert posture.classify(*body((0.5, 0.3), (0.5, 0.6), vis=0.2)) == posture.UNKNOWN


def test_missing_pose_is_unknown():
    assert posture.classify(None, None) == posture.UNKNOWN
