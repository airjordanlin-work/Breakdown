"""Tests for freeze detection and stability scoring, using synthetic IMU data."""

import math

from app.imu_metrics import FreezeTracker
from app.imu_protocol import decode_packet, encode_packet
from app.imu_store import TimedImuSample

DT = 0.02  # 50Hz
LSB = 16.4


def sample(t, wrist_dps, leg_dps, seq=0):
    raw = lambda dps: [0, 0, 4096] + [round(v * LSB) for v in dps]
    return TimedImuSample(t, decode_packet(encode_packet(seq, 0, raw(wrist_dps), raw(leg_dps))))


def segment(start, seconds, wrist_fn, leg_fn):
    n = int(seconds / DT)
    return [sample(start + i * DT, wrist_fn(i), leg_fn(i), i) for i in range(n)]


MOVING = lambda i: (200.0, 50.0, 0.0)
STILL = lambda i: (1.0, -0.5, 1.5)                        # just sensor offset
WOBBLE = lambda i: (18 * math.sin(i * 0.9), 5.0, 0.0)     # visible shake, under 30 dps


def run(*segments):
    tracker = FreezeTracker()
    results = []
    for seg in segments:
        r = tracker.update(seg)
        if r:
            results.append(r)
    return tracker, results


def test_steady_freeze_after_movement_is_rock_solid():
    _, results = run(segment(0, 1, MOVING, MOVING),
                     segment(1, 1.5, STILL, STILL),
                     segment(2.5, 0.5, MOVING, MOVING))
    assert len(results) == 1
    r = results[0]
    assert r.tier == "rock solid"
    assert r.stability >= 95
    assert 1.4 <= r.duration_s <= 1.5


def test_constant_gyro_offset_is_not_counted_as_wobble():
    offset = lambda i: (12.0, -8.0, 5.0)   # large but constant
    _, results = run(segment(0, 1, MOVING, MOVING),
                     segment(1, 1, offset, offset),
                     segment(2, 0.5, MOVING, MOVING))
    assert results[0].wobble_dps < 0.5


def test_wobbly_freeze_is_graded_down_and_blames_the_right_limb():
    _, results = run(segment(0, 1, MOVING, MOVING),
                     segment(1, 1, STILL, WOBBLE),
                     segment(2, 0.5, MOVING, MOVING))
    r = results[0]
    assert r.tier in ("shaky", "unstable")
    assert r.shakiest == "leg"


def test_standing_still_without_prior_movement_is_ignored():
    _, results = run(segment(0, 3, STILL, STILL),
                     segment(3, 0.5, MOVING, MOVING))
    assert results == []


def test_short_pause_is_not_a_freeze():
    _, results = run(segment(0, 1, MOVING, MOVING),
                     segment(1, 0.3, STILL, STILL),
                     segment(1.3, 0.5, MOVING, MOVING))
    assert results == []


def test_result_only_arrives_when_the_hold_ends():
    tracker, results = run(segment(0, 1, MOVING, MOVING),
                           segment(1, 1, STILL, STILL))
    assert results == []                     # still holding
    live = tracker.current()
    assert live is not None and live["tier"] == "rock solid"


def test_settling_into_the_freeze_is_not_graded_as_wobble():
    # Rotation decays from 29 to ~1 deg/s over the first 0.2s: the body
    # slowing down, not shaking. The freeze itself is dead still.
    settling = lambda i: (max(29.0 - i * 3.0, 1.0), -0.5, 1.5)
    _, results = run(segment(0, 1, MOVING, MOVING),
                     segment(1, 1.5, settling, STILL),
                     segment(2.5, 0.5, MOVING, MOVING))
    assert results[0].tier == "rock solid"