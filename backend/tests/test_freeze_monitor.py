"""Tests for FreezeMonitor: IMU detects the hold, camera can veto standing pauses."""

import numpy as np

from app.freeze_monitor import FreezeMonitor
from app.imu_protocol import decode_packet, encode_packet
from app.imu_store import ImuStore

DT = 0.02
MOVING = (200.0, 50.0, 0.0)
STILL = (1.0, -0.5, 1.5)


def add_imu(store, start, seconds, gyro_dps):
    raw = [0, 0, 4096] + [round(v * 16.4) for v in gyro_dps]
    for i in range(int(seconds / DT)):
        store.add(start + i * DT, decode_packet(encode_packet(i, 0, raw, raw)))


def body(shoulder_xy, hip_xy):
    raw = np.zeros((33, 3), dtype=np.float32)
    raw[[11, 12], :2] = shoulder_xy
    raw[[23, 24], :2] = hip_xy
    return raw, np.ones(33, dtype=np.float32)


STANDING = body((0.5, 0.3), (0.5, 0.6))
INVERTED = body((0.5, 0.8), (0.5, 0.5))
BLIND = (None, None)


def play(pose_during_hold):
    """Move 1s, hold 1.5s, move 1s, with a camera frame every 0.1s."""
    store, mon, results, t = ImuStore(), None, [], 100.0
    mon = FreezeMonitor(store)
    for seconds, gyro, pose in [(1.0, MOVING, BLIND), (1.5, STILL, pose_during_hold),
                                (1.0, MOVING, BLIND)]:
        add_imu(store, t, seconds, gyro)
        for k in range(1, int(seconds / 0.1) + 1):
            payload = mon.on_frame(*pose, now=t + k * 0.1)
            if payload["result"]:
                results.append(payload["result"])
        t += seconds
    return mon, results


def test_inverted_hold_counts_as_freeze():
    _, results = play(INVERTED)
    assert len(results) == 1
    assert results[0]["tier"] == "rock solid"


def test_standing_pause_is_vetoed_by_camera():
    mon, results = play(STANDING)
    assert results == []
    assert mon.vetoed == 1


def test_camera_blind_during_hold_still_counts():
    # Real freezes often hide the torso from the camera. Blind is not a veto.
    _, results = play(BLIND)
    assert len(results) == 1


def test_live_meter_hidden_while_standing():
    store = ImuStore()
    mon = FreezeMonitor(store)
    add_imu(store, 100, 1, MOVING)
    add_imu(store, 101, 1, STILL)
    mon.on_frame(*STANDING, now=100.5)
    assert mon.on_frame(*STANDING, now=102.0)["live"] is None


def test_debug_fields_explain_a_veto():
    mon, _ = play(STANDING)
    payload = mon.on_frame(*STANDING, now=200.0)
    assert payload["posture"] == "upright"
    assert payload["vetoed"] == 1
