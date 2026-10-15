"""The frame payload carries this frame's own visibility, not the last good one."""

import base64
import threading
from types import SimpleNamespace

import cv2
import numpy as np

from api.server import CoachingSession, REFERENCE_DIR, _live_visibility
from app.buffer import PoseBuffer
from app.freeze_monitor import FreezeMonitor
from app.imu_store import ImuStore
from app.pose_estimator import PoseFrame
from app.scorer import Scorer


def pose(visibility: float) -> PoseFrame:
    raw = np.random.default_rng(0).random((33, 3)).astype(np.float32)
    return PoseFrame(
        landmarks=raw.copy(),
        visibility=np.full(33, visibility, dtype=np.float32),
        raw_landmarks=raw,
    )


class ScriptedEstimator:
    """Returns the queued poses one per process() call (None means no person found)."""

    def __init__(self, poses):
        self.poses = list(poses)

    def process(self, rgb, keep_raw=False):
        return self.poses.pop(0)


def make_session(poses) -> CoachingSession:
    # Skip __init__ so the test never loads MediaPipe or the reference library.
    s = CoachingSession.__new__(CoachingSession)
    s._lock = threading.Lock()
    s.closed = False
    s.estimator = ScriptedEstimator(poses)
    s.buf = PoseBuffer()
    s.engine = SimpleNamespace(move_name=None)
    s.scorer = Scorer(REFERENCE_DIR)
    s.freeze = FreezeMonitor(ImuStore())
    s.future = None
    s.flash_counter = 0
    s.score_result = None
    s.dtw_result = None
    s.pose_frame = None
    s.frame_idx = 0
    s.last_captured_at = None
    s.lost_frames = 0
    s.frames_since_rep = 0
    s.future_window = []
    return s


def jpeg_b64() -> str:
    ok, buf = cv2.imencode(".jpg", np.zeros((48, 64, 3), dtype=np.uint8))
    assert ok
    return base64.b64encode(buf.tobytes()).decode()


def test_live_visibility_helper():
    assert _live_visibility(None) == []
    assert _live_visibility(pose(0.25)) == [0.25] * 33


def test_live_visibility_is_not_stale_after_dancer_leaves_frame():
    session = make_session([pose(0.9), pose(0.1), None])
    frame = jpeg_b64()

    reliable = session.process_frame_sync(frame)
    assert reliable["live_visibility"] == [np.float32(0.9).item()] * 33

    # Unreliable pose (torso below 0.5): "visibility" is held from the last good
    # frame, but live_visibility must show the current, low values.
    partial = session.process_frame_sync(frame)
    assert partial["visibility"] == reliable["visibility"]
    assert partial["live_visibility"] == [np.float32(0.1).item()] * 33

    # No person at all.
    gone = session.process_frame_sync(frame)
    assert gone["live_visibility"] == []
    assert gone["visibility"] == reliable["visibility"]
