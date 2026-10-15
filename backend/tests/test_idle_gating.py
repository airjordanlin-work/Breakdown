"""A still or missing dancer must not be scored or coached."""

import base64
import threading
from concurrent.futures import Future

import cv2
import numpy as np

from api import server
from api.server import CoachingSession, LOST_RESET_FRAMES
from app.attempts import AttemptTracker
from app.buffer import PoseBuffer, WINDOW_LEN
from app.dtw_engine import DTWResult
from app.pose_estimator import PoseFrame

_, _jpg = cv2.imencode(".jpg", np.zeros((8, 8, 3), np.uint8))
FRAME_B64 = base64.b64encode(_jpg.tobytes()).decode()


def _pose(offset: float) -> PoseFrame:
    lm = np.linspace(-1, 1, 99, dtype=np.float32).reshape(33, 3) + offset
    return PoseFrame(landmarks=lm, visibility=np.ones(33, np.float32), raw_landmarks=lm)


class ScriptedEstimator:
    def __init__(self, poses):
        self.poses = list(poses)

    def process(self, rgb, keep_raw=False):
        return self.poses.pop(0)


class AlwaysAlignedEngine:
    move_name = "br_01"

    def __init__(self):
        self.compares = 0

    def compare_async(self, window):
        self.compares += 1
        f = Future()
        f.set_result(DTWResult(True, 1.0, 0.1, "br_01", 2000.0, WINDOW_LEN, WINDOW_LEN))
        return f

    def get_move(self, name):
        return None


class CountingScorer:
    def __init__(self):
        self.calls = 0

    def score_rep(self, dtw, window):
        self.calls += 1
        return None

    def stats(self):
        from app.scorer import Scorer
        return Scorer(server.REFERENCE_DIR).stats()


class NoFreeze:
    def on_frame(self, *a):
        return {"result": None}


def make_session(poses) -> CoachingSession:
    s = CoachingSession.__new__(CoachingSession)
    s._lock = threading.Lock()
    s.closed = False
    s.estimator = ScriptedEstimator(poses)
    s.buf = PoseBuffer()
    s.engine = AlwaysAlignedEngine()
    s.scorer = CountingScorer()
    s.freeze = NoFreeze()
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
    s.attempts = AttemptTracker()
    s.previous_rep = None
    s.latest_rep = None
    return s


def run(s, n):
    return [s.process_frame_sync(FRAME_B64, 0.0) for _ in range(n)]


def test_standing_still_is_never_scored_or_coached():
    s = make_session([_pose(0.0)] * (WINDOW_LEN + 20))
    out = run(s, WINDOW_LEN + 20)
    assert s.engine.compares > 0
    assert s.scorer.calls == 0
    ctx = out[-1]["_coach_context"]
    assert ctx["tracked"] and not ctx["moving"]
    assert ctx["move_name"] == "BR 01"


def test_moving_dancer_is_scored():
    poses = [_pose(0.2 * np.sin(i / 2)) for i in range(WINDOW_LEN + 20)]
    s = make_session(poses)
    out = run(s, WINDOW_LEN + 20)
    assert s.scorer.calls > 0
    assert out[-1]["_coach_context"]["moving"]


def test_lost_body_clears_the_window_and_stops_scoring():
    moving = [_pose(0.2 * np.sin(i / 2)) for i in range(WINDOW_LEN)]
    s = make_session(moving + [None] * (LOST_RESET_FRAMES + 30))
    run(s, WINDOW_LEN)
    compares, scored = s.engine.compares, s.scorer.calls
    out = run(s, LOST_RESET_FRAMES + 30)
    assert len(s.buf) == 0
    assert s.engine.compares == compares
    assert s.scorer.calls == scored
    ctx = out[-1]["_coach_context"]
    assert not ctx["tracked"] and not ctx["moving"]


def test_one_score_per_rep_not_per_comparison():
    from app.scorer import KeyframeResult, ScoreResult

    class RepScorer(CountingScorer):
        def score_rep(self, dtw, window):
            self.calls += 1
            r = KeyframeResult("left_elbow", 150.0, 160.0, 10.0, "Perfect", 100.0)
            return ScoreResult([r], 100.0, 1.0, "br_01")

    n = WINDOW_LEN * 3
    s = make_session([_pose(0.2 * np.sin(i / 2)) for i in range(n)])
    s.scorer = RepScorer()
    run(s, n)
    # DTW compares on nearly every full-buffer frame, but a rep is scored
    # only once per WINDOW_LEN new frames
    assert s.engine.compares > 3
    assert s.scorer.calls == n // WINDOW_LEN


def test_stopping_after_reps_hands_the_attempt_to_the_debrief_once():
    from app.scorer import KeyframeResult, ScoreResult

    class RepScorer(CountingScorer):
        def score_rep(self, dtw, window):
            self.calls += 1
            r = KeyframeResult("left_elbow", 120.0, 160.0, 40.0, "Miss", 0.0)
            return ScoreResult([r], 0.0, 1.0, "br_01")

    n_move = WINDOW_LEN * 2
    n_still = 60
    poses = [_pose(0.2 * np.sin(i / 2)) for i in range(n_move)] + [_pose(0.0)] * n_still
    s = make_session(poses)
    s.scorer = RepScorer()
    out = [s.process_frame_sync(FRAME_B64, i / 15) for i in range(n_move + n_still)]
    debriefs = [o["_coach_context"]["debrief_reps"] for o in out if o["_coach_context"]["debrief_reps"]]
    assert len(debriefs) == 1
    assert len(debriefs[0]) == s.scorer.calls
    assert debriefs[0][0]["results"][0] == {
        "joint": "left elbow", "tier": "Miss", "diff": 40.0, "fix": "straighten", "points": 0,
    }
