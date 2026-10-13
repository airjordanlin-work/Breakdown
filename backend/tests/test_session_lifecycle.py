"""Ending a session while its socket still sends frames must not crash the socket."""

import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from api import server
from api.server import CoachingSession, SessionClosed, app


class FakeEstimator:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class FakeCoach:
    def can_fire_now(self):
        return False

    def should_fire(self, **kw):
        return False


class FakeEngine:
    def __init__(self):
        self.closed = False

    def shutdown(self):
        self.closed = True


def make_session(frame_delay: float = 0.0) -> CoachingSession:
    # Skip __init__ so the test never loads MediaPipe or the reference library.
    s = CoachingSession.__new__(CoachingSession)
    s._lock = threading.Lock()
    s.closed = False
    s.estimator = FakeEstimator()
    s.engine = FakeEngine()
    s.voice = None
    s.ai_coach = FakeCoach()
    s.calls = 0

    def fake_frame(b64, captured_at):
        s.calls += 1
        time.sleep(frame_delay)
        # the estimator must still be open for the whole frame
        assert not s.estimator.closed
        return {"type": "frame"}

    s._process_frame = fake_frame
    return s


def test_frame_after_close_raises_session_closed():
    s = make_session()
    s.close()
    assert s.engine.closed and s.estimator.closed
    with pytest.raises(SessionClosed):
        s.process_frame_sync("", None)
    assert s.calls == 0


def test_close_is_idempotent():
    s = make_session()
    s.close()
    s.close()


def test_close_waits_for_in_flight_frame():
    s = make_session(frame_delay=0.2)
    worker = threading.Thread(target=s.process_frame_sync, args=("", None))
    worker.start()
    time.sleep(0.05)          # frame is now mid-flight
    s.close()                 # must block until the frame finishes
    worker.join()
    assert s.calls == 1 and s.closed


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("ACCESS_TOKEN", raising=False)
    return TestClient(app)


def test_socket_reports_voice_status_on_connect(client):
    server._sessions["sess-voice"] = make_session()
    try:
        with client.websocket_connect("/ws/sess-voice") as ws:
            msg = json.loads(ws.receive_text())
            assert msg == {
                "type": "voice_status", "available": False,
                "reason": "no voice for this session",
            }
    finally:
        server._sessions.pop("sess-voice", None)


def test_frame_after_end_session_closes_socket_cleanly(client):
    s = make_session()
    server._sessions["sess-end"] = s
    with client.websocket_connect("/ws/sess-end") as ws:
        ws.receive_text()                                   # voice_status
        ws.send_text(json.dumps({"type": "frame", "data": ""}))
        assert json.loads(ws.receive_text())["type"] == "frame"

        assert client.delete("/session/sess-end").status_code == 200
        assert s.closed and "sess-end" not in server._sessions

        ws.send_text(json.dumps({"type": "frame", "data": ""}))
        closed = ws.receive()
        assert closed["type"] == "websocket.close"
        assert closed["code"] == 1000
