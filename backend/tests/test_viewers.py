"""Second-screen viewers: a read-only mirror of a live session."""

import asyncio
import json
import threading

from fastapi.testclient import TestClient

from api import server
from api.server import app
from app.audio_frames import decode_audio_chunk
from app.coach import AICoach
from app.scorer import Scorer
from app.viewers import ViewerHub


class FakeWS:
    def __init__(self, fail=False, slow=False):
        self.texts, self.blobs = [], []
        self.fail, self.slow = fail, slow

    async def send_text(self, t):
        if self.fail:
            raise RuntimeError("closed")
        if self.slow:
            await asyncio.sleep(5)
        self.texts.append(t)

    async def send_bytes(self, b):
        if self.fail:
            raise RuntimeError("closed")
        self.blobs.append(b)


def test_hub_broadcasts_and_drops_dead_viewers():
    async def go():
        hub = ViewerHub()
        good, dead = FakeWS(), FakeWS(fail=True)
        hub.add(good)
        hub.add(dead)
        assert hub.count == 2
        changed = await hub.broadcast_text("hi")
        assert changed and hub.count == 1 and good.texts == ["hi"]
        assert not await hub.broadcast_bytes(b"x") and good.blobs == [b"x"]
    asyncio.run(go())


def test_slow_viewer_is_dropped_not_waited_on(monkeypatch):
    monkeypatch.setattr("app.viewers.SEND_TIMEOUT_S", 0.05)

    async def go():
        hub = ViewerHub()
        hub.add(FakeWS(slow=True))
        assert await hub.broadcast_text("hi")
        assert hub.count == 0
    asyncio.run(go())


def test_send_coach_cue_mirrors_caption_and_skip_notice_to_viewers():
    async def go():
        phone, tv = FakeWS(), FakeWS()
        hub = ViewerHub()
        hub.add(tv)
        sent = await server.send_coach_cue(phone, None, "Stay low", viewers=hub)
        assert sent is False
        assert [json.loads(t)["type"] for t in tv.texts] == ["coach_cue", "coach_audio_skipped"]
        assert tv.texts == phone.texts
    asyncio.run(go())


def make_session():
    s = server.CoachingSession.__new__(server.CoachingSession)
    s._lock = threading.Lock()
    s.closed = False
    s.estimator = type("E", (), {"close": lambda self: None})()
    s.engine = type("En", (), {"get_move": lambda self, n: type("M", (), {"meta": {"duration_s": 3.0}})(),
                              "shutdown": lambda self: None})()
    s.voice = None
    s.ai_coach = AICoach(llm=lambda *a: None)
    s.scorer = Scorer(server.REFERENCE_DIR)
    s.move_id = "br_01"
    s.dtw_result = None
    s.speak_lock = asyncio.Lock()
    s.coach_tasks = set()
    s.viewers = ViewerHub()
    s.controller = None
    return s


def test_viewer_socket_mirrors_the_phone_and_reports_count():
    server._sessions["sess-view"] = make_session()
    try:
        client = TestClient(app)
        with client.websocket_connect("/ws/sess-view/view") as tv:
            assert json.loads(tv.receive_text())["type"] == "voice_status"
            with client.websocket_connect("/ws/sess-view") as phone:
                assert json.loads(phone.receive_text())["type"] == "voice_status"
                assert json.loads(phone.receive_text()) == {"type": "viewers", "count": 1}
                phone.send_text(json.dumps({"type": "demo_request"}))
                assert json.loads(phone.receive_text())["type"] == "demo"
                assert json.loads(tv.receive_text())["type"] == "demo"
                # a ping on the viewer socket gets its own pong, nothing is forwarded to the phone
                tv.send_text(json.dumps({"type": "ping"}))
                assert json.loads(tv.receive_text()) == {"type": "pong"}
            assert server._sessions["sess-view"].viewers.count == 1
        assert server._sessions["sess-view"].viewers.count == 0
    finally:
        server._sessions.pop("sess-view", None)


def test_phone_is_told_when_the_last_viewer_leaves():
    server._sessions["sess-left"] = make_session()
    try:
        client = TestClient(app)
        with client.websocket_connect("/ws/sess-left") as phone:
            phone.receive_text()          # voice status
            with client.websocket_connect("/ws/sess-left/view") as tv:
                tv.receive_text()
                assert json.loads(phone.receive_text()) == {"type": "viewers", "count": 1}
            assert json.loads(phone.receive_text()) == {"type": "viewers", "count": 0}
    finally:
        server._sessions.pop("sess-left", None)


def test_viewer_for_an_unknown_session_is_refused():
    with TestClient(app).websocket_connect("/ws/nope/view") as tv:
        assert json.loads(tv.receive_text()) == {"error": "session not found"}


def test_latest_session_is_the_newest_open_one():
    old, new, closed = make_session(), make_session(), make_session()
    closed.closed = True
    server._sessions.clear()
    server._sessions.update({"a": old, "b": new, "c": closed})
    try:
        assert TestClient(app).get("/sessions/latest").json() == {"session_id": "b"}
        server._sessions.pop("b")
        assert TestClient(app).get("/sessions/latest").json() == {"session_id": "a"}
        server._sessions.clear()
        assert TestClient(app).get("/sessions/latest").json() == {"session_id": None}
    finally:
        server._sessions.clear()
