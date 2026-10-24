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
                phone.send_text(json.dumps({"type": "demo_end"}))
                assert json.loads(tv.receive_text()) == {"type": "demo_end"}
                # a ping on the viewer socket gets its own pong, nothing is forwarded to the phone
                tv.send_text(json.dumps({"type": "ping"}))
                assert json.loads(tv.receive_text()) == {"type": "pong"}
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


def test_latest_session_is_the_newest_open_one_with_a_phone():
    old, new, closed, no_phone = make_session(), make_session(), make_session(), make_session()
    for s in (old, new, closed):
        s.controller = object()
    closed.closed = True
    server._sessions.clear()
    server._sessions.update({"a": old, "b": new, "c": closed, "d": no_phone})
    try:
        assert TestClient(app).get("/sessions/latest").json() == {"session_id": "b"}
        server._sessions.pop("b")
        assert TestClient(app).get("/sessions/latest").json() == {"session_id": "a"}
        server._sessions.clear()
        assert TestClient(app).get("/sessions/latest").json() == {"session_id": None}
    finally:
        server._sessions.clear()


def test_ending_a_session_closes_its_viewers():
    server._sessions["sess-end"] = make_session()
    try:
        client = TestClient(app)
        with client.websocket_connect("/ws/sess-end/view") as tv:
            tv.receive_text()
            assert client.delete("/session/sess-end").json() == {"status": "closed"}
            try:
                tv.receive_text()
                raise AssertionError("viewer should have been closed")
            except Exception as e:
                assert "AssertionError" not in type(e).__name__
    finally:
        server._sessions.pop("sess-end", None)


def test_viewer_survives_junk_input():
    server._sessions["sess-junk"] = make_session()
    try:
        with TestClient(app).websocket_connect("/ws/sess-junk/view") as tv:
            tv.receive_text()
            tv.send_text("not json")
            tv.send_text("[1, 2]")
            tv.send_text(json.dumps({"type": "ping"}))
            assert json.loads(tv.receive_text()) == {"type": "pong"}
    finally:
        server._sessions.pop("sess-junk", None)


def test_viewers_are_closed_when_the_phone_leaves():
    server._sessions["sess-gone"] = make_session()
    try:
        client = TestClient(app)
        with client.websocket_connect("/ws/sess-gone/view") as tv:
            tv.receive_text()
            with client.websocket_connect("/ws/sess-gone") as phone:
                phone.receive_text()
            assert server._sessions["sess-gone"].controller is None
            assert server._sessions["sess-gone"].viewers.count == 0
    finally:
        server._sessions.pop("sess-gone", None)


def test_video_goes_only_to_viewers_that_opted_in():
    async def go():
        hub = ViewerHub()
        a, b = FakeWS(), FakeWS()
        hub.add(a)
        hub.add(b)
        assert hub.video_count == 0
        hub.want_video(a, True)
        await hub.broadcast_video("pic")
        assert a.texts == ["pic"] and b.texts == []
        hub.want_video(a, False)
        await hub.broadcast_video("pic2")
        assert a.texts == ["pic"]
        hub.want_video(FakeWS(), True)   # not a viewer: ignored
        assert hub.video_count == 0
    asyncio.run(go())


def test_discard_forgets_the_video_opt_in():
    hub = ViewerHub()
    a = FakeWS()
    hub.add(a)
    hub.want_video(a, True)
    hub.discard(a)
    assert hub.video_count == 0


def test_make_preview_shrinks_a_frame_and_rejects_junk():
    import base64
    import cv2
    import numpy as np
    img = np.zeros((480, 640, 3), np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    small = server.make_preview(base64.b64encode(buf.tobytes()).decode())
    out = cv2.imdecode(np.frombuffer(base64.b64decode(small), np.uint8), cv2.IMREAD_COLOR)
    assert out.shape[1] == server.PREVIEW_WIDTH
    assert server.make_preview("not an image") is None
