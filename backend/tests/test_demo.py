"""Slow demo on request: the clip, what to highlight, and a grounded line."""

import asyncio
import json
import threading

import pytest
from fastapi.testclient import TestClient

from api import server
from api.server import CoachingSession, app
from app.coach import AICoach
from app.demo import DEMO_LOOPS, DEMO_SPEED, demo_facts, demo_line, demo_message, highlight_for
from app.grounding import ungrounded_claims
from app.move_catalog import PREVIEW_JOINTS
from app.scorer import Scorer


def test_highlight_uses_preview_joint_order():
    assert [PREVIEW_JOINTS[i] for i in highlight_for("hips")] == [23, 24]
    assert [PREVIEW_JOINTS[i] for i in highlight_for("left_elbow")] == [13]
    assert highlight_for("spine") == [] and highlight_for(None) == []


@pytest.mark.parametrize("joint,fix", [
    ("hips", "twist"), ("leg", "reach"), ("left elbow", "straighten"), ("right knee", "bend"), (None, None),
])
def test_demo_line_only_names_the_measured_focus(joint, fix):
    facts = demo_facts("BR 01", joint, fix)
    line = demo_line(facts)
    assert line.startswith("Here's BR 01 at half speed.")
    assert ungrounded_claims(line, facts) == []
    if joint:
        assert f"Watch the {joint}." in line
    else:
        assert "Watch the whole move." in line


def test_unknown_focus_falls_back_to_the_whole_move():
    msg, line, _ = demo_message("br_01", "BR 01", "spine", "twist", 3.0)
    assert msg["focus"] is None and msg["highlight"] == []
    assert "whole move" in line


def test_demo_message_and_quiet_time():
    msg, line, quiet = demo_message("br_01", "BR 01", "hips", "twist", 3.0)
    assert msg == {
        "type": "demo", "move_id": "br_01", "move_name": "BR 01",
        "speed": DEMO_SPEED, "loops": DEMO_LOOPS, "seconds": 3.0, "focus": "hips",
        "highlight": highlight_for("hips"),
    }
    assert quiet == pytest.approx(3.0 / DEMO_SPEED * DEMO_LOOPS)
    assert "hips" in line


def test_coach_stays_quiet_while_a_demo_plays():
    async def never(*a):
        return None
    coach = AICoach(llm=never)
    ready = dict(tracked=True, moving=True, fill_ratio=1.0, has_score=True)
    assert coach.should_fire(**ready)
    coach.hold(60)
    assert not coach.should_fire(**ready)


class FakeEngine:
    def __init__(self, meta=None):
        self.meta = meta or {}

    def get_move(self, name):
        return type("M", (), {"meta": self.meta})()

    def shutdown(self):
        pass


def make_session(move_id):
    s = CoachingSession.__new__(CoachingSession)
    s._lock = threading.Lock()
    s.closed = False
    s.estimator = type("E", (), {"close": lambda self: None})()
    s.engine = FakeEngine({"duration_s": 3.0})
    s.voice = None
    s.ai_coach = AICoach(llm=lambda *a: None)
    s.scorer = Scorer(server.REFERENCE_DIR)
    s.move_id = move_id
    s.dtw_result = None
    s.speak_lock = asyncio.Lock()
    s.coach_tasks = set()
    return s


def test_socket_demo_request_sends_demo_then_spoken_line():
    server._sessions["sess-demo"] = make_session("br_01")
    try:
        with TestClient(app).websocket_connect("/ws/sess-demo") as ws:
            ws.receive_text()   # voice status
            ws.send_text(json.dumps({"type": "demo_request"}))
            demo = json.loads(ws.receive_text())
            cue = json.loads(ws.receive_text())
            assert demo["type"] == "demo" and demo["move_id"] == "br_01"
            assert cue["type"] == "coach_cue" and cue["mode"] == "demo"
            assert cue["cue"].startswith("Here's BR 01 at half speed.")
        assert not server._sessions["sess-demo"].ai_coach.should_fire(
            tracked=True, moving=True, fill_ratio=1.0, has_score=True)
    finally:
        server._sessions.pop("sess-demo", None)


def test_socket_demo_without_a_move_says_so():
    server._sessions["sess-nodemo"] = make_session(None)
    try:
        with TestClient(app).websocket_connect("/ws/sess-nodemo") as ws:
            ws.receive_text()
            ws.send_text(json.dumps({"type": "demo_request"}))
            assert json.loads(ws.receive_text()) == {
                "type": "demo_unavailable", "reason": "Pick a move first"}
    finally:
        server._sessions.pop("sess-nodemo", None)


def test_hold_blocks_event_cues_and_debrief():
    coach = AICoach(llm=lambda *a: None)
    assert not coach.held and coach.can_fire_now()
    coach.hold(5.0)
    assert coach.held and not coach.can_fire_now()


def test_release_ends_a_demo_hold():
    coach = AICoach(llm=lambda *a: None)
    coach.hold(60)
    assert coach.held
    coach.release()
    assert not coach.held and coach.can_fire_now()


def test_socket_survives_a_second_demo_and_a_close():
    server._sessions["sess-twice"] = make_session("br_01")
    try:
        with TestClient(app).websocket_connect("/ws/sess-twice") as ws:
            ws.receive_text()
            for _ in range(2):
                ws.send_text(json.dumps({"type": "demo_request"}))
                assert json.loads(ws.receive_text())["type"] == "demo"
                assert json.loads(ws.receive_text())["mode"] == "demo"
                assert json.loads(ws.receive_text())["type"] == "coach_audio_skipped"
                ws.send_text(json.dumps({"type": "demo_close"}))
            ws.send_text(json.dumps({"type": "ping"}))
            assert json.loads(ws.receive_text()) == {"type": "pong"}
        assert not server._sessions["sess-twice"].ai_coach.held
    finally:
        server._sessions.pop("sess-twice", None)


def test_a_failing_demo_does_not_close_the_socket():
    session = make_session("br_01")
    session.demo = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    server._sessions["sess-boom"] = session
    try:
        with TestClient(app).websocket_connect("/ws/sess-boom") as ws:
            ws.receive_text()
            ws.send_text(json.dumps({"type": "demo_request"}))
            assert json.loads(ws.receive_text())["type"] == "demo_unavailable"
            ws.send_text(json.dumps({"type": "ping"}))
            assert json.loads(ws.receive_text()) == {"type": "pong"}
    finally:
        server._sessions.pop("sess-boom", None)
