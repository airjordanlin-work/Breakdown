"""Live cues are off by default; the debrief is unaffected; the phone can switch them on."""

import json

from fastapi.testclient import TestClient

from api import server
from api.server import app
from app.coach import AICoach
from tests.test_viewers import make_session

READY = dict(tracked=True, moving=True, fill_ratio=1.0)


def test_coach_stays_silent_mid_rep_when_live_cues_are_off():
    coach = AICoach(llm=lambda *a: None)
    assert coach.should_fire(**READY)
    coach.live_cues = False
    assert not coach.should_fire(**READY)


def test_new_sessions_default_to_live_cues_off():
    assert server.LIVE_CUES_DEFAULT is False


def test_phone_can_switch_live_cues_on_and_off():
    session = make_session()
    server._sessions["sess-live"] = session
    try:
        with TestClient(app).websocket_connect("/ws/sess-live") as phone:
            phone.receive_text()  # voice status
            phone.send_text(json.dumps({"type": "live_cues", "on": True}))
            assert json.loads(phone.receive_text()) == {"type": "live_cues", "on": True}
            assert session.ai_coach.live_cues is True
            phone.send_text(json.dumps({"type": "live_cues", "on": False}))
            assert json.loads(phone.receive_text()) == {"type": "live_cues", "on": False}
            assert session.ai_coach.live_cues is False
    finally:
        server._sessions.pop("sess-live", None)
