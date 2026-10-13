"""ACCESS_TOKEN gates every REST route and WebSocket."""

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from api.server import app
from app.access import token_is_valid

client = TestClient(app)


def test_token_is_valid_rules():
    assert token_is_valid(None, "")            # auth off
    assert not token_is_valid(None, "s3cret")
    assert not token_is_valid("wrong", "s3cret")
    assert token_is_valid("s3cret", "s3cret")


def test_open_when_no_token_configured(monkeypatch):
    monkeypatch.delenv("ACCESS_TOKEN", raising=False)
    assert client.get("/health").status_code == 200


def test_rest_requires_token(monkeypatch):
    monkeypatch.setenv("ACCESS_TOKEN", "s3cret")
    assert client.get("/moves").status_code == 401
    assert client.get("/moves", headers={"X-Access-Token": "nope"}).status_code == 401
    assert client.get("/moves", headers={"X-Access-Token": "s3cret"}).status_code == 200
    assert client.get("/moves?token=s3cret").status_code == 200
    assert client.get("/health").status_code == 401


def test_session_start_requires_token(monkeypatch):
    monkeypatch.setenv("ACCESS_TOKEN", "s3cret")
    assert client.post("/session/start", json={}).status_code == 401


def test_websocket_rejected_without_token(monkeypatch):
    monkeypatch.setenv("ACCESS_TOKEN", "s3cret")
    for path in ("/ws/imu", "/ws/some-session"):
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(path):
                pass


def test_websocket_accepts_token(monkeypatch):
    monkeypatch.setenv("ACCESS_TOKEN", "s3cret")
    with client.websocket_connect("/ws/missing?token=s3cret") as ws:
        assert ws.receive_json() == {"error": "session not found"}


def test_docs_and_openapi_are_off(monkeypatch):
    monkeypatch.delenv("ACCESS_TOKEN", raising=False)
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 404


def test_no_route_exposes_env_or_secrets(monkeypatch):
    """Every GET response is free of the API key and the access token."""
    monkeypatch.setenv("ACCESS_TOKEN", "tok-xyz-123")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake-456")
    h = {"X-Access-Token": "tok-xyz-123"}
    for path in ("/health", "/moves", "/imu/status"):
        body = client.get(path, headers=h).text
        assert "tok-xyz-123" not in body and "sk-ant-fake-456" not in body
