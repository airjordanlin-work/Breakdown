"""CORS origins come from CORS_ORIGINS, with localhost dev defaults."""

import pytest

from api.server import DEFAULT_CORS_ORIGINS, cors_origins


@pytest.mark.parametrize("raw", [None, "", "   "])
def test_defaults_when_unset(raw):
    assert cors_origins(raw) == DEFAULT_CORS_ORIGINS


def test_parses_comma_separated_list():
    raw = " https://a.trycloudflare.com/ , http://192.168.1.20:5173,, "
    assert cors_origins(raw) == ["https://a.trycloudflare.com", "http://192.168.1.20:5173"]


def test_defaults_are_not_mutated_by_callers():
    cors_origins(None).append("http://evil.example")
    assert "http://evil.example" not in DEFAULT_CORS_ORIGINS
