"""Tests for coach prompt building. No API calls."""

from app.coach import _build_freeze_prompt


def test_clean_freeze_asks_for_hype():
    p = _build_freeze_prompt({"tier": "rock solid", "duration_s": 2.1,
                              "stability": 94, "shakiest": "leg"}, None)
    assert "2.1 seconds" in p
    assert "hype" in p
    assert "legs" not in p          # don't criticize a clean freeze


def test_shaky_freeze_names_the_limb():
    p = _build_freeze_prompt({"tier": "shaky", "duration_s": 1.2,
                              "stability": 55, "shakiest": "wrist"}, "Lock it in")
    assert "base arm" in p
    assert 'Do NOT repeat: "Lock it in"' in p
