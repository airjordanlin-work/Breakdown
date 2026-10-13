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


# ------------------------------------------------------------ live cues

import asyncio

from app import coach as coach_mod
from app.coach import AICoach, _build_prompt, correction_focus, fallback_cue


def _miss(joint, fix, diff=40.0):
    return {"joint": joint, "tier": "Miss", "diff": diff, "fix": fix}


def test_prompt_names_the_move_and_the_joint_that_is_off():
    p = _build_prompt("BR 01", {"results": [
        {"joint": "left elbow", "tier": "Close", "diff": 20.0, "fix": "bend"},
        _miss("right elbow", "straighten"),
    ]})
    assert '"BR 01"' in p
    assert "right elbow needs to straighten" in p
    assert "left elbow" not in p          # only the worst joint
    assert "40" not in p                  # no numbers for the model to repeat


def test_bend_direction_comes_from_the_score():
    p = _build_prompt("BR 01", {"results": [_miss("left knee", "bend")]})
    assert "left knee needs to bend" in p


def test_no_data_prompt_forbids_body_parts():
    p = _build_prompt("BR 01", None, ["Keep it moving"])
    assert "Do not name any body part" in p
    assert '"Keep it moving"' in p


def test_move_tips_are_offered_when_written():
    p = _build_prompt("Six-step", None, (), ["Keep your hips low"])
    assert "Keep your hips low" in p


def test_focus_picks_worst_miss():
    f = correction_focus({"results": [_miss("left elbow", "bend", 35), _miss("right elbow", "straighten", 70)]})
    assert f == {"tier": "Miss", "joint": "right elbow", "fix": "straighten"}
    assert correction_focus({"results": [{"joint": "x", "tier": "Perfect", "diff": 1}]})["tier"] == "Perfect"
    assert correction_focus(None) is None


def test_fallback_cue_uses_joint_and_direction():
    cue = fallback_cue({"results": [_miss("left elbow", "straighten")]})
    assert "left elbow" in cue
    assert "Bend" not in cue
    cue = fallback_cue({"results": [_miss("left elbow", "bend")]})
    assert "left elbow" in cue and "Straighten" not in cue


def test_fallback_cue_avoids_recent():
    seen = set()
    recent = []
    for _ in range(3):
        cue = fallback_cue(None, recent)
        assert cue not in recent
        recent.append(cue)
        seen.add(cue)
    assert len(seen) == 3


def test_fallback_cue_is_short():
    for score in (None, {"results": [_miss("right shoulder", "bend")]},
                  {"results": [{"joint": "left hip", "tier": "Close", "diff": 20, "fix": "straighten"}]}):
        for _ in range(10):
            assert len(fallback_cue(score).split()) <= coach_mod.MAX_CUE_WORDS


def test_silent_without_tracked_pose_or_movement():
    c = AICoach()
    assert not c.should_fire(tracked=False, moving=False, fill_ratio=1.0)
    assert not c.should_fire(tracked=True, moving=False, fill_ratio=1.0)
    assert not c.should_fire(tracked=False, moving=True, fill_ratio=1.0)
    assert not c.should_fire(tracked=True, moving=True, fill_ratio=0.5)
    assert c.should_fire(tracked=True, moving=True, fill_ratio=1.0)


def test_cooldown_still_applies(monkeypatch):
    c = AICoach()
    now = [1000.0]
    monkeypatch.setattr(coach_mod.time, "time", lambda: now[0])
    asyncio.run(c.get_cue("BR 01", {"results": [_miss("left elbow", "bend")]}))
    now[0] += coach_mod.COOLDOWN_SECONDS - 0.1
    assert not c.should_fire(tracked=True, moving=True, fill_ratio=1.0, has_score=True)
    now[0] += 0.2
    assert c.should_fire(tracked=True, moving=True, fill_ratio=1.0, has_score=True)
    # with no joint data it waits longer before encouraging again
    assert not c.should_fire(tracked=True, moving=True, fill_ratio=1.0, has_score=False)


def test_no_api_key_gives_templated_cue_and_never_repeats(monkeypatch):
    c = AICoach()
    monkeypatch.setattr(c, "_available", False)
    cues = [asyncio.run(c.get_cue("BR 01", None)) for _ in range(6)]
    assert all(cues)
    assert all(a != b for a, b in zip(cues, cues[1:]))


def test_model_failure_falls_back(monkeypatch):
    c = AICoach()
    monkeypatch.setattr(c, "_available", True)

    async def boom(*a, **kw):
        return None
    monkeypatch.setattr(c, "_ask_model", boom)
    cue = asyncio.run(c.get_cue("BR 01", {"results": [_miss("right knee", "bend")]}))
    assert "right knee" in cue
