"""LLM-written live cues and debriefs, with a fake model client. No API calls."""

import asyncio

from app import coach as coach_mod
from app.attempts import attempt_facts
from app.coach import AICoach, fallback_debrief, live_facts
from app.grounding import ungrounded_claims


class FakeLLM:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    async def __call__(self, system, prompt, max_tokens, timeout):
        self.calls.append({"system": system, "prompt": prompt, "max_tokens": max_tokens, "timeout": timeout})
        return self.replies.pop(0) if self.replies else None


def rep(*joints):
    return {"results": [{"joint": j, "tier": t, "fix": f} for j, t, f in joints]}


MISS_L = rep(("left elbow", "Miss", "straighten"), ("right elbow", "Perfect", "bend"))
MISS_R = rep(("left elbow", "Perfect", "bend"), ("right elbow", "Miss", "bend"))
CLEAN = rep(("left elbow", "Perfect", "bend"), ("right elbow", "Perfect", "bend"))


def cue(c, *args, **kw):
    return asyncio.run(c.get_cue(*args, **kw))


# ------------------------------------------------------------ live cues

def test_grounded_model_cue_is_spoken():
    llm = FakeLLM("Still that left elbow, open it")
    c = AICoach(llm=llm)
    out = cue(c, "BR 01", MISS_L, previous_result=MISS_L)
    assert out == "Still that left elbow, open it"
    assert llm.calls[0]["timeout"] == coach_mod.TIMEOUT_SECONDS
    assert "Same joint was off on the rep before" in llm.calls[0]["prompt"]


def test_cue_naming_unmeasured_joint_is_refused():
    c = AICoach(llm=FakeLLM("Drop those hips lower"))
    out = cue(c, "BR 01", MISS_L)
    assert "hips" not in out
    assert "left elbow" in out            # templated from the same facts
    assert c.refused == 1


def test_cue_with_wrong_side_or_invented_number_is_refused():
    c = AICoach(llm=FakeLLM("Straighten that right elbow"))
    assert "left elbow" in cue(c, "BR 01", MISS_L)
    c = AICoach(llm=FakeLLM("Five clean in a row"))
    out = cue(c, "BR 01", CLEAN, clean_streak=3)
    assert "Five" not in out
    assert c.refused == 1


def test_streak_number_from_facts_is_allowed():
    llm = FakeLLM("Three clean in a row, keep going")
    c = AICoach(llm=llm)
    assert cue(c, "BR 01", CLEAN, clean_streak=3) == "Three clean in a row, keep going"
    assert "3 clean reps in a row" in llm.calls[0]["prompt"]


def test_fixed_joint_is_offered_to_the_model():
    f = live_facts("BR 01", MISS_R, previous_result=MISS_L)
    assert f["fixed_since_last_rep"] == ["left elbow"]
    assert f["focus"]["joint"] == "right elbow"
    assert "same_joint_as_last_rep" not in f


def test_short_streak_is_not_offered():
    assert "clean_streak" not in live_facts("BR 01", CLEAN, clean_streak=1)


def test_model_timeout_falls_back():
    c = AICoach(llm=FakeLLM(None))
    assert "left elbow" in cue(c, "BR 01", MISS_L)


# ------------------------------------------------------------ debrief

def debrief(c, reps, session=None):
    return asyncio.run(c.get_debrief("BR 01", reps, session))


def test_grounded_debrief_is_spoken():
    text = "Two of three clean. Left elbow kept folding, straighten it next round. Feel that?"
    llm = FakeLLM(text)
    c = AICoach(llm=llm)
    assert debrief(c, [MISS_L, CLEAN, CLEAN]) == text
    call = llm.calls[0]
    assert call["system"] == coach_mod.DEBRIEF_SYSTEM_PROMPT
    assert call["max_tokens"] == coach_mod.DEBRIEF_MAX_TOKENS
    assert '"clean_reps": 2' in call["prompt"]


def test_debrief_with_invented_progress_is_refused():
    c = AICoach(llm=FakeLLM("Way cleaner than last week, five clean!"))
    out = debrief(c, [MISS_L, CLEAN])
    assert "last week" not in out
    assert out == fallback_debrief(attempt_facts("BR 01", [MISS_L, CLEAN]))


def test_debrief_too_long_or_two_questions_falls_back():
    long = " ".join(["nice"] * (coach_mod.DEBRIEF_MAX_WORDS + 1))
    for reply in (long, "Good? Again?"):
        c = AICoach(llm=FakeLLM(reply))
        out = debrief(c, [CLEAN])
        assert out == fallback_debrief(attempt_facts("BR 01", [CLEAN]))


def test_no_reps_no_debrief():
    llm = FakeLLM("x")
    assert debrief(AICoach(llm=llm), []) is None
    assert llm.calls == []


def test_templated_debrief_is_grounded_and_short():
    cases = [[CLEAN], [CLEAN, CLEAN, CLEAN], [MISS_L], [MISS_L, CLEAN, CLEAN, MISS_R],
             [CLEAN, CLEAN, MISS_L, CLEAN]]
    for reps in cases:
        facts = attempt_facts("BR 01", reps, {"reps": 12, "clean_reps": 9, "best_streak": 5})
        text = fallback_debrief(facts)
        assert ungrounded_claims(text, facts) == [], text
        assert len(text.split()) <= coach_mod.DEBRIEF_MAX_WORDS, text


def test_templated_debrief_names_the_focus_joint():
    text = fallback_debrief(attempt_facts("BR 01", [MISS_L, CLEAN, MISS_L]))
    assert "straighten your left elbow" in text
    assert "3 reps, 1 clean" in text


def test_debrief_without_key_uses_template(monkeypatch):
    c = AICoach()
    monkeypatch.setattr(c, "_available", False)
    assert debrief(c, [CLEAN, CLEAN]) == "All 2 reps clean, nice work. Keep that going, you're on the right track."


def test_debrief_does_not_block_and_does_not_leave_coach_pending():
    c = AICoach(llm=FakeLLM("Clean rep, nice."))
    debrief(c, [CLEAN])
    assert c.can_fire_now()


def test_debrief_that_claims_rhythm_is_refused():
    c = AICoach(llm=FakeLLM("Nice work. Stay on the beat and keep a steady rhythm next round."))
    out = debrief(c, [MISS_L, CLEAN])
    assert "rhythm" not in out
    assert out == fallback_debrief(attempt_facts("BR 01", [MISS_L, CLEAN]))


def test_longer_explaining_debrief_is_allowed():
    text = ("Two clean out of three, that's solid work. Your left elbow keeps folding, "
            "so really straighten it, because a bigger shape reads cleaner. You've got this, go again.")
    c = AICoach(llm=FakeLLM(text))
    assert debrief(c, [MISS_L, CLEAN, CLEAN]) == text
