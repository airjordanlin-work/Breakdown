"""Grounding guard: refuse coach text that states anything not measured."""

from app.grounding import ungrounded_claims

LIVE = {"move": "BR 01", "focus": {"tier": "Miss", "joint": "left elbow", "fix": "straighten"}}
DEBRIEF = {
    "move": "BR 01", "reps": 4, "clean_reps": 3, "best_streak": 3,
    "off_most": {"joint": "right knee", "fix": "bend", "times_off": 1},
    "landed_every_rep": ["left elbow"],
}


def test_measured_joint_and_side_pass():
    assert ungrounded_claims("Straighten that left elbow", LIVE) == []
    assert ungrounded_claims("Open up the elbow", LIVE) == []
    assert ungrounded_claims("Reach through that left arm", LIVE) == []


def test_wrong_side_is_refused():
    assert ungrounded_claims("Straighten your right elbow", LIVE)


def test_unmeasured_body_part_is_refused():
    assert ungrounded_claims("Get your hips lower", LIVE)
    assert ungrounded_claims("Point your toes", LIVE)
    assert ungrounded_claims("Watch your knees", LIVE)


def test_generic_body_is_fine():
    assert ungrounded_claims("Keep your body loose", {"move": "BR 01"}) == []


def test_numbers_must_come_from_facts():
    assert ungrounded_claims("Three clean out of 4, nice", DEBRIEF) == []
    assert ungrounded_claims("Five clean reps, nice", DEBRIEF)
    assert ungrounded_claims("That was 90 percent", DEBRIEF)
    assert ungrounded_claims("That's the one", LIVE) == []


def test_move_name_digits_are_not_claims():
    assert ungrounded_claims("Nice BR 01", LIVE) == []


def test_angles_scores_and_past_sessions_are_refused():
    for text in ("Ten degrees off", "Great points there", "Better than last time",
                 "You're improving", "Cleaner than yesterday", "Elbow at 45°"):
        assert ungrounded_claims(text, LIVE), text


def test_tips_ground_body_parts_when_no_joint_data():
    facts = {"move": "Six-step", "tips": ["Keep your hips low"]}
    assert ungrounded_claims("Hips low, keep stepping", facts) == []
    assert ungrounded_claims("Bend that knee", facts)
