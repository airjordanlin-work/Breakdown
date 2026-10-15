"""Attempt tracking and the facts a debrief may use."""

from app.attempts import AttemptTracker, attempt_facts


def rep(*joints):
    return {"results": [{"joint": j, "tier": t, "fix": f} for j, t, f in joints]}


CLEAN = rep(("left elbow", "Perfect", "bend"), ("right elbow", "Perfect", "bend"))
MISS_L = rep(("left elbow", "Miss", "straighten"), ("right elbow", "Perfect", "bend"))
CLOSE_L = rep(("left elbow", "Close", "straighten"), ("right elbow", "Perfect", "bend"))


def test_attempt_ends_after_a_pause_once():
    t = AttemptTracker(pause_s=2.0)
    assert t.on_frame(True, 0.0) is None
    t.add_rep(CLEAN)
    assert t.on_frame(True, 1.0) is None
    assert t.on_frame(False, 2.5) is None      # only 1.5 s still
    assert t.on_frame(False, 3.1) == [CLEAN]   # 2.1 s still
    assert t.on_frame(False, 5.0) is None      # already debriefed


def test_no_debrief_without_reps():
    t = AttemptTracker(pause_s=1.0)
    t.on_frame(True, 0.0)
    assert t.on_frame(False, 10.0) is None


def test_facts_are_tallies_of_the_reps():
    f = attempt_facts("BR 01", [CLEAN, MISS_L, CLOSE_L, CLEAN], {"reps": 9, "clean_reps": 6, "best_streak": 4, "grade": "S"})
    assert f["reps"] == 4
    assert f["clean_reps"] == 3            # Close counts as clean, like the scorer
    assert f["best_streak"] == 2
    assert f["off_most"] == {"joint": "left elbow", "fix": "straighten", "times_off": 2}
    assert f["landed_every_rep"] == ["right elbow"]
    assert f["session"] == {"reps": 9, "clean_reps": 6, "best_streak": 4}   # no grade


def test_all_clean_has_no_focus():
    f = attempt_facts("BR 01", [CLEAN, CLEAN])
    assert f["off_most"] is None
    assert f["landed_every_rep"] == ["left elbow", "right elbow"]
