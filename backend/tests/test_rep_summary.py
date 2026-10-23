"""The result card: what landed, what to fix, and where in the move."""

from pathlib import Path

from app.attempts import attempt_summary
from app.dtw_engine import compare_to_move, load_reference_library
from app.scorer import Scorer

REF_DIR = Path(__file__).resolve().parents[1] / "reference_moves"


def rep(*joints, moments=()):
    return {"results": [{"joint": j, "tier": t, "fix": f} for j, t, f in joints], "moments": list(moments)}


def moment(checkpoint, joint, of=3, fix="straighten"):
    return {"checkpoint": checkpoint, "of": of, "joint": joint, "tier": "Miss", "fix": fix}


def test_reference_rep_has_no_moments():
    move = next(m for m in load_reference_library(REF_DIR) if m.name == "br_01")
    result = Scorer(REF_DIR).score_rep(compare_to_move(move.sequence, move), list(move.sequence.reshape(-1, 33, 3)))
    assert result.moments == []


def test_bent_elbow_reports_the_checkpoints_where_it_was_off():
    move = next(m for m in load_reference_library(REF_DIR) if m.name == "br_01")
    window = [f.copy() for f in move.sequence.reshape(-1, 33, 3)]
    for f in window:
        f[15] = f[11]  # fold the left wrist onto the shoulder
    result = Scorer(REF_DIR).score_rep(compare_to_move(move.sequence, move), window)
    left = [m for m in result.moments if m["joint"] == "left elbow"]
    assert left and all(m["fix"] == "straighten" and 1 <= m["checkpoint"] <= m["of"] for m in left)
    assert not [m for m in result.moments if m["joint"] == "right elbow" and m["tier"] == "Miss"]


def test_summary_counts_and_what_went_well():
    reps = [
        rep(("left elbow", "Miss", "straighten"), ("left knee", "Perfect", "bend")),
        rep(("left elbow", "Perfect", "straighten"), ("left knee", "Perfect", "bend")),
    ]
    card = attempt_summary("Cross step", reps)
    assert card["type"] == "rep_summary" and card["move"] == "Cross step"
    assert (card["reps"], card["clean_reps"], card["best_streak"]) == (2, 1, 1)
    assert card["went_well"] == ["left knee"]
    assert card["fix"] == {"joint": "left elbow", "fix": "straighten", "times_off": 1}


def test_summary_names_the_checkpoint_where_the_joint_was_off_most():
    reps = [
        rep(("left elbow", "Miss", "straighten"), moments=[moment(2, "left elbow"), moment(3, "left elbow")]),
        rep(("left elbow", "Miss", "straighten"), moments=[moment(2, "left elbow")]),
        rep(("left elbow", "Close", "straighten"), moments=[moment(2, "left elbow"), moment(1, "right elbow")]),
    ]
    fix = attempt_summary("Cross step", reps)["fix"]
    assert fix["checkpoint"] == 2 and fix["of"] == 3 and fix["times_off"] == 3


def test_feature_fix_has_no_checkpoint():
    card = attempt_summary("Cross step", [rep(("hips", "Miss", "twist"))])
    assert card["fix"] == {"joint": "hips", "fix": "twist", "times_off": 1}


def test_clean_attempt_has_no_fix():
    card = attempt_summary("Cross step", [rep(("left knee", "Perfect", "bend"))])
    assert card["fix"] is None and card["went_well"] == ["left knee"]
