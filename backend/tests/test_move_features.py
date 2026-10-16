"""Whole-rep features (hip twist, leg reach): measured against the reference move."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from api.server import _score_payload
from app.coach import _build_prompt, fallback_cue, live_facts
from app.dtw_engine import DTWResult
from app.grounding import ungrounded_claims
from app.move_features import (
    CLOSE_RATIO, PERFECT_RATIO, compare, hip_turn, leg_reach, tier_for,
)
from app.scorer import Scorer

REF_DIR = Path(__file__).resolve().parents[1] / "reference_moves"
L_HIP, R_HIP, L_AN, R_AN = 23, 24, 27, 28


def _ref(name="br_01"):
    return np.load(REF_DIR / f"{name}.npy").reshape(-1, 33, 3).astype(np.float32)


def _square_hips(seq):
    """Same rep, but the hips never turn: hip width stays a fixed share of shoulder width."""
    out = seq.copy()
    mid = seq[:, [L_HIP, R_HIP], 0].mean(axis=1)
    half = np.abs(seq[:, 11, 0] - seq[:, 12, 0]) * 0.5 * 0.7
    out[:, L_HIP, 0] = mid - half * np.sign(seq[:, L_HIP, 0] - seq[:, R_HIP, 0])
    out[:, R_HIP, 0] = mid + half * np.sign(seq[:, L_HIP, 0] - seq[:, R_HIP, 0])
    return out


def _live(seq):
    """What the live pipeline hands the scorer: every frame normalized by its own hip width."""
    from app.pose_estimator import normalize_landmarks
    return np.stack([normalize_landmarks(f) for f in seq])


def _short_steps(seq, factor=0.3):
    """Same rep, feet kept close under the hips."""
    out = seq.copy()
    mid = seq[:, [L_HIP, R_HIP], 0].mean(axis=1)
    for a in (L_AN, R_AN):
        out[:, a, 0] = mid + (seq[:, a, 0] - mid) * factor
    return out


def test_reference_matches_itself():
    ref = _ref()
    results = compare(["hip_turn", "leg_reach"], ref, ref)
    assert [r.tier for r in results] == ["Perfect", "Perfect"]
    assert all(r.ratio == pytest.approx(1.0) for r in results)


def test_square_hips_miss_the_twist():
    ref = _ref()
    assert hip_turn(_square_hips(ref)) == pytest.approx(0.0, abs=1e-6)
    (r,) = compare(["hip_turn"], _square_hips(ref), ref)
    assert (r.body_part, r.fix, r.tier) == ("hips", "twist", "Miss")


def test_short_steps_miss_the_reach():
    ref = _ref()
    assert leg_reach(_short_steps(ref)) < leg_reach(ref)
    (r,) = compare(["leg_reach"], _short_steps(ref), ref)
    assert (r.body_part, r.fix, r.tier) == ("leg", "reach", "Miss")


def test_tiers_only_flag_too_little():
    assert tier_for(1.8) == "Perfect"
    assert tier_for(PERFECT_RATIO) == "Perfect"
    assert tier_for(CLOSE_RATIO) == "Close"
    assert tier_for(CLOSE_RATIO - 0.01) == "Miss"


def test_unknown_or_absent_features_are_skipped():
    ref = _ref()
    flat = ref.copy()
    flat[:, [L_HIP, R_HIP], 0] = 0.0   # reference shows no hip width at all
    assert compare(["no_such_feature"], ref, ref) == []
    assert compare(["hip_turn"], ref, flat) == []


def _aligned(n):
    path = tuple(range(n))
    return DTWResult(True, 1.0, 0.1, "br_01", 2000.0, n, n, live_path=path, reference_path=path)


def test_scorer_adds_features_from_meta_and_names_the_fix():
    ref = _ref()
    scorer = Scorer(REF_DIR)
    result = scorer.score_rep(_aligned(len(ref)), list(_square_hips(ref)))
    by_name = {r.joint_name: r for r in result.results}
    assert by_name["hips"].tier == "Miss" and by_name["hips"].fix_direction == "twist"
    assert by_name["leg"].tier == "Perfect"
    assert scorer.stats().focus_joint == "hips"
    assert scorer.stats().focus_fix == "twist"

    payload = _score_payload(result)
    hips = next(r for r in payload["results"] if r["joint"] == "hips")
    assert hips["fix"] == "twist" and not payload["clean"]


def test_moves_without_features_are_unchanged(tmp_path):
    for f in ("br_03.npy", "br_03_meta.json"):
        shutil.copy(REF_DIR / f, tmp_path / f)
    meta = json.loads((tmp_path / "br_03_meta.json").read_text())
    assert "features" not in meta
    ref = np.load(tmp_path / "br_03.npy").reshape(-1, 33, 3)
    path = tuple(range(len(ref)))
    dtw = DTWResult(True, 1.0, 0.1, "br_03", 2000.0, len(ref), len(ref), live_path=path, reference_path=path)
    result = Scorer(tmp_path).score_rep(dtw, list(ref))
    assert {r.joint_name for r in result.results} == {"left_elbow", "right_elbow", "left_knee", "right_knee"}


def _rep(joint, fix, tier="Miss"):
    return {"results": [{"joint": joint, "tier": tier, "diff": 60.0, "fix": fix}]}


@pytest.mark.parametrize("joint,fix", [("hips", "twist"), ("leg", "reach")])
def test_feature_cues_are_grounded(joint, fix):
    rep = _rep(joint, fix)
    facts = live_facts("BR 01", rep)
    for _ in range(20):
        cue = fallback_cue(rep)
        assert joint in cue
        assert ungrounded_claims(cue, facts) == []
    close = _rep(joint, fix, "Close")
    for _ in range(20):
        assert ungrounded_claims(fallback_cue(close), live_facts("BR 01", close)) == []
    prompt = _build_prompt("BR 01", rep, facts=facts)
    assert f"their {joint} needs to" in prompt


def test_twist_is_refused_unless_measured():
    elbow = live_facts("BR 01", _rep("left elbow", "straighten"))
    assert ungrounded_claims("Twist more", elbow)
    assert ungrounded_claims("Twist your hips", elbow)
    hips = live_facts("BR 01", _rep("hips", "twist"))
    assert ungrounded_claims("Twist those hips more", hips) == []


def test_reference_through_live_normalization_scores_clean():
    """Regression: live frames are rescaled per frame, which once erased the hip twist."""
    ref = _ref()
    results = compare(["hip_turn", "leg_reach"], _live(ref), ref)
    assert [r.tier for r in results] == ["Perfect", "Perfect"]
    assert all(0.9 <= r.ratio <= 1.1 for r in results)


def test_square_hips_still_miss_after_live_normalization():
    ref = _ref()
    (r,) = compare(["hip_turn"], _live(_square_hips(ref)), ref)
    assert r.tier == "Miss"


def test_short_steps_still_miss_after_live_normalization():
    ref = _ref()
    (r,) = compare(["leg_reach"], _live(_short_steps(ref)), ref)
    assert r.tier == "Miss"


def test_scorer_gives_a_normalized_reference_all_perfect():
    scorer = Scorer(REF_DIR)
    ref = _ref()
    result = scorer.score_rep(_aligned(len(ref)), list(_live(ref)))
    assert result is not None
    assert [r.tier for r in result.results] == ["Perfect"] * len(result.results)
