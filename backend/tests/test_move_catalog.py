"""Tests for the move library: measured stats, meta overrides, previews."""

import json

import numpy as np

from app.move_catalog import NECK, load_catalog, preview, up_sign

T = 20


def standing(y_up=True, arm_speed=0.0):
    """Normalized standing figure: hips at 0, head 3 above, ankles 3 below."""
    seq = np.zeros((T, 33, 3), dtype=np.float32)
    sign = 1 if y_up else -1
    seq[:, 0, 1] = 3 * sign                     # nose
    seq[:, [11, 12], 1] = 2 * sign              # shoulders
    seq[:, [27, 28], 1] = -3 * sign             # ankles
    seq[:, [15, 16], 0] = np.arange(T)[:, None] * arm_speed
    return seq


def save(tmp_path, name, seq, **meta):
    np.save(tmp_path / f"{name}.npy", seq.reshape(T, -1))
    (tmp_path / f"{name}_meta.json").write_text(json.dumps({"name": name, **meta}))


def test_detects_coordinate_direction():
    assert up_sign(standing(y_up=True)) == 1
    assert up_sign(standing(y_up=False)) == -1


def test_upright_move_has_no_inversion_or_floor(tmp_path):
    save(tmp_path, "a", standing())
    m = load_catalog(tmp_path)[0]
    assert m.stats["inversion"] == 0 and m.stats["floor"] == 0


def test_inverted_frames_are_measured(tmp_path):
    seq = standing()
    seq[T // 2:, 0, 1] = -2                     # head below hips for half the move
    save(tmp_path, "headspin", seq)
    m = load_catalog(tmp_path)[0]
    assert m.stats["inversion"] == 50
    assert any(r["kind"] == "sensors" for r in m.requirements)


def test_power_is_relative_to_fastest_move(tmp_path):
    save(tmp_path, "slow", standing(arm_speed=0.1))
    save(tmp_path, "fast", standing(arm_speed=0.4))
    stats = {m.id: m.stats["power"] for m in load_catalog(tmp_path)}
    assert stats["fast"] == 100 and stats["slow"] == 25


def test_meta_overrides_and_hidden_moves(tmp_path):
    save(tmp_path, "six_step", standing(), display_name="Six-step",
         description="Footwork around your hands.", difficulty="beginner", tips=["Stay low"])
    save(tmp_path, "calib", standing(), hidden=True)
    moves = load_catalog(tmp_path)
    assert [m.id for m in moves] == ["six_step"]
    m = moves[0]
    assert m.display_name == "Six-step"
    assert m.difficulty == "Beginner" and not m.difficulty_measured
    assert m.tips == ["Stay low"]


def test_preview_is_screen_space_with_neck(tmp_path):
    save(tmp_path, "a", standing(y_up=True))
    p = preview(tmp_path, "a")
    head_y = p["frames"][0][0][1]
    ankle_y = p["frames"][0][11][1]
    assert head_y < ankle_y                     # screen y grows downward
    assert len(p["frames"][0]) == NECK + 1


def test_preview_rejects_unknown_and_path_tricks(tmp_path):
    assert preview(tmp_path, "missing") is None
    assert preview(tmp_path, "../secrets") is None


def test_preview_turns_a_side_view_to_face_the_viewer(tmp_path):
    seq = standing()
    seq[:, 11, 2], seq[:, 12, 2] = 1.0, -1.0    # shoulders spread along depth (side view)
    save(tmp_path, "side", seq)
    f = preview(tmp_path, "side")["frames"][0]
    assert abs(f[1][0] - f[2][0]) > 1.5         # shoulders now spread across the screen


def test_preview_depth_puts_the_face_toward_the_viewer(tmp_path):
    seq = standing()
    seq[:, 0, 2] = -0.5                          # nose points toward -z
    save(tmp_path, "facing", seq)
    f = preview(tmp_path, "facing")["frames"][0]
    assert f[0][2] > f[NECK][2]                  # nose ends up nearer than the neck


def test_preview_includes_depth_per_joint(tmp_path):
    seq = standing()
    seq[:, 15, 2] = 2.0                          # left wrist reaching toward the camera
    save(tmp_path, "reach", seq)
    f = preview(tmp_path, "reach")["frames"][0]
    assert all(len(j) == 3 for j in f)
    assert f[5][2] != f[6][2]                    # left and right wrists at different depths


def test_crouching_is_not_floor_but_sitting_is(tmp_path):
    crouch = standing()
    crouch[:, 0, 1] = 1.0                        # head bent down near hip level
    sit = standing()
    sit[:, [27, 28], 1] = -0.3                   # feet at hip level: hips on the ground
    save(tmp_path, "crouch", crouch)
    save(tmp_path, "sit", sit)
    stats = {m.id: m.stats["floor"] for m in load_catalog(tmp_path)}
    assert stats["crouch"] == 0
    assert stats["sit"] == 100


def test_empty_joints_never_count_as_lowest_point(tmp_path):
    seq = standing()                             # unmapped joints are all 0 = hip level
    save(tmp_path, "a", seq)
    assert load_catalog(tmp_path)[0].stats["floor"] == 0


def test_power_uses_real_duration(tmp_path):
    save(tmp_path, "short", standing(arm_speed=0.2), duration_s=1.0)
    save(tmp_path, "long", standing(arm_speed=0.2), duration_s=4.0)
    stats = {m.id: m.stats["power"] for m in load_catalog(tmp_path)}
    assert stats["short"] == 100 and stats["long"] == 25   # same motion, 4x the time


def test_preview_plays_in_real_time(tmp_path):
    save(tmp_path, "a", standing(), duration_s=2.0)
    assert preview(tmp_path, "a")["fps"] == round((T - 1) / 2.0, 3)
