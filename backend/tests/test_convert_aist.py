"""Tests for the AIST++ converter: references must match live MediaPipe coordinates."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "convert_aist", Path(__file__).resolve().parent.parent / "scripts" / "convert_aist.py")
conv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(conv)


def aist_person(facing_deg, frames=240, sway=0.0):
    """COCO-17 skeleton in AIST++ world style: centimeters, y up, facing any way."""
    f = np.array([np.cos(np.radians(facing_deg)), 0, np.sin(np.radians(facing_deg))])
    left = np.array([f[2], 0, -f[0]])            # anatomical left for a person facing f
    up = np.array([0, 1, 0])
    base = np.array([100.0, 0, -50.0])
    pts = {
        0: 165 * up + 10 * f,                     # nose, in front of the face
        1: 170 * up + 3 * left + 8 * f, 2: 170 * up - 3 * left + 8 * f,
        3: 168 * up + 7 * left, 4: 168 * up - 7 * left,
        5: 140 * up + 20 * left, 6: 140 * up - 20 * left,
        7: 115 * up + 25 * left, 8: 115 * up - 25 * left,
        9: 90 * up + 27 * left, 10: 90 * up - 27 * left,
        11: 90 * up + 15 * left, 12: 90 * up - 15 * left,
        13: 50 * up + 15 * left, 14: 50 * up - 15 * left,
        15: 5 * up + 15 * left, 16: 5 * up - 15 * left,
    }
    seq = np.zeros((frames, 17, 3), dtype=np.float32)
    for t in range(frames):
        for j, p in pts.items():
            seq[t, j] = base + p
        seq[t, 9] += sway * np.sin(t / 10) * up   # left wrist waves
    return seq


@pytest.mark.parametrize("facing", [0, 37, 90, 180, 250])
def test_output_matches_live_mediapipe_conventions(facing):
    flat, _ = conv.convert_sequence(aist_person(facing), seconds=3.0)
    f = flat.reshape(60, 33, 3)[0]
    assert f[0, 1] < 0                          # head above hips (y down)
    assert f[27, 1] > 0                         # ankles below hips
    assert f[0, 2] < 0                          # facing the camera
    assert f[11, 0] > f[12, 0]                  # left shoulder at larger x
    assert np.linalg.norm(f[23] - f[24]) == pytest.approx(1.0, abs=1e-3)


def test_unmapped_joints_stay_zero():
    flat, _ = conv.convert_sequence(aist_person(60), seconds=3.0)
    f = flat.reshape(60, 33, 3)
    unmapped = [i for i in range(33) if i not in conv.MAPPED]
    assert np.all(f[:, unmapped] == 0)


def test_keeps_a_short_window_and_records_real_duration():
    raw = aist_person(0, frames=1200)           # a 20 second routine
    raw[600:780, 9] += 40 * np.sin(np.arange(180) / 3)[:, None]   # busy 3s in the middle
    _, meta = conv.convert_sequence(raw, seconds=3.0)
    assert meta["duration_s"] == 3.0
    assert meta["original_frames"] == 1200
    assert 8.0 <= meta["clip_start_s"] <= 10.5  # found the active part


def test_rebuild_keeps_user_edits(tmp_path, monkeypatch):
    monkeypatch.setattr(conv, "load_keypoints", lambda p: aist_person(0))
    meta = conv.default_meta("br_01", "gBR_x.pkl", 2000.0)
    meta.update(display_name="Six-step", tips=["Stay low"])
    assert conv.convert_file(Path("gBR_x.pkl"), tmp_path, "br_01", 2000.0, 3.0, existing_meta=meta)
    import json
    saved = json.loads((tmp_path / "br_01_meta.json").read_text())
    assert saved["display_name"] == "Six-step" and saved["tips"] == ["Stay low"]
    assert saved["coords_version"] == conv.COORDS_VERSION
