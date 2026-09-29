"""Move library for the frontend: descriptions, stats, requirements, previews.

Stats are measured from each move's reference keypoints, not hand-written, so
the skill card always matches what the move actually contains. Anything the
data can't tell us (a real name, a description, tips) comes from the move's
meta JSON, and the card falls back to plain defaults when it's missing.

Optional meta fields a move can set:
    display_name   "Six-step"
    description    "Footwork in a circle around your hands..."
    difficulty     "Beginner" | "Intermediate" | "Advanced"  (overrides the measured tier)
    tips           ["Keep your hips low", ...]
    hidden         true  (leave it out of the library, e.g. calibration poses)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 0, 11, 12, 13, 14, 15, 16
L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN = 23, 24, 25, 26, 27, 28

# Joints drawn in the preview, in output order. Index 13 is a computed neck.
PREVIEW_JOINTS = [NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR,
                  L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN]
NECK = len(PREVIEW_JOINTS)
PREVIEW_BONES = [
    [0, NECK], [NECK, 1], [NECK, 2], [1, 2],
    [1, 3], [3, 5], [2, 4], [4, 6],
    [1, 7], [2, 8], [7, 8],
    [7, 9], [9, 11], [8, 10], [10, 12],
]
PREVIEW_FPS = 20

LOW_HEAD = 1.2        # head within this many hip-widths above the hips = low / on the floor
HIDDEN_BY_DEFAULT = {"t_pose"}


@dataclass
class MoveInfo:
    id: str
    display_name: str
    source: str
    description: str
    difficulty: str
    difficulty_measured: bool
    stats: dict[str, int]
    requirements: list[dict[str, str]]
    tips: list[str]
    frames: int
    original: str = ""
    difficulty_override: Optional[str] = field(default=None, repr=False)
    _power_raw: float = field(default=0.0, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            # "name" kept for older frontends
            "name": self.display_name.upper(),
            "display_name": self.display_name,
            "source": self.source,
            "description": self.description,
            "difficulty": self.difficulty,
            "difficulty_measured": self.difficulty_measured,
            "stats": self.stats,
            "requirements": self.requirements,
            "tips": self.tips,
            "frames": self.frames,
            "original": self.original,
        }


def _load(npy: Path) -> np.ndarray:
    seq = np.load(npy).astype(np.float32)
    return seq.reshape(seq.shape[0], 33, 3)


def up_sign(seq: np.ndarray) -> float:
    """+1 if this sequence uses y-up coordinates, -1 if y-down.

    AIST++ references are y-up; moves recorded through MediaPipe are y-down
    (image coordinates). Standing people have their nose above their ankles,
    so compare the two on average.
    """
    return 1.0 if seq[:, NOSE, 1].mean() > seq[:, [L_AN, R_AN], 1].mean() else -1.0


def measure(seq: np.ndarray) -> tuple[float, float, float]:
    """(inversion fraction, floor fraction, raw power) for one sequence."""
    y_up = seq[..., 1] * up_sign(seq)
    head = y_up[:, NOSE]                          # hips are at 0 after normalization
    inversion = float(np.mean(head < 0))          # head below the hips
    floor = float(np.mean(head < LOW_HEAD))       # head low: crouched, floor, or inverted
    limbs = seq[:, [L_WR, R_WR, L_AN, R_AN], :2]
    speed = np.linalg.norm(np.diff(limbs, axis=0), axis=-1)
    power = float(speed.mean()) if len(speed) else 0.0
    return inversion, floor, power


def _tier(stats: dict[str, int]) -> str:
    score = 0.4 * stats["power"] + 0.35 * stats["inversion"] + 0.25 * stats["floor"]
    if score < 35:
        return "Beginner"
    if score < 60:
        return "Intermediate"
    return "Advanced"


def _requirements(stats: dict[str, int]) -> list[dict[str, str]]:
    reqs = [
        {"kind": "space", "text": "Clear floor, about 2 by 2 meters"},
        {"kind": "camera", "text": "Whole body visible to the camera"},
    ]
    if stats["floor"] >= 30:
        reqs.append({"kind": "floor", "text": "A smooth floor you can slide on"})
    if stats["inversion"] >= 15:
        reqs.append({"kind": "warmup", "text": "Wrists and shoulders warmed up. You'll be upside down."})
    if stats["inversion"] >= 15 or stats["floor"] >= 30:
        reqs.append({"kind": "sensors", "text": "Wear the sensors for freeze grading"})
    return reqs


def load_catalog(reference_dir: Path) -> list[MoveInfo]:
    moves: list[MoveInfo] = []
    for npy in sorted(Path(reference_dir).glob("*.npy")):
        meta_path = npy.with_name(f"{npy.stem}_meta.json")
        meta: dict[str, Any] = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
        if meta.get("hidden", npy.stem in HIDDEN_BY_DEFAULT):
            continue

        seq = _load(npy)
        inversion, floor, power = measure(seq)
        source = meta.get("source", "custom")
        moves.append(MoveInfo(
            id=npy.stem,
            display_name=meta.get("display_name") or npy.stem.replace("_", " ").upper(),
            source=source,
            description=meta.get("description") or (
                "A breaking sequence from the AIST++ dance dataset."
                if source == "AIST++" else "A custom recorded move."),
            difficulty="",
            difficulty_measured=True,
            stats={"inversion": round(inversion * 100), "floor": round(floor * 100)},
            requirements=[],
            tips=list(meta.get("tips", [])),
            frames=int(seq.shape[0]),
            original=meta.get("original_file", ""),
            difficulty_override=meta.get("difficulty"),
            _power_raw=power,
        ))

    # Power is relative to the fastest move in the library, so it reads 0-100.
    top = max((m._power_raw for m in moves), default=0.0) or 1.0
    for m in moves:
        m.stats = {"power": round(100 * m._power_raw / top), **m.stats}
        if m.difficulty_override:
            m.difficulty = str(m.difficulty_override).capitalize()
            m.difficulty_measured = False
        else:
            m.difficulty = _tier(m.stats)
        m.requirements = _requirements(m.stats)
    return moves


def facing_angle(seq: np.ndarray) -> float:
    """Horizontal rotation (radians) that shows the dancer most front-on.

    Reference data is 3D, and a dancer filmed from the side collapses into a
    stick when projected straight onto x/y. Pick the rotation around the
    vertical axis that makes the shoulders look widest over the whole move.
    """
    shoulders = seq[:, L_SH, [0, 2]] - seq[:, R_SH, [0, 2]]   # (T, 2) in the x/z plane
    best, best_width = 0.0, -1.0
    for deg in range(0, 180, 5):
        a = np.radians(deg)
        width = np.abs(shoulders[:, 0] * np.cos(a) + shoulders[:, 1] * np.sin(a)).mean()
        if width > best_width:
            best, best_width = a, width
    return best


def preview(reference_dir: Path, move_id: str) -> Optional[dict[str, Any]]:
    """Skeleton frames for the animated preview, in screen coordinates (y down)."""
    npy = Path(reference_dir) / f"{move_id}.npy"
    if not npy.exists() or "/" in move_id or ".." in move_id:
        return None
    seq = _load(npy)
    s = up_sign(seq)
    a = facing_angle(seq)
    joints = seq[:, PREVIEW_JOINTS, :]
    x = joints[..., 0] * np.cos(a) + joints[..., 2] * np.sin(a)
    y = joints[..., 1] * -s                        # y-up -> screen y-down
    pts = np.stack([x, y], axis=-1)
    neck = (pts[:, 1] + pts[:, 2]) / 2
    pts = np.concatenate([pts, neck[:, None, :]], axis=1)
    return {
        "id": move_id,
        "fps": PREVIEW_FPS,
        "bones": PREVIEW_BONES,
        "frames": np.round(pts, 3).tolist(),
    }
