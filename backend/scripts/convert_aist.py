"""
Convert AIST++ 3D keypoints to .npy reference moves for Breakdown.

AIST++ and MediaPipe describe bodies differently, so references have to be
converted into the same coordinate system the live camera produces, or DTW
ends up comparing the dancer against an upside-down, sideways reference:

  AIST++ (world, cm)                 MediaPipe live (what we match)
  y points up                        y points down (image coordinates)
  dancer faces any direction         dancer faces the camera
  17 COCO joints                     33 landmarks

Each reference is also cut to a short window (default 3 seconds, the most
active part of the clip) instead of squeezing a whole routine into 60 frames,
and the real duration is saved so playback speed and "power" are accurate.

Usage:
    # add new moves
    python scripts/convert_aist.py --keypoints-dir ~/keypoints3d --target-count 10

    # regenerate existing AIST++ moves in place, keeping names, descriptions,
    # tips, and keyframes you've added to their meta files
    python scripts/convert_aist.py --keypoints-dir ~/keypoints3d --rebuild
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

COCO_TO_MEDIAPIPE: dict[int, int] = {
    0:  0,    # nose
    1:  2,    # left_eye
    2:  5,    # right_eye
    3:  7,    # left_ear
    4:  8,    # right_ear
    5:  11,   # left_shoulder
    6:  12,   # right_shoulder
    7:  13,   # left_elbow
    8:  14,   # right_elbow
    9:  15,   # left_wrist
    10: 16,   # right_wrist
    11: 23,   # left_hip
    12: 24,   # right_hip
    13: 25,   # left_knee
    14: 26,   # right_knee
    15: 27,   # left_ankle
    16: 28,   # right_ankle
}
MAPPED = sorted(COCO_TO_MEDIAPIPE.values())

AIST_FPS = 60.0
TARGET_FRAMES = 60
DEFAULT_SECONDS = 3.0
NOSE, L_SH, R_SH, L_HIP, R_HIP = 0, 11, 12, 23, 24
MAX_VALID_VALUE = 35.0     # anything beyond this is corrupted data
COORDS_VERSION = 2         # bump when the coordinate conversion changes


def load_keypoints(pkl_path: Path) -> np.ndarray:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    key = "keypoints3d_optim" if "keypoints3d_optim" in data else "keypoints3d"
    return data[key].astype(np.float32)


def coco_to_mediapipe(coco_joints: np.ndarray) -> np.ndarray:
    """Place the 17 COCO joints at their MediaPipe indices; the rest stay 0."""
    mp = np.zeros((coco_joints.shape[0], 33, 3), dtype=np.float32)
    for coco_idx, mp_idx in COCO_TO_MEDIAPIPE.items():
        mp[:, mp_idx, :] = coco_joints[:, coco_idx, :]
    return mp


def most_active_window(seq: np.ndarray, frames: int) -> int:
    """Start index of the window with the most limb movement."""
    if seq.shape[0] <= frames:
        return 0
    limbs = seq[:, [15, 16, 27, 28], :]
    speed = np.linalg.norm(np.diff(limbs, axis=0), axis=-1).sum(axis=1)
    csum = np.concatenate([[0.0], np.cumsum(speed)])
    totals = csum[frames - 1:] - csum[: len(csum) - frames + 1]
    return int(np.argmax(totals))


def to_camera_frame(seq: np.ndarray) -> np.ndarray:
    """AIST++ world coordinates -> MediaPipe-style camera coordinates.

    Output matches live MediaPipe data: hip-centered, y down, dancer facing
    the camera (nose closer than the hips, so negative z), and the left
    shoulder at larger x than the right (how MediaPipe labels a person
    facing the camera). Scaled by the clip's hip width.
    """
    out = seq.copy()
    out[:, MAPPED] -= ((out[:, L_HIP] + out[:, R_HIP]) / 2)[:, None, :]

    # y up -> y down
    out[:, MAPPED, 1] *= -1

    # Turn around the vertical axis so the shoulders spread across x.
    sh = out[:, L_SH, :] - out[:, R_SH, :]
    best, best_w = 0.0, -1.0
    for deg in range(0, 180, 2):
        a = np.radians(deg)
        w = np.abs(sh[:, 0] * np.cos(a) + sh[:, 2] * np.sin(a)).mean()
        if w > best_w:
            best, best_w = a, w
    c, s = np.cos(best), np.sin(best)
    x, z = out[:, MAPPED, 0].copy(), out[:, MAPPED, 2].copy()
    out[:, MAPPED, 0] = x * c + z * s
    out[:, MAPPED, 2] = -x * s + z * c

    # Face the camera: a 180 degree turn flips both x and z.
    if out[:, NOSE, 2].mean() > 0:
        out[:, MAPPED, 0] *= -1
        out[:, MAPPED, 2] *= -1

    # MediaPipe labels a camera-facing person's left side at larger x.
    # If that's reversed, mirror x (the live feed is mirrored too).
    if (out[:, L_SH, 0] - out[:, R_SH, 0]).mean() < 0:
        out[:, MAPPED, 0] *= -1

    # One scale for the whole clip, so spins don't blow up the numbers
    # the way a per-frame 2D hip width would.
    hip_width = np.median(np.linalg.norm(out[:, L_HIP] - out[:, R_HIP], axis=-1))
    out /= max(float(hip_width), 1e-6)
    return out


def resample(seq: np.ndarray, target: int = TARGET_FRAMES) -> np.ndarray:
    T = seq.shape[0]
    if T == target:
        return seq
    indices = np.linspace(0, T - 1, target)
    lo = np.floor(indices).astype(int)
    hi = np.minimum(lo + 1, T - 1)
    t = (indices - lo)[:, None, None]
    return (seq[lo] * (1 - t) + seq[hi] * t).astype(np.float32)


def convert_sequence(raw: np.ndarray, seconds: float) -> tuple[np.ndarray, dict]:
    """Returns (flat (60, 99) array, meta fields describing the clip)."""
    mp = coco_to_mediapipe(raw)
    window = min(int(round(seconds * AIST_FPS)), mp.shape[0])
    start = most_active_window(mp, window)
    clip = to_camera_frame(mp[start:start + window])
    flat = resample(clip, TARGET_FRAMES).reshape(TARGET_FRAMES, -1)
    return flat, {
        "coords_version": COORDS_VERSION,
        "source_fps": AIST_FPS,
        "clip_start_s": round(start / AIST_FPS, 3),
        "duration_s": round(window / AIST_FPS, 3),
        "original_frames": int(raw.shape[0]),
    }


def default_meta(move_name: str, pkl_name: str, dtw_threshold: float) -> dict:
    return {
        "name": move_name,
        "dtw_threshold": dtw_threshold,
        "source": "AIST++",
        "original_file": pkl_name,
        "keyframes": [{
            "frame": 59,
            "joints": [
                {"joint_name": "left_elbow", "joint_triplet": [11, 13, 15],
                 "target_angle": 160.0, "threshold": 30.0},
                {"joint_name": "right_elbow", "joint_triplet": [12, 14, 16],
                 "target_angle": 160.0, "threshold": 30.0},
            ],
        }],
    }


def remap_keyframes(meta: dict, clip_meta: dict, old_frames: int) -> list[dict]:
    """Map via source frames; -1 disables scoring while retaining joint edits."""
    # Metadata times are rounded to milliseconds; recover integer source frames
    # before mapping the endpoint-inclusive samples produced by resample().
    fps = clip_meta["source_fps"]
    new_start = round(clip_meta["clip_start_s"] * fps)
    new_span = round(clip_meta["duration_s"] * fps) - 1
    try:
        if not any(k in meta for k in ("coords_version", "clip_start_s", "duration_s")):
            # The original converter resampled the entire source sequence.
            old_start, old_span = 0, clip_meta["original_frames"] - 1
        else:
            old_start = round(meta["clip_start_s"] * fps)
            old_span = round(meta["duration_s"] * fps) - 1
        timing_known = old_start >= 0 and old_span >= 0 and old_frames > 0
    except (KeyError, TypeError, ValueError, OverflowError):
        timing_known = False

    keyframes = []
    for entry in meta.get("keyframes", []):
        frame = entry.get("frame", -1)
        mapped = -1
        if timing_known and isinstance(frame, int) and 0 <= frame < old_frames:
            source_frame = old_start + frame * old_span / max(old_frames - 1, 1)
            if new_start <= source_frame <= new_start + new_span:
                mapped = round((source_frame - new_start) * (TARGET_FRAMES - 1) / max(new_span, 1))
        keyframes.append({**entry, "frame": mapped})
    return keyframes


def convert_file(pkl_path: Path, output_dir: Path, move_name: str,
                 dtw_threshold: float, seconds: float, existing_meta: dict | None = None) -> bool:
    try:
        flat, clip_meta = convert_sequence(load_keypoints(pkl_path), seconds)
        max_val = float(abs(flat).max())
        if max_val > MAX_VALID_VALUE:
            print(f"  SKIPPED, corrupted data (max={max_val:.1f})")
            return False

        # Preserve user edits, but align keyframes with the rebuilt clip.
        meta = dict(existing_meta) if existing_meta is not None else default_meta(
            move_name, pkl_path.name, dtw_threshold)
        if existing_meta is not None:
            old_path = output_dir / f"{move_name}.npy"
            old_frames = np.load(old_path, mmap_mode="r").shape[0] if old_path.exists() else TARGET_FRAMES
            meta["keyframes"] = remap_keyframes(existing_meta, clip_meta, old_frames)
        meta.update(clip_meta)

        np.save(output_dir / f"{move_name}.npy", flat)
        (output_dir / f"{move_name}_meta.json").write_text(json.dumps(meta, indent=2))
        print(f"  OK  {move_name}.npy  {clip_meta['duration_s']}s from "
              f"{clip_meta['clip_start_s']}s  max={max_val:.2f}")
        return True
    except Exception as e:
        print(f"  FAILED {pkl_path.name}: {e}")
        return False


def rebuild(args: argparse.Namespace) -> None:
    """Re-convert every AIST++ move in the output folder from its original file."""
    done = 0
    for meta_path in sorted(args.output.glob("*_meta.json")):
        meta = json.loads(meta_path.read_text())
        if meta.get("source") != "AIST++" or not meta.get("original_file"):
            continue
        name = meta_path.name[: -len("_meta.json")]
        pkl = args.keypoints_dir / meta["original_file"]
        print(f"[rebuild] {name}  <-  {pkl.name}")
        if not pkl.exists():
            print(f"  MISSING {pkl}")
            continue
        done += convert_file(pkl, args.output, name, meta.get("dtw_threshold", args.threshold),
                             args.seconds, existing_meta=meta)
    print(f"\nRebuilt {done} moves in {args.output}/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keypoints-dir", type=Path, default=Path("/Users/jordanlin/keypoints3d"))
    parser.add_argument("--output", type=Path, default=Path("reference_moves"))
    parser.add_argument("--genre", default="BR")
    parser.add_argument("--target-count", type=int, default=10,
                        help="How many clean sequences to collect")
    parser.add_argument("--skip", type=int, default=0,
                        help="Skip first N sequences (already processed)")
    parser.add_argument("--threshold", type=float, default=2000.0)
    parser.add_argument("--seconds", type=float, default=DEFAULT_SECONDS,
                        help="Length of the clip to keep, taken from the most active part")
    parser.add_argument("--rebuild", action="store_true",
                        help="Re-convert existing AIST++ moves in place, keeping their meta edits")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.rebuild:
        rebuild(args)
        return

    all_files = sorted(args.keypoints_dir.glob(f"g{args.genre}_*.pkl"))
    if not all_files:
        raise SystemExit(f"No files found for genre '{args.genre}'")

    existing = len(list(args.output.glob(f"{args.genre.lower()}_*.npy")))
    next_num = existing + 1
    files_to_try = all_files[args.skip:]
    print(f"Found {len(all_files)} total {args.genre} sequences")
    print(f"Skipping first {args.skip}, trying up to {len(files_to_try)} more")
    print(f"Target: {args.target_count} clean sequences (have {existing} already)\n")

    converted = 0
    for tried, pkl_path in enumerate(files_to_try, 1):
        if converted >= args.target_count:
            break
        move_name = f"{args.genre.lower()}_{next_num:02d}"
        print(f"[trying {tried}] {pkl_path.name}  ->  {move_name}")
        if convert_file(pkl_path, args.output, move_name, args.threshold, args.seconds):
            converted += 1
            next_num += 1

    print(f"\nDone, added {converted} clean sequences ({existing + converted} total in {args.output}/)")


if __name__ == "__main__":
    main()
