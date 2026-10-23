"""Vision layer — MediaPipe pose estimation, keypoint extraction, hip-center normalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover
    mp = None  # type: ignore

NUM_LANDMARKS = 33
MIN_TORSO_SCALE = 1e-6

# MediaPipe Pose landmark indices
L_HIP, R_HIP = 23, 24
L_SHOULDER, R_SHOULDER = 11, 12


@dataclass(frozen=True)
class PoseFrame:
    """One frame of body-relative pose data.

    landmarks: (33, 3) float32 — x, y, z normalized (hip origin, torso-scaled).
    visibility: (33,) float32 — per-landmark visibility in [0, 1].
    raw_landmarks: (33, 3) float32 — pre-normalization image-space coords (optional).
    captured_at: when the browser grabbed the frame, in seconds (optional); lets a
        replay on the phone find the picture this pose came from.
    """

    landmarks: np.ndarray
    visibility: np.ndarray
    raw_landmarks: Optional[np.ndarray] = None
    captured_at: Optional[float] = None


def extract_keypoints(pose_landmarks) -> tuple[np.ndarray, np.ndarray]:
    """Extract (33, 3) positions and (33,) visibility from MediaPipe landmark list."""
    raw = np.array([[p.x, p.y, p.z] for p in pose_landmarks], dtype=np.float32)
    visibility = np.array([p.visibility for p in pose_landmarks], dtype=np.float32)
    return raw, visibility


def hip_center(landmarks: np.ndarray) -> np.ndarray:
    """Midpoint between left and right hip."""
    return (landmarks[L_HIP] + landmarks[R_HIP]) / 2.0


def torso_scale(landmarks: np.ndarray) -> float:
    """Hip-to-hip distance in the xy plane; floored to avoid divide-by-zero."""
    left = landmarks[L_HIP, :2]
    right = landmarks[R_HIP, :2]
    return max(float(np.linalg.norm(left - right)), MIN_TORSO_SCALE)


def normalize_landmarks(raw: np.ndarray) -> np.ndarray:
    """Body-size-agnostic coords: hip-center origin, scale by torso width."""
    out = np.asarray(raw, dtype=np.float32).copy()
    out -= hip_center(out)
    out /= torso_scale(out)
    return out


def is_pose_reliable(frame: PoseFrame, min_visibility: float = 0.5) -> bool:
    """Return True only if key landmarks are visible above threshold."""
    key_landmarks = [L_HIP, R_HIP, L_SHOULDER, R_SHOULDER]
    return all(frame.visibility[i] >= min_visibility for i in key_landmarks)


def _landmark_xy_to_pixel(x: float, y: float, width: int, height: int) -> tuple[int, int]:
    """MediaPipe normalized image coords → pixel (x, y)."""
    return int(x * width), int(y * height)


def hip_bbox_pixels(
    raw_landmarks: np.ndarray,
    frame_width: int,
    frame_height: int,
    *,
    padding: float = 0.6,
) -> tuple[int, int, int, int]:
    """Bounding box (x1, y1, x2, y2) around the hips in pixel space."""
    left = raw_landmarks[L_HIP, :2]
    right = raw_landmarks[R_HIP, :2]
    center = hip_center(raw_landmarks)[:2]

    hip_span = float(np.linalg.norm(left - right))
    pad = max(hip_span * padding, 0.04)

    x1_n = min(left[0], right[0], center[0]) - pad
    x2_n = max(left[0], right[0], center[0]) + pad
    y1_n = min(left[1], right[1], center[1]) - pad * 1.2
    y2_n = max(left[1], right[1], center[1]) + pad * 1.2

    x1, y1 = _landmark_xy_to_pixel(x1_n, y1_n, frame_width, frame_height)
    x2, y2 = _landmark_xy_to_pixel(x2_n, y2_n, frame_width, frame_height)

    x1 = max(0, min(x1, frame_width - 1))
    y1 = max(0, min(y1, frame_height - 1))
    x2 = max(0, min(x2, frame_width - 1))
    y2 = max(0, min(y2, frame_height - 1))
    return x1, y1, x2, y2


def hip_center_pixels(
    raw_landmarks: np.ndarray,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int]:
    """Hip midpoint in pixel coordinates."""
    c = hip_center(raw_landmarks)
    return _landmark_xy_to_pixel(float(c[0]), float(c[1]), frame_width, frame_height)


def draw_hip_tracker(
    frame: np.ndarray,
    pose: PoseFrame,
    *,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    trail: Optional[list[tuple[int, int]]] = None,
    show_label: bool = True,
) -> tuple[int, int, int, int] | None:
    """Draw hip box, center dot, and optional trail (requires ``keep_raw=True``)."""
    import cv2

    if pose.raw_landmarks is None:
        return None

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = hip_bbox_pixels(pose.raw_landmarks, w, h)
    cx, cy = hip_center_pixels(pose.raw_landmarks, w, h)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    cv2.circle(frame, (cx, cy), 6, color, -1)
    cv2.circle(frame, (cx, cy), 8, (255, 255, 255), 1)

    if trail is not None:
        trail.append((cx, cy))
        if len(trail) > 45:
            del trail[0]
        for i in range(1, len(trail)):
            fade = int(255 * i / len(trail))
            shade = (0, min(fade, 255), 0)
            cv2.line(frame, trail[i - 1], trail[i], shade, 2)

    if show_label:
        norm = pose.landmarks[L_HIP, :2]
        label = f"hip norm ({norm[0]:+.2f}, {norm[1]:+.2f})"
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 8, 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    return x1, y1, x2, y2


def mediapipe_to_poseframe(
    results,
    *,
    keep_raw: bool = False,
) -> Optional[PoseFrame]:
    """Convert a MediaPipe ``process()`` result into a :class:`PoseFrame`."""
    if results.pose_landmarks is None:
        return None

    raw, visibility = extract_keypoints(results.pose_landmarks.landmark)
    normalized = normalize_landmarks(raw)
    return PoseFrame(
        landmarks=normalized,
        visibility=visibility,
        raw_landmarks=raw.copy() if keep_raw else None,
    )


class PoseEstimator:
    """MediaPipe Pose wrapper for live webcam / video frames."""

    def __init__(
        self,
        *,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        smooth_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        if mp is None:
            raise ImportError("mediapipe is required. Install with: pip install mediapipe")

        self._pose = mp.solutions.pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, rgb_frame: np.ndarray, *, keep_raw: bool = False) -> Optional[PoseFrame]:
        """Run pose estimation on one RGB frame; return normalized keypoints or None."""
        if rgb_frame.ndim != 3 or rgb_frame.shape[2] != 3:
            raise ValueError(f"Expected RGB frame (H, W, 3), got shape {rgb_frame.shape}")

        results = self._pose.process(rgb_frame)
        return mediapipe_to_poseframe(results, keep_raw=keep_raw)

    def close(self) -> None:
        self._pose.close()

    def __enter__(self) -> PoseEstimator:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
