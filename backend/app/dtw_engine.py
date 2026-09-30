"""Comparison engine — fastdtw sequence alignment against the reference move library.

Architecture role (flowchart):
  reference_moves/*.npy + *_meta.json  →  DTWEngine
  60-frame normalized window           →  fastdtw (O(n) approximate DTW)
  distance vs per-move threshold       →  DTWResult.aligned  →  scoring / feedback

Embedding lookup (FAISS) is a separate path in the roadmap; this module owns
the DTW branch only.
"""

from __future__ import annotations

import json
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    from fastdtw import fastdtw
except ImportError:  # pragma: no cover
    fastdtw = None  # type: ignore

from scipy.spatial.distance import euclidean

# 33 MediaPipe landmarks × (x, y, z)
FEATURES_PER_FRAME = 33 * 3
DEFAULT_WINDOW_LEN = 60
DEFAULT_DTW_THRESHOLD = 50.0

# Joints compared by DTW: the 17 that AIST++ references actually contain
# (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles). The
# other 16 MediaPipe landmarks are empty in those references, so including
# them would compare the dancer's hands and feet against zeros.
COMPARE_JOINTS = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


def comparable(sequence: np.ndarray) -> np.ndarray:
    """(T, 99) -> (T, 34): x and y of COMPARE_JOINTS only.

    Depth is left out on purpose: MediaPipe's z from a single webcam is a
    rough estimate, while AIST++ depth is measured, so comparing them adds
    noise rather than signal.
    """
    seq = _ensure_2d(sequence).reshape(-1, 33, 3)
    return seq[:, COMPARE_JOINTS, :2].reshape(seq.shape[0], -1)


@dataclass(frozen=True)
class ReferenceMove:
    """One move from the reference library."""

    name: str
    sequence: np.ndarray  # (T, 99) flattened normalized keypoints
    meta: dict[str, Any] = field(default_factory=dict)
    dtw_threshold: float = DEFAULT_DTW_THRESHOLD
    keyframes: list[dict[str, Any]] = field(default_factory=list)

    @property
    def num_frames(self) -> int:
        return int(self.sequence.shape[0])


@dataclass(frozen=True)
class DTWResult:
    """Outcome of comparing a live window to the best-matching reference move."""

    aligned: bool
    distance: float
    normalized_distance: float
    move_name: str
    threshold: float
    live_frames: int
    reference_frames: int
    # Indices into live / reference sequences along the warping path (for keyframe sync)
    live_path: tuple[int, ...] = ()
    reference_path: tuple[int, ...] = ()

    @property
    def match_label(self) -> str:
        return "ALIGNED" if self.aligned else "NO MATCH"


def flatten_sequence(frames: list[np.ndarray]) -> np.ndarray:
    """Stack pose frames into (T, 99) — 33 landmarks × 3 coords."""
    if not frames:
        raise ValueError("Cannot flatten an empty frame list")
    return np.stack([np.asarray(f, dtype=np.float32).reshape(-1) for f in frames], axis=0)


def _ensure_2d(sequence: np.ndarray) -> np.ndarray:
    """Accept (T, 33, 3) or (T, 99) on disk; always return (T, 99)."""
    seq = np.asarray(sequence, dtype=np.float32)
    if seq.ndim == 3:
        return seq.reshape(seq.shape[0], -1)
    if seq.ndim == 2 and seq.shape[1] == FEATURES_PER_FRAME:
        return seq
    raise ValueError(f"Expected shape (T, 33, 3) or (T, 99), got {seq.shape}")


def _parse_meta(meta_path: Path, move_name: str) -> tuple[dict[str, Any], float, list[dict[str, Any]]]:
    if not meta_path.exists():
        return {}, DEFAULT_DTW_THRESHOLD, []

    with meta_path.open(encoding="utf-8") as f:
        meta = json.load(f)

    threshold = float(meta.get("dtw_threshold", DEFAULT_DTW_THRESHOLD))
    keyframes = list(meta.get("keyframes", []))
    return meta, threshold, keyframes


def load_reference_library(reference_dir: Path) -> list[ReferenceMove]:
    """Load all ``*.npy`` moves and companion ``{name}_meta.json`` files."""
    reference_dir = Path(reference_dir)
    moves: list[ReferenceMove] = []

    for npy_path in sorted(reference_dir.glob("*.npy")):
        name = npy_path.stem
        sequence = _ensure_2d(np.load(npy_path))
        meta_path = reference_dir / f"{name}_meta.json"
        meta, threshold, keyframes = _parse_meta(meta_path, name)
        moves.append(
            ReferenceMove(
                name=meta.get("name", name),
                sequence=sequence,
                meta=meta,
                dtw_threshold=threshold,
                keyframes=keyframes,
            )
        )

    return moves


def dtw_distance(
    live: np.ndarray,
    reference: np.ndarray,
) -> tuple[float, list[tuple[int, int]]]:
    """Run fastdtw between two (T, 99) sequences; return distance and warping path."""
    if fastdtw is None:
        raise ImportError("fastdtw is required. Install with: pip install fastdtw")

    live = np.asarray(live, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    distance, path = fastdtw(live, reference, dist=euclidean)
    return float(distance), path


def compare_to_move(live: np.ndarray, move: ReferenceMove) -> DTWResult:
    """Compare a live window to one reference move and test alignment."""
    live = _ensure_2d(live)
    distance, path = dtw_distance(comparable(live), comparable(move.sequence))
    n_live, n_ref = live.shape[0], move.num_frames
    normalized = distance / max(n_live, n_ref, 1)

    live_path = tuple(i for i, _j in path)
    ref_path = tuple(j for _i, j in path)

    aligned = distance <= move.dtw_threshold
    return DTWResult(
        aligned=aligned,
        distance=distance,
        normalized_distance=normalized,
        move_name=move.name,
        threshold=move.dtw_threshold,
        live_frames=n_live,
        reference_frames=n_ref,
        live_path=live_path,
        reference_path=ref_path,
    )


def compare_to_library(live: np.ndarray, library: list[ReferenceMove]) -> Optional[DTWResult]:
    """Pick the lowest-DTW reference move and apply that move's alignment threshold."""
    if not library:
        return None

    best: Optional[DTWResult] = None
    for move in library:
        result = compare_to_move(live, move)
        if best is None or result.distance < best.distance:
            best = result
    return best


class DTWEngine:
    """Async DTW comparison against the full reference move library.

    Typical usage from the main render loop::

        engine = DTWEngine(reference_dir)
        buffer: deque[np.ndarray]  # maxlen=60, already normalized

        if engine.is_ready(buffer) and frame_idx % 3 == 0:
            pending = engine.compare_async(list(buffer))

        if pending and pending.done():
            result = pending.result()
            if result.aligned:
                ...  # keyframe scoring
            else:
                ...  # mismatch feedback
    """

    def __init__(
        self,
        reference_dir: Path,
        *,
        window_len: int = DEFAULT_WINDOW_LEN,
        executor: Optional[ThreadPoolExecutor] = None,
        only_move: Optional[str] = None,
    ):
        self.reference_dir = Path(reference_dir)
        self.window_len = window_len
        self._executor = executor or ThreadPoolExecutor(max_workers=1, thread_name_prefix="dtw")
        self._library: list[ReferenceMove] = load_reference_library(self.reference_dir)
        # Practicing one move: compare only against it. Unknown ids fall back
        # to the full library instead of leaving nothing to compare against.
        if only_move:
            chosen = [m for m in self._library if m.name == only_move]
            if chosen:
                self._library = chosen
        self._last_result: Optional[DTWResult] = None

    @property
    def library(self) -> list[ReferenceMove]:
        return list(self._library)

    @property
    def move_name(self) -> Optional[str]:
        """Best-matching move from the last completed comparison."""
        if self._last_result is None:
            return self._library[0].name if self._library else None
        return self._last_result.move_name

    @property
    def last_result(self) -> Optional[DTWResult]:
        return self._last_result

    @property
    def has_reference(self) -> bool:
        return len(self._library) > 0

    @property
    def aligned(self) -> bool:
        return bool(self._last_result and self._last_result.aligned)

    def get_move(self, name: str) -> Optional[ReferenceMove]:
        for move in self._library:
            if move.name == name:
                return move
        return None

    def is_ready(self, window: list[np.ndarray] | deque[np.ndarray]) -> bool:
        """True when the sliding window is full enough for a stable DTW read."""
        return self.has_reference and len(window) >= self.window_len

    def compare(self, window: list[np.ndarray]) -> Optional[DTWResult]:
        """Synchronous comparison — use from tests or when blocking is acceptable."""
        if not self.is_ready(window):
            return None

        live = flatten_sequence(window[-self.window_len :])
        result = compare_to_library(live, self._library)
        self._last_result = result
        return result

    def compare_async(self, window: list[np.ndarray]) -> Optional[Future[DTWResult]]:
        """Submit DTW to the worker pool; returns None if the window is not ready."""
        if not self.is_ready(window):
            return None

        live = flatten_sequence(window[-self.window_len :])

        def _run() -> DTWResult:
            result = compare_to_library(live, self._library)
            if result is None:
                raise RuntimeError("Reference library is empty")
            self._last_result = result
            return result

        return self._executor.submit(_run)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)
