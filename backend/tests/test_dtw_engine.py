"""Tests for DTW comparison and alignment gating."""

import json

import numpy as np
import pytest

from app.dtw_engine import (
    DTWEngine,
    compare_to_move,
    flatten_sequence,
    load_reference_library,
)


def _synthetic_move(frames: int = 30, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((frames, 99)).astype(np.float32)


def test_flatten_sequence_shape():
    frames = [_synthetic_move(1)[0].reshape(33, 3) for _ in range(5)]
    flat = flatten_sequence(frames)
    assert flat.shape == (5, 99)


def test_identical_sequences_align(tmp_path):
    seq = _synthetic_move(20)
    np.save(tmp_path / "test_move.npy", seq)
    with (tmp_path / "test_move_meta.json").open("w") as f:
        json.dump({"name": "test_move", "dtw_threshold": 1e6}, f)

    library = load_reference_library(tmp_path)
    live = seq.copy()
    result = compare_to_move(live, library[0])
    assert result.distance == pytest.approx(0.0, abs=1e-6)
    assert result.aligned is True


def test_distant_sequences_do_not_align(tmp_path):
    ref = _synthetic_move(20, seed=1)
    live = _synthetic_move(20, seed=99)
    np.save(tmp_path / "move.npy", ref)
    with (tmp_path / "move_meta.json").open("w") as f:
        json.dump({"dtw_threshold": 1.0}, f)

    library = load_reference_library(tmp_path)
    result = compare_to_move(live, library[0])
    assert result.aligned is False


def test_engine_requires_full_window(tmp_path):
    seq = _synthetic_move(10)
    np.save(tmp_path / "m.npy", seq)
    engine = DTWEngine(tmp_path, window_len=60)
    short = [seq[0].reshape(33, 3)] * 10
    assert engine.compare(short) is None
    assert engine.compare_async(short) is None


def test_engine_picks_closest_move(tmp_path):
    close = _synthetic_move(15, seed=2)
    far = _synthetic_move(15, seed=3)
    np.save(tmp_path / "close.npy", close)
    np.save(tmp_path / "far.npy", far)
    for name in ("close", "far"):
        with (tmp_path / f"{name}_meta.json").open("w") as f:
            json.dump({"dtw_threshold": 1e6}, f)

    engine = DTWEngine(tmp_path, window_len=15)
    window = [close[i].reshape(33, 3) for i in range(15)]
    result = engine.compare(window)
    assert result is not None
    assert result.move_name == "close"

def test_flatten_sequence_empty_raises():
    with pytest.raises(ValueError):
        flatten_sequence([])


def test_dtw_different_length_sequences(tmp_path):
    """DTW should handle sequences of different lengths."""
    live = _synthetic_move(60)
    ref  = _synthetic_move(45)
    np.save(tmp_path / "move.npy", ref)
    with (tmp_path / "move_meta.json").open("w") as f:
        json.dump({"dtw_threshold": 1e6}, f)
    library = load_reference_library(tmp_path)
    result = compare_to_move(live, library[0])
    assert result.distance >= 0


def test_engine_empty_library(tmp_path):
    engine = DTWEngine(tmp_path)
    assert not engine.has_reference


def test_engine_async_returns_future(tmp_path):
    seq = _synthetic_move(60)
    np.save(tmp_path / "windmill.npy", seq)
    with (tmp_path / "windmill_meta.json").open("w") as f:
        json.dump({"dtw_threshold": 1e6}, f)

    from app.dtw_engine import DTWEngine, DTWResult
    engine = DTWEngine(tmp_path, window_len=60)
    window = [seq[i].reshape(33, 3) for i in range(60)]
    future = engine.compare_async(window)
    assert future is not None
    result = future.result(timeout=5)
    assert isinstance(result, DTWResult)
    engine.shutdown()

def test_engine_can_practice_a_single_move(tmp_path):
    for name in ("a", "b"):
        np.save(tmp_path / f"{name}.npy", np.zeros((60, 99), dtype=np.float32))
    only = DTWEngine(tmp_path, only_move="b")
    assert [m.name for m in only.library] == ["b"]
    fallback = DTWEngine(tmp_path, only_move="nope")
    assert len(fallback.library) == 2
