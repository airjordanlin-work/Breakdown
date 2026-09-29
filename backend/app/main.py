"""Entry point — webcam/video loop wiring pose, buffer, DTW, scoring, and overlay."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.app import overlay
from backend.app.buffer import PoseBuffer, WINDOW_LEN
from backend.app.dtw_engine import DTWEngine
from backend.app.pose_estimator import PoseEstimator, is_pose_reliable
from backend.app.scorer import Scorer
from backend.app.voice import VoiceCoach

DTW_EVERY_N_FRAMES = 3
KEYFRAME_FLASH_FRAMES = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Breakdance Coach — real-time move feedback")
    parser.add_argument("--source", default="webcam")
    parser.add_argument("--reference-dir", type=Path, 
                    default=Path(__file__).resolve().parent.parent / "reference_moves")
    parser.add_argument("--model-complexity", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--window-name", default="Breakdance Coach")
    parser.add_argument(
        "--zoom", type=float, default=1.0,
        help="Digital zoom factor. <1.0 zooms out (shows more). Try 0.7"
    )
    parser.add_argument(
        "--no-voice", action="store_true",
        help="Disable voice coaching feedback"
    )
    parser.add_argument(
        "--voice-gender", default="female", choices=["female", "male"],
        help="Voice coach gender: female=Bella (default) or male=Michael"
    )
    return parser.parse_args()


def open_capture(source: str) -> cv2.VideoCapture:
    if source == "webcam":
        return cv2.VideoCapture(0)
    if source.isdigit():
        return cv2.VideoCapture(int(source))
    return cv2.VideoCapture(source)


def warn_if_no_references(reference_dir: Path) -> None:
    if not any(reference_dir.glob("*.npy")):
        print(f"WARNING: no reference moves found in {reference_dir}")


def print_session_summary(scorer: Scorer) -> None:
    stats = scorer.stats()
    print("\nSession complete")
    print(f"Final score: {stats.total_points:.0f}")
    print(f"Grade:       {stats.grade}")
    print(f"Perfect: {stats.perfects} / Close: {stats.closes} / Miss: {stats.misses}")


def apply_zoom(frame: np.ndarray, zoom: float) -> np.ndarray:
    """Digital zoom. <1.0 zooms out by adding black borders. >1.0 zooms in by cropping."""
    if zoom == 1.0:
        return frame
    h, w = frame.shape[:2]
    if zoom < 1.0:
        new_h = int(h / zoom)
        new_w = int(w / zoom)
        canvas = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        y_off = (new_h - h) // 2
        x_off = (new_w - w) // 2
        canvas[y_off:y_off + h, x_off:x_off + w] = frame
        return cv2.resize(canvas, (w, h))
    else:
        cy, cx = h // 2, w // 2
        half_h = max(1, min(int((h / zoom) / 2), cy))
        half_w = max(1, min(int((w / zoom) / 2), cx))
        cropped = frame[cy - half_h:cy + half_h, cx - half_w:cx + half_w]
        return cv2.resize(cropped, (w, h))


def run_loop(args: argparse.Namespace) -> None:
    reference_dir = Path(args.reference_dir)
    warn_if_no_references(reference_dir)

    print("Loading pose model...")
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)

    cap = open_capture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {args.source}")

    estimator = PoseEstimator(model_complexity=args.model_complexity)
    estimator.process(dummy)
    print("Ready — press Q to quit | + zoom out | - zoom in | V toggle voice gender")

    engine = DTWEngine(reference_dir)
    print(f"Reference library loaded: {len(engine.library)} move(s)")

    scorer = Scorer(reference_dir)
    buf    = PoseBuffer()
    voice  = VoiceCoach(gender=args.voice_gender) if not args.no_voice else None
    if voice:
        voice.speak("Voice coach ready")

    current_gender    = args.voice_gender
    future            = None
    flash_counter     = 0
    score_result      = None
    dtw_result        = None
    frame_idx         = 0
    pose_frame        = None
    was_aligned       = False
    buf_was_ready     = False
    no_body_voiced_at = -999

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = apply_zoom(frame, args.zoom)
            frame = cv2.flip(frame, 1)

            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose = estimator.process(rgb, keep_raw=True)

            body_visible = pose is not None and is_pose_reliable(pose)

            if body_visible:
                pose_frame = pose
                buf.add(pose)
            else:
                if voice and frame_idx - no_body_voiced_at > 150:
                    voice.feedback_no_body()
                    no_body_voiced_at = frame_idx

            if buf.is_ready() and not buf_was_ready:
                buf_was_ready = True
                if voice:
                    voice.feedback_buffer_ready()

            if frame_idx % 30 == 0:
                print(f"[frame {frame_idx}] buffer: {len(buf)}/60 | has_ref: {engine.has_reference} | future: {future is not None} | zoom: {args.zoom:.1f}")

            if buf.is_ready() and future is None and frame_idx % DTW_EVERY_N_FRAMES == 0:
                window = list(buf.get_sequence())
                future = engine.compare_async(window)
                print(f"DTW fired on frame {frame_idx}")

            if future is not None and future.done():
                try:
                    dtw_result = future.result()
                except Exception as e:
                    print(f"DTW error: {e}")
                    dtw_result = None
                future = None

                if dtw_result is not None:
                    print(f"DTW distance: {dtw_result.distance:.2f} | threshold: {dtw_result.threshold:.2f} | aligned: {dtw_result.aligned}")

                    if dtw_result.aligned and not was_aligned:
                        if voice:
                            voice.feedback_aligned()
                    was_aligned = dtw_result.aligned

                    if dtw_result.aligned and pose_frame is not None:
                        new_score = scorer.score(dtw_result, pose_frame)
                        print(f"Score result: {new_score}")
                        if new_score is not None:
                            score_result  = new_score
                            flash_counter = KEYFRAME_FLASH_FRAMES
                            if voice:
                                voice.feedback_from_score(new_score)
                        else:
                            print(f"No score — {scorer.explain_no_score(dtw_result)}")

            fill_ratio = len(buf) / WINDOW_LEN
            move_name  = dtw_result.move_name if dtw_result else engine.move_name

            frame, flash_counter = overlay.render(
                frame,
                pose_frame=pose_frame,
                dtw_result=dtw_result,
                score_result=score_result,
                stats=scorer.stats(),
                fill_ratio=fill_ratio,
                flash_counter=flash_counter,
                move_name=move_name,
                buf_len=len(buf),
            )

            cv2.imshow(args.window_name, frame)
            frame_idx += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("=") or key == ord("+"):
                args.zoom = max(0.3, args.zoom - 0.1)
                print(f"Zoom: {args.zoom:.1f}")
            elif key == ord("-"):
                args.zoom = min(2.0, args.zoom + 0.1)
                print(f"Zoom: {args.zoom:.1f}")
            elif key == ord("v"):
                if voice:
                    current_gender = "male" if current_gender == "female" else "female"
                    voice.set_gender(current_gender)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        estimator.close()
        engine.shutdown()
        print_session_summary(scorer)


def main() -> None:
    args = parse_args()
    run_loop(args)


if __name__ == "__main__":
    main()