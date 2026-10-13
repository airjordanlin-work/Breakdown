"""Benchmark the Breakdown pipeline and print numbers you can defend.

Run on the machine you actually demo on, with nothing else heavy open, then
copy the printed summary lines. Every stage also writes a CSV of raw samples to
backend/recordings/ so the numbers can be re-derived.

    python scripts/benchmark_pipeline.py dtw
    python scripts/benchmark_pipeline.py pose --video clip.mp4
    python scripts/benchmark_pipeline.py pose --video clip.mp4 --with-dtw
    python scripts/benchmark_pipeline.py coach --n 5
    python scripts/benchmark_pipeline.py all --video clip.mp4

What each stage measures (and what it does not):

  dtw    Time for one full-library fastdtw comparison of a 60-frame window.
         The "self-match" rate is a plumbing sanity check on noisy slices of
         the references. It is NOT dancer-scoring accuracy.
  pose   MediaPipe + normalization per frame on a real video of a person,
         optionally with DTW running in the background the way the live loop
         does it (every 3rd frame). Excludes browser capture, JPEG decode and
         the WebSocket hop. For true end-to-end latency, use the round-trip
         readout in the frontend debug mode.
  coach  Claude Haiku latency, token use and cost per cue across fixed
         scenarios, plus an automatic grounding check. The check is a rough
         heuristic. Fill in the manual_label column in the CSV by hand and
         report that rate, not the automatic one.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

REFERENCE_DIR = _ROOT / "reference_moves"
OUT_DIR = _ROOT / "recordings"
FRAME_BUDGET_MS = 1000.0 / 30.0  # 30 fps

# Claude Haiku 4.5 list prices in USD per million tokens. Check the Anthropic
# pricing page before quoting a cost figure, these may have changed.
PRICE_IN_PER_MTOK = 1.0
PRICE_OUT_PER_MTOK = 5.0

_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------------- helpers

def pct(values: list[float], p: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def summarize(label: str, values_ms: list[float]) -> str:
    line = (
        f"{label}: n={len(values_ms)}  mean={np.mean(values_ms):.1f} ms  "
        f"p50={pct(values_ms, 50):.1f} ms  p95={pct(values_ms, 95):.1f} ms  "
        f"max={max(values_ms):.1f} ms"
    )
    print(line)
    return line


def write_csv(stage: str, header: list[str], rows: list[list]) -> Path:
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / f"bench_{stage}_{_STAMP}.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"  raw samples -> {path}")
    return path


# -------------------------------------------------------------------- dtw

def bench_dtw(args: argparse.Namespace) -> None:
    from app.dtw_engine import DEFAULT_WINDOW_LEN, compare_to_library, load_reference_library

    library = load_reference_library(REFERENCE_DIR)
    usable = [m for m in library if m.num_frames >= DEFAULT_WINDOW_LEN]
    if not usable:
        sys.exit(f"No reference moves with >= {DEFAULT_WINDOW_LEN} frames in {REFERENCE_DIR}")

    rng = np.random.default_rng(args.seed)
    print(f"\n[dtw] library={len(library)} moves, window={DEFAULT_WINDOW_LEN} frames, runs={args.n}")

    for _ in range(3):  # warm-up so import and cache costs are not counted
        compare_to_library(usable[0].sequence[:DEFAULT_WINDOW_LEN], library)

    times_ms: list[float] = []
    rows: list[list] = []
    self_hits = 0
    for i in range(args.n):
        move = usable[int(rng.integers(len(usable)))]
        start = int(rng.integers(0, move.num_frames - DEFAULT_WINDOW_LEN + 1))
        window = move.sequence[start:start + DEFAULT_WINDOW_LEN]
        window = window + rng.normal(0, args.noise, window.shape).astype(np.float32)

        t0 = time.perf_counter()
        result = compare_to_library(window, library)
        dt = (time.perf_counter() - t0) * 1000.0

        hit = int(result is not None and result.move_name == move.name)
        self_hits += hit
        times_ms.append(dt)
        rows.append([i, move.name, result.move_name if result else "", f"{dt:.3f}", hit])

    summarize("full-library DTW compare", times_ms)
    print(f"self-match sanity (noisy slice of a reference finds itself): "
          f"{self_hits}/{args.n} = {100.0 * self_hits / args.n:.1f}%  (plumbing check, not accuracy)")
    write_csv("dtw", ["run", "true_move", "best_move", "ms", "self_match"], rows)


# ------------------------------------------------------------------- pose

def bench_pose(args: argparse.Namespace) -> None:
    import cv2

    from app.buffer import PoseBuffer
    from app.dtw_engine import DTWEngine
    from app.pose_estimator import PoseEstimator

    if not args.video:
        sys.exit("pose stage needs --video <path to a clip of a person moving>")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"Could not open video: {args.video}")
    frames = []
    while len(frames) < args.max_frames:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))  # decode/convert outside the timed loop
    cap.release()
    if len(frames) < 30:
        sys.exit(f"Need at least 30 frames, got {len(frames)}")

    h, w = frames[0].shape[:2]
    mode = "with background DTW" if args.with_dtw else "pose only"
    print(f"\n[pose] {len(frames)} frames at {w}x{h}, model_complexity={args.model_complexity}, {mode}")

    estimator = PoseEstimator(model_complexity=args.model_complexity)
    buffer = PoseBuffer()
    engine = DTWEngine(REFERENCE_DIR) if args.with_dtw else None
    pending = None
    dtw_ms: list[float] = []

    def _track(start: float):
        def _done(_fut):
            dtw_ms.append((time.perf_counter() - start) * 1000.0)
        return _done

    for rgb in frames[:5]:  # warm-up
        estimator.process(rgb)

    frame_ms: list[float] = []
    detected = 0
    rows: list[list] = []
    wall0 = time.perf_counter()
    for idx, rgb in enumerate(frames):
        t0 = time.perf_counter()
        pose = estimator.process(rgb)
        if pose is not None:
            detected += 1
            buffer.add(pose)
        if engine and buffer.is_ready() and idx % 3 == 0 and (pending is None or pending.done()):
            pending = engine.compare_async(list(buffer.get_sequence()))
            if pending is not None:
                pending.add_done_callback(_track(time.perf_counter()))
        dt = (time.perf_counter() - t0) * 1000.0
        frame_ms.append(dt)
        rows.append([idx, f"{dt:.3f}", int(pose is not None)])
    wall = time.perf_counter() - wall0

    if engine:
        engine.shutdown()
    estimator.close()

    summarize("per-frame processing", frame_ms)
    over = sum(1 for v in frame_ms if v > FRAME_BUDGET_MS)
    print(f"throughput: {len(frames) / wall:.1f} fps processing-only  "
          f"(30 fps needs <= {FRAME_BUDGET_MS:.1f} ms/frame; {over} of {len(frames)} frames over budget)")
    print(f"pose detected in {detected}/{len(frames)} frames = {100.0 * detected / len(frames):.1f}%")
    if detected < 0.5 * len(frames):
        print("WARNING: few frames had a detectable person. Use a clip of a person fully in frame, "
              "or these timings will be too optimistic.")
    if dtw_ms:
        summarize("background DTW (submit -> done)", dtw_ms)
    write_csv("pose", ["frame", "ms", "detected"], rows)


# ------------------------------------------------------------------ coach

_BODY_WORDS = ["elbow", "knee", "wrist", "shoulder", "hip", "ankle", "arm", "leg",
               "hand", "foot", "feet", "head", "chest", "back", "core"]


def _grounding_flags(prompt: str, cue: str) -> list[str]:
    """Heuristic: does the cue mention body parts or numbers the prompt never gave it?"""
    p = prompt.lower().replace("_", " ")
    c = cue.lower()
    flags = []
    for word in _BODY_WORDS:
        if re.search(rf"\b{word}s?\b", c) and not re.search(rf"\b{word}s?\b", p):
            flags.append(f"mentions '{word}' not in data")
    prompt_nums = set(re.findall(r"\d+(?:\.\d+)?", p))
    for num in re.findall(r"\d+(?:\.\d+)?", c):
        if num not in prompt_nums:
            flags.append(f"number '{num}' not in data")
    if len(cue.split()) > 10:
        flags.append("over 10 words")
    return flags


def _scenarios():
    from app.coach import _build_freeze_prompt, _build_prompt

    miss = lambda j, fix: {"results": [{"joint": j, "tier": "Miss", "diff": 40.0, "fix": fix}]}
    return [
        ("moving_no_data", _build_prompt("BR 01", None)),
        ("miss_left_elbow", _build_prompt("BR 01", miss("left elbow", "straighten"))),
        ("miss_right_knee", _build_prompt("BR 01", miss("right knee", "bend"))),
        ("close_left_shoulder", _build_prompt("BR 01",
            {"results": [{"joint": "left shoulder", "tier": "Close", "diff": 22.0, "fix": "bend"}]})),
        ("all_perfect", _build_prompt("BR 01",
            {"results": [{"joint": "left elbow", "tier": "Perfect", "diff": 4.0, "fix": "bend"}]})),
        ("freeze_steady", _build_freeze_prompt(
            {"tier": "steady", "duration_s": 2.4, "stability": 84, "shakiest": "wrist"}, None)),
        ("freeze_wobbly", _build_freeze_prompt(
            {"tier": "wobbly", "duration_s": 0.9, "stability": 38, "shakiest": "leg"}, None)),
    ]


def bench_coach(args: argparse.Namespace) -> None:
    import httpx

    from app import coach

    if not coach.ANTHROPIC_API_KEY:
        sys.exit("ANTHROPIC_API_KEY not set (add it to .env). Coach stage makes real API calls.")

    scenarios = _scenarios()
    total = len(scenarios) * args.n
    print(f"\n[coach] model={coach.MODEL}, {len(scenarios)} scenarios x {args.n} = {total} real API calls")

    headers = {
        "x-api-key": coach.ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    lat_ms: list[float] = []
    in_tok = out_tok = failures = flagged = 0
    rows: list[list] = []

    with httpx.Client(timeout=coach.TIMEOUT_SECONDS * 2) as client:
        for name, prompt in scenarios:
            for k in range(args.n):
                body = {
                    "model": coach.MODEL,
                    "max_tokens": coach.MAX_TOKENS,
                    "system": coach.SYSTEM_PROMPT,
                    "messages": [{"role": "user", "content": prompt}],
                }
                t0 = time.perf_counter()
                try:
                    resp = client.post(coach.ANTHROPIC_URL, headers=headers, json=body)
                    resp.raise_for_status()
                    data = resp.json()
                except Exception as exc:  # network error, timeout, 4xx/5xx
                    failures += 1
                    rows.append([name, k, "", "", "", "", f"ERROR {exc}", "", ""])
                    continue
                dt = (time.perf_counter() - t0) * 1000.0
                cue = data["content"][0]["text"].strip().strip('"')
                usage = data.get("usage", {})
                i_t, o_t = int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))
                flags = _grounding_flags(prompt, cue)
                lat_ms.append(dt)
                in_tok += i_t
                out_tok += o_t
                flagged += int(bool(flags))
                rows.append([name, k, f"{dt:.1f}", i_t, o_t, cue, "; ".join(flags), int(bool(flags)), ""])

    ok = len(lat_ms)
    if not ok:
        sys.exit("All coach calls failed, check your key and network.")
    summarize("Haiku cue latency (API round trip)", lat_ms)
    cost = (in_tok * PRICE_IN_PER_MTOK + out_tok * PRICE_OUT_PER_MTOK) / 1_000_000
    per_cue = cost / ok
    print(f"tokens per cue: {in_tok / ok:.0f} in / {out_tok / ok:.1f} out  ->  ${per_cue:.5f} per cue "
          f"(assumes ${PRICE_IN_PER_MTOK}/${PRICE_OUT_PER_MTOK} per M tokens, verify pricing)")
    print(f"a 10-minute session at the {coach.COOLDOWN_SECONDS:.0f}s cooldown is ~{600 / coach.COOLDOWN_SECONDS:.0f} "
          f"cues = ~${per_cue * 600 / coach.COOLDOWN_SECONDS:.3f}")
    print(f"failures: {failures}/{total}")
    print(f"auto grounding check flagged {flagged}/{ok} cues = {100.0 * flagged / ok:.1f}%  (rough heuristic)")
    print("Next: open the CSV, read every cue, and fill manual_label with grounded / ungrounded. "
          "Report that rate.")
    write_csv("coach", ["scenario", "run", "latency_ms", "in_tokens", "out_tokens", "cue",
                        "auto_flags", "auto_flagged", "manual_label"], rows)


# ------------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)

    p_dtw = sub.add_parser("dtw", help="time full-library DTW comparisons")
    p_pose = sub.add_parser("pose", help="time MediaPipe (and optional background DTW) on a video")
    p_coach = sub.add_parser("coach", help="time and check Claude Haiku cues (real API calls)")
    p_all = sub.add_parser("all", help="run every stage that has what it needs")

    for p in (p_dtw, p_all):
        p.add_argument("--n", type=int, default=200, help="DTW comparisons to time")
        p.add_argument("--noise", type=float, default=0.02, help="noise added to reference slices")
        p.add_argument("--seed", type=int, default=0)
    for p in (p_pose, p_all):
        p.add_argument("--video", help="clip of a person moving, fully in frame")
        p.add_argument("--max-frames", type=int, default=600)
        p.add_argument("--model-complexity", type=int, default=1, choices=[0, 1, 2])
        p.add_argument("--with-dtw", action="store_true", help="run DTW in the background every 3rd frame")
    p_coach.add_argument("--n", type=int, default=5, help="API calls per scenario")

    args = parser.parse_args()
    if args.stage == "all":
        bench_dtw(args)
        if args.video:
            bench_pose(args)
        else:
            print("\n[pose] skipped, pass --video to include it")
        if os.getenv("ANTHROPIC_API_KEY") or (_ROOT.parent / ".env").exists() or (_ROOT / ".env").exists():
            args.n = 5
            bench_coach(args)
        else:
            print("\n[coach] skipped, no ANTHROPIC_API_KEY found")
        return
    {"dtw": bench_dtw, "pose": bench_pose, "coach": bench_coach}[args.stage](args)


if __name__ == "__main__":
    main()