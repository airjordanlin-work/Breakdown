"""Benchmark the Breakdown pipeline and print numbers you can defend.

Run on the machine you actually demo on, with nothing else heavy open, then
copy the printed summary lines. Every stage also writes a CSV of raw samples to
backend/recordings/ so the numbers can be re-derived.

    python scripts/benchmark_pipeline.py dtw
    python scripts/benchmark_pipeline.py pose --video clip.mp4
    python scripts/benchmark_pipeline.py pose --video clip.mp4 --with-dtw
    python scripts/benchmark_pipeline.py coach --n 5
    python scripts/benchmark_pipeline.py stt --synth
    python scripts/benchmark_pipeline.py stt --clips my_clips.csv --models tiny.en base.en
    python scripts/benchmark_pipeline.py chat --n 3
    python scripts/benchmark_pipeline.py threshold --out-dir ../docs/benchmarks
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
  threshold
         DTW distance of each reference move against: its own clip played
         faster and slower with noise added (a stand-in for doing the move),
         the other moves, and a single frame held still. Shows whether the
         per-move dtw_threshold separates them. It uses reference data only,
         so it cannot say how far a real dancer on a phone camera lands.
  coach  Claude Haiku latency, token use and cost per cue across fixed
         scenarios, plus an automatic grounding check. The check is a rough
         heuristic. Fill in the manual_label column in the CSV by hand and
         report that rate, not the automatic one.
  stt    Local speech to text (faster-whisper) time per clip, and word error
         rate against the expected text. --synth speaks fixed coaching
         questions with Kokoro, which is clean synthetic speech: its error
         rate says nothing about a real voice over music. For that, record
         your own clips (16-bit WAV, any rate) and pass --clips with a CSV of
         path,text. Model load time is reported separately.
  chat   Answers to the dancer's questions (app/chat.py) with real Haiku
         calls: latency, tokens and cost per model turn, and how often the
         grounding guard replaced the answer. Also reports how many of the
         fixed sample questions were answered free (no model call); that
         share depends on the sample list, not on real use.
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


def write_csv(stage: str, header: list[str], rows: list[list], out_dir: Path = OUT_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"bench_{stage}_{_STAMP}.csv"
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


# -------------------------------------------------------------- threshold

def _time_warp(seq: np.ndarray, speed: float, length: int) -> np.ndarray:
    """Replay a (T, 99) clip at ``speed`` and return ``length`` frames (looping)."""
    t = (np.arange(length) * speed) % (seq.shape[0] - 1)
    lo = np.floor(t).astype(int)
    frac = (t - lo)[:, None].astype(np.float32)
    return seq[lo] * (1 - frac) + seq[lo + 1] * frac


def bench_threshold(args: argparse.Namespace) -> None:
    from app.dtw_engine import DEFAULT_WINDOW_LEN, compare_to_move, load_reference_library

    library = [m for m in load_reference_library(REFERENCE_DIR) if m.num_frames >= DEFAULT_WINDOW_LEN]
    rng = np.random.default_rng(args.seed)
    n = DEFAULT_WINDOW_LEN
    print(f"\n[threshold] {len(library)} moves, window={n}, noise={args.noise}, speeds={args.speeds}")

    rows: list[list] = []
    by_kind: dict[str, list[float]] = {}
    for move in library:
        cases = []
        for speed in args.speeds:
            live = _time_warp(move.sequence, speed, n)
            live = live + rng.normal(0, args.noise, live.shape).astype(np.float32)
            cases.append(("same_move_warped", f"{move.name}@{speed}x", live))
        for other in library:
            if other.name != move.name:
                cases.append(("other_move", other.name, other.sequence[:n]))
        for frame in (0, move.num_frames // 2):
            cases.append(("held_still", f"{move.name}[{frame}]", np.repeat(move.sequence[frame:frame + 1], n, axis=0)))
        for kind, source, live in cases:
            d = compare_to_move(live, move).distance
            by_kind.setdefault(kind, []).append(d)
            rows.append([move.name, kind, source, f"{d:.1f}", f"{move.dtw_threshold:.1f}", int(d <= move.dtw_threshold)])

    thresholds = sorted({m.dtw_threshold for m in library})
    print(f"current dtw_threshold values: {thresholds}")
    for kind, vals in by_kind.items():
        print(f"{kind:>17}: n={len(vals)}  min={min(vals):.0f}  p50={pct(vals, 50):.0f}  max={max(vals):.0f}")
    write_csv("threshold", ["reference", "kind", "source", "distance", "threshold", "aligned"], rows,
              Path(args.out_dir) if args.out_dir else OUT_DIR)


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


# ------------------------------------------------------------------- stt

def _words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]+", " ", text.lower()).split()


def word_error_rate(reference: str, heard: str) -> float:
    """Word-level edit distance divided by the reference length."""
    ref, hyp = _words(reference), _words(heard)
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1] / len(ref)


def wav_to_pcm16k(data: bytes) -> bytes:
    """16-bit WAV bytes (any rate, mono or stereo) to 16 kHz mono PCM16, what the phone sends."""
    import io
    import wave

    with wave.open(io.BytesIO(data)) as w:
        if w.getsampwidth() != 2:
            raise ValueError("only 16-bit WAV is supported")
        rate, channels = w.getframerate(), w.getnchannels()
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if rate != 16000:
        n_out = int(len(audio) * 16000 / rate)
        audio = np.interp(np.linspace(0, len(audio) - 1, n_out), np.arange(len(audio)), audio)
    return np.clip(audio, -32768, 32767).astype("<i2").tobytes()


STT_PHRASES = [
    "how did I do",
    "what should I fix",
    "what went well",
    "why does my left elbow matter",
    "show me the cross step again",
    "how many clean reps was that",
    "is my right knee bent enough",
    "what's my streak",
]


def _synth_clips() -> list[tuple[str, bytes]]:
    from app.voice import VoiceCoach

    voice = VoiceCoach()
    if not voice.available:
        sys.exit(f"Kokoro is not available ({getattr(voice, 'unavailable_reason', '')}); use --clips instead.")
    clips = []
    for text in STT_PHRASES:
        wav = voice.synthesize_wav(text)
        if wav:
            clips.append((text, wav_to_pcm16k(wav)))
    return clips


def _file_clips(path: str) -> list[tuple[str, bytes]]:
    base = Path(path).resolve().parent
    clips = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            wav = (base / row["path"]).read_bytes()
            clips.append((row["text"], wav_to_pcm16k(wav)))
    return clips


def bench_stt(args: argparse.Namespace) -> None:
    from app.stt import BYTES_PER_S, WhisperSTT

    if args.clips:
        clips = _file_clips(args.clips)
        kind = f"your clips ({args.clips})"
    elif args.synth:
        clips = _synth_clips()
        kind = "Kokoro synthetic speech (clean, not a real voice)"
    else:
        sys.exit("Pass --synth or --clips path.csv (columns: path,text).")
    print(f"\n[stt] {len(clips)} clips, {kind}")

    rows: list[list] = []
    for model in args.models:
        engine = WhisperSTT(model=model)
        if not engine.available:
            sys.exit(engine.unavailable_reason)
        t0 = time.perf_counter()
        engine.warm_up()
        if not engine.available:
            sys.exit(engine.unavailable_reason)
        load_s = time.perf_counter() - t0
        print(f"{model}: model load {load_s:.1f} s (first run includes the download)")
        times, rtf, errors = [], [], []
        for rep in range(args.repeat):
            for text, pcm in clips:
                out = engine.transcribe(pcm)
                secs = len(pcm) / BYTES_PER_S
                wer = word_error_rate(text, out.text)
                times.append(out.ms)
                rtf.append(out.ms / 1000.0 / secs)
                errors.append(wer)
                rows.append([model, rep, f"{secs:.2f}", f"{out.ms:.1f}", f"{wer:.3f}",
                             out.confidence if out.confidence is not None else "", text, out.text])
        summarize(f"{model} time per clip", times)
        print(f"{model}: time / clip length p50={pct(rtf, 50):.2f}  word error rate mean={np.mean(errors):.3f}")
    write_csv("stt", ["model", "repeat", "clip_s", "stt_ms", "wer", "confidence", "expected", "heard"], rows,
              Path(args.out_dir) if args.out_dir else OUT_DIR)


# -------------------------------------------------------------------- chat

CHAT_QUESTIONS = [
    "hello",
    "how did I do",
    "what should I fix",
    "what went well",
    "why does a straight elbow matter for this move",
    "can you explain what the cross step is supposed to look like",
    "was my right shoulder okay",
    "am I on beat",
    "give me one thing to think about next round",
    "what's the capital of France",
]


def _chat_facts() -> dict:
    from app.chat import chat_facts

    stats = {"reps": 6, "clean_reps": 4, "clean_streak": 2, "best_streak": 3,
             "focus_joint": "left elbow", "focus_fix": "straighten"}
    summary = {"reps": 3, "clean_reps": 2, "best_streak": 2, "went_well": ["right knee"],
               "fix": {"joint": "left elbow", "fix": "straighten", "times_off": 1, "checkpoint": 2, "of": 4}}
    return chat_facts("Cross step", stats, summary)


def bench_chat(args: argparse.Namespace) -> None:
    import asyncio

    from app import chat as chat_mod
    from app.turn_gate import TokenBudget, TurnGate

    if not chat_mod.ANTHROPIC_API_KEY:
        sys.exit("ANTHROPIC_API_KEY not set (add it to .env). Chat stage makes real API calls.")
    facts = _chat_facts()
    rows: list[list] = []
    lat, t_in, t_out = [], [], []
    sources: dict[str, int] = {}

    async def run_all():
        for rep in range(args.n):
            # Fresh conversation per round so the cache and history do not hide calls.
            convo = chat_mod.CoachChat(gate=TurnGate(per_minute=10_000, per_session=10_000),
                                       budget=TokenBudget(daily=10_000_000))
            for q in CHAT_QUESTIONS:
                r = await convo.answer(q, facts)
                sources[r.source] = sources.get(r.source, 0) + 1
                if r.llm_ms is not None:
                    lat.append(r.llm_ms)
                if r.tokens_in or r.tokens_out:
                    t_in.append(r.tokens_in)
                    t_out.append(r.tokens_out)
                rows.append([rep, q, r.source, r.reason or "", f"{r.llm_ms:.1f}" if r.llm_ms else "",
                             r.tokens_in, r.tokens_out, r.text, ""])

    print(f"\n[chat] model={chat_mod.MODEL}, {len(CHAT_QUESTIONS)} questions x {args.n} rounds")
    asyncio.run(run_all())
    total = len(CHAT_QUESTIONS) * args.n
    free = sources.get("quick", 0)
    print(f"answer sources: {sources}")
    print(f"answered free (no model call): {free}/{total} of this fixed sample list")
    if lat:
        summarize("Haiku chat latency (API round trip)", lat)
    if t_in:
        per_turn = (np.mean(t_in) * PRICE_IN_PER_MTOK + np.mean(t_out) * PRICE_OUT_PER_MTOK) / 1_000_000
        print(f"tokens per model turn: {np.mean(t_in):.0f} in / {np.mean(t_out):.1f} out  ->  ${per_turn:.5f} "
              f"(assumes ${PRICE_IN_PER_MTOK}/${PRICE_OUT_PER_MTOK} per M tokens, verify pricing)")
        print(f"if a 10-minute session has 10 model turns: ~${per_turn * 10:.4f}")
    print(f"answers replaced by the guard or a length check: {sources.get('fallback', 0)}")
    print("Next: read every model answer in the CSV and fill manual_label with grounded / ungrounded.")
    write_csv("chat", ["round", "question", "source", "reason", "llm_ms", "in_tokens", "out_tokens",
                       "answer", "manual_label"], rows, Path(args.out_dir) if args.out_dir else OUT_DIR)


# ------------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)

    p_dtw = sub.add_parser("dtw", help="time full-library DTW comparisons")
    p_pose = sub.add_parser("pose", help="time MediaPipe (and optional background DTW) on a video")
    p_thr = sub.add_parser("threshold", help="DTW distances: same move vs other moves vs standing still")
    p_thr.add_argument("--noise", type=float, default=0.05, help="noise added to the warped clip")
    p_thr.add_argument("--speeds", type=float, nargs="+", default=[0.8, 1.0, 1.25])
    p_thr.add_argument("--seed", type=int, default=0)
    p_thr.add_argument("--out-dir", help="where to write the CSV (default backend/recordings)")
    p_coach = sub.add_parser("coach", help="time and check Claude Haiku cues (real API calls)")
    p_all = sub.add_parser("all", help="run every stage that has what it needs")
    p_stt = sub.add_parser("stt", help="time local speech to text and its word error rate")
    p_stt.add_argument("--synth", action="store_true", help="speak fixed questions with Kokoro")
    p_stt.add_argument("--clips", help="CSV with columns path,text (16-bit WAV files)")
    p_stt.add_argument("--models", nargs="+", default=["base.en"], help="whisper models to compare")
    p_stt.add_argument("--repeat", type=int, default=3, help="passes over the clips")
    p_stt.add_argument("--out-dir", help="where to write the CSV (default backend/recordings)")
    p_chat = sub.add_parser("chat", help="time and check answers to the dancer's questions (real API calls)")
    p_chat.add_argument("--n", type=int, default=3, help="rounds over the sample questions")
    p_chat.add_argument("--out-dir", help="where to write the CSV (default backend/recordings)")

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
    {"dtw": bench_dtw, "pose": bench_pose, "coach": bench_coach,
     "threshold": bench_threshold, "stt": bench_stt, "chat": bench_chat}[args.stage](args)


if __name__ == "__main__":
    main()