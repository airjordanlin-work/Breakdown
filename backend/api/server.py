"""
FastAPI backend — WebSocket streaming with real-time AI coaching.
"""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import json
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import sys
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.pose_estimator import PoseEstimator, PoseFrame, is_pose_reliable
from app.buffer import PoseBuffer, WINDOW_LEN
from app.dtw_engine import DTWEngine
from app.scorer import Scorer
from app.coach import AICoach
from app.demo import demo_message
from app.voice import VoiceCoach
from app.access import configured_token, require_token
from app.viewers import ViewerHub
from app.audio_frames import encode_audio_chunk
from app.imu_protocol import PacketError, unpack_forwarded
from app.imu_store import ImuStore
from app.freeze_monitor import FreezeMonitor
from app.activity import is_moving
from app.attempts import AttemptTracker, attempt_summary
from app.chat import CoachChat, chat_facts
from app.orchestrator import VoiceOrchestrator
from app.stt import STT_TIMEOUT_S, get_stt
from app.move_catalog import load_catalog, preview as move_preview

# Every route needs the shared ACCESS_TOKEN when one is set.
# The interactive docs are off so they cannot be used to browse the API.
# Pages poll these every few seconds. Drop their successful requests from the
# uvicorn access log so the terminal (and the machine) is not flooded.
QUIET_POLL_PATHS = ("/imu/status", "/sessions/latest")


class _QuietPollFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if isinstance(args, tuple) and len(args) >= 5:
            path, status = str(args[2]).split("?", 1)[0], args[4]
            if path in QUIET_POLL_PATHS and status == 200:
                return False
        return True


logging.getLogger("uvicorn.access").addFilter(_QuietPollFilter())

app = FastAPI(
    title="Breakdance Coach API",
    dependencies=[Depends(require_token)],
    docs_url=None, redoc_url=None, openapi_url=None,
)
if not configured_token():
    print("WARNING: ACCESS_TOKEN is not set. Do not expose this server through a tunnel until you set it.")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REFERENCE_DIR  = Path(__file__).resolve().parent.parent / "reference_moves"
KEYFRAME_FLASH = 30
# Consecutive frames without a tracked body before the pose buffer is cleared,
# so a stale window is never compared or scored again. A few frames of grace
# rides out MediaPipe dropping a single frame mid-move.
LOST_RESET_FRAMES = 10
# One rep = one full window of new tracked frames. Without this, every DTW
# result (several a second) counted as an attempt.
REP_FRAMES = WINDOW_LEN
_sessions: dict[str, "CoachingSession"] = {}
_executor      = ThreadPoolExecutor(max_workers=4)
# TTS gets its own small pool so a slow synth never steals a pose-frame slot.
# Two workers let two sessions synthesize at once; each VoiceCoach serializes
# its own Kokoro calls. The timeout below includes queue time on purpose: audio
# that arrives after the deadline is stale for a live cue.
_tts_executor  = ThreadPoolExecutor(max_workers=2)
_stt_executor  = ThreadPoolExecutor(max_workers=1)   # one clip at a time
TTS_TIMEOUT_SECONDS = 5.0

# One shared IMU store for the whole server: there's one physical device.
# The /ws/imu endpoint writes into it; frame processing reads from it.
_imu_store     = ImuStore()

BODY_CONNECTIONS = [
    [11,12],[11,13],[13,15],[12,14],[14,16],
    [11,23],[12,24],[23,24],
    [23,25],[25,27],[24,26],[26,28],
]

GHOST_BONES = [
    [0.00,-1.80, 0.00,-1.50],
    [-0.50,-1.45, 0.50,-1.45],
    [-0.50,-1.45,-1.10,-1.45],
    [-1.10,-1.45,-1.65,-1.45],
    [0.50,-1.45, 1.10,-1.45],
    [1.10,-1.45, 1.65,-1.45],
    [-0.50,-1.45,-0.20,-0.80],
    [0.50,-1.45, 0.20,-0.80],
    [-0.20,-0.80, 0.20,-0.80],
    [-0.20,-0.80,-0.22,-0.20],
    [0.20,-0.80, 0.22,-0.20],
    [-0.22,-0.20,-0.22, 0.48],
    [0.22,-0.20, 0.22, 0.48],
]


# Spoken cues while the dancer is mid-rep distract newer dancers, so the coach
# waits for the pause and debriefs the attempt. The dancer can switch them on.
LIVE_CUES_DEFAULT = False


class SessionConfig(BaseModel):
    voice_gender: str = "female"
    zoom: float       = 1.0
    move_id: Optional[str] = None   # practice one move; None compares against all


def _diagnose_visibility(pose_frame) -> Optional[str]:
    vis          = pose_frame.visibility
    hips_ok      = vis[23] >= 0.4 and vis[24] >= 0.4
    shoulders_ok = vis[11] >= 0.4 and vis[12] >= 0.4
    feet_ok      = vis[27] >= 0.4 and vis[28] >= 0.4

    if not hips_ok and not shoulders_ok:
        return "Too close — step back until full body is visible"
    if not hips_ok:
        return "Step back — hips not in frame"
    if not feet_ok:
        return "Step back — feet not in frame"
    return None


def _live_visibility(pose_frame) -> list[float]:
    """Per-landmark visibility of the current frame, or [] when no pose was found."""
    return pose_frame.visibility.tolist() if pose_frame is not None else []


def _imu_payload(captured_at: float) -> dict:
    """IMU health plus how closely an IMU sample lines up with this frame.

    offset_ms is the gap between the frame's capture time and the nearest IMU
    sample. Small values (under ~20ms at 50Hz) mean the two streams are in sync.
    None means no IMU sample was close enough to trust.
    """
    status = _imu_store.status()
    nearest = _imu_store.nearest(captured_at)
    status["offset_ms"] = (
        round((nearest.host_time - captured_at) * 1000, 1) if nearest else None
    )
    return status


class SessionClosed(Exception):
    """The session was ended (DELETE /session) while its socket was still sending frames."""


def _track(session: "CoachingSession", task: asyncio.Task) -> None:
    session.coach_tasks.add(task)
    task.add_done_callback(session.coach_tasks.discard)


def _score_payload(result) -> dict:
    """One scored rep as the client (and the coach) see it."""
    return {
        "points":  round(result.points_this_attempt),
        "clean":   all(r.tier != "Miss" for r in result.results),
        "moments": list(getattr(result, "moments", [])),
        "results": [
            {
                "joint":  r.joint_name.replace("_", " "),
                "tier":   r.tier,
                "diff":   round(r.diff, 1),
                # which way to move: straighten/bend for joint angles,
                # twist/reach for whole-rep features
                "fix":    r.fix_direction,
                "points": round(r.points_earned),
            }
            for r in result.results
        ],
    }


class CoachingSession:
    def __init__(self, config: SessionConfig) -> None:
        # close() and frame processing run on different threads; this lock keeps
        # close() from tearing down the estimator or DTW pool mid-frame.
        self._lock          = threading.Lock()
        self.closed         = False
        self.estimator      = PoseEstimator(model_complexity=1)
        self.buf            = PoseBuffer()
        self.move_id        = config.move_id
        self.engine         = DTWEngine(REFERENCE_DIR, only_move=config.move_id)
        self.scorer         = Scorer(REFERENCE_DIR)
        self.ai_coach       = AICoach()
        self.ai_coach.live_cues = LIVE_CUES_DEFAULT
        self.voice          = VoiceCoach(gender=config.voice_gender)
        self.future         = None
        self.flash_counter  = 0
        self.score_result   = None
        self.dtw_result     = None
        self.pose_frame     = None
        self.frame_idx      = 0
        self.last_captured_at: Optional[float] = None
        self.freeze         = FreezeMonitor(_imu_store)
        self.pending_cue: Optional[asyncio.Task] = None
        self.lost_frames    = 0
        self.frames_since_rep = 0
        self.future_window: list = []
        self.future_frames: list = []
        self.attempts       = AttemptTracker()
        # One coach line at a time: a debrief waits for a live cue still in
        # flight instead of talking over it. Tasks are kept so they can't be
        # garbage-collected mid-flight.
        self.speak_lock     = asyncio.Lock()
        self.coach_tasks: set[asyncio.Task] = set()
        self.viewers        = ViewerHub()     # read-only second screens (TV, laptop)
        self.controller: Optional[WebSocket] = None   # the phone's socket
        self.previous_rep: Optional[dict] = None   # payload of the rep before the latest one
        self.latest_rep: Optional[dict] = None
        # What the dancer's own questions may be answered from (app/chat.py).
        self.chat           = CoachChat()
        self.last_stats: Optional[dict] = None
        self.last_summary: Optional[dict] = None
        self.last_move_name = ""
        self.orchestrator: Optional[VoiceOrchestrator] = None

        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self.estimator.process(dummy)

    def process_frame_sync(self, b64: str, captured_at: Optional[float] = None) -> dict:
        """Synchronous frame processing — runs in thread pool.

        captured_at is the laptop time (seconds) when the browser grabbed the
        frame. It uses the same clock as the IMU bridge, so the two streams can
        be lined up. Falls back to arrival time for older frontends.
        """
        with self._lock:
            if self.closed:
                raise SessionClosed()
            return self._process_frame(b64, captured_at)

    def _process_frame(self, b64: str, captured_at: Optional[float]) -> dict:
        captured_at = captured_at if captured_at is not None else time.time()
        self.last_captured_at = captured_at

        data  = base64.b64decode(b64)
        arr   = np.frombuffer(data, np.uint8)
        bgr   = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        frame = cv2.flip(bgr, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose  = self.estimator.process(rgb, keep_raw=True)
        if isinstance(pose, PoseFrame):
            pose = dataclasses.replace(pose, captured_at=captured_at)

        guidance = None
        tracked  = pose is not None and is_pose_reliable(pose)
        if tracked:
            self.lost_frames = 0
            self.pose_frame = pose
            if self.buf.add(pose):
                self.frames_since_rep += 1
            guidance = _diagnose_visibility(pose)
        else:
            guidance = "No body detected — step into frame"
            if pose is not None:
                guidance = _diagnose_visibility(pose)
            self.lost_frames += 1
            if self.lost_frames == LOST_RESET_FRAMES:
                # Body gone: drop the old window so it is not compared or
                # scored again while nobody is dancing.
                self.buf.clear()
                self.frames_since_rep = 0

        window = [f.landmarks for f in self.buf._frames]
        moving = tracked and is_moving(window)

        if tracked and self.buf.is_ready() and self.future is None:
            self.future = self.engine.compare_async(window)
            # DTW path indices point into this exact window, not a later one
            self.future_window = window
            self.future_frames = list(self.buf._frames)

        if self.future is not None and self.future.done():
            try:
                self.dtw_result = self.future.result()
            except Exception:
                self.dtw_result = None
            self.future = None

            # Score only a live, moving body, and once per rep. Standing still
            # or out of frame used to keep scoring the last window over and over.
            rep_ready = self.frames_since_rep >= REP_FRAMES
            if self.dtw_result and self.dtw_result.aligned and tracked and moving and rep_ready:
                s = self.scorer.score_rep(self.dtw_result, self.future_window, self.future_frames)
                if s is not None:
                    self.score_result  = s
                    self.flash_counter = KEYFRAME_FLASH
                    self.frames_since_rep = 0
                    rep_payload = _score_payload(s)
                    self.previous_rep, self.latest_rep = self.latest_rep, rep_payload
                    self.attempts.add_rep(rep_payload)

        # The dancer stopped after scored reps: hand the attempt to the debrief.
        finished_attempt = self.attempts.on_frame(moving, captured_at)

        if self.flash_counter > 0:
            self.flash_counter -= 1

        # IMU decides when a freeze happens and how steady it was; this
        # frame's pose lets the camera veto standing pauses.
        freeze = self.freeze.on_frame(
            pose.raw_landmarks if pose is not None else None,
            pose.visibility if pose is not None else None,
        )

        fill_ratio = len(self.buf) / WINDOW_LEN
        move_name  = (self.dtw_result.move_name
                      if self.dtw_result else self.engine.move_name)
        stats      = self.scorer.stats()
        self.frame_idx += 1

        landmarks  = []
        visibility = []
        hip_cx = hip_cy = scale = None

        if self.pose_frame is not None and self.pose_frame.raw_landmarks is not None:
            raw = self.pose_frame.raw_landmarks
            vis = self.pose_frame.visibility
            landmarks  = raw.tolist()
            visibility = vis.tolist()
            lh, rh = raw[23], raw[24]
            ls, rs = raw[11], raw[12]
            if all(vis[i] >= 0.4 for i in [11, 12, 23, 24]):
                hip_cx = float((lh[0] + rh[0]) / 2)
                hip_cy = float((lh[1] + rh[1]) / 2)
                scale  = float(abs(rs[0] - ls[0]) * 0.9)

        score_payload = None
        if self.score_result and self.flash_counter > 0:
            score_payload = _score_payload(self.score_result)

        ref_meta = self._move_meta(move_name)

        return {
            "type":         "frame",
            "landmarks":    landmarks,
            "visibility":   visibility,
            # This frame's own visibility. "visibility" above is held from the
            # last reliable pose, so only this one shows a dancer leaving frame.
            "live_visibility": _live_visibility(pose),
            "ghost_bones":  GHOST_BONES if fill_ratio >= 1.0 else [],
            "ghost_anchor": {
                "hip_cx": hip_cx,
                "hip_cy": hip_cy,
                "scale":  scale,
            } if hip_cx is not None else None,
            "aligned":      bool(self.dtw_result.aligned) if self.dtw_result else False,
            "fill_ratio":   round(fill_ratio, 3),
            "move_name":    move_name or "",
            "guidance":     guidance,
            "tracked":      tracked,
            "moving":       moving,
            "stats": {
                "total_points": round(stats.total_points),
                "grade":        stats.grade,
                "streak":       stats.streak,
                "perfects":     stats.perfects,
                "closes":       stats.closes,
                "misses":       stats.misses,
                "reps":         stats.reps,
                "clean_reps":   stats.clean_reps,
                "clean_streak": stats.clean_streak,
                "best_streak":  stats.best_streak,
                "focus_joint":  stats.focus_joint.replace("_", " ") if stats.focus_joint else None,
                "focus_fix":    stats.focus_fix,
            },
            "score_result": score_payload,
            "imu":          _imu_payload(captured_at),
            "freeze":       freeze,
            # snapshot for AI coach — passed separately in ws handler
            "_coach_context": {
                "move_name":    ref_meta.get("display_name") or (move_name or "").replace("_", " ").upper(),
                "move_tips":    list(ref_meta.get("tips", [])),
                "tracked":      tracked,
                "moving":       moving,
                "fill_ratio":   round(fill_ratio, 3),
                "score_result": score_payload,
                "previous_result": self.previous_rep if score_payload else None,
                "clean_streak": stats.clean_streak,
                "freeze":       freeze["result"],
                "debrief_reps": finished_attempt,
                "session_totals": {
                    "reps":        stats.reps,
                    "clean_reps":  stats.clean_reps,
                    "best_streak": stats.best_streak,
                },
            },
        }

    def demo(self) -> Optional[tuple[dict, str, float]]:
        """Demo of the practiced move, focused on the joint the scorer flags most.

        None when there is no single move to show (no move picked and none
        matched yet).
        """
        move_id = self.move_id or (self.dtw_result.move_name if self.dtw_result else None) \
            or self._last_matched_move()
        if not move_id or not (REFERENCE_DIR / f"{move_id}.npy").exists():
            return None
        meta = self._move_meta(move_id)
        name = meta.get("display_name") or move_id.replace("_", " ").upper()
        stats = self.scorer.stats()
        try:
            seconds = float(meta.get("duration_s", 3.0))
        except (TypeError, ValueError):
            seconds = 3.0
        return demo_message(move_id, name, stats.focus_joint, stats.focus_fix, seconds)

    def _last_matched_move(self) -> Optional[str]:
        """Move from the last completed comparison, kept when dtw_result is cleared."""
        last = getattr(self.engine, "last_result", None)
        return last.move_name if last is not None else None

    def _move_meta(self, move_name: Optional[str]) -> dict:
        move = self.engine.get_move(move_name) if move_name else None
        return move.meta if move is not None else {}

    def close(self) -> None:
        with self._lock:
            if self.closed:
                return
            self.closed = True
            self.estimator.close()
            self.engine.shutdown()


def voice_status(voice: Optional[VoiceCoach]) -> dict:
    """Whether this session can speak, and why not. Sent once when the socket opens."""
    if voice is None:
        return {"type": "voice_status", "available": False, "reason": "no voice for this session"}
    if not voice.available:
        reason = getattr(voice, "unavailable_reason", None) or "TTS unavailable"
        return {"type": "voice_status", "available": False, "reason": reason}
    return {"type": "voice_status", "available": True, "reason": None}


def warm_up_voice(voice: Optional[VoiceCoach]) -> None:
    """Run one throwaway synth so the first real cue does not pay Kokoro's cold start.

    On slower CPUs the first synth can take longer than TTS_TIMEOUT_SECONDS,
    which would drop the first cue's audio. Logs how long it took either way.
    """
    if voice is None or not voice.available:
        reason = voice_status(voice)["reason"]
        print(f"VoiceCoach: COACH AUDIO OFF ({reason}). Captions only this session.")
        return
    start = time.perf_counter()
    wav = voice.synthesize_wav("Ready")
    ms = (time.perf_counter() - start) * 1000
    if wav:
        print(f"VoiceCoach: warm-up synth took {ms:.0f} ms (cue timeout is {TTS_TIMEOUT_SECONDS:.0f} s)")
    else:
        print(f"VoiceCoach: warm-up synth FAILED after {ms:.0f} ms; cues will be captions only")


async def _send_text(websocket: WebSocket, text: str, viewers: Optional[ViewerHub] = None) -> None:
    """Send to the phone, then mirror to any viewers (a viewer failure never blocks the phone)."""
    await websocket.send_text(text)
    if viewers is not None:
        await viewers.broadcast_text(text)


async def notify_viewer_count(session) -> None:
    """Tell the phone how many second screens are watching, so it can mute its own audio."""
    controller = getattr(session, "controller", None)
    hub = getattr(session, "viewers", None)
    if controller is None or hub is None:
        return
    try:
        await controller.send_text(json.dumps({"type": "viewers", "count": hub.count}))
    except Exception:
        pass  # phone socket closed


async def _audio_skipped(websocket: WebSocket, turn_id: str, cue: str, reason: str,
                         viewers: Optional[ViewerHub] = None) -> bool:
    print(f"TTS skipped ({reason}), sent caption only: '{cue}'")
    await _send_text(websocket, json.dumps(
        {"type": "coach_audio_skipped", "turn_id": turn_id, "reason": reason}
    ), viewers)
    return False


def make_orchestrator(session: "CoachingSession", websocket: WebSocket,
                      viewers: Optional[ViewerHub]) -> VoiceOrchestrator:
    """User turns for this phone socket. The conversation (history, limits) lives on the session."""
    async def emit(msg: dict) -> None:
        try:
            await _send_text(websocket, json.dumps(msg), viewers)
        except Exception:
            pass  # phone socket closed

    async def speak(text: str) -> None:
        async with session.speak_lock:
            try:
                await send_coach_cue(websocket, session.voice, text, mode="reply", viewers=viewers)
            except Exception as e:
                print(f"Reply not sent (socket closed?): {e}")

    def facts() -> dict:
        return chat_facts(getattr(session, "last_move_name", ""), getattr(session, "last_stats", None),
                          getattr(session, "last_summary", None))

    stt = get_stt()

    async def transcribe(pcm: bytes):
        loop = asyncio.get_running_loop()
        return await asyncio.wait_for(loop.run_in_executor(_stt_executor, stt.transcribe, pcm), STT_TIMEOUT_S)

    if getattr(session, "chat", None) is None:
        session.chat = CoachChat()   # sessions built without __init__ (tests)
    session.orchestrator = VoiceOrchestrator(emit, speak, facts, chat=session.chat,
                                             transcribe=transcribe, stt_available=stt.available)
    return session.orchestrator


async def send_coach_cue(
    websocket: WebSocket,
    voice: Optional[VoiceCoach],
    cue: str,
    timeout: float = TTS_TIMEOUT_SECONDS,
    mode: str = "live",
    viewers: Optional[ViewerHub] = None,
) -> bool:
    """Send a coach cue to the client: caption text first, then the spoken audio.

    Text goes first so the caption shows even when TTS is slow, missing or
    fails. When there is no audio, a coach_audio_skipped message says why, so
    the client can show it. Returns True if audio was sent. Everything sent to
    the phone is mirrored to ``viewers`` too.
    """
    turn_id = uuid.uuid4().hex[:12]
    await _send_text(websocket, json.dumps({"type": "coach_cue", "cue": cue, "turn_id": turn_id, "mode": mode}), viewers)

    status = voice_status(voice)
    if not status["available"]:
        return await _audio_skipped(websocket, turn_id, cue, status["reason"], viewers)
    loop = asyncio.get_running_loop()
    start = time.perf_counter()
    try:
        wav = await asyncio.wait_for(
            loop.run_in_executor(_tts_executor, voice.synthesize_wav, cue), timeout
        )
    except asyncio.TimeoutError:
        return await _audio_skipped(websocket, turn_id, cue, f"TTS timed out after {timeout:g} s", viewers)
    if not wav:
        return await _audio_skipped(websocket, turn_id, cue, "TTS synth failed, see log above", viewers)
    frame = encode_audio_chunk(turn_id, wav)
    await websocket.send_bytes(frame)
    if viewers is not None:
        await viewers.broadcast_bytes(frame)
    ms = (time.perf_counter() - start) * 1000
    print(f"TTS sent {len(wav)} bytes in {ms:.0f} ms: '{cue}'")
    return True


@app.get("/health")
async def health():
    return {"status": "ok", "moves": len(list(REFERENCE_DIR.glob("*.npy")))}


@app.get("/moves")
async def list_moves():
    return {"moves": [m.to_dict() for m in load_catalog(REFERENCE_DIR)]}


@app.get("/moves/{move_id}/preview")
async def get_move_preview(move_id: str):
    data = move_preview(REFERENCE_DIR, move_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"No move named {move_id}")
    return data


@app.post("/session/start")
async def start_session(config: SessionConfig):
    sid = str(uuid.uuid4())
    session = CoachingSession(config)
    _sessions[sid] = session
    _tts_executor.submit(warm_up_voice, session.voice)
    stt = get_stt()
    if stt.available and hasattr(stt, "warm_up"):
        _stt_executor.submit(stt.warm_up)   # first use downloads and loads the model
    return {"session_id": sid}


@app.delete("/session/{session_id}")
async def end_session(session_id: str):
    session = _sessions.pop(session_id, None)
    if session is not None:
        # run off the event loop: close() waits for an in-flight frame to finish
        await asyncio.get_running_loop().run_in_executor(_executor, session.close)
        hub = getattr(session, "viewers", None)
        if hub is not None:
            await hub.close_all()
    return {"status": "closed"}


@app.get("/sessions/latest")
async def latest_session():
    """The newest open session with a phone connected, so a second screen (/view) can find it."""
    for sid in reversed(list(_sessions)):
        s = _sessions[sid]
        if not getattr(s, "closed", False) and getattr(s, "controller", None) is not None:
            return {"session_id": sid}
    return {"session_id": None}


@app.get("/stt/status")
async def stt_status():
    """Whether push-to-talk can hear on this backend, and why not."""
    stt = get_stt()
    return {"available": stt.available, "reason": stt.unavailable_reason}


@app.get("/imu/status")
async def imu_status():
    return _imu_store.status()


@app.websocket("/ws/imu")
async def imu_endpoint(websocket: WebSocket):
    """Receives forwarded IMU packets from scripts/imu_bridge.py."""
    await websocket.accept()
    try:
        while True:
            message = await websocket.receive_bytes()
            try:
                host_time, packet = unpack_forwarded(message)
            except PacketError:
                continue  # skip a malformed message instead of dropping the connection
            _imu_store.add(host_time, packet)
    except WebSocketDisconnect:
        pass


@app.websocket("/ws/{session_id}/view")
async def viewer_endpoint(websocket: WebSocket, session_id: str):
    """Read-only second screen: gets what the phone gets, sends nothing but pings."""
    await websocket.accept()
    session = _sessions.get(session_id)
    if not session:
        await websocket.send_json({"error": "session not found"})
        await websocket.close()
        return

    await websocket.send_text(json.dumps(voice_status(session.voice)))
    session.viewers.add(websocket)
    await notify_viewer_count(session)
    try:
        while True:
            try:
                msg = json.loads(await websocket.receive_text())
            except ValueError:
                continue  # a viewer only pings; ignore anything else
            if isinstance(msg, dict) and msg.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        session.viewers.discard(websocket)
        await notify_viewer_count(session)


@app.websocket("/ws/{session_id}")
async def ws_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    session = _sessions.get(session_id)
    if not session:
        await websocket.send_json({"error": "session not found"})
        await websocket.close()
        return

    loop = asyncio.get_running_loop()
    await websocket.send_text(json.dumps(voice_status(session.voice)))
    session.controller = websocket
    viewers = getattr(session, "viewers", None)
    if viewers is not None and viewers.count:
        await notify_viewer_count(session)   # a reconnecting phone learns who is watching
    orchestrator = make_orchestrator(session, websocket, viewers)

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)

            if msg.get("type") == "frame":
                # run pose pipeline in thread pool
                try:
                    payload = await loop.run_in_executor(
                        _executor, session.process_frame_sync, msg["data"], msg.get("t")
                    )
                except SessionClosed:
                    # END SESSION raced a frame still in flight; close cleanly
                    await websocket.close(code=1000)
                    return

                # extract coach context then remove from payload
                coach_ctx = payload.pop("_coach_context", {})

                # send frame data to frontend immediately
                await _send_text(websocket, json.dumps(payload), viewers)
                session.last_stats = payload.get("stats")
                session.last_move_name = coach_ctx.get("move_name", "") or getattr(session, "last_move_name", "")

                # fire AI coach — non-blocking.
                # A finished freeze gets feedback right away (skips the cooldown);
                # everything else waits for the normal cooldown.
                coach = session.ai_coach
                debrief_reps = coach_ctx.get("debrief_reps")
                if debrief_reps and coach.held:
                    # The dancer is watching a demo; the debrief would talk over it.
                    continue
                if debrief_reps:
                    # The result card first (shows even with no API key), then the spoken debrief.
                    summary = attempt_summary(coach_ctx.get("move_name", ""), debrief_reps)
                    session.last_summary = summary
                    await _send_text(websocket, json.dumps(summary), viewers)
                    # The dancer just stopped: one debrief for that attempt.
                    async def _fire_debrief(ctx=coach_ctx, reps=debrief_reps):
                        async with session.speak_lock:
                            text = await session.ai_coach.get_debrief(
                                ctx.get("move_name", ""), reps, ctx.get("session_totals"),
                            )
                            # A demo may have started while the debrief was being written.
                            if text and not session.ai_coach.held:
                                try:
                                    await send_coach_cue(websocket, session.voice, text, mode="debrief", viewers=viewers)
                                except Exception as e:
                                    print(f"Debrief not sent (socket closed?): {e}")

                    _track(session, asyncio.create_task(_fire_debrief()))
                    continue

                freeze_result = coach_ctx.get("freeze")
                # Nothing is said while no body is tracked or the dancer is still.
                fire = coach.can_fire_now() if freeze_result else coach.should_fire(
                    tracked=coach_ctx.get("tracked", False),
                    moving=coach_ctx.get("moving", False),
                    fill_ratio=coach_ctx.get("fill_ratio", 0),
                    has_score=bool(coach_ctx.get("score_result")),
                )
                if fire and orchestrator.busy:
                    fire = False   # the dancer is talking to the coach; no cue over it
                if fire:
                    async def _fire_cue(ctx=coach_ctx):
                        async with session.speak_lock:
                            cue = await session.ai_coach.get_cue(
                                move_name    = ctx.get("move_name", ""),
                                score_result = ctx.get("score_result"),
                                freeze       = ctx.get("freeze"),
                                move_tips    = ctx.get("move_tips"),
                                previous_result = ctx.get("previous_result"),
                                clean_streak = ctx.get("clean_streak", 0),
                            )
                            # A demo may have started while the cue was being written.
                            if cue and not session.ai_coach.held and not orchestrator.busy:
                                # caption + Kokoro audio, streamed to the client
                                try:
                                    await send_coach_cue(websocket, session.voice, cue, viewers=viewers)
                                except Exception:
                                    pass  # socket closed mid-send

                    _track(session, asyncio.create_task(_fire_cue()))

            elif msg.get("type") == "demo_request":
                # "Show me": play the move slowly, say what to watch, and keep
                # live cues quiet while the dancer is watching.
                try:
                    demo = session.demo()
                except Exception as e:
                    print(f"Demo request failed: {e}")
                    demo = None
                if demo is None:
                    await websocket.send_text(json.dumps(
                        {"type": "demo_unavailable", "reason": "Pick a move first"}))
                    continue
                # The demo is silent: the overlay shows the move and the joint to watch.
                demo_msg, _line, quiet_s = demo
                session.ai_coach.hold(quiet_s)
                await _send_text(websocket, json.dumps(demo_msg), viewers)

            elif msg.get("type") == "chat":
                # A typed question. Answered off the receive loop so frames keep flowing.
                text = msg.get("text")
                if isinstance(text, str):
                    _track(session, asyncio.create_task(orchestrator.handle_text(text, "text")))

            elif msg.get("type") == "ptt_start":
                await orchestrator.ptt_start()

            elif msg.get("type") == "ptt_audio":
                orchestrator.ptt_audio(msg.get("data"))

            elif msg.get("type") == "ptt_end":
                # Speech to text can take a while; keep reading frames meanwhile.
                _track(session, asyncio.create_task(orchestrator.ptt_end()))

            elif msg.get("type") == "ptt_cancel":
                await orchestrator.ptt_cancel()

            elif msg.get("type") == "live_cues":
                session.ai_coach.live_cues = bool(msg.get("on"))
                await websocket.send_text(json.dumps(
                    {"type": "live_cues", "on": session.ai_coach.live_cues}))

            elif msg.get("type") == "demo_end":
                # The dancer closed the demo (or it finished): live cues resume.
                session.ai_coach.release()
                if viewers is not None:
                    await viewers.broadcast_text(json.dumps({"type": "demo_end"}))

            elif msg.get("type") == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        pass
    finally:
        # The session is kept for reconnection, but a second screen should not
        # sit on a phone that is gone: closing it makes /view look for the next one.
        if session.controller is websocket:
            session.controller = None
            if viewers is not None:
                await viewers.close_all()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)