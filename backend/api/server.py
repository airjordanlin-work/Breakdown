"""
FastAPI backend — WebSocket streaming with real-time AI coaching.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import sys
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.pose_estimator import PoseEstimator, is_pose_reliable
from app.buffer import PoseBuffer, WINDOW_LEN
from app.dtw_engine import DTWEngine
from app.scorer import Scorer
from app.coach import AICoach
from app.voice import VoiceCoach
from app.imu_protocol import PacketError, unpack_forwarded
from app.imu_store import ImuStore
from app.freeze_monitor import FreezeMonitor

app = FastAPI(title="Breakdance Coach API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REFERENCE_DIR  = Path(__file__).resolve().parent.parent / "reference_moves"
KEYFRAME_FLASH = 30
_sessions: dict[str, "CoachingSession"] = {}
_executor      = ThreadPoolExecutor(max_workers=4)

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


class SessionConfig(BaseModel):
    voice_gender: str = "female"
    zoom: float       = 1.0


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


class CoachingSession:
    def __init__(self, config: SessionConfig) -> None:
        self.estimator      = PoseEstimator(model_complexity=1)
        self.buf            = PoseBuffer()
        self.engine         = DTWEngine(REFERENCE_DIR)
        self.scorer         = Scorer(REFERENCE_DIR)
        self.ai_coach       = AICoach()
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

        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self.estimator.process(dummy)

    def process_frame_sync(self, b64: str, captured_at: Optional[float] = None) -> dict:
        """Synchronous frame processing — runs in thread pool.

        captured_at is the laptop time (seconds) when the browser grabbed the
        frame. It uses the same clock as the IMU bridge, so the two streams can
        be lined up. Falls back to arrival time for older frontends.
        """
        captured_at = captured_at if captured_at is not None else time.time()
        self.last_captured_at = captured_at

        data  = base64.b64decode(b64)
        arr   = np.frombuffer(data, np.uint8)
        bgr   = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        frame = cv2.flip(bgr, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose  = self.estimator.process(rgb, keep_raw=True)

        guidance = None
        if pose is not None and is_pose_reliable(pose):
            self.pose_frame = pose
            self.buf.add(pose)
            guidance = _diagnose_visibility(pose)
        else:
            guidance = "No body detected — step into frame"
            if pose is not None:
                guidance = _diagnose_visibility(pose)

        if self.buf.is_ready() and self.future is None:
            window      = [f.landmarks for f in self.buf._frames]
            self.future = self.engine.compare_async(window)

        if self.future is not None and self.future.done():
            try:
                self.dtw_result = self.future.result()
            except Exception:
                self.dtw_result = None
            self.future = None

            if self.dtw_result and self.dtw_result.aligned and self.pose_frame:
                s = self.scorer.score(self.dtw_result, self.pose_frame)
                if s is not None:
                    self.score_result  = s
                    self.flash_counter = KEYFRAME_FLASH

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
            score_payload = {
                "points":  round(self.score_result.points_this_attempt),
                "results": [
                    {
                        "joint":  r.joint_name.replace("_", " "),
                        "tier":   r.tier,
                        "diff":   round(r.diff, 1),
                        "points": round(r.points_earned),
                    }
                    for r in self.score_result.results
                ],
            }

        return {
            "type":         "frame",
            "landmarks":    landmarks,
            "visibility":   visibility,
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
            "stats": {
                "total_points": round(stats.total_points),
                "grade":        stats.grade,
                "streak":       stats.streak,
                "perfects":     stats.perfects,
                "closes":       stats.closes,
                "misses":       stats.misses,
            },
            "score_result": score_payload,
            "imu":          _imu_payload(captured_at),
            "freeze":       freeze,
            # snapshot for AI coach — passed separately in ws handler
            "_coach_context": {
                "move_name":    move_name or "",
                "aligned":      bool(self.dtw_result.aligned) if self.dtw_result else False,
                "fill_ratio":   round(fill_ratio, 3),
                "score_result": score_payload,
                "freeze":       freeze["result"],
            },
        }

    def speak(self, cue: str) -> None:
        """Speak a cue through Kokoro TTS."""
        if self.voice and cue:
            self.voice.speak(cue)

    def close(self) -> None:
        self.estimator.close()
        self.engine.shutdown()


def _infer_difficulty(stem: str) -> str:
    s = stem.lower()
    if any(x in s for x in ["toprock", "basic", "_00", "_01"]): return "Beginner"
    if any(x in s for x in ["footwork", "freeze", "_02", "_03", "_04"]): return "Intermediate"
    return "Advanced"


@app.get("/health")
async def health():
    return {"status": "ok", "moves": len(list(REFERENCE_DIR.glob("*.npy")))}


@app.get("/moves")
async def list_moves():
    moves = []
    for npy in sorted(REFERENCE_DIR.glob("*.npy")):
        meta_path = REFERENCE_DIR / f"{npy.stem}_meta.json"
        meta = {}
        if meta_path.exists():
            with meta_path.open() as f:
                meta = json.load(f)
        moves.append({
            "id":         npy.stem,
            "name":       meta.get("name", npy.stem).replace("_", " ").upper(),
            "source":     meta.get("source", "custom"),
            "difficulty": _infer_difficulty(npy.stem),
            "original":   meta.get("original_file", ""),
        })
    return {"moves": moves}


@app.post("/session/start")
async def start_session(config: SessionConfig):
    sid = str(uuid.uuid4())
    _sessions[sid] = CoachingSession(config)
    return {"session_id": sid}


@app.delete("/session/{session_id}")
async def end_session(session_id: str):
    if session_id in _sessions:
        _sessions[session_id].close()
        del _sessions[session_id]
    return {"status": "closed"}


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


@app.websocket("/ws/{session_id}")
async def ws_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    session = _sessions.get(session_id)
    if not session:
        await websocket.send_json({"error": "session not found"})
        await websocket.close()
        return

    loop = asyncio.get_running_loop()

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)

            if msg.get("type") == "frame":
                # run pose pipeline in thread pool
                payload = await loop.run_in_executor(
                    _executor, session.process_frame_sync, msg["data"], msg.get("t")
                )

                # extract coach context then remove from payload
                coach_ctx = payload.pop("_coach_context", {})

                # send frame data to frontend immediately
                await websocket.send_text(json.dumps(payload))

                # fire AI coach — non-blocking.
                # A finished freeze gets feedback right away (skips the cooldown);
                # everything else waits for the normal cooldown.
                coach = session.ai_coach
                freeze_result = coach_ctx.get("freeze")
                fire = coach.can_fire_now() if freeze_result else coach.should_fire(
                    aligned=coach_ctx.get("aligned", False),
                    fill_ratio=coach_ctx.get("fill_ratio", 0),
                )
                if fire:
                    async def _fire_cue(ctx=coach_ctx):
                        cue = await session.ai_coach.get_cue(
                            move_name    = ctx.get("move_name", ""),
                            aligned      = ctx.get("aligned", False),
                            fill_ratio   = ctx.get("fill_ratio", 0),
                            score_result = ctx.get("score_result"),
                            freeze       = ctx.get("freeze"),
                        )
                        if cue:
                            # speak through Kokoro
                            session.speak(cue)
                            # also send to frontend for display
                            try:
                                await websocket.send_text(json.dumps({
                                    "type": "coach_cue",
                                    "cue":  cue,
                                }))
                            except Exception:
                                pass

                    asyncio.create_task(_fire_cue())

            elif msg.get("type") == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        pass
    finally:
        pass  # session kept alive for reconnection


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)