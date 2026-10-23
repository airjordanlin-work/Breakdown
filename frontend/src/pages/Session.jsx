import { useState, useEffect, useCallback, useRef } from "react";
import { useWebSocket } from "../hooks/useWebSocket";
import { useCamera }    from "../hooks/useCamera";
import { useCoachAudio } from "../hooks/useCoachAudio";
import { apiFetch }           from "../api";
import ScoreHUD         from "../components/ScoreHUD";
import KeyframeFlash    from "../components/KeyframeFlash";
import SkeletonCanvas   from "../components/SkeletonCanvas";
import CoachCue, { CAPTION_SIZES } from "../components/CoachCue";
import FramingGuide     from "../components/FramingGuide";
import { INITIAL_FRAMING, nextFraming } from "../framing";
import "../styles/session.css";
import FreezeMeter      from "../components/FreezeMeter";
import DemoOverlay      from "../components/DemoOverlay";
import ResultCard       from "../components/ResultCard";
import ReplayOverlay    from "../components/ReplayOverlay";
import { useFrameBuffer } from "../hooks/useFrameBuffer";
import { useMicCapture, MAX_HOLD_MS } from "../hooks/useMicCapture";
import TalkButton       from "../components/TalkButton";
import ChatBox          from "../components/ChatBox";

// How much of the run-up to the mistake the replay shows, in seconds.
const REPLAY_LEAD_S = 1.5;

const CAPTION_SIZE_KEY = "breakdown_caption_size";

// Saved caption size index (0 small, 1 medium, 2 large), or small if unavailable.
function loadCaptionSize() {
  try {
    const n = Number(localStorage.getItem(CAPTION_SIZE_KEY));
    return Number.isInteger(n) && n >= 0 && n < CAPTION_SIZES.length ? n : 0;
  } catch {
    return 0;
  }
}

export default function Session({ sessionId, onEnd }) {
  const [landmarks,   setLandmarks]   = useState([]);
  const [visibility,  setVisibility]  = useState([]);
  const [ghostBones,  setGhostBones]  = useState([]);
  const [ghostAnchor, setGhostAnchor] = useState(null);
  const [aligned,     setAligned]     = useState(false);
  const [fillRatio,   setFillRatio]   = useState(0);
  const [moveName,    setMoveName]    = useState("");
  const [guidance,    setGuidance]    = useState(null);
  const [stats,       setStats]       = useState(null);
  const [scoreResult, setScoreResult] = useState(null);
  const [debugMode,   setDebugMode]   = useState(false);
  const [latencyMs,   setLatencyMs]   = useState(null);
  const [wsReady,     setWsReady]     = useState(false);
  const [imu,         setImu]         = useState(null);
  const [coachCue,    setCoachCue]    = useState(null);
  const [freezeLive,  setFreezeLive]  = useState(null);
  const [freezeDone,  setFreezeDone]  = useState(null);
  const [freezeDebug, setFreezeDebug] = useState(null);
  const [framing,     setFraming]     = useState(INITIAL_FRAMING);
  const [capSize,     setCapSize]     = useState(loadCaptionSize);

  const cycleCaptionSize = () => setCapSize((i) => {
    const next = (i + 1) % CAPTION_SIZES.length;
    try { localStorage.setItem(CAPTION_SIZE_KEY, String(next)); } catch { /* private mode */ }
    return next;
  });
  const [demo,        setDemo]        = useState(null);
  // Result card for the attempt just finished; it clears when the next rep is scored.
  const [summary, setSummary] = useState(null);
  const frameBuf = useFrameBuffer();
  const [replay, setReplay] = useState(null);   // { frames, fix } while the mistake replay is open
  const repsNow = useRef(0);
  useEffect(() => {
    if (summary && (stats?.reps ?? 0) > summary.at) setSummary(null);
  }, [stats, summary]);
  // REPLAY needs the frame from the moment of the miss to still be in this tab's memory.
  const missFix = summary?.fix;
  const replayable = missFix?.t != null && frameBuf.between(missFix.t - 0.3, missFix.t + 0.3).length > 0;
  const openReplay = useCallback(() => {
    if (!missFix) return;
    setReplay({ frames: frameBuf.between(missFix.t - REPLAY_LEAD_S, missFix.t + 0.3), fix: missFix });
  }, [missFix, frameBuf]);
  const demoOpen = useRef(false);   // mirrors `demo` so handleMessage can read it without re-creating

  // Why there is no sound, when the backend knows. null means audio is fine so far.
  const [audioNote, setAudioNote] = useState(null);

  // Talking to the coach: what it is doing, what it heard, and whether it can hear at all.
  const [voiceState, setVoiceState] = useState("idle");
  const [heard, setHeard] = useState(null);
  const heardTimer = useRef(null);
  const [sttOk, setSttOk] = useState(null);   // null until /stt/status answers
  const [micNote, setMicNote] = useState(null);
  const [holding, setHolding] = useState(false);
  useEffect(() => {
    let alive = true;
    apiFetch("/stt/status")
      .then((r) => r.json())
      .then((s) => alive && setSttOk(Boolean(s.available)))
      .catch(() => alive && setSttOk(false));
    return () => { alive = false; };
  }, []);

  // stopCapture is defined below; the ref lets handleEnd stop frames before
  // the backend tears the session down.
  const stopCaptureRef = useRef(null);
  const handleEnd = async () => {
    stopCaptureRef.current?.();
    try {
      await apiFetch(`/session/${sessionId}`, { method: "DELETE" });
    } finally {
      onEnd();
    }
  };

  // Coach audio comes from the backend over the WebSocket.
  const { unlocked: soundOn, unlock: unlockSound, playChunk: playOnPhone, setLatestTurn } = useCoachAudio();

  // With a TV or laptop watching (/view), it plays the voice and the phone stays quiet.
  const viewerCount = useRef(0);
  const playChunk = useCallback((buf) => {
    if (viewerCount.current > 0) return;
    playOnPhone(buf);
  }, [playOnPhone]);

  const handleMessage = useCallback((msg) => {
    if (msg.type === "voice_status") {
      setAudioNote(msg.available ? null : `Coach voice is off: ${msg.reason}`);
      return;
    }
    if (msg.type === "rep_summary") {
      setSummary({ ...msg, at: repsNow.current });
      return;
    }
    if (msg.type === "voice_state") {
      if (!msg.busy) setVoiceState(msg.state);
      return;
    }
    if (msg.type === "transcript") {
      setHeard(msg.text);
      clearTimeout(heardTimer.current);
      heardTimer.current = setTimeout(() => setHeard(null), 6000);
      return;
    }
    if (msg.type === "viewers") {
      viewerCount.current = Number(msg.count) || 0;
      return;
    }
    if (msg.type === "coach_audio_skipped") {
      setAudioNote(`No audio for the last cue: ${msg.reason}`);
      return;
    }
    if (msg.type === "demo") {
      demoOpen.current = true;
      // fresh object and key per request so a repeat SHOW ME remounts the overlay
      setDemo({ ...msg, requestId: Date.now() });
      return;
    }
    if (msg.type === "demo_unavailable") {
      setCoachCue(msg.reason);
      return;
    }
    if (msg.type === "coach_cue") {
      // A live cue that was already on its way when SHOW ME opened would talk over the demo.
      if (demoOpen.current && msg.mode !== "demo") return;
      setCoachCue(msg.cue);
      setLatestTurn(msg.turn_id);
      return;
    }
    if (msg.type !== "frame") return;
    if (msg.freeze) {
      setFreezeLive(msg.freeze.live);
      setFreezeDebug(msg.freeze);
      // new object each time so the result card re-triggers for back-to-back freezes
      if (msg.freeze.result) setFreezeDone({ ...msg.freeze.result });
    }
    if (msg._rttMs !== undefined)       setLatencyMs(msg._rttMs);
    if (msg.imu) setImu(msg.imu);
    if (msg.landmarks)                  setLandmarks(msg.landmarks);
    if (msg.visibility)                 setVisibility(msg.visibility);
    setFraming((f) => nextFraming(f, msg.live_visibility));
    if (msg.ghost_bones)                setGhostBones(msg.ghost_bones);
    if (msg.ghost_anchor !== undefined) setGhostAnchor(msg.ghost_anchor);
    if (msg.aligned     !== undefined)  setAligned(msg.aligned);
    if (msg.fill_ratio  !== undefined)  setFillRatio(msg.fill_ratio);
    if (msg.move_name)                  setMoveName(msg.move_name);
    if (msg.guidance    !== undefined)  setGuidance(msg.guidance);
    if (msg.stats) {
      setStats(msg.stats);
      repsNow.current = msg.stats.reps;
    }
    if (msg.score_result)               setScoreResult(msg.score_result);
  }, [setLatestTurn]);

  // Browsers (iOS Safari above all) need a tap before they play sound, so try
  // to unlock on mount and on any tap, and show a SOUND button until it works.
  useEffect(() => {
    unlockSound().catch(() => {});
    if (soundOn) return;
    const onTap = () => { unlockSound().catch(() => {}); };
    window.addEventListener("pointerdown", onTap);
    return () => window.removeEventListener("pointerdown", onTap);
  }, [soundOn, unlockSound]);

  // Spoken cues mid-rep are off by default; the coach debriefs after each attempt.
  // The setting lives here and is re-sent whenever the socket (re)opens.
  const [liveCues, setLiveCues] = useState(false);
  const liveCuesRef = useRef(false);
  const sendRef = useRef(null);
  const onWsOpen  = useCallback(() => {
    setWsReady(true);
    sendRef.current?.({ type: "live_cues", on: liveCuesRef.current });
  }, []);
  const { sendFrame, canSend, sendJson } = useWebSocket(sessionId, handleMessage, onWsOpen, playChunk, frameBuf.add);
  sendRef.current = sendJson;
  const toggleLiveCues = useCallback(() => {
    const next = !liveCuesRef.current;
    liveCuesRef.current = next;
    setLiveCues(next);
    sendJson({ type: "live_cues", on: next });
  }, [sendJson]);
  const requestDemo = useCallback(() => sendJson({ type: "demo_request" }), [sendJson]);

  // Push-to-talk: the server gets ptt_start, audio chunks while held, then ptt_end.
  const mic = useMicCapture(useCallback((b64) => { sendJson({ type: "ptt_audio", data: b64 }); }, [sendJson]));
  const holdingRef = useRef(false);
  const holdTimer = useRef(null);
  const endTalk = useCallback(() => {
    if (!holdingRef.current) return;
    holdingRef.current = false;
    setHolding(false);
    clearTimeout(holdTimer.current);
    mic.stop();   // sends the last chunk first
    sendJson({ type: "ptt_end" });
  }, [mic, sendJson]);
  const cancelTalk = useCallback(() => {
    if (!holdingRef.current) return;
    holdingRef.current = false;
    setHolding(false);
    clearTimeout(holdTimer.current);
    mic.stop();
    sendJson({ type: "ptt_cancel" });
  }, [mic, sendJson]);
  const startTalk = useCallback(() => {
    setMicNote(null);
    unlockSound().catch(() => {});
    if (!sendJson({ type: "ptt_start" })) return;
    if (sttOk === false) return;   // the coach answers "type instead"; no need for the mic
    holdingRef.current = true;
    setHolding(true);
    holdTimer.current = setTimeout(endTalk, MAX_HOLD_MS);
    mic.start().catch((err) => {
      setMicNote(err?.name === "NotAllowedError"
        ? "Microphone blocked. Allow it for this site in Safari settings."
        : `Microphone unavailable: ${err?.message || err}`);
      cancelTalk();
    });
  }, [mic, sendJson, sttOk, endTalk, cancelTalk, unlockSound]);
  useEffect(() => () => { clearTimeout(holdTimer.current); clearTimeout(heardTimer.current); mic.stop(); }, [mic]);
  const sendChat = useCallback((text) => sendJson({ type: "chat", text }), [sendJson]);
  const closeDemo   = useCallback(() => {
    demoOpen.current = false;
    setDemo(null);
    sendJson({ type: "demo_end" });   // the coach may speak live cues again
  }, [sendJson]);

  // Only send. The debug view now shows the live <video> element directly,
  // so we no longer re-render the whole page with a base64 image every frame.
  const handleFrame = useCallback((b64) => { sendFrame(b64); }, [sendFrame]);

  // fps is now a ceiling, not a fixed rate. Backpressure in canSend() means the
  // real rate automatically matches how fast the backend can keep up.
  const { videoRef, canvasRef, startCapture, stopCapture, error: cameraError } =
    useCamera(handleFrame, { fps: 30, maxWidth: 480, canSend });

  stopCaptureRef.current = stopCapture;

  useEffect(() => {
    const t = setTimeout(() => startCapture(), 800);
    return () => { clearTimeout(t); stopCapture(); };
  }, [startCapture, stopCapture]);

  const displayName = moveName
    ? moveName.replace(/_/g, " ").toUpperCase()
    : null;

  return (
    <div className="sess-root">
      {/* ── top bar ── */}
      <div className="sess-top">
        <div className="sess-title">
          BREAKDANCE COACH
        </div>

        <div className="sess-top-right">
          <div style={{
            fontSize: 11, letterSpacing: "0.2em",
            color:  aligned ? "#4cff4c" : displayName ? "#888" : "#444",
            border: `1px solid ${aligned ? "#4cff4c" : "#222"}`,
            padding: "4px 14px",
          }}>
            {aligned ? "ALIGNED" : displayName ? displayName : "DETECTING..."}
          </div>

          {!soundOn && (
            <button
              className="sess-btn"
              onClick={() => unlockSound().catch(() => {})}
              style={{
                background: "#1a0606", border: "1px solid #ff2d2d",
                color: "#ff2d2d", padding: "4px 14px",
                fontFamily: "'Anton',sans-serif",
                fontSize: 10, letterSpacing: "0.15em", cursor: "pointer",
              }}
            >
              TAP FOR SOUND
            </button>
          )}

          <button
            className="sess-btn"
            aria-label="Caption text size"
            onClick={cycleCaptionSize}
            style={{
              background: "transparent", border: "1px solid #2a2a2a", color: "#888",
              padding: "4px 14px", fontFamily: "'Anton',sans-serif",
              fontSize: 10, letterSpacing: "0.15em", cursor: "pointer",
            }}
          >
            TEXT {["S", "M", "L"][capSize]}
          </button>

          <button
            className="sess-btn"
            onClick={toggleLiveCues}
            disabled={!wsReady}
            aria-pressed={liveCues}
            aria-label="Spoken cues while you dance"
            style={{
              background: "transparent", border: `1px solid ${liveCues ? "#4cff4c" : "#2a2a2a"}`,
              color: liveCues ? "#4cff4c" : "#888", padding: "4px 14px",
              fontFamily: "'Anton',sans-serif",
              fontSize: 10, letterSpacing: "0.15em", cursor: "pointer",
            }}
          >
            LIVE CUES {liveCues ? "ON" : "OFF"}
          </button>

          <button
            className="sess-btn"
            onClick={requestDemo}
            disabled={!wsReady}
            aria-label="Show me the move slowly"
            style={{
              background: "transparent", border: "1px solid #ffc14c",
              color: "#ffc14c", padding: "4px 14px",
              fontFamily: "'Anton',sans-serif",
              fontSize: 10, letterSpacing: "0.15em", cursor: "pointer",
            }}
          >
            SHOW ME
          </button>

          <button
            className="sess-btn"
            onClick={() => setDebugMode((d) => !d)}
            style={{
              background:    debugMode ? "#1a1200" : "transparent",
              border:        `1px solid ${debugMode ? "#ffc14c" : "#2a2a2a"}`,
              color:         debugMode ? "#ffc14c" : "#444",
              padding:       "4px 14px",
              fontFamily:    "'Anton',sans-serif",
              fontSize:      10, letterSpacing: "0.15em", cursor: "pointer",
            }}
          >
            {debugMode ? "DEBUG ON" : "DEBUG"}
          </button>

          <button
            className="sess-btn"
            onClick={handleEnd}
            style={{
              background: "transparent", border: "1px solid #2a2a2a",
              color: "#555", padding: "6px 16px",
              fontFamily: "'Anton',sans-serif",
              fontSize: 11, letterSpacing: "0.15em", cursor: "pointer",
            }}
          >
            END SESSION
          </button>
        </div>
      </div>

      {audioNote && (
        <div style={{
          background: "rgba(255,193,76,0.08)",
          borderBottom: "1px solid rgba(255,193,76,0.2)",
          padding: "7px 24px",
          fontSize: 10, letterSpacing: "0.2em", color: "#ffc14c",
          textAlign: "center", flexShrink: 0,
        }}>
          {audioNote.toUpperCase()}
        </div>
      )}

      {/* ── guidance banner ── */}
      {framing.ready && guidance && !aligned && (
        <div style={{
          background: "rgba(255,45,45,0.07)",
          borderBottom: "1px solid rgba(255,45,45,0.15)",
          padding: "7px 24px",
          fontSize: 10, letterSpacing: "0.25em", color: "#ff5555",
          textAlign: "center", flexShrink: 0,
        }}>
          {guidance.toUpperCase()}
        </div>
      )}

      {/* ── main ── */}
      <div className="sess-main">

        {/* view */}
        <div className="sess-view">
          {/* Live camera feed. Always mounted so the stream stays attached; only visible
              in debug mode. Mirrored to match the backend's cv2.flip. */}
          <video ref={videoRef} autoPlay playsInline muted
            style={debugMode
              ? { position:"absolute", inset:0, width:"100%", height:"100%",
                  objectFit:"contain", transform:"scaleX(-1)" }
              : { display:"none" }} />
          {!wsReady ? (
            <div style={{
              width:"100%", height:"100%",
              display:"flex", alignItems:"center", justifyContent:"center",
              flexDirection:"column", gap:16,
            }}>
              <div style={{ fontSize:11, letterSpacing:"0.3em", color:"#ff2d2d" }}>
                BREAKDANCE COACH
              </div>
              <div style={{ fontSize:10, letterSpacing:"0.25em", color:"#2a2a2a" }}>
                INITIALIZING PIPELINE...
              </div>
            </div>
          ) : debugMode ? (
            <div className="sess-debug" style={{ position:"absolute", top:12, left:12, zIndex:2,
              fontSize:10, letterSpacing:"0.2em", color:"#ff2d2d" }}>
              <div>ROUND TRIP {latencyMs ?? "--"} MS</div>
              <div>
                IMU {imu?.connected ? `${imu.rate_hz}HZ` : "OFF"}
                {" · "}WRIST {imu?.wrist_ok ? "OK" : "--"}
                {" · "}LEG {imu?.leg_ok ? "OK" : "--"}
                {" · "}OFFSET {imu?.offset_ms ?? "--"} MS
              </div>
              <div>
                POSTURE {(freezeDebug?.posture ?? "--").toUpperCase()}
                {" · "}{freezeDebug?.holding ? "HOLDING" : "MOVING"}
                {" · "}FREEZES {freezeDebug?.count ?? 0}
                {" · "}VETOED {freezeDebug?.vetoed ?? 0}
              </div>
            </div>
          ) : (
            <SkeletonCanvas
              landmarks={landmarks}
              visibility={visibility}
              ghostBones={ghostBones}
              ghostAnchor={ghostAnchor}
              aligned={aligned}
              fillRatio={fillRatio}
              moveName={moveName}
            />
          )}

          {scoreResult && (
            <KeyframeFlash result={scoreResult} onDone={() => setScoreResult(null)} />
          )}

          {stats && (
            <div className="sess-chip" aria-label="Score">
              <span>{stats.clean_reps ?? 0}/{stats.reps ?? 0} clean</span>
              <span>{stats.clean_streak ?? 0}×</span>
            </div>
          )}

          <FreezeMeter live={freezeLive} result={freezeDone} />
          {wsReady && !framing.ready && <FramingGuide problem={cameraError || framing.problem} />}
          <ResultCard
            summary={summary}
            onClose={() => setSummary(null)}
            onReplay={replayable ? openReplay : undefined}
          />
          {replay && <ReplayOverlay frames={replay.frames} fix={replay.fix} onClose={() => setReplay(null)} />}
          <DemoOverlay key={demo?.requestId} demo={demo} onClose={closeDemo} />
          <CoachCue cue={coachCue} size={CAPTION_SIZES[capSize]} />
          {wsReady && (debugMode || sttOk === false) && (
            <ChatBox onSend={sendChat} disabled={voiceState !== "idle"} />
          )}
          {wsReady && (
            <TalkButton
              state={voiceState}
              holding={holding}
              disabled={!wsReady}
              onStart={startTalk}
              onEnd={endTalk}
              onCancel={cancelTalk}
              heard={heard}
              note={micNote}
            />
          )}
        </div>

        {/* ── sidebar ── */}
        <div className="sess-side">
          {stats
            ? <ScoreHUD stats={stats} />
            : <div style={{ fontSize:9, letterSpacing:"0.2em", color:"#2a2a2a" }}>
                WAITING FOR DATA...
              </div>
          }

          <div style={{ marginTop: 20 }}>
            <div style={{ fontSize:9, letterSpacing:"0.2em", color:"#2a2a2a", marginBottom:6 }}>
              BUFFER
            </div>
            <div style={{ fontSize:20, letterSpacing:"0.05em",
              color: fillRatio >= 1 ? "#4cff4c" : "#ff2d2d" }}>
              {fillRatio >= 1 ? "READY" : `${Math.round(fillRatio * 100)}%`}
            </div>
          </div>

          <div style={{ marginTop: 20 }}>
            <div style={{ fontSize:9, letterSpacing:"0.2em", color:"#2a2a2a", marginBottom:6 }}>
              CURRENT MOVE
            </div>
            <div style={{ fontSize:13, letterSpacing:"0.08em", color:"#888" }}>
              {displayName || "—"}
            </div>
          </div>

          <div style={{ marginTop: 20 }}>
            <div style={{ fontSize:9, letterSpacing:"0.2em", color:"#2a2a2a", marginBottom:6 }}>
              VIEW
            </div>
            <div style={{ fontSize:10, letterSpacing:"0.1em",
              color: debugMode ? "#ffc14c" : "#4cff4c" }}>
              {debugMode ? "CAMERA / DEBUG" : "SKELETON"}
            </div>
          </div>
        </div>
      </div>

      <canvas ref={canvasRef} style={{ display:"none" }} />
    </div>
  );
}