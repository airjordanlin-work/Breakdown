import { useState, useEffect, useCallback, useRef } from "react";
import { useWebSocket } from "../hooks/useWebSocket";
import { useCamera }    from "../hooks/useCamera";
import ScoreHUD         from "../components/ScoreHUD";
import KeyframeFlash    from "../components/KeyframeFlash";
import SkeletonCanvas   from "../components/SkeletonCanvas";
import CoachCue from "../components/CoachCue";

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
  const [cameraFrame, setCameraFrame] = useState(null);
  const [wsReady,     setWsReady]     = useState(false);
  const [coachCue, setCoachCue] = useState("");

  const handleEnd = async () => {
    await fetch(`http://localhost:8000/session/${sessionId}`, { method: "DELETE" });
    onEnd();
  };

  const handleMessage = useCallback((msg) => {
    if (msg.type !== "frame") return;
    if (msg.landmarks)                  setLandmarks(msg.landmarks);
    if (msg.visibility)                 setVisibility(msg.visibility);
    if (msg.ghost_bones)                setGhostBones(msg.ghost_bones);
    if (msg.ghost_anchor !== undefined) setGhostAnchor(msg.ghost_anchor);
    if (msg.aligned     !== undefined)  setAligned(msg.aligned);
    if (msg.fill_ratio  !== undefined)  setFillRatio(msg.fill_ratio);
    if (msg.move_name)                  setMoveName(msg.move_name);
    if (msg.guidance    !== undefined)  setGuidance(msg.guidance);
    if (msg.stats)                      setStats(msg.stats);
    if (msg.score_result)               setScoreResult(msg.score_result);
    if (msg.type === "coach_cue") {
      setCoachCue(msg.cue);
      return;
    }
  }, []);

  const onWsOpen  = useCallback(() => setWsReady(true), []);
  const { sendFrame } = useWebSocket(sessionId, handleMessage, onWsOpen);

  const handleFrame = useCallback((b64) => {
    sendFrame(b64);
    setCameraFrame(`data:image/jpeg;base64,${b64}`);
  }, [sendFrame]);

  const { videoRef, canvasRef, startCapture, stopCapture } = useCamera(handleFrame, 15);

  useEffect(() => {
    const t = setTimeout(() => startCapture(), 800);
    return () => { clearTimeout(t); stopCapture(); };
  }, [startCapture, stopCapture]);

  const displayName = moveName
    ? moveName.replace(/_/g, " ").toUpperCase()
    : null;

  return (
    <div style={{
      height: "100vh", background: "#060608",
      fontFamily: "'Anton','Arial Black',sans-serif",
      color: "#fff", display: "flex", flexDirection: "column",
      overflow: "hidden",
    }}>
      {/* ── top bar ── */}
      <div style={{
        borderBottom: "2px solid #1a1a1a",
        padding: "12px 24px",
        display: "flex", justifyContent: "space-between", alignItems: "center",
        flexShrink: 0, background: "#060608",
      }}>
        <div style={{ fontSize: 14, letterSpacing: "0.15em", color: "#ff2d2d" }}>
          BREAKDANCE COACH
        </div>

        <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
          <div style={{
            fontSize: 11, letterSpacing: "0.2em",
            color:  aligned ? "#4cff4c" : displayName ? "#888" : "#444",
            border: `1px solid ${aligned ? "#4cff4c" : "#222"}`,
            padding: "4px 14px",
          }}>
            {aligned ? "ALIGNED" : displayName ? displayName : "DETECTING..."}
          </div>

          <button
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

      {/* ── guidance banner ── */}
      {guidance && !aligned && (
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
      <div style={{ flex: 1, display: "flex", overflow: "hidden", minHeight: 0 }}>

        {/* view */}
        <div style={{ flex: 1, position: "relative", overflow: "hidden", background: "#060608" }}>
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
            cameraFrame
              ? <img src={cameraFrame} alt="debug"
                  style={{ width:"100%", height:"100%", objectFit:"contain", display:"block" }} />
              : <div style={{ width:"100%", height:"100%", display:"flex",
                  alignItems:"center", justifyContent:"center",
                  color:"#2a2a2a", fontSize:10, letterSpacing:"0.2em" }}>
                  WAITING FOR CAMERA...
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
          <CoachCue cue={coachCue} />
        </div>

        {/* ── sidebar ── */}
        <div style={{
          width: 210, background: "#07070c",
          borderLeft: "1px solid #111", padding: "20px 16px",
          display: "flex", flexDirection: "column", overflowY: "auto",
          flexShrink: 0,
        }}>
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

      <video ref={videoRef} autoPlay playsInline muted style={{ display:"none" }} />
      <canvas ref={canvasRef} style={{ display:"none" }} />
    </div>
  );
}