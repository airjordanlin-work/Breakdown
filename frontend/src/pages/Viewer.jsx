import { useCallback, useEffect, useRef, useState } from "react";
import SkeletonCanvas from "../components/SkeletonCanvas";
import DemoOverlay from "../components/DemoOverlay";
import ResultCard from "../components/ResultCard";
import { useViewerSocket } from "../hooks/useViewerSocket";
import { useCoachAudio } from "../hooks/useCoachAudio";
import "../styles/viewer.css";

// A big mirror for a TV or laptop (open /view). The phone stays the camera;
// this page only shows what the backend sends: the dancer's skeleton flipped
// like a mirror, the rep tallies, the joint to fix, and the coach's words in
// large text. The camera picture is off until you tap SHOW CAMERA; it is a
// small, low-quality copy of what the phone sends, shown live and never stored.
export default function Viewer() {
  const [frame, setFrame] = useState({});
  const [stats, setStats] = useState(null);
  const [cue, setCue] = useState("");
  const [cueKey, setCueKey] = useState(0);
  const [demo, setDemo] = useState(null);
  const [summary, setSummary] = useState(null);
  const repsNow = useRef(0);
  const [audioNote, setAudioNote] = useState(null);
  const [showCamera, setShowCamera] = useState(false);
  const [picture, setPicture] = useState(null);
  const cueTimer = useRef(null);

  const { unlocked, unlock, playChunk, setLatestTurn } = useCoachAudio();

  const onMessage = useCallback((msg) => {
    if (msg.type === "voice_status") {
      setAudioNote(msg.available ? null : `Coach voice is off: ${msg.reason}`);
    } else if (msg.type === "coach_audio_skipped") {
      setAudioNote(`No audio for the last cue: ${msg.reason}`);
    } else if (msg.type === "coach_cue") {
      if (typeof msg.cue !== "string") return;
      setCue(msg.cue);
      setCueKey((k) => k + 1);
      setLatestTurn(msg.turn_id);
      clearTimeout(cueTimer.current);
      cueTimer.current = setTimeout(() => setCue(""), 8000);
    } else if (msg.type === "video") {
      setPicture(`data:image/jpeg;base64,${msg.data}`);
    } else if (msg.type === "rep_summary") {
      setSummary({ ...msg, at: repsNow.current });
    } else if (msg.type === "demo_end") {
      setDemo(null);
    } else if (msg.type === "demo") {
      setDemo({ ...msg, _n: Date.now() });
    } else if (msg.type === "frame") {
      setFrame({
        landmarks: msg.landmarks, visibility: msg.visibility,
        ghostBones: msg.ghost_bones, ghostAnchor: msg.ghost_anchor,
        aligned: msg.aligned, fillRatio: msg.fill_ratio, moveName: msg.move_name,
      });
      if (msg.stats) { setStats(msg.stats); repsNow.current = msg.stats.reps; }
    }
  }, [setLatestTurn]);

  // A different session means a different dancer run: drop the old pose, tallies and caption.
  const onSession = useCallback(() => {
    setFrame({}); setStats(null); setCue(""); setDemo(null); setSummary(null); setAudioNote(null); setPicture(null);
  }, []);

  const status = useViewerSocket(onMessage, playChunk, onSession, showCamera);
  const closeDemo = useCallback(() => setDemo(null), []);

  // Sound needs a click or tap first on most browsers and TVs, so unlock on the
  // first one (the button does the same).
  useEffect(() => {
    unlock().catch(() => {});
    if (unlocked) return;
    const onTap = () => { unlock().catch(() => {}); };
    window.addEventListener("pointerdown", onTap);
    return () => window.removeEventListener("pointerdown", onTap);
  }, [unlocked, unlock]);

  useEffect(() => {
    if (summary && (stats?.reps ?? 0) > summary.at) setSummary(null);
  }, [stats, summary]);

  const reps = stats?.reps ?? 0;
  const focus = stats?.focus_joint
    ? `${(stats.focus_fix || "").toUpperCase()} ${stats.focus_joint.toUpperCase()}`
    : reps ? "ALL JOINTS ON POINT" : "DANCE A FULL REP";

  return (
    <div className="viewer-root">
      <div className="viewer-top">
        <span className="viewer-title">BREAKDOWN</span>
        <span className={`viewer-status ${status}`}>
          {status === "live" ? "LIVE" : "WAITING FOR THE PHONE"}
        </span>
        <button className="viewer-sound" onClick={() => { setShowCamera((v) => !v); setPicture(null); }}>
          {showCamera ? "HIDE CAMERA" : "SHOW CAMERA"}
        </button>
        {!unlocked && <button className="viewer-sound" onClick={() => unlock().catch(() => {})}>CLICK FOR SOUND</button>}
      </div>

      <div className="viewer-main">
        <div className="viewer-stage">
          {showCamera && picture && <img className="viewer-camera" src={picture} alt="" />}
          <SkeletonCanvas {...frame} mirror clear={showCamera && !!picture} />
          <ResultCard summary={summary} big />
          <DemoOverlay key={demo?._n} demo={demo} onClose={closeDemo} />
        </div>
        <div className="viewer-side">
          <div className="viewer-label">CLEAN REPS</div>
          <div className="viewer-big">{stats?.clean_reps ?? 0}<span> of {reps}</span></div>
          <div className="viewer-label">STREAK</div>
          <div className="viewer-mid">{stats?.clean_streak ?? 0}<span> best {stats?.best_streak ?? 0}</span></div>
          <div className="viewer-label">FOCUS</div>
          <div className={`viewer-focus ${stats?.focus_joint ? "fix" : "ok"}`}>{focus}</div>
        </div>
      </div>

      <div className="viewer-cue" key={cueKey} aria-live="polite">{cue.toUpperCase()}</div>
      {audioNote && <div className="viewer-note">{audioNote}</div>}
    </div>
  );
}
