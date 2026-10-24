// Hold to talk to the coach (push-to-talk). Press and hold, speak, let go.
// Sliding off the button or the browser cancelling the touch drops the clip.
const LABELS = {
  idle: "HOLD TO TALK",
  listening: "LISTENING",
  transcribing: "HEARING YOU",
  thinking: "THINKING",
  speaking: "COACH",
};

export default function TalkButton({ state = "idle", holding, disabled, onStart, onEnd, onCancel, heard, note }) {
  const busy = state !== "idle" && !holding;
  const down = (e) => {
    if (disabled || busy) return;
    e.preventDefault();
    try { e.currentTarget.setPointerCapture(e.pointerId); } catch { /* not supported */ }
    onStart();
  };
  const up = (e) => { e.preventDefault(); if (holding) onEnd(); };
  const cancel = () => { if (holding) onCancel(); };
  return (
    <div className="talk-wrap">
      {heard && <div className="talk-heard">YOU: {heard}</div>}
      {note && <div className="talk-note">{note}</div>}
      <button
        type="button"
        className={`talk-btn ${holding ? "on" : ""} ${busy ? "busy" : ""}`}
        aria-label="Hold to talk to the coach"
        aria-pressed={!!holding}
        disabled={disabled}
        onPointerDown={down}
        onPointerUp={up}
        onPointerCancel={cancel}
        onContextMenu={(e) => e.preventDefault()}
      >
        {holding ? LABELS.listening : LABELS[state] || LABELS.idle}
      </button>
    </div>
  );
}
