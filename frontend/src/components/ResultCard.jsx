// What the coach found in the attempt you just finished: how many reps were
// clean, what landed every rep, and the one thing to fix with where in the
// move it went wrong. Built by the backend from the scorer's results.
export default function ResultCard({ summary, onClose, onReplay, big = false }) {
  if (!summary) return null;
  const { move, reps, clean_reps: clean, went_well: wentWell = [], fix } = summary;
  const size = (n) => (big ? n * 1.8 : n);
  const label = { fontSize: size(11), letterSpacing: "0.2em", color: "#888" };
  const text = { fontFamily: "'Anton',sans-serif", letterSpacing: "0.04em", color: "#eee9e0" };

  return (
    <div role="dialog" aria-label="Result of your last attempt" style={{
      position: "absolute", inset: 0, zIndex: 4, background: "rgba(6,6,8,0.9)",
      display: "flex", flexDirection: "column", justifyContent: "center",
      gap: size(10), padding: size(18),
    }}>
      <div style={label}>{String(move || "").toUpperCase()} · RESULT</div>
      <div style={{ ...text, fontSize: size(44), lineHeight: 1 }}>
        {clean} OF {reps} CLEAN
      </div>

      {wentWell.length > 0 && (
        <div>
          <div style={label}>WENT WELL</div>
          <div style={{ ...text, fontSize: size(20), color: "#4cff4c" }}>
            {wentWell.join(", ").toUpperCase()}
          </div>
        </div>
      )}

      <div>
        <div style={label}>FIX THIS</div>
        {fix ? (
          <>
            <div style={{ ...text, fontSize: size(26), color: "#ffc14c" }}>
              {`${fix.fix} ${fix.joint}`.toUpperCase()}
            </div>
            {fix.checkpoint && (
              <div style={{ ...text, fontSize: size(14), color: "#aaa" }}>
                OFF AT CHECKPOINT {fix.checkpoint} OF {fix.of}
              </div>
            )}
          </>
        ) : (
          <div style={{ ...text, fontSize: size(26), color: "#4cff4c" }}>ALL JOINTS ON POINT</div>
        )}
      </div>

      {onReplay && (
        <button type="button" onClick={onReplay} style={{
          alignSelf: "flex-start", background: "transparent",
          border: "1px solid #ffc14c", color: "#ffc14c", padding: "8px 22px",
          fontFamily: "'Anton',sans-serif", fontSize: 12, letterSpacing: "0.15em", cursor: "pointer",
        }}>
          REPLAY THE MISTAKE
        </button>
      )}

      {onClose && (
        <button type="button" onClick={onClose} style={{
          alignSelf: "flex-start", marginTop: size(6), background: "transparent",
          border: "1px solid #4cff4c", color: "#4cff4c", padding: "8px 22px",
          fontFamily: "'Anton',sans-serif", fontSize: 12, letterSpacing: "0.15em", cursor: "pointer",
        }}>
          GO AGAIN
        </button>
      )}
    </div>
  );
}
