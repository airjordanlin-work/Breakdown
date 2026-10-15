import { useEffect } from "react";
import MovePreview from "./MovePreview";
import "../styles/move-library.css";

// Slow demo of the practiced move, shown over the session view when the
// dancer taps SHOW ME. The camera keeps running underneath; the coach stays
// quiet until the demo has played `loops` times, then the overlay closes.
export default function DemoOverlay({ demo, onClose }) {
  useEffect(() => {
    if (!demo) return;
    const ms = (demo.seconds || 3) / demo.speed * demo.loops * 1000;
    const t = setTimeout(onClose, ms);
    return () => clearTimeout(t);
  }, [demo, onClose]);

  if (!demo) return null;
  return (
    <div role="dialog" aria-label={`Demo of ${demo.move_name}`} style={{
      position: "absolute", inset: 0, zIndex: 5,
      background: "rgba(6,6,8,0.92)",
      display: "flex", flexDirection: "column", padding: 12, gap: 8,
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center",
        fontSize: 11, letterSpacing: "0.2em", color: "#ffc14c" }}>
        <span>
          DEMO · {demo.move_name}{demo.focus ? ` · WATCH THE ${demo.focus.toUpperCase()}` : ""}
        </span>
        <button type="button" onClick={onClose} style={{
          background: "transparent", border: "1px solid #2a2a2a", color: "#aaa",
          padding: "4px 12px", fontFamily: "'Anton',sans-serif",
          fontSize: 10, letterSpacing: "0.15em", cursor: "pointer",
        }}>
          CLOSE
        </button>
      </div>
      <div className="library demo-stage" style={{ flex: 1, minHeight: 0 }}>
        <MovePreview moveId={demo.move_id} title={demo.move_name}
          highlight={demo.highlight} initialSpeed={demo.speed} />
      </div>
    </div>
  );
}
