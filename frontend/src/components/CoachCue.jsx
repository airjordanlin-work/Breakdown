import { useEffect, useState } from "react";

export default function CoachCue({ cue }) {
  const [visible, setVisible] = useState(false);
  const [displayed, setDisplayed] = useState("");

  useEffect(() => {
    if (!cue) return;
    setDisplayed(cue);
    setVisible(true);
    const t = setTimeout(() => setVisible(false), 4000);
    return () => clearTimeout(t);
  }, [cue]);

  if (!displayed) return null;

  return (
    <div style={{
      position:   "absolute",
      bottom:     40,
      left:       "50%",
      transform:  "translateX(-50%)",
      background: "rgba(6,6,8,0.90)",
      border:     "1px solid rgba(255,45,45,0.4)",
      padding:    "12px 28px",
      fontFamily: "'Anton','Arial Black',sans-serif",
      fontSize:   18,
      letterSpacing: "0.06em",
      color:      "#fff",
      whiteSpace: "nowrap",
      opacity:    visible ? 1 : 0,
      transition: "opacity 0.5s ease",
      pointerEvents: "none",
      textShadow: "0 2px 12px rgba(255,45,45,0.4)",
      boxShadow:  "0 0 30px rgba(255,45,45,0.15)",
    }}>
      {displayed.toUpperCase()}
    </div>
  );
}