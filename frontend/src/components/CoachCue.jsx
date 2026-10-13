import { useEffect, useState } from "react";

// Caption sizes in px. The coach's spoken words always show as text (NFR-17);
// the dancer picks a size with the A button in the session bar.
export const CAPTION_SIZES = [16, 22, 30];

export default function CoachCue({ cue, size = CAPTION_SIZES[0] }) {
  const [visible, setVisible] = useState(false);
  const [displayed, setDisplayed] = useState("");

  useEffect(() => {
    if (!cue) return;
    setDisplayed(cue);
    setVisible(true);
    const t = setTimeout(() => setVisible(false), 5000);
    return () => clearTimeout(t);
  }, [cue]);

  if (!displayed) return null;

  return (
    <div className="coach-cue" aria-live="polite" style={{
      fontSize:   size,
      opacity:    visible ? 1 : 0,
    }}>
      {displayed.toUpperCase()}
    </div>
  );
}
