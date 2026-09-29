import { useEffect, useState } from "react";

// Freeze stability from the wearable IMUs.
// While holding: live hold time + stability bar.
// After: a result card for a few seconds.

const TIER_COLOR = {
  "rock solid": "#4cff4c",
  steady:       "#b6ff4c",
  shaky:        "#ffc14c",
  unstable:     "#ff4c4c",
};
const LIMB = { wrist: "ARM", leg: "LEGS" };
const RESULT_MS = 3500;

function Bar({ value, color }) {
  return (
    <div style={{ height: 3, background: "#111", marginTop: 8 }}>
      <div style={{ height: "100%", width: `${value}%`, background: color,
        transition: "width 0.2s" }} />
    </div>
  );
}

export default function FreezeMeter({ live, result }) {
  const [shown, setShown] = useState(null);

  useEffect(() => {
    if (!result) return;
    setShown(result);
    const t = setTimeout(() => setShown(null), RESULT_MS);
    return () => clearTimeout(t);
  }, [result]);

  const data = live || shown;
  if (!data) return null;

  const color = TIER_COLOR[data.tier] || "#888";
  const wobbly = data.tier === "shaky" || data.tier === "unstable";

  return (
    <div style={{
      position: "absolute", top: 20, left: "50%", transform: "translateX(-50%)",
      background: "rgba(6,6,8,0.92)", border: `1px solid ${color}55`,
      padding: "12px 20px", minWidth: 220, textAlign: "center",
      pointerEvents: "none",
    }}>
      <div style={{ fontSize: 9, letterSpacing: "0.25em", color: "#555" }}>
        {live ? "HOLDING FREEZE" : "FREEZE"}
      </div>
      <div style={{ fontSize: 26, letterSpacing: "0.05em", color }}>
        {live ? `${data.duration_s.toFixed(1)}S` : data.tier.toUpperCase()}
      </div>
      <div style={{ fontSize: 10, letterSpacing: "0.15em", color: "#888" }}>
        {live
          ? `STABILITY ${data.stability}`
          : `${data.duration_s.toFixed(1)}S · STABILITY ${data.stability}`}
      </div>
      {!live && wobbly && data.shakiest && (
        <div style={{ fontSize: 10, letterSpacing: "0.15em", color, marginTop: 4 }}>
          SHAKIEST: {LIMB[data.shakiest] || data.shakiest.toUpperCase()}
        </div>
      )}
      <Bar value={data.stability} color={color} />
    </div>
  );
}
