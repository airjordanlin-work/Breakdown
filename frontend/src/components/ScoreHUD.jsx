// Rep view: plain tallies of scored reps from the server, no points or grade.
// A clean rep is one where no checked joint was in the Miss tier.
const label = { fontSize:9, letterSpacing:"0.2em", color:"#555", marginBottom:8 };

export default function ScoreHUD({ stats }) {
  const reps   = stats.reps ?? 0;
  const clean  = stats.clean_reps ?? 0;
  const streak = stats.clean_streak ?? 0;
  const best   = stats.best_streak ?? 0;
  const hot    = streak >= 3;

  return (
    <div>
      <div style={label}>CLEAN REPS</div>
      <div style={{ display:"flex", alignItems:"baseline", gap:8, marginBottom:20 }}>
        <span style={{ fontSize:48, lineHeight:1 }}>{clean}</span>
        <span style={{ fontSize:14, color:"#555" }}>of {reps}</span>
      </div>

      <div style={label}>STREAK</div>
      <div style={{ display:"flex", alignItems:"baseline", gap:8, marginBottom:20 }}>
        <span style={{ fontSize:28, color: hot ? "#ffd700" : "#fff" }}>
          {streak}{hot ? " 🔥" : ""}
        </span>
        <span style={{ fontSize:11, color:"#555" }}>best {best}</span>
      </div>

      <div style={label}>FOCUS</div>
      <div style={{ fontSize:12, letterSpacing:"0.1em", border:"1px solid #222",
        padding:"8px 10px", color: stats.focus_joint ? "#ffc14c" : "#4cff4c" }}>
        {reps === 0
          ? <span style={{ color:"#555" }}>Dance a full rep for your first read</span>
          : stats.focus_joint
            ? `${stats.focus_fix?.toUpperCase()} ${stats.focus_joint.toUpperCase()}`
            : "ALL JOINTS ON POINT"}
      </div>
    </div>
  );
}
