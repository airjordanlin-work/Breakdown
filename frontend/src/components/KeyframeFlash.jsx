import { useEffect } from "react";
const TC = { Perfect:"#4cff4c", Close:"#ffc14c", Miss:"#ff4c4c" };
const TIER_WORDS = { Perfect:"ON POINT", Close:"CLOSE", Miss:"WORK ON IT" };
export default function KeyframeFlash({ result, onDone }) {
  useEffect(() => { const t = setTimeout(onDone, 2500); return () => clearTimeout(t); }, [result, onDone]);
  if (!result) return null;
  return (
    <div style={{ position:"absolute", top:20, right:20, background:"rgba(6,6,8,0.92)",
      border:"1px solid #222", padding:"16px 20px", minWidth:200 }}>
      <div style={{ fontSize:9, letterSpacing:"0.2em", color:"#555", marginBottom:8 }}>REP</div>
      <div style={{ fontSize:28, marginBottom:12, color: result.clean ? "#4cff4c" : "#ffc14c" }}>
        {result.clean ? "CLEAN REP" : "KEEP WORKING"}
      </div>
      {result.results.map((r, i) => (
        <div key={i} style={{ marginBottom:8 }}>
          <div style={{ display:"flex", justifyContent:"space-between",
            fontSize:11, letterSpacing:"0.1em", marginBottom:4 }}>
            <span style={{ color:"#888" }}>{r.joint.toUpperCase()}</span>
            <span style={{ color:TC[r.tier] }}>{TIER_WORDS[r.tier] ?? r.tier.toUpperCase()}</span>
          </div>
          <div style={{ height:2, background:"#111" }}>
            <div style={{ height:"100%", width:`${Math.max(0,(1-r.diff/90)*100)}%`,
              background:TC[r.tier], transition:"width 0.4s" }} />
          </div>
          <div style={{ fontSize:9, color:"#444", marginTop:2 }}>{r.diff} DEG OFF</div>
        </div>
      ))}
    </div>
  );
}
