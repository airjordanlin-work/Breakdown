import { useState, useEffect, useRef } from "react";
import { API } from "../api";

const DIFF_COLOR = {
  Beginner:     { bg: "#0a1f0a", border: "#1a5c1a", text: "#4cff4c" },
  Intermediate: { bg: "#1a0a1f", border: "#4a1a6c", text: "#cc44ff" },
  Advanced:     { bg: "#1f0000", border: "#5c0000", text: "#ff4c4c" },
};

// ── paint splatter helper ─────────────────────────────────────────────────────
function splatter(ctx, cx, cy, r, color, count = 12) {
  ctx.save();
  for (let i = 0; i < count; i++) {
    const angle  = (Math.PI * 2 * i) / count + Math.random() * 0.4;
    const dist   = r * (0.3 + Math.random() * 0.8);
    const size   = r * (0.04 + Math.random() * 0.18);
    const x = cx + Math.cos(angle) * dist;
    const y = cy + Math.sin(angle) * dist;
    ctx.beginPath();
    ctx.arc(x, y, size, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    // drip from some blobs
    if (Math.random() > 0.6) {
      const dripLen = size * (2 + Math.random() * 5);
      ctx.beginPath();
      ctx.moveTo(x, y + size);
      ctx.bezierCurveTo(
        x + size * 0.3, y + dripLen * 0.4,
        x - size * 0.3, y + dripLen * 0.7,
        x + size * 0.1, y + dripLen
      );
      ctx.lineWidth = size * 1.4;
      ctx.strokeStyle = color;
      ctx.stroke();
    }
  }
  // center blob
  ctx.beginPath();
  ctx.arc(cx, cy, r * 0.35, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.restore();
}

// ── brick wall ───────────────────────────────────────────────────────────────
function drawBrickWall(ctx, W, H) {
  const bH = Math.floor(H / 22);
  const bW = Math.floor(W / 7);

  for (let row = 0; row < 24; row++) {
    const y      = row * bH;
    const offset = (row % 2) * (bW / 2);
    for (let col = -1; col < 9; col++) {
      const x = col * bW + offset;
      // brick face — subtle warm dark
      const shade = 10 + Math.floor(Math.random() * 6);
      ctx.fillStyle = `rgb(${shade + 4},${shade},${shade - 2})`;
      ctx.fillRect(x + 1, y + 1, bW - 2, bH - 2);
      // mortar lines
      ctx.strokeStyle = "rgba(0,0,0,0.5)";
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, bW, bH);
    }
  }

  // darken overall
  ctx.fillStyle = "rgba(0,0,0,0.55)";
  ctx.fillRect(0, 0, W, H);
}

// ── 3D graffiti text ─────────────────────────────────────────────────────────
function draw3DText(ctx, text, cx, cy, size, angle = 0) {
  ctx.save();
  ctx.translate(cx, cy);
  ctx.rotate(angle);

  // shadow depth layers
  for (let d = 8; d >= 1; d--) {
    ctx.fillStyle = `rgba(0,0,0,${0.15 + d * 0.04})`;
    ctx.font = `900 ${size}px 'Arial Black', Arial`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(text, d * 1.5, d * 1.5);
  }

  // dark outline
  ctx.strokeStyle = "rgba(0,0,0,0.9)";
  ctx.lineWidth = size * 0.08;
  ctx.lineJoin = "round";
  ctx.font = `900 ${size}px 'Arial Black', Arial`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.strokeText(text, 0, 0);

  // main gradient fill
  const grad = ctx.createLinearGradient(0, -size * 0.6, 0, size * 0.6);
  grad.addColorStop(0.0,  "#ff6060");
  grad.addColorStop(0.25, "#ff2d2d");
  grad.addColorStop(0.5,  "#cc0000");
  grad.addColorStop(0.75, "#ff2d2d");
  grad.addColorStop(1.0,  "#ff8080");
  ctx.fillStyle = grad;
  ctx.fillText(text, 0, 0);

  // highlight streak
  const hl = ctx.createLinearGradient(-size, -size * 0.5, size * 0.3, size * 0.1);
  hl.addColorStop(0,   "rgba(255,255,255,0.35)");
  hl.addColorStop(0.4, "rgba(255,255,255,0.10)");
  hl.addColorStop(1,   "rgba(255,255,255,0)");
  ctx.fillStyle = hl;
  ctx.fillText(text, 0, 0);

  ctx.restore();
}

// ── smaller tag text ─────────────────────────────────────────────────────────
function drawTag(ctx, text, cx, cy, size, color, angle = 0) {
  ctx.save();
  ctx.translate(cx, cy);
  ctx.rotate(angle);
  ctx.font = `900 ${size}px 'Arial Black', Arial`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.strokeStyle = "rgba(0,0,0,0.8)";
  ctx.lineWidth = size * 0.10;
  ctx.lineJoin = "round";
  ctx.strokeText(text, 0, 0);
  ctx.fillStyle = color;
  ctx.fillText(text, 0, 0);
  ctx.restore();
}

// ── spray lines ──────────────────────────────────────────────────────────────
function sprayLine(ctx, x1, y1, x2, y2, color, width = 6) {
  ctx.save();
  const grad = ctx.createLinearGradient(x1, y1, x2, y2);
  grad.addColorStop(0,   "rgba(0,0,0,0)");
  grad.addColorStop(0.1, color);
  grad.addColorStop(0.9, color);
  grad.addColorStop(1,   "rgba(0,0,0,0)");
  ctx.strokeStyle = grad;
  ctx.lineWidth   = width;
  ctx.lineCap     = "round";
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
  ctx.restore();
}

// ── GraffitiBackground component ─────────────────────────────────────────────
function GraffitiBackground() {
  return (
    <div style={{
      position: "fixed",
      inset: 0,
      zIndex: 0,
      pointerEvents: "none",
    }}>
      {/* graffiti image — more visible */}
      <div style={{
        position:           "absolute",
        inset:              0,
        backgroundImage:    "url('/graffiti-bg.jpg')",
        backgroundSize:     "cover",
        backgroundPosition: "center",
        opacity:            0.65,   // ← was 0.35, now much more visible
      }} />

      {/* neutral dark overlay — no color tint */}
      <div style={{
        position:   "absolute",
        inset:      0,
        background: "rgba(0,0,0,0.45)",  // ← plain black, no red
      }} />

      {/* vignette — softer */}
      <div style={{
        position:   "absolute",
        inset:      0,
        background: "radial-gradient(ellipse at center, transparent 20%, rgba(0,0,0,0.60) 100%)",
      }} />
    </div>
  );
}
// ── Landing page ──────────────────────────────────────────────────────────────
export default function Landing({ onStart }) {
  const [moves,    setMoves]    = useState([]);
  const [selected, setSelected] = useState(null);
  const [loading,  setLoading]  = useState(false);
  const [fetching, setFetching] = useState(true);
  const [error,    setError]    = useState(null);

  useEffect(() => {
    fetch(`${API}/moves`)
      .then((r) => r.json())
      .then((d) => {
        setMoves(d.moves);
        if (d.moves.length) setSelected(d.moves[0].id);
      })
      .catch(() => setError("Cannot connect — is the backend running?"))
      .finally(() => setFetching(false));
  }, []);

  const handleStart = async () => {
    setLoading(true);
    await onStart({ move_id: selected });
    setLoading(false);
  };

  if (fetching || error) return (
    <div style={{ minHeight:"100vh", background:"#060608",
      fontFamily:"'Anton','Arial Black',sans-serif",
      display:"flex", alignItems:"center", justifyContent:"center",
      flexDirection:"column", gap:16 }}>
      <GraffitiBackground />
      <div style={{ position:"relative", zIndex:1, fontSize:11,
        letterSpacing:"0.3em", color:"#ff2d2d" }}>BREAKDANCE COACH</div>
      <div style={{ position:"relative", zIndex:1, fontSize:11,
        letterSpacing:"0.2em", color: error ? "#ff4c4c" : "#2a2a2a" }}>
        {error || "LOADING..."}
      </div>
    </div>
  );

  return (
    <div style={{
      minHeight: "100vh",
      fontFamily: "'Anton','Arial Black',sans-serif",
      color: "#fff", position: "relative", overflow: "hidden",
    }}>
      <GraffitiBackground />

      <div style={{ position:"relative", zIndex:1 }}>

        {/* ── header ── */}
        <div style={{
          padding: "22px 40px 18px",
          borderBottom: "3px solid rgba(255,45,45,0.8)",
          display: "flex", justifyContent: "space-between", alignItems: "flex-end",
        }}>
          <div>
            <div style={{ fontSize:9, letterSpacing:"0.4em", color:"#ff2d2d", marginBottom:4 }}>
              AI POWERED
            </div>
            <div style={{ fontSize:"clamp(20px,3.5vw,42px)", letterSpacing:"0.04em", lineHeight:1,
              textShadow:"0 2px 20px rgba(255,45,45,0.4)" }}>
              BREAKDANCE COACH
            </div>
          </div>
          <div style={{ textAlign:"right" }}>
            <div style={{ fontSize:8, color:"#333", letterSpacing:"0.25em", marginBottom:3 }}>POWERED BY</div>
            <div style={{ fontSize:10, color:"#555", letterSpacing:"0.12em" }}>
              AIST++ · MEDIAPIPE · DTW
            </div>
          </div>
        </div>

        {/* ── move selection ── */}
        <div style={{ padding:"32px 40px 40px", maxWidth:1280, margin:"0 auto" }}>

          <div style={{ display:"flex", alignItems:"center", gap:14, marginBottom:22 }}>
            <div style={{ fontSize:9, letterSpacing:"0.3em", color:"#555" }}>SELECT YOUR MOVE</div>
            <div style={{ flex:1, height:1, background:"rgba(255,45,45,0.2)" }} />
            <div style={{ fontSize:9, letterSpacing:"0.2em", color:"#ff2d2d" }}>
              {moves.length} AVAILABLE
            </div>
          </div>

          {/* grid */}
          <div style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(185px, 1fr))",
            gap: 10, marginBottom: 48,
          }}>
            {moves.map((m) => {
              const dc  = DIFF_COLOR[m.difficulty] || DIFF_COLOR.Advanced;
              const sel = selected === m.id;
              return (
                <div
                  key={m.id}
                  onClick={() => setSelected(m.id)}
                  style={{
                    background: sel
                        ? "rgba(20,0,0,0.88)"
                        : "rgba(0,0,0,0.72)",
                    border:       `2px solid ${sel ? "#ff2d2d" : "rgba(255,255,255,0.07)"}`,
                    padding:      "16px 14px",
                    cursor:       "pointer",
                    position:     "relative",
                    transition:   "all 0.12s",
                    backdropFilter: "blur(6px)",
                    boxShadow:    sel ? "0 0 20px rgba(255,45,45,0.25)" : "none",
                  }}
                >
                  {sel && (
                    <div style={{
                      position:"absolute", top:0, right:0,
                      background:"#ff2d2d", color:"#000",
                      fontSize:7, fontWeight:700,
                      padding:"3px 8px", letterSpacing:"0.2em",
                    }}>SELECTED</div>
                  )}
                  <div style={{
                    display:"inline-block",
                    background:dc.bg, border:`1px solid ${dc.border}`,
                    color:dc.text, fontSize:7,
                    padding:"2px 8px", letterSpacing:"0.2em", marginBottom:9,
                  }}>
                    {m.difficulty.toUpperCase()}
                  </div>
                  <div style={{ fontSize:16, letterSpacing:"0.04em", marginBottom:4, lineHeight:1.1 }}>
                    {m.name}
                  </div>
                  <div style={{ fontSize:8, color:"#333", letterSpacing:"0.12em" }}>
                    {m.source === "AIST++" ? "PROFESSIONAL REFERENCE" : "CUSTOM REFERENCE"}
                  </div>
                  {sel && <div style={{
                    position:"absolute", bottom:0, left:0, right:0,
                    height:2, background:"linear-gradient(90deg,transparent,#ff2d2d,transparent)",
                  }} />}
                </div>
              );
            })}
          </div>

          {/* start */}
          <div style={{ textAlign:"center" }}>
            <button
              onClick={handleStart}
              disabled={!selected || loading}
              style={{
                background:    loading ? "#111" : "linear-gradient(135deg,#ff2d2d,#cc0000)",
                color:         loading ? "#333" : "#fff",
                border:        "none",
                padding:       "20px 90px",
                fontSize:      20,
                fontFamily:    "'Anton','Arial Black',sans-serif",
                letterSpacing: "0.2em",
                cursor:        loading ? "not-allowed" : "pointer",
                boxShadow:     loading ? "none" : "0 4px 30px rgba(255,45,45,0.5)",
                transition:    "all 0.15s",
                textShadow:    "0 2px 8px rgba(0,0,0,0.4)",
              }}
              onMouseEnter={(e) => {
                if (!loading) e.currentTarget.style.boxShadow = "0 6px 40px rgba(255,45,45,0.7)";
              }}
              onMouseLeave={(e) => {
                if (!loading) e.currentTarget.style.boxShadow = "0 4px 30px rgba(255,45,45,0.5)";
              }}
            >
              {loading ? "LOADING..." : "START SESSION"}
            </button>
            <div style={{ marginTop:12, fontSize:8, color:"#2a2a2a", letterSpacing:"0.2em" }}>
              MAKE SURE YOUR FULL BODY IS VISIBLE BEFORE STARTING
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}