import { useCallback, useEffect, useMemo, useState } from "react";
import MovePreview from "../components/MovePreview";
import "../styles/move-library.css";

const API = "http://localhost:8000";
const TIERS = ["Beginner", "Intermediate", "Advanced"];
const STATS = [
  { key: "power",     label: "Power",     hint: "How fast the arms and legs move, compared to the other moves" },
  { key: "floor",     label: "Floor",     hint: "Share of the move spent low or on the ground" },
  { key: "inversion", label: "Inversion", hint: "Share of the move spent upside down" },
];

function StatBar({ label, value, hint }) {
  const filled = Math.round(value / 10);
  return (
    <div className="stat" title={hint}>
      <span className="stat-label">{label}</span>
      <span className="stat-track" role="meter" aria-label={label}
        aria-valuemin={0} aria-valuemax={100} aria-valuenow={value}>
        {Array.from({ length: 10 }, (_, i) => (
          <span key={i} className={i < filled ? "seg on" : "seg"} />
        ))}
      </span>
      <span className="stat-value">{value}</span>
    </div>
  );
}

function useSensorStatus() {
  const [connected, setConnected] = useState(false);
  useEffect(() => {
    let alive = true;
    const check = () =>
      fetch(`${API}/imu/status`)
        .then((r) => r.json())
        .then((s) => alive && setConnected(Boolean(s.connected)))
        .catch(() => alive && setConnected(false));
    check();
    const id = setInterval(check, 3000);
    return () => { alive = false; clearInterval(id); };
  }, []);
  return connected;
}

export default function MoveLibrary({ onStart }) {
  const [moves, setMoves] = useState(null);
  const [error, setError] = useState(null);
  const [selectedId, setSelectedId] = useState(null);
  const [starting, setStarting] = useState(false);
  const sensors = useSensorStatus();

  const load = useCallback(() => {
    setError(null);
    fetch(`${API}/moves`)
      .then((r) => r.json())
      .then((d) => {
        setMoves(d.moves);
        if (d.moves.length) setSelectedId((id) => id ?? d.moves[0].id);
      })
      .catch(() => setError("Can't reach the coaching server. Start the backend, then try again."));
  }, []);
  useEffect(load, [load]);

  // Tier order drives both the list and arrow-key navigation.
  const ordered = useMemo(() => {
    if (!moves) return [];
    return TIERS.flatMap((t) => moves.filter((m) => m.difficulty === t))
      .concat(moves.filter((m) => !TIERS.includes(m.difficulty)));
  }, [moves]);

  const move = ordered.find((m) => m.id === selectedId);

  // Keep the selected move visible in the list (matters on the phone strip).
  useEffect(() => {
    document.querySelector(".move-item.selected")?.scrollIntoView({ block: "nearest", inline: "nearest" });
  }, [selectedId]);

  const start = useCallback(async () => {
    if (!move || starting) return;
    setStarting(true);
    try { await onStart({ move_id: move.id }); }
    catch { setError("Couldn't start the session. Check that the backend is running."); setStarting(false); }
  }, [move, starting, onStart]);

  useEffect(() => {
    const onKey = (e) => {
      if (!ordered.length || e.target.closest?.("input, textarea")) return;
      const i = ordered.findIndex((m) => m.id === selectedId);
      if (e.key === "ArrowDown" || e.key === "ArrowRight") {
        e.preventDefault();
        setSelectedId(ordered[(i + 1) % ordered.length].id);
      } else if (e.key === "ArrowUp" || e.key === "ArrowLeft") {
        e.preventDefault();
        setSelectedId(ordered[(i - 1 + ordered.length) % ordered.length].id);
      } else if (e.key === "Enter" && e.target === document.body) {
        start();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [ordered, selectedId, start]);

  if (error && !moves) {
    return (
      <main className="library library--empty">
        <p>{error}</p>
        <button type="button" className="start" onClick={load}>Try again</button>
      </main>
    );
  }
  if (!moves) return <main className="library library--empty"><p>Loading moves</p></main>;
  if (!moves.length) {
    return (
      <main className="library library--empty">
        <p>No moves yet. Add a reference move to backend/reference_moves, then reload.</p>
      </main>
    );
  }

  return (
    <main className="library">
      <header className="library-bar">
        <h1 className="wordmark">Breakdown</h1>
        <span className={sensors ? "sensor-pill on" : "sensor-pill"}>
          {sensors ? "Sensors connected" : "Sensors off"}
        </span>
      </header>

      <nav className="move-list" aria-label="Moves">
        {TIERS.map((tier) => {
          const inTier = ordered.filter((m) => m.difficulty === tier);
          if (!inTier.length) return null;
          return (
            <section key={tier} className="tier" data-tier={tier}>
              <h2 className="tier-name">{tier}</h2>
              <ul>
                {inTier.map((m) => (
                  <li key={m.id}>
                    <button type="button"
                      className={m.id === selectedId ? "move-item selected" : "move-item"}
                      aria-current={m.id === selectedId}
                      onClick={() => setSelectedId(m.id)}
                      onDoubleClick={start}>
                      {m.display_name}
                    </button>
                  </li>
                ))}
              </ul>
            </section>
          );
        })}
      </nav>

      {move && <MovePreview moveId={move.id} title={move.display_name} />}

      {move && (
        <aside className="sheet" data-tier={move.difficulty}>
          <p className="sheet-tier">
            {move.difficulty}
            {move.difficulty_measured && <span className="sheet-tier-note">rated from motion data</span>}
          </p>
          <h2 className="sheet-name">{move.display_name}</h2>
          <p className="sheet-desc">{move.description}</p>

          <div className="stats">
            {STATS.map((s) => (
              <StatBar key={s.key} label={s.label} value={move.stats[s.key]} hint={s.hint} />
            ))}
          </div>

          <h3 className="sheet-heading">Before you start</h3>
          <ul className="reqs">
            {move.requirements.map((r) => {
              const live = r.kind === "sensors";
              const met = live && sensors;
              return (
                <li key={r.kind} className={live ? (met ? "req met" : "req unmet") : "req"}>
                  {r.text}
                  {live && <span className="req-state">{met ? "Connected" : "Not connected"}</span>}
                </li>
              );
            })}
          </ul>

          {move.tips.length > 0 && (
            <>
              <h3 className="sheet-heading">Tips</h3>
              <ul className="tips">{move.tips.map((t) => <li key={t}>{t}</li>)}</ul>
            </>
          )}

          <button type="button" className="start" onClick={start} disabled={starting}>
            {starting ? "Starting" : "Start practice"}
          </button>
          <p className="keys-hint">Arrow keys to browse, Enter to start</p>
        </aside>
      )}
    </main>
  );
}
