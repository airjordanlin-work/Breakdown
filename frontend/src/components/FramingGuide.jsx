// Full-body framing guide shown until the whole dancer has been in frame for a
// moment (see framing.js). The dashed outline is where the body should fit.
export default function FramingGuide({ problem }) {
  return (
    <div className="framing-guide" role="status" aria-live="polite">
      <div className="framing-outline" aria-hidden="true">
        <svg viewBox="0 0 100 200" preserveAspectRatio="xMidYMid meet">
          <circle cx="50" cy="22" r="12" />
          <path d="M50 34 V110 M50 48 L22 92 M50 48 L78 92 M50 110 L30 186 M50 110 L70 186" />
        </svg>
      </div>
      <div className="framing-text">
        <div className="framing-title">FRAME YOUR WHOLE BODY</div>
        <div className="framing-problem">{problem || "HOLD STILL"}</div>
        <div className="framing-hint">Prop the phone up and stand 2 to 3 meters away</div>
      </div>
    </div>
  );
}
