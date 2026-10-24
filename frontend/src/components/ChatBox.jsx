import { useState } from "react";

// Type a question to the coach. Shown in debug mode, or when the backend
// cannot hear (no speech to text installed).
export default function ChatBox({ onSend, disabled }) {
  const [text, setText] = useState("");
  const submit = (e) => {
    e.preventDefault();
    const t = text.trim();
    if (!t || disabled) return;
    if (onSend(t)) setText("");
  };
  return (
    <form className="chat-box" onSubmit={submit}>
      <input
        value={text}
        onChange={(e) => setText(e.target.value)}
        maxLength={300}
        placeholder="Ask the coach"
        aria-label="Ask the coach"
        enterKeyHint="send"
      />
      <button type="submit" disabled={disabled || !text.trim()}>SEND</button>
    </form>
  );
}
