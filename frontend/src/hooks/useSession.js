import { useState, useCallback } from "react";
import { apiFetch } from "../api";

export function useSession() {
  const [sessionId, setSessionId] = useState(null);
  const [loading, setLoading]     = useState(false);
  const startSession = useCallback(async (config = {}) => {
    setLoading(true);
    try {
      const res = await apiFetch(`/session/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config),
      });
      const { session_id } = await res.json();
      setSessionId(session_id);
      return session_id;
    } finally { setLoading(false); }
  }, []);
  const endSession = useCallback(() => setSessionId(null), []);
  return { sessionId, loading, startSession, endSession };
}
