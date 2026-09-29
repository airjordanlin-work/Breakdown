import { useRef, useCallback, useEffect } from "react";

// How long to wait for a reply before assuming it was lost and sending again.
// Prevents the pipeline from freezing if one response never arrives.
const STALE_MS = 1000;

export function useWebSocket(sessionId, onMessage, onOpen) {
  const ws        = useRef(null);
  const onOpenRef = useRef(onOpen);
  const onMsgRef  = useRef(onMessage);

  // Backpressure state: is a frame currently being processed by the backend?
  const inFlight  = useRef(false);
  const sentAt    = useRef(0);

  useEffect(() => { onOpenRef.current = onOpen; }, [onOpen]);
  useEffect(() => { onMsgRef.current  = onMessage; }, [onMessage]);

  useEffect(() => {
    if (!sessionId) return;
    ws.current = new WebSocket(`ws://localhost:8000/ws/${sessionId}`);
    ws.current.onopen = () => {
      inFlight.current = false;
      onOpenRef.current?.();
    };
    ws.current.onmessage = (e) => {
      const msg = JSON.parse(e.data);
      if (msg.type === "frame" || msg.error) {
        // Backend finished the frame we sent, so we're free to send the next one.
        msg._rttMs = Math.round(performance.now() - sentAt.current);
        inFlight.current = false;
      }
      onMsgRef.current?.(msg);
    };
    ws.current.onerror = (e) => console.error("WS error", e);
    ws.current.onclose = () => { inFlight.current = false; };
    return () => ws.current?.close();
  }, [sessionId]);

  // True when the socket is open and the backend isn't busy with a previous frame.
  const canSend = useCallback(() => {
    if (ws.current?.readyState !== WebSocket.OPEN) return false;
    if (!inFlight.current) return true;
    return performance.now() - sentAt.current > STALE_MS;
  }, []);

  const sendFrame = useCallback((b64) => {
    if (!canSend()) return false;
    inFlight.current = true;
    sentAt.current   = performance.now();
    // t = capture time in seconds on the laptop clock. The IMU bridge stamps
    // packets with the same clock, so the backend can line the two up.
    ws.current.send(JSON.stringify({ type: "frame", data: b64, t: Date.now() / 1000 }));
    return true;
  }, [canSend]);

  return { sendFrame, canSend };
}