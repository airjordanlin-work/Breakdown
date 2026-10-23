import { useEffect, useState } from "react";
import { apiFetch, wsUrl } from "../api";

const RETRY_MS = 2000;
const FOLLOW_MS = 3000;

// Follows the phone's newest session as a read-only viewer (a TV or laptop).
// Looks for the latest session every couple of seconds, connects to its
// /view socket, and starts over when the socket closes (the phone ended the
// session or the network dropped). While connected it keeps checking, and
// moves to a newer session when the phone starts one. onSession(id) runs when
// the page joins a different session, so the page can clear the old one.
// status is "searching" until the first message arrives, then "live".
export function useViewerSocket(onMessage, onBinary, onSession) {
  const [status, setStatus] = useState("searching");

  useEffect(() => {
    let stopped = false;
    let ws = null;
    let timer = null;
    let lastId = null;
    const sleep = (ms) => new Promise((r) => { timer = setTimeout(r, ms); });

    async function latest() {
      try {
        const res = await apiFetch("/sessions/latest");
        if (res.ok) return (await res.json()).session_id;
      } catch { /* backend not up yet; keep trying */ }
      return null;
    }

    async function run() {
      while (!stopped) {
        const id = await latest();
        if (stopped) return;
        if (!id) { setStatus("searching"); await sleep(RETRY_MS); continue; }
        if (id !== lastId) { lastId = id; onSession?.(id); }

        await new Promise((resolve) => {
          const sock = new WebSocket(wsUrl(`/ws/${id}/view`));
          ws = sock;
          sock.binaryType = "arraybuffer";
          const ping = setInterval(() => {
            if (sock.readyState === WebSocket.OPEN) sock.send(JSON.stringify({ type: "ping" }));
          }, 15000);
          const follow = setInterval(async () => {
            const newest = await latest();
            if (newest && newest !== id) sock.close();
          }, FOLLOW_MS);
          sock.onmessage = (e) => {
            if (e.data instanceof ArrayBuffer) { onBinary?.(e.data); return; }
            let msg;
            try { msg = JSON.parse(e.data); } catch { return; }
            if (msg.error) { sock.close(); return; }
            setStatus("live");
            onMessage?.(msg);
          };
          sock.onclose = () => { clearInterval(ping); clearInterval(follow); resolve(); };
          sock.onerror = () => sock.close();
        });
        if (!stopped) { setStatus("searching"); await sleep(RETRY_MS); }
      }
    }
    run();
    return () => { stopped = true; clearTimeout(timer); ws?.close(); };
    // handlers are read once; the page passes stable callbacks
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return status;
}
