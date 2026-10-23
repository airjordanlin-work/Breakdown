import { useEffect, useState } from "react";
import { apiFetch, wsUrl } from "../api";

const RETRY_MS = 2000;

// Follows the phone's newest session as a read-only viewer (a TV or laptop).
// Looks for the latest session every couple of seconds, connects to its
// /view socket, and starts over when the socket closes (the phone ended the
// session or the network dropped). status is "searching" or "live".
export function useViewerSocket(onMessage, onBinary) {
  const [status, setStatus] = useState("searching");

  useEffect(() => {
    let stopped = false;
    let ws = null;
    let timer = null;
    const sleep = (ms) => new Promise((r) => { timer = setTimeout(r, ms); });

    async function run() {
      while (!stopped) {
        let id = null;
        try {
          const res = await apiFetch("/sessions/latest");
          if (res.ok) id = (await res.json()).session_id;
        } catch { /* backend not up yet; keep trying */ }
        if (!id) { setStatus("searching"); await sleep(RETRY_MS); continue; }

        await new Promise((resolve) => {
          ws = new WebSocket(wsUrl(`/ws/${id}/view`));
          ws.binaryType = "arraybuffer";
          const ping = setInterval(() => {
            if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "ping" }));
          }, 15000);
          ws.onopen = () => setStatus("live");
          ws.onmessage = (e) => {
            if (e.data instanceof ArrayBuffer) { onBinary?.(e.data); return; }
            const msg = JSON.parse(e.data);
            if (msg.error) { ws.close(); return; }
            onMessage?.(msg);
          };
          ws.onclose = () => { clearInterval(ping); resolve(); };
          ws.onerror = () => ws.close();
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
