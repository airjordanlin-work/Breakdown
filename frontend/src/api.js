// All backend traffic goes through the page's own origin under /api. In dev,
// Vite proxies /api to the backend on :8000 (see vite.config.js). That way a
// phone that opens the app through one HTTPS tunnel reaches the backend too,
// and the WebSocket upgrades to wss:// automatically.
export const API = "/api";

export function wsUrl(path) {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}${API}${path}`;
}
