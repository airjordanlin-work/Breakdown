// All backend traffic goes through the page's own origin under /api. In dev,
// Vite proxies /api to the backend on :8000 (see vite.config.js). That way a
// phone that opens the app through one HTTPS tunnel reaches the backend too,
// and the WebSocket upgrades to wss:// automatically.
export const API = "/api";

// Shared secret (ACCESS_TOKEN in the backend .env). Open the app once with
// ?token=... in the address; it is saved in this browser and removed from the URL.
const TOKEN_KEY = "breakdown_access_token";

function loadToken() {
  try {
    const url = new URL(window.location.href);
    const fromUrl = url.searchParams.get("token");
    if (fromUrl) {
      localStorage.setItem(TOKEN_KEY, fromUrl);
      url.searchParams.delete("token");
      window.history.replaceState(null, "", url.pathname + url.search + url.hash);
    }
    return localStorage.getItem(TOKEN_KEY) || "";
  } catch {
    return "";
  }
}

const token = loadToken();

// fetch() against the backend with the access token attached.
export function apiFetch(path, options = {}) {
  const headers = { ...(options.headers || {}) };
  if (token) headers["X-Access-Token"] = token;
  return fetch(`${API}${path}`, { ...options, headers });
}

export function wsUrl(path) {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  const query = token ? `?token=${encodeURIComponent(token)}` : "";
  return `${proto}//${window.location.host}${API}${path}${query}`;
}
