import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Extra tunnel hostnames, comma-separated (for example a named Cloudflare tunnel).
const extraHosts = (process.env.ALLOWED_HOSTS || "")
  .split(",").map((h) => h.trim()).filter(Boolean);
const allowedHosts = [
  ".trycloudflare.com", ".ngrok-free.app", ".ngrok-free.dev", ".ngrok.app", ".ngrok.dev",
  ...extraHosts,
];

const BACKEND = process.env.BACKEND_URL || "http://localhost:8000";

// /api/* goes to the backend with the prefix stripped, WebSockets included.
// The same proxy serves `vite preview`, so a production build works too.
const proxy = {
  "/api": {
    target: BACKEND,
    changeOrigin: true,
    ws: true,
    rewrite: (path) => path.replace(/^\/api/, ""),
  },
};

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: true,   // listen on the LAN too, so a phone on the same WiFi can connect
    // Allow HTTPS tunnel hostnames (cloudflared, ngrok) in front of the dev server.
    allowedHosts,
    proxy,
  },
  preview: { port: 4173, host: true, allowedHosts, proxy },
});
