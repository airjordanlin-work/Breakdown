// Phone-layout and session-flow tests. See docs/plans/test-automation.md.
//
// The backend is real (REST: moves, session start, previews). The session
// WebSocket is faked inside each test, so no pose model, API key or TTS is
// needed. Camera and mic are Chromium's fake devices.
import { defineConfig, devices } from "@playwright/test";

const PORT = 4173;
const BACKEND_PORT = 8000;
// CI sets BACKEND_PYTHON to the interpreter that has backend/requirements.txt installed.
const python = process.env.BACKEND_PYTHON || "python3";
// Chromium from a pre-installed location (cloud sandbox); unset on a normal machine.
const chromePath = process.env.PW_CHROMIUM_PATH || undefined;

const fakeMedia = [
  "--use-fake-device-for-media-stream",
  "--use-fake-ui-for-media-stream",
  "--autoplay-policy=no-user-gesture-required",
];

export default defineConfig({
  testDir: "./e2e",
  timeout: 30_000,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [["list"], ["html", { open: "never" }]] : "list",
  use: {
    baseURL: `http://localhost:${PORT}`,
    trace: "off",
    screenshot: "only-on-failure",
  },
  projects: [
    {
      // Chromium with iPhone viewport, touch and user agent. Layout only; it is not Safari.
      name: "iphone-chromium",
      use: {
        ...devices["iPhone 14"],
        browserName: "chromium",
        launchOptions: { executablePath: chromePath, args: fakeMedia },
        permissions: ["camera", "microphone"],
      },
    },
    // Real WebKit engine. Needs `npx playwright install webkit`; on by default in CI.
    ...(process.env.PW_WEBKIT
      ? [{
          name: "iphone-webkit",
          use: { ...devices["iPhone 14"] },
        }]
      : []),
  ],
  webServer: [
    {
      command: `${python} -m uvicorn api.server:app --port ${BACKEND_PORT}`,
      cwd: "../backend",
      url: `http://localhost:${BACKEND_PORT}/health`,
      reuseExistingServer: !process.env.CI,
      timeout: 60_000,
      env: { ACCESS_TOKEN: "" },
    },
    {
      command: `npm run build && npx vite preview --port ${PORT} --strictPort`,
      url: `http://localhost:${PORT}`,
      reuseExistingServer: !process.env.CI,
      timeout: 120_000,
    },
  ],
});
