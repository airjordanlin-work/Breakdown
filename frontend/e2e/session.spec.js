// Session screen on an iPhone-sized viewport with a faked coach socket.
// Covers what Jordan checks by hand: the top bar fits, SHOW ME works, the
// result card and captions appear, and hold-to-talk sends the right messages.
import { test, expect } from "@playwright/test";

// Fake the session WebSocket. Returns the list of messages the page sent,
// and a `push` function that sends a message to the page.
async function fakeCoach(page, { onMessage } = {}) {
  const sent = [];
  let route = null;
  await page.routeWebSocket(/\/api\/ws\/[^/]+$/, (ws) => {
    route = ws;
    ws.send(JSON.stringify({ type: "voice_status", available: true }));
    ws.onMessage((raw) => {
      if (typeof raw !== "string") return;
      const msg = JSON.parse(raw);
      if (msg.type === "frame") {
        // Answer every frame, or the page stops sending (backpressure).
        ws.send(JSON.stringify({ type: "frame", move_name: "cross_step", live_visibility: Array(33).fill(0.9) }));
        return;
      }
      sent.push(msg);
      onMessage?.(msg, ws);
    });
  });
  return { sent, push: (m) => route.send(JSON.stringify(m)) };
}

async function startSession(page) {
  await page.goto("/");
  await page.getByRole("button", { name: /start practice/i }).click();
  await expect(page.getByRole("button", { name: "Show me the move slowly" })).toBeVisible();
}

test("top bar fits the phone screen", async ({ page }) => {
  await fakeCoach(page);
  await startSession(page);

  const vp = page.viewportSize();
  // Read every button's position in one step: TAP FOR SOUND can disappear while we look.
  const rects = await page.locator(".sess-top button").evaluateAll((els) =>
    els.map((el) => { const r = el.getBoundingClientRect(); return { x: r.x, right: r.right, w: r.width }; })
      .filter((r) => r.w > 0));
  expect(rects.length).toBeGreaterThanOrEqual(4);
  rects.forEach((r, i) => {
    expect(r.x, `button ${i} left edge`).toBeGreaterThanOrEqual(0);
    expect(r.right, `button ${i} right edge`).toBeLessThanOrEqual(vp.width);
  });
  // No sideways scrolling, and the bar leaves most of the screen for the camera.
  const scrollW = await page.evaluate(() => document.documentElement.scrollWidth);
  expect(scrollW).toBeLessThanOrEqual(vp.width);
  const top = await page.locator(".sess-top").boundingBox();
  expect(top.height).toBeLessThan(vp.height * 0.3);
  await page.screenshot({ path: test.info().outputPath("session-top.png") });
});

test("SHOW ME asks for a demo and the overlay opens and closes", async ({ page }) => {
  const coach = await fakeCoach(page, {
    onMessage: (msg, ws) => {
      if (msg.type === "demo_request") {
        ws.send(JSON.stringify({
          type: "demo", move_id: "br_01", move_name: "cross step",
          speed: 0.5, loops: 2, seconds: 3, focus: "left knee", highlight: [],
        }));
      }
    },
  });
  await startSession(page);
  await page.getByRole("button", { name: "Show me the move slowly" }).click();
  const overlay = page.getByRole("dialog", { name: /Demo of cross step/ });
  await expect(overlay).toBeVisible();
  await expect(overlay).toContainText("WATCH THE LEFT KNEE");
  expect(coach.sent.map((m) => m.type)).toContain("demo_request");
  await overlay.getByRole("button", { name: "CLOSE" }).click();
  await expect(overlay).toBeHidden();
  await expect.poll(() => coach.sent.map((m) => m.type)).toContain("demo_end");
});

test("LIVE CUES starts off and the toggle is sent to the server", async ({ page }) => {
  const coach = await fakeCoach(page);
  await startSession(page);
  const btn = page.getByRole("button", { name: "Spoken cues while you dance" });
  await expect(btn).toHaveAttribute("aria-pressed", "false");
  await expect.poll(() => coach.sent.find((m) => m.type === "live_cues")?.on).toBe(false);
  await btn.click();
  await expect(btn).toHaveAttribute("aria-pressed", "true");
  await expect.poll(() => coach.sent.filter((m) => m.type === "live_cues").pop()?.on).toBe(true);
});

test("caption, result card and missing-audio note show up", async ({ page }) => {
  const coach = await fakeCoach(page);
  await startSession(page);
  coach.push({ type: "coach_cue", cue: "Nice. Straighten the left knee.", turn_id: 1, mode: "debrief" });
  await expect(page.getByText("Straighten the left knee")).toBeVisible();
  coach.push({ type: "coach_audio_skipped", turn_id: 1, reason: "tts timed out" });
  await expect(page.getByText(/no audio for the last cue: tts timed out/i)).toBeVisible();
  coach.push({
    type: "rep_summary", move: "cross step", reps: 5, clean_reps: 3, best_streak: 2,
    went_well: ["hips"], fix: { joint: "left knee", fix: "straighten", checkpoint: 2, of: 4 },
  });
  const card = page.getByRole("dialog", { name: "Result of your last attempt" });
  await expect(card).toContainText("3 OF 5 CLEAN");
  await expect(card).toContainText("STRAIGHTEN LEFT KNEE");
});

test("coach voice off is shown", async ({ page }) => {
  const coach = await fakeCoach(page);
  await startSession(page);
  coach.push({ type: "voice_status", available: false, reason: "Kokoro model file missing" });
  await expect(page.getByText(/coach voice is off: kokoro model file missing/i)).toBeVisible();
});

test("hold to talk sends ptt_start then ptt_end", async ({ page, browserName }) => {
  // WebKit has no fake microphone, so the mic fails to open and the page cancels the turn.
  // Real iOS mic behavior is on the manual checklist.
  test.skip(browserName === "webkit", "no fake mic in WebKit");
  await page.route("**/api/stt/status", (r) =>
    r.fulfill({ json: { available: true, state: "ready", reason: null } }));
  const coach = await fakeCoach(page);
  await startSession(page);
  const talk = page.getByRole("button", { name: "Hold to talk to the coach" });
  await expect(talk).toBeVisible();
  const box = await talk.boundingBox();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await expect.poll(() => coach.sent.map((m) => m.type)).toContain("ptt_start");
  await page.waitForTimeout(600);
  await page.mouse.up();
  await expect.poll(() => coach.sent.map((m) => m.type)).toContain("ptt_end");
});

test("without speech to text the typed question box is shown", async ({ page }) => {
  await page.route("**/api/stt/status", (r) =>
    r.fulfill({ json: { available: false, state: "off", reason: "not installed" } }));
  const coach = await fakeCoach(page);
  await startSession(page);
  const box = page.getByLabel("Ask the coach");
  await expect(box).toBeVisible();
  await box.fill("how did I do?");
  await box.press("Enter");
  await expect.poll(() => coach.sent.find((m) => m.type === "chat")?.text).toBe("how did I do?");
});

test.describe("landscape phone", () => {
  test.use({ viewport: { width: 844, height: 390 } });
  test("top bar leaves room for the camera", async ({ page }) => {
    await fakeCoach(page);
    await startSession(page);
    const top = await page.locator(".sess-top").boundingBox();
    expect(top.height).toBeLessThan(390 * 0.35);
    const scrollW = await page.evaluate(() => document.documentElement.scrollWidth);
    expect(scrollW).toBeLessThanOrEqual(844);
  });
});
