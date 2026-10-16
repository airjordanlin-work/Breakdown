# Phone demo: steps and manual test checklist

This is the M1 demo from the [product spec](PRODUCT_SPEC.html) (section 11): a short video of a phone session with spoken cues. It covers FR-1 (camera over HTTPS), FR-3 (skeleton and framing guide), FR-5 and FR-6 (move matching and scoring), FR-10 (coach audio on the phone) and FR-14 (debrief after an attempt).

Everything here runs on your laptop. The phone only opens a web page. None of these steps can be checked from a cloud session, so the checklist below is filled in by hand on a real device.

## What is in scope

| Step | Status on `main` |
|---|---|
| Open the app on a phone over a cloudflared tunnel, with `ACCESS_TOKEN` | EXISTS, not yet verified on an iPhone |
| Framing guide, skeleton overlay, captions in portrait | EXISTS (PR #28), iPhone recheck owed |
| Live coach cues as captions and Kokoro audio streamed to the phone | EXISTS (PR #24, #31), iPhone check owed |
| Reps, clean streak and focus joint on screen | EXISTS (PR #32) |
| Spoken debrief when you stop after an attempt | EXISTS (PR #31, #32), iPhone check owed |
| Talking back to the coach (push-to-talk, questions) | PROPOSED, PR #33 not on `main` |
| Move-specific cues (hip twist, leg reach) and SHOW ME demo | PROPOSED, PR #34 not on `main` |

Skip the PROPOSED rows for this clip. Add them to the checklist once those PRs land on `main`.

## Before you start

1. Pull `main` and set up the backend, frontend, coach and voice as in the README, sections 1 to 3. You need `ANTHROPIC_API_KEY` in `.env` and the two Kokoro model files in `backend/` or the coach stays silent.
2. Generate an access token and put it in `.env` as `ACCESS_TOKEN` (`.env.example` has the one-line command). Do this before any tunnel starts.
3. Install cloudflared (`brew install cloudflared`).
4. Clear a practice space and find a spot to prop the phone in portrait, about hip height, 2 to 3 meters from where you will dance.
5. Turn the phone's silent switch off and the volume up. Safari plays web audio through the speaker only when the phone is not muted.

## Run it

1. From the repo root, start both servers:

   ```bash
   ./dev.sh
   ```

   Check the backend log. You should see the coach report it is ready and no warning that `ACCESS_TOKEN` is unset. If you see `Coach voice is off` later in the app, Kokoro did not load; the log says why.

2. In a second terminal, start the tunnel:

   ```bash
   cloudflared tunnel --url http://localhost:5173
   ```

3. On the iPhone, open the `https://...trycloudflare.com` address it prints, with your token added once: `/?token=YOUR_TOKEN`. Type it on the phone or AirDrop it. Do not paste the URL or the token into a chat, issue or commit.
4. Allow camera access when Safari asks.
5. In the move library, pick **BR 01**. This is the toprock cross step. Read its card, then tap **Start practice**.
6. Tap the screen once. If **TAP FOR SOUND** is showing, tap it until it goes away.
7. Prop the phone up and step back until the framing guide clears and the skeleton appears.

## What to do on camera

1. Stand still for a moment so the guide clears. The coach should stay quiet while you are still.
2. Do the cross step for a few reps. Watch for the move name in the top bar, the buffer reaching READY, and the reps and clean count on the score chip.
3. Keep going until you hear at least two live cues. Each one should appear as a caption and play out loud at the same time.
4. Stop and stand still. You should hear one short debrief about the attempt you just did.
5. Tap **END SESSION**.

Record the clip with a second phone or a screen recording on the session phone. Screen recording on iOS can capture app audio; turn the microphone off in the recording control so your room noise stays out.

## Manual test checklist

Fill this in on the device. Write down what you saw, not what you expected. If a box fails, note the exact text on screen and the backend log lines around it.

**Setup**
- [ ] Backend log shows no `ACCESS_TOKEN is not set` warning
- [ ] In a Private tab, the tunnel URL without `?token=` shows "Can't reach the coaching server" instead of the move list (the backend answers 401). Do this first, since the token is saved in the browser once used
- [ ] Opening it with `?token=` loads the move library, and the token disappears from the address bar
- [ ] Safari asks for camera access and the session starts after you allow it

**Framing and tracking (FR-1, FR-3)**
- [ ] Portrait layout fits the screen with no sideways scrolling
- [ ] Framing guide shows until your whole body is in view, then clears
- [ ] Skeleton follows you while you move
- [ ] Captions wrap and stay readable at each **TEXT S/M/L** size

**Scoring (FR-5, FR-6)**
- [ ] Top bar shows the move name or ALIGNED once you are doing the move
- [ ] Rep and clean counts on the score chip go up as you dance
- [ ] Sidebar shows a focus joint after a few reps

**Coach voice (FR-10)**
- [ ] No coach cue while you stand still before dancing
- [ ] Live cues appear as captions while you move
- [ ] Each cue is audible from the phone speaker, not the laptop
- [ ] Audio starts close to when the caption shows
- [ ] No **Coach voice is off** or **No audio for the last cue** banner

**Debrief (FR-14)**
- [ ] One debrief plays after you stop, and only one
- [ ] It talks about the attempt you just did

**Ending**
- [ ] **END SESSION** returns to the move library with no error
- [ ] Laptop audio stayed silent the whole time
- [ ] Tunnel stopped with Ctrl+C when done

**Grounding spot check (rule 1 in CLAUDE.md)**
- [ ] Every cue and the debrief only mention things on screen or in the move's tips (a joint, a rep count, a streak). Note any cue that states a number or body fact you cannot see in the app.

## After recording

1. Stop the tunnel (Ctrl+C) and the servers.
2. Watch the clip once for the tunnel URL, the token or anything private in frame, and crop it out.
3. Do not commit the video. Upload it somewhere public and link it in the README in place of "Demo video coming."
4. Report the checklist results in the project thread, with any failing line and its log output.
