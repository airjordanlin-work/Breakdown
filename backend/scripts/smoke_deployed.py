#!/usr/bin/env python3
"""Smoke test for a running Breakdown app, local or through a tunnel.

    python backend/scripts/smoke_deployed.py https://xyz.trycloudflare.com
    python backend/scripts/smoke_deployed.py http://localhost:5173
    python backend/scripts/smoke_deployed.py http://localhost:8000 --api-prefix ""

The token comes from the SMOKE_TOKEN or ACCESS_TOKEN environment variable (never
an argument, so it stays out of shell history) and is never printed. It checks
the page, the REST routes, and the session WebSocket (voice_status, SHOW ME). It
sends no camera frames and spends no model tokens. It cannot check audio, camera
or pose quality; see docs/plans/test-automation.md for the manual list.
Exit code is 0 when every check passes.
"""
import argparse
import asyncio
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

import websockets


def _request(method, url, token, timeout=15):
    req = urllib.request.Request(url, method=method)
    if token:
        req.add_header("X-Access-Token", token)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


async def _ws_checks(ws_url, results):
    async with websockets.connect(ws_url, open_timeout=15) as ws:
        # voice_status is the first thing the server sends on connect.
        msg = None
        for _ in range(5):
            raw = await asyncio.wait_for(ws.recv(), timeout=10)
            if isinstance(raw, str) and json.loads(raw).get("type") == "voice_status":
                msg = json.loads(raw)
                break
        results.append(("ws voice_status", msg is not None, "" if msg is None else
                        f"available={msg.get('available')} reason={msg.get('reason')}"))
        await ws.send(json.dumps({"type": "demo_request"}))
        reply = None
        for _ in range(10):
            raw = await asyncio.wait_for(ws.recv(), timeout=10)
            if isinstance(raw, str):
                m = json.loads(raw)
                if m.get("type") in ("demo", "demo_unavailable"):
                    reply = m
                    break
        results.append(("ws demo_request answered", reply is not None,
                        "" if reply is None else reply["type"]))
        await ws.send(json.dumps({"type": "demo_end"}))


def run(base, api_prefix, token):
    base = base.rstrip("/")
    api = base + api_prefix
    results = []

    def check(name, fn):
        try:
            ok, detail = fn()
        except Exception as e:  # report and keep going
            ok, detail = False, f"{type(e).__name__}: {str(e)[:120]}"
        results.append((name, ok, detail))
        return ok

    if api_prefix:  # only the frontend origin serves the page
        def page():
            status, body = _request("GET", base + "/", token)
            return status == 200 and b'id="root"' in body, f"HTTP {status}"
        check("page loads", page)

    check("health", lambda: (lambda s, b: (s == 200 and json.loads(b).get("status") == "ok", b.decode()[:80]))(
        *_request("GET", api + "/health", token)))

    def moves():
        s, b = _request("GET", api + "/moves", token)
        n = len(json.loads(b)["moves"])
        return s == 200 and n > 0, f"{n} moves"
    check("moves list", moves)

    def stt():
        s, b = _request("GET", api + "/stt/status", token)
        d = json.loads(b)
        return s == 200, f"available={d.get('available')} state={d.get('state')}"
    check("stt status", stt)

    def auth():
        # With a token configured, a request without one must be refused.
        try:
            _request("GET", api + "/health", None)
        except urllib.error.HTTPError as e:
            return e.code == 401, f"HTTP {e.code}"
        return not token, "open (no token set)" if not token else "NOT refused without a token"
    check("auth gate", auth)

    sid = None
    req = urllib.request.Request(api + "/session/start", method="POST", data=b"{}",
                                 headers={"Content-Type": "application/json"})
    if token:
        req.add_header("X-Access-Token", token)
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            sid = json.loads(r.read()).get("session_id")
    except Exception as e:
        results.append(("session start", False, f"{type(e).__name__}: {str(e)[:120]}"))
    else:
        results.append(("session start", bool(sid), "got a session id" if sid else "no session id"))

    if sid:
        scheme = "wss" if api.startswith("https") else "ws"
        q = "?" + urllib.parse.urlencode({"token": token}) if token else ""
        ws_url = f"{scheme}://{api.split('://', 1)[1]}/ws/{sid}{q}"
        try:
            asyncio.run(_ws_checks(ws_url, results))
        except Exception as e:
            results.append(("websocket", False, f"{type(e).__name__}: {str(e)[:120]}"))
        try:
            _request("DELETE", f"{api}/session/{sid}", token)
            results.append(("session end", True, ""))
        except Exception as e:
            results.append(("session end", False, str(e)[:120]))

    width = max(len(n) for n, _, _ in results)
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
    failed = [n for n, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    return 1 if failed else 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("url", help="app origin, e.g. https://xyz.trycloudflare.com or http://localhost:5173")
    p.add_argument("--api-prefix", default="/api",
                   help='path the backend is served under (default /api; use "" for port 8000 directly)')
    args = p.parse_args()
    token = os.environ.get("SMOKE_TOKEN") or os.environ.get("ACCESS_TOKEN") or ""
    sys.exit(run(args.url, args.api_prefix, token))


if __name__ == "__main__":
    main()
