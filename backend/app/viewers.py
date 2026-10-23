"""Second screens that mirror a live session (spec 3.1, a TV or laptop viewer).

The phone is the camera and the only controller. A viewer is a read-only
WebSocket on the same session: it gets every frame result, coach cue and audio
chunk the phone gets, so a TV or laptop can show the dancer a large mirror
view. Viewers never send frames, and nothing here stores video: the payloads
are the landmarks and scores the phone already receives, plus (only for a viewer
that opts in) a small, low-quality camera picture forwarded as it arrives.
"""

from __future__ import annotations

import asyncio
from typing import Any

# A slow viewer must not stall the camera loop, so each send gets this long.
SEND_TIMEOUT_S = 1.0


class ViewerHub:
    """The viewer sockets attached to one session."""

    def __init__(self) -> None:
        self._viewers: set[Any] = set()
        self._video: set[Any] = set()   # viewers that asked for camera pictures

    @property
    def count(self) -> int:
        return len(self._viewers)

    def add(self, ws: Any) -> None:
        self._viewers.add(ws)

    def discard(self, ws: Any) -> None:
        self._viewers.discard(ws)
        self._video.discard(ws)

    @property
    def video_count(self) -> int:
        return len(self._video)

    def want_video(self, ws: Any, on: bool) -> None:
        """A viewer opts in or out of the camera picture. Off by default."""
        if ws not in self._viewers:
            return
        (self._video.add if on else self._video.discard)(ws)

    async def close_all(self) -> None:
        """Close every viewer, so a page following this session looks for the next one."""
        for ws in list(self._viewers):
            self._viewers.discard(ws)
            try:
                await asyncio.wait_for(ws.close(), SEND_TIMEOUT_S)
            except Exception:
                pass  # already closed

    async def _send(self, send_name: str, data: str | bytes, targets: Any = None) -> bool:
        """Send to every viewer (or just ``targets``); drop any that fail or time out. True if the set changed."""
        changed = False
        for ws in list(self._viewers if targets is None else targets):
            try:
                await asyncio.wait_for(getattr(ws, send_name)(data), SEND_TIMEOUT_S)
            except Exception:
                self.discard(ws)
                changed = True
        return changed

    async def broadcast_video(self, text: str) -> bool:
        """One small camera picture, only to the viewers that opted in. Never stored."""
        return await self._send("send_text", text, self._video)

    async def broadcast_text(self, text: str) -> bool:
        return await self._send("send_text", text)

    async def broadcast_bytes(self, data: bytes) -> bool:
        return await self._send("send_bytes", data)
