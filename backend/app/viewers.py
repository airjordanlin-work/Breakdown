"""Second screens that mirror a live session (spec 3.1, a TV or laptop viewer).

The phone is the camera and the only controller. A viewer is a read-only
WebSocket on the same session: it gets every frame result, coach cue and audio
chunk the phone gets, so a TV or laptop can show the dancer a large mirror
view. Viewers never send frames, and nothing here stores video: the payloads
are the landmarks and scores the phone already receives.
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

    @property
    def count(self) -> int:
        return len(self._viewers)

    def add(self, ws: Any) -> None:
        self._viewers.add(ws)

    def discard(self, ws: Any) -> None:
        self._viewers.discard(ws)

    async def _send(self, send_name: str, data: str | bytes) -> bool:
        """Send to every viewer; drop any that fail or time out. True if the set changed."""
        changed = False
        for ws in list(self._viewers):
            try:
                await asyncio.wait_for(getattr(ws, send_name)(data), SEND_TIMEOUT_S)
            except Exception:
                self._viewers.discard(ws)
                changed = True
        return changed

    async def broadcast_text(self, text: str) -> bool:
        return await self._send("send_text", text)

    async def broadcast_bytes(self, data: bytes) -> bool:
        return await self._send("send_bytes", data)
