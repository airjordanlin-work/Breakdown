"""Shared-secret access check for the REST routes and WebSockets.

The phone reaches the backend through a public tunnel URL, so anyone who has
that URL could otherwise start sessions. When ACCESS_TOKEN is set, every route
except /health needs the token in an X-Access-Token header or a ?token= query
parameter (browsers cannot set headers on a WebSocket). When it is unset the
check is off, which is fine on localhost and logged as a warning.
"""

from __future__ import annotations

import hmac
import os

from fastapi import HTTPException
from starlette.requests import HTTPConnection

HEADER = "x-access-token"
QUERY_PARAM = "token"


def configured_token() -> str:
    return os.getenv("ACCESS_TOKEN", "").strip()


def token_is_valid(supplied: str | None, expected: str) -> bool:
    """True when auth is off (no expected token) or the supplied token matches."""
    if not expected:
        return True
    if not supplied:
        return False
    return hmac.compare_digest(supplied.encode(), expected.encode())


async def require_token(conn: HTTPConnection) -> None:
    supplied = conn.headers.get(HEADER) or conn.query_params.get(QUERY_PARAM)
    if not token_is_valid(supplied, configured_token()):
        raise HTTPException(status_code=401, detail="Missing or wrong access token")
