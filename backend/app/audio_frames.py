"""Binary WebSocket frames that carry coach audio to the client.

Layout (spec section 8, coach_audio_chunk):

    [2 bytes] header length N, unsigned big-endian
    [N bytes] UTF-8 JSON header: {type, turn_id, seq, codec, final}
    [rest]    audio payload (for codec "wav", a complete WAV file)

JSON messages stay text frames, so the client can tell the two apart by
frame type alone.
"""

from __future__ import annotations

import json
import struct

AUDIO_CHUNK_TYPE = "coach_audio_chunk"
_LEN = struct.Struct(">H")


class AudioFrameError(ValueError):
    pass


def encode_audio_chunk(
    turn_id: str, payload: bytes, seq: int = 0, codec: str = "wav", final: bool = True
) -> bytes:
    header = json.dumps(
        {"type": AUDIO_CHUNK_TYPE, "turn_id": turn_id, "seq": seq, "codec": codec, "final": final},
        separators=(",", ":"),
    ).encode("utf-8")
    return _LEN.pack(len(header)) + header + payload


def decode_audio_chunk(frame: bytes) -> tuple[dict, bytes]:
    if len(frame) < _LEN.size:
        raise AudioFrameError("frame shorter than its length prefix")
    (n,) = _LEN.unpack_from(frame)
    end = _LEN.size + n
    if len(frame) < end:
        raise AudioFrameError("header length runs past the end of the frame")
    try:
        header = json.loads(frame[_LEN.size:end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise AudioFrameError(f"bad header: {e}") from e
    if not (
        isinstance(header, dict)
        and header.get("type") == AUDIO_CHUNK_TYPE
        and isinstance(header.get("turn_id"), str)
        and isinstance(header.get("seq"), int) and not isinstance(header.get("seq"), bool)
        and isinstance(header.get("codec"), str)
        and isinstance(header.get("final"), bool)
    ):
        raise AudioFrameError(f"header does not match {AUDIO_CHUNK_TYPE}: {header!r}")
    return header, frame[end:]
