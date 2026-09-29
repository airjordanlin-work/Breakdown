"""Decode packets sent by the Breakdown IMU firmware (protocol v1).

Kept separate from the Bluetooth code on purpose: this module has no
hardware dependencies, so it can be unit tested in CI and reused later to
replay recorded sessions into the backend.

Packet layout (32 bytes, little-endian), mirrors ImuPacket in
firmware/imu_ble/imu_ble.ino:

    uint8   version
    uint8   flags        bit0 = wrist ok, bit1 = leg ok
    uint16  seq          +1 per packet, wraps at 65535
    uint32  device_ms    ESP32 uptime when sampled
    int16x6 wrist raw    ax, ay, az, gx, gy, gz
    int16x6 leg raw      ax, ay, az, gx, gy, gz
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Optional, Sequence

SERVICE_UUID = "07c6216a-c07d-4d82-8d60-0f2edfedffff"
IMU_CHAR_UUID = "4850fc92-218d-422a-913f-af8292387c2a"
DEVICE_NAME = "Breakdown-IMU"

PROTOCOL_VERSION = 1
PACKET_SIZE = 32
_STRUCT = struct.Struct("<BBHI6h6h")
assert _STRUCT.size == PACKET_SIZE

# Must match the ranges set in the firmware's mpuInit()
ACCEL_LSB_PER_G = 4096.0   # +/-8 g
GYRO_LSB_PER_DPS = 16.4    # +/-2000 deg/s

FLAG_WRIST = 0x01
FLAG_LEG = 0x02


class PacketError(ValueError):
    """Raised when bytes can't be decoded as a valid packet."""


@dataclass(frozen=True)
class ImuSample:
    """One sensor reading in physical units."""
    accel_g: tuple[float, float, float]
    gyro_dps: tuple[float, float, float]


@dataclass(frozen=True)
class ImuPacket:
    seq: int
    device_ms: int
    wrist: Optional[ImuSample]  # None if the wrist sensor wasn't read
    leg: Optional[ImuSample]    # None if the leg sensor wasn't read


def _to_sample(raw: Sequence[int]) -> ImuSample:
    ax, ay, az, gx, gy, gz = raw
    return ImuSample(
        accel_g=(ax / ACCEL_LSB_PER_G, ay / ACCEL_LSB_PER_G, az / ACCEL_LSB_PER_G),
        gyro_dps=(gx / GYRO_LSB_PER_DPS, gy / GYRO_LSB_PER_DPS, gz / GYRO_LSB_PER_DPS),
    )


def decode_packet(data: bytes) -> ImuPacket:
    """Turn 32 raw bytes from the firmware into an :class:`ImuPacket`."""
    if len(data) != PACKET_SIZE:
        raise PacketError(f"expected {PACKET_SIZE} bytes, got {len(data)}")

    version, flags, seq, device_ms, *raw = _STRUCT.unpack(data)
    if version != PROTOCOL_VERSION:
        raise PacketError(f"unsupported protocol version {version}")

    wrist_raw, leg_raw = raw[:6], raw[6:]
    return ImuPacket(
        seq=seq,
        device_ms=device_ms,
        wrist=_to_sample(wrist_raw) if flags & FLAG_WRIST else None,
        leg=_to_sample(leg_raw) if flags & FLAG_LEG else None,
    )


def encode_packet(
    seq: int,
    device_ms: int,
    wrist_raw: Optional[Sequence[int]],
    leg_raw: Optional[Sequence[int]],
    version: int = PROTOCOL_VERSION,
) -> bytes:
    """Build a packet the same way the firmware does. Used by tests and replay."""
    flags = (FLAG_WRIST if wrist_raw is not None else 0) | (FLAG_LEG if leg_raw is not None else 0)
    return _STRUCT.pack(
        version, flags, seq & 0xFFFF, device_ms & 0xFFFFFFFF,
        *(wrist_raw or (0,) * 6), *(leg_raw or (0,) * 6),
    )


class DropCounter:
    """Counts lost packets from gaps in the sequence number.

    seq is 16-bit, so it wraps from 65535 back to 0; that's not a drop.
    A huge jump usually means the ESP32 rebooted and restarted at 0, so
    it's treated as a reset rather than thousands of drops.
    """

    RESET_GAP = 500  # 10 seconds at 50Hz

    def __init__(self) -> None:
        self.last_seq: Optional[int] = None
        self.received = 0
        self.dropped = 0
        self.resets = 0

    def update(self, seq: int) -> None:
        self.received += 1
        if self.last_seq is not None:
            gap = (seq - self.last_seq) & 0xFFFF
            if gap > self.RESET_GAP:
                self.resets += 1
            elif gap > 1:
                self.dropped += gap - 1
        self.last_seq = seq

    @property
    def drop_rate(self) -> float:
        total = self.received + self.dropped
        return self.dropped / total if total else 0.0


# ---------------------------------------------------------------------------
# Bridge -> backend forwarding format
#
# The bridge stamps each packet with the laptop's clock when it arrives, then
# forwards [8-byte float host_time][32-byte firmware packet] over a WebSocket.
# The backend and the browser read the same laptop clock, so IMU samples and
# camera frames can be lined up directly by timestamp.
# ---------------------------------------------------------------------------
_FWD_HEADER = struct.Struct("<d")
FORWARDED_SIZE = _FWD_HEADER.size + PACKET_SIZE  # 40 bytes


def pack_forwarded(host_time: float, packet: bytes) -> bytes:
    if len(packet) != PACKET_SIZE:
        raise PacketError(f"expected {PACKET_SIZE}-byte packet, got {len(packet)}")
    return _FWD_HEADER.pack(host_time) + packet


def unpack_forwarded(message: bytes) -> tuple[float, ImuPacket]:
    if len(message) != FORWARDED_SIZE:
        raise PacketError(f"expected {FORWARDED_SIZE}-byte message, got {len(message)}")
    (host_time,) = _FWD_HEADER.unpack_from(message)
    return host_time, decode_packet(message[_FWD_HEADER.size:])