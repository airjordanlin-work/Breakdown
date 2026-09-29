"""Tests for the IMU packet decoder. No hardware or Bluetooth needed."""

import pytest

from app.imu_protocol import (
    PACKET_SIZE,
    DropCounter,
    PacketError,
    decode_packet,
    encode_packet,
)

ONE_G = 4096        # raw accel value for 1 g
TEN_DPS = 164       # raw gyro value for 10 deg/s


def test_packet_is_32_bytes():
    assert len(encode_packet(0, 0, [0] * 6, [0] * 6)) == PACKET_SIZE


def test_round_trip_converts_units():
    pkt = decode_packet(encode_packet(7, 1234, [0, 0, ONE_G, TEN_DPS, 0, 0],
                                      [ONE_G, 0, 0, 0, 0, -TEN_DPS]))
    assert pkt.seq == 7
    assert pkt.device_ms == 1234
    assert pkt.wrist.accel_g == pytest.approx((0.0, 0.0, 1.0))
    assert pkt.wrist.gyro_dps == pytest.approx((10.0, 0.0, 0.0))
    assert pkt.leg.accel_g == pytest.approx((1.0, 0.0, 0.0))
    assert pkt.leg.gyro_dps == pytest.approx((0.0, 0.0, -10.0))


def test_missing_sensor_decodes_as_none():
    pkt = decode_packet(encode_packet(1, 0, [0] * 6, None))
    assert pkt.wrist is not None
    assert pkt.leg is None


def test_wrong_length_rejected():
    with pytest.raises(PacketError):
        decode_packet(b"\x00" * 20)


def test_wrong_version_rejected():
    with pytest.raises(PacketError):
        decode_packet(encode_packet(0, 0, [0] * 6, [0] * 6, version=2))


def test_drop_counter_counts_gaps():
    c = DropCounter()
    for seq in [0, 1, 2, 5, 6]:
        c.update(seq)
    assert c.received == 5
    assert c.dropped == 2


def test_drop_counter_handles_wraparound():
    c = DropCounter()
    for seq in [65534, 65535, 0, 1]:
        c.update(seq)
    assert c.dropped == 0


def test_drop_counter_treats_reboot_as_reset():
    c = DropCounter()
    for seq in [3000, 3001, 0, 1]:   # device rebooted, seq restarted
        c.update(seq)
    assert c.dropped == 0
    assert c.resets == 1