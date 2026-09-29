"""Tests for ImuStore and the bridge forwarding format. No hardware needed."""

import pytest

from app.imu_protocol import (
    PacketError,
    decode_packet,
    encode_packet,
    pack_forwarded,
    unpack_forwarded,
)
from app.imu_store import ImuStore


def make_packet(seq):
    return decode_packet(encode_packet(seq, seq * 20, [0, 0, 4096, 0, 0, 0], [0] * 6))


def filled_store(start=100.0, n=50, dt=0.02):
    store = ImuStore()
    for i in range(n):
        store.add(start + i * dt, make_packet(i))
    return store


def test_forwarded_round_trip():
    raw = encode_packet(7, 140, [1] * 6, None)
    host_time, pkt = unpack_forwarded(pack_forwarded(123.456, raw))
    assert host_time == pytest.approx(123.456)
    assert pkt.seq == 7
    assert pkt.leg is None


def test_forwarded_rejects_bad_length():
    with pytest.raises(PacketError):
        unpack_forwarded(b"\x00" * 39)


def test_nearest_picks_closest_sample():
    store = filled_store()
    sample = store.nearest(100.031)       # between 100.02 and 100.04
    assert sample.packet.seq == 2         # 100.04 is closer


def test_nearest_returns_none_when_stream_stalled():
    store = filled_store()
    assert store.nearest(105.0) is None   # seconds after the last sample


def test_window_returns_range_in_order():
    store = filled_store()
    seqs = [s.packet.seq for s in store.window(100.1, 100.2)]
    assert seqs == list(range(5, 11))


def test_old_samples_roll_off():
    store = ImuStore(seconds=1, rate_hz=50)   # holds 50 samples
    for i in range(120):
        store.add(100 + i * 0.02, make_packet(i))
    assert store.window(0, 1e9)[0].packet.seq == 70


def test_out_of_order_sample_ignored():
    store = filled_store(n=3)
    store.add(99.0, make_packet(99))
    assert store.latest().packet.seq == 2


def test_status_reports_connection_and_rate():
    store = filled_store()                       # 100.00 .. 100.98
    live = store.status(now=101.0)
    assert live["connected"] is True
    assert live["rate_hz"] == 50
    assert store.status(now=110.0)["connected"] is False