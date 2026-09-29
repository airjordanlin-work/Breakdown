"""BLE bridge: connects to the Breakdown IMU over Bluetooth and decodes its data.

Run from the backend folder with the venv active:

    python scripts/imu_bridge.py                     # live stats
    python scripts/imu_bridge.py --show              # stats + sensor values
    python scripts/imu_bridge.py --record 30         # save 30s to recordings/
    python scripts/imu_bridge.py --forward           # also send data to the backend

The backend never talks Bluetooth directly. This script owns the messy parts
(scanning, connecting, reconnecting, macOS permissions) and later forwards
clean readings to the backend.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import websockets
from bleak import BleakClient, BleakScanner
from bleak.exc import BleakError

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.imu_protocol import (  # noqa: E402
    DEVICE_NAME,
    IMU_CHAR_UUID,
    SERVICE_UUID,
    DropCounter,
    ImuPacket,
    PacketError,
    decode_packet,
    pack_forwarded,
)

RECORDINGS_DIR = _ROOT / "recordings"
DEFAULT_BACKEND_URL = "ws://localhost:8000/ws/imu"
CSV_HEADER = [
    "host_time", "seq", "device_ms",
    "wrist_ok", "w_ax", "w_ay", "w_az", "w_gx", "w_gy", "w_gz",
    "leg_ok", "l_ax", "l_ay", "l_az", "l_gx", "l_gy", "l_gz",
]


def _sample_cols(sample) -> list:
    if sample is None:
        return [0] + [""] * 6
    return [1, *(round(v, 4) for v in sample.accel_g), *(round(v, 2) for v in sample.gyro_dps)]


def _fmt(sample) -> str:
    if sample is None:
        return "--"
    ax, ay, az = sample.accel_g
    gx, gy, gz = sample.gyro_dps
    return f"acc {ax:5.2f} {ay:5.2f} {az:5.2f}  gyro {gx:7.1f} {gy:7.1f} {gz:7.1f}"


class Forwarder:
    """Sends packets to the backend over a WebSocket, reconnecting as needed.

    Packets go through a bounded queue so a slow or offline backend can never
    block the Bluetooth callback. If the queue fills up, the oldest packets
    are dropped, since for live coaching only fresh data matters.
    """

    def __init__(self, url: str, max_queue: int = 100) -> None:
        self.url = url
        self.queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=max_queue)
        self.connected = False
        self.discarded = 0

    def push(self, message: bytes) -> None:
        if self.queue.full():
            self.queue.get_nowait()
            self.discarded += 1
        self.queue.put_nowait(message)

    async def run(self) -> None:
        warned = False
        while True:
            try:
                async with websockets.connect(self.url) as ws:
                    self.connected = True
                    warned = False
                    print(f"Forwarding to {self.url}")
                    while True:
                        await ws.send(await self.queue.get())
            except (OSError, websockets.exceptions.WebSocketException) as e:
                self.connected = False
                if not warned:
                    print(f"Backend not reachable at {self.url} ({e}). Is the server running? Retrying...")
                    warned = True
                await asyncio.sleep(2.0)


class Bridge:
    def __init__(self, show: bool, record_seconds: float | None,
                 forwarder: Forwarder | None = None) -> None:
        self.show = show
        self.forwarder = forwarder
        self.record_seconds = record_seconds
        self.drops = DropCounter()
        self.bad_packets = 0
        self.latest: ImuPacket | None = None
        self._count_this_sec = 0
        self._csv_file = None
        self._writer = None
        self.record_path: Path | None = None
        self.record_started: float | None = None
        self.recorded = 0

    # --- recording -------------------------------------------------------
    def start_recording(self) -> None:
        RECORDINGS_DIR.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.record_path = RECORDINGS_DIR / f"imu_{stamp}.csv"
        self._csv_file = self.record_path.open("w", newline="")
        self._writer = csv.writer(self._csv_file)
        self._writer.writerow(CSV_HEADER)
        self.record_started = time.time()

    def stop_recording(self) -> None:
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None
            print(f"\nSaved {self.recorded} samples to {self.record_path}")

    def recording_done(self) -> bool:
        return (
            self.record_seconds is not None
            and self.record_started is not None
            and time.time() - self.record_started >= self.record_seconds
        )

    # --- BLE callback, runs for every packet -----------------------------
    def on_packet(self, _sender, data: bytearray) -> None:
        host_time = time.time()  # laptop clock, used later to align with camera frames
        try:
            pkt = decode_packet(bytes(data))
        except PacketError:
            self.bad_packets += 1
            return

        self.drops.update(pkt.seq)
        self.latest = pkt
        if self.forwarder:
            self.forwarder.push(pack_forwarded(host_time, bytes(data)))
        self._count_this_sec += 1

        if self._writer and not self.recording_done():
            self._writer.writerow(
                [f"{host_time:.4f}", pkt.seq, pkt.device_ms,
                 *_sample_cols(pkt.wrist), *_sample_cols(pkt.leg)]
            )
            self.recorded += 1

    # --- once-per-second status line -------------------------------------
    def print_stats(self) -> None:
        rate = self._count_this_sec
        self._count_this_sec = 0
        pkt = self.latest
        wrist = "OK" if pkt and pkt.wrist else "--"
        leg = "OK" if pkt and pkt.leg else "--"
        line = (f"{rate:3d} pkt/s  dropped {self.drops.dropped} "
                f"({self.drops.drop_rate:.1%})  wrist {wrist}  leg {leg}")
        if self.bad_packets:
            line += f"  bad {self.bad_packets}"
        if self._writer:
            line += f"  REC {self.recorded}"
        if self.forwarder:
            line += "  fwd " + ("OK" if self.forwarder.connected else "--")
        print(line)
        if self.show and pkt:
            print(f"    wrist {_fmt(pkt.wrist)}")
            print(f"    leg   {_fmt(pkt.leg)}")


async def find_device(timeout: float = 10.0):
    """Find the ESP32 by its service UUID, falling back to its name."""
    def match(device, adv) -> bool:
        uuids = [u.lower() for u in (adv.service_uuids or [])]
        return SERVICE_UUID in uuids or device.name == DEVICE_NAME

    return await BleakScanner.find_device_by_filter(match, timeout=timeout)


async def run(bridge: Bridge) -> None:
    if bridge.forwarder:
        asyncio.create_task(bridge.forwarder.run())
    while True:
        print(f"Scanning for {DEVICE_NAME}...")
        device = await find_device()
        if device is None:
            print("Not found. Is it powered on, and disconnected from your phone? Retrying.")
            continue

        disconnected = asyncio.Event()
        try:
            async with BleakClient(device, disconnected_callback=lambda _c: disconnected.set()) as client:
                await client.start_notify(IMU_CHAR_UUID, bridge.on_packet)
                print(f"Connected to {device.name or device.address}")
                if bridge.record_seconds is not None and bridge.record_started is None:
                    bridge.start_recording()
                    print(f"Recording {bridge.record_seconds:.0f}s to {bridge.record_path}")

                while not disconnected.is_set():
                    await asyncio.sleep(1.0)
                    bridge.print_stats()
                    if bridge.recording_done():
                        return
        except BleakError as e:
            print(f"Bluetooth error: {e}")

        print("Disconnected, reconnecting in 2s...")
        await asyncio.sleep(2.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Breakdown IMU BLE bridge")
    parser.add_argument("--show", action="store_true", help="print sensor values each second")
    parser.add_argument("--record", type=float, metavar="SECONDS",
                        help="save this many seconds of data to backend/recordings/")
    parser.add_argument("--forward", nargs="?", const=DEFAULT_BACKEND_URL, metavar="URL",
                        help=f"send data to the backend (default {DEFAULT_BACKEND_URL})")
    args = parser.parse_args()

    forwarder = Forwarder(args.forward) if args.forward else None
    bridge = Bridge(show=args.show, record_seconds=args.record, forwarder=forwarder)
    try:
        asyncio.run(run(bridge))
    except KeyboardInterrupt:
        pass
    finally:
        bridge.stop_recording()
        print(f"Total received {bridge.drops.received}, dropped {bridge.drops.dropped}")


if __name__ == "__main__":
    main()