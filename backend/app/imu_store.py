"""Thread-safe rolling store of recent IMU samples, keyed by laptop time.

The IMU WebSocket writes into it at ~50Hz, and camera frame processing
(which runs in worker threads) reads from it to find the IMU sample closest
to each frame's capture time. One store is shared by the whole server since
there's one physical device.
"""

from __future__ import annotations

import bisect
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

from app.imu_protocol import DropCounter, ImuPacket

DEFAULT_SECONDS = 10.0
DEFAULT_RATE_HZ = 50


@dataclass(frozen=True)
class TimedImuSample:
    host_time: float
    packet: ImuPacket


class ImuStore:
    def __init__(self, seconds: float = DEFAULT_SECONDS, rate_hz: int = DEFAULT_RATE_HZ) -> None:
        self._samples: deque[TimedImuSample] = deque(maxlen=int(seconds * rate_hz))
        self._lock = threading.Lock()
        self.drops = DropCounter()

    def add(self, host_time: float, packet: ImuPacket) -> None:
        with self._lock:
            # Keep samples sorted by time. They normally arrive in order,
            # so this is just an append.
            if self._samples and host_time < self._samples[-1].host_time:
                return
            self._samples.append(TimedImuSample(host_time, packet))
            self.drops.update(packet.seq)

    def latest(self) -> Optional[TimedImuSample]:
        with self._lock:
            return self._samples[-1] if self._samples else None

    def nearest(self, t: float, max_gap: float = 0.05) -> Optional[TimedImuSample]:
        """Sample closest to time t, or None if nothing is within max_gap seconds.

        50ms default: at 50Hz a sample arrives every 20ms, so anything farther
        away means the IMU stream stalled and the data shouldn't be trusted.
        """
        with self._lock:
            if not self._samples:
                return None
            times = [s.host_time for s in self._samples]
            i = bisect.bisect_left(times, t)
            candidates = [self._samples[j] for j in (i - 1, i) if 0 <= j < len(times)]
        best = min(candidates, key=lambda s: abs(s.host_time - t))
        return best if abs(best.host_time - t) <= max_gap else None

    def window(self, t0: float, t1: float) -> list[TimedImuSample]:
        """All samples with t0 <= host_time <= t1, oldest first."""
        with self._lock:
            return [s for s in self._samples if t0 <= s.host_time <= t1]

    def status(self, now: Optional[float] = None, stale_after: float = 0.5) -> dict:
        now = time.time() if now is None else now
        with self._lock:
            last = self._samples[-1] if self._samples else None
            recent = [s for s in self._samples if now - s.host_time <= 1.0]
        return {
            "connected": last is not None and now - last.host_time <= stale_after,
            "rate_hz": len(recent),
            "dropped": self.drops.dropped,
            "wrist_ok": bool(last and last.packet.wrist),
            "leg_ok": bool(last and last.packet.leg),
        }