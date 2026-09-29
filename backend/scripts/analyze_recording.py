"""Replay a recorded IMU session and report every freeze it finds.

Use this to tune the thresholds in app/imu_metrics.py against real dancing
instead of guessing:

    python scripts/imu_bridge.py --record 60      # record some freezes
    python scripts/analyze_recording.py recordings/imu_<date>.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.imu_metrics import FreezeTracker  # noqa: E402
from app.imu_protocol import ImuPacket, ImuSample  # noqa: E402
from app.imu_store import TimedImuSample  # noqa: E402


def _sample(row: dict, prefix: str) -> ImuSample | None:
    if row[f"{prefix}_ok"] != "1":
        return None
    p = prefix[0]  # column prefix: "w" for wrist, "l" for leg
    f = lambda k: float(row[f"{p}_{k}"])
    return ImuSample(accel_g=(f("ax"), f("ay"), f("az")), gyro_dps=(f("gx"), f("gy"), f("gz")))


def load(path: Path) -> list[TimedImuSample]:
    with path.open(newline="") as fh:
        return [
            TimedImuSample(
                float(row["host_time"]),
                ImuPacket(int(row["seq"]), int(row["device_ms"]),
                          _sample(row, "wrist"), _sample(row, "leg")),
            )
            for row in csv.DictReader(fh)
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Find freezes in a recorded IMU session")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--still", type=float, help="override STILL_DPS")
    parser.add_argument("--min-hold", type=float, help="override MIN_HOLD_S")
    args = parser.parse_args()

    samples = load(args.csv)
    if not samples:
        print("No samples in file.")
        return

    kwargs = {}
    if args.still is not None:
        kwargs["still_dps"] = args.still
    if args.min_hold is not None:
        kwargs["min_hold_s"] = args.min_hold
    tracker = FreezeTracker(**kwargs)

    t0 = samples[0].host_time
    found = 0
    for s in samples:
        r = tracker.update([s])
        if r:
            found += 1
            print(f"{r.start_time - t0:6.1f}s  held {r.duration_s:4.1f}s  "
                  f"{r.tier:10s}  stability {r.stability:3d}  "
                  f"wobble {r.wobble_dps:5.1f} deg/s  shakiest {r.shakiest}")
    span = samples[-1].host_time - t0
    print(f"\n{found} freeze(s) in {span:.0f}s of data ({len(samples)} samples)")


if __name__ == "__main__":
    main()