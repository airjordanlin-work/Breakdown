# Breakdown IMU Firmware

Streams motion data from two MPU-6050 IMUs (wrist + leg) over Bluetooth Low
Energy so the backend can keep tracking limbs the camera can't see
(inversions, ground work, arm behind the body).

## Wiring

Each sensor gets its own I2C bus, so both can keep the default address 0x68.

| MPU-6050 pin | Sensor A (wrist) | Sensor B (leg) |
|--------------|------------------|----------------|
| VCC          | 3V3              | 3V3            |
| GND          | GND              | GND            |
| SDA          | GPIO 8           | GPIO 5         |
| SCL          | GPIO 9           | GPIO 6         |

If you wired different pins, change `SDA_A`, `SCL_A`, `SDA_B`, `SCL_B` at the
top of `imu_ble/imu_ble.ino`.

## Arduino IDE setup (one time)

1. Arduino IDE > Settings > Additional boards manager URLs, add
   `https://espressif.github.io/arduino-esp32/package_esp32_index.json`
2. Tools > Board > Boards Manager, install **esp32 by Espressif Systems**.
3. Tools menu settings:
   - Board: **ESP32S3 Dev Module**
   - Flash Size: **16MB**
   - PSRAM: **OPI PSRAM**
   - USB CDC On Boot: **Enabled** (so Serial Monitor works over USB)

No extra libraries needed. BLE and Wire come with the board package.

## Flash it

1. Open `firmware/imu_ble/imu_ble.ino`.
2. Plug the board in. If it has two USB-C ports, try the one labeled
   **USB** first; if upload fails, use the one labeled **COM/UART**.
3. Tools > Port, pick the board.
4. Click Upload. If it hangs on "Connecting...", hold **BOOT**, tap
   **RST**, release BOOT, and upload again.

## Test 1: sensors (Serial Monitor)

Tools > Serial Monitor at **115200 baud**. You should see:

```
Sensor A (wrist): found
Sensor B (leg):   found
Advertising as Breakdown-IMU
BLE:off rate:0Hz A[ok] acc(g) 0.01 -0.02 1.00 gyro(dps) 0.4 ...
```

Pass criteria:
- Both sensors say **found**.
- Lying flat and still, one accel axis reads about **1.00 g** (gravity) and
  gyro values sit near **0**.
- Rotating a sensor makes its gyro values jump, and they settle back near 0
  when you stop.

## Test 2: Bluetooth (phone, no backend needed)

1. Install **nRF Connect** (Nordic Semiconductor) on your phone.
2. Scan, connect to **Breakdown-IMU**.
3. Open the service `07c6216a-...`, find characteristic `4850fc92-...`, tap
   the subscribe (triple-arrow) icon.

Pass criteria:
- Values update continuously (they're raw hex bytes, that's expected).
- Serial Monitor now shows **BLE:on** and **rate:50Hz**.
- Disconnect in the app, then reconnect. It should work without resetting
  the board.

## Packet format (protocol v1)

32 bytes, little-endian, one packet per sample at `SAMPLE_HZ`.

| Offset | Type      | Field   | Notes                                   |
|--------|-----------|---------|-----------------------------------------|
| 0      | uint8     | version | 1                                       |
| 1      | uint8     | flags   | bit0 = sensor A ok, bit1 = sensor B ok  |
| 2      | uint16    | seq     | +1 per packet, gaps mean dropped packets|
| 4      | uint32    | t_ms    | ESP32 uptime in ms                      |
| 8      | int16 x 6 | a       | wrist: ax, ay, az, gx, gy, gz (raw)     |
| 20     | int16 x 6 | b       | leg: ax, ay, az, gx, gy, gz (raw)       |

Unit conversion: accel `raw / 4096` = g (range +/-8 g), gyro `raw / 16.4` =
deg/s (range +/-2000 deg/s).

## Troubleshooting

- **Sensor NOT FOUND**: check SDA/SCL aren't swapped and the pins in the
  sketch match your wiring. Some cheap clones report a different chip ID,
  which this code ignores on purpose.
- **Compile error mentioning BLE2902**: newer board package versions add
  that descriptor automatically. Delete the `addDescriptor(...)` line.
- **Nothing in Serial Monitor**: confirm USB CDC On Boot is Enabled and the
  baud rate is 115200.