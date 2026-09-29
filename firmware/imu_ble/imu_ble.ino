/*
 * Breakdown IMU firmware
 *
 * Reads two MPU-6050 IMUs (wrist + leg), each on its own I2C bus, and streams
 * the raw readings over Bluetooth Low Energy (BLE) as small binary packets.
 *
 * Board:  ESP32-S3 (N16R8), Arduino IDE, "esp32 by Espressif" board package
 * Output: BLE notifications on IMU_CHAR_UUID, one 32-byte packet per sample
 *
 * Why BLE: a wearable needs to work untethered, anywhere in the room, on a
 * battery. BLE connects directly to the laptop (no router needed) and uses
 * far less power than Wi-Fi.
 */

#include <Arduino.h>
#include <string.h>
#include <Wire.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

// ------------------------------------------------------------------
// Config: change these to match your wiring and needs
// ------------------------------------------------------------------
#define DEVICE_NAME "Breakdown-IMU"

// How many samples per second to send. 50 is plenty to start; the camera
// runs well below this. Raise to 100 only if the bridge reports 0 drops.
static const uint32_t SAMPLE_HZ = 50;

// Print human-readable values to the Serial Monitor (about 5x per second).
static const bool DEBUG_SERIAL = true;

// Sensor A (wrist) on I2C bus 0. CHANGE to the pins you actually wired.
static const int SDA_A = 8;
static const int SCL_A = 9;

// Sensor B (leg) on I2C bus 1. CHANGE to the pins you actually wired.
static const int SDA_B = 5;
static const int SCL_B = 6;

static const uint32_t I2C_HZ   = 400000;  // 400kHz "fast mode"
static const uint8_t  MPU_ADDR = 0x68;    // AD0 pin low (default on GY-521 boards)

// Custom 128-bit UUIDs so the bridge can find this exact device and data stream.
#define SERVICE_UUID  "07c6216a-c07d-4d82-8d60-0f2edfedffff"
#define IMU_CHAR_UUID "4850fc92-218d-422a-913f-af8292387c2a"

// ------------------------------------------------------------------
// Packet format (little-endian, 32 bytes). Keep in sync with the bridge.
// ------------------------------------------------------------------
struct __attribute__((packed)) ImuPacket {
  uint8_t  version;  // protocol version, bump if the layout changes
  uint8_t  flags;    // bit0 = sensor A ok, bit1 = sensor B ok
  uint16_t seq;      // increments every packet; gaps = dropped packets
  uint32_t t_ms;     // ESP32 uptime in ms when the sample was taken
  int16_t  a[6];     // sensor A raw: ax, ay, az, gx, gy, gz
  int16_t  b[6];     // sensor B raw: ax, ay, az, gx, gy, gz
};
static_assert(sizeof(ImuPacket) == 32, "ImuPacket must be 32 bytes");

static const uint8_t PROTOCOL_VERSION = 1;

// Raw -> physical units for the ranges configured in mpuInit()
static const float ACCEL_LSB_PER_G   = 4096.0f;  // +/-8 g
static const float GYRO_LSB_PER_DPS  = 16.4f;    // +/-2000 deg/s

// ------------------------------------------------------------------
// MPU-6050 helpers (raw register access, works on any TwoWire bus)
// ------------------------------------------------------------------
static bool mpuWrite(TwoWire &bus, uint8_t reg, uint8_t val) {
  bus.beginTransmission(MPU_ADDR);
  bus.write(reg);
  bus.write(val);
  return bus.endTransmission() == 0;
}

static bool mpuInit(TwoWire &bus) {
  if (!mpuWrite(bus, 0x6B, 0x01)) return false;  // wake up, use gyro clock
  delay(10);
  mpuWrite(bus, 0x1A, 0x03);   // low-pass filter ~44Hz, cuts vibration noise
  mpuWrite(bus, 0x19, 0x04);   // internal sample rate 200Hz
  mpuWrite(bus, 0x1B, 0x18);   // gyro +/-2000 deg/s (spins are fast)
  return mpuWrite(bus, 0x1C, 0x10);  // accel +/-8 g (impacts on landings)
}

// Reads accel + gyro in one burst so all 6 values come from the same instant.
static bool mpuRead(TwoWire &bus, int16_t out[6]) {
  bus.beginTransmission(MPU_ADDR);
  bus.write(0x3B);  // ACCEL_XOUT_H, start of the data block
  if (bus.endTransmission(false) != 0) return false;
  if (bus.requestFrom((uint16_t)MPU_ADDR, (size_t)14, true) != 14) return false;

  uint8_t r[14];
  for (int i = 0; i < 14; i++) r[i] = bus.read();

  out[0] = (int16_t)((r[0]  << 8) | r[1]);   // ax
  out[1] = (int16_t)((r[2]  << 8) | r[3]);   // ay
  out[2] = (int16_t)((r[4]  << 8) | r[5]);   // az
  // r[6], r[7] are temperature, skipped
  out[3] = (int16_t)((r[8]  << 8) | r[9]);   // gx
  out[4] = (int16_t)((r[10] << 8) | r[11]);  // gy
  out[5] = (int16_t)((r[12] << 8) | r[13]);  // gz
  return true;
}

// ------------------------------------------------------------------
// BLE
// ------------------------------------------------------------------
static BLECharacteristic *imuChar = nullptr;
static volatile bool deviceConnected = false;

class ServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer *) override { deviceConnected = true; }
  void onDisconnect(BLEServer *) override {
    deviceConnected = false;
    BLEDevice::startAdvertising();  // let the bridge reconnect automatically
  }
};

static void setupBle() {
  BLEDevice::init(DEVICE_NAME);
  BLEDevice::setMTU(185);  // our packet is 32 bytes; default BLE payload is only 20

  BLEServer *server = BLEDevice::createServer();
  server->setCallbacks(new ServerCallbacks());

  BLEService *svc = server->createService(SERVICE_UUID);
  imuChar = svc->createCharacteristic(
      IMU_CHAR_UUID,
      BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY);
  imuChar->addDescriptor(new BLE2902());  // lets clients subscribe to notifications
  svc->start();

  BLEAdvertising *adv = BLEDevice::getAdvertising();
  adv->addServiceUUID(SERVICE_UUID);
  adv->setScanResponse(true);
  BLEDevice::startAdvertising();
}

// ------------------------------------------------------------------
// Main loop state
// ------------------------------------------------------------------
static bool     okA = false, okB = false;
static uint16_t seq = 0;
static uint32_t nextSampleUs = 0;
static uint32_t lastRetryMs  = 0;
static uint32_t sentThisSec  = 0, lastRateMs = 0, measuredHz = 0;

static void printSample(const ImuPacket &p) {
  Serial.printf(
      "BLE:%s  rate:%luHz  "
      "A[%s] acc(g) %5.2f %5.2f %5.2f  gyro(dps) %7.1f %7.1f %7.1f  |  "
      "B[%s] acc(g) %5.2f %5.2f %5.2f  gyro(dps) %7.1f %7.1f %7.1f\n",
      deviceConnected ? "on " : "off", (unsigned long)measuredHz,
      okA ? "ok" : "--",
      p.a[0] / ACCEL_LSB_PER_G, p.a[1] / ACCEL_LSB_PER_G, p.a[2] / ACCEL_LSB_PER_G,
      p.a[3] / GYRO_LSB_PER_DPS, p.a[4] / GYRO_LSB_PER_DPS, p.a[5] / GYRO_LSB_PER_DPS,
      okB ? "ok" : "--",
      p.b[0] / ACCEL_LSB_PER_G, p.b[1] / ACCEL_LSB_PER_G, p.b[2] / ACCEL_LSB_PER_G,
      p.b[3] / GYRO_LSB_PER_DPS, p.b[4] / GYRO_LSB_PER_DPS, p.b[5] / GYRO_LSB_PER_DPS);
}

void setup() {
  Serial.begin(115200);
  delay(500);

  Wire.begin(SDA_A, SCL_A, I2C_HZ);
  Wire1.begin(SDA_B, SCL_B, I2C_HZ);
  okA = mpuInit(Wire);
  okB = mpuInit(Wire1);
  Serial.printf("Sensor A (wrist): %s\n", okA ? "found" : "NOT FOUND, check wiring/pins");
  Serial.printf("Sensor B (leg):   %s\n", okB ? "found" : "NOT FOUND, check wiring/pins");

  setupBle();
  Serial.println("Advertising as " DEVICE_NAME);

  nextSampleUs = micros();
  lastRateMs   = millis();
}

void loop() {
  // If a sensor dropped out (loose wire during a spin), keep trying to recover.
  if ((!okA || !okB) && millis() - lastRetryMs > 1000) {
    lastRetryMs = millis();
    if (!okA) okA = mpuInit(Wire);
    if (!okB) okB = mpuInit(Wire1);
  }

  // Fixed-rate schedule: sample exactly SAMPLE_HZ times per second.
  const uint32_t periodUs = 1000000UL / SAMPLE_HZ;
  if ((int32_t)(micros() - nextSampleUs) < 0) return;
  nextSampleUs += periodUs;

  ImuPacket p = {};
  p.version = PROTOCOL_VERSION;
  p.seq     = seq++;
  p.t_ms    = millis();

  // Read into aligned locals, then copy into the packed struct.
  int16_t a[6] = {0}, b[6] = {0};
  if (okA) okA = mpuRead(Wire, a);
  if (okB) okB = mpuRead(Wire1, b);
  memcpy(p.a, a, sizeof(a));
  memcpy(p.b, b, sizeof(b));
  p.flags = (okA ? 0x01 : 0) | (okB ? 0x02 : 0);

  if (deviceConnected) {
    imuChar->setValue((uint8_t *)&p, sizeof(p));
    imuChar->notify();
    sentThisSec++;
  }

  if (millis() - lastRateMs >= 1000) {
    measuredHz  = sentThisSec;
    sentThisSec = 0;
    lastRateMs  = millis();
  }

  if (DEBUG_SERIAL && (p.seq % (SAMPLE_HZ / 5 ? SAMPLE_HZ / 5 : 1)) == 0) {
    printSample(p);
  }
}
