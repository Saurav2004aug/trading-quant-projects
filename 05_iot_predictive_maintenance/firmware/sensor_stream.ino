/*
  sensor_stream.ino  --  ESP32 vibration + temperature node

  Every PUBLISH_INTERVAL_MS the node:
    1. captures a burst of N_SAMPLES accelerometer readings at ~1 kHz
       (MPU6050 accel output rate is 1 kHz; digital low-pass at 260 Hz
       acts as the anti-aliasing filter below the 500 Hz Nyquist limit),
    2. removes the per-axis mean of the burst (gravity + sensor offset),
       so orientation does not matter,
    3. computes vibration features of the remaining AC signal:
         RMS (g), peak (g), crest factor = peak / RMS, kurtosis
       (crest factor and kurtosis rise early with impulsive bearing faults,
       before RMS moves),
    4. reads a DS18B20 probe without blocking (conversion started one
       interval earlier), rejecting the -127 C "disconnected" value,
    5. publishes one JSON message over MQTT.

  Why not 1 Hz sampling: machine vibration lives at tens to hundreds of Hz.
  Sampling once per second aliases all of it and measures noise.
  Limits: a 1 kHz accelerometer sees shaft-rate and low-order harmonics
  (<~250 Hz). Bearing defect frequencies in the kHz range need a
  higher-bandwidth sensor (e.g. ADXL1002 + ADC, or a piezo accelerometer).

  Libraries: Adafruit_MPU6050, Adafruit_Sensor, OneWire, DallasTemperature,
             PubSubClient (all from the Arduino Library Manager).
*/

#include <Wire.h>
#include <WiFi.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <PubSubClient.h>
#include <math.h>

// ---------- configuration ----------
const char* WIFI_SSID     = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
const char* MQTT_BROKER   = "192.168.1.10";
const uint16_t MQTT_PORT  = 1883;
const char* MACHINE_ID    = "machine_01";
const char* MQTT_TOPIC    = "plant/machine_01/sensors";

const int ONE_WIRE_PIN               = 4;
const uint16_t N_SAMPLES             = 1024;    // ~1.0 s burst at 1 kHz
const uint32_t SAMPLE_PERIOD_US      = 1000;    // 1 kHz
const uint32_t PUBLISH_INTERVAL_MS   = 5000;
const float    G                     = 9.80665f;

// ---------- globals ----------
Adafruit_MPU6050 mpu;
OneWire oneWire(ONE_WIRE_PIN);
DallasTemperature tempSensor(&oneWire);
WiFiClient net;
PubSubClient mqtt(net);

float ax[N_SAMPLES], ay[N_SAMPLES], az[N_SAMPLES];   // 12 KB, fits easily in ESP32 RAM
uint32_t seq = 0;
uint32_t lastPublish = 0;
float lastTempC = NAN;

struct VibFeatures { float rms_g, peak_g, crest, kurtosis; uint16_t n; float fs_hz; };

// ---------- connectivity ----------
void ensureWiFi() {
  if (WiFi.status() == WL_CONNECTED) return;
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  uint32_t t0 = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - t0 < 15000) delay(250);
}

void ensureMqtt() {
  if (mqtt.connected() || WiFi.status() != WL_CONNECTED) return;
  String id = String(MACHINE_ID) + "-" + String((uint32_t)ESP.getEfuseMac(), HEX);
  mqtt.connect(id.c_str());          // retried on the next loop if it fails; never blocks forever
}

// ---------- sensing ----------
VibFeatures captureVibration() {
  sensors_event_t a, g, t;
  uint32_t start = micros(), next = start;
  for (uint16_t i = 0; i < N_SAMPLES; i++) {
    while ((int32_t)(micros() - next) < 0) { }      // pace to SAMPLE_PERIOD_US
    next += SAMPLE_PERIOD_US;
    mpu.getEvent(&a, &g, &t);
    ax[i] = a.acceleration.x; ay[i] = a.acceleration.y; az[i] = a.acceleration.z;
  }
  float fs = N_SAMPLES * 1e6f / (float)(micros() - start);   // measured, not assumed

  double mx = 0, my = 0, mz = 0;
  for (uint16_t i = 0; i < N_SAMPLES; i++) { mx += ax[i]; my += ay[i]; mz += az[i]; }
  mx /= N_SAMPLES; my /= N_SAMPLES; mz /= N_SAMPLES;          // DC = gravity + offset

  double s2 = 0, s4 = 0, peak = 0;
  for (uint16_t i = 0; i < N_SAMPLES; i++) {
    double x = ax[i] - mx, y = ay[i] - my, z = az[i] - mz;
    double m2 = x * x + y * y + z * z;                        // squared AC magnitude
    s2 += m2; s4 += m2 * m2;
    if (m2 > peak) peak = m2;
  }
  double ms = s2 / N_SAMPLES;
  VibFeatures f;
  f.rms_g    = sqrt(ms) / G;
  f.peak_g   = sqrt(peak) / G;
  f.crest    = ms > 0 ? sqrt(peak / ms) : 0;
  f.kurtosis = ms > 0 ? (s4 / N_SAMPLES) / (ms * ms) : 0;    // of the magnitude signal
  f.n = N_SAMPLES; f.fs_hz = fs;
  return f;
}

float readTemperatureNonBlocking() {
  // Read the conversion started during the previous interval, then start the next one.
  float c = tempSensor.getTempCByIndex(0);
  tempSensor.requestTemperatures();
  if (c == DEVICE_DISCONNECTED_C || c < -55 || c > 125) return NAN;
  return c;
}

// ---------- publish ----------
void publish(const VibFeatures& v, float tempC) {
  char payload[256];
  char temp[16];
  if (isnan(tempC)) strcpy(temp, "null"); else snprintf(temp, sizeof(temp), "%.2f", tempC);
  snprintf(payload, sizeof(payload),
           "{\"machine_id\":\"%s\",\"seq\":%lu,\"uptime_ms\":%lu,\"vibration_g\":%.4f,"
           "\"peak_g\":%.4f,\"crest\":%.2f,\"kurtosis\":%.2f,\"fs_hz\":%.0f,"
           "\"n\":%u,\"temperature_c\":%s}",
           MACHINE_ID, (unsigned long)seq, (unsigned long)millis(), v.rms_g, v.peak_g,
           v.crest, v.kurtosis, v.fs_hz, v.n, temp);
  if (mqtt.connected()) mqtt.publish(MQTT_TOPIC, payload);
  Serial.println(payload);          // always log locally, even if offline
  seq++;
}

void setup() {
  Serial.begin(115200);
  Wire.begin();
  Wire.setClock(400000);            // fast-mode I2C is needed to sustain 1 kHz reads
  if (!mpu.begin()) {
    Serial.println("MPU6050 not found - check wiring");
    while (true) delay(1000);
  }
  mpu.setAccelerometerRange(MPU6050_RANGE_4_G);
  mpu.setFilterBandwidth(MPU6050_BAND_260_HZ);

  tempSensor.begin();
  tempSensor.setWaitForConversion(false);
  tempSensor.requestTemperatures();

  mqtt.setServer(MQTT_BROKER, MQTT_PORT);
  mqtt.setBufferSize(512);
  ensureWiFi();
}

void loop() {
  ensureWiFi();
  ensureMqtt();
  mqtt.loop();
  if (millis() - lastPublish >= PUBLISH_INTERVAL_MS) {
    lastPublish = millis();
    VibFeatures v = captureVibration();
    lastTempC = readTemperatureNonBlocking();
    publish(v, lastTempC);
  }
}
