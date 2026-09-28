# Balancia Robot — ESP32-S3-Zero Arduino Firmware 🤖

This folder contains the complete Arduino IDE sketch and C/C++ neural network inference engine for the **Waveshare ESP32-S3-Zero** board.

---

## 1. Directory Structure

```text
firmware/balancia_esp32_s3_zero/
├── balancia_esp32_s3_zero.ino   # Main Arduino sketch (100 Hz control loop, IMU filter, Encoders, PWM)
├── policy_net.h                 # Pure C++ neural network weights & microsecond inference engine
└── README.md                    # Pinout, wiring, and Arduino IDE setup guide
```

---

## 2. Hardware Wiring (ESP32-S3-Zero)

| Module | Pin Name | ESP32-S3-Zero GPIO | Notes |
| :--- | :--- | :---: | :--- |
| **MPU6050 (IMU)** | `SDA` | **GPIO 8** | I2C Data (400 kHz Fast Mode) |
| | `SCL` | **GPIO 9** | I2C Clock |
| | `VCC` / `GND` | `3V3` / `GND` | Power |
| **Left Motor (TB6612FNG)** | `PWMA` | **GPIO 1** | Ultrasonic PWM (20 kHz, 8-bit) |
| | `AIN1` | **GPIO 2** | Direction 1 |
| | `AIN2` | **GPIO 3** | Direction 2 |
| **Right Motor (TB6612FNG)**| `PWMB` | **GPIO 4** | Ultrasonic PWM (20 kHz, 8-bit) |
| | `BIN1` | **GPIO 5** | Direction 1 |
| | `BIN2` | **GPIO 6** | Direction 2 |
| **Left Wheel Encoder** | `Phase A` | **GPIO 7** | Interrupt (Rising edge) |
| | `Phase B` | **GPIO 10** | Direction reading |
| **Right Wheel Encoder** | `Phase A` | **GPIO 11** | Interrupt (Rising edge) |
| | `Phase B` | **GPIO 12** | Direction reading |
| **Onboard RGB LED** | `WS2812` | **GPIO 21** | Built-in LED on ESP32-S3-Zero |

---

## 3. How to Export the Latest Policy

Run from `projects/balancia_robot/`:

```bash
python scripts/export_to_esp32.py --model trained_models/ppo_balance_latest.zip
```

This exports `policy_net.h` directly into this folder.

---

## 4. Arduino IDE Setup

1. Open **[balancia_esp32_s3_zero.ino](file:///home/abok-omen/mujoco-rl/projects/balancia_robot/firmware/balancia_esp32_s3_zero/balancia_esp32_s3_zero.ino)** in Arduino IDE.
2. Select Board: **ESP32S3 Dev Module** (or **Waveshare ESP32-S3-Zero**).
3. Board Settings:
   - **USB CDC On Boot**: `Enabled` *(enables Serial Monitor over USB)*
   - **CPU Frequency**: `240MHz (WiFi/BT)`
   - **Flash Mode**: `QIO 80MHz`
   - **Flash Size**: `4MB`
   - **Upload Speed**: `921600`
4. Hold **BOOT** button $\to$ tap **RESET** button $\to$ release **BOOT** to enter flashing mode.
5. Click **Upload**.
