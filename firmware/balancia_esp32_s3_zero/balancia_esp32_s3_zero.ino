/**
 * ============================================================================
 * Balancia Robot — ESP32-S3-Zero Microcontroller Firmware
 * ============================================================================
 * 
 * Hardware Target: Waveshare ESP32-S3-Zero
 * Architecture   : Dual-Core Xtensa LX7 @ 240 MHz
 * Framework      : Arduino IDE / ESP-IDF (ESP32 core >= v2.0.0)
 * Control Loop   : 100 Hz (10.0 ms deterministic period)
 * 
 * Pinout Map (ESP32-S3-Zero):
 *   - I2C IMU (MPU6050):
 *       SDA          -> GPIO 8
 *       SCL          -> GPIO 9
 *   - Motor Driver (TB6612FNG / DRV8833 / PWM+DIR):
 *       Left PWM     -> GPIO 1  (LEDC Ch 0)
 *       Left DIR1    -> GPIO 2
 *       Left DIR2    -> GPIO 3
 *       Right PWM    -> GPIO 4  (LEDC Ch 1)
 *       Right DIR1   -> GPIO 5
 *       Right DIR2   -> GPIO 6
 *   - Wheel Encoders (Quadrature):
 *       Left Enc A   -> GPIO 7
 *       Left Enc B   -> GPIO 10
 *       Right Enc A  -> GPIO 11
 *       Right Enc B  -> GPIO 12
 *   - Onboard RGB LED:
 *       WS2812 Data  -> GPIO 21
 * ============================================================================
 */

#include <Arduino.h>
#include <Wire.h>
#include "policy_net.h"

// ----------------------------------------------------------------------------
// PIN DEFINITIONS (ESP32-S3-Zero)
// ----------------------------------------------------------------------------
#define PIN_I2C_SDA        8
#define PIN_I2C_SCL        9

#define PIN_MOTOR_L_PWM    1
#define PIN_MOTOR_L_DIR1   2
#define PIN_MOTOR_L_DIR2   3

#define PIN_MOTOR_R_PWM    4
#define PIN_MOTOR_R_DIR1   5
#define PIN_MOTOR_R_DIR2   6

#define PIN_ENC_L_A        7
#define PIN_ENC_L_B        10
#define PIN_ENC_R_A        11
#define PIN_ENC_R_B        12

// ----------------------------------------------------------------------------
// ROBOT PHYSICAL PARAMETERS
// ----------------------------------------------------------------------------
const float WHEEL_RADIUS_M    = 0.020f;   // 20 mm radius (0.04m diameter)
const float ENCODER_TICKS_REV = 360.0f;   // Ticks per full wheel revolution (adjust for your motor gear ratio)
const float TIP_LIMIT_RAD     = 0.610f;   // 35 degrees cutoff
const float MIN_PWM_DEADBAND  = 25.0f;    // Minimum PWM to overcome motor stiction (0-255)
const float MAX_PWM           = 255.0f;

// ----------------------------------------------------------------------------
// PWM TIMER CONFIGURATION (LEDC for ESP32-S3)
// ----------------------------------------------------------------------------
const int PWM_FREQ = 20000;               // 20 kHz ultrasonic (silent motors)
const int PWM_RES  = 8;                   // 8-bit resolution (0 - 255)
const int PWM_CH_L = 0;
const int PWM_CH_R = 1;

// ----------------------------------------------------------------------------
// IMU MPU6050 CONSTANTS & CALIBRATION
// ----------------------------------------------------------------------------
#define MPU6050_ADDR 0x68
float gyro_pitch_bias = 0.0f;
float accel_pitch_bias = 0.0f;

// Complementary filter weight (0.98 gyro + 0.02 accel)
const float ALPHA_COMP = 0.985f;

// ----------------------------------------------------------------------------
// GLOBAL STATE VARIABLES
// ----------------------------------------------------------------------------
volatile long enc_left_ticks  = 0;
volatile long enc_right_ticks = 0;

float current_pitch      = 0.0f; // radians (0 = vertical)
float current_pitch_rate = 0.0f; // rad/s
float left_wheel_speed   = 0.0f; // rad/s
float right_wheel_speed  = 0.0f; // rad/s
float forward_velocity   = 0.0f; // m/s
float target_velocity    = 0.0f; // commanded forward speed (m/s)

unsigned long last_loop_micros = 0;
const unsigned long LOOP_PERIOD_US = 10000; // 10,000 us = 10.0 ms (100 Hz)

// ----------------------------------------------------------------------------
// ENCODER ISR ROUTINES
// ----------------------------------------------------------------------------
void IRAM_ATTR isr_enc_l() {
    int b = digitalRead(PIN_ENC_L_B);
    if (b > 0) enc_left_ticks++;
    else enc_left_ticks--;
}

void IRAM_ATTR isr_enc_r() {
    int b = digitalRead(PIN_ENC_R_B);
    if (b > 0) enc_right_ticks--;
    else enc_right_ticks++;
}

// ----------------------------------------------------------------------------
// MPU6050 LOW-LEVEL DRIVER
// ----------------------------------------------------------------------------
void mpu6050_init() {
    Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL, 400000); // Fast I2C 400kHz
    delay(50);

    // Wake up MPU6050
    Wire.beginTransmission(MPU6050_ADDR);
    Wire.write(0x6B); // PWR_MGMT_1
    Wire.write(0x00); // Wake up
    Wire.endTransmission(true);

    // Set Gyro Range to ±500 deg/s (FS_SEL = 1) -> 65.5 LSB/(deg/s)
    Wire.beginTransmission(MPU6050_ADDR);
    Wire.write(0x1B); // GYRO_CONFIG
    Wire.write(0x08);
    Wire.endTransmission(true);

    // Set Accel Range to ±4g (AFS_SEL = 1) -> 8192 LSB/g
    Wire.beginTransmission(MPU6050_ADDR);
    Wire.write(0x1C); // ACCEL_CONFIG
    Wire.write(0x08);
    Wire.endTransmission(true);
}

void mpu6050_calibrate() {
    Serial.println("Calibrating IMU... Keep robot still at vertical balance position.");
    long gyro_sum = 0;
    long accel_sum = 0;
    const int SAMPLES = 500;

    for (int i = 0; i < SAMPLES; i++) {
        Wire.beginTransmission(MPU6050_ADDR);
        Wire.write(0x3B);
        Wire.endTransmission(false);
        Wire.requestFrom((uint8_t)MPU6050_ADDR, (size_t)14, true);

        int16_t ax = (Wire.read() << 8) | Wire.read();
        int16_t ay = (Wire.read() << 8) | Wire.read();
        int16_t az = (Wire.read() << 8) | Wire.read();
        Wire.read(); Wire.read(); // skip temp
        int16_t gx = (Wire.read() << 8) | Wire.read();
        int16_t gy = (Wire.read() << 8) | Wire.read();
        int16_t gz = (Wire.read() << 8) | Wire.read();

        // Assuming robot tilt axis is around Y-axis (adjust if your IMU is mounted differently)
        gyro_sum += gy;
        accel_sum += atan2f((float)ax, (float)az) * 1000.0f;
        delay(4);
    }

    gyro_pitch_bias = ((float)gyro_sum / SAMPLES) / 65.5f * (PI / 180.0f); // rad/s
    accel_pitch_bias = ((float)accel_sum / SAMPLES) / 1000.0f;             // rad
    current_pitch = 0.0f;
    Serial.printf("Calibration Done! Gyro Bias: %.4f rad/s, Accel Pitch Bias: %.4f rad\n", gyro_pitch_bias, accel_pitch_bias);
}

void mpu6050_read_pitch(float dt) {
    Wire.beginTransmission(MPU6050_ADDR);
    Wire.write(0x3B);
    Wire.endTransmission(false);
    Wire.requestFrom((uint8_t)MPU6050_ADDR, (size_t)14, true);

    int16_t ax = (Wire.read() << 8) | Wire.read();
    int16_t ay = (Wire.read() << 8) | Wire.read();
    int16_t az = (Wire.read() << 8) | Wire.read();
    Wire.read(); Wire.read(); // skip temp
    int16_t gx = (Wire.read() << 8) | Wire.read();
    int16_t gy = (Wire.read() << 8) | Wire.read();
    int16_t gz = (Wire.read() << 8) | Wire.read();

    // Raw rates
    float gyro_rate = (((float)gy / 65.5f) * (PI / 180.0f)) - gyro_pitch_bias; // rad/s
    current_pitch_rate = gyro_rate;

    // Pitch from Accelerometer
    float accel_pitch = atan2f((float)ax, (float)az) - accel_pitch_bias;

    // Complementary Filter Fusion
    current_pitch = ALPHA_COMP * (current_pitch + gyro_rate * dt) + (1.0f - ALPHA_COMP) * accel_pitch;
}

// ----------------------------------------------------------------------------
// MOTOR DRIVER ACTUATION
// ----------------------------------------------------------------------------
void motors_init() {
    pinMode(PIN_MOTOR_L_DIR1, OUTPUT);
    pinMode(PIN_MOTOR_L_DIR2, OUTPUT);
    pinMode(PIN_MOTOR_R_DIR1, OUTPUT);
    pinMode(PIN_MOTOR_R_DIR2, OUTPUT);

    ledcSetup(PWM_CH_L, PWM_FREQ, PWM_RES);
    ledcSetup(PWM_CH_R, PWM_FREQ, PWM_RES);
    ledcAttachPin(PIN_MOTOR_L_PWM, PWM_CH_L);
    ledcAttachPin(PIN_MOTOR_R_PWM, PWM_CH_R);

    // Start with motors stopped
    ledcWrite(PWM_CH_L, 0);
    ledcWrite(PWM_CH_R, 0);
}

void set_motor_speeds(float u_balance) {
    // u_balance in [-1.0, 1.0]
    if (fabs(current_pitch) > TIP_LIMIT_RAD) {
        // Safety Cutoff: Robot fell over! Cut power to protect motors.
        ledcWrite(PWM_CH_L, 0);
        ledcWrite(PWM_CH_R, 0);
        digitalWrite(PIN_MOTOR_L_DIR1, LOW);
        digitalWrite(PIN_MOTOR_L_DIR2, LOW);
        digitalWrite(PIN_MOTOR_R_DIR1, LOW);
        digitalWrite(PIN_MOTOR_R_DIR2, LOW);
        return;
    }

    float u = constrain(u_balance, -1.0f, 1.0f);
    float duty = fabs(u) * (MAX_PWM - MIN_PWM_DEADBAND);
    if (duty > 0.5f) duty += MIN_PWM_DEADBAND;
    duty = constrain(duty, 0.0f, MAX_PWM);

    uint8_t pwm_val = (uint8_t)duty;

    // Symmetrical drive (Left and Right motors)
    if (u >= 0.0f) {
        // Forward
        digitalWrite(PIN_MOTOR_L_DIR1, HIGH);
        digitalWrite(PIN_MOTOR_L_DIR2, LOW);
        digitalWrite(PIN_MOTOR_R_DIR1, HIGH);
        digitalWrite(PIN_MOTOR_R_DIR2, LOW);
    } else {
        // Reverse
        digitalWrite(PIN_MOTOR_L_DIR1, LOW);
        digitalWrite(PIN_MOTOR_L_DIR2, HIGH);
        digitalWrite(PIN_MOTOR_R_DIR1, LOW);
        digitalWrite(PIN_MOTOR_R_DIR2, HIGH);
    }

    ledcWrite(PWM_CH_L, pwm_val);
    ledcWrite(PWM_CH_R, pwm_val);
}

// ----------------------------------------------------------------------------
// SETUP & MAIN LOOP
// ----------------------------------------------------------------------------
void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("\n--- Balancia Robot ESP32-S3-Zero Controller Starting ---");

    // Initialize Encoders
    pinMode(PIN_ENC_L_A, INPUT_PULLUP);
    pinMode(PIN_ENC_L_B, INPUT_PULLUP);
    pinMode(PIN_ENC_R_A, INPUT_PULLUP);
    pinMode(PIN_ENC_R_B, INPUT_PULLUP);
    attachInterrupt(digitalPinToInterrupt(PIN_ENC_L_A), isr_enc_l, RISING);
    attachInterrupt(digitalPinToInterrupt(PIN_ENC_R_A), isr_enc_r, RISING);

    // Initialize Hardware
    motors_init();
    mpu6050_init();
    mpu6050_calibrate();

    last_loop_micros = micros();
    Serial.println("System Ready. Balance loop running at 100 Hz.");
}

void loop() {
    unsigned long now = micros();
    if (now - last_loop_micros < LOOP_PERIOD_US) {
        return; // Deterministic 100 Hz timing
    }
    float dt = (now - last_loop_micros) * 1e-6f;
    last_loop_micros = now;

    // 1. Read IMU
    mpu6050_read_pitch(dt);

    // 2. Read Encoders & Calculate Wheel Speeds (rad/s)
    long ticks_l = enc_left_ticks;
    long ticks_r = enc_right_ticks;
    enc_left_ticks = 0;
    enc_right_ticks = 0;

    left_wheel_speed  = ((float)ticks_l / ENCODER_TICKS_REV) * (2.0f * PI) / dt;
    right_wheel_speed = ((float)ticks_r / ENCODER_TICKS_REV) * (2.0f * PI) / dt;
    forward_velocity  = 0.5f * (left_wheel_speed + right_wheel_speed) * WHEEL_RADIUS_M;

    // 3. Assemble 6D Observation Vector for Neural Network
    // Observation format matching MuJoCo gym environment:
    // [pitch (rad), pitch_rate (rad/s), forward_vel (m/s), lv (rad/s), rv (rad/s), target_vel (m/s)]
    float obs[NN_INPUT_DIM] = {
        current_pitch,
        current_pitch_rate,
        forward_velocity,
        left_wheel_speed,
        right_wheel_speed,
        target_velocity
    };

    // 4. Run Neural Network Policy Inference (< 5 microseconds on ESP32-S3)
    float action[NN_OUTPUT_DIM] = {0.0f};
    policy_predict(obs, action);

    // 5. Output Motor PWM / Torque
    set_motor_speeds(action[0]);

    // 6. Optional Telemetry (every 10 steps / 10 Hz)
    static int print_divider = 0;
    if (++print_divider >= 10) {
        print_divider = 0;
        Serial.printf("Pitch: %+5.1f deg | Gyro: %+5.2f rad/s | Speed: %+4.2f m/s | Action: %+4.2f\n",
                      current_pitch * (180.0f / PI), current_pitch_rate, forward_velocity, action[0]);
    }
}
