# Balancia Robot — Learning Log & Technical Progress Notes 🤖

This document tracks our step-by-step reinforcement learning progress, physical insights, reward formulation tuning, and architecture evolution for the two-wheeled self-balancing robot in MuJoCo.

---

## 1. Core Physics & Control Principles

A 2-wheeled self-balancing robot is an **inverted pendulum on wheels**. Unlike a standard 4-wheeled car:

1. **Non-Minimum Phase Dynamics (Tilt-to-Drive)**:
   - To accelerate forward, the wheels must momentarily kick **backward** for a fraction of a second so the chassis falls into a forward lean ($\theta > 0$).
   - Gravity then pulls the robot forward, and the wheels drive forward beneath it to catch and sustain the forward lean without tipping over.
   - To brake or reverse, the wheels must accelerate forward faster than the chassis to tip the body backward ($\theta < 0$) and resist forward momentum.
2. **Why Simple Torque Addition Fails**:
   - Simply injecting a forward torque offset ($u = u_{\text{balance}} + u_{\text{drive}}$) causes the bottom to shoot forward and the top to tip backward, triggering the balance controller to brake and fight the driving torque.
   - **Solution**: Feeding `target_velocity` directly into the RL observation space allows the neural network to discover and coordinate this lean-and-cruise physics end-to-end.
3. **Control Decimation**:
   - Physics runs at $500\,\text{Hz}$ ($\Delta t = 2\,\text{ms}$).
   - Policy control rate runs at $100\,\text{Hz}$ (`FRAME_SKIP = 5`, $\Delta t = 10\,\text{ms}$).
   - Decimation eliminates high-frequency motor chattering and matches embedded microcontroller loop rates.

---

## 2. Phase 1: Pure Stationary Balancing

### Goal
Keep the robot upright at vertical equilibrium ($\theta \approx 0^\circ$) without drifting or tipping over.

### Observation Space (`Box(4,)`)
```text
obs = [
    pitch,                # Index 0: Body tilt angle (rad, 0 = upright)
    pitch_rate,           # Index 1: Angular velocity of tilt (rad/s)
    left_wheel_velocity,  # Index 2: Left wheel rotational speed (rad/s)
    right_wheel_velocity  # Index 3: Right wheel rotational speed (rad/s)
]
```

### Action Space (`Box(1,)`)
- `action[0]` $\in [-1.0, 1.0]$: Symmetrical torque command driving both left and right direct-drive motors ($\tau_L = \tau_R = u \times \tau_{\max}$).

### Reward Function (Classic Quadratic Penalty)
```python
pitch_cost         = 5.0   * (pitch ** 2)
pitch_rate_cost    = 0.05  * (pitch_rate ** 2)
wheel_cost         = 0.001 * (left_wheel_vel ** 2 + right_wheel_vel ** 2)
action_cost        = 0.002 * (torque ** 2)
action_delta_cost  = 0.01  * ((torque - prev_torque) ** 2)

# Total Step Reward:
reward = 1.0 - pitch_cost - pitch_rate_cost - wheel_cost - action_cost - action_delta_cost

if abs(pitch) > 35.0_degrees:
    reward = 0.0
```

### Key Learnings from Phase 1
- When trained from mild tilts ($\pm 8^\circ$), PPO converges rapidly within ~150k steps to full 1000-step balance episodes (`ep_rew_mean ≈ 980+`).
- Positive alive bonus ($+1.0$) combined with small quadratic penalties provides clear, stable gradients without local traps.

---

## 3. Phase 2: Push Disturbance & Hard Recovery

### Goal
Train the policy to actively catch itself and recover from violent external force impulses ($8\text{–}10\,\text{N}$) without tipping over.

### Key Adjustments Made
1. **Push Shock Parameters**:
   - `PUSH_MAX_FORCE`: Set to `10.0 N` (hard impact shock).
   - `PUSH_DURATION_STEPS`: Set to `6 steps` ($60\,\text{ms}$ pulse).
   - `PUSH_PROB`: `0.025` (2.5% chance per step).
2. **Lowered Action Penalty for Rapid Torque Bursts**:
   - Lowered `action_cost` ($0.01 \to 0.002$) and `action_delta_cost` ($0.05 \to 0.01$).
   - This allows the neural network to output instantaneous maximum torque to sprint under its center of mass when hit by a strong shove.

---

## 4. Phase 3: Dynamic Balancing & Velocity Tracking

### Goal
Allow the robot to balance while accelerating, cruising, braking, and reversing on command ($v_{\text{target}}$).

### Observation Space (`Box(6,)`)
```text
obs = [
    pitch,                # Index 0: Body tilt angle (rad)
    pitch_rate,           # Index 1: Angular tilt velocity (rad/s)
    forward_velocity,     # Index 2: [NEW] Actual forward linear speed (m/s)
    left_wheel_velocity,  # Index 3: Left wheel rotational speed (rad/s)
    right_wheel_velocity, # Index 4: Right wheel rotational speed (rad/s)
    target_velocity       # Index 5: [NEW] Commanded target speed (m/s)
]
```

### Action Space (`Box(1,)`)
- `action[0]` $\in [-1.0, 1.0]$: Symmetrical torque command.

### Updated Reward Function (Balance + Speed Tracking + Yaw Suppression)
```python
vel_error          = forward_velocity - target_velocity

pitch_cost         = 4.0    * (pitch ** 2)
pitch_rate_cost    = 0.05   * (pitch_rate ** 2)
yaw_cost           = 0.2    * (yaw_rate ** 2)                # [NEW] Suppresses yaw rotation / drift
vel_cost           = 1.5    * (vel_error ** 2)               # [NEW] Velocity tracking term
wheel_cost         = 0.0005 * (left_wheel_vel ** 2 + right_wheel_vel ** 2)
action_cost        = 0.002  * (torque ** 2)
action_delta_cost  = 0.01   * ((torque - prev_torque) ** 2)

# Total Step Reward:
reward = (
    1.0
    - pitch_cost
    - pitch_rate_cost
    - yaw_cost
    - vel_cost
    - wheel_cost
    - action_cost
    - action_delta_cost
)

if abs(pitch) > 35.0_degrees:
    reward = 0.0
```

---

## 5. Phase 4: Domain Randomization & Sim-to-Real Transfer 🎯

### Goal
Close the reality gap between MuJoCo physics and the physical robot hardware. Inverted pendulums are notoriously sensitive to **actuation latency**, **IMU bias drift**, **motor deadbands**, and **Center of Mass (CoM) shifts**.

```
   ┌──────────────────────────────────────────────────────────────┐
   │                    SIM-TO-REAL BOTTLENECKS                   │
   ├──────────────────────────────┬───────────────────────────────┤
   │ Simulation (MuJoCo)          │ Physical Hardware (Real World)│
   ├──────────────────────────────┼───────────────────────────────┤
   │ Zero latency (0 ms)          │ 10–30 ms loop/bus lag         │
   │ Perfect IMU angle reading    │ Noise, bias drift, filter lag │
   │ Known exact Center of Mass   │ Battery & wiring shift CoM    │
   │ Linear frictionless motors   │ Stiction, deadband, back-EMF  │
   │ Constant floor friction      │ Tiles, wood, carpet, dust     │
   └──────────────────────────────┴───────────────────────────────┘
```

---

### Domain Randomization Strategy

#### 1. Actuation Latency & Communication Lag (The #1 Sim-to-Real Killer)
- **Problem**: In simulation, torque is applied instantaneously. On physical hardware, the control loop, microcontroller UART/CAN bus, and motor ESC current loop introduce a **10–25 ms delay**.
- **Solution**: Implement an **Action Buffer / Delay Queue**:
  - Sample random step latency $d \sim \mathcal{U}\{0, 1, 2\}$ policy steps ($0\text{–}20\,\text{ms}$ at 100 Hz).
  - Apply delayed action $a_{t-d}$ to MuJoCo actuators instead of $a_t$.

#### 2. Motor Dynamics & Non-Linearities
- **Motor Strength Scaling**: Randomize motor torque gain per episode:
  $$\tau_{\text{actual}} = \kappa \cdot u \cdot \tau_{\max}, \quad \kappa \sim \mathcal{U}(0.85, 1.15)$$
  *(Models battery voltage drops from full charge 12.6V down to 10.5V).*
- **Motor Damping & Friction Randomization**:
  - Viscous damping: $b \sim \mathcal{U}(0.001, 0.005)\,\text{N}\cdot\text{s/rad}$
  - Motor deadband / Coulomb stiction: $\tau_{\text{friction}} = \tau_{\text{dry}} \cdot \text{sign}(\omega)$ where $\tau_{\text{dry}} \sim \mathcal{U}(0.01, 0.04)\,\text{N}\cdot\text{m}$.

#### 3. Sensor Noise, Bias Drift & Quantization
On real robots, observations come from an IMU (e.g. MPU6050 / BNO085) and optical wheel encoders:
- **Pitch Angle ($\theta$)**:
  - Zero-rate bias drift: $b_\theta \sim \mathcal{U}(-0.035, +0.035)\,\text{rad}$ ($\approx \pm 2.0^\circ$).
  - Gaussian measurement noise: $\epsilon_\theta \sim \mathcal{N}(0, 0.015\,\text{rad})$.
  - $\theta_{\text{obs}} = \theta_{\text{true}} + b_\theta + \epsilon_\theta$.
- **Pitch Rate ($\dot{\theta}$)**:
  - Gyro noise: $\epsilon_{\dot{\theta}} \sim \mathcal{N}(0, 0.03\,\text{rad/s})$.
  - Gyro bias: $b_{\dot{\theta}} \sim \mathcal{U}(-0.02, +0.02\,\text{rad/s})$.
- **Wheel Encoders ($\omega_L, \omega_R$)**:
  - Encoder quantization & differentiation noise: $\epsilon_\omega \sim \mathcal{N}(0, 0.15\,\text{rad/s})$.

#### 4. Inertial & Physical Parameter Variations (Sampled at Episode Reset)
| Parameter | Nominal Value | Randomization Range | Real-World Reason |
| :--- | :--- | :--- | :--- |
| **Chassis Mass ($m$)** | $2.0\,\text{kg}$ | $[1.70, 2.30]\,\text{kg}$ ($\pm 15\%$) | Battery swaps, payload additions |
| **CoM Offset ($X$)** | $0.0\,\text{mm}$ | $[-8.0, +8.0]\,\text{mm}$ | Asymmetric electronics/cables |
| **CoM Offset ($Z$)** | Nominal | $[-15.0, +15.0]\,\text{mm}$ | Battery mounting height |
| **Ground Friction ($\mu$)** | $1.0$ | $[0.40, 1.30]$ | Tiles vs. hardwood vs. rug |
| **Push Shocks** | $0.0\,\text{N}$ | $[2.0, 10.0]\,\text{N}$ pulse | External human pushes / bumps |

---

### Key Architectural Upgrades for Robust Sim-to-Real

1. **Include Previous Actions in Observation Space**:
   - Adding $a_{t-1}$ (and optionally $a_{t-2}$) into the observation vector allows the policy to estimate hidden latency and velocity response.
2. **Smoothness Penalties ($\Delta u$)**:
   - High motor jerk kills real motor gearboxes and overheats ESCs. Keeping `action_delta_cost = 0.01` ensures smooth, continuous torque outputs without high-frequency chattering.
3. **Equilibrium Adaptation (Handling Asymmetric CoM)**:
   - When CoM is shifted forward by $+5\,\text{mm}$, the true mechanical balance point is $\theta \approx -1.5^\circ$, not $0.0^\circ$. By training with randomized CoM, the policy learns integral-like balance recovery rather than statically assuming $\theta = 0$ is vertical.

---

## 6. Summary Comparison Matrix

| Component | Phase 1 (Stationary Balance) | Phase 3 (Balance & Drive) | Phase 4 (Domain Randomized Sim-to-Real) |
| :--- | :--- | :--- | :--- |
| **Observation Space** | `Box(4,)` `[pitch, d_pitch, lv, rv]` | `Box(6,)` `[pitch, d_pitch, v_fwd, lv, rv, v_target]` | `Box(7,)` `[pitch+noise, d_pitch+noise, v_fwd, lv, rv, v_target, prev_action]` |
| **Action Space** | `Box(1,)` Symmetric torque | `Box(1,)` Symmetric torque | `Box(1,)` Symmetric torque (with 10–20 ms delay queue) |
| **Latency Modeling** | None ($0\,\text{ms}$) | None ($0\,\text{ms}$) | Random buffer delay $d \in [0, 2]$ steps ($0\text{–}20\,\text{ms}$) |
| **Sensors** | Ideal MuJoCo state | Ideal MuJoCo state | Gaussian noise + IMU bias drift + encoder quantization |
| **Dynamics** | Fixed mass & CoM | Fixed mass & CoM | $\pm 15\%$ mass, $\pm 8\text{mm}$ CoM $X$, $\pm 15\text{mm}$ CoM $Z$, $\mu \in [0.4, 1.3]$ |
| **Motor Model** | Perfect linear actuator | Perfect linear actuator | $\pm 15\%$ torque scaling, stiction deadband, voltage drop |

---

## 7. How to Train & Deploy

### 1. Train Robust Policy with Full Domain Randomization
```bash
python scripts/train_rl.py --timesteps 2000000 --n-envs 8 --randomize-domain --randomize-commands
```

### 2. Interactive Viewer & Stress Testing
```bash
python scripts/play_rl.py --eval-domain-randomization
```
- **`UP` / `DOWN`**: Increase / decrease target speed ($v_{\text{target}}$).
- **`ENTER`**: Stop / brake ($v_{\text{target}} = 0.0\,\text{m/s}$).
- **`SPACE` / `P`**: Inject an $8.0\text{–}10.0\,\text{N}$ force shockwave.
- **`R`**: Reset robot state with newly randomized mass, CoM, and friction.

### 3. Sim-to-Real Deployment Pipeline
```
[ PyTorch Policy (.zip / .pt) ]
             │
             ▼  (ONNX Export)
     [ balancia_policy.onnx ]
             │
             ▼  (Quantization / C++ Header Generation)
     [ policy_weights.h / TFLite Micro ]
             │
             ▼
[ Microcontroller (ESP32 / STM32 / Teensy 4.1 @ 100 Hz) ]
   ├── Inputs : IMU (Pitch, Gyro) + Encoders (Left/Right) + Target Speed
   └── Output : Dual Motor PWM / CAN Torque Commands
```

