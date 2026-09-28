# Balancia Robot 🤖

A 2-wheeled self-balancing mobile robot trained using **Proximal Policy Optimization (PPO)** in **MuJoCo** with **2D differential drive steering**, **100 Hz control decimation**, and **robust push disturbance recovery**.

---

## Key Features

- **1D Direct Balance Torque**: Symmetric torque applied to both motors for pitch balance and stability.
- **100 Hz Control Decimation**: Physics runs at $500\,\text{Hz}$ ($\Delta t = 2\,\text{ms}$) while policy runs at $100\,\text{Hz}$ ($\Delta t = 10\,\text{ms}$, frame skip = 5), matching real-world microcontroller loop rates.
- **Simple Quadratic Penalty Reward**: Clean survival bonus combined with tilt, angular velocity, and control effort penalties.
- **Push Disturbance Recovery**: Random force impulses ($4.5\text{–}5.0\,\text{N}$) to train and evaluate active recovery.
- **Interactive MuJoCo Viewer**: Real-time viewer with live push injection using `SPACE`/`P`.

---

## Specifications

| Parameter | Value |
| :--- | :--- |
| **Chassis Mass** | $2.0\,\text{kg}$ |
| **Wheel Mass** | $0.05\,\text{kg}$ (each) |
| **Wheel Radius** | $0.02\,\text{m}$ |
| **Wheel Base** | $0.09\,\text{m}$ |
| **Actuator Type** | Direct Torque Motors (Left & Right) |
| **Physics Timestep** | $0.002\,\text{s}$ ($500\,\text{Hz}$) |
| **Control Timestep** | $0.010\,\text{s}$ ($100\,\text{Hz}$, `FRAME_SKIP = 5`) |
| **Max Episode Length** | $1000\,\text{steps}$ ($10.0\,\text{seconds}$) |
| **Tip Threshold** | $\pm 35.0^\circ$ ($0.61\,\text{rad}$) |

---

## Observation & Action Space

### Observation Space (`Box(6,)`)

| Index | Symbol | Name | Units | Range | Description |
| :---: | :---: | :--- | :---: | :---: | :--- |
| **0** | $\theta$ | Pitch Angle | $\text{rad}$ | $[-\infty, \infty]$ | Lean angle ($0 = \text{vertical}$, $+ = \text{forward}$) |
| **1** | $\dot{\theta}$ | Pitch Rate | $\text{rad/s}$ | $[-\infty, \infty]$ | Angular tilt velocity |
| **2** | $v_{\text{fwd}}$ | Forward Speed | $\text{m/s}$ | $[-\infty, \infty]$ | Linear velocity along robot heading |
| **3** | $\omega_L$ | Left Wheel Speed | $\text{rad/s}$ | $[-\infty, \infty]$ | Angular velocity of left wheel |
| **4** | $\omega_R$ | Right Wheel Speed | $\text{rad/s}$ | $[-\infty, \infty]$ | Angular velocity of right wheel |
| **5** | $v_{\text{target}}$ | Target Speed | $\text{m/s}$ | $[-0.5, 0.5]$ | Commanded forward velocity goal |

### Action Space (`Box(1,)`)

The policy outputs $\mathbf{a} = [u_{\text{balance}}] \in [-1.0, 1.0]$.

Symmetric torque applied to left and right motors:

$$\tau_{\text{left}} = \tau_{\text{right}} = u_{\text{balance}} \times \tau_{\max}$$

---

## Reward Formulation

A standard balancing and velocity tracking reward consisting of a positive survival bonus minus tilt, velocity tracking error, and control penalties:

$$\text{Reward} = 1.0 - 4.0 \cdot \theta^2 - 0.05 \cdot \dot{\theta}^2 - 0.2 \cdot \dot{\psi}^2 - 1.5 \cdot (v_{\text{fwd}} - v_{\text{target}})^2 - 0.0005 \cdot (\omega_L^2 + \omega_R^2) - 0.002 \cdot u^2 - 0.01 \cdot (\Delta u)^2$$

- **Alive Bonus ($+1.0$)**: Rewards surviving each time step.
- **Pitch Penalty ($-4.0 \cdot \theta^2$)**: Penalizes tilt deviation from upright equilibrium.
- **Pitch Rate Penalty ($-0.05 \cdot \dot{\theta}^2$)**: Damps angular oscillation velocity.
- **Yaw Penalty ($-0.2 \cdot \dot{\psi}^2$)**: Penalizes unwanted chassis spinning / turning drift.
- **Velocity Error Penalty ($-1.5 \cdot (v_{\text{fwd}} - v_{\text{target}})^2$)**: Drives the robot at commanded target speed.
- **Wheel Speed Penalty ($-0.0005 \cdot (\omega_L^2 + \omega_R^2)$)**: Penalizes excess wheel runaway.
- **Effort Penalty ($-0.002 \cdot u^2$)**: Penalizes excessive motor torque.
- **Action Smoothness ($-0.01 \cdot (\Delta u)^2$)**: Penalizes rapid motor jitter.
- **Terminal Condition**: When tipped beyond threshold ($|\theta| > 35^\circ$), reward is set to $0.0$.

---

## Installation & Setup

1. **Activate your environment:**

   ```bash
   source env_mujoco/bin/activate
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

---

## Training

### Train 2D Balance & Steering Policy

```bash
python scripts/train_rl.py --timesteps 1000000 --n-envs 8 --no-pushes
```

### Resume Training with Push Disturbance Recovery

```bash
python scripts/train_rl.py --resume --timesteps 1000000 --n-envs 8 --enable-pushes
```

### Monitor with TensorBoard

```bash
tensorboard --logdir logs
```

Open [http://localhost:6006](http://localhost:6006) in your browser.

---

## Interactive Play & Live Push Testing

Run the trained policy in the interactive MuJoCo viewer:

```bash
python scripts/play_rl.py
```

### Viewer Keyboard Controls

Click inside the MuJoCo viewer window to drive and perturb the robot:

- **`UP` / `DOWN`**: Increase / decrease forward speed target ($v_{\text{target}}$)
- **`LEFT` / `RIGHT`**: Steer left / right ($\omega_{\text{target}}$)
- **`SPACE` or `P`**: Inject an instant **$12\,\text{N}$ impulse push** to test live recovery!
- **`ENTER`**: Stop (reset speed and turn targets to 0)
- **`R`**: Reset robot to initial state
- **`Ctrl+C`**: Quit

---

## Project Structure

```
balancia_robot/
├── assets/
│   └── two_wheeled.xml         # MuJoCo MJCF robot model
├── envs/
│   └── balancing_robot_env.py  # Gymnasium environment (decimation, rewards, pushes)
├── scripts/
│   ├── train_rl.py             # Vectorized PPO training script
│   └── play_rl.py              # Interactive viewer with keyboard control & push testing
├── test/
│   ├── pid_balance.py          # Classical baseline controller
│   ├── test_model.py           # Model sanity tests
│   └── view_model.py           # Basic model visualizer
├── trained_models/             # Saved PPO checkpoints (.zip)
├── logs/                       # TensorBoard training logs
└── requirements.txt            # Python dependencies
```
