"""
Gymnasium environment for the two-wheeled balancing robot.

Action Space (1,):
    [balance_torque] in [-1, 1]
    - Symmetrical torque applied to both left and right wheel motors.

Control Decimation:
    Physics timestep dt = 0.002s (500 Hz).
    Policy control rate = 100 Hz (FRAME_SKIP = 5).
    Decimation eliminates high-frequency 500 Hz chattering and wiggling.

Observation Space (4,):
    [pitch, pitch_rate, left_wheel_velocity, right_wheel_velocity]
"""

from pathlib import Path
import math
from collections import deque

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco


MODEL_PATH = str(
    Path(__file__).resolve().parents[1] / "assets" / "two_wheeled.xml"
)

TIP_THRESHOLD = math.radians(35.0)
MAX_EPISODE_STEPS = 1000        # 1000 steps @ 100 Hz = 10.0 seconds
FRAME_SKIP = 5                  # 5 * 0.002s = 0.010s per policy step (100 Hz control)

# Reset tilt range (curriculum recovery training)
INIT_TILT_RANGE_DEG = 12.0      # uniform in [-12°, +12°]
INIT_TILT_RATE_RANGE = 0.25     # rad/s

# Velocity commands
MAX_TARGET_VEL = 0.5            # max speed: ±0.5 m/s
VEL_RAMP_RATE = 0.4             # acceleration: 0.4 m/s²
CMD_RESAMPLE_STEPS = 150        # resample target speed every 1.5s during training

# Push disturbances (Hard Push Training)
PUSH_PROB = 0.025               # 2.5% probability per step when enabled
PUSH_DURATION_STEPS = 6         # 6 * 10ms = 60 ms
PUSH_MAX_FORCE = 10.0           # Up to 10.0 N hard disturbance
PUSH_MAX_TORQUE = 0.35          # N*m

# Domain Randomization (Sim-to-Real Gap Reduction)
IMU_PITCH_BIAS_MAX = math.radians(2.0)       # ±2.0° pitch bias
IMU_PITCH_RATE_BIAS_MAX = 0.02               # ±0.02 rad/s gyro bias
IMU_PITCH_NOISE_STD = 0.008                  # Gaussian noise on pitch angle (rad)
IMU_PITCH_RATE_NOISE_STD = 0.015             # Gaussian noise on gyro rate (rad/s)

TORQUE_SCALE_MIN = 0.85                      # 85% to 115% torque scaling (battery drop & motor discrepancy)
TORQUE_SCALE_MAX = 1.15

MAX_ACTION_LATENCY_STEPS = 2                 # 0 to 2 steps (0 to 20ms @ 100 Hz)



class TwoWheeledBalanceEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self,
        render_mode=None,
        randomize_target_vel=False,
        enable_pushes=False,
        randomize_domain=False,
        randomize_imu=False,
        randomize_torque=False,
        randomize_latency=False,
        action_delay_steps=0,
    ):
        super().__init__()

        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data = mujoco.MjData(self.model)
        self.physics_dt = self.model.opt.timestep
        self.dt = self.physics_dt * FRAME_SKIP  # 0.010s (100 Hz)

        self.render_mode = render_mode
        self.randomize_target_vel = randomize_target_vel
        self.enable_pushes = enable_pushes

        # Domain randomization flags
        self.randomize_domain = randomize_domain
        self.randomize_imu = randomize_imu or randomize_domain
        self.randomize_torque = randomize_torque or randomize_domain
        self.randomize_latency = randomize_latency or randomize_domain
        self.action_delay_steps = action_delay_steps

        # Action queue & domain randomization variables
        self._action_queue = deque(maxlen=MAX_ACTION_LATENCY_STEPS + 1)
        self._imu_pitch_bias = 0.0
        self._imu_pitch_rate_bias = 0.0
        self._torque_scale = 1.0
        self._current_latency_steps = 0
        self._viewer = None

        # ---- ids ----
        self.left_motor_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_motor"
        )
        self.right_motor_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_motor"
        )
        self.base_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link"
        )
        assert self.left_motor_id >= 0, "left_motor not found in model"
        assert self.right_motor_id >= 0, "right_motor not found in model"
        assert self.base_id >= 0, "base_link not found in model"

        base_jnt_id = self.model.body_jntadr[self.base_id]
        assert self.model.jnt_type[base_jnt_id] == mujoco.mjtJoint.mjJNT_FREE, \
            "base_link needs a free joint"
        self.base_qpos_adr = self.model.jnt_qposadr[base_jnt_id]
        self.base_dof_adr = self.model.jnt_dofadr[base_jnt_id]

        # ---- wheel joints ----
        self.left_wheel_dof = self._wheel_dof("left_wheel_joint")
        self.right_wheel_dof = self._wheel_dof("right_wheel_joint")

        # ---- torque limits ----
        self.left_torque_limit = self._torque_limit(self.left_motor_id)
        self.right_torque_limit = self._torque_limit(self.right_motor_id)

        # ---- spaces ----
        # Action: [balance_torque] in [-1, 1] (symmetric drive to both wheels)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        # 6D Observation: [pitch, pitch_rate, forward_vel, left_wheel_vel, right_wheel_vel, target_vel]
        obs_high = np.array([np.inf] * 6, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-obs_high, high=obs_high, dtype=np.float32
        )

        # Velocity tracking states
        self.target_vel = 0.0
        self._target_vel_goal = 0.0

        # Action history & push tracking
        self._prev_action = np.zeros(1, dtype=np.float32)
        self._step_count = 0
        self._push_remaining_steps = 0
        self._active_push_force = np.zeros(6, dtype=np.float64)

    # ------------------------------------------------------------
    # setup helpers
    # ------------------------------------------------------------
    def _wheel_dof(self, joint_name):
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            return None
        return self.model.jnt_dofadr[jid]

    def _torque_limit(self, actuator_id):
        if self.model.actuator_ctrllimited[actuator_id]:
            lo, hi = self.model.actuator_ctrlrange[actuator_id]
            return float(max(abs(lo), abs(hi)))
        return 5.0

    # ------------------------------------------------------------
    # state readers
    # ------------------------------------------------------------
    def _get_pitch(self):
        R = self.data.xmat[self.base_id].reshape(3, 3)
        return math.atan2(-R[2, 0], R[0, 0])

    def _get_pitch_rate(self):
        return self.data.qvel[self.base_dof_adr + 4]

    def _get_yaw_rate(self):
        return self.data.qvel[self.base_dof_adr + 5]

    def _get_forward_speed(self):
        R = self.data.xmat[self.base_id].reshape(3, 3)
        v_world = self.data.qvel[self.base_dof_adr : self.base_dof_adr + 3]
        return float(np.dot(R[:, 0], v_world))

    def _get_wheel_speeds(self):
        lv = self.data.qvel[self.left_wheel_dof] if self.left_wheel_dof is not None else 0.0
        rv = self.data.qvel[self.right_wheel_dof] if self.right_wheel_dof is not None else 0.0
        return float(lv), float(rv)

    def _get_obs(self):
        pitch = self._get_pitch()
        pitch_rate = self._get_pitch_rate()

        if self.randomize_imu:
            pitch_obs = pitch + self._imu_pitch_bias + float(self.np_random.normal(0, IMU_PITCH_NOISE_STD))
            pitch_rate_obs = pitch_rate + self._imu_pitch_rate_bias + float(self.np_random.normal(0, IMU_PITCH_RATE_NOISE_STD))
        else:
            pitch_obs = pitch
            pitch_rate_obs = pitch_rate

        v_fwd = self._get_forward_speed()
        lv, rv = self._get_wheel_speeds()
        return np.array(
            [pitch_obs, pitch_rate_obs, v_fwd, lv, rv, self.target_vel],
            dtype=np.float32,
        )

    # ------------------------------------------------------------
    # push disturbance handler
    # ------------------------------------------------------------
    def _apply_push_disturbance(self):
        if not self.enable_pushes:
            return

        if self._push_remaining_steps <= 0:
            if self.np_random.uniform(0, 1) < PUSH_PROB:
                self._push_remaining_steps = PUSH_DURATION_STEPS
                fx = self.np_random.uniform(-PUSH_MAX_FORCE, PUSH_MAX_FORCE)
                fy = self.np_random.uniform(-PUSH_MAX_FORCE, PUSH_MAX_FORCE)
                tz = self.np_random.uniform(-PUSH_MAX_TORQUE, PUSH_MAX_TORQUE)
                self._active_push_force = np.array([fx, fy, 0.0, 0.0, 0.0, tz], dtype=np.float64)
            else:
                self._active_push_force.fill(0.0)

        if self._push_remaining_steps > 0:
            self.data.xfrc_applied[self.base_id, :] = self._active_push_force
            self._push_remaining_steps -= 1
        else:
            self.data.xfrc_applied[self.base_id, :].fill(0.0)

    # ------------------------------------------------------------
    # gym API
    # ------------------------------------------------------------
    def reset(self, *, seed=None, randomize=True, **kwargs):
        super().reset(seed=seed)
        rng = self.np_random

        mujoco.mj_resetData(self.model, self.data)
        self.data.xfrc_applied[self.base_id, :].fill(0.0)
        self._push_remaining_steps = 0
        self._active_push_force.fill(0.0)

        # Domain Randomization: IMU biases
        if self.randomize_imu:
            self._imu_pitch_bias = float(rng.uniform(-IMU_PITCH_BIAS_MAX, IMU_PITCH_BIAS_MAX))
            self._imu_pitch_rate_bias = float(rng.uniform(-IMU_PITCH_RATE_BIAS_MAX, IMU_PITCH_RATE_BIAS_MAX))
        else:
            self._imu_pitch_bias = 0.0
            self._imu_pitch_rate_bias = 0.0

        # Domain Randomization: Motor Torque scaling (mass left unchanged)
        if self.randomize_torque:
            self._torque_scale = float(rng.uniform(TORQUE_SCALE_MIN, TORQUE_SCALE_MAX))
        else:
            self._torque_scale = 1.0

        # Domain Randomization: Action latency queue
        if self.randomize_latency:
            self._current_latency_steps = int(rng.integers(0, MAX_ACTION_LATENCY_STEPS + 1))
        else:
            self._current_latency_steps = self.action_delay_steps

        self._action_queue.clear()
        for _ in range(MAX_ACTION_LATENCY_STEPS + 1):
            self._action_queue.append(np.zeros(1, dtype=np.float32))

        tilt_deg = rng.uniform(-INIT_TILT_RANGE_DEG, INIT_TILT_RANGE_DEG) if randomize else 4.0
        theta = math.radians(tilt_deg)
        w1, x1, y1, z1 = math.cos(theta / 2), 0.0, math.sin(theta / 2), 0.0
        w2, x2, y2, z2 = self.data.qpos[self.base_qpos_adr + 3 : self.base_qpos_adr + 7]

        self.data.qpos[self.base_qpos_adr + 3] = w1*w2 - x1*x2 - y1*y2 - z1*z2
        self.data.qpos[self.base_qpos_adr + 4] = w1*x2 + x1*w2 + y1*z2 - z1*y2
        self.data.qpos[self.base_qpos_adr + 5] = w1*y2 - x1*z2 + y1*w2 + z1*x2
        self.data.qpos[self.base_qpos_adr + 6] = w1*z2 + x1*y2 - y1*x2 + z1*w2

        if randomize:
            self.data.qvel[self.base_dof_adr + 4] = rng.uniform(
                -INIT_TILT_RATE_RANGE, INIT_TILT_RATE_RANGE
            )

        mujoco.mj_forward(self.model, self.data)

        self.target_vel = 0.0
        self._target_vel_goal = (
            float(rng.uniform(-MAX_TARGET_VEL, MAX_TARGET_VEL))
            if self.randomize_target_vel
            else 0.0
        )

        self._prev_action = np.zeros(1, dtype=np.float32)
        self._step_count = 0

        return self._get_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        u = float(np.clip(action[0], -1.0, 1.0))
        action = np.array([u], dtype=np.float32)

        # Action latency queue buffer
        self._action_queue.append(action)
        delayed_action = self._action_queue[-(self._current_latency_steps + 1)]
        u_applied = float(delayed_action[0])

        # Resample target velocity periodically during training
        if (
            self.randomize_target_vel
            and CMD_RESAMPLE_STEPS > 0
            and self._step_count > 0
            and self._step_count % CMD_RESAMPLE_STEPS == 0
        ):
            self._target_vel_goal = float(
                self.np_random.uniform(-MAX_TARGET_VEL, MAX_TARGET_VEL)
            )

        # Smooth velocity command ramping
        vel_step = VEL_RAMP_RATE * self.dt
        self.target_vel += float(
            np.clip(self._target_vel_goal - self.target_vel, -vel_step, vel_step)
        )

        # Apply symmetric balance torque with torque randomization
        left_torque = u_applied * self.left_torque_limit * self._torque_scale
        right_torque = u_applied * self.right_torque_limit * self._torque_scale

        self.data.ctrl[self.left_motor_id] = left_torque
        self.data.ctrl[self.right_motor_id] = right_torque

        # Apply disturbance push if active
        self._apply_push_disturbance()

        # Step MuJoCo physics multiple times per policy step (Decimation)
        for _ in range(FRAME_SKIP):
            mujoco.mj_step(self.model, self.data)
            if abs(self._get_pitch()) > TIP_THRESHOLD:
                break

        self._step_count += 1

        pitch = self._get_pitch()
        pitch_rate = self._get_pitch_rate()
        yaw_rate = self._get_yaw_rate()
        v_fwd = self._get_forward_speed()
        lv, rv = self._get_wheel_speeds()

        terminated = abs(pitch) > TIP_THRESHOLD
        truncated = self._step_count >= MAX_EPISODE_STEPS

        vel_error = v_fwd - self.target_vel
        action_delta = action - self._prev_action
        self._prev_action = action.copy()

        # Simple non-exponential balance + velocity tracking + yaw suppression reward:
        pitch_cost = 4.0 * (pitch ** 2)
        pitch_rate_cost = 0.05 * (pitch_rate ** 2)
        yaw_cost = 0.2 * (yaw_rate ** 2)       # Penalizes unwanted yaw rotation / spinning
        vel_cost = 1.5 * (vel_error ** 2)
        wheel_cost = 0.0005 * (lv ** 2 + rv ** 2)
        action_cost = 0.002 * (u ** 2)
        action_delta_cost = 0.01 * float(action_delta[0] ** 2)

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
        if terminated:
            reward = 0.0

        obs = self._get_obs()
        info = {
            "pitch": pitch,
            "pitch_rate": pitch_rate,
            "yaw_rate": yaw_rate,
            "forward_velocity": v_fwd,
            "target_velocity": self.target_vel,
            "wheel_speeds": (lv, rv),
            "wheel_avg": 0.5 * (lv + rv),
            "is_being_pushed": self._push_remaining_steps > 0,
            "reward_breakdown": {
                "alive_bonus": 1.0,
                "pitch_cost": pitch_cost,
                "pitch_rate_cost": pitch_rate_cost,
                "yaw_cost": yaw_cost,
                "vel_cost": vel_cost,
                "wheel_cost": wheel_cost,
                "action_cost": action_cost,
                "action_delta_cost": action_delta_cost,
                "total": reward,
            },
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return
        import mujoco.viewer as mjv
        if self._viewer is None:
            self._viewer = mjv.launch_passive(self.model, self.data)
            self._viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            self._viewer.cam.lookat[:] = [0.0, 0.0, 0.05]
            self._viewer.cam.distance = 0.8
            self._viewer.cam.azimuth = 45
            self._viewer.cam.elevation = -25
        self._viewer.sync()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None


# Optional registration
try:
    gym.register(
        id="TwoWheeledBalance-v0",
        entry_point="envs.balancing_robot_env:TwoWheeledBalanceEnv",
        max_episode_steps=MAX_EPISODE_STEPS,
    )
except gym.error.Error:
    pass