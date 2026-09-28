"""
Load a trained PPO model and watch it balance and recover from pushes in MuJoCo.

Interactive Keyboard Controls (click on the MuJoCo viewer window):
    SPACE or P   : deliver a strong impulse push to test live recovery
    R            : reset robot state
    Ctrl+C       : quit

Usage:
    python scripts/play_rl.py
    python scripts/play_rl.py --model trained_models/ppo_balance_latest.zip
    python scripts/play_rl.py --enable-pushes
"""

import argparse
import os
import signal
import sys
import threading
import time
from pathlib import Path
import numpy as np
import mujoco
import mujoco.viewer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from envs.balancing_robot_env import TwoWheeledBalanceEnv  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402


# GLFW key codes
KEY_SPACE = 32
KEY_ENTER = 257
KEY_RIGHT = 262
KEY_LEFT = 263
KEY_DOWN = 264
KEY_UP = 265
KEY_P = 80
KEY_R = 82

SPEED_STEP = 0.05
TURN_STEP = 0.25
MAX_SPEED = 0.5
MAX_TURN = 1.5

running = True
manual_push_triggered = False
reset_triggered = False


def handle_sigint(signum, frame):
    global running
    running = False


def main():
    global manual_push_triggered, reset_triggered

    parser = argparse.ArgumentParser(description="Watch Trained RL Policy with Interactive Controls")
    parser.add_argument(
        "--model",
        default=str(PROJECT_ROOT / "trained_models" / "ppo_balance_latest.zip"),
        help="Path to trained PPO model .zip",
    )
    parser.add_argument("--target-vel", type=float, default=0.0,
                        help="Initial forward velocity command (m/s).")
    parser.add_argument("--target-yaw-rate", type=float, default=0.0,
                        help="Initial yaw rate command (rad/s).")
    parser.add_argument("--enable-pushes", action="store_true", default=False,
                        help="Enable continuous random background pushes.")
    parser.add_argument("--randomize-domain", action="store_true", default=False,
                        help="Enable full domain randomization during play.")
    parser.add_argument("--randomize-imu", action="store_true", default=False,
                        help="Enable IMU pitch & gyro bias drift.")
    parser.add_argument("--randomize-torque", action="store_true", default=False,
                        help="Enable motor torque scaling variations.")
    parser.add_argument("--randomize-latency", action="store_true", default=False,
                        help="Enable random action latency buffer (0-20ms).")
    parser.add_argument("--action-delay-steps", type=int, default=0,
                        help="Fixed action latency delay in steps (1 step = 10ms).")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, handle_sigint)

    if not Path(args.model).exists():
        print(f"Error: Model checkpoint '{args.model}' not found.")
        print("Please train a model first with: python scripts/train_rl.py")
        sys.exit(1)

    print(f"Loading model from {args.model} ...")
    model = PPO.load(args.model)

    env = TwoWheeledBalanceEnv(
        render_mode=None,  # We manage the viewer loop directly with key callbacks
        randomize_target_vel=False,
        enable_pushes=args.enable_pushes,
        randomize_domain=args.randomize_domain,
        randomize_imu=args.randomize_imu,
        randomize_torque=args.randomize_torque,
        randomize_latency=args.randomize_latency,
        action_delay_steps=args.action_delay_steps,
    )

    cmd_speed = args.target_vel

    def key_callback(keycode):
        nonlocal cmd_speed
        global manual_push_triggered, reset_triggered

        if keycode == KEY_UP:
            cmd_speed = float(np.clip(cmd_speed + SPEED_STEP, -MAX_SPEED, MAX_SPEED))
        elif keycode == KEY_DOWN:
            cmd_speed = float(np.clip(cmd_speed - SPEED_STEP, -MAX_SPEED, MAX_SPEED))
        elif keycode == KEY_ENTER:
            cmd_speed = 0.0
        elif keycode in (KEY_SPACE, KEY_P):
            manual_push_triggered = True
        elif keycode == KEY_R:
            reset_triggered = True

    obs, _ = env.reset()
    env.target_vel = cmd_speed

    print("\n=======================================================")
    print("  Two-Wheeled Robot - Balance & Drive Controller")
    print("=======================================================")
    print("Click inside the MuJoCo viewer window and use:")
    print("  UP / DOWN     : Forward / Backward speed command (m/s)")
    print("  ENTER         : Stop (reset target speed to 0.0)")
    print("  SPACE / P     : Push the robot with a hard impulse shock!")
    print("  R             : Reset robot")
    print("  Ctrl+C        : Quit")
    print("=======================================================\n")

    try:
        with mujoco.viewer.launch_passive(env.model, env.data, key_callback=key_callback) as viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.lookat[:] = [0.0, 0.0, 0.05]
            viewer.cam.distance = 0.8
            viewer.cam.azimuth = 45
            viewer.cam.elevation = -25

            while running and viewer.is_running():
                step_start = time.time()

                if reset_triggered:
                    reset_triggered = False
                    obs, _ = env.reset()

                # Manual push injection from user keystroke
                if manual_push_triggered:
                    manual_push_triggered = False
                    # Realistic impulse shock (matching training distribution)
                    push_angle = np.random.uniform(0, 2 * np.pi)
                    push_mag = 8.0  # 8.0 N hard impulse
                    fx = push_mag * np.cos(push_angle)
                    fy = push_mag * np.sin(push_angle)
                    tz = np.random.uniform(-0.25, 0.25)
                    env._push_remaining_steps = 5  # 50 ms impulse pulse
                    env._active_push_force = np.array([fx, fy, 0.0, 0.0, 0.0, tz], dtype=np.float64)
                    print(f"\n[PUSH] Applied {push_mag:.1f}N hard impulse disturbance!")

                env.target_vel = cmd_speed
                obs[5] = cmd_speed

                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                viewer.sync()

                push_str = " [BEING PUSHED!]" if info["is_being_pushed"] else ""
                print(
                    f"\rpitch={np.degrees(info['pitch']):+5.1f}° | "
                    f"v={info.get('forward_velocity', 0.0):+5.2f}/{cmd_speed:+4.2f} m/s | "
                    f"wheel_avg={info.get('wheel_avg', 0.0):+5.2f} rad/s{push_str}   ",
                    end="",
                    flush=True,
                )

                if terminated:
                    print("\nRobot tipped past limit! Resetting...")
                    obs, _ = env.reset()

                remaining = env.dt - (time.time() - step_start)
                if remaining > 0:
                    time.sleep(remaining)

    finally:
        print("\nClosing...")
        env.close()
        threading.Timer(1.0, lambda: os._exit(0)).start()


if __name__ == "__main__":
    main()