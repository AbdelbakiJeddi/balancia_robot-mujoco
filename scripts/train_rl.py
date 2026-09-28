"""
Train PPO to balance the two-wheeled robot with symmetric torque control.

Usage:
    python scripts/train_rl.py --timesteps 1000000 --n-envs 8
    python scripts/train_rl.py --resume
"""

import argparse
import sys
from pathlib import Path

# Make the sibling "envs" package importable regardless of cwd
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from envs.balancing_robot_env import TwoWheeledBalanceEnv  # noqa: E402

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor


MODEL_DIR = PROJECT_ROOT / "trained_models"
LOG_DIR = PROJECT_ROOT / "logs"


def make_env(
    randomize_commands=True,
    enable_pushes=False,
    randomize_domain=False,
    randomize_imu=False,
    randomize_torque=False,
    randomize_latency=False,
):
    def _init():
        env = TwoWheeledBalanceEnv(
            render_mode=None,
            randomize_target_vel=randomize_commands,
            enable_pushes=enable_pushes,
            randomize_domain=randomize_domain,
            randomize_imu=randomize_imu,
            randomize_torque=randomize_torque,
            randomize_latency=randomize_latency,
        )
        return Monitor(env)
    return _init


def main():
    parser = argparse.ArgumentParser(description="Train PPO for Two-Wheeled Robot Balancing & Driving")
    parser.add_argument("--timesteps", type=int, default=1_000_000,
                        help="Total timesteps to train.")
    parser.add_argument("--n-envs", type=int, default=8,
                        help="Parallel environments; lower this if CPU is constrained.")
    parser.add_argument("--randomize-commands", action="store_true", default=True,
                        help="Randomize target velocity commands during training.")
    parser.add_argument("--no-randomize-commands", action="store_false", dest="randomize_commands",
                        help="Train stationary balance only.")
    parser.add_argument("--enable-pushes", action="store_true", default=False,
                        help="Enable random push disturbances during training for push recovery.")
    parser.add_argument("--randomize-domain", action="store_true", default=False,
                        help="Enable full domain randomization (IMU bias, latency queue, torque scale).")
    parser.add_argument("--randomize-imu", action="store_true", default=False,
                        help="Randomize IMU pitch & gyro rate bias per episode.")
    parser.add_argument("--randomize-torque", action="store_true", default=False,
                        help="Randomize motor torque scale per episode.")
    parser.add_argument("--randomize-latency", action="store_true", default=False,
                        help="Randomize action latency buffer delay (0-2 steps / 0-20ms).")
    parser.add_argument("--resume", action="store_true",
                        help="Load trained_models/ppo_balance_latest.zip and continue training.")
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Training Configuration ===")
    print(f"Total Timesteps:     {args.timesteps:,}")
    print(f"Parallel Envs:       {args.n_envs}")
    print(f"Randomize Speed:     {args.randomize_commands} (tracks dynamic target_vel)")
    print(f"Push Disturbances:   {args.enable_pushes}")
    print(f"Domain Randomize:    {args.randomize_domain or args.randomize_imu or args.randomize_torque or args.randomize_latency}")
    print(f"  ├─ IMU Bias:       {args.randomize_domain or args.randomize_imu}")
    print(f"  ├─ Torque Scale:   {args.randomize_domain or args.randomize_torque}")
    print(f"  └─ Action Latency: {args.randomize_domain or args.randomize_latency}")
    print(f"Resume:              {args.resume}")
    print("==============================\n")

    env_fn = make_env(
        randomize_commands=args.randomize_commands,
        enable_pushes=args.enable_pushes,
        randomize_domain=args.randomize_domain,
        randomize_imu=args.randomize_imu,
        randomize_torque=args.randomize_torque,
        randomize_latency=args.randomize_latency,
    )
    vec_env = make_vec_env(env_fn, n_envs=args.n_envs)

    latest_path = MODEL_DIR / "ppo_balance_latest.zip"

    if args.resume and latest_path.exists():
        print(f"Resuming from {latest_path}")
        model = PPO.load(str(latest_path), env=vec_env, tensorboard_log=str(LOG_DIR))
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            n_steps=2048,
            batch_size=256,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.001,
            tensorboard_log=str(LOG_DIR),
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000 // args.n_envs, 1),
        save_path=str(MODEL_DIR / "checkpoints"),
        name_prefix="ppo_balance",
    )

    model.learn(
        total_timesteps=args.timesteps,
        callback=checkpoint_callback,
        reset_num_timesteps=not args.resume,
        tb_log_name="ppo_balance",
    )

    model.save(str(latest_path))
    print(f"\nTraining done. Saved to {latest_path}")
    print(f"View training curves with:\n  tensorboard --logdir {LOG_DIR}")


if __name__ == "__main__":
    main()