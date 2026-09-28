import shutil
from pathlib import Path

checkpoints_dir = Path("/home/abok-omen/mujoco-rl/projects/balancia_robot/trained_models/checkpoints")

if checkpoints_dir.exists():
    for item in checkpoints_dir.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)
    print(f"Cleaned all checkpoint files in {checkpoints_dir}. Only ppo_balance_latest.zip remains.")
