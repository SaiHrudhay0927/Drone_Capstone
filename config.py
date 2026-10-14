import json
from pathlib import Path

def load_config(path):
    config = json.loads(Path(path).read_text())
    if config["episodes"] < 1 or config["eval_episodes"] < 1: raise ValueError("Episode counts must be positive")
    if not config["seeds"]: raise ValueError("At least one training seed is required")
    a = config["agent"]
    if not 0 <= a["success_fraction"] <= 1: raise ValueError("Invalid success_fraction")
    if a["capacity"] < a["batch_size"] or a["temporary_capacity"] < 1: raise ValueError("Invalid replay sizes")
    return config
