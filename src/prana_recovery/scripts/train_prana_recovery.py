#!/usr/bin/env python3
"""Train a PPO recovery policy on procedural Prana terrain."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from prana_recovery.env import PranaRecoveryEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", help="PPO device: auto, cpu, cuda, or cuda:0")
    parser.add_argument("--output", type=Path, default=Path("runs/prana_recovery"))
    return parser.parse_args()


def make_env(seed: int):
    def factory() -> PranaRecoveryEnv:
        return PranaRecoveryEnv(seed=seed)
    return factory


def main() -> None:
    args = parse_args()
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import SubprocVecEnv
    except ImportError as error:
        raise SystemExit(
            "Stable-Baselines3 is required for training. Install the project requirements with "
            "`python3 -m pip install -r requirements-training.txt`."
        ) from error

    args.output.mkdir(parents=True, exist_ok=True)
    environment = SubprocVecEnv([make_env(args.seed + index) for index in range(args.num_envs)])
    model = PPO(
        "MlpPolicy",
        environment,
        verbose=1,
        seed=args.seed,
        device=args.device,
        tensorboard_log=str(args.output / "tensorboard"),
        n_steps=1024,
        batch_size=256,
    )
    try:
        model.learn(total_timesteps=args.timesteps, progress_bar=True)
        model.save(str(args.output / "ppo_prana_recovery"))
    finally:
        environment.close()


if __name__ == "__main__":
    main()
