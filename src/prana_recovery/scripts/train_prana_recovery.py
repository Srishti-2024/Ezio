#!/usr/bin/env python3
"""Train a PPO recovery policy on procedural Prana terrain."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from prana_recovery.env import PranaRecoveryEnv
from prana_recovery.terrain import TerrainConfig
from prana_recovery.training_viewer import TrainingMosaicCallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", help="PPO device: auto, cpu, cuda, or cuda:0")
    parser.add_argument("--output", type=Path, default=Path("runs/prana_recovery"))
    parser.add_argument(
        "--terrain-height-range",
        type=float,
        default=0.65,
        help="Vertical terrain relief in metres; higher values produce steeper terrain (default: 0.65)",
    )
    parser.add_argument(
        "--spawn-clearance",
        type=float,
        default=1.15,
        help="Initial rover-root height above sampled terrain in metres (default: 1.15)",
    )
    parser.add_argument("--render", action="store_true", help="Show selected real PPO workers in one live grid window")
    parser.add_argument("--render-envs", type=int, default=4, help="Maximum workers to display with --render")
    parser.add_argument("--render-every", type=int, default=25, help="Render every N vector-environment steps")
    return parser.parse_args()


def make_env(seed: int, terrain_config: TerrainConfig, spawn_clearance_m: float):
    def factory() -> PranaRecoveryEnv:
        return PranaRecoveryEnv(
            terrain_config=terrain_config,
            spawn_clearance_m=spawn_clearance_m,
            seed=seed,
        )
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
    if args.render_envs < 1 or args.render_every < 1:
        raise SystemExit("--render-envs and --render-every must both be positive.")
    if args.terrain_height_range <= 0.0 or args.spawn_clearance <= 0.0:
        raise SystemExit("--terrain-height-range and --spawn-clearance must both be positive.")
    terrain_config = TerrainConfig(height_range_m=args.terrain_height_range)
    environment = SubprocVecEnv(
        [make_env(args.seed + index, terrain_config, args.spawn_clearance) for index in range(args.num_envs)]
    )
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
    callback = None
    if args.render:
        callback = TrainingMosaicCallback(
            every_steps=args.render_every,
            worker_count=min(args.render_envs, args.num_envs),
        )
    try:
        model.learn(total_timesteps=args.timesteps, progress_bar=True, callback=callback)
        model.save(str(args.output / "ppo_prana_recovery"))
    finally:
        environment.close()


if __name__ == "__main__":
    main()
