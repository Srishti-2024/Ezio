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
        "--spawn-contact-clearance",
        "--spawn-clearance",
        dest="spawn_contact_clearance",
        type=float,
        default=0.015,
        help="Wheel-to-terrain clearance at spawn in metres (default: 0.015)",
    )
    parser.add_argument("--render", action="store_true", help="Show selected real PPO workers in one live grid window")
    parser.add_argument("--render-envs", type=int, default=4, help="Maximum workers to display with --render")
    parser.add_argument("--render-every", type=int, default=25, help="Render every N vector-environment steps")
    parser.add_argument("--history-frames", type=int, default=4, help="Past camera/state frames per observation")
    parser.add_argument("--precheck-steps", type=int, default=15, help="Control steps in each initial mobility probe")
    parser.add_argument("--precheck-translation", type=float, default=0.30, help="Metres of probe movement that rejects a spawn")
    parser.add_argument("--mobility-window-steps", type=int, default=20, help="Consecutive control steps required for recovery")
    parser.add_argument("--mobility-displacement", type=float, default=0.50, help="Net metres required inside recovery window")
    parser.add_argument("--mobility-path", type=float, default=0.60, help="Path metres required inside recovery window")
    return parser.parse_args()


def make_env(seed: int, terrain_config: TerrainConfig, spawn_contact_clearance_m: float, recovery_options: dict):
    def factory() -> PranaRecoveryEnv:
        return PranaRecoveryEnv(
            terrain_config=terrain_config,
            spawn_contact_clearance_m=spawn_contact_clearance_m,
            seed=seed,
            **recovery_options,
        )
    return factory


def main() -> None:
    args = parse_args()
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import SubprocVecEnv
        from prana_recovery.policy import RoverTerrainExtractor
    except ImportError as error:
        raise SystemExit(
            "Stable-Baselines3 is required for training. Install the project requirements with "
            "`python3 -m pip install -r requirements-training.txt`."
        ) from error

    args.output.mkdir(parents=True, exist_ok=True)
    if args.render_envs < 1 or args.render_every < 1:
        raise SystemExit("--render-envs and --render-every must both be positive.")
    if args.terrain_height_range <= 0.0 or args.spawn_contact_clearance < 0.0:
        raise SystemExit("--terrain-height-range must be positive and --spawn-contact-clearance cannot be negative.")
    if (
        args.history_frames < 1
        or args.precheck_steps < 1
        or args.mobility_window_steps < 1
        or args.precheck_translation <= 0.0
        or args.mobility_displacement <= 0.0
        or args.mobility_path <= 0.0
    ):
        raise SystemExit("Recovery history, step counts, and distance thresholds must all be positive.")
    terrain_config = TerrainConfig(height_range_m=args.terrain_height_range)
    recovery_options = {
        "history_frames": args.history_frames,
        "precheck_control_steps": args.precheck_steps,
        "precheck_translation_m": args.precheck_translation,
        "mobility_window_steps": args.mobility_window_steps,
        "mobility_displacement_m": args.mobility_displacement,
        "mobility_path_m": args.mobility_path,
    }
    environment = SubprocVecEnv(
        [
            make_env(args.seed + index, terrain_config, args.spawn_contact_clearance, recovery_options)
            for index in range(args.num_envs)
        ]
    )
    model = PPO(
        "MultiInputPolicy",
        environment,
        verbose=1,
        seed=args.seed,
        device=args.device,
        tensorboard_log=str(args.output / "tensorboard"),
        n_steps=1024,
        batch_size=256,
        policy_kwargs={"features_extractor_class": RoverTerrainExtractor},
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
