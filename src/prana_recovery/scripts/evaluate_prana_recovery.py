#!/usr/bin/env python3
"""Evaluate a saved Prana PPO policy on fixed, replayable terrain seeds."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from prana_recovery.env import PranaRecoveryEnv
from prana_recovery.terrain import TerrainConfig
from prana_recovery.training_viewer import EvaluationFrameViewer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path, help="PPO model path, with or without the .zip suffix")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=10_000, help="Evaluation seed; keep fixed for comparisons")
    parser.add_argument(
        "--terrain-height-range",
        type=float,
        default=0.65,
        help="Must match the terrain relief used during training (default: 0.65)",
    )
    parser.add_argument(
        "--spawn-clearance",
        type=float,
        default=1.15,
        help="Initial rover-root height above sampled terrain in metres (default: 1.15)",
    )
    parser.add_argument("--render", action="store_true", help="Show the live MuJoCo evaluation episode")
    parser.add_argument(
        "--render-envs",
        type=int,
        default=1,
        help="Evaluation has one environment, so this must be 1 (default: 1)",
    )
    parser.add_argument(
        "--render-every",
        type=int,
        default=1,
        help="Update the live view every N evaluation steps (default: 1)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from stable_baselines3 import PPO
    except ImportError as error:
        raise SystemExit("Install requirements-training.txt before evaluating a PPO model.") from error

    if args.episodes < 1:
        raise SystemExit("--episodes must be positive.")
    if args.render_envs != 1:
        raise SystemExit("Evaluation runs one environment, so --render-envs must be 1.")
    if args.render_every < 1:
        raise SystemExit("--render-every must be positive.")
    if args.terrain_height_range <= 0.0 or args.spawn_clearance <= 0.0:
        raise SystemExit("--terrain-height-range and --spawn-clearance must both be positive.")
    environment = PranaRecoveryEnv(
        terrain_config=TerrainConfig(height_range_m=args.terrain_height_range),
        spawn_clearance_m=args.spawn_clearance,
        seed=args.seed,
    )
    policy = PPO.load(args.model)
    viewer = EvaluationFrameViewer() if args.render else None
    successes = 0
    returns: list[float] = []
    try:
        for episode in range(args.episodes):
            observation, info = environment.reset(seed=args.seed + episode)
            episode_return = 0.0
            step = 0
            while True:
                action, _ = policy.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = environment.step(action)
                episode_return += reward
                step += 1
                if viewer is not None and step % args.render_every == 0:
                    viewer.update(environment.render(), episode=episode, step=step)
                if terminated or truncated:
                    break
            successes += int(info["success"])
            returns.append(episode_return)
            print(
                f"episode={episode:03d} seed={info['terrain_seed']} difficulty={info['difficulty']:.2f} "
                f"success={info['success']} return={episode_return:.2f}"
            )
    finally:
        if viewer is not None:
            viewer.close()
        environment.close()
    print(f"success_rate={successes / args.episodes:.1%} mean_return={sum(returns) / len(returns):.2f}")


if __name__ == "__main__":
    main()
