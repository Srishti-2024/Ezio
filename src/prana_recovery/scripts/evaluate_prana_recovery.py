#!/usr/bin/env python3
"""Evaluate a saved Prana PPO policy on fixed, replayable terrain seeds."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

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
        "--spawn-contact-clearance",
        "--spawn-clearance",
        dest="spawn_contact_clearance",
        type=float,
        default=0.015,
        help="Wheel-to-terrain clearance at spawn in metres (default: 0.015)",
    )
    parser.add_argument("--render", action="store_true", help="Show the live MuJoCo evaluation episode")
    parser.add_argument(
        "--show-precheck",
        action="store_true",
        help="Replay accepted forward/crab/yaw stuckness probes before PPO; requires --render",
    )
    parser.add_argument(
        "--mujoco-viewer",
        action="store_true",
        help="Open MuJoCo's interactive free-camera viewer during each evaluation episode",
    )
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
    parser.add_argument("--history-frames", type=int, default=4, help="Must match training (default: 4)")
    parser.add_argument("--precheck-steering-steps", type=int, default=10, help="Must match training (default: 10)")
    parser.add_argument("--precheck-steps", type=int, default=30, help="Must match training (default: 30)")
    parser.add_argument("--precheck-translation", type=float, default=0.30, help="Must match training (default: 0.30)")
    parser.add_argument("--mobility-window-steps", type=int, default=20, help="Must match training (default: 20)")
    parser.add_argument("--mobility-displacement", type=float, default=0.50, help="Must match training (default: 0.50)")
    parser.add_argument("--mobility-path", type=float, default=0.60, help="Must match training (default: 0.60)")
    return parser.parse_args()


def run_episode(
    environment: PranaRecoveryEnv, policy, observation, viewer=None
) -> tuple[float, dict]:
    """Run one policy episode, optionally synchronising a native MuJoCo viewer."""
    episode_return = 0.0
    while True:
        if viewer is not None and not viewer.is_running():
            raise SystemExit("MuJoCo viewer was closed; stopping evaluation.")
        action, _ = policy.predict(observation, deterministic=True)
        started_at = time.monotonic()
        observation, reward, terminated, truncated, info = environment.step(action)
        episode_return += reward
        if viewer is not None:
            viewer.sync()
            time.sleep(max(0.0, environment.control_dt_s - (time.monotonic() - started_at)))
        if terminated or truncated:
            return episode_return, info


def main() -> None:
    args = parse_args()
    try:
        from stable_baselines3 import PPO
    except ImportError as error:
        raise SystemExit("Install requirements-training.txt before evaluating a PPO model.") from error

    if args.episodes < 1:
        raise SystemExit("--episodes must be positive.")
    if args.render and args.mujoco_viewer:
        raise SystemExit("Choose either --render or --mujoco-viewer, not both.")
    if args.show_precheck and not args.render:
        raise SystemExit("--show-precheck requires --render; native MuJoCo viewer attaches after reset.")
    if args.render_envs != 1:
        raise SystemExit("Evaluation runs one environment, so --render-envs must be 1.")
    if args.render_every < 1:
        raise SystemExit("--render-every must be positive.")
    if args.terrain_height_range <= 0.0 or args.spawn_contact_clearance < 0.0:
        raise SystemExit("--terrain-height-range must be positive and --spawn-contact-clearance cannot be negative.")
    if (
        args.history_frames < 1
        or args.precheck_steering_steps < 1
        or args.precheck_steps < 1
        or args.mobility_window_steps < 1
        or args.precheck_translation <= 0.0
        or args.mobility_displacement <= 0.0
        or args.mobility_path <= 0.0
    ):
        raise SystemExit("Recovery history, step counts, and distance thresholds must all be positive.")
    environment = PranaRecoveryEnv(
        terrain_config=TerrainConfig(height_range_m=args.terrain_height_range),
        spawn_contact_clearance_m=args.spawn_contact_clearance,
        history_frames=args.history_frames,
        precheck_steering_steps=args.precheck_steering_steps,
        precheck_control_steps=args.precheck_steps,
        precheck_translation_m=args.precheck_translation,
        mobility_window_steps=args.mobility_window_steps,
        mobility_displacement_m=args.mobility_displacement,
        mobility_path_m=args.mobility_path,
        record_precheck_frames=args.show_precheck,
        seed=args.seed,
    )
    policy = PPO.load(args.model)
    viewer = EvaluationFrameViewer() if args.render else None
    mujoco_viewer_module = None
    if args.mujoco_viewer:
        import mujoco.viewer

        mujoco_viewer_module = mujoco.viewer
    successes = 0
    returns: list[float] = []
    try:
        for episode in range(args.episodes):
            observation, info = environment.reset(seed=args.seed + episode)
            if viewer is not None and args.show_precheck:
                viewer.replay_precheck(environment.consume_precheck_frames(), episode=episode)
            if mujoco_viewer_module is not None:
                # A seeded reset creates a new MuJoCo model/heightfield, so a
                # native viewer is opened per episode to stay attached to it.
                with mujoco_viewer_module.launch_passive(environment.model, environment.data) as native_viewer:
                    episode_return, info = run_episode(environment, policy, observation, native_viewer)
            else:
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
