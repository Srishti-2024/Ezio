#!/usr/bin/env python3
"""Evaluate a bowl-curriculum policy with optional dual-view MP4 recording."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from evaluate_prana_recovery import Mp4Recorder
from prana_recovery.bowl_env import BowlConfig, BowlRecoveryEnv
from prana_recovery.training_viewer import EvaluationFrameViewer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--bowl-depth", type=float, default=1.4)
    parser.add_argument("--bowl-radius", type=float, default=2.5)
    parser.add_argument("--spawn-radius", type=float, default=0.25)
    parser.add_argument("--escape-margin", type=float, default=0.30)
    parser.add_argument("--max-episode-steps", type=int, default=400)
    parser.add_argument("--history-frames", type=int, default=4)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render-every", type=int, default=1)
    parser.add_argument("--record-videos", action="store_true")
    parser.add_argument("--alternate-view", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from stable_baselines3 import PPO
    except ImportError as error:
        raise SystemExit("Install src/prana_recovery/requirements-training.txt first.") from error
    if args.episodes < 1 or args.render_every < 1 or args.history_frames < 1 or args.max_episode_steps < 1:
        raise SystemExit("episode, render, history, and maximum-step counts must be positive.")
    config = BowlConfig(
        depth_m=args.bowl_depth, rim_radius_m=args.bowl_radius,
        spawn_radius_m=args.spawn_radius, escape_margin_m=args.escape_margin,
    )
    environment = BowlRecoveryEnv(bowl_config=config, max_episode_steps=args.max_episode_steps, history_frames=args.history_frames, seed=args.seed)
    policy = PPO.load(args.model)
    expected_state = (36 * args.history_frames,)
    expected_terrain = (2 * args.history_frames, 32, 32)
    if policy.observation_space.spaces["state"].shape != expected_state or policy.observation_space.spaces["terrain"].shape != expected_terrain:
        raise SystemExit("Checkpoint observation shape does not match this bowl configuration/history length.")
    viewer = EvaluationFrameViewer(alternate_view=args.alternate_view) if args.render else None
    video_directory = args.model.with_suffix("").parent / "bowl_evaluation_videos"
    model_stem = args.model.with_suffix("").name
    successes = 0
    returns: list[float] = []
    try:
        for episode in range(args.episodes):
            observation, info = environment.reset(seed=args.seed + episode)
            primary_recorder = Mp4Recorder(video_directory / f"{model_stem}_episode_{episode:03d}_primary.mp4", fps=20.0) if args.record_videos else None
            alternate_recorder = Mp4Recorder(video_directory / f"{model_stem}_episode_{episode:03d}_opposite_overhead.mp4", fps=20.0) if args.record_videos and args.alternate_view else None
            episode_return = 0.0
            step = 0
            try:
                while True:
                    action, _ = policy.predict(observation, deterministic=True)
                    observation, reward, terminated, truncated, info = environment.step(action)
                    episode_return += reward
                    step += 1
                    if viewer is not None or primary_recorder is not None:
                        primary = environment.render()
                        alternate = environment.render_camera(distance=4.2, azimuth=315.0, elevation=-62.0) if args.alternate_view else None
                        if primary_recorder is not None:
                            primary_recorder.write(primary)
                        if alternate_recorder is not None:
                            assert alternate is not None
                            alternate_recorder.write(alternate)
                        if viewer is not None and step % args.render_every == 0:
                            viewer.update(primary, alternate_frame=alternate, episode=episode, step=step)
                    if terminated or truncated:
                        break
            finally:
                if primary_recorder is not None:
                    primary_recorder.close()
                if alternate_recorder is not None:
                    alternate_recorder.close()
            successes += int(info["success"])
            returns.append(episode_return)
            print(
                f"episode={episode:03d} seed={info['terrain_seed']} slope={info['difficulty']:.2f} "
                f"radius={info['bowl_radius_m']:.2f}/{info['bowl_rim_radius_m']:.2f} "
                f"success={info['success']} return={episode_return:.2f}"
            )
    finally:
        if viewer is not None:
            viewer.close()
        environment.close()
    print(f"success_rate={successes / args.episodes:.1%} mean_return={sum(returns) / len(returns):.2f}")


if __name__ == "__main__":
    main()
