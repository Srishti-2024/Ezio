#!/usr/bin/env python3
"""Train a direct-motor PPO policy to escape a smooth parabolic bowl."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from prana_recovery.bowl_env import BowlConfig, BowlRecoveryEnv
from prana_recovery.training_viewer import TrainingMosaicCallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=Path("runs/prana_bowl_recovery"))
    parser.add_argument("--resume", type=Path, help="Continue a bowl checkpoint; timesteps are additional.")
    parser.add_argument("--bowl-depth", type=float, default=1.4, help="Rim height above bowl floor in metres")
    parser.add_argument("--bowl-radius", type=float, default=2.5, help="Bowl rim radius in metres")
    parser.add_argument("--spawn-radius", type=float, default=0.25, help="Maximum centre-start jitter in metres")
    parser.add_argument("--escape-margin", type=float, default=0.30, help="Metres beyond rim required for success")
    parser.add_argument("--max-episode-steps", type=int, default=400)
    parser.add_argument("--history-frames", type=int, default=4, help="Must remain compatible with later ditch transfer")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render-envs", type=int, default=4)
    parser.add_argument("--render-every", type=int, default=25)
    parser.add_argument("--checkpoint-every", type=int, default=100_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CheckpointCallback
        from stable_baselines3.common.vec_env import SubprocVecEnv
        from prana_recovery.policy import RoverTerrainExtractor
    except ImportError as error:
        raise SystemExit("Install src/prana_recovery/requirements-training.txt first.") from error
    if min(args.timesteps, args.num_envs, args.max_episode_steps, args.history_frames, args.render_envs, args.render_every, args.checkpoint_every) < 1:
        raise SystemExit("step counts and environment counts must be positive.")
    if args.bowl_depth <= 0.0 or args.bowl_radius <= 0.0 or args.spawn_radius < 0.0 or args.escape_margin < 0.0:
        raise SystemExit("Bowl depth/radius must be positive; spawn radius and escape margin cannot be negative.")
    if args.resume is not None and not (args.resume.exists() or args.resume.with_suffix(".zip").exists()):
        raise SystemExit(f"Could not find resume model: {args.resume}")
    args.output.mkdir(parents=True, exist_ok=True)
    bowl_config = BowlConfig(
        depth_m=args.bowl_depth,
        rim_radius_m=args.bowl_radius,
        spawn_radius_m=args.spawn_radius,
        escape_margin_m=args.escape_margin,
    )

    def make_env(worker_seed: int):
        def factory():
            return BowlRecoveryEnv(
                bowl_config=bowl_config,
                max_episode_steps=args.max_episode_steps,
                history_frames=args.history_frames,
                seed=worker_seed,
            )
        return factory

    environment = SubprocVecEnv([make_env(args.seed + index) for index in range(args.num_envs)])
    tensorboard_log = str(args.output / "tensorboard")
    if args.resume is None:
        model = PPO(
            "MultiInputPolicy", environment, verbose=1, seed=args.seed, device=args.device,
            tensorboard_log=tensorboard_log, n_steps=1024, batch_size=256,
            learning_rate=1e-4, n_epochs=5, target_kl=0.04, ent_coef=0.005,
            policy_kwargs={"features_extractor_class": RoverTerrainExtractor, "log_std_init": -0.5},
        )
    else:
        print(f"Resuming bowl policy from {args.resume}")
        model = PPO.load(args.resume, env=environment, device=args.device, tensorboard_log=tensorboard_log, verbose=1)
    callbacks = [CheckpointCallback(save_freq=max(1, args.checkpoint_every // args.num_envs), save_path=str(args.output / "checkpoints"), name_prefix="ppo_prana_bowl")]
    if args.render:
        callbacks.append(TrainingMosaicCallback(every_steps=args.render_every, worker_count=min(args.render_envs, args.num_envs)))
    try:
        model.learn(total_timesteps=args.timesteps, reset_num_timesteps=args.resume is None, callback=callbacks)
    except KeyboardInterrupt:
        model.save(args.output / "ppo_prana_bowl_interrupted")
        print("Interrupted: saved ppo_prana_bowl_interrupted.")
    else:
        model.save(args.output / "ppo_prana_bowl")
    finally:
        environment.close()


if __name__ == "__main__":
    main()
