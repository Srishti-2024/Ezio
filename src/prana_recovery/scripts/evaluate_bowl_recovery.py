#!/usr/bin/env python3
"""Evaluate a bowl-curriculum policy with optional dual-view MP4 recording."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
import sys
import time

import numpy as np

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
    parser.add_argument("--surface-friction", type=float, default=1.0)
    parser.add_argument("--max-wheel-speed", type=float, default=6.5)
    parser.add_argument("--spawn-radius", type=float, default=0.25)
    parser.add_argument("--escape-margin", type=float, default=0.30)
    parser.add_argument("--max-episode-steps", type=int, default=400)
    parser.add_argument("--history-frames", type=int, default=4)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--mujoco-viewer", action="store_true", help="Open MuJoCo's native interactive free-camera viewer")
    parser.add_argument("--render-every", type=int, default=1)
    parser.add_argument("--record-videos", action="store_true")
    parser.add_argument("--alternate-view", action="store_true")
    parser.add_argument(
        "--episode-details",
        action="store_true",
        help="Print maximum/final bowl radius, terrain and base height, path length, and speed for every episode",
    )
    parser.add_argument(
        "--position",
        action="store_true",
        help="Print final 6D base pose plus velocity, steering, wheel contacts, and last motor action for every episode",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from stable_baselines3 import PPO
    except ImportError as error:
        raise SystemExit("Install src/prana_recovery/requirements-training.txt first.") from error
    if args.episodes < 1 or args.render_every < 1 or args.history_frames < 1 or args.max_episode_steps < 1:
        raise SystemExit("episode, render, history, and maximum-step counts must be positive.")
    if args.render and args.mujoco_viewer:
        raise SystemExit("Choose either --render or --mujoco-viewer, not both.")
    config = BowlConfig(
        depth_m=args.bowl_depth, rim_radius_m=args.bowl_radius, surface_friction=args.surface_friction,
        spawn_radius_m=args.spawn_radius, escape_margin_m=args.escape_margin,
    )
    environment = BowlRecoveryEnv(
        bowl_config=config,
        max_episode_steps=args.max_episode_steps,
        history_frames=args.history_frames,
        max_wheel_linear_speed_m_s=args.max_wheel_speed,
        seed=args.seed,
    )
    policy = PPO.load(args.model)
    expected_state = (36 * args.history_frames,)
    expected_terrain = (2 * args.history_frames, 32, 32)
    if policy.observation_space.spaces["state"].shape != expected_state or policy.observation_space.spaces["terrain"].shape != expected_terrain:
        raise SystemExit("Checkpoint observation shape does not match this bowl configuration/history length.")
    viewer = EvaluationFrameViewer(alternate_view=args.alternate_view) if args.render else None
    mujoco_viewer = None
    if args.mujoco_viewer:
        import mujoco.viewer

        mujoco_viewer = mujoco.viewer
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
            initial_position = environment.data.body("prana_mark2_v2").xpos.copy()
            previous_xy = initial_position[:2].copy()
            path_length_m = 0.0
            max_radius_m = info["bowl_radius_m"]
            max_terrain_height_m = environment.bowl.height_at(*previous_xy)
            max_base_height_m = float(initial_position[2])
            max_planar_speed_m_s = 0.0
            try:
                viewer_context = (
                    mujoco_viewer.launch_passive(environment.model, environment.data)
                    if mujoco_viewer is not None
                    else nullcontext(None)
                )
                with viewer_context as native_viewer:
                    while True:
                        if native_viewer is not None and not native_viewer.is_running():
                            raise SystemExit("MuJoCo viewer was closed; stopping evaluation.")
                        started_at = time.monotonic()
                        action, _ = policy.predict(observation, deterministic=True)
                        observation, reward, terminated, truncated, info = environment.step(action)
                        episode_return += reward
                        step += 1
                        position = environment.data.body("prana_mark2_v2").xpos.copy()
                        path_length_m += float(((position[:2] - previous_xy) ** 2).sum() ** 0.5)
                        previous_xy = position[:2]
                        max_radius_m = max(max_radius_m, info["bowl_radius_m"])
                        max_terrain_height_m = max(max_terrain_height_m, environment.bowl.height_at(*position[:2]))
                        max_base_height_m = max(max_base_height_m, float(position[2]))
                        max_planar_speed_m_s = max(max_planar_speed_m_s, environment._planar_speed_m_s())
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
                        if native_viewer is not None:
                            native_viewer.sync()
                            time.sleep(max(0.0, environment.control_dt_s - (time.monotonic() - started_at)))
                        if terminated or truncated:
                            break

                    # Unlike the regular evaluator, keep this viewer attached
                    # after the terminal state so the final pendulum/tip pose
                    # can be examined with MuJoCo's free camera.
                    if native_viewer is not None and native_viewer.is_running():
                        print("Episode finished; native viewer remains open. Close it to continue.")
                        while native_viewer.is_running():
                            native_viewer.sync()
                            time.sleep(0.02)
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
            final_position = environment.data.body("prana_mark2_v2").xpos.copy()
            final_terrain_height_m = environment.bowl.height_at(*final_position[:2])
            if args.episode_details:
                print(
                    f"  details: steps={step} duration={step * environment.control_dt_s:.2f}s "
                    f"radius_final={info['bowl_radius_m']:.3f}m radius_max={max_radius_m:.3f}m "
                    f"terrain_height_final={final_terrain_height_m:.3f}m terrain_height_max={max_terrain_height_m:.3f}m "
                    f"base_z_final={final_position[2]:.3f}m base_z_max={max_base_height_m:.3f}m "
                    f"path={path_length_m:.3f}m speed_max={max_planar_speed_m_s:.3f}m/s "
                    f"tipped={info['tipped']} body_contacts={info['body_contacts']}"
                )
            if args.position:
                roll, pitch = environment._roll_pitch()
                yaw = environment._yaw()
                velocity = environment.data.qvel[environment._free_joint_dofadr : environment._free_joint_dofadr + 3]
                angular_velocity = environment.data.qvel[environment._free_joint_dofadr + 3 : environment._free_joint_dofadr + 6]
                print(
                    "  pose: "
                    f"xyz_m={np.round(final_position, 3).tolist()} "
                    f"rpy_deg={np.round(np.degrees([roll, pitch, yaw]), 2).tolist()} "
                    f"linear_velocity_world_m_s={np.round(velocity, 3).tolist()} "
                    f"angular_velocity_world_rad_s={np.round(angular_velocity, 3).tolist()}"
                )
                print(
                    f"  controls: steering_rad={np.round([environment.data.qpos[environment.model.joint(name).qposadr[0]] for name in ('bracker_lf_joint', 'bracket_rf_joint', 'bracket_lb_joint', 'bracket_rb_joint')], 3).tolist()} "
                    f"wheel_contacts={environment._wheel_contact_mask().astype(int).tolist()} "
                    f"last_action={np.round(environment._last_action, 3).tolist()}"
                )
    finally:
        if viewer is not None:
            viewer.close()
        environment.close()
    print(f"success_rate={successes / args.episodes:.1%} mean_return={sum(returns) / len(returns):.2f}")


if __name__ == "__main__":
    main()
