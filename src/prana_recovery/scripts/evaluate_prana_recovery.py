#!/usr/bin/env python3
"""Evaluate a saved Prana PPO policy on fixed, replayable terrain seeds."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
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
        "--record-videos",
        action="store_true",
        help="Save renderer frames as MP4s in <model run>/evaluation_videos/.",
    )
    parser.add_argument(
        "--alternate-view",
        action="store_true",
        help="Add an opposite, steeper overhead rover-following camera view.",
    )
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
    parser.add_argument("--max-reset-terrain-attempts", type=int, default=20, help="Must match training (default: 20)")
    return parser.parse_args()


class Mp4Recorder:
    """Stream rendered RGB frames straight to ffmpeg as a compact MP4.

    Feeding raw frames over stdin avoids a heavyweight Python video dependency
    and avoids first storing a directory full of temporary PNG frames.
    """

    def __init__(self, path: Path, *, fps: float) -> None:
        self.path = path
        self.fps = fps
        self._process: subprocess.Popen[bytes] | None = None

    def write(self, frame) -> None:
        if frame.dtype.name != "uint8" or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Expected an HxWx3 uint8 RGB frame from MuJoCo.")
        if self._process is None:
            height, width, _ = frame.shape
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._process = subprocess.Popen(
                [
                    "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-vcodec", "rawvideo",
                    "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(self.fps), "-i", "-",
                    "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
                    str(self.path),
                ],
                stdin=subprocess.PIPE,
            )
        assert self._process.stdin is not None
        self._process.stdin.write(frame.tobytes())

    def close(self) -> None:
        if self._process is None:
            return
        assert self._process.stdin is not None
        self._process.stdin.close()
        return_code = self._process.wait()
        self._process = None
        if return_code:
            raise RuntimeError(f"ffmpeg failed while writing {self.path} (exit code {return_code}).")


def run_episode(
    environment: PranaRecoveryEnv, policy, observation, viewer=None, frame_callback=None
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
        if frame_callback is not None:
            frame_callback(step=None)
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
    if args.record_videos and shutil.which("ffmpeg") is None:
        raise SystemExit("--record-videos needs the system 'ffmpeg' command, but it was not found on PATH.")
    if args.alternate_view and not (args.render or args.record_videos):
        raise SystemExit("--alternate-view needs --render and/or --record-videos.")
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
        or args.max_reset_terrain_attempts < 1
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
        max_reset_terrain_attempts=args.max_reset_terrain_attempts,
        record_precheck_frames=args.show_precheck,
        seed=args.seed,
    )
    policy = PPO.load(args.model)
    viewer = EvaluationFrameViewer(alternate_view=args.alternate_view) if args.render else None
    mujoco_viewer_module = None
    if args.mujoco_viewer:
        import mujoco.viewer

        mujoco_viewer_module = mujoco.viewer
    successes = 0
    skipped_episodes = 0
    returns: list[float] = []
    model_stem = args.model.with_suffix("").name
    video_directory = args.model.with_suffix("").parent / "evaluation_videos"
    try:
        for episode in range(args.episodes):
            observation, info = environment.reset(seed=args.seed + episode)
            if viewer is not None and args.show_precheck:
                viewer.replay_precheck(environment.consume_precheck_frames(), episode=episode)
            primary_recorder = None
            alternate_recorder = None
            if args.record_videos:
                primary_recorder = Mp4Recorder(
                    video_directory / f"{model_stem}_episode_{episode:03d}_primary.mp4",
                    fps=1.0 / environment.control_dt_s,
                )
                if args.alternate_view:
                    alternate_recorder = Mp4Recorder(
                        video_directory / f"{model_stem}_episode_{episode:03d}_opposite_overhead.mp4",
                        fps=1.0 / environment.control_dt_s,
                    )

            def handle_frame(*, step: int | None) -> None:
                # Primary is exactly the existing Matplotlib camera: it follows
                # the base at 3.2 m, azimuth 135°, elevation -28°.
                primary = environment.render()
                alternate = None
                if args.alternate_view:
                    alternate = environment.render_camera(distance=4.2, azimuth=315.0, elevation=-62.0)
                if primary_recorder is not None:
                    primary_recorder.write(primary)
                if alternate_recorder is not None:
                    assert alternate is not None
                    alternate_recorder.write(alternate)
                if viewer is not None and step is not None and step % args.render_every == 0:
                    viewer.update(primary, alternate_frame=alternate, episode=episode, step=step)

            if mujoco_viewer_module is not None:
                # A seeded reset creates a new MuJoCo model/heightfield, so a
                # native viewer is opened per episode to stay attached to it.
                try:
                    with mujoco_viewer_module.launch_passive(environment.model, environment.data) as native_viewer:
                        episode_return, info = run_episode(
                            environment, policy, observation, native_viewer,
                            frame_callback=handle_frame if args.record_videos else None,
                        )
                finally:
                    if primary_recorder is not None:
                        primary_recorder.close()
                    if alternate_recorder is not None:
                        alternate_recorder.close()
            else:
                episode_return = 0.0
                step = 0
                try:
                    while True:
                        action, _ = policy.predict(observation, deterministic=True)
                        observation, reward, terminated, truncated, info = environment.step(action)
                        episode_return += reward
                        step += 1
                        if viewer is not None or primary_recorder is not None:
                            handle_frame(step=step)
                        if terminated or truncated:
                            break
                finally:
                    if primary_recorder is not None:
                        primary_recorder.close()
                    if alternate_recorder is not None:
                        alternate_recorder.close()
            if info["skipped"]:
                skipped_episodes += 1
            else:
                successes += int(info["success"])
                returns.append(episode_return)
            print(
                f"episode={episode:03d} seed={info['terrain_seed']} difficulty={info['difficulty']:.2f} "
                f"reset_terrains={info['reset_terrain_attempts']} "
                f"skipped={info['skipped']} success={info['success']} return={episode_return:.2f}"
            )
    finally:
        if viewer is not None:
            viewer.close()
        environment.close()
    evaluated = args.episodes - skipped_episodes
    success_rate = successes / evaluated if evaluated else 0.0
    mean_return = sum(returns) / len(returns) if returns else 0.0
    print(
        f"success_rate={success_rate:.1%} mean_return={mean_return:.2f} "
        f"evaluated={evaluated} skipped={skipped_episodes}"
    )


if __name__ == "__main__":
    main()
