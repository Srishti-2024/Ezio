#!/usr/bin/env python3
"""Physics experiment: drive all four Prana wheels straight out of a bowl.

This is intentionally not an RL policy.  It applies the same normalized
direct-wheel action to every drive actuator, with all steering targets set to
zero, so it answers the simple question: can conventional full-drive climbing
cross this bowl without tipping?
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
from prana_recovery.bowl_env import BowlConfig, BowlRecoveryEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bowl-depth", type=float, default=1.6)
    parser.add_argument("--bowl-radius", type=float, default=2.5)
    parser.add_argument("--surface-friction", type=float, default=1.0)
    parser.add_argument("--max-wheel-speed", type=float, default=6.5, help="Wheel linear-speed target at drive command 1 (default: 6.5 m/s)")
    parser.add_argument(
        "--drive-command",
        type=float,
        default=1.0,
        help="Normalized common wheel command before slowdown in [-1, 1] (default: 1.0)",
    )
    parser.add_argument(
        "--rim-drive-command",
        type=float,
        default=0.35,
        help="Normalized common wheel command at/after the rim (default: 0.35)",
    )
    parser.add_argument(
        "--slowdown-start-radius",
        type=float,
        help="Start smooth deceleration at this bowl-centre radius; default: 70%% of bowl radius",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=400,
        help="Headless control-step limit; viewer mode keeps simulating until its window is closed",
    )
    parser.add_argument(
        "--steering-kp",
        type=float,
        default=500.0,
        help="Position gain holding each steering joint at zero (default: 500; model default: 100)",
    )
    parser.add_argument(
        "--steering-kv",
        type=float,
        default=60.0,
        help="Damping gain for the strengthened steering hold (default: 60; model default: 20)",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-viewer", action="store_true", help="Run headless, printing diagnostics only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bowl_depth <= 0.0 or args.bowl_radius <= 0.0 or args.surface_friction <= 0.0 or args.max_wheel_speed <= 0.0 or args.steps < 1 or args.steering_kp <= 0.0 or args.steering_kv < 0.0:
        raise SystemExit("Bowl dimensions, steps, and steering gains must be non-negative/positive as appropriate.")
    if not -1.0 <= args.drive_command <= 1.0 or not -1.0 <= args.rim_drive_command <= 1.0:
        raise SystemExit("drive commands must be within [-1, 1].")
    slowdown_start_radius = args.slowdown_start_radius
    if slowdown_start_radius is None:
        slowdown_start_radius = 0.70 * args.bowl_radius
    if not 0.0 <= slowdown_start_radius < args.bowl_radius:
        raise SystemExit("--slowdown-start-radius must be non-negative and smaller than --bowl-radius.")

    environment = BowlRecoveryEnv(
        bowl_config=BowlConfig(
            depth_m=args.bowl_depth,
            rim_radius_m=args.bowl_radius,
            surface_friction=args.surface_friction,
            # At the centre every heading is radially outward, so this avoids
            # accidentally testing a diagonal/cross-slope initial path.
            spawn_radius_m=0.0,
        ),
        max_episode_steps=args.steps,
        max_wheel_linear_speed_m_s=args.max_wheel_speed,
        seed=args.seed,
    )
    # Direct action order is drive FL/FR/LB/RB, then steering FL/FR/LB/RB.
    # Zero steering means every wheel remains aligned with rover forward.
    action = np.zeros(8, dtype=np.float32)
    viewer = None
    try:
        environment.reset(seed=args.seed)
        # A MuJoCo position actuator produces kp * (target - position) - kv *
        # joint_velocity. The compiled rover normally uses kp=100, kv=20.
        # This diagnostic keeps steering truly straight under tyre load without
        # changing the rover description or either RL environment.
        for actuator_id in environment._steering_ids:
            environment.model.actuator_gainprm[actuator_id, 0] = args.steering_kp
            environment.model.actuator_biasprm[actuator_id, 1] = -args.steering_kp
            environment.model.actuator_biasprm[actuator_id, 2] = -args.steering_kv
        print(
            f"straight-bowl test: depth={args.bowl_depth:.2f} m radius={args.bowl_radius:.2f} m friction={args.surface_friction:.2f} "
            f"drive={args.drive_command:.2f}->{args.rim_drive_command:.2f} max_speed={args.max_wheel_speed:.2f}m/s "
            f"slowdown={slowdown_start_radius:.2f}m->{args.bowl_radius:.2f}m "
            f"steering_kp={args.steering_kp:.0f} steering_kv={args.steering_kv:.0f}"
        )
        if not args.no_viewer:
            import mujoco.viewer

            viewer = mujoco.viewer.launch_passive(environment.model, environment.data)
            viewer.cam.lookat[:] = environment.data.body("prana_mark2_v2").xpos
            viewer.cam.distance = 5.0
            viewer.cam.azimuth = 135.0
            viewer.cam.elevation = -32.0

        step = 0
        reported_terminal = False
        while viewer is None or viewer.is_running():
            # In headless mode retain a finite run; interactive mode is for
            # examining the post-tip physics, so it ends only when the user
            # closes the MuJoCo window.
            if viewer is None and step >= args.steps:
                print("finished: headless step limit")
                break
            radius = environment._escape_distance()
            # Cubic smoothstep keeps both the command and its rate of change
            # continuous: high speed in the bowl floor, gentle rim crossing.
            interpolation = np.clip(
                (radius - slowdown_start_radius) / (args.bowl_radius - slowdown_start_radius), 0.0, 1.0
            )
            interpolation = interpolation * interpolation * (3.0 - 2.0 * interpolation)
            drive_command = (1.0 - interpolation) * args.drive_command + interpolation * args.rim_drive_command
            action[:4] = drive_command
            started_at = time.monotonic()
            _, _, terminated, truncated, info = environment.step(action)
            step += 1
            if step == 1 or step % 10 == 0 or (terminated and not reported_terminal):
                roll, pitch = environment._roll_pitch()
                print(
                    f"step={step:03d} time={step * environment.control_dt_s:5.2f}s "
                    f"radius={info['bowl_radius_m']:.3f}m drive={drive_command:+.2f} "
                    f"speed={environment._planar_speed_m_s():.3f}m/s "
                    f"roll={np.degrees(roll):+.1f}deg pitch={np.degrees(pitch):+.1f}deg "
                    f"contacts={info['body_contacts']}"
                )
            if viewer is not None:
                viewer.sync()
                time.sleep(max(0.0, environment.control_dt_s - (time.monotonic() - started_at)))
            if (terminated or truncated) and not reported_terminal:
                outcome = "success" if info["success"] else ("tipped" if info["tipped"] else "time limit")
                print(f"terminal signal: {outcome}; continuing simulation until viewer is closed.")
                reported_terminal = True
    finally:
        if viewer is not None:
            viewer.close()
        environment.close()


if __name__ == "__main__":
    main()
