#!/usr/bin/env python3
"""Interactive 4WIS teleoperation viewer for the Prana Mark 2 MuJoCo model.

Keep the terminal focused while using the keys below.  The MuJoCo window is
only a viewer; keyboard input is read from this terminal.

    u  forward + CCW        i  forward        o  forward + CW
    j  counter-clockwise    k  stop           l  right/crab
    J  left/crab (capital J)
    m  reverse + CW         ,  reverse        .  reverse + CCW
    q / Ctrl-C  quit

Each command is converted from a body-frame Twist-like command ``[vx, vy, wz]``
to four steering targets and four wheel velocity targets.  The current keymap
exposes forward/reverse, yaw, and one lateral (crab) command.  Uppercase and
lowercase keys are intentionally distinct.
"""

from __future__ import annotations

import argparse
import math
import select
import sys
import termios
import threading
import time
import tty
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


# The existing script owns the model conversion and rocker-differential
# equality-constraint setup.  Import it without modifying that test script.
DESCRIPTION_SCRIPTS = Path(__file__).resolve().parents[2] / "prana_description" / "scripts"
sys.path.insert(0, str(DESCRIPTION_SCRIPTS))
import mujoco_sim_mark2_v2 as rover_sim  # noqa: E402


# Wheel order used everywhere in this file.  These are nominal wheel contact
# positions in the base frame (X forward, Y left), calculated from rover.urdf.
WHEEL_NAMES = ("fl", "fr", "lb", "rb")
STEERING_JOINTS = {
    "fl": "bracker_lf_joint",  # URDF spelling retained by the source model
    "fr": "bracket_rf_joint",
    "lb": "bracket_lb_joint",
    "rb": "bracket_rb_joint",
}
WHEEL_POSITIONS_M = np.array(
    [
        [0.445, 0.300],   # front-left
        [0.456, -0.294],  # front-right
        [-0.360, 0.299],  # rear-left
        [-0.360, -0.294], # rear-right
    ],
    dtype=np.float64,
)
WHEEL_RADIUS_M = 0.135
MAX_WHEEL_LINEAR_SPEED_M_S = 2.0


def wrap_steering_angle(angle_rad: float, speed_m_s: float) -> tuple[float, float]:
    """Choose the equivalent wheel heading within +/- 90 degrees.

    Reversing the wheel is preferable to turning a steering bracket more than
    90 degrees.  The MuJoCo steering joints use a positive Z axis, so their
    actuator targets have the same CCW-positive sign as ``atan2``.
    """
    angle_rad = (angle_rad + math.pi) % (2.0 * math.pi) - math.pi
    if angle_rad > math.pi / 2.0:
        return angle_rad - math.pi, -speed_m_s
    if angle_rad < -math.pi / 2.0:
        return angle_rad + math.pi, -speed_m_s
    return angle_rad, speed_m_s


def body_twist_to_wheels(vx: float, vy: float, wz: float) -> tuple[np.ndarray, np.ndarray]:
    """Return steering targets (rad) and wheel targets (rad/s) for 4WIS.

    ``vx`` is forward, ``vy`` is left/crab, and positive ``wz`` is
    counter-clockwise.  The calculation mirrors the Gazebo controller supplied
    with this project, adapted to the MuJoCo joint order and signs.
    """
    steering = np.zeros(4, dtype=np.float64)
    linear_speeds = np.zeros(4, dtype=np.float64)

    for index, (px, py) in enumerate(WHEEL_POSITIONS_M):
        wheel_vx = vx - wz * py
        wheel_vy = vy + wz * px
        speed = math.hypot(wheel_vx, wheel_vy)
        angle = math.atan2(wheel_vy, wheel_vx) if speed > 1e-8 else 0.0
        steering[index], linear_speeds[index] = wrap_steering_angle(angle, speed)

    max_speed = float(np.max(np.abs(linear_speeds)))
    if max_speed > MAX_WHEEL_LINEAR_SPEED_M_S:
        linear_speeds *= MAX_WHEEL_LINEAR_SPEED_M_S / max_speed

    return steering, linear_speeds / WHEEL_RADIUS_M


class KeyboardTeleop:
    """Read single characters in a background thread without blocking physics."""

    def __init__(self, linear_speed: float, yaw_speed: float) -> None:
        self.linear_speed = linear_speed
        self.yaw_speed = yaw_speed
        self._twist = np.zeros(3, dtype=np.float64)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._terminal_state: list | None = None

    def start(self) -> None:
        if not sys.stdin.isatty():
            raise RuntimeError("Interactive teleop requires a terminal (stdin is not a TTY).")
        self._terminal_state = termios.tcgetattr(sys.stdin.fileno())
        tty.setcbreak(sys.stdin.fileno())
        self._thread = threading.Thread(target=self._read_keys, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._terminal_state is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._terminal_state)
            self._terminal_state = None

    def twist(self) -> np.ndarray:
        with self._lock:
            return self._twist.copy()

    @property
    def should_quit(self) -> bool:
        return self._stop.is_set()

    def _set_twist(self, vx: float, vy: float, wz: float) -> None:
        with self._lock:
            self._twist[:] = (vx, vy, wz)

    def _read_keys(self) -> None:
        # Standard teleop_twist_keyboard's 3x3 layout, adapted for a 4WIS
        # rover: uppercase J is a true lateral (vy) crab command. Case must
        # remain intact when reading the key below.
        commands = {
            "u": (self.linear_speed, 0.0, self.yaw_speed),
            "i": (self.linear_speed, 0.0, 0.0),
            "o": (self.linear_speed, 0.0, -self.yaw_speed),
            "J": (0.0, self.linear_speed, 0.0),
            "j": (0.0, 0.0, self.yaw_speed),
            "k": (0.0, 0.0, 0.0),
            "L": (0.0, -self.linear_speed, 0.0),
            "l": (0.0, 0.0, -self.yaw_speed),
            "m": (-self.linear_speed, 0.0, -self.yaw_speed),
            ",": (-self.linear_speed, 0.0, 0.0),
            ".": (-self.linear_speed, 0.0, self.yaw_speed),
        }
        while not self._stop.is_set():
            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not ready:
                continue
            key = sys.stdin.read(1)
            if key in ("q", "Q", "\x03"):
                self._stop.set()
            elif key in commands:
                self._set_twist(*commands[key])
                print(f"\rcommand: {key}  twist={self.twist().round(2)}      ", end="", flush=True)


def actuator_ids(model: mujoco.MjModel) -> tuple[list[int], list[int]]:
    steering_ids = []
    drive_ids = []
    for wheel in WHEEL_NAMES:
        # The front-left steering joint intentionally retains its URDF typo:
        # bracker_lf_joint.  Use the model's exact actuator name.
        steering_joint = STEERING_JOINTS[wheel]
        steering_name = f"{steering_joint}_hold"
        drive_name = f"wheel_{wheel}_joint_drive"
        steering_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, steering_name)
        drive_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, drive_name)
        if steering_id < 0 or drive_id < 0:
            raise RuntimeError(f"Missing actuator(s) for {wheel}: {steering_name}, {drive_name}")
        steering_ids.append(steering_id)
        drive_ids.append(drive_id)
    return steering_ids, drive_ids


def apply_twist(data: mujoco.MjData, steering_ids: list[int], drive_ids: list[int], twist: np.ndarray) -> None:
    steering_targets, wheel_velocity_targets = body_twist_to_wheels(*twist)
    for actuator_id, target in zip(steering_ids, steering_targets, strict=True):
        data.ctrl[actuator_id] = target
    for actuator_id, target in zip(drive_ids, wheel_velocity_targets, strict=True):
        data.ctrl[actuator_id] = target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--linear-speed", type=float, default=0.8, help="Translation command speed in m/s (default: 0.8)")
    parser.add_argument("--yaw-speed", type=float, default=0.9, help="Diagonal-command yaw rate in rad/s (default: 0.9)")
    parser.add_argument("--no-viewer", action="store_true", help="Load and validate the model without opening interactive teleop")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = Path(__file__).resolve().parents[2] / "prana_description" / "prana_mark2_v2"
    model = rover_sim.build_model(model_dir)
    data = mujoco.MjData(model)
    steering_ids, drive_ids = actuator_ids(model)
    apply_twist(data, steering_ids, drive_ids, np.zeros(3))

    if args.no_viewer:
        for _ in range(50):
            mujoco.mj_step(model, data)
        print("Prana MuJoCo teleop model loaded and stepped successfully.")
        return

    teleop = KeyboardTeleop(args.linear_speed, args.yaw_speed)
    print(
        "\nPrana MuJoCo 4WIS teleop\n"
        "  u i o   forward-left, forward, forward-right\n"
        "  j k l   counter-clockwise yaw, stop, right/crab\n"
        "  J       left/crab (uppercase)\n"
        "  m , .   reverse-right, reverse, reverse-left\n"
        "  q       quit\n"
        "Keep this terminal focused to send commands.\n"
    )
    try:
        teleop.start()
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running() and not teleop.should_quit:
                apply_twist(data, steering_ids, drive_ids, teleop.twist())
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(model.opt.timestep)
    finally:
        apply_twist(data, steering_ids, drive_ids, np.zeros(3))
        teleop.close()
        print("\nStopped.")


if __name__ == "__main__":
    main()
