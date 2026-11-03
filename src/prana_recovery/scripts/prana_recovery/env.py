"""Gymnasium environment for procedural Prana recovery training."""

from __future__ import annotations

import math
import sys
import tempfile
from collections import deque
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from .scenarios import RecoveryScenario, ScenarioSampler
from .terrain import ProceduralTerrain, TerrainConfig


# This package is separate from the existing drive package, so these are the
# only location changes from the original copied environment.
SOURCE_ROOT = Path(__file__).resolve().parents[3]
DRIVE_SCRIPTS = SOURCE_ROOT / "prana_mujoco_drive" / "scripts"
DESCRIPTION_SCRIPTS = SOURCE_ROOT / "prana_description" / "scripts"

sys.path.insert(0, str(DRIVE_SCRIPTS))
sys.path.insert(0, str(DESCRIPTION_SCRIPTS))
from prana_mujoco_teleop import (  # noqa: E402
    WHEEL_NAMES,
    WHEEL_POSITIONS_M,
    WHEEL_RADIUS_M,
    actuator_ids,
    body_twist_to_wheels,
)
import mujoco_sim_mark2_v2 as rover_sim  # noqa: E402


class PranaRecoveryEnv(gym.Env[dict[str, np.ndarray], np.ndarray]):
    """Recover a Prana rover from difficult positions on seeded heightfields.

    Actions are normalized direct wheel commands in this fixed order:
    ``[drive_fl, drive_fr, drive_lb, drive_rb, steer_fl, steer_fr, steer_lb,
    steer_rb]``.  The policy controls each drive and steering actuator itself;
    it does not receive the base-twist kinematic controller used by teleop.
    """

    metadata = {"render_modes": ["rgb_array"]}

    # This is a deliberately downsampled ZED-like depth view.  The raw ray
    # intersections form a 3-D point cloud which is rasterised below into a
    # compact local elevation map for PPO.
    CAMERA_HORIZONTAL_FOV_RAD = math.radians(65.0)
    CAMERA_WIDTH = 48
    CAMERA_HEIGHT = 30
    CAMERA_RANGE_M = 4.0
    CAMERA_OFFSET_BODY_M = np.array([0.45, 0.0, 0.45], dtype=np.float64)
    TERRAIN_MAP_CELLS = 32
    TERRAIN_MAP_FORWARD_M = 4.0
    TERRAIN_MAP_HALF_WIDTH_M = 2.0
    TERRAIN_MAP_HEIGHT_SCALE_M = 1.5
    MAX_WHEEL_LINEAR_SPEED_M_S = 2.0
    MAX_STEERING_ANGLE_RAD = math.pi / 2.0
    PROPRIOCEPTIVE_STATE_SIZE = 16
    ACTION_SIZE = 8

    def __init__(
        self,
        terrain_config: TerrainConfig | None = None,
        terrain_episodes: int = 8,
        control_dt_s: float = 0.05,
        max_episode_steps: int = 300,
        spawn_contact_clearance_m: float = 0.015,
        settle_max_steps: int = 750,
        settle_stable_steps: int = 50,
        spawn_attempts: int = 10,
        max_reset_terrain_attempts: int = 20,
        spawn_max_body_contacts: int = 8,
        history_frames: int = 4,
        precheck_steering_steps: int = 10,
        precheck_control_steps: int = 30,
        precheck_translation_m: float = 0.30,
        mobility_window_steps: int = 20,
        mobility_displacement_m: float = 0.50,
        mobility_path_m: float = 0.60,
        record_precheck_frames: bool = False,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.terrain_config = terrain_config or TerrainConfig()
        self.terrain_episodes = terrain_episodes
        self.control_dt_s = control_dt_s
        self.max_episode_steps = max_episode_steps
        if spawn_contact_clearance_m < 0.0:
            raise ValueError("spawn_contact_clearance_m must not be negative.")
        if (
            settle_max_steps < 1
            or settle_stable_steps < 1
            or spawn_attempts < 1
            or max_reset_terrain_attempts < 1
        ):
            raise ValueError("settling and spawn attempt counts must be positive.")
        if spawn_max_body_contacts < 0:
            raise ValueError("spawn_max_body_contacts must not be negative.")
        if (
            history_frames < 1
            or precheck_steering_steps < 1
            or precheck_control_steps < 1
            or mobility_window_steps < 1
        ):
            raise ValueError("history, precheck, and mobility window counts must be positive.")
        if precheck_translation_m <= 0.0 or mobility_displacement_m <= 0.0 or mobility_path_m <= 0.0:
            raise ValueError("translation and mobility thresholds must be positive.")
        self.spawn_contact_clearance_m = spawn_contact_clearance_m
        self.settle_max_steps = settle_max_steps
        self.settle_stable_steps = settle_stable_steps
        self.spawn_attempts = spawn_attempts
        self.max_reset_terrain_attempts = max_reset_terrain_attempts
        self.spawn_max_body_contacts = spawn_max_body_contacts
        self.history_frames = history_frames
        self.precheck_steering_steps = precheck_steering_steps
        self.precheck_control_steps = precheck_control_steps
        self.precheck_translation_m = precheck_translation_m
        self.mobility_window_steps = mobility_window_steps
        self.mobility_displacement_m = mobility_displacement_m
        self.mobility_path_m = mobility_path_m
        self.record_precheck_frames = record_precheck_frames
        self._rng = np.random.default_rng(seed)
        self._temporary_directory = tempfile.TemporaryDirectory(prefix="prana_recovery_")
        self._model: mujoco.MjModel | None = None
        self._data: mujoco.MjData | None = None
        self._steering_ids: list[int] = []
        self._drive_ids: list[int] = []
        self._terrain: ProceduralTerrain | None = None
        self._scenario: RecoveryScenario | None = None
        self._episodes_on_terrain = terrain_episodes
        self._step_count = 0
        self._previous_escape_distance_m = 0.0
        self._episode_start_xy = np.zeros(2, dtype=np.float64)
        self._last_action = np.zeros(self.ACTION_SIZE, dtype=np.float32)
        self._terrain_history: deque[np.ndarray] = deque(maxlen=history_frames)
        self._state_history: deque[np.ndarray] = deque(maxlen=history_frames)
        self._mobility_positions: deque[np.ndarray] = deque(maxlen=mobility_window_steps + 1)
        self._mobility_speeds: deque[float] = deque(maxlen=mobility_window_steps)
        self._precheck_max_translation_m = 0.0
        self._precheck_yaw_change_rad = 0.0
        self._reset_terrain_attempts = 0
        self._precheck_frames: list[tuple[str, np.ndarray]] = []
        self._free_joint_qposadr = 0
        self._free_joint_dofadr = 0
        self._substeps = 1
        self._renderer: mujoco.Renderer | None = None
        self._camera_directions_body = self._make_camera_ray_directions()

        # Oldest to newest camera/proprioception frames are concatenated along
        # their feature axes.  The numeric state also carries the last motor
        # command, allowing the policy to notice that a command did not move it.
        self.observation_space = spaces.Dict(
            {
                "terrain": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(2 * history_frames, self.TERRAIN_MAP_CELLS, self.TERRAIN_MAP_CELLS),
                    dtype=np.float32,
                ),
                "state": spaces.Box(
                    low=-100.0,
                    high=100.0,
                    shape=((self.PROPRIOCEPTIVE_STATE_SIZE + self.ACTION_SIZE) * history_frames,),
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.ACTION_SIZE,), dtype=np.float32)

    @property
    def model(self) -> mujoco.MjModel:
        if self._model is None:
            raise RuntimeError("Call reset() before accessing the MuJoCo model.")
        return self._model

    @property
    def data(self) -> mujoco.MjData:
        if self._data is None:
            raise RuntimeError("Call reset() before accessing MuJoCo data.")
        return self._data

    def _build_terrain_model(self) -> None:
        terrain_seed = int(self._rng.integers(0, np.iinfo(np.int32).max))
        self._terrain = ProceduralTerrain(self.terrain_config, terrain_seed)
        workspace = Path(self._temporary_directory.name)
        hfield_path = workspace / f"terrain_{terrain_seed}.bin"
        self._terrain.write_hfield(hfield_path)

        rover_dir = DESCRIPTION_SCRIPTS.parent / "prana_mark2_v2"
        compiled_path = rover_sim.prepare_rover_xml(rover_dir)
        scene_path = workspace / f"scene_{terrain_seed}.xml"
        extent = self.terrain_config.half_extent_m
        amplitude = self.terrain_config.height_range_m
        scene_path.write_text(
            f'''<mujoco model="prana_procedural_recovery">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <asset>
    <hfield name="procedural_terrain" file="{hfield_path}" size="{extent} {extent} {amplitude} 0.1"/>
  </asset>
  <worldbody>
    <light pos="0 0 8" dir="0 0 -1" directional="true"/>
    <geom name="procedural_terrain_geom" type="hfield" hfield="procedural_terrain" friction="1.0 0.005 0.0001" rgba="0.25 0.20 0.14 1"/>
  </worldbody>
  <include file="{compiled_path}"/>
</mujoco>''',
            encoding="utf-8",
        )
        self._model = mujoco.MjModel.from_xml_path(str(scene_path))
        self._data = mujoco.MjData(self._model)
        self._steering_ids, self._drive_ids = actuator_ids(self._model)
        self._free_joint_qposadr = self._model.jnt_qposadr[mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "prana_mark2_v2_freejoint")]
        self._free_joint_dofadr = self._model.jnt_dofadr[mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "prana_mark2_v2_freejoint")]
        self._substeps = max(1, round(self.control_dt_s / self._model.opt.timestep))
        self._episodes_on_terrain = 0

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, np.ndarray], dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        # An explicit seed must fully reproduce an episode, including its
        # heightfield. Ordinary unseeded resets reuse terrain for efficiency.
        if seed is not None or self._model is None or self._episodes_on_terrain >= self.terrain_episodes:
            self._build_terrain_model()

        accepted = False
        unstable_candidates = 0
        mobile_or_tipped_candidates = 0
        for terrain_attempt in range(self.max_reset_terrain_attempts):
            assert self._terrain is not None
            for _ in range(self.spawn_attempts):
                self._scenario = ScenarioSampler(
                    self._terrain, int(self._rng.integers(0, np.iinfo(np.int32).max))
                ).sample()
                if not self._place_and_settle():
                    unstable_candidates += 1
                    continue
                if self._precheck_is_stuck():
                    accepted = True
                    break
                mobile_or_tipped_candidates += 1
            if accepted:
                break
            # This terrain did not offer a stable sampled scenario. Build a
            # new one rather than presenting PPO with a falling rover.
            self._build_terrain_model()
        self._reset_terrain_attempts = terrain_attempt + 1

        if not accepted:
            if unstable_candidates > 0 and mobile_or_tipped_candidates == 0:
                suggestion = "All sampled starts failed to settle; reducing terrain height range may help."
            elif mobile_or_tipped_candidates > 0 and unstable_candidates == 0:
                suggestion = (
                    "All stable starts moved during the precheck; increasing terrain height range or "
                    "sampling more difficult spawn locations may help."
                )
            else:
                suggestion = (
                    "Some starts were unstable and the remaining stable starts moved during the precheck; "
                    "adjust terrain relief more gradually or improve the scenario sampler."
                )
            raise RuntimeError(
                "Could not find a stable, stuck rover spawn after terrain and scenario retries "
                f"({self.max_reset_terrain_attempts} terrain attempts; "
                f"unstable={unstable_candidates}, mobile_or_tipped={mobile_or_tipped_candidates}). {suggestion}"
            )

        self._step_count = 0
        self._episodes_on_terrain += 1
        self._episode_start_xy = self._body_xy()
        self._previous_escape_distance_m = 0.0
        self._last_action.fill(0.0)
        self._mobility_positions.clear()
        self._mobility_positions.append(self._episode_start_xy.copy())
        self._mobility_speeds.clear()
        self._reset_observation_history()
        return self._stacked_observation(), self._info()

    def _place_and_settle(self) -> bool:
        """Place wheels just above terrain, then wait until passive physics is calm."""
        assert self._terrain is not None
        assert self._scenario is not None
        mujoco.mj_resetData(self.model, self.data)
        spawn_x, spawn_y = self._scenario.spawn_xy_m
        rotation = self._terrain_aligned_rotation(spawn_x, spawn_y, self._scenario.spawn_yaw_rad)
        qpos = self.data.qpos
        qpos[self._free_joint_qposadr : self._free_joint_qposadr + 3] = (spawn_x, spawn_y, 0.0)
        quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quaternion, rotation.ravel())
        qpos[self._free_joint_qposadr + 3 : self._free_joint_qposadr + 7] = quaternion
        mujoco.mj_forward(self.model, self.data)

        # At root Z=0 these are each wheel's world positions relative to the
        # terrain. Raise the rover only enough for its lowest wheel to clear.
        root_z = max(
            self._terrain.height_at(*self.data.body(f"wheel_{wheel}").xpos[:2])
            - self.data.body(f"wheel_{wheel}").xpos[2]
            + WHEEL_RADIUS_M
            + self.spawn_contact_clearance_m
            for wheel in WHEEL_NAMES
        )
        qpos[self._free_joint_qposadr + 2] = root_z
        mujoco.mj_forward(self.model, self.data)
        self._apply_direct_action(np.zeros(8, dtype=np.float64))

        stable_steps = 0
        for _ in range(self.settle_max_steps):
            mujoco.mj_step(self.model, self.data)
            if self._is_settled():
                stable_steps += 1
                if stable_steps >= self.settle_stable_steps:
                    return True
            else:
                stable_steps = 0
        return False

    def _terrain_aligned_rotation(self, x_m: float, y_m: float, yaw_rad: float) -> np.ndarray:
        """Return a body rotation with its Z axis normal to a wheel-fit plane."""
        assert self._terrain is not None
        yaw_rotation = np.array(
            [[math.cos(yaw_rad), -math.sin(yaw_rad)], [math.sin(yaw_rad), math.cos(yaw_rad)]],
            dtype=np.float64,
        )
        wheel_xy = np.asarray([x_m, y_m]) + WHEEL_POSITIONS_M @ yaw_rotation.T
        heights = np.array([self._terrain.height_at(*point) for point in wheel_xy])
        plane, *_ = np.linalg.lstsq(
            np.column_stack((wheel_xy, np.ones(len(wheel_xy)))), heights, rcond=None
        )
        normal = np.array([-plane[0], -plane[1], 1.0], dtype=np.float64)
        normal /= np.linalg.norm(normal)
        heading = np.array([math.cos(yaw_rad), math.sin(yaw_rad), 0.0], dtype=np.float64)
        forward = heading - normal * np.dot(heading, normal)
        forward /= np.linalg.norm(forward)
        left = np.cross(normal, forward)
        return np.column_stack((forward, left, normal))

    def _is_settled(self) -> bool:
        linear_velocity = self.data.qvel[self._free_joint_dofadr : self._free_joint_dofadr + 3]
        angular_velocity = self.data.qvel[self._free_joint_dofadr + 3 : self._free_joint_dofadr + 6]
        roll, pitch = self._roll_pitch()
        return (
            np.linalg.norm(linear_velocity) < 0.05
            and np.linalg.norm(angular_velocity) < 0.12
            and abs(roll) < 0.85
            and abs(pitch) < 0.85
            # Stable non-wheel contact is a valid recovery start: a base or
            # rocker resting on a ditch shoulder/ridge is precisely the type
            # of high-centred condition this task should learn to escape.
            and self._body_contact_count() <= self.spawn_max_body_contacts
        )

    def _precheck_is_stuck(self) -> bool:
        """Reject mobile starts, then restore genuinely stuck starts exactly.

        This happens wholly inside ``reset()``.  A mobile candidate is not a
        zero-reward PPO episode: it is discarded and another pose is sampled.
        The yaw probe remains useful diagnostics, but rotation alone never
        makes a rover count as mobile or recovered.
        """
        saved_qpos = self.data.qpos.copy()
        saved_qvel = self.data.qvel.copy()
        saved_act = self.data.act.copy()
        saved_ctrl = self.data.ctrl.copy()
        saved_time = self.data.time
        self._precheck_max_translation_m = 0.0
        self._precheck_yaw_change_rad = 0.0
        self._precheck_frames.clear()

        probes = (
            ("forward", 0.80, 0.0, 0.0),
            ("sideways", 0.0, 0.80, 0.0),
            ("yaw", 0.0, 0.0, 0.80),
        )
        for name, vx, vy, wz in probes:
            # Each probe starts from the same settled state.  Otherwise a
            # forward probe could make the following sideways probe easier.
            self._restore_state(saved_qpos, saved_qvel, saved_act, saved_ctrl, saved_time)
            probe_action = self._twist_probe_action(vx, vy, wz)
            # The old probe measured while the steering brackets were still
            # travelling to their command. On flat ground that incorrectly
            # looked like a stuck rover. Pre-position the steering with zero
            # wheel speed, then begin the actual mobility measurement.
            steering_only_action = probe_action.copy()
            steering_only_action[:4] = 0.0
            self._capture_precheck_frame(f"precheck: {name} (steering)")
            for _ in range(self.precheck_steering_steps):
                self._apply_direct_action(steering_only_action)
                for _ in range(self._substeps):
                    mujoco.mj_step(self.model, self.data)

            start_xy = self._body_xy()
            start_yaw = self._yaw()
            self._capture_precheck_frame(f"precheck: {name} (drive)")
            for step in range(self.precheck_control_steps):
                self._apply_direct_action(probe_action)
                for _ in range(self._substeps):
                    mujoco.mj_step(self.model, self.data)
                # Five frames per probe is enough to see motion without
                # retaining large RGB videos in every vector-environment worker.
                if (step + 1) % max(1, math.ceil(self.precheck_control_steps / 4)) == 0:
                    self._capture_precheck_frame(f"precheck: {name}")
            self._precheck_max_translation_m = max(
                self._precheck_max_translation_m, float(np.linalg.norm(self._body_xy() - start_xy))
            )
            self._precheck_yaw_change_rad = max(
                self._precheck_yaw_change_rad, abs(self._wrapped_angle(self._yaw() - start_yaw))
            )
            roll, pitch = self._roll_pitch()
            if abs(roll) > 0.85 or abs(pitch) > 0.85:
                self._precheck_frames.clear()
                return False

        is_stuck = self._precheck_max_translation_m < self.precheck_translation_m
        if is_stuck:
            self._restore_state(saved_qpos, saved_qvel, saved_act, saved_ctrl, saved_time)
        else:
            self._precheck_frames.clear()
        return is_stuck

    def _capture_precheck_frame(self, label: str) -> None:
        if self.record_precheck_frames:
            self._precheck_frames.append((label, self.render()))

    def consume_precheck_frames(self) -> list[tuple[str, np.ndarray]]:
        """Return-and-clear probe frames from the current accepted reset.

        This is intentionally opt-in. It is for inspecting the reset filter,
        not for normal training, where retaining RGB frames would be wasteful.
        """
        frames = self._precheck_frames
        self._precheck_frames = []
        return frames

    def _restore_state(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        act: np.ndarray,
        ctrl: np.ndarray,
        time_s: float,
    ) -> None:
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.data.act[:] = act
        self.data.ctrl[:] = ctrl
        self.data.time = time_s
        mujoco.mj_forward(self.model, self.data)

    def _twist_probe_action(self, vx: float, vy: float, wz: float) -> np.ndarray:
        steering, wheel_angular_velocity = body_twist_to_wheels(vx, vy, wz)
        drive = wheel_angular_velocity * WHEEL_RADIUS_M / self.MAX_WHEEL_LINEAR_SPEED_M_S
        steer = steering / self.MAX_STEERING_ANGLE_RAD
        return np.clip(np.concatenate((drive, steer)), -1.0, 1.0)

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self._apply_direct_action(action)
        for _ in range(self._substeps):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        escape_distance_m = self._escape_distance()
        progress_m = escape_distance_m - self._previous_escape_distance_m
        self._previous_escape_distance_m = escape_distance_m
        position = self._body_xy()
        self._mobility_positions.append(position)
        planar_speed_m_s = self._planar_speed_m_s()
        self._mobility_speeds.append(planar_speed_m_s)
        roll_rad, pitch_rad = self._roll_pitch()
        tipped = abs(roll_rad) > 0.85 or abs(pitch_rad) > 0.85
        body_contacts = self._body_contact_count()
        success = not tipped and body_contacts == 0 and self._has_sustained_mobility()
        terminated = success or tipped
        truncated = self._step_count >= self.max_episode_steps

        # Target-free reward: get meaningfully away from the stuck start while
        # preferring sustained, upright, smooth translation over thrashing.
        energy = float(np.square(action).mean())
        action_change = float(np.square(action - self._last_action).mean())
        uprightness = max(0.0, 1.0 - (abs(roll_rad) + abs(pitch_rad)) / 0.70)
        sustained_motion = min(planar_speed_m_s / 0.45, 1.0)
        reward = (
            8.0 * progress_m
            + 0.03 * sustained_motion
            + 0.02 * uprightness
            - 0.01
            - 0.015 * energy
            - 0.010 * action_change
            - 0.01 * body_contacts
        )
        if success:
            reward += 15.0
        if tipped:
            reward -= 10.0
        self._last_action = action.astype(np.float32)
        return self._observation(), float(reward), terminated, truncated, self._info(success=success, tipped=tipped)

    def _apply_direct_action(self, action: np.ndarray) -> None:
        """Map normalized direct motor commands to MuJoCo actuator targets."""
        drive_linear_speed = action[:4] * self.MAX_WHEEL_LINEAR_SPEED_M_S
        steering_target = action[4:] * self.MAX_STEERING_ANGLE_RAD
        for actuator_id, target in zip(self._drive_ids, drive_linear_speed / WHEEL_RADIUS_M, strict=True):
            self.data.ctrl[actuator_id] = target
        for actuator_id, target in zip(self._steering_ids, steering_target, strict=True):
            self.data.ctrl[actuator_id] = target

    def _body_rotation(self) -> np.ndarray:
        return self.data.body("prana_mark2_v2").xmat.reshape(3, 3).copy()

    def _body_xy(self) -> np.ndarray:
        return self.data.body("prana_mark2_v2").xpos[:2].copy()

    def _escape_distance(self) -> float:
        return float(np.linalg.norm(self._body_xy() - self._episode_start_xy))

    def _planar_speed_m_s(self) -> float:
        velocity_world = self.data.qvel[self._free_joint_dofadr : self._free_joint_dofadr + 3]
        return float(np.linalg.norm(velocity_world[:2]))

    def _has_sustained_mobility(self) -> bool:
        if len(self._mobility_positions) < self.mobility_window_steps + 1:
            return False
        positions = np.asarray(self._mobility_positions)
        net_displacement = float(np.linalg.norm(positions[-1] - positions[0]))
        path_length = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
        mean_speed = float(np.mean(self._mobility_speeds))
        roll, pitch = self._roll_pitch()
        return (
            net_displacement >= self.mobility_displacement_m
            and path_length >= self.mobility_path_m
            and mean_speed >= 0.15
            and abs(roll) < 0.35
            and abs(pitch) < 0.35
        )

    def _roll_pitch(self) -> tuple[float, float]:
        rotation = self._body_rotation()
        pitch = math.atan2(-rotation[2, 0], math.hypot(rotation[2, 1], rotation[2, 2]))
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        return roll, pitch

    def _yaw(self) -> float:
        rotation = self._body_rotation()
        return math.atan2(rotation[1, 0], rotation[0, 0])

    @staticmethod
    def _wrapped_angle(angle_rad: float) -> float:
        return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi

    def _body_contact_count(self) -> int:
        count = 0
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            names = (self.model.geom(contact.geom1).name, self.model.geom(contact.geom2).name)
            count += any(name is not None and name.startswith("collision_") for name in names)
        return count

    def _observation(self) -> dict[str, np.ndarray]:
        """Append and return a history of camera and proprioceptive frames.

        The terrain view is derived from actual MuJoCo 3-D ray intersections,
        not direct access to the procedural heightfield.  Those intersections
        are the simulated front-camera point cloud; rasterising them produces
        a fixed-size input suited to a CNN.
        """
        terrain, state = self._sensor_frame()
        self._terrain_history.append(terrain)
        self._state_history.append(state)
        return self._stacked_observation()

    def _reset_observation_history(self) -> None:
        """Seed every history slot with the same settled initial observation."""
        terrain, state = self._sensor_frame()
        self._terrain_history.clear()
        self._state_history.clear()
        for _ in range(self.history_frames):
            self._terrain_history.append(terrain.copy())
            self._state_history.append(state.copy())

    def _stacked_observation(self) -> dict[str, np.ndarray]:
        return {
            "terrain": np.concatenate(tuple(self._terrain_history), axis=0).astype(np.float32),
            "state": np.concatenate(tuple(self._state_history), axis=0).astype(np.float32),
        }

    def _sensor_frame(self) -> tuple[np.ndarray, np.ndarray]:
        """Return one target-free camera/proprioception/action frame."""
        rotation = self._body_rotation()
        position = self.data.body("prana_mark2_v2").xpos
        velocity = rotation.T @ self.data.qvel[self._free_joint_dofadr : self._free_joint_dofadr + 3]
        angular_velocity = rotation.T @ self.data.qvel[self._free_joint_dofadr + 3 : self._free_joint_dofadr + 6]
        roll, pitch = self._roll_pitch()
        steering = np.array([self.data.qpos[self.model.joint(name).qposadr[0]] for name in ("bracker_lf_joint", "bracket_rf_joint", "bracket_lb_joint", "bracket_rb_joint")])
        wheel_velocity = np.array([self.data.qvel[self.model.joint(name).dofadr[0]] for name in ("wheel_fl_joint", "wheel_fr_joint", "wheel_lb_joint", "wheel_rb_joint")])
        proprioception = np.concatenate(
            (velocity, angular_velocity, [roll, pitch], steering, wheel_velocity)
        ).astype(np.float32)
        state = np.concatenate((proprioception, self._last_action)).astype(np.float32)
        return self._camera_elevation_map(rotation, position), state

    def _make_camera_ray_directions(self) -> np.ndarray:
        """Create normalized body-frame rays for a 65 degree front camera."""
        horizontal = np.linspace(
            -self.CAMERA_HORIZONTAL_FOV_RAD / 2.0,
            self.CAMERA_HORIZONTAL_FOV_RAD / 2.0,
            self.CAMERA_WIDTH,
        )
        # ZED's camera is mounted above the chassis. Aim its centre slightly
        # downward so the finite depth range covers terrain ahead of the rover.
        vertical = np.linspace(math.radians(-48.0), math.radians(-4.0), self.CAMERA_HEIGHT)
        yaw, pitch = np.meshgrid(horizontal, vertical)
        directions = np.stack(
            (
                np.cos(pitch) * np.cos(yaw),
                np.cos(pitch) * np.sin(yaw),
                np.sin(pitch),
            ),
            axis=-1,
        ).reshape(-1, 3)
        return directions.astype(np.float64)

    def _front_point_cloud_body(self, rotation: np.ndarray, position: np.ndarray) -> np.ndarray:
        """Return visible front-camera surface points in rover body coordinates."""
        directions_world = self._camera_directions_body @ rotation.T
        origin_world = position + rotation @ self.CAMERA_OFFSET_BODY_M
        geomgroup = np.ones(6, dtype=np.uint8)
        rover_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "prana_mark2_v2")
        distances = np.empty(len(directions_world), dtype=np.float64)
        hit_geom_ids = np.empty(len(directions_world), dtype=np.int32)
        mujoco.mj_multiRay(
            self.model,
            self.data,
            origin_world,
            directions_world.ravel(),
            geomgroup,
            True,
            rover_body_id,
            hit_geom_ids,
            distances,
            None,
            len(directions_world),
            self.CAMERA_RANGE_M,
        )
        valid = (distances >= 0.0) & (distances <= self.CAMERA_RANGE_M)
        points_world = origin_world + directions_world[valid] * distances[valid, None]
        return (points_world - position) @ rotation

    def _camera_elevation_map(self, rotation: np.ndarray, position: np.ndarray) -> np.ndarray:
        """Rasterise the front-camera point cloud as elevation and visibility."""
        points_body = self._front_point_cloud_body(rotation, position)
        cells = self.TERRAIN_MAP_CELLS
        # Start at -inf so ``maximum.at`` also preserves terrain below the
        # rover. A zero-filled map would erase all negative elevations.
        elevation = np.full((cells, cells), -np.inf, dtype=np.float32)
        observed = np.full((cells, cells), -1.0, dtype=np.float32)
        if len(points_body) == 0:
            return np.stack((np.zeros_like(elevation), observed))

        forward = points_body[:, 0]
        lateral = points_body[:, 1]
        valid = (
            (forward >= 0.0)
            & (forward < self.TERRAIN_MAP_FORWARD_M)
            & (np.abs(lateral) < self.TERRAIN_MAP_HALF_WIDTH_M)
        )
        if not np.any(valid):
            return np.stack((np.zeros_like(elevation), observed))

        forward = forward[valid]
        lateral = lateral[valid]
        point_heights = points_body[valid, 2] / self.TERRAIN_MAP_HEIGHT_SCALE_M
        rows = np.minimum((forward / self.TERRAIN_MAP_FORWARD_M * cells).astype(np.int32), cells - 1)
        columns = np.minimum(
            ((lateral + self.TERRAIN_MAP_HALF_WIDTH_M) / (2.0 * self.TERRAIN_MAP_HALF_WIDTH_M) * cells).astype(np.int32),
            cells - 1,
        )
        # Multiple depth pixels may land in one grid cell.  The highest hit is
        # safest for a rover: it preserves the visible top of a ledge or rock.
        np.maximum.at(elevation, (rows, columns), point_heights)
        observed[rows, columns] = 1.0
        elevation[observed < 0.0] = 0.0
        return np.stack((np.clip(elevation, -1.0, 1.0), observed))

    def _info(self, *, success: bool = False, tipped: bool = False) -> dict:
        assert self._scenario is not None
        return {
            "terrain_seed": self._scenario.terrain_seed,
            "difficulty": self._scenario.difficulty,
            "displacement_from_start_m": self._escape_distance(),
            "precheck_max_translation_m": self._precheck_max_translation_m,
            "precheck_yaw_change_rad": self._precheck_yaw_change_rad,
            "reset_terrain_attempts": self._reset_terrain_attempts,
            "success": success,
            "tipped": tipped,
        }

    def render(self) -> np.ndarray:
        """Return an RGB image of this worker's current MuJoCo state.

        Training remains headless unless a caller explicitly requests these
        frames.  The live-training viewer uses this method to tile frames from
        several real ``SubprocVecEnv`` workers into one window.
        """
        if self._model is None or self._data is None:
            raise RuntimeError("Call reset() before rendering the environment.")
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self._model, height=288, width=384)

        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = self._data.body("prana_mark2_v2").xpos
        camera.distance = 3.2
        camera.azimuth = 135.0
        camera.elevation = -28.0
        self._renderer.update_scene(self._data, camera=camera)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self._temporary_directory.cleanup()
