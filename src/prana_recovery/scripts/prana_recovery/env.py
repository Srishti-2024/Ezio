"""Gymnasium environment for procedural Prana recovery training."""

from __future__ import annotations

import math
import sys
import tempfile
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
from prana_mujoco_teleop import actuator_ids, apply_twist  # noqa: E402
import mujoco_sim_mark2_v2 as rover_sim  # noqa: E402


class PranaRecoveryEnv(gym.Env[np.ndarray, np.ndarray]):
    """Recover a Prana rover from difficult positions on seeded heightfields.

    Actions are normalized ``[vx, vy, wz]`` body commands.  The environment
    routes them through the same 4-wheel independent-steering controller used
    by the interactive viewer.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        terrain_config: TerrainConfig | None = None,
        terrain_episodes: int = 8,
        control_dt_s: float = 0.05,
        max_episode_steps: int = 300,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.terrain_config = terrain_config or TerrainConfig()
        self.terrain_episodes = terrain_episodes
        self.control_dt_s = control_dt_s
        self.max_episode_steps = max_episode_steps
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
        self._previous_distance_m = 0.0
        self._free_joint_qposadr = 0
        self._free_joint_dofadr = 0
        self._substeps = 1
        self._renderer: mujoco.Renderer | None = None

        # target XY (2), base linear and angular velocity (6), pitch and roll
        # (2), steering joints (4), wheel velocities (4), 16 ray distances.
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(34,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

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

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        # An explicit seed must fully reproduce an episode, including its
        # heightfield. Ordinary unseeded resets reuse terrain for efficiency.
        if seed is not None or self._model is None or self._episodes_on_terrain >= self.terrain_episodes:
            self._build_terrain_model()

        assert self._terrain is not None
        self._scenario = ScenarioSampler(self._terrain, int(self._rng.integers(0, np.iinfo(np.int32).max))).sample()
        mujoco.mj_resetData(self.model, self.data)
        spawn_x, spawn_y = self._scenario.spawn_xy_m
        qpos = self.data.qpos
        qpos[self._free_joint_qposadr : self._free_joint_qposadr + 3] = (
            spawn_x,
            spawn_y,
            self._terrain.height_at(spawn_x, spawn_y) + 0.86,
        )
        half_yaw = self._scenario.spawn_yaw_rad * 0.5
        qpos[self._free_joint_qposadr + 3 : self._free_joint_qposadr + 7] = (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw))
        mujoco.mj_forward(self.model, self.data)
        apply_twist(self.data, self._steering_ids, self._drive_ids, np.zeros(3))
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)

        self._step_count = 0
        self._episodes_on_terrain += 1
        self._previous_distance_m = self._distance_to_target()
        return self._observation(), self._info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        twist = action * np.array([0.8, 0.8, 0.9], dtype=np.float64)
        apply_twist(self.data, self._steering_ids, self._drive_ids, twist)
        for _ in range(self._substeps):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        distance_m = self._distance_to_target()
        progress_m = self._previous_distance_m - distance_m
        self._previous_distance_m = distance_m
        roll_rad, pitch_rad = self._roll_pitch()
        success = distance_m < 0.42 and abs(roll_rad) < 0.35 and abs(pitch_rad) < 0.35
        tipped = abs(roll_rad) > 0.85 or abs(pitch_rad) > 0.85
        terminated = success or tipped
        truncated = self._step_count >= self.max_episode_steps

        body_contacts = self._body_contact_count()
        energy = float(np.square(action).sum())
        reward = 8.0 * progress_m - 0.01 - 0.015 * energy - 0.01 * body_contacts
        if success:
            reward += 15.0
        if tipped:
            reward -= 10.0
        return self._observation(), float(reward), terminated, truncated, self._info(success=success, tipped=tipped)

    def _body_rotation(self) -> np.ndarray:
        return self.data.body("prana_mark2_v2").xmat.reshape(3, 3).copy()

    def _distance_to_target(self) -> float:
        assert self._scenario is not None
        position = self.data.body("prana_mark2_v2").xpos
        target = np.asarray(self._scenario.egress_xy_m)
        return float(np.linalg.norm(target - position[:2]))

    def _roll_pitch(self) -> tuple[float, float]:
        rotation = self._body_rotation()
        pitch = math.atan2(-rotation[2, 0], math.hypot(rotation[2, 1], rotation[2, 2]))
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        return roll, pitch

    def _body_contact_count(self) -> int:
        count = 0
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            names = (self.model.geom(contact.geom1).name, self.model.geom(contact.geom2).name)
            count += any(name.startswith("collision_") for name in names)
        return count

    def _observation(self) -> np.ndarray:
        assert self._scenario is not None
        rotation = self._body_rotation()
        position = self.data.body("prana_mark2_v2").xpos
        target_delta_body = rotation.T @ np.array([
            self._scenario.egress_xy_m[0] - position[0],
            self._scenario.egress_xy_m[1] - position[1],
            0.0,
        ])
        velocity = rotation.T @ self.data.qvel[self._free_joint_dofadr : self._free_joint_dofadr + 3]
        angular_velocity = rotation.T @ self.data.qvel[self._free_joint_dofadr + 3 : self._free_joint_dofadr + 6]
        roll, pitch = self._roll_pitch()
        steering = np.array([self.data.qpos[self.model.joint(name).qposadr[0]] for name in ("bracker_lf_joint", "bracket_rf_joint", "bracket_lb_joint", "bracket_rb_joint")])
        wheel_velocity = np.array([self.data.qvel[self.model.joint(name).dofadr[0]] for name in ("wheel_fl_joint", "wheel_fr_joint", "wheel_lb_joint", "wheel_rb_joint")])
        rays = self._lidar_distances(rotation, position)
        return np.concatenate((target_delta_body[:2], velocity, angular_velocity, [roll, pitch], steering, wheel_velocity, rays)).astype(np.float32)

    def _lidar_distances(self, rotation: np.ndarray, position: np.ndarray) -> np.ndarray:
        distances = np.empty(16, dtype=np.float64)
        origin = position + np.array([0.0, 0.0, 0.20])
        geomgroup = np.ones(6, dtype=np.uint8)
        rover_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "prana_mark2_v2")
        hit_geom_id = np.empty(1, dtype=np.int32)
        for index, angle in enumerate(np.linspace(-np.pi, np.pi, len(distances), endpoint=False)):
            direction = rotation @ np.array([math.cos(angle), math.sin(angle), 0.0])
            distance = mujoco.mj_ray(
                self.model, self.data, origin, direction, geomgroup, True, rover_body_id, hit_geom_id
            )
            distances[index] = min(distance, 4.0) if distance >= 0.0 else 4.0
        return distances / 4.0

    def _info(self, *, success: bool = False, tipped: bool = False) -> dict:
        assert self._scenario is not None
        return {
            "terrain_seed": self._scenario.terrain_seed,
            "difficulty": self._scenario.difficulty,
            "distance_to_egress_m": self._distance_to_target(),
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
