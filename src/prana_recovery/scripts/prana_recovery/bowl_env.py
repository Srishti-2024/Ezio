"""Smooth-bowl curriculum environment for discovering rover escape behaviours."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

from .env import DESCRIPTION_SCRIPTS, WHEEL_RADIUS_M, PranaRecoveryEnv, actuator_ids, rover_sim
from .scenarios import RecoveryScenario
from .terrain import TerrainConfig


@dataclass(frozen=True)
class BowlConfig:
    """Geometry of the smooth parabolic bowl used by this environment."""

    half_extent_m: float = 8.0
    resolution: int = 257
    rim_radius_m: float = 2.5
    depth_m: float = 1.4
    spawn_radius_m: float = 0.25
    escape_margin_m: float = 0.30


class ParabolicBowl:
    """Deterministic hfield plus analytic terrain queries for one bowl."""

    def __init__(self, config: BowlConfig, seed: int) -> None:
        if not 0.0 < config.rim_radius_m < config.half_extent_m:
            raise ValueError("bowl rim radius must be inside the terrain extent.")
        if config.depth_m <= 0.0 or config.spawn_radius_m < 0.0:
            raise ValueError("bowl depth must be positive and spawn radius cannot be negative.")
        self.config = config
        self.seed = int(seed)
        coords = np.linspace(-config.half_extent_m, config.half_extent_m, config.resolution)
        x, y = np.meshgrid(coords, coords, indexing="xy")
        self.heights_m = config.depth_m * np.minimum((x * x + y * y) / config.rim_radius_m**2, 1.0)

    def height_at(self, x_m: float, y_m: float) -> float:
        return float(self.config.depth_m * min((x_m * x_m + y_m * y_m) / self.config.rim_radius_m**2, 1.0))

    def slope_at_radius(self, radius_m: float) -> float:
        return 0.0 if radius_m > self.config.rim_radius_m else 2.0 * self.config.depth_m * radius_m / self.config.rim_radius_m**2

    def write_hfield(self, path: Path) -> None:
        normalized = self.heights_m / self.config.depth_m
        with path.open("wb") as output:
            np.asarray(normalized.shape, dtype=np.int32).tofile(output)
            normalized.astype(np.float32).tofile(output)


class BowlRecoveryEnv(PranaRecoveryEnv):
    """Same sensors/actions as ditch recovery, with a bowl-centred objective.

    It deliberately has no mobility precheck: every episode starts in the bowl
    and is a valid curriculum example, even if the rover could immediately
    drive away.  Passive settling is retained so PPO never starts mid-drop.
    """

    def __init__(self, bowl_config: BowlConfig | None = None, **kwargs) -> None:
        self.bowl_config = bowl_config or BowlConfig()
        terrain_config = TerrainConfig(
            half_extent_m=self.bowl_config.half_extent_m,
            resolution=self.bowl_config.resolution,
            height_range_m=self.bowl_config.depth_m,
        )
        super().__init__(terrain_config=terrain_config, **kwargs)
        self._bowl: ParabolicBowl | None = None
        self._bowl_center_xy = np.zeros(2, dtype=np.float64)

    @property
    def bowl(self) -> ParabolicBowl:
        if self._bowl is None:
            raise RuntimeError("Call reset() before accessing bowl geometry.")
        return self._bowl

    def _build_bowl_model(self, bowl_seed: int) -> None:
        self._bowl = ParabolicBowl(self.bowl_config, bowl_seed)
        # Parent placement and terrain-camera support only need height_at().
        self._terrain = self._bowl  # type: ignore[assignment]
        workspace = Path(self._temporary_directory.name)
        hfield_path = workspace / f"bowl_{bowl_seed}.bin"
        self._bowl.write_hfield(hfield_path)
        rover_dir = DESCRIPTION_SCRIPTS.parent / "prana_mark2_v2"
        compiled_path = rover_sim.prepare_rover_xml(rover_dir)
        scene_path = workspace / f"bowl_scene_{bowl_seed}.xml"
        extent = self.bowl_config.half_extent_m
        depth = self.bowl_config.depth_m
        scene_path.write_text(
            f'''<mujoco model="prana_bowl_recovery">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <asset>
    <hfield name="bowl_terrain" file="{hfield_path}" size="{extent} {extent} {depth} 0.1"/>
  </asset>
  <worldbody>
    <light pos="0 0 8" dir="0 0 -1" directional="true"/>
    <geom name="bowl_terrain_geom" type="hfield" hfield="bowl_terrain" friction="1.8 0.01 0.001" rgba="0.25 0.20 0.14 1"/>
  </worldbody>
  <include file="{compiled_path}"/>
</mujoco>''',
            encoding="utf-8",
        )
        self._model = mujoco.MjModel.from_xml_path(str(scene_path))
        self._data = mujoco.MjData(self._model)
        self._steering_ids, self._drive_ids = actuator_ids(self._model)
        free_joint = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "prana_mark2_v2_freejoint")
        self._free_joint_qposadr = self._model.jnt_qposadr[free_joint]
        self._free_joint_dofadr = self._model.jnt_dofadr[free_joint]
        self._substeps = max(1, round(self.control_dt_s / self._model.opt.timestep))

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        gym.Env.reset(self, seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if self._model is None or seed is not None:
            self._build_bowl_model(int(self._rng.integers(0, np.iinfo(np.int32).max)))

        # No stuckness probe: sample a centre start and only retry if passive
        # physics cannot settle it safely.
        settled = False
        for attempt in range(self.spawn_attempts * 2):
            radius = self.bowl_config.spawn_radius_m * math.sqrt(float(self._rng.random()))
            angle = float(self._rng.uniform(-math.pi, math.pi))
            spawn_xy = (radius * math.cos(angle), radius * math.sin(angle))
            yaw = float(self._rng.uniform(-math.pi, math.pi))
            self._scenario = RecoveryScenario(
                terrain_seed=self.bowl.seed,
                spawn_xy_m=spawn_xy,
                spawn_yaw_rad=yaw,
                difficulty=self.bowl.slope_at_radius(radius),
            )
            if self._place_and_settle():
                settled = True
                break
        if not settled:
            raise RuntimeError("Could not settle rover in smooth bowl; inspect bowl geometry or rover physics.")

        self._step_count = 0
        self._fallback_mobile_start = False
        self._reset_terrain_attempts = attempt + 1
        self._episode_start_xy = self._body_xy()
        self._previous_escape_distance_m = self._escape_distance()
        self._initial_body_contacts = self._body_contact_count()
        self._previous_body_contacts = self._initial_body_contacts
        self._last_action.fill(0.0)
        self._mobility_positions.clear()
        self._mobility_positions.append(self._episode_start_xy.copy())
        self._mobility_speeds.clear()
        self._reset_observation_history()
        return self._stacked_observation(), self._info()

    def _escape_distance(self) -> float:
        """Bowl progress is radial distance from the bowl centre, not spawn."""
        return float(np.linalg.norm(self._body_xy() - self._bowl_center_xy))

    def _has_sustained_mobility(self) -> bool:
        return (
            self._escape_distance() >= self.bowl_config.rim_radius_m + self.bowl_config.escape_margin_m
            and super()._has_sustained_mobility()
        )

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self._apply_direct_action(action)
        for _ in range(self._substeps):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        distance = self._escape_distance()
        progress_m = distance - self._previous_escape_distance_m
        self._previous_escape_distance_m = distance
        self._mobility_positions.append(self._body_xy())
        planar_speed_m_s = self._planar_speed_m_s()
        self._mobility_speeds.append(planar_speed_m_s)
        roll_rad, pitch_rad = self._roll_pitch()
        tipped = abs(roll_rad) > 0.85 or abs(pitch_rad) > 0.85
        body_contacts = self._body_contact_count()
        success = not tipped and body_contacts == 0 and self._has_sustained_mobility()
        terminated = success or tipped
        truncated = self._step_count >= self.max_episode_steps

        # This curriculum intentionally favours trying varied recovery actions.
        # Stationary upright behaviour is negative; uprightness only helps when
        # actual XY motion is present.  It does not reward a spiral explicitly.
        energy = float(np.square(action).mean())
        action_change = float(np.square(action - self._last_action).mean())
        uprightness = max(0.0, 1.0 - (abs(roll_rad) + abs(pitch_rad)) / 0.70)
        sustained_motion = min(planar_speed_m_s / 0.45, 1.0)
        contact_release = float(np.clip(self._previous_body_contacts - body_contacts, -2, 2))
        passive = self._step_count > 40 and planar_speed_m_s < 0.03 and float(np.mean(np.abs(action))) < 0.12
        reward = (
            7.0 * progress_m
            + 0.06 * sustained_motion * uprightness
            + 0.30 * contact_release
            - 0.015
            - 0.005 * energy
            - 0.003 * action_change
            - 0.03 * body_contacts
        )
        if passive:
            reward -= 0.03
        if success:
            reward += 30.0
        if tipped:
            reward -= 5.0
        self._previous_body_contacts = body_contacts
        self._last_action = action.astype(np.float32)
        return self._observation(), float(reward), terminated, truncated, self._info(success=success, tipped=tipped)

    def _info(self, *, success: bool = False, tipped: bool = False, skipped: bool = False) -> dict:
        info = super()._info(success=success, tipped=tipped, skipped=skipped)
        info.update(
            bowl_radius_m=self._escape_distance(),
            bowl_rim_radius_m=self.bowl_config.rim_radius_m,
            bowl_depth_m=self.bowl_config.depth_m,
        )
        return info
