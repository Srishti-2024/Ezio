#!/usr/bin/env python3
"""Run one saved recovery policy on several rovers in one MuJoCo world.

This is deliberately a *showcase*, rather than an evaluator.  It tiles seeded
procedural terrain patches into one large heightfield, puts one rover in a
ditch on each patch, and keeps MuJoCo's normal free-camera viewer alive after
the rovers finish.  Use ``evaluate_prana_recovery.py`` for comparable success
rates: that script performs the full reset/precheck procedure used in training.
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from prana_recovery.env import PranaRecoveryEnv
from prana_recovery.scenarios import RecoveryScenario, ScenarioSampler
from prana_recovery.terrain import ProceduralTerrain, TerrainConfig


# ``SCRIPT_DIR`` is ``.../src/prana_recovery/scripts``; the sibling packages
# live directly under ``src``.
SOURCE_ROOT = SCRIPT_DIR.parents[1]
DRIVE_SCRIPTS = SOURCE_ROOT / "prana_mujoco_drive" / "scripts"
DESCRIPTION_SCRIPTS = SOURCE_ROOT / "prana_description" / "scripts"
sys.path.insert(0, str(DRIVE_SCRIPTS))
sys.path.insert(0, str(DESCRIPTION_SCRIPTS))
from prana_mujoco_teleop import WHEEL_NAMES, WHEEL_POSITIONS_M, WHEEL_RADIUS_M  # noqa: E402
import mujoco_sim_mark2_v2 as rover_sim  # noqa: E402


STEERING_JOINTS = ("bracker_lf_joint", "bracket_rf_joint", "bracket_lb_joint", "bracket_rb_joint")
WHEEL_JOINTS = tuple(f"wheel_{wheel}_joint" for wheel in WHEEL_NAMES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Saved PPO model, with or without .zip")
    parser.add_argument("--rovers", type=int, default=8, help="Number of simultaneous rovers (default: 8)")
    parser.add_argument("--seed", type=int, default=10_000, help="Seed for the whole showcase (default: 10000)")
    parser.add_argument("--terrain-height-range", type=float, default=0.65, help="Terrain relief in metres")
    parser.add_argument("--max-steps", type=int, default=300, help="PPO decisions per rover (default: 300)")
    parser.add_argument("--history-frames", type=int, default=4, help="Must match policy training (default: 4)")
    parser.add_argument("--settle-steps", type=int, default=500, help="Passive physics steps before policy starts")
    parser.add_argument("--device", default="auto", help="Stable-Baselines device for policy inference")
    parser.add_argument("--realtime", action="store_true", help="Run at the 0.05 s policy-control rate")
    parser.add_argument(
        "--close-when-done",
        action="store_true",
        help="Close the viewer after printing results; by default it stays open for inspection.",
    )
    return parser.parse_args()


def prefixed_rover_elements(compiled_root: ET.Element, prefix: str) -> tuple[ET.Element, list[ET.Element], list[ET.Element]]:
    """Copy the compiled rover's body, equality constraints and actuators.

    Mesh assets remain shared. Everything that belongs to a physical rover is
    renamed, which is what makes this a single model with independent robots.
    """
    source_worldbody = compiled_root.find("worldbody")
    source_equality = compiled_root.find("equality")
    source_actuator = compiled_root.find("actuator")
    if source_worldbody is None or source_equality is None or source_actuator is None:
        raise RuntimeError("Compiled rover XML is missing worldbody, equality, or actuator sections.")
    source_body = next((child for child in source_worldbody if child.tag == "body"), None)
    if source_body is None:
        raise RuntimeError("Compiled rover XML has no root rover body.")

    body = copy.deepcopy(source_body)
    equality = [copy.deepcopy(element) for element in source_equality]
    actuators = [copy.deepcopy(element) for element in source_actuator]
    rename: dict[str, str] = {}

    # MuJoCo gives unnamed mesh geoms the mesh name.  That would clash eight
    # times, so give every geom a real unique name too.
    for element in body.iter():
        if element.tag == "geom" and "name" not in element.attrib:
            mesh = element.get("mesh", "geom")
            # The compiled rover uses each visual mesh once, so this also
            # gives tyre contacts stable, readable names such as
            # ``r3_wheel_fl_visual``.
            element.set("name", f"{prefix}_{mesh}")
            # This name was just made unique; it is not a source-model name
            # that another XML element can reference.
            continue
        name = element.get("name")
        if name:
            rename[name] = f"{prefix}_{name}"
    for element in equality + actuators:
        name = element.get("name")
        if name:
            rename[name] = f"{prefix}_{name}"

    # References between the copied parts must be redirected to their local
    # copy. Mesh references are intentionally not touched: assets are shared.
    reference_attributes = {"joint", "site", "site1", "site2", "body", "body1", "body2", "geom", "geom1", "geom2"}
    for root in [body, *equality, *actuators]:
        for element in root.iter():
            name = element.get("name")
            if name in rename:
                element.set("name", rename[name])
            for attribute in reference_attributes:
                value = element.get(attribute)
                if value in rename:
                    element.set(attribute, rename[value])
    return body, equality, actuators


def write_tiled_heightfield(path: Path, terrains: list[ProceduralTerrain], rows: int, columns: int) -> None:
    """Join equal-sized terrain patches into one normalized MuJoCo hfield."""
    resolution = terrains[0].config.resolution
    stride = resolution - 1
    grid = np.zeros((rows * stride + 1, columns * stride + 1), dtype=np.float32)
    height_range = terrains[0].config.height_range_m
    for index, terrain in enumerate(terrains):
        row, column = divmod(index, columns)
        r0, c0 = row * stride, column * stride
        grid[r0 : r0 + resolution, c0 : c0 + resolution] = terrain.heights_m / height_range
    with path.open("wb") as output:
        np.asarray(grid.shape, dtype=np.int32).tofile(output)
        grid.tofile(output)


def build_showcase_model(
    workspace: Path, terrains: list[ProceduralTerrain], rows: int, columns: int
) -> mujoco.MjModel:
    """Assemble the terrain mosaic and all rover copies into one MjModel."""
    rover_dir = DESCRIPTION_SCRIPTS.parent / "prana_mark2_v2"
    compiled_path = rover_sim.prepare_rover_xml(rover_dir)
    compiled_root = ET.parse(compiled_path).getroot()
    terrain_config = terrains[0].config
    hfield_path = workspace / "showcase_terrain.bin"
    write_tiled_heightfield(hfield_path, terrains, rows, columns)

    root = ET.Element("mujoco", {"model": "prana_recovery_showcase"})
    ET.SubElement(root, "compiler", {"angle": "radian"})
    ET.SubElement(root, "option", {"timestep": "0.002", "gravity": "0 0 -9.81"})
    asset = ET.SubElement(root, "asset")
    tile_width = 2.0 * terrain_config.half_extent_m
    ET.SubElement(
        asset,
        "hfield",
        {
            "name": "showcase_terrain",
            "file": str(hfield_path),
            "size": f"{columns * tile_width / 2.0} {rows * tile_width / 2.0} {terrain_config.height_range_m} 0.1",
        },
    )
    source_asset = compiled_root.find("asset")
    if source_asset is None:
        raise RuntimeError("Compiled rover XML has no asset section.")
    for element in source_asset:
        copied = copy.deepcopy(element)
        if copied.get("file"):
            copied.set("file", str((compiled_path.parent / copied.get("file", "")).resolve()))
        asset.append(copied)

    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(worldbody, "light", {"pos": "0 0 28", "dir": "0 0 -1", "directional": "true"})
    ET.SubElement(worldbody, "geom", {"name": "showcase_terrain_geom", "type": "hfield", "hfield": "showcase_terrain", "friction": "1.8 0.01 0.001", "rgba": "0.25 0.20 0.14 1"})
    equality = ET.SubElement(root, "equality")
    actuators = ET.SubElement(root, "actuator")
    for index in range(len(terrains)):
        body, rover_equalities, rover_actuators = prefixed_rover_elements(compiled_root, f"r{index}")
        worldbody.append(body)
        for element in rover_equalities:
            equality.append(element)
        for element in rover_actuators:
            actuators.append(element)
    scene_path = workspace / "showcase_scene.xml"
    ET.ElementTree(root).write(scene_path, encoding="utf-8", xml_declaration=True)
    return mujoco.MjModel.from_xml_path(str(scene_path))


@dataclass
class ShowcaseRover:
    """One policy/observation/recovery state living inside the shared MjData."""

    index: int
    model: mujoco.MjModel
    data: mujoco.MjData
    terrain: ProceduralTerrain
    scenario: RecoveryScenario
    tile_origin_xy: np.ndarray
    history_frames: int
    max_steps: int
    control_dt_s: float = 0.05
    prefix: str = field(init=False)
    drive_ids: list[int] = field(init=False)
    steering_ids: list[int] = field(init=False)
    free_qposadr: int = field(init=False)
    free_dofadr: int = field(init=False)
    root_body_name: str = field(init=False)
    terrain_history: deque[np.ndarray] = field(init=False)
    state_history: deque[np.ndarray] = field(init=False)
    positions: deque[np.ndarray] = field(init=False)
    speeds: deque[float] = field(init=False)
    last_action: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=np.float32))
    start_xy: np.ndarray = field(default_factory=lambda: np.zeros(2))
    previous_escape_distance: float = 0.0
    previous_body_contacts: int = 0
    episode_return: float = 0.0
    step_count: int = 0
    done: bool = False
    success: bool = False

    MOBILITY_WINDOW = 20

    def __post_init__(self) -> None:
        self.prefix = f"r{self.index}"
        self.root_body_name = f"{self.prefix}_prana_mark2_v2"
        self.free_qposadr = self.model.joint(f"{self.prefix}_prana_mark2_v2_freejoint").qposadr[0]
        self.free_dofadr = self.model.joint(f"{self.prefix}_prana_mark2_v2_freejoint").dofadr[0]
        self.steering_ids = [self.model.actuator(f"{self.prefix}_{joint}_hold").id for joint in STEERING_JOINTS]
        self.drive_ids = [self.model.actuator(f"{self.prefix}_{joint}_drive").id for joint in WHEEL_JOINTS]
        self.terrain_history = deque(maxlen=self.history_frames)
        self.state_history = deque(maxlen=self.history_frames)
        self.positions = deque(maxlen=self.MOBILITY_WINDOW + 1)
        self.speeds = deque(maxlen=self.MOBILITY_WINDOW)

    def local_xy(self, world_xy: np.ndarray) -> np.ndarray:
        return world_xy - self.tile_origin_xy

    def terrain_aligned_rotation(self) -> np.ndarray:
        x_m, y_m = self.scenario.spawn_xy_m
        yaw = self.scenario.spawn_yaw_rad
        yaw_rotation = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        wheel_xy = np.asarray([x_m, y_m]) + WHEEL_POSITIONS_M @ yaw_rotation.T
        heights = np.array([self.terrain.height_at(*point) for point in wheel_xy])
        plane, *_ = np.linalg.lstsq(np.column_stack((wheel_xy, np.ones(4))), heights, rcond=None)
        normal = np.array([-plane[0], -plane[1], 1.0])
        normal /= np.linalg.norm(normal)
        heading = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        forward = heading - normal * np.dot(heading, normal)
        forward /= np.linalg.norm(forward)
        return np.column_stack((forward, np.cross(normal, forward), normal))

    def place(self) -> None:
        local_spawn = np.asarray(self.scenario.spawn_xy_m)
        rotation = self.terrain_aligned_rotation()
        qpos = self.data.qpos
        qpos[self.free_qposadr : self.free_qposadr + 3] = (*((self.tile_origin_xy + local_spawn)), 0.0)
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, rotation.ravel())
        qpos[self.free_qposadr + 3 : self.free_qposadr + 7] = quaternion
        mujoco.mj_forward(self.model, self.data)
        root_z = max(
            self.terrain.height_at(*self.local_xy(self.data.body(f"{self.prefix}_wheel_{wheel}").xpos[:2]))
            - self.data.body(f"{self.prefix}_wheel_{wheel}").xpos[2]
            + WHEEL_RADIUS_M
            + 0.015
            for wheel in WHEEL_NAMES
        )
        qpos[self.free_qposadr + 2] = root_z

    def initialize_episode(self) -> None:
        self.start_xy = self.body_xy()
        self.previous_escape_distance = 0.0
        self.previous_body_contacts = self.body_contact_count()
        self.last_action.fill(0.0)
        self.positions.clear()
        self.positions.append(self.start_xy.copy())
        self.speeds.clear()
        terrain, state = self.sensor_frame()
        self.terrain_history.clear()
        self.state_history.clear()
        for _ in range(self.history_frames):
            self.terrain_history.append(terrain.copy())
            self.state_history.append(state.copy())

    def observation(self) -> dict[str, np.ndarray]:
        return {"terrain": np.concatenate(tuple(self.terrain_history)), "state": np.concatenate(tuple(self.state_history))}

    def apply_action(self, action: np.ndarray) -> None:
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        for actuator_id, target in zip(self.drive_ids, action[:4] * PranaRecoveryEnv.MAX_WHEEL_LINEAR_SPEED_M_S / WHEEL_RADIUS_M, strict=True):
            self.data.ctrl[actuator_id] = target
        for actuator_id, target in zip(self.steering_ids, action[4:] * PranaRecoveryEnv.MAX_STEERING_ANGLE_RAD, strict=True):
            self.data.ctrl[actuator_id] = target
        self.pending_action = action

    def update(self) -> tuple[bool, bool]:
        """Update history/reward and return ``(finished, success)``."""
        if self.done:
            return True, self.success
        self.step_count += 1
        position = self.body_xy()
        distance = float(np.linalg.norm(position - self.start_xy))
        progress = distance - self.previous_escape_distance
        self.previous_escape_distance = distance
        self.positions.append(position)
        speed = self.planar_speed()
        self.speeds.append(speed)
        roll, pitch = self.roll_pitch()
        tipped = abs(roll) > 0.85 or abs(pitch) > 0.85
        contacts = self.body_contact_count()
        success = not tipped and contacts == 0 and self.has_sustained_mobility()
        energy = float(np.square(self.pending_action).mean())
        action_change = float(np.square(self.pending_action - self.last_action).mean())
        uprightness = max(0.0, 1.0 - (abs(roll) + abs(pitch)) / 0.70)
        reward = 7.0 * progress + 0.04 * min(speed / 0.45, 1.0) + 0.02 * uprightness
        reward += 0.30 * float(np.clip(self.previous_body_contacts - contacts, -2, 2))
        reward -= 0.01 + 0.020 * energy + 0.015 * action_change + 0.04 * contacts
        if success:
            reward += 25.0
        if tipped:
            reward -= 12.0
        self.episode_return += reward
        self.previous_body_contacts = contacts
        self.last_action = self.pending_action.astype(np.float32)
        terrain, state = self.sensor_frame()
        self.terrain_history.append(terrain)
        self.state_history.append(state)
        self.done = success or tipped or self.step_count >= self.max_steps
        self.success = success
        if self.done:
            for actuator_id in self.drive_ids:
                self.data.ctrl[actuator_id] = 0.0
        return self.done, self.success

    def body_xy(self) -> np.ndarray:
        return self.data.body(self.root_body_name).xpos[:2].copy()

    def body_rotation(self) -> np.ndarray:
        return self.data.body(self.root_body_name).xmat.reshape(3, 3).copy()

    def roll_pitch(self) -> tuple[float, float]:
        rotation = self.body_rotation()
        return math.atan2(rotation[2, 1], rotation[2, 2]), math.atan2(-rotation[2, 0], math.hypot(rotation[2, 1], rotation[2, 2]))

    def planar_speed(self) -> float:
        return float(np.linalg.norm(self.data.qvel[self.free_dofadr : self.free_dofadr + 2]))

    def body_contact_count(self) -> int:
        marker = f"{self.prefix}_collision_"
        count = 0
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            names = (self.model.geom(contact.geom1).name, self.model.geom(contact.geom2).name)
            count += any(name is not None and name.startswith(marker) for name in names)
        return count

    def has_sustained_mobility(self) -> bool:
        if len(self.positions) < self.MOBILITY_WINDOW + 1:
            return False
        positions = np.asarray(self.positions)
        net = float(np.linalg.norm(positions[-1] - positions[0]))
        path = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
        roll, pitch = self.roll_pitch()
        return net >= 0.50 and path >= 0.60 and float(np.mean(self.speeds)) >= 0.15 and abs(roll) < 0.35 and abs(pitch) < 0.35

    def sensor_frame(self) -> tuple[np.ndarray, np.ndarray]:
        rotation = self.body_rotation()
        position = self.data.body(self.root_body_name).xpos
        velocity = rotation.T @ self.data.qvel[self.free_dofadr : self.free_dofadr + 3]
        angular = rotation.T @ self.data.qvel[self.free_dofadr + 3 : self.free_dofadr + 6]
        roll, pitch = self.roll_pitch()
        steering = np.array([self.data.qpos[self.model.joint(f"{self.prefix}_{joint}").qposadr[0]] for joint in STEERING_JOINTS])
        wheel_velocity = np.array([self.data.qvel[self.model.joint(f"{self.prefix}_{joint}").dofadr[0]] for joint in WHEEL_JOINTS])
        suspension = np.array([self.data.qpos[self.model.joint(f"{self.prefix}_{joint}").qposadr[0]] for joint in PranaRecoveryEnv.SUSPENSION_JOINTS])
        wheel_contact = np.zeros(4, dtype=np.float32)
        wheel_geoms = {f"{self.prefix}_wheel_{wheel}_visual": i for i, wheel in enumerate(WHEEL_NAMES)}
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            for geom_id in (contact.geom1, contact.geom2):
                index = wheel_geoms.get(self.model.geom(geom_id).name)
                if index is not None:
                    wheel_contact[index] = 1.0
        state = np.concatenate((velocity, angular, [roll, pitch], steering, wheel_velocity, suspension, wheel_contact, [float(self.body_contact_count())], self.last_action)).astype(np.float32)
        return self.camera_map(rotation, position), state

    def camera_map(self, rotation: np.ndarray, position: np.ndarray) -> np.ndarray:
        directions = _camera_directions() @ rotation.T
        origin = position + rotation @ PranaRecoveryEnv.CAMERA_OFFSET_BODY_M
        distances = np.empty(len(directions))
        hits = np.empty(len(directions), dtype=np.int32)
        mujoco.mj_multiRay(self.model, self.data, origin, directions.ravel(), np.ones(6, dtype=np.uint8), True, self.model.body(self.root_body_name).id, hits, distances, None, len(directions), PranaRecoveryEnv.CAMERA_RANGE_M)
        valid = (distances >= 0.0) & (distances <= PranaRecoveryEnv.CAMERA_RANGE_M)
        points = (origin + directions[valid] * distances[valid, None] - position) @ rotation
        cells = PranaRecoveryEnv.TERRAIN_MAP_CELLS
        elevation = np.full((cells, cells), -np.inf, dtype=np.float32)
        observed = np.full((cells, cells), -1.0, dtype=np.float32)
        if len(points):
            valid = (points[:, 0] >= 0.0) & (points[:, 0] < PranaRecoveryEnv.TERRAIN_MAP_FORWARD_M) & (np.abs(points[:, 1]) < PranaRecoveryEnv.TERRAIN_MAP_HALF_WIDTH_M)
            points = points[valid]
            if len(points):
                rows = np.minimum((points[:, 0] / PranaRecoveryEnv.TERRAIN_MAP_FORWARD_M * cells).astype(np.int32), cells - 1)
                cols = np.minimum(((points[:, 1] + PranaRecoveryEnv.TERRAIN_MAP_HALF_WIDTH_M) / (2 * PranaRecoveryEnv.TERRAIN_MAP_HALF_WIDTH_M) * cells).astype(np.int32), cells - 1)
                np.maximum.at(elevation, (rows, cols), points[:, 2] / PranaRecoveryEnv.TERRAIN_MAP_HEIGHT_SCALE_M)
                observed[rows, cols] = 1.0
        elevation[observed < 0.0] = 0.0
        return np.stack((np.clip(elevation, -1.0, 1.0), observed))


_RAY_DIRECTIONS: np.ndarray | None = None


def _camera_directions() -> np.ndarray:
    global _RAY_DIRECTIONS
    if _RAY_DIRECTIONS is None:
        horizontal = np.linspace(-PranaRecoveryEnv.CAMERA_HORIZONTAL_FOV_RAD / 2, PranaRecoveryEnv.CAMERA_HORIZONTAL_FOV_RAD / 2, PranaRecoveryEnv.CAMERA_WIDTH)
        vertical = np.linspace(math.radians(-48), math.radians(-4), PranaRecoveryEnv.CAMERA_HEIGHT)
        yaw, pitch = np.meshgrid(horizontal, vertical)
        _RAY_DIRECTIONS = np.stack((np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)), axis=-1).reshape(-1, 3)
    return _RAY_DIRECTIONS


def main() -> None:
    args = parse_args()
    if args.rovers < 1 or args.max_steps < 1 or args.history_frames < 1 or args.settle_steps < 0 or args.terrain_height_range <= 0:
        raise SystemExit("--rovers, --max-steps, --history-frames and terrain height range must be positive; --settle-steps cannot be negative.")
    try:
        from stable_baselines3 import PPO
        import mujoco.viewer
    except ImportError as error:
        raise SystemExit("Install requirements-training.txt before running the showcase.") from error

    columns = math.ceil(math.sqrt(args.rovers * 2))
    rows = math.ceil(args.rovers / columns)
    config = TerrainConfig(height_range_m=args.terrain_height_range)
    rng = np.random.default_rng(args.seed)
    terrains = [ProceduralTerrain(config, int(rng.integers(0, np.iinfo(np.int32).max))) for _ in range(args.rovers)]
    scenarios = [ScenarioSampler(terrain, int(rng.integers(0, np.iinfo(np.int32).max))).sample() for terrain in terrains]
    tile_width = 2 * config.half_extent_m

    with tempfile.TemporaryDirectory(prefix="prana_recovery_showcase_") as temporary_directory:
        model = build_showcase_model(Path(temporary_directory), terrains, rows, columns)
        data = mujoco.MjData(model)
        rovers: list[ShowcaseRover] = []
        for index, (terrain, scenario) in enumerate(zip(terrains, scenarios, strict=True)):
            row, column = divmod(index, columns)
            origin = np.array(((column - (columns - 1) / 2) * tile_width, (row - (rows - 1) / 2) * tile_width))
            rover = ShowcaseRover(index, model, data, terrain, scenario, origin, args.history_frames, args.max_steps)
            rover.place()
            rovers.append(rover)
        mujoco.mj_forward(model, data)
        for _ in range(args.settle_steps):
            mujoco.mj_step(model, data)
        for rover in rovers:
            rover.initialize_episode()
        policy = PPO.load(args.model, device=args.device)
        expected_terrain = (2 * args.history_frames, PranaRecoveryEnv.TERRAIN_MAP_CELLS, PranaRecoveryEnv.TERRAIN_MAP_CELLS)
        expected_state = ((PranaRecoveryEnv.PROPRIOCEPTIVE_STATE_SIZE + PranaRecoveryEnv.ACTION_SIZE) * args.history_frames,)
        saved_terrain = policy.observation_space.spaces["terrain"].shape
        saved_state = policy.observation_space.spaces["state"].shape
        if saved_terrain != expected_terrain or saved_state != expected_state:
            raise SystemExit(
                "Model observation layout does not match the current direct-8-action recovery environment: "
                f"model terrain/state={saved_terrain}/{saved_state}, showcase terrain/state={expected_terrain}/{expected_state}. "
                "Use a model trained after the current observation/action-space change."
            )

        print(f"showcase rovers={args.rovers} terrain_tiles={columns}x{rows} seed={args.seed} settle_steps={args.settle_steps}")
        print("This is a visual simultaneous rollout; use evaluate_prana_recovery.py for formal metrics.")
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat[:] = (0.0, 0.0, 0.0)
            viewer.cam.distance = max(columns, rows) * tile_width * 0.9
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -50
            while viewer.is_running() and not all(rover.done for rover in rovers):
                started = time.monotonic()
                for rover in rovers:
                    if not rover.done:
                        action, _ = policy.predict(rover.observation(), deterministic=True)
                        rover.apply_action(action)
                for _ in range(max(1, round(rovers[0].control_dt_s / model.opt.timestep))):
                    mujoco.mj_step(model, data)
                for rover in rovers:
                    was_done = rover.done
                    done, success = rover.update()
                    if done and not was_done:
                        outcome = "success" if success else "failed"
                        print(f"rover={rover.index:02d} terrain_seed={rover.terrain.seed} difficulty={rover.scenario.difficulty:.2f} {outcome} return={rover.episode_return:.2f} steps={rover.step_count}")
                viewer.sync()
                if args.realtime:
                    time.sleep(max(0.0, rovers[0].control_dt_s - (time.monotonic() - started)))
            if all(rover.done for rover in rovers):
                succeeded = sum(rover.success for rover in rovers)
                print(f"showcase complete: successes={succeeded}/{len(rovers)} ({succeeded / len(rovers):.1%})")
                if not args.close_when_done:
                    print("Viewer remains open for free-camera inspection. Close its window when you are done.")
                    while viewer.is_running():
                        viewer.sync()
                        time.sleep(0.02)


if __name__ == "__main__":
    main()
