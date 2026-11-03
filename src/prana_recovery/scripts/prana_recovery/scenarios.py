"""Select reproducible rover spawn poses in generated ditches."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from .terrain import ProceduralTerrain, TerrainConfig


@dataclass(frozen=True)
class RecoveryScenario:
    """One rover start pose on a specific terrain seed."""

    terrain_seed: int
    spawn_xy_m: tuple[float, float]
    spawn_yaw_rad: float
    difficulty: float


class ScenarioSampler:
    """Choose a reproducible start on the floor of a procedural ditch."""

    def __init__(self, terrain: ProceduralTerrain, seed: int) -> None:
        self.terrain = terrain
        self.rng = np.random.default_rng(seed)

    def sample(self) -> RecoveryScenario:
        terrain = self.terrain
        if not terrain.ditches:
            raise RuntimeError("Terrain has no procedural ditches; set TerrainConfig.ditch_count above zero.")

        ditch = terrain.ditches[int(self.rng.integers(len(terrain.ditches)))]
        # Keep the rover on the valley floor, with modest longitudinal jitter.
        # Its heading points across a ditch wall, making the initial recovery
        # question 'can I climb out?' rather than 'can I drive along the ditch?'
        along_m = float(self.rng.uniform(-0.20, 0.20) * ditch.length_m)
        across_m = float(self.rng.uniform(-0.12, 0.12) * ditch.width_m)
        forward = np.array([np.cos(ditch.yaw_rad), np.sin(ditch.yaw_rad)])
        left = np.array([-np.sin(ditch.yaw_rad), np.cos(ditch.yaw_rad)])
        spawn_xy = np.asarray(ditch.center_xy_m) + along_m * forward + across_m * left
        spawn_x, spawn_y = (float(spawn_xy[0]), float(spawn_xy[1]))
        difficulty = terrain.difficulty_at(spawn_x, spawn_y)
        return RecoveryScenario(
            terrain_seed=terrain.seed,
            spawn_xy_m=(spawn_x, spawn_y),
            spawn_yaw_rad=float(ditch.yaw_rad + np.pi / 2.0 + self.rng.uniform(-0.20, 0.20)),
            difficulty=float(difficulty),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample one recovery scenario on procedural terrain.")
    parser.add_argument("--terrain-seed", type=int, default=7)
    parser.add_argument("--scenario-seed", type=int, default=107)
    args = parser.parse_args()

    terrain = ProceduralTerrain(TerrainConfig(), seed=args.terrain_seed)
    scenario = ScenarioSampler(terrain, seed=args.scenario_seed).sample()
    print(f"terrain_seed={scenario.terrain_seed}")
    print(f"spawn_xy_m=({scenario.spawn_xy_m[0]:.2f}, {scenario.spawn_xy_m[1]:.2f})")
    print(f"spawn_yaw_rad={scenario.spawn_yaw_rad:.2f}")
    print(f"spawn_difficulty={scenario.difficulty:.3f}")


if __name__ == "__main__":
    main()
