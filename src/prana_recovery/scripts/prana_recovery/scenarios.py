"""Select reproducible rover spawn poses on generated terrain."""

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
    """Choose a relatively difficult, reproducible rover start."""

    def __init__(self, terrain: ProceduralTerrain, seed: int) -> None:
        self.terrain = terrain
        self.rng = np.random.default_rng(seed)

    def sample(self) -> RecoveryScenario:
        terrain = self.terrain
        margin = 2.0
        candidates: list[tuple[float, float, float]] = []
        for _ in range(400):
            x_m, y_m = self.rng.uniform(
                -terrain.config.half_extent_m + margin,
                terrain.config.half_extent_m - margin,
                size=2,
            )
            difficulty = terrain.difficulty_at(x_m, y_m)
            if 0.22 <= difficulty <= 0.82:
                candidates.append((float(x_m), float(y_m), difficulty))

        if not candidates:
            raise RuntimeError("Terrain had no valid recovery spawn candidates; try another terrain seed.")

        weights = np.array([candidate[2] ** 2 for candidate in candidates], dtype=np.float64)
        spawn_x, spawn_y, difficulty = candidates[
            self.rng.choice(len(candidates), p=weights / weights.sum())
        ]
        return RecoveryScenario(
            terrain_seed=terrain.seed,
            spawn_xy_m=(spawn_x, spawn_y),
            spawn_yaw_rad=float(self.rng.uniform(-np.pi, np.pi)),
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
