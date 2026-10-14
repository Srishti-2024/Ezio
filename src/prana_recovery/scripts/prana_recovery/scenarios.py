"""Select reproducible rover spawn and egress points on generated terrain."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from .terrain import ProceduralTerrain, TerrainConfig


@dataclass(frozen=True)
class RecoveryScenario:
    """One rover start pose and nearby target on a specific terrain seed."""

    terrain_seed: int
    spawn_xy_m: tuple[float, float]
    spawn_yaw_rad: float
    egress_xy_m: tuple[float, float]
    difficulty: float


class ScenarioSampler:
    """Choose a relatively difficult start and a nearby easier target."""

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
        target_x, target_y = self._find_egress_target(spawn_x, spawn_y)
        return RecoveryScenario(
            terrain_seed=terrain.seed,
            spawn_xy_m=(spawn_x, spawn_y),
            spawn_yaw_rad=float(self.rng.uniform(-np.pi, np.pi)),
            egress_xy_m=(target_x, target_y),
            difficulty=float(difficulty),
        )

    def _find_egress_target(self, spawn_x: float, spawn_y: float) -> tuple[float, float]:
        terrain = self.terrain
        best: tuple[float, float, float] | None = None
        for angle in np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False):
            distance = self.rng.uniform(1.5, 2.6)
            x_m = spawn_x + distance * np.cos(angle)
            y_m = spawn_y + distance * np.sin(angle)
            edge = terrain.config.half_extent_m - 1.2
            if abs(x_m) > edge or abs(y_m) > edge:
                continue
            score = terrain.difficulty_at(x_m, y_m)
            if best is None or score < best[0]:
                best = (score, float(x_m), float(y_m))

        if best is None:
            raise RuntimeError("Could not place an egress target within terrain bounds.")
        return best[1], best[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample one recovery scenario on procedural terrain.")
    parser.add_argument("--terrain-seed", type=int, default=7)
    parser.add_argument("--scenario-seed", type=int, default=107)
    args = parser.parse_args()

    terrain = ProceduralTerrain(TerrainConfig(), seed=args.terrain_seed)
    scenario = ScenarioSampler(terrain, seed=args.scenario_seed).sample()
    target_x, target_y = scenario.egress_xy_m
    print(f"terrain_seed={scenario.terrain_seed}")
    print(f"spawn_xy_m=({scenario.spawn_xy_m[0]:.2f}, {scenario.spawn_xy_m[1]:.2f})")
    print(f"spawn_yaw_rad={scenario.spawn_yaw_rad:.2f}")
    print(f"spawn_difficulty={scenario.difficulty:.3f}")
    print(f"egress_xy_m=({target_x:.2f}, {target_y:.2f})")
    print(f"egress_difficulty={terrain.difficulty_at(target_x, target_y):.3f}")


if __name__ == "__main__":
    main()
