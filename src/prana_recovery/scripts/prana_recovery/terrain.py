"""Seeded procedural heightfields for rover recovery episodes."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TerrainConfig:
    """Parameters defining one reproducible terrain family."""

    half_extent_m: float = 8.0
    resolution: int = 129
    height_range_m: float = 0.65
    noise_amplitude_m: float = 0.24
    detail_amplitude_m: float = 0.09
    feature_count: int = 16
    feature_amplitude_m: float = 0.28
    feature_radius_range_m: tuple[float, float] = (0.22, 0.85)
    ditch_count: int = 6
    ditch_width_range_m: tuple[float, float] = (0.35, 0.75)
    ditch_length_range_m: tuple[float, float] = (2.0, 4.5)
    ditch_depth_m: float = 0.65


@dataclass(frozen=True)
class DitchFeature:
    """One seeded elongated depression available to the scenario sampler."""

    center_xy_m: tuple[float, float]
    yaw_rad: float
    width_m: float
    length_m: float


class ProceduralTerrain:
    """A heightfield generated only from a seed and :class:`TerrainConfig`."""
    """ Generates a 12x12m terrain with resolultion of 129x 129 """
    def __init__(self, config: TerrainConfig, seed: int) -> None:
        self.config = config
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.ditches: list[DitchFeature] = []
        self.heights_m = self._generate()
        self.cell_size_m = 2.0 * config.half_extent_m / (config.resolution - 1)
        gradient_y, gradient_x = np.gradient(self.heights_m, self.cell_size_m, self.cell_size_m)
        self.slope = np.hypot(gradient_x, gradient_y)
        self.roughness = self._local_roughness(radius_cells=3)

    def _value_noise(self, grid_size: int) -> np.ndarray:
        """ Fake perlin not actual perlin noise but works for now """
        """ Just generate random values and interpolates between those for smoothness """
        """Smooth value noise, sufficient here without a Perlin dependency."""
        coarse = self.rng.standard_normal((grid_size, grid_size))
        source = np.linspace(0.0, grid_size - 1, self.config.resolution)
        intermediate = np.array([np.interp(source, np.arange(grid_size), row) for row in coarse])
        return np.array([np.interp(source, np.arange(grid_size), intermediate[:, column]) for column in range(self.config.resolution)]).T

    def _generate(self) -> np.ndarray:
        """ Generates a 2d heightmap using just the seed"""
        config = self.config
        terrain = (
            config.noise_amplitude_m * self._value_noise(9)
            + config.detail_amplitude_m * self._value_noise(27)
        )
        coords = np.linspace(-config.half_extent_m, config.half_extent_m, config.resolution)
        x, y = np.meshgrid(coords, coords, indexing="xy")
        margin = 1.5
        for _ in range(config.feature_count):
            center_x, center_y = self.rng.uniform(-config.half_extent_m + margin, config.half_extent_m - margin, size=2)
            radius = self.rng.uniform(*config.feature_radius_range_m)
            amplitude = self.rng.uniform(-config.feature_amplitude_m, config.feature_amplitude_m)
            terrain += amplitude * np.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / (2.0 * radius**2))

        # Smooth random noise alone mostly creates slopes that a 4WIS rover
        # can simply drive across. Add explicit elongated valleys so resets can
        # begin in a geometrically meaningful ditch rather than on a hill.
        ditch_margin = 2.8
        for _ in range(config.ditch_count):
            center_x, center_y = self.rng.uniform(
                -config.half_extent_m + ditch_margin,
                config.half_extent_m - ditch_margin,
                size=2,
            )
            yaw_rad = float(self.rng.uniform(-np.pi, np.pi))
            width_m = float(self.rng.uniform(*config.ditch_width_range_m))
            length_m = float(self.rng.uniform(*config.ditch_length_range_m))
            self.ditches.append(
                DitchFeature(
                    center_xy_m=(float(center_x), float(center_y)),
                    yaw_rad=yaw_rad,
                    width_m=width_m,
                    length_m=length_m,
                )
            )
            along = (x - center_x) * np.cos(yaw_rad) + (y - center_y) * np.sin(yaw_rad)
            across = -(x - center_x) * np.sin(yaw_rad) + (y - center_y) * np.cos(yaw_rad)
            # A super-Gaussian keeps the ditch floor relatively flat while
            # retaining steep shoulders that make climbing out non-trivial.
            shape = np.exp(-0.5 * ((along / (length_m / 2.0)) ** 8 + (across / width_m) ** 4))
            terrain -= config.ditch_depth_m * shape

        terrain -= terrain.min()
        maximum = float(terrain.max())
        if maximum > 1e-8:
            terrain *= config.height_range_m / maximum
        return terrain.astype(np.float64)

    def _local_roughness(self, radius_cells: int) -> np.ndarray:
        """ Calculates local roughenss gives a path and the 2d heightmap """
        padded = np.pad(self.heights_m, radius_cells, mode="edge")
        result = np.empty_like(self.heights_m)
        width = 2 * radius_cells + 1
        for row in range(self.config.resolution):
            for column in range(self.config.resolution):
                window = padded[row : row + width, column : column + width]
                result[row, column] = window.std()
        return result

    def _grid_coordinates(self, x_m: float, y_m: float) -> tuple[float, float]:
        normalized_x = (x_m + self.config.half_extent_m) / (2.0 * self.config.half_extent_m)
        normalized_y = (y_m + self.config.half_extent_m) / (2.0 * self.config.half_extent_m)
        scale = self.config.resolution - 1
        return np.clip(normalized_x * scale, 0.0, scale), np.clip(normalized_y * scale, 0.0, scale)

    def sample(self, values: np.ndarray, x_m: float, y_m: float) -> float:
        """ Finds value(height) at point (x,y ) since this is usually not a grid point it billinearly interpolates to find value"""
        """Bilinearly interpolate a terrain metric at world XY."""
        column, row = self._grid_coordinates(x_m, y_m)
        c0, r0 = int(column), int(row)
        c1, r1 = min(c0 + 1, self.config.resolution - 1), min(r0 + 1, self.config.resolution - 1)
        dc, dr = column - c0, row - r0
        return float(
            (1.0 - dr) * ((1.0 - dc) * values[r0, c0] + dc * values[r0, c1])
            + dr * ((1.0 - dc) * values[r1, c0] + dc * values[r1, c1])
        )

    def height_at(self, x_m: float, y_m: float) -> float:
        return self.sample(self.heights_m, x_m, y_m)

    def difficulty_at(self, x_m: float, y_m: float) -> float:
        # Calculates diffuctily from slope and roughness, which initself is calculated from the 2d heightmap
        slope_score = min(self.sample(self.slope, x_m, y_m) / 1.2, 1.0)
        roughness_score = min(self.sample(self.roughness, x_m, y_m) / 0.12, 1.0)
        return 0.7 * slope_score + 0.3 * roughness_score

    def write_hfield(self, path: Path) -> None:
        """ Final heightfield output to mujoco """
        """Write MuJoCo's normalized row-major heightfield file."""
        normalized = self.heights_m / self.config.height_range_m
        with path.open("wb") as output:
            np.asarray(normalized.shape, dtype=np.int32).tofile(output)
            normalized.astype(np.float32).tofile(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and summarize procedural Prana terrain.")
    parser.add_argument("--seed", type=int, default=7, help="First deterministic terrain seed")
    parser.add_argument("--count", type=int, default=3, help="Number of consecutive terrain seeds to generate")
    args = parser.parse_args()

    for seed in range(args.seed, args.seed + args.count):
        terrain = ProceduralTerrain(TerrainConfig(), seed)
        print(
            f"seed={seed} shape={terrain.heights_m.shape} "
            f"height=[{terrain.heights_m.min():.3f}, {terrain.heights_m.max():.3f}] m "
            f"slope=[{terrain.slope.min():.3f}, {terrain.slope.max():.3f}] "
            f"roughness=[{terrain.roughness.min():.3f}, {terrain.roughness.max():.3f}] m "
            f"center_difficulty={terrain.difficulty_at(0.0, 0.0):.3f}"
        )


if __name__ == "__main__":
    main()
