"""Neural-network feature extractor for camera-based recovery observations."""

from __future__ import annotations

import torch as th
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class RoverTerrainExtractor(BaseFeaturesExtractor):
    """Fuse a local terrain elevation map with the rover's numeric state."""

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256) -> None:
        super().__init__(observation_space, features_dim)
        terrain_shape = observation_space.spaces["terrain"].shape
        state_dim = observation_space.spaces["state"].shape[0]
        assert terrain_shape is not None

        self.terrain_cnn = th.nn.Sequential(
            th.nn.Conv2d(terrain_shape[0], 16, kernel_size=5, stride=2),
            th.nn.ReLU(),
            th.nn.Conv2d(16, 32, kernel_size=3, stride=2),
            th.nn.ReLU(),
            th.nn.Conv2d(32, 32, kernel_size=3, stride=1),
            th.nn.ReLU(),
            th.nn.Flatten(),
        )
        with th.no_grad():
            terrain_features = self.terrain_cnn(th.zeros((1, *terrain_shape))).shape[1]
        self.terrain_projection = th.nn.Sequential(
            th.nn.Linear(terrain_features, 128), th.nn.ReLU()
        )
        self.state_mlp = th.nn.Sequential(th.nn.Linear(state_dim, 64), th.nn.ReLU())
        self.fusion = th.nn.Sequential(
            th.nn.Linear(128 + 64, features_dim), th.nn.ReLU()
        )

    def forward(self, observations: dict[str, th.Tensor]) -> th.Tensor:
        terrain = self.terrain_projection(self.terrain_cnn(observations["terrain"]))
        state = self.state_mlp(observations["state"])
        return self.fusion(th.cat((terrain, state), dim=1))
