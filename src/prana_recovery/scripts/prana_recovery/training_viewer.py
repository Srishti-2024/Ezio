"""One-window mosaic for watching the real PPO workers while they train."""

from __future__ import annotations

import math


class TrainingMosaicCallback:
    """Stable-Baselines callback built lazily to keep headless training light.

    ``SubprocVecEnv`` keeps each rover in a different process and MuJoCo world.
    This callback asks selected workers for RGB frames, then displays those
    frames as a grid in one Matplotlib window.  It never creates duplicate
    showcase physics: every tile is a worker PPO is currently training on.
    """

    def __new__(cls, *, every_steps: int, worker_count: int):
        try:
            import matplotlib.pyplot as plt
            from stable_baselines3.common.callbacks import BaseCallback
        except ImportError as error:
            raise RuntimeError(
                "Live training display needs matplotlib and Stable-Baselines3."
            ) from error

        class _Callback(BaseCallback):
            def __init__(self) -> None:
                super().__init__()
                self._plt = plt
                self._every_steps = every_steps
                self._worker_indices = list(range(worker_count))
                self._figure = None
                self._axes = []
                self._images = []

            def _on_training_start(self) -> None:
                columns = math.ceil(math.sqrt(worker_count))
                rows = math.ceil(worker_count / columns)
                self._plt.ion()
                self._figure, axes = self._plt.subplots(
                    rows, columns, figsize=(4.8 * columns, 3.6 * rows)
                )
                self._axes = list(getattr(axes, "flat", [axes]))
                for index, axis in enumerate(self._axes):
                    axis.set_axis_off()
                    if index < worker_count:
                        axis.set_title(f"PPO worker {index}")
                self._figure.canvas.manager.set_window_title("Prana PPO live training")
                self._figure.tight_layout()

            def _on_step(self) -> bool:
                if self.n_calls % self._every_steps:
                    return True
                frames = self.training_env.env_method("render", indices=self._worker_indices)
                for index, frame in enumerate(frames):
                    if index >= len(self._images):
                        self._images.append(self._axes[index].imshow(frame))
                    else:
                        self._images[index].set_data(frame)
                assert self._figure is not None
                self._figure.canvas.draw_idle()
                self._figure.canvas.flush_events()
                self._plt.pause(0.001)
                return True

            def _on_training_end(self) -> None:
                if self._figure is not None:
                    self._plt.ioff()

        return _Callback()
