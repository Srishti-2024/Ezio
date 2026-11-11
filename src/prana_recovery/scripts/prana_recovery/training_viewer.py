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

    def __new__(cls, *, every_steps: int, worker_count: int, show_precheck: bool = False):
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
                self._show_precheck = show_precheck
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
                if self._show_precheck:
                    probe_frames = self.training_env.env_method(
                        "consume_precheck_frames", indices=self._worker_indices
                    )
                    self._replay_prechecks(probe_frames)
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

            def _replay_prechecks(self, probe_frames) -> None:
                """Replay accepted-start probes in the same worker mosaic."""
                frame_count = max((len(worker_frames) for worker_frames in probe_frames), default=0)
                for frame_index in range(frame_count):
                    for worker_index, worker_frames in enumerate(probe_frames):
                        if frame_index >= len(worker_frames):
                            continue
                        label, frame = worker_frames[frame_index]
                        if worker_index >= len(self._images):
                            self._images.append(self._axes[worker_index].imshow(frame))
                        else:
                            self._images[worker_index].set_data(frame)
                        self._axes[worker_index].set_title(f"worker {worker_index}: {label}")
                    assert self._figure is not None
                    self._figure.canvas.draw_idle()
                    self._figure.canvas.flush_events()
                    self._plt.pause(0.08)

            def _on_training_end(self) -> None:
                if self._figure is not None:
                    self._plt.ioff()

        return _Callback()


class EvaluationFrameViewer:
    """A live evaluation window with an optional second rover-following view."""

    def __init__(self, *, alternate_view: bool = False) -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError as error:
            raise RuntimeError("Live evaluation display needs matplotlib.") from error

        self._plt = plt
        self._plt.ion()
        self._alternate_view = alternate_view
        if alternate_view:
            self._figure, axes = self._plt.subplots(1, 2, figsize=(12.8, 4.8))
            self._axes = list(axes)
        else:
            self._figure, axis = self._plt.subplots(figsize=(6.4, 4.8))
            self._axes = [axis]
        for axis in self._axes:
            axis.set_axis_off()
        self._images = [None] * len(self._axes)
        self._figure.canvas.manager.set_window_title("Prana PPO evaluation")
        self._figure.tight_layout()

    def update(self, frame, *, episode: int, step: int, alternate_frame=None) -> None:
        """Display the exact primary/alternate frames optionally written to MP4."""
        frames = [frame]
        if self._alternate_view:
            if alternate_frame is None:
                raise ValueError("alternate_frame is required when alternate_view is enabled.")
            frames.append(alternate_frame)
        for index, (axis, current) in enumerate(zip(self._axes, frames, strict=True)):
            if self._images[index] is None:
                self._images[index] = axis.imshow(current)
            else:
                self._images[index].set_data(current)
            label = "primary" if index == 0 else "opposite overhead"
            axis.set_title(f"Episode {episode}, step {step}: {label}")
        self._figure.canvas.draw_idle()
        self._figure.canvas.flush_events()
        self._plt.pause(0.001)

    def replay_precheck(self, frames, *, episode: int) -> None:
        """Show the accepted start's recorded forward/crab/yaw probes."""
        for label, frame in frames:
            if self._images[0] is None:
                self._images[0] = self._axes[0].imshow(frame)
            else:
                self._images[0].set_data(frame)
            self._axes[0].set_title(f"Evaluation episode {episode}: {label}")
            self._figure.canvas.draw_idle()
            self._figure.canvas.flush_events()
            self._plt.pause(0.12)

    def close(self) -> None:
        self._plt.ioff()
