# Prana Recovery

This package contains a first procedural recovery-learning environment for the
Prana rover. The existing `prana_mujoco_drive` package remains responsible for
the manual 4-wheel-independent-steering teleop and kinematics.

## Procedural recovery environment

`scripts/prana_recovery/terrain.py` creates a seeded heightfield from two
smooth-noise scales plus randomized Gaussian bumps and depressions.
`scenarios.py` samples a rover spawn where local slope and roughness are high,
then chooses a nearby lower-difficulty egress target. Every terrain seed and
scenario seed is retained in the environment `info` dictionary, so an episode
can be replayed.

`env.py` is the Gymnasium boundary. It exposes a normalized body action:

```text
[vx, vy, wz] in [-1, 1]^3
```

The existing 4WIS controller converts that action to the four steering and four
wheel commands. Its observation contains the body-frame egress target, base
linear and angular velocity, roll/pitch, steering state, wheel speeds, and a
front-facing simulated ZED-like depth view.

The depth view uses `65°` horizontal FOV, a `4 m` range, and 48×30 ray samples.
Each ray produces a real MuJoCo surface intersection, together forming a local
3-D point cloud. The environment rasterises it into a 32×32 rover-frame
elevation map plus an observed/unobserved mask. PPO uses `MultiInputPolicy`:
a small CNN processes that terrain map, an MLP processes the rover state, then
their features are fused before producing the 4WIS velocity command. This
replaces the previous 16 omnidirectional horizontal rays, so old MLP policies
cannot be evaluated with this environment and a new policy must be trained.

Terrain relief defaults to `0.65` metres over the 16×16 metre heightfield.
Use `--terrain-height-range` to make the terrain steeper. For example, start
with `1.0` metres or `1.2` metres; use the same value for training and
evaluation.

Before an episode starts, the environment samples terrain below the four wheel
positions, aligns rover roll and pitch to the fitted support plane, and places
the wheels `1.5 cm` above the terrain. It then applies zero control until the
rover is stable. Failed placements are retried internally and never become PPO
episodes. Adjust `--spawn-contact-clearance` only if the wheel geometry needs
more or less initial clearance; use the same value when evaluating a policy.

For inspection of one evaluated rover in MuJoCo's normal interactive free
camera, use `evaluate_prana_recovery.py --mujoco-viewer --episodes 1`.

The terrain generator and sampler have no PPO dependency. A later focused
scenario family can be added beside `ScenarioSampler`, then selected by a new
environment configuration without changing the controller, rewards, or PPO
entry point.

## Training laptop setup

Use Python 3.10 or newer. Install the appropriate PyTorch wheel for the laptop
from the [PyTorch selector](https://pytorch.org/get-started/locally/), choosing
Linux, pip, and the CUDA version supported by that laptop's NVIDIA driver.
Then install the remaining project packages:

```bash
cd ~/Ezio/src/prana_recovery
python3 -m pip install -r requirements-training.txt
```

Verify PyTorch can see the GPU before training:

```bash
python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

## Commands

Run from the workspace root:

```bash
python3 src/prana_recovery/scripts/train_prana_recovery.py \
  --num-envs 8 --timesteps 2000000 --device cuda \
  --terrain-height-range 1.2
```

Training outputs the PPO policy and TensorBoard logs below `runs/prana_recovery`.
Evaluate a saved policy against a repeatable suite:

```bash
python3 src/prana_recovery/scripts/evaluate_prana_recovery.py \
  runs/prana_recovery/ppo_prana_recovery --episodes 50 --seed 10000 \
  --terrain-height-range 1.2
```

The first version intentionally uses slope and roughness as a fast spawn
heuristic. The next useful improvement is a physics probe that rejects starts
where a short scripted drive already escapes or where no nearby egress target
is physically reachable.
