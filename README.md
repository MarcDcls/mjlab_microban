# MjLab Microban

[![License: Apache-2.0](https://img.shields.io/badge/Software-Apache--2.0-yellow.svg)](LICENSE)

<!-- <img src="" align="right" height="350px"> -->

This repository contains Reinforcement Learning (RL) environments for Microban, a compact, low-cost, fully open-source small humanoid robot. If you are interested in learning more about Microban, or even building your own, check out the [Microban repository](https://github.com/MarcDcls/microban).

The environments are built using the [MjLab](https://github.com/mujocolab/mjlab) framework.
A velocity control task is currently implemented, allowing the robot to follow target linear and angular velocities while resisting external disturbances.

<!-- Add gif of trained agent here -->

## Install

To install the repository, you need the uv package manager.
If you don't have it yet, you can install it by following the instructions [here](https://docs.astral.sh/uv/getting-started/installation/#installation-methods).

Then, clone this repository and run the following command in your terminal:
```
uv sync
```

## Using a velocity agent

<p align="center">
  <img width="480" alt="MicrobanSimu" src="https://github.com/user-attachments/assets/fa79d712-e2ff-4452-b3ef-7ac41b87ff13" />
</p>

You can use a pre-trained agent directly in its MjLab environment (GPU required), where random velocity commands are given to the robot at regular intervals.
Linear velocity commands are represented by a blue arrow, while angular velocity commands are represented by a green vertical one.

```
uv run play Mjlab-Velocity-Microban --checkpoint-file src/mjlab_microban/agents/velocity.pt
```

To push the robot while playing, double-click on the trunk in the simulation window, then hold the left-ctrl key and right-click and drag to apply a force.

## Training your own agent

You can modify the environment configuration at `src/mjlab_microban/tasks/microban_velocity_env_cfg.py`.

To test the environment before training, play with a zero or random agent:

```
uv run play Mjlab-Velocity-Microban --agent zero
uv run play Mjlab-Velocity-Microban --agent random
```

Start the training with:

```
uv run train Mjlab-Velocity-Microban --env.scene.num-envs 4096
```

Once training is complete, play back a checkpoint with:

```
uv run play Mjlab-Velocity-Microban --checkpoint-file [path to your checkpoint]
```

Where `[path to your checkpoint]` is typically located at `logs/rsl_rl/mjlab_microban_velocity/[date]/model_[number].pt`.

You can also play back the last checkpoint in wandb with:

```
uv run play Mjlab-Velocity-Microban --wandb-run-path [path to your wandb run]
```

Where `[path to your wandb run]` is available in the Overview tab of your wandb run.

## Exporting a policy to ONNX

A ONNX is generated during training with the latest checkpoint, but if you want to export a specific checkpoint, you can do so with the following command:

```
uv run python src/mjlab_microban/scripts/export_onnx.py --checkpoint [path to your checkpoint]
```

## Transferring to the real robot

The transfer on the real robot is always a challenge due to the sim-to-real gap. However, the policies trained in this repository have been successfully transferred to the real Microban robot. It is possible due to a combination of domain randomization and a well-tuned modelisation of the actuators (delays, friction, voltage drop, current clipping, etc.). This modelisation is done using the [BAM](https://github.com/Rhoban/bam) library.

Here is a video of the trained agent being transferred to the real robot: [https://youtu.be/1pnFrT_jfXQ](https://youtu.be/1pnFrT_jfXQ)

<p align="center">
  <img width="70%" alt="image" src="https://github.com/user-attachments/assets/dd91b082-faf0-4c73-a216-fe9b633f51b3" />
</p>

## License

This software is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details.
