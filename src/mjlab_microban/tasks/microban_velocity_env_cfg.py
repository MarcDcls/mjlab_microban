"""Microban velocity environment"""

import numpy as np
import math
import pickle
import torch
from copy import deepcopy
from dataclasses import dataclass

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab_microban.robot.microban_constants import MICROBAN_ROBOT_CFG, HOME_FRAME
from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.tasks.velocity.mdp import Entity, UniformVelocityCommandCfg

from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.envs.mdp import dr

from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.lab_api.math import quat_apply_inverse

from mjlab_microban.tasks.mdp import *


def make_microban_velocity_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_velocity_env_cfg()

    cfg.viewer.body_name = "trunk"
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk",)
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk",)

    cfg.scene.entities = {"robot": MICROBAN_ROBOT_CFG}

    ##
    # Sensors configuration
    ##

    feet_ground_sensor_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(foot|foot_2)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    self_collision_sensor_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    cfg.scene.sensors = (feet_ground_sensor_cfg, self_collision_sensor_cfg)

    ##
    # Action adjustment
    ##

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    ##
    # Pose reward adjustment
    ##

    std_standing = {
        # Head
        r".*Head_Yaw.*": 0.3,
        r".*Head_Pitch.*": 0.3,
        # Lower body.
        r".*Knee.*": 0.05,
        r".*Ankle_Pitch.*": 0.05,
        r".*Ankle_Roll.*": 0.05,
        # Waist.
        r".*Hip_Roll.*": 0.05,
        r".*Hip_Pitch.*": 0.05,
        r".*Hip_Yaw.*": 0.05,
        # Arms.
        r".*Shoulder_Pitch.*": 0.1,
        r".*Shoulder_Roll.*": 0.1,
        r".*Elbow.*": 0.1,
    }

    std_walking = {
        # Head
        r".*Head_Yaw.*": 0.3,
        r".*Head_Pitch.*": 0.3,
        # Lower body.
        r".*Knee.*": 0.4,
        r".*Ankle_Pitch.*": 0.15,
        r".*Ankle_Roll.*": 0.15,
        # Waist.
        r".*Hip_Roll.*": 0.2,
        r".*Hip_Pitch.*": 0.4,
        r".*Hip_Yaw.*": 0.3,
        # Arms.
        r".*Shoulder_Pitch.*": 0.4,
        r".*Shoulder_Roll.*": 0.3,
        r".*Elbow.*": 0.25,
    }

    # cfg.rewards["pose"].params["std_standing"] = std_standing
    # cfg.rewards["pose"].params["std_walking"] = std_walking
    # cfg.rewards["pose"].params["std_running"] = std_walking
    del cfg.rewards["pose"]

    ##
    # Specifying site names for foot-related rewards
    ##

    site_names = ["left_foot", "right_foot"]

    for reward_name in ["foot_clearance", "foot_swing_height", "foot_slip"]:
        cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

    # Removing specific rewards
    # cfg.rewards["foot_swing_height"].weight = 0.0
    # cfg.rewards["foot_slip"].weight = 0.0
    # cfg.rewards["foot_clearance"].weight = 0.0
    # cfg.rewards["soft_landing"].weight = 0.0

    cfg.observations["critic"].terms["foot_height"].params[
        "asset_cfg"
    ].site_names = site_names

    ##
    # Self-collision soft reward
    ##

    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_sensor_cfg.name},
    )

    ##
    # Feet friction reward configuration
    ##

    foot_frictions_geom_names = (
        "foot",
        "foot_2",
    )
    cfg.events["foot_friction"].params[
        "asset_cfg"
    ].geom_names = foot_frictions_geom_names

    ##
    # Adjusting reward weights
    ##

    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02
    cfg.rewards["air_time"].weight = 0.0
    cfg.rewards["soft_landing"].weight = -1e-3

    # Removing base lin velocity observation
    del cfg.observations["actor"].terms["base_lin_vel"]

    # Removing height sensor for now
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    ##
    # Masking Head in observation and action
    ##

    dofs_filter = r".*(?<!Head_Yaw)(?<!Head_Pitch)$"

    cfg.observations["actor"].terms["joint_pos"] = ObservationTermCfg(
        func=mdp.joint_pos_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(dofs_filter,))},
        noise=Unoise(n_min=-0.01, n_max=0.01),
    )

    # cfg.observations["actor"].terms["joint_vel"] = ObservationTermCfg(
    #     func=qvel_smooth_rel,
    #     params={
    #         "action_name": "qvel_filter",
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=(dofs_filter,)),
    #     },
    #     noise=Unoise(n_min=-0.5, n_max=0.5),
    # )

    cfg.observations["actor"].terms["joint_vel"] = ObservationTermCfg(
        func=mdp.joint_vel_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(dofs_filter,))},
        noise=Unoise(n_min=-0.25, n_max=0.25),
    )

    cfg.actions["joint_pos"].actuator_names = (dofs_filter,)

    cfg.events["reset_base"].params["pose_range"]["z"] = (0, 0)

    cfg.commands["twist"].viz.z_offset = 1.0

    ##
    # Walking on plane only
    ##
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    ##
    # Disabling default curriculum
    ##
    del cfg.curriculum["terrain_levels"]

    cfg.curriculum["command_vel"] = CurriculumTermCfg(
        func=mdp.commands_vel,
        params={
            "command_name": "twist",
            "velocity_stages": [
                {"step": 0, "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-0.5, 0.5)},
                {
                    "step": 1_500 * 24,
                    "lin_vel_x": (-1.5, 2.0),
                    "ang_vel_z": (-1.5, 1.5),
                },
                {"step": 3_000 * 24, "lin_vel_z": (-3.0, 3.0)},
            ],
        },
    )

    # Increasing action rate over time
    cfg.curriculum["action_rate_l2"] = CurriculumTermCfg(
        func=mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": -0.1},
                {"step": 24 * 2_000, "weight": -0.15},
            ],
        },
    )

    cfg.curriculum["soft_landing"] = CurriculumTermCfg(
        func=mdp.reward_weight,
        params={
            "reward_name": "soft_landing",
            "weight_stages": [
                {"step": 0, "weight": -1e-4},
                {"step": 1_000 * 24, "weight": -5e-4},
                {"step": 2_000 * 24, "weight": -1e-3},
                {"step": 3_000 * 24, "weight": -5e-3},
            ],
        },
    )

    # Adjusting push ranges
    cfg.events["push_robot"].params["velocity_range"] = {
        "x": (-0.75, 0.75),
        "y": (-0.75, 0.75),
    }

    # Slightly increased L2 action rate penalty
    cfg.rewards["action_rate_l2"].weight = -0.1

    # More standing env, disabling heading envs
    command: UniformVelocityCommandCfg = cfg.commands["twist"]
    command.rel_standing_envs = 0.25
    command.rel_heading_envs = 0.0

    cfg.observations["actor"].terms["projected_gravity"] = deepcopy(
        cfg.observations["actor"].terms["projected_gravity"]
    )
    cfg.observations["actor"].terms["base_ang_vel"] = deepcopy(
        cfg.observations["actor"].terms["base_ang_vel"]
    )

    cfg.observations["actor"].terms["base_ang_vel"].delay_min_lag = 0
    cfg.observations["actor"].terms["base_ang_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["base_ang_vel"].delay_update_period = 64

    cfg.observations["actor"].terms["projected_gravity"].delay_min_lag = 1
    cfg.observations["actor"].terms["projected_gravity"].delay_max_lag = 1
    cfg.observations["actor"].terms["projected_gravity"].delay_update_period = 64

    # Randomization on CoM positions
    cfg.events["base_com"].params["ranges"] = {
        0: (-0.025, 0.025),
        1: (-0.05, 0.05),
        2: (-0.05, 0.05),
    }
    cfg.events["base_com"].params["asset_cfg"].body_names = ("trunk",)

    # Slightly randomizing armatures
    cfg.events["dof_armature_randomization"] = EventTermCfg(
        mode="startup",
        func=dr.joint_armature,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=(r".*",)
            ),  # Set per-robot.
            "operation": "scale",
            "ranges": (0.5, 1.5),
        },
    )

    cfg.events["dof_friction_randomization"] = EventTermCfg(
        mode="startup",
        func=dr.joint_friction,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=(r".*",)
            ),  # Set per-robot.
            "operation": "abs",
            "ranges": (0.0, 0.1),
        },
    )

    if play:
        # Disabling push when in play
        cfg.events["push_robot"].params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
        }

        # Custom command
        # cfg.commands["twist"].ranges.ang_vel_z = (0.0, 0.0)
        # cfg.commands["twist"].ranges.lin_vel_y = (0.0, 0.0)
        # cfg.commands["twist"].ranges.lin_vel_x = (0.0, 0.0)
        # cfg.commands["twist"].rel_standing_envs = 0.0

        cfg.observations["actor"].enable_corruption = False

        # Can be used to print something every step
        def debug(env: ManagerBasedRlEnv, _): ...

        cfg.events["debug"] = EventTermCfg(
            func=debug, mode="interval", interval_range_s=(0.0, 0.0)
        )

    return cfg


MicrobanVelocityRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    ),
    wandb_project="mjlab_microban_velocity",
    experiment_name="mjlab_microban_velocity",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=30_000,
)
