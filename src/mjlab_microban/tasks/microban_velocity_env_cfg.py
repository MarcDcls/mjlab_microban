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

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from mjlab.envs.mdp import dr
from mjlab.envs.mdp.terminations import root_height_below_minimum

from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.scene import SceneCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)
from mjlab.utils.lab_api.math import sample_uniform, quat_apply_inverse
from mjlab.utils.lab_api.string import resolve_matching_names_values
from mjlab.tasks.velocity import mdp as mdp_vel
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, ContactSensor

SCENE_CFG = SceneCfg(
    terrain=TerrainEntityCfg(
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=0,
    ),
    num_envs=1,
    extent=2.0,
    entities={"robot": MICROBAN_ROBOT_CFG},
)

VIEWER_CONFIG = ViewerConfig(
    origin_type=ViewerConfig.OriginType.ASSET_BODY,
    entity_name="robot",
    body_name="trunk",
    distance=3.0,
    elevation=-15.0,
    azimuth=90.0,
)

SIM_CFG = SimulationCfg(
    mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
        ccd_iterations=100,
    ),
    nconmax=256,
    njmax=1024,
)

def make_microban_velocity_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_velocity_env_cfg()

    cfg.viewer = VIEWER_CONFIG
    cfg.sim = SIM_CFG
    cfg.scene = SCENE_CFG

    #---------------------------- Sensors ---------------------------
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

    #---------------------------- Terrain ---------------------------
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    #---------------------------- Actions ---------------------------
    dofs_filter = r".*(?<!head)$"

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0
    cfg.actions["joint_pos"].actuator_names = (dofs_filter,)

    #---------------------------- Observations ----------------------
    del cfg.observations["actor"].terms["base_lin_vel"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    cfg.observations["actor"].terms["joint_pos"] = ObservationTermCfg(
        func=mdp.joint_pos_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(dofs_filter,))},
        noise=Unoise(n_min=-0.01, n_max=0.01),
    )

    cfg.observations["actor"].terms["joint_vel"] = ObservationTermCfg(
        func=mdp.joint_vel_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(dofs_filter,))},
        noise=Unoise(n_min=-0.25, n_max=0.25),
    )

    # cfg.observations["actor"].terms["joint_vel"] = ObservationTermCfg(
    #     func=qvel_smooth_rel,
    #     params={
    #         "action_name": "qvel_filter",
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=(dofs_filter,)),
    #     },
    #     noise=Unoise(n_min=-0.5, n_max=0.5),
    # )

    site_names = ["left_foot", "right_foot"]
    cfg.observations["critic"].terms["foot_height"].params[
        "asset_cfg"
    ].site_names = site_names

    # Observation delays to simulate IMU latency
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

    #---------------------------- Rewards ---------------------------
    cfg.rewards["track_linear_velocity"].params["std"] = np.sqrt(0.1)
    cfg.rewards["track_linear_velocity"].weight = 2.0

    cfg.rewards["track_angular_velocity"].weight = 2.0

    std_standing = {
        r".*head.*": 0.3,
        r".*shoulder_pitch.*": 0.1,
        r".*shoulder_roll.*": 0.1,
        r".*elbow.*": 0.1,
        r".*hip_roll.*": 0.1,
        r".*hip_pitch.*": 0.15,
        r".*hip_yaw.*": 0.1,
        r".*knee.*": 0.15,
        r".*ankle_pitch.*": 0.1,
        r".*ankle_roll.*": 0.1,
    }

    std_walking = {
        r".*head.*": 0.3,
        r".*shoulder_pitch.*": 0.4,
        r".*shoulder_roll.*": 0.2,
        r".*elbow.*": 0.2,
        r".*hip_roll.*": 0.2,
        r".*hip_pitch.*": 0.4,
        r".*hip_yaw.*": 0.2,
        r".*knee.*": 0.4,
        r".*ankle_pitch.*": 0.3,
        r".*ankle_roll.*": 0.2,
    }

    cfg.rewards["pose"].params["std_standing"] = std_standing
    cfg.rewards["pose"].params["std_walking"] = std_walking
    cfg.rewards["pose"].params["std_running"] = std_walking
    cfg.rewards["pose"].params["walking_threshold"] = 0.01
    cfg.rewards["pose"].weight = 1.0

    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk",)
    cfg.rewards["upright"].weight = 1.0
    
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk",)
    cfg.rewards["body_ang_vel"].weight = -0.05

    cfg.rewards["angular_momentum"].weight = -0.02

    for reward_name in ["foot_clearance", "foot_swing_height", "foot_slip"]:
        cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

    cfg.rewards["foot_clearance"].params["command_threshold"] = 0.01
    cfg.rewards["foot_clearance"].params["target_height"] = 0.02

    cfg.rewards["foot_swing_height"].params["command_threshold"] = 0.01
    cfg.rewards["foot_swing_height"].params["target_height"] = 0.02

    cfg.rewards["foot_slip"].params["command_threshold"] = 0.01
    cfg.rewards["foot_slip"].weight = -0.1

    cfg.rewards["air_time"].params["command_threshold"] = 0.01
    cfg.rewards["air_time"].params["threshold_min"] = 0.10
    cfg.rewards["air_time"].params["threshold_max"] = 0.25
    cfg.rewards["air_time"].weight = 1.0

    cfg.rewards["soft_landing"].weight = -1e-05

    cfg.rewards["action_rate_l2"].weight = -0.5

    # cfg.rewards["self_collisions"] = RewardTermCfg(
    #     func=mdp.self_collision_cost,
    #     weight=-1.0,
    #     params={"sensor_name": self_collision_sensor_cfg.name},
    # )

    #---------------------------- Commands --------------------------
    command: UniformVelocityCommandCfg = cfg.commands["twist"]
    command.rel_standing_envs = 0.1
    command.rel_heading_envs = 0.0
    command.viz.z_offset = 0.5

    command.ranges.lin_vel_x = (-0.3, 0.3)
    command.ranges.lin_vel_y = (-0.3, 0.3)

    #---------------------------- Events ----------------------------
    cfg.events["reset_base"].params["pose_range"]["z"] = (0, 0)

    # cfg.events["push_robot"].interval_range_s = (3.0, 6.0)
    cfg.events["push_robot"].params["velocity_range"] = {
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
    }

    cfg.events["foot_friction"].params["asset_cfg"].geom_names = (
        "left_foot_collision",
        "right_foot_collision",
    )

    cfg.events["base_com"].params["ranges"] = {
        0: (-0.005, 0.005),
        1: (-0.005, 0.005),
        2: (-0.005, 0.005),
    }
    cfg.events["base_com"].params["asset_cfg"].body_names = ("trunk",)

    # cfg.events["dof_armature_randomization"] = EventTermCfg(
    #     mode="startup",
    #     func=dr.joint_armature,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
    #         "operation": "scale",
    #         "ranges": (0.5, 1.5),
    #     },
    # )

    # cfg.events["dof_friction_randomization"] = EventTermCfg(
    #     mode="startup",
    #     func=dr.joint_friction,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
    #         "operation": "abs",
    #         "ranges": (0.0, 0.1),
    #     },
    # )

    #---------------------------- Curriculum ------------------------
    del cfg.curriculum["terrain_levels"]

    cfg.curriculum["command_vel"] = CurriculumTermCfg(
        func=mdp.commands_vel,
        params={
            "command_name": "twist",
            "velocity_stages": [
                {
                    "step": 0, 
                    "lin_vel_x": (-0.3, 0.3), 
                    "ang_vel_z": (-0.5, 0.5),
                },
                # {
                #     "step": 5000 * 24,
                #     "lin_vel_x": (-0.5, 0.6),
                #     "ang_vel_z": (-1, 1),
                # },
                # {
                #     "step": 10000 * 24, 
                #     "lin_vel_x": (-0.75, 1.0),
                # },
            ],
        },
    )

    # cfg.curriculum["action_rate_l2"] = CurriculumTermCfg(
    #     func=mdp.reward_weight,
    #     params={
    #         "reward_name": "action_rate_l2",
    #         "weight_stages": [
    #             {"step": 0, "weight": -0.1},
    #             # {"step": 2000 * 24, "weight": -0.15},
    #         ],
    #     },
    # )

    # cfg.curriculum["soft_landing"] = CurriculumTermCfg(
    #     func=mdp.reward_weight,
    #     params={
    #         "reward_name": "soft_landing",
    #         "weight_stages": [
    #             {"step": 0, "weight": -1e-4},
    #             {"step": 2000 * 24, "weight": -5e-4},
    #             {"step": 3000 * 24, "weight": -1e-3},
    #             {"step": 4000 * 24, "weight": -5e-3},
    #         ],
    #     },
    # )

    #---------------------------- Terminations ----------------------
    cfg.terminations["fell_over"] = TerminationTermCfg(
        func=root_height_below_minimum,
        params={"minimum_height": 0.10},
    )

    #---------------------------- Play mode -------------------------
    if play:
        # cfg.events["push_robot"].params["velocity_range"] = {
        #     "x": (0.0, 0.0),
        #     "y": (0.0, 0.0),
        # }

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
