# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Microban velocity environment"""

import numpy as np
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from mjlab_microban.robot.microban_constants import HOME_FRAME, MICROBAN_ROBOT_CFG
from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.envs.mdp import dr

from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
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
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, ObjRef, TerrainHeightSensorCfg, RingPatternCfg

from mjlab_microban.tasks.mdp import (
    reward_based_staged_curriculum,
    reward_based_curriculum,
    step_based_staged_curriculum,
    set_command_velocity,
    set_stepping_parameters,
    set_push_parameters,
    no_stepping_penalty, 
    penalize_stepping_while_standing,
    stepping_curriculum,
    UniformVelocityCommandWithRotation,
    upright as local_upright,
)
from mjlab_microban.tasks.reference import (
    ReferenceCommandCfg,
    reference_phase,
    reference_pose_reward,
)

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
    # nconmax=256,
    # njmax=1024,
)

def make_microban_velocity_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_velocity_env_cfg()

    cfg.viewer = deepcopy(VIEWER_CONFIG)
    cfg.sim = deepcopy(SIM_CFG)
    cfg.scene = deepcopy(SCENE_CFG)

    foot_site_names = ["left_foot", "right_foot"]

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

    foot_height_scan_cfg = TerrainHeightSensorCfg(
        name="foot_height_scan",
        frame=tuple(ObjRef(type="site", name=s, entity="robot") for s in foot_site_names),
        pattern=RingPatternCfg.single_ring(radius=0.04, num_samples=2),
        ray_alignment="yaw",
        max_distance=1.0,
        exclude_parent_body=True,
        include_geom_groups=(0,),
        debug_vis=False,
    )

    self_collision_sensor_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    
    cfg.scene.sensors = (
        feet_ground_sensor_cfg,
        foot_height_scan_cfg,
        self_collision_sensor_cfg,
    )

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

    # Observation delays/noises to simulate IMU sensor readings (only for actor, not critic)
    cfg.observations["actor"].terms["projected_gravity"] = deepcopy(cfg.observations["actor"].terms["projected_gravity"])
    cfg.observations["actor"].terms["projected_gravity"].noise = Unoise(n_min=-0.005, n_max=0.005)

    cfg.observations["actor"].terms["base_ang_vel"] = deepcopy(cfg.observations["actor"].terms["base_ang_vel"])
    cfg.observations["actor"].terms["base_ang_vel"].noise = Unoise(n_min=-0.03, n_max=0.03)

    # cfg.observations["actor"].terms["base_ang_vel"].delay_min_lag = 0
    # cfg.observations["actor"].terms["base_ang_vel"].delay_max_lag = 1
    # cfg.observations["actor"].terms["base_ang_vel"].delay_update_period = 64
    # cfg.observations["actor"].terms["projected_gravity"].delay_min_lag = 1
    # cfg.observations["actor"].terms["projected_gravity"].delay_max_lag = 1
    # cfg.observations["actor"].terms["projected_gravity"].delay_update_period = 64

    cfg.observations["actor"].terms["base_ang_vel"].delay_min_lag = 3
    cfg.observations["actor"].terms["base_ang_vel"].delay_max_lag = 4
    cfg.observations["actor"].terms["base_ang_vel"].delay_update_period = 64
    cfg.observations["actor"].terms["projected_gravity"].delay_min_lag = 3
    cfg.observations["actor"].terms["projected_gravity"].delay_max_lag = 4
    cfg.observations["actor"].terms["projected_gravity"].delay_update_period = 64

    #---------------------------- Rewards ---------------------------
    cfg.rewards["track_linear_velocity"].params["std"] = np.sqrt(0.1)
    cfg.rewards["track_linear_velocity"].weight = 2.0

    cfg.rewards["track_angular_velocity"].params["std"] = np.sqrt(0.5)
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

    walking_threshold = 0.01

    cfg.rewards["pose"].params["std_standing"] = std_standing
    cfg.rewards["pose"].params["std_walking"] = std_walking
    cfg.rewards["pose"].params["std_running"] = std_walking
    cfg.rewards["pose"].params["walking_threshold"] = walking_threshold
    cfg.rewards["pose"].weight = 1.0

    cfg.rewards["upright"].func = local_upright
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk",)
    cfg.rewards["upright"].params["pitch"] = np.deg2rad(5.0)
    cfg.rewards["upright"].weight = 1.0
    
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk",)
    cfg.rewards["body_ang_vel"].weight = -0.05

    cfg.rewards["angular_momentum"].weight = -0.02

    for reward_name in ["foot_clearance", "foot_slip"]:
        cfg.rewards[reward_name].params["asset_cfg"].site_names = foot_site_names

    cfg.rewards["foot_clearance"].params["command_threshold"] = walking_threshold
    cfg.rewards["foot_clearance"].params["target_height"] = 0.02

    cfg.rewards["foot_swing_height"].params["command_threshold"] = walking_threshold
    cfg.rewards["foot_swing_height"].params["target_height"] = 0.02

    cfg.rewards["air_time"].params["command_threshold"] = walking_threshold
    cfg.rewards["air_time"].params["threshold_min"] = 0.125
    cfg.rewards["air_time"].params["threshold_max"] = 0.300
    cfg.rewards["air_time"].weight = 3.0

    cfg.rewards["no_stepping"] = RewardTermCfg(
        func=no_stepping_penalty,
        weight=0.0,
        params={
            "sensor_name": feet_ground_sensor_cfg.name,
            "command_name": "twist",
            "command_threshold": walking_threshold,
        },
    )

    del cfg.rewards["soft_landing"]

    cfg.rewards["foot_slip"].params["command_threshold"] = walking_threshold
    cfg.rewards["foot_slip"].weight = -1.0

    cfg.rewards["action_rate_l2"].weight = -0.1

    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_sensor_cfg.name},
    )

    #---------------------------- Commands --------------------------
    command = cfg.commands["twist"]
    command.build = lambda env, _cmd=command: UniformVelocityCommandWithRotation(_cmd, env)
    command.viz.z_offset = 0.5

    command.rel_standing_envs = 0.1
    command.rel_heading_envs = 0.0
    command.rel_rotation_envs = 0.1

    command.ranges.lin_vel_x = (-0.5, 0.5)
    command.ranges.lin_vel_y = (-0.3, 0.3)
    command.ranges.ang_vel_z = (-0.75, 0.75)

    command.rotation_env_ang_vel_range = (-1.5, 1.5)
    command.rotation_min_ang_vel = 0.5

    #---------------------------- Events ----------------------------
    cfg.events["reset_base"].params["pose_range"]["z"] = (0.0, 0.01)

    cfg.events["push_robot"].params["velocity_range"] = {
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
    }

    cfg.events["foot_friction"].params["asset_cfg"].geom_names = (
        r".*left_foot_collision.*",
        r".*right_foot_collision.*",
    )

    cfg.events["base_com"].params["ranges"] = {
        0: (-0.005, 0.005),
        1: (-0.005, 0.005),
        2: (-0.005, 0.005),
    }
    cfg.events["base_com"].params["asset_cfg"].body_names = ("trunk",)

    cfg.events["dof_armature_randomization"] = EventTermCfg(
        mode="startup",
        func=dr.joint_armature,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
            "operation": "scale",
            "ranges": (0.9, 1.1),
        },
    )

    cfg.events["dof_friction_randomization"] = EventTermCfg(
        mode="startup",
        func=dr.joint_friction,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
            "operation": "scale",
            "ranges": (0.9, 1.1),
        },
    )

    # Reference phase

    joint_names_all = list(HOME_FRAME.joint_pos.keys())

    cfg.commands["reference"] = ReferenceCommandCfg(
        resampling_time_range=(1e9, 1e9),  # never externally resampled
        poses_path="poses.pkl",
        twist_command_name="twist",
        velocity_threshold=0.05,
    )

    cfg.observations["actor"].terms["reference_phase"] = ObservationTermCfg(
        func=reference_phase,
        params={"command_name": "reference"},
    )
    cfg.observations["critic"].terms["reference_phase"] = ObservationTermCfg(
        func=reference_phase,
        params={"command_name": "reference"},
    )

    del cfg.rewards["pose"]
    cfg.rewards["reference_pose"] = RewardTermCfg(
        func=reference_pose_reward,
        weight=1.0,
        params={
            "std_standing": std_standing,
            "std_walking": std_walking,
            "command_name": "reference",
            "asset_cfg": SceneEntityCfg("robot", joint_names=joint_names_all),
        },
    )

    #---------------------------- Curriculum ------------------------
    cfg.curriculum = {}

    cfg.curriculum["staged_curriculum"] = CurriculumTermCfg(
        func=step_based_staged_curriculum,
        params={
            "stages": [
                {
                    "name": "penalize stepping + increase velocity",
                    "step": 5000 * 24,
                    "apply": lambda env: {
                        set_command_velocity(
                            env,
                            lin_vel_x=(-0.7, 0.7),
                            ang_vel_z=(-1.5, 1.5),
                            rotation_env_ang_vel_z=(-3.0, 3.0),
                        ),
                        set_stepping_parameters(
                            env,
                            air_time_weight=3.0,
                            no_stepping_penalty_weight=-1.0,
                            rel_standing_envs=0.2,
                            rel_rotation_envs=0.3,
                        ),
                        set_push_parameters(
                            env,
                            velocity_range={
                                "x": (-0.35, 0.35),
                                "y": (-0.35, 0.35),
                            },
                        ),
                    },
                },
                # {
                #     "name": "increase velocity",
                #     "step": 20000 * 24,
                #     "apply": lambda env: set_command_velocity(
                #         env,
                #         lin_vel_x=(-0.9, 0.9),
                #         ang_vel_z=(-3.0, 3.0),
                #         rotation_env_ang_vel_z=(-3.0, 3.0),
                #     ),
                # },
            ],
        },
    )

    # cfg.curriculum["staged_curriculum"] = CurriculumTermCfg(
    #     func=reward_based_staged_curriculum,
    #     params={
    #         "stages": [
    #             {
    #                 "name": "learning to step",
    #                 "reward_term_name": "track_angular_velocity",
    #                 "threshold": 1.0,
    #                 "apply": lambda env: penalize_stepping_while_standing(
    #                     env,
    #                     air_time_weight=1.0,
    #                     no_stepping_penalty_weight=-0.1,
    #                 ),
    #             },
    #             {
    #                 "name": "learning to walk forward",
    #                 "reward_term_name": "track_linear_velocity",
    #                 "threshold": 1.0,
    #                 "apply": lambda env: (
    #                     set_command_velocity(env, lin_vel_x=(-0.7, 0.7)),
    #                     # setattr(env.command_manager.get_term_cfg("twist"), "rel_rotation_envs", 0.3),
    #                 ),
    #             },
    #             {
    #                 "name": "learning to turn",
    #                 "reward_term_name": "track_angular_velocity",
    #                 "threshold": 0.7,
    #                 "apply": lambda env: set_command_velocity(
    #                     env, 
    #                     ang_vel_z=(-2.0, 2.0),
    #                 ),
    #             }
    #         ],
    #     },
    # )

    # cfg.curriculum["track_linear_velocity_reward"] = CurriculumTermCfg(
    #     func=reward_based_curriculum,
    #     params={
    #         "reward_term_name": "track_linear_velocity",
    #         "stages": [
    #             {
    #                 "name": "command lin_vel increase",
    #                 "threshold": 1.5,
    #                 "apply": lambda env: set_command_velocity(
    #                     env,
    #                     lin_vel_x=(-0.7, 0.7),
    #                 ),
    #             },
    #         ],
    #     },
    # )

    # cfg.curriculum["track_angular_velocity_reward"] = CurriculumTermCfg(
    #     func=reward_based_curriculum,
    #     params={
    #         "reward_term_name": "track_angular_velocity",
    #         "stages": [
    #             {
    #                 "name": "command ang_vel increase",
    #                 "threshold": 1.0,
    #                 "apply": lambda env: set_command_velocity(
    #                     env,
    #                     ang_vel_z=(-1.5, 1.5),
    #                 ),
    #             },
    #         ],
    #     },
    # )

    # cfg.curriculum["air_time_reward"] = CurriculumTermCfg(
    #     func=reward_based_curriculum,
    #     params={
    #         "reward_term_name": "air_time",
    #         "stages": [
    #             {
    #                 "name": "penalize stepping while standing",
    #                 "threshold": 1.2,
    #                 "apply": lambda env: penalize_stepping_while_standing(
    #                     env,
    #                     air_time_weight=0.1,
    #                     no_stepping_penalty_weight=-0.1,
    #                 ),
    #             },
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
        cfg.curriculum = {}
        
        cfg.commands["twist"].rel_standing_envs = 0.0
        cfg.commands["twist"].rel_rotation_envs = 0.0

        cfg.events["push_robot"].params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
        }

        cfg.commands["twist"].ranges.lin_vel_x = (0.5, 0.5)
        cfg.commands["twist"].ranges.lin_vel_y = (0.0, 0.0)
        cfg.commands["twist"].ranges.ang_vel_z = (0.0, 0.0)
        cfg.commands["twist"].rotation_env_ang_vel_range = (1.0, 1.0)

        # cfg.commands["twist"].ranges.lin_vel_x = (-0.7, 0.7)
        # cfg.commands["twist"].ranges.lin_vel_y = (-0.3, 0.3)
        # cfg.commands["twist"].ranges.ang_vel_z = (-1.5, 1.5)
        # cfg.commands["twist"].rotation_env_ang_vel_range = (-3.0, 3.0)

        # Can be used to edit neutral pose with a zero agent
        # cfg.events["reset_base"].params["pose_range"]["x"] = (0.0, 0.0)
        # cfg.events["reset_base"].params["pose_range"]["y"] = (0.0, 0.0)
        # cfg.events["reset_base"].params["pose_range"]["z"] = (0.3, 0.3)
        # cfg.events["reset_base"].params["pose_range"]["yaw"] = (0.0, 0.0)
        # cfg.events["reset_base"].interval_range_s = (0.0, 0.0)
        # cfg.events["reset_base"].mode = "interval"

        cfg.observations["actor"].enable_corruption = False

        # cfg.terminations = {}

        # Can be used to print something every step
        def debug(env: ManagerBasedRlEnv, _):
            env.observation_manager.compute_group("actor", update_history=True)

            terms = dict(env.observation_manager.get_active_iterable_terms(env_idx=0))

            base_ang_vel = terms["actor-base_ang_vel"]
            projected_gravity = terms["actor-projected_gravity"]

            print("\n")
            print(f"base_ang_vel: ")
            print(f"x: {base_ang_vel[0]:.3f}")
            print(f"y: {base_ang_vel[1]:.3f}")
            print(f"z: {base_ang_vel[2]:.3f}")
            print("\n")
            print(f"projected_gravity: ")
            print(f"x: {projected_gravity[0]:.3f}")
            print(f"y: {projected_gravity[1]:.3f}")
            print(f"z: {projected_gravity[2]:.3f}")

        # cfg.events["debug"] = EventTermCfg(
        #     func=debug, mode="interval", interval_range_s=(0.0, 0.0)
        # )

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
