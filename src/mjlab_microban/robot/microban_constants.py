# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

import os
import numpy as np
from pathlib import Path

import mujoco
from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

MICROBAN_XML: Path = Path(os.path.dirname(__file__)) / "microban" / "robot.xml"
assert MICROBAN_XML.exists(), f"XML not found: {MICROBAN_XML}"

def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICROBAN_XML))

HOME_FRAME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.168), #0.1676),
    joint_pos={
        "head": float(np.deg2rad(0.0)),
        "left_shoulder_roll": float(np.deg2rad(10.0)),
        "right_shoulder_roll": float(np.deg2rad(-10.0)),
        "left_shoulder_pitch": float(np.deg2rad(10.0)),
        "right_shoulder_pitch": float(np.deg2rad(10.0)),
        "left_elbow": float(np.deg2rad(-20.0)),
        "right_elbow": float(np.deg2rad(-20.0)),
        "left_hip_roll": float(np.deg2rad(5.0)),
        "right_hip_roll": float(np.deg2rad(-5.0)),
        "left_hip_pitch": float(np.deg2rad(0.0)),
        "right_hip_pitch": float(np.deg2rad(0.0)),
        "left_hip_yaw": float(np.deg2rad(0.0)),
        "right_hip_yaw": float(np.deg2rad(0.0)),
        "left_knee": float(np.deg2rad(0.0)),
        "right_knee": float(np.deg2rad(0.0)),
        "left_ankle_roll": float(np.deg2rad(-5.0)),
        "right_ankle_roll": float(np.deg2rad(5.0)),
        "left_ankle_pitch": float(np.deg2rad(0.0)),
        "right_ankle_pitch": float(np.deg2rad(0.0)),
    },
    joint_vel={r".*": 0.0},
)

FULL_COLLISION = CollisionCfg(
    geom_names_expr=(r".*_collision",),
    condim={r"^(left|right)_foot_collision$": 3, r".*_collision": 1},
    priority={r"^(left|right)_foot_collision$": 1},
    friction={r"^(left|right)_foot_collision$": (1.0,)},
)

from bam.mjlab import make_bam_actuator_cfg

actuators = make_bam_actuator_cfg(
    json_path="/home/marc/bam/params/xl330/m6.json",
    kp_fw=125,
    vin=8.0,
    target_names_expr=(r".*",),
    delay_min_lag=0,
    delay_max_lag=3,
)

# -- Old actuator (XML position, MuJoCo built-in PD + friction) --
# actuators = XmlActuatorCfg(
#     target_names_expr=(r".*",),
#     delay_min_lag=0,
#     delay_max_lag=3,
# )

MICROBAN_ROBOT_CFG = EntityCfg(
    spec_fn=get_spec,
    init_state=HOME_FRAME,
    collisions=(FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

if __name__ == "__main__":
    import mujoco.viewer as viewer
    from mjlab.scene import Scene, SceneCfg
    from mjlab.terrains import TerrainEntityCfg

    SCENE_CFG = SceneCfg(
        terrain=TerrainEntityCfg(terrain_type="plane"),
        entities={"robot": MICROBAN_ROBOT_CFG},
    )

    scene = Scene(SCENE_CFG, device="cuda:0")
    model = scene.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("init_state").id)
    viewer.launch(model, data=data)