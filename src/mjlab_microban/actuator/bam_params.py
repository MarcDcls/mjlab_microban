"""Load BAM M4/M6 parameters from JSON and create actuator configs."""

from __future__ import annotations

import json
from pathlib import Path

from mjlab_microban.actuator.bam_actuator import (
    BamM4ActuatorCfg,
    BamM6ActuatorCfg,
)


# M6 params from clean data identification (m6_new.json)
DEFAULT_XL330_M6 = {
    "kt": 0.24702827088535634,
    "R": 2.436537942885361,
    "armature": 0.002231042413951293,
    "friction_base": 0.007805203011273793,
    "friction_stribeck": 0.01299013941785831,
    "load_friction_motor": 0.17679071496643342,
    "load_friction_external": 0.33284617369197755,
    "load_friction_motor_stribeck": 0.04834054555210131,
    "load_friction_external_stribeck": 0.03230746746292114,
    "load_friction_motor_quad": 0.004778286363709164,
    "load_friction_external_quad": 0.004335373885291851,
    "dtheta_stribeck": 0.10838180452009236,
    "alpha": 2.1089115156897034,
    "friction_viscous": 0.01674718702359746,
}


# M4 params (xl330_test/m4.json from Rhoban/bam)
DEFAULT_XL330_M4 = {
    "kt": 0.40807008666379696,
    "R": 2.8004973357212677,
    "armature": 0.0014905005069671938,
    "friction_base": 0.00944647379388224,
    "friction_stribeck": 0.0026283903757616453,
    "load_friction_base": 0.19194381271212732,
    "load_friction_stribeck": 0.07555177546287972,
    "dtheta_stribeck": 0.6445896028101025,
    "alpha": 2.450581810840127,
    "friction_viscous": 0.002447363938065353,
}


def _load_params(json_path: str | Path, expected_model: str) -> dict:
    with open(json_path) as f:
        params = json.load(f)
    got = params.get("model")
    assert got == expected_model, f"Expected {expected_model} model, got {got}"
    return params


def load_bam_m6_params(json_path: str | Path) -> dict:
    return _load_params(json_path, "m6")


def load_bam_m4_params(json_path: str | Path) -> dict:
    return _load_params(json_path, "m4")


def make_bam_m6_actuator_cfg(
    joint_names_expr: tuple[str, ...] = (r".*",),
    params: dict | None = None,
    json_path: str | Path | None = None,
    vin: float = 7.4,
    kp_fw: float = 200.0,
) -> BamM6ActuatorCfg:
    """Create a BamM6ActuatorCfg from BAM M6 parameters."""
    if json_path is not None:
        p = load_bam_m6_params(json_path)
    elif params is not None:
        p = params
    else:
        p = DEFAULT_XL330_M6

    return BamM6ActuatorCfg(
        joint_names_expr=joint_names_expr,
        armature=p["armature"],
        kt=p["kt"],
        R=p["R"],
        vin=vin,
        kp_fw=kp_fw,
        friction_base=p["friction_base"],
        friction_stribeck=p["friction_stribeck"],
        dtheta_stribeck=p["dtheta_stribeck"],
        alpha=p["alpha"],
        friction_viscous=p["friction_viscous"],
        load_friction_motor=p["load_friction_motor"],
        load_friction_external=p["load_friction_external"],
        load_friction_motor_stribeck=p["load_friction_motor_stribeck"],
        load_friction_external_stribeck=p["load_friction_external_stribeck"],
        load_friction_motor_quad=p["load_friction_motor_quad"],
        load_friction_external_quad=p["load_friction_external_quad"],
    )


def make_bam_m4_actuator_cfg(
    joint_names_expr: tuple[str, ...] = (r".*",),
    params: dict | None = None,
    json_path: str | Path | None = None,
    vin: float = 7.4,
    kp_fw: float = 200.0,
) -> BamM4ActuatorCfg:
    """Create a BamM4ActuatorCfg from BAM M4 parameters."""
    if json_path is not None:
        p = load_bam_m4_params(json_path)
    elif params is not None:
        p = params
    else:
        p = DEFAULT_XL330_M4

    return BamM4ActuatorCfg(
        joint_names_expr=joint_names_expr,
        armature=p["armature"],
        kt=p["kt"],
        R=p["R"],
        vin=vin,
        kp_fw=kp_fw,
        friction_base=p["friction_base"],
        friction_stribeck=p["friction_stribeck"],
        dtheta_stribeck=p["dtheta_stribeck"],
        alpha=p["alpha"],
        friction_viscous=p["friction_viscous"],
        load_friction_base=p["load_friction_base"],
        load_friction_stribeck=p["load_friction_stribeck"],
    )
